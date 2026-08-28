//! Per-writer transactional index over property values.
//!
//! [`crate::WriterSession::unique_probe`] answers "which node currently
//! holds this value tuple for `(label, properties)`?" in O(1) after a
//! one-time label scan, instead of re-scanning the label for every row a
//! constraint-bearing bulk write stages. Unlike
//! [`crate::property_index::PropertyIndexCache`] (committed state only,
//! shared across snapshots), this index is private to one `WriterSession`
//! and tracks committed **plus staged** state: it is populated lazily from
//! the read-your-own-writes overlay — the same source the flat scan uses —
//! and then kept current by every staged node upsert/tombstone, so a value
//! freed or claimed earlier in the same uncommitted batch is visible to the
//! next check.
//!
//! Consistency contract: a populated `(label, property-set)` map must agree
//! with a fresh `scan_label` over the overlay snapshot at all times. Node
//! mutations are applied at the staging chokepoints. The first mutation of
//! each node in a pending batch journals its prior tuple, so commit can simply
//! forget the journal while discard restores the already-populated maps
//! without a corpus scan. Maps first populated from an overlay that already
//! contains staged rows are removed on discard because they have no committed
//! baseline. The same maps also back non-unique String equality postings used
//! by indexed `MATCH` / `MERGE`: `holders` already retains every claimant, so
//! no second transactional index is needed. Flush preserves the maps (it
//! changes physical representation, not logical content). Events that bypass
//! the chokepoints — external SST attachment, session reopen, and relevant DDL
//! — reset the index.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

use namidb_core::id::NodeId;
use namidb_core::Value;

/// Outcome of probing the writer's unique-value index.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UniqueProbe {
    /// A node other than the excluded one currently holds the value tuple.
    Conflict(NodeId),
    /// No other node holds the value tuple.
    NoConflict,
    /// At least one probed value has no canonical scalar encoding
    /// (vector/list/map/null/NaN); the caller must fall back to the
    /// scan-based check, which IS the source of truth.
    Unindexable,
}

/// Canonical, hashable form of one scalar property value. Key equality must
/// match [`Value`]'s derived `PartialEq` exactly — that is what the flat
/// scan compares with — so `I64(1)` and `F64(1.0)` stay distinct, `-0.0`
/// is folded into `0.0` (`f64::eq` treats them as equal, bit patterns do
/// not), and NaN is rejected at encode time (`NaN != NaN` means a scan can
/// never observe a NaN conflict).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) enum UniqueKeyPart {
    Str(String),
    I64(i64),
    Bool(bool),
    /// Bit pattern of a non-NaN f64 with -0.0 normalised to 0.0.
    F64(u64),
    Bytes(Vec<u8>),
    Date(i32),
    DateTime(i64),
}

pub(crate) type UniqueKey = Vec<UniqueKeyPart>;

fn key_part(v: &Value) -> Option<UniqueKeyPart> {
    match v {
        Value::Str(s) => Some(UniqueKeyPart::Str(s.clone())),
        Value::I64(n) => Some(UniqueKeyPart::I64(*n)),
        Value::Bool(b) => Some(UniqueKeyPart::Bool(*b)),
        Value::F64(f) if !f.is_nan() => {
            let f = if *f == 0.0 { 0.0 } else { *f };
            Some(UniqueKeyPart::F64(f.to_bits()))
        }
        Value::Bytes(b) => Some(UniqueKeyPart::Bytes(b.clone())),
        Value::Date(d) => Some(UniqueKeyPart::Date(*d)),
        Value::DateTime(m) => Some(UniqueKeyPart::DateTime(*m)),
        Value::F64(_)
        | Value::Null
        | Value::Vec(_)
        | Value::VecI8 { .. }
        | Value::List(_)
        | Value::Map(_) => None,
    }
}

/// Encode the probe values themselves (already paired with their sorted
/// property names). `None` when any value is unindexable.
pub(crate) fn encode_probe_key(values: &[&Value]) -> Option<UniqueKey> {
    values.iter().map(|v| key_part(v)).collect()
}

/// Encode a node's key for a constraint over `names` (sorted) from its
/// property map. `None` when any property is absent or unindexable — such a
/// node cannot equal an indexable probe tuple, so it is simply not filed.
fn encode_node_key(names: &[String], props: &BTreeMap<String, Value>) -> Option<UniqueKey> {
    names
        .iter()
        .map(|n| props.get(n).and_then(key_part))
        .collect()
}

/// One populated `(label, property-set)` constraint map.
///
/// Holder representation optimized for the dominant high-cardinality-index
/// shape: one distinct value per node must not allocate a tiny hash table 1.5M
/// times. A duplicate/non-unique value promotes lazily to a set; deletion
/// collapses it back to the inline singleton.
// Keep the common singleton variant compact: this map has one entry per
// indexed value (about 1.5M in the legal corpus), while duplicate postings
// are rare and can afford the extra indirection.
#[allow(clippy::box_collection)]
#[derive(Debug)]
enum Holders {
    One(NodeId),
    Many(Box<HashSet<NodeId>>),
}

impl Holders {
    fn insert(&mut self, id: NodeId) {
        match self {
            Self::One(existing) => {
                debug_assert_ne!(*existing, id);
                let mut ids = HashSet::with_capacity(2);
                ids.insert(*existing);
                ids.insert(id);
                *self = Self::Many(Box::new(ids));
            }
            Self::Many(ids) => {
                debug_assert!(!ids.contains(&id));
                ids.insert(id);
            }
        }
    }

    /// Remove `id`; return true when no holder remains.
    fn remove(&mut self, id: NodeId) -> bool {
        match self {
            Self::One(existing) => *existing == id,
            Self::Many(ids) => {
                ids.remove(&id);
                match ids.len() {
                    0 => true,
                    1 => {
                        let remaining = *ids.iter().next().expect("len checked");
                        *self = Self::One(remaining);
                        false
                    }
                    _ => false,
                }
            }
        }
    }

    fn first_other(&self, exclude: Option<NodeId>) -> Option<NodeId> {
        match self {
            Self::One(id) => (Some(*id) != exclude).then_some(*id),
            Self::Many(ids) => ids.iter().copied().find(|id| Some(*id) != exclude),
        }
    }

    fn to_vec(&self) -> Vec<NodeId> {
        match self {
            Self::One(id) => vec![*id],
            Self::Many(ids) => ids.iter().copied().collect(),
        }
    }
}

/// `holders` keeps EVERY node currently carrying a value tuple (normally one,
/// but pre-existing duplicates — e.g. a constraint declared over data that
/// already violates it — must keep answering "conflict" exactly like the
/// scan would). `by_node` is the reverse edge that makes staged upserts and
/// tombstones O(1): a full-record upsert first detaches the node from its
/// previous tuple, then files it under the new one.
#[derive(Debug, Default)]
struct ConstraintMap {
    holders: HashMap<UniqueKey, Holders>,
    by_node: HashMap<NodeId, UniqueKey>,
    /// A full map can answer every negative probe. A partial map, seeded from
    /// immutable point sidecars for one MERGE batch, may answer only occupied
    /// keys in `holders` and confirmed misses in `known_misses`; any other key
    /// returns `None` and triggers the existing authoritative population path.
    ///
    /// Hits deliberately are not duplicated here: their presence in
    /// `holders` is already the positive-knowledge bit. This keeps a long
    /// loader from retaining a third copy of every unique value.
    complete: bool,
    known_misses: HashSet<UniqueKey>,
}

impl ConstraintMap {
    fn detach(&mut self, id: NodeId) {
        if let Some(old) = self.by_node.remove(&id) {
            let empty = self
                .holders
                .get_mut(&old)
                .is_some_and(|holders| holders.remove(id));
            if empty {
                self.holders.remove(&old);
                if !self.complete {
                    // A staged delete/re-home of the last holder turns a
                    // previously-known hit into an authoritative miss.
                    self.known_misses.insert(old);
                }
            }
        }
    }

    fn file(&mut self, id: NodeId, key: UniqueKey) {
        debug_assert!(!self.by_node.contains_key(&id));
        self.known_misses.remove(&key);
        match self.holders.entry(key.clone()) {
            std::collections::hash_map::Entry::Vacant(entry) => {
                entry.insert(Holders::One(id));
            }
            std::collections::hash_map::Entry::Occupied(mut entry) => {
                entry.get_mut().insert(id);
            }
        }
        self.by_node.insert(id, key);
    }
}

type ConstraintId = (String, Vec<String>);

/// Rollback state for one constraint map during the current pending batch.
///
/// Existing maps journal each touched node's tuple before its first staged
/// mutation. A map populated from an already-staged overlay has no committed
/// baseline to restore, so `created_in_batch` makes rollback remove it.
#[derive(Debug, Default)]
struct ConstraintUndo {
    created_in_batch: bool,
    by_node: HashMap<NodeId, Option<UniqueKey>>,
    /// Exact pre-batch membership for every partial-map miss touched by this
    /// batch. A key introduced only by a staged upsert was previously
    /// *unknown*, not an authoritative miss; restoring only `by_node` would
    /// otherwise turn it into a false-negative answer on rollback.
    known_misses: HashMap<UniqueKey, bool>,
}

#[derive(Debug, Default)]
struct IndexState {
    maps: HashMap<ConstraintId, ConstraintMap>,
    undo: HashMap<ConstraintId, ConstraintUndo>,
    /// Group-commit request scope (RFC-034): while `Some`, staged
    /// maintenance journals first-touch state HERE instead of `undo`, so a
    /// FAILED statement can be rolled back to its request boundary — the
    /// journaled values are the at-request-start (possibly already-staged)
    /// tuples, not the pre-batch ones. On request success the layer merges
    /// into `undo` keeping the OLDEST first-touch entry per node, which
    /// preserves the pre-batch restore of a full [`discard`] later.
    request_undo: Option<HashMap<ConstraintId, ConstraintUndo>>,
}

impl IndexState {
    /// Split borrow: the maps plus whichever journal staged maintenance
    /// must write (the request layer when a scope is open, else the batch
    /// journal).
    fn maps_and_journal(
        &mut self,
    ) -> (
        &mut HashMap<ConstraintId, ConstraintMap>,
        &mut HashMap<ConstraintId, ConstraintUndo>,
    ) {
        let IndexState {
            maps,
            undo,
            request_undo,
        } = self;
        (maps, request_undo.as_mut().unwrap_or(undo))
    }
}

/// The per-writer index: `(label, sorted property names) → ConstraintMap`.
/// Interior mutability because probes run under `&WriterSession` while
/// staged-write maintenance runs under `&mut` — the writer is single-owner,
/// so the mutex is uncontended.
#[derive(Debug, Default)]
pub struct UniqueConstraintIndex {
    state: Mutex<IndexState>,
    /// Label scans performed to populate a constraint map. Exposed so tests
    /// can assert a bulk write pays exactly one scan, not one per row.
    populate_scans: AtomicU64,
    /// Probes answered from a populated map (i.e. without scanning).
    probes: AtomicU64,
    /// Non-unique posting-list probes answered from a populated map.
    posting_probes: AtomicU64,
}

impl UniqueConstraintIndex {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// Probe a populated map. `None` when `(label, names)` has not been
    /// populated yet — the caller scans and calls [`Self::populate`].
    pub(crate) fn probe(
        &self,
        label: &str,
        names: &[String],
        key: &UniqueKey,
        exclude: Option<NodeId>,
    ) -> Option<UniqueProbe> {
        let state = self.state.lock().expect("unique index lock");
        let map = state.maps.get(&(label.to_string(), names.to_vec()))?;
        if !map.complete && !map.holders.contains_key(key) && !map.known_misses.contains(key) {
            return None;
        }
        self.probes.fetch_add(1, Ordering::Relaxed);
        let conflict = map
            .holders
            .get(key)
            .and_then(|holders| holders.first_other(exclude));
        Some(match conflict {
            Some(id) => UniqueProbe::Conflict(id),
            None => UniqueProbe::NoConflict,
        })
    }

    /// Return every node currently holding `key` for `(label, names)`.
    ///
    /// `None` means the map has not been populated yet; `Some(empty)` is an
    /// authoritative negative answer. This is the non-unique counterpart of
    /// [`Self::probe`], used by String equality indexes in writer/RYOW
    /// snapshots. Cloning the normally-small holder vector keeps the index
    /// mutex out of async point-read confirmation.
    pub(crate) fn probe_all(
        &self,
        label: &str,
        names: &[String],
        key: &UniqueKey,
    ) -> Option<Vec<NodeId>> {
        let state = self.state.lock().expect("unique index lock");
        let map = state.maps.get(&(label.to_string(), names.to_vec()))?;
        if !map.complete && !map.holders.contains_key(key) && !map.known_misses.contains(key) {
            return None;
        }
        self.posting_probes.fetch_add(1, Ordering::Relaxed);
        Some(
            map.holders
                .get(key)
                .map(Holders::to_vec)
                .unwrap_or_default(),
        )
    }

    /// Install the `(label, names)` map from a label scan over the overlay
    /// snapshot. `entries` yields every live node of the label with its
    /// current property map.
    pub(crate) fn populate<'a>(
        &self,
        label: &str,
        names: &[String],
        entries: impl Iterator<Item = (NodeId, &'a BTreeMap<String, Value>)>,
    ) {
        self.populate_inner(label, names, entries, false);
    }

    /// Populate from an overlay that already contains staged mutations.
    ///
    /// The resulting map is exact for the current transaction, but it cannot
    /// be incrementally restored to committed state because the pre-staged
    /// tuples were never observed. Mark it for removal on rollback; commit
    /// promotes it by merely dropping that marker.
    pub(crate) fn populate_staged<'a>(
        &self,
        label: &str,
        names: &[String],
        entries: impl Iterator<Item = (NodeId, &'a BTreeMap<String, Value>)>,
    ) {
        self.populate_inner(label, names, entries, true);
    }

    fn populate_inner<'a>(
        &self,
        label: &str,
        names: &[String],
        entries: impl Iterator<Item = (NodeId, &'a BTreeMap<String, Value>)>,
        staged: bool,
    ) {
        let mut map = ConstraintMap::default();
        for (id, props) in entries {
            if let Some(key) = encode_node_key(names, props) {
                map.file(id, key);
            }
        }
        map.complete = true;
        self.populate_scans.fetch_add(1, Ordering::Relaxed);
        let identity = (label.to_string(), names.to_vec());
        let mut state = self.state.lock().expect("unique index lock");
        state.maps.insert(identity.clone(), map);
        if staged {
            let (_, journal) = state.maps_and_journal();
            let undo = journal.entry(identity).or_default();
            undo.created_in_batch = true;
            undo.by_node.clear();
            undo.known_misses.clear();
        } else {
            state.undo.remove(&identity);
            if let Some(request) = state.request_undo.as_mut() {
                request.remove(&identity);
            }
        }
    }

    /// Seed authoritative point answers without claiming that the whole
    /// `(label, names)` domain is populated.
    ///
    /// `entries` must come from a current committed sidecar-backed lookup and
    /// include misses as `(key, None)`. This is called before a statement
    /// stages node mutations. Subsequent upsert/tombstone chokepoints maintain
    /// the seeded keys exactly; an unseeded probe still returns `None` and
    /// takes the full-population fallback.
    pub(crate) fn seed_committed_keys(
        &self,
        label: &str,
        names: &[String],
        entries: impl IntoIterator<Item = (UniqueKey, Option<NodeId>)>,
    ) {
        let identity = (label.to_string(), names.to_vec());
        let mut state = self.state.lock().expect("unique index lock");
        if state.maps.get(&identity).is_some_and(|map| map.complete) {
            return;
        }
        debug_assert!(
            !state.undo.contains_key(&identity)
                && !state
                    .request_undo
                    .as_ref()
                    .is_some_and(|request| request.contains_key(&identity)),
            "committed key seeding must happen before staged mutations"
        );
        let map = state.maps.entry(identity).or_default();
        for (key, holder) in entries {
            if map.holders.contains_key(&key) || map.known_misses.contains(&key) {
                continue;
            }
            if let Some(id) = holder {
                // A confirmed unique node can only occupy one current tuple.
                // Detaching defensively also makes reseeding robust to a
                // previously-known key whose node was moved.
                map.detach(id);
                map.file(id, key);
            } else {
                map.known_misses.insert(key);
            }
        }
    }

    /// Seed authoritative point answers for the current committed+staged
    /// overlay without claiming the whole `(label, names)` domain.
    ///
    /// Unlike [`Self::seed_committed_keys`], this may run after node mutations
    /// have already been staged. An absent map is therefore disposable on
    /// rollback. When a committed partial map already exists, newly learned
    /// hits/misses are journalled too: an answer may depend on a staged
    /// delete/re-home whose old tuple was unknown to that partial map, so
    /// retaining the answer after rollback would risk a false miss.
    pub(crate) fn seed_staged_keys(
        &self,
        label: &str,
        names: &[String],
        entries: impl IntoIterator<Item = (UniqueKey, Option<NodeId>)>,
    ) {
        let identity = (label.to_string(), names.to_vec());
        let mut state = self.state.lock().expect("unique index lock");
        if state.maps.get(&identity).is_some_and(|map| map.complete) {
            return;
        }

        let created_in_batch = !state.maps.contains_key(&identity);
        let (maps, undo) = state.maps_and_journal();
        let map = maps.entry(identity.clone()).or_default();
        let rollback = undo.entry(identity).or_default();
        if created_in_batch {
            rollback.created_in_batch = true;
        }

        for (key, holder) in entries {
            if map.holders.contains_key(&key) || map.known_misses.contains(&key) {
                continue;
            }
            if let Some(id) = holder {
                if !rollback.created_in_batch {
                    rollback
                        .by_node
                        .entry(id)
                        .or_insert_with(|| map.by_node.get(&id).cloned());
                    rollback
                        .known_misses
                        .entry(key.clone())
                        .or_insert_with(|| map.known_misses.contains(&key));
                    if let Some(old_key) = map.by_node.get(&id) {
                        rollback
                            .known_misses
                            .entry(old_key.clone())
                            .or_insert_with(|| map.known_misses.contains(old_key));
                    }
                }
                map.detach(id);
                map.file(id, key);
            } else {
                if !rollback.created_in_batch {
                    rollback
                        .known_misses
                        .entry(key.clone())
                        .or_insert_with(|| map.known_misses.contains(&key));
                }
                map.known_misses.insert(key);
            }
        }
    }

    /// Maintain every populated map for a staged full-record node upsert:
    /// the node's previous tuple (if any) is freed, and it is re-filed under
    /// each constraint whose label it carries and whose properties are all
    /// present and indexable in the new record.
    pub(crate) fn apply_upsert(
        &self,
        id: NodeId,
        labels: &[&str],
        props: &BTreeMap<String, Value>,
    ) {
        let mut state = self.state.lock().expect("unique index lock");
        let (maps, undo) = state.maps_and_journal();
        for (identity @ (clabel, cnames), map) in maps.iter_mut() {
            let rollback = undo.entry(identity.clone()).or_default();
            if !rollback.created_in_batch {
                rollback
                    .by_node
                    .entry(id)
                    .or_insert_with(|| map.by_node.get(&id).cloned());
            }
            let new_key = if clabel.is_empty() || labels.iter().any(|l| l == clabel) {
                encode_node_key(cnames, props)
            } else {
                None
            };
            if !map.complete && !rollback.created_in_batch {
                if let Some(old_key) = map.by_node.get(&id) {
                    rollback
                        .known_misses
                        .entry(old_key.clone())
                        .or_insert_with(|| map.known_misses.contains(old_key));
                }
                if let Some(new_key) = &new_key {
                    rollback
                        .known_misses
                        .entry(new_key.clone())
                        .or_insert_with(|| map.known_misses.contains(new_key));
                }
            }
            map.detach(id);
            // An empty label is the physical any-label scope used by
            // `MATCH (n {prop: ...})`. Every node, including an unlabelled one,
            // belongs to that global postings map.
            if let Some(key) = new_key {
                map.file(id, key);
            }
        }
    }

    /// Maintain every populated map for a staged node tombstone.
    pub(crate) fn apply_tombstone(&self, id: NodeId) {
        let mut state = self.state.lock().expect("unique index lock");
        let (maps, undo) = state.maps_and_journal();
        for (identity, map) in maps.iter_mut() {
            let rollback = undo.entry(identity.clone()).or_default();
            if !rollback.created_in_batch {
                rollback
                    .by_node
                    .entry(id)
                    .or_insert_with(|| map.by_node.get(&id).cloned());
            }
            if !map.complete && !rollback.created_in_batch {
                if let Some(old_key) = map.by_node.get(&id) {
                    rollback
                        .known_misses
                        .entry(old_key.clone())
                        .or_insert_with(|| map.known_misses.contains(old_key));
                }
            }
            map.detach(id);
        }
    }

    /// Promote the current staged view to committed state. The populated maps
    /// already contain those mutations, so only the rollback journal changes.
    pub(crate) fn commit_staged(&self) {
        let mut state = self.state.lock().expect("unique index lock");
        state.undo = HashMap::new();
        // A dangling request scope is a caller bug, but its journal is
        // obsolete either way once the batch is durable.
        state.request_undo = None;
    }

    /// Open a group-commit request scope (RFC-034): staged maintenance
    /// journals into a request-local layer whose entries capture the
    /// AT-REQUEST-START values, so a failed statement rolls back alone.
    pub(crate) fn begin_request(&self) {
        let mut state = self.state.lock().expect("unique index lock");
        debug_assert!(state.request_undo.is_none(), "request scope already open");
        state.request_undo = Some(HashMap::new());
    }

    /// Close the request scope keeping its mutations: merge the request
    /// journal into the batch journal, keeping the OLDEST first-touch entry
    /// per node/key so a later full [`Self::rollback_staged`] still restores
    /// the pre-BATCH state.
    pub(crate) fn merge_request(&self) {
        let mut state = self.state.lock().expect("unique index lock");
        let Some(request) = state.request_undo.take() else {
            return;
        };
        for (identity, from_request) in request {
            match state.undo.entry(identity) {
                std::collections::hash_map::Entry::Vacant(slot) => {
                    slot.insert(from_request);
                }
                std::collections::hash_map::Entry::Occupied(mut slot) => {
                    // The batch entry predates this request, so its
                    // created_in_batch flag and existing first-touch values
                    // win; only genuinely new touches merge in.
                    let batch = slot.get_mut();
                    for (id, old_key) in from_request.by_node {
                        batch.by_node.entry(id).or_insert(old_key);
                    }
                    for (key, was_known_miss) in from_request.known_misses {
                        batch.known_misses.entry(key).or_insert(was_known_miss);
                    }
                }
            }
        }
    }

    /// Roll back ONLY the open request scope's mutations, restoring every
    /// touched map to its at-request-start (possibly staged-by-earlier-
    /// requests) state. Earlier requests in the batch survive.
    pub(crate) fn rollback_request(&self) {
        let mut state = self.state.lock().expect("unique index lock");
        let Some(request) = state.request_undo.take() else {
            return;
        };
        Self::restore(&mut state, request);
    }

    /// Restore every populated map to its pre-batch committed state without a
    /// label scan. This is proportional to the number of distinct node ids
    /// touched in the discarded batch, not to the stored graph size.
    pub(crate) fn rollback_staged(&self) {
        let mut state = self.state.lock().expect("unique index lock");
        // A dangling request scope rolls back first (its journal holds the
        // newest deltas), then the batch journal restores to pre-batch.
        if let Some(request) = state.request_undo.take() {
            Self::restore(&mut state, request);
        }
        let undo = std::mem::take(&mut state.undo);
        Self::restore(&mut state, undo);
    }

    fn restore(state: &mut IndexState, undo: HashMap<ConstraintId, ConstraintUndo>) {
        for (identity, rollback) in undo {
            if rollback.created_in_batch {
                state.maps.remove(&identity);
                continue;
            }
            let Some(map) = state.maps.get_mut(&identity) else {
                continue;
            };
            for (id, old_key) in rollback.by_node {
                map.detach(id);
                if let Some(key) = old_key {
                    map.file(id, key);
                }
            }
            for (key, was_known_miss) in rollback.known_misses {
                if was_known_miss {
                    debug_assert!(
                        !map.holders.contains_key(&key),
                        "a committed holder cannot also be a known miss"
                    );
                    map.known_misses.insert(key);
                } else {
                    map.known_misses.remove(&key);
                }
            }
        }
    }

    /// Drop every populated map; the next probe repopulates from a scan.
    pub(crate) fn reset(&self) {
        let mut state = self.state.lock().expect("unique index lock");
        // Replacing the state drops the HashMap bucket arrays too. `clear()`
        // would leave allocations sized for the largest historical corpus or
        // staged batch resident after an RSS-pressure pass.
        *state = IndexState::default();
    }

    /// Drop sidecar-seeded partial maps after a successful flush.
    ///
    /// The newly committed SST sidecars are now the authoritative point index,
    /// so retaining every key touched since the previous flush would only
    /// duplicate immutable state. Complete maps populated for generic
    /// multi-property constraints remain hot because they can still answer
    /// the whole domain without I/O.
    pub(crate) fn drop_partial_maps(&self) {
        let mut state = self.state.lock().expect("unique index lock");
        debug_assert!(
            state.undo.is_empty(),
            "partial maps may only be reclaimed after commit"
        );
        state.maps.retain(|_, map| map.complete);
    }

    /// Number of populating label scans performed so far.
    pub fn populate_scans(&self) -> u64 {
        self.populate_scans.load(Ordering::Relaxed)
    }

    /// Number of probes answered from a populated map (no scan).
    pub fn probes(&self) -> u64 {
        self.probes.load(Ordering::Relaxed)
    }

    /// Number of non-unique posting probes served from populated maps.
    pub fn posting_probes(&self) -> u64 {
        self.posting_probes.load(Ordering::Relaxed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn nid(n: u8) -> NodeId {
        NodeId::from_uuid(uuid::Uuid::from_bytes([n; 16]))
    }

    fn props(pairs: &[(&str, Value)]) -> BTreeMap<String, Value> {
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.clone()))
            .collect()
    }

    #[test]
    fn key_encoding_matches_value_equality_semantics() {
        // I64 and F64 stay distinct, exactly like `Value`'s PartialEq.
        assert_ne!(key_part(&Value::I64(1)), key_part(&Value::F64(1.0)));
        // -0.0 and 0.0 compare equal as f64, so they must share a key.
        assert_eq!(key_part(&Value::F64(0.0)), key_part(&Value::F64(-0.0)));
        // NaN never equals anything under the scan; refuse to index it.
        assert_eq!(key_part(&Value::F64(f64::NAN)), None);
        // Non-scalar values are unindexable and force the scan fallback.
        assert_eq!(key_part(&Value::List(vec![Value::I64(1)])), None);
        assert_eq!(key_part(&Value::Null), None);
    }

    #[test]
    fn upsert_rehomes_and_tombstone_frees() {
        let idx = UniqueConstraintIndex::new();
        let names = vec!["email".to_string()];
        let a = props(&[("email", Value::Str("a@x".into()))]);
        idx.populate("User", &names, vec![(nid(1), &a)].into_iter());

        let key = encode_probe_key(&[&Value::Str("a@x".into())]).unwrap();
        assert_eq!(
            idx.probe("User", &names, &key, None),
            Some(UniqueProbe::Conflict(nid(1)))
        );
        // Self-exclusion: the holder rewriting its own value is not a conflict.
        assert_eq!(
            idx.probe("User", &names, &key, Some(nid(1))),
            Some(UniqueProbe::NoConflict)
        );

        // Full-record upsert moves the node to a new value, freeing the old.
        let b = props(&[("email", Value::Str("b@x".into()))]);
        idx.apply_upsert(nid(1), &["User"], &b);
        assert_eq!(
            idx.probe("User", &names, &key, None),
            Some(UniqueProbe::NoConflict)
        );
        let key_b = encode_probe_key(&[&Value::Str("b@x".into())]).unwrap();
        assert_eq!(
            idx.probe("User", &names, &key_b, None),
            Some(UniqueProbe::Conflict(nid(1)))
        );

        idx.apply_tombstone(nid(1));
        assert_eq!(
            idx.probe("User", &names, &key_b, None),
            Some(UniqueProbe::NoConflict)
        );
    }

    #[test]
    fn reset_releases_index_and_undo_bucket_allocations() {
        let idx = UniqueConstraintIndex::new();
        let names = vec!["email".to_string()];
        let original = props(&[("email", Value::Str("a@x".into()))]);
        idx.populate("User", &names, std::iter::once((nid(1), &original)));
        let replacement = props(&[("email", Value::Str("b@x".into()))]);
        idx.apply_upsert(nid(1), &["User"], &replacement);
        {
            let state = idx.state.lock().unwrap();
            assert!(state.maps.capacity() > 0);
            assert!(state.undo.capacity() > 0);
        }

        idx.reset();
        let state = idx.state.lock().unwrap();
        assert_eq!(state.maps.capacity(), 0);
        assert_eq!(state.undo.capacity(), 0);
    }

    #[test]
    fn postings_promote_only_on_duplicate_and_collapse_after_remove() {
        let idx = UniqueConstraintIndex::new();
        let names = vec!["group".to_string()];
        let a = props(&[("group", Value::Str("legal".into()))]);
        let b = props(&[("group", Value::Str("legal".into()))]);
        idx.populate("Doc", &names, vec![(nid(1), &a), (nid(2), &b)].into_iter());
        let key = encode_probe_key(&[&Value::Str("legal".into())]).unwrap();
        let mut both = idx.probe_all("Doc", &names, &key).unwrap();
        both.sort();
        assert_eq!(both, vec![nid(1), nid(2)]);
        {
            let state = idx.state.lock().unwrap();
            let map = state.maps.get(&("Doc".into(), names.clone())).unwrap();
            assert!(matches!(map.holders.get(&key), Some(Holders::Many(_))));
        }

        idx.apply_tombstone(nid(1));
        assert_eq!(idx.probe_all("Doc", &names, &key), Some(vec![nid(2)]));
        {
            let state = idx.state.lock().unwrap();
            let map = state.maps.get(&("Doc".into(), names.clone())).unwrap();
            assert!(matches!(
                map.holders.get(&key),
                Some(Holders::One(id)) if *id == nid(2)
            ));
        }

        idx.apply_tombstone(nid(2));
        assert_eq!(idx.probe_all("Doc", &names, &key), Some(Vec::new()));
    }

    #[test]
    fn partial_sidecar_seed_answers_only_known_keys_and_rolls_back_mutations() {
        let idx = UniqueConstraintIndex::new();
        let names = vec!["key".to_string()];
        let key_a = encode_probe_key(&[&Value::Str("a".into())]).unwrap();
        let key_b = encode_probe_key(&[&Value::Str("b".into())]).unwrap();
        let key_unknown = encode_probe_key(&[&Value::Str("unknown".into())]).unwrap();
        idx.seed_committed_keys(
            "Account",
            &names,
            vec![(key_a.clone(), Some(nid(1))), (key_b.clone(), None)],
        );
        {
            let state = idx.state.lock().unwrap();
            let map = state.maps.get(&("Account".into(), names.clone())).unwrap();
            assert!(
                !map.known_misses.contains(&key_a),
                "a seeded hit must not retain a third copy of its key"
            );
            assert!(map.known_misses.contains(&key_b));
        }

        assert_eq!(
            idx.probe("Account", &names, &key_a, None),
            Some(UniqueProbe::Conflict(nid(1)))
        );
        assert_eq!(
            idx.probe("Account", &names, &key_b, None),
            Some(UniqueProbe::NoConflict),
            "seeded misses are authoritative"
        );
        assert_eq!(
            idx.probe("Account", &names, &key_unknown, None),
            None,
            "an unseeded key must retain the full-population fallback"
        );

        let b = props(&[("key", Value::Str("b".into()))]);
        idx.apply_upsert(nid(2), &["Account"], &b);
        assert_eq!(
            idx.probe("Account", &names, &key_b, None),
            Some(UniqueProbe::Conflict(nid(2)))
        );
        idx.rollback_staged();
        assert_eq!(
            idx.probe("Account", &names, &key_b, None),
            Some(UniqueProbe::NoConflict),
            "rollback restores the seeded committed miss"
        );

        let unknown = props(&[("key", Value::Str("unknown".into()))]);
        idx.apply_upsert(nid(1), &["Account"], &unknown);
        assert_eq!(
            idx.probe("Account", &names, &key_unknown, None),
            Some(UniqueProbe::Conflict(nid(1)))
        );
        idx.rollback_staged();
        assert_eq!(
            idx.probe("Account", &names, &key_a, None),
            Some(UniqueProbe::Conflict(nid(1))),
            "rollback restores the original seeded hit"
        );
        assert_eq!(
            idx.probe("Account", &names, &key_unknown, None),
            None,
            "a key introduced only by a rolled-back write remains unknown"
        );
        assert_eq!(idx.populate_scans(), 0);
    }

    #[test]
    fn staged_point_seed_is_disposable_and_rolls_back_overlay_only_misses() {
        let names = vec!["key".to_string()];
        let staged_key = encode_probe_key(&[&Value::Str("staged".into())]).unwrap();

        // With no committed map, every answer was learned from an already
        // mutated overlay. Rollback must drop the whole partial map.
        let fresh = UniqueConstraintIndex::new();
        fresh.seed_staged_keys("Account", &names, vec![(staged_key.clone(), Some(nid(1)))]);
        assert_eq!(
            fresh.probe("Account", &names, &staged_key, None),
            Some(UniqueProbe::Conflict(nid(1)))
        );
        fresh.rollback_staged();
        assert_eq!(fresh.probe("Account", &names, &staged_key, None), None);

        // A pre-existing partial map can lack the old tuple of a node deleted
        // during this batch. A subsequently seeded overlay miss is valid now,
        // but must become unknown (not a false committed miss) on rollback.
        let warm = UniqueConstraintIndex::new();
        let known_key = encode_probe_key(&[&Value::Str("known".into())]).unwrap();
        let deleted_key = encode_probe_key(&[&Value::Str("deleted".into())]).unwrap();
        warm.seed_committed_keys("Account", &names, vec![(known_key, Some(nid(1)))]);
        warm.apply_tombstone(nid(2));
        warm.seed_staged_keys("Account", &names, vec![(deleted_key.clone(), None)]);
        assert_eq!(
            warm.probe("Account", &names, &deleted_key, None),
            Some(UniqueProbe::NoConflict)
        );
        warm.rollback_staged();
        assert_eq!(
            warm.probe("Account", &names, &deleted_key, None),
            None,
            "rollback must forget a miss that depended on an unknown staged tombstone"
        );
    }

    #[test]
    fn successful_flush_reclamation_drops_only_partial_maps() {
        let idx = UniqueConstraintIndex::new();
        let names = vec!["key".to_string()];
        let partial_key = encode_probe_key(&[&Value::Str("partial".into())]).unwrap();
        idx.seed_committed_keys("Partial", &names, vec![(partial_key.clone(), Some(nid(1)))]);

        let full_props = props(&[("key", Value::Str("full".into()))]);
        let full_key = encode_probe_key(&[&Value::Str("full".into())]).unwrap();
        idx.populate("Full", &names, vec![(nid(2), &full_props)].into_iter());

        idx.drop_partial_maps();
        assert_eq!(idx.probe("Partial", &names, &partial_key, None), None);
        assert_eq!(
            idx.probe("Full", &names, &full_key, None),
            Some(UniqueProbe::Conflict(nid(2)))
        );
    }

    #[test]
    fn rollback_restores_warm_map_and_commit_advances_its_baseline() {
        let idx = UniqueConstraintIndex::new();
        let names = vec!["email".to_string()];
        let a = props(&[("email", Value::Str("a@x".into()))]);
        idx.populate("User", &names, vec![(nid(1), &a)].into_iter());
        let key_a = encode_probe_key(&[&Value::Str("a@x".into())]).unwrap();

        // Rolling back an empty batch is a true no-op: the populated map
        // remains available and still describes committed state.
        idx.rollback_staged();
        assert_eq!(
            idx.probe("User", &names, &key_a, None),
            Some(UniqueProbe::Conflict(nid(1)))
        );

        let b = props(&[("email", Value::Str("b@x".into()))]);
        idx.apply_upsert(nid(1), &["User"], &b);
        idx.apply_upsert(nid(2), &["User"], &a);
        idx.rollback_staged();

        let key_b = encode_probe_key(&[&Value::Str("b@x".into())]).unwrap();
        assert_eq!(
            idx.probe("User", &names, &key_a, None),
            Some(UniqueProbe::Conflict(nid(1)))
        );
        assert_eq!(
            idx.probe("User", &names, &key_b, None),
            Some(UniqueProbe::NoConflict)
        );

        // A committed mutation becomes the next rollback baseline.
        idx.apply_upsert(nid(1), &["User"], &b);
        idx.commit_staged();
        idx.apply_tombstone(nid(1));
        idx.rollback_staged();
        assert_eq!(
            idx.probe("User", &names, &key_b, None),
            Some(UniqueProbe::Conflict(nid(1)))
        );
    }

    #[test]
    fn rollback_drops_map_first_populated_from_staged_overlay() {
        let idx = UniqueConstraintIndex::new();
        let names = vec!["email".to_string()];
        let staged = props(&[("email", Value::Str("staged@x".into()))]);
        idx.populate_staged("User", &names, vec![(nid(1), &staged)].into_iter());
        let key = encode_probe_key(&[&Value::Str("staged@x".into())]).unwrap();
        assert!(idx.probe("User", &names, &key, None).is_some());

        idx.rollback_staged();
        assert_eq!(
            idx.probe("User", &names, &key, None),
            None,
            "no committed baseline existed for this map"
        );
    }

    #[test]
    fn preexisting_duplicates_keep_conflicting_like_the_scan() {
        // Two nodes already carry the same value (constraint declared over
        // violating data). Excluding one must still surface the other.
        let idx = UniqueConstraintIndex::new();
        let names = vec!["code".to_string()];
        let v = props(&[("code", Value::I64(7))]);
        idx.populate("A", &names, vec![(nid(1), &v), (nid(2), &v)].into_iter());
        let key = encode_probe_key(&[&Value::I64(7)]).unwrap();
        assert_eq!(
            idx.probe("A", &names, &key, Some(nid(1))),
            Some(UniqueProbe::Conflict(nid(2)))
        );
    }

    #[test]
    fn label_scoping_and_incomplete_tuples() {
        let idx = UniqueConstraintIndex::new();
        let names = vec!["a".to_string(), "b".to_string()];
        idx.populate("L", &names, std::iter::empty());

        // A node not carrying the constraint's label is never filed.
        let p = props(&[("a", Value::I64(1)), ("b", Value::I64(2))]);
        idx.apply_upsert(nid(1), &["Other"], &p);
        let key = encode_probe_key(&[&Value::I64(1), &Value::I64(2)]).unwrap();
        assert_eq!(
            idx.probe("L", &names, &key, None),
            Some(UniqueProbe::NoConflict)
        );

        // Missing tuple element → not filed either.
        let partial = props(&[("a", Value::I64(1))]);
        idx.apply_upsert(nid(2), &["L"], &partial);
        assert_eq!(
            idx.probe("L", &names, &key, None),
            Some(UniqueProbe::NoConflict)
        );

        // Complete tuple on the right label → conflict.
        idx.apply_upsert(nid(3), &["L"], &p);
        assert_eq!(
            idx.probe("L", &names, &key, None),
            Some(UniqueProbe::Conflict(nid(3)))
        );
    }

    #[test]
    fn unpopulated_probe_reports_none() {
        let idx = UniqueConstraintIndex::new();
        let names = vec!["x".to_string()];
        let key = encode_probe_key(&[&Value::I64(1)]).unwrap();
        assert_eq!(idx.probe("L", &names, &key, None), None);
        assert_eq!(idx.probes(), 0);
    }
}
