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
            }
        }
    }

    fn file(&mut self, id: NodeId, key: UniqueKey) {
        debug_assert!(!self.by_node.contains_key(&id));
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
}

#[derive(Debug, Default)]
struct IndexState {
    maps: HashMap<ConstraintId, ConstraintMap>,
    undo: HashMap<ConstraintId, ConstraintUndo>,
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
        self.populate_scans.fetch_add(1, Ordering::Relaxed);
        let identity = (label.to_string(), names.to_vec());
        let mut state = self.state.lock().expect("unique index lock");
        state.maps.insert(identity.clone(), map);
        if staged {
            let undo = state.undo.entry(identity).or_default();
            undo.created_in_batch = true;
            undo.by_node.clear();
        } else {
            state.undo.remove(&identity);
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
        let IndexState { maps, undo } = &mut *state;
        for (identity @ (clabel, cnames), map) in maps.iter_mut() {
            let rollback = undo.entry(identity.clone()).or_default();
            if !rollback.created_in_batch {
                rollback
                    .by_node
                    .entry(id)
                    .or_insert_with(|| map.by_node.get(&id).cloned());
            }
            map.detach(id);
            // An empty label is the physical any-label scope used by
            // `MATCH (n {prop: ...})`. Every node, including an unlabelled one,
            // belongs to that global postings map.
            if !clabel.is_empty() && !labels.iter().any(|l| l == clabel) {
                continue;
            }
            if let Some(key) = encode_node_key(cnames, props) {
                map.file(id, key);
            }
        }
    }

    /// Maintain every populated map for a staged node tombstone.
    pub(crate) fn apply_tombstone(&self, id: NodeId) {
        let mut state = self.state.lock().expect("unique index lock");
        let IndexState { maps, undo } = &mut *state;
        for (identity, map) in maps.iter_mut() {
            let rollback = undo.entry(identity.clone()).or_default();
            if !rollback.created_in_batch {
                rollback
                    .by_node
                    .entry(id)
                    .or_insert_with(|| map.by_node.get(&id).cloned());
            }
            map.detach(id);
        }
    }

    /// Promote the current staged view to committed state. The populated maps
    /// already contain those mutations, so only the rollback journal changes.
    pub(crate) fn commit_staged(&self) {
        self.state.lock().expect("unique index lock").undo.clear();
    }

    /// Restore every populated map to its pre-batch committed state without a
    /// label scan. This is proportional to the number of distinct node ids
    /// touched in the discarded batch, not to the stored graph size.
    pub(crate) fn rollback_staged(&self) {
        let mut state = self.state.lock().expect("unique index lock");
        let undo = std::mem::take(&mut state.undo);
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
        }
    }

    /// Drop every populated map; the next probe repopulates from a scan.
    pub(crate) fn reset(&self) {
        let mut state = self.state.lock().expect("unique index lock");
        state.maps.clear();
        state.undo.clear();
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
