//! Cross-snapshot in-memory index over `(label, property) → value → NodeId`.
//!
//! Populated lazy on the first `Snapshot::lookup_node_by_property` call
//! per (label, property) pair, then reused across every subsequent
//! snapshot the same `WriterSession` emits — so the LDBC SNB anchor
//! pattern `MATCH (a:Person {id: '...'})` pays the index-build cost
//! exactly once and then becomes an O(1) `HashMap::get` for every
//! warm query that follows.
//!
//! Design:
//! - Keyed by the **string representation** of the property value.
//!   v0 covers LDBC's `id` (always a String); a future bump can add
//!   typed key support for Int64 / Float / etc.
//! - Stored as `Arc<HashMap<String, NodeId>>` so reader-side lookups
//!   don't hold the global `RwLock` while probing.
//! - Negative answers (value not in index) are O(1) `HashMap::get(None)`
//!   — the absence is authoritative under the invariant that the
//!   property is declared `unique` and the index has been populated.
//!
//! Trade-offs:
//! - Memory: ~24 bytes per index entry (HashMap overhead + String key
//!   pointer + NodeId). 10 K Person rows ≈ 240 KiB; 1 M ≈ 24 MiB.
//!   Comfortable on a CCX13 with 8 GiB RAM.
//! - Build time: one full label scan on the first miss. Warm queries
//!   amortise it; cold-from-zero callers pay it on the first request,
//!   which is the right place to pay it.
//! - Invalidation: the cache is tied to a `WriterSession` and advances a
//!   logical generation only when node data changes. Edge-only commits and
//!   representation-only flushes preserve the generation and remain hot.

use std::collections::{BTreeSet, HashMap};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, RwLock};

use imbl::{HashMap as PersistentHashMap, OrdSet};
use namidb_core::id::NodeId;

/// `(label, property)` keys mapped to a shared per-pair `value -> NodeId`
/// index. Aliased so the `RwLock` field below stays under clippy's
/// type-complexity threshold.
type PropertyIndices = HashMap<(u64, String, String), Arc<HashMap<String, NodeId>>>;
type GlobalPropertyIndices = HashMap<(u64, String), Arc<HashMap<String, Vec<NodeId>>>>;

/// One immutable-by-convention claimant posting map. Both levels are
/// persistent: carrying a populated map across an auto-commit is O(1), while
/// inserting/removing one changed node copies only the affected HAMT/B-tree
/// paths. `OrdSet` keeps postings in NodeId order for the bounded equality
/// cursor without a corpus-sized sort or linear removal for low-cardinality
/// values such as Bool.
pub(crate) type MemtableClaimantIndex = PersistentHashMap<String, OrdSet<NodeId>>;
type MemtableClaimantIndices = PersistentHashMap<(String, String), Arc<MemtableClaimantIndex>>;

/// Snapshot-pinned generation of committed-memtable claimant maps.
///
/// Logical node generations are insufficient here: a flush changes the
/// physical memtable while preserving the logical node set. A dedicated cell
/// lets a pre-flush snapshot retain its exact old map while new snapshots see
/// an empty cell backed by the newly written SST sidecars.
#[derive(Debug, Default)]
pub(crate) struct MemtableClaimantCell {
    indices: RwLock<MemtableClaimantIndices>,
    /// Lazily built vector-index tails of the same committed memtable
    /// (see [`crate::vector_tail`]), carried by commits like the claimants.
    vector_tails: RwLock<PersistentHashMap<(String, String), Arc<VectorTailIndex>>>,
    /// Per full-text index (by name), the memtable rows whose indexed text
    /// differs from the version the index covers (see [`TextChangedIndex`]).
    text_changed: RwLock<PersistentHashMap<String, Arc<TextChangedIndex>>>,
}

/// The memtable rows that change what one full-text index would see.
///
/// Every committed memtable row that carries `label` and is NOT in `changed`
/// has the same label membership and the same values of every indexed
/// property as its persisted version, so the persisted index is exact for it
/// (an embedding-only `SET` is the common case). A row is added to `changed`
/// when a commit makes its label membership or any indexed value differ from
/// the version before the commit, and never leaves it while the memtable
/// lives: by induction every row outside the set still equals its persisted
/// version.
#[derive(Debug, Clone)]
pub(crate) struct TextChangedIndex {
    pub(crate) label: String,
    pub(crate) properties: BTreeSet<String>,
    pub(crate) changed: OrdSet<NodeId>,
}

/// Labels and `(label, property)` presence pairs a count seed is asked for.
pub(crate) type CountSeedRequest = (BTreeSet<String>, BTreeSet<(String, String)>);

/// `(index name, label, indexed properties)` of a tracked full-text index.
pub(crate) type TextChangedSpec = (String, String, BTreeSet<String>);

/// The memtable tail of one vector index: the metric and dimension it was
/// encoded for, and its rows.
#[derive(Debug, Clone)]
pub(crate) struct VectorTailIndex {
    pub(crate) metric: crate::manifest::VectorMetric,
    pub(crate) dim: usize,
    pub(crate) tail: crate::vector_tail::VectorTail,
}

/// One commit's change to a tail: the id, its new winning LSN and entry.
pub(crate) type VectorTailChange = (NodeId, u64, crate::vector_tail::TailEntry);

/// The equality-indexable portion of one committed memtable node row.
///
/// This deliberately excludes vectors, maps and other values without a
/// canonical equality encoding, so a 200-row embedding SET does not duplicate
/// the embedding bytes merely to maintain its String key claimant.
#[derive(Clone, Debug, Default)]
pub(crate) struct MemtableClaimantNode {
    pub(crate) labels: BTreeSet<String>,
    pub(crate) properties: HashMap<String, String>,
}

/// One posting-list mutation for a cached `(label, property)` pair.
pub(crate) type MemtableClaimantPairChange = (NodeId, Option<String>, Option<String>);

/// A delta prepared against the exact pairs populated at commit start.
///
/// A lazy reader may install a brand-new pair while the WAL/manifest commit is
/// in flight. `captured_pairs` lets the carry step retain only pairs whose
/// values were captured; any concurrently introduced pair is dropped and
/// rebuilt once, never carried without its mutations. Changes are pre-grouped
/// by relevant pair so carry is O(pairs + affected associations), rather than
/// O(pairs × mutated nodes).
#[derive(Debug)]
pub(crate) struct MemtableClaimantDelta {
    pub(crate) changes_by_pair: HashMap<(String, String), Vec<MemtableClaimantPairChange>>,
    pub(crate) captured_pairs: BTreeSet<(String, String)>,
    pub(crate) rows: usize,
    /// Changes for every vector tail populated when the delta was prepared,
    /// keyed by `(label, property)` together with the metric and dimension
    /// they were encoded for. A tail installed later is dropped, not carried.
    pub(crate) tail_changes:
        HashMap<(String, String), (crate::manifest::VectorMetric, usize, Vec<VectorTailChange>)>,
    /// Rows whose indexed text changed in this commit, for every text index
    /// tracked when the delta was prepared, with the spec it was computed for.
    pub(crate) text_changes: HashMap<String, (TextChangedSpec, Vec<NodeId>)>,
}

impl MemtableClaimantNode {
    pub(crate) fn value_for<'a>(&'a self, label: &str, property: &str) -> Option<&'a String> {
        if !label.is_empty() && !self.labels.contains(label) {
            return None;
        }
        self.properties.get(property)
    }
}

impl MemtableClaimantCell {
    /// Carry every lazily populated `(label, property)` map to the next
    /// committed memtable using only the final rows touched by the batch.
    fn carry(&self, delta: &MemtableClaimantDelta) -> Arc<Self> {
        let source = self
            .indices
            .read()
            .map(|indices| indices.clone())
            .unwrap_or_else(|poisoned| poisoned.into_inner().clone());
        let source_tails = self
            .vector_tails
            .read()
            .map(|tails| tails.clone())
            .unwrap_or_else(|poisoned| poisoned.into_inner().clone());
        let source_text = self
            .text_changed
            .read()
            .map(|text| text.clone())
            .unwrap_or_else(|poisoned| poisoned.into_inner().clone());
        if source.is_empty() && source_tails.is_empty() && source_text.is_empty() {
            return Arc::new(Self::default());
        }

        let mut carried = PersistentHashMap::new();
        for ((label, property), index) in source.iter() {
            let pair = (label.clone(), property.clone());
            if !delta.captured_pairs.contains(&pair) {
                // Installed after delta preparation; force one exact rebuild
                // instead of carrying an unpatched map.
                continue;
            }
            let Some(changes) = delta.changes_by_pair.get(&pair) else {
                carried.insert(pair, Arc::clone(index));
                continue;
            };
            let mut updated = index.as_ref().clone();
            for (id, old_value, new_value) in changes {
                if let Some(value) = old_value {
                    if let Some(mut ids) = updated.get(value).cloned() {
                        ids.remove(id);
                        if ids.is_empty() {
                            updated.remove(value);
                        } else {
                            updated.insert(value.clone(), ids);
                        }
                    }
                }
                if let Some(value) = new_value {
                    let mut ids = updated.get(value).cloned().unwrap_or_default();
                    ids.insert(*id);
                    updated.insert(value.clone(), ids);
                }
            }
            carried.insert(pair, Arc::new(updated));
        }
        let mut carried_tails = PersistentHashMap::new();
        for (pair, tail) in source_tails.iter() {
            // Only a tail whose changes were prepared with its own encoding
            // can be carried; anything else is rebuilt on its next query.
            let Some((metric, dim, changes)) = delta.tail_changes.get(pair) else {
                continue;
            };
            if *metric != tail.metric || *dim != tail.dim {
                continue;
            }
            let mut updated = tail.as_ref().clone();
            for (id, lsn, entry) in changes {
                updated.tail.insert(*id, (*lsn, entry.clone()));
            }
            carried_tails.insert(pair.clone(), Arc::new(updated));
        }
        let mut carried_text = PersistentHashMap::new();
        for (name, tracked) in source_text.iter() {
            // Same rule as the tails: only an index whose changes were
            // prepared for the same label and properties is carried.
            let Some(((_, label, properties), changed)) = delta.text_changes.get(name) else {
                continue;
            };
            if *label != tracked.label || *properties != tracked.properties {
                continue;
            }
            let mut updated = tracked.as_ref().clone();
            for id in changed {
                updated.changed.insert(*id);
            }
            carried_text.insert(name.clone(), Arc::new(updated));
        }
        Arc::new(Self {
            indices: RwLock::new(carried),
            vector_tails: RwLock::new(carried_tails),
            text_changed: RwLock::new(carried_text),
        })
    }

    /// The changed rows of full-text index `name` when they were computed
    /// for `label` and `properties`.
    #[cfg_attr(not(feature = "text-index"), allow(dead_code))]
    pub(crate) fn text_changed(
        &self,
        name: &str,
        label: &str,
        properties: &BTreeSet<String>,
    ) -> Option<Arc<TextChangedIndex>> {
        let text = self.text_changed.read().ok()?;
        text.get(name)
            .filter(|tracked| tracked.label == label && tracked.properties == *properties)
            .cloned()
    }

    /// Install freshly computed changed rows unless some are already there.
    #[cfg_attr(not(feature = "text-index"), allow(dead_code))]
    pub(crate) fn install_text_changed(
        &self,
        name: &str,
        tracked: Arc<TextChangedIndex>,
    ) -> Arc<TextChangedIndex> {
        let Ok(mut text) = self.text_changed.write() else {
            return tracked;
        };
        if let Some(existing) = text.get(name).filter(|existing| {
            existing.label == tracked.label && existing.properties == tracked.properties
        }) {
            return Arc::clone(existing);
        }
        text.insert(name.to_string(), Arc::clone(&tracked));
        tracked
    }

    /// The spec of every tracked full-text index.
    pub(crate) fn text_changed_specs(&self) -> Vec<TextChangedSpec> {
        self.text_changed
            .read()
            .map(|text| {
                text.iter()
                    .map(|(name, tracked)| {
                        (
                            name.clone(),
                            tracked.label.clone(),
                            tracked.properties.clone(),
                        )
                    })
                    .collect()
            })
            .unwrap_or_default()
    }

    /// The tail of `(label, property)` when it was built for `metric` and
    /// `dim`.
    #[cfg_attr(not(feature = "vector-index"), allow(dead_code))]
    pub(crate) fn vector_tail(
        &self,
        label: &str,
        property: &str,
        metric: crate::manifest::VectorMetric,
        dim: usize,
    ) -> Option<Arc<VectorTailIndex>> {
        let tails = self.vector_tails.read().ok()?;
        tails
            .get(&(label.to_string(), property.to_string()))
            .filter(|tail| tail.metric == metric && tail.dim == dim)
            .cloned()
    }

    /// Install a freshly built tail unless one is already there.
    #[cfg_attr(not(feature = "vector-index"), allow(dead_code))]
    pub(crate) fn install_vector_tail(
        &self,
        label: &str,
        property: &str,
        tail: Arc<VectorTailIndex>,
    ) -> Arc<VectorTailIndex> {
        let Ok(mut tails) = self.vector_tails.write() else {
            return tail;
        };
        let key = (label.to_string(), property.to_string());
        if let Some(existing) = tails
            .get(&key)
            .filter(|existing| existing.metric == tail.metric && existing.dim == tail.dim)
        {
            return Arc::clone(existing);
        }
        tails.insert(key, Arc::clone(&tail));
        tail
    }

    /// `(label, property, metric, dim)` of every tail built so far.
    pub(crate) fn vector_tail_specs(
        &self,
    ) -> Vec<((String, String), crate::manifest::VectorMetric, usize)> {
        self.vector_tails
            .read()
            .map(|tails| {
                tails
                    .iter()
                    .map(|(pair, tail)| (pair.clone(), tail.metric, tail.dim))
                    .collect()
            })
            .unwrap_or_default()
    }

    #[cfg(test)]
    fn is_empty(&self) -> bool {
        self.indices
            .read()
            .map(|indices| indices.is_empty())
            .unwrap_or(false)
    }
}

/// Exact logical node cardinalities for one generation. The writer cache keeps
/// only the current cell; older cells live only while an immutable snapshot is
/// pinned, so memory is bounded by labels × live generations rather than nodes
/// or lifetime commits.
///
/// Knowledge may be partial. A full reconciliation (or authoritative manifest
/// metadata) seeds `total` and every label (`labels_complete`). A background
/// label seed ([`PropertyIndexCache::begin_count_seed`]) seeds one label and the
/// presence of some of its properties without knowing the rest of the graph.
/// Every commit then applies its exact old/new deltas to whatever is known.
#[derive(Clone, Debug, Default)]
struct ExactNodeCounts {
    /// Total live nodes, when known.
    total: Option<u64>,
    /// Exact live nodes per label. When `labels_complete`, a label missing
    /// here has none; otherwise only the listed labels are known.
    by_label: HashMap<String, u64>,
    labels_complete: bool,
    /// Live nodes of `label` whose `property` is present and not null, for
    /// the listed pairs. Every listed pair's label is known in `by_label`.
    presence: HashMap<(String, String), u64>,
}

impl ExactNodeCounts {
    fn label(&self, label: &str) -> Option<u64> {
        match self.by_label.get(label) {
            Some(count) => Some(*count),
            None if self.labels_complete => Some(0),
            None => None,
        }
    }

    fn is_empty(&self) -> bool {
        self.total.is_none() && self.by_label.is_empty() && !self.labels_complete
    }

    /// Exact: no impossible relation between the known counts.
    fn consistent(&self) -> bool {
        if let Some(total) = self.total {
            if self.by_label.values().any(|count| *count > total) {
                return false;
            }
        }
        self.presence.iter().all(|((label, _), present)| {
            self.label(label)
                .is_some_and(|label_count| *present <= label_count)
        })
    }
}

/// Labels and tracked-property presence of one node version, as seen by the
/// node-count ledger. `present` lists only properties the ledger tracks.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct NodeCountState {
    pub(crate) labels: BTreeSet<String>,
    pub(crate) present: BTreeSet<String>,
}

impl NodeCountState {
    /// The state of a node version: its labels, and which of `tracked`
    /// properties it carries with a non-null value.
    pub(crate) fn of<'a>(
        labels: BTreeSet<String>,
        tracked: &BTreeSet<String>,
        mut property: impl FnMut(&str) -> Option<&'a namidb_core::Value>,
    ) -> Self {
        let present = tracked
            .iter()
            .filter(|name| {
                property(name.as_str())
                    .is_some_and(|value| !matches!(value, namidb_core::Value::Null))
            })
            .cloned()
            .collect();
        Self { labels, present }
    }

    fn has(&self, label: &str, property: &str) -> bool {
        self.labels.contains(label) && self.present.contains(property)
    }
}

/// Final `(before, after)` state of every node one commit changed. `None` is
/// absence (create / delete). `tracked` is the property set the states were
/// computed against: presence of any other property is unknown.
#[derive(Clone, Debug, Default)]
pub(crate) struct NodeCountChanges {
    pub(crate) tracked: BTreeSet<String>,
    pub(crate) changes: Vec<(Option<NodeCountState>, Option<NodeCountState>)>,
}

/// Snapshot-stable, lazily populated exact cardinalities.
///
/// Each logical node generation owns one cell. Published snapshots retain its
/// `Arc`, so a later node commit can install a new generation without making
/// an older pinned snapshot observe new counts or fall back to a corpus scan.
/// Representation-only cache eviction reuses the same cell.
#[derive(Debug, Default)]
pub(crate) struct ExactNodeCountCell {
    counts: RwLock<Option<Arc<ExactNodeCounts>>>,
}

impl ExactNodeCountCell {
    pub(crate) fn count(&self, label: Option<&str>) -> Option<u64> {
        let guard = self.counts.read().ok()?;
        let counts = guard.as_ref()?;
        match label {
            None => counts.total,
            Some(label) => counts.label(label),
        }
    }

    /// Exact live nodes of `label` whose `property` is present and not null,
    /// when this generation tracks the pair.
    pub(crate) fn presence(&self, label: &str, property: &str) -> Option<u64> {
        let guard = self.counts.read().ok()?;
        guard
            .as_ref()?
            .presence
            .get(&(label.to_string(), property.to_string()))
            .copied()
    }

    fn tracked_properties(&self) -> BTreeSet<String> {
        self.counts
            .read()
            .ok()
            .and_then(|guard| {
                guard.as_ref().map(|counts| {
                    counts
                        .presence
                        .keys()
                        .map(|(_, property)| property.clone())
                        .collect()
                })
            })
            .unwrap_or_default()
    }

    fn snapshot(&self) -> Option<ExactNodeCounts> {
        self.counts
            .read()
            .ok()
            .and_then(|guard| guard.as_ref().map(|counts| counts.as_ref().clone()))
    }

    /// Install a complete total + per-label vector (manifest metadata or a
    /// full reconciliation). Presence already known for this generation is
    /// kept when its label count agrees.
    pub(crate) fn install(&self, total: u64, by_label: HashMap<String, u64>) {
        if let Ok(mut guard) = self.counts.write() {
            if by_label.values().any(|count| *count == 0 || *count > total) {
                *guard = None;
                return;
            }
            let previous = guard.as_ref().map(|counts| counts.as_ref().clone());
            let mut counts = ExactNodeCounts {
                total: Some(total),
                by_label,
                labels_complete: true,
                presence: HashMap::new(),
            };
            if let Some(previous) = previous {
                for (pair, present) in previous.presence {
                    if previous.by_label.get(&pair.0) == counts.by_label.get(&pair.0)
                        || (counts.label(&pair.0) == Some(0)
                            && !previous.by_label.contains_key(&pair.0))
                    {
                        counts.presence.insert(pair, present);
                    }
                }
            }
            *guard = counts.consistent().then(|| Arc::new(counts));
        }
    }

    fn install_snapshot(&self, counts: Option<ExactNodeCounts>) {
        if let Ok(mut guard) = self.counts.write() {
            *guard = counts.filter(|counts| !counts.is_empty()).map(Arc::new);
        }
    }

    /// Read-modify-write the counts under the write lock. `merge` sees what
    /// the cell holds now (a full vector [`install`](Self::install) may have
    /// written a moment ago, for instance) and returns what to keep, or
    /// `None` to leave the cell unchanged. Reading and writing separately
    /// would let a concurrent install land in between and be overwritten.
    fn update<R>(
        &self,
        merge: impl FnOnce(Option<ExactNodeCounts>) -> (Option<Option<ExactNodeCounts>>, R),
    ) -> Option<R> {
        let mut guard = self.counts.write().ok()?;
        let current = guard.as_ref().map(|counts| counts.as_ref().clone());
        let (next, result) = merge(current);
        if let Some(next) = next {
            *guard = next.filter(|counts| !counts.is_empty()).map(Arc::new);
        }
        Some(result)
    }
}

#[cfg(test)]
thread_local! {
    /// Test hook: runs in [`PropertyIndexCache::finish_count_seed`] once the
    /// seed is claimed, right before its counts are merged into the cell.
    static BEFORE_SEED_MERGE: std::cell::RefCell<Option<Box<dyn FnOnce()>>> =
        std::cell::RefCell::new(None);
}

/// What merging a finished seed into a cell's counts gave.
enum SeedMerge {
    /// The seed's counts were merged in.
    Merged,
    /// A label count the cell already knew disagrees with the seed: one of
    /// them is wrong, so the cell must be cleared.
    Disagrees,
    /// The seed cannot be applied (missing scan results, an impossible roll
    /// forward, or an inconsistent outcome): leave the cell as it is.
    Unusable,
}

/// Roll `result` forward over the commits recorded while the seed scanned
/// and merge it into `counts`, which hold the cell's current contents.
fn merge_count_seed(
    counts: &mut ExactNodeCounts,
    seed: &CountSeed,
    result: &CountSeedResult,
) -> SeedMerge {
    for label in &seed.labels {
        let Some(scanned) = result.by_label.get(label) else {
            return SeedMerge::Unusable;
        };
        let delta = seed.delta.by_label.get(label).copied().unwrap_or(0);
        let Some(now) = roll_forward(*scanned, delta) else {
            return SeedMerge::Unusable;
        };
        match counts.label(label) {
            // Already known (and kept exact by every commit since): the
            // two must agree, or one of them is wrong.
            Some(known) if known != now => {
                tracing::warn!(
                    label,
                    known,
                    seeded = now,
                    "node-count seed disagrees; discarding counts"
                );
                return SeedMerge::Disagrees;
            }
            Some(_) => {}
            None => {
                counts.by_label.insert(label.clone(), now);
            }
        }
    }
    for pair in &seed.pairs {
        if counts.presence.len() >= MAX_TRACKED_PRESENCE_PAIRS
            && !counts.presence.contains_key(pair)
        {
            break;
        }
        let Some(scanned) = result.presence.get(pair) else {
            return SeedMerge::Unusable;
        };
        let delta = seed.delta.presence.get(pair).copied().unwrap_or(0);
        let Some(now) = roll_forward(*scanned, delta) else {
            return SeedMerge::Unusable;
        };
        counts.presence.insert(pair.clone(), now);
    }
    if !counts.consistent() {
        return SeedMerge::Unusable;
    }
    SeedMerge::Merged
}

/// Apply a committed node batch to one exact cardinality vector.
///
/// Count metadata is an optimisation, so an impossible delta must discard it
/// and force the exact read-path reconciliation. Saturating or wrapping here
/// would be worse than a scan: it could publish a plausible but stale Cypher
/// `count(*)`. The writer guarantees one final change per node, but this
/// defensive validation also protects migrations from corrupt/legacy seed
/// metadata.
///
/// A tracked presence pair whose property the batch did not compute (it was
/// installed between the batch's preparation and its commit) is forgotten,
/// not guessed.
fn apply_node_count_changes(
    mut counts: ExactNodeCounts,
    changes: &NodeCountChanges,
) -> Option<ExactNodeCounts> {
    counts
        .presence
        .retain(|(_, property), _| changes.tracked.contains(property));
    for (old, new) in &changes.changes {
        if let Some(total) = &mut counts.total {
            match (old.is_some(), new.is_some()) {
                (false, true) => *total = total.checked_add(1)?,
                (true, false) => *total = total.checked_sub(1)?,
                _ => {}
            }
        }

        if let Some(old) = old {
            for label in &old.labels {
                if new.as_ref().is_some_and(|new| new.labels.contains(label)) {
                    continue;
                }
                match counts.by_label.get_mut(label) {
                    Some(count) => {
                        *count = count.checked_sub(1)?;
                        if *count == 0 && counts.labels_complete {
                            counts.by_label.remove(label);
                        }
                    }
                    // Complete knowledge says the label has no node, yet a
                    // node carried it.
                    None if counts.labels_complete => return None,
                    None => {}
                }
            }
        }
        if let Some(new) = new {
            for label in &new.labels {
                if old.as_ref().is_some_and(|old| old.labels.contains(label)) {
                    continue;
                }
                match counts.by_label.get_mut(label) {
                    Some(count) => *count = count.checked_add(1)?,
                    None if counts.labels_complete => {
                        counts.by_label.insert(label.clone(), 1);
                    }
                    None => {}
                }
            }
        }
        for ((label, property), present) in &mut counts.presence {
            let before = old.as_ref().is_some_and(|old| old.has(label, property));
            let after = new.as_ref().is_some_and(|new| new.has(label, property));
            match (before, after) {
                (false, true) => *present = present.checked_add(1)?,
                (true, false) => *present = present.checked_sub(1)?,
                _ => {}
            }
        }
    }

    counts.consistent().then_some(counts)
}

/// Signed change of the counts a running seed is computing, accumulated from
/// the commits that land while it scans.
#[derive(Debug, Default)]
struct CountSeedDelta {
    by_label: HashMap<String, i64>,
    presence: HashMap<(String, String), i64>,
}

/// One background label seed: it scans `labels` at `generation` and the
/// commits after that generation are folded into `delta`, so its result can
/// be rolled forward to whatever generation is current when it finishes.
#[derive(Debug)]
struct CountSeed {
    id: u64,
    labels: BTreeSet<String>,
    pairs: BTreeSet<(String, String)>,
    delta: CountSeedDelta,
    /// A commit whose delta is unknown landed meanwhile.
    poisoned: bool,
}

impl CountSeed {
    fn fold(&mut self, changes: Option<&NodeCountChanges>) {
        let Some(changes) = changes else {
            self.poisoned = true;
            return;
        };
        if self
            .pairs
            .iter()
            .any(|(_, property)| !changes.tracked.contains(property))
        {
            self.poisoned = true;
            return;
        }
        for (old, new) in &changes.changes {
            for label in &self.labels {
                let before = old.as_ref().is_some_and(|old| old.labels.contains(label));
                let after = new.as_ref().is_some_and(|new| new.labels.contains(label));
                if before != after {
                    *self.delta.by_label.entry(label.clone()).or_default() +=
                        if after { 1 } else { -1 };
                }
            }
            for (label, property) in &self.pairs {
                let before = old.as_ref().is_some_and(|old| old.has(label, property));
                let after = new.as_ref().is_some_and(|new| new.has(label, property));
                if before != after {
                    *self
                        .delta
                        .presence
                        .entry((label.clone(), property.clone()))
                        .or_default() += if after { 1 } else { -1 };
                }
            }
        }
    }
}

/// Exact counts a seed scan measured at its start generation.
#[derive(Debug, Default)]
pub(crate) struct CountSeedResult {
    pub(crate) by_label: HashMap<String, u64>,
    pub(crate) presence: HashMap<(String, String), u64>,
}

/// Upper bound on presence pairs the ledger tracks, so the per-commit delta
/// stays a handful of map probes per changed node.
pub(crate) const MAX_TRACKED_PRESENCE_PAIRS: usize = 64;

fn roll_forward(value: u64, delta: i64) -> Option<u64> {
    if delta >= 0 {
        value.checked_add(delta as u64)
    } else {
        value.checked_sub(delta.unsigned_abs())
    }
}

/// Process-unique id of one [`PropertyIndexCache`], i.e. of one writer
/// session. Generations are seeded from the manifest version, which restarts
/// when a namespace is dropped and created again, so a key that must never
/// match another session's data (another store, or an earlier incarnation of
/// the same namespace in this process) carries this id too.
#[derive(Debug)]
struct CacheInstance(u64);

impl Default for CacheInstance {
    fn default() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(1);
        Self(NEXT.fetch_add(1, Ordering::Relaxed))
    }
}

/// Shared cache that lives at `WriterSession` scope and is cloned (as
/// an `Arc`) into every `Snapshot` the session emits.
#[derive(Debug, Default)]
pub struct PropertyIndexCache {
    /// Which session this cache belongs to; see [`CacheInstance`].
    instance: CacheInstance,
    /// `(label_name, property_name) → Arc<value_string → NodeId>`.
    /// `Arc` on the inner so readers can release the outer lock as
    /// soon as they have the per-(label, prop) handle.
    indices: RwLock<PropertyIndices>,
    /// Per-(label, property) claimants from the current committed memtable.
    ///
    /// SST sidecars already provide an immutable map per file. Without this
    /// companion index, every sidecar-backed point lookup still scanned every
    /// node buffered since the last flush, turning a 2,000-key sweep into
    /// O(keys × memtable_nodes). The current cell is replaced incrementally on
    /// node commits and reset on flush/pressure; immutable snapshots pin their
    /// own cell so representation changes cannot create false negatives.
    memtable_claimants: RwLock<Arc<MemtableClaimantCell>>,
    /// Complete label-agnostic fallback indexes. These are populated only
    /// when an older SST lacks the global equality sidecar; one all-node scan
    /// amortises every correlated `(n {key})` lookup that follows.
    global_indices: RwLock<GlobalPropertyIndices>,
    /// Logical node-data generation. Every snapshot captures this value when
    /// it is built and may only read/insert entries under that generation.
    /// This prevents an old pinned reader from repopulating the shared cache
    /// after a newer node commit invalidated it.
    generation: AtomicU64,
    /// Serialises generation transitions that carry or discard exact node
    /// counts. Memory pressure can run concurrently with the single writer; if
    /// it raced a node commit without this lock, both could carry generation N
    /// and the pressure pass could overwrite N+1's label deltas.
    generation_transition: std::sync::Mutex<()>,
    /// Calls routed through the unique point-lookup storage API.
    unique_lookup_calls: AtomicU64,
    /// Calls routed through the non-unique equality posting-list API.
    equality_lookup_calls: AtomicU64,
    /// Candidate node versions hydrated to confirm equality postings against
    /// the current last-write-wins view. Kept separate from lookup calls so
    /// LIMIT-pushdown tests can prove that a five-row query does not
    /// materialise an entire high-cardinality posting list.
    equality_confirmation_candidates: AtomicU64,
    /// Distinct posting candidates consumed by the k-way equality cursor.
    /// This distinguishes true LIMIT pushdown from merely postponing a full
    /// posting union before hydration.
    equality_candidates_iterated: AtomicU64,
    /// Geometric posting-prefix expansions needed to satisfy a label-scoped
    /// equality LIMIT from a global id-primary posting.
    equality_posting_widenings: AtomicU64,
    /// Range-readable equality-index bytes fetched by limited posting probes.
    equality_index_bytes_read: AtomicU64,
    /// Number of full committed-memtable scans used to populate claimant
    /// indexes. Exposed for deterministic performance regressions.
    memtable_population_scans: AtomicU64,
    /// Node memtable entries examined while building claimant indexes.
    memtable_population_rows: AtomicU64,
    /// Final staged node rows applied through persistent claimant-map deltas.
    /// This should grow with write volume, never with the accumulated
    /// committed memtable size.
    memtable_incremental_rows: AtomicU64,
    /// Cached pair/node associations actually changed by incremental carry.
    /// Unlike `memtable_incremental_rows`, this exposes whether maintenance
    /// accidentally regresses to visiting every populated pair for every row.
    memtable_incremental_associations: AtomicU64,
    /// Exact total + per-label node-count cell for the current logical
    /// generation. A commit creates a new cell by applying old/new label
    /// deltas; pinned snapshots retain the old cell independently.
    node_counts: RwLock<Option<(u64, Arc<ExactNodeCountCell>)>>,
    /// Number of full node reconciliations performed to seed `node_counts`.
    node_count_reconciliation_scans: AtomicU64,
    /// The background label seed in flight, if any (one at a time).
    count_seed: std::sync::Mutex<Option<CountSeed>>,
    /// A seed a reader asked for but could not start because a commit moved
    /// the generation past its snapshot. The writer starts it right after its
    /// next node commit, from a snapshot of exactly that generation.
    count_seed_request: std::sync::Mutex<Option<CountSeedRequest>>,
    next_count_seed_id: AtomicU64,
    /// Seeds that finished and installed their counts.
    count_seeds_installed: AtomicU64,
    /// Ordered String-index prefix reads used by `ORDER BY ... SKIP/LIMIT`.
    ordered_prefix_calls: AtomicU64,
    /// Geometric prefix expansions needed when a global id-primary sidecar's
    /// early values mostly belong to labels other than the requested one.
    ordered_prefix_widenings: AtomicU64,
    /// Distinct `(value, NodeId)` tuples consumed by ordered-prefix cursors.
    ordered_prefix_candidates_iterated: AtomicU64,
    /// Ordered-prefix candidates hydrated for last-write-wins confirmation.
    ordered_prefix_confirmation_candidates: AtomicU64,
    /// Range-readable equality-index bytes fetched by ordered prefix probes.
    ordered_prefix_index_bytes_read: AtomicU64,
}

impl PropertyIndexCache {
    pub fn new() -> Self {
        Self::new_at_generation(0)
    }

    /// Process-unique id of this cache (one per writer session).
    pub fn instance(&self) -> u64 {
        self.instance.0
    }

    /// Start a writer-local cache at a namespace-safe generation floor.
    ///
    /// Writer sessions use the current manifest version so a process-wide
    /// NodeView cache cannot confuse generation zero from a reopened session
    /// with an older session's generation zero. Subsequent node mutations
    /// advance from this floor; edge-only commits and physical flushes do not.
    pub(crate) fn new_at_generation(generation: u64) -> Self {
        Self {
            generation: AtomicU64::new(generation),
            node_counts: RwLock::new(Some((generation, Arc::new(ExactNodeCountCell::default())))),
            ..Self::default()
        }
    }

    /// Probe-only: returns `Some(handle)` when the (label, property)
    /// index has already been built, `None` otherwise. Used by the
    /// `lookup_node_by_property` hot path to short-circuit before
    /// taking the write lock + scanning.
    pub fn get(&self, label: &str, property: &str) -> Option<Arc<HashMap<String, NodeId>>> {
        self.get_at(label, property, self.generation())
    }

    pub(crate) fn get_at(
        &self,
        label: &str,
        property: &str,
        generation: u64,
    ) -> Option<Arc<HashMap<String, NodeId>>> {
        self.indices
            .read()
            .ok()?
            .get(&(generation, label.to_string(), property.to_string()))
            .cloned()
    }

    /// Insert a pre-built index. Idempotent — last write wins under a
    /// race; the contents are identical by construction so this is safe.
    pub fn insert(&self, label: String, property: String, index: Arc<HashMap<String, NodeId>>) {
        self.insert_at(label, property, index, self.generation());
    }

    pub(crate) fn insert_at(
        &self,
        label: String,
        property: String,
        index: Arc<HashMap<String, NodeId>>,
        generation: u64,
    ) {
        if generation != self.generation() {
            return;
        }
        if let Ok(mut w) = self.indices.write() {
            // Re-check after acquiring the map lock: reset() advances the
            // generation before clearing, so a reader that raced between the
            // first check and this lock must not repopulate an unreachable
            // stale generation after the clear.
            if generation == self.generation() {
                w.insert((generation, label, property), index);
            }
        }
    }

    pub(crate) fn get_memtable_claimants(
        &self,
        cell: &MemtableClaimantCell,
        label: &str,
        property: &str,
    ) -> Option<Arc<MemtableClaimantIndex>> {
        cell.indices
            .read()
            .ok()?
            .get(&(label.to_string(), property.to_string()))
            .cloned()
    }

    pub(crate) fn insert_memtable_claimants(
        &self,
        cell: &MemtableClaimantCell,
        label: String,
        property: String,
        index: Arc<MemtableClaimantIndex>,
    ) {
        if let Ok(mut w) = cell.indices.write() {
            w.insert((label, property), index);
            self.memtable_population_scans
                .fetch_add(1, Ordering::Relaxed);
        }
    }

    pub(crate) fn current_memtable_claimant_cell(&self) -> Arc<MemtableClaimantCell> {
        self.memtable_claimants
            .read()
            .map(|cell| Arc::clone(&*cell))
            .unwrap_or_else(|poisoned| Arc::clone(&*poisoned.into_inner()))
    }

    /// Whether carrying claimant deltas can benefit the next node commit.
    ///
    /// A concurrent first population after this check is harmless: the commit
    /// receives `None` and deliberately installs a fresh empty cell rather
    /// than carrying an unpatched map, so the next lookup performs one exact
    /// rebuild instead of risking a false negative.
    #[cfg(test)]
    pub(crate) fn has_memtable_claimant_indices(&self) -> bool {
        !self.current_memtable_claimant_cell().is_empty()
    }

    pub(crate) fn memtable_vector_tail_specs(
        &self,
    ) -> Vec<((String, String), crate::manifest::VectorMetric, usize)> {
        self.current_memtable_claimant_cell().vector_tail_specs()
    }

    pub(crate) fn memtable_text_changed_specs(&self) -> Vec<TextChangedSpec> {
        self.current_memtable_claimant_cell().text_changed_specs()
    }

    pub(crate) fn memtable_claimant_pairs(&self) -> BTreeSet<(String, String)> {
        let cell = self.current_memtable_claimant_cell();
        cell.indices
            .read()
            .map(|indices| indices.keys().cloned().collect())
            .unwrap_or_default()
    }

    pub(crate) fn get_global_at(
        &self,
        property: &str,
        generation: u64,
    ) -> Option<Arc<HashMap<String, Vec<NodeId>>>> {
        self.global_indices
            .read()
            .ok()?
            .get(&(generation, property.to_string()))
            .cloned()
    }

    pub(crate) fn insert_global_at(
        &self,
        property: String,
        index: Arc<HashMap<String, Vec<NodeId>>>,
        generation: u64,
    ) {
        if generation != self.generation() {
            return;
        }
        if let Ok(mut w) = self.global_indices.write() {
            if generation == self.generation() {
                w.insert((generation, property), index);
            }
        }
    }

    pub fn generation(&self) -> u64 {
        self.generation.load(Ordering::Acquire)
    }

    /// Capture a generation and both snapshot-pinned metadata cells under the
    /// same transition lock. Cache pressure can advance the logical generation
    /// while a physical flush can replace only the memtable cell; reading
    /// these independently could otherwise build a mixed snapshot.
    pub(crate) fn snapshot_generation_and_cells(
        &self,
    ) -> (u64, Arc<ExactNodeCountCell>, Arc<MemtableClaimantCell>) {
        let _transition = self
            .generation_transition
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let generation = self.generation();
        let mut counts = self
            .node_counts
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let cell = match counts.as_ref() {
            Some((cached, cell)) if *cached == generation => Arc::clone(cell),
            _ => {
                let cell = Arc::new(ExactNodeCountCell::default());
                *counts = Some((generation, Arc::clone(&cell)));
                cell
            }
        };
        let claimant_cell = self.current_memtable_claimant_cell();
        (generation, cell, claimant_cell)
    }

    /// Record one storage-level unique property lookup. Kept on the shared
    /// writer cache so regression tests can assert the chosen read path rather
    /// than merely observing an equal result from a label-scan fallback.
    pub(crate) fn record_unique_lookup(&self) {
        self.unique_lookup_calls.fetch_add(1, Ordering::Relaxed);
    }

    /// Record one storage-level non-unique equality lookup.
    pub(crate) fn record_equality_lookup(&self) {
        self.equality_lookup_calls.fetch_add(1, Ordering::Relaxed);
    }

    pub fn unique_lookup_calls(&self) -> u64 {
        self.unique_lookup_calls.load(Ordering::Relaxed)
    }

    pub fn equality_lookup_calls(&self) -> u64 {
        self.equality_lookup_calls.load(Ordering::Relaxed)
    }

    pub(crate) fn record_equality_confirmation_candidates(&self, count: usize) {
        self.equality_confirmation_candidates
            .fetch_add(count as u64, Ordering::Relaxed);
    }

    pub fn equality_confirmation_candidates(&self) -> u64 {
        self.equality_confirmation_candidates
            .load(Ordering::Relaxed)
    }

    pub(crate) fn record_equality_candidate_iterated(&self) {
        self.equality_candidates_iterated
            .fetch_add(1, Ordering::Relaxed);
    }

    pub fn equality_candidates_iterated(&self) -> u64 {
        self.equality_candidates_iterated.load(Ordering::Relaxed)
    }

    pub(crate) fn record_equality_posting_widening(&self) {
        self.equality_posting_widenings
            .fetch_add(1, Ordering::Relaxed);
    }

    pub fn equality_posting_widenings(&self) -> u64 {
        self.equality_posting_widenings.load(Ordering::Relaxed)
    }

    pub(crate) fn record_equality_index_bytes_read(&self, bytes: usize) {
        self.equality_index_bytes_read
            .fetch_add(u64::try_from(bytes).unwrap_or(u64::MAX), Ordering::Relaxed);
    }

    pub fn equality_index_bytes_read(&self) -> u64 {
        self.equality_index_bytes_read.load(Ordering::Relaxed)
    }

    pub fn memtable_population_scans(&self) -> u64 {
        self.memtable_population_scans.load(Ordering::Relaxed)
    }

    pub(crate) fn record_memtable_population_rows(&self, rows: usize) {
        self.memtable_population_rows
            .fetch_add(u64::try_from(rows).unwrap_or(u64::MAX), Ordering::Relaxed);
    }

    pub fn memtable_population_rows(&self) -> u64 {
        self.memtable_population_rows.load(Ordering::Relaxed)
    }

    pub fn memtable_incremental_rows(&self) -> u64 {
        self.memtable_incremental_rows.load(Ordering::Relaxed)
    }

    pub fn memtable_incremental_associations(&self) -> u64 {
        self.memtable_incremental_associations
            .load(Ordering::Relaxed)
    }

    pub(crate) fn node_count_cell_at(&self, generation: u64) -> Option<Arc<ExactNodeCountCell>> {
        self.node_counts
            .read()
            .ok()?
            .as_ref()
            .filter(|(cached, _)| *cached == generation)
            .map(|(_, cell)| Arc::clone(cell))
    }

    pub(crate) fn node_count_at(&self, label: Option<&str>, generation: u64) -> Option<u64> {
        self.node_count_cell_at(generation)?.count(label)
    }

    /// `true` when the total node count is known at `generation`.
    #[cfg(test)]
    pub(crate) fn node_counts_initialized_at(&self, generation: u64) -> bool {
        self.node_count_cell_at(generation)
            .is_some_and(|cell| cell.count(None).is_some())
    }

    /// The properties whose presence a commit at `generation` must report,
    /// or `None` when nothing is counted (no known count and no seed running),
    /// so the commit can skip computing deltas at all.
    pub(crate) fn count_tracking_at(&self, generation: u64) -> Option<BTreeSet<String>> {
        let cell = self
            .node_count_cell_at(generation)
            .filter(|cell| cell.snapshot().is_some());
        let seed = self
            .count_seed
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if cell.is_none() && seed.is_none() {
            return None;
        }
        let mut tracked = cell
            .map(|cell| cell.tracked_properties())
            .unwrap_or_default();
        if let Some(seed) = seed.as_ref() {
            tracked.extend(seed.pairs.iter().map(|(_, property)| property.clone()));
        }
        Some(tracked)
    }

    /// Presence count of the current generation (not a pinned snapshot's).
    pub(crate) fn current_presence(&self, label: &str, property: &str) -> Option<u64> {
        self.node_count_cell_at(self.generation())?
            .presence(label, property)
    }

    /// Label count of the current generation (not a pinned snapshot's).
    pub(crate) fn current_label_count(&self, label: &str) -> Option<u64> {
        self.node_count_cell_at(self.generation())?
            .count(Some(label))
    }

    /// Register a seed of `labels` (and the presence `pairs`, whose labels
    /// must be among them) scanned at `generation`. Returns its id, or `None`
    /// when another seed is running or a commit already moved past
    /// `generation` (the scan would miss that commit's delta).
    pub(crate) fn begin_count_seed(
        &self,
        generation: u64,
        labels: BTreeSet<String>,
        pairs: BTreeSet<(String, String)>,
    ) -> Option<u64> {
        let _transition = self
            .generation_transition
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if self.generation() != generation {
            return None;
        }
        let mut seed = self
            .count_seed
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if seed.is_some() {
            return None;
        }
        let id = self.next_count_seed_id.fetch_add(1, Ordering::Relaxed) + 1;
        *seed = Some(CountSeed {
            id,
            labels,
            pairs,
            delta: CountSeedDelta::default(),
            poisoned: false,
        });
        Some(id)
    }

    /// `true` while a seed is registered.
    pub(crate) fn count_seed_running(&self) -> bool {
        self.count_seed
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .is_some()
    }

    /// Drop seed `id` without installing anything.
    pub(crate) fn abort_count_seed(&self, id: u64) {
        let mut seed = self
            .count_seed
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if seed.as_ref().is_some_and(|seed| seed.id == id) {
            *seed = None;
        }
    }

    /// Roll seed `id`'s scan result forward over the commits that landed
    /// while it ran and install it into the current generation. Returns
    /// `false` (installing nothing) when the seed was poisoned or replaced.
    pub(crate) fn finish_count_seed(&self, id: u64, result: CountSeedResult) -> bool {
        let _transition = self
            .generation_transition
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let seed = {
            let mut guard = self
                .count_seed
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            if !guard.as_ref().is_some_and(|seed| seed.id == id) {
                return false;
            }
            guard.take().expect("checked above")
        };
        if seed.poisoned {
            return false;
        }
        let generation = self.generation();
        let Some(cell) = self.node_count_cell_at(generation) else {
            return false;
        };
        #[cfg(test)]
        if let Some(hook) = BEFORE_SEED_MERGE.with(|hook| hook.borrow_mut().take()) {
            hook();
        }
        // Merge onto what the cell holds at this moment, under its lock: a
        // full reconciliation can install a complete vector while the seed
        // finishes, and a separate read and write would discard it. Labels
        // the cell already knows (all of them, for a complete vector) are
        // checked, never replaced, so a seed cannot downgrade a vector.
        let installed = cell
            .update(|current| {
                let mut counts = current.unwrap_or_default();
                match merge_count_seed(&mut counts, &seed, &result) {
                    SeedMerge::Merged => (Some(Some(counts)), true),
                    SeedMerge::Disagrees => (Some(None), false),
                    SeedMerge::Unusable => (None, false),
                }
            })
            .unwrap_or(false);
        if installed {
            self.count_seeds_installed.fetch_add(1, Ordering::Relaxed);
        }
        installed
    }

    /// Ask the writer to start a seed after its next node commit.
    pub(crate) fn request_count_seed(
        &self,
        labels: BTreeSet<String>,
        pairs: BTreeSet<(String, String)>,
    ) {
        let mut request = self
            .count_seed_request
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        match request.as_mut() {
            Some((pending_labels, pending_pairs)) => {
                pending_labels.extend(labels);
                pending_pairs.extend(pairs);
            }
            None => *request = Some((labels, pairs)),
        }
    }

    pub(crate) fn count_seed_requested(&self) -> bool {
        self.count_seed_request
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .is_some()
    }

    pub(crate) fn take_count_seed_request(&self) -> Option<CountSeedRequest> {
        self.count_seed_request
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .take()
    }

    /// Seeds that installed their counts (diagnostics and tests).
    pub fn count_seeds_installed(&self) -> u64 {
        self.count_seeds_installed.load(Ordering::Relaxed)
    }

    pub(crate) fn insert_node_counts_at(
        &self,
        generation: u64,
        total: u64,
        by_label: HashMap<String, u64>,
    ) {
        if generation != self.generation() {
            return;
        }
        let cell = {
            let Ok(mut guard) = self.node_counts.write() else {
                return;
            };
            if generation != self.generation() {
                return;
            }
            match guard.as_ref() {
                Some((cached, cell)) if *cached == generation => Arc::clone(cell),
                _ => {
                    let cell = Arc::new(ExactNodeCountCell::default());
                    *guard = Some((generation, Arc::clone(&cell)));
                    cell
                }
            }
        };
        cell.install(total, by_label);
    }

    pub(crate) fn record_node_count_reconciliation_scan(&self) {
        self.node_count_reconciliation_scans
            .fetch_add(1, Ordering::Relaxed);
    }

    pub fn node_count_reconciliation_scans(&self) -> u64 {
        self.node_count_reconciliation_scans.load(Ordering::Relaxed)
    }

    pub(crate) fn record_ordered_prefix_call(&self) {
        self.ordered_prefix_calls.fetch_add(1, Ordering::Relaxed);
    }

    pub fn ordered_prefix_calls(&self) -> u64 {
        self.ordered_prefix_calls.load(Ordering::Relaxed)
    }

    pub(crate) fn record_ordered_prefix_widening(&self) {
        self.ordered_prefix_widenings
            .fetch_add(1, Ordering::Relaxed);
    }

    pub fn ordered_prefix_widenings(&self) -> u64 {
        self.ordered_prefix_widenings.load(Ordering::Relaxed)
    }

    pub(crate) fn record_ordered_prefix_candidate_iterated(&self) {
        self.ordered_prefix_candidates_iterated
            .fetch_add(1, Ordering::Relaxed);
    }

    pub fn ordered_prefix_candidates_iterated(&self) -> u64 {
        self.ordered_prefix_candidates_iterated
            .load(Ordering::Relaxed)
    }

    pub(crate) fn record_ordered_prefix_confirmation_candidates(&self, count: usize) {
        self.ordered_prefix_confirmation_candidates
            .fetch_add(u64::try_from(count).unwrap_or(u64::MAX), Ordering::Relaxed);
    }

    pub fn ordered_prefix_confirmation_candidates(&self) -> u64 {
        self.ordered_prefix_confirmation_candidates
            .load(Ordering::Relaxed)
    }

    pub(crate) fn record_ordered_prefix_index_bytes_read(&self, bytes: usize) {
        self.ordered_prefix_index_bytes_read
            .fetch_add(u64::try_from(bytes).unwrap_or(u64::MAX), Ordering::Relaxed);
    }

    pub fn ordered_prefix_index_bytes_read(&self) -> u64 {
        self.ordered_prefix_index_bytes_read.load(Ordering::Relaxed)
    }

    /// Advance the logical generation after a durable node commit while
    /// retaining exact cardinalities when they were already seeded.
    ///
    /// Each tuple is `(old_labels, new_labels)`. `None` denotes absence, so
    /// create/delete update the total; `Some(empty)` is a real unlabeled node.
    /// The caller publishes this only after manifest/WAL durability succeeds.
    pub(crate) fn advance_with_node_changes(
        &self,
        changes: Option<&NodeCountChanges>,
        claimant_delta: Option<&MemtableClaimantDelta>,
    ) {
        let retired = {
            let _transition = self
                .generation_transition
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            let old_generation = self.generation();
            let carried = self.node_counts.read().ok().and_then(|guard| {
                guard
                    .as_ref()
                    .filter(|(generation, _)| *generation == old_generation)
                    .and_then(|(_, cell)| cell.snapshot())
            });
            if let Some(seed) = self
                .count_seed
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .as_mut()
            {
                seed.fold(changes);
            }
            let carried_claimants = claimant_delta
                .map(|delta| {
                    self.memtable_incremental_rows.fetch_add(
                        u64::try_from(delta.rows).unwrap_or(u64::MAX),
                        Ordering::Relaxed,
                    );
                    let associations = delta
                        .changes_by_pair
                        .values()
                        .fold(0usize, |sum, changes| sum.saturating_add(changes.len()));
                    self.memtable_incremental_associations.fetch_add(
                        u64::try_from(associations).unwrap_or(u64::MAX),
                        Ordering::Relaxed,
                    );
                    self.current_memtable_claimant_cell().carry(delta)
                })
                .unwrap_or_else(|| Arc::new(MemtableClaimantCell::default()));
            let new_generation = self.generation.fetch_add(1, Ordering::AcqRel) + 1;
            let retired_indices = std::mem::take(
                &mut *self
                    .indices
                    .write()
                    .unwrap_or_else(std::sync::PoisonError::into_inner),
            );
            let retired_claimants = std::mem::replace(
                &mut *self
                    .memtable_claimants
                    .write()
                    .unwrap_or_else(std::sync::PoisonError::into_inner),
                carried_claimants,
            );
            let retired_global = std::mem::take(
                &mut *self
                    .global_indices
                    .write()
                    .unwrap_or_else(std::sync::PoisonError::into_inner),
            );
            if let Ok(mut w) = self.node_counts.write() {
                let cell = Arc::new(ExactNodeCountCell::default());
                cell.install_snapshot(carried.and_then(|counts| {
                    changes.and_then(|changes| apply_node_count_changes(counts, changes))
                }));
                *w = Some((new_generation, cell));
            }
            (retired_indices, retired_claimants, retired_global)
        };
        // Corpus-sized maps may take noticeable allocator time to destroy.
        // Never do that while a generation transition or cache lock is held.
        drop(retired);
    }

    /// Invalidate corpus-sized reconstructible maps while retaining the
    /// O(labels) exact-count cache. Used for process memory pressure and
    /// metadata-only schema/search-index transitions, neither of which changes
    /// the logical node set.
    ///
    /// Advancing the generation before clearing prevents a pinned old snapshot
    /// from repopulating any evicted map. Exact counts describe logical state,
    /// not a physical representation, so they are moved unchanged to the new
    /// generation. Node commits and schema resets use the same transition lock:
    /// a concurrent pressure pass therefore cannot overwrite a just-applied
    /// count delta with its pre-commit copy.
    pub(crate) fn reset_preserving_node_counts(&self) {
        let retired = {
            let _transition = self
                .generation_transition
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            let old_generation = self.generation();
            let carried = self.node_counts.read().ok().and_then(|guard| {
                guard
                    .as_ref()
                    .filter(|(generation, _)| *generation == old_generation)
                    .map(|(_, cell)| Arc::clone(cell))
            });
            let new_generation = self.generation.fetch_add(1, Ordering::AcqRel) + 1;
            let retired_indices = std::mem::take(
                &mut *self
                    .indices
                    .write()
                    .unwrap_or_else(std::sync::PoisonError::into_inner),
            );
            let retired_claimants = std::mem::replace(
                &mut *self
                    .memtable_claimants
                    .write()
                    .unwrap_or_else(std::sync::PoisonError::into_inner),
                Arc::new(MemtableClaimantCell::default()),
            );
            let retired_global = std::mem::take(
                &mut *self
                    .global_indices
                    .write()
                    .unwrap_or_else(std::sync::PoisonError::into_inner),
            );
            if let Ok(mut counts) = self.node_counts.write() {
                *counts = Some((
                    new_generation,
                    carried.unwrap_or_else(|| Arc::new(ExactNodeCountCell::default())),
                ));
            }
            (retired_indices, retired_claimants, retired_global)
        };
        drop(retired);
    }

    /// Advance the logical node generation and drop every cached index.
    /// Called after committed node mutations and schema/index changes that
    /// can alter property lookup semantics, not representation-only flushes.
    pub fn reset(&self) {
        let retired = {
            let _transition = self
                .generation_transition
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            if let Some(seed) = self
                .count_seed
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .as_mut()
            {
                // The node set changed by an unknown amount (attached SSTs).
                seed.poisoned = true;
            }
            let new_generation = self.generation.fetch_add(1, Ordering::AcqRel) + 1;
            let retired_indices = std::mem::take(
                &mut *self
                    .indices
                    .write()
                    .unwrap_or_else(std::sync::PoisonError::into_inner),
            );
            let retired_claimants = std::mem::replace(
                &mut *self
                    .memtable_claimants
                    .write()
                    .unwrap_or_else(std::sync::PoisonError::into_inner),
                Arc::new(MemtableClaimantCell::default()),
            );
            let retired_global = std::mem::take(
                &mut *self
                    .global_indices
                    .write()
                    .unwrap_or_else(std::sync::PoisonError::into_inner),
            );
            if let Ok(mut w) = self.node_counts.write() {
                *w = Some((new_generation, Arc::new(ExactNodeCountCell::default())));
            }
            (retired_indices, retired_claimants, retired_global)
        };
        drop(retired);
    }

    /// Drop the claimant map after a successful memtable flush without
    /// perturbing the logical node generation or exact-count cell. Snapshots
    /// published before the flush retain the old cell; snapshots published
    /// after it start from an empty memtable and immutable sidecars.
    pub(crate) fn reset_memtable_claimants(&self) {
        let retired = {
            let _transition = self
                .generation_transition
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            std::mem::replace(
                &mut *self
                    .memtable_claimants
                    .write()
                    .unwrap_or_else(std::sync::PoisonError::into_inner),
                Arc::new(MemtableClaimantCell::default()),
            )
        };
        drop(retired);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use uuid::Uuid;

    fn node(byte: u8) -> NodeId {
        NodeId::from_uuid(Uuid::from_bytes([byte; 16]))
    }

    fn claimant_index(value: &str, ids: &[NodeId]) -> Arc<MemtableClaimantIndex> {
        Arc::new(PersistentHashMap::from_iter([(
            value.to_string(),
            OrdSet::from_iter(ids.iter().copied()),
        )]))
    }

    #[test]
    fn stale_snapshot_cannot_repopulate_after_reset() {
        let cache = PropertyIndexCache::new();
        let old_generation = cache.generation();
        let old_claimants = cache.current_memtable_claimant_cell();
        cache.insert_at(
            "Doc".into(),
            "key".into(),
            Arc::new(HashMap::from([("old".into(), node(1))])),
            old_generation,
        );
        cache.insert_memtable_claimants(
            &old_claimants,
            "Doc".into(),
            "key".into(),
            claimant_index("old", &[node(1)]),
        );
        cache.insert_global_at(
            "key".into(),
            Arc::new(HashMap::from([("old".into(), vec![node(1)])])),
            old_generation,
        );
        assert!(cache.get_at("Doc", "key", old_generation).is_some());
        assert!(cache
            .get_memtable_claimants(&old_claimants, "Doc", "key")
            .is_some());
        assert!(cache.get_global_at("key", old_generation).is_some());
        assert!(cache.indices.read().unwrap().capacity() > 0);
        assert!(!old_claimants.is_empty());
        assert!(cache.global_indices.read().unwrap().capacity() > 0);

        cache.reset();
        let current_generation = cache.generation();
        let current_claimants = cache.current_memtable_claimant_cell();
        assert_ne!(current_generation, old_generation);
        assert_eq!(
            cache.indices.read().unwrap().capacity(),
            0,
            "reset must release the unique-property map buckets"
        );
        assert!(current_claimants.is_empty());
        assert_eq!(cache.global_indices.read().unwrap().capacity(), 0);

        // Simulate an old pinned reader finishing its expensive scan after a
        // newer node commit already reset the shared cache.
        cache.insert_at(
            "Doc".into(),
            "key".into(),
            Arc::new(HashMap::from([("stale".into(), node(2))])),
            old_generation,
        );
        cache.insert_memtable_claimants(
            &old_claimants,
            "Doc".into(),
            "key".into(),
            claimant_index("stale", &[node(2)]),
        );
        cache.insert_global_at(
            "key".into(),
            Arc::new(HashMap::from([("stale".into(), vec![node(2)])])),
            old_generation,
        );
        assert!(
            cache.get_at("Doc", "key", current_generation).is_none(),
            "the new snapshot must never observe an old reader's index"
        );
        assert!(cache
            .get_memtable_claimants(&current_claimants, "Doc", "key")
            .is_none());
        assert!(cache.get_global_at("key", current_generation).is_none());

        cache.insert_at(
            "Doc".into(),
            "key".into(),
            Arc::new(HashMap::from([("fresh".into(), node(3))])),
            current_generation,
        );
        cache.insert_memtable_claimants(
            &current_claimants,
            "Doc".into(),
            "key".into(),
            claimant_index("fresh", &[node(3)]),
        );
        cache.insert_global_at(
            "key".into(),
            Arc::new(HashMap::from([("fresh".into(), vec![node(3)])])),
            current_generation,
        );
        assert_eq!(
            cache
                .get_at("Doc", "key", current_generation)
                .unwrap()
                .get("fresh"),
            Some(&node(3))
        );
        assert_eq!(
            cache
                .get_memtable_claimants(&current_claimants, "Doc", "key")
                .unwrap()
                .get("fresh"),
            Some(&OrdSet::from_iter([node(3)]))
        );
        assert_eq!(
            cache
                .get_global_at("key", current_generation)
                .unwrap()
                .get("fresh"),
            Some(&vec![node(3)])
        );
    }

    #[test]
    fn impossible_exact_count_delta_discards_metadata_instead_of_saturating() {
        let underflow = PropertyIndexCache::new();
        underflow.insert_node_counts_at(underflow.generation(), 0, HashMap::new());
        underflow
            .advance_with_node_changes(Some(&changes(vec![(state(&["Doc"], &[]), None)])), None);
        assert_eq!(
            underflow.node_count_at(None, underflow.generation()),
            None,
            "an impossible delete must force exact reconciliation"
        );

        let overflow = PropertyIndexCache::new();
        overflow.insert_node_counts_at(
            overflow.generation(),
            u64::MAX,
            HashMap::from([("Doc".to_string(), u64::MAX)]),
        );
        overflow
            .advance_with_node_changes(Some(&changes(vec![(None, state(&["Other"], &[]))])), None);
        assert_eq!(
            overflow.node_count_at(None, overflow.generation()),
            None,
            "an overflowing create must never publish a wrapped/saturated count"
        );
    }

    fn state(labels: &[&str], present: &[&str]) -> Option<NodeCountState> {
        Some(NodeCountState {
            labels: labels.iter().map(|l| (*l).to_string()).collect(),
            present: present.iter().map(|p| (*p).to_string()).collect(),
        })
    }

    fn changes(changes: Vec<(Option<NodeCountState>, Option<NodeCountState>)>) -> NodeCountChanges {
        NodeCountChanges {
            tracked: ["embedding".to_string()].into_iter().collect(),
            changes,
        }
    }

    fn pair(label: &str, property: &str) -> (String, String) {
        (label.to_string(), property.to_string())
    }

    #[test]
    fn a_seed_keeps_a_full_vector_installed_while_it_finishes() {
        // A query's full reconciliation installs the complete vector right
        // as the label seed it started finishes. The seed used to read the
        // cell, then write it back with its label added: a vector installed
        // in between was replaced by the seed's partial counts, and the next
        // count of another label (or of all nodes) scanned the corpus again.
        let cache = Arc::new(PropertyIndexCache::new());
        let start = cache.generation();
        let id = cache
            .begin_count_seed(
                start,
                ["Doc".to_string()].into_iter().collect(),
                BTreeSet::new(),
            )
            .expect("no commit moved the generation");
        let reconciled = Arc::new(std::sync::atomic::AtomicBool::new(false));
        {
            let cache = Arc::clone(&cache);
            let reconciled = Arc::clone(&reconciled);
            BEFORE_SEED_MERGE.with(|hook| {
                *hook.borrow_mut() = Some(Box::new(move || {
                    cache.insert_node_counts_at(
                        start,
                        12,
                        HashMap::from([("Doc".to_string(), 10), ("Other".to_string(), 2)]),
                    );
                    reconciled.store(true, Ordering::Relaxed);
                }));
            });
        }
        assert!(cache.finish_count_seed(
            id,
            CountSeedResult {
                by_label: HashMap::from([("Doc".to_string(), 10)]),
                presence: HashMap::new(),
            }
        ));
        assert!(reconciled.load(Ordering::Relaxed), "the hook ran");
        // The full vector survives the seed, complete.
        assert_eq!(cache.node_count_at(None, start), Some(12));
        assert_eq!(cache.node_count_at(Some("Doc"), start), Some(10));
        assert_eq!(cache.node_count_at(Some("Other"), start), Some(2));
        assert_eq!(cache.node_count_at(Some("Absent"), start), Some(0));

        // A seed that disagrees with the installed vector still clears it:
        // one of the two is wrong.
        let id = cache
            .begin_count_seed(
                start,
                ["Other".to_string()].into_iter().collect(),
                BTreeSet::new(),
            )
            .expect("no commit moved the generation");
        assert!(!cache.finish_count_seed(
            id,
            CountSeedResult {
                by_label: HashMap::from([("Other".to_string(), 3)]),
                presence: HashMap::new(),
            }
        ));
        assert_eq!(cache.node_count_at(None, start), None);
        assert_eq!(cache.node_count_at(Some("Other"), start), None);
    }

    #[test]
    fn a_seed_rolls_forward_over_commits_that_land_while_it_scans() {
        let cache = PropertyIndexCache::new();
        let start = cache.generation();
        let id = cache
            .begin_count_seed(
                start,
                ["Doc".to_string()].into_iter().collect(),
                [pair("Doc", "embedding")].into_iter().collect(),
            )
            .expect("no commit moved the generation");
        assert_eq!(
            cache.count_tracking_at(start),
            Some(["embedding".to_string()].into_iter().collect()),
            "a running seed makes commits report presence"
        );
        // While the scan runs: one new embedded Doc, one Doc loses its
        // embedding, one Doc is deleted, one unrelated node is created.
        cache.advance_with_node_changes(
            Some(&changes(vec![
                (None, state(&["Doc"], &["embedding"])),
                (state(&["Doc"], &["embedding"]), state(&["Doc"], &[])),
                (state(&["Doc"], &[]), None),
                (None, state(&["Other"], &["embedding"])),
            ])),
            None,
        );
        // The scan saw 10 Docs, 7 of them embedded, at `start`.
        assert!(cache.finish_count_seed(
            id,
            CountSeedResult {
                by_label: HashMap::from([("Doc".to_string(), 10)]),
                presence: HashMap::from([(pair("Doc", "embedding"), 7)]),
            }
        ));
        let now = cache.generation();
        assert_eq!(cache.node_count_at(Some("Doc"), now), Some(10));
        assert_eq!(cache.current_presence("Doc", "embedding"), Some(7));
        assert_eq!(
            cache.node_count_at(None, now),
            None,
            "the total stays unknown"
        );
        assert_eq!(cache.node_count_at(Some("Other"), now), None);

        // Later commits keep the seeded pair exact.
        cache.advance_with_node_changes(
            Some(&changes(vec![
                (state(&["Doc"], &[]), state(&["Doc"], &["embedding"])),
                (
                    state(&["Doc"], &["embedding"]),
                    state(&["Archived"], &["embedding"]),
                ),
            ])),
            None,
        );
        assert_eq!(cache.current_presence("Doc", "embedding"), Some(7));
        assert_eq!(cache.current_label_count("Doc"), Some(9));
    }

    #[test]
    fn a_seed_is_discarded_when_a_commit_delta_is_unknown_or_the_generation_moved() {
        let cache = PropertyIndexCache::new();
        let start = cache.generation();
        let labels: BTreeSet<String> = ["Doc".to_string()].into_iter().collect();
        let pairs: BTreeSet<(String, String)> = [pair("Doc", "embedding")].into_iter().collect();
        let id = cache
            .begin_count_seed(start, labels.clone(), pairs.clone())
            .unwrap();
        assert!(
            cache
                .begin_count_seed(start, labels.clone(), pairs.clone())
                .is_none(),
            "one seed at a time"
        );
        cache.advance_with_node_changes(None, None);
        assert!(!cache.finish_count_seed(
            id,
            CountSeedResult {
                by_label: HashMap::from([("Doc".to_string(), 1)]),
                presence: HashMap::from([(pair("Doc", "embedding"), 1)]),
            }
        ));
        assert_eq!(cache.current_presence("Doc", "embedding"), None);
        assert!(!cache.count_seed_running());
        assert!(
            cache
                .begin_count_seed(start, labels.clone(), pairs.clone())
                .is_none(),
            "a snapshot older than the current generation cannot seed"
        );

        // A commit prepared before the seed began (so without its property)
        // poisons it instead of being folded as "no presence change".
        let id = cache
            .begin_count_seed(
                cache.generation(),
                labels,
                [pair("Doc", "title")].into_iter().collect(),
            )
            .unwrap();
        cache.advance_with_node_changes(Some(&changes(vec![(None, state(&["Doc"], &[]))])), None);
        assert!(!cache.finish_count_seed(
            id,
            CountSeedResult {
                by_label: HashMap::from([("Doc".to_string(), 1)]),
                presence: HashMap::from([(pair("Doc", "title"), 1)]),
            }
        ));
    }

    #[test]
    fn presence_is_forgotten_when_a_commit_did_not_compute_its_property() {
        let cache = PropertyIndexCache::new();
        let id = cache
            .begin_count_seed(
                cache.generation(),
                ["Doc".to_string()].into_iter().collect(),
                [pair("Doc", "embedding")].into_iter().collect(),
            )
            .unwrap();
        assert!(cache.finish_count_seed(
            id,
            CountSeedResult {
                by_label: HashMap::from([("Doc".to_string(), 2)]),
                presence: HashMap::from([(pair("Doc", "embedding"), 1)]),
            }
        ));
        cache.advance_with_node_changes(
            Some(&NodeCountChanges {
                tracked: BTreeSet::new(),
                changes: vec![(None, state(&["Doc"], &[]))],
            }),
            None,
        );
        assert_eq!(cache.current_presence("Doc", "embedding"), None);
        assert_eq!(cache.current_label_count("Doc"), Some(3));
    }

    #[test]
    fn a_full_install_keeps_agreeing_presence_and_seeds_survive_pressure_resets() {
        let cache = PropertyIndexCache::new();
        let id = cache
            .begin_count_seed(
                cache.generation(),
                ["Doc".to_string()].into_iter().collect(),
                [pair("Doc", "embedding")].into_iter().collect(),
            )
            .unwrap();
        cache.reset_preserving_node_counts();
        assert!(cache.finish_count_seed(
            id,
            CountSeedResult {
                by_label: HashMap::from([("Doc".to_string(), 2)]),
                presence: HashMap::from([(pair("Doc", "embedding"), 2)]),
            }
        ));
        cache.insert_node_counts_at(
            cache.generation(),
            5,
            HashMap::from([("Doc".to_string(), 2), ("Other".to_string(), 3)]),
        );
        assert_eq!(cache.current_presence("Doc", "embedding"), Some(2));
        assert_eq!(cache.node_count_at(None, cache.generation()), Some(5));

        // An attach (unknown node delta) poisons a running seed.
        let id = cache
            .begin_count_seed(
                cache.generation(),
                ["Other".to_string()].into_iter().collect(),
                BTreeSet::new(),
            )
            .unwrap();
        cache.reset();
        assert!(!cache.finish_count_seed(
            id,
            CountSeedResult {
                by_label: HashMap::from([("Other".to_string(), 3)]),
                presence: HashMap::new(),
            }
        ));
    }

    #[test]
    fn claimant_carry_handles_create_delete_and_relabel_without_touching_irrelevant_pairs() {
        let cache = PropertyIndexCache::new();
        let cell = cache.current_memtable_claimant_cell();
        let created = node(3);
        let deleted = node(1);
        let relabelled = node(2);

        let a_key = claimant_index("old-a", &[deleted]);
        let mut a_key_values = a_key.as_ref().clone();
        a_key_values.insert("move".into(), OrdSet::from_iter([relabelled]));
        let a_key = Arc::new(a_key_values);
        let b_key = Arc::new(MemtableClaimantIndex::new());
        let global_key = Arc::clone(&a_key);
        let irrelevant = claimant_index("untouched", &[node(9)]);
        for (label, property, index) in [
            ("A", "key", Arc::clone(&a_key)),
            ("B", "key", Arc::clone(&b_key)),
            ("", "key", global_key),
            ("Other", "other", Arc::clone(&irrelevant)),
        ] {
            cache.insert_memtable_claimants(&cell, label.into(), property.into(), index);
        }

        let captured_pairs = BTreeSet::from([
            ("A".into(), "key".into()),
            ("B".into(), "key".into()),
            ("".into(), "key".into()),
            ("Other".into(), "other".into()),
        ]);
        let delta = MemtableClaimantDelta {
            changes_by_pair: HashMap::from([
                (
                    ("A".into(), "key".into()),
                    vec![
                        (deleted, Some("old-a".into()), None),
                        (created, None, Some("new-a".into())),
                        (relabelled, Some("move".into()), None),
                    ],
                ),
                (
                    ("B".into(), "key".into()),
                    vec![(relabelled, None, Some("move".into()))],
                ),
                (
                    ("".into(), "key".into()),
                    vec![
                        (deleted, Some("old-a".into()), None),
                        (created, None, Some("new-a".into())),
                    ],
                ),
            ]),
            captured_pairs,
            rows: 3,
            tail_changes: HashMap::new(),
            text_changes: HashMap::new(),
        };
        cache.advance_with_node_changes(None, Some(&delta));

        let current = cache.current_memtable_claimant_cell();
        let a = cache.get_memtable_claimants(&current, "A", "key").unwrap();
        assert!(a.get("old-a").is_none());
        assert!(a.get("move").is_none());
        assert_eq!(a.get("new-a"), Some(&OrdSet::from_iter([created])));

        let b = cache.get_memtable_claimants(&current, "B", "key").unwrap();
        assert_eq!(b.get("move"), Some(&OrdSet::from_iter([relabelled])));

        let global = cache.get_memtable_claimants(&current, "", "key").unwrap();
        assert!(global.get("old-a").is_none());
        assert_eq!(global.get("move"), Some(&OrdSet::from_iter([relabelled])));
        assert_eq!(global.get("new-a"), Some(&OrdSet::from_iter([created])));

        let carried_irrelevant = cache
            .get_memtable_claimants(&current, "Other", "other")
            .unwrap();
        assert!(
            Arc::ptr_eq(&irrelevant, &carried_irrelevant),
            "a captured pair with no relevant changes must retain its map O(1)"
        );
        assert_eq!(cache.memtable_incremental_rows(), 3);
        assert_eq!(cache.memtable_incremental_associations(), 6);
    }
}
