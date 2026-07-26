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
type NodeLabelChange = (Option<BTreeSet<String>>, Option<BTreeSet<String>>);

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
}

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
        if source.is_empty() {
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
        Arc::new(Self {
            indices: RwLock::new(carried),
        })
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
#[derive(Clone, Debug, Default)]
struct ExactNodeCounts {
    total: u64,
    by_label: HashMap<String, u64>,
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
        Some(match label {
            None => counts.total,
            Some(label) => counts.by_label.get(label).copied().unwrap_or(0),
        })
    }

    fn snapshot(&self) -> Option<ExactNodeCounts> {
        self.counts
            .read()
            .ok()
            .and_then(|guard| guard.as_ref().map(|counts| counts.as_ref().clone()))
    }

    pub(crate) fn install(&self, total: u64, by_label: HashMap<String, u64>) {
        if let Ok(mut guard) = self.counts.write() {
            *guard = if by_label.values().any(|count| *count == 0 || *count > total) {
                None
            } else {
                Some(Arc::new(ExactNodeCounts { total, by_label }))
            };
        }
    }

    fn install_snapshot(&self, counts: Option<ExactNodeCounts>) {
        if let Ok(mut guard) = self.counts.write() {
            *guard = counts.map(Arc::new);
        }
    }
}

/// Apply a committed node batch to one exact cardinality vector.
///
/// Count metadata is an optimisation, so an impossible delta must discard it
/// and force the exact read-path reconciliation. Saturating or wrapping here
/// would be worse than a scan: it could publish a plausible but stale Cypher
/// `count(*)`. The writer guarantees one final change per node, but this
/// defensive validation also protects migrations from corrupt/legacy seed
/// metadata.
fn apply_node_label_changes(
    mut counts: ExactNodeCounts,
    changes: &[NodeLabelChange],
) -> Option<ExactNodeCounts> {
    for (old_labels, new_labels) in changes {
        match (old_labels.is_some(), new_labels.is_some()) {
            (false, true) => counts.total = counts.total.checked_add(1)?,
            (true, false) => counts.total = counts.total.checked_sub(1)?,
            _ => {}
        }

        if let Some(old_labels) = old_labels {
            for label in old_labels {
                if new_labels
                    .as_ref()
                    .is_some_and(|labels| labels.contains(label))
                {
                    continue;
                }
                let remove = {
                    let count = counts.by_label.get_mut(label)?;
                    *count = count.checked_sub(1)?;
                    *count == 0
                };
                if remove {
                    counts.by_label.remove(label);
                }
            }
        }
        if let Some(new_labels) = new_labels {
            for label in new_labels {
                if old_labels
                    .as_ref()
                    .is_some_and(|labels| labels.contains(label))
                {
                    continue;
                }
                let count = counts.by_label.entry(label.clone()).or_insert(0);
                *count = count.checked_add(1)?;
            }
        }
    }

    if counts.by_label.values().any(|count| *count > counts.total) {
        return None;
    }
    Some(counts)
}

/// Shared cache that lives at `WriterSession` scope and is cloned (as
/// an `Arc`) into every `Snapshot` the session emits.
#[derive(Debug, Default)]
pub struct PropertyIndexCache {
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

    pub(crate) fn node_counts_initialized_at(&self, generation: u64) -> bool {
        self.node_count_cell_at(generation)
            .is_some_and(|cell| cell.count(None).is_some())
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
        changes: Option<&[NodeLabelChange]>,
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
                    changes.and_then(|changes| apply_node_label_changes(counts, changes))
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
        let labels = |names: &[&str]| {
            Some(
                names
                    .iter()
                    .map(|name| (*name).to_string())
                    .collect::<BTreeSet<_>>(),
            )
        };

        let underflow = PropertyIndexCache::new();
        underflow.insert_node_counts_at(underflow.generation(), 0, HashMap::new());
        underflow.advance_with_node_changes(Some(&[(labels(&["Doc"]), None)]), None);
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
        overflow.advance_with_node_changes(Some(&[(None, labels(&["Other"]))]), None);
        assert_eq!(
            overflow.node_count_at(None, overflow.generation()),
            None,
            "an overflowing create must never publish a wrapped/saturated count"
        );
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
