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

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, RwLock};

use namidb_core::id::NodeId;

/// `(label, property)` keys mapped to a shared per-pair `value -> NodeId`
/// index. Aliased so the `RwLock` field below stays under clippy's
/// type-complexity threshold.
type PropertyIndices = HashMap<(u64, String, String), Arc<HashMap<String, NodeId>>>;
type MemtableClaimantIndices = HashMap<(u64, String, String), Arc<HashMap<String, Vec<NodeId>>>>;
type GlobalPropertyIndices = HashMap<(u64, String), Arc<HashMap<String, Vec<NodeId>>>>;

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
    /// O(keys × memtable_nodes). The writer invalidates this tier only when a
    /// committed batch contains node mutations.
    memtable_claimants: RwLock<MemtableClaimantIndices>,
    /// Complete label-agnostic fallback indexes. These are populated only
    /// when an older SST lacks the global equality sidecar; one all-node scan
    /// amortises every correlated `(n {key})` lookup that follows.
    global_indices: RwLock<GlobalPropertyIndices>,
    /// Logical node-data generation. Every snapshot captures this value when
    /// it is built and may only read/insert entries under that generation.
    /// This prevents an old pinned reader from repopulating the shared cache
    /// after a newer node commit invalidated it.
    generation: AtomicU64,
    /// Calls routed through the unique point-lookup storage API.
    unique_lookup_calls: AtomicU64,
    /// Calls routed through the non-unique equality posting-list API.
    equality_lookup_calls: AtomicU64,
    /// Number of full committed-memtable scans used to populate claimant
    /// indexes. Exposed for deterministic performance regressions.
    memtable_population_scans: AtomicU64,
}

impl PropertyIndexCache {
    pub fn new() -> Self {
        Self::default()
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

    pub(crate) fn get_memtable_claimants_at(
        &self,
        label: &str,
        property: &str,
        generation: u64,
    ) -> Option<Arc<HashMap<String, Vec<NodeId>>>> {
        self.memtable_claimants
            .read()
            .ok()?
            .get(&(generation, label.to_string(), property.to_string()))
            .cloned()
    }

    pub(crate) fn insert_memtable_claimants_at(
        &self,
        label: String,
        property: String,
        index: Arc<HashMap<String, Vec<NodeId>>>,
        generation: u64,
    ) {
        if generation != self.generation() {
            return;
        }
        if let Ok(mut w) = self.memtable_claimants.write() {
            if generation == self.generation() {
                w.insert((generation, label, property), index);
                self.memtable_population_scans
                    .fetch_add(1, Ordering::Relaxed);
            }
        }
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

    pub fn memtable_population_scans(&self) -> u64 {
        self.memtable_population_scans.load(Ordering::Relaxed)
    }

    /// Advance the logical node generation and drop every cached index.
    /// Called after committed node mutations and schema/index changes that
    /// can alter property lookup semantics, not representation-only flushes.
    pub fn reset(&self) {
        self.generation.fetch_add(1, Ordering::AcqRel);
        if let Ok(mut w) = self.indices.write() {
            w.clear();
        }
        if let Ok(mut w) = self.memtable_claimants.write() {
            w.clear();
        }
        if let Ok(mut w) = self.global_indices.write() {
            w.clear();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use uuid::Uuid;

    fn node(byte: u8) -> NodeId {
        NodeId::from_uuid(Uuid::from_bytes([byte; 16]))
    }

    #[test]
    fn stale_snapshot_cannot_repopulate_after_reset() {
        let cache = PropertyIndexCache::new();
        let old_generation = cache.generation();
        cache.insert_at(
            "Doc".into(),
            "key".into(),
            Arc::new(HashMap::from([("old".into(), node(1))])),
            old_generation,
        );
        cache.insert_memtable_claimants_at(
            "Doc".into(),
            "key".into(),
            Arc::new(HashMap::from([("old".into(), vec![node(1)])])),
            old_generation,
        );
        cache.insert_global_at(
            "key".into(),
            Arc::new(HashMap::from([("old".into(), vec![node(1)])])),
            old_generation,
        );
        assert!(cache.get_at("Doc", "key", old_generation).is_some());
        assert!(cache
            .get_memtable_claimants_at("Doc", "key", old_generation)
            .is_some());
        assert!(cache.get_global_at("key", old_generation).is_some());

        cache.reset();
        let current_generation = cache.generation();
        assert_ne!(current_generation, old_generation);

        // Simulate an old pinned reader finishing its expensive scan after a
        // newer node commit already reset the shared cache.
        cache.insert_at(
            "Doc".into(),
            "key".into(),
            Arc::new(HashMap::from([("stale".into(), node(2))])),
            old_generation,
        );
        cache.insert_memtable_claimants_at(
            "Doc".into(),
            "key".into(),
            Arc::new(HashMap::from([("stale".into(), vec![node(2)])])),
            old_generation,
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
            .get_memtable_claimants_at("Doc", "key", current_generation)
            .is_none());
        assert!(cache.get_global_at("key", current_generation).is_none());

        cache.insert_at(
            "Doc".into(),
            "key".into(),
            Arc::new(HashMap::from([("fresh".into(), node(3))])),
            current_generation,
        );
        cache.insert_memtable_claimants_at(
            "Doc".into(),
            "key".into(),
            Arc::new(HashMap::from([("fresh".into(), vec![node(3)])])),
            current_generation,
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
                .get_memtable_claimants_at("Doc", "key", current_generation)
                .unwrap()
                .get("fresh"),
            Some(&vec![node(3)])
        );
        assert_eq!(
            cache
                .get_global_at("key", current_generation)
                .unwrap()
                .get("fresh"),
            Some(&vec![node(3)])
        );
    }
}
