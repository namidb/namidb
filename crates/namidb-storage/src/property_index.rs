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
//! - Invalidation: the cache is tied to a `WriterSession`. The session
//!   itself bumps its manifest version on every flush, so any caller
//!   that reuses the cache across flushes must `reset()` it; the
//!   bench harness opens a fresh writer per benchmark, which sidesteps
//!   the concern entirely.

use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use namidb_core::id::NodeId;

/// `(label, property)` keys mapped to a shared per-pair `value -> NodeId`
/// index. Aliased so the `RwLock` field below stays under clippy's
/// type-complexity threshold.
type PropertyIndices = HashMap<(String, String), Arc<HashMap<String, NodeId>>>;

/// Shared cache that lives at `WriterSession` scope and is cloned (as
/// an `Arc`) into every `Snapshot` the session emits.
#[derive(Debug, Default)]
pub struct PropertyIndexCache {
    /// `(label_name, property_name) → Arc<value_string → NodeId>`.
    /// `Arc` on the inner so readers can release the outer lock as
    /// soon as they have the per-(label, prop) handle.
    indices: RwLock<PropertyIndices>,
}

impl PropertyIndexCache {
    pub fn new() -> Self {
        Self::default()
    }

    /// Probe-only: returns `Some(handle)` when the (label, property)
    /// index has already been built, `None` otherwise. Used by the
    /// `lookup_node_by_property` hot path to short-circuit before
    /// taking the write lock + scanning.
    pub fn get(&self, label: &str, property: &str) -> Option<Arc<HashMap<String, NodeId>>> {
        self.indices
            .read()
            .ok()?
            .get(&(label.to_string(), property.to_string()))
            .cloned()
    }

    /// Insert a pre-built index. Idempotent — last write wins under a
    /// race; the contents are identical by construction so this is safe.
    pub fn insert(&self, label: String, property: String, index: Arc<HashMap<String, NodeId>>) {
        if let Ok(mut w) = self.indices.write() {
            w.insert((label, property), index);
        }
    }

    /// Drop every cached index. Called after a flush that changes the
    /// manifest version (any cached `Arc<HashMap>` still points at
    /// post-flush data only as long as nothing changed; on flush we
    /// invalidate to be safe).
    pub fn reset(&self) {
        if let Ok(mut w) = self.indices.write() {
            w.clear();
        }
    }
}
