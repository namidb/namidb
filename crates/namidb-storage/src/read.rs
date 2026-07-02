//! Snapshot-isolated read path.
//!
//! A [`Snapshot`] pins a [`LoadedManifest`] and borrows a [`Memtable`]
//! for the lifetime of one or more reads. Even if the underlying
//! namespace advances (a competing writer commits a new manifest, or a
//! flush moves memtable rows into SSTs), the snapshot continues to see
//! the state as of its pin — `Snapshot::lookup_node` etc. never
//! re-query the manifest store after construction.
//!
//! ## Last-write-wins
//!
//! For each `(label, node_id)` or `(edge_type, src, dst)`, the row with
//! the highest LSN observed across memtable + SSTs wins. A
//! `MemOp::Tombstone` or a Parquet row with `tombstone=true` at the
//! winning LSN produces an absent result.
//!
//! ## Pruning
//!
//! For each SST candidate the snapshot uses:
//! - `kind` and `scope` match (label or edge_type).
//! - `min_key <= target <= max_key` (zero-cost via embedded stats).
//! - **Bloom side-car probe** when the SST carries one (RFC-002 §4.2 —
//! small SSTs omit the side-car entirely). The probe loads the bloom
//! body, verifies its xxhash and tests membership; a negative answer
//! short-circuits the costly body GET without sacrificing correctness.
//!
//! ## What's not here yet (deliberate follow-ups)
//!
//! - Streaming range scans: `scan_label` / `scan_edge_type` exist but
//! buffer the merged result in RAM. The query layer will gain
//! `Stream<Item = Result<NodeView>>` once the executor needs to pipeline.
//! - Concurrent SST GETs: candidates are walked sequentially. Same
//! tradeoff documented in the bug audit — the flush side is
//! already parallelised; read-side concurrency lands with the
//! buffer-pool task to avoid double-blast under cache misses.
//! - `foyer-rs` cache: every GET hits the object store. Cache
//! integration lands with the buffer-pool task — at which point the
//! bloom side-car becomes cache-friendly (constant per SST, tiny).
//! - Declared edge property streams: the read path now decodes the
//! `__overflow_json` stream that the writer emits, so `EdgeView`s
//! coming from SSTs carry their property maps. Splitting properties
//! into per-name streams (RFC-002 §3.2.7) is still a follow-up —
//! relevant for selective predicate push-down rather than
//! correctness.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::sync::{Arc, Mutex};

use arrow_array::RecordBatch;
use arrow_array::{
    Array, BinaryArray, BooleanArray, Date32Array, FixedSizeBinaryArray, FixedSizeListArray,
    Float32Array, Float64Array, Int32Array, Int64Array, LargeStringArray, ListArray, StringArray,
    TimestampMicrosecondArray, UInt32Array, UInt64Array,
};
use bytes::Bytes;
use object_store::path::Path;
use object_store::{ObjectStore, ObjectStoreExt};
use parquet::file::metadata::ParquetMetaData;
use tracing::instrument;
use uuid::Uuid;

use namidb_core::{DataType, LabelDef, LabelDictionary, LabelId, NodeId, Value};

use crate::adjacency::{
    adjacency_enabled, build_adjacency, AdjacencyCache, AdjacencyKey, EdgeAdjacency,
};
use crate::cache::{DecodedNodeRowGroup, EdgeStreamBundle, NodeRowGroupKey, SstCache};
use crate::error::{Error, Result};
use crate::flush::{EdgeWriteRecord, NodeWriteRecord};
use crate::manifest::{LoadedManifest, Manifest, SstDescriptor, SstKind};
use crate::memtable::{MemEntry, MemKey, MemOp, MemtableSnapshot};
use crate::node_cache::{NodeCacheKey, NodeViewCache};
use crate::paths::NamespacePaths;
use crate::sst::bloom::BloomFilter;
use crate::sst::edges::reader::EdgeSstReader;
use crate::sst::edges::EdgeDirection;
use crate::sst::nodes::{
    load_node_sst_metadata_async, parse_node_sst_metadata, prop_column_name, row_groups_for_keys,
    scan_row_groups_async as node_scan_row_groups_async, split_batches_by_row_group,
    targeted_scan_async as node_targeted_scan_async, NodeSstReader, COL_LABELS, COL_LSN,
    COL_NODE_ID, COL_TOMBSTONE, OVERFLOW_JSON, SCHEMA_VERSION,
};
use crate::sst::predicates::{eval_against_value, ScanPredicate};

/// Projection of a node row materialised by the read path.
///
/// A node carries a *set* of labels. Today the set always has exactly one
/// member (the SST scope it was read from); multi-label nodes will populate it
/// from the on-row label column in a later step. Storing a set now lets the
/// query layer match `(n:A:B)` as set-membership without another type flip.
#[derive(Debug, Clone, PartialEq)]
pub struct NodeView {
    pub id: NodeId,
    pub labels: BTreeSet<String>,
    pub properties: BTreeMap<String, Value>,
    pub lsn: u64,
    pub schema_version: u64,
}

/// Projection of an edge row materialised by the read path.
#[derive(Debug, Clone, PartialEq)]
pub struct EdgeView {
    pub edge_type: String,
    pub src: NodeId,
    pub dst: NodeId,
    pub properties: BTreeMap<String, Value>,
    pub lsn: u64,
}

/// Collection of edges incident to a single key (src for forward, dst
/// for inverse). Sorted by the partner identifier for stable iteration.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct EdgeListView {
    pub edges: Vec<EdgeView>,
}

/// Endpoint labels for an edge type, surfaced by
/// [`Snapshot::observed_edge_endpoints`]. For edge types that were
/// declared through `SchemaBuilder` the labels come straight from the
/// manifest and `inferred` is `false`. For edge types that only exist
/// because some `CREATE` ran without a prior declaration, the labels
/// are derived from a sample of the actual edges in the snapshot and
/// `inferred` is `true`. Either label can still be `None` if no
/// matching sample edge could be resolved (a tombstoned-only edge type
/// or a corrupt state — should not happen in practice).
#[derive(Debug, Clone, PartialEq)]
pub struct EdgeEndpoint {
    pub edge_type: String,
    pub src_label: Option<String>,
    pub dst_label: Option<String>,
    pub inferred: bool,
}

/// Pinned read view of a namespace.
pub struct Snapshot<'mt> {
    manifest: LoadedManifest,
    memtable: &'mt MemtableSnapshot,
    store: Arc<dyn ObjectStore>,
    paths: NamespacePaths,
    cache: Option<SstCache>,
    /// Per-snapshot NodeView cache. Many queries access the same node
    /// from multiple sides (e.g., Join probe + reverse Expand, or the
    /// same friend reached through several paths in IC09). Caching the
    /// post-decode `Option<NodeView>` skips bloom probe + SST body
    /// decode + parquet row scan on the second access.
    ///
    /// Scope: the cache lives as long as the `Snapshot`. Snapshots are
    /// cheap and built per query in the executor (`writer.snapshot()`),
    /// so the cache fills during one query and drops when the query
    /// finishes — no cross-query staleness risk.
    ///
    /// Mutex (not RefCell) because the snapshot is shared across the
    /// tokio executor via `&Snapshot<'_>` and the tree-walking executor
    /// drives multiple `lookup_node` calls in async tasks.
    node_cache: Mutex<HashMap<(String, NodeId), Option<NodeView>>>,
    /// Cold node lookup routing (RFC-003):
    /// - `Force(false)` — always full-body GET (legacy, populates
    /// body cache). Used by `read_latency.cold_no_cache`.
    /// - `Force(true)` — always ranged GET (footer + page index +
    /// column pages). Used by `read_latency.cold_ranged_reads`.
    /// - `Auto` (default) — full-body when `desc.size_bytes` is
    /// below `ranged_threshold_bytes`, ranged otherwise. Picks
    /// full-body for small SSTs where RTT dominates transfer, and
    /// ranged for large SSTs where transfer dominates RTT.
    ranged_mode: RangedMode,
    /// Size at which `Auto` mode switches from full-body to ranged.
    /// Default 16 MiB — empirically at ~7 MiB (1 M nodes) full body
    /// wins on a typical R2/laptop deploy; at ~70 MiB (10 M nodes)
    /// ranged dominates. 16 MiB lands the threshold somewhere
    /// reasonable without forcing ranged on for the small SSTs that
    /// hit the test path.
    ranged_threshold_bytes: u64,
    /// Process-wide CSR cache (RFC-018). Populated via
    /// [`Self::with_adjacency_cache`]. Cross-snapshot reuse keyed by
    /// `(manifest_version, edge_type, direction)`. Consulted by
    /// `edge_lookup` only when `NAMIDB_ADJACENCY=1` — guards correctness
    /// for callers that rely on full `EdgeView.properties` for SST-sourced
    /// edges (the slim CSR returns empty maps; see the RFC §4 caveat).
    adjacency_cache: Option<Arc<AdjacencyCache>>,
    /// Process-wide cross-snapshot NodeView cache (RFC-019).
    /// Populated via [`Self::with_shared_node_cache`]. Promotion path:
    /// L1 (per-snap `node_cache`) → L2 (this Arc) → L3 (SST walk).
    /// Slot key is `(manifest_version, label, NodeId)`; same invariants
    /// as the adjacency cache. Caches both positive (`Some(view)`) and
    /// negative (`None`) outcomes — a tombstoned key resolved once
    /// stays resolved for every subsequent snapshot at that manifest
    /// version.
    shared_node_cache: Option<Arc<NodeViewCache>>,
    /// Cross-snapshot lazy index over `(label, property) → value → NodeId`
    /// (RFC-pending). Attached via [`Self::with_property_index_cache`].
    /// `Snapshot::lookup_node_by_property` populates it on first miss
    /// and reuses it for the warm-path point lookups.
    property_index_cache: Option<Arc<crate::property_index::PropertyIndexCache>>,
    /// Per-snapshot fallback for decoded node-SST row groups, keyed by
    /// `(absolute SST path, row-group index)`. Used by
    /// [`Self::batch_lookup_nodes`] ONLY when no process-wide [`SstCache`]
    /// is attached; with a cache attached, decoded row groups live in the
    /// byte-budgeted `SstCache` tier and are shared across snapshots.
    /// Holding row groups (not whole SSTs) bounds this map by what one
    /// query actually touches — an L1-compacted whole-dataset SST no
    /// longer materialises in full per snapshot.
    decoded_node_row_groups: Mutex<HashMap<NodeRowGroupKey, DecodedNodeRowGroup>>,
    /// Read-your-own-writes overlay (RFC-026). A writer's staged-but-
    /// uncommitted batch, materialised as a second memtable and consulted
    /// alongside the committed `memtable`. The staged ops carry LSNs
    /// strictly greater than any committed LSN, so the existing
    /// last-LSN-wins merge resolves a staged upsert over the committed row
    /// and a staged tombstone hides it, with no separate read engine.
    /// `None` for every read outside a write context (auto-commit reads,
    /// the HTTP read path, the Bolt auto-commit branch), which is the only
    /// behaviour that changes nothing.
    ///
    /// The node read paths merge it via [`node_entries`](Self::node_entries)
    /// and [`node_mem_entry`](Self::node_mem_entry); the edge read paths
    /// merge it via [`edge_mem_entries`](Self::edge_mem_entries) (RFC-026
    /// edge overlay), so a traversal over an edge staged earlier in the
    /// same statement or transaction sees it.
    overlay: Option<MemtableSnapshot>,
}

/// Cold-path routing policy for [`Snapshot::lookup_node`]. See
/// [`Snapshot::ranged_mode`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RangedMode {
    /// Pick based on `desc.size_bytes` vs the snapshot's
    /// `ranged_threshold_bytes`.
    Auto,
    /// Always full-body GET (legacy).
    Force(bool),
}

impl RangedMode {
    /// Resolve to a yes/no decision for a specific SST size.
    fn enable_for(self, size_bytes: u64, threshold: u64) -> bool {
        match self {
            RangedMode::Auto => size_bytes >= threshold,
            RangedMode::Force(b) => b,
        }
    }
}

/// Default `ranged_threshold_bytes` when `Auto` mode is in effect.
pub const DEFAULT_RANGED_THRESHOLD_BYTES: u64 = 16 * 1024 * 1024;

impl<'mt> std::fmt::Debug for Snapshot<'mt> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Snapshot")
            .field("version", &self.manifest.manifest.version)
            .field("memtable_entries", &self.memtable.len())
            .field("sst_count", &self.manifest.manifest.ssts.len())
            .field("cache_active", &self.cache.is_some())
            .finish()
    }
}

impl<'mt> Snapshot<'mt> {
    pub fn new(
        manifest: LoadedManifest,
        memtable: &'mt MemtableSnapshot,
        store: Arc<dyn ObjectStore>,
        paths: NamespacePaths,
    ) -> Self {
        Self {
            manifest,
            memtable,
            store,
            paths,
            cache: None,
            node_cache: Mutex::new(HashMap::new()),
            ranged_mode: RangedMode::Auto,
            ranged_threshold_bytes: DEFAULT_RANGED_THRESHOLD_BYTES,
            adjacency_cache: None,
            shared_node_cache: None,
            property_index_cache: None,
            decoded_node_row_groups: Mutex::new(HashMap::new()),
            overlay: None,
        }
    }

    /// Attach a process-wide [`SstCache`] so subsequent body and bloom
    /// GETs go through the cache. Caller-supplied so the same cache can
    /// be shared across multiple `Snapshot`s and `WriterSession`s.
    pub fn with_cache(mut self, cache: SstCache) -> Self {
        self.cache = Some(cache);
        self
    }

    /// Force the cold-path routing policy. `Force(true)` makes every
    /// cold lookup use the RFC-003 ranged reader; `Force(false)`
    /// always issues a full-body GET (legacy). Override the
    /// size-based `Auto` heuristic for benches or for callers that
    /// know their workload.
    pub fn with_ranged_reads(mut self, enabled: bool) -> Self {
        self.ranged_mode = RangedMode::Force(enabled);
        self
    }

    /// Tune the SST size at which `Auto` mode switches from full-body
    /// to ranged. See [`DEFAULT_RANGED_THRESHOLD_BYTES`] for the
    /// rationale; smaller thresholds favour bandwidth savings at the
    /// cost of an extra round-trip on small SSTs.
    pub fn with_ranged_threshold_bytes(mut self, threshold: u64) -> Self {
        self.ranged_threshold_bytes = threshold;
        self
    }

    /// Attach a process-wide [`AdjacencyCache`]. When `NAMIDB_ADJACENCY=1`
    /// is set, `out_edges` / `in_edges` / `edge_lookup` resolve through
    /// the CSR materialised by that cache (RFC-018). When unset,
    /// the cache is ignored and the legacy SST-scan path is used —
    /// preserving full edge-property coverage for tests that need it.
    pub fn with_adjacency_cache(mut self, cache: Arc<AdjacencyCache>) -> Self {
        self.adjacency_cache = Some(cache);
        self
    }

    /// Attach a process-wide cross-snapshot [`NodeViewCache`]. When
    /// `NAMIDB_NODE_CACHE=1` is set, `lookup_node` consults the cache as
    /// L2 between the per-snapshot intra-cache and the SST walk
    /// (RFC-019). When unset the cache is ignored and the legacy
    /// L1-only behaviour is preserved.
    pub fn with_shared_node_cache(mut self, cache: Arc<NodeViewCache>) -> Self {
        self.shared_node_cache = Some(cache);
        self
    }

    /// Attach a cross-snapshot lazy property index. The first call to
    /// [`Self::lookup_node_by_property`] for any (label, prop) pair
    /// scans the label once and builds the `value → NodeId` map; every
    /// subsequent call (even on a different snapshot from the same
    /// writer) is an `O(1)` `HashMap::get`.
    pub fn with_property_index_cache(
        mut self,
        cache: Arc<crate::property_index::PropertyIndexCache>,
    ) -> Self {
        self.property_index_cache = Some(cache);
        self
    }

    /// Attach a read-your-own-writes overlay (RFC-026): a writer's staged
    /// batch, materialised as a second memtable, that the node read paths
    /// consult alongside the committed `memtable`. Built by
    /// [`crate::ingest::WriterSession::overlay_snapshot`]. See the
    /// [`Self::overlay`] field for the merge semantics.
    pub fn with_overlay(mut self, overlay: MemtableSnapshot) -> Self {
        self.overlay = Some(overlay);
        self
    }

    /// Node memtable entries to merge at read time: the committed
    /// `memtable`, with the staged overlay (RFC-026) chained on when this
    /// is an overlay snapshot. Staged LSNs are strictly greater than any
    /// committed LSN, so callers' [`update_node_winner`] last-LSN-wins
    /// merge picks the staged op for a key present in both. The overlay
    /// yields only `MemKey::Node` entries, matching `iter_nodes`.
    fn node_entries(&self) -> impl Iterator<Item = (&MemKey, &MemEntry)> {
        self.memtable
            .iter_nodes()
            .chain(self.overlay.iter().flat_map(|o| o.iter_nodes()))
    }

    /// Memtable entries to consult on the edge read paths (RFC-026 edge
    /// overlay): the committed `memtable`, with the writer's staged batch
    /// chained on when this is an overlay snapshot. Staged LSNs are
    /// strictly greater than any committed LSN, so the per-partner /
    /// per-edge last-LSN-wins merge in every edge read path picks a staged
    /// upsert or tombstone over the committed edge. Unlike
    /// [`node_entries`](Self::node_entries) this yields every entry; the
    /// edge paths filter `MemKey::Edge` inline, exactly as they did over
    /// the bare committed `memtable`, so node entries are skipped. When
    /// nothing is staged (`overlay` is `None`) this is the committed
    /// `memtable` alone.
    fn edge_mem_entries(&self) -> impl Iterator<Item = (&MemKey, &MemEntry)> {
        self.memtable
            .iter()
            .chain(self.overlay.iter().flat_map(|o| o.iter()))
    }

    /// Point read of a single node's memtable entry with the staged
    /// overlay (RFC-026) winning when present. A staged tombstone returns
    /// the tombstone entry (high LSN), so the caller's last-LSN-wins merge
    /// hides the committed row.
    ///
    /// Staged LSNs are strictly greater than any committed LSN (the writer
    /// seeds `next_lsn` past every committed LSN on open). We compare the
    /// two LSNs anyway, rather than blindly trusting the overlay, so a
    /// future regression in LSN allocation degrades to the same
    /// last-LSN-wins rule the scan path uses instead of silently surfacing
    /// a stale row.
    fn node_mem_entry(&self, id: NodeId) -> Option<&MemEntry> {
        let key = MemKey::Node { id };
        let committed = self.memtable.get(&key);
        let staged = self.overlay.as_ref().and_then(|o| o.get(&key));
        match (staged, committed) {
            (Some(s), Some(c)) => {
                debug_assert!(
                    s.lsn > c.lsn,
                    "overlay LSN {} must exceed committed LSN {}",
                    s.lsn,
                    c.lsn
                );
                if s.lsn >= c.lsn {
                    Some(s)
                } else {
                    Some(c)
                }
            }
            (Some(s), None) => Some(s),
            (None, c) => c,
        }
    }

    /// Point-lookup a node by a *unique* user property. The first call
    /// per (label, prop) pays a full label scan to populate the
    /// cross-snapshot cache; subsequent calls are `O(1)`. Caller is
    /// responsible for the unique invariant — without it the lookup
    /// returns an arbitrary matching row.
    ///
    /// Today only `String`-valued properties are indexed (LDBC's `id`).
    /// Non-string types fall back to the scan + filter path.
    pub async fn lookup_node_by_property(
        &self,
        label: &str,
        property: &str,
        value: &str,
    ) -> Result<Option<NodeView>> {
        namidb_core::profile_scope!("Snapshot::lookup_node_by_property");
        // 1. Try the cross-snapshot in-memory index — `O(1)` warm path.
        if let Some(cache) = &self.property_index_cache {
            if let Some(idx) = cache.get(label, property) {
                if let Some(node_id) = idx.get(value).copied() {
                    return self.lookup_node(label, node_id).await;
                } else {
                    // Property is declared unique → "not in index" is a
                    // definitive negative answer, no need to scan.
                    return Ok(None);
                }
            }
        }

        // 2. Sidecar path (RFC-pending): every Nodes SST in this scope
        // emits a `value → NodeId` map alongside the body on flush. If
        // every candidate SST carries the sidecar for `property`, we
        // can resolve the lookup with one bincode decode per SST
        // instead of a full label scan.
        //
        // "Candidate" is scoped to SSTs that can actually contain a live
        // row of `label`: an unrelated label's SST lacking the sidecar must
        // not demote this label's lookups to a full scan (previously any
        // multi-label deployment degraded this way). Excluded SSTs cannot
        // contribute a live match, and their tombstones still apply at
        // confirm time via `lookup_node`, which consults every SST.
        let node_sst_idxs: Vec<usize> = self.manifest.index.node_descriptors();
        let have_node_ssts = !node_sst_idxs.is_empty();
        let sst_idxs: Vec<usize> = node_sst_idxs
            .into_iter()
            .filter(|i| node_sst_can_contain_label(&self.manifest.manifest, *i, label))
            .collect();
        let all_have_sidecar = have_node_ssts
            && sst_idxs.iter().all(|i| {
                self.manifest.manifest.ssts[*i]
                    .unique_property_indices
                    .iter()
                    .any(|d| d.property == property)
            });
        if all_have_sidecar {
            namidb_core::profile_scope!("Snapshot::lookup_node_by_property.sidecar");
            // Memtable wins on conflicts: scan memtable first for any
            // upsert / tombstone of the same `value` so cross-store
            // last-write-wins logic stays correct. Memtable rows
            // without an LSN newer than the SST sidecar's authoritative
            // entry won't override it, but a memtable upsert with the
            // same `value` does: the user might have re-inserted a row
            // under the same property between the SST's flush and
            // ours. Materialise as a full label scan over the memtable
            // — bounded by memtable size, not the SST.
            let mut winner: Option<(u64, namidb_core::id::NodeId, bool)> = None;
            for (mk, e) in self.node_entries() {
                if let MemKey::Node { id } = mk {
                    match &e.op {
                        MemOp::Upsert(payload) => {
                            let rec = NodeWriteRecord::decode(payload)?;
                            if record_carries_label(&rec, label, &self.manifest.manifest.label_dict)
                            {
                                if let Some(namidb_core::Value::Str(s)) =
                                    rec.properties.get(property)
                                {
                                    if s == value {
                                        let bump = winner
                                            .as_ref()
                                            .map(|(lsn, _, _)| e.lsn > *lsn)
                                            .unwrap_or(true);
                                        if bump {
                                            winner = Some((e.lsn, *id, false));
                                        }
                                    }
                                }
                            }
                        }
                        MemOp::Tombstone => {
                            // A node tombstone on a winning id removes it; a
                            // tombstone on a non-winning id is irrelevant.
                            if let Some((lsn, win_id, _)) = winner {
                                if win_id == *id && e.lsn > lsn {
                                    winner = Some((e.lsn, *id, true));
                                }
                            }
                        }
                    }
                }
            }

            // SST sidecar pass: bincode-decode each candidate's
            // `(value → NodeId)` map and probe `value`. Last LSN wins
            // when multiple SSTs carry an entry for the same value;
            // SST LSN is `max_lsn` of the SST.
            for idx in &sst_idxs {
                let desc = &self.manifest.manifest.ssts[*idx];
                let sidecar_desc = desc
                    .unique_property_indices
                    .iter()
                    .find(|d| d.property == property)
                    .expect("all_have_sidecar guard");
                let absolute = format!(
                    "{}/{}",
                    self.paths.namespace_prefix().as_ref(),
                    sidecar_desc.path
                );
                // Reuse the body cache for sidecar bodies too. They're
                // immutable per UUIDv7 path so the standard cache key
                // works without conflict.
                let body = if let Some(b) = self.cache_get(&absolute) {
                    b
                } else {
                    let object_path = object_store::path::Path::from(absolute.clone());
                    let bytes = self
                        .store
                        .get(&object_path)
                        .await
                        .map_err(Error::ObjectStore)?
                        .bytes()
                        .await
                        .map_err(Error::ObjectStore)?;
                    if let Some(cache) = &self.cache {
                        cache.insert(absolute.clone(), bytes.clone());
                    }
                    bytes
                };
                let map: std::collections::BTreeMap<String, [u8; 16]> = bincode::deserialize(&body)
                    .map_err(|e| Error::invariant(format!("unique-index bincode decode: {e}")))?;
                if let Some(id_bytes) = map.get(value) {
                    let id = namidb_core::id::NodeId::from_uuid(Uuid::from_bytes(*id_bytes));
                    let bump = winner
                        .as_ref()
                        .map(|(lsn, _, _)| desc.max_lsn > *lsn)
                        .unwrap_or(true);
                    if bump {
                        winner = Some((desc.max_lsn, id, false));
                    }
                }
            }

            return match winner {
                Some((_, _, true)) => Ok(None), // tombstone-on-winner
                Some((_, id, false)) => match self.lookup_node(label, id).await? {
                    // Re-verify the current value: the SST sidecar maps
                    // `value → id` as of flush time, but a later SET may have
                    // renamed the property. If the live node no longer carries
                    // `value`, the sidecar entry is stale — return no match so a
                    // rename can't trigger a false unique-constraint violation
                    // or a wrong MATCH hit. A memtable winner already matched at
                    // its current value, so this never rejects a live match.
                    Some(view)
                        if matches!(view.properties.get(property),
                            Some(namidb_core::Value::Str(s)) if s == value) =>
                    {
                        Ok(Some(view))
                    }
                    _ => Ok(None),
                },
                None => Ok(None), // not in any sidecar
            };
        }

        // 3. Legacy cold path: full label scan to build the index, then look up.
        //
        // Reached when at least one SST in the scope was written by a
        // pre-sidecar build (or when the property wasn't declared
        // `unique` at flush time). The in-memory cache caches the
        // result so subsequent calls bypass the scan.
        let all_nodes = self.scan_label(label).await?;
        let mut idx: std::collections::HashMap<String, namidb_core::id::NodeId> =
            std::collections::HashMap::with_capacity(all_nodes.len());
        let mut found: Option<NodeView> = None;
        for view in &all_nodes {
            if let Some(namidb_core::Value::Str(s)) = view.properties.get(property) {
                if s == value {
                    found = Some(view.clone());
                }
                idx.insert(s.clone(), view.id);
            }
        }
        if let Some(cache) = &self.property_index_cache {
            cache.insert(
                label.to_string(),
                property.to_string(),
                std::sync::Arc::new(idx),
            );
        }
        Ok(found)
    }

    /// Resolve `MATCH (a:label {property: value})` for a NON-unique
    /// `indexed` property through the equality-index sidecars, returning
    /// every live node carrying that value.
    ///
    /// Each in-scope Nodes SST emits a `value → [NodeId, ...]` posting list
    /// for an `indexed` property. When every in-scope SST carries the
    /// sidecar we union the posting lists (plus any memtable upserts) into a
    /// candidate set, then *confirm* each candidate with `lookup_node`: that
    /// resolves cross-store last-write-wins and tombstones, and we keep only
    /// nodes whose CURRENT value still equals `value`. Confirmation makes
    /// the lookup correct even when a node was deleted or had its value
    /// changed after an older sidecar captured it (both yield a candidate
    /// that fails the re-check). Falls back to a full label scan when any
    /// in-scope SST predates the sidecar. String-valued properties only.
    pub async fn lookup_nodes_by_property(
        &self,
        label: &str,
        property: &str,
        value: &str,
    ) -> Result<Vec<NodeView>> {
        namidb_core::profile_scope!("Snapshot::lookup_nodes_by_property");

        // Same label scoping as `lookup_node_by_property`: only SSTs that
        // can contain a live row of `label` need the sidecar; the rest can
        // contribute no posting and must not disable the fast path.
        let node_sst_idxs: Vec<usize> = self.manifest.index.node_descriptors();
        let have_node_ssts = !node_sst_idxs.is_empty();
        let sst_idxs: Vec<usize> = node_sst_idxs
            .into_iter()
            .filter(|i| node_sst_can_contain_label(&self.manifest.manifest, *i, label))
            .collect();
        let all_have_sidecar = have_node_ssts
            && sst_idxs.iter().all(|i| {
                self.manifest.manifest.ssts[*i]
                    .equality_property_indices
                    .iter()
                    .any(|d| d.property == property)
            });

        // Cold path: a pre-sidecar SST is in scope (or the property was not
        // `indexed` at flush time). Scan + filter, returning every match.
        if !all_have_sidecar {
            let all_nodes = self.scan_label(label).await?;
            return Ok(all_nodes
                .into_iter()
                .filter(|v| {
                    matches!(v.properties.get(property),
                        Some(namidb_core::Value::Str(s)) if s == value)
                })
                .collect());
        }

        namidb_core::profile_scope!("Snapshot::lookup_nodes_by_property.sidecar");
        // Gather candidate ids: memtable upserts carrying `value`, plus the
        // union of every SST posting list under `value`.
        let mut candidates: std::collections::BTreeSet<namidb_core::id::NodeId> =
            std::collections::BTreeSet::new();
        for (mk, e) in self.node_entries() {
            if let MemKey::Node { id } = mk {
                if let MemOp::Upsert(payload) = &e.op {
                    let rec = NodeWriteRecord::decode(payload)?;
                    if record_carries_label(&rec, label, &self.manifest.manifest.label_dict) {
                        if let Some(namidb_core::Value::Str(s)) = rec.properties.get(property) {
                            if s == value {
                                candidates.insert(*id);
                            }
                        }
                    }
                }
            }
        }
        for idx in &sst_idxs {
            let desc = &self.manifest.manifest.ssts[*idx];
            let sidecar_desc = desc
                .equality_property_indices
                .iter()
                .find(|d| d.property == property)
                .expect("all_have_sidecar guard");
            let absolute = format!(
                "{}/{}",
                self.paths.namespace_prefix().as_ref(),
                sidecar_desc.path
            );
            let body = if let Some(b) = self.cache_get(&absolute) {
                b
            } else {
                let object_path = object_store::path::Path::from(absolute.clone());
                let bytes = self
                    .store
                    .get(&object_path)
                    .await
                    .map_err(Error::ObjectStore)?
                    .bytes()
                    .await
                    .map_err(Error::ObjectStore)?;
                if let Some(cache) = &self.cache {
                    cache.insert(absolute.clone(), bytes.clone());
                }
                bytes
            };
            let map: std::collections::BTreeMap<String, Vec<[u8; 16]>> =
                bincode::deserialize(&body)
                    .map_err(|e| Error::invariant(format!("equality-index bincode decode: {e}")))?;
            if let Some(ids) = map.get(value) {
                for id_bytes in ids {
                    candidates.insert(namidb_core::id::NodeId::from_uuid(Uuid::from_bytes(
                        *id_bytes,
                    )));
                }
            }
        }

        // Confirm each candidate against its current value. `lookup_node`
        // returns None for a tombstoned id and the live view otherwise; we
        // drop any whose value no longer matches (the value-changed case).
        let mut out = Vec::with_capacity(candidates.len());
        for id in candidates {
            if let Some(view) = self.lookup_node(label, id).await? {
                if matches!(view.properties.get(property),
                    Some(namidb_core::Value::Str(s)) if s == value)
                {
                    out.push(view);
                }
            }
        }
        Ok(out)
    }

    pub fn manifest(&self) -> &LoadedManifest {
        &self.manifest
    }

    /// Manifest version this snapshot is pinned at. Surfaced in Bolt
    /// bookmarks and observability metrics (RFC-021).
    pub fn manifest_version(&self) -> u64 {
        self.manifest.manifest.version
    }

    /// Every edge type observable through this snapshot — declared in the
    /// manifest schema, present in the borrowed memtable, or persisted in
    /// at least one SST descriptor (forward or inverse). Mirrors
    /// [`crate::ingest::WriterSession::observed_edge_types`] for the
    /// read-side: query executors that need to fan-out across all edge
    /// types (e.g. typeless `Expand`, `DETACH DELETE` on the read path)
    /// can rely on this rather than the bare declared schema, which is
    /// empty for namespaces that never went through `SchemaBuilder`.
    pub fn observed_edge_types(&self) -> Vec<String> {
        use std::collections::BTreeSet;
        let mut set: BTreeSet<String> = self
            .manifest
            .manifest
            .schema
            .edge_types
            .keys()
            .cloned()
            .collect();
        for (key, _) in self.memtable.iter() {
            if let MemKey::Edge { edge_type, .. } = key {
                set.insert(edge_type.clone());
            }
        }
        for sst in &self.manifest.manifest.ssts {
            if matches!(sst.kind, SstKind::EdgesFwd | SstKind::EdgesInv) {
                set.insert(sst.scope.clone());
            }
        }
        set.into_iter().collect()
    }

    /// Endpoint labels for every observable edge type.
    ///
    /// Declared edge types (`SchemaBuilder::edge_type(name, src, dst)`)
    /// come back verbatim from the manifest schema with `inferred = false`.
    /// Edge types that were only ever created by raw Cypher (no
    /// `SchemaBuilder`) are missing endpoints in the declared schema;
    /// for those, sample one live edge from the memtable, resolve its
    /// endpoint labels, and return them with `inferred = true`.
    ///
    /// Sampling is best-effort and cheap: we walk the memtable once to
    /// build a `NodeId → label` map for memtable-resident nodes, then
    /// pick the first edge per type and read its endpoints. If the
    /// sample's endpoints live in SSTs, we fan out one `lookup_node`
    /// per known label until one resolves. Schema reads are infrequent
    /// enough that the linear fallback is acceptable.
    pub async fn observed_edge_endpoints(&self) -> Result<Vec<EdgeEndpoint>> {
        use std::collections::BTreeMap;
        let declared = &self.manifest.manifest.schema.edge_types;

        // Build a memtable NodeId → label map once. Cheap (a few
        // BTreeMap insertions per node) and lets the common case
        // (newly-created edges live alongside their newly-created
        // nodes) skip the SST lookup entirely.
        let mut mem_node_label: BTreeMap<NodeId, String> = BTreeMap::new();
        for (key, entry) in self.memtable.iter() {
            if let MemKey::Node { id } = key {
                if let MemOp::Upsert(payload) = &entry.op {
                    let rec = NodeWriteRecord::decode(payload)?;
                    if let Some(name) = rec
                        .labels
                        .first()
                        .and_then(|&lid| self.manifest.manifest.label_dict.name(LabelId::new(lid)))
                    {
                        mem_node_label.insert(*id, name.to_string());
                    }
                }
            }
        }

        // Pick one sample edge per observed type, preferring memtable
        // edges so we can resolve endpoints synchronously through the
        // map above.
        let observed = self.observed_edge_types();
        let mut samples: BTreeMap<String, (NodeId, NodeId)> = BTreeMap::new();
        for (key, entry) in self.memtable.iter() {
            if let MemKey::Edge {
                edge_type,
                src,
                dst,
            } = key
            {
                if !matches!(entry.op, MemOp::Upsert(_)) {
                    continue;
                }
                if !declared.contains_key(edge_type) && !samples.contains_key(edge_type) {
                    samples.insert(edge_type.clone(), (*src, *dst));
                }
            }
        }

        let mut out: Vec<EdgeEndpoint> = Vec::with_capacity(observed.len());
        for edge_type in observed {
            if let Some(def) = declared.get(&edge_type) {
                out.push(EdgeEndpoint {
                    edge_type,
                    src_label: Some(def.src_label.clone()),
                    dst_label: Some(def.dst_label.clone()),
                    inferred: false,
                });
                continue;
            }
            // Prefer the memtable sample (freshest, resolved synchronously);
            // fall back to a forward-SST sample when the live memtable holds
            // no edge of this type — the common case for a bulk-loaded
            // namespace whose edges have already been flushed.
            let sample = match samples.get(&edge_type) {
                Some(pair) => Some(*pair),
                None => self.first_sst_edge(&edge_type).await?,
            };
            let (src_label, dst_label) = match sample {
                Some((src, dst)) => (
                    self.find_node_label(src, &mem_node_label).await?,
                    self.find_node_label(dst, &mem_node_label).await?,
                ),
                None => (None, None),
            };
            out.push(EdgeEndpoint {
                edge_type,
                src_label,
                dst_label,
                inferred: true,
            });
        }
        Ok(out)
    }

    /// Sample one live `(src, dst)` edge of `edge_type` from the forward
    /// SSTs. Used by [`Self::observed_edge_endpoints`] when the live
    /// memtable carries no edge of the type — the common case for a
    /// bulk-loaded namespace whose edges were flushed to SSTs. Reads the
    /// key columns of forward-SST descriptors in manifest order and
    /// returns the first non-tombstone row; `None` if every forward SST is
    /// empty / all-tombstone. Property streams are never decoded, and we
    /// stop at the first match, so the cost is bounded by one SST's key
    /// section — acceptable for an infrequent schema read.
    async fn first_sst_edge(&self, edge_type: &str) -> Result<Option<(NodeId, NodeId)>> {
        for &idx in self
            .manifest
            .index
            .scope_descriptors(SstKind::EdgesFwd, edge_type)
        {
            let desc = &self.manifest.manifest.ssts[idx];
            let body = self.get_sst_body(desc).await?;
            let reader = EdgeSstReader::open(body)?;
            for row in reader.scan_all_edges()? {
                if row.tombstone {
                    continue;
                }
                let src = NodeId::from_uuid(Uuid::from_bytes(row.key_id));
                let dst = NodeId::from_uuid(Uuid::from_bytes(row.partner_id));
                return Ok(Some((src, dst)));
            }
        }
        Ok(None)
    }

    /// Resolve the label of `id` by checking the memtable map first,
    /// then probing each observed label's SSTs in turn. Used only by
    /// [`Self::observed_edge_endpoints`] to enrich undeclared edge
    /// types, so a linear scan over labels is acceptable.
    async fn find_node_label(
        &self,
        id: NodeId,
        mem_node_label: &std::collections::BTreeMap<NodeId, String>,
    ) -> Result<Option<String>> {
        if let Some(label) = mem_node_label.get(&id) {
            return Ok(Some(label.clone()));
        }
        for label in self.observed_labels() {
            if self.lookup_node(&label, id).await?.is_some() {
                return Ok(Some(label));
            }
        }
        Ok(None)
    }

    /// Every node label observable through this snapshot — declared in the
    /// manifest schema, present in the borrowed memtable, or persisted in
    /// at least one node SST. Sister to [`Self::observed_edge_types`]:
    /// query executors that need to fan-out across all labels (typeless
    /// `NodeScan`, full-graph counts) can rely on this rather than the
    /// declared schema, which is empty for namespaces that never went
    /// through `SchemaBuilder`.
    pub fn observed_labels(&self) -> Vec<String> {
        use std::collections::BTreeSet;
        let mut set: BTreeSet<String> = self
            .manifest
            .manifest
            .schema
            .labels
            .keys()
            .cloned()
            .collect();
        // The dictionary holds every label name ever interned in this
        // namespace (memtable writes intern into it before commit).
        for (_, name) in self.manifest.manifest.label_dict.iter() {
            set.insert(name.to_string());
        }
        // Legacy node SSTs still carry their single label as the scope; id-
        // primary SSTs use an empty scope and contribute nothing here.
        for sst in &self.manifest.manifest.ssts {
            if matches!(sst.kind, SstKind::Nodes) && !sst.scope.is_empty() {
                set.insert(sst.scope.clone());
            }
        }
        set.into_iter().collect()
    }

    /// Observed property names and types for `label`, merging the
    /// declared `LabelDef` with `PropertyColumnStats` from every node
    /// SST in scope.
    ///
    /// Declared properties always win — their `data_type` is
    /// authoritative even when the column also has SST stats. For
    /// labels where every property is declared (the common case) this
    /// is equivalent to reading `schema.label(name).properties` and
    /// stopping there.
    ///
    /// SST stats are consulted as a fallback for the corner cases
    /// where the declared schema and the persisted columns drift apart
    /// (e.g. a schema migration removed a property after some SSTs
    /// already shipped). All-NULL columns end up out of the returned
    /// map; the writer never saw a non-null value to record.
    ///
    /// What this method does *not* report: properties supplied at
    /// `CREATE` time without a matching `PropertyDef`. The flush path
    /// drops those into the `__overflow_json` stream (RFC-002 §2.1)
    /// rather than into typed columns, so the manifest has no type
    /// information to surface. Schema-introspection callers that need
    /// those still have to sample the actual data.
    pub fn observed_property_types_for_label(
        &self,
        label: &str,
    ) -> std::collections::BTreeMap<String, namidb_core::DataType> {
        use std::collections::BTreeMap;
        let mut out: BTreeMap<String, namidb_core::DataType> = BTreeMap::new();
        if let Some(def) = self.manifest.manifest.schema.labels.get(label) {
            for prop in &def.properties {
                out.insert(prop.name.clone(), prop.data_type.clone());
            }
        }
        for sst in &self.manifest.manifest.ssts {
            if !matches!(sst.kind, SstKind::Nodes) || sst.scope != label {
                continue;
            }
            for stat in &sst.property_stats {
                // PropertyColumnStats names carry the `prop_` Arrow
                // prefix; strip it before comparing against the
                // user-facing property name.
                let name = stat
                    .name
                    .strip_prefix("prop_")
                    .unwrap_or(stat.name.as_str());
                if out.contains_key(name) {
                    continue;
                }
                if let Some(dt) = stat.observed_data_type() {
                    out.insert(name.to_string(), dt);
                }
            }
        }
        out
    }

    /// Look up a single node by `(label, id)`. Returns `None` for both
    /// "never inserted" and "winning record is a tombstone" outcomes.
    #[instrument(skip(self), fields(label = label, id = %id))]
    pub async fn lookup_node(&self, label: &str, id: NodeId) -> Result<Option<NodeView>> {
        namidb_core::profile_scope!("Snapshot::lookup_node");
        // L1: intra-snapshot cache. Same (label, node_id) hit
        // repeatedly within one query (~10× reuse for IC09 friends-of-
        // friends, more for highly-connected nodes). Clone is cheap
        // (~100 ns) vs the cold SST walk (~378 µs).
        let intra_key = (label.to_string(), id);
        if let Some(cached) = self.node_cache.lock().unwrap().get(&intra_key).cloned() {
            namidb_core::profile::record("Snapshot::lookup_node.l1_hit", 0);
            return Ok(cached);
        }

        // L2: cross-snapshot NodeViewCache (RFC-019). Optional.
        // Slot key includes manifest_version so promoted entries cannot
        // serve stale data after the writer commits.
        if let Some(shared) = &self.shared_node_cache {
            let shared_key = NodeCacheKey {
                manifest_version: self.manifest.manifest.version,
                label: label.to_string(),
                node_id: id,
            };
            if let Some(cached) = shared.get(&shared_key) {
                namidb_core::profile::record("Snapshot::lookup_node.l2_hit", 0);
                // Promote into L1 so subsequent intra-snap calls skip L2.
                self.node_cache
                    .lock()
                    .unwrap()
                    .insert(intra_key, cached.clone());
                return Ok(cached);
            }
        }

        // L3: cold SST walk. Resolve the id-primary record, then keep it only
        // if it actually carries `label` (the cache slot is per `(label, id)`).
        let result = self
            .lookup_node_by_id(id)
            .await?
            .filter(|v| v.labels.contains(label));
        // Insert into L1.
        self.node_cache
            .lock()
            .unwrap()
            .insert(intra_key, result.clone());
        // Insert into L2 if attached.
        if let Some(shared) = &self.shared_node_cache {
            let shared_key = NodeCacheKey {
                manifest_version: self.manifest.manifest.version,
                label: label.to_string(),
                node_id: id,
            };
            shared.insert(shared_key, result.clone());
        }
        Ok(result)
    }

    /// Batched analogue of [`Self::lookup_node`]: probe many `ids` for
    /// the same `label` in one pass over the `(Nodes, label)` SST set.
    ///
    /// Returns a `Vec<Option<NodeView>>` aligned 1:1 with `ids`. `None`
    /// means absent or tombstoned at this snapshot. Duplicates in `ids`
    /// resolve to the same `NodeView` value (cheap clone — same Arc).
    ///
    /// Why this exists: in cold IC09-shaped workloads
    /// (`(a)-[:KNOWS]->(b)-[:KNOWS]->(c)`) the per-edge `lookup_node`
    /// loop in `walker::execute_expand` issues N×M calls (~2 k for SF1).
    /// Each call decodes the same Person SST once. The batched variant
    /// maps the probe ids to the row groups that can contain them (the
    /// writer keeps `node_id` ascending, so per-row-group stats
    /// partition the key space), decodes ONLY those row groups, and
    /// matches all `ids` against them in one pass. Decoded row groups
    /// are shared process-wide through the byte-budgeted [`SstCache`]
    /// tier, so neither repeated batch calls nor repeated snapshots
    /// (one per commit) re-decode — and an L1-compacted whole-dataset
    /// SST costs only the row groups actually probed, not a full
    /// materialisation.
    ///
    /// Layered consistency: results are LSN-merged across memtable + all
    /// candidate SSTs, exactly like the single-id path. The cache tiers
    /// are checked first (L1 intra-snapshot, L2 cross-snapshot when
    /// attached) so already-resolved ids skip the SST scan entirely.
    /// Fresh resolutions populate L1 and L2 on the way out.
    #[instrument(skip(self, ids), fields(label = label, ids_len = ids.len()))]
    pub async fn batch_lookup_nodes(
        &self,
        label: &str,
        ids: &[NodeId],
    ) -> Result<Vec<Option<NodeView>>> {
        namidb_core::profile_scope!("Snapshot::batch_lookup_nodes");
        if ids.is_empty() {
            return Ok(Vec::new());
        }

        let mut out: Vec<Option<NodeView>> = vec![None; ids.len()];

        // Group output indices by id_bytes so duplicate `ids` map to the
        // same view, and so the L2 / SST passes only do unique work.
        let mut id_to_outputs: HashMap<[u8; 16], Vec<usize>> = HashMap::new();
        for (i, id) in ids.iter().enumerate() {
            id_to_outputs.entry(*id.as_bytes()).or_default().push(i);
        }

        // L1 cache pass: drop any id that's already resolved.
        let mut pending: std::collections::HashSet<[u8; 16]> =
            id_to_outputs.keys().copied().collect();
        {
            let cache = self.node_cache.lock().unwrap();
            for (id_bytes, outputs) in &id_to_outputs {
                let id = NodeId::from_uuid(Uuid::from_bytes(*id_bytes));
                let intra_key = (label.to_string(), id);
                if let Some(cached) = cache.get(&intra_key).cloned() {
                    namidb_core::profile::record("Snapshot::batch_lookup_nodes.l1_hit", 0);
                    for &i in outputs {
                        out[i] = cached.clone();
                    }
                    pending.remove(id_bytes);
                }
            }
        }

        // L2 cache pass: same logic against the cross-snapshot cache.
        if let Some(shared) = &self.shared_node_cache {
            let manifest_version = self.manifest.manifest.version;
            let mut promote: Vec<(NodeCacheKey, Option<NodeView>)> = Vec::new();
            for id_bytes in &pending {
                let id = NodeId::from_uuid(Uuid::from_bytes(*id_bytes));
                let shared_key = NodeCacheKey {
                    manifest_version,
                    label: label.to_string(),
                    node_id: id,
                };
                if let Some(cached) = shared.get(&shared_key) {
                    namidb_core::profile::record("Snapshot::batch_lookup_nodes.l2_hit", 0);
                    for &i in &id_to_outputs[id_bytes] {
                        out[i] = cached.clone();
                    }
                    promote.push((shared_key.clone(), cached));
                }
            }
            let mut cache = self.node_cache.lock().unwrap();
            for (key, view) in promote {
                let id_bytes = *key.node_id.as_bytes();
                pending.remove(&id_bytes);
                cache.insert((key.label, key.node_id), view);
            }
        }

        if pending.is_empty() {
            return Ok(out);
        }

        // Aggregate winners across memtable + every (Nodes, label) SST
        // candidate. Last-LSN-wins, mirroring `lookup_node_uncached`.
        let mut winners: HashMap<[u8; 16], (u64, Option<NodeView>)> = HashMap::new();

        // 1. Memtable: probe each pending id.
        for id_bytes in &pending {
            let id = NodeId::from_uuid(Uuid::from_bytes(*id_bytes));
            if let Some(entry) = self.node_mem_entry(id) {
                let view = match &entry.op {
                    MemOp::Tombstone => None,
                    MemOp::Upsert(payload) => Some(node_view_from_payload(
                        id,
                        entry.lsn,
                        payload,
                        &self.manifest.manifest.label_dict,
                        "",
                    )?),
                };
                // Inline last-LSN-wins: equivalent to update_node_winner
                // but keyed by raw bytes (cheaper than NodeId for the SST
                // harvest path).
                match winners.get(id_bytes) {
                    Some((existing_lsn, _)) if *existing_lsn >= entry.lsn => {}
                    _ => {
                        winners.insert(*id_bytes, (entry.lsn, view));
                    }
                }
            }
        }

        // 2. SST pass: for every node descriptor (id-primary partition + any
        // legacy per-label SSTs), prune to the row groups whose `node_id`
        // min/max range can contain a pending id, decode ONLY those, and
        // harvest every pending id in one sweep over the record batches.
        let mut sorted_pending: Vec<[u8; 16]> = pending.iter().copied().collect();
        sorted_pending.sort_unstable();
        let sst_idxs: Vec<usize> = self.manifest.index.node_descriptors();
        for idx in sst_idxs {
            let desc = &self.manifest.manifest.ssts[idx];
            // Cheap pre-filter: skip the SST if its [min_key, max_key]
            // range is disjoint from every pending id. For typical
            // LDBC IDs this still admits the SST (UUIDv7 hashes spread
            // across the range), but it cheaply rules out partition
            // scenarios where a label is split into per-tenant SSTs.
            let min_key = desc.min_key;
            let max_key = desc.max_key;
            if !pending.iter().any(|id| id >= &min_key && id <= &max_key) {
                continue;
            }
            let absolute = format!("{}/{}", self.paths.namespace_prefix().as_ref(), desc.path);
            let label_def = self.label_def_for_node_sst(desc);
            let md = self.node_sst_metadata(desc, &absolute).await?;
            let needed = row_groups_for_keys(&md, &sorted_pending)?;
            if needed.is_empty() {
                continue;
            }
            // Decoded row-group cache: process-wide + byte-budgeted when an
            // SstCache is attached, per-snapshot fallback otherwise. Either
            // way it amortises the Parquet decode across the N batch calls a
            // factor-path Expand chain issues (one per parent_leaf); without
            // it, SF1 IC09 cold pays the decode ~150 times. Probe the
            // fallback map in a bounded scope so the MutexGuard is released
            // before the decode await below.
            let mut decoded: Vec<Arc<Vec<RecordBatch>>> = Vec::with_capacity(needed.len());
            let mut missing: Vec<usize> = Vec::new();
            for &rg in &needed {
                let hit = match &self.cache {
                    Some(cache) => cache.get_decoded_node_row_group(&absolute, rg),
                    None => self
                        .decoded_node_row_groups
                        .lock()
                        .unwrap()
                        .get(&(absolute.clone(), rg))
                        .cloned(),
                };
                match hit {
                    Some(b) => decoded.push(b),
                    None => missing.push(rg),
                }
            }
            if !missing.is_empty() {
                let fresh = self
                    .decode_node_row_groups(desc, &absolute, &label_def, &md, &missing)
                    .await?;
                for (rg, batches) in fresh {
                    let batches = Arc::new(batches);
                    // Last write wins on a race because both threads
                    // decoded identical bytes.
                    match &self.cache {
                        Some(cache) => {
                            cache.insert_decoded_node_row_group(
                                absolute.clone(),
                                rg,
                                batches.clone(),
                            );
                        }
                        None => {
                            self.decoded_node_row_groups
                                .lock()
                                .unwrap()
                                .insert((absolute.clone(), rg), batches.clone());
                        }
                    }
                    decoded.push(batches);
                }
            }
            for batches in &decoded {
                batch_harvest_node_rows(
                    batches,
                    &label_def,
                    &self.manifest.manifest.label_dict,
                    &desc.scope,
                    &pending,
                    &mut winners,
                )?;
            }
        }

        // 3. Push every (resolved or negative) outcome into the output
        // vector and populate the cache tiers.
        let shared = self.shared_node_cache.clone();
        let manifest_version = self.manifest.manifest.version;
        let mut cache_l1 = self.node_cache.lock().unwrap();
        for id_bytes in &pending {
            let view = winners
                .remove(id_bytes)
                .map(|(_, v)| v)
                .unwrap_or(None)
                .filter(|v| v.labels.contains(label));
            for &i in &id_to_outputs[id_bytes] {
                out[i] = view.clone();
            }
            let id = NodeId::from_uuid(Uuid::from_bytes(*id_bytes));
            cache_l1.insert((label.to_string(), id), view.clone());
            if let Some(ref shared) = shared {
                let shared_key = NodeCacheKey {
                    manifest_version,
                    label: label.to_string(),
                    node_id: id,
                };
                shared.insert(shared_key, view);
            }
        }

        Ok(out)
    }

    /// Force the legacy uncached path. Bypasses both L1 and L2. Used by
    /// parity tests (RFC-019) to compare against the tiered path
    /// without mutating env state.
    pub async fn lookup_node_via_uncached(
        &self,
        label: &str,
        id: NodeId,
    ) -> Result<Option<NodeView>> {
        Ok(self
            .lookup_node_by_id(id)
            .await?
            .filter(|v| v.labels.contains(label)))
    }

    /// The `LabelDef` to open a node SST with. Id-primary node SSTs carry
    /// `scope = ""` and no declared columns (every property in overflow); legacy
    /// single-label SSTs are still typed by their scope label.
    fn label_def_for_node_sst(&self, desc: &SstDescriptor) -> LabelDef {
        if desc.scope.is_empty() {
            LabelDef {
                name: String::new(),
                properties: Vec::new(),
            }
        } else {
            self.manifest
                .manifest
                .schema
                .label(&desc.scope)
                .cloned()
                .unwrap_or_else(|| LabelDef {
                    name: desc.scope.clone(),
                    properties: Vec::new(),
                })
        }
    }

    /// Footer + page-index metadata for a node SST, through the
    /// process-wide metadata cache when one is attached (RFC-003 — SSTs
    /// are immutable per UUIDv7 path, so a cached entry never goes
    /// stale). Cold: parses in-process when the body is local anyway
    /// (full-body routing, or an existing body-cache entry); otherwise
    /// fetches footer + page index over ranged GETs without pulling the
    /// body.
    async fn node_sst_metadata(
        &self,
        desc: &SstDescriptor,
        absolute: &str,
    ) -> Result<Arc<ParquetMetaData>> {
        if let Some(cache) = &self.cache {
            if let Some(md) = cache.get_metadata(absolute) {
                return Ok(md);
            }
        }
        let use_ranged = self
            .ranged_mode
            .enable_for(desc.size_bytes, self.ranged_threshold_bytes);
        let md = if !use_ranged || self.cache_get(absolute).is_some() {
            let body = self.get_sst_body(desc).await?;
            parse_node_sst_metadata(&body)?
        } else {
            load_node_sst_metadata_async(
                self.store.clone(),
                Path::from(absolute),
                desc.size_bytes,
            )
            .await?
        };
        if let Some(cache) = &self.cache {
            cache.insert_metadata(absolute.to_string(), md.clone());
        }
        Ok(md)
    }

    /// Decode `row_groups` (ascending) from the node SST at `desc`,
    /// split back into per-row-group batch vectors ready for the decoded
    /// cache. Routing mirrors the per-id cold path: full-body GET when
    /// ranged reads are off for this SST size (populates the body cache)
    /// or when the body is already cached; byte-ranged GETs of just the
    /// selected row groups otherwise.
    async fn decode_node_row_groups(
        &self,
        desc: &SstDescriptor,
        absolute: &str,
        label_def: &LabelDef,
        md: &Arc<ParquetMetaData>,
        row_groups: &[usize],
    ) -> Result<Vec<(usize, Vec<RecordBatch>)>> {
        let use_ranged = self
            .ranged_mode
            .enable_for(desc.size_bytes, self.ranged_threshold_bytes);
        let local_body = if !use_ranged {
            Some(self.get_sst_body(desc).await?)
        } else {
            self.cache_get(absolute)
        };
        if let Some(body) = local_body {
            // Per-group decode so each cache entry owns right-sized buffers
            // (a multi-group sync scan can emit batches spanning groups,
            // whose slices would pin — and double-count — shared buffers).
            let reader = NodeSstReader::open(label_def.clone(), body)?;
            return reader.scan_row_groups_each(md, row_groups);
        }
        let batches = node_scan_row_groups_async(
            self.store.clone(),
            Path::from(absolute),
            desc.size_bytes,
            label_def,
            row_groups.to_vec(),
            Some(md.clone()),
        )
        .await?;
        split_batches_by_row_group(md, row_groups, batches)
    }

    /// Id-primary cold lookup: resolve the last-LSN-wins record for `id` across
    /// the memtable and every node SST (decoding each row's label set), with no
    /// label scoping. `lookup_node` filters the result by label.
    async fn lookup_node_by_id(&self, id: NodeId) -> Result<Option<NodeView>> {
        namidb_core::profile_scope!("Snapshot::lookup_node_by_id");
        let id_bytes = *id.as_bytes();
        let dict = &self.manifest.manifest.label_dict;
        let mut winner: Option<(u64, Option<NodeView>)> = None;

        // 1. Memtable (highest LSN typically).
        if let Some(entry) = self.node_mem_entry(id) {
            let view = match &entry.op {
                MemOp::Tombstone => None,
                MemOp::Upsert(payload) => {
                    Some(node_view_from_payload(id, entry.lsn, payload, dict, "")?)
                }
            };
            winner = Some((entry.lsn, view));
        }

        // 2. Node SST candidates across every scope (id-primary), pruned by the
        // per-bucket min/max-key index; still bloom-probed + body-fetched.
        let candidates = self
            .manifest
            .index
            .node_candidates(&self.manifest.manifest.ssts, &id_bytes);
        for idx in candidates {
            let desc = &self.manifest.manifest.ssts[idx];
            // Decoded row-group tier, shared with `batch_lookup_nodes`: when
            // this SST's footer metadata is already cached, resolve which row
            // groups could hold `id` and serve the probe straight from the
            // process-wide decoded cache — no bloom fetch, no body GET, no
            // re-decode. This is what keeps a batch prewarm paying off for
            // the per-id lookups that follow it, even across snapshots. Any
            // miss (metadata or row group) falls through to the cold path
            // unchanged. Correctness: the writer keeps `node_id` ascending,
            // so the row-group stats are authoritative — an id outside every
            // kept row group is provably absent from this SST.
            if let Some(cache) = &self.cache {
                let absolute = format!("{}/{}", self.paths.namespace_prefix().as_ref(), desc.path);
                if let Some(md) = cache.get_metadata(&absolute) {
                    let needed = row_groups_for_keys(&md, std::slice::from_ref(&id_bytes))?;
                    let mut cached_groups: Option<Vec<Arc<Vec<RecordBatch>>>> =
                        Some(Vec::with_capacity(needed.len()));
                    for &rg in &needed {
                        match cache.get_decoded_node_row_group(&absolute, rg) {
                            Some(b) => cached_groups.as_mut().unwrap().push(b),
                            None => {
                                cached_groups = None;
                                break;
                            }
                        }
                    }
                    if let Some(groups) = cached_groups {
                        let label_def = self.label_def_for_node_sst(desc);
                        let mut candidate: Option<(u64, Option<NodeView>)> = None;
                        for batches in &groups {
                            if let Some(found) = find_node_row_in_batches(
                                batches,
                                &label_def,
                                id,
                                dict,
                                &desc.scope,
                            )? {
                                candidate = Some(found);
                                break;
                            }
                        }
                        if let Some((lsn, view)) = candidate {
                            match &winner {
                                None => winner = Some((lsn, view)),
                                Some((w_lsn, _)) if lsn > *w_lsn => winner = Some((lsn, view)),
                                _ => {}
                            }
                        }
                        continue;
                    }
                }
            }
            if !self.bloom_admits(desc, &id_bytes).await? {
                continue;
            }
            let label_def = self.label_def_for_node_sst(desc);
            // Cold-path routing (RFC-003):
            // - Ranged disabled (forced off, or `Auto` below the size
            // threshold): full-body GET via `get_sst_body` —
            // populates the body cache for subsequent warm reads.
            // - Ranged enabled (forced on, or `Auto` ≥ threshold):
            // probe the body cache first; on hit, decode in-process
            // (warm), on miss, footer + page index + column pages
            // only. Body cache is *not* populated by this path.
            let use_ranged = self
                .ranged_mode
                .enable_for(desc.size_bytes, self.ranged_threshold_bytes);
            let candidate = if !use_ranged {
                let body = self.get_sst_body(desc).await?;
                let reader = NodeSstReader::open(label_def.clone(), body)?;
                find_node_row(&reader, &label_def, id, dict, &desc.scope)?
            } else {
                let absolute = format!("{}/{}", self.paths.namespace_prefix().as_ref(), desc.path);
                if let Some(body) = self.cache_get(&absolute) {
                    let reader = NodeSstReader::open(label_def.clone(), body)?;
                    find_node_row(&reader, &label_def, id, dict, &desc.scope)?
                } else {
                    // Look up cached parquet metadata first; on hit we
                    // skip the footer + page-index round-trip entirely
                    // (RFC-003 warm-path optimisation).
                    let cached_meta = self.cache.as_ref().and_then(|c| c.get_metadata(&absolute));
                    let (batches, meta) = node_targeted_scan_async(
                        self.store.clone(),
                        Path::from(absolute.clone()),
                        desc.size_bytes,
                        &label_def,
                        &id_bytes,
                        cached_meta,
                    )
                    .await?;
                    // Cache the metadata for the next warm lookup on
                    // this SST. SSTs are immutable per UUIDv7 path so
                    // the entry never goes stale.
                    if let Some(cache) = &self.cache {
                        cache.insert_metadata(absolute, meta);
                    }
                    find_node_row_in_batches(&batches, &label_def, id, dict, &desc.scope)?
                }
            };
            if let Some((lsn, view)) = candidate {
                match &winner {
                    None => winner = Some((lsn, view)),
                    Some((w_lsn, _)) if lsn > *w_lsn => winner = Some((lsn, view)),
                    _ => {}
                }
            }
        }

        Ok(winner.and_then(|(_, view)| view))
    }

    /// Forward edges from `src` along `edge_type` (out-edges).
    #[instrument(skip(self), fields(edge_type = edge_type, src = %src))]
    pub async fn out_edges(&self, edge_type: &str, src: NodeId) -> Result<EdgeListView> {
        self.edge_lookup(edge_type, src, EdgeDirection::Forward)
            .await
    }

    /// Force the legacy SST-scan path for `out_edges`. Bypasses the
    /// `NAMIDB_ADJACENCY` toggle. Used by parity tests (RFC-018)
    /// to compare against [`Self::out_edges_via_csr`] on the same
    /// snapshot without mutating global env state.
    pub async fn out_edges_via_sst(&self, edge_type: &str, src: NodeId) -> Result<EdgeListView> {
        self.edge_lookup_via_sst(edge_type, src, EdgeDirection::Forward)
            .await
    }

    /// Force the CSR path for `out_edges`. Requires an `AdjacencyCache`
    /// attached via [`Self::with_adjacency_cache`]; returns
    /// `Error::invariant` otherwise. Slim path — SST-sourced edges come
    /// back with empty `properties` (see RFC-018 §4). Used by parity
    /// tests; bypasses the `NAMIDB_ADJACENCY` toggle.
    pub async fn out_edges_via_csr(&self, edge_type: &str, src: NodeId) -> Result<EdgeListView> {
        let cache = self
            .adjacency_cache
            .clone()
            .ok_or_else(|| Error::invariant("out_edges_via_csr called without adjacency cache"))?;
        self.edge_lookup_via_csr(cache, edge_type, src, EdgeDirection::Forward)
            .await
    }

    /// Materialise every node row visible under `label` at this snapshot.
    /// Equivalent to `scan_label_with_predicates_and_projection(label, &[], None)`.
    pub async fn scan_label(&self, label: &str) -> Result<Vec<NodeView>> {
        self.scan_label_with_predicates_and_projection(label, &[], None)
            .await
    }

    /// Predicate-pushed variant of [`scan_label`] (RFC-013).
    /// Equivalent to `scan_label_with_predicates_and_projection(label, predicates, None)`.
    pub async fn scan_label_with_predicates(
        &self,
        label: &str,
        predicates: &[ScanPredicate],
    ) -> Result<Vec<NodeView>> {
        self.scan_label_with_predicates_and_projection(label, predicates, None)
            .await
    }

    /// Predicate-pushed + column-projected variant of [`scan_label`]
    /// (RFC-013 + RFC-015, S12.5/S12.6). The SST reader uses
    /// per-row-group statistics to skip row-groups that cannot satisfy
    /// any predicate, and a Parquet `ProjectionMask` to read only the
    /// engine columns plus the property columns named in `projection`.
    ///
    /// Memtable rows are evaluated row-by-row via `eval_against_value`
    /// — already in-memory, so no IO. When `projection.is_some()`, the
    /// resulting `NodeView`'s `properties` map is filtered to the same
    /// set so callers see a uniform shape between SST-sourced and
    /// memtable-sourced rows.
    ///
    /// Empty `predicates` + `projection.is_none()` falls through to the
    /// legacy full scan path. Bloom probes are still intentionally
    /// skipped.
    #[instrument(skip(self, predicates, projection), fields(label = label, predicates = predicates.len(), projection = projection.as_ref().map(|p| p.len()).unwrap_or(0)))]
    pub async fn scan_label_with_predicates_and_projection(
        &self,
        label: &str,
        predicates: &[ScanPredicate],
        projection: Option<&[String]>,
    ) -> Result<Vec<NodeView>> {
        let dict = &self.manifest.manifest.label_dict;

        // (node_id) → (winning lsn, materialised view or tombstone marker).
        // Nodes are id-primary: materialise every node across the label-agnostic
        // memtable + node SSTs and keep only those whose decoded label set
        // contains `label` (filtered at the end).
        let mut latest: BTreeMap<NodeId, (u64, Option<NodeView>)> = BTreeMap::new();

        // 1. Memtable rows. Apply predicates after materialising the view; a
        // failing predicate yields a tombstone-like None so a lower-LSN SST row
        // for the same id is not spuriously surfaced.
        for (mk, entry) in self.node_entries() {
            let MemKey::Node { id } = mk else {
                continue;
            };
            let view = match &entry.op {
                MemOp::Tombstone => None,
                MemOp::Upsert(payload) => {
                    let mut v = node_view_from_payload(*id, entry.lsn, payload, dict, "")?;
                    if !node_view_matches_predicates(&v, predicates) {
                        None
                    } else {
                        // Apply projection to memtable views so SST-sourced
                        // and memtable-sourced rows have a uniform shape.
                        if let Some(keep) = projection {
                            let keep_set: std::collections::BTreeSet<&str> =
                                keep.iter().map(|s| s.as_str()).collect();
                            v.properties.retain(|k, _| keep_set.contains(k.as_str()));
                        }
                        Some(v)
                    }
                }
            };
            update_node_winner(&mut latest, *id, entry.lsn, view);
        }

        // 2. Every node SST: the id-primary partition plus any legacy per-label
        // SSTs. Each row's label set is decoded from `__labels` (or the scope
        // for legacy SSTs); the label filter is applied at the end.
        for idx in self.manifest.index.node_descriptors() {
            let desc = &self.manifest.manifest.ssts[idx];
            let sst_label_def = self.label_def_for_node_sst(desc);
            let body = self.get_sst_body(desc).await?;
            let reader = NodeSstReader::open(sst_label_def.clone(), body)?;
            // Build the projection set once per SST (declared properties
            // ∩ requested). When `projection.is_none()` we iterate every
            // declared property.
            let projection_set: Option<std::collections::BTreeSet<&str>> =
                projection.map(|cols| cols.iter().map(|s| s.as_str()).collect());

            for batch in reader.scan_with_predicates_and_projection(predicates, projection)? {
                // Cooperative cancellation (query timeout): one large SST can
                // decode into many batches, so probe the deadline per batch.
                crate::cancel::check()?;
                let id_col = batch
                    .column_by_name(COL_NODE_ID)
                    .and_then(|c| c.as_any().downcast_ref::<FixedSizeBinaryArray>())
                    .ok_or_else(|| Error::invariant("node_id column missing"))?;
                let tomb_col = batch
                    .column_by_name(COL_TOMBSTONE)
                    .and_then(|c| c.as_any().downcast_ref::<BooleanArray>())
                    .ok_or_else(|| Error::invariant("tombstone column missing"))?;
                let lsn_col = batch
                    .column_by_name(COL_LSN)
                    .and_then(|c| c.as_any().downcast_ref::<UInt64Array>())
                    .ok_or_else(|| Error::invariant("lsn column missing"))?;
                let sv_col = batch
                    .column_by_name(SCHEMA_VERSION)
                    .and_then(|c| c.as_any().downcast_ref::<UInt64Array>())
                    .ok_or_else(|| Error::invariant("__schema_version column missing"))?;
                let ovf_col = batch
                    .column_by_name(OVERFLOW_JSON)
                    .and_then(|c| c.as_any().downcast_ref::<StringArray>())
                    .ok_or_else(|| Error::invariant("__overflow_json column missing"))?;
                for row in 0..batch.num_rows() {
                    let row_id_bytes: [u8; 16] = id_col
                        .value(row)
                        .try_into()
                        .map_err(|_| Error::invariant("node_id row length != 16"))?;
                    let row_id = NodeId::from_uuid(Uuid::from_bytes(row_id_bytes));
                    let lsn = lsn_col.value(row);
                    if tomb_col.value(row) {
                        update_node_winner(&mut latest, row_id, lsn, None);
                        continue;
                    }
                    let mut properties: BTreeMap<String, Value> = BTreeMap::new();
                    for p in &sst_label_def.properties {
                        // Skip properties not in the projection (when one
                        // is set). Engine columns are still required and
                        // were included by the ProjectionMask.
                        if let Some(keep) = &projection_set {
                            if !keep.contains(p.name.as_str()) {
                                continue;
                            }
                        }
                        let col_name = prop_column_name(p);
                        // Defensive: if Parquet's ProjectionMask elided the
                        // column (because the caller asked for a subset)
                        // the column won't be in the batch — skip silently.
                        let Some(col) = batch.column_by_name(&col_name) else {
                            continue;
                        };
                        if let Some(v) = arrow_value_to_value(col.as_ref(), row, &p.data_type)? {
                            properties.insert(p.name.clone(), v);
                        }
                    }
                    if !ovf_col.is_null(row) {
                        let json_str = ovf_col.value(row);
                        let extra: BTreeMap<String, Value> = serde_json::from_str(json_str)?;
                        if let Some(keep) = &projection_set {
                            for (k, v) in extra {
                                if keep.contains(k.as_str()) {
                                    properties.insert(k, v);
                                }
                            }
                        } else {
                            properties.extend(extra);
                        }
                    }
                    let view = NodeView {
                        id: row_id,
                        labels: decode_node_labels(&batch, row, dict, &desc.scope),
                        properties,
                        lsn,
                        schema_version: sv_col.value(row),
                    };
                    // Row-group pruning is conservative; surviving rows
                    // still need a per-row 3VL check so we don't surface
                    // a row that fails any predicate. A failing predicate
                    // produces `None` at this LSN — older versions of the
                    // same id will still lose to a higher-LSN tombstone /
                    // upsert and stay correctly hidden.
                    if node_view_matches_predicates(&view, predicates) {
                        update_node_winner(&mut latest, row_id, lsn, Some(view));
                    } else {
                        update_node_winner(&mut latest, row_id, lsn, None);
                    }
                }
            }
        }

        // 3. Drop tombstones and rows that don't carry `label`; return in
        // ascending-id order (BTreeMap iter).
        Ok(latest
            .into_values()
            .filter_map(|(_, v)| v)
            .filter(|v| v.labels.contains(label))
            .collect())
    }

    /// Materialise every edge row of `edge_type` visible at this snapshot.
    /// Edges are returned grouped by `src` (ascending), then by `dst`
    /// (ascending). Tombstones win over older upserts and are pruned.
    ///
    /// v1 reads the forward partner SSTs (`EdgesFwd`). Declared edge
    /// property streams are still on the TODO list (RFC-002 §3.2.7); the
    /// memtable carries full property maps but SST-sourced edges land with
    /// empty `properties`. The merger consults both sources, so edges
    /// updated in the memtable retain their properties.
    #[instrument(skip(self), fields(edge_type = edge_type))]
    pub async fn scan_edge_type(&self, edge_type: &str) -> Result<Vec<EdgeView>> {
        let mut latest: BTreeMap<(NodeId, NodeId), (u64, Option<EdgeView>)> = BTreeMap::new();

        // 1. Memtable, then the writer's staged overlay (RFC-026 edge RYOW).
        for (mk, entry) in self.edge_mem_entries() {
            let MemKey::Edge {
                edge_type: et,
                src,
                dst,
            } = mk
            else {
                continue;
            };
            if et != edge_type {
                continue;
            }
            let view = match &entry.op {
                MemOp::Tombstone => None,
                MemOp::Upsert(payload) => {
                    let rec = EdgeWriteRecord::decode(payload)?;
                    Some(EdgeView {
                        edge_type: edge_type.to_string(),
                        src: *src,
                        dst: *dst,
                        properties: rec.properties,
                        lsn: entry.lsn,
                    })
                }
            };
            update_edge_winner(&mut latest, (*src, *dst), entry.lsn, view);
        }

        // 2. Forward SSTs only — the inverse partner duplicates the same
        // (src, dst, lsn) tuples in inverse order. Using one direction
        // keeps the merge unambiguous. Pull descriptors via the
        // manifest index instead of re-filtering every SST.
        // Declared property names from the schema (RFC-002 §3.2.7). Used
        // below to fan out the reader's per-stream decode and combine
        // with __overflow_json in `decode_edge_properties`.
        let declared_property_names: Vec<String> = self
            .manifest
            .manifest
            .schema
            .edge_type(edge_type)
            .map(|def| def.properties.iter().map(|p| p.name.clone()).collect())
            .unwrap_or_default();

        for &idx in self
            .manifest
            .index
            .scope_descriptors(SstKind::EdgesFwd, edge_type)
        {
            let desc = &self.manifest.manifest.ssts[idx];
            let body = self.get_sst_body(desc).await?;
            let reader = EdgeSstReader::open(body)?;
            let rows = reader.scan_all_edges()?;
            let overflows = reader.read_overflow_strings()?;
            let declared_streams = load_declared_streams(&reader, &declared_property_names)?;
            for (idx, row) in rows.iter().enumerate() {
                // Cooperative cancellation (query timeout): a strided probe so
                // a huge single edge SST aborts mid-decode, not just per SST.
                if idx % crate::cancel::CHECK_STRIDE == 0 {
                    crate::cancel::check()?;
                }
                let src = NodeId::from_uuid(Uuid::from_bytes(row.key_id));
                let dst = NodeId::from_uuid(Uuid::from_bytes(row.partner_id));
                let view = if row.tombstone {
                    None
                } else {
                    let properties = decode_edge_properties(
                        overflows.as_ref().and_then(|v| v.get(idx)),
                        &declared_streams,
                        idx,
                    )?;
                    Some(EdgeView {
                        edge_type: edge_type.to_string(),
                        src,
                        dst,
                        properties,
                        lsn: row.lsn,
                    })
                };
                update_edge_winner(&mut latest, (src, dst), row.lsn, view);
            }
        }

        Ok(latest.into_values().filter_map(|(_, v)| v).collect())
    }

    /// Count the live edges of `edge_type` visible at this snapshot.
    ///
    /// Same memtable + forward-SST merge as [`Self::scan_edge_type`]
    /// (last-writer-wins by LSN, tombstones pruned) but it never decodes
    /// edge property streams — it only tracks `(src, dst)` liveness. A
    /// global edge count is therefore `O(edges_of_type)` with no per-edge
    /// property decode and, crucially, no scan over every node. This is the
    /// primitive the query optimizer's edge-type-count pushdown calls in
    /// place of `NodeScan + Expand + Aggregate`.
    #[instrument(skip(self), fields(edge_type = edge_type))]
    pub async fn count_edge_type(&self, edge_type: &str) -> Result<u64> {
        // (src, dst) -> (winning_lsn, is_live). Mirrors scan_edge_type's
        // merge exactly, minus the EdgeView materialisation.
        let mut latest: BTreeMap<(NodeId, NodeId), (u64, bool)> = BTreeMap::new();

        // 1. Memtable, then the writer's staged overlay (RFC-026 edge RYOW).
        for (mk, entry) in self.edge_mem_entries() {
            let MemKey::Edge {
                edge_type: et,
                src,
                dst,
            } = mk
            else {
                continue;
            };
            if et != edge_type {
                continue;
            }
            let live = !matches!(entry.op, MemOp::Tombstone);
            update_edge_count_winner(&mut latest, (*src, *dst), entry.lsn, live);
        }

        // 2. Forward SSTs only — the inverse partner duplicates the same
        // (src, dst, lsn) tuples. No property decode: scan_all_edges yields
        // key/partner/lsn/tombstone, which is all a count needs.
        for &idx in self
            .manifest
            .index
            .scope_descriptors(SstKind::EdgesFwd, edge_type)
        {
            let desc = &self.manifest.manifest.ssts[idx];
            let body = self.get_sst_body(desc).await?;
            let reader = EdgeSstReader::open(body)?;
            let rows = reader.scan_all_edges()?;
            for (i, row) in rows.iter().enumerate() {
                // Cooperative cancellation (query timeout): strided probe.
                if i % crate::cancel::CHECK_STRIDE == 0 {
                    crate::cancel::check()?;
                }
                let src = NodeId::from_uuid(Uuid::from_bytes(row.key_id));
                let dst = NodeId::from_uuid(Uuid::from_bytes(row.partner_id));
                update_edge_count_winner(&mut latest, (src, dst), row.lsn, !row.tombstone);
            }
        }

        Ok(latest.into_values().filter(|(_, live)| *live).count() as u64)
    }

    /// Inverse edges into `dst` along `edge_type` (in-edges).
    #[instrument(skip(self), fields(edge_type = edge_type, dst = %dst))]
    pub async fn in_edges(&self, edge_type: &str, dst: NodeId) -> Result<EdgeListView> {
        self.edge_lookup(edge_type, dst, EdgeDirection::Inverse)
            .await
    }

    /// Force the legacy SST-scan path for `in_edges`. See
    /// [`Self::out_edges_via_sst`] for the rationale.
    pub async fn in_edges_via_sst(&self, edge_type: &str, dst: NodeId) -> Result<EdgeListView> {
        self.edge_lookup_via_sst(edge_type, dst, EdgeDirection::Inverse)
            .await
    }

    /// Force the CSR path for `in_edges`. See
    /// [`Self::out_edges_via_csr`] for the rationale.
    pub async fn in_edges_via_csr(&self, edge_type: &str, dst: NodeId) -> Result<EdgeListView> {
        let cache = self
            .adjacency_cache
            .clone()
            .ok_or_else(|| Error::invariant("in_edges_via_csr called without adjacency cache"))?;
        self.edge_lookup_via_csr(cache, edge_type, dst, EdgeDirection::Inverse)
            .await
    }

    /// Return every partner of `key` along `(edge_type, direction)` as a
    /// sorted `Vec<NodeId>` ascending by `NodeId` byte order, with the
    /// memtable overlay applied last-LSN-wins and tombstones removed.
    ///
    /// This is the input shape the leapfrog triejoin executor consumes
    /// (RFC-024): the WCOJ inner loop wraps the returned `Vec<NodeId>`
    /// in `SortedSliceIter` and intersects across the constraints
    /// incident to the current trie level. The CSR partner array is
    /// already sorted by construction (RFC-018); the memtable overlay
    /// can introduce out-of-order partners, so the merge stage funnels
    /// everything through a `BTreeMap` keyed on the raw partner bytes
    /// and drains it in ascending order. Properties are discarded; the
    /// caller only needs topology.
    ///
    /// Cost is `O(deg + memtable_edges_for_type)`. Production memtables
    /// flush at a configurable threshold so the second term is
    /// bounded; the first term comes for free from
    /// `EdgeAdjacency::lookup`.
    #[instrument(skip(self), fields(edge_type = edge_type, key = %key, direction = ?direction))]
    pub async fn sorted_partners(
        &self,
        edge_type: &str,
        key: NodeId,
        direction: EdgeDirection,
    ) -> Result<Vec<NodeId>> {
        namidb_core::profile_scope!("Snapshot::sorted_partners");
        let key_bytes = *key.as_bytes();
        // Partner bytes -> (lsn, is_upsert).
        let mut latest: BTreeMap<[u8; 16], (u64, bool)> = BTreeMap::new();

        // Committed memtable then the staged overlay (RFC-026 edge RYOW)
        // first; the SST/CSR path below shadows whatever they contributed
        // only when its LSN is strictly higher.
        for (mk, entry) in self.edge_mem_entries() {
            let MemKey::Edge {
                edge_type: et,
                src: s,
                dst: d,
            } = mk
            else {
                continue;
            };
            if et != edge_type {
                continue;
            }
            let (my_key_id, partner_id) = match direction {
                EdgeDirection::Forward => (*s.as_bytes(), *d.as_bytes()),
                EdgeDirection::Inverse => (*d.as_bytes(), *s.as_bytes()),
            };
            if my_key_id != key_bytes {
                continue;
            }
            let is_upsert = matches!(entry.op, MemOp::Upsert(_));
            match latest.get(&partner_id) {
                Some((existing_lsn, _)) if *existing_lsn >= entry.lsn => {}
                _ => {
                    latest.insert(partner_id, (entry.lsn, is_upsert));
                }
            }
        }

        // CSR if available + enabled, otherwise SST fallback. Both paths
        // emit (partner, lsn, is_upsert) triples into the same map.
        if adjacency_enabled() {
            if let Some(cache) = self.adjacency_cache.clone() {
                self.merge_sorted_partners_csr(cache, edge_type, key, direction, &mut latest)
                    .await?;
            } else {
                self.merge_sorted_partners_sst(edge_type, key, direction, &mut latest)
                    .await?;
            }
        } else {
            self.merge_sorted_partners_sst(edge_type, key, direction, &mut latest)
                .await?;
        }

        // BTreeMap drains ascending by key. Drop tombstones; rehydrate
        // the bytes back into a NodeId.
        let partners = latest
            .into_iter()
            .filter_map(|(partner_bytes, (_lsn, is_upsert))| {
                if is_upsert {
                    Some(NodeId::from_uuid(Uuid::from_bytes(partner_bytes)))
                } else {
                    None
                }
            })
            .collect();
        Ok(partners)
    }

    async fn merge_sorted_partners_csr(
        &self,
        cache: Arc<AdjacencyCache>,
        edge_type: &str,
        key: NodeId,
        direction: EdgeDirection,
        latest: &mut BTreeMap<[u8; 16], (u64, bool)>,
    ) -> Result<()> {
        let manifest_version = self.manifest.manifest.version;
        let cache_key = AdjacencyKey::new(manifest_version, edge_type, direction);
        let adj: Arc<EdgeAdjacency> = {
            let manifest = self.manifest.clone();
            let store = self.store.clone();
            let paths = self.paths.clone();
            let sst_cache = self.cache.clone();
            let edge_type_owned = edge_type.to_string();
            cache
                .get_or_build(cache_key, || async move {
                    build_adjacency(
                        &manifest,
                        store.as_ref(),
                        &paths,
                        sst_cache.as_ref(),
                        &edge_type_owned,
                        direction,
                    )
                    .await
                })
                .await?
        };
        if let Some(slice) = adj.lookup(key) {
            for i in 0..slice.partners.len() {
                let partner_id = *slice.partners[i].as_bytes();
                let lsn = slice.lsns[i];
                let is_upsert = !slice.tombstones[i];
                match latest.get(&partner_id) {
                    Some((existing_lsn, _)) if *existing_lsn >= lsn => {}
                    _ => {
                        latest.insert(partner_id, (lsn, is_upsert));
                    }
                }
            }
        }
        Ok(())
    }

    async fn merge_sorted_partners_sst(
        &self,
        edge_type: &str,
        key: NodeId,
        direction: EdgeDirection,
        latest: &mut BTreeMap<[u8; 16], (u64, bool)>,
    ) -> Result<()> {
        let key_bytes = *key.as_bytes();
        let want_kind = match direction {
            EdgeDirection::Forward => SstKind::EdgesFwd,
            EdgeDirection::Inverse => SstKind::EdgesInv,
        };
        let candidates = self.manifest.index.lookup_candidates(
            &self.manifest.manifest.ssts,
            want_kind,
            edge_type,
            &key_bytes,
        );
        for idx in candidates {
            let desc = &self.manifest.manifest.ssts[idx];
            if !self.bloom_admits(desc, &key_bytes).await? {
                continue;
            }
            let absolute = format!("{}/{}", self.paths.namespace_prefix().as_ref(), desc.path);
            let reader = self.fetch_edge_reader(&absolute).await?;
            let Some(lookup) = reader.lookup(&key_bytes)? else {
                continue;
            };
            for i in 0..lookup.partners.len() {
                let partner_id = lookup.partners[i];
                let lsn = lookup.lsns[i];
                let is_upsert = !lookup.tombstones[i];
                match latest.get(&partner_id) {
                    Some((existing_lsn, _)) if *existing_lsn >= lsn => {}
                    _ => {
                        latest.insert(partner_id, (lsn, is_upsert));
                    }
                }
            }
        }
        Ok(())
    }

    async fn edge_lookup(
        &self,
        edge_type: &str,
        key: NodeId,
        direction: EdgeDirection,
    ) -> Result<EdgeListView> {
        // CSR route (RFC-018): when an `AdjacencyCache` is attached AND
        // `NAMIDB_ADJACENCY` is not "0", resolve via the in-RAM CSR. Slim
        // CSR means EdgeView.properties is empty for SST-sourced edges —
        // memtable edges still carry their full property maps.
        // plan-aware routing in `namidb_query::exec::walker` calls
        // `edge_lookup_via_sst` directly when the query reads `r` or
        // `r.prop` downstream, so the caveat is invisible to query
        // callers; storage-level consumers that need full properties
        // should call `edge_lookup_via_sst` directly.
        if adjacency_enabled() {
            if let Some(cache) = self.adjacency_cache.clone() {
                return self
                    .edge_lookup_via_csr(cache, edge_type, key, direction)
                    .await;
            }
        }
        self.edge_lookup_via_sst(edge_type, key, direction).await
    }

    async fn edge_lookup_via_sst(
        &self,
        edge_type: &str,
        key: NodeId,
        direction: EdgeDirection,
    ) -> Result<EdgeListView> {
        namidb_core::profile_scope!("Snapshot::edge_lookup_via_sst");
        // Legacy path; behaviour unchanged from
        // the earlier NodeView cache iteration.
        let key_bytes = *key.as_bytes();
        let want_kind = match direction {
            EdgeDirection::Forward => SstKind::EdgesFwd,
            EdgeDirection::Inverse => SstKind::EdgesInv,
        };

        // Per-partner last-write-wins state.
        let mut latest: BTreeMap<[u8; 16], (u64, Option<EdgeView>)> = BTreeMap::new();

        // 1. Memtable, then the writer's staged overlay (RFC-026 edge RYOW).
        for (mk, entry) in self.edge_mem_entries() {
            let MemKey::Edge {
                edge_type: et,
                src: s,
                dst: d,
            } = mk
            else {
                continue;
            };
            if et != edge_type {
                continue;
            }
            let (my_key_id, partner_id) = match direction {
                EdgeDirection::Forward => (*s.as_bytes(), *d.as_bytes()),
                EdgeDirection::Inverse => (*d.as_bytes(), *s.as_bytes()),
            };
            if my_key_id != key_bytes {
                continue;
            }

            let view = match &entry.op {
                MemOp::Tombstone => None,
                MemOp::Upsert(payload) => {
                    let rec = EdgeWriteRecord::decode(payload)?;
                    Some(EdgeView {
                        edge_type: edge_type.to_string(),
                        src: *s,
                        dst: *d,
                        properties: rec.properties,
                        lsn: entry.lsn,
                    })
                }
            };
            update_partner_winner(&mut latest, partner_id, entry.lsn, view);
        }

        // 2. SST candidates — pruned by the manifest index, same as the
        // node lookup path.
        let candidates = self.manifest.index.lookup_candidates(
            &self.manifest.manifest.ssts,
            want_kind,
            edge_type,
            &key_bytes,
        );
        for idx in candidates {
            let desc = &self.manifest.manifest.ssts[idx];
            if !self.bloom_admits(desc, &key_bytes).await? {
                continue;
            }
            let absolute = format!("{}/{}", self.paths.namespace_prefix().as_ref(), desc.path);
            // S18.B: cache `EdgeSstReader` per SST path. `open` is
            // `O(edge_count)` because it precomputes the cumulative-
            // edges prefix sum; caching makes warm `edge_lookup_via_sst`
            // O(deg) instead of O(edge_count) per call.
            let reader = self.fetch_edge_reader(&absolute).await?;
            let Some(lookup) = reader.lookup(&key_bytes)? else {
                continue;
            };
            // S17.3: cache the decoded property streams per SST path.
            // The streams are immutable for the SST's lifetime so the
            // first call decodes O(edge_count) and every subsequent
            // call on this snapshot — or any other snapshot sharing
            // the cache — is a single map probe.
            let streams = self.fetch_edge_streams(&absolute, edge_type, &reader)?;
            for (i, partner_id) in lookup.partners.iter().enumerate() {
                let lsn = lookup.lsns[i];
                let tomb = lookup.tombstones[i];
                let view = if tomb {
                    None
                } else {
                    let partner_node = NodeId::from_uuid(Uuid::from_bytes(*partner_id));
                    let (src_id, dst_id) = match direction {
                        EdgeDirection::Forward => (key, partner_node),
                        EdgeDirection::Inverse => (partner_node, key),
                    };
                    let absolute_idx = lookup.edge_offset + i;
                    let properties = decode_edge_properties(
                        streams.overflow.as_ref().and_then(|v| v.get(absolute_idx)),
                        &streams.declared,
                        absolute_idx,
                    )?;
                    Some(EdgeView {
                        edge_type: edge_type.to_string(),
                        src: src_id,
                        dst: dst_id,
                        properties,
                        lsn,
                    })
                };
                update_partner_winner(&mut latest, *partner_id, lsn, view);
            }
        }

        // 3. Materialise: drop tombstones, sort by partner identifier.
        let mut edges: Vec<EdgeView> = latest.into_values().filter_map(|(_, view)| view).collect();
        edges.sort_by(|a, b| match direction {
            EdgeDirection::Forward => a.dst.cmp(&b.dst),
            EdgeDirection::Inverse => a.src.cmp(&b.src),
        });
        Ok(EdgeListView { edges })
    }

    /// CSR-based edge lookup (RFC-018). Resolves through the
    /// shared [`AdjacencyCache`]; builds the per-`(edge_type, direction)`
    /// adjacency on cache miss and reuses it for the lifetime of the
    /// matching `manifest_version`. SST-sourced edges come back with
    /// **empty `properties`** — memtable-sourced edges retain their full
    /// property maps (decoded from the upsert payload). The slim
    /// trade-off is documented in RFC-018 §4 + the `EdgeView` doc.
    async fn edge_lookup_via_csr(
        &self,
        cache: Arc<AdjacencyCache>,
        edge_type: &str,
        key: NodeId,
        direction: EdgeDirection,
    ) -> Result<EdgeListView> {
        namidb_core::profile_scope!("Snapshot::edge_lookup_via_csr");
        let key_bytes = *key.as_bytes();

        // 1. Resolve (build on miss) the CSR for this (manifest_version,
        // edge_type, direction).
        let manifest_version = self.manifest.manifest.version;
        let cache_key = AdjacencyKey::new(manifest_version, edge_type, direction);
        let adj: Arc<EdgeAdjacency> = {
            namidb_core::profile_scope!("AdjacencyCache::get_or_build");
            // Capture state needed by the build closure so the future
            // doesn't borrow `self` (the closure must be `'static`-ish
            // friendly across the await point).
            let manifest = self.manifest.clone();
            let store = self.store.clone();
            let paths = self.paths.clone();
            let sst_cache = self.cache.clone();
            let edge_type_owned = edge_type.to_string();
            cache
                .get_or_build(cache_key, || async move {
                    build_adjacency(
                        &manifest,
                        store.as_ref(),
                        &paths,
                        sst_cache.as_ref(),
                        &edge_type_owned,
                        direction,
                    )
                    .await
                })
                .await?
        };

        // 2. Per-partner last-write-wins state. Memtable + CSR feed it.
        let mut latest: BTreeMap<[u8; 16], (u64, Option<EdgeView>)> = BTreeMap::new();

        // 2a. Memtable sweep, then the writer's staged overlay (RFC-026
        // edge RYOW): same shape as the SST path. A staged or committed
        // tombstone here shadows a CSR upsert of equal-or-lower LSN.
        for (mk, entry) in self.edge_mem_entries() {
            let MemKey::Edge {
                edge_type: et,
                src: s,
                dst: d,
            } = mk
            else {
                continue;
            };
            if et != edge_type {
                continue;
            }
            let (my_key_id, partner_id) = match direction {
                EdgeDirection::Forward => (*s.as_bytes(), *d.as_bytes()),
                EdgeDirection::Inverse => (*d.as_bytes(), *s.as_bytes()),
            };
            if my_key_id != key_bytes {
                continue;
            }
            let view = match &entry.op {
                MemOp::Tombstone => None,
                MemOp::Upsert(payload) => {
                    let rec = EdgeWriteRecord::decode(payload)?;
                    Some(EdgeView {
                        edge_type: edge_type.to_string(),
                        src: *s,
                        dst: *d,
                        properties: rec.properties,
                        lsn: entry.lsn,
                    })
                }
            };
            update_partner_winner(&mut latest, partner_id, entry.lsn, view);
        }

        // 2b. CSR slice — O(log K + deg). Each row in the slice is a
        // candidate edge. Properties are NOT decoded; callers wishing
        // full property maps must use the SST path (flag off).
        if let Some(slice) = adj.lookup(key) {
            for i in 0..slice.partners.len() {
                let partner_id = *slice.partners[i].as_bytes();
                let lsn = slice.lsns[i];
                let tomb = slice.tombstones[i];
                let view = if tomb {
                    None
                } else {
                    let partner_node = slice.partners[i];
                    let (src_id, dst_id) = match direction {
                        EdgeDirection::Forward => (key, partner_node),
                        EdgeDirection::Inverse => (partner_node, key),
                    };
                    Some(EdgeView {
                        edge_type: edge_type.to_string(),
                        src: src_id,
                        dst: dst_id,
                        properties: BTreeMap::new(),
                        lsn,
                    })
                };
                update_partner_winner(&mut latest, partner_id, lsn, view);
            }
        }

        // 3. Materialise + sort by partner (same shape as SST path).
        let mut edges: Vec<EdgeView> = latest.into_values().filter_map(|(_, view)| view).collect();
        edges.sort_by(|a, b| match direction {
            EdgeDirection::Forward => a.dst.cmp(&b.dst),
            EdgeDirection::Inverse => a.src.cmp(&b.src),
        });
        Ok(EdgeListView { edges })
    }

    async fn get_sst_body(&self, desc: &SstDescriptor) -> Result<Bytes> {
        // Cooperative cancellation (query timeout): every read path fetches a
        // candidate SST body through here once per SST, so this one probe
        // bounds the "scan touches many SSTs" case across all of them. The
        // per-row decode loops add their own strided probes for a single huge
        // SST. A no-op when no deadline is in scope (writes, compaction).
        crate::cancel::check()?;
        let absolute = format!("{}/{}", self.paths.namespace_prefix().as_ref(), desc.path);
        self.fetch_bytes(&absolute).await
    }

    /// RFC-030 (`vector-index`): approximate top-k over the `VectorGraph`
    /// SST(s) registered for `index_name`. Returns `(NodeId, similarity)`
    /// best-first (higher similarity = closer). Unions across every in-scope
    /// VectorGraph SST and re-ranks — there is normally exactly one per index
    /// for an id-primary namespace, but a partial rebuild can briefly leave
    /// two. `ef` is the search beam width (≥ `k`).
    /// Decoded `.vg` index for `desc`, via the process-wide [`SstCache`]:
    /// decoding deserialises every stored vector plus the whole adjacency and
    /// clones the vectors into the navigation space, so paying it once per SST
    /// (instead of per query, and per widening round) is the difference between
    /// `O(k)`-ish and `O(index size)` KNN latency. `Ok(None)` = undecodable
    /// (legacy/corrupt) body — the caller skips it and the flat scan covers.
    #[cfg(feature = "vector-index")]
    async fn fetch_vector_index(
        &self,
        desc: &crate::manifest::SstDescriptor,
    ) -> Result<Option<Arc<crate::sst::vector::VectorGraphIndex>>> {
        use crate::sst::vector::VectorGraphIndex;
        let absolute = format!("{}/{}", self.paths.namespace_prefix().as_ref(), desc.path);
        if let Some(cache) = self.cache.as_ref() {
            if let Some(idx) = cache.get_vector_index(&absolute) {
                return Ok(Some(idx));
            }
        }
        let body = self.get_sst_body(desc).await?;
        let Ok(idx) = VectorGraphIndex::decode(&body) else {
            return Ok(None);
        };
        let idx = Arc::new(idx);
        if let Some(cache) = self.cache.as_ref() {
            cache.insert_vector_index(absolute, idx.clone());
        }
        Ok(Some(idx))
    }

    /// Decoded `.ft` index for `desc`, via the process-wide [`SstCache`] (same
    /// once-per-SST story as [`Self::fetch_vector_index`]). Unlike `.vg`, a
    /// corrupt text body is a hard error (the historical behaviour).
    #[cfg(feature = "text-index")]
    async fn fetch_text_index(
        &self,
        desc: &crate::manifest::SstDescriptor,
    ) -> Result<Arc<crate::sst::text::TextIndex>> {
        use crate::sst::text::TextIndex;
        let absolute = format!("{}/{}", self.paths.namespace_prefix().as_ref(), desc.path);
        if let Some(cache) = self.cache.as_ref() {
            if let Some(idx) = cache.get_text_index(&absolute) {
                return Ok(idx);
            }
        }
        let body = self.get_sst_body(desc).await?;
        let idx = Arc::new(TextIndex::decode(&body)?);
        if let Some(cache) = self.cache.as_ref() {
            cache.insert_text_index(absolute, idx.clone());
        }
        Ok(idx)
    }

    #[cfg(feature = "vector-index")]
    pub async fn vector_search(
        &self,
        index_name: &str,
        query: &[f32],
        k: usize,
        ef: usize,
    ) -> Result<Vec<(NodeId, f32)>> {
        let mut all: Vec<(NodeId, f32)> = Vec::new();
        // Score orientation is metric-dependent: cosine/dot are higher-is-closer,
        // euclidean is lower-is-closer. All `.vg` SSTs for one index share a
        // metric, so the last decoded one's orientation is authoritative.
        let mut higher_is_better = true;
        for desc in &self.manifest.manifest.ssts {
            if desc.kind != SstKind::VectorGraph || desc.scope != index_name {
                continue;
            }
            // A legacy (v1) or corrupt body fails to decode; skip it so the read
            // falls back to the flat scan rather than erroring the whole query.
            let Some(idx) = self.fetch_vector_index(desc).await? else {
                continue;
            };
            higher_is_better = idx.higher_is_better();
            all.extend(
                idx.search(query, k, ef)
                    .into_iter()
                    .map(|(id, s)| (NodeId(Uuid::from_bytes(id)), s)),
            );
        }
        // Best-first by the metric's orientation.
        if higher_is_better {
            all.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
        } else {
            all.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));
        }
        all.truncate(k);
        Ok(all)
    }

    /// `true` if any persisted `Nodes` SST carries writes **newer** than the
    /// `index_name` index's SST(s) — the index has been outrun by node data it
    /// has not absorbed, so the caller must fall back to the exact flat scan.
    ///
    /// An SST-backed index (`.vg` / `.ft`) is rebuilt only on an **authoritative**
    /// (deepest-level) compaction whose merged rows span the full label corpus;
    /// the rebuild stamps the index descriptor's `max_lsn` with that corpus's
    /// high-water LSN. A later flush (→ L0) or partial merge (→ L1+) produces a
    /// `Nodes` SST with a higher `max_lsn`. Comparing LSNs — not levels — is what
    /// makes this correct regardless of which level the newer data landed at,
    /// closing the partial-compaction truncation window: a shallow merge that
    /// rewrites a subset to L1 no longer hides those rows from the freshness
    /// check just because L0 is now empty. The lockstep `Nodes` SST written by
    /// the same authoritative merge shares the index's `max_lsn` exactly, so it is
    /// never (`>`) flagged. `kind` is the index SST kind
    /// (`VectorGraph` / `TextIndex`).
    #[cfg(any(feature = "vector-index", feature = "text-index"))]
    pub fn index_outrun_by_nodes(&self, index_name: &str, kind: SstKind) -> bool {
        let idx_lsn = self
            .manifest
            .manifest
            .ssts
            .iter()
            .filter(|d| d.kind == kind && d.scope == index_name)
            .map(|d| d.max_lsn)
            .max();
        let Some(idx_lsn) = idx_lsn else {
            // No index SST for this name yet. If any persisted `Nodes` SST
            // exists, its flushed rows are unabsorbed by the (nonexistent)
            // index AND are not in the memtable fresh-delta the caller merges,
            // so the index path would silently miss them — report "outrun" to
            // force the exact flat scan. (A vector KNN with a just-registered
            // index but no authoritative `.vg` compaction yet was returning
            // memtable-only top-k, dropping every flushed neighbour.) When
            // there is no Nodes SST either, the whole corpus is still in the
            // memtable, which the caller's fresh-delta merge fully covers, so
            // the index path stays correct.
            return self
                .manifest
                .manifest
                .ssts
                .iter()
                .any(|d| d.kind == SstKind::Nodes);
        };
        self.manifest
            .manifest
            .ssts
            .iter()
            .any(|d| d.kind == SstKind::Nodes && d.max_lsn > idx_lsn)
    }

    /// (`vector-index`) Fresh node deltas (committed memtable + staged overlay)
    /// for a `(label, property)` vector index: every node id touched since the
    /// last compaction the `.vg` has not absorbed. `Some(vec)` is a live
    /// embedding to merge into the KNN; `None` suppresses the id — it is
    /// tombstoned, no longer carries `label`, or dropped its embedding — so a
    /// stale index hit for it is excluded. Highest-LSN entry per id wins (staged
    /// overlay LSNs outrank committed). The executor unions this with the index
    /// result so the ANN answer stays freshness-equivalent to the flat scan
    /// (RFC-030); a node written but not yet compacted is found immediately.
    #[cfg(feature = "vector-index")]
    pub fn vector_fresh_delta(
        &self,
        label: &str,
        property: &str,
    ) -> Result<Vec<(NodeId, Option<Vec<f32>>)>> {
        let dict = &self.manifest.manifest.label_dict;
        // (node_id) → (winning lsn, Some(embedding) | None=suppress).
        let mut latest: BTreeMap<NodeId, (u64, Option<Vec<f32>>)> = BTreeMap::new();
        for (mk, entry) in self.node_entries() {
            let MemKey::Node { id } = mk else {
                continue;
            };
            let val: Option<Vec<f32>> = match &entry.op {
                MemOp::Tombstone => None,
                MemOp::Upsert(payload) => {
                    let rec = NodeWriteRecord::decode(payload)?;
                    if record_carries_label(&rec, label, dict) {
                        embedding_as_f32(rec.properties.get(property))
                    } else {
                        // A memtable version that no longer carries `label`
                        // supersedes any indexed row for this id → suppress.
                        None
                    }
                }
            };
            match latest.get(id) {
                Some((existing_lsn, _)) if *existing_lsn >= entry.lsn => {}
                _ => {
                    latest.insert(*id, (entry.lsn, val));
                }
            }
        }
        Ok(latest.into_iter().map(|(id, (_, v))| (id, v)).collect())
    }

    /// (`text-index`): full BM25 top-k over the `TextIndex` SST(s) registered for
    /// `index_name`, **only when the index is authoritative for `label`**.
    ///
    /// Returns `Ok(None)` — meaning "fall back to the flat scan" — when the index
    /// would not see the full corpus: no built `TextIndex` SST yet, or there is
    /// un-compacted node data for `label` (committed/staged memtable entries, or
    /// an L0 `Nodes` SST not yet folded into the index by compaction). This keeps
    /// the index path freshness-equivalent to the flat scan: a write is visible
    /// to `search.bm25` immediately, regardless of whether an index exists. The
    /// index only serves once compaction has caught the corpus up.
    ///
    /// `Ok(Some(hits))` is the BM25 result: `(NodeId, score)` best-first with a
    /// node-id tie-break, unioned across every in-scope TextIndex SST (normally
    /// one per index; a partial rebuild can briefly leave two). `k = None`
    /// returns every match.
    #[cfg(feature = "text-index")]
    pub async fn text_search(
        &self,
        index_name: &str,
        label: &str,
        query_terms: &[String],
        k: Option<usize>,
    ) -> Result<Option<Vec<(NodeId, f64)>>> {

        // Authoritative only if a TextIndex SST exists for this index...
        let has_index_sst = self
            .manifest
            .manifest
            .ssts
            .iter()
            .any(|d| d.kind == SstKind::TextIndex && d.scope == index_name);
        if !has_index_sst {
            return Ok(None);
        }
        // ...and there is no un-compacted node delta the index has not absorbed:
        // a persisted `Nodes` SST newer than the index (flushed/partially-merged
        // but not yet folded in by an authoritative compaction — the LSN
        // comparison catches both, see `index_outrun_by_nodes`)...
        if self.index_outrun_by_nodes(index_name, SstKind::TextIndex) {
            return Ok(None);
        }
        // ...and no memtable/overlay entry that touches the indexed corpus. The
        // check is label-scoped: an unflushed write to an UNRELATED label must
        // not disable the index (it used to — under live mixed traffic every
        // `search.bm25` became an `O(corpus)` flat scan). BM25 scores depend on
        // corpus-wide stats (N, avgdl, df), so exact flat-scan parity allows
        // serving only when the delta provably does not touch the corpus:
        //   - an upsert CARRYING `label` is a live document delta → flat scan;
        //   - a tombstone, or an upsert NOT carrying `label` (a possible
        //     relabel), affects the corpus only if its id is an indexed
        //     document — probed against the decoded index below.
        let dict = &self.manifest.manifest.label_dict;
        let mut dirty: Vec<[u8; 16]> = Vec::new();
        for (mk, entry) in self.node_entries() {
            let MemKey::Node { id } = mk else {
                continue;
            };
            match &entry.op {
                MemOp::Tombstone => dirty.push(*id.0.as_bytes()),
                MemOp::Upsert(payload) => {
                    let rec = NodeWriteRecord::decode(payload)?;
                    if record_carries_label(&rec, label, dict) {
                        return Ok(None);
                    }
                    dirty.push(*id.0.as_bytes());
                }
            }
        }

        let mut all: Vec<(NodeId, f64)> = Vec::new();
        for desc in &self.manifest.manifest.ssts {
            if desc.kind != SstKind::TextIndex || desc.scope != index_name {
                continue;
            }
            let idx = self.fetch_text_index(desc).await?;
            // A dirty id that IS an indexed document means a stale doc (delete/
            // relabel) the index would still serve, and its removal also shifts
            // the corpus stats — only the flat scan is exact then.
            if dirty.iter().any(|id| idx.contains_doc(id)) {
                return Ok(None);
            }
            all.extend(
                idx.search(query_terms, k)
                    .into_iter()
                    .map(|(id, s)| (NodeId(Uuid::from_bytes(id)), s)),
            );
        }
        all.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.0.cmp(&b.0))
        });
        if let Some(k) = k {
            all.truncate(k);
        }
        Ok(Some(all))
    }

    /// Return the decoded edge property streams for the SST identified by
    /// `absolute`, hitting [`SstCache::get_edge_streams`] first and
    /// decoding via the freshly-opened `reader` on miss.
    ///
    /// — IC07 at SF1 profile showed that every
    /// `edge_lookup_via_sst` call decoded the SST's overflow stream
    /// (`O(edge_count)`) plus every declared property column. The bundle
    /// is immutable per SST path (UUIDv7-keyed, never overwritten), so
    /// a `HashMap<String, Arc<EdgeStreamBundle>>` keyed by absolute path
    /// covers every reader of that SST across the lifetime of the
    /// process.
    /// Return the [`EdgeSstReader`] for the SST identified by `absolute`,
    /// hitting [`SstCache::get_edge_reader`] first. Cache miss path opens
    /// a fresh reader (which precomputes `cumulative_edges`,
    /// `O(edge_count)`) and inserts it.: IC07 at
    /// SF10 surfaced that `EdgeSstReader::open` was the residual
    /// per-call cost not covered by the property stream cache.
    async fn fetch_edge_reader(&self, absolute: &str) -> Result<Arc<EdgeSstReader>> {
        namidb_core::profile_scope!("Snapshot::fetch_edge_reader");
        if let Some(cache) = self.cache.as_ref() {
            if let Some(reader) = cache.get_edge_reader(absolute) {
                namidb_core::profile_scope!("Snapshot::fetch_edge_reader.hit");
                return Ok(reader);
            }
        }
        namidb_core::profile_scope!("Snapshot::fetch_edge_reader.miss");
        let body = self.fetch_bytes(absolute).await?;
        let reader = Arc::new(EdgeSstReader::open(body)?);
        if let Some(cache) = self.cache.as_ref() {
            cache.insert_edge_reader(absolute.to_string(), reader.clone());
        }
        Ok(reader)
    }

    fn fetch_edge_streams(
        &self,
        absolute: &str,
        edge_type: &str,
        reader: &EdgeSstReader,
    ) -> Result<Arc<EdgeStreamBundle>> {
        namidb_core::profile_scope!("Snapshot::fetch_edge_streams");
        if let Some(cache) = self.cache.as_ref() {
            if let Some(bundle) = cache.get_edge_streams(absolute) {
                namidb_core::profile_scope!("Snapshot::fetch_edge_streams.hit");
                return Ok(bundle);
            }
        }
        namidb_core::profile_scope!("Snapshot::fetch_edge_streams.miss");
        let declared_property_names: Vec<String> = self
            .manifest
            .manifest
            .schema
            .edge_type(edge_type)
            .map(|def| def.properties.iter().map(|p| p.name.clone()).collect())
            .unwrap_or_default();
        let bundle = Arc::new(EdgeStreamBundle {
            overflow: reader.read_overflow_strings()?,
            declared: load_declared_streams(reader, &declared_property_names)?,
        });
        if let Some(cache) = self.cache.as_ref() {
            cache.insert_edge_streams(absolute.to_string(), bundle.clone());
        }
        Ok(bundle)
    }

    /// Returns `true` if the SST cannot be ruled out by its bloom side-car
    /// for `key`. SSTs without a side-car (small bodies under the omit
    /// threshold — see [`crate::sst::bloom::BLOOM_OMIT_THRESHOLD_BYTES`])
    /// always admit, falling back to the body GET that follows.
    async fn bloom_admits(&self, desc: &SstDescriptor, key: &[u8; 16]) -> Result<bool> {
        let Some(bloom_desc) = &desc.bloom else {
            return Ok(true);
        };
        let absolute = format!(
            "{}/{}",
            self.paths.namespace_prefix().as_ref(),
            bloom_desc.path
        );
        let body = self.fetch_bytes(&absolute).await?;
        let filter = BloomFilter::from_bytes(&bloom_desc.path, &body)?;
        Ok(filter.contains(key))
    }

    /// Cache-only check: returns `Some(body)` if the cache has it,
    /// `None` if not present or no cache attached. Used by
    /// `lookup_node` to decide between the sync (cache-hit) and async
    /// (cold ranged-read) paths.
    fn cache_get(&self, absolute: &str) -> Option<Bytes> {
        self.cache.as_ref().and_then(|c| c.get(absolute))
    }

    /// Cache-aware fetch by absolute path. On hit, returns the cached
    /// `Bytes` (a cheap `Arc::clone`). On miss, GETs the object store
    /// and inserts the bytes back into the cache so the next reader on
    /// the same SST or bloom side-car can avoid the round-trip.
    ///
    /// SST + bloom bodies are immutable per UUIDv7-keyed path, so the
    /// cache cannot ever return stale bytes — once an object is named,
    /// its content is final.
    async fn fetch_bytes(&self, absolute: &str) -> Result<Bytes> {
        if let Some(cache) = &self.cache {
            if let Some(hit) = cache.get(absolute) {
                return Ok(hit);
            }
        }
        let path = Path::from(absolute);
        let res = self.store.get(&path).await?;
        let body = res.bytes().await?;
        if let Some(cache) = &self.cache {
            cache.insert(absolute.to_string(), body.clone());
        }
        Ok(body)
    }
}

fn update_partner_winner(
    map: &mut BTreeMap<[u8; 16], (u64, Option<EdgeView>)>,
    partner: [u8; 16],
    lsn: u64,
    view: Option<EdgeView>,
) {
    match map.get(&partner) {
        Some((existing_lsn, _)) if *existing_lsn >= lsn => {}
        _ => {
            map.insert(partner, (lsn, view));
        }
    }
}

/// Load every declared property stream from a freshly-opened edge SST
/// reader. Streams for names the SST doesn't carry (legacy bodies,
/// all-null columns elided by the writer) are silently skipped — the
/// caller treats them as absent at every index.
fn load_declared_streams(
    reader: &EdgeSstReader,
    declared_property_names: &[String],
) -> Result<Vec<(String, Vec<Option<String>>)>> {
    let mut out: Vec<(String, Vec<Option<String>>)> =
        Vec::with_capacity(declared_property_names.len());
    for name in declared_property_names {
        if let Some(stream) = reader.read_declared_property_strings(name)? {
            out.push((name.clone(), stream));
        }
    }
    Ok(out)
}

/// Decode the JSON property bag for one edge slot. `None` slots — and a
/// missing entry — produce an empty map, mirroring the writer's "no
/// overflow → null in IPC stream" convention.
fn decode_overflow_props(slot: Option<&Option<String>>) -> Result<BTreeMap<String, Value>> {
    let Some(Some(json)) = slot else {
        return Ok(BTreeMap::new());
    };
    let parsed: BTreeMap<String, Value> = serde_json::from_str(json)?;
    Ok(parsed)
}

/// Materialise the property map for one edge slot by combining the
/// legacy `__overflow_json` (a JSON object) with each declared property
/// stream (RFC-002 §3.2.7, one JSON-encoded `Value` per property).
/// Declared values take precedence on key collision — the writer is
/// expected to route every declared key into its named stream, so a
/// collision in the wire data is anomalous and most likely the result
/// of a legacy SST. Order: overflow first, then declared (so the
/// declared values shadow).
fn decode_edge_properties(
    overflow_slot: Option<&Option<String>>,
    declared_streams: &[(String, Vec<Option<String>>)],
    idx: usize,
) -> Result<BTreeMap<String, Value>> {
    let mut out = decode_overflow_props(overflow_slot)?;
    for (name, stream) in declared_streams {
        if let Some(Some(encoded)) = stream.get(idx) {
            let value: Value = serde_json::from_str(encoded).map_err(|e| {
                Error::invariant(format!(
                    "edge declared property '{name}' decode at idx {idx}: {e}"
                ))
            })?;
            out.insert(name.clone(), value);
        }
    }
    Ok(out)
}

fn update_node_winner(
    map: &mut BTreeMap<NodeId, (u64, Option<NodeView>)>,
    id: NodeId,
    lsn: u64,
    view: Option<NodeView>,
) {
    match map.get(&id) {
        Some((existing_lsn, _)) if *existing_lsn >= lsn => {}
        _ => {
            map.insert(id, (lsn, view));
        }
    }
}

fn update_edge_winner(
    map: &mut BTreeMap<(NodeId, NodeId), (u64, Option<EdgeView>)>,
    key: (NodeId, NodeId),
    lsn: u64,
    view: Option<EdgeView>,
) {
    match map.get(&key) {
        Some((existing_lsn, _)) if *existing_lsn >= lsn => {}
        _ => {
            map.insert(key, (lsn, view));
        }
    }
}

/// `update_edge_winner` for the count path: tracks `(lsn, is_live)` only,
/// no `EdgeView`. Same last-writer-wins semantics (`existing >= lsn`
/// keeps the existing winner) so a count agrees with `scan_edge_type`.
fn update_edge_count_winner(
    map: &mut BTreeMap<(NodeId, NodeId), (u64, bool)>,
    key: (NodeId, NodeId),
    lsn: u64,
    live: bool,
) {
    match map.get(&key) {
        Some((existing_lsn, _)) if *existing_lsn >= lsn => {}
        _ => {
            map.insert(key, (lsn, live));
        }
    }
}

fn node_view_from_payload(
    id: NodeId,
    lsn: u64,
    payload: &Bytes,
    dict: &LabelDictionary,
    scope_fallback: &str,
) -> Result<NodeView> {
    let rec = NodeWriteRecord::decode(payload)?;
    let labels = labels_from_ids(&rec.labels, dict, scope_fallback);
    Ok(NodeView {
        id,
        labels,
        properties: rec.properties,
        lsn,
        schema_version: rec.schema_version,
    })
}

/// Whether a decoded record carries `label`, resolving the name via `dict`.
/// Used to label-filter memtable rows now that the label left the key.
fn record_carries_label(rec: &NodeWriteRecord, label: &str, dict: &LabelDictionary) -> bool {
    dict.id(label)
        .map(|lid| rec.labels.contains(&lid.get()))
        .unwrap_or(false)
}

/// Whether the node SST at `idx` can contain a LIVE row carrying `label`.
///
/// Scopes the sidecar-completeness checks in `lookup_node_by_property` /
/// `lookup_nodes_by_property`: an SST that provably holds no row of `label`
/// must not disable a sidecar fast path it could never contribute to.
/// Conservative — answers `true` unless the manifest proves absence:
///
/// - Legacy per-label SSTs name their single label as `scope`; a different
///   label's SST cannot contain this one's rows.
/// - id-primary SSTs (`scope == ""`) carry per-label posting counts in their
///   label-index descriptor (live rows only): a label with no postings — or
///   one the namespace dictionary never interned — has no live row in the
///   SST. Pre-counts manifests (`per_label_counts` empty) and pre-label-index
///   SSTs stay `true`.
///
/// Excluding tombstone-only coverage is safe: sidecar winners are re-confirmed
/// through `lookup_node`, which resolves last-LSN-wins across EVERY SST.
fn node_sst_can_contain_label(manifest: &Manifest, idx: usize, label: &str) -> bool {
    let desc = &manifest.ssts[idx];
    if !desc.scope.is_empty() {
        return desc.scope == label;
    }
    if let Some(li) = &desc.label_index {
        if !li.per_label_counts.is_empty() {
            return match manifest.label_dict.id(label) {
                Some(lid) => li
                    .per_label_counts
                    .iter()
                    .any(|(id, count)| *id == lid.get() && *count > 0),
                None => false,
            };
        }
    }
    true
}

/// Decode an embedding property value to `Vec<f32>` for a vector delta scan:
/// a stored `Vec` directly, an int8-quantized `VecI8` dequantized via
/// `code * scale` (matching the build hook and `coerce_vector`). Any other
/// value (or absence) yields `None` — the node has no usable embedding.
#[cfg(feature = "vector-index")]
fn embedding_as_f32(v: Option<&Value>) -> Option<Vec<f32>> {
    match v {
        Some(Value::Vec(v)) => Some(v.clone()),
        Some(Value::VecI8 { codes, scale }) => {
            Some(codes.iter().map(|&c| c as f32 * *scale).collect())
        }
        _ => None,
    }
}

/// Resolve interned `LabelId`s to label names via `dict`. Falls back to a
/// singleton `{scope_fallback}` when there are no ids (a legacy single-label
/// record/SST), or to an empty set when the fallback is empty.
fn labels_from_ids(ids: &[u32], dict: &LabelDictionary, scope_fallback: &str) -> BTreeSet<String> {
    let mut set: BTreeSet<String> = ids
        .iter()
        .filter_map(|&lid| dict.name(LabelId::new(lid)).map(String::from))
        .collect();
    if set.is_empty() && !scope_fallback.is_empty() {
        set.insert(scope_fallback.to_string());
    }
    set
}

/// Decode a node SST row's label set from the `__labels` column, resolving
/// `LabelId`s via `dict`. Legacy SSTs lack the column; their single label is
/// the SST scope, supplied as `scope_fallback`.
fn decode_node_labels(
    batch: &RecordBatch,
    row: usize,
    dict: &LabelDictionary,
    scope_fallback: &str,
) -> BTreeSet<String> {
    let Some(list) = batch
        .column_by_name(COL_LABELS)
        .and_then(|c| c.as_any().downcast_ref::<ListArray>())
    else {
        return labels_from_ids(&[], dict, scope_fallback);
    };
    if list.is_null(row) {
        return labels_from_ids(&[], dict, scope_fallback);
    }
    let values = list.value(row);
    let ids: Vec<u32> = match values.as_any().downcast_ref::<UInt32Array>() {
        Some(a) => (0..a.len())
            .filter(|&i| !a.is_null(i))
            .map(|i| a.value(i))
            .collect(),
        None => Vec::new(),
    };
    labels_from_ids(&ids, dict, scope_fallback)
}

/// 3VL evaluation of a conjunctive predicate list against a single
/// `NodeView`. `true` ⇔ every predicate evaluates to `true`. Missing
/// properties evaluate as NULL; ordered predicates against NULL drop.
fn node_view_matches_predicates(view: &NodeView, predicates: &[ScanPredicate]) -> bool {
    for p in predicates {
        let val = view.properties.get(p.column());
        if !eval_against_value(p, val) {
            return false;
        }
    }
    true
}

/// Scan a node SST body for the row with `node_id == target.as_bytes()`.
/// Returns `Some((lsn, Some(view)))` for an upsert, `Some((lsn, None))`
/// for a tombstone, and `None` if the SST does not contain the key.
fn find_node_row(
    reader: &NodeSstReader,
    label_def: &LabelDef,
    target: NodeId,
    dict: &LabelDictionary,
    scope_fallback: &str,
) -> Result<Option<(u64, Option<NodeView>)>> {
    let target_bytes = *target.as_bytes();
    let batches = reader.targeted_scan(&target_bytes)?;
    find_node_row_in_batches(&batches, label_def, target, dict, scope_fallback)
}

/// Backend-agnostic row search over already-decoded record batches.
/// Shared between the sync (cache-hit) and async (cold ranged-read)
/// paths so behavior stays bit-identical regardless of where the
/// batches came from.
fn find_node_row_in_batches(
    batches: &[RecordBatch],
    label_def: &LabelDef,
    target: NodeId,
    dict: &LabelDictionary,
    scope_fallback: &str,
) -> Result<Option<(u64, Option<NodeView>)>> {
    let target_bytes = *target.as_bytes();
    for batch in batches {
        let id_col = batch
            .column_by_name(COL_NODE_ID)
            .and_then(|c| c.as_any().downcast_ref::<FixedSizeBinaryArray>())
            .ok_or_else(|| Error::invariant("node_id column missing"))?;
        for row in 0..batch.num_rows() {
            let row_id: [u8; 16] = id_col
                .value(row)
                .try_into()
                .map_err(|_| Error::invariant("node_id row length != 16"))?;
            if row_id != target_bytes {
                continue;
            }

            let tomb_col = batch
                .column_by_name(COL_TOMBSTONE)
                .and_then(|c| c.as_any().downcast_ref::<BooleanArray>())
                .ok_or_else(|| Error::invariant("tombstone column missing"))?;
            let lsn_col = batch
                .column_by_name(COL_LSN)
                .and_then(|c| c.as_any().downcast_ref::<UInt64Array>())
                .ok_or_else(|| Error::invariant("lsn column missing"))?;
            let tomb = tomb_col.value(row);
            let lsn = lsn_col.value(row);
            if tomb {
                return Ok(Some((lsn, None)));
            }

            let mut properties: BTreeMap<String, Value> = BTreeMap::new();
            for p in &label_def.properties {
                let col_name = prop_column_name(p);
                let col = batch
                    .column_by_name(&col_name)
                    .ok_or_else(|| Error::invariant(format!("missing column {col_name}")))?;
                if let Some(v) = arrow_value_to_value(col.as_ref(), row, &p.data_type)? {
                    properties.insert(p.name.clone(), v);
                }
            }
            let ovf_col = batch
                .column_by_name(OVERFLOW_JSON)
                .and_then(|c| c.as_any().downcast_ref::<StringArray>())
                .ok_or_else(|| Error::invariant("__overflow_json column missing"))?;
            if !ovf_col.is_null(row) {
                let json_str = ovf_col.value(row);
                let extra: BTreeMap<String, Value> = serde_json::from_str(json_str)?;
                properties.extend(extra);
            }
            let sv_col = batch
                .column_by_name(SCHEMA_VERSION)
                .and_then(|c| c.as_any().downcast_ref::<UInt64Array>())
                .ok_or_else(|| Error::invariant("__schema_version column missing"))?;
            let schema_version = sv_col.value(row);

            return Ok(Some((
                lsn,
                Some(NodeView {
                    id: target,
                    labels: decode_node_labels(batch, row, dict, scope_fallback),
                    properties,
                    lsn,
                    schema_version,
                }),
            )));
        }
    }
    Ok(None)
}

/// Batched analogue of `find_node_row_in_batches`: walk `batches` ONCE,
/// emit a `NodeView` (or tombstone marker) for every row whose `node_id`
/// is in `pending`, and last-LSN-merge into `winners`. The hot inner
/// loop short-circuits on rows whose id isn't in the pending set, so
/// the per-row cost on irrelevant rows is one `HashSet::contains`.
fn batch_harvest_node_rows(
    batches: &[RecordBatch],
    label_def: &LabelDef,
    dict: &LabelDictionary,
    scope_fallback: &str,
    pending: &std::collections::HashSet<[u8; 16]>,
    winners: &mut HashMap<[u8; 16], (u64, Option<NodeView>)>,
) -> Result<()> {
    for batch in batches {
        let id_col = batch
            .column_by_name(COL_NODE_ID)
            .and_then(|c| c.as_any().downcast_ref::<FixedSizeBinaryArray>())
            .ok_or_else(|| Error::invariant("node_id column missing"))?;
        let tomb_col = batch
            .column_by_name(COL_TOMBSTONE)
            .and_then(|c| c.as_any().downcast_ref::<BooleanArray>())
            .ok_or_else(|| Error::invariant("tombstone column missing"))?;
        let lsn_col = batch
            .column_by_name(COL_LSN)
            .and_then(|c| c.as_any().downcast_ref::<UInt64Array>())
            .ok_or_else(|| Error::invariant("lsn column missing"))?;
        let ovf_col = batch
            .column_by_name(OVERFLOW_JSON)
            .and_then(|c| c.as_any().downcast_ref::<StringArray>())
            .ok_or_else(|| Error::invariant("__overflow_json column missing"))?;
        let sv_col = batch
            .column_by_name(SCHEMA_VERSION)
            .and_then(|c| c.as_any().downcast_ref::<UInt64Array>())
            .ok_or_else(|| Error::invariant("__schema_version column missing"))?;
        for row in 0..batch.num_rows() {
            let row_id: [u8; 16] = id_col
                .value(row)
                .try_into()
                .map_err(|_| Error::invariant("node_id row length != 16"))?;
            if !pending.contains(&row_id) {
                continue;
            }
            let tomb = tomb_col.value(row);
            let lsn = lsn_col.value(row);
            // Last-LSN-wins early skip: if the existing winner already
            // beats us, decoding the row's properties would be wasted.
            if let Some((existing_lsn, _)) = winners.get(&row_id) {
                if *existing_lsn >= lsn {
                    continue;
                }
            }
            let view = if tomb {
                None
            } else {
                let mut properties: BTreeMap<String, Value> = BTreeMap::new();
                for p in &label_def.properties {
                    let col_name = prop_column_name(p);
                    let col = batch
                        .column_by_name(&col_name)
                        .ok_or_else(|| Error::invariant(format!("missing column {col_name}")))?;
                    if let Some(v) = arrow_value_to_value(col.as_ref(), row, &p.data_type)? {
                        properties.insert(p.name.clone(), v);
                    }
                }
                if !ovf_col.is_null(row) {
                    let json_str = ovf_col.value(row);
                    let extra: BTreeMap<String, Value> = serde_json::from_str(json_str)?;
                    properties.extend(extra);
                }
                let schema_version = sv_col.value(row);
                let id = NodeId::from_uuid(Uuid::from_bytes(row_id));
                Some(NodeView {
                    id,
                    labels: decode_node_labels(batch, row, dict, scope_fallback),
                    properties,
                    lsn,
                    schema_version,
                })
            };
            winners.insert(row_id, (lsn, view));
        }
    }
    Ok(())
}

pub(crate) fn arrow_value_to_value(
    array: &dyn Array,
    row: usize,
    data_type: &DataType,
) -> Result<Option<Value>> {
    if array.is_null(row) {
        return Ok(None);
    }
    let value = match data_type {
        DataType::Bool => {
            let a = array
                .as_any()
                .downcast_ref::<BooleanArray>()
                .ok_or_else(|| Error::invariant("expected BooleanArray"))?;
            Value::Bool(a.value(row))
        }
        DataType::Int32 => {
            let a = array
                .as_any()
                .downcast_ref::<Int32Array>()
                .ok_or_else(|| Error::invariant("expected Int32Array"))?;
            Value::I64(a.value(row) as i64)
        }
        DataType::Int64 => {
            let a = array
                .as_any()
                .downcast_ref::<Int64Array>()
                .ok_or_else(|| Error::invariant("expected Int64Array"))?;
            Value::I64(a.value(row))
        }
        DataType::Float32 => {
            let a = array
                .as_any()
                .downcast_ref::<Float32Array>()
                .ok_or_else(|| Error::invariant("expected Float32Array"))?;
            Value::F64(a.value(row) as f64)
        }
        DataType::Float64 => {
            let a = array
                .as_any()
                .downcast_ref::<Float64Array>()
                .ok_or_else(|| Error::invariant("expected Float64Array"))?;
            Value::F64(a.value(row))
        }
        DataType::Utf8 => {
            let a = array
                .as_any()
                .downcast_ref::<StringArray>()
                .ok_or_else(|| Error::invariant("expected StringArray"))?;
            Value::Str(a.value(row).to_string())
        }
        DataType::LargeUtf8 => {
            let a = array
                .as_any()
                .downcast_ref::<LargeStringArray>()
                .ok_or_else(|| Error::invariant("expected LargeStringArray"))?;
            Value::Str(a.value(row).to_string())
        }
        DataType::Binary => {
            let a = array
                .as_any()
                .downcast_ref::<BinaryArray>()
                .ok_or_else(|| Error::invariant("expected BinaryArray"))?;
            Value::Bytes(a.value(row).to_vec())
        }
        DataType::Date32 => {
            let a = array
                .as_any()
                .downcast_ref::<Date32Array>()
                .ok_or_else(|| Error::invariant("expected Date32Array"))?;
            Value::Date(a.value(row))
        }
        DataType::TimestampMicrosUtc => {
            let a = array
                .as_any()
                .downcast_ref::<TimestampMicrosecondArray>()
                .ok_or_else(|| Error::invariant("expected TimestampMicrosecondArray"))?;
            Value::DateTime(a.value(row))
        }
        DataType::FloatVector { dim } => {
            let a = array
                .as_any()
                .downcast_ref::<FixedSizeListArray>()
                .ok_or_else(|| Error::invariant("expected FixedSizeListArray"))?;
            let inner_ref = a.value(row);
            let f = inner_ref
                .as_any()
                .downcast_ref::<Float32Array>()
                .ok_or_else(|| Error::invariant("expected inner Float32Array"))?;
            if f.len() != *dim as usize {
                return Err(Error::invariant(format!(
                    "FloatVector dim mismatch: expected {dim}, got {}",
                    f.len()
                )));
            }
            Value::Vec(f.values().to_vec())
        }
        DataType::Int8Vector { dim } => {
            let a = array
                .as_any()
                .downcast_ref::<FixedSizeBinaryArray>()
                .ok_or_else(|| Error::invariant("expected FixedSizeBinaryArray"))?;
            let bytes = a.value(row);
            let want = 4 + *dim as usize;
            if bytes.len() != want {
                return Err(Error::invariant(format!(
                    "Int8Vector byte width mismatch: expected {want}, got {}",
                    bytes.len()
                )));
            }
            let scale = f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
            let codes = bytes[4..].iter().map(|&b| b as i8).collect();
            Value::VecI8 { codes, scale }
        }
        DataType::Json => {
            let a = array
                .as_any()
                .downcast_ref::<StringArray>()
                .ok_or_else(|| Error::invariant("expected StringArray for Json"))?;
            serde_json::from_str(a.value(row))?
        }
    };
    Ok(Some(value))
}

/// Owned, lifetime-free read snapshot of a namespace.
///
/// `OwnedSnapshot` carries an `Arc<MemtableSnapshot>` (a frozen copy
/// of the writer's memtable at commit time) plus the manifest, object
/// store and the cross-snapshot caches. Multiple concurrent readers
/// share one `OwnedSnapshot` via `Arc`, so reads run in parallel
/// across the tokio runtime without taking the writer mutex. See
/// RFC-021.
///
/// Each read call materialises a short-lived [`Snapshot`] borrowed
/// from the owned state. The per-query scratch caches (intra-snapshot
/// node lookups, decoded RecordBatch reuse) live on that temporary
/// borrowed snapshot and drop at the end of the query.
pub struct OwnedSnapshot {
    pub(crate) manifest: LoadedManifest,
    pub(crate) memtable: Arc<MemtableSnapshot>,
    pub(crate) store: Arc<dyn ObjectStore>,
    pub(crate) paths: NamespacePaths,
    pub(crate) cache: Option<SstCache>,
    pub(crate) ranged_mode: RangedMode,
    pub(crate) ranged_threshold_bytes: u64,
    pub(crate) adjacency_cache: Option<Arc<AdjacencyCache>>,
    pub(crate) shared_node_cache: Option<Arc<NodeViewCache>>,
    pub(crate) property_index_cache: Option<Arc<crate::property_index::PropertyIndexCache>>,
}

impl std::fmt::Debug for OwnedSnapshot {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OwnedSnapshot")
            .field("manifest_version", &self.manifest.manifest.version)
            .field("memtable_entries", &self.memtable.len())
            .field("sst_count", &self.manifest.manifest.ssts.len())
            .finish()
    }
}

impl OwnedSnapshot {
    pub fn manifest(&self) -> &LoadedManifest {
        &self.manifest
    }

    pub fn manifest_version(&self) -> u64 {
        self.manifest.manifest.version
    }

    /// Build a short-lived [`Snapshot`] borrowed from this owned state.
    /// Hand it to the query executor; the lifetime is bounded by
    /// `&self`, so the owned snapshot must outlive every read it
    /// drives.
    pub fn borrow(&self) -> Snapshot<'_> {
        let mut snap = Snapshot::new(
            self.manifest.clone(),
            &self.memtable,
            self.store.clone(),
            self.paths.clone(),
        );
        if let Some(c) = &self.cache {
            snap = snap.with_cache(c.clone());
        }
        snap = snap.with_ranged_threshold_bytes(self.ranged_threshold_bytes);
        if let RangedMode::Force(b) = self.ranged_mode {
            snap = snap.with_ranged_reads(b);
        }
        if let Some(c) = &self.adjacency_cache {
            snap = snap.with_adjacency_cache(c.clone());
        }
        if let Some(c) = &self.shared_node_cache {
            snap = snap.with_shared_node_cache(c.clone());
        }
        if let Some(c) = &self.property_index_cache {
            snap = snap.with_property_index_cache(c.clone());
        }
        snap
    }
}

/// Tracks the manifest versions live readers are pinned to (RFC-027).
///
/// Each [`SnapshotCell::load`] registers the version of the snapshot it
/// hands out; the returned [`PinnedSnapshot`] deregisters it on drop. The
/// compactor's sweep and version GC read the resulting retention horizon —
/// the oldest version any reader could still need — so they never reclaim
/// an object a live reader can still reach. `min_live()` is monotonically
/// non-decreasing while a given set of readers runs (readers only ever
/// register the current version, which increases), so a sweep that samples
/// it gets a safe lower bound.
#[derive(Debug, Default)]
struct SnapshotRegistry {
    /// `manifest version -> number of live readers pinned to it`.
    live: std::sync::Mutex<BTreeMap<u64, usize>>,
}

impl SnapshotRegistry {
    fn acquire(&self, version: u64) {
        *self
            .live
            .lock()
            .expect("snapshot registry poisoned")
            .entry(version)
            .or_insert(0) += 1;
    }

    fn release(&self, version: u64) {
        let mut g = self.live.lock().expect("snapshot registry poisoned");
        if let Some(count) = g.get_mut(&version) {
            *count -= 1;
            if *count == 0 {
                g.remove(&version);
            }
        }
    }

    /// Oldest manifest version any live reader is pinned to, or `None` when
    /// no reader is active.
    fn min_live(&self) -> Option<u64> {
        self.live
            .lock()
            .expect("snapshot registry poisoned")
            .keys()
            .next()
            .copied()
    }
}

/// An [`OwnedSnapshot`] handed to a reader with its manifest version
/// registered as live for the duration (RFC-027). Deref-transparent to
/// `OwnedSnapshot`, so call sites use `.borrow()` / `.manifest()` as
/// before. Dropping it releases the reader's hold on the retention
/// horizon, letting the sweep / GC reclaim that version once no reader
/// needs it.
pub struct PinnedSnapshot {
    snap: Arc<OwnedSnapshot>,
    registry: Arc<SnapshotRegistry>,
    version: u64,
}

impl std::ops::Deref for PinnedSnapshot {
    type Target = OwnedSnapshot;
    fn deref(&self) -> &OwnedSnapshot {
        &self.snap
    }
}

impl Drop for PinnedSnapshot {
    fn drop(&mut self) {
        self.registry.release(self.version);
    }
}

impl PinnedSnapshot {
    /// Clone the shared `Arc` for callers that need to store or republish
    /// it. The clone is NOT separately registered; the horizon hold lives
    /// with this `PinnedSnapshot`.
    pub fn arc(&self) -> Arc<OwnedSnapshot> {
        Arc::clone(&self.snap)
    }
}

impl std::fmt::Debug for PinnedSnapshot {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PinnedSnapshot")
            .field("version", &self.version)
            .finish()
    }
}

/// Atomic publisher cell for the currently-active [`OwnedSnapshot`].
///
/// `SnapshotCell` is the lock-light handoff between the writer (which
/// rebuilds the snapshot after every successful commit / flush) and the
/// readers (which load the Arc and drop the writer mutex entirely).
///
/// The current implementation guards the inner `Arc` with a
/// `std::sync::Mutex` for clarity. The critical section is exactly
/// one pointer load plus an `Arc` strong-count bump (tens of
/// nanoseconds). A lock-free swap (via `arc-swap`) is the natural
/// follow-up once a flamegraph shows the mutex matters.
#[derive(Debug)]
pub struct SnapshotCell {
    inner: std::sync::Mutex<Arc<OwnedSnapshot>>,
    registry: Arc<SnapshotRegistry>,
}

impl SnapshotCell {
    pub fn new(snap: Arc<OwnedSnapshot>) -> Self {
        Self {
            inner: std::sync::Mutex::new(snap),
            registry: Arc::new(SnapshotRegistry::default()),
        }
    }

    /// Pick up the current snapshot, registering its version as live until
    /// the returned [`PinnedSnapshot`] drops. Cheap: one mutex acquire plus
    /// an `Arc::clone` and a counter bump. The version is registered while
    /// the cell lock is held, so it is selected and recorded atomically and
    /// the retention horizon never excludes a version a reader is about to
    /// read.
    pub fn load(&self) -> PinnedSnapshot {
        let guard = self.inner.lock().expect("snapshot cell poisoned");
        let snap = Arc::clone(&guard);
        let version = snap.manifest_version();
        self.registry.acquire(version);
        drop(guard);
        PinnedSnapshot {
            snap,
            registry: Arc::clone(&self.registry),
            version,
        }
    }

    /// Publish a new snapshot. The previous Arc is dropped once
    /// every reader holding it lets go.
    pub fn store(&self, snap: Arc<OwnedSnapshot>) {
        *self.inner.lock().expect("snapshot cell poisoned") = snap;
    }

    /// Manifest version currently published. Cheap diagnostic for
    /// observability — equivalent to `self.load().manifest_version()`
    /// without the Arc clone path.
    pub fn manifest_version(&self) -> u64 {
        self.inner
            .lock()
            .expect("snapshot cell poisoned")
            .manifest_version()
    }

    /// Retention horizon (RFC-027): the oldest manifest version any live
    /// reader is pinned to, or the currently-published version when no
    /// reader is active. The sweep / GC may reclaim any object that no
    /// manifest version at or above this references — by construction a
    /// reader pinned at version `V` keeps the horizon at or below `V`, so
    /// nothing `V` needs is collected.
    pub fn retention_horizon(&self) -> u64 {
        let current = self.manifest_version();
        self.registry
            .min_live()
            .map(|m| m.min(current))
            .unwrap_or(current)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use namidb_core::{EdgeTypeDef, LabelDef, NamespaceId, PropertyDef, SchemaBuilder};
    use object_store::memory::InMemory;

    use super::*;
    use crate::adjacency::{adjacency_budget_bytes, AdjacencyCache};
    use crate::fence::WriterFence;
    use crate::flush::{flush, NodeWriteRecord};
    use crate::manifest::ManifestStore;
    use crate::memtable::Memtable;
    use crate::wal::WalSegment;

    fn make_store() -> Arc<dyn ObjectStore> {
        Arc::new(InMemory::new())
    }

    fn make_paths(name: &str) -> NamespacePaths {
        NamespacePaths::new("tenants", NamespaceId::new(name).unwrap())
    }

    fn person_label() -> LabelDef {
        LabelDef {
            name: "Person".into(),
            properties: vec![
                PropertyDef::new("name", DataType::Utf8, false).unwrap(),
                PropertyDef::new("age", DataType::Int32, true).unwrap(),
            ],
        }
    }

    fn knows_edge() -> EdgeTypeDef {
        EdgeTypeDef {
            name: "KNOWS".into(),
            src_label: "Person".into(),
            dst_label: "Person".into(),
            properties: vec![],
        }
    }

    fn sorted_node_id(b: u8) -> NodeId {
        let mut bytes = [0u8; 16];
        bytes[15] = b;
        NodeId::from_uuid(Uuid::from_bytes(bytes))
    }

    fn node_payload(name: &str, age: Option<i32>) -> Bytes {
        let mut props: BTreeMap<String, Value> = BTreeMap::new();
        props.insert("name".into(), Value::Str(name.into()));
        if let Some(a) = age {
            props.insert("age".into(), Value::I64(a as i64));
        }
        NodeWriteRecord {
            properties: props,
            schema_version: 1,
            // These tests use the single label "Person", which interns to
            // LabelId(0) on a fresh dict. Carry it on-row so the id-primary
            // read path resolves the node to "Person".
            labels: vec![0],
        }
        .encode()
        .unwrap()
    }

    /// Like `node_payload` but carries an explicit interned `LabelId` on-row.
    /// Used by the multi-label endpoint-inference tests where the two endpoint
    /// nodes need distinct labels (e.g. "Person" -> 0, "Company" -> 1).
    fn labeled_node_payload(name: &str, label_id: u32) -> Bytes {
        let mut props: BTreeMap<String, Value> = BTreeMap::new();
        props.insert("name".into(), Value::Str(name.into()));
        NodeWriteRecord {
            properties: props,
            schema_version: 1,
            labels: vec![label_id],
        }
        .encode()
        .unwrap()
    }

    fn edge_payload() -> Bytes {
        EdgeWriteRecord {
            properties: BTreeMap::new(),
            schema_version: 1,
        }
        .encode()
        .unwrap()
    }

    #[tokio::test]
    async fn lookup_node_finds_row_in_sst_after_flush() {
        let store = make_store();
        let paths = make_paths("read-flush");
        let ms = ManifestStore::new(store.clone(), paths.clone());
        let mut base = ms.bootstrap(Uuid::now_v7()).await.unwrap();
        // Seed the dict so the on-row LabelId(0) resolves to "Person".
        base.manifest.label_dict.intern("Person");
        let fence = WriterFence::new(base.manifest.epoch);
        let schema = SchemaBuilder::new().label(person_label()).unwrap().build();

        let alice = sorted_node_id(1);
        let mut mt = Memtable::new();
        mt.apply(
            MemKey::Node { id: alice },
            10,
            MemOp::Upsert(node_payload("Alice", Some(30))),
        );
        let frozen = mt.freeze();
        let outcome = flush(&ms, &fence, &base, &frozen, schema.clone())
            .await
            .unwrap();

        // After flush the persisted state lives in SSTs; the live memtable
        // is empty.
        let empty = Memtable::new();
        let empty_view = empty.snapshot_view();
        let snap = Snapshot::new(
            outcome.committed.clone(),
            &empty_view,
            store.clone(),
            paths.clone(),
        );
        let view = snap.lookup_node("Person", alice).await.unwrap().unwrap();
        assert_eq!(view.id, alice);
        assert_eq!(view.lsn, 10);
        assert_eq!(view.schema_version, 1);
        assert_eq!(
            view.properties.get("name"),
            Some(&Value::Str("Alice".into()))
        );
        assert_eq!(view.properties.get("age"), Some(&Value::I64(30)));
    }

    #[tokio::test]
    async fn scan_aborts_on_a_passed_deadline() {
        // Cooperative cancellation (query timeout): a scan run under an
        // already-passed deadline aborts inside storage with `Error::Timeout`,
        // at the per-SST body fetch, rather than decoding to completion first.
        // No deadline in scope leaves the scan unguarded.
        let store = make_store();
        let paths = make_paths("read-deadline");
        let ms = ManifestStore::new(store.clone(), paths.clone());
        let mut base = ms.bootstrap(Uuid::now_v7()).await.unwrap();
        base.manifest.label_dict.intern("Person");
        let fence = WriterFence::new(base.manifest.epoch);
        let schema = SchemaBuilder::new().label(person_label()).unwrap().build();

        // Flush a node so the scan reaches an SST; the deadline probe lives in
        // the per-SST body fetch and the per-batch decode loop.
        let mut mt = Memtable::new();
        mt.apply(
            MemKey::Node {
                id: sorted_node_id(1),
            },
            10,
            MemOp::Upsert(node_payload("Alice", Some(30))),
        );
        let committed = flush(&ms, &fence, &base, &mt.freeze(), schema)
            .await
            .unwrap()
            .committed;
        let empty = Memtable::new();
        let empty_view = empty.snapshot_view();
        let snap = Snapshot::new(committed, &empty_view, store, paths);

        // Unguarded: the scan succeeds.
        assert_eq!(snap.scan_label("Person").await.unwrap().len(), 1);

        // Under a deadline already in the past, the same scan aborts.
        let past = std::time::Instant::now() - std::time::Duration::from_secs(1);
        let result = crate::cancel::with_deadline(Some(past), snap.scan_label("Person")).await;
        assert!(
            matches!(result, Err(Error::Timeout)),
            "expected Error::Timeout, got {result:?}"
        );
    }

    // ── secondary equality index (non-unique `indexed` property) ──

    fn indexed_city_label() -> LabelDef {
        LabelDef {
            name: "Person".into(),
            properties: vec![
                PropertyDef::new("name", DataType::Utf8, false).unwrap(),
                PropertyDef::new("city", DataType::Utf8, true)
                    .unwrap()
                    .with_indexed(true),
            ],
        }
    }

    fn city_payload(name: &str, city: &str) -> Bytes {
        let mut props: BTreeMap<String, Value> = BTreeMap::new();
        props.insert("name".into(), Value::Str(name.into()));
        props.insert("city".into(), Value::Str(city.into()));
        NodeWriteRecord {
            properties: props,
            schema_version: 1,
            // Single label "Person" -> LabelId(0) on a fresh dict.
            labels: vec![0],
        }
        .encode()
        .unwrap()
    }

    async fn flush_batch(
        ms: &ManifestStore,
        fence: &WriterFence,
        base: &LoadedManifest,
        schema: &namidb_core::Schema,
        rows: Vec<(NodeId, u64, MemOp)>,
    ) -> LoadedManifest {
        let mut mt = Memtable::new();
        for (id, lsn, op) in rows {
            mt.apply(MemKey::Node { id }, lsn, op);
        }
        let frozen = mt.freeze();
        flush(ms, fence, base, &frozen, schema.clone())
            .await
            .unwrap()
            .committed
    }

    /// Resolve `city == value` against `committed` (empty live memtable) and
    /// return the matched names, sorted.
    async fn lookup_cities(
        committed: &LoadedManifest,
        store: Arc<dyn ObjectStore>,
        paths: NamespacePaths,
        city: &str,
    ) -> Vec<String> {
        let empty = Memtable::new();
        let view = empty.snapshot_view();
        let snap = Snapshot::new(committed.clone(), &view, store, paths);
        let mut names: Vec<String> = snap
            .lookup_nodes_by_property("Person", "city", city)
            .await
            .unwrap()
            .into_iter()
            .filter_map(|v| match v.properties.get("name") {
                Some(Value::Str(s)) => Some(s.clone()),
                _ => None,
            })
            .collect();
        names.sort();
        names
    }

    #[tokio::test]
    async fn equality_index_returns_all_matching_nodes() {
        let store = make_store();
        let paths = make_paths("eqidx-all");
        let ms = ManifestStore::new(store.clone(), paths.clone());
        let mut base = ms.bootstrap(Uuid::now_v7()).await.unwrap();
        // Seed the dict so the on-row LabelId(0) resolves to "Person".
        base.manifest.label_dict.intern("Person");
        let fence = WriterFence::new(base.manifest.epoch);
        let schema = SchemaBuilder::new()
            .label(indexed_city_label())
            .unwrap()
            .build();

        let committed = flush_batch(
            &ms,
            &fence,
            &base,
            &schema,
            vec![
                (
                    sorted_node_id(1),
                    10,
                    MemOp::Upsert(city_payload("Ann", "LA")),
                ),
                (
                    sorted_node_id(2),
                    11,
                    MemOp::Upsert(city_payload("Bob", "LA")),
                ),
                (
                    sorted_node_id(3),
                    12,
                    MemOp::Upsert(city_payload("Cy", "NYC")),
                ),
            ],
        )
        .await;

        assert_eq!(
            lookup_cities(&committed, store.clone(), paths.clone(), "LA").await,
            vec!["Ann".to_string(), "Bob".to_string()]
        );
        assert_eq!(
            lookup_cities(&committed, store.clone(), paths.clone(), "NYC").await,
            vec!["Cy".to_string()]
        );
        assert!(lookup_cities(&committed, store, paths, "SF")
            .await
            .is_empty());
    }

    #[tokio::test]
    async fn equality_index_drops_tombstoned_candidate() {
        let store = make_store();
        let paths = make_paths("eqidx-tomb");
        let ms = ManifestStore::new(store.clone(), paths.clone());
        let mut base = ms.bootstrap(Uuid::now_v7()).await.unwrap();
        // Seed the dict so the on-row LabelId(0) resolves to "Person".
        base.manifest.label_dict.intern("Person");
        let fence = WriterFence::new(base.manifest.epoch);
        let schema = SchemaBuilder::new()
            .label(indexed_city_label())
            .unwrap()
            .build();

        // Two LA nodes flushed into one SST (which carries the sidecar).
        let committed = flush_batch(
            &ms,
            &fence,
            &base,
            &schema,
            vec![
                (
                    sorted_node_id(1),
                    10,
                    MemOp::Upsert(city_payload("Ann", "LA")),
                ),
                (
                    sorted_node_id(2),
                    11,
                    MemOp::Upsert(city_payload("Bob", "LA")),
                ),
            ],
        )
        .await;

        // A live-memtable tombstone on Ann: the sidecar still lists her id,
        // but the confirmation via lookup_node sees the tombstone and drops
        // her.
        let mut mt = Memtable::new();
        mt.apply(
            MemKey::Node {
                id: sorted_node_id(1),
            },
            20,
            MemOp::Tombstone,
        );
        let view = mt.snapshot_view();
        let snap = Snapshot::new(committed, &view, store, paths);
        let rows = snap
            .lookup_nodes_by_property("Person", "city", "LA")
            .await
            .unwrap();
        assert_eq!(rows.len(), 1, "Ann was tombstoned");
        assert_eq!(
            rows[0].properties.get("name"),
            Some(&Value::Str("Bob".into()))
        );
    }

    #[tokio::test]
    async fn equality_index_drops_value_changed_candidate() {
        // The §4 correctness guard: a node whose indexed value changed must
        // not be returned under its stale value.
        let store = make_store();
        let paths = make_paths("eqidx-changed");
        let ms = ManifestStore::new(store.clone(), paths.clone());
        let mut base = ms.bootstrap(Uuid::now_v7()).await.unwrap();
        // Seed the dict so the on-row LabelId(0) resolves to "Person".
        base.manifest.label_dict.intern("Person");
        let fence = WriterFence::new(base.manifest.epoch);
        let schema = SchemaBuilder::new()
            .label(indexed_city_label())
            .unwrap()
            .build();

        // Flush X under "LA" (the sidecar captures X at "LA").
        let committed = flush_batch(
            &ms,
            &fence,
            &base,
            &schema,
            vec![(
                sorted_node_id(1),
                10,
                MemOp::Upsert(city_payload("X", "LA")),
            )],
        )
        .await;

        // X moves to "NYC" in the live memtable (newer lsn, not flushed).
        let mut mt = Memtable::new();
        mt.apply(
            MemKey::Node {
                id: sorted_node_id(1),
            },
            20,
            MemOp::Upsert(city_payload("X", "NYC")),
        );
        let view = mt.snapshot_view();
        let snap = Snapshot::new(committed, &view, store, paths);

        // A query for the stale value must NOT return X.
        let la = snap
            .lookup_nodes_by_property("Person", "city", "LA")
            .await
            .unwrap();
        assert!(la.is_empty(), "stale 'LA' must not return the moved node");
        // The current value does.
        let nyc = snap
            .lookup_nodes_by_property("Person", "city", "NYC")
            .await
            .unwrap();
        assert_eq!(nyc.len(), 1);
        assert_eq!(nyc[0].properties.get("name"), Some(&Value::Str("X".into())));
    }

    #[tokio::test]
    async fn equality_index_survives_compaction() {
        let store = make_store();
        let paths = make_paths("eqidx-compact");
        let ms = ManifestStore::new(store.clone(), paths.clone());
        let mut base = ms.bootstrap(Uuid::now_v7()).await.unwrap();
        // Seed the dict so the on-row LabelId(0) resolves to "Person".
        base.manifest.label_dict.intern("Person");
        let fence = WriterFence::new(base.manifest.epoch);
        let schema = SchemaBuilder::new()
            .label(indexed_city_label())
            .unwrap()
            .build();

        // Two separate flushes → two L0 Person SSTs, each with a partial
        // sidecar.
        let b1 = flush_batch(
            &ms,
            &fence,
            &base,
            &schema,
            vec![(
                sorted_node_id(1),
                10,
                MemOp::Upsert(city_payload("Ann", "LA")),
            )],
        )
        .await;
        let b2 = flush_batch(
            &ms,
            &fence,
            &b1,
            &schema,
            vec![(
                sorted_node_id(2),
                11,
                MemOp::Upsert(city_payload("Bob", "LA")),
            )],
        )
        .await;

        // Compact L0 → L1; the rebuilt L1 sidecar must serve the union.
        let outcome = crate::compact::compact_l0_to_l1(&ms, &fence, &b2, &schema)
            .await
            .unwrap();
        assert!(
            outcome.source_ssts_removed >= 2,
            "expected the two L0 Person SSTs to compact"
        );
        assert_eq!(
            lookup_cities(&outcome.committed, store, paths, "LA").await,
            vec!["Ann".to_string(), "Bob".to_string()]
        );
    }

    #[tokio::test]
    async fn lookup_node_falls_back_to_memtable_when_not_flushed() {
        let store = make_store();
        let paths = make_paths("read-mt");
        let ms = ManifestStore::new(store.clone(), paths.clone());
        let mut base = ms.bootstrap(Uuid::now_v7()).await.unwrap();
        // No flush here: the snapshot reads the live memtable. Seed the dict so
        // the record's on-row LabelId(0) resolves to "Person".
        base.manifest.label_dict.intern("Person");

        let alice = sorted_node_id(2);
        let mut mt = Memtable::new();
        mt.apply(
            MemKey::Node { id: alice },
            7,
            MemOp::Upsert(node_payload("Alice", Some(28))),
        );

        let mt_view = mt.snapshot_view();
        let snap = Snapshot::new(base.clone(), &mt_view, store, paths);
        let view = snap.lookup_node("Person", alice).await.unwrap().unwrap();
        assert_eq!(view.lsn, 7);
        assert_eq!(
            view.properties.get("name"),
            Some(&Value::Str("Alice".into()))
        );
    }

    #[tokio::test]
    async fn lookup_node_returns_none_for_missing_key() {
        let store = make_store();
        let paths = make_paths("read-none");
        let ms = ManifestStore::new(store.clone(), paths.clone());
        let base = ms.bootstrap(Uuid::now_v7()).await.unwrap();
        let mt = Memtable::new();
        let mt_view = mt.snapshot_view();
        let snap = Snapshot::new(base, &mt_view, store, paths);
        let res = snap
            .lookup_node("Person", sorted_node_id(99))
            .await
            .unwrap();
        assert!(res.is_none());
    }

    #[tokio::test]
    async fn memtable_tombstone_overrides_sst_upsert() {
        let store = make_store();
        let paths = make_paths("read-tomb");
        let ms = ManifestStore::new(store.clone(), paths.clone());
        let base = ms.bootstrap(Uuid::now_v7()).await.unwrap();
        let fence = WriterFence::new(base.manifest.epoch);
        let schema = SchemaBuilder::new().label(person_label()).unwrap().build();

        let alice = sorted_node_id(3);

        // Flush an upsert into an SST.
        let mut mt_flush = Memtable::new();
        mt_flush.apply(
            MemKey::Node { id: alice },
            10,
            MemOp::Upsert(node_payload("Alice", Some(30))),
        );
        let frozen = mt_flush.freeze();
        let outcome = flush(&ms, &fence, &base, &frozen, schema.clone())
            .await
            .unwrap();

        // Live memtable now carries a tombstone at LSN 15 (> SST's LSN 10).
        let mut live_mt = Memtable::new();
        live_mt.apply(MemKey::Node { id: alice }, 15, MemOp::Tombstone);

        let live_mt_view = live_mt.snapshot_view();
        let snap = Snapshot::new(outcome.committed.clone(), &live_mt_view, store, paths);
        let res = snap.lookup_node("Person", alice).await.unwrap();
        assert!(res.is_none(), "tombstone at higher LSN must win");
    }

    #[tokio::test]
    async fn out_and_in_edges_traverse_partner_ssts() {
        let store = make_store();
        let paths = make_paths("read-edges");
        let ms = ManifestStore::new(store.clone(), paths.clone());
        let base = ms.bootstrap(Uuid::now_v7()).await.unwrap();
        let fence = WriterFence::new(base.manifest.epoch);
        let schema = SchemaBuilder::new()
            .label(person_label())
            .unwrap()
            .edge_type(knows_edge())
            .unwrap()
            .build();

        let alice = sorted_node_id(1);
        let bob = sorted_node_id(2);
        let carol = sorted_node_id(3);

        let mut mt = Memtable::new();
        mt.apply(
            MemKey::Edge {
                edge_type: "KNOWS".into(),
                src: alice,
                dst: bob,
            },
            10,
            MemOp::Upsert(edge_payload()),
        );
        mt.apply(
            MemKey::Edge {
                edge_type: "KNOWS".into(),
                src: alice,
                dst: carol,
            },
            11,
            MemOp::Upsert(edge_payload()),
        );
        mt.apply(
            MemKey::Edge {
                edge_type: "KNOWS".into(),
                src: bob,
                dst: alice,
            },
            12,
            MemOp::Upsert(edge_payload()),
        );
        let frozen = mt.freeze();
        let outcome = flush(&ms, &fence, &base, &frozen, schema.clone())
            .await
            .unwrap();

        let empty = Memtable::new();
        let empty_view = empty.snapshot_view();
        let snap = Snapshot::new(
            outcome.committed.clone(),
            &empty_view,
            store.clone(),
            paths.clone(),
        );

        // Out-edges of alice: bob and carol.
        let out = snap.out_edges("KNOWS", alice).await.unwrap();
        assert_eq!(out.edges.len(), 2);
        let dsts: Vec<NodeId> = out.edges.iter().map(|e| e.dst).collect();
        assert!(dsts.contains(&bob));
        assert!(dsts.contains(&carol));

        // In-edges of alice: only bob.
        let inn = snap.in_edges("KNOWS", alice).await.unwrap();
        assert_eq!(inn.edges.len(), 1);
        assert_eq!(inn.edges[0].src, bob);
    }

    #[tokio::test]
    async fn out_edges_merges_memtable_and_sst() {
        let store = make_store();
        let paths = make_paths("read-edges-merge");
        let ms = ManifestStore::new(store.clone(), paths.clone());
        let base = ms.bootstrap(Uuid::now_v7()).await.unwrap();
        let fence = WriterFence::new(base.manifest.epoch);
        let schema = SchemaBuilder::new()
            .label(person_label())
            .unwrap()
            .edge_type(knows_edge())
            .unwrap()
            .build();

        let alice = sorted_node_id(1);
        let bob = sorted_node_id(2);
        let carol = sorted_node_id(3);

        // Flush an edge alice→bob at LSN 10.
        let mut mt_flush = Memtable::new();
        mt_flush.apply(
            MemKey::Edge {
                edge_type: "KNOWS".into(),
                src: alice,
                dst: bob,
            },
            10,
            MemOp::Upsert(edge_payload()),
        );
        let frozen = mt_flush.freeze();
        let outcome = flush(&ms, &fence, &base, &frozen, schema.clone())
            .await
            .unwrap();

        // Live memtable: alice→carol (new) and alice→bob (tombstone at LSN 20).
        let mut live = Memtable::new();
        live.apply(
            MemKey::Edge {
                edge_type: "KNOWS".into(),
                src: alice,
                dst: carol,
            },
            15,
            MemOp::Upsert(edge_payload()),
        );
        live.apply(
            MemKey::Edge {
                edge_type: "KNOWS".into(),
                src: alice,
                dst: bob,
            },
            20,
            MemOp::Tombstone,
        );

        let live_view = live.snapshot_view();
        let snap = Snapshot::new(outcome.committed.clone(), &live_view, store, paths);
        let out = snap.out_edges("KNOWS", alice).await.unwrap();
        // bob's edge tombstoned, only carol remains.
        assert_eq!(out.edges.len(), 1);
        assert_eq!(out.edges[0].dst, carol);
    }

    #[tokio::test]
    async fn edge_overlay_read_your_own_writes_sst_and_csr() {
        // RFC-026 edge overlay: a writer's staged-but-uncommitted edge is
        // visible through the overlay (a staged upsert appears, a staged
        // tombstone hides a committed edge) on BOTH edge read paths — the
        // legacy SST scan and the CSR adjacency — and through both the
        // partner list (out/in_edges) and the WCOJ topology
        // (sorted_partners). The overlay is built by hand here; the query
        // and Bolt suites cover the real `overlay_snapshot()` wiring.
        let store = make_store();
        let paths = make_paths("read-edge-overlay");
        let ms = ManifestStore::new(store.clone(), paths.clone());
        let base = ms.bootstrap(Uuid::now_v7()).await.unwrap();
        let fence = WriterFence::new(base.manifest.epoch);
        let schema = SchemaBuilder::new()
            .label(person_label())
            .unwrap()
            .edge_type(knows_edge())
            .unwrap()
            .build();

        let alice = sorted_node_id(1);
        let bob = sorted_node_id(2);
        let carol = sorted_node_id(3);
        let dave = sorted_node_id(4);

        // Commit (flush to SST) alice→bob (LSN 10) and alice→dave (LSN 11).
        let mut mt_flush = Memtable::new();
        for (dst, lsn) in [(bob, 10u64), (dave, 11)] {
            mt_flush.apply(
                MemKey::Edge {
                    edge_type: "KNOWS".into(),
                    src: alice,
                    dst,
                },
                lsn,
                MemOp::Upsert(edge_payload()),
            );
        }
        let frozen = mt_flush.freeze();
        let outcome = flush(&ms, &fence, &base, &frozen, schema.clone())
            .await
            .unwrap();

        // The committed (live) memtable is empty; the staged batch lives only
        // in the overlay: upsert alice→carol (LSN 30) and tombstone alice→bob
        // (LSN 31). Staged LSNs exceed every committed LSN, as the real
        // `overlay_snapshot` guarantees.
        let live = Memtable::new();
        let live_view = live.snapshot_view();

        let mut staged = Memtable::new();
        staged.apply(
            MemKey::Edge {
                edge_type: "KNOWS".into(),
                src: alice,
                dst: carol,
            },
            30,
            MemOp::Upsert(edge_payload()),
        );
        staged.apply(
            MemKey::Edge {
                edge_type: "KNOWS".into(),
                src: alice,
                dst: bob,
            },
            31,
            MemOp::Tombstone,
        );

        // Baseline (no overlay): the committed edges are alice→{bob, dave}.
        let plain = Snapshot::new(
            outcome.committed.clone(),
            &live_view,
            store.clone(),
            paths.clone(),
        );
        let base_out = plain.out_edges_via_sst("KNOWS", alice).await.unwrap();
        assert_eq!(
            base_out.edges.iter().map(|e| e.dst).collect::<Vec<_>>(),
            vec![bob, dave],
            "without the overlay only the committed edges are visible"
        );
        drop(plain);

        // With the overlay attached: bob is hidden by the staged tombstone,
        // carol appears from the staged upsert, dave stays committed. The
        // result must be identical on the SST path and the CSR path.
        for use_csr in [false, true] {
            let mut snap = Snapshot::new(
                outcome.committed.clone(),
                &live_view,
                store.clone(),
                paths.clone(),
            )
            .with_overlay(staged.snapshot_view());
            if use_csr {
                snap = snap
                    .with_adjacency_cache(Arc::new(AdjacencyCache::new(adjacency_budget_bytes())));
            }

            let out = if use_csr {
                snap.out_edges_via_csr("KNOWS", alice).await.unwrap()
            } else {
                snap.out_edges_via_sst("KNOWS", alice).await.unwrap()
            };
            assert_eq!(
                out.edges.iter().map(|e| e.dst).collect::<Vec<_>>(),
                vec![carol, dave],
                "out_edges (csr={use_csr}) must hide the tombstoned edge and surface the staged one"
            );

            let inc = if use_csr {
                snap.in_edges_via_csr("KNOWS", carol).await.unwrap()
            } else {
                snap.in_edges_via_sst("KNOWS", carol).await.unwrap()
            };
            assert_eq!(
                inc.edges.iter().map(|e| e.src).collect::<Vec<_>>(),
                vec![alice],
                "in_edges (csr={use_csr}) must see the staged edge in reverse"
            );

            let partners = snap
                .sorted_partners("KNOWS", alice, EdgeDirection::Forward)
                .await
                .unwrap();
            assert_eq!(
                partners,
                vec![carol, dave],
                "sorted_partners (csr={use_csr}) must reflect the staged upsert and tombstone"
            );
        }
    }

    #[tokio::test]
    async fn count_edge_type_matches_scan_after_memtable_sst_merge() {
        let store = make_store();
        let paths = make_paths("read-edges-count");
        let ms = ManifestStore::new(store.clone(), paths.clone());
        let base = ms.bootstrap(Uuid::now_v7()).await.unwrap();
        let fence = WriterFence::new(base.manifest.epoch);
        let schema = SchemaBuilder::new()
            .label(person_label())
            .unwrap()
            .edge_type(knows_edge())
            .unwrap()
            .build();

        let alice = sorted_node_id(1);
        let bob = sorted_node_id(2);
        let carol = sorted_node_id(3);
        let dave = sorted_node_id(4);

        // Flush two edges: alice→bob (LSN 10) and alice→dave (LSN 11).
        let mut mt_flush = Memtable::new();
        for (dst, lsn) in [(bob, 10u64), (dave, 11)] {
            mt_flush.apply(
                MemKey::Edge {
                    edge_type: "KNOWS".into(),
                    src: alice,
                    dst,
                },
                lsn,
                MemOp::Upsert(edge_payload()),
            );
        }
        let frozen = mt_flush.freeze();
        let outcome = flush(&ms, &fence, &base, &frozen, schema.clone())
            .await
            .unwrap();

        // Live memtable: add alice→carol (LSN 15), tombstone alice→bob (LSN 20).
        let mut live = Memtable::new();
        live.apply(
            MemKey::Edge {
                edge_type: "KNOWS".into(),
                src: alice,
                dst: carol,
            },
            15,
            MemOp::Upsert(edge_payload()),
        );
        live.apply(
            MemKey::Edge {
                edge_type: "KNOWS".into(),
                src: alice,
                dst: bob,
            },
            20,
            MemOp::Tombstone,
        );

        let live_view = live.snapshot_view();
        let snap = Snapshot::new(outcome.committed.clone(), &live_view, store, paths);

        // Live KNOWS edges after the merge: alice→dave (SST) + alice→carol
        // (memtable); alice→bob is tombstoned. So 2.
        let count = snap.count_edge_type("KNOWS").await.unwrap();
        assert_eq!(count, 2);
        // It must agree with the materialising scan, the source of truth.
        let scanned = snap.scan_edge_type("KNOWS").await.unwrap();
        assert_eq!(count, scanned.len() as u64);

        // An unknown edge type counts zero.
        assert_eq!(snap.count_edge_type("FOLLOWS").await.unwrap(), 0);
    }

    #[tokio::test]
    async fn sorted_partners_returns_csr_partners_ascending() {
        let store = make_store();
        let paths = make_paths("read-sp-csr");
        let ms = ManifestStore::new(store.clone(), paths.clone());
        let base = ms.bootstrap(Uuid::now_v7()).await.unwrap();
        let fence = WriterFence::new(base.manifest.epoch);
        let schema = SchemaBuilder::new()
            .label(person_label())
            .unwrap()
            .edge_type(knows_edge())
            .unwrap()
            .build();

        let alice = sorted_node_id(1);
        // Three partners chosen so the order of insertion is not the
        // order of NodeId byte ordering.
        let p_03 = sorted_node_id(3);
        let p_07 = sorted_node_id(7);
        let p_05 = sorted_node_id(5);

        let mut mt = Memtable::new();
        for (dst, lsn) in [(p_07, 10u64), (p_03, 11), (p_05, 12)] {
            mt.apply(
                MemKey::Edge {
                    edge_type: "KNOWS".into(),
                    src: alice,
                    dst,
                },
                lsn,
                MemOp::Upsert(edge_payload()),
            );
        }
        let frozen = mt.freeze();
        let outcome = flush(&ms, &fence, &base, &frozen, schema.clone())
            .await
            .unwrap();

        let empty = Memtable::new();
        let empty_view = empty.snapshot_view();
        let snap = Snapshot::new(outcome.committed.clone(), &empty_view, store, paths)
            .with_adjacency_cache(Arc::new(AdjacencyCache::new(adjacency_budget_bytes())));

        let partners = snap
            .sorted_partners("KNOWS", alice, EdgeDirection::Forward)
            .await
            .unwrap();
        assert_eq!(partners, vec![p_03, p_05, p_07]);
    }

    #[tokio::test]
    async fn sorted_partners_merges_memtable_upsert_into_csr() {
        let store = make_store();
        let paths = make_paths("read-sp-merge");
        let ms = ManifestStore::new(store.clone(), paths.clone());
        let base = ms.bootstrap(Uuid::now_v7()).await.unwrap();
        let fence = WriterFence::new(base.manifest.epoch);
        let schema = SchemaBuilder::new()
            .label(person_label())
            .unwrap()
            .edge_type(knows_edge())
            .unwrap()
            .build();

        let alice = sorted_node_id(1);
        let bob = sorted_node_id(2);
        let carol = sorted_node_id(3);
        let dave = sorted_node_id(4);

        // Flush alice -> bob, alice -> dave at LSN 10/11.
        let mut mt_flush = Memtable::new();
        mt_flush.apply(
            MemKey::Edge {
                edge_type: "KNOWS".into(),
                src: alice,
                dst: bob,
            },
            10,
            MemOp::Upsert(edge_payload()),
        );
        mt_flush.apply(
            MemKey::Edge {
                edge_type: "KNOWS".into(),
                src: alice,
                dst: dave,
            },
            11,
            MemOp::Upsert(edge_payload()),
        );
        let frozen = mt_flush.freeze();
        let outcome = flush(&ms, &fence, &base, &frozen, schema.clone())
            .await
            .unwrap();

        // Live memtable adds alice -> carol at LSN 20.
        let mut live = Memtable::new();
        live.apply(
            MemKey::Edge {
                edge_type: "KNOWS".into(),
                src: alice,
                dst: carol,
            },
            20,
            MemOp::Upsert(edge_payload()),
        );
        let live_view = live.snapshot_view();
        let snap = Snapshot::new(outcome.committed.clone(), &live_view, store, paths)
            .with_adjacency_cache(Arc::new(AdjacencyCache::new(adjacency_budget_bytes())));

        let partners = snap
            .sorted_partners("KNOWS", alice, EdgeDirection::Forward)
            .await
            .unwrap();
        assert_eq!(partners, vec![bob, carol, dave]);
    }

    #[tokio::test]
    async fn sorted_partners_drops_memtable_tombstone() {
        let store = make_store();
        let paths = make_paths("read-sp-tomb");
        let ms = ManifestStore::new(store.clone(), paths.clone());
        let base = ms.bootstrap(Uuid::now_v7()).await.unwrap();
        let fence = WriterFence::new(base.manifest.epoch);
        let schema = SchemaBuilder::new()
            .label(person_label())
            .unwrap()
            .edge_type(knows_edge())
            .unwrap()
            .build();

        let alice = sorted_node_id(1);
        let bob = sorted_node_id(2);
        let carol = sorted_node_id(3);

        // Flush alice -> bob, alice -> carol.
        let mut mt_flush = Memtable::new();
        mt_flush.apply(
            MemKey::Edge {
                edge_type: "KNOWS".into(),
                src: alice,
                dst: bob,
            },
            10,
            MemOp::Upsert(edge_payload()),
        );
        mt_flush.apply(
            MemKey::Edge {
                edge_type: "KNOWS".into(),
                src: alice,
                dst: carol,
            },
            11,
            MemOp::Upsert(edge_payload()),
        );
        let frozen = mt_flush.freeze();
        let outcome = flush(&ms, &fence, &base, &frozen, schema.clone())
            .await
            .unwrap();

        // Live memtable: tombstone alice -> bob at LSN 20.
        let mut live = Memtable::new();
        live.apply(
            MemKey::Edge {
                edge_type: "KNOWS".into(),
                src: alice,
                dst: bob,
            },
            20,
            MemOp::Tombstone,
        );
        let live_view = live.snapshot_view();
        let snap = Snapshot::new(outcome.committed.clone(), &live_view, store, paths)
            .with_adjacency_cache(Arc::new(AdjacencyCache::new(adjacency_budget_bytes())));

        let partners = snap
            .sorted_partners("KNOWS", alice, EdgeDirection::Forward)
            .await
            .unwrap();
        assert_eq!(partners, vec![carol]);
    }

    #[tokio::test]
    async fn sorted_partners_inverse_direction_returns_sources() {
        let store = make_store();
        let paths = make_paths("read-sp-inv");
        let ms = ManifestStore::new(store.clone(), paths.clone());
        let base = ms.bootstrap(Uuid::now_v7()).await.unwrap();
        let fence = WriterFence::new(base.manifest.epoch);
        let schema = SchemaBuilder::new()
            .label(person_label())
            .unwrap()
            .edge_type(knows_edge())
            .unwrap()
            .build();

        let alice = sorted_node_id(1);
        let bob = sorted_node_id(2);
        let carol = sorted_node_id(3);

        // bob -> alice, carol -> alice. Inverse of alice should yield
        // both sources sorted ascending.
        let mut mt = Memtable::new();
        mt.apply(
            MemKey::Edge {
                edge_type: "KNOWS".into(),
                src: bob,
                dst: alice,
            },
            10,
            MemOp::Upsert(edge_payload()),
        );
        mt.apply(
            MemKey::Edge {
                edge_type: "KNOWS".into(),
                src: carol,
                dst: alice,
            },
            11,
            MemOp::Upsert(edge_payload()),
        );
        let frozen = mt.freeze();
        let outcome = flush(&ms, &fence, &base, &frozen, schema.clone())
            .await
            .unwrap();

        let empty = Memtable::new();
        let empty_view = empty.snapshot_view();
        let snap = Snapshot::new(outcome.committed.clone(), &empty_view, store, paths)
            .with_adjacency_cache(Arc::new(AdjacencyCache::new(adjacency_budget_bytes())));

        let partners = snap
            .sorted_partners("KNOWS", alice, EdgeDirection::Inverse)
            .await
            .unwrap();
        assert_eq!(partners, vec![bob, carol]);
    }

    #[tokio::test]
    async fn key_range_prune_skips_irrelevant_ssts() {
        // Two flushes with disjoint node_id ranges: only one SST should be
        // GETted when looking up a key in the second range. We verify via
        // the existence of two SSTs and a successful lookup that respects
        // their min/max bounds.
        let store = make_store();
        let paths = make_paths("read-prune");
        let ms = ManifestStore::new(store.clone(), paths.clone());
        let mut base = ms.bootstrap(Uuid::now_v7()).await.unwrap();
        // Seed the dict so the on-row LabelId(0) resolves to "Person".
        base.manifest.label_dict.intern("Person");
        let fence = WriterFence::new(base.manifest.epoch);
        let schema = SchemaBuilder::new().label(person_label()).unwrap().build();

        let id_low = sorted_node_id(1);
        let id_high = sorted_node_id(200);

        let mut mt1 = Memtable::new();
        mt1.apply(
            MemKey::Node { id: id_low },
            1,
            MemOp::Upsert(node_payload("Low", None)),
        );
        let frozen1 = mt1.freeze();
        let after1 = flush(&ms, &fence, &base, &frozen1, schema.clone())
            .await
            .unwrap();

        let mut mt2 = Memtable::new();
        mt2.apply(
            MemKey::Node { id: id_high },
            2,
            MemOp::Upsert(node_payload("High", None)),
        );
        let frozen2 = mt2.freeze();
        let after2 = flush(&ms, &fence, &after1.committed, &frozen2, schema.clone())
            .await
            .unwrap();

        assert_eq!(after2.committed.manifest.ssts.len(), 2);

        let empty = Memtable::new();
        let empty_view = empty.snapshot_view();
        let snap = Snapshot::new(after2.committed.clone(), &empty_view, store, paths);
        let low = snap.lookup_node("Person", id_low).await.unwrap().unwrap();
        assert_eq!(low.properties.get("name"), Some(&Value::Str("Low".into())));
        let high = snap.lookup_node("Person", id_high).await.unwrap().unwrap();
        assert_eq!(
            high.properties.get("name"),
            Some(&Value::Str("High".into()))
        );
    }

    #[tokio::test]
    async fn snapshot_with_cache_serves_warm_lookups_from_memory() {
        use crate::cache::SstCache;

        let store = make_store();
        let paths = make_paths("read-cache");
        let ms = ManifestStore::new(store.clone(), paths.clone());
        let mut base = ms.bootstrap(Uuid::now_v7()).await.unwrap();
        // Seed the dict so the on-row LabelId(0) resolves to "Person".
        base.manifest.label_dict.intern("Person");
        let fence = WriterFence::new(base.manifest.epoch);
        let schema = SchemaBuilder::new().label(person_label()).unwrap().build();

        let alice = sorted_node_id(1);
        let mut mt = Memtable::new();
        mt.apply(
            MemKey::Node { id: alice },
            5,
            MemOp::Upsert(node_payload("Alice", Some(30))),
        );
        let frozen = mt.freeze();
        let after = flush(&ms, &fence, &base, &frozen, schema.clone())
            .await
            .unwrap();

        let cache = SstCache::new(8 * 1024 * 1024);
        let empty = Memtable::new();
        let empty_view = empty.snapshot_view();
        let snap = Snapshot::new(after.committed.clone(), &empty_view, store, paths)
            .with_cache(cache.clone());

        // Cold read: cache miss, then insert.
        let cold = snap.lookup_node("Person", alice).await.unwrap().unwrap();
        assert_eq!(cold.lsn, 5);
        let cold_inserts = cache.inserts();
        let cold_misses = cache.misses();
        assert!(cold_inserts >= 1, "cold path must insert at least one body");
        assert!(cold_misses >= 1, "cold path must record at least one miss");

        // Warm read: same key, same snapshot — every body and bloom is
        // already cached. Insert count must not grow.
        //
        // Note: the per-snapshot NodeView cache short-circuits the
        // second `lookup_node` BEFORE reaching the `SstCache`, so
        // `cache.hits()` may stay at zero on the warm path. The
        // important invariant is that the warm path performs no new
        // object-store work, which `inserts()` already captures.
        let warm = snap.lookup_node("Person", alice).await.unwrap().unwrap();
        assert_eq!(warm, cold);
        assert_eq!(
            cache.inserts(),
            cold_inserts,
            "warm path must not insert anything new"
        );
        assert!(cache.usage() > 0);
    }

    #[tokio::test]
    async fn edge_properties_round_trip_through_sst_overflow_stream() {
        // Regression for "EdgeView.properties is empty after flush".
        // Edges carry `since`/`weight` overflow JSON; after flush the read
        // path must decode the SST's overflow section and present the
        // same property map a memtable read would have produced.
        let store = make_store();
        let paths = make_paths("edge-props-overflow");
        let ms = ManifestStore::new(store.clone(), paths.clone());
        let base = ms.bootstrap(Uuid::now_v7()).await.unwrap();
        let fence = WriterFence::new(base.manifest.epoch);
        let schema = SchemaBuilder::new()
            .label(person_label())
            .unwrap()
            .edge_type(knows_edge())
            .unwrap()
            .build();

        let alice = sorted_node_id(1);
        let bob = sorted_node_id(2);
        let carol = sorted_node_id(3);

        let mut mt = Memtable::new();
        let mut props_ab: BTreeMap<String, Value> = BTreeMap::new();
        props_ab.insert("since".into(), Value::I64(2020));
        props_ab.insert("weight".into(), Value::F64(0.75));
        mt.apply(
            MemKey::Edge {
                edge_type: "KNOWS".into(),
                src: alice,
                dst: bob,
            },
            10,
            MemOp::Upsert(
                EdgeWriteRecord {
                    properties: props_ab.clone(),
                    schema_version: 1,
                }
                .encode()
                .unwrap(),
            ),
        );
        let mut props_ac: BTreeMap<String, Value> = BTreeMap::new();
        props_ac.insert("since".into(), Value::I64(2024));
        mt.apply(
            MemKey::Edge {
                edge_type: "KNOWS".into(),
                src: alice,
                dst: carol,
            },
            11,
            MemOp::Upsert(
                EdgeWriteRecord {
                    properties: props_ac.clone(),
                    schema_version: 1,
                }
                .encode()
                .unwrap(),
            ),
        );
        let frozen = mt.freeze();
        let after = flush(&ms, &fence, &base, &frozen, schema.clone())
            .await
            .unwrap();

        let empty = Memtable::new();
        let empty_view = empty.snapshot_view();
        let snap = Snapshot::new(after.committed, &empty_view, store, paths);

        let out = snap.out_edges("KNOWS", alice).await.unwrap();
        assert_eq!(out.edges.len(), 2);
        let edge_to_bob = out.edges.iter().find(|e| e.dst == bob).unwrap();
        assert_eq!(edge_to_bob.properties, props_ab);
        let edge_to_carol = out.edges.iter().find(|e| e.dst == carol).unwrap();
        assert_eq!(edge_to_carol.properties, props_ac);

        // scan_edge_type must also surface the properties.
        let all = snap.scan_edge_type("KNOWS").await.unwrap();
        let by_dst: BTreeMap<NodeId, &EdgeView> = all.iter().map(|e| (e.dst, e)).collect();
        assert_eq!(by_dst[&bob].properties, props_ab);
        assert_eq!(by_dst[&carol].properties, props_ac);
    }

    fn knows_edge_with_declared() -> EdgeTypeDef {
        EdgeTypeDef {
            name: "KNOWS".into(),
            src_label: "Person".into(),
            dst_label: "Person".into(),
            properties: vec![
                namidb_core::PropertyDef::new("since", DataType::Int64, true).unwrap(),
                namidb_core::PropertyDef::new("weight", DataType::Float64, true).unwrap(),
            ],
        }
    }

    #[tokio::test]
    async fn declared_edge_properties_round_trip_through_named_streams() {
        // RFC-002 §3.2.7: when the schema declares edge-type properties,
        // the flush writes one Arrow IPC stream per declared name (with
        // JSON-encoded `Value` payloads) under `SECTION_PROPERTY_STREAM`.
        // The reader merges those back into `EdgeView.properties` exactly
        // like the overflow path. This regression test ensures declared
        // and ad-hoc properties both land correctly post-flush.
        let store = make_store();
        let paths = make_paths("edge-props-declared");
        let ms = ManifestStore::new(store.clone(), paths.clone());
        let base = ms.bootstrap(Uuid::now_v7()).await.unwrap();
        let fence = WriterFence::new(base.manifest.epoch);
        let schema = SchemaBuilder::new()
            .label(person_label())
            .unwrap()
            .edge_type(knows_edge_with_declared())
            .unwrap()
            .build();

        let alice = sorted_node_id(1);
        let bob = sorted_node_id(2);
        let carol = sorted_node_id(3);

        let mut mt = Memtable::new();
        // Bob's edge carries `since` + `weight` (both declared) + an
        // ad-hoc `note` (must land in __overflow_json).
        let mut props_ab: BTreeMap<String, Value> = BTreeMap::new();
        props_ab.insert("since".into(), Value::I64(2020));
        props_ab.insert("weight".into(), Value::F64(0.75));
        props_ab.insert("note".into(), Value::Str("close friend".into()));
        mt.apply(
            MemKey::Edge {
                edge_type: "KNOWS".into(),
                src: alice,
                dst: bob,
            },
            10,
            MemOp::Upsert(
                EdgeWriteRecord {
                    properties: props_ab.clone(),
                    schema_version: 1,
                }
                .encode()
                .unwrap(),
            ),
        );
        // Carol's edge only carries `since` (declared, but `weight`
        // omitted) and no ad-hoc properties.
        let mut props_ac: BTreeMap<String, Value> = BTreeMap::new();
        props_ac.insert("since".into(), Value::I64(2024));
        mt.apply(
            MemKey::Edge {
                edge_type: "KNOWS".into(),
                src: alice,
                dst: carol,
            },
            11,
            MemOp::Upsert(
                EdgeWriteRecord {
                    properties: props_ac.clone(),
                    schema_version: 1,
                }
                .encode()
                .unwrap(),
            ),
        );
        let frozen = mt.freeze();
        let after = flush(&ms, &fence, &base, &frozen, schema.clone())
            .await
            .unwrap();

        let empty = Memtable::new();
        let empty_view = empty.snapshot_view();
        let snap = Snapshot::new(after.committed, &empty_view, store, paths);

        // out_edges: both declared and ad-hoc properties round-trip.
        let out = snap.out_edges("KNOWS", alice).await.unwrap();
        assert_eq!(out.edges.len(), 2);
        let edge_to_bob = out.edges.iter().find(|e| e.dst == bob).unwrap();
        assert_eq!(edge_to_bob.properties, props_ab);
        let edge_to_carol = out.edges.iter().find(|e| e.dst == carol).unwrap();
        assert_eq!(edge_to_carol.properties, props_ac);

        // scan_edge_type also surfaces them.
        let all = snap.scan_edge_type("KNOWS").await.unwrap();
        let by_dst: BTreeMap<NodeId, &EdgeView> = all.iter().map(|e| (e.dst, e)).collect();
        assert_eq!(by_dst[&bob].properties, props_ab);
        assert_eq!(by_dst[&carol].properties, props_ac);

        // in_edges (inverse partner): exactly the same property set.
        let in_b = snap.in_edges("KNOWS", bob).await.unwrap();
        assert_eq!(in_b.edges.len(), 1);
        assert_eq!(in_b.edges[0].properties, props_ab);
        let in_c = snap.in_edges("KNOWS", carol).await.unwrap();
        assert_eq!(in_c.edges.len(), 1);
        assert_eq!(in_c.edges[0].properties, props_ac);
    }

    #[tokio::test]
    async fn scan_label_returns_all_nodes_in_id_order_with_tombstones_pruned() {
        let store = make_store();
        let paths = make_paths("scan-nodes");
        let ms = ManifestStore::new(store.clone(), paths.clone());
        let mut base = ms.bootstrap(Uuid::now_v7()).await.unwrap();
        // Seed the dict so the on-row LabelId(0) resolves to "Person" for both
        // the flushed SST rows and the live-memtable rows.
        base.manifest.label_dict.intern("Person");
        let fence = WriterFence::new(base.manifest.epoch);
        let schema = SchemaBuilder::new().label(person_label()).unwrap().build();

        // Flush three nodes at LSNs 1..3.
        let alice = sorted_node_id(1);
        let bob = sorted_node_id(2);
        let carol = sorted_node_id(3);
        let mut mt_flush = Memtable::new();
        for (i, id) in [(1u64, alice), (2, bob), (3, carol)] {
            mt_flush.apply(
                MemKey::Node { id },
                i,
                MemOp::Upsert(node_payload("X", None)),
            );
        }
        let frozen = mt_flush.freeze();
        let after = flush(&ms, &fence, &base, &frozen, schema.clone())
            .await
            .unwrap();

        // Live memtable carries: an update to alice at LSN 10, a tombstone
        // for bob at LSN 11, and a new node dave at LSN 12.
        let dave = sorted_node_id(4);
        let mut live = Memtable::new();
        live.apply(
            MemKey::Node { id: alice },
            10,
            MemOp::Upsert(node_payload("Alice-updated", Some(99))),
        );
        live.apply(MemKey::Node { id: bob }, 11, MemOp::Tombstone);
        live.apply(
            MemKey::Node { id: dave },
            12,
            MemOp::Upsert(node_payload("Dave", None)),
        );

        let live_view = live.snapshot_view();
        let snap = Snapshot::new(after.committed, &live_view, store, paths);
        let rows = snap.scan_label("Person").await.unwrap();

        // bob (tombstoned) absent; alice, carol, dave present (3 nodes).
        let ids: Vec<NodeId> = rows.iter().map(|n| n.id).collect();
        assert_eq!(ids, vec![alice, carol, dave], "ids must be ascending");

        // alice's row must reflect the memtable upsert (lsn 10), not the
        // older SST row (lsn 1).
        let alice_row = rows.iter().find(|n| n.id == alice).unwrap();
        assert_eq!(alice_row.lsn, 10);
        assert_eq!(
            alice_row.properties.get("name"),
            Some(&Value::Str("Alice-updated".into()))
        );
    }

    #[tokio::test]
    async fn scan_edge_type_merges_memtable_and_ssts_with_tombstones() {
        let store = make_store();
        let paths = make_paths("scan-edges");
        let ms = ManifestStore::new(store.clone(), paths.clone());
        let base = ms.bootstrap(Uuid::now_v7()).await.unwrap();
        let fence = WriterFence::new(base.manifest.epoch);
        let schema = SchemaBuilder::new()
            .label(person_label())
            .unwrap()
            .edge_type(knows_edge())
            .unwrap()
            .build();

        let alice = sorted_node_id(1);
        let bob = sorted_node_id(2);
        let carol = sorted_node_id(3);

        // Flush alice→bob and bob→carol.
        let mut mt_flush = Memtable::new();
        mt_flush.apply(
            MemKey::Edge {
                edge_type: "KNOWS".into(),
                src: alice,
                dst: bob,
            },
            1,
            MemOp::Upsert(edge_payload()),
        );
        mt_flush.apply(
            MemKey::Edge {
                edge_type: "KNOWS".into(),
                src: bob,
                dst: carol,
            },
            2,
            MemOp::Upsert(edge_payload()),
        );
        let frozen = mt_flush.freeze();
        let after = flush(&ms, &fence, &base, &frozen, schema.clone())
            .await
            .unwrap();

        // Live memtable: tombstone alice→bob, add carol→alice.
        let mut live = Memtable::new();
        live.apply(
            MemKey::Edge {
                edge_type: "KNOWS".into(),
                src: alice,
                dst: bob,
            },
            10,
            MemOp::Tombstone,
        );
        live.apply(
            MemKey::Edge {
                edge_type: "KNOWS".into(),
                src: carol,
                dst: alice,
            },
            11,
            MemOp::Upsert(edge_payload()),
        );

        let live_view = live.snapshot_view();
        let snap = Snapshot::new(after.committed, &live_view, store, paths);
        let edges = snap.scan_edge_type("KNOWS").await.unwrap();

        // Expected: bob→carol (from SST) and carol→alice (from memtable).
        // alice→bob is tombstoned out.
        let pairs: Vec<(NodeId, NodeId)> = edges.iter().map(|e| (e.src, e.dst)).collect();
        assert_eq!(pairs, vec![(bob, carol), (carol, alice)]);
    }

    #[tokio::test]
    async fn bloom_admits_rejects_absent_key_and_admits_present_one() {
        // Drive `Snapshot::bloom_admits` directly. We synthesise a
        // descriptor + side-car so the test does not depend on whether
        // the flush path happened to keep a bloom for its SST (it does
        // not for tiny bodies — see RFC-002 §4.2).
        use crate::manifest::{KindSpecificStats, SstKind, SstLevel};
        use crate::sst::bloom::{BloomFilter, DEFAULT_BITS_PER_KEY};
        use chrono::Utc;
        use object_store::PutPayload;

        let store = make_store();
        let paths = make_paths("read-bloom-unit");
        let ms = ManifestStore::new(store.clone(), paths.clone());
        let base = ms.bootstrap(Uuid::now_v7()).await.unwrap();

        // Build a filter holding nodes 1..=10 and push it to the store.
        let mut filter = BloomFilter::with_capacity(16, DEFAULT_BITS_PER_KEY);
        for i in 1u8..=10 {
            filter.insert(sorted_node_id(i).as_bytes());
        }
        let bloom_bytes = filter.to_bytes();
        let relative = "sst/level0/bloom-test.bloom".to_string();
        let absolute = paths.sst_object(0, "bloom-test.bloom");
        store
            .put(&absolute, PutPayload::from(bloom_bytes.clone()))
            .await
            .unwrap();
        let bloom_desc =
            crate::sst::bloom::BloomDescriptor::from_body(relative.clone(), &bloom_bytes).unwrap();

        let descriptor = SstDescriptor {
            id: Uuid::now_v7(),
            kind: SstKind::Nodes,
            scope: "Person".into(),
            level: SstLevel::L0,
            path: "sst/level0/bloom-test.parquet".into(),
            size_bytes: 1,
            row_count: 10,
            created_at: Utc::now(),
            min_key: *sorted_node_id(1).as_bytes(),
            max_key: *sorted_node_id(10).as_bytes(),
            min_lsn: 1,
            max_lsn: 10,
            schema_version_min: 0,
            schema_version_max: 0,
            property_stats: vec![],
            kind_specific: KindSpecificStats::Nodes { tombstone_count: 0 },
            bloom: Some(bloom_desc),
            unique_property_indices: Vec::new(),
            equality_property_indices: Vec::new(),
            label_index: None,
            per_label_property_stats: Vec::new(),
        };

        let empty = Memtable::new();
        let empty_view = empty.snapshot_view();
        let snap = Snapshot::new(base, &empty_view, store.clone(), paths);
        assert!(
            snap.bloom_admits(&descriptor, sorted_node_id(5).as_bytes())
                .await
                .unwrap(),
            "inserted key must pass the bloom"
        );
        assert!(
            !snap
                .bloom_admits(&descriptor, sorted_node_id(99).as_bytes())
                .await
                .unwrap(),
            "key never inserted should be rejected by the bloom"
        );

        // Sanity: an SstDescriptor with `bloom = None` admits everything.
        let no_bloom = SstDescriptor {
            bloom: None,
            unique_property_indices: Vec::new(),
            equality_property_indices: Vec::new(),
            label_index: None,
            per_label_property_stats: Vec::new(),
            ..descriptor.clone()
        };
        assert!(snap
            .bloom_admits(&no_bloom, sorted_node_id(99).as_bytes())
            .await
            .unwrap());
    }

    #[tokio::test]
    async fn lookup_node_works_against_recovered_memtable_with_unflushed_wal() {
        // End-to-end smoke: a WAL segment whose records have not yet
        // been flushed is replayed by `recovery::recover_memtable`, and
        // the resulting Memtable feeds a Snapshot that reads the
        // unflushed state alongside any persisted SSTs (here: none).
        use crate::manifest::WalSegmentDescriptor;
        use crate::recovery::{recover_memtable, WalEntry, WalOp};
        use crate::wal::{WalRecord, WalStore};

        let store = make_store();
        let paths = make_paths("read-recovery");
        let ms = ManifestStore::new(store.clone(), paths.clone());
        let wal_store = WalStore::new(store.clone(), paths.clone());
        let mut base = ms.bootstrap(Uuid::now_v7()).await.unwrap();
        // Seed the dict so the recovered record's on-row LabelId(0) resolves to
        // "Person". `next_version` clones the dict forward into `with_wal`.
        base.manifest.label_dict.intern("Person");
        let fence = WriterFence::new(base.manifest.epoch);

        let alice = sorted_node_id(5);

        // Append a WAL segment containing an upsert for Alice.
        let entry = WalEntry {
            key: MemKey::Node { id: alice },
            op: WalOp::Upsert(node_payload("Alice", Some(40)).to_vec()),
            lsn: 30,
        };
        let mut seg = WalSegment::new(1);
        seg.push(WalRecord {
            lsn: 30,
            payload: entry.encode().unwrap(),
        });
        let seg_path = wal_store.append_segment(&seg).await.unwrap();

        // Commit a manifest version that references the segment so the
        // recovery step sees it.
        let mut next = base.manifest.next_version(fence.writer_id);
        next.wal_segments.push(WalSegmentDescriptor {
            seq: seg.seq,
            path: seg_path.as_ref().to_string(),
            last_lsn: seg.last_lsn(),
            xxh3: None,
        });
        let with_wal = ms.commit(&fence, &base, next).await.unwrap();

        let recovered = recover_memtable(&with_wal.manifest, &wal_store)
            .await
            .unwrap();
        assert_eq!(recovered.records_replayed, 1);
        assert_eq!(recovered.max_lsn, 30);

        let view = recovered.memtable.snapshot_view();
        let snap = Snapshot::new(with_wal, &view, store, paths);
        let view = snap.lookup_node("Person", alice).await.unwrap().unwrap();
        assert_eq!(view.lsn, 30);
        assert_eq!(
            view.properties.get("name"),
            Some(&Value::Str("Alice".into()))
        );
    }

    #[tokio::test]
    async fn observed_edge_endpoints_returns_declared_pairs_first() {
        let store = make_store();
        let paths = make_paths("schema-endpoints-declared");
        let ms = ManifestStore::new(store.clone(), paths.clone());
        let mut base = ms.bootstrap(Uuid::now_v7()).await.unwrap();
        // Bake a declared schema into the manifest directly so the
        // snapshot sees it without going through a writer commit.
        base.manifest.schema = SchemaBuilder::new()
            .label(person_label())
            .unwrap()
            .edge_type(knows_edge())
            .unwrap()
            .build();
        let mt = Memtable::new();
        let view = mt.snapshot_view();
        let snap = Snapshot::new(base, &view, store, paths);

        let endpoints = snap.observed_edge_endpoints().await.unwrap();
        assert_eq!(endpoints.len(), 1);
        let ep = &endpoints[0];
        assert_eq!(ep.edge_type, "KNOWS");
        assert_eq!(ep.src_label.as_deref(), Some("Person"));
        assert_eq!(ep.dst_label.as_deref(), Some("Person"));
        assert!(!ep.inferred);
    }

    #[tokio::test]
    async fn observed_edge_endpoints_infers_when_schema_is_empty() {
        let store = make_store();
        let paths = make_paths("schema-endpoints-inferred");
        let ms = ManifestStore::new(store.clone(), paths.clone());
        let mut base = ms.bootstrap(Uuid::now_v7()).await.unwrap();
        // Two distinct labels: "Person" -> LabelId(0), "Company" -> LabelId(1).
        // Seed the dict so each record's on-row id resolves to its name.
        base.manifest.label_dict.intern("Person");
        base.manifest.label_dict.intern("Company");

        // Two nodes with distinct labels, one edge that ties them
        // together, no `SchemaBuilder` ever ran.
        let person = sorted_node_id(1);
        let company = sorted_node_id(2);
        let mut mt = Memtable::new();
        mt.apply(
            MemKey::Node { id: person },
            1,
            MemOp::Upsert(node_payload("Alice", None)),
        );
        mt.apply(
            MemKey::Node { id: company },
            2,
            MemOp::Upsert(labeled_node_payload("Acme", 1)),
        );
        mt.apply(
            MemKey::Edge {
                edge_type: "WORKS_AT".into(),
                src: person,
                dst: company,
            },
            3,
            MemOp::Upsert(edge_payload()),
        );
        let view = mt.snapshot_view();
        let snap = Snapshot::new(base, &view, store, paths);

        let endpoints = snap.observed_edge_endpoints().await.unwrap();
        assert_eq!(endpoints.len(), 1);
        let ep = &endpoints[0];
        assert_eq!(ep.edge_type, "WORKS_AT");
        assert_eq!(ep.src_label.as_deref(), Some("Person"));
        assert_eq!(ep.dst_label.as_deref(), Some("Company"));
        assert!(ep.inferred);
    }

    #[tokio::test]
    async fn observed_edge_endpoints_infers_from_flushed_sst() {
        // Regression: a bulk-loaded namespace flushes its edges into SSTs,
        // leaving the live memtable empty. Endpoint inference must fall
        // back to sampling an edge from the forward SST rather than
        // returning None — otherwise the dashboard's graph explorer cannot
        // collapse its cartesian probe fan-out for such namespaces.
        let store = make_store();
        let paths = make_paths("schema-endpoints-sst");
        let ms = ManifestStore::new(store.clone(), paths.clone());
        let mut base = ms.bootstrap(Uuid::now_v7()).await.unwrap();
        // Two distinct labels: "Person" -> LabelId(0), "Company" -> LabelId(1).
        // Seed the dict so each flushed SST row's on-row id resolves to its
        // name (`next_version` clones the dict forward into the committed
        // manifest).
        base.manifest.label_dict.intern("Person");
        base.manifest.label_dict.intern("Company");
        let fence = WriterFence::new(base.manifest.epoch);

        // Declare the node labels so the flush writes node SSTs, but leave
        // the edge type UNDECLARED — that is exactly the case we infer.
        let company_label = LabelDef {
            name: "Company".into(),
            properties: vec![PropertyDef::new("name", DataType::Utf8, false).unwrap()],
        };
        let schema = SchemaBuilder::new()
            .label(person_label())
            .unwrap()
            .label(company_label)
            .unwrap()
            .build();

        let person = sorted_node_id(1);
        let company = sorted_node_id(2);
        let mut mt = Memtable::new();
        mt.apply(
            MemKey::Node { id: person },
            1,
            MemOp::Upsert(node_payload("Alice", None)),
        );
        mt.apply(
            MemKey::Node { id: company },
            2,
            MemOp::Upsert(labeled_node_payload("Acme", 1)),
        );
        mt.apply(
            MemKey::Edge {
                edge_type: "WORKS_AT".into(),
                src: person,
                dst: company,
            },
            3,
            MemOp::Upsert(edge_payload()),
        );
        let frozen = mt.freeze();
        let outcome = flush(&ms, &fence, &base, &frozen, schema).await.unwrap();

        // Live memtable empty → inference must read the sample edge from
        // the forward SST, not the memtable.
        let empty = Memtable::new();
        let empty_view = empty.snapshot_view();
        let snap = Snapshot::new(outcome.committed.clone(), &empty_view, store, paths);

        let endpoints = snap.observed_edge_endpoints().await.unwrap();
        let ep = endpoints
            .iter()
            .find(|e| e.edge_type == "WORKS_AT")
            .expect("WORKS_AT endpoint present");
        assert_eq!(ep.src_label.as_deref(), Some("Person"));
        assert_eq!(ep.dst_label.as_deref(), Some("Company"));
        assert!(ep.inferred);
    }

    #[tokio::test]
    async fn observed_edge_endpoints_handles_orphan_edge_type() {
        // Edge type observed (tombstone-only memtable entries — no
        // upsert ever present in this snapshot). Should surface with
        // None / None rather than panic or skip.
        let store = make_store();
        let paths = make_paths("schema-endpoints-orphan");
        let ms = ManifestStore::new(store.clone(), paths.clone());
        let base = ms.bootstrap(Uuid::now_v7()).await.unwrap();
        let mut mt = Memtable::new();
        mt.apply(
            MemKey::Edge {
                edge_type: "GHOST".into(),
                src: sorted_node_id(10),
                dst: sorted_node_id(11),
            },
            1,
            MemOp::Tombstone,
        );
        let view = mt.snapshot_view();
        let snap = Snapshot::new(base, &view, store, paths);

        let endpoints = snap.observed_edge_endpoints().await.unwrap();
        assert_eq!(endpoints.len(), 1);
        let ep = &endpoints[0];
        assert_eq!(ep.edge_type, "GHOST");
        assert!(ep.src_label.is_none());
        assert!(ep.dst_label.is_none());
        assert!(ep.inferred);
    }

    #[tokio::test]
    async fn observed_property_types_returns_declared_when_no_ssts() {
        let store = make_store();
        let paths = make_paths("schema-props-declared-only");
        let ms = ManifestStore::new(store.clone(), paths.clone());
        let mut base = ms.bootstrap(Uuid::now_v7()).await.unwrap();
        base.manifest.schema = SchemaBuilder::new().label(person_label()).unwrap().build();
        let mt = Memtable::new();
        let view = mt.snapshot_view();
        let snap = Snapshot::new(base, &view, store, paths);

        let props = snap.observed_property_types_for_label("Person");
        assert_eq!(props.len(), 2);
        assert_eq!(props.get("name"), Some(&DataType::Utf8));
        assert_eq!(props.get("age"), Some(&DataType::Int32));
    }

    #[tokio::test]
    async fn observed_property_types_falls_back_to_sst_stats_when_schema_drifts() {
        // Real-world hook: a schema migration removed a property
        // (`age`) but SSTs from before the migration still ship column
        // stats for it. The schema-introspection caller wants to know
        // the column is still observable so it can warn the user, and
        // the SST stats carry enough type info via min/max to surface
        // it without opening the parquet body.
        use crate::manifest::{KindSpecificStats, SstDescriptor, SstKind, SstLevel};
        use crate::sst::bloom::BloomDescriptor;
        use crate::sst::stats::{PropertyColumnStats, StatScalar};
        use chrono::Utc;

        let store = make_store();
        let paths = make_paths("schema-props-drift");
        let ms = ManifestStore::new(store.clone(), paths.clone());
        let mut base = ms.bootstrap(Uuid::now_v7()).await.unwrap();
        base.manifest.schema = SchemaBuilder::new()
            // Declared schema only knows about `name`.
            .label(LabelDef {
                name: "Person".into(),
                properties: vec![PropertyDef::new("name", DataType::Utf8, false).unwrap()],
            })
            .unwrap()
            .build();
        // Inject a stale SST descriptor that still reports an `age`
        // column from before the migration.
        base.manifest.ssts.push(SstDescriptor {
            id: Uuid::now_v7(),
            kind: SstKind::Nodes,
            scope: "Person".into(),
            level: SstLevel::L0,
            path: "stale.parquet".into(),
            size_bytes: 1,
            row_count: 1,
            created_at: Utc::now(),
            min_key: [0u8; 16],
            max_key: [0u8; 16],
            min_lsn: 1,
            max_lsn: 1,
            schema_version_min: 1,
            schema_version_max: 1,
            property_stats: vec![
                PropertyColumnStats {
                    name: "prop_name".into(),
                    null_count: 0,
                    min: Some(StatScalar::Utf8("a".into())),
                    max: Some(StatScalar::Utf8("z".into())),
                    ndv_estimate: None,
                },
                PropertyColumnStats {
                    name: "prop_age".into(),
                    null_count: 0,
                    min: Some(StatScalar::Int32(18)),
                    max: Some(StatScalar::Int32(90)),
                    ndv_estimate: None,
                },
            ],
            kind_specific: KindSpecificStats::Nodes { tombstone_count: 0 },
            bloom: None::<BloomDescriptor>,
            unique_property_indices: Vec::new(),
            equality_property_indices: Vec::new(),
            label_index: None,
            per_label_property_stats: Vec::new(),
        });
        let mt = Memtable::new();
        let view = mt.snapshot_view();
        let snap = Snapshot::new(base, &view, store, paths);

        let props = snap.observed_property_types_for_label("Person");
        // Declared property keeps its declared type.
        assert_eq!(props.get("name"), Some(&DataType::Utf8));
        // Stale SST-only column surfaces from the recorded scalar.
        assert_eq!(props.get("age"), Some(&DataType::Int32));
    }

    #[tokio::test]
    async fn observed_property_types_declared_overrides_sst_stats() {
        // Declared properties win even when an SST exists. This
        // matters for properties whose declared type is wider than the
        // observed values (e.g. `Int64` declared, `Int32` observed).
        let store = make_store();
        let paths = make_paths("schema-props-declared-wins");
        let ms = ManifestStore::new(store.clone(), paths.clone());
        let base = ms.bootstrap(Uuid::now_v7()).await.unwrap();
        let fence = WriterFence::new(base.manifest.epoch);
        let schema = SchemaBuilder::new().label(person_label()).unwrap().build();
        let alice = sorted_node_id(1);
        let mut mt = Memtable::new();
        mt.apply(
            MemKey::Node { id: alice },
            1,
            MemOp::Upsert(node_payload("Alice", Some(30))),
        );
        let frozen = mt.freeze();
        let outcome = flush(&ms, &fence, &base, &frozen, schema.clone())
            .await
            .unwrap();
        let empty = Memtable::new();
        let view = empty.snapshot_view();
        let snap = Snapshot::new(outcome.committed.clone(), &view, store, paths);

        let props = snap.observed_property_types_for_label("Person");
        // person_label() declares age as Int32. Even though the writer
        // happens to store it as Int64 in the SST, the declared type
        // is what surfaces in the schema introspection.
        assert_eq!(props.get("age"), Some(&DataType::Int32));
    }

    // ── batch_lookup_nodes row-group pruning + decoded cache ──

    /// Serialises the tests that force small node-SST row groups through
    /// `NAMIDB_NODE_SST_ROW_GROUP_ROWS`, restoring the previous value so
    /// parallel tests never observe a partial state.
    static ROW_GROUP_ENV: std::sync::Mutex<()> = std::sync::Mutex::new(());

    // Intentional: the guard serialises env mutation across the whole
    // flush; each test drives its own single-threaded runtime.
    #[allow(clippy::await_holding_lock)]
    async fn flush_batch_with_row_group_rows(
        rows_per_group: usize,
        ms: &ManifestStore,
        fence: &WriterFence,
        base: &LoadedManifest,
        schema: &namidb_core::Schema,
        rows: Vec<(NodeId, u64, MemOp)>,
    ) -> LoadedManifest {
        let _guard = ROW_GROUP_ENV.lock().unwrap_or_else(|e| e.into_inner());
        let prev = std::env::var("NAMIDB_NODE_SST_ROW_GROUP_ROWS").ok();
        std::env::set_var("NAMIDB_NODE_SST_ROW_GROUP_ROWS", rows_per_group.to_string());
        let committed = flush_batch(ms, fence, base, schema, rows).await;
        match prev {
            Some(v) => std::env::set_var("NAMIDB_NODE_SST_ROW_GROUP_ROWS", v),
            None => std::env::remove_var("NAMIDB_NODE_SST_ROW_GROUP_ROWS"),
        }
        committed
    }

    /// `(committed, node SST absolute path)` for a Person namespace whose
    /// single node SST holds ids 1..=n at `rows_per_group` rows per row
    /// group. Id 5 is tombstoned when `n >= 5`.
    async fn multi_row_group_fixture(
        store: &Arc<dyn ObjectStore>,
        paths: &NamespacePaths,
        n: u8,
        rows_per_group: usize,
    ) -> (LoadedManifest, String) {
        let ms = ManifestStore::new(store.clone(), paths.clone());
        let mut base = ms.bootstrap(Uuid::now_v7()).await.unwrap();
        base.manifest.label_dict.intern("Person");
        let fence = WriterFence::new(base.manifest.epoch);
        let schema = SchemaBuilder::new().label(person_label()).unwrap().build();

        let mut rows: Vec<(NodeId, u64, MemOp)> = (1..=n)
            .map(|i| {
                (
                    sorted_node_id(i),
                    10 + i as u64,
                    MemOp::Upsert(node_payload(&format!("n{i}"), Some(i as i32))),
                )
            })
            .collect();
        if n >= 5 {
            rows.push((sorted_node_id(5), 500, MemOp::Tombstone));
        }
        let committed =
            flush_batch_with_row_group_rows(rows_per_group, &ms, &fence, &base, &schema, rows)
                .await;
        let desc = committed
            .manifest
            .ssts
            .iter()
            .find(|d| matches!(d.kind, SstKind::Nodes))
            .expect("flush produced a node SST");
        let absolute = format!("{}/{}", paths.namespace_prefix().as_ref(), desc.path);
        (committed, absolute)
    }

    #[tokio::test]
    async fn batch_lookup_prunes_row_groups_and_matches_uncached() {
        let store = make_store();
        let paths = make_paths("batch-rg-prune");
        // 64 nodes at 8 rows per row group → 8 row groups in one SST.
        let (committed, absolute) = multi_row_group_fixture(&store, &paths, 64, 8).await;

        let cache = SstCache::new(64 * 1024 * 1024);
        let empty = Memtable::new();
        let view = empty.snapshot_view();
        let snap = Snapshot::new(committed.clone(), &view, store.clone(), paths.clone())
            .with_cache(cache.clone());

        // Live ids 2..=4 plus the tombstoned id 5 all live in row group 0
        // (rows 0..8 = ids 1..=8); id 200 is absent; a duplicate id must
        // resolve to the same view.
        let probes = vec![
            sorted_node_id(2),
            sorted_node_id(3),
            sorted_node_id(5),
            sorted_node_id(4),
            sorted_node_id(200),
            sorted_node_id(2),
        ];
        let got = snap.batch_lookup_nodes("Person", &probes).await.unwrap();

        // Correctness parity against the per-id uncached walk.
        let flat = Snapshot::new(committed.clone(), &view, store.clone(), paths.clone());
        for (i, id) in probes.iter().enumerate() {
            let want = flat.lookup_node_via_uncached("Person", *id).await.unwrap();
            assert_eq!(got[i], want, "probe #{i} diverged from the flat walk");
        }
        assert_eq!(
            got[0].as_ref().unwrap().properties.get("name"),
            Some(&Value::Str("n2".into()))
        );
        assert!(got[2].is_none(), "tombstoned id must resolve to None");
        assert!(got[4].is_none(), "absent id must resolve to None");
        assert_eq!(got[5], got[0], "duplicate probe must match");

        // The pruning path is actually in use: the SST really has 8 row
        // groups and the batch decoded ONLY the one that can hold the
        // probes — not the whole SST.
        let md = cache
            .get_metadata(&absolute)
            .expect("batch path caches footer metadata");
        assert_eq!(md.num_row_groups(), 8);
        assert_eq!(
            cache.decoded_node_row_group_inserts(),
            1,
            "ids 2..=5 share row group 0; nothing else may decode"
        );

        // Cross-snapshot reuse: a FRESH snapshot over the same cache
        // re-answers from the decoded tier without re-decoding.
        let snap2 = Snapshot::new(committed.clone(), &view, store.clone(), paths.clone())
            .with_cache(cache.clone());
        let again = snap2.batch_lookup_nodes("Person", &probes).await.unwrap();
        assert_eq!(again, got);
        assert_eq!(cache.decoded_node_row_group_inserts(), 1, "no re-decode");
        assert!(cache.decoded_node_row_group_hits() >= 1);
    }

    #[tokio::test]
    async fn batch_lookup_ranged_path_decodes_only_needed_row_groups() {
        // RFC-003 routing for the batch path: with ranged reads forced on
        // (the post-compaction large-SST scenario) the batch must resolve
        // through footer + row-group GETs — never a full-body pull — and
        // still decode only the row groups that can hold a probe id.
        let store = make_store();
        let paths = make_paths("batch-rg-ranged");
        let (committed, absolute) = multi_row_group_fixture(&store, &paths, 64, 8).await;

        let cache = SstCache::new(64 * 1024 * 1024);
        let empty = Memtable::new();
        let view = empty.snapshot_view();
        let snap = Snapshot::new(committed.clone(), &view, store.clone(), paths.clone())
            .with_cache(cache.clone())
            .with_ranged_reads(true);
        let probes = vec![sorted_node_id(2), sorted_node_id(11), sorted_node_id(200)];
        let got = snap.batch_lookup_nodes("Person", &probes).await.unwrap();

        let flat = Snapshot::new(committed.clone(), &view, store.clone(), paths.clone());
        for (i, id) in probes.iter().enumerate() {
            let want = flat.lookup_node_via_uncached("Person", *id).await.unwrap();
            assert_eq!(got[i], want, "probe #{i} diverged from the flat walk");
        }
        assert_eq!(
            cache.decoded_node_row_group_inserts(),
            2,
            "ids 2 and 11 land in row groups 0 and 1; nothing else may decode"
        );
        assert!(
            cache.get(&absolute).is_none(),
            "ranged batch path must not pull the whole body"
        );
    }

    #[tokio::test]
    async fn batch_prewarm_serves_per_id_lookup_from_shared_row_group_cache() {
        let store = make_store();
        let paths = make_paths("batch-rg-prewarm");
        let (committed, _absolute) = multi_row_group_fixture(&store, &paths, 64, 8).await;

        let cache = SstCache::new(64 * 1024 * 1024);
        let empty = Memtable::new();
        let view = empty.snapshot_view();
        {
            let snap = Snapshot::new(committed.clone(), &view, store.clone(), paths.clone())
                .with_cache(cache.clone());
            let _ = snap
                .batch_lookup_nodes("Person", &[sorted_node_id(2), sorted_node_id(11)])
                .await
                .unwrap();
        }
        assert_eq!(
            cache.decoded_node_row_group_inserts(),
            2,
            "ids 2 and 11 land in row groups 0 and 1"
        );

        // A per-id lookup on a FRESH snapshot (empty L1, no L2 attached)
        // must be served by the shared decoded row-group tier: no new
        // decode, no body or bloom GET.
        let body_misses = cache.misses();
        let rg_hits = cache.decoded_node_row_group_hits();
        let snap2 = Snapshot::new(committed.clone(), &view, store.clone(), paths.clone())
            .with_cache(cache.clone());
        let got = snap2
            .lookup_node("Person", sorted_node_id(11))
            .await
            .unwrap()
            .expect("live node");
        assert_eq!(got.properties.get("name"), Some(&Value::Str("n11".into())));
        assert!(
            cache.decoded_node_row_group_hits() > rg_hits,
            "per-id lookup must hit the decoded row-group tier"
        );
        assert_eq!(cache.decoded_node_row_group_inserts(), 2, "no re-decode");
        assert_eq!(
            cache.misses(),
            body_misses,
            "warm per-id path must not touch the object store"
        );
    }

    #[tokio::test]
    async fn decoded_row_group_cache_respects_byte_budget_across_snapshots() {
        let store = make_store();
        let paths = make_paths("batch-rg-budget");
        // 128 nodes at 4 rows per row group → 32 row groups.
        let (committed, absolute) = multi_row_group_fixture(&store, &paths, 128, 4).await;

        // Ground-truth decoded footprint: decode every row group once and
        // weigh it exactly as the cache does.
        let object_path = Path::from(absolute.clone());
        let body = store
            .get(&object_path)
            .await
            .unwrap()
            .bytes()
            .await
            .unwrap();
        let md = parse_node_sst_metadata(&body).unwrap();
        assert_eq!(md.num_row_groups(), 32);
        let empty_label = LabelDef {
            name: String::new(),
            properties: Vec::new(),
        };
        let reader = NodeSstReader::open(empty_label, body).unwrap();
        let mut total_weight = 0usize;
        let mut max_weight = 0usize;
        for rg in 0..md.num_row_groups() {
            let batches = Arc::new(reader.scan_row_groups(vec![rg]).unwrap());
            let w =
                crate::cache::decoded_node_row_group_weight(&(absolute.clone(), rg), &batches);
            total_weight += w;
            max_weight = max_weight.max(w);
        }

        // Budget for an eighth of the decoded set; every round probes ALL
        // row groups, over fresh snapshots, so an unbounded cache would
        // converge on `total_weight`.
        let budget = total_weight / 8;
        let cache = SstCache::with_budgets(64 * 1024 * 1024, budget);
        let ids: Vec<NodeId> = (1..=128u8)
            .filter(|&i| i != 5) // id 5 is tombstoned by the fixture
            .map(sorted_node_id)
            .collect();
        for round in 0..3 {
            let empty = Memtable::new();
            let view = empty.snapshot_view();
            let snap = Snapshot::new(committed.clone(), &view, store.clone(), paths.clone())
                .with_cache(cache.clone());
            let got = snap.batch_lookup_nodes("Person", &ids).await.unwrap();
            assert!(
                got.iter().all(|v| v.is_some()),
                "round {round}: over-eviction must re-decode, never lose rows"
            );
        }

        let usage = cache.decoded_node_row_groups_usage();
        assert!(
            usage < total_weight / 2,
            "decoded cache must stay bounded: usage={usage}, unbounded total={total_weight}"
        );
        // foyer's 8 shards evict independently, so allow one entry of
        // slack per shard on top of the configured budget.
        assert!(
            usage <= budget + 8 * max_weight,
            "usage={usage} exceeds budget={budget} (+ shard slack {})",
            8 * max_weight
        );
        assert!(
            cache.decoded_node_row_group_inserts() > 32,
            "budget pressure must evict + re-decode, not grow without bound"
        );
    }

    #[tokio::test]
    async fn batch_lookup_single_row_group_sst_stays_equivalent() {
        let store = make_store();
        let paths = make_paths("batch-rg-single");
        // Default row-group sizing → one row group; the pruned path must
        // behave exactly like the historical full decode.
        let ms = ManifestStore::new(store.clone(), paths.clone());
        let mut base = ms.bootstrap(Uuid::now_v7()).await.unwrap();
        base.manifest.label_dict.intern("Person");
        let fence = WriterFence::new(base.manifest.epoch);
        let schema = SchemaBuilder::new().label(person_label()).unwrap().build();
        let rows: Vec<(NodeId, u64, MemOp)> = (1..=6u8)
            .map(|i| {
                (
                    sorted_node_id(i),
                    10 + i as u64,
                    MemOp::Upsert(node_payload(&format!("n{i}"), Some(i as i32))),
                )
            })
            .collect();
        let committed = flush_batch(&ms, &fence, &base, &schema, rows).await;

        let cache = SstCache::new(64 * 1024 * 1024);
        let empty = Memtable::new();
        let view = empty.snapshot_view();
        let snap = Snapshot::new(committed.clone(), &view, store.clone(), paths.clone())
            .with_cache(cache.clone());
        let probes = vec![sorted_node_id(1), sorted_node_id(6), sorted_node_id(99)];
        let got = snap.batch_lookup_nodes("Person", &probes).await.unwrap();

        let flat = Snapshot::new(committed.clone(), &view, store.clone(), paths.clone());
        for (i, id) in probes.iter().enumerate() {
            let want = flat.lookup_node_via_uncached("Person", *id).await.unwrap();
            assert_eq!(got[i], want, "probe #{i} diverged from the flat walk");
        }
        assert_eq!(
            cache.decoded_node_row_group_inserts(),
            1,
            "a single-row-group SST decodes exactly once"
        );
    }

    #[test]
    fn snapshot_registry_tracks_oldest_live_version() {
        // The retention horizon (RFC-027) is min_live(); it must reflect the
        // oldest version with at least one live holder and advance only when
        // every holder of that version releases.
        let reg = SnapshotRegistry::default();
        assert_eq!(reg.min_live(), None);

        reg.acquire(5);
        reg.acquire(7);
        reg.acquire(5);
        assert_eq!(reg.min_live(), Some(5));

        reg.release(5);
        assert_eq!(reg.min_live(), Some(5), "one holder of v5 remains");
        reg.release(5);
        assert_eq!(
            reg.min_live(),
            Some(7),
            "v5 fully released, v7 is now the oldest live version"
        );

        reg.release(7);
        assert_eq!(reg.min_live(), None, "no live readers");
    }

    /// Node payload carrying one string property under an explicit label id.
    fn coded_node_payload(prop: &str, value: &str, label_id: u32) -> Bytes {
        let mut props: BTreeMap<String, Value> = BTreeMap::new();
        props.insert(prop.into(), Value::Str(value.into()));
        NodeWriteRecord {
            properties: props,
            schema_version: 1,
            labels: vec![label_id],
        }
        .encode()
        .unwrap()
    }

    #[tokio::test]
    async fn unique_sidecar_fast_path_is_scoped_to_the_labels_ssts() {
        // Regression (finding 37): the sidecar-completeness check used to run
        // over EVERY node SST, so a different label's SST lacking an unrelated
        // sidecar demoted the lookup to a full label scan in any multi-label
        // deployment. The check must be scoped to SSTs that can contain the
        // label being probed.
        let store = make_store();
        let paths = make_paths("sidecar-scope");
        let ms = ManifestStore::new(store.clone(), paths.clone());
        let mut base = ms.bootstrap(Uuid::now_v7()).await.unwrap();
        let account_lid = base.manifest.label_dict.intern("Account").get();
        let widget_lid = base.manifest.label_dict.intern("Widget").get();
        let fence = WriterFence::new(base.manifest.epoch);
        let schema = SchemaBuilder::new().build();

        // Flush 1: two Account rows carrying the unique property `code`.
        let a1 = sorted_node_id(1);
        let a2 = sorted_node_id(2);
        let mut mt = Memtable::new();
        mt.apply(
            MemKey::Node { id: a1 },
            10,
            MemOp::Upsert(coded_node_payload("code", "a-1", account_lid)),
        );
        mt.apply(
            MemKey::Node { id: a2 },
            11,
            MemOp::Upsert(coded_node_payload("code", "a-2", account_lid)),
        );
        let out1 = flush(&ms, &fence, &base, &mt.freeze(), schema.clone())
            .await
            .unwrap();

        // Flush 2: one Widget row WITHOUT `code` — its SST carries no
        // sidecar for the property (and never could).
        let w1 = sorted_node_id(3);
        let mut mt2 = Memtable::new();
        mt2.apply(
            MemKey::Node { id: w1 },
            12,
            MemOp::Upsert(coded_node_payload("sku", "w-1", widget_lid)),
        );
        let out2 = flush(&ms, &fence, &out1.committed, &mt2.freeze(), schema)
            .await
            .unwrap();

        // Attach a unique-property sidecar to the Account SST, exactly as a
        // per-label build would have emitted it (the id-primary flush path
        // does not): a bincode `value → NodeId` map next to the body.
        let mut committed = out2.committed.clone();
        let account_sst = committed
            .manifest
            .ssts
            .iter()
            .position(|d| {
                d.kind == SstKind::Nodes
                    && d.label_index.as_ref().is_some_and(|li| {
                        li.per_label_counts
                            .iter()
                            .any(|(id, c)| *id == account_lid && *c > 0)
                    })
            })
            .expect("Account SST present");
        let mut sidecar: BTreeMap<String, [u8; 16]> = BTreeMap::new();
        sidecar.insert("a-1".into(), *a1.as_bytes());
        sidecar.insert("a-2".into(), *a2.as_bytes());
        let body = Bytes::from(bincode::serialize(&sidecar).unwrap());
        let relative = "sst/L0/fabricated.idx_code.bin".to_string();
        let absolute = format!("{}/{}", paths.namespace_prefix().as_ref(), relative);
        store
            .put(&object_store::path::Path::from(absolute), body.clone().into())
            .await
            .unwrap();
        committed.manifest.ssts[account_sst].unique_property_indices.push(
            crate::manifest::UniquePropertyIndexDescriptor {
                property: "code".into(),
                path: relative,
                size_bytes: body.len() as u64,
                entry_count: 2,
            },
        );

        let empty = Memtable::new();
        let empty_view = empty.snapshot_view();
        let cache = Arc::new(crate::property_index::PropertyIndexCache::new());
        let snap = Snapshot::new(committed, &empty_view, store.clone(), paths.clone())
            .with_property_index_cache(cache.clone());

        // The lookup resolves through the sidecar even though the Widget SST
        // has none for `code`.
        let hit = snap
            .lookup_node_by_property("Account", "code", "a-2")
            .await
            .unwrap();
        assert_eq!(hit.map(|v| v.id), Some(a2));
        // Path assertion: the legacy fallback populates the property-index
        // cache from its full label scan; the sidecar path never does.
        assert!(
            cache.get("Account", "code").is_none(),
            "lookup fell back to the full label scan — the sidecar check was \
             not scoped to the Account SSTs"
        );

        // A miss through the sidecar path is a definitive negative.
        assert!(snap
            .lookup_node_by_property("Account", "code", "zz")
            .await
            .unwrap()
            .is_none());
        assert!(cache.get("Account", "code").is_none());
    }

    #[tokio::test]
    async fn unique_lookup_for_memtable_only_label_skips_other_labels_ssts() {
        // A label that lives only in the memtable must resolve via the
        // memtable-side sidecar pass; another label's SSTs (which cannot
        // contain it) must neither be required to carry sidecars nor force
        // the full-scan fallback.
        let store = make_store();
        let paths = make_paths("sidecar-scope-mem");
        let ms = ManifestStore::new(store.clone(), paths.clone());
        let mut base = ms.bootstrap(Uuid::now_v7()).await.unwrap();
        let widget_lid = base.manifest.label_dict.intern("Widget").get();
        let fresh_lid = base.manifest.label_dict.intern("Fresh").get();
        let fence = WriterFence::new(base.manifest.epoch);
        let schema = SchemaBuilder::new().build();

        // Widget rows flushed to an SST with no sidecars.
        let w1 = sorted_node_id(1);
        let mut mt = Memtable::new();
        mt.apply(
            MemKey::Node { id: w1 },
            10,
            MemOp::Upsert(coded_node_payload("sku", "w-1", widget_lid)),
        );
        let out = flush(&ms, &fence, &base, &mt.freeze(), schema)
            .await
            .unwrap();

        // A Fresh row only in the live memtable.
        let f1 = sorted_node_id(2);
        let mut live = Memtable::new();
        live.apply(
            MemKey::Node { id: f1 },
            20,
            MemOp::Upsert(coded_node_payload("email", "f@x", fresh_lid)),
        );
        let live_view = live.snapshot_view();
        let cache = Arc::new(crate::property_index::PropertyIndexCache::new());
        let snap = Snapshot::new(out.committed, &live_view, store, paths)
            .with_property_index_cache(cache.clone());

        let hit = snap
            .lookup_node_by_property("Fresh", "email", "f@x")
            .await
            .unwrap();
        assert_eq!(hit.map(|v| v.id), Some(f1));
        assert!(
            cache.get("Fresh", "email").is_none(),
            "memtable-only label fell back to the full label scan"
        );
    }

    #[tokio::test]
    async fn equality_sidecar_fast_path_never_consults_other_labels_ssts() {
        // Same scoping for the non-unique equality index: plant an unreadable
        // equality-sidecar descriptor on the OTHER label's SST. If the lookup
        // consults that SST at all (pre-fix it made `all_have_sidecar` true
        // and probed it), the GET fails and the lookup errors; correctly
        // scoped, the SST is never touched.
        let store = make_store();
        let paths = make_paths("sidecar-scope-eq");
        let ms = ManifestStore::new(store.clone(), paths.clone());
        let mut base = ms.bootstrap(Uuid::now_v7()).await.unwrap();
        let person_lid = base.manifest.label_dict.intern("Person").get();
        let widget_lid = base.manifest.label_dict.intern("Widget").get();
        let fence = WriterFence::new(base.manifest.epoch);
        // `city` is declared indexed, so the Person flush emits an equality
        // sidecar for it.
        let schema = SchemaBuilder::new()
            .label(LabelDef {
                name: "Person".into(),
                properties: vec![PropertyDef::new("city", DataType::Utf8, true)
                    .unwrap()
                    .with_indexed(true)],
            })
            .unwrap()
            .build();

        let p1 = sorted_node_id(1);
        let p2 = sorted_node_id(2);
        let mut mt = Memtable::new();
        mt.apply(
            MemKey::Node { id: p1 },
            10,
            MemOp::Upsert(coded_node_payload("city", "Lisbon", person_lid)),
        );
        mt.apply(
            MemKey::Node { id: p2 },
            11,
            MemOp::Upsert(coded_node_payload("city", "Porto", person_lid)),
        );
        let out1 = flush(&ms, &fence, &base, &mt.freeze(), schema.clone())
            .await
            .unwrap();
        assert!(
            out1.committed.manifest.ssts.iter().any(|d| d
                .equality_property_indices
                .iter()
                .any(|e| e.property == "city")),
            "Person flush must emit the city equality sidecar"
        );

        let w1 = sorted_node_id(3);
        let mut mt2 = Memtable::new();
        mt2.apply(
            MemKey::Node { id: w1 },
            12,
            MemOp::Upsert(coded_node_payload("sku", "w-1", widget_lid)),
        );
        let out2 = flush(&ms, &fence, &out1.committed, &mt2.freeze(), schema)
            .await
            .unwrap();

        let mut committed = out2.committed.clone();
        let widget_sst = committed
            .manifest
            .ssts
            .iter()
            .position(|d| {
                d.kind == SstKind::Nodes
                    && d.label_index.as_ref().is_some_and(|li| {
                        li.per_label_counts
                            .iter()
                            .any(|(id, c)| *id == widget_lid && *c > 0)
                    })
            })
            .expect("Widget SST present");
        committed.manifest.ssts[widget_sst].equality_property_indices.push(
            crate::manifest::EqualityIndexDescriptor {
                property: "city".into(),
                path: "sst/L0/does-not-exist.eqidx_city.bin".into(),
                size_bytes: 1,
                distinct_values: 1,
            },
        );

        let empty = Memtable::new();
        let empty_view = empty.snapshot_view();
        let snap = Snapshot::new(committed, &empty_view, store, paths);

        let hits = snap
            .lookup_nodes_by_property("Person", "city", "Lisbon")
            .await
            .expect("the Widget SST must not be consulted for a Person lookup");
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].id, p1);
    }
}
