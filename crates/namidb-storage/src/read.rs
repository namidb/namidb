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

use std::collections::{BTreeMap, HashMap};
use std::sync::{Arc, Mutex};

use arrow_array::RecordBatch;
use arrow_array::{
    Array, BinaryArray, BooleanArray, Date32Array, FixedSizeBinaryArray, FixedSizeListArray,
    Float32Array, Float64Array, Int32Array, Int64Array, LargeStringArray, StringArray,
    TimestampMicrosecondArray, UInt64Array,
};
use bytes::Bytes;
use object_store::path::Path;
use object_store::{ObjectStore, ObjectStoreExt};
use tracing::instrument;
use uuid::Uuid;

use namidb_core::{DataType, LabelDef, NodeId, Value};

use crate::adjacency::{
    adjacency_enabled, build_adjacency, AdjacencyCache, AdjacencyKey, EdgeAdjacency,
};
use crate::cache::{EdgeStreamBundle, SstCache};
use crate::error::{Error, Result};
use crate::flush::{EdgeWriteRecord, NodeWriteRecord};
use crate::manifest::{LoadedManifest, SstDescriptor, SstKind};
use crate::memtable::{MemKey, MemOp, MemtableSnapshot};
use crate::node_cache::{NodeCacheKey, NodeViewCache};
use crate::paths::NamespacePaths;
use crate::sst::bloom::BloomFilter;
use crate::sst::edges::reader::EdgeSstReader;
use crate::sst::edges::EdgeDirection;
use crate::sst::nodes::{
    prop_column_name, targeted_scan_async as node_targeted_scan_async, NodeSstReader, COL_LSN,
    COL_NODE_ID, COL_TOMBSTONE, OVERFLOW_JSON, SCHEMA_VERSION,
};
use crate::sst::predicates::{eval_against_value, ScanPredicate};

/// Projection of a node row materialised by the read path.
#[derive(Debug, Clone, PartialEq)]
pub struct NodeView {
    pub id: NodeId,
    pub label: String,
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
    /// Intra-snapshot cache of decoded Parquet `RecordBatch`es keyed by
    /// the absolute SST path. Populated by [`Self::batch_lookup_nodes`]
    /// on the first probe of an SST; subsequent probes within the same
    /// snapshot reuse the decoded batches via `Arc::clone` instead of
    /// re-parsing the body. Amortises the SST decode cost across the N
    /// per-parent batch calls the factor-path executor issues during a
    /// 2-hop Expand chain (LDBC SNB IC09: 150+ calls/query).
    decoded_node_sst_batches: Mutex<HashMap<String, Arc<Vec<RecordBatch>>>>,
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
            decoded_node_sst_batches: Mutex::new(HashMap::new()),
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
        let sst_idxs: Vec<usize> = self
            .manifest
            .index
            .scope_descriptors(SstKind::Nodes, label)
            .to_vec();
        let all_have_sidecar = !sst_idxs.is_empty()
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
            for (mk, e) in self.memtable.iter() {
                if let MemKey::Node { label: mlabel, id } = mk {
                    if mlabel != label {
                        continue;
                    }
                    match &e.op {
                        MemOp::Upsert(payload) => {
                            let rec = NodeWriteRecord::decode(payload)?;
                            if let Some(namidb_core::Value::Str(s)) = rec.properties.get(property) {
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
                Some((_, id, false)) => self.lookup_node(label, id).await,
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
            if let MemKey::Node { label, id } = key {
                if matches!(entry.op, MemOp::Upsert(_)) {
                    mem_node_label.insert(*id, label.clone());
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
            let (src_label, dst_label) = match samples.get(&edge_type) {
                Some((src, dst)) => (
                    self.find_node_label(*src, &mem_node_label).await?,
                    self.find_node_label(*dst, &mem_node_label).await?,
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
        for (key, _) in self.memtable.iter() {
            if let MemKey::Node { label, .. } = key {
                set.insert(label.clone());
            }
        }
        for sst in &self.manifest.manifest.ssts {
            if matches!(sst.kind, SstKind::Nodes) {
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

        // L3: cold SST walk.
        let result = self.lookup_node_uncached(label, id).await?;
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
    /// decodes each candidate SST once total and matches all `ids`
    /// against the resulting `RecordBatch`es in one pass — turning a
    /// linear N×M ladder into one SST decode per `(label, scope)`.
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
            if let Some(entry) = self.memtable.get(&MemKey::Node {
                label: label.to_string(),
                id,
            }) {
                let view = match &entry.op {
                    MemOp::Tombstone => None,
                    MemOp::Upsert(payload) => Some(node_view_from_payload(
                        id,
                        label.to_string(),
                        entry.lsn,
                        payload,
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

        // 2. SST pass: iterate every `(Nodes, label)` descriptor exactly
        // once, decode its body, and harvest every pending id in one
        // sweep over the record batches.
        let label_def = self
            .manifest
            .manifest
            .schema
            .label(label)
            .cloned()
            .unwrap_or_else(|| LabelDef {
                name: label.to_string(),
                properties: vec![],
            });
        let sst_idxs: Vec<usize> = self
            .manifest
            .index
            .scope_descriptors(SstKind::Nodes, label)
            .to_vec();
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
            // Intra-snapshot decoded-batches cache: amortises the
            // per-call `reader.scan()` Parquet decode across the N
            // batch calls a factor-path Expand chain issues (one per
            // parent_leaf). Without this, SF1 IC09 cold pays the SST
            // decode ~150 times.
            let absolute = format!("{}/{}", self.paths.namespace_prefix().as_ref(), desc.path);
            // Probe the cache in a separate scope so the MutexGuard is
            // released before the await — futures-Send requires the guard
            // type itself to be Send + Sync, which `MutexGuard` isn't.
            let cached: Option<Arc<Vec<RecordBatch>>> = self
                .decoded_node_sst_batches
                .lock()
                .unwrap()
                .get(&absolute)
                .cloned();
            let batches: Arc<Vec<RecordBatch>> = if let Some(b) = cached {
                b
            } else {
                let body = self.get_sst_body(desc).await?;
                let reader = NodeSstReader::open(label_def.clone(), body)?;
                let decoded = Arc::new(reader.scan()?);
                // Re-acquire the lock to insert — last write wins on
                // a race because both threads decoded identical bytes.
                self.decoded_node_sst_batches
                    .lock()
                    .unwrap()
                    .insert(absolute.clone(), decoded.clone());
                decoded
            };
            batch_harvest_node_rows(&batches, &label_def, label, &pending, &mut winners)?;
        }

        // 3. Push every (resolved or negative) outcome into the output
        // vector and populate the cache tiers.
        let shared = self.shared_node_cache.clone();
        let manifest_version = self.manifest.manifest.version;
        let mut cache_l1 = self.node_cache.lock().unwrap();
        for id_bytes in &pending {
            let view = winners.remove(id_bytes).map(|(_, v)| v).unwrap_or(None);
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
        self.lookup_node_uncached(label, id).await
    }

    async fn lookup_node_uncached(&self, label: &str, id: NodeId) -> Result<Option<NodeView>> {
        namidb_core::profile_scope!("Snapshot::lookup_node_uncached");
        let id_bytes = *id.as_bytes();
        let mut winner: Option<(u64, Option<NodeView>)> = None;

        // 1. Memtable (highest LSN typically).
        if let Some(entry) = self.memtable.get(&MemKey::Node {
            label: label.to_string(),
            id,
        }) {
            let view = match &entry.op {
                MemOp::Tombstone => None,
                MemOp::Upsert(payload) => Some(node_view_from_payload(
                    id,
                    label.to_string(),
                    entry.lsn,
                    payload,
                )?),
            };
            winner = Some((entry.lsn, view));
        }

        // 2. SST candidates — pruned via the manifest's sorted-by-min-key
        // index. The index returns descriptor positions whose
        // `(min_key, max_key)` already straddles `id_bytes` for
        // `(Nodes, label)`; we still bloom-probe + body-fetch.
        let label_def = self
            .manifest
            .manifest
            .schema
            .label(label)
            .cloned()
            .unwrap_or_else(|| LabelDef {
                name: label.to_string(),
                properties: vec![],
            });
        let candidates = self.manifest.index.lookup_candidates(
            &self.manifest.manifest.ssts,
            SstKind::Nodes,
            label,
            &id_bytes,
        );
        for idx in candidates {
            let desc = &self.manifest.manifest.ssts[idx];
            if !self.bloom_admits(desc, &id_bytes).await? {
                continue;
            }
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
                find_node_row(&reader, &label_def, id, label)?
            } else {
                let absolute = format!("{}/{}", self.paths.namespace_prefix().as_ref(), desc.path);
                if let Some(body) = self.cache_get(&absolute) {
                    let reader = NodeSstReader::open(label_def.clone(), body)?;
                    find_node_row(&reader, &label_def, id, label)?
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
                    find_node_row_in_batches(&batches, &label_def, id, label)?
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
        let label_def = self
            .manifest
            .manifest
            .schema
            .label(label)
            .cloned()
            .unwrap_or_else(|| LabelDef {
                name: label.to_string(),
                properties: vec![],
            });

        // (node_id) → (winning lsn, materialised view or tombstone marker).
        let mut latest: BTreeMap<NodeId, (u64, Option<NodeView>)> = BTreeMap::new();

        // 1. Memtable rows for this label. Apply predicates after
        // materialising the view; if any predicate evaluates to
        // false / NULL, drop the view (kept as tombstone-like None
        // so subsequent SST upserts for the same id are not
        // spuriously surfaced).
        for (mk, entry) in self.memtable.iter() {
            let MemKey::Node { label: ml, id } = mk else {
                continue;
            };
            if ml != label {
                continue;
            }
            let view = match &entry.op {
                MemOp::Tombstone => None,
                MemOp::Upsert(payload) => {
                    let mut v = node_view_from_payload(*id, label.to_string(), entry.lsn, payload)?;
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

        // 2. SSTs that scope to `label`, taken straight from the
        // manifest index so we don't re-scan every descriptor.
        for &idx in self.manifest.index.scope_descriptors(SstKind::Nodes, label) {
            let desc = &self.manifest.manifest.ssts[idx];
            let body = self.get_sst_body(desc).await?;
            let reader = NodeSstReader::open(label_def.clone(), body)?;
            // Build the projection set once per SST (declared properties
            // ∩ requested). When `projection.is_none()` we iterate every
            // declared property.
            let projection_set: Option<std::collections::BTreeSet<&str>> =
                projection.map(|cols| cols.iter().map(|s| s.as_str()).collect());

            for batch in reader.scan_with_predicates_and_projection(predicates, projection)? {
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
                    for p in &label_def.properties {
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
                        label: label.to_string(),
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

        // 3. Drop tombstones and return in ascending-id order (BTreeMap iter).
        Ok(latest.into_values().filter_map(|(_, v)| v).collect())
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

        // 1. Memtable.
        for (mk, entry) in self.memtable.iter() {
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

        // Memtable sweep first; the SST/CSR path below shadows whatever
        // the memtable contributed only when its LSN is strictly higher.
        for (mk, entry) in self.memtable.iter() {
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

        // 1. Memtable.
        for (mk, entry) in self.memtable.iter() {
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

        // 2a. Memtable sweep — same shape as the SST path but unchanged
        // because the memtable is already in-RAM. Tombstones from
        // here can shadow CSR upserts of equal-or-lower LSN.
        for (mk, entry) in self.memtable.iter() {
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
        let absolute = format!("{}/{}", self.paths.namespace_prefix().as_ref(), desc.path);
        self.fetch_bytes(&absolute).await
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

fn node_view_from_payload(
    id: NodeId,
    label: String,
    lsn: u64,
    payload: &Bytes,
) -> Result<NodeView> {
    let rec = NodeWriteRecord::decode(payload)?;
    Ok(NodeView {
        id,
        label,
        properties: rec.properties,
        lsn,
        schema_version: rec.schema_version,
    })
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
    label: &str,
) -> Result<Option<(u64, Option<NodeView>)>> {
    let target_bytes = *target.as_bytes();
    let batches = reader.targeted_scan(&target_bytes)?;
    find_node_row_in_batches(&batches, label_def, target, label)
}

/// Backend-agnostic row search over already-decoded record batches.
/// Shared between the sync (cache-hit) and async (cold ranged-read)
/// paths so behavior stays bit-identical regardless of where the
/// batches came from.
fn find_node_row_in_batches(
    batches: &[RecordBatch],
    label_def: &LabelDef,
    target: NodeId,
    label: &str,
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
                    label: label.to_string(),
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
    label: &str,
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
                    label: label.to_string(),
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
}

impl SnapshotCell {
    pub fn new(snap: Arc<OwnedSnapshot>) -> Self {
        Self {
            inner: std::sync::Mutex::new(snap),
        }
    }

    /// Pick up the current snapshot. Cheap: one mutex acquire plus
    /// one `Arc::clone`. The returned `Arc` is independent of any
    /// future `store` calls, so the read can run for as long as it
    /// needs without holding any cell lock.
    pub fn load(&self) -> Arc<OwnedSnapshot> {
        Arc::clone(&self.inner.lock().expect("snapshot cell poisoned"))
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
        let base = ms.bootstrap(Uuid::now_v7()).await.unwrap();
        let fence = WriterFence::new(base.manifest.epoch);
        let schema = SchemaBuilder::new().label(person_label()).unwrap().build();

        let alice = sorted_node_id(1);
        let mut mt = Memtable::new();
        mt.apply(
            MemKey::Node {
                label: "Person".into(),
                id: alice,
            },
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
    async fn lookup_node_falls_back_to_memtable_when_not_flushed() {
        let store = make_store();
        let paths = make_paths("read-mt");
        let ms = ManifestStore::new(store.clone(), paths.clone());
        let base = ms.bootstrap(Uuid::now_v7()).await.unwrap();

        let alice = sorted_node_id(2);
        let mut mt = Memtable::new();
        mt.apply(
            MemKey::Node {
                label: "Person".into(),
                id: alice,
            },
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
            MemKey::Node {
                label: "Person".into(),
                id: alice,
            },
            10,
            MemOp::Upsert(node_payload("Alice", Some(30))),
        );
        let frozen = mt_flush.freeze();
        let outcome = flush(&ms, &fence, &base, &frozen, schema.clone())
            .await
            .unwrap();

        // Live memtable now carries a tombstone at LSN 15 (> SST's LSN 10).
        let mut live_mt = Memtable::new();
        live_mt.apply(
            MemKey::Node {
                label: "Person".into(),
                id: alice,
            },
            15,
            MemOp::Tombstone,
        );

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
        let base = ms.bootstrap(Uuid::now_v7()).await.unwrap();
        let fence = WriterFence::new(base.manifest.epoch);
        let schema = SchemaBuilder::new().label(person_label()).unwrap().build();

        let id_low = sorted_node_id(1);
        let id_high = sorted_node_id(200);

        let mut mt1 = Memtable::new();
        mt1.apply(
            MemKey::Node {
                label: "Person".into(),
                id: id_low,
            },
            1,
            MemOp::Upsert(node_payload("Low", None)),
        );
        let frozen1 = mt1.freeze();
        let after1 = flush(&ms, &fence, &base, &frozen1, schema.clone())
            .await
            .unwrap();

        let mut mt2 = Memtable::new();
        mt2.apply(
            MemKey::Node {
                label: "Person".into(),
                id: id_high,
            },
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
        let base = ms.bootstrap(Uuid::now_v7()).await.unwrap();
        let fence = WriterFence::new(base.manifest.epoch);
        let schema = SchemaBuilder::new().label(person_label()).unwrap().build();

        let alice = sorted_node_id(1);
        let mut mt = Memtable::new();
        mt.apply(
            MemKey::Node {
                label: "Person".into(),
                id: alice,
            },
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
        let base = ms.bootstrap(Uuid::now_v7()).await.unwrap();
        let fence = WriterFence::new(base.manifest.epoch);
        let schema = SchemaBuilder::new().label(person_label()).unwrap().build();

        // Flush three nodes at LSNs 1..3.
        let alice = sorted_node_id(1);
        let bob = sorted_node_id(2);
        let carol = sorted_node_id(3);
        let mut mt_flush = Memtable::new();
        for (i, id) in [(1u64, alice), (2, bob), (3, carol)] {
            mt_flush.apply(
                MemKey::Node {
                    label: "Person".into(),
                    id,
                },
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
            MemKey::Node {
                label: "Person".into(),
                id: alice,
            },
            10,
            MemOp::Upsert(node_payload("Alice-updated", Some(99))),
        );
        live.apply(
            MemKey::Node {
                label: "Person".into(),
                id: bob,
            },
            11,
            MemOp::Tombstone,
        );
        live.apply(
            MemKey::Node {
                label: "Person".into(),
                id: dave,
            },
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
        let base = ms.bootstrap(Uuid::now_v7()).await.unwrap();
        let fence = WriterFence::new(base.manifest.epoch);

        let alice = sorted_node_id(5);

        // Append a WAL segment containing an upsert for Alice.
        let entry = WalEntry {
            key: MemKey::Node {
                label: "Person".into(),
                id: alice,
            },
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
        let base = ms.bootstrap(Uuid::now_v7()).await.unwrap();

        // Two nodes with distinct labels, one edge that ties them
        // together, no `SchemaBuilder` ever ran.
        let person = sorted_node_id(1);
        let company = sorted_node_id(2);
        let mut mt = Memtable::new();
        mt.apply(
            MemKey::Node {
                label: "Person".into(),
                id: person,
            },
            1,
            MemOp::Upsert(node_payload("Alice", None)),
        );
        mt.apply(
            MemKey::Node {
                label: "Company".into(),
                id: company,
            },
            2,
            MemOp::Upsert(node_payload("Acme", None)),
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
            MemKey::Node {
                label: "Person".into(),
                id: alice,
            },
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
}
