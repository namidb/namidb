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
use std::hash::Hash;
#[cfg(any(feature = "text-index", feature = "vector-index"))]
use std::ops::Range;
use std::sync::{Arc, Mutex};

use arrow_array::RecordBatch;
use arrow_array::{
    Array, BinaryArray, BooleanArray, Date32Array, FixedSizeBinaryArray, FixedSizeListArray,
    Float32Array, Float64Array, Int32Array, Int64Array, LargeStringArray, ListArray, StringArray,
    TimestampMicrosecondArray, UInt32Array, UInt64Array,
};
use bytes::Bytes;
use object_store::path::Path;
#[cfg(any(feature = "text-index", feature = "vector-index"))]
use object_store::ObjectMeta;
use object_store::{ObjectStore, ObjectStoreExt};
use parquet::file::metadata::ParquetMetaData;
use tracing::instrument;
use uuid::Uuid;

use namidb_core::{DataType, LabelDef, LabelDictionary, LabelId, NodeId, Value};

use crate::adjacency::{
    adjacency_enabled, build_adjacency, AdjacencyCache, AdjacencyKey, EdgeAdjacency,
};
use crate::cache::{
    DecodedNodeRowGroup, EqualityPropertySidecar, NodeRowGroupKey, SstCache, UniquePropertySidecar,
};
use crate::error::{Error, Result};
use crate::flush::{decode_exact_node_record, EdgeWriteRecord, NodeWriteRecord};
#[cfg(any(feature = "text-index", feature = "vector-index"))]
use crate::manifest::KindSpecificStats;
use crate::manifest::{
    EqualityIndexDescriptor, LoadedManifest, Manifest, PropertyIndexFormat, SstDescriptor, SstKind,
    UniquePropertyIndexDescriptor,
};
use crate::memtable::{MemEntry, MemKey, MemOp, MemtableSnapshot};
use crate::node_cache::{NodeCacheKey, NodeViewCache};
use crate::paths::NamespacePaths;
#[cfg(any(feature = "text-index", feature = "vector-index"))]
use crate::search_lsm::{
    select_search_read_plan, validate_search_barrier, SearchLsmKind, SearchReadPlan,
    SearchSegmentRef, SearchSegmentStats, SearchStatValue,
};
use crate::sst::bloom::BloomFilter;
use crate::sst::edges::format::OVERFLOW_JSON_NAME;
use crate::sst::edges::paged_reader::PagedEdgeReader;
use crate::sst::edges::reader::EdgeSstReader;
use crate::sst::edges::EdgeDirection;
use crate::sst::nodes::property_pages::{
    NodePropertyPageConfig, NodePropertyPageReader, PropertyCell,
    NODE_PROPERTY_PAGES_FORMAT_VERSION,
};
use crate::sst::nodes::{
    load_node_sst_metadata_async, parse_node_sst_metadata, prop_column_name, row_groups_for_keys,
    scan_row_groups_async as node_scan_row_groups_async,
    scan_row_groups_for_keys_async as node_scan_row_groups_for_keys_async,
    scan_rows_by_ordinals_async as node_scan_rows_by_ordinals_async,
    scan_with_predicates_and_projection_async as node_scan_limited_async,
    split_batches_by_row_group, targeted_scan_async as node_targeted_scan_async, NodeSstReader,
    COL_LABELS, COL_LSN, COL_NODE_ID, COL_TOMBSTONE, OVERFLOW_JSON, SCHEMA_VERSION,
};
use crate::sst::predicates::{eval_against_value, ScanPredicate};

#[cfg(any(feature = "text-index", feature = "vector-index"))]
#[path = "search_lsm_read.rs"]
mod search_lsm_read;

const SNAPSHOT_ROW_GROUP_CACHE_MAX_BYTES_ENV: &str = "NAMIDB_SNAPSHOT_ROW_GROUP_CACHE_MAX_BYTES";
const SNAPSHOT_EDGE_READER_CACHE_MAX_BYTES_ENV: &str =
    "NAMIDB_SNAPSHOT_EDGE_READER_CACHE_MAX_BYTES";
const SNAPSHOT_NODE_PROPERTY_READER_CACHE_MAX_BYTES_ENV: &str =
    "NAMIDB_SNAPSHOT_NODE_PROPERTY_READER_CACHE_MAX_BYTES";
const DEFAULT_SNAPSHOT_LOCAL_CACHE_MAX_BYTES: usize = 1024 * 1024;
const SNAPSHOT_ROW_GROUP_CACHE_MAX_ENTRIES: usize = 2;
const SNAPSHOT_EDGE_READER_CACHE_MAX_ENTRIES: usize = 32;
const SNAPSHOT_NODE_PROPERTY_READER_CACHE_MAX_ENTRIES: usize = 32;
const SNAPSHOT_CACHE_ENTRY_OVERHEAD_BYTES: usize = 256;

/// Small byte- and count-bounded cache owned by one immutable query snapshot.
///
/// Process-wide caches use Foyer and richer admission. Snapshot-local caches
/// only avoid repeating metadata/row-group work inside one query, but they
/// still need hard ceilings: query concurrency otherwise multiplies an
/// apparently harmless unbounded `HashMap`. The deliberately tiny entry caps
/// also make the O(entries) LRU victim selection bounded.
#[derive(Debug)]
struct SnapshotByteCache<K, V> {
    entries: HashMap<K, SnapshotByteCacheEntry<V>>,
    used_bytes: usize,
    capacity_bytes: usize,
    max_entries: usize,
    clock: u64,
}

#[derive(Debug)]
struct SnapshotByteCacheEntry<V> {
    value: V,
    weight: usize,
    touched: u64,
}

impl<K, V> SnapshotByteCache<K, V>
where
    K: Eq + Hash + Clone,
    V: Clone,
{
    fn new(capacity_bytes: usize, max_entries: usize) -> Self {
        Self {
            entries: HashMap::new(),
            used_bytes: 0,
            capacity_bytes,
            max_entries,
            clock: 0,
        }
    }

    fn get(&mut self, key: &K) -> Option<V> {
        let entry = self.entries.get_mut(key)?;
        self.clock = self.clock.saturating_add(1);
        entry.touched = self.clock;
        Some(entry.value.clone())
    }

    fn insert(&mut self, key: K, value: V, estimated_bytes: usize) {
        let weight = estimated_bytes.max(SNAPSHOT_CACHE_ENTRY_OVERHEAD_BYTES);
        if self.capacity_bytes == 0 || self.max_entries == 0 || weight > self.capacity_bytes {
            return;
        }
        if let Some(previous) = self.entries.remove(&key) {
            self.used_bytes = self.used_bytes.saturating_sub(previous.weight);
        }
        while self.entries.len() >= self.max_entries
            || self.used_bytes.saturating_add(weight) > self.capacity_bytes
        {
            let Some(victim) = self
                .entries
                .iter()
                .min_by_key(|(_, entry)| entry.touched)
                .map(|(key, _)| key.clone())
            else {
                break;
            };
            if let Some(previous) = self.entries.remove(&victim) {
                self.used_bytes = self.used_bytes.saturating_sub(previous.weight);
            }
        }
        self.clock = self.clock.saturating_add(1);
        self.used_bytes = self.used_bytes.saturating_add(weight);
        self.entries.insert(
            key,
            SnapshotByteCacheEntry {
                value,
                weight,
                touched: self.clock,
            },
        );
    }

    #[cfg(test)]
    fn used_bytes(&self) -> usize {
        self.used_bytes
    }

    #[cfg(test)]
    fn len(&self) -> usize {
        self.entries.len()
    }
}

fn snapshot_local_cache_max_bytes(name: &str) -> usize {
    match std::env::var(name) {
        Ok(value) => match value.trim().parse::<usize>() {
            Ok(bytes) => bytes,
            Err(error) => {
                tracing::warn!(
                    name,
                    value,
                    %error,
                    "disabling snapshot-local cache: setting must be an exact byte count"
                );
                0
            }
        },
        Err(std::env::VarError::NotPresent) => DEFAULT_SNAPSHOT_LOCAL_CACHE_MAX_BYTES.min(
            crate::cache_budget::cache_max_bytes()
                .checked_div(64)
                .unwrap_or(0),
        ),
        Err(std::env::VarError::NotUnicode(_)) => {
            tracing::warn!(
                name,
                "disabling snapshot-local cache: setting is not valid UTF-8"
            );
            0
        }
    }
}

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

/// Result of probing native metadata postings inside a vector `.vg`.
///
/// `Unsupported` is not an empty result: it means the body predates v4 or none
/// of the requested properties survived the bounded posting build, so callers
/// must retain their capped-sidecar/residual fallback.
#[cfg(feature = "vector-index")]
#[derive(Debug)]
pub enum VectorFilterSearch {
    Applied {
        hits: Vec<(NodeId, f32)>,
        point_count: u64,
        eligible_count: usize,
    },
    Unsupported,
}

#[cfg(any(feature = "text-index", feature = "vector-index"))]
#[derive(Debug)]
struct SearchObjectRangeSource {
    pinned: crate::range_cache::PinnedObjectRangeSource,
}

#[cfg(any(feature = "text-index", feature = "vector-index"))]
impl SearchObjectRangeSource {
    async fn new(store: Arc<dyn ObjectStore>, meta: ObjectMeta) -> Result<Self> {
        // Search SST names are create-only UUID paths. Prefer a backend
        // version/ETag precondition; local adapters that expose neither use
        // the common source's explicit immutable-path contract.
        let pinned =
            crate::range_cache::PinnedObjectRangeSource::from_create_only_meta(store, meta).await?;
        Ok(Self { pinned })
    }

    fn file_len(&self) -> u64 {
        self.pinned.object_size()
    }

    async fn read(&self, range: Range<u64>) -> Result<Bytes> {
        self.pinned.read_range(range).await
    }
}

#[cfg(feature = "text-index")]
#[async_trait::async_trait]
impl crate::sst::text::TextIndexRangeSource for SearchObjectRangeSource {
    async fn read_range(&self, range: Range<u64>) -> Result<Bytes> {
        self.read(range).await
    }

    async fn read_ranges(&self, ranges: &[Range<u64>]) -> Result<Vec<Bytes>> {
        self.pinned.read_ranges(ranges).await
    }
}

#[cfg(feature = "vector-index")]
#[async_trait::async_trait]
impl crate::sst::vector::v5::VectorV5RangeSource for SearchObjectRangeSource {
    async fn read_range(&self, range: Range<u64>) -> Result<Bytes> {
        self.read(range).await
    }

    async fn read_ranges(&self, ranges: &[Range<u64>]) -> Result<Vec<Bytes>> {
        self.pinned.read_ranges(ranges).await
    }
}

#[cfg(any(feature = "text-index", feature = "vector-index"))]
#[async_trait::async_trait]
impl crate::sst::search_delta::SearchVersionRangeSource for SearchObjectRangeSource {
    async fn read_range(&self, range: Range<u64>) -> Result<Bytes> {
        self.read(range).await
    }

    async fn read_ranges(&self, ranges: &[Range<u64>]) -> Result<Vec<Bytes>> {
        self.pinned.read_ranges(ranges).await
    }
}

/// `(doc_count, term_count, total_len, min_node_id, max_node_id)` of a
/// range-readable text base.
#[cfg(feature = "text-index")]
type RangedTextMetadata = (u64, u64, u64, [u8; 16], [u8; 16]);

#[cfg(feature = "text-index")]
#[derive(Debug)]
enum TextSearchIndex {
    Legacy(Arc<crate::sst::text::TextIndex>),
    Ranged(Arc<crate::sst::text::TextIndexV3Reader>),
}

#[cfg(feature = "text-index")]
impl TextSearchIndex {
    async fn contains_any_doc(&self, ids: &[[u8; 16]]) -> Result<bool> {
        match self {
            Self::Legacy(index) => Ok(ids.iter().any(|id| index.contains_doc(id))),
            Self::Ranged(index) => index.contains_any_doc(ids).await,
        }
    }

    async fn search_query(
        &self,
        query: &crate::text::TextQuery,
        k: Option<usize>,
    ) -> Result<Vec<([u8; 16], f64)>> {
        match self {
            Self::Legacy(index) => {
                use crate::search_workspace::{
                    search_max_result_bytes, search_max_text_result_hits, shared_search_workspace,
                    MATERIALISED_TEXT_RESULT_BYTES_PER_HIT,
                };

                // V2 is retained for in-place compatibility, but its postings
                // are monolithic and scoring accumulates one map entry per
                // potentially matching document. Admit that worst case
                // explicitly so a legacy snapshot cannot bypass the same
                // process-wide transient-memory ceiling as V3.
                const SCORE_MAP_BYTES_PER_DOC: usize = 64;
                // Phrase evaluation can briefly retain the prior allowed set,
                // the current phrase set, and their intersection.
                const PHRASE_SET_BYTES_PER_DOC: usize = 96;
                let documents = usize::try_from(index.doc_count()).unwrap_or(usize::MAX);
                let retained_hits = match k {
                    Some(limit) => limit.min(documents),
                    None => search_max_text_result_hits()
                        .saturating_add(1)
                        .min(documents),
                };
                let per_doc = SCORE_MAP_BYTES_PER_DOC.saturating_add(if query.phrases.is_empty() {
                    0
                } else {
                    PHRASE_SET_BYTES_PER_DOC
                });
                let required_bytes = documents.saturating_mul(per_doc).saturating_add(
                    retained_hits.saturating_mul(MATERIALISED_TEXT_RESULT_BYTES_PER_HIT),
                );
                let _workspace = shared_search_workspace()
                    .reserve("legacy full-text search", required_bytes)
                    .await?;

                let effective_k = k.or(Some(retained_hits));
                let hits = index.search_query(query, effective_k);
                if k.is_none() {
                    let maximum_hits = search_max_text_result_hits();
                    if hits.len() > maximum_hits {
                        return Err(Error::SearchResultLimitExceeded {
                            index_kind: "full-text",
                            estimated_bytes: hits
                                .len()
                                .saturating_mul(MATERIALISED_TEXT_RESULT_BYTES_PER_HIT),
                            limit_bytes: search_max_result_bytes(),
                        });
                    }
                }
                Ok(hits)
            }
            Self::Ranged(index) => index.search_query(query, k).await,
        }
    }

    /// `(doc_count, term_count, total_len, min_node_id, max_node_id)` of a
    /// range-readable text base.
    fn ranged_metadata(&self) -> Option<RangedTextMetadata> {
        let Self::Ranged(index) = self else {
            return None;
        };
        let (min_node_id, max_node_id) = index.node_id_bounds();
        Some((
            index.doc_count(),
            index.term_count(),
            index.total_len(),
            min_node_id,
            max_node_id,
        ))
    }
}

#[cfg(feature = "vector-index")]
#[derive(Debug)]
enum VectorSearchIndex {
    Legacy(Arc<crate::sst::vector::VectorGraphIndex>),
    Ranged(Arc<crate::sst::vector::v5::VectorV5Reader>),
}

#[cfg(any(feature = "text-index", feature = "vector-index"))]
#[derive(Debug)]
struct SelectedSearchBase {
    descriptor_index: usize,
    active_segment: Option<SearchSegmentRef>,
}

#[cfg(feature = "vector-index")]
impl VectorSearchIndex {
    fn point_count(&self) -> u64 {
        match self {
            Self::Legacy(index) => index.point_count(),
            Self::Ranged(index) => index.point_count(),
        }
    }

    fn higher_is_better(&self) -> bool {
        match self {
            Self::Legacy(index) => index.higher_is_better(),
            Self::Ranged(index) => {
                !matches!(index.metric(), crate::manifest::VectorMetric::Euclidean)
            }
        }
    }

    async fn search(&self, query: &[f32], k: usize, ef: usize) -> Result<Vec<([u8; 16], f32)>> {
        match self {
            Self::Legacy(index) => Ok(index.search(query, k, ef)),
            Self::Ranged(index) => {
                index
                    .search(query, k, vector_v5_search_options(index, ef))
                    .await
            }
        }
    }
}

#[cfg(feature = "vector-index")]
fn vector_v5_search_options(
    index: &crate::sst::vector::v5::VectorV5Reader,
    ef: usize,
) -> crate::sst::vector::v5::VectorV5SearchOptions {
    let page_count = index.page_count().max(1);
    // `ef` remains the public accuracy knob. Four coarse leaves is the cold
    // minimum; widening eventually reaches every page when `ef` approaches
    // the corpus, preserving the executor's exact-fallback/exhaustion logic.
    let nprobe = ef.div_ceil(64).max(4).min(page_count);
    crate::sst::vector::v5::VectorV5SearchOptions {
        nprobe,
        max_nprobe: nprobe.saturating_mul(8).max(nprobe).min(page_count),
        rerank_factor: 8,
    }
}

#[cfg(feature = "vector-index")]
fn active_vector_base_matches(
    index: &VectorSearchIndex,
    descriptor: &SstDescriptor,
    segment: &SearchSegmentRef,
) -> bool {
    let VectorSearchIndex::Ranged(index) = index else {
        return false;
    };
    let KindSpecificStats::VectorGraph {
        dim,
        metric,
        point_count,
        ..
    } = &descriptor.kind_specific
    else {
        return false;
    };
    let expected_metric = match index.metric() {
        crate::manifest::VectorMetric::Cosine => "cosine",
        crate::manifest::VectorMetric::Dot => "dot",
        crate::manifest::VectorMetric::Euclidean => "euclidean",
    };
    let (min_node_id, max_node_id) = index.node_id_bounds();
    let stats_match = matches!(
        segment.stats,
        SearchSegmentStats::Vector {
            live_count: SearchStatValue::Absolute(count)
        } if count == index.point_count()
    );
    *dim == index.dim()
        && metric == expected_metric
        && *point_count == index.point_count()
        && descriptor.row_count == index.point_count()
        && descriptor.min_key == min_node_id
        && descriptor.max_key == max_node_id
        && stats_match
        && segment
            .complete_filter_properties
            .iter()
            .all(|property| index.supports_filter_property(property))
        && segment.content_xxh3
            == crate::search_lsm::legacy_base_content_fingerprint(
                descriptor,
                segment.format,
                &segment.complete_filter_properties,
            )
}

#[cfg(feature = "text-index")]
fn active_text_base_matches(
    index: &TextSearchIndex,
    descriptor: &SstDescriptor,
    segment: &SearchSegmentRef,
) -> bool {
    let Some((doc_count, term_count, total_len, min_node_id, max_node_id)) =
        index.ranged_metadata()
    else {
        return false;
    };
    let KindSpecificStats::TextIndex {
        doc_count: descriptor_docs,
        term_count: descriptor_terms,
        total_len: descriptor_len,
    } = &descriptor.kind_specific
    else {
        return false;
    };
    let stats_match = matches!(
        segment.stats,
        SearchSegmentStats::Text {
            doc_count: SearchStatValue::Absolute(docs),
            total_len: SearchStatValue::Absolute(len),
            term_df_violation_count: 0,
        } if docs == doc_count && len == total_len
    );
    *descriptor_docs == doc_count
        && *descriptor_terms == term_count
        && *descriptor_len == total_len
        && descriptor.row_count == doc_count
        && descriptor.min_key == min_node_id
        && descriptor.max_key == max_node_id
        && segment.complete_filter_properties.is_empty()
        && stats_match
        && segment.content_xxh3
            == crate::search_lsm::legacy_base_content_fingerprint(
                descriptor,
                segment.format,
                &segment.complete_filter_properties,
            )
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

enum EdgePointWinner {
    Tombstone,
    Materialized(BTreeMap<String, Value>),
    Persisted {
        absolute: String,
        edge_offset: usize,
    },
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
    /// `paths.namespace_prefix()` rendered once — the namespace component
    /// stamped into every [`NodeCacheKey`] / [`AdjacencyKey`] this snapshot
    /// builds, so entries in the process-wide shared caches never collide
    /// across namespaces. `Arc<str>` so per-key clones are pointer-cheap.
    cache_namespace: Arc<str>,
    cache: Option<SstCache>,
    /// Byte-bounded per-snapshot NodeView cache. Many queries access the same node
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
    /// [`NodeViewCache`] keeps both positive and negative entries under
    /// `NAMIDB_SNAPSHOT_NODE_CACHE_MAX_BYTES`; every key includes namespace
    /// and logical generation just like the shared L2.
    node_cache: NodeViewCache,
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
    /// Slot key is `(logical_node_generation, label, NodeId)`. The generation
    /// comes from the writer's property-index cache and therefore advances on
    /// node mutations while remaining stable across edge-only commits and
    /// physical flushes. Caches both positive (`Some(view)`) and negative
    /// (`None`) outcomes.
    shared_node_cache: Option<Arc<NodeViewCache>>,
    /// Cross-snapshot lazy index over `(label, property) → value → NodeId`
    /// (RFC-pending). Attached via [`Self::with_property_index_cache`].
    /// `Snapshot::lookup_node_by_property` populates it on first miss
    /// and reuses it for the warm-path point lookups.
    property_index_cache: Option<Arc<crate::property_index::PropertyIndexCache>>,
    /// Cache generation captured with this immutable snapshot. Entries built
    /// by an older pinned reader cannot become visible to a newer snapshot.
    property_index_generation: Option<u64>,
    /// Snapshot-stable exact total/per-label node counts. Unlike the
    /// reconstructible property maps, this cell is retained by pinned
    /// snapshots across cache-pressure resets and replaced only for a logical
    /// node commit. A cold overlapping generation may populate it lazily once.
    exact_node_counts: Option<Arc<crate::property_index::ExactNodeCountCell>>,
    /// Physical committed-memtable claimant generation pinned with this
    /// snapshot. A flush replaces this cell independently of the logical node
    /// generation, so old/new snapshots cannot populate each other's physical
    /// delta index.
    memtable_claimant_cell: Option<Arc<crate::property_index::MemtableClaimantCell>>,
    /// Writer-private transactional property index. Attached only to
    /// [`crate::ingest::WriterSession::overlay_snapshot`], never to published
    /// reader snapshots, so committed + staged postings can be reused without
    /// leaking an uncommitted value change to concurrent readers.
    transactional_property_index: Option<&'mt crate::unique_index::UniqueConstraintIndex>,
    /// Whether the overlay already contains staged node mutations. A map first
    /// populated from such a view has no committed baseline and must be removed
    /// (rather than incrementally restored) if the batch rolls back.
    transactional_property_index_staged: bool,
    /// Per-snapshot fallback for decoded node-SST row groups, keyed by
    /// `(absolute SST path, row-group index)`. Used by
    /// [`Self::batch_lookup_nodes`] ONLY when no process-wide [`SstCache`]
    /// is attached; with a cache attached, decoded row groups live in the
    /// byte-budgeted `SstCache` tier and are shared across snapshots.
    /// This fallback is additionally capped by
    /// `NAMIDB_SNAPSHOT_ROW_GROUP_CACHE_MAX_BYTES` and two entries. An
    /// oversized decoded group remains valid for the current operation but is
    /// not retained.
    decoded_node_row_groups: Mutex<SnapshotByteCache<NodeRowGroupKey, DecodedNodeRowGroup>>,
    /// Metadata-only range readers opened during this snapshot. Immutable data
    /// pages themselves are shared process-wide by the RAM/NVMe range cache;
    /// this small cache only avoids repeating HEAD/footer parsing inside one
    /// batched graph operation. It is capped by
    /// `NAMIDB_SNAPSHOT_EDGE_READER_CACHE_MAX_BYTES` and 32 readers.
    paged_edge_readers: Mutex<SnapshotByteCache<String, Arc<PagedEdgeReader>>>,
    /// Decoded node-property catalogs only; page indexes and value payloads
    /// remain in the generation-pinned shared range cache. Both byte and entry
    /// ceilings prevent high query concurrency from multiplying metadata RAM.
    node_property_readers: Mutex<SnapshotByteCache<String, Arc<NodePropertyPageReader>>>,
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
    /// merge it via the range-pruned edge memtable iterators (RFC-026
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
        #[cfg(debug_assertions)]
        manifest
            .index
            .debug_assert_consistent(&manifest.manifest.ssts);
        let cache_namespace: Arc<str> = Arc::from(paths.namespace_prefix().as_ref());
        Self {
            manifest,
            memtable,
            store,
            paths,
            cache_namespace,
            cache: None,
            node_cache: NodeViewCache::new(crate::node_cache::snapshot_node_cache_max_bytes()),
            ranged_mode: RangedMode::Auto,
            ranged_threshold_bytes: DEFAULT_RANGED_THRESHOLD_BYTES,
            adjacency_cache: None,
            shared_node_cache: None,
            property_index_cache: None,
            property_index_generation: None,
            exact_node_counts: None,
            memtable_claimant_cell: None,
            transactional_property_index: None,
            transactional_property_index_staged: false,
            decoded_node_row_groups: Mutex::new(SnapshotByteCache::new(
                snapshot_local_cache_max_bytes(SNAPSHOT_ROW_GROUP_CACHE_MAX_BYTES_ENV),
                SNAPSHOT_ROW_GROUP_CACHE_MAX_ENTRIES,
            )),
            paged_edge_readers: Mutex::new(SnapshotByteCache::new(
                snapshot_local_cache_max_bytes(SNAPSHOT_EDGE_READER_CACHE_MAX_BYTES_ENV),
                SNAPSHOT_EDGE_READER_CACHE_MAX_ENTRIES,
            )),
            node_property_readers: Mutex::new(SnapshotByteCache::new(
                snapshot_local_cache_max_bytes(SNAPSHOT_NODE_PROPERTY_READER_CACHE_MAX_BYTES_ENV),
                SNAPSHOT_NODE_PROPERTY_READER_CACHE_MAX_ENTRIES,
            )),
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
        let (generation, counts, claimants) = cache.snapshot_generation_and_cells();
        self.exact_node_counts = Some(counts);
        self.memtable_claimant_cell = Some(claimants);
        self.property_index_generation = Some(generation);
        self.property_index_cache = Some(cache);
        self
    }

    pub(crate) fn with_property_index_cache_generation(
        mut self,
        cache: Arc<crate::property_index::PropertyIndexCache>,
        generation: u64,
    ) -> Self {
        self.exact_node_counts = cache.node_count_cell_at(generation);
        self.memtable_claimant_cell = Some(cache.current_memtable_claimant_cell());
        self.property_index_generation = Some(generation);
        self.property_index_cache = Some(cache);
        self
    }

    /// Generation used by the semantic NodeView cache. Writer snapshots carry
    /// the logical property-index generation; standalone snapshots fall back
    /// to the manifest version, preserving the conservative legacy behaviour.
    fn node_cache_generation(&self) -> u64 {
        self.property_index_generation
            .unwrap_or(self.manifest.manifest.version)
    }

    fn node_cache_key(&self, label: &str, node_id: NodeId) -> NodeCacheKey {
        NodeCacheKey {
            namespace: self.cache_namespace.clone(),
            manifest_version: self.node_cache_generation(),
            label: label.to_string(),
            node_id,
        }
    }

    /// Attach the writer-private committed+staged postings index.
    ///
    /// This hook is intentionally crate-private: only a writer overlay may
    /// expose the index. Published/read-only snapshots keep using the
    /// generation-scoped committed cache and immutable SST sidecars.
    pub(crate) fn with_transactional_property_index(
        mut self,
        index: &'mt crate::unique_index::UniqueConstraintIndex,
        staged_node_mutations: bool,
    ) -> Self {
        self.transactional_property_index = Some(index);
        self.transactional_property_index_staged = staged_node_mutations;
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

    /// Memtable entries for one edge type, merging the committed view with a
    /// staged overlay. `MemKey` is ordered by `(kind, edge_type, src, dst)`,
    /// so this is a tight range and never walks buffered node rows or other
    /// edge types.
    fn edge_mem_entries_for_type<'a>(
        &'a self,
        edge_type: &'a str,
    ) -> impl Iterator<Item = (&'a MemKey, &'a MemEntry)> + 'a {
        self.memtable.iter_edge_type(edge_type).chain(
            self.overlay
                .iter()
                .flat_map(move |o| o.iter_edge_type(edge_type)),
        )
    }

    /// Memtable entries that can affect one adjacency probe.
    ///
    /// Forward expansion is the bulk-loader hot path and maps exactly to the
    /// ordered `(edge_type, src)` prefix. Inverse probes cannot use that
    /// ordering directly, so they at least stay inside the edge-type range
    /// before filtering by destination.
    fn edge_mem_entries_for_key<'a>(
        &'a self,
        edge_type: &'a str,
        key: NodeId,
        direction: EdgeDirection,
    ) -> Box<dyn Iterator<Item = (&'a MemKey, &'a MemEntry)> + 'a> {
        match direction {
            EdgeDirection::Forward => Box::new(
                self.memtable.iter_out_edges(edge_type, key).chain(
                    self.overlay
                        .iter()
                        .flat_map(move |o| o.iter_out_edges(edge_type, key)),
                ),
            ),
            EdgeDirection::Inverse => Box::new(
                self.edge_mem_entries_for_type(edge_type)
                    .filter(move |(mk, _)| matches!(mk, MemKey::Edge { dst, .. } if *dst == key)),
            ),
        }
    }

    /// Point-read one physical relationship identity from the committed
    /// memtable plus the staged RYOW overlay.
    fn edge_mem_entry(&self, edge_type: &str, src: NodeId, dst: NodeId) -> Option<&MemEntry> {
        let key = MemKey::Edge {
            edge_type: edge_type.to_string(),
            src,
            dst,
        };
        let committed = self.memtable.get(&key);
        let staged = self.overlay.as_ref().and_then(|o| o.get(&key));
        match (staged, committed) {
            (Some(s), Some(c)) => {
                debug_assert!(
                    s.lsn > c.lsn,
                    "overlay edge LSN {} must exceed committed LSN {}",
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

    /// Build (once per committed memtable generation) the string-property
    /// claimants used to supplement immutable SST sidecars.
    ///
    /// A sidecar lookup must consider rows buffered since the last flush.
    /// Scanning those rows for every key makes a batch lookup quadratic even
    /// though the SST half is indexed. The writer-owned property cache is
    /// invalidated on node commits, so this map is exact for snapshots that
    /// attach it. RYOW overlay snapshots deliberately do not share that cache;
    /// they rebuild from their small staged view to avoid leaking uncommitted
    /// rows across readers.
    fn memtable_property_claimants(
        &self,
        label: &str,
        property: &str,
    ) -> Result<Arc<crate::property_index::MemtableClaimantIndex>> {
        if let (Some(cache), Some(cell)) =
            (&self.property_index_cache, &self.memtable_claimant_cell)
        {
            if let Some(index) = cache.get_memtable_claimants(cell, label, property) {
                return Ok(index);
            }
        }

        let mut index: HashMap<String, imbl::OrdSet<NodeId>> = HashMap::new();
        let mut rows_examined = 0usize;
        for (mk, entry) in self.node_entries() {
            let MemKey::Node { id } = mk else {
                continue;
            };
            rows_examined = rows_examined.saturating_add(1);
            let MemOp::Upsert(payload) = &entry.op else {
                continue;
            };
            let record = NodeWriteRecord::decode(payload)?;
            if !label.is_empty()
                && !record_carries_label(&record, label, &self.manifest.manifest.label_dict)
            {
                continue;
            }
            if let Some(value) = record
                .properties
                .get(property)
                .and_then(crate::cache::encode_equality_property_value)
            {
                index.entry(value).or_default().insert(*id);
            }
        }
        let index: Arc<crate::property_index::MemtableClaimantIndex> =
            Arc::new(index.into_iter().collect());
        if let (Some(cache), Some(cell)) =
            (&self.property_index_cache, &self.memtable_claimant_cell)
        {
            cache.record_memtable_population_rows(rows_examined);
            cache.insert_memtable_claimants(
                cell,
                label.to_string(),
                property.to_string(),
                index.clone(),
            );
        }
        Ok(index)
    }

    /// Probe (and lazily populate) the writer-private committed+staged
    /// postings map attached to a RYOW snapshot.
    ///
    /// `None` means this is an ordinary published/read-only snapshot and the
    /// caller should use committed caches/SST sidecars. `Some(ids)` is
    /// authoritative, including an empty vector. The first lookup for one
    /// `(label, property)` scans the current overlay exactly once; subsequent
    /// snapshots reuse the map, while staged upserts/tombstones maintain it at
    /// the writer chokepoints.
    async fn transactional_property_candidates(
        &self,
        label: &str,
        property: &str,
        value: &Value,
    ) -> Result<Option<Vec<NodeId>>> {
        let Some(index) = self.transactional_property_index else {
            return Ok(None);
        };
        let names = vec![property.to_string()];
        let Some(key) = crate::unique_index::encode_probe_key(&[value]) else {
            return Ok(None);
        };
        if let Some(ids) = index.probe_all(label, &names, &key) {
            return Ok(Some(ids));
        }

        // Populate from exactly the view this snapshot exposes. An empty label
        // denotes the id-primary any-label scope; physical nodes are reconciled
        // once even when they carry multiple labels.
        let views = if label.is_empty() {
            self.scan_all_nodes_with_predicates_and_projection(&[], None)
                .await?
        } else {
            self.scan_label(label).await?
        };
        let entries = views.iter().map(|view| (view.id, &view.properties));
        if self.transactional_property_index_staged {
            index.populate_staged(label, &names, entries);
        } else {
            index.populate(label, &names, entries);
        }
        Ok(Some(
            index
                .probe_all(label, &names, &key)
                .expect("transactional property map was just populated"),
        ))
    }

    /// Return the live node ids matching an indexed scalar equality predicate.
    ///
    /// `Some(ids)` is authoritative (including an empty vector): every
    /// relevant SST carried a compatible equality sidecar, memtable claimants
    /// were included, and each candidate was confirmed against its current
    /// last-write-wins node view. `None` means the property is not indexed, a
    /// legacy/incomplete sidecar set cannot represent this value type, or the
    /// value is not scalar; callers must retain their exact scan fallback.
    ///
    /// This ID-only surface is shared by ordinary `MATCH ... WHERE prop = ...`
    /// and filtered vector search. It deliberately confirms before returning
    /// so a stale posting from a renamed/tombstoned node can never leak into a
    /// downstream pre-filter.
    pub async fn indexed_node_ids_by_property_value(
        &self,
        label: &str,
        property: &str,
        value: &Value,
    ) -> Result<Option<Vec<NodeId>>> {
        Ok(
            match self
                .indexed_node_ids_by_property_value_inner(label, property, value, None, None, None)
                .await?
            {
                IndexedPropertyLookup::Available(ids) => Some(ids),
                IndexedPropertyLookup::Unavailable | IndexedPropertyLookup::Truncated => None,
            },
        )
    }

    /// Limited variant of [`Self::indexed_node_ids_by_property_value`].
    ///
    /// The posting list remains authoritative, but candidate confirmation
    /// stops as soon as `limit` current node versions match. This is the
    /// storage primitive used by a bare Cypher `LIMIT` over an indexed
    /// equality predicate.
    pub async fn indexed_node_ids_by_property_value_limited(
        &self,
        label: &str,
        property: &str,
        value: &Value,
        limit: usize,
    ) -> Result<Option<Vec<NodeId>>> {
        if limit == 0 {
            return Ok(Some(Vec::new()));
        }
        let mut posting_read_limit = limit;
        loop {
            match self
                .indexed_node_ids_by_property_value_inner(
                    label,
                    property,
                    value,
                    Some(limit),
                    None,
                    Some(posting_read_limit),
                )
                .await?
            {
                IndexedPropertyLookup::Available(ids) => return Ok(Some(ids)),
                IndexedPropertyLookup::Unavailable => return Ok(None),
                IndexedPropertyLookup::Truncated => {
                    let widened = posting_read_limit.saturating_mul(4);
                    if widened <= posting_read_limit {
                        return Ok(None);
                    }
                    posting_read_limit = widened;
                    if let Some(cache) = &self.property_index_cache {
                        cache.record_equality_posting_widening();
                    }
                }
            }
        }
    }

    /// Complete indexed equality result only when its posting set is no larger
    /// than `max_candidates`.
    ///
    /// Unlike the limited variant, this never returns a partial eligibility
    /// set: an oversized posting returns `None` before candidate hydration so
    /// ANN callers can choose their high-cardinality fallback safely.
    pub async fn indexed_node_ids_by_property_value_capped(
        &self,
        label: &str,
        property: &str,
        value: &Value,
        max_candidates: usize,
    ) -> Result<Option<Vec<NodeId>>> {
        Ok(
            match self
                .indexed_node_ids_by_property_value_inner(
                    label,
                    property,
                    value,
                    None,
                    Some(max_candidates),
                    None,
                )
                .await?
            {
                IndexedPropertyLookup::Available(ids) => Some(ids),
                IndexedPropertyLookup::Unavailable | IndexedPropertyLookup::Truncated => None,
            },
        )
    }

    async fn indexed_node_ids_by_property_value_inner(
        &self,
        label: &str,
        property: &str,
        value: &Value,
        limit: Option<usize>,
        candidate_cap: Option<usize>,
        posting_read_limit: Option<usize>,
    ) -> Result<IndexedPropertyLookup> {
        namidb_core::profile_scope!("Snapshot::indexed_node_ids_by_property_value");

        let indexed = if label.is_empty() {
            self.manifest.manifest.schema.labels.values().any(|def| {
                def.properties
                    .iter()
                    .find(|p| p.name == property)
                    .is_some_and(|p| p.indexed || p.unique)
            })
        } else {
            self.manifest
                .manifest
                .schema
                .label(label)
                .and_then(|def| def.properties.iter().find(|p| p.name == property))
                .is_some_and(|p| p.indexed || p.unique)
        };
        if !indexed {
            return Ok(IndexedPropertyLookup::Unavailable);
        }
        if limit == Some(0) {
            return Ok(IndexedPropertyLookup::Available(Vec::new()));
        }
        // `prop = NULL` never evaluates true in Cypher.
        if matches!(value, Value::Null) {
            return Ok(IndexedPropertyLookup::Available(Vec::new()));
        }
        let Some(memtable_key) = crate::cache::encode_equality_property_value(value) else {
            return Ok(IndexedPropertyLookup::Unavailable);
        };
        if let Some(cache) = &self.property_index_cache {
            cache.record_equality_lookup();
        }

        let mut cursors: Vec<EqualityNodePostingCursor> = Vec::new();
        let mut advertised_source_entries: Option<usize> = None;
        let mut truncated_source_without_cursor = false;
        if let Some(mut ids) = self
            .transactional_property_candidates(label, property, value)
            .await?
        {
            // The transactional holders use a HashSet for duplicate values;
            // sort once so they can participate in the same streaming merge.
            ids.sort_unstable();
            ids.dedup();
            if !ids.is_empty() {
                cursors.push(EqualityNodePostingCursor::Owned { ids, position: 0 });
            }
        } else {
            let all_node_ssts: Vec<usize> = self.manifest.index.node_descriptors();
            let have_node_ssts = !all_node_ssts.is_empty();
            let sst_idxs: Vec<usize> = all_node_ssts
                .into_iter()
                .filter(|idx| {
                    label.is_empty()
                        || node_sst_can_contain_label(&self.manifest.manifest, *idx, label)
                })
                .collect();
            let sidecars: Option<Vec<_>> = sst_idxs
                .iter()
                .map(|idx| {
                    self.manifest.manifest.ssts[*idx]
                        .equality_property_indices
                        .iter()
                        .find(|desc| desc.property == property && desc.mixed_type_complete)
                        .filter(|desc| equality_sidecar_key(desc.key_encoding, value).is_some())
                        .map(|desc| (*idx, desc))
                })
                .collect();
            let Some(sidecars) = sidecars else {
                crate::route_telemetry::record_property(false);
                return Ok(IndexedPropertyLookup::Unavailable);
            };
            // No SSTs means the committed/staged memtable is the whole store.
            // Otherwise every relevant SST must have contributed one sidecar.
            if have_node_ssts && sidecars.len() != sst_idxs.len() {
                crate::route_telemetry::record_property(false);
                return Ok(IndexedPropertyLookup::Unavailable);
            }

            let memtable = self.memtable_property_claimants(label, property)?;
            advertised_source_entries =
                Some(memtable.get(&memtable_key).map_or(0, |ids| ids.len()));
            if memtable
                .get(&memtable_key)
                .is_some_and(|ids| !ids.is_empty())
            {
                cursors.push(EqualityNodePostingCursor::memtable(memtable, memtable_key));
            }
            for (_idx, sidecar) in sidecars {
                let absolute = format!(
                    "{}/{}",
                    self.paths.namespace_prefix().as_ref(),
                    sidecar.path
                );
                let probe = equality_sidecar_key(sidecar.key_encoding, value)
                    .expect("sidecar compatibility checked above");
                let read_limit = posting_read_limit.or(limit).or(candidate_cap);
                let sidecar_probe = self
                    .probe_equality_property_sidecar_limited(
                        sidecar,
                        &absolute,
                        std::slice::from_ref(&probe),
                        read_limit,
                    )
                    .await;
                let (map, posting_len, truncated) = match sidecar_probe {
                    Ok(result) => result,
                    Err(error) if optional_accelerator_fallback(&error) => {
                        tracing::warn!(
                            path = %sidecar.path,
                            error = %error,
                            "equality accelerator unavailable; falling back to exact scan"
                        );
                        crate::route_telemetry::record_property(false);
                        return Ok(IndexedPropertyLookup::Unavailable);
                    }
                    Err(error) => return Err(error),
                };
                advertised_source_entries = Some(
                    advertised_source_entries
                        .unwrap_or(0)
                        .saturating_add(posting_len),
                );
                if map.get(&probe).is_some_and(|ids| !ids.is_empty()) {
                    cursors.push(EqualityNodePostingCursor::Sidecar {
                        map,
                        key: probe,
                        position: 0,
                        truncated,
                    });
                } else {
                    // A truncated posting with no materialised prefix has an
                    // unknown first NodeId. It can therefore precede any
                    // result confirmed from another source.
                    truncated_source_without_cursor |= truncated;
                }
            }
        }

        // This sum can over-count duplicate ids present in overlapping LSM
        // versions, which is intentional for the capped API: rejecting early
        // chooses the safe fallback and avoids building a corpus-sized union.
        if let Some(cap) = candidate_cap {
            let source_entries = advertised_source_entries.or_else(|| {
                cursors
                    .iter()
                    .try_fold(0_usize, |total, cursor| total.checked_add(cursor.len()))
            });
            if source_entries.is_none_or(|entries| entries > cap) {
                return Ok(IndexedPropertyLookup::Unavailable);
            }
        }

        let target = limit.unwrap_or(usize::MAX);
        let source_entries = advertised_source_entries.unwrap_or_else(|| {
            cursors
                .iter()
                .fold(0_usize, |total, cursor| total.saturating_add(cursor.len()))
        });
        let mut confirmed = Vec::with_capacity(source_entries.min(target));
        const CONFIRM_BATCH: usize = 256;
        while confirmed.len() < target {
            let remaining = target.saturating_sub(confirmed.len());
            let mut batch = Vec::with_capacity(CONFIRM_BATCH.min(remaining));
            while batch.len() < CONFIRM_BATCH.min(remaining) {
                let Some(id) = next_equality_candidate(&mut cursors) else {
                    break;
                };
                if let Some(cache) = &self.property_index_cache {
                    cache.record_equality_candidate_iterated();
                }
                batch.push(id);
            }
            if batch.is_empty() {
                break;
            }
            if let Some(cache) = &self.property_index_cache {
                cache.record_equality_confirmation_candidates(batch.len());
            }
            let views = self.batch_lookup_nodes(label, &batch).await?;
            for (id, view) in batch.into_iter().zip(views) {
                if view
                    .as_ref()
                    .is_some_and(|view| view.properties.get(property) == Some(value))
                {
                    confirmed.push(id);
                    if confirmed.len() == target {
                        break;
                    }
                }
            }
        }
        let truncated_suffix_may_change_result = if confirmed.len() < target {
            truncated_source_without_cursor
                || cursors
                    .iter()
                    .any(EqualityNodePostingCursor::has_truncated_suffix)
        } else {
            let cutoff = *confirmed
                .last()
                .expect("a full equality prefix has a confirmed cutoff");
            truncated_source_without_cursor
                || cursors
                    .iter()
                    .any(|cursor| cursor.unread_may_precede(cutoff))
        };
        if truncated_suffix_may_change_result {
            // A truncated source is not safe merely because another source
            // filled LIMIT. Its unread NodeIds may sort before that confirmed
            // cutoff (for example, an overlapping SST prefix can contain only
            // stale versions while a memtable contributes large live ids).
            // Geometrically widen until the unseen suffix is provably after
            // the cutoff or every posting is exhausted.
            return Ok(IndexedPropertyLookup::Truncated);
        }
        crate::route_telemetry::record_property(true);
        Ok(IndexedPropertyLookup::Available(confirmed))
    }

    /// Read the first `limit` live nodes in ascending String-property order
    /// from a complete equality index.
    ///
    /// The iterator k-way merges immutable per-SST BTree sidecars plus the
    /// current memtable claimants and confirms each candidate against the
    /// snapshot. It therefore positions by value and hydrates only the prefix
    /// needed for `SKIP + LIMIT`, instead of decoding and sorting the label.
    ///
    /// Returns `None` when the property is not a declared String index, any
    /// relevant SST lacks a compatible sidecar, or the non-null indexed prefix
    /// is shorter than `limit` (the ordinary scan must then account for NULL /
    /// missing values at the end of ascending order).
    pub async fn ordered_node_ids_by_string_property(
        &self,
        label: &str,
        property: &str,
        limit: usize,
    ) -> Result<Option<Vec<NodeId>>> {
        namidb_core::profile_scope!("Snapshot::ordered_node_ids_by_string_property");
        if let Some(cache) = &self.property_index_cache {
            cache.record_ordered_prefix_call();
        }
        if limit == 0 {
            return Ok(Some(Vec::new()));
        }
        let indexed_string = self
            .manifest
            .manifest
            .schema
            .label(label)
            .and_then(|def| def.properties.iter().find(|p| p.name == property))
            .is_some_and(|p| {
                (p.indexed || p.unique)
                    && matches!(
                        p.data_type,
                        namidb_core::DataType::Utf8 | namidb_core::DataType::LargeUtf8
                    )
            });
        if !indexed_string {
            return Ok(None);
        }

        let node_ssts: Vec<usize> = self
            .manifest
            .index
            .node_descriptors()
            .into_iter()
            .filter(|idx| node_sst_can_contain_label(&self.manifest.manifest, *idx, label))
            .collect();
        let sidecars: Option<Vec<_>> = node_ssts
            .iter()
            .map(|idx| string_property_sidecar(&self.manifest.manifest.ssts[*idx], label, property))
            .collect();
        let Some(sidecars) = sidecars else {
            return Ok(None);
        };
        let mut legacy_unique_sidecars = Vec::new();
        let mut equality_sidecars = Vec::new();
        for sidecar in sidecars {
            match sidecar {
                StringPropertySidecar::Unique(descriptor) => {
                    let absolute = format!(
                        "{}/{}",
                        self.paths.namespace_prefix().as_ref(),
                        descriptor.path
                    );
                    match self.fetch_unique_property_sidecar(&absolute).await {
                        Ok(sidecar) => legacy_unique_sidecars.push(sidecar),
                        Err(error) if optional_accelerator_fallback(&error) => {
                            tracing::warn!(
                                path = %descriptor.path,
                                error = %error,
                                "legacy ordered-property accelerator unavailable; falling back to exact scan"
                            );
                            return Ok(None);
                        }
                        Err(error) => return Err(error),
                    }
                }
                StringPropertySidecar::Equality(descriptor) => {
                    equality_sidecars.push(descriptor);
                }
            }
        }

        // The committed/staged memtable claimant cache uses ScalarV1 keys but
        // is a HashMap (point-lookup optimised). Sort only that bounded delta
        // into a BTreeMap so it can participate in each widening attempt.
        let memtable = self.memtable_property_claimants(label, property)?;
        let memtable_sorted = if memtable.is_empty() {
            None
        } else {
            let sorted: EqualityPropertySidecar = memtable
                .iter()
                .map(|(key, ids)| (key.clone(), ids.iter().map(|id| *id.as_bytes()).collect()))
                .collect();
            Some(Arc::new(sorted))
        };

        // Equality sidecars are global to the id-primary node SST. The first K
        // values can therefore all belong to another label (e.g. 10k Article
        // keys before a sparse Materia label). Geometrically widen the
        // range-readable prefix until K requested-label rows are confirmed or
        // every sidecar is exhausted. Restarting is deliberate: the final pass
        // dominates the geometric series and preserves a simple exact merge.
        let mut prefix_target = limit;
        loop {
            let mut sources: Vec<OrderedStringPostingCursor> =
                Vec::with_capacity(legacy_unique_sidecars.len() + equality_sidecars.len() + 1);
            let mut truncated_source_without_cursor = false;
            for sidecar in &legacy_unique_sidecars {
                if let Some(cursor) = OrderedStringPostingCursor::new_unique(sidecar.clone()) {
                    sources.push(cursor);
                }
            }
            for sidecar in &equality_sidecars {
                let absolute = format!(
                    "{}/{}",
                    self.paths.namespace_prefix().as_ref(),
                    sidecar.path
                );
                let prefix = self
                    .fetch_equality_property_sidecar_prefix(sidecar, &absolute, prefix_target)
                    .await;
                let (map, truncated) = match prefix {
                    Ok(prefix) => prefix,
                    Err(error) if optional_accelerator_fallback(&error) => {
                        tracing::warn!(
                            path = %sidecar.path,
                            error = %error,
                            "ordered-property accelerator unavailable; falling back to exact scan"
                        );
                        return Ok(None);
                    }
                    Err(error) => return Err(error),
                };
                if let Some(cursor) =
                    OrderedStringPostingCursor::new_equality(map, sidecar.key_encoding, truncated)
                {
                    sources.push(cursor);
                } else {
                    // A malformed/legacy map may contain only empty posting
                    // lists. If the range read stopped there, its first real
                    // tuple is unknown and can precede every visible source.
                    truncated_source_without_cursor |= truncated;
                }
            }
            if let Some(sorted) = &memtable_sorted {
                if let Some(cursor) = OrderedStringPostingCursor::new_equality(
                    sorted.clone(),
                    crate::manifest::EqualityKeyEncoding::ScalarV1,
                    false,
                ) {
                    sources.push(cursor);
                }
            }

            let mut out = Vec::with_capacity(ordered_prefix_initial_capacity(limit));
            let mut seen: std::collections::HashSet<(String, NodeId)> =
                std::collections::HashSet::new();
            let mut confirmed_cutoff: Option<(String, NodeId)> = None;
            while out.len() < limit {
                crate::cancel::check()?;
                // Pull a small ordered candidate window, then confirm all ids in
                // one row-group-aware storage pass. This keeps `SKIP 1000` at a
                // handful of batched SST reads instead of ~1,000 sequential point
                // lookups while still allowing stale postings to be discarded and
                // replenished from the next window.
                const CONFIRM_BATCH: usize = 256;
                let batch_target = CONFIRM_BATCH.min(limit - out.len());
                let mut pending: Vec<(String, NodeId)> = Vec::with_capacity(batch_target);
                while pending.len() < batch_target {
                    let next = sources
                        .iter()
                        .enumerate()
                        .filter_map(|(source, cursor)| {
                            cursor
                                .current()
                                .map(|(value, id)| (source, value.to_string(), id))
                        })
                        .min_by(|(_, av, aid), (_, bv, bid)| av.cmp(bv).then(aid.cmp(bid)));
                    let Some((source, value, id)) = next else {
                        break;
                    };
                    sources[source].advance();
                    if seen.insert((value.clone(), id)) {
                        if let Some(cache) = &self.property_index_cache {
                            cache.record_ordered_prefix_candidate_iterated();
                        }
                        pending.push((value, id));
                    }
                }
                if pending.is_empty() {
                    break;
                }
                if let Some(cache) = &self.property_index_cache {
                    cache.record_ordered_prefix_confirmation_candidates(pending.len());
                }
                let ids: Vec<NodeId> = pending.iter().map(|(_, id)| *id).collect();
                let views = self.batch_lookup_nodes(label, &ids).await?;
                for ((value, id), view) in pending.into_iter().zip(views) {
                    if view.as_ref().is_some_and(|view| {
                        matches!(view.properties.get(property), Some(Value::Str(current)) if current == &value)
                    }) {
                        confirmed_cutoff = Some((value, id));
                        out.push(id);
                        if out.len() == limit {
                            break;
                        }
                    }
                }
            }

            let any_truncated =
                truncated_source_without_cursor || sources.iter().any(|source| source.truncated);
            if out.len() == limit {
                let cutoff = confirmed_cutoff
                    .as_ref()
                    .expect("a full ordered prefix has a confirmed cutoff");
                // A truncated source needs another range only when its first
                // unread tuple could sort before the Kth confirmed result.
                // This is exact per source: it avoids both the old false
                // return (one SST hid a smaller value) and needless widening
                // merely because an unrelated-label candidate was discarded.
                let must_widen = truncated_source_without_cursor
                    || sources
                        .iter()
                        .any(|source| source.unread_may_precede(cutoff));
                if !must_widen {
                    return Ok(Some(out));
                }
            }
            if any_truncated {
                let widened = prefix_target.saturating_mul(4);
                if widened > prefix_target {
                    prefix_target = widened;
                    if let Some(cache) = &self.property_index_cache {
                        cache.record_ordered_prefix_widening();
                    }
                    continue;
                }
            }

            // Every indexed non-null value was exhausted (or usize saturated)
            // before K live rows were proven. The ordinary scan must account
            // for NULL/missing values and remains the exact fallback.
            return Ok(None);
        }
    }

    /// Posting-sidecar coverage for `(label, property)` over the node SSTs
    /// that can contain the label: `(covered, in_scope)`. The lookup routes
    /// demote to a label scan unless every in-scope SST is covered, so
    /// `covered < in_scope` means "the index exists in the schema but this
    /// snapshot pays scan prices" — the physical-route fact EXPLAIN
    /// annotates (item 39). `(0, 0)` = memtable-only (no SSTs in scope;
    /// the claimant map serves natively).
    pub fn property_index_coverage(&self, label: &str, property: &str) -> (usize, usize) {
        let in_scope: Vec<usize> = self
            .manifest
            .index
            .node_descriptors()
            .into_iter()
            .filter(|i| node_sst_can_contain_label(&self.manifest.manifest, *i, label))
            .collect();
        let covered = in_scope
            .iter()
            .filter(|i| {
                string_property_sidecar(&self.manifest.manifest.ssts[**i], label, property)
                    .is_some()
            })
            .count();
        (covered, in_scope.len())
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
        if let Some(cache) = &self.property_index_cache {
            cache.record_unique_lookup();
        }
        // Writer/RYOW path: the private transactional postings map sees both
        // committed and staged rows and is incrementally maintained across
        // overlay snapshots. Confirm against this snapshot so relabels,
        // tombstones, and value changes retain last-write-wins semantics.
        if let Some(ids) = self
            .transactional_property_candidates(label, property, &Value::Str(value.to_string()))
            .await?
        {
            let mut confirmed: Option<NodeView> = None;
            for id in ids {
                let Some(view) = self.lookup_node(label, id).await? else {
                    continue;
                };
                if !matches!(view.properties.get(property), Some(Value::Str(s)) if s == value) {
                    continue;
                }
                let replace = confirmed.as_ref().is_none_or(|current| {
                    view.lsn > current.lsn || (view.lsn == current.lsn && view.id < current.id)
                });
                if replace {
                    confirmed = Some(view);
                }
            }
            return Ok(confirmed);
        }
        // 1. Try the cross-snapshot in-memory index — `O(1)` warm path.
        if let Some(cache) = &self.property_index_cache {
            let generation = self
                .property_index_generation
                .unwrap_or_else(|| cache.generation());
            if let Some(idx) = cache.get_at(label, property, generation) {
                crate::route_telemetry::record_property(true);
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
        // Rolling upgrades can legitimately mix legacy label-scoped SSTs,
        // which carry a single-value unique sidecar, with current id-primary
        // SSTs, which carry a global equality posting sidecar for the same
        // schema-unique property. Coverage is therefore a per-SST property:
        // requiring one homogeneous sidecar class across the whole generation
        // permanently demoted such mixed-scope stores to `scan_label`.
        let sidecars: Option<Vec<_>> = sst_idxs
            .iter()
            .map(|idx| string_property_sidecar(&self.manifest.manifest.ssts[*idx], label, property))
            .collect();
        if let (true, Some(sidecars)) = (have_node_ssts, sidecars) {
            namidb_core::profile_scope!("Snapshot::lookup_node_by_property.sidecar");
            // Gather every id that has ever claimed `value` in an in-scope
            // sidecar, plus current memtable claimants, then confirm each
            // candidate against the snapshot's last-write-wins node view.
            //
            // A unique sidecar stores `value → NodeId`, not the posting row's
            // LSN. Using the enclosing SST's `max_lsn` as that row LSN is
            // incorrect: an unrelated high-LSN row can make a stale claimant
            // appear newer than the value's real reassignment in another SST.
            // If that stale id was subsequently renamed, re-verifying only the
            // false "winner" returns None and hides the real live owner.
            //
            // Candidate cardinality is bounded by the number of in-scope SSTs
            // plus matching memtable rows (normally one under the uniqueness
            // invariant), and BTreeSet gives deterministic probe order.
            let mut candidates: BTreeSet<namidb_core::id::NodeId> = BTreeSet::new();
            let memtable_claimants = self.memtable_property_claimants(label, property)?;
            let memtable_key =
                crate::cache::encode_equality_property_value(&Value::Str(value.to_string()))
                    .expect("String values have an equality key");
            if let Some(ids) = memtable_claimants.get(&memtable_key) {
                for id in ids {
                    candidates.insert(*id);
                }
            }

            // SST sidecar pass: bincode-decode each `(value → NodeId)` map and
            // retain every claimant. Confirmation below resolves tombstones,
            // renames, relabels, and cross-SST last-write-wins by id.
            let mut sidecars_available = true;
            for sidecar in sidecars {
                match sidecar {
                    StringPropertySidecar::Unique(sidecar_desc) => {
                        let absolute = format!(
                            "{}/{}",
                            self.paths.namespace_prefix().as_ref(),
                            sidecar_desc.path
                        );
                        let map = match self
                            .probe_unique_property_sidecar(
                                sidecar_desc,
                                &absolute,
                                &[value.to_string()],
                            )
                            .await
                        {
                            Ok(map) => map,
                            Err(error) if optional_accelerator_fallback(&error) => {
                                tracing::warn!(
                                    path = %sidecar_desc.path,
                                    error = %error,
                                    "unique property accelerator unavailable; falling back to exact label scan"
                                );
                                sidecars_available = false;
                                break;
                            }
                            Err(error) => return Err(error),
                        };
                        if let Some(id_bytes) = map.get(value) {
                            candidates.insert(NodeId::from_uuid(Uuid::from_bytes(*id_bytes)));
                        }
                    }
                    StringPropertySidecar::Equality(sidecar_desc) => {
                        let absolute = format!(
                            "{}/{}",
                            self.paths.namespace_prefix().as_ref(),
                            sidecar_desc.path
                        );
                        let probe = equality_sidecar_key(
                            sidecar_desc.key_encoding,
                            &Value::Str(value.to_string()),
                        )
                        .expect("String sidecar compatibility checked during coverage");
                        let map = match self
                            .probe_equality_property_sidecar(
                                sidecar_desc,
                                &absolute,
                                std::slice::from_ref(&probe),
                            )
                            .await
                        {
                            Ok(map) => map,
                            Err(error) if optional_accelerator_fallback(&error) => {
                                tracing::warn!(
                                    path = %sidecar_desc.path,
                                    error = %error,
                                    "equality property accelerator unavailable; falling back to exact label scan"
                                );
                                sidecars_available = false;
                                break;
                            }
                            Err(error) => return Err(error),
                        };
                        if let Some(ids) = map.get(&probe) {
                            candidates.extend(
                                ids.iter()
                                    .map(|id| NodeId::from_uuid(Uuid::from_bytes(*id))),
                            );
                        }
                    }
                }
            }

            if sidecars_available {
                crate::route_telemetry::record_property(true);
                // Under a valid unique constraint there is at most one
                // confirmed live owner. If legacy/corrupt data contains more,
                // select deterministically by newest row LSN, then lowest
                // NodeId, rather than depending on manifest/SST iteration
                // order.
                let mut confirmed: Option<NodeView> = None;
                for id in candidates {
                    let Some(view) = self.lookup_node(label, id).await? else {
                        continue;
                    };
                    if !matches!(view.properties.get(property),
                        Some(namidb_core::Value::Str(s)) if s == value)
                    {
                        continue;
                    }
                    let replace = confirmed.as_ref().is_none_or(|current| {
                        view.lsn > current.lsn || (view.lsn == current.lsn && view.id < current.id)
                    });
                    if replace {
                        confirmed = Some(view);
                    }
                }
                return Ok(confirmed);
            }
        }

        // 3. Legacy cold path: full label scan to build the index, then look up.
        //
        // Reached when at least one SST in the scope was written by a
        // pre-sidecar build (or when the property wasn't declared
        // `unique` at flush time). The in-memory cache caches the
        // result so subsequent calls bypass the scan.
        crate::route_telemetry::record_property(!have_node_ssts);
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
            let generation = self
                .property_index_generation
                .unwrap_or_else(|| cache.generation());
            cache.insert_at(
                label.to_string(),
                property.to_string(),
                std::sync::Arc::new(idx),
                generation,
            );
        }
        Ok(found)
    }

    /// Batched unique String-property lookup.
    ///
    /// Returns one entry per input value, preserving order, duplicates and
    /// misses. For a complete sidecar generation it gathers every claimant
    /// from the committed memtable and all in-scope sidecars once, then
    /// confirms the deduplicated NodeIds with exactly one
    /// [`Self::batch_lookup_nodes`] call. Confirmation is mandatory: an older
    /// sidecar can still name a node whose current value was changed or
    /// tombstoned by a newer memtable/SST record.
    ///
    /// The caller is responsible for the uniqueness invariant, matching
    /// [`Self::lookup_node_by_property`]. Legacy/incomplete sidecar stores fall
    /// back to one label scan, never one scan per requested value.
    #[instrument(skip(self, values), fields(
        label = label,
        property = property,
        values_len = values.len()
    ))]
    pub async fn batch_lookup_nodes_by_property(
        &self,
        label: &str,
        property: &str,
        values: &[String],
    ) -> Result<Vec<Option<NodeView>>> {
        namidb_core::profile_scope!("Snapshot::batch_lookup_nodes_by_property");
        if values.is_empty() {
            return Ok(Vec::new());
        }
        if let Some(cache) = &self.property_index_cache {
            // One storage operation, regardless of batch cardinality.
            cache.record_unique_lookup();
        }

        // `batch_lookup_nodes` is label-scoped. Preserve the existing
        // label-agnostic sentinel exactly through one all-node scan.
        if label.is_empty() {
            let views = self
                .scan_all_nodes_with_predicates_and_projection(&[], None)
                .await?;
            return self.batch_unique_from_scan(label, property, values, views);
        }

        let mut candidates: BTreeMap<String, BTreeSet<NodeId>> = values
            .iter()
            .cloned()
            .map(|value| (value, BTreeSet::new()))
            .collect();

        // RYOW snapshots use the writer-private postings map. Populating the
        // first distinct value scans the overlay once; every later value is an
        // O(1) probe. Candidate confirmation is still batched below.
        if self.transactional_property_index.is_some() {
            for (value, ids_out) in &mut candidates {
                if let Some(ids) = self
                    .transactional_property_candidates(label, property, &Value::Str(value.clone()))
                    .await?
                {
                    ids_out.extend(ids);
                }
            }
            return self
                .batch_confirm_unique_candidates(label, property, values, candidates)
                .await;
        }

        // A fully materialised legacy index is already a complete claimant
        // source for this logical generation. Batch-confirm its hits so node
        // materialisation is still row-group vectorised.
        if let Some(cache) = &self.property_index_cache {
            let generation = self
                .property_index_generation
                .unwrap_or_else(|| cache.generation());
            if let Some(index) = cache.get_at(label, property, generation) {
                for (value, ids_out) in &mut candidates {
                    if let Some(id) = index.get(value) {
                        ids_out.insert(*id);
                    }
                }
                return self
                    .batch_confirm_unique_candidates(label, property, values, candidates)
                    .await;
            }
        }

        let node_sst_idxs: Vec<usize> = self.manifest.index.node_descriptors();
        let have_node_ssts = !node_sst_idxs.is_empty();
        let sst_idxs: Vec<usize> = node_sst_idxs
            .into_iter()
            .filter(|idx| node_sst_can_contain_label(&self.manifest.manifest, *idx, label))
            .collect();
        let sidecars: Option<Vec<_>> = sst_idxs
            .iter()
            .map(|idx| string_property_sidecar(&self.manifest.manifest.ssts[*idx], label, property))
            .collect();

        // A memtable-only namespace has a complete claimant map. With SSTs,
        // every relevant SST must carry either supported sidecar. The choice is
        // intentionally per-SST so a rolling-upgrade generation can combine a
        // legacy unique map with a current global equality posting map.
        if have_node_ssts && sidecars.is_none() {
            crate::route_telemetry::record_property(false);
            let views = self.scan_label(label).await?;
            return self.batch_unique_from_scan(label, property, values, views);
        }
        crate::route_telemetry::record_property(true);

        let memtable_claimants = self.memtable_property_claimants(label, property)?;
        for (value, ids_out) in &mut candidates {
            let memtable_key =
                crate::cache::encode_equality_property_value(&Value::Str(value.clone()))
                    .expect("String values have an equality key");
            if let Some(ids) = memtable_claimants.get(&memtable_key) {
                ids_out.extend(ids.iter().copied());
            }
        }

        for sidecar in sidecars.unwrap_or_default() {
            match sidecar {
                StringPropertySidecar::Unique(sidecar) => {
                    let absolute = format!(
                        "{}/{}",
                        self.paths.namespace_prefix().as_ref(),
                        sidecar.path
                    );
                    let index = match self
                        .probe_unique_property_sidecar(sidecar, &absolute, values)
                        .await
                    {
                        Ok(index) => index,
                        Err(error) if optional_accelerator_fallback(&error) => {
                            tracing::warn!(
                                path = %sidecar.path,
                                error = %error,
                                "unique property accelerator unavailable; falling back to one exact batch label scan"
                            );
                            crate::route_telemetry::record_property(false);
                            let views = self.scan_label(label).await?;
                            return self.batch_unique_from_scan(label, property, values, views);
                        }
                        Err(error) => return Err(error),
                    };
                    for (value, ids_out) in &mut candidates {
                        if let Some(id) = index.get(value) {
                            ids_out.insert(NodeId::from_uuid(Uuid::from_bytes(*id)));
                        }
                    }
                }
                StringPropertySidecar::Equality(sidecar) => {
                    let absolute = format!(
                        "{}/{}",
                        self.paths.namespace_prefix().as_ref(),
                        sidecar.path
                    );
                    let probes: Vec<String> = values
                        .iter()
                        .map(|value| {
                            equality_sidecar_key(sidecar.key_encoding, &Value::Str(value.clone()))
                                .expect("String sidecar compatibility checked during coverage")
                        })
                        .collect();
                    let index = match self
                        .probe_equality_property_sidecar(sidecar, &absolute, &probes)
                        .await
                    {
                        Ok(index) => index,
                        Err(error) if optional_accelerator_fallback(&error) => {
                            tracing::warn!(
                                path = %sidecar.path,
                                error = %error,
                                "equality property accelerator unavailable; falling back to one exact batch label scan"
                            );
                            crate::route_telemetry::record_property(false);
                            let views = self.scan_label(label).await?;
                            return self.batch_unique_from_scan(label, property, values, views);
                        }
                        Err(error) => return Err(error),
                    };
                    for (value, ids_out) in &mut candidates {
                        let probe =
                            equality_sidecar_key(sidecar.key_encoding, &Value::Str(value.clone()))
                                .expect("String sidecar compatibility checked during coverage");
                        if let Some(ids) = index.get(&probe) {
                            ids_out.extend(
                                ids.iter()
                                    .map(|id| NodeId::from_uuid(Uuid::from_bytes(*id))),
                            );
                        }
                    }
                }
            }
        }

        self.batch_confirm_unique_candidates(label, property, values, candidates)
            .await
    }

    async fn batch_confirm_unique_candidates(
        &self,
        label: &str,
        property: &str,
        values: &[String],
        candidates: BTreeMap<String, BTreeSet<NodeId>>,
    ) -> Result<Vec<Option<NodeView>>> {
        let ids: Vec<NodeId> = candidates
            .values()
            .flat_map(|ids| ids.iter().copied())
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect();
        let resolved = self.batch_lookup_nodes(label, &ids).await?;
        let by_id: HashMap<NodeId, NodeView> = ids
            .into_iter()
            .zip(resolved)
            .filter_map(|(id, view)| view.map(|view| (id, view)))
            .collect();

        let mut confirmed: HashMap<String, Option<NodeView>> =
            HashMap::with_capacity(candidates.len());
        for (value, ids) in candidates {
            let mut winner: Option<NodeView> = None;
            for id in ids {
                let Some(view) = by_id.get(&id) else {
                    continue;
                };
                if !matches!(view.properties.get(property), Some(Value::Str(current)) if current == &value)
                {
                    continue;
                }
                let replace = winner.as_ref().is_none_or(|old| {
                    view.lsn > old.lsn || (view.lsn == old.lsn && view.id < old.id)
                });
                if replace {
                    winner = Some(view.clone());
                }
            }
            confirmed.insert(value, winner);
        }
        Ok(values
            .iter()
            .map(|value| confirmed.get(value).cloned().flatten())
            .collect())
    }

    fn batch_unique_from_scan(
        &self,
        label: &str,
        property: &str,
        values: &[String],
        views: Vec<NodeView>,
    ) -> Result<Vec<Option<NodeView>>> {
        let mut by_value: HashMap<String, NodeView> = HashMap::new();
        for view in views {
            let Some(Value::Str(value)) = view.properties.get(property) else {
                continue;
            };
            let replace = by_value
                .get(value)
                .is_none_or(|old| view.lsn > old.lsn || (view.lsn == old.lsn && view.id < old.id));
            if replace {
                by_value.insert(value.clone(), view);
            }
        }

        if let Some(cache) = &self.property_index_cache {
            let generation = self
                .property_index_generation
                .unwrap_or_else(|| cache.generation());
            let index = by_value
                .iter()
                .map(|(value, view)| (value.clone(), view.id))
                .collect();
            cache.insert_at(
                label.to_string(),
                property.to_string(),
                Arc::new(index),
                generation,
            );
        }

        Ok(values
            .iter()
            .map(|value| by_value.get(value).cloned())
            .collect())
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
        if let Some(cache) = &self.property_index_cache {
            cache.record_equality_lookup();
        }
        if let Some(ids) = self
            .transactional_property_candidates(label, property, &Value::Str(value.to_string()))
            .await?
        {
            let mut out = Vec::with_capacity(ids.len());
            for id in ids {
                if let Some(view) = self.lookup_node(label, id).await? {
                    if matches!(view.properties.get(property), Some(Value::Str(s)) if s == value) {
                        out.push(view);
                    }
                }
            }
            out.sort_by_key(|view| view.id);
            return Ok(out);
        }

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
                    .any(|d| d.property == property && d.mixed_type_complete)
            });

        // Cold path: a pre-sidecar SST is in scope (or the property was not
        // `indexed` at flush time). A memtable-only store is NOT cold: its
        // claimant map is the complete index and is cached per generation.
        if have_node_ssts && !all_have_sidecar {
            crate::route_telemetry::record_property(false);
            let all_nodes = self.scan_label(label).await?;
            return Ok(all_nodes
                .into_iter()
                .filter(|v| {
                    matches!(v.properties.get(property),
                        Some(namidb_core::Value::Str(s)) if s == value)
                })
                .collect());
        }
        crate::route_telemetry::record_property(true);

        namidb_core::profile_scope!("Snapshot::lookup_nodes_by_property.sidecar");
        // Gather candidate ids: memtable upserts carrying `value`, plus the
        // union of every SST posting list under `value`.
        let mut candidates: std::collections::BTreeSet<namidb_core::id::NodeId> =
            std::collections::BTreeSet::new();
        let memtable_claimants = self.memtable_property_claimants(label, property)?;
        let memtable_key =
            crate::cache::encode_equality_property_value(&Value::Str(value.to_string()))
                .expect("String values have an equality key");
        if let Some(ids) = memtable_claimants.get(&memtable_key) {
            for id in ids {
                candidates.insert(*id);
            }
        }
        for idx in &sst_idxs {
            let desc = &self.manifest.manifest.ssts[*idx];
            let sidecar_desc = desc
                .equality_property_indices
                .iter()
                .find(|d| d.property == property && d.mixed_type_complete)
                .expect("all_have_sidecar guard");
            let absolute = format!(
                "{}/{}",
                self.paths.namespace_prefix().as_ref(),
                sidecar_desc.path
            );
            let probe =
                equality_sidecar_key(sidecar_desc.key_encoding, &Value::Str(value.to_string()))
                    .expect("String values are supported by every equality encoding");
            let map = match self
                .probe_equality_property_sidecar(
                    sidecar_desc,
                    &absolute,
                    std::slice::from_ref(&probe),
                )
                .await
            {
                Ok(map) => map,
                Err(error) if optional_accelerator_fallback(&error) => {
                    tracing::warn!(
                        path = %sidecar_desc.path,
                        error = %error,
                        "equality property accelerator unavailable; falling back to exact label scan"
                    );
                    let all_nodes = self.scan_label(label).await?;
                    return Ok(all_nodes
                        .into_iter()
                        .filter(|view| {
                            matches!(view.properties.get(property),
                                Some(namidb_core::Value::Str(current)) if current == value)
                        })
                        .collect());
                }
                Err(error) => return Err(error),
            };
            if let Some(ids) = map.get(&probe) {
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

    /// Resolve a label-agnostic equality predicate, e.g.
    /// `MATCH (n {key: $key})`, without guessing a label scope.
    ///
    /// Fresh id-primary SSTs carry a global equality sidecar for every
    /// schema-declared indexed *or unique* property. We union those postings
    /// with the committed memtable claimant map and confirm candidates by
    /// physical node id, so multi-label nodes appear once and per-label
    /// uniqueness does not incorrectly collapse equal keys from two labels.
    /// Older/incomplete SST sets pay one all-node scan and retain a complete
    /// cross-snapshot fallback index.
    pub async fn lookup_nodes_by_property_any_label(
        &self,
        property: &str,
        value: &str,
    ) -> Result<Vec<NodeView>> {
        namidb_core::profile_scope!("Snapshot::lookup_nodes_by_property_any_label");
        let mut results = self
            .batch_lookup_nodes_by_property_any_label(property, &[value.to_string()])
            .await?;
        Ok(results.pop().unwrap_or_default())
    }

    /// Batched label-agnostic equality lookup.
    ///
    /// Returns one posting result per requested value, preserving input order,
    /// duplicate values, misses, and per-value ascending `NodeId` order. Physical
    /// work is deduplicated: the committed memtable claimant map is consulted
    /// once, each equality sidecar receives every distinct probe in one call,
    /// and all distinct candidate ids are confirmed through one
    /// [`Self::batch_lookup_nodes`] hydration.
    ///
    /// Confirmation remains mandatory because an older posting can name a node
    /// whose current value was renamed or tombstoned. Transactional snapshots
    /// source candidates from the writer-private RYOW index. A legacy/incomplete
    /// sidecar generation falls back to exactly one all-node scan for the whole
    /// batch and seeds the existing complete global cache.
    pub async fn batch_lookup_nodes_by_property_any_label(
        &self,
        property: &str,
        values: &[String],
    ) -> Result<Vec<Vec<NodeView>>> {
        namidb_core::profile_scope!("Snapshot::batch_lookup_nodes_by_property_any_label");
        if values.is_empty() {
            return Ok(Vec::new());
        }
        if let Some(cache) = &self.property_index_cache {
            cache.record_equality_lookup();
            let generation = self
                .property_index_generation
                .unwrap_or_else(|| cache.generation());
            if let Some(index) = cache.get_global_at(property, generation) {
                let candidates = values
                    .iter()
                    .cloned()
                    .collect::<BTreeSet<_>>()
                    .into_iter()
                    .map(|value| {
                        let ids = index
                            .get(&value)
                            .into_iter()
                            .flatten()
                            .copied()
                            .collect::<BTreeSet<_>>();
                        (value, ids)
                    })
                    .collect();
                return self
                    .batch_confirm_global_property_candidates(property, values, candidates)
                    .await;
            }
        }

        let distinct_values: BTreeSet<String> = values.iter().cloned().collect();
        if self.transactional_property_index.is_some() {
            let mut candidates: BTreeMap<String, BTreeSet<NodeId>> = distinct_values
                .iter()
                .cloned()
                .map(|value| (value, BTreeSet::new()))
                .collect();
            for (value, ids_out) in &mut candidates {
                if let Some(ids) = self
                    .transactional_property_candidates("", property, &Value::Str(value.clone()))
                    .await?
                {
                    ids_out.extend(ids);
                }
            }
            return self
                .batch_confirm_global_property_candidates(property, values, candidates)
                .await;
        }

        let sst_idxs: Vec<usize> = self.manifest.index.node_descriptors();
        let have_node_ssts = !sst_idxs.is_empty();
        let all_have_sidecar = have_node_ssts
            && sst_idxs.iter().all(|idx| {
                self.manifest.manifest.ssts[*idx]
                    .equality_property_indices
                    .iter()
                    .any(|d| d.property == property && d.mixed_type_complete)
            });

        // New stores use the sidecars. A memtable-only store has the same
        // candidate shape without an SST half, so it also avoids a full scan.
        if all_have_sidecar || !have_node_ssts {
            let mut candidates: BTreeMap<String, BTreeSet<NodeId>> = distinct_values
                .iter()
                .cloned()
                .map(|value| (value, BTreeSet::new()))
                .collect();
            let memtable_claimants = self.memtable_property_claimants("", property)?;
            for (value, ids_out) in &mut candidates {
                let memtable_key =
                    crate::cache::encode_equality_property_value(&Value::Str(value.clone()))
                        .expect("String values have an equality key");
                if let Some(ids) = memtable_claimants.get(&memtable_key) {
                    ids_out.extend(ids.iter());
                }
            }

            let mut sidecars_available = true;
            if all_have_sidecar {
                for idx in &sst_idxs {
                    let desc = &self.manifest.manifest.ssts[*idx];
                    let sidecar_desc = desc
                        .equality_property_indices
                        .iter()
                        .find(|d| d.property == property && d.mixed_type_complete)
                        .expect("all_have_sidecar guard");
                    let absolute = format!(
                        "{}/{}",
                        self.paths.namespace_prefix().as_ref(),
                        sidecar_desc.path
                    );
                    let encoded: Vec<(String, String)> = distinct_values
                        .iter()
                        .map(|value| {
                            let probe = equality_sidecar_key(
                                sidecar_desc.key_encoding,
                                &Value::Str(value.clone()),
                            )
                            .expect("String values are supported by every equality encoding");
                            (value.clone(), probe)
                        })
                        .collect();
                    let probes: Vec<String> = encoded
                        .iter()
                        .map(|(_, probe)| probe.clone())
                        .collect::<BTreeSet<_>>()
                        .into_iter()
                        .collect();
                    let map = match self
                        .probe_equality_property_sidecar(sidecar_desc, &absolute, &probes)
                        .await
                    {
                        Ok(map) => map,
                        Err(error) if optional_accelerator_fallback(&error) => {
                            tracing::warn!(
                                path = %sidecar_desc.path,
                                error = %error,
                                "global equality accelerator unavailable; falling back to one batch scan"
                            );
                            sidecars_available = false;
                            break;
                        }
                        Err(error) => return Err(error),
                    };
                    for (value, probe) in encoded {
                        if let Some(ids) = map.get(&probe) {
                            candidates
                                .get_mut(&value)
                                .expect("distinct value initialized")
                                .extend(
                                    ids.iter()
                                        .map(|bytes| NodeId::from_uuid(Uuid::from_bytes(*bytes))),
                                );
                        }
                    }
                }
            }

            if sidecars_available {
                return self
                    .batch_confirm_global_property_candidates(property, values, candidates)
                    .await;
            }
        }

        // Legacy/incomplete sidecars: one label-agnostic reconciliation for the
        // whole batch, then keep every posting so later calls are O(1).
        let all_nodes = self
            .scan_all_nodes_with_predicates_and_projection(&[], None)
            .await?;
        let mut index: HashMap<String, Vec<NodeId>> = HashMap::with_capacity(all_nodes.len());
        let mut requested: BTreeMap<String, Vec<NodeView>> = distinct_values
            .iter()
            .cloned()
            .map(|value| (value, Vec::new()))
            .collect();
        for view in all_nodes {
            if let Some(Value::Str(current)) = view.properties.get(property) {
                index.entry(current.clone()).or_default().push(view.id);
                if let Some(out) = requested.get_mut(current) {
                    out.push(view);
                }
            }
        }
        for ids in index.values_mut() {
            ids.sort_unstable();
            ids.dedup();
        }
        for views in requested.values_mut() {
            views.sort_by_key(|view| view.id);
        }
        if let Some(cache) = &self.property_index_cache {
            let generation = self
                .property_index_generation
                .unwrap_or_else(|| cache.generation());
            cache.insert_global_at(property.to_string(), Arc::new(index), generation);
        }
        Ok(values
            .iter()
            .map(|value| requested.get(value).cloned().unwrap_or_default())
            .collect())
    }

    async fn batch_confirm_global_property_candidates(
        &self,
        property: &str,
        values: &[String],
        candidates: BTreeMap<String, BTreeSet<NodeId>>,
    ) -> Result<Vec<Vec<NodeView>>> {
        let ids: Vec<NodeId> = candidates
            .values()
            .flat_map(|ids| ids.iter().copied())
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect();
        let resolved = self.batch_lookup_nodes("", &ids).await?;
        let by_id: HashMap<NodeId, NodeView> = ids
            .into_iter()
            .zip(resolved)
            .filter_map(|(id, view)| view.map(|view| (id, view)))
            .collect();

        let mut confirmed: BTreeMap<String, Vec<NodeView>> = BTreeMap::new();
        for (value, candidate_ids) in candidates {
            let mut matches = Vec::with_capacity(candidate_ids.len());
            for id in candidate_ids {
                let Some(view) = by_id.get(&id) else {
                    continue;
                };
                if matches!(view.properties.get(property), Some(Value::Str(current)) if current == &value)
                {
                    matches.push(view.clone());
                }
            }
            // `candidate_ids` is a BTreeSet, so this is already NodeId order.
            confirmed.insert(value, matches);
        }
        Ok(values
            .iter()
            .map(|value| confirmed.get(value).cloned().unwrap_or_default())
            .collect())
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
        let cache_key = self.node_cache_key(label, id);
        if let Some(cached) = self.node_cache.get(&cache_key) {
            namidb_core::profile::record("Snapshot::lookup_node.l1_hit", 0);
            return Ok(cached);
        }

        // L2: cross-snapshot NodeViewCache (RFC-019). Optional.
        // Slot key uses the logical node generation: an edge-only commit can
        // reuse the view, while a node mutation advances the generation.
        if let Some(shared) = &self.shared_node_cache {
            if let Some(cached) = shared.get(&cache_key) {
                namidb_core::profile::record("Snapshot::lookup_node.l2_hit", 0);
                // Promote into L1 so subsequent intra-snap calls skip L2.
                self.node_cache.insert(cache_key.clone(), cached.clone());
                return Ok(cached);
            }
        }

        // L3: cold SST walk. Resolve the id-primary record, then keep it only
        // if it actually carries `label` (the cache slot is per `(label, id)`).
        // An empty label is unconstrained id-primary resolution, matching
        // `batch_lookup_nodes`: both paths share the `(label, id)` cache
        // keyspace, so diverging semantics would let one path cache an answer
        // the other contradicts.
        let result = self
            .lookup_node_by_id(id)
            .await?
            .filter(|v| label.is_empty() || v.labels.contains(label));
        // Insert into L1.
        self.node_cache.insert(cache_key.clone(), result.clone());
        // Insert into L2 if attached.
        if let Some(shared) = &self.shared_node_cache {
            shared.insert(cache_key, result.clone());
        }
        Ok(result)
    }

    /// Prove label membership for many node ids without materialising their
    /// property maps.
    ///
    /// This accelerator is exact only when persisted node ranges are
    /// disjoint and there is no node memtable: in that layout each id can
    /// belong to at most one SST, and its SST-bound `(LabelId, NodeId)`
    /// sidecar contains the complete live membership set. Missing/corrupt
    /// sidecars, overlapping generations and an empty `labels` constraint
    /// return `None`; callers must then use the ordinary node point reader.
    ///
    /// The result preserves input order and duplicates. `false` covers an
    /// absent/tombstoned node as well as a live node missing any requested
    /// label. B+tree probes are grouped by page and capped inside
    /// `batch_label_contains_from_source`, so a high-degree graph expansion
    /// does not issue one header/footer read or hydrate one embedding per
    /// relationship endpoint.
    pub async fn try_batch_nodes_have_labels(
        &self,
        labels: &[String],
        ids: &[NodeId],
    ) -> Result<Option<Vec<bool>>> {
        if ids.is_empty() {
            return Ok(Some(Vec::new()));
        }
        if labels.is_empty() {
            // Membership in zero labels does not prove that the endpoint
            // itself exists; retain the authoritative node point path.
            if let Some(cache) = &self.cache {
                cache.record_label_membership_fallback();
            }
            return Ok(None);
        }
        let Some(descriptors) = self.disjoint_node_descriptors() else {
            if let Some(cache) = &self.cache {
                cache.record_label_membership_fallback();
            }
            return Ok(None);
        };
        let mut label_ids = Vec::with_capacity(labels.len());
        for label in labels {
            let Some(label_id) = self.manifest.manifest.label_dict.id(label) else {
                if let Some(cache) = &self.cache {
                    cache.record_label_membership_fast_path();
                }
                return Ok(Some(vec![false; ids.len()]));
            };
            label_ids.push(label_id);
        }

        let mut probes_by_descriptor = BTreeMap::<usize, Vec<(usize, [u8; 16])>>::new();
        for (position, id) in ids.iter().enumerate() {
            let id_bytes = *id.as_bytes();
            let candidate = descriptors.partition_point(|descriptor_index| {
                self.manifest.manifest.ssts[*descriptor_index].max_key < id_bytes
            });
            let Some(&descriptor_index) = descriptors.get(candidate) else {
                continue;
            };
            let descriptor = &self.manifest.manifest.ssts[descriptor_index];
            if descriptor.min_key <= id_bytes && id_bytes <= descriptor.max_key {
                probes_by_descriptor
                    .entry(descriptor_index)
                    .or_default()
                    .push((position, id_bytes));
            }
        }

        let mut output = vec![false; ids.len()];
        for (descriptor_index, probes) in probes_by_descriptor {
            let descriptor = &self.manifest.manifest.ssts[descriptor_index];
            let Some(index) = &descriptor.label_index else {
                if let Some(cache) = &self.cache {
                    cache.record_label_membership_fallback();
                }
                return Ok(None);
            };
            if index.format != PropertyIndexFormat::PagedV1
                || index.per_label_counts.is_empty()
                || self
                    .validated_node_descriptor_live_count(descriptor)
                    .is_none()
            {
                if let Some(cache) = &self.cache {
                    cache.record_label_membership_fallback();
                }
                return Ok(None);
            }
            let absolute = format!("{}/{}", self.paths.namespace_prefix().as_ref(), index.path);
            let source = match self
                .pinned_sidecar_source(&absolute, Some(index.size_bytes))
                .await
            {
                Ok(source) => source,
                Err(error) if optional_accelerator_fallback(&error) => {
                    if let Some(cache) = &self.cache {
                        cache.record_label_membership_fallback();
                    }
                    return Ok(None);
                }
                Err(error) => return Err(error),
            };
            let probe_ids = probes.iter().map(|(_, id)| *id).collect::<Vec<_>>();
            let mut descriptor_matches = vec![true; probes.len()];
            for label_id in &label_ids {
                let (matches, stats) =
                    match crate::sst::paged_index::batch_label_contains_from_source(
                        &source,
                        label_id.get(),
                        &probe_ids,
                        *descriptor.id.as_bytes(),
                        &index.per_label_counts,
                    )
                    .await
                    {
                        Ok(result) => result,
                        Err(error) if optional_accelerator_fallback(&error) => {
                            if let Some(cache) = &self.cache {
                                cache.record_label_membership_fallback();
                            }
                            return Ok(None);
                        }
                        Err(error) => return Err(error),
                    };
                if let Some(cache) = &self.cache {
                    cache.record_label_membership_probe(probe_ids.len(), stats);
                }
                if matches.len() != probes.len() || stats.index_entries != index.posting_count {
                    if let Some(cache) = &self.cache {
                        cache.record_label_membership_fallback();
                    }
                    return Ok(None);
                }
                for (combined, one_label) in descriptor_matches.iter_mut().zip(matches) {
                    *combined &= one_label;
                }
                if descriptor_matches.iter().all(|matched| !*matched) {
                    break;
                }
            }
            for ((position, _), matched) in probes.into_iter().zip(descriptor_matches) {
                output[position] = matched;
            }
        }
        if let Some(cache) = &self.cache {
            cache.record_label_membership_fast_path();
        }
        Ok(Some(output))
    }

    /// Batched analogue of [`Self::lookup_node`]: probe many `ids` for
    /// the same `label` in one pass over the node SST set. An empty label is
    /// the internal id-primary scope and returns each node with its complete
    /// label set without applying a membership filter.
    ///
    /// Returns a `Vec<Option<NodeView>>` aligned 1:1 with `ids`. `None`
    /// means absent or tombstoned at this snapshot. Duplicates in `ids`
    /// resolve to equivalent `NodeView` values. `NodeView` owns its maps, so
    /// callers should consume this returned vector directly instead of doing
    /// a second point lookup and clone.
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
        let unique_ids = id_to_outputs.keys().copied().collect::<Vec<_>>();
        let l1_keys = unique_ids
            .iter()
            .map(|id_bytes| {
                self.node_cache_key(label, NodeId::from_uuid(Uuid::from_bytes(*id_bytes)))
            })
            .collect::<Vec<_>>();
        for ((id_bytes, outputs), cached) in unique_ids
            .iter()
            .map(|id_bytes| (id_bytes, &id_to_outputs[id_bytes]))
            .zip(self.node_cache.get_many(&l1_keys))
        {
            if let Some(cached) = cached {
                namidb_core::profile::record("Snapshot::batch_lookup_nodes.l1_hit", 0);
                for &i in outputs {
                    out[i] = cached.clone();
                }
                pending.remove(id_bytes);
            }
        }

        // L2 cache pass: same logic against the cross-snapshot cache.
        if let Some(shared) = &self.shared_node_cache {
            let pending_ids = pending.iter().copied().collect::<Vec<_>>();
            let pending_keys = pending_ids
                .iter()
                .map(|id_bytes| {
                    self.node_cache_key(label, NodeId::from_uuid(Uuid::from_bytes(*id_bytes)))
                })
                .collect::<Vec<_>>();
            for ((id_bytes, key), cached) in pending_ids
                .iter()
                .zip(&pending_keys)
                .zip(shared.get_many(&pending_keys))
            {
                if let Some(cached) = cached {
                    namidb_core::profile::record("Snapshot::batch_lookup_nodes.l2_hit", 0);
                    for &i in &id_to_outputs[id_bytes] {
                        out[i] = cached.clone();
                    }
                    self.node_cache.insert(key.clone(), cached);
                    pending.remove(id_bytes);
                }
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

            // Current `.nloc2` objects append an exact NodeId -> record B+tree
            // after the backward-compatible ordinal locator. Prefer it before
            // even opening the Parquet footer: a random existing-node update
            // then fetches only that node's compressed JSON payload instead of
            // decoding the 1 MiB `__overflow_json` page that happens to contain
            // its row. At legal-corpus scale this is the difference between a
            // few KiB and ~MiB of allocator high-water per updated node.
            if crate::manifest::node_locator_has_exact_records(desc) {
                if let Some(locator) = desc
                    .node_locator
                    .as_ref()
                    .filter(|locator| locator.entry_count == desc.row_count)
                {
                    let locator_absolute = format!(
                        "{}/{}",
                        self.paths.namespace_prefix().as_ref(),
                        locator.path
                    );
                    let probed = match self
                        .pinned_sidecar_source(&locator_absolute, Some(locator.size_bytes))
                        .await
                    {
                        Ok(source) => {
                            crate::sst::paged_index::probe_node_records_from_source(
                                &source,
                                &sorted_pending,
                            )
                            .await
                        }
                        Err(error) => Err(error),
                    };
                    match probed {
                        Ok((records, stats)) if stats.index_entries == locator.entry_count => {
                            let decoded = records
                                .into_iter()
                                .map(|(id_bytes, encoded)| {
                                    let (lsn, op) = decode_exact_node_record(&encoded)?;
                                    let id = NodeId::from_uuid(Uuid::from_bytes(id_bytes));
                                    let view = match op {
                                        MemOp::Tombstone => None,
                                        MemOp::Upsert(payload) => Some(node_view_from_payload(
                                            id,
                                            lsn,
                                            &payload,
                                            &self.manifest.manifest.label_dict,
                                            &desc.scope,
                                        )?),
                                    };
                                    Ok((id_bytes, lsn, view))
                                })
                                .collect::<Result<Vec<_>>>();
                            match decoded {
                                Ok(decoded) => {
                                    if let Some(cache) = &self.cache {
                                        cache.record_node_locator_probe(stats);
                                    }
                                    for (id_bytes, lsn, view) in decoded {
                                        match winners.get(&id_bytes) {
                                            Some((existing_lsn, _)) if *existing_lsn >= lsn => {}
                                            _ => {
                                                winners.insert(id_bytes, (lsn, view));
                                            }
                                        }
                                    }
                                    // Matching keys were returned and checked;
                                    // all other requested keys are authoritative
                                    // misses because both sidecar trees cover
                                    // exactly `row_count` entries.
                                    continue;
                                }
                                Err(error) => {
                                    tracing::warn!(
                                        path = %locator.path,
                                        error = %error,
                                        "exact node-record index is corrupt; falling back to Parquet"
                                    );
                                }
                            }
                        }
                        Ok((_records, _stats)) => {
                            // Count disagreement means the extension cannot be
                            // authoritative for absence. Retain the compatible
                            // ordinal/Parquet fallback below.
                        }
                        Err(error) if optional_accelerator_fallback(&error) => {
                            tracing::warn!(
                                path = %locator.path,
                                error = %error,
                                "exact node-record index unavailable; falling back to Parquet"
                            );
                        }
                        Err(error) => return Err(error),
                    }
                }
            }

            let label_def = self.label_def_for_node_sst(desc);
            let md = self.node_sst_metadata(desc, &absolute).await?;
            if let Some(locator) = desc
                .node_locator
                .as_ref()
                .filter(|locator| locator.entry_count == desc.row_count)
            {
                let locator_absolute = format!(
                    "{}/{}",
                    self.paths.namespace_prefix().as_ref(),
                    locator.path
                );
                let located = match self
                    .pinned_sidecar_source(&locator_absolute, Some(locator.size_bytes))
                    .await
                {
                    Ok(source) => {
                        crate::sst::paged_index::probe_node_locator_from_source(
                            &source,
                            &sorted_pending,
                        )
                        .await
                    }
                    Err(error) => Err(error),
                };
                match located {
                    Ok((located, locator_stats)) => {
                        if locator_stats.index_entries != locator.entry_count {
                            // A partial/stale locator is only an accelerator,
                            // never an authority for absence.
                        } else {
                            if let Some(cache) = &self.cache {
                                cache.record_node_locator_probe(locator_stats);
                            }
                            if located.is_empty() {
                                continue;
                            }
                            let mut ordinals: Vec<u64> = located.values().copied().collect();
                            ordinals.sort_unstable();
                            ordinals.dedup();
                            if ordinals.iter().all(|ordinal| *ordinal < desc.row_count) {
                                let batches = self
                                    .decode_node_rows_by_ordinals(
                                        desc, &absolute, &label_def, &md, &ordinals,
                                    )
                                    .await?;
                                let decoded_ids: BTreeSet<[u8; 16]> = batches
                                    .iter()
                                    .filter_map(|batch| {
                                        batch.column_by_name(COL_NODE_ID).and_then(|column| {
                                            column.as_any().downcast_ref::<FixedSizeBinaryArray>()
                                        })
                                    })
                                    .flat_map(|ids| {
                                        (0..ids.len())
                                            .filter_map(|row| ids.value(row).try_into().ok())
                                    })
                                    .collect();
                                if decoded_ids.len() == located.len()
                                    && located.keys().all(|id| decoded_ids.contains(id))
                                {
                                    batch_harvest_node_rows(
                                        &batches,
                                        &label_def,
                                        &self.manifest.manifest.label_dict,
                                        &desc.scope,
                                        &pending,
                                        &mut winners,
                                    )?;
                                    continue;
                                }
                            }
                            // A valid-looking locator that maps a key to the wrong
                            // or out-of-bounds ordinal is not authoritative. Fall
                            // through to the row-group path rather than returning
                            // a false miss.
                        }
                    }
                    Err(error) if optional_accelerator_fallback(&error) => {
                        tracing::warn!(
                            path = %locator.path,
                            error = %error,
                            "node locator unavailable; falling back to authoritative Parquet"
                        );
                    }
                    Err(error) => return Err(error),
                }
            }
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
                        .get(&(absolute.clone(), rg)),
                };
                match hit {
                    Some(b) => decoded.push(b),
                    None => missing.push(rg),
                }
            }
            if !missing.is_empty() {
                let missing_rows = missing.iter().fold(0_usize, |total, &rg| {
                    total.saturating_add(md.row_group(rg).num_rows() as usize)
                });
                // A large id-primary SST can have very wide rows (overflow
                // JSON, embeddings, text). When a small set of MERGE hits is
                // spread over most row groups, caching/decompressing every
                // complete row recreates a corpus-sized scan. Let Parquet
                // evaluate a narrow node_id RowFilter first and materialise
                // only matching payloads. Dense scans still populate the
                // decoded row-group cache, preserving traversal locality.
                let sparse = missing_rows > sorted_pending.len().saturating_mul(8)
                    && missing_rows > 8 * 1024;
                if sparse {
                    let batches = self
                        .decode_node_rows_for_keys(
                            desc,
                            &absolute,
                            &label_def,
                            &md,
                            &missing,
                            Arc::new(pending.clone()),
                        )
                        .await?;
                    batch_harvest_node_rows(
                        &batches,
                        &label_def,
                        &self.manifest.manifest.label_dict,
                        &desc.scope,
                        &pending,
                        &mut winners,
                    )?;
                } else {
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
                                let key = (absolute.clone(), rg);
                                let weight =
                                    crate::cache::decoded_node_row_group_weight(&key, &batches);
                                self.decoded_node_row_groups.lock().unwrap().insert(
                                    key,
                                    batches.clone(),
                                    weight,
                                );
                            }
                        }
                        decoded.push(batches);
                    }
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
        for id_bytes in &pending {
            let view = winners
                .remove(id_bytes)
                .map(|(_, v)| v)
                .unwrap_or(None)
                .filter(|v| label.is_empty() || v.labels.contains(label));
            for &i in &id_to_outputs[id_bytes] {
                out[i] = view.clone();
            }
            let id = NodeId::from_uuid(Uuid::from_bytes(*id_bytes));
            let cache_key = self.node_cache_key(label, id);
            self.node_cache.insert(cache_key.clone(), view.clone());
            if let Some(ref shared) = shared {
                shared.insert(cache_key, view);
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
        // The RFC-019 parity oracle mirrors `lookup_node` exactly, including
        // the empty-label-is-unconstrained rule.
        Ok(self
            .lookup_node_by_id(id)
            .await?
            .filter(|v| label.is_empty() || v.labels.contains(label)))
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
            load_node_sst_metadata_async(self.store.clone(), Path::from(absolute), desc.size_bytes)
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

    /// Decode only rows whose node_id belongs to `keys`, without admitting
    /// the resulting partial row groups into the complete-row-group cache.
    async fn decode_node_rows_for_keys(
        &self,
        desc: &SstDescriptor,
        absolute: &str,
        label_def: &LabelDef,
        md: &Arc<ParquetMetaData>,
        row_groups: &[usize],
        keys: Arc<std::collections::HashSet<[u8; 16]>>,
    ) -> Result<Vec<RecordBatch>> {
        let use_ranged = self
            .ranged_mode
            .enable_for(desc.size_bytes, self.ranged_threshold_bytes);
        let local_body = if !use_ranged {
            Some(self.get_sst_body(desc).await?)
        } else {
            self.cache_get(absolute)
        };
        if let Some(body) = local_body {
            let reader = NodeSstReader::open(label_def.clone(), body)?;
            let batches = reader.scan_row_groups_for_keys(md, row_groups, keys)?;
            if let Some(cache) = &self.cache {
                cache.record_sparse_node_filter(batches.iter().map(RecordBatch::num_rows).sum());
            }
            return Ok(batches);
        }
        let batches = node_scan_row_groups_for_keys_async(
            self.store.clone(),
            Path::from(absolute),
            desc.size_bytes,
            label_def,
            row_groups.to_vec(),
            keys,
            Some(md.clone()),
        )
        .await?;
        if let Some(cache) = &self.cache {
            cache.record_sparse_node_filter(batches.iter().map(RecordBatch::num_rows).sum());
        }
        Ok(batches)
    }

    /// Decode exact locator-provided physical rows. No partial row-group is
    /// inserted into the complete row-group cache.
    async fn decode_node_rows_by_ordinals(
        &self,
        desc: &SstDescriptor,
        absolute: &str,
        label_def: &LabelDef,
        md: &Arc<ParquetMetaData>,
        ordinals: &[u64],
    ) -> Result<Vec<RecordBatch>> {
        let use_ranged = self
            .ranged_mode
            .enable_for(desc.size_bytes, self.ranged_threshold_bytes);
        let local_body = if !use_ranged {
            Some(self.get_sst_body(desc).await?)
        } else {
            self.cache_get(absolute)
        };
        if let Some(body) = local_body {
            let reader = NodeSstReader::open(label_def.clone(), body)?;
            return reader.scan_rows_by_ordinals(md, ordinals);
        }
        node_scan_rows_by_ordinals_async(
            self.store.clone(),
            Path::from(absolute),
            desc.size_bytes,
            label_def,
            ordinals.to_vec(),
            Some(md.clone()),
        )
        .await
    }

    /// Id-primary lookup: resolve the last-LSN-wins record for `id` across the
    /// memtable and every node SST, decoding and returning its complete label
    /// set. Label-agnostic query operators should use this directly instead of
    /// trial-probing every observed label; the physical node key is already the
    /// id, so that loop repeats the same point read `O(label_count)` times.
    ///
    /// [`Self::lookup_node`] layers its per-label caches and membership filter
    /// over this primitive.
    pub async fn lookup_node_by_id(&self, id: NodeId) -> Result<Option<NodeView>> {
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

    /// Exact `(edge_type, src, dst)` relationship lookup.
    ///
    /// This is the storage primitive behind bound-endpoint `MERGE` and
    /// Expand-Into. It probes the ordered memtable key and then only the
    /// forward SST ranges whose source key can contain `src`; within a
    /// high-degree partner block the destination is binary-searched in place.
    /// Overlapping L0/L1 versions are reconciled by LSN, and property streams
    /// are decoded only for the final live winner.
    #[instrument(skip(self), fields(edge_type = edge_type, src = %src, dst = %dst))]
    pub async fn lookup_edge_via_sst(
        &self,
        edge_type: &str,
        src: NodeId,
        dst: NodeId,
    ) -> Result<Option<EdgeView>> {
        Ok(self
            .batch_edge_point_lookup_via_sst(edge_type, &[(src, dst)], true)
            .await?
            .pop()
            .flatten())
    }

    /// Exact relationship existence probe without decoding property streams.
    pub async fn contains_edge_via_sst(
        &self,
        edge_type: &str,
        src: NodeId,
        dst: NodeId,
    ) -> Result<bool> {
        Ok(self
            .batch_edge_point_lookup_via_sst(edge_type, &[(src, dst)], false)
            .await?
            .pop()
            .flatten()
            .is_some())
    }

    /// Batched exact relationship lookup, preserving input order and
    /// duplicates. Current forward SSTs are probed through one range-readable
    /// B+tree walk per SST; legacy SSTs are opened at most once for the whole
    /// batch.
    pub async fn batch_lookup_edges_via_sst(
        &self,
        edge_type: &str,
        pairs: &[(NodeId, NodeId)],
    ) -> Result<Vec<Option<EdgeView>>> {
        self.batch_edge_point_lookup_via_sst(edge_type, pairs, true)
            .await
    }

    /// Existence-only companion to [`Self::batch_lookup_edges_via_sst`].
    pub async fn batch_contains_edges_via_sst(
        &self,
        edge_type: &str,
        pairs: &[(NodeId, NodeId)],
    ) -> Result<Vec<bool>> {
        Ok(self
            .batch_edge_point_lookup_via_sst(edge_type, pairs, false)
            .await?
            .into_iter()
            .map(|edge| edge.is_some())
            .collect())
    }

    async fn batch_edge_point_lookup_via_sst(
        &self,
        edge_type: &str,
        pairs: &[(NodeId, NodeId)],
        materialize_properties: bool,
    ) -> Result<Vec<Option<EdgeView>>> {
        namidb_core::profile_scope!("Snapshot::batch_edge_point_lookup_via_sst");
        if pairs.is_empty() {
            return Ok(Vec::new());
        }
        let unique_pairs: BTreeSet<(NodeId, NodeId)> = pairs.iter().copied().collect();
        let mut winners: BTreeMap<(NodeId, NodeId), Option<(u64, EdgePointWinner)>> = unique_pairs
            .iter()
            .copied()
            .map(|pair| (pair, None))
            .collect();

        // The physical memtable identity already is `(type, src, dst)`, so
        // both committed and staged writes are direct ordered-map probes.
        for &(src, dst) in &unique_pairs {
            if let Some(entry) = self.edge_mem_entry(edge_type, src, dst) {
                let source = match &entry.op {
                    MemOp::Tombstone => EdgePointWinner::Tombstone,
                    MemOp::Upsert(payload) => {
                        let properties = if materialize_properties {
                            EdgeWriteRecord::decode(payload)?.properties
                        } else {
                            BTreeMap::new()
                        };
                        EdgePointWinner::Materialized(properties)
                    }
                };
                winners.insert((src, dst), Some((entry.lsn, source)));
            }
        }

        // Group every requested pair by the forward SST ranges that can
        // contain its source. A descriptor is opened/probed once regardless
        // of UNWIND cardinality or duplicate endpoint pairs.
        let mut probes_by_sst: BTreeMap<usize, BTreeSet<(NodeId, NodeId)>> = BTreeMap::new();
        for &(src, dst) in &unique_pairs {
            for idx in self.manifest.index.lookup_candidates(
                &self.manifest.manifest.ssts,
                SstKind::EdgesFwd,
                edge_type,
                src.as_bytes(),
            ) {
                probes_by_sst.entry(idx).or_default().insert((src, dst));
            }
        }
        let mut candidates: Vec<usize> = probes_by_sst.keys().copied().collect();
        // Probe newest candidate ranges first. Once a point version wins,
        // `max_lsn` lets us skip every older SST body without opening it.
        // This keeps point probes bounded while an L0 backlog is draining.
        candidates.sort_unstable_by(|left, right| {
            self.manifest.manifest.ssts[*right]
                .max_lsn
                .cmp(&self.manifest.manifest.ssts[*left].max_lsn)
                .then_with(|| right.cmp(left))
        });
        for idx in candidates {
            let desc = &self.manifest.manifest.ssts[idx];
            let desc_pairs: Vec<(NodeId, NodeId)> = probes_by_sst
                .remove(&idx)
                .unwrap_or_default()
                .into_iter()
                .filter(|pair| {
                    winners
                        .get(pair)
                        .and_then(Option::as_ref)
                        .is_none_or(|(winner_lsn, _)| desc.max_lsn > *winner_lsn)
                })
                .collect();
            if desc_pairs.is_empty() {
                continue;
            }

            // Current `.ep.csr` bodies have a deterministic optional point
            // sidecar. It is an accelerator, never the authority: any missing,
            // stale or corrupt sidecar falls through to the exact CSR below.
            if let Some(relative) = crate::manifest::edge_point_sidecar_path(desc) {
                let absolute = format!("{}/{}", self.paths.namespace_prefix().as_ref(), relative);
                let raw_pairs: Vec<([u8; 16], [u8; 16])> = desc_pairs
                    .iter()
                    .map(|(src, dst)| (*src.as_bytes(), *dst.as_bytes()))
                    .collect();
                let probed = match self.pinned_sidecar_source(&absolute, None).await {
                    Ok(source) => {
                        crate::sst::paged_index::probe_edge_points_from_source(&source, &raw_pairs)
                            .await
                    }
                    Err(error) => Err(error),
                };
                let decoded = match probed {
                    Ok((found, stats)) if stats.index_entries == desc.row_count => {
                        if let Some(cache) = &self.cache {
                            cache.record_edge_point_probe(stats);
                        }
                        let mut decoded = BTreeMap::new();
                        let mut valid = true;
                        for (raw_pair, value) in found {
                            match crate::sst::edges::point_index::decode(
                                &value,
                                materialize_properties,
                            ) {
                                Ok(value) => {
                                    decoded.insert(
                                        (
                                            NodeId::from_uuid(Uuid::from_bytes(raw_pair.0)),
                                            NodeId::from_uuid(Uuid::from_bytes(raw_pair.1)),
                                        ),
                                        value,
                                    );
                                }
                                Err(error) if optional_accelerator_fallback(&error) => {
                                    tracing::warn!(
                                        path = %relative,
                                        error = %error,
                                        "edge point accelerator value is corrupt; falling back to CSR"
                                    );
                                    valid = false;
                                    break;
                                }
                                Err(error) => return Err(error),
                            }
                        }
                        valid.then_some(decoded)
                    }
                    Ok((_found, stats)) => {
                        tracing::warn!(
                            path = %relative,
                            expected_entries = desc.row_count,
                            actual_entries = stats.index_entries,
                            "edge point accelerator is stale/partial; falling back to CSR"
                        );
                        None
                    }
                    Err(error) if optional_accelerator_fallback(&error) => {
                        tracing::warn!(
                            path = %relative,
                            error = %error,
                            "edge point accelerator unavailable; falling back to CSR"
                        );
                        None
                    }
                    Err(error) => return Err(error),
                };
                if let Some(decoded) = decoded {
                    for (pair, point) in decoded {
                        update_edge_point_winner(
                            &mut winners,
                            pair,
                            point.lsn,
                            if point.tombstone {
                                EdgePointWinner::Tombstone
                            } else {
                                EdgePointWinner::Materialized(point.properties)
                            },
                        );
                    }
                    // A valid complete sidecar makes every absent key an
                    // authoritative miss for this SST.
                    continue;
                }
            }

            let admitted_sources: BTreeSet<NodeId> = {
                let mut admitted = BTreeSet::new();
                let sources: BTreeSet<NodeId> = desc_pairs.iter().map(|(src, _)| *src).collect();
                for src in sources {
                    if self.bloom_admits(desc, src.as_bytes()).await? {
                        admitted.insert(src);
                    }
                }
                admitted
            };
            if admitted_sources.is_empty() {
                continue;
            }
            let absolute = format!("{}/{}", self.paths.namespace_prefix().as_ref(), desc.path);
            let reader = self.fetch_paged_edge_reader(&absolute).await?;
            for (src, dst) in desc_pairs {
                if !admitted_sources.contains(&src) {
                    continue;
                }
                let Some(point) = reader
                    .lookup_partner(src.as_bytes(), dst.as_bytes())
                    .await?
                else {
                    continue;
                };
                let source = if point.tombstone {
                    EdgePointWinner::Tombstone
                } else {
                    EdgePointWinner::Persisted {
                        absolute: absolute.clone(),
                        edge_offset: point.edge_offset,
                    }
                };
                update_edge_point_winner(&mut winners, (src, dst), point.lsn, source);
            }
        }

        let mut resolved: BTreeMap<(NodeId, NodeId), Option<EdgeView>> = BTreeMap::new();
        for ((src, dst), winner) in winners {
            let Some((lsn, winner)) = winner else {
                resolved.insert((src, dst), None);
                continue;
            };
            let properties = match winner {
                EdgePointWinner::Tombstone => {
                    resolved.insert((src, dst), None);
                    continue;
                }
                EdgePointWinner::Materialized(properties) => properties,
                EdgePointWinner::Persisted {
                    absolute,
                    edge_offset,
                } => {
                    if materialize_properties {
                        // Hydrate exactly the winning row through the paged
                        // property pages; only after LWW has selected a live
                        // winning point. Legacy Arrow sections fall back to
                        // the eager body inside `read_property_rows`.
                        let paged_reader = self.fetch_paged_edge_reader(&absolute).await?;
                        let row_start = u64::try_from(edge_offset)
                            .map_err(|_| Error::invariant("edge offset does not fit u64"))?;
                        let row_range = row_start..row_start.checked_add(1).ok_or_else(|| {
                            Error::invariant("edge point row range overflows u64")
                        })?;
                        let overflow = paged_reader
                            .read_property_rows(OVERFLOW_JSON_NAME, row_range.clone())
                            .await?;
                        let declared_property_names: Vec<String> = self
                            .manifest
                            .manifest
                            .schema
                            .edge_type(edge_type)
                            .map(|def| def.properties.iter().map(|p| p.name.clone()).collect())
                            .unwrap_or_default();
                        let mut declared: Vec<(String, Vec<Option<String>>)> =
                            Vec::with_capacity(declared_property_names.len());
                        for name in declared_property_names {
                            if let Some(values) = paged_reader
                                .read_property_rows(&name, row_range.clone())
                                .await?
                            {
                                declared.push((name, values));
                            }
                        }
                        decode_edge_properties(
                            overflow.as_ref().and_then(|values| values.first()),
                            &declared,
                            0,
                        )?
                    } else {
                        BTreeMap::new()
                    }
                }
            };
            resolved.insert(
                (src, dst),
                Some(EdgeView {
                    edge_type: edge_type.to_string(),
                    src,
                    dst,
                    properties,
                    lsn,
                }),
            );
        }

        Ok(pairs
            .iter()
            .map(|pair| resolved.get(pair).cloned().flatten())
            .collect())
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
        self.scan_nodes_with_optional_label(Some(label), predicates, projection)
            .await
    }

    /// Visit one compacted label scan a bounded batch at a time.
    ///
    /// In the steady-state layout, node SST key ranges are disjoint and there
    /// are no node memtable entries. This path then reads selected Parquet
    /// columns lazily by row group and never materialises the corpus. During a
    /// transient overlapping generation it preserves exact last-write-wins
    /// semantics by falling back to ordinary reconciliation before invoking
    /// the visitor. Callers that make multiple passes (for example exact BM25
    /// corpus statistics followed by scoring) can therefore remain bounded in
    /// the normal object-store-native case without changing snapshot results.
    pub async fn visit_label_with_projection<F, E>(
        &self,
        label: &str,
        projection: &[String],
        mut visitor: F,
    ) -> std::result::Result<(), E>
    where
        F: FnMut(NodeView) -> std::result::Result<(), E>,
        E: From<Error>,
    {
        let Some(descriptors) = self.disjoint_node_descriptors() else {
            for view in self
                .scan_nodes_with_optional_label(Some(label), &[], Some(projection))
                .await
                .map_err(E::from)?
            {
                visitor(view)?;
            }
            return Ok(());
        };

        let dict = &self.manifest.manifest.label_dict;
        let requested_projection: BTreeSet<&str> = projection.iter().map(String::as_str).collect();
        for idx in descriptors {
            if !node_sst_can_contain_label(&self.manifest.manifest, idx, label) {
                continue;
            }
            let desc = &self.manifest.manifest.ssts[idx];
            let sst_label_def = self.label_def_for_node_sst(desc);
            let absolute = format!("{}/{}", self.paths.namespace_prefix().as_ref(), desc.path);
            if let Some(reader) = self.node_property_reader(desc).await.map_err(E::from)? {
                match reader.verify_properties(projection).await {
                    Ok(_) => {
                        let cached_metadata = self
                            .cache
                            .as_ref()
                            .and_then(|cache| cache.get_metadata(&absolute));
                        let metadata_was_cached = cached_metadata.is_some();
                        let (mut stream, metadata) = node_scan_limited_async(
                            self.store.clone(),
                            Path::from(absolute.clone()),
                            desc.size_bytes,
                            &sst_label_def,
                            &[],
                            Some(&[]),
                            cached_metadata,
                        )
                        .await
                        .map_err(E::from)?;
                        if !metadata_was_cached {
                            if let Some(cache) = &self.cache {
                                cache.insert_metadata(absolute.clone(), metadata);
                            }
                        }
                        let mut next_ordinal = 0_u64;
                        while let Some(batches) = stream.next_row_group().await.map_err(E::from)? {
                            for batch in batches {
                                let batch = batch.map_err(|error| {
                                    E::from(Error::invariant(format!(
                                        "projected Parquet visitor read: {error}"
                                    )))
                                })?;
                                let Some(candidates) = self
                                    .project_node_property_batch(
                                        reader.as_ref(),
                                        desc,
                                        projection,
                                        &batch,
                                        next_ordinal,
                                    )
                                    .await
                                    .map_err(E::from)?
                                else {
                                    return Err(E::from(Error::invariant(
                                        "validated node property pages became unreadable",
                                    )));
                                };
                                next_ordinal = next_ordinal
                                    .checked_add(batch.num_rows() as u64)
                                    .ok_or_else(|| {
                                        E::from(Error::invariant(
                                            "node property ordinal exceeds u64",
                                        ))
                                    })?;
                                for (_, _, view) in candidates {
                                    let Some(view) = view else {
                                        continue;
                                    };
                                    if view.labels.contains(label) {
                                        visitor(view)?;
                                    }
                                }
                            }
                        }
                        if next_ordinal != desc.row_count {
                            return Err(E::from(Error::invariant(
                                "node property/Parquet row-count mismatch",
                            )));
                        }
                        continue;
                    }
                    Err(error) if optional_accelerator_fallback(&error) => {
                        tracing::warn!(
                            sst_id = %desc.id,
                            %error,
                            "node property projection scrub failed; using exact Parquet fallback"
                        );
                    }
                    Err(error) => return Err(E::from(error)),
                }
            }
            let context = LimitedNodeBatchContext {
                sst_label_def: &sst_label_def,
                desc,
                dict,
                label: Some(label),
                predicates: &[],
                decode_projection: Some(projection),
                requested_projection: Some(&requested_projection),
                limit: usize::MAX,
            };

            let cached_body = self.cache_get(&absolute);
            if cached_body.is_some() || matches!(self.ranged_mode, RangedMode::Force(false)) {
                let body = match cached_body {
                    Some(body) => body,
                    None => self.get_sst_body(desc).await.map_err(E::from)?,
                };
                let reader = NodeSstReader::open(sst_label_def.clone(), body).map_err(E::from)?;
                let batches = reader
                    .scan_iter_with_predicates_and_projection(&[], Some(projection))
                    .map_err(E::from)?;
                for batch in batches {
                    let mut rows = Vec::new();
                    consume_limited_node_batches(std::iter::once(batch), &context, &mut rows)
                        .map_err(E::from)?;
                    for row in rows {
                        visitor(row)?;
                    }
                }
            } else {
                let cached_metadata = self
                    .cache
                    .as_ref()
                    .and_then(|cache| cache.get_metadata(&absolute));
                let metadata_was_cached = cached_metadata.is_some();
                let (mut stream, metadata) = node_scan_limited_async(
                    self.store.clone(),
                    Path::from(absolute.clone()),
                    desc.size_bytes,
                    &sst_label_def,
                    &[],
                    Some(projection),
                    cached_metadata,
                )
                .await
                .map_err(E::from)?;
                if !metadata_was_cached {
                    if let Some(cache) = &self.cache {
                        cache.insert_metadata(absolute.clone(), metadata);
                    }
                }
                while let Some(batches) = stream.next_row_group().await.map_err(E::from)? {
                    for batch in batches {
                        let mut rows = Vec::new();
                        consume_limited_node_batches(std::iter::once(batch), &context, &mut rows)
                            .map_err(E::from)?;
                        for row in rows {
                            visitor(row)?;
                        }
                    }
                }
            }
        }
        Ok(())
    }

    /// Exact prefix scan used by an order-insensitive Cypher LIMIT.
    ///
    /// In a compacted/disjoint snapshot, physical node order is already the
    /// logical last-write-wins order, so the reader can stop after `limit`
    /// live rows that carry `label` and satisfy every predicate. Any node
    /// memtable entry or overlapping SST range falls back to the ordinary full
    /// reconciliation before truncation.
    pub async fn scan_label_with_predicates_and_projection_limited(
        &self,
        label: &str,
        predicates: &[ScanPredicate],
        projection: Option<&[String]>,
        limit: usize,
    ) -> Result<Vec<NodeView>> {
        self.scan_nodes_with_optional_label_limited(Some(label), predicates, projection, limit)
            .await
    }

    /// Materialise every live node in one label-agnostic pass, with the same
    /// predicate pushdown and property projection as
    /// [`Self::scan_label_with_predicates_and_projection`].
    ///
    /// Each physical node id is reconciled once across the memtable and all
    /// node SSTs, so multi-label nodes appear once and nodes with no labels are
    /// included. This is the storage primitive for Cypher `MATCH (n)` and
    /// typeless vector-search fallback; callers must not emulate it by
    /// concatenating one `scan_label` per observed label.
    #[instrument(skip(self, predicates, projection), fields(predicates = predicates.len(), projection = projection.as_ref().map(|p| p.len()).unwrap_or(0)))]
    pub async fn scan_all_nodes_with_predicates_and_projection(
        &self,
        predicates: &[ScanPredicate],
        projection: Option<&[String]>,
    ) -> Result<Vec<NodeView>> {
        self.scan_nodes_with_optional_label(None, predicates, projection)
            .await
    }

    /// Typeless counterpart to
    /// [`Self::scan_label_with_predicates_and_projection_limited`].
    pub async fn scan_all_nodes_with_predicates_and_projection_limited(
        &self,
        predicates: &[ScanPredicate],
        projection: Option<&[String]>,
        limit: usize,
    ) -> Result<Vec<NodeView>> {
        self.scan_nodes_with_optional_label_limited(None, predicates, projection, limit)
            .await
    }

    /// Prove that every persisted node id has exactly one physical version and
    /// return SST descriptors in global node-id order.
    ///
    /// Ranges are inclusive, so touching endpoints overlap. Empty descriptors
    /// cannot contribute a version and are ignored. Invalid single-descriptor
    /// metadata (`min > max`) is rejected explicitly rather than slipping
    /// through a windows-only overlap check.
    fn disjoint_node_descriptors(&self) -> Option<Vec<usize>> {
        if self.node_entries().next().is_some() {
            return None;
        }
        let mut descriptors: Vec<usize> = self
            .manifest
            .index
            .node_descriptors()
            .into_iter()
            .filter(|idx| self.manifest.manifest.ssts[*idx].row_count > 0)
            .collect();
        if descriptors.iter().any(|idx| {
            let desc = &self.manifest.manifest.ssts[*idx];
            desc.min_key > desc.max_key
        }) {
            return None;
        }
        descriptors.sort_by_key(|idx| self.manifest.manifest.ssts[*idx].min_key);
        if descriptors.windows(2).any(|pair| {
            self.manifest.manifest.ssts[pair[0]].max_key
                >= self.manifest.manifest.ssts[pair[1]].min_key
        }) {
            return None;
        }
        Some(descriptors)
    }

    /// Exact manifest-only node count when the physical node partitions prove
    /// they cannot contain two versions of the same id.
    ///
    /// A single compacted node SST is the dominant steady-state case. Several
    /// SSTs are also safe when their `[min_key, max_key]` ranges are pairwise
    /// disjoint. Any node in the committed/staged memtable, overlapping SST
    /// range, or legacy id-primary descriptor without per-label counts returns
    /// `None`; callers must run the exact reconciliation fallback rather than
    /// trust additive LSM statistics across duplicate versions.
    pub fn metadata_node_count(&self, label: Option<&str>) -> Option<u64> {
        // Validate every node descriptor before disjointness drops empty
        // physical files. Otherwise corrupt metadata such as
        // `row_count = 0, tombstone_count = 1` could silently authorize an
        // exact zero. Per-label counters are cache-like manifest summaries:
        // any contradiction forces the ordinary exact reconciliation path.
        for idx in self.manifest.index.node_descriptors() {
            self.validated_node_descriptor_live_count(&self.manifest.manifest.ssts[idx])?;
        }
        let descriptors = self.disjoint_node_descriptors()?;

        let mut total = 0_u64;
        for idx in descriptors {
            let desc = &self.manifest.manifest.ssts[idx];
            let live = self.validated_node_descriptor_live_count(desc)?;
            match label {
                None => total = total.checked_add(live)?,
                Some(label) => {
                    if let Some(index) = &desc.label_index {
                        if index.per_label_counts.is_empty() {
                            return None;
                        }
                        let label_id = self.manifest.manifest.label_dict.id(label);
                        let count = label_id
                            .and_then(|id| {
                                index
                                    .per_label_counts
                                    .iter()
                                    .find(|(candidate, _)| *candidate == id.get())
                                    .map(|(_, count)| *count)
                            })
                            .unwrap_or(0);
                        total = total.checked_add(count)?;
                    } else if !desc.scope.is_empty() {
                        if desc.scope == label {
                            total = total.checked_add(live)?;
                        }
                    } else {
                        // Pre-label-index id-primary SST: its additive row
                        // count cannot answer one label exactly.
                        return None;
                    }
                }
            }
        }
        Some(total)
    }

    /// Validate the additive metadata attached to one node SST and return its
    /// live physical row count. Empty `per_label_counts` is the legacy marker:
    /// it can still answer the global count, but never a label count.
    fn validated_node_descriptor_live_count(&self, desc: &SstDescriptor) -> Option<u64> {
        let tombstones = match &desc.kind_specific {
            crate::manifest::KindSpecificStats::Nodes { tombstone_count } => *tombstone_count,
            _ => return None,
        };
        let live = desc.row_count.checked_sub(tombstones)?;
        let Some(index) = &desc.label_index else {
            return Some(live);
        };
        if index.per_label_counts.is_empty() {
            return Some(live);
        }
        if index.label_count != u64::try_from(index.per_label_counts.len()).ok()? {
            return None;
        }

        let mut previous_label = None;
        let mut posting_count = 0_u64;
        for &(label_id, count) in &index.per_label_counts {
            if previous_label.is_some_and(|previous| previous >= label_id)
                || self
                    .manifest
                    .manifest
                    .label_dict
                    .name(LabelId::new(label_id))
                    .is_none()
                || count > live
            {
                return None;
            }
            previous_label = Some(label_id);
            posting_count = posting_count.checked_add(count)?;
        }
        if posting_count != index.posting_count {
            return None;
        }
        Some(live)
    }

    /// Count live nodes exactly, using manifest metadata in the compacted /
    /// disjoint case and the ordinary last-write-wins reconciliation
    /// otherwise. A writer-attached snapshot caches the reconciled total and
    /// every per-label count under its logical generation; durable commits
    /// carry that tiny cache forward with exact label deltas. The fallback
    /// projects no properties, so even an overlapping write-heavy store avoids
    /// overflow/property materialisation.
    pub async fn count_nodes(&self, label: Option<&str>) -> Result<u64> {
        if let Some(count) = self
            .exact_node_counts
            .as_ref()
            .and_then(|counts| counts.count(label))
        {
            namidb_core::profile::record("Snapshot::count_nodes.cache", 0);
            return Ok(count);
        }
        let cache_generation = self
            .property_index_cache
            .as_ref()
            .zip(self.property_index_generation);
        if let Some((cache, generation)) = cache_generation {
            if let Some(count) = cache.node_count_at(label, generation) {
                namidb_core::profile::record("Snapshot::count_nodes.cache", 0);
                return Ok(count);
            }
        }

        if let Some(count) = self.metadata_node_count(label) {
            namidb_core::profile::record("Snapshot::count_nodes.metadata", 0);
            // Seed the complete count vector while metadata is authoritative,
            // so the first later write can apply a delta instead of forcing a
            // full-corpus reconciliation.
            if let Some((cache, generation)) = cache_generation {
                if let Some(total) = self.metadata_node_count(None) {
                    let mut by_label = HashMap::new();
                    let mut complete = true;
                    for observed in self.observed_labels() {
                        match self.metadata_node_count(Some(&observed)) {
                            Some(label_count) => {
                                if label_count > 0 {
                                    by_label.insert(observed, label_count);
                                }
                            }
                            None => {
                                complete = false;
                                break;
                            }
                        }
                    }
                    if complete {
                        if let Some(counts) = &self.exact_node_counts {
                            counts.install(total, by_label);
                        } else {
                            cache.insert_node_counts_at(generation, total, by_label);
                        }
                    }
                }
            }
            return Ok(count);
        }
        namidb_core::profile_scope!("Snapshot::count_nodes.reconcile");
        if let Some((cache, generation)) = cache_generation {
            cache.record_node_count_reconciliation_scan();
            let rows = self
                .scan_all_nodes_with_predicates_and_projection(&[], Some(&[]))
                .await?;
            let mut by_label: HashMap<String, u64> = HashMap::new();
            for row in &rows {
                for row_label in &row.labels {
                    *by_label.entry(row_label.clone()).or_insert(0) += 1;
                }
            }
            let total = rows.len() as u64;
            let requested = label
                .and_then(|label| by_label.get(label).copied())
                .unwrap_or_else(|| if label.is_some() { 0 } else { total });
            if let Some(counts) = &self.exact_node_counts {
                counts.install(total, by_label);
            } else {
                cache.insert_node_counts_at(generation, total, by_label);
            }
            return Ok(requested);
        }

        let rows = match label {
            Some(label) => {
                self.scan_label_with_predicates_and_projection(label, &[], Some(&[]))
                    .await?
            }
            None => {
                self.scan_all_nodes_with_predicates_and_projection(&[], Some(&[]))
                    .await?
            }
        };
        Ok(rows.len() as u64)
    }

    /// Resolve one label through current composite `(LabelId, NodeId)`
    /// sidecars when the node SST ranges prove that every id has exactly one
    /// persisted version.
    ///
    /// `None` means the accelerator cannot prove completeness and the caller
    /// must retain its Parquet scan. `Some` is exact: every sidecar page is
    /// bound to the manifest counts, every candidate is confirmed through the
    /// ordinary batch point reader, and the total number of candidates must
    /// equal the manifest's per-label count. A corrupt/missing optional
    /// sidecar therefore never turns into a short result.
    ///
    /// Keeping this fast path on disjoint generations has two useful
    /// properties. First, descriptors sorted by disjoint id range plus sorted
    /// sidecar leaves already produce global NodeId order, so `LIMIT` can stop
    /// without a corpus-sized merge heap. Second, a sidecar entry that fails
    /// point confirmation is necessarily inconsistent rather than merely an
    /// older LSM version, and can safely select the authoritative fallback.
    async fn try_scan_disjoint_label_sidecars(
        &self,
        descriptors: &[usize],
        label: &str,
        predicates: &[ScanPredicate],
        projection: Option<&[String]>,
        limit: Option<usize>,
    ) -> Result<Option<Vec<NodeView>>> {
        const CANDIDATE_BATCH: usize = 512;

        // Exact-record lookups materialise the complete encoded node payload.
        // When projected property pages exist that would pull unrelated wide
        // values (notably embeddings) merely to trim them afterwards. Let the
        // structural-Parquet + property-page path below own projected scans;
        // mixed/legacy generations still fall back per SST.
        if projection.is_some()
            && descriptors.iter().any(|idx| {
                crate::manifest::node_property_pages_sidecar(&self.manifest.manifest.ssts[*idx])
                    .is_some()
            })
        {
            return Ok(None);
        }

        let Some(label_id) = self.manifest.manifest.label_dict.id(label) else {
            return Ok(Some(Vec::new()));
        };
        let requested_projection: Option<BTreeSet<&str>> =
            projection.map(|columns| columns.iter().map(String::as_str).collect());
        let result_limit = limit.unwrap_or(usize::MAX);
        let mut out = Vec::with_capacity(result_limit.min(64));

        for &idx in descriptors {
            if !node_sst_can_contain_label(&self.manifest.manifest, idx, label) {
                continue;
            }
            let desc = &self.manifest.manifest.ssts[idx];
            let Some(index) = &desc.label_index else {
                return Ok(None);
            };
            if index.format != PropertyIndexFormat::PagedV1
                || index.per_label_counts.is_empty()
                || self.validated_node_descriptor_live_count(desc).is_none()
            {
                return Ok(None);
            }
            let Some(expected_candidates) = index
                .per_label_counts
                .iter()
                .find(|(candidate, _)| *candidate == label_id.get())
                .map(|(_, count)| *count)
            else {
                // `node_sst_can_contain_label` admitted this descriptor, so a
                // missing positive count contradicts its completeness proof.
                return Ok(None);
            };
            if expected_candidates == 0 {
                continue;
            }

            let absolute = format!("{}/{}", self.paths.namespace_prefix().as_ref(), index.path);
            let source = match self
                .pinned_sidecar_source(&absolute, Some(index.size_bytes))
                .await
            {
                Ok(source) => source,
                Err(error) if optional_accelerator_fallback(&error) => return Ok(None),
                Err(error) => return Err(error),
            };

            let mut cursor = None;
            let mut observed_candidates = 0_u64;
            loop {
                let page_limit = CANDIDATE_BATCH;
                let (page, stats) = match crate::sst::paged_index::page_label_ids_from_source(
                    &source,
                    label_id.get(),
                    cursor,
                    page_limit,
                    *desc.id.as_bytes(),
                    &index.per_label_counts,
                )
                .await
                {
                    Ok(page) => page,
                    Err(error) if optional_accelerator_fallback(&error) => return Ok(None),
                    Err(error) => return Err(error),
                };
                if stats.index_entries != index.posting_count {
                    return Ok(None);
                }
                observed_candidates = observed_candidates
                    .checked_add(page.ids.len() as u64)
                    .ok_or_else(|| Error::invariant("label candidate count overflows"))?;
                if observed_candidates > expected_candidates
                    || (page.ids.is_empty() && page.next_after.is_some())
                {
                    return Ok(None);
                }

                let ids = page
                    .ids
                    .iter()
                    .map(|id| NodeId::from_uuid(Uuid::from_bytes(*id)))
                    .collect::<Vec<_>>();
                let resolved = self.batch_lookup_nodes(label, &ids).await?;
                if resolved.len() != ids.len() {
                    return Ok(None);
                }
                for view in resolved {
                    let Some(mut view) = view else {
                        return Ok(None);
                    };
                    if !view.labels.contains(label) {
                        return Ok(None);
                    }
                    if !node_view_matches_predicates(&view, predicates) {
                        continue;
                    }
                    if let Some(requested) = &requested_projection {
                        view.properties
                            .retain(|property, _| requested.contains(property.as_str()));
                    }
                    out.push(view);
                    if out.len() == result_limit {
                        return Ok(Some(out));
                    }
                }

                let Some(next) = page.next_after else {
                    break;
                };
                if cursor.is_some_and(|previous| previous >= next) {
                    return Ok(None);
                }
                cursor = Some(next);
            }
            if observed_candidates != expected_candidates {
                return Ok(None);
            }
        }
        Ok(Some(out))
    }

    /// Shared exact implementation for the two public limited node scans.
    async fn scan_nodes_with_optional_label_limited(
        &self,
        label: Option<&str>,
        predicates: &[ScanPredicate],
        projection: Option<&[String]>,
        limit: usize,
    ) -> Result<Vec<NodeView>> {
        if limit == 0 {
            return Ok(Vec::new());
        }

        // Predicate columns are semantically required even when a direct
        // storage caller supplies a narrower output projection. The optimizer
        // already includes them, but widening here keeps this API exact on its
        // own; rows are trimmed back to the caller's projection after filtering.
        let decode_projection: Option<Vec<String>> = projection.map(|requested| {
            let mut columns: BTreeSet<String> = requested.iter().cloned().collect();
            columns.extend(
                predicates
                    .iter()
                    .map(|predicate| predicate.column().to_string()),
            );
            columns.into_iter().collect()
        });
        let decode_projection = decode_projection.as_deref();
        let requested_projection: Option<BTreeSet<&str>> =
            projection.map(|columns| columns.iter().map(String::as_str).collect());

        let Some(descriptors) = self.disjoint_node_descriptors() else {
            if let Some(cache) = &self.cache {
                cache.record_limited_node_scan_fallback();
            }
            let mut rows = self
                .scan_nodes_with_optional_label(label, predicates, decode_projection)
                .await?;
            if let Some(requested) = &requested_projection {
                for row in &mut rows {
                    row.properties
                        .retain(|property, _| requested.contains(property.as_str()));
                }
            }
            rows.truncate(limit);
            return Ok(rows);
        };

        if let Some(label) = label {
            if let Some(rows) = self
                .try_scan_disjoint_label_sidecars(
                    &descriptors,
                    label,
                    predicates,
                    projection,
                    Some(limit),
                )
                .await?
            {
                if let Some(cache) = &self.cache {
                    cache.record_limited_node_scan_fast_path();
                }
                return Ok(rows);
            }
        }

        if let Some(cache) = &self.cache {
            cache.record_limited_node_scan_fast_path();
        }
        let dict = &self.manifest.manifest.label_dict;
        let mut out = Vec::with_capacity(limit.min(64));
        let mut decoded_rows = 0usize;
        let mut examined_rows = 0usize;
        let mut row_groups = 0usize;
        let mut range_bytes = 0u64;

        for idx in descriptors {
            if label.is_some_and(|label| {
                !node_sst_can_contain_label(&self.manifest.manifest, idx, label)
            }) {
                continue;
            }
            let desc = &self.manifest.manifest.ssts[idx];
            let sst_label_def = self.label_def_for_node_sst(desc);
            let absolute = format!("{}/{}", self.paths.namespace_prefix().as_ref(), desc.path);
            let batch_context = LimitedNodeBatchContext {
                sst_label_def: &sst_label_def,
                desc,
                dict,
                label,
                predicates,
                decode_projection,
                requested_projection: requested_projection.as_ref(),
                limit,
            };

            if let Some(projected_properties) = decode_projection {
                if let Some(reader) = self.node_property_reader(desc).await? {
                    let cached_metadata = self
                        .cache
                        .as_ref()
                        .and_then(|cache| cache.get_metadata(&absolute));
                    let metadata_was_cached = cached_metadata.is_some();
                    let (mut stream, metadata) = node_scan_limited_async(
                        self.store.clone(),
                        Path::from(absolute.clone()),
                        desc.size_bytes,
                        &sst_label_def,
                        &[],
                        Some(&[]),
                        cached_metadata,
                    )
                    .await?;
                    if !metadata_was_cached {
                        if let Some(cache) = &self.cache {
                            cache.insert_metadata(absolute.clone(), metadata);
                        }
                    }
                    let io_stats = stream.stats().clone();
                    let out_before = out.len();
                    let decoded_before = decoded_rows;
                    let examined_before = examined_rows;
                    let mut next_ordinal = 0_u64;
                    let mut sidecar_failed = false;
                    'property_rows: while out.len() < limit {
                        let Some(batches) = stream.next_row_group().await? else {
                            break;
                        };
                        row_groups = row_groups.saturating_add(1);
                        for batch in batches {
                            let batch = batch.map_err(|error| {
                                Error::invariant(format!("projected Parquet limited read: {error}"))
                            })?;
                            decoded_rows = decoded_rows.saturating_add(batch.num_rows());
                            let Some(candidates) = self
                                .project_node_property_batch(
                                    reader.as_ref(),
                                    desc,
                                    projected_properties,
                                    &batch,
                                    next_ordinal,
                                )
                                .await?
                            else {
                                sidecar_failed = true;
                                break 'property_rows;
                            };
                            next_ordinal = next_ordinal
                                .checked_add(batch.num_rows() as u64)
                                .ok_or_else(|| {
                                    Error::invariant("node property ordinal exceeds u64")
                                })?;
                            for (_, _, view) in candidates {
                                if out.len() == limit {
                                    break;
                                }
                                examined_rows = examined_rows.saturating_add(1);
                                let Some(mut view) = view else {
                                    continue;
                                };
                                if label.is_some_and(|label| !view.labels.contains(label))
                                    || !node_view_matches_predicates(&view, predicates)
                                {
                                    continue;
                                }
                                if let Some(requested) = &requested_projection {
                                    view.properties.retain(|property, _| {
                                        requested.contains(property.as_str())
                                    });
                                }
                                out.push(view);
                            }
                        }
                    }
                    drop(stream);
                    range_bytes = range_bytes.saturating_add(io_stats.bytes_read());
                    if !sidecar_failed {
                        if out.len() == limit {
                            break;
                        }
                        if next_ordinal != desc.row_count {
                            return Err(Error::invariant(
                                "node property/Parquet row-count mismatch",
                            ));
                        }
                        continue;
                    }
                    // No rows escaped this internal buffer. Rewind only this SST's
                    // contribution and restart through authoritative Parquet.
                    out.truncate(out_before);
                    decoded_rows = decoded_before;
                    examined_rows = examined_before;
                }
            }

            // A cached immutable body is already resident, so consume its
            // synchronous Arrow iterator lazily. Force(false) remains the
            // explicit diagnostic escape hatch for the legacy full-body GET.
            let cached_body = self.cache_get(&absolute);
            if cached_body.is_some() || matches!(self.ranged_mode, RangedMode::Force(false)) {
                let body = match cached_body {
                    Some(body) => body,
                    None => self.get_sst_body(desc).await?,
                };
                let reader = NodeSstReader::open(sst_label_def.clone(), body)?;
                let batches = reader
                    .scan_iter_with_predicates_and_projection(predicates, decode_projection)?;
                let work = consume_limited_node_batches(batches, &batch_context, &mut out)?;
                decoded_rows = decoded_rows.saturating_add(work.decoded_rows);
                examined_rows = examined_rows.saturating_add(work.examined_rows);
            } else {
                // A prefix scan deliberately prefers ranged I/O in Auto mode,
                // even below the ordinary point-read size threshold: fetching
                // the whole body would erase LIMIT's byte savings.
                let cached_metadata = self
                    .cache
                    .as_ref()
                    .and_then(|cache| cache.get_metadata(&absolute));
                let metadata_was_cached = cached_metadata.is_some();
                let (mut stream, metadata) = node_scan_limited_async(
                    self.store.clone(),
                    Path::from(absolute.clone()),
                    desc.size_bytes,
                    &sst_label_def,
                    predicates,
                    decode_projection,
                    cached_metadata,
                )
                .await?;
                if !metadata_was_cached {
                    if let Some(cache) = &self.cache {
                        cache.insert_metadata(absolute.clone(), metadata);
                    }
                }
                let io_stats = stream.stats().clone();
                while out.len() < limit {
                    let Some(batches) = stream.next_row_group().await? else {
                        break;
                    };
                    row_groups = row_groups.saturating_add(1);
                    let work = consume_limited_node_batches(batches, &batch_context, &mut out)?;
                    decoded_rows = decoded_rows.saturating_add(work.decoded_rows);
                    examined_rows = examined_rows.saturating_add(work.examined_rows);
                }
                drop(stream);
                range_bytes = range_bytes.saturating_add(io_stats.bytes_read());
            }

            if out.len() == limit {
                break;
            }
        }

        if let Some(cache) = &self.cache {
            cache.record_limited_node_scan_work(
                decoded_rows,
                examined_rows,
                out.len(),
                row_groups,
                range_bytes,
            );
        }
        Ok(out)
    }

    /// Shared implementation for typed and typeless scans. Keeping the label
    /// filter optional here ensures both routes perform exactly one
    /// memtable+SST reconciliation and cannot drift in predicate/projection
    /// semantics.
    async fn scan_nodes_with_optional_label(
        &self,
        label: Option<&str>,
        predicates: &[ScanPredicate],
        projection: Option<&[String]>,
    ) -> Result<Vec<NodeView>> {
        let dict = &self.manifest.manifest.label_dict;
        // Row-group predicate pruning happens before the LSM winner is known.
        // It is therefore sound only when the manifest proves that every
        // persisted id has exactly one physical version and there are no node
        // memtable rows. With overlap, pruning a newer non-matching version
        // would let an older matching version for the same id resurface.
        let can_prune_before_lww = self.disjoint_node_descriptors().is_some();

        // Predicate properties are needed until after LWW even when the caller
        // requests a narrower output projection. Decode their union and trim
        // the winning live rows back to the requested shape at the end.
        let decode_projection: Option<Vec<String>> = projection.map(|requested| {
            let mut columns: BTreeSet<String> = requested.iter().cloned().collect();
            columns.extend(
                predicates
                    .iter()
                    .map(|predicate| predicate.column().to_string()),
            );
            columns.into_iter().collect()
        });
        let decode_projection = decode_projection.as_deref();
        let requested_projection: Option<BTreeSet<&str>> =
            projection.map(|columns| columns.iter().map(String::as_str).collect());

        if let Some(label) = label {
            if let Some(descriptors) = self.disjoint_node_descriptors() {
                if let Some(rows) = self
                    .try_scan_disjoint_label_sidecars(
                        &descriptors,
                        label,
                        predicates,
                        projection,
                        None,
                    )
                    .await?
                {
                    return Ok(rows);
                }
            }
        }

        // (node_id) → (winning lsn, materialised view or tombstone marker).
        // Nodes are id-primary: materialise every node across the label-agnostic
        // memtable + node SSTs and keep only those whose decoded label set
        // contains `label` (filtered at the end).
        let mut latest: BTreeMap<NodeId, (u64, Option<NodeView>)> = BTreeMap::new();

        // 1. Memtable rows. Keep the complete winner candidate until every
        // source has been reconciled; predicate evaluation belongs after LWW.
        for (mk, entry) in self.node_entries() {
            let MemKey::Node { id } = mk else {
                continue;
            };
            let view = match &entry.op {
                MemOp::Tombstone => None,
                MemOp::Upsert(payload) => {
                    Some(node_view_from_payload(*id, entry.lsn, payload, dict, "")?)
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
            if let Some(projected_properties) = decode_projection {
                if let Some(candidates) = self
                    .try_read_projected_node_sst(desc, &sst_label_def, projected_properties)
                    .await?
                {
                    for (row_id, lsn, view) in candidates {
                        update_node_winner(&mut latest, row_id, lsn, view);
                    }
                    continue;
                }
            }
            let body = self.get_sst_body(desc).await?;
            let reader = NodeSstReader::open(sst_label_def.clone(), body)?;
            // Build the projection set once per SST (declared properties
            // ∩ requested). When `projection.is_none()` we iterate every
            // declared property.
            let projection_set: Option<std::collections::BTreeSet<&str>> =
                decode_projection.map(|cols| cols.iter().map(|s| s.as_str()).collect());
            let reader_predicates = if can_prune_before_lww {
                predicates
            } else {
                &[]
            };

            for batch in
                reader.scan_with_predicates_and_projection(reader_predicates, decode_projection)?
            {
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
                    .and_then(|c| c.as_any().downcast_ref::<StringArray>());
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
                    // Skip the per-row overflow JSON parse entirely when the
                    // projection keeps nothing — an id-only scan was still
                    // paying a serde_json parse per row for values it threw
                    // away immediately.
                    if projection_set.as_ref().is_none_or(|keep| !keep.is_empty()) {
                        if let Some(ovf_col) = ovf_col.filter(|column| !column.is_null(row)) {
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
                    }
                    let view = NodeView {
                        id: row_id,
                        labels: decode_node_labels(&batch, row, dict, &desc.scope),
                        properties,
                        lsn,
                        schema_version: sv_col.value(row),
                    };
                    update_node_winner(&mut latest, row_id, lsn, Some(view));
                }
            }
        }

        // 3. Only the winning live version may be filtered. This ordering is
        // what prevents an older matching version from resurfacing behind a
        // newer non-matching upsert. Return in ascending-id order.
        let mut out = Vec::new();
        for (_, view) in latest.into_values() {
            let Some(mut view) = view else {
                continue;
            };
            if label.is_some_and(|label| !view.labels.contains(label))
                || !node_view_matches_predicates(&view, predicates)
            {
                continue;
            }
            if let Some(requested) = &requested_projection {
                view.properties
                    .retain(|property, _| requested.contains(property.as_str()));
            }
            out.push(view);
        }
        Ok(out)
    }

    /// Every live node id visible at this snapshot, in ONE label-agnostic
    /// pass over the memtable + every node SST, decoding only the
    /// id/tombstone/lsn columns (no property decode, no overflow JSON
    /// parse). This is the whole-graph node set for `CALL algo.*`: the
    /// per-label scan repeats the same full-store merge once per observed
    /// label, making an unfiltered algo-graph build `O(labels × nodes)`
    /// instead of `O(nodes)`. Includes nodes with an empty label set (the
    /// whole graph, GDS-style), which the per-label union would miss.
    pub async fn scan_all_node_ids(&self) -> Result<Vec<NodeId>> {
        // (node_id) → (winning lsn, live?). Highest LSN wins per id.
        let mut latest: BTreeMap<NodeId, (u64, bool)> = BTreeMap::new();
        let update =
            |latest: &mut BTreeMap<NodeId, (u64, bool)>, id: NodeId, lsn: u64, live: bool| {
                match latest.get(&id) {
                    Some((existing, _)) if *existing >= lsn => {}
                    _ => {
                        latest.insert(id, (lsn, live));
                    }
                }
            };

        for (mk, entry) in self.node_entries() {
            let MemKey::Node { id } = mk else {
                continue;
            };
            let live = !matches!(entry.op, MemOp::Tombstone);
            update(&mut latest, *id, entry.lsn, live);
        }

        for idx in self.manifest.index.node_descriptors() {
            let desc = &self.manifest.manifest.ssts[idx];
            let sst_label_def = self.label_def_for_node_sst(desc);
            let body = self.get_sst_body(desc).await?;
            let reader = NodeSstReader::open(sst_label_def, body)?;
            for batch in reader.scan_with_predicates_and_projection(&[], Some(&[]))? {
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
                for row in 0..batch.num_rows() {
                    let id_bytes: [u8; 16] = id_col
                        .value(row)
                        .try_into()
                        .map_err(|_| Error::invariant("node_id row length != 16"))?;
                    let id = NodeId::from_uuid(Uuid::from_bytes(id_bytes));
                    update(&mut latest, id, lsn_col.value(row), !tomb_col.value(row));
                }
            }
        }

        Ok(latest
            .into_iter()
            .filter_map(|(id, (_, live))| live.then_some(id))
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
        for (mk, entry) in self.edge_mem_entries_for_type(edge_type) {
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
        // Steady-state fast path (the 25 TB shape): one compacted forward
        // SST per type. The manifest already carries exact edge and
        // tombstone counts, and every memtable entry is newer than any
        // flushed row (the LSM flush cut), so the count is metadata plus an
        // O(memtable) point-lookup delta — never an O(edges) resident merge.
        let fwd: Vec<usize> = self
            .manifest
            .index
            .scope_descriptors(SstKind::EdgesFwd, edge_type)
            .to_vec();
        if fwd.len() <= 1 {
            let mut winners: BTreeMap<(NodeId, NodeId), (u64, bool)> = BTreeMap::new();
            for (mk, entry) in self.edge_mem_entries_for_type(edge_type) {
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
                update_edge_count_winner(&mut winners, (*src, *dst), entry.lsn, live);
            }
            let Some(&idx) = fwd.first() else {
                return Ok(winners.into_values().filter(|(_, live)| *live).count() as u64);
            };
            let desc = &self.manifest.manifest.ssts[idx];
            if let crate::manifest::KindSpecificStats::Edges {
                tombstone_count, ..
            } = &desc.kind_specific
            {
                let base_live = desc.row_count.saturating_sub(*tombstone_count);
                let absolute = format!("{}/{}", self.paths.namespace_prefix().as_ref(), desc.path);
                let mut delta: i64 = 0;
                if !winners.is_empty() {
                    let paged = self.fetch_paged_edge_reader(&absolute).await?;
                    for ((src, dst), (_, live)) in winners {
                        let sst_live = paged
                            .lookup_partner(src.as_bytes(), dst.as_bytes())
                            .await?
                            .map(|row| !row.tombstone)
                            .unwrap_or(false);
                        match (live, sst_live) {
                            (true, false) => delta += 1,
                            (false, true) => delta -= 1,
                            _ => {}
                        }
                    }
                }
                let total = base_live as i64 + delta;
                return Ok(u64::try_from(total.max(0)).unwrap_or(0));
            }
        }

        // (src, dst) -> (winning_lsn, is_live). Mirrors scan_edge_type's
        // merge exactly, minus the EdgeView materialisation.
        let mut latest: BTreeMap<(NodeId, NodeId), (u64, bool)> = BTreeMap::new();

        // 1. Memtable, then the writer's staged overlay (RFC-026 edge RYOW).
        for (mk, entry) in self.edge_mem_entries_for_type(edge_type) {
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
        self.sorted_partners_inner(edge_type, key, direction, false)
            .await
    }

    /// Identity-only partner lookup that always uses the source-keyed SST
    /// range/bloom path, even when the process-wide CSR cache is enabled.
    ///
    /// This is the sparse mutation primitive: a keyed `DELETE r` needs only
    /// `(edge_type, src, dst)`. Rebuilding a whole-type CSR after every
    /// manifest-changing delete batch would turn that operation into
    /// O(total edges) per batch; the SST route stays proportional to the
    /// candidate SSTs and the selected node's degree.
    pub async fn sorted_partners_via_sst(
        &self,
        edge_type: &str,
        key: NodeId,
        direction: EdgeDirection,
    ) -> Result<Vec<NodeId>> {
        self.sorted_partners_inner(edge_type, key, direction, true)
            .await
    }

    async fn sorted_partners_inner(
        &self,
        edge_type: &str,
        key: NodeId,
        direction: EdgeDirection,
        force_sst: bool,
    ) -> Result<Vec<NodeId>> {
        namidb_core::profile_scope!("Snapshot::sorted_partners");
        let key_bytes = *key.as_bytes();
        // Partner bytes -> (lsn, is_upsert).
        let mut latest: BTreeMap<[u8; 16], (u64, bool)> = BTreeMap::new();

        // Committed memtable then the staged overlay (RFC-026 edge RYOW)
        // first; the SST/CSR path below shadows whatever they contributed
        // only when its LSN is strictly higher.
        for (mk, entry) in self.edge_mem_entries_for_key(edge_type, key, direction) {
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
        if !force_sst && adjacency_enabled() {
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
        let cache_key = AdjacencyKey::new(
            self.cache_namespace.clone(),
            manifest_version,
            edge_type,
            direction,
        );
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
            let reader = self.fetch_paged_edge_reader(&absolute).await?;
            let Some(lookup) = reader.lookup(&key_bytes).await? else {
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
        for (mk, entry) in self.edge_mem_entries_for_key(edge_type, key, direction) {
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
            let paged_reader = self.fetch_paged_edge_reader(&absolute).await?;
            let Some(lookup) = paged_reader.lookup(&key_bytes).await? else {
                continue;
            };
            // Hydrate ONLY this key's property row range through the paged
            // property pages. The former whole-body `EdgeSstReader` +
            // stream-decode pair cost O(edge-SST bytes) per cold lookup —
            // disqualifying at large edge SSTs — while the row range is
            // O(deg). Legacy Arrow-stream sections keep exact behaviour:
            // `read_property_rows` falls back to the eager body for them.
            let row_start = u64::try_from(lookup.edge_offset)
                .map_err(|_| Error::invariant("edge lookup offset does not fit u64"))?;
            let row_range = row_start
                ..row_start
                    .checked_add(lookup.partners.len() as u64)
                    .ok_or_else(|| Error::invariant("edge lookup row range overflows u64"))?;
            let overflow = paged_reader
                .read_property_rows(OVERFLOW_JSON_NAME, row_range.clone())
                .await?;
            let declared_property_names: Vec<String> = self
                .manifest
                .manifest
                .schema
                .edge_type(edge_type)
                .map(|def| def.properties.iter().map(|p| p.name.clone()).collect())
                .unwrap_or_default();
            let mut declared: Vec<(String, Vec<Option<String>>)> =
                Vec::with_capacity(declared_property_names.len());
            for name in declared_property_names {
                if let Some(values) = paged_reader
                    .read_property_rows(&name, row_range.clone())
                    .await?
                {
                    declared.push((name, values));
                }
            }
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
                    let properties = decode_edge_properties(
                        overflow.as_ref().and_then(|values| values.get(i)),
                        &declared,
                        i,
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
        let cache_key = AdjacencyKey::new(
            self.cache_namespace.clone(),
            manifest_version,
            edge_type,
            direction,
        );
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
        for (mk, entry) in self.edge_mem_entries_for_key(edge_type, key, direction) {
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

    /// Fetch an immutable optional accelerator body.
    ///
    /// The node/edge SSTs remain authoritative. A rollback janitor may remove
    /// a newer `.vg`, `.ft`, or sidecar object while an older manifest still
    /// advertises it; structurally corrupt optional bodies are equivalent.
    /// Only those local availability/integrity failures select the exact
    /// fallback. Authentication, network, cancellation, and capacity failures
    /// remain visible to the caller.
    #[cfg(any(feature = "vector-index", feature = "text-index"))]
    async fn get_optional_sst_body(
        &self,
        desc: &SstDescriptor,
        accelerator: &'static str,
    ) -> Result<Option<Bytes>> {
        match self.get_sst_body(desc).await {
            Ok(body) => Ok(Some(body)),
            Err(error) if optional_accelerator_fallback(&error) => {
                tracing::warn!(
                    path = %desc.path,
                    error = %error,
                    accelerator,
                    "optional accelerator unavailable; falling back to exact scan"
                );
                Ok(None)
            }
            Err(error) => Err(error),
        }
    }

    /// Check whether an optional object really exists after decoded-cache
    /// admission rejected it. This HEAD is intentionally confined to the rare
    /// rejection path: a swept object must select the exact fallback, while an
    /// existing valid body that is too large must keep surfacing
    /// [`Error::CacheCapacity`] instead of hiding a sizing error behind an
    /// O(corpus) scan.
    #[cfg(any(feature = "vector-index", feature = "text-index"))]
    async fn optional_accelerator_exists(&self, absolute: &str) -> Result<bool> {
        crate::cancel::check()?;
        match self.store.head(&Path::from(absolute)).await {
            Ok(_) => Ok(true),
            Err(error) => {
                let error = Error::ObjectStore(error);
                if optional_accelerator_fallback(&error) {
                    Ok(false)
                } else {
                    Err(error)
                }
            }
        }
    }

    /// RFC-030 (`vector-index`): approximate top-k over the single authoritative
    /// `VectorGraph` SST registered for `index_name`. Returns
    /// `(NodeId, similarity)` best-first (higher similarity = closer). A
    /// rebuild atomically replaces the previous body; zero or multiple bodies
    /// are treated as an unusable generation and the query layer flat-scans.
    /// `ef` is the search beam width (≥ `k`).
    /// Decoded `.vg` index for `desc`, via the process-wide [`SstCache`]:
    /// decoding deserialises every stored vector plus the whole adjacency and
    /// clones the vectors into the navigation space, so paying it once per SST
    /// (instead of per query, and per widening round) is the difference between
    /// `O(k)`-ish and `O(index size)` KNN latency. `Ok(None)` = undecodable
    /// (legacy/corrupt) body — the caller skips it and the flat scan covers. A
    /// valid body that cannot fit the configured cache returns
    /// [`Error::CacheCapacity`] instead: an O(corpus) fallback would hide an
    /// operator sizing error.
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
            let (point_count, dim) = match &desc.kind_specific {
                KindSpecificStats::VectorGraph {
                    point_count, dim, ..
                } => (
                    usize::try_from(*point_count).unwrap_or(usize::MAX),
                    *dim as usize,
                ),
                // A kind/stats mismatch is an unusable optional artifact, not
                // a capacity decision. Preserve the corrupt-index fallback.
                _ => return Ok(None),
            };
            let wire_bytes = usize::try_from(desc.size_bytes).unwrap_or(usize::MAX);
            if let Err(rejection) =
                cache.admit_vector_index_wire_bytes(&absolute, wire_bytes, point_count, dim)
            {
                if !self.optional_accelerator_exists(&absolute).await? {
                    tracing::warn!(
                        path = %desc.path,
                        "vector index disappeared before cache admission; falling back to exact scan"
                    );
                    return Ok(None);
                }
                return Err(Error::CacheCapacity {
                    index_kind: "vector",
                    path: absolute,
                    required_bytes: rejection.required_bytes,
                    capacity_bytes: rejection.capacity_bytes,
                });
            }
        }
        let Some(body) = self.get_optional_sst_body(desc, "vector").await? else {
            return Ok(None);
        };
        let idx = match VectorGraphIndex::decode(&body) {
            Ok(idx) => idx,
            Err(error) => {
                tracing::warn!(
                    path = %desc.path,
                    error = %error,
                    "vector index is corrupt or legacy; falling back to exact scan"
                );
                return Ok(None);
            }
        };
        let idx = Arc::new(idx);
        if let Some(cache) = self.cache.as_ref() {
            if let Err(rejection) = cache.try_insert_vector_index_with_wire_bytes(
                absolute.clone(),
                idx.clone(),
                body.len(),
            ) {
                return Err(Error::CacheCapacity {
                    index_kind: "vector",
                    path: absolute,
                    required_bytes: rejection.required_bytes,
                    capacity_bytes: rejection.capacity_bytes,
                });
            }
        }
        Ok(Some(idx))
    }

    /// Select NAMIVG05 by its fixed prefix and keep only its centroid footer
    /// resident. NAMIVG03/04 retain the bounded monolithic compatibility path.
    #[cfg(feature = "vector-index")]
    async fn fetch_vector_search_index(
        &self,
        desc: &crate::manifest::SstDescriptor,
    ) -> Result<Option<VectorSearchIndex>> {
        use crate::sst::vector::v5::{VectorV5RangeSource, VectorV5Reader, MAGIC_V5};

        let absolute = format!("{}/{}", self.paths.namespace_prefix().as_ref(), desc.path);
        if let Some(cache) = self.cache.as_ref() {
            if let Some(reader) = cache.get_vector_v5_reader(&absolute) {
                return Ok(Some(VectorSearchIndex::Ranged(reader)));
            }
            if let Some(index) = cache.get_vector_index(&absolute) {
                return Ok(Some(VectorSearchIndex::Legacy(index)));
            }
        }
        if !matches!(&desc.kind_specific, KindSpecificStats::VectorGraph { .. }) {
            return Ok(None);
        }

        crate::cancel::check()?;
        let path = Path::from(absolute.as_str());
        let meta = match self.store.head(&path).await {
            Ok(meta) => meta,
            Err(error) => {
                let error = Error::ObjectStore(error);
                if optional_accelerator_fallback(&error) {
                    tracing::warn!(
                        path = %desc.path,
                        error = %error,
                        "range-readable vector index disappeared; falling back to exact scan"
                    );
                    return Ok(None);
                }
                return Err(error);
            }
        };
        if meta.size != desc.size_bytes {
            tracing::warn!(
                path = %desc.path,
                manifest_size = desc.size_bytes,
                object_size = meta.size,
                "vector object size disagrees with its manifest; falling back to exact scan"
            );
            return Ok(None);
        }
        let source = Arc::new(SearchObjectRangeSource::new(self.store.clone(), meta).await?);
        let magic = match VectorV5RangeSource::read_range(source.as_ref(), 0..MAGIC_V5.len() as u64)
            .await
        {
            Ok(magic) => magic,
            Err(error) if optional_accelerator_fallback(&error) => {
                tracing::warn!(
                    path = %desc.path,
                    error = %error,
                    "vector header is unavailable; falling back to exact scan"
                );
                return Ok(None);
            }
            Err(error) => return Err(error),
        };
        if magic.as_ref() == MAGIC_V5 {
            let reader = match VectorV5Reader::open(source.clone(), source.file_len()).await {
                Ok(reader) => Arc::new(reader),
                Err(error) if optional_accelerator_fallback(&error) => {
                    tracing::warn!(
                        path = %desc.path,
                        error = %error,
                        "range-readable vector index is corrupt; falling back to exact scan"
                    );
                    return Ok(None);
                }
                Err(error) => return Err(error),
            };
            if let Some(cache) = self.cache.as_ref() {
                cache.insert_vector_v5_reader(absolute, reader.clone());
            }
            return Ok(Some(VectorSearchIndex::Ranged(reader)));
        }
        if crate::sst::vector::is_supported_monolithic_magic(&magic) {
            return Ok(self
                .fetch_vector_index(desc)
                .await?
                .map(VectorSearchIndex::Legacy));
        }
        tracing::warn!(
            path = %desc.path,
            magic = ?magic,
            "vector index has an unknown format; falling back to exact scan"
        );
        Ok(None)
    }

    /// Decoded `.ft` index for `desc`, via the process-wide [`SstCache`] (same
    /// once-per-SST story as [`Self::fetch_vector_index`]). Mirrors the `.vg`
    /// convention: a body that fails decode — a legacy magic after a format
    /// bump, or corruption — is `None`, so the caller treats the index as
    /// absent and the flat scan serves until the next authoritative
    /// compaction rebuilds the SST, rather than erroring every query. Capacity
    /// refusal is different and surfaces as [`Error::CacheCapacity`].
    #[cfg(feature = "text-index")]
    async fn fetch_text_index(
        &self,
        desc: &crate::manifest::SstDescriptor,
    ) -> Result<Option<Arc<crate::sst::text::TextIndex>>> {
        use crate::sst::text::TextIndex;
        let absolute = format!("{}/{}", self.paths.namespace_prefix().as_ref(), desc.path);
        if let Some(cache) = self.cache.as_ref() {
            if let Some(idx) = cache.get_text_index(&absolute) {
                return Ok(Some(idx));
            }
            let doc_count = match &desc.kind_specific {
                KindSpecificStats::TextIndex { doc_count, .. } => {
                    usize::try_from(*doc_count).unwrap_or(usize::MAX)
                }
                _ => return Ok(None),
            };
            let wire_bytes = usize::try_from(desc.size_bytes).unwrap_or(usize::MAX);
            if let Err(rejection) =
                cache.admit_text_index_wire_bytes(&absolute, wire_bytes, doc_count)
            {
                if !self.optional_accelerator_exists(&absolute).await? {
                    tracing::warn!(
                        path = %desc.path,
                        "full-text index disappeared before cache admission; falling back to exact scan"
                    );
                    return Ok(None);
                }
                return Err(Error::CacheCapacity {
                    index_kind: "full-text",
                    path: absolute,
                    required_bytes: rejection.required_bytes,
                    capacity_bytes: rejection.capacity_bytes,
                });
            }
        }
        let Some(body) = self.get_optional_sst_body(desc, "full-text").await? else {
            return Ok(None);
        };
        let idx = match TextIndex::decode(&body) {
            Ok(idx) => idx,
            Err(error) => {
                tracing::warn!(
                    path = %desc.path,
                    error = %error,
                    "full-text index is corrupt or legacy; falling back to exact scan"
                );
                return Ok(None);
            }
        };
        let idx = Arc::new(idx);
        if let Some(cache) = self.cache.as_ref() {
            if let Err(rejection) = cache.try_insert_text_index_with_wire_bytes(
                absolute.clone(),
                idx.clone(),
                body.len(),
            ) {
                return Err(Error::CacheCapacity {
                    index_kind: "full-text",
                    path: absolute,
                    required_bytes: rejection.required_bytes,
                    capacity_bytes: rejection.capacity_bytes,
                });
            }
        }
        Ok(Some(idx))
    }

    /// Open the newest range-readable full-text format without applying the
    /// monolithic decoded-index admission rule. Legacy NAMIFT02 objects retain
    /// the old full-body cache path; NAMIFT03 keeps only its sparse
    /// footer/directory resident and resolves postings/doc IDs through the
    /// shared RAM/NVMe object-page cache.
    #[cfg(feature = "text-index")]
    async fn fetch_text_search_index(
        &self,
        desc: &crate::manifest::SstDescriptor,
    ) -> Result<Option<TextSearchIndex>> {
        use crate::sst::text::{
            TextIndexRangeSource, TextIndexV3Reader, LEGACY_MONOLITHIC_MAGIC, RANGE_READABLE_MAGIC,
        };

        let absolute = format!("{}/{}", self.paths.namespace_prefix().as_ref(), desc.path);
        if let Some(cache) = self.cache.as_ref() {
            if let Some(reader) = cache.get_text_v3_reader(&absolute) {
                return Ok(Some(TextSearchIndex::Ranged(reader)));
            }
            if let Some(index) = cache.get_text_index(&absolute) {
                return Ok(Some(TextSearchIndex::Legacy(index)));
            }
        }
        if !matches!(&desc.kind_specific, KindSpecificStats::TextIndex { .. }) {
            return Ok(None);
        }

        crate::cancel::check()?;
        let path = Path::from(absolute.as_str());
        let meta = match self.store.head(&path).await {
            Ok(meta) => meta,
            Err(error) => {
                let error = Error::ObjectStore(error);
                if optional_accelerator_fallback(&error) {
                    tracing::warn!(
                        path = %desc.path,
                        error = %error,
                        "range-readable full-text index disappeared; falling back to exact scan"
                    );
                    return Ok(None);
                }
                return Err(error);
            }
        };
        if meta.size != desc.size_bytes {
            tracing::warn!(
                path = %desc.path,
                manifest_size = desc.size_bytes,
                object_size = meta.size,
                "full-text object size disagrees with its manifest; falling back to exact scan"
            );
            return Ok(None);
        }
        let source = Arc::new(SearchObjectRangeSource::new(self.store.clone(), meta).await?);
        let magic = match source
            .read_range(0..RANGE_READABLE_MAGIC.len() as u64)
            .await
        {
            Ok(magic) => magic,
            Err(error) if optional_accelerator_fallback(&error) => {
                tracing::warn!(
                    path = %desc.path,
                    error = %error,
                    "full-text header is unavailable; falling back to exact scan"
                );
                return Ok(None);
            }
            Err(error) => return Err(error),
        };

        if magic.as_ref() == RANGE_READABLE_MAGIC {
            let reader = match TextIndexV3Reader::open(source.clone(), source.file_len()).await {
                Ok(reader) => Arc::new(reader),
                Err(error) if optional_accelerator_fallback(&error) => {
                    tracing::warn!(
                        path = %desc.path,
                        error = %error,
                        "range-readable full-text index is corrupt; falling back to exact scan"
                    );
                    return Ok(None);
                }
                Err(error) => return Err(error),
            };
            if let Some(cache) = self.cache.as_ref() {
                cache.insert_text_v3_reader(absolute, reader.clone());
            }
            return Ok(Some(TextSearchIndex::Ranged(reader)));
        }
        if magic.as_ref() == LEGACY_MONOLITHIC_MAGIC {
            return Ok(self
                .fetch_text_index(desc)
                .await?
                .map(TextSearchIndex::Legacy));
        }
        tracing::warn!(
            path = %desc.path,
            magic = ?magic,
            "full-text index has an unknown format; falling back to exact scan"
        );
        Ok(None)
    }

    #[cfg(any(feature = "text-index", feature = "vector-index"))]
    async fn select_search_base(
        &self,
        kind: SearchLsmKind,
        index_name: &str,
    ) -> Result<Option<SelectedSearchBase>> {
        let manifest = &self.manifest.manifest;
        // Selection validates the Search-LSM invariants. Reuse that result:
        // asking `index_outrun_by_nodes` first used to run the complete
        // validator twice for every active query.
        let plan = select_search_read_plan(manifest, kind, index_name);
        if matches!(&plan, SearchReadPlan::Legacy { .. })
            && self.legacy_index_outrun_by_nodes(index_name, kind.sst_kind())
        {
            return Ok(None);
        }
        match plan {
            SearchReadPlan::Legacy { sst_id } => {
                let Some(descriptor_index) = manifest
                    .ssts
                    .iter()
                    .position(|descriptor| descriptor.id == sst_id)
                else {
                    return Ok(None);
                };
                Ok(Some(SelectedSearchBase {
                    descriptor_index,
                    active_segment: None,
                }))
            }
            SearchReadPlan::ActiveLegacyBase {
                state,
                base_sst_id,
                barrier_sst_id,
            } => {
                let Some(barrier) = manifest
                    .ssts
                    .iter()
                    .find(|descriptor| descriptor.id == barrier_sst_id)
                else {
                    return Ok(None);
                };
                let Some(body) = self
                    .get_optional_sst_body(barrier, "search-lsm-barrier")
                    .await?
                else {
                    return Ok(None);
                };
                if body.len() as u64 != barrier.size_bytes {
                    tracing::warn!(
                        path = %barrier.path,
                        manifest_size = barrier.size_bytes,
                        object_size = body.len(),
                        "search LSM barrier size disagrees with its descriptor; falling back to exact scan"
                    );
                    return Ok(None);
                }
                if let Err(error) = validate_search_barrier(&state, &body) {
                    tracing::warn!(
                        path = %barrier.path,
                        error = %error,
                        "search LSM barrier is corrupt or belongs to another generation; \
                         falling back to exact scan"
                    );
                    return Ok(None);
                }
                let Some(descriptor_index) = manifest
                    .ssts
                    .iter()
                    .position(|descriptor| descriptor.id == base_sst_id)
                else {
                    return Ok(None);
                };
                let Some(segment) = state
                    .segments
                    .iter()
                    .find(|segment| segment.sst_id == base_sst_id)
                    .cloned()
                else {
                    return Ok(None);
                };
                Ok(Some(SelectedSearchBase {
                    descriptor_index,
                    active_segment: Some(segment),
                }))
            }
            SearchReadPlan::ActiveSegments { state, .. } => {
                tracing::debug!(
                    index = index_name,
                    ?kind,
                    segments = state.segments.len(),
                    "native search segments require the multi-segment coordinator; using exact scan"
                );
                Ok(None)
            }
            SearchReadPlan::FlatFallback(reason) => {
                tracing::debug!(
                    index = index_name,
                    ?kind,
                    ?reason,
                    "search generation unavailable; using exact scan"
                );
                Ok(None)
            }
        }
    }

    /// Search the persisted vector-graph SSTs for `index_name`, preserving the
    /// distinction between an empty result and an unusable index.
    ///
    /// `Ok(None)` means no matching `.vg` exists or at least one matching body
    /// could not be decoded. Callers that merge fresh memtable deltas with the
    /// persisted result must retain this signal and fall back to the flat scan:
    /// treating an undecodable graph as an empty graph is only safe while the
    /// fresh delta itself cannot fill `k`.
    #[cfg(feature = "vector-index")]
    pub async fn try_vector_search(
        &self,
        index_name: &str,
        query: &[f32],
        k: usize,
        ef: usize,
    ) -> Result<Option<Vec<(NodeId, f32)>>> {
        Ok(self
            .try_vector_search_with_point_count(index_name, query, k, ef)
            .await?
            .map(|(hits, _point_count)| hits))
    }

    /// The same availability-preserving vector probe as
    /// [`Self::try_vector_search`], plus the decoded body's exact point count.
    ///
    /// The query executor uses `point_count` only as an exhaustiveness proof:
    /// when a request returns every persisted vector, a `k` larger than the
    /// corpus can return that exact short page without re-scanning all nodes.
    /// The count comes from the validated body rather than manifest statistics,
    /// so corrupt or mismatched optional metadata cannot manufacture a false
    /// completeness signal.
    #[cfg(feature = "vector-index")]
    pub async fn try_vector_search_with_point_count(
        &self,
        index_name: &str,
        query: &[f32],
        k: usize,
        ef: usize,
    ) -> Result<Option<(Vec<(NodeId, f32)>, u64)>> {
        let result = self
            .try_vector_search_with_point_count_inner(index_name, query, k, ef)
            .await;
        if let Ok(outcome) = &result {
            crate::route_telemetry::record_vector(outcome.is_some());
        }
        result
    }

    #[cfg(feature = "vector-index")]
    async fn try_vector_search_with_point_count_inner(
        &self,
        index_name: &str,
        query: &[f32],
        k: usize,
        ef: usize,
    ) -> Result<Option<(Vec<(NodeId, f32)>, u64)>> {
        match search_lsm_read::vector_search(self, index_name, query, k, ef, &[]).await? {
            search_lsm_read::ActiveSearch::Ready(
                search_lsm_read::CoordinatedVectorFilterResult::Applied(result),
            ) => return Ok(Some((result.hits, result.point_count))),
            search_lsm_read::ActiveSearch::Ready(
                search_lsm_read::CoordinatedVectorFilterResult::Unsupported,
            )
            | search_lsm_read::ActiveSearch::Unavailable => return Ok(None),
            search_lsm_read::ActiveSearch::NotActive => {}
        }
        let mut best_by_id: HashMap<NodeId, f32> = HashMap::new();
        let Some(selected) = self
            .select_search_base(SearchLsmKind::Vector, index_name)
            .await?
        else {
            return Ok(None);
        };
        let descriptor_ids = [selected.descriptor_index];
        // Score orientation is metric-dependent: cosine/dot are higher-is-closer,
        // euclidean is lower-is-closer. All `.vg` SSTs for one index share a
        // metric, so the last decoded one's orientation is authoritative.
        let mut higher_is_better = true;
        let mut point_count = 0;
        for &desc_idx in &descriptor_ids {
            let desc = &self.manifest.manifest.ssts[desc_idx];
            // A legacy (v1) or corrupt body makes the persisted answer
            // incomplete. Preserve "index unavailable" so the query layer can
            // fall back to the exact flat scan rather than accidentally serving
            // only a sufficiently-large fresh delta.
            let Some(idx) = self.fetch_vector_search_index(desc).await? else {
                return Ok(None);
            };
            if selected
                .active_segment
                .as_ref()
                .is_some_and(|segment| !active_vector_base_matches(&idx, desc, segment))
            {
                tracing::warn!(
                    path = %desc.path,
                    "active vector base footer disagrees with its Search-LSM binding; \
                     falling back to exact scan"
                );
                return Ok(None);
            }
            higher_is_better = idx.higher_is_better();
            point_count = idx.point_count();
            let hits = match idx.search(query, k, ef).await {
                Ok(hits) => hits,
                Err(error) if optional_accelerator_fallback(&error) => {
                    tracing::warn!(
                        path = %desc.path,
                        error = %error,
                        "vector query page is corrupt; falling back to exact scan"
                    );
                    return Ok(None);
                }
                Err(error) => return Err(error),
            };
            for (id, score) in hits {
                let id = NodeId(Uuid::from_bytes(id));
                best_by_id
                    .entry(id)
                    .and_modify(|best| {
                        let replace = if higher_is_better {
                            score > *best
                        } else {
                            score < *best
                        };
                        if replace {
                            *best = score;
                        }
                    })
                    .or_insert(score);
            }
        }
        let mut all: Vec<(NodeId, f32)> = best_by_id.into_iter().collect();
        // Best-first by the metric's orientation.
        if higher_is_better {
            all.sort_unstable_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        } else {
            all.sort_unstable_by(|a, b| a.1.total_cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
        }
        all.truncate(k);
        Ok(Some((all, point_count)))
    }

    /// Filter-aware vector-index probe.
    ///
    /// Unlike fetching an unfiltered top-k and discarding rows afterwards,
    /// `eligible` is passed into the decoded `.vg`: ineligible ordinals may
    /// still guide Vamana navigation, but are removed before metric reranking
    /// and before the index truncates to `k`. The returned point count lets the
    /// query layer distinguish an under-filled selective beam from a fully
    /// exhausted index while it widens `ef`.
    ///
    /// `Ok(None)` preserves the same unavailable/corrupt-generation contract as
    /// [`Self::try_vector_search`]. This method deliberately accepts only a
    /// complete, caller-proven eligibility set; deriving that set from indexed
    /// metadata (and falling back when no complete metadata index exists) lives
    /// one layer above.
    #[cfg(feature = "vector-index")]
    pub async fn try_vector_search_filtered(
        &self,
        index_name: &str,
        query: &[f32],
        k: usize,
        ef: usize,
        eligible: &BTreeSet<NodeId>,
    ) -> Result<Option<(Vec<(NodeId, f32)>, u64)>> {
        if matches!(
            select_search_read_plan(&self.manifest.manifest, SearchLsmKind::Vector, index_name),
            SearchReadPlan::ActiveSegments { .. }
        ) {
            // Arbitrary NodeId sets have no ordinal bitmap in V5/VG6. Native
            // property groups use the coordinator below; residual sets retain
            // the exact node-scan fallback.
            return Ok(None);
        }
        let Some(selected) = self
            .select_search_base(SearchLsmKind::Vector, index_name)
            .await?
        else {
            return Ok(None);
        };
        let desc = &self.manifest.manifest.ssts[selected.descriptor_index];
        let Some(idx) = self.fetch_vector_search_index(desc).await? else {
            return Ok(None);
        };
        if selected
            .active_segment
            .as_ref()
            .is_some_and(|segment| !active_vector_base_matches(&idx, desc, segment))
        {
            return Ok(None);
        }
        let point_count = idx.point_count();
        let VectorSearchIndex::Legacy(idx) = idx else {
            // NAMIVG05 applies its own persisted metadata bitmaps natively.
            // An arbitrary caller-supplied NodeId set has no on-page bitmap
            // yet, so preserve correctness through the exact fallback.
            return Ok(None);
        };
        let hits = idx
            .search_filtered(query, k, ef, |id| {
                eligible.contains(&NodeId(Uuid::from_bytes(*id)))
            })
            .into_iter()
            .map(|(id, score)| (NodeId(Uuid::from_bytes(id)), score))
            .collect();
        Ok(Some((hits, point_count)))
    }

    /// Apply complete String/Bool equality groups from the metadata postings
    /// embedded in a v4 `.vg`.
    ///
    /// The posting is evaluated as vector ordinals inside the decoded graph;
    /// no matching node corpus or NodeId set is hydrated. `Ok(None)` keeps the
    /// usual unavailable/corrupt-index meaning. `Unsupported` means the vector
    /// body is usable but none of these groups was materialised (legacy v3,
    /// unindexed property, or a property atomically omitted by a build cap).
    #[cfg(feature = "vector-index")]
    pub async fn try_vector_search_filter_groups(
        &self,
        index_name: &str,
        query: &[f32],
        k: usize,
        ef: usize,
        groups: &[(String, Vec<Value>)],
    ) -> Result<Option<VectorFilterSearch>> {
        match search_lsm_read::vector_search(self, index_name, query, k, ef, groups).await? {
            search_lsm_read::ActiveSearch::Ready(
                search_lsm_read::CoordinatedVectorFilterResult::Applied(result),
            ) => {
                return Ok(Some(VectorFilterSearch::Applied {
                    hits: result.hits,
                    point_count: result.point_count,
                    eligible_count: result.eligible_count,
                }));
            }
            search_lsm_read::ActiveSearch::Ready(
                search_lsm_read::CoordinatedVectorFilterResult::Unsupported,
            ) => return Ok(Some(VectorFilterSearch::Unsupported)),
            search_lsm_read::ActiveSearch::Unavailable => return Ok(None),
            search_lsm_read::ActiveSearch::NotActive => {}
        }
        let Some(selected) = self
            .select_search_base(SearchLsmKind::Vector, index_name)
            .await?
        else {
            return Ok(None);
        };
        let desc = &self.manifest.manifest.ssts[selected.descriptor_index];
        let Some(idx) = self.fetch_vector_search_index(desc).await? else {
            return Ok(None);
        };
        if selected
            .active_segment
            .as_ref()
            .is_some_and(|segment| !active_vector_base_matches(&idx, desc, segment))
        {
            return Ok(None);
        }
        let point_count = idx.point_count();
        let (hits, eligible_count) = match idx {
            VectorSearchIndex::Legacy(idx) => {
                let Some(result) = idx.search_filter_groups(query, k, ef, groups) else {
                    return Ok(Some(VectorFilterSearch::Unsupported));
                };
                result
            }
            VectorSearchIndex::Ranged(idx) => {
                let result = match idx
                    .search_filter_groups(query, k, vector_v5_search_options(&idx, ef), groups)
                    .await
                {
                    Ok(result) => result,
                    Err(error) if optional_accelerator_fallback(&error) => {
                        tracing::warn!(
                            path = %desc.path,
                            error = %error,
                            "filtered vector page is corrupt; falling back to exact scan"
                        );
                        return Ok(None);
                    }
                    Err(error) => return Err(error),
                };
                if result.applied_filter_groups != groups.len() {
                    return Ok(Some(VectorFilterSearch::Unsupported));
                }
                let eligible_count = if result.probed_pages == idx.page_count() {
                    result.eligible_rows_seen
                } else {
                    // The executor treats a finite eligible_count as an
                    // exhaustiveness proof. A partial IVF probe cannot make
                    // that claim, so use a sentinel until every page was read.
                    usize::MAX
                };
                (result.hits, eligible_count)
            }
        };
        let hits = hits
            .into_iter()
            .map(|(id, score)| (NodeId(Uuid::from_bytes(id)), score))
            .collect();
        Ok(Some(VectorFilterSearch::Applied {
            hits,
            point_count,
            eligible_count,
        }))
    }

    /// Low-level vector-index probe retained for storage callers and benchmark
    /// harnesses that only need the decoded hits. An absent/undecodable index is
    /// represented as an empty result; query execution must use
    /// [`Self::try_vector_search`] so it can trigger the exact fallback.
    #[cfg(feature = "vector-index")]
    pub async fn vector_search(
        &self,
        index_name: &str,
        query: &[f32],
        k: usize,
        ef: usize,
    ) -> Result<Vec<(NodeId, f32)>> {
        Ok(self
            .try_vector_search(index_name, query, k, ef)
            .await?
            .unwrap_or_default())
    }

    /// `true` if persisted `Nodes` SSTs carry writes that the `index_name`
    /// index SST(s) may not have absorbed.
    ///
    /// An SST-backed index (`.vg` / `.ft`) is rebuilt only on an **authoritative**
    /// compaction spanning the full label corpus. Its descriptor's `max_lsn`
    /// stores that corpus high-water mark; later flushes/partial merges carry a
    /// higher LSN. Comparing LSNs rather than levels catches both.
    ///
    /// The persisted gate is label-scoped without trusting live-label metadata
    /// alone: an SST with no current row of the indexed label may still contain
    /// a tombstone/relabel of an id that the old index serves. Every current
    /// index descriptor stores its exact member-NodeId range. A newer node SST
    /// can be ignored only when its label index proves it has no live row of the
    /// target label **and** its key range is disjoint from each index SST it
    /// outruns. Range and `max_lsn` stay paired per index SST, so a newer index
    /// cannot lend its high-water mark to an older range. Legacy index
    /// descriptors use 00..FF and remain conservatively global until rebuilt.
    ///
    /// `kind` is `VectorGraph` or `TextIndex`.
    #[cfg(any(feature = "vector-index", feature = "text-index"))]
    pub fn index_outrun_by_nodes(&self, index_name: &str, kind: SstKind) -> bool {
        let manifest = &self.manifest.manifest;
        let lsm_kind = match kind {
            SstKind::VectorGraph => SearchLsmKind::Vector,
            SstKind::TextIndex => SearchLsmKind::Text,
            _ => return true,
        };
        match select_search_read_plan(manifest, lsm_kind, index_name) {
            SearchReadPlan::ActiveLegacyBase { .. } | SearchReadPlan::ActiveSegments { .. } => {
                // Active coverage is exact per visible Nodes SST. Object/footer
                // validation still happens before serving; failure there
                // returns None and selects the exact fallback.
                false
            }
            SearchReadPlan::FlatFallback(crate::search_lsm::SearchReadFallback::NoPhysicalBody)
            | SearchReadPlan::Legacy { .. } => self.legacy_index_outrun_by_nodes(index_name, kind),
            SearchReadPlan::FlatFallback(_) => true,
        }
    }

    #[cfg(any(feature = "vector-index", feature = "text-index"))]
    fn legacy_index_outrun_by_nodes(&self, index_name: &str, kind: SstKind) -> bool {
        let manifest = &self.manifest.manifest;
        let index_ssts: Vec<&SstDescriptor> = self
            .manifest
            .index
            .scope_descriptors(kind, index_name)
            .iter()
            .map(|idx| &manifest.ssts[*idx])
            .collect();
        if index_ssts.is_empty() {
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
            return manifest.ssts.iter().any(|d| d.kind == SstKind::Nodes);
        }

        let index_label = match kind {
            SstKind::VectorGraph => manifest
                .vector_indexes
                .iter()
                .find(|d| d.name == index_name)
                .map(|d| d.label.as_str()),
            SstKind::TextIndex => manifest
                .text_indexes
                .iter()
                .find(|d| d.name == index_name)
                .map(|d| d.label.as_str()),
            _ => None,
        };
        let oldest_index_lsn = index_ssts.iter().map(|d| d.max_lsn).min().unwrap_or(0);
        // A missing/malformed registration gives us no safe label scope.
        let Some(index_label) = index_label else {
            return manifest
                .ssts
                .iter()
                .any(|d| d.kind == SstKind::Nodes && d.max_lsn > oldest_index_lsn);
        };

        for (node_idx, node_sst) in manifest.ssts.iter().enumerate() {
            if node_sst.kind != SstKind::Nodes || node_sst.max_lsn <= oldest_index_lsn {
                continue;
            }
            // Missing/legacy label-index metadata deliberately returns true:
            // without a proof of absence this SST may add/update a document.
            if node_sst_can_contain_label(manifest, node_idx, index_label) {
                return true;
            }
            // There is no live target-label row, but this SST may tombstone or
            // relabel a member. Compare each index SST against its own LSN and
            // range rather than combining an old range with a new high-water.
            if index_ssts.iter().any(|index_sst| {
                node_sst.max_lsn > index_sst.max_lsn && key_ranges_may_overlap(node_sst, index_sst)
            }) {
                return true;
            }
        }
        false
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
    /// `Ok(Some(hits))` is the BM25 result from the single authoritative index:
    /// `(NodeId, score)` best-first with a node-id tie-break. Multiple bodies
    /// indicate stale/anomalous generations and force the exact fallback
    /// because their per-corpus BM25 statistics are not comparable. `k = None`
    /// returns every match. `query` is the parsed query — phrases, prefixes and
    /// plain terms — whose semantics `TextIndex::search_query` shares verbatim
    /// with the executor's flat scan.
    #[cfg(feature = "text-index")]
    pub async fn text_search(
        &self,
        index_name: &str,
        label: &str,
        query: &crate::text::TextQuery,
        k: Option<usize>,
    ) -> Result<Option<Vec<(NodeId, f64)>>> {
        let result = self.text_search_inner(index_name, label, query, k).await;
        if let Ok(outcome) = &result {
            crate::route_telemetry::record_text(outcome.is_some());
        }
        result
    }

    #[cfg(feature = "text-index")]
    async fn text_search_inner(
        &self,
        index_name: &str,
        label: &str,
        query: &crate::text::TextQuery,
        k: Option<usize>,
    ) -> Result<Option<Vec<(NodeId, f64)>>> {
        if !self
            .manifest
            .manifest
            .text_indexes
            .iter()
            .any(|descriptor| descriptor.name == index_name && descriptor.label == label)
        {
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

        match search_lsm_read::text_search(self, index_name, query, k, &[], &dirty).await? {
            search_lsm_read::ActiveSearch::Ready(hits) => return Ok(Some(hits)),
            search_lsm_read::ActiveSearch::Unavailable => return Ok(None),
            search_lsm_read::ActiveSearch::NotActive => {}
        }
        let Some(selected) = self
            .select_search_base(SearchLsmKind::Text, index_name)
            .await?
        else {
            return Ok(None);
        };
        let mut best_by_id: HashMap<NodeId, f64> = HashMap::new();
        for desc_idx in [selected.descriptor_index] {
            let desc = &self.manifest.manifest.ssts[desc_idx];
            // An undecodable body (legacy magic after a format bump, or
            // corruption): BM25 depends on whole-corpus stats, so a partial
            // serve would skew them — treat the index as absent and flat-scan.
            let Some(idx) = self.fetch_text_search_index(desc).await? else {
                return Ok(None);
            };
            if selected
                .active_segment
                .as_ref()
                .is_some_and(|segment| !active_text_base_matches(&idx, desc, segment))
            {
                tracing::warn!(
                    path = %desc.path,
                    "active full-text base footer disagrees with its Search-LSM binding; \
                     falling back to exact scan"
                );
                return Ok(None);
            }
            // A dirty id that IS an indexed document means a stale doc (delete/
            // relabel) the index would still serve, and its removal also shifts
            // the corpus stats — only the flat scan is exact then.
            match idx.contains_any_doc(&dirty).await {
                Ok(true) => return Ok(None),
                Ok(false) => {}
                Err(error) if optional_accelerator_fallback(&error) => {
                    tracing::warn!(
                        path = %desc.path,
                        error = %error,
                        "full-text membership page is corrupt; falling back to exact scan"
                    );
                    return Ok(None);
                }
                Err(error) => return Err(error),
            }
            let hits = match idx.search_query(query, k).await {
                Ok(hits) => hits,
                Err(error) if optional_accelerator_fallback(&error) => {
                    tracing::warn!(
                        path = %desc.path,
                        error = %error,
                        "full-text query page is corrupt; falling back to exact scan"
                    );
                    return Ok(None);
                }
                Err(error) => return Err(error),
            };
            for (id, score) in hits {
                let id = NodeId(Uuid::from_bytes(id));
                best_by_id
                    .entry(id)
                    .and_modify(|best| {
                        if score > *best {
                            *best = score;
                        }
                    })
                    .or_insert(score);
            }
        }
        let mut all: Vec<(NodeId, f64)> = best_by_id.into_iter().collect();
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

    /// Exact Search-LSM BM25 with native equality prefilters.
    ///
    /// Filter groups use OR within one property and AND across properties.
    /// Every live FT4 segment must advertise and physically contain every
    /// requested property; otherwise `None` selects the ordinary exact
    /// node-scan fallback. Corpus BM25 statistics are reconstructed before
    /// filtering, so selectivity never changes `N`, average length, or IDF.
    #[cfg(feature = "text-index")]
    pub async fn text_search_filter_groups(
        &self,
        index_name: &str,
        label: &str,
        query: &crate::text::TextQuery,
        k: Option<usize>,
        groups: &[(String, Vec<Value>)],
    ) -> Result<Option<Vec<(NodeId, f64)>>> {
        if !self
            .manifest
            .manifest
            .text_indexes
            .iter()
            .any(|descriptor| descriptor.name == index_name && descriptor.label == label)
        {
            return Ok(None);
        }
        let dict = &self.manifest.manifest.label_dict;
        let mut dirty = Vec::new();
        for (key, entry) in self.node_entries() {
            let MemKey::Node { id } = key else {
                continue;
            };
            match &entry.op {
                MemOp::Tombstone => dirty.push(*id.0.as_bytes()),
                MemOp::Upsert(payload) => {
                    let record = NodeWriteRecord::decode(payload)?;
                    if record_carries_label(&record, label, dict) {
                        return Ok(None);
                    }
                    dirty.push(*id.0.as_bytes());
                }
            }
        }
        match search_lsm_read::text_search(self, index_name, query, k, groups, &dirty).await? {
            search_lsm_read::ActiveSearch::Ready(hits) => Ok(Some(hits)),
            search_lsm_read::ActiveSearch::Unavailable
            | search_lsm_read::ActiveSearch::NotActive => Ok(None),
        }
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
    async fn fetch_paged_edge_reader(&self, absolute: &str) -> Result<Arc<PagedEdgeReader>> {
        namidb_core::profile_scope!("Snapshot::fetch_paged_edge_reader");
        let cache_key = absolute.to_string();
        if let Some(reader) = self.paged_edge_readers.lock().unwrap().get(&cache_key) {
            return Ok(reader);
        }
        let reader =
            Arc::new(PagedEdgeReader::open(self.store.clone(), Path::from(absolute)).await?);
        let weight = absolute
            .len()
            .saturating_add(reader.resident_metadata_bytes())
            .saturating_add(SNAPSHOT_CACHE_ENTRY_OVERHEAD_BYTES);
        self.paged_edge_readers
            .lock()
            .unwrap()
            .insert(cache_key, reader.clone(), weight);
        Ok(reader)
    }

    async fn fetch_unique_property_sidecar(
        &self,
        absolute: &str,
    ) -> Result<Arc<UniquePropertySidecar>> {
        if let Some(cache) = self.cache.as_ref() {
            if let Some(index) = cache.get_unique_property_sidecar(absolute) {
                return Ok(index);
            }
        }
        let body = self.fetch_bytes(absolute).await?;
        let index: UniquePropertySidecar = bincode::deserialize(&body)
            .map_err(|e| Error::invariant(format!("unique-index bincode decode: {e}")))?;
        let index = Arc::new(index);
        if let Some(cache) = self.cache.as_ref() {
            cache.insert_unique_property_sidecar(absolute.to_string(), index.clone());
        }
        Ok(index)
    }

    /// Point-probe only the B+tree pages containing `values`. Legacy
    /// bincode sidecars retain their whole-map fallback for rolling upgrades.
    async fn probe_unique_property_sidecar(
        &self,
        descriptor: &UniquePropertyIndexDescriptor,
        absolute: &str,
        values: &[String],
    ) -> Result<Arc<UniquePropertySidecar>> {
        if let Some(paged) = &descriptor.paged {
            let paged_absolute =
                format!("{}/{}", self.paths.namespace_prefix().as_ref(), paged.path);
            let found = match self
                .pinned_sidecar_source(&paged_absolute, Some(paged.size_bytes))
                .await
            {
                Ok(source) => {
                    crate::sst::paged_index::probe_unique_from_source(&source, values).await
                }
                Err(error) => Err(error),
            };
            match found {
                Ok((found, stats)) if stats.index_entries == descriptor.entry_count => {
                    return Ok(Arc::new(found));
                }
                Ok((_found, stats)) => {
                    tracing::warn!(
                        path = %paged.path,
                        expected_entries = descriptor.entry_count,
                        actual_entries = stats.index_entries,
                        "paged unique accelerator is stale/partial; falling back to legacy sidecar"
                    );
                }
                Err(error) if optional_accelerator_fallback(&error) => {
                    tracing::warn!(
                        path = %paged.path,
                        error = %error,
                        "paged unique accelerator unavailable; falling back to legacy sidecar"
                    );
                }
                Err(error) => return Err(error),
            }
        }
        match descriptor.format {
            PropertyIndexFormat::BincodeV0 => {
                let full = self.fetch_unique_property_sidecar(absolute).await?;
                Ok(Arc::new(
                    values
                        .iter()
                        .filter_map(|value| full.get(value).copied().map(|id| (value.clone(), id)))
                        .collect(),
                ))
            }
            PropertyIndexFormat::PagedV1 => {
                let source = self
                    .pinned_sidecar_source(absolute, Some(descriptor.size_bytes))
                    .await?;
                let (found, stats) =
                    crate::sst::paged_index::probe_unique_from_source(&source, values).await?;
                if stats.index_entries != descriptor.entry_count {
                    return Err(Error::invariant(format!(
                        "paged unique entry count mismatch: manifest {}, header {}",
                        descriptor.entry_count, stats.index_entries
                    )));
                }
                Ok(Arc::new(found))
            }
        }
    }

    async fn fetch_equality_property_sidecar(
        &self,
        absolute: &str,
    ) -> Result<Arc<EqualityPropertySidecar>> {
        if let Some(cache) = self.cache.as_ref() {
            if let Some(index) = cache.get_equality_property_sidecar(absolute) {
                return Ok(index);
            }
        }
        let body = self.fetch_bytes(absolute).await?;
        let index: EqualityPropertySidecar = bincode::deserialize(&body)
            .map_err(|e| Error::invariant(format!("equality-index bincode decode: {e}")))?;
        let index = Arc::new(index);
        if let Some(cache) = self.cache.as_ref() {
            cache.insert_equality_property_sidecar(absolute.to_string(), index.clone());
        }
        Ok(index)
    }

    async fn probe_equality_property_sidecar(
        &self,
        descriptor: &EqualityIndexDescriptor,
        absolute: &str,
        values: &[String],
    ) -> Result<Arc<EqualityPropertySidecar>> {
        if let Some(paged) = &descriptor.paged {
            let paged_absolute =
                format!("{}/{}", self.paths.namespace_prefix().as_ref(), paged.path);
            let found = match self
                .pinned_sidecar_source(&paged_absolute, Some(paged.size_bytes))
                .await
            {
                Ok(source) => {
                    crate::sst::paged_index::probe_equality_from_source(&source, values).await
                }
                Err(error) => Err(error),
            };
            match found {
                Ok((found, stats)) if stats.index_entries == descriptor.distinct_values => {
                    return Ok(Arc::new(found));
                }
                Ok((_found, stats)) => {
                    tracing::warn!(
                        path = %paged.path,
                        expected_entries = descriptor.distinct_values,
                        actual_entries = stats.index_entries,
                        "paged equality accelerator is stale/partial; falling back to legacy sidecar"
                    );
                }
                Err(error) if optional_accelerator_fallback(&error) => {
                    tracing::warn!(
                        path = %paged.path,
                        error = %error,
                        "paged equality accelerator unavailable; falling back to legacy sidecar"
                    );
                }
                Err(error) => return Err(error),
            }
        }
        match descriptor.format {
            PropertyIndexFormat::BincodeV0 => {
                let full = self.fetch_equality_property_sidecar(absolute).await?;
                Ok(Arc::new(
                    values
                        .iter()
                        .filter_map(|value| {
                            full.get(value).cloned().map(|ids| (value.clone(), ids))
                        })
                        .collect(),
                ))
            }
            PropertyIndexFormat::PagedV1 => {
                let source = self
                    .pinned_sidecar_source(absolute, Some(descriptor.size_bytes))
                    .await?;
                let (found, stats) =
                    crate::sst::paged_index::probe_equality_from_source(&source, values).await?;
                if stats.index_entries != descriptor.distinct_values {
                    return Err(Error::invariant(format!(
                        "paged equality entry count mismatch: manifest {}, header {}",
                        descriptor.distinct_values, stats.index_entries
                    )));
                }
                if let Some(cache) = &self.property_index_cache {
                    cache.record_equality_index_bytes_read(stats.bytes_read);
                }
                Ok(Arc::new(found))
            }
        }
    }

    async fn probe_equality_property_sidecar_limited(
        &self,
        descriptor: &EqualityIndexDescriptor,
        absolute: &str,
        values: &[String],
        max_ids_per_value: Option<usize>,
    ) -> Result<(Arc<EqualityPropertySidecar>, usize, bool)> {
        if let Some(paged) = &descriptor.paged {
            let paged_absolute =
                format!("{}/{}", self.paths.namespace_prefix().as_ref(), paged.path);
            let probed = match self
                .pinned_sidecar_source(&paged_absolute, Some(paged.size_bytes))
                .await
            {
                Ok(source) => {
                    crate::sst::paged_index::probe_equality_limited_from_source(
                        &source,
                        values,
                        max_ids_per_value,
                    )
                    .await
                }
                Err(error) => Err(error),
            };
            match probed {
                Ok((found, stats)) if stats.index_entries == descriptor.distinct_values => {
                    if let Some(cache) = &self.property_index_cache {
                        cache.record_equality_index_bytes_read(stats.bytes_read);
                    }
                    return Ok((
                        Arc::new(found),
                        (stats.matched_value_bytes / 16) as usize,
                        stats.values_truncated,
                    ));
                }
                Ok((_found, stats)) => {
                    tracing::warn!(
                        path = %paged.path,
                        expected_entries = descriptor.distinct_values,
                        actual_entries = stats.index_entries,
                        "paged equality accelerator is stale/partial; falling back to legacy sidecar"
                    );
                }
                Err(error) if optional_accelerator_fallback(&error) => {
                    tracing::warn!(
                        path = %paged.path,
                        error = %error,
                        "paged equality accelerator unavailable; falling back to legacy sidecar"
                    );
                }
                Err(error) => return Err(error),
            }
        }
        let map = self
            .probe_equality_property_sidecar(descriptor, absolute, values)
            .await?;
        let total = map.values().map(Vec::len).sum();
        Ok((map, total, false))
    }

    async fn fetch_equality_property_sidecar_all(
        &self,
        descriptor: &EqualityIndexDescriptor,
        absolute: &str,
    ) -> Result<Arc<EqualityPropertySidecar>> {
        match descriptor.format {
            PropertyIndexFormat::BincodeV0 => self.fetch_equality_property_sidecar(absolute).await,
            PropertyIndexFormat::PagedV1 => {
                let body = self.fetch_bytes(absolute).await?;
                let decoded = crate::sst::paged_index::decode_all_equality(&body)?;
                if decoded.len() as u64 != descriptor.distinct_values {
                    return Err(Error::invariant(format!(
                        "paged equality entry count mismatch: manifest {}, decoded {}",
                        descriptor.distinct_values,
                        decoded.len()
                    )));
                }
                Ok(Arc::new(decoded))
            }
        }
    }

    async fn fetch_equality_property_sidecar_prefix(
        &self,
        descriptor: &EqualityIndexDescriptor,
        absolute: &str,
        min_postings: usize,
    ) -> Result<(Arc<EqualityPropertySidecar>, bool)> {
        if let Some(paged) = &descriptor.paged {
            let paged_absolute =
                format!("{}/{}", self.paths.namespace_prefix().as_ref(), paged.path);
            let prefix = match self
                .pinned_sidecar_source(&paged_absolute, Some(paged.size_bytes))
                .await
            {
                Ok(source) => {
                    crate::sst::paged_index::equality_prefix_from_source(&source, min_postings)
                        .await
                }
                Err(error) => Err(error),
            };
            match prefix {
                Ok((map, stats)) if stats.index_entries == descriptor.distinct_values => {
                    if let Some(cache) = &self.property_index_cache {
                        cache.record_ordered_prefix_index_bytes_read(stats.bytes_read);
                    }
                    return Ok((Arc::new(map), stats.values_truncated));
                }
                Ok((_map, stats)) => {
                    tracing::warn!(
                        path = %paged.path,
                        expected_entries = descriptor.distinct_values,
                        actual_entries = stats.index_entries,
                        "paged equality prefix is stale/partial; falling back to legacy sidecar"
                    );
                }
                Err(error) if optional_accelerator_fallback(&error) => {
                    tracing::warn!(
                        path = %paged.path,
                        error = %error,
                        "paged equality prefix unavailable; falling back to legacy sidecar"
                    );
                }
                Err(error) => return Err(error),
            }
        }
        if descriptor.format == PropertyIndexFormat::PagedV1 {
            let source = self
                .pinned_sidecar_source(absolute, Some(descriptor.size_bytes))
                .await?;
            let (map, stats) =
                crate::sst::paged_index::equality_prefix_from_source(&source, min_postings).await?;
            if stats.index_entries != descriptor.distinct_values {
                return Err(Error::invariant(format!(
                    "paged equality entry count mismatch: manifest {}, header {}",
                    descriptor.distinct_values, stats.index_entries
                )));
            }
            if let Some(cache) = &self.property_index_cache {
                cache.record_ordered_prefix_index_bytes_read(stats.bytes_read);
            }
            return Ok((Arc::new(map), stats.values_truncated));
        }
        Ok((
            self.fetch_equality_property_sidecar_all(descriptor, absolute)
                .await?,
            false,
        ))
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
        if let Some(cache) = self.cache.as_ref() {
            if let Some(filter) = cache.get_bloom_filter(&absolute) {
                return Ok(filter.contains(key));
            }
        }
        let body = self.fetch_bytes(&absolute).await?;
        let filter = Arc::new(BloomFilter::from_bytes(&bloom_desc.path, &body)?);
        if let Some(cache) = self.cache.as_ref() {
            cache.insert_bloom_filter(absolute, filter.clone());
        }
        Ok(filter.contains(key))
    }

    /// Cache-only check: returns `Some(body)` if the cache has it,
    /// `None` if not present or no cache attached. Used by
    /// `lookup_node` to decide between the sync (cache-hit) and async
    /// (cold ranged-read) paths.
    fn cache_get(&self, absolute: &str) -> Option<Bytes> {
        self.cache.as_ref().and_then(|c| c.get(absolute))
    }

    /// Open and validate the independently ranged property object for one
    /// Nodes SST. `None` selects the authoritative Parquet fallback.
    async fn node_property_reader(
        &self,
        desc: &SstDescriptor,
    ) -> Result<Option<Arc<NodePropertyPageReader>>> {
        let Some(properties) = crate::manifest::node_property_pages_sidecar(desc) else {
            return Ok(None);
        };
        if properties.format_version != NODE_PROPERTY_PAGES_FORMAT_VERSION
            || !properties.is_bound_to(desc)
        {
            return Ok(None);
        }
        let absolute = format!(
            "{}/{}",
            self.paths.namespace_prefix().as_ref(),
            properties.path
        );
        if let Some(reader) = self
            .node_property_readers
            .lock()
            .expect("node property reader cache mutex poisoned")
            .get(&absolute)
        {
            return Ok(Some(reader));
        }

        let source = match self
            .pinned_sidecar_source(&absolute, Some(properties.size_bytes))
            .await
        {
            Ok(source) => Arc::new(source),
            Err(error) if optional_accelerator_fallback(&error) => {
                tracing::warn!(
                    path = %absolute,
                    %error,
                    "node property pages unavailable; using exact Parquet fallback"
                );
                return Ok(None);
            }
            Err(error) => return Err(error),
        };
        let config = NodePropertyPageConfig::from_env()?;
        let reader = match NodePropertyPageReader::open(source, desc.id, config).await {
            Ok(reader) => Arc::new(reader),
            Err(error) if optional_accelerator_fallback(&error) => {
                tracing::warn!(
                    path = %absolute,
                    %error,
                    "node property pages failed validation; using exact Parquet fallback"
                );
                return Ok(None);
            }
            Err(error) => return Err(error),
        };
        if reader.sst_id() != properties.id
            || reader.node_count() != properties.node_count
            || reader.cell_count() != properties.cell_count
            || reader.property_count() != properties.property_count
            || reader.page_count() != properties.page_count
            || reader.content_xxh3() != properties.content_xxh3
            || !reader.is_complete()
        {
            tracing::warn!(
                path = %absolute,
                "node property descriptor does not match its authenticated footer; \
                 using exact Parquet fallback"
            );
            return Ok(None);
        }
        self.node_property_readers
            .lock()
            .expect("node property reader cache mutex poisoned")
            .insert(absolute, reader.clone(), reader.resident_metadata_bytes());
        Ok(Some(reader))
    }

    async fn project_node_property_batch(
        &self,
        reader: &NodePropertyPageReader,
        desc: &SstDescriptor,
        projection: &[String],
        batch: &RecordBatch,
        first_ordinal: u64,
    ) -> Result<Option<Vec<ProjectedNodeCandidate>>> {
        let id_col = batch
            .column_by_name(COL_NODE_ID)
            .and_then(|column| column.as_any().downcast_ref::<FixedSizeBinaryArray>())
            .ok_or_else(|| Error::invariant("node_id column missing"))?;
        let tomb_col = batch
            .column_by_name(COL_TOMBSTONE)
            .and_then(|column| column.as_any().downcast_ref::<BooleanArray>())
            .ok_or_else(|| Error::invariant("tombstone column missing"))?;
        let lsn_col = batch
            .column_by_name(COL_LSN)
            .and_then(|column| column.as_any().downcast_ref::<UInt64Array>())
            .ok_or_else(|| Error::invariant("lsn column missing"))?;
        let schema_version_col = batch
            .column_by_name(SCHEMA_VERSION)
            .and_then(|column| column.as_any().downcast_ref::<UInt64Array>())
            .ok_or_else(|| Error::invariant("__schema_version column missing"))?;
        let ids = (0..batch.num_rows())
            .map(|row| {
                id_col
                    .value(row)
                    .try_into()
                    .map_err(|_| Error::invariant("node_id row length != 16"))
            })
            .collect::<Result<Vec<[u8; 16]>>>()?;
        let projected = match reader.project_node_ids(projection, &ids).await {
            Ok((projected, _)) => projected,
            Err(error) if optional_accelerator_fallback(&error) => {
                tracing::warn!(
                    sst_id = %desc.id,
                    %error,
                    "node property page probe failed; using exact Parquet fallback"
                );
                return Ok(None);
            }
            Err(error) => return Err(error),
        };
        if projected.len() != ids.len() {
            return Ok(None);
        }
        let mut candidates = Vec::with_capacity(batch.num_rows());
        for row in 0..batch.num_rows() {
            let id = ids[row];
            let ordinal = first_ordinal
                .checked_add(row as u64)
                .ok_or_else(|| Error::invariant("node property ordinal exceeds u64"))?;
            let property_row = &projected[row];
            if property_row.node_id != id
                || property_row.ordinal.is_some_and(|found| found != ordinal)
            {
                tracing::warn!(
                    sst_id = %desc.id,
                    "node property row binding mismatch; using exact Parquet fallback"
                );
                return Ok(None);
            }
            let node_id = NodeId::from_uuid(Uuid::from_bytes(id));
            let lsn = lsn_col.value(row);
            if tomb_col.value(row) {
                candidates.push((node_id, lsn, None));
                continue;
            }
            let mut properties = BTreeMap::new();
            for (name, cell) in &property_row.properties {
                match cell {
                    PropertyCell::Absent => {}
                    PropertyCell::Null => {
                        properties.insert(name.clone(), Value::Null);
                    }
                    PropertyCell::Value(value) => {
                        properties.insert(name.clone(), value.clone());
                    }
                }
            }
            candidates.push((
                node_id,
                lsn,
                Some(NodeView {
                    id: node_id,
                    labels: decode_node_labels(
                        batch,
                        row,
                        &self.manifest.manifest.label_dict,
                        &desc.scope,
                    ),
                    properties,
                    lsn,
                    schema_version: schema_version_col.value(row),
                }),
            ));
        }
        Ok(Some(candidates))
    }

    /// Read only structural Parquet columns plus the explicitly named
    /// property pages. No property/overflow column from the Nodes SST is
    /// fetched. The complete per-SST result is retained until every requested
    /// page has validated, so an optional-sidecar failure can restart from
    /// Parquet without exposing a partial answer.
    async fn try_read_projected_node_sst(
        &self,
        desc: &SstDescriptor,
        sst_label_def: &LabelDef,
        projection: &[String],
    ) -> Result<Option<Vec<ProjectedNodeCandidate>>> {
        let Some(reader) = self.node_property_reader(desc).await? else {
            return Ok(None);
        };
        let absolute = format!("{}/{}", self.paths.namespace_prefix().as_ref(), desc.path);
        let cached_metadata = self
            .cache
            .as_ref()
            .and_then(|cache| cache.get_metadata(&absolute));
        let metadata_was_cached = cached_metadata.is_some();
        let (mut stream, metadata) = node_scan_limited_async(
            self.store.clone(),
            Path::from(absolute.clone()),
            desc.size_bytes,
            sst_label_def,
            &[],
            Some(&[]),
            cached_metadata,
        )
        .await?;
        if !metadata_was_cached {
            if let Some(cache) = &self.cache {
                cache.insert_metadata(absolute, metadata);
            }
        }

        let mut candidates = Vec::with_capacity(
            usize::try_from(desc.row_count)
                .unwrap_or(usize::MAX)
                .min(4096),
        );
        let mut next_ordinal = 0_u64;
        while let Some(batches) = stream.next_row_group().await? {
            for batch in batches {
                let batch = batch.map_err(|error| {
                    Error::invariant(format!("projected Parquet scan read: {error}"))
                })?;
                crate::cancel::check()?;
                let Some(mut batch_candidates) = self
                    .project_node_property_batch(
                        reader.as_ref(),
                        desc,
                        projection,
                        &batch,
                        next_ordinal,
                    )
                    .await?
                else {
                    return Ok(None);
                };
                candidates.append(&mut batch_candidates);
                next_ordinal = next_ordinal
                    .checked_add(batch.num_rows() as u64)
                    .ok_or_else(|| Error::invariant("node property ordinal exceeds u64"))?;
            }
        }
        if next_ordinal != desc.row_count || candidates.len() as u64 != desc.row_count {
            tracing::warn!(
                sst_id = %desc.id,
                expected = desc.row_count,
                observed = next_ordinal,
                "node property/Parquet row-count mismatch; using exact Parquet fallback"
            );
            return Ok(None);
        }
        Ok(Some(candidates))
    }

    /// Open one immutable UUID sidecar through the shared generation-pinned
    /// RAM/NVMe page source. The manifest size, when available, is part of the
    /// integrity envelope rather than merely a cache-accounting hint.
    async fn pinned_sidecar_source(
        &self,
        absolute: &str,
        expected_size: Option<u64>,
    ) -> Result<crate::range_cache::PinnedObjectRangeSource> {
        let path = Path::from(absolute);
        let meta = self.store.head(&path).await?;
        if meta.location != path {
            return Err(Error::Corrupted {
                path: absolute.to_string(),
                detail: format!(
                    "sidecar HEAD returned metadata for unexpected path {}",
                    meta.location
                ),
            });
        }
        if let Some(expected) = expected_size {
            if meta.size != expected {
                return Err(Error::Corrupted {
                    path: absolute.to_string(),
                    detail: format!(
                        "sidecar object size {} differs from manifest size {expected}",
                        meta.size
                    ),
                });
            }
        }
        crate::range_cache::PinnedObjectRangeSource::from_create_only_meta(self.store.clone(), meta)
            .await
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

#[derive(Debug, Default)]
struct LimitedNodeBatchWork {
    decoded_rows: usize,
    examined_rows: usize,
}

struct LimitedNodeBatchContext<'a> {
    sst_label_def: &'a LabelDef,
    desc: &'a SstDescriptor,
    dict: &'a LabelDictionary,
    label: Option<&'a str>,
    predicates: &'a [ScanPredicate],
    decode_projection: Option<&'a [String]>,
    requested_projection: Option<&'a BTreeSet<&'a str>>,
    limit: usize,
}

type ProjectedNodeCandidate = (NodeId, u64, Option<NodeView>);

/// Consume lazy Parquet batches only until `out` contains `limit` exact live
/// matches. The iterator itself remains unconsumed after that point, which is
/// what makes both cached-body and ranged scans stop doing Arrow work.
fn consume_limited_node_batches<I, E>(
    batches: I,
    context: &LimitedNodeBatchContext<'_>,
    out: &mut Vec<NodeView>,
) -> Result<LimitedNodeBatchWork>
where
    I: IntoIterator<Item = std::result::Result<RecordBatch, E>>,
    E: std::fmt::Display,
{
    let mut work = LimitedNodeBatchWork::default();
    for batch in batches {
        let batch =
            batch.map_err(|error| Error::invariant(format!("parquet limited read: {error}")))?;
        work.decoded_rows = work.decoded_rows.saturating_add(batch.num_rows());
        work.examined_rows = work
            .examined_rows
            .saturating_add(append_limited_node_batch(&batch, context, out)?);
        if out.len() == context.limit {
            break;
        }
        crate::cancel::check()?;
    }
    Ok(work)
}

fn append_limited_node_batch(
    batch: &RecordBatch,
    context: &LimitedNodeBatchContext<'_>,
    out: &mut Vec<NodeView>,
) -> Result<usize> {
    let sst_label_def = context.sst_label_def;
    let desc = context.desc;
    let dict = context.dict;
    let label = context.label;
    let predicates = context.predicates;
    let decode_projection = context.decode_projection;
    let requested_projection = context.requested_projection;
    let limit = context.limit;
    let id_col = batch
        .column_by_name(COL_NODE_ID)
        .and_then(|column| column.as_any().downcast_ref::<FixedSizeBinaryArray>())
        .ok_or_else(|| Error::invariant("node_id column missing"))?;
    let tomb_col = batch
        .column_by_name(COL_TOMBSTONE)
        .and_then(|column| column.as_any().downcast_ref::<BooleanArray>())
        .ok_or_else(|| Error::invariant("tombstone column missing"))?;
    let lsn_col = batch
        .column_by_name(COL_LSN)
        .and_then(|column| column.as_any().downcast_ref::<UInt64Array>())
        .ok_or_else(|| Error::invariant("lsn column missing"))?;
    let schema_version_col = batch
        .column_by_name(SCHEMA_VERSION)
        .and_then(|column| column.as_any().downcast_ref::<UInt64Array>())
        .ok_or_else(|| Error::invariant("__schema_version column missing"))?;
    let overflow_col = batch
        .column_by_name(OVERFLOW_JSON)
        .and_then(|column| column.as_any().downcast_ref::<StringArray>());
    let decode_set: Option<BTreeSet<&str>> =
        decode_projection.map(|columns| columns.iter().map(String::as_str).collect());

    let mut examined = 0usize;
    for row in 0..batch.num_rows() {
        if out.len() == limit {
            break;
        }
        examined = examined.saturating_add(1);
        if tomb_col.value(row) {
            continue;
        }
        let labels = decode_node_labels(batch, row, dict, &desc.scope);
        if label.is_some_and(|label| !labels.contains(label)) {
            continue;
        }

        let row_id_bytes: [u8; 16] = id_col
            .value(row)
            .try_into()
            .map_err(|_| Error::invariant("node_id row length != 16"))?;
        let mut properties = BTreeMap::new();
        for property in &sst_label_def.properties {
            if decode_set
                .as_ref()
                .is_some_and(|keep| !keep.contains(property.name.as_str()))
            {
                continue;
            }
            let parquet_name = prop_column_name(property);
            let Some(column) = batch.column_by_name(&parquet_name) else {
                continue;
            };
            if let Some(value) = arrow_value_to_value(column.as_ref(), row, &property.data_type)? {
                properties.insert(property.name.clone(), value);
            }
        }
        if decode_set.as_ref().is_none_or(|keep| !keep.is_empty()) {
            if let Some(overflow_col) = overflow_col.filter(|column| !column.is_null(row)) {
                let extra: BTreeMap<String, Value> = serde_json::from_str(overflow_col.value(row))?;
                if let Some(keep) = &decode_set {
                    properties.extend(
                        extra
                            .into_iter()
                            .filter(|(property, _)| keep.contains(property.as_str())),
                    );
                } else {
                    properties.extend(extra);
                }
            }
        }

        let mut view = NodeView {
            id: NodeId::from_uuid(Uuid::from_bytes(row_id_bytes)),
            labels,
            properties,
            lsn: lsn_col.value(row),
            schema_version: schema_version_col.value(row),
        };
        if !node_view_matches_predicates(&view, predicates) {
            continue;
        }
        if let Some(requested) = requested_projection {
            view.properties
                .retain(|property, _| requested.contains(property.as_str()));
        }
        out.push(view);
    }
    Ok(examined)
}

fn optional_accelerator_fallback(error: &Error) -> bool {
    matches!(
        error,
        Error::ObjectStore(
            object_store::Error::NotFound { .. } | object_store::Error::Precondition { .. }
        ) | Error::Invariant(_)
            | Error::Corrupted { .. }
    )
}

fn update_edge_point_winner(
    winners: &mut BTreeMap<(NodeId, NodeId), Option<(u64, EdgePointWinner)>>,
    pair: (NodeId, NodeId),
    lsn: u64,
    winner: EdgePointWinner,
) {
    let replace = winners
        .get(&pair)
        .and_then(Option::as_ref)
        .is_none_or(|(current_lsn, _)| lsn > *current_lsn);
    if replace {
        winners.insert(pair, Some((lsn, winner)));
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

/// Cursor over one posting list for a single equality value. Each variant owns
/// the backing allocation (or its `Arc`), allowing an async lookup to k-way
/// merge SST + memtable postings without first unioning a high-cardinality
/// value into a second corpus-sized set.
enum IndexedPropertyLookup {
    Available(Vec<NodeId>),
    Unavailable,
    Truncated,
}

enum EqualityNodePostingCursor {
    Owned {
        ids: Vec<NodeId>,
        position: usize,
    },
    Memtable {
        map: Arc<crate::property_index::MemtableClaimantIndex>,
        key: String,
        current: Option<NodeId>,
    },
    Sidecar {
        map: Arc<EqualityPropertySidecar>,
        key: String,
        position: usize,
        truncated: bool,
    },
}

impl EqualityNodePostingCursor {
    fn memtable(map: Arc<crate::property_index::MemtableClaimantIndex>, key: String) -> Self {
        let current = map.get(&key).and_then(imbl::OrdSet::get_min).copied();
        Self::Memtable { map, key, current }
    }

    fn len(&self) -> usize {
        match self {
            Self::Owned { ids, .. } => ids.len(),
            Self::Memtable { map, key, .. } => map.get(key).map_or(0, imbl::OrdSet::len),
            Self::Sidecar { map, key, .. } => map.get(key).map_or(0, Vec::len),
        }
    }

    fn current(&self) -> Option<NodeId> {
        match self {
            Self::Owned { ids, position } => ids.get(*position).copied(),
            Self::Memtable { current, .. } => *current,
            Self::Sidecar {
                map, key, position, ..
            } => map
                .get(key)?
                .get(*position)
                .map(|bytes| NodeId::from_uuid(Uuid::from_bytes(*bytes))),
        }
    }

    fn advance(&mut self) {
        match self {
            Self::Owned { position, .. } | Self::Sidecar { position, .. } => *position += 1,
            Self::Memtable { map, key, current } => {
                let Some(frontier) = *current else {
                    return;
                };
                *current = map.get(key).and_then(|ids| {
                    ids.range((
                        std::ops::Bound::Excluded(frontier),
                        std::ops::Bound::Unbounded,
                    ))
                    .next()
                    .copied()
                });
            }
        }
    }

    fn has_truncated_suffix(&self) -> bool {
        matches!(
            self,
            Self::Sidecar {
                truncated: true,
                ..
            }
        )
    }

    /// Whether an intentionally unread suffix can contain an id below the
    /// current global LIMIT cutoff.
    ///
    /// Posting lists are strictly NodeId-ascending. While a loaded candidate
    /// remains, it is the tightest lower bound for the suffix. Once the loaded
    /// prefix is exhausted, the final loaded id is a conservative lower bound
    /// for the first omitted id, which is known to be strictly greater.
    fn unread_may_precede(&self, cutoff: NodeId) -> bool {
        let Self::Sidecar {
            map,
            key,
            position,
            truncated: true,
        } = self
        else {
            return false;
        };
        let Some(ids) = map.get(key) else {
            return true;
        };
        let frontier = ids.get(*position).or_else(|| ids.last()).copied();
        frontier.is_none_or(|bytes| NodeId::from_uuid(Uuid::from_bytes(bytes)) < cutoff)
    }
}

/// Pop the lowest NodeId across sorted posting cursors and advance every
/// source currently pointing at it. Advancing all equal heads deduplicates
/// stale versions across overlapping LSM levels without a `BTreeSet`.
fn next_equality_candidate(cursors: &mut [EqualityNodePostingCursor]) -> Option<NodeId> {
    let next = cursors.iter().filter_map(|cursor| cursor.current()).min()?;
    for cursor in cursors {
        if cursor.current() == Some(next) {
            cursor.advance();
        }
    }
    Some(next)
}

const ORDERED_PREFIX_INITIAL_CAPACITY: usize = 256;

fn ordered_prefix_initial_capacity(limit: usize) -> usize {
    limit.min(ORDERED_PREFIX_INITIAL_CAPACITY)
}

/// One cursor over a decoded String-property sidecar. Current equality
/// sidecars expose posting lists; legacy label-scoped unique sidecars expose
/// one id per key. It owns the `Arc` and keeps only the current key + posting
/// offset, so ordered pagination does not materialise a second merged map.
enum OrderedStringPostingSource {
    Equality(Arc<EqualityPropertySidecar>),
    Unique(Arc<UniquePropertySidecar>),
}

struct OrderedStringPostingCursor {
    source: OrderedStringPostingSource,
    key: String,
    posting: usize,
    truncated: bool,
    /// Populated only when the loaded prefix is exhausted. While a current
    /// tuple exists it is already the tighter lower bound for unseen tuples.
    last_advanced: Option<(String, NodeId)>,
}

impl OrderedStringPostingCursor {
    fn new_equality(
        map: Arc<EqualityPropertySidecar>,
        _encoding: crate::manifest::EqualityKeyEncoding,
        truncated: bool,
    ) -> Option<Self> {
        // ScalarV1 intentionally keeps String keys raw so an older reader can
        // consume newly-written sidecars during rolling upgrades. Tagged
        // non-String keys may be interleaved here; current-value confirmation
        // below discards those conservative candidates.
        let key = map
            .iter()
            .find(|(_, ids)| !ids.is_empty())
            .map(|(key, _)| key.clone())?;
        Some(Self {
            source: OrderedStringPostingSource::Equality(map),
            key,
            posting: 0,
            truncated,
            last_advanced: None,
        })
    }

    fn new_unique(map: Arc<UniquePropertySidecar>) -> Option<Self> {
        let key = map.keys().next()?.clone();
        Some(Self {
            source: OrderedStringPostingSource::Unique(map),
            key,
            posting: 0,
            truncated: false,
            last_advanced: None,
        })
    }

    fn current(&self) -> Option<(&str, NodeId)> {
        let id = match &self.source {
            OrderedStringPostingSource::Equality(map) => {
                let ids = map.get(&self.key)?;
                *ids.get(self.posting)?
            }
            OrderedStringPostingSource::Unique(map) => *map.get(&self.key)?,
        };
        Some((self.key.as_str(), NodeId::from_uuid(Uuid::from_bytes(id))))
    }

    fn advance(&mut self) {
        let posting_continues = match &self.source {
            OrderedStringPostingSource::Equality(map) => map
                .get(&self.key)
                .is_some_and(|ids| self.posting + 1 < ids.len()),
            OrderedStringPostingSource::Unique(_) => false,
        };
        if posting_continues {
            self.posting += 1;
            return;
        }
        use std::ops::Bound::{Excluded, Unbounded};
        let next = match &self.source {
            OrderedStringPostingSource::Equality(map) => map
                .range((Excluded(self.key.clone()), Unbounded))
                .find(|(_, ids)| !ids.is_empty())
                .map(|(key, _)| key.clone()),
            OrderedStringPostingSource::Unique(map) => map
                .range((Excluded(self.key.clone()), Unbounded))
                .next()
                .map(|(key, _)| key.clone()),
        };
        match next {
            Some(key) => {
                self.key = key;
                self.posting = 0;
            }
            None => {
                let advanced = self.current().map(|(value, id)| (value.to_string(), id));
                self.last_advanced = advanced;
                self.key.clear();
                self.posting = usize::MAX;
            }
        }
    }

    /// Whether this source's unread suffix could contain a tuple strictly
    /// before the confirmed global cutoff. Equality postings are ordered by
    /// `(value, NodeId)`, so a visible current tuple is a lower bound; after
    /// exhaustion, the first omitted tuple is strictly greater than the last
    /// advanced tuple.
    fn unread_may_precede(&self, cutoff: &(String, NodeId)) -> bool {
        if !self.truncated {
            return false;
        }
        let frontier = self.current().or_else(|| {
            self.last_advanced
                .as_ref()
                .map(|(value, id)| (value.as_str(), *id))
        });
        frontier.is_none_or(|(value, id)| {
            value < cutoff.0.as_str() || (value == cutoff.0.as_str() && id < cutoff.1)
        })
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

/// Encode one probe for the sidecar format declared in the manifest.
///
/// Legacy sidecars can answer String probes only. Scalar-v1 uses the tagged
/// representation shared with the flush/compaction harvesters.
fn equality_sidecar_key(
    encoding: crate::manifest::EqualityKeyEncoding,
    value: &Value,
) -> Option<String> {
    match encoding {
        crate::manifest::EqualityKeyEncoding::StringV0 => match value {
            Value::Str(value) => Some(value.clone()),
            _ => None,
        },
        crate::manifest::EqualityKeyEncoding::ScalarV1 => match value {
            Value::Str(_) | Value::Bool(_) => crate::cache::encode_equality_property_value(value),
            _ => None,
        },
    }
}

/// Conservative overlap test for descriptor key ranges. Invalid/reversed
/// metadata cannot prove disjointness and therefore overlaps by definition.
#[cfg(any(feature = "vector-index", feature = "text-index"))]
fn key_ranges_may_overlap(a: &SstDescriptor, b: &SstDescriptor) -> bool {
    if a.min_key > a.max_key || b.min_key > b.max_key {
        return true;
    }
    a.min_key <= b.max_key && b.min_key <= a.max_key
}

/// One authoritative String-property claimant source for a node SST.
///
/// Legacy label-scoped SSTs encode schema-unique properties as a single-value
/// map, while current id-primary SSTs encode the same logical key as a global
/// equality posting map. A rolling-upgrade manifest can contain both at once,
/// so completeness must be decided per SST rather than by requiring one format
/// across the complete generation.
enum StringPropertySidecar<'a> {
    Unique(&'a UniquePropertyIndexDescriptor),
    Equality(&'a EqualityIndexDescriptor),
}

fn string_property_sidecar<'a>(
    descriptor: &'a SstDescriptor,
    label: &str,
    property: &str,
) -> Option<StringPropertySidecar<'a>> {
    // A single-value unique map is authoritative only for a legacy
    // label-scoped SST whose scope is exactly the requested label. On an
    // id-primary SST (`scope == ""`), equal values from different labels are
    // legal and a single-value map can silently discard this label's owner.
    if !descriptor.scope.is_empty() && descriptor.scope == label {
        if let Some(sidecar) = descriptor
            .unique_property_indices
            .iter()
            .find(|sidecar| sidecar.property == property)
        {
            return Some(StringPropertySidecar::Unique(sidecar));
        }
    }
    descriptor
        .equality_property_indices
        .iter()
        .find(|sidecar| {
            sidecar.property == property
                && sidecar.mixed_type_complete
                && equality_sidecar_key(sidecar.key_encoding, &Value::Str(String::new())).is_some()
        })
        .map(StringPropertySidecar::Equality)
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
    pub(crate) property_index_generation: Option<u64>,
    pub(crate) exact_node_counts: Option<Arc<crate::property_index::ExactNodeCountCell>>,
    pub(crate) memtable_claimant_cell: Option<Arc<crate::property_index::MemtableClaimantCell>>,
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
            snap = match self.property_index_generation {
                Some(generation) => {
                    snap.with_property_index_cache_generation(c.clone(), generation)
                }
                None => snap.with_property_index_cache(c.clone()),
            };
        }
        // `with_property_index_cache_generation` can no longer retrieve an
        // old generation after the writer advances. The OwnedSnapshot itself
        // pins the exact-count cell, just like it pins its manifest/memtable.
        snap.exact_node_counts = self.exact_node_counts.clone();
        snap.memtable_claimant_cell = self.memtable_claimant_cell.clone();
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

    #[test]
    fn snapshot_byte_cache_is_byte_and_count_bounded_lru() {
        let mut cache = SnapshotByteCache::new(700, 2);
        cache.insert("first".to_string(), 1_u64, 300);
        cache.insert("second".to_string(), 2_u64, 300);
        assert_eq!(cache.get(&"first".to_string()), Some(1));

        // Touching first makes second the bounded LRU victim.
        cache.insert("third".to_string(), 3_u64, 300);
        assert_eq!(cache.get(&"second".to_string()), None);
        assert_eq!(cache.get(&"first".to_string()), Some(1));
        assert_eq!(cache.get(&"third".to_string()), Some(3));
        assert_eq!(cache.len(), 2);
        assert!(cache.used_bytes() <= 700);

        // One entry above the whole assignment is used by the caller but never
        // retained, and overwrites cannot leak the old charge.
        cache.insert("oversized".to_string(), 4_u64, 701);
        assert_eq!(cache.get(&"oversized".to_string()), None);
        cache.insert("first".to_string(), 10_u64, 256);
        assert_eq!(cache.get(&"first".to_string()), Some(10));
        assert!(cache.used_bytes() <= 700);
    }

    #[cfg(any(feature = "vector-index", feature = "text-index"))]
    fn search_index_sst(
        path: &str,
        size_bytes: u64,
        row_count: u64,
        kind: SstKind,
        kind_specific: KindSpecificStats,
    ) -> SstDescriptor {
        SstDescriptor {
            id: Uuid::now_v7(),
            kind,
            scope: "search_idx".into(),
            level: crate::manifest::SstLevel(1),
            path: path.into(),
            size_bytes,
            row_count,
            created_at: chrono::Utc::now(),
            min_key: [0; 16],
            max_key: [0xFF; 16],
            min_lsn: 0,
            max_lsn: 0,
            schema_version_min: 0,
            schema_version_max: 0,
            property_stats: vec![],
            kind_specific,
            bloom: None,
            unique_property_indices: vec![],
            equality_property_indices: vec![],
            label_index: None,
            node_locator: None,
            per_label_property_stats: vec![],
        }
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

    #[test]
    fn optional_accelerator_fallback_classification_keeps_operational_errors_visible() {
        fn source() -> Box<dyn std::error::Error + Send + Sync> {
            Box::new(std::io::Error::other("test"))
        }

        for error in [
            Error::ObjectStore(object_store::Error::NotFound {
                path: "missing".into(),
                source: source(),
            }),
            Error::ObjectStore(object_store::Error::Precondition {
                path: "stale".into(),
                source: source(),
            }),
            Error::Invariant("optional decode".into()),
            Error::Corrupted {
                path: "optional".into(),
                detail: "checksum".into(),
            },
        ] {
            assert!(
                optional_accelerator_fallback(&error),
                "{error:?} must select the exact fallback"
            );
        }

        for error in [
            Error::ObjectStore(object_store::Error::Generic {
                store: "test",
                source: source(),
            }),
            Error::ObjectStore(object_store::Error::PermissionDenied {
                path: "denied".into(),
                source: source(),
            }),
            Error::ObjectStore(object_store::Error::Unauthenticated {
                path: "auth".into(),
                source: source(),
            }),
            Error::Timeout,
            Error::CacheCapacity {
                index_kind: "vector",
                path: "large.vg".into(),
                required_bytes: 2,
                capacity_bytes: 1,
            },
            Error::Precondition("storage/write precondition".into()),
        ] {
            assert!(
                !optional_accelerator_fallback(&error),
                "{error:?} must remain visible to the caller"
            );
        }
    }

    #[cfg(feature = "vector-index")]
    #[tokio::test]
    async fn valid_vector_index_over_configured_pool_errors_before_flat_fallback() {
        use crate::manifest::{VectorIndexDescriptor, VectorMetric, VectorQuantization};

        let store = make_store();
        let paths = make_paths("vector-cache-capacity");
        let ms = ManifestStore::new(store.clone(), paths.clone());
        let loaded = ms.bootstrap(Uuid::now_v7()).await.unwrap();
        let descriptor = VectorIndexDescriptor {
            name: "search_idx".into(),
            label: "Doc".into(),
            property: "embedding".into(),
            dim: 2,
            metric: VectorMetric::Cosine,
            r: 2,
            l_build: 4,
            alpha: 1.2,
            quantization: VectorQuantization::None,
        };
        let (body, stats) = crate::sst::vector::build_body(
            &descriptor,
            vec![([1; 16], vec![1.0, 0.0]), ([2; 16], vec![0.0, 1.0])],
        )
        .unwrap()
        .unwrap();
        let relative = "sst/level1/capacity.vg";
        let absolute = format!("{}/{}", paths.namespace_prefix().as_ref(), relative);
        store
            .put(&Path::from(absolute.clone()), body.clone().into())
            .await
            .unwrap();
        let sst = search_index_sst(
            relative,
            body.len() as u64,
            stats.point_count,
            SstKind::VectorGraph,
            KindSpecificStats::VectorGraph {
                dim: stats.dim,
                metric: stats.metric,
                point_count: stats.point_count,
                r: stats.r,
                l_build: stats.l_build,
                alpha: stats.alpha,
                entry_medoid: stats.entry_medoid,
            },
        );
        let memtable = Memtable::new();
        let view = memtable.snapshot_view();
        let cache = SstCache::with_uniform_budgets(512);
        let snapshot = Snapshot::new(loaded.clone(), &view, store.clone(), paths.clone())
            .with_cache(cache.clone());

        let error = snapshot.fetch_vector_index(&sst).await.unwrap_err();
        assert!(matches!(
            error,
            Error::CacheCapacity {
                index_kind: "vector",
                path,
                required_bytes,
                capacity_bytes: 512,
            } if path == absolute && required_bytes > 512
        ));
        assert_eq!(cache.vector_index_capacity_rejections(), 1);
        assert_eq!(cache.search_index_usage_bytes(), 0);

        // Admission runs before the body GET. Once a rollback/janitor removes
        // the optional object, the rejection branch must HEAD it and return
        // "index unavailable" rather than preserve a stale CacheCapacity error.
        store.delete(&Path::from(absolute)).await.unwrap();
        assert!(
            snapshot.fetch_vector_index(&sst).await.unwrap().is_none(),
            "a swept oversized .vg must select the exact fallback"
        );

        // Cover the ordinary cold-GET branch too (no decoded cache/preflight).
        let uncached = Snapshot::new(loaded, &view, store, paths);
        assert!(
            uncached.fetch_vector_index(&sst).await.unwrap().is_none(),
            "a swept .vg must be unavailable rather than a storage error"
        );
    }

    #[cfg(feature = "text-index")]
    #[tokio::test]
    async fn valid_text_index_over_configured_pool_errors_before_flat_fallback() {
        let store = make_store();
        let paths = make_paths("text-cache-capacity");
        let ms = ManifestStore::new(store.clone(), paths.clone());
        let loaded = ms.bootstrap(Uuid::now_v7()).await.unwrap();
        let (body, stats) =
            crate::sst::text::build_body(vec![([1; 16], "legal production corpus".into())])
                .unwrap()
                .unwrap();
        let relative = "sst/level1/capacity.ft";
        let absolute = format!("{}/{}", paths.namespace_prefix().as_ref(), relative);
        store
            .put(&Path::from(absolute.clone()), body.clone().into())
            .await
            .unwrap();
        let sst = search_index_sst(
            relative,
            body.len() as u64,
            stats.doc_count,
            SstKind::TextIndex,
            KindSpecificStats::TextIndex {
                doc_count: stats.doc_count,
                term_count: stats.term_count,
                total_len: stats.total_len,
            },
        );
        let memtable = Memtable::new();
        let view = memtable.snapshot_view();
        let cache = SstCache::with_uniform_budgets(512);
        let snapshot = Snapshot::new(loaded.clone(), &view, store.clone(), paths.clone())
            .with_cache(cache.clone());

        let error = snapshot.fetch_text_index(&sst).await.unwrap_err();
        assert!(matches!(
            error,
            Error::CacheCapacity {
                index_kind: "full-text",
                path,
                required_bytes,
                capacity_bytes: 512,
            } if path == absolute && required_bytes > 512
        ));
        assert_eq!(cache.text_index_capacity_rejections(), 1);
        assert_eq!(cache.search_index_usage_bytes(), 0);

        store.delete(&Path::from(absolute)).await.unwrap();
        assert!(
            snapshot.fetch_text_index(&sst).await.unwrap().is_none(),
            "a swept oversized .ft must select the exact fallback"
        );
        let uncached = Snapshot::new(loaded, &view, store, paths);
        assert!(
            uncached.fetch_text_index(&sst).await.unwrap().is_none(),
            "a swept .ft must be unavailable rather than a storage error"
        );
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
    async fn equality_point_apis_decline_deleted_or_corrupt_optional_sidecars() {
        async fn assert_unavailable(snapshot: &Snapshot<'_>, expected: &[NodeId]) {
            let value = Value::Str("LA".into());
            assert!(
                snapshot
                    .indexed_node_ids_by_property_value("Person", "city", &value)
                    .await
                    .unwrap()
                    .is_none(),
                "ordinary equality lookup must retain its exact fallback"
            );
            assert!(
                snapshot
                    .indexed_node_ids_by_property_value_limited("Person", "city", &value, 1)
                    .await
                    .unwrap()
                    .is_none(),
                "limited equality lookup must not return a partial posting"
            );
            assert!(
                snapshot
                    .indexed_node_ids_by_property_value_capped("Person", "city", &value, 8)
                    .await
                    .unwrap()
                    .is_none(),
                "vector eligibility must decline an unavailable sidecar"
            );

            let mut exact: Vec<NodeId> = snapshot
                .scan_label("Person")
                .await
                .unwrap()
                .into_iter()
                .filter(|node| node.properties.get("city") == Some(&value))
                .map(|node| node.id)
                .collect();
            exact.sort_unstable();
            assert_eq!(
                exact, expected,
                "the authoritative Parquet corpus remains readable"
            );
        }

        let store = make_store();
        let paths = make_paths("eqidx-optional-fallback");
        let ms = ManifestStore::new(store.clone(), paths.clone());
        let mut base = ms.bootstrap(Uuid::now_v7()).await.unwrap();
        base.manifest.label_dict.intern("Person");
        let fence = WriterFence::new(base.manifest.epoch);
        let schema = SchemaBuilder::new()
            .label(indexed_city_label())
            .unwrap()
            .build();
        let ann = sorted_node_id(1);
        let bob = sorted_node_id(2);
        let committed = flush_batch(
            &ms,
            &fence,
            &base,
            &schema,
            vec![
                (ann, 10, MemOp::Upsert(city_payload("Ann", "LA"))),
                (bob, 11, MemOp::Upsert(city_payload("Bob", "LA"))),
                (
                    sorted_node_id(3),
                    12,
                    MemOp::Upsert(city_payload("Cy", "NYC")),
                ),
            ],
        )
        .await;
        let equality = committed
            .manifest
            .ssts
            .iter()
            .find(|descriptor| descriptor.kind == SstKind::Nodes)
            .and_then(|descriptor| {
                descriptor
                    .equality_property_indices
                    .iter()
                    .find(|index| index.property == "city")
            })
            .cloned()
            .expect("flush builds the city equality sidecar");
        let absolute = |relative: &str| {
            Path::from(format!(
                "{}/{}",
                paths.namespace_prefix().as_ref(),
                relative
            ))
        };
        let base_path = absolute(&equality.path);
        let paged_path = equality
            .paged
            .as_ref()
            .map(|paged| absolute(&paged.path))
            .expect("current flush builds the paged mirror");
        store.delete(&paged_path).await.unwrap();
        store.delete(&base_path).await.unwrap();

        let empty = Memtable::new();
        let view = empty.snapshot_view();
        let missing = Snapshot::new(committed.clone(), &view, store.clone(), paths.clone());
        assert_unavailable(&missing, &[ann, bob]).await;

        // A malformed bincode body is the integrity-equivalent of absence.
        // The paged mirror remains missing, forcing the base decode branch.
        store
            .put(
                &base_path,
                Bytes::from_static(b"corrupt equality sidecar").into(),
            )
            .await
            .unwrap();
        let corrupt = Snapshot::new(committed, &view, store, paths);
        assert_unavailable(&corrupt, &[ann, bob]).await;
    }

    #[tokio::test]
    async fn label_scoped_property_lookups_scan_when_optional_sidecars_are_unavailable() {
        async fn assert_exact(snapshot: &Snapshot<'_>, ann: NodeId, bob: NodeId) {
            assert_eq!(
                snapshot
                    .lookup_node_by_property("Person", "name", "Ann")
                    .await
                    .unwrap()
                    .map(|node| node.id),
                Some(ann),
                "singular unique lookup must fall back to the authoritative label scan"
            );
            assert!(snapshot
                .lookup_node_by_property("Person", "name", "missing")
                .await
                .unwrap()
                .is_none());

            let values = vec![
                "Bob".to_string(),
                "missing".to_string(),
                "Ann".to_string(),
                "Bob".to_string(),
            ];
            assert_eq!(
                snapshot
                    .batch_lookup_nodes_by_property("Person", "name", &values)
                    .await
                    .unwrap()
                    .into_iter()
                    .map(|node| node.map(|node| node.id))
                    .collect::<Vec<_>>(),
                vec![Some(bob), None, Some(ann), Some(bob)],
                "batched unique lookup must preserve order, duplicates and misses"
            );

            let mut city_ids: Vec<NodeId> = snapshot
                .lookup_nodes_by_property("Person", "city", "LA")
                .await
                .unwrap()
                .into_iter()
                .map(|node| node.id)
                .collect();
            city_ids.sort_unstable();
            assert_eq!(
                city_ids,
                vec![ann, bob],
                "non-unique lookup must fall back without under-returning"
            );
        }

        let store = make_store();
        let paths = make_paths("label-scoped-sidecar-fallback");
        let ms = ManifestStore::new(store.clone(), paths.clone());
        let mut base = ms.bootstrap(Uuid::now_v7()).await.unwrap();
        base.manifest.label_dict.intern("Person");
        let fence = WriterFence::new(base.manifest.epoch);
        let schema = SchemaBuilder::new()
            .label(LabelDef {
                name: "Person".into(),
                properties: vec![
                    PropertyDef::new("name", DataType::Utf8, false)
                        .unwrap()
                        .with_unique(true),
                    PropertyDef::new("city", DataType::Utf8, true)
                        .unwrap()
                        .with_indexed(true),
                ],
            })
            .unwrap()
            .build();
        let ann = sorted_node_id(1);
        let bob = sorted_node_id(2);
        let committed = flush_batch(
            &ms,
            &fence,
            &base,
            &schema,
            vec![
                (ann, 10, MemOp::Upsert(city_payload("Ann", "LA"))),
                (bob, 11, MemOp::Upsert(city_payload("Bob", "LA"))),
            ],
        )
        .await;
        let descriptor = committed
            .manifest
            .ssts
            .iter()
            .find(|descriptor| descriptor.kind == SstKind::Nodes)
            .unwrap();
        let indices: Vec<EqualityIndexDescriptor> = ["name", "city"]
            .into_iter()
            .map(|property| {
                descriptor
                    .equality_property_indices
                    .iter()
                    .find(|index| index.property == property)
                    .cloned()
                    .unwrap()
            })
            .collect();
        let absolute = |relative: &str| {
            Path::from(format!(
                "{}/{}",
                paths.namespace_prefix().as_ref(),
                relative
            ))
        };

        // Simulate a rollback/old-reader sweep: both the paged mirror and the
        // legacy base object disappear while authoritative Parquet remains.
        for index in &indices {
            let paged = index
                .paged
                .as_ref()
                .expect("current flush writes a paged mirror");
            store.delete(&absolute(&paged.path)).await.unwrap();
            store.delete(&absolute(&index.path)).await.unwrap();
        }
        let empty = Memtable::new();
        let view = empty.snapshot_view();
        let missing = Snapshot::new(committed.clone(), &view, store.clone(), paths.clone());
        assert_exact(&missing, ann, bob).await;

        // A malformed base body is equally optional. Keep the paged mirrors
        // absent so every public path reaches the failed base decode.
        for index in &indices {
            store
                .put(
                    &absolute(&index.path),
                    Bytes::from_static(b"corrupt equality sidecar").into(),
                )
                .await
                .unwrap();
        }
        let corrupt = Snapshot::new(committed, &view, store, paths);
        assert_exact(&corrupt, ann, bob).await;
    }

    #[tokio::test]
    async fn global_prefixes_widen_for_sparse_requested_label() {
        fn id(n: u64) -> NodeId {
            let mut bytes = [0u8; 16];
            bytes[8..].copy_from_slice(&n.to_be_bytes());
            NodeId::from_uuid(Uuid::from_bytes(bytes))
        }
        fn payload(key: String, label_id: u32) -> Bytes {
            let mut properties = BTreeMap::new();
            properties.insert("key".into(), Value::Str(key));
            properties.insert("vigente".into(), Value::Bool(true));
            NodeWriteRecord {
                properties,
                schema_version: 1,
                labels: vec![label_id],
            }
            .encode()
            .unwrap()
        }
        fn label(name: &str) -> LabelDef {
            LabelDef {
                name: name.into(),
                properties: vec![
                    PropertyDef::new("key", DataType::Utf8, false)
                        .unwrap()
                        .with_indexed(true),
                    PropertyDef::new("vigente", DataType::Bool, false)
                        .unwrap()
                        .with_indexed(true),
                ],
            }
        }

        let store = make_store();
        let paths = make_paths("global-prefix-sparse-label");
        let ms = ManifestStore::new(store.clone(), paths.clone());
        let mut base = ms.bootstrap(Uuid::now_v7()).await.unwrap();
        let other_label = base.manifest.label_dict.intern("Other").0;
        let norma_label = base.manifest.label_dict.intern("Norma").0;
        let schema = SchemaBuilder::new()
            .label(label("Other"))
            .unwrap()
            .label(label("Norma"))
            .unwrap()
            .build();
        let fence = WriterFence::new(base.manifest.epoch);

        // Both key order and NodeId order put every Other candidate before
        // Norma. A fixed prefix of five would return zero requested-label
        // rows and force the old O(N) scan fallback.
        let mut rows = Vec::new();
        for n in 0..100u64 {
            rows.push((
                id(n + 1),
                n + 1,
                MemOp::Upsert(payload(format!("a-{n:04}"), other_label)),
            ));
        }
        for n in 0..10u64 {
            rows.push((
                id(1_000 + n),
                1_000 + n,
                MemOp::Upsert(payload(format!("z-{n:04}"), norma_label)),
            ));
        }
        let committed = flush_batch(&ms, &fence, &base, &schema, rows).await;
        let empty = Memtable::new();
        let view = empty.snapshot_view();
        let cache = Arc::new(crate::property_index::PropertyIndexCache::new());
        let snap =
            Snapshot::new(committed, &view, store, paths).with_property_index_cache(cache.clone());

        let ordered = snap
            .ordered_node_ids_by_string_property("Norma", "key", 5)
            .await
            .unwrap()
            .expect("geometric prefix must stay on the indexed fast path");
        assert_eq!(ordered, (0..5).map(|n| id(1_000 + n)).collect::<Vec<_>>());
        assert!(cache.ordered_prefix_widenings() > 0);

        let vigente = snap
            .indexed_node_ids_by_property_value_limited("Norma", "vigente", &Value::Bool(true), 5)
            .await
            .unwrap()
            .expect("Bool posting prefix must widen instead of full-scanning");
        assert_eq!(vigente, (0..5).map(|n| id(1_000 + n)).collect::<Vec<_>>());
        assert!(cache.equality_posting_widenings() > 0);
    }

    #[tokio::test]
    async fn limited_index_paths_bound_work_across_overlapping_writes_and_memtable() {
        fn id(n: u64) -> NodeId {
            NodeId::from_uuid(Uuid::from_bytes((n as u128).to_be_bytes()))
        }
        fn payload(key: String, vigente: bool, label_id: u32) -> Bytes {
            NodeWriteRecord {
                properties: BTreeMap::from([
                    ("key".into(), Value::Str(key)),
                    ("vigente".into(), Value::Bool(vigente)),
                ]),
                schema_version: 1,
                labels: vec![label_id],
            }
            .encode()
            .unwrap()
        }

        let store = make_store();
        let paths = make_paths("limited-index-overlap-memtable");
        let ms = ManifestStore::new(store.clone(), paths.clone());
        let mut base = ms.bootstrap(Uuid::now_v7()).await.unwrap();
        let norma = base.manifest.label_dict.intern("Norma").get();
        let schema = SchemaBuilder::new()
            .label(LabelDef {
                name: "Norma".into(),
                properties: vec![
                    PropertyDef::new("key", DataType::Utf8, false)
                        .unwrap()
                        .with_indexed(true),
                    PropertyDef::new("vigente", DataType::Bool, false)
                        .unwrap()
                        .with_indexed(true),
                ],
            })
            .unwrap()
            .build();
        let fence = WriterFence::new(base.manifest.epoch);

        let base_rows = (0..1_000u64)
            .map(|offset| {
                (
                    id(offset + 1),
                    offset + 1,
                    MemOp::Upsert(payload(format!("k-{offset:04}"), true, norma)),
                )
            })
            .collect();
        let first = flush_batch(&ms, &fence, &base, &schema, base_rows).await;

        // A truncated prefix that already supplies the true smallest live ids
        // must not widen merely because more ids exist after the LIMIT cutoff.
        let empty = Memtable::new();
        let empty_view = empty.snapshot_view();
        let live_prefix_cache = Arc::new(crate::property_index::PropertyIndexCache::new());
        let live_prefix_snap =
            Snapshot::new(first.clone(), &empty_view, store.clone(), paths.clone())
                .with_property_index_cache(live_prefix_cache.clone());
        assert_eq!(
            live_prefix_snap
                .indexed_node_ids_by_property_value_limited(
                    "Norma",
                    "vigente",
                    &Value::Bool(true),
                    5,
                )
                .await
                .unwrap(),
            Some((1..=5).map(id).collect())
        );
        assert_eq!(live_prefix_cache.equality_posting_widenings(), 0);

        // Obsolete the first five candidates in another overlapping L0 SST.
        let second_rows = (0..5u64)
            .map(|offset| {
                (
                    id(offset + 1),
                    2_000 + offset,
                    MemOp::Upsert(payload(format!("z-{offset:04}"), false, norma)),
                )
            })
            .collect();
        let committed = flush_batch(&ms, &fence, &first, &schema, second_rows).await;

        // Obsolete the next five candidates in the live memtable and add five
        // new smallest String keys. Both limited paths must merge this delta,
        // reject stale SST postings, and refill without scanning 1,000 rows.
        let mut memtable = Memtable::new();
        for offset in 5..10u64 {
            memtable.apply(
                MemKey::Node { id: id(offset + 1) },
                3_000 + offset,
                MemOp::Upsert(payload(format!("y-{offset:04}"), false, norma)),
            );
        }
        for offset in 0..5u64 {
            memtable.apply(
                MemKey::Node {
                    id: id(2_000 + offset),
                },
                4_000 + offset,
                MemOp::Upsert(payload(format!("a-{offset:04}"), true, norma)),
            );
        }
        let view = memtable.snapshot_view();
        let cache = Arc::new(crate::property_index::PropertyIndexCache::new());
        let snap =
            Snapshot::new(committed, &view, store, paths).with_property_index_cache(cache.clone());

        assert_eq!(
            snap.indexed_node_ids_by_property_value_limited(
                "Norma",
                "vigente",
                &Value::Bool(true),
                5,
            )
            .await
            .unwrap(),
            Some((11..=15).map(id).collect())
        );
        assert!(
            cache.equality_candidates_iterated() <= 32,
            "LIMIT 5 should confirm a bounded posting prefix, got {} candidates",
            cache.equality_candidates_iterated()
        );
        assert_eq!(
            cache.equality_candidates_iterated(),
            cache.equality_confirmation_candidates()
        );
        assert_eq!(
            cache.equality_posting_widenings(),
            1,
            "the stale five-id SST prefix should require exactly one geometric retry"
        );
        assert!(
            cache.equality_index_bytes_read() > 0,
            "the test must exercise range-readable posting I/O"
        );

        let mut expected = (2_000..2_005).map(id).collect::<Vec<_>>();
        expected.extend((11..=15).map(id));
        assert_eq!(
            snap.ordered_node_ids_by_string_property("Norma", "key", 10)
                .await
                .unwrap(),
            Some(expected)
        );
        assert!(
            cache.ordered_prefix_candidates_iterated() <= 96,
            "SKIP/LIMIT prefix should hydrate bounded candidates, got {}",
            cache.ordered_prefix_candidates_iterated()
        );
        assert_eq!(
            cache.ordered_prefix_candidates_iterated(),
            cache.ordered_prefix_confirmation_candidates()
        );
        assert!(
            cache.ordered_prefix_index_bytes_read() > 0,
            "the ordered path must use range-readable sidecar prefixes"
        );
        assert_eq!(cache.memtable_population_scans(), 2);
        assert_eq!(
            cache.memtable_population_rows(),
            20,
            "each indexed property scans only the ten-row write delta once"
        );
    }

    #[tokio::test]
    async fn ordered_prefix_widens_when_one_sst_hides_a_smaller_label_value() {
        fn id(n: u64) -> NodeId {
            NodeId::from_uuid(Uuid::from_bytes((n as u128).to_be_bytes()))
        }
        fn payload(key: &str, label_id: u32) -> Bytes {
            NodeWriteRecord {
                properties: BTreeMap::from([("key".into(), Value::Str(key.into()))]),
                schema_version: 1,
                labels: vec![label_id],
            }
            .encode()
            .unwrap()
        }
        fn label(name: &str) -> LabelDef {
            LabelDef {
                name: name.into(),
                properties: vec![PropertyDef::new("key", DataType::Utf8, false)
                    .unwrap()
                    .with_indexed(true)],
            }
        }

        let store = make_store();
        let paths = make_paths("ordered-prefix-multi-sst-frontier");
        let ms = ManifestStore::new(store.clone(), paths.clone());
        let mut base = ms.bootstrap(Uuid::now_v7()).await.unwrap();
        let other = base.manifest.label_dict.intern("Other").0;
        let norma = base.manifest.label_dict.intern("Norma").0;
        let schema = SchemaBuilder::new()
            .label(label("Other"))
            .unwrap()
            .label(label("Norma"))
            .unwrap()
            .build();
        let fence = WriterFence::new(base.manifest.epoch);

        // SST A's first K global keys belong to Other. Its next unread key is
        // the true smallest Norma value. SST B alone can fill K with z*, so an
        // unconditional return at out.len()==K would silently misorder.
        let mut a_rows = (0..5u64)
            .map(|n| {
                (
                    id(n + 1),
                    n + 1,
                    MemOp::Upsert(payload(&format!("a{n}"), other)),
                )
            })
            .collect::<Vec<_>>();
        a_rows.push((id(6), 6, MemOp::Upsert(payload("a5", norma))));
        let first = flush_batch(&ms, &fence, &base, &schema, a_rows).await;
        let b_rows = (0..5u64)
            .map(|n| {
                (
                    id(100 + n),
                    100 + n,
                    MemOp::Upsert(payload(&format!("z{n}"), norma)),
                )
            })
            .collect();
        let committed = flush_batch(&ms, &fence, &first, &schema, b_rows).await;

        let empty = Memtable::new();
        let view = empty.snapshot_view();
        let cache = Arc::new(crate::property_index::PropertyIndexCache::new());
        let snap =
            Snapshot::new(committed, &view, store, paths).with_property_index_cache(cache.clone());
        assert_eq!(
            snap.ordered_node_ids_by_string_property("Norma", "key", 5)
                .await
                .unwrap(),
            Some(vec![id(6), id(100), id(101), id(102), id(103)])
        );
        assert!(
            cache.ordered_prefix_widenings() > 0,
            "discarded candidates under a truncated SST require widening"
        );
    }

    #[tokio::test]
    async fn ordered_prefix_exact_frontier_avoids_unneeded_multi_sst_widening() {
        fn id(n: u64) -> NodeId {
            NodeId::from_uuid(Uuid::from_bytes((n as u128).to_be_bytes()))
        }
        fn payload(key: &str, label_id: u32) -> Bytes {
            NodeWriteRecord {
                properties: BTreeMap::from([("key".into(), Value::Str(key.into()))]),
                schema_version: 1,
                labels: vec![label_id],
            }
            .encode()
            .unwrap()
        }
        fn label(name: &str) -> LabelDef {
            LabelDef {
                name: name.into(),
                properties: vec![PropertyDef::new("key", DataType::Utf8, false)
                    .unwrap()
                    .with_indexed(true)],
            }
        }

        let store = make_store();
        let paths = make_paths("ordered-prefix-exact-safe-frontier");
        let ms = ManifestStore::new(store.clone(), paths.clone());
        let mut base = ms.bootstrap(Uuid::now_v7()).await.unwrap();
        let other = base.manifest.label_dict.intern("Other").get();
        let norma = base.manifest.label_dict.intern("Norma").get();
        let schema = SchemaBuilder::new()
            .label(label("Other"))
            .unwrap()
            .label(label("Norma"))
            .unwrap()
            .build();
        let fence = WriterFence::new(base.manifest.epoch);

        // SST A's first global candidate sorts before Norma but belongs to
        // Other; the remaining visible/unread A frontier is z0. SST B supplies
        // the exact requested b0..b4 prefix. The old aggregate
        // `truncated && discarded` rule widened A repeatedly even though z0
        // proves that none of its hidden tuples can precede cutoff b4.
        let mut a_rows = vec![(id(1), 1, MemOp::Upsert(payload("a0", other)))];
        a_rows.extend((0..12u64).map(|n| {
            (
                id(n + 2),
                n + 2,
                MemOp::Upsert(payload(&format!("z{n:02}"), other)),
            )
        }));
        let first = flush_batch(&ms, &fence, &base, &schema, a_rows).await;
        let b_rows = (0..5u64)
            .map(|n| {
                (
                    id(100 + n),
                    100 + n,
                    MemOp::Upsert(payload(&format!("b{n}"), norma)),
                )
            })
            .collect();
        let committed = flush_batch(&ms, &fence, &first, &schema, b_rows).await;
        let paged_key_sidecars: Vec<_> = committed
            .manifest
            .ssts
            .iter()
            .filter(|descriptor| descriptor.kind == SstKind::Nodes)
            .flat_map(|descriptor| &descriptor.equality_property_indices)
            .filter(|index| index.property == "key")
            .collect();
        assert_eq!(paged_key_sidecars.len(), 2);
        assert!(
            paged_key_sidecars.iter().all(|index| index.paged.is_some()),
            "the regression must exercise range-truncated paged sidecars"
        );

        let empty = Memtable::new();
        let view = empty.snapshot_view();
        let cache = Arc::new(crate::property_index::PropertyIndexCache::new());
        let snap =
            Snapshot::new(committed, &view, store, paths).with_property_index_cache(cache.clone());
        assert_eq!(
            snap.ordered_node_ids_by_string_property("Norma", "key", 5)
                .await
                .unwrap(),
            Some((0..5).map(|n| id(100 + n)).collect())
        );
        assert_eq!(
            cache.ordered_prefix_widenings(),
            0,
            "a source frontier after the global cutoff must not widen"
        );
    }

    #[test]
    fn ordered_prefix_initial_capacity_is_bounded_for_extreme_skip_limit() {
        assert_eq!(ordered_prefix_initial_capacity(0), 0);
        assert_eq!(ordered_prefix_initial_capacity(17), 17);
        assert_eq!(
            ordered_prefix_initial_capacity(usize::MAX),
            ORDERED_PREFIX_INITIAL_CAPACITY
        );
    }

    #[test]
    fn ordered_cursor_skips_empty_posting_lists_and_keeps_exact_frontier() {
        let first = sorted_node_id(1);
        let map = Arc::new(BTreeMap::from([
            ("a-empty".into(), Vec::new()),
            ("b-live".into(), vec![*first.as_bytes()]),
            ("c-empty".into(), Vec::new()),
        ]));
        let mut cursor = OrderedStringPostingCursor::new_equality(
            map,
            crate::manifest::EqualityKeyEncoding::ScalarV1,
            true,
        )
        .expect("a later non-empty posting must remain visible");
        assert_eq!(cursor.current(), Some(("b-live", first)));
        cursor.advance();
        assert!(cursor.current().is_none());
        assert!(
            !cursor.unread_may_precede(&("b-live".into(), first)),
            "the hidden suffix is strictly greater than an equal cutoff"
        );
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
    async fn exact_edge_lookup_reconciles_dense_sst_memtable_and_overlay() {
        let store = make_store();
        let paths = make_paths("read-edge-point");
        let ms = ManifestStore::new(store.clone(), paths.clone());
        let base = ms.bootstrap(Uuid::now_v7()).await.unwrap();
        let fence = WriterFence::new(base.manifest.epoch);
        let schema = SchemaBuilder::new()
            .label(person_label())
            .unwrap()
            .edge_type(knows_edge())
            .unwrap()
            .build();

        let src = NodeId::from_uuid(Uuid::from_bytes([0x80; 16]));
        let partners: Vec<NodeId> = (0..2048u128)
            .map(|i| NodeId::from_uuid(Uuid::from_bytes((i * 2).to_be_bytes())))
            .collect();
        let target = partners[1536];
        let mut flushed = Memtable::new();
        for (i, dst) in partners.iter().enumerate() {
            let properties = if *dst == target {
                BTreeMap::from([("code".into(), Value::Str("persisted".into()))])
            } else {
                BTreeMap::new()
            };
            flushed.apply(
                MemKey::Edge {
                    edge_type: "KNOWS".into(),
                    src,
                    dst: *dst,
                },
                i as u64 + 10,
                MemOp::Upsert(
                    EdgeWriteRecord {
                        properties,
                        schema_version: 1,
                    }
                    .encode()
                    .unwrap(),
                ),
            );
        }
        let outcome = flush(&ms, &fence, &base, &flushed.freeze(), schema)
            .await
            .unwrap();
        let forward = outcome
            .committed
            .manifest
            .ssts
            .iter()
            .find(|descriptor| descriptor.kind == SstKind::EdgesFwd)
            .expect("flush writes a forward edge SST");
        assert!(
            forward.path.ends_with(".ep.csr"),
            "new forward SSTs advertise a complete exact-edge accelerator"
        );
        let point_relative = crate::manifest::edge_point_sidecar_path(forward)
            .expect("point marker derives its sidecar");
        let point_absolute = Path::from(format!(
            "{}/{}",
            paths.namespace_prefix().as_ref(),
            point_relative
        ));
        let point_body = store
            .get(&point_absolute)
            .await
            .unwrap()
            .bytes()
            .await
            .unwrap();

        let live = Memtable::new();
        let live_view = live.snapshot_view();
        // The decoded CSR and stream budgets are intentionally too small for
        // this dense bucket. A current point lookup must remain cold-cache
        // sparse and never try to admit either graph-sized structure.
        let point_cache = SstCache::with_uniform_budgets(1);
        let persisted = Snapshot::new(
            outcome.committed.clone(),
            &live_view,
            store.clone(),
            paths.clone(),
        )
        .with_cache(point_cache.clone());
        let edge = persisted
            .lookup_edge_via_sst("KNOWS", src, target)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(
            edge.properties.get("code"),
            Some(&Value::Str("persisted".into()))
        );
        assert!(persisted
            .contains_edge_via_sst("KNOWS", src, target)
            .await
            .unwrap());
        let absent = NodeId::from_uuid(Uuid::from_bytes((u128::MAX - 1).to_be_bytes()));
        assert!(!persisted
            .contains_edge_via_sst("KNOWS", src, absent)
            .await
            .unwrap());
        assert_eq!(point_cache.edge_readers_inserts(), 0);
        assert_eq!(point_cache.edge_streams_inserts(), 0);
        assert_eq!(point_cache.edge_point_probes(), 3);
        assert!(
            point_cache.edge_point_bytes() < forward.size_bytes,
            "cold exact probes should read less than the dense CSR body"
        );
        let past = std::time::Instant::now() - std::time::Duration::from_secs(1);
        let timed_out = crate::cancel::with_deadline(
            Some(past),
            persisted.contains_edge_via_sst("KNOWS", src, target),
        )
        .await;
        assert!(matches!(timed_out, Err(Error::Timeout)));
        assert_eq!(
            point_cache.edge_readers_inserts(),
            0,
            "a typed timeout must propagate instead of opening the CSR fallback"
        );

        // A corrupt optional value/index must never turn into a false miss.
        // Replace the sidecar with a full-length corrupt body so the ranged
        // read reaches checksum/header validation, then require one batched
        // CSR fallback for hits, misses and duplicate endpoint pairs.
        let mut corrupt = point_body.to_vec();
        corrupt[0] ^= 0xff;
        store
            .put(&point_absolute, Bytes::from(corrupt).into())
            .await
            .unwrap();
        let fallback_cache = SstCache::with_uniform_budgets(1);
        let fallback = Snapshot::new(
            outcome.committed.clone(),
            &live_view,
            store.clone(),
            paths.clone(),
        )
        .with_cache(fallback_cache.clone());
        let batch = fallback
            .batch_lookup_edges_via_sst("KNOWS", &[(src, target), (src, absent), (src, target)])
            .await
            .unwrap();
        assert_eq!(batch.len(), 3);
        assert_eq!(
            batch[0]
                .as_ref()
                .and_then(|edge| edge.properties.get("code")),
            Some(&Value::Str("persisted".into()))
        );
        assert!(batch[1].is_none());
        assert_eq!(batch[0], batch[2]);
        assert_eq!(
            fallback_cache.edge_readers_inserts(),
            0,
            "the corrupt-sidecar fallback hydrates through paged property \
             rows now; the eager whole-body reader cache must stay untouched"
        );
        assert_eq!(fallback_cache.edge_streams_inserts(), 0);

        // A downgrade-era janitor may remove an unrecognised `.epidx`. The
        // marker remains safe: NotFound falls through to the authoritative CSR
        // and still answers exactly.
        store.delete(&point_absolute).await.unwrap();
        let missing_cache = SstCache::with_uniform_budgets(1);
        let missing = Snapshot::new(
            outcome.committed.clone(),
            &live_view,
            store.clone(),
            paths.clone(),
        )
        .with_cache(missing_cache.clone());
        assert!(missing
            .contains_edge_via_sst("KNOWS", src, target)
            .await
            .unwrap());
        assert_eq!(
            missing_cache.edge_readers_inserts(),
            0,
            "an existence probe resolves the CSR through ranged reads: a \
             missing accelerator must never hydrate the whole edge body"
        );

        // A staged tombstone must hide both the persisted row and any
        // committed-memtable version of the same physical relationship.
        let mut committed = Memtable::new();
        committed.apply(
            MemKey::Edge {
                edge_type: "KNOWS".into(),
                src,
                dst: target,
            },
            10_000,
            MemOp::Upsert(
                EdgeWriteRecord {
                    properties: BTreeMap::from([("code".into(), Value::Str("committed".into()))]),
                    schema_version: 1,
                }
                .encode()
                .unwrap(),
            ),
        );
        let committed_view = committed.snapshot_view();
        let mut staged = Memtable::new();
        staged.apply(
            MemKey::Edge {
                edge_type: "KNOWS".into(),
                src,
                dst: target,
            },
            10_001,
            MemOp::Tombstone,
        );
        let hidden = Snapshot::new(outcome.committed, &committed_view, store, paths)
            .with_overlay(staged.snapshot_view());
        assert!(hidden
            .lookup_edge_via_sst("KNOWS", src, target)
            .await
            .unwrap()
            .is_none());
    }

    #[tokio::test]
    async fn exact_edge_lookup_reconciles_overlapping_ssts_before_decoding_properties() {
        let store = make_store();
        let paths = make_paths("read-edge-point-lww");
        let ms = ManifestStore::new(store.clone(), paths.clone());
        let base = ms.bootstrap(Uuid::now_v7()).await.unwrap();
        let fence = WriterFence::new(base.manifest.epoch);
        let schema = SchemaBuilder::new()
            .label(person_label())
            .unwrap()
            .edge_type(knows_edge())
            .unwrap()
            .build();
        let src = sorted_node_id(1);
        let dst = sorted_node_id(2);
        let key = || MemKey::Edge {
            edge_type: "KNOWS".into(),
            src,
            dst,
        };
        let payload = |code: &str| {
            EdgeWriteRecord {
                properties: BTreeMap::from([("code".into(), Value::Str(code.into()))]),
                schema_version: 1,
            }
            .encode()
            .unwrap()
        };

        let mut old = Memtable::new();
        old.apply(key(), 10, MemOp::Upsert(payload("old-loser")));
        let old_frozen = old.freeze();
        let v1 = flush(&ms, &fence, &base, &old_frozen, schema.clone())
            .await
            .unwrap();

        let mut deleted = Memtable::new();
        deleted.apply(key(), 20, MemOp::Tombstone);
        let deleted_frozen = deleted.freeze();
        let v2 = flush(&ms, &fence, &v1.committed, &deleted_frozen, schema.clone())
            .await
            .unwrap();
        let empty = Memtable::new();
        let empty_view = empty.snapshot_view();
        let deleted_snap = Snapshot::new(
            v2.committed.clone(),
            &empty_view,
            store.clone(),
            paths.clone(),
        );
        assert!(deleted_snap
            .lookup_edge_via_sst("KNOWS", src, dst)
            .await
            .unwrap()
            .is_none());

        let mut resurrected = Memtable::new();
        resurrected.apply(key(), 30, MemOp::Upsert(payload("new-winner")));
        let resurrected_frozen = resurrected.freeze();
        let v3 = flush(&ms, &fence, &v2.committed, &resurrected_frozen, schema)
            .await
            .unwrap();

        let cache = SstCache::new(16 * 1024 * 1024);
        let snap = Snapshot::new(v3.committed, &empty_view, store, paths).with_cache(cache.clone());
        let edge = snap
            .lookup_edge_via_sst("KNOWS", src, dst)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(edge.lsn, 30);
        assert_eq!(
            edge.properties.get("code"),
            Some(&Value::Str("new-winner".into()))
        );
        assert_eq!(
            cache.edge_readers_inserts(),
            0,
            "the exact point sidecar should not open any overlapping CSR"
        );
        assert_eq!(
            cache.edge_streams_inserts(),
            0,
            "the exact point value carries the winner's bounded property map"
        );
        assert_eq!(
            cache.edge_point_probes(),
            1,
            "the newest exact winner should prune every older overlapping SST"
        );
        assert!(snap.contains_edge_via_sst("KNOWS", src, dst).await.unwrap());
        assert_eq!(
            cache.edge_streams_inserts(),
            0,
            "an existence probe must not touch property streams"
        );
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

    /// The metadata fast path (single compacted SST + memtable delta) and
    /// the multi-SST merge fallback must agree on the same logical graph.
    /// The overlay exercises every delta class: a new live edge (+1), a
    /// tombstone of a flushed edge (-1), a re-upsert of a flushed edge (0)
    /// and a tombstone of a never-flushed edge (0).
    #[tokio::test]
    async fn count_edge_type_fast_path_matches_the_multi_sst_merge() {
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
        let eve = sorted_node_id(5);

        let flushed_edges: [(NodeId, u64); 3] = [(bob, 10), (dave, 11), (eve, 12)];
        let mut counts = Vec::new();
        for split in [false, true] {
            let store = make_store();
            let paths = make_paths(if split {
                "read-edges-count-split"
            } else {
                "read-edges-count-single"
            });
            let ms = ManifestStore::new(store.clone(), paths.clone());
            let mut current = ms.bootstrap(Uuid::now_v7()).await.unwrap();
            let fence = WriterFence::new(current.manifest.epoch);
            let chunks: Vec<&[(NodeId, u64)]> = if split {
                vec![&flushed_edges[..1], &flushed_edges[1..]]
            } else {
                vec![&flushed_edges[..]]
            };
            for chunk in chunks {
                let mut mt = Memtable::new();
                for (dst, lsn) in chunk {
                    mt.apply(
                        MemKey::Edge {
                            edge_type: "KNOWS".into(),
                            src: alice,
                            dst: *dst,
                        },
                        *lsn,
                        MemOp::Upsert(edge_payload()),
                    );
                }
                current = flush(&ms, &fence, &current, &mt.freeze(), schema.clone())
                    .await
                    .unwrap()
                    .committed;
            }
            let sst_count = current
                .manifest
                .ssts
                .iter()
                .filter(|d| d.kind == SstKind::EdgesFwd)
                .count();
            assert_eq!(sst_count, if split { 2 } else { 1 });

            let mut live = Memtable::new();
            // +1: brand-new live edge.
            live.apply(
                MemKey::Edge {
                    edge_type: "KNOWS".into(),
                    src: alice,
                    dst: carol,
                },
                20,
                MemOp::Upsert(edge_payload()),
            );
            // -1: tombstone of a flushed edge.
            live.apply(
                MemKey::Edge {
                    edge_type: "KNOWS".into(),
                    src: alice,
                    dst: bob,
                },
                21,
                MemOp::Tombstone,
            );
            // 0: re-upsert of a flushed live edge.
            live.apply(
                MemKey::Edge {
                    edge_type: "KNOWS".into(),
                    src: alice,
                    dst: dave,
                },
                22,
                MemOp::Upsert(edge_payload()),
            );
            // 0: tombstone of an edge that never existed.
            live.apply(
                MemKey::Edge {
                    edge_type: "KNOWS".into(),
                    src: bob,
                    dst: carol,
                },
                23,
                MemOp::Tombstone,
            );
            let view = live.snapshot_view();
            let snapshot = Snapshot::new(current.clone(), &view, store.clone(), paths.clone());
            let count = snapshot.count_edge_type("KNOWS").await.unwrap();
            let scanned = snapshot.scan_edge_type("KNOWS").await.unwrap();
            assert_eq!(
                count,
                scanned.len() as u64,
                "count must equal the merged scan (split={split})"
            );
            counts.push(count);
        }
        assert_eq!(counts[0], counts[1], "fast path and fallback must agree");
        // eve + dave (re-upsert) + carol survive; bob tombstoned: 3 live.
        assert_eq!(counts[0], 3);
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
            node_locator: None,
            per_label_property_stats: Vec::new(),
        };

        let empty = Memtable::new();
        let empty_view = empty.snapshot_view();
        let cache = SstCache::new(1 << 20);
        let snap = Snapshot::new(base, &empty_view, store.clone(), paths).with_cache(cache.clone());
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
        assert_eq!(cache.bloom_inserts(), 1, "bloom decoded only once");
        assert_eq!(cache.bloom_misses(), 1);
        assert_eq!(cache.bloom_hits(), 1);

        // Sanity: an SstDescriptor with `bloom = None` admits everything.
        let no_bloom = SstDescriptor {
            bloom: None,
            unique_property_indices: Vec::new(),
            equality_property_indices: Vec::new(),
            label_index: None,
            node_locator: None,
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
            node_locator: None,
            per_label_property_stats: Vec::new(),
        });
        base = LoadedManifest::new(
            base.pointer,
            base.pointer_etag,
            base.pointer_version,
            base.manifest,
        );
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
    async fn multi_row_group_fixture_mode(
        store: &Arc<dyn ObjectStore>,
        paths: &NamespacePaths,
        n: u8,
        rows_per_group: usize,
        keep_locator: bool,
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
        let mut committed =
            flush_batch_with_row_group_rows(rows_per_group, &ms, &fence, &base, &schema, rows)
                .await;
        let desc = committed
            .manifest
            .ssts
            .iter_mut()
            .find(|d| matches!(d.kind, SstKind::Nodes))
            .expect("flush produced a node SST");
        let absolute = format!("{}/{}", paths.namespace_prefix().as_ref(), desc.path);
        if !keep_locator {
            // Legacy-path fixtures intentionally exercise row-group pruning
            // for pre-locator SSTs.
            desc.node_locator = None;
        }
        (committed, absolute)
    }

    async fn multi_row_group_fixture(
        store: &Arc<dyn ObjectStore>,
        paths: &NamespacePaths,
        n: u8,
        rows_per_group: usize,
    ) -> (LoadedManifest, String) {
        multi_row_group_fixture_mode(store, paths, n, rows_per_group, false).await
    }

    fn limited_doc_label() -> LabelDef {
        LabelDef {
            name: "Doc".into(),
            properties: vec![
                PropertyDef::new("key", DataType::Utf8, false).unwrap(),
                PropertyDef::new("vigente", DataType::Bool, true).unwrap(),
                PropertyDef::new("blob", DataType::Utf8, true).unwrap(),
            ],
        }
    }

    fn deterministic_blob(seed: u64, len: usize) -> String {
        let mut state = seed | 1;
        let mut out = String::with_capacity(len);
        const ALPHABET: &[u8] = b"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_";
        for _ in 0..len {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            out.push(ALPHABET[(state as usize) & 63] as char);
        }
        out
    }

    fn limited_doc_payload(index: u8, vigente: bool, wide: bool) -> Bytes {
        let mut properties = BTreeMap::from([
            ("key".into(), Value::Str(format!("doc-{index:03}"))),
            ("vigente".into(), Value::Bool(vigente)),
        ]);
        if wide {
            properties.insert(
                "blob".into(),
                Value::Str(deterministic_blob(index as u64 + 1, 32 * 1024)),
            );
        }
        NodeWriteRecord {
            properties,
            schema_version: 1,
            labels: vec![0],
        }
        .encode()
        .unwrap()
    }

    async fn limited_doc_fixture(
        store: &Arc<dyn ObjectStore>,
        paths: &NamespacePaths,
        count: u8,
        rows_per_group: usize,
        wide: bool,
    ) -> (
        ManifestStore,
        WriterFence,
        namidb_core::Schema,
        LoadedManifest,
    ) {
        let ms = ManifestStore::new(store.clone(), paths.clone());
        let mut base = ms.bootstrap(Uuid::now_v7()).await.unwrap();
        base.manifest.label_dict.intern("Doc");
        let fence = WriterFence::new(base.manifest.epoch);
        let schema = SchemaBuilder::new()
            .label(limited_doc_label())
            .unwrap()
            .build();
        let rows = (1..=count)
            .map(|index| {
                let op = if index == 5 {
                    MemOp::Tombstone
                } else {
                    MemOp::Upsert(limited_doc_payload(index, index % 2 == 0, wide))
                };
                (sorted_node_id(index), index as u64, op)
            })
            .collect();
        let committed =
            flush_batch_with_row_group_rows(rows_per_group, &ms, &fence, &base, &schema, rows)
                .await;
        (ms, fence, schema, committed)
    }

    fn vigente_predicate() -> ScanPredicate {
        ScanPredicate::Eq {
            column: "vigente".into(),
            value: crate::sst::stats::StatScalar::Bool(true),
        }
    }

    fn object_native_doc_label(embedding_dim: u32) -> LabelDef {
        LabelDef {
            name: "Doc".into(),
            properties: vec![
                PropertyDef::new("title", DataType::Utf8, false).unwrap(),
                PropertyDef::new(
                    "embedding",
                    DataType::FloatVector { dim: embedding_dim },
                    true,
                )
                .unwrap(),
            ],
        }
    }

    fn object_native_doc_payload(index: u8, embedding_dim: usize) -> Bytes {
        let mut state = u64::from(index).saturating_add(1);
        let embedding = (0..embedding_dim)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                (state as u32) as f32 / u32::MAX as f32
            })
            .collect();
        NodeWriteRecord {
            properties: BTreeMap::from([
                ("title".into(), Value::Str(format!("title-{index:03}"))),
                ("embedding".into(), Value::Vec(embedding)),
            ]),
            schema_version: 1,
            labels: vec![0],
        }
        .encode()
        .unwrap()
    }

    async fn object_native_doc_fixture(
        store: &Arc<dyn ObjectStore>,
        paths: &NamespacePaths,
        count: u8,
        embedding_dim: usize,
    ) -> LoadedManifest {
        let ms = ManifestStore::new(store.clone(), paths.clone());
        let mut base = ms.bootstrap(Uuid::now_v7()).await.unwrap();
        base.manifest.label_dict.intern("Doc");
        let fence = WriterFence::new(base.manifest.epoch);
        let schema = SchemaBuilder::new()
            .label(object_native_doc_label(embedding_dim as u32))
            .unwrap()
            .build();
        let rows = (1..=count)
            .map(|index| {
                (
                    sorted_node_id(index),
                    u64::from(index),
                    MemOp::Upsert(object_native_doc_payload(index, embedding_dim)),
                )
            })
            .collect();
        flush_batch_with_row_group_rows(4, &ms, &fence, &base, &schema, rows).await
    }

    async fn projected_titles(
        loaded: LoadedManifest,
        memtable: &MemtableSnapshot,
        store: Arc<dyn ObjectStore>,
        paths: NamespacePaths,
    ) -> Vec<Value> {
        Snapshot::new(loaded, memtable, store, paths)
            .scan_label_with_predicates_and_projection("Doc", &[], Some(&["title".into()]))
            .await
            .unwrap()
            .into_iter()
            .map(|row| row.properties["title"].clone())
            .collect()
    }

    #[tokio::test]
    async fn snapshot_title_projection_never_fetches_embedding_pages() {
        let store = make_store();
        let paths = make_paths("node-property-title-not-embedding");
        let committed = object_native_doc_fixture(&store, &paths, 16, 16_384).await;
        let descriptor = committed
            .manifest
            .ssts
            .iter()
            .find(|descriptor| descriptor.kind == SstKind::Nodes)
            .unwrap()
            .clone();
        let property_descriptor = crate::manifest::node_property_pages_sidecar(&descriptor)
            .unwrap()
            .clone();
        let empty = Memtable::new();
        let empty_view = empty.snapshot_view();

        let title_snapshot =
            Snapshot::new(committed.clone(), &empty_view, store.clone(), paths.clone());
        let titles = title_snapshot
            .scan_label_with_predicates_and_projection("Doc", &[], Some(&["title".into()]))
            .await
            .unwrap();
        assert_eq!(titles.len(), 16);
        assert!(titles
            .iter()
            .all(|row| { row.properties.len() == 1 && row.properties.contains_key("title") }));
        let title_reader = title_snapshot
            .node_property_reader(&descriptor)
            .await
            .unwrap()
            .unwrap();
        let title_io = title_reader.range_stats();
        assert!(
            title_io.logical_bytes < property_descriptor.size_bytes,
            "title projection read the complete property object"
        );

        let embedding_snapshot = Snapshot::new(committed, &empty_view, store, paths);
        let embeddings = embedding_snapshot
            .scan_label_with_predicates_and_projection("Doc", &[], Some(&["embedding".into()]))
            .await
            .unwrap();
        assert_eq!(embeddings.len(), 16);
        let embedding_reader = embedding_snapshot
            .node_property_reader(&descriptor)
            .await
            .unwrap()
            .unwrap();
        let embedding_io = embedding_reader.range_stats();
        assert!(
            embedding_io.logical_bytes > title_io.logical_bytes.saturating_mul(4),
            "title bytes={} embedding bytes={}; title likely touched embedding pages",
            title_io.logical_bytes,
            embedding_io.logical_bytes
        );
    }

    #[tokio::test]
    async fn missing_corrupt_and_incompatible_property_pages_fall_back_exactly() {
        let store = make_store();
        let paths = make_paths("node-property-fallback");
        let committed = object_native_doc_fixture(&store, &paths, 8, 256).await;
        let descriptor = committed
            .manifest
            .ssts
            .iter()
            .find(|descriptor| descriptor.kind == SstKind::Nodes)
            .unwrap();
        let properties = crate::manifest::node_property_pages_sidecar(descriptor).unwrap();
        let property_path = Path::from(format!(
            "{}/{}",
            paths.namespace_prefix().as_ref(),
            properties.path
        ));
        let original = store
            .get(&property_path)
            .await
            .unwrap()
            .bytes()
            .await
            .unwrap();
        let empty = Memtable::new();
        let empty_view = empty.snapshot_view();
        let expected = (1..=8)
            .map(|index| Value::Str(format!("title-{index:03}")))
            .collect::<Vec<_>>();
        store.delete(&property_path).await.unwrap();
        assert_eq!(
            projected_titles(committed.clone(), &empty_view, store.clone(), paths.clone(),).await,
            expected
        );
        store
            .put(&property_path, original.clone().into())
            .await
            .unwrap();

        let mut corrupt = original.to_vec();
        corrupt[0] ^= 0xFF;
        store
            .put(&property_path, Bytes::from(corrupt).into())
            .await
            .unwrap();
        assert_eq!(
            projected_titles(committed.clone(), &empty_view, store.clone(), paths.clone(),).await,
            expected
        );
        store.put(&property_path, original.into()).await.unwrap();

        let mut incompatible = committed.clone();
        incompatible
            .manifest
            .ssts
            .iter_mut()
            .find(|descriptor| descriptor.kind == SstKind::Nodes)
            .unwrap()
            .node_locator
            .as_mut()
            .unwrap()
            .property_pages
            .as_mut()
            .unwrap()
            .format_version += 1;
        let incompatible = LoadedManifest::new(
            incompatible.pointer,
            incompatible.pointer_etag,
            incompatible.pointer_version,
            incompatible.manifest,
        );
        assert_eq!(
            projected_titles(incompatible, &empty_view, store.clone(), paths.clone(),).await,
            expected
        );

        let full_with_pages =
            Snapshot::new(committed.clone(), &empty_view, store.clone(), paths.clone())
                .scan_label("Doc")
                .await
                .unwrap();
        let mut no_pages = committed.clone();
        no_pages
            .manifest
            .ssts
            .iter_mut()
            .find(|descriptor| descriptor.kind == SstKind::Nodes)
            .unwrap()
            .node_locator
            .as_mut()
            .unwrap()
            .property_pages = None;
        let no_pages = LoadedManifest::new(
            no_pages.pointer,
            no_pages.pointer_etag,
            no_pages.pointer_version,
            no_pages.manifest,
        );
        let full_without_pages = Snapshot::new(no_pages, &empty_view, store, paths)
            .scan_label("Doc")
            .await
            .unwrap();
        assert_eq!(full_with_pages, full_without_pages);
    }

    #[tokio::test]
    async fn projected_visit_streams_rows_and_bounds_resident_catalog_cache() {
        let store = make_store();
        let paths = make_paths("node-property-visit-bounded");
        let committed = object_native_doc_fixture(&store, &paths, 64, 2048).await;
        let empty = Memtable::new();
        let empty_view = empty.snapshot_view();
        let snapshot = Snapshot::new(committed, &empty_view, store, paths);
        let mut visited = 0usize;
        snapshot
            .visit_label_with_projection::<_, Error>("Doc", &["title".into()], |row| {
                visited += 1;
                assert_eq!(row.properties.len(), 1);
                assert!(row.properties.contains_key("title"));
                Ok(())
            })
            .await
            .unwrap();
        assert_eq!(visited, 64);
        let cache = snapshot
            .node_property_readers
            .lock()
            .expect("node property reader cache mutex poisoned");
        assert!(cache.len() <= SNAPSHOT_NODE_PROPERTY_READER_CACHE_MAX_ENTRIES);
        assert!(
            cache.used_bytes()
                <= snapshot_local_cache_max_bytes(
                    SNAPSHOT_NODE_PROPERTY_READER_CACHE_MAX_BYTES_ENV,
                )
        );
    }

    #[tokio::test]
    async fn limited_disjoint_scan_stops_before_full_sst_and_keeps_predicate_column_internal() {
        let store = make_store();
        let paths = make_paths("limited-disjoint-ranged");
        let (_ms, _fence, _schema, committed) =
            limited_doc_fixture(&store, &paths, 64, 8, true).await;
        let empty = Memtable::new();
        let empty_view = empty.snapshot_view();
        let predicate = vigente_predicate();

        let baseline = Snapshot::new(committed.clone(), &empty_view, store.clone(), paths.clone())
            .scan_label_with_predicates_and_projection(
                "Doc",
                std::slice::from_ref(&predicate),
                Some(&["key".into()]),
            )
            .await
            .unwrap();

        let cache = SstCache::new(8 * 1024 * 1024);
        let limited = Snapshot::new(committed.clone(), &empty_view, store.clone(), paths.clone())
            .with_cache(cache.clone())
            .with_ranged_reads(true)
            .scan_label_with_predicates_and_projection_limited(
                "Doc",
                std::slice::from_ref(&predicate),
                Some(&["key".into()]),
                5,
            )
            .await
            .unwrap();

        assert_eq!(
            limited.iter().map(|node| node.id).collect::<Vec<_>>(),
            baseline
                .iter()
                .take(5)
                .map(|node| node.id)
                .collect::<Vec<_>>()
        );
        assert!(limited
            .iter()
            .all(|node| node.properties.keys().eq(["key"])));
        assert_eq!(cache.limited_node_scan_fast_paths(), 1);
        assert_eq!(cache.limited_node_scan_fallbacks(), 0);
        assert_eq!(cache.limited_node_scan_output_rows(), 5);
        assert!(
            cache.limited_node_scan_decoded_rows() < 64,
            "LIMIT decoded the complete SST"
        );
        assert!(
            cache.limited_node_scan_row_groups() < 8,
            "LIMIT fetched every row group"
        );
        let node_sst = committed
            .manifest
            .ssts
            .iter()
            .find(|descriptor| descriptor.kind == SstKind::Nodes)
            .unwrap();
        assert!(
            cache.limited_node_scan_range_bytes() < node_sst.size_bytes,
            "ranged LIMIT read {} bytes from a {} byte SST",
            cache.limited_node_scan_range_bytes(),
            node_sst.size_bytes
        );
    }

    #[tokio::test]
    async fn batched_label_membership_is_exact_ordered_and_fails_closed() {
        let store = make_store();
        let paths = make_paths("batched-label-membership");
        let (_ms, _fence, _schema, committed) =
            limited_doc_fixture(&store, &paths, 64, 8, true).await;
        let empty = Memtable::new();
        let empty_view = empty.snapshot_view();
        let ids = vec![
            sorted_node_id(2),
            sorted_node_id(5), // physical tombstone
            sorted_node_id(200),
            sorted_node_id(2), // duplicate/order preservation
        ];
        let doc = vec!["Doc".to_string()];
        let cache = SstCache::new(4 * 1024 * 1024);
        let snapshot = Snapshot::new(committed.clone(), &empty_view, store.clone(), paths.clone())
            .with_cache(cache.clone());
        assert_eq!(
            snapshot
                .try_batch_nodes_have_labels(&doc, &ids)
                .await
                .unwrap(),
            Some(vec![true, false, false, true])
        );
        assert_eq!(
            snapshot
                .try_batch_nodes_have_labels(&["NotInDictionary".into()], &ids)
                .await
                .unwrap(),
            Some(vec![false; ids.len()])
        );
        assert_eq!(
            snapshot
                .try_batch_nodes_have_labels(&[], &ids)
                .await
                .unwrap(),
            None,
            "zero labels cannot prove endpoint existence"
        );
        assert_eq!(cache.label_membership_fast_paths(), 2);
        assert_eq!(cache.label_membership_fallbacks(), 1);
        assert_eq!(cache.label_membership_probes(), 1);
        assert_eq!(
            cache.label_membership_candidates(),
            3,
            "candidates counts probed work, not inputs: id 200 falls outside \
             the only descriptor's key range and never reaches a probe, while \
             the repeated id 2 still costs one each time"
        );
        assert_eq!(
            cache.decoded_node_row_group_inserts(),
            0,
            "label membership must not hydrate complete node rows"
        );

        let descriptor = committed
            .manifest
            .ssts
            .iter()
            .find(|descriptor| descriptor.kind == SstKind::Nodes)
            .unwrap();
        let label_index = descriptor
            .label_index
            .as_ref()
            .expect("current node flush must emit a label sidecar");
        assert_eq!(label_index.format, PropertyIndexFormat::PagedV1);
        assert!(
            cache.label_membership_pages() < 8,
            "four endpoint probes touched too many B+tree pages: {}",
            cache.label_membership_pages()
        );
        assert!(
            cache.label_membership_entries_examined() <= label_index.posting_count,
            "the batch probe examined more entries than the complete sidecar"
        );
        // This fixture's sidecar is a single page plus its header, so no probe
        // can read strictly less than the object. What the batch must never do
        // is read one page once per key: that is what pushes the total past the
        // object size, and it is the regression this bound catches.
        assert!(
            cache.label_membership_bytes() <= label_index.size_bytes,
            "the batch probe read {} bytes of a {}-byte sidecar; one shared \
             descent must not re-read a page per key",
            cache.label_membership_bytes(),
            label_index.size_bytes
        );
        let absolute = Path::from(format!(
            "{}/{}",
            paths.namespace_prefix().as_ref(),
            label_index.path
        ));
        let original = store.get(&absolute).await.unwrap().bytes().await.unwrap();
        store.delete(&absolute).await.unwrap();
        assert_eq!(
            Snapshot::new(committed.clone(), &empty_view, store.clone(), paths.clone())
                .try_batch_nodes_have_labels(&doc, &ids)
                .await
                .unwrap(),
            None,
            "a missing optional sidecar must select the authoritative point path"
        );
        store.put(&absolute, original.clone().into()).await.unwrap();
        store
            .put(
                &absolute,
                Bytes::from_static(b"corrupt label sidecar").into(),
            )
            .await
            .unwrap();
        assert_eq!(
            Snapshot::new(committed.clone(), &empty_view, store.clone(), paths.clone())
                .try_batch_nodes_have_labels(&doc, &ids)
                .await
                .unwrap(),
            None
        );
        store.put(&absolute, original.into()).await.unwrap();

        // Even a byte-valid body cannot be transplanted to another physical
        // node SST: the footer/page CRC salt binds its exact SST UUID.
        let mut transplanted = committed.clone();
        transplanted
            .manifest
            .ssts
            .iter_mut()
            .find(|descriptor| descriptor.kind == SstKind::Nodes)
            .unwrap()
            .id = Uuid::now_v7();
        let transplanted = LoadedManifest::new(
            transplanted.pointer,
            transplanted.pointer_etag,
            transplanted.pointer_version,
            transplanted.manifest,
        );
        assert_eq!(
            Snapshot::new(transplanted, &empty_view, store, paths)
                .try_batch_nodes_have_labels(&doc, &ids)
                .await
                .unwrap(),
            None
        );
    }

    #[tokio::test]
    async fn limited_scan_falls_back_for_memtable_and_overlapping_ssts() {
        let store = make_store();
        let paths = make_paths("limited-fallbacks");
        let (ms, fence, schema, first) = limited_doc_fixture(&store, &paths, 20, 8, false).await;
        let predicate = vigente_predicate();

        // A live node memtable delta makes the immutable prefix non-authoritative.
        let mut live = Memtable::new();
        live.apply(
            MemKey::Node {
                id: sorted_node_id(2),
            },
            100,
            MemOp::Tombstone,
        );
        let live_view = live.snapshot_view();
        let cache = SstCache::new(4 * 1024 * 1024);
        let snap = Snapshot::new(first.clone(), &live_view, store.clone(), paths.clone())
            .with_cache(cache.clone());
        let full = snap
            .scan_label_with_predicates("Doc", std::slice::from_ref(&predicate))
            .await
            .unwrap();
        let limited = snap
            .scan_label_with_predicates_and_projection_limited(
                "Doc",
                std::slice::from_ref(&predicate),
                None,
                3,
            )
            .await
            .unwrap();
        assert_eq!(
            limited.iter().map(|node| node.id).collect::<Vec<_>>(),
            full.iter().take(3).map(|node| node.id).collect::<Vec<_>>()
        );
        assert_eq!(cache.limited_node_scan_fallbacks(), 1);

        // A second L0 range updates existing ids. Even with an empty memtable,
        // global range overlap must retain full LWW/tombstone reconciliation.
        let second = flush_batch(
            &ms,
            &fence,
            &first,
            &schema,
            vec![
                (
                    sorted_node_id(2),
                    200,
                    MemOp::Upsert(limited_doc_payload(2, false, false)),
                ),
                (sorted_node_id(4), 201, MemOp::Tombstone),
            ],
        )
        .await;
        let empty = Memtable::new();
        let empty_view = empty.snapshot_view();
        let overlap_cache = SstCache::new(4 * 1024 * 1024);
        let overlap =
            Snapshot::new(second, &empty_view, store, paths).with_cache(overlap_cache.clone());
        let full = overlap
            .scan_label_with_predicates("Doc", std::slice::from_ref(&predicate))
            .await
            .unwrap();
        let limited = overlap
            .scan_label_with_predicates_and_projection_limited(
                "Doc",
                std::slice::from_ref(&predicate),
                None,
                3,
            )
            .await
            .unwrap();
        assert_eq!(
            limited.iter().map(|node| node.id).collect::<Vec<_>>(),
            full.iter().take(3).map(|node| node.id).collect::<Vec<_>>()
        );
        assert_eq!(overlap_cache.limited_node_scan_fast_paths(), 0);
        assert_eq!(overlap_cache.limited_node_scan_fallbacks(), 1);
    }

    #[tokio::test]
    async fn node_predicates_are_evaluated_after_lww_for_overlap_and_memtable() {
        let store = make_store();
        let paths = make_paths("node-predicate-after-lww");
        let ms = ManifestStore::new(store.clone(), paths.clone());
        let mut base = ms.bootstrap(Uuid::now_v7()).await.unwrap();
        base.manifest.label_dict.intern("Doc");
        let fence = WriterFence::new(base.manifest.epoch);
        let schema = SchemaBuilder::new()
            .label(limited_doc_label())
            .unwrap()
            .build();
        let id = sorted_node_id(1);

        // The old version matches `vigente = true`; its one-row group can be
        // selected by predicate statistics.
        let first = flush_batch(
            &ms,
            &fence,
            &base,
            &schema,
            vec![(id, 10, MemOp::Upsert(limited_doc_payload(1, true, false)))],
        )
        .await;

        // A newer one-row SST for the same id does not match. Pruning this row
        // group before reconciliation used to resurrect the old true version.
        let second = flush_batch(
            &ms,
            &fence,
            &first,
            &schema,
            vec![(id, 20, MemOp::Upsert(limited_doc_payload(1, false, false)))],
        )
        .await;
        let empty = Memtable::new();
        let empty_view = empty.snapshot_view();
        let predicate = vigente_predicate();
        let overlap = Snapshot::new(second, &empty_view, store.clone(), paths.clone());
        assert!(overlap.disjoint_node_descriptors().is_none());
        assert!(
            overlap
                .scan_label_with_predicates_and_projection(
                    "Doc",
                    std::slice::from_ref(&predicate),
                    Some(&["key".into()]),
                )
                .await
                .unwrap()
                .is_empty(),
            "newer non-matching SST version must hide the old matching version"
        );
        assert!(
            overlap
                .scan_label_with_predicates_and_projection_limited(
                    "Doc",
                    std::slice::from_ref(&predicate),
                    Some(&["key".into()]),
                    1,
                )
                .await
                .unwrap()
                .is_empty(),
            "limited overlap fallback must preserve the same LWW semantics"
        );

        // The same shadowing rule applies to an unflushed memtable upsert.
        let mut live = Memtable::new();
        live.apply(
            MemKey::Node { id },
            30,
            MemOp::Upsert(limited_doc_payload(1, false, false)),
        );
        let live_view = live.snapshot_view();
        let memtable_shadow = Snapshot::new(first, &live_view, store, paths);
        assert!(
            memtable_shadow
                .scan_label_with_predicates("Doc", std::slice::from_ref(&predicate))
                .await
                .unwrap()
                .is_empty(),
            "newer non-matching memtable version must hide the old matching SST version"
        );
    }

    #[tokio::test]
    async fn metadata_count_rejects_corrupt_summaries_and_checked_add_overflow() {
        let store = make_store();
        let paths = make_paths("metadata-count-validation");
        let (_ms, _fence, _schema, committed) =
            limited_doc_fixture(&store, &paths, 8, 4, false).await;
        let doc_id = committed.manifest.label_dict.id("Doc").unwrap().get();
        let empty = Memtable::new();
        let empty_view = empty.snapshot_view();

        let count = |loaded: LoadedManifest, label: Option<&str>| {
            let loaded = LoadedManifest::new(
                loaded.pointer,
                loaded.pointer_etag,
                loaded.pointer_version,
                loaded.manifest,
            );
            Snapshot::new(loaded, &empty_view, store.clone(), paths.clone())
                .metadata_node_count(label)
        };
        let mutate_node = |loaded: &mut LoadedManifest,
                           mutate: &mut dyn FnMut(&mut SstDescriptor)| {
            let descriptor = loaded
                .manifest
                .ssts
                .iter_mut()
                .find(|descriptor| descriptor.kind == SstKind::Nodes)
                .unwrap();
            mutate(descriptor);
        };

        assert_eq!(count(committed.clone(), None), Some(7));
        assert_eq!(count(committed.clone(), Some("Doc")), Some(7));

        let mut too_many_tombstones = committed.clone();
        mutate_node(&mut too_many_tombstones, &mut |descriptor| {
            descriptor.kind_specific =
                crate::manifest::KindSpecificStats::Nodes { tombstone_count: 9 };
        });
        assert_eq!(count(too_many_tombstones, None), None);

        // Empty physical descriptors are excluded from disjoint ranges, but
        // their counters must be validated before exclusion.
        let mut corrupt_empty = committed.clone();
        mutate_node(&mut corrupt_empty, &mut |descriptor| {
            descriptor.row_count = 0;
            descriptor.kind_specific =
                crate::manifest::KindSpecificStats::Nodes { tombstone_count: 1 };
        });
        assert_eq!(count(corrupt_empty, None), None);

        let mut wrong_label_count = committed.clone();
        mutate_node(&mut wrong_label_count, &mut |descriptor| {
            descriptor.label_index.as_mut().unwrap().label_count += 1;
        });
        assert_eq!(count(wrong_label_count, None), None);

        let mut wrong_posting_count = committed.clone();
        mutate_node(&mut wrong_posting_count, &mut |descriptor| {
            descriptor.label_index.as_mut().unwrap().posting_count += 1;
        });
        assert_eq!(count(wrong_posting_count, Some("Doc")), None);

        let mut duplicate_label_ids = committed.clone();
        mutate_node(&mut duplicate_label_ids, &mut |descriptor| {
            let index = descriptor.label_index.as_mut().unwrap();
            index.label_count = 2;
            index.posting_count = 7;
            index.per_label_counts = vec![(doc_id, 3), (doc_id, 4)];
        });
        assert_eq!(count(duplicate_label_ids, None), None);

        let mut descending_label_ids = committed.clone();
        let other_id = descending_label_ids
            .manifest
            .label_dict
            .intern("Other")
            .get();
        mutate_node(&mut descending_label_ids, &mut |descriptor| {
            let index = descriptor.label_index.as_mut().unwrap();
            index.label_count = 2;
            index.posting_count = 7;
            index.per_label_counts = vec![(other_id, 1), (doc_id, 6)];
        });
        assert_eq!(count(descending_label_ids, None), None);

        let mut unknown_label_id = committed.clone();
        mutate_node(&mut unknown_label_id, &mut |descriptor| {
            let index = descriptor.label_index.as_mut().unwrap();
            index.label_count = 1;
            index.posting_count = 7;
            index.per_label_counts = vec![(u32::MAX, 7)];
        });
        assert_eq!(count(unknown_label_id, None), None);

        let mut label_exceeds_live_rows = committed.clone();
        mutate_node(&mut label_exceeds_live_rows, &mut |descriptor| {
            let index = descriptor.label_index.as_mut().unwrap();
            index.label_count = 1;
            index.posting_count = 8;
            index.per_label_counts = vec![(doc_id, 8)];
        });
        assert_eq!(count(label_exceeds_live_rows, Some("Doc")), None);

        // An empty per-label vector is the rolling-upgrade marker. It cannot
        // answer a label count, but stale non-zero body summaries must not
        // prevent the independent global row/tombstone count.
        let mut legacy_counts = committed.clone();
        mutate_node(&mut legacy_counts, &mut |descriptor| {
            let index = descriptor.label_index.as_mut().unwrap();
            index.per_label_counts.clear();
            index.label_count = 1;
            index.posting_count = 7;
        });
        assert_eq!(count(legacy_counts.clone(), None), Some(7));
        assert_eq!(count(legacy_counts, Some("Doc")), None);

        // The individual summaries are valid, but their additive global and
        // per-label totals overflow u64. Saturation would turn corrupt
        // metadata into an apparently authoritative answer.
        let mut overflow = committed.clone();
        mutate_node(&mut overflow, &mut |descriptor| {
            descriptor.row_count = u64::MAX;
            descriptor.kind_specific =
                crate::manifest::KindSpecificStats::Nodes { tombstone_count: 0 };
            let index = descriptor.label_index.as_mut().unwrap();
            index.label_count = 1;
            index.posting_count = u64::MAX;
            index.per_label_counts = vec![(doc_id, u64::MAX)];
        });
        let mut second = overflow
            .manifest
            .ssts
            .iter()
            .find(|descriptor| descriptor.kind == SstKind::Nodes)
            .unwrap()
            .clone();
        second.id = Uuid::now_v7();
        second.path = "sst/L0/overflow-second.parquet".into();
        second.min_key = *sorted_node_id(9).as_bytes();
        second.max_key = second.min_key;
        second.row_count = 1;
        second.kind_specific = crate::manifest::KindSpecificStats::Nodes { tombstone_count: 0 };
        let second_index = second.label_index.as_mut().unwrap();
        second_index.label_count = 1;
        second_index.posting_count = 1;
        second_index.per_label_counts = vec![(doc_id, 1)];
        overflow.manifest.ssts.push(second);
        assert_eq!(count(overflow.clone(), None), None);
        assert_eq!(count(overflow, Some("Doc")), None);
    }

    #[tokio::test]
    async fn limited_scan_zero_does_no_io_and_invalid_range_disables_metadata_fast_path() {
        let store = make_store();
        let paths = make_paths("limited-zero-invalid-range");
        let (_ms, _fence, _schema, mut committed) =
            limited_doc_fixture(&store, &paths, 8, 4, false).await;
        let descriptor = committed
            .manifest
            .ssts
            .iter_mut()
            .find(|descriptor| descriptor.kind == SstKind::Nodes)
            .unwrap();
        std::mem::swap(&mut descriptor.min_key, &mut descriptor.max_key);
        committed = LoadedManifest::new(
            committed.pointer,
            committed.pointer_etag,
            committed.pointer_version,
            committed.manifest,
        );

        let empty = Memtable::new();
        let empty_view = empty.snapshot_view();
        let cache = SstCache::new(4 * 1024 * 1024);
        let snap = Snapshot::new(committed, &empty_view, store, paths).with_cache(cache.clone());
        assert!(snap.metadata_node_count(None).is_none());
        assert!(snap
            .scan_label_with_predicates_and_projection_limited("Doc", &[], None, 0)
            .await
            .unwrap()
            .is_empty());
        assert_eq!(cache.limited_node_scan_fast_paths(), 0);
        assert_eq!(cache.limited_node_scan_fallbacks(), 0);
        assert_eq!(cache.limited_node_scan_range_bytes(), 0);
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
    async fn exact_record_locator_skips_parquet_for_interleaved_rows() {
        let store = make_store();
        let paths = make_paths("batch-node-locator");
        let ms = ManifestStore::new(store.clone(), paths.clone());
        let mut base = ms.bootstrap(Uuid::now_v7()).await.unwrap();
        base.manifest.label_dict.intern("Person");
        let fence = WriterFence::new(base.manifest.epoch);
        let schema = SchemaBuilder::new().label(person_label()).unwrap().build();
        const TOTAL: usize = 32_768;
        let ids: Vec<NodeId> = (0..TOTAL)
            .map(|i| NodeId::from_uuid(Uuid::from_bytes(((i as u128 + 1) * 17).to_be_bytes())))
            .collect();
        let rows = ids
            .iter()
            .enumerate()
            .map(|(i, id)| {
                (
                    *id,
                    i as u64 + 1,
                    MemOp::Upsert(node_payload(&format!("n{i}"), Some((i % 100) as i32))),
                )
            })
            .collect();
        let committed =
            flush_batch_with_row_group_rows(256, &ms, &fence, &base, &schema, rows).await;
        let descriptor = committed
            .manifest
            .ssts
            .iter()
            .find(|sst| sst.kind == SstKind::Nodes)
            .unwrap();
        let locator_size = descriptor.node_locator.as_ref().unwrap().size_bytes;
        let absolute = format!("{}/{}", paths.namespace_prefix().as_ref(), descriptor.path);
        // Keep the probe density representative of the reported legal-corpus
        // workload (2k candidates across ~783k rows).  At higher densities a
        // correct B+tree necessarily touches more than half of its leaves,
        // making the sublinear assertion below mathematically impossible.
        const PROBES: usize = 64;
        let probes: Vec<NodeId> = (0..PROBES).map(|i| ids[i * (TOTAL / PROBES)]).collect();

        let cache = SstCache::new(64 * 1024 * 1024);
        let empty = Memtable::new();
        let view = empty.snapshot_view();
        let snap = Snapshot::new(committed, &view, store, paths)
            .with_cache(cache.clone())
            .with_ranged_reads(true);
        let got = snap.batch_lookup_nodes("Person", &probes).await.unwrap();
        assert!(got.iter().all(Option::is_some));
        assert_eq!(cache.node_locator_probes(), 1);
        assert!(
            cache.node_locator_entries_examined() < (TOTAL / 2) as u64,
            "locator must examine search-path leaves, not all {TOTAL} ids"
        );
        assert!(
            cache.node_locator_bytes() < locator_size / 2,
            "range probe must not fetch the complete locator"
        );
        assert_eq!(
            cache.sparse_node_filter_scans(),
            0,
            "exact ordinals must bypass the O(N) node_id RowFilter"
        );
        assert_eq!(
            cache.decoded_node_row_group_inserts(),
            0,
            "exact-record results must not decode or cache Parquet row groups"
        );
        assert_eq!(
            cache.metadata_usage_bytes(),
            0,
            "an exact-record hit must not even open the Parquet footer"
        );
        assert!(
            cache.get(&absolute).is_none(),
            "exact-record path must not fetch the complete node SST"
        );
    }

    #[tokio::test]
    async fn optional_accelerators_fall_back_when_stale_or_swept_by_old_reader() {
        let store = make_store();
        let paths = make_paths("optional-accelerator-fallback");
        let ms = ManifestStore::new(store.clone(), paths.clone());
        let mut base = ms.bootstrap(Uuid::now_v7()).await.unwrap();
        base.manifest.label_dict.intern("Person");
        let fence = WriterFence::new(base.manifest.epoch);
        let schema = SchemaBuilder::new()
            .label(LabelDef {
                name: "Person".into(),
                properties: vec![PropertyDef::new("name", DataType::Utf8, false)
                    .unwrap()
                    .with_unique(true)],
            })
            .unwrap()
            .build();
        let rows: Vec<_> = (1..=16u8)
            .map(|i| {
                (
                    sorted_node_id(i),
                    i as u64,
                    MemOp::Upsert(node_payload(&format!("n{i}"), None)),
                )
            })
            .collect();
        let committed = flush_batch(&ms, &fence, &base, &schema, rows).await;
        let descriptor = committed
            .manifest
            .ssts
            .iter()
            .find(|sst| sst.kind == SstKind::Nodes)
            .unwrap();
        let locator = descriptor.node_locator.clone().unwrap();
        let paged = descriptor
            .equality_property_indices
            .iter()
            .find(|index| index.property == "name")
            .and_then(|index| index.paged.clone())
            .unwrap();

        // A count mismatch makes the locator non-authoritative even when its
        // body is structurally valid.
        let mut stale = committed.clone();
        stale
            .manifest
            .ssts
            .iter_mut()
            .find(|sst| sst.kind == SstKind::Nodes)
            .unwrap()
            .node_locator
            .as_mut()
            .unwrap()
            .entry_count -= 1;
        stale = LoadedManifest::new(
            stale.pointer,
            stale.pointer_etag,
            stale.pointer_version,
            stale.manifest,
        );
        let cache = SstCache::new(8 * 1024 * 1024);
        let empty = Memtable::new();
        let view = empty.snapshot_view();
        let snap =
            Snapshot::new(stale, &view, store.clone(), paths.clone()).with_cache(cache.clone());
        let got = snap
            .batch_lookup_nodes("Person", &[sorted_node_id(7)])
            .await
            .unwrap();
        assert!(got[0].is_some());
        assert_eq!(cache.node_locator_probes(), 0);

        // A structurally valid but partial paged property mirror is equally
        // non-authoritative. Its header count disagrees with the immutable
        // manifest descriptor, so the missing n7 entry must be recovered from
        // the legacy bincode map.
        let partial_equality = crate::sst::paged_index::build_equality(&BTreeMap::from([(
            "n1".into(),
            vec![*sorted_node_id(1).as_bytes()],
        )]))
        .unwrap();
        let prefix = paths.namespace_prefix();
        store
            .put(
                &Path::from(format!("{}/{}", prefix.as_ref(), paged.path)),
                partial_equality.into(),
            )
            .await
            .unwrap();
        let snap = Snapshot::new(committed.clone(), &view, store.clone(), paths.clone());
        let got = snap
            .batch_lookup_nodes_by_property("Person", "name", &["n7".into()])
            .await
            .unwrap();
        assert_eq!(
            got[0].as_ref().and_then(|node| node.properties.get("name")),
            Some(&Value::Str("n7".into()))
        );

        // Decode/checksum corruption in optional accelerators is also
        // recoverable: the legacy bincode map and Parquet body remain the
        // authority. Do not let a damaged cache-like mirror break reads.
        store
            .put(
                &Path::from(format!("{}/{}", prefix.as_ref(), paged.path)),
                Bytes::from_static(b"corrupt-paged-index").into(),
            )
            .await
            .unwrap();
        store
            .put(
                &Path::from(format!("{}/{}", prefix.as_ref(), locator.path)),
                Bytes::from_static(b"corrupt-node-locator").into(),
            )
            .await
            .unwrap();
        let snap = Snapshot::new(committed.clone(), &view, store.clone(), paths.clone());
        let got = snap
            .batch_lookup_nodes_by_property("Person", "name", &["n7".into()])
            .await
            .unwrap();
        assert_eq!(
            got[0].as_ref().and_then(|node| node.properties.get("name")),
            Some(&Value::Str("n7".into()))
        );

        // A 2.0.4 janitor can sweep fields it does not understand. Both
        // optional objects may disappear while the legacy `.bin` and Parquet
        // remain valid; queries must fall back instead of failing.
        store
            .delete(&Path::from(format!("{}/{}", prefix.as_ref(), paged.path)))
            .await
            .unwrap();
        store
            .delete(&Path::from(format!("{}/{}", prefix.as_ref(), locator.path)))
            .await
            .unwrap();
        let snap = Snapshot::new(committed, &view, store, paths);
        let got = snap
            .batch_lookup_nodes_by_property("Person", "name", &["n7".into()])
            .await
            .unwrap();
        assert_eq!(
            got[0].as_ref().and_then(|node| node.properties.get("name")),
            Some(&Value::Str("n7".into()))
        );
    }

    #[tokio::test]
    async fn equality_sidecars_require_complete_marker_and_matching_paged_header() {
        let store = make_store();
        let paths = make_paths("equality-authority-markers");
        let ms = ManifestStore::new(store.clone(), paths.clone());
        let mut base = ms.bootstrap(Uuid::now_v7()).await.unwrap();
        base.manifest.label_dict.intern("Person");
        let fence = WriterFence::new(base.manifest.epoch);
        let schema = SchemaBuilder::new()
            .label(indexed_city_label())
            .unwrap()
            .build();
        let paris = sorted_node_id(1);
        let berlin = sorted_node_id(2);
        let committed = flush_batch(
            &ms,
            &fence,
            &base,
            &schema,
            vec![
                (paris, 1, MemOp::Upsert(city_payload("Alice", "Paris"))),
                (berlin, 2, MemOp::Upsert(city_payload("Bob", "Berlin"))),
            ],
        )
        .await;
        let equality = committed
            .manifest
            .ssts
            .iter()
            .find(|descriptor| descriptor.kind == SstKind::Nodes)
            .and_then(|descriptor| {
                descriptor
                    .equality_property_indices
                    .iter()
                    .find(|index| index.property == "city")
            })
            .cloned()
            .unwrap();
        let paged = equality.paged.clone().unwrap();

        // The mirror contains only Paris while the descriptor advertises two
        // distinct values. Berlin must still resolve through the legacy map on
        // exact, limited and ordered-prefix paths.
        let partial = crate::sst::paged_index::build_equality(&BTreeMap::from([(
            "Paris".into(),
            vec![*paris.as_bytes()],
        )]))
        .unwrap();
        store
            .put(
                &Path::from(format!(
                    "{}/{}",
                    paths.namespace_prefix().as_ref(),
                    paged.path
                )),
                partial.into(),
            )
            .await
            .unwrap();
        let empty = Memtable::new();
        let empty_view = empty.snapshot_view();
        let snap = Snapshot::new(committed.clone(), &empty_view, store.clone(), paths.clone());
        assert_eq!(
            snap.lookup_nodes_by_property("Person", "city", "Berlin")
                .await
                .unwrap()
                .into_iter()
                .map(|node| node.id)
                .collect::<Vec<_>>(),
            vec![berlin]
        );
        assert_eq!(
            snap.indexed_node_ids_by_property_value_limited(
                "Person",
                "city",
                &Value::Str("Berlin".into()),
                1,
            )
            .await
            .unwrap(),
            Some(vec![berlin])
        );
        assert_eq!(
            snap.ordered_node_ids_by_string_property("Person", "city", 2)
                .await
                .unwrap(),
            Some(vec![berlin, paris])
        );

        // Pre-marker manifests default to incomplete. Even a present sidecar
        // cannot answer absence: every authoritative consumer must choose its
        // exact scan fallback without touching a deliberately missing path.
        let mut legacy = committed;
        let equality = legacy
            .manifest
            .ssts
            .iter_mut()
            .find(|descriptor| descriptor.kind == SstKind::Nodes)
            .and_then(|descriptor| {
                descriptor
                    .equality_property_indices
                    .iter_mut()
                    .find(|index| index.property == "city")
            })
            .unwrap();
        equality.mixed_type_complete = false;
        equality.path = "sst/L0/must-not-be-read.eqidx_city.bin".into();
        equality.paged = None;
        let snap = Snapshot::new(legacy, &empty_view, store, paths);
        assert!(snap
            .indexed_node_ids_by_property_value("Person", "city", &Value::Str("Berlin".into()))
            .await
            .unwrap()
            .is_none());
        assert_eq!(
            snap.lookup_nodes_by_property("Person", "city", "Berlin")
                .await
                .unwrap()
                .into_iter()
                .map(|node| node.id)
                .collect::<Vec<_>>(),
            vec![berlin]
        );
        assert!(snap
            .ordered_node_ids_by_string_property("Person", "city", 2)
            .await
            .unwrap()
            .is_none());
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
            let w = crate::cache::decoded_node_row_group_weight(&(absolute.clone(), rg), &batches);
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
        let mut committed = flush_batch(&ms, &fence, &base, &schema, rows).await;
        committed
            .manifest
            .ssts
            .iter_mut()
            .find(|sst| sst.kind == SstKind::Nodes)
            .unwrap()
            .node_locator = None;

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
    async fn unique_lookup_accepts_mixed_legacy_unique_and_current_equality_scopes() {
        // Rolling upgrades retain legacy label-scoped node SSTs while current
        // flushes append id-primary (`scope == ""`) SSTs. The former encode a
        // unique key as `value -> NodeId`; the latter encode the same
        // per-label-unique key as a global equality posting list. Requiring
        // either format to cover every SST makes this valid mixed generation
        // fall back permanently to an O(N) label scan.
        let store = make_store();
        let paths = make_paths("mixed-unique-equality-coverage");
        let ms = ManifestStore::new(store.clone(), paths.clone());
        let mut base = ms.bootstrap(Uuid::now_v7()).await.unwrap();
        let person_lid = base.manifest.label_dict.intern("Person").get();
        let fence = WriterFence::new(base.manifest.epoch);
        let person = LabelDef {
            name: "Person".into(),
            properties: vec![PropertyDef::new("key", DataType::Utf8, false)
                .unwrap()
                .with_unique(true)],
        };
        let schema = SchemaBuilder::new().label(person.clone()).unwrap().build();

        // Synthesize one real legacy typed-column SST. Its non-empty scope is
        // the compatibility marker used by the reader, and its only property
        // accelerator is the legacy unique sidecar.
        let legacy_id = sorted_node_id(1);
        let legacy_row = crate::flush::NodeRow {
            id: *legacy_id.as_bytes(),
            lsn: 10,
            op: MemOp::Upsert(coded_node_payload("key", "legacy", person_lid)),
        };
        let finish =
            crate::flush::build_node_sst(&person, std::slice::from_ref(&legacy_row)).unwrap();
        let legacy_sst_id = Uuid::now_v7();
        let legacy_file = "legacy-person.parquet";
        let legacy_relative = format!("sst/level0/{legacy_file}");
        store
            .put(
                &paths.sst_object(0, legacy_file),
                finish.body.clone().into(),
            )
            .await
            .unwrap();
        let (legacy_unique, legacy_sidecars) = crate::flush::prepare_unique_property_sidecars(
            &paths,
            0,
            &legacy_sst_id,
            "Person",
            &person,
            std::slice::from_ref(&legacy_row),
        )
        .unwrap();
        for (path, body) in legacy_sidecars {
            crate::flush::put_sidecar_payload(store.clone(), &path, body)
                .await
                .unwrap();
        }
        let stats = finish.stats;
        base.manifest.ssts.push(SstDescriptor {
            id: legacy_sst_id,
            kind: SstKind::Nodes,
            scope: "Person".into(),
            level: crate::manifest::SstLevel::L0,
            path: legacy_relative,
            size_bytes: finish.body.len() as u64,
            row_count: stats.row_count,
            created_at: chrono::Utc::now(),
            min_key: stats.min_node_id,
            max_key: stats.max_node_id,
            min_lsn: stats.min_lsn,
            max_lsn: stats.max_lsn,
            schema_version_min: stats.schema_version_min,
            schema_version_max: stats.schema_version_max,
            property_stats: stats.property_stats,
            kind_specific: crate::manifest::KindSpecificStats::Nodes {
                tombstone_count: stats.tombstone_count,
            },
            bloom: None,
            unique_property_indices: legacy_unique,
            equality_property_indices: Vec::new(),
            label_index: None,
            node_locator: None,
            per_label_property_stats: Vec::new(),
        });
        base = LoadedManifest::new(
            base.pointer.clone(),
            base.pointer_etag.clone(),
            base.pointer_version.clone(),
            base.manifest,
        );

        // A normal current flush appends an id-primary SST and re-expresses
        // the schema-unique key as a complete global equality posting sidecar.
        let current_id = sorted_node_id(2);
        let committed = flush_batch(
            &ms,
            &fence,
            &base,
            &schema,
            vec![(
                current_id,
                20,
                MemOp::Upsert(coded_node_payload("key", "current", person_lid)),
            )],
        )
        .await;
        let node_ssts: Vec<&SstDescriptor> = committed
            .manifest
            .ssts
            .iter()
            .filter(|sst| sst.kind == SstKind::Nodes)
            .collect();
        assert_eq!(node_ssts.len(), 2);
        assert!(node_ssts.iter().any(|sst| {
            sst.scope == "Person"
                && sst
                    .unique_property_indices
                    .iter()
                    .any(|sidecar| sidecar.property == "key")
                && sst.equality_property_indices.is_empty()
        }));
        assert!(node_ssts.iter().any(|sst| {
            sst.scope.is_empty()
                && sst.unique_property_indices.is_empty()
                && sst
                    .equality_property_indices
                    .iter()
                    .any(|sidecar| sidecar.property == "key" && sidecar.mixed_type_complete)
        }));
        let legacy_descriptor = node_ssts
            .iter()
            .copied()
            .find(|sst| sst.scope == "Person")
            .unwrap();
        let current_descriptor = node_ssts
            .iter()
            .copied()
            .find(|sst| sst.scope.is_empty())
            .unwrap();
        assert!(matches!(
            string_property_sidecar(legacy_descriptor, "Person", "key"),
            Some(StringPropertySidecar::Unique(_))
        ));
        assert!(matches!(
            string_property_sidecar(current_descriptor, "Person", "key"),
            Some(StringPropertySidecar::Equality(_))
        ));
        let corrupted_manifest = committed.clone();
        let legacy_sidecar_path = legacy_descriptor.unique_property_indices[0].path.clone();

        let empty = Memtable::new();
        let empty_view = empty.snapshot_view();
        let property_cache = Arc::new(crate::property_index::PropertyIndexCache::new());
        let snap = Snapshot::new(committed, &empty_view, store.clone(), paths.clone())
            .with_cache(SstCache::new(4 * 1024 * 1024))
            .with_property_index_cache(property_cache.clone());
        let values = vec![
            "legacy".to_string(),
            "current".to_string(),
            "missing".to_string(),
            "legacy".to_string(),
        ];
        let got = snap
            .batch_lookup_nodes_by_property("Person", "key", &values)
            .await
            .unwrap();
        assert_eq!(
            got.iter()
                .map(|node| node.as_ref().map(|node| node.id))
                .collect::<Vec<_>>(),
            vec![Some(legacy_id), Some(current_id), None, Some(legacy_id)]
        );
        assert!(
            property_cache.get("Person", "key").is_none(),
            "mixed complete sidecars must not populate the scan fallback cache"
        );

        assert_eq!(
            snap.ordered_node_ids_by_string_property("Person", "key", 2)
                .await
                .unwrap(),
            Some(vec![current_id, legacy_id]),
            "ordered pagination must merge current equality and legacy unique sidecars"
        );
        assert_eq!(property_cache.ordered_prefix_calls(), 1);

        assert_eq!(
            snap.lookup_node_by_property("Person", "key", "current")
                .await
                .unwrap()
                .map(|node| node.id),
            Some(current_id)
        );
        assert!(property_cache.get("Person", "key").is_none());

        // The sidecar is an accelerator, not authority. A damaged legacy
        // unique object in a rolling-upgrade manifest must select the ordinary
        // exact scan fallback rather than under-returning or failing the
        // query.
        let corrupted_absolute = Path::from(format!(
            "{}/{}",
            paths.namespace_prefix().as_ref(),
            legacy_sidecar_path
        ));
        store
            .put(&corrupted_absolute, Bytes::from_static(b"corrupt").into())
            .await
            .unwrap();
        let fallback = Snapshot::new(corrupted_manifest, &empty_view, store, paths);
        assert_eq!(
            fallback
                .lookup_node_by_property("Person", "key", "legacy")
                .await
                .unwrap()
                .map(|node| node.id),
            Some(legacy_id),
            "singular lookup must recover from a corrupt legacy unique sidecar"
        );
        assert_eq!(
            fallback
                .batch_lookup_nodes_by_property(
                    "Person",
                    "key",
                    &["current".into(), "legacy".into(), "missing".into()],
                )
                .await
                .unwrap()
                .into_iter()
                .map(|node| node.map(|node| node.id))
                .collect::<Vec<_>>(),
            vec![Some(current_id), Some(legacy_id), None],
            "batch lookup must recover from a corrupt legacy unique sidecar"
        );
        assert_eq!(
            fallback
                .ordered_node_ids_by_string_property("Person", "key", 2)
                .await
                .unwrap(),
            None,
            "corrupt optional sidecars must select the exact scan fallback"
        );
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
            .put(
                &object_store::path::Path::from(absolute),
                body.clone().into(),
            )
            .await
            .unwrap();
        // A single-owner unique sidecar is authoritative only on the legacy
        // label-scoped layout that actually emitted it. Mark this fabricated
        // descriptor accordingly; accepting the same map on an id-primary
        // (`scope == ""`) SST could hide an equal value owned by another
        // label, so current rolling-upgrade code rejects that unsafe shape.
        committed.manifest.ssts[account_sst].scope = "Account".into();
        committed.manifest.ssts[account_sst]
            .unique_property_indices
            .push(crate::manifest::UniquePropertyIndexDescriptor {
                property: "code".into(),
                path: relative,
                size_bytes: body.len() as u64,
                entry_count: 2,
                format: crate::manifest::PropertyIndexFormat::BincodeV0,
                paged: None,
                paged_build_unsupported: false,
            });
        committed = LoadedManifest::new(
            committed.pointer.clone(),
            committed.pointer_etag.clone(),
            committed.pointer_version.clone(),
            committed.manifest.clone(),
        );

        let empty = Memtable::new();
        let empty_view = empty.snapshot_view();
        let cache = Arc::new(crate::property_index::PropertyIndexCache::new());
        let sst_cache = SstCache::new(1 << 20);
        let snap = Snapshot::new(committed, &empty_view, store.clone(), paths.clone())
            .with_property_index_cache(cache.clone())
            .with_cache(sst_cache.clone());

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
        assert_eq!(
            cache.memtable_population_scans(),
            1,
            "the committed memtable claimant map is built once per property"
        );
        assert_eq!(
            sst_cache.property_sidecar_inserts(),
            1,
            "the sidecar is decoded once, not once per lookup"
        );
        assert_eq!(sst_cache.property_sidecar_misses(), 1);
        assert_eq!(sst_cache.property_sidecar_hits(), 1);
    }

    #[tokio::test]
    async fn global_property_lookup_preserves_cross_label_duplicates() {
        let store = make_store();
        let paths = make_paths("global-property-postings");
        let ms = ManifestStore::new(store.clone(), paths.clone());
        let mut base = ms.bootstrap(Uuid::now_v7()).await.unwrap();
        let a_label = base.manifest.label_dict.intern("A").get();
        let b_label = base.manifest.label_dict.intern("B").get();
        let fence = WriterFence::new(base.manifest.epoch);
        let key_def = || {
            PropertyDef::new("key", DataType::Utf8, false)
                .unwrap()
                .with_unique(true)
        };
        let schema = SchemaBuilder::new()
            .label(LabelDef {
                name: "A".into(),
                properties: vec![key_def()],
            })
            .unwrap()
            .label(LabelDef {
                name: "B".into(),
                properties: vec![key_def()],
            })
            .unwrap()
            .build();

        let a = sorted_node_id(1);
        let b = sorted_node_id(2);
        let both = sorted_node_id(3);
        let mut mt = Memtable::new();
        for (id, labels, key, lsn) in [
            (a, vec![a_label], "shared", 1),
            (b, vec![b_label], "shared", 2),
            (both, vec![a_label, b_label], "both", 3),
        ] {
            mt.apply(
                MemKey::Node { id },
                lsn,
                MemOp::Upsert(
                    NodeWriteRecord {
                        properties: BTreeMap::from([("key".into(), Value::Str(key.into()))]),
                        schema_version: 1,
                        labels,
                    }
                    .encode()
                    .unwrap(),
                ),
            );
        }
        let flushed = flush(&ms, &fence, &base, &mt.freeze(), schema.clone())
            .await
            .unwrap();
        assert!(
            flushed.committed.manifest.ssts.iter().any(|sst| sst
                .equality_property_indices
                .iter()
                .any(|index| index.property == "key")),
            "per-label unique keys need a global posting sidecar"
        );

        // A later tombstone-only SST contributes no key values, but must
        // still advertise an empty sidecar as a coverage marker. Otherwise
        // one harmless flush would demote every global lookup back to an
        // all-node scan.
        let mut tombstones = Memtable::new();
        tombstones.apply(
            MemKey::Node {
                id: sorted_node_id(99),
            },
            4,
            MemOp::Tombstone,
        );
        let flushed = flush(
            &ms,
            &fence,
            &flushed.committed,
            &tombstones.freeze(),
            schema,
        )
        .await
        .unwrap();
        let node_ssts: Vec<&SstDescriptor> = flushed
            .committed
            .manifest
            .ssts
            .iter()
            .filter(|sst| sst.kind == SstKind::Nodes)
            .collect();
        assert!(node_ssts.len() >= 2);
        assert!(node_ssts.iter().all(|sst| sst
            .equality_property_indices
            .iter()
            .any(|index| index.property == "key")));
        assert!(node_ssts.iter().any(|sst| sst
            .equality_property_indices
            .iter()
            .any(|index| index.property == "key" && index.distinct_values == 0)));
        let equality_sidecar_path = node_ssts
            .iter()
            .find_map(|sst| {
                sst.equality_property_indices
                    .iter()
                    .find(|index| index.property == "key" && index.distinct_values > 0)
                    .map(|index| index.path.clone())
            })
            .expect("non-empty equality sidecar");

        // Overlay the persisted postings with three newer versions:
        // `a` moves from shared→renamed, `b` is deleted, and a fresh `c`
        // claims shared. The batch must gather every stale/fresh claimant and
        // let current-value confirmation resolve LWW exactly.
        let c = sorted_node_id(4);
        let mut overlay = Memtable::new();
        overlay.apply(
            MemKey::Node { id: a },
            5,
            MemOp::Upsert(
                NodeWriteRecord {
                    properties: BTreeMap::from([("key".into(), Value::Str("renamed".into()))]),
                    schema_version: 1,
                    labels: vec![a_label],
                }
                .encode()
                .unwrap(),
            ),
        );
        overlay.apply(MemKey::Node { id: b }, 6, MemOp::Tombstone);
        overlay.apply(
            MemKey::Node { id: c },
            7,
            MemOp::Upsert(
                NodeWriteRecord {
                    properties: BTreeMap::from([("key".into(), Value::Str("shared".into()))]),
                    schema_version: 1,
                    labels: vec![b_label],
                }
                .encode()
                .unwrap(),
            ),
        );
        let overlay_view = overlay.snapshot_view();
        let property_cache = Arc::new(crate::property_index::PropertyIndexCache::new());
        let generation = property_cache.generation();
        let snap = Snapshot::new(
            flushed.committed.clone(),
            &overlay_view,
            store.clone(),
            paths.clone(),
        )
        .with_cache(SstCache::new(1 << 20))
        .with_property_index_cache(property_cache.clone());
        let lookup_calls = property_cache.equality_lookup_calls();
        let batched = snap
            .batch_lookup_nodes_by_property_any_label(
                "key",
                &[
                    "shared".into(),
                    "renamed".into(),
                    "both".into(),
                    "shared".into(),
                    "missing".into(),
                ],
            )
            .await
            .unwrap();
        assert_eq!(batched.len(), 5);
        assert_eq!(
            batched[0].iter().map(|node| node.id).collect::<Vec<_>>(),
            vec![c],
            "stale persisted claimants must fail current-value confirmation"
        );
        assert_eq!(
            batched[1].iter().map(|node| node.id).collect::<Vec<_>>(),
            vec![a],
            "a renamed memtable claimant must be visible immediately"
        );
        assert_eq!(
            batched[2].iter().map(|node| node.id).collect::<Vec<_>>(),
            vec![both],
            "a multi-label physical node must not be duplicated"
        );
        assert_eq!(
            batched[3].iter().map(|node| node.id).collect::<Vec<_>>(),
            vec![c],
            "a duplicate input value must reproduce the same fan-out slot"
        );
        assert!(batched[4].is_empty(), "a miss keeps its aligned empty slot");
        assert_eq!(
            property_cache.equality_lookup_calls() - lookup_calls,
            1,
            "one logical batch must count as one equality-index lookup"
        );
        assert!(
            property_cache.get_global_at("key", generation).is_none(),
            "complete sidecar coverage must not populate the all-node fallback cache"
        );

        // Equality sidecars are accelerators. If a rollback/janitor removed one
        // that the manifest still advertises, the whole logical batch must fall
        // back to one exact reconciliation rather than fail or scan per value.
        let missing_absolute = format!(
            "{}/{}",
            paths.namespace_prefix().as_ref(),
            equality_sidecar_path
        );
        store.delete(&Path::from(missing_absolute)).await.unwrap();
        let fallback_cache = Arc::new(crate::property_index::PropertyIndexCache::new());
        let fallback_calls = fallback_cache.equality_lookup_calls();
        let fallback = Snapshot::new(flushed.committed, &overlay_view, store, paths)
            .with_cache(SstCache::new(1 << 20))
            .with_property_index_cache(fallback_cache.clone())
            .batch_lookup_nodes_by_property_any_label(
                "key",
                &["shared".into(), "renamed".into(), "missing".into()],
            )
            .await
            .unwrap();
        assert_eq!(
            fallback[0].iter().map(|node| node.id).collect::<Vec<_>>(),
            vec![c]
        );
        assert_eq!(
            fallback[1].iter().map(|node| node.id).collect::<Vec<_>>(),
            vec![a]
        );
        assert!(fallback[2].is_empty());
        assert_eq!(
            fallback_cache.equality_lookup_calls() - fallback_calls,
            1,
            "legacy fallback remains one logical equality batch"
        );
    }

    #[cfg(all(feature = "vector-index", feature = "text-index"))]
    #[tokio::test]
    async fn persisted_index_freshness_pairs_label_member_range_and_lsn() {
        use crate::manifest::{
            KindSpecificStats, LabelIndexDescriptor, SstKind, SstLevel, TextIndexDescriptor,
            VectorIndexDescriptor, VectorMetric, VectorQuantization,
        };
        use chrono::Utc;

        fn index_sst(
            kind: SstKind,
            scope: &str,
            min_key: [u8; 16],
            max_key: [u8; 16],
            max_lsn: u64,
        ) -> SstDescriptor {
            let kind_specific = match kind {
                SstKind::VectorGraph => KindSpecificStats::VectorGraph {
                    dim: 4,
                    metric: "cosine".into(),
                    point_count: 2,
                    r: 16,
                    l_build: 32,
                    alpha: 1.2,
                    entry_medoid: 0,
                },
                SstKind::TextIndex => KindSpecificStats::TextIndex {
                    doc_count: 2,
                    term_count: 2,
                    total_len: 2,
                },
                _ => unreachable!("index_sst only builds vector/text descriptors"),
            };
            SstDescriptor {
                id: Uuid::now_v7(),
                kind,
                scope: scope.into(),
                level: SstLevel(1),
                path: format!("{scope}.idx"),
                size_bytes: 1,
                row_count: 2,
                created_at: Utc::now(),
                min_key,
                max_key,
                min_lsn: 0,
                max_lsn,
                schema_version_min: 0,
                schema_version_max: 0,
                property_stats: vec![],
                kind_specific,
                bloom: None,
                unique_property_indices: vec![],
                equality_property_indices: vec![],
                label_index: None,
                node_locator: None,
                per_label_property_stats: vec![],
            }
        }

        fn node_sst(
            min_key: [u8; 16],
            max_key: [u8; 16],
            max_lsn: u64,
            label_counts: Option<Vec<(u32, u64)>>,
        ) -> SstDescriptor {
            SstDescriptor {
                id: Uuid::now_v7(),
                kind: SstKind::Nodes,
                scope: String::new(),
                level: SstLevel::L0,
                path: format!("nodes-{max_lsn}.parquet"),
                size_bytes: 1,
                row_count: 1,
                created_at: Utc::now(),
                min_key,
                max_key,
                min_lsn: max_lsn,
                max_lsn,
                schema_version_min: 1,
                schema_version_max: 1,
                property_stats: vec![],
                kind_specific: KindSpecificStats::Nodes { tombstone_count: 0 },
                bloom: None,
                unique_property_indices: vec![],
                equality_property_indices: vec![],
                label_index: label_counts.map(|per_label_counts| LabelIndexDescriptor {
                    path: "labels.bin".into(),
                    size_bytes: 1,
                    label_count: per_label_counts.len() as u64,
                    posting_count: per_label_counts.iter().map(|(_, n)| *n).sum(),
                    format: PropertyIndexFormat::BincodeV0,
                    per_label_counts,
                }),
                node_locator: None,
                per_label_property_stats: vec![],
            }
        }

        let store = make_store();
        let paths = make_paths("index-freshness-label-range");
        let ms = ManifestStore::new(store.clone(), paths.clone());
        let mut base = ms.bootstrap(Uuid::now_v7()).await.unwrap();
        let note_label = base.manifest.label_dict.intern("Note").get();
        let other_label = base.manifest.label_dict.intern("Other").get();
        base.manifest.vector_indexes.push(VectorIndexDescriptor {
            name: "note_vec".into(),
            label: "Note".into(),
            property: "embedding".into(),
            dim: 4,
            metric: VectorMetric::Cosine,
            r: 16,
            l_build: 32,
            alpha: 1.2,
            quantization: VectorQuantization::None,
        });
        base.manifest.text_indexes.push(TextIndexDescriptor::new(
            "note_ft".into(),
            "Note".into(),
            vec!["body".into()],
        ));
        let member_min = *sorted_node_id(10).as_bytes();
        let member_max = *sorted_node_id(20).as_bytes();
        base.manifest.ssts.extend([
            index_sst(SstKind::VectorGraph, "note_vec", member_min, member_max, 10),
            index_sst(SstKind::TextIndex, "note_ft", member_min, member_max, 10),
        ]);

        let empty = Memtable::new();
        let empty_view = empty.snapshot_view();
        let is_fresh = |loaded: LoadedManifest, name: &str, kind: SstKind| {
            // This test intentionally mutates cloned manifest fixtures. A
            // production LoadedManifest is immutable and every commit/load
            // rebuilds its derived descriptor index; mirror that boundary
            // before constructing the snapshot.
            let loaded = LoadedManifest::new(
                loaded.pointer,
                loaded.pointer_etag,
                loaded.pointer_version,
                loaded.manifest,
            );
            let snap = Snapshot::new(loaded, &empty_view, store.clone(), paths.clone());
            !snap.index_outrun_by_nodes(name, kind)
        };
        let assert_both = |loaded: LoadedManifest, expected_fresh: bool| {
            assert_eq!(
                is_fresh(loaded.clone(), "note_vec", SstKind::VectorGraph),
                expected_fresh
            );
            assert_eq!(
                is_fresh(loaded, "note_ft", SstKind::TextIndex),
                expected_fresh
            );
        };

        // A new, unrelated-label UUIDv7 range is disjoint from every index
        // member range: it cannot add a Note or remove/relabel an indexed id.
        let mut unrelated = base.clone();
        unrelated.manifest.ssts.push(node_sst(
            *sorted_node_id(30).as_bytes(),
            *sorted_node_id(30).as_bytes(),
            20,
            Some(vec![(other_label, 1)]),
        ));
        assert_both(unrelated, true);

        // A live write carrying the indexed label is always dirty, even when
        // its id range is disjoint (new Note document).
        let mut same_label = base.clone();
        same_label.manifest.ssts.push(node_sst(
            *sorted_node_id(30).as_bytes(),
            *sorted_node_id(30).as_bytes(),
            20,
            Some(vec![(note_label, 1)]),
        ));
        assert_both(same_label, false);

        // Relabel/update of an existing member carries only Other now. Its id
        // still overlaps the member range, so the index cannot serve stale.
        let mut relabel = base.clone();
        relabel.manifest.ssts.push(node_sst(
            *sorted_node_id(15).as_bytes(),
            *sorted_node_id(15).as_bytes(),
            20,
            Some(vec![(other_label, 1)]),
        ));
        assert_both(relabel, false);

        // Even a disjoint range cannot be label-scoped when the Nodes SST has
        // no label-index metadata: it might carry a new Note row.
        let mut unknown_labels = base.clone();
        unknown_labels.manifest.ssts.push(node_sst(
            *sorted_node_id(30).as_bytes(),
            *sorted_node_id(30).as_bytes(),
            20,
            None,
        ));
        assert_both(unknown_labels, false);

        // Tombstone-only SSTs have no live label-index sidecar. Absence of that
        // proof remains conservative and forces fallback.
        let mut tombstone = base.clone();
        tombstone.manifest.ssts.push(node_sst(
            *sorted_node_id(15).as_bytes(),
            *sorted_node_id(15).as_bytes(),
            20,
            None,
        ));
        assert_both(tombstone, false);

        // Legacy index descriptors used 00..FF, so an unrelated SST can never
        // prove range disjointness until authoritative recompaction rebuilds it.
        let mut legacy = base.clone();
        for desc in &mut legacy.manifest.ssts {
            desc.min_key = [0u8; 16];
            desc.max_key = [0xFFu8; 16];
        }
        legacy.manifest.ssts.push(node_sst(
            *sorted_node_id(30).as_bytes(),
            *sorted_node_id(30).as_bytes(),
            20,
            Some(vec![(other_label, 1)]),
        ));
        assert_both(legacy, false);

        // Multiple index SSTs: a fresh descriptor's LSN must not mask a change
        // that is newer than and overlaps the older descriptor.
        let mut multiple = base.clone();
        multiple.manifest.ssts.push(index_sst(
            SstKind::VectorGraph,
            "note_vec",
            *sorted_node_id(40).as_bytes(),
            *sorted_node_id(50).as_bytes(),
            30,
        ));
        multiple.manifest.ssts.push(index_sst(
            SstKind::TextIndex,
            "note_ft",
            *sorted_node_id(40).as_bytes(),
            *sorted_node_id(50).as_bytes(),
            30,
        ));
        multiple.manifest.ssts.push(node_sst(
            *sorted_node_id(15).as_bytes(),
            *sorted_node_id(15).as_bytes(),
            20,
            Some(vec![(other_label, 1)]),
        ));
        let anomalous = LoadedManifest::new(
            multiple.pointer.clone(),
            multiple.pointer_etag.clone(),
            multiple.pointer_version.clone(),
            multiple.manifest.clone(),
        );
        let anomaly_snap = Snapshot::new(anomalous, &empty_view, store.clone(), paths.clone());
        assert!(
            anomaly_snap
                .try_vector_search("note_vec", &[0.0; 4], 1, 8)
                .await
                .unwrap()
                .is_none(),
            "multiple vector generations must force the exact fallback"
        );
        assert!(
            anomaly_snap
                .text_search(
                    "note_ft",
                    "Note",
                    &crate::text::parse_query("hello"),
                    Some(1),
                )
                .await
                .unwrap()
                .is_none(),
            "multiple BM25 generations must not mix corpus-local scores"
        );
        assert_both(multiple, false);
    }

    #[tokio::test]
    async fn unique_sidecar_confirms_all_claimants_instead_of_using_sst_max_lsn() {
        // Regression: the sidecar stores value→NodeId but no per-posting LSN.
        // Treating desc.max_lsn as the posting's LSN lets an unrelated new row
        // in SST #1 make its OLD "shared" claimant beat the true reassignment
        // in SST #2. Reverification then rejects only that stale id and used to
        // return None without ever checking the live new owner.
        let store = make_store();
        let paths = make_paths("unique-sidecar-posting-lsn");
        let ms = ManifestStore::new(store.clone(), paths.clone());
        let mut base = ms.bootstrap(Uuid::now_v7()).await.unwrap();
        let account_lid = base.manifest.label_dict.intern("Account").get();
        let fence = WriterFence::new(base.manifest.epoch);
        let schema = SchemaBuilder::new().build();

        let old_owner = sorted_node_id(1);
        let unrelated = sorted_node_id(2);
        let new_owner = sorted_node_id(3);

        // SST #1: the interesting posting is old (LSN 10), but an unrelated
        // row raises the descriptor max_lsn to 100.
        let mut first = Memtable::new();
        first.apply(
            MemKey::Node { id: old_owner },
            10,
            MemOp::Upsert(coded_node_payload("code", "shared", account_lid)),
        );
        first.apply(
            MemKey::Node { id: unrelated },
            100,
            MemOp::Upsert(coded_node_payload("code", "other", account_lid)),
        );
        let out1 = flush(&ms, &fence, &base, &first.freeze(), schema.clone())
            .await
            .unwrap();

        // SST #2: release "shared" from the old id, then assign it to the new
        // id. Both row LSNs are newer than the old posting, but the descriptor's
        // max (90) is lower than SST #1's unrelated max (100).
        let mut second = Memtable::new();
        second.apply(
            MemKey::Node { id: old_owner },
            80,
            MemOp::Upsert(coded_node_payload("code", "renamed", account_lid)),
        );
        second.apply(
            MemKey::Node { id: new_owner },
            90,
            MemOp::Upsert(coded_node_payload("code", "shared", account_lid)),
        );
        let out2 = flush(&ms, &fence, &out1.committed, &second.freeze(), schema)
            .await
            .unwrap();
        let mut committed = out2.committed;

        let first_sst = committed
            .manifest
            .ssts
            .iter()
            .position(|d| d.kind == SstKind::Nodes && d.max_lsn == 100)
            .expect("first node SST");
        let second_sst = committed
            .manifest
            .ssts
            .iter()
            .position(|d| d.kind == SstKind::Nodes && d.max_lsn == 90)
            .expect("second node SST");

        for (ordinal, sst_idx, entries) in [
            (
                1u8,
                first_sst,
                vec![
                    ("shared".to_string(), *old_owner.as_bytes()),
                    ("other".to_string(), *unrelated.as_bytes()),
                ],
            ),
            (
                2u8,
                second_sst,
                vec![
                    ("renamed".to_string(), *old_owner.as_bytes()),
                    ("shared".to_string(), *new_owner.as_bytes()),
                ],
            ),
        ] {
            let map: BTreeMap<String, [u8; 16]> = entries.into_iter().collect();
            let body = Bytes::from(bincode::serialize(&map).unwrap());
            let relative = format!("sst/L0/fabricated-{ordinal}.idx_code.bin");
            let absolute = format!("{}/{}", paths.namespace_prefix().as_ref(), relative);
            store
                .put(
                    &object_store::path::Path::from(absolute),
                    body.clone().into(),
                )
                .await
                .unwrap();
            committed.manifest.ssts[sst_idx]
                .unique_property_indices
                .push(crate::manifest::UniquePropertyIndexDescriptor {
                    property: "code".into(),
                    path: relative,
                    size_bytes: body.len() as u64,
                    entry_count: map.len() as u64,
                    format: crate::manifest::PropertyIndexFormat::BincodeV0,
                    paged: None,
                    paged_build_unsupported: false,
                });
        }

        let empty = Memtable::new();
        let empty_view = empty.snapshot_view();
        let snap = Snapshot::new(committed, &empty_view, store, paths);
        let hit = snap
            .lookup_node_by_property("Account", "code", "shared")
            .await
            .unwrap()
            .expect("the reassigned value must resolve");
        assert_eq!(hit.id, new_owner);
        assert_eq!(hit.lsn, 90);
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
        committed.manifest.ssts[widget_sst]
            .equality_property_indices
            .push(crate::manifest::EqualityIndexDescriptor {
                property: "city".into(),
                path: "sst/L0/does-not-exist.eqidx_city.bin".into(),
                size_bytes: 1,
                distinct_values: 1,
                key_encoding: crate::manifest::EqualityKeyEncoding::StringV0,
                mixed_type_complete: true,
                format: crate::manifest::PropertyIndexFormat::BincodeV0,
                paged: None,
                paged_build_unsupported: false,
            });

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
