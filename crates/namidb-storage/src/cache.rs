//! In-memory cache for SST body + bloom side-car bytes, backed by
//! [`foyer::Cache`].
//!
//! ## Why this lives here
//!
//! The read path issues one `object_store::get` per SST candidate after
//! the manifest/min-key/bloom filter triage. For "warm" workloads where
//! the working set fits in RAM that is wasted latency — every request
//! pays at least one round-trip to S3 (or the local InMemory store).
//! Threading every body through a process-wide cache turns warm reads
//! into a single `Arc::clone()`.
//!
//! ## Scope (v0)
//!
//! - Memory tier only. Foyer's `HybridCache` with a disk back end is a
//! planned follow-up alongside the buffer pool RFC.
//! - Keys are full absolute object-store paths (a `String`). That avoids
//! any normalisation work on the hot path and matches what
//! `Snapshot::get_sst_body` already constructs.
//! - Eviction policy is `S3FifoConfig` (the foyer default).
//! - Weight is `key.len() + value.len()` so the cache obeys a real-byte
//! budget rather than an entry count.
//! - Legacy per-tier budgets are scaled under the process-wide
//! `NAMIDB_CACHE_MAX_BYTES` ceiling before the shared cache is constructed.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use arrow_array::RecordBatch;
use bytes::Bytes;
use foyer::{Cache, CacheBuilder};
use parquet::file::metadata::ParquetMetaData;

use crate::cache_budget::{legacy_budget_bytes, shared_cache_capacities, CacheCapacities};
use crate::sst::bloom::BloomFilter;

/// Default budget for an [`SstCache`]: 256 MiB. Override via
/// `NAMIDB_SST_CACHE_BUDGET_MIB`.
pub const DEFAULT_SST_CACHE_BUDGET_MIB: usize = 256;

/// Default budget for the decoded node row-group cache: 256 MiB.
/// Override via `NAMIDB_DECODED_NODE_RG_CACHE_BUDGET_MIB`. Decoded
/// `RecordBatch`es are typically several times their on-disk size, so
/// this tier gets its own budget rather than sharing the body budget.
pub const DEFAULT_DECODED_NODE_RG_CACHE_BUDGET_MIB: usize = 256;

/// Default budget for decoded property sidecars: 512 MiB process-wide.
///
/// A global legal-graph key index can contain millions of strings and is far
/// larger decoded than on wire. Keeping this tier unbounded made a long L0
/// backlog retain one full BTreeMap per SST. Override via
/// `NAMIDB_PROPERTY_SIDECAR_CACHE_BUDGET_MIB`.
pub const DEFAULT_PROPERTY_SIDECAR_CACHE_BUDGET_MIB: usize = 512;

/// Default decoded Parquet metadata budget. `ParquetMetaData::memory_size`
/// provides exact heap accounting (apart from allocator overhead).
pub const DEFAULT_SST_METADATA_CACHE_BUDGET_MIB: usize = 64;

/// Default decoded edge property-stream budget. Strings are charged by
/// allocation capacity, not length, so a sparse/high-capacity stream cannot
/// escape the limit.
pub const DEFAULT_EDGE_STREAM_CACHE_BUDGET_MIB: usize = 256;

/// Default parsed edge-reader budget. Readers retain the SST body plus their
/// cumulative-offset/fence structures, so this tier is independent from the
/// raw body cache.
pub const DEFAULT_EDGE_READER_CACHE_BUDGET_MIB: usize = 256;

/// Default parsed bloom-filter budget.
pub const DEFAULT_BLOOM_FILTER_CACHE_BUDGET_MIB: usize = 64;

/// Default decoded full-text-index budget.
pub const DEFAULT_TEXT_INDEX_CACHE_BUDGET_MIB: usize = 512;

/// Default decoded vector-index budget. A decoded graph retains both original
/// vectors and a navigation-space copy, so it receives a larger default.
pub const DEFAULT_VECTOR_INDEX_CACHE_BUDGET_MIB: usize = 512;

/// Read `NAMIDB_DECODED_NODE_RG_CACHE_BUDGET_MIB` or fall back to
/// [`DEFAULT_DECODED_NODE_RG_CACHE_BUDGET_MIB`].
pub fn decoded_node_rg_cache_budget_bytes() -> usize {
    legacy_budget_bytes(
        "NAMIDB_DECODED_NODE_RG_CACHE_BUDGET_MIB",
        DEFAULT_DECODED_NODE_RG_CACHE_BUDGET_MIB,
    )
}

pub fn property_sidecar_cache_budget_bytes() -> usize {
    legacy_budget_bytes(
        "NAMIDB_PROPERTY_SIDECAR_CACHE_BUDGET_MIB",
        DEFAULT_PROPERTY_SIDECAR_CACHE_BUDGET_MIB,
    )
}

/// Key for one decoded node-SST row group: `(absolute SST path,
/// row-group index)`.
pub type NodeRowGroupKey = (String, usize);
/// Decoded batches of one node-SST row group.
pub type DecodedNodeRowGroup = Arc<Vec<RecordBatch>>;

/// Decoded unique-property sidecar (`value -> NodeId bytes`).
///
/// The bincode wire format is intentionally unchanged. Keeping the decoded
/// map next to the immutable sidecar path avoids deserialising every entry for
/// every point lookup.
pub type UniquePropertySidecar = BTreeMap<String, [u8; 16]>;

/// Decoded non-unique equality sidecar (`value -> posting list`).
pub type EqualityPropertySidecar = BTreeMap<String, Vec<[u8; 16]>>;

#[derive(Debug, Clone)]
enum DecodedPropertySidecar {
    Unique(Arc<UniquePropertySidecar>),
    Equality(Arc<EqualityPropertySidecar>),
}

fn decoded_property_sidecar_weight(key: &str, value: &DecodedPropertySidecar) -> usize {
    // BTree node/allocator overhead is implementation-specific. Ninety-six
    // bytes per distinct value is deliberately conservative so the configured
    // budget remains a real upper bound rather than just payload bytes.
    const ENTRY_OVERHEAD: usize = 96;
    let payload = match value {
        DecodedPropertySidecar::Unique(index) => index.iter().fold(0usize, |total, (value, _)| {
            total
                .saturating_add(value.capacity())
                .saturating_add(16)
                .saturating_add(ENTRY_OVERHEAD)
        }),
        DecodedPropertySidecar::Equality(index) => {
            index.iter().fold(0usize, |total, (value, ids)| {
                total
                    .saturating_add(value.capacity())
                    .saturating_add(
                        ids.capacity()
                            .saturating_mul(std::mem::size_of::<[u8; 16]>()),
                    )
                    .saturating_add(ENTRY_OVERHEAD)
            })
        }
    };
    key.len().saturating_add(payload)
}

fn edge_stream_bundle_weight(key: &str, bundle: &Arc<EdgeStreamBundle>) -> usize {
    fn strings_weight(values: &[Option<String>]) -> usize {
        values
            .len()
            .saturating_mul(std::mem::size_of::<Option<String>>())
            .saturating_add(values.iter().fold(0usize, |total, value| {
                total.saturating_add(value.as_ref().map_or(0, String::capacity))
            }))
    }

    let overflow = bundle
        .overflow
        .as_deref()
        .map(strings_weight)
        .unwrap_or_default();
    let declared = bundle.declared.iter().fold(
        bundle
            .declared
            .capacity()
            .saturating_mul(std::mem::size_of::<(String, Vec<Option<String>>)>()),
        |total, (name, values)| {
            total
                .saturating_add(name.capacity())
                .saturating_add(strings_weight(values))
        },
    );
    key.len()
        .saturating_add(std::mem::size_of::<EdgeStreamBundle>())
        .saturating_add(overflow)
        .saturating_add(declared)
}

fn edge_reader_weight(key: &str, reader: &Arc<crate::sst::edges::EdgeSstReader>) -> usize {
    // EdgeSstReader intentionally keeps its `Bytes` body private. The final
    // section end is a safe lower bound for that retained allocation; charge
    // the footer/table and parsed cumulative/fence structures on top. These
    // estimates intentionally overcharge normal files so budget eviction is
    // preferable to an unbounded resident graph.
    let footer = reader.footer();
    let body_end = footer
        .sections
        .iter()
        .map(|section| section.offset.saturating_add(section.length))
        .max()
        .unwrap_or_default();
    let body_bytes = usize::try_from(body_end).unwrap_or(usize::MAX);
    let key_count = usize::try_from(reader.key_count()).unwrap_or(usize::MAX);
    let parsed = key_count
        .saturating_add(1)
        .saturating_mul(16)
        .saturating_add(footer.sections.len().saturating_mul(160))
        .saturating_add(4096);
    key.len().saturating_add(body_bytes).saturating_add(parsed)
}

fn bloom_filter_weight(key: &str, filter: &Arc<BloomFilter>) -> usize {
    key.len()
        .saturating_add(
            (filter.block_count() as usize).saturating_mul(8 * std::mem::size_of::<u32>()),
        )
        .saturating_add(std::mem::size_of::<BloomFilter>())
}

#[cfg(any(feature = "text-index", feature = "vector-index"))]
struct WeightedArc<T> {
    value: Arc<T>,
    estimated_bytes: usize,
}

#[cfg(any(feature = "text-index", feature = "vector-index"))]
fn weighted_arc_weight<T>(key: &str, value: &WeightedArc<T>) -> usize {
    key.len().saturating_add(value.estimated_bytes)
}

#[derive(Debug, Clone, Copy)]
struct DecodedCacheBudgets {
    metadata_bytes: usize,
    edge_stream_bytes: usize,
    edge_reader_bytes: usize,
    bloom_bytes: usize,
    #[cfg(feature = "text-index")]
    text_index_bytes: usize,
    #[cfg(feature = "vector-index")]
    vector_index_bytes: usize,
}

#[derive(Debug, Clone, Copy)]
struct SstCacheBudgets {
    body_bytes: usize,
    node_row_group_bytes: usize,
    property_sidecar_bytes: usize,
    decoded: DecodedCacheBudgets,
}

impl SstCacheBudgets {
    fn aggregate_capacity_bytes(self) -> usize {
        let base = self
            .body_bytes
            .saturating_add(self.node_row_group_bytes)
            .saturating_add(self.property_sidecar_bytes)
            .saturating_add(self.decoded.metadata_bytes)
            .saturating_add(self.decoded.edge_stream_bytes)
            .saturating_add(self.decoded.edge_reader_bytes)
            .saturating_add(self.decoded.bloom_bytes);
        #[cfg(feature = "text-index")]
        let base = base.saturating_add(self.decoded.text_index_bytes);
        #[cfg(feature = "vector-index")]
        let base = base.saturating_add(self.decoded.vector_index_bytes);
        base
    }
}

impl DecodedCacheBudgets {
    fn from_env() -> Self {
        Self {
            metadata_bytes: legacy_budget_bytes(
                "NAMIDB_SST_METADATA_CACHE_BUDGET_MIB",
                DEFAULT_SST_METADATA_CACHE_BUDGET_MIB,
            ),
            edge_stream_bytes: legacy_budget_bytes(
                "NAMIDB_EDGE_STREAM_CACHE_BUDGET_MIB",
                DEFAULT_EDGE_STREAM_CACHE_BUDGET_MIB,
            ),
            edge_reader_bytes: legacy_budget_bytes(
                "NAMIDB_EDGE_READER_CACHE_BUDGET_MIB",
                DEFAULT_EDGE_READER_CACHE_BUDGET_MIB,
            ),
            bloom_bytes: legacy_budget_bytes(
                "NAMIDB_BLOOM_FILTER_CACHE_BUDGET_MIB",
                DEFAULT_BLOOM_FILTER_CACHE_BUDGET_MIB,
            ),
            #[cfg(feature = "text-index")]
            text_index_bytes: legacy_budget_bytes(
                "NAMIDB_TEXT_INDEX_CACHE_BUDGET_MIB",
                DEFAULT_TEXT_INDEX_CACHE_BUDGET_MIB,
            ),
            #[cfg(feature = "vector-index")]
            vector_index_bytes: legacy_budget_bytes(
                "NAMIDB_VECTOR_INDEX_CACHE_BUDGET_MIB",
                DEFAULT_VECTOR_INDEX_CACHE_BUDGET_MIB,
            ),
        }
    }

    fn from_capacities(capacities: CacheCapacities) -> Self {
        Self {
            metadata_bytes: capacities.sst_metadata_bytes,
            edge_stream_bytes: capacities.edge_stream_bytes,
            edge_reader_bytes: capacities.edge_reader_bytes,
            bloom_bytes: capacities.bloom_filter_bytes,
            #[cfg(feature = "text-index")]
            text_index_bytes: capacities.text_index_bytes,
            #[cfg(feature = "vector-index")]
            vector_index_bytes: capacities.vector_index_bytes,
        }
    }

    #[cfg(test)]
    fn uniform(bytes: usize) -> Self {
        Self {
            metadata_bytes: bytes,
            edge_stream_bytes: bytes,
            edge_reader_bytes: bytes,
            bloom_bytes: bytes,
            #[cfg(feature = "text-index")]
            text_index_bytes: bytes,
            #[cfg(feature = "vector-index")]
            vector_index_bytes: bytes,
        }
    }
}

/// Weight of one decoded node row-group entry: key bytes plus the
/// Arrow-reported memory footprint of every decoded batch. Shared with
/// tests so budget assertions use the exact accounting the cache does.
pub(crate) fn decoded_node_row_group_weight(
    key: &NodeRowGroupKey,
    value: &DecodedNodeRowGroup,
) -> usize {
    key.0
        .len()
        .saturating_add(std::mem::size_of::<usize>())
        .saturating_add(value.iter().fold(0usize, |sum, batch| {
            sum.saturating_add(batch.get_array_memory_size())
        }))
}

fn raw_body_weight(key: &str, value: &Bytes) -> usize {
    key.len().saturating_add(value.len())
}

fn metadata_weight(key: &str, value: &Arc<ParquetMetaData>) -> usize {
    key.len().saturating_add(value.memory_size())
}

fn fits_capacity(capacity_bytes: usize, weight: usize) -> bool {
    capacity_bytes > 0 && weight <= capacity_bytes
}

/// Read `NAMIDB_SST_CACHE` and return `false` only for `"0"`. Default
/// flipped to ON — the cross-snapshot edge property stream
/// cache lives here, so attaching the cache on every snapshot is now
/// the desirable default.
pub fn sst_cache_enabled() -> bool {
    std::env::var("NAMIDB_SST_CACHE")
        .map(|v| v != "0")
        .unwrap_or(true)
}

/// Read `NAMIDB_SST_CACHE_BUDGET_MIB` or fall back to
/// [`DEFAULT_SST_CACHE_BUDGET_MIB`].
pub fn sst_cache_budget_bytes() -> usize {
    legacy_budget_bytes("NAMIDB_SST_CACHE_BUDGET_MIB", DEFAULT_SST_CACHE_BUDGET_MIB)
}

/// Process-wide shared [`SstCache`]: one instance for every
/// [`crate::WriterSession`] the process opens. Legacy per-tier ceilings are
/// proportionally scaled under `NAMIDB_CACHE_MAX_BYTES`, so a multi-tenant
/// host serving N namespaces holds one aggregate budget, not N.
///
/// Sharing across namespaces is sound because every key in every tier is
/// an absolute object-store path (namespace-prefixed) or `(absolute path,
/// row-group index)`: two namespaces can never collide on a key.
///
/// The enable flag and budgets are read once, on first use; later env
/// mutations don't resize the shared instance. Returns `None` when
/// `NAMIDB_SST_CACHE=0` or `NAMIDB_CACHE_MAX_BYTES=0` at first use. Callers
/// needing private budgets
/// (tests, embedded hosts with several object stores) construct their own
/// [`SstCache`] and inject it via
/// [`crate::ingest::WriterSession::open_with_caches`].
pub fn shared_sst_cache() -> Option<SstCache> {
    static SHARED: OnceLock<Option<SstCache>> = OnceLock::new();
    SHARED
        .get_or_init(|| {
            let capacities = shared_cache_capacities();
            (capacities.sst_capacity_bytes() > 0)
                .then(|| SstCache::with_shared_capacities(capacities))
        })
        .clone()
}

/// Decoded edge SST property streams — the overflow JSON column plus
/// every declared property column.
///
/// Cached per SST absolute path so the per-call `O(edge_count)` decode
/// of [`crate::sst::edges::reader::EdgeSstReader::read_overflow_strings`]
/// and [`crate::sst::edges::reader::EdgeSstReader::read_declared_property_strings`]
/// only happens once per SST per process. Bundled together so the cache
/// lookup is one map probe — the two streams are always read together
/// on the hot path (`edge_lookup_via_sst`).
#[derive(Debug, Clone)]
pub struct EdgeStreamBundle {
    pub overflow: Option<Vec<Option<String>>>,
    pub declared: Vec<(String, Vec<Option<String>>)>,
}

/// Every path-bearing tier is tracked so manifest commits can eagerly evict
/// dead immutable objects. The registry also records the latest live set per
/// namespace: a decode that started from an old pinned snapshot before a prune
/// cannot race the prune and reinsert an already-obsolete body afterwards.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum TrackedCacheEntry {
    Body(String),
    NodeRowGroup(String, usize),
    Metadata(String),
    EdgeStreams(String),
    EdgeReader(String),
    PropertySidecar(String),
    Bloom(String),
    #[cfg(feature = "text-index")]
    TextIndex(String),
    #[cfg(feature = "vector-index")]
    VectorIndex(String),
}

impl TrackedCacheEntry {
    fn path(&self) -> &str {
        match self {
            Self::Body(path)
            | Self::Metadata(path)
            | Self::EdgeStreams(path)
            | Self::EdgeReader(path)
            | Self::PropertySidecar(path)
            | Self::Bloom(path) => path,
            Self::NodeRowGroup(path, _) => path,
            #[cfg(feature = "text-index")]
            Self::TextIndex(path) => path,
            #[cfg(feature = "vector-index")]
            Self::VectorIndex(path) => path,
        }
    }
}

#[derive(Debug, Default)]
struct CachePathRegistry {
    entries: HashSet<TrackedCacheEntry>,
    /// Normalized namespace prefix (always trailing `/`) -> absolute live
    /// object paths from the latest manifest this process observed.
    live_by_namespace: HashMap<String, HashSet<String>>,
}

impl CachePathRegistry {
    fn admits(&self, path: &str) -> bool {
        // Namespace prefixes should not overlap, but choosing the longest
        // match makes the rule deterministic for custom embedded layouts.
        self.live_by_namespace
            .iter()
            .filter(|(prefix, _)| path.starts_with(prefix.as_str()))
            .max_by_key(|(prefix, _)| prefix.len())
            .is_none_or(|(_, live)| live.contains(path))
    }
}

fn normalized_namespace_prefix(namespace_prefix: &str) -> String {
    let mut prefix = namespace_prefix.to_string();
    if !prefix.ends_with('/') {
        prefix.push('/');
    }
    prefix
}

/// Hit/miss counters for diagnostics + cache-integration tests.
#[derive(Debug, Default)]
struct CacheStats {
    hits: AtomicU64,
    misses: AtomicU64,
    inserts: AtomicU64,
    /// Distinct counters for the ranged-read metadata cache (RFC-003).
    /// Lets benches differentiate body-cache vs metadata-cache hits
    /// without dumping the whole `Debug` representation.
    meta_hits: AtomicU64,
    meta_misses: AtomicU64,
    meta_inserts: AtomicU64,
    /// Decoded edge property stream cache counters (added when
    /// IC07 at SF1 surfaced the O(edge_count) decode-per-call cost of
    /// the SST property path).
    edge_streams_hits: AtomicU64,
    edge_streams_misses: AtomicU64,
    edge_streams_inserts: AtomicU64,
    /// Edge SST reader cache counters (S18.B — IC07 at SF10 surfaced the
    /// O(edge_count) `EdgeSstReader::open` cost per call).
    edge_readers_hits: AtomicU64,
    edge_readers_misses: AtomicU64,
    edge_readers_inserts: AtomicU64,
    /// Decoded node row-group cache counters. `inserts` doubles as the
    /// "row groups decoded" probe for the batch-lookup pruning tests.
    node_rg_hits: AtomicU64,
    node_rg_misses: AtomicU64,
    node_rg_inserts: AtomicU64,
    /// Parsed property sidecars. Body bytes are already byte-cached, but
    /// bincode decoding is O(entries), so it needs its own observable tier.
    property_sidecar_hits: AtomicU64,
    property_sidecar_misses: AtomicU64,
    property_sidecar_inserts: AtomicU64,
    /// Parsed bloom filters. Re-validating the checksum and rebuilding all
    /// blocks per negative point probe otherwise turns a sweep into
    /// O(probes × bloom_size × SSTs).
    bloom_hits: AtomicU64,
    bloom_misses: AtomicU64,
    bloom_inserts: AtomicU64,
}

/// Process-wide cache shared between [`crate::Snapshot`] instances.
#[derive(Clone)]
pub struct SstCache {
    /// Logical hard ceilings. Foyer instances use an internal one-byte dummy
    /// capacity for zero-share tiers, while admission consults these values.
    budgets: SstCacheBudgets,
    inner: Arc<Cache<String, Bytes>>,
    /// Decoded node-SST row groups keyed by `(absolute SST path, row-group
    /// index)`. Populated by `Snapshot::batch_lookup_nodes` and consulted by
    /// the per-id lookup cold path so a batch prewarm keeps paying off across
    /// snapshots. Weighted by the decoded Arrow footprint against its own
    /// byte budget (see [`decoded_node_rg_cache_budget_bytes`]); over-eviction
    /// is safe because the read path re-decodes evicted row groups on demand.
    decoded_node_row_groups: Arc<Cache<NodeRowGroupKey, DecodedNodeRowGroup>>,
    /// Parsed Parquet metadata (footer + page index) per SST path, weighted
    /// by `ParquetMetaData::memory_size`.
    metadata: Arc<Cache<String, Arc<ParquetMetaData>>>,
    /// Decoded edge property streams per SST path, weighted by allocated
    /// `Vec` and `String` capacities.
    edge_streams: Arc<Cache<String, Arc<EdgeStreamBundle>>>,
    /// Edge SST readers (header + footer + fence index + precomputed
    /// `cumulative_edges`) keyed by absolute path. `EdgeSstReader::open`
    /// is `O(edge_count)` because it walks every partner block to build
    /// the cumulative-edges prefix sum. Caching the reader makes the
    /// second + every subsequent `edge_lookup_via_sst` against the same
    /// SST run in `O(deg)` instead of `O(edge_count)`. Memory: ~8 B per
    /// edge in the SST.
    edge_readers: Arc<Cache<String, Arc<crate::sst::edges::EdgeSstReader>>>,
    /// Decoded property sidecars keyed by their absolute immutable path.
    /// Unlike the older insert-only maps, this tier is byte-weighted and
    /// process-wide bounded: a backlog cannot retain an unbounded number of
    /// million-key BTreeMaps.
    property_sidecars: Arc<Cache<String, DecodedPropertySidecar>>,
    /// Parsed bloom filters keyed by absolute immutable sidecar path.
    bloom_filters: Arc<Cache<String, Arc<BloomFilter>>>,
    /// Decoded `.ft` text indexes per SST path. Decoding bincode-deserialises
    /// the whole inverted index; without this every `search.bm25` paid
    /// `O(index size)` per query even with the body bytes cached.
    #[cfg(feature = "text-index")]
    text_indexes: Arc<Cache<String, WeightedArc<crate::sst::text::TextIndex>>>,
    /// Decoded `.vg` vector indexes per SST path. Decoding deserialises every
    /// stored vector plus the full Vamana adjacency AND clones the vectors into
    /// the navigation space; without this every KNN (and each widening round)
    /// paid `O(index size)` per query.
    #[cfg(feature = "vector-index")]
    vector_indexes: Arc<Cache<String, WeightedArc<crate::sst::vector::VectorGraphIndex>>>,
    /// Coordinates insertion with manifest pruning. Foyer itself is
    /// thread-safe; this small mutex covers only path admission/bookkeeping,
    /// never decoding or object-store I/O.
    tracked_paths: Arc<Mutex<CachePathRegistry>>,
    stats: Arc<CacheStats>,
}

impl std::fmt::Debug for SstCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // foyer's `Cache` doesn't impl Debug and only exposes
        // `capacity`/`usage`; surface those alongside our own counters.
        let mut out = f.debug_struct("SstCache");
        out.field("capacity_bytes", &self.budgets.body_bytes)
            .field("usage_bytes", &self.inner.usage())
            .field("hits", &self.stats.hits.load(Ordering::Relaxed))
            .field("misses", &self.stats.misses.load(Ordering::Relaxed))
            .field("inserts", &self.stats.inserts.load(Ordering::Relaxed))
            .field("meta_hits", &self.stats.meta_hits.load(Ordering::Relaxed))
            .field(
                "meta_misses",
                &self.stats.meta_misses.load(Ordering::Relaxed),
            )
            .field(
                "meta_inserts",
                &self.stats.meta_inserts.load(Ordering::Relaxed),
            )
            .field(
                "aggregate_capacity_bytes",
                &self.budgets.aggregate_capacity_bytes(),
            )
            .field("aggregate_usage_bytes", &self.aggregate_usage_bytes())
            .field(
                "metadata_capacity_bytes",
                &self.budgets.decoded.metadata_bytes,
            )
            .field("metadata_usage_bytes", &self.metadata.usage())
            .field(
                "edge_stream_capacity_bytes",
                &self.budgets.decoded.edge_stream_bytes,
            )
            .field("edge_stream_usage_bytes", &self.edge_streams.usage())
            .field(
                "edge_reader_capacity_bytes",
                &self.budgets.decoded.edge_reader_bytes,
            )
            .field("edge_reader_usage_bytes", &self.edge_readers.usage())
            .field("node_rg_usage_bytes", &self.decoded_node_row_groups.usage())
            .field(
                "node_rg_hits",
                &self.stats.node_rg_hits.load(Ordering::Relaxed),
            )
            .field(
                "node_rg_misses",
                &self.stats.node_rg_misses.load(Ordering::Relaxed),
            )
            .field(
                "node_rg_inserts",
                &self.stats.node_rg_inserts.load(Ordering::Relaxed),
            )
            .field(
                "property_sidecar_capacity_bytes",
                &self.budgets.property_sidecar_bytes,
            )
            .field(
                "property_sidecar_usage_bytes",
                &self.property_sidecars.usage(),
            )
            .field(
                "property_sidecar_hits",
                &self.stats.property_sidecar_hits.load(Ordering::Relaxed),
            )
            .field(
                "property_sidecar_misses",
                &self.stats.property_sidecar_misses.load(Ordering::Relaxed),
            )
            .field(
                "property_sidecar_inserts",
                &self.stats.property_sidecar_inserts.load(Ordering::Relaxed),
            )
            .field("bloom_capacity_bytes", &self.budgets.decoded.bloom_bytes)
            .field("bloom_usage_bytes", &self.bloom_filters.usage())
            .field("bloom_hits", &self.stats.bloom_hits.load(Ordering::Relaxed))
            .field(
                "bloom_misses",
                &self.stats.bloom_misses.load(Ordering::Relaxed),
            )
            .field(
                "bloom_inserts",
                &self.stats.bloom_inserts.load(Ordering::Relaxed),
            );
        #[cfg(feature = "text-index")]
        out.field(
            "text_index_capacity_bytes",
            &self.budgets.decoded.text_index_bytes,
        )
        .field("text_index_usage_bytes", &self.text_indexes.usage());
        #[cfg(feature = "vector-index")]
        out.field(
            "vector_index_capacity_bytes",
            &self.budgets.decoded.vector_index_bytes,
        )
        .field("vector_index_usage_bytes", &self.vector_indexes.usage());
        out.finish()
    }
}

impl SstCache {
    /// Build a new cache sized for `capacity_bytes`. Entries weight as
    /// `key.len() + value.len()` so the budget is in real bytes. The decoded
    /// node row-group tier gets its own budget from
    /// [`decoded_node_rg_cache_budget_bytes`].
    pub fn new(capacity_bytes: usize) -> Self {
        Self::with_budgets(capacity_bytes, decoded_node_rg_cache_budget_bytes())
    }

    /// Like [`Self::new`] but with an explicit byte budget for the decoded
    /// node row-group tier. Used by tests that need a tight decoded budget
    /// without touching env state.
    pub fn with_budgets(capacity_bytes: usize, decoded_node_rg_bytes: usize) -> Self {
        Self::with_all_budgets(
            capacity_bytes,
            decoded_node_rg_bytes,
            property_sidecar_cache_budget_bytes(),
            DecodedCacheBudgets::from_env(),
        )
    }

    fn with_shared_capacities(capacities: CacheCapacities) -> Self {
        Self::with_all_budgets(
            capacities.sst_body_bytes,
            capacities.decoded_node_row_group_bytes,
            capacities.property_sidecar_bytes,
            DecodedCacheBudgets::from_capacities(capacities),
        )
    }

    fn with_all_budgets(
        capacity_bytes: usize,
        decoded_node_rg_bytes: usize,
        property_sidecar_bytes: usize,
        decoded: DecodedCacheBudgets,
    ) -> Self {
        let budgets = SstCacheBudgets {
            body_bytes: capacity_bytes,
            node_row_group_bytes: decoded_node_rg_bytes,
            property_sidecar_bytes,
            decoded,
        };
        let inner_capacity = capacity_bytes;
        let inner = CacheBuilder::new(inner_capacity.max(1))
            .with_shards(1)
            .with_weighter(|key: &String, value: &Bytes| raw_body_weight(key, value))
            .with_filter(move |key: &String, value: &Bytes| {
                fits_capacity(inner_capacity, raw_body_weight(key, value))
            })
            .build();
        let node_rg_capacity = decoded_node_rg_bytes;
        let decoded_node_row_groups = CacheBuilder::new(node_rg_capacity.max(1))
            .with_shards(1)
            .with_weighter(decoded_node_row_group_weight)
            .with_filter(move |key: &NodeRowGroupKey, value: &DecodedNodeRowGroup| {
                fits_capacity(node_rg_capacity, decoded_node_row_group_weight(key, value))
            })
            .build();
        let property_sidecar_capacity = property_sidecar_bytes;
        let property_sidecars = CacheBuilder::new(property_sidecar_capacity.max(1))
            .with_shards(1)
            .with_weighter(|key: &String, value: &DecodedPropertySidecar| {
                decoded_property_sidecar_weight(key, value)
            })
            .with_filter(move |key: &String, value: &DecodedPropertySidecar| {
                fits_capacity(
                    property_sidecar_capacity,
                    decoded_property_sidecar_weight(key, value),
                )
            })
            .build();
        let metadata_capacity = decoded.metadata_bytes;
        let metadata = CacheBuilder::new(metadata_capacity.max(1))
            .with_shards(1)
            .with_weighter(|key: &String, value: &Arc<ParquetMetaData>| metadata_weight(key, value))
            .with_filter(move |key: &String, value: &Arc<ParquetMetaData>| {
                fits_capacity(metadata_capacity, metadata_weight(key, value))
            })
            .build();
        let edge_stream_capacity = decoded.edge_stream_bytes;
        let edge_streams = CacheBuilder::new(edge_stream_capacity.max(1))
            .with_shards(1)
            .with_weighter(|key: &String, value: &Arc<EdgeStreamBundle>| {
                edge_stream_bundle_weight(key, value)
            })
            .with_filter(move |key: &String, value: &Arc<EdgeStreamBundle>| {
                fits_capacity(edge_stream_capacity, edge_stream_bundle_weight(key, value))
            })
            .build();
        let edge_reader_capacity = decoded.edge_reader_bytes;
        let edge_readers = CacheBuilder::new(edge_reader_capacity.max(1))
            .with_shards(1)
            .with_weighter(
                |key: &String, value: &Arc<crate::sst::edges::EdgeSstReader>| {
                    edge_reader_weight(key, value)
                },
            )
            .with_filter(
                move |key: &String, value: &Arc<crate::sst::edges::EdgeSstReader>| {
                    fits_capacity(edge_reader_capacity, edge_reader_weight(key, value))
                },
            )
            .build();
        let bloom_capacity = decoded.bloom_bytes;
        let bloom_filters = CacheBuilder::new(bloom_capacity.max(1))
            .with_shards(1)
            .with_weighter(|key: &String, value: &Arc<BloomFilter>| bloom_filter_weight(key, value))
            .with_filter(move |key: &String, value: &Arc<BloomFilter>| {
                fits_capacity(bloom_capacity, bloom_filter_weight(key, value))
            })
            .build();
        #[cfg(feature = "text-index")]
        let text_index_capacity = decoded.text_index_bytes;
        #[cfg(feature = "text-index")]
        let text_indexes = CacheBuilder::new(text_index_capacity.max(1))
            .with_shards(1)
            .with_weighter(
                |key: &String, value: &WeightedArc<crate::sst::text::TextIndex>| {
                    weighted_arc_weight(key, value)
                },
            )
            .with_filter(
                move |key: &String, value: &WeightedArc<crate::sst::text::TextIndex>| {
                    fits_capacity(text_index_capacity, weighted_arc_weight(key, value))
                },
            )
            .build();
        #[cfg(feature = "vector-index")]
        let vector_index_capacity = decoded.vector_index_bytes;
        #[cfg(feature = "vector-index")]
        let vector_indexes = CacheBuilder::new(vector_index_capacity.max(1))
            .with_shards(1)
            .with_weighter(
                |key: &String, value: &WeightedArc<crate::sst::vector::VectorGraphIndex>| {
                    weighted_arc_weight(key, value)
                },
            )
            .with_filter(
                move |key: &String, value: &WeightedArc<crate::sst::vector::VectorGraphIndex>| {
                    fits_capacity(vector_index_capacity, weighted_arc_weight(key, value))
                },
            )
            .build();
        Self {
            budgets,
            inner: Arc::new(inner),
            decoded_node_row_groups: Arc::new(decoded_node_row_groups),
            metadata: Arc::new(metadata),
            edge_streams: Arc::new(edge_streams),
            edge_readers: Arc::new(edge_readers),
            property_sidecars: Arc::new(property_sidecars),
            bloom_filters: Arc::new(bloom_filters),
            #[cfg(feature = "text-index")]
            text_indexes: Arc::new(text_indexes),
            #[cfg(feature = "vector-index")]
            vector_indexes: Arc::new(vector_indexes),
            tracked_paths: Arc::new(Mutex::new(CachePathRegistry::default())),
            stats: Arc::new(CacheStats::default()),
        }
    }

    /// Run `insert` while holding the path-admission mutex. A stale pinned
    /// snapshot that finishes decoding after a manifest prune is rejected
    /// here, closing the old `paths.insert(); cache.insert()` race.
    fn insert_tracked(&self, entry: TrackedCacheEntry, insert: impl FnOnce()) {
        let mut paths = self.tracked_paths.lock().unwrap();
        if paths.admits(entry.path()) {
            insert();
            paths.entries.insert(entry);
        }
    }

    /// Look up a decoded text index for an SST path. Returns `None` on miss;
    /// the caller decodes once and re-inserts via [`Self::insert_text_index`].
    /// SSTs are immutable per UUIDv7-keyed path so cached indexes never go
    /// stale; superseded paths are pruned by [`Self::retain_paths`].
    #[cfg(feature = "text-index")]
    pub fn get_text_index(&self, key: &str) -> Option<Arc<crate::sst::text::TextIndex>> {
        self.text_indexes
            .get(key)
            .map(|entry| entry.value().value.clone())
    }

    /// Cheap preflight for a serialized text-index body.
    ///
    /// Call this with the manifest/object length before downloading or
    /// decoding a monolithic index. `false` means the decoded representation's
    /// six-wire-copy lower estimate cannot fit and the exact flat fallback
    /// should be used instead.
    #[cfg(feature = "text-index")]
    pub fn can_admit_text_index_wire_bytes(&self, key: &str, wire_bytes: usize) -> bool {
        fits_capacity(
            self.budgets.decoded.text_index_bytes,
            key.len().saturating_add(wire_bytes.saturating_mul(6)),
        )
    }

    /// Store a decoded text index for an SST path.
    #[cfg(feature = "text-index")]
    pub fn insert_text_index(&self, key: String, idx: Arc<crate::sst::text::TextIndex>) {
        let wire_bytes = self
            .inner
            .get(&key)
            .map(|entry| entry.value().len())
            .unwrap_or_default();
        self.insert_text_index_with_wire_bytes(key, idx, wire_bytes);
    }

    /// Store a decoded text index, using the serialized body length supplied
    /// by the caller for conservative deep-memory admission.
    ///
    /// The explicit length is required when the raw body itself exceeded its
    /// tier and was deliberately not cached.
    #[cfg(feature = "text-index")]
    pub fn insert_text_index_with_wire_bytes(
        &self,
        key: String,
        idx: Arc<crate::sst::text::TextIndex>,
        wire_bytes: usize,
    ) {
        // Bincode's wire body is a stable lower bound. A 6x multiplier
        // conservatively covers BTree/posting Vec allocator overhead and the
        // separately sorted id copy. An uncached body falls back to a
        // per-document estimate; either way the tier has a hard byte budget.
        let doc_count = usize::try_from(idx.doc_count()).unwrap_or(usize::MAX);
        let estimated_bytes = wire_bytes
            .saturating_mul(6)
            .max(doc_count.saturating_mul(512))
            .max(std::mem::size_of_val(idx.as_ref()));
        let value = WeightedArc {
            value: idx,
            estimated_bytes,
        };
        if !fits_capacity(
            self.budgets.decoded.text_index_bytes,
            weighted_arc_weight(&key, &value),
        ) {
            return;
        }
        let tracked = TrackedCacheEntry::TextIndex(key.clone());
        self.insert_tracked(tracked, || {
            self.text_indexes.insert(key, value);
        });
    }

    /// Look up a decoded vector index for an SST path. Same contract as
    /// [`Self::get_text_index`].
    #[cfg(feature = "vector-index")]
    pub fn get_vector_index(&self, key: &str) -> Option<Arc<crate::sst::vector::VectorGraphIndex>> {
        self.vector_indexes
            .get(key)
            .map(|entry| entry.value().value.clone())
    }

    /// Preflight a serialized vector-index body before GET/decode.
    #[cfg(feature = "vector-index")]
    pub fn can_admit_vector_index_wire_bytes(&self, key: &str, wire_bytes: usize) -> bool {
        fits_capacity(
            self.budgets.decoded.vector_index_bytes,
            key.len().saturating_add(wire_bytes.saturating_mul(6)),
        )
    }

    /// Store a decoded vector index for an SST path.
    #[cfg(feature = "vector-index")]
    pub fn insert_vector_index(&self, key: String, idx: Arc<crate::sst::vector::VectorGraphIndex>) {
        let wire_bytes = self
            .inner
            .get(&key)
            .map(|entry| entry.value().len())
            .unwrap_or_default();
        self.insert_vector_index_with_wire_bytes(key, idx, wire_bytes);
    }

    /// Store a decoded vector index with an explicit serialized-body length.
    ///
    /// This remains safe when the raw body was too large for its own tier and
    /// therefore never became observable through [`Self::get`].
    #[cfg(feature = "vector-index")]
    pub fn insert_vector_index_with_wire_bytes(
        &self,
        key: String,
        idx: Arc<crate::sst::vector::VectorGraphIndex>,
        wire_bytes: usize,
    ) {
        // Decode retains the serialized graph's vectors/adjacency and builds a
        // second navigation-space vector set. Six wire copies is conservative
        // for normal Vamana degrees; the independent point/dimension estimate
        // protects the no-body-cache path.
        let per_point = (idx.dim() as usize).saturating_mul(8).saturating_add(512);
        let point_count = usize::try_from(idx.point_count()).unwrap_or(usize::MAX);
        let estimated_bytes = wire_bytes
            .saturating_mul(6)
            .max(point_count.saturating_mul(per_point))
            .max(std::mem::size_of_val(idx.as_ref()));
        let value = WeightedArc {
            value: idx,
            estimated_bytes,
        };
        if !fits_capacity(
            self.budgets.decoded.vector_index_bytes,
            weighted_arc_weight(&key, &value),
        ) {
            return;
        }
        let tracked = TrackedCacheEntry::VectorIndex(key.clone());
        self.insert_tracked(tracked, || {
            self.vector_indexes.insert(key, value);
        });
    }

    /// Look up a cached [`crate::sst::edges::EdgeSstReader`] for an SST
    /// path. Returns `None` on miss; the caller calls
    /// [`crate::sst::edges::EdgeSstReader::open`] once and re-inserts
    /// via [`Self::insert_edge_reader`].
    pub fn get_edge_reader(&self, key: &str) -> Option<Arc<crate::sst::edges::EdgeSstReader>> {
        match self.edge_readers.get(key) {
            Some(entry) => {
                self.stats.edge_readers_hits.fetch_add(1, Ordering::Relaxed);
                Some(entry.value().clone())
            }
            None => {
                self.stats
                    .edge_readers_misses
                    .fetch_add(1, Ordering::Relaxed);
                None
            }
        }
    }

    /// Store an [`crate::sst::edges::EdgeSstReader`] for an SST path.
    /// SSTs are immutable per UUIDv7-keyed path so cached readers
    /// never go stale.
    pub fn insert_edge_reader(&self, key: String, reader: Arc<crate::sst::edges::EdgeSstReader>) {
        self.stats
            .edge_readers_inserts
            .fetch_add(1, Ordering::Relaxed);
        if !fits_capacity(
            self.budgets.decoded.edge_reader_bytes,
            edge_reader_weight(&key, &reader),
        ) {
            return;
        }
        let tracked = TrackedCacheEntry::EdgeReader(key.clone());
        self.insert_tracked(tracked, || {
            self.edge_readers.insert(key, reader);
        });
    }

    pub fn edge_readers_hits(&self) -> u64 {
        self.stats.edge_readers_hits.load(Ordering::Relaxed)
    }
    pub fn edge_readers_misses(&self) -> u64 {
        self.stats.edge_readers_misses.load(Ordering::Relaxed)
    }
    pub fn edge_readers_inserts(&self) -> u64 {
        self.stats.edge_readers_inserts.load(Ordering::Relaxed)
    }

    /// Look up a decoded unique-property sidecar by immutable object path.
    pub fn get_unique_property_sidecar(&self, key: &str) -> Option<Arc<UniquePropertySidecar>> {
        match self.property_sidecars.get(key).and_then(|entry| {
            if let DecodedPropertySidecar::Unique(index) = entry.value() {
                Some(index.clone())
            } else {
                None
            }
        }) {
            Some(index) => {
                self.stats
                    .property_sidecar_hits
                    .fetch_add(1, Ordering::Relaxed);
                Some(index)
            }
            None => {
                self.stats
                    .property_sidecar_misses
                    .fetch_add(1, Ordering::Relaxed);
                None
            }
        }
    }

    /// Store a decoded unique-property sidecar.
    pub fn insert_unique_property_sidecar(&self, key: String, index: Arc<UniquePropertySidecar>) {
        self.stats
            .property_sidecar_inserts
            .fetch_add(1, Ordering::Relaxed);
        let value = DecodedPropertySidecar::Unique(index);
        if !fits_capacity(
            self.budgets.property_sidecar_bytes,
            decoded_property_sidecar_weight(&key, &value),
        ) {
            return;
        }
        let tracked = TrackedCacheEntry::PropertySidecar(key.clone());
        self.insert_tracked(tracked, || {
            self.property_sidecars.insert(key, value);
        });
    }

    /// Look up a decoded non-unique equality sidecar.
    pub fn get_equality_property_sidecar(&self, key: &str) -> Option<Arc<EqualityPropertySidecar>> {
        match self.property_sidecars.get(key).and_then(|entry| {
            if let DecodedPropertySidecar::Equality(index) = entry.value() {
                Some(index.clone())
            } else {
                None
            }
        }) {
            Some(index) => {
                self.stats
                    .property_sidecar_hits
                    .fetch_add(1, Ordering::Relaxed);
                Some(index)
            }
            None => {
                self.stats
                    .property_sidecar_misses
                    .fetch_add(1, Ordering::Relaxed);
                None
            }
        }
    }

    /// Store a decoded non-unique equality sidecar.
    pub fn insert_equality_property_sidecar(
        &self,
        key: String,
        index: Arc<EqualityPropertySidecar>,
    ) {
        self.stats
            .property_sidecar_inserts
            .fetch_add(1, Ordering::Relaxed);
        let value = DecodedPropertySidecar::Equality(index);
        if !fits_capacity(
            self.budgets.property_sidecar_bytes,
            decoded_property_sidecar_weight(&key, &value),
        ) {
            return;
        }
        let tracked = TrackedCacheEntry::PropertySidecar(key.clone());
        self.insert_tracked(tracked, || {
            self.property_sidecars.insert(key, value);
        });
    }

    pub fn property_sidecar_hits(&self) -> u64 {
        self.stats.property_sidecar_hits.load(Ordering::Relaxed)
    }

    pub fn property_sidecar_misses(&self) -> u64 {
        self.stats.property_sidecar_misses.load(Ordering::Relaxed)
    }

    pub fn property_sidecar_inserts(&self) -> u64 {
        self.stats.property_sidecar_inserts.load(Ordering::Relaxed)
    }

    pub fn property_sidecar_usage_bytes(&self) -> usize {
        self.property_sidecars.usage()
    }

    pub fn property_sidecar_capacity_bytes(&self) -> usize {
        self.budgets.property_sidecar_bytes
    }

    /// Look up a parsed bloom filter by immutable sidecar path.
    pub fn get_bloom_filter(&self, key: &str) -> Option<Arc<BloomFilter>> {
        match self.bloom_filters.get(key) {
            Some(entry) => {
                self.stats.bloom_hits.fetch_add(1, Ordering::Relaxed);
                Some(entry.value().clone())
            }
            None => {
                self.stats.bloom_misses.fetch_add(1, Ordering::Relaxed);
                None
            }
        }
    }

    /// Store a parsed bloom filter.
    pub fn insert_bloom_filter(&self, key: String, filter: Arc<BloomFilter>) {
        self.stats.bloom_inserts.fetch_add(1, Ordering::Relaxed);
        if !fits_capacity(
            self.budgets.decoded.bloom_bytes,
            bloom_filter_weight(&key, &filter),
        ) {
            return;
        }
        let tracked = TrackedCacheEntry::Bloom(key.clone());
        self.insert_tracked(tracked, || {
            self.bloom_filters.insert(key, filter);
        });
    }

    pub fn bloom_hits(&self) -> u64 {
        self.stats.bloom_hits.load(Ordering::Relaxed)
    }

    pub fn bloom_misses(&self) -> u64 {
        self.stats.bloom_misses.load(Ordering::Relaxed)
    }

    pub fn bloom_inserts(&self) -> u64 {
        self.stats.bloom_inserts.load(Ordering::Relaxed)
    }

    pub fn bloom_usage_bytes(&self) -> usize {
        self.bloom_filters.usage()
    }

    pub fn bloom_capacity_bytes(&self) -> usize {
        self.budgets.decoded.bloom_bytes
    }

    /// Look up the decoded batches for one node-SST row group. Returns
    /// `None` on miss (never cached, or evicted under the byte budget);
    /// the caller decodes the row group and re-inserts via
    /// [`Self::insert_decoded_node_row_group`].
    pub fn get_decoded_node_row_group(
        &self,
        key: &str,
        row_group: usize,
    ) -> Option<Arc<Vec<RecordBatch>>> {
        match self
            .decoded_node_row_groups
            .get(&(key.to_string(), row_group))
        {
            Some(entry) => {
                self.stats.node_rg_hits.fetch_add(1, Ordering::Relaxed);
                Some(entry.value().clone())
            }
            None => {
                self.stats.node_rg_misses.fetch_add(1, Ordering::Relaxed);
                None
            }
        }
    }

    /// Store the decoded batches for one node-SST row group. SSTs are
    /// immutable per UUIDv7-keyed path so cached row groups never go stale.
    pub fn insert_decoded_node_row_group(
        &self,
        key: String,
        row_group: usize,
        batches: Arc<Vec<RecordBatch>>,
    ) {
        self.stats.node_rg_inserts.fetch_add(1, Ordering::Relaxed);
        let cache_key = (key.clone(), row_group);
        if !fits_capacity(
            self.budgets.node_row_group_bytes,
            decoded_node_row_group_weight(&cache_key, &batches),
        ) {
            return;
        }
        let tracked = TrackedCacheEntry::NodeRowGroup(key.clone(), row_group);
        self.insert_tracked(tracked, || {
            self.decoded_node_row_groups.insert(cache_key, batches);
        });
    }

    /// Bytes held by the decoded node row-group tier (sum of entry weights).
    pub fn decoded_node_row_groups_usage(&self) -> usize {
        self.decoded_node_row_groups.usage()
    }

    pub fn decoded_node_row_groups_capacity_bytes(&self) -> usize {
        self.budgets.node_row_group_bytes
    }

    pub fn decoded_node_row_group_hits(&self) -> u64 {
        self.stats.node_rg_hits.load(Ordering::Relaxed)
    }
    pub fn decoded_node_row_group_misses(&self) -> u64 {
        self.stats.node_rg_misses.load(Ordering::Relaxed)
    }
    pub fn decoded_node_row_group_inserts(&self) -> u64 {
        self.stats.node_rg_inserts.load(Ordering::Relaxed)
    }

    /// Look up decoded edge property streams for an SST path.
    /// Returns `None` on miss; the caller decodes + re-inserts via
    /// [`Self::insert_edge_streams`].
    pub fn get_edge_streams(&self, key: &str) -> Option<Arc<EdgeStreamBundle>> {
        match self.edge_streams.get(key) {
            Some(entry) => {
                self.stats.edge_streams_hits.fetch_add(1, Ordering::Relaxed);
                Some(entry.value().clone())
            }
            None => {
                self.stats
                    .edge_streams_misses
                    .fetch_add(1, Ordering::Relaxed);
                None
            }
        }
    }

    /// Store decoded edge property streams for an SST path. SSTs are
    /// immutable per UUIDv7-keyed path so cached streams never go stale.
    pub fn insert_edge_streams(&self, key: String, bundle: Arc<EdgeStreamBundle>) {
        self.stats
            .edge_streams_inserts
            .fetch_add(1, Ordering::Relaxed);
        if !fits_capacity(
            self.budgets.decoded.edge_stream_bytes,
            edge_stream_bundle_weight(&key, &bundle),
        ) {
            return;
        }
        let tracked = TrackedCacheEntry::EdgeStreams(key.clone());
        self.insert_tracked(tracked, || {
            self.edge_streams.insert(key, bundle);
        });
    }

    pub fn edge_streams_hits(&self) -> u64 {
        self.stats.edge_streams_hits.load(Ordering::Relaxed)
    }
    pub fn edge_streams_misses(&self) -> u64 {
        self.stats.edge_streams_misses.load(Ordering::Relaxed)
    }
    pub fn edge_streams_inserts(&self) -> u64 {
        self.stats.edge_streams_inserts.load(Ordering::Relaxed)
    }

    pub fn edge_streams_usage_bytes(&self) -> usize {
        self.edge_streams.usage()
    }

    pub fn edge_streams_capacity_bytes(&self) -> usize {
        self.budgets.decoded.edge_stream_bytes
    }

    /// Look up Parquet metadata for an SST path (RFC-003). Returns
    /// `None` on miss; the ranged-read path will fetch the footer +
    /// page index and re-insert via [`Self::insert_metadata`].
    pub fn get_metadata(&self, key: &str) -> Option<Arc<ParquetMetaData>> {
        match self.metadata.get(key) {
            Some(entry) => {
                self.stats.meta_hits.fetch_add(1, Ordering::Relaxed);
                Some(entry.value().clone())
            }
            None => {
                self.stats.meta_misses.fetch_add(1, Ordering::Relaxed);
                None
            }
        }
    }

    /// Store the Parquet metadata for an SST path. SSTs are immutable
    /// per UUIDv7-keyed path, so cached metadata never goes stale.
    pub fn insert_metadata(&self, key: String, meta: Arc<ParquetMetaData>) {
        self.stats.meta_inserts.fetch_add(1, Ordering::Relaxed);
        if !fits_capacity(
            self.budgets.decoded.metadata_bytes,
            metadata_weight(&key, &meta),
        ) {
            return;
        }
        let tracked = TrackedCacheEntry::Metadata(key.clone());
        self.insert_tracked(tracked, || {
            self.metadata.insert(key, meta);
        });
    }

    fn remove_tracked(&self, entry: &TrackedCacheEntry) {
        match entry {
            TrackedCacheEntry::Body(path) => {
                self.inner.remove(path);
            }
            TrackedCacheEntry::NodeRowGroup(path, row_group) => {
                self.decoded_node_row_groups
                    .remove(&(path.clone(), *row_group));
            }
            TrackedCacheEntry::Metadata(path) => {
                self.metadata.remove(path);
            }
            TrackedCacheEntry::EdgeStreams(path) => {
                self.edge_streams.remove(path);
            }
            TrackedCacheEntry::EdgeReader(path) => {
                self.edge_readers.remove(path);
            }
            TrackedCacheEntry::PropertySidecar(path) => {
                self.property_sidecars.remove(path);
            }
            TrackedCacheEntry::Bloom(path) => {
                self.bloom_filters.remove(path);
            }
            #[cfg(feature = "text-index")]
            TrackedCacheEntry::TextIndex(path) => {
                self.text_indexes.remove(path);
            }
            #[cfg(feature = "vector-index")]
            TrackedCacheEntry::VectorIndex(path) => {
                self.vector_indexes.remove(path);
            }
        }
    }

    fn tracked_entry_present(&self, entry: &TrackedCacheEntry) -> bool {
        match entry {
            TrackedCacheEntry::Body(path) => self.inner.get(path).is_some(),
            TrackedCacheEntry::NodeRowGroup(path, row_group) => self
                .decoded_node_row_groups
                .get(&(path.clone(), *row_group))
                .is_some(),
            TrackedCacheEntry::Metadata(path) => self.metadata.get(path).is_some(),
            TrackedCacheEntry::EdgeStreams(path) => self.edge_streams.get(path).is_some(),
            TrackedCacheEntry::EdgeReader(path) => self.edge_readers.get(path).is_some(),
            TrackedCacheEntry::PropertySidecar(path) => self.property_sidecars.get(path).is_some(),
            TrackedCacheEntry::Bloom(path) => self.bloom_filters.get(path).is_some(),
            #[cfg(feature = "text-index")]
            TrackedCacheEntry::TextIndex(path) => self.text_indexes.get(path).is_some(),
            #[cfg(feature = "vector-index")]
            TrackedCacheEntry::VectorIndex(path) => self.vector_indexes.get(path).is_some(),
        }
    }

    /// Drop every cache entry under `namespace_prefix` whose immutable object
    /// path is no longer `live`. All tiers remain independently byte-bounded;
    /// eager pruning releases dropped/compacted SSTs immediately instead of
    /// waiting for budget pressure.
    ///
    /// The prune is scoped to `namespace_prefix` (`<root>/<ns>`, with or
    /// without a trailing slash) because the cache is shared process-wide:
    /// one namespace's flush knows only its OWN live set, so it must never
    /// touch sibling namespaces' entries — a global retain here would evict
    /// every other tenant's warm state on each flush.
    pub fn retain_paths(&self, namespace_prefix: &str, live: &std::collections::HashSet<String>) {
        self.retain_paths_inner(namespace_prefix, live, false);
    }

    fn retain_paths_inner(
        &self,
        namespace_prefix: &str,
        live: &std::collections::HashSet<String>,
        enforce_empty: bool,
    ) {
        // Normalize to a path-segment boundary so "tenants/acme" cannot
        // match "tenants/acme2/...".
        let prefix = normalized_namespace_prefix(namespace_prefix);
        let mut paths = self.tracked_paths.lock().unwrap();
        // Publish the admission rule before removing entries while holding the
        // same lock used by insertions. A decode from a pre-prune snapshot
        // finishing later sees this rule and cannot resurrect a dead path.
        if live.is_empty() && !enforce_empty {
            // A freshly-opened empty namespace has no previous immutable path
            // that could race this call. Leave it permissive so its first
            // flush can mint UUID paths before the post-commit retain.
            paths.live_by_namespace.remove(&prefix);
        } else {
            paths.live_by_namespace.insert(prefix.clone(), live.clone());
        }
        let stale: Vec<TrackedCacheEntry> = paths
            .entries
            .iter()
            .filter(|entry| entry.path().starts_with(&prefix) && !live.contains(entry.path()))
            .cloned()
            .collect();
        for entry in stale {
            self.remove_tracked(&entry);
            paths.entries.remove(&entry);
        }
    }

    /// Eagerly drop every cache entry under `namespace_prefix`. Called
    /// when a multi-tenant host evicts a namespace — its state is being
    /// dropped anyway, so its bodies/row groups/decoded indexes are dead
    /// weight in the shared cache.
    pub fn prune_namespace(&self, namespace_prefix: &str) {
        self.retain_paths_inner(namespace_prefix, &std::collections::HashSet::new(), true);
    }

    /// Count of resident entries across every tier whose SST path sits under
    /// `namespace_prefix`. Observability + test probe for the
    /// namespace-scoped [`Self::retain_paths`] / [`Self::prune_namespace`].
    pub fn namespace_side_entries(&self, namespace_prefix: &str) -> usize {
        let prefix = normalized_namespace_prefix(namespace_prefix);
        let mut paths = self.tracked_paths.lock().unwrap();
        // Foyer evicts independently. Clean tracking tombstones here so this
        // diagnostic reports resident entries and the registry cannot retain
        // naturally-evicted path strings forever.
        let entries: Vec<TrackedCacheEntry> = paths.entries.iter().cloned().collect();
        let mut resident = 0;
        for entry in entries {
            if self.tracked_entry_present(&entry) {
                if entry.path().starts_with(&prefix) {
                    resident += 1;
                }
            } else {
                paths.entries.remove(&entry);
            }
        }
        resident
    }

    pub fn metadata_hits(&self) -> u64 {
        self.stats.meta_hits.load(Ordering::Relaxed)
    }
    pub fn metadata_misses(&self) -> u64 {
        self.stats.meta_misses.load(Ordering::Relaxed)
    }
    pub fn metadata_inserts(&self) -> u64 {
        self.stats.meta_inserts.load(Ordering::Relaxed)
    }

    pub fn metadata_usage_bytes(&self) -> usize {
        self.metadata.usage()
    }

    pub fn metadata_capacity_bytes(&self) -> usize {
        self.budgets.decoded.metadata_bytes
    }

    #[cfg(feature = "text-index")]
    pub fn text_index_usage_bytes(&self) -> usize {
        self.text_indexes.usage()
    }

    #[cfg(feature = "text-index")]
    pub fn text_index_capacity_bytes(&self) -> usize {
        self.budgets.decoded.text_index_bytes
    }

    #[cfg(feature = "vector-index")]
    pub fn vector_index_usage_bytes(&self) -> usize {
        self.vector_indexes.usage()
    }

    #[cfg(feature = "vector-index")]
    pub fn vector_index_capacity_bytes(&self) -> usize {
        self.budgets.decoded.vector_index_bytes
    }

    /// Look up a body. Returns `None` on miss; the caller must perform
    /// the GET and re-insert via [`Self::insert`].
    pub fn get(&self, key: &str) -> Option<Bytes> {
        match self.inner.get(key) {
            Some(entry) => {
                self.stats.hits.fetch_add(1, Ordering::Relaxed);
                Some(entry.value().clone())
            }
            None => {
                self.stats.misses.fetch_add(1, Ordering::Relaxed);
                None
            }
        }
    }

    /// Insert (or replace) the entry for `key`.
    pub fn insert(&self, key: String, value: Bytes) {
        self.stats.inserts.fetch_add(1, Ordering::Relaxed);
        if !fits_capacity(self.budgets.body_bytes, raw_body_weight(&key, &value)) {
            return;
        }
        let tracked = TrackedCacheEntry::Body(key.clone());
        self.insert_tracked(tracked, || {
            self.inner.insert(key, value);
        });
    }

    /// Current cache usage in bytes (sum of weights of live entries).
    pub fn usage(&self) -> usize {
        self.inner.usage()
    }

    /// Raw SST body tier capacity.
    pub fn capacity_bytes(&self) -> usize {
        self.budgets.body_bytes
    }

    /// Sum of all compiled SST cache tier capacities (seven base tiers plus
    /// optional text and vector tiers).
    pub fn aggregate_capacity_bytes(&self) -> usize {
        self.budgets.aggregate_capacity_bytes()
    }

    /// Sum of cache-accounted resident bytes in every compiled SST tier.
    pub fn aggregate_usage_bytes(&self) -> usize {
        let base = self
            .inner
            .usage()
            .saturating_add(self.decoded_node_row_groups.usage())
            .saturating_add(self.property_sidecars.usage())
            .saturating_add(self.metadata.usage())
            .saturating_add(self.edge_streams.usage())
            .saturating_add(self.edge_readers.usage())
            .saturating_add(self.bloom_filters.usage());
        #[cfg(feature = "text-index")]
        let base = base.saturating_add(self.text_indexes.usage());
        #[cfg(feature = "vector-index")]
        let base = base.saturating_add(self.vector_indexes.usage());
        base
    }

    /// Cache hit count since construction.
    pub fn hits(&self) -> u64 {
        self.stats.hits.load(Ordering::Relaxed)
    }

    /// Cache miss count since construction.
    pub fn misses(&self) -> u64 {
        self.stats.misses.load(Ordering::Relaxed)
    }

    /// Cache insert count since construction. Useful in production
    /// dashboards alongside `hits` and `misses`.
    pub fn inserts(&self) -> u64 {
        self.stats.inserts.load(Ordering::Relaxed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tight_cache(bytes: usize) -> SstCache {
        SstCache::with_all_budgets(bytes, bytes, bytes, DecodedCacheBudgets::uniform(bytes))
    }

    #[test]
    fn insert_then_get_returns_same_bytes() {
        let cache = SstCache::new(1 << 20);
        cache.insert("k".into(), Bytes::from_static(b"hello"));
        let got = cache.get("k").unwrap();
        assert_eq!(got, Bytes::from_static(b"hello"));
        assert_eq!(cache.hits(), 1);
        assert_eq!(cache.misses(), 0);
    }

    #[test]
    fn miss_returns_none() {
        let cache = SstCache::new(1 << 20);
        assert!(cache.get("nope").is_none());
        assert_eq!(cache.hits(), 0);
        assert_eq!(cache.misses(), 1);
    }

    #[test]
    fn every_decoded_tier_has_an_explicit_capacity() {
        let budget = 32 * 1024;
        let cache = tight_cache(budget);
        assert_eq!(cache.capacity_bytes(), budget);
        assert_eq!(cache.decoded_node_row_groups.capacity(), budget);
        assert_eq!(cache.metadata.capacity(), budget);
        assert_eq!(cache.edge_streams.capacity(), budget);
        assert_eq!(cache.edge_readers.capacity(), budget);
        assert_eq!(cache.property_sidecars.capacity(), budget);
        assert_eq!(cache.bloom_filters.capacity(), budget);
        #[cfg(feature = "text-index")]
        assert_eq!(cache.text_indexes.capacity(), budget);
        #[cfg(feature = "vector-index")]
        assert_eq!(cache.vector_indexes.capacity(), budget);
        let compiled_tiers = 7
            + usize::from(cfg!(feature = "text-index"))
            + usize::from(cfg!(feature = "vector-index"));
        assert_eq!(cache.aggregate_capacity_bytes(), compiled_tiers * budget);
    }

    #[test]
    fn foyer_never_admits_an_entry_larger_than_its_capacity() {
        let cache = tight_cache(4 * 1024);
        cache.insert("k".into(), Bytes::from(vec![0; 6 * 1024]));

        assert!(cache.get("k").is_none());
        assert_eq!(cache.usage(), 0);
        assert_eq!(cache.aggregate_usage_bytes(), 0);
    }

    #[test]
    fn zero_capacity_uses_a_non_admitting_dummy_cache() {
        let cache = tight_cache(0);
        assert_eq!(cache.aggregate_capacity_bytes(), 0);
        cache.insert(String::new(), Bytes::new());
        assert!(cache.get("").is_none());
        assert_eq!(cache.aggregate_usage_bytes(), 0);
    }

    #[cfg(feature = "text-index")]
    #[test]
    fn oversized_text_index_is_rejected_with_explicit_wire_size() {
        let cache = tight_cache(4 * 1024);
        let key = "tenants/a/sst/level1/search.ft".to_string();
        let (body, _) = crate::sst::text::build_body(vec![([1; 16], "legal text".into())])
            .unwrap()
            .unwrap();
        let index = Arc::new(crate::sst::text::TextIndex::decode(&body).unwrap());

        assert!(!cache.can_admit_text_index_wire_bytes(&key, 1024));
        cache.insert_text_index_with_wire_bytes(key.clone(), index, 1024);
        assert!(cache.get_text_index(&key).is_none());
        assert_eq!(cache.text_index_usage_bytes(), 0);
    }

    #[cfg(feature = "vector-index")]
    #[test]
    fn oversized_vector_index_is_rejected_with_explicit_wire_size() {
        use crate::manifest::{VectorIndexDescriptor, VectorMetric, VectorQuantization};

        let cache = tight_cache(4 * 1024);
        let key = "tenants/a/sst/level1/search.vg".to_string();
        let descriptor = VectorIndexDescriptor {
            name: "emb_idx".into(),
            label: "Doc".into(),
            property: "embedding".into(),
            dim: 2,
            metric: VectorMetric::Cosine,
            r: 2,
            l_build: 4,
            alpha: 1.2,
            quantization: VectorQuantization::None,
        };
        let (body, _) = crate::sst::vector::build_body(
            &descriptor,
            vec![([1; 16], vec![1.0, 0.0]), ([2; 16], vec![0.0, 1.0])],
        )
        .unwrap()
        .unwrap();
        let index = Arc::new(crate::sst::vector::VectorGraphIndex::decode(&body).unwrap());

        assert!(!cache.can_admit_vector_index_wire_bytes(&key, 1024));
        cache.insert_vector_index_with_wire_bytes(key.clone(), index, 1024);
        assert!(cache.get_vector_index(&key).is_none());
        assert_eq!(cache.vector_index_usage_bytes(), 0);
    }

    #[test]
    fn decoded_property_sidecars_and_blooms_are_cached_and_counted() {
        let cache = SstCache::new(1 << 20);
        let unique_path = "tenants/a/sst/level0/nodes.unique-key.idx";
        assert!(cache.get_unique_property_sidecar(unique_path).is_none());
        let unique = Arc::new(BTreeMap::from([("k".to_string(), [7; 16])]));
        cache.insert_unique_property_sidecar(unique_path.to_string(), unique.clone());
        assert!(Arc::ptr_eq(
            &cache.get_unique_property_sidecar(unique_path).unwrap(),
            &unique
        ));

        let equality_path = "tenants/a/sst/level0/nodes.eq-kind.idx";
        assert!(cache.get_equality_property_sidecar(equality_path).is_none());
        let equality = Arc::new(BTreeMap::from([("kind".to_string(), vec![[9; 16]])]));
        cache.insert_equality_property_sidecar(equality_path.to_string(), equality.clone());
        assert!(Arc::ptr_eq(
            &cache.get_equality_property_sidecar(equality_path).unwrap(),
            &equality
        ));
        assert_eq!(cache.property_sidecar_misses(), 2);
        assert_eq!(cache.property_sidecar_inserts(), 2);
        assert_eq!(cache.property_sidecar_hits(), 2);

        let bloom_path = "tenants/a/sst/level0/edges.bloom";
        assert!(cache.get_bloom_filter(bloom_path).is_none());
        let bloom = Arc::new(BloomFilter::with_capacity(10, 10));
        cache.insert_bloom_filter(bloom_path.to_string(), bloom.clone());
        assert!(Arc::ptr_eq(
            &cache.get_bloom_filter(bloom_path).unwrap(),
            &bloom
        ));
        assert_eq!(cache.bloom_misses(), 1);
        assert_eq!(cache.bloom_inserts(), 1);
        assert_eq!(cache.bloom_hits(), 1);

        assert_eq!(cache.namespace_side_entries("tenants/a"), 3);
        cache.prune_namespace("tenants/a");
        assert_eq!(cache.namespace_side_entries("tenants/a"), 0);
    }

    #[test]
    fn retain_paths_only_prunes_the_given_namespace() {
        let cache = SstCache::new(1 << 20);
        let bundle = || {
            Arc::new(EdgeStreamBundle {
                overflow: None,
                declared: Vec::new(),
            })
        };
        let a_live = "tenants/a/sst/level0/live.csr".to_string();
        let a_dead = "tenants/a/sst/level0/dead.csr".to_string();
        let b_entry = "tenants/b/sst/level0/other.csr".to_string();
        for k in [&a_live, &a_dead, &b_entry] {
            cache.insert_edge_streams(k.clone(), bundle());
        }

        // Namespace `a` flushes: only its own dead path may go. A naive
        // global retain would also evict `b`'s entry here.
        let live: std::collections::HashSet<String> = [a_live.clone()].into();
        cache.retain_paths("tenants/a", &live);

        assert!(
            cache.get_edge_streams(&a_live).is_some(),
            "a's live entry kept"
        );
        assert!(
            cache.get_edge_streams(&a_dead).is_none(),
            "a's dead entry pruned"
        );
        assert!(
            cache.get_edge_streams(&b_entry).is_some(),
            "sibling namespace's entry must survive a's retain"
        );
    }

    #[test]
    fn retain_paths_prefix_respects_path_boundary() {
        // "tenants/a" must not claim "tenants/a2/..." entries.
        let cache = SstCache::new(1 << 20);
        let bundle = Arc::new(EdgeStreamBundle {
            overflow: None,
            declared: Vec::new(),
        });
        let a2 = "tenants/a2/sst/level0/x.csr".to_string();
        cache.insert_edge_streams(a2.clone(), bundle);
        cache.retain_paths("tenants/a", &std::collections::HashSet::new());
        assert!(
            cache.get_edge_streams(&a2).is_some(),
            "tenants/a2 is not under tenants/a"
        );
    }

    #[test]
    fn prune_namespace_drops_all_side_entries_for_that_namespace() {
        let cache = SstCache::new(1 << 20);
        let bundle = || {
            Arc::new(EdgeStreamBundle {
                overflow: None,
                declared: Vec::new(),
            })
        };
        cache.insert_edge_streams("tenants/gone/sst/level0/a.csr".into(), bundle());
        cache.insert_edge_streams("tenants/gone/sst/level0/b.csr".into(), bundle());
        cache.insert_edge_streams("tenants/kept/sst/level0/c.csr".into(), bundle());
        assert_eq!(cache.namespace_side_entries("tenants/gone"), 2);

        cache.prune_namespace("tenants/gone");
        assert_eq!(cache.namespace_side_entries("tenants/gone"), 0);
        assert_eq!(cache.namespace_side_entries("tenants/kept"), 1);
    }

    #[test]
    fn stale_decode_cannot_reinsert_after_manifest_prune() {
        let cache = tight_cache(1 << 20);
        let live = "tenants/a/sst/level0/live.csr".to_string();
        let dead = "tenants/a/sst/level0/dead.csr".to_string();
        cache.retain_paths("tenants/a", &HashSet::from([live.clone()]));

        // Model a decode that started against the old manifest and completed
        // only after retain_paths published the new live set.
        cache.insert(dead.clone(), Bytes::from_static(b"obsolete"));
        cache.insert_edge_streams(
            dead.clone(),
            Arc::new(EdgeStreamBundle {
                overflow: None,
                declared: Vec::new(),
            }),
        );
        assert!(cache.get(&dead).is_none());
        assert!(cache.get_edge_streams(&dead).is_none());

        cache.insert(live.clone(), Bytes::from_static(b"current"));
        cache.insert_edge_streams(
            live.clone(),
            Arc::new(EdgeStreamBundle {
                overflow: None,
                declared: Vec::new(),
            }),
        );
        assert!(cache.get(&live).is_some());
        assert!(cache.get_edge_streams(&live).is_some());

        cache.prune_namespace("tenants/a");
        assert!(cache.get(&live).is_none(), "raw body pruned eagerly");
        assert!(
            cache.get_edge_streams(&live).is_none(),
            "decoded tier pruned eagerly"
        );
        assert_eq!(cache.namespace_side_entries("tenants/a"), 0);
    }

    #[test]
    fn decoded_property_and_edge_stream_tiers_evict_under_byte_budgets() {
        let budget = 4 * 1024;
        let cache = tight_cache(budget);
        let inserted = 64usize;

        for i in 0..inserted {
            let property_path = format!("tenants/b/sst/level0/{i}.unique.idx");
            let property = Arc::new(BTreeMap::from([(
                format!("key-{i}-{}", "x".repeat(128)),
                [i as u8; 16],
            )]));
            cache.insert_unique_property_sidecar(property_path, property);

            let stream_path = format!("tenants/b/sst/level0/{i}.csr");
            cache.insert_edge_streams(
                stream_path,
                Arc::new(EdgeStreamBundle {
                    overflow: Some(vec![Some("y".repeat(512))]),
                    declared: vec![("payload".into(), vec![Some("z".repeat(512))])],
                }),
            );
        }

        assert_eq!(cache.property_sidecar_capacity_bytes(), budget);
        assert_eq!(cache.edge_streams_capacity_bytes(), budget);
        assert!(
            cache.property_sidecar_usage_bytes() < inserted * 128,
            "property usage must stay far below the unbounded payload"
        );
        assert!(
            cache.edge_streams_usage_bytes() < inserted * 1024,
            "edge stream usage must stay far below the unbounded payload"
        );
        assert!(
            cache.namespace_side_entries("tenants/b") < inserted * 2,
            "the diagnostic must discard path tombstones for foyer evictions"
        );
    }

    #[test]
    fn bloom_tier_is_weighted_and_bounded() {
        let budget = 8 * 1024;
        let cache = tight_cache(budget);
        for i in 0..64 {
            cache.insert_bloom_filter(
                format!("tenants/c/sst/level0/{i}.bloom"),
                Arc::new(BloomFilter::with_capacity(4096, 10)),
            );
        }
        assert_eq!(cache.bloom_capacity_bytes(), budget);
        assert!(
            cache.bloom_usage_bytes() < 64 * 4096,
            "decoded blooms must evict instead of accumulating"
        );
        assert!(cache.namespace_side_entries("tenants/c") < 64);
    }

    #[test]
    fn evicts_when_capacity_exceeded() {
        // Tight capacity: 16 KiB.
        let cache = SstCache::new(16 * 1024);
        let raw_inserted = 32usize;
        for i in 0..raw_inserted {
            // ~2 KiB per value → 64 KiB total → evictions kick in.
            let value = Bytes::from(vec![0u8; 2048]);
            cache.insert(format!("k-{i}"), value);
        }
        assert_eq!(cache.inserts(), raw_inserted as u64);
        // S3FIFO doesn't hard-cap at every instant (some operations are
        // lazy), but the cache must clearly be smaller than the raw
        // inserted total — otherwise eviction isn't running.
        let raw_total = raw_inserted as u64 * 2048;
        assert!(
            (cache.usage() as u64) < raw_total / 2,
            "cache.usage()={}, expected < {}",
            cache.usage(),
            raw_total / 2
        );
    }
}
