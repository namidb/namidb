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
//! - Weight includes key/value bytes plus conservative cache-record,
//! hash-index and eviction-policy overhead, so the budget is not bypassed by
//! millions of tiny entries.
//! - Legacy per-tier budgets are scaled under the process-wide
//! `NAMIDB_CACHE_MAX_BYTES` ceiling before the shared cache is constructed.

use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock, Weak};

use arrow_array::RecordBatch;
use bytes::Bytes;
use foyer::{Cache, CacheBuilder, Event, EventListener};
use parquet::file::metadata::ParquetMetaData;

use namidb_core::Value;

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

/// Default budget for manifest-admission and resident-path metadata.
///
/// This state owns path strings independently from Foyer values. Keeping it
/// as a first-class tier prevents a long eviction/churn workload from growing
/// cache metadata outside `NAMIDB_CACHE_MAX_BYTES`.
pub const DEFAULT_PATH_REGISTRY_CACHE_BUDGET_MIB: usize = 32;

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

pub fn path_registry_cache_budget_bytes() -> usize {
    legacy_budget_bytes(
        "NAMIDB_CACHE_PATH_REGISTRY_BUDGET_MIB",
        DEFAULT_PATH_REGISTRY_CACHE_BUDGET_MIB,
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

/// Canonical key used by scalar-v1 equality posting sidecars.
///
/// String keys deliberately retain the legacy raw representation. That keeps
/// sidecars written by a new server safe for rolling upgrades and rollback:
/// a 2.0.4 reader ignores the new manifest encoding field, but can still probe
/// every String value correctly. Non-String scalars use tagged encodings.
///
/// A raw String such as `"b:1"` can consequently share a posting with the
/// tagged Bool `true`. Equality and ordered readers always hydrate and confirm
/// candidates against the current typed value, so the collision only adds a
/// conservative candidate; it cannot produce a false result.
pub fn encode_equality_property_value(value: &Value) -> Option<String> {
    let mut out = String::new();
    match value {
        Value::Str(s) => out.push_str(s),
        Value::Bool(v) => out.push_str(if *v { "b:1" } else { "b:0" }),
        Value::I64(v) => {
            use std::fmt::Write as _;
            write!(&mut out, "i:{:016x}", (*v as u64) ^ (1_u64 << 63)).ok()?;
        }
        Value::F64(v) if !v.is_nan() => {
            let normalized = if *v == 0.0 { 0.0 } else { *v };
            use std::fmt::Write as _;
            write!(&mut out, "f:{:016x}", normalized.to_bits()).ok()?;
        }
        Value::Bytes(bytes) => {
            use std::fmt::Write as _;
            out.push_str("x:");
            for byte in bytes {
                write!(&mut out, "{byte:02x}").ok()?;
            }
        }
        Value::Date(v) => {
            use std::fmt::Write as _;
            write!(&mut out, "d:{:08x}", (*v as u32) ^ (1_u32 << 31)).ok()?;
        }
        Value::DateTime(v) => {
            use std::fmt::Write as _;
            write!(&mut out, "t:{:016x}", (*v as u64) ^ (1_u64 << 63)).ok()?;
        }
        Value::Null
        | Value::F64(_)
        | Value::Vec(_)
        | Value::VecI8 { .. }
        | Value::List(_)
        | Value::Map(_) => return None,
    }
    Some(out)
}

/// Canonical, SELF-DELIMITING key for TupleV1 composite posting sidecars.
///
/// Each part is `tag byte + u32-le length + canonical part bytes`, in the
/// index's DECLARATION order. Part semantics follow
/// [`encode_equality_property_value`] and the transactional tuple probe
/// (`unique_index::UniqueKeyPart`): typed tags keep `I64(1)` and `F64(1.0)`
/// distinct, `-0.0` folds into `0.0`, and a NaN / Null / non-scalar member
/// makes the whole tuple unindexable (`None`) — such a row is never filed
/// and such a probe falls back to the scan, so the sidecar cannot diverge
/// from the flat scan. Unlike ScalarV1's raw strings, the length prefix
/// makes the key unambiguous: no member byte pattern can alias another
/// member or a part boundary (`("a","bc")` != `("ab","c")`, and a raw
/// string `"b:1"` member cannot collide with a Bool member).
pub fn encode_equality_tuple_key(values: &[&Value]) -> Option<Vec<u8>> {
    let mut out = Vec::new();
    for value in values {
        let (tag, bytes): (u8, Vec<u8>) = match value {
            Value::Str(s) => (b's', s.as_bytes().to_vec()),
            Value::Bool(v) => (b'b', vec![u8::from(*v)]),
            // Numerics share one canonical tag: Cypher equality coerces
            // integer = float (`30 = 30.0` is TRUE, mirrored from the
            // executor's `is_equal`), so members that compare equal MUST
            // produce identical keys or the index route silently drops rows
            // the scan route returns. `i64 -> f64` loses precision above
            // 2^53, so distinct integers may share a posting — safe, because
            // the read path re-confirms every candidate member-by-member
            // with [`cypher_scalar_equal`].
            Value::I64(v) => (b'n', (*v as f64).to_bits().to_be_bytes().to_vec()),
            Value::F64(v) if !v.is_nan() => {
                let normalized = if *v == 0.0 { 0.0 } else { *v };
                (b'n', normalized.to_bits().to_be_bytes().to_vec())
            }
            Value::Bytes(bytes) => (b'x', bytes.clone()),
            Value::Date(v) => (b'd', v.to_be_bytes().to_vec()),
            Value::DateTime(v) => (b't', v.to_be_bytes().to_vec()),
            Value::Null
            | Value::F64(_)
            | Value::Vec(_)
            | Value::VecI8 { .. }
            | Value::List(_)
            | Value::Map(_) => return None,
        };
        out.push(tag);
        out.extend_from_slice(&(u32::try_from(bytes.len()).ok()?).to_le_bytes());
        out.extend_from_slice(&bytes);
    }
    Some(out)
}

/// Cypher scalar equality — the confirmation twin of
/// [`encode_equality_tuple_key`]. Mirrors the executor's `is_equal` for the
/// scalar subset: `NULL` equals nothing (itself included), integer = float
/// compares mathematically, everything else compares typed. Any two values
/// this accepts as equal encode identical tuple members, and every key
/// collision the lossy numeric canonicalization introduces is separated
/// here.
pub fn cypher_scalar_equal(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Null, _) | (_, Value::Null) => false,
        (Value::I64(x), Value::F64(y)) | (Value::F64(y), Value::I64(x)) => (*x as f64) == *y,
        _ => a == b,
    }
}

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
    cache_key_weight(key).saturating_add(payload)
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
    cache_key_weight(key)
        .saturating_add(std::mem::size_of::<EdgeStreamBundle>())
        .saturating_add(overflow)
        .saturating_add(declared)
}

fn edge_reader_weight(key: &str, reader: &Arc<crate::sst::edges::EdgeSstReader>) -> usize {
    // EdgeSstReader intentionally keeps its `Bytes` body private. The final
    // section end is a safe lower bound for that retained allocation; charge
    // the footer/table and parsed cumulative/fence structures on top. These
    // estimates intentionally overcharge normal files so budget eviction is
    // preferable to an unbounded resident graph. If the raw-body tier shares
    // the same Bytes allocation, both tiers still charge the full body: this
    // conservative double accounting is stable even when either owner evicts
    // or an active query temporarily pins one clone.
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
    cache_key_weight(key)
        .saturating_add(body_bytes)
        .saturating_add(parsed)
}

fn bloom_filter_weight(key: &str, filter: &Arc<BloomFilter>) -> usize {
    cache_key_weight(key)
        .saturating_add(
            (filter.block_count() as usize).saturating_mul(8 * std::mem::size_of::<u32>()),
        )
        .saturating_add(std::mem::size_of::<BloomFilter>())
}

#[cfg(any(feature = "text-index", feature = "vector-index"))]
#[derive(Debug)]
struct WeightedArc<T> {
    value: Arc<T>,
    estimated_bytes: usize,
}

#[cfg(any(feature = "text-index", feature = "vector-index"))]
fn weighted_arc_weight<T>(key: &str, value: &WeightedArc<T>) -> usize {
    cache_key_weight(key).saturating_add(value.estimated_bytes)
}

/// Vector and full-text accelerators share one eviction pool. They are both
/// monolithic decoded indexes and compete for the same bounded memory; keeping
/// separate Foyer instances made a free text allocation unusable by vector
/// search (and vice versa).
#[cfg(any(feature = "text-index", feature = "vector-index"))]
#[derive(Debug)]
enum CachedSearchIndex {
    #[cfg(feature = "text-index")]
    Text(WeightedArc<crate::sst::text::TextIndex>),
    /// Sparse footer/directory for a range-readable NAMIFT03 object. Posting
    /// and document pages are retained by the separate hybrid range cache,
    /// never in this decoded-index entry.
    #[cfg(feature = "text-index")]
    TextV3(WeightedArc<crate::sst::text::TextIndexV3Reader>),
    #[cfg(feature = "vector-index")]
    Vector(WeightedArc<crate::sst::vector::VectorGraphIndex>),
    #[cfg(feature = "vector-index")]
    VectorV5(WeightedArc<crate::sst::vector::v5::VectorV5Reader>),
}

#[cfg(any(feature = "text-index", feature = "vector-index"))]
fn cached_search_index_weight(key: &str, value: &CachedSearchIndex) -> usize {
    match value {
        #[cfg(feature = "text-index")]
        CachedSearchIndex::Text(value) => weighted_arc_weight(key, value),
        #[cfg(feature = "text-index")]
        CachedSearchIndex::TextV3(value) => weighted_arc_weight(key, value),
        #[cfg(feature = "vector-index")]
        CachedSearchIndex::Vector(value) => weighted_arc_weight(key, value),
        #[cfg(feature = "vector-index")]
        CachedSearchIndex::VectorV5(value) => weighted_arc_weight(key, value),
    }
}

/// Capacity refusal returned before a monolithic search index is decoded.
/// The query layer turns this into a typed storage error instead of silently
/// selecting an O(corpus) flat scan.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SearchIndexCapacityError {
    pub required_bytes: usize,
    pub capacity_bytes: usize,
}

#[derive(Debug, Clone, Copy)]
struct DecodedCacheBudgets {
    metadata_bytes: usize,
    edge_stream_bytes: usize,
    edge_reader_bytes: usize,
    bloom_bytes: usize,
    #[cfg(any(feature = "text-index", feature = "vector-index"))]
    search_index_bytes: usize,
}

#[derive(Debug, Clone, Copy)]
struct SstCacheBudgets {
    body_bytes: usize,
    node_row_group_bytes: usize,
    property_sidecar_bytes: usize,
    path_registry_bytes: usize,
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
            .saturating_add(self.decoded.bloom_bytes)
            .saturating_add(self.path_registry_bytes);
        #[cfg(any(feature = "text-index", feature = "vector-index"))]
        let base = base.saturating_add(self.decoded.search_index_bytes);
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
            #[cfg(any(feature = "text-index", feature = "vector-index"))]
            search_index_bytes: (if cfg!(feature = "text-index") {
                legacy_budget_bytes(
                    "NAMIDB_TEXT_INDEX_CACHE_BUDGET_MIB",
                    DEFAULT_TEXT_INDEX_CACHE_BUDGET_MIB,
                )
            } else {
                0
            })
            .saturating_add(if cfg!(feature = "vector-index") {
                legacy_budget_bytes(
                    "NAMIDB_VECTOR_INDEX_CACHE_BUDGET_MIB",
                    DEFAULT_VECTOR_INDEX_CACHE_BUDGET_MIB,
                )
            } else {
                0
            }),
        }
    }

    fn from_capacities(capacities: CacheCapacities) -> Self {
        Self {
            metadata_bytes: capacities.sst_metadata_bytes,
            edge_stream_bytes: capacities.edge_stream_bytes,
            edge_reader_bytes: capacities.edge_reader_bytes,
            bloom_bytes: capacities.bloom_filter_bytes,
            #[cfg(any(feature = "text-index", feature = "vector-index"))]
            search_index_bytes: capacities.search_index_bytes,
        }
    }

    #[cfg(test)]
    fn uniform(bytes: usize) -> Self {
        Self {
            metadata_bytes: bytes,
            edge_stream_bytes: bytes,
            edge_reader_bytes: bytes,
            bloom_bytes: bytes,
            #[cfg(any(feature = "text-index", feature = "vector-index"))]
            search_index_bytes: bytes,
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
    cache_key_weight(&key.0)
        .saturating_add(std::mem::size_of::<usize>())
        .saturating_add(value.iter().fold(0usize, |sum, batch| {
            sum.saturating_add(batch.get_array_memory_size())
        }))
}

fn raw_body_weight(key: &str, value: &Bytes) -> usize {
    cache_key_weight(key).saturating_add(value.len())
}

fn metadata_weight(key: &str, value: &Arc<ParquetMetaData>) -> usize {
    cache_key_weight(key).saturating_add(value.memory_size())
}

/// Conservative cache-owned overhead outside the payload itself: Foyer record,
/// hash-index/eviction-policy nodes, `Arc` bookkeeping, and allocator slack.
/// The path bytes are charged separately by [`cache_key_weight`].
const FOYER_ENTRY_OVERHEAD_BYTES: usize = 192;

fn cache_key_weight(key: &str) -> usize {
    key.len().saturating_add(FOYER_ENTRY_OVERHEAD_BYTES)
}

fn fits_capacity(capacity_bytes: usize, weight: usize) -> bool {
    capacity_bytes > 0 && weight <= capacity_bytes
}

#[cfg(feature = "text-index")]
fn text_index_estimated_weight(key: &str, wire_bytes: usize, doc_count: usize) -> usize {
    // Bincode's wire body is a stable lower bound. A 6x multiplier
    // conservatively covers BTree/posting Vec allocator overhead and the
    // separately sorted id copy. The independent per-document floor protects
    // malformed/legacy descriptors whose serialized size understates the
    // decoded object.
    cache_key_weight(key).saturating_add(
        wire_bytes
            .saturating_mul(6)
            .max(doc_count.saturating_mul(512))
            .max(std::mem::size_of::<crate::sst::text::TextIndex>()),
    )
}

#[cfg(feature = "vector-index")]
fn vector_index_estimated_weight(
    key: &str,
    wire_bytes: usize,
    point_count: usize,
    dim: usize,
) -> usize {
    // Decode retains the serialized graph's vectors/adjacency and builds a
    // second navigation-space vector set. Six wire copies is conservative for
    // normal Vamana degree; the independent corpus-shape estimate protects the
    // no-body-cache path and, importantly, is used both before and after
    // decode so an admitted miss cannot turn into an uncached decode loop.
    let per_point = dim.saturating_mul(8).saturating_add(512);
    cache_key_weight(key).saturating_add(
        wire_bytes
            .saturating_mul(6)
            .max(point_count.saturating_mul(per_point))
            .max(std::mem::size_of::<crate::sst::vector::VectorGraphIndex>()),
    )
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

/// Conservative metadata allowance for one registry entry, in addition to its
/// owned path allocation and inline enum. This covers the hash-table bucket,
/// control byte, load-factor slack and allocator bookkeeping.
const TRACKED_CACHE_ENTRY_OVERHEAD_BYTES: usize = 160;
const REGISTRY_HASH_ENTRY_OVERHEAD_BYTES: usize = 128;
const REGISTRY_QUEUE_ENTRY_OVERHEAD_BYTES: usize = 64;

fn tracked_cache_entry_metadata_bytes(entry: &TrackedCacheEntry) -> usize {
    std::mem::size_of::<TrackedCacheEntry>()
        .saturating_add(entry.path().len())
        .saturating_add(TRACKED_CACHE_ENTRY_OVERHEAD_BYTES)
}

fn owned_string_metadata_bytes(value: &String) -> usize {
    std::mem::size_of::<String>().saturating_add(value.capacity())
}

fn live_rule_metadata_bytes(prefix: &str, live: &HashSet<String>) -> usize {
    std::mem::size_of::<HashSet<String>>()
        .saturating_add(std::mem::size_of::<String>())
        .saturating_add(prefix.len())
        .saturating_add(REGISTRY_HASH_ENTRY_OVERHEAD_BYTES)
        .saturating_add(live.iter().fold(0usize, |total, path| {
            total
                .saturating_add(std::mem::size_of::<String>())
                .saturating_add(path.len())
                .saturating_add(REGISTRY_HASH_ENTRY_OVERHEAD_BYTES)
        }))
        .saturating_add(if live.is_empty() {
            std::mem::size_of::<String>()
                .saturating_add(prefix.len())
                .saturating_add(REGISTRY_QUEUE_ENTRY_OVERHEAD_BYTES)
        } else {
            0
        })
}

fn denied_rule_metadata_bytes(prefix: &str) -> usize {
    std::mem::size_of::<String>()
        .saturating_mul(2)
        .saturating_add(prefix.len().saturating_mul(2))
        .saturating_add(REGISTRY_HASH_ENTRY_OVERHEAD_BYTES)
        .saturating_add(REGISTRY_QUEUE_ENTRY_OVERHEAD_BYTES)
}

#[derive(Debug)]
struct CachePathRegistry {
    entries: HashSet<TrackedCacheEntry>,
    /// Normalized namespace prefix (always trailing `/`) -> absolute live
    /// object paths from the latest manifest this process observed.
    live_by_namespace: HashMap<String, HashSet<String>>,
    /// Recently-evicted namespaces reject cache insertions from decodes that
    /// started before eviction. This is deliberately separate from
    /// `live_by_namespace`: an empty live set is normal for a newly-opened
    /// namespace, while an eviction is a deny-all admission rule.
    denied_namespaces: HashSet<String>,
    /// FIFO for authoritative empty-manifest live rules. Embedded callers can
    /// open and drop arbitrarily many fresh namespaces without a registry
    /// eviction callback, so these guards need the same bounded-lateness
    /// contract as eviction tombstones.
    empty_live_order: VecDeque<String>,
    /// FIFO order for the bounded deny set. Namespace eviction is a cold
    /// control-plane path, so FIFO gives deterministic O(1) admission without
    /// adding an LRU dependency to every cache miss.
    denied_order: VecDeque<String>,
    /// Logical, conservatively weighted metadata bytes. Entry bytes stay O(1)
    /// to update on Foyer's hot eviction path; namespace-rule bytes are
    /// recomputed only on manifest/tenant control-plane operations.
    entry_bytes: usize,
    rule_bytes: usize,
    capacity_bytes: usize,
    /// If an authoritative rule cannot fit, fail closed instead of retaining
    /// unbounded metadata or allowing a stale decode to repopulate old data.
    admission_disabled: bool,
}

const MAX_DENIED_CACHE_NAMESPACES: usize = 4096;
// Bounds metadata even for embedded callers that do not use NamespaceRegistry.
const MAX_EMPTY_LIVE_NAMESPACES: usize = 4096;

impl CachePathRegistry {
    fn new(capacity_bytes: usize) -> Self {
        Self {
            entries: HashSet::new(),
            live_by_namespace: HashMap::new(),
            denied_namespaces: HashSet::new(),
            empty_live_order: VecDeque::new(),
            denied_order: VecDeque::new(),
            entry_bytes: 0,
            rule_bytes: 0,
            capacity_bytes,
            admission_disabled: false,
        }
    }

    fn used_bytes(&self) -> usize {
        self.entry_bytes.saturating_add(self.rule_bytes)
    }

    fn can_reserve(&self, bytes: usize) -> bool {
        bytes <= self.capacity_bytes.saturating_sub(self.used_bytes())
    }

    fn try_insert_entry(&mut self, entry: TrackedCacheEntry) -> bool {
        if self.entries.contains(&entry) {
            return true;
        }
        let bytes = tracked_cache_entry_metadata_bytes(&entry);
        if !self.can_reserve(bytes) {
            return false;
        }
        let inserted = self.entries.insert(entry);
        if inserted {
            self.entry_bytes = self.entry_bytes.saturating_add(bytes);
        }
        inserted
    }

    fn remove_entry(&mut self, entry: &TrackedCacheEntry) -> bool {
        let Some(removed) = self.entries.take(entry) else {
            return false;
        };
        self.entry_bytes = self
            .entry_bytes
            .saturating_sub(tracked_cache_entry_metadata_bytes(&removed));
        // Natural eviction can empty a high-water table. Shrink
        // geometrically so bucket allocations cannot survive churn forever,
        // while avoiding a reallocation on every single eviction.
        if self.entries.capacity() > 64
            && self.entries.len().saturating_mul(2) < self.entries.capacity()
        {
            self.entries.shrink_to_fit();
        }
        true
    }

    fn clear_entries(&mut self) {
        self.entries = HashSet::new();
        self.entry_bytes = 0;
    }

    fn admits(&self, path: &str) -> bool {
        if self.admission_disabled {
            return false;
        }
        // Every engine-owned cache path is `<namespace-prefix>/sst/...`.
        // Recover the normalized prefix as a borrowed slice so the hot path
        // performs two hash probes without allocating or walking up to 4096
        // eviction tombstones. `rfind` tolerates a custom root containing an
        // earlier segment named `sst`.
        if let Some(prefix) = canonical_namespace_prefix(path) {
            if self.denied_namespaces.contains(prefix) {
                return false;
            }
            if let Some(live) = self.live_by_namespace.get(prefix) {
                return live.contains(path);
            }
        }

        // Namespace prefixes should not overlap, but choosing the longest
        // match keeps non-canonical custom embedded layouts deterministic.
        let live = self
            .live_by_namespace
            .iter()
            .filter(|(prefix, _)| path.starts_with(prefix.as_str()))
            .max_by_key(|(prefix, _)| prefix.len());
        let denied = self
            .denied_namespaces
            .iter()
            .filter(|prefix| path.starts_with(prefix.as_str()))
            .max_by_key(|prefix| prefix.len());
        match (live, denied) {
            (None, None) => true,
            (Some((_, live)), None) => live.contains(path),
            (None, Some(_)) => false,
            (Some((live_prefix, live)), Some(denied_prefix)) => {
                live_prefix.len() > denied_prefix.len() && live.contains(path)
            }
        }
    }

    fn allow_namespace(&mut self, prefix: &str, live: &HashSet<String>) {
        self.remove_namespace_rules(prefix);
        if self.admission_disabled {
            return;
        }

        // Empty is an authoritative manifest live set, not "no rule". It
        // occurs both on a fresh namespace and after compaction GCs the final
        // tombstone. Retaining it prevents a decode begun against the previous
        // manifest from repopulating an obsolete body after that transition.
        // The first flush publishes its non-empty live set before any later
        // read can cache the new immutable object.
        if live.is_empty() && self.empty_live_order.len() >= MAX_EMPTY_LIVE_NAMESPACES {
            if let Some(expired) = self.empty_live_order.pop_front() {
                if self
                    .live_by_namespace
                    .get(&expired)
                    .is_some_and(HashSet::is_empty)
                {
                    self.live_by_namespace.remove(&expired);
                }
            }
        }

        self.recompute_rule_bytes();
        let required = live_rule_metadata_bytes(prefix, live);
        if !self.can_reserve(required) {
            self.admission_disabled = true;
            tracing::warn!(
                registry_capacity_bytes = self.capacity_bytes,
                registry_usage_bytes = self.used_bytes(),
                required_rule_bytes = required,
                "cache path registry exhausted; disabling SST cache admission"
            );
            return;
        }

        if live.is_empty() {
            self.empty_live_order.push_back(prefix.to_string());
        }
        // Rebuild instead of `HashSet::clone`: callers can pass a sparse set
        // whose historical bucket capacity is enormous. Only live elements,
        // not the caller's high-water allocation, belong in this registry.
        let compact_live = live.iter().cloned().collect();
        self.live_by_namespace
            .insert(prefix.to_string(), compact_live);
        self.recompute_rule_bytes();
        if self.used_bytes() > self.capacity_bytes {
            self.remove_namespace_rules(prefix);
            self.admission_disabled = true;
            tracing::warn!(
                registry_capacity_bytes = self.capacity_bytes,
                registry_usage_bytes = self.used_bytes(),
                "cache path rule allocation exceeded its conservative estimate; disabling admission"
            );
        }
    }

    fn deny_namespace(&mut self, prefix: String) {
        self.remove_namespace_rules(&prefix);
        if self.admission_disabled {
            return;
        }

        if self.denied_namespaces.len() >= MAX_DENIED_CACHE_NAMESPACES {
            if let Some(expired) = self.denied_order.pop_front() {
                self.denied_namespaces.remove(&expired);
            }
        }
        self.recompute_rule_bytes();
        let required = denied_rule_metadata_bytes(&prefix);
        if !self.can_reserve(required) {
            self.admission_disabled = true;
            tracing::warn!(
                registry_capacity_bytes = self.capacity_bytes,
                registry_usage_bytes = self.used_bytes(),
                required_rule_bytes = required,
                "cache path registry exhausted; disabling SST cache admission"
            );
            return;
        }

        self.denied_namespaces.insert(prefix.clone());
        self.denied_order.push_back(prefix);
        self.recompute_rule_bytes();
        if self.used_bytes() > self.capacity_bytes {
            let prefix = self.denied_order.back().cloned().unwrap_or_default();
            self.remove_namespace_rules(&prefix);
            self.admission_disabled = true;
            tracing::warn!(
                registry_capacity_bytes = self.capacity_bytes,
                registry_usage_bytes = self.used_bytes(),
                "cache deny-rule allocation exceeded its conservative estimate; disabling admission"
            );
        }
    }

    fn remove_namespace_rules(&mut self, prefix: &str) {
        self.live_by_namespace.remove(prefix);
        self.empty_live_order.retain(|empty| empty != prefix);
        self.denied_namespaces.remove(prefix);
        self.denied_order.retain(|denied| denied != prefix);
        if self.live_by_namespace.is_empty()
            || (self.live_by_namespace.capacity() > 64
                && self.live_by_namespace.len().saturating_mul(2)
                    < self.live_by_namespace.capacity())
        {
            self.live_by_namespace.shrink_to_fit();
        }
        if self.denied_namespaces.capacity() > 64
            && self.denied_namespaces.len().saturating_mul(2) < self.denied_namespaces.capacity()
        {
            self.denied_namespaces.shrink_to_fit();
        }
        if self.empty_live_order.capacity() > 64
            && self.empty_live_order.len().saturating_mul(2) < self.empty_live_order.capacity()
        {
            self.empty_live_order.shrink_to_fit();
        }
        if self.denied_order.capacity() > 64
            && self.denied_order.len().saturating_mul(2) < self.denied_order.capacity()
        {
            self.denied_order.shrink_to_fit();
        }
        self.recompute_rule_bytes();
    }

    fn recompute_rule_bytes(&mut self) {
        let live_bytes = self
            .live_by_namespace
            .iter()
            .fold(0usize, |total, (prefix, live)| {
                let map_entry = owned_string_metadata_bytes(prefix)
                    .saturating_add(std::mem::size_of::<HashSet<String>>())
                    .saturating_add(REGISTRY_HASH_ENTRY_OVERHEAD_BYTES);
                let paths = live.iter().fold(0usize, |paths, path| {
                    paths
                        .saturating_add(owned_string_metadata_bytes(path))
                        .saturating_add(REGISTRY_HASH_ENTRY_OVERHEAD_BYTES)
                });
                total.saturating_add(map_entry).saturating_add(paths)
            });
        let empty_order_bytes = self.empty_live_order.iter().fold(0usize, |total, prefix| {
            total
                .saturating_add(owned_string_metadata_bytes(prefix))
                .saturating_add(REGISTRY_QUEUE_ENTRY_OVERHEAD_BYTES)
        });
        let denied_set_bytes = self.denied_namespaces.iter().fold(0usize, |total, prefix| {
            total
                .saturating_add(owned_string_metadata_bytes(prefix))
                .saturating_add(REGISTRY_HASH_ENTRY_OVERHEAD_BYTES)
        });
        let denied_order_bytes = self.denied_order.iter().fold(0usize, |total, prefix| {
            total
                .saturating_add(owned_string_metadata_bytes(prefix))
                .saturating_add(REGISTRY_QUEUE_ENTRY_OVERHEAD_BYTES)
        });
        self.rule_bytes = live_bytes
            .saturating_add(empty_order_bytes)
            .saturating_add(denied_set_bytes)
            .saturating_add(denied_order_bytes);
    }
}

/// Foyer calls listeners after removing records from its shard lock. Only
/// natural eviction and replacement are handled here: explicit `remove` is
/// issued while the registry mutex is held by prune/clear and is accounted by
/// those callers, avoiding listener reentrancy.
struct CacheRegistryListener<K, V> {
    registry: Weak<Mutex<CachePathRegistry>>,
    classify: fn(&K, &V) -> TrackedCacheEntry,
}

impl<K, V> EventListener for CacheRegistryListener<K, V>
where
    K: foyer::Key,
    V: foyer::Value,
{
    type Key = K;
    type Value = V;

    fn on_leave(&self, reason: Event, key: &K, value: &V) {
        if !matches!(reason, Event::Evict | Event::Replace) {
            return;
        }
        let Some(registry) = self.registry.upgrade() else {
            return;
        };
        registry
            .lock()
            .unwrap()
            .remove_entry(&(self.classify)(key, value));
    }
}

fn canonical_namespace_prefix(path: &str) -> Option<&str> {
    let marker = path.rfind("/sst/")?;
    Some(&path[..=marker])
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
    /// Sparse node-id RowFilter probes and complete payload rows they
    /// materialised. These distinguish the MERGE batch path from a hidden
    /// full row-group decode in performance regressions.
    node_sparse_filter_scans: AtomicU64,
    node_sparse_filter_rows: AtomicU64,
    /// Exact LIMIT pushdown over globally-disjoint node SSTs. These counters
    /// distinguish a true lazy prefix scan from the legacy full scan followed
    /// by `take(k)`.
    node_limited_scan_fast_paths: AtomicU64,
    node_limited_scan_fallbacks: AtomicU64,
    node_limited_scan_decoded_rows: AtomicU64,
    node_limited_scan_examined_rows: AtomicU64,
    node_limited_scan_output_rows: AtomicU64,
    node_limited_scan_row_groups: AtomicU64,
    node_limited_scan_range_bytes: AtomicU64,
    /// Range-readable node-locator work. These counters make it possible to
    /// distinguish exact B+tree paths from a hidden full-id-column scan.
    node_locator_probes: AtomicU64,
    node_locator_pages: AtomicU64,
    node_locator_entries_examined: AtomicU64,
    node_locator_bytes: AtomicU64,
    /// Exact `(LabelId, NodeId)` sidecar membership used to prune labelled
    /// graph expansion targets without hydrating complete node rows. A probe
    /// is one descriptor/label batch, not one endpoint.
    label_membership_fast_paths: AtomicU64,
    label_membership_fallbacks: AtomicU64,
    label_membership_probes: AtomicU64,
    label_membership_candidates: AtomicU64,
    label_membership_pages: AtomicU64,
    label_membership_entries_examined: AtomicU64,
    label_membership_bytes: AtomicU64,
    /// Range-readable exact-edge sidecar work. One probe may contain a whole
    /// UNWIND batch and visit each distinct B+tree page once.
    edge_point_probes: AtomicU64,
    edge_point_pages: AtomicU64,
    edge_point_entries_examined: AtomicU64,
    edge_point_bytes: AtomicU64,
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
    /// Valid persisted search indexes refused because their conservative
    /// decoded footprint exceeded the configured shared search-index pool.
    /// These are deliberately distinct from absent/stale/corrupt fallbacks.
    #[cfg(any(feature = "text-index", feature = "vector-index"))]
    vector_index_capacity_rejections: AtomicU64,
    #[cfg(any(feature = "text-index", feature = "vector-index"))]
    text_index_capacity_rejections: AtomicU64,
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
    /// Shared decoded `.vg` / `.ft` pool. Either kind can use all assigned
    /// search memory and evict the other, avoiding an artificial fixed split.
    /// Entries remain deeply weighted and admission rejects a body that cannot
    /// fit as one bounded resident object.
    #[cfg(any(feature = "text-index", feature = "vector-index"))]
    search_indexes: Arc<Cache<String, CachedSearchIndex>>,
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
                "path_registry_capacity_bytes",
                &self.budgets.path_registry_bytes,
            )
            .field(
                "path_registry_usage_bytes",
                &self.path_registry_usage_bytes(),
            )
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
                "node_sparse_filter_scans",
                &self.stats.node_sparse_filter_scans.load(Ordering::Relaxed),
            )
            .field(
                "node_sparse_filter_rows",
                &self.stats.node_sparse_filter_rows.load(Ordering::Relaxed),
            )
            .field(
                "node_limited_scan_fast_paths",
                &self
                    .stats
                    .node_limited_scan_fast_paths
                    .load(Ordering::Relaxed),
            )
            .field(
                "node_limited_scan_fallbacks",
                &self
                    .stats
                    .node_limited_scan_fallbacks
                    .load(Ordering::Relaxed),
            )
            .field(
                "node_limited_scan_decoded_rows",
                &self
                    .stats
                    .node_limited_scan_decoded_rows
                    .load(Ordering::Relaxed),
            )
            .field(
                "node_limited_scan_range_bytes",
                &self
                    .stats
                    .node_limited_scan_range_bytes
                    .load(Ordering::Relaxed),
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
        #[cfg(any(feature = "text-index", feature = "vector-index"))]
        out.field(
            "search_index_capacity_bytes",
            &self.budgets.decoded.search_index_bytes,
        )
        .field("search_index_usage_bytes", &self.search_indexes.usage())
        .field(
            "vector_index_capacity_rejections",
            &self
                .stats
                .vector_index_capacity_rejections
                .load(Ordering::Relaxed),
        )
        .field(
            "text_index_capacity_rejections",
            &self
                .stats
                .text_index_capacity_rejections
                .load(Ordering::Relaxed),
        );
        out.finish()
    }
}

impl SstCache {
    /// Build a new cache sized for `capacity_bytes`. Entries include payload,
    /// key and conservative Foyer-owned metadata in their weight. The decoded
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
            path_registry_cache_budget_bytes(),
            DecodedCacheBudgets::from_env(),
        )
    }

    #[cfg(test)]
    pub(crate) fn with_uniform_budgets(bytes: usize) -> Self {
        Self::with_all_budgets(
            bytes,
            bytes,
            bytes,
            bytes,
            DecodedCacheBudgets::uniform(bytes),
        )
    }

    fn with_shared_capacities(capacities: CacheCapacities) -> Self {
        Self::with_all_budgets(
            capacities.sst_body_bytes,
            capacities.decoded_node_row_group_bytes,
            capacities.property_sidecar_bytes,
            capacities.path_registry_bytes,
            DecodedCacheBudgets::from_capacities(capacities),
        )
    }

    fn with_all_budgets(
        capacity_bytes: usize,
        decoded_node_rg_bytes: usize,
        property_sidecar_bytes: usize,
        path_registry_bytes: usize,
        decoded: DecodedCacheBudgets,
    ) -> Self {
        let budgets = SstCacheBudgets {
            body_bytes: capacity_bytes,
            node_row_group_bytes: decoded_node_rg_bytes,
            property_sidecar_bytes,
            path_registry_bytes,
            decoded,
        };
        let tracked_paths = Arc::new(Mutex::new(CachePathRegistry::new(path_registry_bytes)));
        let inner_capacity = capacity_bytes;
        let inner = CacheBuilder::new(inner_capacity.max(1))
            .with_shards(1)
            .with_weighter(|key: &String, value: &Bytes| raw_body_weight(key, value))
            .with_filter(move |key: &String, value: &Bytes| {
                fits_capacity(inner_capacity, raw_body_weight(key, value))
            })
            .with_event_listener(Arc::new(CacheRegistryListener {
                registry: Arc::downgrade(&tracked_paths),
                classify: |key: &String, _value: &Bytes| TrackedCacheEntry::Body(key.clone()),
            }))
            .build();
        let node_rg_capacity = decoded_node_rg_bytes;
        let decoded_node_row_groups = CacheBuilder::new(node_rg_capacity.max(1))
            .with_shards(1)
            .with_weighter(decoded_node_row_group_weight)
            .with_filter(move |key: &NodeRowGroupKey, value: &DecodedNodeRowGroup| {
                fits_capacity(node_rg_capacity, decoded_node_row_group_weight(key, value))
            })
            .with_event_listener(Arc::new(CacheRegistryListener {
                registry: Arc::downgrade(&tracked_paths),
                classify: |key: &NodeRowGroupKey, _value: &DecodedNodeRowGroup| {
                    TrackedCacheEntry::NodeRowGroup(key.0.clone(), key.1)
                },
            }))
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
            .with_event_listener(Arc::new(CacheRegistryListener {
                registry: Arc::downgrade(&tracked_paths),
                classify: |key: &String, _value: &DecodedPropertySidecar| {
                    TrackedCacheEntry::PropertySidecar(key.clone())
                },
            }))
            .build();
        let metadata_capacity = decoded.metadata_bytes;
        let metadata = CacheBuilder::new(metadata_capacity.max(1))
            .with_shards(1)
            .with_weighter(|key: &String, value: &Arc<ParquetMetaData>| metadata_weight(key, value))
            .with_filter(move |key: &String, value: &Arc<ParquetMetaData>| {
                fits_capacity(metadata_capacity, metadata_weight(key, value))
            })
            .with_event_listener(Arc::new(CacheRegistryListener {
                registry: Arc::downgrade(&tracked_paths),
                classify: |key: &String, _value: &Arc<ParquetMetaData>| {
                    TrackedCacheEntry::Metadata(key.clone())
                },
            }))
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
            .with_event_listener(Arc::new(CacheRegistryListener {
                registry: Arc::downgrade(&tracked_paths),
                classify: |key: &String, _value: &Arc<EdgeStreamBundle>| {
                    TrackedCacheEntry::EdgeStreams(key.clone())
                },
            }))
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
            .with_event_listener(Arc::new(CacheRegistryListener {
                registry: Arc::downgrade(&tracked_paths),
                classify: |key: &String, _value: &Arc<crate::sst::edges::EdgeSstReader>| {
                    TrackedCacheEntry::EdgeReader(key.clone())
                },
            }))
            .build();
        let bloom_capacity = decoded.bloom_bytes;
        let bloom_filters = CacheBuilder::new(bloom_capacity.max(1))
            .with_shards(1)
            .with_weighter(|key: &String, value: &Arc<BloomFilter>| bloom_filter_weight(key, value))
            .with_filter(move |key: &String, value: &Arc<BloomFilter>| {
                fits_capacity(bloom_capacity, bloom_filter_weight(key, value))
            })
            .with_event_listener(Arc::new(CacheRegistryListener {
                registry: Arc::downgrade(&tracked_paths),
                classify: |key: &String, _value: &Arc<BloomFilter>| {
                    TrackedCacheEntry::Bloom(key.clone())
                },
            }))
            .build();
        #[cfg(any(feature = "text-index", feature = "vector-index"))]
        let search_index_capacity = decoded.search_index_bytes;
        #[cfg(any(feature = "text-index", feature = "vector-index"))]
        let search_indexes = CacheBuilder::new(search_index_capacity.max(1))
            .with_shards(1)
            .with_weighter(|key: &String, value: &CachedSearchIndex| {
                cached_search_index_weight(key, value)
            })
            .with_filter(move |key: &String, value: &CachedSearchIndex| {
                fits_capacity(
                    search_index_capacity,
                    cached_search_index_weight(key, value),
                )
            })
            .with_event_listener(Arc::new(CacheRegistryListener {
                registry: Arc::downgrade(&tracked_paths),
                classify: |key: &String, value: &CachedSearchIndex| match value {
                    #[cfg(feature = "text-index")]
                    CachedSearchIndex::Text(_) | CachedSearchIndex::TextV3(_) => {
                        TrackedCacheEntry::TextIndex(key.clone())
                    }
                    #[cfg(feature = "vector-index")]
                    CachedSearchIndex::Vector(_) | CachedSearchIndex::VectorV5(_) => {
                        TrackedCacheEntry::VectorIndex(key.clone())
                    }
                },
            }))
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
            #[cfg(any(feature = "text-index", feature = "vector-index"))]
            search_indexes: Arc::new(search_indexes),
            tracked_paths,
            stats: Arc::new(CacheStats::default()),
        }
    }

    /// Reserve bounded path metadata, insert without holding the registry
    /// mutex, then revalidate both manifest admission and Foyer residency.
    ///
    /// Foyer invokes eviction/replacement listeners synchronously from
    /// `insert`, so holding the mutex across that call would deadlock. The
    /// post-insert validation still closes the old prune race: a stale pinned
    /// snapshot that finishes decoding after a manifest commit is removed
    /// before this function returns.
    fn insert_tracked(&self, entry: TrackedCacheEntry, insert: impl FnOnce()) {
        let reserved = {
            let mut paths = self.tracked_paths.lock().unwrap();
            if !paths.admits(entry.path()) {
                false
            } else {
                let required = tracked_cache_entry_metadata_bytes(&entry);
                if required > paths.capacity_bytes.saturating_sub(paths.rule_bytes) {
                    false
                } else {
                    // Registry pressure must behave like cache pressure. Drop
                    // resident metadata/value pairs until the new reservation
                    // fits; otherwise a full registry could reject every
                    // future insertion even though Foyer would evict for it.
                    while !paths.try_insert_entry(entry.clone()) {
                        let Some(victim) = paths.entries.iter().next().cloned() else {
                            break;
                        };
                        self.remove_tracked(&victim);
                        paths.remove_entry(&victim);
                    }
                    paths.entries.contains(&entry)
                }
            }
        };
        if !reserved {
            return;
        }

        insert();

        {
            let mut paths = self.tracked_paths.lock().unwrap();
            let resident = self.tracked_entry_present(&entry);
            let registered = if resident && paths.admits(entry.path()) {
                // A replacement listener can have removed the reservation for
                // the outgoing value. Re-reserve for the value now resident.
                paths.try_insert_entry(entry.clone())
            } else {
                false
            };
            if !registered {
                paths.remove_entry(&entry);
            }
            if resident && !registered {
                // Explicit removal is deliberately ignored by the listener,
                // so this is safe while holding the registry mutex and leaves
                // no window for a concurrent insert of the same key.
                self.remove_tracked(&entry);
            }
        }
    }

    /// Look up a decoded text index for an SST path. Returns `None` on miss;
    /// the caller decodes once and re-inserts via [`Self::insert_text_index`].
    /// SSTs are immutable per UUIDv7-keyed path so cached indexes never go
    /// stale; superseded paths are pruned by [`Self::retain_paths`].
    #[cfg(feature = "text-index")]
    pub fn get_text_index(&self, key: &str) -> Option<Arc<crate::sst::text::TextIndex>> {
        self.search_indexes
            .get(key)
            .and_then(|entry| match entry.value() {
                CachedSearchIndex::Text(value) => Some(value.value.clone()),
                CachedSearchIndex::TextV3(_) => None,
                #[cfg(feature = "vector-index")]
                CachedSearchIndex::Vector(_) | CachedSearchIndex::VectorV5(_) => None,
            })
    }

    /// Look up the sparse reader for a range-readable NAMIFT03 object.
    #[cfg(feature = "text-index")]
    pub fn get_text_v3_reader(
        &self,
        key: &str,
    ) -> Option<Arc<crate::sst::text::TextIndexV3Reader>> {
        self.search_indexes
            .get(key)
            .and_then(|entry| match entry.value() {
                CachedSearchIndex::TextV3(value) => Some(value.value.clone()),
                CachedSearchIndex::Text(_) => None,
                #[cfg(feature = "vector-index")]
                CachedSearchIndex::Vector(_) | CachedSearchIndex::VectorV5(_) => None,
            })
    }

    /// Retain a sparse NAMIFT03 footer/directory when it fits the shared
    /// search-index pool. Unlike a monolithic decode this is best-effort:
    /// rejection only means the next query reopens a few metadata pages, never
    /// that it must perform an O(corpus) flat scan.
    #[cfg(feature = "text-index")]
    pub fn insert_text_v3_reader(
        &self,
        key: String,
        reader: Arc<crate::sst::text::TextIndexV3Reader>,
    ) -> bool {
        let required_bytes =
            cache_key_weight(&key).saturating_add(reader.estimated_resident_bytes());
        if !fits_capacity(self.budgets.decoded.search_index_bytes, required_bytes) {
            return false;
        }
        let value = WeightedArc {
            value: reader,
            estimated_bytes: required_bytes.saturating_sub(cache_key_weight(&key)),
        };
        let tracked = TrackedCacheEntry::TextIndex(key.clone());
        self.insert_tracked(tracked, || {
            self.search_indexes
                .insert(key, CachedSearchIndex::TextV3(value));
        });
        true
    }

    /// Preflight a serialized text-index body against the shared search pool.
    ///
    /// Call this with the manifest/object length and corpus count before GET.
    /// A refusal is observable and must be returned to the client; it is not
    /// equivalent to an absent/stale/corrupt optional accelerator.
    #[cfg(feature = "text-index")]
    pub fn admit_text_index_wire_bytes(
        &self,
        key: &str,
        wire_bytes: usize,
        doc_count: usize,
    ) -> Result<usize, SearchIndexCapacityError> {
        let required_bytes = text_index_estimated_weight(key, wire_bytes, doc_count);
        self.admit_search_index("text", required_bytes)
    }

    /// Compatibility preflight for callers that only know the serialized
    /// body length.
    ///
    /// Text and vector indexes now share one eviction pool, so this checks the
    /// historical six-wire-copy lower bound against that shared capacity.
    /// Engine read paths should prefer [`Self::admit_text_index_wire_bytes`],
    /// which also accounts for the corpus shape and returns the exact refusal.
    #[cfg(feature = "text-index")]
    pub fn can_admit_text_index_wire_bytes(&self, key: &str, wire_bytes: usize) -> bool {
        fits_capacity(
            self.budgets.decoded.search_index_bytes,
            cache_key_weight(key).saturating_add(wire_bytes.saturating_mul(6)),
        )
    }

    /// Store a decoded text index for an SST path.
    ///
    /// This source-compatible wrapper preserves the historical best-effort
    /// contract: an oversized entry is not cached. Engine read paths that must
    /// surface a capacity refusal use [`Self::try_insert_text_index`].
    #[cfg(feature = "text-index")]
    pub fn insert_text_index(&self, key: String, idx: Arc<crate::sst::text::TextIndex>) {
        let _ = self.try_insert_text_index(key, idx);
    }

    /// Store a decoded text index or return its exact shared-pool capacity
    /// refusal.
    #[cfg(feature = "text-index")]
    pub fn try_insert_text_index(
        &self,
        key: String,
        idx: Arc<crate::sst::text::TextIndex>,
    ) -> Result<(), SearchIndexCapacityError> {
        let wire_bytes = self
            .inner
            .get(&key)
            .map(|entry| entry.value().len())
            .unwrap_or_default();
        self.try_insert_text_index_with_wire_bytes(key, idx, wire_bytes)
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
        let _ = self.try_insert_text_index_with_wire_bytes(key, idx, wire_bytes);
    }

    /// Store a decoded text index with an explicit serialized-body length or
    /// return its exact shared-pool capacity refusal.
    #[cfg(feature = "text-index")]
    pub fn try_insert_text_index_with_wire_bytes(
        &self,
        key: String,
        idx: Arc<crate::sst::text::TextIndex>,
        wire_bytes: usize,
    ) -> Result<(), SearchIndexCapacityError> {
        let doc_count = usize::try_from(idx.doc_count()).unwrap_or(usize::MAX);
        let required_bytes = text_index_estimated_weight(&key, wire_bytes, doc_count);
        self.admit_search_index("text", required_bytes)?;
        let value = WeightedArc {
            value: idx,
            estimated_bytes: required_bytes.saturating_sub(cache_key_weight(&key)),
        };
        let tracked = TrackedCacheEntry::TextIndex(key.clone());
        self.insert_tracked(tracked, || {
            self.search_indexes
                .insert(key, CachedSearchIndex::Text(value));
        });
        Ok(())
    }

    /// Look up a decoded vector index for an SST path. Same contract as
    /// [`Self::get_text_index`].
    #[cfg(feature = "vector-index")]
    pub fn get_vector_index(&self, key: &str) -> Option<Arc<crate::sst::vector::VectorGraphIndex>> {
        self.search_indexes
            .get(key)
            .and_then(|entry| match entry.value() {
                CachedSearchIndex::Vector(value) => Some(value.value.clone()),
                CachedSearchIndex::VectorV5(_) => None,
                #[cfg(feature = "text-index")]
                CachedSearchIndex::Text(_) | CachedSearchIndex::TextV3(_) => None,
            })
    }

    /// Look up the sparse reader for a range-readable NAMIVG05 object.
    #[cfg(feature = "vector-index")]
    pub fn get_vector_v5_reader(
        &self,
        key: &str,
    ) -> Option<Arc<crate::sst::vector::v5::VectorV5Reader>> {
        self.search_indexes
            .get(key)
            .and_then(|entry| match entry.value() {
                CachedSearchIndex::VectorV5(value) => Some(value.value.clone()),
                CachedSearchIndex::Vector(_) => None,
                #[cfg(feature = "text-index")]
                CachedSearchIndex::Text(_) | CachedSearchIndex::TextV3(_) => None,
            })
    }

    /// Best-effort retention of the centroid footer for NAMIVG05. Corpus
    /// pages stay in the bounded hybrid range cache, so a metadata admission
    /// miss never selects the O(corpus) vector fallback.
    #[cfg(feature = "vector-index")]
    pub fn insert_vector_v5_reader(
        &self,
        key: String,
        reader: Arc<crate::sst::vector::v5::VectorV5Reader>,
    ) -> bool {
        let required_bytes =
            cache_key_weight(&key).saturating_add(reader.resident_metadata_bytes());
        if !fits_capacity(self.budgets.decoded.search_index_bytes, required_bytes) {
            return false;
        }
        let value = WeightedArc {
            value: reader,
            estimated_bytes: required_bytes.saturating_sub(cache_key_weight(&key)),
        };
        let tracked = TrackedCacheEntry::VectorIndex(key.clone());
        self.insert_tracked(tracked, || {
            self.search_indexes
                .insert(key, CachedSearchIndex::VectorV5(value));
        });
        true
    }

    /// Preflight a serialized vector-index body before GET/decode.
    #[cfg(feature = "vector-index")]
    pub fn admit_vector_index_wire_bytes(
        &self,
        key: &str,
        wire_bytes: usize,
        point_count: usize,
        dim: usize,
    ) -> Result<usize, SearchIndexCapacityError> {
        let required_bytes = vector_index_estimated_weight(key, wire_bytes, point_count, dim);
        self.admit_search_index("vector", required_bytes)
    }

    /// Compatibility preflight for callers that only know the serialized
    /// body length.
    ///
    /// Both search-index kinds use the same eviction pool. This keeps the
    /// historical lower-bound check available; engine paths should use
    /// [`Self::admit_vector_index_wire_bytes`] for shape-aware admission.
    #[cfg(feature = "vector-index")]
    pub fn can_admit_vector_index_wire_bytes(&self, key: &str, wire_bytes: usize) -> bool {
        fits_capacity(
            self.budgets.decoded.search_index_bytes,
            cache_key_weight(key).saturating_add(wire_bytes.saturating_mul(6)),
        )
    }

    /// Store a decoded vector index for an SST path.
    ///
    /// This source-compatible wrapper is best-effort. Engine read paths that
    /// must report a capacity refusal use [`Self::try_insert_vector_index`].
    #[cfg(feature = "vector-index")]
    pub fn insert_vector_index(&self, key: String, idx: Arc<crate::sst::vector::VectorGraphIndex>) {
        let _ = self.try_insert_vector_index(key, idx);
    }

    /// Store a decoded vector index or return its exact shared-pool capacity
    /// refusal.
    #[cfg(feature = "vector-index")]
    pub fn try_insert_vector_index(
        &self,
        key: String,
        idx: Arc<crate::sst::vector::VectorGraphIndex>,
    ) -> Result<(), SearchIndexCapacityError> {
        let wire_bytes = self
            .inner
            .get(&key)
            .map(|entry| entry.value().len())
            .unwrap_or_default();
        self.try_insert_vector_index_with_wire_bytes(key, idx, wire_bytes)
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
        let _ = self.try_insert_vector_index_with_wire_bytes(key, idx, wire_bytes);
    }

    /// Store a decoded vector index with an explicit serialized-body length or
    /// return its exact shared-pool capacity refusal.
    #[cfg(feature = "vector-index")]
    pub fn try_insert_vector_index_with_wire_bytes(
        &self,
        key: String,
        idx: Arc<crate::sst::vector::VectorGraphIndex>,
        wire_bytes: usize,
    ) -> Result<(), SearchIndexCapacityError> {
        let point_count = usize::try_from(idx.point_count()).unwrap_or(usize::MAX);
        let required_bytes =
            vector_index_estimated_weight(&key, wire_bytes, point_count, idx.dim() as usize);
        self.admit_search_index("vector", required_bytes)?;
        let value = WeightedArc {
            value: idx,
            estimated_bytes: required_bytes.saturating_sub(cache_key_weight(&key)),
        };
        let tracked = TrackedCacheEntry::VectorIndex(key.clone());
        self.insert_tracked(tracked, || {
            self.search_indexes
                .insert(key, CachedSearchIndex::Vector(value));
        });
        Ok(())
    }

    #[cfg(any(feature = "text-index", feature = "vector-index"))]
    fn admit_search_index(
        &self,
        kind: &'static str,
        required_bytes: usize,
    ) -> Result<usize, SearchIndexCapacityError> {
        let capacity_bytes = self.budgets.decoded.search_index_bytes;
        if fits_capacity(capacity_bytes, required_bytes) {
            return Ok(required_bytes);
        }

        match kind {
            "vector" => {
                self.stats
                    .vector_index_capacity_rejections
                    .fetch_add(1, Ordering::Relaxed);
            }
            "text" => {
                self.stats
                    .text_index_capacity_rejections
                    .fetch_add(1, Ordering::Relaxed);
            }
            _ => {}
        }
        tracing::warn!(
            index_kind = kind,
            required_bytes,
            pool_capacity_bytes = capacity_bytes,
            config = "NAMIDB_SEARCH_INDEX_CACHE_MAX_BYTES/NAMIDB_CACHE_MAX_BYTES",
            "decoded search index rejected by configured cache capacity"
        );
        Err(SearchIndexCapacityError {
            required_bytes,
            capacity_bytes,
        })
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

    pub(crate) fn record_sparse_node_filter(&self, rows: usize) {
        self.stats
            .node_sparse_filter_scans
            .fetch_add(1, Ordering::Relaxed);
        self.stats
            .node_sparse_filter_rows
            .fetch_add(rows as u64, Ordering::Relaxed);
    }

    pub fn sparse_node_filter_scans(&self) -> u64 {
        self.stats.node_sparse_filter_scans.load(Ordering::Relaxed)
    }

    pub fn sparse_node_filter_rows(&self) -> u64 {
        self.stats.node_sparse_filter_rows.load(Ordering::Relaxed)
    }

    pub(crate) fn record_limited_node_scan_fast_path(&self) {
        self.stats
            .node_limited_scan_fast_paths
            .fetch_add(1, Ordering::Relaxed);
    }

    pub(crate) fn record_limited_node_scan_fallback(&self) {
        self.stats
            .node_limited_scan_fallbacks
            .fetch_add(1, Ordering::Relaxed);
    }

    pub(crate) fn record_limited_node_scan_work(
        &self,
        decoded_rows: usize,
        examined_rows: usize,
        output_rows: usize,
        row_groups: usize,
        range_bytes: u64,
    ) {
        self.stats
            .node_limited_scan_decoded_rows
            .fetch_add(decoded_rows as u64, Ordering::Relaxed);
        self.stats
            .node_limited_scan_examined_rows
            .fetch_add(examined_rows as u64, Ordering::Relaxed);
        self.stats
            .node_limited_scan_output_rows
            .fetch_add(output_rows as u64, Ordering::Relaxed);
        self.stats
            .node_limited_scan_row_groups
            .fetch_add(row_groups as u64, Ordering::Relaxed);
        self.stats
            .node_limited_scan_range_bytes
            .fetch_add(range_bytes, Ordering::Relaxed);
    }

    pub fn limited_node_scan_fast_paths(&self) -> u64 {
        self.stats
            .node_limited_scan_fast_paths
            .load(Ordering::Relaxed)
    }

    pub fn limited_node_scan_fallbacks(&self) -> u64 {
        self.stats
            .node_limited_scan_fallbacks
            .load(Ordering::Relaxed)
    }

    pub fn limited_node_scan_decoded_rows(&self) -> u64 {
        self.stats
            .node_limited_scan_decoded_rows
            .load(Ordering::Relaxed)
    }

    pub fn limited_node_scan_examined_rows(&self) -> u64 {
        self.stats
            .node_limited_scan_examined_rows
            .load(Ordering::Relaxed)
    }

    pub fn limited_node_scan_output_rows(&self) -> u64 {
        self.stats
            .node_limited_scan_output_rows
            .load(Ordering::Relaxed)
    }

    pub fn limited_node_scan_row_groups(&self) -> u64 {
        self.stats
            .node_limited_scan_row_groups
            .load(Ordering::Relaxed)
    }

    pub fn limited_node_scan_range_bytes(&self) -> u64 {
        self.stats
            .node_limited_scan_range_bytes
            .load(Ordering::Relaxed)
    }

    pub(crate) fn record_node_locator_probe(
        &self,
        stats: crate::sst::paged_index::PagedProbeStats,
    ) {
        self.stats
            .node_locator_probes
            .fetch_add(1, Ordering::Relaxed);
        self.stats
            .node_locator_pages
            .fetch_add(stats.pages_read as u64, Ordering::Relaxed);
        self.stats
            .node_locator_entries_examined
            .fetch_add(stats.leaf_entries_examined as u64, Ordering::Relaxed);
        self.stats
            .node_locator_bytes
            .fetch_add(stats.bytes_read as u64, Ordering::Relaxed);
    }

    pub fn node_locator_probes(&self) -> u64 {
        self.stats.node_locator_probes.load(Ordering::Relaxed)
    }

    pub fn node_locator_pages(&self) -> u64 {
        self.stats.node_locator_pages.load(Ordering::Relaxed)
    }

    pub fn node_locator_entries_examined(&self) -> u64 {
        self.stats
            .node_locator_entries_examined
            .load(Ordering::Relaxed)
    }

    pub fn node_locator_bytes(&self) -> u64 {
        self.stats.node_locator_bytes.load(Ordering::Relaxed)
    }

    pub(crate) fn record_label_membership_fast_path(&self) {
        self.stats
            .label_membership_fast_paths
            .fetch_add(1, Ordering::Relaxed);
    }

    pub(crate) fn record_label_membership_fallback(&self) {
        self.stats
            .label_membership_fallbacks
            .fetch_add(1, Ordering::Relaxed);
    }

    pub(crate) fn record_label_membership_probe(
        &self,
        candidates: usize,
        stats: crate::sst::paged_index::PagedProbeStats,
    ) {
        self.stats
            .label_membership_probes
            .fetch_add(1, Ordering::Relaxed);
        self.stats
            .label_membership_candidates
            .fetch_add(candidates as u64, Ordering::Relaxed);
        self.stats
            .label_membership_pages
            .fetch_add(stats.pages_read as u64, Ordering::Relaxed);
        self.stats
            .label_membership_entries_examined
            .fetch_add(stats.leaf_entries_examined as u64, Ordering::Relaxed);
        self.stats
            .label_membership_bytes
            .fetch_add(stats.bytes_read as u64, Ordering::Relaxed);
    }

    pub fn label_membership_fast_paths(&self) -> u64 {
        self.stats
            .label_membership_fast_paths
            .load(Ordering::Relaxed)
    }

    pub fn label_membership_fallbacks(&self) -> u64 {
        self.stats
            .label_membership_fallbacks
            .load(Ordering::Relaxed)
    }

    pub fn label_membership_probes(&self) -> u64 {
        self.stats.label_membership_probes.load(Ordering::Relaxed)
    }

    pub fn label_membership_candidates(&self) -> u64 {
        self.stats
            .label_membership_candidates
            .load(Ordering::Relaxed)
    }

    pub fn label_membership_pages(&self) -> u64 {
        self.stats.label_membership_pages.load(Ordering::Relaxed)
    }

    pub fn label_membership_entries_examined(&self) -> u64 {
        self.stats
            .label_membership_entries_examined
            .load(Ordering::Relaxed)
    }

    pub fn label_membership_bytes(&self) -> u64 {
        self.stats.label_membership_bytes.load(Ordering::Relaxed)
    }

    pub(crate) fn record_edge_point_probe(&self, stats: crate::sst::paged_index::PagedProbeStats) {
        self.stats.edge_point_probes.fetch_add(1, Ordering::Relaxed);
        self.stats
            .edge_point_pages
            .fetch_add(stats.pages_read as u64, Ordering::Relaxed);
        self.stats
            .edge_point_entries_examined
            .fetch_add(stats.leaf_entries_examined as u64, Ordering::Relaxed);
        self.stats
            .edge_point_bytes
            .fetch_add(stats.bytes_read as u64, Ordering::Relaxed);
    }

    pub fn edge_point_probes(&self) -> u64 {
        self.stats.edge_point_probes.load(Ordering::Relaxed)
    }

    pub fn edge_point_pages(&self) -> u64 {
        self.stats.edge_point_pages.load(Ordering::Relaxed)
    }

    pub fn edge_point_entries_examined(&self) -> u64 {
        self.stats
            .edge_point_entries_examined
            .load(Ordering::Relaxed)
    }

    pub fn edge_point_bytes(&self) -> u64 {
        self.stats.edge_point_bytes.load(Ordering::Relaxed)
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
                self.search_indexes.remove(path);
            }
            #[cfg(feature = "vector-index")]
            TrackedCacheEntry::VectorIndex(path) => {
                self.search_indexes.remove(path);
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
            TrackedCacheEntry::TextIndex(path) => {
                self.search_indexes.get(path).is_some_and(|entry| {
                    matches!(
                        entry.value(),
                        CachedSearchIndex::Text(_) | CachedSearchIndex::TextV3(_)
                    )
                })
            }
            #[cfg(feature = "vector-index")]
            TrackedCacheEntry::VectorIndex(path) => {
                self.search_indexes.get(path).is_some_and(|entry| {
                    matches!(
                        entry.value(),
                        CachedSearchIndex::Vector(_) | CachedSearchIndex::VectorV5(_)
                    )
                })
            }
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
        evicted: bool,
    ) {
        // Normalize to a path-segment boundary so "tenants/acme" cannot
        // match "tenants/acme2/...".
        let prefix = normalized_namespace_prefix(namespace_prefix);
        let mut paths = self.tracked_paths.lock().unwrap();
        // Publish the admission rule before removing entries while holding the
        // same lock used by insertions. A decode from a pre-prune snapshot
        // finishing later sees this rule and cannot resurrect a dead path.
        if evicted {
            paths.deny_namespace(prefix.clone());
        } else {
            // Reopening removes an older eviction tombstone. An empty live set
            // remains an authoritative deny-all until the first successful
            // flush publishes its new immutable paths.
            paths.allow_namespace(&prefix, live);
        }
        let stale: Vec<TrackedCacheEntry> = paths
            .entries
            .iter()
            .filter(|entry| entry.path().starts_with(&prefix) && !live.contains(entry.path()))
            .cloned()
            .collect();
        for entry in stale {
            self.remove_tracked(&entry);
            paths.remove_entry(&entry);
        }
    }

    /// Eagerly drop every cache entry under `namespace_prefix`. Called
    /// when a multi-tenant host evicts a namespace — its state is being
    /// dropped anyway, so its bodies/row groups/decoded indexes are dead
    /// weight in the shared cache.
    pub fn prune_namespace(&self, namespace_prefix: &str) {
        self.retain_paths_inner(namespace_prefix, &std::collections::HashSet::new(), true);
    }

    /// Drop every resident entry from every process-wide SST cache tier.
    ///
    /// Immutable object data remains authoritative in the backing store and
    /// every entry is rebuilt on demand. Per-namespace live-path admission
    /// rules and bounded eviction tombstones are retained so a decode from an
    /// old pinned snapshot still cannot resurrect an object made obsolete by
    /// a newer manifest or an evicted namespace.
    pub fn clear(&self) {
        let mut paths = self.tracked_paths.lock().unwrap();
        // foyer-memory 0.22's bulk `clear()` drains its index but leaves the
        // shard usage counter unchanged. Remove the tracked keys explicitly
        // so both resident values and byte accounting reach zero; insertion
        // is serialized by this same mutex, so no tier can race the sweep.
        let entries: Vec<_> = paths.entries.iter().cloned().collect();
        for entry in entries {
            self.remove_tracked(&entry);
        }
        // `HashSet::clear` retains its peak bucket allocation. This hook runs
        // specifically under RSS pressure, so replace the registry storage as
        // well as dropping its logical entries.
        paths.clear_entries();
        for live in paths.live_by_namespace.values_mut() {
            live.shrink_to_fit();
        }
        paths.live_by_namespace.shrink_to_fit();
        paths.empty_live_order.shrink_to_fit();
        paths.denied_namespaces.shrink_to_fit();
        paths.denied_order.shrink_to_fit();
        paths.recompute_rule_bytes();
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
                paths.remove_entry(&entry);
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

    #[cfg(any(feature = "text-index", feature = "vector-index"))]
    pub fn search_index_usage_bytes(&self) -> usize {
        self.search_indexes.usage()
    }

    #[cfg(any(feature = "text-index", feature = "vector-index"))]
    pub fn search_index_capacity_bytes(&self) -> usize {
        self.budgets.decoded.search_index_bytes
    }

    /// Cache-accounted bytes in the shared text/vector decoded-index pool.
    ///
    /// Kept as a source-compatible alias after the two formerly independent
    /// pools were combined.
    #[cfg(feature = "text-index")]
    pub fn text_index_usage_bytes(&self) -> usize {
        self.search_index_usage_bytes()
    }

    /// Capacity of the shared text/vector decoded-index pool.
    #[cfg(feature = "text-index")]
    pub fn text_index_capacity_bytes(&self) -> usize {
        self.search_index_capacity_bytes()
    }

    /// Cache-accounted bytes in the shared text/vector decoded-index pool.
    ///
    /// Kept as a source-compatible alias after the two formerly independent
    /// pools were combined.
    #[cfg(feature = "vector-index")]
    pub fn vector_index_usage_bytes(&self) -> usize {
        self.search_index_usage_bytes()
    }

    /// Capacity of the shared text/vector decoded-index pool.
    #[cfg(feature = "vector-index")]
    pub fn vector_index_capacity_bytes(&self) -> usize {
        self.search_index_capacity_bytes()
    }

    #[cfg(feature = "vector-index")]
    pub fn vector_index_capacity_rejections(&self) -> u64 {
        self.stats
            .vector_index_capacity_rejections
            .load(Ordering::Relaxed)
    }

    #[cfg(feature = "text-index")]
    pub fn text_index_capacity_rejections(&self) -> u64 {
        self.stats
            .text_index_capacity_rejections
            .load(Ordering::Relaxed)
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

    /// Sum of all compiled SST cache tier capacities (seven Foyer base tiers,
    /// the path registry, plus one optional shared search-index tier).
    pub fn aggregate_capacity_bytes(&self) -> usize {
        self.budgets.aggregate_capacity_bytes()
    }

    pub fn path_registry_capacity_bytes(&self) -> usize {
        self.budgets.path_registry_bytes
    }

    pub fn path_registry_usage_bytes(&self) -> usize {
        self.tracked_paths.lock().unwrap().used_bytes()
    }

    /// Sum of cache-accounted resident bytes and path/admission metadata in
    /// every compiled SST tier.
    pub fn aggregate_usage_bytes(&self) -> usize {
        let base = self
            .inner
            .usage()
            .saturating_add(self.decoded_node_row_groups.usage())
            .saturating_add(self.property_sidecars.usage())
            .saturating_add(self.metadata.usage())
            .saturating_add(self.edge_streams.usage())
            .saturating_add(self.edge_readers.usage())
            .saturating_add(self.bloom_filters.usage())
            .saturating_add(self.path_registry_usage_bytes());
        #[cfg(any(feature = "text-index", feature = "vector-index"))]
        let base = base.saturating_add(self.search_indexes.usage());
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

    #[test]
    fn scalar_equality_keys_preserve_legacy_strings_and_tag_other_types() {
        assert_eq!(
            encode_equality_property_value(&Value::Bool(true)).as_deref(),
            Some("b:1")
        );
        assert_eq!(
            encode_equality_property_value(&Value::Str("true".into())).as_deref(),
            Some("true")
        );
        assert_ne!(
            encode_equality_property_value(&Value::Bool(true)),
            encode_equality_property_value(&Value::Str("true".into()))
        );
        assert_eq!(
            encode_equality_property_value(&Value::Bool(true)),
            encode_equality_property_value(&Value::Str("b:1".into())),
            "legacy-compatible String keys may conservatively share a posting with tagged scalars"
        );
        assert_eq!(
            encode_equality_property_value(&Value::F64(-0.0)),
            encode_equality_property_value(&Value::F64(0.0))
        );
        assert!(encode_equality_property_value(&Value::Null).is_none());
        assert!(encode_equality_property_value(&Value::F64(f64::NAN)).is_none());
    }

    fn tight_cache(bytes: usize) -> SstCache {
        SstCache::with_uniform_budgets(bytes)
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
        #[cfg(any(feature = "text-index", feature = "vector-index"))]
        assert_eq!(cache.search_indexes.capacity(), budget);
        let compiled_tiers =
            8 + usize::from(cfg!(any(feature = "text-index", feature = "vector-index")));
        assert_eq!(cache.aggregate_capacity_bytes(), compiled_tiers * budget);
        assert_eq!(cache.path_registry_capacity_bytes(), budget);
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

        let preflight = cache
            .admit_text_index_wire_bytes(&key, 1024, 1)
            .unwrap_err();
        assert!(preflight.required_bytes > preflight.capacity_bytes);
        assert!(!cache.can_admit_text_index_wire_bytes(&key, 1024));
        let insertion = cache
            .try_insert_text_index_with_wire_bytes(key.clone(), index.clone(), 1024)
            .unwrap_err();
        assert_eq!(insertion, preflight);
        assert!(cache.get_text_index(&key).is_none());
        assert_eq!(cache.search_index_usage_bytes(), 0);
        assert_eq!(
            cache.text_index_usage_bytes(),
            cache.search_index_usage_bytes()
        );
        assert_eq!(
            cache.text_index_capacity_bytes(),
            cache.search_index_capacity_bytes()
        );
        assert_eq!(cache.text_index_capacity_rejections(), 2);

        cache.insert_text_index_with_wire_bytes(key.clone(), index, 1024);
        assert!(cache.get_text_index(&key).is_none());
        assert_eq!(cache.text_index_capacity_rejections(), 3);
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

        let preflight = cache
            .admit_vector_index_wire_bytes(&key, 1024, 2, 2)
            .unwrap_err();
        assert!(preflight.required_bytes > preflight.capacity_bytes);
        assert!(!cache.can_admit_vector_index_wire_bytes(&key, 1024));
        let insertion = cache
            .try_insert_vector_index_with_wire_bytes(key.clone(), index.clone(), 1024)
            .unwrap_err();
        assert_eq!(insertion, preflight);
        assert!(cache.get_vector_index(&key).is_none());
        assert_eq!(cache.search_index_usage_bytes(), 0);
        assert_eq!(
            cache.vector_index_usage_bytes(),
            cache.search_index_usage_bytes()
        );
        assert_eq!(
            cache.vector_index_capacity_bytes(),
            cache.search_index_capacity_bytes()
        );
        assert_eq!(cache.vector_index_capacity_rejections(), 2);

        cache.insert_vector_index_with_wire_bytes(key.clone(), index, 1024);
        assert!(cache.get_vector_index(&key).is_none());
        assert_eq!(cache.vector_index_capacity_rejections(), 3);
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
    fn pruned_namespace_tombstones_are_bounded_without_empty_live_sets() {
        let cache = tight_cache(4 << 20);
        let live_path = "tenants/gone/sst/level0/live.parquet".to_string();
        cache.retain_paths("tenants/gone", &HashSet::from([live_path]));
        assert_eq!(
            cache.tracked_paths.lock().unwrap().live_by_namespace.len(),
            1
        );

        cache.prune_namespace("tenants/gone");
        for i in 0..MAX_DENIED_CACHE_NAMESPACES + 128 {
            cache.prune_namespace(&format!("tenants/empty-{i}"));
        }

        let paths = cache.tracked_paths.lock().unwrap();
        assert!(
            paths.live_by_namespace.is_empty(),
            "deny-all tombstones must not be represented as empty live sets"
        );
        assert_eq!(
            paths.live_by_namespace.capacity(),
            0,
            "removing the last live namespace must release registry buckets"
        );
        assert_eq!(
            paths.denied_namespaces.len(),
            MAX_DENIED_CACHE_NAMESPACES,
            "namespace churn must not grow eviction admission state forever"
        );
        assert_eq!(paths.denied_order.len(), MAX_DENIED_CACHE_NAMESPACES);
        assert!(paths.used_bytes() <= paths.capacity_bytes);
    }

    #[test]
    fn evicted_namespace_rejects_late_decode_until_reopened() {
        let cache = tight_cache(1 << 20);
        let stale = "tenants/evicted/sst/level0/stale.parquet".to_string();
        cache.retain_paths("tenants/evicted", &HashSet::from([stale.clone()]));

        // Model a read that missed the cache and began decoding before the
        // namespace was evicted, but reached its cache insert afterwards.
        cache.prune_namespace("tenants/evicted");
        cache.insert(stale.clone(), Bytes::from_static(b"late body"));
        cache.insert_edge_streams(
            stale.clone(),
            Arc::new(EdgeStreamBundle {
                overflow: None,
                declared: Vec::new(),
            }),
        );
        assert!(cache.get(&stale).is_none());
        assert!(cache.get_edge_streams(&stale).is_none());

        // Reopening clears the eviction tombstone, but its empty manifest is
        // still an authoritative deny-all. Publishing the first successful
        // flush's live set admits only that exact new path.
        cache.retain_paths("tenants/evicted", &HashSet::new());
        let fresh = "tenants/evicted/sst/level0/fresh.parquet".to_string();
        cache.insert(fresh.clone(), Bytes::from_static(b"fresh body"));
        assert!(cache.get(&fresh).is_none());
        cache.retain_paths("tenants/evicted", &HashSet::from([fresh.clone()]));
        cache.insert(fresh.clone(), Bytes::from_static(b"fresh body"));
        assert!(cache.get(&fresh).is_some());
    }

    #[test]
    fn transition_to_empty_manifest_rejects_late_decode_even_after_clear() {
        let cache = tight_cache(1 << 20);
        let stale = "tenants/gc/sst/level1/stale.parquet".to_string();
        cache.retain_paths("tenants/gc", &HashSet::from([stale.clone()]));
        cache.insert(stale.clone(), Bytes::from_static(b"old body"));
        assert!(cache.get(&stale).is_some());

        // Model compaction/GC publishing a manifest with no immutable objects,
        // then a pre-commit read finishing its decode after the retain.
        cache.retain_paths("tenants/gc", &HashSet::new());
        cache.insert(stale.clone(), Bytes::from_static(b"late body"));
        assert!(cache.get(&stale).is_none());

        // RSS pressure must not erase the authoritative empty live rule.
        cache.clear();
        cache.insert(stale.clone(), Bytes::from_static(b"later body"));
        assert!(cache.get(&stale).is_none());
    }

    #[test]
    fn embedded_empty_namespace_rules_are_fifo_bounded() {
        let cache = tight_cache(4 << 20);
        for i in 0..MAX_EMPTY_LIVE_NAMESPACES + 128 {
            cache.retain_paths(&format!("embedded/empty-{i}"), &HashSet::new());
        }
        let paths = cache.tracked_paths.lock().unwrap();
        assert_eq!(paths.empty_live_order.len(), MAX_EMPTY_LIVE_NAMESPACES);
        assert_eq!(
            paths
                .live_by_namespace
                .values()
                .filter(|live| live.is_empty())
                .count(),
            MAX_EMPTY_LIVE_NAMESPACES,
            "fresh embedded namespace churn must not retain empty rules forever"
        );
        assert!(paths.used_bytes() <= paths.capacity_bytes);
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
    fn clear_drops_all_tiers_and_preserves_live_path_admission_rules() {
        let cache = tight_cache(1 << 20);
        let live = "tenants/a/sst/level0/live.parquet".to_string();
        let dead = "tenants/a/sst/level0/dead.parquet".to_string();
        cache.retain_paths("tenants/a", &HashSet::from([live.clone()]));
        cache.insert(live.clone(), Bytes::from_static(b"body"));
        cache.insert_equality_property_sidecar(
            live.clone(),
            Arc::new(BTreeMap::from([("s:k".into(), vec![[1; 16]])])),
        );
        cache.insert_bloom_filter(live.clone(), Arc::new(BloomFilter::with_capacity(10, 10)));
        assert!(cache.aggregate_usage_bytes() > 0);
        assert_eq!(cache.namespace_side_entries("tenants/a"), 3);
        assert!(cache.tracked_paths.lock().unwrap().entries.capacity() > 0);

        cache.clear();
        assert_eq!(
            cache.aggregate_usage_bytes(),
            cache.path_registry_usage_bytes(),
            "clear drops residents but preserves the authoritative live rule"
        );
        assert_eq!(cache.namespace_side_entries("tenants/a"), 0);
        assert_eq!(
            cache.tracked_paths.lock().unwrap().entries.capacity(),
            0,
            "pressure clear must release the tracking table allocation"
        );
        assert!(cache.get(&live).is_none());
        assert!(cache.get_equality_property_sidecar(&live).is_none());
        assert!(cache.get_bloom_filter(&live).is_none());

        // Clearing residents must not let a decode racing an older manifest
        // repopulate a path already declared dead.
        cache.insert(dead.clone(), Bytes::from_static(b"obsolete"));
        assert!(cache.get(&dead).is_none());
        cache.insert(live.clone(), Bytes::from_static(b"current"));
        assert!(cache.get(&live).is_some());
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
    fn natural_foyer_evictions_remove_path_metadata_without_a_diagnostic_sweep() {
        let budget = 4 * 1024;
        let cache = tight_cache(budget);
        let inserted = 256usize;

        for i in 0..inserted {
            cache.insert(
                format!("tenants/churn/sst/level0/{i:04}.parquet"),
                Bytes::from(vec![i as u8; 512]),
            );
        }

        // Inspect Foyer directly; calling `namespace_side_entries` here would
        // mask the original leak by performing its legacy lazy sweep.
        let resident = (0..inserted)
            .filter(|i| {
                cache
                    .inner
                    .get(&format!("tenants/churn/sst/level0/{i:04}.parquet"))
                    .is_some()
            })
            .count();
        let paths = cache.tracked_paths.lock().unwrap();
        assert_eq!(
            paths.entries.len(),
            resident,
            "eviction listeners must remove registry entries synchronously"
        );
        assert!(
            paths.entries.len() < inserted / 4,
            "path metadata must follow the bounded resident set"
        );
        assert_eq!(
            paths.entry_bytes,
            paths
                .entries
                .iter()
                .map(tracked_cache_entry_metadata_bytes)
                .sum::<usize>(),
            "listener removals must keep the metadata counter exact"
        );
        assert!(paths.used_bytes() <= paths.capacity_bytes);
    }

    #[test]
    fn natural_eviction_releases_registry_bucket_high_water() {
        let budget = 64 * 1024;
        let cache = tight_cache(budget);
        let inserted = 256usize;
        for i in 0..inserted {
            cache.insert(
                format!("tenants/shape/sst/level0/{i:04}.parquet"),
                Bytes::new(),
            );
        }
        assert!(cache.tracked_paths.lock().unwrap().entries.capacity() > 64);

        let large_path = "tenants/shape/sst/level0/large.parquet".to_string();
        cache.insert(large_path.clone(), Bytes::from(vec![0; 48 * 1024]));

        let resident = (0..inserted)
            .filter(|i| {
                cache
                    .inner
                    .get(&format!("tenants/shape/sst/level0/{i:04}.parquet"))
                    .is_some()
            })
            .count()
            + usize::from(cache.inner.get(&large_path).is_some());
        let registry = cache.tracked_paths.lock().unwrap();
        assert_eq!(registry.entries.len(), resident);
        assert!(
            registry.entries.capacity() <= registry.entries.len().saturating_mul(2).max(64),
            "listener removals must release historical registry buckets"
        );
        assert!(registry.used_bytes() <= registry.capacity_bytes);
    }

    #[test]
    fn oversized_manifest_rule_fails_closed_with_bounded_metadata() {
        let registry_budget = 1024;
        let cache = SstCache::with_all_budgets(
            64 * 1024,
            64 * 1024,
            64 * 1024,
            registry_budget,
            DecodedCacheBudgets::uniform(64 * 1024),
        );
        let prefix = "tenants/huge";
        let live: HashSet<String> = (0..64)
            .map(|i| format!("{prefix}/sst/level0/{i}-{}.parquet", "x".repeat(128)))
            .collect();
        cache.retain_paths(prefix, &live);

        let path = live.iter().next().unwrap().clone();
        cache.insert(path.clone(), Bytes::from_static(b"body"));
        let registry = cache.tracked_paths.lock().unwrap();
        assert!(registry.admission_disabled);
        assert!(registry.used_bytes() <= registry_budget);
        drop(registry);
        assert!(
            cache.get(&path).is_none(),
            "an unrepresentable authoritative rule must disable admission"
        );
    }

    #[test]
    fn live_rule_does_not_clone_the_callers_sparse_hash_table_capacity() {
        let registry_budget = 8 * 1024;
        let cache = SstCache::with_all_budgets(
            64 * 1024,
            64 * 1024,
            64 * 1024,
            registry_budget,
            DecodedCacheBudgets::uniform(64 * 1024),
        );
        let path = "tenants/sparse/sst/level0/live.parquet".to_string();
        let mut live = HashSet::with_capacity(100_000);
        live.insert(path.clone());
        assert!(live.capacity() > 10_000);

        cache.retain_paths("tenants/sparse", &live);
        cache.insert(path.clone(), Bytes::from_static(b"body"));

        let registry = cache.tracked_paths.lock().unwrap();
        let stored = registry
            .live_by_namespace
            .get("tenants/sparse/")
            .expect("authoritative rule");
        assert!(
            stored.capacity() < 128,
            "the registry must rebuild a compact live set"
        );
        assert!(!registry.admission_disabled);
        assert!(registry.used_bytes() <= registry_budget);
        drop(registry);
        assert!(cache.get(&path).is_some());
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

#[cfg(test)]
mod tuple_key_tests {
    use super::*;

    fn key(values: &[&Value]) -> Option<Vec<u8>> {
        encode_equality_tuple_key(values)
    }

    /// The adversarial aliasing rows of the composite test matrix: no
    /// member byte pattern may alias another member, a part boundary, or a
    /// different type carrying the same surface bytes.
    #[test]
    fn tuple_keys_are_unambiguous_and_typed() {
        // Boundary shifting: ("a","bc") != ("ab","c").
        let a_bc = key(&[&Value::Str("a".into()), &Value::Str("bc".into())]).unwrap();
        let ab_c = key(&[&Value::Str("ab".into()), &Value::Str("c".into())]).unwrap();
        assert_ne!(a_bc, ab_c);
        // A raw string that looks like the scalar Bool tag cannot collide
        // with an actual Bool member.
        let fake = key(&[&Value::Str("b:1".into()), &Value::I64(7)]).unwrap();
        let real = key(&[&Value::Bool(true), &Value::I64(7)]).unwrap();
        assert_ne!(fake, real);
        // Numeric members canonicalize: Cypher's `1 = 1.0` is TRUE, so both
        // encodings MUST collide (the scan route would return the row; the
        // index route must never lose it). -0.0 folds into 0.0, and integer
        // zero joins them.
        let int = key(&[&Value::I64(1), &Value::Str("x".into())]).unwrap();
        let float = key(&[&Value::F64(1.0), &Value::Str("x".into())]).unwrap();
        assert_eq!(int, float);
        let pos = key(&[&Value::F64(0.0)]).unwrap();
        let neg = key(&[&Value::F64(-0.0)]).unwrap();
        let zero = key(&[&Value::I64(0)]).unwrap();
        assert_eq!(pos, neg);
        assert_eq!(pos, zero);
        // Above 2^53 the canonicalization is lossy: DISTINCT integers may
        // share a posting (both round to the same f64). That is a false
        // POSITIVE only — confirmation separates them, because same-type
        // integers compare exactly; false negatives remain impossible.
        let big_a = key(&[&Value::I64((1 << 53) + 1)]).unwrap();
        let big_b = key(&[&Value::I64(1 << 53)]).unwrap();
        assert_eq!(big_a, big_b);
        assert!(!cypher_scalar_equal(
            &Value::I64((1 << 53) + 1),
            &Value::I64(1 << 53)
        ));
        // The confirmation twin agrees with the executor's coercion.
        assert!(cypher_scalar_equal(&Value::I64(30), &Value::F64(30.0)));
        assert!(!cypher_scalar_equal(&Value::I64(30), &Value::F64(30.5)));
        assert!(!cypher_scalar_equal(&Value::Null, &Value::Null));
        // Order matters: declaration order IS the layout.
        let xy = key(&[&Value::Str("x".into()), &Value::Str("y".into())]).unwrap();
        let yx = key(&[&Value::Str("y".into()), &Value::Str("x".into())]).unwrap();
        assert_ne!(xy, yx);
        // Deterministic across calls.
        assert_eq!(
            key(&[&Value::Date(123), &Value::DateTime(456)]),
            key(&[&Value::Date(123), &Value::DateTime(456)])
        );
    }

    /// A NaN / Null / non-scalar member poisons the whole tuple: the row is
    /// never filed and the probe must fall back — mirroring the flat scan
    /// (NaN != NaN) and the transactional tuple probe's Unindexable answer.
    #[test]
    fn unindexable_members_poison_the_tuple() {
        assert!(key(&[&Value::Str("x".into()), &Value::Null]).is_none());
        assert!(key(&[&Value::F64(f64::NAN), &Value::Str("x".into())]).is_none());
        assert!(key(&[&Value::List(Vec::new()), &Value::I64(1)]).is_none());
        assert!(key(&[&Value::Map(Default::default()), &Value::I64(1)]).is_none());
    }
}
