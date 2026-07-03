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

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use arrow_array::RecordBatch;
use bytes::Bytes;
use foyer::{Cache, CacheBuilder};
use parquet::file::metadata::ParquetMetaData;
use std::collections::HashMap;

/// Default budget for an [`SstCache`]: 256 MiB. Override via
/// `NAMIDB_SST_CACHE_BUDGET_MIB`.
pub const DEFAULT_SST_CACHE_BUDGET_MIB: usize = 256;

/// Default budget for the decoded node row-group cache: 256 MiB.
/// Override via `NAMIDB_DECODED_NODE_RG_CACHE_BUDGET_MIB`. Decoded
/// `RecordBatch`es are typically several times their on-disk size, so
/// this tier gets its own budget rather than sharing the body budget.
pub const DEFAULT_DECODED_NODE_RG_CACHE_BUDGET_MIB: usize = 256;

/// Read `NAMIDB_DECODED_NODE_RG_CACHE_BUDGET_MIB` or fall back to
/// [`DEFAULT_DECODED_NODE_RG_CACHE_BUDGET_MIB`].
pub fn decoded_node_rg_cache_budget_bytes() -> usize {
    let mib = std::env::var("NAMIDB_DECODED_NODE_RG_CACHE_BUDGET_MIB")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(DEFAULT_DECODED_NODE_RG_CACHE_BUDGET_MIB);
    mib.saturating_mul(1024 * 1024)
}

/// Key for one decoded node-SST row group: `(absolute SST path,
/// row-group index)`.
pub type NodeRowGroupKey = (String, usize);
/// Decoded batches of one node-SST row group.
pub type DecodedNodeRowGroup = Arc<Vec<RecordBatch>>;

/// Weight of one decoded node row-group entry: key bytes plus the
/// Arrow-reported memory footprint of every decoded batch. Shared with
/// tests so budget assertions use the exact accounting the cache does.
pub(crate) fn decoded_node_row_group_weight(
    key: &NodeRowGroupKey,
    value: &DecodedNodeRowGroup,
) -> usize {
    key.0.len()
        + std::mem::size_of::<usize>()
        + value
            .iter()
            .map(|b| b.get_array_memory_size())
            .sum::<usize>()
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
    let mib = std::env::var("NAMIDB_SST_CACHE_BUDGET_MIB")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(DEFAULT_SST_CACHE_BUDGET_MIB);
    mib.saturating_mul(1024 * 1024)
}

/// Process-wide shared [`SstCache`]: one instance for every
/// [`crate::WriterSession`] the process opens, so `NAMIDB_SST_CACHE_BUDGET_MIB`
/// (and the decoded row-group budget) bound the PROCESS, not each session —
/// a multi-tenant host serving N namespaces holds one budget, not N.
///
/// Sharing across namespaces is sound because every key in every tier is
/// an absolute object-store path (namespace-prefixed) or `(absolute path,
/// row-group index)`: two namespaces can never collide on a key.
///
/// The enable flag and budgets are read once, on first use; later env
/// mutations don't resize the shared instance. Returns `None` when
/// `NAMIDB_SST_CACHE=0` at first use. Callers needing private budgets
/// (tests, embedded hosts with several object stores) construct their own
/// [`SstCache`] and inject it via
/// [`crate::ingest::WriterSession::open_with_caches`].
pub fn shared_sst_cache() -> Option<SstCache> {
    static SHARED: OnceLock<Option<SstCache>> = OnceLock::new();
    SHARED
        .get_or_init(|| sst_cache_enabled().then(|| SstCache::new(sst_cache_budget_bytes())))
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
}

/// Process-wide cache shared between [`crate::Snapshot`] instances.
#[derive(Clone)]
pub struct SstCache {
    inner: Arc<Cache<String, Bytes>>,
    /// Decoded node-SST row groups keyed by `(absolute SST path, row-group
    /// index)`. Populated by `Snapshot::batch_lookup_nodes` and consulted by
    /// the per-id lookup cold path so a batch prewarm keeps paying off across
    /// snapshots. Weighted by the decoded Arrow footprint against its own
    /// byte budget (see [`decoded_node_rg_cache_budget_bytes`]); over-eviction
    /// is safe because the read path re-decodes evicted row groups on demand.
    decoded_node_row_groups: Arc<Cache<NodeRowGroupKey, DecodedNodeRowGroup>>,
    /// Parsed Parquet metadata (footer + page index) per SST path.
    /// Populated by the RFC-003 ranged-read path; saves one round-trip
    /// per warm ranged lookup. Unbounded map for now — capped by the
    /// number of SSTs in the namespace (low hundreds in practice,
    /// each metadata blob ~100–400 KiB). Eviction-by-LRU is a TODO.
    metadata: Arc<Mutex<HashMap<String, Arc<ParquetMetaData>>>>,
    /// Decoded edge property streams per SST path. Same lifetime story
    /// as `metadata` (SSTs are immutable per path), unbounded for now.
    edge_streams: Arc<Mutex<HashMap<String, Arc<EdgeStreamBundle>>>>,
    /// Edge SST readers (header + footer + fence index + precomputed
    /// `cumulative_edges`) keyed by absolute path. `EdgeSstReader::open`
    /// is `O(edge_count)` because it walks every partner block to build
    /// the cumulative-edges prefix sum. Caching the reader makes the
    /// second + every subsequent `edge_lookup_via_sst` against the same
    /// SST run in `O(deg)` instead of `O(edge_count)`. Memory: ~8 B per
    /// edge in the SST.
    edge_readers: Arc<Mutex<HashMap<String, Arc<crate::sst::edges::EdgeSstReader>>>>,
    /// Decoded `.ft` text indexes per SST path. Decoding bincode-deserialises
    /// the whole inverted index; without this every `search.bm25` paid
    /// `O(index size)` per query even with the body bytes cached.
    #[cfg(feature = "text-index")]
    text_indexes: Arc<Mutex<HashMap<String, Arc<crate::sst::text::TextIndex>>>>,
    /// Decoded `.vg` vector indexes per SST path. Decoding deserialises every
    /// stored vector plus the full Vamana adjacency AND clones the vectors into
    /// the navigation space; without this every KNN (and each widening round)
    /// paid `O(index size)` per query.
    #[cfg(feature = "vector-index")]
    vector_indexes: Arc<Mutex<HashMap<String, Arc<crate::sst::vector::VectorGraphIndex>>>>,
    stats: Arc<CacheStats>,
}

impl std::fmt::Debug for SstCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // foyer's `Cache` doesn't impl Debug and only exposes
        // `capacity`/`usage`; surface those alongside our own counters.
        f.debug_struct("SstCache")
            .field("capacity_bytes", &self.inner.capacity())
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
                "node_rg_usage_bytes",
                &self.decoded_node_row_groups.usage(),
            )
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
            .finish()
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
        let inner = CacheBuilder::new(capacity_bytes.max(1))
            .with_weighter(|key: &String, value: &Bytes| key.len() + value.len())
            .build();
        let decoded_node_row_groups = CacheBuilder::new(decoded_node_rg_bytes.max(1))
            .with_weighter(decoded_node_row_group_weight)
            .build();
        Self {
            inner: Arc::new(inner),
            decoded_node_row_groups: Arc::new(decoded_node_row_groups),
            metadata: Arc::new(Mutex::new(HashMap::new())),
            edge_streams: Arc::new(Mutex::new(HashMap::new())),
            edge_readers: Arc::new(Mutex::new(HashMap::new())),
            #[cfg(feature = "text-index")]
            text_indexes: Arc::new(Mutex::new(HashMap::new())),
            #[cfg(feature = "vector-index")]
            vector_indexes: Arc::new(Mutex::new(HashMap::new())),
            stats: Arc::new(CacheStats::default()),
        }
    }

    /// Look up a decoded text index for an SST path. Returns `None` on miss;
    /// the caller decodes once and re-inserts via [`Self::insert_text_index`].
    /// SSTs are immutable per UUIDv7-keyed path so cached indexes never go
    /// stale; superseded paths are pruned by [`Self::retain_paths`].
    #[cfg(feature = "text-index")]
    pub fn get_text_index(&self, key: &str) -> Option<Arc<crate::sst::text::TextIndex>> {
        self.text_indexes.lock().unwrap().get(key).cloned()
    }

    /// Store a decoded text index for an SST path.
    #[cfg(feature = "text-index")]
    pub fn insert_text_index(&self, key: String, idx: Arc<crate::sst::text::TextIndex>) {
        self.text_indexes.lock().unwrap().insert(key, idx);
    }

    /// Look up a decoded vector index for an SST path. Same contract as
    /// [`Self::get_text_index`].
    #[cfg(feature = "vector-index")]
    pub fn get_vector_index(&self, key: &str) -> Option<Arc<crate::sst::vector::VectorGraphIndex>> {
        self.vector_indexes.lock().unwrap().get(key).cloned()
    }

    /// Store a decoded vector index for an SST path.
    #[cfg(feature = "vector-index")]
    pub fn insert_vector_index(&self, key: String, idx: Arc<crate::sst::vector::VectorGraphIndex>) {
        self.vector_indexes.lock().unwrap().insert(key, idx);
    }

    /// Look up a cached [`crate::sst::edges::EdgeSstReader`] for an SST
    /// path. Returns `None` on miss; the caller calls
    /// [`crate::sst::edges::EdgeSstReader::open`] once and re-inserts
    /// via [`Self::insert_edge_reader`].
    pub fn get_edge_reader(&self, key: &str) -> Option<Arc<crate::sst::edges::EdgeSstReader>> {
        let map = self.edge_readers.lock().unwrap();
        match map.get(key) {
            Some(r) => {
                self.stats.edge_readers_hits.fetch_add(1, Ordering::Relaxed);
                Some(r.clone())
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
        let mut map = self.edge_readers.lock().unwrap();
        map.insert(key, reader);
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
        self.decoded_node_row_groups.insert((key, row_group), batches);
    }

    /// Bytes held by the decoded node row-group tier (sum of entry weights).
    pub fn decoded_node_row_groups_usage(&self) -> usize {
        self.decoded_node_row_groups.usage()
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
        let map = self.edge_streams.lock().unwrap();
        match map.get(key) {
            Some(b) => {
                self.stats.edge_streams_hits.fetch_add(1, Ordering::Relaxed);
                Some(b.clone())
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
        let mut map = self.edge_streams.lock().unwrap();
        map.insert(key, bundle);
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

    /// Look up Parquet metadata for an SST path (RFC-003). Returns
    /// `None` on miss; the ranged-read path will fetch the footer +
    /// page index and re-insert via [`Self::insert_metadata`].
    pub fn get_metadata(&self, key: &str) -> Option<Arc<ParquetMetaData>> {
        let map = self.metadata.lock().unwrap();
        match map.get(key) {
            Some(meta) => {
                self.stats.meta_hits.fetch_add(1, Ordering::Relaxed);
                Some(meta.clone())
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
        let mut map = self.metadata.lock().unwrap();
        map.insert(key, meta);
    }

    /// Drop side-map entries (Parquet metadata, decoded edge streams, edge
    /// readers) under `namespace_prefix` whose SST path is no longer `live`.
    /// The maps are keyed by absolute SST path and were insert-only, so under
    /// normal flush/compaction churn they grew without bound (a 10M-edge SST's
    /// decoded streams are ~1 GB per entry). Called after a manifest commit
    /// with the paths the new manifest still references; the byte-bounded body
    /// cache (`inner`) is unaffected. Over-eviction is safe (entries re-decode
    /// on demand).
    ///
    /// The prune is scoped to `namespace_prefix` (`<root>/<ns>`, with or
    /// without a trailing slash) because the cache is shared process-wide:
    /// one namespace's flush knows only its OWN live set, so it must never
    /// touch sibling namespaces' entries — a global retain here would evict
    /// every other tenant's warm state on each flush.
    pub fn retain_paths(&self, namespace_prefix: &str, live: &std::collections::HashSet<String>) {
        // Normalise to a path-segment boundary so "tenants/acme" cannot
        // match "tenants/acme2/...".
        let mut prefix = namespace_prefix.to_string();
        if !prefix.ends_with('/') {
            prefix.push('/');
        }
        let keep = |k: &String| !k.starts_with(&prefix) || live.contains(k);
        self.metadata.lock().unwrap().retain(|k, _| keep(k));
        self.edge_streams.lock().unwrap().retain(|k, _| keep(k));
        self.edge_readers.lock().unwrap().retain(|k, _| keep(k));
        #[cfg(feature = "text-index")]
        self.text_indexes.lock().unwrap().retain(|k, _| keep(k));
        #[cfg(feature = "vector-index")]
        self.vector_indexes.lock().unwrap().retain(|k, _| keep(k));
    }

    /// Eagerly drop every side-map entry under `namespace_prefix`. Called
    /// when a multi-tenant host evicts a namespace — its state is being
    /// dropped anyway, so its decoded metadata/streams/readers/indexes are
    /// dead weight in the shared cache. The byte-budgeted foyer tiers (SST
    /// bodies, decoded node row groups) expose no iteration API; their
    /// entries are reclaimed lazily by budget eviction, which is safe
    /// because both tiers are strictly byte-bounded.
    pub fn prune_namespace(&self, namespace_prefix: &str) {
        self.retain_paths(namespace_prefix, &std::collections::HashSet::new());
    }

    /// Count of side-map entries (across the path-keyed maps) whose SST path
    /// sits under `namespace_prefix`. Observability + test probe for the
    /// namespace-scoped [`Self::retain_paths`] / [`Self::prune_namespace`].
    pub fn namespace_side_entries(&self, namespace_prefix: &str) -> usize {
        let mut prefix = namespace_prefix.to_string();
        if !prefix.ends_with('/') {
            prefix.push('/');
        }
        let count = |keys: Vec<String>| keys.iter().filter(|k| k.starts_with(&prefix)).count();
        let mut n = count(self.metadata.lock().unwrap().keys().cloned().collect());
        n += count(self.edge_streams.lock().unwrap().keys().cloned().collect());
        n += count(self.edge_readers.lock().unwrap().keys().cloned().collect());
        #[cfg(feature = "text-index")]
        {
            n += count(self.text_indexes.lock().unwrap().keys().cloned().collect());
        }
        #[cfg(feature = "vector-index")]
        {
            n += count(self.vector_indexes.lock().unwrap().keys().cloned().collect());
        }
        n
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
        self.inner.insert(key, value);
    }

    /// Current cache usage in bytes (sum of weights of live entries).
    pub fn usage(&self) -> usize {
        self.inner.usage()
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
    fn retain_paths_only_prunes_the_given_namespace() {
        let cache = SstCache::new(1 << 20);
        let bundle = || Arc::new(EdgeStreamBundle {
            overflow: None,
            declared: Vec::new(),
        });
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

        assert!(cache.get_edge_streams(&a_live).is_some(), "a's live entry kept");
        assert!(cache.get_edge_streams(&a_dead).is_none(), "a's dead entry pruned");
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
        let bundle = || Arc::new(EdgeStreamBundle {
            overflow: None,
            declared: Vec::new(),
        });
        cache.insert_edge_streams("tenants/gone/sst/level0/a.csr".into(), bundle());
        cache.insert_edge_streams("tenants/gone/sst/level0/b.csr".into(), bundle());
        cache.insert_edge_streams("tenants/kept/sst/level0/c.csr".into(), bundle());
        assert_eq!(cache.namespace_side_entries("tenants/gone"), 2);

        cache.prune_namespace("tenants/gone");
        assert_eq!(cache.namespace_side_entries("tenants/gone"), 0);
        assert_eq!(cache.namespace_side_entries("tenants/kept"), 1);
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
