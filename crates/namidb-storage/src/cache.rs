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
use std::sync::{Arc, Mutex};

use bytes::Bytes;
use foyer::{Cache, CacheBuilder};
use parquet::file::metadata::ParquetMetaData;
use std::collections::HashMap;

/// Default budget for an [`SstCache`]: 256 MiB. Override via
/// `NAMIDB_SST_CACHE_BUDGET_MIB`.
pub const DEFAULT_SST_CACHE_BUDGET_MIB: usize = 256;

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
}

/// Process-wide cache shared between [`crate::Snapshot`] instances.
#[derive(Clone)]
pub struct SstCache {
    inner: Arc<Cache<String, Bytes>>,
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
            .finish()
    }
}

impl SstCache {
    /// Build a new cache sized for `capacity_bytes`. Entries weight as
    /// `key.len() + value.len()` so the budget is in real bytes.
    pub fn new(capacity_bytes: usize) -> Self {
        let inner = CacheBuilder::new(capacity_bytes.max(1))
            .with_weighter(|key: &String, value: &Bytes| key.len() + value.len())
            .build();
        Self {
            inner: Arc::new(inner),
            metadata: Arc::new(Mutex::new(HashMap::new())),
            edge_streams: Arc::new(Mutex::new(HashMap::new())),
            edge_readers: Arc::new(Mutex::new(HashMap::new())),
            stats: Arc::new(CacheStats::default()),
        }
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
