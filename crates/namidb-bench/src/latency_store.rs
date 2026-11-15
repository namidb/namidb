//! An [`ObjectStore`] layer that simulates a remote object store.
//!
//! Every request waits a per-request latency drawn uniformly from
//! `[min_ms, max_ms]` before it reaches the inner store, plus an optional
//! transfer time at a fixed bandwidth. Requests run concurrently, exactly as
//! they would against S3: the delay is a sleep, not a lock. Each request is
//! also counted per operation and per key class (WAL, manifest body, pointer,
//! SST, ...), so a benchmark can report how many round trips a workload paid.
//!
//! The layer forwards every operation unchanged, conditional puts included,
//! so the engine's commit protocol sees the inner store's semantics.

use std::collections::BTreeMap;
use std::fmt;
use std::ops::Range;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use async_trait::async_trait;
use bytes::Bytes;
use futures::stream::BoxStream;
use futures::{FutureExt, StreamExt};
use object_store::path::Path;
use object_store::{
    CopyOptions, GetOptions, GetResult, ListResult, MultipartUpload, ObjectMeta, ObjectStore,
    PutMultipartOptions, PutOptions, PutPayload, PutResult, UploadPart,
};
use serde::{Deserialize, Serialize};

/// Latency model of the simulated store.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LatencyModel {
    /// Lower bound of the per-request latency.
    pub min: Duration,
    /// Upper bound of the per-request latency.
    pub max: Duration,
    /// Transfer rate charged on request and response bodies, in bytes per
    /// second. `0` disables the transfer cost.
    pub bytes_per_sec: u64,
}

impl LatencyModel {
    pub fn new(min_ms: u64, max_ms: u64, mib_per_sec: u64) -> Self {
        Self {
            min: Duration::from_millis(min_ms.min(max_ms)),
            max: Duration::from_millis(max_ms.max(min_ms)),
            bytes_per_sec: mib_per_sec.saturating_mul(1024 * 1024),
        }
    }
}

/// Request counters, by operation and by key class.
#[derive(Debug, Default)]
pub struct StoreCounters {
    by_op: Mutex<BTreeMap<String, u64>>,
    bytes_in: AtomicU64,
    bytes_out: AtomicU64,
    requests: AtomicU64,
}

/// A point-in-time copy of [`StoreCounters`].
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct StoreCountersSnapshot {
    /// `"<op> <class>" -> count`, e.g. `"HEAD sst"`.
    pub by_op: BTreeMap<String, u64>,
    pub requests: u64,
    pub bytes_written: u64,
    pub bytes_read: u64,
}

impl StoreCountersSnapshot {
    /// Counters accumulated since `earlier`.
    pub fn since(&self, earlier: &StoreCountersSnapshot) -> StoreCountersSnapshot {
        let mut by_op = BTreeMap::new();
        for (key, value) in &self.by_op {
            let delta = value.saturating_sub(earlier.by_op.get(key).copied().unwrap_or(0));
            if delta > 0 {
                by_op.insert(key.clone(), delta);
            }
        }
        StoreCountersSnapshot {
            by_op,
            requests: self.requests.saturating_sub(earlier.requests),
            bytes_written: self.bytes_written.saturating_sub(earlier.bytes_written),
            bytes_read: self.bytes_read.saturating_sub(earlier.bytes_read),
        }
    }
}

impl StoreCounters {
    fn record(&self, op: &str, location: &Path) {
        self.requests.fetch_add(1, Ordering::Relaxed);
        let key = format!("{op} {}", key_class(location));
        let mut by_op = self.by_op.lock().unwrap_or_else(|e| e.into_inner());
        *by_op.entry(key).or_insert(0) += 1;
    }

    pub fn snapshot(&self) -> StoreCountersSnapshot {
        StoreCountersSnapshot {
            by_op: self.by_op.lock().unwrap_or_else(|e| e.into_inner()).clone(),
            requests: self.requests.load(Ordering::Relaxed),
            bytes_written: self.bytes_in.load(Ordering::Relaxed),
            bytes_read: self.bytes_out.load(Ordering::Relaxed),
        }
    }
}

/// Coarse class of an object key, for the counters.
pub fn key_class(location: &Path) -> &'static str {
    let path = location.as_ref();
    if path.contains("/wal/") || path.starts_with("wal/") {
        "wal"
    } else if path.contains("/pointer/") {
        "pointer"
    } else if path.ends_with("manifest/current.json") {
        "current"
    } else if path.contains("/pins/") {
        "pin"
    } else if path.contains("/manifest/") || path.starts_with("manifest/") {
        "manifest"
    } else if path.contains("/sst/") || path.starts_with("sst/") {
        "sst"
    } else if path.contains("memtable") || path.contains("snapshot") {
        "snapshot"
    } else {
        "other"
    }
}

/// The simulating layer. See the module docs.
pub struct LatencyStore {
    inner: Arc<dyn ObjectStore>,
    model: LatencyModel,
    counters: Arc<StoreCounters>,
    seq: AtomicU64,
    started: Instant,
    trace: bool,
}

impl LatencyStore {
    pub fn new(
        inner: Arc<dyn ObjectStore>,
        model: LatencyModel,
        counters: Arc<StoreCounters>,
    ) -> Self {
        Self {
            inner,
            model,
            counters,
            seq: AtomicU64::new(0),
            started: Instant::now(),
            trace: std::env::var_os("NAMIDB_BENCH_STORE_TRACE").is_some(),
        }
    }

    /// Deterministic per-request latency: a splitmix64 of the request number
    /// spread over `[min, max]`.
    fn request_latency(&self) -> Duration {
        let span = self.model.max.saturating_sub(self.model.min);
        if span.is_zero() {
            return self.model.min;
        }
        let mut z = self
            .seq
            .fetch_add(1, Ordering::Relaxed)
            .wrapping_add(0x9E37_79B9_7F4A_7C15);
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        let micros = span.as_micros() as u64;
        self.model.min + Duration::from_micros(z % (micros + 1))
    }

    fn transfer_time(&self, bytes: u64) -> Duration {
        if self.model.bytes_per_sec == 0 || bytes == 0 {
            return Duration::ZERO;
        }
        Duration::from_secs_f64(bytes as f64 / self.model.bytes_per_sec as f64)
    }

    async fn before(&self, op: &str, location: &Path, upload_bytes: u64) {
        self.counters.record(op, location);
        self.counters
            .bytes_in
            .fetch_add(upload_bytes, Ordering::Relaxed);
        if self.trace {
            eprintln!(
                "{:>9.3}s {op:<6} {location}",
                self.started.elapsed().as_secs_f64()
            );
        }
        let wait = self.request_latency() + self.transfer_time(upload_bytes);
        if !wait.is_zero() {
            tokio::time::sleep(wait).await;
        }
    }

    async fn after_download(&self, bytes: u64) {
        self.counters.bytes_out.fetch_add(bytes, Ordering::Relaxed);
        let wait = self.transfer_time(bytes);
        if !wait.is_zero() {
            tokio::time::sleep(wait).await;
        }
    }
}

impl fmt::Debug for LatencyStore {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LatencyStore")
            .field("inner", &self.inner)
            .field("model", &self.model)
            .finish()
    }
}

impl fmt::Display for LatencyStore {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "LatencyStore({})", self.inner)
    }
}

#[async_trait]
impl ObjectStore for LatencyStore {
    async fn put_opts(
        &self,
        location: &Path,
        payload: PutPayload,
        opts: PutOptions,
    ) -> object_store::Result<PutResult> {
        self.before("PUT", location, payload.content_length() as u64)
            .await;
        self.inner.put_opts(location, payload, opts).await
    }

    async fn put_multipart_opts(
        &self,
        location: &Path,
        opts: PutMultipartOptions,
    ) -> object_store::Result<Box<dyn MultipartUpload>> {
        self.before("MPU", location, 0).await;
        let inner = self.inner.put_multipart_opts(location, opts).await?;
        Ok(Box::new(LatencyUpload {
            inner,
            location: location.clone(),
            store: LatencyHandle {
                model: self.model,
                counters: Arc::clone(&self.counters),
                seq: self.seq.fetch_add(1, Ordering::Relaxed),
            },
        }))
    }

    async fn get_opts(
        &self,
        location: &Path,
        options: GetOptions,
    ) -> object_store::Result<GetResult> {
        let op = if options.head {
            "HEAD"
        } else if options.range.is_some() {
            "GETR"
        } else {
            "GET"
        };
        self.before(op, location, 0).await;
        let result = self.inner.get_opts(location, options).await?;
        let bytes = result.range.end.saturating_sub(result.range.start);
        if op != "HEAD" {
            self.after_download(bytes).await;
        }
        Ok(result)
    }

    async fn get_ranges(
        &self,
        location: &Path,
        ranges: &[Range<u64>],
    ) -> object_store::Result<Vec<Bytes>> {
        // One coalesced request, as the S3 client issues it for nearby ranges.
        self.before("GETR", location, 0).await;
        let out = self.inner.get_ranges(location, ranges).await?;
        let bytes: u64 = out.iter().map(|b| b.len() as u64).sum();
        self.after_download(bytes).await;
        Ok(out)
    }

    fn delete_stream(
        &self,
        locations: BoxStream<'static, object_store::Result<Path>>,
    ) -> BoxStream<'static, object_store::Result<Path>> {
        let counters = Arc::clone(&self.counters);
        let locations = locations
            .map(move |location| {
                if let Ok(path) = &location {
                    counters.record("DELETE", path);
                }
                location
            })
            .boxed();
        self.inner.delete_stream(locations)
    }

    fn list(&self, prefix: Option<&Path>) -> BoxStream<'static, object_store::Result<ObjectMeta>> {
        let root = Path::from("");
        self.counters.record("LIST", prefix.unwrap_or(&root));
        let wait = self.request_latency();
        let inner = self.inner.list(prefix);
        futures::stream::once(tokio::time::sleep(wait).map(|_| inner))
            .flatten()
            .boxed()
    }

    async fn list_with_delimiter(&self, prefix: Option<&Path>) -> object_store::Result<ListResult> {
        let root = Path::from("");
        self.before("LIST", prefix.unwrap_or(&root), 0).await;
        self.inner.list_with_delimiter(prefix).await
    }

    async fn copy_opts(
        &self,
        from: &Path,
        to: &Path,
        options: CopyOptions,
    ) -> object_store::Result<()> {
        self.before("COPY", to, 0).await;
        self.inner.copy_opts(from, to, options).await
    }
}

/// What a multipart upload needs from its store to charge latency.
#[derive(Debug, Clone)]
struct LatencyHandle {
    model: LatencyModel,
    counters: Arc<StoreCounters>,
    seq: u64,
}

impl LatencyHandle {
    fn wait(&mut self, bytes: u64) -> Duration {
        self.seq = self.seq.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let span = self.model.max.saturating_sub(self.model.min);
        let jitter = if span.is_zero() {
            Duration::ZERO
        } else {
            Duration::from_micros((self.seq >> 11) % (span.as_micros() as u64 + 1))
        };
        let transfer = if self.model.bytes_per_sec == 0 {
            Duration::ZERO
        } else {
            Duration::from_secs_f64(bytes as f64 / self.model.bytes_per_sec as f64)
        };
        self.model.min + jitter + transfer
    }
}

#[derive(Debug)]
struct LatencyUpload {
    inner: Box<dyn MultipartUpload>,
    location: Path,
    store: LatencyHandle,
}

#[async_trait]
impl MultipartUpload for LatencyUpload {
    fn put_part(&mut self, data: PutPayload) -> UploadPart {
        let bytes = data.content_length() as u64;
        self.store.counters.record("PART", &self.location);
        self.store
            .counters
            .bytes_in
            .fetch_add(bytes, Ordering::Relaxed);
        let wait = self.store.wait(bytes);
        let part = self.inner.put_part(data);
        Box::pin(async move {
            tokio::time::sleep(wait).await;
            part.await
        })
    }

    async fn complete(&mut self) -> object_store::Result<PutResult> {
        self.store.counters.record("MPUDONE", &self.location);
        let wait = self.store.wait(0);
        tokio::time::sleep(wait).await;
        self.inner.complete().await
    }

    async fn abort(&mut self) -> object_store::Result<()> {
        self.inner.abort().await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use object_store::memory::InMemory;
    use object_store::ObjectStoreExt as _;

    #[tokio::test]
    async fn charges_latency_and_counts_by_class() {
        let counters = Arc::new(StoreCounters::default());
        let store = LatencyStore::new(
            Arc::new(InMemory::new()),
            LatencyModel::new(20, 20, 0),
            Arc::clone(&counters),
        );
        let path = Path::from("ns/wal/0000000000000001.wal");
        let started = Instant::now();
        store
            .put(&path, Bytes::from_static(b"abc").into())
            .await
            .unwrap();
        store.head(&path).await.unwrap();
        assert!(started.elapsed() >= Duration::from_millis(40));
        let snap = counters.snapshot();
        assert_eq!(snap.by_op.get("PUT wal"), Some(&1));
        assert_eq!(snap.by_op.get("HEAD wal"), Some(&1));
        assert_eq!(snap.bytes_written, 3);
    }

    #[tokio::test]
    async fn concurrent_requests_overlap() {
        let store = Arc::new(LatencyStore::new(
            Arc::new(InMemory::new()),
            LatencyModel::new(50, 50, 0),
            Arc::new(StoreCounters::default()),
        ));
        let started = Instant::now();
        let puts = (0..8).map(|i| {
            let store = Arc::clone(&store);
            async move {
                store
                    .put(&Path::from(format!("ns/sst/{i}")), Bytes::new().into())
                    .await
                    .unwrap();
            }
        });
        futures::future::join_all(puts).await;
        assert!(started.elapsed() < Duration::from_millis(300));
    }
}
