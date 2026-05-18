//! Read/write concurrent mix bench.
//!
//! Drives the namespace with **two simultaneous tasks** for `BENCH_DURATION_SECS`:
//! - A **writer** task owns one `WriterSession` and runs
//! `upsert_node` + `commit_batch` in a tight loop, pushing new ids
//! beyond the pre-flushed fixture range.
//! - A **reader** task opens its own `ManifestStore` (no shared state
//! with the writer), reloads the manifest every N reads, and runs
//! `lookup_node` on uniformly-random ids inside the fixture range.
//!
//! Both tasks count their own ops; the reader also records per-call
//! latency into a vec the runner drains at stop. The single sample
//! reports four numbers: writer ops/s, reader ops/s, reader p50, reader p99.
//!
//! ## Why a *separate* `ManifestStore` for the reader
//!
//! `WriterSession` borrows its own `Memtable` (`Snapshot<'mt>` carries
//! `&'mt Memtable`), so giving the reader a snapshot off the live
//! session would pin a borrow against the writer's `&mut self`. The
//! realistic production shape is "reader and writer are separate
//! processes connected only by the object store": that's what we
//! mirror here by going through `ManifestStore::load_current` and
//! building `Snapshot::new(loaded, &empty_memtable, store, paths)`.
//! Cost is: reader sees only **flushed** state (SSTs published by the
//! writer's commits), never pending memtable rows.
//!
//! ## What we don't measure
//!
//! - Multi-reader, multi-writer mix. One of each.
//! - Concurrent compaction in flight. The writer here only does
//! `commit_batch`; no mid-run flush, no L0→L1 compaction. Bench
//! isolates the manifest hot path under simultaneous read.
//!
//! ## Scaling knobs
//!
//! | Env | Meaning | Default |
//! |---|---|---|
//! | `BENCH_FIXTURE_NODES` | nodes pre-flushed before mix begins | `100_000` |
//! | `BENCH_DURATION_SECS` | concurrent run wall-clock | `15` |
//! | `BENCH_BATCH_COMMIT` | nodes per writer `commit_batch` | `10_000` |
//! | `BENCH_READER_REFRESH` | reads between manifest reloads | `100` |
//! | `BENCH_CACHE_MB` | warm-path `SstCache` capacity (MiB) | `64` |
//! | `BENCH_STORE` | `inmemory` or `s3` (same env block as others) | `inmemory` |

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use object_store::aws::AmazonS3Builder;
use object_store::memory::InMemory;
use object_store::ObjectStore;
use tokio::runtime::Runtime;
use uuid::Uuid;

use namidb_core::{
 DataType as TgDataType, LabelDef, NamespaceId, NodeId, PropertyDef, SchemaBuilder, Value,
};
use namidb_storage::{
 ManifestStore, Memtable, NamespacePaths, NodeWriteRecord, Snapshot, SstCache, WriterSession,
};

const DEFAULT_FIXTURE_NODES: u64 = 100_000;
const DEFAULT_DURATION_SECS: u64 = 15;
const DEFAULT_BATCH_COMMIT: u64 = 10_000;
const DEFAULT_READER_REFRESH: u64 = 100;
const DEFAULT_CACHE_MB: usize = 64;

fn fixture_nodes() -> u64 {
 std::env::var("BENCH_FIXTURE_NODES")
 .ok()
 .and_then(|s| s.parse().ok())
 .unwrap_or(DEFAULT_FIXTURE_NODES)
}
fn duration_secs() -> u64 {
 std::env::var("BENCH_DURATION_SECS")
 .ok()
 .and_then(|s| s.parse().ok())
 .unwrap_or(DEFAULT_DURATION_SECS)
}
fn batch_commit_size() -> u64 {
 std::env::var("BENCH_BATCH_COMMIT")
 .ok()
 .and_then(|s| s.parse().ok())
 .unwrap_or(DEFAULT_BATCH_COMMIT)
}
fn reader_refresh_every() -> u64 {
 std::env::var("BENCH_READER_REFRESH")
 .ok()
 .and_then(|s| s.parse().ok())
 .unwrap_or(DEFAULT_READER_REFRESH)
}
fn cache_mb() -> usize {
 std::env::var("BENCH_CACHE_MB")
 .ok()
 .and_then(|s| s.parse().ok())
 .unwrap_or(DEFAULT_CACHE_MB)
}

fn install_rustls_provider() {
 let _ = rustls::crypto::aws_lc_rs::default_provider().install_default();
}

fn build_store() -> (Arc<dyn ObjectStore>, String) {
 install_rustls_provider();
 let mode = std::env::var("BENCH_STORE").unwrap_or_else(|_| "inmemory".into());
 match mode.as_str() {
 "s3" => {
 let bucket = std::env::var("NAMIDB_TEST_BUCKET")
 .expect("BENCH_STORE=s3 requires NAMIDB_TEST_BUCKET");
 let s3 = AmazonS3Builder::from_env()
 .with_bucket_name(&bucket)
 .build()
 .expect("AmazonS3 client must build from env");
 (Arc::new(s3), format!("s3://{bucket}"))
 }
 "inmemory" | "" => (Arc::new(InMemory::new()), "inmemory://".into()),
 other => panic!("unknown BENCH_STORE='{other}' (want 'inmemory' or 's3')"),
 }
}

fn synth_node_id(i: u64) -> NodeId {
 let mut bytes = [0u8; 16];
 bytes[8..].copy_from_slice(&i.to_be_bytes());
 NodeId::from_uuid(Uuid::from_bytes(bytes))
}

fn synth_record(i: u64) -> NodeWriteRecord {
 let mut props = std::collections::BTreeMap::new();
 props.insert("name".into(), Value::Str(format!("user-{i}")));
 props.insert("age".into(), Value::I64((i % 100) as i64));
 NodeWriteRecord {
 properties: props,
 schema_version: 1,
 }
}

fn person_label() -> LabelDef {
 LabelDef {
 name: "Person".into(),
 properties: vec![
 PropertyDef::new("name", TgDataType::Utf8, false).unwrap(),
 PropertyDef::new("age", TgDataType::Int32, true).unwrap(),
 ],
 }
}

/// Cheap deterministic PRNG (xorshift64). Avoids pulling `rand` into
/// the workspace just for this bench.
#[derive(Clone, Copy)]
struct Xorshift64(u64);

impl Xorshift64 {
 fn new(seed: u64) -> Self {
 Self(seed.max(1))
 }
 fn next_u64(&mut self) -> u64 {
 let mut x = self.0;
 x ^= x << 13;
 x ^= x >> 7;
 x ^= x << 17;
 self.0 = x;
 x
 }
}

/// Pre-flush `fixture` nodes into a fresh namespace so the reader has
/// real SST candidates to lookup. Returns the namespace handle.
async fn build_fixture(
 store: Arc<dyn ObjectStore>,
 fixture: u64,
 commit_every: u64,
) -> NamespacePaths {
 let unique = Uuid::now_v7().simple().to_string();
 let ns = format!("mx-{}", &unique[..16]);
 let paths = NamespacePaths::new("bench", NamespaceId::new(&ns).unwrap());
 let schema = SchemaBuilder::new().label(person_label()).unwrap().build();

 let mut writer = WriterSession::open(store.clone(), paths.clone())
 .await
 .expect("WriterSession::open (fixture)");

 let commit_every = commit_every.max(1);
 for i in 0..fixture {
 writer
 .upsert_node("Person", synth_node_id(i), &synth_record(i))
 .expect("upsert (fixture)");
 if (i + 1) % commit_every == 0 {
 writer.commit_batch().await.expect("commit_batch (fixture)");
 }
 }
 if writer.pending_len() > 0 {
 writer
 .commit_batch()
 .await
 .expect("commit_batch tail (fixture)");
 }
 let _ = writer.flush(schema).await.expect("flush (fixture)");
 drop(writer);
 paths
}

#[derive(Debug)]
struct MixOutcome {
 duration: Duration,
 writer_ops: u64,
 reader_ops: u64,
 reader_p50: Duration,
 reader_p99: Duration,
 reader_p999: Duration,
 reader_max: Duration,
 cache_hits: u64,
 cache_misses: u64,
}

async fn run_concurrent_mix(
 store: Arc<dyn ObjectStore>,
 paths: NamespacePaths,
 fixture: u64,
 duration: Duration,
 commit_every: u64,
 refresh_every: u64,
 cache_bytes: usize,
) -> MixOutcome {
 let stop = Arc::new(AtomicBool::new(false));
 let writer_ops = Arc::new(AtomicU64::new(0));
 let reader_ops = Arc::new(AtomicU64::new(0));
 let reader_latencies: Arc<Mutex<Vec<Duration>>> =
 Arc::new(Mutex::new(Vec::with_capacity(1_000_000)));
 let cache = SstCache::new(cache_bytes);

 // ── Writer task ──
 let writer_handle = {
 let store = store.clone();
 let paths = paths.clone();
 let stop = stop.clone();
 let writer_ops = writer_ops.clone();
 tokio::spawn(async move {
 let mut writer = WriterSession::open(store, paths)
 .await
 .expect("WriterSession::open (writer)");
 // Start ids strictly above the fixture so we never clash
 // with the reader's lookups.
 let mut next_id = fixture;
 let commit_every = commit_every.max(1);
 while !stop.load(Ordering::Acquire) {
 let target = next_id + commit_every;
 while next_id < target {
 writer
 .upsert_node("Person", synth_node_id(next_id), &synth_record(next_id))
 .expect("upsert (writer)");
 next_id += 1;
 }
 writer.commit_batch().await.expect("commit_batch (writer)");
 writer_ops.fetch_add(commit_every, Ordering::Relaxed);
 // Yield so the reader gets scheduled even on a busy
 // current-thread runtime.
 tokio::task::yield_now().await;
 }
 })
 };

 // ── Reader task ──
 let reader_handle = {
 let store = store.clone();
 let paths = paths.clone();
 let stop = stop.clone();
 let reader_ops = reader_ops.clone();
 let reader_latencies = reader_latencies.clone();
 let cache = cache.clone();
 tokio::spawn(async move {
 let manifest_store = ManifestStore::new(store.clone(), paths.clone());
 let mut current = manifest_store
 .load_current()
 .await
 .expect("ManifestStore::load_current (reader init)");
 let empty = Memtable::new();
 let mut rng = Xorshift64::new(0xC0FFEE_u64);
 let mut count: u64 = 0;

 while !stop.load(Ordering::Acquire) {
 if count > 0 && count % refresh_every == 0 {
 current = manifest_store
 .load_current()
 .await
 .expect("ManifestStore::load_current (reader refresh)");
 }
 let snap = Snapshot::new(current.clone(), &empty, store.clone(), paths.clone())
 .with_cache(cache.clone());

 let pick = rng.next_u64() % fixture;
 let id = synth_node_id(pick);
 let started = Instant::now();
 let _ = snap
 .lookup_node("Person", id)
 .await
 .expect("lookup_node (reader)");
 let elapsed = started.elapsed();

 reader_latencies.lock().unwrap().push(elapsed);
 reader_ops.fetch_add(1, Ordering::Relaxed);
 count += 1;
 }
 })
 };

 tokio::time::sleep(duration).await;
 stop.store(true, Ordering::Release);

 writer_handle.await.expect("writer task panic");
 reader_handle.await.expect("reader task panic");

 let writer_ops = writer_ops.load(Ordering::Relaxed);
 let reader_ops = reader_ops.load(Ordering::Relaxed);
 let mut latencies = reader_latencies.lock().unwrap().clone();
 latencies.sort();
 let n = latencies.len().max(1);
 let p50 = latencies[(n - 1) / 2];
 let p99 = latencies[(n * 99 / 100).min(n - 1)];
 let p999 = latencies[(n * 999 / 1000).min(n - 1)];
 let max = *latencies.last().unwrap_or(&Duration::ZERO);

 // Pluck cache stats from the Debug impl (the public surface).
 let dbg = format!("{:?}", cache);
 let cache_hits = parse_debug_field(&dbg, "hits").unwrap_or(0);
 let cache_misses = parse_debug_field(&dbg, "misses").unwrap_or(0);

 MixOutcome {
 duration,
 writer_ops,
 reader_ops,
 reader_p50: p50,
 reader_p99: p99,
 reader_p999: p999,
 reader_max: max,
 cache_hits,
 cache_misses,
 }
}

fn parse_debug_field(dbg: &str, field: &str) -> Option<u64> {
 let needle = format!("{field}: ");
 let rest = dbg.split(&needle).nth(1)?;
 let tok: String = rest.chars().take_while(|c| c.is_ascii_digit()).collect();
 tok.parse().ok()
}

fn rt() -> Runtime {
 // Multi-thread so the writer's I/O .awaits don't starve the reader
 // and vice versa. The workload is dominated by store round-trips,
 // not CPU.
 tokio::runtime::Builder::new_multi_thread()
 .worker_threads(2)
 .enable_all()
 .build()
 .unwrap()
}

fn bench_concurrent_mix(c: &mut Criterion) {
 let fixture = fixture_nodes();
 let duration = Duration::from_secs(duration_secs());
 let commit_every = batch_commit_size();
 let refresh_every = reader_refresh_every();
 let cache_bytes = cache_mb() * 1024 * 1024;

 let mut group = c.benchmark_group("writer_session/concurrent_mix");
 group.sample_size(10);
 // Criterion's measurement_time is a hint; iter_custom controls the
 // actual wall-time. Each sample is one `duration`-long run.
 group.measurement_time(duration.saturating_mul(11));
 group.throughput(Throughput::Elements(1));

 // NB: we intentionally do NOT memoise the fixture here. Each
 // criterion sample rebuilds the namespace because the writer task
 // in `run_concurrent_mix` mutates the manifest (every commit_batch
 // adds a new WAL segment); reusing the namespace across samples
 // would mean sample N opens against a manifest with N − 1
 // samples' worth of pending WAL segments to replay, making
 // per-sample writer throughput decay with sample index. That
 // would invalidate the "sustained writer ops/s" number. Bench
 // does pay the populate cost per sample — acceptable on inmemory
 // / LocalStack, expensive on real WAN; see the R2 doc.
 group.bench_function("writer_plus_reader", |b| {
 b.iter_custom(|iters| {
 let runtime = rt();
 let mut acc = Duration::ZERO;
 for _ in 0..iters {
 let elapsed = runtime.block_on(async {
 let (store, backend) = build_store();
 let paths =
 build_fixture(store.clone(), fixture, commit_every).await;
 eprintln!(
 "[mix] backend={} ns={} fixture={} duration={}s commit_every={} refresh_every={} cache={}MiB",
 backend,
 paths.namespace(),
 fixture,
 duration.as_secs(),
 commit_every,
 refresh_every,
 cache_bytes / (1024 * 1024)
 );
 let started = Instant::now();
 let outcome = run_concurrent_mix(
 store,
 paths,
 fixture,
 duration,
 commit_every,
 refresh_every,
 cache_bytes,
 )
 .await;
 let elapsed = started.elapsed();
 eprintln!(
 "[mix] writer_ops={} ({:.0}/s) reader_ops={} ({:.0}/s) p50={:?} p99={:?} p999={:?} max={:?} cache={}h/{}m",
 outcome.writer_ops,
 outcome.writer_ops as f64 / outcome.duration.as_secs_f64(),
 outcome.reader_ops,
 outcome.reader_ops as f64 / outcome.duration.as_secs_f64(),
 outcome.reader_p50,
 outcome.reader_p99,
 outcome.reader_p999,
 outcome.reader_max,
 outcome.cache_hits,
 outcome.cache_misses,
 );
 elapsed
 });
 acc += elapsed;
 }
 acc
 });
 });

 group.finish();
}

criterion_group!(benches, bench_concurrent_mix);
criterion_main!(benches);
