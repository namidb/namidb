//! Recovery / cold-open bench.
//!
//! Measures the cost of `WriterSession::open` against a namespace whose
//! manifest references N WAL segments that have **not** been flushed.
//! That's the path a real writer pays on startup after a crash (or any
//! cold process boot) before it can take new writes.
//!
//! The work it covers:
//! - `ManifestStore::load_current` (one GET on the manifest pointer + one
//! on the manifest body).
//! - `claim_writer` (CAS that bumps the manifest epoch; one PUT).
//! - `recover_memtable` (one GET per WAL segment + bincode decode of
//! every record + `Memtable::apply` per record).
//! - `WalStore::list_segments` (one LIST against the WAL prefix to seed
//! `next_wal_seq` past orphans).
//!
//! ## Threshold
//!
//! There is no plan gate on recovery time. We're capturing it as
//! a snapshot so we know how the cold-open path scales with the WAL
//! segment count and the records-per-segment ratio — both knobs the
//! flush cadence policy controls.
//!
//! ## Scaling knobs
//!
//! | Env | Meaning | Default |
//! |---|---|---|
//! | `BENCH_SEGMENTS` | WAL segments pre-created off-clock | `10` |
//! | `BENCH_RECORDS_PER_SEGMENT` | records per segment | `10_000` |
//! | `BENCH_STORE` | `inmemory` or `s3` (same wiring as others) | `inmemory` |
//!
//! Total records replayed per open = `BENCH_SEGMENTS × BENCH_RECORDS_PER_SEGMENT`.
//! With defaults, that's 100 K records — the "writer crashed in the
//! middle of a batch import" case. Higher segment counts model a
//! pathological flush cadence.

use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};

use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use object_store::aws::AmazonS3Builder;
use object_store::memory::InMemory;
use object_store::ObjectStore;
use tokio::runtime::Runtime;
use uuid::Uuid;

use namidb_core::{NamespaceId, NodeId, Value};
use namidb_storage::{NamespacePaths, NodeWriteRecord, WriterSession};

const DEFAULT_SEGMENTS: u64 = 10;
const DEFAULT_RECORDS_PER_SEGMENT: u64 = 10_000;

fn segments_count -> u64 {
 std::env::var("BENCH_SEGMENTS")
.ok
.and_then(|s| s.parse.ok)
.unwrap_or(DEFAULT_SEGMENTS)
}

fn records_per_segment -> u64 {
 std::env::var("BENCH_RECORDS_PER_SEGMENT")
.ok
.and_then(|s| s.parse.ok)
.unwrap_or(DEFAULT_RECORDS_PER_SEGMENT)
}

fn install_rustls_provider {
 let _ = rustls::crypto::aws_lc_rs::default_provider.install_default;
}

fn build_store -> (Arc<dyn ObjectStore>, String) {
 install_rustls_provider;
 let mode = std::env::var("BENCH_STORE").unwrap_or_else(|_| "inmemory".into);
 match mode.as_str {
 "s3" => {
 let bucket = std::env::var("NAMIDB_TEST_BUCKET")
.expect("BENCH_STORE=s3 requires NAMIDB_TEST_BUCKET");
 let s3 = AmazonS3Builder::from_env
.with_bucket_name(&bucket)
.build
.expect("AmazonS3 client must build from env");
 (Arc::new(s3), format!("s3://{bucket}")
 }
 "inmemory" | "" => (Arc::new(InMemory::new), "inmemory://".into),
 other => panic!("unknown BENCH_STORE='{other}' (want 'inmemory' or 's3')"),
 }
}

fn synth_node_id(i: u64) -> NodeId {
 let mut bytes = [0u8; 16];
 bytes[8..].copy_from_slice(&i.to_be_bytes);
 NodeId::from_uuid(Uuid::from_bytes(bytes)
}

fn synth_record(i: u64) -> NodeWriteRecord {
 let mut props = std::collections::BTreeMap::new;
 props.insert("name".into, Value::Str(format!("user-{i}"));
 props.insert("age".into, Value::I64((i % 100) as i64);
 NodeWriteRecord {
 properties: props,
 schema_version: 1,
 }
}

/// Off-clock setup: open a fresh namespace, ingest `segments × per_segment`
/// records calling `commit_batch` between segments, drop the writer
/// without flushing. The store and paths returned point at a namespace
/// whose manifest references `segments` WAL segments waiting to be
/// replayed.
async fn populate_namespace(
 store: Arc<dyn ObjectStore>,
 segments: u64,
 per_segment: u64,
) -> (Arc<dyn ObjectStore>, NamespacePaths, usize) {
 let unique = Uuid::now_v7.simple.to_string;
 let ns = format!("rr-{}", &unique[..16]);
 let paths = NamespacePaths::new("bench", NamespaceId::new(&ns).unwrap);

 let mut writer = WriterSession::open(store.clone, paths.clone)
.await
.expect("WriterSession::open (populate)");

 let mut written = 0usize;
 for s in 0..segments {
 for i in 0..per_segment {
 let global = s * per_segment + i;
 let id = synth_node_id(global);
 let record = synth_record(global);
 writer.upsert_node("Person", id, &record).expect("upsert");
 written += 1;
 }
 writer
.commit_batch
.await
.expect("commit_batch (populate)");
 }
 // Intentionally drop without flushing. The manifest still references
 // every WAL segment we just PUT; a fresh `WriterSession::open` will
 // replay them.
 drop(writer);
 (store, paths, written)
}

fn rt -> Runtime {
 // Multi-thread / 2 workers, consistent with the other storage benches.
 tokio::runtime::Builder::new_multi_thread
.worker_threads(2)
.enable_all
.build
.unwrap
}

fn bench_recovery(c: &mut Criterion) {
 let segments = segments_count;
 let per_segment = records_per_segment;
 let total_records = segments * per_segment;

 let mut group = c.benchmark_group("writer_session/recovery");
 group.sample_size(10);
 group.measurement_time(Duration::from_secs(60);
 group.throughput(Throughput::Elements(total_records);

 // Setup ONCE per *process*, not per bench_function invocation.
 // Criterion calls the bench_function closure multiple times during a
 // single run (warmup, autotune, measurement), and previously each
 // call rebuilt the namespace — invisible on inmemory (~10 ms per
 // populate) but catastrophic against R2 (~30 s per populate, which
 // multiplied the bench's wall time by ~30 ×). A static `OnceLock`
 // memoises the (store, paths, written) tuple so subsequent calls
 // re-use it. Every measured `open` still re-claims the writer
 // (CAS bumps epoch) and re-replays the same set of segments —
 // manifest never flushes them, so the work is deterministic.
 static FIXTURE: OnceLock<(Arc<dyn ObjectStore>, NamespacePaths, usize)> = OnceLock::new;

 group.bench_function("cold_open_with_wal_replay", |b| {
 let runtime = rt;
 let (store, paths, written) = FIXTURE.get_or_init(|| {
 runtime.block_on(async {
 let (store, backend) = build_store;
 let (store, paths, written) =
 populate_namespace(store, segments, per_segment).await;
 eprintln!(
 "[recovery] backend={} ns={} segments={} per_segment={} written={}",
 backend,
 paths.namespace,
 segments,
 per_segment,
 written
 );
 (store, paths, written)
 })
 });
 assert_eq!(*written as u64, total_records);

 b.iter_custom(|iters| {
 let mut total = Duration::ZERO;
 for _ in 0..iters {
 let start = Instant::now;
 let writer = runtime
.block_on(WriterSession::open(store.clone, paths.clone)
.expect("WriterSession::open (measured)");
 total += start.elapsed;
 drop(writer);
 }
 total
 });
 });

 group.finish;
}

criterion_group!(benches, bench_recovery);
criterion_main!(benches);
