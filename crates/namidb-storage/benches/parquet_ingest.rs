//! Parquet → WriterSession throughput.
//!
//! This is the literal "ingest from Parquet" threshold the
//! `ingest_throughput` bench could only approximate (its records were
//! built in-process; no Arrow batches, no Parquet decode). Here we
//! generate a Parquet file off-clock and measure the full read +
//! decode + Arrow → `Value` conversion + `WriterSession::upsert_node`
//! + `commit_batch` + `flush` chain.
//!
//! ## Scaling knobs
//!
//! | Env | Meaning | Default |
//! |---|---|---|
//! | `BENCH_NODES` | rows in the generated Parquet (per sample) | `1_000_000` |
//! | `BENCH_BATCH_COMMIT` | rows per `commit_batch` | `10_000` |
//! | `BENCH_STORE` | `inmemory` or `s3` (same wiring as others) | `inmemory` |
//!
//! The Parquet file is generated **once** per `Criterion` benchmark
//! function and re-used across iterations. Generation is sync and
//! lives off the clock; only the load is measured.
//!
//! ## What we measure
//!
//! `ParquetLoader::load_nodes` covers:
//! - `parquet-rs` row-group decode (Arrow `RecordBatch`).
//! - Per-row column dispatch (Arrow → `Value::*`).
//! - `BTreeMap::insert` for each property.
//! - `WriterSession::upsert_node` (LSN alloc + pending WAL append).
//! - `WriterSession::commit_batch` (WAL PUT + manifest CAS + memtable
//! apply) every `BENCH_BATCH_COMMIT` rows.
//!
//! Plus we run a final `commit_batch` for the tail and a `flush`
//! (build SST + commit) at the end, so the wall-time number covers
//! everything a real bulk-load workflow does.

use std::fs::File;
use std::sync::Arc;
use std::time::{Duration, Instant};

use arrow_array::{FixedSizeBinaryArray, Int32Array, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema as ArrowSchema};
use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use object_store::aws::AmazonS3Builder;
use object_store::memory::InMemory;
use object_store::ObjectStore;
use parquet::arrow::ArrowWriter;
use tempfile::NamedTempFile;
use tokio::runtime::Runtime;
use uuid::Uuid;

use namidb_core::{DataType as TgDataType, LabelDef, NamespaceId, PropertyDef, SchemaBuilder};
use namidb_storage::{load_nodes_from_parquet, NamespacePaths, WriterSession};

const DEFAULT_NODES: u64 = 1_000_000;
const DEFAULT_BATCH_COMMIT: u64 = 10_000;
/// Rows per Arrow `RecordBatch` written into the synthetic Parquet
/// file. The reader will decode in batches of this size when its own
/// batch_size is larger, so this controls row-group layout indirectly.
const PARQUET_WRITE_BATCH: usize = 65_536;

fn nodes_count -> u64 {
 std::env::var("BENCH_NODES")
.ok
.and_then(|s| s.parse.ok)
.unwrap_or(DEFAULT_NODES)
}

fn batch_commit_size -> u64 {
 std::env::var("BENCH_BATCH_COMMIT")
.ok
.and_then(|s| s.parse.ok)
.unwrap_or(DEFAULT_BATCH_COMMIT)
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

fn synth_id_bytes(i: u64) -> [u8; 16] {
 let mut bytes = [0u8; 16];
 bytes[8..].copy_from_slice(&i.to_be_bytes);
 bytes
}

fn person_label -> LabelDef {
 LabelDef {
 name: "Person".into,
 properties: vec![
 PropertyDef::new("name", TgDataType::Utf8, false).unwrap,
 PropertyDef::new("age", TgDataType::Int32, true).unwrap,
 ],
 }
}

/// Write a synthetic Parquet file with `n_rows` rows of (node_id, name,
/// age). Done once per bench function, off the clock.
fn generate_parquet(path: &std::path::Path, n_rows: u64) {
 let schema = Arc::new(ArrowSchema::new(vec![
 Field::new("node_id", DataType::FixedSizeBinary(16), false),
 Field::new("name", DataType::Utf8, false),
 Field::new("age", DataType::Int32, true),
 ]);

 let file = File::create(path).expect("create parquet file");
 let mut writer =
 ArrowWriter::try_new(file, schema.clone, None).expect("ArrowWriter::try_new");

 let mut written: u64 = 0;
 while written < n_rows {
 let chunk = (n_rows - written).min(PARQUET_WRITE_BATCH as u64) as usize;
 let ids: Vec<[u8; 16]> = (written..written + chunk as u64)
.map(synth_id_bytes)
.collect;
 let id_array =
 FixedSizeBinaryArray::try_from_iter(ids.iter.map(|b| b.as_slice).unwrap;
 let names: Vec<String> = (written..written + chunk as u64)
.map(|i| format!("user-{i}")
.collect;
 let name_array = StringArray::from(names);
 let ages: Vec<i32> = (written..written + chunk as u64)
.map(|i| (i % 100) as i32)
.collect;
 let age_array = Int32Array::from(ages);

 let batch = RecordBatch::try_new(
 schema.clone,
 vec![
 Arc::new(id_array),
 Arc::new(name_array),
 Arc::new(age_array),
 ],
 )
.unwrap;
 writer.write(&batch).unwrap;
 written += chunk as u64;
 }
 writer.close.expect("ArrowWriter::close");
}

async fn load_once(parquet_path: &std::path::Path, commit_every: u64) -> Duration {
 let (store, backend) = build_store;
 let unique = Uuid::now_v7.simple.to_string;
 let ns = format!("pq-{}", &unique[..16]);
 eprintln!(
 "[parquet] backend={} ns={} commit_every={}",
 backend, ns, commit_every
 );
 let paths = NamespacePaths::new("bench", NamespaceId::new(&ns).unwrap);
 let schema = SchemaBuilder::new.label(person_label).unwrap.build;

 let mut writer = WriterSession::open(store, paths)
.await
.expect("WriterSession::open");

 let start = Instant::now;
 let _outcome =
 load_nodes_from_parquet(parquet_path, &mut writer, "Person", commit_every as usize)
.await
.expect("load_nodes_from_parquet");
 if writer.pending_len > 0 {
 writer.commit_batch.await.expect("commit_batch tail");
 }
 let _flush_outcome = writer.flush(schema).await.expect("flush");
 start.elapsed
}

fn rt -> Runtime {
 // Multi-thread / 2 workers — same as `concurrent_mix.rs` and
 // `ingest_throughput.rs` so numbers are directly
 // comparable across the three benches.
 tokio::runtime::Builder::new_multi_thread
.worker_threads(2)
.enable_all
.build
.unwrap
}

fn bench_parquet_ingest(c: &mut Criterion) {
 let total = nodes_count;
 let commit_every = batch_commit_size;

 // Generate the source Parquet once. Criterion will re-use it across
 // every sample iteration. Generation does NOT count toward the
 // measured time.
 let parquet_file = NamedTempFile::new.expect("NamedTempFile");
 eprintln!(
 "[parquet] generating fixture: {} rows -> {}",
 total,
 parquet_file.path.display
 );
 let gen_start = Instant::now;
 generate_parquet(parquet_file.path, total);
 let parquet_bytes = std::fs::metadata(parquet_file.path)
.map(|m| m.len)
.unwrap_or(0);
 eprintln!(
 "[parquet] fixture ready: {:.2} MiB in {:?}",
 parquet_bytes as f64 / 1_048_576.0,
 gen_start.elapsed
 );

 let mut group = c.benchmark_group("writer_session/parquet_ingest");
 group.sample_size(10);
 group.measurement_time(Duration::from_secs(120);
 group.throughput(Throughput::Elements(total);

 group.bench_function("load_commit_flush", |b| {
 b.iter_custom(|iters| {
 let runtime = rt;
 let mut acc = Duration::ZERO;
 for _ in 0..iters {
 acc += runtime.block_on(load_once(parquet_file.path, commit_every);
 }
 acc
 });
 });

 group.finish;
}

criterion_group!(benches, bench_parquet_ingest);
criterion_main!(benches);
