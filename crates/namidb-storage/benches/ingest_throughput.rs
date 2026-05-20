//! Ingest-path throughput bench.
//!
//! Measures end-to-end **nodes/s through the public `WriterSession` API**:
//! `upsert_node` + periodic `commit_batch` + final `flush`. This is the
//! surface a real client touches — it covers WAL PUT, manifest CAS,
//! memtable apply, and the SST build that the read-latency bench
//! deliberately bypasses (the read bench builds its fixture by punching
//! straight into `Memtable::apply`).
//!
//! ## Threshold (plan)
//!
//! `>= 10_000 nodes/s` sustained, from Parquet. We don't load from
//! Parquet here — the synthetic generator stands in. Numbers above the
//! threshold under the synthetic load are a necessary-but-not-sufficient
//! signal; a real Parquet loader will add decode cost on top.
//!
//! ## Scaling knobs
//!
//! | Env | Meaning | Default |
//! |---|---|---|
//! | `BENCH_NODES` | total nodes ingested per sample | `1_000_000` |
//! | `BENCH_BATCH_COMMIT` | nodes per `commit_batch` (durability flush)| `10_000` |
//! | `BENCH_STORE` | `inmemory` or `s3` (same wiring as read) | `inmemory` |
//!
//! For `BENCH_STORE=s3` see the env block in `benches/read_latency.rs`.
//!
//! ## What we *don't* measure here
//!
//! - Recovery (cold-open via WAL replay). Lives in `recovery.rs` tests.
//! - Read concurrency under ingest. The bench is single-writer, no
//! reads. Real workloads stress the manifest hot path under mix; that
//! needs a separate harness.
//! - Compaction overhead. We flush a single SST at the end.

use std::sync::Arc;
use std::time::{Duration, Instant};

use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use object_store::aws::AmazonS3Builder;
use object_store::memory::InMemory;
use object_store::ObjectStore;
use tokio::runtime::Runtime;
use uuid::Uuid;

use namidb_core::{DataType, LabelDef, NamespaceId, NodeId, PropertyDef, SchemaBuilder, Value};
use namidb_storage::{NamespacePaths, NodeWriteRecord, WriterSession};

const DEFAULT_NODES: u64 = 1_000_000;
const DEFAULT_BATCH_COMMIT: u64 = 10_000;

fn nodes_count() -> u64 {
    std::env::var("BENCH_NODES")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_NODES)
}

fn batch_commit_size() -> u64 {
    std::env::var("BENCH_BATCH_COMMIT")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_BATCH_COMMIT)
}

/// Install `aws-lc-rs` as the rustls crypto provider. Idempotent —
/// `install_default()` returns `Err` on subsequent calls which we
/// intentionally ignore. Required when targeting Cloudflare R2; harmless
/// for LocalStack / inmemory.
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
            PropertyDef::new("name", DataType::Utf8, false).unwrap(),
            PropertyDef::new("age", DataType::Int32, true).unwrap(),
        ],
    }
}

/// Run one full ingest cycle and return the elapsed wall-clock time.
async fn ingest_once(node_count: u64, commit_every: u64) -> Duration {
    let (store, backend) = build_store();
    let unique = Uuid::now_v7().simple().to_string();
    let ns = format!("it-{}", &unique[..16]);
    eprintln!(
        "[ingest] backend={} ns={} nodes={} commit_every={}",
        backend, ns, node_count, commit_every
    );
    let paths = NamespacePaths::new("bench", NamespaceId::new(&ns).unwrap());
    let schema = SchemaBuilder::new().label(person_label()).unwrap().build();

    let mut writer = WriterSession::open(store, paths)
        .await
        .expect("WriterSession::open");

    let start = Instant::now();
    let commit_every = commit_every.max(1);
    for i in 0..node_count {
        let id = synth_node_id(i);
        let record = synth_record(i);
        writer.upsert_node("Person", id, &record).expect("upsert");
        if (i + 1) % commit_every == 0 {
            writer.commit_batch().await.expect("commit_batch");
        }
    }
    if writer.pending_len() > 0 {
        writer.commit_batch().await.expect("commit_batch tail");
    }
    let _outcome = writer.flush(schema).await.expect("flush");
    start.elapsed()
}

fn rt() -> Runtime {
    // Multi-thread runtime with 2 workers so this bench is directly
    // comparable to `concurrent_mix.rs`. The single-threaded current-
    // thread runtime previously used here was ~3 × slower under the
    // same workload, which made the "isolated vs concurrent" comparison
    // misleading.
    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .unwrap()
}

fn bench_ingest(c: &mut Criterion) {
    let total = nodes_count();
    let commit_every = batch_commit_size();

    let mut group = c.benchmark_group("writer_session/ingest");
    // Throughput tests want few-but-long samples, not 50 short ones.
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(120));
    group.throughput(Throughput::Elements(total));

    group.bench_function("upsert_commit_flush", |b| {
        b.iter_custom(|iters| {
            let runtime = rt();
            let mut acc = Duration::ZERO;
            for _ in 0..iters {
                acc += runtime.block_on(ingest_once(total, commit_every));
            }
            acc
        });
    });

    group.finish();
}

criterion_group!(benches, bench_ingest);
criterion_main!(benches);
