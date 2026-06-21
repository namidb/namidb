//! Cold + warm read-latency micro-bench for the snapshot read path.
//!
//! ## Threshold gate
//!
//! - Cold `lookup_node` against an `InMemory` object_store: p50 must
//! stay well under 1 ms. The goal of "cold <500 ms p50" is
//! defined against real S3/LocalStack; the in-process bench is a
//! regression guard.
//! - Warm `lookup_node` with `SstCache` attached: p50 must stay under
//! 10 µs in this harness. Same caveat — the goal of
//! "warm <10 ms p50" is real-network-shaped.
//!
//! ## Scaling knobs
//!
//! Pass `BENCH_NODES=<n>` to scale the synthetic graph; default is
//! 100_000 to keep the bench under ~10 s on a developer laptop. Bump
//! to 10_000_000 when running the gate.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use bytes::Bytes;
use criterion::{criterion_group, criterion_main, BatchSize, Criterion};
use object_store::aws::AmazonS3Builder;
use object_store::memory::InMemory;
use object_store::ObjectStore;
use tokio::runtime::Runtime;
use uuid::Uuid;

use namidb_core::{DataType, LabelDef, NamespaceId, NodeId, PropertyDef, SchemaBuilder, Value};
use namidb_storage::manifest::{LoadedManifest, ManifestStore};
use namidb_storage::{
    flush, MemKey, MemOp, Memtable, NamespacePaths, NodeWriteRecord, Snapshot, SstCache,
    WriterFence,
};

const DEFAULT_NODES: u64 = 100_000;
const DEFAULT_BATCH: u64 = 10_000_000;
const DEFAULT_CACHE_MB: usize = 64;

fn nodes_count() -> u64 {
    std::env::var("BENCH_NODES")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_NODES)
}

/// How many nodes to stuff into a single memtable before freezing + flushing.
/// Lower values cap memtable RAM usage; the trade-off is one L0 SST per
/// batch in the resulting manifest. Sequential key generation keeps each
/// SST's `min_key`/`max_key` window disjoint, so lookups still prune to a
/// single SST via the manifest filter.
fn batch_size() -> u64 {
    std::env::var("BENCH_BATCH")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_BATCH)
}

/// Capacity of the warm-path `SstCache` in MiB. Default is 64 MiB which is
/// fine for tens-of-thousands of nodes; bump it past the size of a single
/// SST body when running production-scale gates (otherwise warm evicts
/// every iteration and devolves into cold).
fn cache_capacity_bytes() -> usize {
    let mb = std::env::var("BENCH_CACHE_MB")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(DEFAULT_CACHE_MB);
    mb * 1024 * 1024
}

/// Build the object store backing this bench run. `BENCH_STORE=s3` switches
/// to an `AmazonS3Builder::from_env()` configured for LocalStack (or any
/// other endpoint pointed at by `AWS_ENDPOINT_URL`). Default is `InMemory`
/// for the regression-guard path.
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
    // Big-endian byte layout keeps lexicographic order monotonic with `i`.
    let mut bytes = [0u8; 16];
    bytes[8..].copy_from_slice(&i.to_be_bytes());
    NodeId::from_uuid(Uuid::from_bytes(bytes))
}

fn synth_payload(i: u64) -> Bytes {
    let mut props = std::collections::BTreeMap::new();
    props.insert("name".into(), Value::Str(format!("user-{i}")));
    props.insert("age".into(), Value::I64((i % 100) as i64));
    NodeWriteRecord {
        properties: props,
        schema_version: 1,
        labels: Vec::new(),
    }
    .encode()
    .unwrap()
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

#[derive(Clone)]
struct Fixture {
    store: Arc<dyn ObjectStore>,
    paths: NamespacePaths,
    committed: LoadedManifest,
    ids: Arc<Vec<NodeId>>,
}

async fn build_fixture(node_count: u64) -> Fixture {
    let (store, backend_label) = build_store();
    let batch = batch_size().max(1);
    // Unique namespace per run so repeated `cargo bench` invocations against
    // a persistent backend (e.g. LocalStack with `PERSISTENCE=1`) don't
    // collide on manifest CAS. For InMemory it's just cosmetic.
    let unique = Uuid::now_v7().simple().to_string();
    let ns_name = format!("rl-{}", &unique[..16]);
    let batches = node_count.div_ceil(batch);
    eprintln!(
        "[bench] backend={} namespace={} nodes={} batch={} batches={}",
        backend_label, ns_name, node_count, batch, batches
    );
    let paths = NamespacePaths::new("bench", NamespaceId::new(&ns_name).unwrap());
    let ms = ManifestStore::new(store.clone(), paths.clone());
    let base = ms.bootstrap(Uuid::now_v7()).await.unwrap();
    let fence = WriterFence::new(base.manifest.epoch);
    let schema = SchemaBuilder::new().label(person_label()).unwrap().build();

    let mut ids = Vec::with_capacity(node_count as usize);
    let mut current = base;
    for b in 0..batches {
        let lo = b * batch;
        let hi = node_count.min(lo + batch);
        let mut mt = Memtable::new();
        for i in lo..hi {
            let id = synth_node_id(i);
            ids.push(id);
            mt.apply(MemKey::Node { id }, i + 1, MemOp::Upsert(synth_payload(i)));
        }
        let frozen = mt.freeze();
        let outcome = flush(&ms, &fence, &current, &frozen, schema.clone())
            .await
            .expect("flush");
        current = outcome.committed;
        eprintln!(
            "[bench] flushed batch {}/{} (rows {}..{})",
            b + 1,
            batches,
            lo,
            hi
        );
    }

    Fixture {
        store,
        paths,
        committed: current,
        ids: Arc::new(ids),
    }
}

fn rt() -> Runtime {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap()
}

fn lookup_cold(c: &mut Criterion) {
    let runtime = rt();
    let count = nodes_count();
    let fx = runtime.block_on(build_fixture(count));
    let ids_len = fx.ids.len();
    let warm_window = ids_len.min(64);

    let mut group = c.benchmark_group("snapshot/lookup_node");
    group.measurement_time(Duration::from_secs(8));
    group.sample_size(50);

    // ── Cold: no cache ─────────────────────────────────────────────────
    let cold_cursor = Arc::new(AtomicUsize::new(0));
    {
        let fx = fx.clone();
        let cold_cursor = cold_cursor.clone();
        group.bench_function("cold_no_cache", |b| {
            b.to_async(&runtime).iter_batched(
                || {
                    let idx = cold_cursor.fetch_add(1, Ordering::Relaxed) % ids_len;
                    fx.ids[idx]
                },
                |id| {
                    let fx = fx.clone();
                    async move {
                        let memtable = Memtable::new().snapshot_view();
                        let snap = Snapshot::new(
                            fx.committed.clone(),
                            &memtable,
                            fx.store.clone(),
                            fx.paths.clone(),
                        );
                        let _ = snap.lookup_node("Person", id).await.expect("lookup");
                    }
                },
                BatchSize::SmallInput,
            );
        });
    }

    // ── Cold ranged (RFC-003): footer + page-index + column-page GETs ─
    // Skips the full-body GET. First call on each SST incurs the
    // metadata round-trip; subsequent calls hit the metadata cache
    // (added in the same close-out as RFC-003 §"Cache integration").
    let ranged_cursor = Arc::new(AtomicUsize::new(0));
    let ranged_cache = SstCache::new(cache_capacity_bytes());
    {
        let fx = fx.clone();
        let ranged_cursor = ranged_cursor.clone();
        let ranged_cache = ranged_cache.clone();
        group.bench_function("cold_ranged_reads", |b| {
            b.to_async(&runtime).iter_batched(
                || {
                    let idx = ranged_cursor.fetch_add(1, Ordering::Relaxed) % ids_len;
                    fx.ids[idx]
                },
                |id| {
                    let fx = fx.clone();
                    let cache = ranged_cache.clone();
                    async move {
                        let memtable = Memtable::new().snapshot_view();
                        let snap = Snapshot::new(
                            fx.committed.clone(),
                            &memtable,
                            fx.store.clone(),
                            fx.paths.clone(),
                        )
                        .with_cache(cache)
                        .with_ranged_reads(true);
                        let _ = snap.lookup_node("Person", id).await.expect("lookup");
                    }
                },
                BatchSize::SmallInput,
            );
        });
    }

    // ── Warm: SstCache attached, pre-warmed window ────────────────────
    let warm_cache = SstCache::new(cache_capacity_bytes());
    {
        let memtable = Memtable::new().snapshot_view();
        let snap = Snapshot::new(
            fx.committed.clone(),
            &memtable,
            fx.store.clone(),
            fx.paths.clone(),
        )
        .with_cache(warm_cache.clone());
        runtime.block_on(async {
            for id in &fx.ids[..warm_window] {
                let _ = snap.lookup_node("Person", *id).await.expect("warm");
            }
        });
    }
    let warm_cursor = Arc::new(AtomicUsize::new(0));
    {
        let fx = fx.clone();
        let warm_cursor = warm_cursor.clone();
        let warm_cache = warm_cache.clone();
        group.bench_function("warm_with_cache", |b| {
            b.to_async(&runtime).iter_batched(
                || {
                    let idx = warm_cursor.fetch_add(1, Ordering::Relaxed) % warm_window;
                    fx.ids[idx]
                },
                |id| {
                    let fx = fx.clone();
                    let cache = warm_cache.clone();
                    async move {
                        let memtable = Memtable::new().snapshot_view();
                        let snap = Snapshot::new(
                            fx.committed.clone(),
                            &memtable,
                            fx.store.clone(),
                            fx.paths.clone(),
                        )
                        .with_cache(cache);
                        let _ = snap.lookup_node("Person", id).await.expect("warm");
                    }
                },
                BatchSize::SmallInput,
            );
        });
    }

    group.finish();
}

criterion_group!(benches, lookup_cold);
criterion_main!(benches);
