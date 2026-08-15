//! Search under concurrent maintenance — the steady-state operating mode of a
//! large deployment: readers hold committed manifests (the S3 reader-node
//! topology) while a writer upserts, deletes, flushes, compacts and runs the
//! janitor. The invariants are snapshot-exactness — every answer must match an
//! oracle computed from the same committed manifest, never a torn or stale
//! mixture — and sweep safety: a janitor honoring the readers' floor never
//! breaks a snapshot at or above it.
#![cfg(all(feature = "vector-index", feature = "text-index"))]

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;
use std::time::Duration;

use namidb_core::id::{NamespaceId, NodeId};
use namidb_core::schema::{DataType, LabelDef, PropertyDef, Schema, SchemaBuilder};
use namidb_core::value::Value as CoreValue;
use namidb_storage::manifest::{TextIndexDescriptor, VectorIndexDescriptor};
use namidb_storage::manifest::{VectorMetric, VectorQuantization};
use namidb_storage::memtable::Memtable;
use namidb_storage::read::Snapshot;
use namidb_storage::{sweep_orphans, NamespacePaths, NodeWriteRecord, WriterSession};
use object_store::memory::InMemory;
use object_store::ObjectStore;

const DIM: u32 = 8;
const K: usize = 3;

fn schema() -> Schema {
    SchemaBuilder::new()
        .label(LabelDef {
            name: "Doc".into(),
            properties: vec![
                PropertyDef::new("embedding", DataType::FloatVector { dim: DIM }, true).unwrap(),
                PropertyDef::new("body", DataType::Utf8, true).unwrap(),
                PropertyDef::new("ordinal", DataType::Int64, true).unwrap(),
            ],
        })
        .unwrap()
        .build()
}

/// Deterministic, well-separated embedding for document `ordinal` at content
/// `version`: similarity to the fixed probe strictly decreases with a mixed
/// function of both, so rank margins stay wide enough for an exact oracle.
fn embedding(ordinal: u64, version: u64) -> Vec<f32> {
    let mut v = vec![0.0f32; DIM as usize];
    let phase = (ordinal * 37 + version * 101) % 997;
    v[0] = 1.0;
    v[1] = (phase as f32) / 997.0;
    v[2] = ((ordinal % 13) as f32) / 13.0;
    v[3] = ((version % 7) as f32) / 7.0;
    v
}

fn body(ordinal: u64, version: u64) -> String {
    if (ordinal + version) % 3 == 0 {
        format!("alpha content {ordinal} v{version}")
    } else {
        format!("plain content {ordinal} v{version}")
    }
}

fn record(ordinal: u64, version: u64) -> NodeWriteRecord {
    let mut properties = BTreeMap::new();
    properties.insert(
        "embedding".into(),
        CoreValue::Vec(embedding(ordinal, version)),
    );
    properties.insert("body".into(), CoreValue::Str(body(ordinal, version)));
    properties.insert("ordinal".into(), CoreValue::I64(ordinal as i64));
    NodeWriteRecord {
        properties,
        schema_version: 1,
        ..Default::default()
    }
}

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let dot: f64 = a.iter().zip(b).map(|(x, y)| *x as f64 * *y as f64).sum();
    let na = a.iter().map(|x| *x as f64 * *x as f64).sum::<f64>().sqrt();
    let nb = b.iter().map(|x| *x as f64 * *x as f64).sum::<f64>().sqrt();
    if na == 0.0 || nb == 0.0 {
        0.0
    } else {
        dot / (na * nb)
    }
}

struct Oracle {
    vector_top: Vec<(NodeId, f64)>,
    vector_margin_ok: bool,
    alpha_ids: BTreeSet<NodeId>,
}

async fn oracle(snapshot: &Snapshot<'_>, probe: &[f32]) -> Oracle {
    let rows = snapshot.scan_label("Doc").await.unwrap();
    let mut scored = Vec::new();
    let mut alpha_ids = BTreeSet::new();
    for row in &rows {
        if let Some(CoreValue::Vec(embedding)) = row.properties.get("embedding") {
            scored.push((row.id, cosine(embedding, probe)));
        }
        if let Some(CoreValue::Str(text)) = row.properties.get("body") {
            if text.split_whitespace().any(|token| token == "alpha") {
                alpha_ids.insert(row.id);
            }
        }
    }
    scored.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    let vector_margin_ok = scored
        .get(K - 1)
        .zip(scored.get(K))
        .map(|(kth, next)| kth.1 - next.1 > 1e-6)
        .unwrap_or(true);
    scored.truncate(K);
    Oracle {
        vector_top: scored,
        vector_margin_ok,
        alpha_ids,
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn search_stays_snapshot_exact_under_flush_compaction_and_sweeps() {
    let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
    let paths = NamespacePaths::new("tenants", NamespaceId::new("search-concurrency").unwrap());
    let mut writer = WriterSession::open(store.clone(), paths.clone())
        .await
        .unwrap();
    writer
        .register_vector_index(
            VectorIndexDescriptor {
                name: "doc_emb".into(),
                label: "Doc".into(),
                property: "embedding".into(),
                dim: DIM,
                metric: VectorMetric::Cosine,
                r: 16,
                l_build: 32,
                alpha: 1.2,
                quantization: VectorQuantization::None,
            },
            false,
        )
        .await
        .unwrap();
    writer
        .register_text_index(
            TextIndexDescriptor::new("doc_ft".into(), "Doc".into(), vec!["body".into()]),
            false,
        )
        .await
        .unwrap();

    // Seed corpus: 240 documents across two flushes, then one compaction.
    let mut ids: BTreeMap<u64, NodeId> = BTreeMap::new();
    for ordinal in 0..240u64 {
        let id = NodeId::new();
        ids.insert(ordinal, id);
        writer.upsert_node("Doc", id, &record(ordinal, 0)).unwrap();
        if ordinal == 119 {
            writer.flush(schema()).await.unwrap();
        }
    }
    writer.flush(schema()).await.unwrap();
    writer.compact_l0(&schema()).await.unwrap();

    let manifest_store = namidb_storage::manifest::ManifestStore::new(store.clone(), paths.clone());
    // Readers discover manifests exactly as S3 reader nodes do: from the
    // committed pointer in the object store, never from writer internals.
    let seed = manifest_store.load_current().await.unwrap();
    let (manifest_tx, manifest_rx) = tokio::sync::watch::channel(seed);
    // Readers report the lowest manifest version any of them may still hold;
    // the janitor's retention horizon never passes it.
    let (floor_tx, floor_rx) = tokio::sync::watch::channel(manifest_rx.borrow().manifest.version);
    let probe: Vec<f32> = embedding(0, 0);

    let mut readers = Vec::new();
    for reader_index in 0..3 {
        let mut manifest_rx = manifest_rx.clone();
        let floor_tx = floor_tx.clone();
        let store = store.clone();
        let paths = paths.clone();
        let probe = probe.clone();
        readers.push(tokio::spawn(async move {
            let mut native_serves = 0usize;
            let mut iterations = 0usize;
            loop {
                let loaded = manifest_rx.borrow_and_update().clone();
                let done = loaded.manifest.version == u64::MAX;
                if done {
                    break;
                }
                // Pin the version this reader is about to serve from.
                floor_tx.send_if_modified(|floor| {
                    if loaded.manifest.version < *floor {
                        *floor = loaded.manifest.version;
                        true
                    } else {
                        false
                    }
                });
                let memtable = Memtable::new();
                let view = memtable.snapshot_view();
                let snapshot = Snapshot::new(loaded, &view, store.clone(), paths.clone());
                let expected = oracle(&snapshot, &probe).await;

                if let Some((hits, _point_count)) = snapshot
                    .try_vector_search_with_point_count("doc_emb", &probe, K, 64)
                    .await
                    .unwrap()
                {
                    native_serves += 1;
                    if expected.vector_margin_ok {
                        let got: Vec<NodeId> = hits.iter().map(|(id, _)| *id).collect();
                        let want: Vec<NodeId> =
                            expected.vector_top.iter().map(|(id, _)| *id).collect();
                        assert_eq!(
                            got, want,
                            "reader {reader_index}: vector top-{K} must match the \
                             same-snapshot oracle exactly"
                        );
                    }
                }

                if let Some(hits) = snapshot
                    .text_search(
                        "doc_ft",
                        "Doc",
                        &namidb_storage::text::parse_query("alpha"),
                        None,
                    )
                    .await
                    .unwrap()
                {
                    native_serves += 1;
                    let got: BTreeSet<NodeId> = hits.iter().map(|(id, _)| *id).collect();
                    assert_eq!(
                        got, expected.alpha_ids,
                        "reader {reader_index}: the served text matches must be \
                         exactly the snapshot's live alpha documents"
                    );
                }
                iterations += 1;
                tokio::task::yield_now().await;
            }
            (iterations, native_serves)
        }));
    }

    // Writer: six maintenance cycles of updates, deletes, inserts, flushes,
    // compactions and floor-respecting sweeps.
    let mut next_ordinal = 240u64;
    for cycle in 1..=6u64 {
        for slot in 0..10u64 {
            let target = (cycle * 17 + slot * 7) % next_ordinal;
            if let Some(id) = ids.get(&target).copied() {
                writer
                    .upsert_node("Doc", id, &record(target, cycle))
                    .unwrap();
            }
        }
        for slot in 0..5u64 {
            let target = (cycle * 29 + slot * 11) % next_ordinal;
            if let Some(id) = ids.remove(&target) {
                writer.tombstone_node("Doc", id).unwrap();
            }
        }
        for _ in 0..5 {
            let id = NodeId::new();
            ids.insert(next_ordinal, id);
            writer
                .upsert_node("Doc", id, &record(next_ordinal, cycle))
                .unwrap();
            next_ordinal += 1;
        }
        writer.flush(schema()).await.unwrap();
        if cycle % 2 == 0 {
            writer.compact_l0(&schema()).await.unwrap();
        }
        let floor = *floor_rx.borrow();
        sweep_orphans(&manifest_store, floor, Duration::ZERO, 10, true)
            .await
            .unwrap();
        manifest_tx
            .send(manifest_store.load_current().await.unwrap())
            .unwrap();
        tokio::task::yield_now().await;
    }

    // Signal completion with a sentinel version.
    let mut sentinel = manifest_store.load_current().await.unwrap();
    sentinel.manifest.version = u64::MAX;
    manifest_tx.send(sentinel).unwrap();

    let mut total_iterations = 0;
    let mut total_native = 0;
    for reader in readers {
        let (iterations, native) = reader.await.unwrap();
        total_iterations += iterations;
        total_native += native;
    }
    assert!(
        total_iterations >= 6,
        "readers must have observed the maintenance cycles ({total_iterations})"
    );
    assert!(
        total_native > 0,
        "at least some queries must have served from the native routes, or the \
         parity checks above proved nothing about the index paths"
    );
}
