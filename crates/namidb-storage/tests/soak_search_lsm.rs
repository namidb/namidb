//! Long-running Search-LSM soak: the doc-mandated end-to-end exercise above
//! toy corpus sizes. Seeds `NAMIDB_SOAK_ROWS` documents (default 1M) through
//! repeated flush + compaction cycles with interleaved updates and deletes —
//! forcing multi-SST, multi-level, multi-delta-segment state — then proves
//! the native vector and text routes still serve exactly.
//!
//! Oracles are structural, not O(corpus): every document carries a globally
//! unique token (text must return exactly that document) and a deterministic
//! well-separated embedding (top-1 for a document's own embedding must be the
//! document itself). Deleted documents must vanish from both routes; updated
//! documents must serve their newest version.
//!
//! Runs only under `--ignored` (the nightly workflow); scale locally with
//! e.g. `NAMIDB_SOAK_ROWS=20000`.
#![cfg(all(feature = "vector-index", feature = "text-index"))]

use std::collections::BTreeMap;
use std::sync::Arc;

use namidb_core::id::{NamespaceId, NodeId};
use namidb_core::schema::{DataType, LabelDef, PropertyDef, Schema, SchemaBuilder};
use namidb_core::value::Value as CoreValue;
use namidb_storage::manifest::{
    TextIndexDescriptor, VectorIndexDescriptor, VectorMetric, VectorQuantization,
};
use namidb_storage::memtable::Memtable;
use namidb_storage::read::Snapshot;
use namidb_storage::{NamespacePaths, NodeWriteRecord, WriterSession};
use object_store::memory::InMemory;
use object_store::ObjectStore;

const DIM: u32 = 16;
const BATCHES: u64 = 20;
const COMPACT_EVERY: u64 = 4;
const SAMPLES: u64 = 50;

fn soak_rows() -> u64 {
    std::env::var("NAMIDB_SOAK_ROWS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(1_000_000)
}

fn schema() -> Schema {
    SchemaBuilder::new()
        .label(LabelDef {
            name: "Doc".into(),
            properties: vec![
                PropertyDef::new("embedding", DataType::FloatVector { dim: DIM }, true).unwrap(),
                PropertyDef::new("body", DataType::Utf8, true).unwrap(),
            ],
        })
        .unwrap()
        .build()
}

fn splitmix(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9e3779b97f4a7c15);
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
    x ^ (x >> 31)
}

/// Deterministic pseudo-random unit-scale embedding: with `DIM` independent
/// coordinates a document's own embedding is its unique cosine top-1 (self
/// similarity 1.0; any distinct point ties only with negligible probability).
fn embedding(ordinal: u64, version: u64) -> Vec<f32> {
    (0..DIM as u64)
        .map(|j| {
            let bits = splitmix(ordinal ^ version.rotate_left(32) ^ (j << 48).wrapping_add(j));
            0.05 + (bits % 100_000) as f32 / 100_000.0
        })
        .collect()
}

fn doc_id(ordinal: u64) -> NodeId {
    let mut bytes = [0u8; 16];
    bytes[0] = 0x51;
    bytes[8..].copy_from_slice(&ordinal.to_be_bytes());
    NodeId::from_uuid(uuid::Uuid::from_bytes(bytes))
}

fn record(ordinal: u64, version: u64) -> NodeWriteRecord {
    let mut properties = BTreeMap::new();
    properties.insert(
        "embedding".into(),
        CoreValue::Vec(embedding(ordinal, version)),
    );
    properties.insert(
        "body".into(),
        CoreValue::Str(format!("common filler tok{ordinal:09} v{version}")),
    );
    NodeWriteRecord {
        properties,
        schema_version: 1,
        ..Default::default()
    }
}

#[cfg(target_os = "linux")]
fn peak_rss_bytes() -> Option<u64> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    let line = status.lines().find(|line| line.starts_with("VmHWM:"))?;
    let kib: u64 = line.split_whitespace().nth(1)?.parse().ok()?;
    Some(kib * 1024)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "long Search-LSM soak (default 1M rows); nightly runs it, scale locally with NAMIDB_SOAK_ROWS"]
async fn soak_native_search_stays_exact_through_a_million_row_lifecycle() {
    let rows = soak_rows().max(BATCHES * 10);
    let batch_rows = rows / BATCHES;
    let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
    let paths = NamespacePaths::new("tenants", NamespaceId::new("soak-search-lsm").unwrap());
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

    // Version per ordinal: 0 = live at seed version, u64::MAX = deleted.
    let mut versions: BTreeMap<u64, u64> = BTreeMap::new();
    for batch in 0..BATCHES {
        let start = batch * batch_rows;
        for ordinal in start..start + batch_rows {
            writer
                .upsert_node("Doc", doc_id(ordinal), &record(ordinal, 0))
                .unwrap();
            versions.insert(ordinal, 0);
        }
        // Churn over already-seeded ordinals: updates rewrite the embedding
        // and the versioned token suffix; deletes tombstone outright. Both
        // must be reflected exactly by the native routes at the end.
        if batch > 0 {
            let seeded = start;
            for slot in 0..batch_rows / 100 {
                let target = splitmix(batch * 1_000_003 + slot) % seeded;
                if versions.get(&target).is_some_and(|v| *v != u64::MAX) {
                    writer
                        .upsert_node("Doc", doc_id(target), &record(target, batch))
                        .unwrap();
                    versions.insert(target, batch);
                }
            }
            for slot in 0..batch_rows / 200 {
                let target = splitmix(batch * 7_000_009 + slot) % seeded;
                if versions.get(&target).is_some_and(|v| *v != u64::MAX) {
                    writer.tombstone_node("Doc", doc_id(target)).unwrap();
                    versions.insert(target, u64::MAX);
                }
            }
        }
        writer.flush(schema()).await.unwrap();
        if (batch + 1) % COMPACT_EVERY == 0 {
            writer.compact_l0(&schema()).await.unwrap();
        }
    }
    writer.compact_l0(&schema()).await.unwrap();

    // Reader-node topology: the committed manifest from the store, an empty
    // memtable, never writer internals.
    let manifest_store = namidb_storage::manifest::ManifestStore::new(store.clone(), paths.clone());
    let loaded = manifest_store.load_current().await.unwrap();

    // Compaction must keep physical fan-out bounded: a run that leaks one
    // delta segment (or barrier) per flush would pass every parity check and
    // still be unusable at scale.
    for kind_name in ["VectorGraph", "TextIndex"] {
        let count = loaded
            .manifest
            .ssts
            .iter()
            .filter(|descriptor| format!("{:?}", descriptor.kind) == kind_name)
            .count();
        assert!(
            count <= 16,
            "{kind_name} physical fan-out must stay bounded, found {count}"
        );
    }

    let memtable = Memtable::new();
    let view = memtable.snapshot_view();
    let snapshot = Snapshot::new(loaded, &view, store.clone(), paths.clone());

    let live: Vec<(u64, u64)> = versions
        .iter()
        .filter(|(_, version)| **version != u64::MAX)
        .map(|(ordinal, version)| (*ordinal, *version))
        .collect();
    let deleted: Vec<u64> = versions
        .iter()
        .filter(|(_, version)| **version == u64::MAX)
        .map(|(ordinal, _)| *ordinal)
        .collect();
    assert!(
        deleted.len() as u64 >= SAMPLES / 2,
        "churn must have deleted rows"
    );

    for sample in 0..SAMPLES {
        let (ordinal, version) = live[(splitmix(sample) % live.len() as u64) as usize];
        let probe = embedding(ordinal, version);
        let (hits, _points) = snapshot
            .try_vector_search_with_point_count("doc_emb", &probe, 1, 64)
            .await
            .unwrap()
            .expect("the vector route must serve natively after the lifecycle");
        assert_eq!(
            hits.first().map(|(id, _)| *id),
            Some(doc_id(ordinal)),
            "own-embedding top-1 must be the document itself (ordinal {ordinal} v{version})"
        );

        let token_query = namidb_storage::text::parse_query(&format!("tok{ordinal:09}"));
        let hits = snapshot
            .text_search("doc_ft", "Doc", &token_query, None)
            .await
            .unwrap()
            .expect("the text route must serve natively after the lifecycle");
        assert_eq!(
            hits.iter().map(|(id, _)| *id).collect::<Vec<_>>(),
            vec![doc_id(ordinal)],
            "the unique token must match exactly its document (ordinal {ordinal})"
        );

        let dead = deleted[(splitmix(sample * 31 + 7) % deleted.len() as u64) as usize];
        let dead_query = namidb_storage::text::parse_query(&format!("tok{dead:09}"));
        let hits = snapshot
            .text_search("doc_ft", "Doc", &dead_query, None)
            .await
            .unwrap()
            .expect("the text route must serve natively for dead-token probes too");
        assert!(
            hits.is_empty(),
            "a deleted document's unique token must return nothing (ordinal {dead})"
        );
    }

    // The lifecycle above must complete in bounded memory: the streaming
    // build/compaction paths spool to disk instead of holding the corpus.
    // The InMemory object store legitimately retains every live SST body, so
    // the ceiling is (store bytes) + a fixed working-set allowance, not a
    // constant.
    #[cfg(target_os = "linux")]
    if let Some(peak) = peak_rss_bytes() {
        let mut store_bytes = 0u64;
        let mut listing = store.list(None);
        use futures::StreamExt;
        while let Some(meta) = listing.next().await {
            store_bytes += meta.unwrap().size;
        }
        let ceiling = store_bytes.saturating_mul(3) + 4 * 1024 * 1024 * 1024;
        assert!(
            peak <= ceiling,
            "peak RSS {peak} exceeds the bounded-memory ceiling {ceiling} \
             (store holds {store_bytes})"
        );
    }
}
