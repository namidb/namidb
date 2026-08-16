//! Unaudited dimension from docs/testing/25tb-readiness.md: DDL racing
//! queries and maintenance. Index lifecycle was only ever audited
//! sequentially; here DROP/CREATE of a text index races live reader
//! snapshots and flush/compaction of the same label. The contract: readers
//! NEVER error or observe a torn index — every query answers `Ok`, native
//! when a generation is active, flat-signal (`None`) otherwise, and matches
//! a same-snapshot oracle; after the last recreate the index serves again.
#![cfg(all(feature = "vector-index", feature = "text-index"))]

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use namidb_core::id::{NamespaceId, NodeId};
use namidb_core::schema::{DataType, LabelDef, PropertyDef, Schema, SchemaBuilder};
use namidb_core::value::Value as CoreValue;
use namidb_storage::manifest::TextIndexDescriptor;
use namidb_storage::memtable::Memtable;
use namidb_storage::read::Snapshot;
use namidb_storage::{NamespacePaths, NodeWriteRecord, WriterSession};
use object_store::memory::InMemory;
use object_store::ObjectStore;

fn schema() -> Schema {
    SchemaBuilder::new()
        .label(LabelDef {
            name: "Doc".into(),
            properties: vec![PropertyDef::new("body", DataType::Utf8, true).unwrap()],
        })
        .unwrap()
        .build()
}

fn doc(body: &str) -> NodeWriteRecord {
    let mut props = BTreeMap::new();
    props.insert("body".to_string(), CoreValue::Str(body.to_string()));
    NodeWriteRecord {
        properties: props,
        schema_version: 1,
        ..Default::default()
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn index_ddl_races_live_readers_without_errors_or_torn_results() {
    let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
    let paths = NamespacePaths::new("tenants", NamespaceId::new("ddl-races").unwrap());
    let mut writer = WriterSession::open(store.clone(), paths.clone())
        .await
        .unwrap();
    writer
        .register_text_index(
            TextIndexDescriptor::new("doc_ft".into(), "Doc".into(), vec!["body".into()]),
            false,
        )
        .await
        .unwrap();
    let mut alpha_ids = BTreeSet::new();
    for ordinal in 0..30u64 {
        let id = NodeId::new();
        let body = if ordinal % 3 == 0 {
            alpha_ids.insert(id);
            format!("alpha doc {ordinal}")
        } else {
            format!("plain doc {ordinal}")
        };
        writer.upsert_node("Doc", id, &doc(&body)).unwrap();
    }
    writer.flush(schema()).await.unwrap();
    writer.compact_l0(&schema()).await.unwrap();

    let manifest_store = namidb_storage::manifest::ManifestStore::new(store.clone(), paths.clone());
    let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let mut readers = Vec::new();
    for _ in 0..2 {
        let store = store.clone();
        let paths = paths.clone();
        let manifest_store =
            namidb_storage::manifest::ManifestStore::new(store.clone(), paths.clone());
        let stop = stop.clone();
        readers.push(tokio::spawn(async move {
            let mut native_serves = 0usize;
            let mut iterations = 0usize;
            loop {
                if stop.load(std::sync::atomic::Ordering::SeqCst) {
                    break;
                }
                let loaded = manifest_store.load_current().await.unwrap();
                let memtable = Memtable::new();
                let view = memtable.snapshot_view();
                let snapshot = Snapshot::new(loaded, &view, store.clone(), paths.clone());
                // The oracle and the query come from the SAME snapshot, so a
                // mid-DDL manifest can never produce a mismatch — only a
                // clean serve or a clean flat signal.
                let expected: BTreeSet<NodeId> = snapshot
                    .scan_label("Doc")
                    .await
                    .unwrap()
                    .into_iter()
                    .filter(|row| {
                        matches!(row.properties.get("body"), Some(CoreValue::Str(text))
                            if text.split_whitespace().any(|token| token == "alpha"))
                    })
                    .map(|row| row.id)
                    .collect();
                if let Some(hits) = snapshot
                    .text_search(
                        "doc_ft",
                        "Doc",
                        &namidb_storage::text::parse_query("alpha"),
                        None,
                    )
                    .await
                    .expect("a query racing DDL must never error")
                {
                    native_serves += 1;
                    let got: BTreeSet<NodeId> = hits.iter().map(|(id, _)| *id).collect();
                    assert_eq!(
                        got, expected,
                        "a native serve racing DDL must match its own snapshot"
                    );
                }
                iterations += 1;
                tokio::task::yield_now().await;
            }
            (iterations, native_serves)
        }));
    }

    // The DDL storm: drop, recreate, add rows, flush, compact — four cycles.
    for cycle in 0..4u64 {
        writer.drop_text_index("doc_ft", false).await.unwrap();
        writer
            .register_text_index(
                TextIndexDescriptor::new("doc_ft".into(), "Doc".into(), vec!["body".into()]),
                false,
            )
            .await
            .unwrap();
        for ordinal in 0..5u64 {
            let id = NodeId::new();
            let body = format!("alpha extra {cycle}-{ordinal}");
            alpha_ids.insert(id);
            writer.upsert_node("Doc", id, &doc(&body)).unwrap();
        }
        writer.flush(schema()).await.unwrap();
        writer.compact_l0(&schema()).await.unwrap();
        tokio::task::yield_now().await;
    }
    stop.store(true, std::sync::atomic::Ordering::SeqCst);

    let mut total_iterations = 0usize;
    let mut total_native = 0usize;
    for reader in readers {
        let (iterations, native) = reader.await.unwrap();
        total_iterations += iterations;
        total_native += native;
    }
    assert!(total_iterations > 0);
    // Whether any mid-storm read lands on an ACTIVE window is timing
    // dependent (DDL cycles are quick); the deterministic native-serving
    // proof is the post-storm assertion below. Mid-storm serves are recorded
    // for signal, not required.
    let _ = total_native;

    // After the storm the recreated index serves the full corpus.
    let loaded = manifest_store.load_current().await.unwrap();
    let memtable = Memtable::new();
    let view = memtable.snapshot_view();
    let snapshot = Snapshot::new(loaded, &view, store.clone(), paths.clone());
    let hits = snapshot
        .text_search(
            "doc_ft",
            "Doc",
            &namidb_storage::text::parse_query("alpha"),
            None,
        )
        .await
        .unwrap()
        .expect("the recreated index must serve after the DDL storm");
    let got: BTreeSet<NodeId> = hits.iter().map(|(id, _)| *id).collect();
    assert_eq!(got, alpha_ids, "the final generation reflects every write");
}
