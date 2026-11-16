//! An off-lock flush whose base gained only a search compaction installs.
//!
//! The server builds a flush off the writer lock and installs it under the
//! lock. Up to 3.4.0 any manifest change between the two (for example a
//! search-index compaction finishing meanwhile) refused the install, and the
//! server redid the whole flush under the writer lock, blocking every write
//! for the length of a flush build. A change that touched only search-index
//! segments is now rebased: the flushed delta is appended to the compacted
//! generation and nothing is rebuilt.
#![cfg(feature = "text-index")]

use std::collections::BTreeMap;
use std::sync::Arc;

use namidb_core::id::{NamespaceId, NodeId};
use namidb_core::schema::{DataType, LabelDef, PropertyDef, Schema, SchemaBuilder};
use namidb_core::value::Value as CoreValue;
use namidb_storage::manifest::TextIndexDescriptor;
use namidb_storage::{BackgroundFlushInstall, NamespacePaths, NodeWriteRecord, WriterSession};
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
    let mut properties = BTreeMap::new();
    properties.insert("body".into(), CoreValue::Str(body.into()));
    NodeWriteRecord {
        properties,
        schema_version: 1,
        ..Default::default()
    }
}

async fn hits(writer: &WriterSession) -> usize {
    let query = namidb_storage::text::parse_query("texto");
    writer
        .snapshot()
        .text_search("doc_ft", "Doc", &query, None)
        .await
        .unwrap()
        .unwrap_or_else(|| panic!("the index serves natively"))
        .len()
}

fn segments(writer: &WriterSession) -> usize {
    let manifest = writer.snapshot().manifest().manifest.clone();
    namidb_storage::search_lsm::validate_search_lsm(&manifest).unwrap();
    manifest.search_lsm[0].segments.len()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_background_flush_installs_over_a_concurrent_search_compaction() {
    let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
    let paths = NamespacePaths::new("tenants", NamespaceId::new("bg-rebase").unwrap());
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
    // Enough single-document flushes for a delta-run compaction to be due.
    for round in 0..6 {
        writer
            .upsert_node("Doc", NodeId::new(), &doc(&format!("ronda {round} texto")))
            .unwrap();
        writer.flush(schema()).await.unwrap();
    }
    assert!(writer.search_compaction_due());
    let before = segments(&writer);

    // Cut an off-lock flush, then let a search compaction land first.
    writer
        .upsert_node("Doc", NodeId::new(), &doc("pendiente texto"))
        .unwrap();
    writer.commit_batch().await.unwrap();
    let ticket = writer.begin_background_flush().await.unwrap().unwrap();
    let compacted = writer.compact_search_segments().await.unwrap();
    assert!(compacted.source_ssts_removed > 0, "the compaction applied");
    let after_compaction = segments(&writer);
    assert!(after_compaction < before);

    let built = ticket.build().await.unwrap();
    let outcome = writer.install_background_flush(built).await.unwrap();
    assert!(
        matches!(outcome, BackgroundFlushInstall::Installed(_)),
        "a search-only compaction must not supersede the flush"
    );
    assert_eq!(segments(&writer), after_compaction + 1);
    // Native search (it serves only with no unflushed document of the label)
    // sees the compacted segments plus the rebased delta.
    assert_eq!(hits(&writer).await, 7);

    // The rebased generation keeps accepting flushes.
    writer
        .upsert_node("Doc", NodeId::new(), &doc("tardio texto"))
        .unwrap();
    writer.flush(schema()).await.unwrap();
    assert_eq!(hits(&writer).await, 8);

    drop(writer);
    let reopened = WriterSession::open(store, paths).await.unwrap();
    assert_eq!(hits(&reopened).await, 8);
}
