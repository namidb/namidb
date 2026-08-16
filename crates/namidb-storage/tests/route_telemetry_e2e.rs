//! Plan item 34 companion: the route counters must move with the REAL
//! serving decision — a native text serve increments the native counter, a
//! freshness-gated decline increments the fallback counter.
#![cfg(feature = "text-index")]

use std::collections::BTreeMap;
use std::sync::Arc;

use namidb_core::id::{NamespaceId, NodeId};
use namidb_core::schema::{DataType, LabelDef, PropertyDef, Schema, SchemaBuilder};
use namidb_core::value::Value as CoreValue;
use namidb_storage::manifest::TextIndexDescriptor;
use namidb_storage::{route_telemetry, NamespacePaths, NodeWriteRecord, WriterSession};
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

#[tokio::test]
async fn route_counters_track_native_serves_and_freshness_fallbacks() {
    let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
    let paths = NamespacePaths::new("tenants", NamespaceId::new("route-telemetry").unwrap());
    let mut writer = WriterSession::open(store, paths).await.unwrap();
    writer
        .register_text_index(
            TextIndexDescriptor::new("doc_ft".into(), "Doc".into(), vec!["body".into()]),
            false,
        )
        .await
        .unwrap();
    writer
        .upsert_node("Doc", NodeId::new(), &doc("alpha one"))
        .unwrap();
    writer
        .upsert_node("Doc", NodeId::new(), &doc("beta two"))
        .unwrap();
    writer.commit_batch().await.unwrap();
    writer.flush(schema()).await.unwrap();
    writer.compact_l0(&schema()).await.unwrap();

    // Clean snapshot: the index serves and the NATIVE counter moves.
    let before = route_telemetry::snapshot();
    let snapshot = writer.snapshot();
    let outcome = snapshot
        .text_search(
            "doc_ft",
            "Doc",
            &namidb_storage::text::parse_query("alpha"),
            None,
        )
        .await
        .unwrap();
    assert!(outcome.is_some(), "the clean snapshot must serve natively");
    drop(snapshot);
    let after = route_telemetry::snapshot();
    assert!(
        after.text_native > before.text_native,
        "a native serve must move the native counter"
    );

    // Dirty memtable: the freshness gate declines and FALLBACK moves.
    writer
        .upsert_node("Doc", NodeId::new(), &doc("gamma three"))
        .unwrap();
    writer.commit_batch().await.unwrap();
    let before = route_telemetry::snapshot();
    let snapshot = writer.snapshot();
    let outcome = snapshot
        .text_search(
            "doc_ft",
            "Doc",
            &namidb_storage::text::parse_query("alpha"),
            None,
        )
        .await
        .unwrap();
    assert!(outcome.is_none(), "the dirty snapshot must decline");
    let after = route_telemetry::snapshot();
    assert!(
        after.text_fallback > before.text_fallback,
        "a freshness decline must move the fallback counter"
    );
}
