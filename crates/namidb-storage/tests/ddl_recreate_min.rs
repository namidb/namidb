//! Minimal sequential reproduction: DROP + recreate a text index, add rows,
//! flush, compact. No concurrency at all.
#![cfg(all(feature = "vector-index", feature = "text-index"))]

use std::collections::BTreeMap;
use std::sync::Arc;

use namidb_core::id::{NamespaceId, NodeId};
use namidb_core::schema::{DataType, LabelDef, PropertyDef, Schema, SchemaBuilder};
use namidb_core::value::Value as CoreValue;
use namidb_storage::manifest::TextIndexDescriptor;
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

#[tokio::test]
async fn drop_recreate_flush_compact_sequentially() {
    let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
    let paths = NamespacePaths::new("tenants", NamespaceId::new("ddl-min").unwrap());
    let mut writer = WriterSession::open(store, paths).await.unwrap();
    writer
        .register_text_index(
            TextIndexDescriptor::new("doc_ft".into(), "Doc".into(), vec!["body".into()]),
            false,
        )
        .await
        .unwrap();
    for i in 0..6u64 {
        writer
            .upsert_node("Doc", NodeId::new(), &doc(&format!("alpha {i}")))
            .unwrap();
    }
    writer.flush(schema()).await.unwrap();
    writer.compact_l0(&schema()).await.unwrap();

    for cycle in 0..4u64 {
        eprintln!("MINDBG {cycle} drop");
        writer.drop_text_index("doc_ft", false).await.unwrap();
        writer
            .register_text_index(
                TextIndexDescriptor::new("doc_ft".into(), "Doc".into(), vec!["body".into()]),
                false,
            )
            .await
            .unwrap();
        writer
            .upsert_node("Doc", NodeId::new(), &doc(&format!("alpha nuevo {cycle}")))
            .unwrap();
        eprintln!("MINDBG {cycle} flush");
        writer.flush(schema()).await.unwrap();
        eprintln!("MINDBG {cycle} compact");
        writer.compact_l0(&schema()).await.unwrap();
        eprintln!("MINDBG {cycle} done");
    }
}
