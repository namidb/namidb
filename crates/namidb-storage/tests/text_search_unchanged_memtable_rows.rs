//! BM25 stays on its index while unflushed writes leave the indexed text
//! alone. An embedding-only `SET` (or any write to properties the index does
//! not cover) rewrites the whole row into the memtable; the index already
//! holds exactly that document, so `text_search` must keep serving natively,
//! with the same hits as before the write. A write that changes indexed text
//! or label membership is merged in from its memtable values (see
//! `search_fresh_merge_parity` in namidb-query for the exactness proof).
#![cfg(feature = "text-index")]

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
            properties: vec![
                PropertyDef::new("body", DataType::Utf8, true).unwrap(),
                PropertyDef::new("title", DataType::Utf8, true).unwrap(),
                PropertyDef::new("n", DataType::Int64, true).unwrap(),
            ],
        })
        .unwrap()
        .build()
}

fn doc(body: &str, title: &str, n: i64) -> NodeWriteRecord {
    let mut props = BTreeMap::new();
    props.insert("body".to_string(), CoreValue::Str(body.to_string()));
    props.insert("title".to_string(), CoreValue::Str(title.to_string()));
    props.insert("n".to_string(), CoreValue::I64(n));
    NodeWriteRecord {
        properties: props,
        schema_version: 1,
        ..Default::default()
    }
}

async fn search(writer: &WriterSession, query: &str) -> Option<Vec<(NodeId, f64)>> {
    writer
        .snapshot()
        .text_search(
            "doc_ft",
            "Doc",
            &namidb_storage::text::parse_query(query),
            Some(10),
        )
        .await
        .unwrap()
}

#[tokio::test]
async fn unchanged_indexed_text_keeps_bm25_on_the_index() {
    let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
    let paths = NamespacePaths::new("tenants", NamespaceId::new("text-unchanged").unwrap());
    let mut writer = WriterSession::open(store, paths).await.unwrap();
    writer
        .register_text_index(
            TextIndexDescriptor::new("doc_ft".into(), "Doc".into(), vec!["body".into()]),
            false,
        )
        .await
        .unwrap();
    let ids: Vec<NodeId> = (0..40).map(|_| NodeId::new()).collect();
    for (i, id) in ids.iter().enumerate() {
        let body = if i % 4 == 0 {
            "alpha beta"
        } else {
            "beta gamma"
        };
        writer
            .upsert_node("Doc", *id, &doc(body, &format!("t{i}"), i as i64))
            .unwrap();
    }
    writer.commit_batch().await.unwrap();
    writer.flush(schema()).await.unwrap();
    writer.compact_l0(&schema()).await.unwrap();
    let clean = search(&writer, "alpha")
        .await
        .expect("clean snapshot serves");
    assert_eq!(clean.len(), 10);

    // Rows rewritten with the same indexed text: non-indexed properties
    // change, in two commits, and one row twice.
    for (i, id) in ids.iter().enumerate().take(20) {
        let body = if i % 4 == 0 {
            "alpha beta"
        } else {
            "beta gamma"
        };
        writer
            .upsert_node(
                "Doc",
                *id,
                &doc(body, &format!("changed{i}"), 1000 + i as i64),
            )
            .unwrap();
    }
    writer.commit_batch().await.unwrap();
    assert_eq!(
        search(&writer, "alpha").await.as_ref(),
        Some(&clean),
        "unchanged indexed text must keep serving natively"
    );
    writer
        .upsert_node("Doc", ids[0], &doc("alpha beta", "again", 7))
        .unwrap();
    writer.commit_batch().await.unwrap();
    assert_eq!(
        search(&writer, "alpha").await.as_ref(),
        Some(&clean),
        "carried across commits"
    );

    // Changing an indexed value is merged in from the memtable, also when a
    // later commit writes the old text back (the row is not proven equal to
    // its persisted version any more, so it keeps being merged).
    writer
        .upsert_node("Doc", ids[1], &doc("delta", "t1", 1))
        .unwrap();
    writer.commit_batch().await.unwrap();
    assert!(
        search(&writer, "alpha").await.is_some(),
        "changed text is merged, not scanned"
    );
    writer
        .upsert_node("Doc", ids[1], &doc("beta gamma", "t1", 1))
        .unwrap();
    writer.commit_batch().await.unwrap();
    assert!(search(&writer, "alpha").await.is_some(), "still merged");

    // A flush folds the change into the index; serving resumes.
    writer.flush(schema()).await.unwrap();
    writer.compact_l0(&schema()).await.unwrap();
    assert!(
        search(&writer, "alpha").await.is_some(),
        "served after flush"
    );

    // A brand-new document carrying the label is served at once, ranked
    // first (two occurrences in the shortest document).
    let fresh = NodeId::new();
    writer
        .upsert_node("Doc", fresh, &doc("alpha alpha", "new", 99))
        .unwrap();
    writer.commit_batch().await.unwrap();
    let merged = search(&writer, "alpha")
        .await
        .expect("a new document is merged, not scanned");
    assert_eq!(merged[0].0, fresh);
    writer.flush(schema()).await.unwrap();

    // A fresh memtable whose first write to the index's label is unchanged
    // text is computed lazily (no commit tracked it) and serves.
    writer.compact_l0(&schema()).await.unwrap();
    let before = search(&writer, "alpha").await.expect("serves after flush");
    writer
        .upsert_node("Doc", ids[8], &doc("alpha beta", "lazy", 8))
        .unwrap();
    writer.commit_batch().await.unwrap();
    assert_eq!(search(&writer, "alpha").await, Some(before));
}
