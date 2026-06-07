//! End-to-end vector KNN: a `cosine_similarity(...)` ORDER BY over stored
//! embeddings ranks nodes by semantic closeness, with a WHERE pre-filter on the
//! candidate set. This is the Phase 1 "no dedicated operator" path: semantic
//! search expressed through the existing scan + sort + limit operators.

use std::collections::BTreeMap;
use std::sync::Arc;

use namidb_core::id::{NamespaceId, NodeId};
use namidb_core::value::Value as CoreValue;
use namidb_storage::{NamespacePaths, NodeWriteRecord, WriterSession};
use object_store::memory::InMemory;
use object_store::ObjectStore;

use namidb_query::{execute, lower, parse, Params, RuntimeValue};

fn store() -> Arc<dyn ObjectStore> {
    Arc::new(InMemory::new())
}

fn paths(name: &str) -> NamespacePaths {
    NamespacePaths::new("tenants", NamespaceId::new(name).unwrap())
}

/// Seed three `:Doc` nodes whose 3-D embeddings point in clearly different
/// directions, plus one with no embedding to exercise the pre-filter.
async fn seed_docs(name: &str) -> WriterSession {
    let mut writer = WriterSession::open(store(), paths(name)).await.unwrap();
    let docs: [(&str, Option<Vec<f32>>); 4] = [
        ("x-ish", Some(vec![1.0, 0.0, 0.0])),
        ("y-ish", Some(vec![0.0, 1.0, 0.0])),
        ("xy-ish", Some(vec![0.9, 0.1, 0.0])),
        ("no-embedding", None),
    ];
    for (title, emb) in docs {
        let mut p = BTreeMap::new();
        p.insert("title".into(), CoreValue::Str(title.into()));
        if let Some(emb) = emb {
            p.insert("embedding".into(), CoreValue::Vec(emb));
        }
        writer
            .upsert_node(
                "Doc",
                NodeId::new(),
                &NodeWriteRecord {
                    properties: p,
                    schema_version: 1,
                    ..Default::default()
                },
            )
            .unwrap();
    }
    writer.commit_batch().await.unwrap();
    writer
}

fn titles(rows: &[namidb_query::Row]) -> Vec<String> {
    rows.iter()
        .filter_map(|r| match r.get("title") {
            Some(RuntimeValue::String(s)) => Some(s.clone()),
            _ => None,
        })
        .collect()
}

#[tokio::test]
async fn cosine_knn_ranks_by_closeness() {
    let writer = seed_docs("knn-rank").await;
    let snap = writer.snapshot();

    // Query along the x axis: x-ish closest, then xy-ish, then y-ish.
    let mut params = Params::new();
    params.insert("q".into(), RuntimeValue::Vector(vec![1.0, 0.0, 0.0]));

    let plan = lower(
        &parse(
            "MATCH (d:Doc) WHERE d.embedding IS NOT NULL \
             RETURN d.title AS title, cosine_similarity(d.embedding, $q) AS score \
             ORDER BY score DESC LIMIT 2",
        )
        .unwrap(),
    )
    .unwrap();

    let rows = execute(&plan, &snap, &params).await.unwrap();
    assert_eq!(
        titles(&rows),
        vec!["x-ish".to_string(), "xy-ish".to_string()]
    );
}

#[tokio::test]
async fn knn_prefilter_excludes_nodes_without_embedding() {
    let writer = seed_docs("knn-prefilter").await;
    let snap = writer.snapshot();

    let mut params = Params::new();
    params.insert("q".into(), RuntimeValue::Vector(vec![0.0, 1.0, 0.0]));

    // No LIMIT: the WHERE pre-filter must drop the embedding-less node, so only
    // the three real embeddings come back (closest to the y axis first).
    let plan = lower(
        &parse(
            "MATCH (d:Doc) WHERE d.embedding IS NOT NULL \
             RETURN d.title AS title, cosine_similarity(d.embedding, $q) AS score \
             ORDER BY score DESC",
        )
        .unwrap(),
    )
    .unwrap();

    let rows = execute(&plan, &snap, &params).await.unwrap();
    let got = titles(&rows);
    assert_eq!(got.len(), 3, "the no-embedding doc must be filtered out");
    assert_eq!(got.first().map(String::as_str), Some("y-ish"));
    assert!(!got.iter().any(|t| t == "no-embedding"));
}
