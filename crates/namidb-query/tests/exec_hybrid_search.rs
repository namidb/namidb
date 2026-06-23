//! End-to-end hybrid search: `CALL search.hybrid({...})` fuses a dense (vector
//! KNN) and a sparse (BM25) retrieval with Reciprocal Rank Fusion (default) or a
//! weighted-linear blend. Also covers the `search.vector` and Neo4j-compatible
//! `db.index.vector.queryNodes` procedures. These run the flat-scan path (no
//! index required), which is freshness-equivalent to the indexed path.

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

/// Seed three `:Doc` nodes, each with a `title`, a `body` (BM25 corpus) and a
/// 3-D `embedding` (vector corpus).
async fn seed(name: &str) -> WriterSession {
    let mut writer = WriterSession::open(store(), paths(name)).await.unwrap();
    let docs: [(&str, &str, Vec<f32>); 3] = [
        ("alpha", "quantum physics lecture", vec![1.0, 0.0, 0.0]),
        ("beta", "italian pasta recipe", vec![0.0, 1.0, 0.0]),
        ("gamma", "quantum pasta experiment", vec![0.8, 0.2, 0.0]),
    ];
    for (title, body, emb) in docs {
        let mut p = BTreeMap::new();
        p.insert("title".into(), CoreValue::Str(title.into()));
        p.insert("body".into(), CoreValue::Str(body.into()));
        p.insert("embedding".into(), CoreValue::Vec(emb));
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

/// Titles in result order (rows project `node.title AS title`).
fn titles(rows: &[namidb_query::Row]) -> Vec<String> {
    rows.iter()
        .filter_map(|r| match r.get("title") {
            Some(RuntimeValue::String(s)) => Some(s.clone()),
            _ => None,
        })
        .collect()
}

async fn run(w: &WriterSession, cypher: &str, q: Vec<f32>) -> Vec<namidb_query::Row> {
    let snap = w.snapshot();
    let plan = lower(&parse(cypher).unwrap()).unwrap();
    let mut params = Params::new();
    params.insert("q".into(), RuntimeValue::Vector(q));
    execute(&plan, &snap, &params).await.unwrap()
}

#[tokio::test]
async fn hybrid_rrf_rewards_documents_strong_in_both_legs() {
    let w = seed("hybrid-rrf").await;
    // Query: text "quantum", vector along x. `alpha` is the top of both legs;
    // `gamma` is second in both; `beta` is only weakly in the vector leg and
    // absent from the text leg. RRF should rank alpha > gamma > beta.
    let cypher = "CALL search.hybrid({ \
         label: 'Doc', \
         query_text: 'quantum', text_property: 'body', \
         query_vector: $q, vector_property: 'embedding', \
         k: 3 \
       }) YIELD node, score RETURN node.title AS title, score";
    let rows = run(&w, cypher, vec![1.0, 0.0, 0.0]).await;
    assert_eq!(
        titles(&rows),
        vec!["alpha".to_string(), "gamma".to_string(), "beta".to_string()],
        "RRF fused order"
    );
    // Scores are non-increasing.
    let scores: Vec<f64> = rows
        .iter()
        .filter_map(|r| match r.get("score") {
            Some(RuntimeValue::Float(f)) => Some(*f),
            _ => None,
        })
        .collect();
    assert!(scores.windows(2).all(|w| w[0] >= w[1] - 1e-9), "{scores:?}");
}

#[tokio::test]
async fn hybrid_dense_only_equals_vector_ranking() {
    let w = seed("hybrid-dense").await;
    // Dense leg only (no query_text) → identical to a pure vector KNN.
    let hybrid = "CALL search.hybrid({ label: 'Doc', \
         query_vector: $q, vector_property: 'embedding', k: 3 }) \
         YIELD node, score RETURN node.title AS title, score";
    let vector = "CALL search.vector({ label: 'Doc', property: 'embedding', \
         query: $q, k: 3 }) YIELD node, score RETURN node.title AS title, score";
    let h = titles(&run(&w, hybrid, vec![1.0, 0.0, 0.0]).await);
    let v = titles(&run(&w, vector, vec![1.0, 0.0, 0.0]).await);
    assert_eq!(h, v, "dense-only hybrid must equal search.vector");
    assert_eq!(
        h,
        vec!["alpha".to_string(), "gamma".to_string(), "beta".to_string()]
    );
}

#[tokio::test]
async fn hybrid_sparse_only_returns_only_text_matches() {
    let w = seed("hybrid-sparse").await;
    // Sparse leg only ("quantum") → alpha and gamma match; beta does not appear.
    let cypher = "CALL search.hybrid({ label: 'Doc', \
         query_text: 'quantum', text_property: 'body', k: 3 }) \
         YIELD node, score RETURN node.title AS title, score";
    let got = titles(&run(&w, cypher, vec![]).await);
    assert!(got.contains(&"alpha".to_string()) && got.contains(&"gamma".to_string()));
    assert!(
        !got.contains(&"beta".to_string()),
        "beta has no query term: {got:?}"
    );
}

#[tokio::test]
async fn hybrid_linear_fusion_runs_and_orders() {
    let w = seed("hybrid-linear").await;
    let cypher = "CALL search.hybrid({ label: 'Doc', \
         query_text: 'quantum', text_property: 'body', \
         query_vector: $q, vector_property: 'embedding', \
         fusion: 'linear', alpha: 0.5, k: 3 }) \
         YIELD node, score RETURN node.title AS title, score";
    let got = titles(&run(&w, cypher, vec![1.0, 0.0, 0.0]).await);
    // alpha is best in both legs → it wins under any sensible weighting.
    assert_eq!(got.first(), Some(&"alpha".to_string()), "{got:?}");
}

#[tokio::test]
async fn hybrid_requires_at_least_one_leg() {
    let w = seed("hybrid-empty").await;
    let snap = w.snapshot();
    let plan = lower(
        &parse("CALL search.hybrid({ label: 'Doc', k: 3 }) YIELD node, score RETURN node").unwrap(),
    )
    .unwrap();
    let err = execute(&plan, &snap, &Params::new()).await;
    assert!(err.is_err(), "hybrid with no legs configured must error");
}

#[tokio::test]
async fn search_vector_procedure_ranks_by_closeness() {
    let w = seed("vec-proc").await;
    let cypher = "CALL search.vector({ label: 'Doc', property: 'embedding', \
         query: $q, k: 2 }) YIELD node, score RETURN node.title AS title";
    let got = titles(&run(&w, cypher, vec![0.0, 1.0, 0.0]).await);
    // Query along y → beta closest, then gamma.
    assert_eq!(got, vec!["beta".to_string(), "gamma".to_string()]);
}
