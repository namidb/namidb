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

// ── Procedure `filter` (issue c): index-side filtering, not post-truncation ──

#[tokio::test]
async fn vector_procedure_filter_overfetches_not_post_truncates() {
    let w = seed("vec-filter-overfetch").await;
    // q = x-axis ⇒ vector rank is alpha (1,0,0) > gamma (.8,.2,0) > beta (0,1,0).
    // Ask for k=1 but constrain to `beta`, the FARTHEST doc. A naive post-filter
    // over the k=1 top-list would take alpha, then drop it → 0 rows. The
    // over-fetch path must instead surface beta. This is the whole point of (c).
    let cypher = "CALL search.vector({ label: 'Doc', property: 'embedding', \
         query: $q, k: 1, filter: { title: 'beta' } }) \
         YIELD node, score RETURN node.title AS title";
    let got = titles(&run(&w, cypher, vec![1.0, 0.0, 0.0]).await);
    assert_eq!(
        got,
        vec!["beta".to_string()],
        "k=1 + filter must over-fetch, not post-truncate to zero"
    );
}

#[tokio::test]
async fn vector_procedure_filter_equality_and_in() {
    let w = seed("vec-filter-shapes").await;
    // Equality: only gamma, regardless of its vector rank.
    let eq = "CALL search.vector({ label: 'Doc', property: 'embedding', \
         query: $q, k: 3, filter: { title: 'gamma' } }) \
         YIELD node, score RETURN node.title AS title";
    assert_eq!(
        titles(&run(&w, eq, vec![1.0, 0.0, 0.0]).await),
        vec!["gamma".to_string()]
    );
    // List value ⇒ IN: alpha + beta, in vector-rank order — gamma is excluded
    // even though it out-ranks beta by similarity.
    let isin = "CALL search.vector({ label: 'Doc', property: 'embedding', \
         query: $q, k: 3, filter: { title: ['alpha', 'beta'] } }) \
         YIELD node, score RETURN node.title AS title";
    assert_eq!(
        titles(&run(&w, isin, vec![1.0, 0.0, 0.0]).await),
        vec!["alpha".to_string(), "beta".to_string()],
        "IN filter keeps the lower-ranked beta and drops the higher-ranked gamma"
    );
}

#[tokio::test]
async fn hybrid_filter_applies_to_both_legs() {
    let w = seed("hybrid-filter").await;
    // Both legs active, but constrained to `gamma`. The dense leg filters via
    // over-fetch; the sparse ("quantum") leg also surfaces alpha, which the
    // fused-output filter must drop.
    let cypher = "CALL search.hybrid({ label: 'Doc', \
         query_text: 'quantum', text_property: 'body', \
         query_vector: $q, vector_property: 'embedding', \
         k: 3, filter: { title: 'gamma' } }) \
         YIELD node, score RETURN node.title AS title";
    assert_eq!(
        titles(&run(&w, cypher, vec![1.0, 0.0, 0.0]).await),
        vec!["gamma".to_string()],
        "filter drops non-matching nodes from both legs"
    );
}

#[tokio::test]
async fn vector_procedure_filter_in_via_param_map() {
    // The explicit `in` operator (a reserved keyword, unusable as a bare inline
    // map key) is reachable when the filter is supplied as a $param — its runtime
    // map keys are plain strings, never parsed.
    let w = seed("vec-filter-param-in").await;
    let cypher = "CALL search.vector({ label: 'Doc', property: 'embedding', \
         query: $q, k: 3, filter: $f }) \
         YIELD node, score RETURN node.title AS title";
    let snap = w.snapshot();
    let plan = lower(&parse(cypher).unwrap()).unwrap();
    let mut params = Params::new();
    params.insert("q".into(), RuntimeValue::Vector(vec![1.0, 0.0, 0.0]));
    // filter: { title: { in: ['alpha', 'gamma'] } }
    let mut inmap = BTreeMap::new();
    inmap.insert(
        "in".to_string(),
        RuntimeValue::List(vec![
            RuntimeValue::String("alpha".into()),
            RuntimeValue::String("gamma".into()),
        ]),
    );
    let mut filter = BTreeMap::new();
    filter.insert("title".to_string(), RuntimeValue::Map(inmap));
    params.insert("f".into(), RuntimeValue::Map(filter));
    let rows = execute(&plan, &snap, &params).await.unwrap();
    assert_eq!(
        titles(&rows),
        vec!["alpha".to_string(), "gamma".to_string()]
    );
}

#[tokio::test]
async fn vector_procedure_filter_null_value_rejected() {
    // `filter: { k: null }` would 3VL-match nothing silently; it must be rejected.
    let w = seed("vec-filter-null").await;
    let snap = w.snapshot();
    let plan = lower(
        &parse(
            "CALL search.vector({ label: 'Doc', property: 'embedding', query: $q, k: 3, \
             filter: { title: null } }) YIELD node, score RETURN node",
        )
        .unwrap(),
    )
    .unwrap();
    let mut params = Params::new();
    params.insert("q".into(), RuntimeValue::Vector(vec![1.0, 0.0, 0.0]));
    let err = execute(&plan, &snap, &params).await;
    assert!(err.is_err(), "a null filter value must be rejected");
}

#[tokio::test]
async fn hybrid_sparse_filter_does_not_starve_past_k_sparse() {
    // 12 `other`-tenant docs with a high-tf body rank above 1 `target`-tenant doc
    // whose body has tf=1 diluted by length, so `target` ranks last (position 13).
    // A sparse-only filtered hybrid with k=1 (k_sparse default = 8) would, without
    // fetching the full BM25 ranking, truncate to the top-8 (all `other`) and then
    // filter to zero. The fix fetches the complete ranking so `target` survives.
    let mut writer = WriterSession::open(store(), paths("hybrid-starve"))
        .await
        .unwrap();
    for i in 0..12 {
        let mut p = BTreeMap::new();
        p.insert("title".into(), CoreValue::Str(format!("other{i}")));
        p.insert("tenant".into(), CoreValue::Str("other".into()));
        p.insert(
            "body".into(),
            CoreValue::Str("alpha alpha alpha alpha".into()),
        );
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
    let mut tp = BTreeMap::new();
    tp.insert("title".into(), CoreValue::Str("target".into()));
    tp.insert("tenant".into(), CoreValue::Str("acme".into()));
    // tf=1 for "alpha", padded with non-matching terms so length normalization
    // pushes it below every high-tf `other` doc.
    tp.insert(
        "body".into(),
        CoreValue::Str("alpha w1 w2 w3 w4 w5 w6 w7 w8 w9 w10 w11".into()),
    );
    writer
        .upsert_node(
            "Doc",
            NodeId::new(),
            &NodeWriteRecord {
                properties: tp,
                schema_version: 1,
                ..Default::default()
            },
        )
        .unwrap();
    writer.commit_batch().await.unwrap();

    let cypher = "CALL search.hybrid({ label: 'Doc', \
         query_text: 'alpha', text_property: 'body', \
         k: 1, filter: { tenant: 'acme' } }) \
         YIELD node, score RETURN node.title AS title";
    let got = titles(&run(&writer, cypher, vec![]).await);
    assert_eq!(
        got,
        vec!["target".to_string()],
        "the filter-matching doc beyond k_sparse must not be starved"
    );
}

#[tokio::test]
async fn vector_procedure_filter_unknown_operator_errors() {
    let w = seed("vec-filter-bad").await;
    let snap = w.snapshot();
    let plan = lower(
        &parse(
            "CALL search.vector({ label: 'Doc', property: 'embedding', query: $q, k: 3, \
             filter: { title: { wat: 'x' } } }) YIELD node, score RETURN node",
        )
        .unwrap(),
    )
    .unwrap();
    let mut params = Params::new();
    params.insert("q".into(), RuntimeValue::Vector(vec![1.0, 0.0, 0.0]));
    let err = execute(&plan, &snap, &params).await;
    assert!(err.is_err(), "an unknown filter operator must be rejected");
}

#[tokio::test]
async fn hybrid_rejects_alpha_out_of_range() {
    let w = seed("hybrid-alpha").await;
    let snap = w.snapshot();
    let plan = lower(
        &parse(
            "CALL search.hybrid({ label: 'Doc', query_text: 'quantum', text_property: 'body', \
             fusion: 'linear', alpha: 1.5, k: 3 }) YIELD node, score RETURN node",
        )
        .unwrap(),
    )
    .unwrap();
    assert!(
        execute(&plan, &snap, &Params::new()).await.is_err(),
        "alpha outside [0,1] must error"
    );
}

#[tokio::test]
async fn hybrid_rejects_partial_dense_leg() {
    let w = seed("hybrid-partial").await;
    let snap = w.snapshot();
    // query_vector without vector_property → error, not a silently-disabled leg.
    let plan = lower(
        &parse(
            "CALL search.hybrid({ label: 'Doc', query_vector: [1.0, 0.0, 0.0], k: 3 }) \
             YIELD node, score RETURN node",
        )
        .unwrap(),
    )
    .unwrap();
    assert!(execute(&plan, &snap, &Params::new()).await.is_err());
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
