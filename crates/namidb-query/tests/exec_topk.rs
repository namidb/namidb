//! Bounded top-k: the `TopN` heap fast path.
//!
//! `ORDER BY ... LIMIT k` with `k < n` keeps only the `k` best rows in a
//! max-heap (O(n log k) time, O(k) memory) instead of materialising and
//! sorting all `n` keyed rows. This is the hot path for KNN
//! (`ORDER BY cosine_similarity(...) DESC LIMIT k`). The result must be
//! identical to the full sort: correct rows, correct order, correct window.

use std::collections::BTreeMap;
use std::sync::Arc;

use namidb_core::id::{NamespaceId, NodeId};
use namidb_core::value::Value as CoreValue;
use namidb_storage::{NamespacePaths, NodeWriteRecord, WriterSession};
use object_store::memory::InMemory;
use object_store::ObjectStore;

use namidb_query::{execute_with_limits, lower, parse, Params, RuntimeValue};

fn store() -> Arc<dyn ObjectStore> {
    Arc::new(InMemory::new())
}

fn paths(name: &str) -> NamespacePaths {
    NamespacePaths::new("tenants", NamespaceId::new(name).unwrap())
}

/// Seed `n` `Person` nodes with distinct `rank` = `0..n`.
async fn seed_ranks(name: &str, n: i64) -> WriterSession {
    let mut w = WriterSession::open(store(), paths(name)).await.unwrap();
    for r in 0..n {
        let mut p = BTreeMap::new();
        p.insert("rank".into(), CoreValue::I64(r));
        w.upsert_node(
            "Person",
            NodeId::new(),
            &NodeWriteRecord {
                properties: p,
                schema_version: 1,
                ..Default::default()
            },
        )
        .unwrap();
    }
    w.commit_batch().await.unwrap();
    w
}

/// Run `query` and collect the integer `r` column in result order.
async fn ranks(writer: &WriterSession, query: &str) -> Vec<i64> {
    let snap = writer.snapshot();
    let plan = lower(&parse(query).unwrap()).unwrap();
    let rows = execute_with_limits(&plan, &snap, &Params::new(), None, None)
        .await
        .unwrap();
    rows.iter()
        .map(|row| match row.get("r") {
            Some(RuntimeValue::Integer(i)) => *i,
            other => panic!("expected an integer `r`, got {other:?}"),
        })
        .collect()
}

#[tokio::test]
async fn top_k_desc_returns_the_largest_in_order() {
    let w = seed_ranks("topk-desc", 50).await;
    // k = 5 << 50, so the bounded heap path runs.
    let got = ranks(
        &w,
        "MATCH (p:Person) RETURN p.rank AS r ORDER BY p.rank DESC LIMIT 5",
    )
    .await;
    assert_eq!(got, vec![49, 48, 47, 46, 45]);
}

#[tokio::test]
async fn top_k_asc_returns_the_smallest_in_order() {
    let w = seed_ranks("topk-asc", 50).await;
    let got = ranks(
        &w,
        "MATCH (p:Person) RETURN p.rank AS r ORDER BY p.rank ASC LIMIT 5",
    )
    .await;
    assert_eq!(got, vec![0, 1, 2, 3, 4]);
}

#[tokio::test]
async fn top_k_with_skip_takes_the_right_window() {
    let w = seed_ranks("topk-skip", 20).await;
    // k = skip + limit = 5 < 20, heap path. Desc order is 19,18,17,...; SKIP 2
    // LIMIT 3 yields 17,16,15.
    let got = ranks(
        &w,
        "MATCH (p:Person) RETURN p.rank AS r ORDER BY p.rank DESC SKIP 2 LIMIT 3",
    )
    .await;
    assert_eq!(got, vec![17, 16, 15]);
}

#[tokio::test]
async fn limit_at_or_above_cardinality_falls_back_and_is_complete() {
    let w = seed_ranks("topk-fallback", 6).await;
    // k = 100 >= 6, so the full-sort path runs; still correct and complete.
    let got = ranks(
        &w,
        "MATCH (p:Person) RETURN p.rank AS r ORDER BY p.rank DESC LIMIT 100",
    )
    .await;
    assert_eq!(got, vec![5, 4, 3, 2, 1, 0]);
}

#[tokio::test]
async fn limit_one_returns_the_single_extreme() {
    let w = seed_ranks("topk-one", 30).await;
    assert_eq!(
        ranks(
            &w,
            "MATCH (p:Person) RETURN p.rank AS r ORDER BY p.rank DESC LIMIT 1"
        )
        .await,
        vec![29]
    );
    assert_eq!(
        ranks(
            &w,
            "MATCH (p:Person) RETURN p.rank AS r ORDER BY p.rank ASC LIMIT 1"
        )
        .await,
        vec![0]
    );
}

#[tokio::test]
async fn ties_return_a_full_window_of_the_tied_value() {
    // Five nodes share rank 7; ORDER BY rank LIMIT 3 returns three of them.
    // Which three is unspecified under Cypher, but the heap's position
    // tiebreak makes it the same set the full sort would pick.
    let mut w = WriterSession::open(store(), paths("topk-ties"))
        .await
        .unwrap();
    for _ in 0..5 {
        let mut p = BTreeMap::new();
        p.insert("rank".into(), CoreValue::I64(7));
        w.upsert_node(
            "Person",
            NodeId::new(),
            &NodeWriteRecord {
                properties: p,
                schema_version: 1,
                ..Default::default()
            },
        )
        .unwrap();
    }
    w.commit_batch().await.unwrap();

    let got = ranks(
        &w,
        "MATCH (p:Person) RETURN p.rank AS r ORDER BY p.rank ASC LIMIT 3",
    )
    .await;
    assert_eq!(got, vec![7, 7, 7]);
}
