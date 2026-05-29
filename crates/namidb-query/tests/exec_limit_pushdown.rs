//! LIMIT-pushdown correctness tests for the flat executor.
//!
//! A bare `LIMIT` (no `ORDER BY`) lowers to `TopN { keys: [] }`, whose
//! child is run under a row budget (`execute_capped`) so `Expand` /
//! `NodeScan` stop early instead of materialising their full output. These
//! tests pin the CORRECTNESS contract: the capped result must equal the
//! uncapped baseline truncated to the same window, the budget must never
//! under-fill (zero-edge source rows force more input than `limit`), and
//! it must be suppressed whenever an order-imposing / cardinality-altering
//! operator sits between the limit and the scan.

use std::collections::BTreeMap;
use std::sync::Arc;

use namidb_core::id::{NamespaceId, NodeId};
use namidb_core::value::Value as CoreValue;
use namidb_storage::{EdgeWriteRecord, NamespacePaths, NodeWriteRecord, WriterSession};
use object_store::memory::InMemory;
use object_store::ObjectStore;

use namidb_query::{execute_flat_path, lower, parse, Params, Row, RuntimeValue};

fn store() -> Arc<dyn ObjectStore> {
    Arc::new(InMemory::new())
}

fn paths(name: &str) -> NamespacePaths {
    NamespacePaths::new("tenants", NamespaceId::new(name).unwrap())
}

fn node_with(prop: &str, v: i64) -> NodeWriteRecord {
    let mut props: BTreeMap<String, CoreValue> = BTreeMap::new();
    props.insert(prop.into(), CoreValue::I64(v));
    NodeWriteRecord {
        properties: props,
        schema_version: 1,
    }
}

fn edge() -> EdgeWriteRecord {
    EdgeWriteRecord {
        properties: BTreeMap::new(),
        schema_version: 1,
    }
}

/// 6 Users, 3 Movies. Only U3/U4/U5 have RATED edges (U0/U1/U2 are
/// edge-less) → to gather K edges the Expand MUST consume more than K
/// source rows, exercising the "uncap the Expand's own input" rule. 6
/// RATED edges total: U3→{M0,M1}, U4→{M0,M1,M2}, U5→{M0}.
async fn build_graph(writer: &mut WriterSession) {
    let users: Vec<NodeId> = (0..6).map(|_| NodeId::new()).collect();
    for (i, id) in users.iter().enumerate() {
        writer
            .upsert_node("User", *id, &node_with("uid", i as i64))
            .unwrap();
    }
    let movies: Vec<NodeId> = (0..3).map(|_| NodeId::new()).collect();
    for (i, id) in movies.iter().enumerate() {
        writer
            .upsert_node("Movie", *id, &node_with("mid", i as i64))
            .unwrap();
    }
    let rated = [(3usize, 0usize), (3, 1), (4, 0), (4, 1), (4, 2), (5, 0)];
    for (u, m) in rated {
        writer
            .upsert_edge("RATED", users[u], movies[m], &edge())
            .unwrap();
    }
    writer.commit_batch().await.unwrap();
}

async fn run(name: &str, query: &str) -> Vec<Row> {
    let mut writer = WriterSession::open(store(), paths(name)).await.unwrap();
    build_graph(&mut writer).await;
    let snapshot = writer.snapshot();
    let q = parse(query).unwrap();
    let plan = lower(&q).unwrap();
    execute_flat_path(&plan, &snapshot, &Params::new())
        .await
        .unwrap()
}

/// Project the `(ua, mb)` pair columns into comparable tuples.
fn pairs(rows: &[Row]) -> Vec<(i64, i64)> {
    rows.iter()
        .map(|r| match (r.get("ua"), r.get("mb")) {
            (Some(RuntimeValue::Integer(a)), Some(RuntimeValue::Integer(b))) => (*a, *b),
            other => panic!("unexpected row shape: {other:?}"),
        })
        .collect()
}

const PAIR_RETURN: &str = "RETURN a.uid AS ua, b.mid AS mb";

#[tokio::test]
async fn capped_limit_is_a_correct_prefix_of_the_full_result() {
    // The capped LIMIT result must equal the uncapped baseline truncated
    // to the same length — the budget only stops early, it never reorders.
    let full = pairs(
        &run(
            "lp-prefix-full",
            &format!("MATCH (a:User)-[r:RATED]->(b:Movie) {PAIR_RETURN}"),
        )
        .await,
    );
    assert_eq!(full.len(), 6, "fixture has 6 RATED edges");

    let capped = pairs(
        &run(
            "lp-prefix-cap",
            &format!("MATCH (a:User)-[r:RATED]->(b:Movie) {PAIR_RETURN} LIMIT 3"),
        )
        .await,
    );
    assert_eq!(
        capped,
        &full[..3],
        "LIMIT 3 must be the first 3 baseline rows"
    );
}

#[tokio::test]
async fn cap_never_underfills_past_zero_edge_sources() {
    // U0/U1/U2 have no RATED edges. A naive cap on the NodeScan would scan
    // only `limit` users (some/all edge-less) and return < limit rows. The
    // correct design uncaps the Expand's input, so LIMIT 3 returns exactly 3.
    let rows = run(
        "lp-zero-edge",
        &format!("MATCH (a:User)-[r:RATED]->(b:Movie) {PAIR_RETURN} LIMIT 3"),
    )
    .await;
    assert_eq!(
        rows.len(),
        3,
        "must fill the limit despite leading zero-edge users"
    );
}

#[tokio::test]
async fn limit_zero_returns_no_rows() {
    let rows = run(
        "lp-zero",
        &format!("MATCH (a:User)-[r:RATED]->(b:Movie) {PAIR_RETURN} LIMIT 0"),
    )
    .await;
    assert!(
        rows.is_empty(),
        "LIMIT 0 must produce nothing, got {}",
        rows.len()
    );
}

#[tokio::test]
async fn skip_plus_limit_slices_the_baseline() {
    // cap = saturating(skip + limit); TopN still applies skip after, so the
    // window must match baseline[skip..skip+limit].
    let full = pairs(
        &run(
            "lp-skip-full",
            &format!("MATCH (a:User)-[r:RATED]->(b:Movie) {PAIR_RETURN}"),
        )
        .await,
    );
    let sliced = pairs(
        &run(
            "lp-skip-cap",
            &format!("MATCH (a:User)-[r:RATED]->(b:Movie) {PAIR_RETURN} SKIP 2 LIMIT 2"),
        )
        .await,
    );
    assert_eq!(sliced, &full[2..4]);
}

#[tokio::test]
async fn skip_beyond_available_is_empty_and_does_not_overflow() {
    let rows = run(
        "lp-skip-huge",
        &format!("MATCH (a:User)-[r:RATED]->(b:Movie) {PAIR_RETURN} SKIP 1000000 LIMIT 5"),
    )
    .await;
    assert!(rows.is_empty());
}

#[tokio::test]
async fn order_by_limit_is_globally_correct_not_capped() {
    // ORDER BY makes TopN.keys non-empty → the cap is suppressed and the
    // full input is sorted, so we get the GLOBAL top-2 by mid DESC, not a
    // scan-order prefix. Highest mid is M2 (only U4 rated it), then M1.
    let rows = run(
        "lp-order",
        &format!("MATCH (a:User)-[r:RATED]->(b:Movie) {PAIR_RETURN} ORDER BY b.mid DESC LIMIT 2"),
    )
    .await;
    let got = pairs(&rows);
    assert_eq!(got.len(), 2);
    assert_eq!(got[0].1, 2, "top mid must be 2 (global), got {got:?}");
    assert_eq!(got[1].1, 1, "second mid must be 1 (global), got {got:?}");
}

#[tokio::test]
async fn pure_nodescan_limit_returns_exactly_limit() {
    // NodeScan directly under TopN (no Expand shield): the cap is honoured
    // at the scan. 6 users exist; LIMIT 4 returns exactly 4.
    let rows = run("lp-scan", "MATCH (n:User) RETURN n.uid AS ua LIMIT 4").await;
    assert_eq!(rows.len(), 4);
}

#[tokio::test]
async fn multi_hop_expand_under_limit_is_correct() {
    // Two stacked Expands: only the outermost is capped; the inner runs
    // uncapped. Co-rated pairs (a)-[:RATED]->(b)<-[:RATED]-(c). Assert the
    // capped result is a prefix of the full baseline.
    let q = "MATCH (a:User)-[:RATED]->(b:Movie)<-[:RATED]-(c:User) RETURN a.uid AS ua, b.mid AS mb";
    let full_pairs = pairs(&run("lp-multihop-full2", q).await);
    let capped_pairs = pairs(&run("lp-multihop-cap", &format!("{q} LIMIT 3")).await);
    assert!(
        full_pairs.len() >= 3,
        "fixture should yield ≥3 co-rated rows"
    );
    assert_eq!(capped_pairs, &full_pairs[..3]);
}
