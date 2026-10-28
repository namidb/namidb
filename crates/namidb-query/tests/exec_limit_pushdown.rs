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
use namidb_core::schema::Schema;
use namidb_core::value::Value as CoreValue;
use namidb_storage::sst::predicates::ScanPredicate;
use namidb_storage::{
    EdgeWriteRecord, NamespacePaths, NodeWriteRecord, SessionCaches, SstCache, WriterSession,
};
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
        ..Default::default()
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

#[tokio::test]
async fn param_skip_limit_resolves_per_execution_on_one_plan() {
    // Lower ONCE, then execute the same plan with different params: the
    // window must follow the params. This proves the plan carries the
    // `$param` by name (so a cached plan is reused across parameter sets)
    // rather than baking a value at plan time.
    let mut writer = WriterSession::open(store(), paths("lp-param-reuse"))
        .await
        .unwrap();
    build_graph(&mut writer).await;
    let snapshot = writer.snapshot();

    let full = pairs(
        &run(
            "lp-param-full",
            &format!("MATCH (a:User)-[r:RATED]->(b:Movie) {PAIR_RETURN}"),
        )
        .await,
    );

    let q = parse(&format!(
        "MATCH (a:User)-[r:RATED]->(b:Movie) {PAIR_RETURN} SKIP $s LIMIT $l"
    ))
    .unwrap();
    let plan = lower(&q).unwrap();

    let mut p1 = Params::new();
    p1.insert("s".into(), RuntimeValue::Integer(1));
    p1.insert("l".into(), RuntimeValue::Integer(2));
    let win1 = pairs(&execute_flat_path(&plan, &snapshot, &p1).await.unwrap());
    assert_eq!(win1, &full[1..3], "first params -> rows [1,3)");

    let mut p2 = Params::new();
    p2.insert("s".into(), RuntimeValue::Integer(3));
    p2.insert("l".into(), RuntimeValue::Integer(2));
    let win2 = pairs(&execute_flat_path(&plan, &snapshot, &p2).await.unwrap());
    assert_eq!(
        win2,
        &full[3..5],
        "second params -> rows [3,5) on the SAME plan"
    );
}

#[tokio::test]
async fn param_limit_not_provided_errors() {
    let mut writer = WriterSession::open(store(), paths("lp-param-missing"))
        .await
        .unwrap();
    build_graph(&mut writer).await;
    let snapshot = writer.snapshot();
    let q = parse(&format!(
        "MATCH (a:User)-[r:RATED]->(b:Movie) {PAIR_RETURN} LIMIT $l"
    ))
    .unwrap();
    let plan = lower(&q).unwrap();
    let err = execute_flat_path(&plan, &snapshot, &Params::new())
        .await
        .unwrap_err();
    assert!(
        format!("{err}").contains("not provided"),
        "expected a missing-parameter error, got: {err}"
    );
}

/// A capped expansion must return exactly `min(limit, total)` rows even when
/// most edges are REJECTED inside the per-edge loop.
///
/// This is the oracle for bounding an expansion's hydration. The per-edge
/// loop has many `continue` paths — label mismatch, membership, the trail
/// rule, visited-set pruning, an absent node — so any scheme that hydrates
/// "the first N endpoints" and stops UNDERFILLS whenever an endpoint is
/// rejected: fewer rows than the user asked for, which is a wrong answer,
/// not a slow one.
///
/// The fixture makes rejection the common case: one hub with 400 neighbours
/// of which only every fourth carries the queried label, so satisfying
/// `LIMIT 25` requires walking ~100 edges and discarding 75 of them.
#[tokio::test]
async fn cap_never_underfills_when_most_edges_are_rejected() {
    let mut writer = WriterSession::open(store(), paths("cap-underfill"))
        .await
        .unwrap();
    let hub = NodeId::new();
    writer
        .upsert_node("Hub", hub, &node_with("hid", 0))
        .unwrap();

    // 400 neighbours; every 4th is a :Wanted, the rest are :Other.
    const NEIGHBOURS: i64 = 400;
    let mut wanted_total = 0i64;
    for i in 0..NEIGHBOURS {
        let id = NodeId::new();
        if i % 4 == 0 {
            writer
                .upsert_node("Wanted", id, &node_with("v", i))
                .unwrap();
            wanted_total += 1;
        } else {
            writer.upsert_node("Other", id, &node_with("v", i)).unwrap();
        }
        writer.upsert_edge("NEAR", hub, id, &edge()).unwrap();
    }
    writer.commit_batch().await.unwrap();
    let snapshot = writer.snapshot();

    let rows_for = |query: String| {
        let snapshot = &snapshot;
        async move {
            let q = parse(&query).unwrap();
            let plan = lower(&q).unwrap();
            execute_flat_path(&plan, snapshot, &Params::new())
                .await
                .unwrap()
        }
    };

    let baseline = rows_for("MATCH (h:Hub)-[:NEAR]->(t:Wanted) RETURN t.v AS v".to_string()).await;
    assert_eq!(
        baseline.len() as i64,
        wanted_total,
        "fixture sanity: every :Wanted neighbour matches"
    );

    // Every limit from under to over the available rows must be exact, and
    // must be an order-preserving PREFIX of the uncapped result.
    let full: Vec<i64> = baseline
        .iter()
        .map(|r| match r.get("v") {
            Some(RuntimeValue::Integer(v)) => *v,
            other => panic!("unexpected row shape: {other:?}"),
        })
        .collect();
    for limit in [1usize, 2, 25, 99, 100, 101, 400] {
        let rows = rows_for(format!(
            "MATCH (h:Hub)-[:NEAR]->(t:Wanted) RETURN t.v AS v LIMIT {limit}"
        ))
        .await;
        let expected = limit.min(full.len());
        assert_eq!(
            rows.len(),
            expected,
            "LIMIT {limit} must return exactly min(limit, {}) rows",
            full.len()
        );
        let got: Vec<i64> = rows
            .iter()
            .map(|r| match r.get("v") {
                Some(RuntimeValue::Integer(v)) => *v,
                other => panic!("unexpected row shape: {other:?}"),
            })
            .collect();
        assert_eq!(
            got,
            full[..expected].to_vec(),
            "LIMIT {limit} must be an order-preserving prefix of the uncapped result"
        );
    }
}

/// `MATCH (a)-->(b)` scans every node as a seed (Movies included, which
/// have no out-edges) and expands every edge type. The flush puts the nodes
/// in an SST so the capped runs take the seed pull, not the memtable path.
#[tokio::test]
async fn unlabelled_source_limit_over_untyped_expand_is_a_prefix() {
    let cache = SstCache::new(8 * 1024 * 1024);
    let mut caches = SessionCaches::none();
    caches.sst_cache = Some(cache.clone());
    let mut writer = WriterSession::open_with_caches(store(), paths("lp-untyped"), caches)
        .await
        .unwrap();
    build_graph(&mut writer).await;
    writer.flush(Schema::empty()).await.unwrap();
    let snapshot = writer.snapshot();
    let run_query = |query: String| {
        let snapshot = &snapshot;
        async move {
            let plan = lower(&parse(&query).unwrap()).unwrap();
            pairs(
                &execute_flat_path(&plan, snapshot, &Params::new())
                    .await
                    .unwrap(),
            )
        }
    };

    let full = run_query(format!("MATCH (a)-->(b) {PAIR_RETURN}")).await;
    assert_eq!(full.len(), 6);
    for limit in [1usize, 3, 6, 10] {
        let capped = run_query(format!("MATCH (a)-->(b) {PAIR_RETURN} LIMIT {limit}")).await;
        assert_eq!(capped, &full[..limit.min(full.len())], "LIMIT {limit}");
    }
    // One limited scan per capped query, each of the 9 nodes (fewer than
    // the first round of 64 seeds), and none fell back to a full scan.
    assert_eq!(cache.limited_node_scan_fallbacks(), 0);
    assert_eq!(cache.limited_node_scan_output_rows(), 4 * 9);
}

/// The seed pull grows 64, 256, 1024: 500 edgeless nodes ahead of the
/// connected ones force several rounds, and rows found across a round
/// boundary must keep the uncapped order. The flush puts the nodes in an
/// SST, where the pull runs in rounds (from the memtable it scans once).
#[tokio::test]
async fn limit_fills_past_many_edgeless_seeds() {
    let mut writer = WriterSession::open(store(), paths("lp-edgeless"))
        .await
        .unwrap();
    for i in 0..500 {
        writer
            .upsert_node("Lonely", NodeId::new(), &node_with("v", i))
            .unwrap();
    }
    let hub = NodeId::new();
    writer.upsert_node("Hub", hub, &node_with("v", -1)).unwrap();
    for i in 0..200 {
        let id = NodeId::new();
        writer
            .upsert_node("Linked", id, &node_with("v", 1000 + i))
            .unwrap();
        writer.upsert_edge("TO", id, hub, &edge()).unwrap();
    }
    writer.commit_batch().await.unwrap();
    writer.flush(Schema::empty()).await.unwrap();
    let snapshot = writer.snapshot();

    let values = |rows: Vec<Row>| -> Vec<i64> {
        rows.iter()
            .map(|r| match r.get("v") {
                Some(RuntimeValue::Integer(v)) => *v,
                other => panic!("unexpected row shape: {other:?}"),
            })
            .collect()
    };
    let run_query = |query: String| {
        let snapshot = &snapshot;
        async move {
            let plan = lower(&parse(&query).unwrap()).unwrap();
            execute_flat_path(&plan, snapshot, &Params::new())
                .await
                .unwrap()
        }
    };

    let full = values(run_query("MATCH (a)-[:TO]->(h) RETURN a.v AS v".to_string()).await);
    assert_eq!(full.len(), 200);
    for limit in [1usize, 2, 63, 64, 65, 150, 200, 300] {
        let got = values(
            run_query(format!(
                "MATCH (a)-[:TO]->(h) RETURN a.v AS v LIMIT {limit}"
            ))
            .await,
        );
        assert_eq!(got, &full[..limit.min(full.len())], "LIMIT {limit}");
    }
}

#[tokio::test]
async fn nodescan_limit_uses_prefix_on_multi_level_layout() {
    // Two flushes that rewrite some of the same random ids leave two node
    // SSTs whose key ranges overlap, the normal shape of a store with more
    // than one level. The scan used to give up on its LIMIT prefix there and
    // reconcile every node; it must now stop early and still return exactly
    // the uncapped prefix, with or without a WITH in between.
    let sst_cache = SstCache::new(64 * 1024 * 1024);
    let mut caches = SessionCaches::none();
    caches.sst_cache = Some(sst_cache.clone());
    let mut writer = WriterSession::open_with_caches(store(), paths("lp-multi-level"), caches)
        .await
        .unwrap();
    let ids: Vec<NodeId> = (0..40).map(|_| NodeId::new()).collect();
    for (i, id) in ids.iter().enumerate() {
        writer
            .upsert_node("L", *id, &node_with("k", i as i64))
            .unwrap();
    }
    writer.commit_batch().await.unwrap();
    writer.flush(Schema::empty()).await.unwrap();
    for (i, id) in ids.iter().enumerate().step_by(3) {
        writer
            .upsert_node("L", *id, &node_with("k", 100 + i as i64))
            .unwrap();
    }
    writer.commit_batch().await.unwrap();
    writer.flush(Schema::empty()).await.unwrap();

    let snapshot = writer.snapshot();
    assert!(
        snapshot.metadata_node_count(None).is_none(),
        "the two node SSTs must overlap"
    );
    let exec = |query: &str| {
        let plan = lower(&parse(query).unwrap()).unwrap();
        let snapshot = &snapshot;
        async move {
            execute_flat_path(&plan, snapshot, &Params::new())
                .await
                .unwrap()
        }
    };
    let full = exec("MATCH (n:L) RETURN keys(n) AS ks, n.k AS k").await;
    assert_eq!(full.len(), 40);
    let fast_paths = sst_cache.limited_node_scan_fast_paths();
    let fallbacks = sst_cache.limited_node_scan_fallbacks();
    for query in [
        "MATCH (n:L) RETURN keys(n) AS ks, n.k AS k LIMIT 2",
        "MATCH (n:L) WITH n LIMIT 2 RETURN keys(n) AS ks, n.k AS k",
    ] {
        assert_eq!(exec(query).await, full[..2].to_vec(), "{query}");
    }
    assert_eq!(sst_cache.limited_node_scan_fast_paths(), fast_paths + 2);
    assert_eq!(sst_cache.limited_node_scan_fallbacks(), fallbacks);
}

/// `MATCH (a)-->(b) RETURN ... LIMIT k` on two overlapping node SSTs plus
/// memtable rows, with updates, deletes and a relabel spread over all three.
/// The seed pull used to run only on disjoint SSTs, so this layout scanned
/// every node in full (on a 2.4M node store, enough to be killed for
/// memory). The unlabelled limited scan now merges the SSTs in id order and
/// stops early, and the rows must still be the uncapped prefix.
#[tokio::test]
async fn unlabelled_seed_pull_stops_early_on_overlapping_ssts() {
    let sst_cache = SstCache::new(64 * 1024 * 1024);
    let mut caches = SessionCaches::none();
    caches.sst_cache = Some(sst_cache.clone());
    let mut writer = WriterSession::open_with_caches(store(), paths("lp-overlap-seed"), caches)
        .await
        .unwrap();
    const N: usize = 400;
    let ids: Vec<NodeId> = (0..N).map(|_| NodeId::new()).collect();
    for (i, id) in ids.iter().enumerate() {
        writer
            .upsert_node("A", *id, &node_with("v", i as i64))
            .unwrap();
    }
    for i in 0..N {
        writer
            .upsert_edge("TO", ids[i], ids[(i + 1) % N], &edge())
            .unwrap();
    }
    writer.commit_batch().await.unwrap();
    writer.flush(Schema::empty()).await.unwrap();
    // Second SST: rewrites, deletes and a relabel of first-SST ids, plus
    // new nodes.
    for (i, id) in ids.iter().enumerate() {
        if i % 7 == 0 {
            writer.tombstone_node("A", *id).unwrap();
        } else if i % 5 == 0 {
            writer
                .upsert_node_with_labels(["B".to_string()], *id, &node_with("v", 5000 + i as i64))
                .unwrap();
        } else if i % 3 == 0 {
            writer
                .upsert_node("A", *id, &node_with("v", 10_000 + i as i64))
                .unwrap();
        }
    }
    let mut fresh = Vec::new();
    for i in 0..20 {
        let id = NodeId::new();
        fresh.push(id);
        let label = if i % 2 == 0 { "A" } else { "B" };
        writer
            .upsert_node(label, id, &node_with("v", 30_000 + i))
            .unwrap();
        writer.upsert_edge("TO", id, ids[1], &edge()).unwrap();
    }
    writer.commit_batch().await.unwrap();
    writer.flush(Schema::empty()).await.unwrap();
    // Memtable on top: more rewrites and deletes, some undoing the SST ones.
    for (i, id) in ids.iter().enumerate() {
        if i % 13 == 0 {
            writer.tombstone_node("A", *id).unwrap();
        } else if i % 11 == 0 {
            writer
                .upsert_node("A", *id, &node_with("v", 20_000 + i as i64))
                .unwrap();
        }
    }
    // Only the newest version of these carries `A`: a labelled scan that
    // settled on an older one would drop them.
    for (i, id) in fresh.iter().enumerate().skip(1).step_by(2) {
        writer
            .upsert_node_with_labels(["A".to_string()], *id, &node_with("v", 40_000 + i as i64))
            .unwrap();
    }
    writer.commit_batch().await.unwrap();

    let snapshot = writer.snapshot();
    assert!(
        !snapshot.node_scan_is_disjoint(),
        "the layout must overlap or the test proves nothing"
    );
    let values = |rows: Vec<Row>| -> Vec<(i64, i64)> {
        rows.iter()
            .map(|r| match (r.get("v"), r.get("w")) {
                (Some(RuntimeValue::Integer(v)), Some(RuntimeValue::Integer(w))) => (*v, *w),
                (Some(RuntimeValue::Integer(v)), None) => (*v, 0),
                other => panic!("unexpected row shape: {other:?}"),
            })
            .collect()
    };
    let run_query = |query: String| {
        let snapshot = &snapshot;
        async move {
            let plan = lower(&parse(&query).unwrap()).unwrap();
            values(
                execute_flat_path(&plan, snapshot, &Params::new())
                    .await
                    .unwrap(),
            )
        }
    };

    for (body, labelled) in [
        ("MATCH (a)-->(b) RETURN a.v AS v, b.v AS w", false),
        ("MATCH (a) RETURN a.v AS v", false),
        ("MATCH (a:A) RETURN a.v AS v", true),
        ("MATCH (a:A)-->(b) RETURN a.v AS v, b.v AS w", true),
        ("MATCH (a:B)-->(b) RETURN a.v AS v, b.v AS w", true),
    ] {
        let full = run_query(body.to_string()).await;
        assert!(full.len() > 40, "{body}: {}", full.len());
        for limit in [1usize, 5, 63, 64, 65, 300, 1000] {
            let fast_paths = sst_cache.limited_node_scan_fast_paths();
            let fallbacks = sst_cache.limited_node_scan_fallbacks();
            let output = sst_cache.limited_node_scan_output_rows();
            let got = run_query(format!("{body} LIMIT {limit}")).await;
            assert_eq!(got, &full[..limit.min(full.len())], "{body} LIMIT {limit}");
            assert_eq!(
                sst_cache.limited_node_scan_fallbacks(),
                fallbacks,
                "{body} LIMIT {limit} reconciled every node"
            );
            assert!(sst_cache.limited_node_scan_fast_paths() > fast_paths);
            if limit == 5 && !labelled {
                // One round of 64 seeds is plenty: nearly every node has an
                // out-edge.
                assert!(sst_cache.limited_node_scan_output_rows() - output <= 64);
            }
        }
    }

    // A bare labelled LIMIT takes the label postings. A filtered one on a
    // label as dense as `A`, or on no label, is left to the full reconcile:
    // the merge could only test the filter after point-reading each
    // candidate, which on a selective filter is one point read per node.
    let predicate = [ScanPredicate::IsNotNull {
        column: "v".to_string(),
    }];
    // So the seed pull does not run a full reconcile per round either.
    assert!(snapshot.node_scan_stops_early(&[]));
    assert!(!snapshot.node_scan_stops_early(&predicate));
    for label in [Some("A"), None] {
        let full = match label {
            Some(label) => snapshot
                .scan_label_with_predicates_and_projection(label, &predicate, None)
                .await
                .unwrap(),
            None => snapshot
                .scan_all_nodes_with_predicates_and_projection(&predicate, None)
                .await
                .unwrap(),
        };
        assert!(full.len() > 250, "{label:?}: {}", full.len());
        for limit in [1usize, 5, 300, 1000] {
            let fallbacks = sst_cache.limited_node_scan_fallbacks();
            let got = match label {
                Some(label) => snapshot
                    .scan_label_with_predicates_and_projection_limited(
                        label, &predicate, None, limit,
                    )
                    .await
                    .unwrap(),
                None => snapshot
                    .scan_all_nodes_with_predicates_and_projection_limited(&predicate, None, limit)
                    .await
                    .unwrap(),
            };
            assert_eq!(
                got,
                &full[..limit.min(full.len())],
                "{label:?} LIMIT {limit}"
            );
            assert_eq!(sst_cache.limited_node_scan_fallbacks(), fallbacks + 1);
        }
    }
}

/// When the seeds rarely expand, every round of the seed pull restarts the
/// overlap merge and point-reads its whole prefix, so growing the rounds to
/// the store size cost more point reads than the store has nodes. Past
/// 4,096 seeds the last round is one full scan, and the rows are unchanged.
#[tokio::test]
async fn overlapping_seed_pull_stops_growing_when_seeds_rarely_expand() {
    let sst_cache = SstCache::new(64 * 1024 * 1024);
    let mut caches = SessionCaches::none();
    caches.sst_cache = Some(sst_cache.clone());
    let mut writer = WriterSession::open_with_caches(store(), paths("lp-overlap-rare"), caches)
        .await
        .unwrap();
    const N: usize = 6000;
    let ids: Vec<NodeId> = (0..N).map(|_| NodeId::new()).collect();
    for (i, id) in ids.iter().enumerate() {
        writer
            .upsert_node("A", *id, &node_with("v", i as i64))
            .unwrap();
    }
    writer
        .upsert_edge("RARE", ids[N - 2], ids[N - 1], &edge())
        .unwrap();
    writer.commit_batch().await.unwrap();
    writer.flush(Schema::empty()).await.unwrap();
    // A second SST spanning the first one's id range makes them overlap.
    for i in [0, N / 2, N - 1] {
        writer
            .upsert_node("A", ids[i], &node_with("v", 10_000 + i as i64))
            .unwrap();
    }
    writer.commit_batch().await.unwrap();
    writer.flush(Schema::empty()).await.unwrap();

    let snapshot = writer.snapshot();
    assert!(!snapshot.node_scan_is_disjoint());
    let output = sst_cache.limited_node_scan_output_rows();
    let plan = lower(&parse("MATCH (a)-[:RARE]->(b) RETURN a.v AS v, b.v AS w LIMIT 10").unwrap())
        .unwrap();
    let rows = execute_flat_path(&plan, &snapshot, &Params::new())
        .await
        .unwrap();
    assert_eq!(rows.len(), 1);
    assert_eq!(
        rows[0].get("v"),
        Some(&RuntimeValue::Integer((N - 2) as i64))
    );
    assert_eq!(
        rows[0].get("w"),
        Some(&RuntimeValue::Integer(10_000 + (N - 1) as i64))
    );
    let point_read = sst_cache.limited_node_scan_output_rows() - output;
    assert!(
        point_read <= 64 + 256 + 1024 + 4096,
        "{point_read} seeds point-read"
    );
}
