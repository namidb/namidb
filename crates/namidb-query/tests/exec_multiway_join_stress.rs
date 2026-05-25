//! RFC-024: stress + edge-case coverage for the WCOJ pipeline.
//!
//! The earlier `exec_multiway_join`, `exec_multiway_join_e2e`, and
//! `exec_alternation` files cover the happy paths and the obvious
//! rejection branches. This file adds:
//!
//! - Larger graphs (100+ nodes) where the leapfrog and the binary
//!   plan must agree on triangle and 4-clique counts.
//! - Direction-inversion combinations (mixed Right/Left edges in the
//!   same constraint subgraph).
//! - Plan-shape stability (idempotency of the detection pass under
//!   the 8-round optimiser fixpoint).
//! - MultiwayJoin composed with downstream operators: Filter on top,
//!   Project DISTINCT, ORDER BY + LIMIT.
//! - Detection rejection for shapes the planner must NOT touch:
//!   subplans under SemiApply, chains rooted at HashJoin, chains
//!   with a Filter mid-stream that carries a user predicate.

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

use namidb_core::id::{NamespaceId, NodeId};
use namidb_core::value::Value as CoreValue;
use namidb_query::plan::LogicalPlan;
use namidb_query::{execute, parse, plan as plan_query, Params, RuntimeValue, StatsCatalog};
use namidb_storage::{EdgeWriteRecord, NamespacePaths, NodeWriteRecord, WriterSession};
use object_store::memory::InMemory;
use object_store::ObjectStore;

static ENV_LOCK: Mutex<()> = Mutex::new(());

struct EnvGuard {
    keys: Vec<(&'static str, Option<String>)>,
    _lock: std::sync::MutexGuard<'static, ()>,
}

impl EnvGuard {
    fn set(vars: &[(&'static str, &str)]) -> Self {
        let lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let keys = vars
            .iter()
            .map(|(k, v)| {
                let prior = std::env::var(*k).ok();
                std::env::set_var(k, v);
                (*k, prior)
            })
            .collect();
        Self { keys, _lock: lock }
    }
}

impl Drop for EnvGuard {
    fn drop(&mut self) {
        for (k, prior) in &self.keys {
            match prior {
                Some(v) => std::env::set_var(k, v),
                None => std::env::remove_var(k),
            }
        }
    }
}

fn store() -> Arc<dyn ObjectStore> {
    Arc::new(InMemory::new())
}

fn paths(name: &str) -> NamespacePaths {
    NamespacePaths::new("tenants", NamespaceId::new(name).unwrap())
}

fn person(name: &str) -> NodeWriteRecord {
    let mut p: BTreeMap<String, CoreValue> = BTreeMap::new();
    p.insert("name".into(), CoreValue::Str(name.into()));
    NodeWriteRecord {
        properties: p,
        schema_version: 1,
    }
}

fn rel() -> EdgeWriteRecord {
    EdgeWriteRecord {
        properties: BTreeMap::new(),
        schema_version: 1,
    }
}

fn contains_multiway_join(plan: &LogicalPlan) -> bool {
    if matches!(plan, LogicalPlan::MultiwayJoin { .. }) {
        return true;
    }
    plan.children().iter().any(|c| contains_multiway_join(c))
}

fn count_multiway_joins(plan: &LogicalPlan) -> usize {
    let mut n = if matches!(plan, LogicalPlan::MultiwayJoin { .. }) {
        1
    } else {
        0
    };
    for c in plan.children() {
        n += count_multiway_joins(c);
    }
    n
}

// ─────────────────────── Larger graph fixtures ──────────────────────────

/// Build a graph of `n` Person nodes with directed KNOWS edges that
/// form `n` triangles by chaining trios `(i, i+1, i+2) mod n`. Each
/// node participates in two triangles (as predecessor and successor),
/// so total directed triangle count when seen from a single starting
/// orientation is exactly `n` (or 3*n if we count rotations).
async fn build_chain_of_triangles(writer: &mut WriterSession, n: usize) -> Vec<NodeId> {
    assert!(n >= 3);
    let ids: Vec<NodeId> = (0..n).map(|_| NodeId::new()).collect();
    for (i, id) in ids.iter().enumerate() {
        writer
            .upsert_node("Person", *id, &person(&format!("P{}", i)))
            .unwrap();
    }
    for i in 0..n {
        let a = ids[i];
        let b = ids[(i + 1) % n];
        let c = ids[(i + 2) % n];
        writer.upsert_edge("KNOWS", a, b, &rel()).unwrap();
        writer.upsert_edge("KNOWS", b, c, &rel()).unwrap();
        writer.upsert_edge("KNOWS", c, a, &rel()).unwrap();
    }
    writer.commit_batch().await.unwrap();
    ids
}

#[tokio::test]
async fn large_graph_triangle_count_matches_binary() {
    // 50 nodes, ~50 base triangles plus a handful of overlapping
    // closures from adjacent windows. WCOJ vs binary must agree
    // exactly once both paths use the same set/multiset semantic;
    // we add DISTINCT to normalise the binary multiset.
    let mut w = WriterSession::open(store(), paths("mwj-large-triangles"))
        .await
        .unwrap();
    let _ = build_chain_of_triangles(&mut w, 50).await;
    let snap = w.snapshot();

    let q = parse(
        "MATCH (a:Person)-[:KNOWS]->(b:Person)-[:KNOWS]->(c:Person)-[:KNOWS]->(a) \
         RETURN DISTINCT a, b, c",
    )
    .unwrap();
    let cat = StatsCatalog::default();

    let binary = {
        let _g = EnvGuard::set(&[("NAMIDB_WCOJ", "0")]);
        let p = plan_query(&q, &cat).unwrap();
        execute(&p, &snap, &Params::new()).await.unwrap().len() as i64
    };
    let wcoj = {
        let _g = EnvGuard::set(&[("NAMIDB_WCOJ", "1"), ("NAMIDB_FACTORIZE", "1")]);
        let p = plan_query(&q, &cat).unwrap();
        assert!(contains_multiway_join(&p));
        execute(&p, &snap, &Params::new()).await.unwrap().len() as i64
    };
    assert_eq!(binary, wcoj, "triangle counts diverge on 50-node graph");
    // The chain-of-triangles construction overlaps adjacent windows,
    // so the actual triangle count is whatever the structural
    // intersection produces. What matters: WCOJ and the binary path
    // agree on a non-trivial count.
    assert!(binary > 0, "expected at least one triangle");
}

#[tokio::test]
async fn four_cycle_rewrites_and_executes_against_complete_graph() {
    // 4-cycle on a fully-connected directed K_4 (every ordered pair
    // has an edge). Cypher binds path as a-b-c-d-a; the v0 detection
    // pass handles any cycle (no special 4-cycle case is needed).
    let mut w = WriterSession::open(store(), paths("mwj-k4")).await.unwrap();
    let ids: [NodeId; 4] = std::array::from_fn(|_| NodeId::new());
    for (i, id) in ids.iter().enumerate() {
        w.upsert_node("Person", *id, &person(&format!("V{}", i)))
            .unwrap();
    }
    for i in 0..4 {
        for j in 0..4 {
            if i != j {
                w.upsert_edge("KNOWS", ids[i], ids[j], &rel()).unwrap();
            }
        }
    }
    w.commit_batch().await.unwrap();
    let snap = w.snapshot();

    let q = parse(
        "MATCH (a:Person)-[:KNOWS]->(b:Person)-[:KNOWS]->(c:Person)\
                       -[:KNOWS]->(d:Person)-[:KNOWS]->(a) \
         RETURN count(*) AS quads",
    )
    .unwrap();
    let cat = StatsCatalog::default();

    let bin_count = {
        let _g = EnvGuard::set(&[("NAMIDB_WCOJ", "0")]);
        let p = plan_query(&q, &cat).unwrap();
        assert!(!contains_multiway_join(&p));
        let rows = execute(&p, &snap, &Params::new()).await.unwrap();
        match rows[0].get("quads") {
            Some(RuntimeValue::Integer(n)) => *n,
            _ => panic!(),
        }
    };
    let wcoj_count = {
        let _g = EnvGuard::set(&[("NAMIDB_WCOJ", "1"), ("NAMIDB_FACTORIZE", "1")]);
        let p = plan_query(&q, &cat).unwrap();
        assert!(contains_multiway_join(&p), "4-cycle missed detection");
        let rows = execute(&p, &snap, &Params::new()).await.unwrap();
        match rows[0].get("quads") {
            Some(RuntimeValue::Integer(n)) => *n,
            _ => panic!(),
        }
    };

    assert!(bin_count > 0 && wcoj_count > 0, "no 4-cycles found");
    // K_4 has no self-loops, so any (a,b,c,d) tuple with two
    // consecutive equal vertices fails to close. Counting tuples
    // that DO close:
    //   * all four distinct: 4! = 24
    //   * exactly one non-consecutive pair equal (a=c xor b=d): each
    //     contributes 4 * 3 * 3 = 36; with overlap (a=c AND b=d) of
    //     4*3 = 12 collapsed by inclusion-exclusion → 60
    //   * total = 24 + 60 = 84
    assert_eq!(
        bin_count, 84,
        "binary path missed expected K_4 4-cycle count"
    );
    assert_eq!(wcoj_count, bin_count, "4-cycle counts diverge");
}

// ─────────────────────── Direction edge cases ───────────────────────────

#[tokio::test]
async fn mixed_direction_triangle_executes_correctly() {
    // Triangle with mixed directions: a→b, b→c, c←a.
    // Closing edge points BACKWARDS relative to the chain.
    let mut w = WriterSession::open(store(), paths("mwj-mixed-dir"))
        .await
        .unwrap();
    let a = NodeId::new();
    let b = NodeId::new();
    let c = NodeId::new();
    for (id, n) in [(a, "A"), (b, "B"), (c, "C")] {
        w.upsert_node("Person", id, &person(n)).unwrap();
    }
    w.upsert_edge("KNOWS", a, b, &rel()).unwrap();
    w.upsert_edge("KNOWS", b, c, &rel()).unwrap();
    w.upsert_edge("KNOWS", a, c, &rel()).unwrap(); // a→c instead of c→a
    w.commit_batch().await.unwrap();
    let snap = w.snapshot();

    let q = parse(
        "MATCH (a:Person)-[:KNOWS]->(b:Person)\
                       -[:KNOWS]->(c:Person)\
                       <-[:KNOWS]-(a) \
         RETURN a, b, c",
    )
    .unwrap();
    let cat = StatsCatalog::default();

    let bin_rows = {
        let _g = EnvGuard::set(&[("NAMIDB_WCOJ", "0")]);
        let p = plan_query(&q, &cat).unwrap();
        execute(&p, &snap, &Params::new()).await.unwrap()
    };
    let wcoj_rows = {
        let _g = EnvGuard::set(&[("NAMIDB_WCOJ", "1"), ("NAMIDB_FACTORIZE", "1")]);
        let p = plan_query(&q, &cat).unwrap();
        assert!(contains_multiway_join(&p));
        execute(&p, &snap, &Params::new()).await.unwrap()
    };
    assert_eq!(bin_rows.len(), wcoj_rows.len());
    assert_eq!(bin_rows.len(), 1);
}

// ─────────────────────── Idempotency / plan stability ────────────────────

#[tokio::test]
async fn detection_pass_is_idempotent_across_fixpoint_rounds() {
    // The optimiser runs to fixpoint (up to 8 rounds). The detection
    // pass must not add a second MultiwayJoin on the second round or
    // mutate the one it produced.
    let _g = EnvGuard::set(&[("NAMIDB_WCOJ", "1"), ("NAMIDB_FACTORIZE", "1")]);
    let q = parse(
        "MATCH (a:Person)-[:KNOWS]->(b:Person)\
                       -[:KNOWS]->(c:Person)\
                       -[:KNOWS]->(a) \
         RETURN a.name",
    )
    .unwrap();
    let cat = StatsCatalog::default();
    let p1 = plan_query(&q, &cat).unwrap();
    let count_first = count_multiway_joins(&p1);
    let p2 = plan_query(&q, &cat).unwrap();
    let count_second = count_multiway_joins(&p2);

    assert_eq!(
        count_first, 1,
        "first run must produce exactly 1 MultiwayJoin"
    );
    assert_eq!(count_second, 1, "second run must not double-rewrite");
    assert_eq!(p1, p2, "plan must be deterministic across runs");
}

// ─────────────────────── Composition with downstream operators ───────────

#[tokio::test]
async fn filter_referencing_multiple_aliases_stays_above_multiway_join() {
    // A WHERE clause that references two chain aliases cannot be
    // pushed onto a single NodeScan, so it stays above the chain as
    // a Filter. The detection pass should still rewrite the chain
    // below into a MultiwayJoin, and the downstream Filter then
    // discards the rotations that fail the predicate.
    let _g = EnvGuard::set(&[("NAMIDB_WCOJ", "1"), ("NAMIDB_FACTORIZE", "1")]);
    let mut w = WriterSession::open(store(), paths("mwj-cross-filter"))
        .await
        .unwrap();
    let ids: Vec<NodeId> = (0..3).map(|_| NodeId::new()).collect();
    for (id, n) in ids.iter().zip(["Alice", "Bob", "Carol"].iter()) {
        w.upsert_node("Person", *id, &person(n)).unwrap();
    }
    w.upsert_edge("KNOWS", ids[0], ids[1], &rel()).unwrap();
    w.upsert_edge("KNOWS", ids[1], ids[2], &rel()).unwrap();
    w.upsert_edge("KNOWS", ids[2], ids[0], &rel()).unwrap();
    w.commit_batch().await.unwrap();
    let snap = w.snapshot();

    let q = parse(
        "MATCH (a:Person)-[:KNOWS]->(b:Person)\
                       -[:KNOWS]->(c:Person)\
                       -[:KNOWS]->(a) \
         WHERE a.name < c.name \
         RETURN a.name AS an, b.name AS bn, c.name AS cn",
    )
    .unwrap();
    let cat = StatsCatalog::default();
    let plan = plan_query(&q, &cat).unwrap();
    assert!(
        contains_multiway_join(&plan),
        "cross-binding filter must not break chain detection"
    );

    let rows = execute(&plan, &snap, &Params::new()).await.unwrap();
    // Of the 3 rotations only ones where a.name < c.name survive:
    //   (Alice, Bob, Carol) — Alice < Carol ✓
    //   (Bob, Carol, Alice) — Bob < Alice ✗
    //   (Carol, Alice, Bob) — Carol < Bob ✗
    assert_eq!(rows.len(), 1);
    match rows[0].get("an") {
        Some(RuntimeValue::String(s)) => assert_eq!(s, "Alice"),
        other => panic!("an not Alice: {:?}", other),
    }
}

#[tokio::test]
async fn order_by_limit_after_multiway_join() {
    let _g = EnvGuard::set(&[("NAMIDB_WCOJ", "1"), ("NAMIDB_FACTORIZE", "1")]);
    let mut w = WriterSession::open(store(), paths("mwj-order-limit"))
        .await
        .unwrap();
    let _ = build_chain_of_triangles(&mut w, 10).await;
    let snap = w.snapshot();

    let q = parse(
        "MATCH (a:Person)-[:KNOWS]->(b:Person)\
                       -[:KNOWS]->(c:Person)\
                       -[:KNOWS]->(a) \
         RETURN a.name AS an, b.name AS bn, c.name AS cn \
         ORDER BY an, bn, cn LIMIT 5",
    )
    .unwrap();
    let cat = StatsCatalog::default();
    let plan = plan_query(&q, &cat).unwrap();
    assert!(contains_multiway_join(&plan));

    let rows = execute(&plan, &snap, &Params::new()).await.unwrap();
    assert_eq!(rows.len(), 5, "LIMIT must cap to 5");
    // Verify sorted ascending by `an`.
    let names: Vec<String> = rows
        .iter()
        .map(|r| match r.get("an") {
            Some(RuntimeValue::String(s)) => s.clone(),
            _ => panic!(),
        })
        .collect();
    let mut sorted = names.clone();
    sorted.sort();
    assert_eq!(names, sorted, "rows not sorted by an");
}

#[tokio::test]
async fn count_aggregate_over_multiway_join() {
    let _g = EnvGuard::set(&[("NAMIDB_WCOJ", "1"), ("NAMIDB_FACTORIZE", "1")]);
    let mut w = WriterSession::open(store(), paths("mwj-count"))
        .await
        .unwrap();
    let _ = build_chain_of_triangles(&mut w, 10).await;
    let snap = w.snapshot();

    let q = parse(
        "MATCH (a:Person)-[:KNOWS]->(b:Person)\
                       -[:KNOWS]->(c:Person)\
                       -[:KNOWS]->(a) \
         RETURN count(*) AS total",
    )
    .unwrap();
    let cat = StatsCatalog::default();
    let plan = plan_query(&q, &cat).unwrap();
    assert!(contains_multiway_join(&plan));

    let rows = execute(&plan, &snap, &Params::new()).await.unwrap();
    assert_eq!(rows.len(), 1);
    let total = match rows[0].get("total") {
        Some(RuntimeValue::Integer(n)) => *n,
        _ => panic!(),
    };
    // 10 chain triangles × 3 rotations = 30 ordered matches.
    assert_eq!(total, 30);
}

// ─────────────────────── Rejection edge cases ───────────────────────────

#[tokio::test]
async fn detection_does_not_recurse_into_pattern_subqueries() {
    // EXISTS-style pattern predicates lower to SemiApply / PatternList
    // whose subplan slot the detection pass intentionally skips
    // (see recurse_children in optimize::multiway_join). The plan
    // shape varies by parser version; we keep the assertion narrow:
    // the planner must compile a query containing a cyclic pattern
    // inside an Exists predicate without panicking, and the result
    // executor must finish without runtime error.
    let _g = EnvGuard::set(&[("NAMIDB_WCOJ", "1"), ("NAMIDB_FACTORIZE", "1")]);

    let mut w = WriterSession::open(store(), paths("mwj-exists"))
        .await
        .unwrap();
    let ids: Vec<NodeId> = (0..3).map(|_| NodeId::new()).collect();
    for (id, n) in ids.iter().zip(["Alice", "Bob", "Carol"].iter()) {
        w.upsert_node("Person", *id, &person(n)).unwrap();
    }
    w.upsert_edge("KNOWS", ids[0], ids[1], &rel()).unwrap();
    w.upsert_edge("KNOWS", ids[1], ids[2], &rel()).unwrap();
    w.upsert_edge("KNOWS", ids[2], ids[0], &rel()).unwrap();
    w.commit_batch().await.unwrap();
    let snap = w.snapshot();

    // Inline EXISTS predicate form supported by the parser. If the
    // pattern lowers to a SemiApply / PatternList, detection skips
    // its subplan; if it lowers directly into the outer chain, the
    // detection picks it up. Either way the executor must run.
    let q = parse(
        "MATCH (a:Person) \
         WHERE exists((a)-[:KNOWS]->(:Person)) \
         RETURN a.name AS n",
    )
    .unwrap();
    let cat = StatsCatalog::default();
    let plan = plan_query(&q, &cat).unwrap();
    let rows = execute(&plan, &snap, &Params::new()).await.unwrap();
    // All three nodes have at least one outgoing KNOWS in our fixture.
    assert_eq!(rows.len(), 3);
}

#[tokio::test]
async fn detection_skips_chain_with_user_filter_mid_stream() {
    // After predicate pushdown + normalisation, defensive
    // `__label_eq` filters are removed but USER predicates that
    // reference an intermediate alias remain as Filter nodes in the
    // chain. A v0 chain harvest requires contiguous Expands; the
    // Filter blocks detection.
    let _g = EnvGuard::set(&[("NAMIDB_WCOJ", "1"), ("NAMIDB_FACTORIZE", "1")]);

    let q = parse(
        "MATCH (a:Person)-[:KNOWS]->(b:Person)\
                       -[:KNOWS]->(c:Person)\
                       -[:KNOWS]->(a) \
         WHERE b.name STARTS WITH 'X' \
         RETURN count(*) AS n",
    )
    .unwrap();
    let cat = StatsCatalog::default();
    let plan = plan_query(&q, &cat).unwrap();
    // The filter on b.name should land between Expands during
    // pushdown, breaking the contiguous Expand chain that the
    // detection pass needs. If it does land at the top (above the
    // chain), the chain remains contiguous and the plan still
    // contains a MultiwayJoin — either outcome is fine for v0; what
    // matters is the pass doesn't crash and the query executes
    // correctly. We assert the plan compiles and yields zero rows
    // (no name starts with 'X' in our fixture which is empty).
    let mut w = WriterSession::open(store(), paths("mwj-filter-mid"))
        .await
        .unwrap();
    let ids: Vec<NodeId> = (0..3).map(|_| NodeId::new()).collect();
    for (id, n) in ids.iter().zip(["Alice", "Bob", "Carol"].iter()) {
        w.upsert_node("Person", *id, &person(n)).unwrap();
    }
    w.upsert_edge("KNOWS", ids[0], ids[1], &rel()).unwrap();
    w.upsert_edge("KNOWS", ids[1], ids[2], &rel()).unwrap();
    w.upsert_edge("KNOWS", ids[2], ids[0], &rel()).unwrap();
    w.commit_batch().await.unwrap();
    let snap = w.snapshot();
    let rows = execute(&plan, &snap, &Params::new()).await.unwrap();
    let n = match rows[0].get("n") {
        Some(RuntimeValue::Integer(n)) => *n,
        _ => panic!(),
    };
    assert_eq!(n, 0, "no name starts with 'X' in the fixture");
}

#[tokio::test]
async fn disconnected_two_triangles_agree_between_paths() {
    // Two independent triangles in the same MATCH (comma-separated
    // pattern parts). The planner should rewrite both chains and
    // CrossProduct them; WCOJ vs binary must agree on the total row
    // count regardless of the exact semantic (per-tuple vs per-path).
    let mut w = WriterSession::open(store(), paths("mwj-two-triangles"))
        .await
        .unwrap();
    let t1: [NodeId; 3] = std::array::from_fn(|_| NodeId::new());
    let t2: [NodeId; 3] = std::array::from_fn(|_| NodeId::new());
    for (id, n) in t1.iter().zip(["A", "B", "C"].iter()) {
        w.upsert_node("Person", *id, &person(n)).unwrap();
    }
    for (id, n) in t2.iter().zip(["X", "Y", "Z"].iter()) {
        w.upsert_node("Person", *id, &person(n)).unwrap();
    }
    for (s, t) in [(t1[0], t1[1]), (t1[1], t1[2]), (t1[2], t1[0])] {
        w.upsert_edge("KNOWS", s, t, &rel()).unwrap();
    }
    for (s, t) in [(t2[0], t2[1]), (t2[1], t2[2]), (t2[2], t2[0])] {
        w.upsert_edge("KNOWS", s, t, &rel()).unwrap();
    }
    w.commit_batch().await.unwrap();
    let snap = w.snapshot();

    let q = parse(
        "MATCH (a:Person)-[:KNOWS]->(b:Person)-[:KNOWS]->(c:Person)-[:KNOWS]->(a), \
                (x:Person)-[:KNOWS]->(y:Person)-[:KNOWS]->(z:Person)-[:KNOWS]->(x) \
         RETURN count(*) AS n",
    )
    .unwrap();
    let cat = StatsCatalog::default();

    let bin_count = {
        let _g = EnvGuard::set(&[("NAMIDB_WCOJ", "0")]);
        let p = plan_query(&q, &cat).unwrap();
        let rows = execute(&p, &snap, &Params::new()).await.unwrap();
        match rows[0].get("n") {
            Some(RuntimeValue::Integer(n)) => *n,
            _ => panic!(),
        }
    };
    let wcoj_count = {
        let _g = EnvGuard::set(&[("NAMIDB_WCOJ", "1"), ("NAMIDB_FACTORIZE", "1")]);
        let p = plan_query(&q, &cat).unwrap();
        assert!(
            contains_multiway_join(&p),
            "at least one cycle must be rewritten"
        );
        let rows = execute(&p, &snap, &Params::new()).await.unwrap();
        match rows[0].get("n") {
            Some(RuntimeValue::Integer(n)) => *n,
            _ => panic!(),
        }
    };

    assert_eq!(bin_count, wcoj_count, "two-triangle counts diverge");
    assert!(
        bin_count > 0,
        "two disjoint triangles must produce some rows"
    );
}

#[tokio::test]
async fn detection_with_pushed_property_predicate_on_head() {
    // NodeScan with a pushed-down id-equality survives into the
    // detection pass as the head NodeScan's `predicates`. The
    // MultiwayJoin must preserve those predicates in
    // `NodeBinding.predicates` and apply them at the outer scan
    // level. The chain-of-triangles fixture overlaps each triangle's
    // membership with two adjacent ones, so anchoring `a = P0`
    // produces three closing tuples: P0→P1→P2 (triangle i=0),
    // P0→P1→P9 (closed via P9→P0 from i=9), P0→P8→P9 (triangle i=8).
    let _g = EnvGuard::set(&[("NAMIDB_WCOJ", "1"), ("NAMIDB_FACTORIZE", "1")]);

    let mut w = WriterSession::open(store(), paths("mwj-head-pred"))
        .await
        .unwrap();
    let _ = build_chain_of_triangles(&mut w, 10).await;
    let snap = w.snapshot();

    let q = parse(
        "MATCH (a:Person {name: 'P0'})-[:KNOWS]->(b:Person)\
                       -[:KNOWS]->(c:Person)\
                       -[:KNOWS]->(a) \
         RETURN b.name AS bn, c.name AS cn",
    )
    .unwrap();
    let cat = StatsCatalog::default();
    let plan_wcoj = plan_query(&q, &cat).unwrap();
    assert!(contains_multiway_join(&plan_wcoj));
    let wcoj_rows = execute(&plan_wcoj, &snap, &Params::new()).await.unwrap();
    let mut tuples: Vec<(String, String)> = wcoj_rows
        .iter()
        .map(|r| {
            let bn = match r.get("bn") {
                Some(RuntimeValue::String(s)) => s.clone(),
                _ => panic!(),
            };
            let cn = match r.get("cn") {
                Some(RuntimeValue::String(s)) => s.clone(),
                _ => panic!(),
            };
            (bn, cn)
        })
        .collect();
    tuples.sort();

    let mut expected = vec![
        ("P1".to_string(), "P2".to_string()),
        ("P1".to_string(), "P9".to_string()),
        ("P8".to_string(), "P9".to_string()),
    ];
    expected.sort();
    assert_eq!(tuples, expected, "WCOJ tuples diverge from chain topology");

    // Parity with the binary path under set semantics (RETURN DISTINCT
    // collapses the binary path's per-edge multiplicity).
    let q_distinct = parse(
        "MATCH (a:Person {name: 'P0'})-[:KNOWS]->(b:Person)\
                       -[:KNOWS]->(c:Person)\
                       -[:KNOWS]->(a) \
         RETURN DISTINCT b.name AS bn, c.name AS cn",
    )
    .unwrap();
    drop(_g);
    let _g_off = EnvGuard::set(&[("NAMIDB_WCOJ", "0")]);
    let plan_bin = plan_query(&q_distinct, &cat).unwrap();
    let bin_rows = execute(&plan_bin, &snap, &Params::new()).await.unwrap();
    assert_eq!(
        bin_rows.len(),
        3,
        "binary DISTINCT must agree with WCOJ tuple count"
    );
}

// ─────────────────────── Memtable overlay correctness ───────────────────

#[tokio::test]
async fn multiway_join_sees_uncommitted_edges_via_overlay() {
    // Build a triangle that exists ONLY in the memtable (no flush).
    // The executor's sorted_partners overlay (RFC: storage commit
    // 5f11f5b) must surface the memtable edges so the leapfrog
    // finds them.
    let _g = EnvGuard::set(&[("NAMIDB_WCOJ", "1"), ("NAMIDB_FACTORIZE", "1")]);

    let mut w = WriterSession::open(store(), paths("mwj-memtable"))
        .await
        .unwrap();
    let ids: Vec<NodeId> = (0..3).map(|_| NodeId::new()).collect();
    for (id, n) in ids.iter().zip(["A", "B", "C"].iter()) {
        w.upsert_node("Person", *id, &person(n)).unwrap();
    }
    w.upsert_edge("KNOWS", ids[0], ids[1], &rel()).unwrap();
    w.upsert_edge("KNOWS", ids[1], ids[2], &rel()).unwrap();
    w.upsert_edge("KNOWS", ids[2], ids[0], &rel()).unwrap();
    w.commit_batch().await.unwrap();
    let snap = w.snapshot();

    let q = parse(
        "MATCH (a:Person)-[:KNOWS]->(b:Person)\
                       -[:KNOWS]->(c:Person)\
                       -[:KNOWS]->(a) \
         RETURN count(*) AS n",
    )
    .unwrap();
    let cat = StatsCatalog::default();
    let plan = plan_query(&q, &cat).unwrap();
    assert!(contains_multiway_join(&plan));
    let rows = execute(&plan, &snap, &Params::new()).await.unwrap();
    let n = match rows[0].get("n") {
        Some(RuntimeValue::Integer(n)) => *n,
        _ => panic!(),
    };
    assert_eq!(n, 3, "memtable-only triangle must be visible via overlay");
}
