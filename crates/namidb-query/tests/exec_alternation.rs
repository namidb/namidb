//! RFC-024 §"Open questions" Q1: end-to-end coverage for relationship
//! type alternation `[:A|:B]`.
//!
//! Two surfaces exercise alternation:
//!
//! - `LogicalPlan::Expand` directly (non-cyclic queries). The walker
//!   unions partner lists across the alternation set via
//!   `neighbours_of_any`, which iterates each listed type. These tests
//!   confirm the lowering no longer rejects `[:A|:B]` and that the
//!   executor produces every matching path with the correct binding
//!   semantics.
//!
//! - `LogicalPlan::MultiwayJoin.EdgeConstraint.edge_types`. The
//!   cyclic detection pass folds alternation into the constraint vector
//!   and the executor merges partner lists per constraint via
//!   `MergeSortedUnion` before the outer leapfrog intersection. These
//!   tests build cyclic queries that mix multiple edge types in the
//!   constraint graph and assert WCOJ-vs-binary parity.

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

/// Walk a plan and collect every `Expand`'s `edge_type` set so a test
/// can assert that alternation reached the lowered plan unchanged.
fn collect_expand_edge_types(plan: &LogicalPlan) -> Vec<Vec<String>> {
    let mut out = Vec::new();
    walk(plan, &mut out);
    out
}

fn walk(plan: &LogicalPlan, out: &mut Vec<Vec<String>>) {
    if let LogicalPlan::Expand { edge_type, .. } = plan {
        if let Some(v) = edge_type {
            out.push(v.clone());
        } else {
            out.push(Vec::new());
        }
    }
    for c in plan.children() {
        walk(c, out);
    }
}

// ─────────────────────── Expand-side alternation ────────────────────────

#[tokio::test]
async fn lowering_accepts_alternation_and_preserves_order() {
    // Pure lowering check — no executor. Earlier the lowering rejected
    // `[:A|:B]` with UnsupportedFeature; the new lowering hands the
    // types straight to `LogicalPlan::Expand.edge_type` as a Vec.
    let q = parse(
        "MATCH (a:Person)-[:KNOWS|:LIKES|:FOLLOWS]->(b:Person) RETURN a.name, b.name",
    )
    .unwrap();
    let cat = StatsCatalog::default();
    let plan = plan_query(&q, &cat).unwrap();
    let sets = collect_expand_edge_types(&plan);
    assert!(!sets.is_empty(), "lowered plan has no Expand: {:#?}", plan);
    assert_eq!(
        sets[0],
        vec!["KNOWS".to_string(), "LIKES".into(), "FOLLOWS".into()],
        "alternation set must preserve source order"
    );
}

#[tokio::test]
async fn expand_alternation_returns_rows_from_both_types() {
    // Graph:
    //   Alice -[:KNOWS]-> Bob
    //   Alice -[:LIKES]-> Carol
    //   Alice -[:KNOWS]-> Dave
    // `[:KNOWS|:LIKES]` from Alice must reach Bob, Carol, Dave.
    let mut w = WriterSession::open(store(), paths("alt-expand"))
        .await
        .unwrap();
    let alice = NodeId::new();
    let bob = NodeId::new();
    let carol = NodeId::new();
    let dave = NodeId::new();
    for (id, n) in [
        (alice, "Alice"),
        (bob, "Bob"),
        (carol, "Carol"),
        (dave, "Dave"),
    ] {
        w.upsert_node("Person", id, &person(n)).unwrap();
    }
    w.upsert_edge("KNOWS", alice, bob, &rel()).unwrap();
    w.upsert_edge("LIKES", alice, carol, &rel()).unwrap();
    w.upsert_edge("KNOWS", alice, dave, &rel()).unwrap();
    w.commit_batch().await.unwrap();
    let snap = w.snapshot();

    let q = parse(
        "MATCH (a:Person {name: 'Alice'})-[:KNOWS|:LIKES]->(b:Person) RETURN b.name AS name",
    )
    .unwrap();
    let cat = StatsCatalog::default();
    let plan = plan_query(&q, &cat).unwrap();
    let rows = execute(&plan, &snap, &Params::new()).await.unwrap();
    let mut names: Vec<String> = rows
        .iter()
        .map(|r| match r.get("name") {
            Some(RuntimeValue::String(s)) => s.clone(),
            other => panic!("name not a string: {:?}", other),
        })
        .collect();
    names.sort();
    assert_eq!(
        names,
        vec!["Bob".to_string(), "Carol".into(), "Dave".into()],
        "alternation must reach targets across both edge types"
    );
}

#[tokio::test]
async fn expand_alternation_unrelated_third_type_does_not_leak() {
    // Add a :HATES edge that the alternation does NOT mention, then
    // verify the result does not contain its target.
    let mut w = WriterSession::open(store(), paths("alt-expand-leak"))
        .await
        .unwrap();
    let alice = NodeId::new();
    let bob = NodeId::new();
    let eve = NodeId::new();
    for (id, n) in [(alice, "Alice"), (bob, "Bob"), (eve, "Eve")] {
        w.upsert_node("Person", id, &person(n)).unwrap();
    }
    w.upsert_edge("KNOWS", alice, bob, &rel()).unwrap();
    w.upsert_edge("HATES", alice, eve, &rel()).unwrap();
    w.commit_batch().await.unwrap();
    let snap = w.snapshot();

    let q = parse(
        "MATCH (a:Person {name: 'Alice'})-[:KNOWS|:LIKES]->(b:Person) RETURN b.name AS name",
    )
    .unwrap();
    let cat = StatsCatalog::default();
    let plan = plan_query(&q, &cat).unwrap();
    let rows = execute(&plan, &snap, &Params::new()).await.unwrap();
    let names: Vec<String> = rows
        .iter()
        .map(|r| match r.get("name") {
            Some(RuntimeValue::String(s)) => s.clone(),
            other => panic!("name not a string: {:?}", other),
        })
        .collect();
    assert_eq!(names, vec!["Bob".to_string()], "HATES must not appear");
}

#[tokio::test]
async fn expand_alternation_parallel_edges_yield_distinct_rows() {
    // If Alice→Bob has BOTH a KNOWS and a LIKES edge, the alternation
    // produces TWO rows. This matches Cypher's per-path semantics: a
    // RETURN without DISTINCT preserves multiplicities.
    let mut w = WriterSession::open(store(), paths("alt-expand-parallel"))
        .await
        .unwrap();
    let alice = NodeId::new();
    let bob = NodeId::new();
    for (id, n) in [(alice, "Alice"), (bob, "Bob")] {
        w.upsert_node("Person", id, &person(n)).unwrap();
    }
    w.upsert_edge("KNOWS", alice, bob, &rel()).unwrap();
    w.upsert_edge("LIKES", alice, bob, &rel()).unwrap();
    w.commit_batch().await.unwrap();
    let snap = w.snapshot();

    let q = parse(
        "MATCH (a:Person {name: 'Alice'})-[:KNOWS|:LIKES]->(b:Person) RETURN b.name AS name",
    )
    .unwrap();
    let cat = StatsCatalog::default();
    let plan = plan_query(&q, &cat).unwrap();
    let rows = execute(&plan, &snap, &Params::new()).await.unwrap();
    assert_eq!(
        rows.len(),
        2,
        "parallel KNOWS/LIKES edges produce one row each"
    );
}

#[tokio::test]
async fn expand_alternation_with_distinct_collapses_parallel_edges() {
    // Same fixture as the parallel-edges test, but RETURN DISTINCT
    // collapses the (Alice, Bob) pair to a single row.
    let mut w = WriterSession::open(store(), paths("alt-expand-distinct"))
        .await
        .unwrap();
    let alice = NodeId::new();
    let bob = NodeId::new();
    for (id, n) in [(alice, "Alice"), (bob, "Bob")] {
        w.upsert_node("Person", id, &person(n)).unwrap();
    }
    w.upsert_edge("KNOWS", alice, bob, &rel()).unwrap();
    w.upsert_edge("LIKES", alice, bob, &rel()).unwrap();
    w.commit_batch().await.unwrap();
    let snap = w.snapshot();

    let q = parse(
        "MATCH (a:Person {name: 'Alice'})-[:KNOWS|:LIKES]->(b:Person) \
         RETURN DISTINCT b.name AS name",
    )
    .unwrap();
    let cat = StatsCatalog::default();
    let plan = plan_query(&q, &cat).unwrap();
    let rows = execute(&plan, &snap, &Params::new()).await.unwrap();
    assert_eq!(rows.len(), 1, "DISTINCT collapses to one row");
}

#[tokio::test]
async fn expand_alternation_single_type_matches_legacy_path() {
    // `[:KNOWS]` (singleton alternation) must produce the same rows as
    // the pre-RFC `Some("KNOWS")` path did. Parity vs an equivalent
    // hand-written query is the easiest signal.
    let mut w = WriterSession::open(store(), paths("alt-singleton"))
        .await
        .unwrap();
    let alice = NodeId::new();
    let bob = NodeId::new();
    let carol = NodeId::new();
    for (id, n) in [(alice, "Alice"), (bob, "Bob"), (carol, "Carol")] {
        w.upsert_node("Person", id, &person(n)).unwrap();
    }
    w.upsert_edge("KNOWS", alice, bob, &rel()).unwrap();
    w.upsert_edge("KNOWS", alice, carol, &rel()).unwrap();
    w.commit_batch().await.unwrap();
    let snap = w.snapshot();

    let cat = StatsCatalog::default();
    let p_single = plan_query(
        &parse("MATCH (a:Person {name: 'Alice'})-[:KNOWS]->(b:Person) RETURN b.name AS n")
            .unwrap(),
        &cat,
    )
    .unwrap();
    let p_alt = plan_query(
        &parse(
            "MATCH (a:Person {name: 'Alice'})-[:KNOWS|:KNOWS]->(b:Person) RETURN b.name AS n",
        )
        .unwrap(),
        &cat,
    )
    .unwrap();

    let single_rows = execute(&p_single, &snap, &Params::new()).await.unwrap();
    let alt_rows = execute(&p_alt, &snap, &Params::new()).await.unwrap();
    // `[:KNOWS|:KNOWS]` should produce TWO rows per edge (one per listed
    // type) — that's the per-path semantic the executor follows.
    assert_eq!(single_rows.len(), 2);
    assert_eq!(alt_rows.len(), 4, "two types × two edges = four rows");
}

// ─────────────────────── MultiwayJoin alternation ───────────────────────

async fn build_mixed_triangle_graph(
    writer: &mut WriterSession,
) -> (
    [NodeId; 3], // triangle members
    NodeId,      // non-member d (linked but not closing a triangle)
) {
    let ids: [NodeId; 3] = std::array::from_fn(|_| NodeId::new());
    let d = NodeId::new();
    for (id, n) in ids.iter().zip(["A", "B", "C"].iter()) {
        writer.upsert_node("Person", *id, &person(n)).unwrap();
    }
    writer.upsert_node("Person", d, &person("D")).unwrap();

    // Triangle A→B→C→A using mixed types: A→B is KNOWS, B→C is
    // LIKES, C→A is FOLLOWS.
    writer.upsert_edge("KNOWS", ids[0], ids[1], &rel()).unwrap();
    writer.upsert_edge("LIKES", ids[1], ids[2], &rel()).unwrap();
    writer
        .upsert_edge("FOLLOWS", ids[2], ids[0], &rel())
        .unwrap();

    // D is connected to A by KNOWS but no closing edge — must be
    // pruned by the cycle.
    writer.upsert_edge("KNOWS", d, ids[0], &rel()).unwrap();

    writer.commit_batch().await.unwrap();
    (ids, d)
}

#[tokio::test]
async fn multiway_join_alternation_finds_triangle_across_types() {
    // The triangle uses three different edge types. The query lists
    // all three on each hop as `[:KNOWS|:LIKES|:FOLLOWS]`, so a binary
    // chain would have plenty of intermediate combinations to prune;
    // the WCOJ path should be more efficient (parity-checked).
    let _env = EnvGuard::set(&[("NAMIDB_WCOJ", "1"), ("NAMIDB_FACTORIZE", "1")]);

    let mut w = WriterSession::open(store(), paths("mwj-alt-triangle"))
        .await
        .unwrap();
    let (ids, _d) = build_mixed_triangle_graph(&mut w).await;
    let snap = w.snapshot();

    let q = parse(
        "MATCH (a:Person)-[:KNOWS|:LIKES|:FOLLOWS]->(b:Person)\
                       -[:KNOWS|:LIKES|:FOLLOWS]->(c:Person)\
                       -[:KNOWS|:LIKES|:FOLLOWS]->(a) \
         RETURN a, b, c",
    )
    .unwrap();
    let cat = StatsCatalog::default();
    let plan = plan_query(&q, &cat).unwrap();
    assert!(
        contains_multiway_join(&plan),
        "alternation triangle must rewrite to MultiwayJoin: {:#?}",
        plan
    );
    let rows = execute(&plan, &snap, &Params::new()).await.unwrap();

    // Exactly 3 rotations of (A, B, C).
    assert_eq!(rows.len(), 3, "expected 3 triangle rotations");
    let mut triples: Vec<(NodeId, NodeId, NodeId)> = rows
        .into_iter()
        .map(|r| (node_id(&r, "a"), node_id(&r, "b"), node_id(&r, "c")))
        .collect();
    triples.sort();
    let mut expected = vec![
        (ids[0], ids[1], ids[2]),
        (ids[1], ids[2], ids[0]),
        (ids[2], ids[0], ids[1]),
    ];
    expected.sort();
    assert_eq!(triples, expected);
}

#[tokio::test]
async fn multiway_join_alternation_parity_with_binary() {
    // Same triangle fixture, executed under WCOJ on/off. The row sets
    // (sorted by binding fingerprint) must match — alternation in
    // MultiwayJoin and alternation in Expand must agree.
    let mut w = WriterSession::open(store(), paths("mwj-alt-parity"))
        .await
        .unwrap();
    let _ = build_mixed_triangle_graph(&mut w).await;
    let snap = w.snapshot();

    let q = parse(
        "MATCH (a:Person)-[:KNOWS|:LIKES|:FOLLOWS]->(b:Person)\
                       -[:KNOWS|:LIKES|:FOLLOWS]->(c:Person)\
                       -[:KNOWS|:LIKES|:FOLLOWS]->(a) \
         RETURN a.name AS an, b.name AS bn, c.name AS cn",
    )
    .unwrap();
    let cat = StatsCatalog::default();

    let binary = {
        let _env = EnvGuard::set(&[("NAMIDB_WCOJ", "0")]);
        let p = plan_query(&q, &cat).unwrap();
        assert!(!contains_multiway_join(&p));
        execute(&p, &snap, &Params::new()).await.unwrap()
    };
    let wcoj = {
        let _env = EnvGuard::set(&[("NAMIDB_WCOJ", "1"), ("NAMIDB_FACTORIZE", "1")]);
        let p = plan_query(&q, &cat).unwrap();
        assert!(contains_multiway_join(&p));
        execute(&p, &snap, &Params::new()).await.unwrap()
    };

    assert_eq!(binary.len(), wcoj.len(), "row count diverges");
    let mut bk: Vec<String> = binary.iter().map(|r| format!("{:?}", r.bindings)).collect();
    let mut wk: Vec<String> = wcoj.iter().map(|r| format!("{:?}", r.bindings)).collect();
    bk.sort();
    wk.sort();
    assert_eq!(bk, wk, "WCOJ and binary diverge on alternation triangle");
}

#[tokio::test]
async fn multiway_join_mixed_single_and_alternation_edges() {
    // Hybrid constraints: one edge is single-typed, the others use
    // alternation. The detection pass and the executor must handle
    // mixed `edge_types.len() == 1` vs `> 1` in the same operator.
    let _env = EnvGuard::set(&[("NAMIDB_WCOJ", "1"), ("NAMIDB_FACTORIZE", "1")]);

    let mut w = WriterSession::open(store(), paths("mwj-alt-mixed"))
        .await
        .unwrap();
    let (ids, _) = build_mixed_triangle_graph(&mut w).await;
    let snap = w.snapshot();

    let q = parse(
        "MATCH (a:Person)-[:KNOWS]->(b:Person)\
                       -[:LIKES|:KNOWS]->(c:Person)\
                       -[:FOLLOWS|:LIKES]->(a) \
         RETURN a, b, c",
    )
    .unwrap();
    let cat = StatsCatalog::default();
    let plan = plan_query(&q, &cat).unwrap();
    assert!(contains_multiway_join(&plan));
    let rows = execute(&plan, &snap, &Params::new()).await.unwrap();
    assert_eq!(rows.len(), 1, "single rotation of the closed triangle");
    let r = &rows[0];
    assert_eq!(node_id(r, "a"), ids[0]);
    assert_eq!(node_id(r, "b"), ids[1]);
    assert_eq!(node_id(r, "c"), ids[2]);
}

#[tokio::test]
async fn multiway_join_alternation_with_missing_type_drops_rotation() {
    // If a rotation requires an edge type not in the alternation set,
    // it must NOT appear. Triangle is C→A FOLLOWS, but query asks for
    // [:LIKES] only on the closing hop, so the triangle is dropped.
    let _env = EnvGuard::set(&[("NAMIDB_WCOJ", "1"), ("NAMIDB_FACTORIZE", "1")]);

    let mut w = WriterSession::open(store(), paths("mwj-alt-missing"))
        .await
        .unwrap();
    let _ = build_mixed_triangle_graph(&mut w).await;
    let snap = w.snapshot();

    let q = parse(
        "MATCH (a:Person)-[:KNOWS]->(b:Person)\
                       -[:LIKES]->(c:Person)\
                       -[:LIKES|:KNOWS]->(a) \
         RETURN a.name",
    )
    .unwrap();
    let cat = StatsCatalog::default();
    let plan = plan_query(&q, &cat).unwrap();
    assert!(contains_multiway_join(&plan));
    let rows = execute(&plan, &snap, &Params::new()).await.unwrap();
    assert!(
        rows.is_empty(),
        "no triangle closes under [:LIKES|:KNOWS], got {} rows",
        rows.len()
    );
}

#[tokio::test]
async fn multiway_join_alternation_dense_graph_emits_set_semantics() {
    // Stress-test the `MergeSortedUnion` path: every pair of triangle
    // nodes has both a KNOWS and a LIKES edge, so each per-constraint
    // partner list is a unioned super-set. Important semantic note:
    // WCOJ binds *variables*, not paths, so for `RETURN a, b, c` the
    // executor emits one row per `(a, b, c)` tuple regardless of how
    // many edge-type combinations close the triangle. The binary
    // plan, by contrast, emits one row per path (2^3 = 8 type
    // pickings × 3 rotations = 24). The user who wants per-tuple
    // semantics on either path can wrap with `RETURN DISTINCT a, b, c`;
    // the user who wants per-path semantics must bind the rels (which
    // the v0 detection pass refuses anyway, so the plan falls back to
    // binary). RFC-024 §"Drawbacks" tracks the divergence.
    let _env = EnvGuard::set(&[("NAMIDB_WCOJ", "1"), ("NAMIDB_FACTORIZE", "1")]);

    let mut w = WriterSession::open(store(), paths("mwj-alt-dense"))
        .await
        .unwrap();
    let ids: [NodeId; 4] = std::array::from_fn(|_| NodeId::new());
    for (id, n) in ids.iter().zip(["A", "B", "C", "D"].iter()) {
        w.upsert_node("Person", *id, &person(n)).unwrap();
    }
    for (s, t) in [(ids[0], ids[1]), (ids[1], ids[2]), (ids[2], ids[0])] {
        w.upsert_edge("KNOWS", s, t, &rel()).unwrap();
        w.upsert_edge("LIKES", s, t, &rel()).unwrap();
    }
    // D is connected to A and B but never closes a triangle.
    w.upsert_edge("KNOWS", ids[3], ids[0], &rel()).unwrap();
    w.upsert_edge("LIKES", ids[3], ids[1], &rel()).unwrap();
    w.commit_batch().await.unwrap();
    let snap = w.snapshot();

    let q = parse(
        "MATCH (a:Person)-[:KNOWS|:LIKES]->(b:Person)\
                       -[:KNOWS|:LIKES]->(c:Person)\
                       -[:KNOWS|:LIKES]->(a) \
         RETURN a, b, c",
    )
    .unwrap();
    let cat = StatsCatalog::default();
    let plan = plan_query(&q, &cat).unwrap();
    assert!(contains_multiway_join(&plan));
    let rows = execute(&plan, &snap, &Params::new()).await.unwrap();

    // 3 rotations of (A,B,C); WCOJ emits one row per tuple regardless
    // of type-picking multiplicity.
    assert_eq!(rows.len(), 3, "expected 3 (a,b,c) tuples under WCOJ set semantics");

    // D must never appear in any binding.
    for r in &rows {
        for v in ["a", "b", "c"] {
            assert_ne!(node_id(r, v), ids[3], "D leaked into binding {}", v);
        }
    }

    // Binary path on the same query: per-path semantics gives 24 rows
    // (2^3 type pickings × 3 rotations). Confirms the divergence is
    // real and only on the WCOJ path.
    drop(_env);
    let _env_off = EnvGuard::set(&[("NAMIDB_WCOJ", "0")]);
    let p_bin = plan_query(&q, &cat).unwrap();
    assert!(!contains_multiway_join(&p_bin));
    let bin_rows = execute(&p_bin, &snap, &Params::new()).await.unwrap();
    assert_eq!(bin_rows.len(), 24, "binary path multiplies per type-picking");

    // The DISTINCT variant restores parity between the two paths.
    let q_distinct = parse(
        "MATCH (a:Person)-[:KNOWS|:LIKES]->(b:Person)\
                       -[:KNOWS|:LIKES]->(c:Person)\
                       -[:KNOWS|:LIKES]->(a) \
         RETURN DISTINCT a, b, c",
    )
    .unwrap();
    let p_dist = plan_query(&q_distinct, &cat).unwrap();
    let dist_rows = execute(&p_dist, &snap, &Params::new()).await.unwrap();
    assert_eq!(dist_rows.len(), 3, "DISTINCT collapses binary to set semantics");
}

fn node_id(row: &namidb_query::Row, alias: &str) -> NodeId {
    match row.get(alias) {
        Some(RuntimeValue::Node(n)) => n.id,
        other => panic!("{} not bound to a node: {:?}", alias, other),
    }
}
