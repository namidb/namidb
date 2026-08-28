//! NDB-04: `RETURN DISTINCT <endpoint> [LIMIT n]` over a variable-length
//! expand must run as a visited-set BFS, not the exponential trail
//! enumeration. Parity is asserted against the per-seed (non-eligible)
//! shape on both the memtable and flushed routes, and the dense-layered
//! blowup test completing at all is the regression assertion — the
//! trail-enumerating executor holds ~30^6 frontier entries there.
//!
//! Graph (KNOWS, directed):
//!
//! ```text
//!   Alice ─▶ Bob ─▶ Carol ─▶ Dave ─▶ Alice   (cycle back to the seed)
//!     │        └─────▶ Dave ▲
//!     └───────▶ Carol ──────┘
//! ```
//!
//! From Alice with `*1..3`: Bob and Carol at hop 1, Dave at hop 2, and
//! Alice HERSELF at hop 3 (through the cycle) — the seed-as-endpoint case
//! the BFS must not lose by pre-visiting the seed.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use namidb_core::id::{NamespaceId, NodeId};
use namidb_core::value::Value as CoreValue;
use namidb_storage::{EdgeWriteRecord, NamespacePaths, NodeWriteRecord, WriterSession};
use object_store::memory::InMemory;
use object_store::ObjectStore;

use namidb_query::{
    execute, execute_with_limits, lower, optimize, parse, Params, RuntimeValue, StatsCatalog,
};

fn store() -> Arc<dyn ObjectStore> {
    Arc::new(InMemory::new())
}

fn paths(name: &str) -> NamespacePaths {
    NamespacePaths::new("tenants", NamespaceId::new(name).unwrap())
}

fn person(name: &str) -> NodeWriteRecord {
    let mut props: BTreeMap<String, CoreValue> = BTreeMap::new();
    props.insert("name".into(), CoreValue::Str(name.into()));
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

async fn build_graph(writer: &mut WriterSession) {
    let names = ["Alice", "Bob", "Carol", "Dave"];
    let ids: [NodeId; 4] = std::array::from_fn(|_| NodeId::new());
    for (id, name) in ids.iter().zip(names.iter()) {
        writer.upsert_node("Person", *id, &person(name)).unwrap();
    }
    let [alice, bob, carol, dave] = ids;
    for (src, dst) in [
        (alice, bob),
        (alice, carol),
        (bob, carol),
        (bob, dave),
        (carol, dave),
        (dave, alice),
    ] {
        writer.upsert_edge("KNOWS", src, dst, &edge()).unwrap();
    }
    writer.commit_batch().await.unwrap();
}

async fn names(writer: &WriterSession, q: &str, column: &str) -> Vec<String> {
    let snap = writer.snapshot();
    // Through the OPTIMIZER, exactly as the server executes — the optimized
    // plan shape is what the BFS eligibility must match (the ORDER BY
    // sandwich regression was invisible to `lower()`-only tests).
    let plan = optimize(lower(&parse(q).unwrap()).unwrap(), &StatsCatalog::default());
    let rows = execute(&plan, &snap, &Params::new()).await.unwrap();
    rows.iter()
        .map(|r| match r.get(column) {
            Some(RuntimeValue::String(s)) => s.clone(),
            other => panic!("{column} not a string: {other:?}"),
        })
        .collect()
}

async fn assert_parity(writer: &WriterSession) {
    // Eligible shape: endpoint-only DISTINCT (runs the BFS).
    let distinct = names(
        writer,
        "MATCH (a:Person {name: 'Alice'})-[:KNOWS*1..3]->(b) \
         RETURN DISTINCT b.name AS name",
        "name",
    )
    .await;
    let got: BTreeSet<String> = distinct.iter().cloned().collect();
    assert_eq!(
        got.len(),
        distinct.len(),
        "DISTINCT must not emit duplicates: {distinct:?}"
    );
    // Reference: the seed-referencing projection is NOT eligible (per-seed
    // rows through the ordinary executor); its endpoint column deduped by
    // the test is the ground truth.
    let reference = names(
        writer,
        "MATCH (a:Person {name: 'Alice'})-[:KNOWS*1..3]->(b) \
         RETURN DISTINCT a.name AS seed, b.name AS name",
        "name",
    )
    .await;
    let expected: BTreeSet<String> = reference.into_iter().collect();
    assert_eq!(got, expected, "BFS endpoints diverge from exhaustive route");
    // The cycle case: Alice reaches herself at hop 3.
    assert!(
        got.contains("Alice"),
        "seed must be reachable as its own endpoint through the cycle: {got:?}"
    );
    assert_eq!(got.len(), 4, "{got:?}");

    // Path multiplicity must SURVIVE without DISTINCT (the BFS must not
    // leak into non-distinct shapes).
    let all_paths = names(
        writer,
        "MATCH (a:Person {name: 'Alice'})-[:KNOWS*1..3]->(b) \
         RETURN b.name AS name",
        "name",
    )
    .await;
    assert!(
        all_paths.len() > got.len(),
        "non-distinct query lost path multiplicity: {all_paths:?}"
    );
}

#[tokio::test]
async fn distinct_endpoints_match_exhaustive_on_memtable_and_flushed_routes() {
    let mut writer = WriterSession::open(store(), paths("dist-endpoints"))
        .await
        .unwrap();
    build_graph(&mut writer).await;
    assert_parity(&writer).await;

    let schema = namidb_core::schema::SchemaBuilder::new()
        .label(namidb_core::schema::LabelDef {
            name: "Person".into(),
            properties: vec![namidb_core::schema::PropertyDef::new(
                "name",
                namidb_core::schema::DataType::Utf8,
                true,
            )
            .unwrap()],
        })
        .unwrap()
        .build();
    writer.flush(schema).await.unwrap();
    assert_parity(&writer).await;
}

#[tokio::test]
async fn distinct_endpoints_with_limit_are_exact() {
    let mut writer = WriterSession::open(store(), paths("dist-limit"))
        .await
        .unwrap();
    build_graph(&mut writer).await;

    let limited = names(
        &writer,
        "MATCH (a:Person {name: 'Alice'})-[:KNOWS*1..3]->(b) \
         RETURN DISTINCT b.name AS name LIMIT 2",
        "name",
    )
    .await;
    assert_eq!(limited.len(), 2, "LIMIT under-produced: {limited:?}");
    let unique: BTreeSet<&String> = limited.iter().collect();
    assert_eq!(unique.len(), 2, "LIMIT emitted duplicates: {limited:?}");

    // A LIMIT above the endpoint count returns everything.
    let all = names(
        &writer,
        "MATCH (a:Person {name: 'Alice'})-[:KNOWS*1..3]->(b) \
         RETURN DISTINCT b.name AS name LIMIT 100",
        "name",
    )
    .await;
    assert_eq!(all.len(), 4, "{all:?}");
}

/// Dense layered graph: s → L1(30) → ... → L4(30) → t, complete bipartite
/// between consecutive layers. Trail enumeration visits ~30^5 walks for
/// `*1..6` (an effective hang); the endpoint BFS touches each node once
/// per seed. Completing at all is the regression assertion.
#[tokio::test]
async fn distinct_endpoints_survive_dense_layered_blowup() {
    let mut writer = WriterSession::open(store(), paths("dist-blowup"))
        .await
        .unwrap();
    const WIDTH: usize = 30;
    const DEPTH: usize = 4;
    let s = NodeId::new();
    let t = NodeId::new();
    writer.upsert_node("Person", s, &person("s")).unwrap();
    writer.upsert_node("Person", t, &person("t")).unwrap();
    let layers: Vec<Vec<NodeId>> = (0..DEPTH)
        .map(|_| (0..WIDTH).map(|_| NodeId::new()).collect())
        .collect();
    for (li, layer) in layers.iter().enumerate() {
        for (ni, &id) in layer.iter().enumerate() {
            writer
                .upsert_node("Person", id, &person(&format!("l{li}n{ni}")))
                .unwrap();
        }
    }
    for &first in &layers[0] {
        writer.upsert_edge("KNOWS", s, first, &edge()).unwrap();
    }
    for w in layers.windows(2) {
        for &a in &w[0] {
            for &b in &w[1] {
                writer.upsert_edge("KNOWS", a, b, &edge()).unwrap();
            }
        }
    }
    for &last in &layers[DEPTH - 1] {
        writer.upsert_edge("KNOWS", last, t, &edge()).unwrap();
    }
    writer.commit_batch().await.unwrap();

    let endpoints = names(
        &writer,
        "MATCH (a:Person {name: 's'})-[:KNOWS*1..6]->(b) \
         RETURN DISTINCT b.name AS name",
        "name",
    )
    .await;
    // Every layer node plus t, each exactly once.
    assert_eq!(endpoints.len(), WIDTH * DEPTH + 1, "{}", endpoints.len());

    // And the LIMIT form stops early instead of exhausting the graph.
    let five = names(
        &writer,
        "MATCH (a:Person {name: 's'})-[:KNOWS*1..6]->(b) \
         RETURN DISTINCT b.name AS name LIMIT 5",
        "name",
    )
    .await;
    assert_eq!(five.len(), 5);

    // The bare-ORDER-BY lowering ("TopN keys under ProjectDistinct") — the
    // exact shape that hung the 2.2.0 smoke test because the BFS
    // eligibility only matched Project-directly-over-Expand. Must complete
    // AND come back value-sorted (dedup_rows is order-preserving now).
    let ordered = names(
        &writer,
        "MATCH (a:Person {name: 's'})-[:KNOWS*1..6]->(b) \
         RETURN DISTINCT b.name AS name ORDER BY name",
        "name",
    )
    .await;
    assert_eq!(ordered.len(), WIDTH * DEPTH + 1);
    let mut sorted = ordered.clone();
    sorted.sort();
    assert_eq!(ordered, sorted, "ORDER BY must survive the DISTINCT");
}

/// dedup_rows used to sort by fingerprint ("I10;" < "I2;"), so
/// `RETURN DISTINCT x ORDER BY x` returned fingerprint order — a
/// correctness bug independent of graphs. Now order-preserving.
#[tokio::test]
async fn distinct_order_by_returns_value_order() {
    let writer = WriterSession::open(store(), paths("dist-order"))
        .await
        .unwrap();
    let snap = writer.snapshot();
    let plan = optimize(
        lower(&parse("UNWIND [10, 2, 1, 20, 2] AS x RETURN DISTINCT x AS n ORDER BY n").unwrap())
            .unwrap(),
        &StatsCatalog::default(),
    );
    let rows = execute(&plan, &snap, &Params::new()).await.unwrap();
    let got: Vec<i64> = rows
        .iter()
        .map(|r| match r.get("n") {
            Some(RuntimeValue::Integer(n)) => *n,
            other => panic!("{other:?}"),
        })
        .collect();
    assert_eq!(got, vec![1, 2, 10, 20]);

    // Descending too.
    let plan = optimize(
        lower(
            &parse("UNWIND [10, 2, 1, 20, 2] AS x RETURN DISTINCT x AS n ORDER BY n DESC").unwrap(),
        )
        .unwrap(),
        &StatsCatalog::default(),
    );
    let rows = execute(&plan, &snap, &Params::new()).await.unwrap();
    let got: Vec<i64> = rows
        .iter()
        .map(|r| match r.get("n") {
            Some(RuntimeValue::Integer(n)) => *n,
            other => panic!("{other:?}"),
        })
        .collect();
    assert_eq!(got, vec![20, 10, 2, 1]);
}

/// A shape the BFS must NOT take (the projection reads the seed) over a
/// dense graph must hit the deadline / row cap INSIDE the traversal —
/// before these in-loop guards, one seed enumerated millions of walks with
/// no budget probe and the query hung past its 30 s budget.
#[tokio::test]
async fn ineligible_dense_expand_hits_budget_instead_of_hanging() {
    let mut writer = WriterSession::open(store(), paths("dist-budget"))
        .await
        .unwrap();
    // Complete digraph over 16 nodes: ~3.6M six-hop trails from one seed.
    let ids: Vec<NodeId> = (0..16).map(|_| NodeId::new()).collect();
    for (i, &id) in ids.iter().enumerate() {
        writer
            .upsert_node("Person", id, &person(&format!("n{i}")))
            .unwrap();
    }
    for &a in &ids {
        for &b in &ids {
            if a != b {
                writer.upsert_edge("KNOWS", a, b, &edge()).unwrap();
            }
        }
    }
    writer.commit_batch().await.unwrap();
    let snap = writer.snapshot();
    let plan = optimize(
        lower(
            &parse(
                "MATCH (a:Person {name: 'n0'})-[:KNOWS*1..6]->(b) \
                 RETURN DISTINCT a.name AS seed, b.name AS name",
            )
            .unwrap(),
        )
        .unwrap(),
        &StatsCatalog::default(),
    );

    // Row cap: must fire mid-seed, typed.
    let err = execute_with_limits(&plan, &snap, &Params::new(), None, Some(10_000))
        .await
        .expect_err("the dense expansion must hit the row cap");
    assert!(
        err.to_string().contains("row cap"),
        "expected a row-cap error, got: {err}"
    );

    // Deadline: 200 ms budget must abort the same expansion promptly.
    let deadline = std::time::Instant::now() + std::time::Duration::from_millis(200);
    let started = std::time::Instant::now();
    let err = execute_with_limits(&plan, &snap, &Params::new(), Some(deadline), None)
        .await
        .expect_err("the dense expansion must hit the deadline");
    assert!(
        err.to_string().contains("timeout"),
        "expected a timeout error, got: {err}"
    );
    assert!(
        started.elapsed() < std::time::Duration::from_secs(10),
        "deadline abort took {:?} — the in-loop guard is not firing",
        started.elapsed()
    );
}
