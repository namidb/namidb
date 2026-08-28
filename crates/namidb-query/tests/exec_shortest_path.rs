//! RFC-023: end-to-end execution tests for `shortestPath` and
//! `allShortestPaths`.
//!
//! The graph used here is the same 5-person KNOWS chain
//! `exec_match_expand` builds, deliberately shaped so several paths
//! exist between the same pair of endpoints:
//!
//! ```text
//!   Alice ──▶ Bob ──▶ Carol ──▶ Dave ──▶ Eve ──▶ Alice
//!     │                                        ▲
//!     └──────────────▶ Carol ──────────────────┘   (via Carol—shorter)
//! ```
//!
//! - Alice→Carol direct edge: 1 hop.
//! - Alice→Bob→Carol: 2 hops.
//! - Alice→Carol→Dave: 2 hops; Alice→Bob→Carol→Dave: 3 hops.

use std::collections::BTreeMap;
use std::sync::Arc;

use namidb_core::id::{NamespaceId, NodeId};
use namidb_core::value::Value as CoreValue;
use namidb_storage::{EdgeWriteRecord, NamespacePaths, NodeWriteRecord, WriterSession};
use object_store::memory::InMemory;
use object_store::ObjectStore;

use namidb_query::{execute, lower, parse, Params, RuntimeValue};

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

async fn build_graph(writer: &mut WriterSession) -> [NodeId; 5] {
    let names = ["Alice", "Bob", "Carol", "Dave", "Eve"];
    let ids: [NodeId; 5] = std::array::from_fn(|_| NodeId::new());
    for (id, name) in ids.iter().zip(names.iter()) {
        writer.upsert_node("Person", *id, &person(name)).unwrap();
    }
    let edges = [
        (ids[0], ids[1]), // Alice -> Bob
        (ids[0], ids[2]), // Alice -> Carol (direct shortcut)
        (ids[1], ids[2]), // Bob   -> Carol
        (ids[2], ids[3]), // Carol -> Dave
        (ids[3], ids[4]), // Dave  -> Eve
        (ids[4], ids[0]), // Eve   -> Alice (back-edge for variety)
    ];
    for (src, dst) in edges {
        writer.upsert_edge("KNOWS", src, dst, &edge()).unwrap();
    }
    writer.commit_batch().await.unwrap();
    ids
}

#[tokio::test]
async fn shortest_path_alice_to_carol_is_one_hop() {
    let mut writer = WriterSession::open(store(), paths("sp-1hop"))
        .await
        .unwrap();
    let _ = build_graph(&mut writer).await;
    let snapshot = writer.snapshot();

    let q = parse(
        "MATCH (a:Person {name: 'Alice'}), (b:Person {name: 'Carol'}) \
         MATCH p = shortestPath((a)-[:KNOWS*..5]-(b)) \
         RETURN length(p) AS hops",
    )
    .unwrap();
    let plan = lower(&q).unwrap();
    let rows = execute(&plan, &snapshot, &Params::new()).await.unwrap();
    assert_eq!(
        rows.len(),
        1,
        "exactly one shortestPath row, got {}",
        rows.len()
    );
    match rows[0].get("hops") {
        Some(RuntimeValue::Integer(n)) => assert_eq!(*n, 1, "Alice -> Carol direct"),
        other => panic!("hops not integer: {:?}", other),
    }
}

#[tokio::test]
async fn shortest_path_alice_to_dave_is_two_hops() {
    let mut writer = WriterSession::open(store(), paths("sp-2hop"))
        .await
        .unwrap();
    let _ = build_graph(&mut writer).await;
    let snapshot = writer.snapshot();

    let q = parse(
        "MATCH (a:Person {name: 'Alice'}), (b:Person {name: 'Dave'}) \
         MATCH p = shortestPath((a)-[:KNOWS*..5]-(b)) \
         RETURN length(p) AS hops",
    )
    .unwrap();
    let plan = lower(&q).unwrap();
    let rows = execute(&plan, &snapshot, &Params::new()).await.unwrap();
    assert_eq!(rows.len(), 1);
    match rows[0].get("hops") {
        Some(RuntimeValue::Integer(n)) => assert_eq!(*n, 2),
        other => panic!("hops not integer: {:?}", other),
    }
}

#[tokio::test]
async fn all_shortest_paths_alice_to_dave_returns_minimum_length_only() {
    let mut writer = WriterSession::open(store(), paths("sp-all")).await.unwrap();
    let _ = build_graph(&mut writer).await;
    let snapshot = writer.snapshot();

    let q = parse(
        "MATCH (a:Person {name: 'Alice'}), (b:Person {name: 'Dave'}) \
         MATCH p = allShortestPaths((a)-[:KNOWS*..5]-(b)) \
         RETURN length(p) AS hops",
    )
    .unwrap();
    let plan = lower(&q).unwrap();
    let rows = execute(&plan, &snapshot, &Params::new()).await.unwrap();

    // Two paths of length 2:
    //   Alice -> Carol -> Dave
    //   Alice -> Bob -> Carol -> Dave   (this is length 3 — must NOT be emitted)
    // So only the 2-hop ones survive.
    assert!(
        !rows.is_empty(),
        "allShortestPaths must return at least one row"
    );
    for r in &rows {
        match r.get("hops") {
            Some(RuntimeValue::Integer(n)) => assert_eq!(
                *n, 2,
                "allShortestPaths must only emit minimum-length paths"
            ),
            other => panic!("hops not integer: {:?}", other),
        }
    }
}

#[tokio::test]
async fn shortest_path_rejects_unbounded_star() {
    // Open-ended `*` now parses (capped for ordinary expands), but shortestPath
    // still requires a finite upper bound, so the lower must reject it.
    let q = parse(
        "MATCH (a:Person {name: 'Alice'}), (b:Person {name: 'Bob'}) \
         MATCH p = shortestPath((a)-[:KNOWS*]-(b)) \
         RETURN p",
    )
    .expect("open-ended `*` now parses");
    let err = lower(&q).expect_err("shortestPath requires a finite upper bound");
    assert!(
        err.message.to_lowercase().contains("finite")
            || err.message.to_lowercase().contains("bound"),
        "error should mention the finite-bound requirement, got: {}",
        err.message
    );
}

#[tokio::test]
async fn shortest_path_rejects_unbound_endpoint() {
    // `b` is not bound by a preceding MATCH and has no unique-key
    // filter, so the lower must reject the shortestPath call.
    let mut writer = WriterSession::open(store(), paths("sp-unbound"))
        .await
        .unwrap();
    let _ = build_graph(&mut writer).await;

    let q = parse(
        "MATCH (a:Person {name: 'Alice'}) \
         MATCH p = shortestPath((a)-[:KNOWS*..5]-(b:Person)) \
         RETURN p",
    )
    .unwrap();
    // `b` does have a binding (`b:Person`), but it's NEW — not in scope.
    // Until 2.2.0 this lowered silently and `First` mode returned ONE
    // arbitrary reachable node per seed instead of one row per (src, dst)
    // pair — an openCypher divergence that read as data loss in the field
    // (NDB-09). The lower now enforces the v0 previously-bound contract.
    let err = lower(&q).expect_err("unbound shortestPath endpoint must be rejected");
    assert!(
        err.message.contains("must be bound by a previous MATCH"),
        "error should name the previously-bound rule, got: {}",
        err.message
    );
    assert!(
        err.message.contains("`b`"),
        "error should name the offending endpoint, got: {}",
        err.message
    );
}

/// RFC-023 trail correctness: `nodes(p)` must carry the ACTUAL intermediate
/// nodes. The back-reference emission used to fill every hop with the
/// pre-bound endpoint value, so a 2-hop path returned ["n0", "n2", "n2"].
#[tokio::test]
async fn shortest_path_trail_carries_intermediate_nodes() {
    let mut writer = WriterSession::open(store(), paths("sp-trail"))
        .await
        .unwrap();
    let ids: Vec<NodeId> = (0..3).map(|_| NodeId::new()).collect();
    for (i, &id) in ids.iter().enumerate() {
        writer
            .upsert_node("Person", id, &person(&format!("n{i}")))
            .unwrap();
    }
    for pair in ids.windows(2) {
        writer
            .upsert_edge("KNOWS", pair[0], pair[1], &edge())
            .unwrap();
    }
    writer.commit_batch().await.unwrap();
    let snapshot = writer.snapshot();
    let q = parse(
        "MATCH (a:Person {name: 'n0'}), (b:Person {name: 'n2'}) \
         MATCH p = shortestPath((a)-[:KNOWS*..4]->(b)) \
         RETURN [n IN nodes(p) | n.name] AS names",
    )
    .unwrap();
    let rows = execute(&lower(&q).unwrap(), &snapshot, &Params::new())
        .await
        .unwrap();
    assert_eq!(rows.len(), 1);
    match rows[0].get("names") {
        Some(RuntimeValue::List(items)) => {
            let names: Vec<&str> = items
                .iter()
                .map(|v| match v {
                    RuntimeValue::String(s) => s.as_str(),
                    other => panic!("non-string name: {other:?}"),
                })
                .collect();
            assert_eq!(names, ["n0", "n1", "n2"]);
        }
        other => panic!("names not a list: {other:?}"),
    }
}

/// NDB-09 follow-up: many (src, dst) pairs sharing a source run ONE
/// multi-target BFS. 40 bound targets over the layered graph — the per-seed
/// executor redid the whole BFS 40 times; the grouped run answers each pair
/// exactly once with its shortest length.
#[tokio::test]
async fn multi_pair_shortest_paths_answer_every_bound_target() {
    let mut writer = WriterSession::open(store(), paths("sp-multipair"))
        .await
        .unwrap();
    const WIDTH: usize = 20;
    let s = NodeId::new();
    writer.upsert_node("Person", s, &person("s")).unwrap();
    let l0: Vec<NodeId> = (0..WIDTH).map(|_| NodeId::new()).collect();
    let l1: Vec<NodeId> = (0..WIDTH).map(|_| NodeId::new()).collect();
    for (ni, &id) in l0.iter().enumerate() {
        writer
            .upsert_node("Person", id, &person(&format!("a{ni}")))
            .unwrap();
    }
    for (ni, &id) in l1.iter().enumerate() {
        writer
            .upsert_node("Person", id, &person(&format!("b{ni}")))
            .unwrap();
    }
    for &a in &l0 {
        writer.upsert_edge("KNOWS", s, a, &edge()).unwrap();
        for &b in &l1 {
            writer.upsert_edge("KNOWS", a, b, &edge()).unwrap();
        }
    }
    writer.commit_batch().await.unwrap();
    let snapshot = writer.snapshot();

    // First mode: one row per (s, b*) pair, each at its shortest length (2).
    let q = parse(
        "MATCH (a:Person {name: 's'}) MATCH (b:Person) WHERE b.name STARTS WITH 'b' \
         MATCH p = shortestPath((a)-[:KNOWS*..5]->(b)) \
         RETURN b.name AS name, length(p) AS hops",
    )
    .unwrap();
    let rows = execute(&lower(&q).unwrap(), &snapshot, &Params::new())
        .await
        .unwrap();
    assert_eq!(rows.len(), WIDTH, "one row per bound target");
    for row in &rows {
        assert_eq!(
            row.get("hops"),
            Some(&RuntimeValue::Integer(2)),
            "{:?}",
            row.get("name")
        );
    }

    // All mode: every same-length arrival, per target — WIDTH distinct
    // 2-hop paths reach each b* (one through each a*).
    let q = parse(
        "MATCH (a:Person {name: 's'}) MATCH (b:Person {name: 'b0'}) \
         MATCH p = allShortestPaths((a)-[:KNOWS*..5]->(b)) \
         RETURN count(p) AS n",
    )
    .unwrap();
    let rows = execute(&lower(&q).unwrap(), &snapshot, &Params::new())
        .await
        .unwrap();
    assert_eq!(
        rows[0].get("n"),
        Some(&RuntimeValue::Integer(WIDTH as i64)),
        "allShortestPaths must keep every same-level arrival"
    );
}

/// The bidirectional route is ACTUALLY taken for single bound pairs — the
/// reachability discipline: result parity alone would pass identically on
/// the unidirectional route — and stays exact on odd lengths, undirected
/// patterns, and unreachable pairs.
#[tokio::test]
async fn single_pair_shortest_path_takes_the_bidirectional_route() {
    let mut writer = WriterSession::open(store(), paths("sp-bidi"))
        .await
        .unwrap();
    let _ = build_graph(&mut writer).await;
    // An isolated node: reachable by nothing.
    writer
        .upsert_node("Person", NodeId::new(), &person("Zoe"))
        .unwrap();
    writer.commit_batch().await.unwrap();
    let snapshot = writer.snapshot();
    let before = namidb_storage::route_telemetry::snapshot().shortest_bidirectional;

    // Odd length, directed: Bob -> Carol -> Dave -> Eve.
    let q = parse(
        "MATCH (a:Person {name: 'Bob'}), (b:Person {name: 'Eve'}) \
         MATCH p = shortestPath((a)-[:KNOWS*..5]->(b)) \
         RETURN length(p) AS hops",
    )
    .unwrap();
    let rows = execute(&lower(&q).unwrap(), &snapshot, &Params::new())
        .await
        .unwrap();
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].get("hops"), Some(&RuntimeValue::Integer(3)));

    // Undirected: Eve -- Alice -- Bob is 2 against the edge directions.
    let q = parse(
        "MATCH (a:Person {name: 'Eve'}), (b:Person {name: 'Bob'}) \
         MATCH p = shortestPath((a)-[:KNOWS*..5]-(b)) \
         RETURN length(p) AS hops",
    )
    .unwrap();
    let rows = execute(&lower(&q).unwrap(), &snapshot, &Params::new())
        .await
        .unwrap();
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].get("hops"), Some(&RuntimeValue::Integer(2)));

    // Unreachable: no rows, no error.
    let q = parse(
        "MATCH (a:Person {name: 'Alice'}), (b:Person {name: 'Zoe'}) \
         MATCH p = shortestPath((a)-[:KNOWS*..5]->(b)) \
         RETURN length(p) AS hops",
    )
    .unwrap();
    let rows = execute(&lower(&q).unwrap(), &snapshot, &Params::new())
        .await
        .unwrap();
    assert!(rows.is_empty(), "{rows:?}");

    let after = namidb_storage::route_telemetry::snapshot().shortest_bidirectional;
    assert!(
        after >= before + 3,
        "all three single-pair queries must take the bidirectional route \
         (before {before}, after {after})"
    );
}

/// Dense layered graph: s → L1(40) → L2(40) → L3(40) → L4(40) → L5(40) → t,
/// complete bipartite between consecutive layers. The walk-enumerating
/// frontier holds 40^k entries at hop k (~102M Row clones by hop 5 — an
/// effective hang); the BFS visited-set frontier holds at most 40. The test
/// completing at all is the regression assertion.
#[tokio::test]
async fn shortest_path_survives_dense_layered_blowup() {
    let mut writer = WriterSession::open(store(), paths("sp-blowup"))
        .await
        .unwrap();
    const WIDTH: usize = 40;
    const DEPTH: usize = 5;
    let s = NodeId::new();
    let t = NodeId::new();
    writer.upsert_node("Person", s, &person("s")).unwrap();
    writer.upsert_node("Person", t, &person("t")).unwrap();
    let layers: Vec<Vec<NodeId>> = (0..DEPTH)
        .map(|_| (0..WIDTH).map(|_| NodeId::new()).collect())
        .collect();
    for (li, layer) in layers.iter().enumerate() {
        for &id in layer {
            writer
                .upsert_node("Person", id, &person(&format!("l{li}")))
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
    let snapshot = writer.snapshot();

    let q = parse(
        "MATCH (a:Person {name: 's'}), (b:Person {name: 't'}) \
         MATCH p = shortestPath((a)-[:KNOWS*..8]->(b)) \
         RETURN length(p) AS hops",
    )
    .unwrap();
    let plan = lower(&q).unwrap();
    let rows = execute(&plan, &snapshot, &Params::new()).await.unwrap();
    assert_eq!(rows.len(), 1);
    match rows[0].get("hops") {
        Some(RuntimeValue::Integer(n)) => assert_eq!(*n, (DEPTH + 1) as i64),
        other => panic!("hops not integer: {:?}", other),
    }

    // NDB-09: `min >= 2` used to disable pruning entirely, reviving the
    // 40^hop frontier (an effective hang). The (node, hop) frontier dedup
    // bounds it at O(V) per level; completing at all is the regression
    // assertion, and the shortest length is unchanged (already >= 2).
    let q = parse(
        "MATCH (a:Person {name: 's'}), (b:Person {name: 't'}) \
         MATCH p = shortestPath((a)-[:KNOWS*2..8]->(b)) \
         RETURN length(p) AS hops",
    )
    .unwrap();
    let plan = lower(&q).unwrap();
    let rows = execute(&plan, &snapshot, &Params::new()).await.unwrap();
    assert_eq!(rows.len(), 1);
    match rows[0].get("hops") {
        Some(RuntimeValue::Integer(n)) => assert_eq!(*n, (DEPTH + 1) as i64),
        other => panic!("hops not integer: {:?}", other),
    }
}

/// Diamond a→{b,c}→d plus a longer a→e→f→d detour: allShortestPaths must
/// return exactly the two 2-hop paths — the visited-set pruning may not
/// collapse same-level arrivals (each is a distinct shortest path), and the
/// 3-hop detour may not leak in.
#[tokio::test]
async fn all_shortest_paths_keeps_every_same_level_arrival() {
    let mut writer = WriterSession::open(store(), paths("sp-diamond"))
        .await
        .unwrap();
    let names = ["a", "b", "c", "d", "e", "f"];
    let ids: [NodeId; 6] = std::array::from_fn(|_| NodeId::new());
    for (id, name) in ids.iter().zip(names.iter()) {
        writer.upsert_node("Person", *id, &person(name)).unwrap();
    }
    for (src, dst) in [
        (ids[0], ids[1]), // a→b
        (ids[0], ids[2]), // a→c
        (ids[1], ids[3]), // b→d
        (ids[2], ids[3]), // c→d
        (ids[0], ids[4]), // a→e
        (ids[4], ids[5]), // e→f
        (ids[5], ids[3]), // f→d
    ] {
        writer.upsert_edge("KNOWS", src, dst, &edge()).unwrap();
    }
    writer.commit_batch().await.unwrap();
    let snapshot = writer.snapshot();

    let q = parse(
        "MATCH (a:Person {name: 'a'}), (b:Person {name: 'd'}) \
         MATCH p = allShortestPaths((a)-[:KNOWS*..5]->(b)) \
         RETURN length(p) AS hops",
    )
    .unwrap();
    let plan = lower(&q).unwrap();
    let rows = execute(&plan, &snapshot, &Params::new()).await.unwrap();
    assert_eq!(rows.len(), 2, "both 2-hop shortest paths, not the detour");
    for r in &rows {
        match r.get("hops") {
            Some(RuntimeValue::Integer(n)) => assert_eq!(*n, 2),
            other => panic!("hops not integer: {:?}", other),
        }
    }
}
