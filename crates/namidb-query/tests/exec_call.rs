//! End-to-end integration tests for `CALL algo.<name>() YIELD …` (RFC-008
//! PR1): parse → lower → execute against an in-memory storage namespace.
//!
//! `algo.wcc` and `algo.pagerank` run over the full snapshot via the
//! Snapshot→`algo::Graph` bridge in the executor. The graph below has two
//! connected pairs plus one isolated node, so WCC must report three
//! components (exercising the isolate-handling fix) and PageRank must score
//! every node.

use std::collections::BTreeMap;
use std::sync::Arc;

use namidb_core::id::{NamespaceId, NodeId};
use namidb_storage::{EdgeWriteRecord, NamespacePaths, NodeWriteRecord, WriterSession};
use object_store::memory::InMemory;
use object_store::ObjectStore;

use namidb_query::{execute, lower, optimize, parse, Params, RuntimeValue, StatsCatalog};

fn store() -> Arc<dyn ObjectStore> {
    Arc::new(InMemory::new())
}

fn paths(name: &str) -> NamespacePaths {
    NamespacePaths::new("tenants", NamespaceId::new(name).unwrap())
}

fn node() -> NodeWriteRecord {
    NodeWriteRecord {
        properties: BTreeMap::new(),
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

/// Two disjoint pairs (a-b, c-d) plus one isolated node (e) → 3 WCC
/// components; 5 nodes for PageRank.
async fn build(writer: &mut WriterSession) -> [NodeId; 5] {
    let ids: [NodeId; 5] = std::array::from_fn(|_| NodeId::new());
    for id in &ids {
        writer.upsert_node("N", *id, &node()).unwrap();
    }
    writer.upsert_edge("E", ids[0], ids[1], &edge()).unwrap();
    writer.upsert_edge("E", ids[2], ids[3], &edge()).unwrap();
    // ids[4] is intentionally isolated.
    writer.commit_batch().await.unwrap();
    ids
}

async fn run(snapshot: &namidb_storage::Snapshot<'_>, cypher: &str) -> Vec<namidb_query::Row> {
    let q = parse(cypher).unwrap_or_else(|e| panic!("parse {cypher}: {e:?}"));
    let plan = lower(&q).unwrap_or_else(|e| panic!("lower: {e:?}"));
    let plan = optimize(plan, &StatsCatalog::empty());
    execute(&plan, snapshot, &Params::new())
        .await
        .unwrap_or_else(|e| panic!("execute: {e}"))
}

#[tokio::test]
async fn call_wcc_yields_three_components_including_isolate() {
    let mut writer = WriterSession::open(store(), paths("call-wcc"))
        .await
        .unwrap();
    let ids = build(&mut writer).await;
    let snapshot = writer.snapshot();

    let rows = run(&snapshot, "CALL algo.wcc() YIELD node_id, component").await;
    assert_eq!(rows.len(), 5, "one row per node, including the isolate");

    // Map each node id → its component id.
    let mut by_node: BTreeMap<[u8; 16], i64> = BTreeMap::new();
    for r in &rows {
        let nid = match r.get("node_id") {
            Some(RuntimeValue::Node(n)) => *n.id.as_bytes(),
            other => panic!("node_id not a node: {other:?}"),
        };
        let comp = match r.get("component") {
            Some(RuntimeValue::Integer(c)) => *c,
            other => panic!("component not an int: {other:?}"),
        };
        by_node.insert(nid, comp);
    }
    let distinct: BTreeMap<i64, ()> = by_node.values().map(|c| (*c, ())).collect();
    assert_eq!(distinct.len(), 3, "two pairs + one isolate = 3 components");

    // Each pair's two members share a component.
    assert_eq!(
        by_node.get(ids[0].as_bytes()),
        by_node.get(ids[1].as_bytes()),
        "pair a-b shares a component"
    );
    assert_eq!(
        by_node.get(ids[2].as_bytes()),
        by_node.get(ids[3].as_bytes()),
        "pair c-d shares a component"
    );
    // The isolate's component differs from both pairs.
    let iso = by_node.get(ids[4].as_bytes()).copied().unwrap();
    assert_ne!(iso, *by_node.get(ids[0].as_bytes()).unwrap());
    assert_ne!(iso, *by_node.get(ids[2].as_bytes()).unwrap());
}

#[tokio::test]
async fn call_pagerank_scores_sum_to_one_and_cover_all_nodes() {
    let mut writer = WriterSession::open(store(), paths("call-pr"))
        .await
        .unwrap();
    build(&mut writer).await;
    let snapshot = writer.snapshot();

    let rows = run(&snapshot, "CALL algo.pagerank() YIELD node_id, score").await;
    assert_eq!(rows.len(), 5, "one row per node");

    let sum: f64 = rows
        .iter()
        .map(|r| match r.get("score") {
            Some(RuntimeValue::Float(s)) => *s,
            other => panic!("score not a float: {other:?}"),
        })
        .sum();
    assert!(
        (sum - 1.0).abs() < 1e-6,
        "PageRank scores conserve mass (sum ≈ 1.0), got {sum}"
    );
    assert!(rows.iter().all(|r| matches!(
        r.get("score"),
        Some(RuntimeValue::Float(s)) if *s >= 0.0
    )));
}

#[tokio::test]
async fn call_unknown_procedure_is_unsupported() {
    let mut writer = WriterSession::open(store(), paths("call-unknown"))
        .await
        .unwrap();
    build(&mut writer).await;
    let snapshot = writer.snapshot();

    let q = parse("CALL algo.bogus() YIELD x").unwrap();
    let plan = lower(&q).unwrap();
    let plan = optimize(plan, &StatsCatalog::empty());
    let err = execute(&plan, &snapshot, &Params::new()).await.unwrap_err();
    assert!(
        err.is_unsupported(),
        "unknown procedure should surface as an unsupported error, got {err}"
    );
}

#[tokio::test]
async fn call_pagerank_accepts_options_map() {
    let mut writer = WriterSession::open(store(), paths("call-pr-opts"))
        .await
        .unwrap();
    build(&mut writer).await;
    let snapshot = writer.snapshot();

    // A map argument overrides defaults; omitted keys keep them.
    let rows = run(
        &snapshot,
        "CALL algo.pagerank({damping: 0.9, max_iterations: 50, tolerance: 1e-6}) YIELD node_id, score",
    )
    .await;
    assert_eq!(rows.len(), 5, "one row per node");
    let sum: f64 = rows
        .iter()
        .map(|r| match r.get("score") {
            Some(RuntimeValue::Float(s)) => *s,
            _ => 0.0,
        })
        .sum();
    assert!(
        (sum - 1.0).abs() < 1e-6,
        "scores still sum to ~1.0, got {sum}"
    );
}

#[tokio::test]
async fn call_wcc_rejects_arguments() {
    let mut writer = WriterSession::open(store(), paths("call-wcc-args"))
        .await
        .unwrap();
    build(&mut writer).await;
    let snapshot = writer.snapshot();

    let q = parse("CALL algo.wcc({}) YIELD node_id, component").unwrap();
    let plan = lower(&q).unwrap();
    let plan = optimize(plan, &StatsCatalog::empty());
    let err = execute(&plan, &snapshot, &Params::new()).await.unwrap_err();
    assert!(
        err.is_unsupported(),
        "algo.wcc takes no arguments — should be unsupported, got {err}"
    );
}

/// A directed triangle a→b→c→a (one SCC, one undirected triangle) plus a node
/// d→a that is its own SCC and a pendant off the triangle. Returns [a, b, c, d].
async fn build_directed(writer: &mut WriterSession) -> [NodeId; 4] {
    let ids: [NodeId; 4] = std::array::from_fn(|_| NodeId::new());
    for id in &ids {
        writer.upsert_node("N", *id, &node()).unwrap();
    }
    writer.upsert_edge("E", ids[0], ids[1], &edge()).unwrap(); // a→b
    writer.upsert_edge("E", ids[1], ids[2], &edge()).unwrap(); // b→c
    writer.upsert_edge("E", ids[2], ids[0], &edge()).unwrap(); // c→a
    writer.upsert_edge("E", ids[3], ids[0], &edge()).unwrap(); // d→a
    writer.commit_batch().await.unwrap();
    ids
}

fn node_of(r: &namidb_query::Row) -> [u8; 16] {
    match r.get("node_id") {
        Some(RuntimeValue::Node(n)) => *n.id.as_bytes(),
        other => panic!("node_id not a node: {other:?}"),
    }
}

fn int_col(r: &namidb_query::Row, col: &str) -> i64 {
    match r.get(col) {
        Some(RuntimeValue::Integer(v)) => *v,
        other => panic!("{col} not an int: {other:?}"),
    }
}

#[tokio::test]
async fn call_degree_reports_in_out_total() {
    let mut writer = WriterSession::open(store(), paths("call-degree"))
        .await
        .unwrap();
    let ids = build(&mut writer).await; // a→b, c→d, isolate e
    let snapshot = writer.snapshot();

    let rows = run(
        &snapshot,
        "CALL algo.degree() YIELD node_id, in_degree, out_degree, degree",
    )
    .await;
    assert_eq!(rows.len(), 5, "one row per node");

    let find = |id: &NodeId| {
        rows.iter()
            .find(|r| node_of(r) == *id.as_bytes())
            .expect("node present")
    };
    let a = find(&ids[0]);
    assert_eq!(int_col(a, "out_degree"), 1);
    assert_eq!(int_col(a, "in_degree"), 0);
    assert_eq!(int_col(a, "degree"), 1);
    let b = find(&ids[1]);
    assert_eq!(int_col(b, "in_degree"), 1);
    assert_eq!(int_col(b, "out_degree"), 0);
    // The isolate e has degree 0 on every axis.
    let e = find(&ids[4]);
    assert_eq!(int_col(e, "degree"), 0);
    assert_eq!(int_col(e, "in_degree"), 0);
    assert_eq!(int_col(e, "out_degree"), 0);
}

#[tokio::test]
async fn call_scc_separates_cycle_from_bridge_node() {
    let mut writer = WriterSession::open(store(), paths("call-scc"))
        .await
        .unwrap();
    let ids = build_directed(&mut writer).await;
    let snapshot = writer.snapshot();

    let rows = run(&snapshot, "CALL algo.scc() YIELD node_id, component").await;
    assert_eq!(rows.len(), 4);

    let mut by_node: BTreeMap<[u8; 16], i64> = BTreeMap::new();
    for r in &rows {
        by_node.insert(node_of(r), int_col(r, "component"));
    }
    // a, b, c are one strongly connected component; d is its own.
    assert_eq!(by_node[ids[0].as_bytes()], by_node[ids[1].as_bytes()]);
    assert_eq!(by_node[ids[1].as_bytes()], by_node[ids[2].as_bytes()]);
    assert_ne!(by_node[ids[0].as_bytes()], by_node[ids[3].as_bytes()]);
    let distinct: BTreeMap<i64, ()> = by_node.values().map(|c| (*c, ())).collect();
    assert_eq!(distinct.len(), 2);
}

#[tokio::test]
async fn call_triangle_count_finds_the_triangle() {
    let mut writer = WriterSession::open(store(), paths("call-tri"))
        .await
        .unwrap();
    let ids = build_directed(&mut writer).await; // a-b-c triangle + d pendant
    let snapshot = writer.snapshot();

    let rows = run(
        &snapshot,
        "CALL algo.triangle_count() YIELD node_id, triangles, coefficient",
    )
    .await;
    assert_eq!(rows.len(), 4);

    let mut tri: BTreeMap<[u8; 16], i64> = BTreeMap::new();
    for r in &rows {
        tri.insert(node_of(r), int_col(r, "triangles"));
    }
    assert_eq!(tri[ids[0].as_bytes()], 1);
    assert_eq!(tri[ids[1].as_bytes()], 1);
    assert_eq!(tri[ids[2].as_bytes()], 1);
    assert_eq!(tri[ids[3].as_bytes()], 0, "d is a pendant, in no triangle");
}

#[tokio::test]
async fn call_label_propagation_groups_each_pair() {
    let mut writer = WriterSession::open(store(), paths("call-lpa"))
        .await
        .unwrap();
    let ids = build(&mut writer).await; // a-b, c-d, isolate e
    let snapshot = writer.snapshot();

    let rows = run(
        &snapshot,
        "CALL algo.label_propagation() YIELD node_id, community",
    )
    .await;
    assert_eq!(rows.len(), 5);

    let mut comm: BTreeMap<[u8; 16], i64> = BTreeMap::new();
    for r in &rows {
        comm.insert(node_of(r), int_col(r, "community"));
    }
    // Each connected pair is one community; the isolate is its own.
    assert_eq!(comm[ids[0].as_bytes()], comm[ids[1].as_bytes()]);
    assert_eq!(comm[ids[2].as_bytes()], comm[ids[3].as_bytes()]);
    let distinct: BTreeMap<i64, ()> = comm.values().map(|c| (*c, ())).collect();
    assert_eq!(distinct.len(), 3);
}

#[tokio::test]
async fn call_shortest_path_from_source() {
    let mut writer = WriterSession::open(store(), paths("call-sp"))
        .await
        .unwrap();
    let ids = build_directed(&mut writer).await; // a→b→c→a, d→a
    let snapshot = writer.snapshot();

    // From a: a=0, b=1, c=2 hops (a→b→c); d is unreachable (only d→a exists).
    let cypher = format!(
        "CALL algo.shortest_path({{source: \"{}\"}}) YIELD node_id, distance, hops",
        ids[0]
    );
    let rows = run(&snapshot, &cypher).await;

    let mut hops: BTreeMap<[u8; 16], i64> = BTreeMap::new();
    for r in &rows {
        hops.insert(node_of(r), int_col(r, "hops"));
    }
    assert_eq!(hops[ids[0].as_bytes()], 0, "source is 0 hops");
    assert_eq!(hops[ids[1].as_bytes()], 1, "b is one hop");
    assert_eq!(hops[ids[2].as_bytes()], 2, "c is two hops");
    assert!(
        !hops.contains_key(ids[3].as_bytes()),
        "d is unreachable from a"
    );
}

fn node_with_body(text: &str) -> NodeWriteRecord {
    let mut props = BTreeMap::new();
    props.insert(
        "body".to_string(),
        namidb_core::Value::Str(text.to_string()),
    );
    NodeWriteRecord {
        properties: props,
        schema_version: 1,
        ..Default::default()
    }
}

/// Five `:Note` docs. "fox" appears in exactly one (rare → high IDF); "common"
/// appears in four (low IDF). Returns the ids in body order.
async fn build_text_corpus(writer: &mut WriterSession) -> [NodeId; 5] {
    let ids: [NodeId; 5] = std::array::from_fn(|_| NodeId::new());
    let bodies = [
        "fox the cat",
        "common the cat",
        "common the dog",
        "common the bird",
        "common the lizard",
    ];
    for (id, body) in ids.iter().zip(bodies) {
        writer
            .upsert_node("Note", *id, &node_with_body(body))
            .unwrap();
    }
    writer.commit_batch().await.unwrap();
    ids
}

#[tokio::test]
async fn call_search_bm25_ranks_rare_terms_higher() {
    let mut writer = WriterSession::open(store(), paths("call-bm25"))
        .await
        .unwrap();
    let ids = build_text_corpus(&mut writer).await;
    let snapshot = writer.snapshot();

    let rows = run(
        &snapshot,
        "CALL search.bm25({label: 'Note', text_property: 'body', query: 'fox common'}) \
         YIELD node, score RETURN node, score",
    )
    .await;

    // All five docs match at least one query term.
    assert_eq!(rows.len(), 5);

    let score = |r: &namidb_query::Row| match r.get("score") {
        Some(RuntimeValue::Float(s)) => *s,
        other => panic!("score not a float: {other:?}"),
    };
    let id_of = |r: &namidb_query::Row| match r.get("node") {
        Some(RuntimeValue::Node(n)) => n.id,
        other => panic!("node not a node: {other:?}"),
    };

    // The doc with the rare term "fox" (df=1, high IDF) must outscore the four
    // that only share the common term (df=4, low IDF). With IDF=1.0 they would
    // tie; real IDF is what separates them.
    let top = rows
        .iter()
        .max_by(|a, b| score(a).partial_cmp(&score(b)).unwrap())
        .unwrap();
    assert_eq!(id_of(top), ids[0], "the rare-term doc should rank first");
    assert!(score(top) > 0.0);
}

#[tokio::test]
async fn call_search_bm25_mcp_query_shape_executes() {
    // Exactly the query shape the MCP lexical channel generates: a map arg with
    // a list value + a `$param`, then `id(node)` / property access in RETURN and
    // an ORDER BY on the yielded score. Guards the MCP wiring from query-shape
    // regressions without needing an embedder.
    let mut writer = WriterSession::open(store(), paths("call-bm25-shape"))
        .await
        .unwrap();
    let ids = build_text_corpus(&mut writer).await;
    let snapshot = writer.snapshot();

    let cypher =
        "CALL search.bm25({label: 'Note', text_properties: ['body', 'title'], query: $text}) \
                  YIELD node, score \
                  RETURN id(node) AS id, node.path AS path, score \
                  ORDER BY score DESC";
    let parsed = parse(cypher).unwrap_or_else(|e| panic!("parse: {e:?}"));
    let plan = lower(&parsed).unwrap_or_else(|e| panic!("lower: {e:?}"));
    let plan = optimize(plan, &StatsCatalog::empty());
    let mut params = Params::new();
    params.insert("text".into(), RuntimeValue::String("fox common".into()));
    let rows = execute(&plan, &snapshot, &params)
        .await
        .unwrap_or_else(|e| panic!("execute: {e}"));

    assert_eq!(rows.len(), 5);
    // ORDER BY score DESC puts the rare-term doc first; id() yields its uuid.
    let top_id = rows[0].get("id").and_then(|v| match v {
        RuntimeValue::String(s) => Some(s.clone()),
        _ => None,
    });
    assert_eq!(
        top_id,
        Some(ids[0].to_string()),
        "rare-term doc id should be first"
    );
}

#[tokio::test]
async fn call_search_bm25_requires_text_property() {
    let mut writer = WriterSession::open(store(), paths("call-bm25-noprop"))
        .await
        .unwrap();
    build_text_corpus(&mut writer).await;
    let snapshot = writer.snapshot();

    let q = parse("CALL search.bm25({label: 'Note', query: 'fox'}) YIELD node").unwrap();
    let plan = lower(&q).unwrap();
    let plan = optimize(plan, &StatsCatalog::empty());
    let err = execute(&plan, &snapshot, &Params::new()).await.unwrap_err();
    assert!(
        err.is_unsupported(),
        "search.bm25 without a text property should be unsupported, got {err}"
    );
}

#[tokio::test]
async fn call_unknown_search_procedure_is_unsupported() {
    let mut writer = WriterSession::open(store(), paths("call-search-unknown"))
        .await
        .unwrap();
    build_text_corpus(&mut writer).await;
    let snapshot = writer.snapshot();

    let q = parse("CALL search.bogus({label: 'Note'}) YIELD x").unwrap();
    let plan = lower(&q).unwrap();
    let plan = optimize(plan, &StatsCatalog::empty());
    let err = execute(&plan, &snapshot, &Params::new()).await.unwrap_err();
    assert!(
        err.is_unsupported(),
        "unknown search proc should be unsupported, got {err}"
    );
}

#[tokio::test]
async fn call_shortest_path_requires_source() {
    let mut writer = WriterSession::open(store(), paths("call-sp-nosrc"))
        .await
        .unwrap();
    build_directed(&mut writer).await;
    let snapshot = writer.snapshot();

    let q = parse("CALL algo.shortest_path({weighted: true}) YIELD node_id, distance").unwrap();
    let plan = lower(&q).unwrap();
    let plan = optimize(plan, &StatsCatalog::empty());
    let err = execute(&plan, &snapshot, &Params::new()).await.unwrap_err();
    assert!(
        err.is_unsupported(),
        "shortest_path without a source should be unsupported, got {err}"
    );
}
