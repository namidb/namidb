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
async fn call_fastrp_yields_structural_embeddings() {
    let mut writer = WriterSession::open(store(), paths("call-fastrp"))
        .await
        .unwrap();
    let ids = build(&mut writer).await;
    let snapshot = writer.snapshot();

    let rows = run(
        &snapshot,
        "CALL algo.fastRP({dimension: 16, seed: 7}) YIELD node_id, embedding",
    )
    .await;
    assert_eq!(rows.len(), 5, "one embedding per node");

    let mut emb: BTreeMap<[u8; 16], Vec<f32>> = BTreeMap::new();
    for r in &rows {
        let nid = match r.get("node_id") {
            Some(RuntimeValue::Node(n)) => *n.id.as_bytes(),
            other => panic!("node_id not a node: {other:?}"),
        };
        let v = match r.get("embedding") {
            Some(RuntimeValue::Vector(v)) => v.clone(),
            other => panic!("embedding not a vector: {other:?}"),
        };
        assert_eq!(v.len(), 16, "embedding has the requested dimension");
        emb.insert(nid, v);
    }

    let cos = |x: &[f32], y: &[f32]| -> f32 {
        let dot: f32 = x.iter().zip(y).map(|(p, q)| p * q).sum();
        let nx = x.iter().map(|p| p * p).sum::<f32>().sqrt();
        let ny = y.iter().map(|p| p * p).sum::<f32>().sqrt();
        if nx == 0.0 || ny == 0.0 {
            0.0
        } else {
            dot / (nx * ny)
        }
    };
    // The connected pair a-b is more similar than a vs the isolate e.
    let connected = cos(
        emb[ids[0].as_bytes()].as_slice(),
        emb[ids[1].as_bytes()].as_slice(),
    );
    let isolate = cos(
        emb[ids[0].as_bytes()].as_slice(),
        emb[ids[4].as_bytes()].as_slice(),
    );
    assert!(
        connected > isolate,
        "connected pair ({connected}) should embed closer than the isolate ({isolate})"
    );
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
async fn call_wcc_rejects_non_projection_arguments() {
    let mut writer = WriterSession::open(store(), paths("call-wcc-args"))
        .await
        .unwrap();
    build(&mut writer).await;
    let snapshot = writer.snapshot();

    // An empty map is a valid (no-op) projection…
    let rows = run(&snapshot, "CALL algo.wcc({}) YIELD node_id, component").await;
    assert_eq!(rows.len(), 5);

    // …but algorithm options wcc doesn't have, or a non-map argument, error.
    for bad in [
        "CALL algo.wcc({damping: 0.9}) YIELD node_id, component",
        "CALL algo.wcc(1) YIELD node_id, component",
    ] {
        let q = parse(bad).unwrap();
        let plan = lower(&q).unwrap();
        let plan = optimize(plan, &StatsCatalog::empty());
        let err = execute(&plan, &snapshot, &Params::new()).await.unwrap_err();
        assert!(
            err.is_unsupported(),
            "{bad} should be unsupported, got {err}"
        );
    }
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

#[cfg(feature = "text-index")]
#[tokio::test]
async fn call_search_bm25_uses_the_index() {
    use namidb_core::schema::{DataType, LabelDef, PropertyDef, SchemaBuilder};
    use namidb_storage::manifest::{ManifestStore, TextIndexDescriptor};
    use namidb_storage::memtable::{MemKey, MemOp, Memtable};
    use namidb_storage::{compact_l0_to_l1, flush, WriterFence};

    let store = store();
    let p = paths("bm25-index");
    let ms = ManifestStore::new(store.clone(), p.clone());
    let mut base = ms.bootstrap(uuid::Uuid::now_v7()).await.unwrap();
    let note_id = base.manifest.label_dict.intern("Note");
    base.manifest.text_indexes.push(TextIndexDescriptor::new(
        "note_ft".into(),
        "Note".into(),
        vec!["body".into()],
    ));
    let schema = SchemaBuilder::new()
        .label(LabelDef {
            name: "Note".into(),
            properties: vec![PropertyDef::new("body", DataType::Utf8, true).unwrap()],
        })
        .unwrap()
        .build();
    let fence = WriterFence::new(base.manifest.epoch);

    // "fox" is rare (one doc); "common" is in the rest. Two L0 SSTs.
    let bodies = [
        "fox the cat",
        "common the cat",
        "common the dog",
        "common the bird",
    ];
    let mut ids: Vec<NodeId> = Vec::new();
    let mut cur = base;
    let mut i: u64 = 0;
    for chunk in bodies.chunks(2) {
        let mut mt = Memtable::new();
        for b in chunk {
            let id = NodeId::new();
            ids.push(id);
            let mut props = BTreeMap::new();
            props.insert("body".to_string(), namidb_core::Value::Str(b.to_string()));
            let rec = NodeWriteRecord {
                properties: props,
                schema_version: 1,
                labels: vec![note_id.0],
            };
            mt.apply(
                MemKey::Node { id },
                i + 1,
                MemOp::Upsert(rec.encode().unwrap()),
            );
            i += 1;
        }
        cur = flush(&ms, &fence, &cur, &mt.freeze(), schema.clone())
            .await
            .unwrap()
            .committed;
    }
    compact_l0_to_l1(&ms, &fence, &cur, &schema).await.unwrap();

    // Reopen so the snapshot reflects the compacted TextIndex SST, then the
    // procedure answers from the index (feature on).
    let writer = WriterSession::open(store.clone(), p.clone()).await.unwrap();
    let snapshot = writer.snapshot();
    let rows = run(
        &snapshot,
        "CALL search.bm25({label: 'Note', text_property: 'body', query: 'fox common'}) \
         YIELD node, score RETURN node, score",
    )
    .await;

    assert_eq!(rows.len(), bodies.len(), "every doc matches a query term");
    let score = |r: &namidb_query::Row| match r.get("score") {
        Some(RuntimeValue::Float(s)) => *s,
        other => panic!("score not a float: {other:?}"),
    };
    let id_of = |r: &namidb_query::Row| match r.get("node") {
        Some(RuntimeValue::Node(n)) => n.id,
        other => panic!("node not a node: {other:?}"),
    };
    let top = rows
        .iter()
        .max_by(|a, b| score(a).partial_cmp(&score(b)).unwrap())
        .unwrap();
    assert_eq!(
        id_of(top),
        ids[0],
        "the rare-term doc ranks first via the index"
    );

    // Freshness: write a NEW doc (now in the memtable, un-compacted) and search
    // for its unique term. The index does not contain it, but the procedure must
    // still find it — `text_search` detects the delta and falls back to the flat
    // scan, so a fresh write is never silently hidden by the index.
    let mut writer = writer;
    let fresh = NodeId::new();
    writer
        .upsert_node("Note", fresh, &node_with_body("zebra"))
        .unwrap();
    writer.commit_batch().await.unwrap();
    let snap2 = writer.snapshot();
    let fresh_rows = run(
        &snap2,
        "CALL search.bm25({label: 'Note', text_property: 'body', query: 'zebra'}) \
         YIELD node, score RETURN node, score",
    )
    .await;
    assert_eq!(fresh_rows.len(), 1, "the just-written doc must be visible");
    assert_eq!(
        id_of(&fresh_rows[0]),
        fresh,
        "fresh write found via flat fallback"
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

/// Two "domains" in one namespace: Person nodes p0→p1→p2 chained by KNOWS,
/// Post nodes q0, q1, and cross-domain LIKES edges p0→q0, p1→q1. Used by the
/// projection tests: filtering to {Person, KNOWS} must hide the posts and the
/// LIKES edges entirely. Returns ([p0, p1, p2], [q0, q1]).
async fn build_two_domains(writer: &mut WriterSession) -> ([NodeId; 3], [NodeId; 2]) {
    let people: [NodeId; 3] = std::array::from_fn(|_| NodeId::new());
    let posts: [NodeId; 2] = std::array::from_fn(|_| NodeId::new());
    for id in &people {
        writer.upsert_node("Person", *id, &node()).unwrap();
    }
    for id in &posts {
        writer.upsert_node("Post", *id, &node()).unwrap();
    }
    writer
        .upsert_edge("KNOWS", people[0], people[1], &edge())
        .unwrap();
    writer
        .upsert_edge("KNOWS", people[1], people[2], &edge())
        .unwrap();
    writer
        .upsert_edge("LIKES", people[0], posts[0], &edge())
        .unwrap();
    writer
        .upsert_edge("LIKES", people[1], posts[1], &edge())
        .unwrap();
    writer.commit_batch().await.unwrap();
    (people, posts)
}

#[tokio::test]
async fn call_wcc_projection_filters_labels_and_edge_types() {
    let mut writer = WriterSession::open(store(), paths("call-wcc-proj"))
        .await
        .unwrap();
    let (people, _posts) = build_two_domains(&mut writer).await;
    let snapshot = writer.snapshot();

    // Whole graph: everything is connected through LIKES → 1 component, 5 rows.
    let rows = run(&snapshot, "CALL algo.wcc() YIELD node_id, component").await;
    assert_eq!(rows.len(), 5);
    let comps: std::collections::BTreeSet<i64> =
        rows.iter().map(|r| int_col(r, "component")).collect();
    assert_eq!(comps.len(), 1, "LIKES bridges people and posts");

    // Projected to Person/KNOWS: only the 3 people, still 1 chain component.
    let rows = run(
        &snapshot,
        "CALL algo.wcc({labels: ['Person'], edge_types: ['KNOWS']}) \
         YIELD node_id, component",
    )
    .await;
    assert_eq!(rows.len(), 3, "posts are projected out");
    let ids: std::collections::BTreeSet<[u8; 16]> = rows.iter().map(node_of).collect();
    for p in &people {
        assert!(ids.contains(p.as_bytes()), "person missing from projection");
    }

    // Label filter alone induces the subgraph: LIKES edges to projected-out
    // posts must not smuggle the posts back in.
    let rows = run(
        &snapshot,
        "CALL algo.wcc({labels: ['Person']}) YIELD node_id, component",
    )
    .await;
    assert_eq!(rows.len(), 3);
}

#[tokio::test]
async fn call_degree_projection_direction_reverse_and_undirected() {
    let mut writer = WriterSession::open(store(), paths("call-degree-proj"))
        .await
        .unwrap();
    let (people, _posts) = build_two_domains(&mut writer).await;
    let snapshot = writer.snapshot();
    let proj = "labels: ['Person'], edge_types: ['KNOWS']";

    // Natural: p0 →1 out, p1 1 in/1 out, p2 1 in.
    let rows = run(
        &snapshot,
        &format!("CALL algo.degree({{{proj}}}) YIELD node_id, in_degree, out_degree, degree"),
    )
    .await;
    let by_node: BTreeMap<[u8; 16], (i64, i64)> = rows
        .iter()
        .map(|r| (node_of(r), (int_col(r, "in_degree"), int_col(r, "out_degree"))))
        .collect();
    assert_eq!(by_node[people[0].as_bytes()], (0, 1));
    assert_eq!(by_node[people[2].as_bytes()], (1, 0));

    // Reverse: in/out swap.
    let rows = run(
        &snapshot,
        &format!(
            "CALL algo.degree({{{proj}, direction: 'reverse'}}) \
             YIELD node_id, in_degree, out_degree, degree"
        ),
    )
    .await;
    let by_node: BTreeMap<[u8; 16], (i64, i64)> = rows
        .iter()
        .map(|r| (node_of(r), (int_col(r, "in_degree"), int_col(r, "out_degree"))))
        .collect();
    assert_eq!(by_node[people[0].as_bytes()], (1, 0));
    assert_eq!(by_node[people[2].as_bytes()], (0, 1));

    // Undirected: every incident edge counts both ways.
    let rows = run(
        &snapshot,
        &format!(
            "CALL algo.degree({{{proj}, direction: 'undirected'}}) \
             YIELD node_id, in_degree, out_degree, degree"
        ),
    )
    .await;
    let by_node: BTreeMap<[u8; 16], (i64, i64)> = rows
        .iter()
        .map(|r| (node_of(r), (int_col(r, "in_degree"), int_col(r, "out_degree"))))
        .collect();
    assert_eq!(by_node[people[1].as_bytes()], (2, 2));
}

#[tokio::test]
async fn call_projection_unknown_label_errors() {
    let mut writer = WriterSession::open(store(), paths("call-proj-unknown"))
        .await
        .unwrap();
    build_two_domains(&mut writer).await;
    let snapshot = writer.snapshot();

    for bad in [
        "CALL algo.wcc({labels: ['Nope']}) YIELD node_id, component",
        "CALL algo.wcc({edge_types: ['NOPE']}) YIELD node_id, component",
        "CALL algo.wcc({direction: 'sideways'}) YIELD node_id, component",
        "CALL algo.wcc({labels: []}) YIELD node_id, component",
    ] {
        let q = parse(bad).unwrap();
        let plan = lower(&q).unwrap();
        let plan = optimize(plan, &StatsCatalog::empty());
        let err = execute(&plan, &snapshot, &Params::new()).await.unwrap_err();
        assert!(err.is_unsupported(), "{bad} should error, got {err}");
    }
}

#[tokio::test]
async fn call_pagerank_accepts_projection_keys() {
    let mut writer = WriterSession::open(store(), paths("call-pr-proj"))
        .await
        .unwrap();
    let (people, _posts) = build_two_domains(&mut writer).await;
    let snapshot = writer.snapshot();

    let rows = run(
        &snapshot,
        "CALL algo.pagerank({labels: ['Person'], edge_types: ['KNOWS'], damping: 0.85}) \
         YIELD node_id, score",
    )
    .await;
    assert_eq!(rows.len(), 3, "only people are ranked");
    // End of the p0→p1→p2 chain accumulates the most rank.
    assert_eq!(node_of(&rows[0]), *people[2].as_bytes());
}

/// Two triangles of Person/KNOWS joined by one bridge edge → Louvain finds the
/// two triangle communities.
#[tokio::test]
async fn call_louvain_separates_bridged_triangles() {
    let mut writer = WriterSession::open(store(), paths("call-louvain"))
        .await
        .unwrap();
    let t1: [NodeId; 3] = std::array::from_fn(|_| NodeId::new());
    let t2: [NodeId; 3] = std::array::from_fn(|_| NodeId::new());
    for id in t1.iter().chain(t2.iter()) {
        writer.upsert_node("N", *id, &node()).unwrap();
    }
    for tri in [&t1, &t2] {
        writer.upsert_edge("E", tri[0], tri[1], &edge()).unwrap();
        writer.upsert_edge("E", tri[1], tri[2], &edge()).unwrap();
        writer.upsert_edge("E", tri[2], tri[0], &edge()).unwrap();
    }
    writer.upsert_edge("E", t1[0], t2[0], &edge()).unwrap();
    writer.commit_batch().await.unwrap();
    let snapshot = writer.snapshot();

    let rows = run(&snapshot, "CALL algo.louvain() YIELD node_id, community").await;
    assert_eq!(rows.len(), 6);
    let by_node: BTreeMap<[u8; 16], i64> = rows
        .iter()
        .map(|r| (node_of(r), int_col(r, "community")))
        .collect();
    let c1 = by_node[t1[0].as_bytes()];
    let c2 = by_node[t2[0].as_bytes()];
    assert_ne!(c1, c2, "the two triangles are distinct communities");
    assert!(t1.iter().all(|n| by_node[n.as_bytes()] == c1));
    assert!(t2.iter().all(|n| by_node[n.as_bytes()] == c2));
}

/// Directed path a→b→c→d→e: Brandes betweenness has exact known values
/// (b=3, c=4, d=3, endpoints 0), and the rows come back score-descending.
#[tokio::test]
async fn call_betweenness_directed_path() {
    let mut writer = WriterSession::open(store(), paths("call-betweenness"))
        .await
        .unwrap();
    let ids: [NodeId; 5] = std::array::from_fn(|_| NodeId::new());
    for id in &ids {
        writer.upsert_node("N", *id, &node()).unwrap();
    }
    for w in ids.windows(2) {
        writer.upsert_edge("E", w[0], w[1], &edge()).unwrap();
    }
    writer.commit_batch().await.unwrap();
    let snapshot = writer.snapshot();

    let rows = run(&snapshot, "CALL algo.betweenness() YIELD node_id, score").await;
    assert_eq!(rows.len(), 5);
    let by_node: BTreeMap<[u8; 16], f64> = rows
        .iter()
        .map(|r| {
            let s = match r.get("score") {
                Some(RuntimeValue::Float(v)) => *v,
                other => panic!("score not a float: {other:?}"),
            };
            (node_of(r), s)
        })
        .collect();
    assert_eq!(by_node[ids[0].as_bytes()], 0.0);
    assert_eq!(by_node[ids[1].as_bytes()], 3.0);
    assert_eq!(by_node[ids[2].as_bytes()], 4.0);
    assert_eq!(by_node[ids[3].as_bytes()], 3.0);
    assert_eq!(by_node[ids[4].as_bytes()], 0.0);
    // Highest-scoring row first.
    assert_eq!(node_of(&rows[0]), *ids[2].as_bytes());

    // Undirected projection halves the doubled raw scores: c carries the
    // same 4 pass-through pairs in the undirected P5.
    let rows = run(
        &snapshot,
        "CALL algo.betweenness({direction: 'undirected'}) YIELD node_id, score",
    )
    .await;
    let c_score = rows
        .iter()
        .find(|r| node_of(r) == *ids[2].as_bytes())
        .map(|r| match r.get("score") {
            Some(RuntimeValue::Float(v)) => *v,
            other => panic!("score not a float: {other:?}"),
        })
        .unwrap();
    assert_eq!(c_score, 4.0);
}

/// The text-index freshness gate is label-scoped: an unflushed write to an
/// UNRELATED label (or a tombstone of a never-indexed id) must not disable the
/// index, while any delta that touches the indexed corpus — a live :Note
/// upsert, or a relabel/delete of an indexed document — still forces the exact
/// flat scan. Asserted at the storage layer, where `Some` vs `None` IS the
/// index-vs-fallback decision.
#[cfg(feature = "text-index")]
#[tokio::test]
async fn text_index_gate_is_label_scoped() {
    use namidb_core::schema::{DataType, LabelDef, PropertyDef, SchemaBuilder};
    use namidb_storage::manifest::{ManifestStore, TextIndexDescriptor};
    use namidb_storage::memtable::{MemKey, MemOp, Memtable};
    use namidb_storage::{compact_l0_to_l1, flush, WriterFence};

    let store = store();
    let p = paths("bm25-gate");
    let ms = ManifestStore::new(store.clone(), p.clone());
    let mut base = ms.bootstrap(uuid::Uuid::now_v7()).await.unwrap();
    let note_id = base.manifest.label_dict.intern("Note");
    base.manifest.text_indexes.push(TextIndexDescriptor::new(
        "note_ft".into(),
        "Note".into(),
        vec!["body".into()],
    ));
    let schema = SchemaBuilder::new()
        .label(LabelDef {
            name: "Note".into(),
            properties: vec![PropertyDef::new("body", DataType::Utf8, true).unwrap()],
        })
        .unwrap()
        .build();
    let fence = WriterFence::new(base.manifest.epoch);

    let bodies = ["fox the cat", "common the dog", "common the bird"];
    let mut ids: Vec<NodeId> = Vec::new();
    let mut cur = base;
    let mut lsn: u64 = 0;
    for chunk in bodies.chunks(2) {
        let mut mt = Memtable::new();
        for b in chunk {
            let id = NodeId::new();
            ids.push(id);
            let mut props = BTreeMap::new();
            props.insert("body".to_string(), namidb_core::Value::Str(b.to_string()));
            let rec = NodeWriteRecord {
                properties: props,
                schema_version: 1,
                labels: vec![note_id.0],
            };
            lsn += 1;
            mt.apply(MemKey::Node { id }, lsn, MemOp::Upsert(rec.encode().unwrap()));
        }
        cur = flush(&ms, &fence, &cur, &mt.freeze(), schema.clone())
            .await
            .unwrap()
            .committed;
    }
    compact_l0_to_l1(&ms, &fence, &cur, &schema).await.unwrap();

    let mut writer = WriterSession::open(store.clone(), p.clone()).await.unwrap();
    let fox = vec!["fox".to_string()];

    // Baseline: compacted, no delta → the index serves.
    assert!(
        writer
            .snapshot()
            .text_search("note_ft", "Note", &fox, Some(5))
            .await
            .unwrap()
            .is_some(),
        "compacted index must serve"
    );

    // Unrelated-label write (fresh id): the index must KEEP serving — this was
    // the O(corpus)-scan-per-query regression under live mixed traffic.
    writer
        .upsert_node("Other", NodeId::new(), &node_with_body("noise"))
        .unwrap();
    writer.commit_batch().await.unwrap();
    let hits = writer
        .snapshot()
        .text_search("note_ft", "Note", &fox, Some(5))
        .await
        .unwrap()
        .expect("unrelated-label delta must not disable the index");
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].0, ids[0], "the fox doc still ranks");

    // Tombstone of a never-indexed id: still clean for this corpus.
    writer.tombstone_node("Other", NodeId::new()).unwrap();
    writer.commit_batch().await.unwrap();
    assert!(
        writer
            .snapshot()
            .text_search("note_ft", "Note", &fox, Some(5))
            .await
            .unwrap()
            .is_some(),
        "tombstone of a non-corpus id must not disable the index"
    );

    // Relabel of an INDEXED document (upsert without :Note): the index would
    // serve the stale doc and its removal shifts the corpus stats → flat scan.
    writer
        .upsert_node("Other", ids[0], &node_with_body("fox the cat"))
        .unwrap();
    writer.commit_batch().await.unwrap();
    let snap = writer.snapshot();
    assert!(
        snap.text_search("note_ft", "Note", &fox, Some(5))
            .await
            .unwrap()
            .is_none(),
        "a dirty id inside the corpus must force the flat scan"
    );
    // End-to-end parity: the flat fallback no longer finds the relabeled doc.
    let rows = run(
        &snap,
        "CALL search.bm25({label: 'Note', text_property: 'body', query: 'fox'}) \
         YIELD node, score RETURN node",
    )
    .await;
    assert!(rows.is_empty(), "relabeled doc must not be served: {rows:?}");
}
