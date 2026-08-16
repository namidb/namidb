//! Plan item 1 (docs/testing/25tb-readiness.md): every rich traversal
//! semantic was exec-tested only against the unflushed memtable, while a
//! 25 TB deployment serves essentially everything from paged edge SSTs.
//! This matrix runs each traversal shape against four physical states of the
//! SAME logical graph and requires the full row multiset to be identical:
//!
//!   (a) memtable-only        — the historical exec-test baseline
//!   (b) pure SST             — one flush, empty memtable (the 25 TB shape)
//!   (c) SST + neutral overlay— flush, then a same-batch tombstone+re-upsert
//!                              of an existing edge plus an identical node
//!                              re-upsert: real memtable deltas over the SST,
//!                              zero net semantic change
//!   (d) staged flush         — half the graph flushed, half live
//!
//! Any divergence is a serving-route bug, not a fixture artifact.

use std::collections::BTreeMap;
use std::sync::Arc;

use namidb_core::id::{NamespaceId, NodeId};
use namidb_core::schema::{DataType, EdgeTypeDef, LabelDef, PropertyDef, Schema, SchemaBuilder};
use namidb_core::value::Value as CoreValue;
use namidb_storage::{EdgeWriteRecord, NamespacePaths, NodeWriteRecord, WriterSession};
use object_store::memory::InMemory;
use object_store::ObjectStore;

use namidb_query::{execute, lower, parse, Params};

fn schema() -> Schema {
    SchemaBuilder::new()
        .label(LabelDef {
            name: "Person".into(),
            properties: vec![PropertyDef::new("name", DataType::Utf8, false).unwrap()],
        })
        .unwrap()
        .edge_type(EdgeTypeDef {
            name: "KNOWS".into(),
            src_label: "Person".into(),
            dst_label: "Person".into(),
            properties: vec![],
        })
        .unwrap()
        .edge_type(EdgeTypeDef {
            name: "LIKES".into(),
            src_label: "Person".into(),
            dst_label: "Person".into(),
            properties: vec![],
        })
        .unwrap()
        .build()
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

const NAMES: [&str; 6] = ["Alice", "Bob", "Carol", "Dave", "Eve", "Frank"];

/// Deterministic ids so every physical state holds the byte-identical graph.
fn pid(ordinal: usize) -> NodeId {
    let mut bytes = [0u8; 16];
    bytes[0] = 0x77;
    bytes[15] = ordinal as u8 + 1;
    NodeId::from_uuid(uuid::Uuid::from_bytes(bytes))
}

/// KNOWS: a triangle (Alice→Bob→Carol, Alice→Carol), a tail
/// (Carol→Dave→Eve), a 2-cycle (Eve↔Frank). LIKES: Alice→Dave, Bob→Frank.
/// Frank has no outgoing KNOWS beyond the cycle; Dave has no LIKES —
/// OPTIONAL rows exercise real nulls.
fn knows_edges() -> Vec<(usize, usize)> {
    vec![(0, 1), (1, 2), (0, 2), (2, 3), (3, 4), (4, 5), (5, 4)]
}

fn likes_edges() -> Vec<(usize, usize)> {
    vec![(0, 3), (1, 5)]
}

async fn open(name: &str) -> WriterSession {
    let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
    let paths = NamespacePaths::new("tenants", NamespaceId::new(name).unwrap());
    WriterSession::open(store, paths).await.unwrap()
}

fn seed_nodes(writer: &mut WriterSession) {
    for (ordinal, name) in NAMES.iter().enumerate() {
        writer
            .upsert_node("Person", pid(ordinal), &person(name))
            .unwrap();
    }
}

fn seed_edges(writer: &mut WriterSession, knows: &[(usize, usize)], likes: &[(usize, usize)]) {
    for (src, dst) in knows {
        writer
            .upsert_edge("KNOWS", pid(*src), pid(*dst), &edge())
            .unwrap();
    }
    for (src, dst) in likes {
        writer
            .upsert_edge("LIKES", pid(*src), pid(*dst), &edge())
            .unwrap();
    }
}

/// (a) memtable-only.
async fn state_memtable(name: &str) -> WriterSession {
    let mut writer = open(name).await;
    seed_nodes(&mut writer);
    seed_edges(&mut writer, &knows_edges(), &likes_edges());
    writer.commit_batch().await.unwrap();
    writer
}

/// (b) pure SST.
async fn state_flushed(name: &str) -> WriterSession {
    let mut writer = state_memtable(name).await;
    writer.flush(schema()).await.unwrap();
    writer
}

/// (c) SST + a semantically neutral memtable overlay: tombstone AND
/// re-upsert one existing KNOWS edge in the same batch (last write wins),
/// plus a byte-identical node re-upsert. The reader must merge real deltas
/// over the SST and land on the same rows.
async fn state_overlay(name: &str) -> WriterSession {
    let mut writer = state_flushed(name).await;
    writer.tombstone_edge("KNOWS", pid(0), pid(1)).unwrap();
    writer
        .upsert_edge("KNOWS", pid(0), pid(1), &edge())
        .unwrap();
    writer
        .upsert_node("Person", pid(2), &person("Carol"))
        .unwrap();
    writer.commit_batch().await.unwrap();
    writer
}

/// (d) staged flush: nodes plus the first half of each edge set flushed,
/// the rest live in the memtable.
async fn state_staged(name: &str) -> WriterSession {
    let mut writer = open(name).await;
    seed_nodes(&mut writer);
    let knows = knows_edges();
    let likes = likes_edges();
    let (k1, k2) = knows.split_at(knows.len() / 2);
    let (l1, l2) = likes.split_at(likes.len() / 2);
    seed_edges(&mut writer, k1, l1);
    writer.commit_batch().await.unwrap();
    writer.flush(schema()).await.unwrap();
    seed_edges(&mut writer, k2, l2);
    writer.commit_batch().await.unwrap();
    writer
}

/// Canonical row multiset: every binding rendered stably, rows sorted.
async fn canonical_rows(writer: &WriterSession, query: &str) -> Vec<String> {
    let snapshot = writer.snapshot();
    let parsed = parse(query).unwrap_or_else(|error| panic!("parse `{query}`: {error:?}"));
    let plan = lower(&parsed).unwrap_or_else(|error| panic!("lower `{query}`: {error:?}"));
    let rows = execute(&plan, &snapshot, &Params::new())
        .await
        .unwrap_or_else(|error| panic!("execute `{query}`: {error:?}"));
    let mut out: Vec<String> = rows
        .iter()
        .map(|row| {
            let cells: Vec<String> = row
                .bindings
                .iter()
                .map(|(column, value)| format!("{column}={value:?}"))
                .collect();
            cells.join("|")
        })
        .collect();
    out.sort();
    out
}

const QUERIES: [(&str, &str); 12] = [
    (
        "expand",
        "MATCH (a:Person)-[:KNOWS]->(b:Person) RETURN a.name AS a, b.name AS b",
    ),
    (
        "expand-inverse",
        "MATCH (a:Person)<-[:KNOWS]-(b:Person) RETURN a.name AS a, b.name AS b",
    ),
    (
        "undirected",
        "MATCH (a:Person {name: 'Carol'})-[:KNOWS]-(x:Person) RETURN x.name AS x",
    ),
    (
        "var-length-directed",
        "MATCH (a:Person {name: 'Alice'})-[:KNOWS*1..3]->(x:Person) RETURN x.name AS x",
    ),
    (
        "var-length-undirected-exact",
        "MATCH (a:Person {name: 'Alice'})-[:KNOWS*2..2]-(x:Person) RETURN x.name AS x",
    ),
    (
        "var-length-path-binding",
        "MATCH p = (a:Person {name: 'Alice'})-[:KNOWS*1..2]->(b:Person) \
         RETURN length(p) AS hops, b.name AS b",
    ),
    (
        "back-reference-cycle",
        "MATCH (a:Person)-[:KNOWS]->(b:Person)-[:KNOWS]->(a) RETURN a.name AS a",
    ),
    (
        "triangle-multiway",
        "MATCH (a:Person)-[:KNOWS]->(b:Person)-[:KNOWS]->(c:Person), (a)-[:KNOWS]->(c) \
         RETURN a.name AS a, b.name AS b, c.name AS c",
    ),
    (
        "alternation",
        "MATCH (a:Person)-[:KNOWS|:LIKES]->(b:Person) RETURN a.name AS a, b.name AS b",
    ),
    (
        "optional-var-length",
        "MATCH (a:Person) OPTIONAL MATCH (a)-[:LIKES*1..2]->(x:Person) \
         RETURN a.name AS a, x.name AS x",
    ),
    (
        "shortest-path",
        "MATCH (a:Person {name: 'Alice'}), (b:Person {name: 'Eve'}) \
         MATCH p = shortestPath((a)-[:KNOWS*..5]-(b)) RETURN length(p) AS hops",
    ),
    (
        "all-shortest-paths",
        "MATCH (a:Person {name: 'Alice'}), (b:Person {name: 'Dave'}) \
         MATCH p = allShortestPaths((a)-[:KNOWS*..5]->(b)) RETURN length(p) AS hops",
    ),
];

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn traversals_are_route_invariant_across_all_physical_states() {
    let memtable = state_memtable("route-matrix-a").await;
    let flushed = state_flushed("route-matrix-b").await;
    let overlay = state_overlay("route-matrix-c").await;
    let staged = state_staged("route-matrix-d").await;

    for (label, query) in QUERIES {
        let baseline = canonical_rows(&memtable, query).await;
        assert!(
            !baseline.is_empty(),
            "{label}: the fixture must produce rows or the parity below is vacuous"
        );
        for (state, writer) in [
            ("pure-sst", &flushed),
            ("sst+overlay", &overlay),
            ("staged-flush", &staged),
        ] {
            let got = canonical_rows(writer, query).await;
            assert_eq!(
                got, baseline,
                "{label}: the {state} route must serve the exact memtable row \
                 multiset for `{query}`"
            );
        }
    }

    // The optional query above must actually produce null rows (Dave/Eve/
    // Frank have no outgoing LIKES), or the OPTIONAL leg proved nothing.
    let optional = canonical_rows(&flushed, QUERIES[9].1).await;
    assert!(
        optional.iter().any(|row| row.contains("x=Null")),
        "OPTIONAL var-length must emit null rows, got {optional:?}"
    );
}
