//! plan-aware routing tests (RFC-018 §4 caveat eliminated).
//!
//! Each test builds a tiny graph with an edge type that has a declared
//! property (`weight: Float64`) and flushes both nodes and edges to
//! SSTs so that the CSR / SST path divergence is observable. Then it
//! issues a query and asserts:
//!
//! - When the query references the rel binding (`r` as a whole, or
//!   `r.weight` as a property), the executor routes through the
//!   full-property SST path even with `NAMIDB_ADJACENCY=1`.
//! - When the rel binding is unreferenced (no alias, or alias bound
//!   but never read), the slim CSR path is used — visible as a build
//!   on the AdjacencyCache counters.
//!
//! Tests force `NAMIDB_ADJACENCY=1` explicitly so they remain valid
//! regardless of the runtime default flip.

use std::collections::BTreeMap;
use std::sync::Arc;

use namidb_core::id::{NamespaceId, NodeId};
use namidb_core::schema::{DataType, EdgeTypeDef, LabelDef, PropertyDef, SchemaBuilder};
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

fn person_label() -> LabelDef {
    LabelDef {
        name: "Person".into(),
        properties: vec![PropertyDef::new("name", DataType::Utf8, false).unwrap()],
    }
}

fn works_with_edge() -> EdgeTypeDef {
    EdgeTypeDef {
        name: "WORKS_WITH".into(),
        src_label: "Person".into(),
        dst_label: "Person".into(),
        properties: vec![PropertyDef::new("weight", DataType::Float64, false).unwrap()],
    }
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

fn weighted_edge(weight: f64) -> EdgeWriteRecord {
    let mut props: BTreeMap<String, CoreValue> = BTreeMap::new();
    props.insert("weight".into(), CoreValue::F64(weight));
    EdgeWriteRecord {
        properties: props,
        schema_version: 1,
    }
}

/// Builds a small graph: Alice → Bob (weight 1.0), Alice → Carol
/// (weight 2.5), Bob → Carol (weight 3.0). Forces a flush so the edges
/// live in an SST (memtable-resident edges always carry full properties
/// regardless of routing).
async fn build_weighted_graph(name: &str) -> (WriterSession, [NodeId; 3]) {
    // The adjacency / node caches are attached at `WriterSession::open`
    // based on the env var read AT THAT MOMENT. Force ON before open so
    // every test in this file sees the cache regardless of whether the
    // runner inherited `NAMIDB_ADJACENCY=0` from the outside.
    std::env::set_var("NAMIDB_ADJACENCY", "1");
    std::env::set_var("NAMIDB_NODE_CACHE", "1");
    let mut writer = WriterSession::open(store(), paths(name)).await.unwrap();
    let schema = SchemaBuilder::new()
        .label(person_label())
        .unwrap()
        .edge_type(works_with_edge())
        .unwrap()
        .build();

    let alice = NodeId::new();
    let bob = NodeId::new();
    let carol = NodeId::new();
    for (id, name_str) in [(alice, "Alice"), (bob, "Bob"), (carol, "Carol")] {
        writer.upsert_node("Person", id, &person(name_str)).unwrap();
    }
    writer
        .upsert_edge("WORKS_WITH", alice, bob, &weighted_edge(1.0))
        .unwrap();
    writer
        .upsert_edge("WORKS_WITH", alice, carol, &weighted_edge(2.5))
        .unwrap();
    writer
        .upsert_edge("WORKS_WITH", bob, carol, &weighted_edge(3.0))
        .unwrap();
    writer.commit_batch().await.unwrap();
    // Flush so the edges live in an SST and the CSR / SST routing
    // divergence is observable. Memtable-resident edges retain full
    // properties on both paths.
    writer.flush(schema).await.unwrap();

    (writer, [alice, bob, carol])
}

#[tokio::test]
async fn csr_used_when_rel_alias_absent_in_pattern() {
    // `MATCH (a)-[]->(b)` — no rel binding. Slim CSR is always safe;
    // the adjacency cache should record at least one build on the
    // `WORKS_WITH` forward direction.
    let (writer, _ids) = build_weighted_graph("plan-route-no-rel").await;
    let snapshot = writer.snapshot();

    std::env::set_var("NAMIDB_ADJACENCY", "1");
    let q = parse("MATCH (a:Person)-[:WORKS_WITH]->(b:Person) RETURN a.name AS from, b.name AS to")
        .unwrap();
    let plan = lower(&q).unwrap();
    let rows = execute(&plan, &snapshot, &Params::new()).await.unwrap();
    assert_eq!(rows.len(), 3);

    let cache = writer.adjacency_cache().expect("adjacency cache on");
    assert!(
        cache.builds() >= 1,
        "expected at least one CSR build, got {}",
        cache.builds()
    );
}

#[tokio::test]
async fn csr_used_when_rel_alias_unreferenced_downstream() {
    // `MATCH (a)-[r:WORKS_WITH]->(b) RETURN a.name, b.name` — `r` is
    // bound but never read. Same routing as the anonymous case.
    let (writer, _ids) = build_weighted_graph("plan-route-rel-unused").await;
    let snapshot = writer.snapshot();

    std::env::set_var("NAMIDB_ADJACENCY", "1");
    let q = parse(
        "MATCH (a:Person)-[r:WORKS_WITH]->(b:Person) \
         RETURN a.name AS from, b.name AS to",
    )
    .unwrap();
    let plan = lower(&q).unwrap();
    let rows = execute(&plan, &snapshot, &Params::new()).await.unwrap();
    assert_eq!(rows.len(), 3);

    let cache = writer.adjacency_cache().expect("adjacency cache on");
    assert!(
        cache.builds() >= 1,
        "rel unused — expected CSR build, got {}",
        cache.builds()
    );
}

#[tokio::test]
async fn sst_used_when_rel_property_returned() {
    // `RETURN r.weight` — properties must survive routing. CSR would
    // hand back `properties.is_empty()`; plan-aware routing forces
    // SST.
    let (writer, _ids) = build_weighted_graph("plan-route-rel-prop").await;
    let snapshot = writer.snapshot();

    std::env::set_var("NAMIDB_ADJACENCY", "1");
    let q = parse(
        "MATCH (a:Person)-[r:WORKS_WITH]->(b:Person) \
         RETURN a.name AS from, b.name AS to, r.weight AS w \
         ORDER BY w",
    )
    .unwrap();
    let plan = lower(&q).unwrap();
    let rows = execute(&plan, &snapshot, &Params::new()).await.unwrap();

    let weights: Vec<f64> = rows
        .iter()
        .map(|r| match r.get("w") {
            Some(RuntimeValue::Float(f)) => *f,
            other => panic!("unexpected weight: {:?}", other),
        })
        .collect();
    assert_eq!(weights, vec![1.0, 2.5, 3.0]);

    // CSR path NOT used for this Expand — `builds()` should stay at 0
    // because nothing else exercised it.
    let cache = writer.adjacency_cache().expect("adjacency cache on");
    assert_eq!(
        cache.builds(),
        0,
        "rel.weight read — expected zero CSR builds, got {}",
        cache.builds()
    );
}

#[tokio::test]
async fn sst_used_when_rel_returned_whole() {
    // `RETURN r` — whole-rel reference; the RelValue must carry full
    // properties.
    let (writer, _ids) = build_weighted_graph("plan-route-rel-whole").await;
    let snapshot = writer.snapshot();

    std::env::set_var("NAMIDB_ADJACENCY", "1");
    let q = parse("MATCH (a:Person)-[r:WORKS_WITH]->(b:Person) RETURN r").unwrap();
    let plan = lower(&q).unwrap();
    let rows = execute(&plan, &snapshot, &Params::new()).await.unwrap();
    assert_eq!(rows.len(), 3);
    for row in &rows {
        match row.get("r") {
            Some(RuntimeValue::Rel(r)) => {
                assert!(
                    r.properties.contains_key("weight"),
                    "whole-rel reference must carry full properties; got {:?}",
                    r.properties
                );
            }
            other => panic!("unexpected rel value: {:?}", other),
        }
    }

    let cache = writer.adjacency_cache().expect("adjacency cache on");
    assert_eq!(
        cache.builds(),
        0,
        "rel whole-return — expected zero CSR builds, got {}",
        cache.builds()
    );
}

#[tokio::test]
async fn sst_used_when_rel_property_in_where() {
    // `WHERE r.weight > 2.0` — filter must see real values, not Null.
    let (writer, _ids) = build_weighted_graph("plan-route-rel-where").await;
    let snapshot = writer.snapshot();

    std::env::set_var("NAMIDB_ADJACENCY", "1");
    let q = parse(
        "MATCH (a:Person)-[r:WORKS_WITH]->(b:Person) \
         WHERE r.weight > 2.0 \
         RETURN b.name AS name \
         ORDER BY name",
    )
    .unwrap();
    let plan = lower(&q).unwrap();
    let rows = execute(&plan, &snapshot, &Params::new()).await.unwrap();
    let names: Vec<&str> = rows
        .iter()
        .map(|r| match r.get("name") {
            Some(RuntimeValue::String(s)) => s.as_str(),
            other => panic!("unexpected: {:?}", other),
        })
        .collect();
    // Two edges have weight > 2.0: Alice→Carol (2.5) and Bob→Carol (3.0).
    assert_eq!(names, vec!["Carol", "Carol"]);
}

#[tokio::test]
async fn sst_used_when_rel_property_in_order_by() {
    // `ORDER BY r.weight DESC` — sort key must see real values.
    let (writer, _ids) = build_weighted_graph("plan-route-rel-orderby").await;
    let snapshot = writer.snapshot();

    std::env::set_var("NAMIDB_ADJACENCY", "1");
    let q = parse(
        "MATCH (a:Person)-[r:WORKS_WITH]->(b:Person) \
         RETURN a.name AS from, b.name AS to \
         ORDER BY r.weight DESC",
    )
    .unwrap();
    let plan = lower(&q).unwrap();
    let rows = execute(&plan, &snapshot, &Params::new()).await.unwrap();
    assert_eq!(rows.len(), 3);
    // Heaviest first: Bob→Carol (3.0), Alice→Carol (2.5), Alice→Bob (1.0).
    let first_from = match rows[0].get("from") {
        Some(RuntimeValue::String(s)) => s.clone(),
        other => panic!("unexpected: {:?}", other),
    };
    let first_to = match rows[0].get("to") {
        Some(RuntimeValue::String(s)) => s.clone(),
        other => panic!("unexpected: {:?}", other),
    };
    assert_eq!(first_from, "Bob");
    assert_eq!(first_to, "Carol");
}

#[tokio::test]
async fn csr_and_sst_route_independently_within_one_query() {
    // Two Expands in the same query: r1 is read (via `r1.weight`), r2 is
    // bound but unused. Plan-aware routing decides per Expand. r2's
    // Expand should go through CSR; r1's must use SST.
    //
    // Build a small graph where there are TWO edges in a chain:
    // Alice -[r1:WORKS_WITH]-> Bob -[r2:WORKS_WITH]-> Carol. After the
    // single bench above, the chain already exists.
    let (writer, _ids) = build_weighted_graph("plan-route-mixed").await;
    let snapshot = writer.snapshot();

    std::env::set_var("NAMIDB_ADJACENCY", "1");
    let q = parse(
        "MATCH (a:Person)-[r1:WORKS_WITH]->(b:Person)-[r2:WORKS_WITH]->(c:Person) \
         RETURN a.name AS from, c.name AS to, r1.weight AS w1 \
         ORDER BY from, to",
    )
    .unwrap();
    let plan = lower(&q).unwrap();
    let rows = execute(&plan, &snapshot, &Params::new()).await.unwrap();

    // Alice→Bob→Carol is the only 2-hop in the seed graph.
    assert_eq!(rows.len(), 1);
    let w1 = match rows[0].get("w1") {
        Some(RuntimeValue::Float(f)) => *f,
        other => panic!("unexpected w1: {:?}", other),
    };
    assert_eq!(w1, 1.0, "r1.weight comes from SST path (full property)");

    let cache = writer.adjacency_cache().expect("adjacency cache on");
    assert!(
        cache.builds() >= 1,
        "r2 unused — expected at least one CSR build for that Expand, got {}",
        cache.builds()
    );
}
