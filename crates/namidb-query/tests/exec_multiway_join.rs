//! RFC-024: end-to-end execution tests for `LogicalPlan::MultiwayJoin`.
//!
//! No optimiser involvement here. We hand-build the operator and feed
//! it through `execute_factor_path`, which is the entry point the
//! NAMIDB_WCOJ pass will route into once it lands. The graph fixture is
//! a directed cycle so the leapfrog intersection has a non-trivial
//! amount to prune.

use std::collections::BTreeMap;
use std::sync::Arc;

use namidb_core::id::{NamespaceId, NodeId};
use namidb_core::value::Value as CoreValue;
use namidb_storage::{EdgeWriteRecord, NamespacePaths, NodeWriteRecord, WriterSession};
use object_store::memory::InMemory;
use object_store::ObjectStore;

use namidb_query::exec::execute_factor_path;
use namidb_query::parser::RelationshipDirection;
use namidb_query::plan::logical::{EdgeConstraint, LogicalPlan, NodeBinding};
use namidb_query::{Params, RuntimeValue};

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

fn triangle_plan() -> LogicalPlan {
    let vars = vec![
        NodeBinding {
            alias: "a".into(),
            label: Some("Person".into()),
            predicates: Vec::new(),
        },
        NodeBinding {
            alias: "b".into(),
            label: Some("Person".into()),
            predicates: Vec::new(),
        },
        NodeBinding {
            alias: "c".into(),
            label: Some("Person".into()),
            predicates: Vec::new(),
        },
    ];
    let edges = vec![
        // a -[KNOWS]-> b
        EdgeConstraint {
            from_idx: 0,
            to_idx: 1,
            edge_types: vec!["KNOWS".into()],
            direction: RelationshipDirection::Right,
        },
        // b -[KNOWS]-> c
        EdgeConstraint {
            from_idx: 1,
            to_idx: 2,
            edge_types: vec!["KNOWS".into()],
            direction: RelationshipDirection::Right,
        },
        // c -[KNOWS]-> a (closing edge)
        EdgeConstraint {
            from_idx: 2,
            to_idx: 0,
            edge_types: vec!["KNOWS".into()],
            direction: RelationshipDirection::Right,
        },
    ];
    LogicalPlan::MultiwayJoin {
        vars,
        edges,
        ordering: vec![0, 1, 2],
        factorize_required: true,
    }
}

async fn build_triangle_graph(writer: &mut WriterSession, ids: &[NodeId; 3]) {
    let names = ["A", "B", "C"];
    for (id, name) in ids.iter().zip(names.iter()) {
        writer.upsert_node("Person", *id, &person(name)).unwrap();
    }
    let triangle = [
        (ids[0], ids[1]), // A -> B
        (ids[1], ids[2]), // B -> C
        (ids[2], ids[0]), // C -> A
    ];
    for (src, dst) in triangle {
        writer.upsert_edge("KNOWS", src, dst, &edge()).unwrap();
    }
    writer.commit_batch().await.unwrap();
}

#[tokio::test]
async fn multiway_join_finds_directed_triangle() {
    let mut writer = WriterSession::open(store(), paths("mwj-triangle"))
        .await
        .unwrap();
    let ids: [NodeId; 3] = std::array::from_fn(|_| NodeId::new());
    build_triangle_graph(&mut writer, &ids).await;
    let snapshot = writer.snapshot();

    let plan = triangle_plan();
    let rows = execute_factor_path(&plan, &snapshot, &Params::new())
        .await
        .unwrap();

    // Three ordered rotations of the same directed triangle:
    // (A,B,C), (B,C,A), (C,A,B). The leapfrog finds them all because the
    // outer scan is unconstrained.
    assert_eq!(
        rows.len(),
        3,
        "expected 3 triangle rotations, got {}",
        rows.len()
    );

    let mut triples: Vec<(NodeId, NodeId, NodeId)> = rows
        .into_iter()
        .map(|r| {
            let a = node_id(r.get("a")).expect("a bound");
            let b = node_id(r.get("b")).expect("b bound");
            let c = node_id(r.get("c")).expect("c bound");
            (a, b, c)
        })
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
async fn multiway_join_prunes_open_paths() {
    // Five-node graph with one triangle (A,B,C) and a dangling chain
    // (D -> E -> A). The chain does not close back, so D and E should
    // never appear in the output. Demonstrates the leapfrog rejects
    // candidates that satisfy the chain but not the closing edge.
    let mut writer = WriterSession::open(store(), paths("mwj-prune"))
        .await
        .unwrap();
    let ids: [NodeId; 5] = std::array::from_fn(|_| NodeId::new());
    let names = ["A", "B", "C", "D", "E"];
    for (id, name) in ids.iter().zip(names.iter()) {
        writer.upsert_node("Person", *id, &person(name)).unwrap();
    }
    let edges_to_add = [
        (ids[0], ids[1]), // A -> B
        (ids[1], ids[2]), // B -> C
        (ids[2], ids[0]), // C -> A (closes triangle)
        (ids[3], ids[4]), // D -> E (open chain)
        (ids[4], ids[0]), // E -> A (still open)
    ];
    for (src, dst) in edges_to_add {
        writer.upsert_edge("KNOWS", src, dst, &edge()).unwrap();
    }
    writer.commit_batch().await.unwrap();
    let snapshot = writer.snapshot();

    let rows = execute_factor_path(&triangle_plan(), &snapshot, &Params::new())
        .await
        .unwrap();

    assert_eq!(rows.len(), 3, "open chain must not contribute matches");
    for r in rows {
        let a = node_id(r.get("a")).unwrap();
        assert!(
            a == ids[0] || a == ids[1] || a == ids[2],
            "leaked non-triangle binding for `a`: {:?}",
            a
        );
    }
}

fn node_id(v: Option<&RuntimeValue>) -> Option<NodeId> {
    match v? {
        RuntimeValue::Node(n) => Some(n.id),
        _ => None,
    }
}
