//! End-to-end tests for write clauses.
//!
//! Exercises CREATE / MATCH+CREATE / SET / REMOVE / DELETE / DETACH DELETE
//! / MERGE-match / MERGE-create against a fresh `WriterSession`. After each
//! mutation the test snapshots the writer to confirm durability.

use std::collections::BTreeMap;
use std::sync::Arc;

use namidb_core::id::{NamespaceId, NodeId};
use namidb_core::value::Value as CoreValue;
use namidb_storage::{NamespacePaths, NodeWriteRecord, WriterSession};
use object_store::memory::InMemory;
use object_store::ObjectStore;

use namidb_query::{execute_write, lower, parse, Params, RuntimeValue};

fn store() -> Arc<dyn ObjectStore> {
    Arc::new(InMemory::new())
}

fn paths(name: &str) -> NamespacePaths {
    NamespacePaths::new("tenants", NamespaceId::new(name).unwrap())
}

#[tokio::test]
async fn create_single_node_persists() {
    let mut writer = WriterSession::open(store(), paths("w-create-1"))
        .await
        .unwrap();
    let q = parse("CREATE (a:Person {name: 'Ada', age: 36}) RETURN a").unwrap();
    let plan = lower(&q).unwrap();
    let outcome = execute_write(&plan, &mut writer, &Params::new())
        .await
        .unwrap();
    assert_eq!(outcome.nodes_created, 1);
    assert_eq!(outcome.edges_created, 0);
    assert_eq!(outcome.rows.len(), 1);
    // Snapshot reads see the new node.
    let snap = writer.snapshot();
    let nodes = snap.scan_label("Person").await.unwrap();
    assert_eq!(nodes.len(), 1);
    assert_eq!(
        nodes[0].properties.get("name"),
        Some(&CoreValue::Str("Ada".into()))
    );
}

#[tokio::test]
async fn match_then_create_relationship() {
    let mut writer = WriterSession::open(store(), paths("w-match-create"))
        .await
        .unwrap();
    // Seed two persons via the storage API for determinism.
    let alice = NodeId::new();
    let bob = NodeId::new();
    let mut p_alice = BTreeMap::new();
    p_alice.insert("name".into(), CoreValue::Str("Alice".into()));
    let mut p_bob = BTreeMap::new();
    p_bob.insert("name".into(), CoreValue::Str("Bob".into()));
    writer
        .upsert_node(
            "Person",
            alice,
            &NodeWriteRecord {
                properties: p_alice,
                schema_version: 1,
            },
        )
        .unwrap();
    writer
        .upsert_node(
            "Person",
            bob,
            &NodeWriteRecord {
                properties: p_bob,
                schema_version: 1,
            },
        )
        .unwrap();
    writer.commit_batch().await.unwrap();

    let q = parse(
        "MATCH (a:Person {_id: $aid}), (b:Person {_id: $bid}) \
 CREATE (a)-[r:KNOWS]->(b) RETURN r",
    )
    .unwrap();
    let plan = lower(&q).unwrap();
    let mut params = Params::new();
    params.insert("aid".into(), RuntimeValue::String(alice.to_string()));
    params.insert("bid".into(), RuntimeValue::String(bob.to_string()));
    let outcome = execute_write(&plan, &mut writer, &params).await.unwrap();
    assert_eq!(outcome.edges_created, 1);
    let snap = writer.snapshot();
    let edges = snap.out_edges("KNOWS", alice).await.unwrap();
    assert_eq!(edges.edges.len(), 1);
    assert_eq!(edges.edges[0].dst, bob);
}

#[tokio::test]
async fn two_match_clauses_then_create_relationship() {
    // Regression: two separate MATCH clauses must propagate both
    // bindings to CREATE. Previously `combine` discarded the prior
    // plan, so only the second MATCH's binding survived and CREATE
    // failed to resolve the first endpoint.
    let mut writer = WriterSession::open(store(), paths("w-two-match-create"))
        .await
        .unwrap();
    let alice = NodeId::new();
    let bob = NodeId::new();
    let mut p_alice = BTreeMap::new();
    p_alice.insert("name".into(), CoreValue::Str("Alice".into()));
    let mut p_bob = BTreeMap::new();
    p_bob.insert("name".into(), CoreValue::Str("Bob".into()));
    writer
        .upsert_node(
            "Person",
            alice,
            &NodeWriteRecord {
                properties: p_alice,
                schema_version: 1,
            },
        )
        .unwrap();
    writer
        .upsert_node(
            "Person",
            bob,
            &NodeWriteRecord {
                properties: p_bob,
                schema_version: 1,
            },
        )
        .unwrap();
    writer.commit_batch().await.unwrap();

    let q = parse(
        "MATCH (a:Person {_id: $aid}) \
 MATCH (b:Person {_id: $bid}) \
 CREATE (a)-[r:KNOWS]->(b) RETURN r",
    )
    .unwrap();
    let plan = lower(&q).unwrap();
    let mut params = Params::new();
    params.insert("aid".into(), RuntimeValue::String(alice.to_string()));
    params.insert("bid".into(), RuntimeValue::String(bob.to_string()));
    let outcome = execute_write(&plan, &mut writer, &params).await.unwrap();
    assert_eq!(outcome.edges_created, 1);
    let snap = writer.snapshot();
    let edges = snap.out_edges("KNOWS", alice).await.unwrap();
    assert_eq!(edges.edges.len(), 1);
    assert_eq!(edges.edges[0].dst, bob);
}

#[tokio::test]
async fn set_property_round_trips() {
    let mut writer = WriterSession::open(store(), paths("w-set")).await.unwrap();
    let alice = NodeId::new();
    let mut p = BTreeMap::new();
    p.insert("name".into(), CoreValue::Str("Alice".into()));
    p.insert("age".into(), CoreValue::I64(30));
    writer
        .upsert_node(
            "Person",
            alice,
            &NodeWriteRecord {
                properties: p,
                schema_version: 1,
            },
        )
        .unwrap();
    writer.commit_batch().await.unwrap();

    let q = parse("MATCH (a:Person {_id: $aid}) SET a.age = 31").unwrap();
    let plan = lower(&q).unwrap();
    let mut params = Params::new();
    params.insert("aid".into(), RuntimeValue::String(alice.to_string()));
    let outcome = execute_write(&plan, &mut writer, &params).await.unwrap();
    assert_eq!(outcome.properties_set, 1);

    let snap = writer.snapshot();
    let v = snap.lookup_node("Person", alice).await.unwrap().unwrap();
    assert_eq!(v.properties.get("age"), Some(&CoreValue::I64(31)));
    assert_eq!(
        v.properties.get("name"),
        Some(&CoreValue::Str("Alice".into()))
    );
}

#[tokio::test]
async fn remove_property() {
    let mut writer = WriterSession::open(store(), paths("w-remove"))
        .await
        .unwrap();
    let alice = NodeId::new();
    let mut p = BTreeMap::new();
    p.insert("name".into(), CoreValue::Str("Alice".into()));
    p.insert("age".into(), CoreValue::I64(30));
    writer
        .upsert_node(
            "Person",
            alice,
            &NodeWriteRecord {
                properties: p,
                schema_version: 1,
            },
        )
        .unwrap();
    writer.commit_batch().await.unwrap();

    let q = parse("MATCH (a:Person {_id: $aid}) REMOVE a.age").unwrap();
    let plan = lower(&q).unwrap();
    let mut params = Params::new();
    params.insert("aid".into(), RuntimeValue::String(alice.to_string()));
    let _outcome = execute_write(&plan, &mut writer, &params).await.unwrap();

    let snap = writer.snapshot();
    let v = snap.lookup_node("Person", alice).await.unwrap().unwrap();
    assert!(!v.properties.contains_key("age"));
    assert_eq!(
        v.properties.get("name"),
        Some(&CoreValue::Str("Alice".into()))
    );
}

#[tokio::test]
async fn detach_delete_removes_node_and_edges() {
    let mut writer = WriterSession::open(store(), paths("w-detach"))
        .await
        .unwrap();
    let alice = NodeId::new();
    let bob = NodeId::new();
    writer
        .upsert_node(
            "Person",
            alice,
            &NodeWriteRecord {
                properties: BTreeMap::new(),
                schema_version: 1,
            },
        )
        .unwrap();
    writer
        .upsert_node(
            "Person",
            bob,
            &NodeWriteRecord {
                properties: BTreeMap::new(),
                schema_version: 1,
            },
        )
        .unwrap();
    writer
        .upsert_edge(
            "KNOWS",
            alice,
            bob,
            &namidb_storage::EdgeWriteRecord {
                properties: BTreeMap::new(),
                schema_version: 1,
            },
        )
        .unwrap();
    writer.commit_batch().await.unwrap();

    let q = parse("MATCH (a:Person {_id: $aid}) DETACH DELETE a").unwrap();
    let plan = lower(&q).unwrap();
    let mut params = Params::new();
    params.insert("aid".into(), RuntimeValue::String(alice.to_string()));
    let outcome = execute_write(&plan, &mut writer, &params).await.unwrap();
    assert_eq!(outcome.nodes_deleted, 1);
    assert!(outcome.edges_deleted >= 1);

    let snap = writer.snapshot();
    assert!(snap.lookup_node("Person", alice).await.unwrap().is_none());
    let edges = snap.out_edges("KNOWS", alice).await.unwrap();
    assert_eq!(edges.edges.len(), 0);
}

#[tokio::test]
async fn merge_match_path_runs_on_match_sets() {
    let mut writer = WriterSession::open(store(), paths("w-merge-match"))
        .await
        .unwrap();
    let alice = NodeId::new();
    let mut p = BTreeMap::new();
    p.insert("externalId".into(), CoreValue::I64(42));
    p.insert("seen".into(), CoreValue::I64(1));
    writer
        .upsert_node(
            "Person",
            alice,
            &NodeWriteRecord {
                properties: p,
                schema_version: 1,
            },
        )
        .unwrap();
    writer.commit_batch().await.unwrap();

    let q = parse(
        "MERGE (a:Person {externalId: 42}) \
 ON MATCH SET a.seen = 2",
    )
    .unwrap();
    let plan = lower(&q).unwrap();
    let outcome = execute_write(&plan, &mut writer, &Params::new())
        .await
        .unwrap();
    assert_eq!(outcome.nodes_created, 0);
    assert_eq!(outcome.properties_set, 1);

    let snap = writer.snapshot();
    let v = snap.lookup_node("Person", alice).await.unwrap().unwrap();
    assert_eq!(v.properties.get("seen"), Some(&CoreValue::I64(2)));
}

#[tokio::test]
async fn merge_create_path_creates_and_runs_on_create_sets() {
    let mut writer = WriterSession::open(store(), paths("w-merge-create"))
        .await
        .unwrap();

    let q = parse(
        "MERGE (a:Person {externalId: 7}) \
 ON CREATE SET a.firstSeen = 1",
    )
    .unwrap();
    let plan = lower(&q).unwrap();
    let outcome = execute_write(&plan, &mut writer, &Params::new())
        .await
        .unwrap();
    assert_eq!(outcome.nodes_created, 1);
    // properties_set counts the ON CREATE SET application.
    assert_eq!(outcome.properties_set, 1);

    let snap = writer.snapshot();
    let nodes = snap.scan_label("Person").await.unwrap();
    assert_eq!(nodes.len(), 1);
    assert_eq!(
        nodes[0].properties.get("externalId"),
        Some(&CoreValue::I64(7))
    );
    assert_eq!(
        nodes[0].properties.get("firstSeen"),
        Some(&CoreValue::I64(1))
    );
}

#[tokio::test]
async fn id_property_is_user_owned_after_reservation_lifted() {
    // Regression for Bug #1: `id` used to be reserved as the internal
    // NodeId sigil; after the rename to `_id`, `id` is just another
    // user property. `CREATE (n:Foo {_id: $uuid, id: 'external-42'})`
    // must persist `id` and a later `MATCH (n) WHERE n.id = 'external-42'`
    // should find that node by user property.
    let mut writer = WriterSession::open(store(), paths("w-id-prop"))
        .await
        .unwrap();
    let nid = NodeId::new();
    let q = parse("CREATE (n:Foo {_id: $nid, id: 'external-42', name: 'Ada'}) RETURN n").unwrap();
    let plan = lower(&q).unwrap();
    let mut params = Params::new();
    params.insert("nid".into(), RuntimeValue::String(nid.to_string()));
    let outcome = execute_write(&plan, &mut writer, &params).await.unwrap();
    assert_eq!(outcome.nodes_created, 1);

    // Snapshot must show `id` as a real property, while the storage
    // NodeId equals the `_id` we passed in.
    let snap = writer.snapshot();
    let stored = snap
        .lookup_node("Foo", nid)
        .await
        .unwrap()
        .expect("Foo present");
    assert_eq!(
        stored.properties.get("id"),
        Some(&CoreValue::Str("external-42".into())),
        "id must be persisted as a user property",
    );
    assert!(
        !stored.properties.contains_key("_id"),
        "_id must NOT leak into the property map",
    );

    // Read-side: `n._id` should surface the internal NodeId and
    // `n.id` the user value.
    let read_q = parse("MATCH (n:Foo {_id: $nid}) RETURN n._id AS nid, n.id AS biz_id").unwrap();
    let read_plan = lower(&read_q).unwrap();
    let outcome = execute_write(&read_plan, &mut writer, &params)
        .await
        .unwrap();
    assert_eq!(outcome.rows.len(), 1);
    match outcome.rows[0].get("nid") {
        Some(RuntimeValue::String(s)) => assert_eq!(s, &nid.to_string()),
        other => panic!("unexpected nid: {:?}", other),
    }
    match outcome.rows[0].get("biz_id") {
        Some(RuntimeValue::String(s)) => assert_eq!(s, "external-42"),
        other => panic!("unexpected biz_id: {:?}", other),
    }
}

#[tokio::test]
async fn merge_with_relationship_creates_then_matches_idempotently() {
    // Regression: MERGE (a)-[r:R]->(b) was lowering to [Node, Node, Rel]
    // but `find_merge_matches` reads pattern positionally as
    // [Node head, Rel, Node tail]. After the lower_merge reorder, this
    // round-trips: first execution creates both nodes + the edge, second
    // execution finds them and is a no-op.
    let mut writer = WriterSession::open(store(), paths("w-merge-rel"))
        .await
        .unwrap();

    let q = parse(
        "MERGE (a:Person {externalId: 1})-[r:KNOWS]->(b:Person {externalId: 2}) \
 RETURN a, b",
    )
    .unwrap();
    let plan = lower(&q).unwrap();

    // First run: create path. Two nodes + one edge.
    let outcome = execute_write(&plan, &mut writer, &Params::new())
        .await
        .unwrap();
    assert_eq!(outcome.nodes_created, 2);
    assert_eq!(outcome.edges_created, 1);
    let snap = writer.snapshot();
    let people = snap.scan_label("Person").await.unwrap();
    assert_eq!(people.len(), 2);
    let alice = people
        .iter()
        .find(|n| n.properties.get("externalId") == Some(&CoreValue::I64(1)))
        .expect("alice present")
        .id;
    let bob = people
        .iter()
        .find(|n| n.properties.get("externalId") == Some(&CoreValue::I64(2)))
        .expect("bob present")
        .id;
    let edges = snap.out_edges("KNOWS", alice).await.unwrap();
    assert_eq!(edges.edges.len(), 1);
    assert_eq!(edges.edges[0].dst, bob);

    // Second run: match path must find the existing triple and not
    // create duplicates.
    let outcome2 = execute_write(&plan, &mut writer, &Params::new())
        .await
        .unwrap();
    assert_eq!(outcome2.nodes_created, 0, "MERGE must not duplicate nodes");
    assert_eq!(outcome2.edges_created, 0, "MERGE must not duplicate edges");
}

#[tokio::test]
async fn create_chain_node_rel_node() {
    let mut writer = WriterSession::open(store(), paths("w-chain"))
        .await
        .unwrap();
    let q = parse(
        "CREATE (a:Person {name: 'Ada'})-[r:KNOWS {weight: 5}]->(b:Person {name: 'Lin'}) \
 RETURN a.name AS aname, b.name AS bname",
    )
    .unwrap();
    let plan = lower(&q).unwrap();
    let outcome = execute_write(&plan, &mut writer, &Params::new())
        .await
        .unwrap();
    assert_eq!(outcome.nodes_created, 2);
    assert_eq!(outcome.edges_created, 1);
    assert_eq!(outcome.rows.len(), 1);
    match outcome.rows[0].get("aname") {
        Some(RuntimeValue::String(s)) => assert_eq!(s, "Ada"),
        other => panic!("unexpected: {:?}", other),
    }
    match outcome.rows[0].get("bname") {
        Some(RuntimeValue::String(s)) => assert_eq!(s, "Lin"),
        other => panic!("unexpected: {:?}", other),
    }
}

#[tokio::test]
async fn create_node_with_vector_literal_persists_as_corevalue_vec() {
    // The whole point of `vector()` is to land as `CoreValue::Vec` on
    // disk — verify the property survives the writer round-trip and
    // is visible to a snapshot read.
    let mut writer = WriterSession::open(store(), paths("w-create-vector"))
        .await
        .unwrap();
    let q = parse("CREATE (d:Doc {title: 'embedding-1', emb: vector([0.1, 0.2, 0.3])}) RETURN d")
        .unwrap();
    let plan = lower(&q).unwrap();
    let outcome = execute_write(&plan, &mut writer, &Params::new())
        .await
        .unwrap();
    assert_eq!(outcome.nodes_created, 1);

    let snap = writer.snapshot();
    let nodes = snap.scan_label("Doc").await.unwrap();
    assert_eq!(nodes.len(), 1);
    assert_eq!(
        nodes[0].properties.get("emb"),
        Some(&CoreValue::Vec(vec![0.1_f32, 0.2_f32, 0.3_f32])),
        "expected emb to round-trip as CoreValue::Vec"
    );
    assert_eq!(
        nodes[0].properties.get("title"),
        Some(&CoreValue::Str("embedding-1".into()))
    );
}

#[tokio::test]
async fn create_node_with_vector_from_list_parameter() {
    // Embeddings normally arrive as a `$param` — exercise the path
    // where `vector()` consumes a `List` value passed through `Params`.
    let mut writer = WriterSession::open(store(), paths("w-vector-param"))
        .await
        .unwrap();
    let q = parse("CREATE (d:Doc {emb: vector($v)}) RETURN d").unwrap();
    let plan = lower(&q).unwrap();
    let mut params = Params::new();
    params.insert(
        "v".into(),
        RuntimeValue::List(vec![
            RuntimeValue::Float(1.5),
            RuntimeValue::Integer(2),
            RuntimeValue::Float(-3.25),
        ]),
    );
    let outcome = execute_write(&plan, &mut writer, &params).await.unwrap();
    assert_eq!(outcome.nodes_created, 1);

    let snap = writer.snapshot();
    let nodes = snap.scan_label("Doc").await.unwrap();
    assert_eq!(nodes.len(), 1);
    assert_eq!(
        nodes[0].properties.get("emb"),
        Some(&CoreValue::Vec(vec![1.5_f32, 2.0_f32, -3.25_f32])),
        "integer elements must be coerced to f32 alongside floats"
    );
}

#[tokio::test]
async fn bare_list_literal_still_rejected_without_vector_wrapper() {
    // Regression guard: `vector()` is the *only* way to persist a
    // numeric collection. A bare `[…]` literal must keep failing so
    // users do not silently get a List stored as something else.
    let mut writer = WriterSession::open(store(), paths("w-bare-list"))
        .await
        .unwrap();
    let q = parse("CREATE (d:Doc {emb: [0.1, 0.2, 0.3]}) RETURN d").unwrap();
    let plan = lower(&q).unwrap();
    let err = execute_write(&plan, &mut writer, &Params::new())
        .await
        .expect_err("bare list literal must not be storable in v0");
    let msg = format!("{:?}", err);
    assert!(
        msg.contains("only scalars are storable"),
        "unexpected error: {}",
        msg
    );
}
