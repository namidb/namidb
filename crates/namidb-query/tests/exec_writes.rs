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

use namidb_query::cost::StatsCatalog;
use namidb_query::{execute, execute_write, lower, optimize, parse, Params, RuntimeValue};

fn store() -> Arc<dyn ObjectStore> {
    Arc::new(InMemory::new())
}

fn paths(name: &str) -> NamespacePaths {
    NamespacePaths::new("tenants", NamespaceId::new(name).unwrap())
}

/// Lower + execute a write clause against `writer`, returning the outcome.
async fn write_q(writer: &mut WriterSession, text: &str) -> namidb_query::WriteOutcome {
    let plan = lower(&parse(text).unwrap()).unwrap();
    execute_write(&plan, writer, &Params::new()).await.unwrap()
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
async fn create_and_match_multi_label_node() {
    let mut writer = WriterSession::open(store(), paths("w-multilabel"))
        .await
        .unwrap();
    // CREATE a node carrying two labels.
    let q = parse("CREATE (a:Person:Admin {name: 'Ada'}) RETURN a").unwrap();
    let plan = lower(&q).unwrap();
    let outcome = execute_write(&plan, &mut writer, &Params::new())
        .await
        .unwrap();
    assert_eq!(outcome.nodes_created, 1);
    // The created node value already carries both labels.
    match &outcome.rows[0].get("a") {
        Some(RuntimeValue::Node(n)) => {
            assert!(n.labels.contains("Person") && n.labels.contains("Admin"));
        }
        other => panic!("expected node, got {other:?}"),
    }

    // Helper: run a read query and return its row count (raw lowering).
    async fn count(writer: &WriterSession, q_text: &str) -> usize {
        let snap = writer.snapshot();
        let plan = lower(&parse(q_text).unwrap()).unwrap();
        execute(&plan, &snap, &Params::new()).await.unwrap().len()
    }

    // Visible under each of its labels individually...
    assert_eq!(count(&writer, "MATCH (n:Person) RETURN n").await, 1);
    assert_eq!(count(&writer, "MATCH (n:Admin) RETURN n").await, 1);
    // ...and under the conjunction of both (it carries both).
    assert_eq!(count(&writer, "MATCH (n:Person:Admin) RETURN n").await, 1);
    // But NOT under a conjunction that includes a label it lacks.
    assert_eq!(count(&writer, "MATCH (n:Person:Manager) RETURN n").await, 0);

    // The optimized plan (label_eq cleanup + pushdown) must agree.
    let snap = writer.snapshot();
    let catalog = StatsCatalog::from_manifest(&snap.manifest().manifest);
    let opt = optimize(
        lower(&parse("MATCH (n:Admin:Person) RETURN n").unwrap()).unwrap(),
        &catalog,
    );
    assert_eq!(execute(&opt, &snap, &Params::new()).await.unwrap().len(), 1);

    // labels(n) returns the full set, sorted (BTreeSet order).
    let snap = writer.snapshot();
    let plan = lower(&parse("MATCH (n:Person:Admin) RETURN labels(n) AS ls").unwrap()).unwrap();
    let rows = execute(&plan, &snap, &Params::new()).await.unwrap();
    match rows[0].get("ls") {
        Some(RuntimeValue::List(items)) => {
            let got: Vec<&str> = items
                .iter()
                .map(|v| match v {
                    RuntimeValue::String(s) => s.as_str(),
                    _ => panic!("non-string label"),
                })
                .collect();
            assert_eq!(got, vec!["Admin", "Person"]);
        }
        other => panic!("labels(n) should be a list, got {other:?}"),
    }
}

#[tokio::test]
async fn set_and_remove_label_mutate_the_set() {
    let mut writer = WriterSession::open(store(), paths("w-setlabel"))
        .await
        .unwrap();
    write_q(&mut writer, "CREATE (a:Person {name: 'Ada'})").await;

    // SET adds a label (union).
    let out = write_q(&mut writer, "MATCH (a:Person) SET a:Admin RETURN a").await;
    assert_eq!(out.labels_set, 1);
    match out.rows[0].get("a") {
        Some(RuntimeValue::Node(n)) => {
            assert!(n.labels.contains("Person") && n.labels.contains("Admin"));
        }
        other => panic!("expected node, got {other:?}"),
    }
    // The added label is durable: now matchable under :Admin.
    {
        let snap = writer.snapshot();
        let plan = lower(&parse("MATCH (n:Admin) RETURN n").unwrap()).unwrap();
        assert_eq!(
            execute(&plan, &snap, &Params::new()).await.unwrap().len(),
            1
        );
    }

    // REMOVE drops a label (difference); the node stays under its remaining one.
    let out = write_q(&mut writer, "MATCH (a:Admin) REMOVE a:Person RETURN a").await;
    assert_eq!(out.labels_set, 1);
    {
        let snap = writer.snapshot();
        let admin = lower(&parse("MATCH (n:Admin) RETURN n").unwrap()).unwrap();
        let person = lower(&parse("MATCH (n:Person) RETURN n").unwrap()).unwrap();
        assert_eq!(
            execute(&admin, &snap, &Params::new()).await.unwrap().len(),
            1
        );
        assert_eq!(
            execute(&person, &snap, &Params::new()).await.unwrap().len(),
            0,
            "Person was removed"
        );
    }
}

#[tokio::test]
async fn property_update_preserves_label_set() {
    let mut writer = WriterSession::open(store(), paths("w-propkeeplabels"))
        .await
        .unwrap();
    write_q(&mut writer, "CREATE (a:Person:Admin {name: 'Ada'})").await;
    // A property update must NOT collapse the multi-label node to one label.
    write_q(&mut writer, "MATCH (a:Person) SET a.age = 36 RETURN a").await;
    let snap = writer.snapshot();
    let plan = lower(&parse("MATCH (n:Person:Admin) RETURN n").unwrap()).unwrap();
    assert_eq!(
        execute(&plan, &snap, &Params::new()).await.unwrap().len(),
        1,
        "both labels must survive a property update"
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
                ..Default::default()
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
                ..Default::default()
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
                ..Default::default()
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
                ..Default::default()
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
                ..Default::default()
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
                ..Default::default()
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
                ..Default::default()
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
                ..Default::default()
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
                ..Default::default()
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
async fn merge_multi_hop_creates_then_matches_idempotently() {
    // B2: MERGE with two hops — three nodes, two edges. On the first
    // execution the whole path is created; on the second the same path
    // is matched and no duplicates are produced.
    let mut writer = WriterSession::open(store(), paths("w-merge-multi-hop"))
        .await
        .unwrap();

    let q = parse(
        "MERGE (a:Person {externalId: 1})-[r1:KNOWS]->(b:Person {externalId: 2})\
         -[r2:KNOWS]->(c:Person {externalId: 3}) \
         RETURN a, b, c",
    )
    .unwrap();
    let plan = lower(&q).unwrap();

    let outcome = execute_write(&plan, &mut writer, &Params::new())
        .await
        .unwrap();
    assert_eq!(outcome.nodes_created, 3, "expected three Persons created");
    assert_eq!(outcome.edges_created, 2, "expected two KNOWS edges");

    // Second run on the same writer must be a pure match — no creates.
    let outcome2 = execute_write(&plan, &mut writer, &Params::new())
        .await
        .unwrap();
    assert_eq!(
        outcome2.nodes_created, 0,
        "MERGE must not duplicate nodes on rerun"
    );
    assert_eq!(
        outcome2.edges_created, 0,
        "MERGE must not duplicate edges on rerun"
    );
}

#[tokio::test]
async fn bare_list_literal_now_persists_as_list() {
    // Previously bare `[v, ...]` literals failed with
    // "only scalars are storable in v0" because the writer rejected
    // `RuntimeValue::List`. With Value::List landing in core and
    // round-tripping through __overflow_json, bare lists now persist
    // and re-decode as the same shape.
    let mut writer = WriterSession::open(store(), paths("w-bare-list"))
        .await
        .unwrap();
    let q = parse("CREATE (d:Doc {emb: [0.1, 0.2, 0.3]}) RETURN d.emb AS emb").unwrap();
    let plan = lower(&q).unwrap();
    let outcome = execute_write(&plan, &mut writer, &Params::new())
        .await
        .unwrap();
    assert_eq!(outcome.nodes_created, 1);
    match outcome.rows[0].get("emb") {
        Some(RuntimeValue::List(items)) => {
            assert_eq!(items.len(), 3);
            assert!(matches!(
                &items[0],
                RuntimeValue::Float(_) | RuntimeValue::Integer(_)
            ));
        }
        other => panic!("expected list, got {:?}", other),
    }
}

#[tokio::test]
async fn create_node_with_list_property_round_trips() {
    let mut writer = WriterSession::open(store(), paths("w-create-list"))
        .await
        .unwrap();
    // No SchemaBuilder run; `tags` falls into __overflow_json on the
    // storage side. The new Value::List variant survives the JSON
    // round-trip and re-materialises as RuntimeValue::List.
    let q = parse(
        "CREATE (a:Person {name: 'Ada', tags: ['rust', 'ssh']}) \
         RETURN a.tags AS tags",
    )
    .unwrap();
    let plan = lower(&q).unwrap();
    let outcome = execute_write(&plan, &mut writer, &Params::new())
        .await
        .unwrap();
    assert_eq!(outcome.nodes_created, 1);
    match outcome.rows[0].get("tags") {
        Some(RuntimeValue::List(items)) => {
            assert_eq!(items.len(), 2);
            assert!(
                matches!(&items[0], RuntimeValue::String(s) if s == "rust"),
                "got {:?}",
                items[0]
            );
        }
        other => panic!("expected list, got {:?}", other),
    }

    // Snapshot read goes through the overflow JSON column and must
    // give back the same list shape.
    let snap = writer.snapshot();
    let nodes = snap.scan_label("Person").await.unwrap();
    assert_eq!(nodes.len(), 1);
    match nodes[0].properties.get("tags") {
        Some(CoreValue::List(items)) => {
            assert_eq!(items.len(), 2);
            assert!(matches!(&items[0], CoreValue::Str(s) if s == "rust"));
        }
        other => panic!("expected list, got {:?}", other),
    }
}

#[tokio::test]
async fn create_node_with_map_property_round_trips() {
    let mut writer = WriterSession::open(store(), paths("w-create-map"))
        .await
        .unwrap();
    let q = parse(
        "CREATE (a:Doc {title: 'Hello', meta: {source: 'cli', version: 3}}) \
         RETURN a.meta AS meta",
    )
    .unwrap();
    let plan = lower(&q).unwrap();
    let outcome = execute_write(&plan, &mut writer, &Params::new())
        .await
        .unwrap();
    assert_eq!(outcome.nodes_created, 1);
    match outcome.rows[0].get("meta") {
        Some(RuntimeValue::Map(m)) => {
            assert!(matches!(m.get("source"), Some(RuntimeValue::String(s)) if s == "cli"));
            assert!(matches!(m.get("version"), Some(RuntimeValue::Integer(3))));
        }
        other => panic!("expected map, got {:?}", other),
    }

    let snap = writer.snapshot();
    let nodes = snap.scan_label("Doc").await.unwrap();
    assert_eq!(nodes.len(), 1);
    match nodes[0].properties.get("meta") {
        Some(CoreValue::Map(m)) => {
            assert!(matches!(m.get("source"), Some(CoreValue::Str(s)) if s == "cli"));
        }
        other => panic!("expected map, got {:?}", other),
    }
}

#[tokio::test]
async fn merge_pattern_property_reads_outer_row_binding() {
    // UNWIND introduces a row-local alias that the MERGE pattern's
    // properties expression should read against the current outer row.
    // Without that wiring the match-or-create decision falls through
    // and the writer ends up creating one node per call to MERGE.
    let mut writer = WriterSession::open(store(), paths("w-merge-outer-row"))
        .await
        .unwrap();
    // Seed an existing Ada so the first iteration must MATCH, not CREATE.
    let setup = parse("CREATE (a:Person {name: 'Ada', age: 36}) RETURN a").unwrap();
    let plan = lower(&setup).unwrap();
    execute_write(&plan, &mut writer, &Params::new())
        .await
        .unwrap();

    let q = parse(
        "UNWIND ['Ada', 'Bob'] AS who \
         MERGE (a:Person {name: who}) \
         RETURN a.name AS name ORDER BY name",
    )
    .unwrap();
    let plan = lower(&q).unwrap();
    let outcome = execute_write(&plan, &mut writer, &Params::new())
        .await
        .unwrap();

    // Ada already existed, so MERGE should match it. Bob is new, so
    // MERGE creates exactly one node.
    assert_eq!(outcome.nodes_created, 1);
    let names: Vec<&str> = outcome
        .rows
        .iter()
        .map(|r| match r.get("name") {
            Some(RuntimeValue::String(s)) => s.as_str(),
            other => panic!("unexpected: {:?}", other),
        })
        .collect();
    assert_eq!(names, vec!["Ada", "Bob"]);

    // Rerunning the same query must be idempotent.
    let outcome2 = execute_write(&plan, &mut writer, &Params::new())
        .await
        .unwrap();
    assert_eq!(outcome2.nodes_created, 0);
    let snap = writer.snapshot();
    let nodes = snap.scan_label("Person").await.unwrap();
    assert_eq!(nodes.len(), 2);
}

#[tokio::test]
async fn merge_rel_over_matched_nodes_is_idempotent() {
    // MATCH (a), MATCH (b), MERGE (a)-[r:KNOWS]->(b). The MERGE needs
    // to see the matched a and b on the outer row and decide whether
    // to create the edge or reuse it.
    let mut writer = WriterSession::open(store(), paths("w-merge-rel-over-match"))
        .await
        .unwrap();
    let setup = parse(
        "CREATE (a:Person {name: 'Ada'}), (b:Person {name: 'Bob'}) \
         RETURN a, b",
    )
    .unwrap();
    let plan = lower(&setup).unwrap();
    execute_write(&plan, &mut writer, &Params::new())
        .await
        .unwrap();

    let q = parse(
        "MATCH (a:Person {name: 'Ada'}), (b:Person {name: 'Bob'}) \
         MERGE (a)-[r:KNOWS]->(b) \
         RETURN r",
    )
    .unwrap();
    let plan = lower(&q).unwrap();
    let outcome1 = execute_write(&plan, &mut writer, &Params::new())
        .await
        .unwrap();
    assert_eq!(outcome1.nodes_created, 0);
    assert_eq!(outcome1.edges_created, 1);

    // Rerun: edge already exists, MERGE must reuse it.
    let outcome2 = execute_write(&plan, &mut writer, &Params::new())
        .await
        .unwrap();
    assert_eq!(outcome2.nodes_created, 0);
    assert_eq!(
        outcome2.edges_created, 0,
        "second MERGE should not duplicate the edge"
    );
}
