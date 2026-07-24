//! End-to-end tests for write clauses.
//!
//! Exercises CREATE / MATCH+CREATE / SET / REMOVE / DELETE / DETACH DELETE
//! / MERGE-match / MERGE-create against a fresh `WriterSession`. After each
//! mutation the test snapshots the writer to confirm durability.

use std::collections::BTreeMap;
use std::sync::Arc;

use namidb_core::id::{NamespaceId, NodeId};
use namidb_core::schema::{
    Constraint, ConstraintKind, DataType, LabelDef, PropertyDef, SchemaBuilder,
};
use namidb_core::value::Value as CoreValue;
use namidb_storage::{NamespacePaths, NodeWriteRecord, WriterSession};
use object_store::memory::InMemory;
use object_store::ObjectStore;

use namidb_query::cost::StatsCatalog;
use namidb_query::{
    execute, execute_write, execute_write_staged, lower, optimize, parse, Params, RuntimeValue,
};

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
    // A typeless NodeScan is a scan of physical nodes, not a concatenation of
    // label postings: this :Person:Admin node must appear exactly once.
    assert_eq!(count(&writer, "MATCH (n) RETURN n").await, 1);
    // LIMIT takes the executor's capped NodeScan route; it must share the same
    // one-row semantics.
    assert_eq!(count(&writer, "MATCH (n) RETURN n LIMIT 10").await, 1);
    // ...and under the conjunction of both (it carries both).
    assert_eq!(count(&writer, "MATCH (n:Person:Admin) RETURN n").await, 1);
    // But NOT under a conjunction that includes a label it lacks.
    assert_eq!(count(&writer, "MATCH (n:Person:Manager) RETURN n").await, 0);

    // Storage permits an unlabeled node. A typeless MATCH includes it while
    // the labelled scans above remain unchanged.
    writer
        .upsert_node_with_labels(
            Vec::<String>::new(),
            NodeId::new(),
            &NodeWriteRecord {
                properties: BTreeMap::from([("name".into(), CoreValue::Str("Bare".into()))]),
                schema_version: 1,
                ..Default::default()
            },
        )
        .unwrap();
    writer.commit_batch().await.unwrap();
    assert_eq!(count(&writer, "MATCH (n) RETURN n").await, 2);

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
async fn merge_multi_label_matches_or_creates() {
    let mut writer = WriterSession::open(store(), paths("w-mergeml"))
        .await
        .unwrap();
    // First MERGE creates the :Person:Admin node.
    let out = write_q(&mut writer, "MERGE (a:Person:Admin {name: 'Ada'}) RETURN a").await;
    assert_eq!(out.nodes_created, 1);
    match out.rows[0].get("a") {
        Some(RuntimeValue::Node(n)) => {
            assert!(n.labels.contains("Person") && n.labels.contains("Admin"));
        }
        other => panic!("expected node, got {other:?}"),
    }
    // Second MERGE with the same labels + props matches it — no new node.
    let out = write_q(&mut writer, "MERGE (a:Person:Admin {name: 'Ada'}) RETURN a").await;
    assert_eq!(out.nodes_created, 0, "existing :Person:Admin must match");

    // A node carrying only :Person must NOT satisfy MERGE (:Person:Admin): the
    // conjunction requires :Admin too, so MERGE creates a fresh node.
    write_q(&mut writer, "CREATE (b:Person {name: 'Bob'})").await;
    let out = write_q(&mut writer, "MERGE (c:Person:Admin {name: 'Bob'}) RETURN c").await;
    assert_eq!(
        out.nodes_created, 1,
        "Person-only node lacks :Admin, so MERGE must create"
    );
}

#[tokio::test]
async fn multi_label_expand_target_is_conjunctive() {
    let mut writer = WriterSession::open(store(), paths("w-ml-expand"))
        .await
        .unwrap();
    // h1 -> p1(:Person:Admin); h2 -> p2(:Person only).
    write_q(
        &mut writer,
        "CREATE (h:Hub {k: 1})-[:R]->(p1:Person:Admin {n: 'a'})",
    )
    .await;
    write_q(
        &mut writer,
        "CREATE (h:Hub {k: 2})-[:R]->(p2:Person {n: 'b'})",
    )
    .await;
    let snap = writer.snapshot();

    // Non-OPTIONAL multi-label target: only the :Person:Admin neighbour matches.
    let plan = lower(&parse("MATCH (h:Hub)-[:R]->(b:Person:Admin) RETURN b").unwrap()).unwrap();
    assert_eq!(
        execute(&plan, &snap, &Params::new()).await.unwrap().len(),
        1,
        "only the :Person:Admin neighbour matches"
    );

    // OPTIONAL with a multi-label target: both hubs survive. h1 binds its
    // :Person:Admin neighbour; h2 yields b=NULL because its only neighbour
    // lacks :Admin (the Expand enforces the full label set, so a partial-label
    // neighbour is a non-match, not a wrong match).
    let plan = lower(
        &parse("MATCH (h:Hub) OPTIONAL MATCH (h)-[:R]->(b:Person:Admin) RETURN h.k AS k, b")
            .unwrap(),
    )
    .unwrap();
    let rows = execute(&plan, &snap, &Params::new()).await.unwrap();
    assert_eq!(rows.len(), 2, "both hubs preserved by OPTIONAL");
    let (mut bound, mut nulls) = (0, 0);
    for r in &rows {
        match r.get("b") {
            Some(RuntimeValue::Node(_)) => bound += 1,
            Some(RuntimeValue::Null) | None => nulls += 1,
            other => panic!("unexpected b: {other:?}"),
        }
    }
    assert_eq!(bound, 1, "h1's :Person:Admin neighbour binds");
    assert_eq!(
        nulls, 1,
        "h2's :Person-only neighbour is a non-match -> NULL"
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
async fn two_sets_to_same_node_via_different_aliases_both_persist() {
    // MATCH (a) MATCH (b) binding the same node, SET a.c=1 SET b.d=2 must keep
    // BOTH properties. The second SET must rebuild from the node's current
    // staged state (with c=1), not from a stale match-time clone (which loses c).
    let mut writer = WriterSession::open(store(), paths("w-set-alias"))
        .await
        .unwrap();
    let nid = NodeId::new();
    let mut params = Params::new();
    params.insert("nid".into(), RuntimeValue::String(nid.to_string()));
    let q0 = parse("CREATE (n:P {_id: $nid, k: 1}) RETURN n").unwrap();
    execute_write(&lower(&q0).unwrap(), &mut writer, &params)
        .await
        .unwrap();
    writer.commit_batch().await.unwrap();

    let q = parse("MATCH (a:P {k:1}) MATCH (b:P {k:1}) SET a.c = 1 SET b.d = 2 RETURN a").unwrap();
    execute_write(&lower(&q).unwrap(), &mut writer, &Params::new())
        .await
        .unwrap();
    writer.commit_batch().await.unwrap();

    let snap = writer.snapshot();
    let stored = snap
        .lookup_node("P", nid)
        .await
        .unwrap()
        .expect("P present");
    assert_eq!(
        stored.properties.get("c"),
        Some(&CoreValue::I64(1)),
        "c must survive"
    );
    assert_eq!(
        stored.properties.get("d"),
        Some(&CoreValue::I64(2)),
        "d must survive"
    );
}

#[tokio::test]
async fn create_with_colliding_explicit_id_errors() {
    // CREATE must create a NEW node: an explicit `_id` that already exists must
    // fail, not silently overwrite the existing node (a data-integrity /
    // security hole — a client could clobber another node by its id).
    let mut writer = WriterSession::open(store(), paths("w-id-collide"))
        .await
        .unwrap();
    let nid = NodeId::new();
    let mut params = Params::new();
    params.insert("nid".into(), RuntimeValue::String(nid.to_string()));

    let q = parse("CREATE (n:Foo {_id: $nid, name: 'first'}) RETURN n").unwrap();
    let plan = lower(&q).unwrap();
    let outcome = execute_write(&plan, &mut writer, &params).await.unwrap();
    assert_eq!(outcome.nodes_created, 1);
    writer.commit_batch().await.unwrap();

    // Second CREATE with the same _id must be rejected as a constraint error.
    let q2 = parse("CREATE (n:Foo {_id: $nid, name: 'second'}) RETURN n").unwrap();
    let plan2 = lower(&q2).unwrap();
    let err = execute_write(&plan2, &mut writer, &params)
        .await
        .unwrap_err();
    assert!(
        matches!(err, namidb_query::ExecError::Constraint(_)),
        "expected a constraint error on id collision, got: {err:?}"
    );

    // The original node must be untouched (name still 'first').
    writer.discard_batch();
    let snap = writer.snapshot();
    let stored = snap
        .lookup_node("Foo", nid)
        .await
        .unwrap()
        .expect("Foo present");
    assert_eq!(
        stored.properties.get("name"),
        Some(&CoreValue::Str("first".into())),
        "the existing node must not be overwritten",
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
async fn unwind_bulk_edges_match_both_endpoints_then_create() {
    // Issue 01 (bulk-load): a single UNWIND of {from,to} pairs drives a
    // MATCH of BOTH endpoints by the row binding, then CREATE one edge per
    // row. This must create exactly N edges in one round-trip — the shape
    // that previously forced per-edge statements ("binding row not bound").
    let mut writer = WriterSession::open(store(), paths("w-unwind-bulk-edges"))
        .await
        .unwrap();
    for name in ["Alice", "Bob", "Carol"] {
        write_q(
            &mut writer,
            &format!("CREATE (a:Person {{name: '{name}'}}) RETURN a"),
        )
        .await;
    }

    let outcome = write_q(
        &mut writer,
        "UNWIND [{from: 'Alice', to: 'Bob'}, {from: 'Bob', to: 'Carol'}] AS row \
         MATCH (a:Person {name: row.from}), (b:Person {name: row.to}) \
         CREATE (a)-[:KNOWS]->(b)",
    )
    .await;
    assert_eq!(outcome.edges_created, 2, "one KNOWS edge per UNWIND row");

    let snap = writer.snapshot();
    let plan = lower(
        &parse(
            "MATCH (a:Person)-[:KNOWS]->(b:Person) \
             RETURN a.name AS from, b.name AS to ORDER BY from, to",
        )
        .unwrap(),
    )
    .unwrap();
    let rows = execute(&plan, &snap, &Params::new()).await.unwrap();
    let pairs: Vec<(String, String)> = rows
        .iter()
        .map(|r| match (r.get("from"), r.get("to")) {
            (Some(RuntimeValue::String(a)), Some(RuntimeValue::String(b))) => {
                (a.clone(), b.clone())
            }
            other => panic!("unexpected: {:?}", other),
        })
        .collect();
    assert_eq!(
        pairs,
        vec![
            ("Alice".to_string(), "Bob".to_string()),
            ("Bob".to_string(), "Carol".to_string()),
        ],
    );
}

#[tokio::test]
async fn set_plus_map_merges_properties() {
    // Issue 02: `SET n += {map}` merges the map into the node, keeping
    // existing properties not named in the map.
    let mut writer = WriterSession::open(store(), paths("w-set-plus-map"))
        .await
        .unwrap();
    write_q(
        &mut writer,
        "CREATE (a:Person {name: 'Ada', age: 36}) RETURN a",
    )
    .await;
    let outcome = write_q(
        &mut writer,
        "MATCH (a:Person {name: 'Ada'}) SET a += {age: 40, city: 'Quito'} RETURN a",
    )
    .await;
    assert_eq!(outcome.properties_set, 2);
    let snap = writer.snapshot();
    let nodes = snap.scan_label("Person").await.unwrap();
    assert_eq!(nodes.len(), 1);
    let p = &nodes[0].properties;
    assert_eq!(p.get("name"), Some(&CoreValue::Str("Ada".into())));
    assert_eq!(p.get("age"), Some(&CoreValue::I64(40)));
    assert_eq!(p.get("city"), Some(&CoreValue::Str("Quito".into())));
}

#[tokio::test]
async fn set_eq_map_replaces_all_properties() {
    // `SET n = {map}` replaces the whole property set, dropping anything
    // not present in the map.
    let mut writer = WriterSession::open(store(), paths("w-set-eq-map"))
        .await
        .unwrap();
    write_q(
        &mut writer,
        "CREATE (a:Person {name: 'Ada', age: 36}) RETURN a",
    )
    .await;
    write_q(
        &mut writer,
        "MATCH (a:Person {name: 'Ada'}) SET a = {name: 'Bob'} RETURN a",
    )
    .await;
    let snap = writer.snapshot();
    let nodes = snap.scan_label("Person").await.unwrap();
    assert_eq!(nodes.len(), 1);
    let p = &nodes[0].properties;
    assert_eq!(p.get("name"), Some(&CoreValue::Str("Bob".into())));
    assert_eq!(
        p.get("age"),
        None,
        "= replaces, dropping unlisted properties"
    );
}

#[tokio::test]
async fn set_plus_map_null_value_removes_property() {
    let mut writer = WriterSession::open(store(), paths("w-set-plus-null"))
        .await
        .unwrap();
    write_q(
        &mut writer,
        "CREATE (a:Person {name: 'Ada', age: 36}) RETURN a",
    )
    .await;
    write_q(
        &mut writer,
        "MATCH (a:Person {name: 'Ada'}) SET a += {age: null} RETURN a",
    )
    .await;
    let snap = writer.snapshot();
    let nodes = snap.scan_label("Person").await.unwrap();
    let p = &nodes[0].properties;
    assert_eq!(p.get("name"), Some(&CoreValue::Str("Ada".into())));
    assert_eq!(p.get("age"), None, "+= null removes the property");
}

#[tokio::test]
async fn merge_on_create_set_plus_map_is_the_upsert_idiom() {
    // The canonical Cypher upsert: MERGE then ON CREATE SET n += {props}.
    // Flows through the same apply_set arm as a bare SET.
    let mut writer = WriterSession::open(store(), paths("w-merge-set-map"))
        .await
        .unwrap();
    let outcome = write_q(
        &mut writer,
        "MERGE (a:Person {name: 'Ada'}) ON CREATE SET a += {age: 36, city: 'Quito'} RETURN a",
    )
    .await;
    assert_eq!(outcome.nodes_created, 1);
    let snap = writer.snapshot();
    let nodes = snap.scan_label("Person").await.unwrap();
    let p = &nodes[0].properties;
    assert_eq!(p.get("name"), Some(&CoreValue::Str("Ada".into())));
    assert_eq!(p.get("age"), Some(&CoreValue::I64(36)));
    assert_eq!(p.get("city"), Some(&CoreValue::Str("Quito".into())));
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

// ─────────────────── RFC-026: read-your-own-writes ───────────────────

#[tokio::test]
async fn create_then_match_in_one_statement_reads_own_write() {
    // RFC-026 example 1: a MATCH that follows a CREATE in the same
    // statement must see the just-created node. Before read-your-own-
    // writes this returned zero rows.
    let mut writer = WriterSession::open(store(), paths("w-ryow-create-match"))
        .await
        .unwrap();
    let q = parse(
        "CREATE (a:Person {name: 'Ada'}) \
         WITH a \
         MATCH (p:Person {name: 'Ada'}) \
         RETURN p",
    )
    .unwrap();
    let plan = lower(&q).unwrap();
    let outcome = execute_write(&plan, &mut writer, &Params::new())
        .await
        .unwrap();
    assert_eq!(outcome.nodes_created, 1);
    assert_eq!(
        outcome.rows.len(),
        1,
        "the MATCH must see the node staged by the CREATE in the same statement"
    );
    match outcome.rows[0].get("p") {
        Some(RuntimeValue::Node(n)) => {
            assert_eq!(
                n.properties.get("name"),
                Some(&RuntimeValue::String("Ada".into()))
            );
        }
        other => panic!("expected node p, got {other:?}"),
    }
}

#[tokio::test]
async fn staged_edge_is_traversable_via_overlay_snapshot() {
    // RFC-026 edge overlay at the query boundary: an edge staged by a write
    // (not yet committed) is traversable by a MATCH run against the writer's
    // overlay snapshot — the same path the Bolt transaction handler uses for
    // an in-tx read — while a plain committed snapshot does not see it. The
    // intra-statement `CREATE ... WITH ... MATCH (expand)` form would need the
    // executor to run a read pipeline above a write in one statement, which is
    // a separate, not-yet-supported capability for nodes or edges (RFC-026
    // follow-up), so this exercises the staged-then-traverse path instead.
    let mut writer = WriterSession::open(store(), paths("w-ryow-edge-overlay"))
        .await
        .unwrap();

    // Stage two persons and a KNOWS edge between them; do NOT commit
    // (`execute_write_staged` leaves the batch pending, unlike the
    // auto-committing `execute_write`).
    let create =
        lower(&parse("CREATE (a:Person {name: 'Ada'})-[:KNOWS]->(b:Person {name: 'Bo'})").unwrap())
            .unwrap();
    let outcome = execute_write_staged(&create, &mut writer, &Params::new())
        .await
        .unwrap();
    assert_eq!(outcome.edges_created, 1);

    let match_plan =
        lower(&parse("MATCH (:Person {name: 'Ada'})-[:KNOWS]->(x) RETURN x.name AS name").unwrap())
            .unwrap();

    // Committed snapshot: the staged edge (and its endpoints) are invisible.
    let committed = writer.snapshot();
    let rows = execute(&match_plan, &committed, &Params::new())
        .await
        .unwrap();
    assert!(
        rows.is_empty(),
        "a plain committed snapshot must not see the staged edge, got {rows:?}"
    );
    drop(committed);

    // Overlay snapshot: the staged edge is traversable end-to-end.
    let overlay = writer.overlay_snapshot();
    let rows = execute(&match_plan, &overlay, &Params::new())
        .await
        .unwrap();
    assert_eq!(
        rows.len(),
        1,
        "the overlay snapshot must surface the staged edge"
    );
    assert_eq!(
        rows[0].get("name"),
        Some(&RuntimeValue::String("Bo".into()))
    );
}

#[tokio::test]
async fn merge_after_create_in_one_statement_does_not_duplicate() {
    // RFC-026 example 2: MERGE's match phase must see a node the same
    // statement just created, so it matches instead of creating a
    // duplicate.
    let mut writer = WriterSession::open(store(), paths("w-ryow-merge-create"))
        .await
        .unwrap();
    let q = parse(
        "CREATE (a:Person {name: 'Ada'}) \
         MERGE (b:Person {name: 'Ada'}) \
         RETURN b",
    )
    .unwrap();
    let plan = lower(&q).unwrap();
    let outcome = execute_write(&plan, &mut writer, &Params::new())
        .await
        .unwrap();
    assert_eq!(
        outcome.nodes_created, 1,
        "MERGE must match the staged CREATE, not create a second node"
    );

    // Exactly one Person is durable after commit.
    let snap = writer.snapshot();
    assert_eq!(snap.scan_label("Person").await.unwrap().len(), 1);
}

#[tokio::test]
async fn intra_batch_duplicate_unique_value_is_rejected() {
    // RFC-026: the unique-constraint check reads the overlay, so two
    // creates of the same unique value in one uncommitted statement are
    // caught — the second now sees the first.
    let mut writer = WriterSession::open(store(), paths("w-ryow-unique"))
        .await
        .unwrap();
    let schema = SchemaBuilder::new()
        .label(LabelDef {
            name: "Account".into(),
            properties: vec![PropertyDef::new("email", DataType::Utf8, false)
                .unwrap()
                .with_unique(true)],
        })
        .unwrap()
        .build();
    // Seed one committed Account so the flush is non-empty and persists the
    // unique schema into the manifest (an empty flush is a no-op).
    write_q(&mut writer, "CREATE (:Account {email: 'seed@x.com'})").await;
    writer.flush(schema).await.unwrap();

    let q =
        parse("CREATE (:Account {email: 'dup@x.com'}), (:Account {email: 'dup@x.com'})").unwrap();
    let plan = lower(&q).unwrap();
    let err = execute_write(&plan, &mut writer, &Params::new())
        .await
        .expect_err("duplicate unique value in one batch must be rejected");
    let msg = format!("{err:?}");
    assert!(
        msg.contains("unique"),
        "expected a unique-constraint error, got: {msg}"
    );

    // The failed statement discarded its batch: only the seed remains.
    let snap = writer.snapshot();
    assert_eq!(snap.scan_label("Account").await.unwrap().len(), 1);
}

/// Schema with a single integer unique property.
fn int_unique_schema() -> namidb_core::schema::Schema {
    SchemaBuilder::new()
        .label(LabelDef {
            name: "Account".into(),
            properties: vec![PropertyDef::new("account_no", DataType::Int64, false)
                .unwrap()
                .with_unique(true)],
        })
        .unwrap()
        .build()
}

#[tokio::test]
async fn nonstring_unique_create_rejects_duplicate() {
    // A non-string (Int64) unique property is now enforced on CREATE, not
    // just string properties.
    let mut writer = WriterSession::open(store(), paths("w-unique-int-create"))
        .await
        .unwrap();
    write_q(&mut writer, "CREATE (:Account {account_no: 1})").await;
    writer.flush(int_unique_schema()).await.unwrap();

    // A different value is fine.
    write_q(&mut writer, "CREATE (:Account {account_no: 2})").await;

    // A duplicate of a committed value is rejected.
    let plan = lower(&parse("CREATE (:Account {account_no: 1})").unwrap()).unwrap();
    let err = execute_write(&plan, &mut writer, &Params::new())
        .await
        .expect_err("duplicate integer unique value must be rejected");
    assert!(
        format!("{err:?}").contains("unique"),
        "expected a unique-constraint error, got: {err:?}"
    );

    let snap = writer.snapshot();
    assert_eq!(snap.scan_label("Account").await.unwrap().len(), 2);
}

#[tokio::test]
async fn nonstring_unique_intra_batch_duplicate_rejected() {
    // The non-string check reads the overlay too: two creates of the same
    // integer value in one uncommitted statement are caught.
    let mut writer = WriterSession::open(store(), paths("w-unique-int-batch"))
        .await
        .unwrap();
    write_q(&mut writer, "CREATE (:Account {account_no: 7})").await;
    writer.flush(int_unique_schema()).await.unwrap();

    let plan =
        lower(&parse("CREATE (:Account {account_no: 9}), (:Account {account_no: 9})").unwrap())
            .unwrap();
    let err = execute_write(&plan, &mut writer, &Params::new())
        .await
        .expect_err("duplicate integer value in one batch must be rejected");
    assert!(
        format!("{err:?}").contains("unique"),
        "expected a unique-constraint error, got: {err:?}"
    );

    // The failed batch was discarded: only the committed seed remains.
    let snap = writer.snapshot();
    assert_eq!(snap.scan_label("Account").await.unwrap().len(), 1);
}

#[tokio::test]
async fn nonstring_unique_set_rejects_collision_but_allows_self_update() {
    // SET enforces a non-string unique constraint: moving a node onto another
    // node's value is rejected, while a self-update or a move to a free value
    // is allowed.
    let mut writer = WriterSession::open(store(), paths("w-unique-int-set"))
        .await
        .unwrap();
    write_q(&mut writer, "CREATE (:Account {account_no: 1})").await;
    write_q(&mut writer, "CREATE (:Account {account_no: 2})").await;
    writer.flush(int_unique_schema()).await.unwrap();

    // Collision: account 1 -> 2 (held by another node) is rejected.
    let plan =
        lower(&parse("MATCH (a:Account {account_no: 1}) SET a.account_no = 2").unwrap()).unwrap();
    let err = execute_write(&plan, &mut writer, &Params::new())
        .await
        .expect_err("SET onto another node's unique value must be rejected");
    assert!(
        format!("{err:?}").contains("unique"),
        "expected a unique-constraint error, got: {err:?}"
    );

    // Self-update: account 1 -> 1 is allowed (the node's own value).
    write_q(
        &mut writer,
        "MATCH (a:Account {account_no: 1}) SET a.account_no = 1",
    )
    .await;
    // Move to a free value: account 1 -> 3 is allowed.
    write_q(
        &mut writer,
        "MATCH (a:Account {account_no: 1}) SET a.account_no = 3",
    )
    .await;

    let snap = writer.snapshot();
    let rows = snap.scan_label("Account").await.unwrap();
    assert_eq!(rows.len(), 2, "no node was created or dropped by the SETs");
}

#[tokio::test]
async fn expand_above_write_sees_staged_edge_in_one_statement() {
    // RFC-026 Q1: a traversal (Expand) running directly above a write in the
    // same statement must see the edge that write just staged. Before the fix
    // this errored ("write operators require execute_write...") because the
    // whole Expand-over-CREATE subtree was handed to the read-only walker; now
    // the write executor stages the input, then expands over the overlay.
    let mut writer = WriterSession::open(store(), paths("w-expand-above-write"))
        .await
        .unwrap();
    let q = parse(
        "CREATE (a:Person {name: 'A'})-[:R]->(b:Person {name: 'B'}) \
         WITH a MATCH (a)-[:R]->(x) RETURN x",
    )
    .unwrap();
    let plan = lower(&q).unwrap();
    let outcome = execute_write(&plan, &mut writer, &Params::new())
        .await
        .unwrap();
    assert_eq!(outcome.nodes_created, 2);
    assert_eq!(outcome.edges_created, 1);
    assert_eq!(
        outcome.rows.len(),
        1,
        "the just-staged edge must be traversed by the following MATCH"
    );
    match outcome.rows[0].get("x") {
        Some(RuntimeValue::Node(n)) => match n.properties.get("name") {
            Some(RuntimeValue::String(s)) => {
                assert_eq!(s.as_str(), "B", "x must bind to the created target b")
            }
            other => panic!("expected x.name = 'B', got {other:?}"),
        },
        other => panic!("expected node x, got {other:?}"),
    }

    // And it committed: a fresh snapshot sees the edge.
    let snap = writer.snapshot();
    let rows = execute(
        &lower(&parse("MATCH (:Person)-[:R]->(x) RETURN x").unwrap()).unwrap(),
        &snap,
        &Params::new(),
    )
    .await
    .unwrap();
    assert_eq!(rows.len(), 1, "the edge must persist after commit");
}

#[tokio::test]
async fn foreach_creates_a_node_per_list_element() {
    // FOREACH over a list literal: one CREATE per element.
    let mut writer = WriterSession::open(store(), paths("w-foreach"))
        .await
        .unwrap();
    write_q(
        &mut writer,
        "FOREACH (x IN [10, 20, 30] | CREATE (:Item {v: x}))",
    )
    .await;

    let snap = writer.snapshot();
    let plan = lower(&parse("MATCH (n:Item) RETURN n.v AS v ORDER BY v").unwrap()).unwrap();
    let rows = execute(&plan, &snap, &Params::new()).await.unwrap();
    let vs: Vec<i64> = rows
        .iter()
        .filter_map(|r| match r.get("v") {
            Some(RuntimeValue::Integer(n)) => Some(*n),
            _ => None,
        })
        .collect();
    assert_eq!(vs, vec![10, 20, 30], "one Item per list element");
}

#[tokio::test]
async fn foreach_runs_per_matched_row_and_preserves_cardinality() {
    // For each matched Person, FOREACH creates one Tag per list element; the
    // RETURN after FOREACH still sees one row per Person (pass-through).
    let mut writer = WriterSession::open(store(), paths("w-foreach-card"))
        .await
        .unwrap();
    write_q(&mut writer, "CREATE (:Person {name: 'a'})").await;
    write_q(&mut writer, "CREATE (:Person {name: 'b'})").await;

    let plan = optimize(
        lower(
            &parse(
                "MATCH (p:Person) \
                 FOREACH (t IN [1, 2] | CREATE (:Tag {owner: p.name, t: t})) \
                 RETURN p.name AS name ORDER BY name",
            )
            .unwrap(),
        )
        .unwrap(),
        &StatsCatalog::empty(),
    );
    let outcome = execute_write(&plan, &mut writer, &Params::new())
        .await
        .unwrap();
    // FOREACH passes the 2 Person rows through unchanged.
    assert_eq!(outcome.rows.len(), 2, "one row per matched Person");

    // 2 Persons × 2 list elements = 4 Tag nodes.
    let snap = writer.snapshot();
    let count = lower(&parse("MATCH (n:Tag) RETURN n").unwrap()).unwrap();
    let tags = execute(&count, &snap, &Params::new()).await.unwrap();
    assert_eq!(tags.len(), 4, "one Tag per (Person × element)");
}

#[tokio::test]
async fn foreach_read_modify_write_accumulates_across_iterations() {
    // A read-modify-write on a node bound by the outer MATCH accumulates across
    // FOREACH iterations: `SET c.n = c.n + i` over [1,2,3] leaves n = 0+1+2+3 = 6
    // (each iteration sees the previous iteration's write, not the pre-loop row).
    let mut writer = WriterSession::open(store(), paths("w-foreach-set"))
        .await
        .unwrap();
    write_q(&mut writer, "CREATE (:Counter {name: 'c', n: 0})").await;
    write_q(
        &mut writer,
        "MATCH (c:Counter {name: 'c'}) FOREACH (i IN [1, 2, 3] | SET c.n = c.n + i)",
    )
    .await;

    let snap = writer.snapshot();
    let plan = lower(&parse("MATCH (c:Counter) RETURN c.n AS n").unwrap()).unwrap();
    let rows = execute(&plan, &snap, &Params::new()).await.unwrap();
    assert!(matches!(rows[0].get("n"), Some(RuntimeValue::Integer(6))));
}

#[tokio::test]
async fn foreach_body_rejects_non_update_clause() {
    // A read clause (RETURN) inside a FOREACH body is rejected at lowering.
    let parsed = parse("FOREACH (x IN [1] | RETURN x)").unwrap();
    assert!(
        lower(&parsed).is_err(),
        "FOREACH body may only contain update clauses"
    );
}

#[tokio::test]
async fn correlated_call_subquery_writes_per_outer_row() {
    // `MATCH (a) CALL { WITH a CREATE (:City {owner: a.name}) }` runs the write
    // once per matched Person, creating one City each.
    let mut writer = WriterSession::open(store(), paths("w-corr-call-write"))
        .await
        .unwrap();
    write_q(&mut writer, "CREATE (:Person {name: 'a'})").await;
    write_q(&mut writer, "CREATE (:Person {name: 'b'})").await;

    let plan = optimize(
        lower(
            &parse(
                "MATCH (p:Person) \
                 CALL { WITH p CREATE (:City {owner: p.name}) } \
                 RETURN p.name AS name ORDER BY name",
            )
            .unwrap(),
        )
        .unwrap(),
        &StatsCatalog::empty(),
    );
    let outcome = execute_write(&plan, &mut writer, &Params::new())
        .await
        .unwrap();
    // One output row per Person (pass-through), two Cities created.
    assert_eq!(outcome.rows.len(), 2);

    let snap = writer.snapshot();
    let cities = lower(&parse("MATCH (c:City) RETURN c.owner AS o ORDER BY o").unwrap()).unwrap();
    let rows = execute(&cities, &snap, &Params::new()).await.unwrap();
    let owners: Vec<&str> = rows
        .iter()
        .filter_map(|r| match r.get("o") {
            Some(RuntimeValue::String(s)) => Some(s.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(owners, vec!["a", "b"], "one City per Person, correlated");
}

#[tokio::test]
async fn composite_unique_create_rejects_duplicate_tuple() {
    let mut writer = WriterSession::open(store(), paths("w-composite-create"))
        .await
        .unwrap();
    write_q(&mut writer, "CREATE (:Person {name: 'Ann', age: 30})").await;
    // Register a composite uniqueness constraint over (name, age).
    let props = vec!["name".to_string(), "age".to_string()];
    writer
        .create_unique_constraint_named(None, "Person", &props, false)
        .await
        .unwrap();

    // Same name, different age → distinct tuple → allowed.
    write_q(&mut writer, "CREATE (:Person {name: 'Ann', age: 31})").await;
    // Same age, different name → allowed.
    write_q(&mut writer, "CREATE (:Person {name: 'Bob', age: 30})").await;

    // Exact (name, age) duplicate → rejected.
    let plan = lower(&parse("CREATE (:Person {name: 'Ann', age: 30})").unwrap()).unwrap();
    let err = execute_write(&plan, &mut writer, &Params::new())
        .await
        .expect_err("duplicate (name, age) tuple must be rejected");
    assert!(
        format!("{err:?}").contains("composite unique"),
        "expected a composite-unique error, got: {err:?}"
    );

    // A node missing one of the constraint's properties is exempt.
    write_q(&mut writer, "CREATE (:Person {name: 'Cara'})").await;
    write_q(&mut writer, "CREATE (:Person {name: 'Cara'})").await;

    let snap = writer.snapshot();
    assert_eq!(snap.scan_label("Person").await.unwrap().len(), 5);
}

#[tokio::test]
async fn composite_unique_set_rejects_collision_allows_self_update() {
    let mut writer = WriterSession::open(store(), paths("w-composite-set"))
        .await
        .unwrap();
    write_q(&mut writer, "CREATE (:Person {name: 'Ann', age: 30})").await;
    write_q(&mut writer, "CREATE (:Person {name: 'Bob', age: 30})").await;
    let props = vec!["name".to_string(), "age".to_string()];
    writer
        .create_unique_constraint_named(None, "Person", &props, false)
        .await
        .unwrap();

    // Moving Bob onto Ann's (name, age) tuple is rejected.
    let plan = lower(&parse("MATCH (p:Person {name: 'Bob'}) SET p.name = 'Ann'").unwrap()).unwrap();
    let err = execute_write(&plan, &mut writer, &Params::new())
        .await
        .expect_err("SET onto another node's composite tuple must be rejected");
    assert!(
        format!("{err:?}").contains("composite unique"),
        "expected a composite-unique error, got: {err:?}"
    );

    // A self-update (writing the same value) is allowed.
    write_q(&mut writer, "MATCH (p:Person {name: 'Ann'}) SET p.age = 30").await;
}

#[tokio::test]
async fn composite_unique_add_label_rejects_collision() {
    let mut writer = WriterSession::open(store(), paths("w-composite-addlabel"))
        .await
        .unwrap();
    write_q(&mut writer, "CREATE (:Person {a: 1, b: 2})").await;
    let props = vec!["a".to_string(), "b".to_string()];
    writer
        .create_unique_constraint_named(None, "Person", &props, false)
        .await
        .unwrap();

    // A :Tmp node with the same (a, b) is fine — the constraint is on :Person.
    write_q(&mut writer, "CREATE (:Tmp {a: 1, b: 2})").await;

    // Promoting it to :Person would create a duplicate tuple → rejected.
    let plan = lower(&parse("MATCH (x:Tmp) SET x:Person").unwrap()).unwrap();
    let err = execute_write(&plan, &mut writer, &Params::new())
        .await
        .expect_err("gaining :Person must run the composite uniqueness check");
    assert!(
        format!("{err:?}").contains("composite unique"),
        "got: {err:?}"
    );

    // A :Tmp node with a distinct tuple promotes cleanly.
    write_q(&mut writer, "CREATE (:Tmp {a: 9, b: 9})").await;
    write_q(&mut writer, "MATCH (x:Tmp {a: 9}) SET x:Person").await;
}

// ─────────────── unique-constraint fast path (finding 37) ───────────────
//
// Uniqueness checks must probe the writer's transactional unique-value
// index — one label scan per (label, property-set) per batch, O(1) per row
// after that — instead of re-scanning the label for every written row. The
// tests below assert the path taken via the writer's index counters, per
// the parity-test invariant (equal results alone are trivially satisfied
// by the scan fallback).

#[tokio::test]
async fn string_unique_checks_probe_index_once_in_multi_label_deployment() {
    // Multi-label deployment: an unrelated label's SST used to demote the
    // string fast path to a full label scan per written row.
    let mut writer = WriterSession::open(store(), paths("w-unique-idx-multilabel"))
        .await
        .unwrap();
    write_q(&mut writer, "CREATE (:Widget {sku: 'w-1'})").await;
    let schema = SchemaBuilder::new()
        .label(LabelDef {
            name: "Account".into(),
            properties: vec![PropertyDef::new("email", DataType::Utf8, false)
                .unwrap()
                .with_unique(true)],
        })
        .unwrap()
        .build();
    // Persist the schema AND put the Widget rows into an SST.
    writer.flush(schema).await.unwrap();

    let scans_before = writer.unique_index().populate_scans();
    let outcome = write_q(
        &mut writer,
        "CREATE (:Account {email: 'a@x'}), (:Account {email: 'b@x'}), \
         (:Account {email: 'c@x'})",
    )
    .await;
    assert_eq!(outcome.nodes_created, 3);
    assert_eq!(
        writer.unique_index().populate_scans() - scans_before,
        1,
        "one populating scan for the whole statement, not one per row"
    );
    assert!(
        writer.unique_index().probes() >= 3,
        "every row's check must go through the index probe"
    );

    // Conflict against a committed value still surfaces through the index.
    let plan = lower(&parse("CREATE (:Account {email: 'a@x'})").unwrap()).unwrap();
    let err = execute_write(&plan, &mut writer, &Params::new())
        .await
        .expect_err("duplicate string unique value must be rejected");
    assert!(format!("{err:?}").contains("unique"), "got: {err:?}");
}

#[tokio::test]
async fn integer_unique_checks_probe_index_conflict_and_non_conflict() {
    let mut writer = WriterSession::open(store(), paths("w-unique-idx-int"))
        .await
        .unwrap();
    write_q(&mut writer, "CREATE (:Account {account_no: 1})").await;
    writer.flush(int_unique_schema()).await.unwrap();

    // Non-conflict: a fresh value is accepted through the index probe.
    let scans_before = writer.unique_index().populate_scans();
    let probes_before = writer.unique_index().probes();
    write_q(&mut writer, "CREATE (:Account {account_no: 2})").await;
    assert_eq!(writer.unique_index().populate_scans() - scans_before, 1);
    assert!(writer.unique_index().probes() > probes_before);

    // Conflict: a duplicate integer is rejected through the index probe.
    let probes_before = writer.unique_index().probes();
    let plan = lower(&parse("CREATE (:Account {account_no: 1})").unwrap()).unwrap();
    let err = execute_write(&plan, &mut writer, &Params::new())
        .await
        .expect_err("duplicate integer unique value must be rejected");
    assert!(format!("{err:?}").contains("unique"), "got: {err:?}");
    assert!(
        writer.unique_index().probes() > probes_before,
        "the conflict must be found by an index probe, not a scan"
    );
}

#[tokio::test]
async fn composite_unique_checks_probe_index_conflict_and_non_conflict() {
    let mut writer = WriterSession::open(store(), paths("w-unique-idx-composite"))
        .await
        .unwrap();
    write_q(&mut writer, "CREATE (:Person {name: 'Ann', age: 30})").await;
    let props = vec!["name".to_string(), "age".to_string()];
    writer
        .create_unique_constraint_named(None, "Person", &props, false)
        .await
        .unwrap();

    // Non-conflict: same name, different age — one scan, probed per row.
    let scans_before = writer.unique_index().populate_scans();
    let probes_before = writer.unique_index().probes();
    let outcome = write_q(
        &mut writer,
        "CREATE (:Person {name: 'Ann', age: 31}), (:Person {name: 'Bob', age: 30})",
    )
    .await;
    assert_eq!(outcome.nodes_created, 2);
    assert_eq!(writer.unique_index().populate_scans() - scans_before, 1);
    assert!(writer.unique_index().probes() >= probes_before + 2);

    // Conflict: the exact committed tuple is rejected through the probe.
    let probes_before = writer.unique_index().probes();
    let plan = lower(&parse("CREATE (:Person {name: 'Ann', age: 30})").unwrap()).unwrap();
    let err = execute_write(&plan, &mut writer, &Params::new())
        .await
        .expect_err("duplicate composite tuple must be rejected");
    assert!(
        format!("{err:?}").contains("composite unique"),
        "got: {err:?}"
    );
    assert!(writer.unique_index().probes() > probes_before);
}

#[tokio::test]
async fn set_can_reuse_unique_value_freed_earlier_in_same_batch() {
    // RYOW inside one uncommitted transaction: a SET that moves a node off
    // its unique value frees it for a later SET in the same batch; the
    // check sees the staged state, not the committed one.
    let mut writer = WriterSession::open(store(), paths("w-unique-ryow-freed"))
        .await
        .unwrap();
    write_q(&mut writer, "CREATE (:Account {account_no: 1})").await;
    write_q(&mut writer, "CREATE (:Account {account_no: 2})").await;
    writer.flush(int_unique_schema()).await.unwrap();

    // Statement 1 (staged, uncommitted): account 1 → 3, freeing value 1.
    let plan =
        lower(&parse("MATCH (a:Account {account_no: 1}) SET a.account_no = 3").unwrap()).unwrap();
    execute_write_staged(&plan, &mut writer, &Params::new())
        .await
        .unwrap();
    // Statement 2 (same batch): account 2 → 1 must be allowed, because the
    // staged statement above freed the value.
    let plan =
        lower(&parse("MATCH (a:Account {account_no: 2}) SET a.account_no = 1").unwrap()).unwrap();
    execute_write_staged(&plan, &mut writer, &Params::new())
        .await
        .expect("value freed earlier in the batch must be reusable");
    writer.commit_batch().await.unwrap();

    let snap = writer.snapshot();
    let mut values: Vec<i64> = snap
        .scan_label("Account")
        .await
        .unwrap()
        .iter()
        .filter_map(|v| match v.properties.get("account_no") {
            Some(CoreValue::I64(n)) => Some(*n),
            _ => None,
        })
        .collect();
    values.sort_unstable();
    assert_eq!(values, vec![1, 3]);

    // Negative control: moving onto a value that is STILL held is rejected.
    let plan =
        lower(&parse("MATCH (a:Account {account_no: 3}) SET a.account_no = 1").unwrap()).unwrap();
    let err = execute_write(&plan, &mut writer, &Params::new())
        .await
        .expect_err("value still held by another node must be rejected");
    assert!(format!("{err:?}").contains("unique"), "got: {err:?}");
}

#[tokio::test]
async fn bulk_create_under_unique_constraint_pays_one_scan_not_one_per_row() {
    // Smoke test for the O(N²) fix: 2k rows under a unique constraint in a
    // single statement must populate the index once and probe per row. The
    // wall-clock bound is a coarse sanity net, not a benchmark.
    let mut writer = WriterSession::open(store(), paths("w-unique-bulk"))
        .await
        .unwrap();
    write_q(&mut writer, "CREATE (:Account {account_no: 0})").await;
    writer.flush(int_unique_schema()).await.unwrap();

    let scans_before = writer.unique_index().populate_scans();
    let started = std::time::Instant::now();
    let outcome = write_q(
        &mut writer,
        "UNWIND range(1, 2000) AS i CREATE (:Account {account_no: i})",
    )
    .await;
    let elapsed = started.elapsed();
    assert_eq!(outcome.nodes_created, 2000);
    assert_eq!(
        writer.unique_index().populate_scans() - scans_before,
        1,
        "bulk write must not re-scan the label per row"
    );
    assert!(writer.unique_index().probes() >= 2000);
    assert!(
        elapsed < std::time::Duration::from_secs(60),
        "2k constrained creates took {elapsed:?}"
    );

    // The constraint still holds after the bulk load.
    let plan = lower(&parse("CREATE (:Account {account_no: 1234})").unwrap()).unwrap();
    let err = execute_write(&plan, &mut writer, &Params::new())
        .await
        .expect_err("duplicate after bulk load must be rejected");
    assert!(format!("{err:?}").contains("unique"), "got: {err:?}");
}

#[tokio::test]
async fn merge_unique_unwind_500_uses_one_index_population_across_commit_and_flush() {
    // Regression for the legal-graph loader shape. MERGE's implicit match
    // used to call scan_label once per UNWIND row (O(store_size * rows)).
    // The path counters are the assertion: equal final rows alone would also
    // pass with the slow scan implementation.
    let mut writer = WriterSession::open(store(), paths("w-merge-unique-unwind-500"))
        .await
        .unwrap();
    write_q(&mut writer, "CREATE (:Account {account_no: 0})").await;
    writer.flush(int_unique_schema()).await.unwrap();

    let scans_before = writer.unique_index().populate_scans();
    let probes_before = writer.unique_index().probes();
    let query = lower(
        &parse(
            "UNWIND range(1, 500) AS i \
             MERGE (a:Account {account_no: i}) \
             SET a.payload = i",
        )
        .unwrap(),
    )
    .unwrap();
    let first = execute_write(&query, &mut writer, &Params::new())
        .await
        .unwrap();
    assert_eq!(first.nodes_created, 500);
    assert_eq!(
        writer.unique_index().populate_scans() - scans_before,
        1,
        "the 500-row MERGE must populate its unique index once"
    );
    assert!(
        writer.unique_index().probes() >= probes_before + 500,
        "every implicit MERGE match must probe the unique index"
    );

    // Both auto-commit and a physical memtable→SST flush preserve the
    // transactional index: neither changes logical node content.
    writer.flush(int_unique_schema()).await.unwrap();
    let replay_probes = writer.unique_index().probes();
    let replay = execute_write(&query, &mut writer, &Params::new())
        .await
        .unwrap();
    assert_eq!(replay.nodes_created, 0, "replay must be idempotent");
    assert!(
        writer.unique_index().probes() >= replay_probes + 500,
        "existing-key MERGE rows must take the probe path too"
    );
    assert_eq!(
        writer.unique_index().populate_scans() - scans_before,
        1,
        "successful commit + flush must not trigger another population scan"
    );
}

#[tokio::test]
async fn merge_numeric_unique_key_preserves_cross_type_cypher_equality() {
    // Runtime equality treats integer 1 and float 1.0 as equal, whereas the
    // storage uniqueness key intentionally keeps their encodings distinct.
    // A negative numeric unique probe must not make MERGE create a duplicate.
    let mut writer = WriterSession::open(store(), paths("w-merge-unique-numeric-equality"))
        .await
        .unwrap();
    write_q(&mut writer, "CREATE (:Account {account_no: 1})").await;
    writer.flush(int_unique_schema()).await.unwrap();

    let probes_before = writer.unique_index().probes();
    let matched = write_q(
        &mut writer,
        "MERGE (a:Account {account_no: 1.0}) \
         ON MATCH SET a.seen = true RETURN a",
    )
    .await;
    assert_eq!(matched.nodes_created, 0);
    assert_eq!(matched.rows.len(), 1);
    assert!(
        writer.unique_index().probes() >= probes_before + 2,
        "numeric tuples must probe both strict I64/F64 encodings"
    );
    let accounts = writer.snapshot().scan_label("Account").await.unwrap();
    assert_eq!(accounts.len(), 1);
    assert_eq!(
        accounts[0].properties.get("seen"),
        Some(&CoreValue::Bool(true))
    );
}

#[tokio::test]
async fn relationship_create_rejects_reserved_id_inline_and_in_spread() {
    for (name, query, params) in [
        (
            "inline",
            "CREATE (:A)-[:R {_id: 'edge-id'}]->(:B)",
            Params::new(),
        ),
        (
            "spread",
            "CREATE (:A)-[:R $props]->(:B)",
            Params::from([(
                "props".into(),
                RuntimeValue::Map(BTreeMap::from([(
                    "_id".into(),
                    RuntimeValue::String("edge-id".into()),
                )])),
            )]),
        ),
    ] {
        let mut writer = WriterSession::open(store(), paths(&format!("w-rel-create-id-{name}")))
            .await
            .unwrap();
        let plan = lower(&parse(query).unwrap()).unwrap();
        let err = execute_write(&plan, &mut writer, &params)
            .await
            .expect_err("relationships have no user-visible _id slot");
        assert!(
            format!("{err:?}").contains("_id is not valid on a relationship CREATE"),
            "unexpected error for {name}: {err:?}"
        );
    }
}

#[tokio::test]
async fn merge_uses_fully_covered_composite_key_on_secondary_label() {
    // The constraint belongs to the SECOND label in the pattern. Candidate
    // selection must use Account(tenant,key), then apply LegalEntity as a
    // residual label check.
    let mut writer = WriterSession::open(store(), paths("w-merge-composite-multilabel"))
        .await
        .unwrap();
    write_q(
        &mut writer,
        "CREATE (:LegalEntity:Account {tenant: 't1', key: 'k1', marker: 'old'})",
    )
    .await;
    let mut schema = SchemaBuilder::new()
        .label(LabelDef {
            name: "LegalEntity".into(),
            properties: vec![],
        })
        .unwrap()
        .label(LabelDef {
            name: "Account".into(),
            properties: vec![
                PropertyDef::new("tenant", DataType::Utf8, false).unwrap(),
                PropertyDef::new("key", DataType::Utf8, false).unwrap(),
            ],
        })
        .unwrap()
        .build();
    schema.constraints.push(Constraint {
        name: "uniq_account_tenant_key".into(),
        label: "Account".into(),
        properties: vec!["tenant".into(), "key".into()],
        kind: ConstraintKind::Unique,
    });
    writer.flush(schema).await.unwrap();

    let scans_before = writer.unique_index().populate_scans();
    let probes_before = writer.unique_index().probes();
    let matched = write_q(
        &mut writer,
        "MERGE (n:LegalEntity:Account {tenant: 't1', key: 'k1'}) \
         ON MATCH SET n.marker = 'seen' RETURN n",
    )
    .await;
    assert_eq!(matched.nodes_created, 0);
    assert_eq!(matched.rows.len(), 1);
    assert_eq!(writer.unique_index().populate_scans() - scans_before, 1);
    assert!(
        writer.unique_index().probes() > probes_before,
        "composite MERGE lookup must probe the transactional index"
    );
}

#[tokio::test]
async fn merge_node_parameter_map_participates_in_match_and_unique_lookup() {
    // `properties_spread` used to be discarded by find_merge_matches, so this
    // query matched both Account nodes instead of only key='b'.
    let mut writer = WriterSession::open(store(), paths("w-merge-node-spread"))
        .await
        .unwrap();
    write_q(
        &mut writer,
        "CREATE (:Account {key: 'a', payload: 1}), \
         (:Account {key: 'b', payload: 2})",
    )
    .await;
    let schema = SchemaBuilder::new()
        .label(LabelDef {
            name: "Account".into(),
            properties: vec![
                PropertyDef::new("key", DataType::Utf8, false)
                    .unwrap()
                    .with_unique(true),
                PropertyDef::new("payload", DataType::Int64, false).unwrap(),
            ],
        })
        .unwrap()
        .build();
    writer.flush(schema).await.unwrap();

    let plan =
        lower(&parse("MERGE (n:Account $props) ON MATCH SET n.hit = true RETURN n").unwrap())
            .unwrap();
    let mut params = Params::new();
    params.insert(
        "props".into(),
        RuntimeValue::Map(BTreeMap::from([
            ("key".into(), RuntimeValue::String("b".into())),
            ("payload".into(), RuntimeValue::Integer(2)),
        ])),
    );
    let probes_before = writer.unique_index().probes();
    let outcome = execute_write(&plan, &mut writer, &params).await.unwrap();
    assert_eq!(outcome.nodes_created, 0);
    assert_eq!(
        outcome.rows.len(),
        1,
        "spread residuals must select one node"
    );
    assert!(
        writer.unique_index().probes() > probes_before,
        "a unique key supplied through $props must use the probe path"
    );
    let snap = writer.snapshot();
    let nodes = snap.scan_label("Account").await.unwrap();
    let hit_keys: Vec<_> = nodes
        .iter()
        .filter(|node| node.properties.get("hit") == Some(&CoreValue::Bool(true)))
        .map(|node| node.properties.get("key").cloned())
        .collect();
    assert_eq!(hit_keys, vec![Some(CoreValue::Str("b".into()))]);
}

#[tokio::test]
async fn merge_non_unique_index_uses_posting_list_then_residual_filter() {
    let mut writer = WriterSession::open(store(), paths("w-merge-equality-index"))
        .await
        .unwrap();
    write_q(
        &mut writer,
        "CREATE (:Doc {group: 'legal', key: 'a'}), \
         (:Doc {group: 'legal', key: 'b'}), \
         (:Doc {group: 'other', key: 'c'})",
    )
    .await;
    let schema = SchemaBuilder::new()
        .label(LabelDef {
            name: "Doc".into(),
            properties: vec![
                PropertyDef::new("group", DataType::Utf8, false)
                    .unwrap()
                    .with_indexed(true),
                PropertyDef::new("key", DataType::Utf8, false).unwrap(),
            ],
        })
        .unwrap()
        .build();
    writer.flush(schema).await.unwrap();

    let lookups_before = writer.property_index_cache().equality_lookup_calls();
    let outcome = write_q(
        &mut writer,
        "MERGE (d:Doc {group: 'legal', key: 'b'}) \
         ON MATCH SET d.hit = true RETURN d",
    )
    .await;
    assert_eq!(outcome.nodes_created, 0);
    assert_eq!(
        outcome.rows.len(),
        1,
        "residual key must reduce two postings"
    );
    assert!(
        writer.property_index_cache().equality_lookup_calls() > lookups_before,
        "MERGE must call lookup_nodes_by_property, not scan_label"
    );
}

#[tokio::test]
async fn merge_node_by_explicit_id_uses_direct_lookup_and_is_idempotent() {
    let mut writer = WriterSession::open(store(), paths("w-merge-explicit-id"))
        .await
        .unwrap();
    let created = write_q(
        &mut writer,
        "CREATE (n:Account {key: 'a', payload: 1}) RETURN n",
    )
    .await;
    let id = match created.rows[0].get("n") {
        Some(RuntimeValue::Node(node)) => node.id,
        other => panic!("expected created node, got {other:?}"),
    };
    let plan = lower(&parse("MERGE (n:Account {_id: $id, key: 'a'}) RETURN n").unwrap()).unwrap();
    let mut params = Params::new();
    params.insert("id".into(), RuntimeValue::String(id.to_string()));
    let scans_before = writer.unique_index().populate_scans();
    let outcome = execute_write(&plan, &mut writer, &params).await.unwrap();
    assert_eq!(outcome.nodes_created, 0);
    assert_eq!(outcome.rows.len(), 1);
    assert_eq!(
        writer.unique_index().populate_scans(),
        scans_before,
        "_id must bypass label/unique-index population and point-read the node"
    );
}

#[tokio::test]
async fn merge_relationship_parameter_map_participates_in_match() {
    let mut writer = WriterSession::open(store(), paths("w-merge-rel-spread"))
        .await
        .unwrap();
    write_q(
        &mut writer,
        "CREATE (a:A {name: 'a'})-[:R {kind: 'old'}]->(b:B {name: 'b'})",
    )
    .await;
    let plan = lower(
        &parse(
            "MATCH (a:A {name: 'a'}), (b:B {name: 'b'}) \
             MERGE (a)-[r:R $relprops]->(b) RETURN r",
        )
        .unwrap(),
    )
    .unwrap();
    let mut params = Params::new();
    params.insert(
        "relprops".into(),
        RuntimeValue::Map(BTreeMap::from([(
            "kind".into(),
            RuntimeValue::String("new".into()),
        )])),
    );
    let replaced = execute_write(&plan, &mut writer, &params).await.unwrap();
    assert_eq!(
        replaced.edges_created, 1,
        "different spread properties must take MERGE's create branch"
    );
    let replay = execute_write(&plan, &mut writer, &params).await.unwrap();
    assert_eq!(
        replay.edges_created, 0,
        "same spread properties must match on replay"
    );
}
