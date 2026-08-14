//! End-to-end tests for write clauses.
//!
//! Exercises CREATE / MATCH+CREATE / SET / REMOVE / DELETE / DETACH DELETE
//! / MERGE-match / MERGE-create against a fresh `WriterSession`. After each
//! mutation the test snapshots the writer to confirm durability.

use std::collections::BTreeMap;
use std::sync::Arc;

use namidb_core::id::{NamespaceId, NodeId};
use namidb_core::schema::{
    Constraint, ConstraintKind, DataType, EdgeTypeDef, LabelDef, PropertyDef, SchemaBuilder,
};
use namidb_core::value::Value as CoreValue;
use namidb_storage::{
    AdjacencyCache, EdgeWriteRecord, NamespacePaths, NodeWriteRecord, SessionCaches, SstCache,
    WriterSession,
};
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
async fn unwind_repeated_node_set_reads_latest_staged_value() {
    let mut writer = WriterSession::open(store(), paths("w-set-unwind-ryow"))
        .await
        .unwrap();
    write_q(&mut writer, "CREATE (:Account {key: 'same', counter: 0})").await;

    let outcome = write_q(
        &mut writer,
        "MATCH (n:Account {key: 'same'}) \
         UNWIND range(1, 3) AS i \
         SET n.counter = n.counter + 1 \
         RETURN n",
    )
    .await;
    let observed: Vec<i64> = outcome
        .rows
        .iter()
        .map(|row| match row.get("n") {
            Some(RuntimeValue::Node(node)) => match node.properties.get("counter") {
                Some(RuntimeValue::Integer(value)) => *value,
                other => panic!("expected integer counter, got {other:?}"),
            },
            other => panic!("expected node, got {other:?}"),
        })
        .collect();
    assert_eq!(observed, vec![1, 2, 3]);
    let stored = writer.snapshot().scan_label("Account").await.unwrap();
    assert_eq!(
        stored[0].properties.get("counter"),
        Some(&CoreValue::I64(3))
    );
}

#[tokio::test]
async fn unwind_repeated_node_map_merge_keeps_prior_patches() {
    let mut writer = WriterSession::open(store(), paths("w-set-map-unwind-ryow"))
        .await
        .unwrap();
    write_q(&mut writer, "CREATE (:Account {key: 'same', base: 1})").await;

    let outcome = write_q(
        &mut writer,
        "MATCH (n:Account {key: 'same'}) \
         UNWIND [{left: 2}, {right: 3}] AS patch \
         SET n += patch \
         RETURN n",
    )
    .await;
    let final_node = match outcome.rows.last().and_then(|row| row.get("n")) {
        Some(RuntimeValue::Node(node)) => node,
        other => panic!("expected final node, got {other:?}"),
    };
    assert_eq!(
        final_node.properties.get("left"),
        Some(&RuntimeValue::Integer(2))
    );
    assert_eq!(
        final_node.properties.get("right"),
        Some(&RuntimeValue::Integer(3))
    );
    let stored = writer.snapshot().scan_label("Account").await.unwrap();
    assert_eq!(stored[0].properties.get("base"), Some(&CoreValue::I64(1)));
    assert_eq!(stored[0].properties.get("left"), Some(&CoreValue::I64(2)));
    assert_eq!(stored[0].properties.get("right"), Some(&CoreValue::I64(3)));
}

#[tokio::test]
async fn remove_through_two_aliases_does_not_resurrect_prior_removal() {
    let mut writer = WriterSession::open(store(), paths("w-remove-alias-ryow"))
        .await
        .unwrap();
    write_q(
        &mut writer,
        "CREATE (:Account {key: 'same', left: 1, right: 2, keep: 3})",
    )
    .await;

    let outcome = write_q(
        &mut writer,
        "MATCH (n:Account {key: 'same'}) \
         MATCH (m:Account {key: 'same'}) \
         REMOVE n.left, m.right \
         RETURN n, m",
    )
    .await;
    let node = match outcome.rows[0].get("n") {
        Some(RuntimeValue::Node(node)) => node,
        other => panic!("expected node alias n, got {other:?}"),
    };
    assert!(!node.properties.contains_key("left"));
    assert!(!node.properties.contains_key("right"));
    assert_eq!(node.properties.get("keep"), Some(&RuntimeValue::Integer(3)));
    assert_eq!(
        outcome.rows[0].get("n"),
        outcome.rows[0].get("m"),
        "aliases of one physical node must remain coherent"
    );

    let stored = writer.snapshot().scan_label("Account").await.unwrap();
    assert!(!stored[0].properties.contains_key("left"));
    assert!(!stored[0].properties.contains_key("right"));
    assert_eq!(stored[0].properties.get("keep"), Some(&CoreValue::I64(3)));
}

#[tokio::test]
async fn unwind_relationship_merge_set_refreshes_outer_alias() {
    let mut writer = WriterSession::open(store(), paths("w-rel-merge-outer-alias-ryow"))
        .await
        .unwrap();
    write_q(
        &mut writer,
        "CREATE (:Person {key: 'a'})-[:KNOWS {counter: 0}]->(:Person {key: 'b'})",
    )
    .await;

    let outcome = write_q(
        &mut writer,
        "MATCH (a:Person {key: 'a'})-[z:KNOWS]->(b:Person {key: 'b'}) \
         UNWIND range(1, 2) AS i \
         MERGE (a)-[r:KNOWS]->(b) \
         ON MATCH SET z.counter = z.counter + 1 \
         RETURN r, z",
    )
    .await;
    assert_eq!(outcome.edges_created, 0);
    let counters: Vec<(i64, i64)> = outcome
        .rows
        .iter()
        .map(|row| {
            let counter = |alias: &str| match row.get(alias) {
                Some(RuntimeValue::Rel(rel)) => match rel.properties.get("counter") {
                    Some(RuntimeValue::Integer(value)) => *value,
                    other => panic!("expected integer counter on {alias}, got {other:?}"),
                },
                other => panic!("expected relationship alias {alias}, got {other:?}"),
            };
            (counter("r"), counter("z"))
        })
        .collect();
    assert_eq!(counters, vec![(1, 1), (2, 2)]);

    let people = writer.snapshot().scan_label("Person").await.unwrap();
    let src = people
        .iter()
        .find(|node| node.properties.get("key") == Some(&CoreValue::Str("a".into())))
        .unwrap()
        .id;
    let edges = writer
        .snapshot()
        .out_edges("KNOWS", src)
        .await
        .unwrap()
        .edges;
    assert_eq!(edges[0].properties.get("counter"), Some(&CoreValue::I64(2)));
}

#[tokio::test]
async fn unwind_repeated_relationship_map_merge_keeps_prior_patches() {
    let mut writer = WriterSession::open(store(), paths("w-rel-set-map-unwind-ryow"))
        .await
        .unwrap();
    write_q(
        &mut writer,
        "CREATE (:Person {key: 'a'})-[:KNOWS {base: 1}]->(:Person {key: 'b'})",
    )
    .await;

    let outcome = write_q(
        &mut writer,
        "MATCH (:Person {key: 'a'})-[r:KNOWS]->(:Person {key: 'b'}) \
         UNWIND [{left: 2}, {right: 3}] AS patch \
         SET r += patch \
         RETURN r",
    )
    .await;
    let final_rel = match outcome.rows.last().and_then(|row| row.get("r")) {
        Some(RuntimeValue::Rel(rel)) => rel,
        other => panic!("expected final relationship, got {other:?}"),
    };
    assert_eq!(
        final_rel.properties.get("left"),
        Some(&RuntimeValue::Integer(2))
    );
    assert_eq!(
        final_rel.properties.get("right"),
        Some(&RuntimeValue::Integer(3))
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
async fn persisted_relationship_merge_is_sparse_and_preserves_properties_on_match() {
    // A relationship MERGE is a point probe from its already-bound source.
    // Routing that probe through the whole-type CSR rebuilds every persisted
    // edge after each manifest-changing loader batch. Slim CSR also omits
    // properties, which makes a persisted `{weight: 1}` edge look absent and
    // makes `ON MATCH SET` overwrite rather than extend its property map.
    let adjacency = Arc::new(AdjacencyCache::new(64 * 1024 * 1024));
    let sst_cache = SstCache::new(64 * 1024 * 1024);
    let mut caches = SessionCaches::none();
    caches.sst_cache = Some(sst_cache.clone());
    caches.adjacency_cache = Some(Arc::clone(&adjacency));
    let mut writer = WriterSession::open_with_caches(store(), paths("w-merge-rel-sst"), caches)
        .await
        .unwrap();

    let schema = SchemaBuilder::new()
        .label(LabelDef {
            name: "Person".into(),
            properties: vec![PropertyDef::new("key", DataType::Utf8, false)
                .unwrap()
                .with_unique(true)],
        })
        .unwrap()
        .edge_type(EdgeTypeDef {
            name: "KNOWS".into(),
            src_label: "Person".into(),
            dst_label: "Person".into(),
            properties: vec![
                PropertyDef::new("weight", DataType::Int64, false).unwrap(),
                PropertyDef::new("retained", DataType::Utf8, false).unwrap(),
                PropertyDef::new("touched", DataType::Bool, true).unwrap(),
            ],
        })
        .unwrap()
        .build();

    let alice = NodeId::new();
    let bob = NodeId::new();
    let carol = NodeId::new();
    for (id, key) in [(alice, "alice"), (bob, "bob"), (carol, "carol")] {
        writer
            .upsert_node(
                "Person",
                id,
                &NodeWriteRecord {
                    properties: BTreeMap::from([("key".into(), CoreValue::Str(key.to_string()))]),
                    schema_version: 1,
                    ..Default::default()
                },
            )
            .unwrap();
    }
    writer
        .upsert_edge(
            "KNOWS",
            alice,
            bob,
            &EdgeWriteRecord {
                properties: BTreeMap::from([
                    ("weight".into(), CoreValue::I64(1)),
                    ("retained".into(), CoreValue::Str("keep".into())),
                ]),
                schema_version: 1,
            },
        )
        .unwrap();
    // Persist a skew/high-degree source bucket. The bound-endpoint MERGE below
    // must point-probe `alice -> bob`; enumerating this whole bucket recreates
    // the loader's degree-dependent latency.
    for _ in 0..2048 {
        let mut dst = NodeId::new();
        while dst == alice || dst == bob {
            dst = NodeId::new();
        }
        writer
            .upsert_edge(
                "KNOWS",
                alice,
                dst,
                &EdgeWriteRecord {
                    properties: BTreeMap::new(),
                    schema_version: 1,
                },
            )
            .unwrap();
    }
    writer.commit_batch().await.unwrap();
    writer.flush(schema).await.unwrap();
    assert!(
        writer
            .snapshot()
            .manifest()
            .manifest
            .ssts
            .iter()
            .any(|sst| sst.kind == namidb_storage::SstKind::EdgesFwd),
        "the relationship must be persisted so MERGE exercises the SST/CSR choice"
    );

    // A miss with both endpoints bound must not decode the source's property
    // streams. The old degree scan did so even though no alice→carol edge
    // existed; the exact point probe proves absence from the partner block.
    let miss_plan = lower(
        &parse(
            "MATCH (a:Person {key: 'alice'}), (c:Person {key: 'carol'}) \
             MERGE (a)-[:KNOWS {weight: 99, retained: 'new'}]->(c)",
        )
        .unwrap(),
    )
    .unwrap();
    let stream_inserts_before_miss = sst_cache.edge_streams_inserts();
    let miss = execute_write(&miss_plan, &mut writer, &Params::new())
        .await
        .unwrap();
    assert_eq!(miss.edges_created, 1);
    assert_eq!(
        sst_cache.edge_streams_inserts(),
        stream_inserts_before_miss,
        "an exact endpoint miss must not decode any persisted edge properties"
    );

    let propertyless_plan = lower(
        &parse(
            "MATCH (a:Person {key: 'alice'}), (b:Person {key: 'bob'}) \
             MERGE (a)-[:KNOWS]->(b)",
        )
        .unwrap(),
    )
    .unwrap();
    let propertyless = execute_write(&propertyless_plan, &mut writer, &Params::new())
        .await
        .unwrap();
    assert_eq!(propertyless.edges_created, 0);
    assert_eq!(
        sst_cache.edge_streams_inserts(),
        stream_inserts_before_miss,
        "anonymous propertyless MERGE should use an existence-only point probe"
    );

    let query = parse(
        "MATCH (a:Person {key: 'alice'}), (b:Person {key: 'bob'}) \
         MERGE (a)-[r:KNOWS {weight: 1}]->(b) \
         ON MATCH SET r.touched = true \
         RETURN r",
    )
    .unwrap();
    let snapshot = writer.snapshot();
    let catalog = StatsCatalog::from_manifest(&snapshot.manifest().manifest);
    drop(snapshot);
    let plan = optimize(lower(&query).unwrap(), &catalog);

    let csr_builds_before = adjacency.builds();
    let outcome = execute_write(&plan, &mut writer, &Params::new())
        .await
        .unwrap();
    assert_eq!(
        outcome.edges_created, 0,
        "the persisted relationship and its pattern properties must match"
    );
    assert_eq!(outcome.properties_set, 1);
    assert_eq!(
        adjacency.builds(),
        csr_builds_before,
        "relationship MERGE must use one source-keyed SST range, not build a whole-type CSR"
    );
    assert_eq!(
        sst_cache.edge_streams_inserts(),
        stream_inserts_before_miss,
        "the matching exact point carries its bounded property map without decoding CSR streams"
    );
    assert_eq!(
        sst_cache.edge_readers_inserts(),
        0,
        "bound relationship MERGE must not open the persisted CSR"
    );
    assert!(
        sst_cache.edge_point_probes() >= 3,
        "miss, propertyless hit and property match should all use exact sidecar probes"
    );

    let edge = writer
        .snapshot()
        .out_edges_via_sst("KNOWS", alice)
        .await
        .unwrap()
        .edges
        .into_iter()
        .find(|edge| edge.dst == bob)
        .expect("the MERGE target edge remains present among distractors");
    assert_eq!(edge.properties.get("weight"), Some(&CoreValue::I64(1)));
    assert_eq!(
        edge.properties.get("retained"),
        Some(&CoreValue::Str("keep".into())),
        "ON MATCH SET must extend the persisted relationship property map"
    );
    assert_eq!(edge.properties.get("touched"), Some(&CoreValue::Bool(true)));
}

#[tokio::test]
async fn unwind_bound_relationship_merge_batches_exact_probes_and_preserves_ryow() {
    let sst_cache = SstCache::new(1);
    let mut caches = SessionCaches::none();
    caches.sst_cache = Some(sst_cache.clone());
    let mut writer =
        WriterSession::open_with_caches(store(), paths("w-merge-rel-batch-point"), caches)
            .await
            .unwrap();
    let schema = SchemaBuilder::new()
        .label(LabelDef {
            name: "Person".into(),
            properties: vec![PropertyDef::new("key", DataType::Utf8, false)
                .unwrap()
                .with_unique(true)],
        })
        .unwrap()
        .edge_type(EdgeTypeDef {
            name: "KNOWS".into(),
            src_label: "Person".into(),
            dst_label: "Person".into(),
            properties: vec![
                PropertyDef::new("code", DataType::Utf8, false).unwrap(),
                PropertyDef::new("retained", DataType::Utf8, true).unwrap(),
            ],
        })
        .unwrap()
        .build();
    let alice = NodeId::new();
    let bob = NodeId::new();
    let carol = NodeId::new();
    for (id, key) in [(alice, "alice"), (bob, "bob"), (carol, "carol")] {
        writer
            .upsert_node(
                "Person",
                id,
                &NodeWriteRecord {
                    properties: BTreeMap::from([("key".into(), CoreValue::Str(key.into()))]),
                    schema_version: 1,
                    ..Default::default()
                },
            )
            .unwrap();
    }
    writer
        .upsert_edge(
            "KNOWS",
            alice,
            bob,
            &EdgeWriteRecord {
                properties: BTreeMap::from([
                    ("code".into(), CoreValue::Str("A".into())),
                    ("retained".into(), CoreValue::Str("keep".into())),
                ]),
                schema_version: 1,
            },
        )
        .unwrap();
    writer.commit_batch().await.unwrap();
    writer.flush(schema).await.unwrap();
    assert!(writer
        .snapshot()
        .manifest()
        .manifest
        .ssts
        .iter()
        .any(|sst| sst.kind == namidb_storage::SstKind::EdgesFwd && sst.path.ends_with(".ep.csr")));

    let probes_before = sst_cache.edge_point_probes();
    let replay = write_q(
        &mut writer,
        "MATCH (a:Person {key: 'alice'}), (b:Person {key: 'bob'}) \
         UNWIND range(1, 2000) AS i \
         MERGE (a)-[r:KNOWS {code: 'A'}]->(b) \
         RETURN r",
    )
    .await;
    assert_eq!(replay.rows.len(), 2000);
    assert_eq!(replay.edges_created, 0);
    assert_eq!(
        sst_cache.edge_point_probes() - probes_before,
        1,
        "one UNWIND batch must issue one shared exact-index probe, not one CSR walk per row"
    );
    assert_eq!(sst_cache.edge_readers_inserts(), 0);
    let last = match replay.rows.last().and_then(|row| row.get("r")) {
        Some(RuntimeValue::Rel(rel)) => rel,
        other => panic!("expected relationship result, got {other:?}"),
    };
    assert_eq!(
        last.properties.get("retained"),
        Some(&RuntimeValue::String("keep".into()))
    );

    // A duplicated miss creates once. The prefetched miss is refreshed from
    // the staged edge after row one, so every later row matches it without
    // another storage probe or a duplicate physical relationship.
    let probes_before = sst_cache.edge_point_probes();
    let fresh = write_q(
        &mut writer,
        "MATCH (a:Person {key: 'alice'}), (c:Person {key: 'carol'}) \
         UNWIND range(1, 256) AS i \
         MERGE (a)-[:KNOWS {code: 'B'}]->(c)",
    )
    .await;
    assert_eq!(fresh.edges_created, 1);
    assert_eq!(
        sst_cache.edge_point_probes() - probes_before,
        1,
        "a batched miss is probed once and then served from RYOW state"
    );
}

#[tokio::test]
async fn loader_shape_batches_varied_existing_relationships_after_correlated_matches() {
    const ROWS: usize = 128;
    let sst_cache = SstCache::new(1);
    let mut caches = SessionCaches::none();
    caches.sst_cache = Some(sst_cache.clone());
    let mut writer =
        WriterSession::open_with_caches(store(), paths("w-merge-rel-loader-batch"), caches)
            .await
            .unwrap();
    let schema = SchemaBuilder::new()
        .label(LabelDef {
            name: "Entidad".into(),
            properties: vec![PropertyDef::new("key", DataType::Utf8, false)
                .unwrap()
                .with_unique(true)],
        })
        .unwrap()
        .edge_type(EdgeTypeDef {
            name: "CITA".into(),
            src_label: "Entidad".into(),
            dst_label: "Entidad".into(),
            properties: vec![
                PropertyDef::new("codigo", DataType::Utf8, false).unwrap(),
                PropertyDef::new("retained", DataType::Utf8, true).unwrap(),
                PropertyDef::new("seen", DataType::Bool, true).unwrap(),
            ],
        })
        .unwrap()
        .build();

    let mut param_rows = Vec::with_capacity(ROWS);
    for i in 0..ROWS {
        let src = NodeId::new();
        let dst = NodeId::new();
        let src_key = format!("src-{i:04}");
        let dst_key = format!("dst-{i:04}");
        let codigo = format!("BOE-{i:04}");
        for (id, key) in [(src, &src_key), (dst, &dst_key)] {
            writer
                .upsert_node(
                    "Entidad",
                    id,
                    &NodeWriteRecord {
                        properties: BTreeMap::from([(
                            "key".into(),
                            CoreValue::Str((*key).clone()),
                        )]),
                        schema_version: 1,
                        ..Default::default()
                    },
                )
                .unwrap();
        }
        writer
            .upsert_edge(
                "CITA",
                src,
                dst,
                &EdgeWriteRecord {
                    properties: BTreeMap::from([
                        ("codigo".into(), CoreValue::Str(codigo.clone())),
                        (
                            "retained".into(),
                            CoreValue::Str(format!("retained-{i:04}")),
                        ),
                    ]),
                    schema_version: 1,
                },
            )
            .unwrap();
        param_rows.push(RuntimeValue::Map(BTreeMap::from([
            ("src".into(), RuntimeValue::String(src_key)),
            ("dst".into(), RuntimeValue::String(dst_key)),
            ("codigo".into(), RuntimeValue::String(codigo)),
        ])));
    }
    writer.commit_batch().await.unwrap();
    writer.flush(schema).await.unwrap();

    let query = parse(
        "UNWIND $rows AS row \
         MATCH (a:Entidad {key: row.src}) \
         MATCH (b:Entidad {key: row.dst}) \
         MERGE (a)-[r:CITA {codigo: row.codigo}]->(b) \
         ON MATCH SET r.seen = true \
         RETURN r.codigo AS codigo, r.retained AS retained, r.seen AS seen",
    )
    .unwrap();
    let snapshot = writer.snapshot();
    let catalog = StatsCatalog::from_manifest(&snapshot.manifest().manifest);
    drop(snapshot);
    let plan = optimize(lower(&query).unwrap(), &catalog);
    let params = Params::from([("rows".into(), RuntimeValue::List(param_rows))]);

    let probes_before = sst_cache.edge_point_probes();
    let outcome = execute_write(&plan, &mut writer, &params).await.unwrap();
    assert_eq!(outcome.rows.len(), ROWS);
    assert_eq!(outcome.edges_created, 0);
    assert_eq!(outcome.properties_set, ROWS as u64);
    assert_eq!(
        sst_cache.edge_point_probes() - probes_before,
        1,
        "the real loader operator must batch all varied EdgeKeys into one sidecar probe"
    );
    assert_eq!(
        sst_cache.edge_readers_inserts(),
        0,
        "correlated MATCH after UNWIND must not reopen the CSR per relationship"
    );
    for (i, row) in outcome.rows.iter().enumerate() {
        assert_eq!(
            row.get("codigo"),
            Some(&RuntimeValue::String(format!("BOE-{i:04}")))
        );
        assert_eq!(
            row.get("retained"),
            Some(&RuntimeValue::String(format!("retained-{i:04}")))
        );
        assert_eq!(row.get("seen"), Some(&RuntimeValue::Bool(true)));
    }
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

fn string_unique_schema() -> namidb_core::schema::Schema {
    SchemaBuilder::new()
        .label(LabelDef {
            name: "Account".into(),
            properties: vec![PropertyDef::new("key", DataType::Utf8, false)
                .unwrap()
                .with_unique(true)],
        })
        .unwrap()
        .build()
}

async fn writer_with_committed_string_unique_account(namespace: &str) -> (WriterSession, NodeId) {
    let mut writer = WriterSession::open(store(), paths(namespace))
        .await
        .unwrap();
    let id = NodeId::new();
    writer
        .upsert_node(
            "Account",
            id,
            &NodeWriteRecord {
                properties: BTreeMap::from([("key".into(), CoreValue::Str("existing".into()))]),
                schema_version: 1,
                ..Default::default()
            },
        )
        .unwrap();
    writer.commit_batch().await.unwrap();
    writer.flush(string_unique_schema()).await.unwrap();
    (writer, id)
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
async fn unwind_2000_delete_edges_uses_correlated_unique_lookups() {
    // Regression for the legal-graph cleanup shape:
    //
    //   UNWIND $keys AS key
    //   MATCH (n {key: key})-[r]->(:Target)
    //   DELETE r
    //
    // Before the correlated lookup rewrite, the anchor lowered to
    // Filter(CrossProduct(Unwind, NodeScan)) and the edge path swept the whole
    // memtable for every key. One real edge makes the assertion sensitive to
    // false-negative lookup rewrites; the other 1,999 keys exercise the empty
    // adjacency hot path. The label-agnostic posting lookup preserves matches
    // without guessing `:Source`.
    let adjacency = Arc::new(AdjacencyCache::new(64 * 1024 * 1024));
    let mut caches = SessionCaches::shared();
    caches.adjacency_cache = Some(Arc::clone(&adjacency));
    let mut writer =
        WriterSession::open_with_caches(store(), paths("w-delete-edge-correlated-lookup"), caches)
            .await
            .unwrap();
    let mut keys = Vec::with_capacity(2_000);
    let mut first_source = None;
    for i in 0..2_000 {
        let key = format!("legal-{i}");
        keys.push(RuntimeValue::String(key.clone()));
        let id = NodeId::new();
        first_source.get_or_insert(id);
        writer
            .upsert_node(
                "Source",
                id,
                &NodeWriteRecord {
                    properties: BTreeMap::from([("key".into(), CoreValue::Str(key))]),
                    schema_version: 1,
                    ..Default::default()
                },
            )
            .unwrap();
    }

    // Several declared edge types make the untyped `-[r]->` faithful to the
    // production shape. Half target :Target and half target :Other; no edge
    // rows are present for any of them.
    let mut schema = SchemaBuilder::new()
        .label(LabelDef {
            name: "Source".into(),
            properties: vec![PropertyDef::new("key", DataType::Utf8, false)
                .unwrap()
                .with_unique(true)],
        })
        .unwrap()
        .label(LabelDef {
            name: "Target".into(),
            properties: vec![],
        })
        .unwrap()
        .label(LabelDef {
            name: "Other".into(),
            properties: vec![],
        })
        .unwrap();
    for i in 0..8 {
        schema = schema
            .edge_type(EdgeTypeDef {
                name: format!("REL_{i}"),
                src_label: "Source".into(),
                dst_label: if i % 2 == 0 {
                    "Target".into()
                } else {
                    "Other".into()
                },
                properties: vec![],
            })
            .unwrap();
    }
    let schema = schema.build();
    writer.flush(schema.clone()).await.unwrap();

    let first_source = first_source.unwrap();
    let target = NodeId::new();
    writer
        .upsert_node(
            "Target",
            target,
            &NodeWriteRecord {
                schema_version: 1,
                ..Default::default()
            },
        )
        .unwrap();
    writer
        .upsert_edge(
            "REL_0",
            first_source,
            target,
            &EdgeWriteRecord {
                properties: BTreeMap::new(),
                schema_version: 1,
            },
        )
        .unwrap();
    writer.commit_batch().await.unwrap();
    writer.flush(schema).await.unwrap();
    assert!(
        writer
            .snapshot()
            .manifest()
            .manifest
            .ssts
            .iter()
            .any(|sst| matches!(
                sst.kind,
                namidb_storage::SstKind::EdgesFwd | namidb_storage::SstKind::EdgesInv
            )),
        "the relationship must be persisted so the regression exercises the SST/CSR choice"
    );

    let query = parse(
        "UNWIND $keys AS key \
         MATCH (n {key: key})-[r]->(:Target) \
         DELETE r",
    )
    .unwrap();
    let snapshot = writer.snapshot();
    let catalog = StatsCatalog::from_manifest(&snapshot.manifest().manifest);
    drop(snapshot);
    let plan = optimize(lower(&query).unwrap(), &catalog);

    fn count_lookups(plan: &namidb_query::LogicalPlan) -> usize {
        usize::from(matches!(
            plan,
            namidb_query::LogicalPlan::NodeByPropertyValue {
                label,
                property,
                multi: true,
                ..
            } if label.is_empty() && property == "key"
        )) + plan
            .children()
            .iter()
            .map(|p| count_lookups(p))
            .sum::<usize>()
    }
    assert_eq!(
        count_lookups(&plan),
        1,
        "optimized delete must contain one correlated global key posting lookup: {plan:?}"
    );

    let mut params = Params::new();
    params.insert("keys".into(), RuntimeValue::List(keys));
    let lookups_before = writer.property_index_cache().equality_lookup_calls();
    let sidecar_inserts_before = writer
        .sst_cache()
        .map(|cache| cache.property_sidecar_inserts())
        .unwrap_or(0);
    let csr_builds_before = adjacency.builds();
    let outcome = execute_write(&plan, &mut writer, &params).await.unwrap();
    assert_eq!(
        outcome.edges_deleted, 1,
        "the global posting lookup must bind the real source and delete its edge"
    );
    assert!(
        outcome.rows.is_empty(),
        "a terminal write-only delete must discard its internal relationship row"
    );
    assert_eq!(
        writer.property_index_cache().equality_lookup_calls() - lookups_before,
        1,
        "the complete UNWIND must route through one batched global equality posting lookup"
    );
    assert_eq!(
        writer
            .sst_cache()
            .map(|cache| cache.property_sidecar_inserts())
            .unwrap_or(0)
            - sidecar_inserts_before,
        0,
        "native paged equality lookups must not decode/insert the legacy key sidecar"
    );
    assert_eq!(
        adjacency.builds(),
        csr_builds_before,
        "a keyed DELETE must use sparse SST identity ranges, not build a whole-type CSR"
    );

    // Correlated values are runtime data and may be NULL. Cypher's
    // `n.key = NULL` never matches; prove that the point-lookup path cannot
    // accidentally bind the label's first node and delete its edge.
    // Re-create the deleted edge, then prove NULL cannot bind the first node.
    writer
        .upsert_edge(
            "REL_0",
            first_source,
            target,
            &EdgeWriteRecord {
                properties: BTreeMap::new(),
                schema_version: 1,
            },
        )
        .unwrap();
    writer.commit_batch().await.unwrap();

    let null_params = Params::from([("keys".into(), RuntimeValue::List(vec![RuntimeValue::Null]))]);
    let null_outcome = execute_write(&plan, &mut writer, &null_params)
        .await
        .unwrap();
    assert_eq!(null_outcome.edges_deleted, 0);
    assert_eq!(
        writer
            .snapshot()
            .out_edges("REL_0", first_source)
            .await
            .unwrap()
            .edges
            .len(),
        1,
        "a NULL correlated key must not match/delete the first Source node's edge"
    );
}

#[tokio::test]
async fn global_correlated_match_batches_direct_set_and_delete_with_rollback() {
    let mut writer = WriterSession::open(store(), paths("w-global-match-direct-write-batch"))
        .await
        .unwrap();
    let doc_shared = NodeId::new();
    let stub_shared = NodeId::new();
    let doc_only = NodeId::new();
    let numeric = NodeId::new();
    for (label, id, key, kind) in [
        ("Doc", doc_shared, CoreValue::Str("shared".into()), "doc"),
        ("Stub", stub_shared, CoreValue::Str("shared".into()), "stub"),
        ("Doc", doc_only, CoreValue::Str("only".into()), "only"),
        ("Numeric", numeric, CoreValue::I64(7), "numeric"),
    ] {
        writer
            .upsert_node(
                label,
                id,
                &NodeWriteRecord {
                    properties: BTreeMap::from([
                        ("key".into(), key),
                        ("kind".into(), CoreValue::Str(kind.into())),
                        ("hits".into(), CoreValue::I64(0)),
                    ]),
                    schema_version: 1,
                    ..Default::default()
                },
            )
            .unwrap();
    }
    let string_key = || {
        PropertyDef::new("key", DataType::Utf8, false)
            .unwrap()
            .with_indexed(true)
    };
    let schema = SchemaBuilder::new()
        .label(LabelDef {
            name: "Doc".into(),
            properties: vec![string_key()],
        })
        .unwrap()
        .label(LabelDef {
            name: "Stub".into(),
            properties: vec![string_key()],
        })
        .unwrap()
        .label(LabelDef {
            name: "Fresh".into(),
            properties: vec![string_key()],
        })
        .unwrap()
        .label(LabelDef {
            name: "Numeric".into(),
            properties: vec![PropertyDef::new("key", DataType::Int64, false)
                .unwrap()
                .with_indexed(true)],
        })
        .unwrap()
        .build();
    writer.flush(schema).await.unwrap();

    let snapshot = writer.snapshot();
    let catalog = StatsCatalog::from_manifest(&snapshot.manifest().manifest);
    drop(snapshot);
    let set_query = parse(
        "UNWIND $keys AS key \
         MATCH (n {key: key}) \
         SET n.hits = n.hits + 1 \
         RETURN n.key AS key, n.kind AS kind, n.hits AS hits",
    )
    .unwrap();
    let set_plan = optimize(lower(&set_query).unwrap(), &catalog);

    fn count_global_multi_lookups(plan: &namidb_query::LogicalPlan) -> usize {
        usize::from(matches!(
            plan,
            namidb_query::LogicalPlan::NodeByPropertyValue {
                label,
                property,
                multi: true,
                ..
            } if label.is_empty() && property == "key"
        )) + plan
            .children()
            .iter()
            .map(|child| count_global_multi_lookups(child))
            .sum::<usize>()
    }
    assert_eq!(
        count_global_multi_lookups(&set_plan),
        1,
        "direct global MATCH+SET must retain one correlated posting lookup: {set_plan:?}"
    );

    let set_params = Params::from([(
        "keys".into(),
        RuntimeValue::List(vec![
            RuntimeValue::String("shared".into()),
            RuntimeValue::String("missing".into()),
            RuntimeValue::Integer(7),
            RuntimeValue::Null,
            RuntimeValue::String("shared".into()),
            RuntimeValue::String("only".into()),
        ]),
    )]);
    let lookups_before = writer.property_index_cache().equality_lookup_calls();
    let set_outcome = execute_write_staged(&set_plan, &mut writer, &set_params)
        .await
        .unwrap();
    assert_eq!(
        writer.property_index_cache().equality_lookup_calls() - lookups_before,
        1,
        "all correlated String values must share one global storage batch"
    );
    assert_eq!(set_outcome.rows.len(), 6);
    assert_eq!(set_outcome.properties_set, 6);
    assert_eq!(
        set_outcome
            .rows
            .iter()
            .map(|row| match row.get("key") {
                Some(RuntimeValue::String(key)) => key.clone(),
                Some(RuntimeValue::Integer(value)) => value.to_string(),
                other => panic!("expected matched key, got {other:?}"),
            })
            .collect::<Vec<_>>(),
        vec!["shared", "shared", "7", "shared", "shared", "only"],
        "the batch must preserve row order, fan-out, duplicates, misses, NULL and typed fallback"
    );

    let staged = writer.overlay_snapshot();
    assert_eq!(
        staged
            .lookup_node("Doc", doc_shared)
            .await
            .unwrap()
            .unwrap()
            .properties
            .get("hits"),
        Some(&CoreValue::I64(2))
    );
    drop(staged);

    writer.discard_batch();
    let rolled_back = writer.snapshot();
    assert_eq!(
        rolled_back
            .lookup_node("Doc", doc_shared)
            .await
            .unwrap()
            .unwrap()
            .properties
            .get("hits"),
        Some(&CoreValue::I64(0))
    );
    drop(rolled_back);

    // Leave a node staged before MATCH so a second execution must use the
    // transactional global overlay, not only committed sidecars.
    let create_fresh =
        lower(&parse("CREATE (:Fresh {key: 'staged', kind: 'fresh', hits: 0})").unwrap()).unwrap();
    execute_write_staged(&create_fresh, &mut writer, &Params::new())
        .await
        .unwrap();
    let staged_params = Params::from([(
        "keys".into(),
        RuntimeValue::List(vec![RuntimeValue::String("staged".into())]),
    )]);
    let staged_outcome = execute_write_staged(&set_plan, &mut writer, &staged_params)
        .await
        .unwrap();
    assert_eq!(staged_outcome.rows.len(), 1);
    assert_eq!(staged_outcome.properties_set, 1);
    let staged = writer.overlay_snapshot();
    assert_eq!(
        staged
            .scan_label("Fresh")
            .await
            .unwrap()
            .first()
            .and_then(|node| node.properties.get("hits")),
        Some(&CoreValue::I64(1)),
        "global batch lookup must see and update a node staged before MATCH"
    );
    drop(staged);
    writer.discard_batch();
    assert!(
        writer
            .snapshot()
            .scan_label("Fresh")
            .await
            .unwrap()
            .is_empty(),
        "discard must remove both the staged node and its batched SET"
    );

    let delete_query = parse("UNWIND $keys AS key MATCH (n {key: key}) DELETE n").unwrap();
    let delete_plan = optimize(lower(&delete_query).unwrap(), &catalog);
    assert_eq!(
        count_global_multi_lookups(&delete_plan),
        1,
        "direct global MATCH+DELETE must retain one correlated posting lookup: {delete_plan:?}"
    );
    let delete_params = Params::from([(
        "keys".into(),
        RuntimeValue::List(vec![
            RuntimeValue::String("shared".into()),
            RuntimeValue::String("missing".into()),
            RuntimeValue::Integer(7),
            RuntimeValue::Null,
            RuntimeValue::String("only".into()),
        ]),
    )]);
    let lookups_before = writer.property_index_cache().equality_lookup_calls();
    let delete_outcome = execute_write_staged(&delete_plan, &mut writer, &delete_params)
        .await
        .unwrap();
    assert_eq!(
        writer.property_index_cache().equality_lookup_calls() - lookups_before,
        1
    );
    assert!(
        delete_outcome.rows.is_empty(),
        "terminal write-only deletes must not retain matched node rows"
    );
    assert_eq!(delete_outcome.nodes_deleted, 4);
    let staged_delete = writer.overlay_snapshot();
    assert!(staged_delete.scan_label("Doc").await.unwrap().is_empty());
    assert!(staged_delete.scan_label("Stub").await.unwrap().is_empty());
    assert!(staged_delete
        .scan_label("Numeric")
        .await
        .unwrap()
        .is_empty());
    drop(staged_delete);
    writer.discard_batch();
    let rolled_back_delete = writer.snapshot();
    assert_eq!(rolled_back_delete.scan_label("Doc").await.unwrap().len(), 2);
    assert_eq!(
        rolled_back_delete.scan_label("Stub").await.unwrap().len(),
        1
    );
    assert_eq!(
        rolled_back_delete
            .scan_label("Numeric")
            .await
            .unwrap()
            .len(),
        1,
        "discard must restore every node deleted through the batched lookup"
    );
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
async fn merge_existing_unique_keys_batch_hydrates_compacted_node_sst_once() {
    const EXISTING: usize = 256;
    const OTHER: usize = 8_192;

    // Reproduce the incremental legal-loader shape: a small :Materia label
    // lives in one id-primary SST with a much larger unrelated corpus. The
    // unique probe itself is O(1), but its hit returns a NodeId that MERGE
    // historically hydrated through one cold point walk per input row. Misses
    // skipped hydration, explaining the existing/new-key asymmetry.
    let cache = SstCache::new(64 * 1024 * 1024);
    let mut writer = WriterSession::open_with_caches(
        store(),
        paths("w-merge-existing-batch-hydration"),
        SessionCaches {
            sst_cache: Some(cache.clone()),
            node_cache: None,
            adjacency_cache: None,
        },
    )
    .await
    .unwrap();
    for i in 0..EXISTING {
        writer
            .upsert_node(
                "Materia",
                NodeId::new(),
                &NodeWriteRecord {
                    properties: BTreeMap::from([(
                        "key".into(),
                        CoreValue::Str(format!("materia-{i:08}")),
                    )]),
                    schema_version: 1,
                    ..Default::default()
                },
            )
            .unwrap();
    }
    for i in 0..OTHER {
        writer
            .upsert_node(
                "Other",
                NodeId::new(),
                &NodeWriteRecord {
                    properties: BTreeMap::from([(
                        "key".into(),
                        CoreValue::Str(format!("other-{i:08}")),
                    )]),
                    schema_version: 1,
                    ..Default::default()
                },
            )
            .unwrap();
    }
    writer.commit_batch().await.unwrap();
    let schema = SchemaBuilder::new()
        .label(LabelDef {
            name: "Materia".into(),
            properties: vec![PropertyDef::new("key", DataType::Utf8, false)
                .unwrap()
                .with_unique(true)],
        })
        .unwrap()
        .label(LabelDef {
            name: "Other".into(),
            properties: vec![PropertyDef::new("key", DataType::Utf8, false).unwrap()],
        })
        .unwrap()
        .build();
    writer.flush(schema).await.unwrap();

    let plan = lower(&parse("UNWIND $keys AS key MERGE (m:Materia {key: key}) RETURN m").unwrap())
        .unwrap();
    let keys = (0..EXISTING)
        .map(|i| RuntimeValue::String(format!("materia-{i:08}")))
        .collect();
    let params = Params::from([("keys".into(), RuntimeValue::List(keys))]);
    let row_group_inserts = cache.decoded_node_row_group_inserts();
    let sparse_scans = cache.sparse_node_filter_scans();
    let locator_probes = cache.node_locator_probes();
    let locator_entries = cache.node_locator_entries_examined();
    let locator_bytes = cache.node_locator_bytes();
    let population_scans = writer.unique_index().populate_scans();
    let outcome = execute_write(&plan, &mut writer, &params).await.unwrap();
    assert_eq!(outcome.nodes_created, 0);
    assert_eq!(outcome.rows.len(), EXISTING);
    assert_eq!(
        cache.decoded_node_row_group_inserts() - row_group_inserts,
        0,
        "a sparse MERGE batch must not cache/decode the corpus-wide complete row group"
    );
    assert_eq!(
        cache.sparse_node_filter_scans() - sparse_scans,
        0,
        "the NodeId locator must avoid a corpus-wide sparse Parquet filter"
    );
    assert!(
        cache.node_locator_probes() > locator_probes,
        "existing-key hydration must probe the NodeId locator"
    );
    assert!(
        cache.node_locator_entries_examined() - locator_entries < OTHER as u64,
        "the locator must examine fewer entries than the unrelated corpus"
    );
    assert!(
        cache.node_locator_bytes() > locator_bytes,
        "the locator must account for bounded page reads"
    );
    assert_eq!(
        writer.unique_index().populate_scans(),
        population_scans,
        "String MERGE must seed exact sidecar hits/misses instead of scanning the label"
    );

    // A fresh execution remains idempotent and reuses the decoded property
    // sidecar + partial transactional keys. It may run one new sparse payload
    // filter, but it must never populate by scanning the label.
    let scans = writer.unique_index().populate_scans();
    let replay = execute_write(&plan, &mut writer, &params).await.unwrap();
    assert_eq!(replay.nodes_created, 0);
    assert_eq!(replay.rows.len(), EXISTING);
    assert_eq!(writer.unique_index().populate_scans(), scans);
    assert_eq!(
        cache.decoded_node_row_group_inserts(),
        row_group_inserts,
        "replay must not admit a complete corpus row group either"
    );
}

#[tokio::test]
async fn merge_unique_string_after_staged_create_uses_point_seed_not_label_scan() {
    let (mut writer, _) =
        writer_with_committed_string_unique_account("w-merge-seed-after-create").await;
    let scans = writer.unique_index().populate_scans();

    // The CREATE and MERGE share one statement. The unrelated node mutation
    // used to disable the committed sidecar seed globally, making MERGE
    // populate :Account through scan_label even though its exact key is
    // already indexed.
    let plan = lower(
        &parse(
            "CREATE (:Other {value: 1}) \
             MERGE (a:Account {key: 'existing'}) \
             ON MATCH SET a.seen = true \
             RETURN a",
        )
        .unwrap(),
    )
    .unwrap();
    let outcome = execute_write_staged(&plan, &mut writer, &Params::new())
        .await
        .unwrap();
    assert_eq!(outcome.nodes_created, 1, "only :Other should be created");
    assert_eq!(outcome.rows.len(), 1);
    assert_eq!(
        writer.unique_index().populate_scans(),
        scans,
        "a staged CREATE must not force MERGE to scan the stored :Account label"
    );
    writer.discard_batch();
}

#[tokio::test]
async fn merge_unique_string_reconciles_staged_set_without_label_scan() {
    let (mut writer, account) =
        writer_with_committed_string_unique_account("w-merge-seed-after-set").await;
    let scans = writer.unique_index().populate_scans();

    // Explicit transaction statement 1 rewrites the full record through SET,
    // but reaches it by NodeId so no property map is incidentally populated.
    let set = lower(&parse("MATCH (a:Account {_id: $id}) SET a.touch = true").unwrap()).unwrap();
    let params = Params::from([("id".into(), RuntimeValue::String(account.to_string()))]);
    execute_write_staged(&set, &mut writer, &params)
        .await
        .unwrap();
    assert_eq!(writer.unique_index().populate_scans(), scans);

    // Statement 2 must overlay the staged full-record upsert over the
    // committed point candidate, preserving both identity and properties.
    let merge = lower(
        &parse(
            "MERGE (a:Account {key: 'existing'}) \
             ON MATCH SET a.seen = true \
             RETURN a",
        )
        .unwrap(),
    )
    .unwrap();
    let outcome = execute_write_staged(&merge, &mut writer, &Params::new())
        .await
        .unwrap();
    assert_eq!(outcome.nodes_created, 0);
    match outcome.rows[0].get("a") {
        Some(RuntimeValue::Node(node)) => {
            assert_eq!(node.id, account);
            assert_eq!(
                node.properties.get("touch"),
                Some(&RuntimeValue::Bool(true))
            );
        }
        other => panic!("expected merged Account, got {other:?}"),
    }
    assert_eq!(
        writer.unique_index().populate_scans(),
        scans,
        "a staged SET must be reconciled from the bounded overlay, not scan_label"
    );
    writer.discard_batch();
}

#[tokio::test]
async fn merge_unique_string_reconciles_staged_delete_without_label_scan() {
    let (mut writer, deleted) =
        writer_with_committed_string_unique_account("w-merge-seed-after-delete").await;
    let scans = writer.unique_index().populate_scans();

    let delete = lower(&parse("MATCH (a:Account {_id: $id}) DELETE a").unwrap()).unwrap();
    let params = Params::from([("id".into(), RuntimeValue::String(deleted.to_string()))]);
    execute_write_staged(&delete, &mut writer, &params)
        .await
        .unwrap();
    assert_eq!(writer.unique_index().populate_scans(), scans);

    // The immutable sidecar still points at `deleted`; the staged tombstone
    // must suppress that base hit so MERGE takes the create branch.
    let merge = lower(&parse("MERGE (a:Account {key: 'existing'}) RETURN a").unwrap()).unwrap();
    let outcome = execute_write_staged(&merge, &mut writer, &Params::new())
        .await
        .unwrap();
    assert_eq!(outcome.nodes_created, 1);
    match outcome.rows[0].get("a") {
        Some(RuntimeValue::Node(node)) => assert_ne!(node.id, deleted),
        other => panic!("expected replacement Account, got {other:?}"),
    }
    assert_eq!(
        writer.unique_index().populate_scans(),
        scans,
        "a staged DELETE must suppress the base hit without scanning :Account"
    );
    writer.discard_batch();
}

#[tokio::test]
async fn merge_existing_string_key_survives_global_mixed_type_sidecar() {
    // Id-primary SSTs harvest one global equality sidecar per property name.
    // Keep the Bool label lexically first so the schema union historically
    // chose Bool as the synthetic declaration and silently omitted the String
    // value carried by the second label. The resulting incomplete sidecar was
    // still advertised as authoritative: MERGE seeded a false miss and created
    // a duplicate despite B.key being unique.
    let mut writer = WriterSession::open(store(), paths("w-merge-global-mixed-key-type"))
        .await
        .unwrap();
    writer
        .upsert_node(
            "A",
            NodeId::new(),
            &NodeWriteRecord {
                properties: BTreeMap::from([("key".into(), CoreValue::Bool(true))]),
                schema_version: 1,
                ..Default::default()
            },
        )
        .unwrap();
    writer
        .upsert_node(
            "B",
            NodeId::new(),
            &NodeWriteRecord {
                properties: BTreeMap::from([("key".into(), CoreValue::Str("existing".into()))]),
                schema_version: 1,
                ..Default::default()
            },
        )
        .unwrap();
    writer.commit_batch().await.unwrap();

    let schema = SchemaBuilder::new()
        .label(LabelDef {
            name: "A".into(),
            properties: vec![PropertyDef::new("key", DataType::Bool, false)
                .unwrap()
                .with_unique(true)],
        })
        .unwrap()
        .label(LabelDef {
            name: "B".into(),
            properties: vec![PropertyDef::new("key", DataType::Utf8, false)
                .unwrap()
                .with_unique(true)],
        })
        .unwrap()
        .build();
    writer.flush(schema).await.unwrap();

    let scans = writer.unique_index().populate_scans();
    let outcome = write_q(
        &mut writer,
        "MERGE (b:B {key: 'existing'}) \
         ON MATCH SET b.seen = true \
         RETURN b",
    )
    .await;
    assert_eq!(
        outcome.nodes_created, 0,
        "the global sidecar must not turn an existing mixed-type key into a miss"
    );
    assert_eq!(outcome.rows.len(), 1);
    assert_eq!(
        writer.unique_index().populate_scans(),
        scans,
        "mixed-type global postings must answer the existing key without a label scan"
    );

    let snapshot = writer.snapshot();
    let a = snapshot.scan_label("A").await.unwrap();
    let b = snapshot.scan_label("B").await.unwrap();
    assert_eq!(a.len(), 1, "the Bool claimant must remain visible");
    assert_eq!(b.len(), 1, "MERGE must not duplicate the String claimant");
    assert_eq!(
        b[0].properties.get("seen"),
        Some(&CoreValue::Bool(true)),
        "the existing node must take ON MATCH"
    );
}

#[tokio::test]
async fn merge_existing_string_key_uses_global_sidecar_when_first_type_is_unsupported() {
    // A synthetic global property used to inherit the lexically first
    // declaration's type. If that declaration was not encodable by ScalarV1,
    // the collector omitted the entire property even when another label had a
    // String unique key. Correctness survived through the scan fallback, but
    // existing-key MERGE remained O(total nodes).
    let mut writer = WriterSession::open(store(), paths("w-merge-global-int-first-key"))
        .await
        .unwrap();
    writer
        .upsert_node(
            "A",
            NodeId::new(),
            &NodeWriteRecord {
                properties: BTreeMap::from([("key".into(), CoreValue::I64(7))]),
                schema_version: 1,
                ..Default::default()
            },
        )
        .unwrap();
    writer
        .upsert_node(
            "B",
            NodeId::new(),
            &NodeWriteRecord {
                properties: BTreeMap::from([("key".into(), CoreValue::Str("existing".into()))]),
                schema_version: 1,
                ..Default::default()
            },
        )
        .unwrap();
    writer.commit_batch().await.unwrap();

    let schema = SchemaBuilder::new()
        .label(LabelDef {
            name: "A".into(),
            properties: vec![PropertyDef::new("key", DataType::Int64, false)
                .unwrap()
                .with_unique(true)],
        })
        .unwrap()
        .label(LabelDef {
            name: "B".into(),
            properties: vec![PropertyDef::new("key", DataType::Utf8, false)
                .unwrap()
                .with_unique(true)],
        })
        .unwrap()
        .build();
    writer.flush(schema).await.unwrap();

    let scans = writer.unique_index().populate_scans();
    let outcome = write_q(
        &mut writer,
        "MERGE (b:B {key: 'existing'}) \
         ON MATCH SET b.seen = true \
         RETURN b",
    )
    .await;
    assert_eq!(outcome.nodes_created, 0);
    assert_eq!(outcome.rows.len(), 1);
    assert_eq!(
        writer.unique_index().populate_scans(),
        scans,
        "a later String declaration must enable the global sidecar"
    );

    let snapshot = writer.snapshot();
    let a = snapshot.scan_label("A").await.unwrap();
    let b = snapshot.scan_label("B").await.unwrap();
    assert_eq!(a.len(), 1);
    assert_eq!(b.len(), 1);
    assert_eq!(b[0].properties.get("seen"), Some(&CoreValue::Bool(true)));
}

#[tokio::test]
async fn merge_existing_unique_batch_refreshes_on_match_state_between_duplicate_rows() {
    let mut writer = WriterSession::open(store(), paths("w-merge-existing-batch-ryow"))
        .await
        .unwrap();
    write_q(&mut writer, "CREATE (:Account {key: 'same', counter: 0})").await;
    let schema = SchemaBuilder::new()
        .label(LabelDef {
            name: "Account".into(),
            properties: vec![
                PropertyDef::new("key", DataType::Utf8, false)
                    .unwrap()
                    .with_unique(true),
                PropertyDef::new("counter", DataType::Int64, false).unwrap(),
            ],
        })
        .unwrap()
        .build();
    writer.flush(schema).await.unwrap();

    // Both rows prefetch the same committed node. The second row must see the
    // first row's ON MATCH mutation, not the original prefetched NodeValue.
    let outcome = write_q(
        &mut writer,
        "UNWIND range(1, 2) AS i \
         MERGE (a:Account {key: 'same'}) \
         ON MATCH SET a.counter = a.counter + 1 \
         RETURN a",
    )
    .await;
    assert_eq!(outcome.nodes_created, 0);
    assert_eq!(outcome.rows.len(), 2);
    let observed: Vec<i64> = outcome
        .rows
        .iter()
        .map(|row| match row.get("a") {
            Some(RuntimeValue::Node(node)) => match node.properties.get("counter") {
                Some(RuntimeValue::Integer(value)) => *value,
                other => panic!("expected integer counter, got {other:?}"),
            },
            other => panic!("expected merged node, got {other:?}"),
        })
        .collect();
    assert_eq!(observed, vec![1, 2]);
    let nodes = writer.snapshot().scan_label("Account").await.unwrap();
    assert_eq!(nodes[0].properties.get("counter"), Some(&CoreValue::I64(2)));
}

#[tokio::test]
async fn merge_existing_unique_batch_keeps_outer_alias_of_same_node_coherent() {
    let mut writer = WriterSession::open(store(), paths("w-merge-existing-batch-alias-ryow"))
        .await
        .unwrap();
    write_q(&mut writer, "CREATE (:Account {key: 'same', counter: 0})").await;
    let schema = SchemaBuilder::new()
        .label(LabelDef {
            name: "Account".into(),
            properties: vec![
                PropertyDef::new("key", DataType::Utf8, false)
                    .unwrap()
                    .with_unique(true),
                PropertyDef::new("counter", DataType::Int64, false).unwrap(),
            ],
        })
        .unwrap()
        .build();
    writer.flush(schema).await.unwrap();

    // `z` and `a` are two aliases for the same physical node. Historically,
    // the batch-prefetch refresh walked every BTreeMap binding; because `z`
    // sorts after `a`, its stale clone overwrote the ON MATCH value and every
    // duplicate input row observed counter=1.
    let outcome = write_q(
        &mut writer,
        "MATCH (z:Account {key: 'same'}) \
         UNWIND range(1, 2) AS i \
         MERGE (a:Account {key: 'same'}) \
         ON MATCH SET a.counter = a.counter + 1 \
         RETURN a, z",
    )
    .await;
    assert_eq!(outcome.nodes_created, 0);
    assert_eq!(outcome.rows.len(), 2);
    let counters: Vec<(i64, i64)> = outcome
        .rows
        .iter()
        .map(|row| {
            let counter = |alias: &str| match row.get(alias) {
                Some(RuntimeValue::Node(node)) => match node.properties.get("counter") {
                    Some(RuntimeValue::Integer(value)) => *value,
                    other => panic!("expected integer counter on {alias}, got {other:?}"),
                },
                other => panic!("expected node alias {alias}, got {other:?}"),
            };
            (counter("a"), counter("z"))
        })
        .collect();
    assert_eq!(counters, vec![(1, 1), (2, 2)]);
    let nodes = writer.snapshot().scan_label("Account").await.unwrap();
    assert_eq!(nodes[0].properties.get("counter"), Some(&CoreValue::I64(2)));
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
async fn unique_indexed_match_unwind_uses_numeric_transactional_probes() {
    let mut writer = WriterSession::open(store(), paths("w-match-unique-numeric-ryow"))
        .await
        .unwrap();
    // Load before declaring the constraint so the transactional map starts
    // cold, mirroring a reopened/imported store.
    write_q(
        &mut writer,
        "UNWIND range(1, 500) AS i CREATE (:Account {account_no: i})",
    )
    .await;
    writer.flush(int_unique_schema()).await.unwrap();

    let one = writer
        .snapshot()
        .scan_label("Account")
        .await
        .unwrap()
        .into_iter()
        .find(|node| node.properties.get("account_no") == Some(&CoreValue::I64(1)))
        .expect("account 1");
    writer
        .upsert_node(
            "Account",
            one.id,
            &NodeWriteRecord {
                properties: BTreeMap::from([("account_no".into(), CoreValue::I64(1_001))]),
                schema_version: 1,
                ..Default::default()
            },
        )
        .unwrap();

    let query = parse(
        "UNWIND range(1, 500) AS i \
         MATCH (a:Account {account_no: toFloat(i)}) \
         SET a.seen = true \
         RETURN a",
    )
    .unwrap();
    let snapshot = writer.snapshot();
    let catalog = StatsCatalog::from_manifest(&snapshot.manifest().manifest);
    drop(snapshot);
    let plan = optimize(lower(&query).unwrap(), &catalog);
    let scans_before = writer.unique_index().populate_scans();
    let probes_before = writer.unique_index().probes();
    let outcome = execute_write_staged(&plan, &mut writer, &Params::new())
        .await
        .unwrap();
    assert_eq!(outcome.rows.len(), 499, "staged account 1 moved to 1001");
    assert_eq!(
        writer.unique_index().populate_scans() - scans_before,
        1,
        "numeric UNIQUE MATCH must populate the RYOW tuple map once"
    );
    assert!(
        writer.unique_index().probes() >= probes_before + 1_000,
        "each float key probes its strict F64 and Cypher-equal I64 variants"
    );

    let null_query =
        parse("UNWIND [NULL] AS i MATCH (a:Account {account_no: i}) SET a.null_hit = true")
            .unwrap();
    let null_plan = optimize(lower(&null_query).unwrap(), &catalog);
    let null_outcome = execute_write_staged(&null_plan, &mut writer, &Params::new())
        .await
        .unwrap();
    assert_eq!(
        null_outcome.rows.len(),
        0,
        "property equality with NULL must never bind a node"
    );
    writer.discard_batch();
}

#[tokio::test]
async fn unique_endpoint_matches_batch_for_relationship_merge_loader() {
    let mut writer = WriterSession::open(store(), paths("w-match-unique-string-sidecar"))
        .await
        .unwrap();
    write_q(
        &mut writer,
        "UNWIND range(1, 500) AS i CREATE (:Account {key: toString(i)})",
    )
    .await;
    let schema = SchemaBuilder::new()
        .label(LabelDef {
            name: "Account".into(),
            properties: vec![PropertyDef::new("key", DataType::Utf8, false)
                .unwrap()
                .with_unique(true)],
        })
        .unwrap()
        .build();
    writer.flush(schema).await.unwrap();

    let query = parse(
        "UNWIND $rows AS row \
         MATCH (a:Account {key: row.source}) \
         MATCH (b:Account {key: row.target}) \
         MERGE (a)-[r:SEEN {codigo: row.codigo}]->(b) \
         SET r.relacion = row.codigo \
         RETURN a.key AS source_key, b.key AS target_key, row.codigo AS codigo",
    )
    .unwrap();
    let snapshot = writer.snapshot();
    let catalog = StatsCatalog::from_manifest(&snapshot.manifest().manifest);
    drop(snapshot);
    let plan = optimize(lower(&query).unwrap(), &catalog);

    fn count_unique_lookups(plan: &namidb_query::LogicalPlan) -> usize {
        usize::from(matches!(
            plan,
            namidb_query::LogicalPlan::NodeByPropertyValue {
                label,
                property,
                multi: false,
                ..
            } if label == "Account" && property == "key"
        )) + plan
            .children()
            .iter()
            .map(|child| count_unique_lookups(child))
            .sum::<usize>()
    }
    assert_eq!(
        count_unique_lookups(&plan),
        2,
        "optimized loader plan must retain one unique point lookup per endpoint: {plan:?}"
    );

    let rel_row = |source: &str, target: &str, codigo: &str| {
        RuntimeValue::Map(BTreeMap::from([
            ("source".into(), RuntimeValue::String(source.into())),
            ("target".into(), RuntimeValue::String(target.into())),
            ("codigo".into(), RuntimeValue::String(codigo.into())),
        ]))
    };
    let mut params = Params::new();
    params.insert(
        "rows".into(),
        RuntimeValue::List(vec![
            rel_row("3", "4", "r1"),
            rel_row("missing", "5", "missing-source"),
            rel_row("1", "missing", "missing-target"),
            rel_row("3", "4", "r1"),
            rel_row("2", "3", "r2"),
        ]),
    );
    let scans_before = writer.unique_index().populate_scans();
    let point_reads_before = writer.property_index_cache().unique_lookup_calls();
    let outcome = execute_write(&plan, &mut writer, &params).await.unwrap();
    let matched_codes = outcome
        .rows
        .iter()
        .map(|row| match row.get("codigo") {
            Some(RuntimeValue::String(code)) => code.as_str(),
            other => panic!("expected relationship code, got {other:?}"),
        })
        .collect::<Vec<_>>();
    assert_eq!(
        matched_codes,
        vec!["r1", "r1", "r2"],
        "endpoint batches must preserve input order, duplicate hits and misses"
    );
    assert_eq!(
        outcome.edges_created, 2,
        "duplicate MERGE rows must share one relationship"
    );
    assert_eq!(
        writer.unique_index().populate_scans(),
        scans_before,
        "committed String MATCH must use sidecars/cache, not full-label tx population"
    );
    assert_eq!(
        writer.property_index_cache().unique_lookup_calls() - point_reads_before,
        2,
        "all correlated String keys must use one storage batch per endpoint"
    );

    let repeat_point_reads_before = writer.property_index_cache().unique_lookup_calls();
    let repeated = execute_write(&plan, &mut writer, &params).await.unwrap();
    assert_eq!(repeated.rows.len(), 3);
    assert_eq!(
        repeated.edges_created, 0,
        "the second loader pass must hit both existing relationships"
    );
    assert_eq!(
        writer.property_index_cache().unique_lookup_calls() - repeat_point_reads_before,
        2,
        "existing-edge MERGE must still use one node batch per endpoint"
    );

    let read_query =
        parse("UNWIND $keys AS key MATCH (a:Account {key: key}) RETURN a.key AS matched_key")
            .unwrap();
    let read_plan = optimize(lower(&read_query).unwrap(), &catalog);
    assert_eq!(
        count_unique_lookups(&read_plan),
        1,
        "optimized correlated read must retain its unique point lookup: {read_plan:?}"
    );
    let mut read_params = Params::new();
    read_params.insert(
        "keys".into(),
        RuntimeValue::List(vec![
            RuntimeValue::String("3".into()),
            RuntimeValue::String("missing".into()),
            RuntimeValue::Null,
            RuntimeValue::String("1".into()),
            RuntimeValue::String("3".into()),
        ]),
    );
    let read_point_reads_before = writer.property_index_cache().unique_lookup_calls();
    let snapshot = writer.snapshot();
    let rows = execute(&read_plan, &snapshot, &read_params).await.unwrap();
    drop(snapshot);
    let matched_keys = rows
        .iter()
        .map(|row| match row.get("matched_key") {
            Some(RuntimeValue::String(key)) => key.as_str(),
            other => panic!("expected matched key, got {other:?}"),
        })
        .collect::<Vec<_>>();
    assert_eq!(matched_keys, vec!["3", "1", "3"]);
    assert_eq!(
        writer.property_index_cache().unique_lookup_calls() - read_point_reads_before,
        1,
        "read-only correlated MATCH must use one storage batch"
    );
}

#[tokio::test]
async fn correlated_unique_match_set_batches_existing_node_updates() {
    const ROWS: usize = 200;
    let mut writer = WriterSession::open(store(), paths("w-match-set-existing-batch"))
        .await
        .unwrap();
    write_q(
        &mut writer,
        "UNWIND range(1, 2000) AS i \
         CREATE (:Articulo {key: toString(i), titulo: 'old'})",
    )
    .await;
    let schema = SchemaBuilder::new()
        .label(LabelDef {
            name: "Articulo".into(),
            properties: vec![PropertyDef::new("key", DataType::Utf8, false)
                .unwrap()
                .with_unique(true)],
        })
        .unwrap()
        .build();
    writer.flush(schema).await.unwrap();

    let query = parse(
        "UNWIND $rows AS row \
         MATCH (a:Articulo {key: row.key}) \
         SET a.embedding = row.embedding, a.titulo = row.titulo \
         RETURN a.key AS key",
    )
    .unwrap();
    let snapshot = writer.snapshot();
    let catalog = StatsCatalog::from_manifest(&snapshot.manifest().manifest);
    drop(snapshot);
    let plan = optimize(lower(&query).unwrap(), &catalog);

    let rows = (1..=ROWS)
        .map(|i| {
            RuntimeValue::Map(BTreeMap::from([
                ("key".into(), RuntimeValue::String(i.to_string())),
                (
                    "embedding".into(),
                    RuntimeValue::Vector(vec![i as f32, 1.0, -1.0]),
                ),
                (
                    "titulo".into(),
                    RuntimeValue::String(format!("articulo-{i}")),
                ),
            ]))
        })
        .collect();
    let mut params = Params::new();
    params.insert("rows".into(), RuntimeValue::List(rows));

    let scans_before = writer.unique_index().populate_scans();
    let point_reads_before = writer.property_index_cache().unique_lookup_calls();
    let outcome = execute_write(&plan, &mut writer, &params).await.unwrap();
    assert_eq!(outcome.rows.len(), ROWS);
    assert_eq!(outcome.properties_set, (ROWS * 2) as u64);
    assert_eq!(
        writer.unique_index().populate_scans(),
        scans_before,
        "UNWIND MATCH+SET must not populate the transactional index by scanning :Articulo"
    );
    assert_eq!(
        writer.property_index_cache().unique_lookup_calls() - point_reads_before,
        1,
        "all existing keys in a correlated node update must use one storage batch"
    );

    let snapshot = writer.snapshot();
    let updated = snapshot
        .lookup_node_by_property("Articulo", "key", "200")
        .await
        .unwrap()
        .expect("updated article");
    assert_eq!(
        updated.properties.get("titulo"),
        Some(&namidb_core::Value::Str("articulo-200".into()))
    );
    assert_eq!(
        updated.properties.get("embedding"),
        Some(&namidb_core::Value::Vec(vec![200.0, 1.0, -1.0]))
    );
}

#[tokio::test]
async fn write_only_vector_update_discards_rows_and_embedding_results() {
    // Large enough to cross several default correlated-write chunks while
    // remaining practical in the integration suite. Every committed source
    // node already carries a wide vector: this is the incremental
    // re-vectorisation case that previously retained the complete old corpus
    // slice plus a second copy of the request.
    const ROWS: usize = 2_000;
    const DIMENSIONS: usize = 1024;

    let mut writer = WriterSession::open(store(), paths("w-match-set-vector-discard"))
        .await
        .unwrap();
    for i in 1..=ROWS {
        writer
            .upsert_node(
                "Articulo",
                NodeId::new(),
                &NodeWriteRecord {
                    properties: BTreeMap::from([
                        ("key".into(), CoreValue::Str(i.to_string())),
                        (
                            "embedding".into(),
                            CoreValue::Vec(vec![-(i as f32); DIMENSIONS]),
                        ),
                        ("titulo".into(), CoreValue::Str("old".into())),
                    ]),
                    schema_version: 1,
                    ..Default::default()
                },
            )
            .unwrap();
    }
    writer.commit_batch().await.unwrap();
    let schema = SchemaBuilder::new()
        .label(LabelDef {
            name: "Articulo".into(),
            properties: vec![
                PropertyDef::new("key", DataType::Utf8, false)
                    .unwrap()
                    .with_unique(true),
                PropertyDef::new(
                    "embedding",
                    DataType::FloatVector {
                        dim: DIMENSIONS as u32,
                    },
                    false,
                )
                .unwrap(),
                PropertyDef::new("titulo", DataType::Utf8, false).unwrap(),
            ],
        })
        .unwrap()
        .build();
    writer.flush(schema.clone()).await.unwrap();

    let query = parse(
        "UNWIND $rows AS row \
         MATCH (a:Articulo {key: row.key}) \
         SET a.embedding = row.embedding, a.titulo = row.titulo",
    )
    .unwrap();
    let snapshot = writer.snapshot();
    let catalog = StatsCatalog::from_manifest(&snapshot.manifest().manifest);
    drop(snapshot);
    let plan = optimize(lower(&query).unwrap(), &catalog);
    assert!(
        matches!(plan, namidb_query::LogicalPlan::DiscardResult { .. }),
        "write-only statement must keep its explicit result sink after optimization: {plan:?}"
    );

    let mut rows = (1..=ROWS)
        .map(|i| {
            let embedding = (0..DIMENSIONS)
                .map(|dimension| i as f32 + dimension as f32 / 1024.0)
                .collect();
            RuntimeValue::Map(BTreeMap::from([
                ("key".into(), RuntimeValue::String(i.to_string())),
                ("embedding".into(), RuntimeValue::Vector(embedding)),
                (
                    "titulo".into(),
                    RuntimeValue::String(format!("articulo-{i}")),
                ),
            ]))
        })
        .collect::<Vec<_>>();
    // Cross-chunk duplicate: the final rewrite must observe the staged value
    // from the earlier chunk (RYOW) and win without rehydrating/storing every
    // prior row.
    rows.push(RuntimeValue::Map(BTreeMap::from([
        ("key".into(), RuntimeValue::String("1".into())),
        (
            "embedding".into(),
            RuntimeValue::Vector(vec![42.0; DIMENSIONS]),
        ),
        (
            "titulo".into(),
            RuntimeValue::String("articulo-1-final".into()),
        ),
    ])));
    // A miss must stay a miss without forcing the staged-aware fallback or
    // being confused with a prior chunk's local overlay.
    rows.push(RuntimeValue::Map(BTreeMap::from([
        ("key".into(), RuntimeValue::String("does-not-exist".into())),
        (
            "embedding".into(),
            RuntimeValue::Vector(vec![777.0; DIMENSIONS]),
        ),
        (
            "titulo".into(),
            RuntimeValue::String("must-not-be-written".into()),
        ),
    ])));
    let update_rows = rows.len();
    let matched_update_rows = ROWS + 1; // all originals + one duplicate
    let mut params = Params::new();
    params.insert("rows".into(), RuntimeValue::List(rows));

    let point_reads_before = writer.property_index_cache().unique_lookup_calls();
    let staged_scan_rows_before = writer.staged_unique_overlay_rows_scanned();
    let outcome = execute_write(&plan, &mut writer, &params).await.unwrap();
    assert!(
        outcome.rows.is_empty(),
        "a write without RETURN must not expose its internal matched rows"
    );
    assert_eq!(
        outcome.rows.capacity(),
        0,
        "the sink must not allocate an output batch that retains cloned embeddings"
    );
    assert_eq!(
        outcome.properties_set,
        (matched_update_rows * 2) as u64,
        "the missing key must not stage either SET"
    );
    let chunk_rows = outcome.correlated_write_chunk_rows as usize;
    assert!(
        chunk_rows > 0,
        "the write-only correlated SET must select the bounded path"
    );
    assert!(
        outcome.correlated_write_peak_hydrated_rows <= chunk_rows as u64,
        "hydrated old NodeViews escaped the configured chunk: peak={} cap={chunk_rows}",
        outcome.correlated_write_peak_hydrated_rows
    );
    assert!(
        outcome.correlated_write_peak_materialized_rows <= chunk_rows as u64,
        "materialized executor rows escaped the configured chunk: peak={} cap={chunk_rows}",
        outcome.correlated_write_peak_materialized_rows
    );
    assert_eq!(
        outcome.correlated_write_lookup_batches,
        update_rows.div_ceil(chunk_rows) as u64,
        "each chunk must issue exactly one sidecar-backed point batch"
    );
    assert_eq!(
        writer.property_index_cache().unique_lookup_calls() - point_reads_before,
        outcome.correlated_write_lookup_batches,
        "the observable storage batches must agree with the executor counter"
    );
    assert_eq!(
        outcome.correlated_write_local_overlay_hits, 1,
        "only the deliberate cross-chunk duplicate should use the scalar local overlay"
    );
    assert_eq!(
        writer.staged_unique_overlay_rows_scanned(),
        staged_scan_rows_before,
        "the bounded clean-transaction path must never rescan accumulated staged embeddings"
    );
    assert_eq!(
        writer.staged_memtable_len(),
        0,
        "commit must release the transaction-local LWW overlay"
    );

    let snapshot = writer.snapshot();
    let updated = snapshot
        .lookup_node_by_property("Articulo", "key", &ROWS.to_string())
        .await
        .unwrap()
        .expect("updated article");
    let expected_embedding = (0..DIMENSIONS)
        .map(|dimension| ROWS as f32 + dimension as f32 / 1024.0)
        .collect();
    assert_eq!(
        updated.properties.get("embedding"),
        Some(&namidb_core::Value::Vec(expected_embedding))
    );
    assert_eq!(
        updated.properties.get("titulo"),
        Some(&namidb_core::Value::Str(format!("articulo-{ROWS}")))
    );

    let duplicate = snapshot
        .lookup_node_by_property("Articulo", "key", "1")
        .await
        .unwrap()
        .expect("duplicate-key article");
    assert_eq!(
        duplicate.properties.get("embedding"),
        Some(&namidb_core::Value::Vec(vec![42.0; DIMENSIONS])),
        "the final cross-chunk duplicate must win"
    );
    assert_eq!(
        duplicate.properties.get("titulo"),
        Some(&namidb_core::Value::Str("articulo-1-final".into()))
    );
    drop(snapshot);

    // Flush must release the complete updated-vector memtable, not leave one
    // retained payload per existing node behind.
    assert!(
        writer.memtable_bytes() > 0,
        "the committed updates should be visible in the live memtable before flush"
    );
    writer.flush(schema).await.unwrap();
    assert_eq!(
        writer.memtable_bytes(),
        0,
        "flush must release committed update payloads"
    );

    // A later-row expression error can arrive after earlier rows (and even an
    // earlier SET on the failing row) have staged more than one complete
    // lookup chunk. The auto-commit statement boundary must still discard
    // every chunk atomically.
    let failing_query = parse(
        "UNWIND $rows AS row \
         MATCH (a:Articulo {key: row.key}) \
         SET a.embedding = row.embedding, a.titulo = toString(10 / row.divisor)",
    )
    .unwrap();
    let snapshot = writer.snapshot();
    let catalog = StatsCatalog::from_manifest(&snapshot.manifest().manifest);
    drop(snapshot);
    let failing_plan = optimize(lower(&failing_query).unwrap(), &catalog);
    let mut failing_params = Params::new();
    let failing_row_count = chunk_rows + 1;
    let failing_rows = (0..failing_row_count)
        .map(|offset| {
            let key = (offset % ROWS) + 1;
            RuntimeValue::Map(BTreeMap::from([
                ("key".into(), RuntimeValue::String(key.to_string())),
                (
                    "embedding".into(),
                    RuntimeValue::Vector(vec![99.0 + offset as f32; DIMENSIONS]),
                ),
                (
                    "divisor".into(),
                    RuntimeValue::Integer(if offset + 1 == failing_row_count {
                        0
                    } else {
                        2
                    }),
                ),
            ]))
        })
        .collect();
    failing_params.insert("rows".into(), RuntimeValue::List(failing_rows));
    let staged_scan_rows_before_failure = writer.staged_unique_overlay_rows_scanned();
    let error = execute_write(&failing_plan, &mut writer, &failing_params)
        .await
        .expect_err("division by zero must abort the complete correlated update");
    assert!(
        error.to_string().contains("division by zero"),
        "unexpected correlated update error: {error}"
    );
    assert_eq!(
        writer.staged_memtable_len(),
        0,
        "a late error must release every staged wide-node payload"
    );
    assert_eq!(
        writer.staged_unique_overlay_rows_scanned(),
        staged_scan_rows_before_failure,
        "rollback coverage must retain the O(total rows) local-overlay path"
    );

    let snapshot = writer.snapshot();
    let rolled_back = snapshot
        .lookup_node_by_property("Articulo", "key", "2")
        .await
        .unwrap()
        .expect("rolled-back article");
    let expected_embedding = (0..DIMENSIONS)
        .map(|dimension| 2.0 + dimension as f32 / 1024.0)
        .collect();
    assert_eq!(
        rolled_back.properties.get("embedding"),
        Some(&namidb_core::Value::Vec(expected_embedding)),
        "an error after staging rows must not leak a partial vector rewrite"
    );
    assert_eq!(
        rolled_back.properties.get("titulo"),
        Some(&namidb_core::Value::Str("articulo-2".into()))
    );
}

#[tokio::test]
async fn write_only_unique_merge_updates_wide_nodes_in_bounded_chunks() {
    const ROWS: usize = 2_000;
    const DIMENSIONS: usize = 1024;

    let mut writer = WriterSession::open(store(), paths("w-merge-vector-discard"))
        .await
        .unwrap();
    for i in 1..=ROWS {
        writer
            .upsert_node(
                "Articulo",
                NodeId::new(),
                &NodeWriteRecord {
                    properties: BTreeMap::from([
                        ("key".into(), CoreValue::Str(i.to_string())),
                        (
                            "embedding".into(),
                            CoreValue::Vec(vec![-(i as f32); DIMENSIONS]),
                        ),
                        ("titulo".into(), CoreValue::Str("old".into())),
                        ("branch".into(), CoreValue::Str("seed".into())),
                    ]),
                    schema_version: 1,
                    ..Default::default()
                },
            )
            .unwrap();
    }
    writer.commit_batch().await.unwrap();
    let schema = SchemaBuilder::new()
        .label(LabelDef {
            name: "Articulo".into(),
            properties: vec![PropertyDef::new("key", DataType::Utf8, false)
                .unwrap()
                .with_unique(true)],
        })
        .unwrap()
        .build();
    writer.flush(schema.clone()).await.unwrap();

    let query = parse(
        "UNWIND $rows AS row \
         MERGE (a:Articulo {key: row.key}) \
         ON CREATE SET a.branch = 'created' \
         ON MATCH SET a.branch = 'matched' \
         SET a.embedding = row.embedding, a.titulo = row.titulo",
    )
    .unwrap();
    let snapshot = writer.snapshot();
    let catalog = StatsCatalog::from_manifest(&snapshot.manifest().manifest);
    drop(snapshot);
    let plan = optimize(lower(&query).unwrap(), &catalog);
    assert!(
        matches!(plan, namidb_query::LogicalPlan::DiscardResult { .. }),
        "write-only MERGE must retain the terminal result sink: {plan:?}"
    );

    // A new key before the first chunk plus existing/new duplicates after the
    // final complete chunk exercise both MERGE branches and cross-chunk RYOW.
    let mut rows = vec![RuntimeValue::Map(BTreeMap::from([
        ("key".into(), RuntimeValue::String("new".into())),
        (
            "embedding".into(),
            RuntimeValue::Vector(vec![7.0; DIMENSIONS]),
        ),
        ("titulo".into(), RuntimeValue::String("new-first".into())),
    ]))];
    rows.extend((1..=ROWS).map(|i| {
        RuntimeValue::Map(BTreeMap::from([
            ("key".into(), RuntimeValue::String(i.to_string())),
            (
                "embedding".into(),
                RuntimeValue::Vector(vec![i as f32; DIMENSIONS]),
            ),
            ("titulo".into(), RuntimeValue::String(format!("merge-{i}"))),
        ]))
    }));
    rows.push(RuntimeValue::Map(BTreeMap::from([
        ("key".into(), RuntimeValue::String("1".into())),
        (
            "embedding".into(),
            RuntimeValue::Vector(vec![41.0; DIMENSIONS]),
        ),
        (
            "titulo".into(),
            RuntimeValue::String("merge-1-final".into()),
        ),
    ])));
    rows.push(RuntimeValue::Map(BTreeMap::from([
        ("key".into(), RuntimeValue::String("new".into())),
        (
            "embedding".into(),
            RuntimeValue::Vector(vec![42.0; DIMENSIONS]),
        ),
        ("titulo".into(), RuntimeValue::String("new-final".into())),
    ])));
    let row_count = rows.len();
    let mut params = Params::new();
    params.insert("rows".into(), RuntimeValue::List(rows));

    let point_reads_before = writer.property_index_cache().unique_lookup_calls();
    let staged_scan_rows_before = writer.staged_unique_overlay_rows_scanned();
    let outcome = execute_write(&plan, &mut writer, &params).await.unwrap();
    assert!(outcome.rows.is_empty());
    assert_eq!(outcome.rows.capacity(), 0);
    assert_eq!(outcome.nodes_created, 1);
    assert_eq!(outcome.properties_set, (row_count * 3) as u64);
    let chunk_rows = outcome.correlated_write_chunk_rows as usize;
    assert!(chunk_rows > 0, "MERGE must use the bounded terminal path");
    assert!(
        outcome.correlated_write_peak_hydrated_rows <= chunk_rows as u64,
        "MERGE hydrated peak {} exceeded chunk {chunk_rows}",
        outcome.correlated_write_peak_hydrated_rows
    );
    assert!(
        outcome.correlated_write_peak_materialized_rows <= chunk_rows as u64,
        "MERGE materialized peak {} exceeded chunk {chunk_rows}",
        outcome.correlated_write_peak_materialized_rows
    );
    assert_eq!(
        outcome.correlated_write_lookup_batches,
        row_count.div_ceil(chunk_rows) as u64
    );
    assert_eq!(
        writer.property_index_cache().unique_lookup_calls() - point_reads_before,
        outcome.correlated_write_lookup_batches
    );
    assert_eq!(outcome.correlated_write_local_overlay_hits, 2);
    assert_eq!(
        writer.staged_unique_overlay_rows_scanned(),
        staged_scan_rows_before,
        "bounded MERGE must not walk accumulated staged vector records"
    );
    assert_eq!(writer.staged_memtable_len(), 0);

    let snapshot = writer.snapshot();
    let existing = snapshot
        .lookup_node_by_property("Articulo", "key", "1")
        .await
        .unwrap()
        .expect("existing MERGE row");
    assert_eq!(
        existing.properties.get("embedding"),
        Some(&CoreValue::Vec(vec![41.0; DIMENSIONS]))
    );
    assert_eq!(
        existing.properties.get("titulo"),
        Some(&CoreValue::Str("merge-1-final".into()))
    );
    assert_eq!(
        existing.properties.get("branch"),
        Some(&CoreValue::Str("matched".into()))
    );
    let created = snapshot
        .lookup_node_by_property("Articulo", "key", "new")
        .await
        .unwrap()
        .expect("new MERGE row");
    assert_eq!(
        created.properties.get("embedding"),
        Some(&CoreValue::Vec(vec![42.0; DIMENSIONS]))
    );
    assert_eq!(
        created.properties.get("branch"),
        Some(&CoreValue::Str("matched".into())),
        "the repeated new key must take ON MATCH after its ON CREATE row"
    );
    drop(snapshot);

    writer.flush(schema).await.unwrap();
    assert_eq!(writer.memtable_bytes(), 0);

    // Fail after more than one chunk, including a newly-created key. The
    // auto-commit boundary must release every wide staged payload and the
    // created node.
    let failing_query = parse(
        "UNWIND $rows AS row \
         MERGE (a:Articulo {key: row.key}) \
         ON MATCH SET a.embedding = row.embedding \
         ON CREATE SET a.embedding = row.embedding \
         SET a.titulo = toString(10 / row.divisor)",
    )
    .unwrap();
    let snapshot = writer.snapshot();
    let catalog = StatsCatalog::from_manifest(&snapshot.manifest().manifest);
    drop(snapshot);
    let failing_plan = optimize(lower(&failing_query).unwrap(), &catalog);
    let failing_row_count = chunk_rows + 1;
    let failing_rows = (0..failing_row_count)
        .map(|offset| {
            let key = if offset == 0 {
                "rollback-new".to_string()
            } else {
                ((offset % ROWS) + 1).to_string()
            };
            RuntimeValue::Map(BTreeMap::from([
                ("key".into(), RuntimeValue::String(key)),
                (
                    "embedding".into(),
                    RuntimeValue::Vector(vec![99.0 + offset as f32; DIMENSIONS]),
                ),
                (
                    "divisor".into(),
                    RuntimeValue::Integer(if offset + 1 == failing_row_count {
                        0
                    } else {
                        2
                    }),
                ),
            ]))
        })
        .collect();
    let mut failing_params = Params::new();
    failing_params.insert("rows".into(), RuntimeValue::List(failing_rows));
    let staged_scan_rows_before_failure = writer.staged_unique_overlay_rows_scanned();
    let error = execute_write(&failing_plan, &mut writer, &failing_params)
        .await
        .expect_err("late MERGE expression failure must roll back every chunk");
    assert!(error.to_string().contains("division by zero"));
    assert_eq!(writer.staged_memtable_len(), 0);
    assert_eq!(
        writer.staged_unique_overlay_rows_scanned(),
        staged_scan_rows_before_failure
    );
    assert!(
        writer
            .snapshot()
            .lookup_node_by_property("Articulo", "key", "rollback-new")
            .await
            .unwrap()
            .is_none(),
        "the new-key branch must not survive a late rollback"
    );
}

#[tokio::test]
async fn write_only_unique_merge_key_mutation_uses_exact_fallback() {
    let mut writer = WriterSession::open(store(), paths("w-merge-key-mutation-fallback"))
        .await
        .unwrap();
    writer
        .upsert_node(
            "Articulo",
            NodeId::new(),
            &NodeWriteRecord {
                properties: BTreeMap::from([("key".into(), CoreValue::Str("fallback".into()))]),
                schema_version: 1,
                ..Default::default()
            },
        )
        .unwrap();
    writer.commit_batch().await.unwrap();
    let schema = SchemaBuilder::new()
        .label(LabelDef {
            name: "Articulo".into(),
            properties: vec![PropertyDef::new("key", DataType::Utf8, false)
                .unwrap()
                .with_unique(true)],
        })
        .unwrap()
        .build();
    writer.flush(schema).await.unwrap();

    let plan = lower(
        &parse(
            "UNWIND $rows AS row \
             MERGE (a:Articulo {key: row.key}) \
             ON MATCH SET a.key = row.next \
             ON CREATE SET a.marker = 'created'",
        )
        .unwrap(),
    )
    .unwrap();
    let mut params = Params::new();
    params.insert(
        "rows".into(),
        RuntimeValue::List(vec![
            RuntimeValue::Map(BTreeMap::from([
                ("key".into(), RuntimeValue::String("fallback".into())),
                (
                    "next".into(),
                    RuntimeValue::String("fallback-renamed".into()),
                ),
            ])),
            RuntimeValue::Map(BTreeMap::from([
                ("key".into(), RuntimeValue::String("fallback".into())),
                ("next".into(), RuntimeValue::String("fallback-final".into())),
            ])),
        ]),
    );
    let outcome = execute_write(&plan, &mut writer, &params).await.unwrap();
    assert_eq!(
        outcome.correlated_write_chunk_rows, 0,
        "lookup-key mutation must reject the specialised path"
    );
    assert_eq!(outcome.nodes_created, 1);
    let snapshot = writer.snapshot();
    assert!(snapshot
        .lookup_node_by_property("Articulo", "key", "fallback")
        .await
        .unwrap()
        .is_some());
    assert!(snapshot
        .lookup_node_by_property("Articulo", "key", "fallback-renamed")
        .await
        .unwrap()
        .is_some());
    assert!(snapshot
        .lookup_node_by_property("Articulo", "key", "fallback-final")
        .await
        .unwrap()
        .is_none());
    drop(snapshot);

    let prior_write = lower(&parse("CREATE (:Articulo {key: 'prior-staged'})").unwrap()).unwrap();
    execute_write_staged(&prior_write, &mut writer, &Params::new())
        .await
        .unwrap();
    let safe_merge = lower(
        &parse(
            "UNWIND $rows AS row \
             MERGE (a:Articulo {key: row.key}) \
             SET a.marker = 'safe'",
        )
        .unwrap(),
    )
    .unwrap();
    let mut safe_params = Params::new();
    safe_params.insert(
        "rows".into(),
        RuntimeValue::List(vec![RuntimeValue::Map(BTreeMap::from([(
            "key".into(),
            RuntimeValue::String("fallback".into()),
        )]))]),
    );
    let staged_outcome = execute_write_staged(&safe_merge, &mut writer, &safe_params)
        .await
        .unwrap();
    assert_eq!(
        staged_outcome.correlated_write_chunk_rows, 0,
        "a transaction with prior staged node writes must retain the canonical RYOW path"
    );
    writer.discard_batch();
}

#[tokio::test]
async fn write_only_union_discards_each_wide_branch_without_losing_effects() {
    const DIMENSIONS: usize = 1024;

    let mut writer = WriterSession::open(store(), paths("w-union-vector-discard"))
        .await
        .unwrap();
    let query = parse(
        "UNWIND $left AS row \
         CREATE (:Articulo {key: row.key, embedding: row.embedding}) \
         UNION \
         UNWIND $right AS row \
         CREATE (:Articulo {key: row.key, embedding: row.embedding})",
    )
    .unwrap();
    let plan = lower(&query).unwrap();
    assert!(
        matches!(plan, namidb_query::LogicalPlan::DiscardResult { .. }),
        "a write-only UNION must have one terminal sink: {plan:?}"
    );

    let wide_row = |key: &str, offset: f32| {
        RuntimeValue::Map(BTreeMap::from([
            ("key".into(), RuntimeValue::String(key.into())),
            (
                "embedding".into(),
                RuntimeValue::Vector(
                    (0..DIMENSIONS)
                        .map(|dimension| offset + dimension as f32)
                        .collect(),
                ),
            ),
        ]))
    };
    let mut params = Params::new();
    params.insert(
        "left".into(),
        RuntimeValue::List(vec![wide_row("left", 1.0)]),
    );
    params.insert(
        "right".into(),
        RuntimeValue::List(vec![wide_row("right", 2.0)]),
    );

    let outcome = execute_write(&plan, &mut writer, &params).await.unwrap();
    assert_eq!(outcome.nodes_created, 2);
    assert!(outcome.rows.is_empty());
    assert_eq!(
        outcome.rows.capacity(),
        0,
        "UNION/DISTINCT must not rebuild a terminal result batch"
    );
    let snapshot = writer.snapshot();
    assert_eq!(snapshot.scan_label("Articulo").await.unwrap().len(), 2);
}

#[tokio::test]
async fn result_sink_evaluates_terminal_projection_errors_and_rolls_back() {
    let mut writer = WriterSession::open(store(), paths("w-discard-project-error"))
        .await
        .unwrap();
    let plan = lower(&parse("CREATE (:Audit {key: 'must-rollback'}) WITH 1 / 0 AS boom").unwrap())
        .unwrap();
    assert!(matches!(
        plan,
        namidb_query::LogicalPlan::DiscardResult { .. }
    ));

    let error = execute_write(&plan, &mut writer, &Params::new())
        .await
        .expect_err("discarding rows must not discard expression errors");
    assert!(
        error.to_string().contains("division by zero"),
        "unexpected error: {error}"
    );
    assert!(
        writer
            .snapshot()
            .scan_label("Audit")
            .await
            .unwrap()
            .is_empty(),
        "a terminal projection error must roll back the staged CREATE"
    );
}

#[tokio::test]
async fn result_sink_evaluates_cross_product_inputs_before_unit_subquery_writes() {
    let mut writer = WriterSession::open(store(), paths("w-discard-cross-error"))
        .await
        .unwrap();
    let plan =
        lower(&parse("UNWIND 1 AS i CALL { CREATE (:Audit {key: 'must-not-run'}) }").unwrap())
            .unwrap();
    assert!(matches!(
        plan,
        namidb_query::LogicalPlan::DiscardResult { .. }
    ));

    let error = execute_write(&plan, &mut writer, &Params::new())
        .await
        .expect_err("invalid UNWIND must fail before the unit subquery write");
    assert!(
        error.to_string().contains("UNWIND requires a list"),
        "unexpected error: {error}"
    );
    assert!(
        writer
            .snapshot()
            .scan_label("Audit")
            .await
            .unwrap()
            .is_empty(),
        "the right side must not commit after the left side fails"
    );
}

#[tokio::test]
async fn terminal_correlated_unit_subquery_discards_rows_but_runs_once_per_outer_row() {
    let mut writer = WriterSession::open(store(), paths("w-unit-call-discard"))
        .await
        .unwrap();
    let plan = lower(
        &parse(
            "UNWIND [1, 2, 3] AS i \
             CALL { WITH i CREATE (:Audit {key: toString(i)}) }",
        )
        .unwrap(),
    )
    .unwrap();
    assert!(
        matches!(plan, namidb_query::LogicalPlan::DiscardResult { .. }),
        "terminal unit CALL must be wrapped only at the outer result boundary: {plan:?}"
    );

    let outcome = execute_write(&plan, &mut writer, &Params::new())
        .await
        .unwrap();
    assert_eq!(outcome.nodes_created, 3);
    assert!(outcome.rows.is_empty());
    assert_eq!(outcome.rows.capacity(), 0);
    assert_eq!(
        writer.snapshot().scan_label("Audit").await.unwrap().len(),
        3
    );
}

#[tokio::test]
async fn unit_subquery_under_return_preserves_outer_cardinality() {
    let mut writer = WriterSession::open(store(), paths("w-unit-call-cardinality"))
        .await
        .unwrap();
    let plan = lower(
        &parse(
            "UNWIND [1, 2, 3] AS i \
             CALL { WITH i CREATE (:Audit {key: toString(i)}) } \
             RETURN i",
        )
        .unwrap(),
    )
    .unwrap();
    assert!(
        !matches!(plan, namidb_query::LogicalPlan::DiscardResult { .. }),
        "an outer RETURN must remain row-producing: {plan:?}"
    );

    let outcome = execute_write(&plan, &mut writer, &Params::new())
        .await
        .unwrap();
    assert_eq!(outcome.nodes_created, 3);
    assert_eq!(outcome.rows.len(), 3);
    assert_eq!(
        outcome
            .rows
            .iter()
            .filter_map(|row| match row.get("i") {
                Some(RuntimeValue::Integer(value)) => Some(*value),
                _ => None,
            })
            .collect::<Vec<_>>(),
        vec![1, 2, 3]
    );
}

#[tokio::test]
async fn correlated_unique_match_set_preserves_duplicates_misses_and_rollback() {
    let mut writer = WriterSession::open(store(), paths("w-match-set-batch-rollback"))
        .await
        .unwrap();
    write_q(
        &mut writer,
        "UNWIND ['a', 'b'] AS key \
         CREATE (:Articulo {key: key, titulo: 'original', revision: 0})",
    )
    .await;
    let schema = SchemaBuilder::new()
        .label(LabelDef {
            name: "Articulo".into(),
            properties: vec![PropertyDef::new("key", DataType::Utf8, false)
                .unwrap()
                .with_unique(true)],
        })
        .unwrap()
        .build();
    writer.flush(schema).await.unwrap();

    let query = parse(
        "UNWIND $rows AS row \
         MATCH (a:Articulo {key: row.key}) \
         SET a.titulo = row.titulo, a.revision = a.revision + 1 \
         RETURN a.key AS key, a.titulo AS titulo, a.revision AS revision",
    )
    .unwrap();
    let snapshot = writer.snapshot();
    let catalog = StatsCatalog::from_manifest(&snapshot.manifest().manifest);
    drop(snapshot);
    let plan = optimize(lower(&query).unwrap(), &catalog);
    let row = |key: &str, titulo: &str| {
        RuntimeValue::Map(BTreeMap::from([
            ("key".into(), RuntimeValue::String(key.into())),
            ("titulo".into(), RuntimeValue::String(titulo.into())),
        ]))
    };
    let params = Params::from([(
        "rows".into(),
        RuntimeValue::List(vec![
            row("a", "first"),
            row("missing", "ignored"),
            row("a", "second"),
            row("b", "third"),
        ]),
    )]);

    let scans_before = writer.unique_index().populate_scans();
    let point_reads_before = writer.property_index_cache().unique_lookup_calls();
    let outcome = execute_write_staged(&plan, &mut writer, &params)
        .await
        .unwrap();
    assert_eq!(outcome.rows.len(), 3, "the missing key must not reach SET");
    assert_eq!(outcome.properties_set, 6);
    assert_eq!(
        outcome
            .rows
            .iter()
            .map(
                |row| match (row.get("key"), row.get("titulo"), row.get("revision"),) {
                    (
                        Some(RuntimeValue::String(key)),
                        Some(RuntimeValue::String(titulo)),
                        Some(RuntimeValue::Integer(revision)),
                    ) => (key.as_str(), titulo.as_str(), *revision),
                    other => panic!("expected key/title strings, got {other:?}"),
                }
            )
            .collect::<Vec<_>>(),
        vec![("a", "first", 1), ("a", "second", 2), ("b", "third", 1)],
        "duplicate hits must retain input order and see the prior staged update"
    );
    assert_eq!(
        writer.unique_index().populate_scans(),
        scans_before,
        "String hit/miss/duplicate seeding must not populate by label scan"
    );
    assert_eq!(
        writer.property_index_cache().unique_lookup_calls() - point_reads_before,
        1,
        "the whole correlated input must issue one sidecar-backed storage batch"
    );

    writer.discard_batch();
    let snapshot = writer.snapshot();
    for key in ["a", "b"] {
        let node = snapshot
            .lookup_node_by_property("Articulo", "key", key)
            .await
            .unwrap()
            .expect("committed article survives rollback");
        assert_eq!(
            node.properties.get("titulo"),
            Some(&namidb_core::Value::Str("original".into())),
            "discard must restore the pre-batch record for {key}"
        );
        assert_eq!(
            node.properties.get("revision"),
            Some(&namidb_core::Value::I64(0)),
            "discard must restore the pre-batch revision for {key}"
        );
    }
}

#[tokio::test]
async fn unique_string_match_set_stays_transactional_across_auto_commits() {
    const ROWS: i64 = 128;
    let mut writer = WriterSession::open(store(), paths("w-match-unique-string-auto-commits"))
        .await
        .unwrap();
    write_q(
        &mut writer,
        "UNWIND range(1, 128) AS i CREATE (:Account {key: toString(i)})",
    )
    .await;
    let schema = SchemaBuilder::new()
        .label(LabelDef {
            name: "Account".into(),
            properties: vec![PropertyDef::new("key", DataType::Utf8, false)
                .unwrap()
                .with_unique(true)],
        })
        .unwrap()
        .build();
    writer.flush(schema).await.unwrap();

    let query = parse("MATCH (a:Account {key: $key}) SET a.touch = $touch").unwrap();
    let snapshot = writer.snapshot();
    let catalog = StatsCatalog::from_manifest(&snapshot.manifest().manifest);
    drop(snapshot);
    let plan = optimize(lower(&query).unwrap(), &catalog);
    let scans_before = writer.unique_index().populate_scans();
    let point_reads_before = writer.property_index_cache().unique_lookup_calls();
    for i in 1..=ROWS {
        let mut params = Params::new();
        params.insert("key".into(), RuntimeValue::String(i.to_string()));
        params.insert("touch".into(), RuntimeValue::Integer(i));
        let outcome = execute_write(&plan, &mut writer, &params).await.unwrap();
        assert!(outcome.rows.is_empty());
        assert_eq!(outcome.properties_set, 1);
    }
    assert_eq!(
        writer.unique_index().populate_scans(),
        scans_before,
        "MATCH(unique String)+SET must seed exact keys without a label scan"
    );
    assert_eq!(
        writer.property_index_cache().unique_lookup_calls() - point_reads_before,
        ROWS as u64,
        "each one-row statement should issue one sidecar-backed point batch"
    );
}

#[tokio::test]
async fn unique_string_match_delete_stays_transactional_across_auto_commits() {
    const ROWS: i64 = 128;
    let mut writer = WriterSession::open(store(), paths("w-delete-unique-string-auto-commits"))
        .await
        .unwrap();
    write_q(
        &mut writer,
        "UNWIND range(1, 128) AS i CREATE (:Account {key: toString(i)})",
    )
    .await;
    let schema = SchemaBuilder::new()
        .label(LabelDef {
            name: "Account".into(),
            properties: vec![PropertyDef::new("key", DataType::Utf8, false)
                .unwrap()
                .with_unique(true)],
        })
        .unwrap()
        .build();
    writer.flush(schema).await.unwrap();

    let query = parse("MATCH (a:Account {key: $key}) DELETE a").unwrap();
    let snapshot = writer.snapshot();
    let catalog = StatsCatalog::from_manifest(&snapshot.manifest().manifest);
    drop(snapshot);
    let plan = optimize(lower(&query).unwrap(), &catalog);
    let scans_before = writer.unique_index().populate_scans();
    let point_reads_before = writer.property_index_cache().unique_lookup_calls();
    for i in 1..=ROWS {
        let mut params = Params::new();
        params.insert("key".into(), RuntimeValue::String(i.to_string()));
        let outcome = execute_write(&plan, &mut writer, &params).await.unwrap();
        assert_eq!(outcome.nodes_deleted, 1);
    }
    assert_eq!(
        writer.unique_index().populate_scans(),
        scans_before,
        "MATCH(unique String)+DELETE must seed exact keys without a label scan"
    );
    assert_eq!(
        writer.property_index_cache().unique_lookup_calls() - point_reads_before,
        ROWS as u64,
        "each one-row DELETE should issue one sidecar-backed point batch"
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
async fn merge_prefers_preseeded_string_unique_key_over_composite_scan() {
    let mut writer = WriterSession::open(store(), paths("w-merge-preseed-before-composite"))
        .await
        .unwrap();
    write_q(
        &mut writer,
        "CREATE (:Account {key: 'k1', tenant: 't1', part: 'p1'})",
    )
    .await;
    let mut schema = SchemaBuilder::new()
        .label(LabelDef {
            name: "Account".into(),
            properties: vec![
                PropertyDef::new("key", DataType::Utf8, false)
                    .unwrap()
                    .with_unique(true),
                PropertyDef::new("tenant", DataType::Utf8, false).unwrap(),
                PropertyDef::new("part", DataType::Utf8, false).unwrap(),
            ],
        })
        .unwrap()
        .build();
    schema.constraints.push(Constraint {
        name: "uniq_account_tenant_part".into(),
        label: "Account".into(),
        properties: vec!["tenant".into(), "part".into()],
        kind: ConstraintKind::Unique,
    });
    writer.flush(schema).await.unwrap();

    let scans_before = writer.unique_index().populate_scans();
    let matched = write_q(
        &mut writer,
        "MERGE (n:Account {key: 'k1', tenant: 't1', part: 'p1'}) \
         ON MATCH SET n.seen = true RETURN n",
    )
    .await;
    assert_eq!(matched.nodes_created, 0);
    assert_eq!(
        writer.unique_index().populate_scans(),
        scans_before,
        "the batch-seeded String key must answer before the composite key can populate by scan"
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
async fn merge_non_unique_index_unwind_stays_transactional_in_fresh_store() {
    // Regression for a fresh loader namespace: before the writer-private
    // postings map and incremental staged memtable, every MERGE rebuilt and
    // scanned the growing RYOW overlay.
    let mut writer = WriterSession::open(store(), paths("w-merge-equality-fresh-unwind"))
        .await
        .unwrap();
    writer.create_property_index("Doc", "key").await.unwrap();

    let scans_before = writer.unique_index().populate_scans();
    let postings_before = writer.unique_index().posting_probes();
    let query = lower(
        &parse(
            "UNWIND range(1, 500) AS i \
             MERGE (d:Doc {key: toString(i)}) \
             SET d.payload = i",
        )
        .unwrap(),
    )
    .unwrap();
    let created = execute_write(&query, &mut writer, &Params::new())
        .await
        .unwrap();
    assert_eq!(created.nodes_created, 500);
    assert_eq!(
        writer.unique_index().populate_scans() - scans_before,
        1,
        "the growing staged overlay must populate its postings map once"
    );
    assert!(
        writer.unique_index().posting_probes() >= postings_before + 499,
        "every lookup after the first staged row must hit transactional postings"
    );
    assert_eq!(writer.staged_memtable_len(), 0, "auto-commit drains RYOW");

    // Commit and flush only change representation; the postings map remains a
    // valid baseline and an idempotent replay performs no corpus scan.
    let schema = writer.schema().clone();
    writer.flush(schema).await.unwrap();
    let scans_after_first = writer.unique_index().populate_scans();
    let replay = execute_write(&query, &mut writer, &Params::new())
        .await
        .unwrap();
    assert_eq!(replay.nodes_created, 0);
    assert_eq!(
        writer.unique_index().populate_scans(),
        scans_after_first,
        "commit + flush must preserve the warm non-unique postings map"
    );
}

#[tokio::test]
async fn non_unique_index_stays_transactional_across_one_row_auto_commits() {
    const ROWS: i64 = 128;

    // MERGE: every statement begins with an empty staged batch. The
    // writer-private postings map must still survive each commit; relying on
    // the shared committed cache would make every node commit invalidate it
    // and re-scan the growing memtable.
    let mut merge_writer = WriterSession::open(store(), paths("w-merge-equality-auto-commits"))
        .await
        .unwrap();
    merge_writer
        .create_property_index("Doc", "key")
        .await
        .unwrap();
    let merge_plan =
        lower(&parse("MERGE (d:Doc {key: $key}) SET d.touch = $touch").unwrap()).unwrap();
    let merge_scans_before = merge_writer.unique_index().populate_scans();
    let merge_probes_before = merge_writer.unique_index().posting_probes();
    for i in 1..=ROWS {
        let mut params = Params::new();
        params.insert("key".into(), RuntimeValue::String(i.to_string()));
        params.insert("touch".into(), RuntimeValue::Integer(i));
        let outcome = execute_write(&merge_plan, &mut merge_writer, &params)
            .await
            .unwrap();
        assert_eq!(outcome.nodes_created, 1);
    }
    assert_eq!(
        merge_writer.unique_index().populate_scans() - merge_scans_before,
        1,
        "one-row MERGE auto-commits must populate transactional postings once"
    );
    assert!(
        merge_writer.unique_index().posting_probes() >= merge_probes_before + ROWS as u64,
        "every MERGE auto-commit must probe the preserved postings map"
    );

    // MATCH+SET: this is also a write plan, so it must retain the same map
    // across commits even though the read itself happens before each SET.
    let mut match_writer = WriterSession::open(store(), paths("w-match-equality-auto-commits"))
        .await
        .unwrap();
    match_writer
        .create_property_index("Doc", "key")
        .await
        .unwrap();
    write_q(
        &mut match_writer,
        "UNWIND range(1, 128) AS i CREATE (:Doc {key: toString(i)})",
    )
    .await;
    let snapshot = match_writer.snapshot();
    let catalog = StatsCatalog::from_manifest(&snapshot.manifest().manifest);
    drop(snapshot);
    let match_plan = optimize(
        lower(&parse("MATCH (d:Doc {key: $key}) SET d.touch = $touch").unwrap()).unwrap(),
        &catalog,
    );
    let match_scans_before = match_writer.unique_index().populate_scans();
    let match_probes_before = match_writer.unique_index().posting_probes();
    for i in 1..=ROWS {
        let mut params = Params::new();
        params.insert("key".into(), RuntimeValue::String(i.to_string()));
        params.insert("touch".into(), RuntimeValue::Integer(i));
        let outcome = execute_write(&match_plan, &mut match_writer, &params)
            .await
            .unwrap();
        assert!(outcome.rows.is_empty());
        assert_eq!(outcome.properties_set, 1);
    }
    assert_eq!(
        match_writer.unique_index().populate_scans() - match_scans_before,
        1,
        "one-row MATCH+SET auto-commits must populate transactional postings once"
    );
    assert!(
        match_writer.unique_index().posting_probes() >= match_probes_before + ROWS as u64,
        "every MATCH auto-commit must probe the preserved postings map"
    );
}

#[tokio::test]
async fn indexed_match_unwind_after_staged_state_populates_once() {
    let mut writer = WriterSession::open(store(), paths("w-match-equality-ryow-unwind"))
        .await
        .unwrap();
    writer.create_property_index("Doc", "key").await.unwrap();
    write_q(
        &mut writer,
        "UNWIND range(1, 500) AS i CREATE (:Doc {key: toString(i)})",
    )
    .await;

    // Stage a value change before the MATCH statement, as an explicit Bolt
    // transaction would. The delegated read subplan must see it and must not
    // rebuild/scan the overlay for every correlated key.
    let staged = lower(&parse("MATCH (d:Doc {key: '1'}) SET d.key = 'one'").unwrap()).unwrap();
    execute_write_staged(&staged, &mut writer, &Params::new())
        .await
        .unwrap();
    let query = parse(
        "UNWIND range(1, 500) AS i \
         MATCH (d:Doc {key: toString(i)}) \
         SET d.seen = true \
         RETURN d",
    )
    .unwrap();
    let snapshot = writer.snapshot();
    let catalog = StatsCatalog::from_manifest(&snapshot.manifest().manifest);
    drop(snapshot);
    let plan = optimize(lower(&query).unwrap(), &catalog);

    let scans_before = writer.unique_index().populate_scans();
    let postings_before = writer.unique_index().posting_probes();
    let outcome = execute_write_staged(&plan, &mut writer, &Params::new())
        .await
        .unwrap();
    assert_eq!(outcome.rows.len(), 499, "staged rename removes key '1'");
    assert_eq!(
        writer.unique_index().populate_scans() - scans_before,
        1,
        "MATCH over a node-mutated overlay gets one transactional population"
    );
    assert!(
        writer.unique_index().posting_probes() >= postings_before + 500,
        "every correlated MATCH key must probe the transactional postings map"
    );
    writer.discard_batch();
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

#[tokio::test]
async fn merge_relationship_matches_compound_temporal_bytes_and_vector_properties() {
    let mut writer = WriterSession::open(store(), paths("w-merge-rel-complex-props"))
        .await
        .unwrap();
    let schema = SchemaBuilder::new()
        .label(LabelDef {
            name: "A".into(),
            properties: vec![PropertyDef::new("name", DataType::Utf8, false).unwrap()],
        })
        .unwrap()
        .label(LabelDef {
            name: "B".into(),
            properties: vec![PropertyDef::new("name", DataType::Utf8, false).unwrap()],
        })
        .unwrap()
        .edge_type(EdgeTypeDef {
            name: "R".into(),
            src_label: "A".into(),
            dst_label: "B".into(),
            properties: vec![
                PropertyDef::new("nullable", DataType::Json, true).unwrap(),
                PropertyDef::new("enabled", DataType::Bool, false).unwrap(),
                PropertyDef::new("count", DataType::Int64, false).unwrap(),
                PropertyDef::new("ratio", DataType::Float64, false).unwrap(),
                PropertyDef::new("name", DataType::Utf8, false).unwrap(),
                PropertyDef::new("bytes", DataType::Binary, false).unwrap(),
                PropertyDef::new("vector", DataType::FloatVector { dim: 3 }, false).unwrap(),
                PropertyDef::new("vector8", DataType::Int8Vector { dim: 3 }, false).unwrap(),
                PropertyDef::new("date", DataType::Date32, false).unwrap(),
                PropertyDef::new("datetime", DataType::TimestampMicrosUtc, false).unwrap(),
                PropertyDef::new("list", DataType::Json, false).unwrap(),
                PropertyDef::new("map", DataType::Json, false).unwrap(),
                PropertyDef::new("matched_in", DataType::Utf8, false).unwrap(),
            ],
        })
        .unwrap()
        .build();
    let complex_properties = RuntimeValue::Map(BTreeMap::from([
        ("nullable".into(), RuntimeValue::Null),
        ("enabled".into(), RuntimeValue::Bool(true)),
        ("count".into(), RuntimeValue::Integer(7)),
        ("ratio".into(), RuntimeValue::Float(1.5)),
        ("name".into(), RuntimeValue::String("typed".into())),
        ("bytes".into(), RuntimeValue::Bytes(vec![0, 1, 255])),
        ("vector".into(), RuntimeValue::Vector(vec![0.25, -1.5, 3.0])),
        (
            "vector8".into(),
            RuntimeValue::Vector8 {
                codes: vec![4, -7, 12],
                scale: 0.125,
            },
        ),
        ("date".into(), RuntimeValue::Date(20_000)),
        (
            "datetime".into(),
            RuntimeValue::DateTime(1_700_000_000_123_456),
        ),
        (
            "list".into(),
            RuntimeValue::List(vec![
                RuntimeValue::Integer(1),
                RuntimeValue::String("two".into()),
                RuntimeValue::Map(BTreeMap::from([("nested_null".into(), RuntimeValue::Null)])),
            ]),
        ),
        (
            "map".into(),
            RuntimeValue::Map(BTreeMap::from([
                ("nested_date".into(), RuntimeValue::Date(20_001)),
                (
                    "nested_list".into(),
                    RuntimeValue::List(vec![
                        RuntimeValue::Float(3.0),
                        RuntimeValue::Bytes(vec![4, 5]),
                    ]),
                ),
            ])),
        ),
    ]));
    let mut params = Params::new();
    params.insert("relprops".into(), complex_properties.clone());

    let create = lower(
        &parse(
            "CREATE (a:A {name: 'a'})-[r:R $relprops]->(b:B {name: 'b'}) \
             RETURN r",
        )
        .unwrap(),
    )
    .unwrap();
    let created = execute_write(&create, &mut writer, &params).await.unwrap();
    assert_eq!(created.edges_created, 1);
    let (src, dst) = match created.rows[0].get("r") {
        Some(RuntimeValue::Rel(relation)) => (relation.src, relation.dst),
        other => panic!("expected created relationship binding, got {other:?}"),
    };

    let merge = lower(
        &parse(
            "MATCH (a:A {name: 'a'}), (b:B {name: 'b'}) \
             MERGE (a)-[r:R $relprops]->(b) \
             ON MATCH SET r.matched_in = $phase \
             RETURN r",
        )
        .unwrap(),
    )
    .unwrap();
    let expected_properties = match &complex_properties {
        RuntimeValue::Map(properties) => properties.clone(),
        _ => unreachable!(),
    };
    let assert_matched = |outcome: &namidb_query::WriteOutcome, phase: &str| {
        assert_eq!(
            outcome.nodes_created, 0,
            "bound-endpoint relationship MERGE must not create nodes in {phase}"
        );
        assert_eq!(
            outcome.edges_created, 0,
            "every storable property shape must match without replacing the edge in {phase}"
        );
        assert_eq!(
            outcome.properties_set, 1,
            "the existing relationship must take the ON MATCH branch in {phase}"
        );
        let relation = match outcome.rows[0].get("r") {
            Some(RuntimeValue::Rel(relation)) => relation,
            other => panic!("expected relationship binding in {phase}, got {other:?}"),
        };
        assert_eq!(
            (relation.src, relation.dst),
            (src, dst),
            "MERGE must retain the relationship endpoints in {phase}"
        );
        let mut expected = expected_properties.clone();
        expected.insert("matched_in".into(), RuntimeValue::String(phase.to_string()));
        assert_eq!(
            relation.properties, expected,
            "ON MATCH must preserve every original property in {phase}"
        );
    };

    params.insert("phase".into(), RuntimeValue::String("memtable".into()));
    let memtable_match = execute_write(&merge, &mut writer, &params).await.unwrap();
    assert_matched(&memtable_match, "memtable");
    assert_eq!(writer.pending_len(), 0, "auto-commit must seal the match");

    writer.flush(schema.clone()).await.unwrap();
    params.insert("phase".into(), RuntimeValue::String("sst".into()));
    let sst_match = execute_write(&merge, &mut writer, &params).await.unwrap();
    assert_matched(&sst_match, "sst");

    // The ON MATCH update creates a second immutable version of the same edge.
    // Flush it, then compact both forward and inverse R buckets so the final
    // replay resolves the relationship and all of its properties from L1.
    writer.flush(schema.clone()).await.unwrap();
    let compacted = writer.compact_l0(&schema).await.unwrap();
    assert_eq!(
        compacted.source_ssts_removed, 4,
        "two L0 versions in each R direction should be compacted"
    );
    assert_eq!(
        compacted.new_ssts_written, 2,
        "compaction should write one forward and one inverse R SST"
    );

    params.insert("phase".into(), RuntimeValue::String("l1".into()));
    let compacted_match = execute_write(&merge, &mut writer, &params).await.unwrap();
    assert_matched(&compacted_match, "l1");
    writer.flush(schema).await.unwrap();

    let edges = writer
        .snapshot()
        .out_edges_via_sst("R", src)
        .await
        .unwrap()
        .edges;
    assert_eq!(
        edges.len(),
        1,
        "MERGE replays across memtable, SST, and L1 must not duplicate the edge"
    );
    assert_eq!(edges[0].dst, dst);
    let mut expected = expected_properties;
    expected.insert("matched_in".into(), RuntimeValue::String("l1".into()));
    let persisted = edges[0]
        .properties
        .iter()
        .map(|(key, value)| (key.clone(), RuntimeValue::from(value.clone())))
        .collect::<BTreeMap<_, _>>();
    assert_eq!(
        persisted, expected,
        "the final ON MATCH property map must survive commit and flush"
    );
}
