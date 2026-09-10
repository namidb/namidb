//! Plan item 18 (docs/testing/25tb-readiness.md): no exec-level traversal
//! ever crossed a FLUSHED high-degree supernode — skew buckets and dense
//! blocks were unit-tested, but MATCH/Expand/var-length never touched a hub
//! through the snapshot API. A 25 TB graph will have plenty of hubs.

use std::collections::BTreeMap;
use std::sync::Arc;

use namidb_core::id::{NamespaceId, NodeId};
use namidb_core::schema::{DataType, EdgeTypeDef, LabelDef, PropertyDef, Schema, SchemaBuilder};
use namidb_core::value::Value as CoreValue;
use namidb_storage::{EdgeWriteRecord, NamespacePaths, NodeWriteRecord, WriterSession};
use object_store::memory::InMemory;
use object_store::ObjectStore;

use namidb_query::{execute, lower, parse, Params, RuntimeValue};

const FANOUT: usize = 2_500;
const FANIN: usize = 400;

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

async fn count_value(writer: &WriterSession, query: &str) -> i64 {
    let snapshot = writer.snapshot();
    let parsed = parse(query).unwrap();
    let plan = lower(&parsed).unwrap();
    let rows = execute(&plan, &snapshot, &Params::new()).await.unwrap();
    assert_eq!(rows.len(), 1);
    match rows[0].bindings.values().next() {
        Some(RuntimeValue::Integer(count)) => *count,
        other => panic!("expected an integer count, got {other:?}"),
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn traversals_cross_a_flushed_supernode_exactly() {
    let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
    let paths = NamespacePaths::new("tenants", NamespaceId::new("exec-supernode").unwrap());
    let mut writer = WriterSession::open(store, paths).await.unwrap();

    let hub = NodeId::new();
    writer.upsert_node("Person", hub, &person("hub")).unwrap();
    let feeder = NodeId::new();
    writer
        .upsert_node("Person", feeder, &person("feeder"))
        .unwrap();
    writer.upsert_edge("KNOWS", feeder, hub, &edge()).unwrap();

    // FANOUT outgoing spokes and FANIN incoming ones, all flushed so every
    // hop below serves from the paged edge SSTs (dense buckets included).
    for ordinal in 0..FANOUT {
        let spoke = NodeId::new();
        writer
            .upsert_node("Person", spoke, &person(&format!("out-{ordinal}")))
            .unwrap();
        writer.upsert_edge("KNOWS", hub, spoke, &edge()).unwrap();
    }
    for ordinal in 0..FANIN {
        let spoke = NodeId::new();
        writer
            .upsert_node("Person", spoke, &person(&format!("in-{ordinal}")))
            .unwrap();
        writer.upsert_edge("KNOWS", spoke, hub, &edge()).unwrap();
    }
    writer.commit_batch().await.unwrap();
    writer.flush(schema()).await.unwrap();

    // Single hop out of the hub: the full dense partner list.
    let outgoing = count_value(
        &writer,
        "MATCH (h:Person {name: 'hub'})-[:KNOWS]->(x:Person) RETURN count(*) AS c",
    )
    .await;
    assert_eq!(outgoing, FANOUT as i64);

    // Single hop INTO the hub: the inverse dense list (feeder + fan-in).
    let incoming = count_value(
        &writer,
        "MATCH (h:Person {name: 'hub'})<-[:KNOWS]-(x:Person) RETURN count(*) AS c",
    )
    .await;
    assert_eq!(incoming, (FANIN + 1) as i64);

    // Two directed hops THROUGH the hub from one feeder: every outgoing
    // spoke exactly once.
    let through = count_value(
        &writer,
        "MATCH (f:Person {name: 'feeder'})-[:KNOWS*2..2]->(x:Person) \
         RETURN count(*) AS c",
    )
    .await;
    assert_eq!(through, FANOUT as i64);

    // Aggregate pushdown across the whole type at hub scale.
    let total = count_value(
        &writer,
        "MATCH (:Person)-[r:KNOWS]->(:Person) RETURN count(r) AS c",
    )
    .await;
    assert_eq!(total, (FANOUT + FANIN + 1) as i64);
}

/// A hub expansion whose target carries NO label must cost the same as a
/// labelled one and return the same rows.
///
/// The 2.6.2 fan-out fix let a single hop consume its own prewarmed batch
/// instead of re-reading each endpoint through the shared FIFO node cache,
/// but it was gated on the target being labelled. An unlabelled or anonymous
/// target — `(a)-[:R]->()`, one of the most common patterns in Cypher — fell
/// back to one authoritative point read PER EDGE. Measured on a 197k-degree
/// hub: 3.5s labelled versus a 300s timeout unlabelled, for the same answer.
/// The batch resolves by id across every node descriptor, so an empty label
/// is a complete id-primary batch rather than a miss.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn unlabelled_and_anonymous_hub_targets_match_the_labelled_form() {
    let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
    let paths = NamespacePaths::new("tenants", NamespaceId::new("exec-unlabelled").unwrap());
    let mut writer = WriterSession::open(store, paths).await.unwrap();

    let hub = NodeId::new();
    writer.upsert_node("Person", hub, &person("hub")).unwrap();
    for ordinal in 0..FANOUT {
        let spoke = NodeId::new();
        writer
            .upsert_node("Person", spoke, &person(&format!("out-{ordinal}")))
            .unwrap();
        writer.upsert_edge("KNOWS", hub, spoke, &edge()).unwrap();
    }
    writer.commit_batch().await.unwrap();
    writer.flush(schema()).await.unwrap();

    // All three spellings describe the same set of edges.
    let labelled = count_value(
        &writer,
        "MATCH (h:Person {name: 'hub'})-[:KNOWS]->(x:Person) RETURN count(*) AS c",
    )
    .await;
    let unlabelled = count_value(
        &writer,
        "MATCH (h:Person {name: 'hub'})-[:KNOWS]->(x) RETURN count(*) AS c",
    )
    .await;
    let anonymous = count_value(
        &writer,
        "MATCH (h:Person {name: 'hub'})-[:KNOWS]->() RETURN count(*) AS c",
    )
    .await;
    assert_eq!(labelled, FANOUT as i64);
    assert_eq!(
        unlabelled, labelled,
        "an unlabelled target must not change the result"
    );
    assert_eq!(
        anonymous, labelled,
        "an anonymous target must not change the result"
    );

    // The unlabelled binding must still carry real property values, so the
    // batch cannot be answering with id-only stubs.
    let snapshot = writer.snapshot();
    let parsed =
        parse("MATCH (h:Person {name: 'hub'})-[:KNOWS]->(x) RETURN x.name AS n ORDER BY n LIMIT 3")
            .unwrap();
    let plan = lower(&parsed).unwrap();
    let rows = namidb_query::execute(&plan, &snapshot, &Params::new())
        .await
        .unwrap();
    assert_eq!(rows.len(), 3);
    for row in &rows {
        match row.bindings.get("n") {
            Some(RuntimeValue::String(name)) => {
                assert!(name.starts_with("out-"), "unexpected spoke name {name}")
            }
            other => panic!("unlabelled target lost its properties: {other:?}"),
        }
    }
}

/// A hub whose neighbours carry DIFFERENT labels, with the target label
/// referenced downstream.
///
/// `batch_lookup_nodes` sweeps by id but FILTERS its output by the label it
/// is given, so asking for `:Green` drops every `:Red` neighbour from the
/// batch. Those ids are then absent from the Views map, and the miss branch
/// falls through to `scan_node_for_id` — one uncached point read PER EDGE.
/// On a 40k-degree hub split evenly between two labels, this took
/// `-[:KNOWS]->(t:Green) RETURN count(t)` past a 120s deadline where 2.6.3
/// answered in 0.22s. The expand batch therefore asks for NO label and
/// re-proves the label itself, per edge, against the view it gets.
///
/// The timing cannot be asserted here (an in-memory store makes the point
/// read cheap), but nothing in the suite expanded a hub whose neighbours had
/// mixed labels at all, which is why the regression shipped.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn mixed_label_neighbours_resolve_without_per_edge_reads() {
    let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
    let paths = NamespacePaths::new("tenants", NamespaceId::new("exec-mixed-labels").unwrap());
    let mut writer = WriterSession::open(store, paths).await.unwrap();

    let hub = NodeId::new();
    writer.upsert_node("Person", hub, &person("hub")).unwrap();
    // Half the neighbours carry the label the query asks for, half do not.
    for ordinal in 0..FANIN {
        let green = NodeId::new();
        writer
            .upsert_node("Green", green, &person(&format!("green-{ordinal}")))
            .unwrap();
        writer.upsert_edge("KNOWS", hub, green, &edge()).unwrap();
        let red = NodeId::new();
        writer
            .upsert_node("Red", red, &person(&format!("red-{ordinal}")))
            .unwrap();
        writer.upsert_edge("KNOWS", hub, red, &edge()).unwrap();
    }
    writer.commit_batch().await.unwrap();
    writer.flush(schema()).await.unwrap();

    // Referencing the target forces the materialising path (an unreferenced
    // target takes the cheap membership sidecar and would not exercise this).
    let green = count_value(
        &writer,
        "MATCH (h:Person {name: 'hub'})-[:KNOWS]->(t:Green) RETURN count(t) AS c",
    )
    .await;
    assert_eq!(
        green, FANIN as i64,
        "every :Green neighbour must be counted"
    );

    let red = count_value(
        &writer,
        "MATCH (h:Person {name: 'hub'})-[:KNOWS]->(t:Red) RETURN count(t) AS c",
    )
    .await;
    assert_eq!(red, FANIN as i64);

    // A label no neighbour carries must return zero, not the whole fan-out.
    let absent = count_value(
        &writer,
        "MATCH (h:Person {name: 'hub'})-[:KNOWS]->(t:Absent) RETURN count(t) AS c",
    )
    .await;
    assert_eq!(absent, 0);

    // Unlabelled sees both halves.
    let all = count_value(
        &writer,
        "MATCH (h:Person {name: 'hub'})-[:KNOWS]->(t) RETURN count(t) AS c",
    )
    .await;
    assert_eq!(all, (FANIN * 2) as i64);
}

/// An UNLABELLED expansion target must be proven to exist, and deleting a
/// target must remove its row.
///
/// An unlabelled target used to pay a full row decode per endpoint purely to
/// establish that the endpoint was there — the label-membership sidecar
/// proves "carries label L" and there is no label to prove. It is now proven
/// through `try_batch_nodes_exist`, which ORs across the labels each
/// descriptor actually holds (a node can never carry zero labels). At degree
/// 160,000 that took `count(*)` over `()` from 2.2s to 0.53s.
///
/// The correctness half is what this pins: the existence proof must agree
/// with hydration, before AND after a delete.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn unlabelled_targets_are_proven_to_exist_and_track_deletes() {
    let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
    let paths = NamespacePaths::new("tenants", NamespaceId::new("exec-existence").unwrap());
    let mut writer = WriterSession::open(store, paths).await.unwrap();

    let hub = NodeId::new();
    writer.upsert_node("Person", hub, &person("hub")).unwrap();
    let mut spokes = Vec::new();
    for ordinal in 0..FANIN {
        let spoke = NodeId::new();
        writer
            .upsert_node("Person", spoke, &person(&format!("spoke-{ordinal}")))
            .unwrap();
        writer.upsert_edge("KNOWS", hub, spoke, &edge()).unwrap();
        spokes.push(spoke);
    }
    writer.commit_batch().await.unwrap();
    writer.flush(schema()).await.unwrap();

    // Every spelling must agree, and agree with hydration.
    let anonymous = count_value(
        &writer,
        "MATCH (h:Person {name: 'hub'})-[:KNOWS]->() RETURN count(*) AS c",
    )
    .await;
    let unlabelled = count_value(
        &writer,
        "MATCH (h:Person {name: 'hub'})-[:KNOWS]->(t) RETURN count(*) AS c",
    )
    .await;
    let hydrated = count_value(
        &writer,
        "MATCH (h:Person {name: 'hub'})-[:KNOWS]->(t) RETURN count(t.name) AS c",
    )
    .await;
    assert_eq!(anonymous, FANIN as i64);
    assert_eq!(unlabelled, anonymous);
    assert_eq!(
        hydrated, anonymous,
        "the existence proof must agree with hydration"
    );

    // Delete a third of the targets, then re-check every spelling.
    let removed = FANIN / 3;
    for spoke in spokes.iter().take(removed) {
        writer.tombstone_edge("KNOWS", hub, *spoke).unwrap();
        writer.tombstone_node("Person", *spoke).unwrap();
    }
    writer.commit_batch().await.unwrap();
    writer.flush(schema()).await.unwrap();

    let expected = (FANIN - removed) as i64;
    for query in [
        "MATCH (h:Person {name: 'hub'})-[:KNOWS]->() RETURN count(*) AS c",
        "MATCH (h:Person {name: 'hub'})-[:KNOWS]->(t) RETURN count(*) AS c",
        "MATCH (h:Person {name: 'hub'})-[:KNOWS]->(t) RETURN count(t.name) AS c",
        "MATCH (h:Person {name: 'hub'})-[:KNOWS]->(t:Person) RETURN count(*) AS c",
    ] {
        assert_eq!(
            count_value(&writer, query).await,
            expected,
            "after deleting {removed} targets: {query}"
        );
    }
}
