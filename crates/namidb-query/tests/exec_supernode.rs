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
    let parsed = parse(
        "MATCH (h:Person {name: 'hub'})-[:KNOWS]->(x) RETURN x.name AS n ORDER BY n LIMIT 3",
    )
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

