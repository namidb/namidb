//! Composite tuple probe (phase 3): `indexed_node_ids_by_property_tuple`
//! serves declared composite indexes from TupleV1 sidecars, unions the
//! memtable delta, confirms every candidate member-by-member (dropping
//! superseded postings and separating lossy-numeric collisions), and
//! declines — never lies — when any relevant SST lacks coverage.

use std::collections::BTreeMap;
use std::sync::Arc;

use namidb_core::id::{NamespaceId, NodeId};
use namidb_core::schema::Schema;
use namidb_core::value::Value;
use namidb_storage::{route_telemetry, NamespacePaths, NodeWriteRecord, WriterSession};
use object_store::memory::InMemory;
use object_store::ObjectStore;

fn person(city: &str, age: i64) -> NodeWriteRecord {
    let mut props: BTreeMap<String, Value> = BTreeMap::new();
    props.insert("city".into(), Value::Str(city.into()));
    props.insert("age".into(), Value::I64(age));
    NodeWriteRecord {
        properties: props,
        schema_version: 1,
        ..Default::default()
    }
}

async fn open(store: &Arc<dyn ObjectStore>, name: &str) -> WriterSession {
    let paths = NamespacePaths::new("tenants", NamespaceId::new(name).unwrap());
    WriterSession::open(store.clone(), paths).await.unwrap()
}

/// The manifest's schema AFTER DDL — the schema flush/compaction must run
/// with, exactly as the server's maintenance loop passes it.
fn committed_schema(w: &WriterSession) -> Schema {
    w.snapshot().manifest().manifest.schema.clone()
}

fn pair() -> Vec<String> {
    vec!["city".into(), "age".into()]
}

async fn probe(w: &WriterSession, label: &str, values: &[Value]) -> Option<Vec<NodeId>> {
    let snapshot = w.snapshot();
    let mut ids = snapshot
        .indexed_node_ids_by_property_tuple(label, &pair(), values)
        .await
        .unwrap()?;
    ids.sort_unstable();
    Some(ids)
}

fn sorted(mut ids: Vec<NodeId>) -> Vec<NodeId> {
    ids.sort_unstable();
    ids
}

#[tokio::test]
async fn tuple_probe_unions_sst_and_memtable_and_confirms() {
    let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
    let mut w = open(&store, "tp-union").await;

    let a = NodeId::new();
    let b = NodeId::new();
    w.upsert_node("Person", a, &person("quito", 30)).unwrap();
    w.upsert_node("Person", b, &person("lima", 44)).unwrap();
    w.commit_batch().await.unwrap();
    w.create_composite_index_named(None, "Person", &["city".into(), "age".into()], false)
        .await
        .unwrap();
    let schema = committed_schema(&w);
    w.flush(schema).await.unwrap();

    // Memtable delta after the flush: a new claimant of the probed tuple,
    // and `b` moves AWAY from (lima, 44) — its SST posting is now stale.
    let c = NodeId::new();
    w.upsert_node("Person", c, &person("quito", 30)).unwrap();
    w.upsert_node("Person", b, &person("cuzco", 44)).unwrap();
    w.commit_batch().await.unwrap();

    let before = route_telemetry::snapshot();
    assert_eq!(
        probe(&w, "Person", &[Value::Str("quito".into()), Value::I64(30)]).await,
        Some(sorted(vec![a, c])),
        "flushed and memtable claimants must union"
    );
    assert_eq!(
        probe(&w, "Person", &[Value::Str("lima".into()), Value::I64(44)]).await,
        Some(Vec::new()),
        "a superseded SST posting must be dropped by confirmation, authoritatively"
    );
    // Cypher coerces integer = float: the float probe must find the same
    // rows the flat scan would.
    assert_eq!(
        probe(
            &w,
            "Person",
            &[Value::Str("quito".into()), Value::F64(30.0)]
        )
        .await,
        Some(sorted(vec![a, c])),
        "numeric members canonicalize across I64/F64"
    );
    // The any-label scope serves too (harvest follows the stored value).
    assert_eq!(
        probe(&w, "", &[Value::Str("quito".into()), Value::I64(30)]).await,
        Some(sorted(vec![a, c])),
    );
    let after = route_telemetry::snapshot();
    assert!(
        after.tuple_native >= before.tuple_native + 4,
        "each native serve must stamp the tuple route"
    );

    // `member = NULL` never matches, authoritatively.
    assert_eq!(
        probe(&w, "Person", &[Value::Str("quito".into()), Value::Null]).await,
        Some(Vec::new()),
    );
    // Reversed member order is NOT this index: declaration order is the key
    // layout, so the caller gets a decline, not wrong postings.
    let snapshot = w.snapshot();
    assert!(snapshot
        .indexed_node_ids_by_property_tuple(
            "Person",
            &["age".into(), "city".into()],
            &[Value::I64(30), Value::Str("quito".into())],
        )
        .await
        .unwrap()
        .is_none());
    // An undeclared pair declines outright.
    assert!(snapshot
        .indexed_node_ids_by_property_tuple(
            "Person",
            &["city".into(), "name".into()],
            &[Value::Str("quito".into()), Value::Str("x".into())],
        )
        .await
        .unwrap()
        .is_none());
}

#[tokio::test]
async fn tuple_probe_declines_without_coverage_then_serves_after_backfill() {
    let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
    let mut w = open(&store, "tp-backfill").await;

    let a = NodeId::new();
    w.upsert_node("Person", a, &person("quito", 30)).unwrap();
    w.upsert_node("Person", NodeId::new(), &person("lima", 44))
        .unwrap();
    w.commit_batch().await.unwrap();
    // Flushed BEFORE the index existed: the SST has no tuple sidecar.
    let pre_ddl = committed_schema(&w);
    w.flush(pre_ddl).await.unwrap();
    w.create_composite_index_named(None, "Person", &["city".into(), "age".into()], false)
        .await
        .unwrap();

    let before = route_telemetry::snapshot();
    assert_eq!(
        probe(&w, "Person", &[Value::Str("quito".into()), Value::I64(30)]).await,
        None,
        "an uncovered SST must decline the whole lookup, not lose its rows"
    );
    let after = route_telemetry::snapshot();
    assert!(
        after.tuple_fallback > before.tuple_fallback,
        "the coverage decline must stamp the fallback route"
    );

    // The DDL-triggered sweep backfills the sidecar; the probe now serves.
    let post_ddl = committed_schema(&w);
    w.compact_l0(&post_ddl).await.unwrap();
    assert_eq!(
        probe(&w, "Person", &[Value::Str("quito".into()), Value::I64(30)]).await,
        Some(vec![a]),
        "after the backfill rewrite the tuple route must serve"
    );
}

#[tokio::test]
async fn tuple_probe_is_authoritative_on_a_memtable_only_store() {
    let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
    let mut w = open(&store, "tp-memtable").await;

    let a = NodeId::new();
    w.upsert_node("Person", a, &person("quito", 30)).unwrap();
    w.upsert_node("Person", NodeId::new(), &person("lima", 44))
        .unwrap();
    w.commit_batch().await.unwrap();
    w.create_composite_index_named(None, "Person", &["city".into(), "age".into()], false)
        .await
        .unwrap();

    // No SSTs at all: the memtable IS the store and the answer is complete.
    assert_eq!(
        probe(&w, "Person", &[Value::Str("quito".into()), Value::I64(30)]).await,
        Some(vec![a]),
    );
    assert_eq!(
        probe(&w, "Person", &[Value::Str("quito".into()), Value::I64(31)]).await,
        Some(Vec::new()),
    );
}
