//! Composite tuple sidecars (phase 2 of the tuple posting index): flush
//! harvests one PagedV1 `tuple -> [id]` sidecar per declared composite
//! index from the reconciled row stream, compaction re-emits it from the
//! merged winners, and the DDL-backfill arm of the migration predicate
//! forces a rewrite of any pre-existing SST that lacks the declared tuple
//! (the item-38 rule generalized to declaration-ordered member lists).

use std::collections::BTreeMap;
use std::sync::Arc;

use namidb_core::id::{NamespaceId, NodeId};
use namidb_core::schema::{DataType, IndexDef, LabelDef, PropertyDef, Schema, SchemaBuilder};
use namidb_core::value::Value;
use namidb_storage::manifest::{CompositeEqualityIndexDescriptor, SstKind};
use namidb_storage::{NamespacePaths, NodeWriteRecord, WriterSession};
use object_store::memory::InMemory;
use object_store::ObjectStore;

fn plain_schema() -> Schema {
    SchemaBuilder::new()
        .label(LabelDef {
            name: "Person".into(),
            properties: vec![
                PropertyDef::new("city", DataType::Utf8, false).unwrap(),
                PropertyDef::new("age", DataType::Int64, false).unwrap(),
            ],
        })
        .unwrap()
        .build()
}

fn indexed_schema() -> Schema {
    let mut schema = plain_schema();
    schema.indexes.push(IndexDef {
        name: "pair".into(),
        label: "Person".into(),
        properties: vec!["city".into(), "age".into()],
    });
    schema
}

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

async fn open(store: &Arc<dyn ObjectStore>, name: &str) -> (WriterSession, NamespacePaths) {
    let paths = NamespacePaths::new("tenants", NamespaceId::new(name).unwrap());
    let writer = WriterSession::open(store.clone(), paths.clone())
        .await
        .unwrap();
    (writer, paths)
}

/// Every node-SST descriptor's composite entries, flattened.
fn composite_entries(writer: &WriterSession) -> Vec<CompositeEqualityIndexDescriptor> {
    let snap = writer.snapshot();
    snap.manifest()
        .manifest
        .ssts
        .iter()
        .filter(|d| d.kind == SstKind::Nodes)
        .flat_map(|d| d.composite_equality_indices.iter().cloned())
        .collect()
}

fn node_sst_count(writer: &WriterSession) -> usize {
    let snap = writer.snapshot();
    snap.manifest()
        .manifest
        .ssts
        .iter()
        .filter(|d| d.kind == SstKind::Nodes)
        .count()
}

#[tokio::test]
async fn flush_emits_a_tuple_sidecar_per_composite_index() {
    let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
    let (mut w, _) = open(&store, "cix-flush").await;

    // Three distinct tuples across four complete rows; the fifth row is
    // missing a member and must stay unfiled without poisoning the sidecar.
    w.upsert_node("Person", NodeId::new(), &person("quito", 30))
        .unwrap();
    w.upsert_node("Person", NodeId::new(), &person("quito", 30))
        .unwrap();
    w.upsert_node("Person", NodeId::new(), &person("quito", 31))
        .unwrap();
    w.upsert_node("Person", NodeId::new(), &person("lima", 30))
        .unwrap();
    let mut partial: BTreeMap<String, Value> = BTreeMap::new();
    partial.insert("city".into(), Value::Str("cuenca".into()));
    w.upsert_node(
        "Person",
        NodeId::new(),
        &NodeWriteRecord {
            properties: partial,
            schema_version: 1,
            ..Default::default()
        },
    )
    .unwrap();
    w.commit_batch().await.unwrap();
    w.flush(indexed_schema()).await.unwrap();

    let entries = composite_entries(&w);
    assert_eq!(entries.len(), 1, "one declared index -> one sidecar entry");
    let entry = &entries[0];
    assert_eq!(
        entry.properties,
        ["city".to_string(), "age".to_string()],
        "descriptor must keep DECLARATION order"
    );
    assert!(
        entry.path.contains(".eqtix_city+age.pidx"),
        "tuple sidecars use the composite naming, got {}",
        entry.path
    );
    assert_eq!(
        entry.distinct_values, 3,
        "four complete rows over three tuples; the partial row is unfiled"
    );
    assert!(entry.mixed_type_complete);
    assert!(!entry.paged_build_unsupported);
    assert!(entry.size_bytes > 0);
}

#[tokio::test]
async fn ddl_after_flush_backfills_via_compaction_then_settles() {
    let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
    let (mut w, _) = open(&store, "cix-backfill").await;

    w.upsert_node("Person", NodeId::new(), &person("quito", 30))
        .unwrap();
    w.upsert_node("Person", NodeId::new(), &person("lima", 44))
        .unwrap();
    w.commit_batch().await.unwrap();
    // Flushed BEFORE the composite index existed: no tuple coverage.
    w.flush(plain_schema()).await.unwrap();
    assert!(
        composite_entries(&w).is_empty(),
        "pre-DDL SSTs advertise no composite sidecars"
    );

    // The DDL-triggered sweep runs with the index in the schema; the
    // migration predicate must force the single-SST rewrite (this is the
    // whole backfill mechanism — no separate builder exists).
    let outcome = w.compact_l0(&indexed_schema()).await.unwrap();
    assert!(
        outcome.new_ssts_written >= 1,
        "an uncovered SST must be rewritten, not skipped"
    );
    let entries = composite_entries(&w);
    assert_eq!(entries.len(), 1);
    assert_eq!(
        entries[0].properties,
        ["city".to_string(), "age".to_string()]
    );
    assert_eq!(entries[0].distinct_values, 2);

    // Once covered, the same schema must NOT rewrite again (the predicate
    // is satisfied, so the sweep settles instead of looping forever).
    let settled = w.compact_l0(&indexed_schema()).await.unwrap();
    assert_eq!(
        settled.new_ssts_written, 0,
        "a covered SST must not be rewritten by the same declaration"
    );
}

#[tokio::test]
async fn compaction_reemits_tuples_from_the_reconciled_winners() {
    let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
    let (mut w, _) = open(&store, "cix-merge").await;

    // First generation: the target holds ("quito", 30).
    let target = NodeId::new();
    w.upsert_node("Person", target, &person("quito", 30))
        .unwrap();
    w.commit_batch().await.unwrap();
    w.flush(indexed_schema()).await.unwrap();

    // Second generation: the target moves to ("cuenca", 30) and a sibling
    // arrives; the superseded quito tuple must NOT survive the merge.
    w.upsert_node("Person", target, &person("cuenca", 30))
        .unwrap();
    w.upsert_node("Person", NodeId::new(), &person("lima", 44))
        .unwrap();
    w.commit_batch().await.unwrap();
    w.flush(indexed_schema()).await.unwrap();

    w.compact_l0(&indexed_schema()).await.unwrap();
    assert_eq!(
        node_sst_count(&w),
        1,
        "both generations merged into one SST"
    );
    let entries = composite_entries(&w);
    assert_eq!(entries.len(), 1);
    assert_eq!(
        entries[0].distinct_values, 2,
        "winners are (cuenca,30) and (lima,44); the superseded tuple is gone"
    );
    assert!(
        entries[0].path.contains(".eqtix_city+age.pidx"),
        "compaction keeps the composite naming, got {}",
        entries[0].path
    );
}
