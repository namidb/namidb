//! A search filter's eligibility set is computed once per statement and
//! reused by later statements over the same node data, keyed by the filter's
//! resolved values: one statement shape with other `$param` values gets its
//! own set, the same values reuse it, and a committed write invalidates it.
//! (Its own test binary: the hit counter is process-wide.)

use std::collections::BTreeMap;
use std::sync::Arc;

use namidb_core::id::{NamespaceId, NodeId};
use namidb_core::value::Value as CoreValue;
use namidb_storage::{delete_namespace_objects, NamespacePaths, NodeWriteRecord, WriterSession};
use object_store::memory::InMemory;
use object_store::ObjectStore;

use namidb_query::{eligibility_cache_hits, execute, lower, parse, Params, RuntimeValue};

/// The hit counter is process-wide: tests that count hits run one at a time.
static SERIAL: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

async fn upsert(writer: &mut WriterSession, title: &str, cat: &str, emb: Vec<f32>) {
    upsert_id(writer, NodeId::new(), title, cat, emb).await;
}

async fn upsert_id(writer: &mut WriterSession, id: NodeId, title: &str, cat: &str, emb: Vec<f32>) {
    let mut p = BTreeMap::new();
    p.insert("title".into(), CoreValue::Str(title.into()));
    p.insert("cat".into(), CoreValue::Str(cat.into()));
    p.insert("embedding".into(), CoreValue::Vec(emb));
    writer
        .upsert_node(
            "Doc",
            id,
            &NodeWriteRecord {
                properties: p,
                schema_version: 1,
                ..Default::default()
            },
        )
        .unwrap();
    writer.commit_batch().await.unwrap();
}

async fn titles(writer: &WriterSession, cat: &str) -> Vec<String> {
    let cypher = "CALL search.vector({label: 'Doc', property: 'embedding', \
                  query: $q, k: 10, filter: {cat: $cat}}) \
                  YIELD node RETURN node.title AS title ORDER BY title";
    let snapshot = writer.snapshot();
    let plan = lower(&parse(cypher).unwrap()).unwrap();
    let mut params = Params::new();
    params.insert("q".into(), RuntimeValue::Vector(vec![1.0, 0.0]));
    params.insert("cat".into(), RuntimeValue::String(cat.into()));
    execute(&plan, &snapshot, &params)
        .await
        .unwrap()
        .iter()
        .filter_map(|row| match row.get("title") {
            Some(RuntimeValue::String(title)) => Some(title.clone()),
            _ => None,
        })
        .collect()
}

#[tokio::test]
async fn eligibility_cache_keys_on_bound_values_and_node_generation() {
    let _serial = SERIAL.lock().await;
    let store = Arc::new(InMemory::new());
    let paths = NamespacePaths::new("tenants", NamespaceId::new("eligibility-cache").unwrap());
    let mut writer = WriterSession::open(store, paths).await.unwrap();
    upsert(&mut writer, "a1", "a", vec![1.0, 0.0]).await;
    upsert(&mut writer, "a2", "a", vec![0.0, 1.0]).await;
    upsert(&mut writer, "b1", "b", vec![0.7, 0.7]).await;

    let start = eligibility_cache_hits();
    assert_eq!(titles(&writer, "a").await, ["a1", "a2"]);
    assert_eq!(
        eligibility_cache_hits(),
        start,
        "first statement computes the set"
    );
    // Same shape, other bound value: its own set, not the cached one.
    assert_eq!(titles(&writer, "b").await, ["b1"]);
    assert_eq!(eligibility_cache_hits(), start);
    // Same values again: both reuse their sets.
    assert_eq!(titles(&writer, "a").await, ["a1", "a2"]);
    assert_eq!(titles(&writer, "b").await, ["b1"]);
    assert_eq!(eligibility_cache_hits(), start + 2);

    // A committed node write advances the generation (nothing is flushed,
    // so the manifest version is unchanged): the next statement recomputes.
    // A new node entering the set:
    upsert(&mut writer, "a3", "a", vec![0.5, 0.5]).await;
    assert_eq!(titles(&writer, "a").await, ["a1", "a2", "a3"]);
    assert_eq!(eligibility_cache_hits(), start + 2);
    // ...and an existing node leaving one set for another.
    assert_eq!(titles(&writer, "b").await, ["b1"]);
    let moving = NodeId::new();
    upsert_id(&mut writer, moving, "a4", "a", vec![0.9, 0.1]).await;
    assert_eq!(titles(&writer, "a").await, ["a1", "a2", "a3", "a4"]);
    upsert_id(&mut writer, moving, "a4", "b", vec![0.9, 0.1]).await;
    assert_eq!(titles(&writer, "a").await, ["a1", "a2", "a3"]);
    assert_eq!(titles(&writer, "b").await, ["a4", "b1"]);
    assert_eq!(
        eligibility_cache_hits(),
        start + 2,
        "every write recomputes"
    );
}

/// Generations are seeded from the manifest version, so two writer sessions
/// over a namespace of the same name reach the same version and generation:
/// one namespace dropped and created again in the same process, or two stores
/// that share a namespace name. Their sets must never be shared.
#[tokio::test]
async fn eligibility_cache_is_never_shared_between_writer_sessions() {
    let _serial = SERIAL.lock().await;
    let paths = || NamespacePaths::new("tenants", NamespaceId::new("eligibility-reuse").unwrap());

    // Dropped and recreated: same store, same namespace, objects deleted.
    let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
    let mut first = WriterSession::open(store.clone(), paths()).await.unwrap();
    upsert(&mut first, "first", "a", vec![1.0, 0.0]).await;
    assert_eq!(titles(&first, "a").await, ["first"]);
    drop(first);
    assert!(delete_namespace_objects(&store, &paths()).await.unwrap() > 0);
    let mut recreated = WriterSession::open(store, paths()).await.unwrap();
    upsert(&mut recreated, "recreated", "a", vec![1.0, 0.0]).await;
    assert_eq!(titles(&recreated, "a").await, ["recreated"]);

    // Another store with the same namespace name, open at the same time.
    let mut other = WriterSession::open(Arc::new(InMemory::new()), paths())
        .await
        .unwrap();
    upsert(&mut other, "other", "a", vec![1.0, 0.0]).await;
    assert_eq!(titles(&other, "a").await, ["other"]);
    assert_eq!(titles(&recreated, "a").await, ["recreated"]);
}
