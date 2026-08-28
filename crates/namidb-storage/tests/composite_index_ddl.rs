//! Composite `CREATE INDEX` DDL (phase 1 of the tuple posting index): the
//! schema records a named, declaration-ordered `IndexDef`, the definition
//! survives a session reopen from the manifest, and the guard rails
//! (IF NOT EXISTS as a set match, duplicate names, duplicate members,
//! arity) hold.

use std::collections::BTreeMap;
use std::sync::Arc;

use namidb_core::id::{NamespaceId, NodeId};
use namidb_core::value::Value;
use namidb_storage::{NamespacePaths, NodeWriteRecord, WriterSession};
use object_store::memory::InMemory;
use object_store::ObjectStore;

fn store() -> Arc<dyn ObjectStore> {
    Arc::new(InMemory::new())
}

fn paths(name: &str) -> NamespacePaths {
    NamespacePaths::new("tenants", NamespaceId::new(name).unwrap())
}

#[tokio::test]
async fn composite_index_ddl_persists_and_guards() {
    let shared = store();
    let mut w = WriterSession::open(shared.clone(), paths("cix"))
        .await
        .unwrap();
    // A live row so member-type inference has something to look at.
    let mut props: BTreeMap<String, Value> = BTreeMap::new();
    props.insert("city".into(), Value::Str("quito".into()));
    props.insert("age".into(), Value::I64(30));
    w.upsert_node(
        "Person",
        NodeId::new(),
        &NodeWriteRecord {
            properties: props,
            schema_version: 1,
            ..Default::default()
        },
    )
    .unwrap();
    w.commit_batch().await.unwrap();

    // Create with an explicit name; declaration order must persist.
    let v1 = w
        .create_composite_index_named(
            Some("pair"),
            "Person",
            &["city".into(), "age".into()],
            false,
        )
        .await
        .unwrap();
    let snap = w.snapshot();
    let schema = &snap.manifest().manifest.schema;
    assert_eq!(schema.indexes.len(), 1);
    let index = &schema.indexes[0];
    assert_eq!(index.name, "pair");
    assert_eq!(index.properties, ["city".to_string(), "age".to_string()]);
    // Member types were declared from live values, flags untouched.
    let label = schema.label("Person").unwrap();
    let age = label.properties.iter().find(|p| p.name == "age").unwrap();
    assert!(!age.indexed && !age.unique, "member flags must stay clear");

    // IF NOT EXISTS matches the SET (either order) and no-ops.
    let v2 = w
        .create_composite_index_named(None, "Person", &["age".into(), "city".into()], true)
        .await
        .unwrap();
    assert_eq!(v1, v2, "IF NOT EXISTS over the same set must be a no-op");
    // Without IF NOT EXISTS it errors.
    let err = w
        .create_composite_index_named(None, "Person", &["age".into(), "city".into()], false)
        .await
        .unwrap_err();
    assert!(err.to_string().contains("already exists"), "{err}");
    // Name collision with a different definition errors.
    let err = w
        .create_composite_index_named(
            Some("pair"),
            "Person",
            &["city".into(), "name".into()],
            false,
        )
        .await
        .unwrap_err();
    assert!(err.to_string().contains("named `pair`"), "{err}");
    // Arity and duplicate-member guards.
    let err = w
        .create_composite_index_named(None, "Person", &["city".into()], false)
        .await
        .unwrap_err();
    assert!(err.to_string().contains("at least two"), "{err}");
    let err = w
        .create_composite_index_named(None, "Person", &["city".into(), "city".into()], false)
        .await
        .unwrap_err();
    assert!(err.to_string().contains("duplicate property"), "{err}");

    // The definition survives a fresh session over the same store.
    drop(w);
    let w2 = WriterSession::open(shared, paths("cix")).await.unwrap();
    let snap2 = w2.snapshot();
    let schema = &snap2.manifest().manifest.schema;
    assert_eq!(schema.indexes.len(), 1);
    assert_eq!(schema.indexes[0].name, "pair");
    assert_eq!(
        schema.indexes[0].properties,
        ["city".to_string(), "age".to_string()],
        "declaration order must survive the manifest round-trip"
    );
}
