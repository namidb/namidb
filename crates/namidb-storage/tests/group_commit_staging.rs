//! RFC-034 group commit, storage layer: request scopes over the staged
//! batch. A FAILED statement must roll back ALONE — earlier requests'
//! staged rows, their RYOW overlay entries, and their unique-index tuples
//! all survive for the group's single commit — while a later full
//! `discard_batch` still restores the pre-batch committed state.

use std::collections::BTreeMap;
use std::sync::Arc;

use namidb_core::id::{NamespaceId, NodeId};
use namidb_core::value::Value;
use namidb_storage::{NamespacePaths, NodeWriteRecord, UniqueProbe, WriterSession};
use object_store::memory::InMemory;
use object_store::ObjectStore;

fn store() -> Arc<dyn ObjectStore> {
    Arc::new(InMemory::new())
}

fn paths(name: &str) -> NamespacePaths {
    NamespacePaths::new("tenants", NamespaceId::new(name).unwrap())
}

fn person(email: &str) -> NodeWriteRecord {
    let mut props: BTreeMap<String, Value> = BTreeMap::new();
    props.insert("email".into(), Value::Str(email.into()));
    NodeWriteRecord {
        properties: props,
        schema_version: 1,
        ..Default::default()
    }
}

#[tokio::test]
async fn failed_request_rolls_back_alone() {
    let mut w = WriterSession::open(store(), paths("gc-alone"))
        .await
        .unwrap();
    let a = NodeId::new();
    let b = NodeId::new();

    // Request 1 stages A and closes successfully.
    let _m1 = w.begin_staged_request();
    w.upsert_node("Person", a, &person("a@x")).unwrap();
    w.commit_staged_request();
    assert_eq!(w.pending_len(), 1);

    // Request 2 stages B and REWRITES A, then fails: both of its
    // mutations must vanish, request 1's A must survive with its value.
    let m2 = w.begin_staged_request();
    w.upsert_node("Person", b, &person("b@x")).unwrap();
    w.upsert_node("Person", a, &person("a@changed")).unwrap();
    assert_eq!(w.pending_len(), 3);
    w.rollback_staged_request(m2);

    assert_eq!(w.pending_len(), 1, "only request 1's row remains staged");
    assert_eq!(w.staged_memtable_len(), 1, "overlay rebuilt to request 1");

    // The group's single commit makes exactly request 1 durable.
    w.commit_batch().await.unwrap();
    let snap = w.snapshot();
    let node = snap
        .lookup_node("Person", a)
        .await
        .unwrap()
        .expect("request 1's node must be durable");
    assert_eq!(
        node.properties.get("email"),
        Some(&Value::Str("a@x".into())),
        "request 2's rewrite of A must not survive its rollback"
    );
    assert!(
        snap.lookup_node("Person", b).await.unwrap().is_none(),
        "request 2's new node must not be durable"
    );
}

/// The subtle case the request-scoped undo layer exists for: request 2
/// touches a tuple request 1 ALREADY MOVED. Rolling request 2 back must
/// restore request 1's staged value — not the committed pre-batch value —
/// and a later full discard must still restore the committed one.
#[tokio::test]
async fn unique_index_restores_at_request_start_values() {
    let mut w = WriterSession::open(store(), paths("gc-unique"))
        .await
        .unwrap();
    let a = NodeId::new();
    w.upsert_node("Person", a, &person("x@x")).unwrap();
    w.commit_batch().await.unwrap();

    // Populate the constraint map from committed state.
    let probe = w
        .unique_probe("Person", &[("email", &Value::Str("x@x".into()))], None)
        .await
        .unwrap();
    assert_eq!(probe, UniqueProbe::Conflict(a));

    // Request 1: move A's email x -> y (staged), close successfully.
    let _m1 = w.begin_staged_request();
    w.upsert_node("Person", a, &person("y@y")).unwrap();
    w.commit_staged_request();

    // Request 2: move it again y -> z, then FAIL.
    let m2 = w.begin_staged_request();
    w.upsert_node("Person", a, &person("z@z")).unwrap();
    let probe = w
        .unique_probe("Person", &[("email", &Value::Str("z@z".into()))], None)
        .await
        .unwrap();
    assert_eq!(
        probe,
        UniqueProbe::Conflict(a),
        "request 2 sees its own move"
    );
    w.rollback_staged_request(m2);

    // At-request-start restore: A holds request 1's staged `y`, NOT the
    // committed `x` (a pre-batch restore here would silently undo request 1).
    let probe = w
        .unique_probe("Person", &[("email", &Value::Str("y@y".into()))], None)
        .await
        .unwrap();
    assert_eq!(
        probe,
        UniqueProbe::Conflict(a),
        "rollback of request 2 must keep request 1's staged tuple"
    );
    let probe = w
        .unique_probe("Person", &[("email", &Value::Str("z@z".into()))], None)
        .await
        .unwrap();
    assert_eq!(probe, UniqueProbe::NoConflict);

    // Full batch discard: back to the committed pre-batch state.
    w.discard_batch();
    let probe = w
        .unique_probe("Person", &[("email", &Value::Str("x@x".into()))], None)
        .await
        .unwrap();
    assert_eq!(
        probe,
        UniqueProbe::Conflict(a),
        "full discard must restore the committed tuple"
    );
    let probe = w
        .unique_probe("Person", &[("email", &Value::Str("y@y".into()))], None)
        .await
        .unwrap();
    assert_eq!(probe, UniqueProbe::NoConflict);
}
