//! Read-query execution guards: the wall-clock timeout (RFC query-timeout
//! follow-up). A deadline already in the past aborts with
//! `ExecError::Timeout`; a generous deadline (or none) runs to completion.

use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use namidb_core::id::{NamespaceId, NodeId};
use namidb_core::value::Value as CoreValue;
use namidb_storage::{NamespacePaths, NodeWriteRecord, WriterSession};
use object_store::memory::InMemory;
use object_store::ObjectStore;

use namidb_query::{execute_with_limits, lower, parse, ExecError, Params};

fn store() -> Arc<dyn ObjectStore> {
    Arc::new(InMemory::new())
}

fn paths(name: &str) -> NamespacePaths {
    NamespacePaths::new("tenants", NamespaceId::new(name).unwrap())
}

async fn seed(name: &str) -> WriterSession {
    let mut writer = WriterSession::open(store(), paths(name)).await.unwrap();
    for n in ["Alice", "Bob", "Carol"] {
        let mut p = BTreeMap::new();
        p.insert("name".into(), CoreValue::Str(n.into()));
        writer
            .upsert_node(
                "Person",
                NodeId::new(),
                &NodeWriteRecord {
                    properties: p,
                    schema_version: 1,
                    ..Default::default()
                },
            )
            .unwrap();
    }
    writer.commit_batch().await.unwrap();
    writer
}

#[tokio::test]
async fn past_deadline_aborts_with_timeout() {
    let writer = seed("limits-timeout").await;
    let snap = writer.snapshot();
    let plan = lower(&parse("MATCH (p:Person) RETURN p").unwrap()).unwrap();

    // A deadline already in the past: the per-operator guard fires on the
    // first check.
    let past = Instant::now() - Duration::from_millis(1);
    let err = execute_with_limits(&plan, &snap, &Params::new(), Some(past))
        .await
        .expect_err("a past deadline must abort the read");
    assert!(matches!(err, ExecError::Timeout), "got {err:?}");
}

#[tokio::test]
async fn generous_deadline_completes() {
    let writer = seed("limits-ok").await;
    let snap = writer.snapshot();
    let plan = lower(&parse("MATCH (p:Person) RETURN p").unwrap()).unwrap();

    let future = Instant::now() + Duration::from_secs(60);
    let rows = execute_with_limits(&plan, &snap, &Params::new(), Some(future))
        .await
        .expect("a generous deadline must not abort");
    assert_eq!(rows.len(), 3);
}

#[tokio::test]
async fn no_deadline_is_unbounded() {
    let writer = seed("limits-none").await;
    let snap = writer.snapshot();
    let plan = lower(&parse("MATCH (p:Person) RETURN p").unwrap()).unwrap();

    let rows = execute_with_limits(&plan, &snap, &Params::new(), None)
        .await
        .expect("no deadline runs unbounded");
    assert_eq!(rows.len(), 3);
}
