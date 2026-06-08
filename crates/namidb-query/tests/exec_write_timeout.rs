//! Write-query wall-clock timeout.
//!
//! A write that overruns its deadline is aborted cooperatively with
//! [`ExecError::Timeout`] and its pending batch is discarded, so nothing
//! partial is committed and the shared writer is left clean for the next
//! statement. The write-side complement of the read-query timeout exercised
//! through `execute_with_limits`.

use std::sync::Arc;
use std::time::{Duration, Instant};

use namidb_core::id::NamespaceId;
use namidb_storage::{NamespacePaths, WriterSession};
use object_store::memory::InMemory;
use object_store::ObjectStore;

use namidb_query::{execute_write, execute_write_with_deadline, lower, parse, ExecError, Params};

fn store() -> Arc<dyn ObjectStore> {
    Arc::new(InMemory::new())
}

fn paths(name: &str) -> NamespacePaths {
    NamespacePaths::new("tenants", NamespaceId::new(name).unwrap())
}

/// A deadline already in the past, so the first cooperative probe fires.
fn expired() -> Option<Instant> {
    Some(Instant::now() - Duration::from_secs(1))
}

async fn write(writer: &mut WriterSession, text: &str) {
    let plan = lower(&parse(text).unwrap()).unwrap();
    execute_write(&plan, writer, &Params::new()).await.unwrap();
}

async fn person_count(writer: &WriterSession) -> usize {
    writer.snapshot().scan_label("Person").await.unwrap().len()
}

#[tokio::test]
async fn create_past_deadline_times_out_and_commits_nothing() {
    let mut writer = WriterSession::open(store(), paths("wt-create"))
        .await
        .unwrap();

    let plan = lower(&parse("CREATE (a:Person {name: 'Ada'}) RETURN a").unwrap()).unwrap();
    let err = execute_write_with_deadline(&plan, &mut writer, &Params::new(), expired())
        .await
        .unwrap_err();
    assert!(
        matches!(err, ExecError::Timeout),
        "expected timeout, got {err:?}"
    );

    // The node never reached a snapshot: the aborted batch was discarded.
    assert_eq!(person_count(&writer).await, 0);

    // The writer is clean — a later unbounded write succeeds and is the only
    // thing committed, proving the discarded batch left no residue behind.
    write(&mut writer, "CREATE (a:Person {name: 'Grace'}) RETURN a").await;
    assert_eq!(person_count(&writer).await, 1);
}

#[tokio::test]
async fn create_within_deadline_succeeds() {
    let mut writer = WriterSession::open(store(), paths("wt-ok")).await.unwrap();

    let plan = lower(&parse("CREATE (a:Person {name: 'Ada'}) RETURN a").unwrap()).unwrap();
    let deadline = Some(Instant::now() + Duration::from_secs(30));
    let outcome = execute_write_with_deadline(&plan, &mut writer, &Params::new(), deadline)
        .await
        .unwrap();

    assert_eq!(outcome.nodes_created, 1);
    assert_eq!(person_count(&writer).await, 1);
}

#[tokio::test]
async fn detach_delete_past_deadline_preserves_data() {
    let mut writer = WriterSession::open(store(), paths("wt-delete"))
        .await
        .unwrap();

    // Seed two connected nodes with an unbounded, committed write.
    write(
        &mut writer,
        "CREATE (a:Person {name: 'A'})-[:KNOWS]->(b:Person {name: 'B'})",
    )
    .await;
    assert_eq!(person_count(&writer).await, 2);

    // A delete that overruns its deadline must remove nothing.
    let del = lower(&parse("MATCH (n:Person) DETACH DELETE n").unwrap()).unwrap();
    let err = execute_write_with_deadline(&del, &mut writer, &Params::new(), expired())
        .await
        .unwrap_err();
    assert!(
        matches!(err, ExecError::Timeout),
        "expected timeout, got {err:?}"
    );

    // Both nodes survive the aborted delete.
    assert_eq!(person_count(&writer).await, 2);
}

#[tokio::test]
async fn no_deadline_leaves_writes_unbounded() {
    let mut writer = WriterSession::open(store(), paths("wt-none"))
        .await
        .unwrap();

    // `None` is the embedded / CLI / test path: no guard, baseline behaviour.
    let plan = lower(&parse("CREATE (a:Person {name: 'Ada'}) RETURN a").unwrap()).unwrap();
    let outcome = execute_write_with_deadline(&plan, &mut writer, &Params::new(), None)
        .await
        .unwrap();

    assert_eq!(outcome.nodes_created, 1);
    assert_eq!(person_count(&writer).await, 1);
}
