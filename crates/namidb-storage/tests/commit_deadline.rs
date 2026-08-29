//! Item 59: the durability tail respects the write deadline at its
//! DETERMINATE boundaries. A deadline that expired during staging fails
//! `commit_batch` before any object write — pending batch preserved,
//! session healthy, nothing published — and the same batch commits cleanly
//! once the pressure clears. The tail is never aborted between the WAL PUT
//! and the pointer CAS (that zone runs to a definite outcome).

use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use namidb_core::id::{NamespaceId, NodeId};
use namidb_core::value::Value;
use namidb_storage::{cancel, Error, NamespacePaths, NodeWriteRecord, WriterSession};
use object_store::memory::InMemory;
use object_store::ObjectStore;

fn row(name: &str) -> NodeWriteRecord {
    let mut props: BTreeMap<String, Value> = BTreeMap::new();
    props.insert("name".into(), Value::Str(name.into()));
    NodeWriteRecord {
        properties: props,
        schema_version: 1,
        ..Default::default()
    }
}

#[tokio::test]
async fn expired_deadline_fails_commit_before_any_object_write() {
    let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
    let paths = NamespacePaths::new("tenants", NamespaceId::new("cdl").unwrap());
    let mut w = WriterSession::open(store, paths).await.unwrap();

    // A first ordinary commit to establish a baseline manifest version.
    w.upsert_node("T", NodeId::new(), &row("first")).unwrap();
    w.commit_batch().await.unwrap();
    let version_before = w.snapshot().manifest().manifest.version;

    // Stage a second write; its deadline "expired during staging".
    w.upsert_node("T", NodeId::new(), &row("second")).unwrap();
    let expired = Instant::now() - Duration::from_millis(1);
    let err = cancel::with_deadline(Some(expired), w.commit_batch())
        .await
        .expect_err("an expired deadline must fail the commit at the pre-PUT probe");
    assert!(matches!(err, Error::Timeout), "{err:?}");

    // Nothing moved: no manifest advance, the second row is not readable,
    // and the session is not poisoned.
    let snap = w.snapshot();
    assert_eq!(snap.manifest().manifest.version, version_before);
    assert_eq!(
        snap.scan_label("T").await.unwrap().len(),
        1,
        "the aborted batch must not be readable"
    );
    drop(snap);

    // The pending batch survived: the SAME staged rows commit once the
    // deadline pressure is gone.
    w.commit_batch().await.unwrap();
    let snap = w.snapshot();
    assert!(snap.manifest().manifest.version > version_before);
    assert_eq!(snap.scan_label("T").await.unwrap().len(), 2);
}

#[tokio::test]
async fn operator_cancel_also_fails_commit_at_the_boundary() {
    let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
    let paths = NamespacePaths::new("tenants", NamespaceId::new("ccl").unwrap());
    let mut w = WriterSession::open(store, paths).await.unwrap();
    w.upsert_node("T", NodeId::new(), &row("victim")).unwrap();

    let flag = Arc::new(std::sync::atomic::AtomicBool::new(true));
    let err = cancel::with_cancel_flag(flag, w.commit_batch())
        .await
        .expect_err("a flipped cancel flag must fail the commit at the boundary");
    assert!(matches!(err, Error::Cancelled), "{err:?}");

    // Same preservation contract as the deadline.
    w.commit_batch().await.unwrap();
    assert_eq!(w.snapshot().scan_label("T").await.unwrap().len(), 1);
}
