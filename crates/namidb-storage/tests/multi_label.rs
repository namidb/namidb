//! Integration tests for multi-label nodes (id-primary storage).
//!
//! A node carries a *set* of labels on-row; it is keyed by id alone, so the
//! same node surfaces under every one of its labels exactly once, and a
//! tombstone (by id) removes it from all of them. These exercise the full
//! write -> flush -> reopen -> scan path against an in-memory object store.

use std::collections::BTreeMap;
use std::sync::Arc;

use object_store::memory::InMemory;
use object_store::ObjectStore;
use uuid::Uuid;

use namidb_core::{NamespaceId, NodeId, Schema, Value};
use namidb_storage::{NamespacePaths, NodeWriteRecord, WriterSession};

fn store() -> Arc<dyn ObjectStore> {
    Arc::new(InMemory::new())
}

fn paths(ns: &str) -> NamespacePaths {
    NamespacePaths::new("tenants", NamespaceId::new(ns).unwrap())
}

fn nid(b: u8) -> NodeId {
    let mut bytes = [0u8; 16];
    bytes[15] = b;
    NodeId::from_uuid(Uuid::from_bytes(bytes))
}

fn rec(name: &str) -> NodeWriteRecord {
    let mut props = BTreeMap::new();
    props.insert("name".to_string(), Value::Str(name.into()));
    NodeWriteRecord {
        properties: props,
        schema_version: 0,
        ..Default::default()
    }
}

fn labels(set: &[&str]) -> Vec<String> {
    set.iter().map(|s| s.to_string()).collect()
}

/// Write `(:A:B)`, flush, reopen: the node surfaces under BOTH labels exactly
/// once, carries the full label set, and is invisible under a label it lacks.
#[tokio::test]
async fn two_labels_scan_each_once_after_flush() {
    let mut s = WriterSession::open(store(), paths("ml-two")).await.unwrap();
    let id = nid(1);
    s.upsert_node_with_labels(labels(&["A", "B"]), id, &rec("x"))
        .unwrap();
    s.commit_batch().await.unwrap();
    s.flush(Schema::empty()).await.unwrap();

    let snap = s.snapshot();
    let a = snap.scan_label("A").await.unwrap();
    let b = snap.scan_label("B").await.unwrap();
    assert_eq!(a.len(), 1, "exactly one node under A");
    assert_eq!(b.len(), 1, "exactly one node under B (no per-label dup)");
    assert_eq!(a[0].id, id);
    assert_eq!(b[0].id, id);
    // The on-row label set round-trips through the SST.
    assert!(a[0].labels.contains("A") && a[0].labels.contains("B"));
    assert_eq!(a[0].labels.len(), 2);

    // Point lookups are label-scoped: present under A, absent under C.
    assert!(snap.lookup_node("A", id).await.unwrap().is_some());
    assert!(snap.lookup_node("B", id).await.unwrap().is_some());
    assert!(snap.lookup_node("C", id).await.unwrap().is_none());
}

/// Same as above but BEFORE a flush: the memtable path must also surface a
/// multi-label node under each label.
#[tokio::test]
async fn two_labels_scan_each_once_from_memtable() {
    let mut s = WriterSession::open(store(), paths("ml-mem")).await.unwrap();
    let id = nid(1);
    s.upsert_node_with_labels(labels(&["A", "B"]), id, &rec("x"))
        .unwrap();
    s.commit_batch().await.unwrap();
    // No flush: rows live in the memtable.

    let snap = s.snapshot();
    assert_eq!(snap.scan_label("A").await.unwrap().len(), 1);
    assert_eq!(snap.scan_label("B").await.unwrap().len(), 1);
    assert_eq!(snap.lookup_node("B", id).await.unwrap().unwrap().id, id);
}

/// A tombstone is keyed by id: it removes the node from EVERY label scan.
#[tokio::test]
async fn tombstone_clears_all_label_scans() {
    let mut s = WriterSession::open(store(), paths("ml-tomb"))
        .await
        .unwrap();
    let id = nid(1);
    s.upsert_node_with_labels(labels(&["A", "B"]), id, &rec("x"))
        .unwrap();
    s.commit_batch().await.unwrap();
    s.flush(Schema::empty()).await.unwrap();

    s.tombstone_node("A", id).unwrap();
    s.commit_batch().await.unwrap();
    s.flush(Schema::empty()).await.unwrap();

    let snap = s.snapshot();
    assert!(snap.scan_label("A").await.unwrap().is_empty());
    assert!(snap.scan_label("B").await.unwrap().is_empty());
    assert!(snap.lookup_node("A", id).await.unwrap().is_none());
}

/// Re-upserting with a smaller label set (drop B) is last-LSN-wins: the node
/// leaves B's scan and stays in A's, with the updated label set.
#[tokio::test]
async fn relabel_via_reupsert_is_last_write_wins() {
    let mut s = WriterSession::open(store(), paths("ml-relabel"))
        .await
        .unwrap();
    let id = nid(1);
    s.upsert_node_with_labels(labels(&["A", "B"]), id, &rec("x"))
        .unwrap();
    s.commit_batch().await.unwrap();
    s.flush(Schema::empty()).await.unwrap();

    // Rewrite the node as :A only (B removed).
    s.upsert_node_with_labels(labels(&["A"]), id, &rec("x"))
        .unwrap();
    s.commit_batch().await.unwrap();
    s.flush(Schema::empty()).await.unwrap();

    let snap = s.snapshot();
    let a = snap.scan_label("A").await.unwrap();
    assert_eq!(a.len(), 1);
    assert_eq!(a[0].labels.len(), 1);
    assert!(a[0].labels.contains("A"));
    // The stale B membership from the older SST must NOT resurface.
    assert!(
        snap.scan_label("B").await.unwrap().is_empty(),
        "removed label must not resurface from the older SST"
    );
}

/// Reopen a fresh `WriterSession` on the same store after flush: the label set
/// survives via the manifest's dictionary + the on-row `__labels` column, with
/// no live memtable.
#[tokio::test]
async fn labels_survive_a_reopen() {
    let store = store();
    let p = paths("ml-reopen");
    let id = nid(7);
    {
        let mut s = WriterSession::open(store.clone(), p.clone()).await.unwrap();
        s.upsert_node_with_labels(labels(&["A", "B"]), id, &rec("x"))
            .unwrap();
        s.commit_batch().await.unwrap();
        s.flush(Schema::empty()).await.unwrap();
    }
    // Fresh session, fresh in-process state — must read labels off disk.
    let s = WriterSession::open(store, p).await.unwrap();
    let snap = s.snapshot();
    let a = snap.scan_label("A").await.unwrap();
    assert_eq!(a.len(), 1);
    assert!(a[0].labels.contains("A") && a[0].labels.contains("B"));
}
