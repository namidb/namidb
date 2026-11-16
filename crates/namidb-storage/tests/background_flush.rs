//! The off-lock flush: `begin_background_flush` → `BackgroundFlush::build`
//! → `install_background_flush`.
//!
//! Writes keep committing between the checkpoint and the install. These tests
//! pin that none of them is lost or shadowed by the flushed version, that the
//! install retires only the WAL segments the checkpoint covered (so a reopen
//! replays the rest), and that an install whose base changed meanwhile is
//! refused without losing anything.

use std::collections::BTreeMap;
use std::sync::Arc;

use namidb_core::id::{NamespaceId, NodeId};
use namidb_core::schema::Schema;
use namidb_core::value::Value as CoreValue;
use namidb_storage::manifest::SstKind;
use namidb_storage::{BackgroundFlushInstall, NamespacePaths, NodeWriteRecord, WriterSession};
use object_store::memory::InMemory;
use object_store::ObjectStore;

fn doc(value: &str) -> NodeWriteRecord {
    let mut properties = BTreeMap::new();
    properties.insert("v".to_string(), CoreValue::Str(value.to_string()));
    NodeWriteRecord {
        properties,
        schema_version: 1,
        ..Default::default()
    }
}

fn paths(name: &str) -> NamespacePaths {
    NamespacePaths::new("tenants", NamespaceId::new(name).unwrap())
}

async fn value_of(writer: &WriterSession, id: NodeId) -> Option<String> {
    let node = writer.snapshot().lookup_node_by_id(id).await.unwrap()?;
    match node.properties.get("v") {
        Some(CoreValue::Str(value)) => Some(value.clone()),
        other => panic!("unexpected property {other:?}"),
    }
}

fn node_ssts(writer: &WriterSession) -> usize {
    writer
        .snapshot()
        .manifest()
        .manifest
        .ssts
        .iter()
        .filter(|sst| sst.kind == SstKind::Nodes)
        .count()
}

/// The state both the installed writer and a reopened one must show.
async fn check(writer: &WriterSession, flushed: &[NodeId], added: &[NodeId]) {
    assert_eq!(value_of(writer, flushed[0]).await.as_deref(), Some("new-0"));
    assert_eq!(
        value_of(writer, flushed[1]).await,
        None,
        "deleted after the checkpoint"
    );
    assert_eq!(value_of(writer, flushed[2]).await.as_deref(), Some("new-2"));
    assert_eq!(value_of(writer, flushed[3]).await.as_deref(), Some("old-3"));
    for (i, id) in added.iter().enumerate() {
        assert_eq!(
            value_of(writer, *id).await,
            Some(format!("added-{i}")),
            "a row added while the flush was built"
        );
    }
    assert_eq!(writer.snapshot().scan_label("Doc").await.unwrap().len(), 24);
}

#[tokio::test]
async fn writes_made_while_a_flush_builds_survive_its_install_and_a_reopen() {
    let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
    let ns = paths("bg-flush");
    let mut writer = WriterSession::open(store.clone(), ns.clone())
        .await
        .unwrap();
    let flushed: Vec<NodeId> = (0..20).map(|_| NodeId::new()).collect();
    for (i, id) in flushed.iter().enumerate() {
        writer
            .upsert_node("Doc", *id, &doc(&format!("old-{i}")))
            .unwrap();
    }
    writer.commit_batch().await.unwrap();

    let ticket = writer
        .begin_background_flush()
        .await
        .unwrap()
        .expect("a non-empty memtable has something to flush");
    assert_eq!(ticket.rows(), 20);

    // While the flush builds: overwrite two flushed keys, delete one, and add
    // new rows, in two separate commits (two WAL segments).
    writer
        .upsert_node("Doc", flushed[0], &doc("new-0"))
        .unwrap();
    writer.tombstone_node("Doc", flushed[1]).unwrap();
    writer.commit_batch().await.unwrap();
    let added: Vec<NodeId> = (0..5).map(|_| NodeId::new()).collect();
    for (i, id) in added.iter().enumerate() {
        writer
            .upsert_node("Doc", *id, &doc(&format!("added-{i}")))
            .unwrap();
    }
    writer
        .upsert_node("Doc", flushed[2], &doc("new-2"))
        .unwrap();
    writer.commit_batch().await.unwrap();

    let built = ticket.build().await.unwrap();
    let outcome = writer.install_background_flush(built).await.unwrap();
    let BackgroundFlushInstall::Installed(outcome) = outcome else {
        panic!("only WAL commits happened since the checkpoint; the install must apply");
    };
    assert!(outcome.ssts_written > 0);
    assert_eq!(node_ssts(&writer), 1);

    check(&writer, &flushed, &added).await;
    // The memtable keeps only what the flush did not cover.
    assert!(writer.memtable_bytes() > 0);

    // The newer WAL segments were not retired: a reopen replays them.
    let reopened = WriterSession::open(store.clone(), ns.clone())
        .await
        .unwrap();
    drop(writer);
    check(&reopened, &flushed, &added).await;
}

#[tokio::test]
async fn an_install_whose_base_changed_is_refused_and_loses_nothing() {
    let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
    let ns = paths("bg-flush-superseded");
    let mut writer = WriterSession::open(store.clone(), ns.clone())
        .await
        .unwrap();
    let first = NodeId::new();
    writer.upsert_node("Doc", first, &doc("first")).unwrap();
    writer.commit_batch().await.unwrap();
    let ticket = writer.begin_background_flush().await.unwrap().unwrap();

    // A synchronous flush commits a manifest that adds SSTs meanwhile.
    let second = NodeId::new();
    writer.upsert_node("Doc", second, &doc("second")).unwrap();
    writer.flush(Schema::empty()).await.unwrap();
    let ssts_before = node_ssts(&writer);

    let built = ticket.build().await.unwrap();
    let outcome = writer.install_background_flush(built).await.unwrap();
    assert!(matches!(outcome, BackgroundFlushInstall::Superseded));
    assert_eq!(node_ssts(&writer), ssts_before, "nothing was installed");
    assert_eq!(value_of(&writer, first).await.as_deref(), Some("first"));
    assert_eq!(value_of(&writer, second).await.as_deref(), Some("second"));

    drop(writer);
    let reopened = WriterSession::open(store, ns).await.unwrap();
    assert_eq!(value_of(&reopened, first).await.as_deref(), Some("first"));
    assert_eq!(value_of(&reopened, second).await.as_deref(), Some("second"));
}
