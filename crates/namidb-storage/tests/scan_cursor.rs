//! `label_scan_cursor` and `edge_scan_cursor` return what `scan_label` and
//! `scan_edge_type` return, a batch at a time, and a node batch reads about
//! its own rows instead of the whole label.

use std::collections::BTreeMap;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;

use object_store::memory::InMemory;
use object_store::ObjectStore;

use namidb_core::{NamespaceId, NodeId, Schema, Value};
use namidb_storage::read::Snapshot;
use namidb_storage::{
    EdgeView, EdgeWriteRecord, NamespacePaths, NodeView, NodeWriteRecord, SstCache, WriterSession,
};

const ROWS: usize = 3000;

fn node(v: i64) -> NodeWriteRecord {
    let mut properties = BTreeMap::new();
    properties.insert("v".to_string(), Value::I64(v));
    NodeWriteRecord {
        properties,
        schema_version: 0,
        ..Default::default()
    }
}

fn edge(w: i64) -> EdgeWriteRecord {
    let mut properties = BTreeMap::new();
    properties.insert("w".to_string(), Value::I64(w));
    EdgeWriteRecord {
        properties,
        schema_version: 0,
    }
}

/// Two flushed node SSTs over the same random ids (so their key ranges
/// overlap) and a memtable on top, with rewrites, deletes and relabels at
/// every level. Edges get the same treatment.
async fn overlapping_writer() -> (WriterSession, Vec<NodeId>) {
    // Small row groups, so a batch can be told apart from the whole SST.
    std::env::set_var("NAMIDB_NODE_SST_ROW_GROUP_ROWS", "64");
    let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
    let paths = NamespacePaths::new("tenants", NamespaceId::new("cursor").unwrap());
    let mut writer = WriterSession::open(store, paths).await.unwrap();
    let ids: Vec<NodeId> = (0..ROWS).map(|_| NodeId::new()).collect();
    for (i, id) in ids.iter().enumerate() {
        let label = if i % 4 == 0 { "B" } else { "A" };
        writer.upsert_node(label, *id, &node(i as i64)).unwrap();
        writer
            .upsert_edge("TO", *id, ids[(i + 1) % ROWS], &edge(i as i64))
            .unwrap();
    }
    writer.commit_batch().await.unwrap();
    writer.flush(Schema::empty()).await.unwrap();
    for (i, id) in ids.iter().enumerate() {
        if i % 7 == 0 {
            writer.tombstone_node("A", *id).unwrap();
            writer
                .tombstone_edge("TO", *id, ids[(i + 1) % ROWS])
                .unwrap();
        } else if i % 5 == 0 {
            writer
                .upsert_node_with_labels(["B".to_string()], *id, &node(10_000 + i as i64))
                .unwrap();
        } else if i % 3 == 0 {
            writer
                .upsert_node("A", *id, &node(20_000 + i as i64))
                .unwrap();
            writer
                .upsert_edge("TO", *id, ids[(i + 1) % ROWS], &edge(20_000 + i as i64))
                .unwrap();
        }
    }
    writer.commit_batch().await.unwrap();
    writer.flush(Schema::empty()).await.unwrap();
    for (i, id) in ids.iter().enumerate() {
        if i % 13 == 0 {
            writer.tombstone_node("A", *id).unwrap();
        } else if i % 11 == 0 {
            writer
                .upsert_node("A", *id, &node(30_000 + i as i64))
                .unwrap();
            writer
                .upsert_edge("TO", *id, ids[(i + 2) % ROWS], &edge(30_000 + i as i64))
                .unwrap();
        }
    }
    writer.commit_batch().await.unwrap();
    assert!(
        !writer.snapshot().node_scan_is_disjoint(),
        "the layout must overlap or the test proves nothing"
    );
    (writer, ids)
}

async fn drain_nodes(snap: &Snapshot<'_>, label: &str, batch: usize) -> Vec<NodeView> {
    let mut cursor = snap.label_scan_cursor(label).await.unwrap();
    let mut out = Vec::new();
    loop {
        let rows = cursor.next_batch(snap, batch).await.unwrap();
        assert!(rows.len() <= batch);
        if rows.is_empty() {
            return out;
        }
        out.extend(rows);
    }
}

async fn drain_edges(snap: &Snapshot<'_>, edge_type: &str, batch: usize) -> Vec<EdgeView> {
    let mut cursor = snap.edge_scan_cursor(edge_type).await.unwrap();
    let mut out = Vec::new();
    loop {
        let rows = cursor.next_batch(batch).await.unwrap();
        assert!(rows.len() <= batch);
        if rows.is_empty() {
            return out;
        }
        out.extend(rows);
    }
}

#[tokio::test]
async fn node_cursor_matches_scan_label_on_overlapping_ssts() {
    let (writer, _) = overlapping_writer().await;
    let snap = writer.snapshot();
    for label in ["A", "B", "Missing"] {
        let full = snap.scan_label(label).await.unwrap();
        for batch in [1, 7, 64, 1000, 10_000] {
            assert_eq!(
                drain_nodes(&snap, label, batch).await,
                full,
                "{label} by {batch}"
            );
        }
    }
}

#[tokio::test]
async fn node_cursor_first_batch_reads_about_one_batch() {
    let (writer, _) = overlapping_writer().await;
    let cache = SstCache::new(64 * 1024 * 1024);
    let snap = writer.snapshot().with_cache(cache.clone());
    let total = snap.scan_label("A").await.unwrap().len();
    assert!(total > 1500, "{total}");

    let cache = SstCache::new(64 * 1024 * 1024);
    let snap = writer.snapshot().with_cache(cache.clone());
    let mut cursor = snap.label_scan_cursor("A").await.unwrap();
    let first = cursor.next_batch(&snap, 50).await.unwrap();
    assert_eq!(first.len(), 50);
    // 50 rows of a ~2,000-row label spread over 2 x 3,000 SST rows touch a
    // couple of 64-row groups per SST, not the 94 groups of both SSTs.
    assert!(
        cache.limited_node_scan_row_groups() <= 8,
        "row groups: {}",
        cache.limited_node_scan_row_groups()
    );
    assert!(
        cache.limited_node_scan_decoded_rows() <= 8 * 64,
        "structural rows: {}",
        cache.limited_node_scan_decoded_rows()
    );
    // The full label scan was not run behind the cursor's back.
    assert_eq!(cache.label_selected_node_rows(), 0);
}

#[tokio::test]
async fn node_cursor_keeps_its_snapshot_while_writes_land() {
    let (mut writer, ids) = overlapping_writer().await;
    let owned = writer.owned_snapshot();
    let expected = owned.borrow().scan_label("A").await.unwrap();

    let snap = owned.borrow();
    let mut cursor = snap.label_scan_cursor("A").await.unwrap();
    let mut got = cursor.next_batch(&snap, 100).await.unwrap();

    // Delete half the label, add new members and flush, while the cursor
    // is open on the older snapshot.
    for id in ids.iter().step_by(2) {
        writer.tombstone_node("A", *id).unwrap();
    }
    for i in 0..500 {
        writer
            .upsert_node("A", NodeId::new(), &node(50_000 + i))
            .unwrap();
    }
    writer.commit_batch().await.unwrap();
    writer.flush(Schema::empty()).await.unwrap();

    loop {
        let rows = cursor.next_batch(&snap, 100).await.unwrap();
        if rows.is_empty() {
            break;
        }
        got.extend(rows);
    }
    assert_eq!(got, expected);
    assert_ne!(writer.snapshot().scan_label("A").await.unwrap(), expected);
}

#[tokio::test]
async fn edge_cursor_matches_scan_edge_type_on_overlapping_ssts() {
    let (writer, _) = overlapping_writer().await;
    let snap = writer.snapshot();
    for edge_type in ["TO", "Missing"] {
        let full = snap.scan_edge_type(edge_type).await.unwrap();
        for batch in [1, 7, 64, 10_000] {
            assert_eq!(
                drain_edges(&snap, edge_type, batch).await,
                full,
                "{edge_type} by {batch}"
            );
        }
    }
}

#[tokio::test]
async fn sparse_label_cursor_skips_ssts_without_it() {
    std::env::set_var("NAMIDB_NODE_SST_ROW_GROUP_ROWS", "64");
    let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
    let paths = NamespacePaths::new("tenants", NamespaceId::new("sparse").unwrap());
    let mut writer = WriterSession::open(store, paths).await.unwrap();
    let ids: Vec<NodeId> = (0..ROWS).map(|_| NodeId::new()).collect();
    for (i, id) in ids.iter().enumerate() {
        writer.upsert_node("A", *id, &node(i as i64)).unwrap();
    }
    writer.commit_batch().await.unwrap();
    writer.flush(Schema::empty()).await.unwrap();
    // A few reports written after the big load, plus rewrites of some old
    // nodes so the second SST overlaps the first.
    for i in 0..10 {
        writer.upsert_node("R", NodeId::new(), &node(i)).unwrap();
    }
    for id in ids.iter().step_by(100) {
        writer.upsert_node("A", *id, &node(-1)).unwrap();
    }
    writer.commit_batch().await.unwrap();
    writer.flush(Schema::empty()).await.unwrap();
    assert!(!writer.snapshot().node_scan_is_disjoint());

    let cache = SstCache::new(64 * 1024 * 1024);
    let snap = writer.snapshot().with_cache(cache.clone());
    let mut cursor = snap.label_scan_cursor("R").await.unwrap();
    let rows = cursor.next_batch(&snap, 100).await.unwrap();
    assert_eq!(rows, snap.scan_label("R").await.unwrap());
    assert_eq!(rows.len(), 10);
    // Only the 40-row second SST is walked, not the 47 groups of the first.
    assert!(
        cache.limited_node_scan_row_groups() <= 1,
        "row groups: {}",
        cache.limited_node_scan_row_groups()
    );
}

#[tokio::test]
async fn node_cursor_stops_after_a_failed_batch() {
    let (writer, _) = overlapping_writer().await;
    let snap = writer.snapshot();
    let mut cursor = snap.label_scan_cursor("A").await.unwrap();
    assert_eq!(cursor.next_batch(&snap, 10).await.unwrap().len(), 10);

    // A timed out batch, say: it gives up after the merge consumed ids.
    let cancelled = Arc::new(AtomicBool::new(true));
    let failed =
        namidb_storage::cancel::with_cancel_flag(cancelled, cursor.next_batch(&snap, 1000)).await;
    assert!(failed.is_err());
    // Going on would resume after the rows the failed batch lost.
    assert!(cursor.next_batch(&snap, 1000).await.is_err());
}

#[tokio::test]
async fn edge_cursor_stops_after_a_failed_batch() {
    let (writer, _) = overlapping_writer().await;
    let snap = writer.snapshot();
    let mut cursor = snap.edge_scan_cursor("TO").await.unwrap();
    assert_eq!(cursor.next_batch(10).await.unwrap().len(), 10);

    let cancelled = Arc::new(AtomicBool::new(true));
    let failed = namidb_storage::cancel::with_cancel_flag(cancelled, cursor.next_batch(10)).await;
    assert!(failed.is_err());
    assert!(cursor.next_batch(10).await.is_err());
}

#[tokio::test]
async fn full_node_walk_reads_each_record_about_once() {
    let (writer, _) = overlapping_writer().await;
    let cache = SstCache::new(64 * 1024 * 1024);
    let snap = writer.snapshot().with_cache(cache.clone());
    let walked = drain_nodes(&snap, "A", 1024).await;
    assert_eq!(walked, snap.scan_label("A").await.unwrap());
    // The point reads of all batches together examine about one node
    // locator entry per SST row, like a single lookup of every member,
    // not one probe of every SST per row.
    let sst_rows = cache.limited_node_scan_decoded_rows();
    assert!(sst_rows > 4000, "{sst_rows}");
    assert!(
        cache.node_locator_entries_examined() <= sst_rows * 5 / 4,
        "entries {} for {sst_rows} SST rows",
        cache.node_locator_entries_examined()
    );
    assert_eq!(cache.decoded_node_row_group_misses(), 0);
}
