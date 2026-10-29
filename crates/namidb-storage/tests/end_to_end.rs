//! End-to-end integration test of the storage protocol against an in-memory
//! object store.
//!
//! We use [`object_store::memory::InMemory`] (not `LocalFileSystem`) because
//! the upstream `LocalFileSystem` does **not** implement `PutMode::Update`
//! as of `object_store 0.13` — every CAS commit would fail with
//! `NotImplemented`. The unit tests inside the crate use `InMemory` too;
//! this file exercises the *combination* of manifest + WAL + memtable in a
//! single scenario, which is what we want from an integration test.
//!
//! The matching MinIO-backed test lives in `minio_integration.rs` and is
//! `#[ignore]`d by default.

use std::sync::Arc;

use bytes::Bytes;
use object_store::memory::InMemory;
use object_store::ObjectStore;
use uuid::Uuid;

use namidb_core::NamespaceId;
use namidb_storage::{
    Epoch, ManifestStore, MemKey, MemOp, Memtable, NamespacePaths, WalRecord, WalSegment,
    WalSegmentDescriptor, WalStore, WriterFence,
};

fn store() -> Arc<dyn ObjectStore> {
    Arc::new(InMemory::new())
}

#[tokio::test]
async fn bootstrap_then_first_write_cycle() {
    let store = store();
    let paths = NamespacePaths::new("tenants", NamespaceId::new("e2e-acme").unwrap());
    let manifest_store = ManifestStore::new(store.clone(), paths.clone());
    let wal_store = WalStore::new(store.clone(), paths);

    let writer_id = Uuid::now_v7();
    let mut current = manifest_store.bootstrap(writer_id).await.unwrap();
    assert_eq!(current.manifest.version, 0);
    assert_eq!(current.manifest.epoch, Epoch::ZERO);

    let fence = WriterFence::new(current.manifest.epoch);
    let mut memtable = Memtable::new();
    let mut next_lsn = 0u64;

    // Stage a batch of writes: two Person nodes.
    let mut seg = WalSegment::new(1);
    for payload in [b"alice".as_ref(), b"bob".as_ref()] {
        next_lsn += 1;
        seg.push(WalRecord {
            lsn: next_lsn,
            payload: Bytes::copy_from_slice(payload),
        });
        memtable.apply(
            MemKey::Node {
                id: namidb_core::NodeId::new(),
            },
            next_lsn,
            MemOp::Upsert(Bytes::copy_from_slice(payload)),
        );
    }
    let segment_path = wal_store.append_segment(&seg).await.unwrap();
    let segment_path_str = segment_path.as_ref().to_string();

    // Reflect the new WAL segment in the manifest.
    let mut next_manifest = current.manifest.next_version(writer_id);
    next_manifest.wal_segments.push(WalSegmentDescriptor {
        seq: seg.seq,
        path: segment_path_str.clone(),
        last_lsn: seg.last_lsn(),
        xxh3: None,
    });
    current = manifest_store
        .commit(&fence, &current, next_manifest)
        .await
        .unwrap();
    assert_eq!(current.manifest.version, 1);
    assert_eq!(current.manifest.wal_segments.len(), 1);
    assert_eq!(current.manifest.wal_segments[0].last_lsn, next_lsn);

    // A fresh reader (no in-process cache) must see the committed state.
    let reader = ManifestStore::new(store.clone(), manifest_store.paths().clone());
    let reloaded = reader.load_current().await.unwrap();
    assert_eq!(reloaded.manifest, current.manifest);

    // List WAL segments — we should see exactly one.
    let segments = wal_store.list_segments().await.unwrap();
    assert_eq!(segments.len(), 1);
    assert_eq!(segments[0].seq, 1);

    let segment_back = wal_store.read_segment(1).await.unwrap();
    assert_eq!(segment_back.records.len(), seg.records.len());
    assert_eq!(segment_back.records[0].payload, seg.records[0].payload);

    // Memtable still holds what we applied locally — proves read-your-writes
    // works without re-fetching from the object store.
    assert_eq!(memtable.len(), 2);
}

#[tokio::test]
async fn second_writer_fences_first() {
    let store = store();
    let paths = NamespacePaths::new("tenants", NamespaceId::new("e2e-fencing").unwrap());
    let manifest_store = ManifestStore::new(store.clone(), paths);

    let writer_a = Uuid::now_v7();
    let bootstrap = manifest_store.bootstrap(writer_a).await.unwrap();
    let fence_a = WriterFence::new(bootstrap.manifest.epoch);

    // Writer B claims the namespace, bumping the epoch.
    let (loaded_b, fence_b) = manifest_store.claim_writer().await.unwrap();
    assert_eq!(loaded_b.manifest.epoch, bootstrap.manifest.epoch.next());
    assert_eq!(fence_b.epoch, loaded_b.manifest.epoch);

    // Writer A tries to commit a new manifest using its stale fence — must
    // be fenced.
    let stale_next = bootstrap.manifest.next_version(writer_a);
    let err = manifest_store
        .commit(&fence_a, &loaded_b, stale_next)
        .await
        .unwrap_err();
    match err {
        namidb_storage::Error::Fenced { mine, current } => {
            assert_eq!(mine, bootstrap.manifest.epoch.as_u64());
            assert_eq!(current, loaded_b.manifest.epoch.as_u64());
        }
        other => panic!("expected Fenced, got {other:?}"),
    }
}

#[tokio::test]
async fn cas_loss_under_concurrent_commits() {
    let store = store();
    let paths = NamespacePaths::new("tenants", NamespaceId::new("e2e-cas").unwrap());
    let ms = ManifestStore::new(store, paths);
    let writer = Uuid::now_v7();
    let base = ms.bootstrap(writer).await.unwrap();
    let fence = WriterFence::new(base.manifest.epoch);

    // Both tasks observe the same base manifest and race to commit v1.
    let next_a = base.manifest.next_version(writer);
    let next_b = base.manifest.next_version(writer);

    let ms_a = ms.clone();
    let base_a = base.clone();
    let fence_a = fence;
    let join_a = tokio::spawn(async move { ms_a.commit(&fence_a, &base_a, next_a).await });
    let ms_b = ms.clone();
    let base_b = base.clone();
    let fence_b = fence;
    let join_b = tokio::spawn(async move { ms_b.commit(&fence_b, &base_b, next_b).await });

    let res_a = join_a.await.unwrap();
    let res_b = join_b.await.unwrap();

    let (winner, loser) = match (res_a, res_b) {
        (Ok(w), Err(l)) => (w, l),
        (Err(l), Ok(w)) => (w, l),
        (Ok(_), Ok(_)) => panic!("both writers won the CAS race"),
        (Err(e1), Err(e2)) => panic!("both writers lost: {e1:?} / {e2:?}"),
    };
    assert_eq!(winner.manifest.version, 1);
    match loser {
        namidb_storage::Error::ManifestCommitCas { expected, found } => {
            assert_eq!(expected, 0);
            assert_eq!(found, 1);
        }
        other => panic!("expected ManifestCommitCas, got {other:?}"),
    }
}

/// Node payloads travel as JSON from the WAL to a flush and from the source
/// rows to the Parquet body during compaction, while the exact record
/// sidecar and the property pages keep the decoded values. Without an exact
/// float parse the last bit of a 17-digit float moved on the JSON leg, so a
/// point lookup and a scan could return different embeddings for one node.
#[tokio::test]
async fn floats_survive_flush_and_compaction_bit_exact_on_every_route() {
    use std::collections::BTreeMap;

    use namidb_core::id::NodeId;
    use namidb_core::value::Value;
    use namidb_storage::{NodeWriteRecord, WriterSession};

    // Doubles that need all 17 significant digits, the shape of a real
    // embedding. 0.10101787042252375 alone used to come back one ulp off.
    let mut bits = 0x9e37_79b9_7f4a_7c15u64;
    let mut floats = vec![0.10101787042252375, 0.30000000000000004];
    for _ in 0..256 {
        bits = bits
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        floats.push((bits >> 11) as f64 / (1u64 << 53) as f64);
    }
    let emb = Value::List(floats.iter().map(|f| Value::F64(*f)).collect());
    let node = |n: i64| NodeWriteRecord {
        properties: BTreeMap::from([
            ("n".to_string(), Value::I64(n)),
            ("emb".to_string(), emb.clone()),
        ]),
        schema_version: 1,
        ..Default::default()
    };

    let paths = NamespacePaths::new("tenants", NamespaceId::new("e2e-floats").unwrap());
    let mut writer = WriterSession::open(store(), paths).await.unwrap();
    let ids: Vec<NodeId> = (0..3).map(|_| NodeId::new()).collect();
    for (n, id) in ids.iter().enumerate().take(2) {
        writer.upsert_node("E", *id, &node(n as i64)).unwrap();
        writer.commit_batch().await.unwrap();
        let schema = writer.snapshot().manifest().manifest.schema.clone();
        writer.flush(schema).await.unwrap();
    }
    let schema = writer.snapshot().manifest().manifest.schema.clone();
    writer.compact_l0(&schema).await.unwrap();
    // A memtable row sends a limited scan through the id-ordered merge.
    writer.upsert_node("E", ids[2], &node(2)).unwrap();
    writer.commit_batch().await.unwrap();

    let snap = writer.snapshot();
    // How many floats came back with different bits.
    let drifted = |v: Option<&Value>| match v {
        Some(Value::List(items)) => {
            assert_eq!(items.len(), floats.len());
            items
                .iter()
                .zip(&floats)
                .filter(|(item, want)| match item {
                    Value::F64(f) => f.to_bits() != want.to_bits(),
                    other => panic!("not a float: {other:?}"),
                })
                .count()
        }
        other => panic!("emb was {other:?}"),
    };
    for id in &ids {
        let view = snap.lookup_node("E", *id).await.unwrap().unwrap();
        assert_eq!(drifted(view.properties.get("emb")), 0, "lookup_node");
    }
    let scanned = snap.scan_label("E").await.unwrap();
    let limited = snap
        .scan_label_with_predicates_and_projection_limited("E", &[], None, 10)
        .await
        .unwrap();
    for (route, views) in [("scan_label", scanned), ("limited merge", limited)] {
        assert_eq!(views.len(), 3, "{route}");
        for view in views {
            assert_eq!(drifted(view.properties.get("emb")), 0, "{route}");
        }
    }
}
