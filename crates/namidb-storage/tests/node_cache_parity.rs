//! (RFC-019): parity tests between the legacy uncached `lookup_node`
//! path and the 3-tier path served by [`NodeViewCache`].
//!
//! Same shape as the adjacency parity tests: two public APIs on the
//! same Snapshot (`lookup_node_via_uncached` vs `lookup_node`) bypass the
//! `NAMIDB_NODE_CACHE` env var, so the comparison is deterministic across
//! test parallelism.

use std::sync::Arc;

use bytes::Bytes;
use object_store::memory::InMemory;
use object_store::ObjectStore;
use uuid::Uuid;

use namidb_core::{DataType, LabelDef, NamespaceId, NodeId, PropertyDef, SchemaBuilder, Value};
use namidb_storage::{
    flush, ManifestStore, MemKey, MemOp, Memtable, NamespacePaths, NodeCacheKey, NodeViewCache,
    NodeWriteRecord, Snapshot, WriterFence,
};

fn store() -> Arc<dyn ObjectStore> {
    Arc::new(InMemory::new())
}

fn paths(ns: &str) -> NamespacePaths {
    NamespacePaths::new("tenants", NamespaceId::new(ns).unwrap())
}

fn person_label() -> LabelDef {
    LabelDef {
        name: "Person".into(),
        properties: vec![PropertyDef::new("name", DataType::Utf8, true).unwrap()],
    }
}

fn node_payload(name: &str) -> Bytes {
    NodeWriteRecord {
        properties: [("name".to_string(), Value::Str(name.into()))]
            .into_iter()
            .collect(),
        schema_version: 1,
        ..Default::default()
    }
    .encode()
    .unwrap()
}

fn sorted_node_id(byte: u8) -> NodeId {
    let mut b = [0u8; 16];
    b[0] = byte;
    NodeId::from_uuid(Uuid::from_bytes(b))
}

#[tokio::test]
async fn parity_pure_sst_nodes() {
    let store = store();
    let paths = paths("nc-parity-pure-sst");
    let ms = ManifestStore::new(store.clone(), paths.clone());
    let base = ms.bootstrap(Uuid::now_v7()).await.unwrap();
    let fence = WriterFence::new(base.manifest.epoch);
    let schema = SchemaBuilder::new().label(person_label()).unwrap().build();

    let alice = sorted_node_id(1);
    let bob = sorted_node_id(2);

    let mut mt = Memtable::new();
    for (lsn, id, name) in [(10u64, alice, "Alice"), (11, bob, "Bob")] {
        mt.apply(MemKey::Node { id }, lsn, MemOp::Upsert(node_payload(name)));
    }
    let frozen = mt.freeze();
    let outcome = flush(&ms, &fence, &base, &frozen, schema.clone())
        .await
        .unwrap();

    let empty = Memtable::new();
    let empty_view = empty.snapshot_view();
    let cache = Arc::new(NodeViewCache::new(1024 * 1024));
    let snap = Snapshot::new(
        outcome.committed.clone(),
        &empty_view,
        store.clone(),
        paths.clone(),
    )
    .with_shared_node_cache(cache.clone());

    // Force uncached path vs tiered path. Same Snapshot, same input.
    let from_uncached = snap
        .lookup_node_via_uncached("Person", alice)
        .await
        .unwrap();
    let from_tiered = snap.lookup_node("Person", alice).await.unwrap();
    assert_eq!(from_uncached, from_tiered);

    // After the tiered path ran, the cache must hold the entry.
    let key = NodeCacheKey::new(outcome.committed.manifest.version, "Person", alice);
    let cached = cache.get(&key).expect("L2 hit after first lookup");
    assert_eq!(cached, from_tiered);
    assert!(cache.inserts() >= 1);
}

#[tokio::test]
async fn parity_with_tombstone_caches_negative() {
    let store = store();
    let paths = paths("nc-parity-tombstone");
    let ms = ManifestStore::new(store.clone(), paths.clone());
    let base = ms.bootstrap(Uuid::now_v7()).await.unwrap();
    let fence = WriterFence::new(base.manifest.epoch);
    let schema = SchemaBuilder::new().label(person_label()).unwrap().build();

    let alice = sorted_node_id(1);

    // Flush alice@LSN10.
    let mut mt = Memtable::new();
    mt.apply(
        MemKey::Node { id: alice },
        10,
        MemOp::Upsert(node_payload("Alice")),
    );
    let frozen = mt.freeze();
    let outcome = flush(&ms, &fence, &base, &frozen, schema.clone())
        .await
        .unwrap();

    // Live memtable tombstone @ LSN 20 > SST 10.
    let mut live = Memtable::new();
    live.apply(MemKey::Node { id: alice }, 20, MemOp::Tombstone);

    let cache = Arc::new(NodeViewCache::new(1024 * 1024));
    let live_view = live.snapshot_view();
    let snap = Snapshot::new(
        outcome.committed.clone(),
        &live_view,
        store.clone(),
        paths.clone(),
    )
    .with_shared_node_cache(cache.clone());

    let uncached = snap
        .lookup_node_via_uncached("Person", alice)
        .await
        .unwrap();
    let tiered = snap.lookup_node("Person", alice).await.unwrap();
    assert!(uncached.is_none(), "tombstone hides the SST upsert");
    assert!(tiered.is_none());
    assert_eq!(uncached, tiered);

    // L2 must hold the negative cache entry.
    let key = NodeCacheKey::new(outcome.committed.manifest.version, "Person", alice);
    let cached = cache.get(&key).expect("L2 hit");
    assert!(cached.is_none(), "negative cache");
}

#[tokio::test]
async fn cache_reuses_across_snapshots_of_same_manifest_version() {
    let store = store();
    let paths = paths("nc-reuse");
    let ms = ManifestStore::new(store.clone(), paths.clone());
    let base = ms.bootstrap(Uuid::now_v7()).await.unwrap();
    let fence = WriterFence::new(base.manifest.epoch);
    let schema = SchemaBuilder::new().label(person_label()).unwrap().build();

    let alice = sorted_node_id(1);

    let mut mt = Memtable::new();
    mt.apply(
        MemKey::Node { id: alice },
        10,
        MemOp::Upsert(node_payload("Alice")),
    );
    let frozen = mt.freeze();
    let outcome = flush(&ms, &fence, &base, &frozen, schema.clone())
        .await
        .unwrap();

    let empty = Memtable::new();
    let empty_view = empty.snapshot_view();
    let cache = Arc::new(NodeViewCache::new(1024 * 1024));

    // Snapshot #1: cold miss → insert.
    {
        let snap = Snapshot::new(
            outcome.committed.clone(),
            &empty_view,
            store.clone(),
            paths.clone(),
        )
        .with_shared_node_cache(cache.clone());
        let _ = snap.lookup_node("Person", alice).await.unwrap();
    }
    let inserts_1 = cache.inserts();
    assert_eq!(inserts_1, 1);

    // Snapshot #2 over the SAME manifest version: must hit L2.
    {
        let snap = Snapshot::new(
            outcome.committed.clone(),
            &empty_view,
            store.clone(),
            paths.clone(),
        )
        .with_shared_node_cache(cache.clone());
        let _ = snap.lookup_node("Person", alice).await.unwrap();
    }
    // Inserts didn't grow (still 1); hits must include the snap #2 read.
    assert_eq!(
        cache.inserts(),
        inserts_1,
        "no new insert on cross-snapshot reuse"
    );
    assert!(cache.hits() >= 1, "snapshot #2 must have hit");
}

#[tokio::test]
async fn l1_hit_short_circuits_before_l2() {
    let store = store();
    let paths = paths("nc-l1-shortcut");
    let ms = ManifestStore::new(store.clone(), paths.clone());
    let base = ms.bootstrap(Uuid::now_v7()).await.unwrap();
    let fence = WriterFence::new(base.manifest.epoch);
    let schema = SchemaBuilder::new().label(person_label()).unwrap().build();

    let alice = sorted_node_id(1);

    let mut mt = Memtable::new();
    mt.apply(
        MemKey::Node { id: alice },
        10,
        MemOp::Upsert(node_payload("Alice")),
    );
    let frozen = mt.freeze();
    let outcome = flush(&ms, &fence, &base, &frozen, schema.clone())
        .await
        .unwrap();

    let empty = Memtable::new();
    let empty_view = empty.snapshot_view();
    let cache = Arc::new(NodeViewCache::new(1024 * 1024));
    let snap = Snapshot::new(
        outcome.committed.clone(),
        &empty_view,
        store.clone(),
        paths.clone(),
    )
    .with_shared_node_cache(cache.clone());

    // First call: L1 miss + L2 miss + L3 walk + insert into L1 and L2.
    let _ = snap.lookup_node("Person", alice).await.unwrap();
    let hits_after_first = cache.hits();

    // Second call within the same snapshot: L1 hit, must NOT consult L2.
    let _ = snap.lookup_node("Person", alice).await.unwrap();
    assert_eq!(
        cache.hits(),
        hits_after_first,
        "L1 hit must short-circuit before touching L2"
    );
}
