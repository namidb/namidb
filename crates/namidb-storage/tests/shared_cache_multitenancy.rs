//! Multi-tenant shared-cache tests.
//!
//! The read caches (`SstCache`, `NodeViewCache`, `AdjacencyCache`) are
//! process-wide: `WriterSession::open` attaches the shared instances so the
//! `NAMIDB_*_BUDGET_MIB` knobs bound the PROCESS, not each session — 100
//! concurrent namespaces hold one set of budgets, not 100. That sharing is
//! only sound if
//!
//! 1. no key can collide across namespaces (`SstCache` keys are absolute
//!    paths; `NodeCacheKey` / `AdjacencyKey` embed the namespace prefix),
//! 2. one namespace's flush-time pruning never evicts a sibling's entries.
//!
//! These tests pin both properties, mostly through private cache instances
//! injected via `WriterSession::open_with_caches` so assertions stay
//! deterministic regardless of what else runs in the process.

use std::collections::BTreeMap;
use std::sync::Arc;

use bytes::Bytes;
use object_store::memory::InMemory;
use object_store::ObjectStore;
use uuid::Uuid;

use namidb_core::{NamespaceId, NodeId, Schema, Value};
use namidb_storage::{
    EdgeStreamBundle, EdgeWriteRecord, NamespacePaths, NodeViewCache, NodeWriteRecord,
    SessionCaches, SstCache, WriterSession,
};

fn store() -> Arc<dyn ObjectStore> {
    Arc::new(InMemory::new())
}

fn paths(ns: &str) -> NamespacePaths {
    NamespacePaths::new("tenants", NamespaceId::new(ns).unwrap())
}

/// Deterministic UUIDs whose first byte controls sort order — the SAME id
/// bytes are reused across namespaces on purpose.
fn nid(b: u8) -> NodeId {
    let mut bytes = [0u8; 16];
    bytes[0] = b;
    NodeId::from_uuid(Uuid::from_bytes(bytes))
}

fn node_rec(name: &str) -> NodeWriteRecord {
    let mut props = BTreeMap::new();
    props.insert("name".to_string(), Value::Str(name.into()));
    NodeWriteRecord {
        properties: props,
        schema_version: 0,
        ..Default::default()
    }
}

fn empty_bundle() -> Arc<EdgeStreamBundle> {
    Arc::new(EdgeStreamBundle {
        overflow: None,
        declared: Vec::new(),
    })
}

/// `WriterSession::open` (the default constructor every host and embedded
/// user goes through) must attach the process-wide shared instances — that
/// is the aggregate-accounting fix. Two namespaces, one cache set.
#[tokio::test]
async fn open_defaults_to_process_wide_shared_caches() {
    let store = store();
    let a = WriterSession::open(store.clone(), paths("shared-default-a"))
        .await
        .unwrap();
    let b = WriterSession::open(store.clone(), paths("shared-default-b"))
        .await
        .unwrap();

    let (na, nb) = (
        a.node_cache().expect("node cache on by default"),
        b.node_cache().expect("node cache on by default"),
    );
    assert!(
        Arc::ptr_eq(na, nb),
        "two sessions must share one NodeViewCache instance"
    );
    let (aa, ab) = (
        a.adjacency_cache().expect("adjacency cache on by default"),
        b.adjacency_cache().expect("adjacency cache on by default"),
    );
    assert!(
        Arc::ptr_eq(aa, ab),
        "two sessions must share one AdjacencyCache instance"
    );

    // SstCache is a Clone-able handle over inner Arcs; prove sharing by
    // observation: a body inserted through A's handle is readable via B's.
    let sa = a.sst_cache().expect("sst cache on by default");
    let sb = b.sst_cache().expect("sst cache on by default");
    let probe = "tenants/shared-default-a/sst/level0/probe.parquet".to_string();
    sa.insert(probe.clone(), Bytes::from_static(b"shared"));
    assert_eq!(
        sb.get(&probe),
        Some(Bytes::from_static(b"shared")),
        "two sessions must share one SstCache instance"
    );
}

/// One namespace's flush prunes the shared `SstCache` side maps with ITS
/// live set only. A naive global retain (the pre-multi-tenant behaviour)
/// would evict every other namespace's entries on each flush — this test
/// fails against it.
#[tokio::test]
async fn flush_retain_does_not_evict_sibling_namespace_entries() {
    let store = store();
    let cache = SstCache::new(1 << 20);
    let caches = SessionCaches {
        sst_cache: Some(cache.clone()),
        ..SessionCaches::none()
    };

    let mut a = WriterSession::open_with_caches(store.clone(), paths("ret-a"), caches.clone())
        .await
        .unwrap();
    let mut b = WriterSession::open_with_caches(store.clone(), paths("ret-b"), caches.clone())
        .await
        .unwrap();

    // B flushes real SSTs, then tags side-map entries at its LIVE paths —
    // exactly what a warm edge lookup would have cached.
    b.upsert_node("Person", nid(1), &node_rec("bea")).unwrap();
    b.commit_batch().await.unwrap();
    b.flush(Schema::empty()).await.unwrap();
    let b_live: Vec<String> = {
        let snap = b.snapshot();
        let prefix = paths("ret-b").namespace_prefix();
        snap.manifest()
            .manifest
            .ssts
            .iter()
            .map(|d| format!("{}/{}", prefix.as_ref(), d.path))
            .collect()
    };
    assert!(!b_live.is_empty(), "B must have flushed at least one SST");
    for path in &b_live {
        cache.insert_edge_streams(path.clone(), empty_bundle());
    }

    // A dead entry under A's prefix: gone after A's flush. B's live
    // entries: must survive it.
    let a_dead = "tenants/ret-a/sst/level0/superseded.csr".to_string();
    cache.insert_edge_streams(a_dead.clone(), empty_bundle());

    a.upsert_node("Person", nid(1), &node_rec("ann")).unwrap();
    a.commit_batch().await.unwrap();
    a.flush(Schema::empty()).await.unwrap();

    assert!(
        cache.get_edge_streams(&a_dead).is_none(),
        "A's flush must prune A's dead side-map entries"
    );
    for path in &b_live {
        assert!(
            cache.get_edge_streams(path).is_some(),
            "A's flush evicted sibling namespace entry {path}"
        );
    }
}

/// The budget is global: bodies inserted from two namespaces compete for
/// ONE budget, and the cache stays within it (rather than 2× per-session
/// budgets). Under a roomy budget, entries from both namespaces coexist.
#[tokio::test]
async fn sst_cache_budget_is_global_across_namespaces() {
    // Roomy budget: both namespaces' entries coexist side by side.
    let roomy = SstCache::new(1 << 20);
    roomy.insert("tenants/ns-a/sst/level0/a.parquet".into(), Bytes::from(vec![1u8; 1024]));
    roomy.insert("tenants/ns-b/sst/level0/b.parquet".into(), Bytes::from(vec![2u8; 1024]));
    assert!(roomy.get("tenants/ns-a/sst/level0/a.parquet").is_some());
    assert!(roomy.get("tenants/ns-b/sst/level0/b.parquet").is_some());

    // Tight budget: 16 KiB total, 64 KiB inserted alternately from two
    // namespaces. One global budget must bound the union.
    let budget = 16 * 1024;
    let tight = SstCache::new(budget);
    let mut raw_total = 0u64;
    for i in 0..16 {
        for ns in ["ns-a", "ns-b"] {
            let value = Bytes::from(vec![0u8; 2048]);
            raw_total += value.len() as u64;
            tight.insert(format!("tenants/{ns}/sst/level0/f{i}.parquet"), value);
        }
    }
    // S3FIFO applies some operations lazily, so allow slack — but the
    // usage must clearly track ONE budget, not one per namespace.
    assert!(
        (tight.usage() as u64) < raw_total / 2,
        "usage {} should be bounded well below the {} raw bytes inserted",
        tight.usage(),
        raw_total
    );

    // Same property for the NodeViewCache byte accounting: inserts from
    // two namespaces respect one shared capacity.
    let nvc = NodeViewCache::new(4096);
    for v in 0..64u8 {
        for ns in ["tenants/ns-a", "tenants/ns-b"] {
            nvc.insert(
                namidb_storage::NodeCacheKey::new(ns, u64::from(v), "Person", nid(v)),
                None,
            );
        }
    }
    assert!(
        nvc.used_bytes() <= nvc.capacity_bytes(),
        "NodeViewCache used {} exceeds its global capacity {}",
        nvc.used_bytes(),
        nvc.capacity_bytes()
    );
    assert!(nvc.evictions() > 0, "the tight budget must have evicted");
}

/// Two namespaces with the SAME label and the SAME synthetic node ids —
/// and, because they perform identical write sequences, the SAME manifest
/// version — must read their OWN values through one shared NodeViewCache.
/// Without the namespace component in `NodeCacheKey`, B's first lookup
/// would hit A's freshly-promoted entry and surface A's row.
#[tokio::test]
async fn node_view_cache_isolates_same_ids_across_namespaces() {
    let store = store();
    let node_cache = Arc::new(NodeViewCache::new(1 << 20));
    let caches = SessionCaches {
        node_cache: Some(node_cache.clone()),
        ..SessionCaches::none()
    };

    let mut a = WriterSession::open_with_caches(store.clone(), paths("nv-iso-a"), caches.clone())
        .await
        .unwrap();
    let mut b = WriterSession::open_with_caches(store.clone(), paths("nv-iso-b"), caches.clone())
        .await
        .unwrap();

    for (w, name) in [(&mut a, "from-a"), (&mut b, "from-b")] {
        w.upsert_node("Person", nid(1), &node_rec(name)).unwrap();
        w.commit_batch().await.unwrap();
        w.flush(Schema::empty()).await.unwrap();
    }
    // Sanity: the identical write sequences leave both namespaces at the
    // same manifest version, so the pre-fix key triple truly collides.
    assert_eq!(
        a.manifest_version(),
        b.manifest_version(),
        "test setup must force colliding (manifest_version, label, id)"
    );

    let name_of = |view: Option<namidb_storage::NodeView>| match view
        .expect("node present")
        .properties
        .get("name")
    {
        Some(Value::Str(s)) => s.clone(),
        other => panic!("unexpected name: {other:?}"),
    };

    // Cold lookups populate the shared cache; warm lookups (fresh
    // snapshots, so L1 is empty and L2 must answer) re-read through it.
    for _round in 0..2 {
        let snap_a = a.snapshot();
        let snap_b = b.snapshot();
        assert_eq!(
            name_of(snap_a.lookup_node("Person", nid(1)).await.unwrap()),
            "from-a"
        );
        assert_eq!(
            name_of(snap_b.lookup_node("Person", nid(1)).await.unwrap()),
            "from-b"
        );
    }
    assert!(node_cache.hits() >= 2, "second round must be served by L2");
    assert_eq!(node_cache.namespace_entries("tenants/nv-iso-a"), 1);
    assert_eq!(node_cache.namespace_entries("tenants/nv-iso-b"), 1);
}

/// Same collision shape for the CSR adjacency: two namespaces, same edge
/// type, same source id, same manifest version — different partners. Each
/// must traverse its OWN topology through one shared AdjacencyCache;
/// without the namespace component in `AdjacencyKey`, B's lookup would
/// reuse A's CSR and surface A's partner.
#[tokio::test]
async fn adjacency_cache_isolates_same_topology_keys_across_namespaces() {
    let store = store();
    let adjacency = Arc::new(namidb_storage::AdjacencyCache::new(1 << 20));
    let caches = SessionCaches {
        adjacency_cache: Some(adjacency.clone()),
        ..SessionCaches::none()
    };

    let mut a = WriterSession::open_with_caches(store.clone(), paths("adj-iso-a"), caches.clone())
        .await
        .unwrap();
    let mut b = WriterSession::open_with_caches(store.clone(), paths("adj-iso-b"), caches.clone())
        .await
        .unwrap();

    for (w, partner) in [(&mut a, 2u8), (&mut b, 3u8)] {
        w.upsert_edge("KNOWS", nid(1), nid(partner), &EdgeWriteRecord::default())
            .unwrap();
        w.commit_batch().await.unwrap();
        w.flush(Schema::empty()).await.unwrap();
    }
    assert_eq!(
        a.manifest_version(),
        b.manifest_version(),
        "test setup must force colliding (manifest_version, edge_type, direction)"
    );

    let partners = |edges: namidb_storage::EdgeListView| -> Vec<NodeId> {
        edges.edges.iter().map(|e| e.dst).collect()
    };
    let snap_a = a.snapshot();
    let snap_b = b.snapshot();
    assert_eq!(
        partners(snap_a.out_edges_via_csr("KNOWS", nid(1)).await.unwrap()),
        vec![nid(2)],
        "namespace A must see its own partner"
    );
    assert_eq!(
        partners(snap_b.out_edges_via_csr("KNOWS", nid(1)).await.unwrap()),
        vec![nid(3)],
        "namespace B must see its own partner, not A's cached CSR"
    );
    assert_eq!(adjacency.namespace_entries("tenants/adj-iso-a"), 1);
    assert_eq!(adjacency.namespace_entries("tenants/adj-iso-b"), 1);
}
