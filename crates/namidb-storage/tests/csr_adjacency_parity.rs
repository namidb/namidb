//! S17.0 (RFC-018): parity tests between the legacy SST edge_lookup path
//! and the CSR-backed path served by [`AdjacencyCache`].
//!
//! Strategy: same Snapshot, two call sites. `Snapshot::out_edges_via_sst`
//! and `Snapshot::out_edges_via_csr` bypass the `NAMIDB_ADJACENCY` env
//! var so the comparison is deterministic across test parallelism.
//!
//! We compare **topology** (src, dst, lsn, tombstone-after-merge), not
//! `EdgeView.properties` — the slim CSR returns empty maps for
//! SST-sourced edges per RFC-018 §4. Memtable-sourced rows retain their
//! property maps in both paths, but the parity assertions don't depend
//! on that to stay robust against future format changes.

use std::collections::BTreeSet;
use std::sync::Arc;

use bytes::Bytes;
use object_store::memory::InMemory;
use object_store::ObjectStore;
use uuid::Uuid;

use namidb_core::{
    DataType, EdgeTypeDef, LabelDef, NamespaceId, NodeId, PropertyDef, SchemaBuilder, Value,
};
use namidb_storage::{
    flush, AdjacencyCache, EdgeWriteRecord, ManifestStore, MemKey, MemOp, Memtable, NamespacePaths,
    Snapshot, WriterFence,
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

fn knows_edge() -> EdgeTypeDef {
    EdgeTypeDef {
        name: "KNOWS".into(),
        src_label: "Person".into(),
        dst_label: "Person".into(),
        properties: vec![],
    }
}

fn edge_payload() -> Bytes {
    EdgeWriteRecord::default().encode().unwrap()
}

/// Deterministic UUIDs whose first byte controls sort order.
fn sorted_node_id(byte: u8) -> NodeId {
    let mut b = [0u8; 16];
    b[0] = byte;
    NodeId::from_uuid(Uuid::from_bytes(b))
}

/// Tuple shape used for topology comparison — order-stable BTreeSet semantics.
type EdgeKey = (NodeId, NodeId, u64);

#[tokio::test]
async fn csr_path_matches_sst_path_on_pure_sst_edges() {
    let store = store();
    let paths = paths("csr-parity-pure-sst");
    let ms = ManifestStore::new(store.clone(), paths.clone());
    let base = ms.bootstrap(Uuid::now_v7()).await.unwrap();
    let fence = WriterFence::new(base.manifest.epoch);
    let schema = SchemaBuilder::new()
        .label(person_label())
        .unwrap()
        .edge_type(knows_edge())
        .unwrap()
        .build();

    let alice = sorted_node_id(1);
    let bob = sorted_node_id(2);
    let carol = sorted_node_id(3);
    let dave = sorted_node_id(4);

    // Flush a fan-out from alice: alice→bob, alice→carol, alice→dave.
    let mut mt = Memtable::new();
    for (lsn, dst) in [(10u64, bob), (11, carol), (12, dave)] {
        mt.apply(
            MemKey::Edge {
                edge_type: "KNOWS".into(),
                src: alice,
                dst,
            },
            lsn,
            MemOp::Upsert(edge_payload()),
        );
    }
    let frozen = mt.freeze();
    let outcome = flush(&ms, &fence, &base, &frozen, schema.clone())
        .await
        .unwrap();

    let empty = Memtable::new();
    let cache = Arc::new(AdjacencyCache::new(1024 * 1024));
    let snap = Snapshot::new(
        outcome.committed.clone(),
        &empty,
        store.clone(),
        paths.clone(),
    )
    .with_adjacency_cache(cache.clone());

    // Out-edges parity.
    let sst = snap.out_edges_via_sst("KNOWS", alice).await.unwrap();
    let csr = snap.out_edges_via_csr("KNOWS", alice).await.unwrap();
    let topo_sst: BTreeSet<EdgeKey> = sst.edges.iter().map(|e| (e.src, e.dst, e.lsn)).collect();
    let topo_csr: BTreeSet<EdgeKey> = csr.edges.iter().map(|e| (e.src, e.dst, e.lsn)).collect();
    assert_eq!(
        topo_sst, topo_csr,
        "topology divergence between SST and CSR paths"
    );

    // Cache must have built exactly one adjacency entry — both
    // `out_edges_via_sst` (no cache touch) and `out_edges_via_csr` (one
    // build) ran in this test, so builds = 1.
    assert_eq!(
        cache.builds(),
        1,
        "build happens once per (version, scope, dir)"
    );
}

#[tokio::test]
async fn csr_path_handles_tombstones_from_memtable_overlay() {
    let store = store();
    let paths = paths("csr-parity-tombstone");
    let ms = ManifestStore::new(store.clone(), paths.clone());
    let base = ms.bootstrap(Uuid::now_v7()).await.unwrap();
    let fence = WriterFence::new(base.manifest.epoch);
    let schema = SchemaBuilder::new()
        .label(person_label())
        .unwrap()
        .edge_type(knows_edge())
        .unwrap()
        .build();

    let alice = sorted_node_id(1);
    let bob = sorted_node_id(2);
    let carol = sorted_node_id(3);

    // Flush alice→bob (LSN 10) + alice→carol (LSN 11).
    let mut mt = Memtable::new();
    mt.apply(
        MemKey::Edge {
            edge_type: "KNOWS".into(),
            src: alice,
            dst: bob,
        },
        10,
        MemOp::Upsert(edge_payload()),
    );
    mt.apply(
        MemKey::Edge {
            edge_type: "KNOWS".into(),
            src: alice,
            dst: carol,
        },
        11,
        MemOp::Upsert(edge_payload()),
    );
    let frozen = mt.freeze();
    let outcome = flush(&ms, &fence, &base, &frozen, schema.clone())
        .await
        .unwrap();

    // Live memtable: tombstone alice→bob at LSN 20 (> SST 10).
    let mut live = Memtable::new();
    live.apply(
        MemKey::Edge {
            edge_type: "KNOWS".into(),
            src: alice,
            dst: bob,
        },
        20,
        MemOp::Tombstone,
    );

    let cache = Arc::new(AdjacencyCache::new(1024 * 1024));
    let snap = Snapshot::new(
        outcome.committed.clone(),
        &live,
        store.clone(),
        paths.clone(),
    )
    .with_adjacency_cache(cache.clone());

    let sst = snap.out_edges_via_sst("KNOWS", alice).await.unwrap();
    let csr = snap.out_edges_via_csr("KNOWS", alice).await.unwrap();
    let dsts_sst: BTreeSet<NodeId> = sst.edges.iter().map(|e| e.dst).collect();
    let dsts_csr: BTreeSet<NodeId> = csr.edges.iter().map(|e| e.dst).collect();
    assert_eq!(
        dsts_sst, dsts_csr,
        "tombstone must hide alice→bob in both paths"
    );
    assert!(
        !dsts_csr.contains(&bob),
        "alice→bob was tombstoned in the live memtable"
    );
    assert!(dsts_csr.contains(&carol));
}

#[tokio::test]
async fn csr_inverse_partner_serves_in_edges_in_parity() {
    let store = store();
    let paths = paths("csr-parity-inverse");
    let ms = ManifestStore::new(store.clone(), paths.clone());
    let base = ms.bootstrap(Uuid::now_v7()).await.unwrap();
    let fence = WriterFence::new(base.manifest.epoch);
    let schema = SchemaBuilder::new()
        .label(person_label())
        .unwrap()
        .edge_type(knows_edge())
        .unwrap()
        .build();

    let alice = sorted_node_id(1);
    let bob = sorted_node_id(2);
    let carol = sorted_node_id(3);

    // bob→alice + carol→alice (two in-edges into alice).
    let mut mt = Memtable::new();
    mt.apply(
        MemKey::Edge {
            edge_type: "KNOWS".into(),
            src: bob,
            dst: alice,
        },
        10,
        MemOp::Upsert(edge_payload()),
    );
    mt.apply(
        MemKey::Edge {
            edge_type: "KNOWS".into(),
            src: carol,
            dst: alice,
        },
        11,
        MemOp::Upsert(edge_payload()),
    );
    let frozen = mt.freeze();
    let outcome = flush(&ms, &fence, &base, &frozen, schema.clone())
        .await
        .unwrap();

    let empty = Memtable::new();
    let cache = Arc::new(AdjacencyCache::new(1024 * 1024));
    let snap = Snapshot::new(
        outcome.committed.clone(),
        &empty,
        store.clone(),
        paths.clone(),
    )
    .with_adjacency_cache(cache.clone());

    let sst = snap.in_edges_via_sst("KNOWS", alice).await.unwrap();
    let csr = snap.in_edges_via_csr("KNOWS", alice).await.unwrap();
    let srcs_sst: BTreeSet<NodeId> = sst.edges.iter().map(|e| e.src).collect();
    let srcs_csr: BTreeSet<NodeId> = csr.edges.iter().map(|e| e.src).collect();
    assert_eq!(srcs_sst, srcs_csr);
    assert!(srcs_csr.contains(&bob));
    assert!(srcs_csr.contains(&carol));
}

#[tokio::test]
async fn csr_returns_empty_properties_for_sst_sourced_edges() {
    // Documented slim-CSR caveat (RFC-018 §4). When the SST has declared
    // properties the legacy path would surface them; the CSR path does not.
    // We don't yet route plan-awarely so this is the contract for v0.
    let store = store();
    let paths = paths("csr-empty-props");
    let ms = ManifestStore::new(store.clone(), paths.clone());
    let base = ms.bootstrap(Uuid::now_v7()).await.unwrap();
    let fence = WriterFence::new(base.manifest.epoch);
    let knows_with_weight = EdgeTypeDef {
        name: "KNOWS".into(),
        src_label: "Person".into(),
        dst_label: "Person".into(),
        properties: vec![PropertyDef::new("weight", DataType::Float64, true).unwrap()],
    };
    let schema = SchemaBuilder::new()
        .label(person_label())
        .unwrap()
        .edge_type(knows_with_weight)
        .unwrap()
        .build();

    let alice = sorted_node_id(1);
    let bob = sorted_node_id(2);

    let mut mt = Memtable::new();
    let weight_payload = EdgeWriteRecord {
        properties: [("weight".into(), Value::F64(2.5))].into_iter().collect(),
        ..Default::default()
    }
    .encode()
    .unwrap();
    mt.apply(
        MemKey::Edge {
            edge_type: "KNOWS".into(),
            src: alice,
            dst: bob,
        },
        10,
        MemOp::Upsert(weight_payload),
    );
    let frozen = mt.freeze();
    let outcome = flush(&ms, &fence, &base, &frozen, schema.clone())
        .await
        .unwrap();

    let empty = Memtable::new();
    let cache = Arc::new(AdjacencyCache::new(1024 * 1024));
    let snap = Snapshot::new(
        outcome.committed.clone(),
        &empty,
        store.clone(),
        paths.clone(),
    )
    .with_adjacency_cache(cache.clone());

    let sst = snap.out_edges_via_sst("KNOWS", alice).await.unwrap();
    let csr = snap.out_edges_via_csr("KNOWS", alice).await.unwrap();

    // SST path surfaces the property.
    assert_eq!(sst.edges.len(), 1);
    assert_eq!(
        sst.edges[0].properties.get("weight"),
        Some(&Value::F64(2.5))
    );

    // CSR path matches on topology but properties are intentionally empty.
    assert_eq!(csr.edges.len(), 1);
    assert_eq!(csr.edges[0].dst, bob);
    assert!(
        csr.edges[0].properties.is_empty(),
        "slim CSR (RFC-018) must NOT carry SST-decoded properties; got {:?}",
        csr.edges[0].properties
    );
}

#[tokio::test]
async fn csr_cache_reuses_across_snapshots_of_same_manifest_version() {
    let store = store();
    let paths = paths("csr-reuse");
    let ms = ManifestStore::new(store.clone(), paths.clone());
    let base = ms.bootstrap(Uuid::now_v7()).await.unwrap();
    let fence = WriterFence::new(base.manifest.epoch);
    let schema = SchemaBuilder::new()
        .label(person_label())
        .unwrap()
        .edge_type(knows_edge())
        .unwrap()
        .build();

    let alice = sorted_node_id(1);
    let bob = sorted_node_id(2);

    let mut mt = Memtable::new();
    mt.apply(
        MemKey::Edge {
            edge_type: "KNOWS".into(),
            src: alice,
            dst: bob,
        },
        10,
        MemOp::Upsert(edge_payload()),
    );
    let frozen = mt.freeze();
    let outcome = flush(&ms, &fence, &base, &frozen, schema.clone())
        .await
        .unwrap();

    let empty = Memtable::new();
    let cache = Arc::new(AdjacencyCache::new(1024 * 1024));

    // Snapshot #1: cold miss → 1 build.
    {
        let snap = Snapshot::new(
            outcome.committed.clone(),
            &empty,
            store.clone(),
            paths.clone(),
        )
        .with_adjacency_cache(cache.clone());
        let _ = snap.out_edges_via_csr("KNOWS", alice).await.unwrap();
    }
    assert_eq!(cache.builds(), 1);

    // Snapshot #2 over the SAME manifest version: must hit, no new build.
    {
        let snap = Snapshot::new(
            outcome.committed.clone(),
            &empty,
            store.clone(),
            paths.clone(),
        )
        .with_adjacency_cache(cache.clone());
        let _ = snap.out_edges_via_csr("KNOWS", alice).await.unwrap();
    }
    assert_eq!(
        cache.builds(),
        1,
        "second snapshot must reuse the cached adjacency"
    );
    // Snapshot #1 missed (then built); snapshot #2 hit.
    assert_eq!(cache.hits(), 1, "second snapshot's lookup is a hit");
}
