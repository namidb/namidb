//! Fourth field report, items 62/63 storage halves.
//!
//! Item 63: `batch_lookup_nodes_by_property_multi` resolves a whole
//! statement's non-unique String lookups with one claimant pass + one
//! multi-value sidecar probe per SST + one batched confirm — and the
//! per-snapshot pinned-source cache means repeated probes stop paying a
//! fresh HEAD per call.
//!
//! Item 62: `edge_lookup_topology` with no CSR configured serves partner
//! identity from the slim merge — no per-edge property hydration — while
//! matching the property route's partner set exactly.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use namidb_core::id::{NamespaceId, NodeId};
use namidb_core::value::Value;
use namidb_storage::{EdgeWriteRecord, NamespacePaths, NodeWriteRecord, WriterSession};
use object_store::memory::InMemory;
use object_store::path::Path;
use object_store::{
    CopyOptions, GetOptions, GetResult, ListResult, ObjectMeta, ObjectStore, PutMultipartOptions,
    PutOptions, PutPayload, PutResult,
};

/// Delegating spy that counts `head()` calls — the per-lookup cost item 63
/// eliminated for repeated probes within one snapshot.
#[derive(Debug)]
struct HeadCountingStore {
    inner: Arc<dyn ObjectStore>,
    heads: AtomicU64,
}

impl std::fmt::Display for HeadCountingStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "HeadCountingStore({})", self.inner)
    }
}

#[async_trait::async_trait]
impl ObjectStore for HeadCountingStore {
    async fn put_opts(
        &self,
        location: &Path,
        payload: PutPayload,
        opts: PutOptions,
    ) -> object_store::Result<PutResult> {
        self.inner.put_opts(location, payload, opts).await
    }
    async fn put_multipart_opts(
        &self,
        location: &Path,
        opts: PutMultipartOptions,
    ) -> object_store::Result<Box<dyn object_store::MultipartUpload>> {
        self.inner.put_multipart_opts(location, opts).await
    }
    async fn get_opts(
        &self,
        location: &Path,
        options: GetOptions,
    ) -> object_store::Result<GetResult> {
        // In object_store 0.13, `head()` is an extension method over
        // `get_opts` with `head: true` — count it here.
        if options.head {
            self.heads.fetch_add(1, Ordering::Relaxed);
        }
        self.inner.get_opts(location, options).await
    }
    fn list(
        &self,
        prefix: Option<&Path>,
    ) -> futures::stream::BoxStream<'static, object_store::Result<ObjectMeta>> {
        self.inner.list(prefix)
    }
    async fn list_with_delimiter(&self, prefix: Option<&Path>) -> object_store::Result<ListResult> {
        self.inner.list_with_delimiter(prefix).await
    }
    async fn copy_opts(
        &self,
        from: &Path,
        to: &Path,
        options: CopyOptions,
    ) -> object_store::Result<()> {
        self.inner.copy_opts(from, to, options).await
    }
    fn delete_stream(
        &self,
        locations: futures::stream::BoxStream<'static, object_store::Result<Path>>,
    ) -> futures::stream::BoxStream<'static, object_store::Result<Path>> {
        self.inner.delete_stream(locations)
    }
}

fn person(team: &str, seq: i64) -> NodeWriteRecord {
    let mut props: BTreeMap<String, Value> = BTreeMap::new();
    props.insert("team".into(), Value::Str(team.into()));
    props.insert("seq".into(), Value::I64(seq));
    NodeWriteRecord {
        properties: props,
        schema_version: 1,
        ..Default::default()
    }
}

#[tokio::test]
async fn batched_multi_lookup_groups_unions_and_confirms() {
    let spy = Arc::new(HeadCountingStore {
        inner: Arc::new(InMemory::new()),
        heads: AtomicU64::new(0),
    });
    let store: Arc<dyn ObjectStore> = spy.clone();
    let paths = NamespacePaths::new("tenants", NamespaceId::new("bm").unwrap());
    let mut w = WriterSession::open(store, paths).await.unwrap();

    let a1 = NodeId::new();
    let a2 = NodeId::new();
    let b1 = NodeId::new();
    let moved = NodeId::new();
    w.upsert_node("Person", a1, &person("alpha", 1)).unwrap();
    w.upsert_node("Person", a2, &person("alpha", 2)).unwrap();
    w.upsert_node("Person", b1, &person("beta", 3)).unwrap();
    w.upsert_node("Person", moved, &person("beta", 4)).unwrap();
    w.commit_batch().await.unwrap();
    w.create_property_index_named(None, "Person", "team", false)
        .await
        .unwrap();
    let schema = w.snapshot().manifest().manifest.schema.clone();
    w.flush(schema).await.unwrap();

    // Memtable delta: a NEW alpha claimant, and `moved` leaves beta — its
    // flushed posting is now stale and must be dropped by confirmation.
    let a3 = NodeId::new();
    w.upsert_node("Person", a3, &person("alpha", 5)).unwrap();
    w.upsert_node("Person", moved, &person("gamma", 4)).unwrap();
    w.commit_batch().await.unwrap();

    let snap = w.snapshot();
    let values = vec![
        "alpha".to_string(),
        "beta".to_string(),
        "alpha".to_string(),
        "missing".to_string(),
    ];
    let groups = snap
        .batch_lookup_nodes_by_property_multi("Person", "team", &values)
        .await
        .unwrap();
    assert_eq!(
        groups.len(),
        4,
        "one group per input value, duplicates kept"
    );
    let ids = |g: &Vec<namidb_storage::NodeView>| g.iter().map(|v| v.id).collect::<Vec<_>>();
    let mut alpha_expected = vec![a1, a2, a3];
    alpha_expected.sort();
    assert_eq!(ids(&groups[0]), alpha_expected, "SST + memtable union");
    assert_eq!(
        ids(&groups[1]),
        vec![b1],
        "the superseded beta posting must be confirmed away"
    );
    assert_eq!(
        groups[0].iter().map(|v| v.id).collect::<Vec<_>>(),
        ids(&groups[2])
    );
    assert!(groups[3].is_empty(), "a missing value gets an empty group");

    // Item 63's HEAD amortization: a second batch on the SAME snapshot must
    // reuse every pinned sidecar source — zero additional HEADs.
    let before = spy.heads.load(Ordering::Relaxed);
    let again = snap
        .batch_lookup_nodes_by_property_multi("Person", "team", &values)
        .await
        .unwrap();
    assert_eq!(ids(&again[0]), alpha_expected);
    assert_eq!(
        spy.heads.load(Ordering::Relaxed),
        before,
        "repeat probes within one snapshot must not re-HEAD sidecars"
    );
}

#[tokio::test]
async fn batched_multi_lookup_falls_back_to_one_scan_without_a_sidecar() {
    let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
    let paths = NamespacePaths::new("tenants", NamespaceId::new("bmscan").unwrap());
    let mut w = WriterSession::open(store, paths).await.unwrap();
    let a = NodeId::new();
    let b = NodeId::new();
    w.upsert_node("Person", a, &person("alpha", 1)).unwrap();
    w.upsert_node("Person", b, &person("alpha", 2)).unwrap();
    w.commit_batch().await.unwrap();
    // Flushed WITHOUT any index declaration: no sidecar coverage — the
    // batch must serve exactly from one label scan, grouped per value.
    let schema = w.snapshot().manifest().manifest.schema.clone();
    w.flush(schema).await.unwrap();

    let snap = w.snapshot();
    let groups = snap
        .batch_lookup_nodes_by_property_multi("Person", "team", &["alpha".into()])
        .await
        .unwrap();
    let mut expected = vec![a, b];
    expected.sort();
    assert_eq!(groups[0].iter().map(|v| v.id).collect::<Vec<_>>(), expected);
}

#[tokio::test]
async fn topology_fallback_matches_property_route_partners_without_hydration() {
    let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
    let paths = NamespacePaths::new("tenants", NamespaceId::new("topo").unwrap());
    let mut w = WriterSession::open(store, paths).await.unwrap();

    let hub = NodeId::new();
    let s1 = NodeId::new();
    let s2 = NodeId::new();
    let s3 = NodeId::new();
    for id in [hub, s1, s2, s3] {
        w.upsert_node("N", id, &person("x", 0)).unwrap();
    }
    let mut edge_props: BTreeMap<String, Value> = BTreeMap::new();
    edge_props.insert("weight".into(), Value::I64(7));
    let edge = EdgeWriteRecord {
        properties: edge_props,
        schema_version: 1,
    };
    // Reverse-degree hub: s1,s2,s3 -> hub.
    w.upsert_edge("VENTA", s1, hub, &edge).unwrap();
    w.upsert_edge("VENTA", s2, hub, &edge).unwrap();
    w.commit_batch().await.unwrap();
    let schema = w.snapshot().manifest().manifest.schema.clone();
    w.flush(schema).await.unwrap();
    // Memtable delta: a third inbound edge, and s2's edge tombstoned.
    w.upsert_edge("VENTA", s3, hub, &edge).unwrap();
    w.tombstone_edge("VENTA", s2, hub).unwrap();
    w.commit_batch().await.unwrap();

    let snap = w.snapshot();
    // The property route (authoritative comparison point).
    let full = snap.in_edges_via_sst("VENTA", hub).await.unwrap().edges;
    let mut full_partners: Vec<NodeId> = full.iter().map(|e| e.src).collect();
    full_partners.sort();
    // The slim topology route with NO CSR configured.
    let slim = snap
        .edge_lookup_topology("VENTA", hub, namidb_storage::EdgeDirection::Inverse)
        .await
        .unwrap()
        .edges;
    let mut slim_partners: Vec<NodeId> = slim.iter().map(|e| e.src).collect();
    slim_partners.sort();
    assert_eq!(slim_partners, full_partners, "partner sets must match");
    assert!(
        slim.iter().all(|e| e.properties.is_empty()),
        "topology mode must not hydrate properties"
    );
    assert!(
        full.iter().any(|e| !e.properties.is_empty()),
        "the property route still hydrates (sanity check on the contrast)"
    );
    let mut expected = vec![s1, s3];
    expected.sort();
    assert_eq!(slim_partners, expected, "tombstone dropped, delta included");
}

/// Sixth field report: `batch_lookup_nodes` sized every intermediate to the
/// WHOLE id list, so one hop over a 53k-degree endpoint set allocated
/// hundreds of megabytes at once. It now resolves in chunks — with the
/// caller's order, duplicates, and misses preserved exactly.
#[tokio::test]
async fn batch_lookup_nodes_chunks_without_changing_results() {
    std::env::set_var("NAMIDB_BATCH_LOOKUP_CHUNK", "7");
    let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
    let paths = NamespacePaths::new("tenants", NamespaceId::new("chunked").unwrap());
    let mut w = WriterSession::open(store, paths).await.unwrap();

    let mut ids = Vec::new();
    for seq in 0..40i64 {
        let id = NodeId::new();
        ids.push(id);
        w.upsert_node("Person", id, &person("team", seq)).unwrap();
    }
    w.commit_batch().await.unwrap();
    let schema = w.snapshot().manifest().manifest.schema.clone();
    w.flush(schema).await.unwrap();
    let missing = NodeId::new();

    // Interleave duplicates and a miss across several chunk boundaries.
    let mut probe: Vec<NodeId> = Vec::new();
    for (i, id) in ids.iter().enumerate() {
        probe.push(*id);
        if i % 5 == 0 {
            probe.push(*id);
        }
        if i % 11 == 0 {
            probe.push(missing);
        }
    }
    let snap = w.snapshot();
    let views = snap.batch_lookup_nodes("Person", &probe).await.unwrap();
    assert_eq!(views.len(), probe.len(), "one slot per requested id");
    for (requested, view) in probe.iter().zip(&views) {
        if *requested == missing {
            assert!(view.is_none(), "a missing id stays None in its own slot");
        } else {
            assert_eq!(
                view.as_ref().map(|v| v.id),
                Some(*requested),
                "each slot must hold ITS id"
            );
        }
    }
    // The chunked result equals the single-pass result.
    std::env::set_var("NAMIDB_BATCH_LOOKUP_CHUNK", "100000");
    let single = snap.batch_lookup_nodes("Person", &probe).await.unwrap();
    let ids_of = |vs: &[Option<namidb_storage::NodeView>]| {
        vs.iter()
            .map(|v| v.as_ref().map(|v| v.id))
            .collect::<Vec<_>>()
    };
    assert_eq!(ids_of(&views), ids_of(&single));
    std::env::remove_var("NAMIDB_BATCH_LOOKUP_CHUNK");
}
