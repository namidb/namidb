//! Unaudited dimension (25tb-readiness): cache INVALIDATION semantics across
//! flush/compaction/generation swap. At 25 TB the shared caches are always
//! full and always churning; a stale cached node/page served after a
//! compaction replaced its SST is a correctness-envelope failure that
//! per-test-store parity suites cannot catch. The design defense is
//! content-addressed keys — every compaction writes NEW UUID paths and the
//! range cache keys carry the backend generation — and this pins it: with
//! every cache tier at its default (SstCache on, RAM page cache on), reads
//! after each maintenance cycle serve the newest committed value.

use std::collections::BTreeMap;
use std::sync::Arc;

use namidb_core::id::{NamespaceId, NodeId};
use namidb_core::schema::{DataType, LabelDef, PropertyDef, Schema, SchemaBuilder};
use namidb_core::value::Value as CoreValue;
use namidb_storage::memtable::Memtable;
use namidb_storage::read::Snapshot;
use namidb_storage::{NamespacePaths, NodeWriteRecord, WriterSession};
use object_store::memory::InMemory;
use object_store::ObjectStore;

fn schema() -> Schema {
    SchemaBuilder::new()
        .label(LabelDef {
            name: "Doc".into(),
            properties: vec![
                PropertyDef::new("name", DataType::Utf8, false).unwrap(),
                PropertyDef::new("revision", DataType::Int64, true).unwrap(),
            ],
        })
        .unwrap()
        .build()
}

fn doc(name: &str, revision: i64) -> NodeWriteRecord {
    let mut props: BTreeMap<String, CoreValue> = BTreeMap::new();
    props.insert("name".into(), CoreValue::Str(name.into()));
    props.insert("revision".into(), CoreValue::I64(revision));
    NodeWriteRecord {
        properties: props,
        schema_version: 1,
        ..Default::default()
    }
}

async fn revision_of(store: &Arc<dyn ObjectStore>, paths: &NamespacePaths, target: NodeId) -> i64 {
    let manifest_store = namidb_storage::manifest::ManifestStore::new(store.clone(), paths.clone());
    let loaded = manifest_store.load_current().await.unwrap();
    let memtable = Memtable::new();
    let view = memtable.snapshot_view();
    let snapshot = Snapshot::new(loaded, &view, store.clone(), paths.clone());
    let node = snapshot
        .lookup_node("Doc", target)
        .await
        .unwrap()
        .expect("the target row must exist at every revision");
    match node.properties.get("revision") {
        Some(CoreValue::I64(revision)) => *revision,
        other => panic!("revision must hydrate, got {other:?}"),
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn caches_never_serve_a_superseded_row_across_maintenance_cycles() {
    let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
    let paths = NamespacePaths::new("tenants", NamespaceId::new("cache-invalidation").unwrap());
    let mut writer = WriterSession::open(store.clone(), paths.clone())
        .await
        .unwrap();
    let target = NodeId::new();
    writer
        .upsert_node("Doc", target, &doc("target", 0))
        .unwrap();
    // Enough sibling rows that node pages/row groups have real content.
    for ordinal in 0..64u64 {
        writer
            .upsert_node("Doc", NodeId::new(), &doc(&format!("f{ordinal}"), -1))
            .unwrap();
    }
    writer.flush(schema()).await.unwrap();
    writer.compact_l0(&schema()).await.unwrap();

    // Six cycles: read (warming every shared cache), update, flush, compact
    // (replacing the node SST), read again — the fresh revision must win
    // every single time, from a brand-new reader-node snapshot each read.
    for revision in 1..=6i64 {
        let before = revision_of(&store, &paths, target).await;
        assert_eq!(before, revision - 1, "warm read of the prior revision");
        writer
            .upsert_node("Doc", target, &doc("target", revision))
            .unwrap();
        writer.commit_batch().await.unwrap();
        writer.flush(schema()).await.unwrap();
        writer.compact_l0(&schema()).await.unwrap();
        let after = revision_of(&store, &paths, target).await;
        assert_eq!(
            after, revision,
            "a compaction that replaced the SST must never let any cache \
             tier serve the superseded row"
        );
    }
}
