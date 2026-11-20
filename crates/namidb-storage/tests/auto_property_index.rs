//! Automatic property indexes: declared in the ordinary equality-sidecar
//! format with the `auto_indexed` marker, never when the index would change a
//! search index's catalog signature (that retires and rebuilds the search
//! generation), and adopted as a normal index by an explicit `CREATE INDEX`.
#![cfg(all(feature = "vector-index", feature = "text-index"))]

use std::collections::BTreeMap;
use std::sync::Arc;

use namidb_core::id::{NamespaceId, NodeId};
use namidb_core::value::Value as CoreValue;
use namidb_storage::manifest::{
    TextIndexDescriptor, VectorIndexDescriptor, VectorMetric, VectorQuantization,
};
use namidb_storage::{AutoIndexOutcome, NamespacePaths, NodeWriteRecord, WriterSession};
use object_store::memory::InMemory;
use object_store::ObjectStore;

fn doc(ordinal: u64) -> NodeWriteRecord {
    let mut props = BTreeMap::new();
    props.insert(
        "embedding".into(),
        CoreValue::Vec(vec![ordinal as f32, 1.0, 0.0, 0.0]),
    );
    props.insert("body".into(), CoreValue::Str(format!("alpha doc{ordinal}")));
    props.insert("city".into(), CoreValue::Str("madrid".into()));
    props.insert("level".into(), CoreValue::I64((ordinal % 4) as i64));
    NodeWriteRecord {
        properties: props,
        schema_version: 1,
        ..Default::default()
    }
}

async fn searched_writer(name: &str) -> WriterSession {
    let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
    let paths = NamespacePaths::new("tenants", NamespaceId::new(name).unwrap());
    let mut writer = WriterSession::open(store, paths).await.unwrap();
    writer
        .register_vector_index(
            VectorIndexDescriptor {
                name: "doc_emb".into(),
                label: "Doc".into(),
                property: "embedding".into(),
                dim: 4,
                metric: VectorMetric::Cosine,
                r: 16,
                l_build: 32,
                alpha: 1.2,
                quantization: VectorQuantization::None,
            },
            false,
        )
        .await
        .unwrap();
    writer
        .register_text_index(
            TextIndexDescriptor::new("doc_ft".into(), "Doc".into(), vec!["body".into()]),
            false,
        )
        .await
        .unwrap();
    for ordinal in 0..16u64 {
        writer
            .upsert_node("Doc", NodeId::new(), &doc(ordinal))
            .unwrap();
    }
    let schema = writer.schema().clone();
    writer.flush(schema.clone()).await.unwrap();
    writer.compact_l0(&schema).await.unwrap();
    assert_eq!(writer.snapshot().manifest().manifest.search_lsm.len(), 2);
    writer
}

fn property(writer: &WriterSession, name: &str) -> Option<namidb_core::PropertyDef> {
    writer
        .schema()
        .label("Doc")
        .and_then(|label| label.properties.iter().find(|p| p.name == name))
        .cloned()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn auto_index_never_rebuilds_a_search_index() {
    let mut writer = searched_writer("auto-index-search").await;
    let before = writer.snapshot().manifest().manifest.clone();

    // A string property of a searched label is a native-filter candidate:
    // indexing it would change both catalog signatures.
    let outcome = writer
        .create_auto_property_index("Doc", "city")
        .await
        .unwrap();
    let AutoIndexOutcome::WouldRebuildSearch(mut indexes) = outcome else {
        panic!("expected a search-rebuild refusal, got {outcome:?}");
    };
    indexes.sort();
    assert_eq!(indexes, ["doc_emb", "doc_ft"]);
    let after = writer.snapshot().manifest().manifest.clone();
    assert_eq!(after.version, before.version, "nothing may be committed");
    assert!(!property(&writer, "city").is_some_and(|p| p.indexed));

    // A numeric property leaves every signature unchanged.
    let outcome = writer
        .create_auto_property_index("Doc", "level")
        .await
        .unwrap();
    assert!(
        matches!(outcome, AutoIndexOutcome::Created { .. }),
        "{outcome:?}"
    );
    let level = property(&writer, "level").unwrap();
    assert!(level.indexed && level.auto_indexed);
    let manifest = writer.snapshot().manifest().manifest.clone();
    namidb_storage::search_lsm::validate_search_lsm(&manifest).unwrap();
    assert_eq!(
        manifest.search_lsm.len(),
        2,
        "no search generation may be retired by an automatic index"
    );
    assert_eq!(
        manifest
            .search_lsm
            .iter()
            .map(|state| state.generation_id)
            .collect::<Vec<_>>(),
        before
            .search_lsm
            .iter()
            .map(|state| state.generation_id)
            .collect::<Vec<_>>()
    );

    assert_eq!(
        writer
            .create_auto_property_index("Doc", "level")
            .await
            .unwrap(),
        AutoIndexOutcome::AlreadyIndexed
    );
    assert_eq!(
        writer
            .create_auto_property_index("Doc", "missing")
            .await
            .unwrap(),
        AutoIndexOutcome::NoValues
    );

    // Compaction builds the sidecars; the index is then fully populated.
    let schema = writer.schema().clone();
    writer.compact_l0(&schema).await.unwrap();
    let manifest = writer.snapshot().manifest().manifest.clone();
    assert_eq!(
        namidb_storage::property_index_population(&manifest, "Doc", "level"),
        1.0
    );
    let ids = writer
        .snapshot()
        .indexed_node_ids_by_property_value("Doc", "level", &CoreValue::I64(1))
        .await
        .unwrap()
        .expect("the automatic index serves lookups once built");
    assert_eq!(ids.len(), 4);

    // An explicit CREATE INDEX adopts it as an ordinary index.
    writer
        .create_property_index_named(None, "Doc", "level", false)
        .await
        .unwrap();
    let level = property(&writer, "level").unwrap();
    assert!(level.indexed && !level.auto_indexed);
}

/// The marker is an optional field: omitted when false, and a manifest from
/// an engine that does not know it (no field) still loads with a plain index.
#[test]
fn auto_indexed_marker_is_optional_on_the_wire() {
    let mut p = namidb_core::PropertyDef::new("level", namidb_core::DataType::Int64, true)
        .unwrap()
        .with_indexed(true);
    let plain = serde_json::to_string(&p).unwrap();
    assert!(!plain.contains("auto_indexed"));
    p.auto_indexed = true;
    let marked = serde_json::to_string(&p).unwrap();
    assert!(marked.contains("\"auto_indexed\":true"));
    let back: namidb_core::PropertyDef = serde_json::from_str(&marked).unwrap();
    assert!(back.auto_indexed && back.indexed);
    let old: namidb_core::PropertyDef = serde_json::from_str(&plain).unwrap();
    assert!(old.indexed && !old.auto_indexed);
}
