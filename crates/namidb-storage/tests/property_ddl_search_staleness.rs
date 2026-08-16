//! Found live on 2026-08-16 during the 200k-node S3 validation: `CREATE
//! INDEX` on any Bool/Utf8 property of an indexed label changes the search
//! catalog signature (the native-filter set), and the DDL committed the new
//! schema WITHOUT retiring the now-stale Active generations — every
//! subsequent flush and compaction failed the catalog invariant forever
//! (maintenance wedged, search pinned to flat scans). These pin both halves
//! of the fix: the DDL retires stale generations in its own commit, and a
//! flush over an already-poisoned manifest self-heals instead of erroring.
#![cfg(all(feature = "vector-index", feature = "text-index"))]

use std::collections::BTreeMap;
use std::sync::Arc;

use namidb_core::id::{NamespaceId, NodeId};
use namidb_core::schema::{DataType, LabelDef, PropertyDef, Schema, SchemaBuilder};
use namidb_core::value::Value as CoreValue;
use namidb_storage::manifest::{
    TextIndexDescriptor, VectorIndexDescriptor, VectorMetric, VectorQuantization,
};
use namidb_storage::{NamespacePaths, NodeWriteRecord, WriterSession};
use object_store::memory::InMemory;
use object_store::ObjectStore;

fn schema(city_indexed: bool) -> Schema {
    let mut city = PropertyDef::new("city", DataType::Utf8, true).unwrap();
    if city_indexed {
        city = city.with_indexed(true);
    }
    SchemaBuilder::new()
        .label(LabelDef {
            name: "Doc".into(),
            properties: vec![
                PropertyDef::new("embedding", DataType::FloatVector { dim: 4 }, true).unwrap(),
                PropertyDef::new("body", DataType::Utf8, true).unwrap(),
                city,
            ],
        })
        .unwrap()
        .build()
}

fn doc(ordinal: u64) -> NodeWriteRecord {
    let mut props = BTreeMap::new();
    props.insert(
        "embedding".into(),
        CoreValue::Vec(vec![ordinal as f32, 1.0, 0.0, 0.0]),
    );
    props.insert("body".into(), CoreValue::Str(format!("alpha doc{ordinal}")));
    props.insert("city".into(), CoreValue::Str("madrid".into()));
    NodeWriteRecord {
        properties: props,
        schema_version: 1,
        ..Default::default()
    }
}

async fn active_writer(name: &str) -> WriterSession {
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
    for ordinal in 0..8u64 {
        writer
            .upsert_node("Doc", NodeId::new(), &doc(ordinal))
            .unwrap();
    }
    writer.flush(schema(false)).await.unwrap();
    writer.compact_l0(&schema(false)).await.unwrap();
    assert_eq!(
        writer.snapshot().manifest().manifest.search_lsm.len(),
        2,
        "both generations must be active before the DDL"
    );
    writer
}

/// The prevention half: the DDL commit itself retires stale generations,
/// keeps the manifest valid, and the next maintenance cycle rebuilds.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn property_index_ddl_retires_stale_generations_and_rebuilds() {
    let mut writer = active_writer("ddl-signature-index").await;

    writer
        .create_property_index_named(None, "Doc", "city", false)
        .await
        .expect("the property DDL must commit");
    let manifest = writer.snapshot().manifest().manifest.clone();
    namidb_storage::search_lsm::validate_search_lsm(&manifest)
        .expect("the DDL commit must leave a VALID manifest");
    assert!(
        manifest.search_lsm.is_empty(),
        "signature-stale generations must be retired in the DDL commit"
    );

    // The next maintenance cycle rebuilds under the new signature and serves.
    writer.upsert_node("Doc", NodeId::new(), &doc(99)).unwrap();
    writer
        .flush(schema(true))
        .await
        .expect("flush after the DDL must not wedge");
    writer
        .compact_l0(&schema(true))
        .await
        .expect("compaction after the DDL must not wedge");
    let manifest = writer.snapshot().manifest().manifest.clone();
    assert_eq!(
        manifest.search_lsm.len(),
        2,
        "fresh generations must exist under the new signature"
    );
    let snapshot = writer.snapshot();
    let hits = snapshot
        .text_search(
            "doc_ft",
            "Doc",
            &namidb_storage::text::parse_query("alpha"),
            None,
        )
        .await
        .unwrap()
        .expect("the rebuilt text generation must serve natively");
    assert_eq!(hits.len(), 9);
}

/// The self-heal half: a manifest already poisoned by an older writer (schema
/// committed without retiring) must not wedge the next flush — it retires the
/// stale generation and rebuilds instead of erroring forever.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn flush_self_heals_a_catalog_stale_manifest() {
    let mut writer = active_writer("ddl-signature-heal").await;

    // Model the OLD writer's bug: commit the indexed-city schema directly,
    // leaving the generations' signatures stale (what 2.1.0 did).
    writer
        .commit_schema_without_search_retirement_for_test(schema(true))
        .await
        .expect("poisoned commit models the old writer");
    assert!(
        namidb_storage::search_lsm::validate_search_lsm(&writer.snapshot().manifest().manifest)
            .is_err(),
        "the fixture must actually be catalog-stale"
    );

    // The next flush must self-heal, not error.
    writer.upsert_node("Doc", NodeId::new(), &doc(50)).unwrap();
    writer
        .flush(schema(true))
        .await
        .expect("flush over a catalog-stale manifest must self-heal");
    writer.compact_l0(&schema(true)).await.unwrap();
    let manifest = writer.snapshot().manifest().manifest.clone();
    namidb_storage::search_lsm::validate_search_lsm(&manifest).unwrap();
    let snapshot = writer.snapshot();
    let hits = snapshot
        .text_search(
            "doc_ft",
            "Doc",
            &namidb_storage::text::parse_query("alpha"),
            None,
        )
        .await
        .unwrap()
        .expect("the healed generation must serve natively");
    assert_eq!(hits.len(), 9);
}
