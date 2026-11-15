//! A bulk load into a label with search indexes must never wedge the flush.
//!
//! Two shapes did: an explicitly indexed, non-unique identifier (a RANGE
//! index on `id`, so a native search filter) collecting more distinct values
//! in one flush than the per-segment filter cap, and one document whose
//! tokenization estimate exceeded the per-index record workspace. Either made
//! every periodic flush of the namespace fail, so the memtable could only
//! grow and writes queued behind the retried flush. A native filter is an
//! optional accelerator and a document the write accepted must be indexable,
//! so both now degrade (the filter is not advertised by that segment; the
//! document is indexed) instead of failing.
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

/// More distinct identifiers in one flush than the default cap (4096).
const ARTICLES: u64 = 5_000;

fn schema() -> Schema {
    SchemaBuilder::new()
        .label(LabelDef {
            name: "Articulo".into(),
            properties: vec![
                PropertyDef::new("embedding", DataType::FloatVector { dim: 4 }, true).unwrap(),
                PropertyDef::new("texto", DataType::Utf8, true).unwrap(),
                PropertyDef::new("id", DataType::Utf8, true)
                    .unwrap()
                    .with_indexed(true),
                PropertyDef::new("estado", DataType::Utf8, true)
                    .unwrap()
                    .with_indexed(true),
            ],
        })
        .unwrap()
        .build()
}

fn article(ordinal: u64, texto: String) -> NodeWriteRecord {
    let mut props = BTreeMap::new();
    props.insert(
        "embedding".into(),
        CoreValue::Vec(vec![ordinal as f32 + 1.0, 1.0, 0.0, 0.0]),
    );
    props.insert("texto".into(), CoreValue::Str(texto));
    props.insert("id".into(), CoreValue::Str(format!("art-{ordinal:06}")));
    props.insert(
        "estado".into(),
        CoreValue::Str(
            if ordinal % 4 == 0 {
                "derogado"
            } else {
                "vigente"
            }
            .into(),
        ),
    );
    NodeWriteRecord {
        properties: props,
        schema_version: 1,
        ..Default::default()
    }
}

async fn assert_search_serves(writer: &WriterSession, phase: &str) {
    let snapshot = writer.snapshot();
    let query = namidb_storage::text::parse_query("reglamento");
    let hits = snapshot
        .text_search("articulo_ft", "Articulo", &query, None)
        .await
        .unwrap()
        .unwrap_or_else(|| panic!("{phase}: unfiltered text search must serve natively"));
    assert_eq!(hits.len() as u64, ARTICLES + 1, "{phase}");

    // The low-cardinality filter stays a native filter.
    let vigente = snapshot
        .text_search_filter_groups(
            "articulo_ft",
            "Articulo",
            &query,
            None,
            &[("estado".into(), vec![CoreValue::Str("vigente".into())])],
        )
        .await
        .unwrap()
        .unwrap_or_else(|| panic!("{phase}: estado must stay a native filter"));
    assert_eq!(vigente.len() as u64, ARTICLES * 3 / 4, "{phase}");

    // The over-cap identifier is not advertised: the caller takes the exact
    // node-scan fallback instead of trusting an incomplete posting.
    let by_id = snapshot
        .text_search_filter_groups(
            "articulo_ft",
            "Articulo",
            &query,
            None,
            &[("id".into(), vec![CoreValue::Str("art-000007".into())])],
        )
        .await
        .unwrap();
    assert!(by_id.is_none(), "{phase}: id must fall back, got {by_id:?}");

    let nearest = snapshot
        .vector_search("articulo_emb", &[1.0, 1.0, 0.0, 0.0], 5, 64)
        .await
        .unwrap();
    assert_eq!(nearest.len(), 5, "{phase}");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn bulk_load_with_indexed_ids_and_an_oversized_document_flushes_and_compacts() {
    let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
    let paths = NamespacePaths::new("tenants", NamespaceId::new("bulk-search").unwrap());
    let mut writer = WriterSession::open(store, paths).await.unwrap();
    writer
        .register_vector_index(
            VectorIndexDescriptor {
                name: "articulo_emb".into(),
                label: "Articulo".into(),
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
            TextIndexDescriptor::new(
                "articulo_ft".into(),
                "Articulo".into(),
                vec!["texto".into()],
            ),
            false,
        )
        .await
        .unwrap();
    // First flush makes both generations exist.
    writer
        .upsert_node(
            "Articulo",
            NodeId::new(),
            &article(ARTICLES, "reglamento inicial".into()),
        )
        .unwrap();
    writer.flush(schema()).await.unwrap();

    for ordinal in 0..ARTICLES {
        writer
            .upsert_node(
                "Articulo",
                NodeId::new(),
                &article(
                    ordinal,
                    format!("articulo {ordinal} del reglamento general"),
                ),
            )
            .unwrap();
    }
    // ~200 KiB of text: above the per-index record workspace that the
    // default flush build budget gives each of the two search indexes.
    let oversized = (0..20_000)
        .map(|word| format!("palabra{} ", word % 5_000))
        .collect::<String>();
    assert!(oversized.len() > 128 * 1024);
    let mut big = article(ARTICLES + 1, oversized);
    big.properties
        .insert("estado".into(), CoreValue::Str("derogado".into()));
    writer.upsert_node("Articulo", NodeId::new(), &big).unwrap();

    writer
        .flush(schema())
        .await
        .expect("a bulk flush over indexed identifiers must not fail");
    let manifest = writer.snapshot().manifest().manifest.clone();
    namidb_storage::search_lsm::validate_search_lsm(&manifest).unwrap();
    assert_eq!(manifest.search_lsm.len(), 2, "both generations stay");
    assert_search_serves(&writer, "after flush").await;

    writer
        .compact_l0(&schema())
        .await
        .expect("compaction over indexed identifiers must not fail");
    let manifest = writer.snapshot().manifest().manifest.clone();
    namidb_storage::search_lsm::validate_search_lsm(&manifest).unwrap();
    assert_search_serves(&writer, "after compaction").await;
}
