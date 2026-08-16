//! Plan item 29 (docs/testing/25tb-readiness.md): the 2.0.6 downgrade path
//! was only ever simulated by mutating today's structs. This test freezes
//! the ACTUAL 2.0.6 wire contract — the field and variant lists extracted
//! from commit fa126e1 (v2.0.6) and written down here as golden constants —
//! and holds a real 2.1.0 manifest against it:
//!
//!   1. every field a 2.0.6 reader REQUIRES is present with the same shape,
//!   2. every SST descriptor uses only kinds 2.0.6 can decode,
//!   3. pruning the JSON to exactly the 2.0.6 field set — what a downgraded
//!      writer rewrites — still loads in TODAY's decoder (the upgrade after
//!      a downgrade), with every descriptor surviving.
//!
//! Changing any golden list below is a WIRE BREAK against live 2.0.6
//! deployments and must be a deliberate decision, not a drive-by.
#![cfg(all(feature = "vector-index", feature = "text-index"))]

use std::collections::BTreeMap;
use std::sync::Arc;

use namidb_core::id::{NamespaceId, NodeId};
use namidb_core::schema::{DataType, LabelDef, PropertyDef, Schema, SchemaBuilder};
use namidb_core::value::Value as CoreValue;
use namidb_storage::manifest::{
    Manifest, TextIndexDescriptor, VectorIndexDescriptor, VectorMetric, VectorQuantization,
};
use namidb_storage::{NamespacePaths, NodeWriteRecord, WriterSession};
use object_store::memory::InMemory;
use object_store::ObjectStore;

/// Manifest fields as of fa126e1 (v2.0.6). Fields NOT in this list are
/// dropped by a downgraded writer's rewrite.
const MANIFEST_FIELDS_206: &[&str] = &[
    "version",
    "epoch",
    "writer_id",
    "created_at",
    "schema",
    "ssts",
    "wal_segments",
    "label_dict",
    "vector_indexes",
    "text_indexes",
    "search_index_builds",
];

/// Manifest fields 2.0.6 requires (no `serde(default)` at fa126e1).
const MANIFEST_REQUIRED_206: &[&str] = &["version", "epoch", "writer_id", "created_at", "schema"];

/// SstDescriptor required core fields at fa126e1.
const SST_REQUIRED_206: &[&str] = &[
    "id",
    "kind",
    "scope",
    "level",
    "path",
    "size_bytes",
    "row_count",
    "created_at",
    "min_key",
    "max_key",
    "min_lsn",
    "max_lsn",
    "schema_version_min",
    "schema_version_max",
];

/// SstKind variants a 2.0.6 reader can decode.
const SST_KINDS_206: &[&str] = &["Nodes", "EdgesFwd", "EdgesInv", "VectorGraph", "TextIndex"];

fn schema() -> Schema {
    SchemaBuilder::new()
        .label(LabelDef {
            name: "Doc".into(),
            properties: vec![
                PropertyDef::new("embedding", DataType::FloatVector { dim: 4 }, true).unwrap(),
                PropertyDef::new("body", DataType::Utf8, true).unwrap(),
            ],
        })
        .unwrap()
        .build()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_2_1_0_manifest_stays_inside_the_frozen_206_wire_contract() {
    let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
    let paths = NamespacePaths::new("tenants", NamespaceId::new("frozen-wire").unwrap());
    let mut writer = WriterSession::open(store.clone(), paths.clone())
        .await
        .unwrap();
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
    for ordinal in 0..6u64 {
        let mut properties = BTreeMap::new();
        properties.insert(
            "embedding".into(),
            CoreValue::Vec(vec![ordinal as f32, 1.0, 0.0, 0.0]),
        );
        properties.insert("body".into(), CoreValue::Str(format!("word{ordinal}")));
        writer
            .upsert_node(
                "Doc",
                NodeId::new(),
                &NodeWriteRecord {
                    properties,
                    schema_version: 1,
                    ..Default::default()
                },
            )
            .unwrap();
    }
    writer.flush(schema()).await.unwrap();
    writer.compact_l0(&schema()).await.unwrap();

    let ms = namidb_storage::manifest::ManifestStore::new(store.clone(), paths.clone());
    let current = ms.load_current().await.unwrap();
    assert!(
        !current.manifest.search_lsm.is_empty(),
        "the fixture must exercise the 2.1.0-only state a downgrade drops"
    );
    let wire = serde_json::to_value(&current.manifest).unwrap();
    let object = wire
        .as_object()
        .expect("a manifest serializes as an object");

    // 1. Every field 2.0.6 REQUIRES is present.
    for field in MANIFEST_REQUIRED_206 {
        assert!(
            object.contains_key(*field),
            "2.0.6 requires manifest field `{field}`; dropping it bricks \
             live downgraded readers"
        );
    }

    // 2. Every descriptor stays inside the 2.0.6 shape: required core
    // fields present, kind decodable. The compat barrier deliberately rides
    // this shape (an ordinary descriptor with metric \"compat-barrier\").
    let ssts = object["ssts"].as_array().unwrap();
    assert!(!ssts.is_empty());
    for descriptor in ssts {
        let descriptor = descriptor.as_object().unwrap();
        for field in SST_REQUIRED_206 {
            assert!(
                descriptor.contains_key(*field),
                "2.0.6 requires SST field `{field}` on every descriptor"
            );
        }
        let kind = descriptor["kind"].as_str().unwrap();
        assert!(
            SST_KINDS_206.contains(&kind),
            "SST kind `{kind}` is not decodable by a 2.0.6 reader"
        );
    }

    // 3. The downgrade rewrite: prune to exactly the 2.0.6 field set, then
    // load with TODAY's decoder — the upgrade after a downgrade.
    let mut pruned = serde_json::Map::new();
    for (key, value) in object {
        if MANIFEST_FIELDS_206.contains(&key.as_str()) {
            pruned.insert(key.clone(), value.clone());
        }
    }
    assert!(
        pruned.len() < object.len(),
        "the fixture must actually carry 2.1.0-only top-level state"
    );
    let reloaded: Manifest = serde_json::from_value(serde_json::Value::Object(pruned))
        .expect("today's decoder must load a 2.0.6-rewritten manifest");
    assert_eq!(reloaded.version, current.manifest.version);
    assert_eq!(reloaded.ssts.len(), current.manifest.ssts.len());
    assert!(
        reloaded.search_lsm.is_empty(),
        "the pruned manifest models the state drop the adoption tests recover from"
    );
    assert_eq!(
        reloaded.search_index_builds.len(),
        current.manifest.search_index_builds.len(),
        "the interop markers ride a 2.0.6 field and must survive the rewrite"
    );
}
