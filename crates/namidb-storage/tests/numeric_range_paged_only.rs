//! The numeric range route must work in the configuration that actually
//! ships — `NAMIDB_LEGACY_PROPERTY_INDEX_MAX_BYTES=0`, no legacy mirror.
//!
//! That setting changes the DESCRIPTOR SHAPE: the paged body becomes the
//! descriptor's own `path` and the `paged` sidecar field is `None`. The route
//! read only the sidecar field, so it declined on its first SST for every
//! numeric range query in the documented deployment — while every test
//! passed, because no test set the variable and the test-only default is the
//! opposite one.
//!
//! This lives in its OWN test binary on purpose. The variable is read at
//! write time and is process-wide, so flipping it beside tests that assume
//! the default leaks into their corpora; a separate binary is a separate
//! process.

use std::collections::BTreeMap;
use std::sync::Arc;

use namidb_core::id::{NamespaceId, NodeId};
use namidb_core::value::Value;
use namidb_storage::sst::{eval_against_value, ScanPredicate, StatScalar};
use namidb_storage::{NamespacePaths, NodeWriteRecord, WriterSession};
use object_store::memory::InMemory;
use object_store::ObjectStore;

const ROWS: i64 = 2_000;

#[tokio::test]
async fn the_route_works_without_the_legacy_property_sidecar() {
    std::env::set_var("NAMIDB_LEGACY_PROPERTY_INDEX_MAX_BYTES", "0");

    let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
    let paths = NamespacePaths::new("tenants", NamespaceId::new("paged-only").unwrap());
    let mut writer = WriterSession::open(store, paths).await.unwrap();
    for ordinal in 0..ROWS {
        writer
            .upsert_node(
                "T",
                NodeId::new(),
                &NodeWriteRecord {
                    properties: BTreeMap::from([
                        ("idx".to_string(), Value::I64(ordinal)),
                        ("tag".to_string(), Value::Str(format!("row-{ordinal}"))),
                    ]),
                    schema_version: 1,
                    ..Default::default()
                },
            )
            .unwrap();
    }
    writer.commit_batch().await.unwrap();
    writer.create_property_index("T", "idx").await.unwrap();
    let schema = writer.snapshot().manifest().manifest.schema.clone();
    writer.flush(schema.clone()).await.unwrap();
    writer.compact_l0(&schema).await.unwrap();

    let snapshot = writer.snapshot();
    let shapes: Vec<_> = snapshot
        .manifest()
        .manifest
        .ssts
        .iter()
        .flat_map(|sst| sst.equality_property_indices.iter())
        .filter(|index| index.property == "idx")
        .map(|index| (index.paged.is_some(), index.format))
        .collect();
    assert!(
        !shapes.is_empty() && shapes.iter().all(|(has_sidecar, _)| !has_sidecar),
        "this test is only meaningful when the legacy mirror is off, which \
         leaves `paged: None`; got {shapes:?}"
    );

    let predicates = vec![ScanPredicate::Gt {
        column: "idx".into(),
        value: StatScalar::Int64(ROWS - 10),
    }];
    let mut expected: Vec<NodeId> = snapshot
        .scan_label("T")
        .await
        .unwrap()
        .into_iter()
        .filter(|view| {
            predicates
                .iter()
                .all(|p| eval_against_value(p, view.properties.get("idx")))
        })
        .map(|view| view.id)
        .collect();
    expected.sort_unstable();
    assert!(!expected.is_empty(), "the window must not be empty");

    let mut got = snapshot
        .indexed_node_ids_by_numeric_range("T", "idx", &predicates, 4_096)
        .await
        .unwrap()
        .expect(
            "with no legacy mirror the paged body IS the descriptor's own path; \
             reading only the `paged` sidecar field declines every query in the \
             configuration the README documents",
        );
    got.sort_unstable();
    assert_eq!(got, expected);
}
