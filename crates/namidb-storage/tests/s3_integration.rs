//! End-to-end test against a real S3-compatible backend.
//!
//! Marked `#[ignore]` because it requires a running S3 endpoint. The
//! default `docker compose` recipe in `tests/docker-compose.s3.yml`
//! provisions LocalStack (see that file for the rationale — MinIO was
//! archived on 25 April 2026 and Garage / SeaweedFS / RustFS do not
//! advertise support for the conditional-write headers that our CAS
//! protocol depends on).
//!
//! ```bash
//! docker compose -f tests/docker-compose.s3.yml up -d
//!
//! AWS_ACCESS_KEY_ID=test AWS_SECRET_ACCESS_KEY=test \
//! AWS_ENDPOINT_URL=http://127.0.0.1:4566 AWS_ALLOW_HTTP=true \
//! AWS_REGION=us-east-1 \
//! NAMIDB_TEST_BUCKET=namidb-tests \
//! cargo test -p namidb-storage --test s3_integration -- --ignored
//! ```
//!
//! Any S3-compatible endpoint that supports conditional writes works too
//! — point `AWS_ENDPOINT_URL` at it and adjust the credentials.

use std::sync::Arc;

use object_store::aws::AmazonS3Builder;
use object_store::ObjectStore;
use uuid::Uuid;

use namidb_core::NamespaceId;
use namidb_storage::{ManifestStore, NamespacePaths, WriterFence};

fn s3_store_from_env() -> (Arc<dyn ObjectStore>, String) {
    let bucket = std::env::var("NAMIDB_TEST_BUCKET").expect("NAMIDB_TEST_BUCKET must be set");

    // AmazonS3Builder picks up AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY,
    // AWS_ENDPOINT_URL, AWS_REGION, and AWS_ALLOW_HTTP from the env.
    let s3 = AmazonS3Builder::from_env()
        .with_bucket_name(&bucket)
        .build()
        .expect("AmazonS3 client should build from env");

    (Arc::new(s3), bucket)
}

#[tokio::test]
#[ignore]
async fn bootstrap_against_s3() {
    let (store, _bucket) = s3_store_from_env();
    let unique = Uuid::now_v7().simple().to_string();
    let ns_name = format!("it-{}", &unique[..16]);
    let ns = NamespaceId::new(&ns_name).unwrap();
    let paths = NamespacePaths::new("namidb-it", ns);

    let ms = ManifestStore::new(store.clone(), paths);
    let writer = Uuid::now_v7();
    let loaded = ms.bootstrap(writer).await.expect("bootstrap");
    assert_eq!(loaded.manifest.version, 0);

    let fence = WriterFence::new(loaded.manifest.epoch);
    let mut current = loaded;
    for v in 1..=3 {
        let next = current.manifest.next_version(writer);
        current = ms
            .commit(&fence, &current, next)
            .await
            .expect("commit roll forward");
        assert_eq!(current.manifest.version, v);
    }

    // Independent reader should see the latest state.
    let reader = ManifestStore::new(store, ms.paths().clone());
    let reloaded = reader.load_current().await.unwrap();
    assert_eq!(reloaded.manifest.version, 3);
}

/// Plan item 33: the full engine cycle against a REAL S3-compatible backend
/// — ingest with vector+text indexes, flush, compact, reader-node serving
/// of both native routes, a verified backup to a second prefix that serves
/// too, and an orphan sweep. Everything before this ran only on `InMemory`.
#[cfg(all(feature = "vector-index", feature = "text-index"))]
#[tokio::test]
#[ignore]
async fn full_cycle_ingest_search_backup_against_s3() {
    use std::collections::BTreeMap;

    use namidb_core::id::NodeId;
    use namidb_core::schema::{DataType, LabelDef, PropertyDef, SchemaBuilder};
    use namidb_core::value::Value as CoreValue;
    use namidb_storage::manifest::{
        TextIndexDescriptor, VectorIndexDescriptor, VectorMetric, VectorQuantization,
    };
    use namidb_storage::memtable::Memtable;
    use namidb_storage::read::Snapshot;
    use namidb_storage::{copy_namespace_snapshot, sweep_orphans, NodeWriteRecord, WriterSession};

    let (store, _bucket) = s3_store_from_env();
    let unique = Uuid::now_v7().simple().to_string();
    let ns = NamespaceId::new(format!("cycle-{}", &unique[..12])).unwrap();
    let paths = NamespacePaths::new("namidb-it", ns);

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
    let schema = SchemaBuilder::new()
        .label(LabelDef {
            name: "Doc".into(),
            properties: vec![
                PropertyDef::new("embedding", DataType::FloatVector { dim: 4 }, true).unwrap(),
                PropertyDef::new("body", DataType::Utf8, true).unwrap(),
            ],
        })
        .unwrap()
        .build();

    let mut target = None;
    for ordinal in 0..24u64 {
        let id = NodeId::new();
        if ordinal == 0 {
            target = Some(id);
        }
        let mut properties = BTreeMap::new();
        let mut embedding = vec![0.1f32; 4];
        embedding[(ordinal % 4) as usize] = 1.0;
        properties.insert("embedding".into(), CoreValue::Vec(embedding));
        properties.insert(
            "body".into(),
            CoreValue::Str(if ordinal == 0 {
                "unicorn payload".into()
            } else {
                format!("common payload {ordinal}")
            }),
        );
        writer
            .upsert_node(
                "Doc",
                id,
                &NodeWriteRecord {
                    properties,
                    schema_version: 1,
                    ..Default::default()
                },
            )
            .unwrap();
    }
    writer.commit_batch().await.unwrap();
    writer.flush(schema.clone()).await.unwrap();
    writer.compact_l0(&schema).await.unwrap();
    let target = target.unwrap();

    // Reader-node serving over the WAN store: both native routes answer.
    let ms = ManifestStore::new(store.clone(), paths.clone());
    let loaded = ms.load_current().await.unwrap();
    let memtable = Memtable::new();
    let view = memtable.snapshot_view();
    let snapshot = Snapshot::new(loaded, &view, store.clone(), paths.clone());
    let probe = vec![1.0f32, 0.1, 0.1, 0.1];
    let (hits, _) = snapshot
        .try_vector_search_with_point_count("doc_emb", &probe, 3, 64)
        .await
        .unwrap()
        .expect("vector must serve natively against S3");
    assert_eq!(hits.len(), 3);
    let hits = snapshot
        .text_search(
            "doc_ft",
            "Doc",
            &namidb_storage::text::parse_query("unicorn"),
            None,
        )
        .await
        .unwrap()
        .expect("text must serve natively against S3");
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].0, target);

    // Verified backup to a sibling prefix, which must serve identically.
    let backup_ns = NamespaceId::new(format!("cycleb-{}", &unique[..12])).unwrap();
    let backup_paths = NamespacePaths::new("namidb-it", backup_ns);
    copy_namespace_snapshot(
        store.clone(),
        paths.clone(),
        store.clone(),
        backup_paths.clone(),
        None,
        false,
        true,
    )
    .await
    .unwrap();
    let backup_ms = ManifestStore::new(store.clone(), backup_paths.clone());
    let loaded = backup_ms.load_current().await.unwrap();
    let view = memtable.snapshot_view();
    let restored = Snapshot::new(loaded, &view, store.clone(), backup_paths.clone());
    let hits = restored
        .text_search(
            "doc_ft",
            "Doc",
            &namidb_storage::text::parse_query("unicorn"),
            None,
        )
        .await
        .unwrap()
        .expect("the restored namespace must serve the native text route");
    assert_eq!(hits.len(), 1);

    // The janitor runs clean against the live namespace.
    sweep_orphans(&ms, u64::MAX, std::time::Duration::ZERO, 10, true)
        .await
        .unwrap();
    let after = ms.load_current().await.unwrap();
    let view = memtable.snapshot_view();
    let swept = Snapshot::new(after, &view, store.clone(), paths.clone());
    assert!(swept
        .text_search(
            "doc_ft",
            "Doc",
            &namidb_storage::text::parse_query("unicorn"),
            None,
        )
        .await
        .unwrap()
        .is_some());
}
