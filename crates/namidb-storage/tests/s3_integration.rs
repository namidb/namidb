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
