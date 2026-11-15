//! Object-store round trips on the write path (issue #212 and the commit
//! protocol).
//!
//! On a remote store every request costs tens of milliseconds, so the number
//! of requests a write pays (and how many of them are sequential) is the
//! write latency. These tests pin three properties with a counting store:
//!
//! - an immutable SST sidecar is HEADed once per process, not once per
//!   snapshot (every commit publishes a new one);
//! - `commit_batch` does not wait for the advisory `current.json` PUT: a
//!   commit completes even while that PUT cannot;
//! - the advisory it defers is published by the next commit or flush, and a
//!   namespace whose advisory lags still resolves its true current version.

use std::collections::BTreeMap;
use std::fmt;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use futures::stream::BoxStream;
use namidb_core::id::{NamespaceId, NodeId};
use namidb_core::schema::Schema;
use namidb_core::value::Value as CoreValue;
use namidb_storage::manifest::{ManifestPointer, ManifestStore};
use namidb_storage::memtable::Memtable;
use namidb_storage::read::Snapshot;
use namidb_storage::{NamespacePaths, NodeWriteRecord, WriterSession};
use object_store::memory::InMemory;
use object_store::path::Path;
use object_store::{
    CopyOptions, GetOptions, GetResult, ListResult, MultipartUpload, ObjectMeta, ObjectStore,
    ObjectStoreExt, PutMultipartOptions, PutOptions, PutPayload, PutResult,
};

/// Counts HEADs of SST objects and can hold every PUT of `current.json`.
#[derive(Debug)]
struct CountingStore {
    inner: Arc<dyn ObjectStore>,
    sst_heads: AtomicUsize,
    current_puts: AtomicUsize,
    hold_current: AtomicBool,
    released: tokio::sync::Notify,
}

impl CountingStore {
    fn new() -> Arc<Self> {
        Arc::new(Self {
            inner: Arc::new(InMemory::new()),
            sst_heads: AtomicUsize::new(0),
            current_puts: AtomicUsize::new(0),
            hold_current: AtomicBool::new(false),
            released: tokio::sync::Notify::new(),
        })
    }

    fn release_current(&self) {
        self.hold_current.store(false, Ordering::SeqCst);
        self.released.notify_waiters();
    }
}

impl fmt::Display for CountingStore {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "CountingStore")
    }
}

#[async_trait::async_trait]
impl ObjectStore for CountingStore {
    async fn put_opts(
        &self,
        location: &Path,
        payload: PutPayload,
        opts: PutOptions,
    ) -> object_store::Result<PutResult> {
        if location.as_ref().ends_with("manifest/current.json") {
            while self.hold_current.load(Ordering::SeqCst) {
                let released = self.released.notified();
                if !self.hold_current.load(Ordering::SeqCst) {
                    break;
                }
                released.await;
            }
            self.current_puts.fetch_add(1, Ordering::SeqCst);
        }
        self.inner.put_opts(location, payload, opts).await
    }

    async fn put_multipart_opts(
        &self,
        location: &Path,
        opts: PutMultipartOptions,
    ) -> object_store::Result<Box<dyn MultipartUpload>> {
        self.inner.put_multipart_opts(location, opts).await
    }

    async fn get_opts(
        &self,
        location: &Path,
        options: GetOptions,
    ) -> object_store::Result<GetResult> {
        if options.head && location.as_ref().contains("/sst/") {
            self.sst_heads.fetch_add(1, Ordering::SeqCst);
        }
        self.inner.get_opts(location, options).await
    }

    fn delete_stream(
        &self,
        locations: BoxStream<'static, object_store::Result<Path>>,
    ) -> BoxStream<'static, object_store::Result<Path>> {
        self.inner.delete_stream(locations)
    }

    fn list(&self, prefix: Option<&Path>) -> BoxStream<'static, object_store::Result<ObjectMeta>> {
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
}

fn doc(key: &str) -> NodeWriteRecord {
    let mut properties = BTreeMap::new();
    properties.insert("key".to_string(), CoreValue::Str(key.to_string()));
    properties.insert("body".to_string(), CoreValue::Str(format!("body of {key}")));
    NodeWriteRecord {
        properties,
        schema_version: 1,
        ..Default::default()
    }
}

fn paths(name: &str) -> NamespacePaths {
    NamespacePaths::new("tenants", NamespaceId::new(name).unwrap())
}

#[tokio::test]
async fn sidecar_heads_are_not_repeated_for_every_published_snapshot() {
    let counting = CountingStore::new();
    let store: Arc<dyn ObjectStore> = counting.clone();
    let ns = paths("head-cache");
    let mut writer = WriterSession::open(store.clone(), ns.clone())
        .await
        .unwrap();
    // Two flushed generations, so lookups probe more than one SST.
    let mut ids = Vec::new();
    for generation in 0..2 {
        for i in 0..20 {
            let id = NodeId::new();
            writer
                .upsert_node("Doc", id, &doc(&format!("k-{generation}-{i}")))
                .unwrap();
            ids.push(id);
        }
        writer.commit_batch().await.unwrap();
        writer.flush(Schema::empty()).await.unwrap();
    }

    let empty = Memtable::new();
    let empty_view = empty.snapshot_view();
    let mut heads_per_round = Vec::new();
    for round in 0..4 {
        // Every commit publishes a new snapshot. Build one per round from the
        // committed manifest with no logical caches attached, so every lookup
        // goes to the SST sidecars, as a write's first probe does.
        writer
            .upsert_node("Doc", NodeId::new(), &doc(&format!("new-{round}")))
            .unwrap();
        writer.commit_batch().await.unwrap();
        let manifest = ManifestStore::new(store.clone(), ns.clone())
            .load_current()
            .await
            .unwrap();
        let before = counting.sst_heads.load(Ordering::SeqCst);
        let snapshot = Snapshot::new(manifest, &empty_view, store.clone(), ns.clone());
        // A batched id lookup, as `MERGE` and the flush's search before-image
        // pass issue: it probes each SST's node-locator sidecar.
        let found = snapshot
            .batch_lookup_nodes("", &[ids[round], ids[25 + round]])
            .await
            .unwrap();
        assert!(
            found.iter().all(Option::is_some),
            "round {round}: batched lookup by id"
        );
        drop(snapshot);
        heads_per_round.push(counting.sst_heads.load(Ordering::SeqCst) - before);
    }
    assert!(
        heads_per_round[0] > 0,
        "the first snapshot must HEAD the sidecars it opens: {heads_per_round:?}"
    );
    assert!(
        heads_per_round[1..].iter().all(|heads| *heads == 0),
        "immutable sidecars must not be HEADed again by the next snapshot; HEADs per round: \
         {heads_per_round:?}"
    );
}

#[tokio::test]
async fn commit_does_not_wait_for_the_advisory_current_json() {
    let counting = CountingStore::new();
    let store: Arc<dyn ObjectStore> = counting.clone();
    let mut writer = WriterSession::open(store, paths("advisory-defer"))
        .await
        .unwrap();
    // Every `current.json` PUT now hangs until released.
    counting.hold_current.store(true, Ordering::SeqCst);
    writer.upsert_node("Doc", NodeId::new(), &doc("a")).unwrap();
    let committed = tokio::time::timeout(Duration::from_secs(10), writer.commit_batch()).await;
    assert!(
        committed.is_ok(),
        "commit_batch must not wait for the advisory current.json PUT"
    );
    committed.unwrap().unwrap();
    assert!(writer.manifest_store().has_pending_advisory());
    counting.release_current();

    // The owed advisory rides along with the next commit.
    writer.upsert_node("Doc", NodeId::new(), &doc("b")).unwrap();
    writer.commit_batch().await.unwrap();
    let latest = writer.manifest_version();
    let advisory = read_advisory(writer.manifest_store()).await;
    assert_eq!(
        advisory,
        latest - 1,
        "the previous commit's advisory landed"
    );

    // A flush settles the last one even when nothing else is written.
    writer.flush(Schema::empty()).await.unwrap();
    assert!(!writer.manifest_store().has_pending_advisory());
    let advisory = read_advisory(writer.manifest_store()).await;
    assert_eq!(advisory, writer.manifest_version());
}

#[tokio::test]
async fn a_lagging_advisory_still_resolves_the_current_version() {
    let counting = CountingStore::new();
    let store: Arc<dyn ObjectStore> = counting.clone();
    let ns = paths("advisory-lag");
    let mut writer = WriterSession::open(store.clone(), ns.clone())
        .await
        .unwrap();
    for i in 0..3 {
        writer
            .upsert_node("Doc", NodeId::new(), &doc(&format!("{i}")))
            .unwrap();
        writer.commit_batch().await.unwrap();
    }
    let version = writer.manifest_version();
    // Crash with the last advisory still owed.
    assert!(writer.manifest_store().has_pending_advisory());
    drop(writer);
    let advisory = read_advisory(&ManifestStore::new(store.clone(), ns.clone())).await;
    assert!(advisory < version);

    // A fresh reader resolves the true current version...
    let loaded = ManifestStore::new(store.clone(), ns.clone())
        .load_current()
        .await
        .unwrap();
    assert_eq!(loaded.manifest.version, version);
    // ...and a reopened writer sees every committed row.
    let reopened = WriterSession::open(store, ns).await.unwrap();
    assert!(reopened.manifest_version() > version);
    let snapshot = reopened.snapshot();
    let rows = snapshot.scan_label("Doc").await.unwrap();
    assert_eq!(rows.len(), 3);
}

async fn read_advisory(manifests: &ManifestStore) -> u64 {
    let body = manifests
        .store()
        .get(&manifests.paths().current_pointer())
        .await
        .unwrap()
        .bytes()
        .await
        .unwrap();
    serde_json::from_slice::<ManifestPointer>(&body)
        .unwrap()
        .version
}
