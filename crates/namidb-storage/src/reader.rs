//! Read-only access to a namespace that another process writes.
//!
//! [`crate::WriterSession::open`] always claims the writer epoch, which fences
//! whatever process was writing before. A [`ReaderSession`] never claims: it
//! loads the current manifest, replays the WAL segments that manifest
//! references (so commits that were never flushed are visible), and hands out
//! an [`OwnedSnapshot`] of that version. [`ReaderSession::refresh`] moves it
//! to the newest committed version.
//!
//! Another process's janitor cannot see this reader's in-process snapshot, so
//! the session holds a [`RetentionPin`] lease on the version it serves. The
//! pin is best effort: a reader on read-only credentials cannot write the
//! lease, and then only the sweep's `min_age` window protects its objects.
//! The lease expires unless something renews it: [`ReaderSession::keep_alive`]
//! does on every read, and a host whose reads can outlast the TTL calls
//! [`ReaderSession::renew_pins`] on a timer. A refresh keeps the old
//! version's pin until no read holds its snapshot. A session that is never
//! refreshed moves itself to the newest version once its own is
//! [`MAX_VERSION_AGE`] old, or once its lease may have lapsed (the files of a
//! version nobody pinned can be gone), so an idle reader cannot hold back the
//! writer's cleanup indefinitely.
//!
//! Writer-local caches (node views, adjacency, property-index generations,
//! exact counts) are left off. They are keyed by the writer's own
//! generations, and a reader that refreshes would serve stale rows from
//! them. The SST body cache is keyed by immutable object path and is safe to
//! share.

use std::sync::Arc;
use std::time::{Duration, Instant};

use object_store::{ObjectStore, ObjectStoreExt};
use tracing::{debug, warn};

use crate::cache::SstCache;
use crate::error::{Error, Result};
use crate::manifest::{LoadedManifest, ManifestStore};
use crate::paths::NamespacePaths;
use crate::pin::{RetentionPin, DEFAULT_PIN_TTL};
use crate::read::{OwnedSnapshot, RangedMode, DEFAULT_RANGED_THRESHOLD_BYTES};
use crate::recovery::recover_memtable_with_snapshot;
use crate::wal::WalStore;

/// How many times a load re-reads the manifest when an object it references
/// vanished mid-load (a concurrent janitor reclaimed a version that was
/// already superseded). Each retry starts from the newer current version.
const LOAD_ATTEMPTS: usize = 3;

/// How long a session serves one version before [`ReaderSession::keep_alive`]
/// refreshes on its own. Bounds how far a forgotten reader's pin holds back
/// the writer's cleanup.
pub const MAX_VERSION_AGE: Duration = Duration::from_secs(60 * 60);

/// A non-claiming, refreshable view of a namespace. See the module docs.
#[derive(Debug)]
pub struct ReaderSession {
    manifest_store: ManifestStore,
    wal_store: WalStore,
    cache: Option<SstCache>,
    current: Arc<OwnedSnapshot>,
    pin: Option<RetentionPin>,
    /// Versions a refresh moved past while a read still held their snapshot.
    /// Each pin is released once its snapshot has no other holder.
    retired: Vec<(Arc<OwnedSnapshot>, RetentionPin)>,
    /// When `current` was last confirmed to be the newest version.
    checked_at: Instant,
    /// [`Self::release`] dropped the leases; the next read reloads with a
    /// fresh one rather than serve files nothing protects.
    released: bool,
}

impl ReaderSession {
    /// Open the namespace for reading. Fails with the object store's
    /// `NotFound` when the namespace was never bootstrapped: a reader never
    /// creates one.
    pub async fn open(
        store: Arc<dyn ObjectStore>,
        paths: NamespacePaths,
        cache: Option<SstCache>,
    ) -> Result<Self> {
        let manifest_store = ManifestStore::new(store.clone(), paths.clone());
        let wal_store = WalStore::new(store, paths);
        // A missing pointer means the namespace does not exist; that is not a
        // race, so it is never retried.
        let head = manifest_store.load_current().await?;
        let (current, pin) = load(&manifest_store, &wal_store, &cache, head).await?;
        Ok(Self {
            manifest_store,
            wal_store,
            cache,
            current,
            pin,
            retired: Vec::new(),
            checked_at: Instant::now(),
            released: false,
        })
    }

    /// The snapshot this session currently serves. Cheap: an `Arc` clone.
    pub fn snapshot(&self) -> Arc<OwnedSnapshot> {
        Arc::clone(&self.current)
    }

    pub fn manifest_version(&self) -> u64 {
        self.current.manifest_version()
    }

    /// Move to the newest committed version. Returns `false` when the
    /// namespace has not advanced since the last load; the retention pins
    /// are renewed either way. Replays the WAL tail the new manifest
    /// references, so the cost is proportional to the unflushed writes.
    /// After [`Self::release`] it reloads even when nothing advanced, to
    /// take a new lease.
    ///
    /// Cancel-safe: dropped part way, it leaves the session on the version
    /// it had and releases a lease the load already took.
    pub async fn refresh(&mut self) -> Result<bool> {
        let head = self.manifest_store.load_current().await?;
        let advanced = head.manifest.version != self.manifest_version();
        if advanced || self.released {
            let (current, pin) =
                load(&self.manifest_store, &self.wal_store, &self.cache, head).await?;
            let old = std::mem::replace(&mut self.current, current);
            if let Some(old_pin) = std::mem::replace(&mut self.pin, pin) {
                self.retired.push((old, old_pin));
            }
            self.released = false;
        }
        self.checked_at = Instant::now();
        self.renew_pins().await;
        Ok(advanced)
    }

    /// Call once per read. Renews the leases that are due and releases the
    /// ones no read needs any more. Once the served version is
    /// [`MAX_VERSION_AGE`] old, or its lease may have lapsed or was
    /// released, it refreshes instead; a failed refresh is logged and the
    /// session keeps serving the version it has. Cancel-safe, like
    /// [`Self::refresh`] and [`Self::renew_pins`].
    pub async fn keep_alive(&mut self) {
        let lapsed = self.pin.as_ref().is_some_and(RetentionPin::lapsed);
        if self.released || lapsed || self.checked_at.elapsed() >= MAX_VERSION_AGE {
            match self.refresh().await {
                Ok(_) => return,
                Err(e) => warn!(error = %e, "reader could not refresh a stale version"),
            }
        }
        self.renew_pins().await;
    }

    /// Delete every retention lease this session holds, including those of
    /// snapshots a read may still be using. The session stays usable: the
    /// next [`Self::keep_alive`] or [`Self::refresh`] reloads the newest
    /// version under a new lease. Dropping the session without calling this
    /// leaves the leases to expire on their own.
    pub async fn release(&mut self) -> Result<()> {
        self.released = true;
        let retired = std::mem::take(&mut self.retired)
            .into_iter()
            .map(|(_, pin)| pin);
        let mut result = Ok(());
        for pin in self.pin.take().into_iter().chain(retired) {
            if let Err(e) = pin.release().await {
                result = Err(e);
            }
        }
        result
    }

    /// Renew the leases that are due without refreshing, for a timer that
    /// keeps a long read covered between reads. Once the served version is
    /// [`MAX_VERSION_AGE`] old and no read holds it, its lease is left to
    /// lapse: an idle session then stops holding back cleanup, and the next
    /// [`Self::keep_alive`] refreshes.
    ///
    /// Cancel-safe: every pin stays in the session until it is released, so
    /// a read that stops waiting part way cannot drop a lease another read
    /// still needs renewed.
    pub async fn renew_pins(&mut self) {
        let in_use = Arc::strong_count(&self.current) > 1;
        if let Some(pin) = &mut self.pin {
            if in_use || self.checked_at.elapsed() < MAX_VERSION_AGE {
                renew_pin(pin).await;
            }
        }
        let mut i = 0;
        while i < self.retired.len() {
            // This session holds one reference; any other is a read in flight.
            if Arc::strong_count(&self.retired[i].0) == 1 {
                // Cut short, the release leaves a lease nothing reads to
                // expire on its own.
                let (_, pin) = self.retired.swap_remove(i);
                release_pin(pin).await;
            } else {
                renew_pin(&mut self.retired[i].1).await;
                i += 1;
            }
        }
    }
}

async fn load(
    manifest_store: &ManifestStore,
    wal_store: &WalStore,
    cache: &Option<SstCache>,
    mut current: LoadedManifest,
) -> Result<(Arc<OwnedSnapshot>, Option<RetentionPin>)> {
    let mut attempt = 1;
    loop {
        let version = current.manifest.version;
        match load_version(manifest_store, wal_store, cache, current).await {
            Err(Error::ObjectStore(object_store::Error::NotFound { .. }))
                if attempt < LOAD_ATTEMPTS =>
            {
                debug!(version, attempt, "reader load raced a sweep; reloading");
                attempt += 1;
                current = manifest_store.load_current().await?;
            }
            other => return other,
        }
    }
}

async fn load_version(
    manifest_store: &ManifestStore,
    wal_store: &WalStore,
    cache: &Option<SstCache>,
    current: LoadedManifest,
) -> Result<(Arc<OwnedSnapshot>, Option<RetentionPin>)> {
    let store = manifest_store.store().clone();
    let paths = manifest_store.paths();
    let version = current.manifest.version;
    let mut pin = ReleaseOnDrop(
        match RetentionPin::acquire(store.clone(), paths, version, DEFAULT_PIN_TTL).await {
            Ok(pin) => Some(pin),
            Err(e) => {
                warn!(error = %e, version, "reader could not write a retention pin");
                None
            }
        },
    );
    // Close the load-then-pin race the same way a backup does: once the body
    // is confirmed present after the lease exists, every later sweep keeps
    // this version's closure.
    let recovered = async {
        store.head(&paths.manifest_version(version)).await?;
        recover_memtable_with_snapshot(&current.manifest, wal_store, Some(&store)).await
    }
    .await;
    let recovered = match recovered {
        Ok(r) => r,
        Err(e) => {
            if let Some(pin) = pin.0.take() {
                release_pin(pin).await;
            }
            return Err(e);
        }
    };
    let snapshot = OwnedSnapshot {
        manifest: current,
        memtable: Arc::new(recovered.memtable.snapshot_view()),
        store,
        paths: paths.clone(),
        cache: cache.clone(),
        ranged_mode: RangedMode::Auto,
        ranged_threshold_bytes: DEFAULT_RANGED_THRESHOLD_BYTES,
        adjacency_cache: None,
        shared_node_cache: None,
        property_index_cache: None,
        property_index_generation: None,
        exact_node_counts: None,
        memtable_claimant_cell: None,
    };
    Ok((Arc::new(snapshot), pin.0.take()))
}

/// The pin of a load in progress. A load dropped part way (a read that
/// timed out during a refresh) releases it in the background instead of
/// leaving a lease nothing renews to hold back cleanup until it expires.
struct ReleaseOnDrop(Option<RetentionPin>);

impl Drop for ReleaseOnDrop {
    fn drop(&mut self) {
        if let Some(pin) = self.0.take() {
            if let Ok(runtime) = tokio::runtime::Handle::try_current() {
                runtime.spawn(release_pin(pin));
            }
        }
    }
}

async fn renew_pin(pin: &mut RetentionPin) {
    if let Err(e) = pin.renew_if_due().await {
        warn!(error = %e, "failed to renew the reader retention pin");
    }
}

async fn release_pin(pin: RetentionPin) {
    if let Err(e) = pin.release().await {
        // Benign: the lease expires on its own and the sweep reclaims it.
        debug!(error = %e, "failed to release the reader retention pin");
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::sync::Mutex;

    use futures::TryStreamExt;
    use namidb_core::{NamespaceId, NodeId, Value};
    use object_store::memory::InMemory;
    use object_store::path::Path;
    use object_store::{
        CopyOptions, GetOptions, GetResult, ListResult, ObjectMeta, PutMultipartOptions,
        PutOptions, PutPayload, PutResult,
    };
    use uuid::Uuid;

    use super::*;
    use crate::flush::NodeWriteRecord;
    use crate::ingest::WriterSession;

    fn paths(name: &str) -> NamespacePaths {
        NamespacePaths::new("tenants", NamespaceId::new(name).unwrap())
    }

    fn node(name: &str) -> NodeWriteRecord {
        let mut properties = BTreeMap::new();
        properties.insert("name".to_string(), Value::Str(name.to_string()));
        NodeWriteRecord {
            properties,
            schema_version: 1,
            ..Default::default()
        }
    }

    async fn pins(store: &Arc<dyn ObjectStore>, paths: &NamespacePaths) -> usize {
        store
            .list(Some(&paths.pins_dir()))
            .try_collect::<Vec<_>>()
            .await
            .unwrap()
            .len()
    }

    /// Delegating store that can fail one `head` of a chosen path with
    /// `NotFound` (a sweep racing the load), refuse every pin lease PUT
    /// (read-only credentials), and hang every PUT or `head` whose path
    /// contains `stall` (a slow request a timeout gives up on).
    #[derive(Debug)]
    struct FaultyStore {
        inner: Arc<dyn ObjectStore>,
        vanish_once: Mutex<Option<Path>>,
        deny_pins: bool,
        stall: Mutex<Option<String>>,
    }

    impl FaultyStore {
        fn wrap(inner: Arc<dyn ObjectStore>) -> Arc<Self> {
            Arc::new(Self {
                inner,
                vanish_once: Mutex::new(None),
                deny_pins: false,
                stall: Mutex::new(None),
            })
        }

        async fn maybe_stall(&self, location: &Path) {
            let stalled = self
                .stall
                .lock()
                .unwrap()
                .as_ref()
                .is_some_and(|part| location.as_ref().contains(part.as_str()));
            if stalled {
                std::future::pending::<()>().await;
            }
        }
    }

    impl std::fmt::Display for FaultyStore {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "FaultyStore({})", self.inner)
        }
    }

    #[async_trait::async_trait]
    impl ObjectStore for FaultyStore {
        async fn put_opts(
            &self,
            location: &Path,
            payload: PutPayload,
            opts: PutOptions,
        ) -> object_store::Result<PutResult> {
            if self.deny_pins && location.as_ref().contains("/pins/") {
                return Err(object_store::Error::PermissionDenied {
                    path: location.to_string(),
                    source: "read-only credentials".into(),
                });
            }
            self.maybe_stall(location).await;
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
            let vanished = {
                let mut vanish = self.vanish_once.lock().unwrap();
                options.head && vanish.take_if(|p| p == location).is_some()
            };
            if vanished {
                return Err(object_store::Error::NotFound {
                    path: location.to_string(),
                    source: "swept".into(),
                });
            }
            if options.head {
                self.maybe_stall(location).await;
            }
            self.inner.get_opts(location, options).await
        }
        fn list(
            &self,
            prefix: Option<&Path>,
        ) -> futures::stream::BoxStream<'static, object_store::Result<ObjectMeta>> {
            self.inner.list(prefix)
        }
        async fn list_with_delimiter(
            &self,
            prefix: Option<&Path>,
        ) -> object_store::Result<ListResult> {
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

    #[tokio::test]
    async fn reader_sees_wal_resident_commits_without_fencing_the_writer() {
        let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let paths = paths("reader-open");
        let mut writer = WriterSession::open(store.clone(), paths.clone())
            .await
            .unwrap();
        let alice = NodeId::from_uuid(Uuid::now_v7());
        writer.upsert_node("Person", alice, &node("alice")).unwrap();
        writer.commit_batch().await.unwrap();
        let epoch_before = writer.snapshot().manifest().manifest.epoch;

        // The commit lives only in the WAL: nothing was flushed.
        let reader = ReaderSession::open(store.clone(), paths.clone(), None)
            .await
            .unwrap();
        let snap = reader.snapshot();
        let view = snap.borrow().lookup_node("Person", alice).await.unwrap();
        assert_eq!(
            view.unwrap().properties.get("name"),
            Some(&Value::Str("alice".into()))
        );

        // The writer still owns the namespace and can keep committing.
        let bob = NodeId::from_uuid(Uuid::now_v7());
        writer.upsert_node("Person", bob, &node("bob")).unwrap();
        writer.commit_batch().await.unwrap();
        assert_eq!(writer.snapshot().manifest().manifest.epoch, epoch_before);
        assert_eq!(
            ManifestStore::new(store.clone(), paths.clone())
                .load_current()
                .await
                .unwrap()
                .manifest
                .epoch,
            epoch_before,
            "opening a reader must not claim the epoch"
        );
    }

    #[tokio::test]
    async fn refresh_moves_to_the_newest_commit_and_its_pin() {
        let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let paths = paths("reader-refresh");
        let mut writer = WriterSession::open(store.clone(), paths.clone())
            .await
            .unwrap();
        let mut reader = ReaderSession::open(store.clone(), paths.clone(), None)
            .await
            .unwrap();
        assert_eq!(pins(&store, &paths).await, 1);
        assert!(!reader.refresh().await.unwrap(), "nothing new to load");

        let alice = NodeId::from_uuid(Uuid::now_v7());
        writer.upsert_node("Person", alice, &node("alice")).unwrap();
        writer.commit_batch().await.unwrap();

        let before = reader.snapshot();
        assert!(before
            .borrow()
            .lookup_node("Person", alice)
            .await
            .unwrap()
            .is_none());
        assert!(reader.refresh().await.unwrap());
        assert_eq!(reader.manifest_version(), writer.manifest_version());
        assert!(reader
            .snapshot()
            .borrow()
            .lookup_node("Person", alice)
            .await
            .unwrap()
            .is_some());
        // The old snapshot is still a consistent view of its own version.
        assert!(before
            .borrow()
            .lookup_node("Person", alice)
            .await
            .unwrap()
            .is_none());

        // The pin moves with the version once no read holds the old one.
        drop(before);
        reader.keep_alive().await;
        assert_eq!(pins(&store, &paths).await, 1);
        reader.release().await.unwrap();
        assert_eq!(pins(&store, &paths).await, 0);
    }

    #[tokio::test]
    async fn refresh_keeps_the_old_pin_while_a_read_holds_its_snapshot() {
        let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let paths = paths("reader-retired");
        let mut writer = WriterSession::open(store.clone(), paths.clone())
            .await
            .unwrap();
        let mut reader = ReaderSession::open(store.clone(), paths.clone(), None)
            .await
            .unwrap();
        let in_flight = reader.snapshot();

        writer
            .upsert_node("Person", NodeId::from_uuid(Uuid::now_v7()), &node("a"))
            .unwrap();
        writer.commit_batch().await.unwrap();
        assert!(reader.refresh().await.unwrap());
        assert_eq!(
            pins(&store, &paths).await,
            2,
            "the old version stays pinned while a read uses it"
        );

        drop(in_flight);
        reader.keep_alive().await;
        assert_eq!(pins(&store, &paths).await, 1);
        reader.release().await.unwrap();
        assert_eq!(pins(&store, &paths).await, 0);
    }

    #[tokio::test]
    async fn keep_alive_refreshes_a_version_past_its_max_age() {
        let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let paths = paths("reader-stale");
        let mut writer = WriterSession::open(store.clone(), paths.clone())
            .await
            .unwrap();
        let mut reader = ReaderSession::open(store.clone(), paths.clone(), None)
            .await
            .unwrap();
        writer
            .upsert_node("Person", NodeId::from_uuid(Uuid::now_v7()), &node("a"))
            .unwrap();
        writer.commit_batch().await.unwrap();

        reader.keep_alive().await;
        assert_ne!(reader.manifest_version(), writer.manifest_version());
        reader.checked_at = Instant::now().checked_sub(MAX_VERSION_AGE).unwrap();
        reader.keep_alive().await;
        assert_eq!(reader.manifest_version(), writer.manifest_version());
        assert_eq!(pins(&store, &paths).await, 1);
    }

    /// A writer with two flushed L0 SSTs, and the paths of those SSTs.
    async fn writer_with_two_l0_ssts(
        store: &Arc<dyn ObjectStore>,
        paths: &NamespacePaths,
    ) -> (WriterSession, Vec<Path>) {
        let mut writer = WriterSession::open(store.clone(), paths.clone())
            .await
            .unwrap();
        for name in ["a", "b"] {
            writer
                .upsert_node("Person", NodeId::from_uuid(Uuid::now_v7()), &node(name))
                .unwrap();
            writer.commit_batch().await.unwrap();
            let schema = writer.snapshot().manifest().manifest.schema.clone();
            writer.flush(schema).await.unwrap();
        }
        let l0 = writer
            .snapshot()
            .manifest()
            .manifest
            .ssts
            .iter()
            .map(|s| Path::from(format!("{}/{}", paths.namespace_prefix().as_ref(), s.path)))
            .collect::<Vec<_>>();
        assert_eq!(l0.len(), 2);
        (writer, l0)
    }

    async fn compact(writer: &mut WriterSession) {
        let schema = writer.snapshot().manifest().manifest.schema.clone();
        let out = writer.compact_l0(&schema).await.unwrap();
        assert_eq!(out.source_ssts_removed, 2);
    }

    /// Sweep with no age margin, as a writer's janitor would once `min_age`
    /// has passed.
    async fn sweep(store: &Arc<dyn ObjectStore>, paths: &NamespacePaths) {
        let ms = ManifestStore::new(store.clone(), paths.clone());
        crate::sweep_orphans(&ms, u64::MAX, Duration::ZERO, 4, true)
            .await
            .unwrap();
    }

    async fn scan_people(reader: &ReaderSession) -> Result<usize> {
        Ok(reader.snapshot().borrow().scan_label("Person").await?.len())
    }

    #[tokio::test]
    async fn reader_pin_holds_the_sweep_horizon() {
        let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let paths = paths("reader-horizon");
        let (mut writer, l0) = writer_with_two_l0_ssts(&store, &paths).await;
        let mut reader = ReaderSession::open(store.clone(), paths.clone(), None)
            .await
            .unwrap();

        compact(&mut writer).await;
        sweep(&store, &paths).await;
        assert_eq!(scan_people(&reader).await.unwrap(), 2);

        reader.release().await.unwrap();
        sweep(&store, &paths).await;
        for path in &l0 {
            assert!(store.head(path).await.is_err(), "{path} outlived the lease");
        }
    }

    #[tokio::test]
    async fn keep_alive_refreshes_once_the_lease_may_have_lapsed() {
        let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let paths = paths("reader-lapsed");
        let (mut writer, _) = writer_with_two_l0_ssts(&store, &paths).await;
        let mut reader = ReaderSession::open(store.clone(), paths.clone(), None)
            .await
            .unwrap();

        // Nothing renewed the lease for a TTL, and a sweep dropped it and
        // the files only the reader's version needed.
        let pin = reader.pin.as_mut().unwrap();
        pin.expires_at_unix = 0;
        let expired = crate::PinLease {
            version: pin.version(),
            expires_at_unix: 0,
        };
        store
            .put(pin.path(), serde_json::to_vec(&expired).unwrap().into())
            .await
            .unwrap();
        compact(&mut writer).await;
        sweep(&store, &paths).await;
        assert!(
            scan_people(&reader).await.is_err(),
            "the sweep kept the files"
        );

        reader.keep_alive().await;
        assert_eq!(reader.manifest_version(), writer.manifest_version());
        assert_eq!(scan_people(&reader).await.unwrap(), 2);
    }

    #[tokio::test]
    async fn renew_pins_lets_an_idle_stale_version_lapse() {
        let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let paths = paths("reader-renew");
        WriterSession::open(store.clone(), paths.clone())
            .await
            .unwrap();
        let mut reader = ReaderSession::open(store.clone(), paths.clone(), None)
            .await
            .unwrap();
        let expiry = |store: Arc<dyn ObjectStore>, path: Path| async move {
            let body = store.get(&path).await.unwrap().bytes().await.unwrap();
            serde_json::from_slice::<crate::PinLease>(&body)
                .unwrap()
                .expires_at_unix
        };
        let pin_path = reader.pin.as_ref().unwrap().path().clone();
        let written = expiry(store.clone(), pin_path.clone()).await;

        // A read still holds the stale version: its lease is renewed.
        reader.checked_at = Instant::now().checked_sub(MAX_VERSION_AGE).unwrap();
        let in_flight = reader.snapshot();
        tokio::time::sleep(Duration::from_millis(1100)).await;
        reader.pin.as_mut().unwrap().expires_at_unix = crate::pin::unix_now();
        reader.renew_pins().await;
        let renewed = expiry(store.clone(), pin_path.clone()).await;
        assert!(renewed > written);

        // Nothing holds it any more: the lease is left to expire.
        drop(in_flight);
        tokio::time::sleep(Duration::from_millis(1100)).await;
        reader.pin.as_mut().unwrap().expires_at_unix = crate::pin::unix_now();
        reader.renew_pins().await;
        assert_eq!(expiry(store.clone(), pin_path).await, renewed);
    }

    #[tokio::test]
    async fn open_retries_when_the_loaded_version_vanishes() {
        let inner: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let paths = paths("reader-race");
        let writer = WriterSession::open(inner.clone(), paths.clone())
            .await
            .unwrap();
        let faulty = Arc::new(FaultyStore {
            inner,
            vanish_once: Mutex::new(Some(paths.manifest_version(writer.manifest_version()))),
            deny_pins: false,
            stall: Mutex::new(None),
        });
        let store: Arc<dyn ObjectStore> = faulty.clone();
        let reader = ReaderSession::open(store.clone(), paths.clone(), None)
            .await
            .unwrap();
        assert!(
            faulty.vanish_once.lock().unwrap().is_none(),
            "fault never fired"
        );
        assert_eq!(reader.manifest_version(), writer.manifest_version());
        // The failed attempt released its pin; only the retry's is left.
        assert_eq!(pins(&store, &paths).await, 1);
    }

    #[tokio::test]
    async fn open_without_permission_to_pin_still_serves_reads() {
        let inner: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let paths = paths("reader-no-pin");
        let mut writer = WriterSession::open(inner.clone(), paths.clone())
            .await
            .unwrap();
        let alice = NodeId::from_uuid(Uuid::now_v7());
        writer.upsert_node("Person", alice, &node("alice")).unwrap();
        writer.commit_batch().await.unwrap();
        let store: Arc<dyn ObjectStore> = Arc::new(FaultyStore {
            inner,
            vanish_once: Mutex::new(None),
            deny_pins: true,
            stall: Mutex::new(None),
        });
        let mut reader = ReaderSession::open(store.clone(), paths.clone(), None)
            .await
            .unwrap();
        assert!(reader.pin.is_none());
        assert!(reader
            .snapshot()
            .borrow()
            .lookup_node("Person", alice)
            .await
            .unwrap()
            .is_some());
        assert_eq!(pins(&store, &paths).await, 0);
        reader.release().await.unwrap();
    }

    #[tokio::test]
    async fn reader_never_bootstraps_a_missing_namespace() {
        let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let paths = paths("reader-missing");
        let err = ReaderSession::open(store.clone(), paths.clone(), None)
            .await
            .unwrap_err();
        assert!(
            matches!(
                err,
                Error::ObjectStore(object_store::Error::NotFound { .. })
            ),
            "{err:?}"
        );
        assert_eq!(pins(&store, &paths).await, 0);
    }

    #[tokio::test]
    async fn renew_pins_cut_short_keeps_the_leases_reads_still_need() {
        let inner: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let paths = paths("reader-renew-cancel");
        let mut writer = WriterSession::open(inner.clone(), paths.clone())
            .await
            .unwrap();
        let faulty = FaultyStore::wrap(inner);
        let mut reader = ReaderSession::open(faulty.clone(), paths.clone(), None)
            .await
            .unwrap();
        let in_flight = reader.snapshot();
        writer
            .upsert_node("Person", NodeId::from_uuid(Uuid::now_v7()), &node("a"))
            .unwrap();
        writer.commit_batch().await.unwrap();
        assert!(reader.refresh().await.unwrap());
        assert_eq!(reader.retired.len(), 1);

        // The retired lease is due, and its renewal never answers: a read
        // with a timeout gives up while it waits.
        reader.retired[0].1.expires_at_unix = crate::pin::unix_now();
        *faulty.stall.lock().unwrap() = Some("/pins/".into());
        let renew = tokio::time::timeout(Duration::from_millis(50), reader.renew_pins());
        assert!(renew.await.is_err());
        *faulty.stall.lock().unwrap() = None;

        assert_eq!(reader.retired.len(), 1, "the in-flight read lost its lease");
        reader.renew_pins().await;
        assert!(!reader.retired[0].1.lapsed());
        drop(in_flight);
        reader.release().await.unwrap();
    }

    #[tokio::test]
    async fn refresh_cut_short_releases_the_lease_it_took() {
        let inner: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let paths = paths("reader-refresh-cancel");
        let mut writer = WriterSession::open(inner.clone(), paths.clone())
            .await
            .unwrap();
        let faulty = FaultyStore::wrap(inner);
        let store: Arc<dyn ObjectStore> = faulty.clone();
        let mut reader = ReaderSession::open(store.clone(), paths.clone(), None)
            .await
            .unwrap();
        let served = reader.manifest_version();
        writer
            .upsert_node("Person", NodeId::from_uuid(Uuid::now_v7()), &node("a"))
            .unwrap();
        writer.commit_batch().await.unwrap();

        // The load takes its lease, then hangs confirming the new version.
        let next = paths.manifest_version(writer.manifest_version());
        *faulty.stall.lock().unwrap() = Some(next.as_ref().to_string());
        let refresh = tokio::time::timeout(Duration::from_millis(50), reader.refresh());
        assert!(refresh.await.is_err());
        *faulty.stall.lock().unwrap() = None;
        assert_eq!(reader.manifest_version(), served);
        // The dropped load releases its lease in the background.
        for _ in 0..100 {
            if pins(&store, &paths).await == 1 {
                break;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        assert_eq!(
            pins(&store, &paths).await,
            1,
            "the cut-short load leaked a lease"
        );

        assert!(reader.refresh().await.unwrap());
        assert_eq!(pins(&store, &paths).await, 1);
    }

    #[tokio::test]
    async fn a_released_session_takes_a_new_lease_on_its_next_read() {
        let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let paths = paths("reader-reuse");
        WriterSession::open(store.clone(), paths.clone())
            .await
            .unwrap();
        let mut reader = ReaderSession::open(store.clone(), paths.clone(), None)
            .await
            .unwrap();
        reader.release().await.unwrap();
        assert_eq!(pins(&store, &paths).await, 0);

        reader.keep_alive().await;
        assert!(reader.pin.is_some());
        assert_eq!(pins(&store, &paths).await, 1);
        reader.release().await.unwrap();
    }
}
