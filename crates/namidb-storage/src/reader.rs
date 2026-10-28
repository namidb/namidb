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
//!
//! Writer-local caches (node views, adjacency, property-index generations,
//! exact counts) are left off. They are keyed by the writer's own
//! generations, and a reader that refreshes would serve stale rows from
//! them. The SST body cache is keyed by immutable object path and is safe to
//! share.

use std::sync::Arc;

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

/// A non-claiming, refreshable view of a namespace. See the module docs.
#[derive(Debug)]
pub struct ReaderSession {
    manifest_store: ManifestStore,
    wal_store: WalStore,
    cache: Option<SstCache>,
    current: Arc<OwnedSnapshot>,
    pin: Option<RetentionPin>,
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
    /// namespace has not advanced since the last load; the retention pin is
    /// renewed either way. Replays the WAL tail the new manifest references,
    /// so the cost is proportional to the unflushed writes.
    pub async fn refresh(&mut self) -> Result<bool> {
        let head = self.manifest_store.load_current().await?;
        if head.manifest.version == self.manifest_version() {
            self.renew_pin().await;
            return Ok(false);
        }
        let (current, pin) = load(&self.manifest_store, &self.wal_store, &self.cache, head).await?;
        self.current = current;
        if let Some(old) = std::mem::replace(&mut self.pin, pin) {
            release_pin(old).await;
        }
        Ok(true)
    }

    /// Re-write the retention lease if half its TTL has elapsed. Cheap when
    /// nothing is due; callers on a long-lived read path can call it per
    /// query so the pin outlives long gaps between refreshes.
    pub async fn renew_pin(&mut self) {
        if let Some(pin) = &mut self.pin {
            if let Err(e) = pin.renew_if_due().await {
                warn!(error = %e, "failed to renew the reader retention pin");
            }
        }
    }

    /// Delete the retention lease. Dropping the session without calling
    /// this leaves the lease to expire on its own.
    pub async fn release(self) -> Result<()> {
        match self.pin {
            Some(pin) => pin.release().await,
            None => Ok(()),
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
    let pin = match RetentionPin::acquire(store.clone(), paths, version, DEFAULT_PIN_TTL).await {
        Ok(pin) => Some(pin),
        Err(e) => {
            warn!(error = %e, version, "reader could not write a retention pin");
            None
        }
    };
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
            if let Some(pin) = pin {
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
    Ok((Arc::new(snapshot), pin))
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

    use futures::TryStreamExt;
    use namidb_core::{NamespaceId, NodeId, Value};
    use object_store::memory::InMemory;
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

        // The pin moved with the version instead of piling up.
        assert_eq!(pins(&store, &paths).await, 1);
        reader.release().await.unwrap();
        assert_eq!(pins(&store, &paths).await, 0);
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
}
