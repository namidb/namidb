//! Local-filesystem [`ObjectStore`] with manifest-CAS support.
//!
//! `object_store::local::LocalFileSystem` does not implement
//! [`PutMode::Update`] — it returns `NotImplemented` — so it cannot
//! drive the NamiDB manifest commit protocol on its own. This wrapper
//! plugs the gap:
//!
//! - [`PutMode::Create`] is delegated as-is. `LocalFileSystem` already
//!   uses `O_CREAT|O_EXCL` semantics, matching the manifest
//!   write-once invariant.
//! - [`PutMode::Update`] is intercepted. Under an advisory file lock
//!   (`<root>/.namidb/cas.lock`, `flock(LOCK_EX)`), we re-`head` the
//!   target, compare its e-tag against the caller-supplied
//!   [`UpdateVersion`], and — only if it matches — overwrite the file
//!   via the inner `LocalFileSystem` (which performs a tmp+rename
//!   atomic publish). A mismatched e-tag returns
//!   [`object_store::Error::Precondition`], identical to the S3 /
//!   `If-Match` failure mode.
//!
//! Lock scope is namespace-coarse on purpose: only the manifest pointer
//! takes this path, so contention is bounded to commit fan-in. Reads,
//! SST writes, and WAL writes never block on it.
//!
//! ## Listing
//!
//! `list` and `list_with_delimiter` filter out entries under
//! `.namidb/` so callers never see the lock file. Every other method
//! delegates verbatim to the inner store.

use std::fs::OpenOptions;
use std::path::PathBuf;
use std::sync::Arc;

use async_trait::async_trait;
use fs2::FileExt;
use futures::stream::{BoxStream, StreamExt};
use object_store::local::LocalFileSystem;
use object_store::path::Path;
use object_store::{
    CopyOptions, GetOptions, GetResult, ListResult, MultipartUpload, ObjectMeta, ObjectStore,
    ObjectStoreExt, PutMode, PutMultipartOptions, PutOptions, PutPayload, PutResult,
    Result as OsResult, UpdateVersion, UploadPart,
};

/// Filesystem-backed `ObjectStore` with the conditional-write semantics
/// the NamiDB manifest CAS protocol needs.
///
/// Construct with [`LocalFileObjectStore::new`] pointing at an
/// existing or yet-to-exist directory. The directory becomes the
/// store's `root`; every object key is rendered as a file under it.
pub struct LocalFileObjectStore {
    inner: Arc<LocalFileSystem>,
    root: PathBuf,
}

impl std::fmt::Debug for LocalFileObjectStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LocalFileObjectStore")
            .field("root", &self.root)
            .finish()
    }
}

impl std::fmt::Display for LocalFileObjectStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "LocalFileObjectStore({})", self.root.display())
    }
}

/// Path segment used to hide internal control files from `list`.
const INTERNAL_DIR: &str = ".namidb";

impl LocalFileObjectStore {
    /// Open (or create) a local-filesystem object store rooted at
    /// `root`. Creates `root` and the internal control directory
    /// `<root>/.namidb/` if they do not already exist.
    pub fn new<P: Into<PathBuf>>(root: P) -> OsResult<Self> {
        let root = root.into();
        std::fs::create_dir_all(&root).map_err(|e| object_store::Error::Generic {
            store: "LocalFileObjectStore",
            source: Box::new(e),
        })?;
        std::fs::create_dir_all(root.join(INTERNAL_DIR)).map_err(|e| {
            object_store::Error::Generic {
                store: "LocalFileObjectStore",
                source: Box::new(e),
            }
        })?;
        let inner =
            LocalFileSystem::new_with_prefix(&root).map_err(|e| object_store::Error::Generic {
                store: "LocalFileObjectStore",
                source: Box::new(e),
            })?;
        Ok(Self {
            inner: Arc::new(inner),
            root,
        })
    }

    /// Root directory on disk. Useful for tests and operational tooling.
    pub fn root(&self) -> &std::path::Path {
        &self.root
    }

    /// Map an object-store key to its on-disk path under `root`. Relies on
    /// the 1:1 key->path layout `LocalFileSystem` uses for the safe ASCII
    /// keys NamiDB writes (digits, `-`, `_`, `.`, `/`); it does not attempt
    /// the percent-encoding `LocalFileSystem` applies to exotic keys.
    fn fs_path(&self, location: &Path) -> PathBuf {
        self.root.join(location.as_ref())
    }

    /// Acquire the namespace-wide CAS lock for the duration of the
    /// returned guard. Blocking `flock` is offloaded to
    /// `spawn_blocking` so we do not stall the runtime when contention
    /// is high.
    async fn cas_guard(&self) -> OsResult<CasGuard> {
        let path = self.root.join(INTERNAL_DIR).join("cas.lock");
        tokio::task::spawn_blocking(move || -> std::io::Result<CasGuard> {
            let f = OpenOptions::new()
                .create(true)
                .write(true)
                .read(true)
                .truncate(false)
                .open(&path)?;
            f.lock_exclusive()?;
            Ok(CasGuard { file: Some(f) })
        })
        .await
        .map_err(|e| object_store::Error::Generic {
            store: "LocalFileObjectStore",
            source: Box::new(e),
        })?
        .map_err(|e| object_store::Error::Generic {
            store: "LocalFileObjectStore",
            source: Box::new(e),
        })
    }

    async fn put_with_cas(
        &self,
        location: &Path,
        payload: PutPayload,
        uv: UpdateVersion,
        opts: PutOptions,
    ) -> OsResult<PutResult> {
        let _guard = self.cas_guard().await?;

        // Re-read current meta under the lock.
        let current = match self.inner.head(location).await {
            Ok(meta) => Some(meta),
            Err(object_store::Error::NotFound { .. }) => None,
            Err(e) => return Err(e),
        };

        let current_etag = current.as_ref().and_then(|m| m.e_tag.clone());
        let current_version = current.as_ref().and_then(|m| m.version.clone());
        // Match S3 semantics: both fields are matched if supplied; an
        // absent field on either side is treated as "don't care" *only*
        // when both sides agree on that. Otherwise we report
        // Precondition.
        let etag_ok = match (&uv.e_tag, &current_etag) {
            (Some(a), Some(b)) => a == b,
            (None, _) => true,
            (Some(_), None) => false,
        };
        let version_ok = match (&uv.version, &current_version) {
            (Some(a), Some(b)) => a == b,
            (None, _) => true,
            (Some(_), None) => false,
        };
        if !etag_ok || !version_ok {
            return Err(object_store::Error::Precondition {
                path: location.to_string(),
                source: format!(
                    "CAS failed: expected e_tag={:?} version={:?}, found e_tag={:?} version={:?}",
                    uv.e_tag, uv.version, current_etag, current_version,
                )
                .into(),
            });
        }

        // Issue an Overwrite put through the inner store. Inner
        // LocalFileSystem writes to a temp file in the same directory
        // and renames atomically — exactly what we want for CAS
        // publish.
        let overwrite_opts = PutOptions {
            mode: PutMode::Overwrite,
            tags: opts.tags,
            attributes: opts.attributes,
            extensions: opts.extensions,
        };
        self.inner.put_opts(location, payload, overwrite_opts).await
    }
}

/// RAII guard releasing the advisory CAS lock on drop.
#[derive(Debug)]
struct CasGuard {
    file: Option<std::fs::File>,
}

impl Drop for CasGuard {
    fn drop(&mut self) {
        if let Some(f) = self.file.take() {
            let _ = FileExt::unlock(&f);
        }
    }
}

fn is_internal_path(p: &Path) -> bool {
    p.as_ref().starts_with(INTERNAL_DIR)
}

/// The [`object_store::Error::Generic`] this store uses for IO and task
/// join failures.
fn generic_err(e: impl std::error::Error + Send + Sync + 'static) -> object_store::Error {
    object_store::Error::Generic {
        store: "LocalFileObjectStore",
        source: Box::new(e),
    }
}

/// fsync `fs_path` and its parent directory so a published write survives
/// an OS crash or power loss. `LocalFileSystem` publishes via tmp+rename
/// but never fsyncs. A file that is gone (a delete or a racing overwrite
/// moved it) is treated as already durable.
async fn fsync_published(fs_path: PathBuf) -> OsResult<()> {
    tokio::task::spawn_blocking(move || -> std::io::Result<()> {
        match std::fs::File::open(&fs_path) {
            Ok(f) => f.sync_all()?,
            Err(ref e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(()),
            Err(e) => return Err(e),
        }
        // fsync the parent directory so the rename (the new directory
        // entry) is durable. POSIX-only: std offers no portable directory
        // fsync on Windows, where the file `sync_all` above is the knob.
        #[cfg(unix)]
        if let Some(parent) = fs_path.parent() {
            match std::fs::File::open(parent) {
                Ok(dir) => dir.sync_all()?,
                Err(ref e) if e.kind() == std::io::ErrorKind::NotFound => {}
                Err(e) => return Err(e),
            }
        }
        Ok(())
    })
    .await
    .map_err(generic_err)?
    .map_err(generic_err)
}

/// Wraps a [`MultipartUpload`] so `complete()` fsyncs the published file
/// (and its parent directory) before returning, giving local multipart
/// writes the same crash-durability as the single-PUT path in `put_opts`.
#[derive(Debug)]
struct FsyncMultipartUpload {
    inner: Box<dyn MultipartUpload>,
    fs_path: PathBuf,
}

#[async_trait]
impl MultipartUpload for FsyncMultipartUpload {
    fn put_part(&mut self, data: PutPayload) -> UploadPart {
        self.inner.put_part(data)
    }

    async fn complete(&mut self) -> OsResult<PutResult> {
        let result = self.inner.complete().await?;
        fsync_published(self.fs_path.clone()).await?;
        Ok(result)
    }

    async fn abort(&mut self) -> OsResult<()> {
        self.inner.abort().await
    }
}

#[async_trait]
impl ObjectStore for LocalFileObjectStore {
    async fn put_opts(
        &self,
        location: &Path,
        payload: PutPayload,
        opts: PutOptions,
    ) -> OsResult<PutResult> {
        let result = match opts.mode.clone() {
            PutMode::Update(uv) => self.put_with_cas(location, payload, uv, opts).await?,
            // Create + Overwrite both supported by LocalFileSystem; CAS
            // is not involved so we forward.
            _ => self.inner.put_opts(location, payload, opts).await?,
        };
        // LocalFileSystem publishes via tmp+rename but never fsyncs, so a
        // PUT this method just reported as durable could still be lost on an
        // OS crash or power loss. Flush the file and its parent directory
        // before returning: the local-backend equivalent of S3
        // durability-on-ack. This covers the whole commit path — WAL
        // segment, manifest body, and the pointer CAS all go through here.
        fsync_published(self.fs_path(location)).await?;
        Ok(result)
    }

    async fn put_multipart_opts(
        &self,
        location: &Path,
        opts: PutMultipartOptions,
    ) -> OsResult<Box<dyn MultipartUpload>> {
        // Large SST bodies upload multipart, bypassing put_opts; wrap the
        // upload so its `complete()` fsyncs the published file too.
        let inner = self.inner.put_multipart_opts(location, opts).await?;
        Ok(Box::new(FsyncMultipartUpload {
            inner,
            fs_path: self.fs_path(location),
        }))
    }

    async fn get_opts(&self, location: &Path, options: GetOptions) -> OsResult<GetResult> {
        self.inner.get_opts(location, options).await
    }

    fn list(&self, prefix: Option<&Path>) -> BoxStream<'static, OsResult<ObjectMeta>> {
        let inner = self.list_inner(prefix);
        inner
            .filter(|res| {
                let keep = match res {
                    Ok(meta) => !is_internal_path(&meta.location),
                    Err(_) => true,
                };
                async move { keep }
            })
            .boxed()
    }

    async fn list_with_delimiter(&self, prefix: Option<&Path>) -> OsResult<ListResult> {
        let mut res = self.inner.list_with_delimiter(prefix).await?;
        res.objects.retain(|m| !is_internal_path(&m.location));
        res.common_prefixes
            .retain(|p| !p.as_ref().starts_with(INTERNAL_DIR));
        Ok(res)
    }

    async fn copy_opts(&self, from: &Path, to: &Path, options: CopyOptions) -> OsResult<()> {
        self.inner.copy_opts(from, to, options).await
    }

    fn delete_stream(
        &self,
        locations: BoxStream<'static, OsResult<Path>>,
    ) -> BoxStream<'static, OsResult<Path>> {
        self.inner.delete_stream(locations)
    }
}

impl LocalFileObjectStore {
    fn list_inner(&self, prefix: Option<&Path>) -> BoxStream<'static, OsResult<ObjectMeta>> {
        self.inner.list(prefix)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use bytes::Bytes;
    use object_store::ObjectStore;
    use tempfile::tempdir;
    use uuid::Uuid;

    use namidb_core::NamespaceId;

    use super::*;
    use crate::manifest::ManifestStore;
    use crate::paths::NamespacePaths;

    fn store(dir: &std::path::Path) -> (Arc<dyn ObjectStore>, NamespacePaths) {
        let s: Arc<dyn ObjectStore> = Arc::new(LocalFileObjectStore::new(dir).unwrap());
        let paths = NamespacePaths::new("", NamespaceId::new("acme").unwrap());
        (s, paths)
    }

    #[tokio::test]
    async fn put_maps_to_on_disk_path_and_round_trips() {
        // Guards the key->path mapping that `fsync_published` relies on: a
        // PUT lands at root.join(key), reads back, and the fsync in
        // `put_opts` does not error on the commit-path key shape.
        let dir = tempdir().unwrap();
        let s = LocalFileObjectStore::new(dir.path()).unwrap();
        let p = Path::from("wal/0000000001.wal");
        s.put_opts(
            &p,
            PutPayload::from(Bytes::from_static(b"durable")),
            PutOptions::from(PutMode::Create),
        )
        .await
        .unwrap();
        let on_disk = dir.path().join("wal").join("0000000001.wal");
        assert_eq!(std::fs::read(&on_disk).unwrap(), b"durable");
    }

    #[tokio::test]
    async fn create_then_update_round_trip() {
        let dir = tempdir().unwrap();
        let s = LocalFileObjectStore::new(dir.path()).unwrap();
        let p = Path::from("a/b/c.json");

        // Create succeeds the first time, fails the second (write-once).
        let r1 = s
            .put_opts(
                &p,
                PutPayload::from(Bytes::from_static(b"v0")),
                PutOptions::from(PutMode::Create),
            )
            .await
            .unwrap();
        assert!(r1.e_tag.is_some());

        let err = s
            .put_opts(
                &p,
                PutPayload::from(Bytes::from_static(b"v0-conflict")),
                PutOptions::from(PutMode::Create),
            )
            .await
            .unwrap_err();
        assert!(
            matches!(err, object_store::Error::AlreadyExists { .. }),
            "expected AlreadyExists, got {err:?}"
        );

        // Update with the correct e_tag succeeds; with a stale one fails.
        let head_after_create = s.head(&p).await.unwrap();
        let etag = head_after_create.e_tag.clone();

        let r2 = s
            .put_opts(
                &p,
                PutPayload::from(Bytes::from_static(b"v1")),
                PutOptions::from(PutMode::Update(UpdateVersion {
                    e_tag: etag.clone(),
                    version: head_after_create.version.clone(),
                })),
            )
            .await
            .unwrap();
        assert_ne!(r2.e_tag, etag);

        let err = s
            .put_opts(
                &p,
                PutPayload::from(Bytes::from_static(b"v2-stale")),
                PutOptions::from(PutMode::Update(UpdateVersion {
                    e_tag: etag,
                    version: None,
                })),
            )
            .await
            .unwrap_err();
        assert!(
            matches!(err, object_store::Error::Precondition { .. }),
            "expected Precondition, got {err:?}"
        );
    }

    #[tokio::test]
    async fn list_hides_internal_lock_dir() {
        let dir = tempdir().unwrap();
        let s = LocalFileObjectStore::new(dir.path()).unwrap();
        // Touch the lock to make sure it materialises on disk.
        let _ = s.cas_guard().await.unwrap();

        // Put one user-visible object.
        let p = Path::from("manifest/current.json");
        s.put(&p, PutPayload::from(Bytes::from_static(b"x")))
            .await
            .unwrap();

        let mut found = Vec::new();
        let mut s_iter = s.list(None);
        while let Some(item) = s_iter.next().await {
            found.push(item.unwrap().location.to_string());
        }
        assert_eq!(found, vec!["manifest/current.json".to_string()]);
    }

    #[tokio::test]
    async fn manifest_cas_round_trip_drives_through_namidb_protocol() {
        // End-to-end: the real ManifestStore should bootstrap, commit,
        // and detect a CAS loss against this object store, identical to
        // the InMemory test in manifest.rs.
        let dir = tempdir().unwrap();
        let (store, paths) = store(dir.path());
        let ms = ManifestStore::new(store, paths);
        let w = Uuid::now_v7();

        let base = ms.bootstrap(w).await.unwrap();
        let fence = crate::fence::WriterFence::new(base.manifest.epoch);

        // First commit advances to v1.
        let next = base.manifest.next_version(fence.writer_id);
        let v1 = ms.commit(&fence, &base, next).await.unwrap();
        assert_eq!(v1.manifest.version, 1);

        // A stale writer B holding the old base must lose CAS.
        let b_next = base.manifest.next_version(fence.writer_id);
        let err = ms.commit(&fence, &base, b_next).await.unwrap_err();
        match err {
            crate::error::Error::ManifestCommitCas { expected, found } => {
                assert_eq!(expected, 0);
                assert_eq!(found, 1);
            }
            other => panic!("expected ManifestCommitCas, got {other:?}"),
        }

        // Reloading sees v1, and another commit on top of it works.
        let reloaded = ms.load_current().await.unwrap();
        assert_eq!(reloaded.manifest.version, 1);
        let next2 = reloaded.manifest.next_version(fence.writer_id);
        let v2 = ms.commit(&fence, &reloaded, next2).await.unwrap();
        assert_eq!(v2.manifest.version, 2);
    }

    #[tokio::test]
    async fn concurrent_cas_serialises_correctly() {
        // Two parallel writers both observe v0 and try to commit v1.
        // Exactly one must succeed; the other must report CAS loss.
        let dir = tempdir().unwrap();
        let (store, paths) = store(dir.path());
        let ms = Arc::new(ManifestStore::new(store, paths));
        let w = Uuid::now_v7();
        let base = ms.bootstrap(w).await.unwrap();
        let fence = crate::fence::WriterFence::new(base.manifest.epoch);

        let ms_a = Arc::clone(&ms);
        let ms_b = Arc::clone(&ms);
        let base_a = base.clone();
        let base_b = base.clone();
        let fence_a = fence;
        let fence_b = fence;

        let h_a = tokio::spawn(async move {
            let next = base_a.manifest.next_version(fence_a.writer_id);
            ms_a.commit(&fence_a, &base_a, next).await
        });
        let h_b = tokio::spawn(async move {
            let next = base_b.manifest.next_version(fence_b.writer_id);
            ms_b.commit(&fence_b, &base_b, next).await
        });

        let r_a = h_a.await.unwrap();
        let r_b = h_b.await.unwrap();
        let oks = [r_a.is_ok(), r_b.is_ok()].iter().filter(|b| **b).count();
        let cas_losses = [r_a.as_ref().err(), r_b.as_ref().err()]
            .iter()
            .filter(|e| matches!(e, Some(crate::error::Error::ManifestCommitCas { .. })))
            .count();
        assert_eq!(
            oks, 1,
            "exactly one writer should succeed; got {r_a:?} / {r_b:?}"
        );
        assert_eq!(cas_losses, 1, "the other writer must report CAS loss");

        // Final state at v1.
        let reloaded = ms.load_current().await.unwrap();
        assert_eq!(reloaded.manifest.version, 1);
    }

    #[tokio::test]
    async fn update_against_missing_object_is_precondition() {
        let dir = tempdir().unwrap();
        let s = LocalFileObjectStore::new(dir.path()).unwrap();
        let p = Path::from("does/not/exist.json");
        let err = s
            .put_opts(
                &p,
                PutPayload::from(Bytes::from_static(b"x")),
                PutOptions::from(PutMode::Update(UpdateVersion {
                    e_tag: Some("anything".into()),
                    version: None,
                })),
            )
            .await
            .unwrap_err();
        assert!(
            matches!(err, object_store::Error::Precondition { .. }),
            "expected Precondition, got {err:?}"
        );
    }
}
