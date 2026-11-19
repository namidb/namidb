//! Local-filesystem [`ObjectStore`] with conditional-write support.
//!
//! ## Status after RFC-029
//!
//! The manifest commit protocol no longer issues [`PutMode::Update`]: the
//! pointer is now a Create-only versioned family (`manifest/pointer/p<N>.json`),
//! so commits depend only on `If-None-Match: *` (PUT-if-absent), which
//! `LocalFileSystem` supports natively via `O_CREAT|O_EXCL`. The advisory
//! `flock` CAS below is therefore **no longer load-bearing for commits**. It is
//! retained as a faithful `ObjectStore` implementation (and is still exercised
//! by this module's `PutMode::Update` contract tests); fully removing it is a
//! follow-up once we confirm nothing external relies on local conditional
//! overwrite.
//!
//! `object_store::local::LocalFileSystem` does not implement
//! [`PutMode::Update`] — it returns `NotImplemented` — so it could not
//! drive the pre-RFC-029 pointer overwrite on its own. This wrapper
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

/// Threads that serve ranged file reads ([`run_file_io`]).
///
/// Tokio's blocking pool keeps every queued task behind one mutex that the
/// async workers take to submit and the pool threads take to finish and to
/// park. A query issues a few hundred range reads, so with a handful of
/// concurrent queries the two async workers of a 2-CPU server spent most
/// of their time contending for that lock instead of driving requests.
/// These reads get their own threads fed by MPMC channels.
///
/// The threads are split into groups of [`FILE_IO_GROUP`], each with its
/// own queue, and reads are dealt round-robin across the groups: one queue
/// for every thread moved the contention into the queue's wake-up lock,
/// which every submission and every idle thread took. A read stuck on a
/// slow disk delays only the reads queued behind it in its own group.
#[cfg(unix)]
struct FileIoPool {
    groups: Vec<crossbeam_channel::Sender<FileIoJob>>,
    next: std::sync::atomic::AtomicUsize,
}

#[cfg(unix)]
type FileIoJob = Box<dyn FnOnce() + Send>;

/// Threads per [`FileIoPool`] queue.
#[cfg(unix)]
const FILE_IO_GROUP: usize = 4;

/// `NAMIDB_LOCAL_IO_THREADS`; default eight per CPU, between 8 and 64. A
/// read that misses the page cache blocks its thread on the disk, so the
/// pool is sized for I/O concurrency, not for CPU.
#[cfg(unix)]
fn file_io_threads() -> usize {
    std::env::var("NAMIDB_LOCAL_IO_THREADS")
        .ok()
        .and_then(|raw| raw.parse::<usize>().ok())
        .filter(|threads| *threads > 0)
        .unwrap_or_else(|| {
            std::thread::available_parallelism()
                .map_or(1, |n| n.get())
                .saturating_mul(8)
                .clamp(8, 64)
        })
}

#[cfg(unix)]
impl FileIoPool {
    /// `threads` I/O threads in queues of [`FILE_IO_GROUP`].
    fn new(threads: usize) -> Self {
        let threads = threads.max(1);
        let mut groups = Vec::new();
        for first in (0..threads).step_by(FILE_IO_GROUP) {
            let (jobs, queue) = crossbeam_channel::unbounded::<FileIoJob>();
            for index in first..(first + FILE_IO_GROUP).min(threads) {
                let queue = queue.clone();
                std::thread::Builder::new()
                    .name(format!("namidb-file-io-{index}"))
                    .spawn(move || {
                        for job in queue {
                            // A panicking read must not take the thread with
                            // it: its caller sees the dropped reply as an error.
                            let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(job));
                        }
                    })
                    .expect("spawn local file I/O thread");
            }
            groups.push(jobs);
        }
        FileIoPool {
            groups,
            next: std::sync::atomic::AtomicUsize::new(0),
        }
    }

    /// Run blocking file work on the next queue and await its result. A
    /// pool whose threads are gone, or work that panics, is an error, never
    /// a hang.
    async fn run<T, F>(&self, work: F) -> OsResult<T>
    where
        F: FnOnce() -> T + Send + 'static,
        T: Send + 'static,
    {
        let (reply, result) = tokio::sync::oneshot::channel();
        let group =
            self.next.fetch_add(1, std::sync::atomic::Ordering::Relaxed) % self.groups.len();
        self.groups[group]
            .send(Box::new(move || {
                let _ = reply.send(work());
            }))
            .map_err(|_| generic_err(std::io::Error::other("local file I/O pool is closed")))?;
        result
            .await
            .map_err(|_| generic_err(std::io::Error::other("local file read did not complete")))
    }
}

#[cfg(unix)]
fn file_io_pool() -> &'static FileIoPool {
    static POOL: std::sync::OnceLock<FileIoPool> = std::sync::OnceLock::new();
    POOL.get_or_init(|| FileIoPool::new(file_io_threads()))
}

/// Run blocking file work on the process's [`FileIoPool`].
#[cfg(unix)]
async fn run_file_io<T, F>(work: F) -> OsResult<T>
where
    F: FnOnce() -> T + Send + 'static,
    T: Send + 'static,
{
    file_io_pool().run(work).await
}

/// Outcome of [`ranged_read`].
#[cfg(unix)]
enum RangedRead {
    Done(GetResult),
    /// Not a case the fast path reproduces exactly; ask `LocalFileSystem`.
    Fallback(Path, GetOptions),
}

/// Open, stat, check preconditions and read one byte range, all on the
/// calling (blocking) thread. Metadata and the ETag are derived exactly as
/// `object_store::local::LocalFileSystem` derives them, so `If-Match`
/// pinning by the range cache sees the same tags whichever path served the
/// read.
#[cfg(unix)]
fn ranged_read(path: PathBuf, location: Path, options: GetOptions) -> OsResult<RangedRead> {
    use std::os::unix::fs::MetadataExt;
    let file = match std::fs::File::open(&path) {
        Ok(file) => file,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            return Err(object_store::Error::NotFound {
                path: path.to_string_lossy().into_owned(),
                source: Box::new(error),
            });
        }
        Err(_) => return Ok(RangedRead::Fallback(location, options)),
    };
    let Ok(metadata) = file.metadata() else {
        return Ok(RangedRead::Fallback(location, options));
    };
    if metadata.is_dir() {
        return Ok(RangedRead::Fallback(location, options));
    }
    let Ok(modified) = metadata.modified() else {
        return Ok(RangedRead::Fallback(location, options));
    };
    let mtime_micros = modified
        .duration_since(std::time::SystemTime::UNIX_EPOCH)
        .unwrap_or_default()
        .as_micros();
    let size = metadata.len();
    let meta = ObjectMeta {
        location: location.clone(),
        last_modified: modified.into(),
        size,
        e_tag: Some(format!("{:x}-{mtime_micros:x}-{size:x}", metadata.ino())),
        version: None,
    };
    options.check_preconditions(&meta)?;
    // Only a non-empty range inside the file takes the fast path; an empty
    // or out-of-range request gets `LocalFileSystem`'s own answer.
    let range = match options.range.as_ref().map(|r| r.as_range(size)) {
        Some(Ok(range)) if range.start < size && range.start < range.end => range,
        _ => return Ok(RangedRead::Fallback(location, options)),
    };
    let end = range.end.min(size);
    let len = usize::try_from(end - range.start).map_err(generic_err)?;
    let Some(buffer) = read_exact_at_or_short(&file, range.start, len).map_err(generic_err)? else {
        // The file shrank between the stat and the read (replaced or
        // truncated under us): let `LocalFileSystem` stat and read again.
        return Ok(RangedRead::Fallback(location, options));
    };
    let bytes = bytes::Bytes::from(buffer);
    Ok(RangedRead::Done(GetResult {
        payload: object_store::GetResultPayload::Stream(
            futures::stream::once(async move { Ok(bytes) }).boxed(),
        ),
        meta,
        range: range.start..end,
        attributes: Default::default(),
    }))
}

/// Read exactly `len` bytes at `offset`; `None` when the file ends first.
#[cfg(unix)]
fn read_exact_at_or_short(
    file: &std::fs::File,
    offset: u64,
    len: usize,
) -> std::io::Result<Option<Vec<u8>>> {
    use std::os::unix::fs::FileExt;
    let mut buffer = vec![0u8; len];
    match file.read_exact_at(&mut buffer, offset) {
        Ok(()) => Ok(Some(buffer)),
        Err(error) if error.kind() == std::io::ErrorKind::UnexpectedEof => Ok(None),
        Err(error) => Err(error),
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
        #[cfg(unix)]
        {
            self.get_opts_one_hop(location, options).await
        }
        #[cfg(not(unix))]
        {
            self.inner.get_opts(location, options).await
        }
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
    /// Ranged reads (every range-cache miss and SST page read) take one
    /// blocking-pool hop instead of `LocalFileSystem`'s two: it opens the
    /// file on one blocking task and reads it on another. Each hop locks the
    /// blocking pool's single queue mutex from an async worker; with a few
    /// hundred range reads per query that mutex was where the runtime
    /// workers of a busy server spent their time. Anything this path does
    /// not reproduce exactly (HEAD, versions, whole-object reads, an
    /// unsatisfiable range, an unusual open error) goes to `LocalFileSystem`.
    #[cfg(unix)]
    async fn get_opts_one_hop(&self, location: &Path, options: GetOptions) -> OsResult<GetResult> {
        if options.head || options.version.is_some() || options.range.is_none() {
            return self.inner.get_opts(location, options).await;
        }
        let path = self.inner.path_to_filesystem(location)?;
        let owned = location.clone();
        let read = run_file_io(move || ranged_read(path, owned, options)).await??;
        match read {
            RangedRead::Done(result) => Ok(result),
            RangedRead::Fallback(location, options) => {
                self.inner.get_opts(&location, options).await
            }
        }
    }

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

    /// The one-hop ranged read must be indistinguishable from
    /// `LocalFileSystem`'s: same bytes, same metadata and ETag (the range
    /// cache pins pages with `If-Match` on it), same precondition outcomes,
    /// and the cases it does not handle still answered by `LocalFileSystem`.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn ranged_reads_match_local_file_system() {
        use object_store::{GetRange, ObjectStoreExt};
        let dir = tempdir().unwrap();
        let ours = LocalFileObjectStore::new(dir.path()).unwrap();
        let reference = LocalFileSystem::new_with_prefix(dir.path()).unwrap();
        let p = Path::from("sst/level0/page.bin");
        let body: Vec<u8> = (0..100_000u32).map(|i| (i % 251) as u8).collect();
        ours.put_opts(&p, PutPayload::from(body.clone()), PutOptions::default())
            .await
            .unwrap();

        for range in [0..1, 10..4096, 99_000..100_000, 5..100_000] {
            let options = || GetOptions {
                range: Some(GetRange::Bounded(range.clone())),
                ..Default::default()
            };
            let a = ours.get_opts(&p, options()).await.unwrap();
            let b = reference.get_opts(&p, options()).await.unwrap();
            assert_eq!(a.meta, b.meta);
            assert_eq!(a.range, b.range);
            let (a, b) = (a.bytes().await.unwrap(), b.bytes().await.unwrap());
            assert_eq!(a, b);
            assert_eq!(&a[..], &body[range.start as usize..range.end as usize]);
        }
        // Suffix and offset ranges, and a range past the end (clamped).
        for range in [
            GetRange::Suffix(10),
            GetRange::Offset(99_990),
            GetRange::Bounded(99_990..200_000),
        ] {
            let options = || GetOptions {
                range: Some(range.clone()),
                ..Default::default()
            };
            let a = ours
                .get_opts(&p, options())
                .await
                .unwrap()
                .bytes()
                .await
                .unwrap();
            let b = reference
                .get_opts(&p, options())
                .await
                .unwrap()
                .bytes()
                .await
                .unwrap();
            assert_eq!(a, b);
        }

        // If-Match pinning: the right tag reads, a stale one is a precondition failure.
        let etag = reference.head(&p).await.unwrap().e_tag.unwrap();
        let pinned = |tag: &str| GetOptions {
            range: Some(GetRange::Bounded(0..8)),
            if_match: Some(tag.to_string()),
            ..Default::default()
        };
        assert!(ours.get_opts(&p, pinned(&etag)).await.is_ok());
        let stale = ours.get_opts(&p, pinned("0-0-0")).await.unwrap_err();
        assert!(
            matches!(stale, object_store::Error::Precondition { .. }),
            "{stale}"
        );

        // Missing objects are NotFound; an unsatisfiable range is still an error.
        let missing = ours
            .get_opts(
                &Path::from("sst/level0/missing.bin"),
                GetOptions {
                    range: Some(GetRange::Bounded(0..8)),
                    ..Default::default()
                },
            )
            .await
            .unwrap_err();
        assert!(
            matches!(missing, object_store::Error::NotFound { .. }),
            "{missing}"
        );
        assert!(ours
            .get_opts(
                &p,
                GetOptions {
                    range: Some(GetRange::Bounded(200_000..200_010)),
                    ..Default::default()
                },
            )
            .await
            .is_err());
    }

    /// Edge ranges: an empty range, a zero-length object and an empty
    /// suffix get exactly `LocalFileSystem`'s outcome (they fall back).
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn edge_ranges_match_local_file_system() {
        use object_store::{GetRange, ObjectStoreExt};
        let dir = tempdir().unwrap();
        let ours = LocalFileObjectStore::new(dir.path()).unwrap();
        let reference = LocalFileSystem::new_with_prefix(dir.path()).unwrap();
        let full = Path::from("sst/full.bin");
        let empty = Path::from("sst/empty.bin");
        ours.put_opts(
            &full,
            PutPayload::from(vec![7u8; 4096]),
            PutOptions::default(),
        )
        .await
        .unwrap();
        ours.put_opts(
            &empty,
            PutPayload::from(Vec::<u8>::new()),
            PutOptions::default(),
        )
        .await
        .unwrap();
        let cases = [
            (&full, GetRange::Bounded(5..5)),
            (&full, GetRange::Suffix(0)),
            (&full, GetRange::Offset(4096)),
            (&empty, GetRange::Bounded(0..1)),
            (&empty, GetRange::Offset(0)),
            (&empty, GetRange::Suffix(5)),
        ];
        for (path, range) in cases {
            let options = || GetOptions {
                range: Some(range.clone()),
                ..Default::default()
            };
            let a = match ours.get_opts(path, options()).await {
                Ok(result) => Ok((result.range.clone(), result.bytes().await.unwrap())),
                Err(error) => Err(std::mem::discriminant(&error)),
            };
            let b = match reference.get_opts(path, options()).await {
                Ok(result) => Ok((result.range.clone(), result.bytes().await.unwrap())),
                Err(error) => Err(std::mem::discriminant(&error)),
            };
            assert_eq!(a, b, "{path} {range:?}");
        }
    }

    /// A file that shrinks between the stat and the read is reported as a
    /// short read (and the caller falls back), not as bytes past its end.
    #[cfg(unix)]
    #[test]
    fn short_read_after_truncation_is_detected() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("page.bin");
        std::fs::write(&path, vec![1u8; 8192]).unwrap();
        let file = std::fs::File::open(&path).unwrap();
        assert_eq!(
            read_exact_at_or_short(&file, 4096, 4096).unwrap().unwrap(),
            vec![1u8; 4096]
        );
        std::fs::OpenOptions::new()
            .write(true)
            .open(&path)
            .unwrap()
            .set_len(1000)
            .unwrap();
        assert!(read_exact_at_or_short(&file, 4096, 4096).unwrap().is_none());
        assert!(read_exact_at_or_short(&file, 0, 4096).unwrap().is_none());
        assert_eq!(
            read_exact_at_or_short(&file, 0, 0).unwrap().unwrap(),
            Vec::<u8>::new()
        );
    }

    /// Work that panics on an I/O thread is an error for its caller and
    /// leaves every thread serving; a pool with no threads left errors
    /// instead of hanging.
    #[cfg(unix)]
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn file_io_pool_survives_panics_and_reports_a_closed_pool() {
        let pool = FileIoPool::new(6);
        for _ in 0..12 {
            let error = pool
                .run(|| -> u32 { panic!("injected I/O panic") })
                .await
                .unwrap_err();
            assert!(error.to_string().contains("did not complete"), "{error}");
        }
        for i in 0..64u32 {
            assert_eq!(pool.run(move || i * 2).await.unwrap(), i * 2);
        }

        let (jobs, queue) = crossbeam_channel::unbounded::<FileIoJob>();
        drop(queue);
        let closed = FileIoPool {
            groups: vec![jobs],
            next: std::sync::atomic::AtomicUsize::new(0),
        };
        let error = closed.run(|| 1u8).await.unwrap_err();
        assert!(error.to_string().contains("closed"), "{error}");
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
