//! Bounded-memory upload of immutable artifacts assembled on local disk.
//!
//! Search-index and graph builders can emit corpus-sized artifacts into one or
//! more anonymous/named temporary files, optionally keeping a small encoded
//! prefix as scatter/gather [`Bytes`] chunks. [`SpooledObject`] owns those
//! ordered resources until [`put_spooled_object`] reaches a terminal result.
//!
//! Objects smaller than 5 MiB use one `PutMode::Create` request. Larger
//! objects use parts sized to stay within S3's 10,000-part ceiling (never less
//! than 5 MiB, except the final part), with at most eight part futures in
//! flight. Multipart APIs cannot express
//! create-only semantics, so callers must use immutable collision-resistant
//! object paths, just like the existing SST uploader.

use std::collections::VecDeque;
use std::fs::File;
use std::io::{Seek, SeekFrom};
use std::sync::Arc;

use bytes::{Bytes, BytesMut};
use futures::stream::{FuturesUnordered, StreamExt};
use object_store::path::Path;
use object_store::{MultipartUpload, ObjectStore, ObjectStoreExt, PutMode, PutOptions, PutPayload};
use tempfile::{NamedTempFile, TempPath};
use tokio::io::AsyncReadExt;
use tracing::warn;

use crate::error::{Error, Result};

/// S3/R2 minimum size for every non-final multipart part.
pub(crate) const MULTIPART_PART_SIZE: usize = 5 * 1024 * 1024;
/// Objects below this size retain create-only single-PUT semantics.
pub(crate) const MULTIPART_THRESHOLD: usize = MULTIPART_PART_SIZE;
/// Hard ceiling for resident multipart request bodies owned by one upload.
pub(crate) const MULTIPART_MAX_CONCURRENCY: usize = 8;
/// S3 multipart protocol ceilings.
const MULTIPART_MAX_PARTS: u64 = 10_000;
const MULTIPART_MAX_PART_SIZE: u64 = 5 * 1024 * 1024 * 1024;
const S3_MAX_OBJECT_BYTES: u64 = 5 * 1024 * 1024 * 1024 * 1024;

/// File ownership retained by a spooled artifact.
#[derive(Debug)]
enum SpooledFile {
    /// Usually an anonymous file created with `tempfile::tempfile[_in]`.
    File(File),
    /// Named temporary file whose path must be removed at terminal cleanup.
    #[allow(dead_code)] // Public-crate constructor is consumed by external builders.
    Named(NamedTempFile),
}

impl SpooledFile {
    fn file_mut(&mut self) -> &mut File {
        match self {
            Self::File(file) => file,
            Self::Named(file) => file.as_file_mut(),
        }
    }

    fn into_async(self) -> (tokio::fs::File, Option<TempPath>) {
        match self {
            Self::File(file) => (tokio::fs::File::from_std(file), None),
            Self::Named(file) => {
                let (file, cleanup_path) = file.into_parts();
                (tokio::fs::File::from_std(file), Some(cleanup_path))
            }
        }
    }
}

/// Owned, exactly-sized immutable object waiting to be uploaded.
///
/// `exact_len` is the size recorded in the artifact descriptor, not a hint.
/// The uploader reads and validates the complete logical stream, rejecting
/// both truncation and trailing bytes. Dropping this value closes every file;
/// named temporary files are also unlinked.
#[derive(Debug)]
pub(crate) struct SpooledObject {
    prefix: Vec<Bytes>,
    files: Vec<SpooledFile>,
    exact_len: u64,
}

impl SpooledObject {
    /// Construct an artifact whose full body is in `file`.
    #[allow(dead_code)] // Used when compact installs external FTS/vector artifacts.
    pub(crate) fn from_file(file: File, exact_len: u64) -> Self {
        Self {
            prefix: Vec::new(),
            files: vec![SpooledFile::File(file)],
            exact_len,
        }
    }

    /// Construct an artifact with in-memory prefix chunks followed by an
    /// optional file region.
    pub(crate) fn from_parts(prefix: Vec<Bytes>, file: Option<File>, exact_len: u64) -> Self {
        Self {
            prefix,
            files: file.into_iter().map(SpooledFile::File).collect(),
            exact_len,
        }
    }

    /// Construct an artifact with an in-memory prefix followed by multiple
    /// file regions. Regions are streamed in order without concatenating
    /// corpus-sized page/value spools first.
    pub(crate) fn from_files(prefix: Vec<Bytes>, files: Vec<File>, exact_len: u64) -> Self {
        Self {
            prefix,
            files: files.into_iter().map(SpooledFile::File).collect(),
            exact_len,
        }
    }

    /// Construct an artifact backed by a named temporary file. Its path is
    /// removed when the artifact/upload is dropped, including cancellation.
    #[allow(dead_code)] // Optional public-crate ownership form.
    pub(crate) fn from_named_file(file: NamedTempFile, exact_len: u64) -> Self {
        Self {
            prefix: Vec::new(),
            files: vec![SpooledFile::Named(file)],
            exact_len,
        }
    }

    pub(crate) fn len(&self) -> u64 {
        self.exact_len
    }

    /// Rewind all file regions so a finished builder can hand the same artifact
    /// directly to the async uploader.
    pub(crate) fn rewind(&mut self) -> std::io::Result<()> {
        for file in &mut self.files {
            file.file_mut().seek(SeekFrom::Start(0))?;
        }
        Ok(())
    }

    fn into_reader(mut self) -> Result<SpooledObjectReader> {
        self.rewind()?;
        let files = self
            .files
            .into_iter()
            .map(|file| {
                let (file, cleanup_path) = file.into_async();
                AsyncSpooledFile {
                    file,
                    _cleanup_path: cleanup_path,
                }
            })
            .collect();
        Ok(SpooledObjectReader {
            prefix: self.prefix.into_iter(),
            current: None,
            current_offset: 0,
            files,
            exact_len: self.exact_len,
            produced: 0,
            end_validated: false,
        })
    }
}

#[derive(Debug)]
struct AsyncSpooledFile {
    // Field order is intentional: the open file is dropped before its
    // TempPath tries to unlink on platforms that reject open deletes.
    file: tokio::fs::File,
    _cleanup_path: Option<TempPath>,
}

/// Incremental logical reader over `[prefix chunks][file region 0][file region 1]…`.
#[derive(Debug)]
struct SpooledObjectReader {
    prefix: std::vec::IntoIter<Bytes>,
    current: Option<Bytes>,
    current_offset: usize,
    files: VecDeque<AsyncSpooledFile>,
    exact_len: u64,
    produced: u64,
    end_validated: bool,
}

impl SpooledObjectReader {
    async fn next_part(&mut self, part_size: usize) -> Result<Option<PutPayload>> {
        debug_assert!(part_size > 0);
        if self.end_validated {
            return Ok(None);
        }

        let remaining = self.exact_len.checked_sub(self.produced).ok_or_else(|| {
            Error::invariant("spooled object produced more bytes than its descriptor")
        })?;
        if remaining == 0 {
            self.validate_eof().await?;
            self.end_validated = true;
            return Ok(None);
        }

        let target = usize::try_from(remaining.min(part_size as u64))
            .expect("part target is bounded by usize part size");
        let mut part = BytesMut::with_capacity(target);
        while part.len() < target {
            if let Some(current) = &self.current {
                let available = current.len().saturating_sub(self.current_offset);
                if available == 0 {
                    self.current = None;
                    self.current_offset = 0;
                    continue;
                }
                let take = (target - part.len()).min(available);
                part.extend_from_slice(&current[self.current_offset..self.current_offset + take]);
                self.current_offset += take;
                continue;
            }

            if let Some(next) = self.prefix.next() {
                self.current = Some(next);
                continue;
            }

            let Some(file) = self.files.front_mut() else {
                return Err(length_error(
                    self.exact_len,
                    self.produced.saturating_add(part.len() as u64),
                    "spool ended before the descriptor length",
                ));
            };
            let read_limit = target - part.len();
            let mut limited = (&mut file.file).take(read_limit as u64);
            let read = limited.read_buf(&mut part).await?;
            drop(limited);
            if read == 0 {
                self.files.pop_front();
            }
        }

        self.produced = self
            .produced
            .checked_add(part.len() as u64)
            .ok_or_else(|| Error::invariant("spooled object byte count exceeds u64"))?;
        if self.produced == self.exact_len {
            self.validate_eof().await?;
            self.end_validated = true;
        }
        Ok(Some(PutPayload::from(part.freeze())))
    }

    async fn validate_eof(&mut self) -> Result<()> {
        if self
            .current
            .as_ref()
            .is_some_and(|current| self.current_offset < current.len())
        {
            return Err(length_error(
                self.exact_len,
                self.produced.saturating_add(1),
                "prefix contains trailing bytes",
            ));
        }
        self.current = None;
        self.current_offset = 0;

        if self.prefix.any(|chunk| !chunk.is_empty()) {
            return Err(length_error(
                self.exact_len,
                self.produced.saturating_add(1),
                "prefix contains trailing chunks",
            ));
        }

        while let Some(mut file) = self.files.pop_front() {
            let mut extra = [0_u8; 1];
            if file.file.read(&mut extra).await? != 0 {
                return Err(length_error(
                    self.exact_len,
                    self.produced.saturating_add(1),
                    "spool file contains trailing bytes",
                ));
            }
        }
        Ok(())
    }
}

fn length_error(expected: u64, observed_at_least: u64, detail: &str) -> Error {
    Error::invariant(format!(
        "spooled object length disagrees with descriptor: expected {expected} bytes, \
         observed at least {observed_at_least}: {detail}"
    ))
}

/// Owns an in-progress multipart upload until successful completion.
///
/// Explicit failures synchronously abort. Dropping the enclosing future,
/// including Tokio task cancellation, detaches an abort task on the current
/// runtime. A process crash still relies on the bucket's incomplete-multipart
/// lifecycle rule.
pub(crate) struct MultipartUploadGuard {
    upload: Option<Box<dyn MultipartUpload>>,
    path: Path,
}

impl std::fmt::Debug for MultipartUploadGuard {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MultipartUploadGuard")
            .field("armed", &self.upload.is_some())
            .field("path", &self.path)
            .finish()
    }
}

impl MultipartUploadGuard {
    pub(crate) fn new(upload: Box<dyn MultipartUpload>, path: &Path) -> Self {
        Self {
            upload: Some(upload),
            path: path.clone(),
        }
    }

    pub(crate) fn put_part(&mut self, data: PutPayload) -> object_store::UploadPart {
        self.upload
            .as_mut()
            .expect("multipart upload guard used after completion")
            .put_part(data)
    }

    pub(crate) async fn complete(&mut self) -> object_store::Result<object_store::PutResult> {
        let result = self
            .upload
            .as_mut()
            .expect("multipart upload guard used after completion")
            .complete()
            .await;
        if result.is_ok() {
            self.upload = None;
        }
        result
    }

    pub(crate) async fn abort_after_error(&mut self, context: &'static str) {
        let Some(upload) = self.upload.as_mut() else {
            return;
        };
        match upload.abort().await {
            Ok(()) => self.upload = None,
            Err(error) => {
                warn!(
                    path = %self.path,
                    error = %error,
                    context,
                    "failed to abort multipart upload"
                );
            }
        }
    }
}

impl Drop for MultipartUploadGuard {
    fn drop(&mut self) {
        let Some(mut upload) = self.upload.take() else {
            return;
        };
        let path = self.path.clone();
        match tokio::runtime::Handle::try_current() {
            Ok(runtime) => {
                let _cleanup = runtime.spawn(async move {
                    if let Err(error) = upload.abort().await {
                        warn!(
                            path = %path,
                            error = %error,
                            "failed to abort multipart upload during cancellation cleanup"
                        );
                    }
                });
            }
            Err(error) => {
                warn!(
                    path = %path,
                    error = %error,
                    "could not schedule multipart cancellation cleanup"
                );
            }
        }
    }
}

async fn put_create_payload(store: &dyn ObjectStore, path: &Path, body: PutPayload) -> Result<()> {
    store
        .put_opts(path, body, PutOptions::from(PutMode::Create))
        .await
        .map_err(Error::ObjectStore)?;
    Ok(())
}

/// Upload a descriptor-sized spooled artifact without materialising the final
/// object in memory.
pub(crate) async fn put_spooled_object(
    store: Arc<dyn ObjectStore>,
    path: &Path,
    artifact: SpooledObject,
) -> Result<()> {
    let part_size = effective_multipart_part_size(artifact.len(), MULTIPART_PART_SIZE)?;
    put_spooled_object_with_limits(store, path, artifact, part_size, MULTIPART_MAX_CONCURRENCY)
        .await
}

fn effective_multipart_part_size(exact_len: u64, minimum_part_size: usize) -> Result<usize> {
    if exact_len > S3_MAX_OBJECT_BYTES {
        return Err(Error::invariant(format!(
            "spooled object length {exact_len} exceeds the S3 object limit \
             {S3_MAX_OBJECT_BYTES}"
        )));
    }
    let by_part_count = exact_len.div_ceil(MULTIPART_MAX_PARTS);
    let part_size = by_part_count.max(minimum_part_size as u64);
    if part_size > MULTIPART_MAX_PART_SIZE {
        return Err(Error::invariant(format!(
            "spooled multipart part size {part_size} exceeds the S3 limit \
             {MULTIPART_MAX_PART_SIZE}"
        )));
    }
    usize::try_from(part_size)
        .map_err(|_| Error::invariant("spooled multipart part size does not fit usize"))
}

async fn verify_uploaded_object(
    store: &dyn ObjectStore,
    path: &Path,
    expected_len: u64,
) -> Result<()> {
    let meta = store.head(path).await.map_err(Error::ObjectStore)?;
    if meta.location != *path || meta.size != expected_len {
        return Err(Error::invariant(format!(
            "uploaded object verification failed for {path}: HEAD returned {} \
             with {} bytes, expected {expected_len}",
            meta.location, meta.size
        )));
    }
    Ok(())
}

async fn put_spooled_object_with_limits(
    store: Arc<dyn ObjectStore>,
    path: &Path,
    artifact: SpooledObject,
    part_size: usize,
    max_concurrency: usize,
) -> Result<()> {
    if part_size == 0 || max_concurrency == 0 {
        return Err(Error::invariant(
            "spooled multipart part size and concurrency must be non-zero",
        ));
    }

    let exact_len = artifact.len();
    if exact_len > S3_MAX_OBJECT_BYTES {
        return Err(Error::invariant(format!(
            "spooled object length {exact_len} exceeds the S3 object limit \
             {S3_MAX_OBJECT_BYTES}"
        )));
    }
    if part_size as u64 > MULTIPART_MAX_PART_SIZE {
        return Err(Error::invariant(format!(
            "spooled multipart part size {part_size} exceeds the S3 limit \
             {MULTIPART_MAX_PART_SIZE}"
        )));
    }
    let part_count = exact_len.div_ceil(part_size as u64);
    if part_count > MULTIPART_MAX_PARTS {
        return Err(Error::invariant(format!(
            "spooled object would require {part_count} multipart parts, above the S3 limit \
             {MULTIPART_MAX_PARTS}"
        )));
    }
    let mut reader = artifact.into_reader()?;
    if exact_len < part_size as u64 {
        let body = reader.next_part(part_size).await?.unwrap_or_default();
        if body.content_length() as u64 != exact_len || reader.produced != exact_len {
            return Err(Error::invariant(
                "spooled single-PUT byte count disagrees with descriptor",
            ));
        }
        debug_assert!(reader.next_part(1).await?.is_none());
        put_create_payload(store.as_ref(), path, body).await?;
        return verify_uploaded_object(store.as_ref(), path, exact_len).await;
    }

    let upload = store
        .put_multipart(path)
        .await
        .map_err(Error::ObjectStore)?;
    let mut upload = MultipartUploadGuard::new(upload, path);
    let mut pending = FuturesUnordered::new();
    let mut upload_error: Option<Error> = None;

    loop {
        match reader.next_part(part_size).await {
            Ok(Some(part)) => {
                let part_len = part.content_length();
                let is_final = reader.produced == exact_len;
                if (!is_final && part_len != part_size) || part_len > part_size {
                    upload_error = Some(Error::invariant(
                        "spooled multipart produced an invalid part size",
                    ));
                    break;
                }
                pending.push(upload.put_part(part));
            }
            Ok(None) => break,
            Err(error) => {
                upload_error = Some(error);
                break;
            }
        }

        if pending.len() < max_concurrency {
            continue;
        }
        if let Some(Err(source)) = pending.next().await {
            upload_error = Some(Error::ObjectStore(source));
            break;
        }
    }

    while let Some(result) = pending.next().await {
        if upload_error.is_none() {
            if let Err(source) = result {
                upload_error = Some(Error::ObjectStore(source));
            }
        }
    }
    if let Some(error) = upload_error {
        upload.abort_after_error("spooled upload failure").await;
        return Err(error);
    }
    if reader.produced != exact_len || !reader.end_validated {
        upload
            .abort_after_error("spooled upload length mismatch")
            .await;
        return Err(Error::invariant(
            "spooled multipart byte count disagrees with descriptor",
        ));
    }

    if let Err(source) = upload.complete().await {
        upload
            .abort_after_error("spooled upload completion failure")
            .await;
        return Err(Error::ObjectStore(source));
    }
    verify_uploaded_object(store.as_ref(), path, exact_len).await
}

#[cfg(test)]
mod tests {
    use std::fmt;
    use std::io::Write;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex};
    use std::time::Duration;

    use async_trait::async_trait;
    use futures::stream::BoxStream;
    use object_store::memory::InMemory;
    use object_store::{
        CopyOptions, GetOptions, GetResult, ListResult, ObjectMeta, PutMultipartOptions, PutResult,
        UploadPart,
    };
    use tokio::sync::{mpsc, Notify, Semaphore};

    use super::*;

    #[derive(Debug)]
    struct ProbeState {
        single_puts: AtomicUsize,
        single_create_puts: AtomicUsize,
        largest_single_payload: AtomicUsize,
        multipart_started: AtomicUsize,
        part_lengths: Mutex<Vec<usize>>,
        active_parts: AtomicUsize,
        maximum_active_parts: AtomicUsize,
        completes: AtomicUsize,
        aborts: AtomicUsize,
        abort_notify: Notify,
        fail_part: Option<usize>,
        fail_complete: AtomicBool,
        part_gate: Option<Arc<Semaphore>>,
        part_started: Option<mpsc::UnboundedSender<usize>>,
    }

    impl ProbeState {
        fn new(
            fail_part: Option<usize>,
            fail_complete: bool,
            part_gate: Option<Arc<Semaphore>>,
            part_started: Option<mpsc::UnboundedSender<usize>>,
        ) -> Self {
            Self {
                single_puts: AtomicUsize::new(0),
                single_create_puts: AtomicUsize::new(0),
                largest_single_payload: AtomicUsize::new(0),
                multipart_started: AtomicUsize::new(0),
                part_lengths: Mutex::new(Vec::new()),
                active_parts: AtomicUsize::new(0),
                maximum_active_parts: AtomicUsize::new(0),
                completes: AtomicUsize::new(0),
                aborts: AtomicUsize::new(0),
                abort_notify: Notify::new(),
                fail_part,
                fail_complete: AtomicBool::new(fail_complete),
                part_gate,
                part_started,
            }
        }
    }

    #[derive(Debug)]
    struct ProbeStore {
        inner: Arc<InMemory>,
        state: Arc<ProbeState>,
    }

    impl ProbeStore {
        fn new(state: Arc<ProbeState>) -> Arc<Self> {
            Arc::new(Self {
                inner: Arc::new(InMemory::new()),
                state,
            })
        }
    }

    impl fmt::Display for ProbeStore {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.write_str("SpooledObjectProbeStore")
        }
    }

    #[async_trait]
    impl ObjectStore for ProbeStore {
        async fn put_opts(
            &self,
            location: &Path,
            payload: PutPayload,
            opts: PutOptions,
        ) -> object_store::Result<PutResult> {
            self.state.single_puts.fetch_add(1, Ordering::SeqCst);
            self.state
                .largest_single_payload
                .fetch_max(payload.content_length(), Ordering::SeqCst);
            if matches!(&opts.mode, PutMode::Create) {
                self.state.single_create_puts.fetch_add(1, Ordering::SeqCst);
            }
            self.inner.put_opts(location, payload, opts).await
        }

        async fn put_multipart_opts(
            &self,
            location: &Path,
            _opts: PutMultipartOptions,
        ) -> object_store::Result<Box<dyn MultipartUpload>> {
            self.state.multipart_started.fetch_add(1, Ordering::SeqCst);
            Ok(Box::new(ProbeMultipart {
                inner: self.inner.clone(),
                state: self.state.clone(),
                location: location.clone(),
                parts: Arc::new(Mutex::new(Vec::new())),
                next_part: 0,
            }))
        }

        async fn get_opts(
            &self,
            location: &Path,
            options: GetOptions,
        ) -> object_store::Result<GetResult> {
            self.inner.get_opts(location, options).await
        }

        fn list(
            &self,
            prefix: Option<&Path>,
        ) -> BoxStream<'static, object_store::Result<ObjectMeta>> {
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
            locations: BoxStream<'static, object_store::Result<Path>>,
        ) -> BoxStream<'static, object_store::Result<Path>> {
            self.inner.delete_stream(locations)
        }
    }

    #[derive(Debug)]
    struct ProbeMultipart {
        inner: Arc<InMemory>,
        state: Arc<ProbeState>,
        location: Path,
        parts: Arc<Mutex<Vec<Option<Bytes>>>>,
        next_part: usize,
    }

    impl ProbeMultipart {
        fn error(message: &'static str) -> object_store::Error {
            object_store::Error::Generic {
                store: "SpooledObjectProbeStore",
                source: std::io::Error::other(message).into(),
            }
        }
    }

    #[derive(Debug)]
    struct ActivePart {
        state: Arc<ProbeState>,
    }

    impl ActivePart {
        fn enter(state: Arc<ProbeState>) -> Self {
            let active = state.active_parts.fetch_add(1, Ordering::SeqCst) + 1;
            state
                .maximum_active_parts
                .fetch_max(active, Ordering::SeqCst);
            Self { state }
        }
    }

    impl Drop for ActivePart {
        fn drop(&mut self) {
            self.state.active_parts.fetch_sub(1, Ordering::SeqCst);
        }
    }

    #[async_trait]
    impl MultipartUpload for ProbeMultipart {
        fn put_part(&mut self, payload: PutPayload) -> UploadPart {
            let index = self.next_part;
            self.next_part += 1;
            let bytes = collect_payload(payload);
            self.state.part_lengths.lock().unwrap().push(bytes.len());
            self.parts.lock().unwrap().push(None);

            let state = self.state.clone();
            let parts = self.parts.clone();
            Box::pin(async move {
                let _active = ActivePart::enter(state.clone());
                if let Some(started) = &state.part_started {
                    let _ = started.send(index);
                }
                if let Some(gate) = &state.part_gate {
                    let permit = gate
                        .clone()
                        .acquire_owned()
                        .await
                        .map_err(|_| ProbeMultipart::error("test part gate closed"))?;
                    permit.forget();
                }
                if state.fail_part == Some(index) {
                    return Err(ProbeMultipart::error("injected multipart part failure"));
                }
                parts.lock().unwrap()[index] = Some(bytes);
                Ok(())
            })
        }

        async fn complete(&mut self) -> object_store::Result<PutResult> {
            self.state.completes.fetch_add(1, Ordering::SeqCst);
            if self.state.fail_complete.load(Ordering::SeqCst) {
                return Err(Self::error("injected multipart completion failure"));
            }

            let body = {
                let parts = self.parts.lock().unwrap();
                let total = parts.iter().try_fold(0usize, |total, part| {
                    part.as_ref().and_then(|part| total.checked_add(part.len()))
                });
                let Some(total) = total else {
                    return Err(Self::error("multipart completed with a missing part"));
                };
                let mut body = BytesMut::with_capacity(total);
                for part in parts.iter() {
                    body.extend_from_slice(
                        part.as_ref()
                            .expect("missing part rejected by total calculation"),
                    );
                }
                body.freeze()
            };
            self.inner.put(&self.location, PutPayload::from(body)).await
        }

        async fn abort(&mut self) -> object_store::Result<()> {
            self.state.aborts.fetch_add(1, Ordering::SeqCst);
            self.state.abort_notify.notify_one();
            Ok(())
        }
    }

    fn collect_payload(payload: PutPayload) -> Bytes {
        let mut body = BytesMut::with_capacity(payload.content_length());
        for chunk in payload {
            body.extend_from_slice(&chunk);
        }
        body.freeze()
    }

    fn anonymous_file(body: &[u8]) -> File {
        let mut file = tempfile::tempfile().unwrap();
        file.write_all(body).unwrap();
        file
    }

    fn patterned_body(len: usize) -> Vec<u8> {
        (0..len)
            .map(|offset| (offset as u8).wrapping_mul(31).wrapping_add(7))
            .collect()
    }

    fn plain_probe() -> (Arc<ProbeStore>, Arc<ProbeState>) {
        let state = Arc::new(ProbeState::new(None, false, None, None));
        (ProbeStore::new(state.clone()), state)
    }

    #[test]
    fn multipart_part_size_scales_before_the_ten_thousand_part_limit() {
        let just_over_fixed_limit = MULTIPART_PART_SIZE as u64 * MULTIPART_MAX_PARTS + 1;
        let selected =
            effective_multipart_part_size(just_over_fixed_limit, MULTIPART_PART_SIZE).unwrap();
        assert!(selected > MULTIPART_PART_SIZE);
        assert!(just_over_fixed_limit.div_ceil(selected as u64) <= MULTIPART_MAX_PARTS);

        let largest =
            effective_multipart_part_size(S3_MAX_OBJECT_BYTES, MULTIPART_PART_SIZE).unwrap();
        assert!(largest as u64 <= MULTIPART_MAX_PART_SIZE);
        assert!(S3_MAX_OBJECT_BYTES.div_ceil(largest as u64) <= MULTIPART_MAX_PARTS);
        assert!(
            effective_multipart_part_size(S3_MAX_OBJECT_BYTES + 1, MULTIPART_PART_SIZE).is_err()
        );
    }

    #[tokio::test]
    async fn impossible_part_count_fails_before_starting_multipart() {
        let (store, state) = plain_probe();
        let artifact = SpooledObject::from_parts(Vec::new(), None, MULTIPART_MAX_PARTS + 1);
        let error = put_spooled_object_with_limits(
            store,
            &Path::from("too-many-parts.bin"),
            artifact,
            1,
            1,
        )
        .await
        .unwrap_err();
        assert!(matches!(error, Error::Invariant(_)));
        assert_eq!(state.multipart_started.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn small_spool_rewinds_and_uses_one_create_only_put() {
        let (store, state) = plain_probe();
        let path = Path::from("small-spooled.bin");
        let prefix = vec![Bytes::from_static(b"header-"), Bytes::from_static(b"v1:")];
        let suffix = b"bounded file body";
        let expected = [b"header-v1:".as_slice(), suffix.as_slice()].concat();
        let artifact =
            SpooledObject::from_parts(prefix, Some(anonymous_file(suffix)), expected.len() as u64);

        put_spooled_object(store.clone(), &path, artifact)
            .await
            .unwrap();
        assert_eq!(state.single_puts.load(Ordering::SeqCst), 1);
        assert_eq!(state.single_create_puts.load(Ordering::SeqCst), 1);
        assert_eq!(state.multipart_started.load(Ordering::SeqCst), 0);
        assert!(state.largest_single_payload.load(Ordering::SeqCst) < MULTIPART_THRESHOLD);
        assert_eq!(
            store.get(&path).await.unwrap().bytes().await.unwrap(),
            Bytes::from(expected.clone())
        );

        let duplicate = SpooledObject::from_file(anonymous_file(&expected), expected.len() as u64);
        assert!(
            put_spooled_object(store, &path, duplicate).await.is_err(),
            "the small path must preserve PutMode::Create"
        );
    }

    #[tokio::test]
    async fn large_spool_has_exact_five_mib_parts_and_exact_content() {
        let (store, state) = plain_probe();
        let path = Path::from("large-spooled.bin");
        let total = MULTIPART_PART_SIZE * 3 + 123;
        let expected = patterned_body(total);
        let prefix_end = 777_777;
        let prefix = vec![
            Bytes::copy_from_slice(&expected[..13]),
            Bytes::copy_from_slice(&expected[13..prefix_end]),
        ];
        let artifact = SpooledObject::from_parts(
            prefix,
            Some(anonymous_file(&expected[prefix_end..])),
            total as u64,
        );

        put_spooled_object(store.clone(), &path, artifact)
            .await
            .unwrap();
        assert_eq!(state.single_puts.load(Ordering::SeqCst), 0);
        assert_eq!(state.multipart_started.load(Ordering::SeqCst), 1);
        assert_eq!(
            *state.part_lengths.lock().unwrap(),
            vec![
                MULTIPART_PART_SIZE,
                MULTIPART_PART_SIZE,
                MULTIPART_PART_SIZE,
                123
            ]
        );
        assert_eq!(state.aborts.load(Ordering::SeqCst), 0);
        assert_eq!(state.completes.load(Ordering::SeqCst), 1);
        assert_eq!(
            store.get(&path).await.unwrap().bytes().await.unwrap(),
            Bytes::from(expected)
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn multipart_part_futures_never_exceed_the_configured_limit() {
        let part_size = 32 * 1024;
        let parts = MULTIPART_MAX_CONCURRENCY + 3;
        let body = patterned_body(part_size * parts + 7);
        let gate = Arc::new(Semaphore::new(0));
        let (started_tx, mut started_rx) = mpsc::unbounded_channel();
        let state = Arc::new(ProbeState::new(
            None,
            false,
            Some(gate.clone()),
            Some(started_tx),
        ));
        let store = ProbeStore::new(state.clone());
        let path = Path::from("bounded-concurrency.bin");
        let artifact = SpooledObject::from_file(anonymous_file(&body), body.len() as u64);
        let task = tokio::spawn(async move {
            put_spooled_object_with_limits(
                store,
                &path,
                artifact,
                part_size,
                MULTIPART_MAX_CONCURRENCY,
            )
            .await
        });

        for _ in 0..MULTIPART_MAX_CONCURRENCY {
            tokio::time::timeout(Duration::from_secs(2), started_rx.recv())
                .await
                .expect("multipart part did not start")
                .expect("multipart start channel closed");
        }
        tokio::task::yield_now().await;
        assert_eq!(
            state.maximum_active_parts.load(Ordering::SeqCst),
            MULTIPART_MAX_CONCURRENCY
        );
        assert!(
            started_rx.try_recv().is_err(),
            "a ninth part started above the configured cap"
        );

        gate.add_permits(parts + 1);
        task.await.unwrap().unwrap();
        assert!(state.maximum_active_parts.load(Ordering::SeqCst) <= MULTIPART_MAX_CONCURRENCY);
        let lengths = state.part_lengths.lock().unwrap();
        assert!(lengths[..parts].iter().all(|len| *len == part_size));
        assert_eq!(lengths[parts], 7);
    }

    #[tokio::test]
    async fn truncated_and_extra_large_spools_are_rejected_and_aborted() {
        let part_size = 64 * 1024;
        let expected_len = part_size * 2;
        for (name, actual_len) in [
            ("truncated-spool.bin", expected_len - 1),
            ("extra-spool.bin", expected_len + 1),
        ] {
            let state = Arc::new(ProbeState::new(None, false, None, None));
            let store = ProbeStore::new(state.clone());
            let artifact = SpooledObject::from_file(
                anonymous_file(&patterned_body(actual_len)),
                expected_len as u64,
            );
            let error =
                put_spooled_object_with_limits(store, &Path::from(name), artifact, part_size, 2)
                    .await
                    .unwrap_err();
            assert!(matches!(error, Error::Invariant(_)), "{error:?}");
            assert_eq!(state.multipart_started.load(Ordering::SeqCst), 1);
            assert_eq!(state.aborts.load(Ordering::SeqCst), 1);
            assert_eq!(state.completes.load(Ordering::SeqCst), 0);
        }
    }

    #[tokio::test]
    async fn part_and_completion_failures_abort_the_upload() {
        let part_size = 64 * 1024;
        let body = patterned_body(part_size * 4 + 9);
        for (fail_part, fail_complete) in [(Some(1), false), (None, true)] {
            let state = Arc::new(ProbeState::new(fail_part, fail_complete, None, None));
            let store = ProbeStore::new(state.clone());
            let path = Path::from(if fail_part.is_some() {
                "part-failure.bin"
            } else {
                "completion-failure.bin"
            });
            let artifact = SpooledObject::from_file(anonymous_file(&body), body.len() as u64);
            assert!(
                put_spooled_object_with_limits(store, &path, artifact, part_size, 2)
                    .await
                    .is_err()
            );
            assert_eq!(state.aborts.load(Ordering::SeqCst), 1);
            if fail_complete {
                assert_eq!(state.completes.load(Ordering::SeqCst), 1);
            }
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn cancellation_aborts_multipart_and_removes_named_spool() {
        let part_size = 64 * 1024;
        let body = patterned_body(part_size * 3);
        let mut named = NamedTempFile::new().unwrap();
        named.write_all(&body).unwrap();
        let spool_path = named.path().to_owned();
        let gate = Arc::new(Semaphore::new(0));
        let (started_tx, mut started_rx) = mpsc::unbounded_channel();
        let state = Arc::new(ProbeState::new(None, false, Some(gate), Some(started_tx)));
        let store = ProbeStore::new(state.clone());
        let artifact = SpooledObject::from_named_file(named, body.len() as u64);
        let task = tokio::spawn(async move {
            let path = Path::from("cancelled-spool.bin");
            put_spooled_object_with_limits(store, &path, artifact, part_size, 2).await
        });

        tokio::time::timeout(Duration::from_secs(2), started_rx.recv())
            .await
            .expect("multipart part did not start")
            .expect("multipart start channel closed");
        assert!(spool_path.exists());
        let aborted = state.abort_notify.notified();
        task.abort();
        assert!(task.await.unwrap_err().is_cancelled());
        tokio::time::timeout(Duration::from_secs(2), aborted)
            .await
            .expect("multipart cancellation did not abort");
        for _ in 0..20 {
            if !spool_path.exists() {
                break;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        assert!(
            !spool_path.exists(),
            "named spool leaked after cancellation"
        );
        assert_eq!(state.aborts.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn cancelled_edge_point_upload_aborts_while_owning_its_value_spool() {
        let mut builder = crate::sst::paged_index::EdgePointIndexBuilder::new();
        for n in 0..32_u128 {
            let value = patterned_body(32 * 1024 + n as usize);
            builder
                .push(&(n * 2).to_be_bytes(), &(n * 2 + 1).to_be_bytes(), &value)
                .unwrap();
        }
        let upload = builder.finish_upload().unwrap();
        assert!(upload.spooled_value_bytes() > 1024 * 1024);
        let exact_len = upload.size_bytes();
        let (files, described_len) = upload.into_files();
        assert_eq!(files.len(), 2, "page and value spools stay independent");
        assert_eq!(described_len, exact_len);
        let artifact = SpooledObject::from_files(Vec::new(), files, exact_len);

        let gate = Arc::new(Semaphore::new(0));
        let (started_tx, mut started_rx) = mpsc::unbounded_channel();
        let state = Arc::new(ProbeState::new(None, false, Some(gate), Some(started_tx)));
        let store: Arc<dyn ObjectStore> = ProbeStore::new(state.clone());
        let task = tokio::spawn(async move {
            put_spooled_object_with_limits(
                store,
                &Path::from("cancelled-edges.epidx"),
                artifact,
                64 * 1024,
                2,
            )
            .await
        });

        tokio::time::timeout(Duration::from_secs(2), started_rx.recv())
            .await
            .expect("edge point multipart part did not start")
            .expect("edge point multipart start channel closed");
        let aborted = state.abort_notify.notified();
        task.abort();
        assert!(task.await.unwrap_err().is_cancelled());
        tokio::time::timeout(Duration::from_secs(2), aborted)
            .await
            .expect("edge point cancellation did not abort multipart");
        assert_eq!(state.aborts.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn cancelled_edge_body_upload_aborts_while_owning_all_section_spools() {
        use crate::sst::edges::writer::{EdgeRecord, EdgeSstWriter, EdgeSstWriterOptions};
        use crate::sst::edges::EdgeDirection;

        let mut writer = EdgeSstWriter::new(EdgeSstWriterOptions::new(
            EdgeDirection::Inverse,
            "CITES",
            "Articulo",
            "Articulo",
        ));
        for ordinal in 0..10_000u128 {
            writer
                .append(EdgeRecord {
                    key_id: ordinal.to_be_bytes(),
                    partner_id: (ordinal + 10_000).to_be_bytes(),
                    lsn: ordinal as u64 + 1,
                    tombstone: false,
                    declared_properties: Vec::new(),
                    overflow_json: None,
                })
                .unwrap();
        }
        let build = writer.finish_with_point_index().unwrap();
        let artifact = build.body.into_spooled_object();
        assert!(artifact.len() > 64 * 1024);

        let gate = Arc::new(Semaphore::new(0));
        let (started_tx, mut started_rx) = mpsc::unbounded_channel();
        let state = Arc::new(ProbeState::new(None, false, Some(gate), Some(started_tx)));
        let store: Arc<dyn ObjectStore> = ProbeStore::new(state.clone());
        let task = tokio::spawn(async move {
            put_spooled_object_with_limits(
                store,
                &Path::from("cancelled-edge-body.csr"),
                artifact,
                64 * 1024,
                2,
            )
            .await
        });

        tokio::time::timeout(Duration::from_secs(2), started_rx.recv())
            .await
            .expect("edge-body multipart part did not start")
            .expect("edge-body multipart start channel closed");
        let aborted = state.abort_notify.notified();
        task.abort();
        assert!(task.await.unwrap_err().is_cancelled());
        tokio::time::timeout(Duration::from_secs(2), aborted)
            .await
            .expect("edge-body cancellation did not abort multipart");
        assert_eq!(state.aborts.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn dropping_owned_named_spool_removes_it() {
        let mut named = NamedTempFile::new().unwrap();
        named.write_all(b"owned temporary artifact").unwrap();
        let path = named.path().to_owned();
        let artifact = SpooledObject::from_named_file(named, 24);
        assert!(path.exists());
        assert_eq!(artifact.len(), 24);
        drop(artifact);
        assert!(!path.exists());
    }
}
