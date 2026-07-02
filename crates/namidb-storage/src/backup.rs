//! Consistent namespace snapshot copy: backup and restore.
//!
//! A namespace's durable state at a manifest version is the manifest body for
//! that version plus every object it references: the SST bodies and their
//! bloom / unique / equality / label-index side-cars, and the WAL segments
//! still needed for recovery. Every one of those objects is immutable once
//! written (compaction and flush only ever add new objects; the orphan sweep
//! only deletes ones no retained manifest references). So copying the set a
//! pinned manifest names is **consistent by construction**: the pinned version
//! can neither gain nor lose a referenced object while the copy runs.
//!
//! [`copy_namespace_snapshot`] is the one primitive behind both directions.
//! Backup copies a live namespace into a fresh location; restore copies a
//! backup back over a target. They are the same operation — pin a version,
//! copy its closure, write the pointer last — so they share an implementation.
//!
//! The destination is left as a self-contained namespace renumbered to
//! version 0 (a fresh epoch), exactly as a freshly bootstrapped namespace that
//! then ingested the data would look. That keeps the restored namespace from
//! carrying dangling references to manifest versions that were never copied,
//! which the orphan sweep's retention horizon would otherwise try to load.
//!
//! While the copy runs it holds a **retention pin lease** on the source
//! (`manifest/pins/<uuid>.json`, [`crate::pin`]): the orphan sweep unions
//! every unexpired lease into its horizon, so a concurrent compaction plus
//! sweep cannot delete the pinned closure mid-copy. The lease is renewed as
//! the copy progresses and released when it finishes (or errors); a crashed
//! copy leaks a lease that simply expires. Residual window: a sweep already
//! past its own pin listing when the lease lands can still reclaim a version
//! *older than every in-process reader*; the copy then fails loudly with
//! `NotFound` rather than producing a truncated snapshot — retry it.

use std::sync::Arc;

use futures::TryStreamExt;
use object_store::path::Path;
use object_store::{ObjectStore, ObjectStoreExt, PutMode, PutOptions, PutPayload, WriteMultipart};
use tracing::debug;

use crate::error::{Error, Result};
use crate::fence::Epoch;
use crate::manifest::{Manifest, ManifestPointer, ManifestStore};
use crate::paths::NamespacePaths;
use crate::pin::{RetentionPin, DEFAULT_PIN_TTL};

/// Outcome of a completed [`copy_namespace_snapshot`].
#[derive(Debug, Clone)]
pub struct SnapshotCopyReport {
    /// The source manifest version that was copied (before renumbering).
    pub source_version: u64,
    /// Objects written to the destination, including the manifest body and
    /// the pointer.
    pub objects_copied: usize,
    /// Total bytes read from the source and written to the destination.
    pub bytes_copied: u64,
}

/// Copy a consistent snapshot of the namespace at `(src_store, src_paths)`
/// into `(dst_store, dst_paths)`, pinned to manifest `version` (or the current
/// committed version when `None`).
///
/// The destination is left openable: its `current.json` pointer is written
/// **last**, so an interrupted copy leaves an un-pointed (and therefore
/// ignored) set of objects rather than a half-written namespace. The snapshot
/// is renumbered to version 0 with a fresh epoch, so the destination is a
/// self-contained namespace.
///
/// `overwrite` guards the destination: `false` (the default for a backup)
/// refuses when the destination already holds a `current.json`, so a backup
/// can never clobber a live namespace; `true` (a restore over a corrupted or
/// stale namespace) proceeds. The guard is a best-effort `head` check, not a
/// lock: a restore must run against an offline destination, since there is no
/// fencing against a concurrent writer (or a concurrent restore) here. When
/// `overwrite` replaces a populated destination, objects the prior namespace
/// owned (its SST bodies and older manifest versions) are left as orphans the
/// new manifest does not reference; prefer restoring into a fresh location, or
/// run the orphan sweep afterwards to reclaim the space.
///
/// The copy holds a retention pin lease on the source for its whole duration
/// (see the module docs), so it is safe against a concurrent compaction plus
/// orphan sweep — the janitor keeps the pinned closure alive until the lease
/// is released or expires. Restore reads from a backup destination, where no
/// janitor runs; the lease it writes there is inert and removed on completion.
pub async fn copy_namespace_snapshot(
    src_store: Arc<dyn ObjectStore>,
    src_paths: NamespacePaths,
    dst_store: Arc<dyn ObjectStore>,
    dst_paths: NamespacePaths,
    version: Option<u64>,
    overwrite: bool,
    verify: bool,
) -> Result<SnapshotCopyReport> {
    // Refuse to stomp a live destination unless explicitly told to. A live
    // namespace is identified by its pointer: the Create-only family
    // (`manifest/pointer/p<N>.json`, RFC-029) for current namespaces, or the
    // legacy `manifest/current.json` for ones bootstrapped before it.
    if !overwrite {
        let has_family = {
            let mut s = dst_store.list(Some(&dst_paths.pointer_dir()));
            s.try_next().await.map_err(Error::ObjectStore)?.is_some()
        };
        let has_legacy = dst_store.head(&dst_paths.current_pointer()).await.is_ok();
        if has_family || has_legacy {
            return Err(Error::precondition(format!(
                "destination namespace '{}' already has a manifest pointer — pass overwrite/--force to replace it",
                dst_paths.namespace()
            )));
        }
    }

    // Pin the manifest version to copy.
    let src_manifests = ManifestStore::new(src_store.clone(), src_paths.clone());
    let manifest: Manifest = match version {
        Some(v) => src_manifests.load_manifest_at(v).await?,
        None => src_manifests.load_current().await?.manifest,
    };
    let source_version = manifest.version;

    // Make the pin durable: a lease object the source's orphan sweep unions
    // into its horizon, so the pinned closure stays alive across processes
    // while the copy runs.
    let mut pin = RetentionPin::acquire(
        src_store.clone(),
        &src_paths,
        source_version,
        DEFAULT_PIN_TTL,
    )
    .await?;
    // Close the load-then-pin race: a sweep could have reclaimed the pinned
    // version between our load and the lease becoming visible. Once the body
    // is confirmed present *after* the lease exists, every later sweep (which
    // lists pins before deleting) retains the whole closure.
    if let Err(e) = src_store
        .head(&src_paths.manifest_version(source_version))
        .await
    {
        let _ = pin.release().await;
        return Err(Error::precondition(format!(
            "source manifest version {source_version} was reclaimed while the retention pin was \
             being acquired — retry the copy: {e}"
        )));
    }

    let result = copy_snapshot_pinned(
        &src_store, &src_paths, &dst_store, &dst_paths, manifest, overwrite, verify, &mut pin,
    )
    .await;
    if let Err(e) = pin.release().await {
        // Benign: the lease expires on its own; the next sweep reclaims it.
        debug!(error = %e, "failed to release the backup retention pin");
    }
    result
}

/// The copy body, run while `pin` holds the source's retention horizon. The
/// lease is renewed as objects are copied (a cheap elapsed-time check per
/// object) so a long copy — or one large object streaming for a while —
/// cannot outlive it.
#[allow(clippy::too_many_arguments)]
async fn copy_snapshot_pinned(
    src_store: &Arc<dyn ObjectStore>,
    src_paths: &NamespacePaths,
    dst_store: &Arc<dyn ObjectStore>,
    dst_paths: &NamespacePaths,
    mut manifest: Manifest,
    overwrite: bool,
    verify: bool,
    pin: &mut RetentionPin,
) -> Result<SnapshotCopyReport> {
    let source_version = manifest.version;
    let mut objects_copied = 0usize;
    let mut bytes_copied = 0u64;

    let src_prefix = src_paths.namespace_prefix();
    let dst_prefix = dst_paths.namespace_prefix();

    // We copy exactly the manifest's referenced closure and nothing else. In
    // particular `memtable_snapshot.bin` is deliberately skipped: it is not
    // manifest-referenced and reflects the live memtable, so a stale copy would
    // make recovery skip WAL records past its floor (silent data loss). The
    // restored namespace simply writes a fresh one on its next flush.

    // 1. SST bodies and their side-cars, addressed by their relative path
    //    (`<prefix>/<relative>`), which is identical at source and destination.
    for sst in &manifest.ssts {
        let mut rels: Vec<&str> = vec![sst.path.as_str()];
        if let Some(b) = &sst.bloom {
            rels.push(b.path.as_str());
        }
        for u in &sst.unique_property_indices {
            rels.push(u.path.as_str());
        }
        for e in &sst.equality_property_indices {
            rels.push(e.path.as_str());
        }
        if let Some(l) = &sst.label_index {
            rels.push(l.path.as_str());
        }
        for rel in rels {
            let from = Path::from(format!("{}/{}", src_prefix.as_ref(), rel));
            let to = Path::from(format!("{}/{}", dst_prefix.as_ref(), rel));
            bytes_copied += copy_object(src_store, dst_store, &from, &to).await?;
            objects_copied += 1;
            pin.renew_if_due().await?;
        }
    }

    // 2. WAL segments still needed for recovery, addressed by seq (the same
    //    canonical key the recovery path reads them back through).
    for seg in &manifest.wal_segments {
        let from = src_paths.wal_segment(seg.seq);
        let to = dst_paths.wal_segment(seg.seq);
        bytes_copied += copy_object(src_store, dst_store, &from, &to).await?;
        objects_copied += 1;
        pin.renew_if_due().await?;
    }

    // 3. The manifest body, renumbered to a self-contained version 0 / fresh
    //    epoch. Its SST paths are relative, so it transplants unchanged apart
    //    from the version and epoch fields. When NOT overwriting, publish via
    //    `PutMode::Create` (If-None-Match:*) so a concurrent backup that raced
    //    past the pre-check is caught at the linearization point instead of
    //    silently stomping (closing the head()-pre-check TOCTOU).
    manifest.version = 0;
    manifest.epoch = Epoch::ZERO;
    let manifest_path = dst_paths.manifest_version(0);
    let manifest_bytes = serde_json::to_vec(&manifest)?;
    bytes_copied += manifest_bytes.len() as u64;
    let publish_create = !overwrite;
    dst_store
        .put_opts(
            &manifest_path,
            PutPayload::from(manifest_bytes),
            PutOptions::from(if publish_create {
                PutMode::Create
            } else {
                PutMode::Overwrite
            }),
        )
        .await
        .map_err(|e| match e {
            object_store::Error::AlreadyExists { .. } => Error::precondition(format!(
                "destination namespace '{}' was written concurrently — retry or pass overwrite/--force",
                dst_paths.namespace()
            )),
            other => Error::ObjectStore(other),
        })?;
    objects_copied += 1;

    // 4. The pointer LAST, so the destination commits atomically from a
    //    reader's view. Its `manifest_path` is absolute, so it is rebased onto
    //    the destination's own key. RFC-029: written into the Create-only
    //    family as `pointer/p0.json`. On an `overwrite` restore, clear any
    //    existing family first so the renumbered version-0 pointer is the
    //    authoritative maximum (a leftover higher `p<N>` would otherwise shadow
    //    the restore). Also delete the legacy `current.json` if present to avoid
    //    leaving a stale pointer that would shadow the new family.
    if overwrite {
        let mut leftovers = Vec::new();
        let mut s = dst_store.list(Some(&dst_paths.pointer_dir()));
        while let Some(meta) = s.try_next().await.map_err(Error::ObjectStore)? {
            leftovers.push(meta.location);
        }
        for loc in leftovers {
            dst_store.delete(&loc).await.map_err(Error::ObjectStore)?;
        }
        // Delete legacy current.json if present — it would shadow the new
        // pointer family on load (legacy fallback takes precedence when both exist).
        if dst_store.head(&dst_paths.current_pointer()).await.is_ok() {
            dst_store
                .delete(&dst_paths.current_pointer())
                .await
                .map_err(Error::ObjectStore)?;
        }
        // Delete leftover write-once manifest version bodies above the restored
        // v0. They would otherwise collide with the reopened writer's
        // PutMode::Create on v1..vN → ManifestCommitCas → the restored namespace
        // could never accept a write again (permanently bricked). Keep the v0
        // body we just published; skip the pointer/ subdir and current.json.
        let v0_str = dst_paths.manifest_version(0).as_ref().to_string();
        let mut stale_bodies = Vec::new();
        let mut ms = dst_store.list(Some(&dst_paths.manifest_dir()));
        while let Some(meta) = ms.try_next().await.map_err(Error::ObjectStore)? {
            let s = meta.location.as_ref();
            let is_version_body = !s.contains("/pointer/")
                && s.rsplit('/')
                    .next()
                    .map(|f| f.starts_with('v') && f.ends_with(".json"))
                    .unwrap_or(false);
            if is_version_body && s != v0_str {
                stale_bodies.push(meta.location);
            }
        }
        for loc in stale_bodies {
            dst_store.delete(&loc).await.map_err(Error::ObjectStore)?;
        }
        // Delete the destination's stale memtable snapshot: recovery trusts it
        // unconditionally, so leaving it would resurrect pre-restore rows and
        // drop the restored WAL tail.
        if dst_store.head(&dst_paths.memtable_snapshot()).await.is_ok() {
            dst_store
                .delete(&dst_paths.memtable_snapshot())
                .await
                .map_err(Error::ObjectStore)?;
        }
    }
    let pointer = ManifestPointer {
        version: 0,
        epoch: Epoch::ZERO,
        manifest_path: manifest_path.as_ref().to_string(),
    };
    let pointer_bytes = serde_json::to_vec(&pointer)?;
    bytes_copied += pointer_bytes.len() as u64;
    dst_store
        .put_opts(
            &dst_paths.pointer_version(0),
            PutPayload::from(pointer_bytes.clone()),
            PutOptions::from(if publish_create {
                PutMode::Create
            } else {
                PutMode::Overwrite
            }),
        )
        .await
        .map_err(|e| match e {
            object_store::Error::AlreadyExists { .. } => Error::precondition(format!(
                "destination namespace '{}' was written concurrently — retry or pass overwrite/--force",
                dst_paths.namespace()
            )),
            other => Error::ObjectStore(other),
        })?;
    objects_copied += 1;

    // Also publish the advisory `current.json` (see manifest.rs) so the
    // restored namespace is findable via a non-LIST read on EC stores.
    dst_store
        .put(
            &dst_paths.current_pointer(),
            PutPayload::from(pointer_bytes),
        )
        .await
        .map_err(Error::ObjectStore)?;
    objects_copied += 1;

    // Optional post-copy consistency verify: re-open the destination and HEAD
    // every manifest-referenced object to confirm it landed with a non-zero
    // size. Catches a partial copy (a dropped/short write) before the caller
    // trusts the snapshot.
    if verify {
        verify_snapshot(dst_store, dst_paths, &manifest).await?;
    }

    Ok(SnapshotCopyReport {
        source_version,
        objects_copied,
        bytes_copied,
    })
}

/// Re-open the destination and HEAD every SST body, side-car, and WAL segment
/// the (renumbered) manifest references, failing if any is missing or
/// empty. A best-effort integrity gate for `--verify`.
async fn verify_snapshot(
    store: &Arc<dyn ObjectStore>,
    paths: &NamespacePaths,
    manifest: &Manifest,
) -> Result<()> {
    let prefix = paths.namespace_prefix();
    for sst in &manifest.ssts {
        let rels: Vec<&str> = std::iter::once(sst.path.as_str())
            .chain(sst.bloom.as_ref().map(|b| b.path.as_str()))
            .chain(sst.unique_property_indices.iter().map(|u| u.path.as_str()))
            .chain(
                sst.equality_property_indices
                    .iter()
                    .map(|e| e.path.as_str()),
            )
            .chain(sst.label_index.as_ref().map(|l| l.path.as_str()))
            .collect();
        for rel in rels {
            let p = Path::from(format!("{}/{}", prefix.as_ref(), rel));
            let meta = store.head(&p).await.map_err(|e| {
                Error::precondition(format!("verify: missing SST object {rel}: {e}"))
            })?;
            if meta.size == 0 {
                return Err(Error::precondition(format!(
                    "verify: SST object {rel} is empty (0 bytes)"
                )));
            }
        }
    }
    for seg in &manifest.wal_segments {
        let p = paths.wal_segment(seg.seq);
        let meta = store.head(&p).await.map_err(|e| {
            Error::precondition(format!("verify: missing WAL segment {}: {e}", seg.seq))
        })?;
        if meta.size == 0 {
            return Err(Error::precondition(format!(
                "verify: WAL segment {} is empty",
                seg.seq
            )));
        }
    }
    Ok(())
}

/// Part size for the streaming copy path. Objects at or below one part are
/// copied with a plain buffered PUT (the common case: manifests, pointers,
/// small SSTs); larger ones stream through a multipart upload in parts of
/// this size. 8 MiB clears every cloud store's minimum non-final part size
/// (S3: 5 MiB) and, with 10k parts, bounds a single object at 80 GB.
const COPY_PART_SIZE: usize = 8 * 1024 * 1024;

/// Cap on multipart parts concurrently in flight, which bounds the copy's
/// memory at roughly `COPY_MAX_IN_FLIGHT_PARTS * COPY_PART_SIZE` per object.
const COPY_MAX_IN_FLIGHT_PARTS: usize = 4;

/// Stream one object from `src` to `dst`, returning its byte length. Plain
/// GET + PUT so it works across backends (s3 -> file, file -> gs, ...); a
/// same-store fast path via `ObjectStore::copy` is a later optimisation.
async fn copy_object(
    src: &Arc<dyn ObjectStore>,
    dst: &Arc<dyn ObjectStore>,
    from: &Path,
    to: &Path,
) -> Result<u64> {
    copy_object_with_part_size(src, dst, from, to, COPY_PART_SIZE).await
}

/// [`copy_object`] with the part-size threshold explicit so tests can force
/// the multipart path without allocating multi-MiB fixtures. Never buffers
/// more than one part (plus the bounded in-flight uploads) regardless of the
/// object's size, so a multi-GB compacted SST cannot OOM the process.
async fn copy_object_with_part_size(
    src: &Arc<dyn ObjectStore>,
    dst: &Arc<dyn ObjectStore>,
    from: &Path,
    to: &Path,
    part_size: usize,
) -> Result<u64> {
    let result = src.get(from).await?;
    if result.meta.size <= part_size as u64 {
        let bytes = result.bytes().await?;
        let len = bytes.len() as u64;
        dst.put_opts(
            to,
            PutPayload::from(bytes),
            PutOptions::from(PutMode::Overwrite),
        )
        .await?;
        return Ok(len);
    }

    let upload = dst.put_multipart(to).await.map_err(Error::ObjectStore)?;
    let mut write = WriteMultipart::new_with_chunk_size(upload, part_size);
    let mut stream = result.into_stream();
    let streamed = async {
        let mut len = 0u64;
        while let Some(chunk) = stream.try_next().await.map_err(Error::ObjectStore)? {
            len += chunk.len() as u64;
            // Backpressure: don't buffer the source faster than the parts
            // upload, or the "streaming" copy degrades into buffering.
            write
                .wait_for_capacity(COPY_MAX_IN_FLIGHT_PARTS)
                .await
                .map_err(Error::ObjectStore)?;
            write.put(chunk);
        }
        Ok::<u64, Error>(len)
    }
    .await;
    match streamed {
        Ok(len) => {
            write.finish().await.map_err(Error::ObjectStore)?;
            Ok(len)
        }
        Err(e) => {
            // Best effort: reclaim already-uploaded parts. The error we
            // surface is the copy failure, not the abort's.
            if let Err(abort_err) = write.abort().await {
                debug!(error = %abort_err, "failed to abort a partial multipart copy");
            }
            Err(e)
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use namidb_core::id::{NamespaceId, NodeId};
    use namidb_core::schema::{DataType, LabelDef, PropertyDef, Schema, SchemaBuilder};
    use namidb_core::value::Value;
    use object_store::memory::InMemory;

    use super::*;
    use crate::flush::NodeWriteRecord;
    use crate::ingest::WriterSession;
    use crate::manifest::ManifestStore;

    /// Delegating store that (a) counts multipart uploads started against it
    /// and (b), when `pins_prefix` is set, records for every GET of a data
    /// object (SST body, side-car, WAL segment) whether a retention pin lease
    /// existed at that moment. (b) proves the copy actually holds its pin
    /// while reading — equal results alone would pass even without the pin.
    #[derive(Debug)]
    struct ProbeStore {
        inner: Arc<dyn ObjectStore>,
        pins_prefix: Option<Path>,
        data_reads: AtomicUsize,
        unpinned_data_reads: AtomicUsize,
        multipart_uploads: AtomicUsize,
    }

    impl ProbeStore {
        fn new(inner: Arc<dyn ObjectStore>, pins_prefix: Option<Path>) -> Self {
            Self {
                inner,
                pins_prefix,
                data_reads: AtomicUsize::new(0),
                unpinned_data_reads: AtomicUsize::new(0),
                multipart_uploads: AtomicUsize::new(0),
            }
        }
    }

    impl std::fmt::Display for ProbeStore {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "ProbeStore({})", self.inner)
        }
    }

    #[async_trait::async_trait]
    impl ObjectStore for ProbeStore {
        async fn put_opts(
            &self,
            location: &Path,
            payload: PutPayload,
            opts: PutOptions,
        ) -> object_store::Result<object_store::PutResult> {
            self.inner.put_opts(location, payload, opts).await
        }

        async fn put_multipart_opts(
            &self,
            location: &Path,
            opts: object_store::PutMultipartOptions,
        ) -> object_store::Result<Box<dyn object_store::MultipartUpload>> {
            self.multipart_uploads.fetch_add(1, Ordering::SeqCst);
            self.inner.put_multipart_opts(location, opts).await
        }

        async fn get_opts(
            &self,
            location: &Path,
            options: object_store::GetOptions,
        ) -> object_store::Result<object_store::GetResult> {
            if let Some(pins) = &self.pins_prefix {
                let key = location.as_ref();
                if key.contains("/sst/") || key.contains("/wal/") {
                    self.data_reads.fetch_add(1, Ordering::SeqCst);
                    let lease_present = self.inner.list(Some(pins)).try_next().await?.is_some();
                    if !lease_present {
                        self.unpinned_data_reads.fetch_add(1, Ordering::SeqCst);
                    }
                }
            }
            self.inner.get_opts(location, options).await
        }

        fn list(
            &self,
            prefix: Option<&Path>,
        ) -> futures::stream::BoxStream<'static, object_store::Result<object_store::ObjectMeta>>
        {
            self.inner.list(prefix)
        }

        async fn list_with_delimiter(
            &self,
            prefix: Option<&Path>,
        ) -> object_store::Result<object_store::ListResult> {
            self.inner.list_with_delimiter(prefix).await
        }

        async fn copy_opts(
            &self,
            from: &Path,
            to: &Path,
            options: object_store::CopyOptions,
        ) -> object_store::Result<()> {
            self.inner.copy_opts(from, to, options).await
        }

        fn delete_stream(
            &self,
            locations: futures::stream::BoxStream<
                'static,
                object_store::Result<object_store::path::Path>,
            >,
        ) -> futures::stream::BoxStream<'static, object_store::Result<object_store::path::Path>>
        {
            self.inner.delete_stream(locations)
        }
    }

    fn store() -> Arc<dyn ObjectStore> {
        Arc::new(InMemory::new())
    }

    fn paths(ns: &str) -> NamespacePaths {
        NamespacePaths::new("tenants", NamespaceId::new(ns).unwrap())
    }

    fn schema() -> Schema {
        SchemaBuilder::new()
            .label(LabelDef {
                name: "Person".into(),
                properties: vec![PropertyDef::new("name", DataType::Utf8, true).unwrap()],
            })
            .unwrap()
            .build()
    }

    fn person(name: &str) -> NodeWriteRecord {
        let mut props = BTreeMap::new();
        props.insert("name".into(), Value::Str(name.into()));
        NodeWriteRecord {
            properties: props,
            schema_version: 1,
            ..Default::default()
        }
    }

    async fn names_in(store: Arc<dyn ObjectStore>, paths: NamespacePaths) -> Vec<String> {
        let session = WriterSession::open(store, paths).await.unwrap();
        let snap = session.snapshot();
        let mut names: Vec<String> = snap
            .scan_label("Person")
            .await
            .unwrap()
            .iter()
            .filter_map(|n| match n.properties.get("name") {
                Some(Value::Str(s)) => Some(s.clone()),
                _ => None,
            })
            .collect();
        names.sort();
        names
    }

    #[tokio::test]
    async fn snapshot_round_trips_flushed_ssts_and_unflushed_wal() {
        let (src_store, src_paths) = (store(), paths("bk-src"));
        {
            let mut w = WriterSession::open(src_store.clone(), src_paths.clone())
                .await
                .unwrap();
            // Two nodes committed and flushed: they live in SSTs.
            w.upsert_node("Person", NodeId::new(), &person("Ada"))
                .unwrap();
            w.upsert_node("Person", NodeId::new(), &person("Grace"))
                .unwrap();
            w.commit_batch().await.unwrap();
            w.flush(schema()).await.unwrap();
            // A third node committed but NOT flushed: it lives in a WAL
            // segment the manifest still references.
            w.upsert_node("Person", NodeId::new(), &person("Lin"))
                .unwrap();
            w.commit_batch().await.unwrap();
        }

        // Back up to a fresh, separate store + namespace.
        let (dst_store, dst_paths) = (store(), paths("bk-dst"));
        let report = copy_namespace_snapshot(
            src_store,
            src_paths,
            dst_store.clone(),
            dst_paths.clone(),
            None,
            false,
            false,
        )
        .await
        .unwrap();
        assert!(report.objects_copied >= 1);

        // Opening the destination sees every node — SST-backed and WAL-backed.
        assert_eq!(
            names_in(dst_store, dst_paths).await,
            vec!["Ada", "Grace", "Lin"]
        );
    }

    #[tokio::test]
    async fn restored_snapshot_is_a_self_contained_version_zero() {
        let (src_store, src_paths) = (store(), paths("bk-ver-src"));
        {
            let mut w = WriterSession::open(src_store.clone(), src_paths.clone())
                .await
                .unwrap();
            // Several commits + a flush, so the source manifest version is > 0.
            for n in ["a", "b", "c"] {
                w.upsert_node("Person", NodeId::new(), &person(n)).unwrap();
                w.commit_batch().await.unwrap();
            }
            w.flush(schema()).await.unwrap();
            assert!(w.manifest_version() > 0, "source should be past version 0");
        }

        let (dst_store, dst_paths) = (store(), paths("bk-ver-dst"));
        let report = copy_namespace_snapshot(
            src_store,
            src_paths,
            dst_store.clone(),
            dst_paths.clone(),
            None,
            false,
            false,
        )
        .await
        .unwrap();
        assert!(report.source_version > 0);

        // On disk the restored manifest is renumbered to a self-contained
        // version 0, with no dangling references to versions never copied.
        let on_disk = ManifestStore::new(dst_store.clone(), dst_paths.clone())
            .load_current()
            .await
            .unwrap();
        assert_eq!(on_disk.manifest.version, 0);

        // And it opens cleanly (the writer then claims it and advances on).
        WriterSession::open(dst_store, dst_paths).await.unwrap();
    }

    #[tokio::test]
    async fn refuses_to_clobber_a_live_destination_without_overwrite() {
        let (src_store, src_paths) = (store(), paths("bk-guard-src"));
        {
            let mut w = WriterSession::open(src_store.clone(), src_paths.clone())
                .await
                .unwrap();
            w.upsert_node("Person", NodeId::new(), &person("Ada"))
                .unwrap();
            w.commit_batch().await.unwrap();
            w.flush(schema()).await.unwrap();
        }
        let (dst_store, dst_paths) = (store(), paths("bk-guard-dst"));

        // First copy into a fresh destination succeeds.
        copy_namespace_snapshot(
            src_store.clone(),
            src_paths.clone(),
            dst_store.clone(),
            dst_paths.clone(),
            None,
            false,
            false,
        )
        .await
        .unwrap();

        // A second copy without overwrite is refused — the destination is live.
        let err = copy_namespace_snapshot(
            src_store.clone(),
            src_paths.clone(),
            dst_store.clone(),
            dst_paths.clone(),
            None,
            false,
            false,
        )
        .await
        .expect_err("must refuse to clobber a live destination");
        assert!(matches!(err, Error::Precondition(_)), "got {err:?}");

        // With overwrite, it proceeds.
        copy_namespace_snapshot(
            src_store, src_paths, dst_store, dst_paths, None, true, false,
        )
        .await
        .unwrap();
    }

    #[tokio::test]
    async fn snapshot_copies_every_side_car() {
        let (src_store, src_paths) = (store(), paths("bk-side-src"));
        let sc = SchemaBuilder::new()
            .label(LabelDef {
                name: "Person".into(),
                properties: vec![
                    PropertyDef::new("email", DataType::Utf8, true)
                        .unwrap()
                        .with_unique(true),
                    PropertyDef::new("city", DataType::Utf8, true)
                        .unwrap()
                        .with_indexed(true),
                ],
            })
            .unwrap()
            .label(LabelDef {
                name: "Admin".into(),
                properties: vec![],
            })
            .unwrap()
            .build();
        {
            let mut w = WriterSession::open(src_store.clone(), src_paths.clone())
                .await
                .unwrap();
            for (email, city) in [("a@x", "NYC"), ("b@x", "LA"), ("c@x", "SF")] {
                let mut props = BTreeMap::new();
                props.insert("email".into(), Value::Str(email.into()));
                props.insert("city".into(), Value::Str(city.into()));
                let rec = NodeWriteRecord {
                    properties: props,
                    schema_version: 1,
                    ..Default::default()
                };
                // Multi-label nodes so the flush emits a label-index side-car
                // alongside the unique (email) and equality (city) ones.
                w.upsert_node_with_labels(
                    ["Person".to_string(), "Admin".to_string()],
                    NodeId::new(),
                    &rec,
                )
                .unwrap();
            }
            w.commit_batch().await.unwrap();
            w.flush(sc).await.unwrap();
        }

        // Enumerate the side-car objects the source flush produced.
        let manifest = ManifestStore::new(src_store.clone(), src_paths.clone())
            .load_current()
            .await
            .unwrap()
            .manifest;
        let mut sidecars: Vec<String> = Vec::new();
        for sst in &manifest.ssts {
            if let Some(b) = &sst.bloom {
                sidecars.push(b.path.clone());
            }
            for u in &sst.unique_property_indices {
                sidecars.push(u.path.clone());
            }
            for e in &sst.equality_property_indices {
                sidecars.push(e.path.clone());
            }
            if let Some(l) = &sst.label_index {
                sidecars.push(l.path.clone());
            }
        }
        assert!(
            !sidecars.is_empty(),
            "a unique + indexed + multi-label schema should emit side-cars, got none"
        );

        let (dst_store, dst_paths) = (store(), paths("bk-side-dst"));
        copy_namespace_snapshot(
            src_store,
            src_paths,
            dst_store.clone(),
            dst_paths.clone(),
            None,
            false,
            false,
        )
        .await
        .unwrap();

        // Every side-car the manifest named exists at the destination.
        let dst_prefix = dst_paths.namespace_prefix();
        for rel in &sidecars {
            let key = Path::from(format!("{}/{}", dst_prefix.as_ref(), rel));
            assert!(
                dst_store.head(&key).await.is_ok(),
                "side-car missing at destination: {rel}"
            );
        }

        // Both labels round-trip (so the copied label index resolves).
        let restored = WriterSession::open(dst_store, dst_paths).await.unwrap();
        let snap = restored.snapshot();
        assert_eq!(snap.scan_label("Person").await.unwrap().len(), 3);
        assert_eq!(snap.scan_label("Admin").await.unwrap().len(), 3);
    }

    #[tokio::test]
    async fn snapshot_round_trips_empty_namespace() {
        let (src_store, src_paths) = (store(), paths("bk-empty-src"));
        // Bootstrap only — no commits, so no SSTs and no WAL segments.
        WriterSession::open(src_store.clone(), src_paths.clone())
            .await
            .unwrap();

        let (dst_store, dst_paths) = (store(), paths("bk-empty-dst"));
        let report = copy_namespace_snapshot(
            src_store,
            src_paths,
            dst_store.clone(),
            dst_paths.clone(),
            None,
            false,
            false,
        )
        .await
        .unwrap();
        // The manifest body, the pointer, and the advisory current.json.
        assert_eq!(report.objects_copied, 3);

        // The restored empty namespace opens cleanly and carries no nodes.
        assert!(names_in(dst_store, dst_paths).await.is_empty());
    }

    #[tokio::test]
    async fn verify_passes_on_a_clean_copy_and_catches_a_missing_object() {
        let (src_store, src_paths) = (store(), paths("bk-verify-src"));
        {
            let mut w = WriterSession::open(src_store.clone(), src_paths.clone())
                .await
                .unwrap();
            w.upsert_node("Person", NodeId::new(), &person("Ada"))
                .unwrap();
            w.commit_batch().await.unwrap();
            w.flush(schema()).await.unwrap();
        }
        let (dst_store, dst_paths) = (store(), paths("bk-verify-dst"));

        // Clean copy with verify → succeeds.
        copy_namespace_snapshot(
            src_store.clone(),
            src_paths.clone(),
            dst_store.clone(),
            dst_paths.clone(),
            None,
            false,
            true,
        )
        .await
        .expect("verify passes on a clean copy");

        // A second copy with one SST body deleted before verify → must fail
        // loudly rather than report a healthy-but-truncated snapshot.
        let (dst2_store, dst2_paths) = (store(), paths("bk-verify-dst2"));
        copy_namespace_snapshot(
            src_store,
            src_paths,
            dst2_store.clone(),
            dst2_paths.clone(),
            None,
            false,
            false,
        )
        .await
        .unwrap();
        // Delete the first SST body the restored manifest references.
        let dst_ms = ManifestStore::new(dst2_store.clone(), dst2_paths.clone());
        let m = dst_ms.load_current().await.unwrap().manifest;
        let first_sst = &m.ssts[0];
        let rel = first_sst.path.as_str();
        let p = object_store::path::Path::from(format!(
            "{}/{}",
            dst2_paths.namespace_prefix().as_ref(),
            rel
        ));
        dst2_store.delete(&p).await.unwrap();
        let err = copy_namespace_snapshot(
            store(), // fresh src is fine; verify inspects dst
            paths("bk-verify-src-3"),
            dst2_store,
            dst2_paths,
            None,
            true, // overwrite so the copy re-runs; verify then catches the gap
            true,
        )
        .await
        .unwrap_err();
        assert!(
            err.to_string().to_lowercase().contains("verify"),
            "expected a verify failure, got: {err}"
        );
    }

    /// The copy must hold a retention pin lease over EVERY data read (so the
    /// source's orphan sweep keeps the pinned closure alive mid-copy) and
    /// release the lease once it finishes.
    #[tokio::test]
    async fn copy_holds_a_pin_lease_over_every_data_read_and_releases_it() {
        let src_inner = store();
        let src_paths = paths("bk-pin-src");
        {
            let mut w = WriterSession::open(src_inner.clone(), src_paths.clone())
                .await
                .unwrap();
            // Flushed rows (SST reads) plus an unflushed commit (a WAL read).
            w.upsert_node("Person", NodeId::new(), &person("Ada"))
                .unwrap();
            w.commit_batch().await.unwrap();
            w.flush(schema()).await.unwrap();
            w.upsert_node("Person", NodeId::new(), &person("Lin"))
                .unwrap();
            w.commit_batch().await.unwrap();
        }

        let probe = Arc::new(ProbeStore::new(
            src_inner.clone(),
            Some(src_paths.pins_dir()),
        ));
        let src: Arc<dyn ObjectStore> = probe.clone();
        let (dst_store, dst_paths) = (store(), paths("bk-pin-dst"));
        copy_namespace_snapshot(
            src,
            src_paths.clone(),
            dst_store.clone(),
            dst_paths.clone(),
            None,
            false,
            false,
        )
        .await
        .unwrap();

        assert!(
            probe.data_reads.load(Ordering::SeqCst) >= 2,
            "the copy must have read SST and WAL bodies through the probe"
        );
        assert_eq!(
            probe.unpinned_data_reads.load(Ordering::SeqCst),
            0,
            "every data read must happen under a live pin lease"
        );

        // The lease is released once the copy completes.
        let mut pins = src_inner.list(Some(&src_paths.pins_dir()));
        assert!(
            pins.try_next().await.unwrap().is_none(),
            "the pin lease must be released after the copy"
        );

        // And the destination still round-trips.
        assert_eq!(names_in(dst_store, dst_paths).await, vec!["Ada", "Lin"]);
    }

    /// An object larger than one part must stream through a multipart upload
    /// (never the whole body in one buffer) and land byte-identical; objects
    /// at or below one part take the buffered PUT fast path.
    #[tokio::test]
    async fn streaming_copy_round_trips_an_object_larger_than_one_part() {
        let src = store();
        let probe = Arc::new(ProbeStore::new(store(), None));
        let dst: Arc<dyn ObjectStore> = probe.clone();

        // 10 KiB body, 1 KiB parts → 10 parts through the multipart path.
        let body: Vec<u8> = (0..10 * 1024u32).map(|i| (i % 251) as u8).collect();
        let from = Path::from("src/big.bin");
        let to = Path::from("dst/big.bin");
        src.put(&from, PutPayload::from(body.clone()))
            .await
            .unwrap();
        let n = copy_object_with_part_size(&src, &dst, &from, &to, 1024)
            .await
            .unwrap();
        assert_eq!(n, body.len() as u64);
        assert_eq!(
            probe.multipart_uploads.load(Ordering::SeqCst),
            1,
            "an object larger than one part must stream via multipart"
        );
        let copied = dst.get(&to).await.unwrap().bytes().await.unwrap();
        assert_eq!(
            copied.as_ref(),
            body.as_slice(),
            "copy must be byte-identical"
        );

        // At or below one part: plain buffered PUT, no new multipart upload.
        let small = b"tiny".to_vec();
        let from_small = Path::from("src/small.bin");
        let to_small = Path::from("dst/small.bin");
        src.put(&from_small, PutPayload::from(small.clone()))
            .await
            .unwrap();
        let n = copy_object_with_part_size(&src, &dst, &from_small, &to_small, 1024)
            .await
            .unwrap();
        assert_eq!(n, small.len() as u64);
        assert_eq!(
            probe.multipart_uploads.load(Ordering::SeqCst),
            1,
            "a small object must not start a multipart upload"
        );
        let copied = dst.get(&to_small).await.unwrap().bytes().await.unwrap();
        assert_eq!(copied.as_ref(), small.as_slice());
    }
}
