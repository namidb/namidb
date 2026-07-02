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
//! Caveat: run the copy against a quiescent source, or accept that a
//! concurrent compaction plus orphan sweep on the source could delete a pinned
//! object mid-copy if the copy outlives the source's retention horizon. There
//! is no `FREEZE`; pinning a specific committed `version` narrows the window to
//! the copy itself.

use std::sync::Arc;

use futures::TryStreamExt;
use object_store::path::Path;
use object_store::{ObjectStore, ObjectStoreExt, PutMode, PutOptions, PutPayload};

use crate::error::{Error, Result};
use crate::fence::Epoch;
use crate::manifest::{Manifest, ManifestPointer, ManifestStore};
use crate::paths::NamespacePaths;

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
/// Run the copy against a quiescent source. The pinned objects are immutable,
/// but a concurrent compaction plus orphan sweep on the source could delete
/// one mid-copy if the copy outlives the source's retention horizon, which
/// surfaces as a non-retriable `NotFound`. There is no `FREEZE` yet; pinning a
/// committed `version` narrows the window to the copy itself.
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
    let mut manifest: Manifest = match version {
        Some(v) => src_manifests.load_manifest_at(v).await?,
        None => src_manifests.load_current().await?.manifest,
    };
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
            bytes_copied += copy_object(&src_store, &dst_store, &from, &to).await?;
            objects_copied += 1;
        }
    }

    // 2. WAL segments still needed for recovery, addressed by seq (the same
    //    canonical key the recovery path reads them back through).
    for seg in &manifest.wal_segments {
        let from = src_paths.wal_segment(seg.seq);
        let to = dst_paths.wal_segment(seg.seq);
        bytes_copied += copy_object(&src_store, &dst_store, &from, &to).await?;
        objects_copied += 1;
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
        verify_snapshot(&dst_store, &dst_paths, &manifest).await?;
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

/// Stream one object from `src` to `dst`, returning its byte length. Plain
/// GET + PUT so it works across backends (s3 -> file, file -> gs, ...); a
/// same-store fast path via `ObjectStore::copy` is a later optimisation.
async fn copy_object(
    src: &Arc<dyn ObjectStore>,
    dst: &Arc<dyn ObjectStore>,
    from: &Path,
    to: &Path,
) -> Result<u64> {
    let bytes = src.get(from).await?.bytes().await?;
    let len = bytes.len() as u64;
    dst.put_opts(
        to,
        PutPayload::from(bytes),
        PutOptions::from(PutMode::Overwrite),
    )
    .await?;
    Ok(len)
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use namidb_core::id::{NamespaceId, NodeId};
    use namidb_core::schema::{DataType, LabelDef, PropertyDef, Schema, SchemaBuilder};
    use namidb_core::value::Value;
    use object_store::memory::InMemory;

    use super::*;
    use crate::flush::NodeWriteRecord;
    use crate::ingest::WriterSession;
    use crate::manifest::ManifestStore;

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
}
