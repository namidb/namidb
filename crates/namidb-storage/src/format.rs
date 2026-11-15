//! Storage format versioning.
//!
//! Every manifest version carries a [`StorageFormat`] stamp: the on-disk
//! format the committing writer used, the oldest format a reader must
//! understand to read that version, and the oldest format a writer must
//! understand to commit on top of it. This module defines the format numbers,
//! enforces the stamp on load and claim, and inspects stores written before
//! the stamp existed.
//!
//! ## Format numbers
//!
//! | Format | Written by | What changed |
//! |--------|------------|--------------|
//! | 1 | NamiDB 2.x | node SSTs keep every ad-hoc property, embeddings included, in the `__overflow_json` text column |
//! | 2 | NamiDB 3.x | node SSTs carry numeric arrays and byte strings in `__binary_props`; exact node records are binary |
//!
//! A manifest without a stamp was written by NamiDB 3.3.1 or older. Its
//! format is inferred from the data it references (see
//! [`infer_unstamped_format`]): format 1 when nothing in it needs a 3.x
//! reader, format 2 otherwise.
//!
//! ## Rules
//!
//! - A manifest whose `min_reader_version` is above [`STORAGE_FORMAT_VERSION`]
//!   is refused on load ([`Error::NewerStorageFormat`]), so nothing reads,
//!   compacts or sweeps data it does not understand.
//! - A writer refuses to claim a manifest whose `min_writer_version` is above
//!   [`STORAGE_FORMAT_VERSION`]: re-serialising it would drop the fields it
//!   does not know.
//! - Opening an older-format namespace for writing upgrades it the moment
//!   anything new is written. [`FormatUpgrade::Refuse`] makes that an explicit
//!   choice ([`Error::StorageFormatUpgradeRequired`]); rewriting the existing
//!   SSTs into the current layout is a separate, explicit migration (see
//!   [`crate::compact::CompactionBasis::prepare_node_layout_migration`]).

use std::sync::Arc;

use object_store::path::Path;
use object_store::{ObjectStore, ObjectStoreExt};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::error::{Error, Result};
use crate::manifest::{Manifest, SstKind};
use crate::paths::NamespacePaths;

/// The storage format this build writes, and the newest it can read.
pub const STORAGE_FORMAT_VERSION: u32 = 2;

/// The NamiDB 2.x format: embeddings and other numeric arrays inside the
/// node SSTs' overflow JSON.
pub const LEGACY_STORAGE_FORMAT_VERSION: u32 = 1;

/// Oldest format able to read what this build writes.
const MIN_READER_FOR_WRITES: u32 = 2;

/// Oldest format allowed to commit on top of what this build writes.
const MIN_WRITER_FOR_WRITES: u32 = 2;

/// Format stamp carried by every manifest version this build commits.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StorageFormat {
    /// Format the writer of this manifest version wrote with.
    pub version: u32,
    /// Oldest format a reader must support to read this manifest version and
    /// everything it references.
    pub min_reader_version: u32,
    /// Oldest format a writer must support to commit a successor of this
    /// manifest version.
    pub min_writer_version: u32,
    /// Release that committed this version, for operators (`namidb 3.4.0`).
    #[serde(default)]
    pub written_by: String,
}

impl StorageFormat {
    /// The stamp of a fresh namespace written by this build.
    pub fn current() -> Self {
        Self {
            version: STORAGE_FORMAT_VERSION,
            min_reader_version: MIN_READER_FOR_WRITES,
            min_writer_version: MIN_WRITER_FOR_WRITES,
            written_by: written_by(),
        }
    }

    /// The stamp of a manifest this build commits on top of one stamped
    /// `prev`. Requirements only grow: a newer writer's data stays in the
    /// namespace after this build commits next to it.
    pub fn successor(prev: Option<&StorageFormat>) -> Self {
        let mut next = Self::current();
        if let Some(prev) = prev {
            next.min_reader_version = next.min_reader_version.max(prev.min_reader_version);
            next.min_writer_version = next.min_writer_version.max(prev.min_writer_version);
        }
        next
    }
}

fn written_by() -> String {
    concat!("namidb ", env!("CARGO_PKG_VERSION")).to_string()
}

fn describe_writer(format: &StorageFormat) -> String {
    if format.written_by.is_empty() {
        "a newer NamiDB release".to_string()
    } else {
        format.written_by.clone()
    }
}

/// Refuse a manifest this build cannot read.
pub fn ensure_readable(manifest: &Manifest) -> Result<()> {
    let Some(format) = &manifest.storage_format else {
        return Ok(());
    };
    if format.min_reader_version > STORAGE_FORMAT_VERSION {
        return Err(Error::NewerStorageFormat {
            found: format.version,
            required: format.min_reader_version,
            supported: STORAGE_FORMAT_VERSION,
            written_by: describe_writer(format),
            access: "read",
        });
    }
    Ok(())
}

/// Refuse to commit on top of a manifest this build cannot write.
pub fn ensure_writable(manifest: &Manifest) -> Result<()> {
    ensure_readable(manifest)?;
    let Some(format) = &manifest.storage_format else {
        return Ok(());
    };
    if format.min_writer_version > STORAGE_FORMAT_VERSION {
        return Err(Error::NewerStorageFormat {
            found: format.version,
            required: format.min_writer_version,
            supported: STORAGE_FORMAT_VERSION,
            written_by: describe_writer(format),
            access: "write",
        });
    }
    Ok(())
}

/// What a writer does when the namespace it opens holds older-format data
/// that its first write would make unreadable to the release that wrote it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FormatUpgrade {
    /// Open it; new writes use the current format. The behaviour of
    /// [`crate::WriterSession::open`].
    #[default]
    Allow,
    /// Fail with [`Error::StorageFormatUpgradeRequired`] before claiming the
    /// namespace, leaving it untouched.
    Refuse,
}

/// Layout of one node SST, as its Parquet footer records it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct NodeSstLayout {
    pub id: Uuid,
    /// Object path relative to the namespace prefix.
    pub path: String,
    pub level: u32,
    pub scope: String,
    pub row_count: u64,
    pub size_bytes: u64,
    /// `true` for the 2.x layout (no `__binary_props` column).
    pub legacy: bool,
}

/// What [`inspect_storage_format`] found in one manifest version.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct FormatInspection {
    pub manifest_version: u64,
    /// The manifest's stamp; `None` when an older release committed it.
    pub stamp: Option<StorageFormat>,
    /// Every node SST the manifest references.
    pub node_ssts: Vec<NodeSstLayout>,
    /// WAL segments the manifest references that a 2.x reader can decode.
    pub legacy_wal_segments: usize,
    /// WAL segments the manifest references that need a 3.x reader.
    pub current_wal_segments: usize,
}

impl FormatInspection {
    /// Node SSTs still in the 2.x layout.
    pub fn legacy_node_ssts(&self) -> impl Iterator<Item = &NodeSstLayout> {
        self.node_ssts.iter().filter(|sst| sst.legacy)
    }

    /// The oldest format that can read the data: the stamp's
    /// `min_reader_version`, or for an unstamped manifest the format inferred
    /// from its SSTs and WAL (`0` for a namespace with neither).
    pub fn effective_format(&self) -> u32 {
        if let Some(stamp) = &self.stamp {
            return stamp.min_reader_version;
        }
        infer_from_counts(
            self.node_ssts.iter().any(|sst| !sst.legacy) || self.current_wal_segments > 0,
            !self.node_ssts.is_empty() || self.legacy_wal_segments > 0,
        )
    }
}

fn infer_from_counts(any_current: bool, any_data: bool) -> u32 {
    if any_current {
        STORAGE_FORMAT_VERSION
    } else if any_data {
        LEGACY_STORAGE_FORMAT_VERSION
    } else {
        0
    }
}

/// Whether the node SST at `desc` uses the 2.x layout. Reads only the
/// Parquet footer.
pub async fn node_sst_is_legacy(
    store: &Arc<dyn ObjectStore>,
    paths: &NamespacePaths,
    desc: &crate::manifest::SstDescriptor,
) -> Result<bool> {
    let path = Path::from(format!(
        "{}/{}",
        paths.namespace_prefix().as_ref(),
        desc.path
    ));
    let metadata =
        crate::sst::nodes::load_node_sst_metadata_async(store.clone(), path, desc.size_bytes)
            .await?;
    let schema = metadata.file_metadata().schema_descr();
    Ok(!schema
        .root_schema()
        .get_fields()
        .iter()
        .any(|field| field.name() == crate::sst::nodes::BINARY_PROPS))
}

/// WAL format version of the segment at `path` (a full object path, as a
/// [`crate::manifest::WalSegmentDescriptor`] records it), from its six-byte
/// header.
async fn wal_segment_version(store: &Arc<dyn ObjectStore>, path: &str) -> Result<u16> {
    let path = Path::from(path);
    let header = store.get_range(&path, 0..6).await?;
    crate::wal::header_format_version(&header).ok_or_else(|| Error::Corrupted {
        path: path.as_ref().to_string(),
        detail: "WAL segment header is not readable".into(),
    })
}

/// WAL segments from the version that started binary node payloads (3.1)
/// cannot be read by 2.x.
const FIRST_CURRENT_WAL_VERSION: u16 = 2;

/// Inspect every node SST footer and WAL header the manifest references.
pub async fn inspect_storage_format(
    store: &Arc<dyn ObjectStore>,
    paths: &NamespacePaths,
    manifest: &Manifest,
) -> Result<FormatInspection> {
    let mut node_ssts = Vec::new();
    for desc in manifest.ssts.iter().filter(|d| d.kind == SstKind::Nodes) {
        node_ssts.push(NodeSstLayout {
            id: desc.id,
            path: desc.path.clone(),
            level: desc.level.as_u32(),
            scope: desc.scope.clone(),
            row_count: desc.row_count,
            size_bytes: desc.size_bytes,
            legacy: node_sst_is_legacy(store, paths, desc).await?,
        });
    }
    let mut legacy_wal_segments = 0;
    let mut current_wal_segments = 0;
    for segment in &manifest.wal_segments {
        if wal_segment_version(store, &segment.path).await? >= FIRST_CURRENT_WAL_VERSION {
            current_wal_segments += 1;
        } else {
            legacy_wal_segments += 1;
        }
    }
    Ok(FormatInspection {
        manifest_version: manifest.version,
        stamp: manifest.storage_format.clone(),
        node_ssts,
        legacy_wal_segments,
        current_wal_segments,
    })
}

/// The format of an unstamped manifest's data, reading as little as it can:
/// it stops at the first WAL segment or node SST that needs a 3.x reader.
/// Newer node SSTs sit in the shallower levels, so those are probed first.
pub async fn infer_unstamped_format(
    store: &Arc<dyn ObjectStore>,
    paths: &NamespacePaths,
    manifest: &Manifest,
) -> Result<u32> {
    for segment in &manifest.wal_segments {
        if wal_segment_version(store, &segment.path).await? >= FIRST_CURRENT_WAL_VERSION {
            return Ok(STORAGE_FORMAT_VERSION);
        }
    }
    let mut nodes: Vec<_> = manifest
        .ssts
        .iter()
        .filter(|d| d.kind == SstKind::Nodes)
        .collect();
    nodes.sort_by_key(|d| (d.level, std::cmp::Reverse(d.max_lsn)));
    for desc in &nodes {
        if !node_sst_is_legacy(store, paths, desc).await? {
            return Ok(STORAGE_FORMAT_VERSION);
        }
    }
    Ok(infer_from_counts(
        false,
        !manifest.ssts.is_empty() || !manifest.wal_segments.is_empty(),
    ))
}

/// Apply `policy` to the namespace a writer is about to claim. With
/// [`FormatUpgrade::Refuse`], a manifest whose data a 2.x reader can still
/// read fails with [`Error::StorageFormatUpgradeRequired`].
pub async fn check_upgrade(
    store: &Arc<dyn ObjectStore>,
    paths: &NamespacePaths,
    manifest: &Manifest,
    policy: FormatUpgrade,
) -> Result<()> {
    ensure_writable(manifest)?;
    if policy == FormatUpgrade::Allow {
        return Ok(());
    }
    let effective = match &manifest.storage_format {
        Some(stamp) => stamp.min_reader_version,
        None => infer_unstamped_format(store, paths, manifest).await?,
    };
    if effective != 0 && effective < MIN_READER_FOR_WRITES {
        return Err(Error::StorageFormatUpgradeRequired {
            found: effective,
            current: STORAGE_FORMAT_VERSION,
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stamped(min_reader: u32, min_writer: u32) -> Manifest {
        let mut m = Manifest::empty(crate::fence::Epoch::ZERO, Uuid::nil());
        m.storage_format = Some(StorageFormat {
            version: min_reader.max(min_writer),
            min_reader_version: min_reader,
            min_writer_version: min_writer,
            written_by: "namidb 9.0.0".into(),
        });
        m
    }

    #[test]
    fn new_manifests_carry_the_current_stamp() {
        let m = Manifest::empty(crate::fence::Epoch::ZERO, Uuid::nil());
        assert_eq!(m.storage_format, Some(StorageFormat::current()));
        let next = m.next_version(Uuid::nil());
        assert_eq!(next.storage_format, Some(StorageFormat::current()));
    }

    #[test]
    fn successor_keeps_a_newer_writers_requirements() {
        let prev = stamped(STORAGE_FORMAT_VERSION + 1, STORAGE_FORMAT_VERSION);
        let next = prev.next_version(Uuid::nil());
        let stamp = next.storage_format.unwrap();
        assert_eq!(stamp.version, STORAGE_FORMAT_VERSION);
        assert_eq!(stamp.min_reader_version, STORAGE_FORMAT_VERSION + 1);
        assert_eq!(stamp.min_writer_version, STORAGE_FORMAT_VERSION);
    }

    #[test]
    fn newer_reader_requirement_is_refused() {
        let err = ensure_readable(&stamped(
            STORAGE_FORMAT_VERSION + 1,
            STORAGE_FORMAT_VERSION + 1,
        ))
        .unwrap_err();
        let text = err.to_string();
        assert!(matches!(
            err,
            Error::NewerStorageFormat { access: "read", .. }
        ));
        assert!(text.contains("namidb 9.0.0"), "{text}");
        assert!(text.contains("upgrade"), "{text}");
    }

    #[test]
    fn newer_writer_requirement_allows_reads_only() {
        let m = stamped(STORAGE_FORMAT_VERSION, STORAGE_FORMAT_VERSION + 1);
        ensure_readable(&m).unwrap();
        assert!(matches!(
            ensure_writable(&m),
            Err(Error::NewerStorageFormat {
                access: "write",
                ..
            })
        ));
    }

    #[test]
    fn unstamped_manifests_pass_the_stamp_checks() {
        let mut m = Manifest::empty(crate::fence::Epoch::ZERO, Uuid::nil());
        m.storage_format = None;
        ensure_writable(&m).unwrap();
    }

    // ── Sessions over 2.x-layout data ───────────────────────────────────

    use std::collections::BTreeMap;

    use namidb_core::{NamespaceId, NodeId, Value};
    use object_store::memory::InMemory;

    use crate::fence::WriterFence;
    use crate::flush::NodeWriteRecord;
    use crate::ingest::WriterSession;
    use crate::manifest::ManifestStore;
    use crate::sst::nodes::LegacyNodeLayout;

    fn ns(name: &str) -> (Arc<dyn ObjectStore>, NamespacePaths) {
        (
            Arc::new(InMemory::new()),
            NamespacePaths::new("t", NamespaceId::new(name).unwrap()),
        )
    }

    fn embedding(seed: u64) -> Value {
        // Whole numbers survive the JSON layout bit for bit.
        Value::List(
            (0..32)
                .map(|j| Value::F64((seed * 100 + j) as f64))
                .collect(),
        )
    }

    fn doc(i: u64) -> NodeWriteRecord {
        NodeWriteRecord {
            properties: BTreeMap::from([
                ("name".to_string(), Value::Str(format!("doc{i}"))),
                ("n".to_string(), Value::I64(i as i64)),
                ("embedding".to_string(), embedding(i)),
            ]),
            schema_version: 0,
            labels: Vec::new(),
        }
    }

    /// Write `batches` flushed batches of `per` docs each, then commit a
    /// manifest without a stamp, the way NamiDB 3.3.1 and older left it.
    async fn write_store(
        store: &Arc<dyn ObjectStore>,
        paths: &NamespacePaths,
        legacy: bool,
        batches: u64,
        per: u64,
    ) -> Vec<NodeId> {
        let mut ids = Vec::new();
        let mut session = WriterSession::open(store.clone(), paths.clone())
            .await
            .unwrap();
        for b in 0..batches {
            let _layout = legacy.then(LegacyNodeLayout::on);
            for i in 0..per {
                let id = NodeId::new();
                session.upsert_node("Doc", id, &doc(b * per + i)).unwrap();
                ids.push(id);
            }
            session.commit_batch().await.unwrap();
            let schema = session.snapshot().manifest().manifest.schema.clone();
            session.flush(schema).await.unwrap();
        }
        drop(session);
        unstamp(store, paths).await;
        ids
    }

    async fn unstamp(store: &Arc<dyn ObjectStore>, paths: &NamespacePaths) {
        let ms = ManifestStore::new(store.clone(), paths.clone());
        let base = ms.load_current().await.unwrap();
        let fence = WriterFence::new(base.manifest.epoch);
        let mut next = base.manifest.next_version(Uuid::now_v7());
        next.storage_format = None;
        ms.commit(&fence, &base, next).await.unwrap();
    }

    async fn stamp_with(
        store: &Arc<dyn ObjectStore>,
        paths: &NamespacePaths,
        stamp: StorageFormat,
    ) {
        let ms = ManifestStore::new(store.clone(), paths.clone());
        let base = ms.load_current().await.unwrap();
        let fence = WriterFence::new(base.manifest.epoch);
        let mut next = base.manifest.next_version(Uuid::now_v7());
        next.storage_format = Some(stamp);
        ms.commit(&fence, &base, next).await.unwrap();
    }

    async fn current(store: &Arc<dyn ObjectStore>, paths: &NamespacePaths) -> Manifest {
        // Read the body directly: `load_current` refuses newer formats.
        let ms = ManifestStore::new(store.clone(), paths.clone());
        let mut version = 0;
        while store
            .head(&paths.pointer_version(version + 1))
            .await
            .is_ok()
        {
            version += 1;
        }
        let body = store
            .get(&paths.manifest_version(version))
            .await
            .unwrap()
            .bytes()
            .await
            .unwrap();
        let _ = ms;
        serde_json::from_slice(&body).unwrap()
    }

    async fn docs(session: &WriterSession) -> Vec<(NodeId, BTreeMap<String, Value>)> {
        let mut rows: Vec<_> = session
            .snapshot()
            .scan_label("Doc")
            .await
            .unwrap()
            .into_iter()
            .map(|view| (view.id, view.properties))
            .collect();
        rows.sort_by_key(|(id, _)| *id);
        rows
    }

    #[tokio::test]
    async fn a_2x_namespace_is_not_upgraded_without_opt_in() {
        let (store, paths) = ns("legacy-open");
        write_store(&store, &paths, true, 2, 20).await;
        let before = current(&store, &paths).await;
        assert_eq!(before.storage_format, None);
        assert_eq!(
            infer_unstamped_format(&store, &paths, &before)
                .await
                .unwrap(),
            LEGACY_STORAGE_FORMAT_VERSION
        );

        let Err(err) = WriterSession::open_with_options(
            store.clone(),
            paths.clone(),
            crate::ingest::SessionCaches::shared(),
            FormatUpgrade::Refuse,
        )
        .await
        else {
            panic!("a 2.x namespace must not open for writing without opt-in");
        };
        assert!(
            matches!(
                err,
                Error::StorageFormatUpgradeRequired {
                    found: 1,
                    current: 2
                }
            ),
            "{err}"
        );
        assert!(err.to_string().contains("upgrade=True"), "{err}");
        // Nothing was claimed or committed.
        assert_eq!(current(&store, &paths).await, before);

        // Read-only access needs no opt-in.
        crate::reader::ReaderSession::open(store.clone(), paths.clone(), None)
            .await
            .unwrap();

        let session = WriterSession::open_with_options(
            store.clone(),
            paths.clone(),
            crate::ingest::SessionCaches::shared(),
            FormatUpgrade::Allow,
        )
        .await
        .unwrap();
        assert_eq!(docs(&session).await.len(), 40);
        drop(session);
        let after = current(&store, &paths).await;
        assert_eq!(after.storage_format, Some(StorageFormat::current()));
        // Claiming stamped the manifest; it did not rewrite any SST.
        assert_eq!(after.ssts, before.ssts);
    }

    #[tokio::test]
    async fn a_3x_namespace_without_a_stamp_opens_without_opt_in() {
        let (store, paths) = ns("current-open");
        write_store(&store, &paths, false, 1, 10).await;
        let manifest = current(&store, &paths).await;
        assert_eq!(manifest.storage_format, None);
        assert_eq!(
            infer_unstamped_format(&store, &paths, &manifest)
                .await
                .unwrap(),
            STORAGE_FORMAT_VERSION
        );
        WriterSession::open_with_options(
            store.clone(),
            paths.clone(),
            crate::ingest::SessionCaches::shared(),
            FormatUpgrade::Refuse,
        )
        .await
        .unwrap();
    }

    #[tokio::test]
    async fn an_unflushed_3x_namespace_is_inferred_from_its_wal() {
        let (store, paths) = ns("wal-only");
        let mut session = WriterSession::open(store.clone(), paths.clone())
            .await
            .unwrap();
        session.upsert_node("Doc", NodeId::new(), &doc(1)).unwrap();
        session.commit_batch().await.unwrap();
        drop(session);
        unstamp(&store, &paths).await;
        let manifest = current(&store, &paths).await;
        assert!(manifest.ssts.is_empty());
        assert_eq!(manifest.wal_segments.len(), 1);
        let inspection = inspect_storage_format(&store, &paths, &manifest)
            .await
            .unwrap();
        assert_eq!(inspection.current_wal_segments, 1);
        assert_eq!(inspection.effective_format(), STORAGE_FORMAT_VERSION);
        assert_eq!(
            infer_unstamped_format(&store, &paths, &manifest)
                .await
                .unwrap(),
            STORAGE_FORMAT_VERSION
        );
        WriterSession::open_with_options(
            store.clone(),
            paths.clone(),
            crate::ingest::SessionCaches::shared(),
            FormatUpgrade::Refuse,
        )
        .await
        .unwrap();
    }

    #[tokio::test]
    async fn a_namespace_from_a_newer_format_is_refused() {
        let (store, paths) = ns("newer-format");
        write_store(&store, &paths, false, 1, 5).await;
        stamp_with(
            &store,
            &paths,
            StorageFormat {
                version: STORAGE_FORMAT_VERSION + 1,
                min_reader_version: STORAGE_FORMAT_VERSION + 1,
                min_writer_version: STORAGE_FORMAT_VERSION + 1,
                written_by: "namidb 9.0.0".into(),
            },
        )
        .await;
        let before = current(&store, &paths).await;
        let err = WriterSession::open(store.clone(), paths.clone())
            .await
            .err()
            .unwrap();
        assert!(
            matches!(err, Error::NewerStorageFormat { access: "read", .. }),
            "{err}"
        );
        let err = crate::reader::ReaderSession::open(store.clone(), paths.clone(), None)
            .await
            .err()
            .unwrap();
        assert!(matches!(err, Error::NewerStorageFormat { .. }), "{err}");
        // Not claimed, not rewritten: the newer release's stamp survives.
        assert_eq!(current(&store, &paths).await, before);
    }

    #[tokio::test]
    async fn a_newer_writer_format_still_opens_read_only() {
        let (store, paths) = ns("newer-writer");
        write_store(&store, &paths, false, 1, 5).await;
        stamp_with(
            &store,
            &paths,
            StorageFormat {
                version: STORAGE_FORMAT_VERSION + 1,
                min_reader_version: STORAGE_FORMAT_VERSION,
                min_writer_version: STORAGE_FORMAT_VERSION + 1,
                written_by: "namidb 9.0.0".into(),
            },
        )
        .await;
        let before = current(&store, &paths).await;
        let err = WriterSession::open(store.clone(), paths.clone())
            .await
            .err()
            .unwrap();
        assert!(
            matches!(
                err,
                Error::NewerStorageFormat {
                    access: "write",
                    ..
                }
            ),
            "{err}"
        );
        assert_eq!(current(&store, &paths).await, before);
        let reader = crate::reader::ReaderSession::open(store.clone(), paths.clone(), None)
            .await
            .unwrap();
        let snapshot = reader.snapshot();
        assert_eq!(snapshot.borrow().scan_label("Doc").await.unwrap().len(), 5);
    }

    #[tokio::test]
    async fn layout_migration_rewrites_2x_ssts_and_keeps_every_row() {
        let (store, paths) = ns("migrate");
        write_store(&store, &paths, true, 3, 25).await;
        let mut session = WriterSession::open(store.clone(), paths.clone())
            .await
            .unwrap();
        let before_rows = docs(&session).await;
        assert_eq!(before_rows.len(), 75);
        let before = session.snapshot().manifest().manifest.clone();
        let legacy = session.compaction_basis().legacy_node_ssts().await.unwrap();
        assert_eq!(legacy.len(), 3);

        // A rewrite that is prepared but never installed (a crash before
        // its commit) changes nothing; the migration simply resumes.
        let (_abandoned, _) = session
            .compaction_basis()
            .prepare_node_layout_migration(legacy[0].id)
            .await
            .unwrap();
        assert_eq!(session.snapshot().manifest().manifest.ssts, before.ssts);

        for sst in &legacy {
            let (prepared, migration) = session
                .compaction_basis()
                .prepare_node_layout_migration(sst.id)
                .await
                .unwrap();
            assert_eq!(migration.rows, sst.row_count);
            assert_eq!(migration.level, sst.level);
            session.install_prepared_compaction(prepared).await.unwrap();
            assert_eq!(docs(&session).await, before_rows);
        }
        assert!(session
            .compaction_basis()
            .legacy_node_ssts()
            .await
            .unwrap()
            .is_empty());
        let after = session.snapshot().manifest().manifest.clone();
        let levels = |m: &Manifest| {
            let mut levels: Vec<_> = m
                .ssts
                .iter()
                .filter(|d| d.kind == SstKind::Nodes)
                .map(|d| (d.level, d.min_key, d.max_key, d.max_lsn, d.row_count))
                .collect();
            levels.sort();
            levels
        };
        assert_eq!(levels(&after), levels(&before));
        drop(session);

        // A fresh session reads the same rows from the rewritten SSTs.
        let session = WriterSession::open(store.clone(), paths.clone())
            .await
            .unwrap();
        assert_eq!(docs(&session).await, before_rows);
        let inspection =
            inspect_storage_format(&store, &paths, &session.snapshot().manifest().manifest)
                .await
                .unwrap();
        assert_eq!(inspection.legacy_node_ssts().count(), 0);
        assert_eq!(inspection.effective_format(), STORAGE_FORMAT_VERSION);
    }

    #[tokio::test]
    async fn a_migration_step_whose_sst_was_compacted_away_is_refused() {
        let (store, paths) = ns("migrate-race");
        write_store(&store, &paths, true, 2, 10).await;
        let mut session = WriterSession::open(store.clone(), paths.clone())
            .await
            .unwrap();
        let legacy = session.compaction_basis().legacy_node_ssts().await.unwrap();
        let (prepared, _) = session
            .compaction_basis()
            .prepare_node_layout_migration(legacy[0].id)
            .await
            .unwrap();
        // Rewrite the same SST first, then try to install the stale step.
        let (first, _) = session
            .compaction_basis()
            .prepare_node_layout_migration(legacy[0].id)
            .await
            .unwrap();
        session.install_prepared_compaction(first).await.unwrap();
        let err = session
            .install_prepared_compaction(prepared)
            .await
            .unwrap_err();
        assert!(matches!(err, Error::Precondition(_)), "{err}");
        let err = session
            .compaction_basis()
            .prepare_node_layout_migration(legacy[0].id)
            .await
            .unwrap_err();
        assert!(matches!(err, Error::Precondition(_)), "{err}");
    }

    #[test]
    fn stamp_round_trips_and_old_manifests_load_without_it() {
        let m = stamped(2, 2);
        let json = serde_json::to_string(&m).unwrap();
        assert!(json.contains("\"storage_format\""), "{json}");
        let back: Manifest = serde_json::from_str(&json).unwrap();
        assert_eq!(back, m);

        let mut value: serde_json::Value = serde_json::from_str(&json).unwrap();
        value.as_object_mut().unwrap().remove("storage_format");
        let old: Manifest = serde_json::from_value(value).unwrap();
        assert_eq!(old.storage_format, None);
    }
}
