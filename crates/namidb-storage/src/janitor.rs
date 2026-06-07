//! Stateless janitor for orphaned SST + bloom side-car objects.
//!
//! ## Why orphans exist
//!
//! Several legitimate code paths in `namidb-storage` produce objects that
//! survive without being referenced by the current manifest:
//!
//! - **Flush failure between PUT and CAS.** [`crate::flush::flush`] writes
//! SST + bloom bodies via `PutMode::Create`, then commits a manifest
//! version that references them. If the manifest CAS loses, the bodies
//! stay; nothing dangerous, just paid storage. RFC-002 §4 explicitly
//! names "fail-fast with orphans" as the chosen tradeoff over two-phase
//! commit.
//! - **Compaction.** [`crate::compact::compact_l0_to_l1`] removes the
//! source L0 descriptors from the manifest after the L1 SST commits,
//! but the source bodies in `sst/level0/` remain readable. Any reader
//! pinned at the pre-compaction manifest version still relies on them.
//! - **Crashed writers.** A process can die after `wal_store.append_segment`
//! but before the manifest CAS, leaving a WAL segment unreferenced. The
//! write-side WAL janitor lives elsewhere (TODO); here we focus on SSTs
//! and their bloom side-cars.
//!
//! ## What the janitor does
//!
//! 1. Loads `manifest/current.json` and, for every manifest version from
//! the caller-supplied retention horizon to current, unions the "live"
//! relative paths (SST body, bloom side-car, unique/equality/label index
//! side-cars). The horizon is the oldest version any live reader is
//! pinned to (RFC-027), so a reader still reading an old version keeps
//! every object that version needs in the live set.
//! 2. Lists `sst/level0/`, `sst/level1/`, … up to a configurable max level.
//! 3. For every listed object not in the live set, checks its
//! `last_modified` age. Any object younger than `min_age` is skipped —
//! this is a secondary guard against an in-flight writer whose body PUT
//! succeeded a moment ago and whose manifest CAS is still in flight (the
//! object is referenced by no version yet).
//! 4. Older objects are reported as orphans and (when `delete = true`)
//! removed via `ObjectStore::delete`.
//!
//! ## Safety
//!
//! The retention horizon is the correctness mechanism: an object the sweep
//! deletes is referenced by no manifest version at or above the horizon, so
//! no live reader can reach it. This covers both compaction inputs merged
//! away before the horizon and orphans from failed commits, with no
//! time-based guess. `min_age` remains as a small secondary guard for the
//! body-PUT-then-CAS race; `delete = false` keeps a dry-run available for
//! operators who want to review a run before trusting it.

use std::collections::HashSet;

use chrono::Utc;
use futures::TryStreamExt;
use object_store::ObjectStoreExt;
use tracing::{debug, instrument};

use crate::error::{Error, Result};
use crate::manifest::ManifestStore;

/// Outcome of a [`sweep_orphans`] invocation. All counters reflect the
/// behaviour requested by the caller — when `delete = false` (dry run),
/// `orphans_deleted` is always zero and `bytes_freed` reports what *would*
/// be freed.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct JanitorReport {
    /// Distinct objects classified as orphan (not referenced by the
    /// current manifest, older than `min_age`).
    pub orphans_found: usize,
    /// Objects actually deleted by this run. Equal to `orphans_found`
    /// when `delete = true`; zero otherwise.
    pub orphans_deleted: usize,
    /// Bytes freed (or that would have been freed in dry-run mode).
    pub bytes_freed: u64,
    /// Objects that are unreferenced but were spared because their
    /// `last_modified` falls within `min_age`. These are the candidates
    /// the operator should re-evaluate on the next sweep.
    pub skipped_too_young: usize,
}

/// Scan `sst/level{0..max_level}/` for objects not referenced by the
/// current manifest and (when `delete = true`) remove the ones older than
/// `min_age`. See module docs for the safety reasoning.
///
/// The function loads the manifest **once** at the start of the sweep.
/// If a writer commits a fresh manifest while we are listing objects, any
/// SSTs that became newly-referenced after our load are still treated as
/// orphans here — but the `min_age` window protects them from deletion as
/// long as the operator picks a sensible value.
#[instrument(
 skip(manifest_store),
 fields(
 namespace = %manifest_store.paths().namespace(),
 retention_horizon,
 min_age_secs = min_age.as_secs(),
 delete,
 max_level,
 )
)]
pub async fn sweep_orphans(
    manifest_store: &ManifestStore,
    retention_horizon: u64,
    min_age: std::time::Duration,
    max_level: u32,
    delete: bool,
) -> Result<JanitorReport> {
    let current = manifest_store.load_current().await?;
    let current_version = current.manifest.version;
    // The horizon is the oldest manifest version any live reader is pinned
    // to (RFC-027). Clamp defensively to the current version.
    let horizon = retention_horizon.min(current_version);

    // Build the live object set from the union of every retained manifest
    // version from the horizon to current (inclusive). A reader pinned at
    // `horizon` still needs every object that version references, so none of
    // them can be swept; an object only an older version referenced (a
    // compaction input merged away before the horizon, say) drops out of the
    // set and becomes reclaimable. This is what makes deletion safe by
    // construction rather than by a wall-clock guess.
    let mut referenced: HashSet<String> = HashSet::new();
    let mut mark_live = |ssts: &[crate::manifest::SstDescriptor]| {
        for desc in ssts {
            referenced.insert(desc.path.clone());
            if let Some(b) = &desc.bloom {
                referenced.insert(b.path.clone());
            }
            // Side-car bodies live in the same `sst/level{N}/` prefix the
            // sweep scans, so they must be marked live too — otherwise the
            // sweep deletes unique/equality/label-index side-cars a retained
            // manifest still references, breaking point lookups and (with the
            // typed-column layout) label scans.
            for u in &desc.unique_property_indices {
                referenced.insert(u.path.clone());
            }
            for e in &desc.equality_property_indices {
                referenced.insert(e.path.clone());
            }
            if let Some(li) = &desc.label_index {
                referenced.insert(li.path.clone());
            }
        }
    };
    for version in horizon..=current_version {
        if version == current_version {
            mark_live(&current.manifest.ssts);
        } else {
            let manifest = manifest_store.load_manifest_at(version).await?;
            mark_live(&manifest.ssts);
        }
    }

    let store = manifest_store.store().clone();
    let paths = manifest_store.paths();
    let ns_prefix = paths.namespace_prefix();
    let ns_prefix_str = ns_prefix.as_ref();

    let mut report = JanitorReport::default();
    let min_age_secs = min_age.as_secs() as i64;
    let now = Utc::now();

    for level in 0..=max_level {
        let level_dir = paths.sst_dir(level);
        let mut stream = store.list(Some(&level_dir));
        while let Some(meta) = stream.try_next().await.map_err(Error::ObjectStore)? {
            let absolute = meta.location.as_ref();
            // Convert to namespace-relative form so the comparison matches
            // what's stored in `SstDescriptor::path`.
            let Some(relative) = absolute
                .strip_prefix(ns_prefix_str)
                .and_then(|s| s.strip_prefix('/'))
            else {
                debug!(path = %absolute, "list returned object outside namespace prefix; skipping");
                continue;
            };
            if referenced.contains(relative) {
                continue;
            }
            let age_secs = (now - meta.last_modified).num_seconds();
            if age_secs < min_age_secs {
                report.skipped_too_young += 1;
                debug!(path = %absolute, age_secs, "orphan candidate too young, deferring");
                continue;
            }
            report.orphans_found += 1;
            report.bytes_freed = report.bytes_freed.saturating_add(meta.size);
            if delete {
                store
                    .delete(&meta.location)
                    .await
                    .map_err(Error::ObjectStore)?;
                report.orphans_deleted += 1;
            }
        }
    }

    Ok(report)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::time::Duration;

    use bytes::Bytes;
    use namidb_core::{LabelDef, NamespaceId, NodeId, PropertyDef, Schema, SchemaBuilder};
    use object_store::memory::InMemory;
    use object_store::{ObjectStore, PutPayload};
    use uuid::Uuid;

    use super::*;
    use crate::fence::WriterFence;
    use crate::flush::{flush, NodeWriteRecord};
    use crate::manifest::ManifestStore;
    use crate::memtable::{MemKey, MemOp, Memtable};
    use crate::paths::NamespacePaths;
    use namidb_core::{DataType, Value};

    fn make_store() -> Arc<dyn ObjectStore> {
        Arc::new(InMemory::new())
    }

    fn make_paths(name: &str) -> NamespacePaths {
        NamespacePaths::new("tenants", NamespaceId::new(name).unwrap())
    }

    fn person_label() -> LabelDef {
        LabelDef {
            name: "Person".into(),
            properties: vec![PropertyDef::new("name", DataType::Utf8, false).unwrap()],
        }
    }

    fn node_payload(name: &str) -> Bytes {
        let mut props = std::collections::BTreeMap::new();
        props.insert("name".into(), Value::Str(name.into()));
        NodeWriteRecord {
            properties: props,
            schema_version: 1,
            ..Default::default()
        }
        .encode()
        .unwrap()
    }

    fn sorted_node_id(b: u8) -> NodeId {
        let mut bytes = [0u8; 16];
        bytes[15] = b;
        NodeId::from_uuid(Uuid::from_bytes(bytes))
    }

    async fn flush_one_node(
        ms: &ManifestStore,
        fence: &WriterFence,
        base: &crate::manifest::LoadedManifest,
        schema: &Schema,
        id: NodeId,
        name: &str,
        lsn: u64,
    ) -> crate::manifest::LoadedManifest {
        let mut mt = Memtable::new();
        mt.apply(MemKey::Node { id }, lsn, MemOp::Upsert(node_payload(name)));
        let frozen = mt.freeze();
        flush(ms, fence, base, &frozen, schema.clone())
            .await
            .unwrap()
            .committed
    }

    #[tokio::test]
    async fn sweep_finds_no_orphans_when_manifest_references_everything() {
        let store = make_store();
        let paths = make_paths("janitor-clean");
        let ms = ManifestStore::new(store.clone(), paths.clone());
        let base = ms.bootstrap(Uuid::now_v7()).await.unwrap();
        let fence = WriterFence::new(base.manifest.epoch);
        let schema = SchemaBuilder::new().label(person_label()).unwrap().build();

        let _after = flush_one_node(&ms, &fence, &base, &schema, sorted_node_id(1), "A", 1).await;

        let report = sweep_orphans(&ms, u64::MAX, Duration::from_secs(0), 4, true)
            .await
            .unwrap();
        assert_eq!(report.orphans_found, 0);
        assert_eq!(report.orphans_deleted, 0);
        assert_eq!(report.bytes_freed, 0);
        assert_eq!(report.skipped_too_young, 0);
    }

    #[tokio::test]
    async fn sweep_identifies_and_deletes_a_planted_orphan() {
        let store = make_store();
        let paths = make_paths("janitor-orphan");
        let ms = ManifestStore::new(store.clone(), paths.clone());
        let base = ms.bootstrap(Uuid::now_v7()).await.unwrap();
        let fence = WriterFence::new(base.manifest.epoch);
        let schema = SchemaBuilder::new().label(person_label()).unwrap().build();

        // Real, manifest-referenced SST so the live set is non-empty.
        let _after = flush_one_node(&ms, &fence, &base, &schema, sorted_node_id(1), "A", 1).await;

        // Plant an extra body under sst/level0/ that no manifest references.
        let orphan = paths.sst_object(0, "0000-orphan-Person.parquet");
        let body: Bytes = b"orphan-body-bytes".to_vec().into();
        let orphan_size = body.len() as u64;
        store.put(&orphan, PutPayload::from(body)).await.unwrap();

        // Dry run: report should flag the orphan but the body must remain.
        let dry = sweep_orphans(&ms, u64::MAX, Duration::from_secs(0), 4, false)
            .await
            .unwrap();
        assert_eq!(dry.orphans_found, 1);
        assert_eq!(dry.orphans_deleted, 0);
        assert_eq!(dry.bytes_freed, orphan_size);
        assert!(store.head(&orphan).await.is_ok(), "dry run must not delete");

        // Real run: deletes the orphan.
        let real = sweep_orphans(&ms, u64::MAX, Duration::from_secs(0), 4, true)
            .await
            .unwrap();
        assert_eq!(real.orphans_found, 1);
        assert_eq!(real.orphans_deleted, 1);
        assert_eq!(real.bytes_freed, orphan_size);
        assert!(
            store.head(&orphan).await.is_err(),
            "orphan must be gone after real sweep"
        );

        // Idempotent: a second sweep finds nothing.
        let again = sweep_orphans(&ms, u64::MAX, Duration::from_secs(0), 4, true)
            .await
            .unwrap();
        assert_eq!(again.orphans_found, 0);
    }

    #[tokio::test]
    async fn sweep_respects_min_age_safety_window() {
        let store = make_store();
        let paths = make_paths("janitor-young");
        let ms = ManifestStore::new(store.clone(), paths.clone());
        let _base = ms.bootstrap(Uuid::now_v7()).await.unwrap();

        // Plant a fresh orphan.
        let orphan = paths.sst_object(0, "young-orphan.parquet");
        store
            .put(&orphan, PutPayload::from(Bytes::from_static(b"recent")))
            .await
            .unwrap();

        // min_age = 24h → the freshly-written orphan must be skipped.
        let report = sweep_orphans(&ms, u64::MAX, Duration::from_secs(86_400), 4, true)
            .await
            .unwrap();
        assert_eq!(report.orphans_found, 0);
        assert_eq!(report.orphans_deleted, 0);
        assert_eq!(report.skipped_too_young, 1);
        assert!(
            store.head(&orphan).await.is_ok(),
            "young orphan must survive the sweep"
        );
    }

    #[tokio::test]
    async fn sweep_respects_max_level_window() {
        let store = make_store();
        let paths = make_paths("janitor-levels");
        let ms = ManifestStore::new(store.clone(), paths.clone());
        let _base = ms.bootstrap(Uuid::now_v7()).await.unwrap();

        // Plant orphans at L0 and L3.
        let l0 = paths.sst_object(0, "l0-orphan.parquet");
        let l3 = paths.sst_object(3, "l3-orphan.parquet");
        store
            .put(&l0, PutPayload::from(Bytes::from_static(b"l0")))
            .await
            .unwrap();
        store
            .put(&l3, PutPayload::from(Bytes::from_static(b"l3")))
            .await
            .unwrap();

        // max_level = 1 catches only the L0 body.
        let report = sweep_orphans(&ms, u64::MAX, Duration::from_secs(0), 1, true)
            .await
            .unwrap();
        assert_eq!(report.orphans_found, 1);
        assert!(store.head(&l0).await.is_err(), "l0 orphan must be deleted");
        assert!(store.head(&l3).await.is_ok(), "l3 orphan must survive");
    }
}
