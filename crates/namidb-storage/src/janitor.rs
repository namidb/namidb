//! Stateless janitor for orphaned SST + bloom side-car objects and
//! superseded manifest snapshots.
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
//! but before the manifest CAS, leaving a WAL segment unreferenced.
//! - **Flushed WAL segments.** A flush clears `wal_segments` from the new
//! manifest version but never deletes the segment objects, so every
//! commit leaves one immutable `wal/<seq>.wal` behind forever — and
//! `WriterSession::open` LISTs the whole `wal/` prefix, so cold-open
//! cost grows with total history, not live state.
//! - **Superseded memtable snapshots.** `memtable_snapshot.bin` is
//! overwritten in place but a commit or flush can strand it bound to a WAL
//! closure the current manifest no longer references (recovery already
//! ignores it; the object is pure storage cost).
//!
//! ## What the janitor does
//!
//! 0. Lists `manifest/pins/` for retention pin leases ([`crate::pin`]).
//! Every unexpired floor lease lowers the retention horizon to its pinned
//! version, so a cross-process reader keeps the closure it is reading
//! alive; a point lease (a running backup) keeps exactly its version's
//! closure; a window lease (the restore point window,
//! [`crate::restore_point`]) keeps every manifest body from its version up
//! plus every object written since the window opened. Expired leases are
//! ignored and deleted.
//! 1. Loads `manifest/current.json` and, for every manifest version from
//! the caller-supplied retention horizon to current, unions the "live"
//! relative paths (SST body, bloom side-car, unique/equality/label index
//! side-cars). The horizon is the oldest version any live reader is
//! pinned to (RFC-027), so a reader still reading an old version keeps
//! every object that version needs in the live set.
//! 2. Lists the namespace's `sst/` prefix recursively and discovers every
//! canonical numeric `levelN/` present. The configured max level remains a
//! compatibility hint, not a correctness boundary: a failed compaction CAS
//! can strand its output one level deeper than every retained manifest.
//! 3. For every listed object not in the live set, checks its
//! `last_modified` age. Any object younger than `min_age` is skipped —
//! this is a secondary guard against an in-flight writer whose body PUT
//! succeeded a moment ago and whose manifest CAS is still in flight (the
//! object is referenced by no version yet).
//! 4. Older objects are reported as orphans and (when `delete = true`)
//! removed via `ObjectStore::delete`.
//! 5. Lists `manifest/` and reclaims every `manifest/v{N}.json` whose version
//! `N` is strictly below the horizon — a retired version no live reader can
//! load — under the same `min_age` guard. `current.json` and every version
//! at or above the horizon are kept. Without this the `manifest/` prefix
//! grows by one immutable snapshot per commit forever.
//! 6. Lists `wal/` and reclaims every segment referenced by no retained
//! manifest version, under the same `min_age` guard (see the deletion
//! rule inline at the sweep). Also removes a stale `memtable_snapshot.bin`
//! once its exact WAL binding no longer matches the current manifest.
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
use tracing::{debug, instrument, warn};

use crate::error::{Error, Result};
use crate::manifest::ManifestStore;

/// Outcome of a [`sweep_orphans`] invocation. All counters reflect the
/// behaviour requested by the caller — when `delete = false` (dry run),
/// `orphans_deleted` is always zero and `bytes_freed` reports what *would*
/// be freed.
#[derive(Debug, Clone, Default, PartialEq, Eq, serde::Serialize)]
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
    /// Superseded manifest snapshots (`manifest/v{N}.json` strictly below the
    /// retention horizon) reclaimable this sweep. Like `orphans_found`, this is
    /// the candidate count and is populated in dry-run too; the bodies are
    /// physically removed only when `delete = true` (consult the caller's
    /// dry-run flag). Counted separately from `orphans_found` because a retired
    /// manifest version is not an orphan — it is a version no live reader can
    /// still load, reclaimable by the same retention-horizon argument.
    pub manifest_snapshots_reclaimed: usize,
    /// Bytes held by the manifest snapshots in `manifest_snapshots_reclaimed`
    /// (freed when `delete = true`, otherwise what *would* be freed).
    pub manifest_bytes_freed: u64,
    /// Superseded pointer files (`manifest/pointer/p<N>.json` strictly below the
    /// retention horizon, RFC-029) reclaimable this sweep. Counted separately
    /// from `manifest_snapshots_reclaimed` for the same reason: a retired
    /// pointer is not an orphan, it is a version no live reader can still load.
    /// Keeping every pointer at or above the horizon also keeps the family
    /// contiguous, which is what makes `load_current`'s forward HEAD probe
    /// gap-safe. Populated in dry-run too; bodies removed only when
    /// `delete = true`.
    pub pointer_files_reclaimed: usize,
    /// Bytes held by the pointer files in `pointer_files_reclaimed` (freed when
    /// `delete = true`, otherwise what *would* be freed).
    pub pointer_bytes_freed: u64,
    /// Dead WAL segments (`wal/<seq>.wal` referenced by no manifest version
    /// at or above the retention horizon, older than `min_age`) reclaimable
    /// this sweep. Populated in dry-run too; bodies removed only when
    /// `delete = true`. Counted separately from `orphans_found` because a
    /// flushed segment is not an orphan — every retained version stopped
    /// referencing it on purpose.
    pub wal_segments_reclaimed: usize,
    /// Bytes held by the segments in `wal_segments_reclaimed` (freed when
    /// `delete = true`, otherwise what *would* be freed).
    pub wal_bytes_freed: u64,
    /// Stale `memtable_snapshot.bin` objects reclaimable this sweep (0 or 1:
    /// there is one snapshot path per namespace). Stale means its exact v2
    /// `wal_segments` binding differs from the current manifest, so current
    /// recovery cannot consume it. Populated in dry-run too; the body is
    /// removed only when `delete = true`.
    pub memtable_snapshots_reclaimed: usize,
    /// Bytes held by the snapshot in `memtable_snapshots_reclaimed` (freed
    /// when `delete = true`, otherwise what *would* be freed).
    pub memtable_snapshot_bytes_freed: u64,
    /// Unexpired retention pin leases (`manifest/pins/`, see [`crate::pin`])
    /// found by this sweep. Each one lowered the retention horizon to at most
    /// its pinned version, so the closure a cross-process reader (a running
    /// backup) depends on stayed live.
    pub pins_honoured: usize,
    /// Expired pin leases reclaimable this sweep. Ignored for the horizon and
    /// (when `delete = true`) removed, so a crashed holder cannot pin the
    /// namespace forever. Populated in dry-run too.
    pub expired_pins_reclaimed: usize,
    /// `true` when the pre-delete pin re-check found a lease that landed
    /// while the live set was being built, pinning a version below this
    /// sweep's horizon: nothing was deleted this pass; the next tick
    /// recomputes with the lease in view.
    pub aborted_by_pin: bool,
    /// Unreachable SST, side-car and WAL objects kept only because a restore
    /// point window lease ([`crate::restore_point`]) covers their write time:
    /// a manifest version inside the window may reference them. This is the
    /// object-count side of the window's storage cost.
    pub restore_window_retained: usize,
    /// Bytes held by the objects in `restore_window_retained`.
    pub restore_window_bytes: u64,
    /// SST, side-car and WAL objects reclaimed through the superseded fast
    /// path: referenced by a manifest version this sweep retires, so they were
    /// once installed and cannot be an in-flight upload, and guarded only by
    /// [`SweepOptions::superseded_min_age`]. A subset of `orphans_found` +
    /// `wal_segments_reclaimed`. Populated in dry-run too.
    pub superseded_reclaimed: usize,
    /// Bytes held by the objects in `superseded_reclaimed`.
    pub superseded_bytes_freed: u64,
    /// Manifest versions retired by this sweep that it loaded to find the
    /// objects they referenced (the superseded fast path).
    pub retired_versions_loaded: usize,
    /// Manifest versions loaded to build the live set (current included).
    pub live_versions_loaded: usize,
    /// `true` when the live set was built from every manifest version in the
    /// restore point window instead of keeping window history by write time
    /// ([`SweepOptions::exact_window`]).
    pub exact_window: bool,
    /// Total bytes of every object listed under `sst/` (bodies and side-cars),
    /// before this sweep deleted anything.
    pub store_sst_bytes: u64,
    /// Total bytes of every object listed under `wal/`, before deletion.
    pub store_wal_bytes: u64,
    /// Total bytes of every object listed under `manifest/` (bodies, pointers
    /// and pin leases), before deletion.
    pub store_manifest_bytes: u64,
    /// Bytes of the listed `sst/` objects in the live set (referenced by a
    /// retained manifest version).
    pub live_sst_bytes: u64,
    /// Bytes of the listed WAL segments in the live set.
    pub live_wal_bytes: u64,
}

impl JanitorReport {
    /// Bytes of the SST and WAL objects in the live set.
    pub fn live_bytes(&self) -> u64 {
        self.live_sst_bytes.saturating_add(self.live_wal_bytes)
    }

    /// `true` when the sweep found anything worth an info log line.
    pub fn has_activity(&self) -> bool {
        self.orphans_found > 0
            || self.manifest_snapshots_reclaimed > 0
            || self.pointer_files_reclaimed > 0
            || self.wal_segments_reclaimed > 0
            || self.memtable_snapshots_reclaimed > 0
            || self.restore_window_retained > 0
    }
}

/// Default for [`SweepOptions::superseded_min_age`].
pub const DEFAULT_SUPERSEDED_MIN_AGE: std::time::Duration = std::time::Duration::from_secs(3600);

/// How many manifest bodies a sweep reads at once while building its live
/// set or loading the versions it retires. Bounds memory to a few bodies.
const MANIFEST_LOAD_CONCURRENCY: usize = 8;

/// Parameters of [`sweep_orphans_with`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SweepOptions {
    /// Oldest manifest version any in-process reader is pinned to (RFC-027).
    pub retention_horizon: u64,
    /// Age an unreferenced object must reach before deletion when the sweep
    /// cannot prove it was ever installed: it may be an upload whose manifest
    /// CAS has not landed yet.
    pub min_age: std::time::Duration,
    /// Age guard for an object referenced by a manifest version this sweep
    /// retires (and by no retained one). Such an object was installed once,
    /// so it is not an in-flight upload; the guard is only a margin for
    /// readers outside this process that hold no retention pin. A retired
    /// version must also have been superseded (its successor committed) at
    /// least this long ago. Never longer than `min_age`.
    pub superseded_min_age: std::time::Duration,
    /// Expected deepest SST level (a diagnostics hint, not a bound).
    pub max_level: u32,
    /// `false` runs a dry run that only reports.
    pub delete: bool,
    /// Build the live set from EVERY manifest version in the restore point
    /// window (one body at a time) instead of keeping every object written
    /// since the window opened. Exact, so it also reclaims orphans written
    /// inside the window (failed uploads, abandoned compaction outputs);
    /// costs one manifest read per version in the window.
    pub exact_window: bool,
}

impl SweepOptions {
    /// The options [`sweep_orphans`] uses: superseded objects wait
    /// `min(min_age, DEFAULT_SUPERSEDED_MIN_AGE)`, window history is kept by
    /// write time.
    pub fn new(
        retention_horizon: u64,
        min_age: std::time::Duration,
        max_level: u32,
        delete: bool,
    ) -> Self {
        Self {
            retention_horizon,
            min_age,
            superseded_min_age: min_age.min(DEFAULT_SUPERSEDED_MIN_AGE),
            max_level,
            delete,
            exact_window: false,
        }
    }
}

/// Every namespace-relative object path one SST descriptor references: the
/// body plus its bloom and every side-car. This is the per-descriptor half of
/// the sweep's live set, shared with callers that need to know what a
/// manifest (or a prepared compaction) points at.
pub fn descriptor_object_paths(desc: &crate::manifest::SstDescriptor) -> Vec<String> {
    let mut out = Vec::with_capacity(4);
    out.push(desc.path.clone());
    if let Some(b) = &desc.bloom {
        out.push(b.path.clone());
    }
    // Side-car bodies live in the same `sst/level{N}/` prefix the sweep
    // scans, so they must be marked live too — otherwise the sweep deletes
    // unique/equality/label-index side-cars a retained manifest still
    // references, breaking point lookups and (with the typed-column layout)
    // label scans.
    for u in &desc.unique_property_indices {
        out.push(u.path.clone());
        if let Some(paged) = &u.paged {
            out.push(paged.path.clone());
        }
    }
    for e in &desc.equality_property_indices {
        out.push(e.path.clone());
        if let Some(paged) = &e.paged {
            out.push(paged.path.clone());
        }
    }
    for c in &desc.composite_equality_indices {
        out.push(c.path.clone());
        if let Some(paged) = &c.paged {
            out.push(paged.path.clone());
        }
    }
    if let Some(li) = &desc.label_index {
        out.push(li.path.clone());
    }
    if let Some(locator) = &desc.node_locator {
        out.push(locator.path.clone());
        if let Some(properties) = &locator.property_pages {
            out.push(properties.path.clone());
        }
    }
    if let Some(point) = crate::manifest::edge_point_sidecar_path(desc) {
        out.push(point);
    }
    out
}

/// Every namespace-relative `sst/` object path `manifest` references.
pub fn manifest_object_paths(manifest: &crate::manifest::Manifest) -> HashSet<String> {
    manifest
        .ssts
        .iter()
        .flat_map(descriptor_object_paths)
        .collect()
}

/// The objects a set of manifest versions references: `sst/` paths
/// (namespace-relative) and WAL segment seqs.
#[derive(Debug, Default)]
struct ObjectSet {
    paths: HashSet<String>,
    wal: HashSet<u64>,
    /// Deepest SST level any marked version occupies.
    max_level: u32,
}

impl ObjectSet {
    fn mark(&mut self, manifest: &crate::manifest::Manifest) {
        for seg in &manifest.wal_segments {
            self.wal.insert(seg.seq);
        }
        for desc in &manifest.ssts {
            self.max_level = self.max_level.max(desc.level.as_u32());
            for path in descriptor_object_paths(desc) {
                self.paths.insert(path);
            }
        }
    }

    /// Mark what `manifest` references and `live` does not.
    fn mark_outside(&mut self, manifest: &crate::manifest::Manifest, live: &ObjectSet) {
        for seg in &manifest.wal_segments {
            if !live.wal.contains(&seg.seq) {
                self.wal.insert(seg.seq);
            }
        }
        for desc in &manifest.ssts {
            for path in descriptor_object_paths(desc) {
                if !live.paths.contains(&path) {
                    self.paths.insert(path);
                }
            }
        }
    }
}

/// The restore point window a sweep honours: the oldest window lease's floor
/// version and the instant it opened.
#[derive(Debug, Clone, Copy)]
struct RestoreWindow {
    version: u64,
    since: chrono::DateTime<Utc>,
}

/// The pre-delete re-check in [`sweep_orphans`]: `true` when an unexpired
/// lease now asks for more than the sweep honoured when it listed the pins —
/// a lease it never saw, or one rewritten to an older version or an earlier
/// window — and that request reaches below the dense `horizon` (everything
/// at or above it is in the live set whatever the lease's scope). Read-only:
/// no reclaim, no report side effects.
async fn pins_changed_below_horizon(
    store: &dyn object_store::ObjectStore,
    paths: &crate::paths::NamespacePaths,
    now_unix: i64,
    honoured: &std::collections::HashMap<String, crate::pin::PinLease>,
    horizon: u64,
) -> Result<bool> {
    use crate::pin::PinScope;
    let mut pins = store.list(Some(&paths.pins_dir()));
    while let Some(meta) = pins.try_next().await.map_err(Error::ObjectStore)? {
        let body = match store.get(&meta.location).await {
            Ok(res) => res.bytes().await.map_err(Error::ObjectStore)?,
            Err(object_store::Error::NotFound { .. }) => continue,
            Err(other) => return Err(Error::ObjectStore(other)),
        };
        let Ok(lease) = serde_json::from_slice::<crate::pin::PinLease>(&body) else {
            continue;
        };
        if lease.expires_at_unix < now_unix || lease.version >= horizon {
            continue;
        }
        let covered = match honoured.get(meta.location.as_ref()) {
            None => false,
            Some(seen) => match (seen.scope, lease.scope) {
                (PinScope::Floor, _) => seen.version <= lease.version,
                (PinScope::Point, PinScope::Point) => seen.version == lease.version,
                (PinScope::Window { since: a }, PinScope::Window { since: b }) => {
                    seen.version <= lease.version && a <= b
                }
                _ => false,
            },
        };
        if !covered {
            return Ok(true);
        }
    }
    Ok(false)
}

/// Scan every canonical numeric level discovered below `sst/` for objects not
/// referenced by the current manifest and (when `delete = true`) remove the
/// ones older than `min_age`, then reclaim manifest snapshots retired below
/// the retention horizon. `max_level` is retained as an expected-level hint
/// for diagnostics; deeper levels are still scanned because a failed
/// compaction CAS can leave output beyond the manifest high-water mark. See
/// module docs for the safety reasoning.
///
/// The function loads the manifest **once** at the start of the sweep.
/// If a writer commits a fresh manifest while we are listing objects, any
/// SSTs that became newly-referenced after our load are still treated as
/// orphans here — but the `min_age` window protects them from deletion as
/// long as the operator picks a sensible value.
///
/// Retention pin leases (`manifest/pins/`, [`crate::pin`]) are listed once
/// at the start too: every unexpired lease lowers the horizon to its pinned
/// version for this whole sweep, expired leases are reclaimed. A lease
/// written *after* that listing is not seen until the next sweep, which is
/// why pin holders must re-check their pinned root after acquiring.
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
    sweep_orphans_with(
        manifest_store,
        SweepOptions::new(retention_horizon, min_age, max_level, delete),
    )
    .await
}

/// Load manifest versions `versions` (at most [`MANIFEST_LOAD_CONCURRENCY`]
/// bodies in flight) and hand each to `visit` in version order. A version
/// that is gone is passed as `None`; any other error aborts.
async fn for_each_manifest_version(
    manifest_store: &ManifestStore,
    versions: impl IntoIterator<Item = u64>,
    mut visit: impl FnMut(u64, Option<&crate::manifest::Manifest>) -> Result<()>,
) -> Result<()> {
    use futures::StreamExt;
    let mut loads = futures::stream::iter(versions)
        .map(|version| async move {
            match manifest_store.load_manifest_at(version).await {
                Ok(manifest) => Ok((version, Some(manifest))),
                Err(Error::ObjectStore(object_store::Error::NotFound { .. })) => {
                    Ok((version, None))
                }
                Err(e) => Err(e),
            }
        })
        .buffered(MANIFEST_LOAD_CONCURRENCY);
    while let Some(loaded) = loads.next().await {
        let (version, manifest) = loaded?;
        visit(version, manifest.as_ref())?;
    }
    Ok(())
}

/// Parse a `manifest/` object name: `v{16-hex}.json` bodies and
/// `p{16-hex}.json` pointers, as `(is_pointer, version)`.
fn parse_manifest_object(location: &object_store::path::Path) -> Option<(bool, u64)> {
    let filename = location.filename()?;
    filename
        .strip_prefix('v')
        .map(|rest| (false, rest))
        .or_else(|| filename.strip_prefix('p').map(|rest| (true, rest)))
        .and_then(|(is_pointer, rest)| rest.strip_suffix(".json").map(|h| (is_pointer, h)))
        .and_then(|(is_pointer, hex)| u64::from_str_radix(hex, 16).ok().map(|v| (is_pointer, v)))
}

/// [`sweep_orphans`] with every knob exposed (see [`SweepOptions`]).
///
/// On top of the retention-horizon sweep this adds two things:
///
/// - **The superseded fast path.** Every manifest version below the retained
///   floor whose successor committed at least `superseded_min_age` ago is
///   loaded, and the objects it references that no retained version
///   references are reclaimed after that short grace instead of `min_age`:
///   they were installed once, so they cannot be an upload whose manifest
///   CAS is still in flight. Objects go first, then the retired manifest
///   bodies, so an interrupted sweep never forgets what a version held.
/// - **The exact restore window** (`exact_window`). The live set is the union
///   of every manifest version in the restore point window instead of "every
///   object written since the window opened", so garbage written inside the
///   window (failed uploads, abandoned compaction outputs) is reclaimed after
///   `min_age`. If any version inside the window is missing the sweep cannot
///   prove the window's closure and falls back to keeping by write time.
#[instrument(
 skip(manifest_store),
 fields(
 namespace = %manifest_store.paths().namespace(),
 retention_horizon = opts.retention_horizon,
 min_age_secs = opts.min_age.as_secs(),
 superseded_min_age_secs = opts.superseded_min_age.as_secs(),
 exact_window = opts.exact_window,
 delete = opts.delete,
 )
)]
pub async fn sweep_orphans_with(
    manifest_store: &ManifestStore,
    opts: SweepOptions,
) -> Result<JanitorReport> {
    let SweepOptions {
        retention_horizon,
        min_age,
        max_level,
        delete,
        ..
    } = opts;
    let superseded_min_age = opts.superseded_min_age.min(min_age);
    let current = manifest_store.load_current().await?;
    let current_version = current.manifest.version;

    let store = manifest_store.store().clone();
    let paths = manifest_store.paths();
    let mut report = JanitorReport::default();
    let min_age_secs = min_age.as_secs() as i64;
    let superseded_secs = superseded_min_age.as_secs() as i64;
    let now = Utc::now();

    // Retention pin leases (see `crate::pin`): cross-process readers — a
    // running backup, the restore point window — that the in-process horizon
    // cannot see. Listed BEFORE anything is classified or deleted, so every
    // unexpired lease is honoured, for its scope, for the whole sweep: a
    // floor lowers the horizon, a point keeps one version, a window keeps
    // history by write time (see `PinScope`). An expired lease is void (its holder
    // crashed or stalled past the TTL): ignored for the horizon and deleted,
    // so a dead holder cannot pin the namespace forever. An undecodable body
    // under the prefix is not ours to judge — it names no version, so it pins
    // nothing, and it is left alone.
    let mut pin_floor = u64::MAX;
    // Point leases: versions kept on their own, whatever the horizon.
    let mut point_versions: std::collections::BTreeSet<u64> = Default::default();
    // The restore point window (the oldest, if several holders wrote one).
    let mut window: Option<RestoreWindow> = None;
    // Every lease honoured, by path, for the pre-delete re-check.
    let mut honoured: std::collections::HashMap<String, crate::pin::PinLease> = Default::default();
    let now_unix = now.timestamp();
    let mut pins = store.list(Some(&paths.pins_dir()));
    while let Some(meta) = pins.try_next().await.map_err(Error::ObjectStore)? {
        let body = match store.get(&meta.location).await {
            Ok(res) => res.bytes().await.map_err(Error::ObjectStore)?,
            // Raced a holder's release between LIST and GET: the pin is gone.
            Err(object_store::Error::NotFound { .. }) => continue,
            Err(other) => return Err(Error::ObjectStore(other)),
        };
        let Ok(lease) = serde_json::from_slice::<crate::pin::PinLease>(&body) else {
            debug!(path = %meta.location, "undecodable pin lease; leaving it alone");
            continue;
        };
        if lease.expires_at_unix < now_unix {
            report.expired_pins_reclaimed += 1;
            if delete {
                store
                    .delete(&meta.location)
                    .await
                    .map_err(Error::ObjectStore)?;
            }
            continue;
        }
        report.pins_honoured += 1;
        match lease.scope {
            crate::pin::PinScope::Floor => pin_floor = pin_floor.min(lease.version),
            crate::pin::PinScope::Point => {
                point_versions.insert(lease.version);
            }
            crate::pin::PinScope::Window { since } => {
                window = Some(match window {
                    None => RestoreWindow {
                        version: lease.version,
                        since,
                    },
                    Some(w) => RestoreWindow {
                        version: w.version.min(lease.version),
                        since: w.since.min(since),
                    },
                });
            }
        }
        honoured.insert(meta.location.as_ref().to_string(), lease);
    }

    // The advisory `current.json` may lag the newest pointer (a commit
    // defers its PUT, see `ManifestStore::cas_pointer_deferring_advisory`).
    // When a LIST of the pointer family comes back stale and empty, the
    // reader starts its forward probe from the version the advisory names,
    // so that pointer and every one above it must stay. Read it once, before
    // anything is deleted; a newer advisory only raises the version.
    let advisory_floor = match store.get(&paths.current_pointer()).await {
        Ok(res) => {
            let body = res.bytes().await.map_err(Error::ObjectStore)?;
            serde_json::from_slice::<crate::manifest::ManifestPointer>(&body)
                .ok()
                .map(|pointer| pointer.version)
        }
        Err(object_store::Error::NotFound { .. }) => None,
        Err(other) => return Err(Error::ObjectStore(other)),
    };

    // The horizon is the oldest manifest version any live reader is pinned
    // to: the in-process reader horizon (RFC-027) unioned with every
    // unexpired pin lease. Clamp defensively to the current version.
    let horizon = retention_horizon.min(pin_floor).min(current_version);
    // Manifest bodies at or above this version are kept: the dense horizon,
    // or the restore point window's floor when it reaches further back.
    let body_floor = window.map_or(horizon, |w| horizon.min(w.version));
    // Versions at or above this floor are retained for in-process readers and
    // must load; versions below it are retained only because of a pin lease
    // and may already have been reclaimed before the lease existed (the pin
    // holder re-checks its own root and fails loudly in that case).
    let strict_floor = retention_horizon.min(current_version);

    // List `manifest/` once: the store accounting, the retired bodies the
    // superseded fast path loads, and their successors' commit times.
    let mut manifest_objects: Vec<object_store::ObjectMeta> = Vec::new();
    let mut body_modified: std::collections::BTreeMap<u64, chrono::DateTime<Utc>> =
        Default::default();
    let mut manifest_listing = store.list(Some(&paths.manifest_dir()));
    while let Some(meta) = manifest_listing
        .try_next()
        .await
        .map_err(Error::ObjectStore)?
    {
        report.store_manifest_bytes = report.store_manifest_bytes.saturating_add(meta.size);
        if meta.location.as_ref().contains("/pins/") {
            continue;
        }
        if let Some((false, version)) = parse_manifest_object(&meta.location) {
            body_modified.insert(version, meta.last_modified);
        }
        manifest_objects.push(meta);
    }

    // Build the live object set from the union of every retained manifest
    // version from the horizon to current (inclusive). A reader pinned at
    // `horizon` still needs every object that version references, so none of
    // them can be swept; an object only an older version referenced (a
    // compaction input merged away before the horizon, say) drops out of the
    // set and becomes reclaimable. This is what makes deletion safe by
    // construction rather than by a wall-clock guess. The set also unions the
    // WAL seqs of those versions: recovery replays exactly the `wal_segments`
    // list of the manifest it is handed, so that union is the complete read
    // surface of the `wal/` prefix.
    let mut live = ObjectSet::default();
    live.mark(&current.manifest);
    report.live_versions_loaded += 1;
    for_each_manifest_version(
        manifest_store,
        horizon..current_version,
        |version, manifest| match manifest {
            Some(manifest) => {
                live.mark(manifest);
                report.live_versions_loaded += 1;
                Ok(())
            }
            // A version retained only by a pin lease can predate the lease's
            // own visibility and be gone already (the holder detects that
            // itself). It references nothing we could keep alive, so skip
            // it; a missing version at or above the strict floor is a real
            // inconsistency and still aborts the sweep.
            None if version < strict_floor => {
                debug!(
                    version,
                    "pin-retained manifest version already reclaimed; skipping"
                );
                Ok(())
            }
            None => Err(Error::invariant(format!(
                "retained manifest version {version} is missing"
            ))),
        },
    )
    .await?;
    // Versions kept on their own below the dense horizon: point leases (a
    // backup of an older restore point) and the restore point window's floor
    // version, whose closure holds every object older than the window that
    // its versions still reference. Like any version below the strict floor,
    // one that is already gone is skipped (its holder re-checks its root).
    let extra_versions: std::collections::BTreeSet<u64> = point_versions
        .iter()
        .copied()
        .chain(window.map(|w| w.version))
        .filter(|v| *v < horizon)
        .collect();
    for_each_manifest_version(
        manifest_store,
        extra_versions.iter().copied(),
        |version, manifest| {
            match manifest {
                Some(manifest) => {
                    live.mark(manifest);
                    report.live_versions_loaded += 1;
                }
                None => debug!(
                    version,
                    "point/window-pinned manifest version already reclaimed; skipping"
                ),
            }
            Ok(())
        },
    )
    .await?;

    // The exact window: every version strictly between the window's floor and
    // the horizon joins the live set, so the window keeps exactly what its
    // versions reference. One that is missing leaves the window's closure
    // unproven; keep history by write time instead (the default mode).
    let mut exact = opts.exact_window;
    if let Some(w) = window.filter(|_| opts.exact_window) {
        let mut missing = 0usize;
        for_each_manifest_version(
            manifest_store,
            w.version.saturating_add(1)..horizon,
            |_, manifest| {
                match manifest {
                    Some(manifest) => {
                        live.mark(manifest);
                        report.live_versions_loaded += 1;
                    }
                    None => missing += 1,
                }
                Ok(())
            },
        )
        .await?;
        if missing > 0 {
            warn!(
                missing,
                window_floor = w.version,
                horizon,
                "restore point window has missing manifest versions; keeping window history by write time"
            );
            exact = false;
        }
    }
    report.exact_window = exact;
    // Objects written at or after this instant are kept for the window: a
    // version inside it may reference them (see `PinScope::Window`).
    let window_keep_since = window.filter(|_| !exact).map(|w| {
        chrono::Duration::from_std(min_age)
            .ok()
            .and_then(|margin| w.since.checked_sub_signed(margin))
            .unwrap_or(chrono::DateTime::<Utc>::MIN_UTC)
    });

    // The superseded fast path: retired versions (below every retained floor
    // and not point-pinned) whose successor committed at least
    // `superseded_min_age` ago. A successor that is already gone was itself
    // retired, so it committed even earlier.
    let superseded_cutoff = now - chrono::Duration::seconds(superseded_secs);
    let fast_versions: Vec<u64> = body_modified
        .keys()
        .copied()
        .filter(|version| *version < body_floor && !point_versions.contains(version))
        .filter(|version| match body_modified.get(&(version + 1)) {
            Some(successor) => *successor <= superseded_cutoff,
            None => version + 1 < body_floor,
        })
        .collect();
    let mut superseded = ObjectSet::default();
    let mut fast_retired: HashSet<u64> = HashSet::new();
    for_each_manifest_version(
        manifest_store,
        fast_versions.iter().copied(),
        |version, manifest| {
            if let Some(manifest) = manifest {
                superseded.mark_outside(manifest, &live);
                fast_retired.insert(version);
                report.retired_versions_loaded += 1;
            }
            Ok(())
        },
    )
    .await?;

    let ns_prefix = paths.namespace_prefix();
    let ns_prefix_str = ns_prefix.as_ref();

    let expected_max_level = max_level.max(live.max_level);
    // Pre-delete pin re-check (defense in depth): a lease that landed while
    // the live set was being built pins a version this sweep's horizon does
    // not honour — deleting now would race the new holder's copy. Abort the
    // pass instead; the next tick recomputes with the lease in view. The
    // holder's own post-acquire root verification covers the (now much
    // smaller) residual window between this check and the deletes below.
    if delete
        && pins_changed_below_horizon(store.as_ref(), paths, now_unix, &honoured, horizon).await?
    {
        warn!(
            horizon,
            "retention pin arrived mid-sweep; skipping deletions this pass"
        );
        report.aborted_by_pin = true;
        return Ok(report);
    }

    // How one unreferenced object is classified.
    enum Verdict {
        Retain,
        TooYoung,
        Reclaim { superseded: bool },
    }
    let classify = |last_modified: chrono::DateTime<Utc>, superseded: bool| {
        if window_keep_since.is_some_and(|since| last_modified >= since) {
            return Verdict::Retain;
        }
        let age_secs = (now - last_modified).num_seconds();
        if superseded && age_secs >= superseded_secs {
            return Verdict::Reclaim { superseded: true };
        }
        if age_secs < min_age_secs {
            return Verdict::TooYoung;
        }
        Verdict::Reclaim { superseded }
    };

    // One recursive LIST discovers numeric levels that no retained manifest
    // references. Listing `0..=expected_max_level` cannot see a level N+1
    // output stranded when compaction loses its install CAS.
    let sst_root = object_store::path::Path::from(format!("{ns_prefix_str}/sst"));
    let mut sst_objects = store.list(Some(&sst_root));
    while let Some(meta) = sst_objects.try_next().await.map_err(Error::ObjectStore)? {
        report.store_sst_bytes = report.store_sst_bytes.saturating_add(meta.size);
        let absolute = meta.location.as_ref();
        // Convert to namespace-relative form so the comparison matches what's
        // stored in `SstDescriptor::path`.
        let Some(relative) = absolute
            .strip_prefix(ns_prefix_str)
            .and_then(|s| s.strip_prefix('/'))
        else {
            debug!(path = %absolute, "list returned object outside namespace prefix; skipping");
            continue;
        };
        let Some((level, file_name)) = relative
            .strip_prefix("sst/level")
            .and_then(|rest| rest.split_once('/'))
            .and_then(|(raw_level, file_name)| {
                raw_level
                    .parse::<u32>()
                    .ok()
                    .map(|level| (level, file_name))
            })
        else {
            debug!(path = %absolute, "non-canonical object below sst prefix; leaving it alone");
            continue;
        };
        if file_name.is_empty() || file_name.contains('/') {
            debug!(path = %absolute, "nested/non-canonical SST object; leaving it alone");
            continue;
        }
        if level > expected_max_level {
            debug!(
                path = %absolute,
                level,
                expected_max_level,
                "discovered SST level deeper than retained manifests/configuration"
            );
        }
        if live.paths.contains(relative) {
            report.live_sst_bytes = report.live_sst_bytes.saturating_add(meta.size);
            continue;
        }
        match classify(meta.last_modified, superseded.paths.contains(relative)) {
            Verdict::Retain => {
                report.restore_window_retained += 1;
                report.restore_window_bytes = report.restore_window_bytes.saturating_add(meta.size);
                continue;
            }
            Verdict::TooYoung => {
                report.skipped_too_young += 1;
                debug!(path = %absolute, "orphan candidate too young, deferring");
                continue;
            }
            Verdict::Reclaim { superseded } => {
                if superseded {
                    report.superseded_reclaimed += 1;
                    report.superseded_bytes_freed =
                        report.superseded_bytes_freed.saturating_add(meta.size);
                }
            }
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

    // Reclaim dead WAL segments. Without this every `commit_batch` leaves one
    // immutable `wal/<seq>.wal` behind forever (flush only clears the manifest
    // references). Derived safety rule — a segment may be deleted iff BOTH hold:
    //
    // 1. No manifest version at or above the retention horizon lists its `seq`
    //    in `wal_segments`. Recovery (`recover_memtable_with_snapshot`) replays
    //    exactly the segments the manifest it is handed references — it never
    //    lists the `wal/` prefix — and `WriterSession::open` uses the `wal/`
    //    LIST only to seed the next seq above every visible segment. So an
    //    unreferenced segment is read by nobody: it is either an orphan from a
    //    commit that failed before the pointer CAS (its records were never
    //    acked) or a segment a flush already drained into SSTs that every
    //    retained version references (its records are durable elsewhere).
    // 2. The object is older than `min_age` — or, when a retired version
    //    referenced it, older than `superseded_min_age`. A commit PUTs the
    //    segment BEFORE the pointer CAS lands; in that window it is referenced
    //    by no version yet, but the very next manifest will reference it and
    //    the client is acked on the CAS — deleting it then would lose acked
    //    writes at the next crash. A segment a retired version references
    //    already had its CAS land, so only the short grace applies.
    //
    // Deleting a segment cannot cause a later seq collision either: writers
    // pick the next seq strictly above every segment still VISIBLE, and once
    // the object is gone `PutMode::Create` on its path succeeds again, while
    // a re-used seq can never interleave with retained history because every
    // retained seq is, by rule 1, still visible.
    let wal_dir = paths.wal_dir();
    let mut wal_objects = store.list(Some(&wal_dir));
    while let Some(meta) = wal_objects.try_next().await.map_err(Error::ObjectStore)? {
        report.store_wal_bytes = report.store_wal_bytes.saturating_add(meta.size);
        // Segments are `<16-hex-digits>.wal`; anything else under the prefix
        // is not ours to judge, so leave it alone.
        let Some(seq) = meta
            .location
            .filename()
            .and_then(|name| name.strip_suffix(".wal"))
            .and_then(|hex| u64::from_str_radix(hex, 16).ok())
        else {
            continue;
        };
        if live.wal.contains(&seq) {
            report.live_wal_bytes = report.live_wal_bytes.saturating_add(meta.size);
            continue;
        }
        match classify(meta.last_modified, superseded.wal.contains(&seq)) {
            Verdict::Retain => {
                report.restore_window_retained += 1;
                report.restore_window_bytes = report.restore_window_bytes.saturating_add(meta.size);
                continue;
            }
            Verdict::TooYoung => {
                report.skipped_too_young += 1;
                debug!(path = %meta.location, "dead WAL segment too young, deferring");
                continue;
            }
            Verdict::Reclaim { superseded } => {
                if superseded {
                    report.superseded_reclaimed += 1;
                    report.superseded_bytes_freed =
                        report.superseded_bytes_freed.saturating_add(meta.size);
                }
            }
        }
        report.wal_segments_reclaimed += 1;
        report.wal_bytes_freed = report.wal_bytes_freed.saturating_add(meta.size);
        if delete {
            store
                .delete(&meta.location)
                .await
                .map_err(Error::ObjectStore)?;
        }
    }

    // Reclaim superseded manifest snapshots, after the objects they named.
    // Every commit / flush / compaction writes an immutable
    // `manifest/v{N}.json` and nothing else removes the old ones. A snapshot
    // at version N is reachable only through `load_manifest_at(N)`, which the
    // engine calls for versions at or above the horizon (a reader pinned at
    // `horizon` loads exactly `v{horizon}.json`, and `current.json` points at
    // `current_version >= horizon`). Versions strictly below the horizon are
    // reachable by no live reader, so they become reclaimable — the same
    // retention-horizon argument that makes the SST sweep safe. A retired body
    // the fast path loaded goes now (its successor committed at least
    // `superseded_min_age` ago); any other waits out `min_age`, the guard for
    // the body-PUT-then-pointer-CAS race.
    for meta in manifest_objects {
        // Classify by filename. Manifest bodies are `v{16-hex}.json` directly
        // under `manifest/`; pointer files (RFC-029) are `p{16-hex}.json` under
        // `manifest/pointer/`, which the recursive LIST also returned. The
        // legacy `current.json` and anything else fail both parses and are
        // left untouched.
        let Some((is_pointer, version)) = parse_manifest_object(&meta.location) else {
            continue;
        };
        // Keep current and every version a pinned reader could still load. For
        // pointers this also keeps the family contiguous over [horizon,
        // current], which load_current's forward HEAD probe relies on.
        // Bodies also stay for the restore point window and point leases;
        // history is read by `load_manifest_at`, which needs no pointer.
        if version >= horizon {
            continue;
        }
        if is_pointer && advisory_floor.is_some_and(|floor| version >= floor) {
            continue;
        }
        if !is_pointer && (version >= body_floor || point_versions.contains(&version)) {
            continue;
        }
        let fast = !is_pointer && fast_retired.contains(&version);
        let age_secs = (now - meta.last_modified).num_seconds();
        if !fast && age_secs < min_age_secs {
            report.skipped_too_young += 1;
            debug!(path = %meta.location, age_secs, "retired manifest/pointer object too young, deferring");
            continue;
        }
        if is_pointer {
            report.pointer_files_reclaimed += 1;
            report.pointer_bytes_freed = report.pointer_bytes_freed.saturating_add(meta.size);
        } else {
            report.manifest_snapshots_reclaimed += 1;
            report.manifest_bytes_freed = report.manifest_bytes_freed.saturating_add(meta.size);
        }
        if delete {
            match store.delete(&meta.location).await {
                Ok(()) | Err(object_store::Error::NotFound { .. }) => {}
                Err(other) => return Err(Error::ObjectStore(other)),
            }
        }
    }

    // Reclaim a stale `memtable_snapshot.bin`. The snapshot is a cold-start
    // cache over the CURRENT manifest's unflushed WAL closure; version 2 names
    // that exact descriptor vector. A flush can garbage-collect the highest
    // LSN and make an SST high-water decrease, so an LSN comparison is not a
    // safe staleness proof. Exact WAL binding is: once it differs from the
    // current manifest, recovery cannot consume it and the object is pure
    // storage cost. On any doubt — legacy/corrupt/future wire — leave it
    // alone. `min_age` guards the window where the writer overwrites the
    // object with a fresh snapshot right after our GET.
    let snap_path = paths.memtable_snapshot();
    match store.get(&snap_path).await {
        Ok(res) => {
            let meta = res.meta.clone();
            let body = res.bytes().await.map_err(Error::ObjectStore)?;
            let stale = crate::recovery::decode_memtable_snapshot(&body)
                .map(|snapshot| !snapshot.covers_manifest(&current.manifest))
                .unwrap_or(false);
            if stale {
                let age_secs = (now - meta.last_modified).num_seconds();
                if age_secs < min_age_secs {
                    report.skipped_too_young += 1;
                    debug!(path = %snap_path, age_secs, "stale memtable snapshot too young, deferring");
                } else {
                    report.memtable_snapshots_reclaimed += 1;
                    report.memtable_snapshot_bytes_freed = report
                        .memtable_snapshot_bytes_freed
                        .saturating_add(meta.size);
                    if delete {
                        store.delete(&snap_path).await.map_err(Error::ObjectStore)?;
                    }
                }
            }
        }
        Err(object_store::Error::NotFound { .. }) => {}
        Err(other) => return Err(Error::ObjectStore(other)),
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
    use crate::compact::compact_l0_to_l1;
    use crate::fence::WriterFence;
    use crate::flush::{flush, NodeWriteRecord};
    use crate::ingest::{CommitOutcome, WriterSession};
    use crate::manifest::ManifestStore;
    use crate::memtable::{MemKey, MemOp, Memtable};
    use crate::paths::NamespacePaths;
    use crate::pin::{PinLease, PinScope, RetentionPin};
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
            properties: vec![PropertyDef::new("name", DataType::Utf8, false)
                .unwrap()
                .with_indexed(true)],
        }
    }

    fn node_record(name: &str) -> NodeWriteRecord {
        let mut props = std::collections::BTreeMap::new();
        props.insert("name".into(), Value::Str(name.into()));
        NodeWriteRecord {
            properties: props,
            schema_version: 1,
            ..Default::default()
        }
    }

    fn node_payload(name: &str) -> Bytes {
        node_record(name).encode().unwrap()
    }

    fn person_schema() -> Schema {
        SchemaBuilder::new().label(person_label()).unwrap().build()
    }

    /// Commit `names` through a fresh `WriterSession`, one commit (= one WAL
    /// segment) per name, and return the session plus the per-commit
    /// `(manifest_version, wal_seq)` pairs.
    async fn session_with_commits(
        store: &Arc<dyn ObjectStore>,
        paths: &NamespacePaths,
        names: &[&str],
    ) -> (WriterSession, Vec<(u64, u64)>) {
        let mut session = WriterSession::open(store.clone(), paths.clone())
            .await
            .unwrap();
        let mut commits = Vec::with_capacity(names.len());
        for (i, name) in names.iter().enumerate() {
            session
                .upsert_node("Person", sorted_node_id(i as u8 + 1), &node_record(name))
                .unwrap();
            match session.commit_batch().await.unwrap() {
                CommitOutcome::Committed {
                    manifest_version,
                    wal_seq,
                    ..
                } => commits.push((manifest_version, wal_seq)),
                other => panic!("expected Committed, got {other:?}"),
            }
        }
        (session, commits)
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

    /// Plan item 31: the janitor against ACTIVE search state. The barrier,
    /// base and delta objects of an Active generation are live — a sweep at
    /// any horizon must not touch them. Once a consolidation retires the old
    /// generation, a sweep whose horizon still pins the old manifest must
    /// preserve the retired bodies (a pinned reader may still be serving
    /// them), and only a sweep past that horizon reclaims them.
    #[cfg(feature = "text-index")]
    // The std-mutex env guard is deliberately held across awaits: it
    // serializes the whole test body against process-global policy env.
    #[allow(clippy::await_holding_lock)]
    #[tokio::test]
    async fn sweep_preserves_active_search_generations_until_the_horizon_passes() {
        use crate::manifest::TextIndexDescriptor;
        use crate::read::Snapshot;

        let _env_lock = crate::test_support::SEARCH_COMPACTION_ENV
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let _env_restore = crate::test_support::SearchCompactionEnvRestore::configure();

        let store = make_store();
        let paths = make_paths("janitor-active-search");
        let mut writer = WriterSession::open(store.clone(), paths.clone())
            .await
            .unwrap();
        writer
            .register_text_index(
                TextIndexDescriptor::new("doc_ft".into(), "Doc".into(), vec!["body".into()]),
                false,
            )
            .await
            .unwrap();
        let schema = SchemaBuilder::new()
            .label(LabelDef {
                name: "Doc".into(),
                properties: vec![PropertyDef::new("body", DataType::Utf8, true).unwrap()],
            })
            .unwrap()
            .build();
        for body in ["alpha one", "alpha two", "beta three"] {
            let mut props = std::collections::BTreeMap::new();
            props.insert("body".to_string(), Value::Str(body.to_string()));
            writer
                .upsert_node(
                    "Doc",
                    NodeId::new(),
                    &NodeWriteRecord {
                        properties: props,
                        schema_version: 1,
                        ..Default::default()
                    },
                )
                .unwrap();
        }
        writer.flush(schema.clone()).await.unwrap();
        writer.compact_l0(&schema).await.unwrap();

        let ms = ManifestStore::new(store.clone(), paths.clone());
        let settled = ms.load_current().await.unwrap();
        let state = settled
            .manifest
            .search_lsm
            .iter()
            .find(|state| state.index_name == "doc_ft")
            .expect("active text generation");
        let mut generation_paths: Vec<String> = state
            .segments
            .iter()
            .filter_map(|segment| {
                settled
                    .manifest
                    .ssts
                    .iter()
                    .find(|descriptor| descriptor.id == segment.sst_id)
                    .map(|descriptor| descriptor.path.clone())
            })
            .collect();
        if let Some(barrier_id) = state.compat_barrier_sst_id {
            generation_paths.extend(
                settled
                    .manifest
                    .ssts
                    .iter()
                    .filter(|descriptor| descriptor.id == barrier_id)
                    .map(|descriptor| descriptor.path.clone()),
            );
        }
        assert!(
            generation_paths.len() >= 2,
            "generation must have at least a body and a barrier"
        );
        let old_version = settled.manifest.version;

        // 1. Active generation, max horizon: the consolidation's leftovers
        // (pre-consolidation deltas) are legitimate garbage, but nothing of
        // the ACTIVE generation may be touched.
        let _ = sweep_orphans(&ms, u64::MAX, Duration::from_secs(0), 10, true)
            .await
            .unwrap();
        for path in &generation_paths {
            let absolute = format!("{}/{}", paths.namespace_prefix().as_ref(), path);
            store
                .head(&object_store::path::Path::from(absolute))
                .await
                .expect("active generation object must survive the sweep");
        }

        // 2. Retire the generation: more docs, flush, force-base consolidate.
        for body in ["alpha four", "beta five"] {
            let mut props = std::collections::BTreeMap::new();
            props.insert("body".to_string(), Value::Str(body.to_string()));
            writer
                .upsert_node(
                    "Doc",
                    NodeId::new(),
                    &NodeWriteRecord {
                        properties: props,
                        schema_version: 1,
                        ..Default::default()
                    },
                )
                .unwrap();
        }
        writer.flush(schema.clone()).await.unwrap();
        writer.compact_l0(&schema).await.unwrap();
        let current = ms.load_current().await.unwrap();
        let retired: Vec<String> = generation_paths
            .iter()
            .filter(|path| {
                !current
                    .manifest
                    .ssts
                    .iter()
                    .any(|descriptor| &descriptor.path == *path)
            })
            .cloned()
            .collect();
        assert!(
            !retired.is_empty(),
            "the consolidation must have retired at least one old object"
        );

        // 3. A sweep whose horizon still pins the OLD version keeps the
        // retired bodies, and a reader pinned there still serves.
        sweep_orphans(&ms, old_version, Duration::from_secs(0), 10, true)
            .await
            .unwrap();
        for path in &retired {
            let absolute = format!("{}/{}", paths.namespace_prefix().as_ref(), path);
            store
                .head(&object_store::path::Path::from(absolute))
                .await
                .expect("a pinned horizon must preserve retired search objects");
        }
        let pinned_manifest = ms.load_manifest_at(old_version).await.unwrap();
        let pinned_loaded = crate::manifest::LoadedManifest::new(
            current.pointer.clone(),
            None,
            None,
            pinned_manifest,
        );
        let memtable = Memtable::new();
        let view = memtable.snapshot_view();
        let pinned = Snapshot::new(pinned_loaded, &view, store.clone(), paths.clone());
        let hits = pinned
            .text_search("doc_ft", "Doc", &crate::text::parse_query("alpha"), None)
            .await
            .unwrap()
            .expect("the pinned reader must still serve the retired generation");
        assert_eq!(hits.len(), 2, "the old snapshot sees the old corpus");

        // 4. Advancing the horizon reclaims the retired objects; the new
        // generation keeps serving.
        let report = sweep_orphans(&ms, u64::MAX, Duration::from_secs(0), 10, true)
            .await
            .unwrap();
        assert!(
            report.orphans_deleted > 0,
            "past the horizon the retired generation is garbage"
        );
        for path in &retired {
            let absolute = format!("{}/{}", paths.namespace_prefix().as_ref(), path);
            assert!(
                store
                    .head(&object_store::path::Path::from(absolute))
                    .await
                    .is_err(),
                "retired object {path} must be reclaimed"
            );
        }
        let fresh = ms.load_current().await.unwrap();
        let view = memtable.snapshot_view();
        let snapshot = Snapshot::new(fresh, &view, store.clone(), paths.clone());
        let hits = snapshot
            .text_search("doc_ft", "Doc", &crate::text::parse_query("alpha"), None)
            .await
            .unwrap()
            .expect("the new generation must serve after the sweep");
        assert_eq!(hits.len(), 3);
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

        let current = ms.load_current().await.unwrap();
        let nodes = current
            .manifest
            .ssts
            .iter()
            .find(|sst| sst.kind == crate::manifest::SstKind::Nodes)
            .unwrap();
        assert!(
            nodes
                .node_locator
                .as_ref()
                .and_then(|locator| locator.property_pages.as_ref())
                .is_some(),
            "fixture must publish locator-bound node property pages"
        );
        assert!(
            nodes.equality_property_indices.iter().any(|index| {
                index.format == crate::manifest::PropertyIndexFormat::PagedV1
                    || index.paged.is_some()
            }),
            "fixture must cover both new live sidecar classes"
        );

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
    async fn sweep_discovers_deep_orphan_after_failed_compaction_cas() {
        let store = make_store();
        let paths = make_paths("janitor-levels");
        let ms = ManifestStore::new(store.clone(), paths.clone());
        let _base = ms.bootstrap(Uuid::now_v7()).await.unwrap();

        // Model a flush orphan at L0 plus a prepared compaction that wrote its
        // L7 output before losing the manifest CAS. No retained manifest names
        // L7, so manifest high-water discovery alone cannot find it.
        let l0 = paths.sst_object(0, "l0-orphan.parquet");
        let l7 = paths.sst_object(7, "l7-failed-cas-orphan.parquet");
        store
            .put(&l0, PutPayload::from(Bytes::from_static(b"l0")))
            .await
            .unwrap();
        store
            .put(&l7, PutPayload::from(Bytes::from_static(b"l7")))
            .await
            .unwrap();

        // The legacy hint is deliberately shallower than the orphan.
        let report = sweep_orphans(&ms, u64::MAX, Duration::from_secs(0), 1, true)
            .await
            .unwrap();
        assert_eq!(report.orphans_found, 2);
        assert_eq!(report.orphans_deleted, 2);
        assert!(store.head(&l0).await.is_err(), "l0 orphan must be deleted");
        assert!(
            store.head(&l7).await.is_err(),
            "the orphan beyond both configured and manifest levels must be deleted"
        );
    }

    /// With no live reader pinned (horizon clamps to current), every manifest
    /// snapshot below the current version is reclaimed; the current snapshot
    /// and the pointer survive and the namespace still loads.
    #[tokio::test]
    async fn sweep_reclaims_manifest_snapshots_below_horizon() {
        let store = make_store();
        let paths = make_paths("janitor-manifest-reclaim");
        let ms = ManifestStore::new(store.clone(), paths.clone());
        let base = ms.bootstrap(Uuid::now_v7()).await.unwrap();
        let fence = WriterFence::new(base.manifest.epoch);
        let schema = SchemaBuilder::new().label(person_label()).unwrap().build();

        // Four manifest versions: bootstrap v0, then three flushes (v1..v3).
        let v0 = base.manifest.version;
        let m1 = flush_one_node(&ms, &fence, &base, &schema, sorted_node_id(1), "A", 1).await;
        let m2 = flush_one_node(&ms, &fence, &m1, &schema, sorted_node_id(2), "B", 2).await;
        let m3 = flush_one_node(&ms, &fence, &m2, &schema, sorted_node_id(3), "C", 3).await;
        let (v1, v2, current) = (
            m1.manifest.version,
            m2.manifest.version,
            m3.manifest.version,
        );
        assert!(v0 < v1 && v1 < v2 && v2 < current);

        // Every old snapshot body exists before the sweep.
        for v in [v0, v1, v2] {
            assert!(store.head(&paths.manifest_version(v)).await.is_ok());
        }

        // horizon = u64::MAX clamps to the current version: only it is needed.
        let report = sweep_orphans(&ms, u64::MAX, Duration::from_secs(0), 4, true)
            .await
            .unwrap();
        assert_eq!(report.manifest_snapshots_reclaimed, 3);
        assert!(report.manifest_bytes_freed > 0);
        // RFC-029: the retired pointer files p0..p2 are reclaimed alongside the
        // manifest snapshots, counted on their own report fields.
        assert_eq!(report.pointer_files_reclaimed, 3);
        assert!(report.pointer_bytes_freed > 0);
        // The accumulating flushes leave every SST referenced by current, so no
        // SST orphans — only the retired manifest snapshots are reclaimed.
        assert_eq!(report.orphans_found, 0);

        for v in [v0, v1, v2] {
            assert!(
                store.head(&paths.manifest_version(v)).await.is_err(),
                "retired snapshot v{v} must be reclaimed"
            );
            assert!(
                store.head(&paths.pointer_version(v)).await.is_err(),
                "retired pointer p{v} must be reclaimed"
            );
        }
        assert!(
            store.head(&paths.manifest_version(current)).await.is_ok(),
            "the current snapshot must survive"
        );
        assert!(store.head(&paths.pointer_version(current)).await.is_ok());
        assert_eq!(ms.load_current().await.unwrap().manifest.version, current);

        // Idempotent: a second sweep reclaims nothing.
        let again = sweep_orphans(&ms, u64::MAX, Duration::from_secs(0), 4, true)
            .await
            .unwrap();
        assert_eq!(again.manifest_snapshots_reclaimed, 0);
    }

    /// A reader pinned at the retention horizon keeps its snapshot and every
    /// later one; only strictly-older snapshots are reclaimed.
    #[tokio::test]
    async fn sweep_keeps_manifest_snapshots_at_or_above_horizon() {
        let store = make_store();
        let paths = make_paths("janitor-manifest-horizon");
        let ms = ManifestStore::new(store.clone(), paths.clone());
        let base = ms.bootstrap(Uuid::now_v7()).await.unwrap();
        let fence = WriterFence::new(base.manifest.epoch);
        let schema = SchemaBuilder::new().label(person_label()).unwrap().build();

        let v0 = base.manifest.version;
        let m1 = flush_one_node(&ms, &fence, &base, &schema, sorted_node_id(1), "A", 1).await;
        let m2 = flush_one_node(&ms, &fence, &m1, &schema, sorted_node_id(2), "B", 2).await;
        let m3 = flush_one_node(&ms, &fence, &m2, &schema, sorted_node_id(3), "C", 3).await;
        let (v1, v2, current) = (
            m1.manifest.version,
            m2.manifest.version,
            m3.manifest.version,
        );

        // A reader is pinned at v2: the sweep must keep v2 and everything newer,
        // reclaiming only v0 and v1.
        let report = sweep_orphans(&ms, v2, Duration::from_secs(0), 4, true)
            .await
            .unwrap();
        assert_eq!(report.manifest_snapshots_reclaimed, 2);

        assert!(store.head(&paths.manifest_version(v0)).await.is_err());
        assert!(store.head(&paths.manifest_version(v1)).await.is_err());
        assert!(
            store.head(&paths.manifest_version(v2)).await.is_ok(),
            "the pinned reader's snapshot must survive"
        );
        assert!(
            store.head(&paths.manifest_version(current)).await.is_ok(),
            "the current snapshot must survive"
        );
        assert!(store.head(&paths.pointer_version(current)).await.is_ok());
    }

    /// Acked-but-unflushed durability: a WAL segment the current manifest
    /// still references must survive the sweep, and a cold reopen must
    /// replay the committed rows out of it.
    #[tokio::test]
    async fn wal_sweep_preserves_acked_unflushed_writes() {
        let store = make_store();
        let paths = make_paths("janitor-wal-unflushed");
        let (session, commits) = session_with_commits(&store, &paths, &["Ada"]).await;
        let (_, seq) = commits[0];
        drop(session);

        let ms = ManifestStore::new(store.clone(), paths.clone());
        let report = sweep_orphans(&ms, u64::MAX, Duration::from_secs(0), 4, true)
            .await
            .unwrap();
        assert_eq!(
            report.wal_segments_reclaimed, 0,
            "a manifest-referenced segment is not reclaimable"
        );
        assert!(
            store.head(&paths.wal_segment(seq)).await.is_ok(),
            "the referenced segment body must survive the sweep"
        );

        // Cold reopen: recovery replays the segment; the acked row is there.
        let session2 = WriterSession::open(store, paths).await.unwrap();
        let snap = session2.snapshot();
        let view = snap
            .lookup_node("Person", sorted_node_id(1))
            .await
            .unwrap()
            .expect("acked-but-unflushed row must survive the sweep");
        assert_eq!(view.properties.get("name"), Some(&Value::Str("Ada".into())));
    }

    /// Once a flush drains the WAL into SSTs (and the horizon passes the
    /// flush), the segments are dead: the sweep reclaims them after
    /// `min_age`, and the rows still read back from the SSTs on reopen.
    #[tokio::test]
    async fn wal_sweep_reclaims_flushed_segments_after_min_age() {
        let store = make_store();
        let paths = make_paths("janitor-wal-flushed");
        let (mut session, commits) = session_with_commits(&store, &paths, &["Ada", "Bob"]).await;
        session.flush(person_schema()).await.unwrap();

        let ms = ManifestStore::new(store.clone(), paths.clone());

        // min_age = 24h: the segments are unreferenced but too young.
        let young = sweep_orphans(&ms, u64::MAX, Duration::from_secs(86_400), 4, true)
            .await
            .unwrap();
        assert_eq!(young.wal_segments_reclaimed, 0);
        for (_, seq) in &commits {
            assert!(
                store.head(&paths.wal_segment(*seq)).await.is_ok(),
                "young dead segment {seq} must be deferred"
            );
        }

        // Dry run past min_age: counted, not deleted.
        let dry = sweep_orphans(&ms, u64::MAX, Duration::from_secs(0), 4, false)
            .await
            .unwrap();
        assert_eq!(dry.wal_segments_reclaimed, 2);
        assert!(dry.wal_bytes_freed > 0);
        assert!(store.head(&paths.wal_segment(commits[0].1)).await.is_ok());

        // Real run: both segments are reclaimed.
        let real = sweep_orphans(&ms, u64::MAX, Duration::from_secs(0), 4, true)
            .await
            .unwrap();
        assert_eq!(real.wal_segments_reclaimed, 2);
        for (_, seq) in &commits {
            assert!(
                store.head(&paths.wal_segment(*seq)).await.is_err(),
                "dead segment {seq} must be reclaimed"
            );
        }

        // Idempotent, and the flushed rows still read back after a cold
        // reopen (they live in SSTs now).
        let again = sweep_orphans(&ms, u64::MAX, Duration::from_secs(0), 4, true)
            .await
            .unwrap();
        assert_eq!(again.wal_segments_reclaimed, 0);
        drop(session);
        let session2 = WriterSession::open(store, paths).await.unwrap();
        let snap = session2.snapshot();
        for (i, name) in ["Ada", "Bob"].iter().enumerate() {
            let view = snap
                .lookup_node("Person", sorted_node_id(i as u8 + 1))
                .await
                .unwrap()
                .expect("flushed row must survive the WAL sweep");
            assert_eq!(
                view.properties.get("name"),
                Some(&Value::Str((*name).into()))
            );
        }
    }

    /// A segment referenced by a retained older manifest version (retention
    /// horizon below current) is kept even though the current version no
    /// longer references it.
    #[tokio::test]
    async fn wal_sweep_keeps_segments_referenced_by_retained_versions() {
        let store = make_store();
        let paths = make_paths("janitor-wal-horizon");
        let (mut session, commits) = session_with_commits(&store, &paths, &["Ada"]).await;
        let (v_commit, seq) = commits[0];
        // The flush clears the WAL refs from the NEW version; v_commit still
        // references the segment.
        session.flush(person_schema()).await.unwrap();

        let ms = ManifestStore::new(store.clone(), paths.clone());
        let pinned = sweep_orphans(&ms, v_commit, Duration::from_secs(0), 4, true)
            .await
            .unwrap();
        assert_eq!(
            pinned.wal_segments_reclaimed, 0,
            "a segment referenced by a retained version must be kept"
        );
        assert!(store.head(&paths.wal_segment(seq)).await.is_ok());

        // Horizon advances past the flush: the segment becomes dead.
        let free = sweep_orphans(&ms, u64::MAX, Duration::from_secs(0), 4, true)
            .await
            .unwrap();
        assert_eq!(free.wal_segments_reclaimed, 1);
        assert!(store.head(&paths.wal_segment(seq)).await.is_err());
    }

    /// A `memtable_snapshot.bin` is kept while its exact WAL binding matches
    /// current unflushed state and reclaimed once a flush supersedes that
    /// closure.
    #[tokio::test]
    async fn sweep_reclaims_stale_memtable_snapshot_once_superseded() {
        let store = make_store();
        let paths = make_paths("janitor-snap-stale");
        let (mut session, _) = session_with_commits(&store, &paths, &["Ada"]).await;
        session.write_memtable_snapshot_now().await.unwrap();
        let snap_path = paths.memtable_snapshot();
        assert!(store.head(&snap_path).await.is_ok());

        let ms = ManifestStore::new(store.clone(), paths.clone());

        // Fresh snapshot (covers unflushed rows): must be kept.
        let fresh = sweep_orphans(&ms, u64::MAX, Duration::from_secs(0), 4, true)
            .await
            .unwrap();
        assert_eq!(fresh.memtable_snapshots_reclaimed, 0);
        assert!(
            store.head(&snap_path).await.is_ok(),
            "a snapshot covering unflushed state must survive"
        );

        // Flush drains the covered rows into SSTs: the snapshot is stale now.
        session.flush(person_schema()).await.unwrap();

        // min_age guard still applies.
        let young = sweep_orphans(&ms, u64::MAX, Duration::from_secs(86_400), 4, true)
            .await
            .unwrap();
        assert_eq!(young.memtable_snapshots_reclaimed, 0);
        assert!(store.head(&snap_path).await.is_ok());

        // Dry run: counted, not deleted.
        let dry = sweep_orphans(&ms, u64::MAX, Duration::from_secs(0), 4, false)
            .await
            .unwrap();
        assert_eq!(dry.memtable_snapshots_reclaimed, 1);
        assert!(dry.memtable_snapshot_bytes_freed > 0);
        assert!(store.head(&snap_path).await.is_ok());

        // Real run: the stale snapshot is reclaimed.
        let real = sweep_orphans(&ms, u64::MAX, Duration::from_secs(0), 4, true)
            .await
            .unwrap();
        assert_eq!(real.memtable_snapshots_reclaimed, 1);
        assert!(
            store.head(&snap_path).await.is_err(),
            "the superseded snapshot must be reclaimed"
        );

        // Idempotent.
        let again = sweep_orphans(&ms, u64::MAX, Duration::from_secs(0), 4, true)
            .await
            .unwrap();
        assert_eq!(again.memtable_snapshots_reclaimed, 0);
    }

    /// A retention pin lease (`manifest/pins/`, written by a running backup)
    /// holds the sweep's horizon at the pinned version: bodies only that
    /// version references survive a concurrent compaction + sweep, and the
    /// same sweep reclaims them once the lease is released.
    #[tokio::test]
    async fn sweep_honours_an_unexpired_pin_lease_until_released() {
        let store = make_store();
        let paths = make_paths("janitor-pin-honoured");
        let ms = ManifestStore::new(store.clone(), paths.clone());
        let base = ms.bootstrap(Uuid::now_v7()).await.unwrap();
        let fence = WriterFence::new(base.manifest.epoch);
        let schema = person_schema();

        // Two L0 flushes, then a compaction: the L0 bodies stay referenced
        // only by versions strictly below current — exactly the closure a
        // backup pinned at the pre-compaction version is still copying.
        let m1 = flush_one_node(&ms, &fence, &base, &schema, sorted_node_id(1), "A", 1).await;
        let m2 = flush_one_node(&ms, &fence, &m1, &schema, sorted_node_id(2), "B", 2).await;
        let pinned_version = m2.manifest.version;
        let pinned_bodies: Vec<String> = m2.manifest.ssts.iter().map(|s| s.path.clone()).collect();
        assert_eq!(pinned_bodies.len(), 2);
        let out = compact_l0_to_l1(&ms, &fence, &m2, &schema).await.unwrap();
        assert_eq!(out.source_ssts_removed, 2);

        let body_path = |rel: &str| {
            object_store::path::Path::from(format!("{}/{}", paths.namespace_prefix().as_ref(), rel))
        };

        // Pin at the pre-compaction version, as copy_namespace_snapshot does.
        let pin = RetentionPin::acquire(
            store.clone(),
            &paths,
            pinned_version,
            Duration::from_secs(600),
        )
        .await
        .unwrap();

        // No in-process reader is pinned (horizon clamps to current): only
        // the lease protects the pinned closure.
        let held = sweep_orphans(&ms, u64::MAX, Duration::from_secs(0), 4, true)
            .await
            .unwrap();
        assert_eq!(held.pins_honoured, 1);
        assert_eq!(held.expired_pins_reclaimed, 0);
        assert_eq!(held.orphans_found, 0, "the pinned closure is not orphaned");
        for rel in &pinned_bodies {
            assert!(
                store.head(&body_path(rel)).await.is_ok(),
                "pinned L0 body must survive the sweep: {rel}"
            );
        }
        assert!(
            store
                .head(&paths.manifest_version(pinned_version))
                .await
                .is_ok(),
            "the pinned manifest body must survive"
        );
        assert!(
            store.head(pin.path()).await.is_ok(),
            "an unexpired lease must not be deleted by the sweep"
        );

        // Release (the backup finished): the same sweep reclaims the closure.
        pin.release().await.unwrap();
        let released = sweep_orphans(&ms, u64::MAX, Duration::from_secs(0), 4, true)
            .await
            .unwrap();
        assert_eq!(released.pins_honoured, 0);
        assert!(
            released.orphans_found >= 2,
            "the compacted-away L0 bodies become reclaimable once unpinned"
        );
        for rel in &pinned_bodies {
            assert!(
                store.head(&body_path(rel)).await.is_err(),
                "unpinned L0 body must be reclaimed: {rel}"
            );
        }
        assert!(
            store
                .head(&paths.manifest_version(pinned_version))
                .await
                .is_err(),
            "the pinned manifest snapshot is reclaimed after release"
        );
    }

    /// An expired lease pins nothing: the sweep ignores it for the horizon
    /// and reclaims the lease object itself, so a crashed backup cannot pin
    /// the namespace forever. An undecodable body under `pins/` names no
    /// version, pins nothing, and is left alone.
    #[tokio::test]
    async fn sweep_ignores_and_reclaims_expired_pin_leases() {
        let store = make_store();
        let paths = make_paths("janitor-pin-expired");
        let (mut session, commits) = session_with_commits(&store, &paths, &["Ada"]).await;
        let (v_commit, seq) = commits[0];
        session.flush(person_schema()).await.unwrap();
        drop(session);

        let ms = ManifestStore::new(store.clone(), paths.clone());

        // A live lease at the commit version keeps its WAL segment alive even
        // though the flush retired it from every later manifest.
        let lease_path = paths.pin_object("backup-1");
        let live = PinLease {
            version: v_commit,
            expires_at_unix: chrono::Utc::now().timestamp() + 3600,
            scope: PinScope::Floor,
        };
        store
            .put(
                &lease_path,
                PutPayload::from(serde_json::to_vec(&live).unwrap()),
            )
            .await
            .unwrap();
        let held = sweep_orphans(&ms, u64::MAX, Duration::from_secs(0), 4, true)
            .await
            .unwrap();
        assert_eq!(held.pins_honoured, 1);
        assert_eq!(held.wal_segments_reclaimed, 0);
        assert!(store.head(&paths.wal_segment(seq)).await.is_ok());
        assert!(store.head(&lease_path).await.is_ok());

        // The holder crashed and the lease expired. Plant garbage alongside.
        let expired = PinLease {
            version: v_commit,
            expires_at_unix: chrono::Utc::now().timestamp() - 3600,
            scope: PinScope::Floor,
        };
        store
            .put(
                &lease_path,
                PutPayload::from(serde_json::to_vec(&expired).unwrap()),
            )
            .await
            .unwrap();
        let garbage_path = paths.pin_object("not-a-lease");
        store
            .put(
                &garbage_path,
                PutPayload::from(Bytes::from_static(b"not json")),
            )
            .await
            .unwrap();

        // Dry run: the expired lease no longer holds the horizon (the dead
        // segment is a candidate) and is itself a reclaim candidate — but
        // nothing is deleted.
        let dry = sweep_orphans(&ms, u64::MAX, Duration::from_secs(0), 4, false)
            .await
            .unwrap();
        assert_eq!(dry.pins_honoured, 0);
        assert_eq!(dry.expired_pins_reclaimed, 1);
        assert_eq!(dry.wal_segments_reclaimed, 1);
        assert!(
            store.head(&lease_path).await.is_ok(),
            "dry run must not delete the expired lease"
        );
        assert!(store.head(&paths.wal_segment(seq)).await.is_ok());

        // Real run: the expired lease and the dead segment are reclaimed; the
        // undecodable object is not ours to judge and survives.
        let real = sweep_orphans(&ms, u64::MAX, Duration::from_secs(0), 4, true)
            .await
            .unwrap();
        assert_eq!(real.expired_pins_reclaimed, 1);
        assert_eq!(real.wal_segments_reclaimed, 1);
        assert!(
            store.head(&lease_path).await.is_err(),
            "the expired lease must be reclaimed"
        );
        assert!(store.head(&paths.wal_segment(seq)).await.is_err());
        assert!(store.head(&garbage_path).await.is_ok());

        // Idempotent.
        let again = sweep_orphans(&ms, u64::MAX, Duration::from_secs(0), 4, true)
            .await
            .unwrap();
        assert_eq!(again.expired_pins_reclaimed, 0);
    }

    /// Reports every listed object as `age` older than it really is, so a
    /// test can drive the sweep's age guards without waiting for them.
    #[derive(Debug)]
    struct AgedStore {
        inner: Arc<dyn ObjectStore>,
        age: chrono::Duration,
    }

    impl std::fmt::Display for AgedStore {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "AgedStore({})", self.inner)
        }
    }

    #[async_trait::async_trait]
    impl ObjectStore for AgedStore {
        async fn put_opts(
            &self,
            location: &object_store::path::Path,
            payload: PutPayload,
            opts: object_store::PutOptions,
        ) -> object_store::Result<object_store::PutResult> {
            self.inner.put_opts(location, payload, opts).await
        }

        async fn put_multipart_opts(
            &self,
            location: &object_store::path::Path,
            opts: object_store::PutMultipartOptions,
        ) -> object_store::Result<Box<dyn object_store::MultipartUpload>> {
            self.inner.put_multipart_opts(location, opts).await
        }

        async fn get_opts(
            &self,
            location: &object_store::path::Path,
            options: object_store::GetOptions,
        ) -> object_store::Result<object_store::GetResult> {
            self.inner.get_opts(location, options).await
        }

        fn list(
            &self,
            prefix: Option<&object_store::path::Path>,
        ) -> futures::stream::BoxStream<'static, object_store::Result<object_store::ObjectMeta>>
        {
            use futures::StreamExt;
            let age = self.age;
            self.inner
                .list(prefix)
                .map(move |meta| {
                    meta.map(|mut meta| {
                        meta.last_modified -= age;
                        meta
                    })
                })
                .boxed()
        }

        async fn list_with_delimiter(
            &self,
            prefix: Option<&object_store::path::Path>,
        ) -> object_store::Result<object_store::ListResult> {
            self.inner.list_with_delimiter(prefix).await
        }

        async fn copy_opts(
            &self,
            from: &object_store::path::Path,
            to: &object_store::path::Path,
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

    /// The superseded fast path. Every object below is two hours old and the
    /// sweep runs with the 24 h `min_age`: WAL segments a flush retired, L0
    /// bodies a compaction merged away and the manifest versions that
    /// referenced them were all installed once, so they are reclaimed under
    /// the 1 h superseded grace, while an object no manifest version ever
    /// referenced (it could be an upload whose CAS has not landed) still waits
    /// out `min_age`. A version pinned by an in-process reader (the horizon)
    /// or by a retention lease keeps its whole closure.
    #[tokio::test]
    async fn sweep_reclaims_superseded_objects_after_the_short_grace_while_orphans_wait() {
        let store: Arc<dyn ObjectStore> = Arc::new(AgedStore {
            inner: make_store(),
            age: chrono::Duration::hours(2),
        });
        let paths = make_paths("janitor-superseded");
        let ms = ManifestStore::new(store.clone(), paths.clone());
        let abs = |rel: &str| {
            object_store::path::Path::from(format!("{}/{}", paths.namespace_prefix().as_ref(), rel))
        };

        // Two commits (two WAL segments), a flush, one more commit + flush:
        // two L0 bodies; then a compaction merges them into one L1 body.
        let (mut session, commits) = session_with_commits(&store, &paths, &["A", "B"]).await;
        session.flush(person_schema()).await.unwrap();
        session
            .upsert_node("Person", sorted_node_id(3), &node_record("C"))
            .unwrap();
        session.commit_batch().await.unwrap();
        session.flush(person_schema()).await.unwrap();
        let pre_compaction = ms.load_current().await.unwrap().manifest;
        let l0_objects: Vec<String> = pre_compaction
            .ssts
            .iter()
            .flat_map(descriptor_object_paths)
            .collect();
        assert!(l0_objects.len() >= 2);
        session.compact_l0(&person_schema()).await.unwrap();
        let current = ms.load_current().await.unwrap().manifest;
        let live_now = manifest_object_paths(&current);
        assert!(
            l0_objects.iter().all(|p| !live_now.contains(p)),
            "the compaction must supersede every L0 object"
        );
        let wal_seqs: Vec<u64> = commits.iter().map(|(_, seq)| *seq).collect();

        // An object no manifest version ever referenced.
        let orphan = paths.sst_object(0, "planted-orphan.parquet");
        store
            .put(&orphan, PutPayload::from(Bytes::from_static(b"orphan")))
            .await
            .unwrap();

        // 1. A reader pinned at the pre-compaction version (the horizon): the
        // flushed WAL segments are superseded and go, the L0 closure stays.
        let pinned = sweep_orphans(
            &ms,
            pre_compaction.version,
            Duration::from_secs(86_400),
            4,
            true,
        )
        .await
        .unwrap();
        for seq in &wal_seqs {
            assert!(
                store.head(&paths.wal_segment(*seq)).await.is_err(),
                "flushed WAL segment {seq} must be reclaimed under the short grace: {pinned:?}"
            );
        }
        for rel in &l0_objects {
            assert!(
                store.head(&abs(rel)).await.is_ok(),
                "the horizon-pinned version keeps {rel}"
            );
        }
        assert!(store
            .head(&paths.manifest_version(pre_compaction.version))
            .await
            .is_ok());

        // 2. The same closure held by a retention lease instead.
        let pin = RetentionPin::acquire(
            store.clone(),
            &paths,
            pre_compaction.version,
            Duration::from_secs(600),
        )
        .await
        .unwrap();
        sweep_orphans(&ms, u64::MAX, Duration::from_secs(86_400), 4, true)
            .await
            .unwrap();
        for rel in &l0_objects {
            assert!(
                store.head(&abs(rel)).await.is_ok(),
                "the lease-pinned version keeps {rel}"
            );
        }

        // 3. Released: the merged-away L0 objects and every retired manifest
        // version go; the never-referenced orphan still waits for min_age.
        pin.release().await.unwrap();
        let released = sweep_orphans(&ms, u64::MAX, Duration::from_secs(86_400), 4, true)
            .await
            .unwrap();
        for rel in &l0_objects {
            assert!(
                store.head(&abs(rel)).await.is_err(),
                "superseded {rel} must be reclaimed under the short grace: {released:?}"
            );
        }
        for version in 0..current.version {
            assert!(
                store.head(&paths.manifest_version(version)).await.is_err(),
                "retired manifest v{version} must be reclaimed: {released:?}"
            );
        }
        assert!(
            store.head(&orphan).await.is_ok(),
            "a never-referenced object waits out min_age"
        );
        assert!(released.skipped_too_young >= 1, "{released:?}");
        assert!(
            released.superseded_reclaimed >= l0_objects.len(),
            "{released:?}"
        );
        assert!(released.retired_versions_loaded > 0, "{released:?}");
        assert!(!released.exact_window);
        assert!(pinned.superseded_reclaimed >= wal_seqs.len(), "{pinned:?}");

        // The current version still reads every row.
        let rows = session.snapshot().scan_label("Person").await.unwrap();
        assert_eq!(rows.len(), 3);
    }

    /// The restore point window by write time keeps every object written
    /// since it opened, garbage included; the exact window keeps exactly
    /// what its versions reference. A compaction input merged away inside
    /// the window is still referenced by an in-window version and stays in
    /// both modes; an upload no version ever referenced is reclaimed only by
    /// the exact sweep. A missing version inside the window leaves its
    /// closure unproven, so the sweep falls back to keeping by write time.
    #[tokio::test]
    async fn exact_window_reclaims_garbage_written_inside_the_window() {
        let store: Arc<dyn ObjectStore> = Arc::new(AgedStore {
            inner: make_store(),
            age: chrono::Duration::hours(2),
        });
        let paths = make_paths("janitor-exact-window");
        let ms = ManifestStore::new(store.clone(), paths.clone());
        let abs = |rel: &str| {
            object_store::path::Path::from(format!("{}/{}", paths.namespace_prefix().as_ref(), rel))
        };
        let (mut session, commits) = session_with_commits(&store, &paths, &["A", "B"]).await;
        let window_floor = commits[0].0;
        session.flush(person_schema()).await.unwrap();
        session
            .upsert_node("Person", sorted_node_id(3), &node_record("C"))
            .unwrap();
        session.commit_batch().await.unwrap();
        session.flush(person_schema()).await.unwrap();
        let l0_objects: Vec<String> = ms
            .load_current()
            .await
            .unwrap()
            .manifest
            .ssts
            .iter()
            .flat_map(descriptor_object_paths)
            .collect();
        session.compact_l0(&person_schema()).await.unwrap();
        let current = ms.load_current().await.unwrap().manifest.version;
        assert!(current > window_floor + 2);

        // The window opened three hours ago at the first commit.
        let lease = PinLease {
            version: window_floor,
            expires_at_unix: chrono::Utc::now().timestamp() + 3600,
            scope: PinScope::Window {
                since: chrono::Utc::now() - chrono::Duration::hours(3),
            },
        };
        store
            .put(
                &paths.pin_object("restore-points"),
                PutPayload::from(serde_json::to_vec(&lease).unwrap()),
            )
            .await
            .unwrap();
        let orphan = paths.sst_object(0, "abandoned-output.parquet");
        store
            .put(&orphan, PutPayload::from(Bytes::from_static(b"abandoned")))
            .await
            .unwrap();
        let options = |exact_window| SweepOptions {
            exact_window,
            ..SweepOptions::new(u64::MAX, Duration::from_secs(3600), 4, true)
        };

        // By write time the window keeps the abandoned upload.
        let by_time = sweep_orphans_with(&ms, options(false)).await.unwrap();
        assert!(!by_time.exact_window);
        assert!(by_time.restore_window_retained >= 1, "{by_time:?}");
        assert_eq!(by_time.orphans_found, 0, "{by_time:?}");
        assert!(store.head(&orphan).await.is_ok());

        // Exact: the abandoned upload goes, the in-window history stays.
        let exact = sweep_orphans_with(&ms, options(true)).await.unwrap();
        assert!(exact.exact_window, "{exact:?}");
        assert_eq!(exact.restore_window_retained, 0, "{exact:?}");
        assert_eq!(exact.orphans_found, 1, "{exact:?}");
        assert!(
            exact.live_versions_loaded as u64 >= current - window_floor,
            "{exact:?}"
        );
        assert!(store.head(&orphan).await.is_err());
        for rel in &l0_objects {
            assert!(
                store.head(&abs(rel)).await.is_ok(),
                "an in-window version still references {rel}"
            );
        }
        for version in window_floor..=current {
            assert!(store.head(&paths.manifest_version(version)).await.is_ok());
        }
        assert!(exact.store_sst_bytes >= exact.live_sst_bytes);
        assert!(exact.live_sst_bytes > 0);

        // A version missing inside the window: back to keeping by write time.
        store
            .delete(&paths.manifest_version(window_floor + 1))
            .await
            .unwrap();
        store
            .put(&orphan, PutPayload::from(Bytes::from_static(b"abandoned")))
            .await
            .unwrap();
        let fallback = sweep_orphans_with(&ms, options(true)).await.unwrap();
        assert!(!fallback.exact_window, "{fallback:?}");
        assert!(store.head(&orphan).await.is_ok());
        for rel in &l0_objects {
            assert!(store.head(&abs(rel)).await.is_ok());
        }

        let rows = session.snapshot().scan_label("Person").await.unwrap();
        assert_eq!(rows.len(), 3);
    }

    /// Snapshot staleness follows the exact CURRENT WAL closure, not an SST
    /// high-water or retention horizon. Versioned readers recover from their
    /// retained WAL objects; the single overwritten snapshot is only a
    /// current-writer cache.
    #[tokio::test]
    async fn sweep_reclaims_snapshot_with_stale_wal_binding_despite_old_horizon() {
        let store = make_store();
        let paths = make_paths("janitor-snap-horizon");
        let (mut session, _) = session_with_commits(&store, &paths, &["Ada"]).await;
        // First flush: retained version v_f has no WAL.
        let v_f = session
            .flush(person_schema())
            .await
            .unwrap()
            .committed
            .manifest
            .version;
        // Second row: snapshot binds its commit WAL, then flush clears that
        // closure from the current manifest.
        session
            .upsert_node("Person", sorted_node_id(2), &node_record("Bob"))
            .unwrap();
        session.commit_batch().await.unwrap();
        session.write_memtable_snapshot_now().await.unwrap();
        session.flush(person_schema()).await.unwrap();
        let snap_path = paths.memtable_snapshot();

        let ms = ManifestStore::new(store.clone(), paths.clone());
        let pinned = sweep_orphans(&ms, v_f, Duration::from_secs(0), 4, true)
            .await
            .unwrap();
        assert_eq!(
            pinned.memtable_snapshots_reclaimed, 1,
            "a stale current-writer cache is not retained by an old manifest horizon"
        );
        assert!(store.head(&snap_path).await.is_err());

        let free = sweep_orphans(&ms, u64::MAX, Duration::from_secs(0), 4, true)
            .await
            .unwrap();
        assert_eq!(free.memtable_snapshots_reclaimed, 0);
        assert!(store.head(&snap_path).await.is_err());
    }
}
