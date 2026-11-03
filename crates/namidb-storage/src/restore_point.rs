//! Restore points: a retained window of manifest history, so a namespace can
//! be copied exactly as it was at an earlier instant.
//!
//! Every commit, flush, compaction and DDL publishes an immutable manifest
//! version stamped with its `created_at`. The namespace's state at an instant
//! `T` is the newest version committed at or before `T`, and a version is a
//! complete description of that state (its SSTs plus the WAL segments not yet
//! flushed into them). Each retained version is therefore a **restore
//! point**: [`crate::copy_namespace_snapshot`] pinned to it produces the
//! namespace row for row as it was then, whatever was written, deleted,
//! flushed or compacted afterwards.
//!
//! Normally the orphan sweep reclaims a version, and every object only it
//! references, as soon as no reader needs it. The restore point window keeps
//! them for a configured retention instead.
//!
//! ## The window lease
//!
//! The window is one retention pin lease per namespace,
//! `manifest/pins/restore-points.json` ([`crate::pin`], scope
//! [`PinScope::Window`]). Its `version` is the window's **floor**: the newest
//! version committed at or before `now - retention`, i.e. the state at the
//! start of the window. Its `since` is that version's `created_at`. The sweep
//! keeps every manifest body from the floor up, the floor's own closure, and
//! every SST, side-car and WAL object written since `since` (less the sweep's
//! `min_age`, the bound it already assumes between an upload and the commit
//! that references it). That covers every version in the window: an object a
//! version references is either in the floor's closure or was uploaded after
//! the floor committed, because a manifest never re-adds an object an earlier
//! version dropped. Keeping by write time is what makes the window cheap to
//! honour: the sweep loads one extra manifest, not one per version.
//!
//! The writer's maintenance loop calls [`refresh_restore_point_lease`] before
//! each sweep. The floor only ever moves forward. The lease expires at
//! `refresh + retention + renew_margin`: when the writer stops refreshing it
//! (the process is gone, or retention was turned off without a release),
//! every point it protected has aged out of the window by the time the sweep
//! stops honouring it. Only an unexpired lease is trusted on refresh: once a
//! lease lapses, a sweep may already have reclaimed history below current, so
//! the window starts again at the current version and fills up from there.
//! The window likewise starts empty the first time retention is enabled.
//!
//! ## Cost
//!
//! The window keeps everything the namespace wrote during the retention
//! period, plus the sweep's `min_age`: one manifest body per commit (its size
//! grows with the namespace's SST count), every WAL segment (one per commit),
//! every flush output, and every compaction input and output that later
//! compactions superseded. For a namespace that rewrites its data by
//! compaction, that is roughly `retention × (ingest rate × compaction write
//! amplification)` on top of the live data, plus `commits × manifest size`.
//! It also lengthens the sweep's and a cold open's LISTs of `manifest/` and
//! `wal/`, which grow with the number of commits in the window.
//! [`crate::JanitorReport::restore_window_retained`] counts the objects kept
//! only for the window on each sweep.

use std::time::Duration;

use chrono::{DateTime, Utc};
use futures::{StreamExt, TryStreamExt};
use object_store::{ObjectStoreExt, PutPayload};
use serde::{Deserialize, Serialize};
use tracing::{debug, warn};

use crate::error::{Error, Result};
use crate::manifest::{Manifest, ManifestStore};
use crate::pin::PinScope;

/// The fixed lease id of a namespace's restore point window
/// (`manifest/pins/restore-points.json`).
pub const RESTORE_POINT_LEASE_ID: &str = "restore-points";

/// Largest page [`list_restore_points`] returns.
pub const MAX_RESTORE_POINT_PAGE: usize = 1000;

/// How many manifest bodies a listing reads at once.
const LIST_CONCURRENCY: usize = 8;

/// One retained manifest version and the instant it was committed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct RestorePoint {
    pub version: u64,
    pub created_at: DateTime<Utc>,
}

impl RestorePoint {
    pub fn of(manifest: &Manifest) -> Self {
        Self {
            version: manifest.version,
            created_at: manifest.created_at,
        }
    }
}

/// The restore point window lease as the writer last wrote it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RestorePointLease {
    /// The window's floor: the oldest restore point it guarantees.
    pub floor: RestorePoint,
    /// When the sweep stops honouring the lease unless it is refreshed.
    pub expires_at: DateTime<Utc>,
    /// The retention the writer was configured with.
    pub retention: Duration,
}

/// The lease body. A superset of [`crate::PinLease`] (the sweep decodes only
/// `version`, `expires_at_unix` and `scope`); `retention_secs` is reporting
/// only.
#[derive(Debug, Serialize, Deserialize)]
struct WindowLeaseBody {
    version: u64,
    expires_at_unix: i64,
    scope: PinScope,
    retention_secs: u64,
}

/// The restore points a namespace can be copied at right now.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RestorePointWindow {
    /// The oldest guaranteed restore point: the lease's floor, or the current
    /// version when no unexpired lease exists.
    pub earliest: RestorePoint,
    /// The current committed version.
    pub latest: RestorePoint,
    /// The unexpired window lease, `None` when retention is off (or the
    /// writer has not refreshed it since it was turned on).
    pub lease: Option<RestorePointLease>,
}

/// One page of [`list_restore_points`].
#[derive(Debug, Clone)]
pub struct RestorePointPage {
    pub window: RestorePointWindow,
    /// Newest first.
    pub points: Vec<RestorePoint>,
    /// Pass as `before` to continue; `None` on the last page.
    pub next_before: Option<u64>,
}

/// What a backup should be a snapshot of.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AsOf {
    /// Exactly this manifest version.
    Version(u64),
    /// The newest version committed at or before this instant.
    Time(DateTime<Utc>),
}

/// Outcome of [`resolve_restore_point`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RestorePointResolution {
    Found(RestorePoint),
    /// The request falls outside the window (before its earliest point, or a
    /// version that is not retained).
    OutOfWindow(RestorePointWindow),
    /// A timestamp later than now: a snapshot of it cannot exist yet.
    InFuture {
        now: DateTime<Utc>,
    },
}

/// Read the version and commit time of a retained manifest without decoding
/// its SST list.
pub async fn load_restore_point(ms: &ManifestStore, version: u64) -> Result<RestorePoint> {
    #[derive(Deserialize)]
    struct Header {
        version: u64,
        created_at: DateTime<Utc>,
    }
    let path = ms.paths().manifest_version(version);
    let body = ms.store().get(&path).await?.bytes().await?;
    let header: Header = serde_json::from_slice(&body)?;
    if header.version != version {
        return Err(Error::Corrupted {
            path: path.as_ref().to_string(),
            detail: format!("manifest body names version {}", header.version),
        });
    }
    Ok(RestorePoint {
        version,
        created_at: header.created_at,
    })
}

/// The window lease, whether or not it has expired. `None` when absent or
/// not a window lease.
async fn read_window_lease(ms: &ManifestStore) -> Result<Option<WindowLeaseBody>> {
    let path = ms.paths().pin_object(RESTORE_POINT_LEASE_ID);
    let body = match ms.store().get(&path).await {
        Ok(res) => res.bytes().await?,
        Err(object_store::Error::NotFound { .. }) => return Ok(None),
        Err(e) => return Err(Error::ObjectStore(e)),
    };
    match serde_json::from_slice::<WindowLeaseBody>(&body) {
        Ok(lease) if matches!(lease.scope, PinScope::Window { .. }) => Ok(Some(lease)),
        _ => {
            warn!(path = %path, "restore point lease is not a window lease; ignoring it");
            Ok(None)
        }
    }
}

/// The unexpired window lease at `now`, with its floor's commit time.
fn live_lease(lease: Option<WindowLeaseBody>, now: DateTime<Utc>) -> Option<RestorePointLease> {
    let lease = lease?;
    if lease.expires_at_unix < now.timestamp() {
        return None;
    }
    let PinScope::Window { since } = lease.scope else {
        return None;
    };
    Some(RestorePointLease {
        floor: RestorePoint {
            version: lease.version,
            created_at: since,
        },
        expires_at: DateTime::from_timestamp(lease.expires_at_unix, 0)
            .unwrap_or(DateTime::<Utc>::MAX_UTC),
        retention: Duration::from_secs(lease.retention_secs),
    })
}

/// The newest point in `[lo, hi]` committed at or before `t`, by binary
/// search over the manifests in between (`O(log n)` reads). `None` when `lo`
/// itself is later than `t`. Commit times follow version order; were a
/// writer's clock to step back, the search still returns a version
/// committed at or before `t` whose successor was committed after it.
async fn newest_at_or_before(
    ms: &ManifestStore,
    mut lo: RestorePoint,
    mut hi: RestorePoint,
    t: DateTime<Utc>,
) -> Result<Option<RestorePoint>> {
    if hi.created_at <= t {
        return Ok(Some(hi));
    }
    if lo.created_at > t {
        return Ok(None);
    }
    while hi.version - lo.version > 1 {
        let mid = load_restore_point(ms, lo.version + (hi.version - lo.version) / 2).await?;
        if mid.created_at <= t {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    Ok(Some(lo))
}

/// Advance (or create) the namespace's restore point window lease for
/// `retention`, or release it when `retention` is zero. Returns the lease
/// written. `renew_margin` is added to the lease's lifetime so that it
/// cannot lapse between two refreshes: pass at least twice the interval the
/// caller refreshes at.
///
/// Only the namespace's writer calls this, before its sweep. See the module
/// docs for the floor rule.
pub async fn refresh_restore_point_lease(
    ms: &ManifestStore,
    retention: Duration,
    renew_margin: Duration,
) -> Result<Option<RestorePointLease>> {
    refresh_restore_point_lease_at(ms, retention, renew_margin, Utc::now()).await
}

/// [`refresh_restore_point_lease`] at an explicit `now` (tests).
pub async fn refresh_restore_point_lease_at(
    ms: &ManifestStore,
    retention: Duration,
    renew_margin: Duration,
    now: DateTime<Utc>,
) -> Result<Option<RestorePointLease>> {
    if retention.is_zero() {
        release_restore_point_lease(ms).await?;
        return Ok(None);
    }
    let latest = RestorePoint::of(&ms.load_current().await?.manifest);
    let held = live_lease(read_window_lease(ms).await?, now)
        .filter(|lease| lease.floor.version <= latest.version);
    let cutoff = chrono::Duration::from_std(retention)
        .ok()
        .and_then(|r| now.checked_sub_signed(r))
        .unwrap_or(DateTime::<Utc>::MIN_UTC);
    let floor = match held {
        // No trustworthy history: the window starts now.
        None => latest,
        Some(lease) => match load_restore_point(ms, lease.floor.version).await {
            Ok(lo) => newest_at_or_before(ms, lo, latest, cutoff)
                .await?
                // Everything retained is younger than the retention: keep it all.
                .unwrap_or(lo),
            Err(Error::ObjectStore(object_store::Error::NotFound { .. })) => {
                warn!(
                    version = lease.floor.version,
                    "restore point floor is gone; restarting the window at the current version"
                );
                latest
            }
            Err(e) => return Err(e),
        },
    };
    let lifetime = retention.saturating_add(renew_margin);
    let expires_at = chrono::Duration::from_std(lifetime)
        .ok()
        .and_then(|d| now.checked_add_signed(d))
        .unwrap_or(DateTime::<Utc>::MAX_UTC);
    let body = WindowLeaseBody {
        version: floor.version,
        expires_at_unix: expires_at.timestamp(),
        scope: PinScope::Window {
            since: floor.created_at,
        },
        retention_secs: retention.as_secs(),
    };
    ms.store()
        .put(
            &ms.paths().pin_object(RESTORE_POINT_LEASE_ID),
            PutPayload::from(serde_json::to_vec(&body)?),
        )
        .await?;
    debug!(
        floor = floor.version,
        latest = latest.version,
        "restore point window refreshed"
    );
    Ok(Some(RestorePointLease {
        floor,
        expires_at: DateTime::from_timestamp(body.expires_at_unix, 0).unwrap_or(expires_at),
        retention,
    }))
}

/// Delete the namespace's restore point window lease, so the next sweep
/// reclaims the history it kept. A missing lease is not an error.
pub async fn release_restore_point_lease(ms: &ManifestStore) -> Result<()> {
    match ms
        .store()
        .delete(&ms.paths().pin_object(RESTORE_POINT_LEASE_ID))
        .await
    {
        Ok(()) | Err(object_store::Error::NotFound { .. }) => Ok(()),
        Err(e) => Err(Error::ObjectStore(e)),
    }
}

/// The restore points the namespace can be copied at right now.
pub async fn restore_point_window(ms: &ManifestStore) -> Result<RestorePointWindow> {
    restore_point_window_at(ms, Utc::now()).await
}

async fn restore_point_window_at(
    ms: &ManifestStore,
    now: DateTime<Utc>,
) -> Result<RestorePointWindow> {
    let latest = RestorePoint::of(&ms.load_current().await?.manifest);
    let lease = live_lease(read_window_lease(ms).await?, now)
        .filter(|lease| lease.floor.version <= latest.version);
    let earliest = match lease {
        None => latest,
        Some(lease) => match load_restore_point(ms, lease.floor.version).await {
            Ok(point) => point,
            Err(Error::ObjectStore(object_store::Error::NotFound { .. })) => latest,
            Err(e) => return Err(e),
        },
    };
    Ok(RestorePointWindow {
        earliest,
        latest,
        lease,
    })
}

/// List the restore points in the window, newest first: those strictly
/// below `before` (all when `None`), at most `limit` (clamped to
/// `1..=MAX_RESTORE_POINT_PAGE`). Each point costs one manifest read.
pub async fn list_restore_points(
    ms: &ManifestStore,
    before: Option<u64>,
    limit: usize,
) -> Result<RestorePointPage> {
    let window = restore_point_window(ms).await?;
    let limit = limit.clamp(1, MAX_RESTORE_POINT_PAGE) as u64;
    let lo = window.earliest.version;
    let hi = match before {
        Some(0) => None,
        Some(b) => Some((b - 1).min(window.latest.version)),
        None => Some(window.latest.version),
    }
    .filter(|hi| *hi >= lo);
    let Some(hi) = hi else {
        return Ok(RestorePointPage {
            window,
            points: Vec::new(),
            next_before: None,
        });
    };
    let last = hi.saturating_sub(limit - 1).max(lo);
    let points: Vec<Option<RestorePoint>> = futures::stream::iter((last..=hi).rev())
        .map(|version| async move {
            match load_restore_point(ms, version).await {
                Ok(point) => Ok(Some(point)),
                // Reclaimed under us (the lease lapsed): not a restore point.
                Err(Error::ObjectStore(object_store::Error::NotFound { .. })) => Ok(None),
                Err(e) => Err(e),
            }
        })
        .buffered(LIST_CONCURRENCY)
        .try_collect()
        .await?;
    Ok(RestorePointPage {
        window,
        points: points.into_iter().flatten().collect(),
        next_before: (last > lo).then_some(last),
    })
}

/// Resolve `as_of` to the restore point a backup should pin.
pub async fn resolve_restore_point(
    ms: &ManifestStore,
    as_of: AsOf,
) -> Result<RestorePointResolution> {
    resolve_restore_point_at(ms, as_of, Utc::now()).await
}

/// [`resolve_restore_point`] at an explicit `now` (tests).
pub async fn resolve_restore_point_at(
    ms: &ManifestStore,
    as_of: AsOf,
    now: DateTime<Utc>,
) -> Result<RestorePointResolution> {
    if let AsOf::Time(t) = as_of {
        if t > now {
            return Ok(RestorePointResolution::InFuture { now });
        }
    }
    let window = restore_point_window_at(ms, now).await?;
    let found = match as_of {
        AsOf::Version(v) if v == window.latest.version => Some(window.latest),
        AsOf::Version(v) if v < window.earliest.version || v > window.latest.version => None,
        AsOf::Version(v) => match load_restore_point(ms, v).await {
            Ok(point) => Some(point),
            Err(Error::ObjectStore(object_store::Error::NotFound { .. })) => None,
            Err(e) => return Err(e),
        },
        AsOf::Time(t) => newest_at_or_before(ms, window.earliest, window.latest, t).await?,
    };
    Ok(match found {
        Some(point) => RestorePointResolution::Found(point),
        None => RestorePointResolution::OutOfWindow(window),
    })
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::sync::Arc;

    use namidb_core::{
        DataType, LabelDef, NamespaceId, NodeId, PropertyDef, Schema, SchemaBuilder,
    };
    use object_store::memory::InMemory;
    use object_store::ObjectStore;
    use uuid::Uuid;

    use super::*;
    use crate::flush::{EdgeWriteRecord, NodeWriteRecord};
    use crate::ingest::{CommitOutcome, WriterSession};
    use crate::paths::NamespacePaths;
    use crate::pin::PinLease;
    use crate::sweep_orphans;
    use namidb_core::Value;

    const HOUR: Duration = Duration::from_secs(3600);

    fn schema() -> Schema {
        SchemaBuilder::new()
            .label(LabelDef {
                name: "Person".into(),
                properties: vec![
                    PropertyDef::new("name", DataType::Utf8, false).unwrap(),
                    PropertyDef::new("rev", DataType::Int64, false).unwrap(),
                ],
            })
            .unwrap()
            .build()
    }

    fn id(n: u8) -> NodeId {
        let mut bytes = [0u8; 16];
        bytes[15] = n;
        NodeId::from_uuid(Uuid::from_bytes(bytes))
    }

    fn person(name: &str, rev: i64) -> NodeWriteRecord {
        NodeWriteRecord {
            properties: BTreeMap::from([
                ("name".to_string(), Value::Str(name.into())),
                ("rev".to_string(), Value::I64(rev)),
            ]),
            schema_version: 1,
            ..Default::default()
        }
    }

    /// The logical content of a namespace: every Person row and every KNOWS
    /// edge out of the ids this test uses.
    #[derive(Debug, Clone, PartialEq, Eq)]
    struct State {
        nodes: Vec<(NodeId, BTreeMap<String, String>)>,
        edges: Vec<(NodeId, NodeId)>,
    }

    async fn state_of(session: &WriterSession) -> State {
        let snapshot = session.snapshot();
        let mut nodes: Vec<_> = snapshot
            .scan_label("Person")
            .await
            .unwrap()
            .into_iter()
            .map(|n| {
                let props = n
                    .properties
                    .iter()
                    .map(|(k, v)| (k.clone(), format!("{v:?}")))
                    .collect();
                (n.id, props)
            })
            .collect();
        nodes.sort_by_key(|(id, _)| *id);
        let mut edges = Vec::new();
        for n in 1..=12 {
            for e in snapshot.out_edges("KNOWS", id(n)).await.unwrap().edges {
                edges.push((e.src, e.dst));
            }
        }
        edges.sort();
        State { nodes, edges }
    }

    fn paths(name: &str) -> NamespacePaths {
        NamespacePaths::new("tenants", NamespaceId::new(name).unwrap())
    }

    async fn commit(session: &mut WriterSession) -> u64 {
        match session.commit_batch().await.unwrap() {
            CommitOutcome::Committed {
                manifest_version, ..
            } => manifest_version,
            other => panic!("expected a commit, got {other:?}"),
        }
    }

    /// Copy the source at `version` into a fresh store and read it back.
    async fn restored_state(
        store: &Arc<dyn ObjectStore>,
        paths: &NamespacePaths,
        version: u64,
    ) -> crate::Result<State> {
        // A fresh namespace name per restore: the read caches are
        // process-wide and keyed by object path, so restoring two versions
        // under one name would serve the first one's cached bodies.
        static RESTORES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let n = RESTORES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let dst: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let dst_paths = NamespacePaths::new(
            "tenants",
            NamespaceId::new(format!("restored-{n}")).unwrap(),
        );
        crate::copy_namespace_snapshot(
            store.clone(),
            paths.clone(),
            dst.clone(),
            dst_paths.clone(),
            Some(version),
            false,
            true,
        )
        .await?;
        let session = WriterSession::open(dst, dst_paths).await?;
        Ok(state_of(&session).await)
    }

    /// A workload that exercises every way history disappears: commits that
    /// stay in the WAL, flushes that retire WAL segments, updates, deletes,
    /// edges, and compactions that supersede SSTs. Returns the expected state
    /// at every manifest version it published.
    async fn churn(
        store: &Arc<dyn ObjectStore>,
        paths: &NamespacePaths,
    ) -> (WriterSession, Vec<(u64, State)>) {
        let mut session = WriterSession::open(store.clone(), paths.clone())
            .await
            .unwrap();
        let mut history = vec![(session.manifest_version(), state_of(&session).await)];
        for round in 0..3i64 {
            for n in 1..=4u8 {
                let node = n + (round as u8) * 4;
                session
                    .upsert_node("Person", id(node), &person(&format!("p{node}"), round))
                    .unwrap();
                if node > 1 {
                    session
                        .upsert_edge("KNOWS", id(node - 1), id(node), &EdgeWriteRecord::default())
                        .unwrap();
                }
                let v = commit(&mut session).await;
                history.push((v, state_of(&session).await));
            }
            // Rewrite an older row and delete another, so the newest state
            // differs from every SST that held the old values.
            session
                .upsert_node("Person", id(1), &person("renamed", 100 + round))
                .unwrap();
            if round > 0 {
                session
                    .tombstone_node("Person", id(round as u8 * 4))
                    .unwrap();
                session
                    .tombstone_edge("KNOWS", id(round as u8 * 4 - 1), id(round as u8 * 4))
                    .unwrap();
            }
            let v = commit(&mut session).await;
            history.push((v, state_of(&session).await));
            session.flush(schema()).await.unwrap();
            history.push((session.manifest_version(), state_of(&session).await));
            if round > 0 {
                session.compact_l0(&schema()).await.unwrap();
                history.push((session.manifest_version(), state_of(&session).await));
            }
        }
        // A last unflushed commit: the newest points live partly in the WAL.
        session
            .upsert_node("Person", id(12), &person("tail", 9))
            .unwrap();
        let v = commit(&mut session).await;
        history.push((v, state_of(&session).await));
        history.dedup_by_key(|(v, _)| *v);
        (session, history)
    }

    async fn sweep(ms: &ManifestStore) -> crate::JanitorReport {
        // A dense horizon at current (no live readers), no age margin: the
        // most aggressive sweep the writer can run.
        sweep_orphans(ms, u64::MAX, Duration::ZERO, 4, true)
            .await
            .unwrap()
    }

    /// Write a window lease at `version` directly (what a refresh writes).
    async fn plant_window(ms: &ManifestStore, version: u64, expires_at_unix: i64) {
        let since = load_restore_point(ms, version).await.unwrap().created_at;
        let body = WindowLeaseBody {
            version,
            expires_at_unix,
            scope: PinScope::Window { since },
            retention_secs: 3600,
        };
        ms.store()
            .put(
                &ms.paths().pin_object(RESTORE_POINT_LEASE_ID),
                PutPayload::from(serde_json::to_vec(&body).unwrap()),
            )
            .await
            .unwrap();
    }

    #[tokio::test]
    async fn every_point_in_the_window_survives_gc_and_restores_row_for_row() {
        let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let paths = paths("pitr-window");
        let ms = ManifestStore::new(store.clone(), paths.clone());
        // The window opens at the namespace's first version.
        WriterSession::open(store.clone(), paths.clone())
            .await
            .unwrap();
        let floor = ms.load_current().await.unwrap().manifest.version;
        plant_window(&ms, floor, chrono::Utc::now().timestamp() + 3600).await;

        let (session, history) = churn(&store, &paths).await;
        assert!(
            history.len() > 15,
            "the workload must publish many versions"
        );

        // The aggressive sweep runs between every write in production; here
        // once after all of them, which is the worst case for history.
        let report = sweep(&ms).await;
        assert!(
            report.restore_window_retained > 0,
            "the window must be what kept the superseded objects: {report:?}"
        );
        assert_eq!(report.manifest_snapshots_reclaimed, 0, "{report:?}");
        // More flushing and compaction after the sweep must not disturb the
        // retained points either.
        let mut session = session;
        session.flush(schema()).await.unwrap();
        session.compact_l0(&schema()).await.unwrap();
        sweep(&ms).await;

        for (version, expected) in &history {
            let got = restored_state(&store, &paths, *version)
                .await
                .unwrap_or_else(|e| panic!("restore point v{version} broke: {e}"));
            assert_eq!(&got, expected, "v{version} must restore row for row");
        }
        // And the states really differ: the test is not comparing one state
        // to itself.
        assert_ne!(history.first().unwrap().1, history.last().unwrap().1);
    }

    /// A floor in the middle of history: everything from the floor up stays
    /// restorable, including objects written before the window opened that
    /// later versions still reference (the floor's closure keeps them), and
    /// everything older is reclaimed.
    #[tokio::test]
    async fn a_mid_history_floor_keeps_exactly_the_versions_from_it_up() {
        let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let paths = paths("pitr-mid");
        let ms = ManifestStore::new(store.clone(), paths.clone());
        let (_session, history) = churn(&store, &paths).await;
        let k = history.len() / 2;
        let floor = history[k].0;
        plant_window(&ms, floor, chrono::Utc::now().timestamp() + 3600).await;
        let report = sweep(&ms).await;
        assert!(report.manifest_snapshots_reclaimed > 0, "{report:?}");
        for (version, expected) in &history {
            let restored = restored_state(&store, &paths, *version).await;
            if *version >= floor {
                assert_eq!(
                    &restored.unwrap_or_else(|e| panic!("v{version} broke: {e}")),
                    expected
                );
            } else {
                assert!(restored.is_err(), "v{version} is below the floor");
            }
        }
    }

    /// The control for the tests above: without the window the same sweep
    /// reclaims the history, so the window is what kept it.
    #[tokio::test]
    async fn without_the_window_the_same_sweep_reclaims_history() {
        let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let paths = paths("pitr-control");
        let ms = ManifestStore::new(store.clone(), paths.clone());
        let (_session, history) = churn(&store, &paths).await;
        sweep(&ms).await;
        let broken = futures::future::join_all(
            history
                .iter()
                .map(|(version, _)| restored_state(&store, &paths, *version)),
        )
        .await
        .into_iter()
        .filter(|r| r.is_err())
        .count();
        assert!(
            broken >= history.len() - 1,
            "only the current version may survive a sweep without the window ({broken}/{})",
            history.len()
        );
    }

    #[tokio::test]
    async fn an_expired_window_releases_its_history() {
        let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let paths = paths("pitr-expiry");
        let ms = ManifestStore::new(store.clone(), paths.clone());
        WriterSession::open(store.clone(), paths.clone())
            .await
            .unwrap();
        plant_window(&ms, 0, chrono::Utc::now().timestamp() + 3600).await;
        let (_session, history) = churn(&store, &paths).await;
        let held = sweep(&ms).await;
        assert_eq!(held.manifest_snapshots_reclaimed, 0);
        assert!(restored_state(&store, &paths, history[1].0).await.is_ok());

        // The writer stopped refreshing: the lease outlives its window and
        // lapses. The next sweep stops honouring it and reclaims history.
        plant_window(&ms, 0, chrono::Utc::now().timestamp() - 1).await;
        let released = sweep(&ms).await;
        assert_eq!(released.expired_pins_reclaimed, 1);
        assert!(released.manifest_snapshots_reclaimed > 0, "{released:?}");
        assert_eq!(released.restore_window_retained, 0);
        assert!(restored_state(&store, &paths, history[1].0).await.is_err());
        // The lease object itself is gone too.
        assert!(store
            .head(&paths.pin_object(RESTORE_POINT_LEASE_ID))
            .await
            .is_err());
        // The window reads as empty: only the current version.
        let window = restore_point_window(&ms).await.unwrap();
        assert_eq!(window.lease, None);
        assert_eq!(window.earliest, window.latest);
    }

    #[tokio::test]
    async fn refresh_advances_the_floor_with_time_and_zero_retention_releases() {
        let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let paths = paths("pitr-refresh");
        let ms = ManifestStore::new(store.clone(), paths.clone());
        WriterSession::open(store.clone(), paths.clone())
            .await
            .unwrap();
        // First enable: the window starts at the current version.
        let now = chrono::Utc::now();
        let lease = refresh_restore_point_lease_at(&ms, HOUR, HOUR, now)
            .await
            .unwrap()
            .unwrap();
        let start = ms.load_current().await.unwrap().manifest.version;
        assert_eq!(lease.floor.version, start);
        assert_eq!(lease.retention, HOUR);
        assert_eq!(
            lease.expires_at.timestamp(),
            (now + chrono::Duration::hours(2)).timestamp()
        );

        let (_session, history) = churn(&store, &paths).await;
        let latest = history.last().unwrap().0;

        // An hour has not passed: every version is still in the window.
        let lease = refresh_restore_point_lease_at(&ms, HOUR, HOUR, chrono::Utc::now())
            .await
            .unwrap()
            .unwrap();
        assert_eq!(lease.floor.version, start);

        // Pretend the clock reached a point between two commits plus the
        // retention: the floor becomes the newest version at or before the
        // cutoff — the state at the start of the window.
        let pick = history[history.len() / 2].0;
        let at = load_restore_point(&ms, pick).await.unwrap().created_at;
        let lease =
            refresh_restore_point_lease_at(&ms, HOUR, HOUR, at + chrono::Duration::hours(1))
                .await
                .unwrap()
                .unwrap();
        assert!(lease.floor.version >= pick, "{lease:?}");
        let floor_at = lease.floor.created_at;
        assert!(floor_at <= at, "the floor must be committed by the cutoff");
        if lease.floor.version < latest {
            let next = load_restore_point(&ms, lease.floor.version + 1)
                .await
                .unwrap();
            assert!(
                next.created_at > at,
                "the floor must be the NEWEST such version"
            );
        }

        // The floor never moves backwards, even if asked about an earlier
        // instant.
        let again = refresh_restore_point_lease_at(&ms, HOUR, HOUR, at)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(again.floor.version, lease.floor.version);

        // Far in the future: the floor reaches the current version.
        let late = refresh_restore_point_lease_at(
            &ms,
            HOUR,
            HOUR,
            chrono::Utc::now() + chrono::Duration::hours(1),
        )
        .await
        .unwrap()
        .unwrap();
        assert_eq!(late.floor.version, latest);

        // Retention off: the lease is released.
        assert!(refresh_restore_point_lease(&ms, Duration::ZERO, HOUR)
            .await
            .unwrap()
            .is_none());
        assert!(store
            .head(&paths.pin_object(RESTORE_POINT_LEASE_ID))
            .await
            .is_err());
    }

    #[tokio::test]
    async fn an_expired_lease_is_not_trusted_on_refresh() {
        let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let paths = paths("pitr-untrusted");
        let ms = ManifestStore::new(store.clone(), paths.clone());
        WriterSession::open(store.clone(), paths.clone())
            .await
            .unwrap();
        plant_window(&ms, 0, chrono::Utc::now().timestamp() - 1).await;
        let (_session, history) = churn(&store, &paths).await;
        let lease = refresh_restore_point_lease(&ms, HOUR, HOUR)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(
            lease.floor.version,
            history.last().unwrap().0,
            "a lapsed lease may have lost history; the window restarts at current"
        );
    }

    #[tokio::test]
    async fn the_lease_body_is_a_window_pin_lease_to_the_sweep() {
        let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let paths = paths("pitr-lease-body");
        let ms = ManifestStore::new(store.clone(), paths.clone());
        WriterSession::open(store.clone(), paths.clone())
            .await
            .unwrap();
        let lease = refresh_restore_point_lease(&ms, HOUR, HOUR)
            .await
            .unwrap()
            .unwrap();
        let body = store
            .get(&paths.pin_object(RESTORE_POINT_LEASE_ID))
            .await
            .unwrap()
            .bytes()
            .await
            .unwrap();
        let pin: PinLease = serde_json::from_slice(&body).unwrap();
        assert_eq!(pin.version, lease.floor.version);
        assert_eq!(
            pin.scope,
            PinScope::Window {
                since: lease.floor.created_at
            }
        );
        assert_eq!(pin.expires_at_unix, lease.expires_at.timestamp());
    }

    #[tokio::test]
    async fn resolve_and_list_cover_exactly_the_window() {
        let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let paths = paths("pitr-resolve");
        let ms = ManifestStore::new(store.clone(), paths.clone());
        WriterSession::open(store.clone(), paths.clone())
            .await
            .unwrap();
        // Retention off: the window is just the current version.
        let window = restore_point_window(&ms).await.unwrap();
        assert_eq!(window.earliest, window.latest);
        assert_eq!(window.lease, None);

        refresh_restore_point_lease(&ms, HOUR, HOUR).await.unwrap();
        let (_session, history) = churn(&store, &paths).await;
        let window = restore_point_window(&ms).await.unwrap();
        assert_eq!(window.latest.version, history.last().unwrap().0);
        assert!(window.lease.is_some());
        let earliest = window.earliest;

        // Every version resolves to itself, by version and by its own commit
        // time; an instant between two commits resolves to the older one.
        let points: Vec<RestorePoint> = futures::future::try_join_all(
            (earliest.version..=window.latest.version).map(|v| load_restore_point(&ms, v)),
        )
        .await
        .unwrap();
        for pair in points.windows(2) {
            let (a, b) = (pair[0], pair[1]);
            assert_eq!(
                resolve_restore_point(&ms, AsOf::Version(a.version))
                    .await
                    .unwrap(),
                RestorePointResolution::Found(a)
            );
            // Versions committed within the same instant resolve to the
            // newest of them; otherwise to exactly `a`.
            let by_time = resolve_restore_point(&ms, AsOf::Time(a.created_at))
                .await
                .unwrap();
            let RestorePointResolution::Found(found) = by_time else {
                panic!("{by_time:?}");
            };
            assert_eq!(found.created_at, a.created_at);
            assert!(found.version >= a.version);
            if b.created_at > a.created_at + chrono::Duration::microseconds(1) {
                let between = a.created_at + (b.created_at - a.created_at) / 2;
                assert_eq!(
                    resolve_restore_point(&ms, AsOf::Time(between))
                        .await
                        .unwrap(),
                    RestorePointResolution::Found(found)
                );
            }
        }
        // Now (or anything after the last commit) is the latest point.
        assert_eq!(
            resolve_restore_point(&ms, AsOf::Time(chrono::Utc::now()))
                .await
                .unwrap(),
            RestorePointResolution::Found(window.latest)
        );
        // Before the window, after the latest version, in the future.
        let before = earliest.created_at - chrono::Duration::seconds(1);
        assert!(matches!(
            resolve_restore_point(&ms, AsOf::Time(before))
                .await
                .unwrap(),
            RestorePointResolution::OutOfWindow(_)
        ));
        assert!(matches!(
            resolve_restore_point(&ms, AsOf::Version(window.latest.version + 1))
                .await
                .unwrap(),
            RestorePointResolution::OutOfWindow(_)
        ));
        assert!(matches!(
            resolve_restore_point(
                &ms,
                AsOf::Time(chrono::Utc::now() + chrono::Duration::minutes(5))
            )
            .await
            .unwrap(),
            RestorePointResolution::InFuture { .. }
        ));

        // Paging walks the window newest first, without gaps or repeats.
        let mut seen = Vec::new();
        let mut before = None;
        loop {
            let page = list_restore_points(&ms, before, 4).await.unwrap();
            assert!(page.points.len() <= 4);
            seen.extend(page.points.iter().map(|p| p.version));
            match page.next_before {
                Some(next) => before = Some(next),
                None => break,
            }
        }
        let expected: Vec<u64> = (earliest.version..=window.latest.version).rev().collect();
        assert_eq!(seen, expected);
        // A cursor below the window is an empty last page.
        let empty = list_restore_points(&ms, Some(earliest.version), 10)
            .await
            .unwrap();
        assert!(empty.points.is_empty());
        assert_eq!(empty.next_before, None);
    }

    /// A backup of an old restore point pins just that version (a point
    /// lease): the sweep keeps its closure even after the window lease is
    /// gone, without holding back everything newer.
    #[tokio::test]
    async fn a_point_pin_keeps_one_old_version_without_a_window() {
        let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let paths = paths("pitr-point");
        let ms = ManifestStore::new(store.clone(), paths.clone());
        WriterSession::open(store.clone(), paths.clone())
            .await
            .unwrap();
        plant_window(&ms, 0, chrono::Utc::now().timestamp() + 3600).await;
        let (_session, history) = churn(&store, &paths).await;
        let (old, expected) = history[3].clone();
        let pin = crate::RetentionPin::acquire_scoped(
            store.clone(),
            &paths,
            old,
            PinScope::Point,
            crate::DEFAULT_PIN_TTL,
        )
        .await
        .unwrap();
        release_restore_point_lease(&ms).await.unwrap();
        let report = sweep(&ms).await;
        assert!(report.manifest_snapshots_reclaimed > 0, "{report:?}");
        assert_eq!(restored_state(&store, &paths, old).await.unwrap(), expected);
        // A neighbour the point lease does not cover is gone.
        assert!(restored_state(&store, &paths, old + 1).await.is_err());
        pin.release().await.unwrap();
    }
}
