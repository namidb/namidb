//! Automatic writer-session recovery.
//!
//! `WriterSession::commit_batch` / `flush` document that a fenced epoch, a
//! lost manifest CAS, or a poisoned session mean "drop the session and
//! reopen". This module is the server-side owner of that contract: after a
//! failed commit/flush the write paths (HTTP, Bolt, multi-tenant, the
//! periodic flush tasks) call [`recover_writer_if_needed`] while still
//! holding the writer lock. It reopens the namespace with bounded retries
//! and republishes the snapshot so readers observe the recovered session.
//!
//! [`WriterHealth`] carries the outcome to the readiness probe
//! (`/v0/health`): the writer reports `degraded` from the terminal failure
//! until a reopen lands, so an orchestrator can stop routing writes to a
//! server whose writer is permanently broken (e.g. an orphan manifest body
//! blocking every claim) instead of reading a green health check forever.

use std::sync::Arc;
use std::time::Duration;

use namidb_storage::{SnapshotCell, WriterSession};
use tracing::{info, warn};

/// Reopen attempts per recovery pass. The first attempt usually wins (a
/// fence just needs a fresh epoch claim); the retries cover a transient
/// store error during the claim itself.
const REOPEN_ATTEMPTS: u32 = 3;

/// Base backoff between reopen attempts, scaled linearly per attempt.
const REOPEN_BACKOFF: Duration = Duration::from_millis(50);

/// What a writer does after a peer claims its namespace and fences it
/// (`Error::Fenced`). Parsed from `--fence-policy`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum FencePolicy {
    /// Reopen the session, which claims the epoch back. Right for a single
    /// server that should keep writing no matter what touched its bucket.
    #[default]
    Steal,
    /// Stay fenced and report the writer degraded (readiness 503). Right
    /// when something outside this process (a control plane, an
    /// orchestrator) decides which node owns a namespace: the new owner
    /// claims the epoch and the old one steps aside instead of fencing it
    /// back. Once a writer has yielded it stays down until the process (or
    /// the namespace, in multi-tenant mode) is opened again.
    Yield,
}

impl std::str::FromStr for FencePolicy {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_ascii_lowercase().as_str() {
            "steal" => Ok(Self::Steal),
            "yield" => Ok(Self::Yield),
            other => Err(format!(
                "unknown fence policy `{other}` (expected `steal` or `yield`)"
            )),
        }
    }
}

/// Writer status for the readiness probe. `Some(reason)` while the writer
/// session is broken — a terminal commit/flush failure happened and the
/// automatic reopen has not yet succeeded. Read lock-free by `/v0/health`
/// (never the writer mutex, which a long write may hold).
#[derive(Debug, Default)]
pub struct WriterHealth {
    degraded: std::sync::Mutex<Option<String>>,
    /// Local-persistence failure (spool disk full/unwritable during flush).
    /// Kept separate from `degraded`: that slot is owned by the
    /// fence-probe/commit-recovery machinery and its prefix-matching clear
    /// logic, while this one is set and cleared strictly by flush outcomes.
    /// While set, write intake is rejected with a typed error (the memtable
    /// cannot drain, so accepting writes only grows an unbounded buffer and
    /// turns into "writer is busy" confusion); reads keep serving the last
    /// committed snapshot. It self-clears on the first successful flush.
    persistence: std::sync::Mutex<Option<String>>,
    /// Whether a fenced writer reopens or steps aside. See [`FencePolicy`].
    fence_policy: FencePolicy,
    /// Set once a fenced writer yields under [`FencePolicy::Yield`]. Sticky:
    /// the yielded session is still fenced, and later failures on it (a
    /// poisoned-session refusal from the next flush tick, say) no longer
    /// carry `Error::Fenced`, so without the latch they would reopen and
    /// fence the new owner after all.
    yielded: std::sync::atomic::AtomicBool,
}

impl WriterHealth {
    pub fn new() -> Arc<Self> {
        Arc::new(Self::default())
    }

    pub fn with_fence_policy(fence_policy: FencePolicy) -> Arc<Self> {
        Arc::new(Self {
            fence_policy,
            ..Self::default()
        })
    }

    /// The failure keeping the writer degraded, or `None` when healthy.
    /// A local-persistence failure reports here too so `/v0/health` and
    /// readiness surface it without a second field.
    pub fn degraded_reason(&self) -> Option<String> {
        self.degraded
            .lock()
            .expect("writer health poisoned")
            .clone()
            .or_else(|| self.persistence_degraded_reason())
    }

    /// The local-persistence failure currently rejecting write intake, or
    /// `None`. Write handlers consult this before queueing on the writer
    /// mutex; reads never consult it.
    pub fn persistence_degraded_reason(&self) -> Option<String> {
        self.persistence
            .lock()
            .expect("writer health poisoned")
            .clone()
    }

    pub(crate) fn mark_persistence_degraded(&self, reason: String) {
        *self.persistence.lock().expect("writer health poisoned") = Some(reason);
    }

    pub(crate) fn clear_persistence_degraded(&self) {
        *self.persistence.lock().expect("writer health poisoned") = None;
    }

    /// `"ok"` / `"degraded"` — the `writer` field of the health payload.
    pub fn status(&self) -> &'static str {
        if self.is_degraded() {
            "degraded"
        } else {
            "ok"
        }
    }

    pub fn is_degraded(&self) -> bool {
        self.degraded
            .lock()
            .expect("writer health poisoned")
            .is_some()
    }

    fn mark_degraded(&self, reason: String) {
        *self.degraded.lock().expect("writer health poisoned") = Some(reason);
    }

    fn mark_ok(&self) {
        *self.degraded.lock().expect("writer health poisoned") = None;
    }

    /// Feed one read-fence probe observation (RFC-027): `observed` is the
    /// epoch of the store's current manifest pointer, `local` the epoch of
    /// the published snapshot. A higher observed epoch means a peer writer
    /// has fenced this node — its published snapshot is going stale and
    /// every local write will fail, so readiness must drop until the
    /// session reopens (steals the epoch back) or traffic drains. Epoch
    /// parity clears only a probe-set reason, never a commit-failure one
    /// (the recovery path owns that).
    pub(crate) fn observe_peer_epoch(&self, observed: u64, local: u64) {
        let mut slot = self.degraded.lock().expect("writer health poisoned");
        if observed > local {
            *slot = Some(format!(
                "{FENCE_PROBE_PREFIX}: observed epoch {observed} > local {local} \
                 (published snapshot is stale; writes will fail until reopen)"
            ));
        } else if slot
            .as_deref()
            .is_some_and(|r| r.starts_with(FENCE_PROBE_PREFIX))
        {
            *slot = None;
        }
    }
}

/// Reason prefix for degradation set by the read-fence probe, so epoch
/// parity clears only probe-set reasons.
const FENCE_PROBE_PREFIX: &str = "fenced by peer writer";

/// Read-side fence probe: compare the store's current manifest epoch with
/// the published snapshot's, lock-free (no writer mutex, one advisory GET).
/// Fencing was write-path-only — a fenced zombie kept serving stale reads
/// with a green health check indefinitely; this turns that split-brain
/// window into a readiness failure. Called from the periodic maintenance
/// loops (single- and multi-tenant).
pub(crate) async fn probe_read_fence(
    manifest_store: &namidb_storage::ManifestStore,
    snapshot: &SnapshotCell,
    health: &WriterHealth,
    namespace: &str,
) {
    match manifest_store.load_current().await {
        Ok(current) => {
            let observed = current.manifest.epoch.as_u64();
            let local = snapshot.load().manifest().manifest.epoch.as_u64();
            if observed > local {
                warn!(
                    namespace,
                    observed_epoch = observed,
                    local_epoch = local,
                    "read-fence probe: a peer writer holds a higher epoch"
                );
            }
            health.observe_peer_epoch(observed, local);
        }
        // A probe failure is not a health signal by itself (the store may
        // be briefly unreachable); the next tick retries.
        Err(e) => tracing::debug!(namespace, error = %e, "read-fence probe failed"),
    }
}

/// If `err` means the writer session is beyond in-place retry — fenced,
/// lost the manifest CAS, or poisoned by a terminal commit failure — reopen
/// it under the caller's writer lock: up to [`REOPEN_ATTEMPTS`] attempts
/// with backoff, republishing the snapshot on success so readers observe
/// the recovered session. `health` is degraded for the duration and ok
/// again once the reopen lands; if every attempt fails it stays degraded
/// and the next failed write triggers another pass. Under
/// [`FencePolicy::Yield`] a fence does not reopen: the writer stays degraded
/// and the peer that claimed the namespace keeps it.
///
/// Holding the lock across the backoff is deliberate: the writer is broken,
/// so a queued write could only fail — better to have it wait for a
/// recovered session than fail and re-trigger recovery itself.
pub(crate) async fn recover_writer_if_needed(
    writer: &mut WriterSession,
    snapshot: &SnapshotCell,
    health: &WriterHealth,
    namespace: &str,
    err: &namidb_storage::Error,
) {
    if !(err.requires_writer_reopen() || writer.is_poisoned()) {
        return;
    }
    if health.fence_policy == FencePolicy::Yield {
        if health.yielded.load(std::sync::atomic::Ordering::Acquire) {
            // Already yielded: keep the fence reason and stay down.
            return;
        }
        if matches!(err, namidb_storage::Error::Fenced { .. }) {
            // A peer owns the namespace now. Reopening would fence it right
            // back, so stay degraded and leave ownership to whoever placed it.
            health.mark_degraded(err.to_string());
            health
                .yielded
                .store(true, std::sync::atomic::Ordering::Release);
            warn!(
                namespace,
                error = %err,
                "writer session is fenced; yielding the namespace (fence policy: yield)"
            );
            return;
        }
    }
    health.mark_degraded(err.to_string());
    warn!(
        namespace,
        error = %err,
        "writer session is fenced/poisoned; reopening the namespace"
    );
    for attempt in 1..=REOPEN_ATTEMPTS {
        match writer.reopen().await {
            Ok(()) => {
                snapshot.store(writer.owned_snapshot());
                health.mark_ok();
                info!(
                    namespace,
                    attempt,
                    manifest_version = writer.manifest_version(),
                    "writer session reopened; writes restored"
                );
                return;
            }
            Err(e) => {
                warn!(namespace, attempt, error = %e, "writer reopen failed");
                if attempt < REOPEN_ATTEMPTS {
                    tokio::time::sleep(REOPEN_BACKOFF * attempt).await;
                }
            }
        }
    }
    // Still broken: health stays degraded so /v0/health reports it, and the
    // next failed write re-enters this path.
}

/// [`recover_writer_if_needed`] for a failed write statement: the executor
/// wraps commit failures in `ExecError::Storage`, everything else (eval,
/// constraint, timeout, row cap) cannot have broken the writer session.
pub(crate) async fn recover_after_write_error(
    writer: &mut WriterSession,
    snapshot: &SnapshotCell,
    health: &WriterHealth,
    namespace: &str,
    err: &namidb_query::exec::ExecError,
) {
    if let namidb_query::exec::ExecError::Storage(storage_err) = err {
        recover_writer_if_needed(writer, snapshot, health, namespace, storage_err).await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn observe_peer_epoch_degrades_and_clears_only_probe_reasons() {
        let health = WriterHealth::new();

        // Higher observed epoch → degraded with the probe reason.
        health.observe_peer_epoch(3, 1);
        assert!(health.is_degraded());
        assert!(health
            .degraded_reason()
            .unwrap()
            .starts_with(FENCE_PROBE_PREFIX));

        // Parity clears the probe-set reason.
        health.observe_peer_epoch(3, 3);
        assert!(!health.is_degraded());

        // A commit-failure reason is NOT cleared by epoch parity — the
        // recovery path owns it.
        health.mark_degraded("commit failed: fenced".to_string());
        health.observe_peer_epoch(2, 2);
        assert!(health.is_degraded(), "probe must not clear recovery state");
    }

    #[tokio::test]
    async fn probe_detects_a_peer_claim_and_recovers_after_reopen() {
        use namidb_core::NamespaceId;
        use namidb_storage::{ManifestStore, NamespacePaths, WriterSession};
        use std::sync::Arc as StdArc;

        let store: StdArc<dyn object_store::ObjectStore> =
            StdArc::new(object_store::memory::InMemory::new());
        let paths = NamespacePaths::new("tenants", NamespaceId::new("fence-probe").unwrap());
        let ms = ManifestStore::new(store.clone(), paths.clone());

        let mut a = WriterSession::open(store.clone(), paths.clone())
            .await
            .unwrap();
        let cell = SnapshotCell::new(a.owned_snapshot());
        let health = WriterHealth::new();

        // Same-epoch probe: healthy.
        probe_read_fence(&ms, &cell, &health, "fence-probe").await;
        assert!(!health.is_degraded());

        // A peer claims the namespace: its epoch outranks ours.
        let _b = WriterSession::open(store.clone(), paths.clone())
            .await
            .unwrap();
        probe_read_fence(&ms, &cell, &health, "fence-probe").await;
        assert!(
            health.is_degraded(),
            "peer epoch must drop readiness: {:?}",
            health.degraded_reason()
        );

        // Reopen steals the epoch back and republishes; the probe clears.
        a.reopen().await.unwrap();
        cell.store(a.owned_snapshot());
        probe_read_fence(&ms, &cell, &health, "fence-probe").await;
        assert!(!health.is_degraded(), "{:?}", health.degraded_reason());
    }

    /// Under `FencePolicy::Yield` a fenced writer stays degraded and leaves
    /// the namespace to the peer that claimed it; `Steal` reopens and takes
    /// the epoch back.
    #[tokio::test]
    async fn fenced_writer_yields_when_policy_is_yield() {
        use namidb_core::NamespaceId;
        use namidb_storage::{ManifestStore, NamespacePaths, WriterSession};
        use std::sync::Arc as StdArc;

        let store: StdArc<dyn object_store::ObjectStore> =
            StdArc::new(object_store::memory::InMemory::new());
        let paths = NamespacePaths::new("tenants", NamespaceId::new("fence-yield").unwrap());
        let ms = ManifestStore::new(store.clone(), paths.clone());
        let current_epoch = || async { ms.load_current().await.unwrap().manifest.epoch.as_u64() };

        let mut a = WriterSession::open(store.clone(), paths.clone())
            .await
            .unwrap();
        let cell = SnapshotCell::new(a.owned_snapshot());
        let mine = current_epoch().await;
        let _b = WriterSession::open(store.clone(), paths.clone())
            .await
            .unwrap();
        let peer = current_epoch().await;
        assert!(peer > mine);
        let fenced = namidb_storage::Error::Fenced {
            mine,
            current: peer,
        };

        let health = WriterHealth::with_fence_policy(FencePolicy::Yield);
        recover_writer_if_needed(&mut a, &cell, &health, "fence-yield", &fenced).await;
        assert!(
            health.is_degraded(),
            "a yielded writer must report degraded"
        );
        assert_eq!(
            current_epoch().await,
            peer,
            "yield must not reclaim the epoch"
        );

        let health = WriterHealth::new();
        recover_writer_if_needed(&mut a, &cell, &health, "fence-yield", &fenced).await;
        assert!(!health.is_degraded(), "{:?}", health.degraded_reason());
        assert!(current_epoch().await > peer, "steal reopens and reclaims");
    }

    /// The reported steal-back: the peer commits, a straggling write on the
    /// yielded node is fenced and poisons the session, and the next flush
    /// tick fails with a plain poisoned-session refusal. That later error is
    /// not `Fenced`, so it must still not reopen.
    #[tokio::test]
    async fn yielded_writer_stays_down_after_its_session_is_poisoned() {
        use namidb_core::NamespaceId;
        use namidb_storage::{ManifestStore, NamespacePaths, NodeWriteRecord, WriterSession};
        use std::sync::Arc as StdArc;

        let store: StdArc<dyn object_store::ObjectStore> =
            StdArc::new(object_store::memory::InMemory::new());
        let paths = NamespacePaths::new("tenants", NamespaceId::new("fence-latch").unwrap());
        let ms = ManifestStore::new(store.clone(), paths.clone());
        let record = NodeWriteRecord {
            schema_version: 1,
            ..Default::default()
        };

        let mut a = WriterSession::open(store.clone(), paths.clone())
            .await
            .unwrap();
        let cell = SnapshotCell::new(a.owned_snapshot());
        let mut b = WriterSession::open(store.clone(), paths.clone())
            .await
            .unwrap();
        b.upsert_node("P", namidb_core::id::NodeId::new(), &record)
            .unwrap();
        b.commit_batch().await.unwrap();
        let owner_epoch = ms.load_current().await.unwrap().manifest.epoch;

        let health = WriterHealth::with_fence_policy(FencePolicy::Yield);
        for _ in 0..2 {
            a.upsert_node("P", namidb_core::id::NodeId::new(), &record)
                .unwrap();
            if let Err(e) = a.commit_batch().await {
                recover_writer_if_needed(&mut a, &cell, &health, "fence-latch", &e).await;
            }
            let schema = a.snapshot().manifest().manifest.schema.clone();
            if let Err(e) = a.flush(schema).await {
                recover_writer_if_needed(&mut a, &cell, &health, "fence-latch", &e).await;
            }
        }
        assert!(health.is_degraded());
        assert_eq!(
            ms.load_current().await.unwrap().manifest.epoch,
            owner_epoch,
            "a yielded writer must not fence the owner on a later failure"
        );
    }

    #[test]
    fn fence_policy_parses_both_spellings() {
        assert_eq!("steal".parse::<FencePolicy>(), Ok(FencePolicy::Steal));
        assert_eq!("Yield".parse::<FencePolicy>(), Ok(FencePolicy::Yield));
        assert!("maybe".parse::<FencePolicy>().is_err());
    }
}
