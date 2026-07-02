//! Persistent retention pins: lease objects that hold the orphan sweep's
//! horizon at (or below) a manifest version across processes.
//!
//! The in-process retention horizon (RFC-027, `SnapshotCell`) only covers
//! readers inside the server. A backup is a long-running *external* reader of
//! the pinned manifest's closure, so it needs a pin the janitor can see from
//! any process: a small JSON lease under `manifest/pins/<uuid>.json` naming
//! `{version, expires_at_unix}`. [`crate::janitor::sweep_orphans`] lists the
//! prefix before deleting anything and unions every unexpired lease into its
//! horizon; an expired lease is ignored and reclaimed by the sweep, so a
//! crashed holder can never pin the namespace forever.
//!
//! The lease is time-bound on purpose: the holder renews it periodically
//! (see [`RetentionPin::renew_if_due`]) and deletes it when done. Expiry is
//! wall-clock (`expires_at_unix`), which assumes loosely synchronised clocks
//! between the holder and the janitor — the generous default TTL plus
//! half-TTL renewal leaves minutes of skew budget.

use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use object_store::path::Path;
use object_store::{ObjectStore, ObjectStoreExt, PutPayload};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::error::{Error, Result};
use crate::paths::NamespacePaths;

/// Default lease TTL for a retention pin. Long enough that a renewal cadence
/// of one check per copied object keeps a healthy backup pinned even while a
/// single multi-GB object streams; short enough that a crashed backup frees
/// the namespace within minutes rather than wedging retention forever.
pub const DEFAULT_PIN_TTL: Duration = Duration::from_secs(15 * 60);

/// On-disk body of a retention pin lease (`manifest/pins/<uuid>.json`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PinLease {
    /// Manifest version the holder is reading. The sweep keeps every object
    /// referenced by any manifest version at or above the smallest pinned
    /// version, exactly as it does for in-process reader horizons.
    pub version: u64,
    /// Unix seconds after which the lease is void. The sweep ignores and
    /// deletes leases past this instant.
    pub expires_at_unix: i64,
}

/// A live retention pin: a lease object this handle wrote and is responsible
/// for renewing and releasing. Dropping the handle without
/// [`RetentionPin::release`] leaks the lease until it expires — callers
/// should release explicitly on both success and error paths.
#[derive(Debug)]
pub struct RetentionPin {
    store: Arc<dyn ObjectStore>,
    path: Path,
    version: u64,
    ttl: Duration,
    /// When the next [`RetentionPin::renew_if_due`] should actually rewrite
    /// the lease: half the TTL after the last write, so a healthy holder
    /// always renews long before expiry.
    renew_due_at: Instant,
}

impl RetentionPin {
    /// Write a fresh lease pinning `version` and return the handle. The pin
    /// only protects objects that still exist once the lease is visible;
    /// callers must re-check their pinned root (the manifest body) *after*
    /// acquiring, to close the load-then-pin race against an in-flight sweep.
    pub async fn acquire(
        store: Arc<dyn ObjectStore>,
        paths: &NamespacePaths,
        version: u64,
        ttl: Duration,
    ) -> Result<Self> {
        let path = paths.pin_object(&Uuid::new_v4().to_string());
        let mut pin = Self {
            store,
            path,
            version,
            ttl,
            renew_due_at: Instant::now(),
        };
        pin.write_lease().await?;
        Ok(pin)
    }

    pub fn version(&self) -> u64 {
        self.version
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Rewrite the lease with a fresh expiry when at least half the TTL has
    /// elapsed since the last write. Cheap enough to call once per copied
    /// object; a no-op most of the time.
    pub async fn renew_if_due(&mut self) -> Result<()> {
        if Instant::now() >= self.renew_due_at {
            self.write_lease().await?;
        }
        Ok(())
    }

    /// Delete the lease. Failing here is benign — the lease expires on its
    /// own — but callers should log it.
    pub async fn release(self) -> Result<()> {
        match self.store.delete(&self.path).await {
            Ok(()) | Err(object_store::Error::NotFound { .. }) => Ok(()),
            Err(e) => Err(Error::ObjectStore(e)),
        }
    }

    async fn write_lease(&mut self) -> Result<()> {
        let lease = PinLease {
            version: self.version,
            expires_at_unix: unix_now().saturating_add(self.ttl.as_secs() as i64),
        };
        self.store
            .put(&self.path, PutPayload::from(serde_json::to_vec(&lease)?))
            .await
            .map_err(Error::ObjectStore)?;
        self.renew_due_at = Instant::now() + self.ttl / 2;
        Ok(())
    }
}

/// Current wall-clock time as Unix seconds (the lease expiry timebase).
pub(crate) fn unix_now() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use namidb_core::NamespaceId;
    use object_store::memory::InMemory;

    use super::*;

    fn paths() -> NamespacePaths {
        NamespacePaths::new("tenants", NamespaceId::new("pin-unit").unwrap())
    }

    #[tokio::test]
    async fn acquire_writes_a_lease_and_release_removes_it() {
        let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let paths = paths();

        let pin = RetentionPin::acquire(store.clone(), &paths, 7, Duration::from_secs(60))
            .await
            .unwrap();
        let body = store.get(pin.path()).await.unwrap().bytes().await.unwrap();
        let lease: PinLease = serde_json::from_slice(&body).unwrap();
        assert_eq!(lease.version, 7);
        assert!(
            lease.expires_at_unix > unix_now(),
            "a fresh lease must not be born expired"
        );
        assert!(
            pin.path().as_ref().starts_with(paths.pins_dir().as_ref()),
            "lease must live under the pins prefix: {}",
            pin.path()
        );

        let lease_path = pin.path().clone();
        pin.release().await.unwrap();
        assert!(
            store.head(&lease_path).await.is_err(),
            "release must delete the lease object"
        );
    }

    #[tokio::test]
    async fn renew_if_due_extends_the_expiry_when_the_deadline_passed() {
        let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let paths = paths();

        let mut pin = RetentionPin::acquire(store.clone(), &paths, 3, Duration::from_secs(60))
            .await
            .unwrap();
        let before: PinLease =
            serde_json::from_slice(&store.get(pin.path()).await.unwrap().bytes().await.unwrap())
                .unwrap();

        // Not due yet (the deadline is half the TTL away): no rewrite.
        pin.renew_if_due().await.unwrap();
        let unchanged: PinLease =
            serde_json::from_slice(&store.get(pin.path()).await.unwrap().bytes().await.unwrap())
                .unwrap();
        assert_eq!(unchanged.expires_at_unix, before.expires_at_unix);

        // Force the deadline into the present and grow the TTL, so the
        // renewal both fires and observably pushes the expiry forward.
        pin.renew_due_at = Instant::now();
        pin.ttl = Duration::from_secs(3600);
        pin.renew_if_due().await.unwrap();
        let after: PinLease =
            serde_json::from_slice(&store.get(pin.path()).await.unwrap().bytes().await.unwrap())
                .unwrap();
        assert_eq!(after.version, 3);
        assert!(
            after.expires_at_unix > before.expires_at_unix,
            "renewal must push the expiry forward"
        );
    }
}
