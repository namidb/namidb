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
//!
//! A lease also names its [`PinScope`]: what the sweep keeps for `version`.
//! The original (and default) scope is a *floor* — every manifest version from
//! `version` to current stays loadable, exactly like an in-process reader
//! horizon. A *point* lease keeps one version's closure (a backup of an old
//! restore point needs nothing newer, and a floor would make every sweep load
//! every manifest between it and current). A *window* lease is the restore
//! point window ([`crate::restore_point`]). Old janitors ignore the scope and
//! honour every lease as a floor, which keeps strictly more, so mixing
//! releases is safe; a scope this build does not recognise is read as a floor
//! for the same reason.

use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use chrono::{DateTime, Utc};
use object_store::path::Path;
use object_store::{ObjectStore, ObjectStoreExt, PutPayload};
use serde::{Deserialize, Deserializer, Serialize};
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
    /// What the sweep keeps for `version`. Absent on leases written before
    /// scopes existed, which are floors.
    #[serde(
        default,
        skip_serializing_if = "PinScope::is_floor",
        deserialize_with = "scope_or_floor"
    )]
    pub scope: PinScope,
}

/// What a [`PinLease`] holds alive (see the module docs).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum PinScope {
    /// Every manifest version from the lease's version to current, and every
    /// object they reference. The sweep lowers its horizon to the version.
    #[default]
    Floor,
    /// Exactly the lease's version: its manifest body and every object it
    /// references. Newer versions are left to the other horizons.
    Point,
    /// The restore point window ([`crate::restore_point`]): every manifest
    /// body from the lease's version up stays, the version's own closure
    /// stays, and so does every SST, side-car or WAL object written since
    /// `since` minus the sweep's `min_age`. `since` is the floor version's
    /// `created_at`. Keeping by write time is what lets the sweep honour a
    /// window of thousands of versions after loading one extra manifest:
    /// every object a version in the window references either belongs to the
    /// floor version or was uploaded after it (less the upload-to-commit
    /// window `min_age` already bounds).
    Window { since: DateTime<Utc> },
}

impl PinScope {
    pub fn is_floor(&self) -> bool {
        matches!(self, PinScope::Floor)
    }
}

/// Decode a lease scope, reading anything unrecognised (a scope from a newer
/// release) as [`PinScope::Floor`], the scope that keeps the most.
fn scope_or_floor<'de, D: Deserializer<'de>>(
    deserializer: D,
) -> std::result::Result<PinScope, D::Error> {
    let raw = serde_json::Value::deserialize(deserializer)?;
    Ok(serde_json::from_value(raw).unwrap_or(PinScope::Floor))
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
    scope: PinScope,
    ttl: Duration,
    /// The `expires_at_unix` of the last lease written. Renewal and lapse
    /// are judged against it on the wall clock, the one the sweep uses: a
    /// monotonic clock stops while the host sleeps, and a lease written
    /// before a long suspend would still look fresh after it.
    pub(crate) expires_at_unix: i64,
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
        Self::acquire_scoped(store, paths, version, PinScope::Floor, ttl).await
    }

    /// [`Self::acquire`] with an explicit [`PinScope`]. A backup pins a
    /// [`PinScope::Point`]: it reads one version's closure and nothing newer.
    pub async fn acquire_scoped(
        store: Arc<dyn ObjectStore>,
        paths: &NamespacePaths,
        version: u64,
        scope: PinScope,
        ttl: Duration,
    ) -> Result<Self> {
        let path = paths.pin_object(&Uuid::new_v4().to_string());
        let mut pin = Self {
            store,
            path,
            version,
            scope,
            ttl,
            expires_at_unix: 0,
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
        if unix_now() >= self.expires_at_unix - self.ttl_secs() / 2 {
            self.write_lease().await?;
        }
        Ok(())
    }

    /// Whether the lease may already have expired: the written expiry is
    /// less than a quarter of the TTL away, the clock skew budget. A sweep
    /// may have ignored a lapsed lease, so renewing it does not show that
    /// the pinned objects still exist.
    pub fn lapsed(&self) -> bool {
        unix_now() >= self.expires_at_unix - self.ttl_secs() / 4
    }

    fn ttl_secs(&self) -> i64 {
        i64::try_from(self.ttl.as_secs()).unwrap_or(i64::MAX)
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
            expires_at_unix: unix_now().saturating_add(self.ttl_secs()),
            scope: self.scope,
        };
        self.store
            .put(&self.path, PutPayload::from(serde_json::to_vec(&lease)?))
            .await
            .map_err(Error::ObjectStore)?;
        self.expires_at_unix = lease.expires_at_unix;
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

        // Bring the expiry within half a TTL and grow the TTL, so the
        // renewal both fires and observably pushes the expiry forward.
        pin.expires_at_unix = unix_now();
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

    #[test]
    fn scopes_round_trip_and_unknown_or_missing_scopes_read_as_floors() {
        // A lease written before scopes existed is a floor.
        let legacy: PinLease =
            serde_json::from_str(r#"{"version":4,"expires_at_unix":10}"#).unwrap();
        assert_eq!(legacy.scope, PinScope::Floor);
        // A floor is written without the field, byte-identical to before.
        let floor = serde_json::to_string(&legacy).unwrap();
        assert!(!floor.contains("scope"), "{floor}");

        for scope in [
            PinScope::Point,
            PinScope::Window {
                since: chrono::DateTime::from_timestamp(1_700_000_000, 0).unwrap(),
            },
        ] {
            let lease = PinLease {
                version: 9,
                expires_at_unix: 11,
                scope,
            };
            let back: PinLease =
                serde_json::from_slice(&serde_json::to_vec(&lease).unwrap()).unwrap();
            assert_eq!(back.scope, scope);
        }

        // A scope from a newer release keeps the most: a floor.
        let future: PinLease = serde_json::from_str(
            r#"{"version":4,"expires_at_unix":10,"scope":{"kind":"something_new","x":1}}"#,
        )
        .unwrap();
        assert_eq!(future.scope, PinScope::Floor);

        // An older reader that knows only {version, expires_at_unix} still
        // decodes a scoped lease, and honours it as a floor.
        #[derive(serde::Deserialize)]
        struct OldLease {
            version: u64,
            expires_at_unix: i64,
        }
        let window = PinLease {
            version: 3,
            expires_at_unix: 99,
            scope: PinScope::Window {
                since: chrono::DateTime::from_timestamp(1_700_000_000, 0).unwrap(),
            },
        };
        let old: OldLease = serde_json::from_slice(&serde_json::to_vec(&window).unwrap()).unwrap();
        assert_eq!((old.version, old.expires_at_unix), (3, 99));
    }

    #[tokio::test]
    async fn lapsed_follows_the_written_expiry_on_the_wall_clock() {
        let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let mut pin = RetentionPin::acquire(store, &paths(), 1, Duration::from_secs(60))
            .await
            .unwrap();
        assert!(!pin.lapsed());
        // The host slept past the expiry: no monotonic time passed, but the
        // sweep's clock did.
        pin.expires_at_unix = unix_now() - 1;
        assert!(pin.lapsed());
    }
}
