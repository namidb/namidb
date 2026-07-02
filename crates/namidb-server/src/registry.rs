//! Multi-tenant namespace registry.
//!
//! The storage layer already supports multiple namespaces over a single
//! `Arc<dyn ObjectStore>`. This registry extends that support to the HTTP
//! server, allowing a single process to serve N namespaces with in-process
//! routing instead of requiring N OS processes.
//!
//! Each namespace has its own `WriterSession` (single-writer-per-namespace
//! is preserved) and its own maintenance tasks (flush, compaction, orphan
//! sweep). The registry lazily creates sessions on first access and evicts
//! idle namespaces under a cap.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use axum::http::StatusCode;
use axum::response::Response;
use axum::{response::IntoResponse, Json};
use namidb_core::NamespaceId;
use namidb_query::StatsCatalog;
use namidb_storage::{
    sweep_orphans, Manifest, ManifestStore, NamespacePaths, SnapshotCell, WriterSession,
};
use object_store::ObjectStore;
use tokio::sync::{watch, Mutex, RwLock};
use tokio::task::JoinHandle;
use tracing::{error, info};

use crate::metrics::Metrics;
use crate::recovery::{self, WriterHealth};

/// `(manifest_version, catalog)` memoised behind a mutex and shared across
/// cloned [`NamespaceState`]s. `None` until the first read query builds it.
type CatalogCache = Arc<std::sync::Mutex<Option<(u64, Arc<StatsCatalog>)>>>;

/// Per-namespace background-maintenance configuration. Mirrors the
/// single-tenant `Config` fields that drive flush/compaction/orphan-sweep,
/// so a multi-tenant namespace gets the same durability guarantees as a
/// single-tenant process. Zero intervals disable the corresponding task.
#[derive(Clone, Copy)]
pub struct MaintenanceConfig {
    pub flush_interval: Duration,
    pub compaction_interval: Duration,
    pub sweep_min_age: Duration,
    pub sweep_delete: bool,
    /// L0-count high-water mark per bucket that triggers a reactive
    /// compaction on flush. `0` disables it.
    pub compaction_l0_trigger: usize,
}

impl Default for MaintenanceConfig {
    fn default() -> Self {
        Self {
            flush_interval: Duration::ZERO,
            compaction_interval: Duration::ZERO,
            sweep_min_age: Duration::ZERO,
            sweep_delete: false,
            compaction_l0_trigger: 0,
        }
    }
}

/// In-process registry of namespace sessions.
///
/// One instance lives at server top-level; all handlers reach it via
/// `State<NamespaceRegistry>`. A `get_or_open` call lazily creates a
/// `NamespaceState` with its own `WriterSession` (storage already supports
/// N over one store), returning a handle that keeps the session alive.
pub struct NamespaceRegistry {
    /// Shared object store (all namespaces multiplex this one store).
    store: Arc<dyn ObjectStore>,
    /// Root prefix for all namespaces (e.g. "tenants" or "" for flat layout).
    root: String,
    /// Active namespace sessions, keyed by namespace string.
    sessions: RwLock<HashMap<String, Arc<NamespaceState>>>,
    /// Maximum number of concurrent namespaces. `0` means unlimited (no cap,
    /// no eviction). Otherwise, when the cap is reached, idle sessions are
    /// evicted oldest-first.
    max_namespaces: usize,
    /// Idle eviction timeout: a namespace unused for this long is evicted
    /// (subject to the cap; eviction only happens when at capacity).
    idle_timeout: Duration,
    /// Monotonic anchor (registry construction). `last_access` stores
    /// `anchor.elapsed().as_secs()` so idle duration is
    /// `now_secs - last_secs` — a plain arithmetic diff, not the
    /// `Instant::now().elapsed()` near-zero value that previously broke
    /// eviction entirely.
    anchor: Instant,
    /// Process-wide metrics (flush, compaction, orphan-sweep increments).
    /// Held for the per-namespace maintenance tasks to increment; retained on
    /// the struct even when a build configuration doesn't read it directly.
    #[allow(dead_code)]
    metrics: Arc<Metrics>,
    /// Per-namespace background-maintenance schedule (flush/compaction/sweep).
    /// Without this, a multi-tenant namespace never flushed or compacted — a
    /// durability and read-amplification gap vs the single-tenant path.
    maintenance: MaintenanceConfig,
}

impl NamespaceRegistry {
    /// Create a new registry with the given store, root prefix, limits, and
    /// per-namespace maintenance schedule.
    pub fn new(
        store: Arc<dyn ObjectStore>,
        root: String,
        max_namespaces: usize,
        idle_timeout: Duration,
        metrics: Arc<Metrics>,
        maintenance: MaintenanceConfig,
    ) -> Self {
        Self {
            store,
            root,
            sessions: RwLock::new(HashMap::new()),
            max_namespaces,
            idle_timeout,
            anchor: Instant::now(),
            metrics,
            maintenance,
        }
    }

    /// Seconds elapsed since the registry's anchor — the clock `last_access`
    /// is measured in. Cheap, monotonic, and yields a correct idle diff.
    fn now_secs(&self) -> u64 {
        self.anchor.elapsed().as_secs()
    }

    /// `true` if the cap is configured and reached.
    fn at_capacity(&self, len: usize) -> bool {
        self.max_namespaces != 0 && len >= self.max_namespaces
    }

    /// Get or create a `NamespaceState` for `namespace`. Returns an error
    /// if the namespace ID is invalid.
    pub async fn get_or_open(&self, namespace: &str) -> Result<Arc<NamespaceState>, RegistryError> {
        let now = self.now_secs();
        // Fast path: read lock check
        {
            let sessions = self.sessions.read().await;
            if let Some(state) = sessions.get(namespace) {
                state
                    .last_access
                    .store(now, std::sync::atomic::Ordering::Relaxed);
                return Ok(Arc::clone(state));
            }
        }

        // Slow path: write lock, double-check, then create
        let mut sessions = self.sessions.write().await;
        if let Some(state) = sessions.get(namespace) {
            state
                .last_access
                .store(now, std::sync::atomic::Ordering::Relaxed);
            return Ok(Arc::clone(state));
        }

        // Evict if at capacity (0 = unlimited, never evicts).
        while self.at_capacity(sessions.len()) {
            if let Some(to_evict) = self.find_idle_oldest(&sessions) {
                tracing::info!("evicting idle namespace: {}", to_evict);
                if let Some(evicted) = sessions.remove(&to_evict) {
                    // Stop the namespace's flush/compaction loops so the
                    // evicted state (writer, memtable, caches) is actually
                    // released instead of living on as a zombie second
                    // writer. Dropping the memtable loses nothing: acked
                    // writes are WAL-committed before the ack, and reopen
                    // replays the WAL.
                    evicted.cancel_maintenance();
                }
            } else {
                return Err(RegistryError::AtCapacity);
            }
        }

        // Create new session
        let ns_id = NamespaceId::new(namespace)
            .map_err(|e| RegistryError::InvalidNamespace(e.to_string()))?;
        let paths = NamespacePaths::new(&self.root, ns_id);

        let writer = WriterSession::open(self.store.clone(), paths.clone())
            .await
            .map_err(|e| match e {
                // A momentarily-stale pointer resolution is retryable (503),
                // not a server bug (500). Classify it before flattening the
                // rest of the typed storage errors into OpenFailed.
                namidb_storage::Error::PointerResolveStale => {
                    RegistryError::Unavailable(e.to_string())
                }
                other => RegistryError::OpenFailed(other.to_string()),
            })?;

        // Create the snapshot from the writer's owned snapshot (required by
        // SnapshotCell::new).
        let snapshot = Arc::new(SnapshotCell::new(writer.owned_snapshot()));

        let state = Arc::new(NamespaceState {
            namespace: namespace.to_string(),
            writer: Arc::new(tokio::sync::Mutex::new(writer)),
            snapshot,
            last_access: std::sync::atomic::AtomicU64::new(now),
            catalog_cache: Arc::new(std::sync::Mutex::new(None)),
            cancel_tx: watch::channel(false).0,
            maintenance_tasks: std::sync::Mutex::new(Vec::new()),
            writer_health: WriterHealth::new(),
        });

        // Spawn per-namespace background maintenance (flush / compaction /
        // orphan sweep) so a multi-tenant namespace is as durable and
        // read-amplification-bounded as a single-tenant process. The tasks
        // hold their own Arc clones and run until eviction cancels them.
        self.spawn_maintenance(Arc::clone(&state), paths);

        sessions.insert(namespace.to_string(), Arc::clone(&state));
        tracing::info!(
            "opened namespace: {} (total: {})",
            namespace,
            sessions.len()
        );
        Ok(state)
    }

    /// Spawn the periodic flush and compaction+sweep tasks for one namespace,
    /// mirroring the single-tenant `run()` maintenance. Each task takes its
    /// own `Arc<NamespaceState>` clone and a per-namespace `ManifestStore`
    /// (for the lock-free orphan sweep). A zero interval disables that task.
    ///
    /// Both loops `select!` the state's cancellation signal against the tick,
    /// so eviction stops them promptly — even mid-sleep — while an operation
    /// already in flight always runs to completion (the signal is only
    /// observed between operations).
    fn spawn_maintenance(&self, state: Arc<NamespaceState>, paths: NamespacePaths) {
        let maint = self.maintenance;
        let maint_store = Arc::new(ManifestStore::new(self.store.clone(), paths));

        // Periodic flush (+ reactive compaction on L0 high-water).
        if maint.flush_interval > Duration::ZERO {
            let s = Arc::clone(&state);
            let mut cancel = state.cancel_tx.subscribe();
            let interval = maint.flush_interval;
            let l0_trigger = maint.compaction_l0_trigger;
            let ns = state.namespace.clone();
            let handle = tokio::spawn(async move {
                let mut tick = tokio::time::interval(interval);
                tick.tick().await; // first tick fires immediately; skip
                loop {
                    tokio::select! {
                        biased;
                        _ = cancel.wait_for(|evicted| *evicted) => break,
                        _ = tick.tick() => {}
                    }
                    let mut w = s.writer.lock().await;
                    let schema = w.snapshot().manifest().manifest.schema.clone();
                    match w.flush(schema.clone()).await {
                        Ok(_) => {
                            s.snapshot.store(w.owned_snapshot());
                            if l0_trigger > 0 && w.max_l0_bucket_len() >= l0_trigger {
                                if let Err(e) = w.compact_l0(&schema).await {
                                    error!(namespace = %ns, error = %e, "reactive compaction failed");
                                } else {
                                    s.snapshot.store(w.owned_snapshot());
                                }
                            }
                        }
                        Err(e) => {
                            error!(namespace = %ns, error = %e, "periodic flush failed");
                            // A fenced/poisoned writer would fail every later
                            // flush AND every write on this namespace; reopen
                            // it under the lock we already hold.
                            recovery::recover_writer_if_needed(
                                &mut w,
                                &s.snapshot,
                                &s.writer_health,
                                &ns,
                                &e,
                            )
                            .await;
                        }
                    }
                }
            });
            state
                .maintenance_tasks
                .lock()
                .expect("maintenance handles poisoned")
                .push(handle);
        }

        // Periodic compaction (L0->L1) + orphan sweep.
        if maint.compaction_interval > Duration::ZERO {
            let s = Arc::clone(&state);
            let mut cancel = state.cancel_tx.subscribe();
            let ms = Arc::clone(&maint_store);
            let interval = maint.compaction_interval;
            let sweep_min_age = maint.sweep_min_age;
            let sweep_delete = maint.sweep_delete;
            let ns = state.namespace.clone();
            let handle = tokio::spawn(async move {
                let mut tick = tokio::time::interval(interval);
                tick.tick().await; // first tick fires immediately; skip
                loop {
                    tokio::select! {
                        biased;
                        _ = cancel.wait_for(|evicted| *evicted) => break,
                        _ = tick.tick() => {}
                    }
                    {
                        let mut w = s.writer.lock().await;
                        let schema = w.snapshot().manifest().manifest.schema.clone();
                        match w.compact_l0(&schema).await {
                            Ok(outcome) if outcome.source_ssts_removed > 0 => {
                                s.snapshot.store(w.owned_snapshot());
                                info!(
                                    namespace = %ns,
                                    removed = outcome.source_ssts_removed,
                                    written = outcome.new_ssts_written,
                                    "compacted L0 into L1"
                                );
                            }
                            Ok(_) => {}
                            Err(e) => {
                                error!(namespace = %ns, error = %e, "periodic compaction failed")
                            }
                        }
                    }
                    // Orphan sweep — no writer lock; the retention horizon
                    // (RFC-027) keeps it from deleting a body a live reader
                    // still references.
                    let horizon = s.snapshot.retention_horizon();
                    if let Err(e) =
                        sweep_orphans(&ms, horizon, sweep_min_age, 1, sweep_delete).await
                    {
                        error!(namespace = %ns, error = %e, "orphan sweep failed");
                    }
                }
            });
            state
                .maintenance_tasks
                .lock()
                .expect("maintenance handles poisoned")
                .push(handle);
        }
    }

    /// Find the oldest idle namespace (unused for > idle_timeout). Returns
    /// `None` if no namespace is idle. Idle duration is the plain arithmetic
    /// diff `now_secs - last_access_secs` (both measured from the same anchor).
    fn find_idle_oldest(&self, sessions: &HashMap<String, Arc<NamespaceState>>) -> Option<String> {
        let now_secs = self.now_secs();
        let idle_timeout_secs = self.idle_timeout.as_secs();
        let mut oldest: Option<(&str, u64)> = None;

        for (ns, state) in sessions.iter() {
            let last_secs = state.last_access.load(std::sync::atomic::Ordering::Relaxed);
            let idle_secs = now_secs.saturating_sub(last_secs);
            if idle_secs > idle_timeout_secs && oldest.is_none_or(|(_, t)| idle_secs > t) {
                oldest = Some((ns.as_str(), idle_secs));
            }
        }
        oldest.map(|(ns, _)| ns.to_string())
    }

    /// Total number of active namespaces.
    pub async fn len(&self) -> usize {
        self.sessions.read().await.len()
    }

    /// `true` when no namespaces are active.
    pub async fn is_empty(&self) -> bool {
        self.sessions.read().await.is_empty()
    }
}

/// Per-namespace state: one `WriterSession` (single-writer-per-namespace)
/// and the read-side cache (`SnapshotCell`).
pub struct NamespaceState {
    /// Namespace identifier (human-readable, e.g. "acme").
    pub namespace: String,
    /// Single writer for this namespace (epoch-fenced, CAS-protected).
    pub writer: Arc<Mutex<WriterSession>>,
    /// Snapshot cache for read queries.
    pub snapshot: Arc<SnapshotCell>,
    /// Last access time (seconds since Unix epoch). Updated on every
    /// `get_or_open` hit by the registry.
    pub last_access: std::sync::atomic::AtomicU64,
    /// Memoised optimizer stats, keyed by manifest version. Building the
    /// catalog is `O(ssts)`; without this every multi-tenant read query
    /// rebuilt it from scratch.
    pub catalog_cache: CatalogCache,
    /// Flipped to `true` when the registry evicts this namespace. The
    /// maintenance loops `select!` over it so they exit promptly (even
    /// mid-sleep) and drop their `Arc<Self>` clones — without it an evicted
    /// state lived on as a zombie second writer with its memtable and caches.
    cancel_tx: watch::Sender<bool>,
    /// Handles of the spawned maintenance tasks, populated by
    /// `spawn_maintenance`, so an eviction observer can await task exit
    /// (each task finishes any in-flight flush/compaction first).
    maintenance_tasks: std::sync::Mutex<Vec<JoinHandle<()>>>,
    /// Writer status for this namespace's readiness probe: degraded from a
    /// terminal commit/flush failure until the automatic reopen succeeds.
    pub writer_health: Arc<WriterHealth>,
}

impl NamespaceState {
    /// Signal the maintenance tasks to stop. An in-flight flush or
    /// compaction runs to completion; the loops observe the signal between
    /// operations (and while sleeping) and then exit, releasing their
    /// references to this state.
    pub fn cancel_maintenance(&self) {
        let _ = self.cancel_tx.send(true);
    }

    /// Take the maintenance task handles (empty after the first call).
    /// They complete shortly after [`Self::cancel_maintenance`].
    pub fn take_maintenance_handles(&self) -> Vec<JoinHandle<()>> {
        std::mem::take(
            &mut self
                .maintenance_tasks
                .lock()
                .expect("maintenance handles poisoned"),
        )
    }

    /// Optimizer [`StatsCatalog`] for `manifest`, built once per manifest
    /// version and reused across queries until the next write bumps the
    /// version. Mirrors the single-tenant `AppState::catalog_for`.
    pub fn catalog_for(&self, manifest: &Manifest) -> Arc<StatsCatalog> {
        let version = manifest.version;
        let mut slot = self.catalog_cache.lock().expect("catalog cache poisoned");
        if let Some((cached_version, catalog)) = slot.as_ref() {
            if *cached_version == version {
                return Arc::clone(catalog);
            }
        }
        let catalog = Arc::new(StatsCatalog::from_manifest(manifest));
        *slot = Some((version, Arc::clone(&catalog)));
        catalog
    }
}

/// Errors from namespace registry operations.
#[derive(Debug, Clone)]
pub enum RegistryError {
    /// Namespace ID is invalid (e.g. contains a slash).
    InvalidNamespace(String),
    /// Failed to open a `WriterSession` for the namespace.
    OpenFailed(String),
    /// Registry is at capacity and no idle namespace to evict.
    AtCapacity,
    /// A transient condition (e.g. the pointer family is momentarily stale);
    /// the client should retry. Mapped to 503, distinct from `OpenFailed`'s
    /// 500 so a retryable race is not reported as a server bug.
    Unavailable(String),
}

impl std::fmt::Display for RegistryError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidNamespace(msg) => write!(f, "invalid namespace: {}", msg),
            Self::OpenFailed(msg) => write!(f, "failed to open namespace: {}", msg),
            Self::AtCapacity => write!(f, "namespace registry at capacity"),
            Self::Unavailable(msg) => write!(f, "namespace temporarily unavailable: {}", msg),
        }
    }
}

impl std::error::Error for RegistryError {}

impl IntoResponse for RegistryError {
    fn into_response(self) -> Response {
        let (status, message) = match self {
            Self::InvalidNamespace(msg) => (StatusCode::BAD_REQUEST, msg),
            Self::OpenFailed(msg) => (StatusCode::INTERNAL_SERVER_ERROR, msg),
            Self::AtCapacity => (
                StatusCode::SERVICE_UNAVAILABLE,
                "namespace registry at capacity".to_string(),
            ),
            Self::Unavailable(msg) => (StatusCode::SERVICE_UNAVAILABLE, msg),
        };
        (status, Json(serde_json::json!({ "error": message }))).into_response()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use namidb_query::{
        execute, execute_write, parse as cypher_parse, plan as build_plan, Params, RuntimeValue,
        StatsCatalog,
    };

    /// Registry over a fresh in-memory store with `max_namespaces = 1` and
    /// `idle_timeout = 0`: opening a second namespace evicts the first as
    /// soon as it is at least one second idle (the idle clock has 1s
    /// granularity).
    fn evicting_registry(uri_ns: &str, maint: MaintenanceConfig) -> NamespaceRegistry {
        let (store, _) = namidb_storage::parse_uri(&format!("memory://{uri_ns}")).unwrap();
        let metrics = Metrics::new(env!("CARGO_PKG_VERSION"), Duration::ZERO);
        NamespaceRegistry::new(store, String::new(), 1, Duration::ZERO, metrics, maint)
    }

    /// Open `ns`, retrying while the current occupant ages past the idle
    /// threshold (whole-second granularity means the first tries can hit
    /// `AtCapacity`).
    async fn open_evicting(reg: &NamespaceRegistry, ns: &str) -> Arc<NamespaceState> {
        let deadline = Instant::now() + Duration::from_secs(15);
        loop {
            match reg.get_or_open(ns).await {
                Ok(state) => return state,
                Err(RegistryError::AtCapacity) => {
                    assert!(
                        Instant::now() < deadline,
                        "the idle namespace was never evicted to make room for {ns}"
                    );
                    tokio::time::sleep(Duration::from_millis(100)).await;
                }
                Err(e) => panic!("open {ns}: {e}"),
            }
        }
    }

    async fn create_person(state: &NamespaceState, name: &str) {
        let q = format!("CREATE (:Person {{name: '{name}'}})");
        let parsed = cypher_parse(&q).expect("parse");
        let mut w = state.writer.lock().await;
        let catalog = StatsCatalog::from_manifest(&w.snapshot().manifest().manifest);
        let plan = build_plan(&parsed, &catalog).expect("plan");
        execute_write(&plan, &mut w, &Params::new())
            .await
            .expect("write");
        state.snapshot.store(w.owned_snapshot());
    }

    async fn count_persons(state: &NamespaceState) -> i64 {
        let parsed = cypher_parse("MATCH (p:Person) RETURN count(p) AS c").expect("parse");
        let snap = state.snapshot.load();
        let catalog = StatsCatalog::from_manifest(&snap.manifest().manifest);
        let plan = build_plan(&parsed, &catalog).expect("plan");
        let borrowed = snap.borrow();
        let rows = execute(&plan, &borrowed, &Params::new())
            .await
            .expect("read");
        match rows.first().and_then(|r| r.get("c")) {
            Some(RuntimeValue::Integer(n)) => *n,
            other => panic!("unexpected count row: {other:?}"),
        }
    }

    async fn join_with_timeout(handles: Vec<tokio::task::JoinHandle<()>>) {
        for h in handles {
            tokio::time::timeout(Duration::from_secs(5), h)
                .await
                .expect("maintenance task did not exit after eviction")
                .expect("maintenance task panicked");
        }
    }

    /// Eviction must stop BOTH maintenance loops (they used to run forever,
    /// pinning the state) and release every long-lived reference to the
    /// evicted `NamespaceState`.
    #[tokio::test]
    async fn eviction_stops_maintenance_and_releases_the_state() {
        let maint = MaintenanceConfig {
            flush_interval: Duration::from_millis(20),
            compaction_interval: Duration::from_millis(20),
            ..MaintenanceConfig::default()
        };
        let reg = evicting_registry("registry-evict-cancel", maint);
        let acme = reg.get_or_open("acme").await.expect("open acme");
        let handles = acme.take_maintenance_handles();
        assert_eq!(handles.len(), 2, "flush + compaction tasks spawn");

        let _beta = open_evicting(&reg, "beta").await;
        assert_eq!(reg.len().await, 1, "acme was evicted");

        // Without the cancel signal both loops spin forever and this joins
        // time out.
        join_with_timeout(handles).await;

        // With the tasks gone and the registry entry removed, the test's
        // clone is the only remaining reference: the writer, memtable, and
        // caches of the evicted state are released, not leaked.
        assert_eq!(
            Arc::strong_count(&acme),
            1,
            "evicted NamespaceState is still referenced somewhere"
        );
    }

    /// Acked writes survive evict + reopen. They are WAL-committed before
    /// the ack, so dropping the memtable on evict loses nothing: the long
    /// maintenance intervals here guarantee nothing was flushed to SSTs,
    /// and reopen recovers the writes purely from WAL replay.
    #[tokio::test]
    async fn evicted_namespace_retains_acked_writes_on_reopen() {
        let maint = MaintenanceConfig {
            flush_interval: Duration::from_secs(3600),
            compaction_interval: Duration::from_secs(3600),
            ..MaintenanceConfig::default()
        };
        let reg = evicting_registry("registry-evict-durability", maint);
        let acme = reg.get_or_open("acme").await.expect("open acme");
        for name in ["ada", "grace", "edsger"] {
            create_person(&acme, name).await;
        }
        let handles = acme.take_maintenance_handles();
        drop(acme);

        let _beta = open_evicting(&reg, "beta").await;
        join_with_timeout(handles).await;

        let reopened = open_evicting(&reg, "acme").await;
        assert_eq!(
            count_persons(&reopened).await,
            3,
            "acked writes were lost across evict/reopen"
        );
    }

    /// After evict + reopen the new incarnation is the only writer — the
    /// old maintenance tasks are gone, so its writes succeed with no
    /// fencing churn from a zombie sibling.
    #[tokio::test]
    async fn reopened_namespace_accepts_writes_after_evict() {
        let maint = MaintenanceConfig {
            flush_interval: Duration::from_millis(20),
            compaction_interval: Duration::from_millis(20),
            ..MaintenanceConfig::default()
        };
        let reg = evicting_registry("registry-evict-reopen", maint);
        let acme = reg.get_or_open("acme").await.expect("open acme");
        create_person(&acme, "before-evict").await;
        let handles = acme.take_maintenance_handles();
        drop(acme);

        let _beta = open_evicting(&reg, "beta").await;
        join_with_timeout(handles).await;

        let reopened = open_evicting(&reg, "acme").await;
        create_person(&reopened, "after-evict").await;
        assert_eq!(
            count_persons(&reopened).await,
            2,
            "the reopened namespace must see the pre-evict write and accept new ones"
        );
    }
}
