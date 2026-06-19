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

use axum::{Json, response::IntoResponse};
use axum::response::Response;
use axum::http::StatusCode;
use namidb_core::NamespaceId;
use namidb_storage::{NamespacePaths, SnapshotCell, WriterSession};
use object_store::ObjectStore;
use tokio::sync::{Mutex, RwLock};

use crate::metrics::Metrics;

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
    metrics: Arc<Metrics>,
}

impl NamespaceRegistry {
    /// Create a new registry with the given store, root prefix, and limits.
    pub fn new(
        store: Arc<dyn ObjectStore>,
        root: String,
        max_namespaces: usize,
        idle_timeout: Duration,
        metrics: Arc<Metrics>,
    ) -> Self {
        Self {
            store,
            root,
            sessions: RwLock::new(HashMap::new()),
            max_namespaces,
            idle_timeout,
            anchor: Instant::now(),
            metrics,
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
                sessions.remove(&to_evict);
            } else {
                return Err(RegistryError::AtCapacity);
            }
        }

        // Create new session
        let ns_id = NamespaceId::new(namespace).map_err(|e| RegistryError::InvalidNamespace(e.to_string()))?;
        let paths = NamespacePaths::new(&self.root, ns_id);

        let writer = WriterSession::open(self.store.clone(), paths.clone())
            .await
            .map_err(|e| RegistryError::OpenFailed(e.to_string()))?;

        // Create the snapshot from the writer's owned snapshot (required by
        // SnapshotCell::new).
        let snapshot = Arc::new(SnapshotCell::new(writer.owned_snapshot()));

        let state = Arc::new(NamespaceState {
            namespace: namespace.to_string(),
            writer: Arc::new(tokio::sync::Mutex::new(writer)),
            snapshot,
            last_access: std::sync::atomic::AtomicU64::new(now),
        });

        sessions.insert(namespace.to_string(), Arc::clone(&state));
        tracing::info!("opened namespace: {} (total: {})", namespace, sessions.len());
        Ok(state)
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
            if idle_secs > idle_timeout_secs && oldest.map_or(true, |(_, t)| idle_secs > t) {
                oldest = Some((ns.as_str(), idle_secs));
            }
        }
        oldest.map(|(ns, _)| ns.to_string())
    }

    /// Total number of active namespaces.
    pub async fn len(&self) -> usize {
        self.sessions.read().await.len()
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
}

impl std::fmt::Display for RegistryError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidNamespace(msg) => write!(f, "invalid namespace: {}", msg),
            Self::OpenFailed(msg) => write!(f, "failed to open namespace: {}", msg),
            Self::AtCapacity => write!(f, "namespace registry at capacity"),
        }
    }
}

impl std::error::Error for RegistryError {}

impl IntoResponse for RegistryError {
    fn into_response(self) -> Response {
        let (status, message) = match self {
            Self::InvalidNamespace(msg) => (StatusCode::BAD_REQUEST, msg),
            Self::OpenFailed(msg) => (StatusCode::INTERNAL_SERVER_ERROR, msg),
            Self::AtCapacity => (StatusCode::SERVICE_UNAVAILABLE, "namespace registry at capacity".to_string()),
        };
        (status, Json(serde_json::json!({ "error": message }))).into_response()
    }
}
