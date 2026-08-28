//! HTTP server exposing a NamiDB namespace.
//!
//! The crate is split between a thin [`main`] CLI parser and this
//! library so integration tests can exercise the routes directly.
//!
//! See [`build_router`] for the full route surface and [`run`] for
//! the end-to-end boot procedure.

pub mod auth;
pub mod authz;
pub mod bolt;
mod introspect;
mod maintenance;
pub mod memory;
pub mod metrics;
pub mod recovery;
pub mod registry;
pub mod shared;
pub mod tls;
// OIDC/JWT bearer-token validation (RFC-015 Wave A). Optional: only compiled
// with the `jwt` Cargo feature, which adds reqwest + jsonwebtoken.
#[cfg(feature = "jwt")]
pub mod jwt;
// External policy decision point (RFC-015 Wave B). Optional: only compiled with
// the `pdp` Cargo feature (adds reqwest). An OPA-backed `AuthzHook`.
#[cfg(feature = "pdp")]
pub mod pdp;

use std::sync::Arc;
use std::time::Duration;

use axum::body::HttpBody as _;
use axum::extract::{DefaultBodyLimit, Extension, Path, State};
use axum::http::{HeaderValue, StatusCode};
use axum::middleware::{self, Next};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::{Deserialize, Serialize};
use tokio::net::TcpListener;
use tokio::sync::Mutex;
use tracing::{error, info, warn};

use namidb_query::{
    execute_with_limits, execute_write_with_deadline, parse as cypher_parse, plan as build_plan,
    Params, RuntimeValue, StatsCatalog, WriteOutcome,
};
use namidb_storage::{sweep_orphans, Manifest, ManifestStore, SnapshotCell, WriterSession};

use crate::auth::{AuthConfig, Principal};
use crate::maintenance::{request_compaction, CompactionScheduler};
use crate::metrics::{CompactionTrigger, Metrics, Protocol, QueryKind, WriterLockKind};
use crate::recovery::WriterHealth;
use crate::registry::{NamespaceRegistry, NamespaceState};
use crate::shared::SharedAppState;

/// Explicit limit for the JSON body accepted by `/v0/cypher`.
///
/// Axum's `Json` extractor currently defaults to the same two MiB, but wiring
/// the value into both the body-limit layer and pre-decode memory reservation
/// prevents those independently evolving into contradictory bounds.
const HTTP_CYPHER_BODY_LIMIT_BYTES: usize = 2 * 1024 * 1024;
/// Conservative serde_json + RuntimeValue + execution amplification charged
/// before the JSON extractor allocates. This mirrors Bolt's measured 16x
/// decode working-set rail.
const HTTP_MEMORY_BYTES_PER_WIRE_BYTE: usize = 16;
const HTTP_MEMORY_BASE_BYTES: usize = 64 * 1024;

fn estimated_http_request_memory_bytes(wire_bytes: usize) -> usize {
    HTTP_MEMORY_BASE_BYTES
        .saturating_add(wire_bytes.saturating_mul(HTTP_MEMORY_BYTES_PER_WIRE_BYTE))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum HttpBodyAdmissionError {
    InvalidContentLength,
    TooLarge { observed: u64 },
}

/// Conservative wire-size bound available before Axum materialises JSON.
///
/// A valid Content-Length is exact. Without it, a body implementation may
/// still expose an exact/upper size hint (ordinary small test/client bodies
/// do); genuinely chunked/unknown bodies reserve the same real two-MiB limit
/// the extractor enforces instead of being rejected merely for lacking the
/// header.
fn http_cypher_wire_bytes(
    request: &axum::extract::Request,
) -> Result<usize, HttpBodyAdmissionError> {
    let content_length = request
        .headers()
        .get(axum::http::header::CONTENT_LENGTH)
        .map(|value| {
            value
                .to_str()
                .ok()
                .and_then(|raw| raw.parse::<u64>().ok())
                .ok_or(HttpBodyAdmissionError::InvalidContentLength)
        })
        .transpose()?;
    let observed = content_length
        .or_else(|| request.body().size_hint().upper())
        .unwrap_or(HTTP_CYPHER_BODY_LIMIT_BYTES as u64);
    if observed > HTTP_CYPHER_BODY_LIMIT_BYTES as u64 {
        return Err(HttpBodyAdmissionError::TooLarge { observed });
    }
    usize::try_from(observed).map_err(|_| HttpBodyAdmissionError::TooLarge { observed })
}

/// Process-wide configuration assembled from CLI flags or env vars.
#[derive(Debug, Clone)]
pub struct Config {
    pub store_uri: String,
    pub listen: std::net::SocketAddr,
    /// Bearer token granting read-write access. Production callers should
    /// set a long random secret. For read-only tokens or several tokens, use
    /// `auth_tokens_file`. When no auth source is configured at all, the
    /// server refuses to boot unless `no_auth` is set.
    pub auth_token: Option<String>,
    /// Path to a JSON file of tokens, each with a `read-only` or
    /// `read-write` role. Takes precedence over `auth_token` when set. `None`
    /// falls back to `auth_token`.
    pub auth_tokens_file: Option<std::path::PathBuf>,
    /// Explicit opt-in to run without any authentication (every request is
    /// anonymous read-write). Without this, a boot with no auth source
    /// configured fails instead of silently serving open — an unset
    /// `NAMIDB_AUTH_TOKEN` in production should be a crash, not a log line.
    pub no_auth: bool,
    /// OIDC/JWT validation config. `None` = JWT auth disabled (static tokens
    /// or open mode). Only present under the `jwt` feature.
    #[cfg(feature = "jwt")]
    pub jwt: Option<crate::jwt::JwtConfig>,
    /// External policy-decision-point URL (OPA-style). `None` = no PDP
    /// (allow-all NoOp). Only present under the `pdp` feature.
    #[cfg(feature = "pdp")]
    pub pdp_url: Option<String>,
    pub flush_interval: Duration,
    /// Interval for the background maintenance task (L0->L1 compaction +
    /// orphan sweep). `Duration::ZERO` disables it.
    pub compaction_interval: Duration,
    /// Minimum age before the orphan sweep may delete an unreachable immutable
    /// object. Live readers are protected independently by the exact retention
    /// horizon; this age guards the upload-before-manifest/pointer-CAS window.
    pub sweep_min_age: Duration,
    /// When `true` (the default) the orphan sweep deletes unreachable
    /// immutable bodies, WALs, manifests, and pointers; the retention horizon
    /// (RFC-027) makes that safe by construction. Set `false` for a dry-run
    /// that only logs what it would free.
    pub sweep_delete: bool,
    /// Bolt listener address. `None` keeps the protocol off (HTTP only).
    pub bolt_listen: Option<std::net::SocketAddr>,
    /// Maximum body size for one authenticated Bolt message. The
    /// unauthenticated handshake/LOGON path always keeps its strict 64 KiB
    /// ceiling. Must be non-zero.
    pub bolt_max_message_bytes: usize,
    /// Idle timeout for an open Bolt explicit transaction. While a
    /// transaction is open the writer lock is held, so an idle client would
    /// pin it; after this long without a message the transaction is rolled
    /// back and failed. `Duration::ZERO` disables the timeout.
    pub bolt_tx_timeout: Duration,
    /// Wall-clock deadline for a single read query (HTTP and Bolt, including
    /// in-transaction reads). A runaway scan or expansion is aborted with a
    /// timeout error rather than pinning a worker. `Duration::ZERO` disables
    /// it.
    pub query_timeout: Duration,
    /// Wall-clock deadline for a single write query: an HTTP / Bolt
    /// auto-commit statement, or each statement of a Bolt explicit
    /// transaction. A runaway MERGE/DELETE is aborted cooperatively rather
    /// than pinning the single writer, and its pending batch is discarded so
    /// nothing partial is committed. `Duration::ZERO` disables it; the CLI
    /// defaults it to `query_timeout`.
    pub write_timeout: Duration,
    /// Maximum rows a single read-query operator may materialise. A query
    /// whose operator output would exceed this aborts with a row-cap error
    /// instead of risking an out-of-memory blow-up (e.g. a cross product).
    /// `0` disables it.
    pub query_row_cap: usize,
    /// L0-count high-water mark per bucket that triggers a compaction as
    /// soon as a flush crosses it, instead of waiting for the periodic
    /// compaction tick (RFC-027 P5). Keeps read amplification bounded under
    /// sustained writes. `0` disables the reactive trigger.
    pub compaction_l0_trigger: usize,
    /// L0-count per bucket above which writes are softly stalled by
    /// `write_stall_delay` (RFC-027 P5), so the writer cannot outrun
    /// compaction without bound. `0` disables the stall.
    pub write_stall_l0: usize,
    /// Delay applied to a write when L0 is above `write_stall_l0`.
    pub write_stall_delay: Duration,
    /// Memtable byte size at which a committed write triggers a flush
    /// immediately (instead of waiting for `flush_interval`). Bounds the
    /// un-flushed working set by bytes, not just wall clock — a burst loader
    /// at hundreds of MB/s can otherwise accept gigabytes between ticks and
    /// OOM when the flush's CPU phase (~2-3x amplification) runs. `0`
    /// disables the trigger.
    pub memtable_flush_bytes: usize,
    /// Memtable byte size above which writes are softly stalled
    /// (backpressure) until the flush catches up. `0` disables the stall.
    pub memtable_stall_bytes: usize,
    /// Bound on how long a foreground auto-commit write (HTTP `/v0/cypher`,
    /// Bolt auto-commit) or a Bolt `BEGIN` may wait to acquire the writer
    /// mutex before failing fast with 503 / a transient Bolt error — so
    /// request queues stay bounded behind a stuck or long-held writer.
    /// Infrequent DDL and admin flush keep the unbounded wait; background
    /// flush/compaction/recovery always wait as long as it takes.
    /// `Duration::ZERO` disables the bound.
    pub writer_lock_timeout: Duration,
    /// PEM certificate-chain file enabling TLS on the HTTP and Bolt
    /// listeners. Must be set together with `tls_key`; when both are `None`
    /// the server serves plaintext.
    pub tls_cert: Option<std::path::PathBuf>,
    /// PEM private-key file paired with `tls_cert`.
    pub tls_key: Option<std::path::PathBuf>,
    /// Wall-clock at or above which a query is logged at `warn!` as a slow
    /// query (the statement text, never its parameters). The Prometheus
    /// counters and latency histograms at `/v0/metrics` are always on
    /// regardless of this. `Duration::ZERO` disables the slow-query log.
    pub slow_query_threshold: Duration,
    /// Multi-tenant mode: when `true`, the server accepts a namespace via
    /// path parameter (`/:namespace/v0/...`) or header (`X-NamiDB-Namespace`)
    /// and routes to a per-namespace `WriterSession`. When `false`, the server
    /// serves a single namespace (backward-compatible mode).
    pub multi_tenant: bool,
    /// Default namespace for backward compatibility. When `multi_tenant` is
    /// `false`, this namespace is opened at boot and all requests go to it.
    /// When `multi_tenant` is `true`, this is the fallback when no namespace
    /// is specified.
    pub default_namespace: String,
    /// Maximum number of concurrent namespaces in multi-tenant mode. When
    /// the cap is reached, idle namespaces are evicted oldest-first.
    /// `0` means unlimited (use with caution).
    pub max_namespaces: usize,
    /// Idle eviction timeout for namespaces in multi-tenant mode. A namespace
    /// unused for this long is eligible for eviction when at capacity.
    pub namespace_idle_timeout: Duration,
}

/// `(manifest_version, catalog)` memoised behind a mutex and shared across
/// cloned [`AppState`]s. `None` until the first read query builds it.
type CatalogCache = Arc<std::sync::Mutex<Option<(u64, Arc<StatsCatalog>)>>>;

/// Shared application state — one `WriterSession` (single-writer
/// invariant) plus the auth token reference and a [`SnapshotCell`]
/// readers consume to serve reads in parallel without taking the
/// writer mutex. See RFC-021.
#[derive(Clone)]
pub struct AppState {
    pub writer: Arc<Mutex<WriterSession>>,
    pub snapshot: Arc<SnapshotCell>,
    /// Single-flight background compaction scheduler. It admits at most one
    /// worker plus one basis-fresh follow-up without creating a FIFO task per
    /// flush trigger.
    pub(crate) compaction_scheduler: Arc<CompactionScheduler>,
    /// Memoised optimizer stats, keyed by manifest version. Building the
    /// catalog is `O(ssts)`; without this every read query rebuilt it from
    /// scratch. Shared across cloned `AppState`s (the router clones it per
    /// request) via the inner `Arc`, so all handlers hit one cache.
    catalog_cache: CatalogCache,
    /// Accepted bearer tokens and their roles. Empty = open (no auth). Shared
    /// with the Bolt serving path so a read-only token cannot write over
    /// either protocol.
    auth: Arc<AuthConfig>,
    namespace: String,
    /// Per-read-query wall-clock budget. `Duration::ZERO` disables it.
    /// Defaults to disabled; the server sets it from [`Config`] at boot.
    query_timeout: Duration,
    /// Per-write-query wall-clock budget. `Duration::ZERO` disables it.
    /// Defaults to disabled; the server sets it from [`Config`] at boot.
    write_timeout: Duration,
    /// Per-read-query operator row cap. `0` disables it. Defaults to
    /// disabled; the server sets it from [`Config`] at boot.
    query_row_cap: usize,
    /// Soft write-stall threshold and delay (RFC-027 P5). When the worst
    /// bucket's L0 count reaches `write_stall_l0` (and it is non-zero), a
    /// committed write waits `write_stall_delay` before returning, applying
    /// backpressure so the writer cannot outrun compaction. Defaults to
    /// disabled; the server sets them from [`Config`] at boot.
    write_stall_l0: usize,
    write_stall_delay: Duration,
    /// Byte thresholds for memtable-driven flushing: at
    /// `memtable_flush_bytes` a committed write nudges the flush task
    /// (`flush_notify`) instead of waiting out the timer; at
    /// `memtable_stall_bytes` the write is additionally stalled, so a burst
    /// loader cannot grow the memtable to OOM between ticks (the flush's CPU
    /// phase amplifies RAM ~2-3x, so wall-clock alone is not a bound). `0`
    /// disables each. The server sets them from [`Config`] at boot.
    memtable_flush_bytes: usize,
    memtable_stall_bytes: usize,
    /// Process-wide RSS/working-set admission shared by HTTP, Bolt, and every
    /// namespace. Unlike the storage cache budget, this observes total
    /// resident memory.
    pub(crate) memory: Arc<memory::MemoryGovernor>,
    /// Foreground writer-lock acquisition bound (see `Config`). `ZERO`
    /// disables it.
    writer_lock_timeout: Duration,
    /// Wakes the periodic flush task early when the byte threshold is
    /// crossed by a committed write.
    pub flush_notify: Arc<tokio::sync::Notify>,
    /// Last observed memtable size, published by the write/flush paths so
    /// `/v0/health` can report it without touching the writer lock.
    pub memtable_bytes_gauge: Arc<std::sync::atomic::AtomicUsize>,
    /// Process-wide query metrics, shared across every connection on both
    /// serving paths. Rendered at `/v0/metrics` and the home of the
    /// slow-query log. Defaults to a registry with the slow-query log
    /// disabled; the server sets the threshold from [`Config`] at boot.
    pub metrics: Arc<Metrics>,
    /// Pre-execution authorization hook (RFC-015 Wave B). Defaults to
    /// [`authz::NoOpAuthz`] (allow-all), so the gate is behavior-preserving
    /// until a real policy is configured.
    authz: Arc<dyn authz::AuthzHook>,
    /// Writer status for the readiness probe: degraded from a terminal
    /// commit/flush failure (fenced / lost CAS / poisoned) until the
    /// automatic reopen ([`recovery`]) succeeds. Read lock-free by
    /// `/v0/health`.
    pub writer_health: Arc<WriterHealth>,
}

impl AppState {
    pub fn new(writer: WriterSession, auth_token: Option<String>, namespace: String) -> Self {
        let snapshot = Arc::new(SnapshotCell::new(writer.owned_snapshot()));
        // A single non-empty token is read-write; `None` or an empty string is
        // open (an empty secret would otherwise be a bypass — `Bearer ` would
        // match it). The server overrides this with the resolved multi-token
        // config via `with_auth` at boot.
        let auth = match auth_token {
            Some(secret) if !secret.is_empty() => AuthConfig::single_read_write(secret),
            _ => AuthConfig::open(),
        };
        Self {
            writer: Arc::new(Mutex::new(writer)),
            snapshot,
            compaction_scheduler: Arc::new(CompactionScheduler::new()),
            catalog_cache: Arc::new(std::sync::Mutex::new(None)),
            auth: Arc::new(auth),
            namespace,
            query_timeout: Duration::ZERO,
            write_timeout: Duration::ZERO,
            query_row_cap: 0,
            write_stall_l0: 0,
            write_stall_delay: Duration::ZERO,
            memtable_flush_bytes: 0,
            memtable_stall_bytes: 0,
            memory: Arc::new(memory::MemoryGovernor::new(
                memory::DEFAULT_MEMORY_MAX_BYTES,
            )),
            writer_lock_timeout: Duration::ZERO,
            flush_notify: Arc::new(tokio::sync::Notify::new()),
            memtable_bytes_gauge: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
            metrics: Metrics::new(env!("CARGO_PKG_VERSION"), Duration::ZERO),
            authz: Arc::new(authz::NoOpAuthz),
            writer_health: WriterHealth::new(),
        }
    }

    /// Attach a pre-execution authorization hook (builder style). Defaults to
    /// allow-all ([`authz::NoOpAuthz`]).
    pub fn with_authz(mut self, authz: Arc<dyn authz::AuthzHook>) -> Self {
        self.authz = authz;
        self
    }

    /// Set the slow-query threshold (builder style). `Duration::ZERO` leaves
    /// the slow-query log off. Replaces the metrics registry, so call this at
    /// boot before any query is served.
    pub fn with_slow_query_threshold(mut self, threshold: Duration) -> Self {
        self.metrics = Metrics::new(env!("CARGO_PKG_VERSION"), threshold);
        self
    }

    /// Set the soft write-stall threshold and delay (builder style). A
    /// threshold of `0` leaves writes unstalled.
    pub fn with_write_stall(mut self, l0_threshold: usize, delay: Duration) -> Self {
        self.write_stall_l0 = l0_threshold;
        self.write_stall_delay = delay;
        self
    }

    /// Set the memtable byte thresholds (builder style): `flush_bytes` nudges
    /// the flush task early, `stall_bytes` applies write backpressure. `0`
    /// disables each.
    pub fn with_memtable_thresholds(mut self, flush_bytes: usize, stall_bytes: usize) -> Self {
        self.memtable_flush_bytes = flush_bytes;
        self.memtable_stall_bytes = stall_bytes;
        self
    }

    /// Attach the process-wide total-memory admission governor.
    pub fn with_memory_governor(mut self, memory: Arc<memory::MemoryGovernor>) -> Self {
        self.memory = memory;
        self
    }

    /// Set the foreground writer-lock acquisition bound (builder style).
    /// `Duration::ZERO` disables it.
    pub fn with_writer_lock_timeout(mut self, timeout: Duration) -> Self {
        self.writer_lock_timeout = timeout;
        self
    }

    /// The configured foreground writer-lock bound (`ZERO` = disabled).
    pub(crate) fn writer_lock_timeout(&self) -> Duration {
        self.writer_lock_timeout
    }

    /// If a write should be stalled given the worst bucket's current L0
    /// count and the live memtable size, the delay to apply; otherwise
    /// `None`. The caller samples both while holding the writer lock, then
    /// sleeps after releasing it.
    pub(crate) fn write_stall_for(&self, max_l0: usize, memtable_bytes: usize) -> Option<Duration> {
        if self.write_stall_l0 > 0
            && max_l0 >= self.write_stall_l0
            && self.write_stall_delay > Duration::ZERO
        {
            return Some(self.write_stall_delay);
        }
        if self.memtable_stall_bytes > 0 && memtable_bytes >= self.memtable_stall_bytes {
            // The byte backstop must bite even when the L0 stall is not
            // configured; fall back to a small fixed delay.
            return Some(if self.write_stall_delay > Duration::ZERO {
                self.write_stall_delay
            } else {
                Duration::from_millis(20)
            });
        }
        None
    }

    /// Post-commit bookkeeping, sampled while the writer lock is still held:
    /// publish the memtable gauge, nudge the flush task when the byte
    /// threshold is crossed, and return the backpressure delay (if any) to
    /// apply AFTER the lock is released.
    pub(crate) fn after_commit_backpressure(&self, writer: &WriterSession) -> Option<Duration> {
        let bytes = writer.memtable_bytes();
        self.memtable_bytes_gauge
            .store(bytes, std::sync::atomic::Ordering::Relaxed);
        if self.memtable_flush_bytes > 0 && bytes >= self.memtable_flush_bytes {
            self.flush_notify.notify_one();
        }
        self.write_stall_for(writer.max_l0_bucket_len(), bytes)
    }

    /// Acquire the writer mutex within the configured foreground bound.
    /// `None` = timed out; the caller answers 503 (HTTP) or a transient
    /// Bolt failure. Background tasks (flush/compaction/recovery) do NOT
    /// use this — they may wait as long as it takes.
    pub(crate) async fn lock_writer_bounded(
        &self,
        kind: WriterLockKind,
    ) -> Option<tokio::sync::MutexGuard<'_, WriterSession>> {
        let started = std::time::Instant::now();
        let guard = lock_writer_bounded(&self.writer, self.writer_lock_timeout).await;
        self.metrics
            .observe_writer_lock(kind, started.elapsed(), guard.is_some());
        guard
    }

    /// Set the per-read-query timeout (builder style). `Duration::ZERO`
    /// leaves reads unbounded.
    pub fn with_query_timeout(mut self, timeout: Duration) -> Self {
        self.query_timeout = timeout;
        self
    }

    /// Set the per-write-query timeout (builder style). `Duration::ZERO`
    /// leaves writes unbounded.
    pub fn with_write_timeout(mut self, timeout: Duration) -> Self {
        self.write_timeout = timeout;
        self
    }

    /// Replace the auth configuration (builder style). The server calls this
    /// at boot with the resolved token set (single token, tokens file, or
    /// open). Shared by clone with the Bolt serving path.
    pub fn with_auth(mut self, auth: Arc<AuthConfig>) -> Self {
        self.auth = auth;
        self
    }

    /// The accepted tokens, shared with the Bolt listener.
    pub(crate) fn auth(&self) -> Arc<AuthConfig> {
        self.auth.clone()
    }

    /// Set the per-read-query operator row cap (builder style). `0` leaves
    /// reads uncapped.
    pub fn with_query_row_cap(mut self, row_cap: usize) -> Self {
        self.query_row_cap = row_cap;
        self
    }

    /// Deadline for a read query starting now, or `None` when the timeout
    /// is disabled. Computed per query so each read gets the full budget.
    pub(crate) fn query_deadline(&self) -> Option<std::time::Instant> {
        (self.query_timeout > Duration::ZERO)
            .then(|| std::time::Instant::now() + self.query_timeout)
    }

    /// Deadline for a write query starting now, or `None` when the timeout
    /// is disabled. Computed per statement so each write gets the full budget.
    pub(crate) fn write_deadline(&self) -> Option<std::time::Instant> {
        (self.write_timeout > Duration::ZERO)
            .then(|| std::time::Instant::now() + self.write_timeout)
    }

    /// Operator row cap for a read query, or `None` when disabled.
    pub(crate) fn query_row_cap(&self) -> Option<usize> {
        (self.query_row_cap > 0).then_some(self.query_row_cap)
    }

    /// Optimizer [`StatsCatalog`] for `manifest`, built once per manifest
    /// version and reused across queries until the next write bumps the
    /// version. Every commit advances `manifest.version`, so a version
    /// match is sufficient for validity — a stale catalog is never served.
    pub(crate) fn catalog_for(&self, manifest: &Manifest) -> Arc<StatsCatalog> {
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

/// Assemble the `axum` router with every public route + auth
/// middleware. `/v0/livez`, `/v0/health`, `/v0/version` and `/v0/metrics` are
/// intentionally excluded from the auth check (a healthcheck probe or a
/// Prometheus scraper carries no token).
pub fn build_router(state: AppState) -> Router {
    let public = Router::new()
        .route("/v0/livez", get(livez))
        .route("/v0/health", get(health))
        .route("/v0/version", get(version))
        .route("/v0/metrics", get(metrics_handler));

    // Cypher is the only private HTTP work that creates a query working set.
    // Admission reserves conservative Content-Length/body-limit amplification
    // before its JSON extractor and retains it through the complete response.
    let admitted = Router::new()
        .route("/v0/cypher", post(cypher))
        .layer(DefaultBodyLimit::max(HTTP_CYPHER_BODY_LIMIT_BYTES))
        .layer(middleware::from_fn_with_state(
            state.clone(),
            require_memory_admission,
        ))
        .layer(middleware::from_fn_with_state(state.clone(), require_auth));
    // Flush is an authenticated pressure-relief operation: rejecting it at
    // the same gate as new Cypher work can leave a large memtable with no
    // operator escape hatch. It is excluded from the request timeout; the
    // handler has its own process-wide single-flight gate, while the storage
    // flush restores its frozen memtable on cancellation. The outer global
    // concurrency cap bounds clients waiting for that gate.
    let maintenance = Router::new()
        .route("/v0/admin/flush", post(admin_flush))
        .layer(middleware::from_fn_with_state(state.clone(), require_auth));

    limit_router(
        Router::new()
            .merge(timeout_router(Router::new().merge(public).merge(admitted)))
            .merge(maintenance)
            .with_state(state),
    )
}

/// Default request-processing deadline and global in-flight cap for the HTTP
/// listener. The timeout bounds how long a single request (body read + handler)
/// may run so a slow/stuck client cannot pin a task indefinitely; the
/// concurrency limit caps total in-flight requests so slow connections cannot
/// accumulate without bound and starve the server. Overridable via env.
fn http_request_timeout() -> Duration {
    std::env::var("NAMIDB_HTTP_REQUEST_TIMEOUT")
        .ok()
        .and_then(|s| humantime::parse_duration(&s).ok())
        .unwrap_or_else(|| Duration::from_secs(120))
}

fn http_max_concurrency() -> usize {
    std::env::var("NAMIDB_HTTP_MAX_CONCURRENCY")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|&n| n > 0)
        .unwrap_or(1024)
}

/// Apply the shared HTTP request timeout to ordinary request routes.
///
/// Operator maintenance routes are merged outside this layer: a flush can
/// legitimately outlive the request deadline. The storage-layer restore guard
/// makes client cancellation safe after the memtable has frozen.
fn timeout_router<S>(router: Router<S>) -> Router<S>
where
    S: Clone + Send + Sync + 'static,
{
    router.layer(tower_http::timeout::TimeoutLayer::with_status_code(
        axum::http::StatusCode::REQUEST_TIMEOUT,
        http_request_timeout(),
    ))
}

/// Keep one process-wide in-flight request cap, including maintenance.
///
/// This bounds authenticated clients queuing behind the single-flight flush
/// semaphore. Memory-pressure query rejections complete quickly and release
/// their permits, so the relief route remains reachable without admitting an
/// unbounded number of waiting tasks.
fn limit_router<S>(router: Router<S>) -> Router<S>
where
    S: Clone + Send + Sync + 'static,
{
    router.layer(tower::limit::GlobalConcurrencyLimitLayer::new(
        http_max_concurrency(),
    ))
}

/// Build the multi-tenant router with namespace extraction.
///
/// Routes are `/:namespace/v0/...` for all v0 endpoints. The namespace is
/// extracted from the path and used to look up (or create) a per-namespace
/// `WriterSession` via the registry.
///
/// Public endpoints (no auth required):
/// - `/:namespace/v0/livez` - liveness probe
/// - `/:namespace/v0/health` - readiness probe with namespace info
/// - `/v0/version` - process version (no namespace prefix)
/// - `/v0/metrics` - Prometheus metrics (no namespace prefix)
///
/// Private endpoints (auth required):
/// - `/:namespace/v0/cypher` - execute Cypher queries
/// - `/:namespace/v0/admin/flush` - manual flush
pub fn build_multi_tenant_router(shared: SharedAppState) -> Router {
    let public = Router::new()
        .route("/v0/version", get(version))
        .route("/v0/metrics", get(metrics_handler_multi));

    // Multi-tenant namespace-scoped routes: /:namespace/v0/...
    // Also register unprefixed /v0/... routes that resolve the namespace from
    // the X-NamiDB-Namespace header (or the configured default), so clients
    // can target a namespace without a path prefix.
    let namespace_public = Router::new()
        .route("/:namespace/v0/livez", get(livez_multi))
        .route("/:namespace/v0/health", get(health_multi))
        .route("/v0/livez", get(livez_multi))
        .route("/v0/health", get(health_multi_unprefixed));

    let namespace_admitted = Router::new()
        .route("/:namespace/v0/cypher", post(cypher_multi))
        .route("/v0/cypher", post(cypher_multi_unprefixed))
        .layer(DefaultBodyLimit::max(HTTP_CYPHER_BODY_LIMIT_BYTES))
        .layer(middleware::from_fn_with_state(
            shared.clone(),
            require_memory_admission_multi,
        ))
        .layer(middleware::from_fn_with_state(
            shared.clone(),
            require_auth_multi,
        ));
    let namespace_maintenance = Router::new()
        .route("/:namespace/v0/admin/flush", post(admin_flush_multi))
        .route("/v0/admin/flush", post(admin_flush_multi_unprefixed))
        .layer(middleware::from_fn_with_state(
            shared.clone(),
            require_auth_multi,
        ));

    limit_router(
        Router::new()
            .merge(timeout_router(
                Router::new()
                    .merge(public)
                    .merge(namespace_public)
                    .merge(namespace_admitted),
            ))
            .merge(namespace_maintenance)
            .with_state(shared),
    )
}

/// Resolve the namespace for an unprefixed request: the `X-NamiDB-Namespace`
/// header if present and non-empty, else the configured default namespace.
fn namespace_from_header(shared: &SharedAppState, headers: &axum::http::HeaderMap) -> String {
    headers
        .get("x-namidb-namespace")
        .and_then(|v| v.to_str().ok())
        .filter(|s| !s.is_empty())
        .map(str::to_string)
        .unwrap_or_else(|| shared.default_namespace.clone())
}

/// Resolve the namespace a multi-tenant request targets, for the auth
/// middleware's per-namespace scoping check.
///
/// **Correctness is load-bearing:** this MUST resolve the exact same namespace
/// the handler will serve, or a scoped token could be authorized for namespace
/// A while the request runs against namespace B (a cross-tenant bypass). To
/// guarantee that, we read the `:namespace` path parameter axum/matchit already
/// captured for the prefixed `/:namespace/v0/...` routes — the same value the
/// handler's `Path<String>` extractor deserializes — instead of re-parsing the
/// URI (which disagreed with matchit for paths like `/v0/v0/...`). Only when no
/// `:namespace` param was captured (a true unprefixed `/v0/...` route) do we
/// fall back to the `X-NamiDB-Namespace` header / default.
fn resolve_request_namespace(
    shared: &SharedAppState,
    params: &axum::extract::RawPathParams,
    headers: &axum::http::HeaderMap,
) -> String {
    for (key, value) in params.iter() {
        if key == "namespace" {
            return value.to_string();
        }
    }
    namespace_from_header(shared, headers)
}

/// Boot the server: parse URI, open a `WriterSession`, optionally
/// spawn a periodic flush task, and serve until the process receives
/// SIGINT.
pub async fn run(config: Config) -> anyhow::Result<()> {
    let memory_max_bytes = match std::env::var("NAMIDB_MEMORY_MAX_BYTES") {
        Ok(raw) => memory::resolve_memory_max_bytes(&raw).map_err(anyhow::Error::msg)?,
        Err(std::env::VarError::NotPresent) => memory::DEFAULT_MEMORY_MAX_BYTES,
        Err(error) => {
            return Err(anyhow::anyhow!(
                "NAMIDB_MEMORY_MAX_BYTES is not valid UTF-8: {error}"
            ))
        }
    };
    run_with_memory_max_bytes(config, memory_max_bytes).await
}

/// Boot the server with an explicit process-wide RSS/working-set ceiling.
///
/// The standalone binary uses this entry point for its CLI/env-parsed value;
/// [`run`] retains the 2.0 `Config` shape and reads
/// `NAMIDB_MEMORY_MAX_BYTES` for embedded callers.
pub async fn run_with_memory_max_bytes(
    config: Config,
    memory_max_bytes: usize,
) -> anyhow::Result<()> {
    namidb_storage::validate_cache_configuration().map_err(anyhow::Error::msg)?;
    if config.bolt_max_message_bytes == 0 {
        anyhow::bail!("NAMIDB_BOLT_MAX_MESSAGE_BYTES must be greater than zero");
    }
    // Bolt is single-namespace: the multi-tenant serve path never starts the
    // listener, so accepting both flags would silently drop the Bolt port.
    if config.multi_tenant && config.bolt_listen.is_some() {
        anyhow::bail!(
            "--bolt-listen is not supported with --multi-tenant: Bolt is \
             single-namespace (see docs/multi-tenancy.md). Run one \
             single-tenant server per namespace, or omit --bolt-listen / \
             NAMIDB_BOLT_LISTEN"
        );
    }
    // Resolve the auth configuration: a tokens file (with roles) wins, else a
    // single read-write `--auth-token`, else open.
    let auth = match (&config.auth_tokens_file, &config.auth_token) {
        (Some(path), _) => AuthConfig::load_file(path)?,
        // Refuse an empty `--auth-token`: it logs as "auth enabled" but a
        // `Bearer ` request would match the empty secret.
        (None, Some(secret)) if secret.is_empty() => {
            anyhow::bail!(
                "--auth-token is empty; set a real secret, or pass --no-auth \
                 to run without auth on purpose"
            )
        }
        (None, Some(secret)) => AuthConfig::single_read_write(secret.clone()),
        (None, None) => AuthConfig::open(),
    };
    // OIDC/JWT: build the validator (fail-fast on an unreachable JWKS) and
    // attach it. A bearer token is then first interpreted as a JWT.
    #[cfg(feature = "jwt")]
    let (auth, jwt_validator) = match config.jwt.as_ref() {
        Some(jwt_cfg) => {
            let v = Arc::new(crate::jwt::JwtValidator::new(jwt_cfg.clone()).await?);
            (auth.with_jwt(Arc::clone(&v)), Some(v))
        }
        None => (auth, None),
    };
    let auth = Arc::new(auth);
    // Refresh the JWKS hourly so keys can rotate without a restart.
    #[cfg(feature = "jwt")]
    if let Some(v) = &jwt_validator {
        v.spawn_refresh(Duration::from_secs(3600));
        info!("JWT auth enabled (JWKS refreshes hourly)");
    }
    if auth.is_open() {
        // Secure by default: an open server must be an explicit decision,
        // never the silent consequence of a missing env var. This check sits
        // AFTER the JWT attach on purpose — under the `jwt` feature a JWKS
        // URL alone is a valid auth configuration.
        if !config.no_auth {
            anyhow::bail!(
                "no auth configured: set --auth-token / NAMIDB_AUTH_TOKEN or \
                 --auth-tokens-file, or pass --no-auth / NAMIDB_NO_AUTH=1 to \
                 run without auth on purpose"
            );
        }
        warn!(
            "⚠️  namidb-server is running WITHOUT auth (--no-auth). Anyone \
             who can reach {} can issue arbitrary Cypher queries. Set \
             --auth-token (or env NAMIDB_AUTH_TOKEN), or --auth-tokens-file \
             for per-token roles, before exposing this port beyond localhost.",
            config.listen
        );
    } else {
        info!(tokens = auth.len(), "auth enabled");
    }

    // Resolve the authorization hook (RFC-015 Wave B). With the `pdp` feature
    // and a configured endpoint, every query/DDL is checked against an external
    // OPA-style policy (fail-closed); otherwise the allow-all NoOp keeps
    // behavior identical. Built once and shared across both serving paths.
    let authz: Arc<dyn authz::AuthzHook> = {
        #[cfg(feature = "pdp")]
        {
            match &config.pdp_url {
                Some(url) => {
                    info!(endpoint = %url, "external policy decision point (PDP) enabled");
                    Arc::new(crate::pdp::OpaAuthz::new(url.clone())?)
                }
                None => Arc::new(authz::NoOpAuthz),
            }
        }
        #[cfg(not(feature = "pdp"))]
        {
            Arc::new(authz::NoOpAuthz)
        }
    };
    let memory = Arc::new(memory::MemoryGovernor::new(memory_max_bytes));
    if memory_max_bytes > 0 {
        info!(
            memory_max_bytes,
            reclaim_at_bytes = memory_max_bytes.saturating_mul(90) / 100,
            watchdog_interval_ms = 500,
            "process resident-memory admission and watchdog enabled"
        );
        let cache_max = namidb_storage::cache_max_bytes();
        if cache_max >= memory_max_bytes {
            warn!(
                cache_max_bytes = cache_max,
                memory_max_bytes,
                "cache ceiling leaves no headroom under the total-memory ceiling; \
                 lower NAMIDB_CACHE_MAX_BYTES"
            );
        }
    }

    // One shutdown signal is shared by HTTP, optional Bolt, and the resident
    // memory watchdog in both single- and multi-tenant modes. Starting the
    // watchdog before namespace recovery lets it reclaim reconstructible
    // state even when opening a large writer moves RSS without any request.
    let (shutdown_tx, shutdown_rx) = tokio::sync::watch::channel(false);
    tokio::spawn(async move {
        wait_for_shutdown_signal().await;
        let _ = shutdown_tx.send(true);
    });
    if memory_max_bytes > 0 {
        // Dropping a Tokio JoinHandle deliberately detaches the task; the
        // shared shutdown receiver still terminates it cleanly.
        drop(Arc::clone(&memory).spawn_watchdog(shutdown_rx.clone()));
    }

    // Multi-tenant mode: create a registry and build the multi-tenant router.
    // The registry lazily creates WriterSessions per namespace on first access.
    if config.multi_tenant {
        let (store, _) = namidb_storage::parse_uri(&config.store_uri)
            .map_err(|e| anyhow::anyhow!("invalid --store: {e}"))?;
        let metrics = Metrics::new(env!("CARGO_PKG_VERSION"), config.slow_query_threshold);
        let maintenance = registry::MaintenanceConfig {
            flush_interval: config.flush_interval,
            compaction_interval: config.compaction_interval,
            sweep_min_age: config.sweep_min_age,
            sweep_delete: config.sweep_delete,
            compaction_l0_trigger: config.compaction_l0_trigger,
        };
        let registry = NamespaceRegistry::new(
            store,
            String::new(), // flat layout (no root prefix)
            config.max_namespaces,
            config.namespace_idle_timeout,
            metrics.clone(),
            maintenance,
        );
        let registry = Arc::new(registry);
        let shared = SharedAppState::new_with_memory(
            registry,
            auth,
            metrics,
            config.query_timeout,
            config.write_timeout,
            config.query_row_cap,
            config.write_stall_l0,
            config.write_stall_delay,
            config.memtable_flush_bytes,
            config.memtable_stall_bytes,
            memory.clone(),
            config.writer_lock_timeout,
            config.default_namespace.clone(),
        )
        .with_authz(authz.clone());
        let app = build_multi_tenant_router(shared);

        // TLS on the serving path.
        let tls_config: Option<Arc<rustls::ServerConfig>> =
            match (&config.tls_cert, &config.tls_key) {
                (Some(cert), Some(key)) => Some(tls::load_server_config(cert, key)?),
                (None, None) => None,
                _ => anyhow::bail!("set both --tls-cert and --tls-key to enable TLS, or neither"),
            };

        info!(multi_tenant = true, "starting multi-tenant server");
        return serve_http(app, config, tls_config, shutdown_rx).await;
    }

    let (store, paths) = namidb_storage::parse_uri(&config.store_uri)
        .map_err(|e| anyhow::anyhow!("invalid --store: {e}"))?;
    let namespace = paths.namespace().as_str().to_string();
    info!(
        namespace = %namespace,
        store = %config.store_uri,
        "opening namespace"
    );
    // A `ManifestStore` for the background orphan sweep, which loads the
    // committed manifest itself without the writer lock. Built from the
    // same `(store, paths)` before `open` consumes them.
    let maint_manifest_store = ManifestStore::new(store.clone(), paths.clone());
    let writer = WriterSession::open(store, paths).await?;

    let state = AppState::new(writer, None, namespace)
        .with_auth(auth)
        .with_authz(authz.clone())
        .with_query_timeout(config.query_timeout)
        .with_write_timeout(config.write_timeout)
        .with_query_row_cap(config.query_row_cap)
        .with_write_stall(config.write_stall_l0, config.write_stall_delay)
        .with_memtable_thresholds(config.memtable_flush_bytes, config.memtable_stall_bytes)
        .with_memory_governor(memory)
        .with_writer_lock_timeout(config.writer_lock_timeout)
        .with_slow_query_threshold(config.slow_query_threshold);

    // Periodic flush task — keeps the WAL bounded and L0 SSTs current.
    if config.flush_interval > Duration::ZERO {
        let state_for_flush = state.clone();
        let interval = config.flush_interval;
        // Reactive compaction trigger (RFC-027 P5): when a flush leaves a
        // bucket with >= this many L0 SSTs, start an off-lock pass immediately
        // rather than waiting for the periodic tick, so read amplification
        // does not spike between ticks.
        let l0_trigger = config.compaction_l0_trigger;
        tokio::spawn(async move {
            let mut tick = tokio::time::interval(interval);
            tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
            tick.tick().await; // first tick fires immediately; skip.
                               // After a local-persistence flush failure (spool disk full),
                               // an immediate retry re-runs the same doomed O(corpus) build
                               // under the writer lock. Hold retries back so the mutex stays
                               // available and the degraded state can be observed; the next
                               // attempt after the window self-heals once the disk clears.
            let mut retry_after: Option<std::time::Instant> = None;
            loop {
                // Flush on the timer OR when a committed write crossed the
                // memtable byte threshold (`after_commit_backpressure`
                // notifies), so a burst loader's working set is bounded by
                // bytes, not just wall clock.
                tokio::select! {
                    _ = tick.tick() => {}
                    _ = state_for_flush.flush_notify.notified() => {}
                }
                if retry_after.is_some_and(|at| std::time::Instant::now() < at) {
                    continue;
                }
                // Flush under the writer lock, then only enqueue a trigger
                // when the L0 high-water mark trips. The scheduler captures
                // a fresh basis inside its sole worker, so a queued follow-up
                // neither retains a stale manifest nor allocates another task.
                let should_compact = {
                    let lock_started = std::time::Instant::now();
                    let mut w = state_for_flush.writer.lock().await;
                    state_for_flush.metrics.observe_writer_lock(
                        WriterLockKind::Flush,
                        lock_started.elapsed(),
                        true,
                    );
                    let schema = w.snapshot().manifest().manifest.schema.clone();
                    match w.flush(schema.clone()).await {
                        Ok(_) => {
                            retry_after = None;
                            state_for_flush.writer_health.clear_persistence_degraded();
                            state_for_flush.snapshot.store(w.owned_snapshot());
                            state_for_flush
                                .memtable_bytes_gauge
                                .store(w.memtable_bytes(), std::sync::atomic::Ordering::Relaxed);
                            l0_trigger > 0 && w.max_l0_bucket_len() >= l0_trigger
                        }
                        Err(e) => {
                            error!(error = %e, "periodic flush failed");
                            if e.is_local_persistence() {
                                state_for_flush
                                    .writer_health
                                    .mark_persistence_degraded(persistence_degraded_reason(&e));
                                retry_after =
                                    Some(std::time::Instant::now() + FLUSH_FAILURE_RETRY_BACKOFF);
                            }
                            // A fenced/poisoned writer would fail every later
                            // flush AND every write; reopen under the held lock.
                            recovery::recover_writer_if_needed(
                                &mut w,
                                &state_for_flush.snapshot,
                                &state_for_flush.writer_health,
                                &state_for_flush.namespace,
                                &e,
                            )
                            .await;
                            false
                        }
                    }
                };
                if !should_compact {
                    continue;
                }
                // Dropping the handle detaches the sole worker in
                // single-tenant mode. The scheduler owns its admission state
                // and bounds a trigger storm to one worker + one follow-up.
                let _ = request_compaction(
                    &state_for_flush.compaction_scheduler,
                    CompactionTrigger::Reactive,
                    &state_for_flush.writer,
                    &state_for_flush.snapshot,
                    &state_for_flush.writer_health,
                    &state_for_flush.namespace,
                    &state_for_flush.metrics,
                    None,
                );
            }
        });
    }

    // Periodic background maintenance: compact L0 SSTs to L1 (bounds read
    // amplification), then sweep orphaned SST bodies left behind by
    // compaction. Compaction commits through the ONE writer lock — never a
    // second `WriterSession`, which would bump the epoch and fence the
    // foreground writer — but only the manifest CAS holds it: the expensive
    // prepare (downloads, merge, index rebuilds, uploads) runs off-lock
    // from a basis snapshotted under it. The sweep takes no lock (it
    // reads the committed manifest itself); the retention horizon (RFC-027)
    // is what keeps it from deleting a body a slow reader's pinned snapshot
    // still references, so it is safe to enable by default.
    if config.compaction_interval > Duration::ZERO {
        let state_for_maint = state.clone();
        let interval = config.compaction_interval;
        let sweep_min_age = config.sweep_min_age;
        let sweep_delete = config.sweep_delete;
        tokio::spawn(async move {
            let mut tick = tokio::time::interval(interval);
            tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
            tick.tick().await; // first tick fires immediately; skip.
            loop {
                tick.tick().await;
                // Enqueue into the same single-flight scheduler used by
                // reactive flush triggers. It captures its basis only when
                // each admitted pass actually begins.
                let _ = request_compaction(
                    &state_for_maint.compaction_scheduler,
                    CompactionTrigger::Periodic,
                    &state_for_maint.writer,
                    &state_for_maint.snapshot,
                    &state_for_maint.writer_health,
                    &state_for_maint.namespace,
                    &state_for_maint.metrics,
                    None,
                );
                // Sweep only after the entire active+pending burst is idle;
                // otherwise it can race immutable outputs uploaded by the
                // off-lock prepare before their manifest install.
                state_for_maint.compaction_scheduler.wait_idle().await;
                // `wait_idle` is only a point-in-time observation. Take the
                // scheduler's fair write guard as well so a reactive trigger
                // racing this boundary either finishes before the sweep or
                // waits until deletion has completed.
                let _sweep_guard = state_for_maint.compaction_scheduler.sweep_guard().await;
                // Orphan sweep — no writer lock. The `max_level` arg is only a
                // floor now: sweep_orphans scans up to the deepest level any
                // retained manifest occupies, so L2+ compaction outputs are
                // reclaimed too. The retention horizon (RFC-027) is the oldest
                // manifest version any live reader is pinned to; the sweep keeps
                // every object referenced from the horizon to current, so it can
                // never delete a body a reader still needs.
                // Read-side fence probe (RFC-027): drop readiness if a peer
                // writer's epoch has fenced this node, so a zombie replica
                // stops serving stale reads behind a green health check.
                recovery::probe_read_fence(
                    &maint_manifest_store,
                    &state_for_maint.snapshot,
                    &state_for_maint.writer_health,
                    &state_for_maint.namespace,
                )
                .await;
                let horizon = state_for_maint.snapshot.retention_horizon();
                match sweep_orphans(
                    &maint_manifest_store,
                    horizon,
                    sweep_min_age,
                    1,
                    sweep_delete,
                )
                .await
                {
                    Ok(report)
                        if report.orphans_found > 0
                            || report.manifest_snapshots_reclaimed > 0
                            || report.pointer_files_reclaimed > 0
                            || report.wal_segments_reclaimed > 0
                            || report.memtable_snapshots_reclaimed > 0 =>
                    {
                        info!(
                            found = report.orphans_found,
                            deleted = report.orphans_deleted,
                            bytes_freed = report.bytes_freed,
                            manifest_snapshots = report.manifest_snapshots_reclaimed,
                            manifest_bytes_freed = report.manifest_bytes_freed,
                            pointer_files = report.pointer_files_reclaimed,
                            pointer_bytes_freed = report.pointer_bytes_freed,
                            wal_segments = report.wal_segments_reclaimed,
                            wal_bytes_freed = report.wal_bytes_freed,
                            memtable_snapshots = report.memtable_snapshots_reclaimed,
                            memtable_snapshot_bytes_freed = report.memtable_snapshot_bytes_freed,
                            dry_run = !sweep_delete,
                            "orphan sweep"
                        )
                    }
                    Ok(_) => {}
                    Err(e) => error!(error = %e, "orphan sweep failed"),
                }
            }
        });
    }

    // Optional Bolt listener (binds an extra TCP port for native
    // Neo4j drivers — see RFC-022). When not configured we stay
    // HTTP-only.
    // TLS on the serving path: when `--tls-cert` / `--tls-key` are set, both
    // the HTTP server and the Bolt listener speak TLS from one shared config;
    // otherwise the server stays plaintext.
    let tls_config: Option<Arc<rustls::ServerConfig>> = match (&config.tls_cert, &config.tls_key) {
        (Some(cert), Some(key)) => Some(tls::load_server_config(cert, key)?),
        (None, None) => None,
        _ => anyhow::bail!("set both --tls-cert and --tls-key to enable TLS, or neither"),
    };

    if let Some(bolt_addr) = config.bolt_listen {
        let bolt_state = state.clone();
        let bolt_auth = state.auth();
        let tx_timeout = config.bolt_tx_timeout;
        let bolt_shutdown = shutdown_rx.clone();
        let bolt_tls = tls_config.clone().map(tls::acceptor);
        let bolt_max_message_bytes = config.bolt_max_message_bytes;
        tokio::spawn(async move {
            if let Err(e) = bolt::serve(
                bolt_state,
                bolt_addr,
                bolt_auth,
                tx_timeout,
                bolt_max_message_bytes,
                bolt_shutdown,
                bolt_tls,
            )
            .await
            {
                error!(error = %e, "bolt listener exited");
            }
        });
    }

    let app = build_router(state);
    serve_http(app, config, tls_config, shutdown_rx).await
}

/// Serve an HTTP router with TLS and graceful shutdown.
async fn serve_http(
    app: Router,
    config: Config,
    tls_config: Option<Arc<rustls::ServerConfig>>,
    shutdown_rx: tokio::sync::watch::Receiver<bool>,
) -> anyhow::Result<()> {
    let mut http_shutdown = shutdown_rx;

    match tls_config {
        Some(server_config) => {
            let handle = axum_server::Handle::new();
            let drain = handle.clone();
            tokio::spawn(async move {
                let _ = http_shutdown.wait_for(|stop| *stop).await;
                info!("shutdown signalled, draining HTTPS requests…");
                drain.graceful_shutdown(Some(Duration::from_secs(10)));
            });
            let rustls = axum_server::tls_rustls::RustlsConfig::from_config(server_config);
            info!(addr = %config.listen, "namidb-server listening (TLS)");
            axum_server::bind_rustls(config.listen, rustls)
                .handle(handle)
                .serve(app.into_make_service())
                .await?;
        }
        None => {
            let listener = TcpListener::bind(config.listen).await?;
            info!(addr = %config.listen, "namidb-server listening");
            axum::serve(listener, app)
                .with_graceful_shutdown(async move {
                    let _ = http_shutdown.wait_for(|stop| *stop).await;
                    info!("shutdown signalled, draining HTTP requests…");
                })
                .await?;
        }
    }
    Ok(())
}

/// Resolve when the process is asked to stop: Ctrl-C (SIGINT) on every
/// platform, plus SIGTERM on Unix — what `docker stop`, systemd and
/// Kubernetes send. Without the SIGTERM arm the server ignored the orderly
/// stop signal and was hard-killed once the grace period elapsed.
async fn wait_for_shutdown_signal() {
    let ctrl_c = async {
        let _ = tokio::signal::ctrl_c().await;
    };
    #[cfg(unix)]
    let terminate = async {
        match tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()) {
            Ok(mut sig) => {
                sig.recv().await;
            }
            Err(_) => std::future::pending::<()>().await,
        }
    };
    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();
    tokio::select! {
        _ = ctrl_c => info!("SIGINT received, draining…"),
        _ = terminate => info!("SIGTERM received, draining…"),
    }
}

// ── auth ──────────────────────────────────────────────────────────────

/// Reject new private HTTP work before Axum materialises its JSON body.
///
/// This middleware is layered inside authentication, so invalid credentials
/// are still rejected without disclosing process-pressure telemetry.
async fn admit_http_cypher_request(
    memory: &Arc<memory::MemoryGovernor>,
    req: axum::extract::Request,
    next: Next,
) -> Response {
    let wire_bytes = match http_cypher_wire_bytes(&req) {
        Ok(bytes) => bytes,
        Err(HttpBodyAdmissionError::InvalidContentLength) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(ErrorBody {
                    error: "invalid Content-Length for /v0/cypher".into(),
                }),
            )
                .into_response();
        }
        Err(HttpBodyAdmissionError::TooLarge { observed }) => {
            return (
                StatusCode::PAYLOAD_TOO_LARGE,
                Json(ErrorBody {
                    error: format!(
                        "Cypher JSON body is {observed} bytes; maximum is \
                         {HTTP_CYPHER_BODY_LIMIT_BYTES} bytes"
                    ),
                }),
            )
                .into_response();
        }
    };
    let started = std::time::Instant::now();
    let reservation = match memory
        .reserve_query_headroom(estimated_http_request_memory_bytes(wire_bytes))
        .await
    {
        Ok(reservation) => reservation,
        Err(pressure) => return memory_pressure_observation(started, pressure).response,
    };
    let response = next.run(req).await;
    // Keep projected JSON/query amplification charged through extraction,
    // planning, execution, response construction, and every early error.
    drop(reservation);
    response
}

async fn require_memory_admission(
    State(state): State<AppState>,
    req: axum::extract::Request,
    next: Next,
) -> Response {
    admit_http_cypher_request(&state.memory, req, next).await
}

/// Multi-tenant twin of [`require_memory_admission`]. It runs before namespace
/// lookup, preventing a cold tenant from allocating a recovered writer after
/// the process has reached its ceiling.
async fn require_memory_admission_multi(
    State(shared): State<SharedAppState>,
    req: axum::extract::Request,
    next: Next,
) -> Response {
    admit_http_cypher_request(&shared.memory, req, next).await
}

async fn require_auth_multi(
    State(shared): State<SharedAppState>,
    params: axum::extract::RawPathParams,
    mut req: axum::extract::Request,
    next: Next,
) -> Response {
    // Open mode: serve every request as an anonymous read-write principal.
    if shared.auth.is_open() {
        req.extensions_mut().insert(Principal::anonymous_rw());
        return next.run(req).await;
    }
    let presented = req
        .headers()
        .get(axum::http::header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .and_then(strip_bearer);
    // Resolve the target namespace HERE (before auth) so a token scoped to
    // other namespaces is rejected even though it is a valid token overall.
    // Uses axum's captured :namespace param so it can't disagree with the
    // handler (the /v0/v0/... bypass class).
    let namespace = resolve_request_namespace(&shared, &params, req.headers());
    match presented.and_then(|token| shared.auth.principal_for_in(token, &namespace)) {
        Some(principal) => {
            req.extensions_mut().insert(principal);
            next.run(req).await
        }
        None => (
            StatusCode::UNAUTHORIZED,
            [(
                axum::http::header::WWW_AUTHENTICATE,
                HeaderValue::from_static("Bearer realm=\"namidb\""),
            )],
            Json(ErrorBody {
                error: "missing or invalid bearer token, or token not scoped to this namespace"
                    .into(),
            }),
        )
            .into_response(),
    }
}

async fn require_auth(
    State(state): State<AppState>,
    mut req: axum::extract::Request,
    next: Next,
) -> Response {
    // Open mode: serve every request as an anonymous read-write principal.
    if state.auth.is_open() {
        req.extensions_mut().insert(Principal::anonymous_rw());
        return next.run(req).await;
    }
    let presented = req
        .headers()
        .get(axum::http::header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .and_then(strip_bearer);
    match presented.and_then(|token| state.auth.principal_for(token)) {
        Some(principal) => {
            // Carry the resolved principal to the handler (write gate + authz hook).
            req.extensions_mut().insert(principal);
            next.run(req).await
        }
        None => (
            StatusCode::UNAUTHORIZED,
            [(
                axum::http::header::WWW_AUTHENTICATE,
                HeaderValue::from_static("Bearer realm=\"namidb\""),
            )],
            Json(ErrorBody {
                error: "missing or invalid bearer token".into(),
            }),
        )
            .into_response(),
    }
}

/// The token from an `Authorization: Bearer <token>` header value. The scheme
/// is matched case-insensitively (RFC 7235 §2.1), matching the Bolt path.
fn strip_bearer(header: &str) -> Option<&str> {
    let (scheme, token) = header.split_once(' ')?;
    scheme.eq_ignore_ascii_case("bearer").then_some(token)
}

// ── routes ────────────────────────────────────────────────────────────

#[derive(Serialize)]
struct ErrorBody {
    error: String,
}

/// Classify an executor error for the HTTP response: the status code and the
/// machine-readable `code`. A deliberately-unsupported feature is a 400 with
/// `code: "unsupported"` (not a 500), so clients can tell "not implemented"
/// from a genuine server bug.
fn exec_error_classification(
    e: &namidb_query::exec::ExecError,
) -> (StatusCode, Option<&'static str>) {
    use namidb_query::exec::ExecError;
    match e {
        ExecError::Timeout => (StatusCode::GATEWAY_TIMEOUT, Some("timeout")),
        ExecError::RowCap(_) => (StatusCode::PAYLOAD_TOO_LARGE, Some("row_cap")),
        // A unique-constraint violation is a client error (duplicate value), not
        // a server fault — surface it as 409 Conflict.
        ExecError::Constraint(_) => (StatusCode::CONFLICT, Some("constraint")),
        ExecError::Storage(namidb_storage::Error::CacheCapacity { .. }) => (
            StatusCode::SERVICE_UNAVAILABLE,
            Some("search_index_cache_capacity"),
        ),
        ExecError::Storage(namidb_storage::Error::QueryWorkspaceExceeded { .. }) => (
            StatusCode::SERVICE_UNAVAILABLE,
            Some("search_workspace_capacity"),
        ),
        ExecError::Storage(namidb_storage::Error::SearchResultLimitExceeded { .. }) => {
            (StatusCode::PAYLOAD_TOO_LARGE, Some("search_result_limit"))
        }
        ExecError::Storage(namidb_storage::Error::SearchDocumentLimitExceeded { .. }) => {
            (StatusCode::PAYLOAD_TOO_LARGE, Some("search_document_limit"))
        }
        other if other.is_unsupported() => (StatusCode::BAD_REQUEST, Some("unsupported")),
        // A runtime evaluation error (division by zero, missing $parameter,
        // type mismatch) is the caller's program being wrong, not a server
        // fault — a 400, not a 500.
        ExecError::Eval(_) => (StatusCode::BAD_REQUEST, Some("eval_error")),
        _ => (StatusCode::INTERNAL_SERVER_ERROR, None),
    }
}

/// Wire-stable error taxonomy for HTTP bodies, mirroring the Bolt FAILURE
/// codes: a dotted Neo4j-shaped class (retry semantics live in the second
/// segment — `TransientError` is safe to retry, `ClientError` is not) and a
/// GQLSTATUS/SQLSTATE-class status per error family (42001 syntax, 42000
/// semantic, 0A000 not supported, 22000 evaluation, 23000 constraint,
/// 57014 timeout, 54000 result-limit, 53000 insufficient resources, 50N42
/// unclassified). Bolt <= 5.4 cannot carry GQLSTATUS on the wire — GQL-aware
/// drivers polyfill every FAILURE with 50N42 — so the HTTP body is where a
/// client reads the per-family status today.
fn exec_error_taxonomy(e: &namidb_query::exec::ExecError) -> (&'static str, &'static str) {
    use namidb_query::exec::ExecError;
    match e {
        ExecError::Timeout => ("Neo.ClientError.Transaction.TransactionTimedOut", "57014"),
        ExecError::RowCap(_) => ("Neo.ClientError.Statement.ResourceLimitExceeded", "54000"),
        ExecError::Constraint(_) => {
            ("Neo.ClientError.Schema.ConstraintValidationFailed", "23000")
        }
        ExecError::Storage(
            namidb_storage::Error::SearchResultLimitExceeded { .. }
            | namidb_storage::Error::SearchDocumentLimitExceeded { .. },
        ) => ("Neo.ClientError.Statement.ResourceLimitExceeded", "54000"),
        ExecError::Storage(_) => ("Neo.TransientError.General.DatabaseUnavailable", "53000"),
        other if other.is_unsupported() => ("Neo.ClientError.Statement.NotSupported", "0A000"),
        ExecError::Eval(_) => ("Neo.ClientError.Statement.ArgumentError", "22000"),
        ExecError::Runtime(_) => ("Neo.DatabaseError.General.UnknownError", "50N42"),
    }
}

/// 400 response for a statement rejected before execution (parse or plan),
/// carrying the same machine-readable taxonomy fields as executor failures.
fn statement_rejected_response(
    error: String,
    code: &'static str,
    neo4j_code: &'static str,
    gql_status: &'static str,
) -> Response {
    (
        StatusCode::BAD_REQUEST,
        Json(serde_json::json!({
            "error": error,
            "code": code,
            "neo4j_code": neo4j_code,
            "gql_status": gql_status,
        })),
    )
        .into_response()
}

fn parse_error_response(error: String) -> Response {
    statement_rejected_response(
        error,
        "parse_error",
        "Neo.ClientError.Statement.SyntaxError",
        "42001",
    )
}

fn plan_error_response(e: &namidb_query::LowerError) -> Response {
    let (code, neo4j_code, gql_status) = match e.kind {
        namidb_query::LowerErrorKind::UnsupportedFeature => (
            "unsupported",
            "Neo.ClientError.Statement.NotSupported",
            "0A000",
        ),
        _ => (
            "plan_error",
            "Neo.ClientError.Statement.SemanticError",
            "42000",
        ),
    };
    statement_rejected_response(format!("plan error: {e}"), code, neo4j_code, gql_status)
}

/// Build an HTTP error response from an executor failure, classifying it so a
/// deliberately-unsupported feature surfaces as 400/`unsupported` instead of
/// a bare 500. The `code` field is emitted only when classified, so existing
/// clients that deserialize the body loosely see no change on plain 500s.
/// Process-wide admission gate for full-scan queries (item 41).
///
/// A `NodeScan` materializes its whole label (and, through the LWW
/// reconciliation fallback, potentially the whole store) into memory, and
/// nothing else bounds how many of those run at once — the global HTTP
/// concurrency cap (1024) counts a point lookup and a 10M-row scan the
/// same. In field testing, parallel 1M-row scans collapsed aggregate
/// throughput to ~2 rps at 8 GB RSS: each scan's working set ballooned
/// past the memory governor's wire-byte-sized reservation and their
/// combined cache traffic evicted every point-lookup working set. This
/// gate bounds worst-case scan memory to `permits x largest-label` and
/// keeps the box responsive for indexed traffic. Waiters queue fairly
/// INSIDE the request-timeout layer, so a queued scan times out cleanly.
///
/// `NAMIDB_MAX_CONCURRENT_SCANS` overrides the default of 4; `0` disables
/// the gate.
fn scan_gate() -> Option<&'static tokio::sync::Semaphore> {
    static GATE: std::sync::OnceLock<Option<tokio::sync::Semaphore>> = std::sync::OnceLock::new();
    GATE.get_or_init(|| {
        let permits = std::env::var("NAMIDB_MAX_CONCURRENT_SCANS")
            .ok()
            .and_then(|raw| raw.parse::<usize>().ok())
            .unwrap_or(4);
        (permits > 0).then(|| tokio::sync::Semaphore::new(permits))
    })
    .as_ref()
}

/// True when the plan contains a full `NodeScan` anywhere (subplans
/// included) — the operator class the scan gate meters. Index lookups
/// (`NodeByPropertyValue`, `NodeById`) and pure expand chains pass free.
pub(crate) fn plan_contains_node_scan(plan: &namidb_query::LogicalPlan) -> bool {
    matches!(plan, namidb_query::LogicalPlan::NodeScan { .. })
        || plan.children().into_iter().any(plan_contains_node_scan)
}

/// Acquire a scan permit when the plan needs one. Holding the returned
/// guard for the duration of execution is the whole contract.
pub(crate) async fn acquire_scan_permit(
    plan: &namidb_query::LogicalPlan,
) -> Option<tokio::sync::SemaphorePermit<'static>> {
    let gate = scan_gate()?;
    if !plan_contains_node_scan(plan) {
        return None;
    }
    // The semaphore is never closed, so acquire cannot fail.
    gate.acquire().await.ok()
}

/// Serve `EXPLAIN [RAW] [VERBOSE] <query>`: render the plan instead of
/// executing. The AST has always promised this ("the executor honours it by
/// returning the plan tree"), but the server previously ignored the prefix
/// and silently EXECUTED the query — an `EXPLAIN CREATE ...` wrote data.
/// Non-RAW forms render the plan the optimizer actually produced against
/// the real manifest catalog, so index rewrites like `NodeByPropertyValue`
/// are visible (the CLI's offline explain has an empty catalog and can
/// never show them). A `# route:` footer then states the PHYSICAL access
/// path for every index lookup in the plan: posting-sidecar coverage
/// decides index-vs-scan at run time, and until item 39 that decision was
/// observable only through `elapsed_ms`.
fn explain_observation(
    parsed: &namidb_query::Query,
    plan: &namidb_query::LogicalPlan,
    owned: &namidb_storage::OwnedSnapshot,
    catalog: &StatsCatalog,
    started: std::time::Instant,
) -> ObservedQuery {
    let rows = explain_plan_lines(parsed, plan, owned, catalog)
        .into_iter()
        .map(|line| {
            let mut row = serde_json::Map::new();
            row.insert("plan".into(), serde_json::Value::String(line));
            row
        })
        .collect();
    ObservedQuery {
        kind: Some(QueryKind::Read),
        ok: true,
        elapsed: started.elapsed(),
        response: Json(CypherResponse {
            columns: vec!["plan".into()],
            rows,
            write_outcome: None,
        })
        .into_response(),
    }
}

/// The rendered EXPLAIN lines (plan tree + `# route:` footer), shared by
/// the HTTP and Bolt surfaces.
pub(crate) fn explain_plan_lines(
    parsed: &namidb_query::Query,
    plan: &namidb_query::LogicalPlan,
    owned: &namidb_storage::OwnedSnapshot,
    catalog: &StatsCatalog,
) -> Vec<String> {
    let text = if parsed.explain_raw && parsed.explain_verbose {
        namidb_query::explain_query_raw_verbose(parsed, catalog)
            .unwrap_or_else(|e| format!("explain error: {e}"))
    } else if parsed.explain_raw {
        namidb_query::explain_query_raw(parsed).unwrap_or_else(|e| format!("explain error: {e}"))
    } else if parsed.explain_verbose {
        namidb_query::explain_verbose(plan, catalog)
    } else {
        namidb_query::explain(plan)
    };
    let mut lines: Vec<String> = text.lines().map(str::to_string).collect();
    if !parsed.explain_raw {
        let snapshot = owned.borrow();
        collect_route_notes(plan, &snapshot, &mut lines);
    }
    lines
}

/// Append one `# route:` line per index-lookup operator in the plan.
fn collect_route_notes(
    plan: &namidb_query::LogicalPlan,
    snapshot: &namidb_storage::Snapshot<'_>,
    out: &mut Vec<String>,
) {
    if let namidb_query::LogicalPlan::NodeByPropertyValue {
        label,
        property,
        value,
        multi,
        ..
    } = plan
    {
        let shown = if label.is_empty() { "*" } else { label };
        let kind = if *multi { "posting" } else { "unique" };
        let numeric = matches!(
            &value.kind,
            namidb_query::parser::ast::ExpressionKind::Literal(
                namidb_query::parser::ast::Literal::Integer(_)
                    | namidb_query::parser::ast::Literal::Float(_)
            )
        );
        let note = if numeric {
            format!(
                "# route: {shown}.{property} → scan \
                 (numeric equality is not posting-indexed; only String/Bool are)"
            )
        } else {
            let (covered, total) = snapshot.property_index_coverage(label, property);
            if total == 0 {
                format!("# route: {shown}.{property} → memtable ({kind} lookup; no SSTs in scope)")
            } else if covered == total {
                format!(
                    "# route: {shown}.{property} → index \
                     ({kind} lookup; posting sidecars {covered}/{total} SSTs)"
                )
            } else {
                format!(
                    "# route: {shown}.{property} → SCAN FALLBACK \
                     (posting sidecars {covered}/{total} SSTs; \
                     a compaction pass materializes the rest)"
                )
            }
        };
        out.push(note);
    }
    for child in plan.children() {
        collect_route_notes(child, snapshot, out);
    }
}

fn exec_failure_response(prefix: &str, e: &namidb_query::exec::ExecError) -> Response {
    let (status, code) = exec_error_classification(e);
    let (neo4j_code, gql_status) = exec_error_taxonomy(e);
    let error = format!("{prefix}: {e}");
    // `code` stays absent on unclassified 500s (loose clients see no new
    // requirement there); the taxonomy fields are additive everywhere.
    let body = match code {
        Some(c) => Json(serde_json::json!({
            "error": error,
            "code": c,
            "neo4j_code": neo4j_code,
            "gql_status": gql_status,
        })),
        None => Json(serde_json::json!({
            "error": error,
            "neo4j_code": neo4j_code,
            "gql_status": gql_status,
        })),
    };
    (status, body).into_response()
}

/// Bounded writer-mutex acquisition for foreground request paths. A zero
/// `timeout` disables the bound. `None` = timed out.
pub(crate) async fn lock_writer_bounded(
    writer: &tokio::sync::Mutex<WriterSession>,
    timeout: Duration,
) -> Option<tokio::sync::MutexGuard<'_, WriterSession>> {
    if timeout.is_zero() {
        return Some(writer.lock().await);
    }
    tokio::time::timeout(timeout, writer.lock()).await.ok()
}

/// How long the periodic flush loop waits after a local-persistence flush
/// failure before retrying. Immediate retries re-run the same doomed
/// O(corpus) build against the same full disk while holding the writer
/// mutex; this window keeps the mutex available and gives the operator a
/// stable degraded state instead of a grinding loop.
pub(crate) const FLUSH_FAILURE_RETRY_BACKOFF: Duration = Duration::from_secs(30);

/// The reason recorded in `WriterHealth` when a flush fails on local
/// persistence. Written for the operator who reads `/v0/health`.
pub(crate) fn persistence_degraded_reason(e: &namidb_storage::Error) -> String {
    format!(
        "flush failed on local persistence (spool disk full or unwritable?): {e}; \
         writes are rejected with 507 and reads keep serving the last committed \
         state; the flush retries automatically and recovery clears this"
    )
}

/// The uniform 507 body for writes rejected while the namespace is
/// persistence-degraded. 507 (Insufficient Storage) so clients and load
/// balancers can tell "stop sending writes, the disk is the problem" apart
/// from the transient 503 "writer is busy; retry".
fn persistence_degraded_response(reason: &str) -> Response {
    (
        StatusCode::INSUFFICIENT_STORAGE,
        Json(ErrorBody {
            error: format!("namespace degraded: {reason}"),
        }),
    )
        .into_response()
}

/// Typed rejection for write intake while the namespace's local persistence
/// is degraded (spool disk full/unwritable — the flush cannot drain the
/// memtable). Checked BEFORE queueing on the writer mutex so a doomed flush
/// cycle cannot convert every write into an opaque lock-timeout 503, and so
/// rejected writes never pin a concurrency slot waiting on the mutex.
/// Reads are never gated on this.
fn write_intake_rejection(health: &recovery::WriterHealth) -> Option<Response> {
    health
        .persistence_degraded_reason()
        .map(|reason| persistence_degraded_response(&reason))
}

/// The uniform 503 body for a foreground writer-lock timeout.
fn writer_busy_response() -> Response {
    (
        StatusCode::SERVICE_UNAVAILABLE,
        Json(ErrorBody {
            error: "writer is busy: could not acquire the write lock within the \
                    configured bound; retry"
                .into(),
        }),
    )
        .into_response()
}

/// The uniform 503 body returned when a multi-tenant request waited on a
/// namespace incarnation that was retired by eviction. Clients may safely
/// retry: the registry will route the next attempt to the live incarnation.
fn namespace_retired_response() -> Response {
    (
        StatusCode::SERVICE_UNAVAILABLE,
        Json(ErrorBody {
            error: "namespace was evicted while waiting for its writer; retry".into(),
        }),
    )
        .into_response()
}

#[derive(Serialize)]
struct HealthResponse {
    status: &'static str,
    namespace: String,
    manifest_version: u64,
    epoch: u64,
    /// `"ok"` or `"degraded"`. Degraded means the writer session is
    /// fenced/poisoned and the automatic reopen has not yet succeeded:
    /// reads still work (the published snapshot serves them) but every
    /// write fails, so the probe as a whole reports 503 / not-ready.
    writer: &'static str,
    /// The commit/flush failure keeping the writer degraded, when it is.
    #[serde(skip_serializing_if = "Option::is_none")]
    writer_error: Option<String>,
    /// Last observed live-memtable size in bytes (the un-flushed working
    /// set), published lock-free by the write/flush paths. `None` on the
    /// multi-tenant probe, which has no single gauge.
    #[serde(skip_serializing_if = "Option::is_none")]
    memtable_bytes: Option<usize>,
    /// Process RSS/working-set telemetry. `memory_pressure` becomes true only
    /// when a configured non-zero ceiling has actually been reached.
    memory_resident_bytes: usize,
    memory_max_bytes: usize,
    memory_pressure: bool,
}

/// Build the health payload + status code from the published snapshot and
/// the writer health. Shared by the single- and multi-tenant probes. A
/// degraded writer is a readiness failure (503): the server can still
/// serve reads, but an orchestrator must not treat it as fully healthy
/// while every write is failing.
fn health_response(
    namespace: String,
    manifest: &Manifest,
    writer_health: &WriterHealth,
    memtable_bytes: Option<usize>,
    memory: &memory::MemoryGovernor,
) -> Response {
    let writer_error = writer_health.degraded_reason();
    let writer_degraded = writer_error.is_some();
    let memory_pressure = memory.over_limit();
    let degraded = writer_degraded || memory_pressure;
    let body = HealthResponse {
        status: if degraded { "degraded" } else { "ok" },
        namespace,
        manifest_version: manifest.version,
        epoch: manifest.epoch.as_u64(),
        writer: if writer_degraded { "degraded" } else { "ok" },
        writer_error,
        memtable_bytes,
        memory_resident_bytes: memory.resident_bytes(),
        memory_max_bytes: memory.max_bytes(),
        memory_pressure,
    };
    let code = if degraded {
        StatusCode::SERVICE_UNAVAILABLE
    } else {
        StatusCode::OK
    };
    (code, Json(body)).into_response()
}

/// Liveness: the process is up and its async runtime is responsive. Takes no
/// lock and reads no namespace state, so a long write or compaction (which
/// holds the writer lock) can never make it hang — a container liveness probe
/// stays green while the engine is busy. This is the endpoint a Docker
/// HEALTHCHECK or a Kubernetes livenessProbe should target.
async fn livez() -> impl IntoResponse {
    Json(serde_json::json!({ "status": "ok" }))
}

/// Readiness: report the latest published snapshot's manifest version and
/// epoch WITHOUT taking the writer lock. The snapshot is republished after
/// every commit, so it reflects committed state; a long write or compaction
/// holding the writer lock does not stall the probe. The writer status
/// ([`WriterHealth`]) rides along: a fenced/poisoned writer whose automatic
/// reopen has not yet landed turns the probe 503 so writes are not routed
/// to a server that can only fail them.
async fn health(State(state): State<AppState>) -> impl IntoResponse {
    let _ = state.memory.sample();
    let owned = state.snapshot.load();
    let m = &owned.manifest().manifest;
    health_response(
        state.namespace.clone(),
        m,
        &state.writer_health,
        Some(
            state
                .memtable_bytes_gauge
                .load(std::sync::atomic::Ordering::Relaxed),
        ),
        &state.memory,
    )
}

#[derive(Serialize)]
struct VersionResponse {
    version: &'static str,
    build_target: &'static str,
}

async fn version() -> impl IntoResponse {
    Json(VersionResponse {
        version: env!("CARGO_PKG_VERSION"),
        build_target: env!("CARGO_PKG_NAME"),
    })
}

/// Prometheus scrape endpoint. Renders the process query metrics in the text
/// exposition format. Unauthenticated, like the health probes, so a scraper
/// needs no bearer token.
async fn metrics_handler(State(state): State<AppState>) -> impl IntoResponse {
    let _ = state.memory.sample();
    (
        [(
            axum::http::header::CONTENT_TYPE,
            HeaderValue::from_static("text/plain; version=0.0.4; charset=utf-8"),
        )],
        state.metrics.render_with_memory(&state.memory),
    )
}

#[derive(Deserialize)]
struct CypherRequest {
    query: String,
    #[serde(default)]
    params: serde_json::Map<String, serde_json::Value>,
}

#[derive(Serialize)]
struct CypherResponse {
    columns: Vec<String>,
    rows: Vec<serde_json::Map<String, serde_json::Value>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    write_outcome: Option<WriteSummary>,
}

#[derive(Serialize)]
struct WriteSummary {
    nodes_created: u64,
    edges_created: u64,
    nodes_deleted: u64,
    edges_deleted: u64,
    properties_set: u64,
}

impl From<&WriteOutcome> for WriteSummary {
    fn from(o: &WriteOutcome) -> Self {
        Self {
            nodes_created: o.nodes_created,
            edges_created: o.edges_created,
            nodes_deleted: o.nodes_deleted,
            edges_deleted: o.edges_deleted,
            properties_set: o.properties_set,
        }
    }
}

/// One executed query, classified for metrics: read vs write (`None` if it
/// failed before planning), whether it succeeded, the wall-clock it took
/// (measured up to the end of execution, excluding any write-stall sleep), and
/// the HTTP response to return.
struct ObservedQuery {
    kind: Option<QueryKind>,
    ok: bool,
    elapsed: Duration,
    response: Response,
}

fn memory_pressure_observation(
    started: std::time::Instant,
    pressure: memory::MemoryPressure,
) -> ObservedQuery {
    let projected = pressure
        .resident_bytes
        .saturating_add(pressure.requested_headroom_bytes);
    let error = if pressure.requested_headroom_bytes == 0 {
        format!(
            "process memory pressure: resident {} bytes reached configured maximum {} bytes; \
             reconstructible caches were reclaimed, retry after memory falls",
            pressure.resident_bytes, pressure.max_bytes
        )
    } else {
        format!(
            "process memory pressure: resident {} bytes plus {} bytes of projected request \
             headroom would reach {} bytes, at or above configured maximum {} bytes; split the \
             request or retry after memory falls",
            pressure.resident_bytes,
            pressure.requested_headroom_bytes,
            projected,
            pressure.max_bytes
        )
    };
    ObservedQuery {
        kind: None,
        ok: false,
        elapsed: started.elapsed(),
        response: (StatusCode::SERVICE_UNAVAILABLE, Json(ErrorBody { error })).into_response(),
    }
}

fn namespace_retired_observation(started: std::time::Instant) -> ObservedQuery {
    ObservedQuery {
        kind: Some(QueryKind::Write),
        ok: false,
        elapsed: started.elapsed(),
        response: namespace_retired_response(),
    }
}

// ───────────────────── CREATE VECTOR INDEX (DDL) ──────────────────────
//
// `CREATE VECTOR INDEX` is schema DDL — neither a read nor a row write — so
// the server intercepts it after parsing and before planning (it never
// becomes a `LogicalPlan`). The whole path is feature-gated: with
// `vector-index` off the intercept is compiled out and the DDL reaches the
// lowerer, which rejects it (HTTP 400 / Bolt NotSupported). On, it calls
// `WriterSession::register_vector_index` (a metadata-only manifest commit)
// and republishes the snapshot so the next query plans against the new
// catalog. The compaction build hook materializes the `.vg` graph lazily.

#[cfg(feature = "vector-index")]
fn vector_index_descriptor_from(
    cvi: &namidb_query::parser::ast::CreateVectorIndexClause,
) -> namidb_storage::manifest::VectorIndexDescriptor {
    use namidb_query::parser::ast::{VectorMetric as M, VectorQuantization as Q};
    let metric = match cvi.metric {
        M::Cosine => namidb_storage::manifest::VectorMetric::Cosine,
        M::Dot => namidb_storage::manifest::VectorMetric::Dot,
        M::Euclidean => namidb_storage::manifest::VectorMetric::Euclidean,
    };
    let quantization = match cvi.quantization {
        Q::None => namidb_storage::manifest::VectorQuantization::None,
        Q::Int8 => namidb_storage::manifest::VectorQuantization::Int8,
    };
    // Vamana build defaults mirror `namidb_ann::BuildParams::default()`
    // (R=64, L_build=128, α=1.2); the user's `WITH {…}` overrides win.
    namidb_storage::manifest::VectorIndexDescriptor {
        name: cvi.name.name.clone(),
        label: cvi.label.name.clone(),
        property: cvi.property.name.clone(),
        dim: cvi.dim,
        metric,
        r: cvi.r.unwrap_or(64),
        l_build: cvi.l_build.unwrap_or(128),
        alpha: cvi.alpha.unwrap_or(1.2),
        quantization,
    }
}

/// Build the descriptor, commit it via the writer (metadata-only), and
/// republish the snapshot so subsequent reads see the new index. Shared by
/// the HTTP and Bolt DDL paths.
#[cfg(feature = "vector-index")]
async fn apply_create_vector_index(
    writer: &mut WriterSession,
    snapshot: &SnapshotCell,
    cvi: &namidb_query::parser::ast::CreateVectorIndexClause,
) -> Result<u64, namidb_storage::Error> {
    let desc = vector_index_descriptor_from(cvi);
    let version = writer
        .register_vector_index(desc, cvi.if_not_exists)
        .await?;
    // Refresh the published snapshot (catalog_for rebuilds on version bump).
    snapshot.store(writer.owned_snapshot());
    Ok(version)
}

/// HTTP shape for a `CREATE VECTOR INDEX`: classify (write), gate on role,
/// run the DDL, return an empty `CypherResponse` on success. Shared by the
/// single- and multi-tenant paths, which pass their own writer/snapshot.
#[cfg(feature = "vector-index")]
// The DDL params are all distinct (session, publish cell, health, authz,
// clause, principal, clock); bundling them would not aid readability.
#[allow(clippy::too_many_arguments)]
async fn run_create_vector_index(
    writer: &Arc<tokio::sync::Mutex<WriterSession>>,
    snapshot: &Arc<SnapshotCell>,
    writer_health: &Arc<WriterHealth>,
    namespace: &str,
    namespace_state: Option<&NamespaceState>,
    authz: &Arc<dyn authz::AuthzHook>,
    cvi: &namidb_query::parser::ast::CreateVectorIndexClause,
    principal: &Principal,
    started: std::time::Instant,
) -> ObservedQuery {
    // DDL mutates durable schema state, so a read-only token may not run it.
    if !principal.allows_write() {
        return ObservedQuery {
            kind: Some(QueryKind::Write),
            ok: false,
            elapsed: started.elapsed(),
            response: (
                StatusCode::FORBIDDEN,
                Json(ErrorBody {
                    error: "this token is read-only; schema commands are forbidden".into(),
                }),
            )
                .into_response(),
        };
    }
    // Authorization hook: DDL is the most-privileged op, so it must consult the
    // policy too (it is intercepted pre-plan, so via check_schema). NoOp allows.
    let op = authz::SchemaOp::CreateVectorIndex {
        name: &cvi.name.name,
        label: &cvi.label.name,
        property: &cvi.property.name,
    };
    if let Err(denied) = authz.check_schema(principal, op).await {
        return ObservedQuery {
            kind: None,
            ok: false,
            elapsed: started.elapsed(),
            response: (
                StatusCode::FORBIDDEN,
                Json(ErrorBody {
                    error: denied.to_string(),
                }),
            )
                .into_response(),
        };
    }
    if let Some(rejection) = write_intake_rejection(writer_health) {
        return ObservedQuery {
            kind: None,
            ok: false,
            elapsed: started.elapsed(),
            response: rejection,
        };
    }
    let mut w = writer.lock().await;
    if namespace_state.is_some_and(NamespaceState::is_retired) {
        drop(w);
        return namespace_retired_observation(started);
    }
    let result = apply_create_vector_index(&mut w, snapshot, cvi).await;
    if let Err(e) = &result {
        // A fenced/poisoned session would fail every later write; reopen it
        // in place under the lock we already hold (no-op for user errors
        // like a duplicate index name).
        if !namespace_state.is_some_and(NamespaceState::is_retired) {
            recovery::recover_writer_if_needed(&mut w, snapshot, writer_health, namespace, e).await;
        }
    }
    drop(w);
    let elapsed = started.elapsed();
    match result {
        Ok(_) => ObservedQuery {
            kind: Some(QueryKind::Write),
            ok: true,
            elapsed,
            response: Json(CypherResponse {
                columns: vec![],
                rows: vec![],
                write_outcome: None,
            })
            .into_response(),
        },
        Err(e) => {
            // A duplicate name/target is a user error (400); a fence or lost
            // CAS is a server-side condition (503).
            let status = match &e {
                namidb_storage::Error::Precondition(_) | namidb_storage::Error::Invariant(_) => {
                    StatusCode::BAD_REQUEST
                }
                _ => StatusCode::SERVICE_UNAVAILABLE,
            };
            ObservedQuery {
                kind: Some(QueryKind::Write),
                ok: false,
                elapsed,
                response: (
                    status,
                    Json(ErrorBody {
                        error: e.to_string(),
                    }),
                )
                    .into_response(),
            }
        }
    }
}

#[cfg(feature = "text-index")]
fn text_index_descriptor_from(
    cfi: &namidb_query::parser::ast::CreateFulltextIndexClause,
) -> namidb_storage::manifest::TextIndexDescriptor {
    namidb_storage::manifest::TextIndexDescriptor::new(
        cfi.name.name.clone(),
        cfi.label.name.clone(),
        cfi.properties.iter().map(|p| p.name.clone()).collect(),
    )
}

/// Register a full-text index (metadata-only) and republish the snapshot.
/// Shared by the HTTP and Bolt DDL paths. The compaction build hook materializes
/// the `.ft` body lazily.
#[cfg(feature = "text-index")]
async fn apply_create_fulltext_index(
    writer: &mut WriterSession,
    snapshot: &SnapshotCell,
    cfi: &namidb_query::parser::ast::CreateFulltextIndexClause,
) -> Result<u64, namidb_storage::Error> {
    let desc = text_index_descriptor_from(cfi);
    let version = writer.register_text_index(desc, cfi.if_not_exists).await?;
    snapshot.store(writer.owned_snapshot());
    Ok(version)
}

/// HTTP shape for a `CREATE FULLTEXT INDEX`: gate on role + authz, run the DDL,
/// return an empty `CypherResponse` on success. Mirrors `run_create_vector_index`.
#[cfg(feature = "text-index")]
// The DDL params are all distinct (session, publish cell, health, authz,
// clause, principal, clock); bundling them would not aid readability.
#[allow(clippy::too_many_arguments)]
async fn run_create_fulltext_index(
    writer: &Arc<tokio::sync::Mutex<WriterSession>>,
    snapshot: &Arc<SnapshotCell>,
    writer_health: &Arc<WriterHealth>,
    namespace: &str,
    namespace_state: Option<&NamespaceState>,
    authz: &Arc<dyn authz::AuthzHook>,
    cfi: &namidb_query::parser::ast::CreateFulltextIndexClause,
    principal: &Principal,
    started: std::time::Instant,
) -> ObservedQuery {
    if !principal.allows_write() {
        return ObservedQuery {
            kind: Some(QueryKind::Write),
            ok: false,
            elapsed: started.elapsed(),
            response: (
                StatusCode::FORBIDDEN,
                Json(ErrorBody {
                    error: "this token is read-only; schema commands are forbidden".into(),
                }),
            )
                .into_response(),
        };
    }
    let props: Vec<String> = cfi.properties.iter().map(|p| p.name.clone()).collect();
    let op = authz::SchemaOp::CreateFulltextIndex {
        name: &cfi.name.name,
        label: &cfi.label.name,
        properties: &props,
    };
    if let Err(denied) = authz.check_schema(principal, op).await {
        return ObservedQuery {
            kind: None,
            ok: false,
            elapsed: started.elapsed(),
            response: (
                StatusCode::FORBIDDEN,
                Json(ErrorBody {
                    error: denied.to_string(),
                }),
            )
                .into_response(),
        };
    }
    if let Some(rejection) = write_intake_rejection(writer_health) {
        return ObservedQuery {
            kind: None,
            ok: false,
            elapsed: started.elapsed(),
            response: rejection,
        };
    }
    let mut w = writer.lock().await;
    if namespace_state.is_some_and(NamespaceState::is_retired) {
        drop(w);
        return namespace_retired_observation(started);
    }
    let result = apply_create_fulltext_index(&mut w, snapshot, cfi).await;
    if let Err(e) = &result {
        // A fenced/poisoned session would fail every later write; reopen it
        // in place under the lock we already hold (no-op for user errors
        // like a duplicate index name).
        if !namespace_state.is_some_and(NamespaceState::is_retired) {
            recovery::recover_writer_if_needed(&mut w, snapshot, writer_health, namespace, e).await;
        }
    }
    drop(w);
    let elapsed = started.elapsed();
    match result {
        Ok(_) => ObservedQuery {
            kind: Some(QueryKind::Write),
            ok: true,
            elapsed,
            response: Json(CypherResponse {
                columns: vec![],
                rows: vec![],
                write_outcome: None,
            })
            .into_response(),
        },
        Err(e) => {
            let status = match &e {
                namidb_storage::Error::Precondition(_) | namidb_storage::Error::Invariant(_) => {
                    StatusCode::BAD_REQUEST
                }
                _ => StatusCode::SERVICE_UNAVAILABLE,
            };
            ObservedQuery {
                kind: Some(QueryKind::Write),
                ok: false,
                elapsed,
                response: (
                    status,
                    Json(ErrorBody {
                        error: e.to_string(),
                    }),
                )
                    .into_response(),
            }
        }
    }
}

/// Drop a vector index (metadata-only: descriptor + `.vg` SST refs in one
/// commit) and republish the snapshot. Shared by the HTTP and Bolt DDL paths.
#[cfg(feature = "vector-index")]
async fn apply_drop_vector_index(
    writer: &mut WriterSession,
    snapshot: &SnapshotCell,
    dvi: &namidb_query::parser::ast::DropVectorIndexClause,
) -> Result<u64, namidb_storage::Error> {
    let version = writer
        .drop_vector_index(&dvi.name.name, dvi.if_exists)
        .await?;
    snapshot.store(writer.owned_snapshot());
    Ok(version)
}

/// HTTP shape for a `DROP VECTOR INDEX`: gate on role + authz, run the DDL,
/// return an empty `CypherResponse` on success. Mirrors
/// `run_create_vector_index` (same authz treatment as CREATE).
#[cfg(feature = "vector-index")]
// The DDL params are all distinct (session, publish cell, health, authz,
// clause, principal, clock); bundling them would not aid readability.
#[allow(clippy::too_many_arguments)]
async fn run_drop_vector_index(
    writer: &Arc<tokio::sync::Mutex<WriterSession>>,
    snapshot: &Arc<SnapshotCell>,
    writer_health: &Arc<WriterHealth>,
    namespace: &str,
    namespace_state: Option<&NamespaceState>,
    authz: &Arc<dyn authz::AuthzHook>,
    dvi: &namidb_query::parser::ast::DropVectorIndexClause,
    principal: &Principal,
    started: std::time::Instant,
) -> ObservedQuery {
    if !principal.allows_write() {
        return ObservedQuery {
            kind: Some(QueryKind::Write),
            ok: false,
            elapsed: started.elapsed(),
            response: (
                StatusCode::FORBIDDEN,
                Json(ErrorBody {
                    error: "this token is read-only; schema commands are forbidden".into(),
                }),
            )
                .into_response(),
        };
    }
    let op = authz::SchemaOp::DropVectorIndex {
        name: &dvi.name.name,
    };
    if let Err(denied) = authz.check_schema(principal, op).await {
        return ObservedQuery {
            kind: None,
            ok: false,
            elapsed: started.elapsed(),
            response: (
                StatusCode::FORBIDDEN,
                Json(ErrorBody {
                    error: denied.to_string(),
                }),
            )
                .into_response(),
        };
    }
    if let Some(rejection) = write_intake_rejection(writer_health) {
        return ObservedQuery {
            kind: None,
            ok: false,
            elapsed: started.elapsed(),
            response: rejection,
        };
    }
    let mut w = writer.lock().await;
    if namespace_state.is_some_and(NamespaceState::is_retired) {
        drop(w);
        return namespace_retired_observation(started);
    }
    let result = apply_drop_vector_index(&mut w, snapshot, dvi).await;
    if let Err(e) = &result {
        // A fenced/poisoned session would fail every later write; reopen it
        // in place under the lock we already hold (no-op for user errors
        // like a duplicate index name).
        if !namespace_state.is_some_and(NamespaceState::is_retired) {
            recovery::recover_writer_if_needed(&mut w, snapshot, writer_health, namespace, e).await;
        }
    }
    drop(w);
    let elapsed = started.elapsed();
    match result {
        Ok(_) => ObservedQuery {
            kind: Some(QueryKind::Write),
            ok: true,
            elapsed,
            response: Json(CypherResponse {
                columns: vec![],
                rows: vec![],
                write_outcome: None,
            })
            .into_response(),
        },
        Err(e) => {
            // A missing index (without IF EXISTS) is a user error (400); a
            // fence or lost CAS is a server-side condition (503).
            let status = match &e {
                namidb_storage::Error::Precondition(_) | namidb_storage::Error::Invariant(_) => {
                    StatusCode::BAD_REQUEST
                }
                _ => StatusCode::SERVICE_UNAVAILABLE,
            };
            ObservedQuery {
                kind: Some(QueryKind::Write),
                ok: false,
                elapsed,
                response: (
                    status,
                    Json(ErrorBody {
                        error: e.to_string(),
                    }),
                )
                    .into_response(),
            }
        }
    }
}

/// Drop a full-text index (metadata-only: descriptor + `.ft` SST refs in one
/// commit) and republish the snapshot. Shared by the HTTP and Bolt DDL paths.
#[cfg(feature = "text-index")]
async fn apply_drop_fulltext_index(
    writer: &mut WriterSession,
    snapshot: &SnapshotCell,
    dfi: &namidb_query::parser::ast::DropFulltextIndexClause,
) -> Result<u64, namidb_storage::Error> {
    let version = writer
        .drop_text_index(&dfi.name.name, dfi.if_exists)
        .await?;
    snapshot.store(writer.owned_snapshot());
    Ok(version)
}

/// HTTP shape for a `DROP INDEX` / `DROP FULLTEXT INDEX`: gate on role +
/// authz, run the DDL, return an empty `CypherResponse` on success. Mirrors
/// `run_drop_vector_index`.
#[cfg(feature = "text-index")]
// The DDL params are all distinct (session, publish cell, health, authz,
// clause, principal, clock); bundling them would not aid readability.
#[allow(clippy::too_many_arguments)]
async fn run_drop_fulltext_index(
    writer: &Arc<tokio::sync::Mutex<WriterSession>>,
    snapshot: &Arc<SnapshotCell>,
    writer_health: &Arc<WriterHealth>,
    namespace: &str,
    namespace_state: Option<&NamespaceState>,
    authz: &Arc<dyn authz::AuthzHook>,
    dfi: &namidb_query::parser::ast::DropFulltextIndexClause,
    principal: &Principal,
    started: std::time::Instant,
) -> ObservedQuery {
    if !principal.allows_write() {
        return ObservedQuery {
            kind: Some(QueryKind::Write),
            ok: false,
            elapsed: started.elapsed(),
            response: (
                StatusCode::FORBIDDEN,
                Json(ErrorBody {
                    error: "this token is read-only; schema commands are forbidden".into(),
                }),
            )
                .into_response(),
        };
    }
    let op = authz::SchemaOp::DropFulltextIndex {
        name: &dfi.name.name,
    };
    if let Err(denied) = authz.check_schema(principal, op).await {
        return ObservedQuery {
            kind: None,
            ok: false,
            elapsed: started.elapsed(),
            response: (
                StatusCode::FORBIDDEN,
                Json(ErrorBody {
                    error: denied.to_string(),
                }),
            )
                .into_response(),
        };
    }
    if let Some(rejection) = write_intake_rejection(writer_health) {
        return ObservedQuery {
            kind: None,
            ok: false,
            elapsed: started.elapsed(),
            response: rejection,
        };
    }
    let mut w = writer.lock().await;
    if namespace_state.is_some_and(NamespaceState::is_retired) {
        drop(w);
        return namespace_retired_observation(started);
    }
    let result = apply_drop_fulltext_index(&mut w, snapshot, dfi).await;
    if let Err(e) = &result {
        // A fenced/poisoned session would fail every later write; reopen it
        // in place under the lock we already hold (no-op for user errors
        // like a duplicate index name).
        if !namespace_state.is_some_and(NamespaceState::is_retired) {
            recovery::recover_writer_if_needed(&mut w, snapshot, writer_health, namespace, e).await;
        }
    }
    drop(w);
    let elapsed = started.elapsed();
    match result {
        Ok(_) => ObservedQuery {
            kind: Some(QueryKind::Write),
            ok: true,
            elapsed,
            response: Json(CypherResponse {
                columns: vec![],
                rows: vec![],
                write_outcome: None,
            })
            .into_response(),
        },
        Err(e) => {
            let status = match &e {
                namidb_storage::Error::Precondition(_) | namidb_storage::Error::Invariant(_) => {
                    StatusCode::BAD_REQUEST
                }
                _ => StatusCode::SERVICE_UNAVAILABLE,
            };
            ObservedQuery {
                kind: Some(QueryKind::Write),
                ok: false,
                elapsed,
                response: (
                    status,
                    Json(ErrorBody {
                        error: e.to_string(),
                    }),
                )
                    .into_response(),
            }
        }
    }
}

/// Apply a `CREATE CONSTRAINT … IS UNIQUE` (single- or multi-property) and
/// republish the snapshot. A metadata-only schema commit in the writer.
async fn apply_create_constraint(
    writer: &mut WriterSession,
    snapshot: &SnapshotCell,
    name: Option<&str>,
    label: &str,
    properties: &[String],
    if_not_exists: bool,
) -> Result<u64, namidb_storage::Error> {
    let version = writer
        .create_unique_constraint_named(name, label, properties, if_not_exists)
        .await?;
    snapshot.store(writer.owned_snapshot());
    Ok(version)
}

/// Apply a `CREATE INDEX … ON …` (single-property equality index) and republish
/// the snapshot. A metadata-only schema commit in the writer.
async fn apply_create_index(
    writer: &mut WriterSession,
    snapshot: &SnapshotCell,
    name: Option<&str>,
    label: &str,
    property: &str,
    if_not_exists: bool,
) -> Result<u64, namidb_storage::Error> {
    let version = writer
        .create_property_index_named(name, label, property, if_not_exists)
        .await?;
    snapshot.store(writer.owned_snapshot());
    Ok(version)
}

/// HTTP shape for `CREATE CONSTRAINT`/`CREATE INDEX`: gate on role + authz, run
/// the schema DDL, return an empty `CypherResponse`. Mirrors the vector/fulltext
/// DDL handlers. These are always-on (no Cargo feature).
#[allow(clippy::too_many_arguments)]
async fn run_create_property_ddl(
    writer: &Arc<tokio::sync::Mutex<WriterSession>>,
    snapshot: &Arc<SnapshotCell>,
    writer_health: &Arc<WriterHealth>,
    namespace: &str,
    namespace_state: Option<&NamespaceState>,
    authz: &Arc<dyn authz::AuthzHook>,
    compaction_scheduler: &Arc<CompactionScheduler>,
    metrics: &Arc<metrics::Metrics>,
    name: Option<&str>,
    label: &str,
    properties: &[String],
    unique: bool,
    if_not_exists: bool,
    principal: &Principal,
    started: std::time::Instant,
) -> ObservedQuery {
    if !principal.allows_write() {
        return ObservedQuery {
            kind: Some(QueryKind::Write),
            ok: false,
            elapsed: started.elapsed(),
            response: (
                StatusCode::FORBIDDEN,
                Json(ErrorBody {
                    error: "this token is read-only; schema commands are forbidden".into(),
                }),
            )
                .into_response(),
        };
    }
    let op = if unique {
        authz::SchemaOp::CreateConstraint { label, properties }
    } else {
        authz::SchemaOp::CreateIndex {
            label,
            property: &properties[0],
        }
    };
    if let Err(denied) = authz.check_schema(principal, op).await {
        return ObservedQuery {
            kind: None,
            ok: false,
            elapsed: started.elapsed(),
            response: (
                StatusCode::FORBIDDEN,
                Json(ErrorBody {
                    error: denied.to_string(),
                }),
            )
                .into_response(),
        };
    }
    if let Some(rejection) = write_intake_rejection(writer_health) {
        return ObservedQuery {
            kind: None,
            ok: false,
            elapsed: started.elapsed(),
            response: rejection,
        };
    }
    let mut w = writer.lock().await;
    if namespace_state.is_some_and(NamespaceState::is_retired) {
        drop(w);
        return namespace_retired_observation(started);
    }
    let result = if unique {
        apply_create_constraint(&mut w, snapshot, name, label, properties, if_not_exists).await
    } else {
        apply_create_index(&mut w, snapshot, name, label, &properties[0], if_not_exists).await
    };
    if let Err(e) = &result {
        // Same reopen-in-place as the other DDL handlers (no-op for user
        // errors like a duplicate name).
        if !namespace_state.is_some_and(NamespaceState::is_retired) {
            recovery::recover_writer_if_needed(&mut w, snapshot, writer_health, namespace, e).await;
        }
    }
    drop(w);
    let elapsed = started.elapsed();
    match result {
        Ok(_) => {
            // The schema commit alone makes `needs_compaction()` true for
            // every node SST whose descriptors lack the posting sidecar the
            // new index requires. Without this request, materialization
            // waits for the next periodic tick — or never happens when
            // periodic compaction is disabled — leaving the "index" at
            // full-scan speed on already-loaded data (item 38). An
            // IF NOT EXISTS no-op re-request is harmless: the pass gates on
            // metadata and answers Noop.
            let _ = request_compaction(
                compaction_scheduler,
                CompactionTrigger::Ddl,
                writer,
                snapshot,
                writer_health,
                namespace,
                metrics,
                None,
            );
            ObservedQuery {
                kind: Some(QueryKind::Write),
                ok: true,
                elapsed,
                response: Json(CypherResponse {
                    columns: vec![],
                    rows: vec![],
                    write_outcome: None,
                })
                .into_response(),
            }
        }
        Err(e) => {
            // A pre-existing duplicate (constraint) is a user error (400); a
            // fence/lost CAS is a server condition (503).
            let status = match &e {
                namidb_storage::Error::Precondition(_) | namidb_storage::Error::Invariant(_) => {
                    StatusCode::BAD_REQUEST
                }
                _ => StatusCode::SERVICE_UNAVAILABLE,
            };
            ObservedQuery {
                kind: Some(QueryKind::Write),
                ok: false,
                elapsed,
                response: (
                    status,
                    Json(ErrorBody {
                        error: e.to_string(),
                    }),
                )
                    .into_response(),
            }
        }
    }
}

async fn cypher(
    State(state): State<AppState>,
    Extension(principal): Extension<Principal>,
    Json(req): Json<CypherRequest>,
) -> Response {
    // The guard drops at the end of the handler, so the in-flight gauge is
    // correct even on an early error return.
    let _in_flight = state.metrics.track_in_flight();
    let obs = run_cypher(&state, &req, &principal).await;
    state
        .metrics
        .observe_query(Protocol::Http, obs.kind, obs.ok, obs.elapsed, &req.query);
    obs.response
}

/// Run one HTTP Cypher request and classify it for metrics. Mirrors the Bolt
/// `ServerBackend::run` path; the two do not share a chokepoint, so the
/// parse/plan/execute logic is intentionally parallel.
async fn run_cypher(state: &AppState, req: &CypherRequest, principal: &Principal) -> ObservedQuery {
    let started = std::time::Instant::now();

    let parsed = match cypher_parse(&req.query) {
        Ok(p) => p,
        Err(errs) => {
            let first = &errs[0];
            return ObservedQuery {
                kind: None,
                ok: false,
                elapsed: started.elapsed(),
                response: parse_error_response(format!(
                    "parse error: {} at {}",
                    first.message, first.span
                )),
            };
        }
    };

    let params = match params_from_json(&req.params) {
        Ok(p) => p,
        Err(e) => {
            return ObservedQuery {
                kind: None,
                ok: false,
                elapsed: started.elapsed(),
                response: (StatusCode::BAD_REQUEST, Json(ErrorBody { error: e })).into_response(),
            };
        }
    };

    // `CREATE VECTOR INDEX` is schema DDL: intercept before planning.
    #[cfg(feature = "vector-index")]
    if let Some(cvi) = parsed.as_create_vector_index() {
        return run_create_vector_index(
            &state.writer,
            &state.snapshot,
            &state.writer_health,
            &state.namespace,
            None,
            &state.authz,
            cvi,
            principal,
            started,
        )
        .await;
    }

    // `CREATE FULLTEXT INDEX` is schema DDL: intercept before planning.
    #[cfg(feature = "text-index")]
    if let Some(cfi) = parsed.as_create_fulltext_index() {
        return run_create_fulltext_index(
            &state.writer,
            &state.snapshot,
            &state.writer_health,
            &state.namespace,
            None,
            &state.authz,
            cfi,
            principal,
            started,
        )
        .await;
    }

    // `DROP VECTOR INDEX` is schema DDL: intercept before planning.
    #[cfg(feature = "vector-index")]
    if let Some(dvi) = parsed.as_drop_vector_index() {
        return run_drop_vector_index(
            &state.writer,
            &state.snapshot,
            &state.writer_health,
            &state.namespace,
            None,
            &state.authz,
            dvi,
            principal,
            started,
        )
        .await;
    }

    // `DROP INDEX` / `DROP FULLTEXT INDEX` is schema DDL: intercept pre-plan.
    #[cfg(feature = "text-index")]
    if let Some(dfi) = parsed.as_drop_fulltext_index() {
        return run_drop_fulltext_index(
            &state.writer,
            &state.snapshot,
            &state.writer_health,
            &state.namespace,
            None,
            &state.authz,
            dfi,
            principal,
            started,
        )
        .await;
    }

    // `CREATE CONSTRAINT` / `CREATE INDEX` are schema DDL: intercept pre-plan.
    if let Some(c) = parsed.as_create_constraint() {
        let properties: Vec<String> = c.properties.iter().map(|p| p.name.clone()).collect();
        return run_create_property_ddl(
            &state.writer,
            &state.snapshot,
            &state.writer_health,
            &state.namespace,
            None,
            &state.authz,
            &state.compaction_scheduler,
            &state.metrics,
            c.name.as_ref().map(|n| n.name.as_str()),
            &c.label.name,
            &properties,
            true,
            c.if_not_exists,
            principal,
            started,
        )
        .await;
    }
    if let Some(c) = parsed.as_create_index() {
        let properties = [c.property.name.clone()];
        return run_create_property_ddl(
            &state.writer,
            &state.snapshot,
            &state.writer_health,
            &state.namespace,
            None,
            &state.authz,
            &state.compaction_scheduler,
            &state.metrics,
            c.name.as_ref().map(|n| n.name.as_str()),
            &c.label.name,
            &properties,
            false,
            c.if_not_exists,
            principal,
            started,
        )
        .await;
    }

    // `SHOW CONSTRAINTS` / `SHOW INDEXES` are schema introspection: answer them
    // from the published manifest without planning or a writer lock.
    if let Some(c) = parsed.as_show_schema() {
        let owned = state.snapshot.load();
        let manifest = &owned.manifest().manifest;
        let rows = match c.kind {
            namidb_query::parser::ast::ShowKind::Constraints => {
                namidb_query::show_constraints_rows(&manifest.schema)
            }
            namidb_query::parser::ast::ShowKind::Indexes => {
                namidb_query::show_indexes_rows(manifest)
            }
        };
        let (_columns, json_rows) = rows_to_json(&rows);
        let columns = namidb_query::show_schema_columns();
        return ObservedQuery {
            kind: Some(QueryKind::Read),
            ok: true,
            elapsed: started.elapsed(),
            response: Json(CypherResponse {
                columns,
                rows: json_rows,
                write_outcome: None,
            })
            .into_response(),
        };
    }

    // Plan against the latest published snapshot — no writer lock yet.
    let owned = state.snapshot.load();
    let catalog = state.catalog_for(&owned.manifest().manifest);
    let plan = match build_plan(&parsed, &catalog) {
        Ok(p) => p,
        Err(e) => {
            return ObservedQuery {
                kind: None,
                ok: false,
                elapsed: started.elapsed(),
                response: plan_error_response(&e),
            };
        }
    };

    // Pre-execution authorization hook (RFC-015 Wave B): a policy may deny the
    // request based on the principal + plan, before the writer lock or any
    // execution. NoOp by default (allow-all), so this is behavior-preserving.
    if let Err(denied) = state.authz.check(principal, &plan).await {
        return ObservedQuery {
            kind: None,
            ok: false,
            elapsed: started.elapsed(),
            response: (
                StatusCode::FORBIDDEN,
                Json(ErrorBody {
                    error: denied.to_string(),
                }),
            )
                .into_response(),
        };
    }

    if parsed.explain {
        return explain_observation(&parsed, &plan, &owned, &catalog, started);
    }

    if plan.contains_write() {
        // A read-only token may not write. Reject before taking the writer
        // lock so a forbidden write costs nothing.
        if !principal.allows_write() {
            return ObservedQuery {
                kind: Some(QueryKind::Write),
                ok: false,
                elapsed: started.elapsed(),
                response: (
                    StatusCode::FORBIDDEN,
                    Json(ErrorBody {
                        error: "this token is read-only; write queries are forbidden".into(),
                    }),
                )
                    .into_response(),
            };
        }
        if let Some(rejection) = write_intake_rejection(&state.writer_health) {
            return ObservedQuery {
                kind: Some(QueryKind::Write),
                ok: false,
                elapsed: started.elapsed(),
                response: rejection,
            };
        }
        let Some(mut writer) = state.lock_writer_bounded(WriterLockKind::Http).await else {
            return ObservedQuery {
                kind: Some(QueryKind::Write),
                ok: false,
                elapsed: started.elapsed(),
                response: writer_busy_response(),
            };
        };
        let result =
            execute_write_with_deadline(&plan, &mut writer, &params, state.write_deadline()).await;
        // Sample the soft write-stall decision while still holding the lock
        // (RFC-027 P5), then release it and sleep — backpressure applies to
        // this request, not to the writer mutex other connections need.
        let stall = match &result {
            Ok(_) => {
                // Refresh the published snapshot so subsequent reads see the
                // just-committed records (RFC-021).
                state.snapshot.store(writer.owned_snapshot());
                state.after_commit_backpressure(&writer)
            }
            Err(e) => {
                // A fenced/poisoned session would fail every later write;
                // reopen it in place under the lock we already hold.
                recovery::recover_after_write_error(
                    &mut writer,
                    &state.snapshot,
                    &state.writer_health,
                    &state.namespace,
                    e,
                )
                .await;
                None
            }
        };
        drop(writer);
        // Stop the clock before the backpressure sleep: the stall is
        // intentional throttling, not query cost, so it must not inflate the
        // latency histogram or trip the slow-query log.
        let elapsed = started.elapsed();
        if let Some(delay) = stall {
            tokio::time::sleep(delay).await;
        }
        match result {
            Ok(outcome) => {
                let summary = WriteSummary::from(&outcome);
                let (columns, rows) = rows_to_json(&outcome.rows);
                ObservedQuery {
                    kind: Some(QueryKind::Write),
                    ok: true,
                    elapsed,
                    response: Json(CypherResponse {
                        columns,
                        rows,
                        write_outcome: Some(summary),
                    })
                    .into_response(),
                }
            }
            Err(e) => ObservedQuery {
                kind: Some(QueryKind::Write),
                ok: false,
                elapsed,
                response: exec_failure_response("write execution failed", &e),
            },
        }
    } else {
        // Read path: no writer lock. Borrow a short-lived `Snapshot`
        // from the owned one; the `OwnedSnapshot` Arc keeps the
        // underlying memtable alive for the duration of the query.
        let _scan_permit = acquire_scan_permit(&plan).await;
        let snap = owned.borrow();
        let result = execute_with_limits(
            &plan,
            &snap,
            &params,
            state.query_deadline(),
            state.query_row_cap(),
        )
        .await;
        let elapsed = started.elapsed();
        match result {
            Ok(rows) => {
                let (columns, rows) = rows_to_json(&rows);
                ObservedQuery {
                    kind: Some(QueryKind::Read),
                    ok: true,
                    elapsed,
                    response: Json(CypherResponse {
                        columns,
                        rows,
                        write_outcome: None,
                    })
                    .into_response(),
                }
            }
            Err(e) => ObservedQuery {
                kind: Some(QueryKind::Read),
                ok: false,
                elapsed,
                response: exec_failure_response("read execution failed", &e),
            },
        }
    }
}

#[derive(Serialize)]
struct FlushResponse {
    ssts_written: usize,
    bloom_sidecars_written: usize,
    manifest_version: u64,
}

async fn admin_flush(
    State(state): State<AppState>,
    Extension(principal): Extension<Principal>,
) -> Response {
    // Admin flush is an operator maintenance action (no Cypher, no plan), so it
    // is intentionally gated by role only — not by the AuthzHook, which decides
    // on a `LogicalPlan`. A flush touches no user data the way a query does;
    // restricting who may operate the server is the deployment's concern (mTLS /
    // network ACL on the admin route), consistent with how `/v0/admin/*` is
    // treated. A read-only token may not trigger it.
    if !principal.allows_write() {
        return (
            StatusCode::FORBIDDEN,
            Json(ErrorBody {
                error: "this token is read-only; admin flush is forbidden".into(),
            }),
        )
            .into_response();
    }

    // The storage flush owns an RAII restore guard from `memtable.freeze()`
    // until manifest success. A disconnected client therefore cancels this
    // future safely and, unlike a detached task, cannot leave an unbounded
    // queue of hidden flush waiters behind the process-wide semaphore.
    run_admin_flush(state).await
}

/// Bound for the admin-flush waits (the process-wide flush permit and the
/// writer mutex). This route is excluded from the request timeout, so an
/// unbounded wait pins one global HTTP concurrency slot per queued client —
/// enough stuck clients exhaust the cap and starve reads process-wide (the
/// disk-full field outage, item 40). A 503 keeps the slot economy honest.
const ADMIN_FLUSH_WAIT: Duration = Duration::from_secs(30);

fn admin_flush_busy_response() -> Response {
    (
        StatusCode::SERVICE_UNAVAILABLE,
        Json(ErrorBody {
            error: "another flush is already running; retry shortly".into(),
        }),
    )
        .into_response()
}

async fn run_admin_flush(state: AppState) -> Response {
    let Ok(flush_permit) =
        tokio::time::timeout(ADMIN_FLUSH_WAIT, state.memory.admin_flush_permit()).await
    else {
        return admin_flush_busy_response();
    };
    let bound = if state.writer_lock_timeout.is_zero() {
        ADMIN_FLUSH_WAIT
    } else {
        state.writer_lock_timeout
    };
    let Some(mut w) = lock_writer_bounded(&state.writer, bound).await else {
        return writer_busy_response();
    };
    let schema = w.snapshot().manifest().manifest.schema.clone();
    match w.flush(schema).await {
        Ok(outcome) => {
            state.writer_health.clear_persistence_degraded();
            state.snapshot.store(w.owned_snapshot());
            state
                .memtable_bytes_gauge
                .store(w.memtable_bytes(), std::sync::atomic::Ordering::Relaxed);
            let response = FlushResponse {
                ssts_written: outcome.ssts_written,
                bloom_sidecars_written: outcome.bloom_sidecars_written,
                manifest_version: outcome.committed.manifest.version,
            };
            // A flush releases the memtable and its transient Arrow/Parquet
            // allocations, but glibc can retain their now-free arenas in the
            // process RSS indefinitely. Drop the writer lock before asking the
            // allocator to return wholly free pages, and keep that potentially
            // expensive operation off the async serving workers.
            drop(w);
            if flush_needs_allocator_trim(outcome.ssts_written) {
                trim_allocator_after_flush(Arc::clone(&state.memory), flush_permit).await;
            }
            Json(response).into_response()
        }
        Err(e) => {
            recovery::recover_writer_if_needed(
                &mut w,
                &state.snapshot,
                &state.writer_health,
                &state.namespace,
                &e,
            )
            .await;
            if e.is_local_persistence() {
                let reason = persistence_degraded_reason(&e);
                state
                    .writer_health
                    .mark_persistence_degraded(reason.clone());
                return persistence_degraded_response(&reason);
            }
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorBody {
                    error: format!("flush failed: {e}"),
                }),
            )
                .into_response()
        }
    }
}

fn flush_needs_allocator_trim(ssts_written: usize) -> bool {
    ssts_written != 0
}

async fn trim_allocator_after_flush(
    memory: Arc<memory::MemoryGovernor>,
    flush_permit: tokio::sync::OwnedSemaphorePermit,
) {
    // Move the process-wide flush permit into the blocking closure. If the
    // client disconnects while this JoinHandle is awaited, Tokio detaches the
    // already-running blocking job; ownership here keeps a second flush/trim
    // from overlapping it until malloc_trim has really completed.
    let trim = tokio::task::spawn_blocking(move || {
        let _flush_permit = flush_permit;
        memory::trim_allocator();
        let _ = memory.sample();
    });
    if let Err(error) = trim.await {
        tracing::warn!(%error, "post-flush allocator trim task failed");
    }
}

// ── multi-tenant handlers ─────────────────────────────────────────────

/// Liveness probe in multi-tenant mode. Same as single-tenant: no lock,
/// no namespace state.
async fn livez_multi() -> impl IntoResponse {
    Json(serde_json::json!({ "status": "ok" }))
}

/// Readiness probe in multi-tenant mode. Returns the namespace's manifest
/// version and epoch.
async fn health_multi(
    Path(namespace): Path<String>,
    State(shared): State<SharedAppState>,
) -> Response {
    dispatch_health_multi(&shared, namespace).await
}

/// Unprefixed readiness probe: resolve the namespace from the
/// `X-NamiDB-Namespace` header (or default).
async fn health_multi_unprefixed(
    State(shared): State<SharedAppState>,
    headers: axum::http::HeaderMap,
) -> Response {
    dispatch_health_multi(&shared, namespace_from_header(&shared, &headers)).await
}

async fn dispatch_health_multi(shared: &SharedAppState, namespace: String) -> Response {
    let _ = shared.memory.sample();
    if shared.memory.over_limit() {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(ErrorBody {
                error: format!(
                    "process memory pressure: resident {} bytes reached configured maximum {} \
                     bytes; namespace health work is paused until memory falls",
                    shared.memory.resident_bytes(),
                    shared.memory.max_bytes()
                ),
            }),
        )
            .into_response();
    }
    match shared.registry.get_or_open(&namespace).await {
        Ok(ns_state) => {
            // Opening/recovery can itself move RSS over the limit. Refresh the
            // gauge so readiness reports that immediately.
            let _ = shared.memory.sample();
            let owned = ns_state.snapshot.load();
            let m = &owned.manifest().manifest;
            health_response(namespace, m, &ns_state.writer_health, None, &shared.memory)
        }
        Err(e) => (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(ErrorBody {
                error: e.to_string(),
            }),
        )
            .into_response(),
    }
}

/// Execute a Cypher query in multi-tenant mode.
async fn cypher_multi(
    Path(namespace): Path<String>,
    State(shared): State<SharedAppState>,
    Extension(principal): Extension<Principal>,
    Json(req): Json<CypherRequest>,
) -> Response {
    dispatch_cypher_multi(&shared, &namespace, &principal, req).await
}

/// Unprefixed entry point: resolve the namespace from the
/// `X-NamiDB-Namespace` header (or the default), then run the query. Used by
/// the `/v0/cypher` route in multi-tenant mode so clients can target a
/// namespace without a path prefix.
async fn cypher_multi_unprefixed(
    State(shared): State<SharedAppState>,
    Extension(principal): Extension<Principal>,
    headers: axum::http::HeaderMap,
    Json(req): Json<CypherRequest>,
) -> Response {
    let namespace = namespace_from_header(&shared, &headers);
    dispatch_cypher_multi(&shared, &namespace, &principal, req).await
}

/// Shared body of the multi-tenant cypher handler: open the namespace, run,
/// observe metrics.
async fn dispatch_cypher_multi(
    shared: &SharedAppState,
    namespace: &str,
    principal: &Principal,
    req: CypherRequest,
) -> Response {
    let _in_flight = shared.metrics.track_in_flight();

    // Get or create the namespace state.
    let ns_state = match shared.registry.get_or_open(namespace).await {
        Ok(ns) => ns,
        Err(e) => {
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(ErrorBody {
                    error: e.to_string(),
                }),
            )
                .into_response();
        }
    };

    let obs = run_cypher_multi(&ns_state, shared, &req, principal).await;
    shared
        .metrics
        .observe_query(Protocol::Http, obs.kind, obs.ok, obs.elapsed, &req.query);
    obs.response
}

/// Run one HTTP Cypher request in multi-tenant mode.
async fn run_cypher_multi(
    ns_state: &NamespaceState,
    shared: &SharedAppState,
    req: &CypherRequest,
    principal: &Principal,
) -> ObservedQuery {
    let started = std::time::Instant::now();

    let parsed = match cypher_parse(&req.query) {
        Ok(p) => p,
        Err(errs) => {
            let first = &errs[0];
            return ObservedQuery {
                kind: None,
                ok: false,
                elapsed: started.elapsed(),
                response: parse_error_response(format!(
                    "parse error: {} at {}",
                    first.message, first.span
                )),
            };
        }
    };

    let params = match params_from_json(&req.params) {
        Ok(p) => p,
        Err(e) => {
            return ObservedQuery {
                kind: None,
                ok: false,
                elapsed: started.elapsed(),
                response: (StatusCode::BAD_REQUEST, Json(ErrorBody { error: e })).into_response(),
            };
        }
    };

    // `CREATE VECTOR INDEX` is schema DDL: intercept before planning.
    #[cfg(feature = "vector-index")]
    if let Some(cvi) = parsed.as_create_vector_index() {
        return run_create_vector_index(
            &ns_state.writer,
            &ns_state.snapshot,
            &ns_state.writer_health,
            &ns_state.namespace,
            Some(ns_state),
            &shared.authz,
            cvi,
            principal,
            started,
        )
        .await;
    }

    // `CREATE FULLTEXT INDEX` is schema DDL: intercept before planning.
    #[cfg(feature = "text-index")]
    if let Some(cfi) = parsed.as_create_fulltext_index() {
        return run_create_fulltext_index(
            &ns_state.writer,
            &ns_state.snapshot,
            &ns_state.writer_health,
            &ns_state.namespace,
            Some(ns_state),
            &shared.authz,
            cfi,
            principal,
            started,
        )
        .await;
    }

    // `DROP VECTOR INDEX` is schema DDL: intercept before planning.
    #[cfg(feature = "vector-index")]
    if let Some(dvi) = parsed.as_drop_vector_index() {
        return run_drop_vector_index(
            &ns_state.writer,
            &ns_state.snapshot,
            &ns_state.writer_health,
            &ns_state.namespace,
            Some(ns_state),
            &shared.authz,
            dvi,
            principal,
            started,
        )
        .await;
    }

    // `DROP INDEX` / `DROP FULLTEXT INDEX` is schema DDL: intercept pre-plan.
    #[cfg(feature = "text-index")]
    if let Some(dfi) = parsed.as_drop_fulltext_index() {
        return run_drop_fulltext_index(
            &ns_state.writer,
            &ns_state.snapshot,
            &ns_state.writer_health,
            &ns_state.namespace,
            Some(ns_state),
            &shared.authz,
            dfi,
            principal,
            started,
        )
        .await;
    }

    // `CREATE CONSTRAINT` / `CREATE INDEX` are schema DDL: intercept pre-plan.
    if let Some(c) = parsed.as_create_constraint() {
        let properties: Vec<String> = c.properties.iter().map(|p| p.name.clone()).collect();
        return run_create_property_ddl(
            &ns_state.writer,
            &ns_state.snapshot,
            &ns_state.writer_health,
            &ns_state.namespace,
            Some(ns_state),
            &shared.authz,
            &ns_state.compaction_scheduler,
            &shared.metrics,
            c.name.as_ref().map(|n| n.name.as_str()),
            &c.label.name,
            &properties,
            true,
            c.if_not_exists,
            principal,
            started,
        )
        .await;
    }
    if let Some(c) = parsed.as_create_index() {
        let properties = [c.property.name.clone()];
        return run_create_property_ddl(
            &ns_state.writer,
            &ns_state.snapshot,
            &ns_state.writer_health,
            &ns_state.namespace,
            Some(ns_state),
            &shared.authz,
            &ns_state.compaction_scheduler,
            &shared.metrics,
            c.name.as_ref().map(|n| n.name.as_str()),
            &c.label.name,
            &properties,
            false,
            c.if_not_exists,
            principal,
            started,
        )
        .await;
    }

    // `SHOW CONSTRAINTS` / `SHOW INDEXES`: schema introspection from the
    // published manifest (a read; no writer lock).
    if let Some(c) = parsed.as_show_schema() {
        let owned = ns_state.snapshot.load();
        let manifest = &owned.manifest().manifest;
        let rows = match c.kind {
            namidb_query::parser::ast::ShowKind::Constraints => {
                namidb_query::show_constraints_rows(&manifest.schema)
            }
            namidb_query::parser::ast::ShowKind::Indexes => {
                namidb_query::show_indexes_rows(manifest)
            }
        };
        let (_columns, json_rows) = rows_to_json(&rows);
        return ObservedQuery {
            kind: Some(QueryKind::Read),
            ok: true,
            elapsed: started.elapsed(),
            response: Json(CypherResponse {
                columns: namidb_query::show_schema_columns(),
                rows: json_rows,
                write_outcome: None,
            })
            .into_response(),
        };
    }

    // Plan against the latest published snapshot. The optimizer catalog is
    // memoised per manifest version on the namespace state (building it is
    // O(ssts)), so a read-heavy namespace does not rebuild it every query.
    let owned = ns_state.snapshot.load();
    let catalog = ns_state.catalog_for(&owned.manifest().manifest);
    let plan = match build_plan(&parsed, &catalog) {
        Ok(p) => p,
        Err(e) => {
            return ObservedQuery {
                kind: None,
                ok: false,
                elapsed: started.elapsed(),
                response: plan_error_response(&e),
            };
        }
    };

    // Pre-execution authorization hook (RFC-015 Wave B); NoOp by default.
    if let Err(denied) = shared.authz.check(principal, &plan).await {
        return ObservedQuery {
            kind: None,
            ok: false,
            elapsed: started.elapsed(),
            response: (
                StatusCode::FORBIDDEN,
                Json(ErrorBody {
                    error: denied.to_string(),
                }),
            )
                .into_response(),
        };
    }

    if parsed.explain {
        return explain_observation(&parsed, &plan, &owned, &catalog, started);
    }

    if plan.contains_write() {
        // A read-only token may not write.
        if !principal.allows_write() {
            return ObservedQuery {
                kind: Some(QueryKind::Write),
                ok: false,
                elapsed: started.elapsed(),
                response: (
                    StatusCode::FORBIDDEN,
                    Json(ErrorBody {
                        error: "this token is read-only; write queries are forbidden".into(),
                    }),
                )
                    .into_response(),
            };
        }
        if let Some(rejection) = write_intake_rejection(&ns_state.writer_health) {
            return ObservedQuery {
                kind: Some(QueryKind::Write),
                ok: false,
                elapsed: started.elapsed(),
                response: rejection,
            };
        }
        let lock_started = std::time::Instant::now();
        let writer = lock_writer_bounded(&ns_state.writer, shared.writer_lock_timeout).await;
        shared.metrics.observe_writer_lock(
            WriterLockKind::Http,
            lock_started.elapsed(),
            writer.is_some(),
        );
        let Some(mut writer) = writer else {
            return ObservedQuery {
                kind: Some(QueryKind::Write),
                ok: false,
                elapsed: started.elapsed(),
                response: writer_busy_response(),
            };
        };
        // Eviction marks the incarnation retired before it waits for this
        // mutex. Revalidate only after acquisition: an Arc cloned before
        // eviction may have spent arbitrary time in the mutex's FIFO queue.
        if ns_state.is_retired() {
            drop(writer);
            return namespace_retired_observation(started);
        }
        let result =
            execute_write_with_deadline(&plan, &mut writer, &params, shared.write_deadline()).await;
        let stall = match &result {
            Ok(_) => {
                ns_state.snapshot.store(writer.owned_snapshot());
                let bytes = writer.memtable_bytes();
                if shared.memtable_flush_bytes > 0 && bytes >= shared.memtable_flush_bytes {
                    ns_state.flush_notify.notify_one();
                }
                shared.write_stall_for(writer.max_l0_bucket_len(), bytes)
            }
            Err(e) => {
                // Reopen a fenced/poisoned namespace writer in place, under
                // the lock we already hold (mirrors the single-tenant path).
                // Never recover a retired incarnation: doing so after
                // eviction could claim a newer epoch and fence its successor.
                if !ns_state.is_retired() {
                    recovery::recover_after_write_error(
                        &mut writer,
                        &ns_state.snapshot,
                        &ns_state.writer_health,
                        &ns_state.namespace,
                        e,
                    )
                    .await;
                }
                None
            }
        };
        drop(writer);
        let elapsed = started.elapsed();
        if let Some(delay) = stall {
            tokio::time::sleep(delay).await;
        }
        match result {
            Ok(outcome) => {
                let summary = WriteSummary::from(&outcome);
                let (columns, rows) = rows_to_json(&outcome.rows);
                ObservedQuery {
                    kind: Some(QueryKind::Write),
                    ok: true,
                    elapsed,
                    response: Json(CypherResponse {
                        columns,
                        rows,
                        write_outcome: Some(summary),
                    })
                    .into_response(),
                }
            }
            Err(e) => ObservedQuery {
                kind: Some(QueryKind::Write),
                ok: false,
                elapsed,
                response: exec_failure_response("write execution failed", &e),
            },
        }
    } else {
        // Read path.
        let _scan_permit = acquire_scan_permit(&plan).await;
        let snap = owned.borrow();
        let result = execute_with_limits(
            &plan,
            &snap,
            &params,
            shared.query_deadline(),
            shared.query_row_cap(),
        )
        .await;
        let elapsed = started.elapsed();
        match result {
            Ok(rows) => {
                let (columns, rows) = rows_to_json(&rows);
                ObservedQuery {
                    kind: Some(QueryKind::Read),
                    ok: true,
                    elapsed,
                    response: Json(CypherResponse {
                        columns,
                        rows,
                        write_outcome: None,
                    })
                    .into_response(),
                }
            }
            Err(e) => ObservedQuery {
                kind: Some(QueryKind::Read),
                ok: false,
                elapsed,
                response: exec_failure_response("read execution failed", &e),
            },
        }
    }
}

/// Admin flush in multi-tenant mode.
async fn admin_flush_multi(
    Path(namespace): Path<String>,
    State(shared): State<SharedAppState>,
    Extension(principal): Extension<Principal>,
) -> Response {
    dispatch_admin_flush_multi(&shared, &namespace, &principal).await
}

/// Unprefixed admin flush: resolve namespace from header/default.
async fn admin_flush_multi_unprefixed(
    State(shared): State<SharedAppState>,
    Extension(principal): Extension<Principal>,
    headers: axum::http::HeaderMap,
) -> Response {
    let namespace = namespace_from_header(&shared, &headers);
    dispatch_admin_flush_multi(&shared, &namespace, &principal).await
}

async fn dispatch_admin_flush_multi(
    shared: &SharedAppState,
    namespace: &str,
    principal: &Principal,
) -> Response {
    if !principal.allows_write() {
        return (
            StatusCode::FORBIDDEN,
            Json(ErrorBody {
                error: "this token is read-only; admin flush is forbidden".into(),
            }),
        )
            .into_response();
    }

    run_admin_flush_multi(shared.clone(), namespace.to_string()).await
}

async fn run_admin_flush_multi(shared: SharedAppState, namespace: String) -> Response {
    let Ok(flush_permit) =
        tokio::time::timeout(ADMIN_FLUSH_WAIT, shared.memory.admin_flush_permit()).await
    else {
        return admin_flush_busy_response();
    };
    // Refresh the gauge because this route intentionally bypasses query
    // admission. At the hard limit, only an already-open namespace can own a
    // memtable worth flushing; recovering a cold one would allocate memory
    // while providing no relief.
    let _ = shared.memory.sample();
    let ns_state = if shared.memory.over_limit() {
        match shared.registry.get_if_open(&namespace).await {
            Some(ns) => ns,
            None => {
                return (
                    StatusCode::SERVICE_UNAVAILABLE,
                    Json(ErrorBody {
                        error: format!(
                            "process memory pressure: namespace '{namespace}' is not open; \
                             refusing to recover a cold writer for admin flush"
                        ),
                    }),
                )
                    .into_response();
            }
        }
    } else {
        match shared.registry.get_or_open(&namespace).await {
            Ok(ns) => ns,
            Err(e) => {
                return (
                    StatusCode::SERVICE_UNAVAILABLE,
                    Json(ErrorBody {
                        error: e.to_string(),
                    }),
                )
                    .into_response();
            }
        }
    };
    let bound = if shared.writer_lock_timeout.is_zero() {
        ADMIN_FLUSH_WAIT
    } else {
        shared.writer_lock_timeout
    };
    let Some(mut w) = lock_writer_bounded(&ns_state.writer, bound).await else {
        return writer_busy_response();
    };
    // Same post-lock incarnation check as normal writes and schema DDL.
    if ns_state.is_retired() {
        drop(w);
        return namespace_retired_response();
    }
    let schema = w.snapshot().manifest().manifest.schema.clone();
    match w.flush(schema).await {
        Ok(outcome) => {
            ns_state.writer_health.clear_persistence_degraded();
            ns_state.snapshot.store(w.owned_snapshot());
            let response = FlushResponse {
                ssts_written: outcome.ssts_written,
                bloom_sidecars_written: outcome.bloom_sidecars_written,
                manifest_version: outcome.committed.manifest.version,
            };
            drop(w);
            if flush_needs_allocator_trim(outcome.ssts_written) {
                trim_allocator_after_flush(Arc::clone(&shared.memory), flush_permit).await;
            }
            Json(response).into_response()
        }
        Err(e) => {
            if !ns_state.is_retired() {
                recovery::recover_writer_if_needed(
                    &mut w,
                    &ns_state.snapshot,
                    &ns_state.writer_health,
                    &ns_state.namespace,
                    &e,
                )
                .await;
            }
            if e.is_local_persistence() {
                let reason = persistence_degraded_reason(&e);
                ns_state
                    .writer_health
                    .mark_persistence_degraded(reason.clone());
                return persistence_degraded_response(&reason);
            }
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorBody {
                    error: format!("flush failed: {e}"),
                }),
            )
                .into_response()
        }
    }
}

/// Prometheus metrics handler in multi-tenant mode.
async fn metrics_handler_multi(State(shared): State<SharedAppState>) -> impl IntoResponse {
    let _ = shared.memory.sample();
    (
        [(
            axum::http::header::CONTENT_TYPE,
            HeaderValue::from_static("text/plain; version=0.0.4; charset=utf-8"),
        )],
        shared.metrics.render_with_memory(&shared.memory),
    )
}

// ── value <-> json conversions ────────────────────────────────────────

fn params_from_json(m: &serde_json::Map<String, serde_json::Value>) -> Result<Params, String> {
    let mut params = Params::new();
    for (k, v) in m {
        let rv = json_to_runtime(v)?;
        params.insert(k.clone(), rv);
    }
    Ok(params)
}

fn json_to_runtime(v: &serde_json::Value) -> Result<RuntimeValue, String> {
    use serde_json::Value::*;
    Ok(match v {
        Null => RuntimeValue::Null,
        Bool(b) => RuntimeValue::Bool(*b),
        Number(n) => {
            if let Some(i) = n.as_i64() {
                RuntimeValue::Integer(i)
            } else if n.is_u64() {
                // A u64 beyond i64::MAX would silently degrade to a lossy
                // float; Cypher integers are 64-bit signed, so reject it.
                return Err(format!("integer param {n} exceeds the 64-bit signed range"));
            } else if let Some(f) = n.as_f64() {
                RuntimeValue::Float(f)
            } else {
                return Err(format!("unsupported numeric param: {n}"));
            }
        }
        String(s) => RuntimeValue::String(s.clone()),
        Array(a) => {
            let mut out = Vec::with_capacity(a.len());
            for item in a {
                out.push(json_to_runtime(item)?);
            }
            RuntimeValue::List(out)
        }
        Object(o) => {
            let mut out = std::collections::BTreeMap::new();
            for (k, val) in o {
                out.insert(k.clone(), json_to_runtime(val)?);
            }
            RuntimeValue::Map(out)
        }
    })
}

fn rows_to_json(
    rows: &[namidb_query::Row],
) -> (Vec<String>, Vec<serde_json::Map<String, serde_json::Value>>) {
    let columns: Vec<String> = rows
        .first()
        .map(|r| r.bindings.keys().cloned().collect())
        .unwrap_or_default();
    let json_rows: Vec<_> = rows
        .iter()
        .map(|r| {
            r.bindings
                .iter()
                .map(|(k, v)| (k.clone(), runtime_to_json(v)))
                .collect::<serde_json::Map<_, _>>()
        })
        .collect();
    (columns, json_rows)
}

fn runtime_to_json(v: &RuntimeValue) -> serde_json::Value {
    use serde_json::Value as J;
    match v {
        RuntimeValue::Null => J::Null,
        RuntimeValue::Bool(b) => J::Bool(*b),
        RuntimeValue::Integer(n) => J::Number((*n).into()),
        RuntimeValue::Float(f) => serde_json::Number::from_f64(*f)
            .map(J::Number)
            .unwrap_or(J::Null),
        RuntimeValue::String(s) => J::String(s.clone()),
        RuntimeValue::Bytes(b) => {
            use base64::Engine as _;
            J::String(base64::engine::general_purpose::STANDARD.encode(b))
        }
        RuntimeValue::Vector(v) => J::Array(
            v.iter()
                .map(|x| {
                    serde_json::Number::from_f64(*x as f64)
                        .map(J::Number)
                        .unwrap_or(J::Null)
                })
                .collect(),
        ),
        // Dequantize int8 to floats so HTTP clients see a float vector.
        RuntimeValue::Vector8 { codes, scale } => J::Array(
            codes
                .iter()
                .map(|&c| {
                    serde_json::Number::from_f64(c as f64 * *scale as f64)
                        .map(J::Number)
                        .unwrap_or(J::Null)
                })
                .collect(),
        ),
        RuntimeValue::List(items) => J::Array(items.iter().map(runtime_to_json).collect()),
        RuntimeValue::Map(m) => J::Object(
            m.iter()
                .map(|(k, v)| (k.clone(), runtime_to_json(v)))
                .collect(),
        ),
        RuntimeValue::Date(d) => J::String(d.to_string()),
        RuntimeValue::DateTime(micros) => {
            chrono::DateTime::<chrono::Utc>::from_timestamp_micros(*micros)
                .map(|dt| J::String(dt.to_rfc3339()))
                .unwrap_or(J::Null)
        }
        RuntimeValue::Node(n) => {
            let mut o = serde_json::Map::new();
            o.insert("_kind".into(), J::String("node".into()));
            o.insert("id".into(), J::String(n.id.to_string()));
            // `label` = representative (first) for back-compat; `labels` = set.
            o.insert(
                "label".into(),
                J::String(n.labels.iter().next().cloned().unwrap_or_default()),
            );
            o.insert(
                "labels".into(),
                J::Array(n.labels.iter().map(|l| J::String(l.clone())).collect()),
            );
            let props: serde_json::Map<String, J> = n
                .properties
                .iter()
                .map(|(k, v)| (k.clone(), runtime_to_json(v)))
                .collect();
            o.insert("properties".into(), J::Object(props));
            J::Object(o)
        }
        RuntimeValue::Rel(r) => {
            let mut o = serde_json::Map::new();
            o.insert("_kind".into(), J::String("rel".into()));
            o.insert("edge_type".into(), J::String(r.edge_type.clone()));
            o.insert("src".into(), J::String(r.src.to_string()));
            o.insert("dst".into(), J::String(r.dst.to_string()));
            let props: serde_json::Map<String, J> = r
                .properties
                .iter()
                .map(|(k, v)| (k.clone(), runtime_to_json(v)))
                .collect();
            o.insert("properties".into(), J::Object(props));
            J::Object(o)
        }
        RuntimeValue::Path(items) => J::Array(items.iter().map(runtime_to_json).collect()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::{to_bytes, Body};
    use axum::http::{Request, StatusCode};
    use tower::ServiceExt;

    async fn fixture(auth_token: Option<&str>) -> Router {
        let (store, paths) = namidb_storage::parse_uri("memory://test").unwrap();
        let writer = WriterSession::open(store, paths).await.unwrap();
        let state = AppState::new(writer, auth_token.map(|s| s.to_string()), "test".into());
        build_router(state)
    }

    /// Plan item 28: the HTTP JSON parameter route had no unit tests and no
    /// HTTP test posted a non-empty params map.
    #[test]
    fn json_params_convert_nested_shapes_and_reject_out_of_range_integers() {
        let map = serde_json::json!({
            "nested": {"list": [1, 2.5, "x", null, true]},
            "imax": i64::MAX,
            "imin": i64::MIN,
            "tenth": 0.1,
        });
        let params = params_from_json(map.as_object().unwrap()).unwrap();
        match params.get("nested") {
            Some(RuntimeValue::Map(m)) => match m.get("list") {
                Some(RuntimeValue::List(items)) => {
                    assert_eq!(items.len(), 5);
                    assert!(matches!(items[0], RuntimeValue::Integer(1)));
                    assert!(matches!(items[1], RuntimeValue::Float(f) if f == 2.5));
                    assert!(matches!(&items[2], RuntimeValue::String(s) if s == "x"));
                    assert!(matches!(items[3], RuntimeValue::Null));
                    assert!(matches!(items[4], RuntimeValue::Bool(true)));
                }
                other => panic!("nested list must survive: {other:?}"),
            },
            other => panic!("nested map must survive: {other:?}"),
        }
        assert!(matches!(params.get("imax"), Some(RuntimeValue::Integer(i)) if *i == i64::MAX));
        assert!(matches!(params.get("imin"), Some(RuntimeValue::Integer(i)) if *i == i64::MIN));
        assert!(
            matches!(params.get("tenth"), Some(RuntimeValue::Float(f)) if *f == 0.1),
            "0.1 must round-trip bit-exact through serde_json"
        );

        let oversized = serde_json::json!({"big": u64::MAX});
        let error = params_from_json(oversized.as_object().unwrap()).unwrap_err();
        assert!(
            error.contains("64-bit signed range"),
            "a u64 beyond i64::MAX must be rejected, not degraded to a float: {error}"
        );
    }

    #[tokio::test]
    async fn http_cypher_executes_with_a_non_empty_params_map() {
        let app = fixture(None).await;
        let body = serde_json::to_vec(&serde_json::json!({
            "query": "RETURN $flag AS flag, $nums AS nums, $meta AS meta, $tenth AS tenth",
            "params": {
                "flag": true,
                "nums": [1, 2, 3],
                "meta": {"tenant": "acme"},
                "tenth": 0.1,
            }
        }))
        .unwrap();
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v0/cypher")
                    .header("content-type", "application/json")
                    .header(axum::http::header::CONTENT_LENGTH, body.len().to_string())
                    .body(Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let bytes = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let parsed: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        let row = &parsed["rows"][0];
        assert_eq!(row["flag"], serde_json::json!(true));
        assert_eq!(row["nums"], serde_json::json!([1, 2, 3]));
        assert_eq!(row["meta"], serde_json::json!({"tenant": "acme"}));
        assert_eq!(row["tenth"], serde_json::json!(0.1), "float round-trip");
    }

    #[tokio::test]
    async fn http_cypher_rejects_an_out_of_range_integer_param_with_400() {
        let app = fixture(None).await;
        let body = format!(
            "{{\"query\": \"RETURN $big AS big\", \"params\": {{\"big\": {}}}}}",
            u64::MAX
        );
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v0/cypher")
                    .header("content-type", "application/json")
                    .header(axum::http::header::CONTENT_LENGTH, body.len().to_string())
                    .body(Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }

    /// Unaudited dimension (25tb-readiness): the in-process concurrent write
    /// contract. Simultaneous HTTP write transactions serialize on the
    /// writer mutex — both must succeed, their effects must both commit, and
    /// a subsequent read sees every write exactly once.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn concurrent_http_writes_serialize_and_both_commit() {
        let app = fixture(None).await;
        let post = |app: Router, body: serde_json::Value| async move {
            let body = serde_json::to_vec(&body).unwrap();
            app.oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v0/cypher")
                    .header("content-type", "application/json")
                    .header(axum::http::header::CONTENT_LENGTH, body.len().to_string())
                    .body(Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap()
        };
        let mut writers = Vec::new();
        for ordinal in 0..8 {
            let app = app.clone();
            writers.push(tokio::spawn(async move {
                let response = post(
                    app,
                    serde_json::json!({
                        "query": format!("CREATE (:Audit {{slot: {ordinal}}})")
                    }),
                )
                .await;
                response.status()
            }));
        }
        for writer in writers {
            assert_eq!(
                writer.await.unwrap(),
                StatusCode::OK,
                "every concurrent write must serialize and succeed"
            );
        }
        let response = post(
            app,
            serde_json::json!({"query": "MATCH (a:Audit) RETURN count(*) AS c"}),
        )
        .await;
        assert_eq!(response.status(), StatusCode::OK);
        let bytes = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let parsed: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(
            parsed["rows"][0]["c"],
            serde_json::json!(8),
            "all eight serialized writes must be visible exactly once"
        );
    }

    /// Plan item 32 (the in-process half): the memory ceiling exercised
    /// against the REAL resident-set sample, not a synthetic gauge. A
    /// ceiling below the process's actual RSS must reject queries with 503
    /// and count them; a sane ceiling must serve. The cgroup `auto` mode in
    /// a real container stays with the pre-load runbook.
    #[tokio::test]
    async fn real_rss_ceiling_rejects_queries_and_recovers() {
        let (store, paths) = namidb_storage::parse_uri("memory://rss-ceiling").unwrap();
        let writer = WriterSession::open(store, paths).await.unwrap();
        let tiny = Arc::new(memory::MemoryGovernor::new(64 * 1024));
        if tiny.sample().is_none() {
            // Platform without a resident-set source: nothing real to test.
            return;
        }
        assert!(
            tiny.over_limit(),
            "a 64 KiB ceiling must sit below any real process RSS"
        );
        let state = AppState::new(writer, None, "rss-ceiling".into())
            .with_memory_governor(Arc::clone(&tiny));
        let app = build_router(state);
        let body = serde_json::to_vec(&serde_json::json!({"query": "RETURN 1 AS v"})).unwrap();
        let response = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v0/cypher")
                    .header("content-type", "application/json")
                    .header(axum::http::header::CONTENT_LENGTH, body.len().to_string())
                    .body(Body::from(body.clone()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(
            response.status(),
            StatusCode::SERVICE_UNAVAILABLE,
            "a real over-ceiling RSS must reject queries"
        );
        assert!(tiny.rejected_queries() > 0, "the rejection must be counted");

        // Recovery: the same server shape under a sane ceiling serves.
        let (store, paths) = namidb_storage::parse_uri("memory://rss-ceiling-ok").unwrap();
        let writer = WriterSession::open(store, paths).await.unwrap();
        let sane = Arc::new(memory::MemoryGovernor::new(usize::MAX));
        sane.sample();
        let state = AppState::new(writer, None, "rss-ceiling-ok".into()).with_memory_governor(sane);
        let app = build_router(state);
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v0/cypher")
                    .header("content-type", "application/json")
                    .header(axum::http::header::CONTENT_LENGTH, body.len().to_string())
                    .body(Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK, "a sane ceiling serves");
    }

    /// Plan item 34: the serving route must be observable at the server
    /// surface, or total loss of native serving passes every parity check.
    #[tokio::test]
    async fn metrics_expose_the_search_route_counters() {
        let app = fixture(None).await;
        let response = app
            .oneshot(
                Request::builder()
                    .method("GET")
                    .uri("/v0/metrics")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let bytes = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let text = String::from_utf8(bytes.to_vec()).unwrap();
        for series in [
            "namidb_search_route_total{kind=\"text\",route=\"native\"}",
            "namidb_search_route_total{kind=\"text\",route=\"fallback\"}",
            "namidb_search_route_total{kind=\"vector\",route=\"native\"}",
            "namidb_search_route_total{kind=\"vector\",route=\"fallback\"}",
        ] {
            assert!(
                text.contains(series),
                "/metrics must export `{series}`; got:\n{text}"
            );
        }
    }

    #[test]
    fn allocator_trim_is_reserved_for_flushes_that_wrote_an_sst() {
        assert!(!flush_needs_allocator_trim(0));
        assert!(flush_needs_allocator_trim(1));
        assert!(flush_needs_allocator_trim(2));
    }

    #[test]
    fn http_headroom_uses_exact_lengths_and_real_limit_for_unknown_streams() {
        let exact = Request::builder()
            .uri("/v0/cypher")
            .header(axum::http::header::CONTENT_LENGTH, "123")
            .body(Body::empty())
            .unwrap();
        assert_eq!(http_cypher_wire_bytes(&exact), Ok(123));

        let hinted = Request::builder()
            .uri("/v0/cypher")
            .body(Body::from("small"))
            .unwrap();
        assert_eq!(http_cypher_wire_bytes(&hinted), Ok(5));

        let unknown = Request::builder()
            .uri("/v0/cypher")
            .body(Body::from_stream(futures::stream::once(async {
                Ok::<_, std::convert::Infallible>(bytes::Bytes::from_static(b"{}"))
            })))
            .unwrap();
        assert_eq!(
            http_cypher_wire_bytes(&unknown),
            Ok(HTTP_CYPHER_BODY_LIMIT_BYTES),
            "chunked/unknown bodies reserve the exact extractor ceiling instead of bypassing it"
        );

        let oversized = Request::builder()
            .uri("/v0/cypher")
            .header(
                axum::http::header::CONTENT_LENGTH,
                (HTTP_CYPHER_BODY_LIMIT_BYTES + 1).to_string(),
            )
            .body(Body::empty())
            .unwrap();
        assert_eq!(
            http_cypher_wire_bytes(&oversized),
            Err(HttpBodyAdmissionError::TooLarge {
                observed: HTTP_CYPHER_BODY_LIMIT_BYTES as u64 + 1
            })
        );

        let malformed = Request::builder()
            .uri("/v0/cypher")
            .header(axum::http::header::CONTENT_LENGTH, "not-a-number")
            .body(Body::empty())
            .unwrap();
        assert_eq!(
            http_cypher_wire_bytes(&malformed),
            Err(HttpBodyAdmissionError::InvalidContentLength)
        );
        assert_eq!(
            estimated_http_request_memory_bytes(10),
            HTTP_MEMORY_BASE_BYTES + 10 * HTTP_MEMORY_BYTES_PER_WIRE_BYTE
        );
        assert_eq!(
            estimated_http_request_memory_bytes(usize::MAX),
            usize::MAX,
            "projection must saturate instead of wrapping below the memory rail"
        );
    }

    #[tokio::test]
    async fn http_headroom_guard_lives_through_handler_and_is_released() {
        let (store, paths) = namidb_storage::parse_uri("memory://http-headroom-raii").unwrap();
        let writer = WriterSession::open(store, paths).await.unwrap();
        let governor = Arc::new(memory::MemoryGovernor::new(usize::MAX));
        let state = AppState::new(writer, None, "http-headroom-raii".into())
            .with_memory_governor(Arc::clone(&governor));
        let app = build_router(state);
        let body = serde_json::to_vec(&serde_json::json!({
            "query": "RETURN 1 AS value"
        }))
        .unwrap();
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v0/cypher")
                    .header("content-type", "application/json")
                    .header(axum::http::header::CONTENT_LENGTH, body.len().to_string())
                    .body(Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(
            governor.reserved_headroom_bytes(),
            0,
            "the middleware's RAII reservation must release after the full response is built"
        );
    }

    #[tokio::test]
    async fn http_oversized_content_length_is_413_before_json_decode() {
        let app = fixture(None).await;
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v0/cypher")
                    .header("content-type", "application/json")
                    .header(
                        axum::http::header::CONTENT_LENGTH,
                        (HTTP_CYPHER_BODY_LIMIT_BYTES + 1).to_string(),
                    )
                    .body(Body::from("{}"))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::PAYLOAD_TOO_LARGE);
    }

    #[test]
    fn search_index_capacity_is_retryable_http_503() {
        let error = namidb_query::exec::ExecError::Storage(namidb_storage::Error::CacheCapacity {
            index_kind: "vector",
            path: "tenants/test/sst/search.vg".into(),
            required_bytes: 8_000,
            capacity_bytes: 4_000,
        });
        assert_eq!(
            exec_error_classification(&error),
            (
                StatusCode::SERVICE_UNAVAILABLE,
                Some("search_index_cache_capacity")
            )
        );
    }

    #[test]
    fn search_workspace_capacity_is_retryable_http_503() {
        let error =
            namidb_query::exec::ExecError::Storage(namidb_storage::Error::QueryWorkspaceExceeded {
                operation: "exact full-text fallback",
                required_bytes: 512 * 1024 * 1024,
                capacity_bytes: 256 * 1024 * 1024,
            });
        assert_eq!(
            exec_error_classification(&error),
            (
                StatusCode::SERVICE_UNAVAILABLE,
                Some("search_workspace_capacity")
            )
        );
    }

    #[test]
    fn search_result_and_document_limits_are_http_413() {
        let result = namidb_query::exec::ExecError::Storage(
            namidb_storage::Error::SearchResultLimitExceeded {
                index_kind: "full-text",
                estimated_bytes: 80 * 1024 * 1024,
                limit_bytes: 64 * 1024 * 1024,
            },
        );
        assert_eq!(
            exec_error_classification(&result),
            (StatusCode::PAYLOAD_TOO_LARGE, Some("search_result_limit"))
        );

        let document = namidb_query::exec::ExecError::Storage(
            namidb_storage::Error::SearchDocumentLimitExceeded {
                operation: "exact full-text fallback",
                document_bytes: 2 * 1024 * 1024,
                limit_bytes: 1024 * 1024,
            },
        );
        assert_eq!(
            exec_error_classification(&document),
            (StatusCode::PAYLOAD_TOO_LARGE, Some("search_document_limit"))
        );
    }

    #[tokio::test]
    async fn search_limit_response_exposes_stable_machine_code() {
        let error = namidb_query::exec::ExecError::Storage(
            namidb_storage::Error::SearchDocumentLimitExceeded {
                operation: "exact full-text fallback",
                document_bytes: 2 * 1024 * 1024,
                limit_bytes: 1024 * 1024,
            },
        );
        let response = exec_failure_response("query failed", &error);
        assert_eq!(response.status(), StatusCode::PAYLOAD_TOO_LARGE);
        let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["code"], "search_document_limit");
        assert!(json["error"]
            .as_str()
            .is_some_and(|message| message.contains("NAMIDB_BM25_MAX_DOCUMENT_BYTES")));
    }

    /// NDB-03: resource exhaustion carries its own taxonomy — a client must
    /// be able to tell "too expensive" from "malformed" without string
    /// matching, on both the `code` and the Neo4j/GQL-shaped fields.
    #[tokio::test]
    async fn timeout_response_exposes_taxonomy_fields() {
        let response =
            exec_failure_response("read execution failed", &namidb_query::exec::ExecError::Timeout);
        assert_eq!(response.status(), StatusCode::GATEWAY_TIMEOUT);
        let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["code"], "timeout");
        assert_eq!(
            json["neo4j_code"],
            "Neo.ClientError.Transaction.TransactionTimedOut"
        );
        assert_eq!(json["gql_status"], "57014");

        let response = exec_failure_response(
            "read execution failed",
            &namidb_query::exec::ExecError::RowCap(1_000_000),
        );
        assert_eq!(response.status(), StatusCode::PAYLOAD_TOO_LARGE);
        let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["code"], "row_cap");
        assert_eq!(
            json["neo4j_code"],
            "Neo.ClientError.Statement.ResourceLimitExceeded"
        );
        assert_eq!(json["gql_status"], "54000");
    }

    /// NDB-03: a runtime evaluation error is the caller's program being
    /// wrong — a 400 with the argument-error taxonomy, not a bare 500.
    #[tokio::test]
    async fn division_by_zero_is_a_client_error_with_taxonomy() {
        let app = fixture(None).await;
        let response = post_cypher(&app, None, "RETURN 1/0 AS x").await;
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["code"], "eval_error");
        assert_eq!(json["neo4j_code"], "Neo.ClientError.Statement.ArgumentError");
        assert_eq!(json["gql_status"], "22000");
    }

    /// NDB-03: parse errors carry the syntax taxonomy in the HTTP body.
    #[tokio::test]
    async fn parse_error_carries_syntax_taxonomy() {
        let app = fixture(None).await;
        let response = post_cypher(&app, None, "MATCH (").await;
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["code"], "parse_error");
        assert_eq!(json["neo4j_code"], "Neo.ClientError.Statement.SyntaxError");
        assert_eq!(json["gql_status"], "42001");
    }

    /// Router for namespace `ns` whose auth is loaded from `tokens_json` (the
    /// real `--auth-tokens-file` path), exercising per-token roles.
    async fn fixture_with_tokens(ns: &str, tokens_json: &str) -> Router {
        let path = std::env::temp_dir().join(format!("namidb-test-tokens-{ns}.json"));
        std::fs::write(&path, tokens_json).unwrap();
        let auth = crate::auth::AuthConfig::load_file(&path).unwrap();
        std::fs::remove_file(&path).ok();
        let (store, paths) = namidb_storage::parse_uri(&format!("memory://{ns}")).unwrap();
        let writer = WriterSession::open(store, paths).await.unwrap();
        let state = AppState::new(writer, None, ns.into()).with_auth(Arc::new(auth));
        build_router(state)
    }

    /// POST a Cypher query with an optional bearer token; return the response.
    async fn post_cypher(app: &Router, token: Option<&str>, query: &str) -> Response {
        let mut builder = Request::builder()
            .method("POST")
            .uri("/v0/cypher")
            .header("content-type", "application/json");
        if let Some(t) = token {
            builder = builder.header("authorization", format!("Bearer {t}"));
        }
        app.clone()
            .oneshot(
                builder
                    .body(Body::from(
                        serde_json::to_vec(&serde_json::json!({ "query": query })).unwrap(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap()
    }

    // A hook that denies every request — proves the dispatcher honours a deny
    // decision, including for READS (which the allows_write gate cannot block).
    struct DenyAllAuthz;
    #[async_trait::async_trait]
    impl crate::authz::AuthzHook for DenyAllAuthz {
        async fn check(
            &self,
            _p: &Principal,
            _plan: &namidb_query::LogicalPlan,
        ) -> Result<(), crate::authz::Denied> {
            Err(crate::authz::Denied::new("denied by test policy"))
        }
    }

    #[tokio::test]
    async fn authz_hook_can_deny_reads() {
        // Open mode (no token) + a deny-all hook: a plain read is rejected 403,
        // proving the hook runs and can deny what the role gate would allow.
        let (store, paths) = namidb_storage::parse_uri("memory://authz-deny").unwrap();
        let writer = WriterSession::open(store, paths).await.unwrap();
        let state = AppState::new(writer, None, "test".into()).with_authz(Arc::new(DenyAllAuthz));
        let app = build_router(state);

        let resp = post_cypher(&app, None, "MATCH (n) RETURN n").await;
        assert_eq!(resp.status(), StatusCode::FORBIDDEN);
        let body: serde_json::Value =
            serde_json::from_slice(&to_bytes(resp.into_body(), 4096).await.unwrap()).unwrap();
        assert!(body["error"]
            .as_str()
            .unwrap()
            .contains("denied by test policy"));
    }

    #[tokio::test]
    async fn default_authz_is_allow_all() {
        // The default NoOpAuthz must not change behavior: a read still succeeds.
        let app = fixture(None).await;
        let resp = post_cypher(&app, None, "MATCH (n) RETURN n").await;
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn memory_admission_rejects_before_json_but_keeps_admin_flush_available() {
        let (store, paths) = namidb_storage::parse_uri("memory://memory-pre-extractor").unwrap();
        let mut writer = WriterSession::open(store, paths).await.unwrap();
        writer
            .upsert_node(
                "Pressure",
                namidb_core::id::NodeId::new(),
                &namidb_storage::NodeWriteRecord {
                    properties: std::collections::BTreeMap::new(),
                    schema_version: 1,
                    ..Default::default()
                },
            )
            .unwrap();
        writer.commit_batch().await.unwrap();
        assert!(writer.memtable_bytes() > 0);
        let before_version = writer.manifest_version();
        // One byte is deliberately below the resident set of this test
        // process, making the real sampler deterministic without a mock-only
        // admission path.
        let governor = Arc::new(memory::MemoryGovernor::new(1));
        let state = AppState::new(writer, None, "memory-test".into())
            .with_memory_governor(Arc::clone(&governor));
        let app = build_router(state.clone());

        // Malformed JSON would be a 400 if the handler's Json extractor ran.
        let malformed = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v0/cypher")
                    .header("content-type", "application/json")
                    .body(Body::from("{"))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(malformed.status(), StatusCode::SERVICE_UNAVAILABLE);

        // Admin flush is the authenticated escape hatch: it bypasses new-work
        // admission and drains the already-committed memtable.
        let flush = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v0/admin/flush")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(flush.status(), StatusCode::OK);
        assert!(state.snapshot.manifest_version() > before_version);
        assert_eq!(
            state
                .memtable_bytes_gauge
                .load(std::sync::atomic::Ordering::Relaxed),
            0
        );
        assert_eq!(
            governor.rejected_queries(),
            1,
            "watchdog/admin maintenance must not be counted as rejected queries"
        );
    }

    const ROLE_TOKENS: &str = r#"{ "tokens": [
        { "name": "writer", "token": "wkey", "role": "read-write" },
        { "name": "reader", "token": "rkey", "role": "read-only" }
    ] }"#;

    #[tokio::test]
    async fn read_only_token_reads_but_cannot_write() {
        let app = fixture_with_tokens("authz-ro", ROLE_TOKENS).await;

        // A read with the read-only token is allowed.
        let read = post_cypher(&app, Some("rkey"), "MATCH (n) RETURN n").await;
        assert_eq!(read.status(), StatusCode::OK);

        // A write with the read-only token is forbidden (not unauthorized).
        let write = post_cypher(&app, Some("rkey"), "CREATE (:Person {name: 'x'})").await;
        assert_eq!(write.status(), StatusCode::FORBIDDEN);

        // Nothing was written.
        let after = post_cypher(&app, Some("rkey"), "MATCH (p:Person) RETURN p").await;
        let body: serde_json::Value =
            serde_json::from_slice(&to_bytes(after.into_body(), 4096).await.unwrap()).unwrap();
        assert_eq!(body["rows"].as_array().unwrap().len(), 0);
    }

    #[tokio::test]
    async fn read_write_token_can_write() {
        let app = fixture_with_tokens("authz-rw", ROLE_TOKENS).await;
        let write = post_cypher(&app, Some("wkey"), "CREATE (:Person {name: 'x'}) RETURN 1").await;
        assert_eq!(write.status(), StatusCode::OK);
    }

    /// `CREATE VECTOR INDEX` end-to-end over HTTP: registers a descriptor,
    /// rejects a duplicate with 400, and is forbidden for a read-only token.
    #[cfg(feature = "vector-index")]
    #[tokio::test]
    async fn create_vector_index_registers_and_reports_duplicate() {
        let app = fixture(None).await;

        let q = "CREATE VECTOR INDEX doc_emb ON :Doc(emb) METRIC cosine DIMENSION 16";
        let r = post_cypher(&app, None, q).await;
        assert_eq!(r.status(), StatusCode::OK, "first create should succeed");
        let body: serde_json::Value =
            serde_json::from_slice(&to_bytes(r.into_body(), 4096).await.unwrap()).unwrap();
        assert!(body["rows"].as_array().unwrap().is_empty());

        // Same name (or same target) again is a duplicate → 400.
        let dup = post_cypher(&app, None, q).await;
        assert_eq!(dup.status(), StatusCode::BAD_REQUEST);

        // …but the same target with IF NOT EXISTS is an idempotent no-op → 200.
        let ine = post_cypher(
            &app,
            None,
            "CREATE VECTOR INDEX doc_emb IF NOT EXISTS ON :Doc(emb) METRIC cosine DIMENSION 16",
        )
        .await;
        assert_eq!(
            ine.status(),
            StatusCode::OK,
            "IF NOT EXISTS over a duplicate must succeed as a no-op"
        );

        // A read-only token may not run schema DDL.
        let app_ro = fixture_with_tokens("vecidx-ro", ROLE_TOKENS).await;
        let ro = post_cypher(&app_ro, Some("rkey"), q).await;
        assert_eq!(ro.status(), StatusCode::FORBIDDEN);
    }

    /// `CREATE FULLTEXT INDEX` end-to-end over HTTP: registers, rejects a
    /// duplicate with 400, and is forbidden for a read-only token.
    #[cfg(feature = "text-index")]
    #[tokio::test]
    async fn create_fulltext_index_registers_and_reports_duplicate() {
        let app = fixture(None).await;

        let q = "CREATE FULLTEXT INDEX note_ft ON :Note(body, title)";
        let r = post_cypher(&app, None, q).await;
        assert_eq!(r.status(), StatusCode::OK, "first create should succeed");
        let body: serde_json::Value =
            serde_json::from_slice(&to_bytes(r.into_body(), 4096).await.unwrap()).unwrap();
        assert!(body["rows"].as_array().unwrap().is_empty());

        // Same name (or same target) again is a duplicate → 400.
        let dup = post_cypher(&app, None, q).await;
        assert_eq!(dup.status(), StatusCode::BAD_REQUEST);

        // …but the same target with IF NOT EXISTS is an idempotent no-op → 200.
        let ine = post_cypher(
            &app,
            None,
            "CREATE FULLTEXT INDEX note_ft IF NOT EXISTS ON :Note(body, title)",
        )
        .await;
        assert_eq!(
            ine.status(),
            StatusCode::OK,
            "IF NOT EXISTS over a duplicate must succeed as a no-op"
        );

        // A read-only token may not run schema DDL.
        let app_ro = fixture_with_tokens("ftidx-ro", ROLE_TOKENS).await;
        let ro = post_cypher(&app_ro, Some("rkey"), q).await;
        assert_eq!(ro.status(), StatusCode::FORBIDDEN);
    }

    /// `DROP VECTOR INDEX` end-to-end over HTTP: unregisters the descriptor
    /// (so the slot can be re-created), reports a missing index with 400 unless
    /// `IF EXISTS`, and is forbidden for a read-only token — the same authz
    /// treatment as CREATE.
    #[cfg(feature = "vector-index")]
    #[tokio::test]
    async fn drop_vector_index_unregisters_and_reports_missing() {
        let app = fixture(None).await;

        let create = "CREATE VECTOR INDEX doc_emb ON :Doc(emb) METRIC cosine DIMENSION 16";
        assert_eq!(
            post_cypher(&app, None, create).await.status(),
            StatusCode::OK
        );

        // Drop succeeds with an empty response…
        let r = post_cypher(&app, None, "DROP VECTOR INDEX doc_emb").await;
        assert_eq!(r.status(), StatusCode::OK);
        let body: serde_json::Value =
            serde_json::from_slice(&to_bytes(r.into_body(), 4096).await.unwrap()).unwrap();
        assert!(body["rows"].as_array().unwrap().is_empty());

        // …and the slot is free: the same CREATE is no longer a duplicate.
        let recreate = post_cypher(&app, None, create).await;
        assert_eq!(
            recreate.status(),
            StatusCode::OK,
            "re-creating over the dropped slot must succeed"
        );

        // A missing index is a 400 without IF EXISTS, a no-op 200 with it.
        let missing = post_cypher(&app, None, "DROP VECTOR INDEX nope").await;
        assert_eq!(missing.status(), StatusCode::BAD_REQUEST);
        let ine = post_cypher(&app, None, "DROP VECTOR INDEX nope IF EXISTS").await;
        assert_eq!(ine.status(), StatusCode::OK);

        // A read-only token may not run schema DDL.
        let app_ro = fixture_with_tokens("dropvec-ro", ROLE_TOKENS).await;
        let ro = post_cypher(&app_ro, Some("rkey"), "DROP VECTOR INDEX doc_emb IF EXISTS").await;
        assert_eq!(ro.status(), StatusCode::FORBIDDEN);
    }

    /// `DROP INDEX` (fulltext) end-to-end over HTTP: unregisters the
    /// descriptor, accepts the `DROP FULLTEXT INDEX` alias, reports a missing
    /// index with 400 unless `IF EXISTS`, and is forbidden for a read-only
    /// token.
    #[cfg(feature = "text-index")]
    #[tokio::test]
    async fn drop_index_unregisters_fulltext_and_reports_missing() {
        let app = fixture(None).await;

        let create = "CREATE FULLTEXT INDEX note_ft ON :Note(body, title)";
        assert_eq!(
            post_cypher(&app, None, create).await.status(),
            StatusCode::OK
        );

        // Drop, then re-create over the freed (label, properties) slot.
        let r = post_cypher(&app, None, "DROP INDEX note_ft").await;
        assert_eq!(r.status(), StatusCode::OK);
        let recreate = post_cypher(&app, None, create).await;
        assert_eq!(
            recreate.status(),
            StatusCode::OK,
            "re-creating over the dropped slot must succeed"
        );

        // The `DROP FULLTEXT INDEX` alias drops it too.
        let alias = post_cypher(&app, None, "DROP FULLTEXT INDEX note_ft").await;
        assert_eq!(alias.status(), StatusCode::OK);

        // A missing index is a 400 without IF EXISTS, a no-op 200 with it.
        let missing = post_cypher(&app, None, "DROP INDEX note_ft").await;
        assert_eq!(missing.status(), StatusCode::BAD_REQUEST);
        let ine = post_cypher(&app, None, "DROP INDEX note_ft IF EXISTS").await;
        assert_eq!(ine.status(), StatusCode::OK);

        // A read-only token may not run schema DDL.
        let app_ro = fixture_with_tokens("dropft-ro", ROLE_TOKENS).await;
        let ro = post_cypher(&app_ro, Some("rkey"), "DROP INDEX note_ft IF EXISTS").await;
        assert_eq!(ro.status(), StatusCode::FORBIDDEN);
    }

    /// The wrong-dimension unbrick scenario end-to-end over HTTP: a dim-1536
    /// index rejects a 768-dim write (400), `DROP VECTOR INDEX` removes it,
    /// and the identical write then succeeds.
    #[cfg(feature = "vector-index")]
    #[tokio::test]
    async fn drop_vector_index_unbricks_wrong_dim_writes_over_http() {
        let app = fixture(None).await;

        let create = "CREATE VECTOR INDEX doc_emb ON :Doc(embedding) METRIC cosine DIMENSION 1536";
        assert_eq!(
            post_cypher(&app, None, create).await.status(),
            StatusCode::OK
        );

        // A 768-dim embedding violates the (misconfigured) declared dimension.
        let vec768 = format!(
            "CREATE (:Doc {{embedding: vector([{}])}})",
            vec!["0.5"; 768].join(", ")
        );
        let rejected = post_cypher(&app, None, &vec768).await;
        assert_eq!(
            rejected.status(),
            StatusCode::CONFLICT,
            "wrong-dim write must be rejected (dimension constraint) while the index exists"
        );

        // Drop the misconfigured index: the identical write now succeeds.
        let dropped = post_cypher(&app, None, "DROP VECTOR INDEX doc_emb").await;
        assert_eq!(dropped.status(), StatusCode::OK);
        let accepted = post_cypher(&app, None, &vec768).await;
        assert_eq!(
            accepted.status(),
            StatusCode::OK,
            "the write must succeed once the index is dropped"
        );
    }

    #[tokio::test]
    async fn create_constraint_enforces_uniqueness_end_to_end() {
        let app = fixture(None).await;

        // Declare a uniqueness constraint (Neo4j 5 syntax).
        let r = post_cypher(
            &app,
            None,
            "CREATE CONSTRAINT FOR (n:User) REQUIRE n.email IS UNIQUE",
        )
        .await;
        assert_eq!(r.status(), StatusCode::OK, "constraint should be created");

        // First insert is fine.
        let r1 = post_cypher(&app, None, "CREATE (:User {email: 'a@x.com'})").await;
        assert_eq!(r1.status(), StatusCode::OK);

        // A duplicate value is now rejected by the engine (the whole point):
        // 409 Conflict from the unique-constraint violation.
        let r2 = post_cypher(&app, None, "CREATE (:User {email: 'a@x.com'})").await;
        assert_eq!(
            r2.status(),
            StatusCode::CONFLICT,
            "duplicate must violate the unique constraint"
        );

        // A different value still inserts.
        let r3 = post_cypher(&app, None, "CREATE (:User {email: 'b@x.com'})").await;
        assert_eq!(r3.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn create_constraint_rejects_existing_duplicates_and_read_only_token() {
        let app = fixture(None).await;
        // Seed a duplicate, then the constraint must be refused (400).
        post_cypher(&app, None, "CREATE (:Tag {slug: 'x'})").await;
        post_cypher(&app, None, "CREATE (:Tag {slug: 'x'})").await;
        let dup = post_cypher(
            &app,
            None,
            "CREATE CONSTRAINT FOR (n:Tag) REQUIRE n.slug IS UNIQUE",
        )
        .await;
        assert_eq!(dup.status(), StatusCode::BAD_REQUEST);

        // A read-only token may not run schema DDL.
        let app_ro = fixture_with_tokens("constraint-ro", ROLE_TOKENS).await;
        let ro = post_cypher(
            &app_ro,
            Some("rkey"),
            "CREATE INDEX FOR (n:Doc) ON (n.title)",
        )
        .await;
        assert_eq!(ro.status(), StatusCode::FORBIDDEN);
    }

    #[tokio::test]
    async fn create_constraint_legacy_assert_syntax_with_name() {
        let app = fixture(None).await;
        // Neo4j 4 form + a constraint name.
        let r = post_cypher(
            &app,
            None,
            "CREATE CONSTRAINT acct_num ON (n:Acct) ASSERT n.num IS UNIQUE",
        )
        .await;
        assert_eq!(r.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn create_index_legacy_syntax_parses_and_applies() {
        let app = fixture(None).await;
        // Neo4j 4 form: `ON :Label(prop)`.
        let r = post_cypher(&app, None, "CREATE INDEX FOR (n:Doc) ON (n.slug)").await;
        assert_eq!(r.status(), StatusCode::OK);
        let r2 = post_cypher(&app, None, "CREATE INDEX ON :Doc(author)").await;
        assert_eq!(r2.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn composite_constraint_enforces_uniqueness_end_to_end() {
        let app = fixture(None).await;
        let r = post_cypher(
            &app,
            None,
            "CREATE CONSTRAINT FOR (n:Cfg) REQUIRE (n.tenant, n.name) IS UNIQUE",
        )
        .await;
        assert_eq!(r.status(), StatusCode::OK, "composite constraint created");

        let r1 = post_cypher(&app, None, "CREATE (:Cfg {tenant: 't1', name: 'a'})").await;
        assert_eq!(r1.status(), StatusCode::OK);
        // Same tenant, different name → distinct tuple → allowed.
        let r2 = post_cypher(&app, None, "CREATE (:Cfg {tenant: 't1', name: 'b'})").await;
        assert_eq!(r2.status(), StatusCode::OK);
        // Exact duplicate tuple → 409 Conflict.
        let r3 = post_cypher(&app, None, "CREATE (:Cfg {tenant: 't1', name: 'a'})").await;
        assert_eq!(
            r3.status(),
            StatusCode::CONFLICT,
            "duplicate composite tuple must conflict"
        );
    }

    #[tokio::test]
    async fn constraint_if_not_exists_is_idempotent() {
        let app = fixture(None).await;
        let a = post_cypher(
            &app,
            None,
            "CREATE CONSTRAINT c1 IF NOT EXISTS FOR (n:User) REQUIRE n.email IS UNIQUE",
        )
        .await;
        assert_eq!(a.status(), StatusCode::OK);
        // Re-running the exact same DDL with IF NOT EXISTS is a no-op success.
        let b = post_cypher(
            &app,
            None,
            "CREATE CONSTRAINT c1 IF NOT EXISTS FOR (n:User) REQUIRE n.email IS UNIQUE",
        )
        .await;
        assert_eq!(b.status(), StatusCode::OK, "IF NOT EXISTS re-run succeeds");
        // Without IF NOT EXISTS, re-declaring the same constraint is a 400.
        let c = post_cypher(
            &app,
            None,
            "CREATE CONSTRAINT FOR (n:User) REQUIRE n.email IS UNIQUE",
        )
        .await;
        assert_eq!(c.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn show_constraints_lists_declared_constraints() {
        let app = fixture(None).await;
        post_cypher(
            &app,
            None,
            "CREATE CONSTRAINT FOR (n:User) REQUIRE n.email IS UNIQUE",
        )
        .await;
        post_cypher(
            &app,
            None,
            "CREATE CONSTRAINT cfg_uq FOR (n:Cfg) REQUIRE (n.tenant, n.name) IS UNIQUE",
        )
        .await;

        let r = post_cypher(&app, None, "SHOW CONSTRAINTS").await;
        assert_eq!(r.status(), StatusCode::OK);
        let body: serde_json::Value =
            serde_json::from_slice(&to_bytes(r.into_body(), 65536).await.unwrap()).unwrap();
        let cols: Vec<&str> = body["columns"]
            .as_array()
            .unwrap()
            .iter()
            .map(|c| c.as_str().unwrap())
            .collect();
        assert!(cols.contains(&"name") && cols.contains(&"properties"));
        let rows = body["rows"].as_array().unwrap();
        assert_eq!(rows.len(), 2, "both constraints listed");
        let cfg = rows
            .iter()
            .find(|row| row["name"] == "cfg_uq")
            .expect("cfg_uq present");
        let props: Vec<&str> = cfg["properties"]
            .as_array()
            .unwrap()
            .iter()
            .map(|p| p.as_str().unwrap())
            .collect();
        assert_eq!(props, ["tenant", "name"]);
        assert_eq!(cfg["type"], "UNIQUENESS");
        assert_eq!(cfg["entityType"], "NODE");
        assert_eq!(cfg["labelsOrTypes"][0], "Cfg");
    }

    #[tokio::test]
    async fn show_indexes_lists_declared_indexes() {
        let app = fixture(None).await;
        post_cypher(&app, None, "CREATE INDEX FOR (n:Doc) ON (n.slug)").await;
        let r = post_cypher(&app, None, "SHOW INDEXES").await;
        assert_eq!(r.status(), StatusCode::OK);
        let body: serde_json::Value =
            serde_json::from_slice(&to_bytes(r.into_body(), 65536).await.unwrap()).unwrap();
        let rows = body["rows"].as_array().unwrap();
        let doc = rows
            .iter()
            .find(|row| row["labelsOrTypes"][0] == "Doc")
            .expect("Doc index present");
        let props: Vec<&str> = doc["properties"]
            .as_array()
            .unwrap()
            .iter()
            .map(|p| p.as_str().unwrap())
            .collect();
        assert_eq!(props, ["slug"]);
    }

    #[cfg(feature = "text-index")]
    #[tokio::test]
    async fn create_fulltext_index_consults_authz_check_schema() {
        let (store, paths) = namidb_storage::parse_uri("memory://ftidx-authz").unwrap();
        let writer = WriterSession::open(store, paths).await.unwrap();
        let state =
            AppState::new(writer, None, "test".into()).with_authz(Arc::new(DenySchemaAuthz));
        let app = build_router(state);

        let q = "CREATE FULLTEXT INDEX note_ft ON :Note(body)";
        let resp = post_cypher(&app, None, q).await;
        assert_eq!(resp.status(), StatusCode::FORBIDDEN);
    }

    // A hook that allows queries but denies schema (DDL) operations. Only the
    // DDL authz tests construct it, so gate it the same way or the default build
    // flags it as dead code.
    #[cfg(any(feature = "vector-index", feature = "text-index"))]
    struct DenySchemaAuthz;
    #[cfg(any(feature = "vector-index", feature = "text-index"))]
    #[async_trait::async_trait]
    impl crate::authz::AuthzHook for DenySchemaAuthz {
        async fn check(
            &self,
            _p: &Principal,
            _plan: &namidb_query::LogicalPlan,
        ) -> Result<(), crate::authz::Denied> {
            Ok(())
        }
        async fn check_schema(
            &self,
            _p: &Principal,
            _op: crate::authz::SchemaOp<'_>,
        ) -> Result<(), crate::authz::Denied> {
            Err(crate::authz::Denied::new("schema changes denied by policy"))
        }
    }

    #[cfg(feature = "vector-index")]
    #[tokio::test]
    async fn create_vector_index_consults_authz_check_schema() {
        // Open mode (read-write principal) but a hook that denies schema ops:
        // the DDL must be 403'd by the hook, proving DDL is not a policy bypass.
        let (store, paths) = namidb_storage::parse_uri("memory://vecidx-authz").unwrap();
        let writer = WriterSession::open(store, paths).await.unwrap();
        let state =
            AppState::new(writer, None, "test".into()).with_authz(Arc::new(DenySchemaAuthz));
        let app = build_router(state);

        let q = "CREATE VECTOR INDEX doc_emb ON :Doc(emb) METRIC cosine DIMENSION 16";
        let resp = post_cypher(&app, None, q).await;
        assert_eq!(resp.status(), StatusCode::FORBIDDEN);
        let body: serde_json::Value =
            serde_json::from_slice(&to_bytes(resp.into_body(), 4096).await.unwrap()).unwrap();
        assert!(body["error"]
            .as_str()
            .unwrap()
            .contains("schema changes denied"));
    }

    #[tokio::test]
    async fn bearer_scheme_is_case_insensitive() {
        // RFC 7235 §2.1: the scheme is case-insensitive, and the Bolt path
        // already lowercases it. A lowercase `bearer` must be accepted.
        let app = fixture_with_tokens("authz-case", ROLE_TOKENS).await;
        let resp = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v0/cypher")
                    .header("content-type", "application/json")
                    .header("authorization", "bearer wkey")
                    .body(Body::from(
                        serde_json::to_vec(&serde_json::json!({ "query": "MATCH (n) RETURN n" }))
                            .unwrap(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn empty_single_token_is_treated_as_open_not_a_bypass() {
        // An empty `--auth-token` must not become a token a `Bearer ` request
        // matches. `AppState::new` falls back to open mode (the boot path in
        // `run` rejects it outright); either way there is no empty-secret token.
        let (store, paths) = namidb_storage::parse_uri("memory://authz-empty").unwrap();
        let writer = WriterSession::open(store, paths).await.unwrap();
        let app = build_router(AppState::new(writer, Some(String::new()), "t".into()));
        // No token at all is served (open mode), and a `Bearer ` (empty) is not
        // a privileged match — both reach the handler as read-write.
        assert_eq!(
            post_cypher(&app, None, "MATCH (n) RETURN n").await.status(),
            StatusCode::OK
        );
    }

    #[tokio::test]
    async fn unknown_token_is_unauthorized() {
        let app = fixture_with_tokens("authz-bad", ROLE_TOKENS).await;
        let resp = post_cypher(&app, Some("nope"), "MATCH (n) RETURN n").await;
        assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn admin_flush_stays_authenticated_and_forbids_read_only_tokens() {
        let app = fixture_with_tokens("authz-flush", ROLE_TOKENS).await;
        let unauthenticated = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v0/admin/flush")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(unauthenticated.status(), StatusCode::UNAUTHORIZED);

        let flush = |token: &'static str| {
            let app = app.clone();
            async move {
                app.oneshot(
                    Request::builder()
                        .method("POST")
                        .uri("/v0/admin/flush")
                        .header("authorization", format!("Bearer {token}"))
                        .body(Body::empty())
                        .unwrap(),
                )
                .await
                .unwrap()
            }
        };
        assert_eq!(flush("rkey").await.status(), StatusCode::FORBIDDEN);
        assert_eq!(flush("wkey").await.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn catalog_cache_reuses_until_version_changes() {
        let (store, paths) = namidb_storage::parse_uri("memory://test-catalog-cache").unwrap();
        let writer = WriterSession::open(store, paths).await.unwrap();
        let state = AppState::new(writer, None, "test".into());

        let m0 = state.snapshot.load().manifest().manifest.clone();
        let c1 = state.catalog_for(&m0);
        let c2 = state.catalog_for(&m0);
        assert!(
            Arc::ptr_eq(&c1, &c2),
            "same manifest version must reuse the cached catalog"
        );

        // A higher version forces a rebuild (a distinct Arc), then caches it.
        let mut m1 = m0.clone();
        m1.version += 1;
        let c3 = state.catalog_for(&m1);
        assert!(
            !Arc::ptr_eq(&c1, &c3),
            "a new manifest version must rebuild the catalog"
        );
        let c4 = state.catalog_for(&m1);
        assert!(Arc::ptr_eq(&c3, &c4));
    }

    #[tokio::test]
    async fn write_stall_for_respects_threshold() {
        // RFC-027 P5 backpressure decision. Disabled by default.
        let (store, paths) = namidb_storage::parse_uri("memory://test-stall-off").unwrap();
        let off = AppState::new(
            WriterSession::open(store, paths).await.unwrap(),
            None,
            "t".into(),
        );
        assert!(
            off.write_stall_for(1_000, usize::MAX).is_none(),
            "disabled: never stalls"
        );

        // Enabled: stall only at or above the threshold.
        let (store2, paths2) = namidb_storage::parse_uri("memory://test-stall-on").unwrap();
        let on = AppState::new(
            WriterSession::open(store2, paths2).await.unwrap(),
            None,
            "t".into(),
        )
        .with_write_stall(8, Duration::from_millis(50));
        assert_eq!(on.write_stall_for(7, 0), None, "below threshold");
        assert_eq!(
            on.write_stall_for(8, 0),
            Some(Duration::from_millis(50)),
            "at threshold"
        );
        assert_eq!(
            on.write_stall_for(99, 0),
            Some(Duration::from_millis(50)),
            "above threshold"
        );

        // Byte backstop: stalls at/above the memtable threshold even when the
        // L0 stall never trips, with the fixed fallback delay when no
        // write_stall_delay is configured.
        let (store3, paths3) = namidb_storage::parse_uri("memory://test-stall-bytes").unwrap();
        let bytes_only = AppState::new(
            WriterSession::open(store3, paths3).await.unwrap(),
            None,
            "t".into(),
        )
        .with_memtable_thresholds(0, 1024);
        assert_eq!(bytes_only.write_stall_for(0, 1023), None, "below bytes");
        assert_eq!(
            bytes_only.write_stall_for(0, 1024),
            Some(Duration::from_millis(20)),
            "byte backstop with default delay"
        );
    }

    #[tokio::test]
    async fn byte_threshold_nudges_flush_and_publishes_gauge() {
        // A committed write at/above `memtable_flush_bytes` must leave a
        // stored permit on `flush_notify` (so the flush task's next
        // `notified()` resolves immediately instead of waiting out the
        // interval) and publish the lock-free gauge health reads.
        let (store, paths) = namidb_storage::parse_uri("memory://test-bytes-flush").unwrap();
        let state = AppState::new(
            WriterSession::open(store, paths).await.unwrap(),
            None,
            "t".into(),
        )
        .with_memtable_thresholds(1, 0);
        {
            let mut w = state.writer.lock().await;
            w.upsert_node(
                "P",
                namidb_core::id::NodeId::new(),
                &namidb_storage::NodeWriteRecord {
                    properties: std::collections::BTreeMap::new(),
                    schema_version: 1,
                    ..Default::default()
                },
            )
            .unwrap();
            w.commit_batch().await.unwrap();
            let stall = state.after_commit_backpressure(&w);
            assert!(stall.is_none(), "no stall threshold configured");
        }
        tokio::time::timeout(Duration::from_secs(1), state.flush_notify.notified())
            .await
            .expect("the byte trigger must have nudged the flush task");
        assert!(
            state
                .memtable_bytes_gauge
                .load(std::sync::atomic::Ordering::Relaxed)
                > 0,
            "gauge must reflect the committed bytes"
        );
    }

    #[tokio::test]
    async fn bounded_writer_lock_returns_503_when_held() {
        // A write queued behind a held writer lock must fail fast with 503
        // once the configured bound elapses — request queues stay bounded
        // behind a stuck or long-held writer.
        let (store, paths) = namidb_storage::parse_uri("memory://test-lock-timeout").unwrap();
        let writer = WriterSession::open(store, paths).await.unwrap();
        let state = AppState::new(writer, None, "test".into())
            .with_writer_lock_timeout(Duration::from_millis(50));
        let app = build_router(state.clone());

        // Hold the lock as a stand-in for a stuck/long transaction.
        let guard = state.writer.lock().await;
        let resp = post_cypher(&app, None, "CREATE (:P {n: 1}) RETURN 1").await;
        assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
        drop(guard);

        // Released: the same write succeeds.
        let resp = post_cypher(&app, None, "CREATE (:P {n: 1}) RETURN 1").await;
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn health_returns_ok() {
        let app = fixture(None).await;
        let resp = app
            .oneshot(
                Request::builder()
                    .uri("/v0/health")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = to_bytes(resp.into_body(), 1024).await.unwrap();
        let v: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(v["status"], "ok");
        assert_eq!(v["namespace"], "test");
        assert_eq!(v["writer"], "ok");
    }

    /// GET /v0/health; return `(status, body)`.
    async fn get_health(app: &Router) -> (StatusCode, serde_json::Value) {
        let resp = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/v0/health")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        let status = resp.status();
        let body = serde_json::from_slice(&to_bytes(resp.into_body(), 4096).await.unwrap())
            .unwrap_or(serde_json::Value::Null);
        (status, body)
    }

    /// Regression for the fenced-writer dead end: a second `WriterSession`
    /// against the same store fences the server's writer; the failing write
    /// must trigger the automatic reopen so the NEXT write succeeds — no
    /// restart, no operator intervention.
    #[tokio::test]
    async fn write_path_recovers_after_writer_is_fenced() {
        let (store, paths) = namidb_storage::parse_uri("memory://fence-recover").unwrap();
        let writer = WriterSession::open(store.clone(), paths.clone())
            .await
            .unwrap();
        let state = AppState::new(writer, None, "fence-recover".into());
        let app = build_router(state);

        // Seed a committed record that must survive the recovery.
        let seed = post_cypher(&app, None, "CREATE (:Person {name: 'Alice'}) RETURN 1").await;
        assert_eq!(seed.status(), StatusCode::OK);

        // An interloper claims the namespace, fencing the server's writer.
        let interloper = WriterSession::open(store, paths).await.unwrap();
        drop(interloper);

        // The next write hits the fence and fails…
        let fenced = post_cypher(&app, None, "CREATE (:Person {name: 'Bob'}) RETURN 1").await;
        assert_eq!(fenced.status(), StatusCode::INTERNAL_SERVER_ERROR);

        // …but the failure ran the reopen, so a subsequent write succeeds.
        let recovered = post_cypher(&app, None, "CREATE (:Person {name: 'Cara'}) RETURN 1").await;
        assert_eq!(
            recovered.status(),
            StatusCode::OK,
            "the write path must recover automatically after a fence"
        );

        // Committed data survived; the fenced (never-ACKed) write did not
        // resurrect through the reopen's WAL replay.
        let read = post_cypher(&app, None, "MATCH (p:Person) RETURN p.name AS name").await;
        let body: serde_json::Value =
            serde_json::from_slice(&to_bytes(read.into_body(), 65536).await.unwrap()).unwrap();
        let names: Vec<&str> = body["rows"]
            .as_array()
            .unwrap()
            .iter()
            .map(|r| r["name"].as_str().unwrap())
            .collect();
        assert!(names.contains(&"Alice"), "committed row lost: {names:?}");
        assert!(names.contains(&"Cara"), "post-recovery row lost: {names:?}");
        assert!(
            !names.contains(&"Bob"),
            "never-ACKed row must not resurrect: {names:?}"
        );

        // Readiness is green again after the recovery.
        let (status, health) = get_health(&app).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(health["writer"], "ok");
    }

    /// `ObjectStore` wrapper that, while enabled, fails any PUT of a manifest
    /// body that does not already exist. An existing body still surfaces the
    /// real `AlreadyExists` (the fence signal), while every `claim_writer`
    /// (i.e. every reopen attempt) fails cleanly without leaving an orphan
    /// body — so the writer stays broken until the fault is lifted.
    #[derive(Debug)]
    struct BrokenClaimStore {
        inner: Arc<dyn object_store::ObjectStore>,
        fail_new_manifest_bodies: std::sync::atomic::AtomicBool,
    }

    impl BrokenClaimStore {
        fn new(inner: Arc<dyn object_store::ObjectStore>) -> Self {
            Self {
                inner,
                fail_new_manifest_bodies: std::sync::atomic::AtomicBool::new(false),
            }
        }
        fn set_broken(&self, broken: bool) {
            self.fail_new_manifest_bodies
                .store(broken, std::sync::atomic::Ordering::SeqCst);
        }
    }

    impl std::fmt::Display for BrokenClaimStore {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "BrokenClaimStore({})", self.inner)
        }
    }

    #[async_trait::async_trait]
    impl object_store::ObjectStore for BrokenClaimStore {
        async fn put_opts(
            &self,
            location: &object_store::path::Path,
            payload: object_store::PutPayload,
            opts: object_store::PutOptions,
        ) -> object_store::Result<object_store::PutResult> {
            use object_store::ObjectStoreExt as _;
            if self
                .fail_new_manifest_bodies
                .load(std::sync::atomic::Ordering::SeqCst)
                && location.as_ref().contains("manifest/v")
                && matches!(
                    self.inner.head(location).await,
                    Err(object_store::Error::NotFound { .. })
                )
            {
                return Err(object_store::Error::Generic {
                    store: "BrokenClaimStore",
                    source: "injected manifest body put failure".into(),
                });
            }
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
            self.inner.list(prefix)
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

    /// While the writer is fenced AND the reopen cannot succeed, `/v0/health`
    /// must report 503 with `writer: "degraded"`; once the reopen lands it
    /// must report 200 / `writer: "ok"` again.
    #[tokio::test]
    async fn health_reports_degraded_writer_until_reopen_succeeds() {
        let broken = Arc::new(BrokenClaimStore::new(Arc::new(
            object_store::memory::InMemory::new(),
        )));
        let store: Arc<dyn object_store::ObjectStore> = broken.clone();
        let paths = namidb_storage::NamespacePaths::new(
            "",
            namidb_core::NamespaceId::new("health-degraded").unwrap(),
        );

        let writer = WriterSession::open(store.clone(), paths.clone())
            .await
            .unwrap();
        let state = AppState::new(writer, None, "health-degraded".into());
        let app = build_router(state);

        let (status, health) = get_health(&app).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(health["writer"], "ok");

        // Fence the server's writer, then break the claim path so the
        // automatic reopen cannot succeed.
        let interloper = WriterSession::open(store, paths).await.unwrap();
        drop(interloper);
        broken.set_broken(true);

        // The write fails on the fence and every reopen attempt fails too.
        let failed = post_cypher(&app, None, "CREATE (:T {k: 1})").await;
        assert_eq!(failed.status(), StatusCode::INTERNAL_SERVER_ERROR);

        let (status, health) = get_health(&app).await;
        assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE, "{health}");
        assert_eq!(health["status"], "degraded");
        assert_eq!(health["writer"], "degraded");
        assert!(
            health["writer_error"].as_str().unwrap().contains("fenced"),
            "the degraded reason must carry the failure: {health}"
        );

        // Reads still work while the writer is degraded (published snapshot).
        let read = post_cypher(&app, None, "MATCH (n) RETURN n").await;
        assert_eq!(read.status(), StatusCode::OK);

        // Lift the fault: the next failed write triggers a successful reopen…
        broken.set_broken(false);
        let retried = post_cypher(&app, None, "CREATE (:T {k: 2})").await;
        assert_eq!(retried.status(), StatusCode::INTERNAL_SERVER_ERROR);

        // …after which writes succeed and readiness is green again.
        let recovered = post_cypher(&app, None, "CREATE (:T {k: 3}) RETURN 1").await;
        assert_eq!(recovered.status(), StatusCode::OK);
        let (status, health) = get_health(&app).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(health["writer"], "ok");
        assert!(health.get("writer_error").is_none() || health["writer_error"].is_null());
    }

    #[tokio::test]
    async fn livez_and_health_do_not_block_on_the_writer_lock() {
        // A long write or compaction holds the writer lock; the liveness and
        // readiness probes must still answer promptly, or an orchestrator
        // kills a busy-but-healthy server. livez takes no lock; health reads
        // the published snapshot, not the writer.
        let (store, paths) = namidb_storage::parse_uri("memory://livez").unwrap();
        let writer = WriterSession::open(store, paths).await.unwrap();
        let state = AppState::new(writer, None, "test".into());
        let app = build_router(state.clone());

        // Hold the writer lock for the duration of both probes.
        let _guard = state.writer.lock().await;

        for uri in ["/v0/livez", "/v0/health"] {
            let resp = tokio::time::timeout(
                std::time::Duration::from_secs(2),
                app.clone()
                    .oneshot(Request::builder().uri(uri).body(Body::empty()).unwrap()),
            )
            .await
            .unwrap_or_else(|_| panic!("{uri} blocked on the writer lock"))
            .unwrap();
            assert_eq!(resp.status(), StatusCode::OK, "{uri}");
        }
    }

    #[tokio::test]
    async fn version_is_public() {
        let app = fixture(Some("secret")).await;
        let resp = app
            .oneshot(
                Request::builder()
                    .uri("/v0/version")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn cypher_without_auth_is_rejected() {
        let app = fixture(Some("secret")).await;
        let resp = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v0/cypher")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        serde_json::to_vec(&serde_json::json!({"query": "MATCH (n) RETURN n"}))
                            .unwrap(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn cypher_with_auth_roundtrips_create_and_match() {
        let app = fixture(Some("secret")).await;

        // CREATE under auth.
        let create = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v0/cypher")
                    .header("content-type", "application/json")
                    .header("authorization", "Bearer secret")
                    .body(Body::from(
                        serde_json::to_vec(&serde_json::json!({
                            "query": "CREATE (a:Person {name: 'Alice', age: 30}) RETURN a.name AS name"
                        }))
                        .unwrap(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(create.status(), StatusCode::OK);
        let create_body: serde_json::Value =
            serde_json::from_slice(&to_bytes(create.into_body(), 4096).await.unwrap()).unwrap();
        assert_eq!(create_body["write_outcome"]["nodes_created"], 1);

        // MATCH against the just-written node.
        let read = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v0/cypher")
                    .header("content-type", "application/json")
                    .header("authorization", "Bearer secret")
                    .body(Body::from(
                        serde_json::to_vec(&serde_json::json!({
                            "query": "MATCH (p:Person) RETURN p.name AS name, p.age AS age"
                        }))
                        .unwrap(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(read.status(), StatusCode::OK);
        let read_body: serde_json::Value =
            serde_json::from_slice(&to_bytes(read.into_body(), 4096).await.unwrap()).unwrap();
        let rows = read_body["rows"].as_array().unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0]["name"], "Alice");
        assert_eq!(rows[0]["age"], 30);
    }

    #[tokio::test]
    async fn metrics_endpoint_is_public_and_renders_prometheus() {
        // Even with auth set, the scrape carries no bearer token and must
        // still be served, like the health probes.
        let app = fixture(Some("secret")).await;
        let resp = app
            .oneshot(
                Request::builder()
                    .uri("/v0/metrics")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let ct = resp
            .headers()
            .get(axum::http::header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .unwrap_or_default()
            .to_string();
        assert!(ct.starts_with("text/plain"), "content-type was {ct}");
        let body = to_bytes(resp.into_body(), 64 * 1024).await.unwrap();
        let text = String::from_utf8(body.to_vec()).unwrap();
        assert!(text.contains("namidb_build_info{version="));
        assert!(text.contains("namidb_queries_total{protocol=\"http\",status=\"ok\"}"));
        assert!(text.contains("namidb_query_duration_seconds_bucket"));
    }

    #[tokio::test]
    async fn cypher_request_increments_query_metrics() {
        let app = fixture(None).await;

        // One successful read query.
        let read = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v0/cypher")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        serde_json::to_vec(&serde_json::json!({"query": "MATCH (n) RETURN n"}))
                            .unwrap(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(read.status(), StatusCode::OK);

        // The shared metrics registry (Arc on the cloned state) reflects it.
        let scrape = app
            .oneshot(
                Request::builder()
                    .uri("/v0/metrics")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        let body = to_bytes(scrape.into_body(), 64 * 1024).await.unwrap();
        let text = String::from_utf8(body.to_vec()).unwrap();
        assert!(
            text.contains("namidb_queries_total{protocol=\"http\",status=\"ok\"} 1"),
            "metrics did not count the read query:\n{text}"
        );
        assert!(
            text.contains("namidb_query_duration_seconds_count{protocol=\"http\",kind=\"read\"} 1"),
            "read histogram did not record the query:\n{text}"
        );
    }

    #[tokio::test]
    async fn parse_error_is_400() {
        let app = fixture(None).await;
        let resp = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v0/cypher")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        serde_json::to_vec(&serde_json::json!({"query": "NOT VALID CYPHER"}))
                            .unwrap(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn unsupported_function_is_typed_400_not_500() {
        // An unknown function is a deliberately-unsupported feature, not an
        // internal bug — it must surface as 400 with code:"unsupported", not
        // a bare 500 (item 11: typed "not supported" errors).
        let app = fixture(None).await;
        let resp = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v0/cypher")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        serde_json::to_vec(&serde_json::json!({
                            "query": "RETURN bogus_function(1) AS x"
                        }))
                        .unwrap(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let body = to_bytes(resp.into_body(), 4096).await.unwrap();
        let v: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(v["code"], "unsupported", "body was: {v}");
    }

    /// Build a multi-tenant router over a fresh memory store with the given
    /// default namespace, open auth, and maintenance disabled.
    async fn multi_tenant_app(default_ns: &str) -> Router {
        let (store, _) = namidb_storage::parse_uri("memory://multi-tenant-test").unwrap();
        let metrics = Metrics::new(env!("CARGO_PKG_VERSION"), Duration::ZERO);
        let registry = Arc::new(registry::NamespaceRegistry::new(
            store,
            String::new(),
            0,
            Duration::from_secs(3600),
            metrics.clone(),
            registry::MaintenanceConfig::default(),
        ));
        let shared = SharedAppState::new_with_memory(
            registry,
            Arc::new(AuthConfig::open()),
            metrics,
            Duration::ZERO,
            Duration::ZERO,
            0,
            0,
            Duration::ZERO,
            0,
            0,
            Arc::new(memory::MemoryGovernor::new(0)),
            Duration::ZERO,
            default_ns.to_string(),
        );
        build_multi_tenant_router(shared)
    }

    async fn mt_cypher(
        app: &Router,
        uri: &str,
        header_ns: Option<&str>,
        query: &str,
    ) -> StatusCode {
        let mut b = Request::builder()
            .method("POST")
            .uri(uri)
            .header("content-type", "application/json");
        if let Some(ns) = header_ns {
            b = b.header("x-namidb-namespace", ns);
        }
        app.clone()
            .oneshot(
                b.body(Body::from(
                    serde_json::to_vec(&serde_json::json!({ "query": query })).unwrap(),
                ))
                .unwrap(),
            )
            .await
            .unwrap()
            .status()
    }

    #[tokio::test]
    async fn multi_tenant_path_prefix_header_and_default_all_route() {
        let app = multi_tenant_app("default").await;
        // 1. Explicit path prefix.
        let q = "CREATE (:Person {name: 'x'}) RETURN 1";
        assert_eq!(
            mt_cypher(&app, "/acme/v0/cypher", None, q).await,
            StatusCode::OK
        );
        // 2. X-NamiDB-Namespace header on an unprefixed path.
        assert_eq!(
            mt_cypher(&app, "/v0/cypher", Some("beta"), q).await,
            StatusCode::OK
        );
        // 3. No prefix, no header → default namespace.
        assert_eq!(mt_cypher(&app, "/v0/cypher", None, q).await, StatusCode::OK);
        // The default namespace is genuinely distinct: a note written to
        // `acme` is NOT visible via the default namespace.
        let app = multi_tenant_app("default").await;
        let _ = mt_cypher(
            &app,
            "/acme/v0/cypher",
            None,
            "CREATE (:Person {name: 'only-acme'})",
        )
        .await;
        let read = mt_cypher(&app, "/v0/cypher", None, "MATCH (p:Person) RETURN count(p)").await;
        assert_eq!(
            read,
            StatusCode::OK,
            "default namespace is isolated from acme"
        );
    }

    #[tokio::test]
    async fn memory_admission_does_not_open_a_cold_namespace() {
        let (store, _) = namidb_storage::parse_uri("memory://mt-memory-cold").unwrap();
        let metrics = Metrics::new(env!("CARGO_PKG_VERSION"), Duration::ZERO);
        let registry = Arc::new(registry::NamespaceRegistry::new(
            store,
            String::new(),
            0,
            Duration::from_secs(3600),
            metrics.clone(),
            registry::MaintenanceConfig::default(),
        ));
        let governor = Arc::new(memory::MemoryGovernor::new(1));
        let shared = SharedAppState::new_with_memory(
            Arc::clone(&registry),
            Arc::new(AuthConfig::open()),
            metrics,
            Duration::ZERO,
            Duration::ZERO,
            0,
            0,
            Duration::ZERO,
            0,
            0,
            Arc::clone(&governor),
            Duration::ZERO,
            "default".to_string(),
        );
        let app = build_multi_tenant_router(shared);

        let resp = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/cold/v0/cypher")
                    .header("content-type", "application/json")
                    // Also proves admission precedes the namespace handler's
                    // JSON extraction.
                    .body(Body::from("{"))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert!(
            registry.is_empty().await,
            "memory-rejected work must not recover/open a cold namespace"
        );
        assert_eq!(governor.rejected_queries(), 1);

        let flush = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/cold/v0/admin/flush")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(flush.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert!(
            registry.is_empty().await,
            "pressure-relief flush must not recover a namespace with no live memtable"
        );
        assert_eq!(
            governor.rejected_queries(),
            1,
            "admin maintenance is not a rejected query"
        );
    }

    #[tokio::test]
    async fn admin_flush_remains_available_for_an_open_namespace_under_pressure() {
        let (store, _) = namidb_storage::parse_uri("memory://mt-memory-open-flush").unwrap();
        let metrics = Metrics::new(env!("CARGO_PKG_VERSION"), Duration::ZERO);
        let registry = Arc::new(registry::NamespaceRegistry::new(
            store,
            String::new(),
            0,
            Duration::from_secs(3600),
            metrics.clone(),
            registry::MaintenanceConfig::default(),
        ));
        let ns = registry.get_or_open("active").await.unwrap();
        {
            let mut writer = ns.writer.lock().await;
            writer
                .upsert_node(
                    "Pressure",
                    namidb_core::id::NodeId::new(),
                    &namidb_storage::NodeWriteRecord {
                        properties: std::collections::BTreeMap::new(),
                        schema_version: 1,
                        ..Default::default()
                    },
                )
                .unwrap();
            writer.commit_batch().await.unwrap();
        }
        let before_version = ns.snapshot.load().manifest_version();
        let governor = Arc::new(memory::MemoryGovernor::new(1));
        let shared = SharedAppState::new_with_memory(
            Arc::clone(&registry),
            Arc::new(AuthConfig::open()),
            metrics,
            Duration::ZERO,
            Duration::ZERO,
            0,
            0,
            Duration::ZERO,
            0,
            0,
            Arc::clone(&governor),
            Duration::ZERO,
            "default".to_string(),
        );
        let app = build_multi_tenant_router(shared);

        let flush = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/active/v0/admin/flush")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(flush.status(), StatusCode::OK);
        assert!(ns.snapshot.load().manifest_version() > before_version);
        assert_eq!(registry.len().await, 1);
        assert_eq!(governor.rejected_queries(), 0);
    }

    /// A request may clone a namespace state, finish planning, and queue on
    /// its writer before eviction retires that incarnation. It must
    /// revalidate only after the mutex becomes available and leave storage
    /// untouched instead of reviving/fencing the old WriterSession.
    #[tokio::test]
    async fn multi_tenant_write_queued_on_retired_incarnation_is_rejected() {
        let (store, _) = namidb_storage::parse_uri("memory://mt-retired-writer").unwrap();
        let metrics = Metrics::new(env!("CARGO_PKG_VERSION"), Duration::ZERO);
        let registry = Arc::new(registry::NamespaceRegistry::new(
            store,
            String::new(),
            0,
            Duration::from_secs(3600),
            metrics.clone(),
            registry::MaintenanceConfig::default(),
        ));
        let shared = SharedAppState::new_with_memory(
            Arc::clone(&registry),
            Arc::new(AuthConfig::open()),
            metrics,
            Duration::ZERO,
            Duration::ZERO,
            0,
            0,
            Duration::ZERO,
            0,
            0,
            Arc::new(memory::MemoryGovernor::new(0)),
            Duration::ZERO,
            "default".to_string(),
        );
        let state = registry.get_or_open("acme").await.expect("open acme");
        let before_version = state.snapshot.load().manifest().manifest.version;

        let active_writer = state.writer.lock().await;
        let request = CypherRequest {
            query: "CREATE (:Person {name: 'must-not-commit'})".to_string(),
            params: serde_json::Map::new(),
        };
        let principal = Principal::anonymous_rw();
        let queued = run_cypher_multi(&state, &shared, &request, &principal);
        tokio::pin!(queued);

        // Poll through parsing/planning into the held writer mutex. Timing
        // out a borrowed pinned future leaves it alive and queued.
        assert!(
            tokio::time::timeout(Duration::from_millis(20), &mut queued)
                .await
                .is_err(),
            "write unexpectedly completed while the writer mutex was held"
        );
        state.mark_retired();
        drop(active_writer);

        let observed = queued.await;
        assert_eq!(observed.response.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(
            state.snapshot.load().manifest().manifest.version,
            before_version,
            "a request queued on a retired incarnation mutated storage"
        );
    }

    /// The multi-tenant registry holds its own per-namespace writers; a
    /// fenced one must recover through the same reopen orchestration as the
    /// single-tenant path, and the namespace health probe must reflect it.
    #[tokio::test]
    async fn multi_tenant_write_path_recovers_after_fencing() {
        let (store, _) = namidb_storage::parse_uri("memory://mt-fence").unwrap();
        let metrics = Metrics::new(env!("CARGO_PKG_VERSION"), Duration::ZERO);
        let registry = Arc::new(registry::NamespaceRegistry::new(
            store.clone(),
            String::new(),
            0,
            Duration::from_secs(3600),
            metrics.clone(),
            registry::MaintenanceConfig::default(),
        ));
        let shared = SharedAppState::new_with_memory(
            registry,
            Arc::new(AuthConfig::open()),
            metrics,
            Duration::ZERO,
            Duration::ZERO,
            0,
            0,
            Duration::ZERO,
            0,
            0,
            Arc::new(memory::MemoryGovernor::new(0)),
            Duration::ZERO,
            "default".to_string(),
        );
        let app = build_multi_tenant_router(shared);

        let q1 = "CREATE (:P {n: 1}) RETURN 1";
        assert_eq!(
            mt_cypher(&app, "/acme/v0/cypher", None, q1).await,
            StatusCode::OK
        );

        // Fence the registry-held writer for `acme` (flat layout: root "").
        let paths =
            namidb_storage::NamespacePaths::new("", namidb_core::NamespaceId::new("acme").unwrap());
        let interloper = WriterSession::open(store, paths).await.unwrap();
        drop(interloper);

        // The fenced write fails once, triggers the reopen, then writes flow.
        assert_eq!(
            mt_cypher(&app, "/acme/v0/cypher", None, "CREATE (:P {n: 2}) RETURN 1").await,
            StatusCode::INTERNAL_SERVER_ERROR
        );
        assert_eq!(
            mt_cypher(&app, "/acme/v0/cypher", None, "CREATE (:P {n: 3}) RETURN 1").await,
            StatusCode::OK,
            "the multi-tenant write path must recover automatically"
        );

        // The namespace readiness probe is green again.
        let resp = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/acme/v0/health")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let health: serde_json::Value =
            serde_json::from_slice(&to_bytes(resp.into_body(), 4096).await.unwrap()).unwrap();
        assert_eq!(health["writer"], "ok");
    }

    async fn multi_tenant_app_auth(auth: Arc<AuthConfig>, default_ns: &str) -> Router {
        let (store, _) = namidb_storage::parse_uri("memory://multi-tenant-scoped").unwrap();
        let metrics = Metrics::new(env!("CARGO_PKG_VERSION"), Duration::ZERO);
        let registry = Arc::new(registry::NamespaceRegistry::new(
            store,
            String::new(),
            0,
            Duration::from_secs(3600),
            metrics.clone(),
            registry::MaintenanceConfig::default(),
        ));
        let shared = SharedAppState::new_with_memory(
            registry,
            auth,
            metrics,
            Duration::ZERO,
            Duration::ZERO,
            0,
            0,
            Duration::ZERO,
            0,
            0,
            Arc::new(memory::MemoryGovernor::new(0)),
            Duration::ZERO,
            default_ns.to_string(),
        );
        build_multi_tenant_router(shared)
    }

    async fn mt_cypher_token(
        app: &Router,
        uri: &str,
        header_ns: Option<&str>,
        token: &str,
        query: &str,
    ) -> StatusCode {
        let mut b = Request::builder()
            .method("POST")
            .uri(uri)
            .header("content-type", "application/json")
            .header("authorization", format!("Bearer {token}"));
        if let Some(ns) = header_ns {
            b = b.header("x-namidb-namespace", ns);
        }
        app.clone()
            .oneshot(
                b.body(Body::from(
                    serde_json::to_vec(&serde_json::json!({ "query": query })).unwrap(),
                ))
                .unwrap(),
            )
            .await
            .unwrap()
            .status()
    }

    /// Per-namespace token scoping (RFC-015 Wave B): a token scoped to one
    /// namespace is rejected (401) on every other namespace, on BOTH the
    /// prefixed path and the header-routed unprefixed path. Closes the
    /// cross-namespace reach gap.
    #[tokio::test]
    async fn scoped_token_cannot_reach_other_namespaces() {
        let json = r#"{ "tokens": [
            { "name": "acme", "token": "acme-key", "role": "read-write", "namespaces": ["acme"] },
            { "name": "beta", "token": "beta-key", "role": "read-write", "namespaces": ["beta"] }
        ] }"#;
        let path = std::env::temp_dir().join("namidb-test-scoped-tokens.json");
        std::fs::write(&path, json).unwrap();
        let auth = Arc::new(AuthConfig::load_file(&path).unwrap());
        std::fs::remove_file(&path).ok();
        let app = multi_tenant_app_auth(auth, "default").await;
        let q = "RETURN 1";

        // acme-key reaches acme (prefixed path) ...
        assert_eq!(
            mt_cypher_token(&app, "/acme/v0/cypher", None, "acme-key", q).await,
            StatusCode::OK
        );
        // ... but is rejected on beta (prefixed).
        assert_eq!(
            mt_cypher_token(&app, "/beta/v0/cypher", None, "acme-key", q).await,
            StatusCode::UNAUTHORIZED
        );
        // ... and rejected via the unprefixed path + header routing to beta.
        assert_eq!(
            mt_cypher_token(&app, "/v0/cypher", Some("beta"), "acme-key", q).await,
            StatusCode::UNAUTHORIZED
        );
        // beta-key reaches beta but not acme.
        assert_eq!(
            mt_cypher_token(&app, "/beta/v0/cypher", None, "beta-key", q).await,
            StatusCode::OK
        );
        assert_eq!(
            mt_cypher_token(&app, "/acme/v0/cypher", None, "beta-key", q).await,
            StatusCode::UNAUTHORIZED
        );
        // Either token is rejected on the default namespace (neither is scoped to it).
        assert_eq!(
            mt_cypher_token(&app, "/v0/cypher", None, "acme-key", q).await,
            StatusCode::UNAUTHORIZED
        );
    }

    /// Regression for the `/v0/v0/...` scoping bypass: a path whose namespace
    /// segment is literally `v0` routes to the PREFIXED handler (Path = "v0"),
    /// so the auth middleware must gate namespace `v0` — NOT fall back to the
    /// header. An acme-scoped token sending `/v0/v0/cypher` with header
    /// `acme` must be REJECTED (it is not scoped to `v0`). Before the fix the
    /// middleware hand-parsed the path, saw the `/v0/` prefix, and authorized
    /// against the header's `acme` while the handler served `v0` — a bypass.
    #[tokio::test]
    async fn v0_namespace_cannot_be_reached_by_path_shadowing() {
        let json = r#"{ "tokens": [
            { "name": "acme", "token": "acme-key", "role": "read-write", "namespaces": ["acme"] },
            { "name": "v0", "token": "v0-key", "role": "read-write", "namespaces": ["v0"] }
        ] }"#;
        let path = std::env::temp_dir().join("namidb-test-v0-shadow.json");
        std::fs::write(&path, json).unwrap();
        let auth = Arc::new(AuthConfig::load_file(&path).unwrap());
        std::fs::remove_file(&path).ok();
        let app = multi_tenant_app_auth(auth, "default").await;
        let q = "RETURN 1";

        // acme-key targeting the `v0` tenant via /v0/v0/... + header=acme: the
        // gate must check namespace `v0` (the routed param), not `acme`.
        assert_eq!(
            mt_cypher_token(&app, "/v0/v0/cypher", Some("acme"), "acme-key", q).await,
            StatusCode::UNAUTHORIZED,
            "acme-scoped token must not reach the v0 tenant via path shadowing"
        );
        // The correctly-scoped v0-key DOES reach it through the same path.
        assert_eq!(
            mt_cypher_token(&app, "/v0/v0/cypher", None, "v0-key", q).await,
            StatusCode::OK,
            "v0-scoped token reaches the v0 tenant"
        );
    }
}
