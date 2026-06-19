//! HTTP server exposing a NamiDB namespace.
//!
//! The crate is split between a thin [`main`] CLI parser and this
//! library so integration tests can exercise the routes directly.
//!
//! See [`build_router`] for the full route surface and [`run`] for
//! the end-to-end boot procedure.

pub mod auth;
pub mod bolt;
mod introspect;
pub mod metrics;
pub mod registry;
pub mod shared;
pub mod tls;

use std::sync::Arc;
use std::time::Duration;

use axum::extract::{Extension, Path, State};
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

use crate::auth::{AuthConfig, Role};
use crate::metrics::{Metrics, Protocol, QueryKind};
use crate::registry::{NamespaceRegistry, NamespaceState};
use crate::shared::SharedAppState;

/// Process-wide configuration assembled from CLI flags or env vars.
#[derive(Debug, Clone)]
pub struct Config {
    pub store_uri: String,
    pub listen: std::net::SocketAddr,
    /// `None` means "no auth"; the server will log a loud warning at
    /// boot and accept every request. Production callers should set a
    /// long random secret. A single token grants read-write access; for
    /// read-only tokens or several tokens, use `auth_tokens_file`.
    pub auth_token: Option<String>,
    /// Path to a JSON file of tokens, each with a `read-only` or
    /// `read-write` role. Takes precedence over `auth_token` when set. `None`
    /// falls back to `auth_token` (or no auth when that is also `None`).
    pub auth_tokens_file: Option<std::path::PathBuf>,
    pub flush_interval: Duration,
    /// Interval for the background maintenance task (L0->L1 compaction +
    /// orphan sweep). `Duration::ZERO` disables it.
    pub compaction_interval: Duration,
    /// Minimum age before the orphan sweep may delete an unreferenced SST
    /// body — the sole guard against deleting a file a slow reader's pinned
    /// snapshot still references.
    pub sweep_min_age: Duration,
    /// When `true` (the default) the orphan sweep deletes unreferenced SST
    /// bodies; the retention horizon (RFC-027) makes that safe by
    /// construction. Set `false` for a dry-run that only logs what it would
    /// free.
    pub sweep_delete: bool,
    /// Bolt listener address. `None` keeps the protocol off (HTTP only).
    pub bolt_listen: Option<std::net::SocketAddr>,
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
    /// Process-wide query metrics, shared across every connection on both
    /// serving paths. Rendered at `/v0/metrics` and the home of the
    /// slow-query log. Defaults to a registry with the slow-query log
    /// disabled; the server sets the threshold from [`Config`] at boot.
    pub metrics: Arc<Metrics>,
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
            catalog_cache: Arc::new(std::sync::Mutex::new(None)),
            auth: Arc::new(auth),
            namespace,
            query_timeout: Duration::ZERO,
            write_timeout: Duration::ZERO,
            query_row_cap: 0,
            write_stall_l0: 0,
            write_stall_delay: Duration::ZERO,
            metrics: Metrics::new(env!("CARGO_PKG_VERSION"), Duration::ZERO),
        }
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

    /// If a write should be stalled given the worst bucket's current L0
    /// count, the delay to apply; otherwise `None`. The caller samples
    /// `max_l0_bucket_len()` while holding the writer lock, then sleeps
    /// after releasing it.
    pub(crate) fn write_stall_for(&self, max_l0: usize) -> Option<Duration> {
        (self.write_stall_l0 > 0
            && max_l0 >= self.write_stall_l0
            && self.write_stall_delay > Duration::ZERO)
            .then_some(self.write_stall_delay)
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

    let private = Router::new()
        .route("/v0/cypher", post(cypher))
        .route("/v0/admin/flush", post(admin_flush))
        .layer(middleware::from_fn_with_state(state.clone(), require_auth));

    Router::new().merge(public).merge(private).with_state(state)
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
    let namespace_routes = Router::new()
        .route("/:namespace/v0/livez", get(livez_multi))
        .route("/:namespace/v0/health", get(health_multi))
        .route("/:namespace/v0/cypher", post(cypher_multi))
        .route("/:namespace/v0/admin/flush", post(admin_flush_multi))
        .route("/v0/livez", get(livez_multi))
        .route("/v0/health", get(health_multi_unprefixed))
        .route("/v0/cypher", post(cypher_multi_unprefixed))
        .route("/v0/admin/flush", post(admin_flush_multi_unprefixed))
        .layer(middleware::from_fn_with_state(shared.clone(), require_auth_multi));

    Router::new()
        .merge(public)
        .merge(namespace_routes)
        .with_state(shared)
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

/// Boot the server: parse URI, open a `WriterSession`, optionally
/// spawn a periodic flush task, and serve until the process receives
/// SIGINT.
pub async fn run(config: Config) -> anyhow::Result<()> {
    // Resolve the auth configuration: a tokens file (with roles) wins, else a
    // single read-write `--auth-token`, else open.
    let auth = match (&config.auth_tokens_file, &config.auth_token) {
        (Some(path), _) => Arc::new(AuthConfig::load_file(path)?),
        // Refuse an empty `--auth-token`: it logs as "auth enabled" but a
        // `Bearer ` request would match the empty secret. Omit it to run open.
        (None, Some(secret)) if secret.is_empty() => {
            anyhow::bail!(
                "--auth-token is empty; omit it (and NAMIDB_AUTH_TOKEN) to run without auth"
            )
        }
        (None, Some(secret)) => Arc::new(AuthConfig::single_read_write(secret.clone())),
        (None, None) => Arc::new(AuthConfig::open()),
    };
    if auth.is_open() {
        warn!(
            "⚠️  namidb-server is running WITHOUT auth. Anyone who can reach \
             {} can issue arbitrary Cypher queries. Set --auth-token (or env \
             NAMIDB_AUTH_TOKEN), or --auth-tokens-file for per-token roles, \
             before exposing this port beyond localhost.",
            config.listen
        );
    } else {
        info!(tokens = auth.len(), "auth enabled");
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
        let shared = SharedAppState::new(
            registry,
            auth,
            metrics,
            config.query_timeout,
            config.write_timeout,
            config.query_row_cap,
            config.write_stall_l0,
            config.write_stall_delay,
            config.default_namespace.clone(),
        );
        let app = build_multi_tenant_router(shared);

        // Shutdown signal.
        let (shutdown_tx, shutdown_rx) = tokio::sync::watch::channel(false);
        tokio::spawn(async move {
            wait_for_shutdown_signal().await;
            let _ = shutdown_tx.send(true);
        });

        // TLS on the serving path.
        let tls_config: Option<Arc<rustls::ServerConfig>> = match (&config.tls_cert, &config.tls_key) {
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
        .with_query_timeout(config.query_timeout)
        .with_write_timeout(config.write_timeout)
        .with_query_row_cap(config.query_row_cap)
        .with_write_stall(config.write_stall_l0, config.write_stall_delay)
        .with_slow_query_threshold(config.slow_query_threshold);

    // Periodic flush task — keeps the WAL bounded and L0 SSTs current.
    if config.flush_interval > Duration::ZERO {
        let state_for_flush = state.clone();
        let interval = config.flush_interval;
        // Reactive compaction trigger (RFC-027 P5): when a flush leaves a
        // bucket with >= this many L0 SSTs, compact immediately under the
        // same writer lock rather than waiting for the periodic compaction
        // tick, so read amplification does not spike between ticks.
        let l0_trigger = config.compaction_l0_trigger;
        tokio::spawn(async move {
            let mut tick = tokio::time::interval(interval);
            tick.tick().await; // first tick fires immediately; skip.
            loop {
                tick.tick().await;
                let mut w = state_for_flush.writer.lock().await;
                let schema = w.snapshot().manifest().manifest.schema.clone();
                match w.flush(schema.clone()).await {
                    Ok(_) => {
                        state_for_flush.snapshot.store(w.owned_snapshot());
                        if l0_trigger > 0 && w.max_l0_bucket_len() >= l0_trigger {
                            match w.compact_l0(&schema).await {
                                Ok(outcome) if outcome.source_ssts_removed > 0 => {
                                    state_for_flush.snapshot.store(w.owned_snapshot());
                                    info!(
                                        removed = outcome.source_ssts_removed,
                                        written = outcome.new_ssts_written,
                                        "reactive compaction (L0 high-water)"
                                    );
                                }
                                Ok(_) => {}
                                Err(e) => error!(error = %e, "reactive compaction failed"),
                            }
                        }
                    }
                    Err(e) => error!(error = %e, "periodic flush failed"),
                }
            }
        });
    }

    // Periodic background maintenance: compact L0 SSTs to L1 (bounds read
    // amplification), then sweep orphaned SST bodies left behind by
    // compaction. Compaction is a writer mutation, so it goes through the
    // ONE writer lock — never a second `WriterSession`, which would bump the
    // epoch and fence the foreground writer. The sweep takes no lock (it
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
            tick.tick().await; // first tick fires immediately; skip.
            loop {
                tick.tick().await;
                // Compaction under the writer lock. `compact_l0` self-no-ops
                // below 2 L0 SSTs per bucket, so an idle tick is cheap and
                // does not commit. Republish only when it actually merged,
                // so reads pick up the new L1 SST and release the removed L0
                // descriptors.
                {
                    let mut w = state_for_maint.writer.lock().await;
                    let schema = w.snapshot().manifest().manifest.schema.clone();
                    match w.compact_l0(&schema).await {
                        Ok(outcome) if outcome.source_ssts_removed > 0 => {
                            state_for_maint.snapshot.store(w.owned_snapshot());
                            info!(
                                removed = outcome.source_ssts_removed,
                                written = outcome.new_ssts_written,
                                "compacted L0 into L1"
                            );
                        }
                        Ok(_) => {}
                        Err(e) => error!(error = %e, "periodic compaction failed"),
                    }
                }
                // Orphan sweep — no writer lock. `max_level = 1` because the
                // engine only produces L0 + L1 today. The retention horizon
                // (RFC-027) is the oldest manifest version any live reader is
                // pinned to; the sweep keeps every object referenced from the
                // horizon to current, so it can never delete a body a reader
                // still needs.
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
                            || report.pointer_files_reclaimed > 0 =>
                    {
                        info!(
                            found = report.orphans_found,
                            deleted = report.orphans_deleted,
                            bytes_freed = report.bytes_freed,
                            manifest_snapshots = report.manifest_snapshots_reclaimed,
                            manifest_bytes_freed = report.manifest_bytes_freed,
                            pointer_files = report.pointer_files_reclaimed,
                            pointer_bytes_freed = report.pointer_bytes_freed,
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
    // A single shutdown signal, flipped to `true` on SIGINT or SIGTERM, that
    // both the HTTP server and the Bolt listener observe, so a `docker stop`
    // or a Kubernetes pod termination drains cleanly instead of being killed.
    let (shutdown_tx, shutdown_rx) = tokio::sync::watch::channel(false);
    tokio::spawn(async move {
        wait_for_shutdown_signal().await;
        let _ = shutdown_tx.send(true);
    });

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
        let bolt_tls = tls_config.clone().map(|c| tls::acceptor(c));
        tokio::spawn(async move {
            if let Err(e) = bolt::serve(
                bolt_state,
                bolt_addr,
                bolt_auth,
                tx_timeout,
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

async fn require_auth_multi(
    State(shared): State<SharedAppState>,
    mut req: axum::extract::Request,
    next: Next,
) -> Response {
    // Open mode: serve every request as read-write.
    if shared.auth.is_open() {
        req.extensions_mut().insert(Role::ReadWrite);
        return next.run(req).await;
    }
    let presented = req
        .headers()
        .get(axum::http::header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .and_then(strip_bearer);
    match presented.and_then(|token| shared.auth.role_for(token)) {
        Some(role) => {
            req.extensions_mut().insert(role);
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

async fn require_auth(
    State(state): State<AppState>,
    mut req: axum::extract::Request,
    next: Next,
) -> Response {
    // Open mode: serve every request as read-write, recording the role so the
    // handler's write gate has a value to read uniformly.
    if state.auth.is_open() {
        req.extensions_mut().insert(Role::ReadWrite);
        return next.run(req).await;
    }
    let presented = req
        .headers()
        .get(axum::http::header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .and_then(strip_bearer);
    match presented.and_then(|token| state.auth.role_for(token)) {
        Some(role) => {
            // Carry the matched token's role to the handler, which gates writes.
            req.extensions_mut().insert(role);
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

#[derive(Serialize)]
struct HealthResponse {
    status: &'static str,
    namespace: String,
    manifest_version: u64,
    epoch: u64,
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
/// holding the writer lock does not stall the probe.
async fn health(State(state): State<AppState>) -> impl IntoResponse {
    let owned = state.snapshot.load();
    let m = &owned.manifest().manifest;
    Json(HealthResponse {
        status: "ok",
        namespace: state.namespace.clone(),
        manifest_version: m.version,
        epoch: m.epoch.as_u64(),
    })
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
    (
        [(
            axum::http::header::CONTENT_TYPE,
            HeaderValue::from_static("text/plain; version=0.0.4; charset=utf-8"),
        )],
        state.metrics.render(),
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

async fn cypher(
    State(state): State<AppState>,
    Extension(role): Extension<Role>,
    Json(req): Json<CypherRequest>,
) -> Response {
    // The guard drops at the end of the handler, so the in-flight gauge is
    // correct even on an early error return.
    let _in_flight = state.metrics.track_in_flight();
    let obs = run_cypher(&state, &req, role).await;
    state
        .metrics
        .observe_query(Protocol::Http, obs.kind, obs.ok, obs.elapsed, &req.query);
    obs.response
}

/// Run one HTTP Cypher request and classify it for metrics. Mirrors the Bolt
/// `ServerBackend::run` path; the two do not share a chokepoint, so the
/// parse/plan/execute logic is intentionally parallel.
async fn run_cypher(state: &AppState, req: &CypherRequest, role: Role) -> ObservedQuery {
    let started = std::time::Instant::now();

    let parsed = match cypher_parse(&req.query) {
        Ok(p) => p,
        Err(errs) => {
            let first = &errs[0];
            return ObservedQuery {
                kind: None,
                ok: false,
                elapsed: started.elapsed(),
                response: (
                    StatusCode::BAD_REQUEST,
                    Json(ErrorBody {
                        error: format!("parse error: {} at {}", first.message, first.span),
                    }),
                )
                    .into_response(),
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

    // Plan against the latest published snapshot — no writer lock yet.
    let owned = state.snapshot.load();
    let plan = {
        let catalog = state.catalog_for(&owned.manifest().manifest);
        match build_plan(&parsed, &catalog) {
            Ok(p) => p,
            Err(e) => {
                return ObservedQuery {
                    kind: None,
                    ok: false,
                    elapsed: started.elapsed(),
                    response: (
                        StatusCode::BAD_REQUEST,
                        Json(ErrorBody {
                            error: format!("plan error: {e}"),
                        }),
                    )
                        .into_response(),
                };
            }
        }
    };

    if plan.contains_write() {
        // A read-only token may not write. Reject before taking the writer
        // lock so a forbidden write costs nothing.
        if !role.allows_write() {
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
        let mut writer = state.writer.lock().await;
        let result =
            execute_write_with_deadline(&plan, &mut writer, &params, state.write_deadline()).await;
        // Sample the soft write-stall decision while still holding the lock
        // (RFC-027 P5), then release it and sleep — backpressure applies to
        // this request, not to the writer mutex other connections need.
        let stall = if result.is_ok() {
            // Refresh the published snapshot so subsequent reads see the
            // just-committed records (RFC-021).
            state.snapshot.store(writer.owned_snapshot());
            state.write_stall_for(writer.max_l0_bucket_len())
        } else {
            None
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
                response: (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(ErrorBody {
                        error: format!("write execution failed: {e}"),
                    }),
                )
                    .into_response(),
            },
        }
    } else {
        // Read path: no writer lock. Borrow a short-lived `Snapshot`
        // from the owned one; the `OwnedSnapshot` Arc keeps the
        // underlying memtable alive for the duration of the query.
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
                response: (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(ErrorBody {
                        error: format!("read execution failed: {e}"),
                    }),
                )
                    .into_response(),
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

async fn admin_flush(State(state): State<AppState>, Extension(role): Extension<Role>) -> Response {
    // A flush mutates durable state, so a read-only token may not trigger it.
    if !role.allows_write() {
        return (
            StatusCode::FORBIDDEN,
            Json(ErrorBody {
                error: "this token is read-only; admin flush is forbidden".into(),
            }),
        )
            .into_response();
    }
    let mut w = state.writer.lock().await;
    let schema = w.snapshot().manifest().manifest.schema.clone();
    match w.flush(schema).await {
        Ok(outcome) => {
            state.snapshot.store(w.owned_snapshot());
            Json(FlushResponse {
                ssts_written: outcome.ssts_written,
                bloom_sidecars_written: outcome.bloom_sidecars_written,
                manifest_version: outcome.committed.manifest.version,
            })
            .into_response()
        }
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorBody {
                error: format!("flush failed: {e}"),
            }),
        )
            .into_response(),
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
    match shared.registry.get_or_open(&namespace).await {
        Ok(ns_state) => {
            let owned = ns_state.snapshot.load();
            let m = &owned.manifest().manifest;
            Json(HealthResponse {
                status: "ok",
                namespace,
                manifest_version: m.version,
                epoch: m.epoch.as_u64(),
            })
            .into_response()
        }
        Err(e) => {
            (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(ErrorBody { error: e.to_string() }),
            )
                .into_response()
        }
    }
}

/// Execute a Cypher query in multi-tenant mode.
async fn cypher_multi(
    Path(namespace): Path<String>,
    State(shared): State<SharedAppState>,
    Extension(role): Extension<Role>,
    Json(req): Json<CypherRequest>,
) -> Response {
    dispatch_cypher_multi(&shared, &namespace, role, req).await
}

/// Unprefixed entry point: resolve the namespace from the
/// `X-NamiDB-Namespace` header (or the default), then run the query. Used by
/// the `/v0/cypher` route in multi-tenant mode so clients can target a
/// namespace without a path prefix.
async fn cypher_multi_unprefixed(
    State(shared): State<SharedAppState>,
    Extension(role): Extension<Role>,
    headers: axum::http::HeaderMap,
    Json(req): Json<CypherRequest>,
) -> Response {
    let namespace = namespace_from_header(&shared, &headers);
    dispatch_cypher_multi(&shared, &namespace, role, req).await
}

/// Shared body of the multi-tenant cypher handler: open the namespace, run,
/// observe metrics.
async fn dispatch_cypher_multi(
    shared: &SharedAppState,
    namespace: &str,
    role: Role,
    req: CypherRequest,
) -> Response {
    let _in_flight = shared.metrics.track_in_flight();

    // Get or create the namespace state.
    let ns_state = match shared.registry.get_or_open(namespace).await {
        Ok(ns) => ns,
        Err(e) => {
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(ErrorBody { error: e.to_string() }),
            )
                .into_response();
        }
    };

    let obs = run_cypher_multi(&ns_state, shared, &req, role).await;
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
    role: Role,
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
                response: (
                    StatusCode::BAD_REQUEST,
                    Json(ErrorBody {
                        error: format!("parse error: {} at {}", first.message, first.span),
                    }),
                )
                    .into_response(),
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

    // Plan against the latest published snapshot.
    let owned = ns_state.snapshot.load();
    let plan = {
        let catalog = StatsCatalog::from_manifest(&owned.manifest().manifest);
        match build_plan(&parsed, &catalog) {
            Ok(p) => p,
            Err(e) => {
                return ObservedQuery {
                    kind: None,
                    ok: false,
                    elapsed: started.elapsed(),
                    response: (
                        StatusCode::BAD_REQUEST,
                        Json(ErrorBody {
                            error: format!("plan error: {e}"),
                        }),
                    )
                        .into_response(),
                };
            }
        }
    };

    if plan.contains_write() {
        // A read-only token may not write.
        if !role.allows_write() {
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
        let mut writer = ns_state.writer.lock().await;
        let result = execute_write_with_deadline(&plan, &mut writer, &params, shared.write_deadline()).await;
        let stall = if result.is_ok() {
            ns_state.snapshot.store(writer.owned_snapshot());
            shared.write_stall_for(writer.max_l0_bucket_len())
        } else {
            None
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
                response: (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(ErrorBody {
                        error: format!("write execution failed: {e}"),
                    }),
                )
                    .into_response(),
            },
        }
    } else {
        // Read path.
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
                response: (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(ErrorBody {
                        error: format!("read execution failed: {e}"),
                    }),
                )
                    .into_response(),
            },
        }
    }
}

/// Admin flush in multi-tenant mode.
async fn admin_flush_multi(
    Path(namespace): Path<String>,
    State(shared): State<SharedAppState>,
    Extension(role): Extension<Role>,
) -> Response {
    dispatch_admin_flush_multi(&shared, &namespace, role).await
}

/// Unprefixed admin flush: resolve namespace from header/default.
async fn admin_flush_multi_unprefixed(
    State(shared): State<SharedAppState>,
    Extension(role): Extension<Role>,
    headers: axum::http::HeaderMap,
) -> Response {
    let namespace = namespace_from_header(&shared, &headers);
    dispatch_admin_flush_multi(&shared, &namespace, role).await
}

async fn dispatch_admin_flush_multi(
    shared: &SharedAppState,
    namespace: &str,
    role: Role,
) -> Response {
    if !role.allows_write() {
        return (
            StatusCode::FORBIDDEN,
            Json(ErrorBody {
                error: "this token is read-only; admin flush is forbidden".into(),
            }),
        )
            .into_response();
    }
    let ns_state = match shared.registry.get_or_open(namespace).await {
        Ok(ns) => ns,
        Err(e) => {
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(ErrorBody { error: e.to_string() }),
            )
                .into_response();
        }
    };
    let mut w = ns_state.writer.lock().await;
    let schema = w.snapshot().manifest().manifest.schema.clone();
    match w.flush(schema).await {
        Ok(outcome) => {
            ns_state.snapshot.store(w.owned_snapshot());
            Json(FlushResponse {
                ssts_written: outcome.ssts_written,
                bloom_sidecars_written: outcome.bloom_sidecars_written,
                manifest_version: outcome.committed.manifest.version,
            })
            .into_response()
        }
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorBody {
                error: format!("flush failed: {e}"),
            }),
        )
            .into_response(),
    }
}

/// Prometheus metrics handler in multi-tenant mode.
async fn metrics_handler_multi(State(shared): State<SharedAppState>) -> impl IntoResponse {
    (
        [(
            axum::http::header::CONTENT_TYPE,
            HeaderValue::from_static("text/plain; version=0.0.4; charset=utf-8"),
        )],
        shared.metrics.render(),
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
    async fn admin_flush_is_forbidden_for_a_read_only_token() {
        let app = fixture_with_tokens("authz-flush", ROLE_TOKENS).await;
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
            off.write_stall_for(1_000).is_none(),
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
        assert_eq!(on.write_stall_for(7), None, "below threshold");
        assert_eq!(
            on.write_stall_for(8),
            Some(Duration::from_millis(50)),
            "at threshold"
        );
        assert_eq!(
            on.write_stall_for(99),
            Some(Duration::from_millis(50)),
            "above threshold"
        );
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
        let shared = SharedAppState::new(
            registry,
            Arc::new(AuthConfig::open()),
            metrics,
            Duration::ZERO,
            Duration::ZERO,
            0,
            0,
            Duration::ZERO,
            default_ns.to_string(),
        );
        build_multi_tenant_router(shared)
    }

    async fn mt_cypher(app: &Router, uri: &str, header_ns: Option<&str>, query: &str) -> StatusCode {
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
        assert_eq!(mt_cypher(&app, "/acme/v0/cypher", None, q).await, StatusCode::OK);
        // 2. X-NamiDB-Namespace header on an unprefixed path.
        assert_eq!(mt_cypher(&app, "/v0/cypher", Some("beta"), q).await, StatusCode::OK);
        // 3. No prefix, no header → default namespace.
        assert_eq!(mt_cypher(&app, "/v0/cypher", None, q).await, StatusCode::OK);
        // The default namespace is genuinely distinct: a note written to
        // `acme` is NOT visible via the default namespace.
        let app = multi_tenant_app("default").await;
        let _ = mt_cypher(&app, "/acme/v0/cypher", None, "CREATE (:Person {name: 'only-acme'})").await;
        let read = mt_cypher(&app, "/v0/cypher", None, "MATCH (p:Person) RETURN count(p)").await;
        assert_eq!(read, StatusCode::OK, "default namespace is isolated from acme");
    }
}
