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
pub mod metrics;
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

use crate::auth::{AuthConfig, Principal};
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
    /// Pre-execution authorization hook (RFC-015 Wave B). Defaults to
    /// [`authz::NoOpAuthz`] (allow-all), so the gate is behavior-preserving
    /// until a real policy is configured.
    authz: Arc<dyn authz::AuthzHook>,
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
            authz: Arc::new(authz::NoOpAuthz),
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

    harden_router(Router::new().merge(public).merge(private).with_state(state))
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

/// Apply the shared HTTP hardening layers (request timeout + global concurrency
/// limit) to a fully-built router.
fn harden_router(router: Router) -> Router {
    router
        .layer(tower_http::timeout::TimeoutLayer::with_status_code(
            axum::http::StatusCode::REQUEST_TIMEOUT,
            http_request_timeout(),
        ))
        .layer(tower::limit::GlobalConcurrencyLimitLayer::new(
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
    let namespace_routes = Router::new()
        .route("/:namespace/v0/livez", get(livez_multi))
        .route("/:namespace/v0/health", get(health_multi))
        .route("/:namespace/v0/cypher", post(cypher_multi))
        .route("/:namespace/v0/admin/flush", post(admin_flush_multi))
        .route("/v0/livez", get(livez_multi))
        .route("/v0/health", get(health_multi_unprefixed))
        .route("/v0/cypher", post(cypher_multi_unprefixed))
        .route("/v0/admin/flush", post(admin_flush_multi_unprefixed))
        .layer(middleware::from_fn_with_state(
            shared.clone(),
            require_auth_multi,
        ));

    harden_router(
        Router::new()
            .merge(public)
            .merge(namespace_routes)
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
    // Resolve the auth configuration: a tokens file (with roles) wins, else a
    // single read-write `--auth-token`, else open.
    let auth = match (&config.auth_tokens_file, &config.auth_token) {
        (Some(path), _) => AuthConfig::load_file(path)?,
        // Refuse an empty `--auth-token`: it logs as "auth enabled" but a
        // `Bearer ` request would match the empty secret. Omit it to run open.
        (None, Some(secret)) if secret.is_empty() => {
            anyhow::bail!(
                "--auth-token is empty; omit it (and NAMIDB_AUTH_TOKEN) to run without auth"
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
        )
        .with_authz(authz.clone());
        let app = build_multi_tenant_router(shared);

        // Shutdown signal.
        let (shutdown_tx, shutdown_rx) = tokio::sync::watch::channel(false);
        tokio::spawn(async move {
            wait_for_shutdown_signal().await;
            let _ = shutdown_tx.send(true);
        });

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
                // Orphan sweep — no writer lock. The `max_level` arg is only a
                // floor now: sweep_orphans scans up to the deepest level any
                // retained manifest occupies, so L2+ compaction outputs are
                // reclaimed too. The retention horizon (RFC-027) is the oldest
                // manifest version any live reader is pinned to; the sweep keeps
                // every object referenced from the horizon to current, so it can
                // never delete a body a reader still needs.
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
        let bolt_tls = tls_config.clone().map(tls::acceptor);
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
        other if other.is_unsupported() => (StatusCode::BAD_REQUEST, Some("unsupported")),
        _ => (StatusCode::INTERNAL_SERVER_ERROR, None),
    }
}

/// Build an HTTP error response from an executor failure, classifying it so a
/// deliberately-unsupported feature surfaces as 400/`unsupported` instead of
/// a bare 500. The `code` field is emitted only when classified, so existing
/// clients that deserialize the body loosely see no change on plain 500s.
fn exec_failure_response(prefix: &str, e: &namidb_query::exec::ExecError) -> Response {
    let (status, code) = exec_error_classification(e);
    let error = format!("{prefix}: {e}");
    let body = match code {
        Some(c) => Json(serde_json::json!({ "error": error, "code": c })),
        None => Json(serde_json::json!({ "error": error })),
    };
    (status, body).into_response()
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
async fn run_create_vector_index(
    writer: &Arc<tokio::sync::Mutex<WriterSession>>,
    snapshot: &Arc<SnapshotCell>,
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
    let mut w = writer.lock().await;
    let result = apply_create_vector_index(&mut w, snapshot, cvi).await;
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
async fn run_create_fulltext_index(
    writer: &Arc<tokio::sync::Mutex<WriterSession>>,
    snapshot: &Arc<SnapshotCell>,
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
    let mut w = writer.lock().await;
    let result = apply_create_fulltext_index(&mut w, snapshot, cfi).await;
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
    authz: &Arc<dyn authz::AuthzHook>,
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
    let mut w = writer.lock().await;
    let result = if unique {
        apply_create_constraint(&mut w, snapshot, name, label, properties, if_not_exists).await
    } else {
        apply_create_index(&mut w, snapshot, name, label, &properties[0], if_not_exists).await
    };
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

    // `CREATE VECTOR INDEX` is schema DDL: intercept before planning.
    #[cfg(feature = "vector-index")]
    if let Some(cvi) = parsed.as_create_vector_index() {
        return run_create_vector_index(
            &state.writer,
            &state.snapshot,
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
            &state.authz,
            cfi,
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
            &state.authz,
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
            &state.authz,
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
                response: exec_failure_response("write execution failed", &e),
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

    // `CREATE VECTOR INDEX` is schema DDL: intercept before planning.
    #[cfg(feature = "vector-index")]
    if let Some(cvi) = parsed.as_create_vector_index() {
        return run_create_vector_index(
            &ns_state.writer,
            &ns_state.snapshot,
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
            &shared.authz,
            cfi,
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
            &shared.authz,
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
            &shared.authz,
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
    let plan = {
        let catalog = ns_state.catalog_for(&owned.manifest().manifest);
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
        let mut writer = ns_state.writer.lock().await;
        let result =
            execute_write_with_deadline(&plan, &mut writer, &params, shared.write_deadline()).await;
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
                response: exec_failure_response("write execution failed", &e),
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
        let shared = SharedAppState::new(
            registry,
            auth,
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
