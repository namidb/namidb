//! HTTP server exposing a NamiDB namespace.
//!
//! The crate is split between a thin [`main`] CLI parser and this
//! library so integration tests can exercise the routes directly.
//!
//! See [`build_router`] for the full route surface and [`run`] for
//! the end-to-end boot procedure.

pub mod bolt;
mod introspect;

use std::sync::Arc;
use std::time::Duration;

use axum::extract::State;
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
    execute_with_limits, execute_write, parse as cypher_parse, plan as build_plan, Params,
    RuntimeValue, StatsCatalog, WriteOutcome,
};
use namidb_storage::{sweep_orphans, Manifest, ManifestStore, SnapshotCell, WriterSession};

/// Process-wide configuration assembled from CLI flags or env vars.
#[derive(Debug, Clone)]
pub struct Config {
    pub store_uri: String,
    pub listen: std::net::SocketAddr,
    /// `None` means "no auth"; the server will log a loud warning at
    /// boot and accept every request. Production callers should set a
    /// long random secret.
    pub auth_token: Option<String>,
    pub flush_interval: Duration,
    /// Interval for the background maintenance task (L0->L1 compaction +
    /// orphan sweep). `Duration::ZERO` disables it.
    pub compaction_interval: Duration,
    /// Minimum age before the orphan sweep may delete an unreferenced SST
    /// body — the sole guard against deleting a file a slow reader's pinned
    /// snapshot still references.
    pub sweep_min_age: Duration,
    /// When `false` the orphan sweep is a dry-run (logs what it would free
    /// without deleting). Operators opt in after reviewing the volume.
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
    /// timeout error rather than pinning a worker. Writes are bounded by the
    /// transaction lifecycle, not this. `Duration::ZERO` disables it.
    pub query_timeout: Duration,
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
    auth_token: Option<Arc<str>>,
    namespace: String,
    /// Per-read-query wall-clock budget. `Duration::ZERO` disables it.
    /// Defaults to disabled; the server sets it from [`Config`] at boot.
    query_timeout: Duration,
}

impl AppState {
    pub fn new(writer: WriterSession, auth_token: Option<String>, namespace: String) -> Self {
        let snapshot = Arc::new(SnapshotCell::new(writer.owned_snapshot()));
        Self {
            writer: Arc::new(Mutex::new(writer)),
            snapshot,
            catalog_cache: Arc::new(std::sync::Mutex::new(None)),
            auth_token: auth_token.map(Arc::from),
            namespace,
            query_timeout: Duration::ZERO,
        }
    }

    /// Set the per-read-query timeout (builder style). `Duration::ZERO`
    /// leaves reads unbounded.
    pub fn with_query_timeout(mut self, timeout: Duration) -> Self {
        self.query_timeout = timeout;
        self
    }

    /// Deadline for a read query starting now, or `None` when the timeout
    /// is disabled. Computed per query so each read gets the full budget.
    pub(crate) fn query_deadline(&self) -> Option<std::time::Instant> {
        (self.query_timeout > Duration::ZERO)
            .then(|| std::time::Instant::now() + self.query_timeout)
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
/// middleware. `/v0/health` and `/v0/version` are intentionally
/// excluded from the auth check.
pub fn build_router(state: AppState) -> Router {
    let public = Router::new()
        .route("/v0/health", get(health))
        .route("/v0/version", get(version));

    let private = Router::new()
        .route("/v0/cypher", post(cypher))
        .route("/v0/admin/flush", post(admin_flush))
        .layer(middleware::from_fn_with_state(state.clone(), require_auth));

    Router::new().merge(public).merge(private).with_state(state)
}

/// Boot the server: parse URI, open a `WriterSession`, optionally
/// spawn a periodic flush task, and serve until the process receives
/// SIGINT.
pub async fn run(config: Config) -> anyhow::Result<()> {
    if config.auth_token.is_none() {
        warn!(
            "⚠️  namidb-server is running WITHOUT auth. Anyone who can reach \
             {} can issue arbitrary Cypher queries. Set --auth-token (or env \
             NAMIDB_AUTH_TOKEN) before exposing this port beyond localhost.",
            config.listen
        );
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

    let state = AppState::new(writer, config.auth_token.clone(), namespace)
        .with_query_timeout(config.query_timeout);

    // Periodic flush task — keeps the WAL bounded and L0 SSTs current.
    if config.flush_interval > Duration::ZERO {
        let state_for_flush = state.clone();
        let interval = config.flush_interval;
        tokio::spawn(async move {
            let mut tick = tokio::time::interval(interval);
            tick.tick().await; // first tick fires immediately; skip.
            loop {
                tick.tick().await;
                let mut w = state_for_flush.writer.lock().await;
                let schema = w.snapshot().manifest().manifest.schema.clone();
                match w.flush(schema).await {
                    Ok(_) => state_for_flush.snapshot.store(w.owned_snapshot()),
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
    // reads the committed manifest itself) and is gated to a dry-run by
    // default; `sweep_min_age` is the only thing keeping it from deleting a
    // body a slow reader's pinned snapshot still references.
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
                // engine only produces L0 + L1 today.
                match sweep_orphans(&maint_manifest_store, sweep_min_age, 1, sweep_delete).await {
                    Ok(report) if report.orphans_found > 0 => info!(
                        found = report.orphans_found,
                        deleted = report.orphans_deleted,
                        bytes_freed = report.bytes_freed,
                        dry_run = !sweep_delete,
                        "orphan sweep"
                    ),
                    Ok(_) => {}
                    Err(e) => error!(error = %e, "orphan sweep failed"),
                }
            }
        });
    }

    // Optional Bolt listener (binds an extra TCP port for native
    // Neo4j drivers — see RFC-022). When not configured we stay
    // HTTP-only.
    if let Some(bolt_addr) = config.bolt_listen {
        let bolt_state = state.clone();
        let bolt_auth = state.auth_token.clone();
        let tx_timeout = config.bolt_tx_timeout;
        tokio::spawn(async move {
            if let Err(e) = bolt::serve(bolt_state, bolt_addr, bolt_auth, tx_timeout).await {
                error!(error = %e, "bolt listener exited");
            }
        });
    }

    let app = build_router(state);

    let listener = TcpListener::bind(config.listen).await?;
    info!(addr = %config.listen, "namidb-server listening");
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;
    Ok(())
}

async fn shutdown_signal() {
    let _ = tokio::signal::ctrl_c().await;
    info!("ctrl-c received, draining requests…");
}

// ── auth ──────────────────────────────────────────────────────────────

async fn require_auth(
    State(state): State<AppState>,
    req: axum::extract::Request,
    next: Next,
) -> Response {
    let Some(expected) = state.auth_token.as_deref() else {
        return next.run(req).await;
    };
    let presented = req
        .headers()
        .get(axum::http::header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.strip_prefix("Bearer "));
    match presented {
        Some(token) if constant_time_eq(token.as_bytes(), expected.as_bytes()) => {
            next.run(req).await
        }
        _ => (
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

/// Subtle::ConstantTimeEq would pull in another crate; this open-coded
/// variant is good enough for short shared secrets (we walk every byte
/// regardless of mismatch position).
fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut acc: u8 = 0;
    for (x, y) in a.iter().zip(b.iter()) {
        acc |= x ^ y;
    }
    acc == 0
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

async fn health(State(state): State<AppState>) -> impl IntoResponse {
    let w = state.writer.lock().await;
    let snap = w.snapshot();
    Json(HealthResponse {
        status: "ok",
        namespace: state.namespace.clone(),
        manifest_version: snap.manifest().manifest.version,
        epoch: snap.manifest().manifest.epoch.as_u64(),
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

async fn cypher(State(state): State<AppState>, Json(req): Json<CypherRequest>) -> Response {
    let parsed = match cypher_parse(&req.query) {
        Ok(p) => p,
        Err(errs) => {
            let first = &errs[0];
            return (
                StatusCode::BAD_REQUEST,
                Json(ErrorBody {
                    error: format!("parse error: {} at {}", first.message, first.span),
                }),
            )
                .into_response();
        }
    };

    let params = match params_from_json(&req.params) {
        Ok(p) => p,
        Err(e) => {
            return (StatusCode::BAD_REQUEST, Json(ErrorBody { error: e })).into_response();
        }
    };

    // Plan against the latest published snapshot — no writer lock yet.
    let owned = state.snapshot.load();
    let plan = {
        let catalog = state.catalog_for(&owned.manifest().manifest);
        match build_plan(&parsed, &catalog) {
            Ok(p) => p,
            Err(e) => {
                return (
                    StatusCode::BAD_REQUEST,
                    Json(ErrorBody {
                        error: format!("plan error: {e}"),
                    }),
                )
                    .into_response();
            }
        }
    };

    if plan.contains_write() {
        let mut writer = state.writer.lock().await;
        match execute_write(&plan, &mut writer, &params).await {
            Ok(outcome) => {
                // Refresh the published snapshot so subsequent reads
                // see the just-committed records (RFC-021).
                state.snapshot.store(writer.owned_snapshot());
                let summary = WriteSummary::from(&outcome);
                let (columns, rows) = rows_to_json(&outcome.rows);
                Json(CypherResponse {
                    columns,
                    rows,
                    write_outcome: Some(summary),
                })
                .into_response()
            }
            Err(e) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorBody {
                    error: format!("write execution failed: {e}"),
                }),
            )
                .into_response(),
        }
    } else {
        // Read path: no writer lock. Borrow a short-lived `Snapshot`
        // from the owned one; the `OwnedSnapshot` Arc keeps the
        // underlying memtable alive for the duration of the query.
        let snap = owned.borrow();
        match execute_with_limits(&plan, &snap, &params, state.query_deadline()).await {
            Ok(rows) => {
                let (columns, rows) = rows_to_json(&rows);
                Json(CypherResponse {
                    columns,
                    rows,
                    write_outcome: None,
                })
                .into_response()
            }
            Err(e) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorBody {
                    error: format!("read execution failed: {e}"),
                }),
            )
                .into_response(),
        }
    }
}

#[derive(Serialize)]
struct FlushResponse {
    ssts_written: usize,
    bloom_sidecars_written: usize,
    manifest_version: u64,
}

async fn admin_flush(State(state): State<AppState>) -> Response {
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
}
