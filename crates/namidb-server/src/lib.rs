//! HTTP server exposing a NamiDB namespace.
//!
//! The crate is split between a thin [`main`] CLI parser and this
//! library so integration tests can exercise the routes directly.
//!
//! See [`build_router`] for the full route surface and [`run`] for
//! the end-to-end boot procedure.

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
    execute, execute_write, parse as cypher_parse, plan as build_plan, Params, RuntimeValue,
    StatsCatalog, WriteOutcome,
};
use namidb_storage::WriterSession;

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
}

/// Shared application state — one `WriterSession` (single-writer
/// invariant) plus the auth token reference.
#[derive(Clone)]
pub struct AppState {
    writer: Arc<Mutex<WriterSession>>,
    auth_token: Option<Arc<str>>,
    namespace: String,
}

impl AppState {
    pub fn new(writer: WriterSession, auth_token: Option<String>, namespace: String) -> Self {
        Self {
            writer: Arc::new(Mutex::new(writer)),
            auth_token: auth_token.map(Arc::from),
            namespace,
        }
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
    let writer = WriterSession::open(store, paths).await?;

    let state = AppState::new(writer, config.auth_token.clone(), namespace);

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
                if let Err(e) = w.flush(schema).await {
                    error!(error = %e, "periodic flush failed");
                }
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
            nodes_created: o.nodes_created as u64,
            edges_created: o.edges_created as u64,
            nodes_deleted: o.nodes_deleted as u64,
            edges_deleted: o.edges_deleted as u64,
            properties_set: o.properties_set as u64,
        }
    }
}

async fn cypher(
    State(state): State<AppState>,
    Json(req): Json<CypherRequest>,
) -> Response {
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

    let mut writer = state.writer.lock().await;
    let catalog = StatsCatalog::from_manifest(&writer.snapshot().manifest().manifest);
    let plan = match build_plan(&parsed, &catalog) {
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
    };

    if plan.contains_write() {
        match execute_write(&plan, &mut writer, &params).await {
            Ok(outcome) => {
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
        // Snapshot is `Snapshot<'_>` borrowed from the writer, so we
        // hold the lock for the duration of the read. Concurrent
        // readers are serialised by the writer mutex; this is the
        // single-writer-per-namespace invariant pulled up to the
        // request layer. Lifting it requires snapshots that own their
        // state (Arc-only), which is RFC-021 work.
        let snap = writer.snapshot();
        match execute(&plan, &snap, &params).await {
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
        Ok(outcome) => Json(FlushResponse {
            ssts_written: outcome.ssts_written,
            bloom_sidecars_written: outcome.bloom_sidecars_written,
            manifest_version: outcome.committed.manifest.version,
        })
        .into_response(),
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

fn params_from_json(
    m: &serde_json::Map<String, serde_json::Value>,
) -> Result<Params, String> {
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
        RuntimeValue::Float(f) => serde_json::Number::from_f64(*f).map(J::Number).unwrap_or(J::Null),
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
        RuntimeValue::DateTime(micros) => chrono::DateTime::<chrono::Utc>::from_timestamp_micros(
            *micros,
        )
        .map(|dt| J::String(dt.to_rfc3339()))
        .unwrap_or(J::Null),
        RuntimeValue::Node(n) => {
            let mut o = serde_json::Map::new();
            o.insert("_kind".into(), J::String("node".into()));
            o.insert("id".into(), J::String(n.id.to_string()));
            o.insert("label".into(), J::String(n.label.clone()));
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
