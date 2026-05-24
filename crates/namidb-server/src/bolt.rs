//! Bolt listener for `namidb-server`.
//!
//! Wires [`namidb_bolt::Session`] up to the writer session that the
//! HTTP router already owns, so both protocols share one
//! `WriterSession` per process (single-writer invariant from RFC-001).
//!
//! Most of the heavy lifting lives in `namidb-bolt`. This module
//! supplies the [`Backend`] adapter and the `accept()` loop.

use std::sync::Arc;

use async_trait::async_trait;
use namidb_bolt::{
    AuthPolicy, Backend, BackendError, RunOutcome, ServerInfo, Session, StatementType,
};
use namidb_query::{
    execute, execute_write, parse as cypher_parse, plan as build_plan, ExecError, LowerError,
    Params, ParseError, StatsCatalog,
};
use tokio::net::TcpListener;
use tracing::{error, info, warn};
use uuid::Uuid;

use crate::AppState;

/// Adapter that drives Bolt `RUN` requests against the shared
/// [`WriterSession`].
pub struct ServerBackend {
    state: AppState,
}

impl ServerBackend {
    pub fn new(state: AppState) -> Self {
        Self { state }
    }
}

#[async_trait]
impl Backend for ServerBackend {
    async fn run(
        &self,
        cypher: &str,
        params: Params,
    ) -> std::result::Result<RunOutcome, BackendError> {
        let parsed = match cypher_parse(cypher) {
            Ok(p) => p,
            Err(errs) => {
                let first = &errs[0];
                return Err(BackendError::Syntax(format!(
                    "{} at {}",
                    first.message, first.span
                )));
            }
        };
        let mut writer = self.state.writer.lock().await;
        let catalog = StatsCatalog::from_manifest(&writer.snapshot().manifest().manifest);
        let plan = build_plan(&parsed, &catalog).map_err(map_lower_err)?;

        if plan.contains_write() {
            let outcome = execute_write(&plan, &mut writer, &params)
                .await
                .map_err(map_exec_err)?;
            let stype = classify_write(&outcome);
            let fields = field_list(&outcome.rows);
            let mut counters = std::collections::BTreeMap::new();
            counters.insert("nodes-created".into(), outcome.nodes_created as i64);
            counters.insert("nodes-deleted".into(), outcome.nodes_deleted as i64);
            counters.insert("relationships-created".into(), outcome.edges_created as i64);
            counters.insert("relationships-deleted".into(), outcome.edges_deleted as i64);
            counters.insert("properties-set".into(), outcome.properties_set as i64);
            Ok(RunOutcome {
                fields,
                rows: outcome.rows,
                statement_type: stype,
                counters,
            })
        } else {
            let snap = writer.snapshot();
            let rows = execute(&plan, &snap, &params).await.map_err(map_exec_err)?;
            let fields = field_list(&rows);
            Ok(RunOutcome {
                fields,
                rows,
                statement_type: StatementType::Read,
                counters: Default::default(),
            })
        }
    }

    async fn current_bookmark(&self) -> Option<String> {
        let w = self.state.writer.lock().await;
        let snapshot = w.snapshot();
        let version = snapshot.manifest().manifest.version;
        Some(format!("namidb:v{}", version))
    }
}

fn classify_write(o: &namidb_query::WriteOutcome) -> StatementType {
    let any_read = !o.rows.is_empty();
    let any_write = o.nodes_created > 0
        || o.nodes_deleted > 0
        || o.edges_created > 0
        || o.edges_deleted > 0
        || o.properties_set > 0;
    match (any_read, any_write) {
        (true, true) => StatementType::ReadWrite,
        (false, true) => StatementType::Write,
        (true, false) => StatementType::Read,
        (false, false) => StatementType::Write,
    }
}

fn field_list(rows: &[namidb_query::Row]) -> Vec<String> {
    rows.first()
        .map(|r| r.bindings.keys().cloned().collect())
        .unwrap_or_default()
}

fn map_lower_err(e: LowerError) -> BackendError {
    use namidb_query::LowerErrorKind;
    match e.kind {
        LowerErrorKind::UnsupportedFeature => BackendError::Unsupported(e.message),
        _ => BackendError::Semantic(e.message),
    }
}

fn map_exec_err(e: ExecError) -> BackendError {
    // ExecError today is opaque from outside the crate; format and
    // bucket as either an eval or a storage error based on a
    // best-effort substring match.
    let text = format!("{e}");
    if text.contains("storage") || text.contains("manifest") {
        BackendError::Storage(text)
    } else {
        BackendError::Eval(text)
    }
}

/// Translate the server's auth token into a Bolt [`AuthPolicy`].
fn auth_policy(token: &Option<Arc<str>>) -> AuthPolicy {
    match token {
        Some(t) => AuthPolicy::Token(t.clone()),
        None => AuthPolicy::Open,
    }
}

/// Bind the Bolt listener and serve sessions until the process exits.
pub async fn serve(
    state: AppState,
    listen: std::net::SocketAddr,
    auth_token: Option<Arc<str>>,
) -> anyhow::Result<()> {
    let listener = TcpListener::bind(listen).await?;
    info!(addr = %listen, "namidb bolt listening");
    let agent = format!("NamiDB/{}", env!("CARGO_PKG_VERSION"));
    loop {
        let (socket, peer) = match listener.accept().await {
            Ok(p) => p,
            Err(e) => {
                error!(error = %e, "bolt accept failed");
                continue;
            }
        };
        if let Err(e) = socket.set_nodelay(true) {
            warn!(error = %e, %peer, "set_nodelay failed");
        }
        let state = state.clone();
        let policy = auth_policy(&auth_token);
        let agent = agent.clone();
        let connection_id = Uuid::now_v7().to_string();
        tokio::spawn(async move {
            let backend = Arc::new(ServerBackend::new(state));
            let session = Session::new(
                socket,
                ServerInfo {
                    agent,
                    connection_id,
                },
                policy,
                backend,
            );
            if let Err(e) = session.run().await {
                warn!(error = %e, %peer, "bolt session ended with error");
            }
        });
    }
}

// `ParseError` is included for callers that want a custom Bolt error
// shape; today we collapse to a single `Syntax(String)` above.
#[allow(dead_code)]
fn parse_err_to_string(e: &ParseError) -> String {
    format!("{} at {}", e.message, e.span)
}
