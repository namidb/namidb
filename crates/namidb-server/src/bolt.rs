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
    execute, execute_write, execute_write_staged, parse as cypher_parse, plan as build_plan,
    ExecError, LowerError, Params, ParseError, Row, WriteOutcome,
};
use namidb_storage::WriterSession;
use tokio::net::TcpListener;
use tokio::sync::{Mutex, OwnedMutexGuard};
use tracing::{error, info, warn};
use uuid::Uuid;

use crate::AppState;

/// In-flight explicit transaction (BEGIN..COMMIT/ROLLBACK). Holds the
/// global writer lock for the whole transaction so no other writer — nor
/// the flush / compaction tasks — can commit a half-built batch in the
/// middle of it. Staged statements live in the writer's pending batch and
/// are made durable in one commit at COMMIT, or dropped at ROLLBACK.
struct TxState {
    writer: OwnedMutexGuard<WriterSession>,
    /// Whether any statement staged a mutation, so ROLLBACK only discards
    /// when there is something to discard.
    staged: bool,
}

/// Adapter that drives Bolt `RUN` requests against the shared
/// [`WriterSession`]. One is created per connection.
pub struct ServerBackend {
    state: AppState,
    /// Per-connection explicit-transaction slot. `None` outside BEGIN..END.
    tx: Mutex<Option<TxState>>,
}

impl ServerBackend {
    pub fn new(state: AppState) -> Self {
        Self {
            state,
            tx: Mutex::new(None),
        }
    }
}

#[async_trait]
impl Backend for ServerBackend {
    async fn run(
        &self,
        cypher: &str,
        params: Params,
    ) -> std::result::Result<RunOutcome, BackendError> {
        // Memgraph-style schema introspection (gdotv and other Bolt
        // GUIs) hits procedures the Cypher parser has no `CALL` clause
        // for. Answer them from the live snapshot before the parser
        // would reject them as a syntax error. See `crate::introspect`.
        {
            let owned = self.state.snapshot.load();
            let snap = owned.borrow();
            if let Some(result) = crate::introspect::try_introspect(cypher, &snap).await {
                return result;
            }
        }

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
        // Plan against the latest published snapshot — no writer lock.
        let owned = self.state.snapshot.load();
        let catalog = self.state.catalog_for(&owned.manifest().manifest);
        let plan = build_plan(&parsed, &catalog).map_err(map_lower_err)?;

        if plan.contains_write() {
            // Writes still take the writer lock (single-writer invariant).
            // On success we refresh the snapshot cell so subsequent reads
            // see the just-committed records (RFC-021).
            let mut writer = self.state.writer.lock().await;
            let outcome = execute_write(&plan, &mut writer, &params)
                .await
                .map_err(map_exec_err)?;
            self.state.snapshot.store(writer.owned_snapshot());
            Ok(write_run_outcome(outcome))
        } else {
            // Read path: borrow a short-lived `Snapshot` from the owned
            // snapshot; the Arc keeps the underlying memtable alive for
            // the duration of the query, no writer lock needed.
            let snap = owned.borrow();
            let rows = execute(&plan, &snap, &params).await.map_err(map_exec_err)?;
            Ok(read_run_outcome(rows))
        }
    }

    async fn begin_tx(&self) -> std::result::Result<(), BackendError> {
        let mut slot = self.tx.lock().await;
        if slot.is_some() {
            return Err(BackendError::Other("a transaction is already open".into()));
        }
        // Take the global writer lock for the whole transaction. Held across
        // RUNs (and client think-time) until COMMIT/ROLLBACK — see TxState.
        let writer = self.state.writer.clone().lock_owned().await;
        *slot = Some(TxState {
            writer,
            staged: false,
        });
        Ok(())
    }

    async fn run_in_tx(
        &self,
        cypher: &str,
        params: Params,
    ) -> std::result::Result<RunOutcome, BackendError> {
        // Introspection and reads run against the published snapshot, same
        // as auto-commit — an in-tx read does NOT see the tx's own staged
        // writes (no read-your-own-writes; documented limitation).
        {
            let owned = self.state.snapshot.load();
            let snap = owned.borrow();
            if let Some(result) = crate::introspect::try_introspect(cypher, &snap).await {
                return result;
            }
        }
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
        let owned = self.state.snapshot.load();
        let catalog = self.state.catalog_for(&owned.manifest().manifest);
        let plan = build_plan(&parsed, &catalog).map_err(map_lower_err)?;

        if plan.contains_write() {
            // Stage into the transaction's held writer; do NOT commit. The
            // RETURN rows are computed during the apply, so they stream now.
            let mut slot = self.tx.lock().await;
            let tx = slot
                .as_mut()
                .ok_or_else(|| BackendError::Other("no open transaction".into()))?;
            let outcome = execute_write_staged(&plan, &mut tx.writer, &params)
                .await
                .map_err(map_exec_err)?;
            tx.staged = true;
            Ok(write_run_outcome(outcome))
        } else {
            let snap = owned.borrow();
            let rows = execute(&plan, &snap, &params).await.map_err(map_exec_err)?;
            Ok(read_run_outcome(rows))
        }
    }

    async fn commit_tx(&self) -> std::result::Result<(), BackendError> {
        let mut slot = self.tx.lock().await;
        let mut tx = slot
            .take()
            .ok_or_else(|| BackendError::Other("no open transaction".into()))?;
        // One manifest CAS makes the whole transaction durable; then
        // republish so reads see it. Dropping `tx` releases the writer lock.
        tx.writer.commit_batch().await.map_err(map_storage_err)?;
        self.state.snapshot.store(tx.writer.owned_snapshot());
        Ok(())
    }

    async fn rollback_tx(&self) -> std::result::Result<(), BackendError> {
        let mut slot = self.tx.lock().await;
        if let Some(mut tx) = slot.take() {
            if tx.staged {
                tx.writer.discard_batch();
            }
            // Dropping `tx` releases the writer lock.
        }
        Ok(())
    }

    async fn current_bookmark(&self) -> Option<String> {
        Some(format!(
            "namidb:v{}",
            self.state.snapshot.manifest_version()
        ))
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

/// Build the Bolt `RunOutcome` for a write statement (auto-commit or staged
/// in a transaction): the result rows plus the update counters.
fn write_run_outcome(outcome: WriteOutcome) -> RunOutcome {
    let stype = classify_write(&outcome);
    let fields = field_list(&outcome.rows);
    let mut counters = std::collections::BTreeMap::new();
    counters.insert("nodes-created".into(), outcome.nodes_created as i64);
    counters.insert("nodes-deleted".into(), outcome.nodes_deleted as i64);
    counters.insert("relationships-created".into(), outcome.edges_created as i64);
    counters.insert("relationships-deleted".into(), outcome.edges_deleted as i64);
    counters.insert("properties-set".into(), outcome.properties_set as i64);
    RunOutcome {
        fields,
        rows: outcome.rows,
        statement_type: stype,
        counters,
    }
}

/// Build the Bolt `RunOutcome` for a read statement.
fn read_run_outcome(rows: Vec<Row>) -> RunOutcome {
    let fields = field_list(&rows);
    RunOutcome {
        fields,
        rows,
        statement_type: StatementType::Read,
        counters: Default::default(),
    }
}

/// Map a storage commit failure to a Bolt error. A failed manifest CAS
/// poisons the `WriterSession` (its contract is "drop and reopen"); the
/// reopen orchestration is a documented follow-up, so for now the client
/// sees a retryable storage error.
fn map_storage_err(e: namidb_storage::Error) -> BackendError {
    BackendError::Storage(format!("{e}"))
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
    tx_timeout: std::time::Duration,
) -> anyhow::Result<()> {
    // `Duration::ZERO` disables the per-transaction idle timeout.
    let tx_idle_timeout = (!tx_timeout.is_zero()).then_some(tx_timeout);
    let listener = TcpListener::bind(listen).await?;
    info!(addr = %listen, "namidb bolt listening");
    // The HELLO `server` agent must look like a Neo4j build or the
    // official drivers (and GUIs built on them: gdotv, Neo4j Browser,
    // Bloom) reject the connection with "Server does not identify as a
    // genuine Neo4j instance". Memgraph and Amazon Neptune present a
    // `Neo4j/<version>` agent for exactly this reason; the Bolt endpoint
    // exists for driver compatibility, so we default to one too.
    // Override via `NAMIDB_BOLT_SERVER_AGENT` (e.g. to the honest
    // `NamiDB/<version>` when talking to a lenient client).
    let agent =
        std::env::var("NAMIDB_BOLT_SERVER_AGENT").unwrap_or_else(|_| "Neo4j/5.13.0".to_string());
    info!(server_agent = %agent, "bolt server agent");
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
            )
            .with_tx_idle_timeout(tx_idle_timeout);
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
