//! Bolt listener for `namidb-server`.
//!
//! Wires [`namidb_bolt::Session`] up to the writer session that the
//! HTTP router already owns, so both protocols share one
//! `WriterSession` per process (single-writer invariant from RFC-001).
//!
//! Most of the heavy lifting lives in `namidb-bolt`. This module
//! supplies the [`Backend`] adapter and the `accept()` loop.

use std::collections::BTreeMap;
use std::sync::Arc;

use async_trait::async_trait;
use namidb_bolt::{
    AuthPolicy, Authenticator, Backend, BackendError, RunOutcome, ServerInfo, Session,
    StatementType, Value,
};
use namidb_query::{
    execute_with_limits, execute_write_staged_with_deadline, execute_write_with_deadline,
    parse as cypher_parse, plan as build_plan, ExecError, LowerError, Params, ParseError, Row,
    WriteOutcome,
};
use namidb_storage::WriterSession;
use tokio::net::TcpListener;
use tokio::sync::{Mutex, OwnedMutexGuard};
use tracing::{error, info, warn};
use uuid::Uuid;

use crate::auth::{AuthConfig, Principal};
use crate::metrics::{Protocol, QueryKind};
use crate::AppState;

/// One executed Bolt query, classified for metrics: read vs write (`None` if
/// it failed before planning), the wall-clock it took (up to the end of
/// execution, excluding any write-stall sleep), and the outcome the protocol
/// returns to the driver.
struct RunObservation {
    kind: Option<QueryKind>,
    elapsed: std::time::Duration,
    result: std::result::Result<RunOutcome, BackendError>,
}

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
    /// The authenticated principal for this connection, set by the paired
    /// [`TokenAuthenticator`] at LOGON. `None` until authenticated (open mode
    /// leaves it `None`, which `principal()` resolves to an anonymous
    /// read-write caller). A `std::sync::Mutex` so the write gate reads it
    /// without an `.await`; per-connection, so never contended.
    principal: Arc<std::sync::Mutex<Option<Principal>>>,
}

impl ServerBackend {
    pub fn new(state: AppState, principal: Arc<std::sync::Mutex<Option<Principal>>>) -> Self {
        Self {
            state,
            tx: Mutex::new(None),
            principal,
        }
    }

    /// The connection's authenticated principal, or an anonymous read-write
    /// principal when unauthenticated (open mode).
    fn principal(&self) -> Principal {
        self.principal
            .lock()
            .expect("bolt principal lock poisoned")
            .clone()
            .unwrap_or_else(Principal::anonymous_rw)
    }

    /// Consult the pre-execution authorization hook for `plan`. Returns
    /// `Some(Forbidden)` to deny (mapped to a Bolt `Forbidden` error), `None`
    /// to allow. Mirrors the HTTP `authz.check` call so the Bolt path is NOT a
    /// policy bypass (NoOp default ⇒ always allows).
    async fn authz_denied(&self, plan: &namidb_query::LogicalPlan) -> Option<BackendError> {
        match self.state.authz.check(&self.principal(), plan).await {
            Ok(()) => None,
            Err(denied) => Some(BackendError::Forbidden(denied.to_string())),
        }
    }

    /// `Some(error)` when the connection's principal may not write, to reject a
    /// write before it touches the writer lock.
    fn write_forbidden(&self) -> Option<BackendError> {
        (!self.principal().allows_write()).then(|| {
            BackendError::Forbidden("this token is read-only; write queries are forbidden".into())
        })
    }

    /// Bolt shape for `CREATE VECTOR INDEX`: gate on role, run the DDL via
    /// the shared storage helper, return an empty `RunOutcome` tagged
    /// `Schema`. Auto-commit only — the in-transaction path rejects DDL
    /// (a schema command commits immediately and cannot be rolled back).
    #[cfg(feature = "vector-index")]
    async fn run_create_vector_index(
        &self,
        cvi: &namidb_query::parser::ast::CreateVectorIndexClause,
        started: std::time::Instant,
    ) -> RunObservation {
        if let Some(err) = self.write_forbidden() {
            return RunObservation {
                kind: Some(QueryKind::Write),
                elapsed: started.elapsed(),
                result: Err(err),
            };
        }
        // Authorization hook for the schema op (DDL is intercepted pre-plan).
        let op = crate::authz::SchemaOp::CreateVectorIndex {
            name: &cvi.name.name,
            label: &cvi.label.name,
            property: &cvi.property.name,
        };
        if let Err(denied) = self.state.authz.check_schema(&self.principal(), op).await {
            return RunObservation {
                kind: None,
                elapsed: started.elapsed(),
                result: Err(BackendError::Forbidden(denied.to_string())),
            };
        }
        let mut writer = self.state.writer.lock().await;
        let result = crate::apply_create_vector_index(&mut writer, &self.state.snapshot, cvi).await;
        drop(writer);
        let elapsed = started.elapsed();
        match result {
            Ok(_) => RunObservation {
                kind: Some(QueryKind::Write),
                elapsed,
                result: Ok(RunOutcome {
                    fields: vec![],
                    rows: vec![],
                    statement_type: StatementType::Schema,
                    counters: BTreeMap::new(),
                }),
            },
            Err(e) => {
                // A duplicate name/target is a user (semantic) error; a fence
                // or lost CAS is a transient storage error.
                let is_user = matches!(
                    &e,
                    namidb_storage::Error::Precondition(_) | namidb_storage::Error::Invariant(_)
                );
                let err = if is_user {
                    BackendError::Semantic(e.to_string())
                } else {
                    map_storage_err(e)
                };
                RunObservation {
                    kind: Some(QueryKind::Write),
                    elapsed,
                    result: Err(err),
                }
            }
        }
    }

    /// Bolt shape for `CREATE FULLTEXT INDEX` (mirrors `run_create_vector_index`).
    #[cfg(feature = "text-index")]
    async fn run_create_fulltext_index(
        &self,
        cfi: &namidb_query::parser::ast::CreateFulltextIndexClause,
        started: std::time::Instant,
    ) -> RunObservation {
        if let Some(err) = self.write_forbidden() {
            return RunObservation {
                kind: Some(QueryKind::Write),
                elapsed: started.elapsed(),
                result: Err(err),
            };
        }
        let props: Vec<String> = cfi.properties.iter().map(|p| p.name.clone()).collect();
        let op = crate::authz::SchemaOp::CreateFulltextIndex {
            name: &cfi.name.name,
            label: &cfi.label.name,
            properties: &props,
        };
        if let Err(denied) = self.state.authz.check_schema(&self.principal(), op).await {
            return RunObservation {
                kind: None,
                elapsed: started.elapsed(),
                result: Err(BackendError::Forbidden(denied.to_string())),
            };
        }
        let mut writer = self.state.writer.lock().await;
        let result =
            crate::apply_create_fulltext_index(&mut writer, &self.state.snapshot, cfi).await;
        drop(writer);
        let elapsed = started.elapsed();
        match result {
            Ok(_) => RunObservation {
                kind: Some(QueryKind::Write),
                elapsed,
                result: Ok(RunOutcome {
                    fields: vec![],
                    rows: vec![],
                    statement_type: StatementType::Schema,
                    counters: BTreeMap::new(),
                }),
            },
            Err(e) => {
                let is_user = matches!(
                    &e,
                    namidb_storage::Error::Precondition(_) | namidb_storage::Error::Invariant(_)
                );
                let err = if is_user {
                    BackendError::Semantic(e.to_string())
                } else {
                    map_storage_err(e)
                };
                RunObservation {
                    kind: Some(QueryKind::Write),
                    elapsed,
                    result: Err(err),
                }
            }
        }
    }

    /// Auto-commit query: parse, plan, and execute against the published
    /// snapshot (reads) or the writer lock (writes), timing the work for the
    /// metrics. Mirrors the HTTP `run_cypher`. The stopwatch stops at the end
    /// of execution, before the optional write-stall sleep, so backpressure is
    /// not counted as query latency.
    async fn run_query(&self, cypher: &str, params: Params) -> RunObservation {
        let started = std::time::Instant::now();

        let parsed = match cypher_parse(cypher) {
            Ok(p) => p,
            Err(errs) => {
                let first = &errs[0];
                return RunObservation {
                    kind: None,
                    elapsed: started.elapsed(),
                    result: Err(BackendError::Syntax(format!(
                        "{} at {}",
                        first.message, first.span
                    ))),
                };
            }
        };
        // `CREATE VECTOR INDEX` is schema DDL: intercept before planning.
        #[cfg(feature = "vector-index")]
        if let Some(cvi) = parsed.as_create_vector_index() {
            return self.run_create_vector_index(cvi, started).await;
        }

        // `CREATE FULLTEXT INDEX` is schema DDL: intercept before planning.
        #[cfg(feature = "text-index")]
        if let Some(cfi) = parsed.as_create_fulltext_index() {
            return self.run_create_fulltext_index(cfi, started).await;
        }

        // Plan against the latest published snapshot — no writer lock.
        let owned = self.state.snapshot.load();
        let catalog = self.state.catalog_for(&owned.manifest().manifest);
        let plan = match build_plan(&parsed, &catalog).map_err(map_lower_err) {
            Ok(p) => p,
            Err(e) => {
                return RunObservation {
                    kind: None,
                    elapsed: started.elapsed(),
                    result: Err(e),
                };
            }
        };

        // Pre-execution authorization hook (RFC-015 Wave B): a policy may deny
        // before execution. NoOp by default. Mirrors the HTTP path so Bolt is
        // not a policy bypass.
        if let Some(err) = self.authz_denied(&plan).await {
            return RunObservation {
                kind: None,
                elapsed: started.elapsed(),
                result: Err(err),
            };
        }

        if plan.contains_write() {
            // A read-only token may not write — reject before the writer lock.
            if let Some(err) = self.write_forbidden() {
                return RunObservation {
                    kind: Some(QueryKind::Write),
                    elapsed: started.elapsed(),
                    result: Err(err),
                };
            }
            // Writes still take the writer lock (single-writer invariant).
            // On success we refresh the snapshot cell so subsequent reads
            // see the just-committed records (RFC-021).
            let mut writer = self.state.writer.lock().await;
            match execute_write_with_deadline(
                &plan,
                &mut writer,
                &params,
                self.state.write_deadline(),
            )
            .await
            {
                Ok(outcome) => {
                    self.state.snapshot.store(writer.owned_snapshot());
                    // Soft write stall (RFC-027 P5): sample under the lock,
                    // release, then back off this request if L0 is piling up.
                    let stall = self.state.write_stall_for(writer.max_l0_bucket_len());
                    drop(writer);
                    let elapsed = started.elapsed();
                    if let Some(delay) = stall {
                        tokio::time::sleep(delay).await;
                    }
                    RunObservation {
                        kind: Some(QueryKind::Write),
                        elapsed,
                        result: Ok(write_run_outcome(outcome)),
                    }
                }
                Err(e) => {
                    drop(writer);
                    RunObservation {
                        kind: Some(QueryKind::Write),
                        elapsed: started.elapsed(),
                        result: Err(map_exec_err(e)),
                    }
                }
            }
        } else {
            // Read path: borrow a short-lived `Snapshot` from the owned
            // snapshot; the Arc keeps the underlying memtable alive for
            // the duration of the query, no writer lock needed.
            let snap = owned.borrow();
            let rows = execute_with_limits(
                &plan,
                &snap,
                &params,
                self.state.query_deadline(),
                self.state.query_row_cap(),
            )
            .await;
            let elapsed = started.elapsed();
            RunObservation {
                kind: Some(QueryKind::Read),
                elapsed,
                result: rows.map(read_run_outcome).map_err(map_exec_err),
            }
        }
    }

    /// In-transaction query: stage writes into the held transaction's writer
    /// (no commit) or read with the staged batch overlaid (RFC-026), timing the
    /// work for the metrics. Mirrors the auto-commit `run_query`.
    async fn run_query_in_tx(&self, cypher: &str, params: Params) -> RunObservation {
        let started = std::time::Instant::now();

        let parsed = match cypher_parse(cypher) {
            Ok(p) => p,
            Err(errs) => {
                let first = &errs[0];
                return RunObservation {
                    kind: None,
                    elapsed: started.elapsed(),
                    result: Err(BackendError::Syntax(format!(
                        "{} at {}",
                        first.message, first.span
                    ))),
                };
            }
        };
        // DDL commits immediately and cannot be rolled back, so it is
        // rejected inside an explicit transaction (auto-commit only).
        #[cfg(feature = "vector-index")]
        if parsed.as_create_vector_index().is_some() {
            return RunObservation {
                kind: None,
                elapsed: started.elapsed(),
                result: Err(BackendError::Unsupported(
                    "CREATE VECTOR INDEX cannot run inside a transaction".into(),
                )),
            };
        }
        #[cfg(feature = "text-index")]
        if parsed.as_create_fulltext_index().is_some() {
            return RunObservation {
                kind: None,
                elapsed: started.elapsed(),
                result: Err(BackendError::Unsupported(
                    "CREATE FULLTEXT INDEX cannot run inside a transaction".into(),
                )),
            };
        }
        let owned = self.state.snapshot.load();
        let catalog = self.state.catalog_for(&owned.manifest().manifest);
        let plan = match build_plan(&parsed, &catalog).map_err(map_lower_err) {
            Ok(p) => p,
            Err(e) => {
                return RunObservation {
                    kind: None,
                    elapsed: started.elapsed(),
                    result: Err(e),
                };
            }
        };

        // Pre-execution authorization hook (RFC-015 Wave B); NoOp by default.
        if let Some(err) = self.authz_denied(&plan).await {
            return RunObservation {
                kind: None,
                elapsed: started.elapsed(),
                result: Err(err),
            };
        }

        if plan.contains_write() {
            // A read-only token may not write, even inside an open transaction.
            if let Some(err) = self.write_forbidden() {
                return RunObservation {
                    kind: Some(QueryKind::Write),
                    elapsed: started.elapsed(),
                    result: Err(err),
                };
            }
            // Stage into the transaction's held writer; do NOT commit. The
            // RETURN rows are computed during the apply, so they stream now.
            let mut slot = self.tx.lock().await;
            let tx = match slot.as_mut() {
                Some(tx) => tx,
                None => {
                    // A protocol-state error, not an executed query: keep it out
                    // of the latency histogram (kind None) like a parse/plan
                    // error. It still counts toward queries_total status=error.
                    return RunObservation {
                        kind: None,
                        elapsed: started.elapsed(),
                        result: Err(BackendError::Other("no open transaction".into())),
                    };
                }
            };
            let result = match execute_write_staged_with_deadline(
                &plan,
                &mut tx.writer,
                &params,
                self.state.write_deadline(),
            )
            .await
            {
                Ok(outcome) => {
                    tx.staged = true;
                    Ok(write_run_outcome(outcome))
                }
                Err(e) => {
                    // A failed statement aborts the transaction. Drop whatever
                    // it (or an earlier statement) staged so a stray COMMIT
                    // cannot seal a partial write; the session moves to FAILED
                    // and the client must ROLLBACK / RESET.
                    tx.writer.discard_batch();
                    Err(map_exec_err(e))
                }
            };
            RunObservation {
                kind: Some(QueryKind::Write),
                elapsed: started.elapsed(),
                result,
            }
        } else {
            // Read against the transaction's own writer so the staged batch
            // is visible (RFC-026). The writer pins the committed state at
            // tx-begin (no commit happens mid-tx while we hold the lock) and
            // overlays everything statements 1..N-1 staged.
            let mut slot = self.tx.lock().await;
            let tx = match slot.as_mut() {
                Some(tx) => tx,
                None => {
                    // See the write branch: a no-open-transaction error is a
                    // protocol-state error, not an executed query.
                    return RunObservation {
                        kind: None,
                        elapsed: started.elapsed(),
                        result: Err(BackendError::Other("no open transaction".into())),
                    };
                }
            };
            let snap = tx.writer.overlay_snapshot();
            let rows = execute_with_limits(
                &plan,
                &snap,
                &params,
                self.state.query_deadline(),
                self.state.query_row_cap(),
            )
            .await;
            let elapsed = started.elapsed();
            RunObservation {
                kind: Some(QueryKind::Read),
                elapsed,
                result: rows.map(read_run_outcome).map_err(map_exec_err),
            }
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
        // These are schema metadata probes, not user queries, so they are
        // intentionally not counted toward the query metrics.
        {
            let owned = self.state.snapshot.load();
            let snap = owned.borrow();
            if let Some(result) = crate::introspect::try_introspect(cypher, &snap).await {
                return result;
            }
        }

        let _in_flight = self.state.metrics.track_in_flight();
        let obs = self.run_query(cypher, params).await;
        self.state.metrics.observe_query(
            Protocol::Bolt,
            obs.kind,
            obs.result.is_ok(),
            obs.elapsed,
            cypher,
        );
        obs.result
    }

    async fn logoff(&self) {
        // Clear the per-connection identity so a subsequent request on this
        // connection cannot reuse the logged-off principal. Without auth
        // re-established it falls back to anonymous (open mode) or is rejected
        // at the next write/authz gate. (The Authenticator trait documents
        // that an embedder binding identity out-of-band should reset it here.)
        *self.principal.lock().expect("bolt principal lock poisoned") = None;
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
        // Introspection runs against the published snapshot (schema only).
        // Data reads, below, run against the transaction's own writer with
        // its staged batch overlaid, so an in-tx read sees the tx's own
        // staged writes (read-your-own-writes, RFC-026). Introspection is a
        // schema probe, not a user query, so it is not counted in the metrics.
        {
            let owned = self.state.snapshot.load();
            let snap = owned.borrow();
            if let Some(result) = crate::introspect::try_introspect(cypher, &snap).await {
                return result;
            }
        }

        let _in_flight = self.state.metrics.track_in_flight();
        let obs = self.run_query_in_tx(cypher, params).await;
        self.state.metrics.observe_query(
            Protocol::Bolt,
            obs.kind,
            obs.result.is_ok(),
            obs.elapsed,
            cypher,
        );
        obs.result
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
            // Always discard: a statement that failed before `staged` was set
            // can still have left mutations in the pending batch. Discarding
            // an empty batch is a no-op. Dropping `tx` releases the writer.
            tx.writer.discard_batch();
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
    // A deliberately-unsupported feature surfaces as the typed
    // `BackendError::Unsupported` (Neo.ClientError.Statement.NotSupported),
    // not a generic eval/storage bucket — so a driver can tell "not
    // implemented" from a genuine internal bug. This is the exec-side twin
    // of `map_lower_err`'s UnsupportedFeature arm.
    if e.is_unsupported() {
        return BackendError::Unsupported(e.to_string());
    }
    match e {
        // A constraint violation has its own Neo4j error class so drivers
        // can distinguish it from an ordinary evaluation error.
        ExecError::Constraint(m) => BackendError::Constraint(m),
        // The rest are opaque from outside the crate; format and bucket as
        // either an eval or a storage error on a best-effort substring match.
        other => {
            let text = format!("{other}");
            if text.contains("storage") || text.contains("manifest") {
                BackendError::Storage(text)
            } else {
                BackendError::Eval(text)
            }
        }
    }
}

/// Bolt `Custom` authenticator backed by the server's token set. On a
/// successful LOGON it records the resolved [`Principal`] into the
/// per-connection cell the paired [`ServerBackend`] reads to gate writes — the
/// "out of band" per-connection context the [`Authenticator`] contract
/// describes.
struct TokenAuthenticator {
    auth: Arc<AuthConfig>,
    principal: Arc<std::sync::Mutex<Option<Principal>>>,
}

#[async_trait]
impl Authenticator for TokenAuthenticator {
    async fn authenticate(
        &self,
        extra: &BTreeMap<String, Value>,
    ) -> std::result::Result<(), String> {
        let str_field = |key: &str| {
            extra.get(key).and_then(|v| match v {
                Value::String(s) => Some(s.as_str()),
                _ => None,
            })
        };
        let scheme = str_field("scheme").unwrap_or("none");
        if scheme != "basic" && scheme != "bearer" {
            return Err(format!("unsupported auth scheme `{scheme}`"));
        }
        match str_field("credentials").and_then(|c| self.auth.principal_for(c)) {
            Some(p) => {
                *self.principal.lock().expect("bolt principal lock poisoned") = Some(p);
                Ok(())
            }
            None => Err("invalid credentials".into()),
        }
    }
}

/// Build the per-connection [`AuthPolicy`]: `Open` when no tokens are
/// configured, otherwise a [`TokenAuthenticator`] that records the resolved
/// principal for the backend's write gate.
fn make_policy(
    auth: &Arc<AuthConfig>,
    principal: Arc<std::sync::Mutex<Option<Principal>>>,
) -> AuthPolicy {
    if auth.is_open() {
        AuthPolicy::Open
    } else {
        AuthPolicy::Custom(Arc::new(TokenAuthenticator {
            auth: auth.clone(),
            principal,
        }))
    }
}

/// Bind the Bolt listener and serve sessions until the process exits.
pub async fn serve(
    state: AppState,
    listen: std::net::SocketAddr,
    auth: Arc<AuthConfig>,
    tx_timeout: std::time::Duration,
    mut shutdown: tokio::sync::watch::Receiver<bool>,
    tls: Option<tokio_rustls::TlsAcceptor>,
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
        let (socket, peer) = tokio::select! {
            accepted = listener.accept() => match accepted {
                Ok(p) => p,
                Err(e) => {
                    error!(error = %e, "bolt accept failed");
                    continue;
                }
            },
            // Stop accepting new connections on shutdown (SIGTERM/SIGINT). The
            // HTTP server drains in parallel; in-flight Bolt sessions finish on
            // their own tasks.
            _ = shutdown.wait_for(|stop| *stop) => {
                info!("shutdown signalled, bolt listener stopping");
                break;
            }
        };
        if let Err(e) = socket.set_nodelay(true) {
            warn!(error = %e, %peer, "set_nodelay failed");
        }
        let state = state.clone();
        // One principal cell per connection, shared between the authenticator
        // (which sets it at LOGON) and the backend (which reads it on every
        // write). `None` until authenticated; open mode leaves it `None`.
        let principal = Arc::new(std::sync::Mutex::new(None));
        let policy = make_policy(&auth, principal.clone());
        let info = ServerInfo {
            agent: agent.clone(),
            connection_id: Uuid::now_v7().to_string(),
        };
        let tls = tls.clone();
        tokio::spawn(async move {
            let backend: Arc<dyn Backend> = Arc::new(ServerBackend::new(state, principal));
            // `Session` is generic over the transport, so the only fork is the
            // optional TLS handshake on the accepted socket.
            match tls {
                Some(acceptor) => match acceptor.accept(socket).await {
                    Ok(stream) => {
                        run_session(stream, info, policy, backend, tx_idle_timeout, peer).await
                    }
                    Err(e) => warn!(error = %e, %peer, "bolt TLS handshake failed"),
                },
                None => run_session(socket, info, policy, backend, tx_idle_timeout, peer).await,
            }
        });
    }
    Ok(())
}

/// Build and run one Bolt session over any byte stream — a plain `TcpStream`
/// or a TLS stream. `Session` is generic over the transport, so TLS adds only
/// a handshake in front of the same session loop.
async fn run_session<S>(
    socket: S,
    info: ServerInfo,
    policy: AuthPolicy,
    backend: Arc<dyn Backend>,
    tx_idle_timeout: Option<std::time::Duration>,
    peer: std::net::SocketAddr,
) where
    S: tokio::io::AsyncReadExt + tokio::io::AsyncWriteExt + Unpin,
{
    let session = Session::new(socket, info, policy, backend).with_tx_idle_timeout(tx_idle_timeout);
    if let Err(e) = session.run().await {
        warn!(error = %e, %peer, "bolt session ended with error");
    }
}

// `ParseError` is included for callers that want a custom Bolt error
// shape; today we collapse to a single `Syntax(String)` above.
#[allow(dead_code)]
fn parse_err_to_string(e: &ParseError) -> String {
    format!("{} at {}", e.message, e.span)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::auth::Role;
    use std::sync::Arc;

    // A hook that denies everything — to prove the Bolt path consults it (the
    // gap the adversarial review found: Bolt used to skip the AuthzHook).
    struct DenyAll;
    #[async_trait]
    impl crate::authz::AuthzHook for DenyAll {
        async fn check(
            &self,
            _p: &Principal,
            _plan: &namidb_query::LogicalPlan,
        ) -> Result<(), crate::authz::Denied> {
            Err(crate::authz::Denied::new("denied by test policy"))
        }
    }

    async fn backend_with_authz(authz: Arc<dyn crate::authz::AuthzHook>) -> ServerBackend {
        let (store, paths) = namidb_storage::parse_uri("memory://bolt-authz-test").unwrap();
        let writer = namidb_storage::WriterSession::open(store, paths)
            .await
            .unwrap();
        let state = AppState::new(writer, None, "test".into()).with_authz(authz);
        // Authenticated read-write principal, so the deny can't be attributed
        // to the role gate — it must come from the AuthzHook.
        let principal = Arc::new(std::sync::Mutex::new(Some(Principal {
            subject: "tester".into(),
            role: Role::ReadWrite,
            groups: vec![],
        })));
        ServerBackend::new(state, principal)
    }

    #[tokio::test]
    async fn bolt_run_query_consults_authz_hook_and_can_deny_reads() {
        let backend = backend_with_authz(Arc::new(DenyAll)).await;
        // A plain READ — the role gate would allow it; the hook must deny.
        let err = backend
            .run("MATCH (n) RETURN n", Params::new())
            .await
            .expect_err("deny-all hook must reject the read over Bolt");
        assert!(matches!(err, BackendError::Forbidden(_)), "got {err:?}");
    }

    #[tokio::test]
    async fn bolt_default_authz_allows_reads() {
        // NoOp default must not change behavior: the read succeeds.
        let backend = backend_with_authz(Arc::new(crate::authz::NoOpAuthz)).await;
        let out = backend.run("MATCH (n) RETURN n", Params::new()).await;
        assert!(out.is_ok(), "default authz should allow: {out:?}");
    }

    #[tokio::test]
    async fn bolt_logoff_clears_principal() {
        let backend = backend_with_authz(Arc::new(crate::authz::NoOpAuthz)).await;
        // A principal is set; after LOGOFF it must be cleared (falls back to
        // anonymous, so a stale identity can't be reused).
        assert_eq!(backend.principal().subject, "tester");
        backend.logoff().await;
        assert_eq!(
            backend.principal().subject,
            Principal::anonymous_rw().subject,
            "logoff must clear the per-connection principal"
        );
    }
}
