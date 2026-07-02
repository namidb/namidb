//! Per-connection Bolt session.
//!
//! Owns the state machine for a single TCP connection: handshake →
//! HELLO → LOGON → (RUN / PULL / DISCARD / BEGIN / COMMIT / ROLLBACK /
//! RESET / GOODBYE)\*. Delegates to a [`Backend`] trait for the actual
//! Cypher execution so the bolt crate stays independent of
//! `namidb-server` and easy to test.

use std::collections::BTreeMap;
use std::sync::Arc;

use async_trait::async_trait;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tracing::{debug, info, warn};

use crate::chunk::{read_message, write_message};
use crate::error::{BoltError, Result};
use crate::handshake::{negotiate, read_offers, write_response, Version};
use crate::mapping::{params_from_bolt_map, runtime_to_bolt, ElementIdMode};
use crate::message::{Request, Response, POST_AUTH_MESSAGE_BYTES, PRE_AUTH_MESSAGE_BYTES};
use crate::state::State;
use crate::value::Value;

use namidb_query::{Params, Row, RuntimeValue};

/// Server-side identity returned in `SUCCESS` after HELLO.
#[derive(Debug, Clone)]
pub struct ServerInfo {
    /// E.g. `"NamiDB/0.4.1"`.
    pub agent: String,
    /// E.g. `"namidb-prod"`.
    pub connection_id: String,
}

/// Pluggable LOGON / HELLO authenticator.
///
/// Lets an embedder (e.g. the NamiDB cloud gateway) verify Bolt credentials
/// against an external source instead of the built-in [`AuthPolicy::Open`] /
/// [`AuthPolicy::Token`] schemes. The session calls [`authenticate`] with the
/// auth map carried by HELLO (Bolt 4.x) or LOGON (Bolt 5.x) — `scheme`,
/// `principal`, `credentials`. Returning `Err(message)` fails the connection
/// with `Neo.ClientError.Security.Unauthorized` (the message reaches the
/// client); `Ok(())` authenticates it.
///
/// Any per-connection context the authenticator establishes (the resolved
/// principal, the target namespace, …) is shared with the paired [`Backend`]
/// out of band — the embedder constructs both per connection.
///
/// [`authenticate`]: Authenticator::authenticate
#[async_trait]
pub trait Authenticator: Send + Sync {
    /// Authenticate a connection from its HELLO/LOGON auth map.
    async fn authenticate(&self, auth: &BTreeMap<String, Value>)
        -> std::result::Result<(), String>;
}

/// Auth policy applied to LOGON.
#[derive(Clone)]
pub enum AuthPolicy {
    /// Accept any LOGON. Mirrors the REST server's "no auth" mode.
    Open,
    /// Accept `basic` or `bearer` schemes whose credentials match
    /// this token (constant-time compare). Anything else fails.
    Token(Arc<str>),
    /// Delegate authentication to a custom [`Authenticator`] — e.g. the
    /// cloud gateway verifying an API key against the control plane.
    Custom(Arc<dyn Authenticator>),
}

impl std::fmt::Debug for AuthPolicy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AuthPolicy::Open => f.write_str("AuthPolicy::Open"),
            // Never print the token material.
            AuthPolicy::Token(_) => f.write_str("AuthPolicy::Token(***)"),
            AuthPolicy::Custom(_) => f.write_str("AuthPolicy::Custom(..)"),
        }
    }
}

/// What [`Backend::run`] returns. Streamed result production lives
/// behind a separate trait in a follow-up RFC; v0 buffers the whole
/// row set.
#[derive(Debug, Default)]
pub struct RunOutcome {
    /// Field names. `fields[i]` is the column name of the i-th value
    /// in each [`Row`].
    pub fields: Vec<String>,
    /// All rows in execution order. Empty for write-only statements.
    pub rows: Vec<Row>,
    /// Cypher statement type, surfaced in the `t_last` summary
    /// metadata. v0 always emits `"r"` for reads, `"w"` for writes.
    pub statement_type: StatementType,
    /// Write counters (`SUCCESS { stats: {...} }`).
    pub counters: BTreeMap<String, i64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum StatementType {
    #[default]
    Read,
    Write,
    Schema,
    ReadWrite,
}

impl StatementType {
    /// Bolt `type` field reported in the closing `SUCCESS` of a
    /// stream: `"r"`, `"w"`, `"rw"`, `"s"`.
    pub fn as_str(self) -> &'static str {
        match self {
            StatementType::Read => "r",
            StatementType::Write => "w",
            StatementType::Schema => "s",
            StatementType::ReadWrite => "rw",
        }
    }
}

/// Errors a backend can surface. The session translates them to
/// dotted Neo4j error codes via [`backend_error_code`].
#[derive(Debug)]
pub enum BackendError {
    /// Parser rejected the input.
    Syntax(String),
    /// Lowering / planner rejected it.
    Semantic(String),
    /// Parser / lower flagged the feature as out of scope for v0.
    Unsupported(String),
    /// Runtime evaluation error (type mismatch, division by zero, ...).
    Eval(String),
    /// Storage error.
    Storage(String),
    /// A declared schema constraint (e.g. a unique property) was violated.
    Constraint(String),
    /// The authenticated principal is not allowed to run this statement (e.g.
    /// a read-only token attempting a write).
    Forbidden(String),
    /// Anything else.
    Other(String),
}

impl BackendError {
    pub fn code(&self) -> &'static str {
        match self {
            BackendError::Syntax(_) => "Neo.ClientError.Statement.SyntaxError",
            BackendError::Semantic(_) => "Neo.ClientError.Statement.SemanticError",
            BackendError::Unsupported(_) => "Neo.ClientError.Statement.NotSupported",
            BackendError::Eval(_) => "Neo.ClientError.Statement.ArgumentError",
            BackendError::Storage(_) => "Neo.TransientError.General.DatabaseUnavailable",
            BackendError::Constraint(_) => "Neo.ClientError.Schema.ConstraintValidationFailed",
            BackendError::Forbidden(_) => "Neo.ClientError.Security.Forbidden",
            BackendError::Other(_) => "Neo.DatabaseError.General.UnknownError",
        }
    }

    pub fn message(&self) -> &str {
        match self {
            BackendError::Syntax(s)
            | BackendError::Semantic(s)
            | BackendError::Unsupported(s)
            | BackendError::Eval(s)
            | BackendError::Storage(s)
            | BackendError::Constraint(s)
            | BackendError::Forbidden(s)
            | BackendError::Other(s) => s,
        }
    }
}

/// Pluggable Cypher executor. The server crate implements this on top
/// of `WriterSession`; tests implement it with hand-canned rows.
#[async_trait]
pub trait Backend: Send + Sync {
    /// Execute one Cypher statement in auto-commit mode.
    async fn run(
        &self,
        cypher: &str,
        params: Params,
    ) -> std::result::Result<RunOutcome, BackendError>;

    /// Begin an explicit transaction. Subsequent [`Backend::run_in_tx`]
    /// calls stage into it; [`Backend::commit_tx`] makes them durable and
    /// [`Backend::rollback_tx`] discards them. The default is a no-op so a
    /// backend without transaction support keeps working (its in-tx writes
    /// just behave like auto-commit).
    async fn begin_tx(&self) -> std::result::Result<(), BackendError> {
        Ok(())
    }

    /// Execute one statement inside the open explicit transaction. The
    /// default delegates to [`Backend::run`] (auto-commit), which is the
    /// pre-transaction behaviour for backends that do not override it.
    async fn run_in_tx(
        &self,
        cypher: &str,
        params: Params,
    ) -> std::result::Result<RunOutcome, BackendError> {
        self.run(cypher, params).await
    }

    /// Commit the open explicit transaction, making its staged statements
    /// durable. Default is a no-op.
    async fn commit_tx(&self) -> std::result::Result<(), BackendError> {
        Ok(())
    }

    /// Roll back the open explicit transaction, discarding its staged
    /// statements. Default is a no-op.
    async fn rollback_tx(&self) -> std::result::Result<(), BackendError> {
        Ok(())
    }

    /// Optional override for the manifest version reported as the
    /// bookmark after COMMIT. Default returns `None` and the session
    /// emits no bookmark.
    async fn current_bookmark(&self) -> Option<String> {
        None
    }

    /// Called when the client issues LOGOFF, returning the connection to the
    /// unauthenticated state. The default is a no-op. An embedder that binds
    /// per-connection identity to its [`Backend`] out of band — e.g. a cloud
    /// edge that resolved an API key to a tenant at LOGON via a custom
    /// [`Authenticator`] — overrides this to drop that identity, so a
    /// subsequent RESET (which returns the connection to `Ready`) cannot
    /// resume executing as the logged-off principal.
    async fn logoff(&self) {}
}

/// One Bolt connection. Created once per `accept()` and driven to
/// completion in a single task.
pub struct Session<S: AsyncReadExt + AsyncWriteExt + Unpin> {
    socket: S,
    info: ServerInfo,
    auth: AuthPolicy,
    backend: Arc<dyn Backend>,
    state: State,
    version: Option<Version>,
    /// `statement_type` of the in-flight stream, surfaced in the
    /// closing `SUCCESS` after PULL/DISCARD. `None` while no stream
    /// is active.
    pending_statement_type: Option<StatementType>,
    /// Write counters of the in-flight stream, emitted as `stats` in the
    /// closing `SUCCESS` after PULL/DISCARD. Empty for reads.
    pending_counters: BTreeMap<String, i64>,
    /// While an explicit transaction is open the backend holds the writer
    /// lock, so an idle client would pin it indefinitely. When set, a read
    /// that blocks longer than this with a transaction open rolls the
    /// transaction back (releasing the writer) and fails it. `None` (the
    /// default) keeps the legacy unbounded behaviour for test backends.
    tx_idle_timeout: Option<std::time::Duration>,
    /// Whether this session has completed authentication (the v5 HELLO +
    /// LOGON handshake, or the v4 HELLO that carries auth). RESET only
    /// recovers a session to READY once this is set: before auth a RESET
    /// must not grant READY, or a client could skip HELLO/LOGON entirely
    /// (handshake -> RESET -> RUN). LOGOFF clears it, forcing a fresh LOGON
    /// before any further work.
    authenticated: bool,
}

impl<S: AsyncReadExt + AsyncWriteExt + Unpin> Session<S> {
    pub fn new(socket: S, info: ServerInfo, auth: AuthPolicy, backend: Arc<dyn Backend>) -> Self {
        Self {
            socket,
            info,
            auth,
            backend,
            state: State::Negotiation,
            version: None,
            pending_statement_type: None,
            pending_counters: BTreeMap::new(),
            tx_idle_timeout: None,
            authenticated: false,
        }
    }

    /// Set the idle timeout applied while an explicit transaction is open.
    /// `None` disables it (a transaction may stay open indefinitely).
    pub fn with_tx_idle_timeout(mut self, timeout: Option<std::time::Duration>) -> Self {
        self.tx_idle_timeout = timeout;
        self
    }

    /// Run the session to completion. Returns once the client sends
    /// GOODBYE, the socket closes, or a fatal protocol error fires.
    pub async fn run(mut self) -> Result<()> {
        self.do_handshake().await?;
        if self.version.is_none() {
            return Ok(()); // negotiation failed; we already wrote [0;4]
        }
        let element_mode = ElementIdMode::from_major(self.version.unwrap().major);
        loop {
            let max = if self.state == State::Connected || self.state == State::Authentication {
                PRE_AUTH_MESSAGE_BYTES
            } else {
                POST_AUTH_MESSAGE_BYTES
            };
            // Bound how long the writer lock is pinned by an open
            // transaction: if the client idles past `tx_idle_timeout` while
            // in a transaction, roll it back to release the writer and fail
            // the transaction.
            let in_tx = matches!(self.state, State::TxReady | State::TxStreaming);
            let idle = self.tx_idle_timeout;
            let read = read_message(&mut self.socket, max);
            let read_result = match (idle, in_tx) {
                (Some(t), true) => match tokio::time::timeout(t, read).await {
                    Ok(r) => r,
                    Err(_elapsed) => {
                        let _ = self.backend.rollback_tx().await;
                        self.state = State::Failed;
                        self.write_failure(
                            "Neo.TransientError.Transaction.LockClientStopped",
                            "transaction idle timeout; rolled back to release the writer"
                                .to_string(),
                        )
                        .await?;
                        continue;
                    }
                },
                _ => read.await,
            };
            let body = match read_result {
                Ok(b) => b,
                Err(BoltError::Io(e)) if e.kind() == std::io::ErrorKind::UnexpectedEof => {
                    debug!("bolt connection closed by client");
                    // A client that drops mid-transaction (crash, kill, network
                    // partition) never sends ROLLBACK/GOODBYE; without this the
                    // staged batch would linger in the shared writer and be
                    // sealed by the next unrelated commit. Roll it back here.
                    self.rollback_if_open_tx().await;
                    return Ok(());
                }
                Err(e) => {
                    self.rollback_if_open_tx().await;
                    return Err(e);
                }
            };
            let request = match Request::decode(&body, max) {
                Ok(r) => r,
                Err(e) => {
                    self.write_failure(
                        "Neo.ClientError.Request.Invalid",
                        format!("malformed Bolt message: {e}"),
                    )
                    .await?;
                    self.state = State::Failed;
                    continue;
                }
            };
            debug!(name = request.name(), state = ?self.state, "bolt request");
            let goodbye = matches!(request, Request::Goodbye);
            if let Err(e) = self.handle(request, element_mode).await {
                warn!(error = %e, "bolt session error");
                // A transport/protocol error tears the connection down; roll
                // back any open transaction so its staged writes are discarded
                // rather than left in the shared writer for the next commit.
                self.rollback_if_open_tx().await;
                return Err(e);
            }
            if goodbye || self.state == State::Defunct {
                return Ok(());
            }
        }
    }

    /// Roll back any open/dangling transaction to discard its staged writes and
    /// release the writer it pins. A no-op when no transaction is open.
    /// FAILED is included: a statement that failed mid-transaction leaves the
    /// state FAILED while the writer can still be pinned with a staged batch.
    async fn rollback_if_open_tx(&mut self) {
        if matches!(
            self.state,
            State::TxReady | State::TxStreaming | State::Failed
        ) {
            let _ = self.backend.rollback_tx().await;
        }
    }

    async fn do_handshake(&mut self) -> Result<()> {
        let offers = read_offers(&mut self.socket).await?;
        let version = negotiate(&offers);
        write_response(&mut self.socket, version).await?;
        match version {
            Some(v) => {
                info!(version = %v, "bolt session negotiated");
                self.version = Some(v);
                self.state = State::Connected;
            }
            None => {
                warn!("bolt handshake failed — no supported version offered");
                self.state = State::Defunct;
            }
        }
        Ok(())
    }

    async fn handle(&mut self, req: Request, element_mode: ElementIdMode) -> Result<()> {
        // RESET recovers a session to READY, but only once it has
        // authenticated. Before auth (CONNECTED/AUTHENTICATION) a RESET must
        // not grant READY, or a client could skip HELLO/LOGON entirely
        // (handshake -> RESET -> RUN). When unauthenticated it falls through
        // to the per-state handlers below, which reject it. GOODBYE always
        // ends. Either one while a transaction may still be open rolls it back
        // so its staged writes are discarded and the writer it holds is
        // released — including a transaction left dangling by a statement that
        // failed mid-tx, where the state is FAILED (not TX_READY) but the
        // writer can still be pinned. `rollback_tx` is a no-op with no open tx.
        if matches!(req, Request::Reset) && self.authenticated {
            if matches!(
                self.state,
                State::TxReady | State::TxStreaming | State::Failed
            ) {
                let _ = self.backend.rollback_tx().await;
            }
            self.state = State::Ready;
            return self.write_response(Response::success_empty()).await;
        }
        if matches!(req, Request::Goodbye) {
            if matches!(
                self.state,
                State::TxReady | State::TxStreaming | State::Failed
            ) {
                let _ = self.backend.rollback_tx().await;
            }
            self.state = State::Defunct;
            return Ok(());
        }
        // After a FAILURE, every non-RESET non-GOODBYE message is IGNORED.
        if self.state == State::Failed {
            return self.write_response(Response::Ignored).await;
        }

        match self.state {
            State::Connected => self.handle_in_connected(req).await,
            State::Authentication => self.handle_in_authentication(req).await,
            State::Ready => self.handle_in_ready(req, element_mode).await,
            State::Streaming | State::TxStreaming => {
                self.handle_in_streaming(req, element_mode).await
            }
            State::TxReady => self.handle_in_tx_ready(req, element_mode).await,
            State::Negotiation | State::Failed | State::Defunct => {
                // Negotiation is handled in do_handshake; the others
                // were short-circuited above.
                self.write_response(Response::Ignored).await
            }
        }
    }

    async fn handle_in_connected(&mut self, req: Request) -> Result<()> {
        let Request::Hello(extra) = req else {
            return self.invalid_state("HELLO required").await;
        };
        let mut meta = BTreeMap::new();
        meta.insert("server".into(), Value::String(self.info.agent.clone()));
        meta.insert(
            "connection_id".into(),
            Value::String(self.info.connection_id.clone()),
        );
        meta.insert("hints".into(), Value::Map(BTreeMap::new()));
        if let Some(v) = self.version {
            meta.insert("protocol_version".into(), Value::String(format!("{}", v)));
        }
        // HELLO is the only place v4.4 carries auth; v5 splits to LOGON.
        let major = self.version.map(|v| v.major).unwrap_or(5);
        if major >= 5 {
            self.state = State::Authentication;
            // No auth fields in v5 HELLO; just echo the metadata.
            self.write_response(Response::Success(meta)).await?;
        } else {
            // v4 HELLO carries scheme/principal/credentials.
            if let Err(e) = self.authenticate(&extra).await {
                self.state = State::Failed;
                return self
                    .write_failure("Neo.ClientError.Security.Unauthorized", e)
                    .await;
            }
            self.authenticated = true;
            self.state = State::Ready;
            self.write_response(Response::Success(meta)).await?;
        }
        Ok(())
    }

    async fn handle_in_authentication(&mut self, req: Request) -> Result<()> {
        let Request::Logon(extra) = req else {
            return self.invalid_state("LOGON required").await;
        };
        if let Err(e) = self.authenticate(&extra).await {
            self.state = State::Failed;
            return self
                .write_failure("Neo.ClientError.Security.Unauthorized", e)
                .await;
        }
        self.authenticated = true;
        self.state = State::Ready;
        self.write_response(Response::success_empty()).await
    }

    async fn handle_in_ready(&mut self, req: Request, element_mode: ElementIdMode) -> Result<()> {
        match req {
            Request::Run {
                cypher,
                params,
                extra: _,
            } => self.execute_run(&cypher, params, element_mode, false).await,
            Request::Begin(_) => match self.backend.begin_tx().await {
                Ok(()) => {
                    self.state = State::TxReady;
                    self.write_response(Response::success_empty()).await
                }
                Err(e) => {
                    self.state = State::Failed;
                    self.write_failure(e.code(), e.message().to_string()).await
                }
            },
            Request::Route { .. } => self.respond_route().await,
            Request::Logoff => {
                // Drop any per-connection identity the embedder bound out of
                // band, then return to the unauthenticated state. Clearing
                // `authenticated` is what makes a later RESET refuse to recover
                // to Ready until a fresh LOGON re-authenticates.
                self.backend.logoff().await;
                self.authenticated = false;
                self.state = State::Authentication;
                self.write_response(Response::success_empty()).await
            }
            Request::Telemetry(_) => self.write_response(Response::success_empty()).await,
            Request::Pull { .. } | Request::Discard { .. } => {
                self.invalid_state("PULL/DISCARD outside a stream").await
            }
            Request::Commit | Request::Rollback => {
                self.invalid_state("COMMIT/ROLLBACK outside a transaction")
                    .await
            }
            _ => self.invalid_state("unexpected message in READY").await,
        }
    }

    async fn handle_in_tx_ready(
        &mut self,
        req: Request,
        element_mode: ElementIdMode,
    ) -> Result<()> {
        match req {
            Request::Run {
                cypher,
                params,
                extra: _,
            } => self.execute_run(&cypher, params, element_mode, true).await,
            Request::Commit => self.commit(element_mode).await,
            Request::Rollback => match self.backend.rollback_tx().await {
                Ok(()) => {
                    self.state = State::Ready;
                    self.write_response(Response::success_empty()).await
                }
                Err(e) => {
                    self.state = State::Failed;
                    self.write_failure(e.code(), e.message().to_string()).await
                }
            },
            _ => self.invalid_state("unexpected message in TX_READY").await,
        }
    }

    async fn handle_in_streaming(
        &mut self,
        req: Request,
        element_mode: ElementIdMode,
    ) -> Result<()> {
        let _ = element_mode;
        match req {
            Request::Pull { extra: _ } | Request::Discard { extra: _ } => {
                // The current backend buffers the whole result inside
                // `execute_run`, so the cached rows already streamed.
                // We just answer the PULL/DISCARD with an empty
                // SUCCESS marking the stream done.
                let mut meta = BTreeMap::new();
                let stype = self
                    .pending_statement_type
                    .take()
                    .unwrap_or(StatementType::Read);
                meta.insert("type".into(), Value::String(stype.as_str().into()));
                // Emit write counters (Neo4j `stats`) so a client shows
                // "N created / deleted" after a write. Empty for reads.
                let counters = std::mem::take(&mut self.pending_counters);
                if !counters.is_empty() {
                    let stats: BTreeMap<String, Value> = counters
                        .into_iter()
                        .map(|(k, v)| (k, Value::Int(v)))
                        .collect();
                    meta.insert("stats".into(), Value::Map(stats));
                }
                if self.state == State::TxStreaming {
                    self.state = State::TxReady;
                } else {
                    self.state = State::Ready;
                }
                self.write_response(Response::Success(meta)).await
            }
            _ => {
                self.invalid_state("only PULL/DISCARD valid in STREAMING")
                    .await
            }
        }
    }

    async fn execute_run(
        &mut self,
        cypher: &str,
        bolt_params: BTreeMap<String, Value>,
        element_mode: ElementIdMode,
        inside_tx: bool,
    ) -> Result<()> {
        let params = params_from_bolt_map(&bolt_params);
        // Inside an explicit transaction the statement stages into the open
        // tx (committed at COMMIT, discarded at ROLLBACK); a bare RUN
        // auto-commits.
        let run_result = if inside_tx {
            self.backend.run_in_tx(cypher, params).await
        } else {
            self.backend.run(cypher, params).await
        };
        let outcome = match run_result {
            Ok(o) => o,
            Err(e) => {
                // A statement that fails inside an explicit transaction aborts
                // it. The backend has already discarded any staged batch, but
                // the transaction still pins the global writer lock. Roll it
                // back now to release that writer: otherwise the session moves
                // to FAILED (where the tx idle timeout no longer arms), and a
                // client that idles there — or only ever sends RESET — would
                // pin the single writer, wedging every other writer on the
                // server, until it disconnects. `rollback_tx` is a no-op when
                // no transaction is open, so the auto-commit path is unaffected.
                if inside_tx {
                    let _ = self.backend.rollback_tx().await;
                }
                self.state = State::Failed;
                return self.write_failure(e.code(), e.message().to_string()).await;
            }
        };

        // 1) SUCCESS { fields, qid? } announcing the field list.
        let mut head_meta = BTreeMap::new();
        head_meta.insert(
            "fields".into(),
            Value::List(outcome.fields.iter().cloned().map(Value::String).collect()),
        );
        head_meta.insert("t_first".into(), Value::Int(0));
        self.write_response(Response::Success(head_meta)).await?;

        // 2) one RECORD per row.
        for row in &outcome.rows {
            let mut values = Vec::with_capacity(outcome.fields.len());
            for name in &outcome.fields {
                let v = row
                    .bindings
                    .get(name)
                    .cloned()
                    .unwrap_or(RuntimeValue::Null);
                values.push(runtime_to_bolt(&v, element_mode));
            }
            self.write_response(Response::Record(values)).await?;
        }

        // 3) Transition to STREAMING and wait for PULL/DISCARD that
        //    will close the stream. The Bolt protocol requires it.
        self.state = if inside_tx {
            State::TxStreaming
        } else {
            State::Streaming
        };
        self.pending_statement_type = Some(outcome.statement_type);
        self.pending_counters = outcome.counters;
        // Buffered model: rows already emitted. PULL/DISCARD will
        // observe the streaming state and answer with a closing
        // SUCCESS in handle_in_streaming.
        Ok(())
    }

    async fn commit(&mut self, _element_mode: ElementIdMode) -> Result<()> {
        // Make the transaction's staged statements durable. A failure here
        // (e.g. a lost manifest CAS) is the abort surface; surface it as a
        // FAILURE and the client retries.
        if let Err(e) = self.backend.commit_tx().await {
            self.state = State::Failed;
            return self.write_failure(e.code(), e.message().to_string()).await;
        }
        let mut meta = BTreeMap::new();
        if let Some(bm) = self.backend.current_bookmark().await {
            meta.insert("bookmark".into(), Value::String(bm));
        }
        self.state = State::Ready;
        self.write_response(Response::Success(meta)).await
    }

    async fn respond_route(&mut self) -> Result<()> {
        // Single-server cluster — RFC-022 §"Q2 ROUTE behaviour".
        let mut rt = BTreeMap::new();
        rt.insert("ttl".into(), Value::Int(300));
        rt.insert("db".into(), Value::String("namidb".into()));
        let server_block = |role: &str| -> BTreeMap<String, Value> {
            let mut m = BTreeMap::new();
            m.insert("role".into(), Value::String(role.into()));
            m.insert(
                "addresses".into(),
                Value::List(vec![Value::String("self".into())]),
            );
            m
        };
        rt.insert(
            "servers".into(),
            Value::List(vec![
                Value::Map(server_block("WRITE")),
                Value::Map(server_block("READ")),
                Value::Map(server_block("ROUTE")),
            ]),
        );
        let mut meta = BTreeMap::new();
        meta.insert("rt".into(), Value::Map(rt));
        self.write_response(Response::Success(meta)).await
    }

    async fn authenticate(
        &self,
        extra: &BTreeMap<String, Value>,
    ) -> std::result::Result<(), String> {
        // A custom authenticator owns the whole decision and receives the
        // full auth map (scheme / principal / credentials).
        if let AuthPolicy::Custom(authenticator) = &self.auth {
            return authenticator.authenticate(extra).await;
        }
        let scheme = extra
            .get("scheme")
            .and_then(|v| match v {
                Value::String(s) => Some(s.as_str()),
                _ => None,
            })
            .unwrap_or("none");
        match (&self.auth, scheme) {
            (AuthPolicy::Open, _) => Ok(()),
            (AuthPolicy::Token(_), "none") => {
                Err("server requires authentication; got scheme=\"none\"".into())
            }
            (AuthPolicy::Token(expected), "basic") | (AuthPolicy::Token(expected), "bearer") => {
                let presented = extra.get("credentials").and_then(|v| match v {
                    Value::String(s) => Some(s.as_str()),
                    _ => None,
                });
                match presented {
                    Some(token) if constant_time_eq(token.as_bytes(), expected.as_bytes()) => {
                        Ok(())
                    }
                    _ => Err("invalid credentials".into()),
                }
            }
            (_, scheme) => Err(format!("unsupported auth scheme `{scheme}`")),
        }
    }

    async fn invalid_state(&mut self, why: &str) -> Result<()> {
        self.state = State::Failed;
        self.write_failure(
            "Neo.ClientError.Request.Invalid",
            format!("invalid request in state {:?}: {}", self.state, why),
        )
        .await
    }

    async fn write_failure(&mut self, code: &str, message: impl Into<String>) -> Result<()> {
        self.write_response(Response::failure(code, message)).await
    }

    async fn write_response(&mut self, resp: Response) -> Result<()> {
        let body = resp.encode()?;
        write_message(&mut self.socket, &body).await
    }
}

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codec::encode;
    use crate::handshake::MAGIC;
    use bytes::BytesMut;
    use namidb_query::exec::NodeValue;
    use namidb_query::Row;
    use std::sync::Mutex as StdMutex;
    use tokio::io::duplex;

    struct StaticBackend {
        outcome: StdMutex<Option<RunOutcome>>,
    }

    #[async_trait]
    impl Backend for StaticBackend {
        async fn run(
            &self,
            _cypher: &str,
            _params: Params,
        ) -> std::result::Result<RunOutcome, BackendError> {
            Ok(self.outcome.lock().unwrap().take().unwrap_or_default())
        }
    }

    fn fixture_session<S: AsyncReadExt + AsyncWriteExt + Unpin>(
        socket: S,
        outcome: RunOutcome,
        auth: AuthPolicy,
    ) -> Session<S> {
        let backend = Arc::new(StaticBackend {
            outcome: StdMutex::new(Some(outcome)),
        });
        Session::new(
            socket,
            ServerInfo {
                agent: "NamiDB/test".into(),
                connection_id: "test-conn".into(),
            },
            auth,
            backend,
        )
    }

    /// LOGOFF must invoke `Backend::logoff()` so an embedder can drop the
    /// per-connection identity it bound out of band — otherwise a later RESET
    /// (any-state → Ready) would let the connection keep executing as the
    /// logged-off principal. Regression test for that auth-state bypass.
    #[tokio::test]
    async fn logoff_invokes_backend_logoff_hook() {
        use std::sync::atomic::{AtomicBool, Ordering};

        struct LogoffBackend {
            logged_off: Arc<AtomicBool>,
        }
        #[async_trait]
        impl Backend for LogoffBackend {
            async fn run(
                &self,
                _cypher: &str,
                _params: Params,
            ) -> std::result::Result<RunOutcome, BackendError> {
                Ok(RunOutcome::default())
            }
            async fn logoff(&self) {
                self.logged_off.store(true, Ordering::SeqCst);
            }
        }

        let flag = Arc::new(AtomicBool::new(false));
        let (mut client, server) = duplex(64 * 1024);
        let session = Session::new(
            server,
            ServerInfo {
                agent: "NamiDB/test".into(),
                connection_id: "test-conn".into(),
            },
            AuthPolicy::Open,
            Arc::new(LogoffBackend {
                logged_off: flag.clone(),
            }),
        );
        let task = tokio::spawn(async move { session.run().await });

        send_handshake(&mut client).await;
        let _ = read_handshake_reply(&mut client).await;

        // HELLO + LOGON (open auth).
        write_msg(
            &mut client,
            &pack_request(&Value::Struct {
                tag: crate::value::struct_tag::HELLO,
                fields: vec![Value::Map(BTreeMap::new())],
            }),
        )
        .await;
        let _ = read_msg(&mut client).await;
        write_msg(
            &mut client,
            &pack_request(&Value::Struct {
                tag: crate::value::struct_tag::LOGON,
                fields: vec![Value::Map({
                    let mut m = BTreeMap::new();
                    m.insert("scheme".into(), Value::String("none".into()));
                    m
                })],
            }),
        )
        .await;
        let _ = read_msg(&mut client).await;
        assert!(
            !flag.load(Ordering::SeqCst),
            "logoff not called before LOGOFF"
        );

        // LOGOFF must ack AND invoke the hook.
        write_msg(
            &mut client,
            &pack_request(&Value::Struct {
                tag: crate::value::struct_tag::LOGOFF,
                fields: vec![],
            }),
        )
        .await;
        let resp = read_msg(&mut client).await;
        assert!(
            matches!(decode_response(&resp), Response::Success(_)),
            "LOGOFF acked"
        );
        assert!(
            flag.load(Ordering::SeqCst),
            "LOGOFF must invoke Backend::logoff()"
        );

        write_msg(
            &mut client,
            &pack_request(&Value::Struct {
                tag: crate::value::struct_tag::GOODBYE,
                fields: vec![],
            }),
        )
        .await;
        drop(client);
        let _ = task.await.unwrap();
    }

    async fn send_handshake<W: AsyncWriteExt + Unpin>(w: &mut W) {
        let mut bytes = Vec::with_capacity(20);
        bytes.extend_from_slice(&MAGIC);
        bytes.extend_from_slice(&[0, 0, 4, 5]); // 5.4
        bytes.extend_from_slice(&[0; 12]);
        w.write_all(&bytes).await.unwrap();
    }

    async fn read_handshake_reply<R: AsyncReadExt + Unpin>(r: &mut R) -> [u8; 4] {
        let mut buf = [0u8; 4];
        r.read_exact(&mut buf).await.unwrap();
        buf
    }

    fn pack_request(req: &Value) -> Vec<u8> {
        let mut buf = BytesMut::new();
        encode(&mut buf, req).unwrap();
        buf.to_vec()
    }

    async fn write_msg<W: AsyncWriteExt + Unpin>(w: &mut W, body: &[u8]) {
        write_message(w, body).await.unwrap();
    }

    async fn read_msg<R: AsyncReadExt + Unpin>(r: &mut R) -> Vec<u8> {
        read_message(r, POST_AUTH_MESSAGE_BYTES).await.unwrap()
    }

    fn decode_response(body: &[u8]) -> Response {
        let mut slice: &[u8] = body;
        let v = crate::codec::decode(&mut slice).unwrap();
        let (tag, mut fields) = match v {
            Value::Struct { tag, fields } => (tag, fields),
            other => panic!("expected struct, got {:?}", other),
        };
        match tag {
            crate::value::struct_tag::SUCCESS => Response::Success(
                fields
                    .pop()
                    .and_then(|v| match v {
                        Value::Map(m) => Some(m),
                        _ => None,
                    })
                    .unwrap_or_default(),
            ),
            crate::value::struct_tag::RECORD => Response::Record(
                fields
                    .pop()
                    .and_then(|v| match v {
                        Value::List(l) => Some(l),
                        _ => None,
                    })
                    .unwrap_or_default(),
            ),
            crate::value::struct_tag::IGNORED => Response::Ignored,
            crate::value::struct_tag::FAILURE => Response::Failure(
                fields
                    .pop()
                    .and_then(|v| match v {
                        Value::Map(m) => Some(m),
                        _ => None,
                    })
                    .unwrap_or_default(),
            ),
            other => panic!("unexpected response tag 0x{:02X}", other),
        }
    }

    #[tokio::test]
    async fn happy_path_run_with_one_row() {
        let outcome = RunOutcome {
            fields: vec!["n".into()],
            rows: vec![{
                let mut bindings = std::collections::BTreeMap::new();
                bindings.insert("n".into(), RuntimeValue::Integer(42));
                Row { bindings }
            }],
            ..Default::default()
        };
        let (mut client, server) = duplex(64 * 1024);
        let session = fixture_session(server, outcome, AuthPolicy::Open);
        let task = tokio::spawn(async move { session.run().await });

        // Handshake.
        send_handshake(&mut client).await;
        let reply = read_handshake_reply(&mut client).await;
        assert_eq!(reply, [0, 0, 4, 5]);

        // HELLO.
        let hello = Value::Struct {
            tag: crate::value::struct_tag::HELLO,
            fields: vec![Value::Map(BTreeMap::new())],
        };
        write_msg(&mut client, &pack_request(&hello)).await;
        let resp = read_msg(&mut client).await;
        assert!(matches!(decode_response(&resp), Response::Success(_)));

        // LOGON (open auth).
        let logon = Value::Struct {
            tag: crate::value::struct_tag::LOGON,
            fields: vec![Value::Map({
                let mut m = BTreeMap::new();
                m.insert("scheme".into(), Value::String("none".into()));
                m
            })],
        };
        write_msg(&mut client, &pack_request(&logon)).await;
        let resp = read_msg(&mut client).await;
        assert!(matches!(decode_response(&resp), Response::Success(_)));

        // RUN.
        let run = Value::Struct {
            tag: crate::value::struct_tag::RUN,
            fields: vec![
                Value::String("RETURN 42 AS n".into()),
                Value::Map(BTreeMap::new()),
                Value::Map(BTreeMap::new()),
            ],
        };
        write_msg(&mut client, &pack_request(&run)).await;
        // First response: SUCCESS { fields: ["n"] }
        let r1 = read_msg(&mut client).await;
        match decode_response(&r1) {
            Response::Success(meta) => assert!(meta.contains_key("fields")),
            other => panic!("expected SUCCESS, got {:?}", other),
        }
        // Second response: RECORD [42]
        let r2 = read_msg(&mut client).await;
        match decode_response(&r2) {
            Response::Record(values) => {
                assert_eq!(values, vec![Value::Int(42)]);
            }
            other => panic!("expected RECORD, got {:?}", other),
        }

        // PULL — closes the stream.
        let pull = Value::Struct {
            tag: crate::value::struct_tag::PULL,
            fields: vec![Value::Map({
                let mut m = BTreeMap::new();
                m.insert("n".into(), Value::Int(-1));
                m
            })],
        };
        write_msg(&mut client, &pack_request(&pull)).await;
        let r3 = read_msg(&mut client).await;
        assert!(matches!(decode_response(&r3), Response::Success(_)));

        // GOODBYE.
        let bye = Value::Struct {
            tag: crate::value::struct_tag::GOODBYE,
            fields: vec![],
        };
        write_msg(&mut client, &pack_request(&bye)).await;
        drop(client);
        let res = task.await.unwrap();
        assert!(res.is_ok());
    }

    #[tokio::test]
    async fn bad_auth_returns_failure_then_resets() {
        let (mut client, server) = duplex(16 * 1024);
        let session = fixture_session(
            server,
            RunOutcome::default(),
            AuthPolicy::Token(Arc::from("correct-token")),
        );
        let task = tokio::spawn(async move { session.run().await });

        send_handshake(&mut client).await;
        let _ = read_handshake_reply(&mut client).await;

        write_msg(
            &mut client,
            &pack_request(&Value::Struct {
                tag: crate::value::struct_tag::HELLO,
                fields: vec![Value::Map(BTreeMap::new())],
            }),
        )
        .await;
        let _ = read_msg(&mut client).await;

        write_msg(
            &mut client,
            &pack_request(&Value::Struct {
                tag: crate::value::struct_tag::LOGON,
                fields: vec![Value::Map({
                    let mut m = BTreeMap::new();
                    m.insert("scheme".into(), Value::String("basic".into()));
                    m.insert("credentials".into(), Value::String("wrong".into()));
                    m
                })],
            }),
        )
        .await;
        let r = read_msg(&mut client).await;
        let resp = decode_response(&r);
        match resp {
            Response::Failure(meta) => {
                assert_eq!(
                    meta.get("code"),
                    Some(&Value::String(
                        "Neo.ClientError.Security.Unauthorized".into()
                    ))
                );
            }
            other => panic!("expected FAILURE, got {:?}", other),
        }

        // After FAILURE, RUN should be IGNORED.
        write_msg(
            &mut client,
            &pack_request(&Value::Struct {
                tag: crate::value::struct_tag::RUN,
                fields: vec![
                    Value::String("RETURN 1".into()),
                    Value::Map(BTreeMap::new()),
                    Value::Map(BTreeMap::new()),
                ],
            }),
        )
        .await;
        let r = read_msg(&mut client).await;
        assert!(matches!(decode_response(&r), Response::Ignored));

        // GOODBYE closes.
        write_msg(
            &mut client,
            &pack_request(&Value::Struct {
                tag: crate::value::struct_tag::GOODBYE,
                fields: vec![],
            }),
        )
        .await;
        drop(client);
        let _ = task.await.unwrap();
    }

    #[tokio::test]
    async fn reset_before_auth_does_not_bypass_authentication() {
        // A client that completes the handshake but never sends HELLO/LOGON
        // must not reach READY (and run queries) by sending RESET. Before the
        // fix, RESET was handled ahead of the per-state dispatch and jumped
        // unconditionally to READY (handshake -> RESET -> RUN bypass).
        let (mut client, server) = duplex(16 * 1024);
        let session = fixture_session(
            server,
            RunOutcome::default(),
            AuthPolicy::Token(Arc::from("correct-token")),
        );
        let task = tokio::spawn(async move { session.run().await });

        send_handshake(&mut client).await;
        let _ = read_handshake_reply(&mut client).await;

        // RESET straight after the handshake, with no HELLO/LOGON.
        write_msg(
            &mut client,
            &pack_request(&Value::Struct {
                tag: crate::value::struct_tag::RESET,
                fields: vec![],
            }),
        )
        .await;
        // Must be rejected, not answered with SUCCESS (a SUCCESS would mean
        // the session reached READY unauthenticated).
        match decode_response(&read_msg(&mut client).await) {
            Response::Failure(meta) => assert_eq!(
                meta.get("code"),
                Some(&Value::String("Neo.ClientError.Request.Invalid".into())),
                "pre-auth RESET must fail as an invalid request"
            ),
            other => panic!("expected FAILURE for pre-auth RESET, got {:?}", other),
        }

        // And a RUN must still be refused (IGNORED after the failure),
        // proving no query executes on an unauthenticated connection.
        write_msg(
            &mut client,
            &pack_request(&Value::Struct {
                tag: crate::value::struct_tag::RUN,
                fields: vec![
                    Value::String("RETURN 1".into()),
                    Value::Map(BTreeMap::new()),
                    Value::Map(BTreeMap::new()),
                ],
            }),
        )
        .await;
        assert!(matches!(
            decode_response(&read_msg(&mut client).await),
            Response::Ignored
        ));

        drop(client);
        let _ = task.await.unwrap();
    }

    #[tokio::test]
    async fn reset_after_auth_recovers_to_ready() {
        // RESET on an authenticated session still recovers to READY: the fix
        // gates pre-auth RESET only, it must not break the normal recovery.
        let (mut client, server) = duplex(16 * 1024);
        let session = fixture_session(
            server,
            RunOutcome::default(),
            AuthPolicy::Token(Arc::from("correct-token")),
        );
        let task = tokio::spawn(async move { session.run().await });

        send_handshake(&mut client).await;
        let _ = read_handshake_reply(&mut client).await;

        write_msg(
            &mut client,
            &pack_request(&Value::Struct {
                tag: crate::value::struct_tag::HELLO,
                fields: vec![Value::Map(BTreeMap::new())],
            }),
        )
        .await;
        let _ = read_msg(&mut client).await;

        write_msg(
            &mut client,
            &pack_request(&Value::Struct {
                tag: crate::value::struct_tag::LOGON,
                fields: vec![Value::Map({
                    let mut m = BTreeMap::new();
                    m.insert("scheme".into(), Value::String("basic".into()));
                    m.insert("credentials".into(), Value::String("correct-token".into()));
                    m
                })],
            }),
        )
        .await;
        assert!(matches!(
            decode_response(&read_msg(&mut client).await),
            Response::Success(_)
        ));

        // RESET on the authenticated session returns SUCCESS.
        write_msg(
            &mut client,
            &pack_request(&Value::Struct {
                tag: crate::value::struct_tag::RESET,
                fields: vec![],
            }),
        )
        .await;
        assert!(matches!(
            decode_response(&read_msg(&mut client).await),
            Response::Success(_)
        ));

        drop(client);
        let _ = task.await.unwrap();
    }

    #[tokio::test]
    async fn failed_in_tx_statement_rolls_back_to_release_writer() {
        // A statement that fails inside an explicit transaction must roll the
        // transaction back so the backend releases the global writer it took at
        // BEGIN. Otherwise the session sits in FAILED still holding the writer
        // until the connection closes — a single client whose in-tx statement
        // fails (e.g. a mid-tx timeout) could wedge every other writer on the
        // server. Regression test for that writer-lock leak.
        use std::sync::atomic::{AtomicUsize, Ordering};

        struct TxFailBackend {
            rollbacks: Arc<AtomicUsize>,
        }
        #[async_trait]
        impl Backend for TxFailBackend {
            async fn run(
                &self,
                _cypher: &str,
                _params: Params,
            ) -> std::result::Result<RunOutcome, BackendError> {
                Ok(RunOutcome::default())
            }
            async fn run_in_tx(
                &self,
                _cypher: &str,
                _params: Params,
            ) -> std::result::Result<RunOutcome, BackendError> {
                // The in-tx statement fails (stands in for a mid-tx timeout or
                // eval error).
                Err(BackendError::Eval("boom".into()))
            }
            async fn rollback_tx(&self) -> std::result::Result<(), BackendError> {
                self.rollbacks.fetch_add(1, Ordering::SeqCst);
                Ok(())
            }
        }

        let rollbacks = Arc::new(AtomicUsize::new(0));
        let (mut client, server) = duplex(64 * 1024);
        let session = Session::new(
            server,
            ServerInfo {
                agent: "NamiDB/test".into(),
                connection_id: "test-conn".into(),
            },
            AuthPolicy::Open,
            Arc::new(TxFailBackend {
                rollbacks: rollbacks.clone(),
            }),
        );
        let task = tokio::spawn(async move { session.run().await });

        send_handshake(&mut client).await;
        let _ = read_handshake_reply(&mut client).await;

        // HELLO + LOGON (open auth) -> READY.
        write_msg(
            &mut client,
            &pack_request(&Value::Struct {
                tag: crate::value::struct_tag::HELLO,
                fields: vec![Value::Map(BTreeMap::new())],
            }),
        )
        .await;
        let _ = read_msg(&mut client).await;
        write_msg(
            &mut client,
            &pack_request(&Value::Struct {
                tag: crate::value::struct_tag::LOGON,
                fields: vec![Value::Map({
                    let mut m = BTreeMap::new();
                    m.insert("scheme".into(), Value::String("none".into()));
                    m
                })],
            }),
        )
        .await;
        let _ = read_msg(&mut client).await;

        // BEGIN -> TX_READY (the backend takes the writer here).
        write_msg(
            &mut client,
            &pack_request(&Value::Struct {
                tag: crate::value::struct_tag::BEGIN,
                fields: vec![Value::Map(BTreeMap::new())],
            }),
        )
        .await;
        assert!(matches!(
            decode_response(&read_msg(&mut client).await),
            Response::Success(_)
        ));

        // An in-tx RUN that fails must roll the transaction back (releasing the
        // writer) even as the session moves to FAILED.
        write_msg(
            &mut client,
            &pack_request(&Value::Struct {
                tag: crate::value::struct_tag::RUN,
                fields: vec![
                    Value::String("CREATE (n)".into()),
                    Value::Map(BTreeMap::new()),
                    Value::Map(BTreeMap::new()),
                ],
            }),
        )
        .await;
        assert!(matches!(
            decode_response(&read_msg(&mut client).await),
            Response::Failure(_)
        ));
        assert_eq!(
            rollbacks.load(Ordering::SeqCst),
            1,
            "a failed in-tx statement must roll the transaction back to release the writer"
        );

        // RESET still recovers the (already released) session to READY.
        write_msg(
            &mut client,
            &pack_request(&Value::Struct {
                tag: crate::value::struct_tag::RESET,
                fields: vec![],
            }),
        )
        .await;
        assert!(matches!(
            decode_response(&read_msg(&mut client).await),
            Response::Success(_)
        ));

        drop(client);
        let _ = task.await.unwrap();
    }

    #[tokio::test]
    async fn rejects_unsupported_handshake_version() {
        let (mut client, server) = duplex(64);
        let session = fixture_session(server, RunOutcome::default(), AuthPolicy::Open);
        let task = tokio::spawn(async move { session.run().await });

        let mut bytes = Vec::with_capacity(20);
        bytes.extend_from_slice(&MAGIC);
        // Bolt 3 only, not supported.
        bytes.extend_from_slice(&[0, 0, 0, 3]);
        bytes.extend_from_slice(&[0; 12]);
        client.write_all(&bytes).await.unwrap();

        let reply = read_handshake_reply(&mut client).await;
        assert_eq!(reply, [0, 0, 0, 0]);
        drop(client);
        let _ = task.await.unwrap();
    }

    #[test]
    fn node_value_record_test_compiles() {
        // Sanity check that NodeValue can be exported as a test row
        // value. Real coverage lives in `mapping::tests` and in the
        // server-side integration test.
        let _ = NodeValue {
            id: namidb_core::id::NodeId::new(),
            labels: std::iter::once("X".to_string()).collect(),
            properties: BTreeMap::new(),
        };
    }

    /// Test authenticator: accept only when `credentials` matches.
    struct ApiKeyAuth(&'static str);

    #[async_trait]
    impl Authenticator for ApiKeyAuth {
        async fn authenticate(
            &self,
            auth: &BTreeMap<String, Value>,
        ) -> std::result::Result<(), String> {
            match auth.get("credentials") {
                Some(Value::String(s)) if s == self.0 => Ok(()),
                _ => Err("invalid api key".into()),
            }
        }
    }

    /// Drive handshake → HELLO (v5) → LOGON with `creds` under `policy`,
    /// returning the LOGON reply.
    async fn drive_logon(creds: &str, policy: AuthPolicy) -> Response {
        let (mut client, server) = duplex(16 * 1024);
        let session = fixture_session(server, RunOutcome::default(), policy);
        let task = tokio::spawn(async move { session.run().await });

        send_handshake(&mut client).await;
        let _ = read_handshake_reply(&mut client).await;

        // v5 HELLO carries no auth; it just moves to Authentication.
        write_msg(
            &mut client,
            &pack_request(&Value::Struct {
                tag: crate::value::struct_tag::HELLO,
                fields: vec![Value::Map(BTreeMap::new())],
            }),
        )
        .await;
        let _ = read_msg(&mut client).await;

        // LOGON carries the credentials the custom authenticator checks.
        write_msg(
            &mut client,
            &pack_request(&Value::Struct {
                tag: crate::value::struct_tag::LOGON,
                fields: vec![Value::Map({
                    let mut m = BTreeMap::new();
                    m.insert("scheme".into(), Value::String("basic".into()));
                    m.insert("credentials".into(), Value::String(creds.into()));
                    m
                })],
            }),
        )
        .await;
        let reply = decode_response(&read_msg(&mut client).await);

        write_msg(
            &mut client,
            &pack_request(&Value::Struct {
                tag: crate::value::struct_tag::GOODBYE,
                fields: vec![],
            }),
        )
        .await;
        drop(client);
        let _ = task.await.unwrap();
        reply
    }

    #[tokio::test]
    async fn custom_authenticator_accepts_valid_credentials() {
        let policy = AuthPolicy::Custom(Arc::new(ApiKeyAuth("good-key")));
        assert!(matches!(
            drive_logon("good-key", policy).await,
            Response::Success(_)
        ));
    }

    #[tokio::test]
    async fn custom_authenticator_rejects_bad_credentials() {
        let policy = AuthPolicy::Custom(Arc::new(ApiKeyAuth("good-key")));
        match drive_logon("wrong", policy).await {
            Response::Failure(meta) => assert_eq!(
                meta.get("code"),
                Some(&Value::String(
                    "Neo.ClientError.Security.Unauthorized".into()
                )),
            ),
            other => panic!("expected FAILURE, got {other:?}"),
        }
    }
}
