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

/// Auth policy applied to LOGON.
#[derive(Debug, Clone)]
pub enum AuthPolicy {
    /// Accept any LOGON. Mirrors the REST server's "no auth" mode.
    Open,
    /// Accept `basic` or `bearer` schemes whose credentials match
    /// this token (constant-time compare). Anything else fails.
    Token(Arc<str>),
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

    /// Optional override for the manifest version reported as the
    /// bookmark after COMMIT. Default returns `None` and the session
    /// emits no bookmark.
    async fn current_bookmark(&self) -> Option<String> {
        None
    }
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
        }
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
            let body = match read_message(&mut self.socket, max).await {
                Ok(b) => b,
                Err(BoltError::Io(e)) if e.kind() == std::io::ErrorKind::UnexpectedEof => {
                    debug!("bolt connection closed by client");
                    return Ok(());
                }
                Err(e) => return Err(e),
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
                return Err(e);
            }
            if goodbye || self.state == State::Defunct {
                return Ok(());
            }
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
        // RESET always recovers; GOODBYE always ends.
        if matches!(req, Request::Reset) {
            self.state = State::Ready;
            return self.write_response(Response::success_empty()).await;
        }
        if matches!(req, Request::Goodbye) {
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
            if let Err(e) = self.authenticate(&extra) {
                self.state = State::Failed;
                return self
                    .write_failure("Neo.ClientError.Security.Unauthorized", e)
                    .await;
            }
            self.state = State::Ready;
            self.write_response(Response::Success(meta)).await?;
        }
        Ok(())
    }

    async fn handle_in_authentication(&mut self, req: Request) -> Result<()> {
        let Request::Logon(extra) = req else {
            return self.invalid_state("LOGON required").await;
        };
        if let Err(e) = self.authenticate(&extra) {
            self.state = State::Failed;
            return self
                .write_failure("Neo.ClientError.Security.Unauthorized", e)
                .await;
        }
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
            Request::Begin(_) => {
                self.state = State::TxReady;
                self.write_response(Response::success_empty()).await
            }
            Request::Route { .. } => self.respond_route().await,
            Request::Logoff => {
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
            Request::Rollback => {
                self.state = State::Ready;
                self.write_response(Response::success_empty()).await
            }
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
        let _ = params.clone(); // explicit clone to avoid moves later
        let outcome = match self.backend.run(cypher, params).await {
            Ok(o) => o,
            Err(e) => {
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

    fn authenticate(&self, extra: &BTreeMap<String, Value>) -> std::result::Result<(), String> {
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
            label: "X".into(),
            properties: BTreeMap::new(),
        };
    }
}
