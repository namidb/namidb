//! Per-connection Bolt session.
//!
//! Owns the state machine for a single TCP connection: handshake →
//! HELLO → LOGON → (RUN / PULL / DISCARD / BEGIN / COMMIT / ROLLBACK /
//! RESET / GOODBYE)\*. Delegates to a [`Backend`] trait for the actual
//! Cypher execution so the bolt crate stays independent of
//! `namidb-server` and easy to test.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use async_trait::async_trait;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::sync::Notify;
use tracing::{debug, info, warn};

use crate::budget::{MessageMemoryBudget, MessageMemoryLease, CONTROL_FRAME_MAX_BYTES};
use crate::chunk::write_message;
use crate::error::{BoltError, Result};
use crate::handshake::{negotiate, read_offers, write_response, Version};
use crate::mapping::{
    params_from_bolt_map_owned, runtime_to_bolt, runtime_to_bolt_owned, ElementIdMode,
};
use crate::message::{Request, Response, DEFAULT_POST_AUTH_MESSAGE_BYTES, PRE_AUTH_MESSAGE_BYTES};
use crate::state::State;
use crate::value::Value;

use namidb_query::{Params, Row};

/// Cooperative cancellation signal for one in-flight `RUN`.
///
/// The Bolt session flips this when its transport reaches EOF while the
/// backend is still executing. A backend may override
/// [`Backend::run_with_cancellation`] / [`Backend::run_in_tx_with_cancellation`]
/// to stop only at a cancellation-safe boundary, clean up staged mutations,
/// and then return. The session deliberately awaits that cleanup before it
/// tears the connection task down.
#[derive(Clone, Debug, Default)]
pub struct RunCancellation {
    inner: Arc<RunCancellationInner>,
}

#[derive(Debug, Default)]
struct RunCancellationInner {
    cancelled: AtomicBool,
    notify: Notify,
}

impl RunCancellation {
    pub fn new() -> Self {
        Self::default()
    }

    /// Publish cancellation. Idempotent and safe to call before the backend
    /// starts waiting.
    pub fn cancel(&self) {
        if !self.inner.cancelled.swap(true, Ordering::AcqRel) {
            self.inner.notify.notify_waiters();
        }
    }

    pub fn is_cancelled(&self) -> bool {
        self.inner.cancelled.load(Ordering::Acquire)
    }

    /// Wait until cancellation is published without a check/subscribe race.
    pub async fn cancelled(&self) {
        loop {
            if self.is_cancelled() {
                return;
            }
            let notified = self.inner.notify.notified();
            tokio::pin!(notified);
            notified.as_mut().enable();
            if self.is_cancelled() {
                return;
            }
            notified.await;
        }
    }
}

/// Cancellation-safe Bolt chunk decoder.
///
/// `chunk::read_message` intentionally owns its header/body buffers inside one
/// future, which is ideal for the ordinary sequential session loop but cannot
/// be selected against backend execution: cancelling it after a partial
/// socket read would discard bytes already removed from TCP. This decoder
/// retains every partial header/chunk in the `Session`, and uses Tokio's
/// cancellation-safe `read` operation for each increment. If RUN completes
/// halfway through a pipelined PULL frame, the next loop iteration resumes at
/// the exact byte offset.
#[derive(Debug)]
struct BudgetedMessage {
    body: Vec<u8>,
    /// Retained through decode, parameter conversion, backend execution, or
    /// RUN prefetch. Dropping the envelope releases the shared byte budget.
    _lease: Option<MessageMemoryLease>,
}

impl BudgetedMessage {
    #[cfg(test)]
    fn unbudgeted(body: Vec<u8>) -> Self {
        Self { body, _lease: None }
    }

    fn len(&self) -> usize {
        self.body.len()
    }
}

#[derive(Debug, Default)]
struct StatefulMessageReader {
    header: [u8; 2],
    header_read: usize,
    /// A chunk header has been consumed, but its body has not yet been
    /// allocated/read. This explicit state keeps budget acquisition
    /// cancellation-safe inside RUN's `select!`.
    pending_chunk_len: Option<usize>,
    chunk: Vec<u8>,
    chunk_read: usize,
    message: Vec<u8>,
    message_complete: bool,
    /// Fixed deadline from the first byte of a frame. Idle authenticated
    /// connections have no deadline and hold no budget; a partial slowloris
    /// cannot retain its bounded raw-memory lease indefinitely.
    message_deadline: Option<tokio::time::Instant>,
    budget_lease: Option<MessageMemoryLease>,
    /// When a chunk header crosses the configured message ceiling, its body
    /// has not been consumed yet. Retain that exact remainder so an
    /// authenticated session can bounded-time drain through the terminator,
    /// emit a reliable FAILURE (without TCP RST from unread input), and close.
    oversized_chunk_remaining: usize,
}

impl StatefulMessageReader {
    async fn read_message<R: AsyncReadExt + Unpin>(
        &mut self,
        reader: &mut R,
        max_message_bytes: usize,
        budget: Option<&Arc<MessageMemoryBudget>>,
        partial_message_timeout: Option<std::time::Duration>,
    ) -> Result<BudgetedMessage> {
        loop {
            if self.message_complete {
                let control_bypass = self.message.len() <= CONTROL_FRAME_MAX_BYTES
                    && is_pressure_relief_frame(&self.message);
                if !control_bypass {
                    if let Some(budget) = budget {
                        // Upgrade the raw framing charge to the conservative
                        // decoded/runtime working-set charge atomically while
                        // retaining the raw lease. Failure is immediate, so no
                        // task waits while holding partial permits.
                        self.ensure_budget(budget, self.message.len(), true)?;
                    }
                }

                self.message_complete = false;
                self.message_deadline = None;
                return Ok(BudgetedMessage {
                    body: std::mem::take(&mut self.message),
                    _lease: self.budget_lease.take(),
                });
            }

            // An in-progress chunk is the decoder's explicit "body" phase.
            // Do not read a fresh header until its full body has arrived: the
            // future may have been dropped by `select!` after consuming only
            // part of that body while a concurrent RUN completed.
            if self.chunk.is_empty() {
                if self.pending_chunk_len.is_none() {
                    while self.header_read < self.header.len() {
                        let deadline = self.message_deadline;
                        let read = await_message_step(
                            deadline,
                            reader.read(&mut self.header[self.header_read..]),
                        )
                        .await??;
                        if read == 0 {
                            return Err(
                                std::io::Error::from(std::io::ErrorKind::UnexpectedEof).into()
                            );
                        }
                        if self.message_deadline.is_none() {
                            self.message_deadline = partial_message_timeout
                                .map(|timeout| tokio::time::Instant::now() + timeout);
                        }
                        self.header_read += read;
                    }

                    let chunk_len = u16::from_be_bytes(self.header) as usize;
                    self.header_read = 0;
                    self.pending_chunk_len = Some(chunk_len);
                }

                let chunk_len = self
                    .pending_chunk_len
                    .expect("chunk length is retained across cancellation");
                if chunk_len == 0 {
                    self.pending_chunk_len = None;
                    self.chunk_read = 0;
                    self.message_complete = true;
                    continue;
                }

                let next_len =
                    self.message
                        .len()
                        .checked_add(chunk_len)
                        .ok_or(BoltError::TooLarge {
                            what: "Bolt message",
                            len: usize::MAX,
                            max: max_message_bytes,
                        })?;
                if next_len > max_message_bytes {
                    self.pending_chunk_len = None;
                    self.oversized_chunk_remaining = chunk_len;
                    return Err(BoltError::TooLarge {
                        what: "Bolt message",
                        len: next_len,
                        max: max_message_bytes,
                    });
                }

                if next_len > CONTROL_FRAME_MAX_BYTES {
                    if let Some(budget) = budget {
                        if let Err(error) = self.ensure_budget(budget, next_len, false) {
                            // The chunk header is already off the transport but
                            // its body is untouched. Reuse the ordinary
                            // oversized-frame drain path before closing.
                            self.pending_chunk_len = None;
                            self.oversized_chunk_remaining = chunk_len;
                            return Err(error);
                        }
                    }
                }

                self.pending_chunk_len = None;
                self.chunk.resize(chunk_len, 0);
                self.chunk_read = 0;
            }

            while self.chunk_read < self.chunk.len() {
                let read = await_message_step(
                    self.message_deadline,
                    reader.read(&mut self.chunk[self.chunk_read..]),
                )
                .await??;
                if read == 0 {
                    return Err(std::io::Error::from(std::io::ErrorKind::UnexpectedEof).into());
                }
                self.chunk_read += read;
            }
            self.message.extend_from_slice(&self.chunk);
            self.chunk.clear();
            self.chunk_read = 0;
        }
    }

    fn ensure_budget(
        &mut self,
        budget: &Arc<MessageMemoryBudget>,
        wire_bytes: usize,
        decoded_working_set: bool,
    ) -> Result<()> {
        let desired = if decoded_working_set {
            budget.decoded_units_for_wire(wire_bytes)?
        } else {
            budget.framing_units_for_wire(wire_bytes)?
        };
        let current = self
            .budget_lease
            .as_ref()
            .map(MessageMemoryLease::units)
            .unwrap_or(0);
        if desired <= current {
            return Ok(());
        }
        let additional = budget.try_acquire_units(desired - current)?;
        match &mut self.budget_lease {
            Some(lease) => lease.merge(additional),
            None => self.budget_lease = Some(MessageMemoryLease::new(additional)),
        }
        Ok(())
    }

    /// Release every retained allocation/permit while preserving the framing
    /// cursor needed by [`Self::discard_oversized_message`].
    fn release_buffered_allocations(&mut self) {
        drop(std::mem::take(&mut self.message));
        drop(std::mem::take(&mut self.chunk));
        self.chunk_read = 0;
        self.budget_lease = None;
    }

    /// Discard the rest of a frame whose next chunk crossed the size limit.
    ///
    /// Bolt chunks are at most 65,535 bytes, so this uses fixed scratch memory
    /// no matter how large the rejected message is. Callers impose a wall
    /// clock timeout, preventing an authenticated slowloris from pinning the
    /// connection merely to obtain a diagnostic.
    async fn discard_oversized_message<R: AsyncReadExt + Unpin>(
        &mut self,
        reader: &mut R,
    ) -> Result<()> {
        // A sub-4-KiB data frame is buffered before its tag can be classified.
        // If its completed body cannot fit the shared budget, the terminating
        // zero chunk is already consumed. Do not mistake the next frame for a
        // remainder and drain it.
        let already_complete = std::mem::take(&mut self.message_complete);
        let mut scratch = [0_u8; 8 * 1024];
        // `clear()` would retain the rejected frame's potentially very large
        // allocation until the connection task finally exits. Release it
        // before spending up to the drain timeout reading the rest of the
        // frame.
        self.release_buffered_allocations();
        if already_complete {
            self.header_read = 0;
            self.pending_chunk_len = None;
            self.oversized_chunk_remaining = 0;
            self.message_deadline = None;
            return Ok(());
        }

        let mut remaining = std::mem::take(&mut self.oversized_chunk_remaining);
        loop {
            while remaining > 0 {
                let take = remaining.min(scratch.len());
                reader.read_exact(&mut scratch[..take]).await?;
                remaining -= take;
            }

            let mut header = [0_u8; 2];
            reader.read_exact(&mut header).await?;
            let chunk_len = u16::from_be_bytes(header) as usize;
            if chunk_len == 0 {
                self.header_read = 0;
                self.pending_chunk_len = None;
                self.message_complete = false;
                self.message_deadline = None;
                return Ok(());
            }
            remaining = chunk_len;
        }
    }
}

async fn await_message_step<F, T>(deadline: Option<tokio::time::Instant>, future: F) -> Result<T>
where
    F: std::future::Future<Output = T>,
{
    match deadline {
        Some(deadline) => tokio::time::timeout_at(deadline, future)
            .await
            .map_err(|_| {
                std::io::Error::new(
                    std::io::ErrorKind::TimedOut,
                    "authenticated Bolt frame exceeded its partial-message deadline",
                )
            })
            .map_err(BoltError::Io),
        None => Ok(future.await),
    }
}

fn raw_request_tag(body: &[u8]) -> Option<u8> {
    match body {
        [marker @ 0xB0..=0xBF, tag, ..] => {
            let _fields = marker & 0x0F;
            Some(*tag)
        }
        [0xDC, _fields, tag, ..] => Some(*tag),
        [0xDD, _fields_hi, _fields_lo, tag, ..] => Some(*tag),
        _ => None,
    }
}

fn is_pressure_relief_frame(body: &[u8]) -> bool {
    matches!(
        raw_request_tag(body),
        Some(
            crate::value::struct_tag::PULL
                | crate::value::struct_tag::DISCARD
                | crate::value::struct_tag::COMMIT
                | crate::value::struct_tag::ROLLBACK
                | crate::value::struct_tag::RESET
                | crate::value::struct_tag::GOODBYE
                | crate::value::struct_tag::LOGOFF
        )
    )
}

fn requires_decode_admission(body: &[u8]) -> bool {
    body.len() > CONTROL_FRAME_MAX_BYTES || !is_pressure_relief_frame(body)
}

fn is_frame_memory_rejection(error: &BoltError) -> bool {
    matches!(
        error,
        BoltError::TooLarge { .. }
            | BoltError::DecodedTooLarge { .. }
            | BoltError::MemoryBudgetExhausted { .. }
    )
}

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
///
/// Retry semantics are encoded in the code's second segment, which official
/// Neo4j drivers classify on: `Neo.TransientError.*` is auto-retried,
/// `Neo.ClientError.*` is not. Deterministic limits (timeout, row cap) are
/// therefore ClientError on purpose — a driver auto-retrying a query that
/// will exceed the same budget again is a retry livelock.
#[derive(Debug)]
#[non_exhaustive]
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
    /// The query ran past its configured wall-clock budget. A re-run would
    /// time out again, so this is deliberately NOT a transient error.
    Timeout(String),
    /// The query exceeded a configured deterministic result limit (row cap,
    /// search result caps). Also deliberately not transient.
    ResourceLimit(String),
    /// An administrator cancelled this execution. Transient by Neo4j
    /// convention (Transaction.Terminated): the STATEMENT may be retried —
    /// whether it should be is the operator's conversation to have.
    Cancelled(String),
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
            BackendError::Timeout(_) => "Neo.ClientError.Transaction.TransactionTimedOut",
            BackendError::Cancelled(_) => "Neo.TransientError.Transaction.Terminated",
            BackendError::ResourceLimit(_) => "Neo.ClientError.Statement.ResourceLimitExceeded",
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
            | BackendError::Timeout(s)
            | BackendError::ResourceLimit(s)
            | BackendError::Cancelled(s)
            | BackendError::Other(s) => s,
        }
    }
}

/// Pluggable Cypher executor. The server crate implements this on top
/// of `WriterSession`; tests implement it with hand-canned rows.
pub trait DecodeAdmissionGuard: Send {}

impl<T: Send> DecodeAdmissionGuard for T {}

#[async_trait]
pub trait Backend: Send + Sync {
    /// Admit a potentially allocation-heavy request before PackStream decode
    /// and parameter conversion. After authentication the session invokes
    /// this for every frame except bounded pressure-relief controls (small
    /// `PULL`/`DISCARD`/`COMMIT`/`ROLLBACK`/`RESET`/`GOODBYE`/`LOGOFF`).
    /// The default is a no-op for embedders without a process RSS governor.
    ///
    /// A production backend may return an opaque RAII reservation, which the
    /// session retains through handling. It should still keep ordinary
    /// execution-time admission to close races with non-Bolt allocations.
    async fn admit_request_decode(
        &self,
        _wire_bytes: usize,
    ) -> std::result::Result<Option<Box<dyn DecodeAdmissionGuard>>, BackendError> {
        Ok(None)
    }

    /// Execute one Cypher statement in auto-commit mode.
    async fn run(
        &self,
        cypher: &str,
        params: Params,
    ) -> std::result::Result<RunOutcome, BackendError>;

    /// Execute an auto-commit statement while observing transport
    /// cancellation. The default preserves the legacy behaviour and lets the
    /// statement finish: blindly dropping an arbitrary backend future could
    /// interrupt a durability commit. Backends that can distinguish their
    /// cancel-safe apply phase from their non-cancellable commit phase should
    /// override this method.
    async fn run_with_cancellation(
        &self,
        cypher: &str,
        params: Params,
        _cancellation: RunCancellation,
    ) -> std::result::Result<RunOutcome, BackendError> {
        self.run(cypher, params).await
    }

    /// Auto-commit RUN carrying the Bolt `db` routing field from the
    /// message's `extra` map (`Some("acme")` when the driver session was
    /// opened with `database="acme"`; `None` otherwise). The default ignores
    /// the database and delegates, so single-namespace embedders and test
    /// backends keep working unchanged; a multi-tenant backend overrides
    /// this to route the statement to the named namespace.
    async fn run_with_cancellation_on(
        &self,
        db: Option<&str>,
        cypher: &str,
        params: Params,
        cancellation: RunCancellation,
    ) -> std::result::Result<RunOutcome, BackendError> {
        let _ = db;
        self.run_with_cancellation(cypher, params, cancellation)
            .await
    }

    /// Begin an explicit transaction. Subsequent [`Backend::run_in_tx`]
    /// calls stage into it; [`Backend::commit_tx`] makes them durable and
    /// [`Backend::rollback_tx`] discards them. The default is a no-op so a
    /// backend without transaction support keeps working (its in-tx writes
    /// just behave like auto-commit).
    async fn begin_tx(&self) -> std::result::Result<(), BackendError> {
        Ok(())
    }

    /// BEGIN carrying the Bolt `db` routing field from the message's
    /// `extra` map. The transaction is pinned to that database for its
    /// whole lifetime (statements inside it carry no `db` of their own).
    /// The default ignores the database and delegates to
    /// [`Backend::begin_tx`].
    async fn begin_tx_on(&self, db: Option<&str>) -> std::result::Result<(), BackendError> {
        let _ = db;
        self.begin_tx().await
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

    /// Cancellation-aware explicit-transaction statement. As with
    /// [`Backend::run_with_cancellation`], the conservative default finishes
    /// the call. A supporting backend may abort the current apply safely; the
    /// session will subsequently invoke `rollback_tx` when the disconnected
    /// connection unwinds.
    async fn run_in_tx_with_cancellation(
        &self,
        cypher: &str,
        params: Params,
        _cancellation: RunCancellation,
    ) -> std::result::Result<RunOutcome, BackendError> {
        self.run_in_tx(cypher, params).await
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

/// The Bolt `db` routing field of a RUN/BEGIN `extra` map. An absent key or
/// an empty string both mean "the default database" (drivers send `""` for
/// an unset session database on some protocol versions) — normalised to
/// `None`.
fn db_from_extra(extra: &BTreeMap<String, Value>) -> Option<String> {
    match extra.get("db") {
        Some(Value::String(s)) if !s.is_empty() => Some(s.clone()),
        _ => None,
    }
}

/// Bound valid Bolt pipelining while a long RUN is in flight. A normal driver
/// pipelines at most RUN + PULL/DISCARD; the higher ceiling leaves room for
/// RESET/telemetry without allowing an executing query to become an
/// unbounded per-connection message queue.
const MAX_PREFETCHED_MESSAGES: usize = 16;
const OVERSIZED_MESSAGE_DRAIN_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(5);

/// One Bolt connection. Created once per `accept()` and driven to
/// completion in a single task.
pub struct Session<S: AsyncReadExt + AsyncWriteExt + Unpin> {
    socket: S,
    message_reader: StatefulMessageReader,
    info: ServerInfo,
    auth: AuthPolicy,
    backend: Arc<dyn Backend>,
    state: State,
    version: Option<Version>,
    /// `statement_type` of the in-flight stream, surfaced in the
    /// closing `SUCCESS` after PULL/DISCARD. `None` while no stream
    /// is active.
    pending_statement_type: Option<StatementType>,
    /// Projection order for runtime rows buffered by the last RUN.
    pending_fields: Vec<String>,
    /// Parallel to `pending_fields`: true only for the final occurrence of a
    /// name, where consuming the row binding cannot affect a later column.
    pending_field_is_last_occurrence: Vec<bool>,
    /// Runtime rows moved from the last RUN and drained lazily by PULL in
    /// client-demanded batches (`n` per PULL). DISCARD drops rows without
    /// expanding their values to Bolt containers. Cleared on RESET.
    pending_rows: std::collections::VecDeque<Row>,
    /// Write counters of the in-flight stream, emitted as `stats` in the
    /// closing `SUCCESS` after PULL/DISCARD. Empty for reads.
    pending_counters: BTreeMap<String, i64>,
    /// Messages pipelined by the client while a RUN is executing. Reading the
    /// transport concurrently with the backend is what lets us notice EOF and
    /// cancel a runaway statement; valid early PULL/DISCARD messages are kept
    /// in order and consumed by the normal state machine afterwards.
    prefetched_messages: std::collections::VecDeque<BudgetedMessage>,
    prefetched_bytes: usize,
    /// Maximum body size for one authenticated Bolt message. The strict
    /// pre-authentication cap remains [`PRE_AUTH_MESSAGE_BYTES`] regardless
    /// of this setting.
    post_auth_message_bytes: usize,
    /// Process-wide authenticated-message working-set budget. `None` preserves
    /// standalone library compatibility; the production server always injects
    /// one shared instance across every accepted connection.
    message_memory_budget: Option<Arc<MessageMemoryBudget>>,
    /// Deadline for a frame once its first byte arrives. It does not close a
    /// completely idle authenticated socket, which owns no budget.
    partial_message_timeout: Option<std::time::Duration>,
    /// While an explicit transaction is open the backend holds the writer
    /// lock, so an idle client would pin it indefinitely. When set, a read
    /// that blocks longer than this with a transaction open rolls the
    /// transaction back (releasing the writer) and fails it. `None` (the
    /// default) keeps the legacy unbounded behaviour for test backends.
    tx_idle_timeout: Option<std::time::Duration>,
    /// Bounds how long a not-yet-authenticated connection may block on a read
    /// (the version handshake and every pre-auth message). A slowloris client
    /// that opens a socket and sends nothing (or one byte) would otherwise pin
    /// a spawned task and a file descriptor forever. `None` (the default)
    /// disables it for test backends.
    handshake_timeout: Option<std::time::Duration>,
    /// Maximum total lifetime of an open transaction. The idle timeout only
    /// bounds the gap between messages, so a client that sends a trivial message
    /// just under the idle window can pin the writer forever; this caps the
    /// whole transaction. `None` disables it. Checked at message boundaries.
    max_tx_lifetime: Option<std::time::Duration>,
    /// When the currently-open transaction first blocked on a read. Reset to
    /// `None` whenever no transaction is open. Used to enforce `max_tx_lifetime`.
    tx_started: Option<tokio::time::Instant>,
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
            message_reader: StatefulMessageReader::default(),
            info,
            auth,
            backend,
            state: State::Negotiation,
            version: None,
            pending_statement_type: None,
            pending_counters: BTreeMap::new(),
            pending_fields: Vec::new(),
            pending_field_is_last_occurrence: Vec::new(),
            pending_rows: std::collections::VecDeque::new(),
            prefetched_messages: std::collections::VecDeque::new(),
            prefetched_bytes: 0,
            post_auth_message_bytes: DEFAULT_POST_AUTH_MESSAGE_BYTES,
            message_memory_budget: None,
            partial_message_timeout: None,
            tx_idle_timeout: None,
            handshake_timeout: None,
            max_tx_lifetime: None,
            tx_started: None,
            authenticated: false,
        }
    }

    /// Set the idle timeout applied while an explicit transaction is open.
    /// `None` disables it (a transaction may stay open indefinitely).
    pub fn with_tx_idle_timeout(mut self, timeout: Option<std::time::Duration>) -> Self {
        self.tx_idle_timeout = timeout;
        self
    }

    /// Set the read timeout applied before authentication (handshake + pre-auth
    /// messages). `None` disables it. Bounds slowloris connections that never
    /// complete the handshake.
    pub fn with_handshake_timeout(mut self, timeout: Option<std::time::Duration>) -> Self {
        self.handshake_timeout = timeout;
        self
    }

    /// Set the maximum total lifetime of an open transaction. `None` disables
    /// it (a transaction may stay open indefinitely as long as it stays under
    /// the idle timeout).
    pub fn with_max_tx_lifetime(mut self, lifetime: Option<std::time::Duration>) -> Self {
        self.max_tx_lifetime = lifetime;
        self
    }

    /// Set the maximum body size for one authenticated Bolt message.
    ///
    /// This does not relax the fixed 64 KiB pre-authentication ceiling. The
    /// server validates that the configured value is non-zero before creating
    /// sessions; the defensive clamp keeps direct library callers from
    /// accidentally constructing a session that rejects every message.
    pub fn with_post_auth_message_bytes(mut self, max_bytes: usize) -> Self {
        self.post_auth_message_bytes = max_bytes.max(1);
        self
    }

    /// Attach the process-wide authenticated Bolt message working-set budget.
    ///
    /// The lease is acquired by the framer, before it grows a data message
    /// beyond the bounded control prefix, and follows complete messages through
    /// decode, RUN execution, and prefetch.
    pub fn with_message_memory_budget(mut self, budget: Arc<MessageMemoryBudget>) -> Self {
        self.message_memory_budget = Some(budget);
        self
    }

    /// Bound a partially-sent authenticated frame without timing out a fully
    /// idle connection. This prevents a slowloris from retaining the shared
    /// raw-message byte permits indefinitely.
    pub fn with_partial_message_timeout(mut self, timeout: Option<std::time::Duration>) -> Self {
        self.partial_message_timeout = timeout.filter(|timeout| !timeout.is_zero());
        self
    }

    fn current_message_limit(&self) -> usize {
        if self.authenticated {
            self.post_auth_message_bytes
        } else {
            PRE_AUTH_MESSAGE_BYTES
        }
    }

    /// Remaining aggregate body budget while RUN executes.
    ///
    /// The stateful reader may already hold a partial frame in addition to
    /// complete queued frames, so each read must be bounded by the bytes left,
    /// not by the full per-message ceiling again.
    fn prefetch_read_limit(&self) -> usize {
        if self.prefetched_messages.len() >= MAX_PREFETCHED_MESSAGES {
            0
        } else {
            self.post_auth_message_bytes
                .saturating_sub(self.prefetched_bytes)
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
            let max = self.current_message_limit();
            // Bound how long the writer lock is pinned by an open
            // transaction: if the client idles past `tx_idle_timeout` while
            // in a transaction, roll it back to release the writer and fail
            // the transaction.
            let in_tx = matches!(self.state, State::TxReady | State::TxStreaming);
            // Enforce total transaction lifetime at message boundaries: a client
            // that stays just under the idle timeout could otherwise hold the
            // writer forever. Roll back and fail once the cap is exceeded.
            if in_tx {
                let started = *self
                    .tx_started
                    .get_or_insert_with(tokio::time::Instant::now);
                if let Some(max) = self.max_tx_lifetime {
                    if started.elapsed() >= max {
                        self.fail_request(
                            "Neo.TransientError.Transaction.LockClientStopped",
                            "transaction exceeded maximum lifetime; rolled back to release \
                             the writer"
                                .to_string(),
                        )
                        .await?;
                        continue;
                    }
                }
            } else {
                self.tx_started = None;
            }
            let pre_auth = !self.authenticated;
            let read_result = if let Some(body) = self.prefetched_messages.pop_front() {
                self.prefetched_bytes = self.prefetched_bytes.saturating_sub(body.len());
                Ok(body)
            } else {
                let message_budget = if pre_auth {
                    None
                } else {
                    self.message_memory_budget.as_ref()
                };
                let partial_timeout = (!pre_auth)
                    .then_some(self.partial_message_timeout)
                    .flatten();
                let read = self.message_reader.read_message(
                    &mut self.socket,
                    max,
                    message_budget,
                    partial_timeout,
                );
                if in_tx {
                    match self.tx_idle_timeout {
                        Some(t) => match tokio::time::timeout(t, read).await {
                            Ok(r) => r,
                            Err(_elapsed) => {
                                self.fail_request(
                                    "Neo.TransientError.Transaction.LockClientStopped",
                                    "transaction idle timeout; rolled back to release the writer"
                                        .to_string(),
                                )
                                .await?;
                                continue;
                            }
                        },
                        None => read.await,
                    }
                } else if pre_auth {
                    // Drop a not-yet-authenticated client that stalls a read past
                    // the handshake timeout (slowloris): it never sends HELLO/LOGON
                    // and would otherwise pin this task + FD indefinitely.
                    match self.handshake_timeout {
                        Some(t) => match tokio::time::timeout(t, read).await {
                            Ok(r) => r,
                            Err(_elapsed) => {
                                debug!("bolt pre-auth read timed out; closing idle connection");
                                return Ok(());
                            }
                        },
                        None => read.await,
                    }
                } else {
                    read.await
                }
            };
            let message = match read_result {
                Ok(b) => b,
                Err(BoltError::Io(e)) if e.kind() == std::io::ErrorKind::UnexpectedEof => {
                    debug!("bolt connection closed by client");
                    self.message_reader.release_buffered_allocations();
                    // A client that drops mid-transaction (crash, kill, network
                    // partition) never sends ROLLBACK/GOODBYE; without this the
                    // staged batch would linger in the shared writer and be
                    // sealed by the next unrelated commit. Roll it back here.
                    self.rollback_if_open_tx().await;
                    return Ok(());
                }
                Err(e) if is_frame_memory_rejection(&e) => {
                    // Rollback/recovery may wait on the single writer. The
                    // framing cursor is enough to drain later; return the
                    // rejected body's memory permits immediately.
                    self.message_reader.release_buffered_allocations();
                    self.rollback_if_open_tx().await;
                    // The rejected message has not been drained, so this
                    // connection cannot safely continue at the next Bolt
                    // frame. After authentication, however, the transport is
                    // full-duplex: emit one explicit FAILURE and flush it
                    // before closing instead of making the driver diagnose an
                    // unexplained EOF/defunct connection.
                    self.write_oversized_message_failure(&e, true).await;
                    return Err(e);
                }
                Err(e) => {
                    self.message_reader.release_buffered_allocations();
                    self.rollback_if_open_tx().await;
                    return Err(e);
                }
            };
            let BudgetedMessage {
                body,
                _lease: message_lease,
            } = message;
            let decode_admission_guard = if self.authenticated && requires_decode_admission(&body) {
                match self.backend.admit_request_decode(body.len()).await {
                    Ok(guard) => guard,
                    Err(error) => {
                        // Do not retain the raw body/lease while rollback
                        // and the pressure diagnostic run. RESET remains
                        // budget-bypassed and can recover FAILED.
                        drop(body);
                        drop(message_lease);
                        self.fail_request(error.code(), error.message().to_string())
                            .await?;
                        continue;
                    }
                }
            } else {
                None
            };
            let decoded = Request::decode(&body, max);
            // Request now owns every decoded allocation. Drop the contiguous
            // wire copy before parameter conversion/query execution.
            drop(body);
            let request = match decoded {
                Ok(r) => r,
                Err(e) => {
                    drop(message_lease);
                    drop(decode_admission_guard);
                    self.fail_request(
                        "Neo.ClientError.Request.Invalid",
                        format!("malformed Bolt message: {e}"),
                    )
                    .await?;
                    continue;
                }
            };
            debug!(name = request.name(), state = ?self.state, "bolt request");
            let goodbye = matches!(request, Request::Goodbye);
            if let Err(e) = self.handle(request, element_mode).await {
                warn!(error = %e, "bolt session error");
                drop(message_lease);
                drop(decode_admission_guard);
                // A transport/protocol error tears the connection down; roll
                // back any open transaction so its staged writes are discarded
                // rather than left in the shared writer for the next commit.
                self.rollback_if_open_tx().await;
                return Err(e);
            }
            // Keep the shared message charge through parameter conversion and
            // backend execution; only a fully-consumed request releases it.
            drop(message_lease);
            drop(decode_admission_guard);
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
        // Bound the 20-byte version handshake read: a client that connects and
        // sends nothing must not pin this task forever (slowloris).
        let offers = match self.handshake_timeout {
            Some(t) => match tokio::time::timeout(t, read_offers(&mut self.socket)).await {
                Ok(r) => r?,
                Err(_elapsed) => {
                    debug!("bolt handshake read timed out; closing idle connection");
                    self.state = State::Defunct;
                    return Ok(());
                }
            },
            None => read_offers(&mut self.socket).await?,
        };
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
            drop(std::mem::take(&mut self.pending_rows));
            drop(std::mem::take(&mut self.pending_fields));
            drop(std::mem::take(&mut self.pending_field_is_last_occurrence));
            self.pending_statement_type = None;
            self.pending_counters.clear();
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
                return self
                    .fail_request("Neo.ClientError.Security.Unauthorized", e)
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
            return self
                .fail_request("Neo.ClientError.Security.Unauthorized", e)
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
                extra,
            } => {
                let db = db_from_extra(&extra);
                self.execute_run(&cypher, params, element_mode, false, db)
                    .await
            }
            Request::Begin(extra) => {
                let db = db_from_extra(&extra);
                match self.backend.begin_tx_on(db.as_deref()).await {
                    Ok(()) => {
                        self.state = State::TxReady;
                        self.write_response(Response::success_empty()).await
                    }
                    Err(e) => self.fail_request(e.code(), e.message().to_string()).await,
                }
            }
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
            // In-tx statements carry no `db` of their own — the transaction
            // was pinned to its database at BEGIN.
            Request::Run {
                cypher,
                params,
                extra: _,
            } => {
                self.execute_run(&cypher, params, element_mode, true, None)
                    .await
            }
            Request::Commit => self.commit(element_mode).await,
            Request::Rollback => match self.backend.rollback_tx().await {
                Ok(()) => {
                    self.state = State::Ready;
                    self.write_response(Response::success_empty()).await
                }
                Err(e) => self.fail_request(e.code(), e.message().to_string()).await,
            },
            _ => self.invalid_state("unexpected message in TX_READY").await,
        }
    }

    async fn handle_in_streaming(
        &mut self,
        req: Request,
        element_mode: ElementIdMode,
    ) -> Result<()> {
        match req {
            Request::Pull { ref extra } | Request::Discard { ref extra } => {
                // `n` is the batch size the client is ready for; -1 (or a
                // missing key, which real drivers never send) means "all
                // remaining". PULL emits that many RECORDs, DISCARD drops
                // them unsent; a non-empty remainder answers
                // SUCCESS { has_more: true } and stays STREAMING so the next
                // PULL continues — this is what makes a driver's fetch_size
                // actually page a large result.
                let is_pull = matches!(req, Request::Pull { .. });
                let n = match extra.get("n") {
                    Some(Value::Int(i)) => *i,
                    _ => -1,
                };
                let take = if n < 0 {
                    self.pending_rows.len()
                } else {
                    (n as usize).min(self.pending_rows.len())
                };
                for _ in 0..take {
                    let mut row = self
                        .pending_rows
                        .pop_front()
                        .expect("take is bounded by len");
                    if is_pull {
                        let values = self
                            .pending_fields
                            .iter()
                            .zip(&self.pending_field_is_last_occurrence)
                            .map(|(name, is_last_occurrence)| {
                                if *is_last_occurrence {
                                    row.bindings
                                        .remove(name)
                                        .map(|value| runtime_to_bolt_owned(value, element_mode))
                                        .unwrap_or(Value::Null)
                                } else {
                                    row.bindings
                                        .get(name)
                                        .map(|value| runtime_to_bolt(value, element_mode))
                                        .unwrap_or(Value::Null)
                                }
                            })
                            .collect();
                        self.write_response(Response::Record(values)).await?;
                    }
                }
                if !self.pending_rows.is_empty() {
                    let mut meta = BTreeMap::new();
                    meta.insert("has_more".into(), Value::Bool(true));
                    return self.write_response(Response::Success(meta)).await;
                }
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
                // Popping every row leaves VecDeque's peak allocation behind.
                // Release it and the projection fields when the stream closes.
                drop(std::mem::take(&mut self.pending_rows));
                drop(std::mem::take(&mut self.pending_fields));
                drop(std::mem::take(&mut self.pending_field_is_last_occurrence));
                self.write_response(Response::Success(meta)).await
            }
            _ => {
                self.invalid_state("only PULL/DISCARD valid in STREAMING")
                    .await
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    async fn execute_run(
        &mut self,
        cypher: &str,
        bolt_params: BTreeMap<String, Value>,
        _element_mode: ElementIdMode,
        inside_tx: bool,
        db: Option<String>,
    ) -> Result<()> {
        let params = params_from_bolt_map_owned(bolt_params);
        // Inside an explicit transaction the statement stages into the open
        // tx (committed at COMMIT, discarded at ROLLBACK); a bare RUN
        // auto-commits.
        //
        // While it executes, keep reading complete framed messages. This is
        // necessary to observe TCP/TLS EOF: a single-task session otherwise
        // cannot discover that the client disappeared until RUN returns.
        // Legitimate pipelined PULL/DISCARD messages are replayed through the
        // normal state machine afterwards.
        let cancellation = RunCancellation::new();
        let backend = Arc::clone(&self.backend);
        let run = if inside_tx {
            Box::pin(backend.run_in_tx_with_cancellation(cypher, params, cancellation.clone()))
        } else {
            Box::pin(backend.run_with_cancellation_on(
                db.as_deref(),
                cypher,
                params,
                cancellation.clone(),
            ))
        };
        tokio::pin!(run);
        let run_result = loop {
            let prefetch_read_limit = self.prefetch_read_limit();
            tokio::select! {
                // Prefer a ready EOF over a simultaneously-ready backend so a
                // closed client never receives (or appears to receive) an ACK.
                biased;
                read = self
                    .message_reader
                    .read_message(
                        &mut self.socket,
                        prefetch_read_limit,
                        self.message_memory_budget.as_ref(),
                        self.partial_message_timeout,
                    ) => {
                    match read {
                        Ok(body) => {
                            let next_bytes = self.prefetched_bytes.saturating_add(body.len());
                            if self.prefetched_messages.len() >= MAX_PREFETCHED_MESSAGES
                                || next_bytes > self.post_auth_message_bytes
                            {
                                cancellation.cancel();
                                // This connection is closing. Release the body
                                // that crossed the aggregate cap and all
                                // already-prefetched leases before awaiting a
                                // potentially non-cancellable durability tail.
                                drop(body);
                                self.prefetched_messages.clear();
                                self.prefetched_bytes = 0;
                                // The backend owns cleanup semantics. Await it
                                // before unwinding so staged mutations and the
                                // single-writer guard cannot escape this
                                // connection task.
                                let _ = run.as_mut().await;
                                let error = BoltError::TooLarge {
                                    what: "pipelined messages during RUN",
                                    len: next_bytes,
                                    max: self.post_auth_message_bytes,
                                };
                                self.write_oversized_message_failure(&error, false).await;
                                return Err(error);
                            }
                            self.prefetched_bytes = next_bytes;
                            self.prefetched_messages.push_back(body);
                        }
                        Err(error) => {
                            cancellation.cancel();
                            self.prefetched_messages.clear();
                            self.prefetched_bytes = 0;
                            self.message_reader.release_buffered_allocations();
                            // Never drop the backend future at an arbitrary
                            // durability boundary. A cancellation-aware backend
                            // stops its apply phase, rolls staged state back,
                            // or completes an already-started commit, then
                            // returns and releases the writer.
                            let _ = run.as_mut().await;
                            if is_frame_memory_rejection(&error) {
                                self.write_oversized_message_failure(&error, true).await;
                            }
                            return Err(error);
                        }
                    }
                }
                result = run.as_mut() => break result,
            }
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
                return self.fail_request(e.code(), e.message().to_string()).await;
            }
        };

        // 1) SUCCESS { fields } announcing the field list. RECORDs are NOT
        //    emitted here: the Bolt contract is demand-driven — the client
        //    asks for batches with PULL {n}, and a driver's fetch_size must
        //    actually bound what is in flight. Runtime rows are moved into the
        //    session unchanged and converted only when PULL demands them.
        let RunOutcome {
            fields,
            rows,
            statement_type,
            counters,
        } = outcome;
        let mut head_meta = BTreeMap::new();
        head_meta.insert(
            "fields".into(),
            Value::List(fields.iter().cloned().map(Value::String).collect()),
        );
        head_meta.insert("t_first".into(), Value::Int(0));
        self.write_response(Response::Success(head_meta)).await?;

        let field_is_last_occurrence = {
            let mut last_occurrence = BTreeMap::new();
            for (index, name) in fields.iter().enumerate() {
                last_occurrence.insert(name.as_str(), index);
            }
            fields
                .iter()
                .enumerate()
                .map(|(index, name)| last_occurrence.get(name.as_str()).copied() == Some(index))
                .collect()
        };
        self.pending_fields = fields;
        self.pending_field_is_last_occurrence = field_is_last_occurrence;
        self.pending_rows = rows.into();

        // 2) Transition to STREAMING; PULL/DISCARD drain or drop the buffer.
        self.state = if inside_tx {
            State::TxStreaming
        } else {
            State::Streaming
        };
        self.pending_statement_type = Some(statement_type);
        self.pending_counters = counters;
        Ok(())
    }

    async fn commit(&mut self, _element_mode: ElementIdMode) -> Result<()> {
        // Make the transaction's staged statements durable. A failure here
        // (e.g. a lost manifest CAS) is the abort surface; surface it as a
        // FAILURE and the client retries.
        if let Err(e) = self.backend.commit_tx().await {
            return self.fail_request(e.code(), e.message().to_string()).await;
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
        let prior_state = self.state;
        self.fail_request(
            "Neo.ClientError.Request.Invalid",
            format!("invalid request in state {prior_state:?}: {why}"),
        )
        .await
    }

    /// Enter FAILED without ever leaving an explicit transaction's writer
    /// pinned behind disabled idle/lifetime checks.
    async fn fail_request(&mut self, code: &str, message: String) -> Result<()> {
        let was_in_tx = matches!(self.state, State::TxReady | State::TxStreaming);

        // Release potentially corpus-sized result memory before waiting on
        // rollback or socket I/O.
        drop(std::mem::take(&mut self.pending_rows));
        drop(std::mem::take(&mut self.pending_fields));
        drop(std::mem::take(&mut self.pending_field_is_last_occurrence));
        self.pending_statement_type = None;
        self.pending_counters.clear();

        if was_in_tx {
            let _ = self.backend.rollback_tx().await;
        }
        self.tx_started = None;
        self.state = State::Failed;
        self.write_failure(code, message).await
    }

    /// Send a client-visible diagnostic before closing an authenticated
    /// connection whose current frame cannot be drained safely.
    ///
    /// The pre-auth path deliberately remains silent: it is both
    /// unauthenticated and subject to the separate fixed 64 KiB defense.
    async fn write_oversized_message_failure(
        &mut self,
        error: &BoltError,
        drain_current_frame: bool,
    ) {
        if !self.authenticated {
            return;
        }
        if drain_current_frame {
            match tokio::time::timeout(
                OVERSIZED_MESSAGE_DRAIN_TIMEOUT,
                self.message_reader
                    .discard_oversized_message(&mut self.socket),
            )
            .await
            {
                Ok(Ok(())) => {}
                Ok(Err(drain_error)) => {
                    warn!(
                        error = %drain_error,
                        "could not drain oversized Bolt message before diagnostic"
                    );
                    return;
                }
                Err(_) => {
                    warn!(
                        timeout_ms = OVERSIZED_MESSAGE_DRAIN_TIMEOUT.as_millis(),
                        "timed out draining oversized Bolt message before diagnostic"
                    );
                    return;
                }
            }
        }
        let (code, message) = match error {
            BoltError::MemoryBudgetExhausted { .. } => (
                "Neo.TransientError.General.DatabaseUnavailable",
                format!(
                    "{error}; authenticated Bolt working-set capacity is temporarily busy; \
                     retry with a smaller batch"
                ),
            ),
            BoltError::TooLarge {
                what: "Bolt in-flight message memory" | "Bolt framed message",
                ..
            } => (
                "Neo.ClientError.Request.Invalid",
                format!(
                    "{error}; split the request into smaller batches or raise \
                     NAMIDB_BOLT_MEMORY_BUDGET_BYTES with matching process/cgroup headroom"
                ),
            ),
            _ => (
                "Neo.ClientError.Request.Invalid",
                format!(
                    "{error}; authenticated Bolt messages are limited to {} bytes by \
                     NAMIDB_BOLT_MAX_MESSAGE_BYTES; split the request into smaller batches or \
                     raise the server limit",
                    self.post_auth_message_bytes
                ),
            ),
        };
        if let Err(write_error) = self.write_failure(code, message).await {
            warn!(
                error = %write_error,
                "failed to send Bolt oversized-message diagnostic"
            );
        }
    }

    async fn write_failure(&mut self, code: &str, message: impl Into<String>) -> Result<()> {
        // Every FAILURE funnels through here, so the negotiated-version
        // check upgrades all of them at once: >= 5.7 carries the GQL error
        // fields, older protocols keep the exact two-key shape.
        let supports_gql = self.version.is_some_and(|v| (v.major, v.minor) >= (5, 7));
        let resp = if supports_gql {
            Response::failure_with_gql(code, message)
        } else {
            Response::failure(code, message)
        };
        self.write_response(resp).await
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
    use crate::message::POST_AUTH_MESSAGE_BYTES;
    use bytes::BytesMut;
    use namidb_query::exec::NodeValue;
    use namidb_query::{Row, RuntimeValue};
    use std::pin::Pin;
    use std::sync::Mutex as StdMutex;
    use std::task::{Context, Poll};
    use tokio::io::{duplex, AsyncRead, AsyncWrite, DuplexStream, ReadBuf};

    struct StaticBackend {
        outcome: StdMutex<Option<RunOutcome>>,
    }

    /// Test transport that reports how many bytes the server consumed after
    /// it is armed. This lets the partial-frame regression prove the decoder
    /// reached the middle of a chunk body before RUN is allowed to finish.
    struct ReadObservedStream {
        inner: DuplexStream,
        armed: Arc<std::sync::atomic::AtomicBool>,
        bytes_read: Arc<std::sync::atomic::AtomicUsize>,
        target: usize,
        target_reached: Arc<Notify>,
    }

    impl AsyncRead for ReadObservedStream {
        fn poll_read(
            mut self: Pin<&mut Self>,
            cx: &mut Context<'_>,
            buf: &mut ReadBuf<'_>,
        ) -> Poll<std::io::Result<()>> {
            let before = buf.filled().len();
            let result = Pin::new(&mut self.inner).poll_read(cx, buf);
            if matches!(result, Poll::Ready(Ok(())))
                && self.armed.load(std::sync::atomic::Ordering::Acquire)
            {
                let read = buf.filled().len().saturating_sub(before);
                let total = self
                    .bytes_read
                    .fetch_add(read, std::sync::atomic::Ordering::AcqRel)
                    .saturating_add(read);
                if total >= self.target {
                    self.target_reached.notify_one();
                }
            }
            result
        }
    }

    impl AsyncWrite for ReadObservedStream {
        fn poll_write(
            mut self: Pin<&mut Self>,
            cx: &mut Context<'_>,
            buf: &[u8],
        ) -> Poll<std::io::Result<usize>> {
            Pin::new(&mut self.inner).poll_write(cx, buf)
        }

        fn poll_flush(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<std::io::Result<()>> {
            Pin::new(&mut self.inner).poll_flush(cx)
        }

        fn poll_shutdown(
            mut self: Pin<&mut Self>,
            cx: &mut Context<'_>,
        ) -> Poll<std::io::Result<()>> {
            Pin::new(&mut self.inner).poll_shutdown(cx)
        }
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

    #[test]
    fn configured_post_auth_limit_never_relaxes_strict_pre_auth_cap() {
        let (_client, server) = duplex(1024);
        let mut session = fixture_session(server, RunOutcome::default(), AuthPolicy::Open)
            .with_post_auth_message_bytes(128 * 1024 * 1024);

        session.state = State::Connected;
        assert_eq!(session.current_message_limit(), PRE_AUTH_MESSAGE_BYTES);
        session.state = State::Authentication;
        assert_eq!(session.current_message_limit(), PRE_AUTH_MESSAGE_BYTES);
        session.state = State::Failed;
        assert_eq!(
            session.current_message_limit(),
            PRE_AUTH_MESSAGE_BYTES,
            "FAILED before LOGON must not bypass the pre-authentication cap"
        );

        session.state = State::Ready;
        session.authenticated = true;
        assert_eq!(session.current_message_limit(), 128 * 1024 * 1024);
    }

    #[test]
    fn session_defaults_to_production_post_auth_limit_and_defends_zero() {
        let (_client, server) = duplex(1024);
        let session = fixture_session(server, RunOutcome::default(), AuthPolicy::Open);
        assert_eq!(
            session.post_auth_message_bytes,
            DEFAULT_POST_AUTH_MESSAGE_BYTES
        );

        let (_client, server) = duplex(1024);
        let session = fixture_session(server, RunOutcome::default(), AuthPolicy::Open)
            .with_post_auth_message_bytes(0);
        assert_eq!(session.post_auth_message_bytes, 1);
    }

    #[test]
    fn only_small_pressure_relief_controls_bypass_decode_admission() {
        for tag in [
            crate::value::struct_tag::PULL,
            crate::value::struct_tag::DISCARD,
            crate::value::struct_tag::COMMIT,
            crate::value::struct_tag::ROLLBACK,
            crate::value::struct_tag::RESET,
            crate::value::struct_tag::GOODBYE,
            crate::value::struct_tag::LOGOFF,
        ] {
            let small = [0xB0, tag];
            assert!(
                !requires_decode_admission(&small),
                "small pressure-relief tag 0x{tag:02X} must remain available"
            );

            let mut large = vec![0; CONTROL_FRAME_MAX_BYTES + 1];
            large[0] = 0xB0;
            large[1] = tag;
            assert!(
                requires_decode_admission(&large),
                "an oversized control-shaped frame can amplify decode memory"
            );
        }

        for tag in [
            crate::value::struct_tag::RUN,
            crate::value::struct_tag::BEGIN,
            crate::value::struct_tag::ROUTE,
            crate::value::struct_tag::TELEMETRY,
            0xEE,
        ] {
            assert!(
                requires_decode_admission(&[0xB0, tag]),
                "data/unknown tag 0x{tag:02X} must consult decode admission"
            );
        }
    }

    #[test]
    fn prefetch_limit_is_aggregate_and_stops_at_message_count_cap() {
        let (_client, server) = duplex(1024);
        let mut session = fixture_session(server, RunOutcome::default(), AuthPolicy::Open)
            .with_post_auth_message_bytes(1024);

        assert_eq!(session.prefetch_read_limit(), 1024);
        session
            .prefetched_messages
            .push_back(BudgetedMessage::unbudgeted(vec![0; 900]));
        session.prefetched_bytes = 900;
        assert_eq!(session.prefetch_read_limit(), 124);

        for _ in session.prefetched_messages.len()..MAX_PREFETCHED_MESSAGES {
            session
                .prefetched_messages
                .push_back(BudgetedMessage::unbudgeted(Vec::new()));
        }
        assert_eq!(session.prefetch_read_limit(), 0);
    }

    #[tokio::test]
    async fn prefetch_remaining_budget_also_bounds_an_in_progress_frame() {
        let (mut client, mut server) = duplex(64);
        // Simulate 900 queued bytes plus 100 bytes already retained for the
        // next frame. A further 30-byte chunk would cross the 1,024-byte
        // aggregate even though that chunk alone is tiny.
        client.write_all(&30_u16.to_be_bytes()).await.unwrap();

        let mut reader = StatefulMessageReader {
            message: vec![0; 100],
            ..StatefulMessageReader::default()
        };
        let error = reader
            .read_message(&mut server, 124, None, None)
            .await
            .unwrap_err();
        assert!(matches!(
            error,
            BoltError::TooLarge {
                what: "Bolt message",
                len: 130,
                max: 124,
            }
        ));
        assert_eq!(reader.oversized_chunk_remaining, 30);
    }

    #[tokio::test]
    async fn oversized_reader_can_drain_to_the_next_frame_without_buffering_it() {
        let (mut client, mut server) = duplex(8 * 1024);
        let sender = tokio::spawn(async move {
            // Force several complete writer chunks into the retained message
            // Vec before the next chunk crosses the limit.
            let oversized = vec![0xAB; 130 * 1024];
            write_message(&mut client, &oversized).await.unwrap();
            write_message(&mut client, b"next").await.unwrap();
        });

        let mut reader = StatefulMessageReader::default();
        let max = 70 * 1024;
        let error = reader
            .read_message(&mut server, max, None, None)
            .await
            .unwrap_err();
        assert!(matches!(
            error,
            BoltError::TooLarge {
                what: "Bolt message",
                max: 71_680,
                ..
            }
        ));
        assert!(reader.message.capacity() >= 64 * 1024);
        assert!(reader.chunk.capacity() >= crate::chunk::DEFAULT_CHUNK_SIZE);
        reader.discard_oversized_message(&mut server).await.unwrap();
        assert_eq!(reader.message.capacity(), 0);
        assert_eq!(reader.chunk.capacity(), 0);
        assert_eq!(
            reader
                .read_message(&mut server, max, None, None)
                .await
                .unwrap()
                .body,
            b"next"
        );
        sender.await.unwrap();
    }

    #[tokio::test]
    async fn completed_frame_upgrade_contention_does_not_drain_the_next_frame() {
        let mut data = vec![0; 300];
        data[0] = 0xB0;
        data[1] = crate::value::struct_tag::RUN;
        let budget = Arc::new(
            MessageMemoryBudget::try_new(MessageMemoryBudget::estimated_bytes_for_wire(data.len()))
                .unwrap(),
        );

        let (mut first_client, mut first_server) = duplex(1024);
        write_msg(&mut first_client, &data).await;
        let mut first_reader = StatefulMessageReader::default();
        let first = first_reader
            .read_message(
                &mut first_server,
                POST_AUTH_MESSAGE_BYTES,
                Some(&budget),
                None,
            )
            .await
            .unwrap();

        let (mut client, mut server) = duplex(1024);
        write_msg(&mut client, &data).await;
        write_msg(&mut client, b"next").await;

        let mut reader = StatefulMessageReader::default();
        let error = reader
            .read_message(&mut server, POST_AUTH_MESSAGE_BYTES, Some(&budget), None)
            .await
            .expect_err("the completed frame upgrade must see temporary contention");
        assert!(matches!(error, BoltError::MemoryBudgetExhausted { .. }));
        assert!(
            reader.message_complete,
            "the frame terminator was consumed before budget rejection"
        );

        // Match the production error path: release retained allocations, then
        // drain only if framing says bytes remain on the transport.
        reader.release_buffered_allocations();
        reader.discard_oversized_message(&mut server).await.unwrap();
        assert_eq!(
            reader
                .read_message(&mut server, POST_AUTH_MESSAGE_BYTES, None, None)
                .await
                .unwrap()
                .body,
            b"next"
        );
        drop(first);
    }

    #[tokio::test]
    async fn shared_message_budget_fails_fast_but_control_frames_bypass_pressure() {
        let data = pack_request(&Value::Struct {
            tag: crate::value::struct_tag::RUN,
            fields: vec![
                Value::String("RETURN $payload".into()),
                Value::Map(BTreeMap::from([(
                    "payload".into(),
                    Value::String("x".repeat(512)),
                )])),
                Value::Map(BTreeMap::new()),
            ],
        });
        assert!(data.len() < CONTROL_FRAME_MAX_BYTES);
        let budget = Arc::new(
            MessageMemoryBudget::try_new(MessageMemoryBudget::estimated_bytes_for_wire(data.len()))
                .unwrap(),
        );

        let (mut first_client, mut first_server) = duplex(16 * 1024);
        write_msg(&mut first_client, &data).await;
        let mut first_reader = StatefulMessageReader::default();
        let first = first_reader
            .read_message(
                &mut first_server,
                POST_AUTH_MESSAGE_BYTES,
                Some(&budget),
                None,
            )
            .await
            .unwrap();
        assert_eq!(first.body, data);
        assert!(
            budget.available_bytes() < 4 * 1024,
            "the first complete RUN must retain its exact shared lease"
        );

        let (mut second_client, mut second_server) = duplex(16 * 1024);
        write_msg(&mut second_client, &data).await;
        let mut second_reader = StatefulMessageReader::default();
        let second_error = tokio::time::timeout(
            std::time::Duration::from_millis(50),
            second_reader.read_message(
                &mut second_server,
                POST_AUTH_MESSAGE_BYTES,
                Some(&budget),
                None,
            ),
        )
        .await
        .expect("budget contention must fail fast, never wait holding partial permits")
        .expect_err("a second RUN unexpectedly exceeded the shared budget");
        assert!(
            matches!(second_error, BoltError::MemoryBudgetExhausted { .. }),
            "unexpected contention error: {second_error:?}"
        );

        for tag in [
            crate::value::struct_tag::PULL,
            crate::value::struct_tag::DISCARD,
            crate::value::struct_tag::COMMIT,
            crate::value::struct_tag::ROLLBACK,
            crate::value::struct_tag::RESET,
            crate::value::struct_tag::GOODBYE,
            crate::value::struct_tag::LOGOFF,
        ] {
            let control = pack_request(&Value::Struct {
                tag,
                fields: Vec::new(),
            });
            let (mut control_client, mut control_server) = duplex(1024);
            write_msg(&mut control_client, &control).await;
            let mut control_reader = StatefulMessageReader::default();
            let admitted = tokio::time::timeout(
                std::time::Duration::from_millis(50),
                control_reader.read_message(
                    &mut control_server,
                    POST_AUTH_MESSAGE_BYTES,
                    Some(&budget),
                    None,
                ),
            )
            .await
            .expect("pressure-relief control frame blocked behind data budget")
            .unwrap();
            assert_eq!(admitted.body, control);
            assert!(
                admitted._lease.is_none(),
                "small control frame unexpectedly consumed the data budget"
            );
        }

        drop(first);
        let (mut third_client, mut third_server) = duplex(16 * 1024);
        write_msg(&mut third_client, &data).await;
        let mut third_reader = StatefulMessageReader::default();
        let admitted = third_reader
            .read_message(
                &mut third_server,
                POST_AUTH_MESSAGE_BYTES,
                Some(&budget),
                None,
            )
            .await
            .expect("a new RUN must be admitted after the first lease drops");
        assert_eq!(admitted.body, data);
    }

    #[tokio::test]
    async fn temporary_message_budget_pressure_returns_retryable_failure() {
        let data = pack_request(&Value::Struct {
            tag: crate::value::struct_tag::RUN,
            fields: vec![
                Value::String("RETURN $payload".into()),
                Value::Map(BTreeMap::from([(
                    "payload".into(),
                    Value::String("x".repeat(512)),
                )])),
                Value::Map(BTreeMap::new()),
            ],
        });
        let budget = Arc::new(
            MessageMemoryBudget::try_new(MessageMemoryBudget::estimated_bytes_for_wire(data.len()))
                .unwrap(),
        );

        let (mut holder_client, mut holder_server) = duplex(4096);
        write_msg(&mut holder_client, &data).await;
        let mut holder_reader = StatefulMessageReader::default();
        let held = holder_reader
            .read_message(
                &mut holder_server,
                POST_AUTH_MESSAGE_BYTES,
                Some(&budget),
                None,
            )
            .await
            .unwrap();

        let (mut client, server) = duplex(64 * 1024);
        let session = fixture_session(server, RunOutcome::default(), AuthPolicy::Open)
            .with_message_memory_budget(Arc::clone(&budget));
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
                fields: vec![Value::Map(BTreeMap::from([(
                    "scheme".into(),
                    Value::String("none".into()),
                )]))],
            }),
        )
        .await;
        let _ = read_msg(&mut client).await;

        write_msg(&mut client, &data).await;
        match decode_response(&read_msg(&mut client).await) {
            Response::Failure(meta) => assert_eq!(
                meta.get("code"),
                Some(&Value::String(
                    "Neo.TransientError.General.DatabaseUnavailable".into()
                ))
            ),
            other => panic!("expected retryable memory FAILURE, got {other:?}"),
        }
        let error = task
            .await
            .expect("session task panicked")
            .expect_err("memory contention must close the rejected connection");
        assert!(matches!(error, BoltError::MemoryBudgetExhausted { .. }));
        drop(held);
    }

    #[tokio::test]
    async fn partial_data_frame_does_not_block_other_data_and_releases_budget_at_deadline() {
        let wire_bytes = CONTROL_FRAME_MAX_BYTES + 1024;
        // Two concurrent messages each carry the fixed base charge, so the
        // shared budget needs two full estimates (the partial frame's
        // framing charge plus the unrelated RUN's decode charge).
        let budget = Arc::new(
            MessageMemoryBudget::try_new(
                MessageMemoryBudget::estimated_bytes_for_wire(wire_bytes) * 2,
            )
            .unwrap(),
        );
        let available_before = budget.available_bytes();
        let (mut slow_client, slow_server) = duplex(16 * 1024);
        slow_client
            .write_all(&(wire_bytes as u16).to_be_bytes())
            .await
            .unwrap();
        let slow_budget = Arc::clone(&budget);
        let slow = tokio::spawn(async move {
            let mut server = slow_server;
            let mut reader = StatefulMessageReader::default();
            reader
                .read_message(
                    &mut server,
                    POST_AUTH_MESSAGE_BYTES,
                    Some(&slow_budget),
                    Some(std::time::Duration::from_millis(50)),
                )
                .await
        });

        tokio::time::timeout(std::time::Duration::from_millis(40), async {
            while budget.available_bytes() == available_before {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("partial data frame never acquired the shared budget");
        let raw_charge = available_before - budget.available_bytes();
        let expected_raw_charge =
            (crate::MESSAGE_MEMORY_BASE_BYTES + 2 * wire_bytes).div_ceil(4096) * 4096;
        assert_eq!(
            raw_charge, expected_raw_charge,
            "an incomplete frame must retain raw Vec headroom, not 16x decode amplification"
        );

        let other_data = pack_request(&Value::Struct {
            tag: crate::value::struct_tag::RUN,
            fields: vec![
                Value::String("RETURN $v".into()),
                Value::Map(BTreeMap::from([(
                    "v".into(),
                    Value::String("x".repeat(256)),
                )])),
                Value::Map(BTreeMap::new()),
            ],
        });
        let (mut other_client, mut other_server) = duplex(4096);
        write_msg(&mut other_client, &other_data).await;
        let mut other_reader = StatefulMessageReader::default();
        let other = tokio::time::timeout(
            std::time::Duration::from_millis(40),
            other_reader.read_message(
                &mut other_server,
                POST_AUTH_MESSAGE_BYTES,
                Some(&budget),
                None,
            ),
        )
        .await
        .expect("a slow partial frame globally blocked unrelated data")
        .expect("unrelated small RUN should fit beside the partial raw charge");
        assert_eq!(other.body, other_data);
        drop(other);

        let error = tokio::time::timeout(std::time::Duration::from_secs(1), slow)
            .await
            .expect("partial-frame deadline did not fire")
            .expect("slow reader task panicked")
            .expect_err("slow partial frame unexpectedly completed");
        assert!(
            matches!(error, BoltError::Io(ref io) if io.kind() == std::io::ErrorKind::TimedOut),
            "unexpected partial-frame error: {error:?}"
        );
        assert_eq!(
            budget.available_bytes(),
            available_before,
            "dropping the timed-out reader must release its byte permits"
        );
    }

    #[tokio::test]
    async fn backend_can_reject_run_before_decode_and_reset_still_recovers() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        struct RejectDecodeBackend {
            admitted_wire_bytes: AtomicUsize,
            runs: AtomicUsize,
        }

        #[async_trait]
        impl Backend for RejectDecodeBackend {
            async fn admit_request_decode(
                &self,
                wire_bytes: usize,
            ) -> std::result::Result<Option<Box<dyn DecodeAdmissionGuard>>, BackendError>
            {
                self.admitted_wire_bytes.store(wire_bytes, Ordering::SeqCst);
                Err(BackendError::Storage(
                    "decode admission rejected projected memory".into(),
                ))
            }

            async fn run(
                &self,
                _cypher: &str,
                _params: Params,
            ) -> std::result::Result<RunOutcome, BackendError> {
                self.runs.fetch_add(1, Ordering::SeqCst);
                Ok(RunOutcome::default())
            }
        }

        let backend = Arc::new(RejectDecodeBackend {
            admitted_wire_bytes: AtomicUsize::new(0),
            runs: AtomicUsize::new(0),
        });
        let (mut client, server) = duplex(64 * 1024);
        let session = Session::new(
            server,
            ServerInfo {
                agent: "NamiDB/test".into(),
                connection_id: "test-pre-decode-admission".into(),
            },
            AuthPolicy::Open,
            backend.clone(),
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
        assert!(matches!(
            decode_response(&read_msg(&mut client).await),
            Response::Success(_)
        ));
        write_msg(
            &mut client,
            &pack_request(&Value::Struct {
                tag: crate::value::struct_tag::LOGON,
                fields: vec![Value::Map(BTreeMap::from([(
                    "scheme".into(),
                    Value::String("none".into()),
                )]))],
            }),
        )
        .await;
        assert!(matches!(
            decode_response(&read_msg(&mut client).await),
            Response::Success(_)
        ));

        // Struct(3), RUN tag, then a value marker lacking its payload. Decode
        // would fail immediately, so receiving the backend message proves the
        // hook ran before PackStream allocation/parameter conversion.
        let malformed_run = [0xB3, crate::value::struct_tag::RUN, 0xD0];
        write_msg(&mut client, &malformed_run).await;
        match decode_response(&read_msg(&mut client).await) {
            Response::Failure(meta) => {
                let message = meta.get("message").and_then(|value| match value {
                    Value::String(message) => Some(message.as_str()),
                    _ => None,
                });
                assert_eq!(message, Some("decode admission rejected projected memory"));
            }
            other => panic!("expected pre-decode FAILURE, got {other:?}"),
        }
        assert_eq!(
            backend.admitted_wire_bytes.load(Ordering::SeqCst),
            malformed_run.len()
        );
        assert_eq!(backend.runs.load(Ordering::SeqCst), 0);

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

        write_msg(
            &mut client,
            &pack_request(&Value::Struct {
                tag: crate::value::struct_tag::GOODBYE,
                fields: vec![],
            }),
        )
        .await;
        drop(client);
        task.await
            .expect("session task panicked")
            .expect("session failed after RESET recovery");
    }

    #[tokio::test]
    async fn decode_admission_guard_lives_through_backend_and_drops_after_handle() {
        use std::sync::atomic::{AtomicBool, Ordering};

        struct ProbeGuard {
            alive: Arc<AtomicBool>,
        }

        impl Drop for ProbeGuard {
            fn drop(&mut self) {
                self.alive.store(false, Ordering::Release);
            }
        }

        struct GuardBackend {
            guard_alive: Arc<AtomicBool>,
            observed_in_run: Arc<AtomicBool>,
        }

        #[async_trait]
        impl Backend for GuardBackend {
            async fn admit_request_decode(
                &self,
                _wire_bytes: usize,
            ) -> std::result::Result<Option<Box<dyn DecodeAdmissionGuard>>, BackendError>
            {
                self.guard_alive.store(true, Ordering::Release);
                Ok(Some(Box::new(ProbeGuard {
                    alive: Arc::clone(&self.guard_alive),
                })))
            }

            async fn run(
                &self,
                _cypher: &str,
                _params: Params,
            ) -> std::result::Result<RunOutcome, BackendError> {
                assert!(
                    self.guard_alive.load(Ordering::Acquire),
                    "pre-decode reservation dropped before backend execution"
                );
                self.observed_in_run.store(true, Ordering::Release);
                Ok(RunOutcome::default())
            }
        }

        let guard_alive = Arc::new(AtomicBool::new(false));
        let observed_in_run = Arc::new(AtomicBool::new(false));
        let backend = Arc::new(GuardBackend {
            guard_alive: Arc::clone(&guard_alive),
            observed_in_run: Arc::clone(&observed_in_run),
        });
        let (mut client, server) = duplex(64 * 1024);
        let session = Session::new(
            server,
            ServerInfo {
                agent: "NamiDB/test".into(),
                connection_id: "test-decode-guard".into(),
            },
            AuthPolicy::Open,
            backend,
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
                fields: vec![Value::Map(BTreeMap::from([(
                    "scheme".into(),
                    Value::String("none".into()),
                )]))],
            }),
        )
        .await;
        let _ = read_msg(&mut client).await;

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
            Response::Success(_)
        ));
        assert!(observed_in_run.load(Ordering::Acquire));
        tokio::time::timeout(std::time::Duration::from_secs(1), async {
            while guard_alive.load(Ordering::Acquire) {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("decode reservation guard leaked after handle");

        write_msg(
            &mut client,
            &pack_request(&Value::Struct {
                tag: crate::value::struct_tag::PULL,
                fields: vec![Value::Map(BTreeMap::from([("n".into(), Value::Int(-1))]))],
            }),
        )
        .await;
        let _ = read_msg(&mut client).await;
        write_msg(
            &mut client,
            &pack_request(&Value::Struct {
                tag: crate::value::struct_tag::GOODBYE,
                fields: vec![],
            }),
        )
        .await;
        drop(client);
        task.await.unwrap().unwrap();
    }

    #[tokio::test]
    async fn decode_admission_rejection_in_tx_rolls_back_before_failed() {
        use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

        struct AdmissionTxBackend {
            decode_admissions: AtomicUsize,
            tx_open: AtomicBool,
            rollbacks: AtomicUsize,
        }

        #[async_trait]
        impl Backend for AdmissionTxBackend {
            async fn admit_request_decode(
                &self,
                _wire_bytes: usize,
            ) -> std::result::Result<Option<Box<dyn DecodeAdmissionGuard>>, BackendError>
            {
                let prior = self.decode_admissions.fetch_add(1, Ordering::SeqCst);
                if prior == 0 {
                    Ok(None) // BEGIN
                } else {
                    Err(BackendError::Storage(
                        "projected decode memory rejected in transaction".into(),
                    ))
                }
            }

            async fn run(
                &self,
                _cypher: &str,
                _params: Params,
            ) -> std::result::Result<RunOutcome, BackendError> {
                panic!("rejected RUN must not reach the backend")
            }

            async fn begin_tx(&self) -> std::result::Result<(), BackendError> {
                self.tx_open.store(true, Ordering::SeqCst);
                Ok(())
            }

            async fn rollback_tx(&self) -> std::result::Result<(), BackendError> {
                if self.tx_open.swap(false, Ordering::SeqCst) {
                    self.rollbacks.fetch_add(1, Ordering::SeqCst);
                }
                Ok(())
            }
        }

        let backend = Arc::new(AdmissionTxBackend {
            decode_admissions: AtomicUsize::new(0),
            tx_open: AtomicBool::new(false),
            rollbacks: AtomicUsize::new(0),
        });
        let (mut client, server) = duplex(64 * 1024);
        let session = Session::new(
            server,
            ServerInfo {
                agent: "NamiDB/test".into(),
                connection_id: "test-tx-decode-admission".into(),
            },
            AuthPolicy::Open,
            backend.clone(),
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
                fields: vec![Value::Map(BTreeMap::from([(
                    "scheme".into(),
                    Value::String("none".into()),
                )]))],
            }),
        )
        .await;
        let _ = read_msg(&mut client).await;

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
        assert!(backend.tx_open.load(Ordering::SeqCst));

        let malformed_run = [0xB3, crate::value::struct_tag::RUN, 0xD0];
        write_msg(&mut client, &malformed_run).await;
        assert!(matches!(
            decode_response(&read_msg(&mut client).await),
            Response::Failure(_)
        ));
        assert!(
            !backend.tx_open.load(Ordering::SeqCst),
            "the writer-pinning transaction must be closed before FAILURE"
        );
        assert_eq!(
            backend.rollbacks.load(Ordering::SeqCst),
            1,
            "decode admission must roll back the transaction immediately"
        );

        drop(client);
        task.await
            .expect("session task panicked")
            .expect("session failed after rolling back rejected transaction");
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

    /// A session used to await `Backend::run` without touching the socket, so
    /// dropping the client could leave an unbounded write holding the server's
    /// single writer forever. The RUN reader now notices EOF, publishes the
    /// cancellation token, and waits for backend cleanup before the session
    /// exits.
    #[tokio::test]
    async fn disconnect_during_run_notifies_backend_and_finishes_session() {
        use std::sync::atomic::{AtomicBool, Ordering};

        struct DisconnectAwareBackend {
            started: Arc<Notify>,
            cancelled: Arc<AtomicBool>,
        }

        #[async_trait]
        impl Backend for DisconnectAwareBackend {
            async fn run(
                &self,
                _cypher: &str,
                _params: Params,
            ) -> std::result::Result<RunOutcome, BackendError> {
                std::future::pending().await
            }

            async fn run_with_cancellation(
                &self,
                _cypher: &str,
                _params: Params,
                cancellation: RunCancellation,
            ) -> std::result::Result<RunOutcome, BackendError> {
                self.started.notify_one();
                cancellation.cancelled().await;
                self.cancelled.store(true, Ordering::SeqCst);
                Err(BackendError::Other("client disconnected".into()))
            }
        }

        let started = Arc::new(Notify::new());
        let cancelled = Arc::new(AtomicBool::new(false));
        let (mut client, server) = duplex(64 * 1024);
        let session = Session::new(
            server,
            ServerInfo {
                agent: "NamiDB/test".into(),
                connection_id: "disconnect-run".into(),
            },
            AuthPolicy::Open,
            Arc::new(DisconnectAwareBackend {
                started: Arc::clone(&started),
                cancelled: Arc::clone(&cancelled),
            }),
        );
        let task = tokio::spawn(async move { session.run().await });

        send_handshake(&mut client).await;
        let _ = read_handshake_reply(&mut client).await;
        for (tag, auth) in [
            (crate::value::struct_tag::HELLO, BTreeMap::new()),
            (crate::value::struct_tag::LOGON, {
                let mut auth = BTreeMap::new();
                auth.insert("scheme".into(), Value::String("none".into()));
                auth
            }),
        ] {
            write_msg(
                &mut client,
                &pack_request(&Value::Struct {
                    tag,
                    fields: vec![Value::Map(auth)],
                }),
            )
            .await;
            let _ = read_msg(&mut client).await;
        }

        write_msg(
            &mut client,
            &pack_request(&Value::Struct {
                tag: crate::value::struct_tag::RUN,
                fields: vec![
                    Value::String("UNBOUNDED WRITE".into()),
                    Value::Map(BTreeMap::new()),
                    Value::Map(BTreeMap::new()),
                ],
            }),
        )
        .await;
        tokio::time::timeout(std::time::Duration::from_secs(5), started.notified())
            .await
            .expect("backend RUN did not start");

        drop(client);
        let result = tokio::time::timeout(std::time::Duration::from_secs(5), task)
            .await
            .expect("session stayed pinned after client EOF")
            .expect("session task panicked");
        assert!(result.is_err(), "EOF must terminate the session");
        assert!(
            cancelled.load(Ordering::SeqCst),
            "backend never observed disconnect cancellation"
        );
    }

    /// RUN and PULL are commonly pipelined. If RUN wins `select!` after the
    /// session has consumed a chunk header plus part of the PULL body, the
    /// next state-machine iteration must resume that same body. Treating its
    /// next two bytes as a fresh header corrupts the Bolt stream.
    #[tokio::test]
    async fn run_completion_preserves_partially_read_pipelined_pull() {
        struct ControlledBackend {
            started: Arc<Notify>,
            release: Arc<Notify>,
        }

        #[async_trait]
        impl Backend for ControlledBackend {
            async fn run(
                &self,
                _cypher: &str,
                _params: Params,
            ) -> std::result::Result<RunOutcome, BackendError> {
                self.started.notify_one();
                self.release.notified().await;
                let mut bindings = BTreeMap::new();
                bindings.insert("n".into(), RuntimeValue::Integer(7));
                Ok(RunOutcome {
                    fields: vec!["n".into()],
                    rows: vec![Row { bindings }],
                    ..Default::default()
                })
            }
        }

        let started = Arc::new(Notify::new());
        let release = Arc::new(Notify::new());
        let armed = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let bytes_read = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let target_reached = Arc::new(Notify::new());
        let (mut client, server) = duplex(64 * 1024);
        let observed_server = ReadObservedStream {
            inner: server,
            armed: Arc::clone(&armed),
            bytes_read: Arc::clone(&bytes_read),
            // Two chunk-header bytes plus exactly one body byte.
            target: 3,
            target_reached: Arc::clone(&target_reached),
        };
        let session = Session::new(
            observed_server,
            ServerInfo {
                agent: "NamiDB/test".into(),
                connection_id: "partial-pipeline".into(),
            },
            AuthPolicy::Open,
            Arc::new(ControlledBackend {
                started: Arc::clone(&started),
                release: Arc::clone(&release),
            }),
        );
        let task = tokio::spawn(async move { session.run().await });

        send_handshake(&mut client).await;
        let _ = read_handshake_reply(&mut client).await;
        for (tag, auth) in [
            (crate::value::struct_tag::HELLO, BTreeMap::new()),
            (crate::value::struct_tag::LOGON, {
                let mut auth = BTreeMap::new();
                auth.insert("scheme".into(), Value::String("none".into()));
                auth
            }),
        ] {
            write_msg(
                &mut client,
                &pack_request(&Value::Struct {
                    tag,
                    fields: vec![Value::Map(auth)],
                }),
            )
            .await;
            assert!(matches!(
                decode_response(&read_msg(&mut client).await),
                Response::Success(_)
            ));
        }

        write_msg(
            &mut client,
            &pack_request(&Value::Struct {
                tag: crate::value::struct_tag::RUN,
                fields: vec![
                    Value::String("MATCH (n) RETURN n".into()),
                    Value::Map(BTreeMap::new()),
                    Value::Map(BTreeMap::new()),
                ],
            }),
        )
        .await;
        tokio::time::timeout(std::time::Duration::from_secs(5), started.notified())
            .await
            .expect("backend RUN did not start");

        let pull_body = pack_request(&Value::Struct {
            tag: crate::value::struct_tag::PULL,
            fields: vec![Value::Map({
                let mut extra = BTreeMap::new();
                extra.insert("n".into(), Value::Int(-1));
                extra
            })],
        });
        let chunk_len = u16::try_from(pull_body.len()).expect("test PULL fits one chunk");
        let mut framed_pull = Vec::with_capacity(pull_body.len() + 4);
        framed_pull.extend_from_slice(&chunk_len.to_be_bytes());
        framed_pull.extend_from_slice(&pull_body);
        framed_pull.extend_from_slice(&[0, 0]);

        // Arm only after RUN has been decoded, then stop exactly after the
        // PULL header and first body byte have reached StatefulMessageReader.
        armed.store(true, std::sync::atomic::Ordering::Release);
        client.write_all(&framed_pull[..3]).await.unwrap();
        tokio::time::timeout(std::time::Duration::from_secs(5), target_reached.notified())
            .await
            .expect("session did not consume the partial PULL body");
        assert_eq!(bytes_read.load(std::sync::atomic::Ordering::Acquire), 3);

        // Complete RUN while its concurrent frame read is suspended in the
        // body phase. The RUN header SUCCESS arrives before the rest of PULL.
        release.notify_one();
        assert!(matches!(
            decode_response(&read_msg(&mut client).await),
            Response::Success(_)
        ));

        client.write_all(&framed_pull[3..]).await.unwrap();
        match decode_response(&read_msg(&mut client).await) {
            Response::Record(values) => assert_eq!(values, vec![Value::Int(7)]),
            other => panic!("expected intact pipelined PULL RECORD, got {other:?}"),
        }
        assert!(matches!(
            decode_response(&read_msg(&mut client).await),
            Response::Success(_)
        ));

        write_msg(
            &mut client,
            &pack_request(&Value::Struct {
                tag: crate::value::struct_tag::GOODBYE,
                fields: vec![],
            }),
        )
        .await;
        drop(client);
        tokio::time::timeout(std::time::Duration::from_secs(5), task)
            .await
            .expect("session did not finish")
            .expect("session task panicked")
            .expect("session failed after resumed partial PULL");
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
        crate::chunk::read_message(r, POST_AUTH_MESSAGE_BYTES)
            .await
            .unwrap()
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
    async fn run_moves_runtime_vectors_and_pull_converts_one_page_at_a_time() {
        let first = vec![0.25_f32, 0.5];
        let second = vec![0.75_f32, 1.0];
        let first_allocation = first.as_ptr() as usize;
        let second_allocation = second.as_ptr() as usize;
        let rows = vec![
            Row::new()
                .with("embedding", RuntimeValue::Vector(first))
                .with("ordinal", RuntimeValue::Integer(1)),
            Row::new()
                .with("embedding", RuntimeValue::Vector(second))
                .with("ordinal", RuntimeValue::Integer(2)),
        ];
        let outcome = RunOutcome {
            fields: vec!["embedding".into(), "ordinal".into(), "missing".into()],
            rows,
            ..Default::default()
        };
        let (mut client, server) = duplex(64 * 1024);
        let mut session = fixture_session(server, outcome, AuthPolicy::Open);
        session.state = State::Ready;
        session.authenticated = true;

        session
            .execute_run(
                "MATCH (n) RETURN n.embedding, n.ordinal",
                BTreeMap::new(),
                ElementIdMode::Include,
                false,
                None,
            )
            .await
            .unwrap();

        match decode_response(&read_msg(&mut client).await) {
            Response::Success(meta) => assert_eq!(
                meta.get("fields"),
                Some(&Value::List(vec![
                    Value::String("embedding".into()),
                    Value::String("ordinal".into()),
                    Value::String("missing".into()),
                ]))
            ),
            other => panic!("expected RUN field SUCCESS, got {other:?}"),
        }
        assert_eq!(session.pending_rows.len(), 2);
        assert_eq!(session.pending_fields.len(), 3);
        match session.pending_rows[0].get("embedding") {
            Some(RuntimeValue::Vector(values)) => {
                assert_eq!(values.as_ptr() as usize, first_allocation);
            }
            other => panic!("RUN expanded the first runtime vector: {other:?}"),
        }
        match session.pending_rows[1].get("embedding") {
            Some(RuntimeValue::Vector(values)) => {
                assert_eq!(values.as_ptr() as usize, second_allocation);
            }
            other => panic!("RUN expanded the second runtime vector: {other:?}"),
        }

        let pull_one = || Request::Pull {
            extra: [("n".into(), Value::Int(1))].into_iter().collect(),
        };
        session
            .handle_in_streaming(pull_one(), ElementIdMode::Include)
            .await
            .unwrap();
        match decode_response(&read_msg(&mut client).await) {
            Response::Record(values) => assert_eq!(
                values,
                vec![
                    Value::List(vec![Value::Float(0.25), Value::Float(0.5)]),
                    Value::Int(1),
                    Value::Null,
                ]
            ),
            other => panic!("expected first lazy RECORD, got {other:?}"),
        }
        match decode_response(&read_msg(&mut client).await) {
            Response::Success(meta) => {
                assert_eq!(meta.get("has_more"), Some(&Value::Bool(true)))
            }
            other => panic!("expected paged SUCCESS, got {other:?}"),
        }
        assert_eq!(session.pending_rows.len(), 1);
        match session.pending_rows[0].get("embedding") {
            Some(RuntimeValue::Vector(values)) => {
                assert_eq!(values.as_ptr() as usize, second_allocation);
            }
            other => panic!("PULL eagerly expanded the next page: {other:?}"),
        }

        session
            .handle_in_streaming(pull_one(), ElementIdMode::Include)
            .await
            .unwrap();
        match decode_response(&read_msg(&mut client).await) {
            Response::Record(values) => assert_eq!(
                values,
                vec![
                    Value::List(vec![Value::Float(0.75), Value::Float(1.0)]),
                    Value::Int(2),
                    Value::Null,
                ]
            ),
            other => panic!("expected second lazy RECORD, got {other:?}"),
        }
        assert!(matches!(
            decode_response(&read_msg(&mut client).await),
            Response::Success(_)
        ));
        assert_eq!(session.state, State::Ready);
        assert!(session.pending_rows.is_empty());
        assert!(session.pending_fields.is_empty());
    }

    #[tokio::test]
    async fn pull_repeats_duplicate_projection_columns_before_consuming_binding() {
        let outcome = RunOutcome {
            fields: vec!["x".into(), "x".into()],
            rows: vec![Row::new().with("x", RuntimeValue::Vector(vec![0.25, 0.5]))],
            ..Default::default()
        };
        let (mut client, server) = duplex(64 * 1024);
        let mut session = fixture_session(server, outcome, AuthPolicy::Open);
        session.state = State::Ready;
        session.authenticated = true;

        session
            .execute_run(
                "RETURN x, x",
                BTreeMap::new(),
                ElementIdMode::Include,
                false,
                None,
            )
            .await
            .unwrap();
        let _ = read_msg(&mut client).await;
        assert_eq!(session.pending_field_is_last_occurrence, vec![false, true]);

        session
            .handle_in_streaming(
                Request::Pull {
                    extra: [("n".into(), Value::Int(-1))].into_iter().collect(),
                },
                ElementIdMode::Include,
            )
            .await
            .unwrap();
        let expected = Value::List(vec![Value::Float(0.25), Value::Float(0.5)]);
        match decode_response(&read_msg(&mut client).await) {
            Response::Record(values) => {
                assert_eq!(values, vec![expected.clone(), expected]);
            }
            other => panic!("expected duplicate-column RECORD, got {other:?}"),
        }
        assert!(matches!(
            decode_response(&read_msg(&mut client).await),
            Response::Success(_)
        ));
        assert!(session.pending_field_is_last_occurrence.is_empty());
    }

    #[tokio::test]
    async fn reset_releases_pending_runtime_rows_and_stream_metadata() {
        let outcome = RunOutcome {
            fields: vec!["embedding".into()],
            rows: vec![Row::new().with("embedding", RuntimeValue::Vector(vec![1.0; 1024]))],
            statement_type: StatementType::Write,
            counters: [("properties-set".into(), 1)].into_iter().collect(),
        };
        let (mut client, server) = duplex(64 * 1024);
        let mut session = fixture_session(server, outcome, AuthPolicy::Open);
        session.state = State::Ready;
        session.authenticated = true;

        session
            .execute_run(
                "RETURN $embedding",
                BTreeMap::new(),
                ElementIdMode::Include,
                false,
                None,
            )
            .await
            .unwrap();
        let _ = read_msg(&mut client).await;
        assert_eq!(session.state, State::Streaming);
        assert_eq!(session.pending_rows.len(), 1);

        session
            .handle(Request::Reset, ElementIdMode::Include)
            .await
            .unwrap();
        assert!(matches!(
            decode_response(&read_msg(&mut client).await),
            Response::Success(_)
        ));
        assert_eq!(session.state, State::Ready);
        assert!(session.pending_rows.is_empty());
        assert!(session.pending_fields.is_empty());
        assert!(session.pending_field_is_last_occurrence.is_empty());
        assert!(session.pending_statement_type.is_none());
        assert!(session.pending_counters.is_empty());
    }

    #[tokio::test]
    async fn invalid_tx_stream_message_rolls_back_and_releases_result_buffers() {
        let outcome = RunOutcome {
            fields: vec!["embedding".into()],
            rows: vec![Row::new().with("embedding", RuntimeValue::Vector(vec![1.0; 16 * 1024]))],
            statement_type: StatementType::Read,
            counters: [("properties-set".into(), 1)].into_iter().collect(),
        };
        let (mut client, server) = duplex(64 * 1024);
        let mut session = fixture_session(server, outcome, AuthPolicy::Open);
        session.state = State::TxReady;
        session.authenticated = true;

        session
            .execute_run(
                "RETURN $embedding",
                BTreeMap::new(),
                ElementIdMode::Include,
                true,
                None,
            )
            .await
            .unwrap();
        let _ = read_msg(&mut client).await;
        assert_eq!(session.state, State::TxStreaming);
        assert_eq!(session.pending_rows.len(), 1);

        session
            .handle(Request::Commit, ElementIdMode::Include)
            .await
            .unwrap();
        assert!(matches!(
            decode_response(&read_msg(&mut client).await),
            Response::Failure(_)
        ));
        assert_eq!(session.state, State::Failed);
        assert!(session.pending_rows.is_empty());
        assert_eq!(session.pending_rows.capacity(), 0);
        assert!(session.pending_fields.is_empty());
        assert_eq!(session.pending_fields.capacity(), 0);
        assert!(session.pending_field_is_last_occurrence.is_empty());
        assert_eq!(session.pending_field_is_last_occurrence.capacity(), 0);
        assert!(session.pending_statement_type.is_none());
        assert!(session.pending_counters.is_empty());
    }

    #[tokio::test]
    async fn pull_n_pages_the_result_with_has_more() {
        // Five rows, PULL {n: 2} three times: 2 + 2 + 1 records, the first
        // two closes carry has_more=true, the final one the summary — the
        // demand-driven contract a driver's fetch_size relies on. A DISCARD
        // mid-stream must also drop the remainder and close.
        let rows: Vec<Row> = (0..5)
            .map(|i| {
                let mut bindings = std::collections::BTreeMap::new();
                bindings.insert("n".to_string(), RuntimeValue::Integer(i));
                Row { bindings }
            })
            .collect();
        let outcome = RunOutcome {
            fields: vec!["n".into()],
            rows,
            ..Default::default()
        };
        let (mut client, server) = duplex(64 * 1024);
        let session = fixture_session(server, outcome, AuthPolicy::Open);
        let task = tokio::spawn(async move { session.run().await });

        send_handshake(&mut client).await;
        let _ = read_handshake_reply(&mut client).await;
        for (tag, m) in [
            (crate::value::struct_tag::HELLO, BTreeMap::new()),
            (crate::value::struct_tag::LOGON, {
                let mut m = BTreeMap::new();
                m.insert("scheme".into(), Value::String("none".into()));
                m
            }),
        ] {
            write_msg(
                &mut client,
                &pack_request(&Value::Struct {
                    tag,
                    fields: vec![Value::Map(m)],
                }),
            )
            .await;
            let _ = read_msg(&mut client).await;
        }

        let run = Value::Struct {
            tag: crate::value::struct_tag::RUN,
            fields: vec![
                Value::String("MATCH (n) RETURN n".into()),
                Value::Map(BTreeMap::new()),
                Value::Map(BTreeMap::new()),
            ],
        };
        write_msg(&mut client, &pack_request(&run)).await;
        assert!(matches!(
            decode_response(&read_msg(&mut client).await),
            Response::Success(_)
        ));

        let pull_two = || Value::Struct {
            tag: crate::value::struct_tag::PULL,
            fields: vec![Value::Map({
                let mut m = BTreeMap::new();
                m.insert("n".into(), Value::Int(2));
                m
            })],
        };
        // Batch 1: records 0, 1 + has_more.
        write_msg(&mut client, &pack_request(&pull_two())).await;
        for want in [0i64, 1] {
            match decode_response(&read_msg(&mut client).await) {
                Response::Record(v) => assert_eq!(v, vec![Value::Int(want)]),
                other => panic!("expected RECORD {want}, got {other:?}"),
            }
        }
        match decode_response(&read_msg(&mut client).await) {
            Response::Success(meta) => {
                assert_eq!(meta.get("has_more"), Some(&Value::Bool(true)))
            }
            other => panic!("expected has_more SUCCESS, got {other:?}"),
        }
        // Batch 2: records 2, 3 + has_more.
        write_msg(&mut client, &pack_request(&pull_two())).await;
        for want in [2i64, 3] {
            match decode_response(&read_msg(&mut client).await) {
                Response::Record(v) => assert_eq!(v, vec![Value::Int(want)]),
                other => panic!("expected RECORD {want}, got {other:?}"),
            }
        }
        match decode_response(&read_msg(&mut client).await) {
            Response::Success(meta) => {
                assert_eq!(meta.get("has_more"), Some(&Value::Bool(true)))
            }
            other => panic!("expected has_more SUCCESS, got {other:?}"),
        }
        // Batch 3: record 4 + closing summary (no has_more).
        write_msg(&mut client, &pack_request(&pull_two())).await;
        match decode_response(&read_msg(&mut client).await) {
            Response::Record(v) => assert_eq!(v, vec![Value::Int(4)]),
            other => panic!("expected RECORD 4, got {other:?}"),
        }
        match decode_response(&read_msg(&mut client).await) {
            Response::Success(meta) => {
                assert!(!meta.contains_key("has_more"));
                assert!(meta.contains_key("type"), "closing summary meta");
            }
            other => panic!("expected closing SUCCESS, got {other:?}"),
        }

        // A fresh RUN then DISCARD {n: -1} drops everything and closes.
        write_msg(&mut client, &pack_request(&run)).await;
        let _ = read_msg(&mut client).await; // fields SUCCESS
        let discard = Value::Struct {
            tag: crate::value::struct_tag::DISCARD,
            fields: vec![Value::Map({
                let mut m = BTreeMap::new();
                m.insert("n".into(), Value::Int(-1));
                m
            })],
        };
        write_msg(&mut client, &pack_request(&discard)).await;
        match decode_response(&read_msg(&mut client).await) {
            Response::Success(meta) => assert!(!meta.contains_key("has_more")),
            other => panic!("expected closing SUCCESS, got {other:?}"),
        }

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
        // First response: SUCCESS { fields: ["n"] } — no records yet: they
        // are demand-driven by PULL.
        let r1 = read_msg(&mut client).await;
        match decode_response(&r1) {
            Response::Success(meta) => assert!(meta.contains_key("fields")),
            other => panic!("expected SUCCESS, got {:?}", other),
        }

        // PULL — emits the RECORD then the closing SUCCESS.
        let pull = Value::Struct {
            tag: crate::value::struct_tag::PULL,
            fields: vec![Value::Map({
                let mut m = BTreeMap::new();
                m.insert("n".into(), Value::Int(-1));
                m
            })],
        };
        write_msg(&mut client, &pack_request(&pull)).await;
        let r2 = read_msg(&mut client).await;
        match decode_response(&r2) {
            Response::Record(values) => {
                assert_eq!(values, vec![Value::Int(42)]);
            }
            other => panic!("expected RECORD, got {:?}", other),
        }
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
