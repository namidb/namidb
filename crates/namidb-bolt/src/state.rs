//! Per-connection Bolt state machine.
//!
//! See RFC-022 §State machine for the full transition diagram. The
//! enum below is the minimal set the session needs to gate incoming
//! messages.

/// Current state of a Bolt session.
///
/// The state changes in response to inbound messages. The session
/// rejects messages that don't match the current state with a
/// `FAILURE { code: "Neo.ClientError.Request.Invalid", … }` response.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum State {
    /// Pre-handshake. The 20-byte handshake hasn't been exchanged yet.
    /// No messages are accepted; only raw handshake bytes.
    Negotiation,
    /// Handshake done, awaiting `HELLO`.
    Connected,
    /// `HELLO` accepted, awaiting `LOGON` (Bolt v5.1+ split auth).
    /// On Bolt v4.4 the session jumps from Connected straight to
    /// Ready inside `HELLO`.
    Authentication,
    /// Auth done. Accepts `RUN`, `BEGIN`, `RESET`, `GOODBYE`, `ROUTE`,
    /// `LOGOFF`, `TELEMETRY`.
    Ready,
    /// `RUN` accepted, server is producing rows. Accepts `PULL`,
    /// `DISCARD`, `RESET`, `GOODBYE`. (Auto-commit transaction.)
    Streaming,
    /// `BEGIN` accepted, waiting for `RUN` / `COMMIT` / `ROLLBACK`.
    TxReady,
    /// Inside a tx, `RUN` accepted, server is producing rows.
    TxStreaming,
    /// A request raised a server error. Only `RESET` / `GOODBYE`
    /// recover from this state; everything else is `IGNORED`.
    Failed,
    /// Terminal — connection is going away.
    Defunct,
}

impl State {
    /// True iff a `RECORD`/`SUCCESS` stream can be produced from this
    /// state. Used by the session to decide whether to emit `IGNORED`.
    pub fn is_streaming(self) -> bool {
        matches!(self, State::Streaming | State::TxStreaming)
    }
}
