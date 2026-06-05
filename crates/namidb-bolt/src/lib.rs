//! Bolt wire protocol implementation for NamiDB.
//!
//! See [RFC-022](../../../docs/rfc/022-bolt-protocol.md) for the
//! design. The crate is split into:
//!
//! - [`codec`] — PackStream encode / decode.
//! - [`value`] — the value model the codec ingests + Node / Rel /
//!   Path helpers.
//! - [`state`] — connection state machine.
//! - [`error`] — crate-local error enum.
//!
//! Higher-level pieces (chunked framing, handshake, message types,
//! per-connection async session) land in subsequent commits — see
//! the matching tasks under task #1 (Bolt protocol).

#![warn(rust_2018_idioms)]

pub mod chunk;
pub mod codec;
pub mod error;
pub mod handshake;
pub mod mapping;
pub mod message;
pub mod session;
pub mod state;
pub mod value;

pub use error::{BoltError, Result};
pub use handshake::{Version, SUPPORTED_VERSIONS};
pub use mapping::{bolt_to_runtime, params_from_bolt_map, runtime_to_bolt, ElementIdMode};
pub use message::{Request, Response};
pub use session::{
    AuthPolicy, Authenticator, Backend, BackendError, RunOutcome, ServerInfo, Session,
    StatementType,
};
pub use state::State;
pub use value::{struct_tag, Node, Relationship, Value};
