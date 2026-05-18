//! Top-level error type shared across NamiDB crates.
//!
//! Each crate is free to define its own internal errors but should expose a
//! conversion into [`Error`] for cross-crate boundaries.

use std::result::Result as StdResult;

use thiserror::Error;

/// Convenience alias.
pub type Result<T, E = Error> = StdResult<T, E>;

/// Top-level error type for NamiDB.
///
/// We intentionally keep this small and let nested errors live in their crates;
/// only conditions that surface to callers across crates land here.
#[derive(Debug, Error)]
pub enum Error {
 /// Schema-level error: invalid label name, conflicting type, etc.
 #[error("schema error: {0}")]
 Schema(String),

 /// Value coercion / type mismatch.
 #[error("type error: {0}")]
 Type(String),

 /// Identifier could not be parsed or formatted.
 #[error("invalid identifier: {0}")]
 InvalidId(String),

 /// Generic invariant violation — caller passed garbage we did not
 /// otherwise classify. We do **not** use this for "the world changed
 /// underneath us"; that maps to specific storage errors.
 #[error("invariant violation: {0}")]
 Invariant(String),
}

impl Error {
 pub fn schema(msg: impl Into<String>) -> Self {
 Error::Schema(msg.into())
 }
 pub fn typ(msg: impl Into<String>) -> Self {
 Error::Type(msg.into())
 }
 pub fn invariant(msg: impl Into<String>) -> Self {
 Error::Invariant(msg.into())
 }
}
