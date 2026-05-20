//! Storage-engine errors.
//!
//! These are the failure modes that surface at the boundary of the
//! `namidb-storage` crate. Cross-crate consumers can decide whether to
//! retry (most CAS losses), abort (corruption), or fence (epoch mismatch).

use std::result::Result as StdResult;

use thiserror::Error;

pub type Result<T, E = Error> = StdResult<T, E>;

#[derive(Debug, Error)]
pub enum Error {
    /// The object store returned an error we did not specifically classify.
    #[error("object store error: {0}")]
    ObjectStore(#[from] object_store::Error),

    /// Path parsing failed.
    #[error("invalid object store path: {0}")]
    Path(#[from] object_store::path::Error),

    /// JSON ser/de.
    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),

    /// Standard I/O error (used for in-memory adapters and bincode codecs).
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    /// We lost the manifest CAS race; another writer advanced the version.
    /// Callers should reload the manifest and retry.
    #[error("manifest commit lost CAS race; expected version {expected}, found {found}")]
    ManifestCommitCas { expected: u64, found: u64 },

    /// A precondition we believed held was violated by the world we observed.
    #[error("precondition failed: {0}")]
    Precondition(String),

    /// The expected manifest version does not exist yet.
    #[error("manifest version {0} not found")]
    ManifestNotFound(u64),

    /// We were fenced: our local epoch is older than the current one in
    /// object storage. Caller should drop its writer state.
    #[error("writer fenced: local epoch {mine}, current epoch {current}")]
    Fenced { mine: u64, current: u64 },

    /// A CRC or schema check on a stored artefact failed.
    #[error("corrupted artefact at {path}: {detail}")]
    Corrupted { path: String, detail: String },

    /// The caller asked us to do something that violates the contract of the
    /// storage layer (e.g. write to a namespace it does not own).
    #[error("invariant violation: {0}")]
    Invariant(String),

    /// Bubble-up from `namidb-core`.
    #[error("core error: {0}")]
    Core(#[from] namidb_core::Error),
}

impl Error {
    pub fn invariant(msg: impl Into<String>) -> Self {
        Error::Invariant(msg.into())
    }
    pub fn precondition(msg: impl Into<String>) -> Self {
        Error::Precondition(msg.into())
    }
}
