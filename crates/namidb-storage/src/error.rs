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

    /// The pointer family could not be resolved to a definitive current version
    /// from the cached lower bound — typically the forward probe exhausted its
    /// window because the namespace advanced far past the bound while a LIST was
    /// stale. **Retryable:** the read-after-write advisory (`current.json`)
    /// catches up, so the caller should retry rather than treat it as fatal.
    #[error("manifest pointer resolution is temporarily stale; retry")]
    PointerResolveStale,

    /// A precondition we believed held was violated by the world we observed.
    #[error("precondition failed: {0}")]
    Precondition(String),

    /// A read query ran past its wall-clock deadline while this crate was
    /// decoding or merging SSTs (cooperative cancellation, see
    /// [`crate::cancel`]). The query layer maps it to its own timeout error.
    /// Only raised when a deadline is in scope.
    #[error("read query exceeded its deadline")]
    Timeout,

    /// The expected manifest version does not exist yet.
    #[error("manifest version {0} not found")]
    ManifestNotFound(u64),

    /// We were fenced: our local epoch is older than the current one in
    /// object storage. Caller should drop its writer state.
    #[error("writer fenced: local epoch {mine}, current epoch {current}")]
    Fenced { mine: u64, current: u64 },

    /// `claim_writer` kept losing the manifest CAS to a body that already
    /// exists at `version` while the pointer never advanced to it — the
    /// signature of an orphan manifest body left by a writer that wrote
    /// the body but crashed before the pointer CAS (e.g. a transient
    /// error in `cas_pointer`). Nobody can supersede that version under
    /// `PutMode::Create`, so the namespace cannot be claimed until the
    /// orphan at `manifest/v{version}.json` is removed. Distinct from
    /// `ManifestCommitCas` (a live race that resolves on retry) so callers
    /// and operators can tell a recoverable race from a stuck namespace.
    #[error(
        "orphan manifest body blocks claim: a body exists at version {version} \
         but the pointer never advanced to it"
    )]
    OrphanManifestBody { version: u64 },

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
