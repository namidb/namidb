//! Single-writer epoch fencing.
//!
//! A writer process is **fenced** when another writer has incremented the
//! epoch on the canonical manifest. The CAS protocol on the manifest guarantees
//! that at most one writer can win each epoch transition; everyone else needs
//! to discover this and stop issuing writes.
//!
//! This module is intentionally I/O-free. The wiring to the manifest store
//! lives in [`crate::manifest`].

use std::fmt;

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::error::{Error, Result};

/// Monotonic epoch counter. Incremented every time a new writer claims a
/// namespace.
#[derive(Debug, Clone, Copy, Eq, PartialEq, Ord, PartialOrd, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Epoch(pub u64);

impl Epoch {
    pub const ZERO: Epoch = Epoch(0);

    pub fn next(self) -> Epoch {
        Epoch(self.0 + 1)
    }
    pub fn as_u64(self) -> u64 {
        self.0
    }
}

impl fmt::Display for Epoch {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "e{}", self.0)
    }
}

/// A writer's local fencing token. Compare against the current manifest before
/// every mutation; if `current_epoch > self.epoch`, we have been fenced.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct WriterFence {
    pub epoch: Epoch,
    pub writer_id: Uuid,
}

impl WriterFence {
    pub fn new(epoch: Epoch) -> Self {
        Self {
            epoch,
            writer_id: Uuid::now_v7(),
        }
    }

    /// Returns `Err(Error::Fenced)` when `current` has surpassed our epoch.
    /// Callers should propagate this and drop any in-flight writes.
    pub fn assert_alive(&self, current: Epoch) -> Result<()> {
        if current > self.epoch {
            return Err(Error::Fenced {
                mine: self.epoch.as_u64(),
                current: current.as_u64(),
            });
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fence_passes_when_equal() {
        let f = WriterFence::new(Epoch(7));
        f.assert_alive(Epoch(7)).unwrap();
    }

    #[test]
    fn fence_passes_when_local_ahead() {
        // local ahead can't actually happen in practice but the assertion
        // models "I am still alive" — only `current > mine` fences.
        let f = WriterFence::new(Epoch(7));
        f.assert_alive(Epoch(6)).unwrap();
    }

    #[test]
    fn fence_trips_when_advanced() {
        let f = WriterFence::new(Epoch(7));
        let err = f.assert_alive(Epoch(8)).unwrap_err();
        match err {
            Error::Fenced { mine, current } => {
                assert_eq!(mine, 7);
                assert_eq!(current, 8);
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }
}
