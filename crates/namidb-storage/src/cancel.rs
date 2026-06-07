//! Cooperative cancellation for read queries (the query timeout).
//!
//! A read query's wall-clock deadline rides a tokio task-local scoped over
//! the whole execution. Because the deadline lives in this crate, the
//! CPU-bound SST decode and merge loops can probe it directly and abort a
//! single long-running operator mid-flight, not only at the query operator
//! boundaries above them (a giant single-SST decode used to run to
//! completion regardless of the deadline). The query layer scopes the
//! deadline through [`with_deadline`] and reads it through the same
//! task-local.
//!
//! When no deadline is in scope (writes, tests, the no-timeout server
//! config) every probe is a cheap task-local miss and the read path keeps
//! its baseline cost.

use std::future::Future;
use std::time::Instant;

use crate::error::{Error, Result};

tokio::task_local! {
    static DEADLINE: Instant;
}

/// Run `fut` with `deadline` scoped on the current task, so any read this
/// task performs can probe it. `None` runs `fut` unguarded: no task-local is
/// installed, so [`check`] and [`deadline_exceeded`] stay no-ops.
pub async fn with_deadline<F: Future>(deadline: Option<Instant>, fut: F) -> F::Output {
    match deadline {
        Some(at) => DEADLINE.scope(at, fut).await,
        None => fut.await,
    }
}

/// `true` when a deadline is in scope and has passed.
#[inline]
pub fn deadline_exceeded() -> bool {
    DEADLINE
        .try_with(|at| Instant::now() >= *at)
        .unwrap_or(false)
}

/// `Err(Error::Timeout)` when a deadline is in scope and has passed, else
/// `Ok(())`. Call it periodically inside a long CPU-bound loop so the work
/// aborts cooperatively instead of pinning a worker until it returns.
#[inline]
pub fn check() -> Result<()> {
    if deadline_exceeded() {
        Err(Error::Timeout)
    } else {
        Ok(())
    }
}

/// How many rows a decode/merge loop processes between deadline probes.
/// Probing every row would put an `Instant::now()` on the hot path; a power
/// of two lets the compiler turn the modulus into a mask.
pub const CHECK_STRIDE: usize = 1024;
