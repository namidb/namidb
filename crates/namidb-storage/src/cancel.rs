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
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;

use crate::error::{Error, Result};

/// What interrupted a cooperative probe: the wall clock, or an operator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Interrupt {
    Timeout,
    Cancelled,
}

/// The guards a task runs under: an optional wall-clock deadline and an
/// optional operator-flippable cancel flag. Both are probed by the same
/// ~hundred cooperative check sites, so an admin cancel aborts exactly
/// where a timeout would.
#[derive(Debug, Clone, Default)]
struct CancelCtx {
    deadline: Option<Instant>,
    cancel: Option<Arc<AtomicBool>>,
}

tokio::task_local! {
    static CTX: CancelCtx;
}

/// Run `fut` with `deadline` scoped on the current task, so any read this
/// task performs can probe it. `None` runs `fut` unguarded (an
/// already-scoped cancel flag from an enclosing [`with_cancel_flag`] stays
/// visible). A `Some` deadline inherits any enclosing cancel flag rather
/// than masking it.
pub async fn with_deadline<F: Future>(deadline: Option<Instant>, fut: F) -> F::Output {
    match deadline {
        Some(at) => {
            let cancel = CTX.try_with(|ctx| ctx.cancel.clone()).ok().flatten();
            CTX.scope(
                CancelCtx {
                    deadline: Some(at),
                    cancel,
                },
                fut,
            )
            .await
        }
        None => fut.await,
    }
}

/// Run `fut` with an operator cancel flag scoped on the current task. The
/// server registers one per in-flight query; flipping it makes every
/// cooperative probe under this scope return [`Error::Cancelled`]. Inherits
/// any enclosing deadline; an inner [`with_deadline`] inherits this flag.
pub async fn with_cancel_flag<F: Future>(flag: Arc<AtomicBool>, fut: F) -> F::Output {
    let deadline = CTX.try_with(|ctx| ctx.deadline).ok().flatten();
    CTX.scope(
        CancelCtx {
            deadline,
            cancel: Some(flag),
        },
        fut,
    )
    .await
}

/// The operator cancel flag in scope, if any — for re-scoping onto a
/// spawned task (task-locals do not cross `tokio::spawn`).
pub fn current_cancel_flag() -> Option<Arc<AtomicBool>> {
    CTX.try_with(|ctx| ctx.cancel.clone()).ok().flatten()
}

/// The active interrupt, if any. Operator cancel wins over the clock so the
/// surfaced error names the actual cause.
#[inline]
pub fn interrupted() -> Option<Interrupt> {
    CTX.try_with(|ctx| {
        if ctx
            .cancel
            .as_ref()
            .is_some_and(|flag| flag.load(Ordering::Relaxed))
        {
            Some(Interrupt::Cancelled)
        } else if ctx.deadline.is_some_and(|at| Instant::now() >= at) {
            Some(Interrupt::Timeout)
        } else {
            None
        }
    })
    .ok()
    .flatten()
}

/// `true` when a guard is in scope and has fired (deadline passed OR the
/// query was cancelled). The historical name predates operator cancel.
#[inline]
pub fn deadline_exceeded() -> bool {
    interrupted().is_some()
}

/// `Err(Error::Timeout)` / `Err(Error::Cancelled)` when a guard in scope has
/// fired, else `Ok(())`. Call it periodically inside a long CPU-bound loop
/// so the work aborts cooperatively instead of pinning a worker until it
/// returns.
#[inline]
pub fn check() -> Result<()> {
    match interrupted() {
        None => Ok(()),
        Some(Interrupt::Timeout) => Err(Error::Timeout),
        Some(Interrupt::Cancelled) => Err(Error::Cancelled),
    }
}

/// How many rows a decode/merge loop processes between deadline probes.
/// Probing every row would put an `Instant::now()` on the hot path; a power
/// of two lets the compiler turn the modulus into a mask.
pub const CHECK_STRIDE: usize = 1024;
