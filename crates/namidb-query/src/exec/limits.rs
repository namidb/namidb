//! Read-query execution guards: a wall-clock timeout and a row cap.
//!
//! Both are carried by a single task-local scoped on the current tokio
//! task and probed cheaply by the executor. Mirrors the
//! [`ProfileCollector`](crate::profile) pattern: the regular query path
//! (no guard in scope, e.g. tests, the write executor, profiling) keeps
//! its baseline cost.
//!
//! - The deadline is checked at operator boundaries and inside the long
//!   scan / expand loops.
//! - The row cap bounds the rows any single operator materialises: the
//!   executor aborts a query whose operator output would exceed it, plus a
//!   fast pre-check on the multiplicative `CrossProduct` and a fast-fail
//!   inside the `Expand` accumulation loop so a runaway never fully
//!   materialises.
//!
//! The server scopes both from its configuration (env `NAMIDB_QUERY_TIMEOUT`
//! and `NAMIDB_QUERY_ROW_CAP`). Writes are not guarded here; an open Bolt
//! transaction is bounded separately by its idle timeout.

use std::future::Future;
use std::time::Instant;

use super::walker::ExecError;

/// The active limits for one read query.
struct Limits {
    /// Wall-clock deadline; `None` leaves the query untimed.
    deadline: Option<Instant>,
    /// Maximum rows a single operator may materialise; `None` is unbounded.
    row_cap: Option<usize>,
}

tokio::task_local! {
    static LIMITS: Limits;
}

/// Run `fut` with an optional deadline and row cap scoped on the current
/// task. When both are `None` the future runs unguarded (no overhead).
pub(crate) async fn with_limits<F: Future>(
    deadline: Option<Instant>,
    row_cap: Option<usize>,
    fut: F,
) -> F::Output {
    if deadline.is_none() && row_cap.is_none() {
        return fut.await;
    }
    LIMITS.scope(Limits { deadline, row_cap }, fut).await
}

/// `Err(ExecError::Timeout)` when a deadline is in scope and has passed,
/// `Ok(())` otherwise. A cheap task-local probe plus an `Instant`
/// comparison; a no-op when no deadline is scoped.
#[inline]
pub(crate) fn check_deadline() -> Result<(), ExecError> {
    LIMITS
        .try_with(|l| match l.deadline {
            Some(at) if Instant::now() >= at => Err(ExecError::Timeout),
            _ => Ok(()),
        })
        .unwrap_or(Ok(()))
}

/// `Err(ExecError::RowCap)` when a row cap is in scope and `len` exceeds
/// it, `Ok(())` otherwise. `len` is the row count an operator produced (or
/// is about to produce). A no-op when no cap is scoped.
#[inline]
pub(crate) fn check_row_cap(len: usize) -> Result<(), ExecError> {
    LIMITS
        .try_with(|l| match l.row_cap {
            Some(cap) if len > cap => Err(ExecError::RowCap(cap)),
            _ => Ok(()),
        })
        .unwrap_or(Ok(()))
}
