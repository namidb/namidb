//! Read-query execution guards: a wall-clock timeout and a row cap.
//!
//! The deadline rides a tokio task-local in [`namidb_storage::cancel`] so the
//! storage crate's CPU-bound SST decode and merge loops can probe it too and
//! abort a single long operator mid-flight (cooperative cancellation), not
//! only at the query operator boundaries here. The row cap is a query-local
//! concern on a separate task-local. The regular query path (no guard in
//! scope: tests, the write executor, profiling) keeps its baseline cost.
//!
//! - The deadline is checked at operator boundaries, inside the long scan /
//!   expand loops, and inside the storage decode loops.
//! - The row cap bounds the rows any single operator materialises: the
//!   executor aborts a query whose operator output would exceed it, plus a
//!   fast pre-check on the multiplicative `CrossProduct` and a fast-fail
//!   inside the `Expand` accumulation loop so a runaway never fully
//!   materialises.
//!
//! The server scopes both from its configuration (env `NAMIDB_QUERY_TIMEOUT`
//! and `NAMIDB_QUERY_ROW_CAP`). Writes carry only a deadline (no row cap):
//! the write executor scopes it through [`with_limits`] in
//! `execute_write_with_deadline` and probes it in its per-row loops, so a
//! runaway statement aborts before commit. The bare `execute_write` and an
//! open Bolt transaction's own idle timeout remain separate, unrelated
//! bounds.

use std::future::Future;
use std::time::Instant;

use super::walker::ExecError;

tokio::task_local! {
    /// Maximum rows a single operator may materialise, when a cap is in scope.
    static ROW_CAP: usize;
}

/// Run `fut` with an optional deadline and row cap scoped on the current
/// task. The deadline is installed on [`namidb_storage::cancel`]'s task-local
/// so storage decode loops observe it; the row cap on a query-local one. When
/// both are `None` the future runs unguarded (no overhead).
pub(crate) async fn with_limits<F: Future>(
    deadline: Option<Instant>,
    row_cap: Option<usize>,
    fut: F,
) -> F::Output {
    let capped = async move {
        match row_cap {
            Some(cap) => ROW_CAP.scope(cap, fut).await,
            None => fut.await,
        }
    };
    namidb_storage::cancel::with_deadline(deadline, capped).await
}

/// `Err(ExecError::Timeout)` when a deadline is in scope and has passed,
/// `Ok(())` otherwise. Delegates to the shared storage task-local so the
/// query and storage layers observe the same deadline. A no-op when none is
/// scoped.
#[inline]
pub(crate) fn check_deadline() -> Result<(), ExecError> {
    if namidb_storage::cancel::deadline_exceeded() {
        Err(ExecError::Timeout)
    } else {
        Ok(())
    }
}

/// `Err(ExecError::RowCap)` when a row cap is in scope and `len` exceeds it,
/// `Ok(())` otherwise. `len` is the row count an operator produced (or is
/// about to produce). A no-op when no cap is scoped.
#[inline]
pub(crate) fn check_row_cap(len: usize) -> Result<(), ExecError> {
    ROW_CAP
        .try_with(|cap| {
            if len > *cap {
                Err(ExecError::RowCap(*cap))
            } else {
                Ok(())
            }
        })
        .unwrap_or(Ok(()))
}
