//! Read-query execution guards.
//!
//! A wall-clock deadline scoped on the current tokio task and checked at
//! operator boundaries and inside the long scan / expand loops. Mirrors
//! the [`ProfileCollector`](crate::profile) task-local pattern: the
//! executor probes the task-local cheaply and the regular query path (no
//! guard in scope, e.g. tests, the write executor, profiling) keeps its
//! baseline cost.
//!
//! The server scopes a deadline around read queries from its configured
//! query timeout (env `NAMIDB_QUERY_TIMEOUT`). Writes are not guarded
//! here; an open Bolt transaction is bounded separately by its idle
//! timeout.

use std::future::Future;
use std::time::Instant;

use super::walker::ExecError;

tokio::task_local! {
    static QUERY_DEADLINE: Instant;
}

/// Run `fut` with an optional wall-clock deadline scoped on the current
/// task. `None` runs `fut` unguarded (no overhead, no timeout).
pub(crate) async fn with_deadline<F: Future>(deadline: Option<Instant>, fut: F) -> F::Output {
    match deadline {
        Some(at) => QUERY_DEADLINE.scope(at, fut).await,
        None => fut.await,
    }
}

/// `Err(ExecError::Timeout)` when a deadline is in scope and has passed,
/// `Ok(())` otherwise. A cheap task-local probe plus an `Instant`
/// comparison; a no-op when no deadline is scoped.
#[inline]
pub(crate) fn check_deadline() -> Result<(), ExecError> {
    QUERY_DEADLINE
        .try_with(|at| {
            if Instant::now() >= *at {
                Err(ExecError::Timeout)
            } else {
                Ok(())
            }
        })
        .unwrap_or(Ok(()))
}
