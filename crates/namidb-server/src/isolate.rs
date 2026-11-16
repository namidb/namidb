//! Read execution on threads of its own.
//!
//! The executor is cooperative: it checks the deadline, the cancel flag and
//! the memory budget every few thousand rows, but a CPU-bound stretch of a
//! query (rows already in the cache, a cross product, a sort) never yields
//! to the async runtime. Run inline on a runtime worker, one such query
//! pinned that worker for its whole run: with as many heavy queries as
//! workers the server stopped answering anything, health checks included,
//! and a client that disconnected (HTTP) or sent RESET / closed the socket
//! (Bolt) could not be noticed until the query finished, because the task
//! that would notice was the one running the query.
//!
//! [`run_read_isolated`] runs a read on a blocking-pool thread instead. The
//! operating system schedules those threads preemptively, so a huge query
//! shares the CPU with the others instead of starving them, and the
//! connection task stays free to notice its client. When the awaiting
//! future is dropped (client gone, request timeout, Bolt RESET) it raises
//! the query's cancel flag, and the query stops at its next cooperative
//! check, within milliseconds. Concurrency stays bounded by the read
//! admission gate (`NAMIDB_MAX_CONCURRENT_QUERIES`).

use std::future::Future;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;

/// Reads currently executing on an isolated thread, including any whose
/// caller has gone away and that have not reached their next cooperative
/// check yet.
static RUNNING: AtomicUsize = AtomicUsize::new(0);

/// How many reads are executing on isolated threads right now. A read whose
/// client disconnected stops counting once it has actually stopped, so this
/// is how tests (and operators, through the query metrics) can tell that a
/// cancel really ended the work rather than only the response.
#[doc(hidden)]
pub fn running_isolated_reads() -> usize {
    RUNNING.load(Ordering::Acquire)
}

/// Decrements [`RUNNING`] when the isolated read ends, however it ends.
struct Running;

impl Running {
    fn enter() -> Self {
        RUNNING.fetch_add(1, Ordering::AcqRel);
        Running
    }
}

impl Drop for Running {
    fn drop(&mut self) {
        RUNNING.fetch_sub(1, Ordering::AcqRel);
    }
}

/// Raises the query's cancel flag when dropped while armed.
struct CancelOnDrop(Option<Arc<AtomicBool>>);

impl Drop for CancelOnDrop {
    fn drop(&mut self) {
        if let Some(flag) = &self.0 {
            flag.store(true, Ordering::Relaxed);
        }
    }
}

/// Run `fut` on a blocking-pool thread under the cancel flag in scope (the
/// per-query flag the admin cancel flips) or a fresh one. Dropping the
/// returned future before it completes cancels the query; awaiting it
/// returns the query's own output. The memory budget, deadline and row cap
/// must be scoped inside `fut` (task-locals do not cross threads).
pub(crate) async fn run_read_isolated<T, F>(fut: F) -> T
where
    F: Future<Output = T> + Send + 'static,
    T: Send + 'static,
{
    let flag = namidb_storage::cancel::current_cancel_flag().unwrap_or_default();
    let mut guard = CancelOnDrop(Some(Arc::clone(&flag)));
    let handle = tokio::runtime::Handle::current();
    let running = Running::enter();
    let worker = tokio::task::spawn_blocking(move || {
        let _running = running;
        handle.block_on(namidb_storage::cancel::with_cancel_flag(flag, fut))
    });
    let output = worker.await;
    guard.0 = None;
    match output {
        Ok(output) => output,
        Err(error) => std::panic::resume_unwind(error.into_panic()),
    }
}

/// Run a read that borrows connection state (a Bolt explicit transaction's
/// snapshot or writer) and so cannot move to another thread. On a
/// multi-threaded runtime the current worker hands its other tasks to the
/// rest of the pool for the duration ([`tokio::task::block_in_place`]), so
/// a long statement does not stall unrelated connections; the statement's
/// own deadline, cancel flag and memory budget still stop it. On a
/// current-thread runtime (tests) it simply runs inline.
pub(crate) async fn run_inline_unblocking<F: Future>(fut: F) -> F::Output {
    match tokio::runtime::Handle::try_current() {
        Ok(handle) if handle.runtime_flavor() == tokio::runtime::RuntimeFlavor::MultiThread => {
            tokio::task::block_in_place(move || handle.block_on(fut))
        }
        _ => fut.await,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{Duration, Instant};

    /// A CPU-bound "query" that only stops when its cancel flag is raised.
    async fn spin_until_cancelled() -> bool {
        let started = Instant::now();
        while started.elapsed() < Duration::from_secs(30) {
            if namidb_storage::cancel::check().is_err() {
                return true;
            }
            std::hint::spin_loop();
        }
        false
    }

    #[tokio::test(flavor = "current_thread")]
    async fn a_busy_read_does_not_block_the_runtime_and_stops_when_dropped() {
        let flag = Arc::new(AtomicBool::new(false));
        let query = namidb_storage::cancel::with_cancel_flag(
            Arc::clone(&flag),
            run_read_isolated(spin_until_cancelled()),
        );
        // On a single-threaded runtime an inline busy loop would never let
        // this timeout fire.
        let started = Instant::now();
        let outcome = tokio::time::timeout(Duration::from_millis(200), query).await;
        assert!(outcome.is_err(), "the read finished on its own");
        assert!(started.elapsed() < Duration::from_secs(5));
        // Dropping the read raised its flag.
        assert!(flag.load(Ordering::Relaxed));
    }

    // What the read owns (the call sites move their admission permits in)
    // stays held until the work has really stopped, not only until the
    // caller gave up: otherwise clients that disconnect over and over could
    // keep more reads running than the admission gates allow.
    #[tokio::test(flavor = "current_thread")]
    async fn what_the_read_owns_outlives_an_abandoned_caller() {
        let permit = Arc::new(());
        let held = Arc::clone(&permit);
        let read = run_read_isolated(async move {
            let _held = held;
            spin_until_cancelled().await;
            // A stretch without a cooperative check after the cancel.
            std::thread::sleep(Duration::from_millis(300));
        });
        let outcome = tokio::time::timeout(Duration::from_millis(100), read).await;
        assert!(outcome.is_err(), "the read finished on its own");
        assert_eq!(
            Arc::strong_count(&permit),
            2,
            "released before the read stopped"
        );
        let started = Instant::now();
        while Arc::strong_count(&permit) > 1 {
            assert!(started.elapsed() < Duration::from_secs(5), "never released");
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    }

    #[tokio::test(flavor = "current_thread")]
    async fn the_output_comes_back() {
        assert_eq!(run_read_isolated(async { 42 }).await, 42);
    }
}
