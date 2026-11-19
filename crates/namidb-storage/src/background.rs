//! CPU-heavy maintenance (compaction merges, search index builds) on threads
//! of its own at a lower scheduling priority.
//!
//! Reads run on the runtime's blocking pool. Maintenance used to run there
//! too, at the same priority, so on a small engine (two CPUs) a compaction or
//! an index build took half of the CPU from every query for its whole run,
//! and the same statement took several times longer while one was going.
//! [`spawn_blocking`] runs the closure on a dedicated thread whose nice value
//! is raised by [`BACKGROUND_NICE`] (Linux; elsewhere it only isolates), so
//! the kernel gives queries the CPU first and maintenance the rest: it still
//! uses every idle cycle, it only yields when a query wants to run. A
//! dedicated thread is used because an unprivileged process cannot lower a
//! nice value again, and a pool thread would carry the low priority into the
//! next read it runs.
//!
//! `NAMIDB_BACKGROUND_NICE` sets the increment (0 keeps the normal priority).

use std::sync::OnceLock;

/// Default nice increment of maintenance threads.
pub const BACKGROUND_NICE: i32 = 10;

fn background_nice() -> i32 {
    static NICE: OnceLock<i32> = OnceLock::new();
    *NICE.get_or_init(|| {
        std::env::var("NAMIDB_BACKGROUND_NICE")
            .ok()
            .and_then(|value| value.trim().parse::<i32>().ok())
            .map(|value| value.clamp(0, 19))
            .unwrap_or(BACKGROUND_NICE)
    })
}

/// Raise the calling thread's nice value by `increment` (Linux only).
fn lower_current_thread_priority(increment: i32) {
    if increment <= 0 {
        return;
    }
    #[cfg(target_os = "linux")]
    {
        // SAFETY: plain syscalls on the calling thread; `setpriority` with
        // PRIO_PROCESS and a thread id changes only that thread on Linux.
        unsafe {
            let tid = libc::syscall(libc::SYS_gettid) as libc::id_t;
            let current = libc::getpriority(libc::PRIO_PROCESS, tid);
            let target = (current + increment).min(19);
            if libc::setpriority(libc::PRIO_PROCESS, tid, target) != 0 {
                tracing::debug!("could not lower a maintenance thread's priority");
            }
        }
    }
}

/// Start `f` on a new low-priority thread now, and return a future of its
/// result (like `tokio::task::spawn_blocking`, the work starts before the
/// future is polled). The error is the panic message when `f` panicked. It
/// cannot be stopped once started; the caller's runtime is entered on the
/// thread, so `f` may use runtime handles.
pub fn spawn_blocking<F, T>(f: F) -> impl std::future::Future<Output = Result<T, String>>
where
    F: FnOnce() -> T + Send + 'static,
    T: Send + 'static,
{
    let (sender, receiver) = tokio::sync::oneshot::channel();
    let handle = tokio::runtime::Handle::try_current().ok();
    let nice = background_nice();
    let spawned = std::thread::Builder::new()
        .name("namidb-maint".into())
        .spawn(move || {
            let _entered = handle.as_ref().map(|handle| handle.enter());
            lower_current_thread_priority(nice);
            let result =
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)).map_err(|panic| {
                    panic
                        .downcast_ref::<&str>()
                        .map(|s| s.to_string())
                        .or_else(|| panic.downcast_ref::<String>().cloned())
                        .unwrap_or_else(|| "maintenance task panicked".to_string())
                });
            let _ = sender.send(result);
        })
        .map_err(|error| format!("could not start a maintenance thread: {error}"));
    async move {
        spawned?;
        receiver
            .await
            .unwrap_or_else(|_| Err("maintenance thread ended without a result".to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn returns_the_value_and_reports_a_panic() {
        assert_eq!(spawn_blocking(|| 41 + 1).await, Ok(42));
        let error = spawn_blocking(|| -> u8 { panic!("boom") })
            .await
            .unwrap_err();
        assert!(error.contains("boom"), "{error}");
    }

    #[cfg(target_os = "linux")]
    #[tokio::test]
    async fn the_thread_runs_at_a_lower_priority() {
        let base = unsafe { libc::getpriority(libc::PRIO_PROCESS, 0) };
        let inside = spawn_blocking(|| unsafe {
            let tid = libc::syscall(libc::SYS_gettid) as libc::id_t;
            libc::getpriority(libc::PRIO_PROCESS, tid)
        })
        .await
        .unwrap();
        assert!(
            inside > base || base >= 19,
            "base {base}, maintenance {inside}"
        );
        // The caller's own thread keeps its priority.
        let after = unsafe {
            let tid = libc::syscall(libc::SYS_gettid) as libc::id_t;
            libc::getpriority(libc::PRIO_PROCESS, tid)
        };
        assert_eq!(after, base);
    }
}
