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
//!
//! A lower priority only orders threads that wait for the same CPU. A
//! container's CPU limit is a quota over every core of the host, so a
//! maintenance thread on an idle core runs at full speed beside the query,
//! spends the quota, and the kernel then throttles the whole container,
//! queries included. Maintenance therefore also yields by itself:
//!
//! - **Build slots.** At most [`build_threads`] maintenance threads run at
//!   once (`NAMIDB_INDEX_BUILD_THREADS`, default half the CPUs, at least
//!   one); the others wait at their next [`pace`] point. A slot is taken for
//!   a slice of work at a time, so a long build (a full-text re-analysis)
//!   and a short one (a node compaction) share it instead of queueing.
//! - **Yielding to queries.** While a statement runs ([`Foreground`]), a
//!   maintenance thread works one short slice at a time and sleeps between
//!   slices, leaving most of the CPU (and the disk) to the statement
//!   (`NAMIDB_INDEX_BUILD_DUTY_PERCENT`, default
//!   [`DEFAULT_FOREGROUND_DUTY_PERCENT`]); with no statement running it
//!   works at full speed.
//!
//! Both act at [`pace`], which the hot loops of every build call (once per
//! row, record or posting block): it costs a thread-local increment except
//! every [`PACE_CHECK_EVERY`] calls.

use std::cell::Cell;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Condvar, Mutex, OnceLock};
use std::time::{Duration, Instant};

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
            let _maintenance = MaintenanceThread::enter();
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

/// Default share of the time a maintenance thread works while a statement
/// runs: one [`SLICE`] of work, then a pause four times as long.
pub const DEFAULT_FOREGROUND_DUTY_PERCENT: u32 = 20;

/// Work between two slot hand-offs (and, while a statement runs, between two
/// pauses).
const SLICE: Duration = Duration::from_millis(4);

/// [`pace`] reads the clock once per this many calls.
pub const PACE_CHECK_EVERY: u32 = 64;

/// Longest wait for a build slot. A thread that waited this long runs
/// anyway: slots bound CPU use, they must never be able to deadlock a
/// maintenance thread that holds one while it waits on another.
const MAX_SLOT_WAIT: Duration = Duration::from_millis(250);

/// Maintenance threads allowed to run at once: `NAMIDB_INDEX_BUILD_THREADS`,
/// else half the CPUs this process may use, at least one.
pub fn build_threads() -> usize {
    static THREADS: OnceLock<usize> = OnceLock::new();
    *THREADS.get_or_init(|| {
        std::env::var("NAMIDB_INDEX_BUILD_THREADS")
            .ok()
            .and_then(|value| value.trim().parse::<usize>().ok())
            .filter(|threads| *threads > 0)
            .unwrap_or_else(|| default_build_threads(available_cpus()))
    })
}

/// Half of `cpus`, at least one.
pub fn default_build_threads(cpus: usize) -> usize {
    (cpus / 2).max(1)
}

fn available_cpus() -> usize {
    std::thread::available_parallelism().map_or(1, |n| n.get())
}

/// Percent of the time maintenance works while a statement runs
/// (`NAMIDB_INDEX_BUILD_DUTY_PERCENT`, 1..=100; 100 turns yielding off).
fn foreground_duty_percent() -> u32 {
    static DUTY: OnceLock<u32> = OnceLock::new();
    *DUTY.get_or_init(|| {
        std::env::var("NAMIDB_INDEX_BUILD_DUTY_PERCENT")
            .ok()
            .and_then(|value| value.trim().parse::<u32>().ok())
            .map(|value| value.clamp(1, 100))
            .unwrap_or(DEFAULT_FOREGROUND_DUTY_PERCENT)
    })
}

/// The pause after each slice while a statement runs.
fn foreground_pause() -> Duration {
    let duty = foreground_duty_percent();
    SLICE * (100 - duty) / duty
}

static FOREGROUND: AtomicUsize = AtomicUsize::new(0);

/// Held for the run of a statement (query, write, DDL): maintenance yields
/// while any is alive.
#[derive(Debug)]
pub struct Foreground(());

impl Foreground {
    pub fn new() -> Self {
        FOREGROUND.fetch_add(1, Ordering::Relaxed);
        Self(())
    }
}

impl Default for Foreground {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for Foreground {
    fn drop(&mut self) {
        FOREGROUND.fetch_sub(1, Ordering::Relaxed);
    }
}

/// Whether a statement is running.
pub fn foreground_active() -> bool {
    FOREGROUND.load(Ordering::Relaxed) > 0
}

/// For maintenance that runs as an async task (its reads await on the
/// runtime) rather than on a maintenance thread: called after each read,
/// with how long it took. While a statement runs, it sleeps for `worked`
/// scaled by the duty cycle (`NAMIDB_INDEX_BUILD_DUTY_PERCENT`), at most a
/// second, as a maintenance thread pauses between slices; otherwise it
/// returns at once.
pub async fn yield_to_foreground(worked: Duration) {
    let duty = foreground_duty_percent();
    if !foreground_active() || duty >= 100 {
        return;
    }
    let pause = worked.max(SLICE) * (100 - duty) / duty;
    tokio::time::sleep(pause.min(Duration::from_secs(1))).await;
}

/// Counting semaphore of build slots, for plain threads.
struct BuildSlots {
    free: Mutex<usize>,
    released: Condvar,
}

fn build_slots() -> &'static BuildSlots {
    static SLOTS: OnceLock<BuildSlots> = OnceLock::new();
    SLOTS.get_or_init(|| BuildSlots {
        free: Mutex::new(build_threads()),
        released: Condvar::new(),
    })
}

impl BuildSlots {
    /// Take a slot, waiting at most [`MAX_SLOT_WAIT`]; `false` when the
    /// thread goes on without one.
    fn acquire(&self) -> bool {
        let deadline = Instant::now() + MAX_SLOT_WAIT;
        let mut free = self.free.lock().unwrap_or_else(|e| e.into_inner());
        loop {
            if *free > 0 {
                *free -= 1;
                return true;
            }
            let now = Instant::now();
            if now >= deadline {
                return false;
            }
            free = self
                .released
                .wait_timeout(free, deadline - now)
                .unwrap_or_else(|e| e.into_inner())
                .0;
        }
    }

    fn release(&self) {
        let mut free = self.free.lock().unwrap_or_else(|e| e.into_inner());
        *free += 1;
        drop(free);
        self.released.notify_one();
    }
}

#[derive(Clone, Copy)]
struct SliceState {
    holds_slot: bool,
    slice_start: Instant,
    calls: u32,
}

thread_local! {
    /// Set on maintenance threads ([`spawn_blocking`]); [`pace`] is a no-op
    /// elsewhere (a read that shares a decoder with a build never yields).
    static MAINTENANCE: Cell<Option<SliceState>> = const { Cell::new(None) };
}

/// Marks the current thread as a maintenance thread for its lifetime and
/// holds its build slot.
struct MaintenanceThread;

impl MaintenanceThread {
    fn enter() -> Self {
        let holds_slot = build_slots().acquire();
        MAINTENANCE.with(|state| {
            state.set(Some(SliceState {
                holds_slot,
                slice_start: Instant::now(),
                calls: 0,
            }))
        });
        Self
    }
}

impl Drop for MaintenanceThread {
    fn drop(&mut self) {
        let state = MAINTENANCE.with(|state| state.take());
        if state.is_some_and(|state| state.holds_slot) {
            build_slots().release();
        }
    }
}

/// A yield point of maintenance work; call it from every hot loop of a
/// build (per row, record, posting block). On a maintenance thread, at the
/// end of each slice of work it hands its build slot on, pauses while a
/// statement runs, and takes a slot again. Anywhere else it does nothing.
#[inline]
pub fn pace() {
    MAINTENANCE.with(|cell| {
        let Some(mut state) = cell.get() else {
            return;
        };
        state.calls = state.calls.wrapping_add(1);
        if state.calls % PACE_CHECK_EVERY != 0 {
            cell.set(Some(state));
            return;
        }
        if state.slice_start.elapsed() < SLICE {
            cell.set(Some(state));
            return;
        }
        cell.set(Some(yield_slice(state)));
    })
}

#[cold]
fn yield_slice(mut state: SliceState) -> SliceState {
    if state.holds_slot {
        build_slots().release();
    }
    if foreground_active() && foreground_duty_percent() < 100 {
        std::thread::sleep(foreground_pause());
    }
    wait_out_memory_pressure();
    state.holds_slot = build_slots().acquire();
    state.slice_start = Instant::now();
    state
}

// ── Memory gate ─────────────────────────────────────────────────────────
//
// Maintenance allocates outside every query account: a merge's decoders and
// writers, a base build's buffers. The process memory governor (the server)
// publishes what it last sampled through [`set_memory_pressure`]; at every
// slice boundary a maintenance thread then
//
// - works at half speed while the process is over its reclaim threshold;
// - stops while it is at its ceiling, for at most
//   `NAMIDB_COMPACTION_PRESSURE_PAUSE` (default 10 s), and, when the ceiling
//   still holds after that, marks the thread deferred: the merge loops see it
//   through [`deferred`] and give their memory back by failing the pass with
//   [`crate::Error::MemoryLimit`], which the caller retries later with a
//   smaller plan instead of the kernel killing the process. A loop that does
//   not check it (a vector or full-text base build) goes on throttled as
//   over the reclaim threshold, and is no longer deferred once the process
//   leaves the ceiling.
//
// `NAMIDB_COMPACTION_MEMORY_GATE=0` turns the gate off (3.5.4 behaviour).

/// The process's memory state as the governor last sampled it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum MemoryPressure {
    /// Below the reclaim threshold, or no governor.
    Normal,
    /// Over the reclaim threshold, under the ceiling.
    Reclaim,
    /// At or over the ceiling.
    Ceiling,
}

static MEMORY_PRESSURE: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);
static PRESSURE_PAUSES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static PRESSURE_THROTTLES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static PRESSURE_DEFERRALS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Publish the process's memory state (the server's governor calls this on
/// every sample).
pub fn set_memory_pressure(level: MemoryPressure) {
    let raw = match level {
        MemoryPressure::Normal => 0,
        MemoryPressure::Reclaim => 1,
        MemoryPressure::Ceiling => 2,
    };
    MEMORY_PRESSURE.store(raw, Ordering::Relaxed);
}

/// The last published memory state.
pub fn memory_pressure() -> MemoryPressure {
    match MEMORY_PRESSURE.load(Ordering::Relaxed) {
        0 => MemoryPressure::Normal,
        1 => MemoryPressure::Reclaim,
        _ => MemoryPressure::Ceiling,
    }
}

/// Counters of the memory gate since start: `(pauses at the ceiling, slices
/// throttled over the reclaim threshold, passes deferred)`.
pub fn memory_gate_stats() -> (u64, u64, u64) {
    (
        PRESSURE_PAUSES.load(Ordering::Relaxed),
        PRESSURE_THROTTLES.load(Ordering::Relaxed),
        PRESSURE_DEFERRALS.load(Ordering::Relaxed),
    )
}

/// Whether the memory gate is on (`NAMIDB_COMPACTION_MEMORY_GATE`, default
/// on; `0`, `false`, `off` or `no` turn it off).
pub fn memory_gate_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| !env_flag_off("NAMIDB_COMPACTION_MEMORY_GATE"))
}

/// `true` when `name` is set to `0`, `false`, `off` or `no`.
pub fn env_flag_off(name: &str) -> bool {
    matches!(
        std::env::var(name)
            .map(|value| value.trim().to_ascii_lowercase())
            .as_deref(),
        Ok("0" | "false" | "off" | "no")
    )
}

/// Longest stop at the memory ceiling before a maintenance pass is deferred.
fn pressure_pause_limit() -> Duration {
    static LIMIT: OnceLock<Duration> = OnceLock::new();
    *LIMIT.get_or_init(|| {
        std::env::var("NAMIDB_COMPACTION_PRESSURE_PAUSE")
            .ok()
            .and_then(|value| humantime_secs(&value))
            .unwrap_or(Duration::from_secs(10))
    })
}

/// `"10s"`, `"500ms"` or plain seconds.
fn humantime_secs(raw: &str) -> Option<Duration> {
    let raw = raw.trim();
    if let Some(ms) = raw.strip_suffix("ms") {
        return ms.trim().parse::<u64>().ok().map(Duration::from_millis);
    }
    raw.strip_suffix('s')
        .unwrap_or(raw)
        .trim()
        .parse::<u64>()
        .ok()
        .map(Duration::from_secs)
}

const PRESSURE_POLL: Duration = Duration::from_millis(50);

thread_local! {
    /// Set on a maintenance thread whose pause at the memory ceiling ran
    /// out (see [`deferred`]).
    static DEFERRED: Cell<bool> = const { Cell::new(false) };
}

/// Whether this maintenance thread was told to give up its pass because the
/// process stayed at its memory ceiling. The merge loops check it after
/// [`pace`] and fail with [`crate::Error::MemoryLimit`].
#[inline]
pub fn deferred() -> bool {
    DEFERRED.with(|flag| flag.get())
}

fn wait_out_memory_pressure() {
    if !memory_gate_enabled() {
        return;
    }
    let pressure = memory_pressure();
    if pressure != MemoryPressure::Ceiling {
        // Out of the ceiling the thread is no longer deferred: a loop that
        // carried on (one that does not check [`deferred`]) pauses again at
        // the next ceiling. A merge loop gave its pass up already.
        DEFERRED.with(|flag| flag.set(false));
    }
    match pressure {
        MemoryPressure::Normal => {}
        MemoryPressure::Reclaim => {
            PRESSURE_THROTTLES.fetch_add(1, Ordering::Relaxed);
            std::thread::sleep(SLICE);
        }
        MemoryPressure::Ceiling => {
            if deferred() {
                // Still at the ceiling after this thread's pause ran out, and
                // its loop did not give the pass up (a vector or full-text
                // base build does not check `deferred`): it goes on, but no
                // faster than over the reclaim threshold.
                PRESSURE_THROTTLES.fetch_add(1, Ordering::Relaxed);
                std::thread::sleep(SLICE);
                return;
            }
            PRESSURE_PAUSES.fetch_add(1, Ordering::Relaxed);
            let started = Instant::now();
            let limit = pressure_pause_limit();
            while memory_pressure() == MemoryPressure::Ceiling {
                if started.elapsed() >= limit {
                    PRESSURE_DEFERRALS.fetch_add(1, Ordering::Relaxed);
                    DEFERRED.with(|flag| flag.set(true));
                    tracing::warn!(
                        paused_ms = started.elapsed().as_millis() as u64,
                        "memory pressure: deferring a maintenance pass that is still at the \
                         process memory ceiling"
                    );
                    return;
                }
                std::thread::sleep(PRESSURE_POLL);
            }
        }
    }
}

/// Run `wait` (a blocking wait for input: a channel receive) without holding
/// a build slot, so another maintenance thread can work meanwhile.
pub fn idle<T>(wait: impl FnOnce() -> T) -> T {
    let held = MAINTENANCE.with(|cell| {
        let mut state = cell.get()?;
        let held = state.holds_slot;
        state.holds_slot = false;
        cell.set(Some(state));
        Some(held)
    });
    if held == Some(true) {
        build_slots().release();
    }
    let result = wait();
    if held.is_some() {
        let reacquired = build_slots().acquire();
        MAINTENANCE.with(|cell| {
            if let Some(mut state) = cell.get() {
                state.holds_slot = reacquired;
                state.slice_start = Instant::now();
                cell.set(Some(state));
            } else if reacquired {
                build_slots().release();
            }
        });
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_threads_default_to_half_the_cpus() {
        assert_eq!(default_build_threads(1), 1);
        assert_eq!(default_build_threads(2), 1);
        assert_eq!(default_build_threads(3), 1);
        assert_eq!(default_build_threads(8), 4);
        assert!(build_threads() >= 1);
    }

    #[test]
    fn pace_is_a_no_op_off_maintenance_threads() {
        let started = Instant::now();
        for _ in 0..1_000_000 {
            pace();
        }
        assert!(started.elapsed() < Duration::from_secs(2));
    }

    /// Spin `for_how_long` on a maintenance thread, calling `pace` between
    /// units of work; the number of units done.
    async fn paced_units(for_how_long: Duration) -> u64 {
        spawn_blocking(move || {
            let started = Instant::now();
            let mut units = 0u64;
            while started.elapsed() < for_how_long {
                // ~10 µs of work per unit.
                let unit = Instant::now();
                while unit.elapsed() < Duration::from_micros(10) {
                    std::hint::spin_loop();
                }
                units += 1;
                pace();
            }
            units
        })
        .await
        .unwrap()
    }

    /// While a statement runs, maintenance works a fraction of the time.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn maintenance_yields_while_a_statement_runs() {
        let alone = paced_units(Duration::from_millis(400)).await;
        let statement = Foreground::new();
        let beside = paced_units(Duration::from_millis(400)).await;
        drop(statement);
        // The default duty is 20%; leave room for a loaded test host.
        assert!(
            beside * 2 < alone,
            "alone {alone} units, beside a statement {beside}"
        );
    }

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
