//! Process-wide resident-memory admission and pressure relief.
//!
//! `NAMIDB_CACHE_MAX_BYTES` bounds retained cache entries, not the complete
//! process. A loader also owns memtables, transactional indexes, active query
//! rows, allocator arenas, and transient flush/compaction buffers. This
//! governor observes RSS/working-set bytes directly and provides a second,
//! opt-in ceiling for admitting new Cypher work.

use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Disabled by default for backward compatibility.
pub const DEFAULT_MEMORY_MAX_BYTES: usize = 0;

const RECLAIM_PERCENT: usize = 90;
const RESUME_PERCENT: usize = 80;
const HARD_RECLAIM_COOLDOWN: Duration = Duration::from_secs(1);
const WATCHDOG_INTERVAL: Duration = Duration::from_millis(500);

#[derive(Debug, Default)]
struct ReclaimState {
    running: bool,
    last_started: Option<Instant>,
}

/// A rejected admission with the current measured resident set.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MemoryPressure {
    pub resident_bytes: usize,
    pub max_bytes: usize,
}

/// Process-wide memory governor shared by HTTP, Bolt, and every namespace.
#[derive(Debug)]
pub struct MemoryGovernor {
    max_bytes: usize,
    reclaim_bytes: usize,
    resume_bytes: usize,
    resident_bytes: AtomicUsize,
    pressure_active: AtomicBool,
    /// At most one blocking-pool reclaim job may be queued or running.
    ///
    /// `ReclaimState::running` protects the actual cache clear, but setting it
    /// only inside the blocking closure is too late: a request burst could
    /// enqueue hundreds of closures first. This gate spans queue + execution.
    reclaim_job_scheduled: AtomicBool,
    /// Completion generation for hard-limit admissions that arrived while the
    /// one reclaim job was already queued. A watch channel avoids lost wakeups
    /// without retaining one task per notification.
    reclaim_completed: tokio::sync::watch::Sender<u64>,
    reclaim_state: Mutex<ReclaimState>,
    reclaim_events: AtomicU64,
    rejected_queries: AtomicU64,
    admin_flush_gate: tokio::sync::Semaphore,
}

impl MemoryGovernor {
    pub fn new(max_bytes: usize) -> Self {
        let (reclaim_completed, _) = tokio::sync::watch::channel(0);
        Self {
            max_bytes,
            reclaim_bytes: percent(max_bytes, RECLAIM_PERCENT),
            resume_bytes: percent(max_bytes, RESUME_PERCENT),
            resident_bytes: AtomicUsize::new(0),
            pressure_active: AtomicBool::new(false),
            reclaim_job_scheduled: AtomicBool::new(false),
            reclaim_completed,
            reclaim_state: Mutex::new(ReclaimState::default()),
            reclaim_events: AtomicU64::new(0),
            rejected_queries: AtomicU64::new(0),
            admin_flush_gate: tokio::sync::Semaphore::new(1),
        }
    }

    /// Sample current RSS/working-set and admit a new query only while it is
    /// below the configured maximum.
    ///
    /// At 90% we first clear reconstructible storage caches and ask glibc to
    /// return free arenas on the official Linux image. The sample is repeated
    /// before deciding whether to reject, so successful reclaim does not
    /// produce a spurious 503. Hysteresis normally re-arms below 80%; a
    /// successful pass that gets back below 90% also re-arms immediately.
    /// While the hard ceiling remains exceeded, a monotonic one-second
    /// cooldown permits another pass without allowing retry storms to clear
    /// the same caches concurrently.
    pub async fn admit_query(self: &Arc<Self>) -> Result<(), MemoryPressure> {
        // Keep the ordinary below-threshold path to one cheap RSS sample.
        // Cache destruction and `malloc_trim`, however, can walk millions of
        // allocations and must never run on an async serving worker.
        if self.max_bytes == 0 {
            return Ok(());
        }
        let Some(initial) = self.observe_resident() else {
            return Ok(());
        };
        let resident = self.reclaim_off_worker(initial).await;

        if resident >= self.max_bytes {
            self.rejected_queries.fetch_add(1, Ordering::Relaxed);
            return Err(MemoryPressure {
                resident_bytes: resident,
                max_bytes: self.max_bytes,
            });
        }
        Ok(())
    }

    fn observe_resident(&self) -> Option<usize> {
        let resident = sample_resident_bytes()?;
        self.resident_bytes.store(resident, Ordering::Relaxed);
        if resident <= self.resume_bytes {
            self.pressure_active.store(false, Ordering::Release);
        }
        Some(resident)
    }

    /// Whether the current sample can start a useful reclaim pass.
    ///
    /// During an active 90%-to-100% pressure episode one clear is enough:
    /// requests continue but do not enqueue no-op blocking jobs. At the hard
    /// ceiling, another pass becomes eligible only after the monotonic retry
    /// cooldown.
    fn reclaim_job_due(&self, resident: usize) -> bool {
        if resident < self.reclaim_bytes {
            return false;
        }
        if !self.pressure_active.load(Ordering::Acquire) {
            return true;
        }
        if resident < self.max_bytes {
            return false;
        }
        let now = Instant::now();
        let state = self
            .reclaim_state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        !state.running
            && state
                .last_started
                .is_none_or(|last| now.duration_since(last) >= HARD_RECLAIM_COOLDOWN)
    }

    fn try_schedule_reclaim(self: &Arc<Self>, resident: usize) -> Option<ReclaimJobPermit> {
        if !self.reclaim_job_due(resident)
            || self
                .reclaim_job_scheduled
                .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
                .is_err()
        {
            return None;
        }
        Some(ReclaimJobPermit(Arc::clone(self)))
    }

    async fn reclaim_off_worker(self: &Arc<Self>, initial: usize) -> usize {
        let Some(permit) = self.try_schedule_reclaim(initial) else {
            // At the hard ceiling, do not reject from a sample taken while
            // the one useful reclaim is still in flight. Subscribe before the
            // second gate check so completion cannot be lost between them.
            if initial >= self.max_bytes && self.reclaim_job_scheduled.load(Ordering::Acquire) {
                let mut completed = self.reclaim_completed.subscribe();
                if self.reclaim_job_scheduled.load(Ordering::Acquire) {
                    let _ = completed.changed().await;
                }
                return self.observe_resident().unwrap_or(initial);
            }
            return initial;
        };
        let governor = Arc::clone(self);
        match tokio::task::spawn_blocking(move || {
            // Dropping the queued closure without running it also drops this
            // permit, so runtime shutdown cannot leave the scheduling gate set.
            let _permit = permit;
            governor.reclaim_if_needed()
        })
        .await
        {
            Ok(Some(after)) => after,
            Ok(None) => initial,
            Err(error) => {
                tracing::error!(%error, "resident-memory reclaim task failed");
                // We have a valid pre-spawn sample. Preserve fail-closed
                // behavior at the configured hard ceiling if the blocking
                // worker itself failed.
                initial
            }
        }
    }

    /// Sample RSS and reclaim reconstructible state when pressure thresholds
    /// require it, without classifying the observation as a query admission.
    ///
    /// Both foreground admission and the process watchdog use this exact
    /// single-flight/hysteresis path. Keeping rejection accounting outside it
    /// ensures an idle process under pressure does not manufacture "rejected
    /// query" metrics merely because the watchdog ticked.
    pub(crate) fn reclaim_if_needed(&self) -> Option<usize> {
        // Keep the default-disabled path genuinely free: sampling RSS reads
        // procfs / asks the OS, and must not become per-query overhead for
        // deployments that did not opt into this governor.
        if self.max_bytes == 0 {
            return None;
        }
        let Some(mut resident) = sample_resident_bytes() else {
            // Unsupported/temporarily unavailable sampling must not turn an
            // otherwise healthy server into a permanent fail-closed outage.
            return None;
        };
        self.resident_bytes.store(resident, Ordering::Relaxed);

        let new_pressure_episode = resident >= self.reclaim_bytes
            && self
                .pressure_active
                .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
                .is_ok();
        let reclaim = self.try_begin_reclaim(new_pressure_episode, resident >= self.max_bytes);
        if reclaim {
            namidb_storage::clear_shared_caches();
            trim_allocator();
            self.reclaim_events.fetch_add(1, Ordering::Relaxed);
            if let Some(after) = sample_resident_bytes() {
                resident = after;
                self.resident_bytes.store(after, Ordering::Relaxed);
            }
            self.finish_reclaim(resident);
        }

        if resident <= self.resume_bytes {
            self.pressure_active.store(false, Ordering::Release);
        }

        Some(resident)
    }

    /// Start the one process-wide pressure watchdog.
    ///
    /// Cache reclamation and `malloc_trim` can walk/deallocate a large working
    /// set, so each pass runs on Tokio's blocking pool rather than occupying a
    /// serving worker. Missed ticks are skipped and every pass is awaited,
    /// guaranteeing there is never more than one watchdog job in flight.
    pub(crate) fn spawn_watchdog(
        self: Arc<Self>,
        mut shutdown: tokio::sync::watch::Receiver<bool>,
    ) -> tokio::task::JoinHandle<()> {
        tokio::spawn(async move {
            let mut tick = tokio::time::interval(WATCHDOG_INTERVAL);
            tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
            loop {
                {
                    // `watch::Receiver::wait_for` returns a borrowed `Ref`.
                    // Keeping that value in `select!` while the tick branch
                    // awaits `spawn_blocking` makes the watchdog future
                    // non-`Send`. Consume the current value before selecting
                    // and wait only for the next change notification below.
                    let stop = *shutdown.borrow_and_update();
                    if stop {
                        break;
                    }
                }
                tokio::select! {
                    biased;
                    changed = shutdown.changed() => {
                        if changed.is_err() {
                            break;
                        }
                    },
                    _ = tick.tick() => {
                        if let Some(initial) = self.observe_resident() {
                            let _ = self.reclaim_off_worker(initial).await;
                        }
                    }
                }
            }
        })
    }

    /// Serialize operator-requested flushes process-wide.
    ///
    /// A flush is deliberately allowed while Cypher admission is closed
    /// because it can release a large memtable. Its SST build temporarily
    /// amplifies memory, though, so multi-tenant callers must never run several
    /// relief flushes concurrently.
    pub(crate) async fn admin_flush_permit(&self) -> tokio::sync::SemaphorePermit<'_> {
        self.admin_flush_gate
            .acquire()
            .await
            .expect("admin flush semaphore is never closed")
    }

    fn try_begin_reclaim(&self, new_pressure_episode: bool, at_hard_limit: bool) -> bool {
        let now = Instant::now();
        let mut state = self
            .reclaim_state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if state.running {
            return false;
        }
        let hard_retry_due = at_hard_limit
            && state
                .last_started
                .is_none_or(|last| now.duration_since(last) >= HARD_RECLAIM_COOLDOWN);
        if !new_pressure_episode && !hard_retry_due {
            return false;
        }
        state.running = true;
        state.last_started = Some(now);
        true
    }

    fn finish_reclaim(&self, resident: usize) {
        if resident < self.reclaim_bytes {
            // This pass actually relieved the pressure episode. Re-arm at the
            // threshold it crossed so repopulated caches can trigger another
            // pass; the 80% hysteresis remains for natural (non-reclaim) decay.
            self.pressure_active.store(false, Ordering::Release);
        }
        let mut state = self
            .reclaim_state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        state.running = false;
    }

    /// Refresh the resident-byte gauge without applying admission/reclaim.
    pub fn sample(&self) -> Option<usize> {
        let resident = sample_resident_bytes()?;
        self.resident_bytes.store(resident, Ordering::Relaxed);
        Some(resident)
    }

    pub fn max_bytes(&self) -> usize {
        self.max_bytes
    }

    pub fn resident_bytes(&self) -> usize {
        self.resident_bytes.load(Ordering::Relaxed)
    }

    pub fn reclaim_events(&self) -> u64 {
        self.reclaim_events.load(Ordering::Relaxed)
    }

    pub fn rejected_queries(&self) -> u64 {
        self.rejected_queries.load(Ordering::Relaxed)
    }

    pub fn over_limit(&self) -> bool {
        self.max_bytes > 0 && self.resident_bytes() >= self.max_bytes
    }

    #[cfg(test)]
    fn observe_for_test(&self, resident: usize) -> Result<bool, MemoryPressure> {
        self.resident_bytes.store(resident, Ordering::Relaxed);
        if self.max_bytes == 0 {
            return Ok(false);
        }
        let reclaim = resident >= self.reclaim_bytes
            && self
                .pressure_active
                .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
                .is_ok();
        if resident <= self.resume_bytes {
            self.pressure_active.store(false, Ordering::Release);
        }
        if resident >= self.max_bytes {
            return Err(MemoryPressure {
                resident_bytes: resident,
                max_bytes: self.max_bytes,
            });
        }
        Ok(reclaim)
    }
}

/// Owns the queue-to-completion scheduling slot for one blocking reclaim job.
struct ReclaimJobPermit(Arc<MemoryGovernor>);

impl Drop for ReclaimJobPermit {
    fn drop(&mut self) {
        self.0.reclaim_job_scheduled.store(false, Ordering::Release);
        self.0.reclaim_completed.send_modify(|generation| {
            *generation = generation.wrapping_add(1);
        });
    }
}

fn percent(value: usize, percentage: usize) -> usize {
    ((value as u128 * percentage as u128) / 100).min(usize::MAX as u128) as usize
}

fn sample_resident_bytes() -> Option<usize> {
    memory_stats::memory_stats().map(|stats| stats.physical_mem)
}

#[cfg(all(target_os = "linux", target_env = "gnu"))]
fn trim_allocator() {
    // SAFETY: `malloc_trim(0)` has no pointer preconditions and only asks the
    // process-global glibc allocator to release wholly free pages.
    unsafe {
        libc::malloc_trim(0);
    }
}

#[cfg(not(all(target_os = "linux", target_env = "gnu")))]
fn trim_allocator() {}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn disabled_governor_never_rejects() {
        let governor = Arc::new(MemoryGovernor::new(0));
        assert_eq!(governor.admit_query().await, Ok(()));
        assert_eq!(
            governor.resident_bytes(),
            0,
            "disabled admission must not sample or mutate the RSS gauge"
        );
        assert_eq!(governor.observe_for_test(usize::MAX), Ok(false));
    }

    #[test]
    fn reclaim_has_hysteresis_and_limit_rejects() {
        let governor = MemoryGovernor::new(1_000);
        assert_eq!(governor.observe_for_test(899), Ok(false));
        assert_eq!(governor.observe_for_test(900), Ok(true));
        assert_eq!(governor.observe_for_test(950), Ok(false));
        assert_eq!(
            governor.observe_for_test(1_000),
            Err(MemoryPressure {
                resident_bytes: 1_000,
                max_bytes: 1_000,
            })
        );
        assert_eq!(governor.observe_for_test(800), Ok(false));
        assert_eq!(governor.observe_for_test(900), Ok(true));
    }

    #[test]
    fn reclaim_is_single_flight_and_hard_retries_are_cooled_down() {
        let governor = MemoryGovernor::new(1_000);
        assert!(governor.try_begin_reclaim(true, false));
        assert!(
            !governor.try_begin_reclaim(false, true),
            "a second caller must not overlap an active clear"
        );
        governor.finish_reclaim(950);
        assert!(
            !governor.try_begin_reclaim(false, true),
            "hard-limit retries are rate limited"
        );

        {
            let mut state = governor.reclaim_state.lock().unwrap();
            state.last_started = Some(Instant::now() - HARD_RECLAIM_COOLDOWN);
        }
        assert!(governor.try_begin_reclaim(false, true));
        governor.pressure_active.store(true, Ordering::Release);
        governor.finish_reclaim(850);
        assert!(
            !governor.pressure_active.load(Ordering::Acquire),
            "an effective clear below 90% must re-arm pressure detection"
        );
    }

    #[tokio::test]
    async fn blocking_reclaim_jobs_are_pre_gated_across_queue_and_cooldown() {
        let governor = Arc::new(MemoryGovernor::new(1_000));
        let permit = governor
            .try_schedule_reclaim(900)
            .expect("first pressure observer schedules one job");
        assert!(
            governor.try_schedule_reclaim(900).is_none(),
            "a burst cannot enqueue a second blocking job before the first starts"
        );
        let mut completed = governor.reclaim_completed.subscribe();
        drop(permit);
        tokio::time::timeout(Duration::from_secs(1), completed.changed())
            .await
            .expect("dropping a queued/running permit notifies hard-limit waiters")
            .expect("the governor retains its completion sender");

        governor.pressure_active.store(true, Ordering::Release);
        assert!(
            governor.try_schedule_reclaim(950).is_none(),
            "one clear serves the active soft-pressure episode"
        );
        {
            let mut state = governor.reclaim_state.lock().unwrap();
            state.last_started = Some(Instant::now());
        }
        assert!(
            governor.try_schedule_reclaim(1_000).is_none(),
            "hard-pressure retries respect the cooldown before queueing"
        );
        {
            let mut state = governor.reclaim_state.lock().unwrap();
            state.last_started = Some(Instant::now() - HARD_RECLAIM_COOLDOWN);
        }
        assert!(
            governor.try_schedule_reclaim(1_000).is_some(),
            "one hard-pressure retry is admitted after the cooldown"
        );
    }

    #[test]
    fn maintenance_reclaim_never_counts_as_a_rejected_query() {
        let governor = MemoryGovernor::new(1);
        let _ = governor.reclaim_if_needed();
        assert_eq!(governor.rejected_queries(), 0);
    }

    #[tokio::test]
    async fn watchdog_samples_without_requests_and_stops_on_shutdown() {
        let sampling_supported = sample_resident_bytes().is_some();
        let governor = Arc::new(MemoryGovernor::new(usize::MAX));
        let (shutdown_tx, shutdown_rx) = tokio::sync::watch::channel(false);
        let handle = Arc::clone(&governor).spawn_watchdog(shutdown_rx);

        if sampling_supported {
            tokio::time::timeout(Duration::from_secs(2), async {
                while governor.resident_bytes() == 0 {
                    tokio::task::yield_now().await;
                }
            })
            .await
            .expect("watchdog must sample RSS without a foreground request");
        }
        assert_eq!(governor.rejected_queries(), 0);

        shutdown_tx.send(true).unwrap();
        tokio::time::timeout(Duration::from_secs(1), handle)
            .await
            .expect("watchdog must observe process shutdown")
            .expect("watchdog task must exit cleanly");
    }

    #[tokio::test]
    async fn admin_flush_gate_is_process_wide_single_flight() {
        let governor = MemoryGovernor::new(0);
        let first = governor.admin_flush_permit().await;
        assert!(
            tokio::time::timeout(Duration::from_millis(10), governor.admin_flush_permit())
                .await
                .is_err(),
            "a second relief flush must wait for the process-wide permit"
        );
        drop(first);
        let _second = tokio::time::timeout(Duration::from_secs(1), governor.admin_flush_permit())
            .await
            .expect("dropping the first permit must unblock the next flush");
    }
}
