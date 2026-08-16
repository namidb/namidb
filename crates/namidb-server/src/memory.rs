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
use std::{collections::HashSet, path::PathBuf};

/// Disabled by default for backward compatibility.
pub const DEFAULT_MEMORY_MAX_BYTES: usize = 0;
/// Fraction of a finite cgroup hard limit used by the `auto` admission rail.
///
/// The remaining ten percent is deliberately outside NamiDB's admission
/// ceiling. It covers allocations already in flight, allocator/runtime
/// overhead, and the delay between two 500 ms watchdog samples before the
/// kernel's OOM killer reaches the container limit.
pub const AUTO_MEMORY_LIMIT_PERCENT: usize = 90;

const CGROUP_ROOT: &str = "/sys/fs/cgroup";
const SELF_CGROUP: &str = "/proc/self/cgroup";
// cgroup v1 reports values close to i64::MAX for "unlimited" instead of a
// textual `max`. No realistic container allocation belongs in this range.
const CGROUP_V1_UNLIMITED_FLOOR: u128 = 1u128 << 60;

const RECLAIM_PERCENT: usize = 90;
const RESUME_PERCENT: usize = 80;
const HARD_RECLAIM_COOLDOWN: Duration = Duration::from_secs(1);
const WATCHDOG_INTERVAL: Duration = Duration::from_millis(500);

/// Parse an exact-byte process ceiling or resolve `auto` from the current
/// container's finite cgroup memory limit.
///
/// `auto` intentionally fails startup when the process has no finite cgroup
/// limit. Falling back to a percentage of host RAM would be unsafe on a shared
/// machine and silently resolving to zero would disable the very safety rail
/// the operator requested.
pub fn resolve_memory_max_bytes(raw: &str) -> Result<usize, String> {
    let raw = raw.trim();
    if raw.eq_ignore_ascii_case("auto") {
        let hard_limit = finite_cgroup_memory_limit_bytes().ok_or_else(|| {
            "NAMIDB_MEMORY_MAX_BYTES=auto requires a finite cgroup memory limit; \
             set an exact byte count or start the container with --memory/Compose mem_limit"
                .to_string()
        })?;
        return Ok(percent(hard_limit, AUTO_MEMORY_LIMIT_PERCENT));
    }
    raw.parse::<usize>().map_err(|error| {
        format!("NAMIDB_MEMORY_MAX_BYTES must be an exact byte count, 0, or auto: {error}")
    })
}

/// Current finite cgroup hard limit, if one is configured.
///
/// cgroup v2 is preferred. The v1 fallback keeps the official image safe on
/// older Docker hosts. Errors and the respective unlimited sentinels return
/// `None`; callers decide whether absence is fatal.
pub fn finite_cgroup_memory_limit_bytes() -> Option<usize> {
    cgroup_memory_limit_paths()
        .into_iter()
        .find_map(|(path, v1)| {
            std::fs::read_to_string(path)
                .ok()
                .and_then(|raw| parse_cgroup_limit(&raw, v1))
        })
}

fn parse_cgroup_limit(raw: &str, v1: bool) -> Option<usize> {
    let raw = raw.trim();
    if raw.is_empty() || raw.eq_ignore_ascii_case("max") {
        return None;
    }
    let bytes = raw.parse::<u128>().ok()?;
    if bytes == 0 || (v1 && bytes >= CGROUP_V1_UNLIMITED_FLOOR) {
        return None;
    }
    usize::try_from(bytes).ok()
}

fn cgroup_memory_limit_paths() -> Vec<(PathBuf, bool)> {
    let mut candidates = Vec::new();
    if let Ok(membership) = std::fs::read_to_string(SELF_CGROUP) {
        for line in membership.lines() {
            let mut fields = line.splitn(3, ':');
            let Some(_hierarchy) = fields.next() else {
                continue;
            };
            let Some(controllers) = fields.next() else {
                continue;
            };
            let Some(relative) = fields.next() else {
                continue;
            };
            if controllers.is_empty() {
                if let Some(mut directory) = cgroup_directory(CGROUP_ROOT, relative) {
                    directory.push("memory.max");
                    candidates.push((directory, false));
                }
            } else if controllers
                .split(',')
                .any(|controller| controller == "memory")
            {
                if let Some(mut directory) =
                    cgroup_directory(&format!("{CGROUP_ROOT}/memory"), relative)
                {
                    directory.push("memory.limit_in_bytes");
                    candidates.push((directory, true));
                }
            }
        }
    }

    // Container cgroup namespaces commonly expose the current group directly
    // at the mount root. These fallbacks also cover hosts where
    // `/proc/self/cgroup` is hidden but the controller file is mounted.
    candidates.push((PathBuf::from(CGROUP_ROOT).join("memory.max"), false));
    candidates.push((
        PathBuf::from(CGROUP_ROOT)
            .join("memory")
            .join("memory.limit_in_bytes"),
        true,
    ));

    let mut seen = HashSet::new();
    candidates.retain(|(path, _)| seen.insert(path.clone()));
    candidates
}

fn cgroup_directory(root: &str, relative: &str) -> Option<PathBuf> {
    let mut out = PathBuf::from(root);
    for component in std::path::Path::new(relative).components() {
        match component {
            std::path::Component::RootDir | std::path::Component::CurDir => {}
            std::path::Component::Normal(component) => out.push(component),
            // The membership path is kernel-owned, nevertheless refuse
            // traversal rather than allowing a malformed mount/proc fixture
            // to make startup read an arbitrary host file.
            std::path::Component::ParentDir | std::path::Component::Prefix(_) => return None,
        }
    }
    Some(out)
}

#[derive(Debug, Default)]
struct ReclaimState {
    running: bool,
    last_started: Option<Instant>,
}

/// A rejected admission with the current measured resident set and any
/// conservative working-set headroom requested before allocating.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MemoryPressure {
    pub resident_bytes: usize,
    pub requested_headroom_bytes: usize,
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
    /// Working-set headroom promised to authenticated Bolt requests that have
    /// passed RSS admission but have not finished decode/execution yet.
    reserved_headroom_bytes: AtomicUsize,
    admin_flush_gate: Arc<tokio::sync::Semaphore>,
}

/// RAII reservation against [`MemoryGovernor`]'s total-memory ceiling.
///
/// The session retains this from pre-decode admission through request
/// handling. Cancellation and every error path release it automatically.
#[derive(Debug)]
pub struct MemoryHeadroomReservation {
    governor: Arc<MemoryGovernor>,
    bytes: usize,
}

impl Drop for MemoryHeadroomReservation {
    fn drop(&mut self) {
        if self.bytes > 0 {
            let previous = self
                .governor
                .reserved_headroom_bytes
                .fetch_sub(self.bytes, Ordering::AcqRel);
            debug_assert!(previous >= self.bytes);
        }
    }
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
            reserved_headroom_bytes: AtomicUsize::new(0),
            admin_flush_gate: Arc::new(tokio::sync::Semaphore::new(1)),
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

        if let Some(pressure) = Self::projected_pressure(resident, 0, self.max_bytes) {
            self.rejected_queries.fetch_add(1, Ordering::Relaxed);
            return Err(pressure);
        }
        Ok(())
    }

    /// Atomically reserve projected decode/runtime headroom after normal RSS
    /// sampling and reclaim.
    ///
    /// The compare/exchange includes reservations held by every concurrent
    /// request, so multiple individually admissible frames cannot all race
    /// past the same remaining process headroom.
    pub async fn reserve_query_headroom(
        self: &Arc<Self>,
        additional_bytes: usize,
    ) -> Result<MemoryHeadroomReservation, MemoryPressure> {
        if self.max_bytes == 0 {
            return Ok(MemoryHeadroomReservation {
                governor: Arc::clone(self),
                bytes: 0,
            });
        }
        let Some(initial) = self.observe_resident() else {
            return Ok(MemoryHeadroomReservation {
                governor: Arc::clone(self),
                bytes: 0,
            });
        };
        let resident = self.reclaim_off_worker(initial).await;
        self.try_reserve_observed(resident, additional_bytes)
    }

    fn try_reserve_observed(
        self: &Arc<Self>,
        resident_bytes: usize,
        additional_bytes: usize,
    ) -> Result<MemoryHeadroomReservation, MemoryPressure> {
        loop {
            let reserved = self.reserved_headroom_bytes.load(Ordering::Acquire);
            let projected_headroom = reserved.saturating_add(additional_bytes);
            if let Some(pressure) =
                Self::projected_pressure(resident_bytes, projected_headroom, self.max_bytes)
            {
                self.rejected_queries.fetch_add(1, Ordering::Relaxed);
                return Err(pressure);
            }
            let next = reserved.checked_add(additional_bytes).ok_or_else(|| {
                self.rejected_queries.fetch_add(1, Ordering::Relaxed);
                MemoryPressure {
                    resident_bytes,
                    requested_headroom_bytes: usize::MAX,
                    max_bytes: self.max_bytes,
                }
            })?;
            match self.reserved_headroom_bytes.compare_exchange_weak(
                reserved,
                next,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => {
                    return Ok(MemoryHeadroomReservation {
                        governor: Arc::clone(self),
                        bytes: additional_bytes,
                    });
                }
                Err(_) => continue,
            }
        }
    }

    fn projected_pressure(
        resident_bytes: usize,
        requested_headroom_bytes: usize,
        max_bytes: usize,
    ) -> Option<MemoryPressure> {
        (max_bytes > 0 && resident_bytes.saturating_add(requested_headroom_bytes) >= max_bytes)
            .then_some(MemoryPressure {
                resident_bytes,
                requested_headroom_bytes,
                max_bytes,
            })
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
    /// relief flushes concurrently. The owned permit can move into a
    /// post-flush blocking allocator trim, keeping the gate closed even if the
    /// HTTP request waiting on that trim is cancelled.
    pub(crate) async fn admin_flush_permit(self: &Arc<Self>) -> tokio::sync::OwnedSemaphorePermit {
        Arc::clone(&self.admin_flush_gate)
            .acquire_owned()
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

    /// Projected request working sets currently promised by admission guards.
    ///
    /// Exposed inside the server crate for deterministic protocol tests and
    /// metrics; it is not a resident-byte measurement.
    pub fn reserved_headroom_bytes(&self) -> usize {
        self.reserved_headroom_bytes.load(Ordering::Acquire)
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
                requested_headroom_bytes: 0,
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
pub(crate) fn trim_allocator() {
    // SAFETY: `malloc_trim(0)` has no pointer preconditions and only asks the
    // process-global glibc allocator to release wholly free pages.
    unsafe {
        libc::malloc_trim(0);
    }
}

#[cfg(not(all(target_os = "linux", target_env = "gnu")))]
pub(crate) fn trim_allocator() {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn memory_limit_parser_accepts_exact_bytes_and_rejects_ambiguous_units() {
        assert_eq!(resolve_memory_max_bytes("0"), Ok(0));
        assert_eq!(resolve_memory_max_bytes(" 3758096384 "), Ok(3_758_096_384));
        let error = resolve_memory_max_bytes("4GiB").unwrap_err();
        assert!(error.contains("exact byte count, 0, or auto"));
    }

    #[test]
    fn cgroup_limit_parser_rejects_unlimited_and_overflow_sentinels() {
        assert_eq!(parse_cgroup_limit("max\n", false), None);
        assert_eq!(parse_cgroup_limit("0", false), None);
        assert_eq!(
            parse_cgroup_limit("4294967296", false),
            usize::try_from(4_294_967_296u64).ok()
        );
        assert_eq!(
            parse_cgroup_limit(&(CGROUP_V1_UNLIMITED_FLOOR - 1).to_string(), true),
            usize::try_from(CGROUP_V1_UNLIMITED_FLOOR - 1).ok()
        );
        assert_eq!(
            parse_cgroup_limit(&CGROUP_V1_UNLIMITED_FLOOR.to_string(), true),
            None
        );
        assert_eq!(
            parse_cgroup_limit(&(usize::MAX as u128 + 1).to_string(), false),
            None
        );
    }

    #[test]
    fn cgroup_membership_paths_stay_beneath_the_controller_mount() {
        assert_eq!(
            cgroup_directory("/sys/fs/cgroup", "/tenant.slice/namidb.service"),
            Some(PathBuf::from("/sys/fs/cgroup/tenant.slice/namidb.service"))
        );
        assert_eq!(
            cgroup_directory("/sys/fs/cgroup", "/"),
            Some(PathBuf::from("/sys/fs/cgroup"))
        );
        assert_eq!(
            cgroup_directory("/sys/fs/cgroup", "../../etc"),
            None,
            "kernel membership text must never be allowed to escape the mount"
        );
    }

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
                requested_headroom_bytes: 0,
                max_bytes: 1_000,
            })
        );
        assert_eq!(governor.observe_for_test(800), Ok(false));
        assert_eq!(governor.observe_for_test(900), Ok(true));
    }

    #[test]
    fn projected_headroom_rejects_before_allocation_and_saturates() {
        assert_eq!(
            MemoryGovernor::projected_pressure(800, 200, 1_000),
            Some(MemoryPressure {
                resident_bytes: 800,
                requested_headroom_bytes: 200,
                max_bytes: 1_000,
            }),
            "reaching the ceiling with the projected decode working set must reject"
        );
        assert!(
            MemoryGovernor::projected_pressure(800, 199, 1_000).is_none(),
            "one byte below the projected ceiling remains admissible"
        );
        assert_eq!(
            MemoryGovernor::projected_pressure(1, usize::MAX, usize::MAX),
            Some(MemoryPressure {
                resident_bytes: 1,
                requested_headroom_bytes: usize::MAX,
                max_bytes: usize::MAX,
            }),
            "overflow must conservatively saturate instead of wrapping"
        );
        assert!(
            MemoryGovernor::projected_pressure(usize::MAX, usize::MAX, 0).is_none(),
            "a disabled governor remains disabled regardless of headroom"
        );
    }

    #[test]
    fn headroom_reservations_are_atomic_and_drop_readmits() {
        let governor = Arc::new(MemoryGovernor::new(1_000));
        let first = governor
            .try_reserve_observed(100, 400)
            .expect("first projected working set fits");
        assert_eq!(
            governor.reserved_headroom_bytes.load(Ordering::Acquire),
            400
        );

        let rejected = governor
            .try_reserve_observed(100, 500)
            .expect_err("concurrent reservations must be included in projection");
        assert_eq!(rejected.resident_bytes, 100);
        assert_eq!(rejected.requested_headroom_bytes, 900);

        drop(first);
        assert_eq!(governor.reserved_headroom_bytes.load(Ordering::Acquire), 0);
        let second = governor
            .try_reserve_observed(100, 500)
            .expect("dropping the first guard must readmit the second");
        assert_eq!(
            governor.reserved_headroom_bytes.load(Ordering::Acquire),
            500
        );
        drop(second);

        assert!(governor.try_reserve_observed(1, usize::MAX).is_err());
        assert_eq!(
            governor.reserved_headroom_bytes.load(Ordering::Acquire),
            0,
            "overflow rejection must not leak a reservation"
        );
    }

    #[tokio::test]
    async fn disabled_headroom_reservation_is_a_zero_byte_guard() {
        let governor = Arc::new(MemoryGovernor::new(0));
        let guard = governor
            .reserve_query_headroom(usize::MAX)
            .await
            .expect("disabled governor never rejects");
        assert_eq!(guard.bytes, 0);
        assert_eq!(governor.reserved_headroom_bytes.load(Ordering::Acquire), 0);
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
        let governor = Arc::new(MemoryGovernor::new(0));
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

    #[tokio::test]
    async fn owned_admin_flush_permit_survives_cancelled_trim_waiter() {
        let governor = Arc::new(MemoryGovernor::new(0));
        let first = governor.admin_flush_permit().await;
        let (started_tx, started_rx) = tokio::sync::oneshot::channel();
        let (release_tx, release_rx) = std::sync::mpsc::channel();
        let trim = tokio::task::spawn_blocking(move || {
            let _first = first;
            let _ = started_tx.send(());
            release_rx
                .recv()
                .expect("test must release the simulated trim");
        });
        started_rx.await.expect("simulated trim must start");

        // Dropping/aborting a waiter does not cancel spawn_blocking. The owned
        // permit lives in the closure, so a disconnected admin request cannot
        // admit a second flush while its allocator trim is still running.
        trim.abort();
        assert!(
            tokio::time::timeout(Duration::from_millis(10), governor.admin_flush_permit())
                .await
                .is_err(),
            "the detached blocking trim must retain the process-wide gate"
        );
        release_tx.send(()).unwrap();
        let _second = tokio::time::timeout(Duration::from_secs(1), governor.admin_flush_permit())
            .await
            .expect("finishing the detached trim must release the gate");
    }
}
