//! Automatic property indexes.
//!
//! A read statement whose search filter, or pushed-down `WHERE` equality or
//! range, had to scan a label because the property has no index reports the
//! `(label, property)` pair ([`namidb_query::AutoIndexHints`]). This module
//! turns those hints into ordinary property indexes (the existing equality
//! sidecar format, flagged `auto_indexed` and listed as
//! `auto_index_<label>_<property>`), under a budget:
//!
//! - one build at a time for the whole process: the next index is declared
//!   only after the previous one is online (every node SST of its label
//!   carries the sidecar) or its wait timed out;
//! - at most `NAMIDB_AUTO_INDEX_MAX_PER_NAMESPACE` (default 8) automatic
//!   indexes per namespace;
//! - only labels with at least `NAMIDB_AUTO_INDEX_MIN_NODES` (default 10000)
//!   nodes: a scan of a smaller label is already cheap;
//! - never an index that would change the catalog signature of a vector or
//!   full-text index (a string/bool property of a searched label): that
//!   retires and rebuilds the search index, which a query must not trigger.
//!   Those cases are logged and counted, and stay with an explicit
//!   `CREATE INDEX`.
//!
//! `NAMIDB_AUTO_INDEX=0` (or `false`/`off`) disables the feature.

use std::collections::{HashSet, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::Duration;

use namidb_storage::AutoIndexOutcome;
use tracing::{info, warn};

use std::sync::Arc;

use namidb_storage::{SnapshotCell, WriterSession};

use crate::maintenance::{request_compaction, CompactionScheduler};
use crate::metrics::{CompactionTrigger, Metrics};
use crate::recovery::WriterHealth;

/// The handles of one namespace an automatic build needs.
#[derive(Clone)]
pub(crate) struct Target {
    pub namespace: String,
    pub writer: Arc<tokio::sync::Mutex<WriterSession>>,
    pub snapshot: Arc<SnapshotCell>,
    pub compaction_scheduler: Arc<CompactionScheduler>,
    pub writer_health: Arc<WriterHealth>,
    pub metrics: Arc<Metrics>,
}

const DEFAULT_MAX_PER_NAMESPACE: usize = 8;
const DEFAULT_MIN_NODES: u64 = 10_000;
/// Pending requests kept at most; later hints are dropped (they come back
/// with the next statement that scans).
const MAX_QUEUED: usize = 64;
/// How long one build may hold the slot before the next one is declared.
const DEFAULT_BUILD_WAIT: Duration = Duration::from_secs(30 * 60);
const POLL: Duration = Duration::from_secs(2);

fn env_flag_enabled() -> bool {
    !matches!(
        std::env::var("NAMIDB_AUTO_INDEX")
            .map(|value| value.trim().to_ascii_lowercase())
            .as_deref(),
        Ok("0" | "false" | "off" | "no")
    )
}

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|value| value.trim().parse().ok())
        .unwrap_or(default)
}

/// Counters rendered on `/v0/metrics`.
#[derive(Default)]
pub(crate) struct AutoIndexCounters {
    pub created: AtomicU64,
    pub online: AtomicU64,
    pub skipped_search_rebuild: AtomicU64,
    pub skipped_cap: AtomicU64,
    pub skipped_other: AtomicU64,
    pub failed: AtomicU64,
}

pub(crate) fn counters() -> &'static AutoIndexCounters {
    static COUNTERS: OnceLock<AutoIndexCounters> = OnceLock::new();
    COUNTERS.get_or_init(AutoIndexCounters::default)
}

struct Job {
    state: Target,
    label: String,
    property: String,
}

#[derive(Default)]
struct Queue {
    jobs: VecDeque<Job>,
    /// `(namespace, label, property)` queued or decided in this process, so
    /// a hint repeated by every statement costs one set lookup.
    seen: HashSet<(String, String, String)>,
    worker_running: bool,
}

fn queue() -> &'static Mutex<Queue> {
    static QUEUE: OnceLock<Mutex<Queue>> = OnceLock::new();
    QUEUE.get_or_init(Mutex::default)
}

/// Whether hints should be collected at all.
pub(crate) fn enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(env_flag_enabled)
}

/// Queue the statement's unindexed filter properties of `state`'s namespace.
/// Cheap and non-blocking; the build runs on a background task.
pub(crate) fn offer(state: &Target, hints: Vec<(String, String)>) {
    if hints.is_empty() || !enabled() {
        return;
    }
    let mut queue = queue().lock().unwrap_or_else(|e| e.into_inner());
    for (label, property) in hints {
        let key = (state.namespace.clone(), label.clone(), property.clone());
        if queue.jobs.len() >= MAX_QUEUED || queue.seen.contains(&key) {
            continue;
        }
        queue.seen.insert(key);
        queue.jobs.push_back(Job {
            state: state.clone(),
            label,
            property,
        });
    }
    if !queue.worker_running && !queue.jobs.is_empty() {
        queue.worker_running = true;
        tokio::spawn(worker());
    }
}

async fn worker() {
    loop {
        let job = {
            let mut queue = queue().lock().unwrap_or_else(|e| e.into_inner());
            match queue.jobs.pop_front() {
                Some(job) => job,
                None => {
                    queue.worker_running = false;
                    return;
                }
            }
        };
        let forget = run(&job).await;
        if forget {
            // Transient refusal (writer degraded, I/O): let a later hint
            // retry it.
            queue()
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .seen
                .remove(&(
                    job.state.namespace.clone(),
                    job.label.clone(),
                    job.property.clone(),
                ));
        }
    }
}

/// Build one index. Returns `true` when the job should be forgotten so a
/// later hint can retry it.
async fn run(job: &Job) -> bool {
    let state = &job.state;
    let namespace = state.namespace.as_str();
    let (label, property) = (job.label.as_str(), job.property.as_str());
    {
        let pinned = state.snapshot.load();
        let snapshot = pinned.borrow();
        let schema = &snapshot.manifest().manifest.schema;
        let auto_count = schema
            .labels
            .values()
            .flat_map(|def| def.properties.iter())
            .filter(|p| p.auto_indexed)
            .count();
        let cap = env_usize(
            "NAMIDB_AUTO_INDEX_MAX_PER_NAMESPACE",
            DEFAULT_MAX_PER_NAMESPACE,
        );
        if auto_count >= cap {
            counters().skipped_cap.fetch_add(1, Ordering::Relaxed);
            info!(
                namespace,
                label, property, cap, "auto-index: namespace cap reached"
            );
            return false;
        }
        let min_nodes = env_usize("NAMIDB_AUTO_INDEX_MIN_NODES", DEFAULT_MIN_NODES as usize) as u64;
        if snapshot
            .metadata_node_count(Some(label))
            .is_some_and(|count| count < min_nodes)
        {
            counters().skipped_other.fetch_add(1, Ordering::Relaxed);
            return false;
        }
    }
    if state.writer_health.persistence_degraded_reason().is_some() {
        return true;
    }
    let outcome = {
        let mut writer = state.writer.lock().await;
        let outcome = writer.create_auto_property_index(label, property).await;
        match &outcome {
            Ok(AutoIndexOutcome::Created { .. }) => {
                state.snapshot.store(writer.owned_snapshot());
            }
            Err(error) => {
                crate::recovery::recover_writer_if_needed(
                    &mut writer,
                    &state.snapshot,
                    &state.writer_health,
                    namespace,
                    error,
                )
                .await;
            }
            Ok(_) => {}
        }
        outcome
    };
    match outcome {
        Ok(AutoIndexOutcome::Created { version }) => {
            counters().created.fetch_add(1, Ordering::Relaxed);
            info!(
                namespace,
                label, property, version, "auto-index: created; building"
            );
            let _ = request_compaction(
                &state.compaction_scheduler,
                CompactionTrigger::Ddl,
                &state.writer,
                &state.snapshot,
                &state.writer_health,
                namespace,
                &state.metrics,
                None,
            );
            wait_online(state, label, property).await;
            false
        }
        Ok(AutoIndexOutcome::WouldRebuildSearch(indexes)) => {
            counters()
                .skipped_search_rebuild
                .fetch_add(1, Ordering::Relaxed);
            warn!(
                namespace,
                label,
                property,
                ?indexes,
                "auto-index: not created, an index on this property would rebuild these \
                 search indexes; create it explicitly when a rebuild is acceptable"
            );
            false
        }
        Ok(_) => {
            counters().skipped_other.fetch_add(1, Ordering::Relaxed);
            false
        }
        Err(error) => {
            counters().failed.fetch_add(1, Ordering::Relaxed);
            warn!(namespace, label, property, %error, "auto-index: declaring the index failed");
            true
        }
    }
}

/// Hold the build slot until every node SST of `label` carries the sidecar
/// (re-requesting compaction while it is missing) or the wait times out.
async fn wait_online(state: &Target, label: &str, property: &str) {
    let wait = Duration::from_secs(env_usize(
        "NAMIDB_AUTO_INDEX_BUILD_WAIT_SECS",
        DEFAULT_BUILD_WAIT.as_secs() as usize,
    ) as u64);
    let started = std::time::Instant::now();
    let mut last_request = started;
    loop {
        let population = {
            let pinned = state.snapshot.load();
            let snapshot = pinned.borrow();
            namidb_storage::property_index_population(
                &snapshot.manifest().manifest,
                label,
                property,
            )
        };
        if population >= 1.0 {
            counters().online.fetch_add(1, Ordering::Relaxed);
            info!(
                namespace = state.namespace.as_str(),
                label,
                property,
                elapsed_ms = started.elapsed().as_millis() as u64,
                "auto-index: online"
            );
            return;
        }
        if started.elapsed() >= wait {
            warn!(
                namespace = state.namespace.as_str(),
                label, property, population, "auto-index: still building; releasing the slot"
            );
            return;
        }
        if last_request.elapsed() >= Duration::from_secs(30) {
            last_request = std::time::Instant::now();
            let _ = request_compaction(
                &state.compaction_scheduler,
                CompactionTrigger::Ddl,
                &state.writer,
                &state.snapshot,
                &state.writer_health,
                state.namespace.as_str(),
                &state.metrics,
                None,
            );
        }
        tokio::time::sleep(POLL).await;
    }
}
