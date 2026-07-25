//! Shared background-compaction scheduling.
//!
//! Single-tenant and registry namespaces use the same two-phase pass:
//! capture a basis briefly under the writer mutex, prepare every selected
//! bucket off-lock, then re-acquire the mutex only to validate and publish
//! the manifest.
//!
//! [`CompactionScheduler`] is a per-namespace single-flight scheduler. A
//! trigger either starts the sole worker, reserves the sole pending follow-up,
//! or is coalesced into that follow-up. Pending work stores only its trigger:
//! the worker captures a fresh basis when the follow-up actually starts. This
//! both bounds task growth under a flush storm and prevents a stale
//! compaction basis from retaining a large manifest while it waits.

use std::sync::{Arc, Mutex as StdMutex};
use std::time::Instant;

use namidb_storage::{CompactionOutcome, SnapshotCell, WriterSession};
use tokio::sync::{watch, Mutex, RwLock, RwLockReadGuard, RwLockWriteGuard};
use tokio::task::JoinHandle;
use tracing::{error, info};

use crate::metrics::{
    CompactionPhase, CompactionStatus, CompactionTrigger, Metrics, WriterLockKind,
};
use crate::recovery::{self, WriterHealth};

#[derive(Debug, Default)]
struct SchedulerState {
    worker_active: bool,
    pending: Option<CompactionTrigger>,
    #[cfg(test)]
    workers_started: u64,
    #[cfg(test)]
    passes_started: u64,
}

/// Per-namespace single-flight compaction scheduler.
///
/// At most one worker is alive and at most one follow-up pass is pending.
/// This is deliberately a synchronous state machine: trigger admission never
/// awaits and therefore never creates a FIFO waiter task or retains a
/// compaction basis. The worker takes a fresh basis at the beginning of every
/// pass.
#[derive(Debug, Default)]
pub(crate) struct CompactionScheduler {
    state: StdMutex<SchedulerState>,
    idle: tokio::sync::Notify,
    /// Fair exclusion between immutable compaction output publication and the
    /// orphan janitor. Compaction passes hold a read guard from before their
    /// first input GET through manifest install; a sweep holds the write
    /// guard while it lists and deletes. Tokio's write-preferring FIFO lock
    /// prevents a trigger stream from starving a queued janitor.
    maintenance_gate: RwLock<()>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Admission {
    StartWorker,
    QueuedFollowUp,
    Coalesced,
}

impl CompactionScheduler {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    fn admit(&self, trigger: CompactionTrigger) -> Admission {
        let mut state = self.state.lock().expect("compaction scheduler poisoned");
        if !state.worker_active {
            state.worker_active = true;
            #[cfg(test)]
            {
                state.workers_started += 1;
            }
            return Admission::StartWorker;
        }
        if state.pending.is_none() {
            state.pending = Some(trigger);
            Admission::QueuedFollowUp
        } else {
            Admission::Coalesced
        }
    }

    fn pass_started(&self) {
        #[cfg(test)]
        {
            self.state
                .lock()
                .expect("compaction scheduler poisoned")
                .passes_started += 1;
        }
    }

    /// Select the one pending follow-up, or atomically make the scheduler
    /// idle. A trigger racing this transition either becomes that follow-up
    /// (if it wins first) or starts the next sole worker (if it wins second);
    /// it cannot be lost.
    fn next_trigger_or_stop(&self) -> Option<CompactionTrigger> {
        let mut state = self.state.lock().expect("compaction scheduler poisoned");
        match state.pending.take() {
            Some(trigger) => Some(trigger),
            None => {
                state.worker_active = false;
                self.idle.notify_waiters();
                None
            }
        }
    }

    fn worker_dropped(&self) {
        let mut state = self.state.lock().expect("compaction scheduler poisoned");
        state.worker_active = false;
        state.pending = None;
        self.idle.notify_waiters();
    }

    /// Wait until the currently scheduled burst has drained. Callers do not
    /// need this for trigger admission; it is useful when maintenance wants
    /// to sequence a janitor sweep after compaction.
    pub(crate) async fn wait_idle(&self) {
        loop {
            let notified = self.idle.notified();
            tokio::pin!(notified);
            // `notified()` does not join the wait list until first poll.
            // Enable it before checking state so `notify_waiters()` cannot
            // race through the check/await gap.
            notified.as_mut().enable();
            let worker_active = self
                .state
                .lock()
                .expect("compaction scheduler poisoned")
                .worker_active;
            if !worker_active {
                return;
            }
            notified.await;
        }
    }

    /// Guard one compaction pass against an overlapping orphan sweep.
    async fn compaction_guard(&self) -> RwLockReadGuard<'_, ()> {
        self.maintenance_gate.read().await
    }

    /// Exclude every current or newly-triggered compaction pass while the
    /// caller lists and deletes orphaned immutable objects.
    pub(crate) async fn sweep_guard(&self) -> RwLockWriteGuard<'_, ()> {
        self.maintenance_gate.write().await
    }

    #[cfg(test)]
    fn stats(&self) -> (bool, bool, u64, u64) {
        let state = self.state.lock().expect("compaction scheduler poisoned");
        (
            state.worker_active,
            state.pending.is_some(),
            state.workers_started,
            state.passes_started,
        )
    }
}

/// RAII reset for cancellation, abort, or panic. A namespace being evicted
/// aborts its tracked worker; without this guard a scheduler reused by tests
/// or shutdown code would remain permanently marked active.
struct WorkerLease {
    scheduler: Arc<CompactionScheduler>,
    armed: bool,
}

impl WorkerLease {
    fn new(scheduler: Arc<CompactionScheduler>) -> Self {
        Self {
            scheduler,
            armed: true,
        }
    }

    fn disarm(&mut self) {
        self.armed = false;
    }
}

impl Drop for WorkerLease {
    fn drop(&mut self) {
        if self.armed {
            self.scheduler.worker_dropped();
        }
    }
}

/// Terminal result of one scheduled pass.
#[derive(Debug)]
pub(crate) enum CompactionPass {
    /// Namespace retirement cancelled this pass before manifest installation.
    Cancelled,
    /// The captured manifest shape did not require a merge.
    Noop,
    /// One prepared result was installed.
    Applied {
        outcome: Box<CompactionOutcome>,
        l0_before: usize,
        l0_after: usize,
    },
}

/// Request compaction for one namespace.
///
/// Returns a handle only when this request starts the namespace's sole worker.
/// The caller may track that one handle for namespace retirement. A request
/// arriving while a worker runs reserves one basis-fresh follow-up; further
/// requests are represented by the `coalesced` metric and allocate no task.
#[allow(clippy::too_many_arguments)]
pub(crate) fn request_compaction(
    scheduler: &Arc<CompactionScheduler>,
    trigger: CompactionTrigger,
    writer: &Arc<Mutex<WriterSession>>,
    snapshot: &Arc<SnapshotCell>,
    writer_health: &Arc<WriterHealth>,
    namespace: &str,
    metrics: &Arc<Metrics>,
    cancel: Option<watch::Receiver<bool>>,
) -> Option<JoinHandle<()>> {
    match scheduler.admit(trigger) {
        Admission::QueuedFollowUp => return None,
        Admission::Coalesced => {
            metrics.observe_compaction_result(trigger, CompactionStatus::Coalesced, 0, 0, 0, 0);
            return None;
        }
        Admission::StartWorker => {}
    }

    // Construct the lease before spawning. If eviction aborts the task before
    // its first poll, dropping the unpolled future still resets admission.
    let lease = WorkerLease::new(Arc::clone(scheduler));
    let writer = Arc::clone(writer);
    let snapshot = Arc::clone(snapshot);
    let writer_health = Arc::clone(writer_health);
    let namespace = namespace.to_string();
    let metrics = Arc::clone(metrics);
    Some(tokio::spawn(async move {
        run_worker(
            lease,
            trigger,
            writer,
            snapshot,
            writer_health,
            namespace,
            metrics,
            cancel,
        )
        .await;
    }))
}

#[allow(clippy::too_many_arguments)]
async fn run_worker(
    mut lease: WorkerLease,
    mut trigger: CompactionTrigger,
    writer: Arc<Mutex<WriterSession>>,
    snapshot: Arc<SnapshotCell>,
    writer_health: Arc<WriterHealth>,
    namespace: String,
    metrics: Arc<Metrics>,
    cancel: Option<watch::Receiver<bool>>,
) {
    let scheduler = Arc::clone(&lease.scheduler);
    loop {
        if is_cancelled(cancel.as_ref()) {
            metrics.observe_compaction_result(trigger, CompactionStatus::Cancelled, 0, 0, 0, 0);
            return;
        }
        // Close the wait_idle→sweep race: even if a janitor observed the
        // scheduler idle immediately before this trigger, either this read
        // guard wins and the sweep waits for install, or the fair write guard
        // wins and this pass cannot upload an unreferenced body during it.
        let _maintenance_guard = scheduler.compaction_guard().await;
        if is_cancelled(cancel.as_ref()) {
            metrics.observe_compaction_result(trigger, CompactionStatus::Cancelled, 0, 0, 0, 0);
            return;
        }
        scheduler.pass_started();
        match run_compaction_pass(
            trigger,
            &writer,
            &snapshot,
            &writer_health,
            &namespace,
            &metrics,
            cancel.as_ref(),
        )
        .await
        {
            Ok(CompactionPass::Applied {
                outcome,
                l0_before,
                l0_after,
            }) => {
                info!(
                    namespace = %namespace,
                    ?trigger,
                    removed = outcome.source_ssts_removed,
                    written = outcome.new_ssts_written,
                    l0_before,
                    l0_after,
                    "compacted L0 backlog"
                );
            }
            Ok(CompactionPass::Cancelled) => return,
            Ok(CompactionPass::Noop) => {}
            Err(error) => {
                error!(
                    namespace = %namespace,
                    ?trigger,
                    %error,
                    "background compaction failed"
                );
            }
        }

        match scheduler.next_trigger_or_stop() {
            Some(next) => {
                trigger = next;
                // Give ready foreground work a scheduling opportunity between
                // passes during a sustained trigger stream.
                tokio::task::yield_now().await;
            }
            None => {
                lease.disarm();
                return;
            }
        }
    }
}

/// Run one compaction attempt without holding the writer mutex across its
/// expensive phase. Every invocation captures a fresh basis; only the
/// scheduler calls this function.
#[allow(clippy::too_many_arguments)]
async fn run_compaction_pass(
    trigger: CompactionTrigger,
    writer: &Mutex<WriterSession>,
    snapshot: &SnapshotCell,
    writer_health: &WriterHealth,
    namespace: &str,
    metrics: &Metrics,
    cancel: Option<&watch::Receiver<bool>>,
) -> namidb_storage::Result<CompactionPass> {
    let basis = {
        let wait_started = Instant::now();
        let writer = writer.lock().await;
        metrics.observe_writer_lock(
            WriterLockKind::CompactionBasis,
            wait_started.elapsed(),
            true,
        );
        if is_cancelled(cancel) {
            metrics.observe_compaction_result(trigger, CompactionStatus::Cancelled, 0, 0, 0, 0);
            return Ok(CompactionPass::Cancelled);
        }
        writer.compaction_basis()
    };

    let l0_before = basis.max_l0_bucket_len();
    if !basis.needs_compaction() {
        metrics.observe_compaction_result(
            trigger,
            CompactionStatus::Noop,
            l0_before,
            l0_before,
            0,
            0,
        );
        return Ok(CompactionPass::Noop);
    }

    // Storage moves the synchronous node/edge merges and Vamana/BM25 builds
    // to Tokio's blocking pool; this outer future retains async object-store
    // I/O without monopolising a reactor thread.
    let schema = basis.schema().clone();
    let prepare_started = Instant::now();
    let prepared = match basis.prepare(&schema).await {
        Ok(prepared) => {
            metrics.observe_compaction_phase(
                trigger,
                CompactionPhase::Prepare,
                prepare_started.elapsed(),
            );
            prepared
        }
        Err(error) => {
            metrics.observe_compaction_phase(
                trigger,
                CompactionPhase::Prepare,
                prepare_started.elapsed(),
            );
            metrics.observe_compaction_result(
                trigger,
                CompactionStatus::PrepareError,
                l0_before,
                l0_before,
                0,
                0,
            );
            return Err(error);
        }
    };
    if is_cancelled(cancel) {
        metrics.observe_compaction_result(
            trigger,
            CompactionStatus::Cancelled,
            l0_before,
            l0_before,
            0,
            0,
        );
        return Ok(CompactionPass::Cancelled);
    }

    // A long prepare may make foreground tasks runnable. Yield before joining
    // the FIFO writer queue so already-ready client work gets a chance to
    // enqueue; once acquired, only manifest validation/CAS runs under it.
    tokio::task::yield_now().await;
    let install_wait_started = Instant::now();
    let mut writer = writer.lock().await;
    let install_wait = install_wait_started.elapsed();
    metrics.observe_writer_lock(WriterLockKind::CompactionInstall, install_wait, true);
    metrics.observe_compaction_phase(trigger, CompactionPhase::InstallWait, install_wait);
    if is_cancelled(cancel) {
        metrics.observe_compaction_result(
            trigger,
            CompactionStatus::Cancelled,
            l0_before,
            writer.max_l0_bucket_len(),
            0,
            0,
        );
        return Ok(CompactionPass::Cancelled);
    }

    let install_started = Instant::now();
    let install = writer.install_prepared_compaction(prepared).await;

    match install {
        Ok(outcome) => {
            let l0_after = writer.max_l0_bucket_len();
            let status = if outcome.source_ssts_removed > 0 {
                snapshot.store(writer.owned_snapshot());
                CompactionStatus::Applied
            } else {
                CompactionStatus::Noop
            };
            metrics.observe_compaction_result(
                trigger,
                status,
                l0_before,
                l0_after,
                outcome.source_ssts_removed,
                outcome.new_ssts_written,
            );
            metrics.observe_compaction_phase(
                trigger,
                CompactionPhase::InstallHold,
                install_started.elapsed(),
            );
            if outcome.source_ssts_removed > 0 {
                Ok(CompactionPass::Applied {
                    outcome: Box::new(outcome),
                    l0_before,
                    l0_after,
                })
            } else {
                Ok(CompactionPass::Noop)
            }
        }
        Err(error) => {
            let status = if matches!(error, namidb_storage::Error::Precondition(_)) {
                // A competing compaction won the install race. Its result is
                // already current; the abandoned immutable outputs are swept.
                CompactionStatus::Stale
            } else {
                CompactionStatus::InstallError
            };
            let l0_after = writer.max_l0_bucket_len();
            metrics.observe_compaction_result(trigger, status, l0_before, l0_after, 0, 0);
            if !is_cancelled(cancel) {
                recovery::recover_writer_if_needed(
                    &mut writer,
                    snapshot,
                    writer_health,
                    namespace,
                    &error,
                )
                .await;
            }
            // Recovery (not just manifest CAS) runs while this writer guard is
            // held. Include it so the metric reports the foreground-visible
            // critical section instead of understating S3 failure stalls.
            metrics.observe_compaction_phase(
                trigger,
                CompactionPhase::InstallHold,
                install_started.elapsed(),
            );
            Err(error)
        }
    }
}

fn is_cancelled(cancel: Option<&watch::Receiver<bool>>) -> bool {
    cancel.is_some_and(|receiver| *receiver.borrow())
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;
    use std::time::Duration;

    use async_trait::async_trait;
    use namidb_core::{NodeId, Schema, Value};
    use namidb_storage::{NodeWriteRecord, WriterSession};
    use object_store::ObjectStore;
    use tokio::sync::Notify;

    use super::*;
    use crate::AppState;

    /// Blocks the first compaction SST GET on an explicit barrier. WAL,
    /// manifest, and flush traffic passes through immediately.
    #[derive(Debug)]
    struct BlockFirstSstGet {
        inner: Arc<dyn ObjectStore>,
        should_block: AtomicBool,
        started: Notify,
        release: Notify,
    }

    impl std::fmt::Display for BlockFirstSstGet {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "BlockFirstSstGet({})", self.inner)
        }
    }

    #[async_trait]
    impl ObjectStore for BlockFirstSstGet {
        async fn put_opts(
            &self,
            location: &object_store::path::Path,
            payload: object_store::PutPayload,
            opts: object_store::PutOptions,
        ) -> object_store::Result<object_store::PutResult> {
            self.inner.put_opts(location, payload, opts).await
        }

        async fn put_multipart_opts(
            &self,
            location: &object_store::path::Path,
            opts: object_store::PutMultipartOptions,
        ) -> object_store::Result<Box<dyn object_store::MultipartUpload>> {
            self.inner.put_multipart_opts(location, opts).await
        }

        async fn get_opts(
            &self,
            location: &object_store::path::Path,
            options: object_store::GetOptions,
        ) -> object_store::Result<object_store::GetResult> {
            if location.as_ref().contains("/sst/")
                && self.should_block.swap(false, Ordering::SeqCst)
            {
                self.started.notify_one();
                self.release.notified().await;
            }
            self.inner.get_opts(location, options).await
        }

        fn list(
            &self,
            prefix: Option<&object_store::path::Path>,
        ) -> futures::stream::BoxStream<'static, object_store::Result<object_store::ObjectMeta>>
        {
            self.inner.list(prefix)
        }

        async fn list_with_delimiter(
            &self,
            prefix: Option<&object_store::path::Path>,
        ) -> object_store::Result<object_store::ListResult> {
            self.inner.list_with_delimiter(prefix).await
        }

        async fn copy_opts(
            &self,
            from: &object_store::path::Path,
            to: &object_store::path::Path,
            options: object_store::CopyOptions,
        ) -> object_store::Result<()> {
            self.inner.copy_opts(from, to, options).await
        }

        fn delete_stream(
            &self,
            locations: futures::stream::BoxStream<
                'static,
                object_store::Result<object_store::path::Path>,
            >,
        ) -> futures::stream::BoxStream<'static, object_store::Result<object_store::path::Path>>
        {
            self.inner.delete_stream(locations)
        }
    }

    fn record(name: &str) -> NodeWriteRecord {
        NodeWriteRecord {
            properties: BTreeMap::from([("name".into(), Value::Str(name.into()))]),
            schema_version: 0,
            labels: vec![],
        }
    }

    async fn commit_and_flush(state: &AppState, id: NodeId, name: &str) {
        let mut writer = state.writer.lock().await;
        writer
            .upsert_node("Person", id, &record(name))
            .expect("stage node");
        writer.commit_batch().await.expect("commit node");
        writer.flush(Schema::empty()).await.expect("flush node");
        state.snapshot.store(writer.owned_snapshot());
    }

    fn request_for_state(
        state: &AppState,
        scheduler: &Arc<CompactionScheduler>,
        trigger: CompactionTrigger,
    ) -> Option<JoinHandle<()>> {
        request_compaction(
            scheduler,
            trigger,
            &state.writer,
            &state.snapshot,
            &state.writer_health,
            &state.namespace,
            &state.metrics,
            None,
        )
    }

    /// Barrier-based fairness and burst-coalescing regression:
    ///
    /// * the sole worker blocks inside the first pass's SST GET;
    /// * foreground writes create two more L0 files without waiting for it;
    /// * a 64-trigger burst allocates no additional task and reserves exactly
    ///   one follow-up pass;
    /// * that follow-up captures a fresh basis and drains the L0 files flushed
    ///   after the first pass began.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn burst_has_one_worker_one_fresh_follow_up_and_no_fifo_tasks() {
        let (_unused, paths) = namidb_storage::parse_uri("memory://maintenance-fairness").unwrap();
        let blocking_store = Arc::new(BlockFirstSstGet {
            inner: Arc::new(object_store::memory::InMemory::new()),
            should_block: AtomicBool::new(true),
            started: Notify::new(),
            release: Notify::new(),
        });
        let store: Arc<dyn ObjectStore> = blocking_store.clone();
        let writer = WriterSession::open(store, paths).await.unwrap();
        let state = AppState::new(writer, None, "maintenance-fairness".into());

        commit_and_flush(&state, NodeId::new(), "first").await;
        commit_and_flush(&state, NodeId::new(), "second").await;
        assert_eq!(state.writer.lock().await.max_l0_bucket_len(), 2);

        let scheduler = Arc::new(CompactionScheduler::new());
        let worker = request_for_state(&state, &scheduler, CompactionTrigger::Periodic)
            .expect("the first trigger must start the sole worker");

        tokio::time::timeout(Duration::from_secs(10), blocking_store.started.notified())
            .await
            .expect("compaction prepare must reach the SST GET barrier");

        // The prepare is now definitely in flight. Foreground commit+flush
        // must still acquire the writer mutex; two flushes also create enough
        // post-basis L0 work to require the pending follow-up.
        let interleaved_a = NodeId::new();
        tokio::time::timeout(
            Duration::from_secs(10),
            commit_and_flush(&state, interleaved_a, "interleaved-a"),
        )
        .await
        .expect("foreground writer must not wait for prepare");
        let interleaved_b = NodeId::new();
        tokio::time::timeout(
            Duration::from_secs(10),
            commit_and_flush(&state, interleaved_b, "interleaved-b"),
        )
        .await
        .expect("second foreground writer must not wait for prepare");
        assert_eq!(state.writer.lock().await.max_l0_bucket_len(), 4);

        for _ in 0..64 {
            assert!(
                request_for_state(&state, &scheduler, CompactionTrigger::Reactive).is_none(),
                "a burst while active must never allocate another worker task"
            );
        }
        assert_eq!(
            scheduler.stats(),
            (true, true, 1, 1),
            "one worker and one pending pass must bound the blocked burst"
        );

        blocking_store.release.notify_one();
        tokio::time::timeout(Duration::from_secs(15), worker)
            .await
            .expect("the two-pass burst must drain")
            .expect("compaction worker must not panic");
        scheduler.wait_idle().await;
        assert_eq!(
            scheduler.stats(),
            (false, false, 1, 2),
            "the burst must execute only its active pass and one follow-up"
        );
        assert_eq!(
            state.writer.lock().await.max_l0_bucket_len(),
            0,
            "the fresh follow-up basis must include both interleaved flushes"
        );

        let pinned = state.snapshot.load();
        let borrowed = pinned.borrow();
        for id in [interleaved_a, interleaved_b] {
            assert!(
                borrowed.lookup_node("Person", id).await.unwrap().is_some(),
                "a commit that landed during prepare must survive both installs"
            );
        }
        let rendered = state.metrics.render();
        assert!(rendered
            .contains("namidb_compactions_total{trigger=\"periodic\",status=\"applied\"} 1"));
        assert!(rendered
            .contains("namidb_compactions_total{trigger=\"reactive\",status=\"applied\"} 1"));
        assert!(rendered
            .contains("namidb_compactions_total{trigger=\"reactive\",status=\"coalesced\"} 63"));
    }

    /// `wait_idle()` alone cannot protect a janitor: a reactive trigger may
    /// arrive immediately after it returns. The sweep write guard must fence
    /// that newly-started worker before its first SST GET/output upload. This
    /// makes even a zero-age orphan sweep safe.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn sweep_guard_excludes_trigger_racing_after_idle_observation() {
        let (_unused, paths) =
            namidb_storage::parse_uri("memory://maintenance-sweep-exclusion").unwrap();
        let blocking_store = Arc::new(BlockFirstSstGet {
            inner: Arc::new(object_store::memory::InMemory::new()),
            should_block: AtomicBool::new(true),
            started: Notify::new(),
            release: Notify::new(),
        });
        let store: Arc<dyn ObjectStore> = blocking_store.clone();
        let writer = WriterSession::open(store, paths).await.unwrap();
        let state = AppState::new(writer, None, "maintenance-sweep-exclusion".into());
        commit_and_flush(&state, NodeId::new(), "first").await;
        commit_and_flush(&state, NodeId::new(), "second").await;

        let scheduler = Arc::new(CompactionScheduler::new());
        scheduler.wait_idle().await;
        let sweep_guard = scheduler.sweep_guard().await;
        let worker = request_for_state(&state, &scheduler, CompactionTrigger::Reactive)
            .expect("the racing trigger starts the sole worker");

        assert!(
            tokio::time::timeout(
                Duration::from_millis(100),
                blocking_store.started.notified()
            )
            .await
            .is_err(),
            "compaction reached an immutable SST while the sweep guard was held"
        );

        drop(sweep_guard);
        tokio::time::timeout(Duration::from_secs(10), blocking_store.started.notified())
            .await
            .expect("compaction must begin after the sweep guard is released");
        blocking_store.release.notify_one();
        tokio::time::timeout(Duration::from_secs(15), worker)
            .await
            .expect("compaction must finish after the sweep")
            .expect("worker must not panic");
        assert_eq!(
            state.writer.lock().await.max_l0_bucket_len(),
            0,
            "the deferred worker must still drain its captured backlog"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn retirement_during_prepare_never_installs_on_the_old_writer() {
        let (_unused, paths) =
            namidb_storage::parse_uri("memory://maintenance-cancel-install").unwrap();
        let blocking_store = Arc::new(BlockFirstSstGet {
            inner: Arc::new(object_store::memory::InMemory::new()),
            should_block: AtomicBool::new(true),
            started: Notify::new(),
            release: Notify::new(),
        });
        let store: Arc<dyn ObjectStore> = blocking_store.clone();
        let writer = WriterSession::open(store, paths).await.unwrap();
        let state = AppState::new(writer, None, "maintenance-cancel-install".into());
        commit_and_flush(&state, NodeId::new(), "first").await;
        commit_and_flush(&state, NodeId::new(), "second").await;

        let scheduler = Arc::new(CompactionScheduler::new());
        let (cancel_tx, cancel_rx) = watch::channel(false);
        let worker = request_compaction(
            &scheduler,
            CompactionTrigger::Periodic,
            &state.writer,
            &state.snapshot,
            &state.writer_health,
            &state.namespace,
            &state.metrics,
            Some(cancel_rx),
        )
        .expect("first trigger starts a worker");

        tokio::time::timeout(Duration::from_secs(10), blocking_store.started.notified())
            .await
            .expect("prepare must reach the blocked SST GET");
        cancel_tx.send_replace(true);
        blocking_store.release.notify_one();
        tokio::time::timeout(Duration::from_secs(15), worker)
            .await
            .expect("cancelled prepare must terminate")
            .expect("worker must not panic");

        scheduler.wait_idle().await;
        assert_eq!(
            state.writer.lock().await.max_l0_bucket_len(),
            2,
            "retirement after prepare starts must leave the old manifest untouched"
        );
        assert!(state
            .metrics
            .render()
            .contains("namidb_compactions_total{trigger=\"periodic\",status=\"cancelled\"} 1"));
    }
}
