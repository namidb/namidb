//! A1: background compaction + orphan sweep.
//!
//! The server's maintenance task collapses L0 SSTs into L1 (bounding read
//! amplification) and sweeps the orphaned L0 bodies compaction leaves
//! behind. This test drives ONE maintenance tick inline (no wall-clock
//! dependency) against the same `AppState` the server uses, and pins the
//! contract:
//!
//! 1. Seeding three flushed batches creates >= 2 L0 SSTs in one bucket.
//! 2. `compact_l0` collapses them to a single L1 SST and removes the L0s.
//! 3. Reads return the same data before and after — including a snapshot
//!    pinned *before* compaction (the source bodies survive for it).
//! 4. The orphan sweep finds the now-unreferenced L0 bodies, deletes them
//!    when asked, and a re-run finds nothing — and reads still hold.

use std::sync::Arc;
use std::time::{Duration, Instant};

use namidb_query::{
    execute, execute_write, parse as cypher_parse, plan as build_plan, Params, RuntimeValue,
    StatsCatalog,
};
use namidb_storage::{
    sweep_orphans, ManifestStore, OwnedSnapshot, SstKind, SstLevel, WriterSession,
};
use object_store::ObjectStore;

const NS: &str = "maint-test";

async fn create_person(state: &namidb_server::AppState, i: usize) {
    let q = format!(
        "CREATE (a:Person {{name: 'p{i}', age: {}}})",
        (i % 80) as i64
    );
    let parsed = cypher_parse(&q).expect("parse");
    let mut w = state.writer.lock().await;
    let catalog = StatsCatalog::from_manifest(&w.snapshot().manifest().manifest);
    let plan = build_plan(&parsed, &catalog).expect("plan");
    execute_write(&plan, &mut w, &Params::new())
        .await
        .expect("write");
    state.snapshot.store(w.owned_snapshot());
}

/// Flush the live memtable into one L0 SST per touched bucket and
/// republish.
async fn flush(state: &namidb_server::AppState) {
    let mut w = state.writer.lock().await;
    let schema = w.snapshot().manifest().manifest.schema.clone();
    w.flush(schema).await.expect("flush");
    state.snapshot.store(w.owned_snapshot());
}

// id-primary node SSTs are no longer partitioned by label (`scope == ""`), so
// count the whole node bucket at a level. The fixture only writes Person nodes,
// so this is exactly the count of Person SSTs.
fn node_ssts(snap: &OwnedSnapshot, level: SstLevel) -> usize {
    snap.manifest()
        .manifest
        .ssts
        .iter()
        .filter(|d| d.kind == SstKind::Nodes && d.level == level)
        .count()
}

/// `MATCH (p:Person) RETURN count(p)` against a pinned snapshot.
async fn count_persons(snap: &OwnedSnapshot) -> i64 {
    let parsed = cypher_parse("MATCH (p:Person) RETURN count(p) AS c").expect("parse");
    let catalog = StatsCatalog::from_manifest(&snap.manifest().manifest);
    let plan = build_plan(&parsed, &catalog).expect("plan");
    let borrowed = snap.borrow();
    let rows = execute(&plan, &borrowed, &Params::new())
        .await
        .expect("read");
    match rows.first().and_then(|r| r.get("c")) {
        Some(RuntimeValue::Integer(n)) => *n,
        other => panic!("unexpected count row: {other:?}"),
    }
}

#[tokio::test]
async fn compaction_collapses_l0_and_sweep_reclaims_orphans() {
    let (store, paths) = namidb_storage::parse_uri(&format!("memory://{NS}")).unwrap();
    // The sweep loads the committed manifest itself, so it needs its own
    // ManifestStore built from the same (store, paths) — exactly what the
    // server does in `run()`.
    let manifest_store = ManifestStore::new(store.clone(), paths.clone());
    let writer = WriterSession::open(store, paths).await.unwrap();
    let state = namidb_server::AppState::new(writer, None, NS.into());

    // Three flushed batches → three L0 SSTs in the (Nodes, Person) bucket.
    const BATCHES: usize = 3;
    const PER_BATCH: usize = 4;
    for b in 0..BATCHES {
        for j in 0..PER_BATCH {
            create_person(&state, b * PER_BATCH + j).await;
        }
        flush(&state).await;
    }
    let total = (BATCHES * PER_BATCH) as i64;

    {
        let before = state.snapshot.load();
        assert!(
            node_ssts(&before, SstLevel::L0) >= 2,
            "expected >= 2 L0 Person SSTs before compaction, got {}",
            node_ssts(&before, SstLevel::L0)
        );
        assert_eq!(count_persons(&before).await, total, "baseline count");
    } // drop `before` so only the explicit pin below holds the old version

    // Pin a snapshot from before compaction: its source L0 bodies must
    // remain readable even after compaction drops them from the manifest,
    // and the retention horizon must keep the sweep off them (RFC-027).
    let pinned_pre_compaction = state.snapshot.load();
    let pinned_version = pinned_pre_compaction.manifest_version();

    // ── One maintenance tick: compaction under the writer lock ──
    {
        let mut w = state.writer.lock().await;
        let schema = w.snapshot().manifest().manifest.schema.clone();
        let outcome = w.compact_l0(&schema).await.expect("compact");
        assert!(
            outcome.source_ssts_removed >= 2,
            "compaction should consume >= 2 L0 SSTs, removed {}",
            outcome.source_ssts_removed
        );
        assert!(
            outcome.new_ssts_written >= 1,
            "compaction should write >= 1 L1 SST, wrote {}",
            outcome.new_ssts_written
        );
        state.snapshot.store(w.owned_snapshot());
    }

    {
        let after = state.snapshot.load();
        assert_eq!(
            node_ssts(&after, SstLevel::L0),
            0,
            "no L0 Person SSTs should remain after compaction"
        );
        assert_eq!(
            node_ssts(&after, SstLevel(1)),
            1,
            "the bucket should collapse to a single L1 SST"
        );
        // Reads stay correct on the new snapshot AND on the pre-compaction one.
        assert_eq!(count_persons(&after).await, total, "count after compaction");
    } // drop `after` so the post-compaction version is not separately pinned

    assert_eq!(
        count_persons(&pinned_pre_compaction).await,
        total,
        "a snapshot pinned before compaction still reads its source bodies"
    );

    // ── Horizon-aware orphan sweep (RFC-027). min_age = 0 in-test
    // (production default is 24h); max_level = 1. ──
    //
    // While the pre-compaction reader is alive it holds the horizon at its
    // version, which still references the dropped L0 bodies, so the sweep
    // must NOT delete them even with delete = true.
    let horizon_pinned = state.snapshot.retention_horizon();
    assert_eq!(
        horizon_pinned, pinned_version,
        "the pinned reader holds the retention horizon at its version"
    );
    let guarded = sweep_orphans(
        &manifest_store,
        horizon_pinned,
        std::time::Duration::ZERO,
        1,
        true,
    )
    .await
    .expect("guarded sweep");
    assert_eq!(
        guarded.orphans_found, 0,
        "the pinned version still references the L0 bodies; none are orphans"
    );
    assert_eq!(guarded.orphans_deleted, 0, "nothing deleted under the pin");
    assert_eq!(
        count_persons(&pinned_pre_compaction).await,
        total,
        "the pinned reader still reads its source bodies after the guarded sweep"
    );

    // Drop the reader: the horizon advances to current and the L0 bodies,
    // now referenced by no retained version, become reclaimable.
    drop(pinned_pre_compaction);
    let horizon_free = state.snapshot.retention_horizon();
    assert_eq!(
        horizon_free,
        state.snapshot.manifest_version(),
        "with no readers the horizon is the current version"
    );
    let del = sweep_orphans(
        &manifest_store,
        horizon_free,
        std::time::Duration::ZERO,
        1,
        true,
    )
    .await
    .expect("delete sweep");
    assert!(
        del.orphans_deleted >= 2,
        "after the reader leaves, the orphaned L0 bodies are reclaimed, deleted {}",
        del.orphans_deleted
    );

    // Idempotent: a second sweep finds nothing.
    let again = sweep_orphans(
        &manifest_store,
        state.snapshot.retention_horizon(),
        std::time::Duration::ZERO,
        1,
        true,
    )
    .await
    .expect("second sweep");
    assert_eq!(again.orphans_found, 0, "no orphans remain after the sweep");

    // The live L1 SST was untouched: a fresh read still returns everything.
    let final_snap = state.snapshot.load();
    assert_eq!(
        count_persons(&final_snap).await,
        total,
        "count still correct after the sweep deleted the orphaned bodies"
    );
}

/// `ObjectStore` wrapper that sleeps before every GET of an SST body, making
/// a compaction prepare (which downloads all its inputs) deterministically
/// slow while writes — WAL and manifest PUTs, no SST GETs — stay fast.
#[derive(Debug)]
struct SlowSstGets {
    inner: Arc<dyn ObjectStore>,
    delay: Duration,
}

impl std::fmt::Display for SlowSstGets {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "SlowSstGets({})", self.inner)
    }
}

#[async_trait::async_trait]
impl ObjectStore for SlowSstGets {
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
        if location.as_ref().contains("/sst/") {
            tokio::time::sleep(self.delay).await;
        }
        self.inner.get_opts(location, options).await
    }

    fn list(
        &self,
        prefix: Option<&object_store::path::Path>,
    ) -> futures::stream::BoxStream<'static, object_store::Result<object_store::ObjectMeta>> {
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
    ) -> futures::stream::BoxStream<'static, object_store::Result<object_store::path::Path>> {
        self.inner.delete_stream(locations)
    }
}

/// Finding 32: the maintenance loops snapshot a compaction basis under a
/// brief writer lock, run the expensive prepare (input downloads, merge,
/// index rebuilds, output uploads) WITHOUT the lock, and re-take it only
/// for the manifest CAS. This drives that exact sequence against an
/// `AppState` whose SST GETs are artificially slow and asserts that writes
/// issued while the prepare is in flight complete without waiting for it —
/// under the old under-lock compaction every one of them would block for
/// the full prepare duration.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn writes_complete_while_a_slow_prepare_runs_off_lock() {
    const SST_GET_DELAY: Duration = Duration::from_millis(300);
    const SLOW_NS: &str = "maint-offlock";

    // Reuse parse_uri only for the canonical paths; the store itself is a
    // fresh in-memory one behind the GET-delaying wrapper.
    let (_discard, paths) = namidb_storage::parse_uri(&format!("memory://{SLOW_NS}")).unwrap();
    let store: Arc<dyn ObjectStore> = Arc::new(SlowSstGets {
        inner: Arc::new(object_store::memory::InMemory::new()),
        delay: SST_GET_DELAY,
    });
    let writer = WriterSession::open(store, paths).await.unwrap();
    let state = namidb_server::AppState::new(writer, None, SLOW_NS.into());

    // Two flushed batches → two L0 SSTs in the node bucket, so the prepare
    // downloads (at least) two bodies and takes >= 2 * SST_GET_DELAY.
    const SEEDED: usize = 6;
    for b in 0..2 {
        for j in 0..3 {
            create_person(&state, b * 3 + j).await;
        }
        flush(&state).await;
    }
    assert!(node_ssts(&state.snapshot.load(), SstLevel::L0) >= 2);

    // ── The off-lock maintenance sequence: brief lock → basis → UNLOCK ──
    let (basis, schema) = {
        let w = state.writer.lock().await;
        let schema = w.snapshot().manifest().manifest.schema.clone();
        (w.compaction_basis(), schema)
    };
    assert!(basis.needs_compaction());

    let prepare_started = Instant::now();
    let prepare = tokio::spawn(async move {
        let prepared = basis.prepare(&schema).await.expect("prepare");
        (prepared, prepare_started.elapsed())
    });

    // Writes issued while the slow prepare is in flight.
    let mut writes_during_prepare = 0usize;
    let mut max_write = Duration::ZERO;
    while !prepare.is_finished() {
        let t0 = Instant::now();
        create_person(&state, 100 + writes_during_prepare).await;
        max_write = max_write.max(t0.elapsed());
        writes_during_prepare += 1;
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    let (prepared, prepare_elapsed) = prepare.await.expect("prepare task");

    assert!(
        prepare_elapsed >= 2 * SST_GET_DELAY,
        "the delayed SST GETs must make the prepare measurably slow, took {prepare_elapsed:?}"
    );
    assert!(
        writes_during_prepare >= 1,
        "at least one write must land while the prepare runs"
    );
    assert!(
        max_write < prepare_elapsed / 2,
        "a write ({max_write:?}) must not wait out the prepare ({prepare_elapsed:?})"
    );

    // ── Re-lock only for the install; the interleaved writes survive. ──
    {
        let mut w = state.writer.lock().await;
        let outcome = w
            .install_prepared_compaction(prepared)
            .await
            .expect("install");
        assert!(outcome.source_ssts_removed >= 2);
        state.snapshot.store(w.owned_snapshot());
    }
    let after = state.snapshot.load();
    assert_eq!(
        count_persons(&after).await,
        (SEEDED + writes_during_prepare) as i64,
        "every write issued during the prepare must survive the install"
    );
}
