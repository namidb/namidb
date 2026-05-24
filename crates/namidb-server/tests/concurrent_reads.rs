//! RFC-021: reads no longer queue behind the writer mutex.
//!
//! Strategy:
//!
//! 1. Bootstrap a memory namespace with N nodes via the REST writer.
//! 2. Spawn K tokio tasks that each run a `MATCH (p:Person) RETURN count(p)`
//!    read against the *same* `AppState` for `D` seconds.
//! 3. Measure total queries served and assert that K-way fan-out exceeds
//!    the single-reader baseline by a non-trivial multiplier.
//!
//! Before RFC-021, `cypher` took `state.writer.lock()` for the entire
//! duration of the read, so K-way fan-out was approximately equal to
//! single-reader throughput. After RFC-021 the read path doesn't touch
//! the writer mutex at all and the K tasks share cores.
//!
//! The assertion is conservative (`K_runs >= 2 * single_runs`) so the
//! test stays green on small CI runners. On a modern multi-core box
//! the ratio is closer to K.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use namidb_query::{
    execute, execute_write, parse as cypher_parse, plan as build_plan, Params, StatsCatalog,
};
use namidb_storage::WriterSession;

const N_PERSONS: usize = 200;
const READ_BUDGET: Duration = Duration::from_millis(750);
const READ_TASKS: usize = 8;

async fn seed_persons(state: &namidb_server::AppState) {
    for i in 0..N_PERSONS {
        let query = format!(
            "CREATE (a:Person {{name: 'p{i}', age: {age}}})",
            i = i,
            age = (i % 80) as i64,
        );
        let parsed = cypher_parse(&query).expect("parse");
        let mut writer = state.writer.lock().await;
        let catalog = StatsCatalog::from_manifest(&writer.snapshot().manifest().manifest);
        let plan = build_plan(&parsed, &catalog).expect("plan");
        execute_write(&plan, &mut writer, &Params::new())
            .await
            .expect("write");
        state.snapshot.store(writer.owned_snapshot());
    }
}

/// Spawn `tasks` readers, each running for `budget`. Returns the
/// total number of completed reads across all tasks.
async fn run_readers(state: namidb_server::AppState, tasks: usize, budget: Duration) -> u64 {
    let total = Arc::new(AtomicU64::new(0));
    let mut joins = Vec::with_capacity(tasks);
    for _ in 0..tasks {
        let state = state.clone();
        let total = Arc::clone(&total);
        joins.push(tokio::spawn(async move {
            let parsed = cypher_parse("MATCH (p:Person) WHERE p.age >= $min RETURN count(p) AS c")
                .expect("parse");
            let mut params = Params::new();
            params.insert("min".into(), namidb_query::RuntimeValue::Integer(0));
            let start = Instant::now();
            while start.elapsed() < budget {
                let owned = state.snapshot.load();
                let catalog = StatsCatalog::from_manifest(&owned.manifest().manifest);
                let plan = build_plan(&parsed, &catalog).expect("plan");
                let snap = owned.borrow();
                execute(&plan, &snap, &params).await.expect("read");
                total.fetch_add(1, Ordering::Relaxed);
            }
        }));
    }
    for j in joins {
        j.await.expect("task panic");
    }
    total.load(Ordering::Relaxed)
}

async fn fresh_state() -> namidb_server::AppState {
    let (store, paths) = namidb_storage::parse_uri("memory://concurrent-reads-test").unwrap();
    let writer = WriterSession::open(store, paths).await.unwrap();
    let state = namidb_server::AppState::new(writer, None, "concurrent-reads-test".into());
    seed_persons(&state).await;
    state
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn many_readers_outpace_single_reader() {
    let baseline_state = fresh_state().await;
    let single = run_readers(baseline_state, 1, READ_BUDGET).await;

    let fanout_state = fresh_state().await;
    let many = run_readers(fanout_state, READ_TASKS, READ_BUDGET).await;

    let ratio = many as f64 / single.max(1) as f64;
    eprintln!("single={single} many={many} ratio={ratio:.2} (tasks={READ_TASKS})");
    // Before RFC-021 this ratio was ~1.0 because every read held the
    // writer mutex. Conservative bound for CI: at least 2x.
    assert!(
        ratio >= 2.0,
        "expected at least 2x fan-out (got {ratio:.2}); did the writer mutex creep back into the read path?"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn writes_publish_a_new_snapshot_visible_to_readers() {
    let state = fresh_state().await;
    let v_before = state.snapshot.manifest_version();

    // Run a write that bumps the manifest.
    {
        let mut writer = state.writer.lock().await;
        let schema = writer.snapshot().manifest().manifest.schema.clone();
        writer.flush(schema).await.expect("flush");
        state.snapshot.store(writer.owned_snapshot());
    }

    let v_after = state.snapshot.manifest_version();
    assert!(
        v_after > v_before,
        "manifest version must advance after flush ({v_before} -> {v_after})"
    );

    // A reader picks up the post-flush snapshot.
    let owned = state.snapshot.load();
    assert_eq!(owned.manifest_version(), v_after);
}
