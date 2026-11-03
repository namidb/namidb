//! The query timeout and the admin cancel must stop a cartesian product
//! mid-operator. `MATCH (a:N), (b:N) RETURN count(*)` over 3,000 nodes is a
//! 9M-row cross product with no storage read underneath it, so the only
//! cooperative probe it used to reach was the operator boundary: it ran for
//! more than 40 s under `--query-timeout 1s` and ignored an admin cancel.
//! Now every CPU-only operator loop probes the guards (amortised), and the
//! query stops within a fraction of a second of the deadline.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use namidb_core::id::{NamespaceId, NodeId};
use namidb_core::value::Value as CoreValue;
use namidb_storage::{NamespacePaths, NodeWriteRecord, WriterSession};
use object_store::memory::InMemory;

use namidb_query::{
    execute_factor_path, execute_flat_path, execute_with_limits, lower, optimize, parse, ExecError,
    LogicalPlan, Params, StatsCatalog,
};

/// The repro's bound: the query must stop within ~1.5 s of its deadline.
const GRACE: Duration = Duration::from_millis(1_500);

/// Plan as the server does (parse, lower, optimize).
fn plan(query: &str) -> LogicalPlan {
    optimize(
        lower(&parse(query).unwrap()).unwrap(),
        &StatsCatalog::empty(),
    )
}

async fn seed(name: &str, nodes: i64) -> WriterSession {
    let mut writer = WriterSession::open(
        Arc::new(InMemory::new()),
        NamespacePaths::new("tenants", NamespaceId::new(name).unwrap()),
    )
    .await
    .unwrap();
    for i in 0..nodes {
        let mut p = BTreeMap::new();
        p.insert("i".into(), CoreValue::I64(i));
        writer
            .upsert_node(
                "N",
                NodeId::new(),
                &NodeWriteRecord {
                    properties: p,
                    schema_version: 1,
                    ..Default::default()
                },
            )
            .unwrap();
    }
    writer.commit_batch().await.unwrap();
    writer
}

/// The reported reproduction, on the default executor: a 1 s timeout must
/// surface `Timeout` well before the product could finish.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cartesian_count_honours_the_query_timeout() {
    let writer = seed("cartesian-timeout", 3_000).await;
    let snap = writer.snapshot();
    let plan = plan("MATCH (a:N), (b:N) RETURN count(*) AS n");

    let timeout = Duration::from_secs(1);
    let start = Instant::now();
    let err = execute_with_limits(&plan, &snap, &Params::new(), Some(start + timeout), None)
        .await
        .expect_err("a 9M-row cross product cannot finish in 1 s");
    let elapsed = start.elapsed();
    assert!(matches!(err, ExecError::Timeout), "got {err:?}");
    assert!(
        elapsed < timeout + GRACE,
        "stopped {elapsed:?} after start, past the {timeout:?} deadline + {GRACE:?}"
    );
}

/// Both executors (flat and factorised) stop on the deadline.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cartesian_stops_on_both_executor_paths() {
    let writer = seed("cartesian-paths", 3_000).await;
    let snap = writer.snapshot();
    let params = Params::new();
    for query in [
        "MATCH (a:N), (b:N) RETURN count(*) AS n",
        // Aggregation, sort and DISTINCT over the product.
        "MATCH (a:N), (b:N) RETURN a.i + b.i AS s, count(*) AS n ORDER BY s",
        "MATCH (a:N), (b:N) RETURN DISTINCT a.i - b.i AS d",
        // A filter that rejects every pair keeps the output empty, so only
        // the in-loop probes can stop it.
        "MATCH (a:N), (b:N) WHERE a.i + b.i < 0 RETURN a, b",
    ] {
        let plan = plan(query);
        for factor in [false, true] {
            let timeout = Duration::from_millis(500);
            let start = Instant::now();
            let deadline = Some(start + timeout);
            let result = if factor {
                namidb_storage::cancel::with_deadline(
                    deadline,
                    execute_factor_path(&plan, &snap, &params),
                )
                .await
            } else {
                namidb_storage::cancel::with_deadline(
                    deadline,
                    execute_flat_path(&plan, &snap, &params),
                )
                .await
            };
            let elapsed = start.elapsed();
            let err = result.expect_err(query);
            assert!(
                matches!(err, ExecError::Timeout),
                "{query} (factor={factor}): {err:?}"
            );
            assert!(
                elapsed < timeout + GRACE,
                "{query} (factor={factor}) stopped after {elapsed:?}"
            );
        }
    }
}

/// The admin cancel flips the same flag the server registers per query; the
/// product must observe it mid-loop, not run to completion first. The flag
/// is flipped from a plain thread because the query never yields to tokio.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cartesian_count_honours_admin_cancel() {
    let writer = seed("cartesian-cancel", 3_000).await;
    let snap = writer.snapshot();
    let plan = plan("MATCH (a:N), (b:N) RETURN count(*) AS n");

    let flag = Arc::new(AtomicBool::new(false));
    let delay = Duration::from_millis(300);
    let flipper = {
        let flag = flag.clone();
        std::thread::spawn(move || {
            std::thread::sleep(delay);
            flag.store(true, Ordering::Relaxed);
            Instant::now()
        })
    };
    let err = namidb_storage::cancel::with_cancel_flag(
        flag,
        execute_with_limits(&plan, &snap, &Params::new(), None, None),
    )
    .await
    .expect_err("a cancelled cross product must abort");
    let stopped = Instant::now();
    let cancelled_at = flipper.join().unwrap();
    assert!(matches!(err, ExecError::Cancelled), "got {err:?}");
    assert!(
        stopped.duration_since(cancelled_at) < GRACE,
        "kept running {:?} after the cancel",
        stopped.duration_since(cancelled_at)
    );
}

/// The probes are no-ops without a guard, and with a generous one: a small
/// product still returns the exact count.
#[tokio::test]
async fn small_cartesian_still_completes_under_a_generous_deadline() {
    let writer = seed("cartesian-ok", 120).await;
    let snap = writer.snapshot();
    let plan = plan("MATCH (a:N), (b:N) RETURN count(*) AS n");
    for deadline in [None, Some(Instant::now() + Duration::from_secs(120))] {
        let rows = execute_with_limits(&plan, &snap, &Params::new(), deadline, None)
            .await
            .unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(
            rows[0].get("n"),
            Some(&namidb_query::RuntimeValue::Integer(14_400))
        );
    }
}
