//! CPU-only operator hot loops, with and without a query deadline in scope.
//!
//! The cross product, joins, filter, projection, aggregation, DISTINCT and
//! ORDER BY loops run over rows already in memory, so they reach no storage
//! read and probe the query timeout / admin cancel through an amortised
//! in-loop check instead (`exec::limits::DeadlineProbe`, one clock read per
//! 1,024 rows). This bench pins what that costs: each workload runs
//! `unguarded` (no deadline: the probe is a counter increment) and `guarded`
//! (a deadline an hour away, as the server runs every read: the probe also
//! reads the task-local and the clock every 1,024 rows, and an ORDER BY over
//! more than 16,384 rows takes the interruptible run-and-merge sort).
//!
//! ## Threshold
//!
//! `guarded` must stay within 2% of `unguarded`, and of the same workload on
//! a build without the probes. Run it the same way on both builds and compare
//! the medians:
//!
//! ```text
//! cargo bench -p namidb-query --bench cpu_operators
//! ```
//!
//! `BENCH_NODES=<n>` scales the `:N` label (default 20,000; the cross product
//! uses the first 700 of them through `n.i < 700`).

use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use criterion::{criterion_group, criterion_main, Criterion};
use namidb_core::id::{NamespaceId, NodeId};
use namidb_core::value::Value as CoreValue;
use namidb_query::{
    execute, execute_with_limits, lower, optimize, parse, LogicalPlan, Params, StatsCatalog,
};
use namidb_storage::{NamespacePaths, NodeWriteRecord, WriterSession};
use object_store::memory::InMemory;
use tokio::runtime::Runtime;

const WORKLOADS: &[(&str, &str)] = &[
    (
        "cross_product_count",
        "MATCH (a:N), (b:N) WHERE a.i < 700 AND b.i < 700 RETURN count(*) AS n",
    ),
    (
        "filter_project",
        "MATCH (n:N) WHERE n.i % 3 = 0 RETURN n.i * 2 AS x, n.i + 1 AS y",
    ),
    (
        "group_aggregate",
        "MATCH (n:N) RETURN n.i % 97 AS k, count(*) AS c, collect(n.i) AS xs",
    ),
    ("distinct", "MATCH (n:N) RETURN DISTINCT n.i % 1000 AS k"),
    ("order_by", "MATCH (n:N) RETURN n.i AS i ORDER BY i DESC"),
    (
        "hash_join",
        "MATCH (a:N), (b:N) WHERE a.i < 5000 AND b.i < 5000 AND a.i = b.i RETURN count(*) AS n",
    ),
];

fn plan(query: &str) -> LogicalPlan {
    optimize(
        lower(&parse(query).unwrap()).unwrap(),
        &StatsCatalog::empty(),
    )
}

async fn seed(nodes: i64) -> WriterSession {
    let mut writer = WriterSession::open(
        Arc::new(InMemory::new()),
        NamespacePaths::new("bench", NamespaceId::new("cpu-operators").unwrap()),
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

fn cpu_operators(c: &mut Criterion) {
    let nodes: i64 = std::env::var("BENCH_NODES")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(20_000);
    let rt = Runtime::new().unwrap();
    let writer = rt.block_on(seed(nodes));
    let snap = writer.snapshot();
    let params = Params::new();

    let mut group = c.benchmark_group("cpu_operators");
    group.sample_size(20);
    group.measurement_time(Duration::from_secs(8));
    for (name, query) in WORKLOADS {
        let plan = plan(query);
        group.bench_function(format!("{name}/unguarded"), |b| {
            b.to_async(&rt)
                .iter(|| async { execute(&plan, &snap, &params).await.unwrap() })
        });
        group.bench_function(format!("{name}/guarded"), |b| {
            b.to_async(&rt).iter(|| async {
                let deadline = Instant::now() + Duration::from_secs(3600);
                execute_with_limits(&plan, &snap, &params, Some(deadline), None)
                    .await
                    .unwrap()
            })
        });
    }
    group.finish();
}

criterion_group!(benches, cpu_operators);
criterion_main!(benches);
