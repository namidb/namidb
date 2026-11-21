//! A list bound in an earlier `WITH` and then iterated (UNWIND, reduce, a
//! list comprehension, a quantifier) used to copy the whole row, list
//! included, once per element: quadratic time and memory, so 20,000 items
//! needed gigabytes. Each loop now copies the row once, and an UNWIND whose
//! source list is not read again moves the list out of the row instead of
//! carrying it on every output row. The answers do not change, and a list
//! that is still read downstream stays bound.

use std::sync::Arc;
use std::time::{Duration, Instant};

use namidb_core::id::NamespaceId;
use namidb_query::{
    execute_with_limits, lower, optimize, parse, with_query_memory, ExecError, Params,
    QueryMemoryBudget, Row, RuntimeValue, StatsCatalog,
};
use namidb_storage::{NamespacePaths, WriterSession};
use object_store::memory::InMemory;

const MIB: usize = 1024 * 1024;

async fn writer(name: &str) -> WriterSession {
    WriterSession::open(
        Arc::new(InMemory::new()),
        NamespacePaths::new("tenants", NamespaceId::new(name).unwrap()),
    )
    .await
    .unwrap()
}

async fn run(w: &WriterSession, query: &str, budget: usize) -> Result<Vec<Row>, ExecError> {
    let snap = w.snapshot();
    let plan = optimize(
        lower(&parse(query).unwrap()).unwrap(),
        &StatsCatalog::empty(),
    );
    let budget = Arc::new(QueryMemoryBudget::new(budget));
    with_query_memory(
        Some(budget),
        execute_with_limits(
            &plan,
            &snap,
            &Params::new(),
            Some(Instant::now() + Duration::from_secs(60)),
            None,
        ),
    )
    .await
}

fn int(row: &Row, column: &str) -> i64 {
    match row.get(column) {
        Some(RuntimeValue::Integer(v)) => *v,
        other => panic!("{column}: {other:?}"),
    }
}

#[tokio::test]
async fn loops_over_a_bound_list_are_linear() {
    let w = writer("bound-list-loops").await;
    let n: i64 = 40_000;
    let sum = n * (n + 1) / 2;
    for (query, column, expected) in [
        (
            format!("WITH range(1, {n}) AS xs UNWIND xs AS x RETURN count(x) AS c"),
            "c",
            n,
        ),
        (
            format!("WITH range(1, {n}) AS xs UNWIND xs AS x WITH x WHERE x > 0 RETURN sum(x) AS s"),
            "s",
            sum,
        ),
        (
            format!("WITH range(1, {n}) AS xs RETURN reduce(s = 0, x IN xs | s + x) AS s"),
            "s",
            sum,
        ),
        (
            format!("WITH range(1, {n}) AS xs RETURN size([x IN xs WHERE x % 7 = 0]) AS c"),
            "c",
            n / 7,
        ),
        (
            format!("WITH range(1, {n}) AS xs RETURN size([x IN xs | x * 2]) AS c"),
            "c",
            n,
        ),
        (
            format!("WITH range(1, {n}) AS xs RETURN CASE WHEN all(x IN xs WHERE x > 0) THEN 1 ELSE 0 END AS c"),
            "c",
            1,
        ),
        (
            format!("WITH range(1, {n}) AS xs RETURN CASE WHEN any(x IN xs WHERE x = {n}) THEN 1 ELSE 0 END AS c"),
            "c",
            1,
        ),
    ] {
        let started = Instant::now();
        // 40,000 copies of a 40,000-item list would be about 50 GB.
        let rows = run(&w, &query, 256 * MIB)
            .await
            .unwrap_or_else(|e| panic!("{query}: {e:?}"));
        assert_eq!(int(&rows[0], column), expected, "{query}");
        assert!(
            started.elapsed() < Duration::from_secs(20),
            "{query}: took {:?}",
            started.elapsed()
        );
    }
}

#[tokio::test]
async fn a_list_read_after_its_unwind_stays_bound() {
    let w = writer("bound-list-live").await;
    // RETURN *, a later projection, a grouping key, a subquery import and a
    // second UNWIND all read the source list after the first UNWIND.
    let star = run(&w, "WITH [1, 2, 3] AS xs UNWIND xs AS x RETURN *", 64 * MIB)
        .await
        .unwrap();
    assert_eq!(star.len(), 3);
    for row in &star {
        assert_eq!(
            row.get("xs"),
            Some(&RuntimeValue::List(vec![
                RuntimeValue::Integer(1),
                RuntimeValue::Integer(2),
                RuntimeValue::Integer(3),
            ]))
        );
    }
    for (query, expected) in [
        (
            "WITH [1, 2, 3] AS xs UNWIND xs AS x RETURN sum(x * size(xs)) AS v",
            18,
        ),
        (
            "WITH [1, 2, 3] AS xs UNWIND xs AS x WITH x, xs RETURN sum(x + xs[0]) AS v",
            9,
        ),
        (
            "WITH [1, 2, 3] AS xs UNWIND xs AS x \
             CALL { WITH xs RETURN size(xs) AS n } RETURN sum(x * n) AS v",
            18,
        ),
        (
            "WITH [1, 2, 3] AS xs UNWIND xs AS x UNWIND xs AS y RETURN count(*) AS v",
            9,
        ),
        (
            "WITH [1, 2, 3] AS xs UNWIND xs AS x \
             RETURN sum(reduce(s = 0, y IN xs | s + y) + x) AS v",
            24,
        ),
    ] {
        let rows = run(&w, query, 64 * MIB)
            .await
            .unwrap_or_else(|e| panic!("{query}: {e:?}"));
        assert_eq!(int(&rows[0], "v"), expected, "{query}");
    }
    // A list that is still read keeps one copy per output row, so a large
    // one runs into the memory budget, and quickly.
    let started = Instant::now();
    let err = run(
        &w,
        "WITH range(1, 20000) AS xs UNWIND xs AS x RETURN xs[0] AS f, count(x) AS c",
        64 * MIB,
    )
    .await
    .expect_err("20,000 copies of a 20,000-item list exceed 64 MiB");
    assert!(matches!(err, ExecError::MemoryLimit(_)), "{err:?}");
    assert!(
        started.elapsed() < Duration::from_secs(5),
        "the memory limit took {:?}",
        started.elapsed()
    );
}
