//! Per-query working-memory budget (issue #203).
//!
//! A query scoped with a `QueryMemoryBudget` must stop with
//! `ExecError::MemoryLimit` as soon as its estimated working set (result
//! rows, `collect()` lists, sort and DISTINCT state, cross products) would
//! exceed the budget, while queries that fit return exactly what they
//! return without a budget. The row cap keeps working alongside it.

use std::collections::BTreeMap;
use std::sync::Arc;

use namidb_core::id::{NamespaceId, NodeId};
use namidb_core::value::Value as CoreValue;
use namidb_storage::{EdgeWriteRecord, NamespacePaths, NodeWriteRecord, WriterSession};
use object_store::memory::InMemory;

use namidb_query::{
    execute, execute_with_limits, lower, parse, with_query_memory, ExecError, Params,
    QueryMemoryBudget, QueryMemoryPool, QueryMemoryScope, Row,
};

const MIB: usize = 1024 * 1024;

async fn seed(name: &str, people: usize) -> WriterSession {
    let paths = NamespacePaths::new("tenants", NamespaceId::new(name).unwrap());
    let mut writer = WriterSession::open(Arc::new(InMemory::new()), paths)
        .await
        .unwrap();
    for i in 0..people {
        let mut p = BTreeMap::new();
        p.insert("name".into(), CoreValue::Str(format!("person-{i}")));
        writer
            .upsert_node(
                "Person",
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

async fn run(
    writer: &WriterSession,
    cypher: &str,
    budget: Option<Arc<QueryMemoryBudget>>,
    row_cap: Option<usize>,
) -> Result<Vec<Row>, ExecError> {
    let snap = writer.snapshot();
    let plan = lower(&parse(cypher).unwrap()).unwrap();
    with_query_memory(
        budget,
        execute_with_limits(&plan, &snap, &Params::new(), None, row_cap),
    )
    .await
}

fn expect_query_scope(err: ExecError, limit: usize) {
    match err {
        ExecError::MemoryLimit(exceeded) => {
            assert_eq!(exceeded.scope, QueryMemoryScope::Query);
            assert_eq!(exceeded.limit_bytes, limit);
            assert!(exceeded.required_bytes > limit);
            assert!(!exceeded.is_transient());
        }
        other => panic!("expected MemoryLimit, got {other:?}"),
    }
}

/// The reproduction from #203: a 600k-row UNWIND that OOM-killed a 384 MiB
/// container. Under a 64 MiB budget it must fail fast, typed, and without
/// the budget ever being charged far past its limit (the growth meters stop
/// it mid-flight rather than after the operator finished).
#[tokio::test]
async fn six_hundred_thousand_row_unwind_fails_under_a_small_budget() {
    let writer = seed("mem-repro", 0).await;
    let limit = 64 * MIB;
    let budget = Arc::new(QueryMemoryBudget::new(limit));
    let err = run(
        &writer,
        "UNWIND range(1, 600000) AS i RETURN i, toString(i) AS s",
        Some(Arc::clone(&budget)),
        // The default row cap (1M) is in force too and does not fire.
        Some(1_000_000),
    )
    .await
    .unwrap_err();
    expect_query_scope(err, limit);
    // Never charged past the limit: refusals are not kept.
    assert!(budget.peak_bytes() <= limit, "peak {}", budget.peak_bytes());
    assert!(err_message_names_the_knob(&budget));
}

fn err_message_names_the_knob(budget: &QueryMemoryBudget) -> bool {
    // The Display text is what HTTP and Bolt clients read.
    let text = ExecError::MemoryLimit(namidb_query::QueryMemoryExceeded {
        scope: QueryMemoryScope::Query,
        required_bytes: budget.limit_bytes() + 1,
        limit_bytes: budget.limit_bytes(),
    })
    .to_string();
    text.contains("working-memory budget") && text.contains("NAMIDB_QUERY_MEMORY_MAX_BYTES")
}

#[tokio::test]
async fn queries_that_fit_return_the_same_rows() {
    let writer = seed("mem-fits", 50).await;
    let cypher = "UNWIND range(1, 5000) AS i RETURN i, toString(i) AS s ORDER BY i DESC";
    let snap = writer.snapshot();
    let plan = lower(&parse(cypher).unwrap()).unwrap();
    let unbudgeted = execute(&plan, &snap, &Params::new()).await.unwrap();
    let budget = Arc::new(QueryMemoryBudget::new(64 * MIB));
    let budgeted = run(&writer, cypher, Some(Arc::clone(&budget)), None)
        .await
        .unwrap();
    assert_eq!(budgeted, unbudgeted);
    assert_eq!(budgeted.len(), 5000);
    // Only the root result stays charged once the query is done, and it
    // was accounted at a realistic size (5000 narrow rows ~ 3-6 MiB).
    let charged = budget.used_bytes();
    assert!(charged >= 5000 * 650, "charged {charged}");
    assert!(charged < 16 * MIB, "charged {charged}");
    assert!(budget.peak_bytes() >= charged);

    let rows = run(
        &writer,
        "MATCH (p:Person) WHERE p.name STARTS WITH 'person-1' RETURN p.name AS name",
        Some(Arc::new(QueryMemoryBudget::new(64 * MIB))),
        None,
    )
    .await
    .unwrap();
    assert_eq!(rows.len(), 11); // person-1, person-10 .. person-19
}

#[tokio::test]
async fn row_cap_still_fires_first_when_it_is_lower() {
    let writer = seed("mem-rowcap", 0).await;
    let err = run(
        &writer,
        "UNWIND range(1, 900) AS a UNWIND range(1, 900) AS b RETURN a, b",
        Some(Arc::new(QueryMemoryBudget::new(1024 * MIB))),
        Some(1000),
    )
    .await
    .unwrap_err();
    assert!(matches!(err, ExecError::RowCap(1000)), "{err:?}");
}

/// A single row can be arbitrarily wide: `collect()` folds every input row
/// into one list, which a row cap cannot see at all.
#[tokio::test]
async fn collect_into_one_huge_list_is_bounded() {
    let writer = seed("mem-collect", 0).await;
    let limit = 32 * MIB;
    let err = run(
        &writer,
        "UNWIND range(1, 50000) AS i WITH i \
         RETURN collect(toString(i) + '-padding-padding-padding-padding') AS xs",
        Some(Arc::new(QueryMemoryBudget::new(limit))),
        Some(1_000_000),
    )
    .await
    .unwrap_err();
    expect_query_scope(err, limit);
}

#[tokio::test]
async fn sort_and_distinct_state_is_bounded() {
    let writer = seed("mem-sort", 0).await;
    let limit = 48 * MIB;
    for cypher in [
        "UNWIND range(1, 120000) AS i RETURN i, toString(i) AS s ORDER BY s",
        "UNWIND range(1, 120000) AS i RETURN DISTINCT toString(i) AS s",
    ] {
        let err = run(
            &writer,
            cypher,
            Some(Arc::new(QueryMemoryBudget::new(limit))),
            None,
        )
        .await
        .unwrap_err();
        expect_query_scope(err, limit);
    }
}

/// A cross product is refused from its estimated size before it allocates.
#[tokio::test]
async fn cross_product_is_refused_before_it_materialises() {
    let writer = seed("mem-cross", 1500).await;
    let limit = 32 * MIB;
    let budget = Arc::new(QueryMemoryBudget::new(limit));
    let err = run(
        &writer,
        "MATCH (a:Person), (b:Person) RETURN a.name, b.name",
        Some(Arc::clone(&budget)),
        Some(10_000_000),
    )
    .await
    .unwrap_err();
    expect_query_scope(err, limit);
}

/// Several queries that each fit their own budget cannot add up past the
/// shared pool: the one that finds it full gets the transient scope.
#[tokio::test]
async fn shared_pool_is_enforced_across_queries() {
    let writer = seed("mem-pool", 0).await;
    let pool = Arc::new(QueryMemoryPool::new(40 * MIB));
    // Something else holds most of the pool for the duration.
    let holder = QueryMemoryBudget::new(usize::MAX).with_pool(Arc::clone(&pool));
    holder.charge(36 * MIB).unwrap();

    let budget = Arc::new(QueryMemoryBudget::new(64 * MIB).with_pool(Arc::clone(&pool)));
    let err = run(
        &writer,
        "UNWIND range(1, 100000) AS i RETURN i, toString(i) AS s",
        Some(budget),
        None,
    )
    .await
    .unwrap_err();
    match err {
        ExecError::MemoryLimit(exceeded) => {
            assert_eq!(exceeded.scope, QueryMemoryScope::Shared);
            assert!(exceeded.is_transient());
            assert_eq!(exceeded.limit_bytes, 40 * MIB);
        }
        other => panic!("expected a shared-pool MemoryLimit, got {other:?}"),
    }
    // The failed query returned everything it had reserved.
    assert_eq!(pool.reserved_bytes(), 36 * MIB);
    drop(holder);
    assert_eq!(pool.reserved_bytes(), 0);

    // With the pool free again the same query fits.
    let budget = Arc::new(QueryMemoryBudget::new(64 * MIB).with_pool(Arc::clone(&pool)));
    let rows = run(
        &writer,
        "UNWIND range(1, 10000) AS i RETURN i, toString(i) AS s",
        Some(Arc::clone(&budget)),
        None,
    )
    .await
    .unwrap();
    assert_eq!(rows.len(), 10_000);
    assert!(pool.reserved_bytes() > 0);
    drop(budget);
    assert_eq!(pool.reserved_bytes(), 0);
}

/// A correlated subquery evaluated once per outer row must not accumulate
/// every evaluation's output in the account. Over a complete graph of 200
/// people, each `EXISTS` evaluation materialises ~199 expanded rows; kept
/// charged across all 200 outer rows that would be ~40k rows, well past a
/// 16 MiB budget, while any single evaluation fits easily.
#[tokio::test]
async fn per_row_subqueries_do_not_accumulate() {
    let paths = NamespacePaths::new("tenants", NamespaceId::new("mem-correlated").unwrap());
    let mut writer = WriterSession::open(Arc::new(InMemory::new()), paths)
        .await
        .unwrap();
    let ids: Vec<NodeId> = (0..200).map(|_| NodeId::new()).collect();
    for (i, id) in ids.iter().enumerate() {
        let mut p = BTreeMap::new();
        p.insert("name".into(), CoreValue::Str(format!("person-{i}")));
        p.insert("bio".into(), CoreValue::Str("x".repeat(256)));
        writer
            .upsert_node(
                "Person",
                *id,
                &NodeWriteRecord {
                    properties: p,
                    schema_version: 1,
                    ..Default::default()
                },
            )
            .unwrap();
    }
    for a in &ids {
        for b in &ids {
            if a != b {
                writer
                    .upsert_edge("KNOWS", *a, *b, &EdgeWriteRecord::default())
                    .unwrap();
            }
        }
    }
    writer.commit_batch().await.unwrap();

    let cypher = "MATCH (a:Person) WHERE EXISTS { MATCH (a)-[:KNOWS]->(b:Person) } \
                  RETURN count(a) AS n";
    // Sanity: the budget would be blown if every evaluation stayed charged.
    let one = run(
        &writer,
        "MATCH (a:Person {name: 'person-0'})-[:KNOWS]->(b:Person) RETURN a, b",
        None,
        None,
    )
    .await
    .unwrap();
    let one_evaluation = namidb_query::exec::memory::estimate_row_vec_bytes(&one);
    let limit = 16 * MIB;
    assert!(
        one_evaluation * 200 > limit,
        "one evaluation is {one_evaluation} bytes"
    );

    let budget = Arc::new(QueryMemoryBudget::new(limit));
    let rows = run(&writer, cypher, Some(Arc::clone(&budget)), None)
        .await
        .unwrap();
    assert_eq!(
        rows[0].get("n"),
        Some(&namidb_query::RuntimeValue::Integer(200))
    );
    assert!(budget.used_bytes() < MIB, "used {}", budget.used_bytes());
}

#[tokio::test]
async fn range_is_checked_before_it_allocates() {
    let writer = seed("mem-range", 0).await;
    let limit = 8 * MIB;
    let err = run(
        &writer,
        "RETURN size(range(1, 5000000)) AS n",
        Some(Arc::new(QueryMemoryBudget::new(limit))),
        None,
    )
    .await
    .unwrap_err();
    expect_query_scope(err, limit);
}
