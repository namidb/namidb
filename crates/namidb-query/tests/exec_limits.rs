//! Read-query execution guards: the wall-clock timeout and the operator
//! row cap. A deadline already in the past aborts with `ExecError::Timeout`
//! (a generous deadline or none runs to completion); an operator that would
//! materialise more rows than the cap aborts with `ExecError::RowCap`,
//! including the multiplicative cross product, which is rejected before it
//! materialises.

use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use namidb_core::id::{NamespaceId, NodeId};
use namidb_core::value::Value as CoreValue;
use namidb_storage::{NamespacePaths, NodeWriteRecord, WriterSession};
use object_store::memory::InMemory;
use object_store::ObjectStore;

use namidb_query::{execute_with_limits, lower, parse, ExecError, Params};

fn store() -> Arc<dyn ObjectStore> {
    Arc::new(InMemory::new())
}

fn paths(name: &str) -> NamespacePaths {
    NamespacePaths::new("tenants", NamespaceId::new(name).unwrap())
}

async fn seed(name: &str) -> WriterSession {
    let mut writer = WriterSession::open(store(), paths(name)).await.unwrap();
    for n in ["Alice", "Bob", "Carol"] {
        let mut p = BTreeMap::new();
        p.insert("name".into(), CoreValue::Str(n.into()));
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

#[tokio::test]
async fn past_deadline_aborts_with_timeout() {
    let writer = seed("limits-timeout").await;
    let snap = writer.snapshot();
    let plan = lower(&parse("MATCH (p:Person) RETURN p").unwrap()).unwrap();

    // A deadline already in the past: the per-operator guard fires on the
    // first check.
    let past = Instant::now() - Duration::from_millis(1);
    let err = execute_with_limits(&plan, &snap, &Params::new(), Some(past), None)
        .await
        .expect_err("a past deadline must abort the read");
    assert!(matches!(err, ExecError::Timeout), "got {err:?}");
}

#[tokio::test]
async fn generous_deadline_completes() {
    let writer = seed("limits-ok").await;
    let snap = writer.snapshot();
    let plan = lower(&parse("MATCH (p:Person) RETURN p").unwrap()).unwrap();

    let future = Instant::now() + Duration::from_secs(60);
    let rows = execute_with_limits(&plan, &snap, &Params::new(), Some(future), None)
        .await
        .expect("a generous deadline must not abort");
    assert_eq!(rows.len(), 3);
}

#[tokio::test]
async fn no_deadline_is_unbounded() {
    let writer = seed("limits-none").await;
    let snap = writer.snapshot();
    let plan = lower(&parse("MATCH (p:Person) RETURN p").unwrap()).unwrap();

    let rows = execute_with_limits(&plan, &snap, &Params::new(), None, None)
        .await
        .expect("no deadline runs unbounded");
    assert_eq!(rows.len(), 3);
}

#[tokio::test]
async fn row_cap_aborts_a_scan_over_the_cap() {
    // Three Person nodes; a cap of 2 must reject the scan.
    let writer = seed("limits-rowcap").await;
    let snap = writer.snapshot();
    let plan = lower(&parse("MATCH (p:Person) RETURN p").unwrap()).unwrap();

    let err = execute_with_limits(&plan, &snap, &Params::new(), None, Some(2))
        .await
        .expect_err("a scan over the row cap must abort");
    assert!(matches!(err, ExecError::RowCap(2)), "got {err:?}");
}

#[tokio::test]
async fn row_cap_at_or_above_result_size_completes() {
    let writer = seed("limits-rowcap-ok").await;
    let snap = writer.snapshot();
    let plan = lower(&parse("MATCH (p:Person) RETURN p").unwrap()).unwrap();

    // Exactly the result size is allowed (cap is "must not exceed").
    let rows = execute_with_limits(&plan, &snap, &Params::new(), None, Some(3))
        .await
        .expect("a cap at the result size must not abort");
    assert_eq!(rows.len(), 3);
}

#[tokio::test]
async fn row_cap_rejects_a_cross_product_before_materialising() {
    // Two Person scans (3 each) crossed = 9 rows; a cap of 5 must abort,
    // and via the pre-multiply check, before building the product.
    let writer = seed("limits-rowcap-cross").await;
    let snap = writer.snapshot();
    let plan = lower(&parse("MATCH (a:Person) MATCH (b:Person) RETURN a, b").unwrap()).unwrap();

    let err = execute_with_limits(&plan, &snap, &Params::new(), None, Some(5))
        .await
        .expect_err("a cross product over the cap must abort");
    assert!(matches!(err, ExecError::RowCap(5)), "got {err:?}");
}

/// Item 60: the row cap must bound MEMORY, not merely veto the result
/// after the full expansion — a bare `UNWIND range(1, 5M)` used to
/// materialise every row (and the whole range list) before the cap was
/// evaluated, OOM-killing a 4 GB container. The range() guard now rejects
/// the list before it allocates, and the unwind loop checks incrementally.
#[tokio::test]
async fn giant_unwind_range_is_rejected_before_it_materialises() {
    let writer = seed("unwind-cap").await;
    let snap = writer.snapshot();

    // The range() materialization guard fires first: the list itself would
    // exceed the cap, so nothing multi-million-sized is ever allocated.
    let plan =
        lower(&parse("UNWIND range(1, 5000000) AS i RETURN count(i) AS n").unwrap()).unwrap();
    let err = execute_with_limits(&plan, &snap, &Params::new(), None, Some(1000))
        .await
        .unwrap_err();
    assert!(
        matches!(&err, ExecError::Eval(e) if e.message.contains("row cap")),
        "range() must be rejected at construction: {err:?}"
    );

    // Per-row lists individually UNDER the cap still trip the INCREMENTAL
    // unwind check once the produced rows multiply past it: 3 rows x
    // 400-item lists = 1200 produced rows against a 1000 cap (each range()
    // passes its own construction guard).
    let plan =
        lower(&parse("MATCH (p:Person) UNWIND range(1, 400) AS i RETURN count(i) AS n").unwrap())
            .unwrap();
    let err = execute_with_limits(&plan, &snap, &Params::new(), None, Some(1000))
        .await
        .unwrap_err();
    assert!(matches!(err, ExecError::RowCap(1000)), "got {err:?}");

    // Without a cap in scope (embedded / CLI), behavior is unchanged.
    let plan = lower(&parse("UNWIND range(1, 5000) AS i RETURN count(i) AS n").unwrap()).unwrap();
    let rows = execute_with_limits(&plan, &snap, &Params::new(), None, None)
        .await
        .unwrap();
    assert_eq!(rows.len(), 1);
}

/// The guard extends to non-UNWIND range() uses when a cap is scoped: a
/// list the statement may not materialise as rows may not be materialised
/// as one giant value either.
#[tokio::test]
async fn range_materialization_respects_the_scoped_cap() {
    let writer = seed("range-cap").await;
    let snap = writer.snapshot();
    let plan = lower(&parse("RETURN size(range(1, 100000)) AS n").unwrap()).unwrap();
    let err = execute_with_limits(&plan, &snap, &Params::new(), None, Some(1000))
        .await
        .unwrap_err();
    assert!(
        matches!(&err, ExecError::Eval(e) if e.message.contains("row cap")),
        "{err:?}"
    );
    // Under the cap: exact size still returned.
    let plan = lower(&parse("RETURN size(range(1, 500)) AS n").unwrap()).unwrap();
    let rows = execute_with_limits(&plan, &snap, &Params::new(), None, Some(1000))
        .await
        .unwrap();
    assert_eq!(rows.len(), 1);
    // Opposite-direction ranges are empty, never rejected.
    let plan = lower(&parse("RETURN size(range(10, 1)) AS n").unwrap()).unwrap();
    let rows = execute_with_limits(&plan, &snap, &Params::new(), None, Some(5))
        .await
        .unwrap();
    assert_eq!(rows.len(), 1);
}
