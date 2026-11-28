//! A write statement's node lookups count the nodes they add to its rows
//! against the statement's working-memory budget, including the rows after
//! the last metering stride.

mod write_lookup_support;

use write_lookup_support::{lookup_plan, run, seed_docs};

use namidb_query::{ExecError, QueryMemoryBudget};
use std::sync::Arc;

/// 900 lookup rows of ~4 KiB each stop short of the first metering stride
/// (1024 rows); they are still counted at the end of the lookup.
#[tokio::test]
async fn a_lookup_short_of_one_stride_is_still_metered() {
    let mut writer = seed_docs("lookup-meter-tail", 900).await;
    let plan = lookup_plan(&writer);
    let stops_before = namidb_query::exec::write_lookup_meter_stops();
    let err = run(
        &mut writer,
        &plan,
        900,
        Some(Arc::new(QueryMemoryBudget::new(1024 * 1024))),
    )
    .await
    .expect_err("3.6 MB of lookup rows do not fit a 1 MiB budget");
    assert!(matches!(err, ExecError::MemoryLimit(_)), "got {err:?}");
    assert!(namidb_query::exec::write_lookup_meter_stops() > stops_before);

    // Without a budget the same statement runs.
    let outcome = run(&mut writer, &plan, 900, None).await.unwrap();
    assert_eq!(outcome.edges_created, 900);
}
