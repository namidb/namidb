//! `NAMIDB_WRITE_LOOKUP_METER=0` restores 3.5.4: a write statement's node
//! lookups are not counted against its budget.
//!
//! This file must stay a single-test integration binary: it sets the
//! process-global switch before the first lookup reads it.

mod write_lookup_support;

use write_lookup_support::{lookup_plan, run, seed_docs};

use namidb_query::QueryMemoryBudget;
use std::sync::Arc;

#[tokio::test]
async fn the_write_lookup_meter_can_be_switched_off() {
    std::env::set_var("NAMIDB_WRITE_LOOKUP_METER", "0");
    let mut writer = seed_docs("lookup-meter-off", 900).await;
    let plan = lookup_plan(&writer);
    let outcome = run(
        &mut writer,
        &plan,
        900,
        Some(Arc::new(QueryMemoryBudget::new(1024 * 1024))),
    )
    .await
    .expect("with the meter off the lookup rows are not counted");
    assert_eq!(outcome.edges_created, 900);
    assert_eq!(namidb_query::exec::write_lookup_meter_stops(), 0);
}
