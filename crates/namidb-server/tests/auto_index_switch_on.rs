//! Positive control of `auto_index_switch_off.rs`: with the builder on, the
//! same statements declare `auto_index_Item_level` and it comes online, so
//! the off test is not passing vacuously.

#![cfg(feature = "text-index")]

#[path = "support/auto_index.rs"]
mod support;

use std::time::Duration;

use support::*;

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn the_builder_declares_the_offered_index_when_on() {
    // Read once per process, before the first statement: lower the label
    // threshold so the small fixture qualifies, and keep builds short.
    std::env::set_var("NAMIDB_AUTO_INDEX_MIN_NODES", "10");
    std::env::set_var("NAMIDB_AUTO_INDEX_BUILD_WAIT_SECS", "20");
    std::env::remove_var("NAMIDB_AUTO_INDEX");
    let dir = tempfile::tempdir().unwrap();
    let base = boot(format!("file://{}?ns={NS}", dir.path().display())).await;
    offer_hints(&base).await;
    let created = eventually(Duration::from_secs(60), || async {
        index_names(&base)
            .await
            .iter()
            .any(|name| name == AUTO_INDEX)
    })
    .await;
    if !created {
        let mut outcomes = Vec::new();
        for outcome in [
            "created",
            "online",
            "skipped_search_rebuild",
            "skipped_cap",
            "skipped_other",
            "failed",
        ] {
            outcomes.push((outcome, auto_index_counter(&base, outcome).await));
        }
        panic!(
            "no {AUTO_INDEX}: {:?}; outcomes {outcomes:?}",
            index_names(&base).await
        );
    }
    assert!(auto_index_counter(&base, "created").await >= 1);
}
