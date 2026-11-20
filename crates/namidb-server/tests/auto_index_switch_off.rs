//! `NAMIDB_AUTO_INDEX=0` turns the automatic property index builder off for
//! the whole process: statements that would offer an index queue nothing, no
//! index is declared, and the manifest's schema is left as the user wrote it.
//! The positive control is `auto_index_switch_on.rs` (a separate process,
//! since the switch is read once).

#![cfg(feature = "text-index")]

#[path = "support/auto_index.rs"]
mod support;

use std::time::Duration;

use support::*;

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn the_off_switch_disables_the_automatic_index_builder() {
    // Read once per process, before the first statement: lower the label
    // threshold so the small fixture qualifies, and keep builds short.
    std::env::set_var("NAMIDB_AUTO_INDEX_MIN_NODES", "10");
    std::env::set_var("NAMIDB_AUTO_INDEX_BUILD_WAIT_SECS", "20");
    std::env::set_var("NAMIDB_AUTO_INDEX", "0");
    let dir = tempfile::tempdir().unwrap();
    let base = boot(format!("file://{}?ns={NS}", dir.path().display())).await;
    offer_hints(&base).await;
    // Longer than the worker's poll and a short build would take with the
    // builder on (the positive control sees it within this window).
    let created = eventually(Duration::from_secs(8), || async {
        index_names(&base)
            .await
            .iter()
            .any(|name| name.starts_with("auto_index_"))
    })
    .await;
    assert!(
        !created,
        "an automatic index was declared with NAMIDB_AUTO_INDEX=0: {:?}",
        index_names(&base).await
    );
    for outcome in [
        "created",
        "online",
        "skipped_search_rebuild",
        "skipped_cap",
        "skipped_other",
        "failed",
    ] {
        assert_eq!(
            auto_index_counter(&base, outcome).await,
            0,
            "outcome {outcome}"
        );
    }
    // A restart with the switch still off leaves the schema untouched too.
    let (store, paths) =
        namidb_storage::parse_uri(&format!("file://{}?ns={NS}", dir.path().display())).unwrap();
    let manifest = namidb_storage::ManifestStore::new(store, paths)
        .load_current()
        .await
        .unwrap();
    assert!(
        manifest
            .manifest
            .schema
            .labels
            .values()
            .flat_map(|l| l.properties.iter())
            .all(|p| !p.auto_indexed),
        "the manifest gained an automatic index"
    );
}
