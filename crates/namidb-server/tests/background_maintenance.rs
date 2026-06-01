//! A1: background compaction + orphan sweep.
//!
//! The server's maintenance task collapses L0 SSTs into L1 (bounding read
//! amplification) and sweeps the orphaned L0 bodies compaction leaves
//! behind. This test drives ONE maintenance tick inline (no wall-clock
//! dependency) against the same `AppState` the server uses, and pins the
//! contract:
//!
//! 1. Seeding three flushed batches creates >= 2 L0 SSTs in one bucket.
//! 2. `compact_l0` collapses them to a single L1 SST and removes the L0s.
//! 3. Reads return the same data before and after — including a snapshot
//!    pinned *before* compaction (the source bodies survive for it).
//! 4. The orphan sweep finds the now-unreferenced L0 bodies, deletes them
//!    when asked, and a re-run finds nothing — and reads still hold.

use std::sync::Arc;

use namidb_query::{
    execute, execute_write, parse as cypher_parse, plan as build_plan, Params, RuntimeValue,
    StatsCatalog,
};
use namidb_storage::{
    sweep_orphans, ManifestStore, OwnedSnapshot, SstKind, SstLevel, WriterSession,
};

const NS: &str = "maint-test";

async fn create_person(state: &namidb_server::AppState, i: usize) {
    let q = format!(
        "CREATE (a:Person {{name: 'p{i}', age: {}}})",
        (i % 80) as i64
    );
    let parsed = cypher_parse(&q).expect("parse");
    let mut w = state.writer.lock().await;
    let catalog = StatsCatalog::from_manifest(&w.snapshot().manifest().manifest);
    let plan = build_plan(&parsed, &catalog).expect("plan");
    execute_write(&plan, &mut w, &Params::new())
        .await
        .expect("write");
    state.snapshot.store(w.owned_snapshot());
}

/// Flush the live memtable into one L0 SST per touched bucket and
/// republish.
async fn flush(state: &namidb_server::AppState) {
    let mut w = state.writer.lock().await;
    let schema = w.snapshot().manifest().manifest.schema.clone();
    w.flush(schema).await.expect("flush");
    state.snapshot.store(w.owned_snapshot());
}

fn person_ssts(snap: &OwnedSnapshot, level: SstLevel) -> usize {
    snap.manifest()
        .manifest
        .ssts
        .iter()
        .filter(|d| d.kind == SstKind::Nodes && d.scope == "Person" && d.level == level)
        .count()
}

/// `MATCH (p:Person) RETURN count(p)` against a pinned snapshot.
async fn count_persons(snap: &Arc<OwnedSnapshot>) -> i64 {
    let parsed = cypher_parse("MATCH (p:Person) RETURN count(p) AS c").expect("parse");
    let catalog = StatsCatalog::from_manifest(&snap.manifest().manifest);
    let plan = build_plan(&parsed, &catalog).expect("plan");
    let borrowed = snap.borrow();
    let rows = execute(&plan, &borrowed, &Params::new())
        .await
        .expect("read");
    match rows.first().and_then(|r| r.get("c")) {
        Some(RuntimeValue::Integer(n)) => *n,
        other => panic!("unexpected count row: {other:?}"),
    }
}

#[tokio::test]
async fn compaction_collapses_l0_and_sweep_reclaims_orphans() {
    let (store, paths) = namidb_storage::parse_uri(&format!("memory://{NS}")).unwrap();
    // The sweep loads the committed manifest itself, so it needs its own
    // ManifestStore built from the same (store, paths) — exactly what the
    // server does in `run()`.
    let manifest_store = ManifestStore::new(store.clone(), paths.clone());
    let writer = WriterSession::open(store, paths).await.unwrap();
    let state = namidb_server::AppState::new(writer, None, NS.into());

    // Three flushed batches → three L0 SSTs in the (Nodes, Person) bucket.
    const BATCHES: usize = 3;
    const PER_BATCH: usize = 4;
    for b in 0..BATCHES {
        for j in 0..PER_BATCH {
            create_person(&state, b * PER_BATCH + j).await;
        }
        flush(&state).await;
    }
    let total = (BATCHES * PER_BATCH) as i64;

    let before = state.snapshot.load();
    assert!(
        person_ssts(&before, SstLevel::L0) >= 2,
        "expected >= 2 L0 Person SSTs before compaction, got {}",
        person_ssts(&before, SstLevel::L0)
    );
    assert_eq!(count_persons(&before).await, total, "baseline count");

    // Pin a snapshot from before compaction: its source L0 bodies must
    // remain readable even after compaction drops them from the manifest.
    let pinned_pre_compaction = state.snapshot.load();

    // ── One maintenance tick: compaction under the writer lock ──
    {
        let mut w = state.writer.lock().await;
        let schema = w.snapshot().manifest().manifest.schema.clone();
        let outcome = w.compact_l0(&schema).await.expect("compact");
        assert!(
            outcome.source_ssts_removed >= 2,
            "compaction should consume >= 2 L0 SSTs, removed {}",
            outcome.source_ssts_removed
        );
        assert!(
            outcome.new_ssts_written >= 1,
            "compaction should write >= 1 L1 SST, wrote {}",
            outcome.new_ssts_written
        );
        state.snapshot.store(w.owned_snapshot());
    }

    let after = state.snapshot.load();
    assert_eq!(
        person_ssts(&after, SstLevel::L0),
        0,
        "no L0 Person SSTs should remain after compaction"
    );
    assert_eq!(
        person_ssts(&after, SstLevel(1)),
        1,
        "the bucket should collapse to a single L1 SST"
    );

    // Reads stay correct on the new snapshot AND on the pre-compaction one.
    assert_eq!(count_persons(&after).await, total, "count after compaction");
    assert_eq!(
        count_persons(&pinned_pre_compaction).await,
        total,
        "a snapshot pinned before compaction still reads its source bodies"
    );

    // ── Orphan sweep: the dropped L0 bodies are now unreferenced. ──
    // min_age = 0 in-test (production default is 24h); max_level = 1.
    let dry = sweep_orphans(&manifest_store, std::time::Duration::ZERO, 1, false)
        .await
        .expect("dry-run sweep");
    assert!(
        dry.orphans_found >= 2,
        "dry-run should find >= 2 orphaned L0 bodies, found {}",
        dry.orphans_found
    );
    assert_eq!(dry.orphans_deleted, 0, "dry-run deletes nothing");

    let del = sweep_orphans(&manifest_store, std::time::Duration::ZERO, 1, true)
        .await
        .expect("delete sweep");
    assert_eq!(
        del.orphans_deleted, dry.orphans_found,
        "delete sweep removes exactly the orphans the dry-run found"
    );

    let dry2 = sweep_orphans(&manifest_store, std::time::Duration::ZERO, 1, false)
        .await
        .expect("second dry-run");
    assert_eq!(dry2.orphans_found, 0, "no orphans remain after the sweep");

    // The live L1 SST was untouched: a fresh read still returns everything.
    let final_snap = state.snapshot.load();
    assert_eq!(
        count_persons(&final_snap).await,
        total,
        "count still correct after the sweep deleted the orphaned bodies"
    );
}
