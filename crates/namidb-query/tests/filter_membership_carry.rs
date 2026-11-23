//! A carry or patch of a kept filter membership set that reads new rows
//! runs once per set, in the background, off every statement's deadline: a
//! statement waits at most a short grace on it and otherwise answers by
//! resolving the filter whole. A failed carry backs off instead of being
//! retried by every statement. Answers stay exact throughout.
//!
//! The carry knobs are process-wide, so this file holds a single test.

use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use namidb_core::id::{NamespaceId, NodeId};
use namidb_core::schema::{DataType, LabelDef, PropertyDef, Schema, SchemaBuilder};
use namidb_core::value::Value as CoreValue;
use namidb_query::{
    cancel_filter_membership_carries, filter_membership_carries, filter_membership_carry_pending,
    filter_membership_ids_for_tests, set_filter_membership_carry_for_tests, with_query_memory,
    ExecError, QueryMemoryBudget, QueryMemoryPool, RuntimeValue,
};
use namidb_storage::{NamespacePaths, NodeWriteRecord, WriterSession};
use object_store::memory::InMemory;
use object_store::ObjectStore;

const LABEL: &str = "Doc";

fn schema() -> Schema {
    SchemaBuilder::new()
        .label(LabelDef {
            name: LABEL.into(),
            properties: vec![PropertyDef::new("vigente", DataType::Bool, true).unwrap()],
        })
        .unwrap()
        .build()
}

fn record(vigente: bool) -> NodeWriteRecord {
    let mut properties = BTreeMap::new();
    properties.insert("vigente".into(), CoreValue::Bool(vigente));
    NodeWriteRecord {
        properties,
        schema_version: 1,
        ..Default::default()
    }
}

fn vigente() -> RuntimeValue {
    RuntimeValue::Map(
        [("vigente".to_string(), RuntimeValue::Bool(true))]
            .into_iter()
            .collect(),
    )
}

/// The pool every statement's (and so every carry's) account reserves from.
fn pool() -> &'static Arc<QueryMemoryPool> {
    static POOL: std::sync::OnceLock<Arc<QueryMemoryPool>> = std::sync::OnceLock::new();
    POOL.get_or_init(|| Arc::new(QueryMemoryPool::new(1 << 30)))
}

/// One statement's answer under `deadline` and a pooled memory account.
async fn answer(
    writer: &WriterSession,
    deadline: Duration,
) -> Result<Option<Vec<NodeId>>, ExecError> {
    let snapshot = writer.snapshot();
    let budget = Arc::new(QueryMemoryBudget::new(256 << 20).with_pool(pool().clone()));
    with_query_memory(
        Some(budget),
        namidb_storage::cancel::with_deadline(
            Some(Instant::now() + deadline),
            filter_membership_ids_for_tests(&snapshot, LABEL, &vigente(), false),
        ),
    )
    .await
}

fn assert_pool_idle(context: &str) {
    assert_eq!(
        pool().reserved_bytes(),
        0,
        "{context}: the pool is back at its baseline"
    );
}

/// One statement under `deadline`: its answer, which must equal the model,
/// and how long it took.
async fn statement(
    writer: &WriterSession,
    docs: &BTreeMap<NodeId, bool>,
    deadline: Duration,
    context: &str,
) -> Duration {
    let started = Instant::now();
    let served = answer(writer, deadline)
        .await
        .unwrap_or_else(|error| panic!("{context}: {error}"));
    let elapsed = started.elapsed();
    let model = docs
        .iter()
        .filter(|(_, vigente)| **vigente)
        .map(|(id, _)| *id)
        .collect::<Vec<_>>();
    assert_eq!(served.as_deref(), Some(model.as_slice()), "{context}");
    elapsed
}

async fn flip_and_flush(writer: &mut WriterSession, docs: &mut BTreeMap<NodeId, bool>, n: usize) {
    let ids = docs.keys().copied().take(n).collect::<Vec<_>>();
    for id in ids {
        let value = docs.get_mut(&id).unwrap();
        *value = !*value;
        writer.upsert_node(LABEL, id, &record(*value)).unwrap();
    }
    writer.commit_batch().await.unwrap();
    writer.flush(schema()).await.unwrap();
}

async fn settle() {
    let until = Instant::now() + Duration::from_secs(20);
    while filter_membership_carry_pending() {
        assert!(Instant::now() < until, "a background carry never settled");
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
}

fn knobs(wait_ms: u64, backoff_ms: u64, max: Option<u64>, delay_ms: u64, fail: bool) {
    set_filter_membership_carry_for_tests(Some(wait_ms), Some(backoff_ms), max, delay_ms, fail);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_slow_or_failing_carry_never_holds_a_statement_past_its_deadline() {
    let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
    let mut writer = WriterSession::open(store, paths()).await.unwrap();
    let mut docs = BTreeMap::new();
    for n in 0..300 {
        let id = NodeId::new();
        writer.upsert_node(LABEL, id, &record(n % 3 == 0)).unwrap();
        docs.insert(id, n % 3 == 0);
    }
    writer.commit_batch().await.unwrap();
    writer.flush(schema()).await.unwrap();
    let deadline = Duration::from_millis(1500);
    let long = Duration::from_secs(30);
    // Kept at this flushed corpus.
    statement(&writer, &docs, long, "built").await;

    // A carry far slower than the deadline: every statement answers within
    // it, through the whole-filter route, and the carry starts once.
    knobs(50, 400, None, 4000, false);
    flip_and_flush(&mut writer, &mut docs, 20).await;
    let before = filter_membership_carries();
    for round in 0..3 {
        let elapsed = statement(&writer, &docs, deadline, &format!("slow carry {round}")).await;
        assert!(elapsed < deadline, "round {round} took {elapsed:?}");
    }
    let after = filter_membership_carries();
    assert_eq!(after.started, before.started + 1, "one carry per kept set");
    assert_eq!(
        after.deferred,
        before.deferred + 3,
        "every statement answered without it"
    );

    // Once it settles, its set is kept: answered without a new carry, and
    // the carry's own account has returned its reservation.
    settle().await;
    assert_pool_idle("after a carry completed");
    let before = filter_membership_carries();
    statement(&writer, &docs, deadline, "carried").await;
    assert_eq!(
        filter_membership_carries(),
        before,
        "the carried set answers"
    );

    // A failing carry backs off: statements answer exactly meanwhile, and
    // only one attempt is made until the back-off passes.
    knobs(50, 400, None, 0, true);
    flip_and_flush(&mut writer, &mut docs, 20).await;
    let before = filter_membership_carries();
    statement(&writer, &docs, deadline, "failing carry").await;
    settle().await;
    for round in 0..3 {
        statement(&writer, &docs, deadline, &format!("backing off {round}")).await;
    }
    let after = filter_membership_carries();
    assert_eq!(
        after.started,
        before.started + 1,
        "no retry before the back-off passes"
    );
    assert_eq!(after.failed, before.failed + 1, "the attempt failed");
    assert_eq!(
        after.deferred,
        before.deferred + 4,
        "each statement answered without it"
    );

    // After the back-off a healthy carry runs and its set answers again.
    tokio::time::sleep(Duration::from_millis(450)).await;
    knobs(10_000, 400, None, 0, false);
    let before = filter_membership_carries();
    statement(&writer, &docs, long, "retried").await;
    let after = filter_membership_carries();
    assert_eq!(
        after.started,
        before.started + 1,
        "retried after the back-off"
    );
    assert_eq!(
        after.deferred, before.deferred,
        "the retried carry answered"
    );
    statement(&writer, &docs, deadline, "kept after the retry").await;

    // The starting statement times out, or its client goes away: the carry
    // has its own task, token and account, and still completes and is kept.
    knobs(1000, 400, None, 300, false);
    flip_and_flush(&mut writer, &mut docs, 20).await;
    let before = filter_membership_carries();
    let started = Instant::now();
    let timed = answer(&writer, Duration::from_millis(50)).await;
    assert!(
        matches!(timed, Ok(_) | Err(ExecError::Timeout)),
        "a statement past its deadline: {timed:?}"
    );
    assert!(
        started.elapsed() < Duration::from_millis(300),
        "it did not wait out the carry"
    );
    settle().await;
    assert_pool_idle("after the timed-out statement's carry");
    let after = filter_membership_carries();
    assert_eq!(after.started, before.started + 1);
    assert_eq!(
        (after.cancelled, after.failed),
        (before.cancelled, before.failed)
    );
    statement(&writer, &docs, deadline, "kept after a timed-out start").await;
    assert_eq!(
        filter_membership_carries(),
        after,
        "the carried set answers"
    );

    flip_and_flush(&mut writer, &mut docs, 20).await;
    let before = filter_membership_carries();
    let dropped = tokio::time::timeout(Duration::from_millis(20), answer(&writer, long)).await;
    assert!(dropped.is_err(), "the statement was dropped mid-wait");
    settle().await;
    assert_pool_idle("after the dropped statement's carry");
    let after = filter_membership_carries();
    assert_eq!(after.started, before.started + 1);
    assert_eq!(
        (after.cancelled, after.failed),
        (before.cancelled, before.failed)
    );
    statement(&writer, &docs, deadline, "kept after a dropped start").await;
    assert_eq!(
        filter_membership_carries(),
        after,
        "the carried set answers"
    );

    // Over the concurrency cap a carry is neither started nor queued.
    knobs(50, 400, Some(0), 0, false);
    flip_and_flush(&mut writer, &mut docs, 20).await;
    let before = filter_membership_carries();
    statement(&writer, &docs, deadline, "over the cap").await;
    statement(&writer, &docs, deadline, "still over the cap").await;
    let after = filter_membership_carries();
    assert_eq!(after.started, before.started, "no carry over the cap");
    assert_eq!(after.deferred, before.deferred + 2);
    assert!(!filter_membership_carry_pending(), "nothing queued");

    // A carry toward a corpus a newer flush replaced is cancelled once; the
    // carry that replaced it is not cancelled again, so it finishes.
    knobs(50, 400, None, 4000, false);
    statement(&writer, &docs, deadline, "toward the first corpus").await;
    flip_and_flush(&mut writer, &mut docs, 20).await;
    let before = filter_membership_carries();
    statement(&writer, &docs, deadline, "toward the second corpus").await;
    flip_and_flush(&mut writer, &mut docs, 20).await;
    statement(&writer, &docs, deadline, "toward the third corpus").await;
    let until = Instant::now() + Duration::from_secs(5);
    while filter_membership_carries().cancelled == before.cancelled {
        assert!(Instant::now() < until, "the stale carry was not cancelled");
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    let after = filter_membership_carries();
    assert_eq!(after.cancelled, before.cancelled + 1, "cancelled once");
    assert_eq!(
        after.started,
        before.started + 1,
        "one replacement, then none"
    );
    settle().await;
    assert_eq!(filter_membership_carries().cancelled, before.cancelled + 1);
    knobs(10_000, 400, None, 0, false);
    statement(&writer, &docs, long, "patched from the replacement").await;

    // Closing the namespace cancels its carries and drops its kept sets.
    knobs(50, 400, None, 4000, false);
    flip_and_flush(&mut writer, &mut docs, 20).await;
    statement(&writer, &docs, deadline, "before close").await;
    assert!(filter_membership_carry_pending());
    let before = filter_membership_carries();
    cancel_filter_membership_carries(paths().namespace_prefix().as_ref());
    assert!(
        !filter_membership_carry_pending(),
        "closing forgets its carries"
    );
    let until = Instant::now() + Duration::from_secs(5);
    while filter_membership_carries().cancelled == before.cancelled {
        assert!(Instant::now() < until, "closing did not cancel the carry");
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    let after = filter_membership_carries();
    assert_eq!(
        after.cancelled,
        before.cancelled + 1,
        "closing cancels the carry"
    );
    assert_pool_idle("after a carry was cancelled");
    let before = after;
    statement(&writer, &docs, long, "after close").await;
    let after = filter_membership_carries();
    assert_eq!(after.started, before.started, "nothing kept to carry");
    assert_eq!(after.deferred, before.deferred);

    set_filter_membership_carry_for_tests(None, None, None, 0, false);
}

fn paths() -> NamespacePaths {
    NamespacePaths::new("tenants", NamespaceId::new("membership-carry").unwrap())
}
