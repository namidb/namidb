//! Composite index planner routing (phase 4): equality conjuncts covering
//! every member of a declared composite index must PLAN the
//! `NodeByPropertyTuple` lookup (reachability asserted on the plan shape
//! AND the storage route counter — result parity alone is trivially
//! satisfied by the scan fallback), serve the same rows as the scan
//! semantics, respect conjunct order and priority rules, and leave
//! partial-member shapes on their scan route.

use std::collections::BTreeMap;
use std::sync::Arc;

use namidb_core::id::{NamespaceId, NodeId};
use namidb_core::value::Value as CoreValue;
use namidb_storage::{route_telemetry, NamespacePaths, NodeWriteRecord, WriterSession};
use object_store::memory::InMemory;
use object_store::ObjectStore;

use namidb_query::{execute, lower, optimize, parse, LogicalPlan, Params, StatsCatalog};

const PEOPLE: i64 = 24;

fn person(ordinal: i64) -> NodeWriteRecord {
    let mut props: BTreeMap<String, CoreValue> = BTreeMap::new();
    let city = ["quito", "lima", "cuzco"][(ordinal % 3) as usize];
    props.insert("city".into(), CoreValue::Str(city.into()));
    props.insert("age".into(), CoreValue::I64(30 + ordinal % 4));
    props.insert("email".into(), CoreValue::Str(format!("p{ordinal}@x")));
    NodeWriteRecord {
        properties: props,
        schema_version: 1,
        ..Default::default()
    }
}

/// (quito, 30) holders: ordinal % 3 == 0 && ordinal % 4 == 0.
fn expected_quito_30() -> usize {
    (0..PEOPLE).filter(|o| o % 3 == 0 && o % 4 == 0).count()
}

async fn corpus(name: &str, flush: bool) -> WriterSession {
    let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
    let paths = NamespacePaths::new("tenants", NamespaceId::new(name).unwrap());
    let mut writer = WriterSession::open(store, paths).await.unwrap();
    for ordinal in 0..PEOPLE {
        writer
            .upsert_node("Person", NodeId::new(), &person(ordinal))
            .unwrap();
    }
    writer.commit_batch().await.unwrap();
    // Real DDL, exactly as the server routes it: the manifest schema (and
    // therefore the optimizer's catalog) must carry both declarations.
    writer
        .create_unique_constraint("Person", "email")
        .await
        .unwrap();
    writer
        .create_composite_index_named(None, "Person", &["city".into(), "age".into()], false)
        .await
        .unwrap();
    if flush {
        // Flush/compact with the POST-DDL manifest schema, exactly as the
        // server maintenance loop does — this is what materializes and
        // backfills the tuple sidecars.
        let committed = writer.snapshot().manifest().manifest.schema.clone();
        writer.flush(committed.clone()).await.unwrap();
        writer.compact_l0(&committed).await.unwrap();
    }
    writer
}

fn plan_has_tuple_lookup(plan: &LogicalPlan) -> bool {
    matches!(plan, LogicalPlan::NodeByPropertyTuple { .. })
        || plan.children().into_iter().any(plan_has_tuple_lookup)
}

fn plan_has_node_scan(plan: &LogicalPlan) -> bool {
    matches!(plan, LogicalPlan::NodeScan { .. })
        || plan.children().into_iter().any(plan_has_node_scan)
}

fn optimized(writer: &WriterSession, query: &str) -> LogicalPlan {
    let snapshot = writer.snapshot();
    let catalog = StatsCatalog::from_manifest(&snapshot.manifest().manifest);
    optimize(lower(&parse(query).unwrap()).unwrap(), &catalog)
}

async fn run(writer: &WriterSession, query: &str) -> Vec<String> {
    let snapshot = writer.snapshot();
    let plan = optimized(writer, query);
    let rows = execute(&plan, &snapshot, &Params::new()).await.unwrap();
    let mut out: Vec<String> = rows
        .iter()
        .map(|row| {
            let cells: Vec<String> = row
                .bindings
                .iter()
                .map(|(column, value)| format!("{column}={value:?}"))
                .collect();
            cells.join("|")
        })
        .collect();
    out.sort();
    out
}

const COVERED: &str = "MATCH (p:Person) WHERE p.city = 'quito' AND p.age = 30 \
                       RETURN p.email AS email";
const COVERED_FLIPPED: &str = "MATCH (p:Person) WHERE p.age = 30 AND p.city = 'quito' \
                               RETURN p.email AS email";
const COVERED_FLOAT: &str = "MATCH (p:Person) WHERE p.city = 'quito' AND p.age = 30.0 \
                             RETURN p.email AS email";
const RESIDUAL: &str = "MATCH (p:Person) WHERE p.city = 'quito' AND p.age = 30 \
                        AND p.email <> 'nobody@x' RETURN p.email AS email";

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn covered_conjuncts_plan_the_tuple_lookup_and_serve_natively() {
    for (route, flush) in [("memtable", false), ("flushed", true)] {
        let writer = corpus(&format!("cix-plan-{route}"), flush).await;

        // Reachability, half one: the PLAN carries the tuple operator (in
        // both conjunct spellings — the rewrite reorders to declaration
        // order) and the Person label scan is gone.
        for query in [COVERED, COVERED_FLIPPED] {
            let plan = optimized(&writer, query);
            assert!(
                plan_has_tuple_lookup(&plan),
                "{route}: covered conjuncts must plan NodeByPropertyTuple: {plan:?}"
            );
            assert!(
                !plan_has_node_scan(&plan),
                "{route}: the label scan must be replaced: {plan:?}"
            );
        }

        // Reachability, half two: execution actually SERVES through the
        // tuple route (parity alone would pass on the scan fallback).
        let before = route_telemetry::snapshot();
        let rows = run(&writer, COVERED).await;
        assert_eq!(rows.len(), expected_quito_30(), "{route}: row parity");
        let after = route_telemetry::snapshot();
        assert!(
            after.tuple_native > before.tuple_native,
            "{route}: the tuple route must serve natively"
        );
        assert_eq!(
            rows,
            run(&writer, COVERED_FLIPPED).await,
            "{route}: conjunct order must not change results"
        );
        // Cypher numeric coercion holds on the index route.
        assert_eq!(
            rows,
            run(&writer, COVERED_FLOAT).await,
            "{route}: 30 = 30.0 must hold through the tuple key"
        );
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn residual_conjuncts_keep_filtering_above_the_tuple_lookup() {
    let writer = corpus("cix-plan-residual", true).await;
    let plan = optimized(&writer, RESIDUAL);
    assert!(plan_has_tuple_lookup(&plan), "{plan:?}");
    let rows = run(&writer, RESIDUAL).await;
    assert_eq!(
        rows.len(),
        expected_quito_30(),
        "the <> residual filters nothing here but must not break results"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn partial_member_conjuncts_keep_the_scan_route() {
    let writer = corpus("cix-plan-partial", true).await;
    // Only one member: a partial tuple can NOT use the index (the recon's
    // negative row) — the plan keeps its scan, results stay exact.
    let query = "MATCH (p:Person) WHERE p.city = 'quito' RETURN p.email AS email";
    let plan = optimized(&writer, query);
    assert!(
        !plan_has_tuple_lookup(&plan),
        "a partial member set must not plan the tuple lookup: {plan:?}"
    );
    let rows = run(&writer, query).await;
    assert_eq!(rows.len(), (0..PEOPLE).filter(|o| o % 3 == 0).count());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn unique_property_lookup_outranks_the_composite() {
    let writer = corpus("cix-plan-priority", true).await;
    // A unique-property conjunct gives at most one row; it must win over
    // the composite even though the tuple is also fully covered.
    let query = "MATCH (p:Person) WHERE p.email = 'p0@x' AND p.city = 'quito' \
                 AND p.age = 30 RETURN p.email AS email";
    let plan = optimized(&writer, query);
    assert!(
        !plan_has_tuple_lookup(&plan),
        "the unique email lookup must outrank the composite: {plan:?}"
    );
    fn has_unique_lookup(plan: &LogicalPlan) -> bool {
        matches!(plan, LogicalPlan::NodeByPropertyValue { multi: false, .. })
            || plan.children().into_iter().any(has_unique_lookup)
    }
    assert!(has_unique_lookup(&plan), "{plan:?}");
    let rows = run(&writer, query).await;
    assert_eq!(rows.len(), 1);
}
