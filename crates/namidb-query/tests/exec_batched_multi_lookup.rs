//! Item 63 executor half: the non-unique labeled equality arm batches the
//! whole statement (mirroring the unique arm) — and the results are
//! byte-identical to the per-row route, including duplicate lookup values,
//! misses, and a mixed-type UNWIND list that forces the per-row fallback
//! to interleave with the batch without losing alignment. Per the project
//! reachability rule, the plan is asserted to carry the multi lookup
//! operator, not just equal results.

use std::collections::BTreeMap;
use std::sync::Arc;

use namidb_core::id::{NamespaceId, NodeId};
use namidb_core::value::Value as CoreValue;
use namidb_storage::{NamespacePaths, NodeWriteRecord, WriterSession};
use object_store::memory::InMemory;
use object_store::ObjectStore;

use namidb_query::{
    execute, lower, optimize, parse, LogicalPlan, Params, RuntimeValue, StatsCatalog,
};

fn person(team: &str, seq: i64) -> NodeWriteRecord {
    let mut props: BTreeMap<String, CoreValue> = BTreeMap::new();
    props.insert("team".into(), CoreValue::Str(team.into()));
    props.insert("seq".into(), CoreValue::I64(seq));
    NodeWriteRecord {
        properties: props,
        schema_version: 1,
        ..Default::default()
    }
}

fn plan_has_multi_lookup(plan: &LogicalPlan) -> bool {
    matches!(plan, LogicalPlan::NodeByPropertyValue { multi: true, .. })
        || plan.children().into_iter().any(plan_has_multi_lookup)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn batched_multi_arm_matches_expected_fanout() {
    let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
    let paths = NamespacePaths::new("tenants", NamespaceId::new("bml").unwrap());
    let mut writer = WriterSession::open(store, paths).await.unwrap();
    // teams: alpha x3, beta x2, gamma x1.
    for (i, team) in ["alpha", "alpha", "alpha", "beta", "beta", "gamma"]
        .iter()
        .enumerate()
    {
        writer
            .upsert_node("Person", NodeId::new(), &person(team, i as i64))
            .unwrap();
    }
    writer.commit_batch().await.unwrap();
    writer
        .create_property_index_named(None, "Person", "team", false)
        .await
        .unwrap();
    let schema = writer.snapshot().manifest().manifest.schema.clone();
    writer.flush(schema).await.unwrap();
    // Memtable delta: one more beta after the flush.
    writer
        .upsert_node("Person", NodeId::new(), &person("beta", 6))
        .unwrap();
    writer.commit_batch().await.unwrap();

    let snapshot = writer.snapshot();
    let catalog = StatsCatalog::from_manifest(&snapshot.manifest().manifest);
    let query = "UNWIND $teams AS t MATCH (p:Person) WHERE p.team = t \
                 RETURN t AS team, p.seq AS seq";
    let plan = optimize(lower(&parse(query).unwrap()).unwrap(), &catalog);
    assert!(
        plan_has_multi_lookup(&plan),
        "the indexed non-unique equality must plan the multi lookup: {plan:?}"
    );

    // Duplicates kept per input row, misses produce no rows, and the
    // memtable delta is unioned in.
    let mut params = Params::new();
    params.insert(
        "teams".to_string(),
        RuntimeValue::List(vec![
            RuntimeValue::String("alpha".into()),
            RuntimeValue::String("beta".into()),
            RuntimeValue::String("alpha".into()),
            RuntimeValue::String("missing".into()),
        ]),
    );
    let rows = execute(&plan, &snapshot, &params).await.unwrap();
    let count_for = |team: &str| {
        rows.iter()
            .filter(|r| matches!(r.get("team"), Some(RuntimeValue::String(t)) if t == team))
            .count()
    };
    assert_eq!(count_for("alpha"), 6, "3 matches x 2 duplicate inputs");
    assert_eq!(count_for("beta"), 3, "2 flushed + 1 memtable");
    assert_eq!(count_for("missing"), 0);
    assert_eq!(rows.len(), 9);

    // Mixed-type list: the non-String entries take the per-row exact route
    // and alignment with the batched String groups must hold.
    let mut params = Params::new();
    params.insert(
        "teams".to_string(),
        RuntimeValue::List(vec![
            RuntimeValue::String("gamma".into()),
            RuntimeValue::Integer(42),
            RuntimeValue::String("beta".into()),
        ]),
    );
    let rows = execute(&plan, &snapshot, &params).await.unwrap();
    let seqs_for = |team: &str| {
        let mut seqs: Vec<i64> = rows
            .iter()
            .filter(|r| matches!(r.get("team"), Some(RuntimeValue::String(t)) if t == team))
            .map(|r| match r.get("seq") {
                Some(RuntimeValue::Integer(s)) => *s,
                other => panic!("seq must be an integer: {other:?}"),
            })
            .collect();
        seqs.sort_unstable();
        seqs
    };
    assert_eq!(seqs_for("gamma"), vec![5]);
    assert_eq!(seqs_for("beta"), vec![3, 4, 6]);
    assert_eq!(rows.len(), 4, "the integer entry matches nothing");
}
