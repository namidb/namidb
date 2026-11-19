//! A path count's working state (oriented edge lists, per-node weight maps)
//! is charged to the query's memory budget, so a whole-graph grouped count
//! over a graph larger than the budget fails with a memory error instead of
//! growing past it.

use std::collections::BTreeMap;
use std::sync::Arc;

use namidb_core::id::{NamespaceId, NodeId};
use namidb_core::schema::{DataType, LabelDef, PropertyDef, SchemaBuilder};
use namidb_core::value::Value as CoreValue;
use namidb_storage::{EdgeWriteRecord, NamespacePaths, NodeWriteRecord, WriterSession};
use object_store::memory::InMemory;

use namidb_query::{
    execute_with_limits, lower, optimize, parse, with_query_memory, ExecError, LogicalPlan, Params,
    QueryMemoryBudget, RuntimeValue, StatsCatalog,
};

const MIB: usize = 1024 * 1024;
const N: usize = 40_000;

async fn ring() -> WriterSession {
    let paths = NamespacePaths::new("tenants", NamespaceId::new("pc-mem").unwrap());
    let mut w = WriterSession::open(Arc::new(InMemory::new()), paths)
        .await
        .unwrap();
    let mut ids = Vec::with_capacity(N);
    for i in 0..N {
        let id = NodeId::new();
        let mut props: BTreeMap<String, CoreValue> = BTreeMap::new();
        props.insert("k".into(), CoreValue::I64(i as i64));
        let rec = NodeWriteRecord {
            properties: props,
            schema_version: 0,
            labels: vec![],
        };
        w.upsert_node_with_labels(["A".to_string()], id, &rec)
            .unwrap();
        ids.push(id);
    }
    for i in 0..N {
        w.upsert_edge(
            "R",
            ids[i],
            ids[(i * 7 + 1) % N],
            &EdgeWriteRecord::default(),
        )
        .unwrap();
    }
    w.commit_batch().await.unwrap();
    let schema = SchemaBuilder::new()
        .label(LabelDef {
            name: "A".into(),
            properties: vec![PropertyDef::new("k", DataType::Int64, true).unwrap()],
        })
        .unwrap()
        .build();
    w.flush(schema).await.unwrap();
    w
}

fn uses_path_count(plan: &LogicalPlan) -> bool {
    matches!(plan, LogicalPlan::PathCount { .. })
        || plan.children().into_iter().any(uses_path_count)
}

async fn run(w: &WriterSession, budget: usize) -> Result<i64, ExecError> {
    let snap = w.snapshot();
    let q = "MATCH (a:A)-[:R]-(b) RETURN count(*) AS c";
    let plan = optimize(lower(&parse(q).unwrap()).unwrap(), &StatsCatalog::empty());
    assert!(uses_path_count(&plan), "{}", namidb_query::explain(&plan));
    let budget = Arc::new(QueryMemoryBudget::new(budget));
    let rows = with_query_memory(
        Some(budget),
        execute_with_limits(&plan, &snap, &Params::new(), None, None),
    )
    .await?;
    match rows[0].get("c") {
        Some(RuntimeValue::Integer(c)) => Ok(*c),
        other => panic!("count: {other:?}"),
    }
}

#[tokio::test]
async fn path_count_state_is_charged_to_the_budget() {
    let w = ring().await;
    // `i -> 7i+1 mod N` is a permutation without fixed points, so every
    // node has one edge out and one in: 2N undirected paths.
    for proof in ["never", "always"] {
        std::env::set_var("NAMIDB_PATH_COUNT_SCAN_PROOF", proof);
        assert_eq!(
            run(&w, 64 * MIB).await.expect("ample budget answers"),
            2 * N as i64,
            "proof {proof}"
        );
        // The edge list is 40k pairs (1.3 MB) and fits; the fold doubles it
        // for the undirected hop and keeps a path-count map per side, which
        // a 3 MiB budget cannot hold. Unmetered, the probe proof answered
        // under 2 MiB.
        match run(&w, 3 * MIB).await {
            Err(ExecError::MemoryLimit(_)) => {}
            other => panic!("proof {proof}: expected a memory error under 3 MiB, got {other:?}"),
        }
    }
    std::env::remove_var("NAMIDB_PATH_COUNT_SCAN_PROOF");
}
