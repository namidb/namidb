//! Shared fixture: small `Doc` nodes with a unique `key`, and a correlated
//! write lookup over them, driven by rows that each carry ~4 KiB, that the
//! generic write executor runs (the bounded terminal-SET path does not take
//! a CREATE). Storage keeps little for the lookup itself; the rows it
//! produces are what is large.

use std::collections::BTreeMap;
use std::sync::Arc;

use namidb_core::id::{NamespaceId, NodeId};
use namidb_core::schema::{DataType, LabelDef, PropertyDef, SchemaBuilder};
use namidb_core::value::Value as CoreValue;
use namidb_query::cost::StatsCatalog;
use namidb_query::{
    execute_write, lower, optimize, parse, with_query_memory, ExecError, LogicalPlan, Params,
    QueryMemoryBudget, RuntimeValue, WriteOutcome,
};
use namidb_storage::{NamespacePaths, NodeWriteRecord, WriterSession};
use object_store::memory::InMemory;

pub async fn seed_docs(name: &str, docs: usize) -> WriterSession {
    let paths = NamespacePaths::new("tenants", NamespaceId::new(name).unwrap());
    let mut writer = WriterSession::open(Arc::new(InMemory::new()), paths)
        .await
        .unwrap();
    let schema = SchemaBuilder::new()
        .label(LabelDef {
            name: "Doc".into(),
            properties: vec![PropertyDef::new("key", DataType::Utf8, false)
                .unwrap()
                .with_unique(true)],
        })
        .unwrap()
        .build();
    for i in 0..docs {
        writer
            .upsert_node(
                "Doc",
                NodeId::new(),
                &NodeWriteRecord {
                    properties: BTreeMap::from([(
                        "key".into(),
                        CoreValue::Str(format!("doc-{i}")),
                    )]),
                    schema_version: 1,
                    ..Default::default()
                },
            )
            .unwrap();
    }
    writer.commit_batch().await.unwrap();
    writer.flush(schema).await.unwrap();
    writer
}

pub fn lookup_plan(writer: &WriterSession) -> LogicalPlan {
    let query =
        parse("UNWIND $rows AS r MATCH (d:Doc {key: r.k}) CREATE (d)-[:R]->(:Org {n: 1})").unwrap();
    let snapshot = writer.snapshot();
    let catalog = StatsCatalog::from_manifest(&snapshot.manifest().manifest);
    drop(snapshot);
    let plan = optimize(lower(&query).unwrap(), &catalog);
    fn has_lookup(plan: &LogicalPlan) -> bool {
        matches!(plan, LogicalPlan::NodeByPropertyValue { .. })
            || plan.children().into_iter().any(has_lookup)
    }
    assert!(has_lookup(&plan), "{plan:?}");
    plan
}

/// Run the lookup over the first `keys` docs (one ~4 KiB row each), then
/// drop what it staged.
pub async fn run(
    writer: &mut WriterSession,
    plan: &LogicalPlan,
    keys: usize,
    budget: Option<Arc<QueryMemoryBudget>>,
) -> Result<WriteOutcome, ExecError> {
    let mut params = Params::new();
    let pad = "x".repeat(4096);
    params.insert(
        "rows".into(),
        RuntimeValue::List(
            (0..keys)
                .map(|i| {
                    RuntimeValue::Map(BTreeMap::from([
                        ("k".to_string(), RuntimeValue::String(format!("doc-{i}"))),
                        ("pad".to_string(), RuntimeValue::String(pad.clone())),
                    ]))
                })
                .collect(),
        ),
    );
    let outcome = with_query_memory(budget, execute_write(plan, writer, &params)).await;
    writer.discard_batch();
    outcome
}
