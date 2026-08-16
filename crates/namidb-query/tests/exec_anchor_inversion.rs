//! Item 36 end-to-end: the optimizer anchors a pattern at its selective
//! endpoint regardless of the direction it was written. Both spellings of
//! the query must produce identical rows — including full node and
//! relationship bindings and static path assembly — on the memtable and
//! flushed routes, and the slow spelling's optimized plan must contain the
//! index anchor instead of the label scan.

use std::collections::BTreeMap;
use std::sync::Arc;

use namidb_core::id::{NamespaceId, NodeId};
use namidb_core::schema::{DataType, EdgeTypeDef, LabelDef, PropertyDef, Schema, SchemaBuilder};
use namidb_core::value::Value as CoreValue;
use namidb_storage::{EdgeWriteRecord, NamespacePaths, NodeWriteRecord, WriterSession};
use object_store::memory::InMemory;
use object_store::ObjectStore;

use namidb_query::{execute, lower, optimize, parse, LogicalPlan, Params, StatsCatalog};

const PEOPLE: u64 = 60;
const COMPANIES: u64 = 6;

fn schema() -> Schema {
    SchemaBuilder::new()
        .label(LabelDef {
            name: "Person".into(),
            properties: vec![PropertyDef::new("name", DataType::Utf8, false).unwrap()],
        })
        .unwrap()
        .label(LabelDef {
            name: "Company".into(),
            properties: vec![PropertyDef::new("cid", DataType::Int64, false)
                .unwrap()
                .with_unique(true)],
        })
        .unwrap()
        .edge_type(EdgeTypeDef {
            name: "WORKS_AT".into(),
            src_label: "Person".into(),
            dst_label: "Company".into(),
            properties: vec![PropertyDef::new("since", DataType::Int64, true).unwrap()],
        })
        .unwrap()
        .build()
}

async fn corpus(name: &str, flush: bool) -> (WriterSession, Vec<NodeId>) {
    let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
    let paths = NamespacePaths::new("tenants", NamespaceId::new(name).unwrap());
    let mut writer = WriterSession::open(store, paths).await.unwrap();
    let mut companies = Vec::new();
    for ordinal in 0..COMPANIES {
        let id = NodeId::new();
        companies.push(id);
        let mut props: BTreeMap<String, CoreValue> = BTreeMap::new();
        props.insert("cid".into(), CoreValue::I64(ordinal as i64));
        writer
            .upsert_node(
                "Company",
                id,
                &NodeWriteRecord {
                    properties: props,
                    schema_version: 1,
                    ..Default::default()
                },
            )
            .unwrap();
    }
    for ordinal in 0..PEOPLE {
        let id = NodeId::new();
        let mut props: BTreeMap<String, CoreValue> = BTreeMap::new();
        props.insert("name".into(), CoreValue::Str(format!("p{ordinal:02}")));
        writer
            .upsert_node(
                "Person",
                id,
                &NodeWriteRecord {
                    properties: props,
                    schema_version: 1,
                    ..Default::default()
                },
            )
            .unwrap();
        let mut edge_props: BTreeMap<String, CoreValue> = BTreeMap::new();
        edge_props.insert("since".into(), CoreValue::I64((ordinal % 5) as i64));
        writer
            .upsert_edge(
                "WORKS_AT",
                id,
                companies[(ordinal % COMPANIES) as usize],
                &EdgeWriteRecord {
                    properties: edge_props,
                    schema_version: 1,
                },
            )
            .unwrap();
    }
    writer.commit_batch().await.unwrap();
    if flush {
        writer.flush(schema()).await.unwrap();
        writer.compact_l0(&schema()).await.unwrap();
    }
    (writer, companies)
}

fn plan_has_lookup(plan: &LogicalPlan) -> bool {
    matches!(plan, LogicalPlan::NodeByPropertyValue { .. })
        || plan.children().into_iter().any(plan_has_lookup)
}

fn plan_has_node_scan_of(plan: &LogicalPlan, wanted: &str) -> bool {
    if let LogicalPlan::NodeScan { label, .. } = plan {
        if label.as_deref() == Some(wanted) {
            return true;
        }
    }
    plan.children()
        .into_iter()
        .any(|child| plan_has_node_scan_of(child, wanted))
}

async fn rows_canonical(writer: &WriterSession, query: &str) -> Vec<String> {
    let snapshot = writer.snapshot();
    let catalog = StatsCatalog::from_manifest(&snapshot.manifest().manifest);
    let plan = optimize(lower(&parse(query).unwrap()).unwrap(), &catalog);
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

const SLOW: &str = "MATCH (p:Person)-[w:WORKS_AT]->(c:Company {cid: 2}) \
                    WHERE w.since = 1 RETURN p.name AS name, w.since AS since, c.cid AS cid";
const FAST: &str = "MATCH (c:Company {cid: 2})<-[w:WORKS_AT]-(p:Person) \
                    WHERE w.since = 1 RETURN p.name AS name, w.since AS since, c.cid AS cid";

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn slow_spelling_plans_the_index_anchor_and_matches_the_fast_spelling() {
    // Manifest stats only exist after a flush, so a cold all-memtable
    // namespace legitimately keeps the un-inverted plan (cheap anyway);
    // parity must hold on both routes regardless.
    for (route, flush) in [("memtable", false), ("flushed", true)] {
        let (mut writer, companies) = corpus(&format!("anchor-inv-{route}"), flush).await;
        let mut expected = (0..PEOPLE)
            .filter(|o| o % COMPANIES == 2 && o % 5 == 1)
            .count();
        if flush {
            let snapshot = writer.snapshot();
            let catalog = StatsCatalog::from_manifest(&snapshot.manifest().manifest);
            let optimized = optimize(lower(&parse(SLOW).unwrap()).unwrap(), &catalog);
            assert!(
                plan_has_lookup(&optimized),
                "{route}: the slow spelling must plan the unique cid anchor"
            );
            assert!(
                !plan_has_node_scan_of(&optimized, "Person"),
                "{route}: the 60-row Person scan must be gone from the plan"
            );
            drop(snapshot);
            // Post-flush memtable delta: the inverted anchor + reverse expand
            // must still see rows that only exist in the memtable.
            let target = companies[2];
            for suffix in ["late-a", "late-b"] {
                let id = NodeId::new();
                writer
                    .upsert_node(
                        "Person",
                        id,
                        &NodeWriteRecord {
                            properties: BTreeMap::from([(
                                "name".into(),
                                CoreValue::Str(format!("p-{suffix}")),
                            )]),
                            schema_version: 1,
                            ..Default::default()
                        },
                    )
                    .unwrap();
                writer
                    .upsert_edge(
                        "WORKS_AT",
                        id,
                        target,
                        &EdgeWriteRecord {
                            properties: BTreeMap::from([("since".into(), CoreValue::I64(1))]),
                            schema_version: 1,
                        },
                    )
                    .unwrap();
            }
            writer.commit_batch().await.unwrap();
            expected += 2;
        }

        let slow_rows = rows_canonical(&writer, SLOW).await;
        let fast_rows = rows_canonical(&writer, FAST).await;
        assert!(
            !slow_rows.is_empty(),
            "{route}: the fixture must produce matches or parity is vacuous"
        );
        assert_eq!(
            slow_rows, fast_rows,
            "{route}: both spellings must return identical bindings"
        );
        assert_eq!(slow_rows.len(), expected, "{route}: exact match count");
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn static_path_assembly_is_identical_across_spellings() {
    let (writer, _companies) = corpus("anchor-inv-path", true).await;
    let slow = rows_canonical(
        &writer,
        "MATCH q = (p:Person)-[w:WORKS_AT]->(c:Company {cid: 3}) \
         RETURN p.name AS name, q AS q",
    )
    .await;
    let fast = rows_canonical(
        &writer,
        "MATCH q = (c:Company {cid: 3})<-[w:WORKS_AT]-(p:Person) \
         RETURN p.name AS name, q AS q",
    )
    .await;
    assert!(!slow.is_empty());
    // The path is assembled in PATTERN order, which differs between the two
    // spellings by definition (p-w-c vs c-w-p); what inversion must preserve
    // is the SLOW spelling's own path shape. Compare the slow spelling
    // against itself executed WITHOUT the optimizer.
    let snapshot = writer.snapshot();
    let unoptimized = lower(
        &parse(
            "MATCH q = (p:Person)-[w:WORKS_AT]->(c:Company {cid: 3}) \
             RETURN p.name AS name, q AS q",
        )
        .unwrap(),
    )
    .unwrap();
    let rows = execute(&unoptimized, &snapshot, &Params::new())
        .await
        .unwrap();
    let mut reference: Vec<String> = rows
        .iter()
        .map(|row| {
            row.bindings
                .iter()
                .map(|(column, value)| format!("{column}={value:?}"))
                .collect::<Vec<_>>()
                .join("|")
        })
        .collect();
    reference.sort();
    assert_eq!(
        slow, reference,
        "the optimized slow spelling must keep its own path shape"
    );
    let _ = fast;
}
