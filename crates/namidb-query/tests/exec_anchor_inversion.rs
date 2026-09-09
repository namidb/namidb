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

/// Fifth field report: `(o:OFERTA|PROMOCION)-[:VIGENTE_EN]->(f:FECHA
/// {fecha: X})` used to re-scan the whole namespace per date because the
/// label-disjunction source hid its scan behind an OR filter the anchor
/// inversion refused. The full pipeline must now plan the f-anchor, keep
/// the disjunction as a residual filter, and match the un-inverted
/// results exactly.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn disjunction_source_anchors_at_the_dated_target() {
    let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
    let paths = NamespacePaths::new("tenants", NamespaceId::new("anchor-disj").unwrap());
    let mut writer = WriterSession::open(store, paths).await.unwrap();
    let mut fechas = Vec::new();
    for d in 0..30 {
        let id = NodeId::new();
        fechas.push(id);
        let mut props: BTreeMap<String, CoreValue> = BTreeMap::new();
        props.insert("fecha".into(), CoreValue::Str(format!("d{d}")));
        writer
            .upsert_node(
                "FECHA",
                id,
                &NodeWriteRecord {
                    properties: props,
                    schema_version: 1,
                    ..Default::default()
                },
            )
            .unwrap();
    }
    let edge = EdgeWriteRecord {
        properties: BTreeMap::new(),
        schema_version: 1,
    };
    // 40 ofertas + 20 promociones, each valid on (ordinal % 30).
    for ordinal in 0..60 {
        let label = if ordinal < 40 { "OFERTA" } else { "PROMOCION" };
        let id = NodeId::new();
        writer
            .upsert_node(
                label,
                id,
                &NodeWriteRecord {
                    properties: BTreeMap::new(),
                    schema_version: 1,
                    ..Default::default()
                },
            )
            .unwrap();
        writer
            .upsert_edge("VIGENTE_EN", id, fechas[ordinal % 30], &edge)
            .unwrap();
    }
    writer.commit_batch().await.unwrap();
    writer
        .create_unique_constraint("FECHA", "fecha")
        .await
        .unwrap();
    let schema = writer.snapshot().manifest().manifest.schema.clone();
    writer.flush(schema).await.unwrap();

    let snapshot = writer.snapshot();
    let catalog = StatsCatalog::from_manifest(&snapshot.manifest().manifest);
    let query = "MATCH (o:OFERTA|PROMOCION)-[:VIGENTE_EN]->(f:FECHA {fecha: 'd7'}) \
                 RETURN count(o) AS n";
    let plan = optimize(lower(&parse(query).unwrap()).unwrap(), &catalog);
    fn has_f_anchor(plan: &LogicalPlan) -> bool {
        matches!(plan, LogicalPlan::NodeByPropertyValue { alias, .. } if alias == "f")
            || plan.children().into_iter().any(has_f_anchor)
    }
    fn has_unlabeled_scan(plan: &LogicalPlan) -> bool {
        matches!(plan, LogicalPlan::NodeScan { label: None, .. })
            || plan.children().into_iter().any(has_unlabeled_scan)
    }
    assert!(
        has_f_anchor(&plan),
        "the dated target must anchor the disjunction pattern: {plan:?}"
    );
    assert!(
        !has_unlabeled_scan(&plan),
        "the whole-namespace scan must be gone: {plan:?}"
    );

    let rows = execute(&plan, &snapshot, &Params::new()).await.unwrap();
    // ordinal % 30 == 7 -> ordinals 7 and 37: both OFERTA. Plus none from
    // PROMOCION (40..59 -> %30 in 10..29): 37 is OFERTA, 47 -> d17. So d7
    // matches ordinals 7 and 37 = 2.
    assert!(
        matches!(
            rows[0].get("n"),
            Some(namidb_query::RuntimeValue::Integer(2))
        ),
        "{rows:?}"
    );
}

/// Fifth/sixth field report: `WITH v, p.cod_item AS c WHERE c = '…'` is the
/// same query as the in-pattern spelling, but the alias hid the equality
/// above a Project, so the anchor was never visible and the plan scanned
/// the whole source label (the reporter's 160k-row scan that killed the
/// process, while the in-pattern form finished instantly). The predicate
/// must now substitute back through the projection, anchor, and return
/// identical rows.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn with_alias_predicate_still_anchors_at_the_indexed_target() {
    let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
    let paths = NamespacePaths::new("tenants", NamespaceId::new("alias-anchor").unwrap());
    let mut writer = WriterSession::open(store, paths).await.unwrap();

    // 20 products; 200 sales spread over them and over 5 dates.
    let mut productos = Vec::new();
    for i in 0..20 {
        let id = NodeId::new();
        productos.push(id);
        let mut props: BTreeMap<String, CoreValue> = BTreeMap::new();
        props.insert("cod_item".into(), CoreValue::Str(format!("cod-{i}")));
        writer
            .upsert_node(
                "PRODUCTO",
                id,
                &NodeWriteRecord {
                    properties: props,
                    schema_version: 1,
                    ..Default::default()
                },
            )
            .unwrap();
    }
    let mut fechas = Vec::new();
    for d in 0..5 {
        let id = NodeId::new();
        fechas.push(id);
        let mut props: BTreeMap<String, CoreValue> = BTreeMap::new();
        props.insert("fecha".into(), CoreValue::Str(format!("f{d}")));
        writer
            .upsert_node(
                "FECHA",
                id,
                &NodeWriteRecord {
                    properties: props,
                    schema_version: 1,
                    ..Default::default()
                },
            )
            .unwrap();
    }
    let edge = EdgeWriteRecord {
        properties: BTreeMap::new(),
        schema_version: 1,
    };
    for i in 0..200usize {
        let v = NodeId::new();
        let mut props: BTreeMap<String, CoreValue> = BTreeMap::new();
        props.insert("venta_neta".into(), CoreValue::I64((i % 7) as i64 + 1));
        writer
            .upsert_node(
                "VENTA",
                v,
                &NodeWriteRecord {
                    properties: props,
                    schema_version: 1,
                    ..Default::default()
                },
            )
            .unwrap();
        writer
            .upsert_edge("VENTA_DE_PRODUCTO", v, productos[i % 20], &edge)
            .unwrap();
        writer
            .upsert_edge("VENTA_EN_FECHA", v, fechas[i % 5], &edge)
            .unwrap();
    }
    writer.commit_batch().await.unwrap();
    writer
        .create_unique_constraint("PRODUCTO", "cod_item")
        .await
        .unwrap();
    let schema = writer.snapshot().manifest().manifest.schema.clone();
    writer.flush(schema).await.unwrap();

    let snapshot = writer.snapshot();
    let catalog = StatsCatalog::from_manifest(&snapshot.manifest().manifest);
    let with_alias = "MATCH (v:VENTA)-[:VENTA_DE_PRODUCTO]->(p:PRODUCTO) \
                      WITH v, p.cod_item AS c WHERE c = 'cod-3' \
                      MATCH (v)-[:VENTA_EN_FECHA]->(f:FECHA) \
                      RETURN f.fecha AS fecha, sum(v.venta_neta) AS total \
                      ORDER BY fecha";
    let in_pattern = "MATCH (v:VENTA)-[:VENTA_DE_PRODUCTO]->(p:PRODUCTO {cod_item: 'cod-3'}) \
                      MATCH (v)-[:VENTA_EN_FECHA]->(f:FECHA) \
                      RETURN f.fecha AS fecha, sum(v.venta_neta) AS total \
                      ORDER BY fecha";

    fn anchors_at_producto(plan: &LogicalPlan) -> bool {
        matches!(
            plan,
            LogicalPlan::NodeByPropertyValue { label, .. } if label == "PRODUCTO"
        ) || plan.children().into_iter().any(anchors_at_producto)
    }
    fn scans_ventas(plan: &LogicalPlan) -> bool {
        matches!(
            plan,
            LogicalPlan::NodeScan { label: Some(l), .. } if l == "VENTA"
        ) || plan.children().into_iter().any(scans_ventas)
    }

    let aliased_plan = optimize(lower(&parse(with_alias).unwrap()).unwrap(), &catalog);
    assert!(
        anchors_at_producto(&aliased_plan),
        "the WITH-aliased predicate must anchor at PRODUCTO: {aliased_plan:?}"
    );
    assert!(
        !scans_ventas(&aliased_plan),
        "the whole-VENTA scan must be gone: {aliased_plan:?}"
    );

    // Same rows as the in-pattern spelling, which already anchored.
    let rows_aliased = execute(&aliased_plan, &snapshot, &Params::new())
        .await
        .unwrap();
    let pattern_plan = optimize(lower(&parse(in_pattern).unwrap()).unwrap(), &catalog);
    let rows_pattern = execute(&pattern_plan, &snapshot, &Params::new())
        .await
        .unwrap();
    let fmt = |rows: &[namidb_query::Row]| {
        rows.iter()
            .map(|r| format!("{:?}|{:?}", r.get("fecha"), r.get("total")))
            .collect::<Vec<_>>()
    };
    assert_eq!(fmt(&rows_aliased), fmt(&rows_pattern));
    assert!(!rows_aliased.is_empty());
}
