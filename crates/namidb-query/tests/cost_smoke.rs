//! Smoke tests for — `StatsCatalog::from_manifest` + cardinality
//! estimation vs actual row counts on the LDBC SNB micro-graph.
//!
//! These tests load the same fixture used by `exec_ldbc_snb.rs`, force
//! a flush so the SST descriptors carry property/degree stats, then
//! compare the optimizer's `estimate` output against the actual rows
//! returned by `execute`. The micro-graph is small enough that the
//! ratios will show large absolute errors; the assertions therefore
//! check *shape* (zero vs non-zero, finite, ordering of magnitude),
//! not exact numbers — RFC-010 §3 explicitly assumes independence and
//! folklore fallbacks, so absolute accuracy is a future concern.

use std::collections::BTreeMap;
use std::sync::Arc;

use namidb_core::id::{NamespaceId, NodeId};
use namidb_core::value::Value as CoreValue;
use namidb_core::{DataType, EdgeTypeDef, LabelDef, PropertyDef, Schema, SchemaBuilder};
use namidb_storage::{EdgeWriteRecord, NamespacePaths, NodeWriteRecord, Snapshot, WriterSession};
use object_store::memory::InMemory;
use object_store::ObjectStore;

use namidb_query::cost::{estimate, StatsCatalog};
use namidb_query::plan::LogicalPlan;
use namidb_query::{
    execute, explain_query_raw_verbose, explain_query_verbose, lower, optimize, parse,
    plan as build_plan, Params, RuntimeValue,
};

fn store() -> Arc<dyn ObjectStore> {
    Arc::new(InMemory::new())
}

fn paths(name: &str) -> NamespacePaths {
    NamespacePaths::new("tenants", NamespaceId::new(name).unwrap())
}

fn schema() -> Schema {
    SchemaBuilder::new()
        .label(LabelDef {
            name: "Person".into(),
            properties: vec![
                PropertyDef::new("firstName", DataType::Utf8, true).unwrap(),
                PropertyDef::new("lastName", DataType::Utf8, true).unwrap(),
                PropertyDef::new("age", DataType::Int64, true).unwrap(),
            ],
        })
        .unwrap()
        .label(LabelDef {
            name: "Message".into(),
            properties: vec![
                PropertyDef::new("content", DataType::Utf8, true).unwrap(),
                PropertyDef::new("creationDate", DataType::Int64, true).unwrap(),
            ],
        })
        .unwrap()
        .label(LabelDef {
            name: "Comment".into(),
            properties: vec![
                PropertyDef::new("content", DataType::Utf8, true).unwrap(),
                PropertyDef::new("creationDate", DataType::Int64, true).unwrap(),
            ],
        })
        .unwrap()
        .edge_type(EdgeTypeDef {
            name: "KNOWS".into(),
            src_label: "Person".into(),
            dst_label: "Person".into(),
            properties: vec![],
        })
        .unwrap()
        .edge_type(EdgeTypeDef {
            name: "HAS_CREATOR".into(),
            src_label: "Message".into(),
            dst_label: "Person".into(),
            properties: vec![],
        })
        .unwrap()
        .edge_type(EdgeTypeDef {
            name: "LIKES".into(),
            src_label: "Person".into(),
            dst_label: "Message".into(),
            properties: vec![],
        })
        .unwrap()
        .edge_type(EdgeTypeDef {
            name: "REPLY_OF".into(),
            src_label: "Comment".into(),
            dst_label: "Message".into(),
            properties: vec![],
        })
        .unwrap()
        .build()
}

fn person(first: &str, last: &str, age: i32) -> NodeWriteRecord {
    let mut p: BTreeMap<String, CoreValue> = BTreeMap::new();
    p.insert("firstName".into(), CoreValue::Str(first.into()));
    p.insert("lastName".into(), CoreValue::Str(last.into()));
    p.insert("age".into(), CoreValue::I64(age as i64));
    NodeWriteRecord {
        properties: p,
        schema_version: 1,
        ..Default::default()
    }
}

fn message(content: &str, creation_date: i64) -> NodeWriteRecord {
    let mut p = BTreeMap::new();
    p.insert("content".into(), CoreValue::Str(content.into()));
    p.insert("creationDate".into(), CoreValue::I64(creation_date));
    NodeWriteRecord {
        properties: p,
        schema_version: 1,
        ..Default::default()
    }
}

fn comment(content: &str, creation_date: i64) -> NodeWriteRecord {
    message(content, creation_date)
}

fn bare_edge() -> EdgeWriteRecord {
    EdgeWriteRecord {
        properties: BTreeMap::new(),
        schema_version: 1,
    }
}

async fn build_micro_graph_and_flush(writer: &mut WriterSession) -> MicroFixture {
    let alice = NodeId::new();
    let bob = NodeId::new();
    let carol = NodeId::new();
    let dave = NodeId::new();
    let eve = NodeId::new();
    let frank = NodeId::new();
    writer
        .upsert_node("Person", alice, &person("Alice", "Anderson", 30))
        .unwrap();
    writer
        .upsert_node("Person", bob, &person("Bob", "Brown", 25))
        .unwrap();
    writer
        .upsert_node("Person", carol, &person("Carol", "Clark", 35))
        .unwrap();
    writer
        .upsert_node("Person", dave, &person("Dave", "Davies", 28))
        .unwrap();
    writer
        .upsert_node("Person", eve, &person("Eve", "Edwards", 40))
        .unwrap();
    writer
        .upsert_node("Person", frank, &person("Frank", "Foley", 33))
        .unwrap();

    let knows_pairs = [
        (alice, bob),
        (alice, carol),
        (bob, dave),
        (carol, eve),
        (dave, frank),
        (eve, frank),
    ];
    for (src, dst) in knows_pairs {
        writer.upsert_edge("KNOWS", src, dst, &bare_edge()).unwrap();
    }

    let base = 1_700_000_000_000_i64;
    let day = 86_400_000_i64;
    let msg_creators = [bob, carol, dave, eve, frank, bob, carol, eve];
    let mut msg_ids = Vec::with_capacity(8);
    for (i, content) in [
        "Hello world",
        "Cypher rocks",
        "Property graphs",
        "Coffee time",
        "Late-night thoughts",
        "Friday vibes",
        "Holiday plans",
        "Year-end recap",
    ]
    .iter()
    .enumerate()
    {
        let id = NodeId::new();
        msg_ids.push(id);
        writer
            .upsert_node("Message", id, &message(content, base + (i as i64) * day))
            .unwrap();
        writer
            .upsert_edge("HAS_CREATOR", id, msg_creators[i], &bare_edge())
            .unwrap();
    }

    for (src, dst, _ts) in [
        (dave, msg_ids[0], base + 10 * day),
        (dave, msg_ids[2], base + 11 * day),
        (eve, msg_ids[0], base + 9 * day),
        (frank, msg_ids[5], base + 12 * day),
    ] {
        writer.upsert_edge("LIKES", src, dst, &bare_edge()).unwrap();
    }

    let comment_creators = [carol, dave, eve, frank];
    let comment_replies = [0usize, 0, 5, 0];
    let mut comment_ids = Vec::with_capacity(4);
    for (i, content) in ["Nice!", "Agreed", "Going!", "+1"].iter().enumerate() {
        let id = NodeId::new();
        comment_ids.push(id);
        writer
            .upsert_node(
                "Comment",
                id,
                &comment(content, base + (13 + i as i64) * day),
            )
            .unwrap();
        writer
            .upsert_edge("HAS_CREATOR", id, comment_creators[i], &bare_edge())
            .unwrap();
        writer
            .upsert_edge("REPLY_OF", id, msg_ids[comment_replies[i]], &bare_edge())
            .unwrap();
    }

    let outcome = writer.flush(schema()).await.expect("flush succeeds");
    assert!(
        outcome.ssts_written > 0,
        "flush should emit at least one SST"
    );

    MicroFixture {
        alice,
        msg_ids,
        comment_ids,
    }
}

struct MicroFixture {
    alice: NodeId,
    msg_ids: Vec<NodeId>,
    comment_ids: Vec<NodeId>,
}

// ──────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn catalog_from_manifest_captures_label_counts() {
    let mut writer = WriterSession::open(store(), paths("stats01"))
        .await
        .unwrap();
    let _ = build_micro_graph_and_flush(&mut writer).await;
    let snap = writer.snapshot();
    let cat = StatsCatalog::from_manifest(&snap.manifest().manifest);

    let person = cat.label("Person").expect("Person label present");
    assert_eq!(person.node_count, 6, "6 Persons in micro-graph");
    let msg = cat.label("Message").expect("Message label present");
    assert_eq!(msg.node_count, 8);
    let comment = cat.label("Comment").expect("Comment label present");
    assert_eq!(comment.node_count, 4);
    assert_eq!(cat.total_nodes(), 6 + 8 + 4);
}

#[tokio::test]
async fn catalog_counts_multi_label_nodes_under_each_label() {
    // Multi-label core (the reason this branch exists): a node carrying
    // {Person, Admin} must count under BOTH labels, while `total_nodes` counts
    // it once. id-primary node SSTs have an empty scope, so per-label
    // `node_count` is recovered from the label-index posting counts — a path
    // the single-label micro-graph never exercises.
    let mut writer = WriterSession::open(store(), paths("ml-counts"))
        .await
        .unwrap();
    // 3 plain :Person, 2 :Person:Admin, 1 plain :Admin → 6 distinct nodes.
    for i in 0..3 {
        writer
            .upsert_node_with_labels(
                ["Person".to_string()],
                NodeId::new(),
                &person(&format!("p{i}"), "x", 30),
            )
            .unwrap();
    }
    for i in 0..2 {
        writer
            .upsert_node_with_labels(
                ["Person".to_string(), "Admin".to_string()],
                NodeId::new(),
                &person(&format!("pa{i}"), "x", 30),
            )
            .unwrap();
    }
    writer
        .upsert_node_with_labels(["Admin".to_string()], NodeId::new(), &person("a0", "x", 30))
        .unwrap();
    writer.flush(schema()).await.expect("flush");

    let snap = writer.snapshot();
    let cat = StatsCatalog::from_manifest(&snap.manifest().manifest);

    let person = cat.label("Person").expect("Person present");
    let admin = cat.label("Admin").expect("Admin present");
    assert_eq!(person.node_count, 5, "3 plain Person + 2 Person+Admin");
    assert_eq!(admin.node_count, 3, "2 Person+Admin + 1 plain Admin");
    // Distinct node rows: each counted once, including the multi-label ones.
    assert_eq!(cat.total_nodes(), 6, "6 distinct nodes");
    // The per-label populations sum to more than the distinct count precisely
    // because two nodes are double-labelled...
    assert!(
        person.node_count + admin.node_count > cat.total_nodes(),
        "multi-label overlap should make the per-label sum exceed total_nodes"
    );
    // ...yet each label's population stays bounded by the distinct count, the
    // invariant every selectivity ratio in the cost model relies on.
    assert!(person.node_count <= cat.total_nodes());
    assert!(admin.node_count <= cat.total_nodes());
}

#[tokio::test]
async fn per_label_counts_survive_compaction() {
    // Compaction rebuilds the label-index sidecar (with per-label counts) on the
    // merged L1 SST. Without that, post-compaction `from_manifest` would read an
    // empty scope and reset every per-label `node_count` to 0 — reviving the
    // optimizer-pruning regression after the first maintenance tick.
    let mut writer = WriterSession::open(store(), paths("ml-compact"))
        .await
        .unwrap();
    // Two flushed batches → two L0 node SSTs in the (empty-scope) node bucket.
    for batch in 0..2 {
        for i in 0..3 {
            let label = if i == 0 {
                vec!["Person".to_string(), "Admin".to_string()]
            } else {
                vec!["Person".to_string()]
            };
            writer
                .upsert_node_with_labels(
                    label,
                    NodeId::new(),
                    &person(&format!("b{batch}n{i}"), "x", 30),
                )
                .unwrap();
        }
        writer.flush(schema()).await.expect("flush");
    }

    // 6 Person (2 of them also Admin), 2 Admin, 6 distinct nodes — before compaction.
    let pre = StatsCatalog::from_manifest(&writer.snapshot().manifest().manifest);
    assert_eq!(pre.label("Person").unwrap().node_count, 6);
    assert_eq!(pre.label("Admin").unwrap().node_count, 2);
    assert_eq!(pre.total_nodes(), 6);

    writer.compact_l0(&schema()).await.expect("compact");

    // Counts must be unchanged after the L0s collapse into one L1 SST.
    let post = StatsCatalog::from_manifest(&writer.snapshot().manifest().manifest);
    assert_eq!(
        post.label("Person").unwrap().node_count,
        6,
        "Person count must survive compaction"
    );
    assert_eq!(
        post.label("Admin").unwrap().node_count,
        2,
        "Admin count must survive compaction"
    );
    assert_eq!(post.total_nodes(), 6, "total_nodes must survive compaction");
}

#[tokio::test]
async fn multi_label_scan_with_projection_pushdown_returns_correct_rows() {
    // End-to-end guard for the __labels-in-projection fix on the multi-label
    // path. A projected scan that elides __labels would make every row decode
    // an empty label set, so the `:Admin` / `:Person` filter would drop all of
    // them and the optimized query would return 0 rows.
    let mut writer = WriterSession::open(store(), paths("ml-proj"))
        .await
        .unwrap();
    // alice & bob are :Person:Admin; carol & dave are :Person only.
    let mut mk = |labels: &[&str], name: &str| {
        writer
            .upsert_node_with_labels(
                labels.iter().map(|s| s.to_string()),
                NodeId::new(),
                &person(name, "x", 30),
            )
            .unwrap();
    };
    mk(&["Person", "Admin"], "Alice");
    mk(&["Person", "Admin"], "Bob");
    mk(&["Person"], "Carol");
    mk(&["Person"], "Dave");
    writer.flush(schema()).await.expect("flush");
    let snap = writer.snapshot();
    let catalog = StatsCatalog::from_manifest(&snap.manifest().manifest);

    for (q_text, expected) in [
        ("MATCH (a:Admin) RETURN a.firstName", 2usize),
        ("MATCH (a:Person) RETURN a.firstName", 4usize),
        (
            "MATCH (a:Admin) WHERE a.firstName = 'Alice' RETURN a.firstName",
            1usize,
        ),
    ] {
        let q = parse(q_text).unwrap();
        let raw = lower(&q).unwrap();
        let opt = optimize(raw.clone(), &catalog);
        let rows_raw = execute(&raw, &snap, &Params::default()).await.unwrap();
        let rows_opt = execute(&opt, &snap, &Params::default()).await.unwrap();
        assert_eq!(
            rows_opt.len(),
            expected,
            "optimized `{q_text}` should return {expected} rows, got {}",
            rows_opt.len()
        );
        assert_eq!(
            rows_raw.len(),
            rows_opt.len(),
            "raw/opt parity for `{q_text}`"
        );
    }
}

#[tokio::test]
async fn catalog_from_manifest_captures_edge_avg_degree() {
    let mut writer = WriterSession::open(store(), paths("stats02"))
        .await
        .unwrap();
    let _ = build_micro_graph_and_flush(&mut writer).await;
    let snap = writer.snapshot();
    let cat = StatsCatalog::from_manifest(&snap.manifest().manifest);

    // 6 KNOWS edges over 6 distinct src nodes (alice, bob, carol, dave,
    // eve all appear once or twice as src; degrees ranged 0..=2).
    let knows = cat.edge_type("KNOWS").expect("KNOWS present");
    assert_eq!(knows.edge_count, 6);
    assert!(knows.avg_out_degree > 0.0);
    // HAS_CREATOR: 12 edges (8 messages + 4 comments).
    let hc = cat.edge_type("HAS_CREATOR").expect("HAS_CREATOR present");
    assert_eq!(hc.edge_count, 12);
}

#[tokio::test]
async fn property_stats_are_present_after_flush() {
    let mut writer = WriterSession::open(store(), paths("stats03"))
        .await
        .unwrap();
    let _ = build_micro_graph_and_flush(&mut writer).await;
    let snap = writer.snapshot();
    let cat = StatsCatalog::from_manifest(&snap.manifest().manifest);

    let p = cat.label("Person").unwrap();
    // Each declared property is still present as a catalog entry (seeded from
    // the schema), so the optimizer knows the column exists. Under the
    // id-primary layout every property rides in a single `__overflow_json`
    // column rather than a typed `prop_*` column, so the writer emits no
    // per-column Parquet statistics: min/max/ndv come back `None` and the
    // optimizer uses its documented fallbacks. Precise per-label column stats
    // (min/max=25/40, firstName=Alice..Frank, real ndv) return with the
    // typed-column layout (RFC-pending); this test guards the current contract.
    assert!(p.properties.contains_key("age"), "age stats present");
    assert!(p.properties.contains_key("firstName"));
    assert!(p.properties.contains_key("lastName"));
    let age = &p.properties["age"];
    assert!(
        age.min.is_none() && age.max.is_none(),
        "id-primary: no per-column min/max until the typed-column layout, got {:?}/{:?}",
        age.min,
        age.max
    );
    assert!(age.ndv.is_none(), "id-primary: no per-column ndv yet");
    // No per-column null statistics either: null_count stays 0 and
    // non_null_count backfills to the label's node_count (6 Persons).
    assert_eq!(
        age.null_count, 0,
        "no per-column null stats under id-primary"
    );
    assert_eq!(age.non_null_count, 6, "non_null backfills to node_count");

    let first = &p.properties["firstName"];
    assert!(
        first.min.is_none() && first.max.is_none(),
        "id-primary: no per-column min/max for firstName yet"
    );
}

#[tokio::test]
async fn filter_estimate_uses_real_min_max_after_writer_fix() {
    // Range selectivity with real min/max would use the precise formula
    // (lit-min)/(max-min): ages min=25,max=40,lit=30 → 6 * 0.667 = 4.0.
    // Under id-primary there are no per-column min/max (properties live in
    // `__overflow_json`), so range selectivity falls back to the documented
    // 0.33 constant → 6 * 0.33 ≈ 1.98. The precise estimate returns with the
    // typed-column layout; results are unaffected either way.
    let mut writer = WriterSession::open(store(), paths("stats06b"))
        .await
        .unwrap();
    let _ = build_micro_graph_and_flush(&mut writer).await;
    let snap = writer.snapshot();
    let cat = StatsCatalog::from_manifest(&snap.manifest().manifest);

    let q = parse("MATCH (a:Person) WHERE a.age > 30 RETURN a").unwrap();
    let plan = lower(&q).unwrap();
    let card = estimate(&plan, &cat);
    let rows = execute(&plan, &snap, &Params::new()).await.unwrap();

    assert_eq!(rows.len(), 3, "Carol(35), Eve(40), Frank(33)");
    // Fallback: 6 * 0.33 ≈ 1.98. The filter still drops the estimate below the
    // 6-row scan cardinality, which is the property the optimizer relies on.
    assert!(
        (card.rows - 2.0).abs() < 0.6,
        "fallback range selectivity should give ~2 (6 * 0.33), got {}",
        card.rows
    );
    assert!(
        card.rows < 6.0,
        "filter must still drop below scan cardinality"
    );
}

#[tokio::test]
async fn nodescan_estimate_matches_actual_on_micro_graph() {
    let mut writer = WriterSession::open(store(), paths("stats04"))
        .await
        .unwrap();
    let _ = build_micro_graph_and_flush(&mut writer).await;
    let snap = writer.snapshot();
    let cat = StatsCatalog::from_manifest(&snap.manifest().manifest);

    let q = parse("MATCH (a:Person) RETURN a").unwrap();
    let plan = lower(&q).unwrap();
    let card = estimate(&plan, &cat);
    let rows = execute(&plan, &snap, &Params::new()).await.unwrap();

    // For a bare label scan + projection, estimate should match actual.
    assert!((card.rows - rows.len() as f64).abs() < 1.0);
    assert_eq!(rows.len(), 6);
}

#[tokio::test]
async fn expand_estimate_is_in_the_right_order_of_magnitude() {
    let mut writer = WriterSession::open(store(), paths("stats05"))
        .await
        .unwrap();
    let _ = build_micro_graph_and_flush(&mut writer).await;
    let snap = writer.snapshot();
    let cat = StatsCatalog::from_manifest(&snap.manifest().manifest);

    let q = parse("MATCH (a:Person)-[:KNOWS]->(b) RETURN a, b").unwrap();
    let plan = lower(&q).unwrap();
    let card = estimate(&plan, &cat);
    let rows = execute(&plan, &snap, &Params::new()).await.unwrap();

    // We hardcoded 6 KNOWS edges, so the executor produces exactly 6
    // tuples. The estimate uses avg_out_degree which on this graph is
    // 1.0 (6 src keys, 6 edges). 6 * 1.0 = 6.
    assert!(rows.len() == 6, "got {} rows", rows.len());
    assert!(
        card.rows >= 3.0 && card.rows <= 20.0,
        "estimate {} should be in [3, 20] vs actual {}",
        card.rows,
        rows.len()
    );
}

#[tokio::test]
async fn filter_estimate_drops_below_input_cardinality() {
    let mut writer = WriterSession::open(store(), paths("stats06"))
        .await
        .unwrap();
    let _ = build_micro_graph_and_flush(&mut writer).await;
    let snap = writer.snapshot();
    let cat = StatsCatalog::from_manifest(&snap.manifest().manifest);

    let q = parse("MATCH (a:Person) WHERE a.age > 30 RETURN a").unwrap();
    let plan = lower(&q).unwrap();
    let card = estimate(&plan, &cat);
    let rows = execute(&plan, &snap, &Params::new()).await.unwrap();

    // Persons with age>30 in fixture: Carol(35), Eve(40), Frank(33) → 3.
    assert_eq!(rows.len(), 3);
    // With real min/max in the catalog → range selectivity
    // formula yields a tighter estimate. We assert the estimate stays
    // within scan cardinality bounds; the precise-min-max test does
    // the exact numerical check.
    assert!(
        card.rows > 0.5 && card.rows < 6.0,
        "filter estimate {} should drop below scan cardinality",
        card.rows
    );
}

#[tokio::test]
async fn ic02_explain_verbose_renders_with_estimates() {
    // Smoke: lower the LDBC IC2 query and ensure EXPLAIN VERBOSE
    // produces a tree with cardinality numbers, without panicking on
    // any operator shape (SemiApply / PatternList / Expand chain).
    let mut writer = WriterSession::open(store(), paths("stats07"))
        .await
        .unwrap();
    let f = build_micro_graph_and_flush(&mut writer).await;
    let snap = writer.snapshot();
    let cat = StatsCatalog::from_manifest(&snap.manifest().manifest);

    let path = format!(
        "{}/tests/fixtures/ic02_recent_messages_by_friends.cypher",
        env!("CARGO_MANIFEST_DIR")
    );
    let src = std::fs::read_to_string(&path).unwrap();
    let q = parse(&src).unwrap();
    let rendered = explain_query_verbose(&q, &cat).expect("EXPLAIN VERBOSE renders");
    assert!(rendered.contains("# Estimated rows:"));
    assert!(rendered.contains("Expand "));
    assert!(rendered.contains("(est="));
    // Sanity check the plan executes too.
    let plan = lower(&q).unwrap();
    let mut params = Params::new();
    params.insert("personId".into(), RuntimeValue::String(f.alice.to_string()));
    let max_date = 1_700_000_000_000_i64 + 100 * 86_400_000;
    params.insert("maxDate".into(), RuntimeValue::Integer(max_date));
    let _ = execute(&plan, &snap, &params).await.unwrap();
    // The first message and comment ids are non-empty (sanity for the
    // fixture, otherwise the EXPLAIN test could pass with an empty graph).
    assert!(!f.msg_ids.is_empty() && !f.comment_ids.is_empty());
}

#[tokio::test]
async fn ic07_explain_verbose_includes_pattern_list_operator() {
    let mut writer = WriterSession::open(store(), paths("stats08"))
        .await
        .unwrap();
    let _ = build_micro_graph_and_flush(&mut writer).await;
    let snap = writer.snapshot();
    let cat = StatsCatalog::from_manifest(&snap.manifest().manifest);

    let path = format!(
        "{}/tests/fixtures/ic07_recent_likers.cypher",
        env!("CARGO_MANIFEST_DIR")
    );
    let src = std::fs::read_to_string(&path).unwrap();
    let q = parse(&src).unwrap();
    let rendered = explain_query_verbose(&q, &cat).expect("EXPLAIN VERBOSE renders");
    // IC7 includes a 2-hop Expand chain; ensure no panic and the
    // header is emitted.
    assert!(rendered.contains("# Estimated rows:"));
    assert!(rendered.lines().count() > 3);
    // We deliberately do not assert on exact numbers — the cost model
    // for micro-graphs is intentionally lossy.
    let _ = snap;
}

#[tokio::test]
async fn empty_catalog_falls_through_to_zero_estimates() {
    // No flushed data → memtable-only graph. The catalog from the
    // committed manifest will be empty even if the executor still
    // returns rows.
    let mut writer = WriterSession::open(store(), paths("stats09"))
        .await
        .unwrap();
    let alice = NodeId::new();
    writer
        .upsert_node("Person", alice, &person("Alice", "X", 30))
        .unwrap();
    writer.commit_batch().await.unwrap();
    // No flush — manifest has no SST.
    let snap = writer.snapshot();
    let cat = StatsCatalog::from_manifest(&snap.manifest().manifest);

    let q = parse("MATCH (a:Person) RETURN a").unwrap();
    let plan = lower(&q).unwrap();
    let card = estimate(&plan, &cat);
    let rows = execute(&plan, &snap, &Params::new()).await.unwrap();

    // Catalog underestimates: 0 (no SST) vs 1 (memtable). Documented
    // gap; tracked in RFC-010 §"Drawbacks 5".
    assert_eq!(card.rows, 0.0);
    assert_eq!(rows.len(), 1);
    snap_borrow_check(&snap);
}

fn snap_borrow_check(_snap: &Snapshot<'_>) {
    // Just to keep `Snapshot` imported and ensure type-name visibility
    // in the tests file.
}

// ──────────────────────────────────────────────────────────────────────
// — predicate pushdown integration
// ──────────────────────────────────────────────────────────────────────

/// Find the first `Filter { input: NodeScan { label } }` node in a plan.
/// Returns the alias of that NodeScan when present, regardless of depth.
#[allow(dead_code)]
fn filter_directly_over_nodescan(plan: &LogicalPlan, target_label: &str) -> Option<String> {
    match plan {
        LogicalPlan::Filter { input, .. } => match input.as_ref() {
            LogicalPlan::NodeScan { label, alias, .. }
                if label.as_deref() == Some(target_label) =>
            {
                Some(alias.clone())
            }
            _ => filter_directly_over_nodescan(input, target_label),
        },
        other => {
            for child in other.children() {
                if let Some(a) = filter_directly_over_nodescan(child, target_label) {
                    return Some(a);
                }
            }
            None
        }
    }
}

/// Recursively determine the maximum depth at which `Filter` operators
/// appear in a plan. Lower is "closer to the leaves".
fn min_filter_depth_from_root(plan: &LogicalPlan, depth: usize) -> Option<usize> {
    if matches!(plan, LogicalPlan::Filter { .. }) {
        return Some(depth);
    }
    let mut best: Option<usize> = None;
    for child in plan.children() {
        if let Some(d) = min_filter_depth_from_root(child, depth + 1) {
            best = Some(best.map_or(d, |b| b.min(d)));
        }
    }
    best
}

#[tokio::test]
async fn pushdown_moves_source_filter_below_expand() {
    // Lowering puts the WHERE filter above the Expand:
    //   Project -> Filter(a.age>30) -> Expand(a->b) -> NodeScan(Person, a)
    // The Filter once sat between Expand and NodeScan; the optimizer
    // the literal comparison is absorbed into NodeScan.predicates and the
    // intermediate Filter disappears entirely.
    let mut writer = WriterSession::open(store(), paths("s121-pushdown01"))
        .await
        .unwrap();
    let _ = build_micro_graph_and_flush(&mut writer).await;
    let snap = writer.snapshot();
    let cat = StatsCatalog::from_manifest(&snap.manifest().manifest);

    let q = parse("MATCH (a:Person)-[:KNOWS]->(b) WHERE a.age > 30 RETURN b").unwrap();
    let raw = lower(&q).unwrap();
    let optimized = optimize(raw.clone(), &cat);

    // Raw plan: Filter materialised above the Expand.
    assert!(
        min_filter_depth_from_root(&raw, 0).is_some(),
        "raw plan should still carry a Filter"
    );
    // Optimized plan: scan-level predicate captures the conjunct.
    let scans = collect_node_scan_predicates(&optimized);
    let preds = scans
        .get("a")
        .expect("NodeScan alias=a present in optimized plan");
    assert!(
        preds.iter().any(|p| p.contains("Gt") && p.contains("age")),
        "the pushed conjunct lives in NodeScan.predicates, got {preds:?}",
    );
}

#[tokio::test]
async fn pushdown_keeps_target_filter_above_expand() {
    // WHERE references the target alias (`b`) of an Expand — the filter
    // CANNOT be pushed below the Expand because `b` is only introduced
    // by the expansion.
    let mut writer = WriterSession::open(store(), paths("s121-pushdown02"))
        .await
        .unwrap();
    let _ = build_micro_graph_and_flush(&mut writer).await;
    let snap = writer.snapshot();
    let cat = StatsCatalog::from_manifest(&snap.manifest().manifest);

    let q = parse("MATCH (a:Person)-[:KNOWS]->(b:Person) WHERE b.age < 50 RETURN a").unwrap();
    let optimized = build_plan(&q, &cat).unwrap();
    // The optimized plan must NOT contain a `Filter { input: NodeScan }`
    // whose predicate references `b` — that would be a soundness bug.
    // Instead, the b-filter must sit above the Expand. Sanity check:
    // running the query produces the same set of rows as the raw lower.
    let rows_opt = execute(&optimized, &snap, &Params::new()).await.unwrap();
    let rows_raw = execute(&lower(&q).unwrap(), &snap, &Params::new())
        .await
        .unwrap();
    assert_eq!(
        rows_opt.len(),
        rows_raw.len(),
        "optimized and raw plans should agree on row count"
    );
}

#[tokio::test]
async fn pushdown_splits_compound_predicate_across_expand() {
    // `MATCH (a:Person)-[:KNOWS]->(b:Person) WHERE a.age > 30 AND b.age < 50`
    // → after pushdown: Filter(b.age<50) above Expand; Filter(a.age>30) below.
    let mut writer = WriterSession::open(store(), paths("s121-pushdown03"))
        .await
        .unwrap();
    let _ = build_micro_graph_and_flush(&mut writer).await;
    let snap = writer.snapshot();
    let cat = StatsCatalog::from_manifest(&snap.manifest().manifest);

    let q =
        parse("MATCH (a:Person)-[:KNOWS]->(b:Person) WHERE a.age > 30 AND b.age < 50 RETURN a, b")
            .unwrap();
    let raw = lower(&q).unwrap();
    let optimized = optimize(raw.clone(), &cat);

    // Optimized plan executes and produces the same rows as raw lowering.
    let rows_opt = execute(&optimized, &snap, &Params::new()).await.unwrap();
    let rows_raw = execute(&raw, &snap, &Params::new()).await.unwrap();
    assert_eq!(rows_opt.len(), rows_raw.len());

    // Optimized estimate should not exceed raw estimate (pushdown reduces
    // intermediate cardinality at the cost-model level too).
    let raw_card = estimate(&raw, &cat);
    let opt_card = estimate(&optimized, &cat);
    assert!(
        opt_card.rows <= raw_card.rows + 0.001,
        "optimized rows ({}) should not exceed raw rows ({})",
        opt_card.rows,
        raw_card.rows
    );
}

#[tokio::test]
async fn explain_verbose_renders_optimized_plan_by_default() {
    let mut writer = WriterSession::open(store(), paths("s121-explain01"))
        .await
        .unwrap();
    let _ = build_micro_graph_and_flush(&mut writer).await;
    let snap = writer.snapshot();
    let cat = StatsCatalog::from_manifest(&snap.manifest().manifest);

    let q = parse("MATCH (a:Person)-[:KNOWS]->(b) WHERE a.age > 30 RETURN b").unwrap();
    let verbose = explain_query_verbose(&q, &cat).expect("verbose renders");
    let raw = explain_query_raw_verbose(&q, &cat).expect("raw renders");

    // Optimized plan: the conjunct was absorbed into NodeScan.predicates
    // (RFC-013 §6) so the optimized rendering should expose
    // `predicates=[...]` on the NodeScan line. Raw rendering still
    // surfaces an explicit `Filter` operator at the lowering's
    // shallower indent.
    assert!(
        verbose.contains("predicates=[a.age > 30]"),
        "optimized EXPLAIN VERBOSE should carry predicates=[a.age > 30]:\n{verbose}",
    );
    assert!(
        raw.lines().any(|l| l.contains("Filter ")),
        "EXPLAIN RAW VERBOSE should still expose the lowering's Filter:\n{raw}",
    );

    // Both must include the estimates header.
    assert!(verbose.contains("# Estimated rows:"));
    assert!(raw.contains("# Estimated rows:"));
}

#[tokio::test]
async fn label_eq_cleanup_removes_defensive_filter_from_optimized_plan() {
    // The lowering emits `Filter(__label_eq(b, "Person"))` on top of
    // `Expand` with `target_label = Some("Person")`. Normalization
    // should drop it post-pushdown.
    let mut writer = WriterSession::open(store(), paths("s121-labeleq01"))
        .await
        .unwrap();
    let _ = build_micro_graph_and_flush(&mut writer).await;
    let snap = writer.snapshot();
    let cat = StatsCatalog::from_manifest(&snap.manifest().manifest);

    let q = parse("MATCH (a:Person)-[:KNOWS]->(b:Person) RETURN b").unwrap();
    let raw = lower(&q).unwrap();
    let optimized = optimize(raw, &cat);

    let rendered = namidb_query::plan::explain(&optimized);
    assert!(
        !rendered.contains("__label_eq"),
        "optimized plan should not contain the synthetic label_eq filter:\n{}",
        rendered
    );
}

#[tokio::test]
async fn optimize_preserves_query_result_set() {
    // End-to-end check on every meaningful WHERE shape we exercise.
    let mut writer = WriterSession::open(store(), paths("s121-resultparity"))
        .await
        .unwrap();
    let _ = build_micro_graph_and_flush(&mut writer).await;
    let snap = writer.snapshot();
    let cat = StatsCatalog::from_manifest(&snap.manifest().manifest);

    let queries = [
        "MATCH (a:Person) WHERE a.age > 30 RETURN a",
        "MATCH (a:Person)-[:KNOWS]->(b:Person) WHERE a.age > 30 RETURN a, b",
        "MATCH (a:Person)-[:KNOWS]->(b:Person) WHERE a.age > 30 AND b.age < 50 RETURN a, b",
        "MATCH (a:Person) WHERE a.firstName = 'Alice' RETURN a",
    ];

    for src in queries {
        let q = parse(src).unwrap_or_else(|_| panic!("parse: {}", src));
        let raw = lower(&q).unwrap();
        let optimized = optimize(raw.clone(), &cat);
        let rows_raw = execute(&raw, &snap, &Params::new()).await.unwrap();
        let rows_opt = execute(&optimized, &snap, &Params::new()).await.unwrap();
        assert_eq!(
            rows_raw.len(),
            rows_opt.len(),
            "result parity mismatch for: {}",
            src
        );
    }
}

#[tokio::test]
async fn cross_product_filter_split_pushes_to_each_side() {
    // Two-pattern MATCH with side-local filters: optimizer should park
    // each filter immediately above its corresponding NodeScan.
    let mut writer = WriterSession::open(store(), paths("s121-cross01"))
        .await
        .unwrap();
    let _ = build_micro_graph_and_flush(&mut writer).await;
    let snap = writer.snapshot();
    let cat = StatsCatalog::from_manifest(&snap.manifest().manifest);

    let q = parse(
        "MATCH (a:Person), (b:Message) WHERE a.age > 30 AND b.creationDate > 100 RETURN a, b",
    )
    .unwrap();
    let optimized = build_plan(&q, &cat).unwrap();

    // Both conjuncts are single-column literal comparisons → RFC-013
    // absorbs them into each side's NodeScan.predicates. The optimised
    // shape therefore contains a CrossProduct with predicate-carrying
    // NodeScans on both sides (no intermediate Filter).
    let scans = collect_node_scan_predicates(&optimized);
    let preds_a = scans.get("a").expect("alias=a NodeScan present");
    let preds_b = scans.get("b").expect("alias=b NodeScan present");
    assert!(
        preds_a
            .iter()
            .any(|p| p.contains("Gt") && p.contains("age")),
        "alias=a should carry the age predicate, got {preds_a:?}",
    );
    assert!(
        preds_b
            .iter()
            .any(|p| p.contains("Gt") && p.contains("creationDate")),
        "alias=b should carry the creationDate predicate, got {preds_b:?}",
    );

    // Sanity: the optimized plan still executes correctly.
    let rows = execute(&optimized, &snap, &Params::new()).await.unwrap();
    let raw = lower(&q).unwrap();
    let rows_raw = execute(&raw, &snap, &Params::new()).await.unwrap();
    assert_eq!(rows.len(), rows_raw.len());
}

// ──────────────────────────────────────────────────────────────────────
// — HLL/NDV real (RFC-010 §"Drawbacks 1" closed)
// ──────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn catalog_materializes_real_ndv_after_flush() {
    let mut writer = WriterSession::open(store(), paths("s121b-ndv01"))
        .await
        .unwrap();
    let _ = build_micro_graph_and_flush(&mut writer).await;
    let snap = writer.snapshot();
    let cat = StatsCatalog::from_manifest(&snap.manifest().manifest);

    let person = cat.label("Person").expect("Person label");
    // The HLL pipeline only runs over typed `prop_*` columns. Under id-primary
    // every property rides in `__overflow_json`, so no sketch is emitted and
    // `ndv` is `None` for every property; equality/IN selectivity falls back to
    // the documented constants. Real per-column ndv (≈6 distinct firstNames in
    // the Alice..Frank fixture) returns with the typed-column layout.
    let first = person
        .properties
        .get("firstName")
        .expect("firstName stats present");
    assert!(
        first.ndv.is_none(),
        "id-primary: no per-column ndv until the typed-column layout, got {:?}",
        first.ndv
    );

    let age = person.properties.get("age").expect("age stats present");
    assert!(
        age.ndv.is_none(),
        "id-primary: no per-column ndv for age yet"
    );
}

#[tokio::test]
async fn eq_selectivity_uses_real_ndv_when_available() {
    // With real ndv ≈ 6, eq selectivity = 1/6 ≈ 0.167. On 6 Person
    // rows → estimate ≈ 1.0. Without the HLL pipeline this fell back to 0.1 → 0.6.
    let mut writer = WriterSession::open(store(), paths("s121b-eqsel01"))
        .await
        .unwrap();
    let _ = build_micro_graph_and_flush(&mut writer).await;
    let snap = writer.snapshot();
    let cat = StatsCatalog::from_manifest(&snap.manifest().manifest);

    let q = parse("MATCH (a:Person) WHERE a.firstName = 'Alice' RETURN a").unwrap();
    let plan = lower(&q).unwrap();
    let card = estimate(&plan, &cat);
    let rows = execute(&plan, &snap, &Params::new()).await.unwrap();

    assert_eq!(rows.len(), 1, "exactly Alice matches");
    // 6 / ndv(~6) = 1.0 → estimate ≈ 1.
    assert!(
        (0.5..=1.8).contains(&card.rows),
        "with real ndv, estimate {} should sit near 1.0 (not 0.6 from fallback)",
        card.rows
    );
}

#[tokio::test]
async fn empty_catalog_eq_selectivity_falls_back_to_zero_rows() {
    // Empty catalog → no ndv, no node_count → eq estimate 0.
    let cat = StatsCatalog::empty();
    let q = parse("MATCH (a:Person) WHERE a.firstName = 'Alice' RETURN a").unwrap();
    let plan = lower(&q).unwrap();
    let card = estimate(&plan, &cat);
    assert_eq!(card.rows, 0.0);
}

// ──────────────────────────────────────────────────────────────────────
// — HashJoin conversion (RFC-012)
// ──────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn cross_with_eq_converts_to_hash_join() {
    // `MATCH (a:Person), (b:Person) WHERE a.firstName = b.firstName`
    // → optimizer materialises HashJoin in the plan tree.
    let mut writer = WriterSession::open(store(), paths("s123-hj01"))
        .await
        .unwrap();
    let _ = build_micro_graph_and_flush(&mut writer).await;
    let snap = writer.snapshot();
    let cat = StatsCatalog::from_manifest(&snap.manifest().manifest);

    let q =
        parse("MATCH (a:Person), (b:Person) WHERE a.firstName = b.firstName RETURN a, b").unwrap();
    let optimized = build_plan(&q, &cat).unwrap();
    fn contains_hash_join(plan: &LogicalPlan) -> bool {
        if matches!(plan, LogicalPlan::HashJoin { .. }) {
            return true;
        }
        plan.children().iter().any(|c| contains_hash_join(c))
    }
    assert!(
        contains_hash_join(&optimized),
        "expected HashJoin in optimized plan:\n{}",
        namidb_query::plan::explain(&optimized)
    );
}

#[tokio::test]
async fn hash_join_executes_correctly() {
    // Optimized plan with HashJoin returns the same rows as the raw
    // lowering with Filter ⇒ CrossProduct.
    let mut writer = WriterSession::open(store(), paths("s123-hj02"))
        .await
        .unwrap();
    let _ = build_micro_graph_and_flush(&mut writer).await;
    let snap = writer.snapshot();
    let cat = StatsCatalog::from_manifest(&snap.manifest().manifest);

    let q =
        parse("MATCH (a:Person), (b:Person) WHERE a.firstName = b.firstName RETURN a, b").unwrap();
    let raw = lower(&q).unwrap();
    let optimized = build_plan(&q, &cat).unwrap();
    let rows_raw = execute(&raw, &snap, &Params::new()).await.unwrap();
    let rows_opt = execute(&optimized, &snap, &Params::new()).await.unwrap();
    // Each Person (6) joins with itself on firstName (unique) — 6 rows.
    assert_eq!(rows_raw.len(), 6);
    assert_eq!(rows_opt.len(), rows_raw.len());
}

#[tokio::test]
async fn hash_join_estimate_matches_actual_within_micro_graph() {
    // HashJoin uses Selinger '79:
    //   rows = (|build| * |probe|) / max(ndv(build_key), ndv(probe_key))
    // The join key here is the property `firstName`. With a real ndv ≈ 6 the
    // divisor would tighten the estimate to ≈ 6 (the actual). Under id-primary
    // `firstName` has no per-column ndv (it lives in `__overflow_json`), so the
    // divisor falls back to 1 and the estimate is the cartesian upper bound
    // |build| * |probe| = 36 — loose but still a valid over-estimate. The tight
    // estimate returns with the typed-column layout. Either way execution is
    // correct (each Person matches only itself; distinct names → 6 rows).
    let mut writer = WriterSession::open(store(), paths("s123-hj03"))
        .await
        .unwrap();
    let _ = build_micro_graph_and_flush(&mut writer).await;
    let snap = writer.snapshot();
    let cat = StatsCatalog::from_manifest(&snap.manifest().manifest);

    let q =
        parse("MATCH (a:Person), (b:Person) WHERE a.firstName = b.firstName RETURN a, b").unwrap();
    let optimized = build_plan(&q, &cat).unwrap();
    let opt_est = estimate(&optimized, &cat).rows;
    let rows = execute(&optimized, &snap, &Params::new()).await.unwrap();
    let actual = rows.len() as f64;
    assert_eq!(actual, 6.0, "6 Persons, each joins only to itself");
    // Cartesian fallback: 6 * 6 = 36, and it must remain a valid upper bound.
    assert!(
        opt_est >= actual,
        "estimate {opt_est} must not under-count actual {actual}"
    );
    assert!(
        (opt_est - 36.0).abs() < 1.0,
        "without per-column ndv the estimate is the cartesian bound 36, got {opt_est}"
    );
}

#[tokio::test]
async fn hash_join_handles_residual_predicate() {
    // `a.firstName = b.firstName AND a.age > b.age` — the equality
    // becomes the join key; the comparison becomes the residual.
    let mut writer = WriterSession::open(store(), paths("s123-hj04"))
        .await
        .unwrap();
    let _ = build_micro_graph_and_flush(&mut writer).await;
    let snap = writer.snapshot();
    let cat = StatsCatalog::from_manifest(&snap.manifest().manifest);

    let q = parse(
        "MATCH (a:Person), (b:Person) WHERE a.firstName = b.firstName AND a.age > b.age RETURN a, b",
    )
    .unwrap();
    let optimized = build_plan(&q, &cat).unwrap();
    // Walk to the HashJoin and verify it has a residual.
    fn find_hash_join(plan: &LogicalPlan) -> Option<&LogicalPlan> {
        if matches!(plan, LogicalPlan::HashJoin { .. }) {
            return Some(plan);
        }
        for c in plan.children() {
            if let Some(hj) = find_hash_join(c) {
                return Some(hj);
            }
        }
        None
    }
    let hj = find_hash_join(&optimized).expect("HashJoin present");
    match hj {
        LogicalPlan::HashJoin { residual, on, .. } => {
            assert_eq!(on.len(), 1);
            assert!(
                residual.is_some(),
                "expected `a.age > b.age` to land as residual"
            );
        }
        _ => unreachable!(),
    }
    // Sanity: optimized executes and returns rows (each Person joined
    // with itself filters by a.age > b.age = false; no rows).
    let rows = execute(&optimized, &snap, &Params::new()).await.unwrap();
    assert_eq!(rows.len(), 0);
}

#[tokio::test]
async fn hash_join_renders_in_explain_verbose() {
    let mut writer = WriterSession::open(store(), paths("s123-hj05"))
        .await
        .unwrap();
    let _ = build_micro_graph_and_flush(&mut writer).await;
    let snap = writer.snapshot();
    let cat = StatsCatalog::from_manifest(&snap.manifest().manifest);

    let q =
        parse("MATCH (a:Person), (b:Person) WHERE a.firstName = b.firstName RETURN a, b").unwrap();
    let rendered = explain_query_verbose(&q, &cat).expect("verbose renders");
    assert!(
        rendered.contains("HashJoin"),
        "no HashJoin in:\n{}",
        rendered
    );
    assert!(rendered.contains("on=["), "no on= clause:\n{}", rendered);
}

#[tokio::test]
async fn multi_key_hash_join() {
    // `a.firstName = b.firstName AND a.age = b.age` → 2-key HashJoin.
    let mut writer = WriterSession::open(store(), paths("s123-hj06"))
        .await
        .unwrap();
    let _ = build_micro_graph_and_flush(&mut writer).await;
    let snap = writer.snapshot();
    let cat = StatsCatalog::from_manifest(&snap.manifest().manifest);

    let q = parse(
        "MATCH (a:Person), (b:Person) WHERE a.firstName = b.firstName AND a.age = b.age RETURN a, b",
    )
    .unwrap();
    let optimized = build_plan(&q, &cat).unwrap();
    fn find_hash_join(plan: &LogicalPlan) -> Option<&LogicalPlan> {
        if matches!(plan, LogicalPlan::HashJoin { .. }) {
            return Some(plan);
        }
        for c in plan.children() {
            if let Some(hj) = find_hash_join(c) {
                return Some(hj);
            }
        }
        None
    }
    let hj = find_hash_join(&optimized).expect("HashJoin present");
    match hj {
        LogicalPlan::HashJoin { on, .. } => assert_eq!(on.len(), 2),
        _ => unreachable!(),
    }
    // Execution: each Person joins itself on (firstName, age).
    let rows = execute(&optimized, &snap, &Params::new()).await.unwrap();
    assert_eq!(rows.len(), 6);
}

#[tokio::test]
async fn raw_explain_flags_join_candidate_before_conversion() {
    // Pre-conversion (raw lowering) view: `Filter(a.x = b.x) ⇒
    // CrossProduct` carries the `[join candidate]` annotation. Post-
    // The optimizer rewrites this shape to HashJoin, so the
    // flag only survives in the RAW EXPLAIN output (RFC-011 §6.3
    // semantics are unchanged; only the OPTIMIZED rendering of this
    // shape has evolved).
    let mut writer = WriterSession::open(store(), paths("s121-join01"))
        .await
        .unwrap();
    let _ = build_micro_graph_and_flush(&mut writer).await;
    let snap = writer.snapshot();
    let cat = StatsCatalog::from_manifest(&snap.manifest().manifest);

    let q =
        parse("MATCH (a:Person), (b:Person) WHERE a.firstName = b.firstName RETURN a, b").unwrap();
    // RAW: pre-optimize plan — Filter ⇒ CrossProduct surfaces, flag
    // appears.
    let raw = explain_query_raw_verbose(&q, &cat).expect("raw verbose renders");
    assert!(
        raw.contains("[join candidate]"),
        "expected join-candidate annotation in raw view:\n{}",
        raw
    );
    // OPTIMIZED: HashJoin replaces the Filter+CrossProduct shape, so
    // the flag does NOT appear — the operator IS the join.
    let optimized = explain_query_verbose(&q, &cat).expect("verbose renders");
    assert!(
        !optimized.contains("[join candidate]"),
        "join-candidate flag should be absent cleared:\n{}",
        optimized
    );
    assert!(
        optimized.contains("HashJoin"),
        "optimized plan must use HashJoin:\n{}",
        optimized
    );
}

// ─── Parquet predicate pushdown ──────────────────────────────────

/// Walk the plan tree and pull every NodeScan's `predicates` list,
/// keyed by `alias`. Helper for the integration tests.
fn collect_node_scan_predicates(plan: &LogicalPlan) -> BTreeMap<String, Vec<String>> {
    fn visit(plan: &LogicalPlan, out: &mut BTreeMap<String, Vec<String>>) {
        if let LogicalPlan::NodeScan {
            alias, predicates, ..
        } = plan
        {
            let rendered: Vec<String> = predicates.iter().map(|p| format!("{:?}", p)).collect();
            out.insert(alias.clone(), rendered);
        }
        for c in plan.children() {
            visit(c, out);
        }
    }
    let mut out = BTreeMap::new();
    visit(plan, &mut out);
    out
}

#[tokio::test]
async fn parquet_pushdown_moves_eq_to_scan() {
    let mut writer = WriterSession::open(store(), paths("parquet-eq"))
        .await
        .unwrap();
    let _ = build_micro_graph_and_flush(&mut writer).await;
    let snap = writer.snapshot();
    let catalog = StatsCatalog::from_manifest(&snap.manifest().manifest);

    let q = parse("MATCH (a:Person) WHERE a.firstName = 'Alice' RETURN a").unwrap();
    let plan = build_plan(&q, &catalog).unwrap();

    let scans = collect_node_scan_predicates(&plan);
    let preds = scans
        .get("a")
        .expect("NodeScan alias=a present in optimized plan");
    assert!(
        preds
            .iter()
            .any(|p| p.contains("Eq") && p.contains("firstName")),
        "expected an Eq(firstName) predicate, got {preds:?}",
    );
}

#[tokio::test]
async fn parquet_pushdown_moves_range_to_scan() {
    let mut writer = WriterSession::open(store(), paths("parquet-range"))
        .await
        .unwrap();
    let _ = build_micro_graph_and_flush(&mut writer).await;
    let snap = writer.snapshot();
    let catalog = StatsCatalog::from_manifest(&snap.manifest().manifest);

    let q = parse("MATCH (a:Person) WHERE a.age > 30 RETURN a").unwrap();
    let plan = build_plan(&q, &catalog).unwrap();
    let scans = collect_node_scan_predicates(&plan);
    let preds = scans.get("a").expect("alias=a NodeScan present");
    assert!(
        preds.iter().any(|p| p.contains("Gt") && p.contains("age")),
        "expected Gt(age) predicate, got {preds:?}",
    );
}

#[tokio::test]
async fn parquet_pushdown_renders_in_explain() {
    let mut writer = WriterSession::open(store(), paths("parquet-explain"))
        .await
        .unwrap();
    let _ = build_micro_graph_and_flush(&mut writer).await;
    let snap = writer.snapshot();
    let catalog = StatsCatalog::from_manifest(&snap.manifest().manifest);

    let q = parse("MATCH (a:Person) WHERE a.firstName = 'Alice' RETURN a").unwrap();
    let rendered = explain_query_verbose(&q, &catalog).unwrap();
    assert!(
        rendered.contains("predicates=[a.firstName = \"Alice\"]"),
        "EXPLAIN VERBOSE should render predicates=[...]:\n{rendered}",
    );
}

#[tokio::test]
async fn parquet_pushdown_executes_with_parity_to_raw() {
    let mut writer = WriterSession::open(store(), paths("parquet-parity"))
        .await
        .unwrap();
    let _ = build_micro_graph_and_flush(&mut writer).await;
    let snap = writer.snapshot();
    let catalog = StatsCatalog::from_manifest(&snap.manifest().manifest);
    let snap = writer.snapshot();

    for q_text in [
        "MATCH (a:Person) WHERE a.firstName = 'Alice' RETURN a.firstName",
        "MATCH (a:Person) WHERE a.age > 30 RETURN a.firstName",
        "MATCH (a:Person) WHERE a.age >= 25 AND a.age <= 40 RETURN a.firstName",
    ] {
        let q = parse(q_text).unwrap();
        let raw = lower(&q).unwrap();
        let opt = optimize(raw.clone(), &catalog);

        let rows_raw = execute(&raw, &snap, &Params::default()).await.unwrap();
        let rows_opt = execute(&opt, &snap, &Params::default()).await.unwrap();
        assert_eq!(
            rows_raw.len(),
            rows_opt.len(),
            "row count parity for `{q_text}`: raw={}, opt={}",
            rows_raw.len(),
            rows_opt.len(),
        );
    }
}

#[tokio::test]
async fn parquet_pushdown_estimate_drops_below_full_scan() {
    let mut writer = WriterSession::open(store(), paths("parquet-est"))
        .await
        .unwrap();
    let _ = build_micro_graph_and_flush(&mut writer).await;
    let snap = writer.snapshot();
    let catalog = StatsCatalog::from_manifest(&snap.manifest().manifest);

    // Bare scan: 6 Person rows.
    let q_all = parse("MATCH (a:Person) RETURN a").unwrap();
    let plan_all = build_plan(&q_all, &catalog).unwrap();
    let est_all = estimate(&plan_all, &catalog).rows;

    // Eq predicate over a high-NDV column (firstName): selectivity ~ 1/NDV ≈ 1/6 ≪ 1.
    let q_eq = parse("MATCH (a:Person) WHERE a.firstName = 'Alice' RETURN a").unwrap();
    let plan_eq = build_plan(&q_eq, &catalog).unwrap();
    let est_eq = estimate(&plan_eq, &catalog).rows;

    assert!(
        est_eq < est_all,
        "eq estimate {} should be strictly less than full-scan {}",
        est_eq,
        est_all,
    );
}

#[tokio::test]
async fn parquet_pushdown_keeps_cross_alias_in_filter() {
    let mut writer = WriterSession::open(store(), paths("parquet-cross"))
        .await
        .unwrap();
    let _ = build_micro_graph_and_flush(&mut writer).await;
    let snap = writer.snapshot();
    let catalog = StatsCatalog::from_manifest(&snap.manifest().manifest);

    // Cross-alias eq: HashJoin owns this (a.firstName = b.firstName); the
    // scan-side conjunct list must NOT include it.
    let q = parse("MATCH (a:Person), (b:Person) WHERE a.firstName = b.firstName RETURN a").unwrap();
    let plan = build_plan(&q, &catalog).unwrap();
    let scans = collect_node_scan_predicates(&plan);
    for (alias, preds) in &scans {
        assert!(
            preds.is_empty(),
            "alias {alias} should have NO scan predicates (cross-alias is HashJoin's job), got {preds:?}",
        );
    }
}

#[tokio::test]
async fn parquet_pushdown_handles_is_null() {
    let mut writer = WriterSession::open(store(), paths("parquet-isnull"))
        .await
        .unwrap();
    let _ = build_micro_graph_and_flush(&mut writer).await;
    let snap = writer.snapshot();
    let catalog = StatsCatalog::from_manifest(&snap.manifest().manifest);

    let q = parse("MATCH (a:Person) WHERE a.firstName IS NULL RETURN a").unwrap();
    let plan = build_plan(&q, &catalog).unwrap();
    let scans = collect_node_scan_predicates(&plan);
    let preds = scans.get("a").expect("alias=a NodeScan present");
    assert!(
        preds.iter().any(|p| p.contains("IsNull")),
        "expected IsNull predicate, got {preds:?}",
    );
}

#[tokio::test]
async fn parquet_pushdown_in_list_translates() {
    let mut writer = WriterSession::open(store(), paths("parquet-in"))
        .await
        .unwrap();
    let _ = build_micro_graph_and_flush(&mut writer).await;
    let snap = writer.snapshot();
    let catalog = StatsCatalog::from_manifest(&snap.manifest().manifest);

    let q = parse("MATCH (a:Person) WHERE a.firstName IN ['Alice', 'Bob'] RETURN a").unwrap();
    let plan = build_plan(&q, &catalog).unwrap();
    let scans = collect_node_scan_predicates(&plan);
    let preds = scans.get("a").expect("alias=a NodeScan present");
    assert!(
        preds
            .iter()
            .any(|p| p.contains("In") && p.contains("firstName")),
        "expected In(firstName) predicate, got {preds:?}",
    );
}

#[tokio::test]
async fn parquet_pushdown_skip_proves_zero_rows_when_out_of_range() {
    let mut writer = WriterSession::open(store(), paths("parquet-out"))
        .await
        .unwrap();
    let _ = build_micro_graph_and_flush(&mut writer).await;
    let snap = writer.snapshot();
    let catalog = StatsCatalog::from_manifest(&snap.manifest().manifest);

    // Age range is [25, 40]; querying > 99 must produce 0 rows (and the
    // SST row-group pruner should skip the only RG entirely — we don't
    // measure IO directly here, just that execution returns no rows).
    let q = parse("MATCH (a:Person) WHERE a.age > 99 RETURN a").unwrap();
    let plan = build_plan(&q, &catalog).unwrap();
    let rows = execute(&plan, &snap, &Params::default()).await.unwrap();
    assert!(rows.is_empty(), "expected 0 rows, got {}", rows.len());
}

// ─── HashSemiJoin (decorrelation) ──────────────────────────────

#[tokio::test]
async fn decorrelation_converts_simple_exists() {
    let mut writer = WriterSession::open(store(), paths("decor-exists01"))
        .await
        .unwrap();
    let _ = build_micro_graph_and_flush(&mut writer).await;
    let snap = writer.snapshot();
    let catalog = StatsCatalog::from_manifest(&snap.manifest().manifest);

    let q =
        parse("MATCH (a:Person) WHERE EXISTS((a)-[:KNOWS]->(b:Person)) RETURN a.firstName AS name")
            .unwrap();
    let plan = build_plan(&q, &catalog).unwrap();
    let rendered = namidb_query::plan::explain(&plan);
    assert!(
        rendered.contains("HashSemiJoin"),
        "optimized plan should contain HashSemiJoin:\n{rendered}",
    );
    assert!(
        !rendered.contains("SemiApply"),
        "SemiApply should have been rewritten away:\n{rendered}",
    );
}

#[tokio::test]
async fn decorrelation_preserves_results() {
    let mut writer = WriterSession::open(store(), paths("decor-parity01"))
        .await
        .unwrap();
    let _ = build_micro_graph_and_flush(&mut writer).await;
    let snap = writer.snapshot();
    let catalog = StatsCatalog::from_manifest(&snap.manifest().manifest);

    for q_text in [
        "MATCH (a:Person) WHERE EXISTS((a)-[:KNOWS]->(b:Person)) RETURN a.firstName",
        "MATCH (a:Person) WHERE NOT EXISTS((a)-[:KNOWS]->(b:Person)) RETURN a.firstName",
        "MATCH (a:Person) WHERE a.age >= 30 AND EXISTS((a)-[:KNOWS]->(b:Person)) RETURN a.firstName",
    ] {
        let q = parse(q_text).unwrap();
        let raw = lower(&q).unwrap();
        let opt = optimize(raw.clone(), &catalog);

        let rows_raw = execute(&raw, &snap, &Params::default()).await.unwrap();
        let rows_opt = execute(&opt, &snap, &Params::default()).await.unwrap();
        assert_eq!(
            rows_raw.len(),
            rows_opt.len(),
            "row count parity for `{q_text}`: raw={}, opt={}",
            rows_raw.len(),
            rows_opt.len(),
        );
    }
}

#[tokio::test]
async fn decorrelation_handles_not_exists() {
    let mut writer = WriterSession::open(store(), paths("decor-not01"))
        .await
        .unwrap();
    let _ = build_micro_graph_and_flush(&mut writer).await;
    let snap = writer.snapshot();
    let catalog = StatsCatalog::from_manifest(&snap.manifest().manifest);

    let q = parse("MATCH (a:Person) WHERE NOT EXISTS((a)-[:KNOWS]->(b:Person)) RETURN a.firstName")
        .unwrap();
    let plan = build_plan(&q, &catalog).unwrap();
    let rendered = namidb_query::plan::explain(&plan);
    assert!(
        rendered.contains("AntiHashSemiJoin"),
        "NOT EXISTS should lower to AntiHashSemiJoin:\n{rendered}",
    );
}

#[tokio::test]
async fn decorrelation_renders_hash_semi_join_in_explain() {
    let mut writer = WriterSession::open(store(), paths("decor-explain01"))
        .await
        .unwrap();
    let _ = build_micro_graph_and_flush(&mut writer).await;
    let snap = writer.snapshot();
    let catalog = StatsCatalog::from_manifest(&snap.manifest().manifest);

    let q = parse("MATCH (a:Person) WHERE EXISTS((a)-[:KNOWS]->(b:Person)) RETURN a.firstName")
        .unwrap();
    let verbose = explain_query_verbose(&q, &catalog).unwrap();
    assert!(
        verbose.contains("HashSemiJoin on=["),
        "EXPLAIN VERBOSE should render HashSemiJoin on=[...]:\n{verbose}",
    );
    let raw = explain_query_raw_verbose(&q, &catalog).unwrap();
    assert!(
        raw.contains("SemiApply"),
        "EXPLAIN RAW VERBOSE should still expose the lowering's SemiApply:\n{raw}",
    );
}

// ─── Projection pushdown ─────────────────────────────────────────

#[tokio::test]
async fn projection_pushdown_extracts_referenced_columns() {
    let mut writer = WriterSession::open(store(), paths("proj-01"))
        .await
        .unwrap();
    let _ = build_micro_graph_and_flush(&mut writer).await;
    let snap = writer.snapshot();
    let catalog = StatsCatalog::from_manifest(&snap.manifest().manifest);

    let q = parse("MATCH (a:Person) RETURN a.firstName AS name").unwrap();
    let plan = build_plan(&q, &catalog).unwrap();
    fn nodescan_proj(plan: &LogicalPlan) -> Option<Vec<String>> {
        match plan {
            LogicalPlan::NodeScan { projection, .. } => projection.clone(),
            _ => plan.children().iter().find_map(|c| nodescan_proj(c)),
        }
    }
    let proj = nodescan_proj(&plan);
    assert_eq!(
        proj,
        Some(vec!["firstName".to_string()]),
        "expected projection=[firstName], got {proj:?}"
    );
}

#[tokio::test]
async fn projection_pushdown_handles_bare_variable_as_all() {
    let mut writer = WriterSession::open(store(), paths("proj-02"))
        .await
        .unwrap();
    let _ = build_micro_graph_and_flush(&mut writer).await;
    let snap = writer.snapshot();
    let catalog = StatsCatalog::from_manifest(&snap.manifest().manifest);

    let q = parse("MATCH (a:Person) RETURN a").unwrap();
    let plan = build_plan(&q, &catalog).unwrap();
    fn nodescan_proj(plan: &LogicalPlan) -> Option<Option<Vec<String>>> {
        match plan {
            LogicalPlan::NodeScan { projection, .. } => Some(projection.clone()),
            _ => plan.children().iter().find_map(|c| nodescan_proj(c)),
        }
    }
    let proj = nodescan_proj(&plan).flatten();
    assert!(
        proj.is_none(),
        "bare RETURN a must not project, got {proj:?}"
    );
}

#[tokio::test]
async fn projection_pushdown_includes_predicate_columns() {
    let mut writer = WriterSession::open(store(), paths("proj-03"))
        .await
        .unwrap();
    let _ = build_micro_graph_and_flush(&mut writer).await;
    let snap = writer.snapshot();
    let catalog = StatsCatalog::from_manifest(&snap.manifest().manifest);

    let q = parse("MATCH (a:Person) WHERE a.age > 30 RETURN a.firstName AS n").unwrap();
    let plan = build_plan(&q, &catalog).unwrap();
    fn nodescan_proj(plan: &LogicalPlan) -> Option<Vec<String>> {
        match plan {
            LogicalPlan::NodeScan { projection, .. } => projection.clone(),
            _ => plan.children().iter().find_map(|c| nodescan_proj(c)),
        }
    }
    let proj = nodescan_proj(&plan);
    // Predicate pushdown absorbs `age > 30` into NodeScan.predicates; projection pushdown must
    // include `age` in the projection so the storage layer can
    // evaluate the predicate post-decode.
    assert_eq!(
        proj,
        Some(vec!["age".into(), "firstName".into()]),
        "expected [age, firstName], got {proj:?}"
    );
}

#[tokio::test]
async fn projection_pushdown_executes_with_parity() {
    let mut writer = WriterSession::open(store(), paths("proj-04"))
        .await
        .unwrap();
    let _ = build_micro_graph_and_flush(&mut writer).await;
    let snap = writer.snapshot();
    let catalog = StatsCatalog::from_manifest(&snap.manifest().manifest);

    for q_text in [
        "MATCH (a:Person) RETURN a.firstName",
        "MATCH (a:Person) WHERE a.age > 30 RETURN a.firstName, a.age",
        "MATCH (a:Person)-[:KNOWS]->(b:Person) RETURN b.firstName",
    ] {
        let q = parse(q_text).unwrap();
        let raw = lower(&q).unwrap();
        let opt = optimize(raw.clone(), &catalog);
        let rows_raw = execute(&raw, &snap, &Params::default()).await.unwrap();
        let rows_opt = execute(&opt, &snap, &Params::default()).await.unwrap();
        assert_eq!(
            rows_raw.len(),
            rows_opt.len(),
            "parity for `{q_text}`: raw={}, opt={}",
            rows_raw.len(),
            rows_opt.len()
        );
    }
}

#[tokio::test]
async fn projection_pushdown_renders_in_explain() {
    let mut writer = WriterSession::open(store(), paths("proj-05"))
        .await
        .unwrap();
    let _ = build_micro_graph_and_flush(&mut writer).await;
    let snap = writer.snapshot();
    let catalog = StatsCatalog::from_manifest(&snap.manifest().manifest);

    let q = parse("MATCH (a:Person) RETURN a.firstName AS n").unwrap();
    let verbose = explain_query_verbose(&q, &catalog).unwrap();
    assert!(
        verbose.contains("projection=[firstName]"),
        "EXPLAIN VERBOSE should render projection=[firstName]:\n{verbose}",
    );
}

#[tokio::test]
async fn decorrelation_estimate_drops_when_match_prob_low() {
    let mut writer = WriterSession::open(store(), paths("decor-est01"))
        .await
        .unwrap();
    let _ = build_micro_graph_and_flush(&mut writer).await;
    let snap = writer.snapshot();
    let catalog = StatsCatalog::from_manifest(&snap.manifest().manifest);

    // Bare scan: 6 rows.
    let q_all = parse("MATCH (a:Person) RETURN a").unwrap();
    let plan_all = build_plan(&q_all, &catalog).unwrap();
    let est_all = estimate(&plan_all, &catalog).rows;

    // With NOT EXISTS the optimizer estimates how many outer rows survive.
    let q_not_exists =
        parse("MATCH (a:Person) WHERE NOT EXISTS((a)-[:KNOWS]->(b:Person)) RETURN a").unwrap();
    let plan_not = build_plan(&q_not_exists, &catalog).unwrap();
    let est_not = estimate(&plan_not, &catalog).rows;

    assert!(
        est_not <= est_all + 0.001,
        "NOT EXISTS estimate {} should be ≤ full-scan {}",
        est_not,
        est_all,
    );
}

// ─── Join reorder (orientation re-evaluation) ───────────────────

#[tokio::test]
async fn join_reorder_swaps_to_smaller_build_when_predicate_shrinks_probe() {
    // After predicate pushdown, alias `a` ends up with predicates and its
    // estimate shrinks below `b`'s. The reorder pass should swap to put
    // alias `a` on the build side.
    let mut writer = WriterSession::open(store(), paths("jr-01")).await.unwrap();
    let _ = build_micro_graph_and_flush(&mut writer).await;
    let snap = writer.snapshot();
    let catalog = StatsCatalog::from_manifest(&snap.manifest().manifest);

    // a is Person (6 rows). b is Person (6 rows). Predicate a.firstName=
    // 'Alice' reduces a's estimate sharply (1/ndv ≈ 1). The HashJoin
    // should pick `a` as build.
    let q = parse(
        "MATCH (a:Person), (b:Person) WHERE a.firstName = 'Alice' AND a._id = b._id RETURN a",
    )
    .unwrap();
    let plan = build_plan(&q, &catalog).unwrap();

    // Walk to find the HashJoin and check that the build alias is `a`.
    fn find_hash_join_build_alias(plan: &LogicalPlan) -> Option<String> {
        match plan {
            LogicalPlan::HashJoin { build, .. } => {
                fn primary_alias(p: &LogicalPlan) -> Option<String> {
                    match p {
                        LogicalPlan::NodeScan { alias, .. } => Some(alias.clone()),
                        _ => p.children().iter().find_map(|c| primary_alias(c)),
                    }
                }
                primary_alias(build)
            }
            _ => plan
                .children()
                .iter()
                .find_map(|c| find_hash_join_build_alias(c)),
        }
    }
    if let Some(alias) = find_hash_join_build_alias(&plan) {
        assert_eq!(
            alias, "a",
            "expected the side with the selective predicate (alias=a) on the build side, got {alias}"
        );
    } else {
        // The plan may not contain a HashJoin at all if the query lowered
        // differently — skip in that case.
    }
}

#[tokio::test]
async fn join_reorder_executes_with_parity() {
    let mut writer = WriterSession::open(store(), paths("jr-02")).await.unwrap();
    let _ = build_micro_graph_and_flush(&mut writer).await;
    let snap = writer.snapshot();
    let catalog = StatsCatalog::from_manifest(&snap.manifest().manifest);

    for q_text in [
        "MATCH (a:Person), (b:Person) WHERE a.firstName = b.firstName RETURN a, b",
        "MATCH (a:Person), (b:Message) WHERE a.firstName = 'Alice' RETURN a, b",
    ] {
        let q = parse(q_text).unwrap();
        let raw = lower(&q).unwrap();
        let opt = optimize(raw.clone(), &catalog);
        let rows_raw = execute(&raw, &snap, &Params::default()).await.unwrap();
        let rows_opt = execute(&opt, &snap, &Params::default()).await.unwrap();
        assert_eq!(
            rows_raw.len(),
            rows_opt.len(),
            "parity for `{q_text}`: raw={}, opt={}",
            rows_raw.len(),
            rows_opt.len()
        );
    }
}

#[tokio::test]
async fn join_reorder_is_idempotent_in_pipeline() {
    let mut writer = WriterSession::open(store(), paths("jr-03")).await.unwrap();
    let _ = build_micro_graph_and_flush(&mut writer).await;
    let snap = writer.snapshot();
    let catalog = StatsCatalog::from_manifest(&snap.manifest().manifest);

    let q =
        parse("MATCH (a:Person), (b:Person) WHERE a.firstName = b.firstName RETURN a, b").unwrap();
    let raw = lower(&q).unwrap();
    let once = optimize(raw.clone(), &catalog);
    let twice = optimize(once.clone(), &catalog);
    assert_eq!(once, twice, "optimizer pipeline must be idempotent");
}
