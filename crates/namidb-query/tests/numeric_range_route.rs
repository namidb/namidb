//! Item 80: a range predicate on an indexed numeric property must be served
//! by the posting index, and must return exactly what the scan returns.
//!
//! Measured on 160,000 rows with `idx` indexed, before and after:
//!
//! | query | before | after |
//! |---|---|---|
//! | `WHERE t.idx > 999999` (0 rows) | 0.311 s | 0.000068 s |
//! | `WHERE t.idx < -5` (0 rows) | 0.399 s | 0.000053 s |
//! | `WHERE t.idx > 159990` (9 rows) | 0.292 s | 0.000109 s |
//! | `WHERE t.idx > 80000` (79,999 rows) | 0.340 s | scan, declined |
//!
//! Two halves, both load-bearing, same as the equality suite:
//!
//!   1. EQUIVALENCE against an UNINDEXED twin holding identical values. The
//!      scan is already right, so this passes before the change; its job is
//!      to guard the new route, whose candidate set is deliberately a
//!      superset (a lossy key above 2^53, stale postings, and raw String
//!      keys inside the numeric byte window).
//!   2. NON-INERT: the route counter must actually move, or the agreement is
//!      the trivial kind where everything declined to the scan.

use std::collections::BTreeMap;
use std::sync::Arc;

use namidb_core::id::{NamespaceId, NodeId};
use namidb_core::value::Value as CoreValue;
use namidb_storage::{route_telemetry, NamespacePaths, NodeWriteRecord, WriterSession};
use object_store::memory::InMemory;
use object_store::ObjectStore;

use namidb_query::{execute, lower, optimize, parse, Params, Row, RuntimeValue, StatsCatalog};

const ROWS: i64 = 3_000;

fn thing(idx: i64, tag: &str) -> NodeWriteRecord {
    NodeWriteRecord {
        properties: BTreeMap::from([
            // `idx` is indexed; `uidx` carries the SAME values with no index,
            // so its spelling is always served by the scan.
            ("idx".to_string(), CoreValue::I64(idx)),
            ("uidx".to_string(), CoreValue::I64(idx)),
            ("tag".to_string(), CoreValue::Str(tag.to_string())),
        ]),
        schema_version: 1,
        ..Default::default()
    }
}

fn odd(idx: CoreValue, uidx: CoreValue, tag: &str) -> NodeWriteRecord {
    NodeWriteRecord {
        properties: BTreeMap::from([
            ("idx".to_string(), idx),
            ("uidx".to_string(), uidx),
            ("tag".to_string(), CoreValue::Str(tag.to_string())),
        ]),
        schema_version: 1,
        ..Default::default()
    }
}

async fn corpus(name: &str) -> WriterSession {
    let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
    let paths = NamespacePaths::new("tenants", NamespaceId::new(name).unwrap());
    let mut writer = WriterSession::open(store, paths).await.unwrap();

    for ordinal in 0..ROWS {
        writer
            .upsert_node("T", NodeId::new(), &thing(ordinal, &format!("r{ordinal}")))
            .unwrap();
    }
    // The values a key-window walk can pick up by accident, or miss.
    for (value, tag) in [
        (CoreValue::F64(10.5), "fractional"),
        (CoreValue::F64(20.0), "integral-float"),
        (CoreValue::F64(-3.25), "negative-float"),
        (CoreValue::I64(-7), "negative-int"),
        (CoreValue::Str("500".into()), "string-on-numeric"),
        (CoreValue::Null, "null"),
        (CoreValue::I64(9_007_199_254_740_992), "big-low"),
        (CoreValue::I64(9_007_199_254_740_993), "big-high"),
    ] {
        writer
            .upsert_node("T", NodeId::new(), &odd(value.clone(), value, tag))
            .unwrap();
    }
    writer.commit_batch().await.unwrap();
    writer.create_property_index("T", "idx").await.unwrap();
    let schema = writer.snapshot().manifest().manifest.schema.clone();
    writer.flush(schema.clone()).await.unwrap();
    writer.compact_l0(&schema).await.unwrap();
    writer
}

fn render(v: &RuntimeValue) -> String {
    match v {
        RuntimeValue::Null => "null".into(),
        RuntimeValue::String(s) => s.clone(),
        RuntimeValue::Integer(n) => n.to_string(),
        RuntimeValue::Float(f) => format!("{f:?}"),
        RuntimeValue::Bool(b) => b.to_string(),
        other => format!("{other:?}"),
    }
}

fn multiset(rows: &[Row]) -> Vec<String> {
    let mut out: Vec<String> = rows
        .iter()
        .map(|r| {
            r.bindings
                .iter()
                .map(|(k, v)| format!("{k}={}", render(v)))
                .collect::<Vec<_>>()
                .join(",")
        })
        .collect();
    out.sort();
    out
}

async fn run(writer: &WriterSession, query: &str) -> Vec<String> {
    let snapshot = writer.snapshot();
    let catalog = StatsCatalog::from_manifest(&snapshot.manifest().manifest);
    let parsed = parse(query).unwrap_or_else(|e| panic!("parse `{query}`: {e:?}"));
    let plan = optimize(
        lower(&parsed).unwrap_or_else(|e| panic!("lower `{query}`: {e:?}")),
        &catalog,
    );
    let rows = execute(&plan, &snapshot, &Params::new())
        .await
        .unwrap_or_else(|e| panic!("execute `{query}`: {e:?}"));
    multiset(&rows)
}

/// `(name, predicate on the indexed property)`. The twin is derived by
/// swapping `idx` for the unindexed `uidx`.
fn shapes() -> Vec<(&'static str, &'static str)> {
    vec![
        ("a narrow tail", "t.IDX > 2990"),
        ("inclusive tail", "t.IDX >= 2990"),
        ("a narrow head", "t.IDX < 5"),
        ("inclusive head", "t.IDX <= 5"),
        ("a band spanning the floats", "t.IDX > 10 AND t.IDX < 25"),
        ("a band that is one value", "t.IDX >= 20 AND t.IDX <= 20"),
        ("spanning the negatives", "t.IDX > -10 AND t.IDX < 1"),
        ("a negative bound", "t.IDX < -5"),
        ("matching nothing, low", "t.IDX < -100"),
        ("matching nothing, high", "t.IDX > 100000"),
        ("beyond 2^53", "t.IDX > 9007199254740992"),
        ("float bounds", "t.IDX > 10.4 AND t.IDX < 10.6"),
        ("an inverted band", "t.IDX > 2999 AND t.IDX < 2991"),
        ("a bound reversed", "2990 < t.IDX"),
        ("negated", "NOT t.IDX <= 2990"),
        (
            "with an unrelated conjunct",
            "t.IDX > 2990 AND t.tag IS NOT NULL",
        ),
        ("ORed with another property", "t.IDX > 2995 OR t.tag = 'r1'"),
    ]
}

#[tokio::test]
async fn numeric_ranges_agree_with_the_scan_and_use_the_index() {
    let writer = corpus("range-route").await;

    let mut served = 0_usize;
    for (name, shape) in shapes() {
        let indexed = format!(
            "MATCH (t:T) WHERE {} RETURN t.tag AS tag",
            shape.replace("IDX", "idx")
        );
        let unindexed = format!(
            "MATCH (t:T) WHERE {} RETURN t.tag AS tag",
            shape.replace("IDX", "uidx")
        );

        let before = route_telemetry::snapshot();
        let got = run(&writer, &indexed).await;
        let after = route_telemetry::snapshot();
        let expected = run(&writer, &unindexed).await;

        assert!(
            got == expected,
            "\n{name}: the indexed range disagreed with its unindexed twin.\n  \
             {indexed}\n    -> {got:?}\n  {unindexed}\n    -> {expected:?}\n"
        );
        if after.property_native > before.property_native {
            served += 1;
        }
    }

    // Agreement is trivial if every shape declined. Most of these are narrow
    // windows over 3,000 rows and must take the index.
    assert!(
        served >= 8,
        "only {served} of {} range shapes took the index route",
        shapes().len()
    );
}

/// A range wider than the label's own share must hand the query back rather
/// than read every posting AND hydrate every row.
#[tokio::test]
async fn a_range_matching_most_of_the_label_stays_on_the_scan() {
    let writer = corpus("range-wide").await;

    let before = route_telemetry::snapshot();
    let wide = run(
        &writer,
        "MATCH (t:T) WHERE t.idx > -1000000 RETURN t.tag AS tag",
    )
    .await;
    let after = route_telemetry::snapshot();
    let twin = run(
        &writer,
        "MATCH (t:T) WHERE t.uidx > -1000000 RETURN t.tag AS tag",
    )
    .await;

    assert_eq!(wide, twin, "declining must not change the answer");
    assert_eq!(
        after.property_native, before.property_native,
        "a range matching nearly the whole label must not take the index route"
    );
}
