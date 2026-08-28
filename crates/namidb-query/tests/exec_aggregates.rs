//! Semantic tests for the aggregate family beyond `count`.
//!
//! openCypher fixes the empty/null envelope precisely: `count` and `sum` over
//! zero rows are 0, `avg`/`min`/`max` are NULL, `collect` is the empty list,
//! and every aggregate skips NULL inputs. These are asserted on the unflushed
//! memtable route and re-asserted after a flush, since a 25 TB corpus answers
//! virtually every aggregate from persisted SSTs.

use std::collections::BTreeMap;
use std::sync::Arc;

use namidb_core::id::{NamespaceId, NodeId};
use namidb_core::schema::{DataType, LabelDef, PropertyDef, SchemaBuilder};
use namidb_core::value::Value as CoreValue;
use namidb_storage::{NamespacePaths, NodeWriteRecord, WriterSession};
use object_store::memory::InMemory;
use object_store::ObjectStore;

use namidb_query::{execute, lower, parse, Params, RuntimeValue};

fn store() -> Arc<dyn ObjectStore> {
    Arc::new(InMemory::new())
}

fn paths(name: &str) -> NamespacePaths {
    NamespacePaths::new("tenants", NamespaceId::new(name).unwrap())
}

fn record(props: Vec<(&str, CoreValue)>) -> NodeWriteRecord {
    let mut properties = BTreeMap::new();
    for (k, v) in props {
        properties.insert(k.to_string(), v);
    }
    NodeWriteRecord {
        properties,
        schema_version: 1,
        ..Default::default()
    }
}

/// Corpus: four Person rows covering ints, a float, a NULL age and a
/// duplicate value, plus one row of an unrelated (and otherwise empty) label.
async fn corpus(name: &str) -> (WriterSession, namidb_core::Schema) {
    let mut writer = WriterSession::open(store(), paths(name)).await.unwrap();
    let schema = SchemaBuilder::new()
        .label(LabelDef {
            name: "Person".into(),
            properties: vec![
                PropertyDef::new("age", DataType::Int64, true).unwrap(),
                PropertyDef::new("score", DataType::Float64, true).unwrap(),
                PropertyDef::new("team", DataType::Utf8, true).unwrap(),
            ],
        })
        .unwrap()
        .label(LabelDef {
            name: "Ghost".into(),
            properties: vec![PropertyDef::new("age", DataType::Int64, true).unwrap()],
        })
        .unwrap()
        .build();
    let rows: Vec<Vec<(&str, CoreValue)>> = vec![
        vec![
            ("age", CoreValue::I64(30)),
            ("score", CoreValue::F64(1.5)),
            ("team", CoreValue::Str("a".into())),
        ],
        vec![
            ("age", CoreValue::I64(40)),
            ("score", CoreValue::F64(2.5)),
            ("team", CoreValue::Str("a".into())),
        ],
        vec![
            // age NULL by omission: every aggregate must skip it.
            ("score", CoreValue::F64(4.0)),
            ("team", CoreValue::Str("b".into())),
        ],
        vec![
            // Duplicate age for the DISTINCT variants.
            ("age", CoreValue::I64(30)),
            ("score", CoreValue::F64(2.0)),
            ("team", CoreValue::Str("b".into())),
        ],
    ];
    for props in rows {
        writer
            .upsert_node("Person", NodeId::new(), &record(props))
            .unwrap();
    }
    writer.commit_batch().await.unwrap();
    (writer, schema)
}

async fn one_value(writer: &WriterSession, q: &str) -> RuntimeValue {
    let snap = writer.snapshot();
    let plan = lower(&parse(q).unwrap()).unwrap();
    let rows = execute(&plan, &snap, &Params::new()).await.unwrap();
    assert_eq!(rows.len(), 1, "expected one aggregate row for {q}");
    rows[0]
        .bindings
        .values()
        .next()
        .expect("aggregate row must carry one column")
        .clone()
}

async fn assert_semantics(writer: &WriterSession) {
    // Null-skipping and DISTINCT.
    assert_eq!(
        one_value(writer, "MATCH (p:Person) RETURN count(p.age) AS c").await,
        RuntimeValue::Integer(3),
        "count(expr) skips the NULL age"
    );
    assert_eq!(
        one_value(writer, "MATCH (p:Person) RETURN count(DISTINCT p.age) AS c").await,
        RuntimeValue::Integer(2),
        "count(DISTINCT) collapses the duplicate 30"
    );
    assert_eq!(
        one_value(writer, "MATCH (p:Person) RETURN sum(p.age) AS s").await,
        RuntimeValue::Integer(100),
        "sum skips NULL and keeps integer typing"
    );
    assert_eq!(
        one_value(writer, "MATCH (p:Person) RETURN sum(DISTINCT p.age) AS s").await,
        RuntimeValue::Integer(70),
    );
    assert_eq!(
        one_value(writer, "MATCH (p:Person) RETURN sum(p.score) AS s").await,
        RuntimeValue::Float(10.0),
        "float sum stays float"
    );
    assert_eq!(
        one_value(writer, "MATCH (p:Person) RETURN avg(p.age) AS a").await,
        RuntimeValue::Float(100.0 / 3.0),
        "avg divides by non-null count only"
    );
    assert_eq!(
        one_value(writer, "MATCH (p:Person) RETURN min(p.age) AS m").await,
        RuntimeValue::Integer(30),
    );
    assert_eq!(
        one_value(writer, "MATCH (p:Person) RETURN max(p.age) AS m").await,
        RuntimeValue::Integer(40),
    );
    match one_value(writer, "MATCH (p:Person) RETURN collect(p.age) AS l").await {
        RuntimeValue::List(values) => {
            assert_eq!(
                values.len(),
                3,
                "collect skips NULL and keeps the duplicate"
            );
            assert!(values
                .iter()
                .all(|value| !matches!(value, RuntimeValue::Null)));
        }
        other => panic!("collect must yield a list, got {other:?}"),
    }
    match one_value(
        writer,
        "MATCH (p:Person) RETURN collect(DISTINCT p.age) AS l",
    )
    .await
    {
        RuntimeValue::List(values) => assert_eq!(values.len(), 2),
        other => panic!("collect DISTINCT must yield a list, got {other:?}"),
    }

    // Empty-input envelope (a label with zero rows).
    assert_eq!(
        one_value(writer, "MATCH (g:Ghost) RETURN count(g) AS c").await,
        RuntimeValue::Integer(0),
    );
    assert_eq!(
        one_value(writer, "MATCH (g:Ghost) RETURN sum(g.age) AS s").await,
        RuntimeValue::Integer(0),
        "openCypher: sum over zero rows is 0, not NULL"
    );
    assert_eq!(
        one_value(writer, "MATCH (g:Ghost) RETURN avg(g.age) AS a").await,
        RuntimeValue::Null,
        "openCypher: avg over zero rows is NULL"
    );
    assert_eq!(
        one_value(writer, "MATCH (g:Ghost) RETURN min(g.age) AS m").await,
        RuntimeValue::Null,
    );
    assert_eq!(
        one_value(writer, "MATCH (g:Ghost) RETURN max(g.age) AS m").await,
        RuntimeValue::Null,
    );
    assert_eq!(
        one_value(writer, "MATCH (g:Ghost) RETURN collect(g.age) AS l").await,
        RuntimeValue::List(Vec::new()),
        "collect over zero rows is the empty list"
    );

    // Grouped: team `b` has one NULL age and one 30, so its avg divides by 1.
    let snap = writer.snapshot();
    let plan = lower(
        &parse(
            "MATCH (p:Person) RETURN p.team AS team, avg(p.age) AS a, \
             count(p.age) AS c ORDER BY team",
        )
        .unwrap(),
    )
    .unwrap();
    let rows = execute(&plan, &snap, &Params::new()).await.unwrap();
    assert_eq!(rows.len(), 2);
    assert_eq!(rows[0].get("team"), Some(&RuntimeValue::String("a".into())));
    assert_eq!(rows[0].get("a"), Some(&RuntimeValue::Float(35.0)));
    assert_eq!(rows[0].get("c"), Some(&RuntimeValue::Integer(2)));
    assert_eq!(rows[1].get("team"), Some(&RuntimeValue::String("b".into())));
    assert_eq!(
        rows[1].get("a"),
        Some(&RuntimeValue::Float(30.0)),
        "the group's NULL age must not join the divisor"
    );
    assert_eq!(rows[1].get("c"), Some(&RuntimeValue::Integer(1)));
}

#[tokio::test]
async fn aggregate_semantics_on_memtable_and_flushed_routes() {
    let (mut writer, schema) = corpus("agg-semantics").await;
    assert_semantics(&writer).await;

    writer.flush(schema).await.unwrap();
    assert_semantics(&writer).await;
}

/// NDB-10: statistical aggregates — stdev/stdevp (sample vs population
/// denominator), percentileCont (linear interpolation) and percentileDisc
/// (nearest-rank), with the standard NULL-skipping/empty-group envelope.
async fn assert_statistical_semantics(writer: &WriterSession) {
    // Ages present: [30, 40, 30]. Sample stdev vs population stdev.
    let mean = (30.0_f64 + 40.0 + 30.0) / 3.0;
    let ss = (30.0 - mean).powi(2) + (40.0 - mean).powi(2) + (30.0 - mean).powi(2);
    for (q, expected) in [
        (
            "MATCH (p:Person) RETURN stdev(p.age) AS s",
            (ss / 2.0).sqrt(),
        ),
        (
            "MATCH (p:Person) RETURN stdevp(p.age) AS s",
            (ss / 3.0).sqrt(),
        ),
        // DISTINCT: [30, 40] -> mean 35, ss 50, sample denominator 1.
        (
            "MATCH (p:Person) RETURN stdev(DISTINCT p.age) AS s",
            50.0_f64.sqrt(),
        ),
        // Scores sorted [1.5, 2.0, 2.5, 4.0]: rank 0.5*3 = 1.5 -> 2.25.
        (
            "MATCH (p:Person) RETURN percentileCont(p.score, 0.5) AS s",
            2.25,
        ),
    ] {
        match one_value(writer, q).await {
            RuntimeValue::Float(f) => {
                assert!((f - expected).abs() < 1e-9, "{q}: {f} != {expected}")
            }
            other => panic!("{q}: expected float, got {other:?}"),
        }
    }

    // Nearest-rank keeps the input value (and its type): ceil(0.5*4) = rank 2.
    assert_eq!(
        one_value(
            writer,
            "MATCH (p:Person) RETURN percentileDisc(p.score, 0.5) AS s"
        )
        .await,
        RuntimeValue::Float(2.0)
    );
    assert_eq!(
        one_value(
            writer,
            "MATCH (p:Person) RETURN percentileDisc(p.age, 1.0) AS s"
        )
        .await,
        RuntimeValue::Integer(40)
    );
    assert_eq!(
        one_value(
            writer,
            "MATCH (p:Person) RETURN percentileDisc(p.age, 0.0) AS s"
        )
        .await,
        RuntimeValue::Integer(30),
        "p = 0.0 clamps to the minimum"
    );

    // Empty group -> NULL; single value -> stdev 0.0.
    assert_eq!(
        one_value(writer, "MATCH (g:Ghost) RETURN stdev(g.age) AS s").await,
        RuntimeValue::Null
    );
    let snap = writer.snapshot();
    let plan = lower(
        &parse("MATCH (p:Person) RETURN p.team AS team, stdev(p.age) AS s ORDER BY team").unwrap(),
    )
    .unwrap();
    let rows = execute(&plan, &snap, &Params::new()).await.unwrap();
    assert_eq!(rows.len(), 2);
    // Team a: [30, 40] -> sqrt(50); team b: NULL age skipped, single 30 -> 0.
    match rows[0].get("s") {
        Some(RuntimeValue::Float(f)) => assert!((f - 50.0_f64.sqrt()).abs() < 1e-9, "{f}"),
        other => panic!("team a stdev: {other:?}"),
    }
    assert_eq!(rows[1].get("s"), Some(&RuntimeValue::Float(0.0)));

    // Out-of-range percentile and non-numeric input are typed errors.
    let plan =
        lower(&parse("MATCH (p:Person) RETURN percentileCont(p.age, 1.5) AS s").unwrap()).unwrap();
    let error = execute(&plan, &snap, &Params::new()).await.unwrap_err();
    assert!(
        error.to_string().contains("percentile must be between"),
        "{error}"
    );
    let plan = lower(&parse("MATCH (p:Person) RETURN stdev(p.team) AS s").unwrap()).unwrap();
    let error = execute(&plan, &snap, &Params::new()).await.unwrap_err();
    assert!(
        error.to_string().contains("requires numeric values"),
        "{error}"
    );
}

#[tokio::test]
async fn statistical_aggregates_on_memtable_and_flushed_routes() {
    let (mut writer, schema) = corpus("agg-statistical").await;
    assert_statistical_semantics(&writer).await;

    writer.flush(schema).await.unwrap();
    assert_statistical_semantics(&writer).await;
}

#[tokio::test]
async fn integer_sum_overflow_is_a_typed_error_not_a_wrap() {
    let mut writer = WriterSession::open(store(), paths("agg-overflow"))
        .await
        .unwrap();
    for _ in 0..2 {
        writer
            .upsert_node(
                "Person",
                NodeId::new(),
                &record(vec![("age", CoreValue::I64(i64::MAX / 2 + 5))]),
            )
            .unwrap();
    }
    writer.commit_batch().await.unwrap();
    let snap = writer.snapshot();
    let plan = lower(&parse("MATCH (p:Person) RETURN sum(p.age) AS s").unwrap()).unwrap();
    let error = execute(&plan, &snap, &Params::new()).await.unwrap_err();
    assert!(
        error.to_string().contains("overflow"),
        "a wrapped sum would be a plausible wrong number; got {error}"
    );
}
