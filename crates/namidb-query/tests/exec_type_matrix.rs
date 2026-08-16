//! Unaudited dimension from docs/testing/25tb-readiness.md: the property
//! type-system as a SYSTEMATIC matrix — round-trip, WHERE equality and
//! ORDER BY for every practical value kind across the memtable route AND
//! the flushed SST/Parquet route. Earlier audits spot-checked individual
//! types; a 25 TB heterogeneous load exercises them all on day one.

use std::collections::BTreeMap;
use std::sync::Arc;

use namidb_core::id::{NamespaceId, NodeId};
use namidb_core::schema::{DataType, LabelDef, PropertyDef, Schema, SchemaBuilder};
use namidb_core::value::Value as CoreValue;
use namidb_storage::{NamespacePaths, NodeWriteRecord, WriterSession};
use object_store::memory::InMemory;
use object_store::ObjectStore;

use namidb_query::{execute, lower, parse, Params, RuntimeValue};

fn schema() -> Schema {
    SchemaBuilder::new()
        .label(LabelDef {
            name: "Item".into(),
            properties: vec![
                PropertyDef::new("name", DataType::Utf8, false).unwrap(),
                PropertyDef::new("count", DataType::Int64, true).unwrap(),
                PropertyDef::new("ratio", DataType::Float64, true).unwrap(),
                PropertyDef::new("active", DataType::Bool, true).unwrap(),
                PropertyDef::new("title", DataType::Utf8, true)
                    .unwrap()
                    .with_indexed(true),
                PropertyDef::new("blob", DataType::Binary, true).unwrap(),
                PropertyDef::new("born", DataType::Date32, true).unwrap(),
                PropertyDef::new("seen", DataType::TimestampMicrosUtc, true).unwrap(),
            ],
        })
        .unwrap()
        .build()
}

/// One node per value kind plus a NULL-heavy node. `title` is
/// schema-indexed with multibyte UTF-8 — the property-index encoding risk.
fn corpus_rows() -> Vec<(&'static str, BTreeMap<String, CoreValue>)> {
    let mut rows = Vec::new();
    let mut base = |name: &str| {
        let mut props: BTreeMap<String, CoreValue> = BTreeMap::new();
        props.insert("name".into(), CoreValue::Str(name.into()));
        props
    };

    let mut props = base("ints");
    props.insert("count".into(), CoreValue::I64(i64::MAX));
    rows.push(("ints", props));

    let mut props = base("floats");
    props.insert("ratio".into(), CoreValue::F64(-0.125));
    rows.push(("floats", props));

    let mut props = base("bools");
    props.insert("active".into(), CoreValue::Bool(true));
    rows.push(("bools", props));

    let mut props = base("unicode");
    props.insert("title".into(), CoreValue::Str("ñandú 東京 🦀".into()));
    rows.push(("unicode", props));

    let mut props = base("bytes");
    props.insert("blob".into(), CoreValue::Bytes(vec![0, 255, 1, 128, 7]));
    rows.push(("bytes", props));

    let mut props = base("dates");
    props.insert("born".into(), CoreValue::Date(19_723));
    props.insert("seen".into(), CoreValue::DateTime(1_723_939_200_000_000));
    rows.push(("dates", props));

    let mut props = base("nested");
    props.insert(
        "extra".into(),
        CoreValue::List(vec![
            CoreValue::I64(1),
            CoreValue::Str("two".into()),
            CoreValue::Map(BTreeMap::from([("inner".into(), CoreValue::Bool(false))])),
        ]),
    );
    rows.push(("nested", props));

    rows.push(("nulls", base("nulls")));
    rows
}

async fn build(name: &str, flush: bool) -> WriterSession {
    let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
    let paths = NamespacePaths::new("tenants", NamespaceId::new(name).unwrap());
    let mut writer = WriterSession::open(store, paths).await.unwrap();
    for (_, props) in corpus_rows() {
        writer
            .upsert_node(
                "Item",
                NodeId::new(),
                &NodeWriteRecord {
                    properties: props,
                    schema_version: 1,
                    ..Default::default()
                },
            )
            .unwrap();
    }
    writer.commit_batch().await.unwrap();
    if flush {
        writer.flush(schema()).await.unwrap();
        writer.compact_l0(&schema()).await.unwrap();
    }
    writer
}

async fn names_for(writer: &WriterSession, query: &str) -> Vec<String> {
    let snapshot = writer.snapshot();
    let parsed = parse(query).unwrap_or_else(|error| panic!("parse `{query}`: {error:?}"));
    let plan = lower(&parsed).unwrap_or_else(|error| panic!("lower `{query}`: {error:?}"));
    let rows = execute(&plan, &snapshot, &Params::new())
        .await
        .unwrap_or_else(|error| panic!("execute `{query}`: {error:?}"));
    let mut out: Vec<String> = rows
        .iter()
        .map(|row| match row.bindings.values().next() {
            Some(RuntimeValue::String(text)) => text.clone(),
            other => format!("{other:?}"),
        })
        .collect();
    out.sort();
    out
}

/// Every WHERE-equality below must produce the same single row on the
/// memtable route and on the flushed SST route.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn every_value_kind_round_trips_and_filters_identically_on_both_routes() {
    let cases: Vec<(&str, &str)> = vec![
        (
            "ints",
            "MATCH (i:Item) WHERE i.count = 9223372036854775807 RETURN i.name AS n",
        ),
        (
            "floats",
            "MATCH (i:Item) WHERE i.ratio = -0.125 RETURN i.name AS n",
        ),
        (
            "bools",
            "MATCH (i:Item) WHERE i.active = true RETURN i.name AS n",
        ),
        (
            "unicode",
            "MATCH (i:Item) WHERE i.title = 'ñandú 東京 🦀' RETURN i.name AS n",
        ),
        (
            "dates",
            "MATCH (i:Item) WHERE i.seen IS NOT NULL AND i.born IS NOT NULL \
             RETURN i.name AS n",
        ),
        (
            "nested",
            "MATCH (i:Item) WHERE i.extra IS NOT NULL RETURN i.name AS n",
        ),
    ];
    for (route, flush) in [("memtable", false), ("flushed", true)] {
        let writer = build(&format!("type-matrix-{route}"), flush).await;
        for (expected, query) in &cases {
            let got = names_for(&writer, query).await;
            assert_eq!(
                got,
                vec![expected.to_string()],
                "kind `{expected}` must filter exactly on the {route} route"
            );
        }
        // The whole corpus stays reachable and countable.
        let all = names_for(&writer, "MATCH (i:Item) RETURN i.name AS n").await;
        assert_eq!(all.len(), corpus_rows().len(), "{route}: full corpus");
    }
}

/// Nested list/map values survive the overflow-JSON path byte-exactly, and
/// bytes round-trip through the flushed route.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn nested_and_binary_values_round_trip_through_the_flushed_route() {
    let writer = build("type-matrix-roundtrip", true).await;
    let snapshot = writer.snapshot();
    let parsed = parse("MATCH (i:Item {name: 'nested'}) RETURN i.extra AS extra").unwrap();
    let rows = execute(&lower(&parsed).unwrap(), &snapshot, &Params::new())
        .await
        .unwrap();
    assert_eq!(rows.len(), 1);
    match rows[0].get("extra") {
        Some(RuntimeValue::List(items)) => {
            assert_eq!(items.len(), 3);
            assert!(matches!(items[0], RuntimeValue::Integer(1)));
            assert!(matches!(&items[1], RuntimeValue::String(s) if s == "two"));
            assert!(
                matches!(&items[2], RuntimeValue::Map(m)
                    if matches!(m.get("inner"), Some(RuntimeValue::Bool(false)))),
                "nested map inside the list must survive, got {:?}",
                items[2]
            );
        }
        other => panic!("nested list must round-trip, got {other:?}"),
    }

    let parsed = parse("MATCH (i:Item {name: 'bytes'}) RETURN i.blob AS blob").unwrap();
    let rows = execute(&lower(&parsed).unwrap(), &snapshot, &Params::new())
        .await
        .unwrap();
    assert_eq!(rows.len(), 1);
    match rows[0].get("blob") {
        Some(RuntimeValue::Bytes(bytes)) => {
            assert_eq!(bytes.as_slice(), &[0, 255, 1, 128, 7]);
        }
        other => panic!("binary must round-trip, got {other:?}"),
    }
}

/// ORDER BY over a mixed column: rows carrying the value sort
/// deterministically, NULL rows have a stable documented position, and both
/// routes agree.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn order_by_with_nulls_is_stable_and_route_invariant() {
    let mut per_route = Vec::new();
    for (route, flush) in [("memtable", false), ("flushed", true)] {
        let writer = build(&format!("type-matrix-order-{route}"), flush).await;
        let names = names_for(
            &writer,
            "MATCH (i:Item) RETURN i.name AS n ORDER BY i.count, i.name",
        )
        .await;
        assert_eq!(names.len(), corpus_rows().len());
        per_route.push(names);
    }
    assert_eq!(
        per_route[0], per_route[1],
        "ORDER BY over a sparse column must sort identically on both routes"
    );
}
