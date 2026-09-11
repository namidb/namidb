//! Item 80: a range predicate on an indexed numeric property must be served
//! from the equality sidecar, and must agree EXACTLY with the flat scan.
//!
//! Measured before this route existed, on 160,000 rows with `idx` indexed:
//! `WHERE t.idx > 999999` (matching nothing) cost 0.311 s, and
//! `WHERE t.idx > 80000` (matching half) cost 0.340 s — completely
//! insensitive to selectivity, because nothing in the node layout carries
//! property statistics to prune on.
//!
//! The route is only sound because the numeric key is order-preserving, so a
//! key window IS a value window. Three things make its candidate set a
//! SUPERSET of the answer rather than a subset, and each is removed by the
//! confirm step:
//!
//!   * the key is lossy above 2^53, so distinct integers share a posting;
//!   * a posting can name a node whose current version no longer matches;
//!   * a raw String key can fall inside the numeric byte window.
//!
//! A subset would be a silent wrong answer, so every case here is checked
//! against a flat scan filtered with the SAME evaluator the scan route uses.

use std::collections::BTreeMap;
use std::sync::Arc;

use namidb_core::id::{NamespaceId, NodeId};
use namidb_core::value::Value;
use namidb_storage::sst::{eval_against_value, ScanPredicate, StatScalar};
use namidb_storage::{NamespacePaths, NodeWriteRecord, WriterSession};
use object_store::memory::InMemory;
use object_store::ObjectStore;

const ROWS: i64 = 2_000;

fn node(props: Vec<(&str, Value)>) -> NodeWriteRecord {
    NodeWriteRecord {
        properties: props
            .into_iter()
            .map(|(k, v)| (k.to_string(), v))
            .collect::<BTreeMap<_, _>>(),
        schema_version: 1,
        ..Default::default()
    }
}

/// `idx` is Int64 and indexed. The corpus also carries rows that are NOT
/// plain integers on that property, because those are exactly what a
/// key-window walk can pick up by accident:
///
///   * floats, including one equal to an integer already present;
///   * a legacy String value on the numeric property;
///   * a NULL and an absent property;
///   * two i64 beyond 2^53 that share one key.
async fn corpus(name: &str, flush: bool, compact: bool) -> WriterSession {
    let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
    let paths = NamespacePaths::new("tenants", NamespaceId::new(name).unwrap());
    let mut writer = WriterSession::open(store, paths).await.unwrap();

    for ordinal in 0..ROWS {
        writer
            .upsert_node(
                "T",
                NodeId::new(),
                &node(vec![
                    ("idx", Value::I64(ordinal)),
                    ("tag", Value::Str(format!("row-{ordinal}"))),
                ]),
            )
            .unwrap();
    }
    for (value, tag) in [
        (Value::F64(10.5), "fractional"),
        (Value::F64(20.0), "integral-float"),
        (Value::F64(-3.25), "negative-float"),
        (Value::I64(-7), "negative-int"),
        (Value::Str("500".into()), "string-on-numeric-prop"),
        (Value::Str("n:0000000000000000".into()), "string-shaped-key"),
        (Value::Null, "null"),
        (Value::I64(9_007_199_254_740_992), "big-low"),
        (Value::I64(9_007_199_254_740_993), "big-high"),
    ] {
        writer
            .upsert_node(
                "T",
                NodeId::new(),
                &node(vec![("idx", value), ("tag", Value::Str(tag.into()))]),
            )
            .unwrap();
    }
    // No `idx` at all.
    writer
        .upsert_node(
            "T",
            NodeId::new(),
            &node(vec![("tag", Value::Str("absent".into()))]),
        )
        .unwrap();
    writer.commit_batch().await.unwrap();
    writer.create_property_index("T", "idx").await.unwrap();

    if flush {
        let schema = writer.snapshot().manifest().manifest.schema.clone();
        writer.flush(schema.clone()).await.unwrap();
        if compact {
            writer.compact_l0(&schema).await.unwrap();
        }
    }
    writer
}

fn gt(v: i64) -> ScanPredicate {
    ScanPredicate::Gt {
        column: "idx".into(),
        value: StatScalar::Int64(v),
    }
}
fn gte(v: i64) -> ScanPredicate {
    ScanPredicate::GtEq {
        column: "idx".into(),
        value: StatScalar::Int64(v),
    }
}
fn lt(v: i64) -> ScanPredicate {
    ScanPredicate::Lt {
        column: "idx".into(),
        value: StatScalar::Int64(v),
    }
}
fn lte(v: i64) -> ScanPredicate {
    ScanPredicate::LtEq {
        column: "idx".into(),
        value: StatScalar::Int64(v),
    }
}
fn gt_f(v: f64) -> ScanPredicate {
    ScanPredicate::Gt {
        column: "idx".into(),
        value: StatScalar::Float64(v),
    }
}
fn lt_f(v: f64) -> ScanPredicate {
    ScanPredicate::Lt {
        column: "idx".into(),
        value: StatScalar::Float64(v),
    }
}

fn families() -> Vec<(&'static str, Vec<ScanPredicate>)> {
    vec![
        ("> 1990 (a narrow tail)", vec![gt(1_990)]),
        (">= 1990", vec![gte(1_990)]),
        ("< 5 (a narrow head)", vec![lt(5)]),
        ("<= 5", vec![lte(5)]),
        ("> 10 AND < 25 (spans the floats)", vec![gt(10), lt(25)]),
        (
            ">= 20 AND <= 20 (the integral float)",
            vec![gte(20), lte(20)],
        ),
        ("> -10 AND < 1 (spans the negatives)", vec![gt(-10), lt(1)]),
        ("< -100 (matches nothing)", vec![lt(-100)]),
        ("> 100000 (above every plain row)", vec![gt(100_000)]),
        (
            "> 2^53 (the two colliding integers)",
            vec![gt(9_007_199_254_740_992)],
        ),
        (
            ">= 2^53 (both colliding integers)",
            vec![gte(9_007_199_254_740_992)],
        ),
        (
            "> 10.4 AND < 10.6 (float bounds)",
            vec![gt_f(10.4), lt_f(10.6)],
        ),
        ("> 1999 AND < 1991 (inverted)", vec![gt(1_999), lt(1_991)]),
    ]
}

/// What the scan route returns: every live node of the label whose current
/// `idx` satisfies every predicate, by the same evaluator.
async fn by_scan(writer: &WriterSession, predicates: &[ScanPredicate]) -> Vec<NodeId> {
    let snapshot = writer.snapshot();
    let mut ids: Vec<NodeId> = snapshot
        .scan_label("T")
        .await
        .unwrap()
        .into_iter()
        .filter(|view| {
            let value = view.properties.get("idx");
            predicates
                .iter()
                .all(|predicate| eval_against_value(predicate, value))
        })
        .map(|view| view.id)
        .collect();
    ids.sort_unstable();
    ids
}

#[tokio::test]
async fn numeric_range_lookup_agrees_with_the_scan() {
    for (flush, compact, stage) in [
        (false, false, "memtable"),
        (true, false, "flushed"),
        (true, true, "compacted"),
    ] {
        let writer = corpus(&format!("range-{stage}"), flush, compact).await;
        let snapshot = writer.snapshot();
        let mut served = 0_usize;

        for (name, predicates) in families() {
            let expected = by_scan(&writer, &predicates).await;
            let got = snapshot
                .indexed_node_ids_by_numeric_range("T", "idx", &predicates, 4_096)
                .await
                .unwrap();
            let Some(mut got) = got else {
                // Declining is always allowed — the caller scans, which is
                // correct. What is never allowed is answering WRONG.
                continue;
            };
            served += 1;
            got.sort_unstable();
            assert_eq!(
                got, expected,
                "[{stage}] {name}: the index route disagreed with the scan"
            );
        }

        // …and it must actually serve something, or the agreement above is
        // the trivial kind: every family declined.
        if flush {
            assert!(
                served >= families().len() - 2,
                "[{stage}] only {served} of {} families took the index route",
                families().len()
            );
        }
    }
}

#[tokio::test]
async fn an_unselective_range_declines_rather_than_paying_twice() {
    let writer = corpus("range-cap", true, true).await;
    let snapshot = writer.snapshot();

    // Matching (almost) the whole label. Reading every posting AND hydrating
    // every row costs strictly more than the scan that reads each row once,
    // so this must decline — and it must decline on EITHER bound.
    let wide = vec![gt(-1_000_000)];
    assert!(
        snapshot
            .indexed_node_ids_by_numeric_range("T", "idx", &wide, 16)
            .await
            .unwrap()
            .is_none(),
        "a window over the posting cap must decline"
    );
    assert!(
        snapshot
            .indexed_node_ids_by_numeric_range("T", "idx", &wide, 1 << 20)
            .await
            .unwrap()
            .is_none(),
        "and still decline with a generous posting cap, because the walk \
         itself spans more leaf pages than the route will read — that page \
         bound is what makes declining cheap, and it is independent of how \
         many ids the caller was willing to take"
    );

    // A window inside BOTH bounds serves, and still has to agree with the
    // scan.
    let narrow = vec![gt(1_900), lt(1_990)];
    let expected = by_scan(&writer, &narrow).await;
    let mut got = snapshot
        .indexed_node_ids_by_numeric_range("T", "idx", &narrow, 1 << 20)
        .await
        .unwrap()
        .expect("a window inside both bounds must serve");
    got.sort_unstable();
    assert_eq!(got, expected);
    assert!(!expected.is_empty(), "the window must not be empty");
}

/// The hole the flushed-corpus cap test did not cover: BEFORE any flush
/// there are no SST descriptors, so a budget derived only from them has no
/// information at all.
///
/// That is not a rare state — it is every store between a bulk load and its
/// first flush, which is exactly when someone runs their first query. A
/// window covering half a memtable-resident label was being served and
/// hydrated row by row, at twice the cost of the scan it was meant to beat
/// (measured: 130.7 ms against 69.0 ms on 60,000 rows).
#[tokio::test]
async fn a_wide_window_over_a_memtable_resident_label_declines_too() {
    let writer = corpus("range-memtable", false, false).await;
    let snapshot = writer.snapshot();

    // Half the label, with no SSTs in existence.
    assert!(
        snapshot.manifest().manifest.ssts.is_empty(),
        "this test is only meaningful before a flush"
    );
    let wide = vec![gt(ROWS / 2)];
    assert!(
        snapshot
            .indexed_node_ids_by_numeric_range("T", "idx", &wide, usize::MAX)
            .await
            .unwrap()
            .is_none(),
        "a window over half a memtable-resident label must decline, whatever \
         cap the caller offers: scanning a memtable is cheap, and hydrating \
         every candidate the postings named is not"
    );

    // A narrow one is still served, and still agrees.
    let narrow = vec![gt(ROWS - 10)];
    let expected = by_scan(&writer, &narrow).await;
    let mut got = snapshot
        .indexed_node_ids_by_numeric_range("T", "idx", &narrow, usize::MAX)
        .await
        .unwrap()
        .expect("a narrow window must still be served from the memtable");
    got.sort_unstable();
    assert_eq!(got, expected);
    assert!(!expected.is_empty());
}

#[tokio::test]
async fn shapes_this_route_must_refuse() {
    let writer = corpus("range-refuse", true, true).await;
    let snapshot = writer.snapshot();

    let refused: Vec<(&str, Vec<ScanPredicate>)> = vec![
        (
            "a predicate on another property",
            vec![ScanPredicate::Gt {
                column: "tag".into(),
                value: StatScalar::Int64(1),
            }],
        ),
        (
            "IS NULL — the sidecar files no key for an absent value",
            vec![ScanPredicate::IsNull {
                column: "idx".into(),
            }],
        ),
        (
            "IS NOT NULL — unbounded, and the scan reads the same rows once",
            vec![ScanPredicate::IsNotNull {
                column: "idx".into(),
            }],
        ),
        (
            "a String bound is not a numeric window",
            vec![ScanPredicate::Gt {
                column: "idx".into(),
                value: StatScalar::Utf8("5".into()),
            }],
        ),
        ("no predicates at all", vec![]),
    ];
    for (name, predicates) in refused {
        assert!(
            snapshot
                .indexed_node_ids_by_numeric_range("T", "idx", &predicates, 4_096)
                .await
                .unwrap()
                .is_none(),
            "{name}: this route must decline, not guess"
        );
    }

    // An UNINDEXED property has no sidecar to read.
    assert!(
        snapshot
            .indexed_node_ids_by_numeric_range("T", "tag", &[gt(1)], 4_096)
            .await
            .unwrap()
            .is_none(),
        "an unindexed property must decline"
    );
}
