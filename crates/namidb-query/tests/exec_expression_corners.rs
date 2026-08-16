//! Plan items 26 and 27 (docs/testing/25tb-readiness.md): the expression
//! evaluator's dark corners — STARTS WITH / ENDS WITH / CONTAINS, both CASE
//! forms, missing-ELSE — and top-level UNION / UNION ALL, including the
//! column-name compatibility contract across branches.

use std::collections::BTreeMap;
use std::sync::Arc;

use namidb_core::id::{NamespaceId, NodeId};
use namidb_core::value::Value as CoreValue;
use namidb_storage::{NamespacePaths, NodeWriteRecord, WriterSession};
use object_store::memory::InMemory;
use object_store::ObjectStore;

use namidb_query::{execute, lower, parse, Params, RuntimeValue};

async fn corpus(name: &str) -> WriterSession {
    let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
    let paths = NamespacePaths::new("tenants", NamespaceId::new(name).unwrap());
    let mut writer = WriterSession::open(store, paths).await.unwrap();
    let people = [
        ("Alice", Some("Madrid")),
        ("alastair", Some("Berlin")),
        ("Bob", None),
    ];
    for (name, city) in people {
        let mut props: BTreeMap<String, CoreValue> = BTreeMap::new();
        props.insert("name".into(), CoreValue::Str(name.into()));
        if let Some(city) = city {
            props.insert("city".into(), CoreValue::Str(city.into()));
        }
        writer
            .upsert_node(
                "Person",
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
    writer
}

async fn names(writer: &WriterSession, query: &str) -> Vec<String> {
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
            Some(other) => format!("{other:?}"),
            None => "<no column>".to_string(),
        })
        .collect();
    out.sort();
    out
}

#[tokio::test]
async fn string_predicates_match_case_sensitively_and_null_out_gracefully() {
    let writer = corpus("expr-string-preds").await;

    // Case-sensitive prefix: 'Al' matches Alice but NOT alastair.
    assert_eq!(
        names(
            &writer,
            "MATCH (p:Person) WHERE p.name STARTS WITH 'Al' RETURN p.name AS n"
        )
        .await,
        vec!["Alice"]
    );
    assert_eq!(
        names(
            &writer,
            "MATCH (p:Person) WHERE p.name ENDS WITH 'ce' RETURN p.name AS n"
        )
        .await,
        vec!["Alice"]
    );
    assert_eq!(
        names(
            &writer,
            "MATCH (p:Person) WHERE p.name CONTAINS 'asta' RETURN p.name AS n"
        )
        .await,
        vec!["alastair"]
    );
    // NULL operand (Bob has no city): the predicate evaluates to NULL, the
    // row is filtered — never an error, never a match.
    assert_eq!(
        names(
            &writer,
            "MATCH (p:Person) WHERE p.city STARTS WITH 'M' RETURN p.name AS n"
        )
        .await,
        vec!["Alice"]
    );
    // Non-string operand → NULL → filtered, for all three operators.
    for operator in ["STARTS WITH", "ENDS WITH", "CONTAINS"] {
        let rows = names(
            &writer,
            &format!("MATCH (p:Person) WHERE 42 {operator} '4' RETURN p.name AS n"),
        )
        .await;
        assert!(
            rows.is_empty(),
            "non-string {operator} must be NULL-filtered, got {rows:?}"
        );
    }
}

#[tokio::test]
async fn both_case_forms_evaluate_and_missing_else_is_null() {
    let writer = corpus("expr-case-forms").await;

    // Searched CASE.
    let rows = names(
        &writer,
        "MATCH (p:Person) RETURN CASE WHEN p.name STARTS WITH 'A' THEN 'upper' \
         WHEN p.name STARTS WITH 'a' THEN 'lower' ELSE 'other' END AS bucket",
    )
    .await;
    assert_eq!(rows, vec!["lower", "other", "upper"]);

    // Simple (scrutinee) CASE.
    let rows = names(
        &writer,
        "MATCH (p:Person) RETURN CASE p.name WHEN 'Bob' THEN 'bob' \
         WHEN 'Alice' THEN 'alice' ELSE 'other' END AS bucket",
    )
    .await;
    assert_eq!(rows, vec!["alice", "bob", "other"]);

    // Missing ELSE → NULL.
    let rows = names(
        &writer,
        "MATCH (p:Person) RETURN CASE p.name WHEN 'Bob' THEN 'bob' END AS bucket",
    )
    .await;
    assert_eq!(rows, vec!["Null", "Null", "bob"]);
}

#[tokio::test]
async fn union_dedupes_and_union_all_keeps_duplicates() {
    let writer = corpus("expr-union").await;

    let rows = names(
        &writer,
        "MATCH (p:Person) WHERE p.name = 'Bob' RETURN p.name AS n \
         UNION MATCH (p:Person) WHERE p.name = 'Bob' RETURN p.name AS n",
    )
    .await;
    assert_eq!(rows, vec!["Bob"], "UNION must dedupe identical rows");

    let rows = names(
        &writer,
        "MATCH (p:Person) WHERE p.name = 'Bob' RETURN p.name AS n \
         UNION ALL MATCH (p:Person) WHERE p.name = 'Bob' RETURN p.name AS n",
    )
    .await;
    assert_eq!(
        rows,
        vec!["Bob", "Bob"],
        "UNION ALL must keep both branch rows"
    );
}

#[tokio::test]
async fn union_rejects_mismatched_column_names() {
    for query in [
        // Different alias.
        "RETURN 1 AS a UNION RETURN 1 AS b",
        // Same names, different order.
        "RETURN 1 AS a, 2 AS b UNION RETURN 2 AS b, 1 AS a",
        // Different arity.
        "RETURN 1 AS a UNION ALL RETURN 1 AS a, 2 AS b",
    ] {
        let parsed = parse(query).unwrap();
        let error = lower(&parsed).expect_err("mismatched UNION columns must be rejected");
        let message = format!("{error:?}");
        assert!(
            message.contains("same column names"),
            "error must explain the column contract for `{query}`, got: {message}"
        );
    }
    // Matching names and order across branches stay valid.
    let parsed = parse("RETURN 1 AS a, 2 AS b UNION RETURN 3 AS a, 4 AS b").unwrap();
    lower(&parsed).expect("matching UNION columns must lower");
}
