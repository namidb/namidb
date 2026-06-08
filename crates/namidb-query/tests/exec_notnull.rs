//! Strict NOT NULL enforcement on the write path.
//!
//! A property a label declares `nullable = false` must carry a non-null value
//! on every node bearing that label. The writer rejects the four ways a write
//! could otherwise leave it null: creating a node that omits it or sets it to
//! `NULL`, `SET`ting it to `NULL`, `REMOVE`ing it, and adding the declaring
//! label to a node that lacks it. Mirrors the unique-constraint enforcement;
//! both surface `ExecError::Constraint`. Edges are out of scope (they carry no
//! declared-property validation today).

use std::sync::Arc;

use namidb_core::id::NamespaceId;
use namidb_core::schema::{DataType, LabelDef, PropertyDef, Schema, SchemaBuilder};
use namidb_storage::{NamespacePaths, WriterSession};
use object_store::memory::InMemory;
use object_store::ObjectStore;

use namidb_query::{execute_write, lower, parse, ExecError, Params};

fn store() -> Arc<dyn ObjectStore> {
    Arc::new(InMemory::new())
}

fn paths(name: &str) -> NamespacePaths {
    NamespacePaths::new("tenants", NamespaceId::new(name).unwrap())
}

/// `Person.name` is NOT NULL; `Person.nick` is nullable.
fn person_schema() -> Schema {
    SchemaBuilder::new()
        .label(LabelDef {
            name: "Person".into(),
            properties: vec![
                PropertyDef::new("name", DataType::Utf8, false).unwrap(),
                PropertyDef::new("nick", DataType::Utf8, true).unwrap(),
            ],
        })
        .unwrap()
        .build()
}

async fn write_q(writer: &mut WriterSession, text: &str) -> namidb_query::WriteOutcome {
    let plan = lower(&parse(text).unwrap()).unwrap();
    execute_write(&plan, writer, &Params::new()).await.unwrap()
}

async fn write_err(writer: &mut WriterSession, text: &str) -> ExecError {
    let plan = lower(&parse(text).unwrap()).unwrap();
    execute_write(&plan, writer, &Params::new())
        .await
        .expect_err("expected the write to be rejected")
}

/// Open a writer with [`person_schema`] persisted into the manifest, so
/// `writer.schema()` sees it and enforcement is live. A flush needs a
/// non-empty memtable, so we seed one node under an unrelated, undeclared
/// label (`Seed`) that the `Person` assertions ignore.
async fn open_enforcing(ns: &str) -> WriterSession {
    let mut writer = WriterSession::open(store(), paths(ns)).await.unwrap();
    write_q(&mut writer, "CREATE (:Seed {x: 1})").await;
    writer.flush(person_schema()).await.unwrap();
    writer
}

fn is_notnull(err: &ExecError) -> bool {
    matches!(err, ExecError::Constraint(msg) if msg.contains("not-null"))
}

async fn person_count(writer: &WriterSession) -> usize {
    writer.snapshot().scan_label("Person").await.unwrap().len()
}

#[tokio::test]
async fn create_rejects_missing_required_property() {
    let mut writer = open_enforcing("nn-create-missing").await;
    let err = write_err(&mut writer, "CREATE (:Person {nick: 'ada'})").await;
    assert!(
        is_notnull(&err),
        "expected not-null constraint, got {err:?}"
    );
    assert_eq!(person_count(&writer).await, 0);
}

#[tokio::test]
async fn create_rejects_explicit_null_required_property() {
    let mut writer = open_enforcing("nn-create-null").await;
    let err = write_err(&mut writer, "CREATE (:Person {name: null})").await;
    assert!(
        is_notnull(&err),
        "expected not-null constraint, got {err:?}"
    );
    assert_eq!(person_count(&writer).await, 0);
}

#[tokio::test]
async fn create_allows_required_property_with_value() {
    let mut writer = open_enforcing("nn-create-ok").await;
    // `nick` (nullable) may be omitted entirely.
    let outcome = write_q(&mut writer, "CREATE (:Person {name: 'Ada'})").await;
    assert_eq!(outcome.nodes_created, 1);
    assert_eq!(person_count(&writer).await, 1);
}

#[tokio::test]
async fn set_rejects_null_on_required_property() {
    let mut writer = open_enforcing("nn-set-null").await;
    write_q(&mut writer, "CREATE (:Person {name: 'Ada'})").await;
    let err = write_err(&mut writer, "MATCH (p:Person) SET p.name = null").await;
    assert!(
        is_notnull(&err),
        "expected not-null constraint, got {err:?}"
    );
    // The node keeps its original name; the rejected SET committed nothing.
    let people = writer.snapshot().scan_label("Person").await.unwrap();
    assert_eq!(people.len(), 1);
    assert_eq!(
        people[0].properties.get("name"),
        Some(&namidb_core::value::Value::Str("Ada".into()))
    );
}

#[tokio::test]
async fn set_allows_value_on_required_and_null_on_nullable() {
    let mut writer = open_enforcing("nn-set-ok").await;
    write_q(&mut writer, "CREATE (:Person {name: 'Ada', nick: 'a'})").await;
    // A real value on the required property is fine.
    write_q(&mut writer, "MATCH (p:Person) SET p.name = 'Grace'").await;
    // NULL on a nullable property is fine.
    write_q(&mut writer, "MATCH (p:Person) SET p.nick = null").await;
    let people = writer.snapshot().scan_label("Person").await.unwrap();
    assert_eq!(
        people[0].properties.get("name"),
        Some(&namidb_core::value::Value::Str("Grace".into()))
    );
}

#[tokio::test]
async fn remove_rejects_required_property() {
    let mut writer = open_enforcing("nn-remove-required").await;
    write_q(&mut writer, "CREATE (:Person {name: 'Ada'})").await;
    let err = write_err(&mut writer, "MATCH (p:Person) REMOVE p.name").await;
    assert!(
        is_notnull(&err),
        "expected not-null constraint, got {err:?}"
    );
    let people = writer.snapshot().scan_label("Person").await.unwrap();
    assert_eq!(
        people[0].properties.get("name"),
        Some(&namidb_core::value::Value::Str("Ada".into()))
    );
}

#[tokio::test]
async fn remove_allows_nullable_property() {
    let mut writer = open_enforcing("nn-remove-nullable").await;
    write_q(&mut writer, "CREATE (:Person {name: 'Ada', nick: 'a'})").await;
    write_q(&mut writer, "MATCH (p:Person) REMOVE p.nick").await;
    let people = writer.snapshot().scan_label("Person").await.unwrap();
    assert_eq!(people[0].properties.get("nick"), None);
    // The required property is untouched.
    assert_eq!(
        people[0].properties.get("name"),
        Some(&namidb_core::value::Value::Str("Ada".into()))
    );
}

#[tokio::test]
async fn add_label_rejects_when_required_property_missing() {
    let mut writer = open_enforcing("nn-addlabel-missing").await;
    // A bare node with no `name`, created under an undeclared label.
    write_q(&mut writer, "CREATE (:Ghost {x: 1})").await;
    // Promoting it to :Person would leave Person.name null.
    let err = write_err(&mut writer, "MATCH (g:Ghost) SET g:Person").await;
    assert!(
        is_notnull(&err),
        "expected not-null constraint, got {err:?}"
    );
    assert_eq!(person_count(&writer).await, 0);
}

#[tokio::test]
async fn add_label_allows_when_required_property_present() {
    let mut writer = open_enforcing("nn-addlabel-ok").await;
    write_q(&mut writer, "CREATE (:Ghost {name: 'Ada'})").await;
    write_q(&mut writer, "MATCH (g:Ghost) SET g:Person").await;
    assert_eq!(person_count(&writer).await, 1);
}

#[tokio::test]
async fn merge_create_branch_enforces_required_property() {
    let mut writer = open_enforcing("nn-merge").await;
    // No Person matches, so MERGE takes the create branch — which must still
    // reject a node missing the required property.
    let err = write_err(&mut writer, "MERGE (:Person {nick: 'ada'})").await;
    assert!(
        is_notnull(&err),
        "expected not-null constraint, got {err:?}"
    );
    assert_eq!(person_count(&writer).await, 0);
}

#[tokio::test]
async fn undeclared_label_is_unconstrained() {
    let mut writer = open_enforcing("nn-undeclared").await;
    // `Robot` is not in the schema, so it has no NOT NULL contract.
    let outcome = write_q(&mut writer, "CREATE (:Robot {serial: 7})").await;
    assert_eq!(outcome.nodes_created, 1);
}
