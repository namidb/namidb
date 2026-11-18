//! `count(n)` and `count(n) WHERE n.p IS NOT NULL` over a label are answered
//! from the node-count ledger. They must equal the live set after every kind
//! of write that changes them (create, `SET n.p`, `REMOVE n.p`, `SET n.p =
//! null`, `DETACH DELETE`), across commits, flushes and a reopen.

use std::sync::Arc;

use namidb_core::id::NamespaceId;
use namidb_core::schema::{DataType, LabelDef, PropertyDef, Schema, SchemaBuilder};
use namidb_storage::{NamespacePaths, WriterSession};
use object_store::memory::InMemory;
use object_store::ObjectStore;

use namidb_query::{execute, execute_write, lower, parse, Params, RuntimeValue};

fn schema() -> Schema {
    SchemaBuilder::new()
        .label(LabelDef {
            name: "Articulo".into(),
            properties: vec![
                PropertyDef::new("n", DataType::Int64, true).unwrap(),
                PropertyDef::new("texto", DataType::Utf8, true).unwrap(),
                PropertyDef::new("embedding", DataType::FloatVector { dim: 3 }, true).unwrap(),
            ],
        })
        .unwrap()
        .build()
}

async fn write(writer: &mut WriterSession, text: &str) {
    let plan = lower(&parse(text).unwrap()).unwrap();
    execute_write(&plan, writer, &Params::new()).await.unwrap();
}

async fn count(writer: &WriterSession, text: &str) -> i64 {
    let snapshot = writer.snapshot();
    let plan = lower(&parse(text).unwrap()).unwrap();
    let rows = execute(&plan, &snapshot, &Params::new()).await.unwrap();
    match rows[0].bindings.values().next() {
        Some(RuntimeValue::Integer(value)) => *value,
        other => panic!("`{text}` returned {other:?}"),
    }
}

const ALL: &str = "MATCH (a:Articulo) RETURN count(a)";
const EMBEDDED: &str = "MATCH (a:Articulo) WHERE a.embedding IS NOT NULL RETURN count(a)";
const EMBEDDED_PROPERTY: &str = "MATCH (a:Articulo) RETURN count(a.embedding)";

async fn assert_counts(writer: &WriterSession, total: i64, embedded: i64, at: &str) {
    assert_eq!(count(writer, ALL).await, total, "{at}: count(a)");
    // Twice: the second answer comes from the seeded ledger.
    for _ in 0..2 {
        assert_eq!(count(writer, EMBEDDED).await, embedded, "{at}: IS NOT NULL");
        assert_eq!(
            count(writer, EMBEDDED_PROPERTY).await,
            embedded,
            "{at}: count(a.embedding)"
        );
    }
}

#[tokio::test]
async fn presence_counts_follow_embedding_batches_removals_and_deletes() {
    let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
    let paths = NamespacePaths::new("tenants", NamespaceId::new("meta-count").unwrap());
    let mut writer = WriterSession::open(store.clone(), paths.clone())
        .await
        .unwrap();
    write(
        &mut writer,
        "UNWIND range(0, 599) AS i CREATE (:Articulo {n: i, texto: 'art ' + toString(i)})",
    )
    .await;
    writer.flush(schema()).await.unwrap();
    assert_counts(&writer, 600, 0, "loaded").await;

    // The 300-row embedding batch shape, in two batches with a flush between.
    write(
        &mut writer,
        "UNWIND range(0, 299) AS i MATCH (a:Articulo {n: i}) SET a.embedding = [1.0, 0.0, toFloat(i)]",
    )
    .await;
    assert_counts(&writer, 600, 300, "first batch in the memtable").await;
    writer.flush(schema()).await.unwrap();
    write(
        &mut writer,
        "UNWIND range(300, 599) AS i MATCH (a:Articulo {n: i}) SET a.embedding = [0.0, 1.0, toFloat(i)]",
    )
    .await;
    assert_counts(&writer, 600, 600, "second batch").await;

    // Re-embedding a row does not change presence.
    write(
        &mut writer,
        "MATCH (a:Articulo) WHERE a.n < 50 SET a.embedding = [0.5, 0.5, 0.5]",
    )
    .await;
    assert_counts(&writer, 600, 600, "re-embedded").await;

    write(
        &mut writer,
        "MATCH (a:Articulo) WHERE a.n < 10 REMOVE a.embedding",
    )
    .await;
    write(
        &mut writer,
        "MATCH (a:Articulo) WHERE a.n >= 10 AND a.n < 15 SET a.embedding = null",
    )
    .await;
    assert_counts(&writer, 600, 585, "removed").await;

    write(
        &mut writer,
        "MATCH (a:Articulo) WHERE a.n >= 590 DETACH DELETE a",
    )
    .await;
    write(
        &mut writer,
        "MATCH (a:Articulo) WHERE a.n < 3 DETACH DELETE a",
    )
    .await;
    assert_counts(&writer, 587, 575, "deleted").await;

    writer.flush(schema()).await.unwrap();
    assert_counts(&writer, 587, 575, "flushed").await;
    write(
        &mut writer,
        "CREATE (:Articulo {n: 1000, embedding: [1.0, 1.0, 1.0]})",
    )
    .await;
    assert_counts(&writer, 588, 576, "created with an embedding").await;
    writer.flush(schema()).await.unwrap();
    drop(writer);

    let reopened = WriterSession::open(store, paths).await.unwrap();
    assert_counts(&reopened, 588, 576, "reopened").await;
}
