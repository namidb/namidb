//! Write-time embedding-dimension enforcement (issue g).
//!
//! When a vector index covers `(label, property)`, a write that stores an
//! embedding of the wrong dimension is rejected with `ExecError::Constraint`
//! instead of silently poisoning the next index build (a single mismatched row
//! makes `build_body` error and skip the whole `.vg`). A correct-dimension bare
//! list (`[f, …]`, no `vector()`) is coerced to a dense vector so it is actually
//! indexed rather than silently skipped at build time. With no index registered,
//! writes are unconstrained (the pre-existing behaviour the RFC's Part B targets).
//!
//! Enforcement is feature-independent — it only consults the registered
//! descriptors — so these tests need no `vector-index` feature: they register a
//! descriptor directly and exercise the write path.

use std::sync::Arc;

use namidb_core::id::NamespaceId;
use namidb_storage::manifest::{VectorIndexDescriptor, VectorMetric, VectorQuantization};
use namidb_storage::{NamespacePaths, WriterSession};
use object_store::memory::InMemory;
use object_store::ObjectStore;

use namidb_query::{execute, execute_write, lower, parse, ExecError, Params, RuntimeValue};

fn store() -> Arc<dyn ObjectStore> {
    Arc::new(InMemory::new())
}

fn paths(name: &str) -> NamespacePaths {
    NamespacePaths::new("tenants", NamespaceId::new(name).unwrap())
}

fn doc_index(dim: u32) -> VectorIndexDescriptor {
    VectorIndexDescriptor {
        name: "doc_emb".into(),
        label: "Doc".into(),
        property: "embedding".into(),
        dim,
        metric: VectorMetric::Cosine,
        r: 32,
        l_build: 64,
        alpha: 1.2,
        quantization: VectorQuantization::None,
    }
}

/// A writer with a dim-4 cosine index on `:Doc(embedding)` registered.
async fn open_with_index(ns: &str) -> WriterSession {
    let mut w = WriterSession::open(store(), paths(ns)).await.unwrap();
    w.register_vector_index(doc_index(4), false).await.unwrap();
    w
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

fn is_dim_constraint(err: &ExecError) -> bool {
    matches!(err, ExecError::Constraint(msg) if msg.contains("dimension constraint"))
}

async fn doc_count(writer: &WriterSession) -> usize {
    writer.snapshot().scan_label("Doc").await.unwrap().len()
}

#[tokio::test]
async fn create_wrong_dim_rejected() {
    let mut w = open_with_index("vdim-create-bad").await;
    let err = write_err(&mut w, "CREATE (:Doc {embedding: vector([1.0, 2.0, 3.0])})").await;
    assert!(is_dim_constraint(&err), "{err:?}");
    if let ExecError::Constraint(msg) = &err {
        assert!(msg.contains("dim 3"), "message names the wrong dim: {msg}");
        assert!(msg.contains('4'), "message names the declared dim: {msg}");
    }
    assert_eq!(
        doc_count(&w).await,
        0,
        "the rejected node must not be staged"
    );
}

#[tokio::test]
async fn create_correct_dim_accepted() {
    let mut w = open_with_index("vdim-create-ok").await;
    let out = write_q(
        &mut w,
        "CREATE (:Doc {embedding: vector([1.0, 2.0, 3.0, 4.0])})",
    )
    .await;
    assert_eq!(out.nodes_created, 1);
    assert_eq!(doc_count(&w).await, 1);
}

#[tokio::test]
async fn set_wrong_dim_rejected_and_leaves_value_unchanged() {
    let mut w = open_with_index("vdim-set-bad").await;
    write_q(
        &mut w,
        "CREATE (:Doc {embedding: vector([1.0, 0.0, 0.0, 0.0])})",
    )
    .await;
    let err = write_err(
        &mut w,
        "MATCH (d:Doc) SET d.embedding = vector([1.0, 2.0, 3.0])",
    )
    .await;
    assert!(is_dim_constraint(&err), "{err:?}");
    // The original 4-d embedding is untouched.
    let snap = w.snapshot();
    let plan = lower(&parse("MATCH (d:Doc) RETURN d.embedding AS e").unwrap()).unwrap();
    let rows = execute(&plan, &snap, &Params::new()).await.unwrap();
    match rows[0].get("e") {
        Some(RuntimeValue::Vector(v)) => assert_eq!(v.len(), 4),
        other => panic!("embedding must be unchanged 4-vector, got {other:?}"),
    }
}

#[tokio::test]
async fn set_map_merge_wrong_dim_rejected() {
    let mut w = open_with_index("vdim-setmap-bad").await;
    write_q(
        &mut w,
        "CREATE (:Doc {embedding: vector([1.0, 0.0, 0.0, 0.0])})",
    )
    .await;
    let err = write_err(
        &mut w,
        "MATCH (d:Doc) SET d += {embedding: vector([1.0, 2.0, 3.0])}",
    )
    .await;
    assert!(is_dim_constraint(&err), "{err:?}");
}

#[tokio::test]
async fn add_label_brings_index_validates_existing_embedding() {
    let mut w = open_with_index("vdim-addlabel").await;
    // An :Item (no index) holding a 3-d embedding, then it gains :Doc whose index
    // declares dim 4 → the gained label's contract rejects the existing value.
    write_q(
        &mut w,
        "CREATE (:Item {embedding: vector([1.0, 2.0, 3.0])})",
    )
    .await;
    let err = write_err(&mut w, "MATCH (n:Item) SET n:Doc").await;
    assert!(is_dim_constraint(&err), "{err:?}");
}

#[tokio::test]
async fn bare_list_correct_dim_is_coerced_to_a_vector() {
    // The exact user scenario: `embedding = [floats]` with no `vector()`. A bare
    // list of the right dim is accepted AND coerced to a dense vector so the index
    // build covers it (a list is silently skipped at build time).
    let mut w = open_with_index("vdim-list-ok").await;
    write_q(&mut w, "CREATE (:Doc {embedding: [1.0, 2.0, 3.0, 4.0]})").await;
    let snap = w.snapshot();
    let plan = lower(&parse("MATCH (d:Doc) RETURN d.embedding AS e").unwrap()).unwrap();
    let rows = execute(&plan, &snap, &Params::new()).await.unwrap();
    match rows[0].get("e") {
        Some(RuntimeValue::Vector(v)) => assert_eq!(v.len(), 4, "coerced to a 4-vector"),
        other => panic!("bare list must be coerced to a Vector, got {other:?}"),
    }
}

#[tokio::test]
async fn bare_list_wrong_dim_rejected() {
    let mut w = open_with_index("vdim-list-bad").await;
    let err = write_err(&mut w, "CREATE (:Doc {embedding: [1.0, 2.0, 3.0]})").await;
    assert!(is_dim_constraint(&err), "{err:?}");
}

#[tokio::test]
async fn unrelated_set_on_legacy_wrong_dim_node_succeeds() {
    use namidb_core::id::NodeId;
    use namidb_core::value::Value as CoreValue;
    use namidb_storage::NodeWriteRecord;
    let mut w = open_with_index("vdim-legacy").await;
    // Seed a node with a dim-3 embedding via the STORAGE API, bypassing query-layer
    // enforcement — a row written before the index existed.
    let mut p = std::collections::BTreeMap::new();
    p.insert("title".to_string(), CoreValue::Str("legacy".into()));
    p.insert("embedding".to_string(), CoreValue::Vec(vec![1.0, 2.0, 3.0]));
    w.upsert_node(
        "Doc",
        NodeId::new(),
        &NodeWriteRecord {
            properties: p,
            schema_version: 1,
            ..Default::default()
        },
    )
    .unwrap();
    w.commit_batch().await.unwrap();
    // Setting an UNRELATED property must NOT re-validate the legacy embedding.
    let out = write_q(&mut w, "MATCH (d:Doc) SET d.title = 'updated'").await;
    assert_eq!(out.properties_set, 1, "unrelated SET must succeed");
    // …but re-writing the embedding itself to a wrong dim is still rejected.
    let err = write_err(&mut w, "MATCH (d:Doc) SET d.embedding = vector([1.0, 2.0])").await;
    assert!(is_dim_constraint(&err), "{err:?}");
}

#[tokio::test]
async fn drop_vector_index_unbricks_wrong_dim_writes() {
    // The misconfiguration remedy end-to-end: an index declared with the wrong
    // dimension (1536) rejects every write of the real embedding size (768)
    // — with no other way out, since the index would otherwise be permanent.
    // DROP VECTOR INDEX removes the descriptor; the same write then succeeds,
    // and a corrected re-create over the same slot enforces the right dim.
    let mut w = WriterSession::open(store(), paths("vdim-drop-unbrick"))
        .await
        .unwrap();
    w.register_vector_index(doc_index(1536), false)
        .await
        .unwrap();

    let mut params = Params::new();
    params.insert("v".into(), RuntimeValue::Vector(vec![0.5; 768]));
    let plan = lower(&parse("CREATE (:Doc {embedding: $v})").unwrap()).unwrap();
    let err = execute_write(&plan, &mut w, &params)
        .await
        .expect_err("a 768-dim write against the dim-1536 index must be rejected");
    assert!(is_dim_constraint(&err), "{err:?}");
    assert_eq!(doc_count(&w).await, 0);

    // Drop the misconfigured index: the identical write now succeeds.
    w.drop_vector_index("doc_emb", false).await.unwrap();
    execute_write(&plan, &mut w, &params)
        .await
        .expect("the write must succeed once the index is dropped");
    assert_eq!(doc_count(&w).await, 1);

    // Re-create corrected (dim 768) over the same (label, property, metric)
    // slot: accepted, and it enforces the corrected dimension.
    w.register_vector_index(doc_index(768), false)
        .await
        .unwrap();
    execute_write(&plan, &mut w, &params)
        .await
        .expect("a correct-dim write against the corrected index must succeed");
    assert_eq!(doc_count(&w).await, 2);
    let err = write_err(&mut w, "CREATE (:Doc {embedding: vector([1.0, 2.0])})").await;
    assert!(is_dim_constraint(&err), "{err:?}");
}

#[tokio::test]
async fn no_index_means_no_dim_enforcement() {
    // Part A boundary: with no vector index registered, any dimension is accepted
    // (the silent-mismatch behaviour the RFC's Part B would close via a typed
    // property / VECTOR(dim) constraint).
    let mut w = WriterSession::open(store(), paths("vdim-noindex"))
        .await
        .unwrap();
    let out = write_q(&mut w, "CREATE (:Doc {embedding: vector([1.0, 2.0, 3.0])})").await;
    assert_eq!(out.nodes_created, 1);
}
