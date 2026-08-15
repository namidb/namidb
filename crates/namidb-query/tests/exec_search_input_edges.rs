//! Plan items 23 and 25 (docs/testing/25tb-readiness.md): empty/edge search
//! inputs at the procedure layer (empty and whitespace queries, `k: 0`,
//! omitted `k`, unknown labels, `LIMIT 0`) and the non-cosine metric forms
//! (`metric: 'dot' | 'euclidean' | 'l2'`, unknown-metric error, and
//! `db.index.vector.queryNodes` against a dot-metric descriptor).
#![cfg(all(feature = "vector-index", feature = "text-index"))]

use std::collections::BTreeMap;
use std::sync::Arc;

use namidb_core::id::{NamespaceId, NodeId};
use namidb_core::schema::{DataType, LabelDef, PropertyDef, Schema, SchemaBuilder};
use namidb_core::value::Value as CoreValue;
use namidb_storage::manifest::{
    TextIndexDescriptor, VectorIndexDescriptor, VectorMetric, VectorQuantization,
};
use namidb_storage::{NamespacePaths, NodeWriteRecord, WriterSession};
use object_store::memory::InMemory;
use object_store::ObjectStore;

use namidb_query::{execute, lower, parse, Params, RuntimeValue};

const DIM: u32 = 4;

fn schema() -> Schema {
    SchemaBuilder::new()
        .label(LabelDef {
            name: "Doc".into(),
            properties: vec![
                PropertyDef::new("name", DataType::Utf8, false).unwrap(),
                PropertyDef::new("embedding", DataType::FloatVector { dim: DIM }, true).unwrap(),
                PropertyDef::new("body", DataType::Utf8, true).unwrap(),
            ],
        })
        .unwrap()
        .build()
}

fn doc(name: &str, embedding: [f32; 4], body: &str) -> NodeWriteRecord {
    let mut props: BTreeMap<String, CoreValue> = BTreeMap::new();
    props.insert("name".into(), CoreValue::Str(name.into()));
    props.insert("embedding".into(), CoreValue::Vec(embedding.to_vec()));
    props.insert("body".into(), CoreValue::Str(body.into()));
    NodeWriteRecord {
        properties: props,
        schema_version: 1,
        ..Default::default()
    }
}

fn descriptor(metric: VectorMetric) -> VectorIndexDescriptor {
    VectorIndexDescriptor {
        name: "doc_emb".into(),
        label: "Doc".into(),
        property: "embedding".into(),
        dim: DIM,
        metric,
        r: 16,
        l_build: 32,
        alpha: 1.2,
        quantization: VectorQuantization::None,
    }
}

/// Three vectors with hand-computable rankings under every metric for the
/// probe [1, 0, 0, 0]:
///   near   = [0.9, 0.1, 0, 0]   — best cosine AND best euclidean
///   big    = [3, 3, 0, 0]       — best dot (magnitude wins), worse cosine
///   far    = [0, 1, 0, 0]       — worst under every metric
async fn corpus(name: &str, metric: VectorMetric) -> WriterSession {
    let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
    let paths = NamespacePaths::new("tenants", NamespaceId::new(name).unwrap());
    let mut writer = WriterSession::open(store, paths).await.unwrap();
    writer
        .register_vector_index(descriptor(metric), false)
        .await
        .unwrap();
    writer
        .register_text_index(
            TextIndexDescriptor::new("doc_ft".into(), "Doc".into(), vec!["body".into()]),
            false,
        )
        .await
        .unwrap();
    writer
        .upsert_node(
            "Doc",
            NodeId::new(),
            &doc("near", [0.9, 0.1, 0.0, 0.0], "alpha common"),
        )
        .unwrap();
    writer
        .upsert_node(
            "Doc",
            NodeId::new(),
            &doc("big", [3.0, 3.0, 0.0, 0.0], "beta common"),
        )
        .unwrap();
    writer
        .upsert_node(
            "Doc",
            NodeId::new(),
            &doc("far", [0.0, 1.0, 0.0, 0.0], "gamma common"),
        )
        .unwrap();
    writer.commit_batch().await.unwrap();
    writer.flush(schema()).await.unwrap();
    writer
}

async fn run(writer: &WriterSession, query: &str) -> Result<Vec<String>, namidb_query::ExecError> {
    let snapshot = writer.snapshot();
    let parsed = parse(query).unwrap_or_else(|error| panic!("parse `{query}`: {error:?}"));
    let plan = lower(&parsed).unwrap_or_else(|error| panic!("lower `{query}`: {error:?}"));
    let rows = execute(&plan, &snapshot, &Params::new()).await?;
    Ok(rows
        .iter()
        .map(|row| match row.bindings.values().next() {
            Some(RuntimeValue::String(text)) => text.clone(),
            Some(RuntimeValue::Node(node)) => match node.properties.get("name") {
                Some(RuntimeValue::String(name)) => name.clone(),
                other => format!("{other:?}"),
            },
            other => format!("{other:?}"),
        })
        .collect())
}

#[tokio::test]
async fn empty_and_whitespace_text_queries_return_no_rows_without_error() {
    let writer = corpus("edge-empty-text", VectorMetric::Cosine).await;
    for query_text in ["", "   ", "\\t"] {
        let rows = run(
            &writer,
            &format!(
                "CALL search.bm25({{label: 'Doc', text_properties: ['body'], \
                 query: '{query_text}'}}) YIELD node, score RETURN node"
            ),
        )
        .await
        .expect("an empty query is a valid question with zero answers");
        assert!(
            rows.is_empty(),
            "query {query_text:?} must match nothing, got {rows:?}"
        );
    }
}

#[tokio::test]
async fn zero_k_and_limit_zero_return_no_rows_and_omitted_k_defaults() {
    let writer = corpus("edge-zero-k", VectorMetric::Cosine).await;

    let rows = run(
        &writer,
        "CALL search.vector({label: 'Doc', property: 'embedding', \
         query: [1.0, 0.0, 0.0, 0.0], k: 0}) YIELD node RETURN node",
    )
    .await
    .expect("k: 0 is a valid ask for zero results");
    assert!(rows.is_empty());

    let rows = run(
        &writer,
        "CALL search.bm25({label: 'Doc', text_properties: ['body'], \
         query: 'common', k: 0}) YIELD node, score RETURN node",
    )
    .await
    .expect("k: 0 on bm25 is valid");
    assert!(rows.is_empty());

    // Omitted k → a bounded default, never an error or unbounded blowup.
    let rows = run(
        &writer,
        "CALL search.vector({label: 'Doc', property: 'embedding', \
         query: [1.0, 0.0, 0.0, 0.0]}) YIELD node RETURN node",
    )
    .await
    .expect("omitted k must default");
    assert_eq!(rows.len(), 3, "all three docs fit inside the default k");

    let rows = run(
        &writer,
        "CALL search.vector({label: 'Doc', property: 'embedding', \
         query: [1.0, 0.0, 0.0, 0.0], k: 2}) YIELD node RETURN node LIMIT 0",
    )
    .await
    .expect("LIMIT 0 is valid");
    assert!(rows.is_empty());
}

#[tokio::test]
async fn unknown_label_is_an_empty_answer_not_a_crash() {
    let writer = corpus("edge-unknown-label", VectorMetric::Cosine).await;
    let result = run(
        &writer,
        "CALL search.bm25({label: 'Ghost', text_properties: ['body'], \
         query: 'common'}) YIELD node, score RETURN node",
    )
    .await;
    match result {
        Ok(rows) => assert!(
            rows.is_empty(),
            "an unknown label matches nothing, got {rows:?}"
        ),
        Err(error) => {
            let message = format!("{error:?}");
            assert!(
                !message.to_lowercase().contains("panic"),
                "an unknown label may error cleanly but never panic: {message}"
            );
        }
    }
}

#[tokio::test]
async fn dot_and_euclidean_metrics_rank_by_their_own_geometry() {
    let writer = corpus("edge-metric-forms", VectorMetric::Cosine).await;

    // Cosine and euclidean agree: `near` first.
    for metric in ["euclidean", "l2"] {
        let rows = run(
            &writer,
            &format!(
                "CALL search.vector({{label: 'Doc', property: 'embedding', \
                 query: [1.0, 0.0, 0.0, 0.0], k: 1, metric: '{metric}'}}) \
                 YIELD node RETURN node"
            ),
        )
        .await
        .expect("euclidean form must run");
        assert_eq!(rows, vec!["near"], "metric {metric} must rank by distance");
    }

    // Dot rewards magnitude: `big` (3+0) beats `near` (0.9).
    let rows = run(
        &writer,
        "CALL search.vector({label: 'Doc', property: 'embedding', \
         query: [1.0, 0.0, 0.0, 0.0], k: 1, metric: 'dot'}) YIELD node RETURN node",
    )
    .await
    .expect("dot form must run");
    assert_eq!(rows, vec!["big"], "dot must rank by inner product");

    let error = run(
        &writer,
        "CALL search.vector({label: 'Doc', property: 'embedding', \
         query: [1.0, 0.0, 0.0, 0.0], metric: 'manhattan'}) YIELD node RETURN node",
    )
    .await
    .expect_err("unknown metric must error");
    assert!(
        format!("{error:?}").contains("unknown metric"),
        "error must name the metric contract, got {error:?}"
    );
}

#[tokio::test]
async fn query_nodes_serves_a_dot_metric_descriptor() {
    let writer = corpus("edge-dot-descriptor", VectorMetric::Dot).await;
    let snapshot = writer.snapshot();
    let parsed = parse(
        "CALL db.index.vector.queryNodes('doc_emb', 1, [1.0, 0.0, 0.0, 0.0]) \
         YIELD node, score RETURN node.name AS name",
    )
    .unwrap();
    let plan = lower(&parsed).unwrap();
    let rows = execute(&plan, &snapshot, &Params::new()).await.unwrap();
    let names: Vec<&str> = rows
        .iter()
        .map(|row| match row.get("name") {
            Some(RuntimeValue::String(name)) => name.as_str(),
            other => panic!("unexpected: {other:?}"),
        })
        .collect();
    assert_eq!(
        names,
        vec!["big"],
        "a Dot descriptor must rank by inner product through queryNodes"
    );
}
