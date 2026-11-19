//! Full-text indexes created before accent folding keep their lowercase-only
//! analyzer, serve queries parsed with it, and are rebuilt with the folding
//! analyzer off-manifest, then swapped in with one commit. A rebuild outrun
//! by writes to the index's documents is refused and leaves the old index
//! serving; writes elsewhere do not hold it back.
#![cfg(feature = "text-index")]

use std::collections::BTreeMap;
use std::sync::Arc;

use namidb_core::id::{NamespaceId, NodeId};
use namidb_core::schema::{DataType, LabelDef, PropertyDef, Schema, SchemaBuilder};
use namidb_core::value::Value as CoreValue;
use namidb_storage::manifest::TextIndexDescriptor;
use namidb_storage::text::{parse_query_with, TextAnalyzer};
use namidb_storage::{Error, NamespacePaths, NodeWriteRecord, WriterSession};
use object_store::memory::InMemory;
use object_store::ObjectStore;

const BODIES: &[&str] = &[
    "La pensión jubilar se calcula sobre el último sueldo",
    "Ação judicial sobre a indenização",
    "Le cœur du problème est la procédure",
    "Straße und Brücke",
    "plain ascii text about pension rules",
];

fn schema() -> Schema {
    SchemaBuilder::new()
        .label(LabelDef {
            name: "Doc".into(),
            properties: vec![PropertyDef::new("body", DataType::Utf8, true).unwrap()],
        })
        .unwrap()
        .label(LabelDef {
            name: "Other".into(),
            properties: vec![PropertyDef::new("n", DataType::Int64, true).unwrap()],
        })
        .unwrap()
        .build()
}

fn doc(body: &str) -> NodeWriteRecord {
    let mut properties = BTreeMap::new();
    properties.insert("body".to_string(), CoreValue::Str(body.to_string()));
    NodeWriteRecord {
        properties,
        schema_version: 1,
        ..Default::default()
    }
}

fn other(n: i64) -> NodeWriteRecord {
    let mut properties = BTreeMap::new();
    properties.insert("n".to_string(), CoreValue::I64(n));
    NodeWriteRecord {
        properties,
        schema_version: 1,
        ..Default::default()
    }
}

async fn search(
    writer: &WriterSession,
    analyzer: TextAnalyzer,
    query: &str,
) -> Option<Vec<(NodeId, f64)>> {
    writer
        .snapshot()
        .text_search(
            "doc_ft",
            "Doc",
            &parse_query_with(analyzer, query),
            Some(50),
        )
        .await
        .unwrap()
}

fn analyzer_of(writer: &WriterSession) -> Option<TextAnalyzer> {
    writer
        .current_manifest()
        .text_indexes
        .iter()
        .find(|descriptor| descriptor.name == "doc_ft")
        .expect("index registered")
        .analyzer
}

/// A namespace with a legacy (pre-folding) index over `BODIES`, repeated, with
/// an Active generation and a base.
async fn legacy_namespace(name: &str) -> (WriterSession, Vec<NodeId>) {
    let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
    let paths = NamespacePaths::new("tenants", NamespaceId::new(name).unwrap());
    let mut writer = WriterSession::open(store, paths).await.unwrap();
    let legacy = TextIndexDescriptor {
        analyzer: None,
        ..TextIndexDescriptor::new("doc_ft".into(), "Doc".into(), vec!["body".into()])
    };
    writer.register_text_index(legacy, false).await.unwrap();
    let mut ids = Vec::new();
    for round in 0..8 {
        for body in BODIES {
            let id = NodeId::new();
            writer
                .upsert_node("Doc", id, &doc(&format!("{body} r{round}")))
                .unwrap();
            ids.push(id);
        }
    }
    writer.commit_batch().await.unwrap();
    writer.flush(schema()).await.unwrap();
    writer.compact_l0(&schema()).await.unwrap();
    assert_eq!(analyzer_of(&writer), None);
    (writer, ids)
}

#[tokio::test]
async fn legacy_index_serves_its_own_analyzer_until_rebuilt() {
    let (mut writer, ids) = legacy_namespace("text-reanalysis").await;

    // The legacy index answers exact-analyzed queries natively ...
    let exact = search(&writer, TextAnalyzer::Exact, "pensión")
        .await
        .expect("legacy index serves its analyzer");
    assert_eq!(exact.len(), 8);
    // ... and declines folded ones, which the caller answers by exact scan.
    assert_eq!(search(&writer, TextAnalyzer::Folded, "pension").await, None);

    let basis = writer.compaction_basis();
    assert_eq!(basis.legacy_text_indexes(), vec!["doc_ft".to_string()]);
    let prepared = basis
        .prepare_text_reanalysis("doc_ft")
        .await
        .unwrap()
        .expect("legacy index is a candidate");
    assert_eq!(prepared.index_name(), "doc_ft");
    // Still legacy until the install.
    assert!(search(&writer, TextAnalyzer::Exact, "pensión")
        .await
        .is_some());
    writer.install_text_reanalysis(prepared).await.unwrap();

    assert_eq!(analyzer_of(&writer), Some(TextAnalyzer::Folded));
    assert!(writer.compaction_basis().legacy_text_indexes().is_empty());
    let folded = search(&writer, TextAnalyzer::Folded, "PENSION")
        .await
        .expect("rebuilt index serves folded queries");
    // "pensión" (8 docs) and the ascii "pension" (8 docs) are one term now.
    assert_eq!(folded.len(), 16);
    for (query, expected) in [
        ("acao", 8),
        ("indenizacao", 8),
        ("coeur", 8),
        ("strasse", 8),
    ] {
        let hits = search(&writer, TextAnalyzer::Folded, query)
            .await
            .expect("rebuilt index serves");
        assert_eq!(hits.len(), expected, "query {query}");
    }
    assert_eq!(search(&writer, TextAnalyzer::Exact, "pensión").await, None);

    // Later writes are analyzed the same way, through flush and compaction.
    let fresh = NodeId::new();
    writer
        .upsert_node("Doc", fresh, &doc("Educación y Jubilación"))
        .unwrap();
    writer.tombstone_node("Doc", ids[0]).unwrap();
    writer.commit_batch().await.unwrap();
    writer.flush(schema()).await.unwrap();
    let hits = search(&writer, TextAnalyzer::Folded, "educacion")
        .await
        .expect("delta segment serves");
    assert_eq!(
        hits.iter().map(|(id, _)| *id).collect::<Vec<_>>(),
        vec![fresh]
    );
    writer.compact_l0(&schema()).await.unwrap();
    assert_eq!(
        search(&writer, TextAnalyzer::Folded, "pension")
            .await
            .expect("serves after compaction")
            .len(),
        15
    );

    // The swapped manifest reopens as is.
    let store = writer.manifest_store().store().clone();
    let paths = writer.manifest_store().paths().clone();
    drop(writer);
    let reopened = WriterSession::open(store, paths).await.unwrap();
    assert_eq!(analyzer_of(&reopened), Some(TextAnalyzer::Folded));
    assert_eq!(
        search(&reopened, TextAnalyzer::Folded, "JUBILACION")
            .await
            .expect("reopened index serves")
            .len(),
        1
    );
}

#[tokio::test]
async fn rebuild_outrun_by_indexed_writes_is_refused() {
    let (mut writer, _) = legacy_namespace("text-reanalysis-outrun").await;
    let prepared = writer
        .compaction_basis()
        .prepare_text_reanalysis("doc_ft")
        .await
        .unwrap()
        .expect("candidate");
    writer
        .upsert_node("Doc", NodeId::new(), &doc("nueva resolución"))
        .unwrap();
    writer.commit_batch().await.unwrap();
    writer.flush(schema()).await.unwrap();
    let error = writer.install_text_reanalysis(prepared).await.unwrap_err();
    assert!(matches!(error, Error::Precondition(_)), "{error}");
    assert_eq!(analyzer_of(&writer), None);
    assert_eq!(
        search(&writer, TextAnalyzer::Exact, "resolución")
            .await
            .expect("legacy index still serves")
            .len(),
        1
    );

    // The retry, from a fresh basis, lands.
    let prepared = writer
        .compaction_basis()
        .prepare_text_reanalysis("doc_ft")
        .await
        .unwrap()
        .expect("still a candidate");
    writer.install_text_reanalysis(prepared).await.unwrap();
    assert_eq!(
        search(&writer, TextAnalyzer::Folded, "resolucion")
            .await
            .expect("rebuilt")
            .len(),
        1
    );
}

#[tokio::test]
async fn writes_outside_the_index_do_not_hold_the_rebuild_back() {
    let (mut writer, _) = legacy_namespace("text-reanalysis-elsewhere").await;
    let prepared = writer
        .compaction_basis()
        .prepare_text_reanalysis("doc_ft")
        .await
        .unwrap()
        .expect("candidate");
    for n in 0..5 {
        writer
            .upsert_node("Other", NodeId::new(), &other(n))
            .unwrap();
    }
    writer.commit_batch().await.unwrap();
    writer.flush(schema()).await.unwrap();
    writer.install_text_reanalysis(prepared).await.unwrap();
    assert_eq!(analyzer_of(&writer), Some(TextAnalyzer::Folded));
    assert_eq!(
        search(&writer, TextAnalyzer::Folded, "pension")
            .await
            .expect("rebuilt index serves")
            .len(),
        16
    );
}

#[tokio::test]
async fn opted_out_and_new_indexes_are_not_candidates() {
    let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
    let paths = NamespacePaths::new("tenants", NamespaceId::new("text-optout").unwrap());
    let mut writer = WriterSession::open(store, paths).await.unwrap();
    writer
        .register_text_index(
            TextIndexDescriptor::with_analyzer(
                "doc_ft".into(),
                "Doc".into(),
                vec!["body".into()],
                TextAnalyzer::Exact,
            ),
            false,
        )
        .await
        .unwrap();
    for body in BODIES {
        writer
            .upsert_node("Doc", NodeId::new(), &doc(body))
            .unwrap();
    }
    writer.commit_batch().await.unwrap();
    writer.flush(schema()).await.unwrap();
    writer.compact_l0(&schema()).await.unwrap();
    assert!(writer.compaction_basis().legacy_text_indexes().is_empty());
    assert!(writer
        .compaction_basis()
        .prepare_text_reanalysis("doc_ft")
        .await
        .unwrap()
        .is_none());
    assert_eq!(
        search(&writer, TextAnalyzer::Exact, "pensión")
            .await
            .expect("opted-out index serves exact queries")
            .len(),
        1
    );
    assert_eq!(
        search(&writer, TextAnalyzer::Exact, "pension")
            .await
            .unwrap()
            .len(),
        1
    );
}
