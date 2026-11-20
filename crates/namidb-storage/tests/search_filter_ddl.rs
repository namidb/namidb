//! A property-index or uniqueness DDL on a label with search indexes changes
//! their native-filter set and with it their catalog signature. Up to 3.5.1
//! the DDL retired the serving generations outright and every search on the
//! label fell back to a scan until a new generation was rebuilt from
//! scratch. Since 3.5.2 the DDL re-signs them in place: they keep serving and
//! receiving writes, and one background base rebuild adds the new filter and
//! swaps in. These pin the cases around that: a restart in the middle of the
//! rebuild, a constraint that removes a filter, a composite index, and a
//! filter whose values exceed the per-segment distinct cap (omitted from the
//! built base, which must not request rebuilds forever), and a DDL during
//! a full-text re-analysis.
#![cfg(all(feature = "vector-index", feature = "text-index"))]

use std::collections::BTreeMap;
use std::sync::Arc;

use namidb_core::id::{NamespaceId, NodeId};
use namidb_core::schema::{DataType, LabelDef, PropertyDef, Schema, SchemaBuilder};
use namidb_core::value::Value as CoreValue;
use namidb_storage::manifest::{
    TextIndexDescriptor, VectorIndexDescriptor, VectorMetric, VectorQuantization,
};
use namidb_storage::search_lsm::{filter_rebuild_due, validate_search_lsm, SearchLsmStatus};
use namidb_storage::search_progress::{index_status, IndexState};
use namidb_storage::{Error, NamespacePaths, NodeWriteRecord, WriterSession};
use object_store::memory::InMemory;
use object_store::ObjectStore;

fn schema() -> Schema {
    SchemaBuilder::new()
        .label(LabelDef {
            name: "Doc".into(),
            properties: vec![
                PropertyDef::new("embedding", DataType::FloatVector { dim: 4 }, true).unwrap(),
                PropertyDef::new("body", DataType::Utf8, true).unwrap(),
                PropertyDef::new("city", DataType::Utf8, true).unwrap(),
                PropertyDef::new("code", DataType::Utf8, true).unwrap(),
            ],
        })
        .unwrap()
        .build()
}

fn doc(ordinal: u64) -> NodeWriteRecord {
    let mut props = BTreeMap::new();
    props.insert(
        "embedding".into(),
        CoreValue::Vec(vec![ordinal as f32, 1.0, (ordinal % 7) as f32, 0.5]),
    );
    props.insert(
        "body".into(),
        CoreValue::Str(format!("alpha doc{ordinal} w{}", ordinal % 13)),
    );
    props.insert(
        "city".into(),
        CoreValue::Str(["madrid", "lima", "quito"][(ordinal % 3) as usize].into()),
    );
    props.insert("code".into(), CoreValue::Str(format!("code-{ordinal}")));
    NodeWriteRecord {
        properties: props,
        schema_version: 1,
        ..Default::default()
    }
}

async fn writer_with_indexes(
    name: &str,
    documents: u64,
) -> (WriterSession, Arc<dyn ObjectStore>, NamespacePaths) {
    let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
    let paths = NamespacePaths::new("tenants", NamespaceId::new(name).unwrap());
    let mut writer = WriterSession::open(store.clone(), paths.clone())
        .await
        .unwrap();
    writer
        .register_vector_index(
            VectorIndexDescriptor {
                name: "doc_emb".into(),
                label: "Doc".into(),
                property: "embedding".into(),
                dim: 4,
                metric: VectorMetric::Cosine,
                r: 16,
                l_build: 32,
                alpha: 1.2,
                quantization: VectorQuantization::None,
            },
            false,
        )
        .await
        .unwrap();
    writer
        .register_text_index(
            TextIndexDescriptor::new("doc_ft".into(), "Doc".into(), vec!["body".into()]),
            false,
        )
        .await
        .unwrap();
    for ordinal in 0..documents {
        writer
            .upsert_node("Doc", NodeId::new(), &doc(ordinal))
            .unwrap();
    }
    writer.flush(schema()).await.unwrap();
    writer.compact_l0(&schema()).await.unwrap();
    assert_eq!(generations(&writer).len(), 2);
    (writer, store, paths)
}

fn generations(writer: &WriterSession) -> Vec<(String, uuid::Uuid, SearchLsmStatus)> {
    let mut generations = writer
        .current_manifest()
        .search_lsm
        .iter()
        .map(|state| (state.index_name.clone(), state.generation_id, state.status))
        .collect::<Vec<_>>();
    generations.sort_by(|left, right| left.0.cmp(&right.0));
    generations
}

/// Every generation is valid, ONLINE, and has a rebuild pending or not.
fn assert_serving(writer: &WriterSession, rebuild_pending: bool) {
    let manifest = writer.current_manifest();
    validate_search_lsm(manifest).expect("a valid manifest");
    for state in &manifest.search_lsm {
        assert_eq!(
            state.status,
            SearchLsmStatus::Active,
            "{}",
            state.index_name
        );
        assert_eq!(
            filter_rebuild_due(manifest, state),
            rebuild_pending,
            "{}",
            state.index_name
        );
        let status = index_status(manifest, state.kind, &state.index_name);
        assert_eq!(status.state, IndexState::Online, "{}", state.index_name);
        assert_eq!(status.percent, 100.0);
        assert_eq!(
            status.migration_percent.is_some(),
            rebuild_pending,
            "{}",
            state.index_name
        );
    }
}

async fn text_hits(writer: &WriterSession, groups: &[(String, Vec<CoreValue>)]) -> Option<usize> {
    let query = namidb_storage::text::parse_query("alpha");
    let snapshot = writer.snapshot();
    if groups.is_empty() {
        snapshot.text_search("doc_ft", "Doc", &query, None).await
    } else {
        snapshot
            .text_search_filter_groups("doc_ft", "Doc", &query, None, groups)
            .await
    }
    .unwrap()
    .map(|hits| hits.len())
}

async fn analyzed_hits(
    writer: &WriterSession,
    analyzer: namidb_storage::text::TextAnalyzer,
    groups: Vec<(String, Vec<CoreValue>)>,
) -> Option<usize> {
    let query = namidb_storage::text::parse_query_with(analyzer, "alpha");
    let snapshot = writer.snapshot();
    if groups.is_empty() {
        snapshot.text_search("doc_ft", "Doc", &query, None).await
    } else {
        snapshot
            .text_search_filter_groups("doc_ft", "Doc", &query, None, &groups)
            .await
    }
    .unwrap()
    .map(|hits| hits.len())
}

fn city(value: &str) -> Vec<(String, Vec<CoreValue>)> {
    vec![("city".to_string(), vec![CoreValue::Str(value.into())])]
}

/// A crash between the rebuild's upload and its swap: the uploaded base is
/// an orphan, the re-signed generation still serves after the restart, and
/// the rebuild runs again from the manifest alone.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_restart_in_the_middle_of_the_filter_rebuild_keeps_serving_and_resumes() {
    let (mut writer, store, paths) = writer_with_indexes("filter-ddl-restart", 60).await;
    let before = generations(&writer);
    writer
        .create_property_index_named(None, "Doc", "city", false)
        .await
        .unwrap();
    assert_eq!(generations(&writer), before, "kept, not retired");
    assert_serving(&writer, true);
    assert_eq!(text_hits(&writer, &[]).await, Some(60));
    assert_eq!(
        text_hits(&writer, &city("lima")).await,
        None,
        "no filter yet"
    );

    // The rebuild is prepared (its base uploaded) but never installed.
    let basis = writer.compaction_basis();
    let prepared = basis.prepare_search_only(&schema()).await.unwrap();
    drop(prepared);
    drop(writer);

    let mut writer = WriterSession::open(store, paths).await.unwrap();
    assert_eq!(generations(&writer), before);
    assert_serving(&writer, true);
    assert!(writer.search_compaction_due(), "the rebuild resumes");
    assert_eq!(text_hits(&writer, &[]).await, Some(60));

    writer.compact_search_segments().await.unwrap();
    assert_serving(&writer, false);
    assert!(!writer.search_compaction_due(), "and runs once");
    assert_eq!(text_hits(&writer, &city("lima")).await, Some(20));
}

/// A uniqueness constraint on a property that was a native filter removes
/// it from the set: the generations are re-signed, their segments keep the
/// now extra filter (still exact for their documents), and nothing is
/// rebuilt.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_constraint_that_removes_a_filter_needs_no_rebuild() {
    let (mut writer, _store, _paths) = writer_with_indexes("filter-ddl-unique", 40).await;
    let before = generations(&writer);
    writer
        .create_property_index_named(None, "Doc", "code", false)
        .await
        .unwrap();
    writer.compact_search_segments().await.unwrap();
    assert_serving(&writer, false);
    let carry_code = |writer: &WriterSession| {
        writer.current_manifest().search_lsm.iter().all(|state| {
            state
                .segments
                .iter()
                .all(|segment| segment.complete_filter_properties == ["code"])
        })
    };
    assert!(carry_code(&writer));
    let rebuilt = generations(&writer);

    writer
        .create_unique_constraint_named(None, "Doc", &["code".to_string()], false)
        .await
        .unwrap();
    assert_eq!(generations(&writer), rebuilt);
    assert_eq!(
        rebuilt.iter().map(|(name, ..)| name).collect::<Vec<_>>(),
        before.iter().map(|(name, ..)| name).collect::<Vec<_>>()
    );
    assert!(
        carry_code(&writer),
        "the extra filter stays on the segments"
    );
    assert_serving(&writer, false);
    assert!(
        !writer.search_compaction_due(),
        "no rebuild for a removed filter"
    );
    assert_eq!(text_hits(&writer, &[]).await, Some(40));

    // New deltas no longer carry it; still valid, still no rebuild.
    writer.upsert_node("Doc", NodeId::new(), &doc(500)).unwrap();
    writer.flush(schema()).await.unwrap();
    assert_serving(&writer, false);
    assert!(!writer.search_compaction_due());
    assert_eq!(text_hits(&writer, &[]).await, Some(41));
}

/// A composite index declares no single-property filter: nothing changes.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_composite_index_keeps_the_generations() {
    let (mut writer, _store, _paths) = writer_with_indexes("filter-ddl-composite", 30).await;
    let before = generations(&writer);
    writer
        .create_composite_index_named(
            None,
            "Doc",
            &["city".to_string(), "code".to_string()],
            false,
        )
        .await
        .unwrap();
    assert_eq!(generations(&writer), before);
    validate_search_lsm(writer.current_manifest()).unwrap();
    for state in &writer.current_manifest().search_lsm {
        assert_eq!(
            index_status(writer.current_manifest(), state.kind, &state.index_name).state,
            IndexState::Online
        );
    }
    assert_eq!(text_hits(&writer, &[]).await, Some(30));
}

/// More distinct values than a segment's filter cap: the rebuilt bases
/// omit the filter (filtered searches on it take the exact fallback) and
/// record the omission, so the rebuild is not requested again.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_filter_omitted_over_the_distinct_cap_does_not_loop() {
    let (mut writer, _store, _paths) = writer_with_indexes("filter-ddl-over-cap", 4_200).await;
    writer
        .create_property_index_named(None, "Doc", "code", false)
        .await
        .unwrap();
    assert_serving(&writer, true);
    writer.compact_search_segments().await.unwrap();
    let manifest = writer.current_manifest();
    let text = manifest
        .search_lsm
        .iter()
        .find(|state| state.index_name == "doc_ft")
        .unwrap();
    assert!(
        text.segments
            .iter()
            .all(|segment| segment.complete_filter_properties.is_empty()),
        "the text base omits the over-cap filter"
    );
    assert_eq!(text.filters_omitted_over_cap, ["code"]);
    assert_serving(&writer, false);
    assert!(!writer.search_compaction_due(), "no rebuild loop");
    let outcome = writer.compact_search_segments().await.unwrap();
    assert_serving(&writer, false);
    let _ = outcome;
    assert_eq!(text_hits(&writer, &[]).await, Some(4_200));
}

/// A filter DDL that lands while a full-text re-analysis (the move to the
/// accent-folding analyzer) builds its base outruns that attempt: its
/// catch-up and install refuse it, and the server restarts it rather than
/// counting a failure. The old generation serves throughout, and the
/// restarted re-analysis builds from the re-signed generation, so its base
/// carries the new filter and no filter rebuild is left behind.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_filter_ddl_during_a_text_reanalysis_restarts_it_with_the_new_filter() {
    use namidb_storage::text::TextAnalyzer;

    let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
    let paths = NamespacePaths::new(
        "tenants",
        NamespaceId::new("filter-ddl-reanalysis").unwrap(),
    );
    let mut writer = WriterSession::open(store, paths).await.unwrap();
    let legacy = TextIndexDescriptor {
        analyzer: None,
        ..TextIndexDescriptor::new("doc_ft".into(), "Doc".into(), vec!["body".into()])
    };
    writer.register_text_index(legacy, false).await.unwrap();
    for ordinal in 0..60 {
        writer
            .upsert_node("Doc", NodeId::new(), &doc(ordinal))
            .unwrap();
    }
    writer.flush(schema()).await.unwrap();
    writer.compact_l0(&schema()).await.unwrap();

    let basis = writer.compaction_basis();
    assert_eq!(basis.legacy_text_indexes(), ["doc_ft".to_string()]);
    let mut prepared = basis
        .prepare_text_reanalysis("doc_ft")
        .await
        .unwrap()
        .expect("a legacy index is a candidate");
    drop(basis);

    // While it builds: a write, then the filter DDL.
    writer
        .upsert_node("Doc", NodeId::new(), &doc(1_000))
        .unwrap();
    writer.flush(schema()).await.unwrap();
    writer
        .create_property_index_named(None, "Doc", "city", false)
        .await
        .unwrap();
    assert_serving(&writer, true);
    assert_eq!(
        analyzed_hits(&writer, TextAnalyzer::Exact, vec![]).await,
        Some(61)
    );

    // The attempt is outrun by the re-sign, not by a definition change.
    assert!(prepared.outrun_by_filter_resign(writer.current_manifest()));
    let refused = writer
        .compaction_basis()
        .catch_up_text_reanalysis(&mut prepared)
        .await;
    assert!(
        matches!(refused, Err(Error::Precondition(_))),
        "{refused:?}"
    );
    let refused = writer.install_text_reanalysis(prepared).await;
    assert!(
        matches!(refused, Err(Error::Precondition(_))),
        "{refused:?}"
    );
    assert_serving(&writer, true);
    assert_eq!(
        analyzed_hits(&writer, TextAnalyzer::Exact, vec![]).await,
        Some(61)
    );

    // The restart builds from the re-signed generation.
    let basis = writer.compaction_basis();
    assert_eq!(basis.legacy_text_indexes(), ["doc_ft".to_string()]);
    let prepared = basis
        .prepare_text_reanalysis("doc_ft")
        .await
        .unwrap()
        .expect("still a candidate");
    drop(basis);
    assert!(!prepared.outrun_by_filter_resign(writer.current_manifest()));
    writer.install_text_reanalysis(prepared).await.unwrap();

    let manifest = writer.current_manifest();
    let text = manifest
        .search_lsm
        .iter()
        .find(|state| state.index_name == "doc_ft")
        .unwrap();
    assert!(text
        .segments
        .iter()
        .all(|segment| segment.complete_filter_properties == ["city"]));
    assert_serving(&writer, false);
    assert!(!writer.search_compaction_due(), "no filter rebuild left");
    assert_eq!(
        analyzed_hits(&writer, TextAnalyzer::Folded, vec![]).await,
        Some(61)
    );
    // ordinals 1, 4, ..., 58 and 1000
    assert_eq!(
        analyzed_hits(&writer, TextAnalyzer::Folded, city("lima")).await,
        Some(21)
    );
}
