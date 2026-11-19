//! A full-text index built by 3.4.x can still have a range-readable V3 base
//! (the format before FT4), wrapped as the base of its Search-LSM generation.
//! 3.5.x flushes append FT4 deltas to it, and up to 3.5.1 no reader accepted
//! that mix: every search fell back to a corpus scan, `SHOW INDEXES` reported
//! the index BUILDING 0%, and nothing scheduled the base rebuild that would
//! have repaired it until gigabytes of deltas accumulated.
//!
//! Now the coordinator merges the V3 base with the FT4 deltas under
//! snapshot-wide BM25 statistics, and the first compaction pass migrates the
//! base to FT4 in the background. These tests build the V3 base the way a
//! 3.4.x engine did (a logical index rebuild inside a Nodes merge, adopted as
//! the generation's base) and check that the merged answers are exactly the
//! answers of the migrated FT4 base, scores included.
#![cfg(feature = "text-index")]

use std::collections::BTreeMap;
use std::sync::Arc;

use namidb_core::id::{NamespaceId, NodeId};
use namidb_core::schema::{DataType, LabelDef, PropertyDef, Schema, SchemaBuilder};
use namidb_core::value::Value as CoreValue;
use namidb_storage::manifest::TextIndexDescriptor;
use namidb_storage::search_lsm::{
    select_search_read_plan, SearchLsmKind, SearchReadPlan, SearchSegmentFormat,
};
use namidb_storage::search_progress::{index_status, IndexState};
use namidb_storage::{NamespacePaths, NodeWriteRecord, WriterSession};
use object_store::memory::InMemory;
use object_store::ObjectStore;

const INDEX: &str = "art_ft";
const LABEL: &str = "Article";

fn schema() -> Schema {
    SchemaBuilder::new()
        .label(LabelDef {
            name: LABEL.into(),
            properties: vec![
                PropertyDef::new("title", DataType::Utf8, true).unwrap(),
                PropertyDef::new("body", DataType::Utf8, true).unwrap(),
                PropertyDef::new("cat", DataType::Utf8, true).unwrap(),
            ],
        })
        .unwrap()
        .build()
}

/// Deterministic pseudo-random words: a small vocabulary so terms repeat
/// across documents (real `df` values) plus a few topical words.
struct Words(u64);

impl Words {
    fn next(&mut self) -> String {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        const TOPICAL: [&str; 8] = [
            "graph", "storage", "object", "engine", "search", "index", "lexical", "vector",
        ];
        let pick = (self.0 >> 33) % 160;
        if pick < 8 {
            TOPICAL[pick as usize].to_string()
        } else {
            format!("w{}", pick - 8)
        }
    }

    fn text(&mut self, words: usize) -> String {
        (0..words)
            .map(|_| self.next())
            .collect::<Vec<_>>()
            .join(" ")
    }
}

fn article(title: &str, body: &str, cat: &str) -> NodeWriteRecord {
    let mut properties = BTreeMap::new();
    properties.insert("title".into(), CoreValue::Str(title.into()));
    properties.insert("body".into(), CoreValue::Str(body.into()));
    properties.insert("cat".into(), CoreValue::Str(cat.into()));
    NodeWriteRecord {
        properties,
        schema_version: 1,
        ..Default::default()
    }
}

fn formats(writer: &WriterSession) -> Vec<SearchSegmentFormat> {
    writer
        .snapshot()
        .manifest()
        .manifest
        .search_lsm
        .iter()
        .find(|state| state.kind == SearchLsmKind::Text && state.index_name == INDEX)
        .map(|state| {
            state
                .segments
                .iter()
                .map(|segment| segment.format)
                .collect()
        })
        .unwrap_or_default()
}

fn plan(writer: &WriterSession) -> SearchReadPlan {
    select_search_read_plan(
        &writer.snapshot().manifest().manifest,
        SearchLsmKind::Text,
        INDEX,
    )
}

/// A namespace whose full-text index has a V3 base, built the way a 3.4.x
/// engine built it: the nodes are flushed first, the index is registered,
/// and the next compaction rebuilds it logically inside the Nodes merge and
/// adopts that body as the generation's base before any flush starts one.
async fn legacy_v3_writer(
    name: &str,
    documents: usize,
) -> (
    WriterSession,
    Arc<dyn ObjectStore>,
    NamespacePaths,
    Vec<NodeId>,
    Words,
) {
    let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
    let paths = NamespacePaths::new("tenants", NamespaceId::new(name).unwrap());
    let mut writer = WriterSession::open(store.clone(), paths.clone())
        .await
        .unwrap();
    let mut words = Words(7);
    let mut ids = Vec::with_capacity(documents);
    for ordinal in 0..documents {
        let id = NodeId::new();
        let title = words.text(4);
        let body = words.text(30);
        writer
            .upsert_node(
                LABEL,
                id,
                &article(&title, &body, &format!("c{}", ordinal % 5)),
            )
            .unwrap();
        ids.push(id);
    }
    writer.flush(schema()).await.unwrap();
    writer
        .register_text_index(
            TextIndexDescriptor::new(
                INDEX.into(),
                LABEL.into(),
                vec!["title".into(), "body".into()],
            ),
            false,
        )
        .await
        .unwrap();
    writer.compact_l0(&schema()).await.unwrap();
    assert_eq!(
        formats(&writer),
        [SearchSegmentFormat::TextV3Base],
        "the fixture must reproduce a legacy V3 base"
    );
    assert!(matches!(
        plan(&writer),
        SearchReadPlan::ActiveLegacyBase { .. }
    ));
    (writer, store, paths, ids, words)
}

const QUERIES: [&str; 9] = [
    "graph storage",
    "w1 w2 w3",
    "engine",
    "\"object engine\"",
    "w1*",
    "lexical vector search",
    "w150 w17",
    "index",
    "graph \"search index\" w3*",
];

type Answers = Vec<(String, Option<usize>, Vec<(NodeId, f64)>)>;

/// Every query at several `k`, each required to be served natively (a
/// `None` would be the exact-scan fallback).
async fn native_answers(writer: &WriterSession) -> Answers {
    let mut answers = Vec::new();
    for query in QUERIES {
        for k in [Some(1), Some(3), Some(10), Some(50), None] {
            let hits = writer
                .snapshot()
                .text_search(INDEX, LABEL, &namidb_storage::text::parse_query(query), k)
                .await
                .unwrap()
                .unwrap_or_else(|| panic!("{query:?} k={k:?} fell back to the exact scan"));
            answers.push((query.to_string(), k, hits));
        }
    }
    answers
}

type Stats = Vec<(
    namidb_storage::text::TextQuery,
    namidb_storage::sst::text::v4::TextV4GlobalStats,
)>;

/// The corpus-wide statistics the selective filtered route scores with.
async fn corpus_stats(writer: &WriterSession) -> Stats {
    let mut stats = Vec::new();
    for query in QUERIES {
        stats.push(
            writer
                .snapshot()
                .text_query_corpus_stats(INDEX, LABEL, &namidb_storage::text::parse_query(query))
                .await
                .unwrap()
                .unwrap_or_else(|| panic!("{query:?} has no native corpus statistics")),
        );
    }
    stats
}

fn assert_same_answers(merged: &Answers, migrated: &Answers) {
    assert_eq!(merged.len(), migrated.len());
    for ((query, k, left), (_, _, right)) in merged.iter().zip(migrated) {
        let left_ids = left.iter().map(|(id, _)| *id).collect::<Vec<_>>();
        let right_ids = right.iter().map(|(id, _)| *id).collect::<Vec<_>>();
        assert_eq!(left_ids, right_ids, "{query:?} k={k:?}: ids differ");
        for ((id, a), (_, b)) in left.iter().zip(right) {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "{query:?} k={k:?}: score of {id:?} differs ({a} vs {b})"
            );
        }
    }
}

/// Updates (some toward the topical words), deletes and inserts, flushed as
/// one FT4 delta.
async fn mutate(
    writer: &mut WriterSession,
    ids: &mut Vec<NodeId>,
    words: &mut Words,
    round: usize,
) {
    for (offset, id) in ids.iter().skip(round * 7).take(5).enumerate() {
        let body = if offset % 2 == 0 {
            format!("graph storage engine {}", words.text(12))
        } else {
            words.text(25)
        };
        writer
            .upsert_node(LABEL, *id, &article(&words.text(3), &body, "c1"))
            .unwrap();
    }
    for id in ids.drain(40 + round * 3..43 + round * 3) {
        writer.tombstone_node(LABEL, id).unwrap();
    }
    for _ in 0..6 {
        let id = NodeId::new();
        writer
            .upsert_node(
                LABEL,
                id,
                &article(
                    "lexical graph",
                    &format!("object engine {}", words.text(20)),
                    "c2",
                ),
            )
            .unwrap();
        ids.push(id);
    }
    writer.flush(schema()).await.unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_v3_base_with_ft4_deltas_serves_natively_and_scores_exactly_like_its_ft4_rebuild() {
    let (mut writer, store, paths, mut ids, mut words) =
        legacy_v3_writer("legacy-v3-merge", 240).await;
    let lone = native_answers(&writer).await;

    // The regression: the first 3.5 flush appends an FT4 delta to the V3 base.
    mutate(&mut writer, &mut ids, &mut words, 0).await;
    assert_eq!(
        formats(&writer),
        [SearchSegmentFormat::TextV3Base, SearchSegmentFormat::TextV4]
    );
    assert!(
        matches!(plan(&writer), SearchReadPlan::ActiveSegments { .. }),
        "a V3 base with FT4 deltas is a native segment set, not a scan"
    );
    let status = index_status(
        &writer.snapshot().manifest().manifest,
        SearchLsmKind::Text,
        INDEX,
    );
    assert_eq!(status.state, IndexState::Online);
    assert_eq!(status.percent, 100.0);
    assert_eq!(status.migration_percent, Some(0.0), "migration pending");
    assert!(
        writer.search_compaction_due(),
        "the migration is planned whatever the delta debt"
    );
    let merged = native_answers(&writer).await;
    let merged_stats = corpus_stats(&writer).await;
    assert_ne!(
        lone.iter()
            .map(|(_, _, hits)| hits.clone())
            .collect::<Vec<_>>(),
        merged
            .iter()
            .map(|(_, _, hits)| hits.clone())
            .collect::<Vec<_>>(),
        "the delta changes the answers"
    );

    // Same snapshot after a reopen: still native.
    drop(writer);
    let mut writer = WriterSession::open(store.clone(), paths.clone())
        .await
        .unwrap();
    assert_same_answers(&merged, &native_answers(&writer).await);

    // The migration: one base rebuild, FT4, then nothing more to do.
    writer.compact_search_segments().await.unwrap();
    assert_eq!(formats(&writer), [SearchSegmentFormat::TextV4]);
    let status = index_status(
        &writer.snapshot().manifest().manifest,
        SearchLsmKind::Text,
        INDEX,
    );
    assert_eq!(status.state, IndexState::Online);
    assert_eq!(status.migration_percent, None);
    assert!(
        !writer.search_compaction_due(),
        "the migrated base is not rebuilt again"
    );
    let migrated = native_answers(&writer).await;
    assert_same_answers(&merged, &migrated);
    assert_eq!(merged_stats, corpus_stats(&writer).await);

    // Ordinary FT4 deltas on the migrated base keep working.
    mutate(&mut writer, &mut ids, &mut words, 1).await;
    assert_eq!(
        formats(&writer),
        [SearchSegmentFormat::TextV4, SearchSegmentFormat::TextV4]
    );
    native_answers(&writer).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_merged_v3_set_matches_its_rebuild_after_several_deltas() {
    let (mut writer, _store, _paths, mut ids, mut words) =
        legacy_v3_writer("legacy-v3-multi-delta", 200).await;
    for round in 0..3 {
        mutate(&mut writer, &mut ids, &mut words, round).await;
    }
    assert_eq!(formats(&writer)[0], SearchSegmentFormat::TextV3Base);
    assert_eq!(formats(&writer).len(), 4);
    let merged = native_answers(&writer).await;
    let merged_stats = corpus_stats(&writer).await;
    writer.compact_search_segments().await.unwrap();
    assert_eq!(formats(&writer), [SearchSegmentFormat::TextV4]);
    assert_same_answers(&merged, &native_answers(&writer).await);
    assert_eq!(merged_stats, corpus_stats(&writer).await);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_lone_v3_base_is_migrated_once_with_the_same_answers() {
    let (mut writer, _store, _paths, _ids, _words) = legacy_v3_writer("legacy-v3-lone", 150).await;
    let status = index_status(
        &writer.snapshot().manifest().manifest,
        SearchLsmKind::Text,
        INDEX,
    );
    assert_eq!(status.state, IndexState::Online);
    assert_eq!(status.migration_percent, Some(0.0));
    assert!(
        writer.search_compaction_due(),
        "a lone V3 base is migrated too"
    );
    let before = native_answers(&writer).await;
    writer.compact_search_segments().await.unwrap();
    assert_eq!(formats(&writer), [SearchSegmentFormat::TextV4]);
    assert!(!writer.search_compaction_due());
    assert_same_answers(&before, &native_answers(&writer).await);
}

/// An index created over existing nodes is BUILDING until a compaction
/// activates its generation. That must hold after a restart with no write
/// since: the activation is planned from the manifest alone, and the index
/// reports progress while it is pending.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_building_index_activates_after_a_restart_without_writes() {
    let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
    let paths = NamespacePaths::new("tenants", NamespaceId::new("building-reopen").unwrap());
    let mut writer = WriterSession::open(store.clone(), paths.clone())
        .await
        .unwrap();
    let mut words = Words(11);
    for ordinal in 0..120 {
        writer
            .upsert_node(
                LABEL,
                NodeId::new(),
                &article(
                    &words.text(4),
                    &words.text(20),
                    &format!("c{}", ordinal % 3),
                ),
            )
            .unwrap();
    }
    writer.flush(schema()).await.unwrap();
    writer
        .register_text_index(
            TextIndexDescriptor::new(
                INDEX.into(),
                LABEL.into(),
                vec!["title".into(), "body".into()],
            ),
            false,
        )
        .await
        .unwrap();
    assert!(writer.start_idle_search_generations().await.unwrap());
    let status = index_status(
        &writer.snapshot().manifest().manifest,
        SearchLsmKind::Text,
        INDEX,
    );
    assert_eq!(status.state, IndexState::Building);
    assert!(status.percent < 100.0);
    assert_eq!(status.migration_percent, None);

    drop(writer);
    let mut writer = WriterSession::open(store, paths).await.unwrap();
    assert!(
        writer.search_compaction_due(),
        "a Building generation is always planned for activation"
    );
    writer.compact_search_segments().await.unwrap();
    let status = index_status(
        &writer.snapshot().manifest().manifest,
        SearchLsmKind::Text,
        INDEX,
    );
    assert_eq!(status.state, IndexState::Online);
    assert_eq!(formats(&writer), [SearchSegmentFormat::TextV4]);
    assert!(!writer.search_compaction_due());
    native_answers(&writer).await;
}
