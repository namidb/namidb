//! A search index created over existing nodes starts as a Building
//! generation. Up to 3.4.1 only a logical rebuild inside a Nodes merge could
//! activate it, and that rebuild was abandoned whenever a flush appended to
//! the generation while it ran: under sustained writes the generation stayed
//! Building forever, accumulated one delta per flush up to the segment
//! ceiling, and blocked flushes. A physical base build now activates it and
//! rebases over the flushes that land meanwhile.
#![cfg(all(feature = "vector-index", feature = "text-index"))]

use std::collections::BTreeMap;
use std::sync::Arc;

use namidb_core::id::{NamespaceId, NodeId};
use namidb_core::schema::{DataType, LabelDef, PropertyDef, Schema, SchemaBuilder};
use namidb_core::value::Value as CoreValue;
use namidb_storage::manifest::TextIndexDescriptor;
use namidb_storage::search_lsm::{SearchLsmKind, SearchLsmStatus};
use namidb_storage::{NamespacePaths, NodeWriteRecord, WriterSession};
use object_store::memory::InMemory;
use object_store::ObjectStore;

fn schema(status_indexed: bool) -> Schema {
    let mut status = PropertyDef::new("status", DataType::Utf8, true).unwrap();
    if status_indexed {
        status = status.with_indexed(true);
    }
    SchemaBuilder::new()
        .label(LabelDef {
            name: "Doc".into(),
            properties: vec![
                PropertyDef::new("body", DataType::Utf8, true).unwrap(),
                PropertyDef::new("title", DataType::Utf8, true).unwrap(),
                status,
            ],
        })
        .unwrap()
        .build()
}

fn doc(ordinal: u64) -> NodeWriteRecord {
    let mut properties = BTreeMap::new();
    properties.insert(
        "body".into(),
        CoreValue::Str(format!("body common text{ordinal}")),
    );
    properties.insert(
        "title".into(),
        CoreValue::Str(format!("title common t{ordinal}")),
    );
    properties.insert("status".into(), CoreValue::Str("open".into()));
    NodeWriteRecord {
        properties,
        schema_version: 1,
        ..Default::default()
    }
}

fn status(writer: &WriterSession, name: &str) -> Option<SearchLsmStatus> {
    writer
        .snapshot()
        .manifest()
        .manifest
        .search_lsm
        .iter()
        .find(|state| state.kind == SearchLsmKind::Text && state.index_name == name)
        .map(|state| state.status)
}

async fn native_hits(writer: &WriterSession, index: &str, query: &str) -> Option<usize> {
    writer
        .snapshot()
        .text_search(
            index,
            "Doc",
            &namidb_storage::text::parse_query(query),
            None,
        )
        .await
        .unwrap()
        .map(|hits| hits.len())
}

async fn writer_with_corpus(
    name: &str,
    docs: u64,
) -> (WriterSession, Arc<dyn ObjectStore>, NamespacePaths) {
    let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
    let paths = NamespacePaths::new("tenants", NamespaceId::new(name).unwrap());
    let mut writer = WriterSession::open(store.clone(), paths.clone())
        .await
        .unwrap();
    writer
        .register_text_index(
            TextIndexDescriptor::new("body_ft".into(), "Doc".into(), vec!["body".into()]),
            false,
        )
        .await
        .unwrap();
    for ordinal in 0..docs {
        writer
            .upsert_node("Doc", NodeId::new(), &doc(ordinal))
            .unwrap();
        if ordinal % 10 == 9 {
            writer.flush(schema(false)).await.unwrap();
        }
    }
    writer.flush(schema(false)).await.unwrap();
    writer.compact_l0(&schema(false)).await.unwrap();
    assert_eq!(status(&writer, "body_ft"), Some(SearchLsmStatus::Active));
    (writer, store, paths)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn an_index_created_over_existing_nodes_activates_under_concurrent_flushes() {
    let (mut writer, store, paths) = writer_with_corpus("building-activation", 30).await;
    writer
        .register_text_index(
            TextIndexDescriptor::new("title_ft".into(), "Doc".into(), vec!["title".into()]),
            false,
        )
        .await
        .unwrap();
    let mut next = 30u64;
    writer
        .upsert_node("Doc", NodeId::new(), &doc(next))
        .unwrap();
    next += 1;
    writer.flush(schema(false)).await.unwrap();
    assert_eq!(
        status(&writer, "title_ft"),
        Some(SearchLsmStatus::Building),
        "an index created over existing nodes starts Building"
    );
    assert!(writer.search_compaction_due(), "activation is scheduled");

    // Prepare the activation off-lock, then let flushes append to the
    // generation (and to the healthy one) before it installs.
    let basis = writer.compaction_basis();
    let prepared = basis.prepare_search_only(&schema(false)).await.unwrap();
    for _ in 0..3 {
        writer
            .upsert_node("Doc", NodeId::new(), &doc(next))
            .unwrap();
        next += 1;
        writer.flush(schema(false)).await.unwrap();
    }
    let outcome = writer.install_prepared_compaction(prepared).await.unwrap();
    assert!(outcome.new_ssts_written > 0);
    assert_eq!(status(&writer, "title_ft"), Some(SearchLsmStatus::Active));
    assert_eq!(status(&writer, "body_ft"), Some(SearchLsmStatus::Active));
    namidb_storage::search_lsm::validate_search_lsm(&writer.snapshot().manifest().manifest)
        .unwrap();
    assert!(
        !writer.search_compaction_due() || {
            // Only routine delta consolidation may remain due.
            let manifest = writer.snapshot().manifest().manifest.clone();
            manifest
                .search_lsm
                .iter()
                .all(|state| state.status == SearchLsmStatus::Active)
        }
    );

    // Every document — before the index, during the build, after it.
    assert_eq!(
        native_hits(&writer, "title_ft", "common").await,
        Some(next as usize)
    );
    assert_eq!(native_hits(&writer, "title_ft", "t3").await, Some(1));
    assert_eq!(native_hits(&writer, "title_ft", "t33").await, Some(1));
    assert_eq!(
        native_hits(&writer, "body_ft", "common").await,
        Some(next as usize)
    );

    // The activated generation keeps serving across flushes, a Nodes merge
    // and a reopen.
    writer
        .upsert_node("Doc", NodeId::new(), &doc(next))
        .unwrap();
    next += 1;
    writer.flush(schema(false)).await.unwrap();
    writer.compact_l0(&schema(false)).await.unwrap();
    assert_eq!(
        native_hits(&writer, "title_ft", "common").await,
        Some(next as usize)
    );
    drop(writer);
    let reopened = WriterSession::open(store, paths).await.unwrap();
    assert_eq!(
        native_hits(&reopened, "title_ft", "common").await,
        Some(next as usize)
    );
}

fn two_label_schema(status_indexed: bool) -> Schema {
    let mut status = PropertyDef::new("status", DataType::Utf8, true).unwrap();
    if status_indexed {
        status = status.with_indexed(true);
    }
    SchemaBuilder::new()
        .label(LabelDef {
            name: "Doc".into(),
            properties: vec![
                PropertyDef::new("body", DataType::Utf8, true).unwrap(),
                PropertyDef::new("title", DataType::Utf8, true).unwrap(),
                status,
            ],
        })
        .unwrap()
        .label(LabelDef {
            name: "Note".into(),
            properties: vec![PropertyDef::new("body", DataType::Utf8, true).unwrap()],
        })
        .unwrap()
        .build()
}

/// A schema change that makes one text index's native-filter set stale
/// retires only that generation at the next flush: an index on another label
/// keeps its Active generation (3.4.1 restarted every generation as
/// Building, and none could activate under sustained writes).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn one_stale_generation_does_not_reset_the_others() {
    let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
    let paths = NamespacePaths::new("tenants", NamespaceId::new("stale-isolation").unwrap());
    let mut writer = WriterSession::open(store, paths).await.unwrap();
    for (name, label) in [("doc_ft", "Doc"), ("note_ft", "Note")] {
        writer
            .register_text_index(
                TextIndexDescriptor::new(name.into(), label.into(), vec!["body".into()]),
                false,
            )
            .await
            .unwrap();
    }
    for ordinal in 0..10u64 {
        writer
            .upsert_node("Doc", NodeId::new(), &doc(ordinal))
            .unwrap();
        writer
            .upsert_node("Note", NodeId::new(), &doc(ordinal))
            .unwrap();
    }
    writer.flush(two_label_schema(false)).await.unwrap();
    writer.compact_l0(&two_label_schema(false)).await.unwrap();
    let generation = |writer: &WriterSession, name: &str| {
        writer
            .snapshot()
            .manifest()
            .manifest
            .search_lsm
            .iter()
            .find(|state| state.index_name == name)
            .map(|state| (state.generation_id, state.status))
    };
    let note_before = generation(&writer, "note_ft").unwrap();
    assert_eq!(note_before.1, SearchLsmStatus::Active);

    // Model an older writer that committed the schema without retiring the
    // generation whose native-filter set it changed (only Doc's).
    writer
        .commit_schema_without_search_retirement_for_test(two_label_schema(true))
        .await
        .unwrap();
    writer.upsert_node("Doc", NodeId::new(), &doc(10)).unwrap();
    writer.upsert_node("Note", NodeId::new(), &doc(10)).unwrap();
    writer.flush(two_label_schema(true)).await.unwrap();
    assert_eq!(
        generation(&writer, "note_ft"),
        Some(note_before),
        "the healthy generation must survive another index going stale"
    );
    assert_eq!(
        generation(&writer, "doc_ft").map(|(_, status)| status),
        Some(SearchLsmStatus::Building)
    );
    writer.compact_search_segments().await.unwrap();
    assert_eq!(
        generation(&writer, "doc_ft").map(|(_, status)| status),
        Some(SearchLsmStatus::Active)
    );
    assert_eq!(native_hits_label(&writer, "doc_ft", "Doc").await, Some(11));
    assert_eq!(
        native_hits_label(&writer, "note_ft", "Note").await,
        Some(11)
    );
}

async fn native_hits_label(writer: &WriterSession, index: &str, label: &str) -> Option<usize> {
    writer
        .snapshot()
        .text_search(
            index,
            label,
            &namidb_storage::text::parse_query("common"),
            None,
        )
        .await
        .unwrap()
        .map(|hits| hits.len())
}

/// Per-index precondition. An index registered over existing nodes has no
/// generation yet, so the next Nodes merge plans a logical rebuild of it.
/// When a flush lands while that merge is prepared (it appends to the healthy
/// index and starts the new one), 3.4.1 abandoned the whole compaction: the
/// Nodes tables were never merged and the rebuild was redone, and abandoned,
/// on every pass. Now only the conflicting index's rebuilt body is
/// discarded; the merge and every other index install.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_flush_during_a_rebuild_discards_only_that_index() {
    let (mut writer, _store, _paths) = writer_with_corpus("per-index-precondition", 30).await;
    // Fresh L0 so the next compaction has a Nodes merge to do.
    for ordinal in 30..40u64 {
        writer
            .upsert_node("Doc", NodeId::new(), &doc(ordinal))
            .unwrap();
    }
    writer.flush(schema(false)).await.unwrap();
    writer
        .register_text_index(
            TextIndexDescriptor::new("title_ft".into(), "Doc".into(), vec!["title".into()]),
            false,
        )
        .await
        .unwrap();
    assert_eq!(
        status(&writer, "title_ft"),
        None,
        "no generation before a flush"
    );

    let basis = writer.compaction_basis();
    let prepared = basis.prepare(&schema(false)).await.unwrap();
    writer.upsert_node("Doc", NodeId::new(), &doc(40)).unwrap();
    writer.flush(schema(false)).await.unwrap();
    let outcome = writer
        .install_prepared_compaction(prepared)
        .await
        .expect("a flush during the rebuild must not abandon the Nodes merge");
    assert!(outcome.source_ssts_removed > 0, "the Nodes merge installed");
    let manifest = writer.snapshot().manifest().manifest.clone();
    namidb_storage::search_lsm::validate_search_lsm(&manifest).unwrap();
    assert_eq!(status(&writer, "body_ft"), Some(SearchLsmStatus::Active));
    assert_eq!(native_hits(&writer, "body_ft", "common").await, Some(41));

    // The discarded index activates through the physical path.
    writer.compact_search_segments().await.unwrap();
    assert_eq!(status(&writer, "title_ft"), Some(SearchLsmStatus::Active));
    assert_eq!(native_hits(&writer, "title_ft", "common").await, Some(41));
}

fn generation_segments(writer: &WriterSession, name: &str) -> usize {
    writer
        .snapshot()
        .manifest()
        .manifest
        .search_lsm
        .iter()
        .find(|state| state.kind == SearchLsmKind::Text && state.index_name == name)
        .map(|state| state.segments.len())
        .unwrap()
}

/// A Building generation gains one delta per flush until its activation
/// installs. Delta runs merge those deltas meanwhile (only past the prefix
/// the activation in flight replaces), and a run whose activation installs
/// first still installs after it instead of being abandoned: in 3.4.1 a
/// Building generation's deltas were never merged and it reached the segment
/// ceiling, then held flushes back.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn delta_runs_merge_a_building_generation_and_survive_its_activation() {
    let (mut writer, _store, _paths) = writer_with_corpus("building-deltas", 30).await;
    writer
        .register_text_index(
            TextIndexDescriptor::new("title_ft".into(), "Doc".into(), vec!["title".into()]),
            false,
        )
        .await
        .unwrap();
    let mut next = 30u64;
    let mut flush_one = |writer: &mut WriterSession| {
        writer
            .upsert_node("Doc", NodeId::new(), &doc(next))
            .unwrap();
        next += 1;
    };
    for _ in 0..6 {
        flush_one(&mut writer);
        writer.flush(schema(false)).await.unwrap();
    }
    assert_eq!(status(&writer, "title_ft"), Some(SearchLsmStatus::Building));
    let building_deltas = generation_segments(&writer, "title_ft");
    assert!(
        building_deltas >= 6,
        "one delta per flush: {building_deltas}"
    );

    // A delta pass alone merges the Building generation's deltas.
    let deltas = writer
        .compaction_basis()
        .prepare_search_deltas(&schema(false))
        .await
        .unwrap();
    assert!(
        !deltas.is_noop(),
        "a Building generation's deltas are merged"
    );
    writer.install_prepared_compaction(deltas).await.unwrap();
    assert_eq!(status(&writer, "title_ft"), Some(SearchLsmStatus::Building));
    let merged = generation_segments(&writer, "title_ft");
    assert!(merged < building_deltas, "{merged} >= {building_deltas}");

    // The activation is prepared over the current segments; flushes land
    // and a delta pass merges only the deltas past that prefix.
    let basis = writer.compaction_basis();
    let protected = basis.planned_search_bases();
    assert!(
        protected
            .iter()
            .any(|((kind, name), _)| *kind == SearchLsmKind::Text && name == "title_ft"),
        "the activation is planned: {protected:?}"
    );
    let activation = basis.prepare_search_bases(&schema(false)).await.unwrap();
    for _ in 0..6 {
        flush_one(&mut writer);
        writer.flush(schema(false)).await.unwrap();
    }
    let tail = writer
        .compaction_basis()
        .protecting_search_prefixes(protected)
        .prepare_search_deltas(&schema(false))
        .await
        .unwrap();
    assert!(
        !tail.is_noop(),
        "the deltas past the protected prefix merge"
    );

    // The activation installs first; the delta run rebases after the new
    // base instead of being abandoned.
    writer
        .install_prepared_compaction(activation)
        .await
        .unwrap();
    assert_eq!(status(&writer, "title_ft"), Some(SearchLsmStatus::Active));
    let before_tail = generation_segments(&writer, "title_ft");
    let outcome = writer.install_prepared_compaction(tail).await.unwrap();
    assert!(outcome.source_ssts_removed > 0);
    let after_tail = generation_segments(&writer, "title_ft");
    assert!(after_tail < before_tail, "{after_tail} >= {before_tail}");
    assert_eq!(status(&writer, "title_ft"), Some(SearchLsmStatus::Active));
    assert_eq!(native_hits(&writer, "title_ft", "common").await, Some(42));
    assert_eq!(native_hits(&writer, "title_ft", "t41").await, Some(1));
    assert_eq!(native_hits(&writer, "body_ft", "common").await, Some(42));

    // A Nodes merge and a full search pass keep every document.
    writer.compact_l0(&schema(false)).await.unwrap();
    writer.compact_search_segments().await.unwrap();
    assert_eq!(native_hits(&writer, "title_ft", "common").await, Some(42));
}

/// A delta consolidation does not depend on the schema beyond its own
/// index's catalog, so a new label or property written meanwhile (which
/// changes the schema) no longer abandons it. In 3.4.1 it was abandoned
/// ("schema changed in manifest vN").
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_delta_run_installs_across_an_unrelated_schema_change() {
    let (mut writer, _store, _paths) = writer_with_corpus("delta-schema-drift", 30).await;
    for ordinal in 30..36 {
        writer
            .upsert_node("Doc", NodeId::new(), &doc(ordinal))
            .unwrap();
        writer.flush(schema(false)).await.unwrap();
    }
    let before = generation_segments(&writer, "body_ft");
    let deltas = writer
        .compaction_basis()
        .prepare_search_deltas(&schema(false))
        .await
        .unwrap();
    assert!(
        !deltas.is_noop(),
        "a delta run is due over {before} segments"
    );

    let mut widened = schema(false);
    widened.labels.insert(
        "Note".into(),
        LabelDef {
            name: "Note".into(),
            properties: vec![PropertyDef::new("text", DataType::Utf8, true).unwrap()],
        },
    );
    let mut note = BTreeMap::new();
    note.insert("text".into(), CoreValue::Str("note".into()));
    writer
        .upsert_node(
            "Note",
            NodeId::new(),
            &NodeWriteRecord {
                properties: note,
                schema_version: 1,
                ..Default::default()
            },
        )
        .unwrap();
    writer.flush(widened.clone()).await.unwrap();
    assert!(writer
        .snapshot()
        .manifest()
        .manifest
        .schema
        .labels
        .contains_key("Note"));

    writer.install_prepared_compaction(deltas).await.unwrap();
    assert!(generation_segments(&writer, "body_ft") < before);
    assert_eq!(native_hits(&writer, "body_ft", "common").await, Some(36));
}

fn vector_status(writer: &WriterSession, name: &str) -> Option<SearchLsmStatus> {
    writer
        .snapshot()
        .manifest()
        .manifest
        .search_lsm
        .iter()
        .find(|state| state.kind == SearchLsmKind::Vector && state.index_name == name)
        .map(|state| state.status)
}

fn bare_list(ordinal: u64, dim: usize) -> CoreValue {
    // A JSON array parameter is stored as a list of floats when no vector
    // index covers the property at write time. Integers can appear in it too.
    CoreValue::List(
        (0..dim)
            .map(|axis| {
                if axis == 0 {
                    CoreValue::I64(1)
                } else {
                    CoreValue::F64(((ordinal * 31 + axis as u64 * 7) % 97) as f64 / 97.0)
                }
            })
            .collect(),
    )
}

/// The embeddings were written as bare lists BEFORE the vector index existed,
/// then the index was created and nothing was written afterwards. Up to 3.4.1
/// no flush ever started its generation (nothing to flush), and the build
/// skipped bare lists anyway: the index never served and every search
/// scanned the label. The idle generation now starts at the next flush tick,
/// its activation base coerces lists of the declared dimension, and both
/// search indexes created after the data activate.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_vector_index_created_after_bare_list_embeddings_activates_while_idle() {
    const DIM: usize = 8;
    let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
    let paths = NamespacePaths::new("tenants", NamespaceId::new("late-vector").unwrap());
    let mut writer = WriterSession::open(store.clone(), paths.clone())
        .await
        .unwrap();
    let mut ids = Vec::new();
    for ordinal in 0..40u64 {
        let mut record = doc(ordinal);
        match ordinal {
            // A wrong-dimension list is not an embedding of this index.
            36 | 37 => {
                record
                    .properties
                    .insert("embedding".into(), bare_list(ordinal, DIM - 1));
            }
            // No embedding at all.
            38 | 39 => {}
            _ => {
                record
                    .properties
                    .insert("embedding".into(), bare_list(ordinal, DIM));
            }
        }
        let id = NodeId::new();
        writer.upsert_node("Doc", id, &record).unwrap();
        ids.push(id);
        if ordinal % 10 == 9 {
            writer.flush(schema(false)).await.unwrap();
        }
    }
    writer.flush(schema(false)).await.unwrap();
    writer.compact_l0(&schema(false)).await.unwrap();

    writer
        .register_vector_index(
            namidb_storage::manifest::VectorIndexDescriptor {
                name: "doc_emb".into(),
                label: "Doc".into(),
                property: "embedding".into(),
                dim: DIM as u32,
                metric: namidb_storage::manifest::VectorMetric::Cosine,
                r: 16,
                l_build: 32,
                alpha: 1.2,
                quantization: namidb_storage::manifest::VectorQuantization::None,
            },
            false,
        )
        .await
        .unwrap();
    writer
        .register_text_index(
            TextIndexDescriptor::new("title_ft".into(), "Doc".into(), vec!["title".into()]),
            false,
        )
        .await
        .unwrap();
    // Idle: the periodic flush finds nothing to flush, but starts both
    // generations.
    writer.flush(schema(false)).await.unwrap();
    assert_eq!(
        vector_status(&writer, "doc_emb"),
        Some(SearchLsmStatus::Building)
    );
    assert_eq!(status(&writer, "title_ft"), Some(SearchLsmStatus::Building));
    namidb_storage::search_lsm::validate_search_lsm(&writer.snapshot().manifest().manifest)
        .unwrap();
    // What `SHOW INDEXES` and `/metrics` report while it builds.
    {
        use namidb_storage::search_progress::{index_state, IndexState, IndexStateCounts};
        let snapshot = writer.snapshot();
        let manifest = &snapshot.manifest().manifest;
        assert_eq!(
            index_state(manifest, SearchLsmKind::Vector, "doc_emb"),
            (IndexState::Building, 0.0)
        );
        assert_eq!(
            index_state(manifest, SearchLsmKind::Text, "title_ft").0,
            IndexState::Building
        );
        let counts = IndexStateCounts::of(manifest);
        assert_eq!((counts.vector_building, counts.vector_online), (1, 0));
        assert_eq!((counts.text_building, counts.text_online), (1, 0));
    }
    // A second idle tick changes nothing.
    let version = writer.snapshot().manifest_version();
    writer.flush(schema(false)).await.unwrap();
    assert_eq!(writer.snapshot().manifest_version(), version);
    assert!(writer.search_compaction_due(), "activation is scheduled");

    let basis = writer.compaction_basis();
    let prepared = basis.prepare_search_only(&schema(false)).await.unwrap();
    writer.install_prepared_compaction(prepared).await.unwrap();
    assert_eq!(
        vector_status(&writer, "doc_emb"),
        Some(SearchLsmStatus::Active)
    );
    assert_eq!(status(&writer, "title_ft"), Some(SearchLsmStatus::Active));
    namidb_storage::search_lsm::validate_search_lsm(&writer.snapshot().manifest().manifest)
        .unwrap();
    {
        use namidb_storage::search_progress::{index_state, IndexState, IndexStateCounts};
        let snapshot = writer.snapshot();
        let manifest = &snapshot.manifest().manifest;
        assert_eq!(
            index_state(manifest, SearchLsmKind::Vector, "doc_emb"),
            (IndexState::Online, 100.0)
        );
        let counts = IndexStateCounts::of(manifest);
        assert_eq!((counts.vector_building, counts.vector_online), (0, 1));
        assert_eq!((counts.text_building, counts.text_online), (0, 1));
    }

    // Every bare list of the declared dimension is indexed, the rest is not.
    let target = 5usize;
    let CoreValue::List(items) = bare_list(target as u64, DIM) else {
        unreachable!()
    };
    let query = items
        .iter()
        .map(|item| match item {
            CoreValue::F64(value) => *value as f32,
            CoreValue::I64(value) => *value as f32,
            _ => unreachable!(),
        })
        .collect::<Vec<_>>();
    let hits = writer
        .snapshot()
        .try_vector_search("doc_emb", &query, 100, 200)
        .await
        .unwrap()
        .expect("the activated index serves natively");
    assert_eq!(hits.len(), 36, "{hits:?}");
    assert_eq!(hits[0].0, ids[target]);
    assert!(hits.iter().all(|(id, _)| !ids[36..].contains(id)));
    assert_eq!(native_hits(&writer, "title_ft", "common").await, Some(40));

    // A later write and a reopen keep it serving.
    let mut record = doc(40);
    record
        .properties
        .insert("embedding".into(), bare_list(40, DIM));
    writer.upsert_node("Doc", NodeId::new(), &record).unwrap();
    writer.flush(schema(false)).await.unwrap();
    drop(writer);
    let reopened = WriterSession::open(store, paths).await.unwrap();
    let hits = reopened
        .snapshot()
        .try_vector_search("doc_emb", &query, 100, 200)
        .await
        .unwrap()
        .expect("still native after a reopen");
    assert_eq!(hits.len(), 37);
}
