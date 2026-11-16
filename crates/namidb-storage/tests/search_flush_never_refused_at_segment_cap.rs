//! A flush is never refused by a search index's segment cap.
//!
//! Each flush adds one delta segment per changed search index, and a
//! generation may hold at most `NAMIDB_SEARCH_LSM_MAX_SEGMENTS` of them. Up to
//! 3.4.0 a flush that found an index at the cap failed ("… has N active
//! segments …") and waited for a compaction, so under sustained writes the
//! memtable filled, writes stalled and clients saw 503 "writer is busy". A
//! flush at the cap now consolidates that index's deltas first (a bounded,
//! search-only compaction that never merges the node tables) and then
//! succeeds; the cap is never exceeded.
#![cfg(feature = "text-index")]

use std::collections::BTreeMap;
use std::sync::Arc;

use namidb_core::id::{NamespaceId, NodeId};
use namidb_core::schema::{DataType, LabelDef, PropertyDef, Schema, SchemaBuilder};
use namidb_core::value::Value as CoreValue;
use namidb_storage::manifest::TextIndexDescriptor;
use namidb_storage::{NamespacePaths, NodeWriteRecord, WriterSession};
use object_store::memory::InMemory;
use object_store::ObjectStore;

const CAP: usize = 3;

fn schema() -> Schema {
    SchemaBuilder::new()
        .label(LabelDef {
            name: "Doc".into(),
            properties: vec![PropertyDef::new("body", DataType::Utf8, true).unwrap()],
        })
        .unwrap()
        .build()
}

fn doc(body: &str) -> NodeWriteRecord {
    let mut properties = BTreeMap::new();
    properties.insert("body".into(), CoreValue::Str(body.into()));
    NodeWriteRecord {
        properties,
        schema_version: 1,
        ..Default::default()
    }
}

fn active_segments(writer: &WriterSession) -> usize {
    let manifest = writer.snapshot().manifest().manifest.clone();
    namidb_storage::search_lsm::validate_search_lsm(&manifest).unwrap();
    manifest
        .search_lsm
        .iter()
        .map(|state| state.segments.len())
        .max()
        .unwrap_or(0)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_flush_at_the_segment_cap_compacts_instead_of_failing() {
    // The only test in this binary: the variable is process-wide.
    std::env::set_var("NAMIDB_SEARCH_LSM_MAX_SEGMENTS", CAP.to_string());

    let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
    let paths = NamespacePaths::new("tenants", NamespaceId::new("segment-cap").unwrap());
    let mut writer = WriterSession::open(store.clone(), paths.clone())
        .await
        .unwrap();
    writer
        .register_text_index(
            TextIndexDescriptor::new("doc_ft".into(), "Doc".into(), vec!["body".into()]),
            false,
        )
        .await
        .unwrap();
    assert!(!writer.search_compaction_due(), "no segments yet");

    let mut ids = Vec::new();
    for round in 0..(CAP * 4) {
        let id = NodeId::new();
        ids.push(id);
        writer
            .upsert_node("Doc", id, &doc(&format!("ronda {round} texto")))
            .unwrap();
        // Rewrite an earlier document too, so deltas carry superseded rows.
        if round > 0 {
            writer
                .upsert_node(
                    "Doc",
                    ids[round - 1],
                    &doc(&format!("ronda {} texto", round - 1)),
                )
                .unwrap();
        }
        writer
            .flush(schema())
            .await
            .unwrap_or_else(|error| panic!("round {round}: the flush was refused: {error}"));
        let segments = active_segments(&writer);
        assert!(
            segments <= CAP,
            "round {round}: {segments} active segments exceed the cap {CAP}"
        );
    }

    // Every document is still found, exactly once, natively.
    let query = namidb_storage::text::parse_query("texto");
    let hits = writer
        .snapshot()
        .text_search("doc_ft", "Doc", &query, None)
        .await
        .unwrap()
        .expect("the index serves natively after the inline compactions");
    assert_eq!(hits.len(), ids.len());

    // An off-lock style flush (no inline compaction) goes past the configured
    // ceiling instead of failing, up to the protocol ceiling every reader
    // accepts; a later compaction brings the generation back down.
    for round in 0..3 {
        let id = NodeId::new();
        ids.push(id);
        writer
            .upsert_node("Doc", id, &doc(&format!("extra {round} texto")))
            .unwrap();
        writer
            .try_flush(schema())
            .await
            .unwrap_or_else(|error| panic!("extra round {round}: refused: {error}"));
    }
    let over = active_segments(&writer);
    assert!(
        over > CAP,
        "try_flush did not go past the configured ceiling"
    );
    assert!(over <= namidb_storage::search_lsm::MAX_ACTIVE_SEARCH_SEGMENTS);
    writer.compact_search_segments().await.unwrap();
    assert!(active_segments(&writer) < over);
    let hits = writer
        .snapshot()
        .text_search("doc_ft", "Doc", &query, None)
        .await
        .unwrap()
        .expect("the index serves natively after the over-ceiling flushes");
    assert_eq!(hits.len(), ids.len());

    // And a reopen reads the same state.
    drop(writer);
    let reopened = WriterSession::open(store, paths).await.unwrap();
    let hits = reopened
        .snapshot()
        .text_search("doc_ft", "Doc", &query, None)
        .await
        .unwrap()
        .expect("the reopened index serves natively");
    assert_eq!(hits.len(), ids.len());
}
