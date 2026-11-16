//! A flush that finds a search index at its segment cap must leave a signal
//! that a compaction is due.
//!
//! Each flush adds one delta segment per changed search index and refuses to
//! add one past `NAMIDB_SEARCH_LSM_MAX_SEGMENTS`. Servers used to request a
//! compaction after a flush only on the L0 high-water mark, so a namespace
//! whose search index reached the cap failed every flush until the periodic
//! compaction tick fired, and writes queued behind the retried flush. The
//! flush loop now asks `search_compaction_due()` after every flush, including
//! a failed one.
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

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_search_index_at_its_segment_cap_reports_compaction_due() {
    // The only test in this binary: the variable is process-wide.
    std::env::set_var("NAMIDB_SEARCH_LSM_MAX_SEGMENTS", "3");

    let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
    let paths = NamespacePaths::new("tenants", NamespaceId::new("segment-cap").unwrap());
    let mut writer = WriterSession::open(store, paths).await.unwrap();
    writer
        .register_text_index(
            TextIndexDescriptor::new("doc_ft".into(), "Doc".into(), vec!["body".into()]),
            false,
        )
        .await
        .unwrap();
    assert!(!writer.search_compaction_due(), "no segments yet");

    let mut capped = None;
    for round in 0..10 {
        writer
            .upsert_node("Doc", NodeId::new(), &doc(&format!("ronda {round} texto")))
            .unwrap();
        match writer.flush(schema()).await {
            Ok(_) => {}
            Err(error) => {
                capped = Some((round, error.to_string()));
                break;
            }
        }
    }
    let (round, error) = capped.expect("the segment cap must stop a flush within 10 rounds");
    assert!(error.contains("active segments"), "round {round}: {error}");
    assert!(
        writer.search_compaction_due(),
        "a flush refused at the segment cap must report a compaction as due"
    );

    writer.compact_l0(&schema()).await.unwrap();
    writer
        .flush(schema())
        .await
        .expect("the flush refused at the cap succeeds once compaction ran");
    let manifest = writer.snapshot().manifest().manifest.clone();
    namidb_storage::search_lsm::validate_search_lsm(&manifest).unwrap();
}
