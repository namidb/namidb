//! `CREATE INDEX` on a property the schema does not type yet infers the type
//! from one live value. The server does that lookup on a published snapshot
//! before it takes the writer lock ([`PropertyTypeHint`]), so writes keep
//! flowing meanwhile; the commit re-validates the hint against the manifest
//! it builds on. Before 3.5.2 the lookup decoded the whole label under the
//! writer lock (a minute on a label of large embeddings, every write refused
//! meanwhile).

use std::collections::BTreeMap;
use std::sync::Arc;

use namidb_core::id::{NamespaceId, NodeId};
use namidb_core::value::Value as CoreValue;
use namidb_core::DataType;
use namidb_storage::{
    HintedDdlOutcome, NamespacePaths, NodeWriteRecord, PropertyTypeHint, WriterSession,
};
use object_store::memory::InMemory;
use object_store::ObjectStore;

fn doc(ordinal: u64, with_city: bool) -> NodeWriteRecord {
    let mut props = BTreeMap::new();
    props.insert("n".into(), CoreValue::I64(ordinal as i64));
    props.insert("blob".into(), CoreValue::Str("x".repeat(512)));
    if with_city {
        props.insert("city".into(), CoreValue::Str("quito".into()));
    }
    NodeWriteRecord {
        properties: props,
        schema_version: 1,
        ..Default::default()
    }
}

async fn writer(name: &str) -> WriterSession {
    let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
    let paths = NamespacePaths::new("tenants", NamespaceId::new(name).unwrap());
    let mut writer = WriterSession::open(store, paths).await.unwrap();
    // The property appears only on the last rows: the lookup has to look
    // past the rows without it.
    for ordinal in 0..64u64 {
        writer
            .upsert_node("Doc", NodeId::new(), &doc(ordinal, ordinal >= 60))
            .unwrap();
    }
    writer.commit_batch().await.unwrap();
    let schema = writer.schema().clone();
    writer.flush(schema).await.unwrap();
    writer
}

fn city(writer: &WriterSession) -> Option<namidb_core::PropertyDef> {
    writer
        .schema()
        .label("Doc")
        .and_then(|label| label.properties.iter().find(|p| p.name == "city"))
        .cloned()
}

#[tokio::test]
async fn a_hint_from_a_snapshot_types_the_index() {
    let mut writer = writer("hinted").await;
    let snapshot = writer.owned_snapshot();
    let hint = PropertyTypeHint::infer(&snapshot, "Doc", "city")
        .await
        .unwrap();
    assert_eq!(hint.inferred(), Some(&DataType::Utf8));
    // Writes land between the lookup and the commit; the hint still holds.
    writer
        .upsert_node("Doc", NodeId::new(), &doc(100, true))
        .unwrap();
    writer.commit_batch().await.unwrap();
    let outcome = writer
        .create_property_index_hinted(None, "Doc", "city", false, &hint)
        .await
        .unwrap();
    assert!(
        matches!(outcome, HintedDdlOutcome::Committed(_)),
        "{outcome:?}"
    );
    let city = city(&writer).expect("city declared");
    assert!(city.indexed && !city.type_pending, "{city:?}");
    assert_eq!(city.data_type, DataType::Utf8);
    // IF NOT EXISTS on the same hint is the no-op it always was.
    let again = writer
        .create_property_index_hinted(None, "Doc", "city", true, &hint)
        .await
        .unwrap();
    assert!(matches!(again, HintedDdlOutcome::Committed(_)));
}

#[tokio::test]
async fn a_type_declared_after_the_lookup_wins_over_the_hint() {
    let mut writer = writer("hint-declared").await;
    let hint = PropertyTypeHint::infer(&writer.owned_snapshot(), "Doc", "city")
        .await
        .unwrap();
    // Another DDL types the property between the lookup and the commit.
    writer
        .create_composite_index_named(
            Some("doc_city_n"),
            "Doc",
            &["city".to_string(), "n".to_string()],
            false,
        )
        .await
        .unwrap();
    let before = writer.snapshot().manifest().manifest.version;
    let outcome = writer
        .create_property_index_hinted(None, "Doc", "city", false, &hint)
        .await
        .unwrap();
    assert_eq!(outcome, HintedDdlOutcome::Committed(before + 1));
    let city = city(&writer).expect("city declared");
    assert!(city.indexed && !city.type_pending, "{city:?}");
    assert_eq!(city.data_type, DataType::Utf8);
}

#[tokio::test]
async fn a_hint_that_saw_no_value_is_looked_up_again_under_the_lock() {
    let mut writer = writer("hint-empty").await;
    let hint = PropertyTypeHint::infer(&writer.owned_snapshot(), "Doc", "rank")
        .await
        .unwrap();
    assert_eq!(hint.inferred(), None);
    // The first value arrives after the lookup.
    let mut late = doc(7, false);
    late.properties.insert("rank".into(), CoreValue::I64(3));
    writer.upsert_node("Doc", NodeId::new(), &late).unwrap();
    writer.commit_batch().await.unwrap();
    writer
        .create_property_index_hinted(None, "Doc", "rank", false, &hint)
        .await
        .unwrap();
    let rank = writer
        .schema()
        .label("Doc")
        .and_then(|label| label.properties.iter().find(|p| p.name == "rank"))
        .cloned()
        .expect("rank declared");
    assert!(rank.indexed && !rank.type_pending, "{rank:?}");
    assert_eq!(rank.data_type, DataType::Int64);
}
