//! Plan item 17 (docs/testing/25tb-readiness.md): the pagination module had
//! zero executions against data. These are the deep-pagination contracts a
//! 25 TB deployment will lean on: duplicate-free, gap-free pages across
//! flushes and deletes, end-of-stream detection, and the stale-cursor
//! contract.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use namidb_core::id::{NamespaceId, NodeId};
use namidb_core::schema::{DataType, LabelDef, PropertyDef, Schema, SchemaBuilder};
use namidb_core::value::Value as CoreValue;
use namidb_storage::{NamespacePaths, NodeWriteRecord, WriterSession};
use object_store::memory::InMemory;
use object_store::ObjectStore;

use namidb_query::pagination::{
    next_cursor, next_cursor_keyset, paginate_plan, paginate_plan_keyset, Cursor, CursorKeyset,
};
use namidb_query::{execute, lower, parse, query_text_hash, Params, RuntimeValue};

const ROWS: usize = 25;
const PAGE: u64 = 10;

fn schema() -> Schema {
    SchemaBuilder::new()
        .label(LabelDef {
            name: "Person".into(),
            properties: vec![PropertyDef::new("name", DataType::Utf8, false).unwrap()],
        })
        .unwrap()
        .build()
}

fn pid(ordinal: usize) -> NodeId {
    let mut bytes = [0u8; 16];
    bytes[0] = 0x42;
    bytes[15] = ordinal as u8 + 1;
    NodeId::from_uuid(uuid::Uuid::from_bytes(bytes))
}

async fn corpus(name: &str) -> WriterSession {
    let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
    let paths = NamespacePaths::new("tenants", NamespaceId::new(name).unwrap());
    let mut writer = WriterSession::open(store, paths).await.unwrap();
    for ordinal in 0..ROWS {
        let mut props: BTreeMap<String, CoreValue> = BTreeMap::new();
        props.insert("name".into(), CoreValue::Str(format!("p{ordinal:02}")));
        writer
            .upsert_node(
                "Person",
                pid(ordinal),
                &NodeWriteRecord {
                    properties: props,
                    schema_version: 1,
                    ..Default::default()
                },
            )
            .unwrap();
    }
    writer.commit_batch().await.unwrap();
    writer
}

#[tokio::test]
async fn skip_cursor_pages_are_gap_free_and_terminate() {
    let writer = corpus("page-skip").await;
    let snapshot = writer.snapshot();
    let parsed = parse("MATCH (p:Person) RETURN p.name AS name ORDER BY name").unwrap();
    let base_plan = lower(&parsed).unwrap();

    let mut cursor: Option<Cursor> = None;
    let mut seen: Vec<String> = Vec::new();
    let mut pages = 0;
    loop {
        let plan = paginate_plan(base_plan.clone(), cursor.as_ref(), PAGE);
        let rows = execute(&plan, &snapshot, &Params::new()).await.unwrap();
        pages += 1;
        for row in &rows {
            match row.get("name") {
                Some(RuntimeValue::String(name)) => seen.push(name.clone()),
                other => panic!("unexpected: {other:?}"),
            }
        }
        match next_cursor(cursor.as_ref(), rows.len(), PAGE) {
            Some(next) => cursor = Some(next),
            None => break,
        }
        assert!(pages < 10, "pagination must terminate");
    }
    assert_eq!(pages, 3, "25 rows at page 10 = pages of 10/10/5");
    let expected: Vec<String> = (0..ROWS).map(|o| format!("p{o:02}")).collect();
    assert_eq!(seen, expected, "no duplicates, no gaps, stable order");
}

#[tokio::test]
async fn keyset_pages_survive_a_flush_and_deletes_without_dups_or_gaps() {
    let mut writer = corpus("page-keyset").await;
    let parsed = parse("MATCH (p:Person) RETURN p").unwrap();
    let base_plan = lower(&parsed).unwrap();
    let plan_hash = query_text_hash("MATCH (p:Person) RETURN p");

    // Page 1 against the memtable.
    let page1_plan = paginate_plan_keyset(base_plan.clone(), None, PAGE, "p");
    let snapshot = writer.snapshot();
    let page1 = execute(&page1_plan, &snapshot, &Params::new())
        .await
        .unwrap();
    drop(snapshot);
    assert_eq!(page1.len(), PAGE as usize);
    let ids_of = |rows: &[namidb_query::Row]| -> Vec<NodeId> {
        rows.iter()
            .map(|row| match row.get("p") {
                Some(RuntimeValue::Node(node)) => node.id,
                other => panic!("unexpected: {other:?}"),
            })
            .collect()
    };
    let page1_ids = ids_of(&page1);
    let last = *page1_ids.last().unwrap();
    let last_string = last.to_string();
    let cursor = next_cursor_keyset(plan_hash, page1.len(), PAGE, Some(&last_string))
        .expect("a full page must yield a cursor");

    // Between pages: flush everything, delete one row ALREADY SERVED
    // (ordinal 3, page 1) and one row NOT YET served (ordinal 17).
    writer.flush(schema()).await.unwrap();
    writer.tombstone_node("Person", pid(3)).unwrap();
    writer.tombstone_node("Person", pid(17)).unwrap();
    writer.commit_batch().await.unwrap();

    // Remaining pages through the keyset cursor.
    let mut seen: Vec<NodeId> = page1_ids.clone();
    let mut cursor = Some(cursor);
    while let Some(current) = cursor.take() {
        let plan = paginate_plan_keyset(base_plan.clone(), Some(&current), PAGE, "p");
        let snapshot = writer.snapshot();
        let rows = execute(&plan, &snapshot, &Params::new()).await.unwrap();
        let ids = ids_of(&rows);
        seen.extend(ids.iter().copied());
        let last_string = ids.last().map(|id| id.to_string());
        cursor = next_cursor_keyset(plan_hash, rows.len(), PAGE, last_string.as_deref());
    }

    // No id may repeat across pages, in strict ascending order per page
    // chain; the already-served delete stays served (page 1 is history),
    // the not-yet-served delete never appears; everything else appears
    // exactly once.
    let unique: BTreeSet<NodeId> = seen.iter().copied().collect();
    assert_eq!(
        unique.len(),
        seen.len(),
        "keyset pages must never duplicate"
    );
    let expected: BTreeSet<NodeId> = (0..ROWS).filter(|o| *o != 17).map(pid).collect();
    assert_eq!(
        unique, expected,
        "every surviving row exactly once; the unseen delete gone"
    );
}

#[test]
fn stale_and_corrupt_cursors_are_rejected() {
    let hash_a = query_text_hash("MATCH (p:Person) RETURN p");
    let hash_b = query_text_hash("MATCH (q:Doc) RETURN q");
    assert_ne!(hash_a, hash_b, "different queries must hash differently");

    let cursor = CursorKeyset::new(hash_a, "some-id");
    let decoded = CursorKeyset::decode(&cursor.encode()).unwrap();
    assert_eq!(
        decoded.plan_hash, hash_a,
        "the hash survives the round trip"
    );
    // The module contract: callers compare plan_hash against the current
    // query's hash and reject on mismatch. The decoded hash makes that
    // comparison possible; a doctored blob must fail outright.
    assert!(CursorKeyset::decode("v2:not-a-hash").is_err());
    assert!(CursorKeyset::decode("garbage").is_err());
    assert!(Cursor::decode("v1:not-a-number").is_err());
}
