//! Unaudited dimension (25tb-readiness): planner statistics freshness after
//! a bulk load. `StatsCatalog::from_manifest` derives everything from the
//! committed manifest, so the lifecycle contract is: unflushed writes are
//! invisible (memtable-only stats stay at zero — the documented shape), and
//! one flush+compaction later every label and edge-type count is exact.
//! Stale zero-row stats after ingesting 25 TB would otherwise produce
//! catastrophically wrong join orders.

use std::collections::BTreeMap;
use std::sync::Arc;

use namidb_core::id::{NamespaceId, NodeId};
use namidb_core::schema::{DataType, EdgeTypeDef, LabelDef, PropertyDef, Schema, SchemaBuilder};
use namidb_core::value::Value as CoreValue;
use namidb_storage::{EdgeWriteRecord, NamespacePaths, NodeWriteRecord, WriterSession};
use object_store::memory::InMemory;
use object_store::ObjectStore;

use namidb_query::StatsCatalog;

const PEOPLE: u64 = 120;
const CITIES: u64 = 8;
const EDGES: u64 = 240;

fn schema() -> Schema {
    SchemaBuilder::new()
        .label(LabelDef {
            name: "Person".into(),
            properties: vec![PropertyDef::new("name", DataType::Utf8, false).unwrap()],
        })
        .unwrap()
        .label(LabelDef {
            name: "City".into(),
            properties: vec![PropertyDef::new("name", DataType::Utf8, false).unwrap()],
        })
        .unwrap()
        .edge_type(EdgeTypeDef {
            name: "LIVES_IN".into(),
            src_label: "Person".into(),
            dst_label: "City".into(),
            properties: vec![],
        })
        .unwrap()
        .build()
}

fn named(name: String) -> NodeWriteRecord {
    let mut props: BTreeMap<String, CoreValue> = BTreeMap::new();
    props.insert("name".into(), CoreValue::Str(name));
    NodeWriteRecord {
        properties: props,
        schema_version: 1,
        ..Default::default()
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn stats_go_from_zero_to_exact_after_the_bulk_load_flushes() {
    let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
    let paths = NamespacePaths::new("tenants", NamespaceId::new("stats-lifecycle").unwrap());
    let mut writer = WriterSession::open(store, paths).await.unwrap();

    let mut people = Vec::new();
    for ordinal in 0..PEOPLE {
        let id = NodeId::new();
        people.push(id);
        writer
            .upsert_node("Person", id, &named(format!("p{ordinal}")))
            .unwrap();
    }
    let mut cities = Vec::new();
    for ordinal in 0..CITIES {
        let id = NodeId::new();
        cities.push(id);
        writer
            .upsert_node("City", id, &named(format!("c{ordinal}")))
            .unwrap();
    }
    for ordinal in 0..EDGES {
        writer
            .upsert_edge(
                "LIVES_IN",
                people[(ordinal % PEOPLE) as usize],
                cities[((ordinal / PEOPLE) % CITIES) as usize],
                &EdgeWriteRecord {
                    properties: BTreeMap::new(),
                    schema_version: 1,
                },
            )
            .unwrap();
    }
    writer.commit_batch().await.unwrap();

    // Unflushed: manifest-derived stats are zero by contract. A planner
    // consuming them mid-ingest sees an empty graph — the documented shape,
    // pinned so a change here is a conscious decision.
    let unflushed = StatsCatalog::from_manifest(&writer.snapshot().manifest().manifest);
    assert_eq!(unflushed.total_nodes(), 0, "memtable writes are invisible");
    assert_eq!(unflushed.total_edges(), 0);

    // One flush + compaction: exact per-label and per-edge-type counts.
    writer.flush(schema()).await.unwrap();
    writer.compact_l0(&schema()).await.unwrap();
    let fresh = StatsCatalog::from_manifest(&writer.snapshot().manifest().manifest);
    assert_eq!(fresh.total_nodes(), PEOPLE + CITIES);
    assert_eq!(
        fresh.label("Person").map(|stats| stats.node_count),
        Some(PEOPLE),
        "per-label counts must be exact after the load"
    );
    assert_eq!(
        fresh.label("City").map(|stats| stats.node_count),
        Some(CITIES)
    );
    // 240 edge slots collapse onto 120 distinct (person, city) pairs by
    // last-write-wins: person index cycles twice as fast as the city index.
    let expected_edges = {
        use std::collections::BTreeSet;
        let mut pairs = BTreeSet::new();
        for ordinal in 0..EDGES {
            pairs.insert((ordinal % PEOPLE, (ordinal / PEOPLE) % CITIES));
        }
        pairs.len() as u64
    };
    assert_eq!(
        fresh.edge_type("LIVES_IN").map(|stats| stats.edge_count),
        Some(expected_edges),
        "edge-type stats must reflect live last-write-wins edges"
    );

    // Deletes register on the next flush too.
    for id in people.iter().take(20) {
        writer.tombstone_node("Person", *id).unwrap();
    }
    writer.commit_batch().await.unwrap();
    writer.flush(schema()).await.unwrap();
    writer.compact_l0(&schema()).await.unwrap();
    let after_delete = StatsCatalog::from_manifest(&writer.snapshot().manifest().manifest);
    assert_eq!(
        after_delete.label("Person").map(|stats| stats.node_count),
        Some(PEOPLE - 20),
        "stats must track deletions through the next maintenance cycle"
    );
}
