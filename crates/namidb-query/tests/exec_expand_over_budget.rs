//! A single hop whose distinct targets outgrow the endpoint view budget
//! resolves each edge on its own, but the neighbours the label sidecar
//! already proved lack the target label must not be read again. Its own
//! binary because the budget is a process-wide environment variable.

use std::collections::BTreeMap;
use std::sync::Arc;

use namidb_core::id::{NamespaceId, NodeId};
use namidb_core::schema::{DataType, LabelDef, PropertyDef, SchemaBuilder};
use namidb_core::value::Value as CoreValue;
use namidb_storage::{
    EdgeWriteRecord, NamespacePaths, NodeViewCache, NodeWriteRecord, SessionCaches, WriterSession,
};
use object_store::memory::InMemory;
use object_store::ObjectStore;

use namidb_query::{execute, lower, parse, Params};

const FANIN: usize = 200;

fn person(name: &str) -> NodeWriteRecord {
    let mut props: BTreeMap<String, CoreValue> = BTreeMap::new();
    props.insert("name".into(), CoreValue::Str(name.into()));
    NodeWriteRecord {
        properties: props,
        schema_version: 1,
        ..Default::default()
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn over_budget_hop_skips_proven_non_members() {
    // Every hop is over budget, so it takes the per-edge route.
    std::env::set_var("NAMIDB_EXPAND_TARGET_VIEW_BUDGET", "0");
    let node_cache = Arc::new(NodeViewCache::new(256 << 20));
    let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
    let paths = NamespacePaths::new("tenants", NamespaceId::new("exec-over-budget").unwrap());
    let mut writer = WriterSession::open_with_caches(
        store,
        paths,
        SessionCaches {
            sst_cache: None,
            node_cache: Some(node_cache.clone()),
            adjacency_cache: None,
        },
    )
    .await
    .unwrap();

    let hub = NodeId::new();
    writer.upsert_node("Person", hub, &person("hub")).unwrap();
    for ordinal in 0..FANIN {
        for label in ["Green", "Red"] {
            let id = NodeId::new();
            writer
                .upsert_node(label, id, &person(&format!("{label}-{ordinal}")))
                .unwrap();
            writer
                .upsert_edge("KNOWS", hub, id, &EdgeWriteRecord::default())
                .unwrap();
        }
    }
    writer.commit_batch().await.unwrap();
    let schema = SchemaBuilder::new()
        .label(LabelDef {
            name: "Person".into(),
            properties: vec![PropertyDef::new("name", DataType::Utf8, false).unwrap()],
        })
        .unwrap()
        .build();
    writer.flush(schema).await.unwrap();

    let snapshot = writer.snapshot();
    let plan = lower(
        &parse("MATCH (h:Person {name: 'hub'})-[:KNOWS]->(t:Green) RETURN t.name AS n").unwrap(),
    )
    .unwrap();
    let inserts = node_cache.inserts();
    let rows = execute(&plan, &snapshot, &Params::new()).await.unwrap();
    let resolved = node_cache.inserts() - inserts;
    assert_eq!(rows.len(), FANIN);
    // The batch prewarm and the per-edge read each resolve the :Green views.
    assert!(
        resolved < (2 * FANIN + FANIN / 2) as u64,
        "the {FANIN} :Red neighbours were proven non-members and must not be \
         looked up per edge ({resolved} node views resolved)"
    );
}
