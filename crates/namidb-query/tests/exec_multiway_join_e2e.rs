//! RFC-024: end-to-end Cypher → plan → execute coverage for the
//! multiway-join detection pass + executor.
//!
//! These tests touch `NAMIDB_WCOJ` and `NAMIDB_FACTORIZE`, which are
//! process-global. We serialise them via a single mutex so they do not
//! race each other under `cargo test`'s default per-binary
//! parallelism.

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

use namidb_core::id::{NamespaceId, NodeId};
use namidb_core::value::Value as CoreValue;
use namidb_query::plan::LogicalPlan;
use namidb_query::{execute, parse, plan as plan_query, Params, StatsCatalog};
use namidb_storage::{EdgeWriteRecord, NamespacePaths, NodeWriteRecord, WriterSession};
use object_store::memory::InMemory;
use object_store::ObjectStore;

// All env-mutating tests grab this lock before touching NAMIDB_WCOJ /
// NAMIDB_FACTORIZE. cargo runs test functions in parallel within a
// binary; the lock keeps each test's view of those vars consistent
// without forcing `--test-threads=1`.
static ENV_LOCK: Mutex<()> = Mutex::new(());

struct EnvGuard {
    keys: Vec<(&'static str, Option<String>)>,
    _lock: std::sync::MutexGuard<'static, ()>,
}

impl EnvGuard {
    fn set(vars: &[(&'static str, &str)]) -> Self {
        let lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let keys = vars
            .iter()
            .map(|(k, v)| {
                let prior = std::env::var(*k).ok();
                std::env::set_var(k, v);
                (*k, prior)
            })
            .collect();
        Self { keys, _lock: lock }
    }
}

impl Drop for EnvGuard {
    fn drop(&mut self) {
        for (k, prior) in &self.keys {
            match prior {
                Some(v) => std::env::set_var(k, v),
                None => std::env::remove_var(k),
            }
        }
    }
}

fn store() -> Arc<dyn ObjectStore> {
    Arc::new(InMemory::new())
}

fn paths(name: &str) -> NamespacePaths {
    NamespacePaths::new("tenants", NamespaceId::new(name).unwrap())
}

fn person(name: &str) -> NodeWriteRecord {
    let mut p: BTreeMap<String, CoreValue> = BTreeMap::new();
    p.insert("name".into(), CoreValue::Str(name.into()));
    NodeWriteRecord {
        properties: p,
        schema_version: 1,
    }
}

fn edge() -> EdgeWriteRecord {
    EdgeWriteRecord {
        properties: BTreeMap::new(),
        schema_version: 1,
    }
}

async fn build_triangle_graph(writer: &mut WriterSession) -> [NodeId; 3] {
    let ids: [NodeId; 3] = std::array::from_fn(|_| NodeId::new());
    let names = ["A", "B", "C"];
    for (id, name) in ids.iter().zip(names.iter()) {
        writer.upsert_node("Person", *id, &person(name)).unwrap();
    }
    let triangle = [(ids[0], ids[1]), (ids[1], ids[2]), (ids[2], ids[0])];
    for (src, dst) in triangle {
        writer.upsert_edge("KNOWS", src, dst, &edge()).unwrap();
    }
    writer.commit_batch().await.unwrap();
    ids
}

fn contains_multiway_join(plan: &LogicalPlan) -> bool {
    if matches!(plan, LogicalPlan::MultiwayJoin { .. }) {
        return true;
    }
    plan.children().iter().any(|c| contains_multiway_join(c))
}

#[tokio::test]
async fn wcoj_rewrites_directed_triangle_cypher() {
    let _env = EnvGuard::set(&[("NAMIDB_WCOJ", "1"), ("NAMIDB_FACTORIZE", "1")]);

    let mut writer = WriterSession::open(store(), paths("wcoj-tri-cypher"))
        .await
        .unwrap();
    let _ = build_triangle_graph(&mut writer).await;
    let snapshot = writer.snapshot();

    let q = parse(
        "MATCH (a:Person)-[:KNOWS]->(b:Person)-[:KNOWS]->(c:Person)-[:KNOWS]->(a) \
         RETURN a.name AS an, b.name AS bn, c.name AS cn",
    )
    .unwrap();
    let cat = StatsCatalog::default();
    let plan = plan_query(&q, &cat).unwrap();
    assert!(
        contains_multiway_join(&plan),
        "expected MultiwayJoin in plan tree, got:\n{:#?}",
        plan
    );

    let rows = execute(&plan, &snapshot, &Params::new()).await.unwrap();
    // Three rotations of the directed triangle (A,B,C), (B,C,A), (C,A,B).
    assert_eq!(
        rows.len(),
        3,
        "expected 3 triangle rotations, got {}",
        rows.len()
    );
}

#[tokio::test]
async fn wcoj_parity_with_binary_path_on_triangle() {
    // Build a fresh graph, then plan twice — once with WCOJ off and
    // once with it on — and assert the produced row sets are equal.
    let mut writer = WriterSession::open(store(), paths("wcoj-parity"))
        .await
        .unwrap();
    let _ = build_triangle_graph(&mut writer).await;
    let snapshot = writer.snapshot();

    let q = parse(
        "MATCH (a:Person)-[:KNOWS]->(b:Person)-[:KNOWS]->(c:Person)-[:KNOWS]->(a) \
         RETURN a.name AS an, b.name AS bn, c.name AS cn",
    )
    .unwrap();
    let cat = StatsCatalog::default();

    let binary_rows = {
        let _env = EnvGuard::set(&[("NAMIDB_WCOJ", "0")]);
        let p = plan_query(&q, &cat).unwrap();
        assert!(
            !contains_multiway_join(&p),
            "binary path produced a MultiwayJoin"
        );
        execute(&p, &snapshot, &Params::new()).await.unwrap()
    };

    let wcoj_rows = {
        let _env = EnvGuard::set(&[("NAMIDB_WCOJ", "1"), ("NAMIDB_FACTORIZE", "1")]);
        let p = plan_query(&q, &cat).unwrap();
        assert!(
            contains_multiway_join(&p),
            "WCOJ path missed the MultiwayJoin rewrite"
        );
        execute(&p, &snapshot, &Params::new()).await.unwrap()
    };

    let mut bin_keys: Vec<String> = binary_rows
        .iter()
        .map(|r| format!("{:?}", r.bindings))
        .collect();
    let mut wcoj_keys: Vec<String> = wcoj_rows
        .iter()
        .map(|r| format!("{:?}", r.bindings))
        .collect();
    bin_keys.sort();
    wcoj_keys.sort();
    assert_eq!(bin_keys, wcoj_keys, "WCOJ and binary row sets diverge");
}

#[tokio::test]
async fn wcoj_does_not_touch_open_chain() {
    // An open chain has no closing edge, so the constraint graph is
    // acyclic and the pass must leave it alone.
    let _env = EnvGuard::set(&[("NAMIDB_WCOJ", "1"), ("NAMIDB_FACTORIZE", "1")]);
    let q = parse(
        "MATCH (a:Person)-[:KNOWS]->(b:Person)-[:KNOWS]->(c:Person) \
         RETURN a.name, b.name, c.name",
    )
    .unwrap();
    let cat = StatsCatalog::default();
    let plan = plan_query(&q, &cat).unwrap();
    assert!(
        !contains_multiway_join(&plan),
        "open chain unexpectedly rewritten to MultiwayJoin"
    );
}

#[tokio::test]
async fn wcoj_does_not_touch_variable_length_cycle() {
    // `[:KNOWS*1..3]` lowers to an Expand with `length = Some(_)`,
    // which the v0 detection pass refuses to rewrite.
    let _env = EnvGuard::set(&[("NAMIDB_WCOJ", "1"), ("NAMIDB_FACTORIZE", "1")]);
    let q = parse(
        "MATCH (a:Person)-[:KNOWS*1..3]->(b:Person)-[:KNOWS]->(a) \
         RETURN a.name, b.name",
    )
    .unwrap();
    let cat = StatsCatalog::default();
    let plan = plan_query(&q, &cat).unwrap();
    assert!(
        !contains_multiway_join(&plan),
        "variable-length cycle unexpectedly rewritten"
    );
}

#[tokio::test]
async fn wcoj_skipped_without_factorize() {
    // RFC-024 §"Feature flag matrix" row: WCOJ=1 + FACTORIZE=0 must NOT
    // produce a MultiwayJoin (the executor would refuse to run it).
    let _env = EnvGuard::set(&[("NAMIDB_WCOJ", "1"), ("NAMIDB_FACTORIZE", "0")]);
    let q = parse(
        "MATCH (a:Person)-[:KNOWS]->(b:Person)-[:KNOWS]->(c:Person)-[:KNOWS]->(a) \
         RETURN a.name, b.name, c.name",
    )
    .unwrap();
    let cat = StatsCatalog::default();
    let plan = plan_query(&q, &cat).unwrap();
    assert!(
        !contains_multiway_join(&plan),
        "WCOJ=1 without FACTORIZE should fall back to the binary plan"
    );
}
