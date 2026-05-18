//! Parity tests for the factorisation path.
//!
//! Each test runs the SAME logical plan through both `execute_flat_path`
//! (the legacy `Vec<Row>` executor) and `execute_factor_path` (the
//! `FactorRowSet` executor) and asserts that the produced row sets
//! are equal (order-insensitive). Multiplicity is preserved by sorting
//! a stable fingerprint and comparing the sorted vectors directly.
//!
//! Fixture is a hand-rolled mini graph:
//!
//! ```text
//! Alice ──KNOWS──▶ Bob ──KNOWS──▶ Carol
//! │ │ │
//! │ └─HAS_CREATOR── post_b1
//! │ │
//! └─KNOWS──▶ Dave │
//! │ │
//! └─HAS_CREATOR── post_a1, post_a2
//! │
//! Bob ── HAS_CREATOR ── post_b1
//! Carol ── HAS_CREATOR ── post_c1
//! Dave ── HAS_CREATOR ── post_d1
//! ```
//!
//! Five Persons, four Posts. Edge fan-out chosen so multi-hop queries
//! exercise the cartesian path (Alice→Bob→Carol with HAS_CREATOR
//! reachable through several friends-of-friends).

use std::collections::BTreeMap;
use std::sync::Arc;

use namidb_core::id::{NamespaceId, NodeId};
use namidb_core::value::Value as CoreValue;
use namidb_query::{
 execute_factor_path, execute_flat_path, lower, parse, plan, Params, Row, StatsCatalog,
};
use namidb_storage::{EdgeWriteRecord, NamespacePaths, NodeWriteRecord, WriterSession};
use object_store::memory::InMemory;
use object_store::ObjectStore;

fn store() -> Arc<dyn ObjectStore> {
 Arc::new(InMemory::new())
}

fn paths(name: &str) -> NamespacePaths {
 NamespacePaths::new("tenants", NamespaceId::new(name).unwrap())
}

fn person(first: &str, last: &str) -> NodeWriteRecord {
 let mut p: BTreeMap<String, CoreValue> = BTreeMap::new();
 p.insert("firstName".into(), CoreValue::Str(first.into()));
 p.insert("lastName".into(), CoreValue::Str(last.into()));
 NodeWriteRecord {
 properties: p,
 schema_version: 1,
 }
}

fn post(content: &str, date: i64) -> NodeWriteRecord {
 let mut p: BTreeMap<String, CoreValue> = BTreeMap::new();
 p.insert("content".into(), CoreValue::Str(content.into()));
 p.insert("creationDate".into(), CoreValue::I64(date));
 NodeWriteRecord {
 properties: p,
 schema_version: 1,
 }
}

async fn setup_fixture() -> WriterSession {
 let mut w = WriterSession::open(store(), paths("parity"))
 .await
 .expect("open writer");

 let alice = NodeId::new();
 let bob = NodeId::new();
 let carol = NodeId::new();
 let dave = NodeId::new();
 let eve = NodeId::new();

 let post_a1 = NodeId::new();
 let post_a2 = NodeId::new();
 let post_b1 = NodeId::new();
 let post_c1 = NodeId::new();
 let post_d1 = NodeId::new();

 for (id, rec) in [
 (alice, person("Alice", "Andersen")),
 (bob, person("Bob", "Brown")),
 (carol, person("Carol", "Clark")),
 (dave, person("Dave", "Davies")),
 (eve, person("Eve", "Edwards")),
 ] {
 w.upsert_node("Person", id, &rec).unwrap();
 }

 for (id, rec) in [
 (post_a1, post("Alice post 1", 1_000_000)),
 (post_a2, post("Alice post 2", 1_000_100)),
 (post_b1, post("Bob post 1", 1_000_200)),
 (post_c1, post("Carol post 1", 1_000_300)),
 (post_d1, post("Dave post 1", 1_000_400)),
 ] {
 w.upsert_node("Post", id, &rec).unwrap();
 }

 // KNOWS: Alice→{Bob, Dave}; Bob→{Carol, Eve}; Carol→{Bob}; Dave→{Eve}.
 for (src, dst) in [
 (alice, bob),
 (alice, dave),
 (bob, carol),
 (bob, eve),
 (carol, bob),
 (dave, eve),
 ] {
 w.upsert_edge(
 "KNOWS",
 src,
 dst,
 &EdgeWriteRecord {
 properties: BTreeMap::new(),
 schema_version: 1,
 },
 )
 .unwrap();
 }

 // HAS_CREATOR: each post → its author.
 for (post_id, author) in [
 (post_a1, alice),
 (post_a2, alice),
 (post_b1, bob),
 (post_c1, carol),
 (post_d1, dave),
 ] {
 w.upsert_edge(
 "HAS_CREATOR",
 post_id,
 author,
 &EdgeWriteRecord {
 properties: BTreeMap::new(),
 schema_version: 1,
 },
 )
 .unwrap();
 }

 w
}

fn fingerprint(row: &Row) -> String {
 let mut s = String::new();
 // BTreeMap iterates in key order, so the fingerprint is stable.
 for (k, v) in &row.bindings {
 s.push_str(k);
 s.push('=');
 s.push_str(&format!("{:?}", v));
 s.push('|');
 }
 s
}

fn sorted_fingerprints(rows: &[Row]) -> Vec<String> {
 let mut fp: Vec<String> = rows.iter().map(fingerprint).collect();
 fp.sort();
 fp
}

async fn assert_parity(cypher: &str, params: &Params, snapshot_label: &str) {
 let writer = setup_fixture().await;
 let snap = writer.snapshot();
 let query = parse(cypher).expect("parse");
 let lp = lower(&query).expect("lower");

 let flat = execute_flat_path(&lp, &snap, params)
 .await
 .expect("flat execute");
 let fact = execute_factor_path(&lp, &snap, params)
 .await
 .expect("factor execute");

 assert_eq!(
 sorted_fingerprints(&flat),
 sorted_fingerprints(&fact),
 "parity failure on `{}`",
 snapshot_label
 );
 assert_eq!(
 flat.len(),
 fact.len(),
 "multiplicity mismatch on `{}`: flat={} factor={}",
 snapshot_label,
 flat.len(),
 fact.len()
 );
}

/// Same as `assert_parity` but routes through the full `plan()`
/// pipeline (lower + optimize). Used to assert parity on plans that
/// contain HashJoin / HashSemiJoin, which only appear post-optimisation
/// (RFC-011 / RFC-012 / RFC-014). The snapshot's manifest provides the
/// `StatsCatalog` so cardinality / selectivity estimates match what the
/// CLI sees in production.
async fn assert_parity_optimized(cypher: &str, params: &Params, snapshot_label: &str) {
 let writer = setup_fixture().await;
 let snap = writer.snapshot();
 let query = parse(cypher).expect("parse");
 let catalog = StatsCatalog::from_manifest(&snap.manifest().manifest);
 let lp = plan(&query, &catalog).expect("plan");

 let flat = execute_flat_path(&lp, &snap, params)
 .await
 .expect("flat execute");
 let fact = execute_factor_path(&lp, &snap, params)
 .await
 .expect("factor execute");

 assert_eq!(
 sorted_fingerprints(&flat),
 sorted_fingerprints(&fact),
 "parity failure on `{}` (optimized)",
 snapshot_label
 );
 assert_eq!(
 flat.len(),
 fact.len(),
 "multiplicity mismatch on `{}` (optimized): flat={} factor={}",
 snapshot_label,
 flat.len(),
 fact.len()
 );
}

#[tokio::test]
async fn parity_simple_node_scan() {
 assert_parity(
 "MATCH (p:Person) RETURN p.firstName AS name",
 &Params::default(),
 "MATCH (p:Person)",
 )
 .await;
}

#[tokio::test]
async fn parity_single_hop_knows() {
 // Single Expand from a NodeScan. Should produce all KNOWS pairs.
 assert_parity(
 "MATCH (a:Person)-[:KNOWS]->(b:Person) RETURN a.firstName AS aName, b.firstName AS bName",
 &Params::default(),
 "single-hop KNOWS",
 )
 .await;
}

#[tokio::test]
async fn parity_two_hop_friends_of_friends() {
 // Two-hop Expand. Without factorisation this materialises every
 // (a, b, c) triple. With factor: arena chain p→f→fof.
 assert_parity(
 "MATCH (a:Person)-[:KNOWS]->(b:Person)-[:KNOWS]->(c:Person) \
 RETURN a.firstName AS aName, b.firstName AS bName, c.firstName AS cName",
 &Params::default(),
 "2-hop friends-of-friends",
 )
 .await;
}

#[tokio::test]
async fn parity_two_hop_with_filter() {
 // Filter post-Expand. The factor path applies the filter on the
 // arena leaves (materialise → predicate → drop or keep idx).
 assert_parity(
 "MATCH (a:Person)-[:KNOWS]->(b:Person) \
 WHERE a.firstName = 'Alice' \
 RETURN b.firstName AS friend",
 &Params::default(),
 "1-hop + Filter",
 )
 .await;
}

#[tokio::test]
async fn parity_ic09_shaped() {
 // IC9-shaped: friends-of-friends → has_creator. This is the
 // ~382x query in the gate; here we only assert parity
 // (the bench measures latency separately).
 assert_parity(
 "MATCH (a:Person)-[:KNOWS]->(b:Person)-[:KNOWS]->(c:Person)\
 <-[:HAS_CREATOR]-(msg:Post) \
 RETURN c.firstName AS personFirstName, msg.content AS messageContent, \
 msg.creationDate AS messageCreationDate \
 ORDER BY messageCreationDate DESC LIMIT 20",
 &Params::default(),
 "IC9 shape",
 )
 .await;
}

#[tokio::test]
async fn parity_ic02_shaped() {
 // IC2-shaped: friend + has_creator. Exercises Expand → reverse-Expand.
 assert_parity(
 "MATCH (a:Person)-[:KNOWS]->(b:Person)<-[:HAS_CREATOR]-(msg:Post) \
 RETURN b.firstName AS friendName, msg.content AS msgContent \
 ORDER BY msg.creationDate DESC LIMIT 10",
 &Params::default(),
 "IC2 shape",
 )
 .await;
}

#[tokio::test]
async fn parity_optional_match_unbound_friend() {
 // OPTIONAL MATCH semantics: every Person in the input survives even
 // if no friend. Factor path uses Slot { value: Null }.
 assert_parity(
 "MATCH (a:Person) \
 OPTIONAL MATCH (a)-[:KNOWS]->(b:Person) \
 RETURN a.firstName AS aName, b.firstName AS bName",
 &Params::default(),
 "OPTIONAL MATCH",
 )
 .await;
}

// ──────────────────────────────────────────────────────────────────────
// — Parity tests for CrossProduct, HashJoin, HashSemiJoin
// ──────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn parity_disconnected_cross_product() {
 // Two disconnected pattern parts with no shared bindings → naive
 // CrossProduct in the lowered plan. Exercises `cross_product_factor`
 // bridge node semantics: each (a, p) pair becomes one FactorNode
 // under a's leaf carrying p's bindings.
 assert_parity(
 "MATCH (a:Person), (p:Post) \
 RETURN a.firstName AS aName, p.content AS pContent",
 &Params::default(),
 "disconnected CrossProduct",
 )
 .await;
}

#[tokio::test]
async fn parity_cross_product_with_filter() {
 // CrossProduct followed by Filter. Lowered (un-optimized) — the
 // filter sits outside the cross product so we still hit
 // cross_product_factor for the join, then `Filter` materialises per
 // leaf to evaluate the predicate.
 assert_parity(
 "MATCH (a:Person), (p:Post) \
 WHERE a.firstName = 'Alice' \
 RETURN a.firstName AS aName, p.content AS pContent",
 &Params::default(),
 "CrossProduct + Filter",
 )
 .await;
}

#[tokio::test]
async fn parity_hash_join_optimized() {
 // Equi-join lowered as CrossProduct + Filter(eq), optimised to
 // HashJoin conversion. Run through full plan() pipeline
 // so the optimizer actually fires; without it the plan stays
 // CrossProduct.
 assert_parity_optimized(
 "MATCH (a:Person), (b:Person) \
 WHERE a.firstName = b.firstName AND a.lastName <> b.lastName \
 RETURN a.firstName AS aName, b.lastName AS bLast",
 &Params::default(),
 "HashJoin (a.firstName = b.firstName)",
 )
 .await;
}

#[tokio::test]
async fn parity_hash_semi_join_exists() {
 // EXISTS sub-pattern → decorrelated HashSemiJoin (RFC-014).
 // Only the outer survives; inner is set lookup. No bindings flow
 // from inner.
 assert_parity_optimized(
 "MATCH (a:Person) \
 WHERE EXISTS((a)-[:KNOWS]->(b:Person)) \
 RETURN a.firstName AS aName",
 &Params::default(),
 "HashSemiJoin via EXISTS",
 )
 .await;
}

#[tokio::test]
async fn parity_hash_semi_join_not_exists() {
 // NOT EXISTS variant. Same decorrelation path, negated truth table.
 // In the fixture Eve has no out-KNOWS, so she should be the only
 // survivor under NOT EXISTS.
 assert_parity_optimized(
 "MATCH (a:Person) \
 WHERE NOT EXISTS((a)-[:KNOWS]->(b:Person)) \
 RETURN a.firstName AS aName",
 &Params::default(),
 "HashSemiJoin via NOT EXISTS",
 )
 .await;
}
