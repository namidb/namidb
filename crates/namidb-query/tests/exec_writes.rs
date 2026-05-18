//! End-to-end tests for write clauses.
//!
//! Exercises CREATE / MATCH+CREATE / SET / REMOVE / DELETE / DETACH DELETE
//! / MERGE-match / MERGE-create against a fresh `WriterSession`. After each
//! mutation the test snapshots the writer to confirm durability.

use std::collections::BTreeMap;
use std::sync::Arc;

use namidb_core::id::{NamespaceId, NodeId};
use namidb_core::value::Value as CoreValue;
use namidb_storage::{NamespacePaths, NodeWriteRecord, WriterSession};
use object_store::memory::InMemory;
use object_store::ObjectStore;

use namidb_query::{execute_write, lower, parse, Params, RuntimeValue};

fn store() -> Arc<dyn ObjectStore> {
 Arc::new(InMemory::new())
}

fn paths(name: &str) -> NamespacePaths {
 NamespacePaths::new("tenants", NamespaceId::new(name).unwrap())
}

#[tokio::test]
async fn create_single_node_persists() {
 let mut writer = WriterSession::open(store(), paths("w-create-1"))
 .await
 .unwrap();
 let q = parse("CREATE (a:Person {name: 'Ada', age: 36}) RETURN a").unwrap();
 let plan = lower(&q).unwrap();
 let outcome = execute_write(&plan, &mut writer, &Params::new())
 .await
 .unwrap();
 assert_eq!(outcome.nodes_created, 1);
 assert_eq!(outcome.edges_created, 0);
 assert_eq!(outcome.rows.len(), 1);
 // Snapshot reads see the new node.
 let snap = writer.snapshot();
 let nodes = snap.scan_label("Person").await.unwrap();
 assert_eq!(nodes.len(), 1);
 assert_eq!(
 nodes[0].properties.get("name"),
 Some(&CoreValue::Str("Ada".into()))
 );
}

#[tokio::test]
async fn match_then_create_relationship() {
 let mut writer = WriterSession::open(store(), paths("w-match-create"))
 .await
 .unwrap();
 // Seed two persons via the storage API for determinism.
 let alice = NodeId::new();
 let bob = NodeId::new();
 let mut p_alice = BTreeMap::new();
 p_alice.insert("name".into(), CoreValue::Str("Alice".into()));
 let mut p_bob = BTreeMap::new();
 p_bob.insert("name".into(), CoreValue::Str("Bob".into()));
 writer
 .upsert_node(
 "Person",
 alice,
 &NodeWriteRecord {
 properties: p_alice,
 schema_version: 1,
 },
 )
 .unwrap();
 writer
 .upsert_node(
 "Person",
 bob,
 &NodeWriteRecord {
 properties: p_bob,
 schema_version: 1,
 },
 )
 .unwrap();
 writer.commit_batch().await.unwrap();

 let q = parse(
 "MATCH (a:Person {id: $aid}), (b:Person {id: $bid}) \
 CREATE (a)-[r:KNOWS]->(b) RETURN r",
 )
 .unwrap();
 let plan = lower(&q).unwrap();
 let mut params = Params::new();
 params.insert("aid".into(), RuntimeValue::String(alice.to_string()));
 params.insert("bid".into(), RuntimeValue::String(bob.to_string()));
 let outcome = execute_write(&plan, &mut writer, &params).await.unwrap();
 assert_eq!(outcome.edges_created, 1);
 let snap = writer.snapshot();
 let edges = snap.out_edges("KNOWS", alice).await.unwrap();
 assert_eq!(edges.edges.len(), 1);
 assert_eq!(edges.edges[0].dst, bob);
}

#[tokio::test]
async fn set_property_round_trips() {
 let mut writer = WriterSession::open(store(), paths("w-set")).await.unwrap();
 let alice = NodeId::new();
 let mut p = BTreeMap::new();
 p.insert("name".into(), CoreValue::Str("Alice".into()));
 p.insert("age".into(), CoreValue::I64(30));
 writer
 .upsert_node(
 "Person",
 alice,
 &NodeWriteRecord {
 properties: p,
 schema_version: 1,
 },
 )
 .unwrap();
 writer.commit_batch().await.unwrap();

 let q = parse("MATCH (a:Person {id: $aid}) SET a.age = 31").unwrap();
 let plan = lower(&q).unwrap();
 let mut params = Params::new();
 params.insert("aid".into(), RuntimeValue::String(alice.to_string()));
 let outcome = execute_write(&plan, &mut writer, &params).await.unwrap();
 assert_eq!(outcome.properties_set, 1);

 let snap = writer.snapshot();
 let v = snap.lookup_node("Person", alice).await.unwrap().unwrap();
 assert_eq!(v.properties.get("age"), Some(&CoreValue::I64(31)));
 assert_eq!(
 v.properties.get("name"),
 Some(&CoreValue::Str("Alice".into()))
 );
}

#[tokio::test]
async fn remove_property() {
 let mut writer = WriterSession::open(store(), paths("w-remove"))
 .await
 .unwrap();
 let alice = NodeId::new();
 let mut p = BTreeMap::new();
 p.insert("name".into(), CoreValue::Str("Alice".into()));
 p.insert("age".into(), CoreValue::I64(30));
 writer
 .upsert_node(
 "Person",
 alice,
 &NodeWriteRecord {
 properties: p,
 schema_version: 1,
 },
 )
 .unwrap();
 writer.commit_batch().await.unwrap();

 let q = parse("MATCH (a:Person {id: $aid}) REMOVE a.age").unwrap();
 let plan = lower(&q).unwrap();
 let mut params = Params::new();
 params.insert("aid".into(), RuntimeValue::String(alice.to_string()));
 let _outcome = execute_write(&plan, &mut writer, &params).await.unwrap();

 let snap = writer.snapshot();
 let v = snap.lookup_node("Person", alice).await.unwrap().unwrap();
 assert!(!v.properties.contains_key("age"));
 assert_eq!(
 v.properties.get("name"),
 Some(&CoreValue::Str("Alice".into()))
 );
}

#[tokio::test]
async fn detach_delete_removes_node_and_edges() {
 let mut writer = WriterSession::open(store(), paths("w-detach"))
 .await
 .unwrap();
 let alice = NodeId::new();
 let bob = NodeId::new();
 writer
 .upsert_node(
 "Person",
 alice,
 &NodeWriteRecord {
 properties: BTreeMap::new(),
 schema_version: 1,
 },
 )
 .unwrap();
 writer
 .upsert_node(
 "Person",
 bob,
 &NodeWriteRecord {
 properties: BTreeMap::new(),
 schema_version: 1,
 },
 )
 .unwrap();
 writer
 .upsert_edge(
 "KNOWS",
 alice,
 bob,
 &namidb_storage::EdgeWriteRecord {
 properties: BTreeMap::new(),
 schema_version: 1,
 },
 )
 .unwrap();
 writer.commit_batch().await.unwrap();

 let q = parse("MATCH (a:Person {id: $aid}) DETACH DELETE a").unwrap();
 let plan = lower(&q).unwrap();
 let mut params = Params::new();
 params.insert("aid".into(), RuntimeValue::String(alice.to_string()));
 let outcome = execute_write(&plan, &mut writer, &params).await.unwrap();
 assert_eq!(outcome.nodes_deleted, 1);
 assert!(outcome.edges_deleted >= 1);

 let snap = writer.snapshot();
 assert!(snap.lookup_node("Person", alice).await.unwrap().is_none());
 let edges = snap.out_edges("KNOWS", alice).await.unwrap();
 assert_eq!(edges.edges.len(), 0);
}

#[tokio::test]
async fn merge_match_path_runs_on_match_sets() {
 let mut writer = WriterSession::open(store(), paths("w-merge-match"))
 .await
 .unwrap();
 let alice = NodeId::new();
 let mut p = BTreeMap::new();
 p.insert("externalId".into(), CoreValue::I64(42));
 p.insert("seen".into(), CoreValue::I64(1));
 writer
 .upsert_node(
 "Person",
 alice,
 &NodeWriteRecord {
 properties: p,
 schema_version: 1,
 },
 )
 .unwrap();
 writer.commit_batch().await.unwrap();

 let q = parse(
 "MERGE (a:Person {externalId: 42}) \
 ON MATCH SET a.seen = 2",
 )
 .unwrap();
 let plan = lower(&q).unwrap();
 let outcome = execute_write(&plan, &mut writer, &Params::new())
 .await
 .unwrap();
 assert_eq!(outcome.nodes_created, 0);
 assert_eq!(outcome.properties_set, 1);

 let snap = writer.snapshot();
 let v = snap.lookup_node("Person", alice).await.unwrap().unwrap();
 assert_eq!(v.properties.get("seen"), Some(&CoreValue::I64(2)));
}

#[tokio::test]
async fn merge_create_path_creates_and_runs_on_create_sets() {
 let mut writer = WriterSession::open(store(), paths("w-merge-create"))
 .await
 .unwrap();

 let q = parse(
 "MERGE (a:Person {externalId: 7}) \
 ON CREATE SET a.firstSeen = 1",
 )
 .unwrap();
 let plan = lower(&q).unwrap();
 let outcome = execute_write(&plan, &mut writer, &Params::new())
 .await
 .unwrap();
 assert_eq!(outcome.nodes_created, 1);
 // properties_set counts the ON CREATE SET application.
 assert_eq!(outcome.properties_set, 1);

 let snap = writer.snapshot();
 let nodes = snap.scan_label("Person").await.unwrap();
 assert_eq!(nodes.len(), 1);
 assert_eq!(
 nodes[0].properties.get("externalId"),
 Some(&CoreValue::I64(7))
 );
 assert_eq!(
 nodes[0].properties.get("firstSeen"),
 Some(&CoreValue::I64(1))
 );
}

#[tokio::test]
async fn create_chain_node_rel_node() {
 let mut writer = WriterSession::open(store(), paths("w-chain"))
 .await
 .unwrap();
 let q = parse(
 "CREATE (a:Person {name: 'Ada'})-[r:KNOWS {weight: 5}]->(b:Person {name: 'Lin'}) \
 RETURN a.name AS aname, b.name AS bname",
 )
 .unwrap();
 let plan = lower(&q).unwrap();
 let outcome = execute_write(&plan, &mut writer, &Params::new())
 .await
 .unwrap();
 assert_eq!(outcome.nodes_created, 2);
 assert_eq!(outcome.edges_created, 1);
 assert_eq!(outcome.rows.len(), 1);
 match outcome.rows[0].get("aname") {
 Some(RuntimeValue::String(s)) => assert_eq!(s, "Ada"),
 other => panic!("unexpected: {:?}", other),
 }
 match outcome.rows[0].get("bname") {
 Some(RuntimeValue::String(s)) => assert_eq!(s, "Lin"),
 other => panic!("unexpected: {:?}", other),
 }
}
