//! End-to-end LDBC SNB Interactive Update queries (IU1, IU2, IU6, IU8).
//!
//! Each test seeds a small base graph, executes the canonical update
//! fixture (simplified versions of the official LDBC IU queries — see
//! `fixtures/iu*_*.cypher`), then snapshots the writer and asserts the
//! mutation is durably visible.

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

fn read_fixture(name: &str) -> String {
    let path = format!("{}/tests/fixtures/{}", env!("CARGO_MANIFEST_DIR"), name);
    std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {}", path, e))
}

async fn seed_person(writer: &mut WriterSession, first: &str, last: &str, age: i32) -> NodeId {
    let id = NodeId::new();
    let mut p = BTreeMap::new();
    p.insert("firstName".into(), CoreValue::Str(first.into()));
    p.insert("lastName".into(), CoreValue::Str(last.into()));
    p.insert("age".into(), CoreValue::I64(age as i64));
    writer
        .upsert_node(
            "Person",
            id,
            &NodeWriteRecord {
                properties: p,
                schema_version: 1,
                ..Default::default()
            },
        )
        .unwrap();
    id
}

async fn seed_message(writer: &mut WriterSession, content: &str, creation_date: i64) -> NodeId {
    let id = NodeId::new();
    let mut p = BTreeMap::new();
    p.insert("content".into(), CoreValue::Str(content.into()));
    p.insert("creationDate".into(), CoreValue::I64(creation_date));
    writer
        .upsert_node(
            "Message",
            id,
            &NodeWriteRecord {
                properties: p,
                schema_version: 1,
                ..Default::default()
            },
        )
        .unwrap();
    id
}

// ──────────────────────────────────────────────────────────────────────
// IU1 — Insert Person + initial KNOWS to a single existing friend.
// ──────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn iu01_insert_person_plus_knows() {
    let mut writer = WriterSession::open(store(), paths("iu01")).await.unwrap();
    let alice = seed_person(&mut writer, "Alice", "Anderson", 30).await;
    writer.commit_batch().await.unwrap();

    let src = read_fixture("iu01_insert_person.cypher");
    let q = parse(&src).expect("parse IU1");
    let plan = lower(&q).expect("lower IU1");

    // Generate a fresh NodeId for the new Person.
    let new_person = NodeId::new();
    let mut params = Params::new();
    params.insert(
        "personId".into(),
        RuntimeValue::String(new_person.to_string()),
    );
    params.insert("firstName".into(), RuntimeValue::String("Bob".into()));
    params.insert("lastName".into(), RuntimeValue::String("Brown".into()));
    params.insert("age".into(), RuntimeValue::Integer(25));
    params.insert("friendId".into(), RuntimeValue::String(alice.to_string()));

    let outcome = execute_write(&plan, &mut writer, &params)
        .await
        .expect("exec IU1");
    assert_eq!(outcome.nodes_created, 1);
    assert_eq!(outcome.edges_created, 1);
    assert_eq!(outcome.rows.len(), 1);

    // Verify Bob is now in storage with the requested NodeId.
    let snap = writer.snapshot();
    let view = snap
        .lookup_node("Person", new_person)
        .await
        .expect("lookup")
        .expect("Bob should exist");
    assert_eq!(
        view.properties.get("firstName"),
        Some(&CoreValue::Str("Bob".into()))
    );
    // And the KNOWS edge from Bob to Alice exists.
    let out_edges = snap.out_edges("KNOWS", new_person).await.unwrap();
    assert_eq!(out_edges.edges.len(), 1);
    assert_eq!(out_edges.edges[0].dst, alice);
}

// ──────────────────────────────────────────────────────────────────────
// IU2 — addPostLike.
// ──────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn iu02_add_post_like() {
    let mut writer = WriterSession::open(store(), paths("iu02")).await.unwrap();
    let alice = seed_person(&mut writer, "Alice", "Anderson", 30).await;
    let post = seed_message(&mut writer, "Hello world", 1_700_000_000_000).await;
    writer.commit_batch().await.unwrap();

    let src = read_fixture("iu02_add_post_like.cypher");
    let q = parse(&src).expect("parse IU2");
    let plan = lower(&q).expect("lower IU2");

    let mut params = Params::new();
    params.insert("personId".into(), RuntimeValue::String(alice.to_string()));
    params.insert("messageId".into(), RuntimeValue::String(post.to_string()));
    let like_date = 1_700_086_400_000_i64;
    params.insert("creationDate".into(), RuntimeValue::Integer(like_date));

    let outcome = execute_write(&plan, &mut writer, &params)
        .await
        .expect("exec IU2");
    assert_eq!(outcome.edges_created, 1);

    let snap = writer.snapshot();
    let out_edges = snap.out_edges("LIKES", alice).await.unwrap();
    assert_eq!(out_edges.edges.len(), 1);
    assert_eq!(out_edges.edges[0].dst, post);
    assert_eq!(
        out_edges.edges[0].properties.get("creationDate"),
        Some(&CoreValue::I64(like_date))
    );
}

// ──────────────────────────────────────────────────────────────────────
// IU6 — addPost.
// ──────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn iu06_add_post() {
    let mut writer = WriterSession::open(store(), paths("iu06")).await.unwrap();
    let alice = seed_person(&mut writer, "Alice", "Anderson", 30).await;
    writer.commit_batch().await.unwrap();

    let src = read_fixture("iu06_add_post.cypher");
    let q = parse(&src).expect("parse IU6");
    let plan = lower(&q).expect("lower IU6");

    let new_msg = NodeId::new();
    let mut params = Params::new();
    params.insert("authorId".into(), RuntimeValue::String(alice.to_string()));
    params.insert(
        "messageId".into(),
        RuntimeValue::String(new_msg.to_string()),
    );
    params.insert(
        "content".into(),
        RuntimeValue::String("Cypher rocks".into()),
    );
    let when = 1_700_000_000_000_i64;
    params.insert("creationDate".into(), RuntimeValue::Integer(when));

    let outcome = execute_write(&plan, &mut writer, &params)
        .await
        .expect("exec IU6");
    assert_eq!(outcome.nodes_created, 1);
    assert_eq!(outcome.edges_created, 1);

    let snap = writer.snapshot();
    let msg = snap
        .lookup_node("Message", new_msg)
        .await
        .unwrap()
        .expect("Message persisted");
    assert_eq!(
        msg.properties.get("content"),
        Some(&CoreValue::Str("Cypher rocks".into()))
    );
    let creator_edges = snap.out_edges("HAS_CREATOR", new_msg).await.unwrap();
    assert_eq!(creator_edges.edges.len(), 1);
    assert_eq!(creator_edges.edges[0].dst, alice);
}

// ──────────────────────────────────────────────────────────────────────
// IU8 — addFriendship.
// ──────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn iu08_add_friendship() {
    let mut writer = WriterSession::open(store(), paths("iu08")).await.unwrap();
    let alice = seed_person(&mut writer, "Alice", "Anderson", 30).await;
    let carol = seed_person(&mut writer, "Carol", "Clark", 35).await;
    writer.commit_batch().await.unwrap();

    let src = read_fixture("iu08_add_friendship.cypher");
    let q = parse(&src).expect("parse IU8");
    let plan = lower(&q).expect("lower IU8");

    let mut params = Params::new();
    params.insert("person1Id".into(), RuntimeValue::String(alice.to_string()));
    params.insert("person2Id".into(), RuntimeValue::String(carol.to_string()));
    let when = 1_700_172_800_000_i64;
    params.insert("creationDate".into(), RuntimeValue::Integer(when));

    let outcome = execute_write(&plan, &mut writer, &params)
        .await
        .expect("exec IU8");
    assert_eq!(outcome.edges_created, 1);

    let snap = writer.snapshot();
    let out_edges = snap.out_edges("KNOWS", alice).await.unwrap();
    assert_eq!(out_edges.edges.len(), 1);
    assert_eq!(out_edges.edges[0].dst, carol);
    assert_eq!(
        out_edges.edges[0].properties.get("creationDate"),
        Some(&CoreValue::I64(when))
    );
}
