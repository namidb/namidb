//! End-to-end integration tests: parse → lower → execute against an
//! in-memory storage namespace populated via `WriterSession`.
//!
//! milestone — exercises `NodeScan`, `Expand`, `Filter`, `Project`,
//! `TopN` and the expression evaluator over real `Snapshot` reads.

use std::collections::BTreeMap;
use std::sync::Arc;

use namidb_core::id::{NamespaceId, NodeId};
use namidb_core::schema::{DataType, EdgeTypeDef, LabelDef, PropertyDef, SchemaBuilder};
use namidb_core::value::Value as CoreValue;
use namidb_storage::{EdgeWriteRecord, NamespacePaths, NodeWriteRecord, WriterSession};
use object_store::memory::InMemory;
use object_store::ObjectStore;

use namidb_query::{execute, lower, parse, Params, RuntimeValue};

fn store() -> Arc<dyn ObjectStore> {
    Arc::new(InMemory::new())
}

fn paths(name: &str) -> NamespacePaths {
    NamespacePaths::new("tenants", NamespaceId::new(name).unwrap())
}

fn person_label() -> LabelDef {
    LabelDef {
        name: "Person".into(),
        properties: vec![
            PropertyDef::new("name", DataType::Utf8, false).unwrap(),
            PropertyDef::new("age", DataType::Int32, true).unwrap(),
        ],
    }
}

fn knows_edge() -> EdgeTypeDef {
    EdgeTypeDef {
        name: "KNOWS".into(),
        src_label: "Person".into(),
        dst_label: "Person".into(),
        properties: vec![],
    }
}

fn person(name: &str, age: i32) -> NodeWriteRecord {
    let mut props: BTreeMap<String, CoreValue> = BTreeMap::new();
    props.insert("name".into(), CoreValue::Str(name.into()));
    props.insert("age".into(), CoreValue::I64(age as i64));
    NodeWriteRecord {
        properties: props,
        schema_version: 1,
    }
}

fn edge() -> EdgeWriteRecord {
    EdgeWriteRecord {
        properties: BTreeMap::new(),
        schema_version: 1,
    }
}

/// Insert five Person nodes plus a hand-picked set of KNOWS edges,
/// returning the writer ready for snapshotting.
async fn build_friend_graph(writer: &mut WriterSession) -> [NodeId; 5] {
    let names = ["Alice", "Bob", "Carol", "Dave", "Eve"];
    let ages = [30, 25, 40, 35, 28];
    let ids: [NodeId; 5] = std::array::from_fn(|_| NodeId::new());

    for ((id, name), age) in ids.iter().zip(names.iter()).zip(ages.iter()) {
        writer
            .upsert_node("Person", *id, &person(name, *age))
            .unwrap();
    }
    // KNOWS edges (directed):
    //   Alice -> Bob, Carol
    //   Bob   -> Carol
    //   Carol -> Dave
    //   Dave  -> Eve
    //   Eve   -> Alice
    let edges = [
        (ids[0], ids[1]),
        (ids[0], ids[2]),
        (ids[1], ids[2]),
        (ids[2], ids[3]),
        (ids[3], ids[4]),
        (ids[4], ids[0]),
    ];
    for (src, dst) in edges {
        writer.upsert_edge("KNOWS", src, dst, &edge()).unwrap();
    }
    writer.commit_batch().await.unwrap();
    ids
}

#[tokio::test]
async fn match_return_lists_all_persons() {
    let mut writer = WriterSession::open(store(), paths("exec-list"))
        .await
        .unwrap();
    build_friend_graph(&mut writer).await;
    let snapshot = writer.snapshot();

    let q = parse("MATCH (a:Person) RETURN a.name AS name ORDER BY name").unwrap();
    let plan = lower(&q).unwrap();
    let rows = execute(&plan, &snapshot, &Params::new()).await.unwrap();

    let names: Vec<&str> = rows
        .iter()
        .map(|r| match r.get("name") {
            Some(RuntimeValue::String(s)) => s.as_str(),
            other => panic!("unexpected: {:?}", other),
        })
        .collect();
    assert_eq!(names, vec!["Alice", "Bob", "Carol", "Dave", "Eve"]);
}

#[tokio::test]
async fn match_with_where_filters_by_age() {
    let mut writer = WriterSession::open(store(), paths("exec-where"))
        .await
        .unwrap();
    build_friend_graph(&mut writer).await;
    let snapshot = writer.snapshot();

    let q = parse(
        "MATCH (a:Person) WHERE a.age >= 30 \
         RETURN a.name AS name ORDER BY a.age DESC",
    )
    .unwrap();
    let plan = lower(&q).unwrap();
    let rows = execute(&plan, &snapshot, &Params::new()).await.unwrap();

    let names: Vec<&str> = rows
        .iter()
        .map(|r| match r.get("name") {
            Some(RuntimeValue::String(s)) => s.as_str(),
            other => panic!("unexpected: {:?}", other),
        })
        .collect();
    assert_eq!(names, vec!["Carol", "Dave", "Alice"]);
}

#[tokio::test]
async fn match_expand_returns_pairs() {
    let mut writer = WriterSession::open(store(), paths("exec-expand"))
        .await
        .unwrap();
    let ids = build_friend_graph(&mut writer).await;
    let _ = ids;
    let snapshot = writer.snapshot();

    let q = parse(
        "MATCH (a:Person)-[:KNOWS]->(b:Person) \
         RETURN a.name AS from, b.name AS to \
         ORDER BY from, to",
    )
    .unwrap();
    let plan = lower(&q).unwrap();
    let rows = execute(&plan, &snapshot, &Params::new()).await.unwrap();

    let pairs: Vec<(String, String)> = rows
        .iter()
        .map(|r| match (r.get("from"), r.get("to")) {
            (Some(RuntimeValue::String(a)), Some(RuntimeValue::String(b))) => {
                (a.clone(), b.clone())
            }
            other => panic!("unexpected: {:?}", other),
        })
        .collect();

    // Edges inserted by `build_friend_graph`:
    let expected = vec![
        ("Alice".to_string(), "Bob".to_string()),
        ("Alice".to_string(), "Carol".to_string()),
        ("Bob".to_string(), "Carol".to_string()),
        ("Carol".to_string(), "Dave".to_string()),
        ("Dave".to_string(), "Eve".to_string()),
        ("Eve".to_string(), "Alice".to_string()),
    ];
    assert_eq!(pairs, expected);
}

#[tokio::test]
async fn match_expand_with_limit() {
    let mut writer = WriterSession::open(store(), paths("exec-limit"))
        .await
        .unwrap();
    build_friend_graph(&mut writer).await;
    let snapshot = writer.snapshot();

    let q = parse(
        "MATCH (a:Person)-[:KNOWS]->(b:Person) \
         RETURN a.name AS from, b.name AS to \
         ORDER BY from, to LIMIT 3",
    )
    .unwrap();
    let plan = lower(&q).unwrap();
    let rows = execute(&plan, &snapshot, &Params::new()).await.unwrap();
    assert_eq!(rows.len(), 3);
    match rows[0].get("from") {
        Some(RuntimeValue::String(s)) => assert_eq!(s, "Alice"),
        other => panic!("unexpected: {:?}", other),
    }
}

#[tokio::test]
async fn match_with_id_parameter_uses_node_by_id() {
    let mut writer = WriterSession::open(store(), paths("exec-id"))
        .await
        .unwrap();
    let ids = build_friend_graph(&mut writer).await;
    let snapshot = writer.snapshot();

    let q = parse("MATCH (a:Person {id: $personId}) RETURN a.name AS name").unwrap();
    let plan = lower(&q).unwrap();

    let mut params = Params::new();
    params.insert("personId".into(), RuntimeValue::String(ids[2].to_string()));
    let rows = execute(&plan, &snapshot, &params).await.unwrap();
    assert_eq!(rows.len(), 1);
    match rows[0].get("name") {
        Some(RuntimeValue::String(s)) => assert_eq!(s, "Carol"),
        other => panic!("unexpected: {:?}", other),
    }
}

#[tokio::test]
async fn count_aggregate_groups_by_label() {
    let mut writer = WriterSession::open(store(), paths("exec-count"))
        .await
        .unwrap();
    build_friend_graph(&mut writer).await;
    let snapshot = writer.snapshot();

    let q = parse("MATCH (a:Person) RETURN count(*) AS n").unwrap();
    let plan = lower(&q).unwrap();
    let rows = execute(&plan, &snapshot, &Params::new()).await.unwrap();
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].get("n"), Some(&RuntimeValue::Integer(5)));
}

#[tokio::test]
async fn count_friends_per_person() {
    let mut writer = WriterSession::open(store(), paths("exec-cf"))
        .await
        .unwrap();
    build_friend_graph(&mut writer).await;
    let snapshot = writer.snapshot();

    let q = parse(
        "MATCH (a:Person)-[:KNOWS]->(b:Person) \
         RETURN a.name AS person, count(b) AS friend_count \
         ORDER BY person",
    )
    .unwrap();
    let plan = lower(&q).unwrap();
    let rows = execute(&plan, &snapshot, &Params::new()).await.unwrap();

    let counts: Vec<(String, i64)> = rows
        .iter()
        .map(|r| match (r.get("person"), r.get("friend_count")) {
            (Some(RuntimeValue::String(p)), Some(RuntimeValue::Integer(c))) => (p.clone(), *c),
            other => panic!("unexpected: {:?}", other),
        })
        .collect();

    // Alice -> 2 (Bob, Carol)
    // Bob   -> 1 (Carol)
    // Carol -> 1 (Dave)
    // Dave  -> 1 (Eve)
    // Eve   -> 1 (Alice)
    assert_eq!(
        counts,
        vec![
            ("Alice".to_string(), 2),
            ("Bob".to_string(), 1),
            ("Carol".to_string(), 1),
            ("Dave".to_string(), 1),
            ("Eve".to_string(), 1),
        ]
    );
}

#[tokio::test]
async fn optional_match_yields_null_when_no_neighbour() {
    // Build a graph where Eve has no out-edges via `LIKES` (we never
    // inserted any LIKES). OPTIONAL MATCH should yield `b = NULL` for
    // every Person.
    let mut writer = WriterSession::open(store(), paths("exec-opt"))
        .await
        .unwrap();
    build_friend_graph(&mut writer).await;

    // Add LIKES edge type to schema by inserting one row of it then
    // removing — actually we can just rely on memtable: declare via
    // building schema explicitly. For this test we use a relation type
    // that simply has no edges.
    let snapshot = writer.snapshot();

    let q = parse(
        "MATCH (a:Person) \
         OPTIONAL MATCH (a)-[:LIKES]->(b:Person) \
         RETURN a.name AS name, b AS friend \
         ORDER BY name",
    )
    .unwrap();
    let plan = lower(&q).unwrap();
    let rows = execute(&plan, &snapshot, &Params::new()).await.unwrap();

    // Every row should have `friend = NULL` because no LIKES edges exist.
    for row in &rows {
        assert!(matches!(row.get("friend"), Some(RuntimeValue::Null)));
    }
    assert_eq!(rows.len(), 5);
}

#[tokio::test]
async fn exists_predicate_filters_via_semiapply() {
    let mut writer = WriterSession::open(store(), paths("exec-exists"))
        .await
        .unwrap();
    build_friend_graph(&mut writer).await;
    let snapshot = writer.snapshot();

    // Eve has no out-KNOWS edges other than to Alice → still has 1 friend.
    // But Alice is the source for two KNOWS edges, while no Person is
    // unreachable. Use NOT EXISTS to find people who don't know anyone
    // — which in this graph is no one (everyone has at least one KNOWS).
    let q = parse(
        "MATCH (a:Person) \
         WHERE EXISTS((a)-[:KNOWS]->(b:Person)) \
         RETURN a.name AS name ORDER BY name",
    )
    .unwrap();
    let plan = lower(&q).unwrap();
    let rows = execute(&plan, &snapshot, &Params::new()).await.unwrap();
    let names: Vec<&str> = rows
        .iter()
        .map(|r| match r.get("name") {
            Some(RuntimeValue::String(s)) => s.as_str(),
            other => panic!("unexpected: {:?}", other),
        })
        .collect();
    // Every node has at least one outgoing KNOWS in this fixture.
    assert_eq!(names, vec!["Alice", "Bob", "Carol", "Dave", "Eve"]);
}

#[tokio::test]
async fn not_exists_predicate_excludes_matching_rows() {
    let mut writer = WriterSession::open(store(), paths("exec-not-exists"))
        .await
        .unwrap();
    build_friend_graph(&mut writer).await;
    let snapshot = writer.snapshot();

    // Bob has 1 outgoing KNOWS edge (to Carol). NOT EXISTS pattern that
    // selects people without a KNOWS to Alice should exclude Eve (Eve→Alice).
    let q = parse(
        "MATCH (a:Person) \
         WHERE NOT EXISTS((a)-[:KNOWS]->(b:Person)) \
         RETURN a.name AS name",
    )
    .unwrap();
    let plan = lower(&q).unwrap();
    let rows = execute(&plan, &snapshot, &Params::new()).await.unwrap();
    // Everyone has at least one out-KNOWS in this graph → 0 results.
    assert_eq!(rows.len(), 0);
}

#[tokio::test]
async fn exists_combined_with_scalar_predicate() {
    let mut writer = WriterSession::open(store(), paths("exec-exists-and"))
        .await
        .unwrap();
    build_friend_graph(&mut writer).await;
    let snapshot = writer.snapshot();

    let q = parse(
        "MATCH (a:Person) \
         WHERE a.age >= 30 AND EXISTS((a)-[:KNOWS]->(b:Person)) \
         RETURN a.name AS name ORDER BY name",
    )
    .unwrap();
    let plan = lower(&q).unwrap();
    let rows = execute(&plan, &snapshot, &Params::new()).await.unwrap();
    let names: Vec<&str> = rows
        .iter()
        .map(|r| match r.get("name") {
            Some(RuntimeValue::String(s)) => s.as_str(),
            other => panic!("unexpected: {:?}", other),
        })
        .collect();
    assert_eq!(names, vec!["Alice", "Carol", "Dave"]);
}

#[tokio::test]
async fn pattern_comprehension_materialises_neighbours_per_node() {
    let mut writer = WriterSession::open(store(), paths("exec-pc"))
        .await
        .unwrap();
    build_friend_graph(&mut writer).await;
    let snapshot = writer.snapshot();

    let q = parse(
        "MATCH (a:Person) \
         RETURN a.name AS name, \
                [(a)-[:KNOWS]->(b:Person) | b.name] AS friends \
         ORDER BY name",
    )
    .unwrap();
    let plan = lower(&q).unwrap();
    let rows = execute(&plan, &snapshot, &Params::new()).await.unwrap();
    let by_name: std::collections::BTreeMap<String, Vec<String>> = rows
        .iter()
        .map(|r| {
            let name = match r.get("name") {
                Some(RuntimeValue::String(s)) => s.clone(),
                other => panic!("unexpected name: {:?}", other),
            };
            let friends = match r.get("friends") {
                Some(RuntimeValue::List(items)) => items
                    .iter()
                    .map(|v| match v {
                        RuntimeValue::String(s) => s.clone(),
                        other => panic!("unexpected friend element: {:?}", other),
                    })
                    .collect::<Vec<_>>(),
                other => panic!("unexpected friends: {:?}", other),
            };
            (name, friends)
        })
        .collect();

    // Edges in build_friend_graph:
    //   Alice -> Bob, Carol      ⇒ friends = [Bob, Carol] (insertion order)
    //   Bob   -> Carol           ⇒ [Carol]
    //   Carol -> Dave            ⇒ [Dave]
    //   Dave  -> Eve             ⇒ [Eve]
    //   Eve   -> Alice           ⇒ [Alice]
    let mut alice = by_name["Alice"].clone();
    alice.sort();
    assert_eq!(alice, vec!["Bob".to_string(), "Carol".to_string()]);
    assert_eq!(by_name["Bob"], vec!["Carol".to_string()]);
    assert_eq!(by_name["Carol"], vec!["Dave".to_string()]);
    assert_eq!(by_name["Dave"], vec!["Eve".to_string()]);
    assert_eq!(by_name["Eve"], vec!["Alice".to_string()]);
}

#[tokio::test]
async fn pattern_comprehension_with_predicate_filters_elements() {
    let mut writer = WriterSession::open(store(), paths("exec-pc-pred"))
        .await
        .unwrap();
    build_friend_graph(&mut writer).await;
    let snapshot = writer.snapshot();

    // Only keep friends older than 30.
    let q = parse(
        "MATCH (a:Person) WHERE a.name = 'Alice' \
         RETURN [(a)-[:KNOWS]->(b:Person) WHERE b.age > 30 | b.name] AS older",
    )
    .unwrap();
    let plan = lower(&q).unwrap();
    let rows = execute(&plan, &snapshot, &Params::new()).await.unwrap();
    assert_eq!(rows.len(), 1);
    match rows[0].get("older") {
        // Alice's friends: Bob (25), Carol (40). Only Carol > 30.
        Some(RuntimeValue::List(items)) => {
            let names: Vec<&str> = items
                .iter()
                .map(|v| match v {
                    RuntimeValue::String(s) => s.as_str(),
                    other => panic!("unexpected: {:?}", other),
                })
                .collect();
            assert_eq!(names, vec!["Carol"]);
        }
        other => panic!("unexpected: {:?}", other),
    }
}

#[tokio::test]
async fn path_binding_materialises_alternating_node_rel_list() {
    let mut writer = WriterSession::open(store(), paths("exec-path"))
        .await
        .unwrap();
    build_friend_graph(&mut writer).await;
    let snapshot = writer.snapshot();

    let q = parse(
        "MATCH p = (a:Person)-[r:KNOWS]->(b:Person) \
         WHERE a.name = 'Alice' AND b.name = 'Bob' \
         RETURN p",
    )
    .unwrap();
    let plan = lower(&q).unwrap();
    let rows = execute(&plan, &snapshot, &Params::new()).await.unwrap();
    assert_eq!(rows.len(), 1);
    match rows[0].get("p") {
        Some(RuntimeValue::Path(items)) => {
            assert_eq!(items.len(), 3);
            match (&items[0], &items[1], &items[2]) {
                (RuntimeValue::Node(a), RuntimeValue::Rel(r), RuntimeValue::Node(b)) => {
                    let aname = match a.properties.get("name") {
                        Some(RuntimeValue::String(s)) => s.as_str(),
                        _ => "?",
                    };
                    let bname = match b.properties.get("name") {
                        Some(RuntimeValue::String(s)) => s.as_str(),
                        _ => "?",
                    };
                    assert_eq!(aname, "Alice");
                    assert_eq!(bname, "Bob");
                    assert_eq!(r.edge_type, "KNOWS");
                }
                other => panic!("unexpected path triple: {:?}", other),
            }
        }
        other => panic!("expected Path, got {:?}", other),
    }
}

#[tokio::test]
async fn path_binding_two_hop_chain() {
    let mut writer = WriterSession::open(store(), paths("exec-path-2h"))
        .await
        .unwrap();
    build_friend_graph(&mut writer).await;
    let snapshot = writer.snapshot();

    let q = parse(
        "MATCH p = (a:Person)-[r1:KNOWS]->(b:Person)-[r2:KNOWS]->(c:Person) \
         WHERE a.name = 'Alice' AND c.name = 'Carol' \
         RETURN p",
    )
    .unwrap();
    let plan = lower(&q).unwrap();
    let rows = execute(&plan, &snapshot, &Params::new()).await.unwrap();
    // Alice → Bob → Carol via the seeded edges.
    assert!(!rows.is_empty(), "expected at least one 2-hop path");
    match rows[0].get("p") {
        Some(RuntimeValue::Path(items)) => {
            assert_eq!(items.len(), 5);
        }
        other => panic!("expected Path, got {:?}", other),
    }
}

#[tokio::test]
async fn path_binding_rejects_anonymous_rel() {
    let q = parse("MATCH p = (a:Person)-[:KNOWS]->(b:Person) RETURN p").unwrap();
    let err = lower(&q).expect_err("expected lowering error");
    assert!(
        err.message
            .contains("relationship to have an explicit alias"),
        "unexpected message: {}",
        err.message
    );
}

#[tokio::test]
async fn return_star_exports_all_named_bindings() {
    let mut writer = WriterSession::open(store(), paths("exec-star"))
        .await
        .unwrap();
    build_friend_graph(&mut writer).await;
    let snapshot = writer.snapshot();

    let q = parse(
        "MATCH (a:Person)-[:KNOWS]->(b:Person) \
         WHERE a.name = 'Alice' \
         RETURN * ORDER BY b.name",
    )
    .unwrap();
    let plan = lower(&q).unwrap();
    let rows = execute(&plan, &snapshot, &Params::new()).await.unwrap();

    assert_eq!(rows.len(), 2);
    for row in &rows {
        let a = row.get("a").expect("a binding");
        let b = row.get("b").expect("b binding");
        assert!(matches!(a, RuntimeValue::Node(_)));
        assert!(matches!(b, RuntimeValue::Node(_)));
    }
}

/// Smoke test: build a schema explicitly via SchemaBuilder so we can
/// confirm scan_label still works when the manifest knows about the
/// label. This is the path the SaaS gateway will use.
#[tokio::test]
async fn explicit_schema_round_trip() {
    let mut writer = WriterSession::open(store(), paths("exec-schema"))
        .await
        .unwrap();
    let _ids = build_friend_graph(&mut writer).await;
    let _schema = SchemaBuilder::new()
        .label(person_label())
        .unwrap()
        .edge_type(knows_edge())
        .unwrap()
        .build();
    let snapshot = writer.snapshot();
    let q = parse("MATCH (a:Person) RETURN count(*) AS n").unwrap();
    let plan = lower(&q).unwrap();
    let rows = execute(&plan, &snapshot, &Params::new()).await.unwrap();
    assert_eq!(rows[0].get("n"), Some(&RuntimeValue::Integer(5)));
}

#[tokio::test]
async fn match_expand_without_rel_type_enumerates_all_edge_types() {
    // Regression for Bug #4: `MATCH (a)-[r]->(b)` without an explicit
    // edge type must traverse every type declared on the manifest
    // schema. Previously the executor aborted with "Expand requires
    // explicit edge type".
    let mut writer = WriterSession::open(store(), paths("exec-no-rel-type"))
        .await
        .unwrap();
    let alice = NodeId::new();
    let bob = NodeId::new();
    let carol = NodeId::new();
    writer
        .upsert_node("Person", alice, &person("Alice", 30))
        .unwrap();
    writer
        .upsert_node("Person", bob, &person("Bob", 25))
        .unwrap();
    writer
        .upsert_node("Person", carol, &person("Carol", 40))
        .unwrap();
    writer.upsert_edge("KNOWS", alice, bob, &edge()).unwrap();
    writer.upsert_edge("LIKES", bob, carol, &edge()).unwrap();
    writer.commit_batch().await.unwrap();

    let snapshot = writer.snapshot();
    let q = parse(
        "MATCH (a:Person)-[r]->(b:Person) \
         RETURN a.name AS from, b.name AS to \
         ORDER BY from, to",
    )
    .unwrap();
    let plan = lower(&q).unwrap();
    let rows = execute(&plan, &snapshot, &Params::new()).await.unwrap();
    let pairs: Vec<(String, String)> = rows
        .iter()
        .map(|r| match (r.get("from"), r.get("to")) {
            (Some(RuntimeValue::String(a)), Some(RuntimeValue::String(b))) => {
                (a.clone(), b.clone())
            }
            other => panic!("unexpected: {:?}", other),
        })
        .collect();
    assert_eq!(
        pairs,
        vec![
            ("Alice".to_string(), "Bob".to_string()),
            ("Bob".to_string(), "Carol".to_string()),
        ],
    );
}

#[tokio::test]
async fn var_length_expand_without_rel_type_crosses_heterogeneous_types() {
    // Regression for Bug #6: `[*1..N]` without an explicit edge type
    // must traverse every type at each hop. With `Alice -KNOWS-> Bob
    // -LIKES-> Carol`, the pattern `(Alice)-[*1..2]->(c)` should reach
    // Bob at hop 1 (via KNOWS) and Carol at hop 2 (KNOWS then LIKES).
    let mut writer = WriterSession::open(store(), paths("exec-no-rel-type-varlen"))
        .await
        .unwrap();
    let alice = NodeId::new();
    let bob = NodeId::new();
    let carol = NodeId::new();
    writer
        .upsert_node("Person", alice, &person("Alice", 30))
        .unwrap();
    writer
        .upsert_node("Person", bob, &person("Bob", 25))
        .unwrap();
    writer
        .upsert_node("Person", carol, &person("Carol", 40))
        .unwrap();
    writer.upsert_edge("KNOWS", alice, bob, &edge()).unwrap();
    writer.upsert_edge("LIKES", bob, carol, &edge()).unwrap();
    writer.commit_batch().await.unwrap();

    let snapshot = writer.snapshot();
    let q = parse(
        "MATCH (a:Person {name: 'Alice'})-[*1..2]->(c:Person) \
         RETURN c.name AS reached \
         ORDER BY reached",
    )
    .unwrap();
    let plan = lower(&q).unwrap();
    let rows = execute(&plan, &snapshot, &Params::new()).await.unwrap();
    let reached: Vec<String> = rows
        .iter()
        .map(|r| match r.get("reached") {
            Some(RuntimeValue::String(s)) => s.clone(),
            other => panic!("unexpected: {:?}", other),
        })
        .collect();
    assert_eq!(reached, vec!["Bob".to_string(), "Carol".to_string()]);
}
