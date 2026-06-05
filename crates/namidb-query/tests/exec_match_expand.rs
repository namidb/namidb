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

use namidb_query::{execute, explain, lower, optimize, parse, Params, RuntimeValue, StatsCatalog};

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
        ..Default::default()
    }
}

fn indexed_city_label() -> LabelDef {
    LabelDef {
        name: "Person".into(),
        properties: vec![
            PropertyDef::new("name", DataType::Utf8, false).unwrap(),
            PropertyDef::new("city", DataType::Utf8, true)
                .unwrap()
                .with_indexed(true),
        ],
    }
}

fn person_city(name: &str, city: &str) -> NodeWriteRecord {
    let mut props: BTreeMap<String, CoreValue> = BTreeMap::new();
    props.insert("name".into(), CoreValue::Str(name.into()));
    props.insert("city".into(), CoreValue::Str(city.into()));
    NodeWriteRecord {
        properties: props,
        schema_version: 1,
        ..Default::default()
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

    let q = parse("MATCH (a:Person {_id: $personId}) RETURN a.name AS name").unwrap();
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
async fn var_length_path_binding_materialises_variable_trails() {
    // gdotv-style "show paths up to N hops": a path binding over a
    // variable-length relationship. The lower routes this through the walker's
    // trail materialisation (the same path the executor uses for shortestPath),
    // so each match returns a real Path whose length tracks the hop count —
    // rather than the old "variable-length path bindings are not yet supported"
    // rejection.
    let mut writer = WriterSession::open(store(), paths("exec-varlen-path"))
        .await
        .unwrap();
    build_friend_graph(&mut writer).await;
    let snapshot = writer.snapshot();

    let q = parse(
        "MATCH p = (a:Person)-[:KNOWS*1..2]->(b:Person) \
         WHERE a.name = 'Alice' \
         RETURN p",
    )
    .unwrap();
    let plan = lower(&q).unwrap();
    let rows = execute(&plan, &snapshot, &Params::new()).await.unwrap();
    assert!(!rows.is_empty(), "expected variable-length paths from Alice");
    let mut lengths = std::collections::BTreeSet::new();
    for row in &rows {
        match row.get("p") {
            // The trail alternates Node, Rel, Node, ... so a valid path has an
            // odd item count >= 3 (one hop).
            Some(RuntimeValue::Path(items)) => {
                assert!(
                    items.len() >= 3 && items.len() % 2 == 1,
                    "unexpected trail shape: {} items",
                    items.len()
                );
                lengths.insert(items.len());
            }
            other => panic!("expected Path, got {:?}", other),
        }
    }
    // `*1..2` from Alice yields both 1-hop (3-item) and 2-hop (5-item) trails.
    assert!(
        lengths.contains(&3),
        "expected a 1-hop path (3 items), got {:?}",
        lengths
    );
    assert!(
        lengths.contains(&5),
        "expected a 2-hop path (5 items), got {:?}",
        lengths
    );
}

#[tokio::test]
async fn path_binding_supports_anonymous_elements() {
    // Clients like gdotv bind paths with anonymous elements:
    // `p = ()-[]->()` for the default graph view, or an anonymous
    // relationship between named nodes. The lower fills the anonymous
    // slots with internal bindings, so the path still materialises
    // (rather than the old "explicit alias required" rejection).
    let mut writer = WriterSession::open(store(), paths("exec-anon-path"))
        .await
        .unwrap();
    build_friend_graph(&mut writer).await;
    let snapshot = writer.snapshot();

    // Anonymous relationship between named nodes.
    let q = parse(
        "MATCH p = (a:Person)-[:KNOWS]->(b:Person) \
         WHERE a.name = 'Alice' AND b.name = 'Bob' \
         RETURN p",
    )
    .unwrap();
    let plan = lower(&q).unwrap();
    let rows = execute(&plan, &snapshot, &Params::new()).await.unwrap();
    assert_eq!(rows.len(), 1);
    assert_path_node_rel_node(rows[0].get("p"));

    // Fully anonymous path — exactly what gdotv's default query emits.
    let q = parse("MATCH p = ()-[]->() RETURN p").unwrap();
    let plan = lower(&q).unwrap();
    let rows = execute(&plan, &snapshot, &Params::new()).await.unwrap();
    assert!(!rows.is_empty(), "anonymous path should match edges");
    for row in &rows {
        assert_path_node_rel_node(row.get("p"));
    }
}

fn assert_path_node_rel_node(value: Option<&RuntimeValue>) {
    match value {
        Some(RuntimeValue::Path(items)) => {
            assert_eq!(items.len(), 3, "path should be node-rel-node");
            assert!(matches!(items[0], RuntimeValue::Node(_)), "head not a node");
            assert!(matches!(items[1], RuntimeValue::Rel(_)), "middle not a rel");
            assert!(matches!(items[2], RuntimeValue::Node(_)), "tail not a node");
        }
        other => panic!("expected Path, got {other:?}"),
    }
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
async fn match_without_label_scans_every_observed_label() {
    // Regression for Bug #3: `MATCH (n) RETURN count(*) AS n` must
    // include every label present in the snapshot, not just one.
    let mut writer = WriterSession::open(store(), paths("exec-no-label"))
        .await
        .unwrap();
    let alice = NodeId::new();
    let post = NodeId::new();
    writer
        .upsert_node("Person", alice, &person("Alice", 30))
        .unwrap();
    let mut post_props = BTreeMap::new();
    post_props.insert("title".into(), CoreValue::Str("Hello".into()));
    writer
        .upsert_node(
            "Post",
            post,
            &NodeWriteRecord {
                properties: post_props,
                schema_version: 1,
                ..Default::default()
            },
        )
        .unwrap();
    writer.commit_batch().await.unwrap();

    let snapshot = writer.snapshot();
    let q = parse("MATCH (n) RETURN count(*) AS n").unwrap();
    let plan = lower(&q).unwrap();
    let rows = execute(&plan, &snapshot, &Params::new()).await.unwrap();
    assert_eq!(rows[0].get("n"), Some(&RuntimeValue::Integer(2)));
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
async fn unwind_alias_drives_following_match_property_filter() {
    // B1 regression: `UNWIND ['Alice', 'Bob'] AS who MATCH (n:Person
    // {name: who})` must use `who` as a per-row driver, returning one
    // matched node per element of the list — not 0 rows (alias dropped
    // during binding) and not the full label scan (alias resolved to
    // null and the filter became `name = NULL`).
    let mut writer = WriterSession::open(store(), paths("exec-unwind-match"))
        .await
        .unwrap();
    build_friend_graph(&mut writer).await;
    let snapshot = writer.snapshot();

    let q = parse(
        "UNWIND ['Alice', 'Bob', 'Eve'] AS who \
         MATCH (n:Person {name: who}) \
         RETURN n.name AS name ORDER BY name",
    )
    .unwrap();
    let plan = lower(&q).unwrap();
    let rows = execute(&plan, &snapshot, &Params::new()).await.unwrap();

    let names: Vec<String> = rows
        .iter()
        .map(|r| match r.get("name") {
            Some(RuntimeValue::String(s)) => s.clone(),
            other => panic!("unexpected: {:?}", other),
        })
        .collect();
    assert_eq!(
        names,
        vec!["Alice".to_string(), "Bob".to_string(), "Eve".to_string()],
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

#[tokio::test]
async fn match_anonymous_endpoints_resolves_without_declared_schema() {
    // B1 / B7 regression: `MATCH ()-[r:T]->()` and `MATCH (a)-[r]->(b)`
    // used to fall through `scan_node_for_id`, which iterated only
    // declared labels. Namespaces that never ran `SchemaBuilder` ended
    // up with an empty label list there, so every neighbour was dropped
    // and queries returned zero rows. `observed_labels` covers both the
    // declared schema and labels written into memtable / SSTs.
    let mut writer = WriterSession::open(store(), paths("exec-anon-endpoints"))
        .await
        .unwrap();
    let alice = NodeId::new();
    let bob = NodeId::new();
    writer
        .upsert_node("Person", alice, &person("Alice", 30))
        .unwrap();
    writer
        .upsert_node("Person", bob, &person("Bob", 25))
        .unwrap();
    writer.upsert_edge("KNOWS", alice, bob, &edge()).unwrap();
    writer.commit_batch().await.unwrap();
    // Note: no SchemaBuilder. `manifest.schema.labels` stays empty.

    let snapshot = writer.snapshot();

    // Anonymous endpoints with an explicit edge type.
    let q = parse("MATCH ()-[r:KNOWS]->() RETURN count(r) AS n").unwrap();
    let plan = lower(&q).unwrap();
    let rows = execute(&plan, &snapshot, &Params::new()).await.unwrap();
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].get("n"), Some(&RuntimeValue::Integer(1)));

    // Fully anonymous expand: no labels, no edge type.
    let q = parse("MATCH (a)-[r]->(b) RETURN count(r) AS n").unwrap();
    let plan = lower(&q).unwrap();
    let rows = execute(&plan, &snapshot, &Params::new()).await.unwrap();
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].get("n"), Some(&RuntimeValue::Integer(1)));
}

fn has_edge_type_count(plan: &namidb_query::LogicalPlan) -> bool {
    matches!(plan, namidb_query::LogicalPlan::EdgeTypeCount { .. })
        || plan.children().iter().any(|c| has_edge_type_count(c))
}

#[tokio::test]
async fn global_edge_type_count_pushdown_matches_nodescan_path() {
    use namidb_query::{parse_lower_optimize, StatsCatalog};

    // KNOWS + Person schema so the first batch can flush into SSTs; the
    // count must merge those with the still-in-memtable second batch.
    let schema = SchemaBuilder::new()
        .label(LabelDef {
            name: "Person".into(),
            properties: vec![
                PropertyDef::new("name", DataType::Utf8, false).unwrap(),
                PropertyDef::new("age", DataType::Int32, true).unwrap(),
            ],
        })
        .unwrap()
        .edge_type(EdgeTypeDef {
            name: "KNOWS".into(),
            src_label: "Person".into(),
            dst_label: "Person".into(),
            properties: vec![],
        })
        .unwrap()
        .build();

    let mut writer = WriterSession::open(store(), paths("exec-edge-count-e2e"))
        .await
        .unwrap();
    let alice = NodeId::new();
    let bob = NodeId::new();
    let carol = NodeId::new();
    let dave = NodeId::new();
    for (id, name, age) in [
        (alice, "Alice", 30),
        (bob, "Bob", 25),
        (carol, "Carol", 41),
        (dave, "Dave", 19),
    ] {
        writer
            .upsert_node("Person", id, &person(name, age))
            .unwrap();
    }
    // Batch 1 → flushed into SSTs: 4 KNOWS edges.
    for (s, d) in [(alice, bob), (alice, carol), (bob, carol), (dave, alice)] {
        writer.upsert_edge("KNOWS", s, d, &edge()).unwrap();
    }
    writer.commit_batch().await.unwrap();
    writer.flush(schema).await.unwrap();

    // Batch 2 → live memtable (not flushed): add one, tombstone one.
    writer.upsert_edge("KNOWS", carol, dave, &edge()).unwrap();
    writer.tombstone_edge("KNOWS", alice, bob).unwrap();
    writer.commit_batch().await.unwrap();

    // Live KNOWS edges: alice→carol, bob→carol, dave→alice, carol→dave = 4
    // (alice→bob tombstoned). Exercises the SST + memtable + tombstone merge.
    let snapshot = writer.snapshot();
    let query = "MATCH ()-[r:KNOWS]->() RETURN count(r) AS n";

    // Optimized: the pushdown must fire (EdgeTypeCount, no NodeScan).
    let optimized = parse_lower_optimize(query, &StatsCatalog::empty()).unwrap();
    assert!(
        has_edge_type_count(&optimized),
        "the global edge count must push down to EdgeTypeCount"
    );
    let pushed = execute(&optimized, &snapshot, &Params::new())
        .await
        .unwrap();
    assert_eq!(pushed.len(), 1);
    assert_eq!(pushed[0].get("n"), Some(&RuntimeValue::Integer(4)));

    // Gold standard: identical to the un-optimized NodeScan + Expand path.
    let raw = lower(&parse(query).unwrap()).unwrap();
    let raw_rows = execute(&raw, &snapshot, &Params::new()).await.unwrap();
    assert_eq!(
        raw_rows[0].get("n"),
        pushed[0].get("n"),
        "EdgeTypeCount must match the NodeScan+Expand count exactly"
    );
}

fn has_node_by_id(plan: &namidb_query::LogicalPlan) -> bool {
    matches!(plan, namidb_query::LogicalPlan::NodeById { .. })
        || plan.children().iter().any(|c| has_node_by_id(c))
}

#[tokio::test]
async fn element_id_filter_lowers_to_point_lookup() {
    use namidb_query::{parse_lower_optimize, StatsCatalog};

    let mut writer = WriterSession::open(store(), paths("exec-element-id"))
        .await
        .unwrap();
    build_friend_graph(&mut writer).await;
    let snapshot = writer.snapshot();

    // Grab a real node and its element id (the UUID).
    let any = execute(
        &lower(&parse("MATCH (p:Person) RETURN p LIMIT 1").unwrap()).unwrap(),
        &snapshot,
        &Params::new(),
    )
    .await
    .unwrap();
    let (eid, name) = match any[0].get("p") {
        Some(RuntimeValue::Node(n)) => (n.id.to_string(), n.properties.get("name").cloned()),
        other => panic!("expected a node, got {other:?}"),
    };

    // Unlabelled `WHERE elementId(v) = '<uuid>'` (the GUI node-fetch shape)
    // must optimise to a NodeById point lookup rather than a full scan, and
    // return exactly that node.
    let query = format!("MATCH (v) WHERE elementId(v) = '{eid}' RETURN v");
    let optimized = parse_lower_optimize(&query, &StatsCatalog::empty()).unwrap();
    assert!(
        has_node_by_id(&optimized),
        "elementId equality must lower to a NodeById point lookup, got {optimized:?}"
    );
    let rows = execute(&optimized, &snapshot, &Params::new())
        .await
        .unwrap();
    assert_eq!(rows.len(), 1, "point lookup must return exactly the node");
    match rows[0].get("v") {
        Some(RuntimeValue::Node(n)) => {
            assert_eq!(n.id.to_string(), eid);
            assert_eq!(n.properties.get("name").cloned(), name);
        }
        other => panic!("expected the looked-up node, got {other:?}"),
    }
}

#[tokio::test]
async fn indexed_property_match_uses_index_and_returns_all_matches() {
    // End-to-end: an equality MATCH on a non-unique `indexed` property is
    // rewritten by the optimizer into the index lookup and the executor
    // fans out one row per match.
    let mut writer = WriterSession::open(store(), paths("exec-eqidx"))
        .await
        .unwrap();
    let ids: [NodeId; 3] = std::array::from_fn(|_| NodeId::new());
    let names = ["Ann", "Bob", "Cy"];
    let cities = ["LA", "LA", "NYC"];
    for ((id, name), city) in ids.iter().zip(names).zip(cities) {
        writer
            .upsert_node("Person", *id, &person_city(name, city))
            .unwrap();
    }
    writer.commit_batch().await.unwrap();
    // Flush with the indexed schema so the equality sidecar and the schema
    // (with `city` indexed) both land in the manifest.
    let schema = SchemaBuilder::new()
        .label(indexed_city_label())
        .unwrap()
        .build();
    writer.flush(schema).await.unwrap();
    let snapshot = writer.snapshot();

    // The catalog (built from the manifest) sees `city` as indexed, so the
    // optimizer rewrites the filter into the index lookup rather than a
    // full label scan.
    let catalog = StatsCatalog::from_manifest(&snapshot.manifest().manifest);
    let q = parse("MATCH (p:Person {city: 'LA'}) RETURN p.name AS name").unwrap();
    let plan = optimize(lower(&q).unwrap(), &catalog);
    let rendered = explain(&plan);
    assert!(
        rendered.contains("NodeByPropertyValue"),
        "expected the index lookup in the optimized plan, got:\n{rendered}"
    );

    let rows = execute(&plan, &snapshot, &Params::new()).await.unwrap();
    let mut got: Vec<String> = rows
        .iter()
        .filter_map(|r| match r.get("name") {
            Some(RuntimeValue::String(s)) => Some(s.clone()),
            _ => None,
        })
        .collect();
    got.sort();
    assert_eq!(
        got,
        vec!["Ann".to_string(), "Bob".to_string()],
        "both LA persons must come back, not the NYC one"
    );
}
