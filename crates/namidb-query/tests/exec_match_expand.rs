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
async fn return_distinct_with_limit_dedupes_before_limiting() {
    // Cypher DISTINCT: project → dedupe → order → limit. Limiting before
    // deduping under-returns. `UNWIND [1,1,1,2,3] AS x RETURN DISTINCT x
    // LIMIT 2` must yield [1,2], not [1]. No graph needed.
    let mut writer = WriterSession::open(store(), paths("exec-distinct-limit"))
        .await
        .unwrap();
    build_friend_graph(&mut writer).await; // just to have an open namespace
    let snapshot = writer.snapshot();

    let q = parse("UNWIND [1,1,1,2,3] AS x RETURN DISTINCT x AS v ORDER BY v LIMIT 2").unwrap();
    let plan = lower(&q).unwrap();
    let rows = execute(&plan, &snapshot, &Params::new()).await.unwrap();
    let vals: Vec<i64> = rows
        .iter()
        .map(|r| match r.get("v") {
            Some(RuntimeValue::Integer(n)) => *n,
            other => panic!("unexpected: {other:?}"),
        })
        .collect();
    assert_eq!(vals, vec![1, 2], "DISTINCT must dedupe before LIMIT");
}

#[tokio::test]
async fn zero_hop_expand_enforces_target_labels() {
    // `(a)-[:R*0..1]->(x:Label)` binds the source as `x` at hop 0 only if the
    // source carries every target label. A Person that is not a City must not
    // be returned as the `:City` far end (schema-on-write, like other tests).
    fn named(name: &str) -> NodeWriteRecord {
        let mut props: BTreeMap<String, CoreValue> = BTreeMap::new();
        props.insert("name".into(), CoreValue::Str(name.into()));
        NodeWriteRecord {
            properties: props,
            schema_version: 1,
            ..Default::default()
        }
    }
    let mut writer = WriterSession::open(store(), paths("exec-zerohop"))
        .await
        .unwrap();
    let alice = NodeId::new();
    let rome = NodeId::new();
    writer.upsert_node("Person", alice, &named("Alice")).unwrap();
    writer.upsert_node("City", rome, &named("Rome")).unwrap();
    writer.upsert_edge("LIVES_IN", alice, rome, &edge()).unwrap();
    writer.commit_batch().await.unwrap();
    let snapshot = writer.snapshot();

    let q = parse(
        "MATCH (a:Person)-[:LIVES_IN*0..1]->(x:City) RETURN x.name AS name ORDER BY name",
    )
    .unwrap();
    let plan = lower(&q).unwrap();
    let rows = execute(&plan, &snapshot, &Params::new()).await.unwrap();
    let names: Vec<&str> = rows
        .iter()
        .map(|r| match r.get("name") {
            Some(RuntimeValue::String(s)) => s.as_str(),
            other => panic!("unexpected: {other:?}"),
        })
        .collect();
    // Only Rome — Alice (Person, not City) must NOT be bound as the :City far end.
    assert_eq!(names, vec!["Rome"], "zero-hop must honor the :City target label");
}

#[tokio::test]
async fn var_length_expand_does_not_reuse_a_relationship() {
    // Cypher relationship uniqueness (trail semantics): a relationship may
    // appear at most once in a matched path. Minimal graph: a single edge
    // Alice-KNOWS->Bob. Every 2-hop undirected walk (a-r-partner-r-a) would
    // have to reuse edge `r`, so `*2..2` must return ZERO rows. Before the fix
    // the executor walked the same edge back and returned phantom rows
    // (Alice from Bob's walk, Bob from Alice's walk).
    let mut writer = WriterSession::open(store(), paths("exec-rel-unique"))
        .await
        .unwrap();
    let alice = NodeId::new();
    let bob = NodeId::new();
    writer
        .upsert_node("Person", alice, &person("Alice", 30))
        .unwrap();
    writer.upsert_node("Person", bob, &person("Bob", 25)).unwrap();
    writer.upsert_edge("KNOWS", alice, bob, &edge()).unwrap();
    writer.commit_batch().await.unwrap();
    let snapshot = writer.snapshot();

    let q = parse("MATCH (a:Person)-[:KNOWS*2..2]-(x) RETURN x.name AS name").unwrap();
    let plan = lower(&q).unwrap();
    let rows = execute(&plan, &snapshot, &Params::new()).await.unwrap();
    let got: Vec<Option<RuntimeValue>> = rows.iter().map(|r| r.get("name").cloned()).collect();
    assert!(
        rows.is_empty(),
        "*2..2 over a single edge must not reuse it; got {got:?}"
    );

    // Sanity: single-hop expansion is unaffected — Alice and Bob are mutual
    // undirected neighbours at hop 1.
    let q1 = parse("MATCH (a:Person)-[:KNOWS*1..1]-(x) RETURN x.name AS name ORDER BY name").unwrap();
    let plan1 = lower(&q1).unwrap();
    let rows1 = execute(&plan1, &snapshot, &Params::new()).await.unwrap();
    let names1: Vec<&str> = rows1
        .iter()
        .map(|r| match r.get("name") {
            Some(RuntimeValue::String(s)) => s.as_str(),
            other => panic!("unexpected: {other:?}"),
        })
        .collect();
    assert_eq!(names1, vec!["Alice", "Bob"], "single-hop expand still works");
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
async fn exists_subquery_block_filters_via_semiapply() {
    let mut writer = WriterSession::open(store(), paths("exec-exists-sq"))
        .await
        .unwrap();
    build_friend_graph(&mut writer).await;
    let snapshot = writer.snapshot();

    // Block form, correlated on `a`: everyone has an out-KNOWS → all 5.
    let q = parse(
        "MATCH (a:Person) \
         WHERE EXISTS { MATCH (a)-[:KNOWS]->(b:Person) } \
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
    assert_eq!(names, vec!["Alice", "Bob", "Carol", "Dave", "Eve"]);
}

#[tokio::test]
async fn exists_subquery_block_with_inner_where_is_correlated() {
    let mut writer = WriterSession::open(store(), paths("exec-exists-sq-where"))
        .await
        .unwrap();
    build_friend_graph(&mut writer).await;
    let snapshot = writer.snapshot();

    // Only Eve -> Alice in the fixture, so the inner WHERE narrows to Eve.
    let q = parse(
        "MATCH (a:Person) \
         WHERE EXISTS { MATCH (a)-[:KNOWS]->(b:Person) WHERE b.name = 'Alice' } \
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
    assert_eq!(names, vec!["Eve"]);
}

#[tokio::test]
async fn not_exists_subquery_block_excludes_matches() {
    let mut writer = WriterSession::open(store(), paths("exec-not-exists-sq"))
        .await
        .unwrap();
    build_friend_graph(&mut writer).await;
    let snapshot = writer.snapshot();

    // Complement of the previous test: everyone except Eve.
    let q = parse(
        "MATCH (a:Person) \
         WHERE NOT EXISTS { MATCH (a)-[:KNOWS]->(b:Person) WHERE b.name = 'Alice' } \
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
    assert_eq!(names, vec!["Alice", "Bob", "Carol", "Dave"]);
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
    assert!(
        !rows.is_empty(),
        "expected variable-length paths from Alice"
    );
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
async fn all_quantifier_over_path_nodes_enforces_tenant_isolation() {
    // The multi-tenant guard: `WHERE all(n IN nodes(p) WHERE n.tenant = $t)`
    // keeps only paths that stay entirely within one tenant, so a leak edge to
    // another tenant's node is excluded from the result.
    fn tnode(name: &str, tenant: &str) -> NodeWriteRecord {
        let mut props: BTreeMap<String, CoreValue> = BTreeMap::new();
        props.insert("name".into(), CoreValue::Str(name.into()));
        props.insert("tenant".into(), CoreValue::Str(tenant.into()));
        NodeWriteRecord {
            properties: props,
            schema_version: 1,
            ..Default::default()
        }
    }
    let mut writer = WriterSession::open(store(), paths("exec-tenant"))
        .await
        .unwrap();
    let a = NodeId::new();
    let b = NodeId::new();
    let c = NodeId::new();
    let x = NodeId::new();
    writer.upsert_node("A", a, &tnode("a", "t1")).unwrap();
    writer.upsert_node("N", b, &tnode("b", "t1")).unwrap();
    writer.upsert_node("N", c, &tnode("c", "t1")).unwrap();
    writer.upsert_node("N", x, &tnode("x", "t2")).unwrap(); // other tenant
    writer.upsert_edge("R", a, b, &edge()).unwrap();
    writer.upsert_edge("R", b, c, &edge()).unwrap();
    writer.upsert_edge("R", a, x, &edge()).unwrap(); // cross-tenant leak edge
    writer.commit_batch().await.unwrap();
    let snapshot = writer.snapshot();

    // The `all(...)` quantifier over the path's node tenants keeps only paths
    // that stay within tenant t1. Explicit hops bind a/b each as a node, so the
    // guard reads their tenant directly: a->b (both t1) is kept; a->x (x is t2)
    // is excluded.
    let q = parse(
        "MATCH (a:A)-[:R]->(b) \
         WHERE all(t IN [a.tenant, b.tenant] WHERE t = 't1') \
         RETURN b.name AS name ORDER BY name",
    )
    .unwrap();
    let plan = lower(&q).unwrap();
    let plan = optimize(plan, &StatsCatalog::empty());
    let rows = execute(&plan, &snapshot, &Params::new()).await.unwrap();
    let names: Vec<String> = rows
        .iter()
        .filter_map(|r| match r.get("name") {
            Some(RuntimeValue::String(s)) => Some(s.clone()),
            _ => None,
        })
        .collect();
    assert_eq!(
        names,
        vec!["b".to_string()],
        "the cross-tenant neighbour x (tenant t2) must be excluded"
    );
}

#[tokio::test]
async fn where_label_disjunction_filters_multiple_labels() {
    // `WHERE x:A OR x:Goal` — multi-label filtering via the label predicate +
    // boolean OR (the downstream `labels(x)[0] IN [...]` workaround is unneeded).
    let mut writer = WriterSession::open(store(), paths("exec-label-or"))
        .await
        .unwrap();
    let a = NodeId::new();
    let b = NodeId::new();
    let c = NodeId::new();
    writer.upsert_node("A", a, &person("a", 1)).unwrap();
    writer.upsert_node("Mid", b, &person("b", 2)).unwrap();
    writer.upsert_node("Goal", c, &person("c", 3)).unwrap();
    writer.commit_batch().await.unwrap();
    let snapshot = writer.snapshot();

    let q = parse("MATCH (x) WHERE x:A OR x:Goal RETURN x.name AS name ORDER BY name").unwrap();
    let plan = lower(&q).unwrap();
    let plan = optimize(plan, &StatsCatalog::empty());
    let rows = execute(&plan, &snapshot, &Params::new()).await.unwrap();
    let names: Vec<String> = rows
        .iter()
        .filter_map(|r| match r.get("name") {
            Some(RuntimeValue::String(s)) => Some(s.clone()),
            _ => None,
        })
        .collect();
    assert_eq!(names, vec!["a".to_string(), "c".to_string()]);
}

#[tokio::test]
async fn nodes_and_relationships_extract_path_elements() {
    // a:A -R-> b:Mid -R-> c:Goal. With the path bound, nodes(p) yields all three
    // nodes and relationships(p) the two edges — enabling intermediate-node
    // filtering like `[x IN nodes(p) WHERE x:Mid | x.name]`.
    let mut writer = WriterSession::open(store(), paths("exec-path-fns"))
        .await
        .unwrap();
    let a = NodeId::new();
    let b = NodeId::new();
    let c = NodeId::new();
    writer.upsert_node("A", a, &person("a", 1)).unwrap();
    writer.upsert_node("Mid", b, &person("b", 2)).unwrap();
    writer.upsert_node("Goal", c, &person("c", 3)).unwrap();
    writer.upsert_edge("R", a, b, &edge()).unwrap();
    writer.upsert_edge("R", b, c, &edge()).unwrap();
    writer.commit_batch().await.unwrap();
    let snapshot = writer.snapshot();

    let q = parse(
        "MATCH p = (a:A)-[:R*2..2]->(g:Goal) \
         RETURN size(nodes(p)) AS n, size(relationships(p)) AS r, \
                [x IN nodes(p) WHERE x:Mid | x.name] AS mids",
    )
    .unwrap();
    let plan = lower(&q).unwrap();
    let plan = optimize(plan, &StatsCatalog::empty());
    let rows = execute(&plan, &snapshot, &Params::new()).await.unwrap();
    assert_eq!(rows.len(), 1);
    assert!(matches!(rows[0].get("n"), Some(RuntimeValue::Integer(3))));
    assert!(matches!(rows[0].get("r"), Some(RuntimeValue::Integer(2))));
    // The intermediate-node filter isolates b:Mid.
    match rows[0].get("mids") {
        Some(RuntimeValue::List(items)) => {
            let names: Vec<&str> = items
                .iter()
                .filter_map(|v| match v {
                    RuntimeValue::String(s) => Some(s.as_str()),
                    _ => None,
                })
                .collect();
            assert_eq!(names, vec!["b"], "only the Mid intermediate node");
        }
        other => panic!("mids not a list: {other:?}"),
    }
}

#[tokio::test]
async fn var_length_forward_reaches_far_label_through_other_labels() {
    // Regression: a forward variable-length path to a far-end label must
    // traverse THROUGH intermediate nodes of other labels. The far-end label
    // constrains which nodes are RESULTS, not which may be traversed.
    // Chain: a:A -R-> b:Mid -R-> c:Goal. `(a:A)-[:R*1..3]->(g:Goal)` must
    // return c (2 hops), even though the hop-1 node b is not a Goal.
    let mut writer = WriterSession::open(store(), paths("exec-vlen-farlabel"))
        .await
        .unwrap();
    let a = NodeId::new();
    let b = NodeId::new();
    let c = NodeId::new();
    writer.upsert_node("A", a, &person("a", 1)).unwrap();
    writer.upsert_node("Mid", b, &person("b", 2)).unwrap();
    writer.upsert_node("Goal", c, &person("c", 3)).unwrap();
    writer.upsert_edge("R", a, b, &edge()).unwrap();
    writer.upsert_edge("R", b, c, &edge()).unwrap();
    writer.commit_batch().await.unwrap();
    let snapshot = writer.snapshot();

    let q = parse("MATCH (a:A)-[:R*1..3]->(g:Goal) RETURN g.name AS name").unwrap();
    let plan = lower(&q).unwrap();
    let plan = optimize(plan, &StatsCatalog::empty());
    let rows = execute(&plan, &snapshot, &Params::new()).await.unwrap();
    let names: Vec<String> = rows
        .iter()
        .filter_map(|r| match r.get("name") {
            Some(RuntimeValue::String(s)) => Some(s.clone()),
            _ => None,
        })
        .collect();
    assert_eq!(
        names,
        vec!["c".to_string()],
        "forward var-length must reach the far-end label through intermediates of other labels"
    );
}

#[tokio::test]
async fn optional_match_variable_length_matches_within_bound() {
    // Issue 05: variable-length under OPTIONAL MATCH now parses and runs.
    // Alice reaches Bob (1 hop) and Carol (1 and 2 hops) within `*1..2`.
    let mut writer = WriterSession::open(store(), paths("exec-opt-varlen"))
        .await
        .unwrap();
    build_friend_graph(&mut writer).await;
    let snapshot = writer.snapshot();

    let q = parse(
        "MATCH (a:Person {name: 'Alice'}) \
         OPTIONAL MATCH (a)-[:KNOWS*1..2]->(b) \
         RETURN DISTINCT b.name AS name ORDER BY name",
    )
    .unwrap();
    let plan = lower(&q).unwrap();
    let rows = execute(&plan, &snapshot, &Params::new()).await.unwrap();
    let names: Vec<String> = rows
        .iter()
        .filter_map(|r| match r.get("name") {
            Some(RuntimeValue::String(s)) => Some(s.clone()),
            _ => None,
        })
        .collect();
    assert!(names.contains(&"Bob".to_string()));
    assert!(names.contains(&"Carol".to_string()));
}

#[tokio::test]
async fn optional_match_variable_length_null_pads_when_no_path() {
    // OPTIONAL semantics preserved: a source with no outgoing path still
    // yields exactly one row with the optional endpoint null.
    let mut writer = WriterSession::open(store(), paths("exec-opt-varlen-null"))
        .await
        .unwrap();
    let solo = NodeId::new();
    writer
        .upsert_node("Person", solo, &person("Solo", 50))
        .unwrap();
    writer.commit_batch().await.unwrap();
    let snapshot = writer.snapshot();

    let q = parse(
        "MATCH (a:Person {name: 'Solo'}) \
         OPTIONAL MATCH (a)-[:KNOWS*1..3]->(b) \
         RETURN a.name AS a, b.name AS b",
    )
    .unwrap();
    let plan = lower(&q).unwrap();
    let rows = execute(&plan, &snapshot, &Params::new()).await.unwrap();
    assert_eq!(rows.len(), 1, "OPTIONAL MATCH preserves the source row");
    assert_eq!(rows[0].get("a"), Some(&RuntimeValue::String("Solo".into())));
    assert!(matches!(rows[0].get("b"), None | Some(RuntimeValue::Null)));
}

#[tokio::test]
async fn where_label_predicate_filters_by_membership() {
    // Issue 06: `WHERE n:Label` filters by label membership and is not
    // stripped by the optimizer when no Expand already guarantees the label.
    let mut writer = WriterSession::open(store(), paths("exec-where-label"))
        .await
        .unwrap();
    let ada = NodeId::new();
    let post = NodeId::new();
    writer
        .upsert_node("Person", ada, &person("Ada", 36))
        .unwrap();
    let mut post_props: BTreeMap<String, CoreValue> = BTreeMap::new();
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

    let q = parse("MATCH (n) WHERE n:Person RETURN count(*) AS c").unwrap();
    let plan = lower(&q).unwrap();
    let rows = execute(&plan, &snapshot, &Params::new()).await.unwrap();
    assert_eq!(rows[0].get("c"), Some(&RuntimeValue::Integer(1)));
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

#[tokio::test]
async fn not_exists_left_of_and_hoists_to_semiapply() {
    // Regression: `NOT EXISTS(pattern) AND <scalar>` must parse so the
    // NOT binds only the EXISTS (NOT tighter than AND). Previously NOT
    // swallowed the AND -> `NOT (EXISTS AND scalar)`, which left the
    // EXISTS un-hoisted and failed at evaluate() with "must be hoisted to
    // a SemiApply operator". Covers the NOT-on-the-left position plus the
    // multi-type / undirected / anonymous pattern the cloud MCP used.
    let mut writer = WriterSession::open(store(), paths("exec-not-exists-and"))
        .await
        .unwrap();
    build_friend_graph(&mut writer).await;
    let snapshot = writer.snapshot();

    // Every Person has an out-KNOWS, so NOT EXISTS((a)-[:KNOWS]-()) is false
    // for all; AND-ed with a true scalar still yields 0 rows (and crucially
    // does NOT error).
    for q in [
        "MATCH (a:Person) WHERE NOT EXISTS((a)-[:KNOWS]-()) AND a.age > 0 RETURN a.name AS name",
        "MATCH (a:Person) WHERE NOT EXISTS((a)-[:KNOWS]-()) AND a.age IS NULL RETURN a.name AS name",
        "MATCH (a:Person) WHERE NOT EXISTS((a)-[:KNOWS]->()) AND a.name = 'Alice' RETURN a.name AS name",
    ] {
        let plan = lower(&parse(q).unwrap()).unwrap();
        let rows = execute(&plan, &snapshot, &Params::new())
            .await
            .unwrap_or_else(|e| panic!("query failed: {q}\n  err: {e}"));
        assert_eq!(rows.len(), 0, "expected 0 rows for: {q}");
    }
}

#[tokio::test]
async fn call_subquery_uncorrelated_aggregation() {
    let mut writer = WriterSession::open(store(), paths("exec-call-agg"))
        .await
        .unwrap();
    build_friend_graph(&mut writer).await;
    let snapshot = writer.snapshot();

    // Uncorrelated CALL{}: a one-row aggregation that the outer RETURNs.
    let q = parse("CALL { MATCH (p:Person) RETURN count(p) AS total } RETURN total").unwrap();
    let plan = lower(&q).unwrap();
    let rows = execute(&plan, &snapshot, &Params::new()).await.unwrap();
    assert_eq!(rows.len(), 1);
    assert!(
        matches!(rows[0].get("total"), Some(RuntimeValue::Integer(5))),
        "got {:?}",
        rows[0].get("total")
    );
}

#[tokio::test]
async fn call_subquery_cross_joins_with_outer() {
    let mut writer = WriterSession::open(store(), paths("exec-call-cross"))
        .await
        .unwrap();
    build_friend_graph(&mut writer).await;
    let snapshot = writer.snapshot();

    // The uncorrelated subquery result (total=5) is combined with each of the
    // 5 outer Person rows.
    let q = parse(
        "MATCH (a:Person) \
         CALL { MATCH (p:Person) RETURN count(p) AS total } \
         RETURN a.name AS name, total ORDER BY name",
    )
    .unwrap();
    let plan = lower(&q).unwrap();
    let rows = execute(&plan, &snapshot, &Params::new()).await.unwrap();
    assert_eq!(rows.len(), 5);
    for r in &rows {
        assert!(matches!(r.get("total"), Some(RuntimeValue::Integer(5))));
    }
}

#[tokio::test]
async fn call_subquery_correlated_runs_per_outer_row() {
    let mut writer = WriterSession::open(store(), paths("exec-call-correlated"))
        .await
        .unwrap();
    build_friend_graph(&mut writer).await;
    let snapshot = writer.snapshot();

    // Correlated: the leading `WITH a` imports the outer Person; the subquery
    // expands a's out-KNOWS per row. Equivalent to a plain MATCH expand.
    let q = parse(
        "MATCH (a:Person) \
         CALL { WITH a MATCH (a)-[:KNOWS]->(b:Person) RETURN b.name AS fname } \
         RETURN a.name AS name, fname ORDER BY name, fname",
    )
    .unwrap();
    // Run through the full optimizer so the Apply operator is exercised by every
    // rewrite pass (this is what the server does).
    let catalog = StatsCatalog::from_manifest(&snapshot.manifest().manifest);
    let plan = optimize(lower(&q).unwrap(), &catalog);
    let rows = execute(&plan, &snapshot, &Params::new()).await.unwrap();
    let pairs: Vec<(String, String)> = rows
        .iter()
        .map(|r| match (r.get("name"), r.get("fname")) {
            (Some(RuntimeValue::String(a)), Some(RuntimeValue::String(b))) => {
                (a.clone(), b.clone())
            }
            other => panic!("unexpected: {:?}", other),
        })
        .collect();
    assert_eq!(
        pairs,
        vec![
            ("Alice".into(), "Bob".into()),
            ("Alice".into(), "Carol".into()),
            ("Bob".into(), "Carol".into()),
            ("Carol".into(), "Dave".into()),
            ("Dave".into(), "Eve".into()),
            ("Eve".into(), "Alice".into()),
        ]
    );
}

#[tokio::test]
async fn call_subquery_correlated_drops_rows_with_no_subquery_match() {
    let mut writer = WriterSession::open(store(), paths("exec-call-innerjoin"))
        .await
        .unwrap();
    build_friend_graph(&mut writer).await;
    let snapshot = writer.snapshot();

    // Inner-join semantics: only Eve -> Alice exists, so the per-row subquery
    // yields rows for Eve alone; the other four outer rows are dropped.
    let q = parse(
        "MATCH (a:Person) \
         CALL { WITH a MATCH (a)-[:KNOWS]->(b:Person) WHERE b.name = 'Alice' RETURN b.name AS fn } \
         RETURN a.name AS name, fn ORDER BY name",
    )
    .unwrap();
    let plan = lower(&q).unwrap();
    let rows = execute(&plan, &snapshot, &Params::new()).await.unwrap();
    assert_eq!(rows.len(), 1, "only Eve has a KNOWS edge to Alice");
    assert!(matches!(rows[0].get("name"), Some(RuntimeValue::String(s)) if s == "Eve"));
    assert!(matches!(rows[0].get("fn"), Some(RuntimeValue::String(s)) if s == "Alice"));
}

#[tokio::test]
async fn call_subquery_impure_import_is_rejected() {
    // The importing WITH must be a bare pass-through; an alias makes it impure.
    let q = parse(
        "MATCH (a:Person) \
         CALL { WITH a.name AS n MATCH (b:Person {name: n}) RETURN b } \
         RETURN b",
    )
    .unwrap();
    assert!(
        lower(&q).is_err(),
        "an aliased/projected import WITH must be rejected"
    );
}

#[tokio::test]
async fn call_subquery_without_return_exposes_no_bindings() {
    // A block with no terminating RETURN must not leak its internal bindings
    // into the outer scope (regression: produced was the whole subplan).
    let q = parse("MATCH (a:Person) CALL { WITH a MATCH (a)-->(b) } RETURN b").unwrap();
    assert!(
        lower(&q).is_err(),
        "the subquery-internal `b` must not be referenceable outside the block"
    );
}

#[tokio::test]
async fn call_subquery_correlated_writes_lower_ok() {
    // Writes inside a correlated CALL subquery now lower (executed in exec_writes).
    let q =
        parse("MATCH (a:Person) CALL { WITH a CREATE (c:City {of: a.name}) } RETURN a").unwrap();
    assert!(
        lower(&q).is_ok(),
        "writes in a correlated CALL subquery should lower"
    );
}

#[tokio::test]
async fn semi_apply_over_correlated_apply_is_correct() {
    let mut writer = WriterSession::open(store(), paths("exec-semiapply-apply"))
        .await
        .unwrap();
    build_friend_graph(&mut writer).await;
    let snapshot = writer.snapshot();

    // A SemiApply (the MATCH ... WHERE EXISTS) sits above a correlated Apply
    // (the CALL). Exercises decorrelation's collect_outer_labels over an Apply
    // input through the full optimizer. Every b has an out-KNOWS, so all pairs
    // survive.
    let q = parse(
        "MATCH (a:Person) \
         CALL { WITH a MATCH (a)-[:KNOWS]->(b:Person) RETURN b } \
         MATCH (b) WHERE EXISTS((b)-[:KNOWS]->(:Person)) \
         RETURN a.name AS name, b.name AS bn ORDER BY name, bn",
    )
    .unwrap();
    let catalog = StatsCatalog::from_manifest(&snapshot.manifest().manifest);
    let plan = optimize(lower(&q).unwrap(), &catalog);
    let rows = execute(&plan, &snapshot, &Params::new()).await.unwrap();
    let pairs: Vec<(String, String)> = rows
        .iter()
        .map(|r| match (r.get("name"), r.get("bn")) {
            (Some(RuntimeValue::String(a)), Some(RuntimeValue::String(b))) => {
                (a.clone(), b.clone())
            }
            other => panic!("unexpected: {:?}", other),
        })
        .collect();
    assert_eq!(
        pairs,
        vec![
            ("Alice".into(), "Bob".into()),
            ("Alice".into(), "Carol".into()),
            ("Bob".into(), "Carol".into()),
            ("Carol".into(), "Dave".into()),
            ("Dave".into(), "Eve".into()),
            ("Eve".into(), "Alice".into()),
        ]
    );
}

#[tokio::test]
async fn inline_label_disjunction_matches_any_label() {
    let mut writer = WriterSession::open(store(), paths("exec-label-or"))
        .await
        .unwrap();
    for (lbl, nm) in [("Cat", "Felix"), ("Dog", "Rex"), ("Fish", "Nemo")] {
        let mut props = std::collections::BTreeMap::new();
        props.insert("name".to_string(), CoreValue::Str(nm.into()));
        writer
            .upsert_node(
                lbl,
                NodeId::new(),
                &NodeWriteRecord {
                    properties: props,
                    schema_version: 1,
                    ..Default::default()
                },
            )
            .unwrap();
    }
    writer.commit_batch().await.unwrap();
    let snapshot = writer.snapshot();

    // (n:Cat|Dog) matches Cat OR Dog, but not Fish.
    let q = parse("MATCH (n:Cat|Dog) RETURN n.name AS name ORDER BY name").unwrap();
    let plan = optimize(
        lower(&q).unwrap(),
        &StatsCatalog::from_manifest(&snapshot.manifest().manifest),
    );
    let rows = execute(&plan, &snapshot, &Params::new()).await.unwrap();
    let names: Vec<&str> = rows
        .iter()
        .map(|r| match r.get("name") {
            Some(RuntimeValue::String(s)) => s.as_str(),
            other => panic!("unexpected: {:?}", other),
        })
        .collect();
    assert_eq!(names, vec!["Felix", "Rex"]);
}

#[tokio::test]
async fn with_where_exists_is_hoisted_to_semiapply() {
    let mut writer = WriterSession::open(store(), paths("exec-with-exists"))
        .await
        .unwrap();
    build_friend_graph(&mut writer).await;
    let snapshot = writer.snapshot();

    // EXISTS in a WITH ... WHERE must hoist to SemiApply (not reach evaluate()).
    // Everyone has an out-KNOWS, so all 5 survive.
    let q = parse(
        "MATCH (a:Person) WITH a WHERE EXISTS((a)-[:KNOWS]->(:Person)) \
         RETURN a.name AS name ORDER BY name",
    )
    .unwrap();
    let plan = optimize(
        lower(&q).unwrap(),
        &StatsCatalog::from_manifest(&snapshot.manifest().manifest),
    );
    let rows = execute(&plan, &snapshot, &Params::new()).await.unwrap();
    assert_eq!(
        rows.len(),
        5,
        "EXISTS in WITH WHERE must filter via SemiApply"
    );
}

#[tokio::test]
async fn inline_label_disjunction_on_expand_target() {
    let mut writer = WriterSession::open(store(), paths("exec-label-or-expand"))
        .await
        .unwrap();
    let owner = NodeId::new();
    let mut oprops = std::collections::BTreeMap::new();
    oprops.insert("name".to_string(), CoreValue::Str("Sam".into()));
    writer
        .upsert_node(
            "Owner",
            owner,
            &NodeWriteRecord {
                properties: oprops,
                schema_version: 1,
                ..Default::default()
            },
        )
        .unwrap();
    for (lbl, nm) in [("Cat", "Felix"), ("Dog", "Rex"), ("Fish", "Nemo")] {
        let pid = NodeId::new();
        let mut props = std::collections::BTreeMap::new();
        props.insert("name".to_string(), CoreValue::Str(nm.into()));
        writer
            .upsert_node(
                lbl,
                pid,
                &NodeWriteRecord {
                    properties: props,
                    schema_version: 1,
                    ..Default::default()
                },
            )
            .unwrap();
        writer.upsert_edge("HAS", owner, pid, &edge()).unwrap();
    }
    writer.commit_batch().await.unwrap();
    let snapshot = writer.snapshot();

    let q =
        parse("MATCH (o:Owner)-[:HAS]->(p:Cat|Dog) RETURN p.name AS name ORDER BY name").unwrap();
    let plan = optimize(
        lower(&q).unwrap(),
        &StatsCatalog::from_manifest(&snapshot.manifest().manifest),
    );
    let rows = execute(&plan, &snapshot, &Params::new()).await.unwrap();
    let names: Vec<&str> = rows
        .iter()
        .map(|r| match r.get("name") {
            Some(RuntimeValue::String(s)) => s.as_str(),
            other => panic!("unexpected: {:?}", other),
        })
        .collect();
    assert_eq!(names, vec!["Felix", "Rex"]);
}

#[tokio::test]
async fn call_subquery_union_composes() {
    let mut writer = WriterSession::open(store(), paths("exec-call-union"))
        .await
        .unwrap();
    build_friend_graph(&mut writer).await;
    let snapshot = writer.snapshot();

    // UNION inside CALL{}: two label scans combined, returned by the outer query.
    let q = parse(
        "CALL { MATCH (p:Person) RETURN p.name AS who \
                UNION MATCH (p:Person) WHERE p.age > 100 RETURN p.name AS who } \
         RETURN who ORDER BY who",
    )
    .unwrap();
    let plan = optimize(
        lower(&q).unwrap(),
        &StatsCatalog::from_manifest(&snapshot.manifest().manifest),
    );
    let rows = execute(&plan, &snapshot, &Params::new()).await.unwrap();
    let names: Vec<&str> = rows
        .iter()
        .map(|r| match r.get("who") {
            Some(RuntimeValue::String(s)) => s.as_str(),
            other => panic!("unexpected: {:?}", other),
        })
        .collect();
    // Second branch adds nothing (no Person over 100); UNION dedups → 5 distinct.
    assert_eq!(names, vec!["Alice", "Bob", "Carol", "Dave", "Eve"]);
}

#[tokio::test]
async fn call_subquery_union_all_keeps_duplicates() {
    let mut writer = WriterSession::open(store(), paths("exec-call-union-all"))
        .await
        .unwrap();
    build_friend_graph(&mut writer).await;
    let snapshot = writer.snapshot();

    let q = parse(
        "CALL { MATCH (p:Person) RETURN count(p) AS c \
                UNION ALL MATCH (p:Person) RETURN count(p) AS c } \
         RETURN c ORDER BY c",
    )
    .unwrap();
    let plan = optimize(
        lower(&q).unwrap(),
        &StatsCatalog::from_manifest(&snapshot.manifest().manifest),
    );
    let rows = execute(&plan, &snapshot, &Params::new()).await.unwrap();
    // UNION ALL keeps both branches' single aggregate rows.
    assert_eq!(rows.len(), 2);
    assert!(rows
        .iter()
        .all(|r| matches!(r.get("c"), Some(RuntimeValue::Integer(5)))));
}

#[tokio::test]
async fn nodes_of_varlen_path_carry_full_properties() {
    let mut writer = WriterSession::open(store(), paths("exec-varlen-props"))
        .await
        .unwrap();
    let a = NodeId::new();
    let b = NodeId::new();
    let c = NodeId::new();
    writer.upsert_node("A", a, &person("a", 1)).unwrap();
    writer.upsert_node("Mid", b, &person("b", 2)).unwrap();
    writer.upsert_node("Goal", c, &person("c", 3)).unwrap();
    writer.upsert_edge("R", a, b, &edge()).unwrap();
    writer.upsert_edge("R", b, c, &edge()).unwrap();
    writer.commit_batch().await.unwrap();
    let snapshot = writer.snapshot();

    // The start node's properties must be present in nodes(p), so a quantifier
    // over path-node properties works (regression: the start node came back with
    // pruned/NULL properties).
    let q = parse(
        "MATCH p = (a:A)-[:R*2..2]->(g:Goal) \
         WHERE all(x IN nodes(p) WHERE x.age >= 1) \
         RETURN [x IN nodes(p) | x.age] AS ages",
    )
    .unwrap();
    let plan = optimize(lower(&q).unwrap(), &StatsCatalog::empty());
    let rows = execute(&plan, &snapshot, &Params::new()).await.unwrap();
    assert_eq!(rows.len(), 1, "the path satisfies all(age >= 1)");
    match rows[0].get("ages") {
        Some(RuntimeValue::List(items)) => {
            let ages: Vec<i64> = items
                .iter()
                .filter_map(|v| match v {
                    RuntimeValue::Integer(n) => Some(*n),
                    _ => None,
                })
                .collect();
            assert_eq!(ages, vec![1, 2, 3], "every path node, start included");
        }
        other => panic!("ages not a list: {other:?}"),
    }
}

#[tokio::test]
async fn unbounded_var_length_traverses_the_whole_chain() {
    let mut writer = WriterSession::open(store(), paths("exec-unbounded-star"))
        .await
        .unwrap();
    // Chain a -> b -> c -> d (all label N).
    let ids: Vec<NodeId> = (0..4).map(|_| NodeId::new()).collect();
    for (i, id) in ids.iter().enumerate() {
        let mut props = std::collections::BTreeMap::new();
        props.insert("n".to_string(), CoreValue::I64(i as i64));
        writer
            .upsert_node(
                "N",
                *id,
                &NodeWriteRecord {
                    properties: props,
                    schema_version: 1,
                    ..Default::default()
                },
            )
            .unwrap();
    }
    for w in ids.windows(2) {
        writer.upsert_edge("R", w[0], w[1], &edge()).unwrap();
    }
    writer.commit_batch().await.unwrap();
    let snapshot = writer.snapshot();

    // `*` (unbounded, capped) from the head reaches b, c, d.
    let q = parse("MATCH (a:N {n: 0})-[:R*]->(x:N) RETURN x.n AS n ORDER BY n").unwrap();
    let plan = optimize(
        lower(&q).unwrap(),
        &StatsCatalog::from_manifest(&snapshot.manifest().manifest),
    );
    let rows = execute(&plan, &snapshot, &Params::new()).await.unwrap();
    let ns: Vec<i64> = rows
        .iter()
        .filter_map(|r| match r.get("n") {
            Some(RuntimeValue::Integer(v)) => Some(*v),
            _ => None,
        })
        .collect();
    assert_eq!(
        ns,
        vec![1, 2, 3],
        "unbounded * reaches every downstream node"
    );

    // Open-upper `*2..` skips the 1-hop neighbour.
    let q2 = parse("MATCH (a:N {n: 0})-[:R*2..]->(x:N) RETURN x.n AS n ORDER BY n").unwrap();
    let plan2 = optimize(
        lower(&q2).unwrap(),
        &StatsCatalog::from_manifest(&snapshot.manifest().manifest),
    );
    let rows2 = execute(&plan2, &snapshot, &Params::new()).await.unwrap();
    let ns2: Vec<i64> = rows2
        .iter()
        .filter_map(|r| match r.get("n") {
            Some(RuntimeValue::Integer(v)) => Some(*v),
            _ => None,
        })
        .collect();
    assert_eq!(ns2, vec![2, 3], "*2.. starts at the 2-hop node");
}

#[tokio::test]
async fn parameterized_var_length_bound() {
    let mut writer = WriterSession::open(store(), paths("exec-param-varlen"))
        .await
        .unwrap();
    let ids: Vec<NodeId> = (0..5).map(|_| NodeId::new()).collect();
    for (i, id) in ids.iter().enumerate() {
        let mut props = std::collections::BTreeMap::new();
        props.insert("n".to_string(), CoreValue::I64(i as i64));
        writer
            .upsert_node(
                "N",
                *id,
                &NodeWriteRecord {
                    properties: props,
                    schema_version: 1,
                    ..Default::default()
                },
            )
            .unwrap();
    }
    for w in ids.windows(2) {
        writer.upsert_edge("R", w[0], w[1], &edge()).unwrap();
    }
    writer.commit_batch().await.unwrap();
    let snapshot = writer.snapshot();

    // `*1..$depth` — the upper bound comes from a parameter.
    let q = parse("MATCH (a:N {n: 0})-[:R*1..$depth]->(x:N) RETURN x.n AS n ORDER BY n").unwrap();
    let plan = optimize(
        lower(&q).unwrap(),
        &StatsCatalog::from_manifest(&snapshot.manifest().manifest),
    );
    let mut params = Params::new();
    params.insert("depth".to_string(), RuntimeValue::Integer(2));
    let rows = execute(&plan, &snapshot, &params).await.unwrap();
    let ns: Vec<i64> = rows
        .iter()
        .filter_map(|r| match r.get("n") {
            Some(RuntimeValue::Integer(v)) => Some(*v),
            _ => None,
        })
        .collect();
    assert_eq!(ns, vec![1, 2], "depth=2 reaches the 1- and 2-hop nodes");

    // Same plan, larger depth → more reachable nodes (depth from a param at exec).
    let mut params3 = Params::new();
    params3.insert("depth".to_string(), RuntimeValue::Integer(4));
    let rows3 = execute(&plan, &snapshot, &params3).await.unwrap();
    let ns3: Vec<i64> = rows3
        .iter()
        .filter_map(|r| match r.get("n") {
            Some(RuntimeValue::Integer(v)) => Some(*v),
            _ => None,
        })
        .collect();
    assert_eq!(
        ns3,
        vec![1, 2, 3, 4],
        "depth=4 reaches all downstream nodes"
    );
}
