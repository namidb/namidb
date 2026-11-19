//! The graph-explorer neighbourhood statement: one correlated
//! `CALL { WITH n OPTIONAL MATCH (n)-[r]->(m) RETURN ... LIMIT $per }` per
//! seed, over every relationship type, in both directions.
//!
//! Correlated subqueries over several input rows run concurrently. These
//! tests pin what that must not change: exactly one output row per seed
//! (the grouping `collect` decides the row order, so rows are matched by
//! seed); at most `$per` neighbours per seed, each a real edge; the full
//! adjacency when `$per` is large; an empty list for a seed without
//! neighbours; and the same answer on every run.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use namidb_core::id::{NamespaceId, NodeId};
use namidb_core::schema::{DataType, EdgeTypeDef, LabelDef, PropertyDef, SchemaBuilder};
use namidb_core::value::Value as CoreValue;
use namidb_storage::{EdgeWriteRecord, NamespacePaths, NodeWriteRecord, WriterSession};
use object_store::memory::InMemory;
use object_store::ObjectStore;

use namidb_query::{execute, lower, parse, Params, RuntimeValue};

const TYPES: [&str; 3] = ["KNOWS", "LIKES", "WORKS_WITH"];
const PEOPLE: usize = 60;

fn store() -> Arc<dyn ObjectStore> {
    Arc::new(InMemory::new())
}

fn paths(name: &str) -> NamespacePaths {
    NamespacePaths::new("tenants", NamespaceId::new(name).unwrap())
}

fn person(i: usize) -> NodeWriteRecord {
    let mut props: BTreeMap<String, CoreValue> = BTreeMap::new();
    props.insert("name".into(), CoreValue::Str(format!("p{i:02}")));
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

/// Directed edges `(type, src index, dst index)`: person 0 is a hub with an
/// edge of every type to everyone, persons 1..50 have a few edges each over
/// all types, and persons 50..60 have none (the OPTIONAL padding case).
fn edge_list() -> Vec<(&'static str, usize, usize)> {
    let mut edges = Vec::new();
    for dst in 1..PEOPLE {
        for t in TYPES {
            edges.push((t, 0, dst));
        }
    }
    for src in 1..50 {
        for k in 0..(src % 5) {
            let dst = 1 + (src * 7 + k * 11) % 49;
            if dst != src {
                edges.push((TYPES[(src + k) % TYPES.len()], src, dst));
            }
        }
    }
    edges.sort();
    edges.dedup();
    edges
}

/// A namespace holding the graph above, flushed to SSTs (the object-store
/// read path), and each person's elementId by index.
async fn graph(name: &str) -> (WriterSession, Vec<String>) {
    let mut schema = SchemaBuilder::new()
        .label(LabelDef {
            name: "Person".into(),
            properties: vec![PropertyDef::new("name", DataType::Utf8, false).unwrap()],
        })
        .unwrap();
    for t in TYPES {
        schema = schema
            .edge_type(EdgeTypeDef {
                name: t.into(),
                src_label: "Person".into(),
                dst_label: "Person".into(),
                properties: vec![],
            })
            .unwrap();
    }
    let schema = schema.build();
    let mut writer = WriterSession::open(store(), paths(name)).await.unwrap();
    let ids: Vec<NodeId> = (0..PEOPLE).map(|_| NodeId::new()).collect();
    for (i, id) in ids.iter().enumerate() {
        writer.upsert_node("Person", *id, &person(i)).unwrap();
    }
    for (t, src, dst) in edge_list() {
        writer.upsert_edge(t, ids[src], ids[dst], &edge()).unwrap();
    }
    writer.commit_batch().await.unwrap();
    writer.flush(schema).await.unwrap();

    let snapshot = writer.snapshot();
    let plan = lower(&parse("MATCH (n:Person) RETURN n.name AS name, elementId(n) AS e").unwrap())
        .unwrap();
    let rows = execute(&plan, &snapshot, &Params::new()).await.unwrap();
    let mut element_ids = vec![String::new(); PEOPLE];
    for row in rows {
        let (Some(RuntimeValue::String(name)), Some(RuntimeValue::String(e))) =
            (row.get("name"), row.get("e"))
        else {
            panic!("unexpected row {row:?}");
        };
        let index: usize = name[1..].parse().unwrap();
        element_ids[index] = e.clone();
    }
    drop(snapshot);
    (writer, element_ids)
}

/// The console's neighbourhood statement for both directions.
const NEIGHBOURS: &str = "UNWIND $ids AS id MATCH (n) WHERE elementId(n) = id \
     CALL { WITH n OPTIONAL MATCH (n)-[r]->(m) RETURN type(r) AS t, elementId(m) AS b LIMIT $per } \
     WITH n, id, collect([t, b]) AS o \
     CALL { WITH n OPTIONAL MATCH (n)<-[r]-(m) RETURN type(r) AS t, elementId(m) AS b LIMIT $per } \
     RETURN id, o, collect([t, b]) AS i";

type Pairs = Vec<(String, String)>;

/// `(seed elementId, out pairs, in pairs)` per output row, in output order.
async fn neighbours(
    writer: &WriterSession,
    seeds: &[String],
    per: i64,
) -> Vec<(String, Pairs, Pairs)> {
    let mut params = Params::new();
    params.insert(
        "ids".into(),
        RuntimeValue::List(seeds.iter().cloned().map(RuntimeValue::String).collect()),
    );
    params.insert("per".into(), RuntimeValue::Integer(per));
    let snapshot = writer.snapshot();
    let plan = lower(&parse(NEIGHBOURS).unwrap()).unwrap();
    let rows = execute(&plan, &snapshot, &params).await.unwrap();
    let pairs = |value: Option<&RuntimeValue>| -> Pairs {
        let Some(RuntimeValue::List(items)) = value else {
            panic!("expected a list, got {value:?}");
        };
        items
            .iter()
            .filter_map(|item| match item {
                RuntimeValue::List(pair) => match (&pair[0], &pair[1]) {
                    (RuntimeValue::String(t), RuntimeValue::String(b)) => {
                        Some((t.clone(), b.clone()))
                    }
                    (RuntimeValue::Null, RuntimeValue::Null) => None,
                    other => panic!("unexpected pair {other:?}"),
                },
                other => panic!("unexpected item {other:?}"),
            })
            .collect()
    };
    rows.iter()
        .map(|row| {
            let Some(RuntimeValue::String(id)) = row.get("id") else {
                panic!("unexpected row {row:?}");
            };
            (id.clone(), pairs(row.get("o")), pairs(row.get("i")))
        })
        .collect()
}

/// Each seed's `(type, partner elementId)` set.
type Adjacency = Vec<BTreeSet<(String, String)>>;

/// Every seed's real out/in adjacency.
fn adjacency(element_ids: &[String]) -> (Adjacency, Adjacency) {
    let mut out = vec![BTreeSet::new(); PEOPLE];
    let mut inc = vec![BTreeSet::new(); PEOPLE];
    for (t, src, dst) in edge_list() {
        out[src].insert((t.to_string(), element_ids[dst].clone()));
        inc[dst].insert((t.to_string(), element_ids[src].clone()));
    }
    (out, inc)
}

/// The rows keyed by seed index, checking every seed has exactly one row.
fn by_seed<'a>(
    rows: &'a [(String, Pairs, Pairs)],
    element_ids: &[String],
) -> Vec<(usize, &'a Pairs, &'a Pairs)> {
    let mut seen = BTreeSet::new();
    let found: Vec<(usize, &Pairs, &Pairs)> = rows
        .iter()
        .map(|(id, o, i)| {
            let index = element_ids
                .iter()
                .position(|e| e == id)
                .unwrap_or_else(|| panic!("row for an unknown seed {id}"));
            assert!(seen.insert(index), "seed {index} has more than one row");
            (index, o, i)
        })
        .collect();
    assert_eq!(seen.len(), element_ids.len(), "every seed has a row");
    found
}

/// Seeds in a scrambled order with the hub, ordinary nodes and isolated
/// nodes interleaved, so concurrent runs finish out of input order.
fn seed_order() -> Vec<usize> {
    let mut order: Vec<usize> = (0..PEOPLE).map(|i| (i * 37 + 11) % PEOPLE).collect();
    order.dedup();
    assert_eq!(order.iter().collect::<BTreeSet<_>>().len(), PEOPLE);
    order
}

#[tokio::test]
async fn limited_neighbourhoods_cover_every_seed_with_real_edges() {
    let (writer, element_ids) = graph("explore-limited").await;
    let (out, inc) = adjacency(&element_ids);
    let order = seed_order();
    let seeds: Vec<String> = order.iter().map(|&i| element_ids[i].clone()).collect();

    let rows = neighbours(&writer, &seeds, 4).await;
    assert_eq!(rows.len(), seeds.len(), "one row per seed");
    for (index, o, i) in by_seed(&rows, &element_ids) {
        assert_eq!(o.len(), out[index].len().min(4), "seed {index} out");
        assert_eq!(i.len(), inc[index].len().min(4), "seed {index} in");
        for pair in o {
            assert!(
                out[index].contains(pair),
                "seed {index}: {pair:?} is not an out edge"
            );
        }
        for pair in i {
            assert!(
                inc[index].contains(pair),
                "seed {index}: {pair:?} is not an in edge"
            );
        }
    }

    // The same statement answers the same rows every time.
    for _ in 0..3 {
        assert_eq!(neighbours(&writer, &seeds, 4).await, rows);
    }
}

#[tokio::test]
async fn unlimited_neighbourhoods_are_the_whole_adjacency() {
    let (writer, element_ids) = graph("explore-unlimited").await;
    let (out, inc) = adjacency(&element_ids);
    let order = seed_order();
    let seeds: Vec<String> = order.iter().map(|&i| element_ids[i].clone()).collect();

    let rows = neighbours(&writer, &seeds, 10_000).await;
    assert_eq!(rows.len(), seeds.len());
    for (index, o, i) in by_seed(&rows, &element_ids) {
        assert_eq!(
            o.iter().cloned().collect::<BTreeSet<_>>(),
            out[index],
            "seed {index} out"
        );
        assert_eq!(
            o.len(),
            out[index].len(),
            "seed {index}: no duplicate out edge"
        );
        assert_eq!(
            i.iter().cloned().collect::<BTreeSet<_>>(),
            inc[index],
            "seed {index} in"
        );
        assert_eq!(
            i.len(),
            inc[index].len(),
            "seed {index}: no duplicate in edge"
        );
    }
}
