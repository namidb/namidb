//! Differential tests for the whole-graph aggregation, anchor-selection and
//! shortest-path work: every optimised plan must return EXACTLY what the
//! naive plan (lowered, never optimised) returns, over randomised graphs.
//!
//! Each seed builds a graph with three labels (some nodes carry two), three
//! relationship types with self-loops, hubs and isolated nodes, spread over
//! flushed SSTs (one or two per type), unflushed memtable edges, deleted
//! edges and, for some seeds, deleted nodes whose edges dangle. Every query
//! runs naive and optimised (under both an empty and a manifest-derived
//! statistics catalog) and the rows must match: in order when the query
//! fixes an order (aggregates group in a deterministic order, ORDER BY on a
//! total key), as a multiset otherwise.

use std::collections::BTreeMap;
use std::sync::Arc;

use namidb_core::id::{NamespaceId, NodeId};
use namidb_core::schema::{DataType, LabelDef, PropertyDef, Schema, SchemaBuilder};
use namidb_core::value::Value as CoreValue;
use namidb_storage::{EdgeWriteRecord, NamespacePaths, NodeWriteRecord, WriterSession};
use object_store::memory::InMemory;
use object_store::ObjectStore;

use namidb_query::{
    execute, lower, optimize, parse, LogicalPlan, Params, Row, RuntimeValue, StatsCatalog,
};

/// xorshift64*: tiny, deterministic, good enough to shape test graphs.
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n.max(1)
    }
    fn chance(&mut self, pct: u64) -> bool {
        self.below(100) < pct
    }
}

const NAMES: [&str; 6] = [
    "ley",
    "código",
    "decreto",
    "jubilación",
    "acción",
    "reglamento",
];

fn schema() -> Schema {
    let props = || {
        vec![
            PropertyDef::new("k", DataType::Int64, true).unwrap(),
            PropertyDef::new("name", DataType::Utf8, true).unwrap(),
        ]
    };
    SchemaBuilder::new()
        .label(LabelDef {
            name: "A".into(),
            properties: props(),
        })
        .unwrap()
        .label(LabelDef {
            name: "B".into(),
            properties: props(),
        })
        .unwrap()
        .label(LabelDef {
            name: "C".into(),
            properties: props(),
        })
        .unwrap()
        .build()
}

fn node(k: i64, name: &str) -> NodeWriteRecord {
    let mut props: BTreeMap<String, CoreValue> = BTreeMap::new();
    props.insert("k".into(), CoreValue::I64(k));
    props.insert("name".into(), CoreValue::Str(name.into()));
    NodeWriteRecord {
        properties: props,
        schema_version: 0,
        labels: vec![],
    }
}

pub struct Graph {
    pub writer: WriterSession,
    pub ids: Vec<NodeId>,
    pub hub: NodeId,
}

/// Build one randomised graph. `seed` decides everything.
pub async fn graph(seed: u64) -> Graph {
    let mut rng = Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1);
    let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
    let paths = NamespacePaths::new("tenants", NamespaceId::new(format!("diff-{seed}")).unwrap());
    let mut w = WriterSession::open(store, paths).await.unwrap();
    let n = 24 + rng.below(40) as usize;
    let mut ids = Vec::with_capacity(n);
    for i in 0..n {
        let id = NodeId::new();
        let label_sets: [&[&str]; 6] = [&["A"], &["B"], &["C"], &["A"], &["A", "B"], &["B", "C"]];
        let labels = label_sets[rng.below(label_sets.len() as u64) as usize];
        let rec = node(rng.below(5) as i64, NAMES[i % NAMES.len()]);
        w.upsert_node_with_labels(labels.iter().map(|l| l.to_string()), id, &rec)
            .unwrap();
        ids.push(id);
    }
    // A hub that collects a large share of R edges (power-law flavour).
    let hub = ids[rng.below(n as u64) as usize];
    let mut edges: Vec<(&str, NodeId, NodeId)> = Vec::new();
    for t in ["R", "S", "T"] {
        let m = n + rng.below(2 * n as u64) as usize;
        for _ in 0..m {
            let s = ids[rng.below(n as u64) as usize];
            let d = if t == "R" && rng.chance(30) {
                hub
            } else if rng.chance(5) {
                s // self-loop
            } else {
                ids[rng.below(n as u64) as usize]
            };
            edges.push((t, s, d));
        }
    }
    let split = edges.len() * (40 + rng.below(40) as usize) / 100;
    for (t, s, d) in &edges[..split] {
        w.upsert_edge(*t, *s, *d, &EdgeWriteRecord::default())
            .unwrap();
    }
    w.commit_batch().await.unwrap();
    w.flush(schema()).await.unwrap();
    if rng.chance(50) {
        // A second SST per type: exercises the multi-SST merge.
        let extra = split + (edges.len() - split) / 2;
        for (t, s, d) in &edges[split..extra] {
            w.upsert_edge(*t, *s, *d, &EdgeWriteRecord::default())
                .unwrap();
        }
        // Delete a few flushed edges so a tombstone lands in the newer SST.
        for (t, s, d) in edges[..split].iter().step_by(7) {
            w.tombstone_edge(*t, *s, *d).unwrap();
        }
        w.commit_batch().await.unwrap();
        w.flush(schema()).await.unwrap();
        for (t, s, d) in &edges[extra..] {
            w.upsert_edge(*t, *s, *d, &EdgeWriteRecord::default())
                .unwrap();
        }
    } else {
        for (t, s, d) in &edges[split..] {
            w.upsert_edge(*t, *s, *d, &EdgeWriteRecord::default())
                .unwrap();
        }
    }
    // Unflushed deletes of both flushed and unflushed edges.
    for (t, s, d) in edges.iter().skip(3).step_by(11) {
        w.tombstone_edge(*t, *s, *d).unwrap();
    }
    if rng.chance(30) {
        // A deleted node whose edges dangle: every plan must drop them.
        let victim = ids[rng.below(n as u64) as usize];
        if victim != hub {
            w.tombstone_node("A", victim).unwrap();
        }
    }
    w.commit_batch().await.unwrap();
    Graph {
        writer: w,
        ids,
        hub,
    }
}

fn render(v: &RuntimeValue) -> String {
    match v {
        RuntimeValue::Node(n) => {
            let props: Vec<String> = n
                .properties
                .iter()
                .map(|(k, v)| format!("{k}:{}", render(v)))
                .collect();
            format!(
                "N({:?};{:?};{})",
                n.id,
                n.labels.iter().collect::<Vec<_>>(),
                props.join(",")
            )
        }
        RuntimeValue::List(xs) => {
            format!("[{}]", xs.iter().map(render).collect::<Vec<_>>().join(","))
        }
        RuntimeValue::Float(f) => format!("{f:?}"),
        other => format!("{other:?}"),
    }
}

fn rows_text(rows: &[Row]) -> Vec<String> {
    rows.iter()
        .map(|r| {
            r.bindings
                .iter()
                .filter(|(k, _)| !k.starts_with("__"))
                .map(|(k, v)| format!("{k}={}", render(v)))
                .collect::<Vec<_>>()
                .join(",")
        })
        .collect()
}

fn contains_op(plan: &LogicalPlan, name: &str) -> bool {
    plan.operator_name() == name || plan.children().iter().any(|c| contains_op(c, name))
}

#[derive(Clone, Copy, PartialEq)]
pub enum Order {
    /// Rows must match in order.
    Exact,
    /// Rows must match as a multiset.
    Any,
}

/// Run `q` naive and optimised; assert equal. Returns the optimised plans
/// (empty catalog, manifest catalog) for shape assertions.
pub async fn check(
    g: &Graph,
    seed: u64,
    q: &str,
    params: &Params,
    order: Order,
) -> (LogicalPlan, LogicalPlan) {
    let snapshot = g.writer.snapshot();
    let parsed = parse(q).unwrap_or_else(|e| panic!("parse `{q}`: {e:?}"));
    let naive_plan = lower(&parsed).unwrap_or_else(|e| panic!("lower `{q}`: {e:?}"));
    let naive = execute(&naive_plan, &snapshot, params)
        .await
        .unwrap_or_else(|e| panic!("seed {seed}: naive `{q}`: {e:?}"));
    let mut expected = rows_text(&naive);
    if order == Order::Any {
        expected.sort();
    }
    let manifest_catalog = StatsCatalog::from_manifest(&snapshot.manifest().manifest);
    let mut plans = Vec::new();
    for (cat_name, catalog) in [
        ("empty", StatsCatalog::empty()),
        ("manifest", manifest_catalog),
    ] {
        let plan = optimize(naive_plan.clone(), &catalog);
        // A PathCount proves its endpoints by probing ids or by scanning the
        // label; both must be exact. Each test file runs in its own process,
        // so the switch reaches only this file's plans.
        for proof in ["never", "always"] {
            std::env::set_var("NAMIDB_PATH_COUNT_SCAN_PROOF", proof);
            let rows = execute(&plan, &snapshot, params).await;
            std::env::remove_var("NAMIDB_PATH_COUNT_SCAN_PROOF");
            let rows = rows.unwrap_or_else(|e| {
                panic!("seed {seed}: optimised ({cat_name}, {proof}) `{q}`: {e:?}")
            });
            let mut got = rows_text(&rows);
            if order == Order::Any {
                got.sort();
            }
            assert_eq!(
                got,
                expected,
                "seed {seed}, catalog {cat_name}, scan proof {proof}: optimised plan disagrees with naive for\n  {q}\nplan:\n{}",
                namidb_query::explain(&plan)
            );
        }
        plans.push(plan);
    }
    let manifest = plans.pop().unwrap();
    (plans.pop().unwrap(), manifest)
}

const SEEDS: u64 = 40;

/// Grouped and global counts over chains: the `PathCount` rewrite.
#[tokio::test]
async fn grouped_counts_match_naive() {
    // (query, must the empty-catalog plan use PathCount?)
    let cases: &[(&str, bool)] = &[
        ("MATCH (a:A)-[:R]->(b) RETURN a.k AS k, count(b) AS c", true),
        ("MATCH (a:A)<-[:R]-(b) RETURN a.k AS k, count(*) AS c", true),
        ("MATCH (a:A)-[:R]-(b) RETURN a.k AS k, count(b) AS c", true),
        ("MATCH (a:A)-[:R]-(b) RETURN a, count(b) AS c", true),
        ("MATCH (a)-[:R]->(b:B) RETURN b.k AS k, count(a) AS c", true),
        ("MATCH (a)-[:R]->(b:B) RETURN b, count(a) AS c", true),
        ("MATCH (a:A)-[r:R|S]->(b:B) RETURN a, count(r) AS c", true),
        ("MATCH (a:A)-[:R]->(b)-[:S]->(c:C) RETURN a.k AS k, count(c) AS c", true),
        ("MATCH (a:A)-[:R]->(b)-[:S]->(c:C) RETURN b.k AS k, count(*) AS c", true),
        ("MATCH (a:A)-[:R]->(b)-[:S]->(c:C) RETURN b, count(*) AS c", true),
        ("MATCH (a:A)-[:R]->(b)-[:S]->(c:C)-[:T]-(d) RETURN d.k AS k, count(*) AS c", true),
        ("MATCH (a:A)-[:R]->(b)-[:S]->(c:C)-[:T]-(d) RETURN d, count(*) AS c", true),
        ("MATCH (a:A)-[:R]->(b)-[:S]->(c) RETURN count(*) AS c", true),
        ("MATCH (a)-[:R]->(b)<-[:S]-(c)-[:T]->(d:B) RETURN count(d) AS c, count(*) AS c2", true),
        ("MATCH (a:A) WHERE a.k > 2 MATCH (a)-[:R]->(b) RETURN a.k AS k, count(b) AS c", false),
        ("MATCH (a:A) WHERE a.k > 2 MATCH (a)-[:R]->(b)-[:S]->(c) RETURN c.name AS n, count(*) AS c", false),
        ("MATCH (a)-[:R]->(b) RETURN labels(b) AS l, count(*) AS c", true),
        ("MATCH (a:A)-[:R]->(b) RETURN a.name AS n, count(b) AS c ORDER BY c DESC, n LIMIT 3", true),
        ("MATCH (a)<-[]-(b) RETURN a.k AS k, count(*) AS c", true),
        ("MATCH (a:B)-[]-(b) RETURN a.name AS k, count(*) AS c", true),
        ("MATCH (x)-[:R]->(d:A) RETURN d.name AS t, count(*) AS c ORDER BY c DESC, t LIMIT 5", true),
        ("MATCH (a:A)-[:R]->(b:B)<-[:S]-(c) RETURN b.k AS k, count(c) AS c", true),
        ("MATCH (a:A)-[:R]->(b:B)<-[:S]-(c) WITH b, count(c) AS c RETURN b.name AS n, c ORDER BY c DESC, n", true),
        ("MATCH (a:C)-[:T]->(b:C) RETURN a.k + 1 AS k, count(*) AS c", true),
        ("MATCH (a:A:B)-[:T]->(b) RETURN a.k AS k, count(*) AS c", false),
        // Shapes that must NOT rewrite (and still agree).
        ("MATCH (a:A)-[:R]->(b)-[:R]->(c) RETURN a.k AS k, count(*) AS c", false),
        ("MATCH (a:A)-[:R]->(b) RETURN a.k AS k, count(DISTINCT b) AS c", false),
        ("MATCH (a:A)-[:R]->(b) RETURN a.k AS k, b.k AS bk, count(*) AS c", false),
        ("MATCH (a:A)-[:R]->(b) RETURN a.k AS k, count(b.k) AS c", false),
        ("MATCH (a:A)-[:R]->(b) RETURN a.k AS k, sum(b.k) AS s", false),
        ("MATCH (a:A)-[:R*1..2]->(b) RETURN a.k AS k, count(*) AS c", false),
        ("MATCH (a:A) OPTIONAL MATCH (a)-[:R]->(b) RETURN a.k AS k, count(b) AS c", false),
    ];
    for seed in 0..SEEDS {
        let g = graph(seed).await;
        for (q, must) in cases {
            let order = if q.contains("ORDER BY") || q.contains("count(") {
                Order::Exact
            } else {
                Order::Any
            };
            let (empty_plan, _) = check(&g, seed, q, &Params::new(), order).await;
            assert_eq!(
                contains_op(&empty_plan, "PathCount"),
                *must,
                "seed {seed}: PathCount expectation for\n  {q}\nplan:\n{}",
                namidb_query::explain(&empty_plan)
            );
        }
        let _ = (&g.ids, g.hub);
    }
}

/// Anchor selection: re-anchored plans (index pin, non-indexed pin, starred
/// hop toward a pinned node) return the naive rows.
#[tokio::test]
async fn anchored_patterns_match_naive() {
    let cases: &[&str] = &[
        "MATCH (x)-[:R]->(d:A {k: 2}) RETURN x.name AS x, d.name AS d",
        "MATCH (d:A {k: 2})<-[:R]-(x) RETURN x.name AS x, d.name AS d",
        "MATCH (x)-[:R]->(d:A) WHERE d.name = 'ley' RETURN x, d",
        "MATCH (x)-[r:R]->(d:A) WHERE d.name = 'ley' AND x.k >= 1 RETURN x.k AS xk, r, d.k AS dk",
        "MATCH (x:B)-[:R]->(d:A) WHERE d.k = 1 AND x.k > 0 RETURN x.k AS xk, d.k AS dk",
        "MATCH (x)-[:R]-(d:C) WHERE d.k < 2 RETURN x.name AS x, d.name AS d",
        "MATCH (x)-[:R*1..3]->(d:A) WHERE d.name = 'código' RETURN x.name AS x, d.name AS d",
        "MATCH (x)-[:R*0..2]->(d:A {k: 3}) RETURN x, d",
        "MATCH (x)-[:R|S*1..2]-(d:B) WHERE d.k = 4 RETURN x.name AS x, d.name AS d",
        "MATCH (x)-[:R]->(y)-[:S]->(d:C) WHERE d.k = 0 RETURN x.name AS x, y.name AS y",
        "MATCH (x)-[:R]->(d) WHERE elementId(d) = $hub RETURN x.name AS x",
        "MATCH (x)-[:R*1..2]->(d) WHERE elementId(d) = $hub RETURN count(DISTINCT x) AS c",
        "MATCH (d) WHERE elementId(d) = $hub MATCH (d)<-[:R*1..2]-(x) RETURN count(DISTINCT x) AS c",
        // Zero-hop rows bind the far alias to the near node's own value, so
        // both aliases must decode the union of their read properties.
        "MATCH (x)-[:R*0..1]-(d:B) WHERE d.k = 4 RETURN x.name AS x",
        "MATCH (d:B) WHERE d.k = 4 MATCH (d)-[:R*0..1]-(x) RETURN x.name AS x",
        "MATCH (x)-[:R*0..2]->(d:A) WHERE d.k = 3 RETURN x.name AS x",
        "MATCH (x:B)-[:R*0..2]->(d:A) WHERE x.k > 0 RETURN d.name AS d",
        "MATCH (x)-[:R*0..1]->(y)-[:S*0..1]->(d:C) WHERE d.k = 0 RETURN x.name AS x",
        "MATCH (x)-[:R*$lo..1]-(d:B) WHERE d.k = 4 RETURN x.name AS x",
        // Further anchored shapes: exact length, left starred, path-bound
        // starred (never inverted), filtered far side, OR / IN / param pins,
        // DISTINCT and OPTIONAL.
        "MATCH (x)-[:R*2..2]->(d:A {k: 1}) RETURN x.name AS x, d.name AS d",
        "MATCH (x)<-[:R*1..3]-(d:A {k: 2}) RETURN x.name AS x",
        "MATCH p=(x)-[:R*1..2]->(d:A {k: 2}) RETURN length(p) AS l, [n IN nodes(p) | n.name] AS ns",
        "MATCH (x)-[:R*1..2]->(d:A {k: 2}) WHERE x.k > 1 RETURN x.name AS x",
        "MATCH (y:C)-[:S]->(x)-[:R*1..2]->(d:A {k: 2}) RETURN y.name AS y, x.name AS x",
        "MATCH (x)-[:R]->(d:A) WHERE d.name = 'ley' OR d.k = 1 RETURN x.name AS x, d.name AS d",
        "MATCH (x)-[:R]->(d:A) WHERE d.name = $n RETURN x.name AS x",
        "MATCH (x)-[:R]->(d:A) WHERE d.k = x.k RETURN x.name AS x",
        "MATCH (x)-[:R]->(d:A) WHERE d.k IN [1, 2] RETURN x.name AS x, d.k AS dk",
        "MATCH (x)-[:R]->(d:A) WHERE d.k = 2 RETURN DISTINCT x.k AS x",
        "OPTIONAL MATCH (x)-[:R]->(d:A {k: 9}) RETURN x, d",
    ];
    for seed in 0..SEEDS {
        let g = graph(seed).await;
        let mut params = Params::new();
        params.insert("hub".into(), RuntimeValue::String(g.hub.to_string()));
        params.insert("lo".into(), RuntimeValue::Integer(0));
        params.insert("n".into(), RuntimeValue::String("decreto".into()));
        for q in cases {
            check(&g, seed, q, &params, Order::Any).await;
        }
    }
}

/// Every live edge as `(type, src, dst)`, read through the naive plan.
async fn live_edges(g: &Graph) -> Vec<(String, String, String)> {
    let snapshot = g.writer.snapshot();
    let plan = lower(
        &parse("MATCH (s)-[r]->(t) RETURN type(r) AS t, elementId(s) AS s, elementId(t) AS d")
            .unwrap(),
    )
    .unwrap();
    execute(&plan, &snapshot, &Params::new())
        .await
        .unwrap()
        .into_iter()
        .map(|r| {
            let get = |k: &str| match r.get(k) {
                Some(RuntimeValue::String(s)) => s.clone(),
                other => panic!("{k}: {other:?}"),
            };
            (get("t"), get("s"), get("d"))
        })
        .collect()
}

/// Oracle BFS distance over `edges` restricted to `types` / `dir`.
fn oracle_distance(
    edges: &[(String, String, String)],
    types: Option<&[&str]>,
    dir: &str,
    from: &str,
    to: &str,
    max: usize,
) -> Option<usize> {
    use std::collections::{HashMap, HashSet, VecDeque};
    let mut adj: HashMap<&str, Vec<&str>> = HashMap::new();
    for (t, s, d) in edges {
        if types.is_some_and(|ts| !ts.contains(&t.as_str())) {
            continue;
        }
        if dir == "->" || dir == "-" {
            adj.entry(s).or_default().push(d);
        }
        if dir == "<-" || dir == "-" {
            adj.entry(d).or_default().push(s);
        }
    }
    let mut seen: HashSet<&str> = HashSet::from([from]);
    let mut q = VecDeque::from([(from, 0usize)]);
    while let Some((u, du)) = q.pop_front() {
        if du >= max {
            continue;
        }
        for v in adj.get(u).into_iter().flatten() {
            if *v == to {
                return Some(du + 1);
            }
            if seen.insert(v) {
                q.push_back((v, du + 1));
            }
        }
    }
    None
}

/// shortestPath (bidirectional, degree-aware, concurrent level reads) finds
/// a path of exactly the oracle's length, whose relationships really chain
/// from source to target under the pattern's types and direction.
#[tokio::test]
async fn shortest_paths_match_oracle() {
    let variants: &[(&str, Option<&[&str]>, &str)] = &[
        ("-[*..6]-", None, "-"),
        ("-[:R*..6]->", Some(&["R"]), "->"),
        ("<-[:R|S*..6]-", Some(&["R", "S"]), "<-"),
        ("-[:T*..3]-", Some(&["T"]), "-"),
    ];
    for seed in 0..SEEDS {
        let g = graph(seed).await;
        let edges = live_edges(&g).await;
        let live: std::collections::HashSet<String> = edges
            .iter()
            .flat_map(|(_, s, d)| [s.clone(), d.clone()])
            .collect();
        let mut rng = Rng(seed + 77);
        for _ in 0..6 {
            let a = g.ids[rng.below(g.ids.len() as u64) as usize].to_string();
            let b = if rng.chance(30) {
                g.hub.to_string()
            } else {
                g.ids[rng.below(g.ids.len() as u64) as usize].to_string()
            };
            if a == b || !live.contains(&a) || !live.contains(&b) {
                continue;
            }
            for (pat, types, dir) in variants {
                let max: usize = if pat.contains("..3") { 3 } else { 6 };
                let q = format!(
                    "MATCH (a), (b) WHERE elementId(a) = $a AND elementId(b) = $b \
                     MATCH p = shortestPath((a){pat}(b)) \
                     RETURN length(p) AS l, [n IN nodes(p) | elementId(n)] AS ns, relationships(p) AS rs"
                );
                let mut params = Params::new();
                params.insert("a".into(), RuntimeValue::String(a.clone()));
                params.insert("b".into(), RuntimeValue::String(b.clone()));
                let snapshot = g.writer.snapshot();
                let plan = optimize(lower(&parse(&q).unwrap()).unwrap(), &StatsCatalog::empty());
                let rows = execute(&plan, &snapshot, &params).await.unwrap();
                let want = oracle_distance(&edges, *types, dir, &a, &b, max);
                match (rows.first(), want) {
                    (None, None) => {}
                    (Some(row), Some(d)) => {
                        assert_eq!(
                            row.get("l"),
                            Some(&RuntimeValue::Integer(d as i64)),
                            "seed {seed} {q}"
                        );
                        let RuntimeValue::List(ns) = row.get("ns").unwrap() else {
                            panic!()
                        };
                        let RuntimeValue::List(rs) = row.get("rs").unwrap() else {
                            panic!()
                        };
                        assert_eq!(ns.len(), d + 1);
                        assert_eq!(ns.first(), Some(&RuntimeValue::String(a.clone())));
                        assert_eq!(ns.last(), Some(&RuntimeValue::String(b.clone())));
                        for (i, r) in rs.iter().enumerate() {
                            let RuntimeValue::Rel(rel) = r else {
                                panic!("{r:?}")
                            };
                            let (t, start, end) = (
                                rel.edge_type.clone(),
                                rel.src.to_string(),
                                rel.dst.to_string(),
                            );
                            assert!(
                                edges.contains(&(t.clone(), start.clone(), end.clone())),
                                "seed {seed}: rel not live"
                            );
                            if let Some(ts) = types {
                                assert!(ts.contains(&t.as_str()));
                            }
                            let (u, v) = (&ns[i], &ns[i + 1]);
                            let fwd = RuntimeValue::String(start.clone()) == *u
                                && RuntimeValue::String(end.clone()) == *v;
                            let bwd = RuntimeValue::String(end) == *u
                                && RuntimeValue::String(start) == *v;
                            match *dir {
                                "->" => assert!(fwd, "seed {seed}: hop {i} not forward"),
                                "<-" => assert!(bwd, "seed {seed}: hop {i} not backward"),
                                _ => assert!(fwd || bwd, "seed {seed}: hop {i} not incident"),
                            }
                        }
                    }
                    (got, want) => panic!("seed {seed}: {q}\n  got {got:?}\n  oracle {want:?}"),
                }
            }
        }
    }
}
