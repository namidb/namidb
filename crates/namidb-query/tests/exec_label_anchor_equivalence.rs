//! Label predicates and hub anchors: the rewritten plan must return exactly
//! what the naive plan returns.
//!
//! Two planner rewrites change where a pattern starts:
//!
//! - `WHERE n:L` (and `n:A:B`, `n:A OR n:B`, `(n:A|B)`) on an unlabelled
//!   pattern variable becomes a label scan, a disjoint `UNION ALL` of label
//!   scans, or the label check of the Expand that reaches the variable,
//!   instead of a scan of every node with a filter on top;
//! - a hop that starts at an id-pinned hub (`WHERE elementId(d) IN $ids`)
//!   starts from a scan of the hop's small target label instead, when the
//!   edge type's degree statistics say the hub side is far larger.
//!
//! Each query below runs twice against the same snapshot: through the full
//! optimizer, and as lowered (no rewrite at all), and the two row multisets
//! must agree. A property test then does the same for random label/property
//! predicates in random positions.

use std::collections::BTreeMap;
use std::sync::Arc;

use namidb_core::id::{NamespaceId, NodeId};
use namidb_core::value::Value as CoreValue;
use namidb_query::{
    execute, explain, lower, optimize, parse, Params, Row, RuntimeValue, StatsCatalog,
};
use namidb_storage::{EdgeWriteRecord, NamespacePaths, NodeWriteRecord, WriterSession};
use object_store::memory::InMemory;
use object_store::ObjectStore;
use proptest::prelude::*;
use proptest::test_runner::{Config, TestRunner};

fn node(name: String, extra: Vec<(&str, CoreValue)>) -> NodeWriteRecord {
    let mut properties: BTreeMap<String, CoreValue> =
        extra.into_iter().map(|(k, v)| (k.to_string(), v)).collect();
    properties.insert("name".into(), CoreValue::Str(name));
    NodeWriteRecord {
        properties,
        schema_version: 0,
        labels: vec![],
    }
}

/// A small graph of the reported shape: three `:Device` hubs, the first one
/// holding most `ON_DEVICE` edges; a small `:Call` label (some calls also
/// `:Recorded`, one node `:Contact:Call`); `:Message` and `:Contact` bulk.
/// Flushed to an SST so the catalog has label and degree statistics, then a
/// few memtable writes on top (a new call, a deleted message, a relabelled
/// node) so the rewrites run against a mixed snapshot.
async fn fixture() -> (WriterSession, Vec<String>) {
    let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
    let paths = NamespacePaths::new("tenants", NamespaceId::new("label-anchor").unwrap());
    let mut w = WriterSession::open(store, paths).await.unwrap();

    let devices: Vec<NodeId> = (0..3).map(|_| NodeId::new()).collect();
    for (i, d) in devices.iter().enumerate() {
        w.upsert_node(
            "Device",
            *d,
            &node(
                format!("device-{i}"),
                vec![("kind", CoreValue::I64(i as i64))],
            ),
        )
        .unwrap();
    }
    let device_of = |i: usize| -> NodeId {
        match i % 10 {
            0..=7 => devices[0],
            8 => devices[1],
            _ => devices[2],
        }
    };
    let mut deleted = None;
    for i in 0..1200usize {
        let id = NodeId::new();
        let ts = CoreValue::I64((i * 37 % 101) as i64);
        let (labels, extra): (Vec<&str>, Vec<(&str, CoreValue)>) = match i {
            0..=13 => (
                vec!["Call"],
                vec![("ts", ts), ("kind", CoreValue::I64((i % 3) as i64))],
            ),
            14..=17 => (vec!["Call", "Recorded"], vec![("ts", ts)]),
            18 => (vec!["Contact", "Call"], vec![("ts", ts)]),
            // No `ts` at all: NULL comparisons must agree too.
            19 => (vec!["Call"], vec![]),
            _ if i % 3 == 0 => (
                vec!["Contact"],
                vec![("kind", CoreValue::I64((i % 4) as i64))],
            ),
            _ => (vec!["Message"], vec![("ts", ts)]),
        };
        let record = node(format!("n{i:03}"), extra);
        if labels.len() == 1 {
            w.upsert_node(labels[0], id, &record).unwrap();
        } else {
            w.upsert_node_with_labels(labels.iter().map(|l| l.to_string()), id, &record)
                .unwrap();
        }
        if i == 21 {
            deleted = Some(id);
        }
        w.upsert_edge("ON_DEVICE", id, device_of(i), &EdgeWriteRecord::default())
            .unwrap();
    }
    w.commit_batch().await.unwrap();
    let schema = w.schema().clone();
    w.flush(schema).await.unwrap();

    // Memtable layer on top of the SST.
    let fresh = NodeId::new();
    w.upsert_node(
        "Call",
        fresh,
        &node("fresh-call".into(), vec![("ts", CoreValue::I64(5))]),
    )
    .unwrap();
    w.upsert_edge("ON_DEVICE", fresh, devices[0], &EdgeWriteRecord::default())
        .unwrap();
    w.tombstone_node("Message", deleted.unwrap()).unwrap();
    w.commit_batch().await.unwrap();

    let hubs = devices.iter().map(|d| d.to_string()).collect();
    (w, hubs)
}

fn multiset(rows: &[Row]) -> Vec<String> {
    let mut out: Vec<String> = rows
        .iter()
        .map(|r| {
            r.bindings
                .iter()
                .map(|(k, v)| format!("{k}={}", render(v)))
                .collect::<Vec<_>>()
                .join(",")
        })
        .collect();
    out.sort();
    out
}

fn render(v: &RuntimeValue) -> String {
    match v {
        RuntimeValue::Null => "null".into(),
        RuntimeValue::String(s) => s.clone(),
        RuntimeValue::Integer(n) => n.to_string(),
        RuntimeValue::Float(f) => format!("{f:?}"),
        RuntimeValue::Bool(b) => b.to_string(),
        RuntimeValue::Node(n) => format!(
            "node({},{:?})",
            n.properties
                .get("name")
                .map(render)
                .unwrap_or_else(|| "?".into()),
            n.labels
        ),
        RuntimeValue::List(items) => {
            let mut parts: Vec<String> = items.iter().map(render).collect();
            parts.sort();
            format!("[{}]", parts.join(","))
        }
        other => format!("{other:?}"),
    }
}

struct Harness {
    writer: WriterSession,
    params: Params,
}

impl Harness {
    async fn new() -> Self {
        let (writer, hubs) = fixture().await;
        let mut params = Params::new();
        params.insert(
            "hubs".into(),
            RuntimeValue::List(hubs.iter().cloned().map(RuntimeValue::String).collect()),
        );
        params.insert("hub".into(), RuntimeValue::String(hubs[0].clone()));
        params.insert(
            "some".into(),
            RuntimeValue::List(vec![
                RuntimeValue::String(hubs[0].clone()),
                RuntimeValue::String(hubs[0].to_uppercase()),
                RuntimeValue::String(NodeId::new().to_string()),
                RuntimeValue::Null,
            ]),
        );
        Self { writer, params }
    }

    /// `(optimized plan text, optimized rows, naive rows)`.
    async fn run(&self, query: &str) -> (String, Vec<String>, Vec<String>) {
        let snapshot = self.writer.snapshot();
        let catalog = StatsCatalog::from_manifest(&snapshot.manifest().manifest);
        let parsed = parse(query).unwrap_or_else(|e| panic!("parse `{query}`: {e:?}"));
        let naive = lower(&parsed).unwrap_or_else(|e| panic!("lower `{query}`: {e:?}"));
        let optimized = optimize(naive.clone(), &catalog);
        let fast = execute(&optimized, &snapshot, &self.params)
            .await
            .unwrap_or_else(|e| panic!("optimized `{query}`: {e:?}"));
        let slow = execute(&naive, &snapshot, &self.params)
            .await
            .unwrap_or_else(|e| panic!("naive `{query}`: {e:?}"));
        (explain(&optimized), multiset(&fast), multiset(&slow))
    }

    async fn assert_same(&self, query: &str) -> String {
        let (plan, fast, slow) = self.run(query).await;
        assert_eq!(
            fast, slow,
            "\n`{query}` disagrees with its naive plan\n{plan}"
        );
        plan
    }
}

#[tokio::test]
async fn label_predicates_match_the_naive_plan_and_never_scan_everything() {
    let h = Harness::new().await;
    for query in [
        "MATCH (n) WHERE n:Device RETURN count(n) AS c",
        "MATCH (n) WHERE n:Call RETURN n.name AS v",
        "MATCH (n) WHERE n:Call AND n:Recorded RETURN n.name AS v",
        "MATCH (n) WHERE n:Recorded:Call RETURN n.name AS v",
        "MATCH (n) WHERE n:Call AND NOT n:Recorded RETURN n.name AS v",
        "MATCH (n) WHERE n:Call AND NOT n:Contact AND n.ts > 40 RETURN n.name AS v",
        "MATCH (n) WHERE n:Device OR n:Call RETURN count(n) AS c",
        "MATCH (n) WHERE n:Call OR n:Contact RETURN n.name AS v",
        "MATCH (n) WHERE (n:Call OR n:Contact) AND n.ts > 20 RETURN n.name AS v",
        "MATCH (n) WHERE (n:Call AND n.ts > 50) OR (n:Contact AND n.kind = 1) RETURN n.name AS v",
        "MATCH (n) WHERE n:Call OR n:Recorded OR n:Nope RETURN n.name AS v",
        "MATCH (n:Call|Contact) RETURN n.name AS v",
        "MATCH (n:Call|Device) WHERE n.ts IS NULL RETURN n.name AS v",
        "MATCH (n) WHERE n:Nope RETURN count(n) AS c",
        "MATCH (n) WHERE n:Call RETURN n",
        "MATCH (n) WHERE n:Call RETURN labels(n) AS l",
        "MATCH (d:Device) WITH d MATCH (n) WHERE n:Call RETURN d.name AS d, n.name AS v",
        "MATCH (d:Device) WITH count(d) AS k MATCH (n) WHERE n:Call RETURN k, count(n) AS c",
        "MATCH (d:Device) OPTIONAL MATCH (n) WHERE n:Call AND n.kind = d.kind \
         RETURN d.name AS d, n.name AS v",
        "MATCH (d:Device) OPTIONAL MATCH (d)<-[:ON_DEVICE]-(n) WHERE n:Call \
         RETURN d.name AS d, n.name AS v",
        "MATCH (d:Device)<-[:ON_DEVICE]-(n) WHERE n:Call RETURN d.name AS d, n.name AS v",
        "MATCH (d)<-[:ON_DEVICE]-(n) WHERE d:Device AND n:Call:Recorded \
         RETURN d.name AS d, n.name AS v",
        "MATCH (a), (b) WHERE a:Device AND b:Call AND a.kind = b.kind \
         RETURN a.name AS a, b.name AS b",
        "MATCH (n) WHERE n:Call WITH n WHERE n.ts > 10 RETURN n.name AS v",
        "MATCH (n) WHERE n:Call OR n:Contact WITH n ORDER BY n.name LIMIT 5 RETURN n.name AS v",
        "MATCH (n) WHERE (n:Call) RETURN n.name AS v",
        "MATCH (n) WHERE (n:Call OR n:Device) RETURN n.name AS v",
    ] {
        let plan = h.assert_same(query).await;
        assert!(
            !plan.contains("label=*"),
            "`{query}` must not scan every node:\n{plan}"
        );
    }
    // No label to start from: a full scan is the only correct plan, and it
    // still agrees.
    for query in [
        "MATCH (n) WHERE NOT n:Message RETURN n.name AS v",
        "MATCH (n) WHERE n:Call OR n.kind = 1 RETURN n.name AS v",
    ] {
        h.assert_same(query).await;
    }
}

#[tokio::test]
async fn hub_anchored_hops_match_the_naive_plan_and_start_from_the_scan() {
    let h = Harness::new().await;
    for query in [
        "MATCH (d:Device)<-[:ON_DEVICE]-(c:Call) WHERE elementId(d) IN $hubs \
         RETURN d.name AS d, c.name AS c",
        "MATCH (c:Call)-[:ON_DEVICE]->(d:Device) WHERE elementId(d) IN $hubs \
         RETURN d.name AS d, c.name AS c",
        "MATCH (d)<-[:ON_DEVICE]-(c) WHERE elementId(d) IN $hubs AND c:Call RETURN c.name AS c",
        "MATCH (d:Device)<-[:ON_DEVICE]-(c:Call) WHERE elementId(d) = $hub RETURN c.ts AS ts",
        "MATCH (d:Device)<-[:ON_DEVICE]-(c:Call) WHERE id(d) = $hub RETURN c",
        "MATCH (d:Device)<-[r:ON_DEVICE]-(c:Call) WHERE elementId(d) IN $some \
         RETURN c.name AS c, type(r) AS t",
        "MATCH (d:Device)<-[:ON_DEVICE]-(c:Call:Recorded) WHERE elementId(d) IN $hubs \
         RETURN c.name AS c",
        "MATCH (d:Device)<-[:ON_DEVICE]-(c:Call) WHERE elementId(d) IN $hubs AND d.kind > 0 \
         RETURN c.name AS c",
        "MATCH (d:Device)<-[:ON_DEVICE]-(c:Call) WHERE elementId(d) IN $hubs AND c.ts > 30 \
         RETURN d.name AS d, count(c) AS n",
    ] {
        let plan = h.assert_same(query).await;
        assert!(
            !plan.contains("NodeById"),
            "`{query}` must not start at the hub:\n{plan}"
        );
    }
    // Forced-start spellings and both-ends-bound shapes agree as well.
    for query in [
        "MATCH (c:Call) WITH c MATCH (d:Device)<-[:ON_DEVICE]-(c) WHERE elementId(d) IN $hubs \
         RETURN c.name AS c",
        "MATCH (d:Device) WHERE elementId(d) IN $hubs OPTIONAL MATCH (d)<-[:ON_DEVICE]-(c:Call) \
         RETURN d.name AS d, c.name AS c",
        "MATCH (d:Device)<-[:ON_DEVICE]-(m:Message) WHERE elementId(d) = $hub \
         RETURN count(m) AS n",
    ] {
        h.assert_same(query).await;
    }
}

/// A random WHERE over `n`: label tests, negations, property comparisons
/// and NULL tests under AND / OR / NOT, in any nesting.
fn predicate() -> impl Strategy<Value = String> {
    let label = prop::sample::select(vec![
        "Device", "Call", "Recorded", "Message", "Contact", "Nope",
    ]);
    let leaf = prop_oneof![
        3 => label.clone().prop_map(|l| format!("n:{l}")),
        1 => (label.clone(), label.clone()).prop_map(|(a, b)| format!("n:{a}:{b}")),
        1 => label.prop_map(|l| format!("NOT n:{l}")),
        1 => (prop::sample::select(vec!["ts", "kind"]), prop::sample::select(vec!["<", ">", "="]), 0..80i64)
            .prop_map(|(k, op, v)| format!("n.{k} {op} {v}")),
        1 => prop::sample::select(vec!["n.ts IS NULL", "n.kind IS NOT NULL"]).prop_map(str::to_string),
    ];
    leaf.prop_recursive(3, 12, 2, |inner| {
        prop_oneof![
            (inner.clone(), inner.clone()).prop_map(|(a, b)| format!("({a} AND {b})")),
            (inner.clone(), inner.clone()).prop_map(|(a, b)| format!("({a} OR {b})")),
            inner.prop_map(|a| format!("NOT ({a})")),
        ]
    })
}

fn query() -> impl Strategy<Value = String> {
    let templates = vec![
        "MATCH (n) WHERE {p} RETURN n.name AS v",
        "MATCH (n) WHERE {p} RETURN count(n) AS c",
        "MATCH (d:Device)<-[:ON_DEVICE]-(n) WHERE {p} RETURN d.name AS d, n.name AS v",
        "MATCH (d)<-[:ON_DEVICE]-(n) WHERE elementId(d) IN $hubs AND {p} RETURN n.name AS v",
        "MATCH (d:Device) WITH d MATCH (n) WHERE {p} AND n.kind = d.kind \
         RETURN d.name AS d, n.name AS v",
        "MATCH (d:Device) OPTIONAL MATCH (d)<-[:ON_DEVICE]-(n) WHERE {p} \
         RETURN d.name AS d, n.name AS v",
        "MATCH (n:Call|Contact) WHERE {p} RETURN n.name AS v",
        "MATCH (n) WHERE ({p}) AND (n)-[:ON_DEVICE]->() RETURN n.name AS v",
        "MATCH (n) WHERE ({p}) AND NOT (n)-[:ON_DEVICE]->(:Device) RETURN n.name AS v",
        "MATCH (d:Device) CALL { WITH d MATCH (d)<-[:ON_DEVICE]-(n) WHERE {p} \
         RETURN count(n) AS c } RETURN d.name AS d, c",
    ];
    (prop::sample::select(templates), predicate()).prop_map(|(t, p)| t.replace("{p}", &p))
}

#[test]
fn random_label_predicates_match_the_naive_plan() {
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let h = rt.block_on(Harness::new());
    let cases = std::env::var("NAMIDB_PROPTEST_CASES")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(192);
    let mut runner = TestRunner::new(Config {
        cases,
        failure_persistence: None,
        ..Config::default()
    });
    runner
        .run(&query(), |q| {
            let (plan, fast, slow) = rt.block_on(h.run(&q));
            prop_assert_eq!(
                fast,
                slow,
                "`{}` disagrees with its naive plan\n{}",
                q,
                plan
            );
            Ok(())
        })
        .unwrap();
}

/// A single-hop Expand reads the first hop of all its seeds with several
/// reads in flight, and an Expand-Into (both ends bound) runs its edge
/// probes the same way. Every seed must still get exactly the partners a
/// query over that one seed finds, in every direction, with the SST edges
/// and the memtable edge of the fixture alike.
#[tokio::test]
async fn multi_seed_expand_matches_one_seed_at_a_time() {
    let h = Harness::new().await;
    let (_, names, _) = h.run("MATCH (c:Call) RETURN c.name AS c").await;
    assert!(names.len() > 10, "{names:?}");
    for pattern in [
        "(c)-[:ON_DEVICE]->(d)",
        "(c)-[:ON_DEVICE]-(d)",
        "(d)<-[:ON_DEVICE]-(c)",
    ] {
        let (_, batch, _) = h
            .run(&format!(
                "MATCH (c:Call) MATCH {pattern} RETURN c.name AS c, d.name AS d"
            ))
            .await;
        let (_, into, _) = h
            .run(&format!(
                "MATCH (c:Call), (d:Device) MATCH {pattern} RETURN c.name AS c, d.name AS d"
            ))
            .await;
        let mut single = Vec::new();
        for name in &names {
            let name = name.strip_prefix("c=").unwrap();
            let (_, rows, _) = h
                .run(&format!(
                    "MATCH (c:Call {{name: '{name}'}}) MATCH {pattern} \
                     RETURN c.name AS c, d.name AS d"
                ))
                .await;
            single.extend(rows);
        }
        single.sort();
        assert_eq!(batch.len(), names.len(), "{pattern}: {batch:?}");
        assert_eq!(batch, single, "{pattern}");
        assert_eq!(into, single, "{pattern}");
    }
}

/// A single hop from a fresh node to one an earlier clause bound walks from
/// the bound node instead of scanning every node and probing each pair.
#[tokio::test]
async fn hop_to_a_bound_node_starts_from_it_and_agrees() {
    let h = Harness::new().await;
    for query in [
        "MATCH (c:Call) MATCH (d)<-[:ON_DEVICE]-(c) RETURN c.name AS c, d.name AS d",
        "MATCH (c:Call) WITH c MATCH (d)<-[:ON_DEVICE]-(c) RETURN c.name AS c, d.name AS d",
        "MATCH (c:Call) MATCH (d)-[:ON_DEVICE]-(c) RETURN c.name AS c, d.name AS d",
        "MATCH (c:Call) MATCH (d)<-[r:ON_DEVICE]-(c) RETURN type(r) AS t, d.name AS d",
        "MATCH (c:Call) MATCH (d)<-[:ON_DEVICE]-(c) WHERE d.name STARTS WITH 'device' \
         RETURN c.name AS c",
        "MATCH (c:Call) MATCH (d)<-[:ON_DEVICE]-(c) WHERE d.kind = c.kind RETURN c.name AS c",
        "MATCH (c:Call) MATCH (d)<--(c) RETURN c.name AS c, d.name AS d",
        "MATCH (c:Call) MATCH (d:Device)<-[:ON_DEVICE]-(c) RETURN c.name AS c, d.name AS d",
        "MATCH (c:Call) MATCH (d:Device:Nope)<-[:ON_DEVICE]-(c) RETURN count(*) AS n",
        "MATCH (d:Device) MATCH (m)-[:ON_DEVICE]->(d) RETURN d.name AS d, count(*) AS n",
        "MATCH (d:Device) MATCH (m)-[:ON_DEVICE]->(d) RETURN d.name AS d, m.name AS m",
        "MATCH (d:Device) MATCH (c:Call)-[:ON_DEVICE]->(d) RETURN d.name AS d, c.name AS c",
        "MATCH (c) WHERE c.name STARTS WITH 'n00' WITH c MATCH (d)<-[:ON_DEVICE]-(c:Call) \
         RETURN c.name AS c, d.name AS d",
        "MATCH (c:Call)-[r1:ON_DEVICE]->(x), (d)<-[r2:ON_DEVICE]-(c) RETURN c.name AS c, d.name AS d",
        "MATCH (c:Call)-[r1:ON_DEVICE]->(x) MATCH (d)<-[r2:ON_DEVICE]-(c) \
         RETURN c.name AS c, d.name AS d",
        "MATCH (c:Call) OPTIONAL MATCH (d)<-[:ON_DEVICE]-(c) WHERE d.name ENDS WITH '2' \
         RETURN c.name AS c, d.name AS d",
        "MATCH (c:Call) OPTIONAL MATCH (d)-[:NOPE]->(c) RETURN c.name AS c, d.name AS d",
        "MATCH (c:Call) MATCH (d)<-[:ON_DEVICE]-(c) MATCH (e)<-[:ON_DEVICE]-(c) \
         RETURN c.name AS c, d.name AS d, e.name AS e",
        "UNWIND range(1, 2) AS i MATCH (n:Call) WITH n, i MATCH (d)<-[:ON_DEVICE]-(n) \
         RETURN count(*) AS c",
    ] {
        let plan = h.assert_same(query).await;
        assert!(
            !plan.contains("label=* alias=d") && !plan.contains("label=* alias=m"),
            "`{query}` must not scan every node:\n{plan}"
        );
    }
    // Shapes that keep the scan (a literal filter folded into it, an id
    // seek on the new end, a named path) must still agree.
    for query in [
        "MATCH (c:Call) MATCH (d)<-[:ON_DEVICE]-(c) WHERE d.kind = 1 RETURN c.name AS c",
        "MATCH (c:Call) OPTIONAL MATCH (d)<-[:ON_DEVICE]-(c) WHERE d.kind = 2 \
         RETURN c.name AS c, d.name AS d",
        "MATCH (c:Call) MATCH (d)<-[:ON_DEVICE]-(c) WHERE elementId(d) IN $hubs \
         RETURN c.name AS c, d.name AS d",
        "MATCH (c:Call) MATCH p = (d)<-[:ON_DEVICE]-(c) \
         RETURN c.name AS c, [x IN nodes(p) | x.name] AS names",
    ] {
        h.assert_same(query).await;
    }
}

/// The walk from the bound node and the probe from the scanned node must
/// agree on a self-loop and on edges in both directions between two nodes,
/// for every direction of the hop.
#[tokio::test]
async fn hop_to_a_bound_node_agrees_on_self_loops_and_two_way_edges() {
    let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
    let paths = NamespacePaths::new("tenants", NamespaceId::new("bound-loop").unwrap());
    let mut w = WriterSession::open(store, paths).await.unwrap();
    let (a, b, c) = (NodeId::new(), NodeId::new(), NodeId::new());
    w.upsert_node("X", a, &node("a".into(), vec![])).unwrap();
    w.upsert_node("Y", b, &node("b".into(), vec![])).unwrap();
    w.upsert_node("Y", c, &node("c".into(), vec![])).unwrap();
    for (from, to) in [(a, a), (a, b), (b, a), (c, a)] {
        w.upsert_edge("L", from, to, &EdgeWriteRecord::default())
            .unwrap();
    }
    w.commit_batch().await.unwrap();
    let schema = w.schema().clone();
    w.flush(schema).await.unwrap();
    let snapshot = w.snapshot();
    let catalog = StatsCatalog::from_manifest(&snapshot.manifest().manifest);
    for query in [
        "MATCH (x:X) MATCH (o)-[:L]-(x) RETURN x.name AS x, o.name AS o",
        "MATCH (x:X) MATCH (o)-[:L]->(x) RETURN x.name AS x, o.name AS o",
        "MATCH (x:X) MATCH (o)<-[:L]-(x) RETURN x.name AS x, o.name AS o",
        "MATCH (x:X) MATCH (o)-[r:L]-(x) RETURN x.name AS x, o.name AS o, type(r) AS t",
        "MATCH (x) MATCH (o)-[:L]-(x) RETURN x.name AS x, o.name AS o",
    ] {
        let naive = lower(&parse(query).unwrap()).unwrap();
        let optimized = optimize(naive.clone(), &catalog);
        let fast = execute(&optimized, &snapshot, &Params::new())
            .await
            .unwrap();
        let slow = execute(&naive, &snapshot, &Params::new()).await.unwrap();
        assert_eq!(
            multiset(&fast),
            multiset(&slow),
            "`{query}`\n{}",
            explain(&optimized)
        );
    }
}

/// More input rows than one read-ahead window (4,096), with every seed
/// repeated: the windowed prefetch of both Expand forms must still give every
/// row exactly its own partners.
#[tokio::test]
async fn multi_seed_expand_across_read_ahead_windows() {
    let h = Harness::new().await;
    let count = |rows: Vec<String>| -> i64 {
        rows[0]
            .rsplit('=')
            .next()
            .unwrap()
            .parse()
            .unwrap_or_else(|e| panic!("{rows:?}: {e}"))
    };
    let (_, messages, _) = h.run("MATCH (n:Message) RETURN count(n) AS c").await;
    let messages = count(messages);
    assert!(messages * 6 > 4_096, "{messages}");
    for (query, expected) in [
        (
            "UNWIND range(1, 6) AS i MATCH (n:Message) WITH n, i \
             MATCH (n)-[:ON_DEVICE]->(d) RETURN count(*) AS c",
            messages * 6,
        ),
        (
            "UNWIND range(1, 6) AS i MATCH (n:Message) WITH n, i \
             MATCH (d)<-[:ON_DEVICE]-(n) RETURN count(*) AS c",
            messages * 6,
        ),
        (
            "UNWIND range(1, 6) AS i MATCH (n:Message), (d:Device) WITH n, d, i \
             MATCH (n)-[:ON_DEVICE]->(d) RETURN count(*) AS c",
            messages * 6,
        ),
        (
            "UNWIND range(1, 6) AS i MATCH (n:Message), (d:Device) WITH n, d, i \
             MATCH (n)-[:ON_DEVICE]-(d) RETURN count(*) AS c",
            messages * 6,
        ),
    ] {
        // Optimized only: the counts are known, and the lowered form of a
        // hop to a bound node is the rows x nodes cross product the
        // optimizer exists to avoid.
        let snapshot = h.writer.snapshot();
        let catalog = StatsCatalog::from_manifest(&snapshot.manifest().manifest);
        let plan = optimize(lower(&parse(query).unwrap()).unwrap(), &catalog);
        let rows = execute(&plan, &snapshot, &h.params).await.unwrap();
        assert_eq!(
            count(multiset(&rows)),
            expected,
            "`{query}`\n{}",
            explain(&plan)
        );
    }
}

/// Shapes the rewrites meet inside other operators: subqueries, pattern
/// comprehensions and predicates, UNION, ordering, DISTINCT, aggregation,
/// variable-length and multi-hop patterns, and labels on both ends of a hop.
#[tokio::test]
async fn label_rewrites_agree_inside_subqueries_unions_and_multi_hop_patterns() {
    let h = Harness::new().await;
    for query in [
        "MATCH (n) WHERE n:Call OR n:Call RETURN n.name AS v",
        "MATCH (n) WHERE n:Call OR (n:Call AND n.ts > 3) RETURN n.name AS v",
        "MATCH (n) WHERE (n:Call AND n.ts > 50) OR (n:Recorded AND n.ts < 60) RETURN n.name AS v",
        "MATCH (n) WHERE (n:Recorded AND n.ts IS NULL) OR n:Call RETURN n.name AS v",
        "MATCH (n) WHERE n:Call OR n:Recorded RETURN DISTINCT labels(n) AS l",
        "MATCH (n) WHERE n:Call OR n:Contact RETURN labels(n) AS l, count(*) AS c",
        "MATCH (n) WHERE n:Call OR n:Contact RETURN n.kind AS k, collect(n.name) AS names",
        "MATCH (d:Device) WHERE EXISTS { MATCH (n) WHERE n:Call OR n:Contact } RETURN d.name AS d",
        "MATCH (d:Device) WHERE EXISTS { MATCH (d)<-[:ON_DEVICE]-(n) WHERE n:Call OR n:Contact } \
         RETURN d.name AS d",
        "MATCH (d:Device) CALL { WITH d MATCH (n) WHERE (n:Call OR n:Contact) AND n.kind = d.kind \
         RETURN count(n) AS c } RETURN d.name AS d, c",
        "MATCH (d:Device) CALL { MATCH (n) WHERE n:Call OR n:Recorded RETURN count(n) AS c } \
         RETURN d.name AS d, c",
        "MATCH (d:Device) RETURN d.name AS d, [(d)<-[:ON_DEVICE]-(n) WHERE n:Call | n.name] AS calls",
        "MATCH (n) WHERE (n:Call OR n:Contact) AND (n)-[:ON_DEVICE]->() RETURN n.name AS v",
        "MATCH (n) WHERE n:Call AND NOT (n)-[:ON_DEVICE]->(:Device) RETURN n.name AS v",
        "MATCH (n) WHERE n:Call RETURN n.name AS v \
         UNION MATCH (n) WHERE n:Contact OR n:Call RETURN n.name AS v",
        "MATCH (n) WHERE n:Call RETURN n.name AS v \
         UNION ALL MATCH (n) WHERE n:Recorded OR n:Call RETURN n.name AS v",
        "MATCH (d:Device) OPTIONAL MATCH (n) WHERE (n:Call OR n:Contact) AND n.kind = d.kind \
         RETURN d.name AS d, n.name AS v",
        "MATCH (d:Device) OPTIONAL MATCH (n) WHERE n:Nope OR n:Nope2 RETURN d.name AS d, n.name AS v",
        "MATCH (d:Device) OPTIONAL MATCH (n) WHERE n:Nope RETURN d.name AS d, n.name AS v",
        "MATCH (d:Device)<-[:ON_DEVICE]-(n) WHERE NOT n:Call RETURN count(n) AS c",
        "MATCH (d:Device)<-[:ON_DEVICE]-(n) WHERE n:Call OR n:Contact RETURN count(n) AS c",
        "MATCH (d:Device)<-[:ON_DEVICE]-(n) WHERE n:Call AND n:Recorded RETURN n.name AS v",
        "MATCH (a)-[:ON_DEVICE]->(d)<-[:ON_DEVICE]-(b) WHERE a:Recorded AND b:Call AND d:Device \
         RETURN a.name AS a, b.name AS b",
        "MATCH (d:Device)<-[:ON_DEVICE*1..1]-(n) WHERE n:Call RETURN count(n) AS c",
        "MATCH (d:Device)<-[:ON_DEVICE*0..1]-(n) WHERE n:Device RETURN count(n) AS c",
        "MATCH p = (d:Device)<-[:ON_DEVICE]-(n) WHERE n:Call RETURN length(p) AS l, n.name AS v",
        "MATCH (d:Device)<-[r:ON_DEVICE]-(n) WHERE n:Call RETURN type(r) AS t, n.name AS v",
        "MATCH (n), (m) WHERE n:Device AND (m:Device OR m:Recorded) RETURN n.name AS a, m.name AS b",
        "MATCH (n) WHERE n:Call WITH n, n.ts AS ts WHERE ts IS NULL OR ts > 40 \
         MATCH (m) WHERE m:Device AND m.kind = n.kind RETURN n.name AS a, m.name AS b",
        "UNWIND [1, 2] AS i MATCH (n) WHERE n:Device OR n:Recorded RETURN i, n.name AS v",
        "MATCH (n) WHERE n:Call AND n:Call RETURN n.name AS v",
        "MATCH (n) WHERE n:Call AND (n:Recorded OR n:Contact) RETURN n.name AS v",
        "MATCH (n) WHERE NOT (NOT n:Call) RETURN n.name AS v",
        "MATCH (n) WHERE n:Call XOR n:Recorded RETURN n.name AS v",
        "MATCH (n) WHERE coalesce(n:Call, false) RETURN n.name AS v",
        "MATCH (n) WHERE (n:Call OR n:Contact) AND n.kind IN [0, 1] RETURN n.name AS v",
    ] {
        h.assert_same(query).await;
    }
    // Ordered results must agree row for row, not only as a multiset.
    for query in [
        "MATCH (n) WHERE n:Call OR n:Contact RETURN n.name AS v ORDER BY v SKIP 3 LIMIT 7",
        "MATCH (n) WHERE n:Call OR n:Recorded RETURN n.name AS v ORDER BY n.ts DESC, v LIMIT 9",
        "MATCH (d:Device)<-[:ON_DEVICE]-(c:Call) WHERE elementId(d) IN $hubs \
         RETURN c.name AS c ORDER BY c LIMIT 5",
    ] {
        let snapshot = h.writer.snapshot();
        let catalog = StatsCatalog::from_manifest(&snapshot.manifest().manifest);
        let naive = lower(&parse(query).unwrap()).unwrap();
        let optimized = optimize(naive.clone(), &catalog);
        let fast = execute(&optimized, &snapshot, &h.params).await.unwrap();
        let slow = execute(&naive, &snapshot, &h.params).await.unwrap();
        let render_rows = |rows: &[Row]| -> Vec<String> {
            rows.iter()
                .map(|r| {
                    r.bindings
                        .iter()
                        .map(|(k, v)| format!("{k}={}", render(v)))
                        .collect::<Vec<_>>()
                        .join(",")
                })
                .collect()
        };
        assert_eq!(
            render_rows(&fast),
            render_rows(&slow),
            "`{query}`\n{}",
            explain(&optimized)
        );
    }
}

/// Id lists with duplicates, NULLs, ids of the wrong label, undirected and
/// multi-type hops, longer chains with relationship uniqueness, and an id
/// anchor inside a subquery.
#[tokio::test]
async fn hub_deanchor_agrees_on_odd_id_lists_and_chains() {
    let mut h = Harness::new().await;
    let snapshot = h.writer.snapshot();
    let hubs = match h.params.get("hubs") {
        Some(RuntimeValue::List(items)) => items.clone(),
        _ => unreachable!(),
    };
    drop(snapshot);
    let some_call = {
        let (_, rows, _) = h
            .run("MATCH (c:Call) WHERE c.name = 'n001' RETURN elementId(c) AS id")
            .await;
        rows[0].strip_prefix("id=").unwrap().to_string()
    };
    h.params.insert(
        "dups".into(),
        RuntimeValue::List(vec![hubs[0].clone(), hubs[0].clone(), hubs[1].clone()]),
    );
    h.params.insert(
        "mixed".into(),
        RuntimeValue::List(vec![
            hubs[0].clone(),
            RuntimeValue::String(some_call),
            RuntimeValue::Null,
        ]),
    );
    h.params
        .insert("nothing".into(), RuntimeValue::List(vec![]));
    h.params.insert("null".into(), RuntimeValue::Null);
    for query in [
        "MATCH (d:Device)<-[:ON_DEVICE]-(c:Call) WHERE elementId(d) IN $dups RETURN c.name AS c",
        "MATCH (d:Device)<-[:ON_DEVICE]-(c:Call) WHERE elementId(d) IN $mixed RETURN c.name AS c",
        "MATCH (d)<-[:ON_DEVICE]-(c:Call) WHERE elementId(d) IN $mixed RETURN c.name AS c",
        "MATCH (d:Device)<-[:ON_DEVICE]-(c:Call) WHERE elementId(d) IN $nothing RETURN c.name AS c",
        "MATCH (d:Device)<-[:ON_DEVICE]-(c:Call) WHERE elementId(d) IN $null RETURN c.name AS c",
        "MATCH (d:Device)<-[:ON_DEVICE]-(c:Call) WHERE elementId(d) = $null RETURN c.name AS c",
        "MATCH (d:Device)-[:ON_DEVICE]-(c:Call) WHERE elementId(d) IN $hubs RETURN c.name AS c",
        "MATCH (d:Device)<-[:ON_DEVICE|NOPE]-(c:Call) WHERE elementId(d) IN $hubs RETURN c.name AS c",
        "MATCH (d:Device)<-[r1:ON_DEVICE]-(c:Call)-[r2:ON_DEVICE]->(e) WHERE elementId(d) IN $hubs \
         RETURN c.name AS c, e.name AS e",
        "MATCH (d:Device)<-[:ON_DEVICE]-(c:Call)-[:ON_DEVICE]->(e:Device) WHERE elementId(d) IN $hubs \
         RETURN c.name AS c, e.name AS e",
        "MATCH (d:Device)<-[:ON_DEVICE]-(c:Call) WHERE elementId(d) IN $hubs AND c.kind = d.kind \
         RETURN c.name AS c",
        "MATCH (d:Device)<-[:ON_DEVICE]-(c:Call) WHERE elementId(d) IN $hubs \
         RETURN DISTINCT d.name AS d",
        "MATCH (d:Device)<-[:ON_DEVICE]-(c:Call) WHERE elementId(d) IN $hubs \
         RETURN d, count(c) AS n",
        "MATCH (x:Device) WHERE x.kind = 0 CALL { WITH x MATCH (d:Device)<-[:ON_DEVICE]-(c:Call) \
         WHERE elementId(d) = elementId(x) RETURN count(c) AS n } RETURN x.name AS x, n",
        "MATCH (d:Device)<-[:ON_DEVICE]-(c:Call) WHERE elementId(d) IN $hubs \
         OPTIONAL MATCH (c)-[:ON_DEVICE]->(e:Device) RETURN c.name AS c, e.name AS e",
    ] {
        h.assert_same(query).await;
    }
}

/// A write followed by a read in the same statement: the label scan the
/// rewrite introduces must see the statement's own writes exactly as the
/// full scan does.
#[tokio::test]
async fn label_rewrites_see_writes_of_the_same_statement() {
    for query in [
        "CREATE (:Call {name: 'created'}) WITH 1 AS one MATCH (n) WHERE n:Call \
         RETURN count(n) AS c",
        "CREATE (:Call {name: 'created'}) WITH 1 AS one MATCH (n) WHERE n:Call OR n:Device \
         RETURN count(n) AS c",
        "MATCH (m:Message) WHERE m.name = 'n020' SET m:Call WITH count(*) AS k \
         MATCH (n) WHERE n:Call RETURN k, count(n) AS c",
        "MATCH (m:Call) WHERE m.name = 'n001' REMOVE m:Call WITH count(*) AS k \
         MATCH (n) WHERE n:Call OR n:Recorded RETURN k, count(n) AS c",
        "MATCH (m:Call) WHERE m.name = 'n002' DETACH DELETE m WITH count(*) AS k \
         MATCH (n) WHERE n:Call RETURN k, count(n) AS c",
        "MATCH (n) WHERE n:Call OR n:Recorded SET n.seen = true RETURN count(n) AS c",
        "MATCH (n) WHERE n:Call AND n.ts > 50 DETACH DELETE n RETURN count(*) AS c",
        "MATCH (m:Message) WHERE m.name = 'n020' SET m:Call WITH count(*) AS k \
         MATCH (d:Device)<-[:ON_DEVICE]-(n) WHERE n:Call RETURN k, count(n) AS c",
        "CREATE (c:Call {name: 'z'})-[:ON_DEVICE]->(:Device {name: 'zd'}) WITH c \
         MATCH (d)<-[:ON_DEVICE]-(c) RETURN d.name AS d",
        "MATCH (d:Device {name: 'device-1'}) CREATE (:Message {name: 'late'})-[:ON_DEVICE]->(d) \
         WITH d MATCH (m)-[:ON_DEVICE]->(d) RETURN count(m) AS c",
    ] {
        let mut outcomes = Vec::new();
        for optimized in [true, false] {
            let (mut writer, _) = fixture().await;
            let catalog = StatsCatalog::from_manifest(&writer.snapshot().manifest().manifest);
            let naive = lower(&parse(query).unwrap()).unwrap();
            let plan = if optimized {
                optimize(naive, &catalog)
            } else {
                naive
            };
            let outcome = namidb_query::execute_write(&plan, &mut writer, &Params::new())
                .await
                .unwrap_or_else(|e| panic!("`{query}` optimized={optimized}: {e:?}"));
            let after = execute(
                &lower(
                    &parse("MATCH (n) RETURN labels(n) AS l, n.name AS v, n.seen AS s").unwrap(),
                )
                .unwrap(),
                &writer.snapshot(),
                &Params::new(),
            )
            .await
            .unwrap();
            outcomes.push((multiset(&outcome.rows), multiset(&after)));
        }
        assert_eq!(outcomes[0], outcomes[1], "`{query}`");
    }
}
