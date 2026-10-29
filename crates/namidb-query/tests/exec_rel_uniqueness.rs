//! Cypher relationship uniqueness: within one MATCH clause, two
//! relationship variables (named or anonymous, across every comma-separated
//! part) never bind the same relationship. Separate MATCH clauses do not
//! constrain each other. Every expected count below is computed by brute
//! force over the edge list and checked on the flat and factorized
//! executors, with and without the optimizer.

use std::collections::BTreeMap;
use std::sync::Arc;

use namidb_core::id::{NamespaceId, NodeId};
use namidb_core::value::Value as CoreValue;
use namidb_query::{
    execute_factor_path, execute_flat_path, lower, optimize, parse, LogicalPlan, Params,
    RuntimeValue, StatsCatalog,
};
use namidb_storage::{EdgeWriteRecord, NamespacePaths, NodeWriteRecord, WriterSession};
use object_store::memory::InMemory;
use object_store::ObjectStore;

const NODES: usize = 7;

/// Directed `K` edges. No self-loop here, so the undirected counts do not
/// depend on how a self-loop is walked; the directed tests add one.
const EDGES: &[(usize, usize)] = &[
    (0, 1),
    (1, 2),
    (2, 0),
    (2, 3),
    (3, 4),
    (4, 2),
    (1, 0),
    (5, 6),
    (3, 5),
];

async fn graph(name: &str, edges: &[(usize, usize)], flush: bool) -> WriterSession {
    let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
    let paths = NamespacePaths::new("tenants", NamespaceId::new(name).unwrap());
    let mut writer = WriterSession::open(store, paths).await.unwrap();
    let ids: Vec<NodeId> = (0..NODES).map(|_| NodeId::new()).collect();
    for (i, id) in ids.iter().enumerate() {
        let mut props: BTreeMap<String, CoreValue> = BTreeMap::new();
        props.insert("n".into(), CoreValue::I64(i as i64));
        writer
            .upsert_node(
                "P",
                *id,
                &NodeWriteRecord {
                    properties: props,
                    schema_version: 1,
                    ..Default::default()
                },
            )
            .unwrap();
    }
    for &(s, d) in edges {
        writer
            .upsert_edge(
                "K",
                ids[s],
                ids[d],
                &EdgeWriteRecord {
                    properties: BTreeMap::new(),
                    schema_version: 1,
                },
            )
            .unwrap();
    }
    writer.commit_batch().await.unwrap();
    if flush {
        let schema = namidb_core::schema::SchemaBuilder::new().build();
        writer.flush(schema).await.unwrap();
    }
    writer
}

/// Every way to walk one edge: (edge index, from, to).
fn steps(edges: &[(usize, usize)], undirected: bool) -> Vec<(usize, usize, usize)> {
    let mut out = Vec::new();
    for (i, &(s, d)) in edges.iter().enumerate() {
        out.push((i, s, d));
        if undirected && s != d {
            out.push((i, d, s));
        }
    }
    out
}

/// Trails of `len` steps (no edge repeated), as edge index lists plus the
/// start and end node.
fn trails(
    edges: &[(usize, usize)],
    undirected: bool,
    len: usize,
) -> Vec<(Vec<usize>, usize, usize)> {
    let st = steps(edges, undirected);
    let mut out: Vec<(Vec<usize>, usize, usize)> = (0..NODES).map(|n| (vec![], n, n)).collect();
    for _ in 0..len {
        let mut next = Vec::new();
        for (used, start, end) in &out {
            for &(e, f, t) in &st {
                if f == *end && !used.contains(&e) {
                    let mut u = used.clone();
                    u.push(e);
                    next.push((u, *start, t));
                }
            }
        }
        out = next;
    }
    out
}

async fn count_all_ways(writer: &WriterSession, query: &str) -> Vec<usize> {
    let snapshot = writer.snapshot();
    let parsed = parse(query).unwrap();
    let lowered = lower(&parsed).unwrap();
    let optimized = optimize(lowered.clone(), &StatsCatalog::default());
    let mut counts = Vec::new();
    for plan in [&lowered, &optimized] {
        for factor in [false, true] {
            let rows = if factor {
                execute_factor_path(plan, &snapshot, &Params::new()).await
            } else {
                execute_flat_path(plan, &snapshot, &Params::new()).await
            }
            .unwrap_or_else(|e| panic!("{query}: {e}"));
            assert_eq!(rows.len(), 1, "{query}");
            match rows[0].get("c") {
                Some(RuntimeValue::Integer(n)) => counts.push(*n as usize),
                other => panic!("{query}: count was {other:?}"),
            }
        }
    }
    counts
}

async fn check(writer: &WriterSession, query: &str, expected: usize) {
    let counts = count_all_ways(writer, query).await;
    assert!(
        counts.iter().all(|&c| c == expected),
        "{query}: expected {expected} everywhere, got {counts:?} \
         (lowered flat, lowered factor, optimized flat, optimized factor)"
    );
}

#[tokio::test]
async fn fixed_chains_never_reuse_a_relationship() {
    for flush in [false, true] {
        let w = graph(&format!("uniq-chain-{flush}"), EDGES, flush).await;
        let two_undirected = trails(EDGES, true, 2).len();
        let three_undirected = trails(EDGES, true, 3).len();
        check(
            &w,
            "MATCH (a:P)-[:K]-(b:P)-[:K]-(c:P) RETURN count(*) AS c",
            two_undirected,
        )
        .await;
        check(
            &w,
            "MATCH (a)-[r:K]-(b)-[s:K]-(x) RETURN count(*) AS c",
            two_undirected,
        )
        .await;
        check(
            &w,
            "MATCH (a)-[r]-(b)-[]-(x) RETURN count(*) AS c",
            two_undirected,
        )
        .await;
        check(
            &w,
            "MATCH (a)--(b)--(x)--(y) RETURN count(*) AS c",
            three_undirected,
        )
        .await;
        // An undirected 2-hop that ends where it started can only do so
        // over two distinct edges between the same pair: here 0<->1 only.
        let back = trails(EDGES, true, 2)
            .iter()
            .filter(|(_, s, e)| s == e)
            .count();
        assert_eq!(back, 4);
        check(&w, "MATCH (a)-[:K]-(b)-[:K]-(a) RETURN count(*) AS c", back).await;
        // Undirected triangles: closed 3-trails.
        let tri = trails(EDGES, true, 3)
            .iter()
            .filter(|(_, s, e)| s == e)
            .count();
        check(
            &w,
            "MATCH (a)-[:K]-(b)-[:K]-(x)-[:K]-(a) RETURN count(*) AS c",
            tri,
        )
        .await;
    }
}

#[tokio::test]
async fn comma_separated_parts_share_one_uniqueness_scope() {
    let w = graph("uniq-parts", EDGES, false).await;
    let e = EDGES.len();
    check(
        &w,
        "MATCH (a)-[:K]->(b), (x)-[:K]->(y) RETURN count(*) AS c",
        e * (e - 1),
    )
    .await;
    check(
        &w,
        "MATCH (a)-[r:K]->(b), (x)-[s]->(y) RETURN count(*) AS c",
        e * (e - 1),
    )
    .await;
    // Two parts sharing a node, written as separate parts.
    let two_undirected = trails(EDGES, true, 2).len();
    check(
        &w,
        "MATCH (a)-[:K]-(b), (b)-[:K]-(x) RETURN count(*) AS c",
        two_undirected,
    )
    .await;
}

#[tokio::test]
async fn separate_match_clauses_do_not_constrain_each_other() {
    let w = graph("uniq-clauses", EDGES, false).await;
    let st = steps(EDGES, true);
    let walks: usize = st
        .iter()
        .map(|&(_, _, t)| st.iter().filter(|&&(_, f, _)| f == t).count())
        .sum();
    check(
        &w,
        "MATCH (a)-[:K]-(b) MATCH (b)-[:K]-(x) RETURN count(*) AS c",
        walks,
    )
    .await;
    let e = EDGES.len();
    check(
        &w,
        "MATCH (a)-[:K]->(b) MATCH (x)-[:K]->(y) RETURN count(*) AS c",
        e * e,
    )
    .await;
}

#[tokio::test]
async fn variable_length_hops_share_the_scope_with_fixed_hops() {
    let w = graph("uniq-varlen", EDGES, false).await;
    let expected = trails(EDGES, true, 2).len() + trails(EDGES, true, 3).len();
    check(
        &w,
        "MATCH (a)-[:K*1..2]-(b)-[:K]-(x) RETURN count(*) AS c",
        expected,
    )
    .await;
    check(
        &w,
        "MATCH (a)-[:K]-(b)-[:K*1..2]-(x) RETURN count(*) AS c",
        expected,
    )
    .await;
    check(
        &w,
        "MATCH (a)-[rs:K*1..2]-(b)-[r:K]-(x) RETURN count(*) AS c",
        expected,
    )
    .await;
}

#[tokio::test]
async fn directed_chain_through_a_self_loop_uses_it_once() {
    let mut edges = EDGES.to_vec();
    edges.push((6, 6));
    let w = graph("uniq-selfloop", &edges, false).await;
    let expected = trails(&edges, false, 2).len();
    check(
        &w,
        "MATCH (a)-[:K]->(b)-[:K]->(x) RETURN count(*) AS c",
        expected,
    )
    .await;
    // Directed triangle over the self-loop alone would reuse it three times.
    let tri = trails(&edges, false, 3)
        .iter()
        .filter(|(_, s, e)| s == e)
        .count();
    check(
        &w,
        "MATCH (a:P)-[:K]->(b:P)-[:K]->(x:P)-[:K]->(a) RETURN count(*) AS c",
        tri,
    )
    .await;
}

#[tokio::test]
async fn optional_match_enforces_uniqueness_within_its_own_pattern() {
    // Node 6's only edge is 5->6, and 5 has no other outgoing edge, so
    // `(6)<-(5)->(x)` could only go back over that same edge: no match, and
    // OPTIONAL keeps the row with NULLs.
    let w = graph("uniq-optional", EDGES, false).await;
    check(
        &w,
        "MATCH (a:P {n: 6}) OPTIONAL MATCH (a)<-[:K]-(b)-[:K]->(x) \
         WITH a, x WHERE x IS NULL RETURN count(*) AS c",
        1,
    )
    .await;
    // Undirected, the only walk left is 6-5-3.
    check(
        &w,
        "MATCH (a:P {n: 6}) OPTIONAL MATCH (a)-[:K]-(b)-[:K]-(x) RETURN count(x) AS c",
        1,
    )
    .await;
    // The MATCH before it does not constrain the OPTIONAL MATCH.
    check(
        &w,
        "MATCH (a:P {n: 6})-[:K]-(b) OPTIONAL MATCH (b)-[:K]-(x) RETURN count(x) AS c",
        2,
    )
    .await;
}

#[tokio::test]
async fn pattern_predicates_and_comprehensions_apply_uniqueness() {
    let w = graph("uniq-subpattern", EDGES, false).await;
    // Only 0 and 1 get back to themselves over two different edges.
    let back: std::collections::BTreeSet<usize> = trails(EDGES, true, 2)
        .iter()
        .filter(|(_, s, e)| s == e)
        .map(|(_, s, _)| *s)
        .collect();
    assert_eq!(back.len(), 2);
    for q in [
        "MATCH (a:P) WHERE (a)-[:K]-()-[:K]-(a) RETURN count(*) AS c",
        "MATCH (a:P) WHERE EXISTS((a)-[:K]-()-[:K]-(a)) RETURN count(*) AS c",
    ] {
        check(&w, q, back.len()).await;
    }
    // From 6 the only 2-trail is 6-5-3.
    check(
        &w,
        "MATCH (a:P {n: 6}) WITH [(a)-[:K]-(b)-[:K]-(x) | x] AS l RETURN size(l) AS c",
        1,
    )
    .await;
}

#[test]
fn lowering_marks_the_later_hop_against_the_earlier_one() {
    fn expands(plan: &LogicalPlan, out: &mut Vec<(Option<String>, Vec<String>)>) {
        if let LogicalPlan::Expand {
            rel_alias,
            unique_rels,
            ..
        } = plan
        {
            out.push((rel_alias.clone(), unique_rels.clone()));
        }
        for c in plan.children() {
            expands(c, out);
        }
    }
    let plan = lower(&parse("MATCH (a)-[r:K]-(b)-[:K]-(c) RETURN c").unwrap()).unwrap();
    let mut found = Vec::new();
    expands(&plan, &mut found);
    // Top-down: the second hop first.
    assert_eq!(found.len(), 2);
    let hidden = found[0]
        .0
        .clone()
        .expect("anonymous hop gets an internal alias");
    assert_eq!(found[0].1, vec!["r".to_string()]);
    assert_eq!(found[1], (Some("r".into()), vec![]));
    assert!(hidden.starts_with("__anon"));
    // A variable-length hop nothing after it compares against stays
    // unnamed, so the factorized executor keeps it; it still checks the
    // hop before it.
    let plan = lower(&parse("MATCH (a)-[:K]->(b)-[:K*1..3]->(c) RETURN c").unwrap()).unwrap();
    let mut found = Vec::new();
    expands(&plan, &mut found);
    assert_eq!(found.len(), 2);
    let first = found[1].0.clone().expect("the fixed hop is named");
    assert_eq!(found[0], (None, vec![first]));
    // Types that cannot overlap are not compared.
    let plan = lower(&parse("MATCH (a)-[:K]-(b)-[:L]-(c) RETURN c").unwrap()).unwrap();
    let mut found = Vec::new();
    expands(&plan, &mut found);
    assert!(found
        .iter()
        .all(|(alias, uniq)| alias.is_none() && uniq.is_empty()));
}

/// Rows of `query` as `(b, x)` pairs of `n` values, sorted, after checking
/// that all four ways to run it agree.
async fn pairs_all_ways(writer: &WriterSession, query: &str) -> Vec<(Option<i64>, Option<i64>)> {
    let snapshot = writer.snapshot();
    let lowered = lower(&parse(query).unwrap()).unwrap();
    let optimized = optimize(lowered.clone(), &StatsCatalog::default());
    let mut all = Vec::new();
    for plan in [&lowered, &optimized] {
        for factor in [false, true] {
            let rows = if factor {
                execute_factor_path(plan, &snapshot, &Params::new()).await
            } else {
                execute_flat_path(plan, &snapshot, &Params::new()).await
            }
            .unwrap_or_else(|e| panic!("{query}: {e}"));
            let n = |v: Option<&RuntimeValue>| match v {
                Some(RuntimeValue::Integer(n)) => Some(*n),
                Some(RuntimeValue::Null) => None,
                other => panic!("{query}: {other:?}"),
            };
            let mut pairs: Vec<_> = rows
                .iter()
                .map(|r| (n(r.get("b")), n(r.get("x"))))
                .collect();
            pairs.sort();
            all.push(pairs);
        }
    }
    assert!(all.windows(2).all(|w| w[0] == w[1]), "{query}: {all:?}");
    all.pop().unwrap()
}

#[tokio::test]
async fn optional_match_is_all_or_nothing() {
    let w = graph("optional-whole", EDGES, false).await;
    // Into 6 only 5->6, and 5 has no other out-edge: no match, so `b` is
    // NULL too, not the 5 the first hop found.
    assert_eq!(
        pairs_all_ways(
            &w,
            "MATCH (a:P {n: 6}) OPTIONAL MATCH (a)<-[:K]-(b)-[:K]->(x) RETURN b.n AS b, x.n AS x",
        )
        .await,
        vec![(None, None)]
    );
    // Into 2: from 1 (which goes on to 0) and from 4 (whose only out-edge
    // is the one into 2). Only the full match comes back.
    assert_eq!(
        pairs_all_ways(
            &w,
            "MATCH (a:P {n: 2}) OPTIONAL MATCH (a)<-[:K]-(b)-[:K]->(x) RETURN b.n AS b, x.n AS x",
        )
        .await,
        vec![(Some(1), Some(0))]
    );
    // The WHERE belongs to the pattern too.
    assert_eq!(
        pairs_all_ways(
            &w,
            "MATCH (a:P {n: 3}) OPTIONAL MATCH (a)-[:K]->(b)-[:K]->(x) WHERE x.n = 2 \
             RETURN b.n AS b, x.n AS x",
        )
        .await,
        vec![(Some(4), Some(2))]
    );
    // A WHERE or an inline property that rules out every neighbour keeps
    // the row with NULLs instead of dropping it.
    for q in [
        "MATCH (a:P {n: 6}) OPTIONAL MATCH (a)<-[:K]-(b) WHERE b.n = 99 RETURN b.n AS b, a.n AS x",
        "MATCH (a:P {n: 6}) OPTIONAL MATCH (a)<-[:K]-(b {n: 99}) RETURN b.n AS b, a.n AS x",
        "MATCH (a:P {n: 6}) OPTIONAL MATCH (b:Nope) RETURN b AS b, a.n AS x",
    ] {
        assert_eq!(pairs_all_ways(&w, q).await, vec![(None, Some(6))], "{q}");
    }
}
