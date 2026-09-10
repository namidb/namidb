//! Differential correctness: query rewrites that MUST return identical rows.
//!
//! Four silent wrong-answer bugs surfaced in this engine on 2026-09-10 —
//! an anonymous intermediate node re-anchoring the next hop, quantifiers over
//! a target's list property reading an empty stub, a `*0..n` pattern losing
//! its own source, and `RETURN` columns reported in the wrong order. None of
//! them raised an error. Every one of them returned plausible rows.
//!
//! The suite could not catch that class because it asserts each query against
//! a hand-written expectation: if the expectation is written from observed
//! behaviour, it locks the bug in. This file asserts something the engine
//! cannot satisfy by being consistently wrong — that two SPELLINGS of the
//! same question agree with each other.
//!
//! Add a family here whenever a rewrite is supposed to be meaning-preserving.

use std::collections::BTreeMap;
use std::sync::Arc;

use namidb_core::id::{NamespaceId, NodeId};
use namidb_core::schema::{DataType, EdgeTypeDef, LabelDef, PropertyDef, SchemaBuilder};
use namidb_core::value::Value as CoreValue;
use namidb_storage::{EdgeWriteRecord, NamespacePaths, NodeWriteRecord, WriterSession};
use object_store::memory::InMemory;
use object_store::ObjectStore;

use namidb_query::{execute, lower, optimize, parse, Params, Row, RuntimeValue, StatsCatalog};

fn node(props: Vec<(&str, CoreValue)>) -> NodeWriteRecord {
    NodeWriteRecord {
        properties: props
            .into_iter()
            .map(|(k, v)| (k.to_string(), v))
            .collect::<BTreeMap<_, _>>(),
        schema_version: 0,
        labels: vec![],
    }
}

fn s(v: &str) -> CoreValue {
    CoreValue::Str(v.into())
}

/// `(p1)-[:E]->(q1:Q:S)-[:E]->(r1)`, `(p1)-[:E]->(q2)-[:E]->(r2)`,
/// `(p2)-[:E]->(q3)`. `q*` carry a `tags` list.
async fn fixture() -> WriterSession {
    let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
    let paths = NamespacePaths::new("tenants", NamespaceId::new("rewrite-equiv").unwrap());
    let mut w = WriterSession::open(store, paths).await.unwrap();
    let schema = SchemaBuilder::new()
        .label(LabelDef {
            name: "P".into(),
            properties: vec![PropertyDef::new("n", DataType::Utf8, false).unwrap()],
        })
        .unwrap()
        .label(LabelDef {
            name: "Q".into(),
            properties: vec![PropertyDef::new("n", DataType::Utf8, false).unwrap()],
        })
        .unwrap()
        .label(LabelDef {
            name: "R".into(),
            properties: vec![PropertyDef::new("n", DataType::Utf8, false).unwrap()],
        })
        .unwrap()
        .edge_type(EdgeTypeDef {
            name: "E".into(),
            src_label: "P".into(),
            dst_label: "Q".into(),
            properties: vec![],
        })
        .unwrap()
        .build();

    let p1 = NodeId::new();
    let p2 = NodeId::new();
    w.upsert_node("P", p1, &node(vec![("n", s("p1"))])).unwrap();
    w.upsert_node("P", p2, &node(vec![("n", s("p2"))])).unwrap();

    // q1 carries a SECOND label, which is what a `*0..n` source must not lose.
    let q1 = NodeId::new();
    w.upsert_node_with_labels(
        ["Q".to_string(), "S".to_string()],
        q1,
        &node(vec![
            ("n", s("q1")),
            ("tags", CoreValue::List(vec![s("t1"), s("t2")])),
        ]),
    )
    .unwrap();
    let q2 = NodeId::new();
    w.upsert_node(
        "Q",
        q2,
        &node(vec![
            ("n", s("q2")),
            ("tags", CoreValue::List(vec![s("t3")])),
        ]),
    )
    .unwrap();
    let q3 = NodeId::new();
    w.upsert_node(
        "Q",
        q3,
        &node(vec![
            ("n", s("q3")),
            ("tags", CoreValue::List(vec![s("t1")])),
        ]),
    )
    .unwrap();

    // Numeric properties of BOTH families, so a filter mixing Int and Float
    // has something to be wrong about. `amount` is a float and `units` an
    // integer — how money and quantities are actually stored.
    let n1 = NodeId::new();
    w.upsert_node(
        "NUM",
        n1,
        &node(vec![
            ("amount", CoreValue::F64(12.5)),
            ("units", CoreValue::I64(3)),
        ]),
    )
    .unwrap();
    let n2 = NodeId::new();
    w.upsert_node(
        "NUM",
        n2,
        &node(vec![
            ("amount", CoreValue::F64(0.0)),
            ("units", CoreValue::I64(0)),
        ]),
    )
    .unwrap();

    let r1 = NodeId::new();
    let r2 = NodeId::new();
    w.upsert_node("R", r1, &node(vec![("n", s("r1"))])).unwrap();
    w.upsert_node("R", r2, &node(vec![("n", s("r2"))])).unwrap();

    for (a, b) in [(p1, q1), (p1, q2), (p2, q3), (q1, r1), (q2, r2)] {
        w.upsert_edge("E", a, b, &EdgeWriteRecord::default())
            .unwrap();
    }
    w.commit_batch().await.unwrap();
    w.flush(schema).await.unwrap();
    w
}

/// Rows as an order-insensitive multiset of `(column, value)` maps, so a
/// family can compare spellings whose row ORDER legitimately differs.
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
        other => format!("{other:?}"),
    }
}

/// Assert every spelling in `family` returns the same multiset of rows.
/// Runs through the FULL optimizer, as the server does.
async fn assert_equivalent(writer: &WriterSession, name: &str, family: &[&str]) {
    let snapshot = writer.snapshot();
    let catalog = StatsCatalog::from_manifest(&snapshot.manifest().manifest);
    let mut baseline: Option<(String, Vec<String>)> = None;
    for query in family {
        let parsed = parse(query).unwrap_or_else(|e| panic!("{name}: parse `{query}`: {e:?}"));
        let plan = optimize(
            lower(&parsed).unwrap_or_else(|e| panic!("{name}: lower `{query}`: {e:?}")),
            &catalog,
        );
        let rows = execute(&plan, &snapshot, &Params::new())
            .await
            .unwrap_or_else(|e| panic!("{name}: execute `{query}`: {e:?}"));
        let got = multiset(&rows);
        match &baseline {
            None => baseline = Some(((*query).to_string(), got)),
            Some((first, expected)) => assert!(
                &got == expected,
                "\n{name}: these must agree but do not.\n  \
                 {first}\n    -> {expected:?}\n  {query}\n    -> {got:?}\n"
            ),
        }
    }
}

#[tokio::test]
async fn meaning_preserving_rewrites_agree() {
    let w = fixture().await;

    assert_equivalent(
        &w,
        "inline label vs labels() predicate",
        &[
            "MATCH (a:P)-[:E]->(b:Q) RETURN b.n AS n",
            "MATCH (a:P)-[:E]->(b) WHERE 'Q' IN labels(b) RETURN b.n AS n",
        ],
    )
    .await;

    // The 2.6.5 bug: an anonymous intermediate re-anchored the next hop.
    assert_equivalent(
        &w,
        "anonymous vs named intermediate",
        &[
            "MATCH (a:P)-[:E]->(m:Q)-[:E]->(z:R) RETURN z.n AS n",
            "MATCH (a:P)-[:E]->()-[:E]->(z:R) RETURN z.n AS n",
            "MATCH (a:P)-[:E*2..2]->(z:R) RETURN z.n AS n",
        ],
    )
    .await;

    assert_equivalent(
        &w,
        "direction reversal",
        &[
            "MATCH (a:P)-[:E]->(b:Q) RETURN b.n AS n",
            "MATCH (b:Q)<-[:E]-(a:P) RETURN b.n AS n",
        ],
    )
    .await;

    assert_equivalent(
        &w,
        "inline property vs WHERE vs WITH-WHERE",
        &[
            "MATCH (a:P)-[:E]->(b:Q {n: 'q1'}) RETURN b.n AS n",
            "MATCH (a:P)-[:E]->(b:Q) WHERE b.n = 'q1' RETURN b.n AS n",
            "MATCH (a:P)-[:E]->(b:Q) WITH b WHERE b.n = 'q1' RETURN b.n AS n",
        ],
    )
    .await;

    // The 2.6.8 bug: the target was bound as an empty stub, so `b.tags` was
    // null and the quantifier matched nothing.
    //
    // The alias must appear ONLY inside the quantifier. Returning `b.n` as
    // well registers `b` by that reference and suppresses the stub entirely —
    // a first draft of this family did exactly that and passed with the fix
    // REVERTED, which is worse than having no family at all.
    assert_equivalent(
        &w,
        "quantifier vs IN, target referenced only inside the predicate",
        &[
            "MATCH (a:P)-[:E]->(b:Q) WHERE any(x IN b.tags WHERE x = 't1') RETURN count(*) AS c",
            "MATCH (a:P)-[:E]->(b:Q) WHERE 't1' IN b.tags RETURN count(*) AS c",
        ],
    )
    .await;
    assert_equivalent(
        &w,
        "all()/none() with the target referenced only in the predicate",
        &[
            "MATCH (a:P)-[:E]->(b:Q) WHERE none(x IN b.tags WHERE x = 't1') RETURN count(*) AS c",
            "MATCH (a:P)-[:E]->(b:Q) WHERE NOT ('t1' IN b.tags) RETURN count(*) AS c",
        ],
    )
    .await;
    assert_equivalent(
        &w,
        "list comprehension with the target referenced only in the predicate",
        &[
            "MATCH (a:P)-[:E]->(b:Q) WHERE size([x IN b.tags WHERE x <> 'zz']) > 0 \
             RETURN count(*) AS c",
            "MATCH (a:P)-[:E]->(b:Q) WHERE size(b.tags) > 0 RETURN count(*) AS c",
        ],
    )
    .await;

    assert_equivalent(
        &w,
        "count(*) vs size(collect())",
        &[
            "MATCH (a:P)-[:E]->(b:Q) RETURN count(*) AS c",
            "MATCH (a:P)-[:E]->(b:Q) WITH collect(b) AS bs RETURN size(bs) AS c",
        ],
    )
    .await;

    assert_equivalent(
        &w,
        "count(DISTINCT x) vs WITH DISTINCT",
        &[
            "MATCH (a:P)-[:E]->(b:Q) RETURN count(DISTINCT b) AS c",
            "MATCH (a:P)-[:E]->(b:Q) WITH DISTINCT b RETURN count(*) AS c",
        ],
    )
    .await;

    assert_equivalent(
        &w,
        "single hop vs *1..1",
        &[
            "MATCH (a:P)-[:E]->(b:Q) RETURN b.n AS n",
            "MATCH (a:P)-[:E*1..1]->(b:Q) RETURN b.n AS n",
        ],
    )
    .await;

    // The 2.6.8 bug: a `*0..n` source bound as a stub lost its second label.
    // Same trap here: `q` must not be referenced outside the `*0..0` pattern,
    // or the stub it depends on is never built.
    assert_equivalent(
        &w,
        "zero-hop source keeps its own labels",
        &[
            "MATCH (a:P)-[:E]->(q:Q)-[:E*0..0]->(z:S) RETURN count(*) AS c",
            "MATCH (a:P)-[:E]->(q:Q) WHERE 'S' IN labels(q) RETURN count(*) AS c",
        ],
    )
    .await;

    // A type alternation is a SET: writing the same type twice describes the
    // same edges, so it must return the same rows. It used to return each row
    // once per listed type.
    assert_equivalent(
        &w,
        "repeated type in an alternation",
        &[
            "MATCH (a:P)-[:E]->(b:Q) RETURN b.n AS n",
            "MATCH (a:P)-[:E|E]->(b:Q) RETURN b.n AS n",
            "MATCH (a:P)-[:E|E|E]->(b:Q) RETURN b.n AS n",
        ],
    )
    .await;

    // The identity-only whitelist: an alias referenced ONLY through
    // `count(x)` / `count(DISTINCT x)` / `count(id(x))` may be bound as an
    // id-only stub. These must all agree with the unreferenced spelling.
    assert_equivalent(
        &w,
        "identity-only counts agree with count(*)",
        &[
            "MATCH (a:P)-[:E]->(b:Q) RETURN count(*) AS c",
            "MATCH (a:P)-[:E]->(b:Q) RETURN count(b) AS c",
            "MATCH (a:P)-[:E]->(b:Q) RETURN count(id(b)) AS c",
        ],
    )
    .await;

    assert_equivalent(
        &w,
        "identity-only DISTINCT agrees across spellings",
        &[
            "MATCH (a:P)-[:E]->(b:Q) RETURN count(DISTINCT b) AS c",
            "MATCH (a:P)-[:E]->(b:Q) RETURN count(DISTINCT id(b)) AS c",
        ],
    )
    .await;

    // THE DANGEROUS CASE. `b` is counted AND read for a property in the same
    // statement, so it must NOT be classified identity-only. If it were, the
    // stub's empty properties would make `max(b.n)` null while the count
    // stayed right — a half-correct row, which is the worst kind.
    assert_equivalent(
        &w,
        "an alias both counted and value-read stays hydrated",
        &[
            "MATCH (a:P)-[:E]->(b:Q) RETURN count(b) AS c, max(b.n) AS m",
            "MATCH (a:P)-[:E]->(b:Q) RETURN count(*) AS c, max(b.n) AS m",
        ],
    )
    .await;

    // `count(b.n)` counts a PROPERTY, not the node: the argument is not a
    // bare variable, so it must take the full path.
    assert_equivalent(
        &w,
        "count over a property is not an identity reference",
        &[
            "MATCH (a:P)-[:E]->(b:Q) RETURN count(b.n) AS c",
            "MATCH (a:P)-[:E]->(b:Q) WHERE b.n IS NOT NULL RETURN count(*) AS c",
        ],
    )
    .await;

    // A comparison must not depend on whether the literal is spelled as an
    // integer or a float, nor on whether it lands in a FILTER or an
    // EXPRESSION. `WHERE n.amount > 0` over a float column silently returned
    // nothing while `RETURN n.amount > 0` returned true.
    assert_equivalent(
        &w,
        "float property vs integer and float literals",
        &[
            "MATCH (n:NUM) WHERE n.amount > 0 RETURN count(*) AS c",
            "MATCH (n:NUM) WHERE n.amount > 0.0 RETURN count(*) AS c",
        ],
    )
    .await;

    assert_equivalent(
        &w,
        "integer property vs float and integer literals",
        &[
            "MATCH (n:NUM) WHERE n.units >= 1 RETURN count(*) AS c",
            "MATCH (n:NUM) WHERE n.units >= 1.0 RETURN count(*) AS c",
        ],
    )
    .await;

    assert_equivalent(
        &w,
        "equality across numeric families",
        &[
            "MATCH (n:NUM) WHERE n.units = 3 RETURN count(*) AS c",
            "MATCH (n:NUM) WHERE n.units = 3.0 RETURN count(*) AS c",
        ],
    )
    .await;

    // The filter/expression split: the same comparison as a WHERE and as a
    // projected boolean must agree on which rows satisfy it.
    assert_equivalent(
        &w,
        "the same comparison as a filter and as an expression",
        &[
            "MATCH (n:NUM) WHERE n.amount > 0 RETURN count(*) AS c",
            "MATCH (n:NUM) WITH n, (n.amount > 0) AS keep WHERE keep RETURN count(*) AS c",
        ],
    )
    .await;

    assert_equivalent(
        &w,
        "conjunctive multi-label vs labels() predicate",
        &[
            "MATCH (b:Q:S) RETURN b.n AS n",
            "MATCH (b:Q) WHERE 'S' IN labels(b) RETURN b.n AS n",
        ],
    )
    .await;
}
