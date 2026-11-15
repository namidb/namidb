//! A parenthesised primary that opens like a node pattern — `(n:A OR n:B)`,
//! `(n {k: v})` — is tried as a pattern and as a parenthesised expression.
//! Nesting such primaries must not multiply the work: every level used to
//! parse its inner text once per alternative, 2^depth in all, so a short
//! query string could pin a CPU core for hours before execution (and its
//! timeout) even began.

use std::sync::mpsc;
use std::time::Duration;

fn parses_quickly(query: String) -> bool {
    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || {
        let _ = namidb_query::parse(&query);
        let _ = tx.send(());
    });
    rx.recv_timeout(Duration::from_secs(20)).is_ok()
}

fn nested(depth: usize, open: &str, close: &str, inner: &str) -> String {
    let mut q = String::from(inner);
    for _ in 0..depth {
        q = format!("{open}{q}{close}");
    }
    q
}

#[test]
fn nested_node_like_parens_parse_quickly() {
    for (open, close, inner) in [
        // Map projection vs node pattern with properties: both parse.
        ("(n {a: ", "})", "1"),
        // Both alternatives fail at the end of every level.
        ("(n {a: ", "}) -", "1"),
        // Pattern with a relationship tail vs arithmetic on a variable.
        ("(n {a: ", "}) - 1", "1"),
        ("(n:A OR ", ")", "n:B"),
        // The pattern attempt reads the nested text, fails after it, and the
        // expression fallback (`n`, then `--(m {…})` as arithmetic) reads it
        // again.
        ("(n)--(m {a: ", "} x)", "1"),
        ("(n)-[:R {a: ", "}]-5", "1"),
        ("(n)<-(m {a: ", "})", "1"),
    ] {
        let query = format!("RETURN {}", nested(120, open, close, inner));
        assert!(
            parses_quickly(query.clone()),
            "parsing did not finish: {query}"
        );
    }
}

#[test]
fn alternating_node_like_parens_parse_quickly() {
    let shapes = [
        ("(n)--(m {a: ", "} x)"),
        ("(n:A OR ", ")"),
        ("(n {a: ", "}) - 1"),
        ("(n)-[:R {a: ", "}]-5"),
        ("(n)<-(m {a: ", "})"),
    ];
    let mut q = String::from("1");
    for i in 0..120 {
        let (open, close) = shapes[i % shapes.len()];
        q = format!("{open}{q}{close}");
    }
    let query = format!("MATCH (n) WHERE {q} RETURN n");
    assert!(
        parses_quickly(query.clone()),
        "parsing did not finish: {query}"
    );
}

#[test]
fn parenthesised_label_predicates_still_parse_as_before() {
    for query in [
        "MATCH (n) WHERE (n:A OR n:B) RETURN n",
        "MATCH (n) WHERE (n:A) RETURN n",
        "MATCH (n) WHERE (n)-[:R]->() RETURN n",
        "MATCH (n) WHERE (n)-[:R]->(:B {x: (1 + 2)}) RETURN n",
        "MATCH (n) WHERE NOT (n:A) AND (n {x: 1}).x = 1 RETURN n",
        "MATCH (n) RETURN (n) - 1 AS x",
        "MATCH (n) WHERE (n:A {x: 1})-[:R]-() RETURN n",
    ] {
        namidb_query::parse(query).unwrap_or_else(|e| panic!("`{query}`: {e:?}"));
    }
}
