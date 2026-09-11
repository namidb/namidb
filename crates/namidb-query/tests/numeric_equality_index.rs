//! Numeric equality must be posting-indexed, and must agree with the scan.
//!
//! Before this suite, `{txt: '12345'}` served from the equality sidecar in
//! ~1 ms while `{num: 12345}` over the same 160k rows scanned the label for
//! ~1.6 s — 1600x, purely because the property was numeric. The index was
//! declared on both, and EXPLAIN said so in as many words: "numeric equality
//! is not posting-indexed; only String/Bool are".
//!
//! Two halves, and BOTH are load-bearing:
//!
//!   1. EQUIVALENCE. An indexed numeric equality must return exactly what an
//!      unindexed twin returns. The scan route is already correct, so this
//!      half passes trivially before the change; its job is to guard the
//!      *new* index route, which is where a silent wrong answer would come
//!      from. The cases attack the canonical key: `5` and `5.0` are one
//!      Cypher value and must share a posting, a string `'5'` must never
//!      match a number, and two i64 beyond 2^53 that round to the same f64
//!      must still be separated by the confirm step.
//!
//!   2. NON-INERT. Equal results also happen when the "optimization" never
//!      fires — item 74 shipped one that passed every test and changed no
//!      timing. So the route counter must show the property lookup taking
//!      the native path, after flush AND after compaction. Compaction
//!      re-derives sidecars from scratch; a flush-only harvester change
//!      would vanish at the first merge.

use std::collections::BTreeMap;
use std::sync::Arc;

use namidb_core::id::{NamespaceId, NodeId};
use namidb_core::value::Value as CoreValue;
use namidb_storage::{route_telemetry, NamespacePaths, NodeWriteRecord, WriterSession};
use object_store::memory::InMemory;
use object_store::ObjectStore;

use namidb_query::{
    execute, lower, optimize, parse, LogicalPlan, Params, Row, RuntimeValue, StatsCatalog,
};

/// Two i64 that share an f64: both round to 2^53. A canonical numeric key
/// therefore hands the confirm step a false positive, and only an exact
/// typed comparison can drop it.
const BIG_LOW: i64 = 9_007_199_254_740_992; // 2^53
const BIG_HIGH: i64 = 9_007_199_254_740_993; // 2^53 + 1, not representable

const ROWS: i64 = 600;

fn thing(num: i64, fnum: f64, txt: &str) -> NodeWriteRecord {
    NodeWriteRecord {
        properties: BTreeMap::from([
            // `num`/`fnum` get an index; `unum`/`ufnum` carry the SAME values
            // with no index, so their spelling is always served by the scan
            // whatever the planner does with the indexed one.
            ("num".to_string(), CoreValue::I64(num)),
            ("fnum".to_string(), CoreValue::F64(fnum)),
            ("unum".to_string(), CoreValue::I64(num)),
            ("ufnum".to_string(), CoreValue::F64(fnum)),
            ("txt".to_string(), CoreValue::Str(txt.to_string())),
        ]),
        schema_version: 1,
        ..Default::default()
    }
}

async fn corpus(name: &str) -> WriterSession {
    let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
    let paths = NamespacePaths::new("tenants", NamespaceId::new(name).unwrap());
    let mut writer = WriterSession::open(store, paths).await.unwrap();

    for ordinal in 0..ROWS {
        writer
            .upsert_node(
                "T",
                NodeId::new(),
                &thing(ordinal, ordinal as f64 + 0.5, &ordinal.to_string()),
            )
            .unwrap();
    }
    // Hand-picked adversarial values.
    for (num, fnum, txt) in [
        // An integral float beside an equal integer: `5` and `5.0` are the
        // same Cypher value and must land in the same posting.
        (5_i64, 5.0_f64, "five-integral"),
        (-7, -7.0, "negative"),
        (0, -0.0, "negative-zero"),
        (BIG_LOW, BIG_LOW as f64, "big-low"),
        (BIG_HIGH, BIG_HIGH as f64, "big-high"),
    ] {
        writer
            .upsert_node("T", NodeId::new(), &thing(num, fnum, txt))
            .unwrap();
    }
    writer.commit_batch().await.unwrap();

    // Real DDL, exactly as the server routes it: the type is inferred from
    // the first live value, so `num` is declared Int64 and `fnum` Float64.
    writer.create_property_index("T", "num").await.unwrap();
    writer.create_property_index("T", "fnum").await.unwrap();
    writer
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

/// Rows as an order-insensitive multiset, so the two spellings may legally
/// differ in row order.
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

async fn run(writer: &WriterSession, query: &str) -> Vec<String> {
    let snapshot = writer.snapshot();
    run_on(&snapshot, query).await
}

/// The same query against the TRANSACTIONAL overlay a write statement reads
/// through. Its candidate map keys off `unique_index::key_part`, which keeps
/// `I64(5)` and `F64(5.0)` in distinct variants — so an index route taken
/// there would answer `{num: 5.0}` with an authoritative empty result while
/// the scan returns the `I64(5)` rows. Storage declines the index in this
/// scope for exactly that reason, and this stage is what proves it.
async fn run_staged(writer: &WriterSession, query: &str) -> Vec<String> {
    let snapshot = writer.transactional_overlay_snapshot();
    run_on(&snapshot, query).await
}

async fn run_on(snapshot: &namidb_storage::Snapshot<'_>, query: &str) -> Vec<String> {
    let catalog = StatsCatalog::from_manifest(&snapshot.manifest().manifest);
    let parsed = parse(query).unwrap_or_else(|e| panic!("parse `{query}`: {e:?}"));
    let plan = optimize(
        lower(&parsed).unwrap_or_else(|e| panic!("lower `{query}`: {e:?}")),
        &catalog,
    );
    let rows = execute(&plan, snapshot, &Params::new())
        .await
        .unwrap_or_else(|e| panic!("execute `{query}`: {e:?}"));
    multiset(&rows)
}

/// `(name, indexed spelling, unindexed twin)`.
fn cases() -> Vec<(String, String, String)> {
    let mut out: Vec<(String, String, String)> = Vec::new();
    let mut pair = |name: &str, indexed: String, unindexed: String| {
        out.push((name.to_string(), indexed, unindexed));
    };

    pair(
        "int literal",
        "MATCH (t:T {num: 5}) RETURN t.txt AS txt".into(),
        "MATCH (t:T) WHERE t.unum = 5 RETURN t.txt AS txt".into(),
    );
    pair(
        "integral float literal must find the integer rows",
        "MATCH (t:T {num: 5.0}) RETURN t.txt AS txt".into(),
        "MATCH (t:T) WHERE t.unum = 5.0 RETURN t.txt AS txt".into(),
    );
    pair(
        "float property, int literal",
        "MATCH (t:T {fnum: 5}) RETURN t.txt AS txt".into(),
        "MATCH (t:T) WHERE t.ufnum = 5 RETURN t.txt AS txt".into(),
    );
    pair(
        "float property, float literal",
        "MATCH (t:T {fnum: 5.0}) RETURN t.txt AS txt".into(),
        "MATCH (t:T) WHERE t.ufnum = 5.0 RETURN t.txt AS txt".into(),
    );
    pair(
        "fractional float",
        "MATCH (t:T {fnum: 10.5}) RETURN t.txt AS txt".into(),
        "MATCH (t:T) WHERE t.ufnum = 10.5 RETURN t.txt AS txt".into(),
    );
    pair(
        "negative int",
        "MATCH (t:T {num: -7}) RETURN t.txt AS txt".into(),
        "MATCH (t:T) WHERE t.unum = -7 RETURN t.txt AS txt".into(),
    );
    pair(
        "zero finds the -0.0 row too",
        "MATCH (t:T {fnum: 0.0}) RETURN t.txt AS txt".into(),
        "MATCH (t:T) WHERE t.ufnum = 0.0 RETURN t.txt AS txt".into(),
    );
    pair(
        "a string never equals a number",
        "MATCH (t:T {num: '5'}) RETURN t.txt AS txt".into(),
        "MATCH (t:T) WHERE t.unum = '5' RETURN t.txt AS txt".into(),
    );
    pair(
        "no such value",
        "MATCH (t:T {num: 999999}) RETURN t.txt AS txt".into(),
        "MATCH (t:T) WHERE t.unum = 999999 RETURN t.txt AS txt".into(),
    );
    pair(
        "null never equals",
        "MATCH (t:T) WHERE t.num = null RETURN t.txt AS txt".into(),
        "MATCH (t:T) WHERE t.unum = null RETURN t.txt AS txt".into(),
    );
    pair(
        "an int in a list predicate",
        "MATCH (t:T) WHERE t.num IN [5, -7] RETURN t.txt AS txt".into(),
        "MATCH (t:T) WHERE t.unum IN [5, -7] RETURN t.txt AS txt".into(),
    );
    // Beyond 2^53 the canonical key is lossy: these two share a posting, and
    // each spelling must still return only its own row.
    pair(
        "i64 beyond 2^53, low",
        format!("MATCH (t:T {{num: {BIG_LOW}}}) RETURN t.txt AS txt"),
        format!("MATCH (t:T) WHERE t.unum = {BIG_LOW} RETURN t.txt AS txt"),
    );
    pair(
        "i64 beyond 2^53, high",
        format!("MATCH (t:T {{num: {BIG_HIGH}}}) RETURN t.txt AS txt"),
        format!("MATCH (t:T) WHERE t.unum = {BIG_HIGH} RETURN t.txt AS txt"),
    );
    out
}

/// The PLAN must anchor on the index, not merely consult it per row from
/// inside a label scan. A scan that calls the index once per row still pays
/// scan prices — which is the whole 1600x — and result parity cannot see the
/// difference.
fn assert_plans_as_index_lookup(writer: &WriterSession, query: &str) {
    fn has_lookup(plan: &LogicalPlan) -> bool {
        matches!(plan, LogicalPlan::NodeByPropertyValue { .. })
            || plan.children().into_iter().any(has_lookup)
    }
    fn has_scan(plan: &LogicalPlan) -> bool {
        matches!(plan, LogicalPlan::NodeScan { .. }) || plan.children().into_iter().any(has_scan)
    }
    let snapshot = writer.snapshot();
    let catalog = StatsCatalog::from_manifest(&snapshot.manifest().manifest);
    let plan = optimize(lower(&parse(query).unwrap()).unwrap(), &catalog);
    assert!(
        has_lookup(&plan) && !has_scan(&plan),
        "`{query}` must plan as an index lookup, not a scan:\n{plan:#?}"
    );
}

async fn assert_all_agree(writer: &WriterSession, stage: &str) {
    for (name, indexed, unindexed) in cases() {
        let got = run(writer, &indexed).await;
        let expected = run(writer, &unindexed).await;
        assert!(
            got == expected,
            "\n[{stage}] {name}: indexed and unindexed numeric equality disagree.\n  \
             {indexed}\n    -> {got:?}\n  {unindexed}\n    -> {expected:?}\n"
        );

        // …and the same through the transactional overlay, which a write
        // statement reads through and which encodes its candidate keys
        // differently.
        let staged = run_staged(writer, &indexed).await;
        assert!(
            staged == expected,
            "\n[{stage}/staged] {name}: the transactional overlay disagrees \
             with the scan.\n  {indexed}\n    -> {staged:?}\n  {unindexed}\n    \
             -> {expected:?}\n"
        );
    }
}

/// The lookup must be served by the posting index, not by an O(label) scan
/// that happens to return the same rows.
async fn assert_native_route(writer: &WriterSession, stage: &str) {
    let before = route_telemetry::snapshot();
    let five = run(writer, "MATCH (t:T {num: 5}) RETURN t.txt AS txt").await;
    let after = route_telemetry::snapshot();
    assert!(
        after.property_native > before.property_native,
        "[{stage}] numeric equality took the scan, not the posting index \
         (native {} -> {}, fallback {} -> {})",
        before.property_native,
        after.property_native,
        before.property_fallback,
        after.property_fallback,
    );
    // …and served the right rows through that route: `5` and `5.0` are one
    // value, so both rows must come back.
    assert!(
        five.contains(&"txt=5".to_string()) && five.contains(&"txt=five-integral".to_string()),
        "[{stage}] `num: 5` must find both I64(5) rows: {five:?}"
    );
}

/// The field measurement, reproduced: 160k rows carrying the same key as a
/// string and as an integer, an index declared on both. Run explicitly:
///
///   cargo test -p namidb-query --test numeric_equality_index --release \
///     -- --ignored --nocapture
#[tokio::test]
#[ignore = "measurement, not a gate: builds a 160k-row corpus"]
async fn numeric_equality_is_not_orders_of_magnitude_slower_than_string() {
    const N: i64 = 160_000;
    let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
    let paths = NamespacePaths::new("tenants", NamespaceId::new("numeric-eq-bench").unwrap());
    let mut writer = WriterSession::open(store, paths).await.unwrap();
    for ordinal in 0..N {
        writer
            .upsert_node(
                "V",
                NodeId::new(),
                &NodeWriteRecord {
                    properties: BTreeMap::from([
                        ("num".to_string(), CoreValue::I64(ordinal)),
                        // Same values, no index: the route numeric equality
                        // was forced onto before this change, measured in the
                        // same process so the factor is honest.
                        ("unum".to_string(), CoreValue::I64(ordinal)),
                        ("txt".to_string(), CoreValue::Str(ordinal.to_string())),
                    ]),
                    schema_version: 1,
                    ..Default::default()
                },
            )
            .unwrap();
        if ordinal % 20_000 == 0 {
            writer.commit_batch().await.unwrap();
        }
    }
    writer.commit_batch().await.unwrap();
    writer.create_property_index("V", "num").await.unwrap();
    writer.create_property_index("V", "txt").await.unwrap();
    let committed = writer.snapshot().manifest().manifest.schema.clone();
    writer.flush(committed.clone()).await.unwrap();
    writer.compact_l0(&committed).await.unwrap();

    async fn timed(writer: &WriterSession, query: &str) -> std::time::Duration {
        let _ = run(writer, query).await; // warm the sidecar cache
        let start = std::time::Instant::now();
        let rows = run(writer, query).await;
        let elapsed = start.elapsed();
        assert_eq!(rows.len(), 1, "`{query}` must find exactly its one row");
        elapsed
    }

    let string = timed(&writer, "MATCH (v:V {txt: '12345'}) RETURN v.txt AS t").await;
    let numeric = timed(&writer, "MATCH (v:V {num: 12345}) RETURN v.txt AS t").await;
    let scan = timed(
        &writer,
        "MATCH (v:V) WHERE v.unum = 12345 RETURN v.txt AS t",
    )
    .await;
    println!(
        "string  {string:?}\nnumeric {numeric:?}\nscan    {scan:?}  ({:.0}x)",
        scan.as_secs_f64() / numeric.as_secs_f64()
    );
    // The field report measured 1600x. Numeric equality must now be within
    // the same order of magnitude as its string twin on identical data.
    assert!(
        numeric < string * 20 + std::time::Duration::from_millis(50),
        "numeric equality is still far slower than the string twin on \
         identical data: string {string:?} vs numeric {numeric:?}"
    );
}

#[tokio::test]
async fn numeric_equality_matches_the_scan_and_uses_the_index() {
    let mut writer = corpus("numeric-eq").await;

    assert_all_agree(&writer, "memtable").await;

    let committed = writer.snapshot().manifest().manifest.schema.clone();
    writer.flush(committed.clone()).await.unwrap();
    assert_all_agree(&writer, "flushed").await;
    assert_native_route(&writer, "flushed").await;
    assert_plans_as_index_lookup(&writer, "MATCH (t:T {num: 5}) RETURN t.txt AS txt");
    assert_plans_as_index_lookup(&writer, "MATCH (t:T {fnum: 10.5}) RETURN t.txt AS txt");
    assert_plans_as_index_lookup(&writer, "MATCH (t:T) WHERE t.num = 5 RETURN t.txt AS txt");

    writer.compact_l0(&committed).await.unwrap();
    assert_all_agree(&writer, "compacted").await;
    assert_native_route(&writer, "compacted").await;
}
