//! "Query of death" suite: shapes that used to grow until the process ran
//! out of memory must stop with a typed memory-limit error, promptly, and
//! without the real heap growing far past the query's budget.
//!
//! This binary installs a counting global allocator, so it holds a single
//! test that runs every shape in turn: the counters are process-wide and
//! parallel tests would pollute them.
//!
//! The graph is built once and flushed to SSTs, so the scans read the
//! steady-state (compacted) layout: 40k `:Item` nodes with a 1 KiB `blob`,
//! each linked to the next three (`:LINK`, a dense chain that makes
//! variable-length paths explode) and to one of four `:Hub` nodes
//! (`:ON_HUB`, 10k edges per hub).

use std::alloc::{GlobalAlloc, Layout, System};
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use namidb_core::id::{NamespaceId, NodeId};
use namidb_core::value::Value as CoreValue;
use namidb_storage::{EdgeWriteRecord, NamespacePaths, NodeWriteRecord, WriterSession};
use object_store::memory::InMemory;

use namidb_query::{
    execute_with_limits, lower, optimize, parse, with_query_memory, ExecError, Params,
    QueryMemoryBudget, QueryMemoryScope, StatsCatalog,
};

struct Counting;

static CURRENT: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);

// SAFETY: delegates every call to the system allocator unchanged and only
// updates two atomic counters on the side.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = System.alloc(layout);
        if !ptr.is_null() {
            let now = CURRENT.fetch_add(layout.size(), Ordering::Relaxed) + layout.size();
            PEAK.fetch_max(now, Ordering::Relaxed);
        }
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        System.dealloc(ptr, layout);
        CURRENT.fetch_sub(layout.size(), Ordering::Relaxed);
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let new = System.realloc(ptr, layout, new_size);
        if !new.is_null() {
            if new_size >= layout.size() {
                let grow = new_size - layout.size();
                let now = CURRENT.fetch_add(grow, Ordering::Relaxed) + grow;
                PEAK.fetch_max(now, Ordering::Relaxed);
            } else {
                CURRENT.fetch_sub(layout.size() - new_size, Ordering::Relaxed);
            }
        }
        new
    }
}

#[global_allocator]
static ALLOCATOR: Counting = Counting;

const MIB: usize = 1024 * 1024;
const ITEMS: usize = 40_000;
const BLOB: usize = 1024;
/// Per-query budget every shape runs under.
const BUDGET: usize = 8 * MIB;
/// How far past the budget the real heap may grow: one decoded storage
/// batch (8,192 rows of the widest columns), the meters' flush stride, and
/// allocator slack. The unbounded shapes grow by hundreds of MiB.
const HEAP_SLACK: usize = 40 * MIB;
/// Every shape must stop well within this, on a debug-ish test build.
const PROMPT: Duration = Duration::from_secs(20);

async fn seed() -> WriterSession {
    let paths = NamespacePaths::new("tenants", NamespaceId::new("qod").unwrap());
    let mut writer = WriterSession::open(Arc::new(InMemory::new()), paths)
        .await
        .unwrap();
    let hubs: Vec<NodeId> = (0..4).map(|_| NodeId::new()).collect();
    for (i, hub) in hubs.iter().enumerate() {
        let mut p = BTreeMap::new();
        p.insert("name".into(), CoreValue::Str(format!("hub-{i}")));
        writer
            .upsert_node(
                "Hub",
                *hub,
                &NodeWriteRecord {
                    properties: p,
                    schema_version: 1,
                    ..Default::default()
                },
            )
            .unwrap();
    }
    let items: Vec<NodeId> = (0..ITEMS).map(|_| NodeId::new()).collect();
    for (i, id) in items.iter().enumerate() {
        let mut p = BTreeMap::new();
        p.insert("k".into(), CoreValue::I64(i as i64));
        p.insert(
            "blob".into(),
            CoreValue::Str(format!("{i:0>width$}", width = BLOB)),
        );
        writer
            .upsert_node(
                "Item",
                *id,
                &NodeWriteRecord {
                    properties: p,
                    schema_version: 1,
                    ..Default::default()
                },
            )
            .unwrap();
        let edge = EdgeWriteRecord {
            properties: BTreeMap::new(),
            schema_version: 0,
        };
        for step in 1..=3 {
            writer
                .upsert_edge("LINK", *id, items[(i + step) % ITEMS], &edge)
                .unwrap();
        }
        writer
            .upsert_edge("ON_HUB", *id, hubs[i % hubs.len()], &edge)
            .unwrap();
        if i % 5_000 == 4_999 {
            writer.commit_batch().await.unwrap();
        }
    }
    writer.commit_batch().await.unwrap();
    let schema = writer.snapshot().manifest().manifest.schema.clone();
    writer.flush(schema).await.unwrap();
    writer
}

struct Outcome {
    result: Result<usize, ExecError>,
    heap_growth: usize,
    elapsed: Duration,
}

async fn run(writer: &WriterSession, cypher: &str) -> Outcome {
    let snap = writer.snapshot();
    let catalog = StatsCatalog::from_manifest(&snap.manifest().manifest);
    let plan = optimize(lower(&parse(cypher).unwrap()).unwrap(), &catalog);
    let params = Params::new();
    let budget = Arc::new(QueryMemoryBudget::new(BUDGET));
    let baseline = CURRENT.load(Ordering::Relaxed);
    PEAK.store(baseline, Ordering::Relaxed);
    let started = Instant::now();
    let result = with_query_memory(
        Some(budget),
        execute_with_limits(&plan, &snap, &params, None, None),
    )
    .await
    .map(|rows| rows.len());
    let elapsed = started.elapsed();
    let heap_growth = PEAK.load(Ordering::Relaxed).saturating_sub(baseline);
    Outcome {
        result,
        heap_growth,
        elapsed,
    }
}

#[tokio::test(flavor = "current_thread")]
async fn queries_of_death_stop_at_the_budget_with_bounded_heap() {
    let writer = seed().await;
    // Every shape below needs far more than the budget.
    let shapes = [
        // Scans that materialise every node (the label-less one reads every
        // SST in full; the label one goes through the label postings).
        "MATCH (n) RETURN n",
        "MATCH (n:Item) RETURN n",
        "MATCH (n) WHERE n:Item RETURN count(n)",
        // Cartesian product.
        "MATCH (a:Item), (b:Item) RETURN a.k, b.k",
        // Unbounded variable-length paths over a dense chain.
        "MATCH (a:Item)-[:LINK*]->(b:Item) RETURN a.k, b.k",
        // collect() of everything.
        "MATCH (n:Item) RETURN collect(n) AS all",
        // ORDER BY over everything.
        "MATCH (n:Item) RETURN n ORDER BY n.blob DESC",
        // DISTINCT and grouping over everything.
        "MATCH (n:Item) RETURN DISTINCT n.blob AS b",
        "MATCH (n:Item) RETURN n.blob AS b, count(*) AS c",
        // Huge UNWIND and a huge list value.
        "UNWIND range(1, 100000000) AS i RETURN i",
        "RETURN size(range(1, 100000000)) AS s",
        // Expansion into a hub (10k in-edges each) for every hub row,
        // carrying the wide node through.
        "MATCH (h:Hub)<-[:ON_HUB]-(i:Item) RETURN h, i",
    ];
    let mut failures = Vec::new();
    for cypher in shapes {
        let outcome = run(&writer, cypher).await;
        let line = format!(
            "{cypher}: {:?}, heap +{} MiB, {:?}",
            outcome.result.as_ref().map_err(|e| e.to_string()),
            outcome.heap_growth / MIB,
            outcome.elapsed
        );
        eprintln!("{line}");
        let typed = matches!(
            &outcome.result,
            Err(ExecError::MemoryLimit(exceeded))
                if exceeded.scope == QueryMemoryScope::Query && exceeded.limit_bytes == BUDGET
        );
        if !typed || outcome.heap_growth > BUDGET + HEAP_SLACK || outcome.elapsed > PROMPT {
            failures.push(line);
        }
    }
    assert!(
        failures.is_empty(),
        "unbounded shapes:\n{}",
        failures.join("\n")
    );

    // Queries that fit still answer under the same budget.
    let fits = run(&writer, "MATCH (n:Item) WHERE n.k < 100 RETURN n.k").await;
    assert_eq!(fits.result.unwrap(), 100);
    let count = run(&writer, "MATCH (n:Item) RETURN count(n) AS c").await;
    assert_eq!(count.result.unwrap(), 1);
}
