//! Measured heap against the working-memory budget (issue #203).
//!
//! This binary installs a counting global allocator, so it holds a single
//! test: the allocation counters are process-wide and parallel tests would
//! pollute them. It checks the two properties the budget relies on:
//!
//! 1. the estimate is conservative: for a query that fits, the budget's peak
//!    charge is at least the heap the query really allocated at its peak;
//! 2. the bound holds: when the #203 reproduction is stopped by a budget,
//!    the real heap never grew much past that budget.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use namidb_core::id::NamespaceId;
use namidb_storage::{NamespacePaths, WriterSession};
use object_store::memory::InMemory;

use namidb_query::{
    execute_with_limits, lower, parse, with_query_memory, ExecError, Params, QueryMemoryBudget,
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

/// Run `cypher` under `budget` and return (result, heap growth at peak).
async fn measured(
    writer: &WriterSession,
    cypher: &str,
    budget: Arc<QueryMemoryBudget>,
) -> (Result<usize, ExecError>, usize) {
    let snap = writer.snapshot();
    let plan = lower(&parse(cypher).unwrap()).unwrap();
    let params = Params::new();
    let baseline = CURRENT.load(Ordering::Relaxed);
    PEAK.store(baseline, Ordering::Relaxed);
    let result = with_query_memory(
        Some(budget),
        execute_with_limits(&plan, &snap, &params, None, None),
    )
    .await
    .map(|rows| rows.len());
    let grown = PEAK.load(Ordering::Relaxed).saturating_sub(baseline);
    (result, grown)
}

#[tokio::test(flavor = "current_thread")]
async fn budget_estimate_is_conservative_and_bounds_the_real_heap() {
    let paths = NamespacePaths::new("tenants", NamespaceId::new("mem-peak").unwrap());
    let writer = WriterSession::open(Arc::new(InMemory::new()), paths)
        .await
        .unwrap();
    let cypher = |n: usize| format!("UNWIND range(1, {n}) AS i RETURN i, toString(i) AS s");

    // 1. Conservative: a 100k-row result under an ample budget. glibc adds
    //    its own per-block overhead on top of the requested sizes counted
    //    here, which the estimate also covers.
    let budget = Arc::new(QueryMemoryBudget::new(1024 * MIB));
    let (rows, real_peak) = measured(&writer, &cypher(100_000), Arc::clone(&budget)).await;
    assert_eq!(rows.unwrap(), 100_000);
    let estimated_peak = budget.peak_bytes();
    eprintln!("100k rows: estimated peak {estimated_peak} B, real heap peak {real_peak} B");
    assert!(
        estimated_peak >= real_peak,
        "estimate {estimated_peak} below the real peak {real_peak}"
    );
    // ... but not absurdly so, or ordinary queries would be refused.
    assert!(
        estimated_peak <= real_peak.saturating_mul(3),
        "estimate {estimated_peak} more than 3x the real peak {real_peak}"
    );
    drop(budget);

    // 2. Bounded: the #203 reproduction (600k rows) under 64 MiB stops
    //    while the real heap is still close to the budget. Without the
    //    budget this query allocates several hundred MiB.
    let limit = 64 * MIB;
    let budget = Arc::new(QueryMemoryBudget::new(limit));
    let (rows, real_peak) = measured(&writer, &cypher(600_000), budget).await;
    assert!(matches!(rows, Err(ExecError::MemoryLimit(_))), "{rows:?}");
    eprintln!("600k rows under {limit} B: stopped, real heap peak {real_peak} B");
    assert!(
        real_peak <= limit + limit / 4,
        "real heap peaked at {real_peak} bytes under a {limit}-byte budget"
    );
}
