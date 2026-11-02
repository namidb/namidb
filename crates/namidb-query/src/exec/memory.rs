//! Per-query working-memory budget (issue #203).
//!
//! The executor is eager: every operator hands its parent a complete
//! `Vec<Row>`, so a query's resident working set is the rows its operators
//! have materialised and not yet dropped, plus operator state (sort keys,
//! grouping state, `collect()` lists, DISTINCT fingerprints). The server's
//! process governor (`NAMIDB_MEMORY_MAX_BYTES`) only samples RSS when a query
//! is admitted, so before this module one admitted query could grow until
//! the kernel OOM-killed the process.
//!
//! A [`QueryMemoryBudget`] is scoped on the query's task with
//! [`with_query_memory`]. The executor charges it in two ways:
//!
//! - **Operator boundaries.** `execute_inner_with_routing` records the
//!   charged total when an operator starts. When the operator returns, its
//!   output is estimated and the peak (inputs still charged plus the new
//!   output) is checked against the budget; the operator's inputs are then
//!   released and only its output stays charged until the parent finishes.
//!   This mirrors the eager executor's real lifetimes: a child's rows live
//!   until the parent has consumed them. The factorised executor
//!   (`NAMIDB_FACTORIZE=1`) settles the same way in
//!   `execute_factor_inner_with_routing`, charging each operator's whole
//!   arena, and charges the rows it materialises from a set in place of
//!   the set.
//! - **Growth meters.** The producers that can grow far beyond their input
//!   in a single call (`UNWIND`, `Expand`, projections, joins, cross
//!   products, aggregation and `collect()`, DISTINCT, sorting, `range()`)
//!   estimate rows as they push them and check the budget every
//!   [`METER_FLUSH_BYTES`], so an oversized operator fails mid-flight
//!   instead of after it has allocated everything.
//!
//! Estimates are approximate but deliberately conservative: they count the
//! heap blocks the Rust collections actually allocate (B-tree nodes at
//! minimum fill, string and vector capacities, allocator headers), not the
//! logical payload. Measured against glibc on the repro shape from #203
//! (`UNWIND range(1, n) AS i RETURN i, toString(i) AS s`), the estimate is
//! within a few percent of the observed ~650 bytes per row and never below
//! it.
//!
//! Three limits can stop a query, distinguished by [`QueryMemoryScope`]:
//!
//! - [`QueryMemoryScope::Query`]: the query alone needs more than its own
//!   budget. Deterministic: the same statement fails the same way again.
//! - [`QueryMemoryScope::Shared`]: the process-wide pool that every
//!   concurrent query reserves from ([`QueryMemoryPool`]) is exhausted by
//!   other in-flight queries. Transient.
//! - [`QueryMemoryScope::Process`]: the optional resident-memory probe saw
//!   the process at its configured ceiling while this query was growing.
//!   Transient.
//!
//! Nothing here runs unless a budget is scoped: tests, embedded callers and
//! the CLI keep their existing cost.

use std::fmt;
use std::future::Future;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;

use super::factor::{FactorArena, FactorIdx, FactorNode, FactorRowSet, Slot};
use super::row::Row;
use super::value::{NodeValue, RelValue, RuntimeValue};
use super::walker::ExecError;

/// How often (in estimated bytes) a growth meter re-checks the budget. Small
/// enough that an operator overshoots its budget by at most this much,
/// large enough that the check costs nothing next to building the rows.
pub const METER_FLUSH_BYTES: usize = 256 * 1024;

/// Pool reservations are made in whole chunks so concurrent queries do not
/// contend on the shared counter for every row.
const POOL_CHUNK_BYTES: usize = 1024 * 1024;

/// How much a query's charged total must grow between two resident-memory
/// probes. A probe reads procfs, which is cheap but not free.
const PROBE_INTERVAL_BYTES: usize = 16 * 1024 * 1024;

/// Which limit a query ran into. See the module docs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QueryMemoryScope {
    /// The per-query budget (`NAMIDB_QUERY_MEMORY_MAX_BYTES`).
    Query,
    /// The process-wide pool shared by all in-flight queries.
    Shared,
    /// The process resident set reached the configured memory ceiling.
    Process,
}

/// A query stopped because it needed more working memory than a limit
/// allows. Carried by [`ExecError::MemoryLimit`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QueryMemoryExceeded {
    /// Which limit was hit.
    pub scope: QueryMemoryScope,
    /// For `Query` and `Shared`: the estimated working set the query needed
    /// at the moment it was stopped. For `Process`: the sampled resident
    /// bytes of the whole process.
    pub required_bytes: usize,
    /// The limit that was exceeded, in bytes.
    pub limit_bytes: usize,
}

impl QueryMemoryExceeded {
    /// `true` when a retry can succeed once other queries finish (`Shared`,
    /// `Process`); `false` for the deterministic per-query budget.
    pub fn is_transient(&self) -> bool {
        !matches!(self.scope, QueryMemoryScope::Query)
    }
}

impl fmt::Display for QueryMemoryExceeded {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.scope {
            QueryMemoryScope::Query => write!(
                f,
                "query exceeded its working-memory budget: it needs at least {} bytes, \
                 the per-query limit (NAMIDB_QUERY_MEMORY_MAX_BYTES) is {} bytes; \
                 return fewer or narrower rows (LIMIT, pagination, aggregation)",
                self.required_bytes, self.limit_bytes
            ),
            QueryMemoryScope::Shared => write!(
                f,
                "query working memory unavailable: it needs {} bytes but concurrent \
                 queries hold the shared query-memory pool of {} bytes; retry after \
                 other queries finish",
                self.required_bytes, self.limit_bytes
            ),
            QueryMemoryScope::Process => write!(
                f,
                "process memory pressure: resident {} bytes reached the configured \
                 maximum of {} bytes while the query was running; retry after memory falls",
                self.required_bytes, self.limit_bytes
            ),
        }
    }
}

/// Process-wide byte pool that every concurrent query's budget reserves
/// from, so many queries that each fit their own budget cannot add up past
/// the process ceiling together.
#[derive(Debug)]
pub struct QueryMemoryPool {
    capacity: usize,
    reserved: AtomicUsize,
    peak: AtomicUsize,
    rejections: AtomicU64,
}

impl QueryMemoryPool {
    /// A pool of `capacity_bytes`. Zero rejects every non-empty reservation.
    pub fn new(capacity_bytes: usize) -> Self {
        Self {
            capacity: capacity_bytes,
            reserved: AtomicUsize::new(0),
            peak: AtomicUsize::new(0),
            rejections: AtomicU64::new(0),
        }
    }

    /// Configured capacity in bytes.
    pub fn capacity_bytes(&self) -> usize {
        self.capacity
    }

    /// Bytes currently reserved by in-flight queries.
    pub fn reserved_bytes(&self) -> usize {
        self.reserved.load(Ordering::Acquire)
    }

    /// High-water mark of [`Self::reserved_bytes`] since start.
    pub fn peak_reserved_bytes(&self) -> usize {
        self.peak.load(Ordering::Acquire)
    }

    /// Reservations refused because the pool was full.
    pub fn rejections(&self) -> u64 {
        self.rejections.load(Ordering::Relaxed)
    }

    fn try_reserve(&self, bytes: usize) -> bool {
        let mut current = self.reserved.load(Ordering::Acquire);
        loop {
            let Some(next) = current.checked_add(bytes).filter(|n| *n <= self.capacity) else {
                self.rejections.fetch_add(1, Ordering::Relaxed);
                return false;
            };
            match self.reserved.compare_exchange_weak(
                current,
                next,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => {
                    self.peak.fetch_max(next, Ordering::AcqRel);
                    return true;
                }
                Err(observed) => current = observed,
            }
        }
    }

    fn release(&self, bytes: usize) {
        if bytes > 0 {
            let previous = self.reserved.fetch_sub(bytes, Ordering::AcqRel);
            debug_assert!(previous >= bytes);
        }
    }
}

/// Resident-memory probe: returns `Some((resident_bytes, max_bytes))` when
/// the process is at or above its ceiling, `None` otherwise.
pub type MemoryProbe = Arc<dyn Fn() -> Option<(usize, usize)> + Send + Sync>;

/// One query's working-memory account. Create one per statement, scope it
/// with [`with_query_memory`], and keep the `Arc` to charge work done after
/// execution (for example, building a response body) with [`Self::charge`].
/// Dropping the last reference returns the query's pool reservation.
pub struct QueryMemoryBudget {
    limit: usize,
    used: AtomicUsize,
    peak: AtomicUsize,
    pool: Option<Arc<QueryMemoryPool>>,
    pool_held: AtomicUsize,
    probe: Option<MemoryProbe>,
    next_probe_at: AtomicUsize,
}

impl fmt::Debug for QueryMemoryBudget {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("QueryMemoryBudget")
            .field("limit", &self.limit)
            .field("used", &self.used_bytes())
            .field("peak", &self.peak_bytes())
            .field("pool", &self.pool.as_ref().map(|p| p.capacity_bytes()))
            .field("probe", &self.probe.is_some())
            .finish()
    }
}

impl QueryMemoryBudget {
    /// A budget of `limit_bytes` for one query. `usize::MAX` leaves only
    /// the shared pool and the probe (if attached) in force.
    pub fn new(limit_bytes: usize) -> Self {
        Self {
            limit: limit_bytes,
            used: AtomicUsize::new(0),
            peak: AtomicUsize::new(0),
            pool: None,
            pool_held: AtomicUsize::new(0),
            probe: None,
            next_probe_at: AtomicUsize::new(PROBE_INTERVAL_BYTES),
        }
    }

    /// Reserve this query's working memory from a shared pool as it grows.
    pub fn with_pool(mut self, pool: Arc<QueryMemoryPool>) -> Self {
        self.pool = Some(pool);
        self
    }

    /// Probe process resident memory every 16 MiB of growth and stop the
    /// query when the probe reports the process at its ceiling.
    pub fn with_probe(mut self, probe: MemoryProbe) -> Self {
        self.probe = Some(probe);
        self
    }

    /// The per-query limit in bytes.
    pub fn limit_bytes(&self) -> usize {
        self.limit
    }

    /// Bytes currently charged.
    pub fn used_bytes(&self) -> usize {
        self.used.load(Ordering::Acquire)
    }

    /// The largest working set this query was charged for.
    pub fn peak_bytes(&self) -> usize {
        self.peak.load(Ordering::Acquire)
    }

    /// Charge `bytes` more on top of what is already charged and keep it
    /// charged (until the budget is dropped).
    pub fn charge(&self, bytes: usize) -> Result<(), QueryMemoryExceeded> {
        let projected = self.used_bytes().saturating_add(bytes);
        self.ensure(projected)?;
        self.set_used(projected);
        Ok(())
    }

    /// Check that the query may hold `projected` bytes in total right now,
    /// growing its pool reservation to cover it. Does not change the charged
    /// total.
    fn ensure(&self, projected: usize) -> Result<(), QueryMemoryExceeded> {
        if projected > self.limit {
            return Err(QueryMemoryExceeded {
                scope: QueryMemoryScope::Query,
                required_bytes: projected,
                limit_bytes: self.limit,
            });
        }
        if let Some(pool) = &self.pool {
            let held = self.pool_held.load(Ordering::Acquire);
            if projected > held {
                // Round up to whole chunks, but never past what the pool
                // could ever give this query: a budget clamped to the pool's
                // (not chunk-aligned) capacity must still fit when it runs
                // alone, so `Shared` only means other queries hold the pool.
                let exact = projected - held;
                let need = round_up(exact, POOL_CHUNK_BYTES)
                    .min(pool.capacity_bytes().saturating_sub(held))
                    .max(exact);
                if !pool.try_reserve(need) {
                    return Err(QueryMemoryExceeded {
                        scope: QueryMemoryScope::Shared,
                        required_bytes: projected,
                        limit_bytes: pool.capacity_bytes(),
                    });
                }
                self.pool_held.fetch_add(need, Ordering::AcqRel);
            }
        }
        if let Some(probe) = &self.probe {
            let due = self.next_probe_at.load(Ordering::Acquire);
            if projected >= due {
                self.next_probe_at.store(
                    projected.saturating_add(PROBE_INTERVAL_BYTES),
                    Ordering::Release,
                );
                if let Some((resident, max)) = probe() {
                    return Err(QueryMemoryExceeded {
                        scope: QueryMemoryScope::Process,
                        required_bytes: resident,
                        limit_bytes: max,
                    });
                }
            }
        }
        self.peak.fetch_max(projected, Ordering::AcqRel);
        Ok(())
    }

    /// Set the charged total, returning pool chunks the query no longer
    /// needs. Keeps up to two spare chunks so a query hovering around a
    /// chunk boundary does not churn the shared counter.
    fn set_used(&self, value: usize) {
        self.used.store(value, Ordering::Release);
        if let Some(pool) = &self.pool {
            let keep = round_up(value, POOL_CHUNK_BYTES);
            let held = self.pool_held.load(Ordering::Acquire);
            if held > keep.saturating_add(2 * POOL_CHUNK_BYTES) {
                let surplus = held - keep;
                self.pool_held.fetch_sub(surplus, Ordering::AcqRel);
                pool.release(surplus);
            }
        }
        // Re-arm the probe relative to the new level, so a query that
        // shrinks and grows again is still sampled every interval.
        let next = value.saturating_add(PROBE_INTERVAL_BYTES);
        if next < self.next_probe_at.load(Ordering::Acquire) {
            self.next_probe_at.store(next, Ordering::Release);
        }
    }
}

impl Drop for QueryMemoryBudget {
    fn drop(&mut self) {
        if let Some(pool) = &self.pool {
            pool.release(*self.pool_held.get_mut());
        }
    }
}

tokio::task_local! {
    static BUDGET: Arc<QueryMemoryBudget>;
}

/// Run `fut` with `budget` charged by the executor. `None` runs it
/// unbudgeted at no extra cost.
pub async fn with_query_memory<F: Future>(
    budget: Option<Arc<QueryMemoryBudget>>,
    fut: F,
) -> F::Output {
    match budget {
        Some(budget) => BUDGET.scope(budget, fut).await,
        None => fut.await,
    }
}

fn with_budget<R>(f: impl FnOnce(&QueryMemoryBudget) -> R) -> Option<R> {
    BUDGET.try_with(|budget| f(budget)).ok()
}

/// Whether a budget is scoped on this task.
#[inline]
pub(crate) fn memory_scoped() -> bool {
    with_budget(|_| ()).is_some()
}

/// The charged total when an operator starts, or `None` without a budget.
#[inline]
pub(crate) fn frame_base() -> Option<usize> {
    with_budget(QueryMemoryBudget::used_bytes)
}

/// Check that `extra` bytes of transient state fit on top of what is
/// charged, without keeping them charged.
#[inline]
pub(crate) fn check_transient(extra: usize) -> Result<(), ExecError> {
    match with_budget(|budget| budget.ensure(budget.used_bytes().saturating_add(extra))) {
        Some(Err(exceeded)) => Err(ExecError::MemoryLimit(exceeded)),
        _ => Ok(()),
    }
}

/// How an operator's output relates to its inputs, for [`settle`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum OutputKind {
    /// New allocations built while the inputs were still alive (Project,
    /// Expand, joins, aggregation, scans): peak = inputs + output.
    Additive,
    /// The output rows were moved out of the inputs (Filter, SemiApply,
    /// Union ALL): nothing new was allocated, so the inputs' charge carries
    /// over unchanged and no estimate is needed.
    Moved,
    /// Rows moved out of the input and possibly truncated (TopN, DISTINCT):
    /// peak = max(inputs, output); the dropped remainder is released.
    Replaced,
}

/// Account an operator's finished output. `base` is [`frame_base`] at the
/// operator's start. The inputs charged since `base` are released and the
/// output stays charged until the parent settles.
pub(crate) fn settle(base: usize, rows: &Vec<Row>, kind: OutputKind) -> Result<(), ExecError> {
    settle_with(base, kind, || estimate_row_vec_bytes(rows))
}

/// [`settle`] for a factorised operator's output: the whole arena it owns
/// (inherited input nodes included) plus its leaf list.
pub(crate) fn settle_factor(
    base: usize,
    set: &FactorRowSet,
    kind: OutputKind,
) -> Result<(), ExecError> {
    settle_with(base, kind, || estimate_factor_set_bytes(set))
}

fn settle_with(
    base: usize,
    kind: OutputKind,
    estimate: impl FnOnce() -> usize,
) -> Result<(), ExecError> {
    let outcome = with_budget(|budget| -> Result<(), QueryMemoryExceeded> {
        let current = budget.used_bytes();
        let (peak, keep) = match kind {
            OutputKind::Moved => (current, current),
            OutputKind::Additive => {
                let out = estimate();
                (current.saturating_add(out), base.saturating_add(out))
            }
            OutputKind::Replaced => {
                let out = estimate();
                let keep = base.saturating_add(out);
                (current.max(keep), keep)
            }
        };
        budget.ensure(peak)?;
        budget.set_used(keep);
        Ok(())
    });
    match outcome {
        Some(Err(exceeded)) => Err(ExecError::MemoryLimit(exceeded)),
        _ => Ok(()),
    }
}

/// Swap `release` charged bytes for `charge` new ones: an input the
/// operator has just dropped for the copy it made of it (the factorised
/// path materialising a set into rows). The copy stays charged until the
/// operator settles.
pub(crate) fn exchange(release: usize, charge: usize) -> Result<(), ExecError> {
    let outcome = with_budget(|budget| -> Result<(), QueryMemoryExceeded> {
        let next = budget
            .used_bytes()
            .saturating_sub(release)
            .saturating_add(charge);
        budget.ensure(next)?;
        budget.set_used(next);
        Ok(())
    });
    match outcome {
        Some(Err(exceeded)) => Err(ExecError::MemoryLimit(exceeded)),
        _ => Ok(()),
    }
}

/// Return the charged total to `mark` (a [`frame_base`] taken earlier).
/// Used after a correlated subquery's per-row output has been consumed, so
/// a subquery evaluated once per outer row does not accumulate every
/// evaluation's output.
#[inline]
pub(crate) fn restore(mark: Option<usize>) {
    if let Some(mark) = mark {
        let _ = with_budget(|budget| budget.set_used(mark));
    }
}

/// Incremental accounting for a producer loop. Call [`Self::observe`] with
/// the output vector (rows pushed since the last call are estimated once)
/// or [`Self::add`] for other state; the budget is checked every
/// [`METER_FLUSH_BYTES`]. The bytes are transient: the operator boundary
/// recomputes and charges the finished output.
#[derive(Debug)]
pub(crate) struct GrowthMeter {
    active: bool,
    upto: usize,
    /// Bytes of the observed vector's buffer, by capacity, included in
    /// `bytes`.
    buffer: usize,
    bytes: usize,
    checked: usize,
}

impl GrowthMeter {
    pub(crate) fn new() -> Self {
        Self {
            active: memory_scoped(),
            upto: 0,
            buffer: 0,
            bytes: 0,
            checked: 0,
        }
    }

    /// Account rows appended to `rows` since the last call.
    #[inline]
    pub(crate) fn observe(&mut self, rows: &Vec<Row>) -> Result<(), ExecError> {
        if !self.active {
            return Ok(());
        }
        if rows.len() < self.upto {
            // The caller drained or replaced the vector; start over.
            self.upto = 0;
        }
        let fresh: usize = rows[self.upto..]
            .iter()
            .map(estimate_row_bytes)
            .fold(0usize, usize::saturating_add);
        self.upto = rows.len();
        // A doubling `Vec` holds up to twice its length in slots.
        let buffer = alloc(rows.capacity().saturating_mul(std::mem::size_of::<Row>()));
        self.bytes = self.bytes.saturating_sub(self.buffer);
        self.checked = self.checked.min(self.bytes);
        self.buffer = buffer;
        self.add(fresh.saturating_add(buffer))
    }

    /// Account `bytes` of operator state.
    #[inline]
    pub(crate) fn add(&mut self, bytes: usize) -> Result<(), ExecError> {
        if !self.active {
            return Ok(());
        }
        self.bytes = self.bytes.saturating_add(bytes);
        if self.bytes - self.checked >= METER_FLUSH_BYTES {
            self.checked = self.bytes;
            check_transient(self.bytes)?;
        }
        Ok(())
    }

    /// Stop accounting `bytes` added earlier (state the operator freed).
    #[inline]
    pub(crate) fn release(&mut self, bytes: usize) {
        self.bytes = self.bytes.saturating_sub(bytes);
        self.checked = self.checked.min(self.bytes);
    }

    /// Whether a budget is in scope (callers can skip building estimates).
    #[inline]
    pub(crate) fn active(&self) -> bool {
        self.active
    }

    /// Bytes accounted so far.
    #[inline]
    pub(crate) fn bytes(&self) -> usize {
        self.bytes
    }
}

// ───────────────────────────── estimators ─────────────────────────────
//
// The numbers model a 64-bit target with glibc malloc: every heap block
// carries an 8-byte header, is rounded up to 16 bytes, and is at least 32
// bytes. B-tree maps allocate fixed-size nodes of 11 slots whatever their
// fill; non-root nodes hold at least 5 entries, which is what we assume.

const BTREE_CAPACITY: usize = 11;
const BTREE_MIN_LEN: usize = 5;
/// Parent pointer, parent index and length of a B-tree leaf node.
const BTREE_LEAF_HEADER: usize = 16;
/// Child pointers an internal node adds on top of a leaf's layout.
const BTREE_EDGES: usize = (BTREE_CAPACITY + 1) * std::mem::size_of::<usize>();
const VALUE_SIZE: usize = std::mem::size_of::<RuntimeValue>();
const STRING_SIZE: usize = std::mem::size_of::<String>();
const SLOT_SIZE: usize = std::mem::size_of::<Slot>();

#[inline]
fn round_up(value: usize, multiple: usize) -> usize {
    value.div_ceil(multiple).saturating_mul(multiple)
}

/// Bytes malloc hands out for a request of `size` bytes.
#[inline]
fn alloc(size: usize) -> usize {
    if size == 0 {
        0
    } else {
        round_up(size.saturating_add(8), 16).max(32)
    }
}

/// Heap bytes of the nodes of a B-tree map with `len` entries.
fn btree_nodes(len: usize, entry_size: usize) -> usize {
    if len == 0 {
        return 0;
    }
    let leaf = BTREE_LEAF_HEADER + BTREE_CAPACITY * entry_size;
    if len <= BTREE_CAPACITY {
        return alloc(leaf);
    }
    let leaves = len.div_ceil(BTREE_MIN_LEN);
    let internals = leaves.div_ceil(BTREE_MIN_LEN + 1);
    leaves
        .saturating_mul(alloc(leaf))
        .saturating_add(internals.saturating_mul(alloc(leaf + BTREE_EDGES)))
}

#[inline]
fn string_heap(s: &str, capacity: usize) -> usize {
    alloc(capacity.max(s.len()))
}

fn map_heap(map: &std::collections::BTreeMap<String, RuntimeValue>) -> usize {
    map.iter()
        .map(|(k, v)| string_heap(k, k.capacity()).saturating_add(estimate_value_bytes(v)))
        .fold(
            btree_nodes(map.len(), STRING_SIZE + VALUE_SIZE),
            usize::saturating_add,
        )
}

fn node_heap(node: &NodeValue) -> usize {
    let labels = node
        .labels
        .iter()
        .map(|l| string_heap(l, l.capacity()))
        .fold(
            btree_nodes(node.labels.len(), STRING_SIZE),
            usize::saturating_add,
        );
    alloc(std::mem::size_of::<NodeValue>())
        .saturating_add(labels)
        .saturating_add(map_heap(&node.properties))
}

fn rel_heap(rel: &RelValue) -> usize {
    alloc(std::mem::size_of::<RelValue>())
        .saturating_add(string_heap(&rel.edge_type, rel.edge_type.capacity()))
        .saturating_add(map_heap(&rel.properties))
}

fn list_heap(items: &Vec<RuntimeValue>) -> usize {
    items.iter().map(estimate_value_bytes).fold(
        alloc(items.capacity().saturating_mul(VALUE_SIZE)),
        usize::saturating_add,
    )
}

/// Heap bytes owned by one value, not counting the value's own inline slot.
pub fn estimate_value_bytes(value: &RuntimeValue) -> usize {
    match value {
        RuntimeValue::Null
        | RuntimeValue::Bool(_)
        | RuntimeValue::Integer(_)
        | RuntimeValue::Float(_)
        | RuntimeValue::Date(_)
        | RuntimeValue::DateTime(_) => 0,
        RuntimeValue::String(s) => string_heap(s, s.capacity()),
        RuntimeValue::Bytes(b) => alloc(b.capacity()),
        RuntimeValue::Vector(v) => alloc(v.capacity().saturating_mul(4)),
        RuntimeValue::Vector8 { codes, .. } => alloc(codes.capacity()),
        RuntimeValue::List(items) | RuntimeValue::Path(items) => list_heap(items),
        RuntimeValue::Map(map) => map_heap(map),
        RuntimeValue::Node(node) => node_heap(node),
        RuntimeValue::Rel(rel) => rel_heap(rel),
    }
}

/// Heap bytes of a value vector: its buffer (by capacity) plus every
/// value's heap.
pub fn estimate_values_bytes(values: &Vec<RuntimeValue>) -> usize {
    list_heap(values)
}

/// Heap bytes owned by one row, not counting the row's own inline slot.
pub fn estimate_row_bytes(row: &Row) -> usize {
    map_heap(&row.bindings)
}

/// Heap bytes of a whole row vector: its buffer (by capacity) plus every
/// row's heap.
pub fn estimate_row_vec_bytes(rows: &Vec<Row>) -> usize {
    rows.iter().map(estimate_row_bytes).fold(
        alloc(rows.capacity().saturating_mul(std::mem::size_of::<Row>())),
        usize::saturating_add,
    )
}

/// Heap bytes of a shared binding name. Factor nodes usually share one
/// `Arc<str>` per binding; each holder is charged its share of the block so
/// a set's total counts the block once.
fn arc_name_heap(name: &Arc<str>) -> usize {
    alloc(16 + name.len()) / Arc::strong_count(name).max(1)
}

/// Heap bytes one factor node adds to an arena: its slot in the arena's
/// node vector (twice, for the doubling buffer), its slot buffer, and each
/// slot's value and name.
pub(crate) fn estimate_factor_slots_bytes(slots: &Vec<Slot>) -> usize {
    slots
        .iter()
        .map(|slot| estimate_value_bytes(&slot.value).saturating_add(arc_name_heap(&slot.name)))
        .fold(
            alloc(slots.capacity().saturating_mul(SLOT_SIZE))
                .saturating_add(2 * std::mem::size_of::<FactorNode>()),
            usize::saturating_add,
        )
}

/// Heap bytes of the slots on a leaf's whole chain, root-ward: what one copy
/// of that leaf's bindings costs as factor slots.
pub(crate) fn estimate_factor_chain_bytes(arena: &FactorArena, leaf: FactorIdx) -> usize {
    let mut bytes = 0usize;
    let mut cur = Some(leaf);
    while let Some(idx) = cur {
        let node = arena.node(idx);
        bytes = bytes.saturating_add(estimate_factor_slots_bytes(&node.slots));
        cur = node.parent;
    }
    bytes
}

/// Heap bytes of a factorised result: the arena's node vector (by
/// capacity), every node's slots, and the leaf list.
pub fn estimate_factor_set_bytes(set: &FactorRowSet) -> usize {
    let nodes = set.arena.nodes();
    nodes
        .iter()
        .map(|node| {
            node.slots
                .iter()
                .map(|slot| {
                    estimate_value_bytes(&slot.value).saturating_add(arc_name_heap(&slot.name))
                })
                .fold(
                    alloc(node.slots.capacity().saturating_mul(SLOT_SIZE)),
                    usize::saturating_add,
                )
        })
        .fold(
            alloc(
                nodes
                    .capacity()
                    .saturating_mul(std::mem::size_of::<FactorNode>()),
            )
            .saturating_add(alloc(
                set.leaves
                    .capacity()
                    .saturating_mul(std::mem::size_of::<FactorIdx>()),
            )),
            usize::saturating_add,
        )
}

/// Upper estimate of the JSON text a value serialises to: strings are
/// counted at twice their length to cover escaping, numbers at their
/// longest form.
pub fn estimate_value_text_bytes(value: &RuntimeValue) -> usize {
    let props = |map: &std::collections::BTreeMap<String, RuntimeValue>| {
        map.iter()
            .map(|(k, v)| (2 * k.len() + 4).saturating_add(estimate_value_text_bytes(v)))
            .fold(2usize, usize::saturating_add)
    };
    match value {
        RuntimeValue::Null => 4,
        RuntimeValue::Bool(_) => 5,
        RuntimeValue::Integer(_) => 20,
        RuntimeValue::Float(_) => 24,
        RuntimeValue::Date(_) => 12,
        RuntimeValue::DateTime(_) => 36,
        RuntimeValue::String(s) => 2 * s.len() + 2,
        RuntimeValue::Bytes(b) => b.len().div_ceil(3) * 4 + 2,
        RuntimeValue::Vector(v) => v.len().saturating_mul(25) + 2,
        RuntimeValue::Vector8 { codes, .. } => codes.len().saturating_mul(25) + 2,
        RuntimeValue::List(items) | RuntimeValue::Path(items) => items
            .iter()
            .map(|v| estimate_value_text_bytes(v).saturating_add(1))
            .fold(2usize, usize::saturating_add),
        RuntimeValue::Map(map) => props(map),
        RuntimeValue::Node(node) => node
            .labels
            .iter()
            .map(|l| 2 * l.len() + 3)
            .fold(96usize, usize::saturating_add)
            .saturating_add(props(&node.properties)),
        RuntimeValue::Rel(rel) => {
            (128 + 2 * rel.edge_type.len()).saturating_add(props(&rel.properties))
        }
    }
}

/// Upper estimate of the JSON text a result serialises to.
pub fn estimate_rows_text_bytes(rows: &[Row]) -> usize {
    rows.iter()
        .map(|row| {
            row.bindings
                .iter()
                .map(|(k, v)| (2 * k.len() + 4).saturating_add(estimate_value_text_bytes(v)))
                .fold(3usize, usize::saturating_add)
        })
        .fold(2usize, usize::saturating_add)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn int_row(i: i64) -> Row {
        Row::new()
            .with("i", RuntimeValue::Integer(i))
            .with("s", RuntimeValue::String(i.to_string()))
    }

    #[test]
    fn row_estimate_covers_the_measured_cost_of_a_narrow_row() {
        // #203 measured ~650 bytes per materialised two-column row. The
        // estimate must not be below that, or a budget would let the very
        // query from the issue through.
        let rows: Vec<Row> = (0..1000).map(int_row).collect();
        let per_row = estimate_row_vec_bytes(&rows) / rows.len();
        assert!(per_row >= 650, "estimated {per_row} bytes per row");
        assert!(
            per_row <= 1300,
            "estimate {per_row} is not conservative but absurd"
        );
    }

    #[test]
    fn estimates_grow_with_payload() {
        let small = RuntimeValue::String("x".into());
        let big = RuntimeValue::String("x".repeat(10_000));
        assert!(estimate_value_bytes(&big) >= 10_000);
        assert!(estimate_value_bytes(&small) >= 32);
        let list = RuntimeValue::List((0..1000).map(RuntimeValue::Integer).collect());
        assert!(estimate_value_bytes(&list) >= 1000 * VALUE_SIZE);
        let map: std::collections::BTreeMap<String, RuntimeValue> = (0..100)
            .map(|i| (format!("k{i}"), RuntimeValue::Integer(i)))
            .collect();
        // 100 entries need at least ten 11-slot nodes.
        assert!(
            estimate_value_bytes(&RuntimeValue::Map(map))
                >= 10 * (BTREE_CAPACITY * (STRING_SIZE + VALUE_SIZE))
        );
        assert!(estimate_value_text_bytes(&big) >= 10_000);
    }

    #[test]
    fn budget_charges_and_reports_the_query_scope() {
        let budget = QueryMemoryBudget::new(1000);
        budget.charge(600).unwrap();
        assert_eq!(budget.used_bytes(), 600);
        let err = budget.charge(600).unwrap_err();
        assert_eq!(err.scope, QueryMemoryScope::Query);
        assert_eq!(err.required_bytes, 1200);
        assert_eq!(err.limit_bytes, 1000);
        assert!(!err.is_transient());
        assert_eq!(budget.used_bytes(), 600, "a refused charge is not kept");
        assert!(err.to_string().contains("NAMIDB_QUERY_MEMORY_MAX_BYTES"));
    }

    #[test]
    fn pool_is_shared_and_released_on_drop() {
        let pool = Arc::new(QueryMemoryPool::new(4 * POOL_CHUNK_BYTES));
        let a = QueryMemoryBudget::new(usize::MAX).with_pool(Arc::clone(&pool));
        let b = QueryMemoryBudget::new(usize::MAX).with_pool(Arc::clone(&pool));
        a.charge(3 * POOL_CHUNK_BYTES).unwrap();
        let err = b.charge(2 * POOL_CHUNK_BYTES).unwrap_err();
        assert_eq!(err.scope, QueryMemoryScope::Shared);
        assert!(err.is_transient());
        assert_eq!(pool.rejections(), 1);
        drop(a);
        assert_eq!(pool.reserved_bytes(), 0);
        b.charge(2 * POOL_CHUNK_BYTES).unwrap();
        assert_eq!(pool.reserved_bytes(), 2 * POOL_CHUNK_BYTES);
        drop(b);
        assert_eq!(pool.reserved_bytes(), 0);
        assert_eq!(pool.peak_reserved_bytes(), 3 * POOL_CHUNK_BYTES);
    }

    #[test]
    fn lone_query_at_an_unaligned_pool_capacity_is_not_refused_as_shared() {
        // The server clamps the per-query budget to the pool's capacity,
        // which is a percentage of the ceiling and not chunk-aligned. A
        // query alone on the server that stays within its budget must
        // always reserve, however close to the limit it gets.
        let capacity = 3 * POOL_CHUNK_BYTES + 12_345;
        let pool = Arc::new(QueryMemoryPool::new(capacity));
        let budget = QueryMemoryBudget::new(capacity).with_pool(Arc::clone(&pool));
        budget.charge(3 * POOL_CHUNK_BYTES + 1).unwrap();
        budget.charge(12_000).unwrap();
        budget.charge(capacity - budget.used_bytes()).unwrap();
        assert_eq!(budget.used_bytes(), capacity);
        assert_eq!(pool.reserved_bytes(), capacity);
        assert_eq!(pool.rejections(), 0);
        // One byte more is the query's own limit, not the shared pool.
        let err = budget.charge(1).unwrap_err();
        assert_eq!(err.scope, QueryMemoryScope::Query);
        drop(budget);
        assert_eq!(pool.reserved_bytes(), 0);

        // With another query holding part of the pool, the same need is
        // transient.
        let other = QueryMemoryBudget::new(capacity).with_pool(Arc::clone(&pool));
        other.charge(POOL_CHUNK_BYTES).unwrap();
        let budget = QueryMemoryBudget::new(capacity).with_pool(Arc::clone(&pool));
        let err = budget.charge(3 * POOL_CHUNK_BYTES).unwrap_err();
        assert_eq!(err.scope, QueryMemoryScope::Shared);
    }

    #[test]
    fn shrinking_returns_pool_chunks() {
        let pool = Arc::new(QueryMemoryPool::new(64 * POOL_CHUNK_BYTES));
        let budget = QueryMemoryBudget::new(usize::MAX).with_pool(Arc::clone(&pool));
        budget.charge(32 * POOL_CHUNK_BYTES).unwrap();
        budget.set_used(POOL_CHUNK_BYTES);
        assert!(pool.reserved_bytes() <= 3 * POOL_CHUNK_BYTES);
    }

    #[test]
    fn probe_stops_a_growing_query() {
        let budget = QueryMemoryBudget::new(usize::MAX).with_probe(Arc::new(|| Some((900, 800))));
        // Below the first probe interval the probe is not consulted.
        budget.charge(1024).unwrap();
        let err = budget.charge(PROBE_INTERVAL_BYTES).unwrap_err();
        assert_eq!(err.scope, QueryMemoryScope::Process);
        assert_eq!((err.required_bytes, err.limit_bytes), (900, 800));
        assert!(err.to_string().starts_with("process memory pressure"));
    }

    #[tokio::test]
    async fn settle_keeps_only_the_output_charged() {
        let budget = Arc::new(QueryMemoryBudget::new(usize::MAX));
        with_query_memory(Some(Arc::clone(&budget)), async {
            let base = frame_base().unwrap();
            assert_eq!(base, 0);
            // A child leaves 1000 rows charged.
            let child: Vec<Row> = (0..1000).map(int_row).collect();
            settle(base, &child, OutputKind::Additive).unwrap();
            let child_bytes = frame_base().unwrap();
            assert_eq!(child_bytes, estimate_row_vec_bytes(&child));
            // A parent that keeps 10 rows: peak includes the child, the
            // child is released afterwards.
            let out: Vec<Row> = (0..10).map(int_row).collect();
            settle(base, &out, OutputKind::Replaced).unwrap();
            assert_eq!(frame_base().unwrap(), estimate_row_vec_bytes(&out));
            assert!(budget.peak_bytes() >= child_bytes);
            // A per-outer-row subquery's output is released by `restore`.
            let mark = frame_base();
            settle(mark.unwrap(), &child, OutputKind::Additive).unwrap();
            assert!(frame_base().unwrap() > mark.unwrap());
            restore(mark);
            assert_eq!(frame_base(), mark);
            // Moved output carries the inputs' charge over unchanged.
            let before = frame_base().unwrap();
            settle(0, &child, OutputKind::Moved).unwrap();
            assert_eq!(frame_base().unwrap(), before);
        })
        .await;
    }

    #[tokio::test]
    async fn meter_fails_mid_growth() {
        let budget = Arc::new(QueryMemoryBudget::new(2 * METER_FLUSH_BYTES));
        let err = with_query_memory(Some(budget), async {
            let mut meter = GrowthMeter::new();
            let mut out = Vec::new();
            for i in 0..1_000_000 {
                out.push(int_row(i));
                if out.len() % 64 == 0 {
                    meter.observe(&out)?;
                }
            }
            Ok::<_, ExecError>(out.len())
        })
        .await
        .unwrap_err();
        match err {
            ExecError::MemoryLimit(e) => assert_eq!(e.scope, QueryMemoryScope::Query),
            other => panic!("unexpected {other:?}"),
        }
    }

    #[test]
    fn without_a_budget_everything_is_free() {
        assert!(!memory_scoped());
        assert!(frame_base().is_none());
        check_transient(usize::MAX).unwrap();
        let mut meter = GrowthMeter::new();
        meter.add(usize::MAX).unwrap();
        settle(0, &vec![int_row(1)], OutputKind::Additive).unwrap();
        restore(Some(0));
    }
}
