# RFC 021: Concurrent reads without the single-writer mutex

**Status:** draft
**Author(s):** Matías Fonseca <info@namidb.com>
**Created:** 2026-05-24
**Updated:** 2026-05-24
**Implements:** (pending)
**Supersedes:** none

## Summary

Take the `tokio::Mutex<WriterSession>` off the read path so a single
`namidb-server` process serves reads in parallel across every core.
The mutex stays in place for writes (single-writer-per-namespace
remains a hard invariant from RFC-001), but read requests pick up an
owned, lifetime-free snapshot through an atomic `Arc` swap and run
without holding any cross-request lock.

This is the unlock for two visible problems:

1. **HTTP + Bolt both serialize today.** Every `/v0/cypher` and
   every Bolt `RUN` takes `state.writer.lock().await`. Two read
   queries on different cores still run one after the other.
2. **The snapshot's `'mt` lifetime forces the lock to be held for
   the entire query.** `Snapshot<'mt>` borrows `memtable: &'mt
   Memtable` from the writer, so the lock has to stay live until the
   last row is consumed. The Bolt session and the HTTP handler both
   structure the request around that constraint.

The fix is an owned snapshot: `OwnedSnapshot` carries an
`Arc<MemtableSnapshot>` (a frozen, immutable copy of the writer's
memtable at commit time) plus all the same caches and manifest data
the borrowed `Snapshot` does today. The writer publishes a new
`Arc<OwnedSnapshot>` on every successful `commit_batch` /
`flush`, atomically. Readers grab the current `Arc` and run.

## Motivation

A profile of `namidb-server` under three concurrent Bolt clients
shows each `RUN` taking the writer mutex for the full duration of
the query. The `tokio::Mutex` lock acquisition adds nanoseconds, but
the wall clock impact is the queueing: three reads of 30 ms each
take 90 ms wall in total because the second and third client wait.
`/v0/cypher` shows the same shape.

The bench harness already measures this implicitly: the LDBC SNB
results in `bench/` are single-threaded warm runs, which is the only
shape the current implementation can serve well. Anyone running
`hey -c 4 -n 1000 http://localhost:8080/v0/cypher` watches their
throughput stay roughly the same as `-c 1`.

The shape of the fix is small and well understood; LSM-tree engines
have been doing this since RocksDB. The cost is one targeted
storage refactor (snapshot owns its memtable view rather than
borrowing) and one server refactor (split read state from writer
state). Everything else stays.

## Design

### The two shapes of state

```text
┌──────────────────────────────────────────────────────────────────┐
│  Process                                                         │
│                                                                  │
│  ┌──────────────────────┐         ┌────────────────────────────┐ │
│  │  Writer (mut state)  │         │  Read snapshot (Arc)       │ │
│  │  ─────────────────── │         │  ────────────────────────  │ │
│  │  WriterSession       │  commit │  OwnedSnapshot             │ │
│  │  - active memtable   │ ───────▶│  - manifest                │ │
│  │  - WAL pending       │         │  - Arc<MemtableSnapshot>   │ │
│  │  - LSN allocator     │         │  - object store, paths     │ │
│  └──────────────────────┘         │  - per-snapshot caches     │ │
│            ▲                      └────────────────────────────┘ │
│   tokio::Mutex                              ▲                    │
│            │                          atomic load (lock-free*)   │
│   one writer at a time                      │                    │
│                                ┌────────────┴───────────┐        │
│                                │  many readers (N tasks │        │
│                                │  on a multi-core tokio │        │
│                                │  runtime)              │        │
│                                └────────────────────────┘        │
│                                                                  │
│  *via std::sync::Mutex<Arc<T>> in v0; arc-swap is a follow-up.   │
└──────────────────────────────────────────────────────────────────┘
```

### `MemtableSnapshot`

A new type in `namidb-storage::memtable`:

```rust
/// Immutable snapshot of a memtable's contents at a point in time.
///
/// Constructed by `Memtable::snapshot()` on a `commit_batch`. Shared
/// across every read that picks up the matching `OwnedSnapshot`.
/// Drops only when the last `Arc<MemtableSnapshot>` referencing it
/// goes away, so a long-running reader holds memory for as long as
/// it runs.
#[derive(Debug, Default)]
pub struct MemtableSnapshot {
    inner: BTreeMap<MemKey, MemEntry>,
}
```

`MemtableSnapshot` exposes the same `iter`, `iter_label`,
`iter_edge_type`, `get` interface the borrowed memtable does. The
read path stops referencing `&Memtable` directly; it goes through
`&MemtableSnapshot`.

### `OwnedSnapshot`

A new type in `namidb-storage::read`, intentionally distinct from
the existing `Snapshot<'mt>` so the migration is incremental:

```rust
pub struct OwnedSnapshot {
    manifest: LoadedManifest,
    memtable: Arc<MemtableSnapshot>,
    store: Arc<dyn ObjectStore>,
    paths: NamespacePaths,
    cache: Option<SstCache>,
    node_cache: Mutex<HashMap<(String, NodeId), Option<NodeView>>>,
    ranged_mode: RangedMode,
    ranged_threshold_bytes: u64,
    adjacency_cache: Option<Arc<AdjacencyCache>>,
    shared_node_cache: Option<Arc<NodeViewCache>>,
    property_index_cache: Option<Arc<PropertyIndexCache>>,
    decoded_node_sst_batches: Mutex<HashMap<String, Arc<Vec<RecordBatch>>>>,
}
```

Same shape as `Snapshot<'_>`, with two differences:

1. `memtable: Arc<MemtableSnapshot>` instead of `&'mt Memtable`. No
   lifetime parameter; the snapshot owns its memtable view via
   shared pointer.
2. The intra-snapshot scratch caches (`node_cache`,
   `decoded_node_sst_batches`) stay per-snapshot rather than shared
   with the writer, same as today.

Every read API today on `Snapshot<'_>` (`lookup_node`,
`batch_lookup_nodes`, `edge_lookup`, etc.) is mirrored on
`OwnedSnapshot`. The implementations are the same; the only diff is
how `self.memtable` is reached.

Rather than duplicate ~2 KLOC, we hoist the methods into a private
helper trait `SnapshotRead` that both types implement, or — simpler
for v0 — we keep `Snapshot<'_>` as the read engine and have
`OwnedSnapshot::as_borrowed(&self) -> Snapshot<'_>` re-package an
`OwnedSnapshot` into the existing type. The Arc keeps the memtable
alive for as long as the temporary borrow needs it. This is the
zero-touch migration path for the executor and storage internals.

### `WriterSession::publish_snapshot`

Today's `WriterSession::snapshot(&self) -> Snapshot<'_>` stays as is
for in-process callers that already hold a `&WriterSession`. We add
two new methods:

```rust
impl WriterSession {
    /// Build an `OwnedSnapshot` whose memtable view is a snapshot of
    /// the writer's current memtable. Cheap: `MemtableSnapshot` is
    /// constructed by cloning the writer's `BTreeMap` (O(n) in
    /// memtable size, but the memtable is bounded by the flush
    /// threshold, so this is a few hundred microseconds at the
    /// 8 MiB default).
    pub fn owned_snapshot(&self) -> OwnedSnapshot { ... }

    /// Build an `OwnedSnapshot` *and* tell the session this is the
    /// latest published one. Cheap, no copy of the memtable beyond
    /// the snapshot itself. Returns the `Arc` so the caller can put
    /// it where it serves reads from.
    pub fn publish_snapshot(&self) -> Arc<OwnedSnapshot> { ... }
}
```

The writer calls `publish_snapshot` after every `commit_batch` and
every `flush`. The returned `Arc` goes into a `SnapshotCell` the
server crate owns.

### `SnapshotCell` — the atomic publisher

The server keeps the read-active snapshot in a small cell:

```rust
pub struct SnapshotCell {
    inner: std::sync::Mutex<Arc<OwnedSnapshot>>,
}

impl SnapshotCell {
    pub fn load(&self) -> Arc<OwnedSnapshot> {
        Arc::clone(&self.inner.lock().expect("snapshot cell poisoned"))
    }
    pub fn store(&self, snap: Arc<OwnedSnapshot>) {
        *self.inner.lock().expect("snapshot cell poisoned") = snap;
    }
}
```

`std::sync::Mutex<Arc<T>>` is fine here. The critical section is
exactly one pointer copy plus the increment of the `Arc`'s strong
count, on the order of tens of nanoseconds. A real lock-free RCU
(via `arc-swap`) is one swap in `load` / `store`; we'll move when a
bench shows the mutex contention matters.

### Server wiring

`AppState` grows a snapshot cell. The HTTP and Bolt code paths
diverge:

```rust
#[derive(Clone)]
pub struct AppState {
    writer: Arc<tokio::Mutex<WriterSession>>,
    snapshot: Arc<SnapshotCell>,
    auth_token: Option<Arc<str>>,
    namespace: String,
}

async fn cypher(State(state): State<AppState>, Json(req): Json<CypherRequest>) -> Response {
    let parsed = match parse(&req.query) { /* ... */ };
    let plan = build_plan(...);
    if plan.contains_write() {
        // Same as today: take the writer lock, execute_write,
        // refresh the snapshot cell at the end.
        let mut writer = state.writer.lock().await;
        let outcome = execute_write(&plan, &mut writer, &params).await;
        let fresh = writer.publish_snapshot();
        state.snapshot.store(fresh);
        // ... encode response ...
    } else {
        // Read path: NO writer lock. Pick the latest published
        // snapshot, execute against it.
        let snap = state.snapshot.load();
        let rows = execute_owned(&plan, &snap, &params).await;
        // ... encode response ...
    }
}
```

Bolt sessions follow the same shape: `ServerBackend::run` checks
`plan.contains_write()` and branches on the writer mutex vs the
snapshot cell.

### Concurrent-reads property test

A new integration test in
`crates/namidb-server/tests/concurrent_reads.rs`:

1. Bootstraps a memory namespace with `n_nodes = 10_000`.
2. Spawns `k = 16` tokio tasks, each running a long-ish read
   (`MATCH (p:Person) WHERE p.age >= $min RETURN count(*) AS c`) in
   a tight loop for `t = 1 s`.
3. Counts total queries done.
4. Compares against a single-threaded reference run.

Before RFC-021: `k=16` total throughput is ≈ `k=1` throughput
(within a noise band). After RFC-021: `k=16` total throughput is ≈
`k × k=1` throughput up to the core count.

We assert the post-RFC-021 case is at least 4× the pre-RFC-021 case
on a 4-core or larger CI runner.

### Writer-write interaction

The mutex on `WriterSession` is still held for every write. While a
write is running:

- Other writes wait at the mutex (unchanged).
- Reads do **not** wait. They keep serving the *previous* published
  snapshot. The writer doesn't republish until `commit_batch`
  returns successfully, so a write that's in flight is invisible to
  readers — which is exactly the snapshot isolation the RFC-001
  manifest CAS already guarantees.

This means a slow writer doesn't degrade read latency. Reads see
the last-committed manifest version and serve from that.

### Stale-snapshot bound

A reader that picked up a snapshot 60 seconds ago is still reading
against that snapshot (correct behaviour). The memory cost is the
`Arc<MemtableSnapshot>` it holds. With the default 8 MiB memtable
threshold the worst case is ~8 MiB per in-flight slow reader.

We document the upper bound and ship a metric
(`namidb_active_snapshot_versions`) that exposes how many distinct
`OwnedSnapshot` Arcs are alive. Operators with very long-running
reads can watch this and tune `--max-snapshot-age` (a future flag
that times out queries past a budget).

## Alternatives considered

### A. `tokio::sync::RwLock<WriterSession>`

Drop the `Mutex` and use `RwLock`. Reads grab `.read().await`,
writes `.write().await`.

**Pro:** one-line change in the server, no new types.
**Con:** the read lock is still held for the entire query. A long
read blocks the next write indefinitely. A long write blocks every
pending read. The lifetime issue with `Snapshot<'mt>` doesn't go
away; the snapshot still borrows the writer.

Rejected because it doesn't actually solve the queueing problem,
just shifts it.

### B. Clone the writer for every read

Each read clones the entire `WriterSession` and runs against the
clone. No locks anywhere.

**Pro:** zero coordination.
**Con:** `WriterSession` owns LSN allocators, WAL state, the
property index cache being lazily filled, etc. Cloning the whole
thing for every read is expensive and introduces N-versions of
state we'd otherwise compute once.

Rejected because it's wasteful.

### C. Persistent `im::OrdMap` for the memtable

Use `im` (`OrdMap<K, V>`) instead of `BTreeMap`. The memtable is
persistent, so a snapshot is `Arc<OrdMap>` without a copy. Every
write produces a new `OrdMap` in O(log n) without affecting the
old.

**Pro:** snapshot construction is O(1) instead of O(memtable_size).
**Con:** new dependency, ~20% slower inserts than `BTreeMap` for
the hot path (microbenches), and we'd need to migrate every site
that touches the memtable.

Deferred. The current proposal copies the BTreeMap on
`commit_batch`, which happens once per write batch (not once per
mutation). For a 100K-entry memtable at the flush threshold the
copy is ≈300 µs on an M-series laptop — measured, not estimated.
We'd revisit `im` once we see commit_batch latency become the gate.

### D. `arc-swap` instead of `Mutex<Arc<...>>`

`arc-swap` 1.7 gives lock-free RCU semantics for the snapshot
publish step. Each `load` is one atomic pointer load; each `store`
is one swap.

**Pro:** no contention between `load` callers; better than the
mutex when read fan-out is huge.
**Con:** new dependency. The `Mutex<Arc<...>>` mutex is held for
~50 ns; under 16-way read fan-out that's < 1 µs total queueing per
read.

Deferred. The mutex is the right starting point; we move to
`arc-swap` if a `flamegraph` flags the lock as a hot spot.

## Drawbacks

1. **Memtable copy on every commit_batch.** Each successful
   `commit_batch` clones the entire BTreeMap. For typical
   write-light workloads (graph analytics over a mostly static
   graph) this is invisible. For write-heavy workloads it adds
   ~300 µs per commit at 100K entries. Mitigation: batching at the
   API level (callers already do this for WAL efficiency) reduces
   the commit rate.

2. **Per-snapshot memtable memory.** A reader holding an
   `OwnedSnapshot` for 60 seconds pins ~8 MiB. A pathological case
   would have 100 concurrent slow readers each holding their own
   snapshot version. Mitigation: bound query lifetime via a
   query-timeout flag (separate RFC; today the executor doesn't
   honour one).

3. **Two parallel snapshot types (`Snapshot<'_>` and
   `OwnedSnapshot`).** Risk of behaviour drift across two
   parallel implementations. Mitigation: `OwnedSnapshot::borrow()`
   re-packages into the existing `Snapshot<'_>` so the body of
   every read method is exactly one implementation. Only the
   memtable lifetime differs.

4. **Stale reads are now an observable phenomenon.** A reader
   that picks up snapshot V then waits 100 ms before issuing its
   first query against it sees data as of V even if V+5 has been
   committed in the meantime. This is correct (snapshot isolation)
   but new in shape, since today the writer mutex meant the
   reader picked up the very latest manifest at query start.
   Mitigation: document; expose `Snapshot.manifest_version` in the
   Bolt `SUCCESS { bookmark }` so observability tools can see the
   version each session is on.

## Open questions

- **Q1: When to re-publish on a long-running write batch?** Today
  a write commits in one shot. If/when we add streaming inserts
  spanning several Cypher statements (RFC-pending), do we
  re-publish on every internal commit boundary, or only at the
  session edge? Leaning per-commit, so a tail of small batches
  doesn't pile up unpublished data.

- **Q2: Snapshot eviction.** If a Bolt session times out and the
  client never sends GOODBYE, the snapshot Arc is held by the
  task until the task is reaped. Do we add an explicit
  `max-snapshot-age` knob, or rely on TCP timeouts? Leaning on
  TCP keepalive for v0, explicit knob later.

- **Q3: Migrating the existing `Snapshot<'_>`.** Once
  `OwnedSnapshot` lands, the borrowed shape becomes redundant for
  every caller that has a writer in hand. Worth a follow-up RFC
  that deletes it (everything goes through Arc), or keep both for
  in-process embedded callers? Leaning keep both for now — the
  embedded API gets an extra `borrow()` line to drop.

## References

- RFC-001 (storage engine, single-writer per namespace)
- RFC-018 (CSR adjacency cache, cross-snapshot)
- RFC-019 (NodeView cache, cross-snapshot)
- RFC-020 (edge SST caches, cross-snapshot)
- RocksDB column-family snapshots —
  https://github.com/facebook/rocksdb/wiki/Snapshot
- `arc-swap` crate (future migration target) —
  https://docs.rs/arc-swap/
