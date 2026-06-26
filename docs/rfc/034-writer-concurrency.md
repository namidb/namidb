# RFC 034: Writer concurrency — the single-writer-per-namespace mutex

**Status:** draft
**Author(s):** Matías Fonseca <info@namidb.com>
**Created:** 2026-06-26
**Updated:** 2026-06-26
**Implements:** writer-lock-wait/held instrumentation (proposed, behaviour-preserving); group/pipelined commit (proposed)
**Back-refs:** RFC-001 §"Write path"/§"Epoch fencing", RFC-021, RFC-026, RFC-027, RFC-029

## Summary

Every write to a namespace serializes on one `tokio::sync::Mutex<WriterSession>`
(`crates/namidb-server/src/lib.rs:161`, `AppState::writer`). That mutex is held
across the commit's object-store IO — two round-trips on the critical path
(`crates/namidb-storage/src/ingest.rs:665`, `WriterSession::commit_batch`) — so
per-namespace write throughput is bounded above by roughly `1 / (commit RTT +
any in-lock read IO)`, regardless of how many connections push writes. This RFC
(a) characterizes the bottleneck precisely against the code, being explicit that
it is **architecturally real but quantitatively unmeasured today**; (b) fixes the
read-your-own-writes guarantee (RFC-026, RFC-021) and the single-writer / epoch-
fencing invariant (RFC-001) as hard constraints any redesign must preserve;
(c) recommends a low-risk first step — instrument the lock wait/hold so the
bottleneck stops being a hypothesis — and then a group-commit redesign that
amortizes the per-commit round-trips across concurrently arriving writes without
adding a second writer. Pipelined commit and sharded/multi-writer designs are
evaluated and deferred.

## Motivation

### What is implemented today (v1.4)

Each namespace owns exactly one writer session behind one async mutex:

```rust
// crates/namidb-server/src/lib.rs:161
pub struct AppState {
    pub writer: Arc<Mutex<WriterSession>>,
    pub snapshot: Arc<SnapshotCell>,
    ...
}
```

Three code paths take that mutex, and every write in the system goes through one
of them:

- **HTTP auto-commit, single-tenant** — `lib.rs:1625` takes the lock, runs
  `execute_write_with_deadline`, republishes the snapshot, then drops the lock.
- **HTTP auto-commit, multi-tenant** — `lib.rs:2047` (`ns_state.writer.lock()`),
  identical shape against the per-namespace state.
- **Bolt auto-commit** — `crates/namidb-server/src/bolt.rs:437`, same shape.
- **Bolt explicit transaction** — `begin_tx` takes the lock with `lock_owned()`
  and holds it for the *entire* transaction, across RUNs and client think-time,
  until `COMMIT`/`ROLLBACK` (`bolt.rs:703`, `bolt.rs:748`).

The server deliberately forbids a *second* `WriterSession`: even background
compaction goes through the one writer lock rather than opening its own session,
because a second session would claim a new epoch and fence the foreground writer
(`lib.rs:620`, comment "the ONE writer lock — never a second `WriterSession`").
This is not an accident of the server; it is the storage engine's single-writer
invariant (RFC-001 §"Epoch fencing").

**Reads take no writer lock.** The read path borrows a published `OwnedSnapshot`
from the `SnapshotCell` and runs entirely lock-free (`lib.rs:1671-1682`, RFC-021).
So this bottleneck is a *write*-throughput ceiling only; read concurrency is
unaffected.

### The critical section is IO-bound, and the lock is held across the IO

The auto-commit critical section (`lib.rs:1625-1639`) is, in order, under the
lock the whole time:

1. `execute_write_with_deadline(&plan, &mut writer, …)`
   (`crates/namidb-query/src/exec/writer.rs:112`). This calls
   `execute_write_staged_with_deadline` (`writer.rs:77`) to stage the mutations,
   then `writer.commit_batch()` (`writer.rs:129`).
   - **Staging is purely in-memory**: `upsert_*` / `tombstone_*` push into
     `pending` / `pending_payloads` via `append_pending`
     (`ingest.rs:507-602`, `ingest.rs:1428` region). No IO.
   - **A read sub-plan inside the same statement can do object-store IO**: a
     `MERGE`/`MATCH`/constraint-check phase reads through `overlay_snapshot`
     (`ingest.rs:449`), which serves the committed snapshot plus the staged
     overlay and can issue SST GETs. This IO also happens under the writer lock.
2. `commit_batch` (`ingest.rs:665-798`) — the IO-bound core. On the critical
   path it issues **two object-store round-trips**:
   - `tokio::join!(WAL append PUT, manifest body PUT)` (`ingest.rs:693-697`),
     pipelined into one round-trip (`max(WAL, body)` rather than their sum — the
     WAL path is fully determined by `seq`, so the body can be built and PUT
     before the WAL lands, see the comment at `ingest.rs:683-690`);
   - then the pointer CAS PUT, `cas_pointer` (`ingest.rs:706`), which is the
     linearization point (RFC-029: a `PutMode::Create` of `pointer/p<N>.json`).
   The method's own doc measures the round-trip cost: **~750 ms against
   Cloudflare R2 from a laptop, ~5–15 ms same-region EC2**
   (`ingest.rs:615-616`). On loopback / the in-memory store it is invisible.
3. `state.snapshot.store(writer.owned_snapshot())` (`lib.rs:1634`). This is the
   read-your-own-writes republish and it is **cheap** — `owned_snapshot`
   (`ingest.rs:391`) clones a handful of `Arc`s and the manifest, no IO — but it
   is correctly done *in-lock* (see constraints below).

There is exactly one avoidable in-lock PUT, and it is off by default: the
auto-snapshot tick (`ingest.rs:771-783`) clones the whole memtable and PUTs it
inside the lock, but `DEFAULT_AUTO_SNAPSHOT_EVERY = 0` (`ingest.rs:82`) disables
it, so it is not on the default critical path. The soft write-stall sleep is
already correctly moved *out* of the lock — the stall decision is sampled in-lock
and the `sleep` runs after `drop(writer)` (`lib.rs:1635-1646`), so backpressure
throttles the request, not the mutex other connections need.

### Commit is not group-committed

One HTTP POST = one statement = one `commit_batch` = one WAL segment + one
manifest CAS. There is no cross-request batching anywhere in the server. RFC-001's
write-path diagram lists a step "Buffer in WAL batcher (group commit, 100ms or
1MB)" (`docs/rfc/001-storage-engine.md:193`) and names group commit as a
mitigation for the PUT-latency floor (`001:304`), but that batcher was never
implemented at the request level. The engine deliberately leaves commit cadence
to the caller (`ingest.rs:610-635`, the "Cadence trade-off" doc), and the server's
caller commits once per request. The result: under N concurrent writers to one
namespace, the commits run strictly back-to-back, each paying its own two
round-trips; the N-th writer waits behind `N-1` full commit IOs.

### Where the manifest CAS sits

The CAS is the *last* IO in the critical section (`ingest.rs:706`,
`cas_pointer`), after the pipelined WAL+body PUT. It is the true serialization
point of the storage protocol — whoever creates `pointer/p<N>.json` first owns
version `N` (RFC-029). Everything before it in the critical section (staging,
read sub-plan IO, the WAL+body PUT) does **not** logically require the global
lock to be a single point; only the CAS does. That observation is what the
redesign options below exploit.

### Throughput ceiling, and what is measured

Per namespace, sustained write throughput is bounded above by approximately:

```
throughput  ≈  1 / (commit_RTT + in_lock_read_IO)
```

On object stores with high PUT latency (R2-from-laptop ~750 ms) this ceiling is
low and real. On same-region object storage (~5–15 ms) it is far higher but still
finite and still a single-lane queue.

**This is the honest part: the ceiling is architecturally real but
quantitatively unmeasured.** There is no writer-lock-wait or lock-hold metric in
the server today — `Metrics` (`crates/namidb-server/src/metrics.rs:158`) carries
only per-protocol read/write *latency* histograms (`metrics.rs:140-141`), which
fold lock-wait time, commit IO, and execution together and cannot tell us whether
a slow write is slow because it waited behind other writers or because its own
commit IO was slow. We have a per-RTT cost figure from the engine doc and the
structural argument above; we do **not** have a measured p50/p99 of time-spent-
waiting-on-the-writer-mutex under real concurrency. Any investment in the
redesign below should be gated on that measurement.

## Design

This RFC is split into a behaviour-preserving step that should land first, the
hard constraints any redesign must hold, and the proposed redesign. The current
state of the code is "Implemented today (v1.4)" above; everything in this section
is **proposed**.

### Proposed — immediate, low-risk: measure the bottleneck

Before changing the locking model, make the bottleneck observable. This is
additive, behaviour-preserving, and gates whether the rest of the RFC is worth
building.

1. Add two histograms to `Metrics` (`metrics.rs:158`), reusing the existing
   `Histogram` type (`metrics.rs:72`):
   - `writer_lock_wait` → exposed as `namidb_writer_lock_wait_seconds` — time a
     write spent *waiting to acquire* the writer mutex.
   - `writer_lock_held` → exposed as `namidb_writer_lock_held_seconds` — time the
     mutex was *held* by an auto-commit write (acquire → `drop`).
   Render both through the existing `Histogram::render_into`
   (`metrics.rs:113-132`), which already emits Prometheus-safe cumulative
   `_bucket`/`_sum`/`_count` lines. Exposition-only additions are backward
   compatible.

2. Wrap each auto-commit acquisition site:

   ```rust
   let t = Instant::now();
   let mut writer = state.writer.lock().await;
   metrics.observe_writer_lock_wait(t.elapsed());
   let held = Instant::now();
   // … execute_write_with_deadline, snapshot.store …
   drop(writer);
   metrics.observe_writer_lock_held(held.elapsed());
   ```

   at `lib.rs:1625` (single-tenant), `lib.rs:2047` (multi-tenant), and
   `bolt.rs:437` (Bolt auto-commit). Cost is two `Instant::now()` plus one
   histogram add per write — negligible against a commit round-trip.

3. The Bolt explicit-transaction lock (`bolt.rs:703`, `lock_owned`) holds the
   mutex for the whole transaction including client think-time, which is a
   different phenomenon (already bounded by `tx_idle_timeout`). Record its *wait*
   in the same `writer_lock_wait` histogram, but expose its *hold* separately as
   `namidb_writer_tx_lock_held_seconds` so a long-held transaction lock does not
   masquerade as a slow commit.

This turns "potential bottleneck (unmeasured)" into a graphable p50/p99. **Do
this first.** If the measured `writer_lock_wait` p99 is small under production
concurrency, the redesign is not worth its risk.

### Hard constraints any redesign must keep

Three invariants are load-bearing and a redesign that breaks any of them is
wrong, not faster:

1. **Read-your-own-writes within a statement/transaction (RFC-026).** A read
   sub-plan inside a write statement reads through `overlay_snapshot`
   (`ingest.rs:449`), which layers `pending_payloads` over the committed
   snapshot. The intra-batch unique check and `MERGE`-after-`CREATE` semantics
   depend on a write seeing *its own* staged rows — and on *not* seeing another
   in-flight write's unACKed staged rows. The overlay's caches are deliberately
   not the cross-session manifest-keyed caches precisely to avoid leaking staged
   rows to concurrent readers (`ingest.rs:444-448`). Any design that shares one
   `pending_payloads` buffer across multiple concurrent stagers must preserve
   this isolation.

2. **Cross-POST read-your-own-writes (RFC-021).** A write republishes the
   snapshot in-lock (`lib.rs:1634`, `bolt.rs:447`, `commit_batch`'s
   `refresh_published` at `ingest.rs:757`) *before* it returns 200, so a *later*
   POST that reads `state.snapshot.load()` (`lib.rs:1671`) is guaranteed to see
   the committed write. A redesign must not ACK a write before its effects are
   both durable and visible in the published snapshot.

3. **Single writer enforced by epoch fencing + manifest CAS, no consensus
   (RFC-001 §"Epoch fencing").** There is exactly one live `WriterSession` per
   namespace. A second session bumps the epoch and fences the first; the engine
   relies on this instead of Raft/ZooKeeper/file locks. A redesign must keep
   "one writer session, one epoch" — it may add *concurrency around* the session,
   not a second session.

### Proposed — primary recommendation: group commit

Amortize the per-commit round-trips across writes that arrive concurrently,
keeping **one** `WriterSession` (so the single-writer + epoch-fencing invariant
is untouched). Split the critical section into a cheap, serialized *stage* phase
and an expensive, batched *commit* phase — reusing the stage/commit split that
already exists in `execute_write_staged_with_deadline` (`writer.rs:77`) vs the
`commit_batch` call in `execute_write_with_deadline` (`writer.rs:129`).

Sketch:

- A request takes a short stage-lock, runs the statement to completion against
  the writer — staging its mutations into `pending`/`pending_payloads` and
  running any read sub-plan — records its last LSN (`pending.last_lsn()`,
  `ingest.rs:680`), registers a `oneshot` waiter, and releases the stage-lock.
  Crucially the request stages **and reads to completion** under the stage-lock
  before its rows are merged into the group buffer, so its `MERGE`/constraint
  reads never observe another request's staged-but-unACKed rows (preserves
  constraint 1).
- A leader/committer drains the accumulated pending batch, issues **one**
  `commit_batch` (one merged WAL segment + one pointer CAS), republishes the
  snapshot once (`refresh_published`), then wakes every waiter whose
  `last_lsn ≤ committed_last_lsn` with `Ok` (preserves constraint 2: nobody is
  woken until the merged commit is durable and the snapshot republished).
- Knobs mirror RFC-001's never-built "100ms or 1MB": `group_commit_window`
  (e.g. ≤ 5 ms) and `group_commit_max_bytes`. A window of `0` reproduces exactly
  today's per-request behaviour, which makes it a safe default during rollout and
  a kill-switch if a regression appears.

Failure semantics (must be documented, not hidden): a terminal `commit_batch`
failure poisons the session (`ingest.rs:670-674`, `ingest.rs:734-736`). For a
group that means **all** waiters in the group receive the same error and the
group rolls back together — there is no per-request partial durability. This is
acceptable and is the only sane behaviour for a shared WAL segment, but callers
must understand grouped writes share fate.

Why this is the recommendation: it batches *exactly* the expensive part (the
commit IO), leaves the overlay/RYOW semantics unchanged by staging-and-reading
each request to completion before merge, and keeps one writer session and one
epoch. It is the in-process realization of "many writers" that does not violate
the invariant — many stagers, one committer.

### Proposed — alternative: pipelined commit

Let request N+1 stage while request N's commit IO is in flight, serializing only
at the `cas_pointer` linearization point (`ingest.rs:706`). This lowers the
latency *floor* relative to group commit (no `group_commit_window` added to a
lone write) but is more invasive: the WAL `seq`/LSN allocation
(`ingest.rs:679`, `ingest.rs:750-753`, `next_wal_seq`) must be reservable ahead
of an in-flight CAS, and the memtable drain (`ingest.rs:746-749`) must stay
strictly ordered behind the CAS that authorized it. It touches `commit_batch`'s
`self.pending`/`next_wal_seq` bookkeeping directly. Pick this over group commit
only if the instrumentation shows the pain is single-write *latency* under low
concurrency rather than *throughput* under high concurrency.

### Implementation order

1. Land the instrumentation (immediate, low-risk).
2. Measure `writer_lock_wait` p50/p99 under representative concurrency on the
   target object store.
3. If contention is real, build group commit with `group_commit_window` defaulting
   to `0` (no behaviour change), then tune the window up under measurement.
4. Consider pipelined commit only if latency, not throughput, is the residual
   problem.

## Alternatives considered

- **Multiple true `WriterSession`s serialized only at the manifest CAS.**
  Rejected: a second session claims a new epoch and fences the first
  (RFC-001 §"Epoch fencing"), which is exactly what `lib.rs:620` forbids in the
  server. In-process "many independent writers" inevitably collapses to "many
  stagers, one committer" — which *is* group commit. There is no correct version
  of this that keeps one epoch and multiple sessions.

- **Sharded writers by key range (one writer/manifest per shard).** Rejected for
  v1. It breaks the single-manifest-per-namespace model (RFC-001), loses
  cross-shard statement atomicity (a statement spanning two shards needs
  two-phase commit), and has the largest blast radius of any option. It aligns
  with the key-range-partitioned leveled compaction that RFC-027 explicitly
  defers (`027` "Implements" line: "key-range-partitioned leveled compaction
  remains a follow-up"), so if it is ever built it should be co-designed with
  that. Genuinely multi-master writes are a stated v1 non-goal and probable v2
  (RFC-001 §"Drawbacks" item 2, §"Open questions").

- **Do nothing.** Cheapest, and defensible *if* the bottleneck never bites the
  target workloads (analytical / KG / RAG, which RFC-001 §"Drawbacks" notes are
  fine with single-writer). But "do nothing" without the instrumentation means we
  keep guessing. The instrumentation alone is the minimum responsible action.

- **Cross-process multi-writer with merge/CRDT semantics.** Out of scope; RFC-001
  §"Open questions" parks this as "probably v2".

## Drawbacks

- **Group commit adds up to `group_commit_window` of latency to a lone write.**
  Mitigation: an adaptive window (0 when no contention is observed), and the
  default of `0` ships the redesign as a no-op until deliberately tuned.

- **Grouped writes share fate.** A poisoned session
  (`ingest.rs:670-674`) fails the whole in-flight group, not one request. This is
  inherent to a shared WAL segment and must be documented as part of the write
  contract, not smoothed over.

- **More moving parts in the commit path.** A leader/committer + waiter set is
  more complex than one mutex. The `window=0` escape hatch and the existing
  RYOW/epoch tests (below) are the guardrails that keep the complexity honest.

- **The instrumentation itself is the only thing that is unambiguously safe.**
  It adds exposition lines and two timers per write; everything past it is a real
  concurrency change and should not be rushed ahead of the measurement it exists
  to provide.

## Open questions

- What is the measured `writer_lock_wait` p99 on same-region object storage under
  realistic concurrency? Until this is known, the redesign priority is unsettled.
- Should `group_commit_window` be static config or adaptive (auto-0 when the wait
  histogram shows no contention)?
- Does the Bolt explicit-transaction model (whole-transaction lock,
  `bolt.rs:703`) benefit from any of this, or is it intentionally a serialized,
  think-time-bounded path that group commit should simply not touch? Current lean:
  leave explicit transactions serialized; group commit targets the auto-commit
  paths only.
- Should the in-lock auto-snapshot PUT (`ingest.rs:771-783`) be reworked to
  snapshot-data-in-lock then PUT-after-release before it is ever enabled by
  default? Default-off today, so no action required now, but it is the one
  remaining avoidable in-lock IO.

## Test plan

- **Instrumentation (lands with step 1).** A `metrics.rs` unit test mirroring the
  existing `histogram_buckets_are_cumulative_with_sum_and_count` style: assert
  `namidb_writer_lock_wait_seconds_count` increments once per write and `_sum` is
  monotonic; assert a *read* query does **not** bump the lock-wait histogram
  (proves reads stay lock-free, RFC-021).
- **Cross-POST RYOW regression (must stay green for any redesign).** POST a
  `CREATE`, await 200, POST a `MATCH`, assert the row is present *and* that the
  second request planned against the republished snapshot rather than a flat
  fallback — assert the lowered plan actually consults the index/snapshot, not
  just that results are equal.
- **Group commit (when built):**
  - *Durability/ACK ordering:* N concurrent writes, fault-inject the CAS to fail
    for the group; assert all N see the error and none observe each other's rows
    after the failure (no partial durability).
  - *Isolation under RFC-026:* two concurrent `MERGE`s on the same unique value →
    exactly one node, not two.
  - *Throughput:* against the in-memory store with an injected per-PUT delay,
    assert grouped commits issue fewer `put`/`cas` calls than ungrouped for the
    same row count (count store ops, not wall-clock).
- **Bolt transaction unaffected.** The existing writer-lock-leak regression and
  the `tx_idle_timeout` test must still pass.

## References

- RFC-001 (storage engine) §"Write path" (`001:185-200`, the never-built group
  commit step), §"Epoch fencing" (`001:160-185`), §"Drawbacks"/§"Open questions"
  (single-writer, multi-master as v2).
- RFC-021 (concurrent reads) — lock-free reads off the published `SnapshotCell`;
  the cross-POST RYOW republish.
- RFC-026 (read-your-own-writes) — the `pending_payloads` overlay
  (`ingest.rs:449`) that any group-commit design must keep isolated.
- RFC-027 (compaction and space reclamation) — the soft write-stall sampled
  in-lock and slept out-of-lock (`lib.rs:1635`); the deferred key-range-
  partitioned compaction the sharded-writer alternative would have to co-design
  with.
- RFC-029 (create-only versioned pointer) — the `cas_pointer` linearization point
  (`ingest.rs:706`) that is the true serialization boundary the redesign exploits.
