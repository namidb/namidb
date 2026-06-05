# RFC 026: Read-your-own-writes within a statement and transaction

**Status:** accepted
**Author(s):** Matías Fonseca <info@namidb.com>
**Created:** 2026-06-05
**Updated:** 2026-06-05
**Implements:** (pending)
**Supersedes:** none

## Summary

Make reads inside a write context see the writes that ran before them.
Today every read sub-plan and every in-transaction read runs against the
last committed snapshot, so a node a query just created is invisible to a
later clause in the same statement, and a statement in an open transaction
cannot see what an earlier statement in that transaction staged. Standard
Cypher idioms (`CREATE` then `MATCH`, `MERGE` after `CREATE`, multi-step
transactions) silently return wrong or missing rows.

The fix is a pending overlay: a read-only view layered on top of the
committed snapshot that resolves the writer's staged-but-uncommitted
mutations first. Reads in a write context use the overlay; reads outside a
write context keep using the plain committed snapshot.

## Motivation

The write executor documents the gap in its own header
(`crates/namidb-query/src/exec/writer.rs`): "No read-your-own-writes: each
read sub-plan sees the pre-call snapshot. Pieces created mid-query are not
visible until commit." The Bolt backend documents the same for
transactions (`crates/namidb-server/src/bolt.rs`, `run_in_tx`): "an in-tx
read does NOT see the tx's own staged writes."

Concrete failures a user hits:

1. **Create then match in one statement.**
   `CREATE (a:Person {id: 1}) WITH a MATCH (p:Person {id: 1}) RETURN p`
   returns zero rows. The `MATCH` reads the committed snapshot, which does
   not yet contain `a`.

2. **Merge after create in one transaction.**
   ```
   BEGIN
   CREATE (a:Person {id: 1})
   MERGE (b:Person {id: 1})   // should match a, instead creates a duplicate
   COMMIT
   ```
   `MERGE`'s match phase scans the committed snapshot, misses the staged
   `a`, and creates a second node. This also blocks the intra-batch half of
   the unique-constraint work in RFC follow-ups: the constraint check in
   `apply_create` reads the committed snapshot, so two creates of the same
   unique value in one uncommitted batch both pass.

3. **Multi-statement transaction.** Any transaction whose later statements
   depend on earlier ones (the normal reason to open a transaction) is
   wrong.

The cost of doing nothing is that NamiDB is not correct for the most
common write-then-read Cypher patterns over the Bolt wire, which is a trust
problem more than a feature gap.

## Design

### What is already in place

`WriterSession` stages mutations in two parallel structures before a commit
(`crates/namidb-storage/src/ingest.rs`):

- `pending: WalSegment` is the queued WAL records.
- `pending_payloads: Vec<(MemKey, u64, MemOp)>` is the matching memtable
  operations in LSN order, where `MemOp` is `Upsert(bytes)` or a tombstone.

`commit_batch` drains `pending_payloads` into the live memtable only after
the manifest pointer CAS lands, and `snapshot()` reads the
`published_memtable` plus the SSTs, never the pending batch. That is why a
read in a write context cannot see staged work: the staged work is by
design absent from the snapshot until commit.

### The pending overlay

Add a read-only overlay that resolves the staged mutations on top of a
committed snapshot.

```rust
/// A read-only view of a writer's staged-but-uncommitted mutations,
/// resolved on top of a committed snapshot. Built from `pending_payloads`,
/// so it reflects exactly what `commit_batch` would make durable.
pub struct PendingOverlay {
    /// Latest staged op per key (last write wins by LSN). An `Upsert`
    /// shadows the snapshot value; a tombstone hides it.
    nodes: HashMap<NodeId, MemOp>,
    edges: HashMap<EdgeKey, MemOp>,
    /// Label and edge-type indexes over the staged upserts, so a scan can
    /// merge staged rows in without walking the whole overlay.
    by_label: HashMap<LabelId, Vec<NodeId>>,
    by_edge_type: HashMap<EdgeTypeId, Vec<EdgeKey>>,
}
```

`WriterSession` builds it from the current pending batch:

```rust
impl WriterSession {
    /// Snapshot the committed state and overlay the staged batch on top.
    /// The result resolves staged upserts and tombstones first, then falls
    /// through to the committed snapshot.
    pub fn overlay_snapshot(&self) -> OverlaySnapshot<'_> { ... }
}
```

`OverlaySnapshot` wraps the existing `Snapshot` and consults the overlay
before delegating:

- `lookup_node(id)`: if the overlay has the id, return its upsert (or
  `None` for a tombstone); otherwise delegate to the base snapshot.
- `scan_label(label)`: take the base snapshot rows, drop any the overlay
  tombstones, replace any the overlay upserts, and append staged upserts
  that the base does not have. De-duplicate by id.
- `lookup_node_by_property(label, prop, value)`: check the staged upserts
  for that label first, then delegate. This is what makes the unique-
  constraint check catch intra-batch duplicates.
- Edge and adjacency lookups follow the same overlay-then-base shape.

The base read methods already live on `Snapshot`; the overlay only adds a
resolve-staged-first step in front of each, so there is one read engine,
not two.

### Where the overlay is used

1. **Read sub-plans inside a write statement.** `execute_write_inner`
   currently delegates its read operators to the read-only walker against a
   plain snapshot. Switch those to the overlay snapshot built from the
   writer's current pending batch, so a `MATCH`/`WHERE`/`MERGE`-match that
   follows a `CREATE` in the same statement sees the created rows.

2. **Reads in an open transaction.** In `run_in_tx` the read branch reads
   `self.state.snapshot` (committed). Switch it to the transaction writer's
   overlay snapshot so statement N sees statements 1..N-1.

3. **MERGE match phase.** `apply_merge`'s `find_merge_matches` scans the
   committed snapshot; move it to the overlay so a `MERGE` matches rows the
   same statement or transaction just created.

Reads with no writer in scope (auto-commit reads, the HTTP read path, the
Bolt auto-commit read branch) keep using the plain committed snapshot.
There is nothing staged for them to see.

### Visibility ordering

Cypher's semantics for read-after-write inside one statement are subtle
(the standard separates read and write phases). For v1 we adopt the
operationally useful rule that matches user intent and Neo4j behaviour for
the common cases: a read operator sees every mutation staged by operators
that ran before it in execution order. Because `execute_write_inner` is a
depth-first row-at-a-time evaluator, "before in execution order" is
"already in `pending_payloads`," which the overlay reflects exactly. We do
not attempt the full standard read/write phase separation in v1; the open
questions track the cases where the two differ.

### Cost and lifecycle

The overlay is rebuilt from `pending_payloads` when a write statement
starts its read sub-plan, or once per in-tx read. Building it is O(pending
size), which is bounded by the batch the caller staged. For a single
`CREATE ... MATCH` the pending set is tiny. For a large transaction the
overlay grows with the transaction, which is the same memory the pending
batch already holds, so the overlay adds index structures over data already
resident, not a second copy of it.

The overlay is dropped when the read completes. It never outlives the
writer borrow, so there is no published-snapshot lifecycle to manage (it is
not an `OwnedSnapshot` from RFC-021; it is a borrowed, in-writer view).

## Alternatives considered

### A. Commit each statement inside a transaction

Auto-commit every statement so the next one reads it from the committed
snapshot. Rejected: it destroys transaction atomicity (a later `ROLLBACK`
could not undo earlier statements) and multiplies manifest CAS traffic.

### B. Apply the pending batch into the live memtable, roll back on discard

Stage directly into the live memtable and reads see it for free. Rejected:
it mixes committed and uncommitted state in the one structure the snapshot
reads, so a concurrent reader on the published snapshot could observe
uncommitted rows, and a discard would have to surgically unwind memtable
edits. The pending batch exists precisely to keep uncommitted work out of
the readable state.

### C. Persistent memtable (im::OrdMap) with a staged layer

Use a persistent map so the overlay is a cheap structural share. Rejected
for the same reasons as RFC-021 alternative C: new dependency and a slower
insert hot path. Revisit only if overlay build cost shows up in a profile.

## Drawbacks

1. **Two read entry points to keep in step.** The overlay resolve-first
   logic has to mirror every base read method. Mitigation: the overlay
   delegates to the one base implementation and only prepends the staged
   resolution, so drift is bounded to the small overlay layer.

2. **Edge and adjacency overlay is more involved than nodes.** Staged edges
   have to merge into adjacency scans, including DETACH DELETE's incident-
   edge enumeration. v1 can land nodes first and edges in a fast follow if
   the edge overlay proves large; the open questions track this.

3. **Visibility is the operational rule, not the full standard.** Queries
   that depend on the standard's read/write phase separation (rare, mostly
   pathological `SET` plus `MATCH` on the same pattern) may differ from a
   strict openCypher reading. Documented, with the TCK cases called out as
   they surface.

## Open questions

- **Q1: Node-only v1, edges in a follow-up?** Nodes unblock CREATE-then-
  MATCH, MERGE, and the intra-batch unique check. Edge overlay unblocks
  traversals over just-created edges. Leaning land nodes plus the
  property-lookup overlay first, edges immediately after.

- **Q2: Property index cache interaction.** The cross-snapshot property
  index cache (see the read path) must not serve a stale negative for a
  value the overlay has staged. The overlay's `lookup_node_by_property`
  has to check staged upserts before consulting the cached index. Confirm
  there is no path that bypasses the overlay.

- **Q3: Statement boundary rebuild vs incremental.** Rebuild the overlay
  per read, or maintain it incrementally as the write executor stages
  mutations? Leaning rebuild-per-read for v1 (simpler, correct), measure
  before optimising.

## References

- RFC-001 (storage engine, pending batch and manifest CAS)
- RFC-009 (write clauses)
- RFC-021 (concurrent reads, snapshot model)
- openCypher semantics, read/write phase separation
