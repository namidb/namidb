# RFC 027: Multi-level compaction, tombstone GC, and safe space reclamation

**Status:** accepted
**Author(s):** Matías Fonseca <info@namidb.com>
**Created:** 2026-06-05
**Updated:** 2026-06-05
**Implements:** P1 (horizon) + P2 (horizon-aware sweep) + P3 (tombstone/version GC via full-bucket merge) + P5 (reactive L0 trigger + soft write stall) landed; P4 (leveled compaction) deferred — see the follow-up note in Piece 1
**Supersedes:** none

## Summary

Bound space and read amplification on a namespace that takes sustained
updates and deletes. Today compaction only merges L0 into L1, never
re-compacts L1, never reclaims tombstones or superseded versions, and the
orphan sweeper runs in dry-run mode by default. An update-heavy or
delete-heavy workload grows object-store bytes without an upper bound, and
no operator lever brings them back down.

This RFC proposes three pieces that work together: leveled compaction past
L1, a snapshot-retention horizon that lets compaction drop tombstones and
old versions safely, and a reference-counted sweep that deletes unreferenced
objects (including compaction orphans) instead of leaking them.

## Motivation

The compactor is explicit about its limits
(`crates/namidb-storage/src/compact.rs` module docs): "Stateless L0 to L1
compaction," with non-goals "Multi-level compaction (L1 to L2, etc.). Only
L0 to L1 in v1," and a note that a deleted-row tombstone "preserved in the
L1 SST has no snapshot-retention policy so a tombstone might still be
load-bearing for a reader pinned at an old version."

The consequences:

1. **Tombstones live forever.** A delete writes a tombstone that is carried
   through every compaction and never dropped, because dropping it could
   change what a reader pinned at an old snapshot sees. Delete a million
   rows and the million tombstones stay on disk indefinitely.

2. **Old versions live forever.** An update writes a new version; the old
   value is shadowed at read time but never physically removed, for the
   same reason. A hot row updated a million times keeps a million versions.

3. **No L1 reclamation.** With only L0 to L1, the L1 footprint per bucket
   grows with the merged history and is never reorganised, so read
   amplification and bytes both trend up.

4. **Orphans leak.** A failed commit (WAL or body PUT succeeded, pointer
   CAS lost) or a superseded compaction input leaves objects the live
   manifest no longer references. The janitor that should sweep them runs
   dry-run by default, so on a real deployment they accumulate.

The net effect is that NamiDB's on-disk (on-object-store) size is a
high-water mark of everything ever written, not a function of the live data
set. For an object store billed per byte and per request, that is a direct
cost and operability problem.

## Design

### Piece 1: leveled compaction past L1

Generalise the existing `(kind, scope)` bucketing to a small number of
levels with a size ratio, in the spirit of RocksDB leveled compaction:

- L0: one SST per flush, overlapping key ranges, scanned together on read.
- L1..Lk: each level holds non-overlapping SSTs per bucket, each level
  roughly a fixed factor larger than the one above.

Compaction picks a level whose size exceeds its budget and merges it down
into the next level, rewriting only the overlapping key ranges rather than
the whole bucket. This bounds read amplification to roughly the number of
levels and lets the bytes for a bucket settle near the live data size for
that bucket instead of its full history.

The existing L0 to L1 merge becomes the first level transition; the merge
machinery (`compact_node_ssts`, `compact_edge_ssts`) is reused, with the
input set chosen by the level picker rather than always "all L0 in the
bucket."

**Status / follow-up (P4 deferred).** P1, P2, P3 and P5 have landed; P4 is
deferred. Compaction today is full-bucket (every compaction merges all of a
bucket's SSTs into one L1), which bounds space and read amplification but
trades write amplification — it rewrites the whole bucket each time. True
leveled compaction as described above needs key-range-partitioned SSTs
(multiple non-overlapping files per level), which the current one-SST-per-
bucket writer does not produce; that is its own piece of work.

The interim plan for the follow-up is **leveled-lite**: keep one SST per
`(bucket, level)` across L1..Lk with a per-level size budget
(`budget(Li) = base * ratio^(i-1)`). Most compactions merge `L0 + L1 -> L1`
(cheap); a level only cascades down (`Li + Li+1 -> Li+1`) when it exceeds
its budget, so the large base level is rewritten rarely. Tombstone GC moves
to "the merge whose output is the deepest occupied level," which is safe
because the LSM invariant (lower level number holds the newer LSN for a
key) guarantees any copy in a shallower level is newer and wins at read
time, so dropping a deepest-level tombstone can never resurrect a row.
Key-range-partitioned leveled (rewriting only overlapping ranges) is a
later step on top of that.

### Piece 2: snapshot-retention horizon

A tombstone or a superseded version can be physically dropped only once no
live reader could still need it. Define the retention horizon as the oldest
manifest version any live reader is pinned to.

RFC-021 already publishes `OwnedSnapshot`s carrying a `manifest` version and
proposes a `namidb_active_snapshot_versions` metric. Extend that into an
authoritative low-water mark:

```rust
/// Oldest manifest version any live reader is pinned to. Compaction may
/// drop tombstones and shadowed versions strictly older than this, because
/// no reader can observe them.
fn retention_horizon(&self) -> u64 { ... }
```

In a single process this is the minimum `manifest.version` over the live
`OwnedSnapshot` Arcs (plus the writer's current version). During a merge,
when several versions of a key are present, keep the newest version at or
below the horizon and every version above it, and drop the rest. A tombstone
whose newest covering version is below the horizon, with no live value above
it, is dropped entirely.

This is the standard MVCC GC rule: collect versions no active reader can
reach. It is safe by construction: a reader pinned at version V keeps the
horizon at or below V, so nothing V needs is collected.

### Piece 3: reference-counted, horizon-aware sweep

Replace the wall-clock min-age dry-run sweep with a reference-counted one.
An object is safe to delete when no manifest version at or above the
retention horizon references it.

```text
live set = union of objects referenced by every manifest version
           from retention_horizon() to current, inclusive
sweep    = (objects present in the store) minus (live set)
```

The sweeper lists the namespace, subtracts the live set, and deletes the
remainder. This covers both compaction inputs that have been merged away and
orphans from failed commits, with no time heuristic. Enable it by default,
gated behind the horizon so it can never delete an object a retained
manifest version still points at.

A dry-run mode stays available for operators who want to inspect first, but
it is no longer the default, and every run logs what it deleted (or would
delete) so a bounded sweep is never mistaken for "nothing to do."

### Piece 4: reactive compaction trigger and soft write stall

Compaction today runs on a periodic tick. Add an L0-count high-water mark
that triggers a compaction as soon as L0 for a bucket crosses it, so read
amplification does not spike between ticks under sustained writes. Pair it
with a soft write stall: when L0 grows faster than compaction drains it,
slow the writer (a short delay on `commit_batch`) rather than letting L0 and
read amplification grow without bound. Both thresholds are config, with
defaults chosen from the bench workloads.

### Interaction and ordering

The three pieces are independent but compose:

- Leveled compaction reorganises bytes and is correct on its own, but
  without the horizon it still carries tombstones and old versions forever.
- The horizon makes a merge able to drop them.
- The sweep reclaims the objects the merges leave behind, plus orphans.

A reasonable landing order is: horizon plumbing first (it is also needed by
the sweep), then the horizon-aware sweep enabled by default (immediate win
on orphan leaks), then tombstone and version GC during the existing L0 to L1
merge, then the general leveled scheme, then the reactive trigger and stall.

## Alternatives considered

### A. TTL-based tombstone GC

Drop tombstones older than a fixed wall-clock TTL (the Cassandra
`gc_grace_seconds` model). Simpler than a horizon, no reader tracking.
Rejected as the primary mechanism: it is unsafe for a reader pinned past the
TTL (it would resurrect deleted rows for that reader), and once read
replicas exist the TTL cannot know about a replica's pinned version. The
horizon is correct by construction; a TTL can be layered on later as an
upper bound on retained history, not as the safety mechanism.

### B. Size-tiered compaction

Merge SSTs of similar size into a larger one (the Cassandra STCS model)
instead of leveled. Lower write amplification, higher space amplification
(up to roughly 2x during a major compaction) and worse read amplification.
Rejected for the default because the goal here is to bound space; leveled
trades some write amplification for predictable space and read bounds, which
matches an object-store cost model. STCS stays a candidate for an
append-mostly bulk-load profile.

### C. Keep the wall-clock min-age sweep, just enable it

Flip the existing sweep to enabled and trust the min-age window. Rejected:
min-age is a guess. Too short and it can delete an object a slow reader or
an in-flight commit still needs; too long and it leaks. The reference-
counted horizon sweep has neither failure mode.

## Drawbacks

1. **Write amplification.** Leveled compaction rewrites data multiple times
   as it moves down levels. This is the standard leveled tradeoff; the
   reactive trigger and stall keep it from running away, and the object-
   store cost model rewards the smaller, predictable footprint.

2. **GC correctness is load-bearing.** Deleting a live SST corrupts reads.
   The horizon and the reference-counted live set are the safety net, and
   both need careful testing (a deliberate fault-injection suite that pins a
   reader at an old version and asserts compaction and sweep never collect
   what it can reach).

3. **Cross-process horizon is harder.** In one process the horizon is a
   local minimum. Once read replicas exist (a future RFC), the horizon must
   account for versions pinned on other processes, which needs a shared
   low-water mark (a lease or heartbeat). v1 computes the horizon locally
   and documents that read replicas depend on extending it.

4. **More moving background work.** Another scheduler dimension to tune and
   observe. Mitigated by surfacing compaction and sweep activity as metrics
   (bytes reclaimed, levels, L0 count, horizon) so it is observable rather
   than opaque.

## Open questions

- **Q1: Per-bucket leveling or global?** The current bucketing is
  `(kind, scope)`. Does each bucket get its own level hierarchy, or is
  leveling global with bucket as a partition key? Leaning per-bucket, since
  reads already scope by bucket.

- **Q2: How aggressive should version GC be?** Drop every shadowed version
  below the horizon, or keep a small history for future point-in-time reads
  or time-travel? Leaning drop-all-below-horizon for v1; a retained-history
  knob is a separate feature.

- **Q3: Sweep cadence and listing cost.** A full namespace list per sweep is
  an object-store cost. Run it on the compaction tick, or less often? Can the
  manifest carry enough to avoid a full list? Leaning piggyback on
  compaction and measure the list cost on a real backend.

- **Q4: Horizon for very long readers.** A reader pinned for minutes holds
  the horizon down and stalls reclamation. This is the same stale-snapshot
  tension RFC-021 raises; a max-snapshot-age (query timeout) bounds it. Track
  alongside the query-timeout follow-up.

## References

- RFC-001 (storage engine, manifest, single writer)
- RFC-002 (SST format, tombstone column)
- RFC-021 (concurrent reads, snapshot versions, the basis for the horizon)
- RocksDB leveled compaction and compaction styles
- MVCC garbage collection (collect versions below the oldest live read)
