# Design: Search 3.5 — fresh, incremental, quantized search on object storage

**Status:** in progress (S1 and part of S2 shipped in 3.5.0; see "Implementation status")
**Author(s):** NamiDB team
**Created:** 2026-11-17
**Targets:** NamiDB 3.5.x (branches from the 3.4.3 release)
**Amends:** [`docs/architecture/search-lsm.md`](../architecture/search-lsm.md),
RFC-030 (vector index), RFC-035 (incremental `.vg` maintenance)
**Scope:** `namidb-ann`, `namidb-storage` (Search-LSM flush / compaction / read,
vector and text segment formats, manifest, range cache), `namidb-query`
(count rewrite, vector / BM25 / hybrid execution), `namidb-server`
(maintenance scheduling, prewarm), `namidb-bench` (measurement)

## Implementation status (3.5.0)

3.5.0 ships the parts of S1 and S2 that need no format change. Everything
below is in memory and rebuilt after a restart, so a 3.5.0 store opens with
3.4.x and back.

| Step | Shipped in 3.5.0 | Not yet |
|---|---|---|
| S1 counts | Exact per-label and per-`(label, property)` presence ledger, maintained by every commit and flush; seeded in the background (at open for vector-indexed labels, otherwise on the first count) and rolled forward over the commits that land during the seed; `count(n)`, `count(n) WHERE n.p IS NOT NULL`, `count(n.p)` answered from it | Persisting the ledger in the manifest (§5.7.1); after a restart the first count of an untracked property waits for a seed |
| S2 vector freshness | Per-index int8 memtable tail with exact score bounds (§5.2.6 with int8 instead of RaBitQ codes); only rows that can reach the top-k are decoded; adaptive over-fetch instead of `k + \|tail\|`; resident int8 codes for VG6 deltas so only pages that can hold the top-k are read | Recovery-time tail rebuild is lazy (first query after open or flush), not at replay |
| S2 BM25 freshness | Rows whose label membership and indexed text equal the persisted version no longer force the flat scan (the embedding-only `SET` case of G6) | A text tail for rows whose indexed text did change (§5.5.1): those still take the exact path until flushed |
| Filters (part of S7) | A property-only filter the index cannot apply that admits at most 4,096 nodes is evaluated on its own columns and only those nodes are scored exactly (the index path hands over after one under-filled round) | Native per-leaf filter bitmaps; filtered search under sustained overlapping writes |
| S3–S9 | — | Streaming backfill, VG7 partitioned RaBitQ index with SPFresh maintenance, cold path, FT5 blocked postings, hybrid, storage bounds |

Measured on the live-workload bench (M shape, 120k documents, 100k
embeddings, then 80k more embedding writes), 3.4.2 → 3.5.0, p50: presence
count settled 7.2 s → 7 ms; filtered vector top-10 settled 21.5 s (3/10
timeouts) → 3.0 s; BM25 under embedding writes 23.6 s (10/15 timeouts) →
1.5 s; a fresh embedding found by top-1 18.9 s → 3.5 s; unfiltered vector
top-10 settled 948 → 801 ms. Under sustained re-embedding writes, vector top-k latency stays dominated
by the V5 base and its maintenance competing for the same two CPUs and the
bandwidth cap; that is what S4/S5 address.

## 1. Summary

Search-LSM (3.0–3.4) made vector and full-text indexes correct under live
writes: every `Nodes` SST is covered by a search segment or a proof that it
changes nothing, deltas carry winner tables, and any missing proof falls back
to the exact flat scan. What it did not make is *fast and fresh at the same
time*. On a 377k × 1024-d cosine corpus, under a 10–20 ms object-store
round trip, 3.4.x shows:

- an index created over existing data stays `Building` until a whole-corpus
  base exists; until then every query is an exact flat scan of every embedding
  (about 1.5 GB), and the base build takes minutes;
- once `Active`, flat VG6 deltas are scanned exhaustively, so query cost grows
  with the unmerged tail;
- the clustered V5 base needs about 120 range reads and 30 MB per query for
  recall@10 0.955 and plateaus near 0.96;
- any unflushed write to a label (even an embedding-only `SET`) sends BM25 on
  that label to a two-pass tokenizing flat scan;
- `count(n) WHERE n.embedding IS NOT NULL` decodes every vector of the label;
- maintenance rebuilds whole indexes from `Nodes` SSTs, with scratch and
  memory proportional to the corpus.

This document specifies the 3.5 design that removes each of these, with
measurable targets on the live-workload bench. The design rests on four ideas,
all published and none exotic:

1. **Three freshness tiers, each bounded.** The memtable tail is searched
   exactly from an in-memory per-index structure maintained at commit; flushed
   deltas are searched through quantized codes; the settled part is searched
   through a partitioned index. A committed write is visible to the next query,
   always, with cost bounded by the memtable cap rather than the corpus.
2. **A partitioned, quantized vector index (VG7).** Balanced k-means leaves of
   about 256 vectors, a small resident centroid directory, 1-bit RaBitQ codes
   plus extended-RaBitQ refinement bits, and row-addressable exact f32 rows used
   only to re-rank candidates whose error bound can still change the top-k.
   Scores returned to the user remain exact f32.
3. **Incremental maintenance instead of rebuilds.** New vectors are assigned to
   their nearest existing leaf *at flush*, so a delta is partitioned the moment
   it is written. Leaves split, merge and reassign boundary vectors locally
   (SPFresh/LIRE). The base is a set of shards, each with its own frontier, so
   a fold rewrites only the shards with pending work. Full text gets the same
   treatment through term-range shards and posting-level merges.
4. **Counts are metadata.** An exact per-label and per-indexed-property
   presence ledger is maintained on commit and persisted at flush, so
   `count(n)` and `count(n) WHERE n.p IS NOT NULL` are O(1).

The work is split into ten independently shippable steps (§7), ordered so the
largest user-visible wins (counts, freshness, no `Building` window) ship first
and the format changes ship behind them.

## 2. Goals, non-goals and targets

### 2.1 Goals

- **G-fresh.** A committed vector or document is returned by the next search
  that should return it. No search waits for an index build to be usable.
- **G-incremental.** No hot-path whole-index rebuild for vectors or full text.
  Maintenance cost and scratch are proportional to the change set, not the
  corpus. A global re-cluster exists only as a background, drift-triggered,
  shard-by-shard operation.
- **G-latency.** Vector, BM25 and hybrid targets in §2.3.
- **G-count.** Label counts and indexed-property presence counts from metadata.
- **G-writes.** Writes never wait for index maintenance; memory stays bounded
  in a 1 GiB engine; storage does not accumulate unreclaimed index artifacts.

### 2.2 Non-goals

- Graph-based ANN (DiskANN/Vamana, HNSW) as the primary index on object
  storage. A graph search is a chain of dependent reads whose length grows with
  the corpus; a cluster index needs a fixed number of round trips. RFC-030's
  Vamana path stays for the legacy `.vg` bodies only.
- Approximate counts. Every count answered from metadata is exact or the query
  takes the existing exact path.
- Changing BM25 semantics. Default BM25 stays globally exact (global `N`,
  `avgdl`, `df`) as specified in `search-lsm.md`.
- Arbitrary-predicate counts in O(1). Only label counts and presence of
  properties that carry an index are covered (§5.7); other predicates get a
  streaming, non-materialising count but no latency target.

### 2.3 Targets

All targets are measured with `namidb-bench` (§8) on a corpus of 386k
`Document` nodes with 1–4 KB of text, four full-text indexes on that label,
and 377k distinct 1024-d cosine embeddings written by 300-row
`UNWIND … MATCH … SET a.embedding = r.v` batches. The store is the bench's
simulated object store: 10–20 ms per request, 100 MiB/s bandwidth cap.

Two engine shapes:

| Shape | cgroup | CPU | `NAMIDB_CACHE_MAX_BYTES` | local disk cache tier |
|---|---|---|---|---|
| **M** | 4 GiB | 2 | 1 GiB | 8 GiB |
| **S** | 1 GiB | 1 | 256 MiB | 4 GiB |

Definitions:

- **warm**: the query's index pages have been read at least once by the same
  engine process (RAM or local-disk tier).
- **cold**: engine restarted, RAM and local-disk caches emptied, manifest open
  completed before timing starts. Index footers are *not* preloaded.
- **settled**: no writes for 60 s and maintenance idle.
- **under writes**: the embedding phase is running at its normal rate (≥ 300
  vectors/s) and queries are issued concurrently at 2 QPS.
- **recall@10**: against brute-force exact top-10 over the vectors visible in
  the same snapshot, averaged over 200 queries.
- Latencies are end-to-end HTTP query latencies as the client sees them.

| # | Metric | M (4 GiB / 2 CPU) | S (1 GiB / 1 CPU) | 3.4.x today (M) |
|---|---|---|---|---|
| T1 | Commit-to-searchable (vector and BM25) | next query; 100% of just-committed probe vectors found in top-10 | same | until base rebuild (minutes) for `Building`; flat scan otherwise |
| T2 | Longest window with no usable index (vector p95 > 1 s) | 0 s after index creation over existing data completes backfill; backfill ≤ 3 min | backfill ≤ 8 min | 10–15 min, never under sustained writes |
| T3 | Vector top-10, settled, warm | p50 ≤ 50 ms, p95 ≤ 150 ms | p50 ≤ 100 ms, p95 ≤ 300 ms | p50 422 ms |
| T4 | Vector top-10, cold | p50 ≤ 500 ms, p95 ≤ 900 ms | p50 ≤ 800 ms | p50 627 ms (ef 64) |
| T5 | Vector top-10, under writes | p50 ≤ 100 ms, p95 ≤ 300 ms | p50 ≤ 200 ms, p95 ≤ 600 ms | p50 5.6 s with many deltas |
| T6 | Vector recall@10 (settled and under writes) | ≥ 0.95 | ≥ 0.95 | 0.955 settled, 0.893 with deltas |
| T7 | BM25 top-10, settled, warm | p50 ≤ 50 ms, p95 ≤ 150 ms | p50 ≤ 100 ms | 132–1,873 ms by label |
| T8 | BM25 top-10, under (embedding-only) writes | p50 ≤ 100 ms | p50 ≤ 200 ms | O(corpus) flat scan |
| T9 | Hybrid (vector + BM25 + one equality filter, RRF), warm | p50 ≤ 150 ms | p50 ≤ 300 ms | not measured |
| T10 | `MATCH (n:L) RETURN count(n)` and `… WHERE n.p IS NOT NULL` for an indexed `p` | ≤ 100 ms cold and warm, exact | same | > 30 s for the presence count |
| T11 | Vector write batch (300 rows) | p95 ≤ 2 s, max ≤ 10 s, 0 memory-pressure rejections | p95 ≤ 4 s, max ≤ 20 s | p95 1.95 s, max 13 s (3.4.2 candidate) |
| T12 | Write p95 with maintenance running vs. idle | ≤ 1.25× | ≤ 1.5× | not bounded |
| T13 | Peak cgroup memory | ≤ 80% of limit | ≤ 85% | — |
| T14 | Largest single maintenance job | ≤ 60 s, scratch ≤ 512 MiB | ≤ 120 s, scratch ≤ 256 MiB | 661 s, 6.7 GB scratch |
| T15 | Store bytes after the GC horizon | ≤ 1.5× live referenced bytes | same | ~6× |

The S-shape latency targets are relaxed because it has half the CPU and a
quarter of the RAM page tier; freshness, recall, count and memory targets are
the same on both.

## 3. Where 3.4 stands

Line anchors are at the v3.4.1 tag. 3.4.2/3.4.3 change compaction, flush,
memory accounting, full-text merging and GC; they do not change the read paths
listed here.

| Gap | Root cause | Where |
|---|---|---|
| **G1** `Building` window | A generation created over existing `Nodes` SSTs is `Building`; `Building` is never queried and its deltas are ignored; activation needs a whole-corpus base | `search_lsm_flush.rs:1385-1392`, `search_lsm.rs:1171`, `read.rs:9670-9690` |
| **G2** Base build cost | The base is rebuilt from `Nodes` SSTs via a spool rewritten once per tree level (about 4 × 1.5 GB at 377k); the opportunistic rebuild is skipped when all `Nodes` bytes exceed half the cgroup | `sst/vector/v5/external.rs:889-1060`, `search_lsm_compact.rs:244-246, 290, 592-597` |
| **G3** Weak partitions, heavy rerank | One principal direction per level, int8 nav codes at 1 KB/vector, rerank reads whole 2 MiB exact pages for about 8 rows | `sst/vector/v5.rs:571-690, 838`, `read.rs:495-516` |
| **G4** Flat deltas | VG6 is exact f32, 256 rows/page, scanned exhaustively; up to 32k rows per index by policy, more while compaction lags | `sst/vector/v6.rs:96, 756`, `search_lsm_read.rs:964` |
| **G5** Memtable re-decode | `vector_fresh_delta` decodes every memtable record on every vector query | `read.rs:9767-9800` |
| **G6** BM25 under writes | Any unflushed upsert carrying the label returns `None`, forcing two-pass tokenizing flat BM25 | `read.rs:9862-9878`, `walker.rs:3929-3942` |
| **G7** BM25 over many segments | Block-max pruning only for a single clean base; otherwise exhaustive posting decode per segment with 4× overfetch | `search_lsm_read.rs:1688-1767`, `sst/text/v4.rs:1537, 1845` |
| **G8** Presence count | `try_metadata_node_count` bails on any predicate; `IsNotNull` never prunes; embeddings sit in the per-row `__binary_props` map | `walker.rs:8325`, `sst/predicates.rs:171-176`, `sst/nodes.rs:20-28` |
| **G9** Flat vector fallback memory | Materialises every node with its embedding, sorts, then truncates | `walker.rs:6537-6574` |
| **G10** Scratch and bloat | Logical full-text rebuilds, superseded objects retained | addressed in 3.4.2/3.4.3 (posting-level FT4 merge, bounded k-way merge, store GC); this design must not reintroduce them |

The correctness machinery is sound and is kept: coverage per `Nodes` SST,
`NAMISV01` winner tables, signed statistics, the compatibility barrier, and
"any unproven state selects the exact flat scan".

## 4. Invariants

These hold across every step:

- **I1 Acceleration, not a different answer.** For BM25 the indexed result
  equals the exact flat result (scores and order, ties by `NodeId`). For
  vectors, membership may be approximate only in the settled partitioned part,
  bounded by the recall target; every returned score is the exact f32 metric;
  no write, delete, relabel or property removal is ever missed or resurrected.
- **I2 Coverage.** A generation is queried only when the manifest proves
  coverage (`search-lsm.md`, "Active-generation invariants"), generalised in
  §5.3.5 for shards and backfill.
- **I3 Bounded per-query work.** Every per-query structure is bounded by a
  configured cap that does not grow with the corpus: memtable tails (by the
  memtable cap), quantized tail (by the delta tiering policy), probed leaves
  (by `nprobe`), rerank rows (by the error bound and a hard cap). Exceeding a
  cap selects a slower *bounded* path, never an unbounded allocation.
- **I4 Maintenance off the write path.** Flush cost is O(memtable). No commit
  or flush waits for a merge, fold, split or re-cluster. Maintenance holds no
  writer lock, uses its own memory reservation and CPU share, is preemptible at
  leaf/shard boundaries, and is discarded and retried if its inputs changed
  under it.
- **I5 Change-proportional maintenance.** Bytes read, written and spooled by a
  maintenance job are O(inputs it rewrites). No job reads all `Nodes` SSTs
  except the one-time backfill of a new index (§5.3.6).
- **I6 Everything superseded is reclaimable.** Every object a maintenance job
  supersedes is unreferenced in the same manifest commit and becomes eligible
  for the janitor after the retention horizon.

## 5. Design

### 5.1 Three freshness tiers

```
query ──► memtable tail (exact, in memory, ≤ memtable cap)
      ├─► unsettled deltas (quantized codes → bound-based exact rerank)
      └─► settled shards (partitioned: centroids → probed leaves → codes → rerank)
                     │
          winner reconciliation by (NodeId, LSN) across all tiers
```

A query snapshot pins one memtable generation and one manifest. Each tier
yields candidates `(NodeId, LSN, exact score)` or suppressions; the existing
winner tables decide which version of a node is current. The tiers differ only
in how candidates are found:

| Tier | Contents | Search | Bound |
|---|---|---|---|
| Memtable tail | Committed, not yet flushed rows of the index | Exact f32 / exact BM25 from per-index in-memory structures (§5.2.6, §5.5.1) | memtable vector bytes cap (default `min(64 MiB, budget/8)`); reaching it requests a flush |
| Deltas | Flushed segments after each shard's frontier | VG7: probed leaves' codes (or the whole segment's codes when it is small); FT: block-max MaxScore | delta tiering keeps ≤ 8 delta segments per index; codes are 144 B/vector |
| Shards | Settled, partitioned base | Centroid directory → `nprobe` leaves → codes → rerank | `nprobe`, rerank cap |

There is no "unindexed tail scanned exactly" beyond the memtable: deltas are
partitioned and quantized at flush (§5.3.1), so the exact-scan portion is
always bounded by the memtable cap. This is what replaces both the 3.4
`Building` flat scan and the VG6 exhaustive delta scan.

### 5.2 Vector index format VG7

#### 5.2.1 Centroid directory (`NAMIVC01`)

A small immutable object per index generation and centroid epoch:

- `epoch: u64`, `dim`, `metric`, rotation seed, quantizer parameters;
- an upper level of about `√L` centroids and the leaf level of `L` centroids,
  stored as f16 (`L × dim × 2` bytes: 1,500 leaves × 1024 d ≈ 3 MB);
- per leaf: `leaf_id: u32`, parent, state `Live | Retired { children } |
  Merged { into }`, target and observed row counts, mean residual at creation
  (for drift detection);
- a checksum and the generation binding (`generation_id`, catalog signature).

The directory is always resident (it is a few MB even at 10M vectors with a
three-level tree) and is loaded with the manifest on open. A new epoch is
written by a split, merge or re-cluster; older epochs stay readable for
segments that still reference retired leaves.

#### 5.2.2 Quantization: RaBitQ codes with extended-bit refinement

- **Rotation.** One fixed random rotation per index generation: a randomized
  Hadamard transform (three rounds of random sign flips plus fast Walsh–Hadamard)
  on `dim` padded to the next power of two. Cost O(d log d) per vector, no
  stored matrix.
- **Residuals.** Each vector is normalised (cosine), its residual to its leaf
  centroid is rotated and encoded. Segments without a centroid directory
  (an index too small to partition, §5.3.6) use the segment mean.
- **1-bit plane.** `dim` sign bits (128 B at 1024 d) plus two f32 factors
  (residual norm and the RaBitQ inner-product correction): the unbiased
  estimator and its error bound from the RaBitQ paper.
- **Extended plane.** `B−1` further bits per dimension (default `B = 4`, so
  384 B at 1024 d), stored in a separate section so it can be fetched only when
  useful. Extended RaBitQ shrinks the error bound exponentially in `B`.
- **Row checksum.** An xxh3-64 of the exact row, stored next to the 1-bit plane
  (8 B), so a single exact row can be range-read and verified.

Per vector at 1024 d: 144 B resident-class codes, 384 B refinement, 4,096 B
exact. At 377k vectors: 54 MB, 145 MB and 1.54 GB.

**Kernels** (`namidb-ann/src/rabitq.rs`): the query is rotated once per leaf
centroid group; the 1-bit estimate uses byte-indexed lookup tables (256-entry
table per 8 dimensions, 128 lookups per vector at 1024 d) with scalar, AVX2 and
NEON implementations behind runtime detection. The extended estimate uses
4-bit lookup over the combined code. Both return `(estimate, lower, upper)`.

#### 5.2.3 Segment layout (`NAMIVG07`)

The same layout serves deltas and base shards; the role lives in the manifest.

```
+---------------------------------------------------------------+
| exact section     f32 rows, uncompressed, grouped by leaf      |  row-addressable
| extended section  (B−1)-bit planes, grouped by leaf            |  leaf-addressable
| code section      1-bit plane + factors + row xxh3, by leaf    |  leaf-addressable
| filter section    per-leaf native filter bitmaps (existing)    |
| NAMISV01 winner table (existing)                               |
| footer            centroid epoch, leaf directory, section CRCs |
| trailer                                                        |
+---------------------------------------------------------------+
```

- **Leaf directory:** `leaf_id → (row_start, row_count, code range,
  extended range, exact range, filter range)`, plus per-row `NodeId` ordinals
  into the winner table. Leaves appear in centroid-tree order so the leaves a
  query probes are usually adjacent and their ranges coalesce.
- **Exact rows are not compressed.** Embeddings compress poorly and per-row
  random access matters more; integrity is the per-row xxh3 plus the section
  CRC used by full reads.
- **Small-segment tail read.** The footer, winner-table directory and code
  section are written last; when the code section is ≤ 1 MiB the reader fetches
  footer and codes with one tail read, so a small delta costs one round trip.
- Unknown magic, version, section CRC or row checksum mismatch makes the
  segment unusable and selects the exact path (I2), as in VG6.

#### 5.2.4 Query algorithm (k-NN, `k` default 10)

1. **Rank leaves.** Score the upper level, descend into the best upper nodes,
   and rank leaf centroids (f16 → f32 dot). Select `nprobe` live leaves
   (default 48 at 1,500 leaves, i.e. about 3%; calibrated per index from `ef`,
   §5.2.5). Retired leaves are mapped to their children; the children's
   probe also reads the retired leaf's rows in segments written before the
   split.
2. **Gather codes.** For each probed leaf, the 1-bit codes from every shard and
   delta that holds that leaf, and the whole code section of small deltas.
   Native filters are applied per leaf from the filter bitmaps before scoring.
3. **Estimate.** Compute `(est, lo, hi)` for every candidate with the 1-bit
   kernel. Keep the `C1 = max(20k, 200)` best by estimate.
4. **Refine (if cheap).** If the extended sections of those leaves are cached,
   refine `C1` with the 4-bit estimate. If not (cold), skip refinement and use a
   wider rerank set in step 5 instead of paying another round trip.
5. **Bound-based rerank.** Let `τ` be the k-th smallest upper bound. Every
   candidate whose lower bound is below `τ` is reranked with its exact row
   (range read, checksum verified); others cannot enter the top-k at this
   probe depth. Hard cap `R_max` (default 512) on reranked rows; exceeding it
   reranks the `R_max` best by estimate and records `rerank_capped` in the
   profile.
6. **Reconcile.** Merge with the memtable tail's exact candidates and resolve
   winners by `(NodeId, LSN)` using the existing winner tables; a suppressed or
   superseded candidate is dropped and the next one is admitted. Duplicates of
   the same `(NodeId, LSN)` (relocation, spilling, §5.3.4) are deduplicated.
7. **Return** `k` rows with exact f32 scores.

Expected warm cost at 377k (M shape): leaf ranking < 1 ms; 48 × 256 ≈ 12k
code estimates ≈ 1–2 ms; refinement < 1 ms; 20–60 exact rows from the RAM or
local-disk tier ≈ 2–10 ms. The remaining latency budget is query parsing,
planning and HTTP.

#### 5.2.5 Recall control

- `ef` (existing user surface, RFC-036) maps to `nprobe` through a per-index
  calibration table written at shard fold: the fold samples 64 of the shard's
  own vectors as queries, compares probed top-k with exact top-k within the
  index, and records the smallest `nprobe` reaching 0.95 / 0.98 / 0.99.
  Without calibration data the default is `max(32, 3% of leaves)`.
- **Adaptive stop:** after the first `nprobe/2` leaves, further leaves whose
  centroid lower bound cannot beat `τ` (triangle inequality on residual norms
  stored per leaf) are skipped.
- **Spilling (optional, step S5b):** a vector whose second-nearest centroid is
  within `1 + ε` of its nearest also stores its codes (not its exact row) in
  the second leaf, SOAR-style. Winner reconciliation already tolerates
  duplicate `(NodeId, LSN)`. Enabled only if the bench shows recall below
  target at the default `nprobe`.
- **Continuous recall check:** a configurable fraction of production queries
  (default 0.5%) is re-run in the background as an exact scan of the same
  snapshot within a CPU budget; recall@10 is exported as a metric. This is how
  drift (§5.3.5) is detected in practice.

#### 5.2.6 Memtable vector tail

`VectorTail` per registered vector index, owned by the memtable generation:

- `NodeId → (LSN, Live(Arc<[f32]>) | Suppress)`, plus a contiguous f32 arena
  for scanning;
- maintained on commit from the same after-image the memtable receives
  (`ingest.rs`, where committed payloads are drained into the memtable);
  membership rules (label, dimension, zero vector for cosine) are the index's
  own rules from `search_lsm_flush.rs`;
- rebuilt during WAL replay on recovery;
- immutable snapshots share it by `Arc`, like the memtable claimant cell in
  `property_index.rs`.

A query scans it exactly (16k × 1024 d ≈ 17M multiply-adds, a few ms on one
core) and its suppressions shadow persisted hits. This replaces the per-query
memtable decode in `vector_fresh_delta`.

### 5.3 Incremental vector maintenance

#### 5.3.1 Flush-time assignment

When a flush prepares the vector delta for an index (`search_lsm_flush.rs`), it
assigns each live vector to its nearest live leaf of the current centroid epoch
(upper level, then the best two upper nodes' leaves: about 120 centroid
comparisons ≈ 30 µs per vector at 1024 d) and writes a VG7 delta grouped by
leaf. A 30k-vector flush spends about 1 s of CPU on assignment. The delta is
therefore partitioned at birth: queries probe only its relevant leaves, and
later maintenance never re-clusters it.

#### 5.3.2 Delta tiering

Deltas of the same epoch merge leaf by leaf in a streaming k-way merge with
winner reconciliation (scratch ≤ output). Policy: merge when more than 4
deltas are below 64 MiB, or more than 8 deltas exist in total. Deltas from an
older epoch are re-assigned during the merge (retired leaves to children).

#### 5.3.3 Base shards with per-shard frontiers

The settled base is a set of **shards**, each a VG7 object holding a contiguous
range of leaves in tree order, sized to about 64 MiB of exact rows (about 16k
vectors, about 24 shards at 377k). Each shard carries its own event
**frontier**. A delta applies to shard `s` if and only if its events are after
`s.frontier`.

A **fold** of shard `s` streams `s` and, for its leaves only, the rows of every
delta after its frontier; reconciles winners; applies splits and merges
(§5.3.4); writes the new shard; and advances `s.frontier` to the newest folded
event. Folds run when a shard's pending delta rows exceed 10% of its rows or
when the delta count cap needs space. A delta object is dropped once every
shard's frontier has passed it. Scratch is one shard; memory is one leaf plus
split working sets. The 3.4 `vector_base_rebuild_due` triggers and the V5
external builder are retired for VG7 generations.

#### 5.3.4 Split, merge, reassign (SPFresh/LIRE)

During a fold:

- **Split** a leaf whose row count exceeds 512 with 2-means on its rows
  (≤ ~1k vectors, in memory). The parent becomes `Retired { children }` in a new
  centroid epoch.
- **Reassign** boundary vectors: for the split leaf's rows and the rows of the
  16 nearest leaves inside the same shard, move any vector whose nearest live
  centroid changed. Moves to leaves of other shards are written as
  **relocation records** in a maintenance delta: same `(NodeId, LSN, payload
  fingerprint)`, flagged so statistics are not counted twice. The stale copy is
  harmless (same `(NodeId, LSN)`, deduplicated on read) and is removed when its
  shard next folds.
- **Merge** a leaf under 64 rows into its nearest sibling (`Merged { into }`).
- Splits are bounded per fold (default 64) so a fold's duration stays bounded;
  remaining oversized leaves wait for the next fold.

The published SPFresh evaluation reports that only a fraction of a percent of
inserts trigger rebalancing, with recall stable under skew, which is why local
maintenance can replace global rebuilds.

#### 5.3.5 Drift and rolling re-cluster

Each leaf records the mean residual at its creation. When the index's mean
residual grows by more than 20%, the leaf-size coefficient of variation exceeds
1.0, or the continuous recall check drops below target at the calibrated
`nprobe`, maintenance trains a new upper level and re-assigns shard by shard,
each step an ordinary bounded fold. Reads continue on the mixed epochs; there is
no window without an index.

**Validation (amends invariants 5–7 of `search-lsm.md`).** A VG7 generation is
valid when: the centroid directory objects referenced by the state exist and
their checksums and bindings match; shards have disjoint leaf ranges whose
union covers every live and retired leaf of the epochs in use; every delta's
event range is covered; and for every shard, every event after its frontier is
held by a listed delta or a `ProvenEmpty` coverage entry. Relocation deltas
cover no new events and are validated like `LogicalRewrite` coverage.

#### 5.3.6 First index over existing data: streaming backfill

`CREATE VECTOR INDEX` (or a catalog-signature change) on a label with data
starts a generation in state `Backfilling { snapshot_frontier, done_upto:
NodeId }` instead of `Building`:

1. Flushes write deltas for the generation from the start, as today.
2. A backfill task streams the label's rows at `snapshot_frontier` in `NodeId`
   order (LWW-reconciled k-way merge over `Nodes` SSTs, projecting only label
   membership, the indexed property and native-filter properties), in
   chunks of at most `NAMIDB_INDEX_BUILD_MEMORY_BYTES`.
3. The first chunk (or a reservoir of 16k vectors from it) trains the centroid
   directory; an index with fewer than 16k vectors stays one flat leaf.
4. Each chunk is assigned and written as a shard of `NodeId` range
   `[prev, done_upto)` with frontier `snapshot_frontier`, and committed with
   `done_upto` advanced.
5. **Progressive serving.** A query answers the `NodeId` range below
   `done_upto` from shards plus deltas, and the range above it from a
   range-restricted exact scan of the current snapshot (row groups outside the
   range are skipped by `node_id` statistics), ignoring deltas for that range.
   The two ranges partition the id space, so no node is counted twice or
   missed. Query cost falls as backfill advances.
6. When `done_upto` reaches the end, the generation becomes `Active`.

The backfill is one streaming pass: about 1.5 GB read at the bench's 100 MiB/s
(≈ 15 s), about 11 s of assignment CPU, and about 1.5 GB written. Target T2
allows 3 min on M. Backfill shards are leaf-partitioned per `NodeId` range;
the first ordinary folds merge them into leaf-range shards. Crash recovery
resumes from the committed `done_upto`.

The same state machine serves full-text indexes (§5.5.3), and it replaces
3.4's adoption of a single legacy base (`compact.rs` `prepare_search_lsm_activations`)
for new generations.

### 5.4 Cold path: at most three dependent round trips

| Wave | Reads (all concurrent within the wave) | M-shape estimate |
|---|---|---|
| 1 | centroid directory (if not resident), footers + small-segment code sections of every segment | 15–20 ms + about 3 MB transfer |
| 2 | 1-bit code ranges of the probed leaves in shards (coalesced by tree order) | 15–20 ms + about 2 MB |
| 3 | exact rows for the rerank set (100–300 rows of 4 KB) | 15–20 ms × ⌈rows / 64⌉ + transfer |

Total ≈ 150–300 ms on the bench store, inside T4. Required changes:

- **Row-granular admission** in `range_cache.rs`: the exact section of a VG7
  object uses a 16 KiB canonical page instead of 256 KiB, so 200 rows do not
  pull 50 MB; extended and code sections keep larger pages.
- **Per-class concurrency:** rerank reads may use up to 64 concurrent range
  requests (today `MAX_PINNED_OBJECT_RANGE_READS = 16`), governed by a global
  semaphore so concurrent queries cannot multiply it without bound.
- **Prewarm:** on engine open and after every shard install, load centroid
  directories, footers and, within the RAM page budget, the 1-bit code
  sections (54 MB at 377k). Prewarm runs at background priority and stops at
  the budget. The local-disk tier keeps extended and exact sections across
  restarts.
- **Profile:** each search records waves, ranges, bytes, codes estimated,
  rows reranked, tail rows scanned exactly and segments opened, visible in
  `PROFILE` output and the counters file.

### 5.5 Full text

#### 5.5.1 Memtable text tail (no fallback under writes)

`text_search_inner` stops returning `None` because a memtable row carries the
label. Instead, per full-text index and memtable generation, a lazily built and
cached `TextTail` classifies the label's memtable rows:

- **Unchanged:** the fingerprint of the after-image's indexed fields equals the
  persisted winner's `payload_fingerprint` (one batched `probe_winners` per
  memtable generation, then incremental for new rows). Embedding-only writes
  land here and have no effect on BM25.
- **Changed / new / removed:** the after-image is tokenized once and held in an
  in-memory posting map. At query time, for each query term, signed corrections
  to `df`, `N` and `total_len` are computed from the after-image and from the
  persisted before-image (membership by a posting probe at the winner's payload
  ordinal; document length from the segment's length table).

Scoring then treats the tail as one more signed delta. Cost is bounded by the
changed rows of the memtable times the query terms, never the corpus.

#### 5.5.2 Cross-segment block-max MaxScore

Replace per-segment exhaustive scoring with one global top-k:

- Global `N`, `avgdl` and per-term `df` are reconciled once (as today).
- Each `(term, segment)` posting list is a cursor whose block upper bound is
  computed from the block descriptor's `max_tf` and `min_doc_len` with the
  *global* statistics (FT4 already stores both).
- MaxScore over all cursors: terms split into essential and non-essential by
  cumulative upper bound against the heap threshold; blocks whose bound cannot
  beat the threshold are skipped without decoding.
- A candidate is checked against the winner tables before it can enter the
  heap (batched per block), so superseded postings never occupy a slot and the
  per-segment overfetch and refill loop disappear.
- Multiple full-text indexes or fields in one `search.bm25` / hybrid call run
  as one multi-field MaxScore (sum of per-field BM25 with field boosts), not one
  index scan per field.

Exactness: block bounds are upper bounds under the same global statistics, so
pruning never drops a true top-k document. A property test enforces this.

#### 5.5.3 Incremental full-text maintenance (FT5)

3.4.2 merges FT4 deltas at posting level (no re-tokenizing) with bounded k-way
scratch. FT5 removes the remaining corpus-proportional step, the base rewrite:

- **Term shards:** the base is split by term range into shards of about
  64 MiB of postings. Each shard has its own frontier; a fold merges pending
  delta postings for its term range only (posting-level, as `v4_merge.rs`).
- **Document shards:** winner table, document lengths and native-filter
  bitmaps live in shards partitioned by `NodeId` range, folded the same way.
- **Corpus statistics:** absolute `N` and `total_len` are kept in the
  generation state and advanced with each delta's signed values; per-term `df`
  is the term shard's absolute value plus signed deltas after its frontier.
- Postings stay in FT4's block encoding (block descriptors with `max_tf`,
  `min_doc_len`) and gain a per-block maximum impact for tighter MaxScore
  bounds. Compaction keeps neighbouring blocks of a term contiguous, so one
  range read fetches several blocks.
- Backfill of a new full-text index follows §5.3.6 (tokenize once per row).

Tokenization therefore happens exactly once per changed document, at flush or
backfill; folds and merges never re-tokenize, and scratch is one shard.

### 5.6 Hybrid search

`search.hybrid` (`walker.rs`, `hybrid_search_procedure`) runs the dense and
sparse legs concurrently on the same `Snapshot` and fuses with the existing
RRF (`fusion.rs`, `k = 60`). Equality and range filters are pushed into both
legs through native filter bitmaps (per leaf for vectors, per document shard
for text) instead of post-filtering a capped candidate set
(`NAMIDB_HYBRID_TEXT_FILTER_CANDIDATE_CAP`). When a filter's selectivity
leaves fewer than `nprobe × leaf_target / 10` matches in the probed leaves, the
vector leg continues probing the next-nearest leaves that contain a match
(cluster-level filter summary in the footer), so filtered recall does not
collapse.

### 5.7 Counts from metadata

#### 5.7.1 Count ledger

Generalise `ExactNodeCounts` (`property_index.rs`) into a `CountLedger`:

```rust
struct CountLedger {
    through_lsn: u64,                       // last flushed LSN it is exact for
    total: u64,
    by_label: HashMap<LabelId, u64>,
    presence: HashMap<(LabelId, PropertyId), u64>, // tracked properties only
}
```

- **Tracked properties:** every `(label, property)` covered by any index
  (RANGE, UNIQUE, VECTOR, FULLTEXT). Tracking starts at index creation; the
  index backfill (§5.3.6) or an equivalent streaming presence pass seeds the
  count at its snapshot frontier, then deltas apply.
- **Commit:** the committed batch's `(old_labels, new_labels)` changes, which
  the writer already produces for label counts, are extended with
  `(old_presence, new_presence)` bitsets over the tracked properties of those
  labels. The before-state comes from the same staged before-image the writer
  already uses for labels. A batch for which the before-state is unknown marks
  the memtable's delta as unknown.
- **Flush:** the persisted ledger in the manifest becomes `persisted +
  frozen-memtable delta`, written in the flush CAS. If the memtable delta is
  unknown (for example after WAL replay), the flush computes it exactly from
  the before-images it already reads for search-indexed labels, or by batched
  point reads for the flushed rows of tracked labels.
- **Compaction** carries the ledger unchanged (no logical change).
- **Staleness guard:** the ledger is used only when `through_lsn` equals the
  manifest's flushed frontier. A writer that does not maintain it (a 3.4
  binary after a downgrade) advances the frontier without the ledger, so a 3.5
  reader discards it and re-seeds in the background.

#### 5.7.2 Query rewrite

`try_metadata_node_count` (`walker.rs`) accepts, besides the plain label scan:

- `MATCH (n:L) WHERE n.p IS NOT NULL RETURN count(n)` and
  `RETURN count(n.p)` when `(L, p)` is tracked;
- conjunctions of label predicates on the same node.

The result is `ledger(L, p) + memtable delta(L, p)` at the snapshot. Anything
else (another predicate, `OPTIONAL MATCH`, grouping) takes the normal path, now
with a streaming count that projects nothing but the predicate's presence
(`__binary_props` key presence is read without decoding values) and never
collects rows. "Has an embedding" means non-null, consistent with Cypher; a
value that is not a valid vector still counts, which is why presence is
tracked independently of the vector index's `live_count`.

### 5.8 Write path, memory and CPU budgets

- Flush builds VG7 deltas (assignment + encoding) and FT deltas (tokenizing
  changed rows only) in O(memtable). Flush preparation happens off the writer
  lock (3.4.2), so commits continue into the next memtable.
- Maintenance (tiering, folds, splits, re-cluster, backfill) runs in one
  scheduler with: a memory reservation from `NAMIDB_INDEX_BUILD_MEMORY_BYTES`;
  a CPU share (at most `max(1, cores − 1)` worker threads, a single
  low-priority thread with cooperative yields on 1 CPU); preemption at leaf or
  shard boundaries; and CAS rebasing — a result whose inputs are still
  referenced is installed against the newest manifest, otherwise it is
  discarded and the next cycle redoes the (small) job.
- Under memory pressure maintenance pauses first; flush never waits for it.

Steady-state budget, S shape (1 GiB cgroup, 377k × 1024 d, after §5.2–5.5):

| Component | Budget |
|---|---|
| Centroid directories, footers, leaf directories, winner-table directories | ≈ 10 MB |
| 1-bit codes (RAM page tier) | 54 MB |
| Extended codes and exact rows | local-disk tier / object store (not RAM) |
| Memtable incl. vector and text tails | ≤ 256 MiB (existing cap) |
| Query workspace (per query: heaps, code buffers, rerank rows) | ≤ 16 MiB, ≤ 4 concurrent heavy queries |
| Maintenance reservation | 128 MiB |
| FT dictionaries / block descriptors resident | ≤ 64 MiB (RAM page tier) |

### 5.9 Storage amplification

- Maintenance writes per ingested vector: flush (1×), at most two tier merges
  while small (≤ 1× amortised), one fold (1×) — about 3× the vector bytes in
  total, versus a full-corpus rewrite per base rebuild in 3.4.
- Every superseded delta, shard and centroid directory is removed from the
  manifest in the commit that supersedes it (I6) and reclaimed by the 3.4.3
  janitor after the retention horizon.
- Counters: `search_maintenance_bytes_written`, per kind and role, and
  `search_unreferenced_bytes` from the janitor's sweep, so T15 is measurable.

### 5.10 Compatibility and rollout

- **Formats:** `NAMIVG07`, `NAMIVC01` and FT5 objects are new magics; manifest
  fields are additive (`#[serde(default)]`). Every VG7/FT5 generation keeps the
  `search-lsm.md` compatibility barrier, so a 3.4 reader sees multiple or
  undecodable bodies and takes its exact flat scan: slow but never incomplete.
- **Migration:** existing V5/VG6/FT4 generations remain readable. A V5 base is
  converted on its first fold (leaves re-assigned to a trained centroid
  directory, shard by shard); VG6 deltas are converted when tiered.
- **Switches:** `NAMIDB_SEARCH_VECTOR_FORMAT=vg6|vg7`,
  `NAMIDB_SEARCH_TEXT_FORMAT=ft4|ft5`, `NAMIDB_COUNT_LEDGER=on|off`; defaults
  flip to the new value one minor release after each step ships, and the old
  writers remain for one more minor release.

## 6. Cost model (377k × 1024 d)

| Item | f32 | 4-bit RaBitQ | 1-bit RaBitQ + factors |
|---|---|---|---|
| per vector | 4,096 B | 512 B (+ factors) | 144 B |
| corpus | 1.54 GB | 196 MB | 54 MB |
| 10M vectors | 41 GB | 5.2 GB | 1.44 GB |

| Query path | Bytes touched warm | Bytes fetched cold |
|---|---|---|
| 3.4 flat scan (`Building`) | 1.5 GB | 1.5 GB |
| 3.4 V5 base, ef 64 | ~30 MB | ~30 MB in ~120 GETs |
| 3.5 VG7, `nprobe` 48 | ~1.8 MB codes + ≤ 0.6 MB exact | ~5 MB in ≤ 3 waves |

## 7. Shippable steps

Each step is one PR series with its own CHANGELOG entry and bench report.
Steps S1–S3 change no segment format.

### S0 — Measurement first

Purpose: make every target in §2.3 a number produced by one command.

- `crates/namidb-bench/src/live_load.rs`, `main.rs`:
  - `--documents N` (default unchanged; the 3.5 profile uses 386,000) and
    `--fulltext-fields` (several `FULLTEXT` indexes on `Document`), so the
    vector phase writes 377k *distinct* embeddings instead of cycling over
    60k documents;
  - phase `fresh`: commit a batch, then immediately query with each just-written
    vector and with a unique term from its text; report the share found in
    top-10 and the commit-to-visible delay;
  - phase `count`: label count and presence count, cold and warm;
  - phase `hybrid`: `search.hybrid` with an equality filter;
  - `--concurrent-search-qps` during the `vectors` phase (under-writes
    latency and recall, T5/T8);
  - per-phase latency percentiles, recall@10 against exact ground truth over the
    snapshot's visible vectors.
- Cold-run support in the bench scripts: restart, empty RAM and local-disk
  caches, open, then time.
- Query profile counters (§5.4) in `search_lsm_read.rs` and
  `namidb-core` profiling.
- **Tests:** unit tests for the new phases' report math (percentiles, recall).
- **Proof:** a baseline report for 3.4.3 on M and S, checked in under
  `docs/benchmarks/` as the reference every later step is compared with.

### S1 — Counts from metadata (T10)

- `namidb-storage/src/property_index.rs`: `CountLedger` with presence;
  `apply_node_label_changes` generalised.
- `namidb-storage/src/ingest.rs`: commit computes presence changes beside
  label changes; recovery marks the memtable delta unknown.
- `namidb-storage/src/manifest.rs`: `count_ledger` field.
- `namidb-storage/src/flush.rs`: ledger fold in the flush CAS; exact
  recomputation when the memtable delta is unknown.
- `namidb-storage/src/compact.rs`: carry the ledger.
- `namidb-storage/src/read.rs`: `Snapshot::count_nodes_with_presence`;
  streaming presence count without materialisation for untracked cases.
- `namidb-query/src/exec/walker.rs`: `try_metadata_node_count` accepts
  `IS NOT NULL` / `count(n.p)` on tracked properties.
- **Tests:** property-based random sequences of create / delete / relabel /
  `SET p = null` / `REMOVE` / `SET p = <non-vector>` with ledger checked against
  an exact scan after every commit, flush, compaction, crash-recovery and
  restart; downgrade simulation (ledger frontier mismatch → discard and
  re-seed); planner tests that untracked predicates are not rewritten.
- **Proof:** `count` phase ≤ 100 ms cold and warm on M and S, also while the
  `vectors` phase runs.

### S2 — Bounded freshness without format change (T1, T8, G5/G6/G9)

- `namidb-storage/src/memtable.rs`, `ingest.rs`, `recovery.rs`: `VectorTail`
  per index, maintained on commit and on replay.
- `namidb-storage/src/read.rs`: `vector_fresh_delta` returns the tail;
  `text_search_inner` uses `TextTail` instead of falling back.
- `namidb-storage/src/search_lsm_read.rs`: winner fingerprint, payload-ordinal,
  doc-length and term-membership probes for the text tail.
- `namidb-query/src/exec/walker.rs`: flat vector fallback streams through
  `visit_label_with_projection` with a bounded top-k heap and fetches full
  nodes only for the final `k`.
- **Tests:** exact parity (vector exact scan, BM25 scores and order) with
  random memtable contents including updates of indexed text, embedding-only
  updates, deletes and relabels; recovery rebuilds tails; allocation test that
  the flat fallback's peak is O(k + page), not O(corpus).
- **Proof:** BM25 p50 under embedding-only writes ≤ 100 ms (M); vector query
  CPU no longer grows with memtable size (profile counter); `fresh` phase 100%.

### S3 — Streaming backfill and progressive serving (T2)

- `namidb-storage/src/search_lsm.rs`: `Backfilling { snapshot_frontier,
  done_upto }` state and validation; `select_search_read_plan` returns a
  progressive plan.
- New `namidb-storage/src/search_lsm_backfill.rs`: the backfill task, chunked
  by memory budget, resumable.
- `namidb-storage/src/search_lsm_flush.rs`: new generations start in
  `Backfilling`, not `Building`.
- `namidb-storage/src/search_lsm_read.rs`: id-range split between indexed and
  range-restricted exact paths.
- `namidb-storage/src/compact.rs`: legacy adoption only for legacy
  generations.
- `namidb-server/src/maintenance.rs`: schedule backfill under the maintenance
  budget.
- Uses VG6/FT4 until S4/S7 land.
- **Tests:** crash at every chunk boundary; writes to ids on both sides of
  `done_upto` during backfill; `DROP INDEX` and signature change mid-backfill;
  parity with the exact scan at every progress point; fault injection on
  segment upload.
- **Proof:** index created after 386k documents exist reaches `Active` in
  ≤ 3 min (M) / ≤ 8 min (S); time spent with a non-`Active` index after
  creation equals the backfill time; no `Building` state observed in manifest
  samples.

### S4 — Quantized vector segments (VG7, unpartitioned) (G4)

- New `namidb-ann/src/rabitq.rs`: rotation, 1-bit and extended encoders,
  estimators with bounds, scalar/AVX2/NEON kernels.
- New `namidb-storage/src/sst/vector/v7.rs` (reader) and `v7_external.rs`
  (streaming writer); VG7 with a single leaf.
- `search_lsm.rs` (`SearchSegmentFormat::Vg7`), `search_lsm_flush.rs` (write
  VG7 deltas), `search_lsm_compact.rs` (VG7 tiering; VG6 → VG7 on merge),
  `search_lsm_read.rs` (§5.2.4 steps 3–7).
- `range_cache.rs`: per-section canonical page size and per-class concurrency.
- **Tests:** estimator unbiasedness and bound coverage (bounds hold for
  ≥ 99.9% of pairs over fixed-seed Gaussian and clustered data at d = 128,
  768, 1024, 1536); returned scores bit-equal to exact f32; flat-VG7 recall@10
  ≥ 0.99 against exact; corruption tests per section and per row; decode fuzz
  target; cross-arch kernel equivalence.
- **Proof:** with deltas present, bytes per query from deltas drop ≥ 20×;
  under-writes vector p50 ≤ 300 ms (interim, before S5).

### S5 — Partitioned index with incremental maintenance (T3, T5, T6, T12, T14)

- New `namidb-ann/src/kmeans.rs` (balanced hierarchical k-means on a sample,
  2-means split) and `namidb-ann/src/partition.rs` (directory, assignment).
- New `namidb-storage/src/sst/vector/centroids.rs` (`NAMIVC01`).
- `sst/vector/v7.rs`: leaf directory, filter sections per leaf.
- `search_lsm.rs`: partitioning reference, shard role with leaf range and
  frontier, relocation deltas, validation of §5.3.5.
- `search_lsm_flush.rs`: flush-time assignment.
- `search_lsm_compact.rs`: tiering, folds, split/merge/reassign, drift
  re-cluster, V5 → VG7 conversion; retire V5 rebuild triggers for VG7
  generations.
- `search_lsm_read.rs`: leaf ranking, cross-segment leaf gather, retired-leaf
  mapping, relocation dedup, `nprobe` calibration lookup.
- `namidb-server/src/maintenance.rs`: scheduler budgets and preemption.
- S5b (optional): spilling.
- **Tests:** model-based test — random inserts, updates, deletes, splits,
  merges and relocations — checking no member lost or duplicated and recall
  against exact; crash at every maintenance step; maintenance/writer CAS
  conflicts; validation fuzz (drop or duplicate a shard, delta or directory →
  generation unavailable, exact path chosen); drift re-cluster correctness;
  migration from V5 and VG6.
- **Proof:** full `live-load` run at 377k: T3, T5, T6, T11, T12, T14 on M and S;
  no maintenance job above the scratch cap; recall stable across the run.

### S6 — Cold path and prewarm (T4)

- `search_lsm_read.rs`: three-wave plan, small-segment tail reads, adaptive
  rerank width when extended sections are cold.
- `range_cache.rs`: prewarm API within budget; 64-way rerank class.
- `namidb-server/src/registry.rs`, `maintenance.rs`: prewarm on open and after
  shard installs.
- **Tests:** latency-store instrumentation asserts at most 3 dependent read
  waves per vector query on a cold engine; prewarm respects the RAM budget.
- **Proof:** cold p50 ≤ 500 ms (M), ≤ 800 ms (S); round-trip depth ≤ 3 in
  profiles.

### S7 — Full-text read path and FT5 (T7, T8, G7)

- S7a (read path, no format change): `sst/text/v4.rs` block cursors with
  global-statistics bounds; `search_lsm_read.rs` global MaxScore with winner
  verification; multi-field evaluation in `namidb-query/src/exec/text_scoring.rs`
  and `walker.rs`.
- S7b (format): new `sst/text/v5.rs`, `v5_merge.rs` (term shards, document
  shards, per-block impact); `search_lsm.rs` state and validation;
  `search_lsm_compact.rs` text folds; backfill via S3.
- **Tests:** BM25 parity with the exact flat scorer (scores, order, ties) over
  random segment layouts, deletes, relabels and analyzer settings; bound-safety
  property test (no pruned document can belong to the true top-k); scratch
  bound per fold; crash at every fold step.
- **Proof:** T7 and T8; no logical (re-tokenizing) rebuild after index creation
  in the run's counters; text maintenance scratch ≤ 2× its inputs and
  ≤ 512 MiB.

### S8 — Hybrid (T9)

- `namidb-query/src/exec/walker.rs` (`hybrid_search_procedure`): concurrent
  legs on one snapshot; filter pushdown into both legs; filtered leaf
  continuation (§5.6).
- **Tests:** parity of hybrid results with sequential evaluation; filtered
  recall at 1%, 10% and 50% selectivity.
- **Proof:** `hybrid` phase T9 on M and S.

### S9 — Storage bounds (T15)

- Maintenance byte counters; janitor integration for every superseded VG7,
  centroid and FT5 object (builds on the 3.4.3 sweep).
- **Tests:** after a run with a short retention horizon, referenced bytes equal
  stored bytes for search objects.
- **Proof:** store bytes ≤ 1.5× referenced after the horizon on the full run.

### Ordering

```
S0 ─► S1 ─► S2 ─► S3 ─► S4 ─► S5 ─► S6
                         └──► S7a ─► S7b
                                S5 + S7a ─► S8
                                S5 + S7b ─► S9
```

S1–S3 deliver the user-visible freshness and count fixes on existing formats.
S4 removes the exhaustive delta cost. S5 and S6 deliver the vector latency and
recall targets at scale. S7 does the same for full text.

## 8. Bench plan

Server under test (both shapes), using the bench's simulated store:

```
namidb-bench serve --store file://<store>?ns=bench --listen 127.0.0.1:<port> \
  --latency-min-ms 10 --latency-max-ms 20 --bandwidth-mib-per-sec 100 \
  --memory-max-bytes <0.8 × cgroup> --flush-interval 30s \
  --compaction-interval 300s --counters-file <out>/counters.json
```

with the cgroup, CPU quota, cache and local-disk-tier sizes of §2.3. Sequence
per run: `text` (386k documents, four full-text indexes), `count`, `vectors`
(377k distinct, 300-row batches, `--concurrent-search-qps 2`), `fresh`,
settle 60 s, `search` warm, `hybrid`, restart cold, `search` cold, `count`
cold. Reported: every target T1–T15, manifest samples (generation states,
segment counts), memory samples, maintenance job durations and scratch,
store bytes by object kind. Bench stores and spools are deleted after each
run. `http-load --scenario vector-search --count 377000` remains the
isolated vector check (recall versus `ef`, cold/warm).

A step is accepted when its targets pass on both shapes in two consecutive
runs and no other target regresses by more than 10% against the previous
step's report.

## 9. Alternatives considered

- **Keep V5 and tune it** (more probes, smaller exact pages). Fixes neither the
  rebuild (G2) nor the plateau from one-direction partitions (G3); rejected.
- **Graph ANN with streaming merges (FreshDiskANN).** Strong in memory, but
  each hop is a dependent read on object storage, and a 1 GiB engine cannot
  hold the graph. Rejected for the primary index.
- **Product quantization.** Needs codebook training per index and gives no
  error bound for safe rerank; RaBitQ needs only a rotation and gives bounds.
- **Exhaustive exact tail with write backpressure** (cap the unindexed tail and
  reject writes above it). Contradicts G-writes; flush-time assignment makes
  the tail partitioned instead, so no cap on writes is needed.
- **Count from the vector index's `live_count`.** Not equal to "non-null" (a
  non-vector or wrong-dimension value is non-null but not a member); the
  presence ledger answers the actual Cypher semantics.
- **Approximate BM25 statistics (deleted documents stay in `df` until merge).**
  Simpler, but breaks the exact-parity contract; rejected as default.

## 10. Risks and open questions

- **Recall at `nprobe` 48 on real 1024-d embeddings** is an estimate; the
  synthetic clustered vectors in the bench can be easier or harder than real
  data. Calibration (§5.2.5) and spilling (S5b) are the mitigations; the
  continuous recall metric is the guard in production.
- **Bound-based rerank width** depends on the 1-bit bound's tightness at
  1024 d; if the cold path reranks too many rows, a 2-bit first plane is the
  fallback (still 272 B/vector).
- **Text tail cost under heavy text rewrites** grows with changed rows × query
  terms; the memtable cap bounds it, but the constant needs measuring.
- **FT5 complexity** (two shard families plus generation statistics) is the
  largest change in this document; S7a alone may meet T7 on the bench, in which
  case S7b is scheduled by storage and maintenance needs rather than latency.
- **Before-image availability for the count ledger** on every write path
  (bulk import, `MERGE`, Bolt, embedded) must be audited; any path without it
  marks the memtable delta unknown, which is correct but costs a flush-time
  recomputation.
- **Real object-store tail latency** is higher than the bench's uniform
  10–20 ms; the three-wave cold budget must be re-checked against measured p99
  time-to-first-byte.

## 11. References

- Xu et al., *SPFresh: Incremental In-Place Update for Billion-Scale Vector
  Search*, SOSP 2023 (LIRE rebalancing).
- Chen et al., *SPANN: Highly-efficient Billion-scale Approximate Nearest
  Neighbor Search*, NeurIPS 2021.
- Gao and Long, *RaBitQ: Quantizing High-Dimensional Vectors with a Theoretical
  Error Bound for Approximate Nearest Neighbor Search*, SIGMOD 2024.
- Gao et al., *Practical and Asymptotically Optimal Quantization of
  High-Dimensional Vectors in Euclidean Space for Approximate Nearest Neighbor
  Search* (Extended RaBitQ), SIGMOD 2025.
- Sun et al., *SOAR: Improved Indexing for Approximate Nearest Neighbor
  Search*, NeurIPS 2023.
- Turtle and Flood, *Query Evaluation: Strategies and Optimizations* (MaxScore),
  1995; Ding and Suel, *Faster Top-k Document Retrieval Using Block-Max
  Indexes*, SIGIR 2011.
- Singh et al., *FreshDiskANN*, 2021 (alternative considered).
- Published architecture notes of object-storage search services on
  centroid-tree ANN with RaBitQ and blocked posting lists (turbopuffer
  engineering blog: "ANN v3", "FTS v2"), and LanceDB's IVF_RQ index.
- NamiDB: `docs/architecture/search-lsm.md`, RFC-030, RFC-031, RFC-035, RFC-036.
