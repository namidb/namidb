# RFC 035: Incremental `.vg` maintenance and int8-by-default navigation

**Status:** draft
**Author(s):** NamiDB team
**Created:** 2026-06-26
**Updated:** 2026-06-26
**Implements:** (proposed) incremental `.vg` maintenance + int8-by-default
graph navigation across `namidb-ann`, `namidb-storage`, `namidb-query`
**Amends:** RFC-030 §"Proposed / Future" (resolves the *Incremental / mergeable
index maintenance* and the navigation half of *Richer quantization and memory
budgeting* forward-references)

## Summary

The DiskANN/Vamana vector index (RFC-030) has two scaling limits that this RFC
addresses. **CPU and freshness:** a `.vg` body is *rebuilt from scratch* on
every authoritative compaction and never upserted, so a freshly written batch
of embeddings sits on the exact `O(n)` flat-scan path for the whole label until
the next authoritative merge folds it in. **RAM:** the default (f32) index
loads the entire corpus into memory *twice* — once as the serialized
`VectorStorage::F32` body and once again as the navigation space — and
navigates on full f32, so resident memory per live index is ~`2 · N · 4 · dim`
bytes plus the serialized bytes, paid afresh on every query because there is no
decoded-index cache.

This RFC proposes (1) **incremental Vamana insertion** — greedy-search to find
a new point's neighbours, write its pruned out-list, add the back-edges and
re-prune recipients — so an authoritative compaction *merges* a small delta
into the prior graph instead of rebuilding it; (2) a **tiered base + delta
`.vg`** built at flush so the exact-flat-scan freshness window shrinks from
"nodes newer than the last authoritative compaction" to "nodes newer than the
last flush"; and (3) **int8-by-default navigation with exact f32 rerank** — keep
the int8 codes inline for cheap graph navigation and store the f32 vectors in a
separately-addressable (mmap/range-readable) region used only to rerank the ~`ef`
final candidates, so navigation RAM stops scaling with the f32 corpus while the
returned score stays f32-exact. A self-contained low-risk win — dropping the
duplicate corpus in `VectorGraphIndex::decode` — is flagged for landing ahead of
the larger format change.

The whole subsystem stays behind the `vector-index` Cargo feature, and the
governing invariant from RFC-030 is preserved end-to-end: **the index is an
acceleration of the flat scan, never a different answer.** Every change below is
gated by the same freshness machinery (`Snapshot::index_outrun_by_nodes`,
`Snapshot::vector_fresh_delta`, and the walker's exact flat fallback) that makes
that guarantee hold today.

## Motivation

Two costs grow with the corpus, and neither is bounded by the current design.

**The rebuild-not-merge freshness window is `O(n)` and lasts until the next
authoritative compaction.** `compact.rs::build_vector_indexes_for_nodes`
rebuilds the index from the merged node rows and drops the prior `.vg`
(`removed.extend(old.iter().map(|d| d.id))`), with the module-level note that
"A Vamana graph is not row-mergeable." The rebuild is *only sound when the merge
is authoritative* — i.e. the output is the bucket's deepest level so
`merged_rows` spans the full label corpus — and the hook returns early
(`if !authoritative { return Ok((Vec::new(), Vec::new())); }`) on any partial
merge, leaving the old `.vg` in place. Between those rebuilds, a flush (→ L0) or
a partial merge (→ L1+) writes a `Nodes` SST whose `max_lsn` exceeds the index
descriptor's stamped `max_lsn` (set to `corpus_max_lsn` at build).
`Snapshot::index_outrun_by_nodes` detects that with an **LSN compare** (`d.kind
== Nodes && d.max_lsn > idx_lsn`) and the walker (`try_index_search`) falls back
to the exact flat scan for the whole label. Writes are *visible* immediately —
`Snapshot::vector_fresh_delta` unions committed-memtable and staged-overlay rows
into the KNN, and the executor merges them with `seen` dedup — but the index
only *materializes* on the next authoritative compaction. A namespace that
ingests a large batch and does not happen to trigger a deepest-level merge pays
`O(n · dim)` per vector query for that whole window. This is correct, but it is
the opposite of what an index is for.

**Default-f32 navigation pins ~2× the corpus in RAM, per query.** The
`.vg` body is `bincode`-deserialized whole on **every** `vector_search` call —
`Snapshot::vector_search` loops every in-scope `VectorGraph` SST, calls
`get_sst_body` and `VectorGraphIndex::decode`, with no cache. `decode` then
**clones the entire corpus a second time** into the navigation space
(`F32CosineSpace::new(v.clone())` / `L2Space::new(v.clone())`) while
`body.storage` still holds the first copy, which `search` uses via
`VectorStorage::f32_at` to rerank candidates with the true metric. So a live f32
index holds two full f32 copies of the corpus plus the serialized bytes, and the
default quantization is `None` (full f32: `VectorQuantization::None` is
`#[default]`). For a million 768-d embeddings that is ~3 GB resident *before*
the serialized body, recomputed on every query. The int8 path is already much
leaner (it navigates on int8 codes, see below), but it is opt-in, cosine-only,
and returns the *quantized* score rather than the exact metric.

The cost of doing nothing is that the index's two headline numbers — time to
first indexed query after a write burst, and bytes resident per index — both
scale the wrong way as corpora grow into the millions, which is exactly the
regime the index exists to serve.

## Design

This RFC is split, like RFC-030, into **Implemented (v1.4 + this change set)** —
the authoritative description of what ships today, which the proposals build on —
and **Proposed / Future**.

---

### Implemented (v1.4 + this change set)

These are the load-bearing facts the design rests on; all are confirmed in the
current tree.

#### Rebuild-not-merge, authoritative-only

`build_vector_indexes_for_nodes` (`namidb-storage/src/compact.rs`) is the only
writer of `.vg` bodies. For each registered `VectorIndexDescriptor` whose label
has rows in the merge, it collects `(NodeId, embedding)` pairs from
`merged_rows`, calls `sst::vector::build_body` (a full Vamana build), writes the
new SST, and **marks every prior `.vg` for the same scope for removal**. It runs
only when `authoritative` is true; on a non-authoritative merge it returns
empty and leaves the existing `.vg` untouched. The new descriptor's `max_lsn` is
stamped with `corpus_max_lsn` — the high-water LSN of the rows the graph saw.

#### The exact-flat-scan freshness window

Reads gate on LSN, not level. `Snapshot::index_outrun_by_nodes`
(`namidb-storage/src/read.rs`) takes `max(max_lsn)` over the index's `.vg` SSTs
and returns `true` if any `Nodes` SST has a strictly greater `max_lsn`; the
walker's `try_index_search` (`namidb-query/src/exec/walker.rs`) returns
`Ok(None)` in that case, which routes the query to the exact flat scan.
Independently, `Snapshot::vector_fresh_delta` returns the committed-memtable and
staged-overlay delta for `(label, property)` as `(NodeId, Option<Vec<f32>>)` —
`Some(emb)` to merge, `None` to suppress a tombstoned / relabelled / embedding-
dropped id — and the walker pre-scores that delta once and unions it with the
index hits, starting its `seen` set from the delta ids so a superseded index hit
is dropped. The net guarantee: while the index is stale the answer is the exact
flat scan, and even while it is fresh the delta keeps just-written nodes
visible. The window is *correct* but `O(n)` for the whole label.

#### `vector_search` already unions across `.vg` SSTs

`Snapshot::vector_search` iterates **all** in-scope `VectorGraph` SSTs for the
index name and concatenates their hits before the final sort/truncate — "there
is normally exactly one per index … but a partial rebuild can briefly leave
two." It performs no dedup; the walker dedups via `seen`. This union loop is the
hook a delta tier reuses with no executor change.

#### int8 already navigates on int8 codes — confirmed, precisely

This is the pivotal fact for the memory axis, so it is stated exactly.

- **Build.** For `VectorQuantization::Int8`, `build_body` quantizes each vector
  (`quantize_i8`), builds the graph over an `Int8Space`, and stores **only**
  `VectorStorage::Int8 { codes, scales }`. No f32 vectors are written to an int8
  body.
- **Decode.** `VectorGraphIndex::decode` reconstructs `NavSpace::Int8(Int8Space::
  new(members))` from those codes+scales. `Int8Space` holds `members: Vec<(Vec<i8>,
  f32)>` and a `dim` — **int8 codes, not f32**.
- **Navigate.** `Int8Space::query_distance` scores a query against a member with
  `dot_i8_asymmetric` / `norm_i8` over the int8 codes (scale cancels in the cosine
  ratio). The whole beam search runs on int8.
- **Score.** int8 has **no f32 rerank**: `search` returns `1.0 - nb.dist`, the
  quantized cosine the navigation already computed. A `WHERE score >= t` threshold
  therefore compares against the quantized score, and recall sits at the
  documented ~0.80 floor versus ~0.85 for f32.

So the int8 path already cuts the `.vg` and navigation RAM ~4× — but at
quantized-score fidelity and cosine-only.

- **The f32 default is the heavy one.** For `VectorQuantization::None`,
  `build_body` keeps the original vectors (`VectorStorage::F32`) and builds over
  `F32CosineSpace` (cosine/dot) or `L2Space` (euclidean); `decode` clones them
  into the nav space; and `search` reranks each candidate with the exact metric
  (`metric_score(self.metric, &self.body.storage.f32_at(nb.id), query)`). f32
  **loads full f32 for both navigation and rerank** and holds two copies.

#### The Vamana primitives are insert-shaped already

`namidb-ann/src/build.rs::build` is a per-point loop: for each point `i` it runs
`beam_search` from the entry medoid for `l_build` candidates, `robust_prune`s
them to `R` for `i`'s out-list, then pushes the back-edge into each chosen
neighbour and re-prunes any neighbour whose list overflows `R`. `robust_prune`
(the α-robust DiskANN prune) and `beam_search` are already generic over a
`VectorSpace`. What is missing for incremental maintenance is purely structural:
`VamanaGraph` (`namidb-ann/src/graph.rs`) exposes `adjacency: Vec<Vec<u32>>` and
`entry: u32` with read-only accessors (`neighbors`, `max_degree`, …) and **no
mutation/append API**.

---

### Proposed / Future

#### 1. Incremental Vamana insertion (`namidb-ann`)

Add a single-point insert that mirrors the body of `build`'s refine loop, so a
graph can grow without a from-scratch rebuild:

```rust
// namidb-ann/src/build.rs
pub fn insert<S: VectorSpace>(
    space: &S,        // already includes the new member at dense id `new_id`
    graph: &mut VamanaGraph,
    new_id: u32,
    params: BuildParams,
) {
    // 1. greedy-search the graph-so-far for new_id's candidate neighbours
    let found = beam_search(&graph.adjacency, space.len(), graph.entry,
                            params.l_build, params.l_build,
                            |id| space.pair_distance(new_id, id));
    let cands = found.into_iter()
        .filter(|nb| nb.id != new_id)
        .map(|nb| (nb.dist, nb.id))
        .collect();
    // 2. robust-prune to R; write new_id's out-list
    let out = robust_prune(space, new_id, cands, params.alpha, params.r);
    graph.adjacency[new_id as usize] = out.clone();
    // 3. back-edges; re-prune any recipient that overflows R
    for &j in &out {
        let list = &mut graph.adjacency[j as usize];
        if !list.contains(&new_id) { list.push(new_id); }
        if list.len() > params.r {
            let cj = list.iter().map(|&nb| (space.pair_distance(j, nb), nb)).collect();
            graph.adjacency[j as usize] = robust_prune(space, j, cj, params.alpha, params.r);
        }
    }
}
```

This is the existing per-point step lifted out of the `for &i in &order` loop —
no new algorithm, the same `beam_search` + `robust_prune` + back-edge/re-prune
sequence. It requires a small mutation surface on `VamanaGraph`: grow
`adjacency` by one slot for a new dense id (`push_member() -> u32`) and an
`entry` that need not move for a small delta (the approximate medoid is stable
under a few-percent corpus growth; force a full rebuild past a growth
threshold). Batch inserts call `insert` per new id over a random permutation, as
`build` does, so back-edges interleave.

#### 2. Incremental `.vg` at authoritative compaction (`namidb-storage`)

Replace the unconditional from-scratch `build_body` in
`build_vector_indexes_for_nodes` with a *merge when cheap, rebuild when not*
policy, **without weakening the authoritative-merge soundness gate**:

- When a current `.vg` exists for the scope, covers the corpus, and the changed
  set is small relative to its `point_count`, **load the prior body, `insert` the
  new/changed dense ids, re-emit**, instead of rebuilding. The prior body already
  carries the graph and vectors; only the delta pays Vamana work.
- This still runs only under `if authoritative`. The merge is sound for the same
  reason the rebuild is: at the deepest level `merged_rows` spans the full label,
  so the load+insert sees the same corpus a rebuild would. A non-authoritative
  merge still leaves the `.vg` alone and relies on the freshness gate.
- **Updates and deletes.** Dense ids cannot be renumbered without rewriting the
  graph, so a changed embedding is an append of a new dense id plus a tombstone
  on the old one, and a delete is a tombstone alone. Carry a `tombstones:
  Vec<u32>` (or roaring bitmap) in the body, filtered at `search`, and **compact
  it away on the next full rebuild** (force a rebuild once tombstones or appended
  ids exceed a fraction of `point_count`, which also re-seats the medoid and
  reclaims the drift). This needs the body to persist a `NodeId → dense-id` map
  (today the body carries only `ids: Vec<[u8;16]>` parallel to the graph, which is
  the reverse direction); add the forward map so an update can find the dense id
  to tombstone.

This cuts authoritative-compaction CPU for a touched index from a full rebuild
(brute-force init is `O(N²)` below `AUTO_BRUTEFORCE_MAX = 4_000`, `O(N · L_build
· log)` above) to `O(delta · L_build · log)`, while preserving the exact same
read semantics.

#### 3. Tiered base + delta `.vg` to shrink the flat-scan window

Incremental insert at *compaction* still leaves the `O(n)` flat-scan window open
between compactions. Close it by building a small **L0 delta `.vg`** at flush (or
on the reactive trigger) over just the newly flushed nodes, searched alongside
the base graph:

- The union already exists. `Snapshot::vector_search` loops every in-scope `.vg`
  and merges, and the walker's `seen` dedup already tolerates an id appearing in
  two `.vg`s, so a base+delta pair needs **no executor change** — they are
  searched and merged exactly like the "partial rebuild briefly left two"
  case the code already documents.
- Freshness accounting falls out for free. `index_outrun_by_nodes` takes
  `max(max_lsn)` over the index's `.vg` SSTs. If the delta `.vg` is stamped with
  the flushed `Nodes` SST's `max_lsn`, then base+delta jointly cover up to
  `max(max_lsn)`, the gate stops flagging those rows, and the flat-scan window
  shrinks to "nodes newer than the *delta*" — i.e. one flush, not one
  authoritative compaction.
- **Atomicity.** The delta `.vg` descriptor must be committed in the same
  manifest version as the flush that produced its rows, via the create-only
  pointer CAS (RFC-029); otherwise a reader could momentarily see the delta's
  rows in a `Nodes` SST with the index not yet covering them (correct — it would
  just flat-scan — but it wastes the delta) or, worse, see the delta stamped at an
  LSN the corpus has not reached. Stamp-then-CAS, never the reverse.
- **Bounding the tiers.** Deltas accumulate until an authoritative compaction
  rebuilds (or, with proposal 2, incrementally re-merges) the base, at which point
  the existing `removed.extend(old…)` sweep drops every prior `.vg` for the scope,
  deltas included. Cap the live delta count (fold the oldest deltas into the base
  out of band when the count exceeds a small bound) so the per-query decode/search
  cost stays bounded.

The delta tier is purely additive and does **not** depend on the authoritative
gate — unlike proposal 2, it never rewrites the base, so it cannot truncate
anything. The two compose: deltas keep the window small continuously; the
incremental authoritative merge folds them in cheaply.

#### 4. int8-by-default navigation with exact f32 rerank (the memory axis)

This is the RAM proposal. Today the engine offers two extremes: f32 (exact
score, ~2× corpus RAM, navigates on f32) and int8 (quantized score, ~4× smaller,
navigates on int8, no rerank). The proposal is a third mode that takes the best
of both — **navigate cheap on int8, rerank exact on f32** — and makes it the
default memory profile:

- **Storage.** A new `VectorStorage` variant (bump `MAGIC` to `NAMIVG04`) keeps
  the int8 `codes` + `scales` **inline** (navigated in RAM, exactly as the int8
  path does today) **plus** the full f32 vectors in a **separately-addressable,
  range-readable region** of the `.vg` — an offset table keyed by dense id — that
  is *not* eagerly deserialized. Conceptually the int8 codes are the resident
  navigation structure and the f32 vectors are a paged column.
- **Decode.** Build `NavSpace::Int8` (cheap, ~4× smaller than f32) and keep a
  handle (mmap or a `Bytes` slice) to the f32 region. Navigation RAM is now the
  int8 corpus, independent of the f32 corpus size.
- **Search.** Navigate on int8 to gather `ef` candidates (as today), then
  **rerank only those ~`ef` ids** by reading their f32 vectors from the paged
  region and calling the existing exact `metric_score`. The returned score is the
  true cosine/dot/L2, equal to the flat scan — recovering the f32 path's fidelity
  (recall should land *above* pure-int8's ~0.80, approaching the ~0.85 f32 floor,
  because the final ranking is exact) at int8 navigation RAM. Per query the f32
  bytes touched are `ef · dim · 4`, not `N · dim · 4`.
- **Back-compat.** Older readers skip on magic mismatch and flat-scan, so
  `NAMIVG04` is back-compat-safe. A mixed base+delta set for one index must share
  orientation and metric; `vector_search` already takes the last-decoded
  orientation, so keep metric identical across tiers (it already is — all `.vg` for
  one index share a metric).
- **Paging substrate.** If full mmap of the f32 region is premature, the same
  shape works with a range read of the `ef` candidate rows from the object store
  (the `.vg` is already a single object fetched per search; a sidecar f32 object
  addressed by offset is the object-storage-native form). The decoded-index cache
  below makes the navigation structure resident across queries while the f32
  region stays paged.

#### 5. Low-risk incremental win: stop double-storing the corpus in `decode`

Flagged as the cheapest, lowest-risk step, landable ahead of the format change
and independent of it. `VectorGraphIndex` holds both `body` (with
`body.storage`) and `nav`, each a full copy of the corpus. After `nav` is built,
`body.storage` is used **only** by the f32 rerank in `search`
(`self.body.storage.f32_at(nb.id)`); the int8 path never touches it (it returns
`1.0 - nb.dist`). The nav space already owns the vectors and exposes them —
`F32CosineSpace::vector(id) -> &[f32]` and `L2Space::vector(id) -> &[f32]` are
public. So:

- Materialize the corpus **once**: keep `nav`, drop `body.storage` from the
  resident struct, retain only `ids`, `dim`, `metric`, `nav`.
- Replace the rerank read `self.body.storage.f32_at(nb.id)` with the nav space's
  borrowed `vector(nb.id)` (and feed it to the same `metric_score`). The int8 arm
  is unchanged — it must keep returning `1.0 - nb.dist` and must not be routed
  through the f32 rerank.

This halves resident f32 RAM (and ~halves int8) per live index with **zero format
or behaviour change** — `search` returns byte-identical scores because the rerank
reads the same vectors, just from the single owned copy. Length validation in
`decode` reads from `nav`/`ids` instead of `storage`.

#### 6. Decoded-index cache (orthogonal, helps every proposal)

`Snapshot::vector_search` re-`decode`s the full graph on **every** query — the
whole-body `bincode` deserialize is paid per query, per `.vg`. Cache the decoded
`VectorGraphIndex` keyed by SST id (the bodies are immutable; a new `.vg` gets a
new `Uuid::now_v7`, so cache invalidation is "evict ids no longer in the
manifest"). This is independent of the format change but compounds with it: with
proposal 4 the cached resident structure is the small int8 nav space, and the
heavy f32 region stays paged.

---

### Soundness and the freshness contract

Every proposal is downstream of the same gate, and none of them is allowed to
weaken it:

- The exact-flat fallback (`index_outrun_by_nodes` → `try_index_search` returns
  `Ok(None)`) and the memtable/overlay delta merge (`vector_fresh_delta`) are the
  backstop. As long as a delta or incrementally-merged `.vg` stamps a *correct*
  `max_lsn`, a too-short answer is impossible: anything newer than the stamp
  flat-scans, anything in the memtable is unioned in, and the walker's final
  "fewer than k survivors → flat scan" guard catches the rest.
- The delta-tier `max_lsn` must be committed atomically with the flush manifest
  (RFC-029 create-only pointer CAS). A delta stamped at an LSN its rows have not
  reached is the one way to serve a stale answer; stamp from the produced `Nodes`
  SST's high-water LSN and commit them together.
- Incremental insertion can drift recall versus a from-scratch build (stale
  medoid, under-pruned back-edges accumulating). Cap the delta/tombstone fraction
  and force a full rebuild past the threshold; the rebuild is still the periodic
  ground-truth reconstruction.
- All call sites stay `#[cfg(feature = "vector-index")]`; off-feature the buckets
  stay empty and no `.vg` is ever written.

## Alternatives considered

- **HNSW for native incremental insert.** HNSW's multi-layer structure supports
  online insert naturally, but RFC-030 already chose Vamana for the object-storage
  fit (one self-contained body, one fetch per top-k). Vamana's build loop is
  itself per-point (`build` inserts members one at a time), so the incremental
  insert here reuses that machinery rather than adopting a second graph family and
  a second on-disk format.
- **Delete by graph repair instead of tombstones.** Removing a dense id and
  re-linking its neighbours (the DiskANN "lazy delete + consolidate" approach) is
  the principled fix, but it needs a consolidation pass and complicates the dense-id
  invariant the body relies on. Tombstone-filter-at-search plus rebuild-on-threshold
  is simpler and rides the existing periodic rebuild, at the cost of carrying dead
  nodes in the graph until then.
- **Store only int8 and skip the f32 rerank (make int8 the default outright).**
  This is the existing int8 mode; it forfeits exact scores, the non-cosine metrics,
  and breaks `WHERE score >= t` against the true metric. Proposal 4 keeps int8's
  navigation RAM win *and* exact scores by paging the f32 rerank column, which is
  why it is preferred over simply defaulting to int8.
- **Build the delta tier at flush for every index unconditionally.** Building a
  `.vg` at every flush adds flush-path CPU even for write-light indexes. Gate delta
  builds on a batch-size / outrun-window threshold (build a delta only when the
  flat-scan window would otherwise be large), so write-light indexes keep the
  zero-overhead "just flat-scan the small delta" behaviour.
- **A per-query decoded-index cache only (do nothing else).** It removes the
  repeated decode cost but not the underlying ~2× f32 residency or the `O(n)`
  freshness window; it is included as proposal 6 because it compounds with the
  others, not as a substitute.

## Drawbacks

- **Incremental graphs drift.** A long run of inserts without a rebuild can
  degrade recall relative to a from-scratch build (stale medoid, accumulated
  back-edge churn). The threshold-triggered rebuild bounds this but adds a tuning
  knob (`delta_fraction`) the operator does not have today.
- **Tombstones inflate the graph until rebuild.** Deleted/updated nodes remain
  routing waypoints and consume degree budget until the next full rebuild; a
  delete-heavy index carries dead mass between rebuilds.
- **A new on-disk format (`NAMIVG04`) and an offset table.** The paged-f32 variant
  is more format surface than the current single-`bincode`-blob body, and the f32
  region adds I/O on the hot rerank path (mitigated by reading only `ef` ids and by
  the decoded-index cache). The body still has no checksum (RFC-030 drawback
  inherited); a paged region makes a checksum more attractive and is called out as
  an open question.
- **More moving parts in the freshness story.** A delta tier multiplies the number
  of `.vg` SSTs per index and adds a stamp-then-CAS ordering requirement at flush.
  The union/dedup already tolerate multiple `.vg`s, but the LSN-stamp atomicity is a
  new place to get freshness wrong.
- **The duplicate-corpus fix changes a public struct's internals.** Dropping
  `body.storage` from the resident `VectorGraphIndex` is behaviour-preserving for
  `search`, but any code reaching into `storage` directly would need to read from
  `nav` instead. It is contained to `sst/vector.rs`.

## Open questions

- The `delta_fraction` / tombstone-fraction thresholds that trigger a full
  rebuild, and the live-delta-tier cap — both want tuning against real write/delete
  mixes, not first principles.
- Whether the f32 rerank region should be mmap of the `.vg`, a range read of the
  single object, or a separate sidecar object addressed by offset — and how that
  interacts with the existing per-search single-fetch model and the node-view
  caches.
- Whether `NAMIVG04` should finally carry a body checksum now that a paged region
  makes partial/corrupt reads more likely than a whole-blob `bincode` decode.
- Whether to also page the f32 rerank column for the *existing* f32 default (so
  `quantization: none` navigates on f32 in RAM but reranks from a paged copy), or
  to steer new indexes toward the int8-nav-with-f32-rerank mode and leave the pure
  f32 mode as the "small corpus, all in RAM" choice.
- How the incremental insert interacts with RFC-027 P4 (key-range-partitioned
  leveled compaction): a range-partitioned authoritative merge sees a key *range*,
  not the whole label, which changes what "authoritative for the corpus" means for
  the index build gate.

## References

- RFC-030 (DiskANN/Vamana vector index) — the subsystem this extends; its
  "Proposed / Future" lists incremental/mergeable maintenance and richer
  quantization/memory budgeting, which this RFC designs.
- RFC-027 (Multi-level compaction, tombstone GC, space reclamation) — the
  authoritative/deepest-level merge and retention horizon the build gate and the
  rebuild-on-threshold rely on; P4 (key-range-partitioned leveled compaction)
  remains an open interaction.
- RFC-029 (Create-only versioned manifest pointer) — the commit-time CAS that a
  delta-tier `.vg` descriptor must use to stamp its `max_lsn` atomically with the
  flush that produced its rows.
- Subramanya et al., "DiskANN: Fast Accurate Billion-point Nearest Neighbor
  Search on a Single Node" (NeurIPS 2019) — Algorithm 1 (RobustPrune) and
  Algorithm 2 (Vamana Index), the basis for `namidb-ann::build`/`robust_prune` and
  the incremental insert; and the FreshDiskANN lazy-insert/lazy-delete +
  consolidate model behind the tombstone and delta-tier design.
- Code of record: `namidb-storage/src/compact.rs::build_vector_indexes_for_nodes`,
  `namidb-storage/src/sst/vector.rs` (`VectorStorage`, `VectorGraphIndex::decode`/
  `search`, `MAGIC`), `namidb-storage/src/read.rs` (`vector_search`,
  `index_outrun_by_nodes`, `vector_fresh_delta`),
  `namidb-query/src/exec/walker.rs::try_index_search`,
  `namidb-ann/src/{build.rs,graph.rs,space.rs}`,
  `namidb-storage/src/manifest.rs` (`VectorQuantization`, `VectorIndexDescriptor`).
