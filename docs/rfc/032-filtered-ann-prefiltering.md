# RFC 032: Filtered ANN pre-filtering (filtered-DiskANN)

**Status:** draft
**Author(s):** NamiDB team
**Created:** 2026-06-26
**Updated:** 2026-06-26
**Relates-to:** RFC-030 (DiskANN/Vamana vector index) — this RFC is the v2
successor to the adaptive-widening *post*-filter shipped there, for the
cheap-equality predicate slice.
**Implements:** (proposed; no PR yet)

## Summary

A KNN query with a residual `WHERE` predicate (`MATCH (d:Doc) WHERE
d.tenant = $t RETURN d ORDER BY cosine_similarity(d.embedding, $q) LIMIT k`)
is, today, served by *post*-filtering: the Vamana index returns the top-`k′`
nearest nodes ignoring the predicate, the executor materializes each, applies
the predicate, keeps up to `k`, and adaptively widens `k′` before falling back
to an exact flat scan. That path is correct and already shipped (RFC-030), but
the predicate never enters graph navigation, so a selective filter does
`O(k′)` wasted index work and then an `O(n)` flat scan on every query.

This RFC proposes **true pre-filtering**: push an equality/`IN` predicate into
the beam search so that only *matching* ordinals count toward `k`/`ef` while
non-matching nodes remain routing waypoints. The mechanism is a filter-aware
`beam_search_filtered(…, keep: impl Fn(u32) -> bool)` in `namidb-ann`, fed by
**per-value ordinal bitmaps materialized into the `.vg` body at compaction**.
Bitmaps share the index's `max_lsn`, so the existing freshness gate covers
staleness for free; the fresh-write delta is still handled by the residual
post-filter; and the adaptive widening + flat fallback stay underneath as the
safety net for everything bitmaps cannot express. This follows the
filtered-DiskANN line of work (Gollapudi et al., WWW 2023) that extends DiskANN
(Subramanya et al., NeurIPS 2019).

## Motivation

The shared-index, multi-tenant case is the one the current path serves worst.
When one `.vg` covers many tenants (or kinds, or statuses) and a query filters
to a single one, the predicate is *selective* but the matching rows are still
the genuine top-`k` for that tenant. Post-filtering pays for that selectivity
twice: once over-fetching candidates that get thrown away, and once more in the
`O(n)` flat scan when the over-fetch under-fills. The deeper problem is that the
index is *filter-unaware by construction* — selectivity that a B-tree index
would exploit to prune is, here, invisible to navigation.

The goal is to make a moderately-to-highly selective equality predicate served
*from the index*, sub-linearly, the same way the unfiltered KNN already is —
without giving up the RFC-030 invariant that **the indexed path returns exactly
what a brute-force flat scan would** (freshness gate, delta merge, exact
fallback). Pre-filtering must compose with those guarantees, not replace them.

## Design

### Background — the shipped post-filter path (Implemented: v1.4 + this change set)

This is the behavior the proposal extends; it is correct and stays as the
fallback.

**The index is filter-unaware.** `beam_search`
(`crates/namidb-ann/src/search.rs:54`) is parameterized only by a distance
closure `dist: impl Fn(u32) -> f32`. Admission to the `results` beam
(`search.rs:109`) and the convergence test (`search.rs:98`) count *every*
visited node toward `ef`/`k`; there is no predicate anywhere in the loop. The
`VectorSpace` trait (`crates/namidb-ann/src/space.rs:24`) exposes only dense
ordinals `0..len()` plus distances (`pair_distance`, `query_distance`) — no
`NodeId`, no node properties. So the `ann` crate *cannot* see a predicate even
in principle: the ordinal→`NodeId` map (`VectorGraphBody.ids: Vec<[u8; 16]>`,
`crates/namidb-storage/src/sst/vector.rs:90`, applied at
`self.body.ids[nb.id as usize]`, `vector.rs:401`) and the properties live two
crates up.

**So the query layer post-filters and widens.** `try_index_search`
(`crates/namidb-query/src/exec/walker.rs:3033`) over-fetches `k′ = k · mult`
nearest candidates (`Snapshot::vector_search`,
`crates/namidb-storage/src/read.rs`), merges the freshness delta, materializes
each candidate in rank order via `lookup_node`, evaluates the residual
`post_filter`, and keeps up to `k`. The change set just landed replaced the
historical fixed `×8` over-fetch with **adaptive iterative widening** (`walker.rs:3158-3234`):

- `OVERFETCH_BASE = 8`, `WIDEN_GROWTH = 4`, `MAX_WIDEN_ROUNDS = 4` → the
  over-fetch multiplier grows `8 → 32 → 128 → 512` whenever a round leaves fewer
  than `k` survivors, with `ef` raised in lockstep (`ef = …max(k′)`) so the beam
  actually surfaces the wider fetch.
- With no filter it is exactly one round at `mult = 1` — byte-identical to an
  exact top-`k` single shot.
- The freshness delta is pre-scored once (`delta_scored`, `walker.rs:3133`) and
  re-merged each round but never re-scored.
- A per-candidate `check_deadline()` (`walker.rs:3208`) keeps a wide widen
  interruptible.

**And falls back to the exact flat scan.** When even the widest fetch yields
`< k` survivors, or the index is exhausted (`hits.len() < k′`,
`walker.rs:3179`), `try_index_search` returns `Ok(None)` and `vector_search_rows`
(`walker.rs:2933`) runs the exact `scan_label_with_predicates_and_projection`
over the whole label. That scan is the ground truth — it applies the same
predicate to every node and is never short.

**Why this is correct but `O(n)`-in-worst-case.** The flat fallback guarantees
correctness, and the bounded widen (`≤ 512 · k` index candidates) makes a
*moderately* selective filter index-served. But the predicate never prunes
navigation: the greedy walk still expends its beam on geometrically-near nodes
that the filter rejects, and a filter selective enough to survive `< k` of
`512 · k` candidates collapses to an `O(n)` flat scan *every* query. Selectivity
is observed after the fact, never pushed down.

### Proposed — bitmap pre-filter (filtered-DiskANN)

Pre-filtering makes the predicate visible to navigation so that only matching
nodes count toward `k`/`ef`. It has two parts: a filter-aware search core in
`namidb-ann`, and a per-value bitmap materialized in the `.vg` to feed it.

#### 1. Filter-aware beam search (`namidb-ann`)

Add a sibling to `beam_search` that takes a keep predicate over ordinals:

```rust
pub(crate) fn beam_search_filtered(
    adjacency: &[Vec<u32>],
    n: usize,
    entry: u32,
    k: usize,
    ef: usize,
    dist: impl Fn(u32) -> f32,
    keep: impl Fn(u32) -> bool,  // matching ordinal? counts toward k/ef
) -> Vec<Neighbor>
```

Semantics, relative to the existing loop (`search.rs:94-117`):

- The `candidates` frontier (`search.rs:79`) still admits **all** visited
  nodes. A non-matching node remains a *routing waypoint* — it is expanded so
  the graph stays connected, but it never enters the result set. This is the
  whole point: the walk must be allowed to traverse *through* filtered-out
  regions to reach matching ones.
- A node enters the `results` beam (`search.rs:89-115`) and the final output
  **only if `keep(id)`**. The beam therefore holds the `ef` closest *matching*
  nodes seen so far.
- Convergence (`search.rs:98`) compares the closest unexpanded candidate against
  the worst *kept* result, so the search keeps walking while a closer matching
  node could still exist.
- The entry node still seeds expansion even when `!keep(entry)`.
- `search` / `search_with` (`search.rs:144` / `132`) and the build's call site
  (`crates/namidb-ann/src/build.rs:159`) delegate with `keep = |_| true` —
  exact back-compat; the build is never filtered.

`keep` is hot (called per visited node) and `beam_search` is sync, so `keep`
must be O(1) and allocation-free. That constraint is what drives the data
source below.

#### 2. The `ann ↔ properties` layering problem

`keep` is expressed over dense ordinals, but the predicate is expressed over
`NodeId`s and node *properties* that live in `namidb-storage` /
`namidb-query`. Bridging them inside a sync, per-node loop is the core
difficulty. Three options, in increasing generality and cost:

**Option A — precomputed per-tenant subgraphs.** Build one `.vg` per partition
key; the index *is* the filter and `keep` is implicit (`|_| true` over a graph
that already contains only the tenant). Optimal traversal, nothing in the hot
loop. But it is a combinatorial index blowup with heavy write amplification,
works only for a small set of statically-known low-cardinality keys, and
supports no ad-hoc predicates. Reserve it for a hard multi-tenant *isolation*
boundary, not general filtering.

**Option B — attribute bitmaps materialized at compaction, keyed by ordinal
(RECOMMENDED).** When the authoritative compaction builds the `.vg`
(`build_body`, `vector.rs:174`, the same pass that assigns ordinals and writes
`body.ids`), also materialize — for each indexed low-cardinality property value
— a bitmap over ordinals `0..n`. Store it in the `.vg` body next to `ids`. At
query time, `keep(ord) = selection.contains(ord)`, which is O(1), in-memory, and
sync. The query layer compiles the equality/`IN` predicate into the `selection`
(an AND/OR of value bitmaps) and threads it through
`Snapshot::vector_search → VectorGraphIndex::search → beam_search_filtered`.
True pre-filter, cheap, no async in traversal. Limited to low-cardinality
equality/`IN`; range and expression predicates are not covered, and bitmaps add
`.vg` bytes.

**Option C — query-supplied ordinal→`NodeId`→predicate callback.**
`VectorGraphIndex::search` takes `keep: &dyn Fn([u8; 16]) -> bool`, adapting the
ordinal via `self.body.ids[ord]` (`vector.rs:401`); the query closure maps
`NodeId` → predicate. Fully general (any residual `post_filter`), but a general
predicate needs node *properties*, which means an async `lookup_node` — and
`beam_search` is sync with `keep` on the hot path. So the callback can only be
backed by **in-memory** data (the fresh-write delta, plus at most a small
cached attribute column); general async-in-traversal is the open hard problem.
Use it as the residual *escape hatch*, never the primary filter.

#### 3. Bitmap materialization (Option B, in detail)

- **Where.** In `build_body` (`vector.rs:174`), where the ordinal→`NodeId`
  mapping is constructed. The index descriptor declares which low-cardinality
  properties are bitmap-indexed; for each such property and each distinct value
  observed, the builder sets bit `ord` for every member carrying that value.
- **Storage.** A new `VectorGraphBody` field, e.g.
  `filter_bitmaps: BTreeMap<(PropertyKey, ValueKey), OrdinalBitmap>`, where
  `OrdinalBitmap` is a compact bitset (a plain `Vec<u64>` bitset, or a roaring
  bitmap if a dependency proves justified by cardinality). Serialized in the
  same bincode body after `ids`.
- **Predicate compilation.** The optimizer/walker compiles a residual predicate
  over an indexed property into a bitmap `selection`: `prop = v` → that value's
  bitmap; `prop IN [a, b, c]` → OR of value bitmaps; a conjunction over several
  indexed properties → AND of selections. Anything that does *not* compile
  (range, expression, non-indexed property, high cardinality) stays in
  `post_filter` and is handled exactly as today. The compiler must be
  **conservative**: a wrongly-narrow selection would silently drop matching
  nodes — strictly worse than over-fetching — so when in doubt, leave it to the
  post-filter.

#### 4. Freshness

Pre-filtering inherits RFC-030's freshness model unchanged:

- **Stale index → flat scan, for free.** The bitmaps are part of the `.vg`
  body, so they carry the index's `max_lsn`. The existing gate
  `index_outrun_by_nodes` (`walker.rs:3084`) already falls back to the flat
  scan whenever a `Nodes` SST newer than the index exists — so a stale bitmap
  can never serve a query. No new freshness machinery is required.
- **Fresh delta → residual post-filter.** Writes the index has not yet absorbed
  are not in any bitmap. They are merged by `vector_fresh_delta`
  (`walker.rs:3118`) — `Some(emb)` for a live embedding, `None` to suppress a
  now-stale id — and the residual `post_filter` is evaluated against each
  materialized delta row, just as it is today. A delta node matching the
  equality predicate still surfaces (through the post-filter); one that does
  not is dropped. Bitmap pre-filter over the indexed corpus, post-filter over
  the delta: their union equals the flat scan.

#### 5. Recommended composition

Layer the four mechanisms so each is exact-or-better and the whole stays
flat-equivalent:

1. **Bitmap pre-filter** (Option B) inside `beam_search_filtered` for the cheap
   equality/`IN` slice of the predicate — pushes selectivity into navigation.
2. **Residual `post_filter`** for everything bitmaps cannot express (range,
   expression, non-indexed properties) — applied after materialization, as
   today, and over the fresh delta.
3. **Adaptive widening** (the shipped `walker.rs:3158` loop) as the safety net
   when the residual filter still under-fills a round.
4. **Exact flat scan** as the ground-truth fallback.

The pre-filter narrows the navigated field so the returned candidates are
already mostly matching; widening therefore rarely fires, and the flat fallback
even more rarely. Because every layer is exact-or-better, the composition
preserves the RFC-030 invariant.

## Alternatives considered

- **Do nothing — keep post-filter + adaptive widening + flat fallback.** The
  shipped path. Correct and zero-risk, but `O(n)` per query for any genuinely
  selective filter, and it never exploits selectivity the way an index should.
  This RFC exists to close exactly that gap.
- **Per-tenant subgraphs (Option A).** Optimal traversal but combinatorial
  index blowup and write amplification; only static low-cardinality keys, no
  ad-hoc predicates. Kept in reserve for hard isolation boundaries.
- **General query-supplied callback (Option C).** Fully general, but a general
  predicate needs async property lookups inside a sync per-node loop. Viable
  only when backed by in-memory data; retained as the residual escape hatch,
  not the primary mechanism.
- **FilteredVamana / label-aware build.** Add label-aware edges at *build* time
  so each label/value is internally navigable, eliminating traversal stalls at
  the source (Gollapudi et al., WWW 2023). This is the structurally correct
  long-term answer, but it changes `build.rs` and the `.vg` graph itself and
  interacts with int8 quantization and the single-fetch property — too large
  for a first cut. The bitmap pre-filter is the incremental step that reuses the
  existing graph.

## Drawbacks

- **Traversal stalls on a filter-unaware graph.** The Vamana graph is built with
  no knowledge of labels, so a value's matching nodes can lie in a region the
  greedy walk reaches only through many non-matching hops. With a selective
  filter the frontier can drain (visiting `O(n)` nodes) without ever filling
  `ef` with `k` matches. Bitmaps make `keep` *cheap* but do **not** fix
  *reachability* — that needs label-aware edges (FilteredVamana). Until then,
  widening + flat fallback bound the damage, but a pathological filter still
  degrades to the flat scan.
- **Bitmap size and cardinality.** One bitmap per `(property, value)` over
  `0..n` ordinals grows the `.vg`. Only low-cardinality equality/`IN` predicates
  fit; high-cardinality, range, or expression predicates fall through to the
  residual post-filter. Needs a cardinality cap (skip materializing a property
  with too many distinct values) and a compact encoding.
- **Compilation must be conservative.** Only equality/`IN` over an indexed
  property compiles to a selection (AND for conjunctions, OR within one
  property). A mis-compiled, too-narrow selection would silently drop matching
  nodes — worse than the current over-fetch. Parity tests must assert that the
  plan *uses* the bitmap (flat fallback makes equal-results pass trivially, so
  equal-results alone is not evidence the pre-filter ran).
- **A second search entry point.** `beam_search_filtered` adds a `keep` branch
  per visited node even in the `|_| true` delegate; mitigated by monomorphizing
  the closure (predictable branch), but it is real surface added to the hottest
  loop in the crate.

## Open questions

- Bitmap encoding (plain bitset vs roaring) and the cardinality cap at which a
  property is *not* bitmap-indexed.
- Where predicate→selection compilation lives (optimizer vs walker), and how the
  index descriptor declares which properties are bitmap-indexed.
- Whether to pursue FilteredVamana (label-aware build) next, and how it
  composes with int8 quantization and the single-`.vg`-fetch property.
- Whether Option C's in-memory residual callback earns its keep, or whether
  bitmap pre-filter + widening + flat covers enough of the predicate space.
- A traversal-stall guard: cap visited nodes and bail to the flat scan when a
  filtered search walks too far without filling `ef`, so a disconnected matching
  set degrades predictably rather than draining the whole graph.

## References

- RFC-030 (DiskANN/Vamana vector index) — the subsystem this extends; defines
  the freshness gate, the delta merge, the adaptive widening, and the exact flat
  fallback that pre-filtering composes with. RFC-030 forward-references this RFC
  as the v2 successor to its post-filter path.
- Subramanya, Jayaram Kumar; Devvrit; Kadekodi, Rohan; Krishaswamy, Ravishankar;
  Simhadri, Harsha Vardhan. "DiskANN: Fast Accurate Billion-point Nearest
  Neighbor Search on a Single Node." NeurIPS 2019. (Already cited in
  `crates/namidb-ann/src/build.rs:9-12` — Algorithm 1 RobustPrune, Algorithm 2
  Vamana.)
- Gollapudi, Siddharth; et al. "Filtered-DiskANN: Graph Algorithms for
  Approximate Nearest Neighbor Search with Filters." The Web Conference (WWW)
  2023 — FilteredVamana / StitchedVamana, the label-aware build that is the
  deeper fix to the traversal-stall drawback.
- `crates/namidb-ann/src/search.rs` (`beam_search`), `…/space.rs`
  (`VectorSpace`), `…/build.rs` (`build`); `crates/namidb-storage/src/sst/vector.rs`
  (`VectorGraphBody`, `build_body`, `VectorGraphIndex::search`);
  `crates/namidb-query/src/exec/walker.rs` (`try_index_search`,
  `vector_search_rows`).
