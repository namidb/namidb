# RFC 030: DiskANN/Vamana vector index

**Status:** accepted
**Author(s):** NamiDB team
**Created:** 2026-06-26
**Updated:** 2026-06-26
**Implements:** the `vector-index` Cargo feature across `namidb-ann`,
`namidb-storage`, `namidb-query`, `namidb-server`, `namidb-graph`; the
full-metric ANN + int8 change set; the filtered-ANN adaptive-widening and
procedure `filter`/`ef` change set.

## Summary

NamiDB indexes node embeddings with a DiskANN/Vamana approximate-nearest-
neighbour graph. A `CREATE VECTOR INDEX` registers a descriptor in the
manifest; the next authoritative compaction materializes a self-contained
`SstKind::VectorGraph` (`.vg`) body for it; and the query optimizer rewrites a
KNN-shaped Cypher pattern into a `LogicalPlan::VectorSearch` leaf the executor
serves from that body. All three metrics — cosine, dot, euclidean — are
indexable, optionally with per-vector int8 quantization (cosine-only). The
index is always an *acceleration of the flat scan, never a different answer*:
freshness gates, a memtable/overlay delta merge, an adaptive over-fetch for
filtered queries, and an exact flat fallback together guarantee that the
indexed path returns exactly what a brute-force scan would. This RFC is the
authoritative description of the subsystem as shipped in v1.4 plus the change
set just landed (full-metric ANN + int8, filtered-ANN adaptive widening, the
procedure `filter`/`ef` arguments, the natural-form `$__vector_ef` knob, and
`CREATE VECTOR INDEX … IF NOT EXISTS`). It is referenced throughout the
codebase as "RFC-030".

The whole integration is gated behind the `vector-index` Cargo feature
(`namidb-storage`'s `vector-index = ["dep:namidb-ann"]`, re-exported by
`namidb-query`, `namidb-server`, and `namidb-bench`). With the feature off, no
`.vg` is ever written, the compaction buckets stay empty, and the optimizer
rewrite is absent — every call site is `#[cfg(feature = "vector-index")]` and
the rest of each crate compiles without it.

## Motivation

Embedding-based retrieval (semantic search, RAG, recommendation) needs top-k
nearest-neighbour over high-dimensional vectors. A brute-force scan scoring
every node of a label is `O(n · dim)` per query — fine for thousands of nodes,
untenable for millions. We want sub-linear KNN that:

- lives in the same object-storage-first LSM the rest of the engine uses (the
  index is just another SST kind, built during compaction, fetched on read),
  so it inherits the manifest, snapshot, and caching machinery rather than
  bolting on a second storage system;
- is **exact-or-better**, never silently wrong: a freshly written embedding
  must be findable immediately, a deleted node must disappear, and a query the
  index cannot answer correctly (wrong dimension, too-selective filter, stale
  index) must transparently fall back to the flat scan;
- is reachable from ordinary Cypher (`MATCH … RETURN cosine_similarity(…)
  ORDER BY … LIMIT k`) without the user having to name a procedure, while also
  exposing Neo4j-compatible procedures for explicit control.

DiskANN/Vamana (Subramanya et al., NeurIPS 2019) is the natural fit for an
object-storage-first engine: it builds a single bounded-degree graph whose
adjacency and vectors serialize into one self-contained body, so answering a
top-k needs exactly one object fetch (the `.vg`) and no random I/O fan-out.

## Design

The subsystem spans six crates, layered bottom-up:

```text
  namidb-graph   algo.fastRP — structural embeddings that can feed an index
        │
  namidb-core    quantize.rs — per-vector int8 quantize/dequantize/scoring
        │
  namidb-ann     Vamana build + beam search, generic over a VectorSpace
        │            (storage-agnostic: never touches object storage)
        │
  namidb-storage .vg SST body, manifest descriptor, build-on-compaction,
        │            read path + freshness
        │
  namidb-query   optimizer KNN→VectorSearch rewrite, executor
        │            (index-or-flat), search.*/db.index.vector procedures
        │
  namidb-server  CREATE VECTOR INDEX DDL intercept (HTTP + Bolt)
```

The sections below are split into **Implemented (v1.4 + this change set)** —
the authoritative description of shipped behaviour — and **Proposed / Future**.

---

### Implemented (v1.4 + this change set)

#### 1. Data model and DDL

A vector index is declared over one `(label, property)`:

```text
CREATE VECTOR INDEX <name> [IF NOT EXISTS] ON :<Label>(<property>)
  METRIC <cosine | dot | euclidean>
  DIMENSION <n>
  [WITH { r: <int>, l_build: <int>, alpha: <float>,
          quantization: <none | int8> }]
```

`parse_create_vector_index` (`namidb-query/src/parser/grammar.rs`) parses it.
`VECTOR`, `INDEX`, `METRIC`, and `DIMENSION` are **soft keywords** (ordinary
identifiers elsewhere); `ON` is a hard token; `WITH` is the reserved
`Token::With`. `IF NOT EXISTS` is optional and sits between the name and `ON`
(the next token is unambiguously `IF` or the hard `Token::On`). The build
overrides in `WITH {…}` are each optional; `parse_optional_vector_with`
returns `(None, None, None, None)` when the clause is absent, and an unknown
key is a parse error. **int8 quantization is cosine-only and is rejected at
parse time** for any other metric, so a misconfiguration never reaches the
build.

`CREATE VECTOR INDEX` is a standalone schema command — it is never lowered to
a `LogicalPlan`. The server intercepts the parsed
`CreateVectorIndexClause` pre-plan on both the HTTP and Bolt paths
(`namidb-server/src/lib.rs`, `bolt.rs`). `vector_index_descriptor_from` maps
the AST to a storage `VectorIndexDescriptor`, filling the Vamana defaults
`r = 64`, `l_build = 128`, `alpha = 1.2` — explicitly commented to **mirror
`namidb_ann::BuildParams::default()`**. (These defaults live in two places
that must stay in sync: `BuildParams::default()` and this `unwrap_or`. Changing
one without the other is the classic cross-file version-sync gotcha; keep them
equal.) `apply_create_vector_index` then commits the descriptor through
`writer.register_vector_index(desc, if_not_exists)` (a metadata-only commit —
no `.vg` is built yet) and republishes the snapshot so subsequent reads see
the new index. DDL is a write: a read-only token is forbidden, and the
`authz::SchemaOp::CreateVectorIndex { name, label, property }` hook is
consulted via `check_schema`. The `.vg` graph itself is materialized **lazily,
on the next authoritative compaction** (§4).

`register_vector_index` honours `if_not_exists`: with it set, re-creating an
existing index is a no-op rather than an error.

#### 2. The Vamana algorithm layer (`namidb-ann`)

`namidb-ann` is the algorithm, storage-agnostic and generic over a
`VectorSpace` — it never touches a byte of object storage. The query side
reaches stored vectors only through the trait, by dense ordinal `0..len()`.

`VectorSpace` (`space.rs`) exposes `len`, `dim`, `pair_distance(a, b)` (member
to member, used by build), and `query_distance(query, b)` (external f32 query
to member, used by search). The contract is **"lower is closer" and distances
must be finite** — the beam-search heaps use a total order and a converged-
search comparison that assume no `NaN`. Three implementations ship:

- `F32CosineSpace` — full f32 vectors, cosine distance `1 − dot/(|a||b|)`,
  clamped to `[-1, 1]` before `1 −`. Zero-vs-zero is distance `0.0`,
  zero-vs-nonzero is `1.0` (orthogonal/maximally distant but finite). The
  recall-golden path used to validate the graph.
- `Int8Space` — per-vector `(codes: Vec<i8>, scale: f32)`. **Cosine on int8 is
  scale-invariant**: the per-vector scale appears identically in the dot
  numerator and the norm denominator, so it cancels; the impl still computes it
  with the `quantize` primitives so there is one definition of the score. Its
  zero-norm convention mirrors the f32 space — distance `0.0` **only when both
  sides are zero-norm** (keyed on `q_norm == 0 && norm == 0`, never on `dot`,
  which is forced to 0 in every zero-norm case and so cannot tell the two
  apart). This is the shipped storage path.
- `L2Space` — euclidean `sqrt(Σ(a−b)²)`, magnitude-sensitive. Because L2
  induces a genuinely different neighbour graph from cosine, a euclidean index
  **must navigate with L2** (a cosine graph would mis-rank whenever magnitudes
  vary).

**Build** (`build.rs`, DiskANN Algorithm 2). `BuildParams { r, l_build, alpha,
init }` defaults to `{ r: 64, l_build: 128, alpha: 1.2, init: Auto }`.
`build`:

- `n == 0` → empty graph; `n == 1` → one node, zero edges;
- forces `l_build = max(l_build, r + 1)` so prune always has enough candidates;
- picks the entry point with `approximate_medoid` (exact medoid for
  `n ≤ MEDOID_SAMPLE = 256`, else the sample of 256 minimizing total
  intra-sample distance);
- seeds adjacency by `InitStrategy`: `BruteForce` (exact `R`-NN, `O(N²)`) below
  `AUTO_BRUTEFORCE_MAX = 4_000`, `Random` above, under `Auto`;
- refines over a **random permutation**: for each point, `beam_search` the
  graph-so-far for `l_build` candidates, `robust_prune` them to `R` diverse
  out-edges, write them, add reverse edges, and re-prune any recipient that
  overflows `R`.

`robust_prune` (Algorithm 1) drops the anchor, sorts ascending by distance
(id tie-break), dedups by id keeping the closest, then greedily keeps `p*` and
removes any `p''` lying in its shadow — `α · d(p*, p'') ≤ d(anchor, p'')` —
capping at `R`. Larger `α` prunes less → more diverse neighbours → higher
recall. `build_with_seed(space, params, seed)` runs the build over a
`ChaCha8Rng` seeded from `seed`, so the same `(data, params, seed)` always
yields the same graph; storage seeds it from `xxh3_64(index_name)` for a
deterministic, per-index build.

**Search** (`search.rs`). `beam_search(adjacency, n, entry, k, ef, dist)` is a
greedy best-first beam shared by build and query (parameterized only by the
`dist: Fn(u32) -> f32` closure). It clamps `k = k.min(n)` and **`ef =
ef.max(k).min(n)`**; an out-of-range entry (`entry ≥ n`, possible from a
corrupt/foreign body since the `.vg` has no checksum) returns empty rather than
panicking. It converges when the beam is full (`results.len() == ef`) and the
closest unexpanded candidate is farther than the current worst result.
`search(space, graph, query, k, ef)` wraps it for the query case.

#### 3. int8 quantization (`namidb-core/src/quantize.rs`)

One shared definition of int8 quantization across the write path, the
`Int8Space` scorer, and the recall harness. Quantization is **per-vector and
symmetric** (max-abs scale): `quantize_i8(v) → (codes, scale)` takes the
max-abs over finite components only (non-finite components map to code 0 so the
scale never poisons to `NaN`), sets `scale = max_abs / 127`, and rounds/clamps
each component to `[-127, 127]`; an all-zero or zero-max input yields all-zero
codes with `scale = 0.0`. `dequantize_i8(codes, scale) = codes · scale`.
`dot_i8_asymmetric(query_f32, codes, scale) = scale · Σ qᵢ·codeᵢ` keeps the
query in f32 (the stored side is never expanded). `norm_i8(codes, scale) =
scale · sqrt(Σ codeᵢ²)`. A single fixed scale collapses recall at high
dimension (~0.87 at dim 1536); per-vector scaling restores it.

#### 4. The `.vg` SST format (`namidb-storage/src/sst/vector.rs`)

A `.vg` body is **self-contained**: it carries the indexed vectors plus the
Vamana graph, so a read needs no extra GETs. Layout is an 8-byte magic
`MAGIC = b"NAMIVG03"` (`NAMI` `VG` `\0` major=3) followed by a
bincode-serialized `VectorGraphBody`. **There is no checksum** — which is
exactly why `decode` validates ranges (below) and the read path treats any
decode error as "index absent". (v2 stored the original f32 vectors and
reranked with the true metric, making all three metrics indexable; v3
generalized the vector store to f32 *or* per-vector int8. An older body
mismatches the magic and is skipped.)

```text
VectorGraphBody {
  dim:     u32,
  metric:  String,                 // "cosine" | "dot" | "euclidean"
  ids:     Vec<[u8;16]>,           // NodeId per graph node i
  storage: VectorStorage,          // F32 | Int8, parallel to ids
  graph:   VamanaGraph,            // adjacency + entry, parallel to ids
}
VectorStorage = F32(Vec<Vec<f32>>)
              | Int8 { codes: Vec<Vec<i8>>, scales: Vec<f32> }
```

**Build** (`build_body(desc, members)`): returns `Ok(None)` when fewer than 2
members (the caller keeps the flat scan). It validates each vector's length
against `desc.dim`. **For cosine only, all-zero vectors are excluded** (a
zero-norm vector is not cosine-rankable; the flat scan's `vector_score(Cosine,
…)` returns `None` and drops it, so the indexed corpus must drop it too); dot
and L2 keep the zero vector. If filtering leaves fewer than 2, returns `None`.
**int8 + non-cosine metric is rejected** here as an invariant error (a second
guard behind the parse-time one). Navigation space is chosen by
quantization × metric: int8 → quantize each, build over `Int8Space`, store
`Int8`; f32 euclidean → `L2Space`; f32 cosine/dot → `F32CosineSpace`; store the
original (un-normalized) `F32` vectors. The build is deterministic via the
`xxh3_64(name)` seed.

**Decode** (`VectorGraphIndex::decode`): rejects a too-short buffer, a magic
mismatch (incl. a legacy body), an unknown metric, a length mismatch among
`storage`/`ids`/`adjacency`, an out-of-range graph entry, or mismatched int8
`codes.len()`/`scales.len()`. Because the read path treats any of these as
"index absent" and flat-falls-back, **a corrupt `.vg` never panics a query**.

**Search** (`VectorGraphIndex::search(query, k, ef)`): `k == 0` or a
dimension mismatch returns empty (the caller flat-falls-back, which raises the
canonical dimension-mismatch error — never a prefix-scored wrong answer).
Otherwise it clamps `ef = ef.max(k)`, **navigates** the graph for up to `ef`
candidates with the metric's navigation space, then **reranks** them with the
true metric and truncates to `k`:

- f32 cosine/dot/euclidean → `metric_score(metric, original_vector, query)`,
  computed in f64 to match the engine's `vector_score`: cosine similarity and
  raw dot are higher-is-closer; L2 distance is lower-is-closer. cosine/dot
  navigate with cosine (scale-invariant, rank-correlated for dot) and the wider
  pool surfaces the true dot-nearest; euclidean navigates with L2.
- int8 → the navigation distance already *is* the (quantized cosine) score, so
  the score is `1.0 − dist`.

Results are sorted by `higher_is_better() = !Euclidean`. **The approximation is
only in *which* nodes the graph visits, not in the score** — the returned score
equals the flat scan's `vector_score` to f32 tolerance, so a `WHERE score >= t`
threshold compares against the same number either way (for int8, against the
quantized score).

#### 5. The manifest descriptor (`namidb-storage/src/manifest.rs`)

`Manifest::vector_indexes: Vec<VectorIndexDescriptor>` holds the registered
indexes.

- `VectorMetric { Cosine, Dot, Euclidean }` (serde lowercase). `builtin_name()`
  maps to the Cypher function the optimizer matches against:
  `cosine_similarity` / `dot_product` / `euclidean_distance`.
- `VectorQuantization { None (default), Int8 }` (serde lowercase). `None` is
  ~`4·dim` bytes/vector; `Int8` is ~`dim+4` bytes/vector (~4× smaller),
  cosine-only and lossy (the `namidb-ann` recall floor is ~0.80 vs ~0.85 at
  f32).
- `VectorIndexDescriptor { name, label, property, dim: u32, metric, r: usize,
  l_build: usize, alpha: f32, quantization }`. `quantization` is
  `#[serde(default)]` so manifests written before the field decode as `None`.
  `matches(label, property, metric)` is the optimizer/executor lookup key.

`SstKind::VectorGraph` (path tag `vector-graph`) has no meaningful
lexicographic key range, so its descriptors carry full-range
`min_key = [0;16]` / `max_key = [0xFF;16]` and are looked up by
`(kind, scope = index_name)`. `KindSpecificStats::VectorGraph { dim, metric,
point_count, r, l_build, alpha, entry_medoid }` lets the read path and
observability reason about an index without decoding its body.

#### 6. Build-on-compaction (`namidb-storage/src/compact.rs`)

A Vamana graph is not row-mergeable, so the index is **rebuilt, not merged**.
`build_vector_indexes_for_nodes` runs inside the node-bucket loop after the
merged Nodes SST is written, feature-gated, bucketed by index name.

The critical gate: an index is rebuilt **only on an authoritative compaction**
(`authoritative == plan.is_deepest`, where the merged rows span the full label
corpus). On a non-authoritative (shallow) merge the existing `.vg` is left
untouched and the freshness gate (§7) flat-falls-back until the next
authoritative merge rebuilds it. This prevents truncating the index to a
shallow subset, which would be a permanent recall loss.

Members are gathered per descriptor from the merged node rows: `Upsert` records
carrying the index's label, with a `Vec` value taken as-is and a `VecI8` value
dequantized; everything else skipped. A per-index build error is logged and
skipped (one bad index never wedges namespace compaction); a `build_body` that
returns `None` (too few members) is skipped. The new SST descriptor is stamped
with **`max_lsn = corpus_max_lsn`** — the high-water LSN of the merged
corpus — which is the freshness anchor (§7); prior `.vg` ids for this scope are
removed.

#### 7. The query path

**Optimizer rewrite** (`namidb-query/src/optimize/vector_search.rs`).
`apply_vector_search` recurses bottom-up (so a KNN nested in a `UNION` branch,
a `CALL {}` subquery, a join, or an aggregate is rewritten too) and is
registered in `optimize::mod` right after `unique_lookup`, before pushdown — so
the downstream pushdowns, which treat `VectorSearch` as an opaque leaf, see the
new operator and don't re-introduce a Filter above it. Two lowered KNN shapes
are matched:

- terminal `RETURN`: `Project[…]{ TopN{ [Filter] NodeScan } }` → the ranking
  sub-tree collapses to a `VectorSearch` leaf and the outer `Project` is kept
  (the score alias is read from the outer projection);
- non-terminal `WITH`: `TopN{ Project[…, dist AS score]{ [Filter] NodeScan } }`
  → a bare `VectorSearch`;
- a threshold on the ranked output, `Filter{ TopN }` (e.g. `WHERE score >=
  0.86`), is folded into the `VectorSearch`'s `post_filter`.

`try_match` is conservative and bails (leaving the flat plan, which still
honours everything) on: a non-zero `SKIP`; an unbounded `ORDER BY` with no
`LIMIT` (`k = u64::MAX`); more than one order key; an order key in the wrong
direction (it must be nearest-first — `DESC` for cosine/dot, `ASC` for
euclidean; the wrong direction asks for the *farthest* k, which the index does
not compute); a non-vector key function; a scan alias that doesn't match the
distance call; a predicate already pushed into `NodeScan.predicates` (it can't
be reconstructed, so the flat path must run it); a missing index; or a
`post_filter` that references anything other than the searched binding / score
alias. `StatsCatalog::vector_index_for(label, property, metric)` does the
descriptor lookup; `VectorGraph` SSTs contribute nothing to cost statistics.

**Executor** (`namidb-query/src/exec/walker.rs`). `vector_search_rows` is the
shared core for both the `VectorSearch` operator and the procedures. It tries
`try_index_search` first (feature-gated); on `Ok(None)` it runs the **exact
flat scan** — scanning every node of the label(s), scoring with `vector_score`,
sorting, applying the `post_filter` *before* truncating to the limit, and
materializing whole nodes (a downstream projection may read any property).

`try_index_search` returns `Ok(None)` (→ flat scan) when any of these hold,
otherwise serves from the index:

- no label, no matching descriptor, or the query is not a vector;
- **dimension mismatch** (`qv.len() != index_dim`) — a wrong-length query would
  otherwise be silently prefix-scored;
- the **freshness gate** trips: `index_outrun_by_nodes(index_name,
  VectorGraph)` is LSN-based — it returns true when any Nodes SST has a
  `max_lsn` greater than the index's, i.e. a node was persisted but not yet
  folded into an authoritative `.vg`. The lockstep Nodes SST from the same
  authoritative merge shares the index `max_lsn` exactly, so it is never
  flagged; this closes the partial-compaction truncation window without a
  level-based heuristic.
- a **zero-magnitude cosine query** — cosine is undefined, the flat path drops
  every candidate (returns empty), but the index rerank would score a
  similarity of `0.0` instead of dropping; the cosine-only guard
  (`metric == Cosine && qv.all(== 0)`) flat-falls-back so the index path agrees
  with the `cosine_similarity` builtin's NULL semantics. (Dot/L2 are
  well-defined on a zero query, so the guard is cosine-only, mirroring
  `build_body`'s cosine-only zero-vector skip.)

When it does serve from the index, the path merges the index hits with a
**freshness delta** (`vector_fresh_delta(label, property)`: the committed
memtable plus the staged overlay, highest-LSN-per-id wins; `Some(emb)` is a
live embedding to merge, `None` suppresses a now-stale id — tombstoned, label
removed, or embedding dropped). The delta is **pre-scored once** with
`vector_score(distance, …)` (for an int8 index the delta vector is round-tripped
through the same quantizer so both halves of the merge sit on the same quantized
cosine scale).

**Adaptive over-fetch / widening + flat fallback** is the filtered-ANN
correctness mechanism. A residual `post_filter` is a *post*-filter — the index
navigates without knowing the predicate, then the engine scores, sorts, and
applies the filter while materializing up to `k` survivors. Because a selective
filter can leave fewer than `k` of an over-fetched candidate set, the path
widens geometrically before giving up:

```text
const OVERFETCH_BASE = 8; WIDEN_GROWTH = 4; MAX_WIDEN_ROUNDS = 4;
widen = post_filter.is_some();
mult  = widen ? 8 : 1;                  // no-filter ⇒ exactly one round, mult=1
for round in 0..(widen ? 4 : 1):
    kprime = max(k, k*mult + delta_ids.len());
    ef     = ef_search ? ef_search.max(kprime) : max(kprime, 64);
    hits   = snapshot.vector_search(index_name, qv, kprime, ef);   // unions .vg SSTs
    merge hits (deduped, delta-suppressed) with the pre-scored delta;
    sort by orientation; materialize ≤ k applying post_filter (per-candidate deadline);
    if survivors >= k: return them;
    if hits.len() < kprime: break;       // index exhausted — wider can't help
    mult *= 4;                           // 8 → 32 → 128 → 512
return Ok(None);                         // fall back to the exact flat scan
```

So a filtered query over-fetches `8k`, then `32k`, `128k`, `512k`, and only
then falls back to the `O(n)` flat scan — a moderately selective filter (the
shared-index multi-tenant case) is served from the index instead of a flat
scan every time. With **no** filter, `max_rounds = 1` and `mult = 1`: a single
exact top-`k`, byte-identical to the pre-widening behaviour (an exact top-k
cannot under-fill from selectivity). The `delta` is re-merged each round but
**never re-scored**; `delta_ids` seeds the `seen` set each round so suppression
and dedup (storage `vector_search` unions across `.vg` SSTs without deduping)
are identical to the old single-shot merge. A per-candidate `check_deadline()`
keeps a widened filtered ANN interruptible like the flat scan.

The widening never changes the answer — `ef` only ever *grows* the beam (recall
up, never wrong), and the `Ok(None)` flat fallback (after the round cap or once
`hits.len() < kprime` signals the index is exhausted) is the ground truth.

**Beam width (`ef`).** The procedures take a first-class `ef`. The
natural/operator form has no Cypher syntax for it, so the executor reads a
reserved, namespaced parameter `$__vector_ef` in `flat_vector_search` and
threads it into the same `ef_search` slot. Because it only ever raises the beam
(`ef_search.max(kprime)` downstream), it cannot corrupt results; absent, the
per-round default `max(kprime, 64)` applies. **`$__vector_ef` is an explicitly
NON-STABLE knob** — it is global to the query (every `VectorSearch` node in the
query sees the same value), it is namespaced to avoid clashing with a user's own
`$ef`, and it is intended to be superseded by a first-class `OPTIONS { ef }`
surface (RFC-036). Do not depend on its name.

#### 8. Procedures

Three procedures reach the same `vector_search_rows` core (each
independently index-or-flat, so they are freshness-equivalent to the natural
form):

- `CALL search.vector({ label, property, query, k?=10, ef?, metric?=cosine,
  filter? }) YIELD node, score` — the procedure counterpart to the optimizer
  rewrite. `filter` becomes the index-side `post_filter` (over-fetch + exact
  fallback); `ef` is the beam width.
- `CALL db.index.vector.queryNodes(indexName, k, queryVector [, { ef, filter
  }]) YIELD node, score` — Neo4j-compatible. Resolves the index **by name** to
  `(label, property, metric)` from its descriptor, then takes the same path;
  errors if no index is named.
- `CALL search.hybrid({ label, query_text?, text_property(ies)?, query_vector?,
  vector_property?, k?=10, ef?, fusion?='rrf', rrf_k?, alpha?=0.5, metric?,
  filter?, k_dense?, k_sparse? }) YIELD node, score` — fuses a dense leg
  (vector KNN via `vector_search_rows`) and a sparse leg (BM25 via
  `bm25_ranked`). Default fusion is Reciprocal Rank Fusion (rank-based, needs no
  cross-scale calibration); `fusion: 'linear'` is a min-max blend weighted by
  `alpha` on the dense leg. Each leg over-fetches `k·8` candidates by default
  (`k_dense`/`k_sparse`). The dense leg requires **both** `query_vector` and
  `vector_property` (supplying exactly one is an error). A `filter` applies to
  the dense leg as an index-side `post_filter` *and* to the fused output (so a
  sparse-only hit that fails the predicate is also dropped).

#### 9. Write-time dimension enforcement (`namidb-query/src/exec/writer.rs`)

`enforce_vector_dims` runs on every node create/merge/set path. For each
registered index whose label the node carries, a present (non-null) value for
the indexed property must be a numeric vector of the **declared dimension** — a
mismatch is a `Constraint` error at write time, not a silent build-time
truncation. A null clears the property (not-null is enforced separately). A
correct-dimension value stored as a bare numeric **list** is coerced to a dense
`Vec` (`numeric_list_to_f32`) so the index build covers it. This keeps the
corpus the compaction build sees uniform and correctly typed.

#### 10. FastRP structural embeddings (`namidb-graph/src/algo.rs`)

`CALL algo.fastRP(...)` (aliases `fast_rp` / `fastrp`) computes FastRP (Chen et
al., CIKM 2019) structural node embeddings and yields `(node_id, embedding)`
where `embedding` is exactly the `(NodeId, Vec<f32>)` shape a vector index
ingests — so a graph's structure can be written straight into a `.vg` and
searched. `FastRpOptions` defaults follow Neo4j GDS: `dimension = 256`,
`iteration_weights = [0.0, 1.0, 1.0, 1.0]` (the hop-0 raw projection is
dropped; the embedding is `Σ_k w[k] · R_k`), `normalization_strength = 0.0`
(plain mean propagation `D⁻¹A`), `seed = 42`. The algorithm is deterministic
for a fixed seed (sparse `±√3` projection from `splitmix64`, neighbour lists
built in node-insertion order because f32 sums are non-associative),
near-linear `O(iterations · E · d)`, and cancellable per hop. The
`fastrp_options` parser accepts `{ dimension?, iterations?, iteration_weights?,
normalization_strength?, seed? }`; `iterations: n` expands to
`iteration_weights = [0.0] ++ [1.0; n]`, and an explicit `iteration_weights`
list overrides it.

#### Parameters and defaults

| Parameter | Where | Default | Notes |
|---|---|---|---|
| `r` (`R`, max out-degree) | `BuildParams`, DDL `WITH`, descriptor | `64` | larger → better recall, more edge storage; default in both `BuildParams::default()` and the server `unwrap_or` |
| `l_build` (`L_build`, build beam) | `BuildParams`, DDL `WITH`, descriptor | `128` | forced to `max(l_build, r + 1)` inside `build` |
| `alpha` (`α`, prune diversification) | `BuildParams`, DDL `WITH`, descriptor | `1.2` | `> 1.0` keeps more diverse neighbours → higher recall |
| `init` (seed strategy) | `BuildParams` | `Auto` | brute-force below `AUTO_BRUTEFORCE_MAX = 4000`, random above |
| `metric` | DDL `METRIC`, descriptor | required | `cosine` \| `dot` \| `euclidean` |
| `dimension` | DDL `DIMENSION`, descriptor | required | enforced at write time and against each embedding at build time |
| `quantization` | DDL `WITH`, descriptor | `none` | `int8` is cosine-only (~4× smaller, recall floor ~0.80) |
| `ef` (query beam width) | procedures `ef`; natural-form `$__vector_ef` | per-round `max(kprime, 64)` | only ever clamped up (`ef.max(kprime)`); `$__vector_ef` is NON-STABLE |
| `k` (top-k) | query `LIMIT` / procedure `k` | `10` (procedures) | |
| over-fetch multiplier | executor, filtered path | `8 → 32 → 128 → 512` | `OVERFETCH_BASE=8`, `WIDEN_GROWTH=4`, `MAX_WIDEN_ROUNDS=4`; `1` with no filter |
| FastRP `dimension` | `FastRpOptions` | `256` | |
| FastRP `iteration_weights` | `FastRpOptions` | `[0, 1, 1, 1]` | |
| FastRP `normalization_strength` | `FastRpOptions` | `0.0` | |
| FastRP `seed` | `FastRpOptions` | `42` | |

---

### Proposed / Future

The items below are **not implemented**. They are recorded here so the
forward references scattered through the code and this RFC resolve.

- **First-class beam-width / query-hint surface (RFC-036).** Replace the
  non-stable global `$__vector_ef` parameter with a real query surface — an
  `OPTIONS { ef: … }` clause (reusing the DDL `WITH {…}` map shape) or a planner
  hint — plumbed into an `ef: Option<RowCount>` field on
  `LogicalPlan::VectorSearch`, forwarded through both executor dispatch sites
  into `vector_search_rows`. The executor would read the logical field and fall
  back to the param only while the bridge is being retired, so both can coexist
  during migration. Touches the parser/AST/lowering/optimizer/explain chain.

- **True pre-filtering / filtered-DiskANN (RFC-032).** Today a `WHERE`
  alongside a KNN is a *post*-filter: the predicate never enters Vamana
  navigation, so a selective filter widens (§7) and ultimately flat-falls-back.
  True pre-filtering makes the predicate count toward `k`/`ef` during traversal
  (non-matching nodes remain routing waypoints). The layering gap is that
  `beam_search` is sync and sees only ordinals, while the predicate lives two
  crates up over `NodeId` + properties. The recommended composition is
  attribute bitmaps over ordinals materialized at compaction (keyed by the
  index's `max_lsn`, so the existing freshness gate covers staleness for free)
  for the cheap equality slice, plus the residual `post_filter` and the
  adaptive widening from §7 as the safety net. Per-tenant subgraphs and a
  general ordinal→predicate callback are alternatives with worse trade-offs.

- **Incremental / mergeable index maintenance (RFC-035).** The index is
  rebuilt only on an authoritative (deepest) compaction, so a large delta sits
  on the flat-fallback path until then. An incrementally maintainable graph
  would shrink that freshness window.

- **Richer quantization and memory budgeting (RFC-035).** int8 is the only
  quantization today and is cosine-only. Product/binary quantization, and a
  per-index memory/recall budget that the build honours, would extend the
  storage win to larger corpora and other metrics.

- **ANN-aware cost model (future work).** The optimizer rewrites to `VectorSearch`
  whenever a descriptor matches, and `VectorGraph` SSTs contribute nothing to
  cost statistics. A recall/latency-aware cost model would let the planner
  choose between the index and the flat scan (and size the beam) on cost rather
  than on presence alone.

## Alternatives considered

- **HNSW instead of Vamana.** HNSW is the other dominant graph ANN, but its
  multi-layer structure and incremental insert model fit an in-memory mutable
  index better than an object-storage-first, rebuilt-on-compaction one. Vamana
  builds a single bounded-degree graph that serializes into one self-contained
  body and answers a top-k with one object fetch — a better match for the LSM.

- **A separate vector store / sidecar service.** Regains best-in-class ANN
  features at the cost of a second storage system, a second consistency story,
  and a second operational surface — breaking "your S3 bucket is the database."
  Making the index just another SST kind keeps one manifest, one snapshot, one
  cache.

- **Pre-normalize and store only int8 / only f32.** Storing only int8 would
  forfeit exact-metric reranking and the non-cosine metrics; storing only f32
  would forfeit the ~4× storage win. The format stores the original f32 vectors
  by default (rerank-with-true-metric, all three metrics) and offers int8 as an
  opt-in, cosine-only quantization.

- **True pre-filtering in v1.** Filtered-DiskANN needs label-aware graph
  construction or attribute bitmaps and a way to push a predicate into a sync,
  ordinal-only traversal — a real design with its own trade-offs (RFC-032).
  Shipping post-filter + adaptive widening + exact flat fallback gets correct
  filtered results now, with the index serving the common moderately-selective
  case, and defers the harder navigation change.

- **A first-class `ef` hint in v1.** No query-level hint/`OPTIONS` mechanism
  exists in the grammar. Rather than design and plumb one immediately, the
  procedures expose `ef` directly and the natural form gets the namespaced,
  explicitly non-stable `$__vector_ef` bridge that lands in the already-present
  `ef_search` slot with zero parser/plan changes (RFC-036 will do it properly).

## Drawbacks

- **Rebuild-not-merge.** Each authoritative compaction rebuilds the whole `.vg`
  for a touched index from the merged corpus. This is simple and always
  correct, but it is `O(N²)`-ish for brute-force init below
  `AUTO_BRUTEFORCE_MAX` and re-does work the previous build already did. Until
  RFC-035, a large freshly written delta stays on the flat-fallback path until
  the next authoritative merge.

- **Post-filter, not pre-filter.** A selective `WHERE` widens the over-fetch and
  can still flat-fall-back to `O(n)`. The widening bounds the damage (a
  moderately selective filter is served from the index), but a highly selective
  filter over a large label is a flat scan every query until RFC-032.

- **No checksum on the `.vg` body.** Integrity rests on `decode`'s range
  validation plus the flat fallback. A corrupt-but-in-range body could in
  principle return a wrong-but-plausible ranking rather than being rejected;
  the conservative validation and the exactness of the score reranking mitigate
  this, but a checksum would be stronger.

- **Two homes for the build defaults.** `BuildParams::default()` and the
  server's `unwrap_or(64/128/1.2)` must stay equal; they are commented to that
  effect, but the duplication is a maintenance hazard.

- **int8 is cosine-only and lossy.** It trades ~0.05 recall and the non-cosine
  metrics for a ~4× smaller body; it is not a free win.

- **Repeated decode/navigate under widening.** Each widening round re-decodes
  the `.vg` body and re-runs the beam search; geometric growth makes the final
  round dominate and the round cap bounds it at ≤4×, and it is still far cheaper
  than the `O(n)` flat scan it avoids — but a per-query decoded-index cache
  would help.

## Open questions

- Whether to add a body checksum to `.vg` (and how that interacts with the
  decode-error-as-absent fallback).
- Whether `$__vector_ef` should be retired hard once RFC-036 lands, or kept as
  a documented fallback during migration.
- The right default over-fetch growth (`×4`) and round cap (`4`): they were
  chosen to bound decode cost while serving moderately selective filters, but
  have not been tuned against production selectivity distributions.
- Whether to collapse the two homes of the Vamana defaults into a single shared
  constant exported by `namidb-ann`.

## References

- Subramanya, Devvrit, Kadekodi, Krishnaswamy, Simhadri, "DiskANN: Fast
  Accurate Billion-point Nearest Neighbor Search on a Single Node" (NeurIPS
  2019) — Algorithm 1 (RobustPrune), Algorithm 2 (Vamana). Cited in
  `namidb-ann/src/build.rs`.
- Chen, Yang, et al., "Fast and Accurate Network Embeddings via Very Sparse
  Random Projection" (FastRP, CIKM 2019). Cited in `namidb-graph/src/algo.rs`.
- RFC-002 (SST format / manifest descriptors), RFC-027 (compaction and space
  reclamation), RFC-026 (read-your-own-writes overlay — the freshness delta
  source).
- Conformance suite (the implementation's test coverage):
  `namidb-ann` `build.rs`/`search.rs`/`space.rs` unit tests (recall floors,
  determinism, robust-prune, out-of-range-entry safety, metric correctness);
  `namidb-core` `quantize.rs` round-trip/finiteness tests;
  `namidb-storage` `sst/vector.rs` (all-three-metrics faithful, int8
  smaller-and-recalls, decode rejects bad-magic/out-of-range-entry, int8
  requires cosine) and the `compact.rs` end-to-end-through-real-compaction
  test; `namidb-query` `tests/exec_vector_knn.rs` (freshness delta-union +
  filtered ANN) and `exec_hybrid_search.rs`; `namidb-bench` `ann_bench.rs`
  (recall@k + a `cypher_index_path_reachable` check that asserts the plan
  actually uses the index, not just that results match the flat scan).
</content>
</invoke>
