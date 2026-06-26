# RFC 031: ANN recall@k-vs-QPS benchmark methodology

**Status:** draft
**Author(s):** NamiDB team
**Created:** 2026-06-26
**Updated:** 2026-06-26
**Implements:** (this RFC documents the existing `namidb-bench` vector tracks and proposes the harness additions needed for a recall-vs-QPS curve)

## Summary

This RFC defines a rigorous, reproducible methodology for answering the only
question that matters for the Vamana `.vg` ANN index: **does recall hold as the
corpus grows?** It pins a self-contained synthetic protocol — clustered
unit-norm vectors with a brute-force `exact_top_k` ground truth, fixed seed — so
anyone can reproduce a recall@k number with no external data, then layers the
honest latency/throughput axis on top. It publishes the numbers we have
*actually measured today* alongside the **full parameter set** for each, because
recall on this index is workload- and `ef`-sensitive (a measured `recall@10 =
0.68` on a hard corpus versus the `namidb-ann` unit test's `>= 0.90` on an easy
one — see [Why 0.68 and 0.90 are both true](#why-068-and-090-are-both-true)).
It separates what runs in `namidb-bench` today from the small harness additions
needed for a real recall-vs-`ef`/QPS curve, and reserves a clearly-labelled
future track for the external `ann-benchmarks` HDF5 datasets (`sift-128`,
`glove-100-angular`, `gist-960`) that make a number *publishable* rather than
merely *internal*.

The guiding rule: **synthetic data is for CI and iteration; HDF5 is for
publishable parity.** Neither replaces the other.

## Motivation

The engine ships a DiskANN/Vamana index (`SstKind::VectorGraph`, the `.vg`
body) materialised by compaction, with int8 quantization (RFC-030,
the 1.4 release) and a `VectorSearch` optimizer rewrite. The standing claim is
"recall stays high while the index out-runs a flat scan." That claim is only as
good as the evidence behind it, and today the evidence has three honesty gaps:

1. **A single recall number is not a guarantee.** Recall on a graph index is a
   function of `(dim, num_vectors, clusters, spread, ef, k, metric,
   quantization)`. Publishing one number without its parameter vector invites
   exactly the over-claiming we want to avoid: the `namidb-ann` unit test
   asserts `recall@10 >= 0.90` (`crates/namidb-ann/src/build.rs:386`) while a
   real-engine run on a larger, higher-dimensional corpus measures `0.68`. Both
   are correct; the difference is entirely the workload and is invisible unless
   the parameters travel with the number.

2. **There is no QPS axis.** `ann-bench` emits serial per-query `p50`/`p99`
   latency, never queries-per-second under concurrency. "QPS at scale" is the
   industry-standard x-axis (`ann-benchmarks.com`), and we do not produce it.

3. **There is no single-process `ef` sweep.** `--ef` is one value
   (`crates/namidb-bench/src/main.rs:174`) and the `.vg` build runs once *per
   invocation* (measured at 42 s for 5 000 vectors). Producing a recall-vs-`ef`
   curve today means rebuilding the index N times.

The cost of doing nothing is that we keep asserting "recall holds at scale" from
one cherry-picked point. This RFC makes the claim falsifiable and reproducible.

## Design

### Two harnesses, two jobs

The vector work already ships two distinct, deterministic harnesses. They are
not redundant; they measure different layers.

| Harness | Engine? | What it measures | Source |
|---|---|---|---|
| `vector-recall` | no (pure arithmetic) | int8-vs-f32 quantization recall@k + latency + bytes/vector | `crates/namidb-bench/src/vector_recall.rs` |
| `ann-bench` | yes (real `.vg`) | Vamana index recall@k vs exact flat KNN + index-vs-scan latency + Cypher reachability | `crates/namidb-bench/src/ann_bench.rs` |

`vector-recall` answers "is int8 a safe on-disk format?" by quantizing stored
vectors and scoring with the asymmetric scorer the engine actually uses
(`namidb_core::quantize::dot_i8_asymmetric`), then comparing the int8 ranking
against the exact f32 ranking. No graph, no compaction — it isolates the
quantization arithmetic. `ann-bench` answers "does the *index* return the right
neighbours?" end to end: it builds a namespace on an `InMemory` object store,
registers a cosine `VectorIndexDescriptor`, writes the corpus across two L0
SSTs, runs `compact_l0` so the compactor materialises the `.vg`, and then scores
the graph's top-k against a brute-force `exact_top_k`.

### The synthetic protocol (self-contained ground truth)

Both harnesses share one generator (`vector_recall::random_unit_vector` /
`perturbed_unit_vector`), seeded by a `ChaCha8Rng` from `--seed` (default 42),
so every run is bit-for-bit reproducible.

- **Corpus.** With `--clusters N > 0`, draw `N` random centroids on the unit
  sphere, then place each stored vector as `centroid + spread·gaussian`,
  renormalised (`--spread` default 0.35). This mimics real embeddings, where
  true neighbours sit in the same semantic cluster and are well separated from
  the rest.
- **Queries.** Drawn around the same centroids but tighter — query spread is
  `spread · 0.5` (`ann_bench.rs:167`) — so each query's true top-k are that
  cluster's members.
- **Ground truth.** `exact_top_k` is an exact brute-force cosine ranking over
  the identical corpus (`ann_bench.rs:115`). Because it is exact, it is *both*
  the recall oracle and the compute floor for the flat-scan latency. There is no
  external label set to download or trust.
- **The pessimistic floor.** `--clusters 0` draws the corpus and queries
  uniformly on the sphere. Random high-dimensional vectors have no meaningful
  neighbours, so near-tied ranks reshuffle under any approximation — this is the
  worst case and the honest lower bound. Always run it next to the clustered
  case; a recall number without its floor is marketing.

**Parameter grid this RFC standardises on:**

- `dim ∈ {256, 768, 1536}` — local embedders (256), sentence-transformer class
  (768), and OpenAI `text-embedding-3` (1536).
- `num ∈ {10_000, 100_000, 1_000_000}` — the "at scale" ladder. See the
  [build-cost wall](#the-build-cost-wall-at-1m) for why 1M is not yet runnable
  unattended.
- `clusters ∈ {0, 64, 256, 1024}` — `0` is the floor; the rest sweep how
  separable the neighbourhoods are.
- `k = 10`, `ef ∈ {16, 32, 64, 128, 256}` — the recall/latency knob.

### What `recall@k` means here, exactly

Per query, let `A` be the index's top-k node ids and `T` the exact top-k. The
harness accumulates `hits += |A ∩ T|` and `total += |T|`, then reports

```text
recall@k = (Σ over queries |A ∩ T|) / (Σ over queries |T|)
```

(`ann_bench.rs:266-276`). With `num ≥ k` every `|T| = k`, so this is the mean
over queries of `|A ∩ T| / k` — the standard ANN recall@k. The comparison is
keyed on corpus index (the harness maps each `NodeId` hit back to its generated
vector via `id_to_idx`), so it is exact set-membership, not an approximation of
the metric. `ef` is clamped up to at least `k` (`ann_bench.rs:144`); a higher
`ef` only raises recall, never corrupts it.

### The QPS axis — what we have and what we propose

**Honest status: the harness emits serial `p50`/`p99` latency, not QPS.** The
loop at `ann_bench.rs:248-268` times one query at a time on one thread; there is
no concurrency and no queries-per-second anywhere in `crates/namidb-bench/src`.
A single-thread `QPS = 1 / mean_latency` is the *throughput floor*, not
throughput under load, and must be labelled as such.

The reported `speedup_p50 = flat_p50 / index_p50` (17.2× measured below) is a
**compute-only** comparison: both the index path (`Snapshot::vector_search`,
`crates/namidb-storage/src/read.rs:2231`) and the flat path (`exact_top_k`)
exclude row materialisation, which each would pay alike. It is therefore a
conservative lower bound on what a real end-to-end scan costs.

The following are **proposed harness additions**, each on the order of tens of
lines, scoped here so the methodology is complete even before they land. They
are *not* implemented today.

- **`--ef-list 16,32,64,128,256` (single-build sweep).** Refactor `ann_bench::run`
  so the build phase (`ann_bench.rs:196-219`) runs once and the query loop runs
  once per `ef` against the *same* `.vg`, emitting a JSON array of
  `AnnBenchReport`. This turns an N-point recall-vs-`ef` curve from N builds into
  one, killing the repeated 42 s build. Keep single-`--ef` output as a
  one-element array (or gate the array behind `--ef-list`) so existing
  one-object consumers do not break.
- **`--concurrency M` + `qps`/`mean_us` fields.** Wall-clock the whole query
  batch and report `qps = num_queries / total_search_secs`; with `--concurrency
  M`, spawn `M` tokio tasks sharing the read-only `Arc<Snapshot>` for true
  throughput-under-load. The snapshot is immutable so this is safe, but a
  multi-core box inflates QPS versus a 1-core deployment — **pin and report the
  core count** with every QPS number.
- **`--quantization {none,int8}` on the `.vg`.** `ann-bench` hardcodes
  `VectorQuantization::None` (`ann_bench.rs:189`). Parametrising it exercises the
  int8 path on the *real index*, not just `vector-recall`'s arithmetic. Valid
  because the bench uses `VectorMetric::Cosine` and int8 requires cosine
  (`crates/namidb-storage/src/manifest.rs:430-445`); int8 with euclidean/dot must
  error clearly, never silently fall back.
- **`--filter-keep <frac>` through the executor.** Today `ann-bench` calls the
  low-level `Snapshot::vector_search` reader directly, which has no filter
  support, so it bypasses the executor's over-fetch path. To bench *filtered*
  ANN honestly, write a `bucket: Int` property per `Doc` and route the KNN
  through the executor — `parse`/`lower`/`optimize`/execute a `MATCH (d:Doc)
  WHERE d.bucket < N AND d.embedding IS NOT NULL ... ORDER BY score DESC LIMIT k`
  — so the adaptive over-fetch in `try_index_search`
  (`crates/namidb-query/src/exec/walker.rs:3033`) fires: it starts at `×8`
  (`OVERFETCH_BASE`) and widens `×4` per round up to `512×` (`MAX_WIDEN_ROUNDS =
  4`) before any O(n) flat fallback, so a selective filter is still served from
  the index. Measure filtered *and* unfiltered through the **same** executor path
  — filtered latency includes freshness merge + full-node materialisation and is
  not comparable to the raw-reader unfiltered number. The `search.vector` /
  `search.hybrid` / `db.index.vector.queryNodes` procedures
  (`walker.rs:2241-2398`) already expose a tunable `ef`, so a procedure-driven
  bench gets the knob for free.

### Measured results (real numbers, full parameters)

Everything below was produced by running the harnesses under `--release` on this
tree. Each number carries its **entire** parameter set; do not quote a cell
without its row.

**`ann-bench` — real `.vg` index, cosine, `quantization=None`, seed 42.** A
recall-vs-`ef` sweep on the SAME corpus (each row rebuilds the index; `cypher
reaches index = true` for every row), which is the direct answer to "does recall
hold?": recall is a **tunable dial**, not a fixed property of the index.

| dim | num | clusters | ef | k | queries | recall@10 | index p50 | flat p50 | speedup | build |
|---|---|---|---|---|---|---|---|---|---|---|
| 256 | 5 000 | 64  | 64  | 10 | 50 | **0.68**  | 4 462 µs | 75 837 µs | 17.0× | 42.2 s |
| 256 | 5 000 | 256 | 128 | 10 | 50 | **0.898** | 5 173 µs | 75 520 µs | 14.6× | 43.4 s |
| 256 | 5 000 | 64  | 256 | 10 | 50 | **0.99**  | 6 083 µs | 75 768 µs | 12.5× | 43.3 s |

So at a wider beam the same index reaches **0.99 recall@10 while still serving
~12.5× faster than the exact flat scan** — the `ef=64` row is a deliberately-hard,
low-`ef` operating point, not a ceiling. The latency cost of recall is sub-linear
in `ef` here (4.5→6.1 ms p50 for a 0.68→0.99 recall jump). Tighter clustering
(`clusters=256`) raises recall at a given `ef` because the graph's neighbourhoods
are more separable.

Commands that produced these exact rows:

```bash
# recall 0.68 (hard, low ef)
cargo run --release -p namidb-bench --features vector-index -- ann-bench \
    --dim 256 --num 5000 --queries 50 --k 10 --clusters 64 --ef 64
# recall 0.898 (more realistic clustering)
cargo run --release -p namidb-bench --features vector-index -- ann-bench \
    --dim 256 --num 5000 --queries 50 --k 10 --clusters 256 --ef 128
# recall 0.99 (wide beam)
cargo run --release -p namidb-bench --features vector-index -- ann-bench \
    --dim 256 --num 5000 --queries 50 --k 10 --clusters 64 --ef 256
```

**`vector-recall` — int8 quantization arithmetic (no engine), seed 42:**

| dim | num | clusters | k | queries | recall@10 (per-vec scale) | recall@10 (fixed-127) | exact p50 | int8 p50 | compression |
|---|---|---|---|---|---|---|---|---|---|
| 256 | 5 000 | 64 | 10 | 50 | **0.988** | 0.936 | 1 163 µs | 1 246 µs | **3.94×** |

Command that produced this exact row:

```bash
cargo run --release -p namidb-bench -- vector-recall \
    --dim 256 --num 5000 --queries 50 --k 10 --clusters 64
```

The compression ratio is `4·dim / (dim + 4)` (`vector_recall.rs:233`): 3.94×
at dim 256, rising to ~3.98× at 768 and ~3.99× at 1536. The per-vector max-abs
scale (0.988) beats a naive fixed-127 scale (0.936) decisively — this is the
arithmetic justification for the on-disk int8 format choice
(`VectorQuantization::Int8`).

#### Why 0.68 and 0.90 are both true

The measured `recall@10 = 0.68` (`ann-bench`, above) sits *below* the
`namidb-ann` unit test's asserted `recall@10 >= 0.90`
(`crates/namidb-ann/src/build.rs:386`). This is not a contradiction — it is the
entire reason this RFC exists. Both runs use `ef = 64`, `k = 10`, `alpha = 1.2`;
they differ only in the workload:

| | unit test `recall_on_clustered_data_f32` | measured `ann-bench` |
|---|---|---|
| `num` | 500 | 5 000 |
| `clusters` | 20 | 64 |
| vectors / cluster | 25 | ~78 |
| `dim` | 48 | 256 |
| query spread | 0.10 | 0.175 (`0.35·0.5`) |
| build `r` / `l_build` | 32 / 64 | 32 / 64 |
| recall@10 | ≥ 0.90 | 0.68 |

The bench corpus is larger, ~3× denser per cluster, 5× higher-dimensional, and
queried with looser perturbation — every axis makes the top-10 harder to recover
at a fixed beam width. The unit test deliberately picks an *easy* corpus to
guard against graph-build regressions; the bench deliberately picks a *harder*
one to avoid over-claiming. A single point proves nothing about scale; the
recall-vs-`ef` curve (proposed `--ef-list`) is what actually answers "does recall
hold?", and the expectation is monotonic non-decreasing recall as `ef` grows.

### Commands that run today, no external data

All of these work on the current tree. The 5 000-vector rows above are *measured*;
the larger invocations are *runnable but not yet measured here* — run them to
populate the grid.

```bash
# 1. Measured int8 arithmetic recall (the row above).
cargo run --release -p namidb-bench -- vector-recall \
    --dim 256 --num 5000 --queries 50 --k 10 --clusters 64

# 2. int8 recall at OpenAI dimensionality (runnable; populate the grid).
cargo run --release -p namidb-bench -- vector-recall \
    --dim 1536 --num 10000 --k 10 --clusters 256

# 3. Measured real-index recall + latency (the ann-bench row above).
cargo run --release -p namidb-bench --features vector-index -- ann-bench \
    --dim 256 --num 5000 --queries 50 --k 10 --clusters 64 --ef 64

# 4. Larger realistic corpus (runnable; build cost grows — see the wall below).
cargo run --release -p namidb-bench --features vector-index -- ann-bench \
    --dim 256 --num 50000 --queries 200 --k 10 --clusters 256 --ef 64

# 5. Pessimistic floor (uniform sphere, no meaningful neighbours).
cargo run --release -p namidb-bench --features vector-index -- ann-bench \
    --dim 256 --num 50000 --queries 200 --k 10 --clusters 0

# 6. Recall-vs-ef curve TODAY (rebuilds the .vg each iteration — slow but
#    correct; this is exactly what the proposed --ef-list collapses to 1 build).
for ef in 16 32 64 128 256; do
  cargo run --release -p namidb-bench --features vector-index -- ann-bench \
      --dim 256 --num 5000 --queries 50 --k 10 --clusters 64 --ef "$ef"
done
```

Always run `--release`: a debug build inflates both the graph build and the
per-query latency several-fold (`bench/README.md`).

### The build-cost wall at 1M

The Vamana build is a **single-threaded sequential** refinement pass —
`for &i in &order { beam_search(...); robust_prune(...) }`
(`crates/namidb-ann/src/build.rs:157`) — with no rayon anywhere in the build.
It cost 42 s for 5 000 vectors. The pass is roughly `O(n · l_build · log n)`
beam searches, so `num = 1_000_000` is hours of wall-clock and the *build*, not
the search, becomes the experiment's bottleneck. An honest "at scale" answer at
1M is therefore **blocked on build parallelism** (rayon over the refinement
order, or compaction-side sharding). Until that lands, treat `num = 100_000` as
the practical ceiling for an unattended run and call out 1M as a prerequisite,
not a result.

## Implemented today (v1.4) vs proposed

### Implemented and runnable now

- `vector-recall`: int8-vs-f32 quantization recall@k, latency, compression
  (`crates/namidb-bench/src/vector_recall.rs`). No engine.
- `ann-bench`: real `.vg` recall@k vs exact flat KNN, serial `p50`/`p99`
  index-vs-scan latency, `cypher_index_path_reachable`
  (`crates/namidb-bench/src/ann_bench.rs`, behind `--features vector-index`).
- Deterministic clustered + uniform-floor synthetic generator, fixed `--seed`.
- The `clusters = 0` pessimistic floor.

### Proposed (this RFC, not yet landed)

- `--ef-list` single-build recall-vs-`ef` sweep emitting a JSON array.
- `--concurrency M` + `qps`/`mean_us` fields for true throughput-under-load.
- `--quantization {none,int8}` wired into the `.vg` descriptor for real-index
  int8 recall/latency/size.
- `--filter-keep <frac>` routed through the executor over-fetch path for
  filtered-ANN recall at selectivities `{1.0, 0.5, 0.1, 0.01}`.
- A nightly (not per-PR) recall smoke gate; see CI below.

### Future: external HDF5 parity

The synthetic protocol is for **CI and fast iteration**. It cannot make a
*publishable* claim, because reviewers trust the field's shared datasets, not our
generator. The follow-up track imports the standard `ann-benchmarks` HDF5
corpora (no `hdf5` reader exists anywhere in the repo today — this is greenfield):

- **`sift-128-euclidean`** — 1M SIFT image descriptors, 128-d, euclidean. The
  canonical "does it scale to 1M" dataset (and the one that needs the parallel
  build above).
- **`glove-100-angular`** — 1.18M GloVe word vectors, 100-d, angular/cosine.
- **`gist-960-euclidean`** — 1M GIST descriptors, 960-d, the high-dimensional
  stress case.

Each HDF5 file ships a `train` matrix, a `test` query matrix, and a
`neighbors` ground-truth matrix, so recall is scored against the *dataset's*
labels rather than our brute force. This is what lets us plot a recall-vs-QPS
curve directly comparable to published results. euclidean datasets require the
`.vg` to use `VectorMetric::Euclidean`; angular maps to `Cosine`. **Synthetic
for CI, HDF5 for parity — ship both, conflate neither.**

## Alternatives considered

- **Keep emitting a single recall number per run.** Cheapest, and what we do
  today. Rejected: a point is not a curve, and the 0.68-vs-0.90 gap shows a
  single point invites over-claiming. The methodology must carry parameters with
  every number.
- **Go straight to HDF5 and skip the synthetic track.** Tempting for
  credibility, but HDF5 corpora are hundreds of MB to download, cannot run in CI
  without network and disk, and the 1M builds need parallelism we do not have.
  The synthetic generator gives a deterministic, zero-dependency recall gate
  *now*; HDF5 is the paired follow-up, exactly as the LDBC bench pairs synthetic
  data with the real SNB datagen (`bench/README.md`).
- **Approximate QPS as `1 / serial_p50` and call it throughput.** Rejected as
  dishonest: serial latency ignores contention, cache effects under load, and
  core count. The `--concurrency` path measures real throughput; the serial
  number is explicitly the floor.
- **Bench filtered ANN through the low-level reader.**
  `Snapshot::vector_search` has no filter argument
  (`crates/namidb-storage/src/read.rs:2231`), so filtered ANN *cannot* be
  measured there. Routing through the executor is the only path that exercises
  the over-fetch/widening logic real queries hit.

## Drawbacks

- Synthetic clustered data is kind to a graph index — well-separated clusters
  are close to the easy case. The `clusters = 0` floor and the eventual HDF5
  track exist precisely to counter this, but anyone reading a clustered number in
  isolation can still over-read it. Mitigation: never publish a clustered number
  without its `clusters = 0` neighbour.
- The proposed `--concurrency` QPS is sensitive to host core count; a number from
  a 32-core CI box overstates a 1-core deployment. Mitigation: pin and report
  cores.
- Changing `ann-bench` stdout from one JSON object to an array for `--ef-list`
  risks breaking any one-object consumer. Mitigation: single-`--ef` stays a
  1-element array, or the array is gated behind `--ef-list`.
- The full grid is expensive: `dim × num × clusters × ef` with a 42 s/5k build
  is a nightly, not a per-PR job. CI (`.github/workflows/ci.yml`) already
  excludes benches; a fast smoke (`--num 2000 --queries 20 --clusters 32 --ef
  64`, recall floor ~0.6) can gate graph-build regressions nightly, while the
  per-PR guard stays the library-level `namidb-ann` recall tests
  (`crates/namidb-ann/src/build.rs:356,392`) that build without compaction.

## Open questions

- Whether to land build parallelism (rayon over the refinement order) before or
  alongside the 1M HDF5 track — the 1M result is gated on it either way.
- The recall floor for the nightly smoke gate: `0.6` is conservative given the
  observed `0.68` at 5k/clusters=64/ef=64, but the right value depends on the
  smoke's exact parameters and wants its own short measurement pass.
- Whether filtered-ANN recall should be scored against the *filtered* ground
  truth (`exact_top_k` restricted to the kept bucket) — it should, so that
  over-fetch's compensation for selectivity is what is being measured, not a
  selectivity penalty mislabelled as a recall loss.
- Which HDF5 reader to depend on, and whether to vendor a minimal parser rather
  than pull a full `hdf5` crate into the bench's dependency graph.

## References

- `crates/namidb-bench/src/ann_bench.rs` — real-engine `.vg` recall/latency
  harness (`run`, `AnnBenchReport`, `exact_top_k`, `cypher_reaches_index`).
- `crates/namidb-bench/src/vector_recall.rs` — int8 quantization recall harness
  (`run`, `VectorRecallReport`).
- `crates/namidb-bench/src/main.rs:149-178` — the `AnnBench` CLI subcommand
  (single `--ef` today).
- `crates/namidb-ann/src/build.rs:157` — single-threaded Vamana refinement loop
  (the build-cost wall); `:356`/`:392` — the `recall_on_clustered_data_f32`
  (≥ 0.90) and `recall_int8_tracks_f32` (≥ 0.80) library tests.
- `crates/namidb-storage/src/read.rs:2231` — `Snapshot::vector_search`, the
  low-level `.vg` reader the bench calls (no filter support).
- `crates/namidb-storage/src/manifest.rs:428-480` — `VectorQuantization`
  (int8 requires cosine), `VectorIndexDescriptor`.
- `crates/namidb-query/src/exec/walker.rs:3033` — `try_index_search` adaptive
  over-fetch/widening (`OVERFETCH_BASE = 8`, `×4`, up to `512×`); `:2241-2398` —
  the `search.vector` / `db.index.vector.queryNodes` procedures with a tunable
  `ef`.
- `bench/README.md` — the existing vector/ANN runbook this methodology extends.
- The `ann-benchmarks` project (`ann-benchmarks.com`) — recall-vs-QPS
  methodology and the `sift-128`, `glove-100-angular`, `gist-960` HDF5 datasets.
- DiskANN / Vamana (Subramanya et al., NeurIPS 2019) — the index this benchmarks.
