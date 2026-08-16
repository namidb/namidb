# NamiDB LDBC-shaped bench

A synthetic, deterministic benchmark for the four LDBC SNB Complex Read
queries that NamiDB already runs end to end today:

| Query | Shape | Notes |
|---|---|---|
| **IC02** | `friend <- KNOWS <- p; msg -> HAS_CREATOR -> friend` | recent messages by friends |
| **IC07** | `p <- HAS_CREATOR <- msg <- LIKES <- fan` | recent likers of my messages |
| **IC08** | `p <- HAS_CREATOR <- post <- REPLY_OF <- reply` | recent replies to my messages |
| **IC09** | `p -> KNOWS -> friend -> KNOWS -> fof <- HAS_CREATOR <- msg` | recent messages by friends-of-friends |

Same dataset, same Cypher (modulo each backend's parameter syntax), same
warm-up and sample-count protocol. The output JSON is shape-compatible,
so you can diff the two backends directly.

## Layout

- `crates/namidb-bench/`: the Rust crate. Subcommands:
  - `generate`: write the dataset out as CSV files.
  - `run`: load (or reuse) the dataset into an in-memory NamiDB
    namespace, time each query, and print JSON.
  - `vector-recall`: in-memory int8-quantization recall@k vs exact f32 (the gate
    for the on-disk int8 vector format). No engine, pure arithmetic.
  - `ann-bench` (needs `--features vector-index`): the Vamana ANN **index**
    recall@k vs the exact flat KNN, plus index-vs-scan latency, over the real
    engine — see [Vector / ANN benchmarks](#vector--ann-benchmarks).
  - `object-native` (needs `--features object-native`): acceptance gates for
    the range-readable V5/FT3 base formats, incremental VG6/FT4 Search-LSM and
    forward/inverse paged graph SSTs. It builds through the external-memory
    writers and queries through instrumented immutable byte ranges with
    explicit zero/sized search caches.
- `bench/kuzu_runner.py`: a Python harness that runs the same CSVs
  against Kuzu (via the `kuzu` PyPI package).

## End-to-end workflow

```bash
# 0. Prerequisites.
rustup show          # cargo + rustc stable
python3 -m pip install kuzu

# 1. Generate the synthetic dataset once. Pick scale to fit RAM:
#    scale=1.0 -> 10k Person + 100k Post + 50k Comment + ~430k edges.
#    scale=0.1 -> ~1k Person + ~10k Post + ...  (fast smoke).
DATASET=/tmp/snb-0.1
cargo run --release -p namidb-bench -- generate \
    --scale 0.1 --seed 42 --out "$DATASET"

# 2. Bench NamiDB.
cargo run --release -p namidb-bench -- run \
    --scale 0.1 --dataset-dir "$DATASET" \
    --warm-runs 50 --param-count 3 \
    > /tmp/bench-namidb.json

# 3. Bench Kuzu over the SAME CSVs.
python3 bench/kuzu_runner.py \
    --dataset-dir "$DATASET" \
    --warm-runs 50 --param-count 3 \
    > /tmp/bench-kuzu.json

# 4. Diff: every record has (query, param, rows, cold_us,
#    warm_p50_us, warm_p95_us, warm_p99_us). Quick eyeballing:
jq -r '.results[] | [.query, .param[:8], .rows, .warm_p50_us] | @tsv' \
    /tmp/bench-namidb.json
jq -r '.results[] | [.query, .param[:8], .rows, .warm_p50_us] | @tsv' \
    /tmp/bench-kuzu.json
```

## How the dataset is structured

| Label / type | Count @ scale=1.0 | Property columns |
|---|---|---|
| `Person` | 10 000 | firstName, lastName, age, creationDate |
| `Post` | 100 000 | content, creationDate, length |
| `Comment` | 50 000 | content, creationDate, length |
| `KNOWS` (Person -> Person) | 100 000 | since |
| `HAS_CREATOR` (Post/Comment -> Person) | 150 000 | (none) |
| `LIKES` (Person -> Post/Comment) | 100 000 | creationDate |
| `REPLY_OF` (Comment -> Post/Comment) | 30 000 | (none) |

The generator is `crates/namidb-bench/src/dataset.rs`. The RNG is
`ChaCha8Rng(seed=42)` by default, so two runs at the same scale produce
identical files (and Kuzu sees the same edges as NamiDB).

Node ids are 32-hex-char strings (16 bytes) with a prefix byte tagging
the label (`P=Person, O=Post, C=Comment`), so the same numeric index
maps to distinct ids across labels.

## Object-store-native acceptance gates

`object-native` produces one `namidb-object-native-gate-v2` JSON document. It
retains the original V5/FT3 acceptance tracks unchanged: one
`NAMIVG05` artifact for each of cosine, dot-product and L2, and one `NAMIFT03`
artifact. Corpus rows are regenerated from `(seed, ordinal)` and streamed into
the external builders; the harness does not keep a second vector corpus in
memory. Exact vector top-k is also computed as a bounded streaming pass.

For every vector metric the report includes build time and spool accounting,
artifact size, decoded reader metadata, exact recall@k, a selective native
`tenant` filter, and cold/warm p50+p95. Full text checks ordinary BM25, a
quoted phrase, a prefix and a combined phrase+prefix query against an
independent exact scorer. The RangeSource reports logical requests, underlying
fetches, fetched bytes, cache hits and peak in-flight requests. On Linux the
report also records `/proc/self/status` RSS and high-water RSS.

V2 adds two production-architecture sections:

- `search_lsm`: an authoritative clustered `NAMIVG05` vector `Base` and an
  authoritative absolute FT4 text `Base`, each followed by `1..N` ordered
  VG6/FT4 mutation deltas. Every delta has deterministic payload updates,
  tombstones and filter-only updates. The report asserts that V5 has
  `SearchSegmentRole::Base`, `SearchSegmentFormat::VectorV5Base`, an absolute
  live count, no suppressions and a non-empty clustered page directory. It
  likewise requires FT4 Base plus absolute document-count/total-length
  statistics and no suppressions. A bench-only coordinator uses the public V5,
  VG6 and FT4 readers and `NAMISV01` version tables, rejects a V5 candidate when
  any newer VG6 winner exists, widens stale-heavy segment candidate lists, and
  accepts only the exact highest-LSN live fingerprint. `vector` is the real
  serving path: clustered V5 `nprobe`/adaptive widening/rerank plus exhaustive
  small VG6 deltas. Its filtered/unfiltered recall is measured against an
  independently regenerated final-corpus oracle with cache capacity zero, a
  cleared sized cache, and the same sized cache warm. Returned winner versions
  and f32 scores are mandatory even when ANN top-k IDs differ from the oracle.
  `vector_exact_shadow` separately exhausts V5 one page at a time with an
  `O(k)` heap to prove exact Base+delta reconciliation. It is correctness-only:
  its range traffic and latency are explicitly excluded from every serving
  I/O/latency field. The JSON carries serving physical fetch bytes/requests,
  logical reads, cache hits, fanout, peak in-flight requests, reader metadata
  and builder logical high-water.
- `graph`: independently built forward and inverse v1.2 edge SSTs, each with a
  deliberately high-degree hub, regular adjacency, exact endpoint lookup and
  exact `codigo` property-row hydration. It opens each direction through the
  public explicit-cache constructor twice: once with `cache=None`, and once
  with an isolated bounded `ImmutableRangeCache`. Instrumentation below the
  cache distinguishes reader-logical ranges from physical object-store GETs.
  Every operation is measured no-cache, sized-cold after a clear, and
  sized-warm as an immediate identical repeat. Parity and range completeness
  are mandatory, eager-body fallback is forbidden, and a warm pass fetching
  any backing-store byte fails the gate.

The public V5 external builder exposes an explicit authoritative-Base
constructor and emits the existing clustered, range-readable V5 artifact
without retaining a corpus `Vec`; the public FT4 external builder likewise
emits its real absolute `Base`. The fixture therefore has no Search-LSM API
limitation: vector is V5 Base + VG6 deltas and text is FT4 Base + FT4 deltas.

`builder_workspace` reports the maximum logical workspace observed across V5,
FT3, VG6 and FT4. `rss` reports current RSS before/after plus Linux `VmHWM`.
Exact-shadow/vector-winner/text parity gates are unconditional. Serving ANN
recall becomes a release gate only when the reviewed optional
`--min-recall-at-k` is supplied. Fanout defaults to 32 and immutable range
concurrency to 16; both are configurable.

Threshold arguments are optional. When supplied, every failure is retained in
`gates.failures`, the complete JSON is still written to stdout, and the process
exits non-zero. This makes a failed CI run diagnosable without parsing stderr.

```bash
# Canonical deterministic V2 smoke/freeze command. This is the command to use
# before changing the report schema or any object-native reader contract.
CARGO_BUILD_JOBS=1 cargo run --release -p namidb-bench \
  --features object-native -- object-native \
  --vectors 256 --documents 256 --dim 16 --queries 3 --k 5 \
  --clusters 4 --filter-buckets 4 --page-rows 32 \
  --branch-factor 4 --nprobe 4 --max-nprobe 16 --rerank-factor 8 \
  --delta-segments 2 --graph-keys 64 --graph-high-degree 128 \
  --cache-bytes 2097152 --build-memory-bytes 2097152 \
  --max-fanout 32 --max-in-flight 16 --seed 42 \
  > /tmp/namidb-object-native-v2-smoke.json

jq -e '
  .format == "namidb-object-native-gate-v2" and
  .gates.passed and
  # Every parity assertion below asks the reader for exactly as many hits as the
  # exact oracle produced, and an empty-vs-empty comparison is trivially equal.
  # Require the oracle to have found something first, or a corpus regression
  # would silently turn the whole contract into a tautology.
  (.search_lsm.vector_serving_ann.unfiltered.parity.expected_hits > 0) and
  (.search_lsm.vector_serving_ann.active_filter.parity.expected_hits > 0) and
  (.search_lsm.vector_exact_shadow.unfiltered.expected_hits > 0) and
  (.search_lsm.vector_exact_shadow.active_filter.expected_hits > 0) and
  (.search_lsm.text.unfiltered.parity.expected_hits > 0) and
  (.search_lsm.text.active_filter.parity.expected_hits > 0) and
  (.search_lsm.vector_serving_ann.unfiltered.query_count > 0) and
  (.search_lsm.text.unfiltered.query_count > 0) and
  .search_lsm.vector_serving_ann.unfiltered.result_model == "serving_ann_vs_exact_oracle" and
  .search_lsm.vector_serving_ann.unfiltered.returned_winners_valid and
  .search_lsm.vector_serving_ann.unfiltered.returned_scores_exact and
  .search_lsm.vector_serving_ann.unfiltered.cache_results_identical and
  (.search_lsm.vector_serving_ann.unfiltered.recall_at_k >= 0 and
   .search_lsm.vector_serving_ann.unfiltered.recall_at_k <= 1) and
  .search_lsm.vector_serving_ann.active_filter.native_filter_applied and
  .search_lsm.vector_serving_ann.active_filter.returned_winners_valid and
  .search_lsm.vector_serving_ann.active_filter.returned_scores_exact and
  .search_lsm.vector_serving_ann.active_filter.cache_results_identical and
  (.search_lsm.vector_serving_ann.active_filter.recall_at_k >= 0 and
   .search_lsm.vector_serving_ann.active_filter.recall_at_k <= 1) and
  .search_lsm.vector_exact_shadow.unfiltered.node_ids_exact and
  .search_lsm.vector_exact_shadow.unfiltered.scores_exact and
  .search_lsm.vector_exact_shadow.active_filter.node_ids_exact and
  .search_lsm.vector_exact_shadow.active_filter.scores_exact and
  .search_lsm.v5_base_contract.role_is_base and
  .search_lsm.v5_base_contract.format_is_v5 and
  .search_lsm.v5_base_contract.live_count_is_absolute and
  .search_lsm.v5_base_contract.suppress_count_is_zero and
  .search_lsm.v5_base_contract.native_footer_validated and
  .search_lsm.v5_base_contract.legacy_manifest_binding_nonzero and
  .search_lsm.vector_segment_roles == ["base", "delta", "delta"] and
  .search_lsm.vector_segment_stats == ["absolute", "delta", "delta"] and
  .search_lsm.text_segment_stats == ["absolute", "delta", "delta"] and
  .search_lsm.text.unfiltered.parity.node_ids_exact and
  .search_lsm.text.active_filter.parity.node_ids_exact and
  .search_lsm.text.active_filter.native_filter_applied and
  .search_lsm.ft4_base_contract.role_is_base and
  .search_lsm.ft4_base_contract.doc_count_is_absolute and
  .search_lsm.ft4_base_contract.total_len_is_absolute and
  (.search_lsm.api_limitations | length == 0) and
  .graph.parity_exact and
  .graph.cache_modes.explicit_zero_and_sized_available and
  ([.graph.forward.operations[], .graph.inverse.operations[]]
    | all(.sized_cache_warm.fetched_bytes == 0))
' /tmp/namidb-object-native-v2-smoke.json

# CI runs this corpus and contract in the `object-native acceptance` job and
# retains the JSON artifact even on failure.

# Reproducible 1M calibration. Its optional latency/byte/metadata/RSS thresholds
# intentionally remain absent (`null` in the report) until this exact command
# is run on the release machine. `--range-latency-ms 5` models non-zero request
# cost without making byte/request accounting dependent on a live R2 service.
CARGO_BUILD_JOBS=1 cargo run --release -p namidb-bench \
  --features object-native -- object-native \
  --vectors 1000000 --documents 1000000 --dim 256 --queries 50 --k 10 \
  --clusters 256 --spread 0.20 --filter-buckets 64 \
  --page-rows 512 --branch-factor 8 --nprobe 8 --max-nprobe 128 \
  --rerank-factor 8 --cache-bytes 67108864 \
  --delta-segments 3 --graph-keys 100000 --graph-high-degree 100000 \
  --build-memory-bytes 268435456 --range-latency-ms 5 --seed 42 \
  --max-fanout 32 --max-in-flight 16 \
  > /tmp/namidb-object-native-1m.json

# After calibration, a release owner may rerun with measured, reviewed SLOs.
# No numeric SLO is guessed by this repository:
#   --min-recall-at-k "$MIN_RECALL" \
#   --max-cold-bytes-ratio "$MAX_COLD_RATIO" \
#   --max-reader-metadata-ratio "$MAX_METADATA_RATIO" \
#   --max-query-p95-ms "$MAX_QUERY_P95_MS" \
#   --max-rss-bytes "$MAX_RSS_BYTES"

# 10M capacity run (documented only; it has not been executed for this V2
# freeze and is intentionally not a normal CI job). Peak
# builder-owned logical memory remains capped at 512 MiB; scratch space must
# hold the external sort/partition runs and one final artifact. Exact recall
# scans are CPU-heavy but remain O(k) in retained corpus state.
CARGO_BUILD_JOBS=1 cargo run --release -p namidb-bench \
  --features object-native -- object-native \
  --vectors 10000000 --documents 10000000 --dim 1024 --queries 100 --k 10 \
  --clusters 1024 --spread 0.20 --filter-buckets 128 \
  --page-rows 512 --branch-factor 8 --nprobe 16 --max-nprobe 256 \
  --rerank-factor 8 --cache-bytes 134217728 \
  --delta-segments 3 --graph-keys 1000000 --graph-high-degree 1000000 \
  --build-memory-bytes 536870912 --range-latency-ms 5 --seed 42 \
  --max-fanout 32 --max-in-flight 16 \
  > /tmp/namidb-object-native-10m.json
```

Do not treat the documented 1M/10M commands as results. The V2 freeze gate is
the deterministic 256-row smoke above; capacity runs are deliberately deferred
until the storage/compaction surfaces are frozen and a dedicated NVMe scratch
volume is available.

Use a local NVMe-backed `TMPDIR` (or the production spool environment selected
by the builders) for the 1M/10M runs. The harness intentionally does not
contact R2: its byte/request gates are deterministic and can be combined with a
separate live-R2 latency test without conflating network health with format
read amplification.

## Vector / ANN benchmarks

The vector tracks are separate from the LDBC graph bench above, and both run
**today with no external data**: the corpora are generated in-process from a
fixed `--seed`, and an exact brute-force KNN is the ground truth, so there is no
HDF5 / SIFT / GloVe download to stage. Two harnesses:

- `ann-bench` (needs the `vector-index` feature — it links the Vamana engine):
  the real `.vg` **index** recall@k vs the exact flat KNN, plus index-vs-scan
  latency, over the live storage engine.
- `vector-recall` (no feature, no engine): int8-vs-exact-f32 quantization recall
  arithmetic — the gate for the on-disk int8 vector format.

The full sampling protocol — fixed-seed generator, recall-vs-`ef` and
recall-vs-QPS curves, the int8 / filtered variants, the pessimistic floor, and
the HDF5 external-validation follow-up — lives in
`docs/rfc/031-ann-benchmark-methodology.md`.

### Implemented (v1.4 + this change set)

**`ann-bench` — real index recall + latency.** `ann_bench.rs::run` builds a
namespace on an `InMemory` store, registers a **cosine** `VectorIndexDescriptor`
(`r=32`, `l_build=64`, `alpha=1.2`; `quantization` is hardcoded to
`VectorQuantization::None`), writes the corpus across two L0 SSTs, `compact_l0`s
so the compactor materialises the `.vg`, then for each query calls the low-level
`Snapshot::vector_search` reader and scores it against `exact_top_k`
brute-force cosine as the ground truth. The JSON report carries `recall_at_k`
(`|index ∩ flat| / k`), `index_p50_us` / `index_p99_us`, `flat_p50_us` /
`flat_p99_us`, `speedup_p50` (`flat_p50 / index_p50`), `build_secs`, and
`cypher_index_path_reachable` — whether the optimizer rewrites a plain KNN Cypher
query onto the index (`cypher_reaches_index` inspects the optimized plan; it does
**not** execute it). Always run under `--release`: a debug build inflates both
the graph build and the per-query latency several-fold.

```bash
# Realistic clustered embeddings (true neighbours well separated). Produces:
# recall_at_k, index/flat p50+p99 latency, speedup_p50, build_secs,
# cypher_index_path_reachable. The Vamana build is single-threaded, so at
# --num 50000 the build (not the search) dominates the wall time.
cargo run --release -p namidb-bench --features vector-index -- ann-bench \
    --dim 256 --num 50000 --queries 200 --k 10 --clusters 256 --ef 64

# Pessimistic floor (uniform on the sphere, no meaningful neighbours): recall
# collapses by construction — this is the lower bound, not a target.
cargo run --release -p namidb-bench --features vector-index -- ann-bench \
    --dim 256 --num 50000 --queries 200 --k 10 --clusters 0

# Fast smoke (smaller corpus) with a verified representative result:
cargo run --release -p namidb-bench --features vector-index -- ann-bench \
    --dim 256 --num 5000 --queries 50 --k 10 --clusters 64 --ef 64
#   -> recall@10 ≈ 0.68, cypher_index_path_reachable=true,
#      index p50 ≈ 4.4 ms, flat p50 ≈ 75 ms, speedup ≈ 17x, build ≈ 42 s.
```

Recall is **workload- and `ef`-sensitive**: the same graph code that the
`namidb-ann` `recall_on_clustered_data_f32` unit test pins at `>= 0.90` (ef=64)
measures ≈0.68 on the harder corpus above. Always report the full parameter set
(`dim` / `num` / `clusters` / `spread` / `ef` / `seed`) with every number — a
lone recall figure is not portable across workloads.

**`ef` sweep (recall-vs-`ef` curve), runnable today.** `--ef` takes a single
value, so the curve is a shell loop. Each iteration rebuilds the `.vg` from
scratch — the build is single-threaded and dominates (≈42 s at 5k vectors) — so
this is honest but slow; the one-build sweep is the `--ef-list` proposal below.

```bash
for ef in 16 32 64 128 256; do
  cargo run --release -p namidb-bench --features vector-index -- ann-bench \
      --dim 256 --num 5000 --queries 50 --k 10 --clusters 64 --ef "$ef"
done
# Each line is one (ef, recall_at_k, index_p50_us, speedup_p50) point: recall
# should be non-decreasing in ef, latency rising with it.
```

**`vector-recall` — int8 quantization arithmetic.** No engine: it quantizes each
synthetic unit-norm vector with the per-vector max-abs scale the engine ships
(`quantize_i8`) and scores with the asymmetric f32×int8 scorer
(`dot_i8_asymmetric`), then reports `recall_at_k`, `recall_at_k_fixed_scale`
(a naive fixed-127 scale, for contrast), exact/int8 p50+p99, and the size change:
f32 costs `4 * dim` bytes/vector, int8 costs `dim + 4` (the codes plus one f32
scale), so `compression_ratio = 4*dim / (dim + 4)`.

```bash
# int8 arithmetic recall + compression. Produces: recall_at_k (per-vector
# scale), recall_at_k_fixed_scale, exact/int8 p50+p99, compression_ratio.
cargo run --release -p namidb-bench -- vector-recall \
    --dim 256 --num 5000 --queries 50 --k 10 --clusters 64
#   -> recall@10 ≈ 0.988 (fixed-scale ≈ 0.936), compression ≈ 3.94x
#      (1024 B -> 260 B/vector), exact p50 ≈ 1.2 ms, int8 p50 ≈ 1.2 ms.

# At OpenAI's 1536-dim the codes dominate the 4-byte scale, so the ratio rises
# toward ~3.99x:
cargo run --release -p namidb-bench -- vector-recall --dim 1536 --num 10000 --k 10
```

**QPS vs latency.** **QPS** (queries per second) = `num_queries / wall-clock
seconds spent in the search loop`. Both harnesses today time queries
**serially** — one at a time on one thread — and report only p50/p99 latency, so
`1 / p50` is a single-thread **latency floor**, not throughput under concurrent
load. There is no QPS axis or concurrency in the harness yet (see `--concurrency`
below).

### Proposed / future (tracked in RFC-031)

Not yet implemented; `docs/rfc/031-ann-benchmark-methodology.md` specifies the
harness additions:

- `--ef-list 16,32,64,128,256` — build the `.vg` **once** and sweep `ef` over the
  same graph, emitting a JSON array of reports (kills the repeated ≈42 s build
  the shell loop above pays per `ef`).
- `--concurrency N` — drive N tasks over one shared `Arc<Snapshot>` and report a
  real `qps` + `mean_us`, turning the serial latency floor into a
  throughput-under-load number (pin and report core count — a multi-core QPS is
  not a single-core deployment's QPS).
- `--quantization {none,int8}` — wire the choice into the `VectorIndexDescriptor`
  (valid because the bench already uses cosine, which int8 requires) to measure
  int8 recall / latency / size on the **real `.vg`**, not just `vector-recall`'s
  arithmetic.
- `--filter-keep <frac>` — route a filtered KNN through the executor
  (`try_index_search` in `crates/namidb-query/src/exec/walker.rs`) instead of the
  raw `Snapshot::vector_search` reader, so the oversample / post-filter path is
  exercised at a few selectivities.
- External-dataset validation against the ann-benchmarks HDF5 corpora
  (`sift-128-euclidean`, `glove-100-angular`, `gist-960`) as the publishable
  parity follow-up; the synthetic generator stays the fast CI / iteration path.

## What this bench does **not** cover yet

- The remaining 10 LDBC Complex Read queries (IC01/IC03/.../IC14).
  Several need features the parser and lowering don't have yet
  (recursive variable-length paths, `STDEV`, multi-pattern `WITH`
  threading). Each one is tracked in
  `crates/namidb-query/tests/parser_ldbc_snb_interactive.rs`.
- LDBC Short Reads (IS1-IS7). Trivial once the Complex set works.
- LDBC SNB **Updates** (IU1-IU8). Already covered in
  `crates/namidb-query/tests/exec_ldbc_snb_updates.rs`, just not benched
  here, because Kuzu's update semantics differ enough that an
  apples-to-apples comparison is harder.
- LDBC SF1/SF10 *real* datasets from the official Hadoop datagen. The
  synthetic data stays in-process for fast iteration; real LDBC is the
  paired follow-up.
