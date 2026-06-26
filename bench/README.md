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
