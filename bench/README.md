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

The vector tracks are separate from the LDBC graph bench above. `ann-bench`
needs the `vector-index` feature (it links the Vamana engine):

```bash
# Realistic clustered embeddings (true neighbours well separated):
cargo run --release -p namidb-bench --features vector-index -- ann-bench \
    --dim 256 --num 50000 --queries 200 --k 10 --clusters 256 --ef 64

# Pessimistic floor (uniform on the sphere, no meaningful neighbours):
cargo run --release -p namidb-bench --features vector-index -- ann-bench \
    --dim 256 --num 50000 --queries 200 --k 10 --clusters 0
```

It builds a real namespace, registers a cosine index, writes the corpus across
two L0 SSTs, `compact_l0`s to materialise the `.vg`, then reports JSON with
`recall_at_k` (indexed top-k vs the exact flat top-k), `index_p50_us` /
`flat_p50_us` / `speedup_p50`, and `cypher_index_path_reachable` (whether a plain
KNN Cypher query is rewritten onto the index). Run under `--release`: a debug
build inflates both the graph build and the per-query latency several-fold.

`vector-recall` is the lower-level companion — int8 quantization recall vs exact
f32, no engine — used to justify the on-disk vector format:

```bash
cargo run --release -p namidb-bench -- vector-recall --dim 1536 --num 10000 --k 10
```

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
