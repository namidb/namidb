# NamiDB LDBC-shaped bench

Synthetic, deterministic benchmark for the four LDBC SNB Complex Read
queries that NamiDB already supports end-to-end today:

| Query | Shape | Notes |
|---|---|---|
| **IC02** | `friend ← KNOWS ← p; msg → HAS_CREATOR → friend` | recent messages by friends |
| **IC07** | `p ← HAS_CREATOR ← msg ← LIKES ← fan` | recent likers of my messages |
| **IC08** | `p ← HAS_CREATOR ← post ← REPLY_OF ← reply` | recent replies to my messages |
| **IC09** | `p → KNOWS → friend → KNOWS → fof ← HAS_CREATOR ← msg` | recent messages by friends-of-friends |

Same dataset, same Cypher (modulo each backend's parameter syntax),
same warm-up + sample-count protocol. Output JSON is shape-compatible
so the two backends can be diffed directly.

## Layout

- `crates/namidb-bench/` — Rust crate. Subcommands:
  - `generate` — write CSV files for the dataset.
  - `run` — load (or reuse) the dataset into an in-memory NamiDB
    namespace, time each query, print JSON.
- `bench/kuzu_runner.py` — Python harness over the same CSVs against
  Kuzu (via the `kuzu` PyPI package).

## End-to-end workflow

```bash
# 0. Prerequisites.
rustup show          # cargo + rustc stable
python3 -m pip install kuzu

# 1. Generate the synthetic dataset once. Pick scale to fit RAM:
#    scale=1.0 → 10k Person + 100k Post + 50k Comment + ~430k edges.
#    scale=0.1 → ~1k Person + ~10k Post + ...      (fast smoke).
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
| `KNOWS` (Person→Person) | 100 000 | since |
| `HAS_CREATOR` (Post/Comment→Person) | 150 000 | — |
| `LIKES` (Person→Post/Comment) | 100 000 | creationDate |
| `REPLY_OF` (Comment→Post/Comment) | 30 000 | — |

Generator is `crates/namidb-bench/src/dataset.rs`. RNG is
`ChaCha8Rng(seed=42)` by default so two runs at the same scale
produce identical files (and Kuzu sees the same edges as NamiDB).

Node ids are 32-hex-char strings (16-byte) with a prefix byte tagging
the label (`P=Person, O=Post, C=Comment`), so the same numeric index
maps to distinct ids across labels.

## What this bench does **not** cover yet

- The remaining 10 LDBC Complex Read queries (IC01/IC03/.../IC14).
  Several require features NamiDB's parser/lowering doesn't have
  (recursive variable-length paths, `STDEV`, multi-pattern `WITH`
  threading). Track each in `crates/namidb-query/tests/parser_ldbc_snb_interactive.rs`.
- LDBC Short Reads (IS1-IS7). Trivial when the Complex set works.
- LDBC SNB **Updates** (IU1-IU8). Already covered in
  `crates/namidb-query/tests/exec_ldbc_snb_updates.rs`; not
  benched here because Kuzu's update semantics differ enough that
  apples-to-apples is harder.
- LDBC SF1/SF10 *real* datasets from the official Hadoop datagen.
  Synthetic stays in-process for fast iteration; real LDBC is the
  paired follow-up.
