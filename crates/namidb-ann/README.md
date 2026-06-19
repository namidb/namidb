# namidb-ann

DiskANN/Vamana approximate-nearest-neighbor (ANN) graph index for NamiDB vector
search.

This crate holds the **algorithm** — pure, in-memory, storage-agnostic. The
[`VectorSpace`] trait abstracts the stored representation and its distance
function, so the same Vamana build + greedy search runs over:

- f32 unit-normalized vectors (the recall-golden path, `F32CosineSpace`), and
- int8-quantized vectors (the shipped path, `Int8Space`, scoring the query in
  f32 against per-vector-scaled int8 codes via `namidb_core::quantize`).

## What it provides

- [`VamanaGraph`] — a bounded-degree search graph (`Vec<Vec<u32>>` adjacency +
  entry medoid), produced by [`build::build`] with the Vamana robust-prune
  algorithm (`α` diversification, degree bound `R`, build beam `L_build`).
- [`search::search`] — best-first beam search with a candidate min-heap, a
  result max-heap capped at `ef_search`, and a visited bitset; returns the
  top-`k` ids with their distances.
- [`build::RobustPrune`] — the α-pruning diversification procedure, exposed so
  incremental/L0 builds can reuse it.

## Not in this crate

Object-store wiring (`SstKind::VectorGraph`), the compaction-time build hook,
the `CREATE VECTOR INDEX` DDL, and the optimizer `VectorSearch` rewrite live in
`namidb-storage` / `namidb-query` behind the `vector-index` Cargo feature. This
crate deliberately has no storage dependency so the algorithm is testable in
isolation and the recall harness can drive it directly.

## Recall

The greedy search is approximate; recall@k is bounded by `R`, `L_build`,
`α`, and `ef_search`. See the `recall_*` tests in `src/build.rs` for the
expected floor on clustered synthetic data.
