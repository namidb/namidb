# Changelog

All notable changes to NamiDB will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project loosely follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

While the engine is pre-1.0, breaking changes can land in minor
versions. They will always be called out in the **Breaking** section
below and in the release notes.

## [Unreleased]

### Added
- (nothing yet)

### Changed
- (nothing yet)

### Fixed
- (nothing yet)

### Breaking
- (nothing yet)

---

## [0.1.0] — initial public release

First public release of the NamiDB engine under
[Business Source License 1.1](LICENSE) (Change Date: 2029-05-18,
Change License: Apache License 2.0).

### Engine

- Cypher / GQL parser covering a strict subset of GQL (ISO/IEC
  39075:2024) + openCypher 9. End-to-end execution of LDBC SNB
  Interactive Complex Read queries IC01–IC12.
- Writes via Cypher: `CREATE`, `MERGE`, `SET`, `DELETE`, `DETACH
  DELETE`, `REMOVE`. Durable on `commit_batch` (WAL append + manifest
  CAS).
- Cost-based optimizer with predicate pushdown, projection pushdown,
  join reorder, hash-join conversion, hash semi-join (`EXISTS`
  decorrelation), and Parquet row-group pruning.
- Morsel-driven vectorized executor with optional factorized
  intermediate representation (RFC-017) for path-heavy queries.

### Storage

- Columnar storage on object storage: Parquet node SSTs, custom
  edge-SST format with CSR adjacency (RFC-002), zstd compression,
  bloom filters, fence-pointer indices.
- Coordination-free correctness: single-writer-per-namespace with
  epoch fencing via manifest CAS. Conditional writes (`If-Match`,
  `If-None-Match`) replace external consensus.
- Tiered caches: process-wide `AdjacencyCache` (CSR), `NodeViewCache`,
  and `SstCache` (decoded body + edge property streams + reader).
  Cross-snapshot reuse with `Arc`-shared, byte-budgeted memory.

### Clients

- Python bindings (`pip install namidb`), abi3 wheels for Linux
  (x86_64 + aarch64), macOS (arm64) and Windows (x86_64). Intel macOS
  installs via sdist. Sync + async (`acypher`). Arrow / pandas /
  polars output. `s3://` and `memory://` URIs.
- CLI: `namidb parse`, `namidb explain --verbose`, `namidb run`.

### Project

- Workspace of 8 crates (`namidb-core`, `-storage`, `-graph`,
  `-query`, `-cli`, `-py`, `-bench`, façade `namidb`).
- 18 design RFCs in [`docs/rfc/`](./docs/rfc/) covering storage
  engine, SST format, read path, Cypher subset, logical plan IR,
  write clauses, cost-based optimizer, predicate pushdown, hash join,
  Parquet predicate pushdown, hash semi-join, projection pushdown,
  join reorder, factorization, CSR adjacency, NodeView cache, and
  edge SST caches.
- LDBC-shaped synthetic benchmark harness with a paired Kùzu runner
  under [`bench/`](./bench/).

[Unreleased]: https://github.com/namidb/namidb/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/namidb/namidb/releases/tag/v0.1.0
