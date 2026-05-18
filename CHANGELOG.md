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

## [0.2.0] — 2026-05-18 · self-host story

### Added

- **`file://` storage backend** with full manifest CAS via per-namespace
  `flock` + atomic `rename(2)` (`namidb-storage::local::LocalFileObjectStore`).
  Previously rejected with a `ValueError`; now a first-class durable
  backend. Works in CI fixtures, single-machine deployments, and
  anywhere a real bucket is overkill.
- **`gs://` storage backend** for Google Cloud Storage. Credentials
  via `GOOGLE_APPLICATION_CREDENTIALS` or `?service_account=` query
  parameter. Previously rejected as "planned"; now stable.
- **`az://` storage backend** for Azure Blob Storage. Credentials via
  the standard `AZURE_STORAGE_*` env vars; supports the Azurite
  emulator via `?use_emulator=true`. Previously rejected as "planned";
  now stable.
- **`namidb-server` crate and binary** — Rust HTTP daemon exposing a
  REST API over any backend. Endpoints: `POST /v0/cypher`,
  `GET /v0/health`, `GET /v0/version`, `POST /v0/admin/flush`. Bearer
  token auth (`--auth-token`), periodic memtable flush
  (`--flush-interval`), multi-stage Dockerfile, full JSON ↔ Cypher
  type mapping for Node / Rel / Path values.
- **`docker-compose.yml`** at the repo root — copy-paste recipe that
  brings up MinIO + bucket-init + `namidb-server` and exposes an
  authenticated graph database on `:8080`.
- **Shared URI parser** (`namidb-storage::uri::parse_uri`) used by
  the Python client, the CLI, and the server.
- **Architecture and deployment diagrams** as native SVGs, with
  matching dark-mode variants (`*-dark.svg`) selected by GitHub
  automatically via `<picture media="(prefers-color-scheme: dark)">`.
  System-font stack only; the dark palette swaps the slate ink for
  a near-white on `#0f172a` ground and brightens the accent teal
  to `#5eb5c8` for legibility.

### Changed

- **CLI `namidb run` learns `--store <uri>`** — accepts any supported
  scheme (`memory://`, `file://`, `s3://`, `gs://`, `az://`) for
  durable runs. Defaults to `memory://default` when omitted, preserving
  the previous one-shot ephemeral UX.
- **Python `tg.Client(uri)`** now delegates URI parsing to the shared
  Rust implementation. `PyValueError` is raised on malformed URIs and
  `PyRuntimeError` on backend-init failures; messages unchanged.
- **README** reorganised into an S3-first self-host guide: hero hook
  ("Your graph database lives in your S3 bucket"), "The shape"
  paragraph, AWS S3 / Cloudflare R2 as starred backends, MinIO and the
  others tucked into collapsible sections, and a new Roadmap section.
- **`clap`** workspace feature set now includes `env` so server flags
  can be supplied via `NAMIDB_*` env vars.

### Fixed

- `plan::explain::tests::explain_renders_full_chain` indent
  expectation aligned with the tree-renderer's per-depth indentation.

### Breaking

- (none) — every previously-rejected scheme now returns a working
  client instead of a `ValueError`; all existing `memory://` and
  `s3://` URIs continue to work unchanged.

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

[Unreleased]: https://github.com/namidb/namidb/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/namidb/namidb/releases/tag/v0.2.0
[0.1.0]: https://github.com/namidb/namidb/releases/tag/v0.1.0
