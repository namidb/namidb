# Changelog

All notable changes to NamiDB will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project loosely follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

While the engine is pre-1.0, breaking changes can land in minor
versions. They will always be called out in the **Breaking** section
below and in the release notes.

## [Unreleased]

### Added

- **`Value::Date(i32)` and `Value::DateTime(i64)`** in `namidb-core`,
  with custom serde that tags them as `{"$date": <days>}` and
  `{"$datetime": <us>}` on JSON so the typing survives a round-trip
  through `__overflow_json` (undeclared properties). Declared
  columns of type `Date32` and `TimestampMicrosUtc` now decode to
  these variants instead of the previous lossy `Value::I64`, and
  the executor's `runtime_to_core` + `node_runtime_props_to_core`
  pass them through. The flush-side `PropertyBuilder` learns the
  two new match arms. Closes the limit found while smoke-testing
  Bolt: `datetime()` parameters from a Neo4j driver now persist and
  read back as `neo4j.time.DateTime` / `neo4j.time.Date` instead of
  raw integers.
- **Bolt protocol listener** in `namidb-server`. Opt-in via
  `--bolt-listen 0.0.0.0:7687` (or `NAMIDB_BOLT_LISTEN`). Speaks Bolt
  4.4 / 5.0 / 5.4 so the official Neo4j drivers (Python, Java,
  JavaScript, .NET, Go, Rust) connect unmodified through
  `bolt://host:7687`. The HTTP and Bolt listeners share one
  `WriterSession` per process and the same `--auth-token`. Design in
  [RFC-022](docs/rfc/022-bolt-protocol.md); see
  `crates/namidb-bolt` for the codec, handshake and state machine
  and `crates/namidb-server/src/bolt.rs` for the wiring.
- **`namidb-bolt` crate.** PackStream encoder/decoder, chunked
  framing, handshake (`0x6060B017` magic + four 4-byte version
  offers, with the `range` form supported), full request /
  response message vocabulary (HELLO / LOGON / LOGOFF / RUN / PULL /
  DISCARD / BEGIN / COMMIT / ROLLBACK / RESET / ROUTE / TELEMETRY /
  GOODBYE), a `Session` driver around a `Backend` trait, and a
  total `RuntimeValue` ↔ Bolt `Value` mapping including Node /
  Relationship / UnboundRelationship / Path / Date / LocalDateTime.
  Covered by 43 unit tests (including proptest round-trips) plus a
  two-test integration suite in
  `crates/namidb-server/tests/bolt_integration.rs` that drives a
  real `namidb-server` instance through the Bolt 5.4 handshake,
  authenticates, and round-trips CREATE / MATCH.
- **`tests/bolt_neo4j_driver_smoke.py`** — manual smoke script that
  connects the official `neo4j` PyPI driver to a running
  `namidb-server` and verifies a CREATE / MATCH round-trip end to end.

### Changed
- `namidb_server::Config` gained `bolt_listen: Option<SocketAddr>`.
  When unset the server stays HTTP-only (the previous behaviour).

### Fixed
- (nothing yet)

### Breaking
- (none) — Bolt is opt-in. Existing `Config` construction sites need
  to add `bolt_listen: None` for source compatibility.

---

## [0.4.1] - 2026-05-19: vector() + reproducible Docker build

Small follow-up to 0.4.0 driven by an end-to-end run against the
published Docker image: one packaging fix that was blocking a clean
`docker build`, and one Cypher surface that was blocking the only
test in the E2E battery that did not pass on 0.4.0 (vector
properties).

### Added
- **`vector()` Cypher builtin.** Lifts a numeric list literal or
  parameter into a first-class `Vector(Vec<f32>)`, the only shape that
  round-trips through `runtime_to_core` into `CoreValue::Vec` and the
  Parquet column writer (`crates/namidb-query/src/exec/expr.rs`,
  `crates/namidb-query/src/exec/writer.rs`). Accepts homogeneous
  `[Integer | Float]` lists (ints are coerced to `f32`) and propagates
  `NULL`. Non-numeric or non-list arguments produce a typed
  `EvalError` that names the offending element index. Bare list
  literals (e.g. `[0.1, 0.2]`) still error with `only scalars are
  storable in v0`. The constructor is the explicit opt-in. Engine
  vector capability has existed since v0.3 but lacked a Cypher entry
  point; the missing surface was flagged by an E2E run against the
  Docker image.

### Fixed
- **Track `Cargo.lock` in the repository.** The workspace ships
  distributable binaries (`namidb-server`, `namidb-cli`); the lockfile
  is required by `crates/namidb-server/Dockerfile` (its `COPY
  Cargo.toml Cargo.lock` line) and by anyone wanting reproducible
  release builds. Previously `.gitignore` excluded `Cargo.lock`, so
  the documented `docker build` recipe failed on a fresh clone unless
  the user ran `cargo generate-lockfile` first.

---

## [0.4.0] - 2026-05-19: engine perf sweep

Performance gains over 0.3.0 (LDBC SNB SF1, M-series laptop, 30 warm
runs x 3 params; reproducible from `scripts/bench_publish/`):

- Cold IC09 SF1: 9.0 s to 170 ms (52x), from `batch_lookup_nodes` +
  decoded RecordBatch cache + persisted unique-property sidecar +
  skip intermediate target materialise in chained Expand.
- Cold IC02 SF1: 720 ms to 51 ms (17x), from the sidecar property
  index + decoded batches cache.
- Engine warm vs Kùzu: NamiDB now beats Kùzu warm on every IC02 / 07
  / 08 / 09 (3-4x on IC02 and IC08).
- Bulk-write to R2: 5.5 K to 31.9 K elem/s (laptop, 5.5x) and 51.5 K
  elem/s in-region (9x) via 5 MiB multipart upload at 8-way
  concurrency.

Workspace tests: ~700 passing across storage / query / server /
bench / control / gateway / worker / CLI crates.

### Added

- **`Snapshot::batch_lookup_nodes(label, &[NodeId])`** materialises
  many node views in one pass over the candidate SST set. Last-LSN
  merge across memtable + SSTs preserves consistency; `NodeViewCache`
  and `SstCache` populate on the way out
  (`crates/namidb-storage/src/read.rs`,
  `crates/namidb-query/src/exec/walker.rs`).
- **Persisted unique-property index sidecar.**
  `SstDescriptor.unique_property_indices` + a bincode sidecar
  alongside every Node SST. `lookup_node_by_property` resolves the
  point query with one bincode decode per candidate SST instead of
  scanning the full label. Re-emitted on L0 to L1 compaction so the
  fast path survives the merge (`crates/namidb-storage/src/flush.rs`,
  `compact.rs`, `manifest.rs`, `read.rs`,
  `crates/namidb-query/src/cost/stats.rs`).
- **`PropertyDef::unique: bool` schema flag + planner rewrite.**
  `Filter(NodeScan {label})` with an equality on a unique property is
  rewritten to `NodeByPropertyValue` for SST-level pushdown. New
  optimizer pass `crates/namidb-query/src/optimize/unique_lookup.rs`;
  schema in `crates/namidb-core/src/schema.rs`.
- **In-memory property index on the write session.** Closes the
  warm-path gap on repeated unique-property lookups before flush
  (new file `crates/namidb-storage/src/property_index.rs`,
  `ingest.rs`, `lib.rs`, `read.rs`).
- **Intra-snapshot decoded RecordBatch cache** keyed by SST path.
  `decoded_node_sst_batches: Mutex<HashMap<path, Arc<Vec<RecordBatch>>>>`
  amortises the per-call Parquet decode across N `batch_lookup_nodes`
  invocations inside a single query (`crates/namidb-storage/src/read.rs`).
- **Multipart PUT for SST bodies >= 4 MiB on flush.**
  `flush::put_object` switches to `object_store::buffered::BufWriter`
  (5 MiB parts, 8 in flight). Small bodies keep the single-PUT +
  `PutMode::Create` collision protection
  (`crates/namidb-storage/src/flush.rs`).
- **`namidb-bench load`.** Write-throughput timing for Bench D
  (`crates/namidb-bench/src/main.rs`).

### Changed

- **Chained `Expand` skips intermediate target materialise** when the
  target alias is only consumed as the next `Expand`'s source.
  `walker::PlanRouting` extended with a target-alias-references-out
  check (`crates/namidb-query/src/exec/walker.rs`,
  `crates/namidb-query/src/cost/cardinality.rs`,
  `crates/namidb-query/src/cost/selectivity.rs`,
  `crates/namidb-query/src/optimize/join_conversion.rs`,
  `crates/namidb-query/src/plan/explain.rs`).

### Fixed

- The bench loader declares `id` as a user property so the LDBC
  IC02 / 07 / 08 / 09 fixtures bind rows correctly under the v0.3.0
  `id` to `_id` semantics (`crates/namidb-bench/src/loader.rs`).

### Breaking

- (none)

---

## [0.3.0] - 2026-05-18: Cypher v0.2.1 limitation sweep

Closes the six query-engine limitations documented in the v0.2.1
README (`MATCH (n)` rejected, MERGE with relationship broken, `id`
reserved, etc.). One of them, the `id` reservation, is breaking; see
**Breaking** below.

### Fixed

- **#5** `lower::combine` now emits `CrossProduct` between two
  non-Empty plans instead of dropping the earlier one, so
  `MATCH (a:A) MATCH (b:B) CREATE (a)-[:R]->(b)` finally propagates
  both bindings to `CREATE` (`crates/namidb-query/src/plan/lower.rs`).
- **#2** `find_merge_matches` indexes the `Vec<CreateElement>` by
  alias instead of positionally, so `MERGE (a)-[r:R]->(b)` works
  against the CREATE-shaped pattern the lowerer produces
  (`crates/namidb-query/src/exec/writer.rs`).
- **#4 / #6** `execute_expand` (and its factor sibling) accept
  `edge_type=None` and fan out across every type observable through
  the snapshot, so `MATCH (a)-[r]->(b)` and `-[*1..N]->` work without
  an explicit relationship type. Backed by a new
  `Snapshot::observed_edge_types` that unions declared schema +
  memtable + persisted SSTs, needed because the declared schema is
  empty for namespaces that never went through `SchemaBuilder`
  (`crates/namidb-storage/src/read.rs`,
  `crates/namidb-query/src/exec/walker.rs`).
- **#3** `LogicalPlan::NodeScan.label` becomes `Option<String>`;
  walker resolves the set via `Snapshot::observed_labels` so
  `MATCH (n)` without a label predicate fans out across every label.
  Cardinality falls back to `catalog.total_nodes()`; `EXPLAIN`
  renders `label=*`. The id-lookup branch (`{_id: $x}`, see Breaking)
  still requires an explicit label because `NodeById` needs a
  specific column family (`crates/namidb-query/src/plan/logical.rs`,
  `crates/namidb-query/src/plan/lower.rs`,
  `crates/namidb-query/src/exec/walker.rs`, and cascade).

### Breaking

- **#1 `id` is now a user property; the internal NodeId moves to
  `_id`.** Previously `id` hijacked Cypher map literals as the
  internal NodeId sigil: a `CREATE (n:Foo {id: $uuid})` parsed
  `$uuid` as a `NodeId` and refused to persist `id` as a property.
  After this release, `id` is treated like any other property; the
  internal NodeId is addressed via `_id`. The Cypher `id(n)`
  function keeps returning the internal NodeId for callers that want
  it.

  **Migration.** Anywhere a query passes `{id: $uuid}` to refer to
  the internal NodeId, rename the key to `{_id: $uuid}`. Likewise
  `n.id` (accessor) becomes `n._id` when you want the NodeId, or
  `id(n)` for the function form. Reading `n.id` now returns the user
  property (or `Null` when absent). Failures are loud rather than
  silent: a wrong UUID lands as a plain `Filter` over a missing
  property and returns no rows rather than throwing.

  Behavioural pivots:
  - `CREATE (n:Foo {_id: $uuid, id: 'external-42'})` assigns the
    storage NodeId from `_id` and persists `id` in the property map.
  - `MATCH (n:Foo {_id: $uuid})` lowers to `NodeById`; `{id: ...}`
    falls through to `NodeScan + Filter`.
  - `n._id` and `id(n)` materialise the internal NodeId; `n.id`
    reads the user-owned property (or `Null`).

  Sites updated alongside the engine change: every LDBC fixture in
  `crates/namidb-query/tests/fixtures/`, the optimizer's
  decorrelation join-key
  (`crates/namidb-query/src/optimize/decorrelation.rs`), and the
  integration tests in `exec_writes`, `exec_match_expand`,
  `cost_smoke`, `exec_ldbc_snb`.

---

## [0.2.1] - 2026-05-18: CI fix

Tag `py-v0.2.0` built every wheel and the sdist, but the smoke-test
job (`pytest` against the installed wheel) flagged three stale
expectations and the publish step was skipped, so nothing reached
PyPI. `0.2.1` ships the same code with the test expectations brought
up to date.

### Fixed

- `crates/namidb-py/tests/test_uri.py`: three tests were asserting
  the *pre-0.2.0* contract (`file://`, `gs://`, `az://` raise
  `ValueError`). Replaced with:
  - `test_file_uri_round_trip`: full CREATE / MATCH against a
    temp-dir-backed namespace, exercising the new
    `LocalFileObjectStore` end-to-end from Python.
  - `test_gs_uri_missing_namespace_raises`,
    `test_az_uri_missing_container_raises`,
    `test_az_uri_missing_namespace_raises`: grammar checks that
    surface before the GCS / Azure client is built, so they don't
    need real cloud credentials on CI runners.

---

## [0.2.0] - 2026-05-18: self-host story

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
- **`namidb-server` crate and binary.** Rust HTTP daemon exposing a
  REST API over any backend. Endpoints: `POST /v0/cypher`,
  `GET /v0/health`, `GET /v0/version`, `POST /v0/admin/flush`. Bearer
  token auth (`--auth-token`), periodic memtable flush
  (`--flush-interval`), multi-stage Dockerfile, full two-way
  JSON/Cypher type mapping for Node / Rel / Path values.
- **`docker-compose.yml`** at the repo root: a copy-paste recipe that
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

- **CLI `namidb run` learns `--store <uri>`.** Accepts any supported
  scheme (`memory://`, `file://`, `s3://`, `gs://`, `az://`) for
  durable runs. Defaults to `memory://default` when omitted, preserving
  the previous one-shot ephemeral UX.
- **Python `namidb.Client(uri)`** now delegates URI parsing to the shared
  Rust implementation. `PyValueError` is raised on malformed URIs and
  `PyRuntimeError` on backend-init failures; messages unchanged.
- **README** reorganised into an S3-first self-host guide: the hero
  line ("Your graph database lives in your S3 bucket"), a "The shape"
  paragraph, AWS S3 / Cloudflare R2 as starred backends, MinIO and the
  others tucked into collapsible sections, and a new Roadmap section.
- **`clap`** workspace feature set now includes `env` so server flags
  can be supplied via `NAMIDB_*` env vars.

### Fixed

- `plan::explain::tests::explain_renders_full_chain` indent
  expectation aligned with the tree-renderer's per-depth indentation.

### Breaking

- (none). Every previously-rejected scheme now returns a working
  client instead of a `ValueError`; all existing `memory://` and
  `s3://` URIs continue to work unchanged.

---

## [0.1.0] - initial public release

First public release of the NamiDB engine under
[Business Source License 1.1](LICENSE) (Change Date: 2029-05-18,
Change License: Apache License 2.0).

### Engine

- Cypher / GQL parser covering a strict subset of GQL (ISO/IEC
  39075:2024) + openCypher 9. End-to-end execution of LDBC SNB
  Interactive Complex Read queries IC01-IC12.
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

[Unreleased]: https://github.com/namidb/namidb/compare/v0.4.1...HEAD
[0.4.1]: https://github.com/namidb/namidb/releases/tag/v0.4.1
[0.4.0]: https://github.com/namidb/namidb/releases/tag/v0.4.0
[0.3.0]: https://github.com/namidb/namidb/releases/tag/v0.3.0
[0.2.1]: https://github.com/namidb/namidb/releases/tag/v0.2.1
[0.2.0]: https://github.com/namidb/namidb/releases/tag/v0.2.0
[0.1.0]: https://github.com/namidb/namidb/releases/tag/v0.1.0
