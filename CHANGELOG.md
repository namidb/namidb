# Changelog

All notable changes to NamiDB will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project loosely follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

While the engine is pre-1.0, breaking changes can land in minor
versions. They will always be called out in the **Breaking** section
below and in the release notes.

## [Unreleased]

### Added

- **Markdown links as graph edges.** Standard markdown links `[text](note.md)`
  to a local `.md`/`.markdown` file now produce a `LINKS_TO` edge alongside
  `[[wikilinks]]`, resolved by basename (percent-decoded). External URLs,
  mail/other schemes, anchors and non-markdown files are ignored, and a
  destination that does not reduce to a clean note name is skipped rather than
  creating a dangling edge. Docs-style vaults (and the repo's own `MEMORY.md`
  index, which uses `[Title](file.md)`) become fully connected.
- **Inline `#tags` collected into the `tags` property.** Inline tags in a
  note body (excluding code, headings and URLs; nested `#area/topic` kept;
  `#123` is not a tag) are merged with any frontmatter `tags` into one
  deduplicated `tags` list. A frontmatter `tags` value that is not a string or
  list is left untouched, and non-string list items are preserved.

### Changed

### Fixed

### Breaking

---

## [0.8.0] - 2026-05-30: vault prune, name resolution, and prebuilt binaries

### Added

- **`prune` for vault loads.** Re-loading a vault can now mirror it instead of
  merging: with prune enabled the loader tombstones notes and links the vault
  no longer contains, so the graph stays a faithful, rebuildable index rather
  than accumulating stale nodes and edges. Exposed as `LoadOptions::prune`, the
  CLI `--prune` flag, and the Python `Client.load_vault(prune=...)` argument;
  the local MCP server mirrors on load. The default load stays additive.
- **Resolve notes by name in the MCP server.** The vault loader stores a
  normalized `key` property on each note, and the `backlinks`, `neighbors` and
  `get_note` tools now resolve their argument by that key as well as by exact
  title or path. An agent can address a note as `User Role`, `user-role` or
  `user_role` regardless of the file stem's casing or separators.
- **Prebuilt `namidb` and `namidb-mcp` binaries.** A `release-binaries`
  workflow builds standalone binaries for Linux (x86_64, aarch64), macOS
  (arm64) and Windows (x86_64) on every `v*` tag and attaches them to the
  GitHub Release, so the CLI and MCP server run without a Rust toolchain.
- **Offline SST builder and `attach_ssts`.** Build SST files offline (outside a
  live `WriterSession`) and attach them to a namespace's manifest via
  `attach_ssts` (RFC-023 tasks 4/5).

### Changed

### Fixed

### Breaking

---

## [0.7.0] - 2026-05-30: markdown vault ingest + local MCP server

### Added

- **Markdown vault ingest (`namidb-markdown`).** Load an Obsidian-style
  vault of `.md` files into a graph: each note becomes a `Note` node,
  each `[[wikilink]]` a `LINKS_TO` edge, and YAML frontmatter becomes
  node properties. The raw note body is kept as a `body` property, so the
  files stay the source of truth and the graph is a derived, rebuildable
  index. Wikilinks resolve by normalized basename (kebab, snake, and
  spaces collapse to one key), links inside fenced or inline code are
  excluded, and node ids are derived with BLAKE3 so re-ingesting a vault
  is idempotent.
- **`namidb load-vault` (CLI).** Load a vault into any namespace, with
  `--store`, `--namespace`, `--label`, and `--edge-type`.
- **`Client.load_vault` (Python).** Load a vault from the Python client;
  it commits the load and returns a dict of counts.
- **Local MCP server (`namidb-mcp`).** A Model Context Protocol server
  (JSON-RPC 2.0 over stdio) that exposes a namespace to an agent as
  read-only graph tools: `list_notes`, `get_note`, `backlinks`,
  `neighbors`, `orphans`, `search`, and a read-only `cypher`. Point it at
  a loaded namespace or pass `--vault` to load one on startup.

### Changed

### Fixed

### Breaking

---

## [0.6.0] - 2026-05-28: edge-type-count pushdown + orphan-segment durability

### Added

- **Edge-type-count pushdown.** A global `count(*)` / `count(r)` over a
  directed, single-hop, unfiltered typed expand
  (`MATCH ()-[r:T]->() RETURN count(r)`) is now answered straight from the
  edge index via a new `EdgeTypeCount` operator, skipping the `NodeScan` +
  `Expand` over every node. The rewrite is conservative: a labelled or
  predicated source, a target label, an undirected, variable-length,
  optional, or `shortestPath` expand, an untyped edge, `GROUP BY`, or a
  count over anything but the relationship binding all fall back to the
  ordinary plan. `EXPLAIN` renders the operator.
- **`Snapshot::count_edge_type`.** Counts live edges of a type by merging
  the memtable and forward SSTs (last-writer-wins, tombstones pruned)
  without decoding edge property streams.

### Changed

- **The server caches the optimizer `StatsCatalog` per manifest version**
  instead of rebuilding it on every read query. Every commit bumps the
  version, so a version match is enough to keep the cache valid.

### Fixed

- **Intra-session orphan WAL recovery.** When a prior commit left a WAL
  segment durable but failed before the manifest commit, an in-session
  retry of `commit_batch` no longer wedges on a repeated `Precondition`.
  It re-picks a fresh seq and retries once, recovering when the manifest
  body slot is still free and otherwise terminating with a clean
  `ManifestCommitCas` that poisons the session for a drop-and-reopen.
- **`claim_writer` no longer hangs on an orphan manifest body.** A body
  written at `version + 1` whose pointer CAS failed transiently used to
  spin `claim_writer` forever. It now bounds the stall and returns the new
  `OrphanManifestBody` error.

### Breaking

- None.

---

## [0.5.1] - 2026-05-27: Value::Bytes JSON round-trip

### Fixed

- **`Value::Bytes` round-trips through `__overflow_json` again.** The
  serialiser wrote bytes as an untagged JSON array (`[0, 1, 2]`); the
  deserialiser's `visit_seq` could not tell that apart from a
  `Vec<f32>` vector and silently turned the blob into a float vector
  on the way back. The smoke test `test_property_types_roundtrip`
  caught the regression at release time. Fixed by tagging bytes as
  `{"$bytes": [0, 1, 2]}` (matching the `$date` / `$datetime` /
  `$list` / `$map` shapes already in this module). Old SST bodies
  that still encode bytes untagged keep decoding as `Vec<f32>` —
  forward compatible, no backfill required.

### Breaking

- (pre-1.0 semver-relax) Newly-written `Value::Bytes` use the tagged
  JSON wire shape. Downstream services that parsed the bytes-as-array
  shape directly must accept the tagged form going forward. SST
  bodies written before 0.5.1 keep working through the legacy
  untagged path.

---

## [0.5.0] - 2026-05-26: cloud-readiness sweep

### Added

- **`profile_query_tree` with per-operator runtime stats.** PROFILE
  now reports `rows_returned` and `elapsed_us` on every operator in
  the returned `ExplainNode`, not only the root. Plumbed through a
  `ProfileCollector` scoped on `tokio::task_local!` so a plain
  `execute` (no scope) keeps its baseline cost. Times are inclusive
  (parent includes children). Per-op `attribute_profiles` walks the
  plan and explain trees in lockstep, keying by stable `LogicalPlan`
  pointer.
- **`profile.rs` module** exposing `ProfileCollector`,
  `profile_query_tree`, `ProfileError`, plus `RuntimeStats` on
  `ExplainNode` (`Option<RuntimeStats>` field, `#[serde(skip)]` when
  absent so existing EXPLAIN JSON payloads stay byte-compatible).
- **Structured `ExplainNode` tree** alongside the existing string
  renderer (`explain_tree`, `explain_tree_verbose`,
  `explain_query_tree*`). The cloud worker / CLI consume the
  `Serialize` shape directly without depending on `serde_json` from
  this crate.
- **`pagination.rs`: offset cursors (`v1`).** `Cursor`, `CursorError`,
  `paginate_plan`, `next_cursor`. Wire shape `v1:<decimal-skip>`.
  Wraps the plan in a `TopN { skip, limit }` and is the
  zero-assumptions default the dashboard's paginated tables hit.
- **`pagination.rs`: keyset cursors (`v2`).** `CursorKeyset` with
  `encode` / `decode`, `paginate_plan_keyset`, `next_cursor_keyset`.
  Rewrites the plan into `WHERE alias._id > cursor.last_id ORDER BY
  alias._id ASC LIMIT page_size` so deep pages stay flat in cost.
  Plan-hash mismatch must reject the request — documented as caller
  contract.
- **`plan_cache.rs`: plan-cache helpers.** `query_text_hash` produces
  a stable xxh3-64 fingerprint of a Cypher query with whitespace
  normalised. `parse_lower_optimize(text, catalog)` is the one-shot
  entry point the cache wraps. Cache key layout the caller is
  expected to wire up: `format!("{ENGINE_VERSION}:{hash}")`.
- **`LogicalPlan` + AST: `Serialize` / `Deserialize` derives.**
  Every node (LogicalPlan, AggregateExpr, CreateElement, SetOp,
  RemoveOp, ShortestMode, OrderKey, Expression, Literal, MapLiteral,
  PatternProperties, NodePattern, RelationshipPattern, …) plus
  `SourceSpan` and storage `ScanPredicate` derive serde. Cross-process
  plan caches (Redis, R2, Supabase) can round-trip a cached plan
  byte-for-byte.
- **`Snapshot::observed_edge_endpoints`.** For declared edge types
  the endpoints come straight from `EdgeTypeDef`. For undeclared
  types we sample one upserted edge per type and resolve its
  endpoint labels via the memtable's `NodeId → label` map with a
  `lookup_node` fallback for SST-resident endpoints. Carries an
  `inferred` flag.
- **`Snapshot::observed_property_types_for_label`** + new
  `PropertyColumnStats::observed_data_type`. Merges the declared
  `LabelDef` with SST `PropertyColumnStats` so the schema response
  reports property types even when the namespace skipped
  `SchemaBuilder`.
- **`Value::List(Vec<Value>)` and `Value::Map(BTreeMap<String, Value>)`**
  in `namidb-core`. JSON-tagged as `{"$list": [...]}` and `{"$map":
  {...}}` so the typing survives a `__overflow_json` round-trip and
  bare JSON arrays keep decoding as `Vec<f32>`. The executor accepts
  list and map runtime values; declared columns stay scalar-only
  (separate RFC).
- **`CREATE (n:L $params)`: parameter-as-map property spread.** New
  `PatternProperties` enum on `NodePattern` / `RelationshipPattern`
  (`Literal` | `Parameter`). `CreateElement` grows a
  `properties_spread: Option<Expression>` the executor merges into
  the new node / edge at runtime. Explicit literal entries still
  win on key collisions. MATCH / MERGE patterns accept the syntax
  too but lower rejects them today with a clear pointer to the
  WHERE alternative.
- **`expect_in(token, ctx)` helper in the parser** + contextual
  `help:` line on six closing-token sites (node pattern,
  relationship pattern, map literal, function call arguments, list
  literal, `CASE` expression). `E001` payloads now say "while parsing
  node pattern" instead of the bare token name.
- **`Cursor`'s namespace got namesake structured `ExplainNode`
  variants** (`explain_query_raw_tree`, `explain_query_raw_tree_verbose`)
  so callers can render the pre-optimise plan in the same shape as
  the optimised one.
- **`MemtableSnapshotFile` cold-start fast path** +
  `WriterSession::write_memtable_snapshot_now()` and the
  `NAMIDB_MEMTABLE_SNAPSHOT_EVERY` env var. The writer auto-writes
  the bincode snapshot every N commits when the env var is set; a
  cold-starting writer always tries the snapshot path before WAL
  replay. Best-effort: failed snapshot writes log and continue.

### Changed

- **`commit_batch` pipelines the WAL append with the manifest body
  PUT.** `ManifestStore::commit` split into `put_body` +
  `cas_pointer`; `WriterSession::commit_batch` runs the WAL append
  and the manifest body PUT under `tokio::join!`, then `cas_pointer`
  once both are durable. Critical path drops from three round-trips
  to two (`max(WAL, body) + CAS`).
- **`scan_node_for_id` consults `observed_labels`** instead of the
  declared label map only. The typeless Expand path no longer
  silently drops every neighbour for namespaces that skipped
  `SchemaBuilder`.
- **MERGE `find_merge_matches` accepts back-references on both sides.**
  `MATCH (a), (b) MERGE (a)-[r:KNOWS]->(b)` now succeeds instead of
  erroring with "MERGE head `a` not found"; the rel binding (`r`) is
  populated on the resulting row too. The matcher classifies each
  pattern position as `Fresh` (scan + filter) vs `BackReference`
  (constrain by existing NodeId) and chooses accordingly.

### Fixed

- **`MATCH ()-[r:T]->()` and `MATCH (a)-[r]->(b)` return their edges.**
  `scan_node_for_id` walked the declared label map and dropped every
  neighbour when the namespace had no `SchemaBuilder`, returning 0
  rows. `observed_labels` covers the same surface as
  `resolve_edge_types` already does on the edge-type side.
- **MERGE pattern accepts back-referenced sources / tails.** See the
  *Changed* entry above; tracked as the same bug from two angles.
- **`CREATE (n:L $params)` works.** The parser accepted only literal
  `{...}` maps; the lowerer rejected anything else. New
  `PatternProperties` enum + `properties_spread` field through the
  pipeline.
- **Parse errors name the production.** "expected `)`, found `RETURN`"
  now ships a `help: while parsing node pattern` line; same for
  relationship pattern, map literal, function call arguments, list
  literal, and `CASE`.
- **Schema response carries edge endpoints / property types.** The
  cloud worker can answer `/schema` without falling back to a
  client-side sample. (Engine side; cloud handler picks this up
  separately.)
- **`Value::List` / `Value::Map` storable.** "only scalars are
  storable in v0" stops being a wall for tag-style lists and
  metadata maps.
- **`__overflow_json` round-trips list + map values.** The serde
  visitor learned `$list` / `$map` tags so a stored value comes back
  as the same `Value` variant it went in as.

### Breaking

- (pre-1.0 semver-relax) `Value` and `LogicalPlan` are wider:
  exhaustive matches downstream need new arms for the new variants
  (`Value::List`, `Value::Map`, `LogicalPlan::Merge`'s
  `properties_spread` field on `CreateElement::Node/Rel`).
- `NodePattern.properties` and `RelationshipPattern.properties`
  changed from `Option<MapLiteral>` to `Option<PatternProperties>`.
  External AST consumers must add the new enum arms.

### Earlier in this release window (previously in `[Unreleased]`)

The items below landed between 0.4.1 and the 0.5.0 tag and were
already in `main` ahead of this session; they ride along in 0.5.0.

#### Added

- **Worst-case optimal join via leapfrog triejoin (RFC-024).** Cyclic
  Cypher patterns that used to expand as a chain of binary `HashJoin`
  / `Expand` operators now fold into a single `LogicalPlan::
  MultiwayJoin` that runs Veldhuizen 2014 leapfrog over the sorted
  partner lists `Snapshot::sorted_partners` produces. The new path
  is opt-in via `NAMIDB_WCOJ=1` (and requires `NAMIDB_FACTORIZE=1`);
  when off, the planner stays on the existing binary chain so
  behaviour is unchanged for production. The detection pass at
  `optimize::multiway_join` walks the plan top-down, harvests a
  contiguous `Expand` chain rooted at a labelled `NodeScan`, runs
  union-find to spot a cycle, and emits the `MultiwayJoin`; chains
  that don't satisfy the v0 preconditions (variable-length,
  `rel_alias` set, undirected edges, missing target label, mid-chain
  `Filter` with user predicates) silently fall back to the binary
  plan. The executor binds variables in the heuristic ordering
  produced by `variable_ordering` (head NodeScan first, rest by
  constraint-graph degree), leapfrog-intersects the per-constraint
  partner lists at each level, and at the leaf scales the per-tuple
  WCOJ set to the per-path multiset binary emits via
  `count_edge_multiplicity` so `RETURN a, b, c` (no `DISTINCT`) gets
  the same row count from both paths.
- **Relationship type alternation `[:A|:B|...]` (RFC-024 §Q1).** The
  lowering at `lower.rs:877` no longer rejects alternation;
  `LogicalPlan::Expand.edge_type` and
  `EdgeConstraint.edge_types` now carry a non-empty
  `Vec<String>` of accepted types. The non-cyclic executor unions
  partner lists across the listed types through the existing
  per-type iteration in `neighbours_of_any`; the cyclic executor
  uses a new `MergeSortedUnion` primitive to fold per-type lists
  into one ascending stream before the outer leapfrog intersection.
  Singleton `[:KNOWS]` keeps working bit-identically;
  `[:KNOWS|:LIKES|:FOLLOWS]` now matches across all listed types.
  An exhaustive sweep
  (`exec_alternation::multiway_join_alternation_per_path_count_matches
  _binary_in_all_cases`) covers every single-type / mixed /
  all-both pair combination on a triangle and asserts WCOJ and
  binary row counts agree exactly.
- **AGM-tight cost model for `MultiwayJoin` (RFC-024 §Cost model).**
  `cost::cardinality::agm_bound_rows` returns the Atserias-Grohe-Marx
  upper bound for the cyclic match's output. For the shapes the v0
  detection pass actually produces (triangle, k-clique, k-cycle,
  triangle-with-dangling-edge, K_{m,n}) the greedy
  `w_e = 1 / min(deg(from), deg(to))` is the LP optimum exactly;
  for irregular shapes it remains a guaranteed upper bound. Per-edge
  cardinality sums catalog `edge_count` across the alternation set,
  and the result is clipped from above by the cartesian product of
  per-variable label counts so tiny graphs don't get
  astronomically pessimistic estimates. 9 closed-form unit tests
  in `cost::cardinality::tests` (triangle, K_4, 4-cycle, alternation
  sum, dangling-edge, cartesian clip, no-stats fallback, two-var
  single-edge, regression vs the prior naïve formula).
- **`exec::leapfrog::MergeSortedUnion`** — k-way ascending dedup
  union via min-heap, the companion to `LeapfrogIntersect`. 11 unit
  tests cover passthrough, disjoint interleave, dedup, empty
  inputs, zero iterators, identical lists, dense overlap, the
  alternation-in-cycle composition, five-iterator rotating minima,
  and `collect()` vs iterative drain parity.
- **`Snapshot::sorted_partners`** in `namidb-storage`. Returns the
  partner `NodeId`s for `(edge_type, key, direction)` sorted
  ascending, merging the CSR adjacency cache (or SST fallback)
  with the memtable overlay last-LSN-wins. Drops tombstones at the
  same key. This is the storage primitive WCOJ leapfrogs over.
- **`shortestPath` and `allShortestPaths` (RFC-023).** The parser
  accepts the wrapping function form
  (`MATCH p = shortestPath((a)-[*..N]-(b))`), the lower validates
  the v0 rules (path binding required, single hop, finite upper
  bound, both endpoints in scope), and the executor terminates the
  BFS in `execute_expand` at the hop where the back-reference target
  first appears. `shortestPath` emits one row per (source, target)
  pair; `allShortestPaths` emits every distinct path of the minimum
  length and stops the BFS at that hop. The variable-length parser
  also accepts the `*..M` form (min defaults to 1) so `-[:KNOWS*..15]-`
  matches the Neo4j surface. `length(p)` now answers correctly on
  `RuntimeValue::Path` (number of hops). Closes the LDBC SNB
  Interactive IC13 and IC14 parser gap: 15/15 fixtures round-trip.
  Design in [RFC-023](docs/rfc/023-shortest-path.md). 5 new
  end-to-end tests in
  `crates/namidb-query/tests/exec_shortest_path.rs`.
- **Concurrent reads without the writer mutex (RFC-021).** Reads no
  longer take `state.writer.lock()`. A new `OwnedSnapshot` carries an
  `Arc<MemtableSnapshot>` plus the manifest, object store, and the
  cross-snapshot caches; multiple readers share it through a
  `SnapshotCell` (`std::sync::Mutex<Arc<OwnedSnapshot>>`). Writes
  refresh the cell after each successful `commit_batch` / `flush`,
  so subsequent reads see the latest durable state. Snapshot
  isolation, the single-writer-per-namespace invariant from RFC-001,
  and the Bolt bookmark format all stay intact. Integration test
  `crates/namidb-server/tests/concurrent_reads.rs` measures a ~7x
  fan-out at 8 readers on a 4-core box (~1x before this change).
  Design in [RFC-021](docs/rfc/021-concurrent-reads.md).
- **`MemtableSnapshot`** in `namidb-storage`: a read-only,
  point-in-time view of a `Memtable` with the same iter / get /
  iter_label / iter_edge_type surface. Snapshots own their memtable
  view via `Arc` instead of borrowing from the writer.
- **`OwnedSnapshot`, `SnapshotCell`, `WriterSession::owned_snapshot`**
  in `namidb-storage::read`. The cell lives in
  `namidb_server::AppState` so HTTP and Bolt share one published
  snapshot per process.
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

#### Changed
- `namidb_server::Config` gained `bolt_listen: Option<SocketAddr>`.
  When unset the server stays HTTP-only (the previous behaviour).

#### Fixed
- **WCOJ leaf-multiplicity matches binary per-path semantics.**
  Before, `MultiwayJoin` emitted one row per `(a, b, c, ...)` tuple
  regardless of how many type combinations or parallel edges
  actually closed the cycle, because `Snapshot::sorted_partners`
  collapses partners to a set. The fix walks `out_edges` /
  `in_edges` per listed type at the leaf and multiplies the counts
  across constraints, so `RETURN a, b, c` without `DISTINCT` gets
  the same row count from the WCOJ and the binary paths even on
  alternation queries that match multiple edge types between the
  same pair of nodes.
- **`namidb-py::value_to_py` handles `Date` / `DateTime`.** The
  Python binding's value mapping kept an exhaustive match against
  the original 7 `Value` variants and stopped compiling after
  `Date(i32)` / `DateTime(i64)` landed in `namidb-core`. Mirror
  the conversion the runtime-value path already does: turn
  `Date(days)` into a `chrono::NaiveDate` and `DateTime(micros)`
  into a `chrono::DateTime<Utc>` so the caller gets a real
  `datetime.date` / `datetime.datetime` from pyo3.

#### Breaking
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
