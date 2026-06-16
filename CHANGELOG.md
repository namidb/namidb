# Changelog

All notable changes to NamiDB will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project loosely follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

While the engine is pre-1.0, breaking changes can land in minor
versions. They will always be called out in the **Breaking** section
below and in the release notes.

## [Unreleased]

## [0.18.0] - 2026-06-15: Cypher write ergonomics and bulk-load

### Added

- `SET n += {map}` (merge) and `SET n = {map}` (replace). The map forms
  previously parsed but errored at execution. `+=` merges the map into the
  node or relationship (a null value removes a key); `=` replaces the whole
  property set. Uniqueness and NOT NULL are checked against the final set, so
  a `=` that drops a required column is rejected rather than committed.
- `datetime()` and `date()` constructors. No-arg returns the current UTC
  instant / today, a single ISO-8601 string parses to the same. Previously
  every temporal constructor fell through to "not supported in v0".
- Label predicate in `WHERE`: `WHERE n:Label`, `n:A:B`, and `NOT n:Person`,
  reusing the existing label-membership builtin.
- Bulk-load edges from Parquet: `load_edges` in the storage loader plus
  Python `Client.load_parquet_nodes` / `load_parquet_edges`, the
  file-to-graph fast path with no per-row dict construction. The loader was
  nodes-only.

### Changed

- Variable-length paths are now allowed under `OPTIONAL MATCH`. Lowering and
  the walker already handled the combination, only the parser rejected it.

### Fixed

- `UNWIND list AS row MATCH (a {x: row.a}), (b {x: row.b})` now propagates the
  row binding across comma-separated pattern parts, so the canonical bulk-edge
  load (look up both endpoints per row, then CREATE the edge) runs in one
  round-trip instead of failing with "binding row not bound".
- The Python low-level bulk API (`upsert_node`, `upsert_node_with_labels`,
  `merge_nodes`) now enforces declared unique constraints, the same check the
  Cypher CREATE path runs, instead of committing duplicate unique-property
  values silently.

## [0.17.0] - 2026-06-14: int8 vector storage and scoring

### Added

- int8 vector storage. A new `Int8Vector(dim)` property type stores an
  embedding as one `FixedSizeBinary(4 + dim)` column: a 4-byte per-vector f32
  scale followed by the int8 codes (`x_i ≈ code_i * scale`), 4x smaller than
  `FloatVector`. Writing an f32 vector to an int8 column quantizes it on the
  fly with a per-vector max-abs scale, which 0.16.0's `namidb-bench
  vector-recall` harness measured at recall@10 around 0.98 to 0.99 at dim 256
  and 1536.
- The similarity builtins (`cosine_similarity`, `dot_product`,
  `euclidean_distance`) and `size()` now accept a stored int8 vector, scoring
  an f32 query against it by dequantizing on the fly (the asymmetric case:
  f32 query, int8 stored), with f64 accumulation. Encoding is fixed per
  property, so compaction never sees a mixed f32/int8 column. Declaring an
  int8 column goes through the programmatic / offline-builder schema path for
  now; a vault-load `--quantize` opt-in is a follow-up.

## [0.16.0] - 2026-06-13: read-your-own-writes traversals, int8 quantizer foundation, writer-lock and space-leak fixes

### Added

- MCP `vector_search` gains an optional `where` argument: a Cypher predicate
  over the matched node that pre-filters the candidate set *before* cosine
  ranking, so a metadata-constrained semantic search returns the true top-k
  within the filter instead of post-filtering (and truncating) the global
  top-k. Still read-only.
- A traversal that runs directly above a write in one statement now sees the
  edge that write just staged. `CREATE (a)-[:R]->(b) WITH a MATCH (a)-[:R]->(x)
  RETURN x` previously failed with a "write operators require execute_write"
  error and committed nothing; it now stages the write and expands over the
  read-your-own-writes overlay, returning `b`. Closes the last RFC-026 Q1 gap.
- `namidb-bench vector-recall`: a harness that measures int8 quantization
  recall@k and latency against exact f32, plus a per-vector int8 quantizer in
  `namidb-core` (`quantize_i8` / `dequantize_i8`, max-abs scale so the full
  int8 range is used at any dimension). Foundation for int8 vector storage; no
  on-disk format change yet.

### Changed

- MCP `vector_search` memoises the query embedding (keyed by embedder id +
  text) and the namespace's stored embedder id, so a query repeated in a RAG
  loop is embedded once — one API call instead of N for a remote embedder —
  and the embedder-mismatch guard no longer runs an extra lookup per search.
- The server now applies soft write-stall backpressure by default
  (`--write-stall-l0` defaults to `24`, three times the reactive-compaction
  trigger). Before, backpressure was off, so a writer could outrun compaction
  and let L0 grow unbounded, inflating read amplification. It is invisible
  under normal load and only delays a committed write under sustained
  overload; set `--write-stall-l0 0` to restore the old unbounded behaviour.

### Fixed

- A Bolt statement that failed inside an explicit transaction (for example a
  mid-transaction query timeout) left the session holding the global writer
  lock. A `RESET` after the failure recovered the session to a usable state but
  never released the writer, so one client could wedge every other write on the
  server until its connection closed. The transaction is now rolled back as
  soon as an in-transaction statement fails, and `RESET`/`GOODBYE` release it
  from the failed state too.
- The orphan sweep now reclaims superseded manifest snapshots
  (`manifest/v{N}.json`) below the retention horizon. Before, every commit,
  flush, and compaction wrote an immutable manifest snapshot that was never
  deleted, so the `manifest/` prefix grew by one object per write forever:
  unbounded space amplification independent of how much data was stored.

## [0.15.0] - 2026-06-10: production hardening — write timeouts, NOT NULL, backup/restore, token roles, bounded top-k

### Breaking

- Server-initiated writes are now bounded by a wall-clock timeout that
  defaults to `--query-timeout` (30s). Before this, writes ran unbounded; a
  bulk load or large `MERGE`/`DELETE` that takes longer than the budget now
  aborts and rolls back. To keep the old unbounded behaviour set
  `--write-timeout 0s` / `NAMIDB_WRITE_TIMEOUT=0s`, or raise the budget to a
  value that fits the workload. Embedded callers are unaffected: the bare
  `execute_write` / `execute_write_staged` stay unbounded.
- A property a label declares `nullable = false` is now enforced on write,
  where before the flag was advisory. A `CREATE` that omits it or sets it to
  `NULL`, a `SET p = NULL`, a `REMOVE p`, or adding the declaring label to a
  node that lacks the value are all rejected with a constraint error. Schemas
  that declared `nullable = false` without supplying the value on every write
  will now see those writes fail; mark the property nullable, or supply a
  value. Enforcement is node-only and pure (no extra read): edges still carry
  no declared-property validation.

### Added

- Write-query timeout. A write statement now honours a wall-clock deadline,
  so a runaway `MERGE`/`DELETE` is aborted instead of pinning the single
  writer of a namespace. The deadline rides the same cooperative
  cancellation the read path uses, and a write that overruns has its pending
  batch discarded, so nothing partial is committed. Configure it with
  `--write-timeout` / `NAMIDB_WRITE_TIMEOUT`; it defaults to the read budget
  (`--query-timeout`), and `0s` opts a write back into running unbounded. It
  applies to HTTP and Bolt auto-commit statements and to each statement of a
  Bolt explicit transaction. Embedded callers reach it through the new
  `execute_write_with_deadline` / `execute_write_staged_with_deadline`; the
  existing `execute_write` / `execute_write_staged` stay unbounded.
- NOT NULL constraint enforcement. Declaring a property `nullable = false`
  now makes it a hard write-time invariant, alongside the existing unique
  constraint, so a label's required properties cannot be left null through
  `CREATE`, `SET`, `REMOVE`, `MERGE`, or a label addition. Violations surface
  as `ExecError::Constraint` (HTTP 4xx / Bolt failure), the same path unique
  violations take.
- Consistent backup and restore. `namidb backup --from <uri> --to <uri>`
  copies a point-in-time snapshot of a namespace: it pins a manifest version
  and copies its closure (the manifest, every SST and its bloom / unique /
  equality / label-index side-cars, and the WAL segments still needed for
  recovery). Every one of those objects is immutable once written, so the
  snapshot is consistent by construction rather than a racy `aws s3 sync`.
  `namidb restore --from <uri> --to <uri>` is the same copy in the recovery
  direction. The destination is left as a self-contained, openable namespace
  (renumbered to a fresh version 0). `--version N` pins a specific committed
  version; `--force` overwrites a destination that already holds a namespace.
  Also exposed as the library function
  `namidb_storage::copy_namespace_snapshot`. Run against a quiescent source;
  there is no `FREEZE` yet, so a concurrent compaction plus orphan sweep
  could delete a pinned object mid-copy.
- Per-token roles and multiple tokens. A new `--auth-tokens-file` /
  `NAMIDB_AUTH_TOKENS_FILE` points at a JSON file of tokens, each granting
  `read-only` or `read-write`:

  ```json
  { "tokens": [
      { "name": "ci",        "token": "…", "role": "read-write" },
      { "name": "dashboard", "token": "…", "role": "read-only"  }
  ] }
  ```

  A read-only token may run reads but is refused on any write or admin flush,
  over both HTTP (`403 Forbidden`) and Bolt (`Neo.ClientError.Security.
  Forbidden`). Keeping secrets in a file also keeps them out of the process
  arguments. The existing single `--auth-token` still works and grants
  read-write; the tokens file takes precedence when both are set.
  Per-namespace token scoping is deferred until multi-namespace routing
  exists (the server serves one namespace today).

### Changed

- Bounded top-k for `ORDER BY ... LIMIT`. When a limit bounds the result to
  `k = skip + limit` rows and `k` is smaller than the number of candidates,
  the `TopN` operator now keeps only the `k` best in a max-heap instead of
  materialising and sorting every candidate: O(n log k) time and O(k) memory
  rather than O(n log n) and O(n). This is the hot path for K-nearest-neighbour
  vector search (`ORDER BY cosine_similarity(n.embedding, $q) DESC LIMIT k`),
  which previously sorted the whole scanned set. Results are identical to the
  full sort, ties included. (The flat O(n) scan and uncompressed f32 vectors
  remain; int8 quantization and an ANN index are the next steps.)

## [0.14.0] - 2026-06-07: vector search and embeddings, TLS, Prometheus metrics, leveled-lite compaction

### Added

- Vector search. Three scalar similarity/distance builtins,
  `cosine_similarity`, `dot_product` and `euclidean_distance`, operate on a
  stored vector property or a numeric `$param` array, so K-nearest-neighbour
  search is expressible through the existing scan + `ORDER BY` + `LIMIT` path:
  `MATCH (n:Note) WHERE n.embedding IS NOT NULL RETURN n ORDER BY
  cosine_similarity(n.embedding, $q) DESC LIMIT 10`, with the `WHERE` clause
  acting as a pre-filter on the candidate set. NULL propagates, a
  zero-magnitude vector makes cosine NULL, and a dimension mismatch is a clear
  error; `size()` returns a vector's dimension.
- Embeddings on vault load. `load-vault --embed` (and the MCP server by
  default) computes a text embedding for each note and stores it as an
  `embedding` property, so semantic search works over an Obsidian vault. The
  default embedder is local, deterministic and dependency-free (a hashing
  embedder; lexical similarity). Build with `--features remote-embedder` and
  set `NAMIDB_EMBEDDER=remote` plus `NAMIDB_EMBED_PROVIDER` (openai, voyage,
  cohere, gemini or jina) and an API key to embed with a real model instead;
  the load batches notes into one request per call. Each note is stamped with
  the embedder identity, and a search refuses (rather than ranking wrongly) if
  the namespace was embedded by a different model than the one querying it; a
  sync that would switch the embedder is likewise refused.
- MCP `vector_search` tool: semantic K-NN over the vault. It takes
  natural-language query text, embeds it server-side with the same embedder
  that indexed the notes, and returns the closest notes by cosine similarity.
- Read-your-own-writes for edges (RFC-026 edge overlay). A traversal that
  runs after an edge is staged in the same transaction now sees that edge:
  every edge read path (`out_edges` / `in_edges` over both the SST scan and
  the CSR adjacency, plus the WCOJ `sorted_partners` and the edge-type scan
  and count) merges the writer's staged batch last-LSN-wins, so a staged
  upsert is traversable and a staged tombstone hides a committed edge. This
  completes the node overlay shipped in 0.13.0; a read against a plain
  committed snapshot is unchanged, and reads outside a write context have
  nothing staged and pay nothing. Running a read pipeline directly above a
  write within one statement (`CREATE (a)-[:R]->(b) WITH a MATCH
  (a)-[:R]->(x)`) is still a follow-up: the staged edge is visible to a later
  statement or an in-transaction read, not to an expand stacked on the same
  statement's write.
- Operability: a lock-free liveness probe, graceful `SIGTERM`, and a container
  healthcheck. A new unauthenticated `GET /v0/livez` answers without taking any
  lock or reading namespace state, so a long write or compaction (which holds
  the writer lock) no longer makes a liveness probe hang and get the server
  killed; `GET /v0/health` now reports the published snapshot's version and
  epoch without the writer lock too. The server drains on `SIGTERM` (what
  `docker stop`, systemd and Kubernetes send), not only on Ctrl-C: a shared
  signal stops the HTTP server and the Bolt listener together. A `Dockerfile`
  ships with a `HEALTHCHECK` targeting `/v0/livez`.
- TLS on the serving path (`--tls-cert` / `--tls-key`, env `NAMIDB_TLS_CERT` /
  `NAMIDB_TLS_KEY`). One PEM certificate chain and key enable rustls on both
  the HTTP REST API (HTTPS, served via `axum-server`) and the Bolt listener
  (a TLS handshake in front of the same session loop, since the Bolt session
  is generic over its transport). The `ring` crypto provider is selected
  explicitly, so the build needs no aws-lc-rs C toolchain. Both `--tls-cert`
  and `--tls-key` must be set together; with neither the server stays
  plaintext exactly as before, and the graceful-shutdown drain works on both
  the TLS and plaintext paths.
- Prometheus metrics and a slow-query log. A new unauthenticated `GET
  /v0/metrics` renders the process query metrics in the Prometheus text
  exposition format: `namidb_queries_total` and `namidb_query_duration_seconds`
  (a latency histogram), both split by protocol (`http` / `bolt`) and read vs
  write, plus `namidb_queries_in_flight`, `namidb_slow_queries_total`,
  `namidb_build_info` and `namidb_uptime_seconds`. The registry is a small
  hand-rolled set of lock-free atomic counters, so the hot path stays
  allocation-free and pulls in no new dependency. Both serving paths feed one
  shared registry, and the stopwatch stops before the optional write-stall
  sleep, so backpressure is not counted as query latency; Bolt schema
  introspection probes are not counted as queries. Separately,
  `--slow-query-threshold` (env `NAMIDB_SLOW_QUERY_THRESHOLD`, default `1s`,
  `0s` disables) logs any query at or above that wall-clock at WARN with its
  protocol, kind, status, elapsed and statement text. The statement text only,
  never its parameters, which can carry sensitive values.

### Changed

- Leveled-lite compaction (RFC-027 P4). Compaction keeps one SST per `(kind,
  scope, level)` across L1..Lk with a per-level byte budget
  (`NAMIDB_COMPACTION_BASE_BYTES` / `NAMIDB_COMPACTION_LEVEL_RATIO`, defaults
  8 MiB / 10). New L0s drain into L1, and a merge cascades into a deeper level
  only when the accumulated bytes exceed that level's budget, so the large
  base levels are rewritten rarely. This bounds write amplification, the cost
  the previous full-bucket compaction traded for its space bound, while space
  and read amplification stay bounded. Tombstone and superseded-version GC now
  runs only on the merge whose output is the bucket's deepest occupied level,
  where the LSM invariant (a shallower level holds the newer LSN for a key)
  guarantees the dropped tombstone shadows nothing.

### Fixed

- Unique constraints are enforced for non-string properties. A property
  declared unique is now checked on `CREATE` and `SET` regardless of type,
  not only for strings: a duplicate integer, float, bool, date or other value
  is rejected with a constraint error, the same as a duplicate string. String
  values keep using the `O(log N)` property index; other types fall back to a
  label scan and a typed-value compare (a typed index is a later
  optimisation). The check reads through the read-your-own-writes overlay, so
  an intra-batch duplicate is caught too.
- The read-query timeout now cancels cooperatively inside the storage decode,
  not only at query operator boundaries. The deadline rides a task-local in
  `namidb-storage`, so the CPU-bound SST body fetch and the per-batch /
  per-row decode and merge loops probe it and abort a single long-running
  operator (for example a large scan or a big leveled SST decode) mid-flight
  with a timeout, instead of pinning a worker until the operator returns.
  Untimed reads, writes and compaction are unaffected (the probe is a no-op
  when no deadline is in scope).

## [0.13.0] - 2026-06-07: read-your-own-writes for nodes, compaction space reclamation, query timeout and row cap

### Added

- Unique constraint enforcement on `SET`. A property declared unique is now
  enforced when an existing node's value changes through `SET`, not only on
  `CREATE`: a `SET` that would collide with another node's value for that
  property is rejected with a constraint-validation error, while rewriting a
  node's own value (or a no-op write) is allowed. The check reads through the
  read-your-own-writes overlay, so a value staged earlier in the same
  uncommitted batch is considered too.
- Read query timeout (`NAMIDB_QUERY_TIMEOUT` / `--query-timeout`, default
  `30s`, `0s` disables). A single HTTP or Bolt read, including a read
  inside an open transaction, is bounded by a wall-clock deadline checked
  at operator boundaries and inside the scan and expand loops; a query
  that runs past it aborts with a timeout error instead of pinning a
  worker. Writes are bounded by the transaction lifecycle, not by this.
- Read query row cap (`NAMIDB_QUERY_ROW_CAP` / `--query-row-cap`, default
  `0` = unlimited). Bounds the rows any single read-query operator may
  materialise; a query whose operator output would exceed the cap aborts
  with a row-cap error. The multiplicative cross product is rejected
  before it builds, and a runaway expansion fails fast mid-loop, so a
  pathological query cannot blow up memory first.
- Reactive compaction trigger and soft write stall (RFC-027 P5).
  `NAMIDB_COMPACTION_L0_TRIGGER` (default `8`) compacts a bucket as soon
  as a flush leaves it with that many L0 SSTs, instead of waiting for the
  periodic compaction tick, so read amplification stays bounded under
  sustained writes. `NAMIDB_WRITE_STALL_L0` (default `0` = off) with
  `NAMIDB_WRITE_STALL_DELAY` (default `50ms`) applies backpressure to a
  committed write when L0 climbs past the threshold, so the writer cannot
  outrun compaction without bound.

### Changed

- Compaction reclaims tombstones and superseded versions (RFC-027 P3).
  Each compaction is now full-bucket: it merges a bucket's existing L1
  with its new L0s into a single L1, so the result is the bucket's only
  SST at the new version and a key whose newest version is a tombstone (or
  a fully-deleted node/edge) is dropped entirely instead of carried
  forever. A reader pinned at an older version still observes the delete
  through the retained source bodies. This bounds on-disk size for
  delete- and update-heavy workloads; the cost is a full-bucket rewrite
  (write amplification), which leveled compaction will later bound.
- Orphan sweep is now reference-counted and snapshot-horizon aware
  (RFC-027), and enabled by default. It keeps every object referenced by
  any manifest version from the retention horizon (the oldest version a
  live reader is pinned to) up to current, then deletes the rest, so it
  reclaims compaction inputs and failed-commit orphans without a
  wall-clock guess and can never delete a body a live reader still needs.
  `min_age` stays as a small secondary guard for the body-PUT-then-CAS
  race; `NAMIDB_SWEEP_DELETE=false` keeps a dry-run available.

### Fixed

- Read-your-own-writes within a statement and an open transaction
  (RFC-026, node overlay). A read sub-plan that runs after a write in the
  same statement or transaction now sees the staged rows, so `CREATE` then
  `MATCH`, `MERGE` after `CREATE`, and duplicate detection inside one
  uncommitted batch all return the right result instead of reading the
  pre-call committed snapshot. Reads outside a write context are
  unchanged. Staged edges are not yet visible to traversals; that is a
  follow-up.

### Breaking

- The orphan sweep deletes by default (`NAMIDB_SWEEP_DELETE` now defaults
  to `true`); the retention horizon makes that safe. Set it to `false` to
  keep the previous dry-run behaviour. The `namidb_storage::sweep_orphans`
  function gained a `retention_horizon` parameter.

---

## [0.12.0] - 2026-06-05: multi-label nodes, secondary indexes, per-label stats, pluggable Bolt auth, and a hardening pass

This release reconciles two lines that forked at 0.11.0 and advanced in
parallel: the published 0.11.x tags (pluggable Bolt auth, the logoff hook,
variable-length path bindings) and main (multi-label nodes, the secondary
equality index, per-label statistics). They are unified here, and releases
are cut from `main` from now on. The intervening 0.11.0, 0.11.1 and 0.11.2
tags shipped without changelog entries; their changes are folded below.

### Added

- **Multi-label nodes, end-to-end.** A node carries a set of labels rather
  than one. New `LabelId`/`LabelDictionary` in the core, an id-primary
  storage core that keeps the label set per node, Cypher that matches on any
  subset of labels, intersection-aware cardinality for multi-label `MATCH`,
  and Python bindings that read and write the label set.
- **Secondary equality index for non-unique properties.** Indexed properties
  that are not declared unique now get a value to node-set index (storage
  half), and the planner uses it for equality predicates instead of scanning.
- **Per-label property statistics (RFC-025, Phase 1).** Statistics are kept
  per `(label, property)` so selectivity estimation no longer blends
  unrelated labels.
- **Pluggable Bolt authenticator.** Embedders can supply a custom
  `Authenticator` instead of the built-in open/token schemes, plus a
  `Backend::logoff` hook so they can drop per-connection identity on `LOGOFF`.
- **Variable-length path bindings.** `MATCH p = (a)-[*1..2]->(b) RETURN p`
  now binds the whole path to `p`.
- **Real Bolt transactions.** `BEGIN`/`COMMIT`/`ROLLBACK` run as genuine
  multi-statement transactions over the single-writer session.
- **Background compaction scheduler.** A server task runs L0->L1 compaction
  and orphan sweep on a tick.
- **GUI client support.** G.V()/gdotv support (Neo4j connection type, write
  counters, elementId point lookup) and Memgraph schema-introspection
  procedures for GUI clients.
- **Query surface.** `timestamp()` (epoch milliseconds), standard string,
  math and list scalar builtins, `SKIP`/`LIMIT $parameter` resolution at
  execution time, and synthesised bindings for anonymous elements in a
  bound path.
- **Unique constraint enforcement on CREATE.** A property declared unique is
  now enforced on write: creating a node whose unique string property
  duplicates an existing value is rejected (over Bolt as
  `Neo.ClientError.Schema.ConstraintValidationFailed`) instead of silently
  upserting. `MERGE`'s create branch inherits the check.

### Fixed

- **Read-after-write through the property index.** `commit_batch` now resets
  the cross-snapshot property index, the same way `flush` and `attach_ssts`
  already did. Before this, a node committed without a flush could be
  invisible to `lookup_node_by_property` once that `(label, property)` pair
  had been warmed, returning stale or missing rows. Covered by a regression
  test.
- **Failed writes no longer leak into the next commit.** A write statement
  that errored after staging some mutations left them in the shared writer's
  pending batch, where the next write's commit sealed them. The pending batch
  is now discarded on a staged-execution error and always on ROLLBACK.
- **Crash durability on the local backend.** Writes through the local
  filesystem backend now fsync the file and its parent directory (and the
  multipart path on completion), so a committed write survives an OS crash or
  power loss. Previously the backend relied on `LocalFileSystem`'s tmp+rename
  with no fsync, so self-hosted (non-S3) deployments were not crash-safe.
- Python bindings adapted to the `NodeView` label set.

### Security

- **Bolt RESET no longer bypasses authentication.** A client could complete
  the handshake and then send `RESET` to reach the READY state without
  `HELLO`/`LOGON`, running queries unauthenticated even with a token
  configured. RESET now only recovers an already-authenticated session.
- **Parser recursion is bounded.** Deeply nested input (thousands of nested
  parens, lists or maps) could overflow the stack and abort the whole
  process. Expression nesting past a fixed depth is now rejected with a parse
  error, which also bounds the expression evaluator.

---

## [0.10.0] - 2026-05-31: Live incremental sync (--watch, frontmatter links and aliases, nested tags)

### Added

- **Incremental vault sync.** `sync_vault`/`sync_graph` parse the vault, read
  the prior `content_hash` state through a column projection, and re-index only
  what changed. Unchanged notes are not re-written and their bodies are never
  loaded; edges and tags are reconciled exactly as a prune-load. The contract is
  asserted directly: after a sync the graph is byte-identical to a fresh
  prune-load of the same disk state, across add/modify/delete/unchanged with
  link, embed and tag changes, with placeholders on and off. `VaultSyncOutcome`
  reports the change counts.
- **Live `--watch` in the CLI and MCP server.** `namidb load-vault --watch
  <dir>` does an initial mirrored sync, then watches the vault (debounced 400ms)
  and re-syncs on every change until Ctrl-C, so the graph stays a live index.
  The MCP server gains `--watch` (requires `--vault`): a background task
  re-syncs incrementally on each change and republishes the snapshot, so agent
  reads keep flowing while the graph updates under them. A missed or coalesced
  filesystem event never desyncs the graph, because each sync re-walks and
  re-hashes the vault rather than trusting the event.
- **Nested tags as a `:SUBTAG_OF` hierarchy.** A nested tag like `#area/db` now
  materializes its ancestor `:Tag` nodes (`area`) and a child-to-parent
  `:SUBTAG_OF` edge per level, so the tag tree is a real sub-graph an agent can
  traverse. The note stays `:TAGGED` to the leaf it wrote. Prune and the
  incremental sync reconcile `:SUBTAG_OF` like the other edge types. The load
  outcome gains `subtag_edges` and `subtag_edges_pruned` (surfaced in the CLI
  and Python).
- **MCP tag-tree queries.** `notes_by_tag` now returns notes carrying the tag or
  any tag nested under it (`area` also returns notes tagged `area/db`), matched
  by name prefix. A new `subtags` tool lists a tag's immediate children via the
  `:SUBTAG_OF` edges, so an agent can walk the tag tree. The `cypher` tool
  description and the tool list note `:SUBTAG_OF`.
- **Frontmatter wikilinks as `LINKS_TO` edges.** A frontmatter property whose
  value is wholly a `[[Note]]` wikilink (or a list of them, for example `up:
  "[[Parent]]"`) now produces a `LINKS_TO` edge alongside body links, the way
  Obsidian links frontmatter properties. A value that merely contains `[[...]]`
  inside prose or a code snippet does not grow a spurious edge, and the `tags`
  property is never scanned.
- **Frontmatter `aliases` resolve links.** A note's `aliases` list now registers
  alternate names, so `[[U-R]]` anywhere resolves to the note aliased "U-R"
  instead of dangling. A real note key always wins over an alias, and the first
  note in path order wins an alias clash. Resolution covers links and embeds.
  The load outcome gains `aliases_registered` (surfaced in the CLI and Python).

### Changed

- **Latin diacritics folded in note-name resolution.** Note-name matching now
  folds the Latin-1 accented letters to their base (`á` to `a`, `ñ` to `n`, `ü`
  to `u`, and so on) before lowercasing, so `[[Matías]]` resolves to
  `matias.md` and accented and unaccented spellings collapse to one note, which
  is what a Spanish or Western European vault needs. ASCII names are unaffected.
- **MCP reads serve from a published snapshot.** Read queries no longer take the
  writer lock for the whole `execute()`; the server holds an
  `Arc<SnapshotCell>`, publishes the committed state after each commit, and
  serves reads from that snapshot without the lock. A vault load or sync no
  longer blocks every agent read for its duration. No behavior change for reads.

### Fixed

- **Duplicate-key frontmatter no longer drops the whole note.** A doubled
  top-level key (for example two `tags:`) made the YAML parser reject the entire
  document, silently dropping the note's title, role and everything else.
  Recovery is now scoped to exactly that error: regroup by top-level key, keep
  the last value (the way Obsidian resolves duplicates), and re-parse once. Any
  other malformed YAML still yields no properties, and a note that already
  parsed is never affected.
- **Non-string frontmatter `title` is kept as a string** instead of being
  coerced or dropped.
- **Engine-reserved frontmatter keys are dropped on ingest**, so a vault cannot
  overwrite the engine's own node properties.
- **Engine-owned frontmatter keys are not scanned for links**, so the body
  property the engine adds is never double-scanned for wikilinks.
- **The `placeholders` flag is exposed on the MCP loader** to match the CLI and
  Python loaders, and placeholder stubs are kept out of the note-listing tools.

### Breaking

- **Notes with accented names get a new `NodeId`.** Because a note's id derives
  from its normalized (now diacritics-folded) key, a vault that was indexed
  before this release must be reloaded or synced to rebuild the index. ASCII-only
  vaults are unaffected.

---

## [0.9.0] - 2026-05-30: Obsidian fidelity (markdown links, tags, embeds, placeholders)

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
- **Tags as graph nodes.** Each distinct tag becomes a shared `:Tag` node (one
  per name, matched case-sensitively), linked from a note by a `:TAGGED` edge,
  so tag traversals run on the graph: `MATCH (n:Note)-[:TAGGED]->(:Tag
  {name:$t})` for "notes tagged X", or `(:Note)-[:TAGGED]->(:Tag)<-[:TAGGED]-(o)`
  for "notes that share a tag". Prune reconciles stale tag nodes and edges too.
  Exposed via the load outcome (`tags_loaded`, `tag_links`, `tags_pruned`,
  `tag_links_pruned`) in the CLI and Python client.
- **MCP tag tools.** The local MCP server gains `list_tags`, `notes_by_tag`
  (accepts the tag with or without a leading `#`) and `tags_of`, so an agent
  can traverse the tag graph without writing Cypher. The `cypher` tool's
  description now names the `:Note`/`:Tag` and `:LINKS_TO`/`:TAGGED` schema.
- **Embeds as a distinct edge type.** An embed `![[note]]` now produces an
  `EMBEDS` edge instead of `LINKS_TO`, so "what does this note embed" is its
  own relation. Reference traversals span both: the MCP `backlinks`, `neighbors`
  and `orphans` tools now match `[:LINKS_TO|:EMBEDS]`, so embeds still count as
  references (an embedder is a backlink, an embed-only note is not an orphan).
  The load outcome gains `embeds_resolved`, `embeds_dangling` and
  `embeds_pruned`, surfaced in the CLI and Python client.
- **Placeholder nodes for unresolved references (opt-in).** With
  `--placeholders` (CLI) / `placeholders=True` (Python) /
  `LoadOptions::placeholders`, a link or embed whose target has no real note
  gets a stub `:Note` (`placeholder: true`, no `path`/`body`) and a real edge,
  so unresolved references show in the graph like Obsidian. The stub's id is
  the one the real note would have, so creating that note later upserts over
  the stub. Prune keeps stubs that are still referenced and tombstones the
  rest. The load outcome gains `placeholders_created`. Default off, so existing
  behavior (count dangling, no node) is unchanged.

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

[Unreleased]: https://github.com/namidb/namidb/compare/v0.13.0...HEAD
[0.13.0]: https://github.com/namidb/namidb/releases/tag/v0.13.0
[0.4.1]: https://github.com/namidb/namidb/releases/tag/v0.4.1
[0.4.0]: https://github.com/namidb/namidb/releases/tag/v0.4.0
[0.3.0]: https://github.com/namidb/namidb/releases/tag/v0.3.0
[0.2.1]: https://github.com/namidb/namidb/releases/tag/v0.2.1
[0.2.0]: https://github.com/namidb/namidb/releases/tag/v0.2.0
[0.1.0]: https://github.com/namidb/namidb/releases/tag/v0.1.0
