# Changelog

All notable changes to NamiDB will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the released CLI/server binaries, HTTP/Bolt and Cypher surfaces, PyPI
package, official container images, and durable on-disk formats follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html). Breaking changes to
those released surfaces bump the major version and are called out in the
**Breaking** section and release notes.

The Rust workspace crates are not published to crates.io. Their low-level
`pub` implementation types are not yet a stable package API; a future
crates.io release will establish and document that API explicitly.

## [Unreleased]

**Added**
- Operators can now see and kill in-flight queries: `GET /v0/admin/queries`
  lists them (id, protocol, namespace, sanitized statement, elapsed) and
  `POST /v0/admin/queries/:id/cancel` aborts one at the executor's
  cooperative probe points — the same ~hundred sites the query timeout
  uses, covering reads and writes over HTTP and Bolt, single- and
  multi-tenant (namespace-scoped: a tenant token can neither see nor
  cancel another tenant's queries). A cancelled write surfaces
  "query cancelled by administrator" (HTTP 409 `cancelled`,
  Bolt `Neo.TransientError.Transaction.Terminated`), discards its staged
  batch through the same recovery path an error takes, and leaves the
  writer immediately usable. Both routes sit behind auth and the
  write-role gate; statement text never reaches unauthenticated
  `/v0/metrics`. Found along the way: a bare giant `UNWIND range(...)`
  expansion had no cooperative probes at all — it could neither time out
  nor be cancelled; the loops now probe like every other operator.
- The static auth tokens file hot-reloads: the server re-reads
  `--auth-tokens-file` on `--auth-tokens-reload-interval` (default 10s,
  `0s` disables) and swaps the token set atomically, so onboarding or
  rotating a tenant's token no longer restarts every namespace. Change
  detection is content-compare (rename-safe); a malformed, truncated, or
  emptied file keeps the last good set serving — a reload can never
  widen access or flip auth open. Revocation applies to new requests and
  new Bolt LOGONs immediately; live single-tenant Bolt sessions keep
  their LOGON principal (documented in docs/multi-tenancy.md).

**Fixed**
- A write's durability tail (WAL PUT + manifest body PUT + pointer CAS +
  orphan retries) ran OUTSIDE the scope of both `--write-timeout` and
  `--writer-lock-timeout` on every foreground path — and Bolt has no
  outer request timeout, which is how a production log showed a
  24.5-minute write completing with status ok under 30s timeouts (a
  store stall or retry storm that eventually succeeded). The commit now
  probes the deadline (and the admin cancel flag) at its DETERMINATE
  boundaries — before any object write, and before the orphan-WAL
  retry — with the pending batch preserved on abort; it is never
  interrupted between the WAL PUT and the pointer CAS, so outcomes stay
  definite. The HTTP write deadline scope now covers the commit, Bolt
  re-arms it around its commits, the group-commit ACK wait is bounded
  by the write deadline (elapsing yields an honest
  outcome-unknown error without cancelling the committer), commits
  slower than 10s log a warn naming the durability tail, and optional
  `NAMIDB_STORE_REQUEST_TIMEOUT` / `NAMIDB_STORE_RETRY_TIMEOUT` /
  `NAMIDB_STORE_MAX_RETRIES` bound individual object-store ops
  (defaults unchanged).
- The Bolt per-message decode guard rejected legitimate small-row
  batches: its fixed allowance was 64 KiB on top of 8x the message's own
  wire size, so ~889 single-key row maps (~13x heap-to-wire
  amplification) failed while larger multi-field rows passed at any
  count — and because the limit tracked each message's size, the
  reported maximum looked like it moved between runs. The fixed
  allowance is now 2 MiB, a documented client contract: any message
  whose estimated decoded heap fits 2 MiB always decodes regardless of
  shape (roughly 11,000 tiny row maps), stable across runs. The
  shared Bolt memory semaphore's base charge moved in lockstep, and the
  rejection now reports actionable units — estimated decoded bytes, the
  limit, the message's wire size, and the formula — instead of
  "decoded Map heap length". Amplification attacks stay rejected
  (declared lengths still require bytes actually present on the wire).
- Server boolean flags now accept `1/0/true/false/yes/no/on/off` on the
  CLI (`--multi-tenant=1`) and through their env vars
  (`NAMIDB_MULTI_TENANT=1`, `NAMIDB_NO_AUTH=1`, `NAMIDB_SWEEP_DELETE=0`)
  — previously only the literal strings `true`/`false` parsed, and
  `--multi-tenant=true` was rejected outright. Unrecognized values stay
  hard errors, so junk can never silently disable auth.
- Multi-tenant mode no longer demands `?ns=` in `--store`: the value was
  parsed, validated, and thrown away (an operator writing `?ns=main`
  silently got fallback namespace `default`). The parameter is now
  optional; if present it is ignored with a boot warning naming
  `--default-namespace` as the real source of the fallback tenant.

## [2.4.1] - 2026-08-28

**Fixed**
- `EXPLAIN` in the CLI's `run` and the embedded Python client rendered
  nothing and silently EXECUTED the query — an `EXPLAIN CREATE ...`
  wrote data (the same trap the HTTP surface fixed in 2.1.3). Both now
  return the identical rendering the server serves: the optimized plan
  against the real catalog plus the `# route:` physical-access footer.
  The footer also learned the composite tuple route
  (`# route: L.(a, b) → index (tuple lookup; ...)`), including the
  memtable and scan-fallback coverage states.

## [2.4.0] - 2026-08-28

**Added**
- Composite indexes: `CREATE INDEX [name] [IF NOT EXISTS] FOR (n:Label)
  ON (n.a, n.b, ...)` now builds a real tuple posting index instead of
  falling back to a scan. Declarations are schema DDL (durable in the
  manifest, visible in `SHOW INDEXES`, accepted over HTTP, Bolt, and the
  CLI); flush materializes one paged `tuple -> [node ids]` sidecar per
  declared index and DDL-triggered compaction backfills pre-existing
  SSTs. The planner routes `WHERE n.a = ... AND n.b = ...` conjuncts
  (any order; extras stay as residual filters) through the new
  `NodeByPropertyTuple` operator, with a unique-property conjunct still
  taking priority and an exact scan fallback whenever coverage or
  freshness cannot be proven — results never depend on the sidecar.
  Numeric members follow Cypher equality (`30 = 30.0` matches through
  the index). Route observability:
  `namidb_tuple_lookup_route_total{route="native"|"fallback"}` on
  `/metrics`.
- `shortestPath` on a single bound pair now runs a bidirectional
  meet-in-the-middle BFS (exactness kept via the classical stopping
  criterion): the frontier is O(deg^(d/2)) per side instead of
  O(deg^d). Multi-target groups, `allShortestPaths`, `min >= 2`, and
  self-pairs keep the unidirectional visited BFS; the executor stamps
  the route in telemetry (`shortest_bidirectional`) so tests and
  operators can prove it engages.
- The admin backup endpoint now serves multi-tenant deployments:
  `/:namespace/v0/admin/backup` (and the header-routed unprefixed form)
  with the same destination allowlist, a process-wide single-flight,
  namespace-scoped token enforcement, and per-namespace source paths
  resolved without opening the namespace (cold namespaces stay cold).
- `namidb run` accepts `;`-separated multi-statement scripts: statements
  split outside strings/backticks/comments, run sequentially against one
  session, and stop at the first error (reported with its position). The
  CLI also gained schema DDL support — `CREATE CONSTRAINT` and
  `CREATE INDEX` now execute as schema commands (previously the CLI
  could not run DDL at all) — so a pasted schema-bootstrap script works
  end to end.
- Group commit now covers multi-tenant namespaces: the registry spawns
  one committer per namespace (cancelled on eviction, stragglers failed
  rather than hung), `--group-commit-window` applies to the multi-tenant
  HTTP and Bolt write paths, and grouping stays per-namespace.

**Fixed**
- A manifest pointer-CAS transport failure could falsify a negative ACK:
  the client was told the write failed, but the dangling WAL segment +
  manifest body were later (correctly, by crash semantics) published by
  the stalled-commit repair — silently resurrecting rows the caller
  believed rejected. The writer now RESOLVES the indeterminate outcome
  before reporting: if the pointer landed (proven by version + fence
  writer id + the segment's content hash) the commit is adopted and
  ACKed as success; if definitively absent, the orphan body and WAL
  segment are deleted first so the negative ACK stays true forever; only
  when the store keeps failing does the session poison, with an error
  message that explicitly says the write may become durable after
  recovery. Found by the new RFC-034 shared-fate fault-injection test;
  the hazard predates group commit (the inline commit path had it too).

## [2.3.0] - 2026-08-28

**Added**
- Group commit (RFC-034): `--group-commit-window` /
  `NAMIDB_GROUP_COMMIT_WINDOW` (default `0s` = off, inline commits exactly
  as before) coalesces concurrently arriving writes into one WAL append +
  one manifest CAS. Writes ACK only after the group's commit is durable
  and the snapshot republished (read-your-own-writes preserved); a
  statement failure rolls back alone; grouped writes share fate on a
  commit failure. Lifts the interactive write floor — previously ~1
  commit-round-trip per request (~309 nodes/s at 3 ms RTT in the field
  report) — to roughly group-size per round-trip. Single-tenant only.
- Multi-tenant Bolt: with `--multi-tenant`, the Bolt listener now starts
  and statements route to the namespace named by the driver's
  `database=` session parameter (the Bolt `db` field of RUN/BEGIN),
  falling back to `--default-namespace`. Namespace-scoped tokens are
  enforced per statement exactly like the HTTP middleware
  (`Neo.ClientError.Security.Forbidden` outside the scope), and an
  explicit transaction is pinned to the database named at BEGIN. The
  2.2.x startup refusal of `--bolt-listen` + `--multi-tenant` is gone.
- `POST /v0/admin/backup`: copy a live, point-in-time snapshot of the
  namespace to an operator-allowlisted destination (retention-pin lease;
  no writer pause). Disabled unless `--backup-target-uri` /
  `NAMIDB_BACKUP_TARGET_URI` allowlists destinations — the server's
  ambient cloud credentials write wherever the request says, so the
  allowlist is the security boundary. Restore stays CLI-only (offline
  destination required). Single-tenant only.
- GQLSTATUS on the Bolt wire: protocol 5.7 is negotiated, and FAILURE
  metadata carries `gql_status` (per-family, mirroring the HTTP
  taxonomy), `description`, `diagnostic_record`, and `neo4j_code`.
  GQL-aware drivers now show e.g. `57014` for a timeout instead of the
  `50N42` polyfill. Older protocol versions keep the exact legacy
  FAILURE shape.
- `shortestPath` seed grouping: consecutive seed rows sharing a source
  (the all-pairs `MATCH (a), (b)` shape) run one multi-target BFS
  instead of one identical BFS per row.

**Fixed**
- `nodes(p)` on a `shortestPath` binding returned the endpoint value at
  every intermediate hop (a 2-hop path came back as
  `["n0", "n2", "n2"]`). The trail now carries the node actually
  reached at each hop.

## [2.2.1] - 2026-08-28

Found smoke-testing the released 2.2.0 artifacts against a dense graph.

**Fixed**
- `RETURN DISTINCT <endpoint> ORDER BY <endpoint key>` over a
  variable-length pattern hung: the bare-ORDER-BY lowering places the
  sort between the distinct projection and the expand, so the 2.2.0
  endpoint-BFS eligibility missed the shape and fell back to the
  exponential path enumeration. The BFS now looks through that sandwich
  and re-applies the sort to its output.
- `RETURN DISTINCT x ORDER BY x` returned fingerprint order, not value
  order (e.g. `[10, 1, 20, 2]` for ascending) — the DISTINCT dedup
  silently re-sorted its input by an internal fingerprint whose
  lexicographic order diverges from value order. Long-standing bug,
  predates 2.2.0; dedup is now order-preserving (first occurrence).
- The query deadline and row cap now fire INSIDE a single seed's
  variable-length expansion (per hop and every few thousand edges, in
  both executors). Previously both were probed only at seed boundaries,
  so one seed's dense enumeration burned unbounded time and memory —
  a 30 s query budget was simply ignored by a six-hop expansion over a
  20-node complete graph.

## [2.2.0] - 2026-08-28

Closes the second production field report (12 findings against 2.1.4).

**Breaking**
- Secure by default: the server now **refuses to start** when no auth
  source is configured (`--auth-token` / `NAMIDB_AUTH_TOKEN`,
  `--auth-tokens-file`, or a JWT config). Pass `--no-auth` /
  `NAMIDB_NO_AUTH=1` to explicitly run open for local development.
  Previously a missing token silently booted an anonymous read-write
  server behind a log line.
- `--bolt-listen` together with `--multi-tenant` now fails at startup.
  The multi-tenant serve path never started the Bolt listener, so the
  combination silently dropped the Bolt port; Bolt is single-namespace
  (run one single-tenant server per namespace). Docs now carry the
  HTTP-only caveat at every multi-tenant routing mention.
- `shortestPath` endpoints must be bound by a previous `MATCH`; an
  unbound endpoint is now a lowering error. Previously it was silently
  accepted and returned ONE arbitrary reachable node per seed instead
  of one row per (src, dst) pair.
- `'text' + <map/node/list/...>` is now a type error instead of
  concatenating the internal Debug representation into the result
  (silent data corruption); the same applies to `toString()` on
  non-scalar values. String `+` still coerces numbers, booleans, and
  temporals (temporals render as ISO-8601). List `+` gains proper
  Cypher semantics: `list + list` concatenates, `list + element`
  appends, `element + list` prepends.
- Bolt: query timeouts and result-limit errors now carry their own
  FAILURE codes (`Neo.ClientError.Transaction.TransactionTimedOut`,
  `Neo.ClientError.Statement.ResourceLimitExceeded`) instead of
  `Neo.ClientError.Statement.ArgumentError`; deterministic search-cap
  errors are no longer the auto-retried
  `Neo.TransientError.General.DatabaseUnavailable`. Clients that
  string-matched the old codes should switch to the new ones.

**Added**
- `reduce(acc = init, x IN list | expr)` — the canonical Cypher fold.
  `reduce` stays usable as a plain identifier (soft keyword).
- Statistical aggregates: `stdev`, `stdevp`, `percentileCont`,
  `percentileDisc` (with `DISTINCT` support on the stdev family,
  NULL-skipping, and typed errors on non-numeric input).
- Machine-readable error taxonomy on HTTP: every statement failure now
  carries `neo4j_code` and `gql_status` fields alongside `code`
  (per-family GQLSTATUS/SQLSTATE-class values — `42001` syntax, `0A000`
  unsupported, `22000` evaluation, `57014` timeout, `54000` result
  limits, `53000` transient pressure; table in the server README).
  Runtime evaluation errors (division by zero, missing `$param`) are
  now 400 `eval_error` instead of a bare 500.
- `RETURN DISTINCT <endpoint> [LIMIT n]` over a variable-length pattern
  now runs as a visited-set BFS instead of enumerating every path:
  each node is expanded once per seed and emitted once, and the LIMIT
  stops the traversal early. Previously `DISTINCT ... LIMIT 200` cost
  exactly the same as counting all paths (exponential in hops).
- `shortestPath` with a minimum bound (`*2..N`) now prunes its frontier
  (`(node, hop)` dedup) instead of enumerating every walk — the shape
  previously hung on realistic graphs.

**Fixed**
- The official Docker image pre-creates `/var/lib/namidb` owned by its
  non-root uid (65532), so a named volume mounted at the canonical
  `file://` store path inherits the right ownership instead of
  crash-looping with `Permission denied`. Bind mounts (and named
  volumes created root-owned by older images) still need a one-time
  `chown -R 65532:65532` — now documented.
- `namidb backup` help no longer demands a quiescent source: the copy
  has been live-safe since the retention-pin lease design (point-in-time
  snapshot at the pinned manifest version; the janitor honours the
  lease; the residual race fails loudly instead of truncating).
- The stale claim that coordination uses "S3 conditional writes
  (`If-Match`, `If-None-Match`, ETag)" is corrected everywhere: the
  commit path depends on exactly one primitive — PUT-if-absent
  (`If-None-Match: *`, RFC-029). `If-Match` is NOT required, which
  re-qualifies stores that only implement the create precondition. The
  README now states the object-store requirement explicitly.

## [2.1.4] - 2026-08-17

**Added**
- Admission gate for concurrent full scans: queries whose plan contains a
  full `NodeScan` now acquire one of `NAMIDB_MAX_CONCURRENT_SCANS`
  permits (default 4; `0` disables) before executing, on HTTP and Bolt.
  Index lookups and expand chains are unaffected. Previously up to 1024
  concurrent scans could each materialize an entire label — in field
  testing, parallel 1M-row scans collapsed aggregate throughput to
  ~2 requests/s at 8 GB RSS while evicting every point-lookup cache
  working set. Worst-case scan memory is now `permits x largest-label`
  and indexed traffic keeps its cache locality.

**Fixed**
- The from-source install instructions now include
  `--features vector-index,text-index`: the official binaries, image,
  and wheels always shipped both, but a bare
  `cargo install --path crates/namidb-server` produced a server that
  rejects `CREATE VECTOR INDEX` / `CREATE FULLTEXT INDEX`.

## [2.1.3] - 2026-08-17

**Fixed**
- A disk-full (or unwritable-spool) flush no longer wedges the namespace
  permanently. Previously the flush retried a doomed O(corpus) build in a
  loop while holding the writer lock: every write got "writer is busy"
  forever, stuck admin/DDL requests exhausted the global HTTP concurrency
  cap until even reads and health checks hung, and only a restart
  recovered. Now the namespace degrades to read-only with a typed 507
  (`namespace degraded: ...`) on every write surface, reads keep serving
  the last committed state, `/v0/health` carries the reason, flush
  retries back off 30 s, and the first successful flush clears the state
  — self-healing once the disk frees, no restart.

**Added**
- `CREATE CONSTRAINT` / `CREATE INDEX` on already-loaded data now takes
  effect immediately: the DDL schedules the compaction pass that
  materializes the posting sidecars on pre-existing SSTs (visible as
  `namidb_compactions_total{trigger="ddl"}`). Previously the index only
  materialized on the next periodic compaction tick — or never, with
  periodic compaction disabled.
- The server now serves `EXPLAIN [RAW] [VERBOSE]` over HTTP and Bolt:
  the optimized plan against the live catalog, one row per line, without
  executing the query (previously the prefix was silently ignored and
  the query ran). A `# route:` footer states the physical access path
  per index lookup — index with sidecar coverage, memtable, scan
  fallback, or the numeric-equality caveat.
- `namidb_property_lookup_route_total{route="native"|"fallback"}` on
  `/v0/metrics`: silent index-to-scan demotions in property lookups are
  now observable, mirroring `namidb_search_route_total`.

## [2.1.2] - 2026-08-17

**Fixed**
- The first large flush no longer stalls all writes for hours on
  string-heavy indexed data: the property-posting spool called `fsync`
  once per distinct indexed value (O(rows) ext4 journal commits when
  values are per-row distinct, e.g. names), crawling at KB/s while
  holding the writer lock. The per-value fsync is gone — the scratch is
  an anonymous deleted temp file whose bytes are checksummed end-to-end
  into the index, and disk-full errors still surface at the builder's
  amortized sync. A 5M-record live load that previously wedged at ~265k
  rows now completes in 210 seconds with flushes finishing in ~20 s.

## [2.1.1] - 2026-08-16

**Fixed**
- Property DDL (`CREATE CONSTRAINT` / `CREATE INDEX` on an existing
  property) no longer wedges maintenance: schema commits now retire
  search-LSM generations whose catalog signature the DDL invalidated, and
  the flush validator treats a pure signature mismatch as staleness
  (rebuild) instead of a fatal invariant. Previously flush and compaction
  looped on the invariant error and search degraded to flat scans until a
  manual intervention.

**Performance**
- The optimizer now anchors a single-hop pattern at its selective
  endpoint regardless of the direction it was written:
  `MATCH (p:Person)-[w:WORKS_AT]->(c:Company {cid: 0})` plans the same
  index lookup + inverse expansion as the hand-anchored
  `(c {cid: 0})<-[w]-(p)` form. On the 200k-node validation dataset this
  takes the written-forward form from ~18 s (cold timeout) to instant.

## [2.1.0] - 2026-08-16: Incremental vector and full-text indexes

### Production-hardening campaign

Between the feature freeze and this release the engine went through a
systematic 25 TB readiness campaign (`docs/testing/25tb-readiness.md`): every
blocker and high-priority item closed, with the notable fixes below.

**Fixed**
- Compaction no longer wedges after `DROP` + `CREATE` INDEX: install exempted
  the states its own rebuild replaces from coverage validation.
- Traversal partner lists hydrate edge properties by row range instead of
  fetching the whole edge SST body per cold lookup.
- 2.0.6-interop markers are minted with the catalog signatures the downgrade
  paths accept, and an unadoptable base now falls back to a full rebuild
  instead of stalling on flat scan.
- Variable-length relationship aliases (`[rs:KNOWS*1..2]`) bind the
  openCypher relationship list, not the last hop.
- `search.vector` / `search.hybrid` reject unknown option keys (a typoed
  `filter` no longer runs unfiltered), and HTTP integer params beyond the
  64-bit signed range are rejected instead of silently degrading to floats.
- `UNION` branches must return the same column names in the same order.
- Shared-cache keys carry a per-store-instance token, closing a cross-store
  page collision.

**Performance**
- The immutable RAM page cache now defaults on (carved out of
  `NAMIDB_CACHE_MAX_BYTES`, no new memory bound).
- `count(r)` answers from manifest statistics in the compacted steady state.
- Vector segments are searched concurrently per coordinator round; FT4
  dictionary blocks are cached per reader.

**Observability & operations**
- `namidb_search_route_total{kind,route}` on `/v0/metrics` exposes whether
  search serves natively or fell back to flat scans.
- A nightly workflow runs the bounded-memory builder soaks, a million-row
  Search-LSM lifecycle soak, and a full ingest/search/backup cycle against
  LocalStack S3. `docs/testing/preload-runbook.md` documents the manual
  validation against a real bucket before any large production load.


Vector and full-text indexes are maintained as an immutable **base plus ordered
delta segments** committed in the same manifest CAS as the `Nodes` SSTs they
cover, replacing the full-corpus rebuild that previously ran only during a
deepest single-scope merge. A search generation is usable only when the manifest
proves that every visible `Nodes` SST is covered by a materialized segment or a
`ProvenEmpty` marker; anything unproven selects the authoritative flat scan and
is never read as an empty result. See `docs/architecture/search-lsm.md` for the
full correctness contract.

**Compatibility scope.** 2.1.0 reads 2.0.6 stores. A pre-existing full-corpus
`.vg`/`.ft` body is adopted in place as the base of a new generation, so no
reindex is required. HTTP, Bolt, CLI, Python and existing index definitions
remain compatible.

**Rolling back to 2.0.6 keeps answers correct but degrades search to an exact
scan.** Once a 2.1.0 writer commits a delta it publishes a compatibility barrier
over that index. A 2.0.6 reader cannot decode the barrier body and takes its
existing optional-accelerator fallback, so vector and full-text queries continue
to return correct results — computed by flat scan rather than served from the
index, and therefore far slower — until the index is rebuilt under 2.0.6. This
is deliberate: the base alone no longer reflects the committed deltas, so
serving from it would return stale answers instead of slow ones.

### Added — Search LSM

- **Incremental delta segments.** Each flush prepares at most one VG6/FT4 delta
  per registered index from the before/after images of that flush, so index
  maintenance no longer scans the whole node corpus. Every non-empty delta
  carries a sorted `NodeId -> version` table, letting updates, deletes,
  relabels, removal of the indexed property and native-filter changes shadow
  older payloads without mutating published objects.
- **Globally exact BM25 across segments.** Delta segments store signed corpus
  statistic changes, and every segment is scored with the same reconstructed
  global `N`, `avgdl` and per-query-term `df`, rather than re-deriving
  statistics per segment.
- **Range-readable object-native formats.** Vector and text bodies, node
  property pages and the paged adjacency index are fetched by byte range with
  page-local native filters, so a query reads the pages it needs instead of a
  whole body.
- **`object-native acceptance` CI gate.** A deterministic corpus exercises the
  physical formats with caching disabled, cold and warm, and fails on parity
  loss, post-`k` filtering, unexpected body reads, or a warm cache that still
  reaches the object store.

### Fixed

- **Zero-norm vectors under Cosine are no longer indexed inconsistently.** The
  delta classifier and the base compaction winner stream now apply the same
  membership rule as the flat scan and the V5 builders. Previously an all-zero
  embedding was written as a live delta payload and served with score `0.0`,
  displacing a legitimate neighbour the flat scan would have returned, and it
  made base compaction fail its input-count invariant.
- **Correlated `MERGE` accepts duplicate keys within one batch.** The per-chunk
  overlay only records keys whose row actually staged a mutation, so a `MERGE`
  that matched an existing node with no `ON MATCH` or trailing `SET` no longer
  fails the statement when the same key recurs.
- **Vector search through a search generation prunes again.** The coordinator
  probed every page of a clustered V5 base, which built the IVF index and then
  read past it — a KNN query cost the whole corpus. It now derives its probe
  budget from `ef` through the same policy as the direct route, so the two
  cannot drift. `eligible_rows_seen` is only accepted as an exhaustion proof
  when the probe actually covered every page, and a segment that comes up short
  widens its page budget rather than re-running an identical scan.
- **BM25 across segments no longer materialises every match.** Each segment was
  asked for its complete match set regardless of `LIMIT`, and a generation whose
  matches exceeded the materialisation cap fell back to a full node scan. Scores
  come from reconciled global statistics and are therefore comparable across
  segments, so a bounded over-fetch with refill returns the same top `k`.
- **Segment lookups within one query run concurrently.** Document-frequency
  reconciliation, prefix expansion and version-table winner probes issued one
  dependent object-store round trip per segment, per term, before anything could
  be scored. They are now issued as ordered waves; every fold stays in segment
  order, so results are unchanged.
- **The `object-native acceptance` gate compiles and runs.** It referenced an
  unbound identifier and no CI job built it, so the gate had never executed.
  The workflow now type-checks the bench before running it, and enforces a
  recall floor and a cold bytes-read ceiling instead of only structural
  assertions that could not fail.
- **The object-range page cache no longer strands callers across runtimes.**
  A RAM-only cache was built as a Foyer *hybrid* cache with no device, and a
  hybrid cache starts background workers on whichever runtime first builds it.
  Because the tier is reached through a process-global cell, any host that
  creates and drops runtimes — an embedded caller opening one per call, or a
  test binary — left every later user awaiting workers that no longer existed.
  A memory-only tier is now a pure in-memory cache with no workers and no
  runtime affinity; the hybrid path is unchanged and still requires NVMe.

### Changed — configuration

- `validate_cache_configuration` rejects a malformed value on every exact-byte
  memory rail it owns, so the official server fails to start instead of
  silently selecting a much larger default. A malformed
  `NAMIDB_RAM_PAGE_CACHE_MAX_BYTES` no longer resolves to zero, which had
  disabled an explicitly requested cache tier.
- `NAMIDB_INDEX_BUILD_MEMORY_BYTES` is documented with both of its defaults:
  256 MiB on the compaction rebuild path and 64 MiB for per-flush delta
  builders. One explicit setting overrides both.

## [2.0.6] - 2026-07-26: Bounded existing-node vector updates

This patch removes the retained multi-megabyte working set behind
`MATCH/MERGE ... SET` updates of existing vector-bearing nodes and bounds the
remaining compaction and object-store paths for long legal-corpus loads.

**Compatibility scope.** 2.0.6 reads 2.0.5 stores and migrates settled node
SSTs online. The `.nloc2` exact-record extension keeps the 2.0.5 node-locator
body as its prefix, so a rollback reader continues to use the ordinal locator
and safely ignores the appended accelerator. HTTP, Bolt, CLI, Python and
existing vector/full-text index definitions remain compatible.

### Fixed — existing-node updates and memory

- **Point updates no longer hydrate a wide Parquet page.** Current node SSTs
  append a checksummed, range-readable `NodeId -> compressed exact record`
  B+tree to the compatible locator body. Existing-node updates fetch only the
  requested records and do not open the Parquet footer or decompress the
  unrelated 1 MiB `__overflow_json` page.
- **Write-only Cypher no longer retains internal result rows.** An explicit
  execution-only result sink consumes every write but does not accumulate the
  matched nodes, `UNWIND` maps or 1,024d embeddings for Bolt/HTTP to discard.
  `WITH`, `UNWIND`, filters, ordering and aggregations are still evaluated, so
  expression errors retain atomic rollback semantics.
- **Flushes return free allocator arenas to the OS.** Admin flush drops the
  writer lock before running `malloc_trim` on glibc and keeps a process-wide
  owned permit through the blocking trim even if the HTTP request disconnects.
  Empty flushes skip the trim.
- **Exact-record construction is disk-backed.** Variable-size locator values
  and node Parquet output stream through anonymous files in
  `NAMIDB_SPOOL_DIR`; completed Parquet output is exposed through an immutable
  mmap, while B+tree pages, Arrow batches and multipart windows stay bounded
  instead of coexisting with a corpus-sized heap buffer. Flush encoding,
  compression and spool writeback run on the blocking pool. A process-wide
  single-flight permit remains owned by that blocking task after request
  cancellation, so repeated retries cannot accumulate detached corpus-sized
  builds. `sync_data` on Parquet and exact-record spools surfaces
  delayed-allocation failures and makes their pages reclaimable before object
  upload.

### Changed — compaction and object storage

- Every compaction input is file-mapped. Remote objects, including a large
  fan-in of individually small L0 files, first stream to disk, synchronise
  writeback and then mmap; local file-store bodies map directly.
- Node merge cursors are lazy, retain at most 64 decoded rows per active
  source, and activate sources only when their manifest `min_key` reaches the
  heap frontier. The complete L0 backlog still drains in one pass and winner
  order remains `(NodeId asc, LSN desc, source order)`.
- A settled 2.0.5 L1 migrates by attaching a fresh `.nloc2` sidecar without
  rewriting its Parquet body. Physical-only node migrations preserve fresh
  vector/FTS bodies, IDs and durable build generations rather than cloning and
  rebuilding the search corpus.
- Flush and offline attach admit at most four independent object uploads, with
  at most eight 5 MiB multipart parts per object. All siblings drain after an
  error, and a cancellation-safe multipart guard aborts unfinished uploads on
  task eviction; complete unreferenced objects remain janitor-reclaimable.
- The official image creates a writable disk-backed
  `/var/tmp/namidb-spool`, and the example Compose deployment mounts a
  dedicated volume there. Bare deployments can override
  `NAMIDB_SPOOL_DIR`.

### Changed — Bolt and release delivery

- Authenticated Bolt messages now default to a configurable 64 MiB ceiling
  (`NAMIDB_BOLT_MAX_MESSAGE_BYTES`), admitting 2,000-row batches of 1,024d
  vectors that exceeded the previous fixed 16 MiB cap. The unauthenticated
  path remains fixed at 64 KiB, and oversized authenticated frames receive an
  explicit `FAILURE` diagnostic before their connection closes.
- PackStream decoding now enforces a cumulative heap/cardinality budget across
  nested lists, maps and structs, caps aggregate chunk prefetch, and transfers
  RUN parameters without avoidable full-tree clones. Malicious container
  lengths cannot turn one bounded wire message into an unbounded allocation.
- Authenticated connections now share a process-wide, two-phase Bolt working
  budget (`NAMIDB_BOLT_MEMORY_BUDGET_BYTES`). Partial frames reserve
  `64 KiB + 2 × wire bytes` with fail-fast growth; only a complete data frame
  upgrades atomically to `64 KiB + 16 × wire bytes` before decode. A stalled
  client therefore retains only its bounded framing allocation and cannot
  monopolise a global ingress lock. Temporary exhaustion is retryable, partial
  frames have a configurable deadline
  (`NAMIDB_BOLT_PARTIAL_MESSAGE_TIMEOUT`, default 120 s), and small
  PULL/DISCARD/transaction-reset controls remain available under pressure.
- With `NAMIDB_MEMORY_MAX_BYTES`, pre-decode admission now holds an RAII
  reservation for the request's projected working set through execution.
  Concurrent frames cannot all race through the same sampled RSS headroom.
  Every transition to Bolt `FAILED` also rolls back an open transaction and
  releases pending result rows before its writer and timeout protections could
  be stranded.
- Bolt results stay as runtime rows until demanded by `PULL`; each page is
  converted with ownership transfer, while `DISCARD` drops rows without
  expanding vectors into PackStream values. Duplicate projection names retain
  their prior semantics, and stream completion/RESET returns the backing row
  allocation immediately.
- PyPI's post-publication integrity gate now treats an initial version-JSON
  `404` as propagation lag and keeps polling. A successful five-file OIDC
  upload no longer leaves the workflow red during the brief visibility window.

## [2.0.5] - 2026-07-26: Indexed incremental graph reads and native vector filters

This patch removes corpus-sized work from existing-key node `MERGE`, filtered
listings, label counts and metadata-filtered vector search. It also adds an
RSS-aware admission rail for long-running loaders. Existing Parquet, manifest,
property-sidecar and search-index data remain readable; the new accelerators
are optional and fall back to the 2.0.4 paths if they are absent or swept
during a rollback.

**Compatibility scope.** 2.0.5 reads 2.0.4 stores and preserves the released
HTTP/Bolt, CLI, container and Python surfaces. The workspace-only Rust crates
remain unpublished; git/path consumers constructing low-level `Manifest`,
`SstDescriptor`, property-index descriptors, `VectorGraphBody` or
`SharedAppState`, or exhaustively matching storage `Error`, must update for the
new accelerator and memory metadata.

### Changed — graph and listing performance

- **Existing-key node `MERGE` is page-addressed end to end.** Batched unique
  probes use fixed-page equality B+trees to resolve `key -> NodeId`, then a
  companion `NodeId -> physical ordinal` locator and Parquet offset indexes
  hydrate only the matching rows. A 783k-id / 2k-interleaved regression asserts
  sublinear index reads; unrelated labels no longer determine the cost of
  re-merging a small label such as `Materia`.
- **Correlated existing-node updates use one lookup per `UNWIND` batch.**
  `UNWIND $rows ... MATCH (n:Label {key: row.key}) SET ...` seeds the requested
  unique String keys from the same sidecar path, reconciles only the bounded
  transaction overlay and applies every mutation to already-hydrated IDs.
  Misses, duplicate keys, rollback and read-your-own-writes remain exact; a
  node-mutating plan no longer populates its transactional map by scanning the
  label before the first row.
- **Committed-memtable claimants carry forward incrementally.** Once a
  `(label, property)` equality map is warm, each auto-commit applies only the
  final changed rows to persistent posting maps instead of rescanning every
  node accumulated since the last flush. Unrelated cached pairs are untouched,
  embedding payloads are not retained by the accelerator, and old snapshots
  pin their physical memtable generation across commits, pressure clears and
  flushes.
- **Label-agnostic correlated anchors are batched too.**
  `UNWIND ... MATCH (n {key: row.key})-[r]->(:Label) DELETE r` deduplicates all
  keys into one multi-key equality probe per SST and one node hydration batch.
  Direct global `MATCH (n {key: row.key}) SET/DELETE n` uses that same batch
  path instead of scanning once per input row.
  Input duplicates, misses, cross-label fan-out, tombstones, renames and
  read-your-own-writes remain exact; an empty expand no longer pays thousands
  of sequential point reads before discovering that no relationship exists.
- **Bound relationship `MERGE` has a range-readable exact-point index.**
  Current forward edge SSTs carry an optional checksummed
  `(source, target) -> {LSN, tombstone, properties}` B+tree, and correlated
  endpoint pairs are probed together rather than reopening the CSR for every
  row. `NAMIDB_EDGE_POINT_MAX_ENTRY_BYTES` and
  `NAMIDB_EDGE_POINT_MAX_SST_BYTES` make admission all-or-nothing; oversized,
  absent, stale or corrupt sidecars fall back to the authoritative CSR without
  changing relationship identity or results.
- **Compacted 2.0.4 stores migrate automatically.** A lone L1 node SST is
  rewritten once to add paged property accelerators and the node locator.
  Oversized keys, missing/corrupt optional objects and old-reader janitor
  sweeps select the exact legacy fallback instead of failing the namespace or
  looping maintenance.
- **Rolling upgrades keep the unique-key fast path.** Completeness is decided
  per SST, so a legacy label-scoped `unique` sidecar and a current id-primary
  equality sidecar can jointly answer one lookup. The legacy single-owner map
  is accepted only for its exact non-empty label scope; candidates from both
  formats are last-write-wins confirmed without populating the scan fallback.
- **Label counts avoid node materialisation.** Unfiltered `count(*)` and
  `count(n)` over a node scan use manifest label cardinalities when physical
  ranges prove them authoritative. Overlapping/write-active layouts reconcile
  once, cache the exact total/per-label vector by logical generation, and
  carry it across commits with checked old/new label deltas. Immutable
  snapshots pin their generation's count cell, so a concurrent commit,
  pressure clear, flush or DDL transition cannot make an older reader observe
  a newer total or rescan the corpus.
- **Secondary filters and pagination consume bounded prefixes.** String/Bool
  equality postings support indexed `WHERE ... LIMIT`, while ascending
  indexed String ordering reads only `SKIP + LIMIT` candidates. Global
  id-primary postings widen geometrically when their first pages belong to
  other labels, and rolling manifests merge legacy unique and current equality
  cursors before last-write-wins confirmation instead of dropping to a label
  scan. A conservative capped scan also stops early for unindexed predicates
  only when disjoint immutable ranges prove that doing so is last-write-wins
  exact.
- **Global mixed-type sidecars are complete for supported scalar values.**
  Reusing a property name as Bool and String across labels no longer lets the
  synthetic schema omit one type and turn an existing unique key into a false
  miss.

### Changed — vector search

- **`search.vector` and `db.index.vector.queryNodes` filter before `k`.**
  Indexed String/Bool equality and `IN` groups are stored as bounded adaptive
  ordinal postings inside the v4 `.vg` body and intersected during ANN
  retrieval. Rejected vectors remain navigation waypoints but never consume a
  result slot or require node hydration.
- Sparse postings use sorted `u32` ordinals and dense values use bitmaps.
  `NAMIDB_VECTOR_FILTER_MAX_DISTINCT` and
  `NAMIDB_VECTOR_FILTER_MAX_BYTES` bound each build; crossing either omits the
  whole property, never a truncated posting. Legacy/high-cardinality bodies
  retain bounded sidecar eligibility plus adaptive widening and the exact
  fallback, so filters cannot silently under-fill `k`.
- Search build markers include the indexed metadata schema. Creating,
  dropping or recreating a filter property index therefore triggers exactly
  one necessary vector rebuild, including empty and all-tombstoned corpora,
  without repeated maintenance work. An authoritative deterministic vector or
  FTS build rejection is also recorded for its exact catalog signature and
  node-LSN generation: reads remain on the exact fallback, while unchanged
  idle maintenance no longer rewrites the same L1 forever. A catalog or node
  generation change retries the build.
- **Filtered hybrid BM25 no longer under-fills its sparse leg.** Authoritative
  FTS results widen geometrically under
  `NAMIDB_HYBRID_TEXT_FILTER_CANDIDATE_CAP` and evaluate the predicate before
  sparse top-k. Reaching the cap selects an exact two-pass flat fallback, so
  the setting bounds fast-path hydration without changing results.
- Natural KNN preserves Cypher's filter boundary: a source `WHERE` runs before
  embedding evaluation and before `k`, while a filter placed above
  `ORDER BY ... LIMIT` remains post-k and may return a short page. The
  optimizer no longer merges those two meanings or refills a filtered ranked
  page from lower positions.

### Changed — memory and operations

- **`NAMIDB_MEMORY_MAX_BYTES` observes total process RSS/working set.** At 90%
  the server clears shared caches plus weakly registered, reconstructible
  property/constraint maps and asks glibc to release free arenas. At the
  configured ceiling it rejects new HTTP/Bolt Cypher work with a retryable
  error while leaving `COMMIT`/`ROLLBACK` available to release staged state.
- **A valid search index that cannot fit no longer triggers an unbounded
  corpus scan.** Decoded `.vg` and `.ft` bodies share one reserved pool; a
  conservative footprint above its capacity returns retryable HTTP 503/Bolt
  `DatabaseUnavailable` with the required and assigned bytes. Consequently
  `NAMIDB_CACHE_MAX_BYTES=0` disables serving persisted vector/full-text
  indexes as well as retention. Missing, stale or corrupt optional indexes
  still use their exact correctness fallback.
- **Pressure relief no longer waits for the next request.** A process-wide
  500 ms watchdog runs the same single-flight cache reclamation. Both watchdog
  and foreground pressure winners execute cache destruction and `malloc_trim`
  on Tokio's blocking pool rather than an async serving worker. Authenticated
  `/v0/admin/flush` remains available while Cypher admission is closed and is
  serialized process-wide; multi-tenant hard pressure flushes only an
  already-open namespace instead of allocating a cold writer. Maintenance
  flushes are excluded from the ordinary HTTP timeout, and a storage-layer
  cancellation guard restores a frozen memtable synchronously if the client
  disconnects during an object-store PUT; the process-wide request cap still
  bounds callers waiting for the flush gate.
- **Evicted namespace cache guards are bounded without reopening a race.**
  `SstCache` retains at most 4096 deny tombstones, looked up allocation-free
  from the canonical `/sst/` path prefix, so a late decode for a retained
  eviction cannot repopulate dead state and namespace churn cannot grow the
  registry indefinitely. An authoritative empty manifest also retains an
  empty live-set rule across pressure clears, closing the same late-decode race
  after compaction removes the final immutable object.
- The official Compose example exposes an optional
  `NAMIDB_CONTAINER_MEMORY_LIMIT` cgroup limit. The RSS setting is an admission
  rail, not a replacement for OS containment of an already-running query.
- The official image now embeds the BSL license, OCI version/revision labels
  and a TLS/listen-aware liveness probe. Release preflights exercise the full
  native OS/architecture matrix before tagging, and GNU/Linux archives are
  gated at a `GLIBC_2.35` baseline on both x86_64 and arm64. Its builder pins
  Rust 1.85.1 instead of letting the workspace's local `stable` override
  silently replace the declared container toolchain.
- Release actions are pinned to immutable commits. GitHub verifies the exact
  16-asset archive/checksum set before making a draft public, and PyPI accepts
  resumable uploads only when any pre-existing file is byte-identical before
  verifying the final five-file release. Docker now creates and validates the
  immutable version in GHCR and Docker Hub before promoting rolling tags;
  historical dispatches cannot move `latest`, `X` or `X.Y` backwards.
- Paged sidecars are built by streaming borrowed maps/postings rather than
  cloning the corpus. Sidecar-seeded transactional maps are dropped after
  flush, and pressure reclamation no longer leaves writer-local fallback maps
  resident indefinitely.
- Documentation now distinguishes the exact retained-cache budget from total
  RSS and explains why one manifest/pointer version per durable commit creates
  temporary, janitor-reclaimable history during a bulk load.

### Fixed

- Repeated node or relationship `SET`/`REMOVE` operations now refresh the
  writer-private last-write-wins value before evaluating the next mutation and
  keep every alias of the same entity coherent. A materialised `UNWIND` batch
  can no longer overwrite an earlier patch with its stale input row.
- Memtable snapshots are checksummed and bound to the manifest's exact WAL
  descriptor closure. Legacy, truncated, corrupt or stale snapshots are cache
  misses followed by authoritative WAL replay; the v2 wire retains the public
  bincode prefix so a 2.0.4 rollback also falls back cleanly.
- Offline SST attachment carries a stable complete label dictionary and
  rejects conflicting id/name maps before the first object-store PUT.
- Vector graph decode validates dimensions, finite numeric state, graph
  ordinals, unique IDs and metadata postings before constructing the ANN
  search space. Corrupt optional bodies select the exact flat fallback.
- Optional `.vg`, `.ft` and equality-index objects removed by a rollback or
  older-reader janitor are treated as unavailable accelerators, including
  filtered vector, BM25, hybrid, label-scoped singular/batch property lookups
  and capped-posting paths. Missing/corrupt objects fall back to the
  authoritative exact scan; a valid index that exceeds its decoded-cache
  allocation still returns `CacheCapacity`, while network, authentication and
  timeout errors remain visible.
- Cache-pressure clearing removes tracked Foyer entries individually so both
  resident values and byte accounting reach zero while stale-path admission
  guards remain active.

## [2.0.4] - 2026-07-25: Batched relationship loading and bounded caches

This patch removes the remaining per-row endpoint work from idempotent
relationship loads and places every shared retained-cache tier under one
process-wide admission budget. The public API and on-disk formats remain
compatible.

### Changed — performance

- **Correlated relationship endpoints resolve once per batch.** Each unique
  `NodeByPropertyValue` operator now collects all String keys from an `UNWIND`,
  reads the unique/equality sidecars and committed memtable once, and confirms
  the deduplicated node ids with one row-group-vectorized lookup. Input order,
  duplicate keys and missing endpoints are preserved. The same path serves
  read-only queries and relationship writes.
- **Relationship-only writes keep the committed node index path enabled.**
  Plan routing distinguishes proven relationship aliases in `MERGE`, `SET`
  and `REMOVE` from node mutations. Consequently the common
  `MATCH a` + `MATCH b` + `MERGE (a)-[r:T {key: ...}]->(b) SET r...` loader
  performs exactly two endpoint batches instead of thousands of sequential
  awaits or a transactional node-index population.
- **Node views stay warm throughout edge loads.** A logical node generation
  now advances on committed node mutations, but survives edge-only commits
  and representation-only flushes. Reopened writers seed it from the durable
  manifest version, retaining cache isolation without throwing away endpoint
  locality after every edge batch.
- **The exact relationship seek remains the physical existence index.**
  Bound `(type, source, target)` probes continue to select source ranges
  through the descriptor index, binary-search the partner CSR block and
  decode only the winning property stream. A 20k-node/10k-edge debug fixture
  measured property-bearing idempotent `MERGE` at about 0.135 ms per edge,
  versus the reported 15 ms per edge before endpoint batching.

### Changed — memory and operations

- **`NAMIDB_CACHE_MAX_BYTES` caps shared retained caches.** The exact-byte
  setting defaults to 1 GiB; `0` disables all shared caches. Existing per-tier
  MiB knobs remain compatible ceilings and are scaled proportionally and
  deterministically when their sum exceeds the aggregate maximum.
- **Oversized entries are rejected before retention.** SST bodies, Arrow row
  groups, property sidecars, metadata, edge readers/streams, blooms, decoded
  vector/full-text indexes, deep `NodeView` values and CSR adjacency all use
  weighted admission. Foyer tiers use one shard, eliminating the previous
  per-shard oversized-entry multiplication.
- **Large search indexes fail safe before allocation.** Vector and full-text
  bodies that cannot fit their assigned decoded tier are not downloaded and
  decoded merely to be discarded; the query selects its existing exact flat
  fallback instead.
- **Cache pressure is observable.** Prometheus now exports
  `namidb_cache_max_bytes`, `namidb_cache_capacity_bytes` and
  `namidb_cache_resident_bytes`. The official container sets bounded glibc
  arena/trim defaults so temporary vector, BM25 and compaction working sets do
  not remain resident across many worker threads.

### Fixed

- Legacy node SSTs without property sidecars pay at most one label scan for a
  complete batch map, rather than one scan per requested key.
- Stale sidecar claimants are batch-confirmed against the current node view,
  preserving tombstone, rename, relabel and last-LSN-wins semantics.
- A single cache value larger than its tier can no longer remain resident
  above the configured budget; oversized adjacency remains queryable but is
  returned uncached.

## [2.0.3] - 2026-07-25: Exact relationship probes

This patch removes degree-dependent work from bound relationship probes,
reduces manifest and search-path overhead, and gives the published Python
distribution the same persistent vector/full-text index lifecycle as the
server. The on-disk formats and existing APIs remain compatible; the Python
compaction methods are additive.

### Changed — performance

- **Bound relationship `MERGE` uses an exact endpoint-pair seek.** When both
  endpoints are already bound, forward and inverse edge SSTs binary-search the
  endpoint key and then the partner inside its CSR block instead of decoding
  and filtering the endpoint's full degree. Anonymous propertyless `MERGE`
  uses an existence-only probe; property-bearing matches decode only the live
  winner's property bundle. Expand-Into and the final WCOJ edge-membership
  check share the same path.
- **Manifest candidate selection is indexed by kind, scope, level and key
  range.** Disjoint leveled ranges use binary search and overlapping L0 or
  legacy ranges use interval trees, reducing each point read from a manifest
  scan to O(levels · log SSTs + overlaps). Exact edge candidates are visited
  newest-LSN first so an authoritative winner can skip older bodies.
- **High-degree edge blocks retain a corpus-independent point-probe bound.**
  The dense/split skew threshold is fixed at 1,024 partners; dense UUID blocks
  use fixed-width binary search and split blocks stop once their sorted range
  passes the target.
- **ANN visited state scales with the work performed.** One-off Vamana queries
  use sparse visited marks rather than allocating and clearing a corpus-sized
  bit vector. Index construction reuses one dense epoch array across all
  refinement searches, removing an O(N²) stream of zero-fill writes.
- **Vector and BM25 result hydration is batched.** ANN and int8 rescoring fetch
  candidates in 64-node groups; BM25 uses 256-node groups and projects only
  requested text properties on the flat path. Hydrated `NodeView`s are
  consumed directly instead of being looked up and cloned a second time, with
  deadline checks before every cold batch.
- **PyPI wheels now include persistent Vamana and full-text indexes.** Embedded
  Python supports `CREATE`/`DROP VECTOR INDEX`, `CREATE`/`DROP FULLTEXT INDEX`,
  `compact()` and `acompact()`. Compaction prepares object reads, merges, index
  builds and immutable uploads outside the writer mutex, then holds it only
  for the validated manifest install.

### Fixed

- **Exact relationship reads preserve last-LSN-wins across every layer.**
  Memtable, staged overlay, overlapping SST upserts and tombstones reconcile
  before properties are decoded; a newer deletion cannot resurrect an older
  edge and a later upsert can safely restore it.
- **Search never combines incompatible index generations.** Vector and
  full-text reads require one authoritative immutable body. Zero or multiple
  generations select the flat fallback, avoiding stale vector membership and
  BM25 scores computed from incompatible corpus statistics.
- **Bound Expand-Into validates the live target label.** Flat and factorized
  execution now reject a pre-bound endpoint with the wrong label, handle NULL
  correctly, and preserve OPTIONAL MATCH null-padding.
- **Relationship `MERGE` compares every persisted property type.** Bytes,
  f32/int8 vectors, dates, datetimes, nested lists/maps and explicit NULLs
  participate in the implicit match; int8 vectors round-trip back to storage
  rather than being dropped during runtime-to-core conversion.
- **Descriptor-index fixtures cannot silently drift in debug builds.** A
  fingerprint covers every structural SST field and rejects snapshots whose
  manifest was mutated without rebuilding the paired index. A deterministic
  differential matrix checks indexed candidates against a linear reference
  across kinds, scopes, levels, overlaps, gaps and inclusive key boundaries.
- **Factorized execution parity tests now read their staged fixture.** The
  suite uses the writer overlay snapshot, so label, NULL and OPTIONAL
  assertions exercise real rows instead of an accidentally empty committed
  snapshot.

## [2.0.2] - 2026-07-25: Sustained bulk graph loads

This patch keeps idempotent multi-million-row graph loads on indexed paths,
drains compaction backlog without starving foreground writes, and bounds the
decoded read working set. The public API and on-disk format are unchanged.

### Changed — performance

- **Node `MERGE` now performs the same index seek as an explicit `MATCH`.**
  Correlated `UNWIND … MERGE (n:Label {key: row.key})` plans preserve the outer
  binding while probing the unique/equality sidecar, instead of converting the
  implicit existence check into a growing label scan. Label-free indexed
  equality probes use the global id-primary posting index and preserve
  cross-label duplicates; every canonical scalar type is rechecked with
  Cypher equality semantics.
- **Read-your-own-writes lookups stay indexed throughout large transactions.**
  The writer maintains its staged memtable incrementally, so every overlay
  snapshot is O(1) instead of replaying the growing pending WAL. Transactional
  postings cover both unique and non-unique indexed properties across
  `MERGE` and correlated `MATCH … SET/DELETE`, including commit, rollback,
  relabel, value-change and tombstone transitions.
- **Relationship `MERGE` probes only the bound endpoint's SST range.**
  Idempotent edge batches no longer rebuild a manifest-versioned whole-type CSR
  after every commit. Persisted relationship properties remain available for
  pattern matching and `ON MATCH SET`, so the sparse route is both bounded and
  semantically identical to an in-memory match.
- **Keyed relationship sweeps stay proportional to the requested keys and
  matched edges.** `UNWIND … MATCH (n {key})-[r]->(:Label) DELETE r` uses one
  indexed anchor lookup per key, exact typed/source memtable edge ranges, and
  sparse source-keyed SST identity ranges when relationship properties are not
  read. The 2,000-key regression covers both the zero-match path and a real
  deletion without materialising the complete relationship type.
- **One compaction pass consumes the complete captured L0 backlog.** Planning
  scales beyond the former three-file-shaped workload and folds every eligible
  L0 in each bucket into the leveled output, avoiding repeated passes whose
  read amplification grew with a heavy loader.
- **Compaction is single-flight and prepared off the writer lock.** Periodic
  and reactive triggers coalesce per namespace; input reads, merge/index
  rebuilds, and immutable object uploads run concurrently with foreground
  writes, followed by a short validated manifest install. A trigger received
  during a pass schedules one fresh follow-up so L0 files created meanwhile
  are drained without maintenance-task storms.
- **Decoded graph and search caches have process-wide byte budgets.** Property
  posting sidecars, graph metadata/readers/streams and vector/FTS indexes use
  weighted eviction with configurable budgets. Manifest publication retires
  superseded immutable paths, and namespace eviction removes that tenant's
  decoded entries so old snapshots cannot repopulate dead cache state.

### Fixed

- **Prepared compaction cannot publish across DDL drift.** Install validates
  the schema plus vector and full-text index catalogs captured by the prepare;
  concurrent data commits and flushes remain valid and their newer L0 files
  survive the install.
- **Aborted batches keep warm unique indexes correct.** The transactional
  unique-key map journals only touched nodes, restores them on rollback, and
  advances its baseline on commit instead of forcing a corpus rescan. Node
  commits invalidate stale property maps by generation, while edge-only
  commits and rollbacks keep unaffected maps hot.
- **Namespace retirement fences stale maintenance work.** Eviction marks the
  old incarnation retired, cancels and joins its flush/compaction tasks, and
  quiesces its writer before a replacement session can claim the namespace,
  preventing zombie recovery from fencing the new writer.
- **Bolt disconnects cancel reversible write application.** The session keeps
  reading with cancellation-safe framing while `RUN` executes, preserves
  pipelined partial frames, discards a partially staged batch on EOF, and
  releases the single writer. A durability commit that has already begun still
  runs to a definite outcome before the guard is released.
- **Traversal labels are verified from the live endpoint.** An edge type's
  declared source/destination labels are no longer treated as proof for raw
  writes that the storage API does not yet validate, preventing false-positive
  matches without changing the on-disk format.
- **Compaction pressure is observable.** Metrics separate prepare,
  install-wait and install-hold time, report per-trigger outcomes and L0/SST
  backlog, and attribute writer-lock waits across HTTP, Bolt, flush and
  maintenance paths.

## [2.0.1] - 2026-07-23: Indexed MERGE and search correctness

This patch removes the last quadratic path from idempotent bulk graph loads,
hardens index fallbacks, and improves large-object uploads without changing
the public API or on-disk format.

### Changed — performance

- **`MERGE` node existence checks use indexed candidates.** `_id`, declared
  single/composite unique keys (including keys on a secondary label), and
  non-unique string equality indexes now select candidates through point or
  transactional index probes before applying the complete pattern as a
  residual. A 500-row `UNWIND … MERGE` populates a unique-key map once per
  writer and then stays O(1) per row across commits and memtable flushes,
  instead of scanning the growing label for every row.
- **Label-agnostic node resolution is one id-primary point read.** Typeless
  graph expansion and node-by-id operations no longer repeat the same storage
  lookup once for every observed label.
- **Typeless node scans reconcile the physical node stream once.** `MATCH (n)`
  and label-free vector fallbacks no longer fan out across labels, so
  multi-label nodes appear exactly once and unlabeled nodes remain visible.
- **Persisted vector/FTS freshness is label-scoped without serving stale
  results.** New index descriptors retain the exact member NodeId range, so a
  newer SST proven to contain only another label can stay on the indexed path
  when its IDs are disjoint. Relabels, deletes, same-label writes, legacy
  descriptors, and ambiguous metadata still force the exact fallback.
- **Large compaction outputs use concurrent multipart upload.** Node/edge SSTs,
  vector and full-text indexes, bloom filters, and property sidecars at least
  5 MiB use the multipart protocol with 5 MiB chunks, bounded concurrency, and
  explicit abort on part/completion failure. Small immutable objects retain
  `PutMode::Create`; manifest CAS and commit durability are unchanged.

### Fixed

- **Unique-key `MERGE` is no longer O(N²).** Both existing and new keys take
  the same warm transactional index path, and node/relationship parameter-map
  properties now participate in the implicit match instead of being ignored.
  Numeric keys probe both integer and float encodings to retain Cypher's
  cross-type equality without giving up the indexed path; only ambiguous
  integral floats beyond the exact 53-bit range use the safe scan fallback.
- **Relationship `_id` handling is consistent.** Both inline and parameter-map
  relationship properties reject the node-only reserved `_id` slot in
  `CREATE` and `MERGE`.
- **Unique property sidecars resolve value reassignment correctly.** Reads
  confirm every historical claimant against its current node view instead of
  treating an SST's unrelated `max_lsn` as the posting's LSN and potentially
  hiding the live owner.
- **Vector search never drops persisted neighbours after index corruption.**
  Missing, legacy, or undecodable `.vg` data now explicitly selects the exact
  flat fallback even when fresh delta rows alone could fill `k`.
- **BM25 field semantics are stable across index freshness.** Requested
  property names are sorted and deduplicated for both indexed and flat paths,
  preventing fallback-only phrase/order changes or duplicate-field score
  inflation.
- **Release artifacts are version- and license-checked before publication.**
  GitHub binaries, four Python wheels plus the sdist, and multi-architecture
  Docker manifests now share a release metadata gate. Python distributions
  carry PEP 639 license metadata and the complete repository license.
- **The declared Rust 1.85 MSRV is reproducible again.** The object-store,
  Arrow table-rendering, and URL/IDNA ICU dependency lines are pinned to their
  final Rust-1.85-compatible releases, so local builds and the `rust:1.85`
  official image do not resolve crates that require newer language syntax.

## [2.0.0] - 2026-07-03: Enterprise-grade hardening — durability, resilience, and search completeness

A deep audit of the vector / text / graph search stack and the enterprise
durability path, followed by the full remediation. The major version marks the
milestone — the engine is now hardened for critical multi-tenant workloads —
**not** a breaking API change: everything that worked on 1.5.0 keeps working,
so the **Breaking** section is intentionally empty. New surfaces are additive
and new tuning knobs default to safe values.

### Added

- **`DROP VECTOR INDEX <name> [IF EXISTS]`** and **`DROP INDEX <name> [IF
  EXISTS]`** (drops a fulltext index) — a mis-created index is no longer
  permanent. The descriptor and the index's SSTs are removed in one manifest
  commit (the janitor reclaims the bodies), writes constrained by a
  wrong-dimension vector index are immediately un-bricked, and the freed
  `(label, properties)` slot can be re-created corrected. Wired through HTTP,
  Bolt, and the authz hook, mirroring the `CREATE` path.
- **Louvain and Brandes betweenness** — `CALL algo.louvain()` (modularity
  community detection, reports the partition's modularity) and `CALL
  algo.betweenness()` (bridge-node centrality), both deterministic and
  deadline-aware. Also exposed through the MCP `graph_algorithm` tool.
- **Graph projections on every `algo.*`.** An optional `{labels, edge_types,
  direction: 'natural'|'reverse'|'undirected'}` map restricts a procedure to the
  induced subgraph (unknown labels/types error rather than silently projecting
  nothing) — the canonical Neo4j GDS graph-projection workflow.
- **Phrase and prefix full-text queries.** The `search.bm25` query string now
  understands quoted `"exact phrase"` (position-adjacency, backed by positional
  postings in the new `.ft` v2 format) and trailing-`*` prefixes (vocabulary
  expansion), with identical semantics on the index path and the flat-scan
  fallback. Plain-term queries are byte-for-byte unchanged.
- **Demand-driven Bolt result streaming.** `RUN` answers `SUCCESS {fields}` only;
  each `PULL {n}` emits at most `n` records and reports `has_more` while rows
  remain, and `DISCARD` drops the remainder unsent — so a driver's `fetch_size`
  finally bounds what is buffered in flight instead of the server materialising
  and streaming the whole result at `RUN` time.
- **Writer health in the readiness probe.** `/v0/health` now reports `writer:
  "ok" | "degraded"` (plus a reason) and returns 503 while the writer is
  fenced/poisoned and the automatic reopen has not yet landed, so an orchestrator
  stops routing writes to a server that can only fail them. A lock-free read-side
  fence probe degrades readiness when a peer writer's epoch has fenced this node
  — a zombie replica no longer serves stale reads behind a green health check.
- **Byte-based flush + write backpressure knobs.** `NAMIDB_MEMTABLE_FLUSH_BYTES`
  (default 64 MiB) triggers a flush as soon as a committed write crosses it, and
  `NAMIDB_MEMTABLE_STALL_BYTES` (default 256 MiB) stalls writes until the flush
  catches up — bounding the un-flushed working set by bytes, not just the wall
  clock, so a burst loader cannot OOM the process between flush ticks.
  `/v0/health` exposes the live memtable size.
- **Bounded writer-lock acquisition.** `NAMIDB_WRITER_LOCK_TIMEOUT` (default 30s)
  caps how long a foreground write / DDL / admin flush / Bolt `BEGIN` waits for
  the writer mutex before failing fast with 503 (or a transient Bolt error), so a
  stuck or long-held writer no longer grows an unbounded request queue.
  Background tasks keep waiting as long as it takes.
- **Consistent online backups.** A backup now holds a persistent retention pin
  (a lease object under `manifest/pins/`, renewed as it copies) that the janitor
  honours, so a concurrent compaction + sweep can no longer delete objects out
  from under a running backup. Object copies stream via multipart instead of
  buffering each object whole in memory.

### Changed — performance

- **Compaction runs off the writer lock.** Split into a `prepare` phase
  (downloads, k-way merge, index rebuilds, and all object PUTs — off-lock) and a
  brief `install` phase that only re-takes the lock for the fence-checked
  manifest CAS. A multi-second compaction no longer blocks every concurrent
  write, and an abandoned prepare's bodies (at immutable UUID paths) are
  reclaimed by the orphan sweep.
- **Streaming compaction merge.** The merge is now a k-way streaming merge over
  per-source row-group cursors (only winning rows are materialised, into bounded
  chunks; sidecar / label-stats / vector / text builders observe the winner
  stream) instead of materialising the entire merged level in RAM, so peak
  compaction memory no longer scales with the level budget up to OOM.
- **O(1) published-memtable snapshots.** The memtable is a persistent
  `imbl::OrdMap`; publishing the read snapshot on every commit is now structural
  sharing instead of a full tree clone (which grew per-commit cost linearly with
  memtable size, quadratically across a flush interval, under the writer lock).
- **O(1) unique-constraint checks.** A per-writer transactional index replaces
  the full-label scan per written row, so constraint-bearing bulk writes are no
  longer O(N²); the sidecar fast-path scoping bug that disabled it in any
  multi-label deployment is fixed.
- **MIPS-correct dot-metric vector search.** Dot indexes are built and navigated
  over MIPS-augmented vectors, so the large-norm vectors that dominate a true
  inner-product top-k are actually surfaced (plain cosine navigation missed
  them). int8 index hits are rescored with the exact f32 metric, so served
  scores and top-k membership match the flat scan.
- **Faster graph & lookup paths.** Triangle counting is the compact-forward
  `O(E^1.5)` algorithm (was quadratic in hub degree); `shortestPath` /
  `allShortestPaths` use a per-seed visited-set BFS (was enumerating every walk);
  whole-graph `algo.*` builds do one label-agnostic id pass (was one full-store
  pass per label); `batch_lookup_nodes` prunes to the needed row groups with a
  byte-budgeted decoded cache (was decoding whole node SSTs into an unbounded
  per-snapshot cache); decoded `.vg`/`.ft` indexes are cached process-wide.
- **Multi-tenant cache memory is bounded.** The SST / node-view / adjacency
  caches are process-wide shared instances with global byte budgets (was a
  per-namespace budget each, so 100 namespaces could hold ~125 GiB); cache keys
  carry the namespace so tenants never collide, and eviction prunes the evicted
  namespace's entries.

### Fixed

- **A fenced or poisoned writer now recovers automatically.** Previously a single
  transient CAS/fence failure — or a second replica pointed at the same bucket —
  left every subsequent write failing forever while `/v0/health` still reported
  ok. All write paths, DDL handlers, flush ticks, and compaction-install paths
  now reopen the session in place (bounded retries), and health reports degraded
  until it lands.
- **Interrupted commits no longer wedge a namespace.** A crash between the
  manifest body PUT and the pointer CAS left an orphan body that blocked every
  future writer claim. `WriterSession::open` now repairs the stall — adopting the
  orphan when its WAL segment is durable and **content-verified** (an xxh3 in the
  descriptor proves the durable segment is ours, so a fenced peer's segment with
  a coinciding LSN can never be adopted, which would have committed
  negatively-acked writes), otherwise deleting it.
- **WAL segments and stale memtable snapshots are garbage-collected.** They were
  never deleted — unbounded object-store growth and an ever-growing cold-open
  `LIST` on every namespace. The janitor now reclaims dead WAL segments (under
  the same retention-horizon + min-age safety rule as SSTs) and superseded
  snapshots.
- **Multi-tenant namespace eviction no longer leaks.** Evicting a namespace now
  cancels its flush/compaction tasks (previously zombie tasks kept an `Arc` to
  the whole state — memtable and ~1 GiB of caches — alive forever and raced the
  namespace's next incarnation as a second writer).
- **Full-text index no longer disabled by an unrelated write.** The freshness
  gate is label-scoped: a write to a different label no longer turns every
  `search.bm25` into an `O(corpus)` flat scan; exact index-vs-flat parity is
  preserved by probing dirty ids against the index's document set.
- **Bolt decode is bounded** (a depth guard stops a pre-auth stack-overflow
  abort), the pre-delete pin re-check closes a mid-sweep backup race, and a batch
  of query-correctness fixes (relationship-uniqueness in variable-length expand,
  `LIMIT $k`, `DISTINCT` before `LIMIT`, `fingerprint_value` collisions, NULL
  ordering, self-loop double-counting, `SET` from a stale row clone,
  unique-sidecar re-verification) round out the audit.

## [1.5.0] - 2026-06-26: Vector search hardening — filtering, idempotent DDL, dimension safety

### Added

- **Structured `filter` on the KNN procedures.** `search.vector`,
  `search.hybrid`, and `db.index.vector.queryNodes` now take a `filter` map and
  compile it into the index-side predicate path — the same over-fetch + exact
  flat fallback the natural `MATCH … WHERE … ORDER BY score` form gets — instead
  of post-filtering an already-truncated top-`k` (which could starve a sparse
  tenant in a shared index to zero results). Qdrant-style shape: a scalar is
  equality, a list is `IN`, and a `{ gte, gt, lte, lt, eq, ne }` map is a range,
  AND-combined across keys (e.g. `filter: { tenant_id: $t, tier: [1, 2, 3] }`).
- **`CREATE VECTOR INDEX … IF NOT EXISTS`** and **`CREATE FULLTEXT INDEX …
  IF NOT EXISTS`** — re-declaring an index is now an idempotent no-op success
  instead of an "already exists" error (matching `CREATE INDEX` / `CREATE
  CONSTRAINT`), over both HTTP and Bolt. The int8-requires-cosine check is never
  suppressed.
- **Tunable beam width on the natural form.** A reserved, namespaced
  `$__vector_ef` parameter widens (or narrows) the Vamana beam for the
  operator-form filtered ANN, so a filtered query can finally trade recall for
  latency — previously only the procedures exposed `ef`. Explicitly **non-stable**
  (see RFC-036 for the first-class `OPTIONS { ef }` surface that supersedes it).
- **Prebuilt HTTP server binary.** Each GitHub Release now ships a
  `namidb-server-<tag>-<target>` archive built with `--features
  vector-index,text-index`, so `CREATE VECTOR INDEX` / `CREATE FULLTEXT INDEX`
  and index-backed KNN/BM25 work out of the box without building from source. A
  feature-on CI job now exercises the vector/text code paths that the default
  build compiles out, and the Docker image is built with the features on.
- **RFC-030** (the DiskANN/Vamana vector index, previously cited in code but
  undocumented) plus design RFCs **031** (ANN benchmark methodology, with
  reproducible recall/latency numbers), **032** (true pre-filtering /
  filtered-DiskANN), **033** (rich properties + named/sparse/multi-vector),
  **034** (writer concurrency), **035** (incremental index maintenance), and
  **036** (first-class `ef` surface); a `docs/multi-tenancy.md` operator guide.

### Changed

- **Filtered ANN adaptively widens before falling back to a flat scan.** A
  selective `WHERE` / `filter` no longer over-fetches a single fixed ×8 and then
  drops to an `O(n)` flat scan: the index fetch grows geometrically
  (×8 → ×32 → ×128 → ×512) until enough candidates survive the predicate, so a
  moderately selective filter is served from the index. The exact flat scan
  remains the ground-truth fallback. No-filter queries are unchanged.
- **Hybrid sparse leg no longer starves a filter.** With a `filter` present, the
  BM25 leg fetches a much deeper ranking (`k × 512`, matching the dense leg's
  maximum widening depth) before applying the predicate at fusion, rather than
  truncating to `k_sparse` first — so a filter-matching document ranked past
  `k_sparse` is no longer dropped. The depth is bounded (not the whole corpus) to
  avoid a resource cliff on a common query term. `linear` fusion's score
  calibration is window-sensitive under filtering; the default RRF fusion is
  rank-based and unaffected.

### Fixed

- **Zero-magnitude query vector divergence (cosine).** A zero-magnitude query
  made `cosine_similarity` undefined; the flat path correctly returned no rows
  (3-valued-logic `NULL`), but the index path returned `k` rows scored `0.0`,
  breaking the invariant that the index returns exactly what the flat scan would.
  The index path now agrees (empty) for a zero cosine query. Dot/L2 remain
  well-defined on a zero query.

### Changed — behaviour

- **Write-time embedding dimension enforcement.** When a vector index covers a
  `(label, property)`, writing an embedding of the wrong dimension to that
  property is now **rejected** (`ExecError::Constraint`) instead of being silently
  accepted — a single mismatched row previously poisoned the entire `.vg` build,
  permanently dropping every query for that index to the flat scan. A
  correct-dimension *bare list* (`embedding = [f, …]`, no `vector()`) is coerced
  to a dense vector so it is actually indexed rather than silently skipped at
  build time. Enforcement is scoped to the property a write touches (a `SET` of
  an unrelated property on a node with a legacy wrong-dimension embedding still
  succeeds), and a label-add validates the existing embedding against the gained
  label's index. Pre-existing rows are not retro-validated.

## [1.4.0] - 2026-06-23: World-class vector & graph — hybrid search, full-metric ANN, int8, FastRP

### Added

- **Hybrid search — `CALL search.hybrid({…})`.** Fuses a dense (vector KNN) and
  a sparse (BM25) retrieval into one ranking. Default fusion is **Reciprocal
  Rank Fusion** (RRF, k=60) — rank-based, so it needs no score calibration across
  the cosine and BM25 scales (the same default Elasticsearch / Weaviate / Qdrant /
  pgvector ship); `fusion: 'linear'` does a weighted min-max blend (`alpha` on the
  dense leg). Each leg independently serves from its index or its exact flat scan,
  so hybrid is freshness-equivalent to running the two separately and fusing.
  Configure either or both legs: `query_vector`+`vector_property` (dense),
  `query_text`+`text_property(ies)` (sparse), plus `k`, `ef`, `rrf_k`, `alpha`,
  `metric`.
- **`CALL search.vector({…})`** — vector KNN as an ergonomic procedure, with a
  tunable `ef` beam width (recall vs latency), mirroring `search.bm25`.
- **Neo4j-compatible `CALL db.index.vector.queryNodes(indexName, k, queryVector
  [, {ef}])`** — resolves the index by name and serves it through the same path.
- **FastRP structural embeddings — `CALL algo.fastRP({dimension, iterations,
  iteration_weights, normalization_strength, seed})`.** Turns pure graph structure
  into dense f32 embeddings (Fast Random Projection) with no model or external
  service. The output is exactly the `(node, embedding)` shape the vector index
  ingests, so "find structurally similar nodes" becomes a vector KNN over a `.vg`
  built from the graph itself. Deterministic for a fixed seed, near-linear,
  cancellable.
- **All three vector metrics now serve from the Vamana index.** Previously only
  `cosine` used the `.vg`; `dot_product` and `euclidean_distance` fell back to the
  O(n) flat scan. The `.vg` now stores the **original (un-normalised) vectors**
  and reranks candidates with the **true metric**, so the index score equals the
  flat scan's exactly (cosine similarity / raw dot, higher = closer; L2 distance,
  lower = closer). `euclidean` navigates with a new L2 space; `cosine`/`dot`
  navigate with cosine. (`.vg` format bumped; old `.vg` files are skipped and
  rebuilt by compaction.)
- **int8 quantization for vector indexes** — `CREATE VECTOR INDEX … WITH
  {quantization: int8}`. Stores per-vector int8 codes + a scale instead of full
  f32 (~4× smaller `.vg`, the DiskANN memory/storage win for object-storage-first
  indexes where the whole index is fetched per search). Cosine-only (the
  scale-invariant int8 space); recall stays above ~0.80 and the score is the
  quantized cosine. Opt-in per index; the default stays full-precision f32.

### Fixed

- **CRITICAL — a partial compaction silently truncated the vector/text index.**
  The `.vg`/`.ft` rebuild ran on every node merge from the *merged subset* and
  unconditionally dropped the prior index, so a shallow L0+L1 sweep deleted the
  deep index that covered L2/L3 — permanent recall loss vs the flat scan, with no
  error. The rebuild now runs only on an **authoritative** (deepest-level) merge
  that spans the full label corpus; the freshness gate is now LSN-based
  (`index_outrun_by_nodes`), so a newer `Nodes` SST at *any* level (not just L0)
  makes the read fall back to the exact flat scan until an authoritative rebuild.
- **CRITICAL — a `WHERE` predicate could be silently dropped from an indexed KNN.**
  When predicate pushdown folded a filter into `NodeScan.predicates` before the
  vector rewrite ran, the rewrite discarded those predicates. The rewrite now
  refuses to fire when the scan carries pushed predicates (the flat path keeps and
  honours them); the common filtered-ANN case (a `Filter` above the scan) is
  unaffected.
- **HIGH — a dimension-mismatched query returned silently-wrong results from the
  index** (a prefix-scored cosine) where the flat path errors. The index path now
  validates the query dimension and falls back to the flat scan (which raises the
  canonical mismatch error).
- **HIGH — `ORDER BY` with no `LIMIT` (k = u64::MAX) panicked** the executor via a
  `Vec::with_capacity` overflow; such plans are no longer rewritten to the index.
- **`euclidean_distance` KNN was rewritten in the wrong direction.** The rewrite
  required `DESC` (farthest-k) and never fired for the correct nearest-first `ASC`
  euclidean query; it now matches the metric's natural direction.
- The KNN→`VectorSearch` rewrite now reaches a KNN nested in a UNION branch, a
  `CALL {}` subquery, a join, or an aggregate (previously only top-level shapes).
- A `.vg` decoded from a corrupt/foreign body with an out-of-range entry point
  **panicked** the search; `decode` now validates the graph and the read path
  skips an undecodable `.vg` (→ flat scan) instead of erroring the query.
- The indexed-KNN materialisation loop now polls the query deadline (a filtered
  ANN can do up to `k×8` cold node lookups), and a plain-KNN flat fallback now
  materialises the full node so a `RETURN d.<prop>` is no longer null.
- **PageRank produced negative (and >1) scores** when a node's out-edges mixed
  signs but summed positive — a negative edge injected negative mass. Negative
  weights are now treated as absent (PageRank is defined for non-negative
  weights).
- **`algo.wcc` assigned non-deterministic component ids** (HashMap iteration
  order); ids are now assigned in node-insertion order, like `label_propagation`.
- `Int8Space::query_distance` returned distance 0 (a false perfect match) for a
  zero-vs-nonzero pair; it now returns 1.0, matching the f32 / pair-distance
  convention.

## [1.3.0] - 2026-06-22: Vector search reaches the index — filtered and fresh

### Fixed

- **The Vamana vector index (`vector-index`) was never used by Cypher queries.**
  The optimizer's KNN → `VectorSearch` rewrite only matched the non-terminal
  `WITH d ORDER BY cosine_similarity(d.emb, $q) DESC LIMIT k` shape (projection
  *inside* the `TopN`). The natural, common form — a terminal
  `RETURN d.x AS t, cosine_similarity(d.emb, $q) AS score ORDER BY score DESC
  LIMIT k` — lowers with the projection *outside* the `TopN`, which the matcher
  did not recognize, so every vector query silently fell back to the O(n) flat
  scan. The result-equivalence tests passed regardless, because the flat
  fallback is exact. The matcher now handles both shapes (preserving the outer
  projection on top of the new `VectorSearch`), so an indexed KNN actually serves
  from the `.vg`. A new test asserts the optimized plan contains `VectorSearch`,
  and the `ann-bench` harness reports it end-to-end. The index serves **cosine**
  KNN; dot-product and Euclidean still use the exact flat scan (the index would
  return a score on a different scale than the flat path).
- The freshness gate for both the vector and BM25 (`text-index`) indexes tested
  `scope == label` for an L0 `Nodes` delta, but node SSTs are id-primary (one SST
  spans every label, flushed with an empty scope), so the check never matched —
  a node flushed to L0 but not yet compacted into the index was invisible to the
  indexed read (neither in the index nor the cleared memtable), silently
  returning a stale top-k. The gate now detects any L0 node delta and falls back
  to the exact flat scan for that window.

### Added

- **Filtered ANN.** A `WHERE` on the KNN candidate set — a label/property
  predicate or a `cosine_similarity(...) >= t` threshold — that references only
  the searched binding is now folded into the `VectorSearch` as a residual
  `post_filter` and evaluated per candidate, instead of disabling the index
  rewrite. The index over-fetches so the filter can't starve the top-k, and
  falls back to the exact flat scan when a selective filter leaves fewer than `k`
  survivors. This is the entity-resolution pattern (vector + label filter +
  similarity threshold) served from the index. A predicate touching another
  binding still bails to the flat path, unchanged.
- **Index freshness.** The indexed KNN path unions the committed-memtable and
  staged-overlay delta that the `.vg` has not yet absorbed: freshly written /
  updated embeddings are merged into the candidate set and re-scored, while ids
  that were tombstoned, lost the indexed label, or dropped their embedding are
  suppressed. While an un-compacted L0 node delta exists the path falls back to
  the exact flat scan (mirroring the BM25 text index). A node written after the
  last compaction is visible to the index search immediately, so the ANN answer
  stays equivalent to the flat scan instead of going stale until the next compaction.
- **`namidb-bench ann-bench`** (gated behind the crate's new `vector-index`
  feature): builds a real `.vg` by compaction and measures the Vamana index's
  recall@k against the exact flat KNN plus index-vs-scan latency, over clustered
  or uniform synthetic embeddings. It also reports `cypher_index_path_reachable`
  — whether a plain KNN Cypher query is rewritten to the indexed path — so the
  reachability fix above stays regression-tested from the outside.

## [1.2.0] - 2026-06-21: Composite constraints, IF NOT EXISTS, and SHOW

### Added

- Composite (multi-property) uniqueness constraints:
  `CREATE CONSTRAINT [name] FOR (n:Label) REQUIRE (n.a, n.b, …) IS UNIQUE`.
  The tuple is unique per label; a node is exempt unless every listed property is
  present and non-null (matching Cypher semantics). Enforced on `CREATE`,
  `MERGE`-create and `SET` against the read-your-own-writes overlay, and the
  existing data is validated when the constraint is declared (a pre-existing
  duplicate tuple is rejected). Single-property uniqueness is unchanged — it
  still sets the planner point-lookup hint and emits the equality sidecar.
- `IF NOT EXISTS` on `CREATE CONSTRAINT` and `CREATE INDEX`: re-declaring an
  existing object is a no-op success instead of an error. Without it, declaring a
  constraint that already exists (by name or by the same label + property set) is
  now rejected, matching Neo4j.
- Optional names on constraints are recorded; `SHOW CONSTRAINTS` and
  `SHOW INDEXES` list the declared schema objects (columns `name`, `type`,
  `entityType`, `labelsOrTypes`, `properties`). Available over HTTP, Bolt, and
  the embedded Python client.
- The embedded Python client now executes `CREATE CONSTRAINT` / `CREATE INDEX` /
  `SHOW …` directly (intercepted before planning), so schema DDL no longer
  requires a Bolt/HTTP round-trip.

## [1.1.0] - 2026-06-21: Cypher DDL, subqueries, FOREACH, and pattern extensions

### Fixed

- `nodes(path)` over a variable-length path now carries the **start** node's full
  properties, not just its id. Previously projection pushdown pruned the source
  scan (since only `nodes(p)` referenced it), so the first path node came back
  with NULL properties and a predicate like `all(x IN nodes(p) WHERE x.age >= 1)`
  wrongly filtered the path out.

### Added

- Open-ended and parameterised variable-length relationships. `*` (any length),
  `*N..` (N or more) and `*..M` now parse, lifting the previous requirement of an
  explicit upper bound; an open upper bound is clamped to a hop cap
  (`UNBOUNDED_VAR_LENGTH_CAP`, 64) at execution. A bound may also be a query
  parameter — `*1..$n`, `*$n`, `*$a..$b` — resolved per execution, so the same
  plan traverses to a depth chosen at call time. (`shortestPath` still requires a
  statically finite bound.)
- Inline label disjunction in node patterns: `(n:A|B)` matches a node carrying
  ANY of the listed labels (vs `(n:A:B)`, which still requires ALL). Works on a
  scanned node, on an expand target `(a)-[:R]->(b:A|B)`, and as a WHERE-position
  predicate; the two separators may not be mixed. (An OPTIONAL-MATCH target with
  `|` is not supported yet.)
- `EXISTS(…)` / `EXISTS { … }` in a `WITH … WHERE` or `RETURN`-position `WHERE`
  now hoists to a SemiApply like a `MATCH … WHERE` does, instead of failing at
  evaluation.
- `CALL { <subquery> }` — subquery blocks, both uncorrelated and correlated.
  Uncorrelated: a self-contained `RETURN`-terminated query whose result rows
  combine (cartesian) with the enclosing scope and whose `RETURN` columns become
  outer bindings — so `CALL { MATCH … RETURN count(*) AS c }` brings an aggregate
  into the outer query. Correlated (`CALL { WITH <bound vars> … }`): the leading
  `WITH` imports outer bindings and the subquery runs once per outer row (a
  lateral join via the new `Apply` operator), e.g. per-row neighbour expansion or
  top-N. The importing `WITH` must be a bare pass-through of bound variables.
  Subquery bodies expose only their RETURN columns to the outer scope; a body
  without a RETURN leaks nothing. A `UNION` / `UNION ALL` inside the block is
  supported for the uncorrelated form (`CALL { MATCH … RETURN x UNION MATCH …
  RETURN x }`). A correlated subquery body may also write — `MATCH (a) CALL { WITH
  a CREATE (a)-[:R]->(:X) }` runs the write once per outer row. (UNION in a
  correlated subquery is still unsupported.)
- `EXISTS { MATCH … [WHERE …] }` — the Neo4j 5 existential subquery form, in
  addition to the existing `EXISTS(pattern)` function. Correlated on outer
  bindings, supports an inner `WHERE` (and nested `EXISTS`), and `NOT EXISTS {…}`.
  Lowers to the same SemiApply operator, so it runs on both executor paths.
- `FOREACH (x IN list | <update clauses>)` — run side-effecting updates
  (CREATE / SET / MERGE / REMOVE / DELETE, and nested FOREACH) once per list
  element. It is a pass-through over its input rows (a clause after FOREACH keeps
  the same cardinality), so it covers both the bulk-write idiom
  `FOREACH (x IN $rows | CREATE …)` and per-matched-row updates. A read-modify-
  write on a node bound by the outer clause accumulates across iterations
  (`SET c.n = c.n + i`). The body is restricted to updating clauses (a read
  clause inside FOREACH is rejected at planning).
- `CREATE CONSTRAINT [name] FOR (n:Label) REQUIRE n.prop IS UNIQUE` (and the
  legacy `ON (n:Label) ASSERT …`) and `CREATE INDEX [name] FOR (n:Label) ON
  (n.prop)` (and legacy `ON :Label(prop)`) — schema DDL in Cypher for uniqueness
  constraints and secondary indexes, instead of only the programmatic schema
  API. A unique constraint validates existing data first (rejects creation with
  409-style "duplicate" if already violated) and then the write path enforces it
  on `CREATE`/`MERGE`/`SET`; a duplicate now returns **409 Conflict**. Both are
  always-on (no Cargo feature), intercepted on HTTP and Bolt, gated by the
  read-only/`AuthzHook` policy, and rejected inside a transaction.
- Cypher path functions `nodes(path)` and `relationships(path)`, and the list
  quantifier predicates `all`/`any`/`none`/`single(x IN list WHERE pred)`.
  Together they express intermediate-node filtering, e.g. a per-tenant guard
  `WHERE all(t IN [a.tenant, b.tenant] WHERE t = $t)`.
- `CALL search.bm25({label, text_property | text_properties, query, k?})` —
  full BM25 lexical search with **real IDF**. It scans the label's text
  property, builds corpus statistics (document count, average length, per-term
  document frequency) in one pass, then weights rare query terms above common
  ones — the signal the per-row `bm25(text, query)` scalar (IDF = 1.0) cannot
  see. Yields `node` + `score`, ordered by relevance. The MCP `hybrid_search`
  lexical channel now ranks through it, so hybrid results reflect true term
  rarity. The corpus-free scalar remains for inline use.
- `CREATE FULLTEXT INDEX <name> ON :Label(prop[, prop…])` — a **persistent
  inverted index** for BM25, behind the new `text-index` Cargo feature (default
  off; byte-identical when disabled). It is built during compaction (an inverted
  index of postings + corpus statistics, mirroring the vector index), and
  `CALL search.bm25` answers from it automatically when its `(label, properties)`
  match — touching only documents that contain the query terms instead of
  re-scanning the corpus — falling back to the flat scan otherwise. The index
  reflects the compacted corpus and is rebuilt on each compaction.

### Fixed

- Variable-length paths to a labelled far end now traverse through intermediate
  nodes of other labels: `(s:Service)-[:R*1..n]->(a:Algorithm)` previously
  returned empty whenever the intermediate hops were not themselves `Algorithm`
  (the far-end label filter pruned the frontier instead of just the result), and
  `shortestPath` to a labelled target shared the bug. The label now constrains
  results, not traversal.
- Path-binding plan analysis: a `WHERE`/`RETURN` over `p` from a variable-length
  `MATCH p = …` no longer mis-binds. The Expand's `path_binding` is now counted
  as a produced alias (so predicate pushdown keeps a `p`-referencing filter above
  it) and the factorised executor materialises the trail instead of dropping it.

## [1.0.0] - 2026-06-20: first stable release

First stable release. The on-disk format, the Cypher surface, the HTTP/Bolt
servers, and the embedded Python/Rust APIs are now covered by semantic
versioning.

### Added

- Graph algorithms expanded from 2 to 7 procedures, exposed identically
  through Cypher `CALL algo.*` and the MCP `graph_algorithm` tool:
  - `algo.degree` — in/out/total degree centrality.
  - `algo.scc` — strongly connected components (iterative Tarjan, no
    recursion limit on deep graphs).
  - `algo.triangle_count` — triangles per node plus the local clustering
    coefficient.
  - `algo.label_propagation` — community detection (asynchronous,
    deterministic label propagation).
  - `algo.shortest_path` — single-source shortest paths, BFS (hop count) by
    default or Dijkstra over non-negative `weight`s with `{weighted: true}`.
  Every kernel has a cancellable variant that honours the query deadline, so
  a heavy `CALL` on a large graph stays interruptible. Output rows and the
  MCP ranking/grouping use deterministic tie-breaks.

### Changed

- The project is now developed and released by NamiDB, Inc.

## [0.18.1] - 2026-06-20: ownership and durability

### Fixed

- s3b versioned-pointer forward probe is now fail-closed when its scan window
  is exhausted (a stale pointer is never served), and `bootstrap()` recovers a
  half-written namespace instead of wedging.

### Changed

- Ownership and licensing updated to NamiDB, Inc.

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

[Unreleased]: https://github.com/namidb/namidb/compare/v2.4.1...HEAD
[2.4.1]: https://github.com/namidb/namidb/compare/v2.4.0...v2.4.1
[2.4.0]: https://github.com/namidb/namidb/compare/v2.3.0...v2.4.0
[2.3.0]: https://github.com/namidb/namidb/compare/v2.2.1...v2.3.0
[2.2.1]: https://github.com/namidb/namidb/compare/v2.2.0...v2.2.1
[2.2.0]: https://github.com/namidb/namidb/compare/v2.1.4...v2.2.0
[2.1.4]: https://github.com/namidb/namidb/compare/v2.1.3...v2.1.4
[2.1.3]: https://github.com/namidb/namidb/compare/v2.1.2...v2.1.3
[2.1.2]: https://github.com/namidb/namidb/compare/v2.1.1...v2.1.2
[2.1.1]: https://github.com/namidb/namidb/compare/v2.1.0...v2.1.1
[2.1.0]: https://github.com/namidb/namidb/compare/v2.0.6...v2.1.0
[2.0.6]: https://github.com/namidb/namidb/compare/v2.0.5...v2.0.6
[2.0.5]: https://github.com/namidb/namidb/compare/v2.0.4...v2.0.5
[2.0.4]: https://github.com/namidb/namidb/compare/v2.0.3...v2.0.4
[2.0.3]: https://github.com/namidb/namidb/compare/v2.0.2...v2.0.3
[2.0.2]: https://github.com/namidb/namidb/compare/v2.0.1...v2.0.2
[2.0.1]: https://github.com/namidb/namidb/compare/v2.0.0...v2.0.1
[2.0.0]: https://github.com/namidb/namidb/releases/tag/v2.0.0
[0.13.0]: https://github.com/namidb/namidb/releases/tag/v0.13.0
[0.4.1]: https://github.com/namidb/namidb/releases/tag/v0.4.1
[0.4.0]: https://github.com/namidb/namidb/releases/tag/v0.4.0
[0.3.0]: https://github.com/namidb/namidb/releases/tag/v0.3.0
[0.2.1]: https://github.com/namidb/namidb/releases/tag/v0.2.1
[0.2.0]: https://github.com/namidb/namidb/releases/tag/v0.2.0
[0.1.0]: https://github.com/namidb/namidb/releases/tag/v0.1.0
