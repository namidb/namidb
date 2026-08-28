# namidb-server

An HTTP server that exposes a NamiDB namespace over a small REST API.
It's the same engine as the embedded library; all this binary adds is
the HTTP boundary, bearer-token auth, and a periodic flush loop.

## Install

From source (workspace root). Note the feature flags: vector and
full-text search are compile-time features that the official binaries,
image, and wheels all enable, but a bare `cargo install` does NOT — a
from-source build without them rejects `CREATE VECTOR INDEX` /
`CREATE FULLTEXT INDEX` with HTTP 400:

```bash
cargo install --path crates/namidb-server --features vector-index,text-index
```

Container image (official, multi-arch amd64/arm64, from
[Docker Hub](https://hub.docker.com/r/namidb/namidb-server)):

```bash
docker pull namidb/namidb-server:2
```

The image runs as non-root (uid 65532) and pre-creates `/var/lib/namidb`
(the canonical `file://` store path), so a fresh **named volume** mounted
there inherits the right ownership and just works. A **bind mount** — or a
named volume created root-owned by an image older than 2.2.0 — needs a
one-time `chown -R 65532:65532 <dir>` on the host first.

Or build it yourself from the repo root:

```bash
docker build -t namidb/namidb-server:2 -f crates/namidb-server/Dockerfile .
```

## Run

```bash
namidb-server \
  --store "s3://my-bucket/data?ns=prod&region=us-east-1" \
  --listen 0.0.0.0:8080 \
  --auth-token "$NAMIDB_AUTH_TOKEN" \
  --flush-interval 30s
```

Every flag can also be set as an env var (`NAMIDB_STORE`,
`NAMIDB_LISTEN`, `NAMIDB_AUTH_TOKEN`, `NAMIDB_FLUSH_INTERVAL`). The
`--store` URI follows the same scheme grammar as the Python client and
the CLI, see [`namidb-storage/src/uri.rs`](../namidb-storage/src/uri.rs).

For shared machines, `--memory-max-bytes` /
`NAMIDB_MEMORY_MAX_BYTES` sets an exact-byte RSS/working-set admission
ceiling (`0` disables it). At 90% the server clears shared caches and weakly
registered, reconstructible writer property/constraint maps. A process-wide
500 ms watchdog performs the same single-flight reclamation even when no
request arrives; at the ceiling new Cypher work returns HTTP 503 or Bolt
`Neo.TransientError.General.DatabaseUnavailable` until memory falls. Pair it
with Docker `--memory` or another cgroup/OS limit for hard containment of an
already-running query. Authenticated `POST /v0/admin/flush` bypasses Cypher
admission and is serialized process-wide so an operator can drain a live
memtable. It is excluded from the ordinary request timeout and its storage
work survives a disconnected client; the global request cap still bounds
waiting callers. Under hard pressure
multi-tenant mode refuses to open a cold namespace merely to flush it.

Flush and compaction spool node Parquet outputs, corpus-sized exact-node values
and remote compaction inputs to `NAMIDB_SPOOL_DIR` instead of retaining them in
RSS. The bare binary defaults to disk-backed `/var/tmp` on Unix (the native
temp directory elsewhere); the official image uses
`/var/tmp/namidb-spool`. During full-backlog node compaction, provision
`sum(inputs) + parquet_output + exact_record_output` — roughly 3× the
compacted live node bytes (about 12–15 GiB for one million 1024d nodes), plus
headroom for superseded versions. Spools are synced before mmap/upload, and
flush builds remain process-wide single-flight after caller cancellation.
Avoid a RAM-backed `/tmp`/`tmpfs`.

Large `.vg` and `.ft` bodies use one shared decoded-index eviction pool.
`NAMIDB_SEARCH_INDEX_CACHE_MAX_BYTES` optionally reserves exact bytes for that
pool inside `NAMIDB_CACHE_MAX_BYTES`; the remaining cache tiers are fitted into
the remainder. If a valid index's conservative decoded estimate cannot fit,
HTTP returns retryable 503 with `code: "search_index_cache_capacity"` and Bolt
returns `Neo.TransientError.General.DatabaseUnavailable`, rather than silently
running an O(corpus) flat scan. The error reports required and assigned bytes.
Missing/stale/corrupt index generations still use the exact correctness
fallback.

Current forward edge SSTs also build an optional exact relationship sidecar
for bound-endpoint `MERGE`. `NAMIDB_EDGE_POINT_MAX_ENTRY_BYTES` (64 KiB) and
`NAMIDB_EDGE_POINT_MAX_SST_BYTES` (512 MiB) bound its per-record and per-SST
footprint; setting either to `0`, crossing either limit, or losing/corrupting
the optional object selects the authoritative CSR path.

Filtered vector indexes keep bounded String/Bool ordinal postings inside each
`.vg`. Tune `NAMIDB_VECTOR_FILTER_MAX_DISTINCT` (default 4096 values/property)
and `NAMIDB_VECTOR_FILTER_MAX_BYTES` (default 64 MiB/body); crossing a bound
omits the whole property rather than writing an incomplete posting. A legacy or
omitted-property query may lazily use at most
`NAMIDB_VECTOR_FILTER_ID_CANDIDATE_CAP` complete sidecar IDs (default 8192,
`0` disables) before retaining the exact scan fallback.

Filtered `search.hybrid` widens its authoritative BM25 prefix up to
`NAMIDB_HYBRID_TEXT_FILTER_CANDIDATE_CAP` (default 65536) and applies the
predicate before sparse top-k. Reaching the cap switches to the exact flat
scorer, so the cap bounds indexed hydration without reducing recall.

Manifest/pointer versions are append-only per durable commit and are retained
temporarily for pinned readers and crash-safe CAS. The maintenance janitor
reclaims versions below the live-reader horizon after
`NAMIDB_SWEEP_MIN_AGE` (24 h by default) only while
`NAMIDB_COMPACTION_INTERVAL` is non-zero and `NAMIDB_SWEEP_DELETE=true`
(otherwise maintenance is disabled or the sweep is dry-run). Thus
object-count growth observed during the first hours of a bulk load is bounded
history under the default maintenance settings.

If you don't configure any auth source, the server **refuses to start**
(since 2.2.0). Pass `--no-auth` / `NAMIDB_NO_AUTH=1` to explicitly boot in
unauthenticated mode for local development — it prints a loud warning, and
every request gets anonymous read-write access. Don't expose that port to
the public internet.

## Security & auth

A bearer token is **required by default** — configure one of the schemes
below, or opt out explicitly with `--no-auth`. All of them resolve a bearer
token to a role (read-only vs read-write) through one path, so HTTP and Bolt
resolve the same role. Namespace *scoping* is enforced on both protocols:
HTTP checks the routed namespace in its auth middleware, and multi-tenant
Bolt (since 2.3.0) re-resolves the token against the namespace each
statement routes to.

| Scheme | Flags | Notes |
|---|---|---|
| **Static token** | `--auth-token` (or `NAMIDB_AUTH_TOKEN`) | Single read-write token. |
| **Static token file** | `--auth-tokens-file` | Per-token roles **and** per-namespace scoping; hand out read-only tokens or tokens scoped to a namespace set. Takes precedence over `--auth-token`. |
| **OIDC / JWT** | `--jwt-jwks-url` (enables), `--jwt-issuer`, `--jwt-audience`, `--jwt-groups-claim` (default `groups`), `--jwt-write-group`, `--jwt-read-group`, `--jwt-namespaces-claim` | Verifies bearer tokens against a JWKS URL (RS/ES* sig, `exp`, optional `iss`/`aud`), maps a group claim → role and a claim → namespace scope. Requires building with `--features jwt`. Fail-closed: a validation failure is a 401; an unreachable JWKS at boot aborts startup. |
| **External policy (PDP)** | `--pdp-url` | POSTs `{subject, role, groups, action, …}` (or a schema op) to an OPA-style endpoint and denies unless it allows. **Fail-closed** on any error. Requires `--features pdp`. Can deny even reads, and gates DDL via `check_schema`. |

TLS: pass `--tls-cert` and `--tls-key` (PEM) to serve HTTPS (and Bolt over TLS);
omit them for plaintext (terminate TLS at a proxy/mesh instead). The Bolt
listener shares the same token and TLS config as HTTP.

The official container's liveness probe follows `NAMIDB_LISTEN` and switches
to HTTPS when `NAMIDB_TLS_CERT` or `NAMIDB_TLS_KEY` is set. Its loopback-only
HTTPS request deliberately skips certificate verification because deployment
certificates commonly omit loopback addresses. When overriding listen or TLS
with container command-line flags, mirror the same values in those environment
variables so the probe sees the effective endpoint, or explicitly replace the
image healthcheck in your orchestrator.

`jwt` and `pdp` are optional Cargo features (default off → the build is
byte-identical to static-token-only). Build the server with, e.g.,
`cargo build -p namidb-server --features jwt,pdp` to enable them.

## Endpoints (v0)

| Method | Path | Auth | Description |
|---|---|---|---|
| `GET`  | `/v0/livez`        | public  | Lock-free liveness (process is up) |
| `GET`  | `/v0/health`       | public  | Readiness + manifest version + epoch + writer status (503 while the writer is degraded) |
| `GET`  | `/v0/version`      | public  | Server build version |
| `GET`  | `/v0/metrics`      | public  | Prometheus metrics (text exposition) |
| `POST` | `/v0/cypher`       | bearer  | Run a Cypher query (read or write) |
| `POST` | `/v0/admin/flush`  | bearer  | Force a memtable -> L0 SST flush; remains available and globally serialized under RSS pressure |
| `POST` | `/v0/admin/backup` | bearer (read-write) | Copy a point-in-time snapshot of the namespace to an allowlisted destination (disabled unless `--backup-target-uri` is set; multi-tenant: `/:namespace/v0/admin/backup`) |

### `POST /v0/admin/backup`

Live-safe by design: the copy pins a manifest version with a durable
retention lease the janitor honours, so no writer pause is needed and the
writer lock is never taken. The result is a self-contained, openable
namespace at the destination — point-in-time at the pinned version
(committed-but-unflushed WAL included; writes after the pin excluded).

```json
{ "to": "s3://backups-bucket/namidb/nightly?ns=prod", "force": false, "verify": true }
```

Optional fields: `version` (pin an older manifest version), `force`
(overwrite an existing destination pointer), `verify` (re-read and checksum
every copied object). Returns `{"source_version", "objects_copied",
"bytes_copied"}`.

**The destination allowlist is the security boundary.** The server's
ambient cloud credentials write wherever `to` points, so the endpoint is
disabled (403) until the operator sets `--backup-target-uri` /
`NAMIDB_BACKUP_TARGET_URI`; `to` must equal that prefix or extend it at a
`/` (or `?ns=`) boundary. One backup runs at a time — process-wide in
multi-tenant mode, where `/:namespace/v0/admin/backup` resolves the
namespace's committed state without opening it (cold namespaces stay
cold) and namespace-scoped tokens are enforced by the auth middleware —
with waiters getting a 503 after 30 s. Restore stays CLI-only on
purpose: it requires an offline destination, and the serving process
*is* the writer.

### `POST /v0/cypher`

Request:

```json
{
  "query": "MATCH (p:Person) WHERE p.age >= $min RETURN p.name AS name",
  "params": {"min": 18}
}
```

Response (read):

```json
{
  "columns": ["name"],
  "rows": [{"name": "Alice"}, {"name": "Bob"}]
}
```

Response (write):

```json
{
  "columns": ["a"],
  "rows": [{"a": {"_kind": "node", "id": "...", "label": "Person", "properties": {}}}],
  "write_outcome": {
    "nodes_created": 1,
    "edges_created": 0,
    "nodes_deleted": 0,
    "edges_deleted": 0,
    "properties_set": 0
  }
}
```

A `curl` round-trip:

```bash
TOKEN=$(openssl rand -hex 32)

namidb-server --store memory://demo --listen 127.0.0.1:8080 --auth-token "$TOKEN" &

curl -s http://127.0.0.1:8080/v0/health | jq .

curl -s -X POST http://127.0.0.1:8080/v0/cypher \
  -H "Authorization: Bearer $TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{"query": "CREATE (a:Person {name: \"Alice\", age: 30}) RETURN a.name AS name"}' \
  | jq .

curl -s -X POST http://127.0.0.1:8080/v0/cypher \
  -H "Authorization: Bearer $TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{"query": "MATCH (p:Person) RETURN p.name AS name, p.age AS age"}' \
  | jq .
```

### Error taxonomy

Failed statements return a JSON body with machine-readable fields (since
2.2.0) so a client can decide "fix the query" vs "back off" vs "retry"
without string-matching the message:

```json
{
  "error": "read execution failed: query exceeded the configured timeout",
  "code": "timeout",
  "neo4j_code": "Neo.ClientError.Transaction.TransactionTimedOut",
  "gql_status": "57014"
}
```

| Condition | HTTP | `code` | `neo4j_code` | `gql_status` |
|---|---|---|---|---|
| Syntax error | 400 | `parse_error` | `Neo.ClientError.Statement.SyntaxError` | `42001` |
| Semantic/plan error | 400 | `plan_error` | `Neo.ClientError.Statement.SemanticError` | `42000` |
| Feature not supported | 400 | `unsupported` | `Neo.ClientError.Statement.NotSupported` | `0A000` |
| Runtime evaluation error (division by zero, missing `$param`, type mismatch) | 400 | `eval_error` | `Neo.ClientError.Statement.ArgumentError` | `22000` |
| Unique-constraint violation | 409 | `constraint` | `Neo.ClientError.Schema.ConstraintValidationFailed` | `23000` |
| Query timeout | 504 | `timeout` | `Neo.ClientError.Transaction.TransactionTimedOut` | `57014` |
| Row cap / search result caps | 413 | `row_cap`, `search_result_limit`, `search_document_limit` | `Neo.ClientError.Statement.ResourceLimitExceeded` | `54000` |
| Transient resource pressure | 503 | `search_index_cache_capacity`, `search_workspace_capacity` | `Neo.TransientError.General.DatabaseUnavailable` | `53000` |
| Unclassified server fault | 500 | — | `Neo.DatabaseError.General.UnknownError` | `50N42` |

The same classes drive the Bolt `FAILURE` code, so official Neo4j drivers —
which auto-retry `Neo.TransientError.*` and nothing else — retry exactly the
transient conditions. Deterministic budgets (timeout, row cap) are
`ClientError` **on purpose**: a re-run exceeds the same budget again, so
auto-retrying them is a retry livelock. Bolt ≤ 5.4 has no GQLSTATUS on the
wire (GQL-aware drivers show the `50N42` polyfill for every failure); the
`gql_status` field in the HTTP body is the per-family status.

## Type mapping (JSON and Cypher)

| Cypher `RuntimeValue` | JSON |
|---|---|
| `Null`                | `null` |
| `Bool`                | `true` / `false` |
| `Integer`             | number (i64) |
| `Float`               | number (f64) |
| `String`              | string |
| `Bytes`               | base64 string |
| `Vector(f32)`         | array of numbers |
| `List`                | array |
| `Map`                 | object |
| `Date`                | ISO-8601 date string |
| `DateTime` (UTC, microseconds) | RFC-3339 timestamp string |
| `Node`                | `{"_kind": "node", "id", "label", "properties"}` |
| `Rel`                 | `{"_kind": "rel", "edge_type", "src", "dst", "properties"}` |
| `Path`                | array of alternating node/rel objects |

## Group commit (RFC-034)

`--group-commit-window` / `NAMIDB_GROUP_COMMIT_WINDOW` (default `0s` =
disabled) coalesces concurrently arriving auto-commit writes into one WAL
append + one manifest CAS: requests stage under the writer lock, register a
waiter, and are ACKed only after the group's single commit is durable and
the snapshot republished — read-your-own-writes is preserved, and a
statement failure rolls back alone. With the window at zero every write
commits inline, exactly as before. On real object storage (5-15 ms per
commit round-trip same-region) a `2ms`-`5ms` window lifts the interactive
write floor from ~1/commit-RTT to roughly group-size/commit-RTT; grouped
writes share fate on a commit failure (one merged WAL segment cannot be
partially durable). Single-tenant only for now.

## Concurrency model

`namidb-server` opens one `WriterSession` per process and serialises
every request behind a tokio `Mutex`. That's the single-writer-per-
namespace invariant from RFC-001, lifted up to the request layer: at
most one Cypher statement is in flight against the namespace at a time.
Read latency stays predictable, and throughput is bounded by the slowest
mutator. Concurrent read fan-out without holding the writer mutex is
RFC-021 work.

If you need horizontal read scale today, point several `namidb-server`
processes at the same `--store` URI. Each one serves reads off the same
manifest version, and only one is allowed to commit writes (the rest get
fenced via epoch CAS).

## Periodic flush

`--flush-interval` (default `30s`) controls how often the background
task turns the memtable into L0 SSTs. Set it to `0s` to disable the loop
and call `POST /v0/admin/flush` from cron or a sidecar instead. Manual flush is
also the authenticated pressure-relief escape hatch when new Cypher admission
is closed; its SST build still needs transient headroom.

## Metrics and the slow-query log

`GET /v0/metrics` renders the process query metrics in the Prometheus
text exposition format. It is unauthenticated, like `/v0/livez` and
`/v0/health`, so a scraper needs no bearer token. When TLS is on it is
served over HTTPS on the same listener.

```bash
curl -s http://127.0.0.1:8080/v0/metrics
```

| Metric | Type | Labels | What it is |
|---|---|---|---|
| `namidb_queries_total`          | counter   | `protocol`, `status` | Queries executed, by `http`/`bolt` and `ok`/`error` |
| `namidb_query_duration_seconds` | histogram | `protocol`, `kind`   | Execution wall-clock, by `http`/`bolt` and `read`/`write` |
| `namidb_queries_in_flight`      | gauge     |                      | Queries currently executing |
| `namidb_cache_max_bytes`        | gauge     |                      | Configured aggregate cache ceiling |
| `namidb_memory_max_bytes`       | gauge     |                      | Configured process RSS/working-set admission ceiling |
| `namidb_memory_resident_bytes`  | gauge     |                      | Most recently sampled process RSS/working set |
| `namidb_memory_reclaims_total`  | counter   |                      | Cache pressure-relief passes |
| `namidb_memory_rejected_queries_total` | counter |                | Queries rejected at the memory ceiling |
| `namidb_cache_capacity_bytes`   | gauge     |                      | Capacity assigned to enabled cache tiers |
| `namidb_cache_resident_bytes`   | gauge     |                      | Cache-accounted bytes currently resident |
| `namidb_search_index_cache_capacity_bytes` | gauge |              | Capacity assigned to the shared decoded `.vg`/`.ft` pool |
| `namidb_search_index_cache_admission_rejections_total` | counter | `kind` | Valid vector/text indexes rejected by the configured pool |
| `namidb_vector_filter_bitmap_searches_total` | counter |             | Vector queries that applied embedded `.vg` metadata postings (vector-enabled builds) |
| `namidb_slow_queries_total`     | counter   |                      | Queries that crossed the slow-query threshold |
| `namidb_build_info`             | gauge     | `version`            | Always `1`; carries the build version |
| `namidb_uptime_seconds`         | gauge     |                      | Seconds since the server started |

Duration is measured per query and stops at the end of execution, so the
optional write-stall backpressure sleep is not counted as query latency.
Bolt schema-introspection probes (the `CALL` / `SHOW` calls GUIs issue)
are not counted as queries.

The **slow-query log** is separate from the metrics and controlled by
`--slow-query-threshold` (env `NAMIDB_SLOW_QUERY_THRESHOLD`, default
`1s`, set `0s` to disable). Any query at or above that wall-clock is
logged at `WARN`:

```
WARN slow query protocol="http" kind="read" status="ok" elapsed_ms=1840 query="MATCH (a:Person)-[:KNOWS*2]-(b) RETURN count(b)"
```

The statement text is logged truncated; parameters are never logged,
since they can carry sensitive values. The statement text itself is, so
a value inlined as a literal in the Cypher source (rather than passed as
a `$param`) does land in the log, the same as any SQL slow-query log.
Parameterise sensitive values to keep them out of it.

## Bolt protocol

Pass `--bolt-listen 0.0.0.0:7687` (or `NAMIDB_BOLT_LISTEN`) to expose
a Bolt 4.4 / 5.0 / 5.4 listener alongside the HTTP API. Both protocols
share the same `WriterSession`, the same auth token, and the same
single-writer-per-namespace invariant.

In single-tenant mode the listener serves the `?ns=` of `--store`. Under
`--multi-tenant` (since 2.3.0), statements route to the namespace named by
the Bolt `db` field — `session(database="acme")` in the official drivers —
falling back to `--default-namespace` when unset. An explicit transaction
is pinned to the database named at BEGIN, and a namespace-scoped token is
rejected (`Neo.ClientError.Security.Forbidden`) outside its scope, exactly
like the HTTP middleware.

Authenticated Bolt messages are capped at 64 MiB by default. Override the
exact-byte ceiling with `--bolt-max-message-bytes` or
`NAMIDB_BOLT_MAX_MESSAGE_BYTES`; the unauthenticated handshake and LOGON path
always remains capped at 64 KiB. An oversized authenticated request receives a
`Neo.ClientError.Request.Invalid` diagnostic and its connection closes cleanly,
so clients should split very large parameter sets into batches.

Authenticated data frames also share one process-wide working-set admission
budget before allocation and PackStream decode. It defaults to half of
`NAMIDB_MEMORY_MAX_BYTES` when that RSS governor is enabled, otherwise
`1073807360` bytes (~1 GiB), and can be set explicitly with
`NAMIDB_BOLT_MEMORY_BUDGET_BYTES`. An incomplete frame holds
`64 KiB + 2 × wire body bytes`; after its terminator, a data frame atomically
upgrades the same fail-fast lease to `64 KiB + 16 × wire body bytes` through
decode, parameter conversion, execution and any RUN prefetch. The server also
atomically reserves measured RSS headroom, including concurrent reservations,
before decode and retains that RAII guard through request handling; normal
admission repeats at execution time. `NAMIDB_BOLT_PARTIAL_MESSAGE_TIMEOUT`
(default `120s`, `0s` disables) bounds a frame from its first byte; completely
idle authenticated connections hold no message permit. Small PULL, DISCARD,
COMMIT, ROLLBACK, RESET, GOODBYE and LOGOFF frames remain available under
pressure.

```bash
namidb-server \
  --store memory://demo \
  --listen 0.0.0.0:8080 \
  --bolt-listen 0.0.0.0:7687 \
  --auth-token "$NAMIDB_AUTH_TOKEN"
```

```python
from neo4j import GraphDatabase
driver = GraphDatabase.driver("bolt://localhost:7687",
                              auth=("namidb", "$NAMIDB_AUTH_TOKEN"))
with driver.session() as s:
    s.run("CREATE (:Person {name: 'Alice'})")
    for r in s.run("MATCH (p:Person) RETURN p.name AS name"):
        print(r["name"])
```

See [RFC-022](../../docs/rfc/022-bolt-protocol.md) for the wire-level
design.

## Roadmap

- `/v0/cypher/stream`: NDJSON streaming for large read result sets.
- `/v0/cypher/arrow`: an Arrow IPC body for zero-copy DataFrame
  ingestion.
- Cache hit-rate gauges on `/v0/metrics` (adjacency, node, SST caches).

See the project [`README`](../../README.md) and [`docs/rfc/`](../../docs/rfc/)
for engine internals.

## License

[Business Source License 1.1](../../LICENSE), © NamiDB, Inc.
