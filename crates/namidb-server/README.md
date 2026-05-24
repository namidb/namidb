# namidb-server

An HTTP server that exposes a NamiDB namespace over a small REST API.
It's the same engine as the embedded library; all this binary adds is
the HTTP boundary, bearer-token auth, and a periodic flush loop.

## Install

From source (workspace root):

```bash
cargo install --path crates/namidb-server
```

Container image (build from the repo root):

```bash
docker build -t namidb-server:0.1 -f crates/namidb-server/Dockerfile .
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

If you don't set `--auth-token`, the server boots in **unauthenticated**
mode and prints a loud warning. Don't expose that port to the public
internet.

## Endpoints (v0)

| Method | Path | Auth | Description |
|---|---|---|---|
| `GET`  | `/v0/health`       | public  | Liveness + manifest version + epoch |
| `GET`  | `/v0/version`      | public  | Server build version |
| `POST` | `/v0/cypher`       | bearer  | Run a Cypher query (read or write) |
| `POST` | `/v0/admin/flush`  | bearer  | Force a memtable -> L0 SST flush |

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
and call `POST /v0/admin/flush` from cron or a sidecar instead.

## Bolt protocol

Pass `--bolt-listen 0.0.0.0:7687` (or `NAMIDB_BOLT_LISTEN`) to expose
a Bolt 4.4 / 5.0 / 5.4 listener alongside the HTTP API. Both protocols
share the same `WriterSession`, the same auth token, and the same
single-writer-per-namespace invariant.

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
- `/v0/metrics`: Prometheus exposition (counters, latency histogram,
  cache hit rates).

See the project [`README`](../../README.md) and [`docs/rfc/`](../../docs/rfc/)
for engine internals.

## License

[Business Source License 1.1](../../LICENSE), © LESAI, Corp.
