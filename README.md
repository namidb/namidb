<div align="center">

<p>
  <img src=".assets/namidb-logo.jpeg" alt="NamiDB" width="640" />
</p>

# NamiDB

### A graph database that lives in your S3 bucket.

It embeds like DuckDB, runs as a standalone HTTP server, or sits on our hosted cloud. Same engine in all three, and the bucket is always the source of truth.

[![License: BSL 1.1](https://img.shields.io/badge/License-BSL%201.1-1f6feb.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/Rust-1.85%2B-dea584.svg?logo=rust&logoColor=white)](https://www.rust-lang.org)
[![PyPI](https://img.shields.io/badge/PyPI-namidb-3776ab.svg?logo=pypi&logoColor=white)](https://pypi.org/project/namidb/)
[![Docker](https://img.shields.io/badge/Docker-namidb--server-2496ed.svg?logo=docker&logoColor=white)](crates/namidb-server/Dockerfile)
[![Website](https://img.shields.io/badge/Website-namidb.com-0a7ea4.svg)](https://namidb.com)
[![Docs](https://img.shields.io/badge/Docs-docs.namidb.com-0a7ea4.svg)](https://docs.namidb.com)

[**Website**](https://namidb.com) · [**Documentation**](https://docs.namidb.com) · [**RFCs**](./docs/rfc/) · [**Request early access**](https://namidb.com)

</div>

---

NamiDB is a graph database engine built around object storage. You write Cypher, it lays your nodes and edges out as columnar files in a bucket, and that bucket is the only source of truth. There's nothing else to run and nothing to coordinate outside the bucket itself. The same engine ships three ways: embedded as a library, as an HTTP server, or on our hosted cloud.

<br />

## Why now

A few things had to line up before this made sense.

**S3 finally got conditional writes.** In 2024, S3 shipped `If-Match` and `If-None-Match`, which was the last primitive we were missing. With compare-and-swap on the bucket you can build a coordinated, durable system where object storage *is* the database. There's no Raft, no ZooKeeper, no etcd in the picture. People had already pulled this off for vectors, for queues, for analytics. Nobody had done it for graphs.

**The best columnar graph engine left the field.** Apple bought Kùzu in October 2025 and archived the repo. It was the most carefully thought-out columnar graph engine anyone had published, and it just went quiet. That left a hole.

**Agents need graphs.** Vector search is necessary but it isn't enough. Knowledge graphs are what agent memory, deep retrieval, and reasoning under uncertainty actually sit on once you're past the demo. A lot of the interesting AI work this decade is going to be about relationships, not just embeddings.

So that's what we're building.

<br />

## Where this came from

NamiDB started inside LESAI as the graph database behind a hosted product we're building. We've been at it for about a year now, and every Cypher query, every manifest CAS, every CSR adjacency table in here has been run against real workloads, not just unit tests.

We're open-sourcing the engine now because two things finally lined up:

1. Apple archived Kùzu in October 2025, so the columnar property-graph space lost its one maintained option. We'd independently landed on more or less the design Kùzu pioneered, so putting NamiDB out there felt like the most useful thing we could do about that gap.
2. Our own roadmap moved to a hosted product, [NamiDB Cloud](https://namidb.com), which is multi-tenant and scales to zero per namespace. The engine doesn't need to be a competitive secret anymore. The engine is open, the cloud is the business.

<br />

## The shape

**NamiDB writes Cypher to your S3 bucket.**

There's no control plane to provision, no Raft to tune, no etcd to babysit. Conditional writes (`If-Match` / `If-None-Match`) on the bucket take the place of a consensus tier, so the bucket itself holds the truth. Your graph database is just files: durability is whatever S3, R2, GCS or Azure already give you, cost drops to zero when nobody is querying, a backup is `aws s3 sync`, and a tenant is a folder.

The engine is the same whether you run it as a library inside your app, as a Rust daemon over HTTP, or on our hosted cloud. It works just as well against AWS S3, Cloudflare R2, GCS, Azure Blob, MinIO, or your local disk.

<br />

## Three deployments, one engine

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset=".assets/namidb-deployments-dark.svg" />
    <img src=".assets/namidb-deployments.svg" alt="NamiDB deployments: Server, Embedded, Cloud, converging on a single object-storage bucket" width="900" />
  </picture>
</p>

| Mode | Status | Best for | How it ships |
|---|---|---|---|
| **Server** | ✅ v0.1 | **Self-hosted production over your S3 / R2 / GCS / Azure bucket** | `namidb-server` binary + Docker image |
| **Embedded** | ✅ v0.1 | Notebooks, single-process apps, local dev, CI fixtures | `pip install namidb`, talks to a bucket from inside your process |
| **Cloud** | 🔒 closed beta | Multi-tenant SaaS, agent memory, scale-to-zero per namespace | Managed by LESAI on namidb.com, [request access](https://namidb.com) |

It's the same engine across all three. Server and Embedded write to an identical bucket layout, so you can open an embedded notebook against the exact `s3://...` URI a production daemon is serving.

<br />

## What's in the engine today

- **Cypher and GQL parsing.** A strict subset of GQL (ISO/IEC 39075:2024) plus openCypher 9. All 12 in-scope LDBC SNB Interactive Complex Read queries (IC01 through IC12) parse, plan and run end to end.
- **Writes through Cypher.** `CREATE`, `MERGE`, `SET`, `DELETE`, `DETACH DELETE`, `REMOVE`. Durable on `commit_batch` (WAL append plus manifest CAS).
- **Cost-based optimizer.** Predicate pushdown, projection pushdown, join reorder, hash-join conversion, hash semi-join (`EXISTS` decorrelation), Parquet row-group pruning. `EXPLAIN VERBOSE` prints the chosen plan with selectivity and cost annotations.
- **Vectorized execution.** A morsel-driven executor with an optional factorized intermediate representation (RFC-017) for path-heavy queries.
- **Columnar storage on object storage.** Parquet node SSTs, a custom edge-SST format with CSR adjacency (RFC-002), zstd compression, bloom filters, fence-pointer indices.
- **Coordination-free correctness.** One writer per namespace, with epoch fencing via manifest CAS. Conditional writes (`If-Match`, `If-None-Match`) stand in for external consensus.
- **Tiered caches.** A process-wide `AdjacencyCache` (CSR), a `NodeViewCache`, and an `SstCache` (decoded body, edge property streams, and the reader). Cross-snapshot reuse, `Arc`-shared and byte-budgeted.
- **Six storage backends.** `memory://`, `file://` (with `flock`-based CAS), `s3://` (AWS S3, R2, MinIO, Tigris, LocalStack), `gs://`, `az://`.
- **Python bindings.** `pip install namidb`. abi3 wheels for Linux (x86_64 and aarch64), macOS (arm64) and Windows (x86_64), with an sdist fallback everywhere else. Sync and async (`acypher`). Arrow, pandas and polars output.
- **CLI.** `namidb parse`, `namidb explain --verbose`, `namidb run --store <uri>` for ad-hoc query work against any backend.
- **HTTP server.** The `namidb-server` binary, with bearer-token auth, a periodic flush loop, and a small REST API (`/v0/cypher`, `/v0/health`, `/v0/admin/flush`).
- **Bench harness.** A synthetic, deterministic LDBC SNB Interactive harness under [`bench/`](./bench/).

<br />

## Quickstart

Two ways in. Same engine behind both.

### Door 1: a real graph database in your S3 bucket

This is the headline use case. Point it at a bucket, write Cypher, and durability is whatever S3 already gives you.

```bash
pip install namidb
export AWS_ACCESS_KEY_ID=AKIA...
export AWS_SECRET_ACCESS_KEY=...
```

```python
import namidb

# Open (or bootstrap) the `prod` namespace on your bucket.
client = namidb.Client("s3://my-bucket/data?ns=prod&region=us-east-1")

client.cypher("CREATE (a:Person {name: 'Alice', age: 30})")
client.cypher("CREATE (b:Person {name: 'Bob',   age: 25})")
client.cypher(
    "MATCH (a:Person {name: 'Alice'}), (b:Person {name: 'Bob'}) "
    "CREATE (a)-[:KNOWS {since: 2020}]->(b)"
)

result = client.cypher(
    "MATCH (p:Person) WHERE p.age >= $min RETURN p.name AS name, p.age AS age",
    params={"min": 18},
)
print(result.to_pandas())
```

Kill the process and start it again. Open a notebook on another machine pointed at the same URI. The graph is still there, because the bucket is the database.

### Door 2: a 30-second taste, no credentials

For when you just want to poke at the engine before wiring up a bucket. In-process, ephemeral, zero setup:

```python
import namidb
client = namidb.Client("memory://acme")
client.cypher("CREATE (a:Person {name: 'Alice'})")
print(client.cypher("MATCH (p:Person) RETURN p.name").rows())
```

The same handful of lines work against `file://`, `gs://`, `az://`, or any S3-compatible endpoint. Only the URI changes.

<br />

## Pick your storage backend

The URI tells the client which bucket and which namespace to use.

| Scheme | Backend |
|---|---|
| `s3://<bucket>[/<prefix>]?ns=<ns>` | **AWS S3, Cloudflare R2, MinIO, Tigris, LocalStack, anything S3-compatible** |
| `gs://<bucket>?ns=<ns>` | Google Cloud Storage |
| `az://<account>/<container>?ns=<ns>` | Azure Blob Storage |
| `file:///abs/dir?ns=<ns>` | Local filesystem (CAS via `flock` + atomic rename) |
| `memory://<ns>` | In-process and ephemeral, for testing only |

Every backend speaks the same Cypher, exposes the same Python, Rust and HTTP APIs, and gives you the same snapshot-isolated reads.

### AWS S3 (the primary path)

```python
import os
os.environ["AWS_ACCESS_KEY_ID"]     = "AKIA..."
os.environ["AWS_SECRET_ACCESS_KEY"] = "..."

client = namidb.Client(
    "s3://my-bucket/data?ns=prod"
    "&region=us-west-2"
)
```

Credentials come from the standard AWS env vars (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_SESSION_TOKEN`, `AWS_DEFAULT_REGION`). IAM roles on EC2, EKS, Lambda and ECS just work, with no NamiDB-specific auth to wire up.

The only IAM permissions NamiDB needs on the bucket are `s3:GetObject`, `s3:PutObject`, `s3:DeleteObject` and `s3:ListBucket`. That's it. No DynamoDB lock table, no separate metadata service.

### Cloudflare R2 (no egress fees)

R2 charges nothing for egress and has full S3-compatible conditional writes. Same scheme, just point at the R2 endpoint with `region=auto`:

```python
import os
os.environ["AWS_ACCESS_KEY_ID"]     = "<R2 access key>"
os.environ["AWS_SECRET_ACCESS_KEY"] = "<R2 secret>"

client = namidb.Client(
    "s3://my-bucket?ns=prod"
    "&endpoint=https://<ACCOUNT_ID>.r2.cloudflarestorage.com"
    "&region=auto"
)
```

If you're running NamiDB anywhere outside AWS (Cloudflare Workers, Fly.io, your own VPS, your laptop), R2 is almost always the right call.

### Other backends

Same `namidb.Client(...)` call, just a different URI. Expand for the copy-paste credentials.

<details>
<summary><strong>Google Cloud Storage</strong> (<code>gs://</code>)</summary>

```python
import os
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/etc/gcs-key.json"
client = namidb.Client("gs://my-bucket/data?ns=prod")
```

You can also pass the service-account path in the URI:
`gs://my-bucket?ns=prod&service_account=/etc/gcs-key.json`.
</details>

<details>
<summary><strong>Azure Blob Storage</strong> (<code>az://</code>)</summary>

```python
import os
os.environ["AZURE_STORAGE_ACCOUNT_NAME"] = "myacct"
os.environ["AZURE_STORAGE_ACCESS_KEY"]   = "..."
client = namidb.Client("az://myacct/mycontainer?ns=prod")
```

For Azurite (the local emulator) tack on `&use_emulator=true`.
</details>

<details>
<summary><strong>MinIO</strong> (self-hosted S3), <code>s3://</code> with an <code>endpoint=...</code></summary>

```bash
docker run -d --rm -p 9000:9000 -p 9001:9001 \
  -e MINIO_ROOT_USER=minioadmin -e MINIO_ROOT_PASSWORD=minioadmin \
  --name minio minio/minio server /data --console-address ":9001"
docker exec minio mc alias set local http://127.0.0.1:9000 minioadmin minioadmin
docker exec minio mc mb local/namidb
```

```python
import os
os.environ["AWS_ACCESS_KEY_ID"]     = "minioadmin"
os.environ["AWS_SECRET_ACCESS_KEY"] = "minioadmin"
client = namidb.Client(
    "s3://namidb?ns=dev"
    "&endpoint=http://127.0.0.1:9000"
    "&region=us-east-1"
    "&allow_http=true"
)
```

For the production-style MinIO plus `namidb-server` plus docker-compose stack,
see [Self-host as a database](#self-host-as-a-database) below.
</details>

<details>
<summary><strong>LocalStack</strong> (S3 mock for tests), <code>s3://</code> with an <code>endpoint=...</code></summary>

```bash
docker run -p 4566:4566 -e SERVICES=s3 localstack/localstack
aws --endpoint-url=http://localhost:4566 s3 mb s3://namidb-dev
export AWS_ACCESS_KEY_ID=test AWS_SECRET_ACCESS_KEY=test
```

```python
client = namidb.Client(
    "s3://namidb-dev?ns=local"
    "&endpoint=http://localhost:4566"
    "&allow_http=true"
    "&region=us-east-1"
)
```
</details>

<details>
<summary><strong>Local filesystem</strong> (<code>file://</code>)</summary>

For CI fixtures or single-machine dev when you want durability without
a bucket. Full manifest CAS via per-namespace `flock` plus atomic
`rename(2)`.

```python
client = namidb.Client("file:///var/lib/namidb?ns=prod")
# relative paths work too:
client = namidb.Client("file://./data?ns=dev")
```
</details>

<br />

## Self-host as a database

There are two ways to run NamiDB as a database you fully own. Pick whichever matches how your app wants to talk to it.

### Option A: embedded library plus your bucket

Your app (Python or Rust) imports NamiDB directly and points at a bucket you control. Lowest latency, no extra hop, no network boundary, nothing to authenticate against. This is the "DuckDB for graphs" mode.

```python
# Python service
import namidb
client = namidb.Client("s3://your-bucket/data?ns=prod&region=us-east-1")
result = client.cypher("MATCH (n:Person) RETURN count(n) AS n")
```

```rust
// Rust service
use namidb::{
    core::id::NamespaceId,
    storage::{parse_uri, WriterSession},
};

let (store, paths) = parse_uri("s3://your-bucket/data?ns=prod")?;
let mut writer = WriterSession::open(store, paths).await?;
// upserts, commit_batch, snapshot reads...
```

Reach for this when your read fan-out fits in a single process and you don't want any network overhead. Because object storage is the source of truth, two replicas of your service can open the same namespace independently, and NamiDB's epoch-CAS protocol fences out the stale writer for you.

### Option B: the `namidb-server` daemon over REST

A single Rust binary (or container image) opens a namespace and serves it over HTTP. This is the one for when the database lives on a different machine than the app, or when you want a network boundary with bearer-token auth.

```bash
# Install from source
cargo install --path crates/namidb-server

# Or build the Docker image (from the repo root)
docker build -t namidb-server:0.1 -f crates/namidb-server/Dockerfile .
```

```bash
namidb-server \
  --store "s3://your-bucket/data?ns=prod&region=us-east-1" \
  --listen 0.0.0.0:8080 \
  --auth-token "$NAMIDB_AUTH_TOKEN"
```

```bash
curl -X POST http://your-host:8080/v0/cypher \
  -H "Authorization: Bearer $NAMIDB_AUTH_TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{"query": "MATCH (n:Person) RETURN count(n) AS n"}'
# {"columns":["n"],"rows":[{"n": 42}]}
```

See [`crates/namidb-server/README.md`](./crates/namidb-server/README.md)
for the full route reference (`/v0/cypher`, `/v0/health`,
`/v0/version`, `/v0/admin/flush`), the JSON to Cypher type mapping, and
the concurrency model.

### Recipe: docker-compose with MinIO plus namidb-server

A complete self-hosted database in one file. Bring your own auth token,
everything else is wired up:

```yaml
# docker-compose.yml
services:
  minio:
    image: minio/minio
    command: server /data --console-address ":9001"
    environment:
      MINIO_ROOT_USER: minioadmin
      MINIO_ROOT_PASSWORD: minioadmin
    volumes:
      - minio-data:/data
    healthcheck:
      test: ["CMD", "mc", "ready", "local"]
      interval: 3s
      retries: 30

  bucket-init:
    image: minio/mc
    depends_on:
      minio:
        condition: service_healthy
    entrypoint: >
      sh -c "
        mc alias set local http://minio:9000 minioadmin minioadmin &&
        mc mb --ignore-existing local/namidb
      "

  namidb-server:
    image: namidb-server:0.1   # built from crates/namidb-server/Dockerfile
    depends_on:
      bucket-init:
        condition: service_completed_successfully
    environment:
      NAMIDB_STORE: "s3://namidb?ns=prod&endpoint=http://minio:9000&region=us-east-1&allow_http=true"
      NAMIDB_LISTEN: "0.0.0.0:8080"
      NAMIDB_AUTH_TOKEN: "${NAMIDB_AUTH_TOKEN:?set NAMIDB_AUTH_TOKEN in your env}"
      NAMIDB_FLUSH_INTERVAL: "30s"
      AWS_ACCESS_KEY_ID: "minioadmin"
      AWS_SECRET_ACCESS_KEY: "minioadmin"
    ports:
      - "8080:8080"

volumes:
  minio-data: {}
```

```bash
export NAMIDB_AUTH_TOKEN=$(openssl rand -hex 32)
docker compose up -d
curl -s http://localhost:8080/v0/health | jq .
```

That's it. A graph database, your data sitting in MinIO, and an
authenticated REST API on `:8080`. Swap the `NAMIDB_STORE` URI and the
same setup moves to AWS S3, R2, GCS or Azure without touching anything
else.

<br />

## CLI

```bash
# Ephemeral in-memory namespace, same as the quickstart.
namidb run "CREATE (a:Person {name: 'Alice'}), (b:Person {name: 'Bob'})"
namidb run "MATCH (p:Person) RETURN p.name"

# Persistent. Any URI scheme works.
namidb run --store "file:///var/lib/namidb?ns=prod" \
  "CREATE (a:Person {name: 'Alice'})"
namidb run --store "file:///var/lib/namidb?ns=prod" \
  "MATCH (p:Person) RETURN p.name"

namidb run --store "s3://my-bucket/data?ns=prod&region=us-west-2" \
  "MATCH (p:Person) RETURN count(*) AS n"

# Plan inspection. Doesn't touch storage.
namidb explain --verbose \
  "MATCH (a:Person)-[:KNOWS]->(b) RETURN b ORDER BY b.id LIMIT 20"
```

See [`crates/namidb-cli/README.md`](./crates/namidb-cli/README.md)
for every subcommand.

<br />

## Rust (embedded)

```rust
use std::sync::Arc;

use namidb_core::id::NamespaceId;
use namidb_query::{execute, lower, parse, Params};
use namidb_storage::{parse_uri, WriterSession};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Any supported URI scheme: memory://, file://, s3://, gs://, az://.
    let (store, paths) = parse_uri("memory://demo")?;
    let mut writer = WriterSession::open(store, paths).await?;

    // ... upsert nodes / edges, then commit_batch + flush ...

    let snap = writer.snapshot();
    let query = parse("MATCH (a:Person) RETURN count(*) AS n")?;
    let plan  = lower(&query)?;
    let rows  = execute(&plan, &snap, &Params::new()).await?;

    println!("{rows:?}");
    Ok(())
}
```

The umbrella crate ([`crates/namidb/`](./crates/namidb/)) re-exports
the stable surface, so a downstream `Cargo.toml` only needs the one
line.

<br />

## Architecture

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset=".assets/namidb-architecture-dark.svg" />
    <img src=".assets/namidb-architecture.svg" alt="NamiDB architecture: Query / Graph / Storage (LSM) / Object store layers, with a cross-snapshot caches side-car" width="900" />
  </picture>
</p>

```
┌─────────────────────────────────────────────────────────────────────┐
│  Cypher · GQL (ISO/IEC 39075:2024)                                  │
│  Cost-based optimizer · Morsel-driven executor · Factorization      │
├─────────────────────────────────────────────────────────────────────┤
│  Property graph · CSR adjacency · Columnar SSTs                     │
├─────────────────────────────────────────────────────────────────────┤
│  LSM tree · WAL · Memtable · SST · Manifest CAS                     │
│  Hybrid buffer pool (memory + NVMe)                                 │
├─────────────────────────────────────────────────────────────────────┤
│  S3 · R2 · GCS · Azure Blob · MinIO · Tigris · Local FS             │
└─────────────────────────────────────────────────────────────────────┘
```

Design proposals live in [`docs/rfc/`](./docs/rfc/). Start with
[RFC-001](./docs/rfc/001-storage-engine.md) for the storage engine and
[RFC-002](./docs/rfc/002-sst-format.md) for the SST format.

<br />

## Configuration

A handful of env vars you can tune. The defaults are fine for almost
everything; you mostly reach for these when you're chasing down a
performance or memory problem.

| Env var | Default | What it does |
|---|---|---|
| `NAMIDB_ADJACENCY` | ON | CSR adjacency in RAM, shared across snapshots (RFC-018). |
| `NAMIDB_NODE_CACHE` | ON | Cross-snapshot `NodeView` lookup cache (RFC-019). |
| `NAMIDB_SST_CACHE` | ON | SST body, decoded edge property streams, and the parsed `EdgeSstReader` (RFC-020). |
| `NAMIDB_FACTORIZE` | OFF | Factorized intermediate results in the executor (RFC-017). |
| `NAMIDB_PROFILE_DUMP` | OFF | Dump per-stage profile counters to stderr after each query. |

`namidb-server` adds a few of its own:

| Env var | Default | What it does |
|---|---|---|
| `NAMIDB_STORE` | (required) | Storage URI, e.g. `s3://bucket?ns=prod`. |
| `NAMIDB_LISTEN` | `0.0.0.0:8080` | TCP bind address. |
| `NAMIDB_AUTH_TOKEN` | unset (open) | Bearer token. When it's unset the server warns and accepts every request. |
| `NAMIDB_FLUSH_INTERVAL` | `30s` | Background memtable -> L0 flush cadence. `0s` disables it. |

<br />

## Repository layout

```
.
├── Cargo.toml              # Workspace manifest
├── rust-toolchain.toml     # Pinned toolchain
├── LICENSE                 # BSL 1.1 (auto-converts to Apache 2.0)
├── README.md
├── CONTRIBUTING.md
├── docs/
│   └── rfc/                # Design proposals (RFC-001 to RFC-020)
├── crates/
│   ├── namidb-core/        # Common types, errors, schema
│   ├── namidb-storage/     # LSM, WAL, manifest, SST, memtable, URI parser, file:// CAS
│   ├── namidb-graph/       # Property columns + CSR adjacency
│   ├── namidb-query/       # Cypher / GQL parser, optimizer, executor
│   ├── namidb-cli/         # `namidb` command-line tool
│   ├── namidb-py/          # Python bindings (PyO3 + maturin)
│   ├── namidb-server/      # `namidb-server` HTTP daemon + Dockerfile
│   ├── namidb-bench/       # LDBC-shaped synthetic bench harness
│   └── namidb/             # Public façade crate
├── bench/                  # LDBC SNB Interactive bench harness
└── tests/                  # Integration helpers (LocalStack, R2 wrapper)
```

<br />

## Documentation

| Resource | Where |
|---|---|
| **Website** | [namidb.com](https://namidb.com) |
| **Reference docs & guides** | [docs.namidb.com](https://docs.namidb.com) |
| **Design RFCs** | [`docs/rfc/`](./docs/rfc/) |
| **Python bindings** | [`crates/namidb-py/README.md`](./crates/namidb-py/README.md) |
| **HTTP server** | [`crates/namidb-server/README.md`](./crates/namidb-server/README.md) |
| **CLI** | [`crates/namidb-cli/README.md`](./crates/namidb-cli/README.md) |
| **Benchmark harness** | [`bench/README.md`](./bench/README.md) |

<br />

## Roadmap

- **Cloud (closed beta).** Multi-tenant SaaS on namidb.com with
  per-namespace scale-to-zero, encrypted-at-rest tenants, and a hosted
  control plane. [Request access](https://namidb.com).
- **Streaming responses.** `/v0/cypher/stream` (NDJSON) and
  `/v0/cypher/arrow` (Arrow IPC) for zero-copy DataFrame ingestion.
- **Bolt protocol.** Wire compatibility with the Neo4j drivers (Python,
  Java, JS and the rest) on top of the same engine.
- **Concurrent reads.** RFC-021 takes the single-writer mutex off the
  read path so a `namidb-server` can fan reads out across every core.

<br />

## Contributing

We develop in the open. Have a look at [`CONTRIBUTING.md`](./CONTRIBUTING.md)
and the RFCs in [`docs/rfc/`](./docs/rfc/) before you start. Anything
non-trivial goes through an RFC first.

<br />

## License

NamiDB is licensed under the [**Business Source License 1.1**](LICENSE).

- Free for development, testing, internal production use, and anything
  that doesn't compete with a hosted NamiDB offering from the Licensor.
- Converts automatically to the **Apache License 2.0** three years
  after each release.
- A separate commercial license is available if you need to embed or
  redistribute NamiDB outside what BSL allows, including running it as
  a hosted database service. Reach us at
  [`info@namidb.com`](mailto:info@namidb.com).

<br />

## Acknowledgements

A few projects this leans on, directly or for ideas:

- **Kùzu**, for showing that columnar storage, CSR adjacency and
  factorization are the right model for property graphs.
- **SlateDB**, for the canonical recipe for LSM trees on object
  storage.
- **turbopuffer**, for proving that namespace-per-tenant on S3 is a
  viable SaaS architecture.
- **Apache Arrow, Parquet and DataFusion**, for the columnar
  foundation.
- **foyer-rs**, for the hybrid memory and disk cache.

<br />

---

<div align="center">

### The bucket is the database.

<sub>NamiDB is a product of <a href="https://namidb.com"><b>LESAI, Corp.</b></a>, Delaware, USA.</sub><br />
<sub>© 2026 LESAI, Corp. All rights reserved.</sub>

</div>
