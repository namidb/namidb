<div align="center">

<p>
  <img src=".assets/namidb-logo.jpeg" alt="NamiDB — The graph database, native to the cloud." width="640" />
</p>

# NamiDB

**Embedded like DuckDB. Multi-tenant on object storage. Built for the AI of this decade.**

[![License: BSL 1.1](https://img.shields.io/badge/License-BSL%201.1-1f6feb.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/Rust-1.85%2B-dea584.svg?logo=rust&logoColor=white)](https://www.rust-lang.org)
[![PyPI](https://img.shields.io/badge/PyPI-namidb-3776ab.svg?logo=pypi&logoColor=white)](https://pypi.org/project/namidb/)
[![Website](https://img.shields.io/badge/Website-namidb.com-0a7ea4.svg)](https://namidb.com)
[![Docs](https://img.shields.io/badge/Docs-docs.namidb.com-0a7ea4.svg)](https://docs.namidb.com)

[**Website**](https://namidb.com) · [**Documentation**](https://docs.namidb.com) · [**RFCs**](./docs/rfc/) · [**Request early access**](https://namidb.com)

</div>

---

> ### The graph is the shape of how things relate.

NamiDB is a graph database engine built from first principles for the era of object storage, columnar execution, and AI agents. One engine. Three deployments. Object storage is the source of truth.

<br />

## Why now

Three things changed. They changed everything.

**1. Object storage grew up.**
In 2024, S3 shipped conditional writes (`If-Match` / `If-None-Match`). The last missing primitive. For the first time, you can build a coordinated, durable system where object storage *is* the database — no Raft, no ZooKeeper, no etcd. The recipe has paid off for vectors, for queues, for analytics. It had not been done for graphs.

**2. The best columnar graph engine left the market.**
In October 2025, Apple acquired Kùzu and archived the repository. The most thoughtful columnar graph engine ever published went quiet. A hole opened.

**3. Agents need graphs.**
Vector search is necessary. It is not sufficient. Knowledge graphs are the substrate of agent memory, deep retrieval, and reasoning under uncertainty. The next decade of AI will run on relationships.

So we are building the database for that decade.

<br />

## Three deployments, one engine

<!-- ─────────────────────────────────────────────────────────────────── -->
<!-- TODO: deployments diagram. Suggested: three columns, one engine     -->
<!-- icon at the centre, arrows showing the same binary fanning out to   -->
<!-- Embedded / Server / Cloud.                                          -->
<!-- ─────────────────────────────────────────────────────────────────── -->
<p align="center">
  <img src=".assets/namidb-deployments.png" alt="NamiDB deployments — Embedded, Server, Cloud" width="780" />
</p>

| Mode | Best for | How it ships |
|---|---|---|
| **Embedded** | Notebooks, single-process apps, local development, CI fixtures | `pip install namidb` — file-based or in-memory, no daemon |
| **Server** | Single-node production, persistent workloads | A single Rust binary backed by any S3-compatible object store |
| **Cloud** | Multi-tenant SaaS, agent memory, scale-to-zero per namespace | Namespace-per-tenant on S3 with snapshot isolation |

Same engine across all three. No rewrites when you graduate from a notebook to a cluster.

<br />

## What's in the engine today

- **Cypher + GQL parsing** — strict subset of GQL (ISO/IEC 39075:2024) + openCypher 9. The 12 in-scope LDBC SNB Interactive Complex Read queries (IC01–IC12) parse, plan and execute end-to-end.
- **Writes via Cypher** — `CREATE`, `MERGE`, `SET`, `DELETE`, `DETACH DELETE`, `REMOVE`. Durable on `commit_batch` (WAL append + manifest CAS).
- **Cost-based optimizer** — predicate pushdown, projection pushdown, join reorder, hash-join conversion, hash semi-join (`EXISTS` decorrelation), Parquet row-group pruning. EXPLAIN VERBOSE prints the chosen plan with selectivity and cost annotations.
- **Vectorized execution** — morsel-driven executor with optional **factorized intermediate representation** (RFC-017) for path-heavy queries.
- **Columnar storage on object storage** — Parquet node SSTs, custom edge-SST format with CSR adjacency (RFC-002), zstd compression, bloom filters, fence-pointer indices.
- **Coordination-free correctness** — single-writer-per-namespace with epoch fencing via manifest CAS. Conditional writes (`If-Match`, `If-None-Match`) replace external consensus.
- **Tiered caches** — process-wide `AdjacencyCache` (CSR), `NodeViewCache`, and `SstCache` (decoded body + edge property streams + reader). Cross-snapshot reuse with `Arc`-shared, byte-budgeted memory.
- **Python bindings** — `pip install namidb`, abi3 wheels for Linux (x86_64 + aarch64), macOS (x86_64 + arm64) and Windows (x86_64). Sync + async (`acypher`). Arrow / pandas / polars output. `s3://` and `memory://` URIs.
- **CLI** — `namidb parse`, `namidb explain --verbose`, `namidb run` for ad-hoc query work.
- **Bench harness** — synthetic, deterministic LDBC SNB Interactive harness with a paired Kùzu runner under [`bench/`](./bench/).

<br />

## Quick start

### Python

```bash
pip install namidb
```

```python
import namidb as tg

# Embedded, in-memory namespace.
client = tg.Client("memory://acme")

client.cypher("CREATE (a:Person {name: 'Alice', age: 30})")
client.cypher("CREATE (b:Person {name: 'Bob',   age: 25})")
client.cypher("MATCH (a:Person {name: 'Alice'}), (b:Person {name: 'Bob'}) "
              "CREATE (a)-[:KNOWS {since: 2020}]->(b)")

result = client.cypher(
    "MATCH (p:Person) WHERE p.age >= $min RETURN p.name AS name, p.age AS age",
    params={"min": 18},
)

print(result.columns)  # ['name', 'age']
print(result.first())  # {'name': 'Alice', 'age': 30}
df = result.to_pandas()
```

S3 / R2 / GCS / Azure / LocalStack are reachable via the `s3://` URI:

```python
client = tg.Client(
    "s3://my-bucket/data?ns=prod"
    "&region=us-west-2"
)
```

Bulk APIs, async (`acypher`), Arrow output and the LocalStack flow are
documented in [**`crates/namidb-py/README.md`**](./crates/namidb-py/README.md).

### CLI

```bash
$ namidb run "CREATE (a:Person {id: 'alice', name: 'Alice'}), \
              (b:Person {id: 'bob',   name: 'Bob'}), (a)-[:KNOWS]->(b)"

$ namidb run "MATCH (a:Person)-[:KNOWS]->(b) RETURN a.name, b.name"

$ namidb explain --verbose \
    "MATCH (a:Person)-[:KNOWS]->(b) RETURN b ORDER BY b.id LIMIT 20"
```

### Rust (embedded)

```rust
use std::sync::Arc;

use namidb_core::id::NamespaceId;
use namidb_query::{execute, lower, parse, Params};
use namidb_storage::{NamespacePaths, WriterSession};
use object_store::{memory::InMemory, ObjectStore};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
    let paths   = NamespacePaths::new("tenants", NamespaceId::new("demo")?);
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

<br />

## Architecture

<!-- ─────────────────────────────────────────────────────────────────── -->
<!-- TODO: detailed architecture illustration.                           -->
<!-- Suggested: parser → logical plan → optimizer → executor on top,     -->
<!-- LSM + SST + manifest CAS in the middle, S3/R2/GCS/Azure at the      -->
<!-- bottom, with caches as side-cars.                                   -->
<!-- ─────────────────────────────────────────────────────────────────── -->
<p align="center">
  <img src=".assets/namidb-architecture.png" alt="NamiDB architecture — query, storage and object-store tiers" width="820" />
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
│  S3 · R2 · GCS · Azure Blob · MinIO · Tigris                        │
└─────────────────────────────────────────────────────────────────────┘
```

Design proposals live in [`docs/rfc/`](./docs/rfc/). Start with
[RFC-001 — Storage Engine](./docs/rfc/001-storage-engine.md) and
[RFC-002 — SST Format](./docs/rfc/002-sst-format.md).

<br />

## Configuration

NamiDB attaches three cross-snapshot caches by default. Set the env
var to `0` to disable individually — useful for performance debugging
or memory-constrained environments.

| Env var | Default | What it does |
|---|---|---|
| `NAMIDB_ADJACENCY` | ON | CSR adjacency in-RAM, shared across snapshots (RFC-018). |
| `NAMIDB_NODE_CACHE` | ON | Cross-snapshot `NodeView` lookup cache (RFC-019). |
| `NAMIDB_SST_CACHE` | ON | SST body + decoded edge property streams + parsed `EdgeSstReader` (RFC-020). |
| `NAMIDB_FACTORIZE` | OFF | Factorized intermediate results in the executor (RFC-017). |
| `NAMIDB_PROFILE_DUMP` | OFF | Dump per-stage profile counters to stderr after each query. |

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
│   └── rfc/                # Design proposals (RFC-001 → RFC-020)
├── crates/
│   ├── namidb-core/        # Common types, errors, schema
│   ├── namidb-storage/     # LSM, WAL, manifest, SST, memtable
│   ├── namidb-graph/       # Property columns + CSR adjacency
│   ├── namidb-query/       # Cypher / GQL parser, optimizer, executor
│   ├── namidb-cli/         # `namidb` command-line tool
│   ├── namidb-py/          # Python bindings (PyO3 + maturin)
│   ├── namidb-bench/       # LDBC-shaped synthetic bench harness
│   └── namidb/             # Public façade crate
├── bench/                  # Kùzu runner + cross-engine comparator
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
| **Benchmark harness** | [`bench/README.md`](./bench/README.md) |

<br />

## Contributing

We develop in the open. Read [`CONTRIBUTING.md`](./CONTRIBUTING.md) and
the RFCs in [`docs/rfc/`](./docs/rfc/). All non-trivial design changes
go through an RFC.

<br />

## License

NamiDB is licensed under the [**Business Source License 1.1**](LICENSE).

- Free for development, testing, internal production use, and any use
  that does not compete with a hosted NamiDB offering from the
  Licensor.
- Automatically converts to **Apache License 2.0** three years after
  each release.
- A separate commercial license is available for teams that need to
  embed or redistribute NamiDB outside the bounds of BSL — including
  offering it as a hosted database service. Contact
  [`info@namidb.com`](mailto:info@namidb.com).

<br />

## Acknowledgements

NamiDB stands on the shoulders of giants:

- **Kùzu** — for showing that columnar storage + CSR adjacency +
  factorization is the right model for property graphs.
- **SlateDB** — for the canonical recipe for LSM trees on object
  storage.
- **turbopuffer** — for proving namespace-per-tenant on S3 is a viable
  SaaS architecture.
- **Apache Arrow / Parquet / DataFusion** — for the columnar
  foundation.
- **foyer-rs** — for the hybrid memory + disk cache.

<br />

---

<div align="center">

### The graph is the shape of how things relate.

<sub>NamiDB is a product of <a href="https://namidb.com"><b>Fonles Studios, Corp.</b></a> — Delaware, USA.</sub><br />
<sub>© 2026 Fonles Studios, Corp. All rights reserved.</sub>

</div>
