<div align="center">

<p>
  <img src=".assets/namidb-logo.jpeg" alt="NamiDB" width="640" />
</p>

# NamiDB

### A graph database that lives in your S3 bucket — with vectors, full-text, and your Obsidian vault built in.

Point it at a bucket (or a local folder, or nothing at all), write Cypher, and you have a property graph with vector search and hybrid retrieval. The same engine embeds in Python, runs as an HTTP/Bolt server, and speaks the Model Context Protocol so your agents can query it directly.

[![License: BSL 1.1](https://img.shields.io/badge/License-BSL%201.1-1f6feb.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/Rust-1.85%2B-dea584.svg?logo=rust&logoColor=white)](https://www.rust-lang.org)
[![PyPI](https://img.shields.io/badge/PyPI-namidb-3776ab.svg?logo=pypi&logoColor=white)](https://pypi.org/project/namidb/)
[![Docker](https://img.shields.io/badge/Docker-namidb--server-2496ed.svg?logo=docker&logoColor=white)](crates/namidb-server/Dockerfile)
[![Website](https://img.shields.io/badge/Website-namidb.com-0a7ea4.svg)](https://namidb.com)
[![Docs](https://img.shields.io/badge/Docs-docs.namidb.com-0a7ea4.svg)](https://docs.namidb.com)

[**Website**](https://namidb.com) · [**Documentation**](https://docs.namidb.com) · [**RFCs**](./docs/rfc/) · [**Request early access**](https://namidb.com)

</div>

---

NamiDB is a graph engine built on object storage. You write Cypher; it lays your nodes and edges out as columnar files in a bucket, and that bucket is the only source of truth. No Raft, no ZooKeeper, no separate metadata service — just the bucket. The same engine ships embedded as a Python or Rust library, as a standalone server, and as an MCP server for agents.

What you get out of the box:

- **A property graph** you query with Cypher / GQL.
- **Vector search** — store embeddings as node properties, rank with `cosine_similarity`, or build a real `CREATE VECTOR INDEX` (DiskANN/Vamana) for ANN.
- **Hybrid search** — BM25 lexical + semantic, fused with reciprocal rank fusion, in one call.
- **Graph algorithms** — connected components and PageRank over `CALL algo.*`.
- **Obsidian / Markdown ingestion** — turn a folder of notes into a live graph (wikilinks, embeds, tags, frontmatter) in one command.
- **Auth that's real** — static tokens, OIDC/JWT, per-namespace scoping, and an external policy hook (OPA).

<br />

## 60-second start (no credentials, nothing to install but pip)

```bash
pip install namidb
```

```python
import namidb

# Ephemeral, in-process. Swap the URI for s3://… / file://… when you're ready.
db = namidb.Client("memory://demo")

db.cypher("CREATE (a:Person {name: 'Alice', age: 30})")
db.cypher("CREATE (b:Person {name: 'Bob',   age: 25})")
db.cypher("""
  MATCH (a:Person {name: 'Alice'}), (b:Person {name: 'Bob'})
  CREATE (a)-[:KNOWS {since: 2020}]->(b)
""")

rows = db.cypher(
    "MATCH (p:Person) WHERE p.age >= $min RETURN p.name AS name, p.age AS age",
    params={"min": 18},
).rows()
print(rows)   # [{'name': 'Alice', 'age': 30}, {'name': 'Bob', 'age': 25}]
```

Writes are durable the moment `cypher()` returns. Want a DataFrame instead? `.to_pandas()`, `.to_polars()`, or `.to_arrow()` on any result.

<br />

## Make it persistent (one line changes)

The URI is the whole config. Everything else stays identical.

```python
# Local folder — great for a single machine.
db = namidb.Client("file:///var/lib/namidb?ns=prod")

# AWS S3 — durability is whatever S3 gives you.
db = namidb.Client("s3://my-bucket/data?ns=prod&region=us-west-2")

# Cloudflare R2 — no egress fees, same code.
db = namidb.Client(
    "s3://my-bucket?ns=prod"
    "&endpoint=https://<ACCOUNT_ID>.r2.cloudflarestorage.com&region=auto"
)
```

Credentials come from the standard env vars (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, …); IAM roles on EC2/EKS/Lambda just work. Kill the process, start it on another machine pointed at the same URI — the graph is still there, because the bucket *is* the database.

| Scheme | Backend |
|---|---|
| `s3://<bucket>[/<prefix>]?ns=<ns>` | AWS S3, Cloudflare R2, MinIO, Tigris, LocalStack — anything S3-compatible |
| `gs://<bucket>?ns=<ns>` | Google Cloud Storage |
| `az://<account>/<container>?ns=<ns>` | Azure Blob Storage |
| `file:///abs/dir?ns=<ns>` | Local filesystem (CAS via `flock` + atomic rename) |
| `memory://<ns>` | In-process, ephemeral — for tests and demos |

<br />

## Quick win: your Obsidian vault as a graph

Turn a folder of Markdown into a queryable graph in one command. Wikilinks `[[...]]` become `LINKS_TO` edges, `![[...]]` embeds become `EMBEDS`, `#tags` become a `:Tag` tree (`:TAGGED` / `:SUBTAG_OF`), and YAML frontmatter becomes node properties. Add `--embed` and every note also gets a vector you can search semantically.

```bash
# Build the CLI once.
cargo build --release -p namidb-cli

# Ingest a vault into a durable namespace, with embeddings.
./target/release/namidb load-vault \
  --store "file:///var/lib/namidb?ns=vault" \
  --embed \
  ./path/to/your/vault
```

Now query it like any graph:

```bash
# What links to a note?
./target/release/namidb run --store "file:///var/lib/namidb?ns=vault" \
  "MATCH (n:Note {title: 'Project X'})<-[:LINKS_TO]-(m:Note) RETURN m.title"

# Notes that share a tag.
./target/release/namidb run --store "file:///var/lib/namidb?ns=vault" \
  "MATCH (n:Note)-[:TAGGED]->(:Tag {name: 'research'}) RETURN n.title"
```

Re-run with `--prune` to mirror the vault (delete notes you removed) or `--watch` to keep the graph live as you edit. The default `--embed` uses a fast offline embedder; for real semantic quality, build with `--features remote-embedder` and set `NAMIDB_EMBED_PROVIDER` (`openai` | `voyage` | `cohere` | `gemini` | `jina`) plus the matching API key.

<br />

## Quick win: vector & hybrid search

Store embeddings as a `list[float]` property and rank with the built-in distance functions — no extra service:

```python
db = namidb.Client("file:///var/lib/namidb?ns=docs")

db.cypher(
    "CREATE (:Doc {title: $t, embedding: $v})",
    params={"t": "intro", "v": [0.1, 0.2, 0.3]},
)

# K-nearest by cosine similarity.
hits = db.cypher(
    """
    MATCH (d:Doc)
    RETURN d.title AS title, cosine_similarity(d.embedding, $q) AS score
    ORDER BY score DESC LIMIT 5
    """,
    params={"q": [0.1, 0.2, 0.25]},
).rows()
```

For large collections, promote it to a real ANN index (DiskANN/Vamana) so the optimizer serves the same query from the index instead of scanning. Build the server or CLI with `--features vector-index`, then:

```cypher
CREATE VECTOR INDEX doc_emb ON :Doc(embedding) METRIC cosine DIMENSION 3;
```

And `bm25(...)` gives you real lexical relevance — the MCP `hybrid_search` tool below fuses it with vector scores automatically.

<br />

## Quick win: plug it into your agents (MCP)

NamiDB ships an MCP server (`namidb-mcp`) that exposes a namespace to any MCP client over stdio. Point it at a vault or a durable namespace and your agent gets graph traversal, tag queries, vector + hybrid search, and graph algorithms — read-only by design.

```bash
cargo build --release -p namidb-mcp
```

Drop this into your MCP client config (e.g. Claude Desktop's `mcpServers`):

```json
{
  "mcpServers": {
    "namidb": {
      "command": "/abs/path/to/target/release/namidb-mcp",
      "args": ["--store", "file:///var/lib/namidb?ns=vault"]
    }
  }
}
```

Or load a vault on startup and keep it live: `"args": ["--vault", "./my-vault", "--watch"]`.

The tools it exposes:

| Tool | What it does |
|---|---|
| `list_notes`, `get_note`, `search` | List, fetch, and substring-search notes |
| `backlinks`, `neighbors`, `orphans` | Graph traversal — what links here, N-hop neighbours, dangling notes |
| `list_tags`, `notes_by_tag`, `subtags`, `tags_of` | Tag queries over the `:Tag` tree |
| `vector_search` | Semantic K-NN by cosine similarity |
| `hybrid_search` | BM25 lexical + semantic, fused with reciprocal rank fusion |
| `graph_algorithm` | Run `wcc` (connected components) or `pagerank` over a subgraph |
| `cypher` | Read-only Cypher escape hatch |

<br />

## Run it as a server (HTTP + Bolt)

```bash
# Plain HTTP on :8080.
cargo run --release -p namidb-server -- --store "s3://my-bucket?ns=prod&region=us-west-2"
```

```bash
curl -s localhost:8080/v0/cypher \
  -H 'content-type: application/json' \
  -d '{"query":"RETURN 1 + 41 AS n"}'
# {"columns":["n"],"rows":[{"n":42}]}
```

Add `--bolt-listen 0.0.0.0:7687` and point any Neo4j driver or `cypher-shell` at `bolt://localhost:7687`. Both protocols share one writer per namespace, so they never disagree.

**Auth and authorization**, all optional and off by default:

```bash
cargo run --release -p namidb-server --features jwt,pdp,vector-index -- \
  --store "s3://my-bucket?ns=prod" \
  --bolt-listen 0.0.0.0:7687 \
  --auth-token "$NAMIDB_AUTH_TOKEN" \                       # static bearer token
  --jwt-jwks-url "https://issuer/.well-known/jwks.json" \   # OIDC/JWT, group → role
  --jwt-namespaces-claim tenants \                          # scope a token to namespaces
  --pdp-url "http://opa:8181/v1/data/namidb/allow"          # external policy (OPA), fail-closed
```

| Flag (env var) | What it does |
|---|---|
| `--store` (`NAMIDB_STORE`) | Storage URI. Required. |
| `--listen` (`NAMIDB_LISTEN`) | HTTP bind, default `0.0.0.0:8080`. |
| `--bolt-listen` (`NAMIDB_BOLT_LISTEN`) | Enable the Bolt listener (e.g. `0.0.0.0:7687`). |
| `--auth-token` / `--auth-tokens-file` | Static bearer token(s) with per-token roles + namespace scopes. |
| `--jwt-*` *(feature `jwt`)* | Validate OIDC JWTs against a JWKS, map a group claim to a role, scope by a namespaces claim. |
| `--pdp-url` *(feature `pdp`)* | Send each query to an OPA-style policy endpoint; deny unless it allows (fail-closed). |
| `--multi-tenant` / `--default-namespace` | Serve many namespaces, routed by path (`/<ns>/v0/cypher`) or the `X-NamiDB-Namespace` header. |

> Build features: `jwt` (OIDC), `pdp` (external policy), `vector-index` (`CREATE VECTOR INDEX`). Omit them for a smaller binary; the default build is static-token auth only.

<br />

## Embed it in Rust

```rust
use namidb_query::{execute, lower, parse, Params};
use namidb_storage::{parse_uri, WriterSession};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let (store, paths) = parse_uri("memory://demo")?;        // or file://, s3://, gs://, az://
    let mut writer = WriterSession::open(store, paths).await?;

    // ... upsert nodes / edges, then commit_batch ...

    let snap  = writer.snapshot();
    let query = parse("MATCH (a:Person) RETURN count(*) AS n")?;
    let rows  = execute(&lower(&query)?, &snap, &Params::new()).await?;
    println!("{rows:?}");
    Ok(())
}
```

The umbrella crate [`crates/namidb/`](./crates/namidb/) re-exports the stable surface, so a downstream `Cargo.toml` needs only one line.

<br />

## CLI cheatsheet

```bash
# One-shot query against any backend.
namidb run --store "file:///var/lib/namidb?ns=prod" "MATCH (p:Person) RETURN count(*) AS n"

# Ingest an Obsidian / Markdown vault (see the quick win above).
namidb load-vault --store "s3://bucket?ns=vault" --embed --prune ./vault

# Inspect a plan without touching storage.
namidb explain --verbose "MATCH (a:Person)-[:KNOWS]->(b) RETURN b LIMIT 20"

# Consistent backup / restore of a namespace.
namidb backup  --store "s3://bucket?ns=prod" ./snapshot
namidb restore --store "s3://bucket?ns=restored" ./snapshot
```

See [`crates/namidb-cli/README.md`](./crates/namidb-cli/README.md) for every subcommand.

<br />

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│  Cypher · GQL (ISO/IEC 39075:2024)                                   │
│  Cost-based optimizer · Morsel-driven executor · Factorization       │
│  Vector / hybrid search · Graph algorithms (CALL algo.*)             │
├─────────────────────────────────────────────────────────────────────┤
│  Property graph · CSR adjacency · Columnar SSTs · Vamana ANN index   │
├─────────────────────────────────────────────────────────────────────┤
│  LSM tree · WAL · Memtable · SST · Manifest CAS                      │
│  Hybrid buffer pool (memory + NVMe)                                  │
├─────────────────────────────────────────────────────────────────────┤
│  S3 · R2 · GCS · Azure Blob · MinIO · Tigris · Local FS             │
└─────────────────────────────────────────────────────────────────────┘
```

Design proposals live in [`docs/rfc/`](./docs/rfc/) — start with [RFC-001](./docs/rfc/001-storage-engine.md) (storage engine) and [RFC-002](./docs/rfc/002-sst-format.md) (SST format).

<br />

## Configuration

The defaults are fine for almost everything; reach for these when chasing a performance or memory problem.

| Env var | Default | What it does |
|---|---|---|
| `NAMIDB_ADJACENCY` | ON | CSR adjacency in RAM, shared across snapshots. |
| `NAMIDB_NODE_CACHE` | ON | Cross-snapshot `NodeView` lookup cache. |
| `NAMIDB_SST_CACHE` | ON | SST body + decoded edge property streams cache. |
| `NAMIDB_FACTORIZE` | OFF | Factorized intermediate results in the executor. |
| `NAMIDB_EMBED_PROVIDER` | unset | Remote embedder for `load-vault --embed` (`openai`/`voyage`/`cohere`/`gemini`/`jina`; needs `--features remote-embedder`). |

<br />

## Repository layout

```
crates/
├── namidb-core/        # Common types, errors, schema
├── namidb-storage/     # LSM, WAL, manifest CAS, SST, URI parser, file:// CAS
├── namidb-graph/       # Property columns, CSR adjacency, WCC + PageRank
├── namidb-ann/         # DiskANN / Vamana vector index
├── namidb-query/       # Cypher / GQL parser, optimizer, executor, BM25
├── namidb-markdown/    # Obsidian / Markdown vault → graph (+ embedders)
├── namidb-cli/         # `namidb` command-line tool
├── namidb-py/          # Python bindings (PyO3 + maturin)
├── namidb-server/      # `namidb-server` HTTP + Bolt daemon (auth, JWT, PDP)
├── namidb-mcp/         # `namidb-mcp` Model Context Protocol server
├── namidb-bench/       # LDBC-shaped synthetic bench harness
└── namidb/             # Public façade crate
```

<br />

## Documentation

| Resource | Where |
|---|---|
| Website | [namidb.com](https://namidb.com) |
| Reference docs & guides | [docs.namidb.com](https://docs.namidb.com) |
| Design RFCs | [`docs/rfc/`](./docs/rfc/) |
| Python bindings | [`crates/namidb-py/README.md`](./crates/namidb-py/README.md) |
| HTTP / Bolt server | [`crates/namidb-server/README.md`](./crates/namidb-server/README.md) |
| MCP server | [`crates/namidb-mcp/README.md`](./crates/namidb-mcp/README.md) |
| CLI | [`crates/namidb-cli/README.md`](./crates/namidb-cli/README.md) |

<br />

## Contributing

We develop in the open. Read [`CONTRIBUTING.md`](./CONTRIBUTING.md) and the RFCs in [`docs/rfc/`](./docs/rfc/) before you start — anything non-trivial goes through an RFC first.

<br />

## License

NamiDB is licensed under the [**Business Source License 1.1**](LICENSE).

- Free for development, testing, internal production use, and anything that doesn't compete with a hosted NamiDB offering from the Licensor.
- Converts automatically to the **Apache License 2.0** three years after each release.
- A separate commercial license is available if you need to embed or redistribute NamiDB outside what BSL allows, including running it as a hosted database service. Reach us at [`info@namidb.com`](mailto:info@namidb.com).

<br />

---

<div align="center">

### The bucket is the database.

<sub>NamiDB is built by <a href="https://namidb.com"><b>NamiDB, Inc.</b></a>, Delaware, USA.</sub><br />
<sub>© 2026 NamiDB, Inc. All rights reserved.</sub>

</div>
