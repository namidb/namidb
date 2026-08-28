<div align="center">

<p>
  <img src=".assets/namidb_3.png" alt="NamiDB — the bucket is the database" width="820" />
</p>

# NamiDB

### A graph database that lives in your S3 bucket — with vectors, full-text, and your Obsidian vault built in.

Point it at a bucket (or a local folder, or nothing at all), write Cypher, and you have a property graph with vector search and hybrid retrieval. The same engine embeds in Python, runs as an HTTP/Bolt server, and speaks the Model Context Protocol so your agents can query it directly.

[![License: BSL 1.1](https://img.shields.io/badge/License-BSL%201.1-1f6feb.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/Rust-1.85%2B-dea584.svg?logo=rust&logoColor=white)](https://www.rust-lang.org)
[![PyPI](https://img.shields.io/badge/PyPI-namidb-3776ab.svg?logo=pypi&logoColor=white)](https://pypi.org/project/namidb/)
[![Docker Hub](https://img.shields.io/docker/v/namidb/namidb-server?sort=semver&label=Docker&logo=docker&logoColor=white&color=2496ed)](https://hub.docker.com/r/namidb/namidb-server)
[![Website](https://img.shields.io/badge/Website-namidb.com-0a7ea4.svg)](https://namidb.com)
[![Docs](https://img.shields.io/badge/Docs-docs.namidb.com-0a7ea4.svg)](https://docs.namidb.com)

[**Website**](https://namidb.com) · [**Documentation**](https://docs.namidb.com) · [**RFCs**](./docs/rfc/) · [**Request early access**](https://namidb.com)

</div>

---

NamiDB is a graph engine built on object storage. You write Cypher; it lays your nodes and edges out as columnar files in a bucket, and that bucket is the only source of truth. No Raft, no ZooKeeper, no separate metadata service — just the bucket. The same engine ships embedded as a Python or Rust library, as a standalone server, and as an MCP server for agents.

What you get out of the box:

- **A property graph** you query with Cypher / GQL — `CALL { … }` subqueries (correlated and uncorrelated), `EXISTS { … }`, `FOREACH`, label disjunction `(n:A|B)`, and open-ended / parameterised variable-length paths (`*`, `*1..$n`).
- **Schema when you want it** — `CREATE CONSTRAINT` for uniqueness (single- *and* multi-property), `NOT NULL`, and `CREATE INDEX` for equality lookups, with `IF NOT EXISTS` and `SHOW CONSTRAINTS` / `SHOW INDEXES`. Schema-optional: the engine enforces only what you declare.
- **Vector search** — store embeddings as node properties and rank with
  `cosine_similarity`, or build a range-readable clustered ANN index for
  cosine, dot, *and* Euclidean. Full-precision vectors are fetched only for a
  bounded rerank set; optional int8 navigation/data codes reduce the large
  payload by about 4×. Reachable from idiomatic Cypher KNN or
  `CALL search.vector` / Neo4j-style `db.index.vector.queryNodes`.
- **Hybrid search** — BM25 lexical + dense vector, fused with Reciprocal Rank Fusion (or a weighted blend), in one `CALL search.hybrid`.
- **Graph algorithms** — connected components (weak & strong), PageRank, degree & betweenness centrality, triangle count, community detection (label propagation *and* Louvain modularity), shortest paths, and FastRP structural embeddings over `CALL algo.*`, each with an optional `{labels, edge_types, direction}` subgraph projection.
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

For production object-store deployments, configure a bucket lifecycle rule
that aborts incomplete multipart uploads after a short retention window.
NamiDB aborts recoverable upload failures itself; the lifecycle rule covers
process or host termination while S3/R2 still owns uploaded parts.

| Scheme | Backend |
|---|---|
| `s3://<bucket>[/<prefix>]?ns=<ns>` | AWS S3, Cloudflare R2, MinIO, Tigris, LocalStack — anything S3-compatible |
| `gs://<bucket>?ns=<ns>` | Google Cloud Storage |
| `az://<account>/<container>?ns=<ns>` | Azure Blob Storage |
| `file:///abs/dir?ns=<ns>` | Local filesystem (Create-only pointer CAS via `O_CREAT\|O_EXCL`) |
| `memory://<ns>` | In-process, ephemeral — for tests and demos |

**Object store requirements.** NamiDB needs exactly one conditional-write
capability from the bucket: **PUT-if-absent** (`If-None-Match: *`) — the
compare-and-swap behind every manifest, pointer, and WAL commit (RFC-029).
AWS S3 (native since Aug 2024), GCS, Azure Blob, Cloudflare R2, Tigris, and
current MinIO all support it. Conditional *overwrite* (`If-Match`) is **not**
required for writes. An "S3-compatible" service that ignores
`If-None-Match: *` preconditions cannot host NamiDB safely — two writers
could both believe they committed. Eventually-consistent LIST is tolerated
(the versioned pointer probe closes that window).

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

## Quick win: graph algorithms

Run analytics over the whole graph with `CALL algo.<name>()`. Every procedure yields one row per node, so you compose it with `YIELD` / `RETURN` / `ORDER BY` like any other query — no separate analytics job, no export.

```cypher
-- Which notes are the hubs? (authority via PageRank)
CALL algo.pagerank() YIELD node_id, score
RETURN node_id, score ORDER BY score DESC LIMIT 10;

-- Most-connected notes (degree centrality).
CALL algo.degree() YIELD node_id, in_degree, out_degree, degree
RETURN node_id, degree ORDER BY degree DESC LIMIT 10;

-- Communities, and clusters of mutual reachability.
CALL algo.label_propagation() YIELD node_id, community RETURN node_id, community;
CALL algo.wcc() YIELD node_id, component RETURN node_id, component;   -- undirected
CALL algo.scc() YIELD node_id, component RETURN node_id, component;   -- directed cycles

-- How tightly knit is each node? (triangles + clustering coefficient)
CALL algo.triangle_count() YIELD node_id, triangles, coefficient RETURN *;

-- Hop distance from a starting node (BFS; pass weighted: true for Dijkstra).
CALL algo.shortest_path({source: "<node-uuid>"}) YIELD node_id, distance, hops RETURN *;

-- Modularity communities (Louvain) and bridge nodes (Brandes betweenness).
CALL algo.louvain() YIELD node_id, community RETURN node_id, community;
CALL algo.betweenness() YIELD node_id, score RETURN node_id, score ORDER BY score DESC LIMIT 10;

-- Structural embeddings from pure graph shape (FastRP) — no model, no service.
-- The output is a FloatVector, ready to store and serve from a vector index:
-- "find structurally similar nodes" becomes a KNN over the graph itself.
CALL algo.fastRP({dimension: 256, iterations: 4, seed: 42}) YIELD node_id, embedding
RETURN node_id, embedding;

-- Every algo.* takes an optional graph projection: restrict to labels /
-- edge types (the induced subgraph) and pick the orientation.
CALL algo.pagerank({labels: ['Person'], edge_types: ['KNOWS'], direction: 'undirected'})
YIELD node_id, score RETURN node_id, score ORDER BY score DESC LIMIT 10;
```

The full set: `wcc`, `scc`, `pagerank`, `degree`, `triangle_count`, `label_propagation`, `louvain`, `betweenness`, `shortest_path`, `fastRP`. They run exact (no sampling), are deterministic, and honour the query deadline, so a heavy call on a large graph is interruptible. The same algorithms are one call away for agents through the MCP `graph_algorithm` tool.

<br />

## Quick win: constraints, schema & richer Cypher

Declare just the invariants you care about — the engine enforces them on write and leaves everything else schema-less. Uniqueness can span several properties, equality indexes speed up point lookups, and `IF NOT EXISTS` makes a migration script idempotent.

```cypher
-- Uniqueness on a single property…
CREATE CONSTRAINT user_email IF NOT EXISTS
  FOR (u:User) REQUIRE u.email IS UNIQUE;

-- …or on a tuple (composite): no two configs share (tenant, name, parameterSet).
CREATE CONSTRAINT cfg_uq IF NOT EXISTS
  FOR (c:Config) REQUIRE (c.tenant, c.name, c.parameterSet) IS UNIQUE;

-- An equality index for fast point lookups.
CREATE INDEX FOR (d:Doc) ON (d.slug);

-- Introspect what's declared.
SHOW CONSTRAINTS;
SHOW INDEXES;
```

A write that violates a uniqueness constraint is rejected (HTTP `409 Conflict`); a composite constraint applies only when every listed property is present. The same `SHOW` / `CREATE CONSTRAINT` commands work over HTTP, Bolt, and the embedded Python client.

The query language goes well beyond basic `MATCH` — subqueries, existential checks, per-row updates, and flexible patterns all compose like any other clause:

```cypher
-- Correlated subquery: the 3 most-liked posts per author, in one query.
MATCH (a:Author)
CALL { WITH a MATCH (a)-[:WROTE]->(p:Post) RETURN p.title AS title ORDER BY p.likes DESC LIMIT 3 }
RETURN a.name AS author, title;

-- Existential filter + conditional per-row write (the FOREACH idiom).
MATCH (u:User) WHERE NOT EXISTS { MATCH (u)-[:HAS]->(:Profile) }
FOREACH (x IN [1] | CREATE (u)-[:HAS]->(:Profile {created: true}));

-- Any-of labels, plus a parameterised variable-length path.
MATCH (n:Person|Org)-[:KNOWS*1..$hops]->(m) RETURN DISTINCT m;
```

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

For large collections, promote it to a range-readable clustered ANN index so
the optimizer can route to a bounded set of object pages instead of scanning
or loading the vector corpus. Build the server with `--features vector-index`
(or grab the prebuilt server binary — see below), then:

```cypher
CREATE VECTOR INDEX doc_emb IF NOT EXISTS ON :Doc(embedding) METRIC cosine DIMENSION 3;
```

`IF NOT EXISTS` goes between the name and `ON`, so re-running the statement is a no-op when the index already exists — the same idempotent-migration story as `CREATE CONSTRAINT` / `CREATE INDEX`.

All three metrics are served from the index — `cosine` and `dot_product` (rank `… DESC`, higher is closer) and `euclidean_distance` (rank `… ASC`, lower is closer) — and the indexed score equals the flat scan's exactly. For large corpora, add `WITH { quantization: int8 }` to store the vectors as int8 (~4× smaller index, cosine-only). Or call it as a procedure with a tunable beam width, including the Neo4j-compatible form:

```cypher
CALL search.vector({label: 'Doc', property: 'embedding', query: $q, k: 5, ef: 200}) YIELD node, score;
CALL db.index.vector.queryNodes('doc_emb', 5, $q) YIELD node, score;
```

`search.vector`, `search.bm25`, `search.hybrid`, and
`db.index.vector.queryNodes` all take an optional `filter` map. This is the
right tool for current/tenant-scoped retrieval inside a shared index:

```cypher
-- Optional but recommended: equality/IN metadata indexes become native
-- pre-filters for vector retrieval (typed scalars include BOOLEAN).
CREATE INDEX IF NOT EXISTS FOR (d:Doc) ON (d.tenant_id);

CALL search.vector({
  label: 'Doc', property: 'embedding', query: $q, k: 5,
  filter: { tenant_id: $t }
}) YIELD node, score;

CALL search.bm25({
  label: 'Doc', text_properties: ['title', 'body'], query: $text, k: 10,
  filter: { tenant_id: $t, vigente: true }
}) YIELD node, score;

-- queryNodes carries it (with `ef`) in the optional 4th map.
CALL db.index.vector.queryNodes('doc_emb', 5, $q, { ef: 200, filter: { tenant_id: $t } }) YIELD node, score;
```

A scalar value means equality (`tenant_id = $t`), a list means `IN` (`{ tier: [1, 2, 3] }` → `tier IN [1, 2, 3]`), and a `{ gte, gt, lte, lt, eq, ne }` map is a range (`{ score: { gte: 0.5 } }`) — keys AND-combine. On `search.vector` and `db.index.vector.queryNodes`, the filter is applied before the returned `k`, so a selective predicate does not merely truncate a shared unfiltered top-k.

Idiomatic KNN receives the same pre-`k` treatment: `MATCH (d:Doc) WHERE
d.vigente = true AND d.ambito IN $ambitos RETURN d ORDER BY
cosine_similarity(d.embedding, $q) DESC LIMIT 50` carries lossless String/Bool
equality groups into the `.vg` postings. Unique-property and `elementId`
predicates remain cheaper exact point seeks; unsupported conjuncts stay as
residual checks with adaptive widening and exact fallback.

For indexed string and boolean equality/`IN` conjuncts, authoritative compaction embeds complete postings directly in the `.vg` as vector ordinals: sparse values use sorted `u32`s and high-frequency/dense postings use bitmaps. A query ORs alternatives and ANDs properties inside the decoded vector body, before metric reranking and before `k`; rejected nodes can still guide ANN navigation, but their full records and a corpus-sized NodeId set are never hydrated. An absent value under a materialised property is an exact empty slice; an absent property means “unsupported” and retains the residual path.

The posting build is bounded and never truncates: a property is omitted atomically after `NAMIDB_VECTOR_FILTER_MAX_DISTINCT` (default `4096`) or the per-body `NAMIDB_VECTOR_FILTER_MAX_BYTES` budget (default `64 MiB`). Legacy v3 bodies and omitted/high-cardinality properties first try a complete equality-sidecar result capped by `NAMIDB_VECTOR_FILTER_ID_CANDIDATE_CAP` (default `8192` IDs), then use adaptive widening plus the exact scan fallback. Unindexed, numeric, and range predicates follow that residual path too, so an index changes cost, not results. (`1 = 1.0` is valid Cypher, so numeric prefiltering cannot safely probe only one typed posting.) `YIELD node WHERE …` is not the procedure-filter syntax—put `filter: {...}` in the argument map as above (a later `WITH node WHERE …` remains an ordinary post-filter).

Recall is tunable. The procedures take a first-class `ef` beam width (shown above); the natural `MATCH … ORDER BY cosine_similarity(…)` form has no syntax for it, so it reads a reserved `$__vector_ef` param. An explicit value replaces the default beam and is clamped to at least the requested candidate count; values below the default `64` can trade recall for latency. `$__vector_ef` is a **non-stable** knob — expect it to be superseded by an `OPTIONS { ef }` clause.

FT4 stores the same complete String/Bool equality/`IN` postings alongside each
incremental text segment. `search.bm25` and the sparse leg of `search.hybrid`
intersect them before sparse top-k while reconstructing `N`, average document
length and every query-term `df` globally, before filtering. Unsupported
range/negative predicates widen the ranked prefix geometrically up to
`NAMIDB_HYBRID_TEXT_FILTER_CANDIDATE_CAP`; if that is not enough, an exact
two-pass scorer takes over. Selective filters therefore cannot shorten a page
merely because ineligible documents ranked first, and the cap bounds fast-path
hydration rather than recall.

Two correctness details worth knowing. When a vector index covers a property, embeddings are dimension-checked on write: a wrong-dimension value is rejected, and a correct-dimension bare `list[float]` is coerced to a dense vector so it actually enters the index (otherwise a bare list reads back fine via flat scan yet is silently skipped at build time). And cosine is undefined for a zero-magnitude vector — `cosine_similarity` returns NULL and that row drops out of a KNN — a contract the index enforces too, so the indexed result equals the flat scan's row-for-row.

For lexical relevance, `CALL search.bm25(...)` ranks documents with full BM25 — real IDF (rare query terms outweigh common ones), term-frequency saturation, and corpus-derived length normalization:

```cypher
CALL search.bm25({label: 'Doc', text_properties: ['title', 'body'], query: 'graph storage', k: 10})
YIELD node, score
RETURN node.title AS title, score ORDER BY score DESC;
```

The query string understands quoted **phrases** and trailing-`*` **prefixes** alongside plain terms — `'"graph database" stor*'` requires the exact adjacent phrase and expands the prefix over the vocabulary — with identical semantics on the index path and the flat-scan fallback.

By default this scans the label and computes corpus statistics on the fly. For large collections, register a persistent inverted index so the same query answers from precomputed postings instead of re-scanning — build the server with `--features text-index` (or use the prebuilt server, below), then:

```cypher
CREATE FULLTEXT INDEX doc_ft IF NOT EXISTS ON :Doc(title, body);
```

`IF NOT EXISTS` sits between the name and `ON` here too, keeping migration scripts idempotent.

The first base is built during migration/compaction. Subsequent node flushes
publish immutable FT4 deltas and their exact update/delete suppress records in
the same manifest commit; `CALL search.bm25` uses the active base+delta
generation automatically when its `(label, properties)` match. Any incomplete
coverage or corrupt/missing segment falls back atomically to the exact scan.

A mis-created index is not permanent: `DROP INDEX doc_ft [IF EXISTS]` removes a fulltext index and `DROP VECTOR INDEX doc_emb [IF EXISTS]` a vector one — the descriptor and the index's SSTs go in one commit, writes constrained by a wrong-dimension vector index are immediately un-bricked, and the freed `(label, properties)` slot can be re-created corrected.

**Hybrid search** fuses both channels natively — `CALL search.hybrid(...)` runs the dense (vector KNN) and sparse (BM25) retrievals and combines them with **Reciprocal Rank Fusion** (the default; `fusion: 'linear'` for a weighted blend). Each leg serves from its index or its exact flat scan, so the result is always fresh:

```cypher
CALL search.hybrid({
  label: 'Doc',
  query_text: 'graph storage', text_properties: ['title', 'body'],
  query_vector: $q,             vector_property: 'embedding',
  k: 10
}) YIELD node, score
RETURN node.title AS title, score ORDER BY score DESC;
```

There's also a per-row `bm25(text, query)` scalar for inline use when you don't need corpus-wide IDF, and the MCP `hybrid_search` tool exposes the same fusion to agents.

The full vector-index design — freshness gating, all three metrics, int8 quantization, and the index-vs-flat equivalence — is [RFC-030](./docs/rfc/030-vector-index.md); the multi-tenant filtered-ANN pre-filtering path behind the `filter` argument is [RFC-032](./docs/rfc/032-filtered-ann-prefiltering.md).

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
| `graph_algorithm` | Run `wcc`/`scc`/`pagerank`/`degree`/`betweenness`/`triangle_count`/`label_propagation`/`louvain`/`shortest_path` over a subgraph |
| `cypher` | Read-only Cypher escape hatch |

<br />

## Run it as a server (HTTP + Bolt)

The official image is on [Docker Hub](https://hub.docker.com/r/namidb/namidb-server) (multi-arch: `linux/amd64` + `linux/arm64`), built with vector + full-text search on:

```bash
# Official image, plain HTTP on :8080.
docker run --rm -p 8080:8080 \
  -e AWS_ACCESS_KEY_ID -e AWS_SECRET_ACCESS_KEY \
  -e NAMIDB_AUTH_TOKEN="$(openssl rand -hex 32)" \
  namidb/namidb-server:2 --store "s3://my-bucket?ns=prod&region=us-west-2"
```

A token (or `--no-auth`) is required: since 2.2.0 the server refuses to boot
with no auth configured instead of silently serving open. For a local
`file://` store, mount a **named volume** at `/var/lib/namidb` — the image
pre-creates it owned by its non-root uid (65532), so the volume inherits the
right ownership. A **bind mount** needs a one-time
`chown -R 65532:65532 <hostdir>` first (that also fixes named volumes created
root-owned by images before 2.2.0):

```bash
docker run --rm -p 8080:8080 -v namidb-data:/var/lib/namidb \
  -e NAMIDB_AUTH_TOKEN=... \
  namidb/namidb-server:2 --store "file:///var/lib/namidb?ns=prod"
```

For a full self-hosted stack (server + MinIO as the bucket) see [`docker-compose.yml`](docker-compose.yml). Or run it from source:

```bash
# Plain HTTP on :8080. --no-auth is for local dev only;
# set --auth-token / NAMIDB_AUTH_TOKEN before exposing the port.
cargo run --release -p namidb-server -- --no-auth --store "s3://my-bucket?ns=prod&region=us-west-2"
```

```bash
curl -s localhost:8080/v0/cypher \
  -H 'content-type: application/json' \
  -d '{"query":"RETURN 1 + 41 AS n"}'
# {"columns":["n"],"rows":[{"n":42}]}
```

Add `--bolt-listen 0.0.0.0:7687` and point any Neo4j driver or `cypher-shell` at `bolt://localhost:7687`. Both protocols share one writer per namespace, so they never disagree. In single-tenant mode Bolt serves the `?ns=` of `--store`; under `--multi-tenant` (since 2.3.0) statements route to the namespace named by the driver's `database=` session parameter (the Bolt `db` field), with the same token namespace scoping HTTP enforces. Bolt keeps a fixed 64 KiB pre-authentication ceiling and defaults authenticated messages to 64 MiB. Before growing or decoding a data frame, all connections share a weighted memory budget and the server checks current RSS plus the request's projected working set. Nested PackStream values also have one cumulative decoded-heap/cardinality budget, and result values are converted only as each `PULL` page demands them.

**Auth and authorization.** A bearer token is required by default — the server refuses to boot with no auth source configured; `--no-auth` is the explicit opt-out for local development. JWT and PDP stay optional:

```bash
cargo run --release -p namidb-server --features jwt,pdp,vector-index -- \
  --store "s3://my-bucket?ns=prod" \
  --bolt-listen 0.0.0.0:7687 \
  --auth-token "$NAMIDB_AUTH_TOKEN" \                       # static bearer token
  --jwt-jwks-url "https://issuer/.well-known/jwks.json" \   # OIDC/JWT, group → role
  --jwt-namespaces-claim tenants \                          # scope a token to namespaces (enforced on HTTP and Bolt)
  --pdp-url "http://opa:8181/v1/data/namidb/allow"          # external policy (OPA), fail-closed
```

| Flag (env var) | What it does |
|---|---|
| `--store` (`NAMIDB_STORE`) | Storage URI. Required. |
| `--listen` (`NAMIDB_LISTEN`) | HTTP bind, default `0.0.0.0:8080`. |
| `--bolt-listen` (`NAMIDB_BOLT_LISTEN`) | Enable the Bolt listener (e.g. `0.0.0.0:7687`). |
| `--bolt-max-message-bytes` (`NAMIDB_BOLT_MAX_MESSAGE_BYTES`) | Authenticated Bolt message cap, default 64 MiB; the pre-auth cap stays fixed at 64 KiB. |
| `--auth-token` / `--auth-tokens-file` | Static bearer token(s) with per-token roles + namespace scopes. One of these (or a JWT config) is required unless `--no-auth` is set. |
| `--no-auth` (`NAMIDB_NO_AUTH`) | Explicitly run without authentication (anonymous read-write). Without it, a boot with no auth configured fails instead of silently serving open. |
| `--jwt-*` *(feature `jwt`)* | Validate OIDC JWTs against a JWKS, map a group claim to a role, scope by a namespaces claim. |
| `--pdp-url` *(feature `pdp`)* | Send each query to an OPA-style policy endpoint; deny unless it allows (fail-closed). |
| `--multi-tenant` / `--default-namespace` | Serve many namespaces: HTTP routes by path (`/<ns>/v0/cypher`) or the `X-NamiDB-Namespace` header; Bolt (since 2.3.0) routes by the driver's `database=` session parameter. |

> Build features: `jwt` (OIDC), `pdp` (external policy), `vector-index` (`CREATE VECTOR INDEX`), `text-index` (`CREATE FULLTEXT INDEX`). Omit them for a smaller binary; the default build is static-token auth only.
>
> **Prebuilt server binary.** Each GitHub Release ships a `namidb-server-<tag>-<target>` archive built with `--features vector-index,text-index`, so `CREATE VECTOR INDEX` / `CREATE FULLTEXT INDEX` work out of the box — the [official Docker image](https://hub.docker.com/r/namidb/namidb-server) is the same configuration. The `namidb` and `namidb-mcp` archives are feature-light — no optional features — so build the server from source with your own `--features` set when you also want `jwt`/`pdp` or a slimmer binary.
>
> Linux GNU archives are built natively on Ubuntu 22.04 for both x86_64 and
> arm64, and the release gate rejects symbols newer than `GLIBC_2.35`. Use the
> official container or build from source when targeting an older libc.

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

The embedded Rust API is currently an unpublished, low-level workspace
surface. The example above uses `namidb-query` and `namidb-storage` directly;
git/path consumers should pin an exact revision and expect those APIs to evolve.
The [`namidb`](./crates/namidb/) façade currently re-exports only
`namidb-core`; it is not yet the complete embedded-engine API.

<br />

## CLI cheatsheet

```bash
# One-shot query against any backend.
namidb run --store "file:///var/lib/namidb?ns=prod" "MATCH (p:Person) RETURN count(*) AS n"

# Ingest an Obsidian / Markdown vault (see the quick win above).
namidb load-vault --store "s3://bucket?ns=vault" --embed --prune ./vault

# Inspect a plan without touching storage.
namidb explain --verbose "MATCH (a:Person)-[:KNOWS]->(b) RETURN b LIMIT 20"

# Consistent backup / restore of a namespace (copy between URIs).
namidb backup  --from "s3://bucket?ns=prod"      --to "file:///snapshots/prod"
namidb restore --from "file:///snapshots/prod"   --to "s3://bucket?ns=restored"
```

See [`crates/namidb-cli/README.md`](./crates/namidb-cli/README.md) for every subcommand.

<br />

## Architecture

<p align="center">
  <img src=".assets/namidb_2.png" alt="NamiDB — the oracle of graphs" width="820" />
</p>

```
┌─────────────────────────────────────────────────────────────────────┐
│  Cypher · GQL (ISO/IEC 39075:2024)                                   │
│  Cost-based optimizer · Morsel-driven executor · Factorization       │
│  Vector / hybrid search · Graph algorithms (CALL algo.*)             │
├─────────────────────────────────────────────────────────────────────┤
│  Paged graph/property SSTs · Clustered ANN · Block-Max full text     │
├─────────────────────────────────────────────────────────────────────┤
│  Graph LSM · Search-LSM · WAL · Manifest CAS                         │
│  Bounded range cache (RAM + optional NVMe) · bounded workspaces      │
├─────────────────────────────────────────────────────────────────────┤
│  S3 · R2 · GCS · Azure Blob · MinIO · Tigris · Local FS             │
└─────────────────────────────────────────────────────────────────────┘
```

Design proposals live in [`docs/rfc/`](./docs/rfc/) — start with [RFC-001](./docs/rfc/001-storage-engine.md) (storage engine) and [RFC-002](./docs/rfc/002-sst-format.md) (SST format).

<br />

## Configuration

The defaults cover ordinary graph workloads. Size search-index memory
explicitly for large vector or full-text corpora. The durability, range-read,
cache, and admission guarantees are specified in the
[object-native storage contract](./docs/architecture/object-native-storage.md).

| Env var | Default | What it does |
|---|---|---|
| `NAMIDB_CACHE_MAX_BYTES` | `1073741824` (1 GiB) | Process-wide admission ceiling for cache-accounted payloads and metadata shared by SST, decoded, object-range page, node-view, and adjacency caches. `0` disables all shared caches; malformed values fail server startup. |
| `NAMIDB_SEARCH_INDEX_CACHE_MAX_BYTES` | unset (proportional) | Exact-byte reservation, carved out of `NAMIDB_CACHE_MAX_BYTES`, for one shared decoded `.vg`/`.ft` eviction pool. Empty means unset; malformed values fail server startup. |
| `NAMIDB_CACHE_PATH_REGISTRY_BUDGET_MIB` | `32` | Requested ceiling for SST path/admission metadata, including resident-path tracking and authoritative manifest rules. It is scaled inside `NAMIDB_CACHE_MAX_BYTES`; natural Foyer evictions remove their metadata synchronously, and exhaustion fails cache admission closed. |
| `NAMIDB_RAM_PAGE_CACHE_MAX_BYTES` | `0`; `67108864` (64 MiB) when both NVMe settings are present | Requested RAM assignment for immutable object-range pages, carved out of `NAMIDB_CACHE_MAX_BYTES` rather than added to it. Every paged read path — vector, full-text, property index and edge adjacency — serves point lookups from small ranges and benefits from this tier. In hybrid mode this includes both Foyer write buffers and a conservative bound for its persistent hash index; only the remainder is resident page capacity. |
| `NAMIDB_NVME_CACHE_PATH` | unset | Enables the persistent local range-cache tier. Must be paired with `NAMIDB_NVME_CACHE_MAX_BYTES`; the directory is process-exclusive and reconstructible from object storage. |
| `NAMIDB_NVME_CACHE_MAX_BYTES` | unset | Exact Foyer filesystem-device capacity. It is independent of the RAM ceiling and never defaults to a percentage of the filesystem. |
| `NAMIDB_RANGE_CACHE_NAMESPACE` | `memory-only`; required with NVMe | Stable, non-secret deployment + bucket/store identity mixed into every persistent cache key. A persistent cache refuses to start without it, preventing cross-bucket hits when an NVMe path is reused. |
| `NAMIDB_RANGE_CACHE_PAGE_BYTES` | `262144` (256 KiB) | Aligned remote-read/cache page. Overlapping reads of the same generation coalesce onto the same single-flight page key. |
| `NAMIDB_RANGE_CACHE_MAX_ENTRY_BYTES` | `4194304` (4 MiB) | Largest immutable range admitted to RAM/NVMe. |
| `NAMIDB_NVME_CACHE_BLOCK_BYTES` | `16777216` (16 MiB) | Foyer disk eviction unit and serialized-entry ceiling. The device must hold at least eight blocks. |
| `NAMIDB_NVME_CACHE_WRITE_BUFFER_BYTES` | one quarter of the effective page-cache RAM, capped at 16 MiB | Size of each Foyer active/rotating disk write buffer. Two such buffers are deducted from the page-cache RAM assignment rather than added to it. |
| `NAMIDB_SEARCH_WORKSPACE_MAX_BYTES` | `268435456` (256 MiB) | Process-wide fair semaphore for compressed ranges, decoded posting/vector pages, candidate heaps and other transient object-native search memory. Queries wait behind the shared cap; a single query that cannot fit fails explicitly without reducing accuracy. |
| `NAMIDB_SEARCH_MAX_RESULT_BYTES` | `67108864` (64 MiB) | Materialised-result ceiling for legacy unbounded (`k = None`) FTS APIs. Crossing it is an explicit error, never silent truncation; finite `k` stays bounded by its heap. |
| `NAMIDB_BM25_MAX_DOCUMENT_BYTES` | `1048576` (1 MiB) | Maximum UTF-8 bytes across one document's configured string fields in the exact BM25 fallback. Tokenization is streaming, but the unique-term map reserves a conservative 160× document allowance together with result memory; raise this and `NAMIDB_SEARCH_WORKSPACE_MAX_BYTES` together with matching process headroom. |
| `NAMIDB_MEMORY_MAX_BYTES` | `0` (disabled); `auto` in the official Compose example | Server-only total RSS/working-set admission ceiling. An exact byte count is accepted; `auto` selects 90% of a finite cgroup limit and fails startup if no hard limit exists. A 500 ms watchdog clears reconstructible caches at 90% of the resulting rail even without incoming work; at the ceiling new Cypher receives a retryable 503/Bolt transient error until memory falls. |
| `NAMIDB_BOLT_MEMORY_BUDGET_BYTES` | half of `NAMIDB_MEMORY_MAX_BYTES`, or `1073807360` (~1 GiB) when that governor is disabled | Process-wide admission budget shared by framed, decoded, converted and prefetched authenticated Bolt data. An incomplete frame holds `64 KiB + 2 × wire body bytes`; at its terminator a data frame must atomically upgrade to `64 KiB + 16 × wire body bytes` or fail retryably. Use smaller batches or raise this only with matching process/cgroup headroom. |
| `NAMIDB_BOLT_PARTIAL_MESSAGE_TIMEOUT` | `120s` | Deadline from the first byte of an authenticated partial frame through framing/budget admission. `0s` disables it; a completely idle connection holds no message-memory permit and has no partial-frame deadline. |
| `NAMIDB_CORRELATED_WRITE_CHUNK_ROWS` | `128` (hard-clamped to `1024`) | Existing-node rows point-probed and hydrated at once for a write-only `UNWIND … MATCH (n {unique_key: …}) SET …`. The request parameters remain caller-owned, while old wide node values (including embeddings) are released one bounded chunk at a time. |
| `NAMIDB_SPOOL_DIR` | `/var/tmp` on Unix; native temp directory elsewhere; `/var/tmp/namidb-spool` in the official image | Disk directory for corpus-sized exact-node values and remote compaction inputs. Size it for all compacted inputs plus the new Parquet and exact-record outputs (roughly 3× the compacted live node bytes; commonly 12–15 GiB per million 1024d nodes, with extra headroom for superseded versions). |
| `NAMIDB_INDEX_BUILD_SPOOL_DIR` | compaction spool, then `NAMIDB_SPOOL_DIR` | Dedicated disk directory for external FTS/vector sort runs and immutable output bodies. |
| `NAMIDB_INDEX_BUILD_MEMORY_BYTES` | `268435456` (256 MiB) on the compaction rebuild path; `67108864` (64 MiB) for per-flush Search-LSM delta builders | Aggregate logical memory ceiling used by external search-index builders; corpus-sized state spills to the index spool. One explicit setting overrides both paths, so size it for the larger full-corpus rebuild. |
| `NAMIDB_SEARCH_LSM_MAX_SEGMENTS` | `8` | Hard live-segment cap per active vector/full-text generation. Values must be at least `2` and are capped at `32`. A flush that needs another physical delta is backpressured until compaction frees a slot; an exactly proven-empty event needs no slot. |
| `NAMIDB_SEARCH_LSM_COMPACT_SEGMENTS` | `max(MAX_SEGMENTS - 2, 2)` | Number of complete adjacent Delta segments consolidated from the physical tail in one routine delta-only pass, clamped to the hard cap. This keeps ordinary maintenance proportional to changed IDs instead of rebuilding the corpus. |
| `NAMIDB_SEARCH_LSM_BASE_COMPACT_BYTES` | `8589934592` (8 GiB) | Triggers a full authoritative Base rebuild when accumulated **Delta** descriptor bytes reach the threshold. The existing Base size is deliberately excluded, so a large clean Base does not rebuild forever. Must be positive. |
| `NAMIDB_SEARCH_LSM_BASE_STALE_PERCENT` | `800` | Triggers a full Base rebuild when accumulated Delta mutations reach this percentage of the current live corpus (`800` = 8×). Must be positive. |
| `NAMIDB_SEARCH_LSM_FORCE_BASE_COMPACTION` | `false` | Boolean one-shot debt-repayment trigger. It rebuilds only when Delta/proven-empty debt exists; a clean singleton Base remains idle. A `ShadowOnly` segment always schedules Base repair independently of this flag. |
| `NAMIDB_SIDECAR_SORT_MEMORY_BYTES` | `8388608` (8 MiB) | Per-collector external-sort buffer for unique, equality and label sidecars (minimum 64 KiB). Checksummed runs merge with bounded fan-in; corpus-sized keys/postings live on the spool, not in RAM. |
| `NAMIDB_LEGACY_PROPERTY_INDEX_MAX_BYTES` | `0` (disabled) | Rolling-downgrade-only cap for also emitting the old monolithic bincode property sidecar. Current writers emit authoritative PagedV2 directly; a positive value enables the legacy mirror and fails the build if it would cross the cap. |
| `NAMIDB_ADJACENCY` | OFF (opt-in) | Enables the reconstructible CSR adjacency cache shared across snapshots. Leave unset for object-native deployments; enable explicitly only with a sized `NAMIDB_ADJACENCY_*` cache budget. |
| `NAMIDB_NODE_CACHE` | ON | Cross-snapshot `NodeView` lookup cache. |
| `NAMIDB_SNAPSHOT_NODE_CACHE_MAX_BYTES` | min(`1048576`, `NAMIDB_CACHE_MAX_BYTES / 64`) | Exact-byte ceiling for the short-lived node-view cache owned by one active snapshot/query. `0` disables this L1 without disabling the shared cache. |
| `NAMIDB_SNAPSHOT_ROW_GROUP_CACHE_MAX_BYTES` | min(`1048576`, `NAMIDB_CACHE_MAX_BYTES / 64`) | Exact-byte ceiling for decoded Parquet row groups retained by one snapshot when no shared SST cache is attached. At most two groups are retained; an oversized group is used once and released. |
| `NAMIDB_SNAPSHOT_EDGE_READER_CACHE_MAX_BYTES` | min(`1048576`, `NAMIDB_CACHE_MAX_BYTES / 64`) | Exact-byte ceiling for range-reader metadata retained by one graph snapshot, additionally capped at 32 immutable edge objects. Data pages remain in the shared RAM/NVMe range cache. |
| `NAMIDB_SST_CACHE` | ON | Raw SST body and decoded SST cache tiers. |
| `NAMIDB_FACTORIZE` | OFF | Factorized intermediate results in the executor. |
| `NAMIDB_VECTOR_FILTER_MAX_DISTINCT` | `4096` | Maximum distinct String/Bool values materialised per property in one `.vg`; crossing omits that property atomically. |
| `NAMIDB_VECTOR_FILTER_MAX_BYTES` | `67108864` (64 MiB) | Per-`.vg` budget for adaptive ordinal filter postings. |
| `NAMIDB_VECTOR_FILTER_ID_CANDIDATE_CAP` | `8192` | Maximum complete sidecar IDs used lazily when a `.vg` cannot apply any filter group; `0` disables this fallback. |
| `NAMIDB_HYBRID_TEXT_FILTER_CANDIDATE_CAP` | `65536` | Maximum authoritative FTS candidates hydrated while refilling a filtered hybrid sparse leg; reaching it switches to the exact flat fallback. |
| `NAMIDB_EDGE_POINT_MAX_ENTRY_BYTES` | `0` (unlimited) | Optional hard ceiling for one complete relationship record in an exact `(source,target)` point sidecar. A positive limit is enforced explicitly by flush/compaction; crossing it fails the build rather than silently omitting `.epidx`. |
| `NAMIDB_EDGE_POINT_MAX_SST_BYTES` | `0` (unlimited) | Optional hard ceiling for one complete exact-edge sidecar per forward SST. A positive limit is enforced explicitly by flush/compaction; crossing it fails the build rather than silently dropping the accelerator. |
| `NAMIDB_EDGE_DECODE_MAX_BYTES` | `67108864` (64 MiB) | Maximum materialised adjacency bytes for a single legacy partner block. Larger supernodes require paged expansion; corrupt degrees are rejected before allocation. |
| `NAMIDB_EMBED_PROVIDER` | unset | Remote embedder for `load-vault --embed` (`openai`/`voyage`/`cohere`/`gemini`/`jina`; needs `--features remote-embedder`). |

The caches are **process-wide** — one shared set across every namespace, so a
busy tenant cannot multiply the memory limit. `NAMIDB_CACHE_MAX_BYTES` is the
aggregate hard-admission ceiling for cache-accounted entry weights. Allocator
and hash-table costs are included through conservative per-entry weights, and
SST path/manifest metadata has its own bounded tier. Values retained by an
active query after cache eviction remain request working-set bytes rather than
cache residency. The legacy per-tier knobs remain compatible ceilings; when
their sum is larger than the aggregate maximum, NamiDB scales every active
ceiling proportionally and deterministically. No single cache entry is
admitted when its deep estimated weight exceeds its assigned tier.

Current V5/VG6 vector and FT3/FT4 full-text readers do **not** put the corpus in
that decoded-index pool. They retain bounded routing/dictionary metadata and
fetch immutable pages through the shared RAM/NVMe range cache while charging
candidate heaps and decoded pages to `NAMIDB_SEARCH_WORKSPACE_MAX_BYTES`.
Consequently a ten-million-vector namespace does not require a
ten-million-vector RAM cache: smaller cache/workspace settings trade warm
latency and concurrency for more range reads, not correctness. Literally zero
RAM is still impossible because active query state, network buffers, the
memtable, and bounded metadata must live somewhere.

`NAMIDB_SEARCH_INDEX_CACHE_MAX_BYTES` remains the shared pool for legacy
monolithic `.vg`/`.ft` readers. Unused text capacity is available to vectors
and vice versa; an explicit reservation is carved out of, and never increases,
`NAMIDB_CACHE_MAX_BYTES`. Admission estimates serialized expansion before a
full-body GET/decode. If a valid monolithic index does not fit, NamiDB returns
HTTP 503 with `code: "search_index_cache_capacity"` or a retryable Bolt
storage error instead of silently selecting an `O(corpus)` scan. Missing,
stale, legacy-unsupported, or corrupt optional indexes retain their
correctness-preserving exact fallback.

Keep additional headroom outside the cache budget for memtables, active
queries, compaction, allocator metadata, and the RSS admission rail described
below. The legacy knobs are
`NAMIDB_SST_CACHE_BUDGET_MIB` (256), `NAMIDB_DECODED_NODE_RG_CACHE_BUDGET_MIB`
(256), `NAMIDB_SST_METADATA_CACHE_BUDGET_MIB` (64),
`NAMIDB_EDGE_STREAM_CACHE_BUDGET_MIB` (256),
`NAMIDB_EDGE_READER_CACHE_BUDGET_MIB` (256),
`NAMIDB_PROPERTY_SIDECAR_CACHE_BUDGET_MIB` (512),
`NAMIDB_BLOOM_FILTER_CACHE_BUDGET_MIB` (64),
`NAMIDB_TEXT_INDEX_CACHE_BUDGET_MIB` (512), and
`NAMIDB_VECTOR_INDEX_CACHE_BUDGET_MIB` (512),
`NAMIDB_CACHE_PATH_REGISTRY_BUDGET_MIB` (32), plus the existing
`NAMIDB_NODE_CACHE_*` and `NAMIDB_ADJACENCY_*` pools. Legacy values are MiB;
the two search-index requests are combined into the shared pool when no
explicit reservation is set. `NAMIDB_CACHE_MAX_BYTES` and
`NAMIDB_SEARCH_INDEX_CACHE_MAX_BYTES` are expressed as exact bytes.
This ceiling covers retained cache entries; memtables, active query rows,
transient flush/compaction work, and writer-local reconstructible
property/uniqueness maps are separate. Those writer-local maps are not charged
to `NAMIDB_CACHE_MAX_BYTES`; an RSS pressure pass clears them
through weak registrations, without extending a namespace's lifetime. The
official image also bounds glibc allocation arenas so dropped large-index
working sets are returned instead of accumulating per-thread arenas.
Authenticated Bolt framing is admitted before a large body allocation or
PackStream decode. A partial frame grows a fail-fast raw-memory lease at
`64 KiB + 2× wire bytes`; once its terminator arrives, a non-control frame must
atomically upgrade that same lease to the `64 KiB + 16× wire bytes` decoded
working set. The shared `NAMIDB_BOLT_MEMORY_BUDGET_BYTES` lease follows the
message through parameter conversion, execution and RUN prefetch. An early
RSS admission also takes a process-wide RAII reservation for that projected
headroom through request handling, followed by the ordinary execution-time
check to cover unrelated allocations.
Small bounded pressure-relief controls (`PULL`, `DISCARD`, `COMMIT`,
`ROLLBACK`, `RESET`, `GOODBYE`, `LOGOFF`) bypass data admission so clients can
drain results, release a transaction or recover under pressure. No task waits
while holding partial permits, and the partial-frame timeout releases a
slowloris's bounded raw-memory lease.
Node Parquet outputs, exact-record values and remote compaction inputs stream
through anonymous files in `NAMIDB_SPOOL_DIR`; only bounded Arrow batches,
compact B+tree pages and multipart windows remain in RAM. The files are
synced before mmap/upload so delayed allocation errors surface early and dirty
pages can be reclaimed, then unlinked/removed automatically on success,
failure, or process exit. Corpus-sized flush builds are process-wide
single-flight even if their caller disconnects. Do not
point this at a RAM-backed `/tmp`/`tmpfs` for large vector corpora. During a
full-backlog node compaction, every mapped input coexists with the new Parquet
output and its exact-record value spool. Provision local disk for
`sum(inputs) + parquet_output + exact_record_output` — roughly 3× the compacted
live node bytes, plus headroom for superseded versions. The official image
defaults to `/var/tmp/namidb-spool`, and the example Compose file mounts a
dedicated volume there.
Current node SSTs pair Parquet with a rollback-compatible `.nloc2` sidecar:
the 2.0.5 `NodeId → row ordinal` tree remains its prefix, followed by a
range-readable `NodeId → compressed exact record` tree. Point updates fetch
that record instead of hydrating an unrelated wide Parquet page. Compaction
maps both local inputs and remote spools, activates each source only when its
manifest key range reaches the merge frontier, and retains at most 64 decoded
rows per active node source. Upgrading a settled 2.0.5 SST can therefore attach
only `.nloc2`; fresh `.vg` vector and `.ft` full-text generations, their object
IDs and durable build markers remain unchanged.
For a process-wide server safety rail, set `NAMIDB_MEMORY_MAX_BYTES` above the
cache budget and expected memtable/compaction headroom. The value may be an
exact byte count or `auto`; the latter takes 90% of a finite cgroup v2/v1 hard
limit and refuses to start without one, rather than guessing from shared host
RAM. It measures RSS on
Linux/macOS and working set on Windows, reclaims shared caches and writer-local
reconstructible maps at 90%, and stops admitting new Cypher work at the
configured byte ceiling. A process-wide 500 ms watchdog performs the same
single-flight reclamation while the server is idle or a background
flush/compaction is running; reclamation no longer waits for the next request.
Authenticated `POST /v0/admin/flush` remains available as a pressure-relief
operation and is serialized process-wide. The route is outside the ordinary
HTTP timeout and the storage task survives client disconnection, so an
operator cannot strand a memtable halfway through a relief flush; the global
request cap still bounds callers waiting for it. In multi-tenant mode, a
hard-pressure flush may target an already-open namespace, but will not recover
a cold namespace that has no live memtable to release. Flush and compaction
can temporarily amplify memory, so this rail does not replace an OS-enforced
container/cgroup limit or correctly-sized headroom for an already-running
operation. The official Compose example supplies a 4 GiB hard limit and uses
`auto`, so both safeguards are active by default there. The setting belongs to
`namidb-server`; embedded Rust and Python
clients do not install this admission governor. For example, pair
`NAMIDB_MEMORY_MAX_BYTES=3758096384` (3.5 GiB admission) with
`docker run --memory=4g ...` (4 GiB hard containment).

One immutable manifest body and one versioned pointer are intentionally
written per durable commit. During a long load, hundreds of small versions
(for example, 345 versions / 36 MiB) are therefore normal temporary history,
not an unbounded leak. The background janitor removes manifest and pointer
versions below the oldest live-reader horizon once
`NAMIDB_SWEEP_MIN_AGE` has elapsed (24 h by default), provided periodic
maintenance is enabled (`NAMIDB_COMPACTION_INTERVAL` is non-zero) and
`NAMIDB_SWEEP_DELETE=true`. With deletion disabled the same pass is dry-run
only. The horizon protects readers exactly; the age window protects a body
uploaded just before its CAS, so lower it only when that object-store race
window is well understood.

The server also takes durability/backpressure knobs for critical workloads:

| Flag (env var) | Default | What it does |
|---|---|---|
| `--memtable-flush-bytes` (`NAMIDB_MEMTABLE_FLUSH_BYTES`) | 64 MiB | Flush as soon as a committed write crosses this — bounds the un-flushed working set by bytes, not just the flush interval. |
| `--memtable-stall-bytes` (`NAMIDB_MEMTABLE_STALL_BYTES`) | 256 MiB | Soft write backpressure so a burst loader can't OOM the process between flushes. |
| `--writer-lock-timeout` (`NAMIDB_WRITER_LOCK_TIMEOUT`) | 30s | Cap how long a foreground write/DDL waits for the writer; a stuck writer returns 503 fast instead of growing an unbounded queue. |

A fenced or poisoned writer reopens itself automatically, and `/v0/health` reports `writer: degraded` (503) until it recovers — a rolling deploy or an accidental second replica on the same bucket drains cleanly instead of serving stale reads behind a green check.

<br />

## Repository layout

```
crates/
├── namidb-core/        # Common types, errors, schema
├── namidb-storage/     # LSM, WAL, manifest CAS, SST, URI parser, file:// CAS
├── namidb-graph/       # Property columns, CSR adjacency, graph algorithm kernels
├── namidb-ann/         # DiskANN / Vamana vector index
├── namidb-query/       # Cypher / GQL parser, optimizer, executor, BM25
├── namidb-bolt/        # Bolt wire protocol (PackStream, handshake, state machine)
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
| **Engine internals (technical report)** | [`docs/ARCHITECTURE.md`](./docs/ARCHITECTURE.md) |
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
- The Change License is **Apache License 2.0**. Conversion occurs on
  **May 18, 2029**, or on the fourth anniversary of a specific version's first
  public distribution if that is earlier, exactly as stated in [`LICENSE`](LICENSE).
- A separate commercial license is available if you need to embed or redistribute NamiDB outside what BSL allows, including running it as a hosted database service. Reach us at [`info@namidb.com`](mailto:info@namidb.com).

<br />

---

<div align="center">

<img src=".assets/logo_namidb.png" alt="NamiDB" width="120" />

### The bucket is the database.

<sub>NamiDB is built by <a href="https://namidb.com"><b>NamiDB, Inc.</b></a>, Delaware, USA.</sub><br />
<sub>© 2026 NamiDB, Inc. All rights reserved.</sub>

</div>
