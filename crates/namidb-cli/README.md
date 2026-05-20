# namidb-cli

The `namidb` command-line tool. It wraps the engine for ad-hoc query
work (parse, explain, run) against any supported storage backend.

## Install

From source:

```bash
git clone https://github.com/namidb/namidb.git
cd namidb
cargo install --path crates/namidb-cli
```

The resulting `namidb` binary needs no daemon. Without `--store` it
spins up an ephemeral in-memory namespace for one-shot work. With
`--store <uri>` it opens a durable namespace on any supported backend
(`file://`, `s3://`, `gs://`, `az://`, `memory://`).

## Usage

```bash
# Show the canonical form of a query (lexer + parser round-trip).
namidb parse "MATCH (a:Person)-[:KNOWS]->(b:Person) RETURN b.name LIMIT 5"

# Show the optimised logical plan with cost and selectivity annotations.
namidb explain --verbose \
  "MATCH (a:Person)-[:KNOWS]->(b) RETURN b ORDER BY b.id LIMIT 20"

# Run a query against an ephemeral in-memory namespace.
namidb run \
  "CREATE (a:Person {name: 'Alice'}), (b:Person {name: 'Bob'}), (a)-[:KNOWS]->(b)"
namidb run "MATCH (a:Person)-[:KNOWS]->(b) RETURN a.name, b.name"

# Same queries, but durable on a local directory.
namidb run --store "file:///var/lib/namidb?ns=prod" \
  "CREATE (a:Person {name: 'Alice'}), (b:Person {name: 'Bob'}), (a)-[:KNOWS]->(b)"
namidb run --store "file:///var/lib/namidb?ns=prod" \
  "MATCH (a:Person)-[:KNOWS]->(b) RETURN a.name, b.name"

# Or against any cloud bucket.
namidb run --store "s3://my-bucket/data?ns=prod&region=us-east-1" \
  "MATCH (p:Person) RETURN count(*) AS n"
namidb run --store "gs://my-bucket?ns=prod" \
  "MATCH (p:Person) RETURN count(*) AS n"
namidb run --store "az://acct/container?ns=prod" \
  "MATCH (p:Person) RETURN count(*) AS n"
```

The full URI grammar (endpoint overrides for R2, MinIO and LocalStack,
GCS service-account paths, Azure emulator mode) lives in the
[project README](../../README.md#pick-your-storage-backend) and in
[`namidb-storage::uri`](../namidb-storage/src/uri.rs).

See the [NamiDB README](../../README.md) for the project overview and
the [RFCs](../../docs/rfc/) for the design rationale.

## License

[Business Source License 1.1](../../LICENSE), © LESAI, Corp.
