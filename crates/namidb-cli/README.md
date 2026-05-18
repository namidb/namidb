# namidb-cli

The `namidb` command-line tool. Wraps the engine for ad-hoc query
work: parse, explain, run.

## Install

From source:

```bash
git clone https://github.com/namidb/namidb.git
cd namidb
cargo install --path crates/namidb-cli
```

The resulting `namidb` binary needs no daemon and no configuration —
it spins up an in-memory namespace for one-shot work.

## Usage

```bash
# Show the canonical form of a query (lexer + parser round-trip).
namidb parse "MATCH (a:Person)-[:KNOWS]->(b:Person) RETURN b.name LIMIT 5"

# Show the optimised logical plan with cost / selectivity annotations.
namidb explain --verbose \
  "MATCH (a:Person)-[:KNOWS]->(b) RETURN b ORDER BY b.id LIMIT 20"

# Run a query against an ephemeral in-memory namespace.
namidb run "CREATE (a:Person {id: 'alice', name: 'Alice'}), \
            (b:Person {id: 'bob',   name: 'Bob'}), (a)-[:KNOWS]->(b)"

namidb run "MATCH (a:Person)-[:KNOWS]->(b) RETURN a.name, b.name"
```

See the [NamiDB README](../../README.md) for the project overview and
the [RFCs](../../docs/rfc/) for design rationale.

## License

[Business Source License 1.1](../../LICENSE) — © Fonles Studios, Corp.
