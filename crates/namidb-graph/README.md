# namidb-graph

Property-graph layer on top of [`namidb-storage`](../namidb-storage/):
node and relationship views, label / type catalogues, and the CSR
adjacency materialisation used by the query engine.

This crate sits between the raw storage substrate and the query
engine. It does not parse Cypher and does not plan; for that, see
[`namidb-query`](../namidb-query/).

## What lives here

- `NodeView`, `EdgeView`, `RelView` — the property-graph value types
  exposed to readers.
- CSR adjacency materialisation per
  `(manifest_version, edge_type, direction)`
  ([RFC-018](../../docs/rfc/018-csr-adjacency.md)).
- Schema bridges between the storage-layer manifest and the
  graph-level surface (labels, edge types, declared property columns).

See the [NamiDB README](../../README.md) for the project overview.

## License

[Business Source License 1.1](../../LICENSE) — © Fonles Studios, Corp.
