# namidb-graph

The property-graph layer on top of [`namidb-storage`](../namidb-storage/):
node and relationship views, the label and type catalogues, and the CSR
adjacency materialisation the query engine runs on.

This crate sits between the raw storage substrate and the query engine.
It doesn't parse Cypher and it doesn't plan; that's
[`namidb-query`](../namidb-query/).

## What lives here

- `NodeView`, `EdgeView`, `RelView`: the property-graph value types
  readers see.
- CSR adjacency materialisation per
  `(manifest_version, edge_type, direction)`
  ([RFC-018](../../docs/rfc/018-csr-adjacency.md)).
- The schema bridges between the storage-layer manifest and the
  graph-level surface (labels, edge types, declared property columns).

See the [NamiDB README](../../README.md) for the project overview.

## License

[Business Source License 1.1](../../LICENSE), © LESAI, Corp.
