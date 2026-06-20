# namidb-core

The foundational types, identifiers, schema primitives and error
definitions that every other crate in the
[NamiDB](https://github.com/namidb/namidb) workspace shares.

It's deliberately small and light on dependencies. If you're embedding
NamiDB into your own application you almost always want the top-level
[`namidb`](../namidb/) façade crate, not `namidb-core` directly.

## What lives here

- `id`: `NamespaceId`, `NodeId`, `EdgeId`, `Lsn`, and the UUIDv7-based
  generation helpers.
- `value`: `Value` (the property value type) and the runtime variants
  used across storage, query and bindings.
- `schema`: label and property schemas, plus serialisation, comparison
  and type-promotion rules.
- `error`: the shared error and `Result` types, built on `thiserror`.
- `profile`: the `profile_scope!` macros behind per-stage profiling
  (`NAMIDB_PROFILE_DUMP=1`).

See the [NamiDB README](../../README.md) for the project overview and
the [RFCs](../../docs/rfc/) for the design rationale.

## License

[Business Source License 1.1](../../LICENSE), © NamiDB, Inc.
