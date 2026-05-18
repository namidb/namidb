# namidb-core

Foundational types, identifiers, schema primitives and error definitions
shared by every other crate in the [NamiDB](https://github.com/namidb/namidb)
workspace.

This crate is intentionally small and dependency-light. If you are
embedding NamiDB into your own application you usually want the
top-level [`namidb`](../namidb/) façade crate, not `namidb-core`
directly.

## What lives here

- `id` — `NamespaceId`, `NodeId`, `EdgeId`, `Lsn`, and the UUIDv7-based
  generation helpers.
- `value` — `Value` (the property value type) and the runtime variants
  used across storage, query and bindings.
- `schema` — label and property schemas; serialisation, comparison,
  type-promotion rules.
- `error` — shared error and `Result` types built with `thiserror`.
- `profile` — `profile_scope!` macros used by per-stage profiling
  (`NAMIDB_PROFILE_DUMP=1`).

See the [NamiDB README](../../README.md) for the project overview and
the [RFCs](../../docs/rfc/) for design rationale.

## License

[Business Source License 1.1](../../LICENSE) — © Fonles Studios, Corp.
