# namidb-query

Cypher / GQL parser, logical-plan IR, cost-based optimizer and
vectorized morsel-driven executor for
[NamiDB](https://github.com/namidb/namidb).

The pipeline is:

1. **Parser** — Cypher source text → AST
   ([RFC-004](../../docs/rfc/004-cypher-subset.md)).
2. **Lowering** — AST → `LogicalPlan`
   ([RFC-008](../../docs/rfc/008-logical-plan-ir.md),
   [RFC-009](../../docs/rfc/009-write-clauses.md)).
3. **Cost-based optimizer** — `LogicalPlan` → optimised
   `LogicalPlan`. Predicate pushdown
   ([RFC-011](../../docs/rfc/011-predicate-pushdown.md)),
   projection pushdown
   ([RFC-015](../../docs/rfc/015-projection-pushdown.md)), join
   reorder ([RFC-016](../../docs/rfc/016-join-reorder.md)), hash join
   conversion ([RFC-012](../../docs/rfc/012-hash-join.md)), hash
   semi-join ([RFC-014](../../docs/rfc/014-hash-semi-join.md)),
   Parquet row-group pruning
   ([RFC-013](../../docs/rfc/013-parquet-predicate-pushdown.md)).
4. **Executor** — morsel-driven, optionally factorized
   ([RFC-017](../../docs/rfc/017-factorization.md)), with `EXPLAIN`
   and `EXPLAIN VERBOSE` support.

Public surface (top-level): `parse`, `lower`, `execute`, `Params`,
`Row`, `RuntimeValue`. End-to-end coverage targets the 12 in-scope
LDBC SNB Interactive Complex Read queries (IC01–IC12).

See the [NamiDB README](../../README.md) for the project overview and
the [RFCs](../../docs/rfc/) for design rationale.

## License

[Business Source License 1.1](../../LICENSE) — © Fonles Studios, Corp.
