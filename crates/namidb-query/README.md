# namidb-query

The Cypher and GQL parser, the logical-plan IR, the cost-based optimizer
and the vectorized morsel-driven executor for
[NamiDB](https://github.com/namidb/namidb).

The pipeline goes:

1. **Parser.** Cypher source text into an AST
   ([RFC-004](../../docs/rfc/004-cypher-subset.md)).
2. **Lowering.** AST into a `LogicalPlan`
   ([RFC-008](../../docs/rfc/008-logical-plan-ir.md),
   [RFC-009](../../docs/rfc/009-write-clauses.md)).
3. **Cost-based optimizer.** `LogicalPlan` into a better `LogicalPlan`:
   predicate pushdown
   ([RFC-011](../../docs/rfc/011-predicate-pushdown.md)), projection
   pushdown ([RFC-015](../../docs/rfc/015-projection-pushdown.md)), join
   reorder ([RFC-016](../../docs/rfc/016-join-reorder.md)), hash-join
   conversion ([RFC-012](../../docs/rfc/012-hash-join.md)), hash
   semi-join ([RFC-014](../../docs/rfc/014-hash-semi-join.md)), and
   Parquet row-group pruning
   ([RFC-013](../../docs/rfc/013-parquet-predicate-pushdown.md)).
4. **Executor.** Morsel-driven, optionally factorized
   ([RFC-017](../../docs/rfc/017-factorization.md)), with `EXPLAIN` and
   `EXPLAIN VERBOSE`.

The public surface (top-level) is `parse`, `lower`, `execute`, `Params`,
`Row`, `RuntimeValue`. End-to-end coverage targets the 12 in-scope LDBC
SNB Interactive Complex Read queries (IC01 through IC12).

See the [NamiDB README](../../README.md) for the project overview and
the [RFCs](../../docs/rfc/) for the design rationale.

## License

[Business Source License 1.1](../../LICENSE), © LESAI, Corp.
