# RFC 036: First-class query beam-width (`ef`) surface

**Status:** draft
**Author(s):** NamiDB team
**Created:** 2026-06-26
**Updated:** 2026-06-26
**Amends:** [RFC-030](030-vector-index.md) — replaces the non-stable
`$__vector_ef` bridge with a first-class beam-width surface for the natural
(operator-form) vector search.

## Summary

The Vamana beam width `ef` trades recall for latency. Today it is reachable on
two query surfaces, but never on the same query as a *filter* in a stable,
first-class way:

- **Procedures** (`search.vector`, `search.hybrid`, `db.index.vector.queryNodes`)
  take an `ef` map key *and* (since the RFC-030 change set) a `filter` key, so
  the procedure surface already composes filtered ANN with a tunable beam.
- **Natural form** (`MATCH (n:L) … WITH cosine_similarity(n.embedding, $q) AS
  score WHERE … ORDER BY score DESC LIMIT k`) is the surface the optimizer
  rewrites into `LogicalPlan::VectorSearch` and the *only* one that carries a
  real `post_filter` (RFC-030 §7). It exposes the beam through a deliberately
  **non-stable** reserved parameter, `$__vector_ef`, read in `flat_vector_search`
  (`crates/namidb-query/src/exec/walker.rs`) and dropped into the already-present
  `ef_search` slot of `vector_search_rows`.

`$__vector_ef` is a bridge, not a destination: it is global to the query (every
`VectorSearch` node sees the same value), undiscoverable, and namespaced only to
avoid clashing with a user's own `$ef`. This RFC proposes the stable surface that
supersedes it.

## Motivation

Retrieval-augmented and semantic-search workloads routinely need to raise recall
for a *specific* filtered query (e.g. a sparse multi-tenant slice) without
globally widening the engine default. The procedure surface already supports
this; the natural form — which the MCP semantic/hybrid tools and most
hand-written Cypher use — does not, except through an internal parameter we have
explicitly documented as unstable. A first-class surface lets a query say "use a
wider beam here" in a way that is discoverable, per-operator, and survives schema
review, while keeping the strong RFC-030 guarantee that a larger `ef` only ever
*raises* recall (it can never change correctness: the over-fetch, the residual
`post_filter`, and the exact flat fallback are unchanged).

## Design

### Surface syntax

Add an `OPTIONS { … }` clause that attaches to a reading query and carries
engine hints, the first being `ef`:

```cypher
MATCH (d:Doc)
WHERE d.tenant_id = $t
RETURN d.title, cosine_similarity(d.embedding, $q) AS score
ORDER BY score DESC
LIMIT 10
OPTIONS { ef: 200 }
```

`OPTIONS` reuses the map-literal shape the DDL `CREATE VECTOR INDEX … WITH { r,
l_build, alpha }` form already parses, so the lexer/parser delta is small and the
syntax is familiar. Unknown option keys are rejected (not silently ignored) so a
typo is an error rather than a no-op. `ef` must be a positive integer or an
integer `$param`.

Alternatives weighed (see *Alternatives considered*): a Neo4j-style planner-hint
comment, and a `USING VECTOR INDEX … EF n` clause. `OPTIONS { … }` wins on the
smallest grammar change and reuse of the existing map parser.

### Plan and execution plumbing

1. **AST/parser** — parse the trailing `OPTIONS { … }` into a small
   `QueryOptions { ef: Option<RowCount> }` carried on the query/return structure.
   `RowCount` (already `Const(u64) | Param(String)`) covers literal and
   parameterized `ef`.
2. **Logical plan** — add `ef: Option<RowCount>` to `LogicalPlan::VectorSearch`
   (mirroring its existing `k: RowCount`). The optimizer's KNN rewrite
   (`crates/namidb-query/src/optimize/vector_search.rs`) populates it from the
   parsed options; because the `VectorSearch` node is registered as an opaque
   leaf after `unique_lookup`, downstream pushdowns already leave it intact.
3. **Executor** — both `VectorSearch` dispatch sites in
   `crates/namidb-query/src/exec/walker.rs` forward the logical `ef` into
   `flat_vector_search → vector_search_rows(ef_search = …)`. The executor reads
   the logical field first and falls back to the `$__vector_ef` parameter only
   while the bridge is being retired, so both can coexist during migration.
4. **EXPLAIN / cost** — `plan/explain.rs` renders `ef` on the `VectorSearch`
   node; `cost/cardinality.rs` is unchanged (`ef` affects latency/recall, not
   cardinality).

### Interaction with the existing guarantees

- `ef` is clamped to `≥ kprime` in `try_index_search` (RFC-030 §7), so a small
  `OPTIONS { ef }` cannot *narrow* the beam below what the over-fetch needs; it
  can only widen it.
- A larger `ef` raises recall and latency; it never changes the *answer*, because
  the residual `post_filter`, the adaptive widening, and the exact flat fallback
  remain the ground truth.
- `OPTIONS { ef }` and a `WHERE` filter compose on one query — the central gap
  this RFC closes for the natural form (the procedure surface already composes
  them).

### Migration

`$__vector_ef` stays as a documented, non-stable fallback for one release after
`OPTIONS { ef }` lands (the executor prefers the logical field, then the param).
RFC-030 §"Open questions" tracks whether to retire it hard or keep it during
migration.

## Alternatives considered

- **Keep `$__vector_ef` only.** Zero further work, but a magic global parameter
  is not a surface: undiscoverable, query-global, and easy to typo into silence.
  Rejected as the end state; kept as the migration bridge.
- **Planner-hint comment** (`/*+ VECTOR_EF(200) */`). Familiar to Postgres/Neo4j
  users, but the parser has no comment-hint infrastructure, and hints-in-comments
  are easy to drop on reformat. More plumbing for less clarity.
- **`USING VECTOR INDEX … EF n`.** Explicit and index-scoped, but a larger
  grammar addition and it conflates index *selection* with beam *sizing*.
- **Per-procedure only.** Declaring the procedure `ef` the supported surface and
  leaving the natural form without one. Rejected because the natural form is the
  only filtered-ANN surface and the one most Cypher and the MCP tools use.

## Drawbacks

- Adds a query-level `OPTIONS` surface to the grammar/AST/lowering/optimizer/
  EXPLAIN chain — a real, if mechanical, plumbing cost across ~8 sites that
  destructure `VectorSearch` (some via `..`); both executor dispatch sites must
  forward `ef` or one path silently ignores it.
- `OPTIONS` is a general-purpose extension point; scoping it to a small, rejected-
  on-unknown key set keeps it from becoming a dumping ground.

## Open questions

- Should `OPTIONS` be query-global or attachable per-`MATCH`/per-`CALL`? A query
  with several `VectorSearch` nodes may want different beams. Per-clause is more
  expressive but a larger grammar change; query-global mirrors the current
  `$__vector_ef` semantics and is the proposed v1.
- Should other engine knobs (e.g. an over-fetch override, a "force flat" debug
  switch) live under the same `OPTIONS`? Likely yes, behind the same
  reject-unknown-keys discipline.

## References

- [RFC-030: DiskANN/Vamana vector index](030-vector-index.md) — the `ef_search`
  slot, the `$__vector_ef` bridge, and the over-fetch/flat-fallback guarantees
  this surface plugs into.
- [RFC-004: Cypher subset](004-cypher-subset.md) — the grammar this `OPTIONS`
  clause extends.
