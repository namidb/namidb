# RFC 023: `shortestPath` and `allShortestPaths`

**Status:** draft
**Author(s):** Matías Fonseca <info@namidb.com>
**Created:** 2026-05-24
**Updated:** 2026-05-24
**Implements:** (pending)
**Supersedes:** none

## Summary

Land `shortestPath((a)-[*..N]-(b))` and `allShortestPaths(...)` so the
two LDBC SNB Interactive queries that hang on them (IC13 single shortest
path, IC14 all shortest paths weighted) parse and execute end to end,
and any `MATCH p = shortestPath(...) RETURN p` query a Neo4j user
already writes works against NamiDB. Both forms compile down to a new
shape of the existing variable-length `Expand` operator that the
RFC-002 BFS already powers: walk the frontier hop by hop until the
first hop where the target appears, then stop and emit one or many
rows depending on the variant.

## Motivation

`shortestPath` is the most common path query a Neo4j user writes after
plain `MATCH`. Without it:

- LDBC SNB Interactive IC13 + IC14 are out of reach. The full IC1–IC14
  suite never finishes.
- Agent-memory and knowledge-graph use cases that rephrase a question
  as "shortest causal chain from A to B" don't have a primitive to
  reach for. Today they have to materialise every path with a bounded
  `*1..N` expand and post-filter by length, which blows up at any
  non-trivial graph.
- The Bolt + neo4j-driver path inherits the parser's hard rejection.
  A user typing `s.run("MATCH p = shortestPath((a)-[*..5]-(b))
  RETURN p")` against `bolt://namidb-server:7687` lands on
  `Neo.ClientError.Statement.NotSupported`, which is exactly the
  surface this RFC closes.

The cost of adding this is small. The variable-length `Expand` in
`crates/namidb-query/src/exec/walker.rs:603` already runs a hop-by-hop
BFS over the CSR adjacency cache; the new shape only changes when the
BFS *stops* and how many of the surviving rows it emits. No new
storage primitives, no new optimizer pass.

## Design

### Syntax

The two functions ride on top of the existing pattern grammar:

```
MATCH p = shortestPath( (a)-[r?:TYPE?*min..max?]-(b) ) ...
MATCH p = allShortestPaths( (a)-[r?:TYPE?*min..max?]-(b) ) ...
```

Rules enforced by the lower:

1. **Pattern must bind a path** (`p = ...`). The result row materialises
   `p` as `RuntimeValue::Path`; without a path binding the call has no
   useful effect.
2. **Both endpoints must be bound or have an inline filter that
   makes them unique.** Concretely: `a` is either already in scope
   from an earlier clause, or has a `{prop: value}` map with at
   least one unique-property predicate. Same for `b`. We reject
   anything else with a typed error
   (`E010_ShortestPathUnboundEndpoint`) so the user doesn't
   accidentally launch an all-pairs shortest-path scan.
3. **Relationship length must have a finite upper bound.** `*..` and
   `*1..` (open-ended) are rejected. The bench harness uses
   `*..15` in IC13, which is the conservative default we recommend.
4. **Exactly one relationship hop in the chain.** `shortestPath(
   (a)-[r1]-(b)-[r2]-(c))` is rejected; that's a multi-leg path and
   needs a different operator (RFC-pending; not in scope here).
5. **No relationship-type alternation inside `shortestPath`.** The
   wrapping function form still rejects `(a)-[:A|:B*]-(b)`. Plain
   `Expand` and cyclic `MultiwayJoin` learned alternation in
   RFC-024, but lifting it into the BFS path-binding executor is a
   separate diff (the path emission has to know which type each hop
   actually used).

Mirrors the surface Neo4j accepts for IC13:

```cypher
MATCH (n:Person {id: $personId}), (m:Person {id: $messageId})
MATCH p = shortestPath((n)-[:KNOWS*..15]-(m))
RETURN length(p) AS shortestPathLength
```

### LogicalPlan

Reuse `LogicalPlan::Expand`. The structure already carries
`length`, `back_reference`, `target_alias`, and the BFS in
`execute_expand` is exactly the shape we want — we only need to tell
it to stop and emit at the first hop the target appears.

Two options:

**A.** Add a new operator `LogicalPlan::ShortestPath { ... }`.
**B.** Add a `shortest: ShortestMode` enum to `Expand`.

We pick **B**. The plumbing — `back_reference` enforcement, edge-type
resolution, neighbour batching, cache pre-warm — is exactly the same.
A new operator would duplicate ~120 LoC for the sake of one extra
field. The enum encodes the variant directly:

```rust
pub enum ShortestMode {
    /// No shortest-path mode. The Expand emits every reachable
    /// path through the frontier (today's behaviour).
    None,
    /// `shortestPath(...)`: emit at most one row per (source, target)
    /// pair, the first time the target is reached.
    First,
    /// `allShortestPaths(...)`: at the first hop where the target
    /// appears, emit every distinct path of that length. Stop the
    /// BFS after that hop completes.
    All,
}
```

`LogicalPlan::Expand` gains one field:

```rust
Expand {
    ...
    /// `ShortestMode::None` for the regular `[*min..max]` expand;
    /// `First` / `All` for the shortest-path variants.
    shortest: ShortestMode,
}
```

### Executor

Modify `execute_expand` to honour `shortest`:

- `ShortestMode::None` — today's behaviour, unchanged.
- `ShortestMode::First` — record `found_at_hop: Option<u64>`. When
  the target id appears in the frontier at hop `H`, push the first
  matching row to `hop_results` and stop the loop after that hop
  completes (no further hops scanned, no more rows of length `H+`
  emitted). One row per (source, target) pair — even if multiple
  edges connect them at the same hop, take the first.
- `ShortestMode::All` — same `found_at_hop` tracking. At the
  identifying hop, push *every* matching row of that length, then
  break out of the outer hop loop. All-shortest-paths semantics
  match Neo4j's: multiple paths of the same minimum length all
  show up; longer paths don't.

The `back_reference` path is mandatory here because both endpoints
must be bound; the lower enforces that. So `existing_target_id` is
always `Some(...)`, and the BFS only flags a hit when the frontier
tail equals it.

### Path materialisation

The result row materialises `p` as `RuntimeValue::Path`. The path is
constructed from the BFS step trail: every `Step` carries its `row`
already; we only need to thread the visited node + relationship
sequence into a `RuntimeValue::Path`. The shape matches the existing
path-binding lowering (`build_path_constructor`).

### Manifest version + caching

The BFS reads from the CSR adjacency cache (`AdjacencyCache`) when
`NAMIDB_ADJACENCY=1` and falls back to the SST edge path otherwise.
No change to the cache layer; shortestPath traffic flows through the
same code path RFC-018 already powers.

### Errors

| Condition | Error code | Message |
|---|---|---|
| Open-ended length | `E002` | `shortestPath requires a finite upper bound (e.g. *..15)` |
| Unbound endpoint, no uniqueness | `E010` | `shortestPath endpoints must be bound or filtered by a unique property` |
| Multi-hop chain inside the call | `E006` | `shortestPath accepts a single relationship hop; use multiple MATCH clauses for longer chains` |
| Type alternation | `E006` | `relationship type alternation inside shortestPath is not supported yet (plain Expand and MultiwayJoin do support it; see RFC-024)` |

## Alternatives considered

### A. Dedicated `ShortestPath` operator

Already weighed in §LogicalPlan. The duplication isn't worth the
isolation; the BFS body in `execute_expand` is the right place to
add the early-exit logic.

### B. Dijkstra with edge weights

LDBC IC14 reasons about weighted shortest paths. Neo4j ships the
weighted variant as `allShortestPaths` with `WHERE` post-filtering
on edge properties, or as APOC `apoc.algo.dijkstra`. We deliberately
keep v0 to unweighted (BFS); weighted Dijkstra lands later when GDS
(graph data science) primitives become a thing.

### C. Bidirectional BFS

Bidirectional BFS (search from both endpoints, meet in the middle)
roughly halves the explored frontier on dense graphs. We could
implement it inside the same `execute_expand`. Reason to defer:
correct path reconstruction across the bidirectional meet point
needs more careful bookkeeping than the unidirectional BFS we
already have, and the unidirectional variant already meets the
IC13 baseline. Revisit when a bench shows the saving matters.

## Drawbacks

1. **The lower gains two more validation rules.** Open-ended
   `*..` and unbound endpoints now have specific error codes.
   Mitigation: the error messages cite this RFC, so the user sees
   *why* and what to write instead.

2. **No properties on relationships in the path.** The current BFS
   skips the per-edge `lookup_node` for back-reference paths
   (`back_reference=true`), so the `RuntimeValue::Path` we
   materialise carries Rel structs whose `properties` map is empty
   (the synthetic edge id we use for Bolt's `element_id` still
   works). Drivers consuming the path get the topology and the
   edge type but not the property map. Mitigation: the docs note
   it; a flag on `Path` lets the executor request full materialise
   when the caller actually projects `r.property` later.

3. **No detection of unreachable target.** Today the BFS stops when
   the frontier empties; a `shortestPath` between two disconnected
   nodes emits zero rows. Neo4j returns `null` for `p`. We follow
   the LDBC IC13 expected output (`shortestPathLength` is just
   absent when there's no path) for v0; a follow-up makes the
   `OPTIONAL MATCH p = shortestPath(...)` shape return one row with
   `p = NULL` so the Neo4j behaviour is reproducible.

## Open questions

- **Q1: Path materialisation for `allShortestPaths`.** When two
  shortest paths share a prefix and diverge mid-walk, the BFS today
  emits both via `next_frontier` branching, which means the row
  trail diverges naturally. Open: do we keep that path-per-row
  emission or fold to one row with a list-of-paths? Leaning
  path-per-row (matches Neo4j; `collect(p)` aggregates downstream).

- **Q2: `length(p)` builtin.** The Cypher function `length` on a
  Path returns the number of relationships. The executor already
  implements it for the regular path-binding case; need to verify
  it picks up the BFS-materialised path identically and add a test
  if not.

- **Q3: When does `shortestPath` participate in cost-based
  optimization?** Today the optimizer treats variable-length
  `Expand` as a fixed-cost node, which is the right default. After
  this RFC the BFS may terminate early at hop 2 even when `max=15`;
  the optimizer should reflect that. Out of scope here; the cost
  model gets a follow-up RFC.

## References

- LDBC SNB Interactive Workload, v0.4 — Erling et al., SIGMOD 2015.
- Neo4j Cypher 25 § "Shortest paths" —
  https://neo4j.com/docs/cypher-manual/current/patterns/concepts/#path-patterns
- RFC-002 (storage / CSR adjacency — feeds the BFS)
- RFC-018 (`AdjacencyCache` — cross-snapshot CSR cache the BFS reads)
- RFC-009 (write clauses; `shortestPath` was originally allocated
  here but lands in this dedicated RFC instead)
