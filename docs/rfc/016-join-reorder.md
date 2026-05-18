# RFC 016: Join reorder DP/greedy

**Status:** draft
**Author(s):** Matías Fonseca <info@namidb.com>
**Builds on:** RFC-010 (cost model), RFC-012 (HashJoin), RFC-013/015 (storage pushdowns)
**Supersedes:** —

## Summary

El HashJoin (RFC-012) produce localmente — escoge build/probe del par
actual. Cuando hay un *chain* `HashJoin { HashJoin { ... }, R3 }` (3+
relations unidas), el orden importa: el par que se construye primero
produce el hash table cuyos size domina la memoria, y el output
intermedio cuyo size se propaga al siguiente join. Hoy el optimizer
mantiene el orden literal del lowering, que sigue el orden textual del
WHERE — frecuentemente sub-óptimo.

Esta RFC enumera todos los órdenes left-deep para chains de HashJoins,
usa `estimate()` (cost model RFC-010 + ndv real) para elegir el de
menor cost intermedio total, y reconstruye el árbol con ese orden.
Selinger '79 DP O(N²·2^N), capeado N≤8 (LDBC SF1 IC8/IC9 tienen 4-5
patterns → factible).

Alcance v0:

- **HashJoin chains únicamente**. Detectar un subtree donde TODOS los
 operadores internos son `HashJoin` y las hojas son sub-trees
 arbitrarios (NodeScan, Expand, Filter, NodeById, etc.).
- **Equi-join keys preservadas**. Cada `HashJoin.on` queda como
 predicate sobre el par que se produce en ese paso. El rewriter
 re-distribuye las equalities a los pares correctos.
- **Left-deep DP** (Selinger '79). En cada paso del DP elige el par
 (S, R) que minimiza el cost acumulado, donde S es un subset y R una
 relation. Bushy plans podrían ganar 10-30% más en queries
 particulares; diferido.
- **Cap N≤8 relations**. Para N>8 (raro en LDBC), saltamos el
 reorder (mantenemos orden literal). 2^N=256 subsets manejables.

Out-of-scope:

- **Expand chain reordering**. Re-anclar un `(a)→(b)→(c)` chain a
 empezar desde `c` requiere reverse-direction Expand y conocer el
 label de cada alias. Complicado v0; diferido.
- **Cross-product reorder**. CrossProduct sin equi-keys queda como
 nested-loop — no hay decisión de orden útil.
- **HashSemiJoin reorder**. Los SemiJoins ya tienen orden fijo
 (outer probe, inner build); reorder no aplica directamente.
- **Cost-based reorder of CrossProducts dentro de chain**. v0 solo
 reordena el subtree HashJoin-only.

## Motivation

Plan IC8-like pre-rewrite:

```
HashJoin on=[(b.id, c.knows_id)]
 HashJoin on=[(a.id, b.knows_id)]
 NodeScan(Person, a) predicates=[a.id=$personId] (est=1)
 NodeScan(Person, b) (est=1000000)
 NodeScan(Person, c) (est=1000000)
```

Intermediate sizes:
- Inner HashJoin (a × b): 1 × 1M / ndv(KNOWS_id≈100) = 10000
- Outer HashJoin (ab × c): 10000 × 1M / 100 = 100000

Si el optimizer reorderaría a `(a × c) × b` (asumiendo `a-b` y `a-c`
joins son ambos válidos):
- Inner HashJoin (a × c): 1 × 1M / 100 = 10000 (similar)
- Outer HashJoin (ac × b): 10000 × 1M / 100 = 100000 (similar)

Hmm — en este ejemplo no cambia mucho porque las cardinalidades de
salida son similares. Pero cuando los predicados varían por
selectividad, el orden ÓPTIMO marca la diferencia. La forma canónica
es:

```
Total cost = Σ |intermediate_i|
```

Y minimizamos la suma. Para 3 relations el óptimo siempre es construir
sobre la relation con menor cardinalidad first.

## Design

### 1. Detección del subtree

```rust
struct JoinSubtree {
 /// Each leaf is an "atomic relation" — a plan subtree that does
 /// NOT contain a HashJoin at the root. (It may contain nested
 /// joins below; those were chosen by earlier passes.)
 leaves: Vec<LogicalPlan>,
 /// All the equi-join predicates pooled from every HashJoin in
 /// the subtree. Each is a pair of expressions; v0 supports only
 /// `(Property(alias_l, key_l), Property(alias_r, key_r))`.
 equalities: Vec<JoinEdge>,
 /// Residual expressions (non-equi) pooled from every HashJoin's
 /// `residual` field. Will be re-attached to whatever pair
 /// produces both halves of the binding map.
 residuals: Vec<Expression>,
}

struct JoinEdge {
 left_leaf_idx: usize,
 right_leaf_idx: usize,
 build_expr: Expression,
 probe_expr: Expression,
}
```

Pre-walk: recursive descent. When a HashJoin is hit, decompose into
the `build`'s recursion + `probe`'s recursion + add the join edges
to the pool. Any non-HashJoin descendant becomes a leaf with the
aliases it produces tracked.

### 2. Left-deep DP (Selinger '79)

```rust
struct DpState {
 /// Bitset of leaves included in this subplan.
 leaves_mask: u32,
 /// Best plan covering exactly `leaves_mask`.
 best_plan: LogicalPlan,
 /// Estimated cost (sum of intermediate sizes).
 cost: f64,
 /// Estimated rows of this subplan's output.
 rows: f64,
}
```

Selinger '79:

1. Base case (single leaf): `cost=0, rows=estimate(leaf)`.
2. Build: for size = 2..=N:
 - For each subset S of leaves with |S|=size:
 - For each (a) sub-subset T ⊂ S with |T|=size-1, (b) the
 remaining single leaf r = S \ T:
 - If there's an equi-key between T's aliases and r's
 aliases: candidate plan = HashJoin(build=DP[T], probe=r)
 with cost = DP[T].cost + cost_of_hash_join(DP[T].rows, r.rows, keys).
 - Pick the candidate with the lowest cost; record in DP[S].
3. Pick DP[full_set] as the final reorder.

`cost_of_hash_join` v0: `build.rows + probe.rows + estimated_output_rows`.
Estimated output rows = Selinger '79 formula (reusing
`cost::cardinality::estimate_hash_join`).

### 3. Cap & fallback

N>8 → skip reorder (keep literal). The cap is also a guard against
catastrophic blow-up when the user writes pathological queries.

### 4. Pipeline integration

`optimize::optimize` runs the join_reorder AFTER `convert_cross_to_hash`
and `convert_semi_apply_to_hash_semi_join` (so it sees the full
HashJoin shape) but BEFORE `apply_projection_pushdown` (so the
projection can prune the FINAL shape).

Idempotency: re-running on a plan that was already optimal produces
the same plan (the DP picks the same min-cost order deterministically).

### 5. Edge cases

- **No equalities between subsets**. If the only way to bridge two
 subsets is a CrossProduct (no equi-key), we leave them as
 CrossProduct (HashJoin's pre-condition still applies). The DP just
 picks the lower-cost arm available.
- **Residuals**. After DP picks the final tree, residuals are
 re-attached to the LOWEST HashJoin whose bindings include all the
 residual's referenced aliases. v0 attaches all residuals to the
 ROOT of the reordered tree (conservative).
- **Multi-key joins**. When two relations share multiple equi-keys,
 the DP picks them all (the `on` list grows).

## Alternatives considered

### A. Greedy bottom-up (no DP)

Pick the cheapest pair, merge, repeat. O(N²) instead of O(N²·2^N).
**Rejected**: known to produce sub-optimal plans on chains with
varying selectivities. DP at N≤8 is cheap enough.

### B. IKKBZ (Krishnamurthy-Kim-Boral)

Optimal left-deep enumeration in polynomial time using ranking
functions. **Rejected v0**: complex to implement; DP at N≤8 is fast
enough (~1ms even for N=8 = 256 subsets).

### C. Bushy DP

Try every binary partition, not just left-deep. **Deferred**.
Gains ~10-30% on specific cyclic patterns; doesn't apply to most
LDBC SNB queries.

### D. Hyper-graph reorder

DSDP (Moerkotte) or similar. **Out of scope**. Diferido si
benchmarks show v0 left-deep DP loses to bushy.

## Drawbacks

1. **Capped at N=8**. Queries with 9+ pattern parts (rare in LDBC)
 keep the literal order. Mitigated by `> 8` being uncommon.

2. **Residual placement is conservative**. v0 always attaches the
 union of residuals at the root. If a residual references only 2
 leaves, attaching it lower would prune earlier. Defer.

3. **Doesn't reorder Expand chains**. The biggest wins on LDBC IC2
 would come from re-anchoring an Expand chain — that's a structural
 rewrite this RFC explicitly punts.

4. **Cost model assumptions**. Selinger '79 assumes uniform
 distribution and independent keys. RFC-010 §"Drawbacks" tracks
 the broader issue. A futuro puede revisitarse.

## References

- Selinger et al., *Access Path Selection in a Relational Database
 Management System* (SIGMOD '79).
- Krishnamurthy, Kim, Boral, *Optimization of Nonrecursive Queries*
 (1986) — IKKBZ.
- Moerkotte, *Building Query Compilers* — bushy enumeration.
- `docs/rfc/012-hash-join.md` — HashJoin this RFC reorders.

## Plan de implementación

1. **`crates/namidb-query/src/optimize/join_reorder.rs`**
 (~500 LoC + 10 unit tests):
 - `reorder_joins(plan, &catalog) -> LogicalPlan`.
 - Collect/decompose helpers.
 - Selinger DP with bitmask state.
 - Rebuild HashJoin tree from DP result.

2. **`crates/namidb-query/src/optimize/mod.rs`** (~5 LoC):
 - Pipeline step `reorder_joins(...)` after decorrelation, before
 projection pushdown.

3. **`crates/namidb-query/tests/cost_smoke.rs`** (+4 tests):
 - `join_reorder_prefers_smaller_build_side`,
 - `join_reorder_keeps_plan_when_no_alternatives`,
 - `join_reorder_executes_with_parity`,
 - `join_reorder_caps_at_8_relations`.

Snapshot esperado:
- `cargo test --workspace --exclude namidb-py`: 627 → ~641 passed.
- `cargo clippy --workspace --all-targets -- -D warnings`: clean.
- `cargo fmt --all -- --check`: clean.
- LoC nuevo: ~500 src + ~250 tests + ~420 RFC.
