# RFC 014: HashSemiJoin via decorrelation

**Status:** draft
**Author(s):** Matías Fonseca <info@namidb.com>
**Builds on:** RFC-008 (Logical Plan IR), RFC-010 (cost model), RFC-011 (predicate pushdown), RFC-012 (HashJoin)
**Supersedes:** —

## Summary

Cosecha el ÚLTIMO operador nested-loop del read-path: `SemiApply`.
Hoy `SemiApply { input, subplan, negated }` ejecuta `subplan` UNA VEZ
por cada row de `input` — O(N·M) sobre `EXISTS` subqueries. Para LDBC
SF1 (3M outer rows × 100 avg degree por subplan = 3×10⁸ ops) eso es
infactible. Esta RFC decorrelaciona el subplan (sustituye el
`Argument` leaf por una `NodeScan` independiente), construye un hash
table una sola vez, y filtra el outer probando contra él: O(N+M).

Alcance:

- Nuevo operador `LogicalPlan::HashSemiJoin { outer, inner, on,
 negated, residual }`. Forma EXACTAMENTE como `HashJoin` excepto
 que:
 - `outer` ↔ probe semantic (NO se duplican rows del inner — máx 1
 output row por outer row matching);
 - `negated` flag para `AntiSemiJoin` (NOT EXISTS).
- Rewriter `convert_semi_apply_to_hash_semi_join(plan, &catalog)`
 bottom-up. Detecta `SemiApply` cuyo subplan:
 1. tiene EXACTAMENTE un `Argument` leaf,
 2. cuyas `bindings` son un SUBSET de 1 alias `X`,
 3. cuyo label puede inferirse del outer scope (NodeScan o Expand
 con `target_label`).
 Sustituye `Argument(X)` por `NodeScan { label: <X's label>, alias:
 X, predicates: vec![] }` y emite `HashSemiJoin` con
 `on=[JoinKey{ build: Property(X,"id"), probe: Property(X,"id") }]`.
- Executor `execute_hash_semi_join`: build phase materializa un
 `BTreeSet<NodeId>` (no full row buffering — solo necesitamos
 "any match"); probe phase emite outer row si lookup acierta
 (semi) o si NO acierta (anti).
- Cardinality `HashSemiJoin` estima rows como
 `outer.rows · min(1.0, inner.rows / outer_X_distinct)` (semi-join
 retains at most all outer rows).
- EXPLAIN VERBOSE: `HashSemiJoin on=[(a.id, a.id)] negated=false`
 o `AntiHashSemiJoin` cuando `negated=true`.

Out-of-scope:

- **SemiApply cuyo subplan tiene Argument con MÚLTIPLES bindings**.
 Requiere multi-column hash key — diferente shape. Iteración futura.
- **SemiApply cuyo subplan no contiene Argument** (subplan independiente
 del outer). Es trivialmente "ejecutar una vez", pero requiere otro
 rewrite path (cache+broadcast). Iteración futura.
- **PatternList decorrelation**. Mismo shape pero materializa lista en
 vez de boolean. Iteración futura.
- **Pushdown sobre HashSemiJoin**. Heredado del existing pushdown
 (`hash_semi_join` arm en `optimize::pushdown::pushdown_at`):
 conjuncts del outer pueden bajar al outer-side, conjuncts del
 inner-only no aplican (el inner no contribuye bindings al output).
- **Multi-pattern EXISTS** (`EXISTS { (a)-[]->(b)-[]->(c) }`). El
 subplan tiene un solo Argument leaf; el cuerpo es un chain de
 Expands. Funciona automáticamente — el rewriter solo reemplaza el
 Argument; el resto del subplan se mantiene literal y se ejecuta
 como inner en build phase.

## Motivation

Sin decorrelation, una query como:

```cypher
MATCH (a:Person)
WHERE EXISTS { (a)-[:KNOWS]->(b:Person) }
RETURN a.firstName
```

produce el plan:

```
Project [a.firstName]
 SemiApply { negated: false }
 NodeScan { Person, a } (outer)
 Expand { source=a, edge_type=KNOWS, target=b } (subplan)
 Argument { bindings: [a] }
```

Sobre micro-graph 6 Persons / 6 KNOWS, esto ya cuesta 6 × scan_label
= 6 evaluaciones del subplan (que itera todos los Persons + edges per
outer row). Sobre LDBC SF1 (3M Persons / 100 avg degree), cuesta
3M × scan_label = infinito.

Con decorrelation el plan optimizado es:

```
Project [a.firstName]
 HashSemiJoin on=[(a.id, a.id)]
 NodeScan { Person, a } (outer)
 Expand { source=a, edge_type=KNOWS, target=b } (inner)
 NodeScan { Person, a } (decorrelated leaf)
```

Build phase ejecuta el inner UNA vez: 6M edges. Probe phase: 3M outer
rows × O(1) lookup = 3M ops. Total: 6M build + 3M probe = 9M ops.
**Mejora de 3M / 30 = 100 000× para este caso típico**.

## Design

### 1. IR: `LogicalPlan::HashSemiJoin`

```rust
HashSemiJoin {
 /// The "probe" side. Bindings from `outer` are the ones that
 /// survive into the output.
 outer: Box<LogicalPlan>,
 /// The "build" side. Bindings introduced by `inner` are
 /// DROPPED — only used to decide whether each outer row matches.
 inner: Box<LogicalPlan>,
 /// Equi-join keys. `build_side` is evaluated on each `inner`
 /// row at build time, `probe_side` on each `outer` row at
 /// probe time. Single-key in v0 (multi-key is OK if needed).
 on: Vec<JoinKey>,
 /// `false`: keep outer rows that have at least one inner match
 /// (`EXISTS`). `true`: keep outer rows with NO inner match
 /// (`NOT EXISTS`).
 negated: bool,
 /// Residual predicate evaluated on the joined row (outer
 /// bindings + inner bindings, 3VL). Optional — most simple
 /// EXISTS lower to no residual.
 residual: Option<Expression>,
}
```

Semantics:

- `outer.bindings` ⊆ output bindings. Inner bindings are dropped (the
 semi-join semantics).
- Build phase: for each `inner` row, evaluate `JoinKey::build_side`
 expressions; if any key component is NULL, skip the row (3VL).
 Build a `BTreeSet<Vec<String>>` (canonical key fingerprint, reusing
 `dedup_rows`'s helper).
- Probe phase: for each `outer` row, evaluate `JoinKey::probe_side`
 expressions; if any is NULL, skip (3VL). Lookup in the set. Emit
 outer row iff `(matched, negated)` matches the desired truth
 table:
 - `(true, false)` → keep (EXISTS).
 - `(false, true)` → keep (NOT EXISTS).
 - else → drop.
- Residual: when present, evaluate on the JOINED row (outer ∪ build's
 full row recovered from a secondary `Vec<Row>` map). For v0 we
 default `residual = None` since the lowering of bare EXISTS
 doesn't generate one.

### 2. Rewriter `optimize::decorrelation::convert_semi_apply_to_hash_semi_join`

Pre-walk the plan to populate `outer_labels: BTreeMap<String,
Option<String>>` — alias → declared label for every NodeScan and
labeled Expand target in scope.

Then walk top-down. For each `SemiApply { input, subplan, negated }`:

1. Recurse into `input` and `subplan` (independent decorrelation).
2. Detect `subplan` shape:
 - Has exactly ONE `Argument { bindings: [X] }` leaf (depth-first
 descent through Expand/Filter/NodeById; reject if multiple
 Arguments or any operator that's not in the decorrelation-safe
 list).
 - The Argument's `bindings` is exactly `[X]` (a single alias).
 - `outer_labels[X] == Some(L)` for some label `L`.
3. Build `inner` by walking the subplan, replacing the unique
 `Argument` with `NodeScan { label: L, alias: X, predicates: vec![] }`.
4. Emit `HashSemiJoin { outer: input, inner: new_subplan,
 on: vec![JoinKey { build_side: Property(X, "id"),
 probe_side: Property(X, "id") }], negated, residual: None }`.

Decorrelation-safe operators (descend into to find Argument):

- `Expand`
- `Filter` (residual conjuncts stay on the inner; semantics is "the
 subplan still filters its rows; HashSemiJoin probes whether any
 filtered row matches the outer key")
- `NodeById` (only if its `input` is the Argument leaf)
- `Project` with `discard_input_bindings: false` (rare in subplans
 but possible)
- `NodeScan`, `Empty` — never contain Argument; the Argument has to
 be at the leaf.

If any other operator appears (`Aggregate`, `TopN`, `Distinct`,
`Union`, `CrossProduct`, `HashJoin`, write ops, `SemiApply` itself),
the rewriter bails and the original SemiApply is kept. v0 keeps the
detection conservative — false negatives leave performance on the
table but never produce incorrect plans.

Idempotency: `HashSemiJoin` is not a `SemiApply`, so the second pass
of the fixpoint won't re-trigger. Verified in unit tests.

### 3. Executor

```rust
async fn execute_hash_semi_join(
 outer: &LogicalPlan,
 inner: &LogicalPlan,
 on: &[JoinKey],
 negated: bool,
 residual: &Option<Expression>,
 snapshot: &Snapshot<'_>,
 params: &Params,
 outer_bindings: Option<&Row>,
) -> Result<Vec<Row>, ExecError>
```

Phase 1 (build): execute `inner` once (no outer context). For each
inner row, evaluate every `JoinKey::build_side` expression. If ANY is
NULL, skip the row (3VL). Otherwise, push the fingerprint into a
`BTreeSet<Vec<String>>`.

Phase 2 (probe): execute `outer`. For each outer row, evaluate every
`JoinKey::probe_side`. Compute matched = `set.contains(fingerprint)`.
Keep iff `(matched, negated)` is `(true, false)` or `(false, true)`.

Residual: when `residual.is_some()`, the build phase additionally
stores the full inner row alongside the fingerprint, and the probe
phase iterates matching inner rows to evaluate the residual on the
joined binding map. v0 ships `residual = None` from the rewriter so
this path is exercised only by future RFC iterations.

### 4. Cardinality

```rust
LogicalPlan::HashSemiJoin { outer, inner, on, negated, residual: _ } => {
 let o = estimate_inner(outer, catalog);
 let i = estimate_inner(inner, catalog);
 // P(at least one inner match for an outer row) ≈
 // 1 - (1 - i.rows/distinct(inner_key))^(o.rows/distinct(outer_key))
 // Simplification: i.rows / max(distinct_outer, 1.0) treated as the
 // probability a random outer row matches.
 let frac_match = (i.rows / o.rows.max(1.0)).min(1.0);
 let rows = if negated {
 o.rows * (1.0 - frac_match)
 } else {
 o.rows * frac_match
 };
 ...
}
```

The estimate is folklore for now — multi-key correlation and inner
NDV are revisited a futuro. The output is clamped to `[0, o.rows]`.

### 5. EXPLAIN VERBOSE

```
HashSemiJoin on=[(a.id, a.id)] (est=4)
 NodeScan label=Person alias=a (est=6)
 Expand source=a edge_type=KNOWS target=b (est=12)
 NodeScan label=Person alias=a (est=6)
```

When `negated=true`, the operator name is `AntiHashSemiJoin` (mirrors
the existing `AntiSemiApply` rendering).

### 6. Integration with the pipeline

- `optimize::optimize` (in `optimize::mod`) runs
 `convert_semi_apply_to_hash_semi_join` AFTER
 `convert_cross_to_hash` in the same fixpoint round. Order: pushdown
 → normalize → cross-to-hash → semi-to-hashsemi.
- `optimize::pushdown::pushdown_at` gets a `HashSemiJoin` arm: conjuncts
 that reference only `outer` bindings push to the outer side;
 conjuncts that reference inner-only bindings are nonsensical
 (inner bindings are dropped) so they stay above the HashSemiJoin
 defensively.
- `optimize::normalize` gets a `HashSemiJoin` arm: recurse on outer
 and inner.

### 7. Bindings analysis

A subtle point: `Argument { bindings: [X] }` may carry multiple names
when the subplan needs more than one outer variable. v0 rejects
those (`arg_bindings.len() != 1`). They will be common in deeper
LDBC queries with chained patterns; una iteración futura lifts the
restriction via multi-key joins.

When the subplan introduces NEW bindings (e.g. `Expand` introduces
`target_alias`), those are local to the subplan and DO NOT leak to
the outer output — same as the existing SemiApply semantics.

## Alternatives considered

### A. Hash table over outer + iterate inner

Build over outer (keyed by `X.id`), iterate inner and emit
`outer_row` once when matched. **Rejected**: requires deduplication
on the inner side (the same outer might match multiple inner rows
emitting duplicates). Cheaper to build over inner.

Actually we go the other way: **build over INNER** (because inner is
the small EXISTS side typically — friends-of-friend, etc.) and probe
outer. The build side stores the SET of key values; each outer is
checked once. Result preserves outer row order.

### B. Adaptive at runtime

Decide per-query whether to decorrelate based on the actual sizes of
outer/inner. **Rejected**: defeats EXPLAIN/PROFILE story. The cost
model picks the side (build vs probe) statically.

### C. Apply pushdown into the subplan

A more aggressive rewrite: pushing outer predicates INTO the inner
subplan so the inner produces only the relevant subset. **Rejected for
v0**: requires correlation analysis beyond the equality on the
`X.id` (parameter propagation). Una iteración futura may revisit.

## Drawbacks

1. **Restricted to single-Argument-binding subplans**. Multi-binding
 correlation is common in real LDBC queries — `EXISTS { (a)-[]->(b)
 ... b.x = a.y }`. v0 keeps these as nested-loop SemiApply.

2. **Inner duplication**. The decorrelated inner enumerates every X
 in the corpus (not just those referenced by the outer). When the
 outer is much smaller than the inner's universe (e.g. outer is a
 single row), nested-loop SemiApply is cheaper — `inner.rows /
 outer.rows` becomes lopsided. v1 could compare estimates and
 choose accordingly; v0 always decorrelates when shape matches.

3. **NULL on join key drops outer / inner rows silently**. Same as
 `HashJoin` 3VL semantics. Documented inline. For typical
 `EXISTS { (a)-[]->(b) }` this never triggers because node ids are
 never NULL.

4. **Cost-model estimate is folklore**. The independence and uniform-
 distribution assumptions over-simplify. RFC-010 §"Drawbacks 1"
 tracks the broader observation; a futuro puede refinarse.

## Open questions

- **OQ1**. Should the rewriter try to lift `Filter` arms from inside
 the subplan to the outer-side when the conjunct references only
 outer bindings? Today the subplan is rewritten verbatim. Defer.

- **OQ2**. How does `HashSemiJoin` interact with `PatternList`?
 `PatternList` is semantically a multi-row apply that materialises a
 list. Decorrelation produces a `HashJoin` (NOT semi-join) with
 array aggregation per outer key. Separate RFC.

## References

- Selinger et al., *Access Path Selection in a Relational Database
 Management System* (SIGMOD '79) — semi-join cardinality.
- Galindo-Legaria & Joshi, *Orthogonal Optimization of Subqueries
 and Aggregation* (SIGMOD '01) — formal decorrelation rewrites.
- `docs/rfc/008-logical-plan-ir.md` — IR this RFC extends.
- `docs/rfc/012-hash-join.md` — HashJoin executor this RFC mirrors.

## Plan de implementación

1. **`crates/namidb-query/src/plan/logical.rs`** (~50 LoC + 2 tests):
 - `LogicalPlan::HashSemiJoin` variant.
 - `operator_name` returns `"HashSemiJoin"` / `"AntiHashSemiJoin"`.
 - `children` → `[outer, inner]`. `contains_write` → false (rewriter
 never touches subtrees with writes).

2. **`crates/namidb-query/src/optimize/decorrelation.rs`** (~250 LoC + 8 tests):
 - `convert_semi_apply_to_hash_semi_join(plan, &catalog)`.
 - `outer_label_map(plan)` walks NodeScan/Expand collecting
 alias → label.
 - `find_unique_argument(plan)` returns
 `Option<(&Argument bindings, parent path)>`.
 - `replace_argument(plan, x, label)` substitutes the unique
 Argument with a fresh `NodeScan { label, alias: X, predicates:
 vec![] }`.
 - Tests cover: simple EXISTS → decorrelates, NOT EXISTS → negated,
 no-Argument subplan → kept as SemiApply, multi-binding
 Argument → kept, label unknown → kept, EXISTS with extra
 Filter → still decorrelates (filter remains in inner),
 nested SemiApply → outer SemiApply NOT touched if its
 subplan has SemiApply, idempotency.

3. **`crates/namidb-query/src/optimize/mod.rs`** (~10 LoC):
 - `pub mod decorrelation`.
 - `optimize` pipeline runs
 `convert_semi_apply_to_hash_semi_join(plan, catalog)` AFTER
 `convert_cross_to_hash`.

4. **`crates/namidb-query/src/optimize/pushdown.rs`** (~30 LoC + 3 tests):
 - HashSemiJoin arm (split pending by outer/inner-aliases).

5. **`crates/namidb-query/src/optimize/normalize.rs`** (~5 LoC):
 - HashSemiJoin arm in `recurse_children`.

6. **`crates/namidb-query/src/cost/cardinality.rs`** (~40 LoC + 3 tests):
 - HashSemiJoin arm with §4 formula.

7. **`crates/namidb-query/src/exec/walker.rs`** (~80 LoC + 2 tests):
 - `execute_hash_semi_join` build/probe phases.

8. **`crates/namidb-query/src/exec/writer.rs`** (~10 LoC):
 - HashSemiJoin arm (defensive, never produced by writes).

9. **`crates/namidb-query/src/plan/explain.rs`** (~30 LoC + 2 tests):
 - `HashSemiJoin` rendering. `negated` flag selects
 `AntiHashSemiJoin`.

10. **`crates/namidb-query/tests/cost_smoke.rs`** (+5 integration tests):
 - `decorrelation_converts_simple_exists`,
 - `decorrelation_preserves_results`,
 - `decorrelation_handles_not_exists`,
 - `decorrelation_keeps_multi_binding_subplan_as_semi_apply`,
 - `decorrelation_renders_hash_semi_join_in_explain`.

Snapshot esperado:
- `cargo test --workspace --exclude namidb-py`: 596 → ~625 passed.
- `cargo clippy --workspace --all-targets -- -D warnings`: clean.
- `cargo fmt --all -- --check`: clean.
- LoC nuevo: ~500 src + ~250 tests + ~400 RFC.
