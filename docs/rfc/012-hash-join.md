# RFC 012: HashJoin

**Status:** draft
**Author(s):** Matías Fonseca <info@namidb.com>
**Builds on:** RFC-008 (Logical Plan IR), RFC-010 (cost model), RFC-011 (predicate pushdown)
**Supersedes:** —

## Summary

Cosecha el `[join candidate]` annotation que el predicate pushdown
(RFC-011) dejó sembrado en EXPLAIN VERBOSE: convierte el shape
`Filter(a.x = b.y) ⇒ CrossProduct(L, R)` a un
`HashJoin { build, probe, on: [(a.x, b.y)] }` que el executor
materializa con build/probe phases. Operación O(N×M) → O(N+M).

Alcance:

- Nuevo operador `LogicalPlan::HashJoin { build, probe, on, residual }`.
- Rewriter `convert_cross_to_hash` en el pipeline de `optimize`,
 posterior al pushdown.
- Estimador de cardinalidad para `HashJoin`.
- Executor que materializa hash table en build-side, streams probe-side.
- EXPLAIN VERBOSE renders `HashJoin on=[(a.x, b.y)]` con build/probe
 como children.

Out-of-scope explícito:

- **HashSemiJoin via decorrelation** — convertir `SemiApply` cuando el
 subplan tiene correlación con bindings del outer. Requiere
 correlation analysis del subplan (separar la equality que correlaciona
 del resto, ejecutar subplan no-correlado, hash por la output column de
 correlación, probe outer). Iteración independiente.
- **Sort-merge join** — alternativa cuando los inputs ya están sorted.
 El executor no propaga ordering, así que solo HashJoin v0.
- **Broadcast / partitioned join** — distribuido.
- **Hash table spilling a disk** — cuando build no entra en memoria.
 v0 asume single-node build fits in memory. Documentado como drawback.
- **Equality detection sin AND-root** — solo extraemos cross-side
 equalities del top-level AND-tree del Filter. `Filter(a.x = b.y OR
 ...)` queda sin convertir.

## Motivation

Tras predicate pushdown el plan para
`MATCH (a:Person), (b:Person) WHERE a.firstName = b.firstName` queda:

```
Filter (a.firstName = b.firstName) [join candidate]
 CrossProduct (est=1000000) // 1000 * 1000
 NodeScan(Person, a) (est=1000)
 NodeScan(Person, b) (est=1000)
```

EXPLAIN VERBOSE flag-ea el shape como `[join candidate]` pero el plan
sigue ejecutando nested-loop. Sobre LDBC SF1 (3M Person), eso son
9×10¹² pairs antes de filtrar — query nunca termina. Con HashJoin:

```
HashJoin on=[(a.firstName, b.firstName)] (est=10000)
 build:
 NodeScan(Person, a) (est=1000)
 probe:
 NodeScan(Person, b) (est=1000)
```

Build phase: ~1M Person → 1M hash table entries (~80MB en RAM con
ndv(firstName)≈1k). Probe phase: 1M Person, lookup en hash → 1k
matches por probe (asumiendo distribución uniforme sobre 1k buckets).
Total: ~1M matches finales. **O(N+M) en vez de O(N×M)**, factor
ahorro de ~3 órdenes de magnitud.

Sin HashJoin, las queries multi-pattern de LDBC SNB Interactive (IC2 con
EXISTS, IC9 con sub-pattern) toman seconds/minutos sobre micro-graphs y
nunca terminan sobre SF1.

## Design

### 1. IR: `LogicalPlan::HashJoin`

```rust
/// Inner hash join (RFC-012). Equivalent to
/// `Filter(on AND residual) ⇒ CrossProduct(build, probe)` but
/// executed in two phases: build a hash table over `build`'s rows
/// keyed by each `JoinKey::build_side` expression; then stream
/// `probe`, evaluating `JoinKey::probe_side` and looking up matches.
///
/// The optimizer picks the side with smaller estimated cardinality
/// as `build` so the hash table stays compact.
///
/// `residual` is any non-equi predicate that survived from the
/// pre-conversion Filter (e.g. a >= b in `WHERE a.x = b.y AND a.z >= b.w`).
/// It is evaluated on the *joined* row in 3VL.
HashJoin {
 build: Box<LogicalPlan>,
 probe: Box<LogicalPlan>,
 on: Vec<JoinKey>,
 residual: Option<Expression>,
},
```

```rust
#[derive(Clone, Debug, PartialEq)]
pub struct JoinKey {
 /// Expression evaluated on each `build`-side row to compute the
 /// hash-table key. References only aliases produced by `build`.
 pub build_side: Expression,
 /// Expression evaluated on each `probe`-side row. References only
 /// aliases produced by `probe`.
 pub probe_side: Expression,
}
```

`children()` returns `vec![build, probe]` in that order — keeps
EXPLAIN rendering predictable.

`operator_name()` returns `"HashJoin"`.

`contains_write()` returns `false` (joins are read-side).

### 2. Conversion rewriter (`optimize::join_conversion`)

```rust
pub fn convert_cross_to_hash(
 plan: LogicalPlan,
 catalog: &StatsCatalog,
) -> LogicalPlan;
```

Bottom-up rewrite. The trigger shape after the post-pushdown plan is:

```
Filter { input: CrossProduct { left, right }, predicate }
```

Algorithm:

1. **Recurse** into children first (so we convert any inner joins
 before considering the current node).
2. **Match the trigger**. If the current plan is a `Filter` whose
 immediate child is a `CrossProduct`:
 a. **AND-split** the predicate into conjuncts.
 b. Compute `produced_aliases(left)` and `produced_aliases(right)`.
 c. For each conjunct `c`:
 - If `c` is `Binary { op: Eq, left: lhs, right: rhs }` and
 (`expression_aliases(lhs) ⊆ left_aliases ∧ expression_aliases(rhs) ⊆ right_aliases`) → push `(lhs, rhs)` to `on`.
 - Mirror case (`lhs ⊆ right ∧ rhs ⊆ left`) → push `(rhs, lhs)` so build_side always lines up with `build` operand. We canonicalize.
 - Otherwise → push to `residual_terms`.
 d. If `on.is_empty()` → no conversion possible; emit the original
 `Filter ⇒ CrossProduct` unchanged.
 e. **Build vs probe decision**: compute `estimate(left, catalog).rows`
 and `estimate(right, catalog).rows`. Whichever has fewer rows
 becomes `build`. If equal, prefer left as build (deterministic).
 Swap `on` keys if we picked right as build.
 f. **Coalesce residual**: `residual = and_chain(residual_terms)`
 → `Option<Expression>`.
 g. Emit `HashJoin { build, probe, on, residual }`.
3. Otherwise (or if step 2 doesn't apply): preserve the operator and
 recurse over its children.

#### Edge cases

- **Eq with a literal on one side** (`a.x = 5`): not cross-side, falls
 through to residual / pushdown. Already handled by RFC-011.
- **Eq with both sides referencing only one alias** (`a.x = a.y`):
 same-side, stays in residual. Already pushable to that side's
 subtree.
- **Eq with parameter** (`a.x = $param`): `expression_aliases($param) = ∅`.
 Falls through to residual; selectivity has it.
- **AND-only predicate**: `a.x = b.y AND a.z > b.w` → `on = [(a.x, b.y)]`,
 `residual = Some(a.z > b.w)`.
- **Multiple cross-side eqs**: `a.x = b.x AND a.y = b.y` → `on = [(a.x, b.x), (a.y, b.y)]`.
 Coalesced into a multi-key hash join.
- **No eq at all** (`Filter(a.x > b.x)`): `on.is_empty()`, no
 conversion. Plan remains nested-loop.

The rewriter must NOT trigger on Filters whose immediate child is not
a CrossProduct — those are pushdown leftovers and irrelevant.

### 3. Conversion entry point (`optimize::optimize`)

Integrates into the existing pipeline:

```rust
pub fn optimize(plan: LogicalPlan, catalog: &StatsCatalog) -> LogicalPlan {
 let mut current = plan;
 for _ in 0..MAX_FIXPOINT_ROUNDS {
 let next = normalize_filters(predicate_pushdown(current.clone()));
 let next = convert_cross_to_hash(next, catalog); // NEW
 if next == current { return next; }
 current = next;
 }
 current
}
```

Order matters: pushdown runs first so any pushable filter has been
moved out of the way; the only Filters remaining above CrossProduct
are by definition cross-side mixers. The rewriter then has the
cleanest possible signal.

### 4. Cardinality estimate

Add an arm to `cost::cardinality::estimate_inner` for `HashJoin`:

```rust
LogicalPlan::HashJoin { build, probe, on, residual } => {
 let b = estimate_inner(build, catalog);
 let p = estimate_inner(probe, catalog);

 // Selinger '79: inner equi-join cardinality.
 // rows = (|build| * |probe|) / max(ndv(build_key), ndv(probe_key))
 // For multi-key, assume independence: divide by product.
 let mut divisor = 1.0_f64;
 for key in on {
 let build_ndv = ndv_for_expr_opt(&key.build_side, catalog, &b.bindings).unwrap_or(1.0);
 let probe_ndv = ndv_for_expr_opt(&key.probe_side, catalog, &p.bindings).unwrap_or(1.0);
 divisor *= build_ndv.max(probe_ndv).max(1.0);
 }
 let mut rows = (b.rows * p.rows / divisor).max(0.0);

 // Residual reduces further. Use the existing selectivity machinery.
 if let Some(res) = residual {
 let mut combined = b.bindings.clone();
 for (k, v) in &p.bindings { combined.insert(k.clone(), v.clone()); }
 let bs = make_binding_stats(catalog, &combined);
 rows *= selectivity(res, &bs);
 }

 let mut bindings = b.bindings.clone();
 for (k, v) in &p.bindings { bindings.insert(k.clone(), v.clone()); }

 Cardinality {
 rows,
 children: vec![b, p],
 bindings,
 operator: "HashJoin",
 }
}
```

#### Why Selinger and not "min(|L|,|R|)"

A common shortcut estimate is `min(|L|, |R|)` for foreign-key joins.
That's correct when the join key is unique on one side and present in
every row of the other. For graph joins on arbitrary properties the
distribution is wider — Selinger captures both extremes:

- Unique on both sides → `min(|L|, |R|)` (since the join key has ndv = |L| ≈ |R|).
- Replicated key → much larger output.

We fall back to `divisor = 1` (= no reduction → CrossProduct
cardinality) when ndv is `None`. That keeps the estimate sound (never
under-estimates) at the cost of being pessimistic for queries the
catalog doesn't know about.

### 5. Executor

Two-phase implementation in `exec::walker`:

```rust
async fn execute_hash_join(
 build: &LogicalPlan,
 probe: &LogicalPlan,
 on: &[JoinKey],
 residual: &Option<Expression>,
 snapshot: &Snapshot<'_>,
 params: &Params,
) -> Result<Vec<Row>, ExecError> {
 // Build phase.
 let build_rows = execute_inner(build, snapshot, params, /*outer=*/ None).await?;
 let mut table: HashMap<Vec<RuntimeValue>, Vec<Row>> = HashMap::new();
 for row in build_rows {
 let mut key = Vec::with_capacity(on.len());
 let mut has_null = false;
 for jk in on {
 let v = evaluate(&jk.build_side, &row, params)?;
 if matches!(v, RuntimeValue::Null) {
 has_null = true;
 break;
 }
 key.push(v);
 }
 if has_null { continue; } // NULL keys never match (3VL).
 table.entry(key).or_default().push(row);
 }

 // Probe phase.
 let probe_rows = execute_inner(probe, snapshot, params, None).await?;
 let mut out = Vec::new();
 for prow in probe_rows {
 let mut key = Vec::with_capacity(on.len());
 let mut has_null = false;
 for jk in on {
 let v = evaluate(&jk.probe_side, &prow, params)?;
 if matches!(v, RuntimeValue::Null) { has_null = true; break; }
 key.push(v);
 }
 if has_null { continue; }
 if let Some(matches) = table.get(&key) {
 for brow in matches {
 let mut combined = brow.clone();
 for (k, v) in &prow.bindings { combined.bindings.insert(k.clone(), v.clone()); }
 if let Some(res) = residual {
 match evaluate(res, &combined, params)? {
 RuntimeValue::Bool(true) => out.push(combined),
 _ => {} // False or NULL drops.
 }
 } else {
 out.push(combined);
 }
 }
 }
 }
 Ok(out)
}
```

#### NULL semantics

`a.x = b.y` is `NULL` when either side is `NULL` (Cypher 3VL).
`Filter` drops rows where the predicate evaluates to NULL. Our hash
join replicates that: any NULL component in the join key skips both
the build insert and the probe lookup. Test coverage explicitly
exercises this.

#### Hash key representation

`Vec<RuntimeValue>` as the HashMap key. `RuntimeValue` implements
`Hash + Eq` through derive (numeric, string, bool variants are
straightforward). `RuntimeValue::Float` requires the bit-level
canonical form to make NaN sort to one bucket — already in the existing
`Hash` impl since the value layer.

#### Memory footprint

Build hash table size: roughly `|build| * (avg_key_size + avg_row_size)`.
For SF1-scale build of 3M Person × 200B/row + 50B key = ~750MB. That
fits comfortably in a 8GB machine. **Drawback**: no spill, so jobs
that pick the wrong build side OOM. Defended by the catalog-based
build-vs-probe decision. Future RFC adds spill.

#### Bindings combine

When we emit a joined row, we take the build row, then `.extend()`
its bindings with the probe row's bindings. If the two sides share a
binding name (shouldn't happen in well-formed plans, but defensive),
probe wins. This matches the lowering invariant: two pattern parts
share no fresh aliases (lowering uses `CrossProduct` precisely when
they don't).

### 6. EXPLAIN rendering

`write_header` for HashJoin:

```
HashJoin on=[(a.firstName, b.firstName)] residual=(a.id < b.id)
```

`residual` omitted when None. Multi-key:

```
HashJoin on=[(a.x, b.x), (a.y, b.y)]
```

The `[join candidate]` annotation that the predicate pushdown emitted
on the original `Filter ⇒ CrossProduct` disappears post-conversion —
the operator IS the join now. This is verifiable with a test.

`EXPLAIN VERBOSE` cardinality numbers show the dramatic improvement:
the HashJoin estimate is much smaller than the pre-conversion
CrossProduct estimate.

### 7. Interaction with subsequent rewrites

- **Predicate pushdown above HashJoin**: predicates that reference
 only build or only probe aliases can be pushed below the HashJoin
 into the respective subtree. The pushdown rule for HashJoin is
 identical to CrossProduct (split by side; mixed-eq is now in `on`
 or `residual`, doesn't reach pushdown). We add an arm to
 `predicate_pushdown` to support this.
- **Join reorder**: when reorder triggers, it can swap build
 and probe by re-evaluating `estimate` — the on-keys remain valid
 (just the symbol order in each pair swaps).
- **HashSemiJoin**: a `SemiApply` with a correlated subplan
 is decorrelated by extracting the correlation key, executing the
 subplan unparametrised, hashing on the correlation key, and probing
 the outer. The IR delta is small (`HashSemiJoin` is just `HashJoin`
 + emit-only-outer + optional negation flag).

## Alternatives considered

### A. Nested-loop with Bloom filter probe

DuckDB-style: build a Bloom filter on `build`'s join keys, probe each
`probe` row by Bloom-checking, and fall through to nested-loop on the
positives. **Rejected**: Bloom-filter false-positive rate ~1% means
99% of work is short-circuited, but the remaining 1% is still
O(N×M) — for N,M = 10⁶ that's 10¹⁰ comparisons. HashJoin is strictly
better when memory fits.

### B. Sort-merge join

If both sides are already sorted on the join key, a sort-merge join
avoids materialising a hash table. **Rejected for v0**: the executor
does not propagate ordering. Adding ordering metadata to the IR is a
separate, larger change (morsel-driven executor — vectorised, often
comes with sort ordering as a metadata column).

### C. Stream both sides and use partition-hash-join

Modern approach (DuckDB, ClickHouse, MapReduce-style). Both inputs
partition by hash; each partition joins independently. **Rejected**:
parallelism over partitions is a morsel feature, out-of-scope here.
Single-threaded HashJoin v0 is the simplest correct approach.

### D. Rely on graph-native joins (WCOJ / Worst-Case Optimal Join)

For cyclic / multi-way joins, WCOJ (RFC-009-eve) outperforms binary
hash joins. **Rejected aquí**: WCOJ es RFC-009's concern.
Binary hash joins cover the LDBC SNB interactive queries que esta RFC
ataca (IC2, IC4, IC10 with cross-pattern equi-joins) — WCOJ
gains kick in on truly cyclic queries (IC9 path patterns).

### E. Defer join conversion until query execution time

Adaptive: run a tiny sample of both sides, pick the join algorithm at
runtime. **Rejected**: adds runtime branching to the executor and
defeats the EXPLAIN/PROFILE story. Adaptive execution can revisit a
futuro.

## Drawbacks

1. **Unbounded hash table memory**. Build side fits in RAM is an
 assumption. For LDBC SF1 we're fine (build of 3M rows × 250B = ~750MB),
 but pathological queries (joins on rare keys with replicated rows)
 could blow memory. Mitigated by the build-vs-probe decision; not
 eliminated. Spill to disk queda como follow-up.

2. **No correlated subquery conversion**. SemiApply with a correlated
 subplan (the typical `EXISTS` shape) still nested-loops. Una
 iteración independiente lo resuelve.

3. **No multi-pattern join graph**. With 3+ pattern parts the optimizer
 sees a tree of CrossProducts; converting bottom-up means we lose
 the chance to pick a globally-optimal join order. Join reorder
 (RFC-016) addresses this by enumerating join trees BEFORE the
 conversion rewriter.

4. **The build side is materialised in full**. For very large build,
 this prevents streaming output. Modern hash joins emit matched
 rows as the probe streams — we do too, but only after the full
 build is in memory.

5. **Conversion conservative on residual**: when AND-split leaves
 non-eq conjuncts, we keep them as `residual` on the HashJoin. This
 means the residual evaluates on every joined row, which can be
 expensive. Future rewrites (`predicate_pushdown` over HashJoin) can
 further push residual conjuncts below the join if they reference
 only one side. Already supported by the standard pushdown rules
 once HashJoin is a recognised plan node.

## Open questions

- **OQ1**. `RuntimeValue::Float` as hash key — NaN canonical hashing
 must be enforced. Today's `Hash` impl uses raw bits, which makes
 distinct NaN bit patterns hash differently. We normalise during
 `evaluate` for the join key? Or in the `Hash` impl? Decided
 pragmatically: normalise on insert/lookup via a helper.

- **OQ2**. Should HashJoin emit a "join-key" column in the output so
 downstream operators can dedup cheaply? Today the executor does
 not — the build row's aliases survive verbatim; downstream Distinct
 re-hashes. Defer.

- **OQ3**. The cost-model assumes independence between multi-key
 components. For LDBC IC9 with `WHERE a.firstName = b.firstName AND
 a.lastName = b.lastName`, the two equalities are strongly
 correlated. The estimate over-reduces. Multi-column histograms a
 futuro lo arreglan.

## References

- Selinger et al., *Access Path Selection in a Relational Database
 Management System* (SIGMOD '79) — origin of cost-based join order
 and cardinality estimates we reuse here.
- DuckDB's *Push-Based Execution* (Mark Raasveldt 2022) — modern
 reference implementation for HashJoin.
- *Worst-Case Optimal Joins* (Ngo, Porat, Ré, Rudra; PODS '14) —
 WCOJ baseline para trabajo futuro. Mentioned to contrast our scope.
- `docs/rfc/008-logical-plan-ir.md` — IR this RFC extends.
- `docs/rfc/010-cost-based-optimizer.md` — cost model this RFC
 consumes via `StatsCatalog`.
- `docs/rfc/011-predicate-pushdown.md` — the `[join candidate]`
 annotation que esta RFC cosechas.

## Plan de implementación

1. **`crates/namidb-query/src/plan/logical.rs`** (~80 LoC + 3 tests):
 - Agregar `LogicalPlan::HashJoin` variant + `JoinKey` struct.
 - Actualizar `children()`, `operator_name()`, `contains_write()`,
 test del IR.

2. **`crates/namidb-query/src/optimize/join_conversion.rs`** (~400 LoC + 12-15 tests):
 - `convert_cross_to_hash(plan, catalog)` recursivo.
 - Helper `extract_cross_side_equalities(predicate, left_aliases,
 right_aliases) -> (Vec<JoinKey>, Vec<Expression>)`.
 - Helper `pick_build_side(left, right, catalog) -> Side`.

3. **`crates/namidb-query/src/optimize/pushdown.rs`** (~50 LoC + 4 tests):
 - Agregar HashJoin arm en `pushdown_at`: split por side, push
 pushable conjuncts a build o probe. Same shape as CrossProduct.

4. **`crates/namidb-query/src/optimize/mod.rs`** (~10 LoC):
 - Llamar `convert_cross_to_hash` post-pushdown en `optimize` pipeline.

5. **`crates/namidb-query/src/cost/cardinality.rs`** (~60 LoC + 4 tests):
 - Nuevo arm para `HashJoin` con la fórmula de §4.

6. **`crates/namidb-query/src/exec/walker.rs`** (~150 LoC + 5 tests):
 - `execute_hash_join` con build/probe phases.

7. **`crates/namidb-query/src/plan/explain.rs`** (~30 LoC + 3 tests):
 - `write_header` arm para HashJoin.
 - `plan_has_stats` arm.

8. **`crates/namidb-query/tests/cost_smoke.rs`** (+6 integration tests).

Snapshot esperado:
- `cargo test --workspace --exclude namidb-py`: 509 → ~555 passed.
- `cargo clippy --workspace --all-targets -- -D warnings`: clean.
- `cargo fmt --all -- --check`: clean.
- LoC nuevo: ~800 src + ~400 tests.
- Sin cambios en `namidb-storage`.
