# RFC 013: Parquet predicate pushdown

**Status:** draft
**Author(s):** Matías Fonseca <info@namidb.com>
**Builds on:** RFC-008 (Logical Plan IR), RFC-010 (cost model), RFC-011 (predicate pushdown), RFC-002 §4 (SST stats)
**Supersedes:** —

## Summary

Cosecha los `min/max` que el writer ya emite por row-group (RFC-002 §4
+ read-back desde el Parquet footer) para evitar decodificar
row-groups que no pueden contener filas que satisfacen el WHERE.

El predicate pushdown estructural (RFC-011) empujó cada Filter lo más
cerca del leaf (NodeScan) que pudo; este siguiente paso convierte el
`Filter` inmediatamente sobre `NodeScan` en `predicates` *del
NodeScan*, que el storage layer consume durante el scan para descartar
row-groups completos sin decodificar.

Alcance:

- Tipo `ScanPredicate` en `namidb-storage::sst::predicates`
 (Eq / Lt / LtEq / Gt / GtEq / Between / IsNull / IsNotNull / In)
 contra una sola columna y un literal canónico (`StatScalar`).
- `eval_row_group(predicate, &PropertyColumnStats) -> RowGroupVerdict`
 conservador: `Absent` solo si los stats demuestran imposibilidad,
 `MaybePresent` cuando los stats faltan o no son concluyentes.
- `NodeSstReader::scan_with_predicates(&[ScanPredicate])` lee el
 Parquet metadata, evalúa cada predicate contra los stats por
 row-group, skipea cualquier row-group con verdict `Absent` para
 CUALQUIER predicate.
- `Snapshot::scan_label_with_predicates(label, &[ScanPredicate])` y
 `Snapshot::scan_label(label)` mantenido como wrapper de
 `scan_label_with_predicates(label, &[])` (compat).
- `LogicalPlan::NodeScan` gana `predicates: Vec<ScanPredicate>` field.
 Default vacío para callers existentes.
- Rewriter en `optimize::pushdown`: cuando llega a `NodeScan` con
 `pending` no-vacío, intenta convertir cada conjunct a un
 `ScanPredicate` sobre `alias.property`. Los pushables van al
 NodeScan.predicates; los no-pushables permanecen como `Filter`.
- Executor pasa los predicates al storage en el callsite de
 `walker::execute_node_scan`.
- Cardinality evalúa selectividad de los predicates ya empujados
 sobre el `node_count` del catalog (consistente con RFC-010 §3.1).
- EXPLAIN VERBOSE: `NodeScan label=Person alias=a predicates=[a.age > 30, a.firstName = "Alice"]`.

Out-of-scope explícito:

- **Parquet row-level filtering** (Arrow `filter` operator dentro del
 reader). El executor ya tiene `Filter` y aplicarlo dos veces
 duplicaría trabajo. Solo hacemos *row-group* pruning en storage.
- **Predicates sobre Expand / edge SST**. La RFC-002 sí define stats
 por edge SST (`DegreeHistogram`) pero las edges no tienen stats
 por propiedad arbitraria. Se mantiene fuera del v0.
- **OR predicates**. Cada `ScanPredicate` es un single-column AND
 conjunct. `WHERE a.x = 1 OR a.y = 2` no se empuja en v0 —
 requeriría unión de row-groups con bookkeeping del verdict
 por-row. Conjunctive-only.
- **Predicates cross-alias** (`WHERE a.age = b.age`). El storage no
 conoce `b`; el Filter cross-alias permanece arriba del NodeScan y
 cuando el HashJoin rewrite ya lo está convirtiendo a HashJoin, esto
 NO le quita nada.
- **Predicates derivados de parámetros**. Los parámetros se
 resuelven en runtime; el storage layer no los ve. Si el lowering
 conoce el valor (constante), se baja como literal; si es un
 parameter abierto, el Filter permanece arriba. Una posible
 extensión post-v0: resolver parameters en `optimize::optimize`
 cuando `&Params` se pase al pipeline.
- **Page-index pruning** (Parquet 2.0 column index + offset index).
 Vale para SSTs grandes con row-groups grandes pero requiere otra
 layer de stats. Skip v0; el writer ya emite chunk-level stats
 (`EnabledStatistics::Chunk`) — suficiente.
- **Bloom filter check sobre eq predicates**. El writer emite
 bloom filters para `node_id` pero no para propiedades arbitrarias.
 Extender el bloom a properties es un trade-off de espacio (bloom
 bytes son ~7×rows) — esto queda diferido si las stats min/max no
 alcanzan.
- **Cardinality estimate con dependencia entre predicates**. Cada
 predicate aplica selectividad independiente, como RFC-010 §3.2. El
 catalog HLL daría correlación pero v0 mantiene la asunción de
 independencia.

## Motivation

Sobre LDBC SNB SF1 (3M Person nodes en ~30 SSTs de 100k rows c/u, con
row-groups de 8192 rows = ~366 row-groups por label), una query
`MATCH (a:Person) WHERE a.creationDate > '2020-01-01' RETURN a` que
hoy:

- Lee TODOS los SSTs (cada uno ~10–40 MB sobre S3).
- Decodifica TODOS los row-groups (~366 Parquet decompressions).
- Filtra row-level en el executor: descarta ~99% de las filas.

Con predicate pushdown:

- Lee TODOS los SSTs **footer + page index** (~64 KiB c/u, no body).
- Por cada SST consulta `min/max(creationDate)` por row-group.
- Skip los row-groups cuyo `max(creationDate) < '2020-01-01'`.
- Decodifica solo los row-groups que pueden contener matches
 (~10–30 row-groups en lugar de 366).

**Ahorro de IO sobre S3 (dominante en cloud):** 10× típico para
queries selectivas. Ahorro de CPU en decoding (Parquet
deserialización): factor ~30×.

Esto es la última pieza del pushdown end-to-end del query layer al
storage. Sin ella, los `min/max` que el writer emite no se usan: solo
están en el catalog para el cost model.

## Design

### 1. `namidb-storage::sst::predicates`

```rust
/// A single-column conjunctive predicate that the SST reader can
/// evaluate against per-row-group stats to skip entire row-groups
/// without decoding them.
///
/// Each variant references a column **by its declared property name**
/// (not by Parquet leaf path). The reader resolves to leaf index at
/// scan time.
#[derive(Clone, Debug, PartialEq)]
pub enum ScanPredicate {
 Eq { column: String, value: StatScalar },
 Lt { column: String, value: StatScalar },
 LtEq { column: String, value: StatScalar },
 Gt { column: String, value: StatScalar },
 GtEq { column: String, value: StatScalar },
 /// `Between { low, high }` is INCLUSIVE both sides. Equivalent to
 /// `Gte(low) AND Lte(high)`.
 Between { column: String, low: StatScalar, high: StatScalar },
 IsNull { column: String },
 IsNotNull { column: String },
 In { column: String, values: Vec<StatScalar> },
}
```

The literal type — `StatScalar` — is the same one the writer emits
into `PropertyColumnStats`. This guarantees comparison between
predicate and stats lives in a single ordering.

#### Verdict

```rust
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RowGroupVerdict {
 /// Stats prove no row in this row-group can satisfy the predicate.
 Absent,
 /// Stats are insufficient (missing min/max, type mismatch) OR
 /// stats overlap — rows may or may not match. Decode the
 /// row-group.
 MaybePresent,
}
```

Conservatism: we only return `Absent` when the stats *prove* the row-group
contains no match. Missing min/max ⇒ `MaybePresent`. Type mismatch
(predicate is `Utf8` but stats are `Int32`) ⇒ `MaybePresent` (defensive;
treats the predicate as inapplicable rather than asserting).

#### Evaluation

```rust
pub fn eval_row_group(
 predicate: &ScanPredicate,
 stats: &PropertyColumnStats,
) -> RowGroupVerdict;
```

Algorithm per predicate (column known to match `stats.name`):

- `Eq(v)`:
 - If `min ≤ v ≤ max` → MaybePresent. Else Absent.
 - If min/max missing → MaybePresent.
- `Lt(v)`:
 - If `min < v` → MaybePresent (some row may be < v). Else Absent
 (all rows ≥ v).
- `LtEq(v)`: `min ≤ v` → MaybePresent. Else Absent.
- `Gt(v)`: `max > v` → MaybePresent. Else Absent.
- `GtEq(v)`: `max ≥ v` → MaybePresent. Else Absent.
- `Between { low, high }`: equivalent to `GtEq(low) AND LtEq(high)`.
 Apply both; AND of verdicts = Absent if either is Absent.
- `IsNull`: `null_count > 0` → MaybePresent. Else Absent.
- `IsNotNull`: row-group has at least one non-null value
 (`null_count < row_count`); we don't have `row_count` per stats here
 but we DO have `null_count` and if `null_count == 0` we're sure
 non-nulls exist (defensive: always MaybePresent in v0 when stats
 exist; falls back to row-level Filter for the per-row check).
- `In { values }`: build the closed interval `[min(values), max(values)]`;
 apply same logic as `Between`. False-positives are accepted (some
 intermediate value may not be in the `In` list — caught by the
 Filter operator above the NodeScan that the rewriter leaves intact
 as residual when In is partial).

Comparison between `StatScalar` variants follows the obvious type
matching: `Int32 vs Int32`, `Float64 vs Float64`, `Utf8 vs Utf8`, etc.
Cross-type comparison (e.g. `Int32 vs Float64`) returns `MaybePresent`
in v0; the optimizer doesn't generate cross-type predicates because the
property type is declared in the schema.

NULL handling: `min` and `max` in `PropertyColumnStats` are computed
over non-null values only (writer convention; see RFC-002 §4.1). So a
column where every value is NULL has `min=None, max=None,
null_count=N`. We evaluate `IsNull` from null_count alone, and any
ordered predicate (`Eq/Lt/Gt/...`) on a min/max=None column returns
MaybePresent (defensive — the row-group has no non-null rows that
could satisfy the predicate, but evaluating the predicate at row-level
will correctly drop NULLs via 3VL).

### 2. `NodeSstReader::scan_with_predicates`

```rust
impl NodeSstReader {
 pub fn scan_with_predicates(
 &self,
 predicates: &[ScanPredicate],
 ) -> Result<Vec<RecordBatch>>;
}
```

Algorithm:

1. Build the Parquet reader once: `ParquetRecordBatchReaderBuilder::try_new(body)`.
2. Read the metadata; collect the leaf index for every property column
 referenced by any predicate.
3. For each row-group:
 - For each predicate, locate the column leaf in the row-group's
 ColumnChunkMetaData → `cc.statistics()`. Map to a per-row-group
 `PropertyColumnStats` synthesizing only the fields evaluation needs
 (`null_count`, `min`, `max` — same coding as
 `compute_property_stats`).
 - Evaluate each predicate with `eval_row_group`. If ANY returns
 `Absent`, skip the row-group.
4. Collect surviving row-group indices into `keep`.
5. If `keep` is empty, return `Vec::new()` (no decode).
6. Otherwise build the reader `with_row_groups(keep)` and decode as in
 `scan()`.

Cost: ~few µs per row-group of metadata inspection (the metadata is
already in memory from `try_new`). When all row-groups survive we
fall through to the same path as `scan()` and pay no extra IO.

### 3. `Snapshot::scan_label_with_predicates`

```rust
impl Snapshot<'_> {
 pub async fn scan_label(&self, label: &str) -> Result<Vec<NodeView>> {
 self.scan_label_with_predicates(label, &[]).await
 }

 pub async fn scan_label_with_predicates(
 &self,
 label: &str,
 predicates: &[ScanPredicate],
 ) -> Result<Vec<NodeView>>;
}
```

The new variant:

- Iterates the memtable as `scan_label` does today, but additionally
 evaluates each predicate against the materialised NodeView's
 property map. Memtable values are decoded already, so this is
 cheap (no IO).
- For each SST scoped to `label`, calls
 `reader.scan_with_predicates(predicates)` instead of `reader.scan()`.
- Returns the same `BTreeMap<NodeId, …>` semantics: tombstones win;
 last-write-wins by LSN.

Memtable predicate evaluation is row-by-row in v0 and uses the same
3VL semantics as the executor's Filter (`Bool(true)` → keep,
`Bool(false)` / `Null` → drop).

### 4. `LogicalPlan::NodeScan` change

```rust
NodeScan {
 label: String,
 alias: String,
 /// Predicates that have been pushed into the scan from a Filter
 /// directly above it. Empty for the lowering output; populated
 /// by `optimize::pushdown` when conjuncts qualify (see §5).
 /// The executor passes them verbatim to
 /// `Snapshot::scan_label_with_predicates`.
 predicates: Vec<ScanPredicate>,
}
```

`PartialEq`, `Clone`, `Debug` derive over the new field. All existing
constructions of `NodeScan` (lowering, tests) now use
`predicates: Vec::new()`.

`operator_name()` remains `"NodeScan"`. EXPLAIN VERBOSE renders the
predicates inline (see §6).

### 5. Rewriter in `optimize::pushdown`

The existing leaf arm:

```rust
LogicalPlan::Empty | LogicalPlan::Argument { .. } | LogicalPlan::NodeScan { .. } => {
 apply_filters(plan, pending)
}
```

becomes:

```rust
LogicalPlan::NodeScan { label, alias, predicates } => {
 let (pushable, residual) = classify_pending_for_scan(pending, &alias, &label_def);
 let mut merged = predicates;
 merged.extend(pushable);
 apply_filters(
 LogicalPlan::NodeScan { label, alias, predicates: merged },
 residual,
 )
}
```

`classify_pending_for_scan(pending, alias, label_def)` returns:

- `pushable: Vec<ScanPredicate>` — conjuncts that are single-column
 comparisons on `alias.<property>` with a literal/parameter (only
 literals in v0; parameters deferred) and reference NO
 other alias. The property must be declared in `label_def`.
- `residual: Vec<Expression>` — everything else; stays as `Filter`
 above the NodeScan.

The classification function lives in
`optimize::parquet_pushdown::classify` (new module) and is unit
tested independently. The integration into `pushdown_at` is a single
arm change.

Why fold the parquet pushdown into the same `pushdown_at` pass instead
of a separate post-pass: the `pending` accumulator already carries
every conjunct that the existing pushdown was about to materialise as
`Filter` over `NodeScan`. Classifying them at the leaf is the natural
place — it costs O(|pending|) per leaf and avoids a second tree walk.

### 6. EXPLAIN VERBOSE

```
Project [a] (est=1500)
 NodeScan label=Person alias=a predicates=[a.age > 30, a.firstName = "Alice"] (est=1500)
```

Plain `EXPLAIN` (no VERBOSE) also renders the predicates — they are
part of the operator shape, not annotations. `EXPLAIN RAW` shows the
pre-optimize lowering, where `NodeScan` has `predicates: vec![]` and
the conjuncts live in a `Filter` above.

`predicates=[...]` rendering uses each predicate's `Display`:

- `Eq { column, value }` → `a.col = <lit>`
- `Lt { column, value }` → `a.col < <lit>`
- `Between { column, low, high }` → `a.col BETWEEN <low> AND <high>`
- `IsNull { column }` → `a.col IS NULL`
- `IsNotNull { column }` → `a.col IS NOT NULL`
- `In { column, values }` → `a.col IN [<v1>, <v2>, ...]`

The `alias.col` prefix comes from the NodeScan's `alias`. Literals
render via their `StatScalar` Display.

### 7. Cardinality

The `NodeScan` arm in `cost::cardinality::estimate_inner` becomes:

```rust
LogicalPlan::NodeScan { label, alias, predicates } => {
 let base = catalog.label(label).map(|l| l.node_count as f64).unwrap_or(0.0);
 let sel = predicates_selectivity(predicates, catalog, label);
 let rows = base * sel;
 // bindings, leaf as before
}
```

`predicates_selectivity` reuses the existing `selectivity` machinery
from `cost::selectivity` by translating each `ScanPredicate` to its
`Expression` analogue (`Eq → BinaryOp::Eq`, etc.) and calling
`selectivity(&expr, &binding_stats)` where `binding_stats` is
seeded from the property stats in the catalog. Multi-predicate
combines under the independence assumption (RFC-010 §3.2).

Trade-off: we double-evaluate selectivity (once at NodeScan level for
the pushed predicates, once at any residual Filter above). This is
correct — Filter applies on top of the already-reduced NodeScan
estimate.

### 8. Edge SSTs

Out of scope (see §"Out-of-scope"). `EdgesFwd/Inv` SSTs ship
`DegreeHistogram` but no per-property stats. Adding edge-property
stats requires the writer to track them per edge_type — a separate
RFC if/when needed.

## Alternatives considered

### A. Filter pushdown using DataFusion's Expr

DataFusion has a full Expr language with a `PhysicalPlanner` that
converts pushable Expr to Parquet `RowFilter`. **Rejected**: bringing
DataFusion as a dependency for the pushdown ergonomics alone is a
mismatch — we'd still need a translator from our `Expression` to
their `Expr`, and the rest of our executor wouldn't share the path.
A future morsel/vectorized iteration may revisit, but pushdown alone
doesn't pay it.

### B. Runtime adaptive sampling

Detect that a query is selective by sampling the first N rows and
deciding pushdown on the fly. **Rejected**: defeats EXPLAIN/PROFILE
story (plan changes at runtime), and our static stats are good enough
for the v0 regime.

### C. Encode predicates in a server-side filter pushdown to S3 Select

S3 Select supports SQL filters server-side but requires the body to be
in CSV/JSON. Parquet Select is not GA. **Rejected**:
incompatible with our storage format. Future feasibility check goes
with edge storage RFCs.

### D. Build a custom bloom filter per property column for eq pushdown

The writer would emit a bloom over each property's hashed values. Eq
predicates probe the bloom before reading the row-group. **Deferred**:
RFC-002 explicitly limits blooms to `node_id` (for point lookups). A
property bloom is ~7×rows bytes — for 100k row SSTs that's ~700 KiB
per property column. The space cost only pays off on cardinalities
that min/max-based pruning misses (which is rare — eq on high-NDV
columns is already covered by min/max for the values inside a row-group
and a NodeId-bloom alike). Track for follow-up if real workloads show it.

## Drawbacks

1. **No row-level filter in storage**. Surviving row-groups still
 decode in full; the executor's Filter then drops non-matching rows.
 For row-groups with ~50% selectivity this double-touches values.
 Mitigated by the executor's Filter living in the same process —
 it's cheap. Row-level pushdown in storage would couple Arrow's
 `filter` operator to the reader (morsel direction).

2. **Single-column predicates only**. Multi-column predicates
 (`a.x + a.y > 100`) stay as `Filter`. v0 accepted.

3. **No parameter substitution in v0**. `WHERE a.age > $minAge` keeps
 the Filter above the NodeScan since we don't resolve `$minAge` until
 execution. A later optimization passes `&Params` into
 `optimize::optimize` to constant-fold them; deferred.

4. **OR predicates are not pushed**. `WHERE a.age > 30 OR a.firstName =
 "Alice"` is one conjunct in the AND-split (the OR root), and
 pushability requires single-column ⇒ rejected. The Filter survives.
 Could be added by extending `ScanPredicate` to a tree, but row-group
 verdict combination for OR gets messier (union of MaybePresent
 verdicts).

5. **`IsNotNull` is conservative**. We don't have per-row-group row_count
 to verify `null_count < row_count`. Always returns MaybePresent
 when stats exist; the executor's Filter drops nulls at row level.
 Negligible cost.

6. **Cross-type predicate comparison returns MaybePresent**. By
 design (defensive). The optimizer doesn't construct cross-type
 predicates because schemas declare property types — but if a future
 path introduces them, this is the safety net.

7. **Memtable predicate eval is row-by-row** (not vectorised). The
 memtable is typically small (<10k rows before flush), so this is
 negligible vs SST decoding. Morsel-driven execution can revisit.

## Open questions

- **OQ1**. `predicates_selectivity` reuse path: should it translate
 `ScanPredicate` → `Expression` and call `selectivity`, or have its
 own simpler arm? Decided: translate (single source of truth for
 selectivity heuristics).

- **OQ2**. Should `NodeById` also accept predicates? Today
 `NodeById` is a point-lookup. Predicates on the same alias COULD be
 applied during the lookup. v0: no — the lookup decodes one row
 group with at most one row anyway; Filter on top is fine. Track as
 follow-up if benchmarks show a hot path.

- **OQ3**. Should the writer emit per-row-group HLL sketches (not just
 per-SST)? This would enable approx-NDV reasoning per row-group
 for eq pushdown. Deferred; the current per-SST HLL is
 sufficient for query-level cardinality estimates.

## References

- *Parquet 2.0 column index + offset index* — Apache Parquet
 specification §6.2 (column index for page-level pruning).
- DuckDB's *predicate pushdown into Parquet readers* (Raasveldt 2022)
 — modern reference implementation.
- `docs/rfc/002-sst-format.md` §4 — stats embedded in SST.
- `docs/rfc/008-logical-plan-ir.md` — IR this RFC extends.
- `docs/rfc/010-cost-based-optimizer.md` — cost model this RFC reuses.
- `docs/rfc/011-predicate-pushdown.md` — the structural pushdown
 this RFC builds on.

## Plan de implementación

1. **`crates/namidb-storage/src/sst/predicates.rs`** (~250 LoC + 18 tests):
 - `ScanPredicate` enum + `RowGroupVerdict` + `eval_row_group`
 evaluator. Helpers `scalar_cmp(a, b) -> Ordering` and
 `scalar_eq(a, b) -> bool` (delegating to PartialOrd / PartialEq
 of `StatScalar`).
 - Module `pub` in `sst/mod.rs`.
 - Unit tests cubren: Eq in/out range, Lt boundary, GtEq with NULL
 min, IsNull positive/negative, In with single/multi values,
 Between, missing min/max → MaybePresent, type mismatch →
 MaybePresent.

2. **`crates/namidb-storage/src/sst/nodes.rs`** (~150 LoC + 5 tests):
 - `NodeSstReader::scan_with_predicates(&[ScanPredicate])`
 implementing §2 algorithm. `scan()` becomes a wrapper of
 `scan_with_predicates(&[])`.
 - Helper `row_group_stats_for_column(rg, col_name, prop_def)
 -> Option<PropertyColumnStats>` reusing the mapping from
 `compute_property_stats`.
 - Unit tests: predicate skips all row-groups, predicate skips
 some, no predicates fall through to full scan,
 `IsNull` with NULL row-group survives, multi-predicate AND,
 no-stats fallback keeps row-group.

3. **`crates/namidb-storage/src/read.rs`** (~80 LoC + 3 tests):
 - `Snapshot::scan_label_with_predicates(label, &[ScanPredicate])`
 implementing §3 algorithm. `scan_label(label)` wraps it with
 `&[]`.
 - Memtable predicate eval helper using `node_view_matches_predicates`
 (NULL-safe 3VL).
 - Unit tests: memtable filtering, SST filtering, predicate over
 tombstoned row, ND ndv (just kidding — verifies catalog isn't
 used in scan path).

4. **`crates/namidb-storage/src/sst/mod.rs` + `lib.rs`**:
 - `pub mod predicates` + re-export `ScanPredicate`, `RowGroupVerdict`,
 `eval_row_group`.

5. **`crates/namidb-query/src/plan/logical.rs`** (~30 LoC + 1 test):
 - `LogicalPlan::NodeScan` adds `predicates: Vec<ScanPredicate>`.
 - Type alias `pub use namidb_storage::sst::predicates::ScanPredicate`
 at module root so the rest of the query crate doesn't need to know
 the storage path.
 - Updates to `children()` (no children added — predicates are flat),
 `operator_name()` (still "NodeScan"), `contains_write()` (still
 false).
 - Test ensures NodeScan with predicates equals NodeScan with same
 predicates and not equal when predicates differ.

6. **`crates/namidb-query/src/optimize/parquet_pushdown.rs`** (~250 LoC + 14 tests):
 - `classify_pending_for_scan(pending: Vec<Expression>, alias: &str,
 label_def: &LabelDef) -> (Vec<ScanPredicate>, Vec<Expression>)`.
 - Conversion `try_into_scan_predicate(expr, alias) -> Option<ScanPredicate>`
 case-analysing each `Expression::kind`. Supports: BinaryOp
 {Eq/Lt/LtEq/Gt/GtEq} with `PropertyAccess(alias, prop)` on one
 side and `Literal(lit)` on the other; the literal converts to
 `StatScalar` via a helper. `IS NULL / IS NOT NULL`. `IN [list]`
 when every element is a literal. `BETWEEN` decomposes to
 Gte+Lte at lowering time so the AND-split already gives us two
 conjuncts.
 - Tests: eq pushable, eq with non-matching alias rejected, eq
 with literal-on-left, range pushable, IS NULL pushable, IS NOT
 NULL pushable, IN with all literals, IN with non-literal
 rejected, cross-alias rejected, non-declared property rejected,
 parameter rejected, complex arithmetic rejected, idempotency.

7. **`crates/namidb-query/src/optimize/pushdown.rs`** (~30 LoC + 4 tests):
 - NodeScan arm in `pushdown_at` now consults
 `parquet_pushdown::classify_pending_for_scan`. The non-pushable
 conjuncts materialise as `Filter` above; the pushable accumulate
 into `predicates`.
 - Tests verify: filter eq on declared prop ends up in NodeScan
 predicates; filter on parameter stays as Filter; filter on
 undeclared prop stays as Filter; filter on different alias
 stays as Filter (and would have been pushed elsewhere by the
 CrossProduct arm).

8. **`crates/namidb-query/src/optimize/mod.rs`** (~10 LoC):
 - `pub mod parquet_pushdown` + re-export `classify_pending_for_scan`
 so tests can reach it.
 - The `optimize` pipeline doesn't add a new pass; the NodeScan arm
 change in `pushdown_at` covers it.

9. **`crates/namidb-query/src/optimize/normalize.rs`** (~5 LoC):
 - `recurse_children` arm for NodeScan recurses on... nothing
 (NodeScan is a leaf). The change is to preserve `predicates`
 when the arm clones the variant — trivial.

10. **`crates/namidb-query/src/exec/walker.rs`** (~10 LoC):
 - `execute_node_scan` callsite (line ~140) passes `predicates`
 to `snapshot.scan_label_with_predicates(label, predicates).await?`.

11. **`crates/namidb-query/src/cost/cardinality.rs`** (~40 LoC + 3 tests):
 - NodeScan arm applies multiplicative selectivity over predicates
 using the existing `selectivity::selectivity` and
 `BindingStats` machinery.
 - Tests: NodeScan with eq predicate estimate drops below base;
 NodeScan with range predicate estimate proportional to range;
 NodeScan with empty predicates equals base.

12. **`crates/namidb-query/src/plan/explain.rs`** (~50 LoC + 2 tests):
 - `write_header` arm for NodeScan with predicates renders as
 §6. Predicate Display uses `format_scan_predicate(p, alias)`
 helper, also unit-tested.

13. **`crates/namidb-query/src/plan/lower.rs`** (~10 LoC):
 - Lowering creates `NodeScan { predicates: vec![] }`. Mecánica;
 sin test new.

14. **`crates/namidb-query/tests/cost_smoke.rs`** (+8 integration tests):
 - `parquet_pushdown_moves_eq_to_scan`
 - `parquet_pushdown_moves_range_to_scan`
 - `parquet_pushdown_keeps_cross_alias_in_filter`
 - `parquet_pushdown_keeps_undeclared_property_in_filter`
 - `parquet_pushdown_renders_in_explain`
 - `parquet_pushdown_estimate_drops_below_full_scan`
 - `parquet_pushdown_executes_with_parity_to_raw`
 - `parquet_pushdown_skips_all_row_groups_when_out_of_range`

Snapshot esperado:
- `cargo test --workspace --exclude namidb-py`: 528 → ~580 passed.
- `cargo clippy --workspace --all-targets --exclude namidb-py -- -D warnings`: clean.
- `cargo fmt --all -- --check`: clean.
- LoC nuevo: ~900 src + ~500 tests + ~650 RFC.
