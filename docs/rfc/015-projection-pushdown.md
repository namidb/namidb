# RFC 015: Projection pushdown / column pruning

**Status:** draft
**Author(s):** Matías Fonseca <info@namidb.com>
**Builds on:** RFC-008 (Logical Plan IR), RFC-010 (cost model), RFC-013 (Parquet predicate pushdown)
**Supersedes:** —

## Summary

Hoy `NodeSstReader::scan()` decodifica TODAS las columnas Parquet
declaradas en el `LabelDef`, aunque la query referencie sólo una
fracción. Sobre Person en LDBC SF1 (~12 columnas, ~3M filas), un
`RETURN a.firstName` decodifica 12× más datos del necesario. Esta RFC
cierra el end-to-end del pushdown:

1. **Analyze** — walk del plan top-down recolectando, por alias, el
 conjunto de propiedades que las expresiones referencian (RETURN,
 WHERE residual, ORDER BY, predicados de filtro intermedios).
2. **Annotate** — `LogicalPlan::NodeScan` gana
 `projection: Option<Vec<String>>` (None = todas las columns,
 default para back-compat). El rewriter lo populates con el set
 inferido del analyze step.
3. **Storage** — `NodeSstReader::scan_with_predicates_and_projection`
 construye un `ProjectionMask` de Parquet que sólo lee las column
 leafs necesarias. Las engine columns (`node_id`, `tombstone`,
 `lsn`, `__schema_version`, `__overflow_json`) se incluyen
 siempre.
4. **Reader** — el resto del path (`Snapshot::scan_label_*`) se
 adapta a transparentar la projection.
5. **EXPLAIN VERBOSE** — `NodeScan label=Person alias=a
 projection=[firstName]` cuando hay projection no-trivial.

Sobre LDBC SF1 con `RETURN a.firstName` esperamos: 12× menos bytes
leídos desde S3 + 12× menos decoding CPU.

Alcance:

- Property-column pruning para NodeScan. Edge SSTs y NodeById quedan
 out-of-scope v0.
- Análisis conservador: cuando una expresión usa `Variable(a)` sin
 PropertyAccess (e.g. `RETURN a`), la projection es `None` (lee
 todas las columnas, incluso `__overflow_json`).
- Análisis de subplans de SemiApply/PatternList/HashSemiJoin va
 recursivo dentro de cada scope (los inner emiten su propio
 projection).
- Predicates ya pushados a `NodeScan.predicates` (RFC-013) cuentan
 como referencias a sus columnas — el storage los necesita para
 filtrarlas.

Out-of-scope:

- **EdgesFwd/Inv property streams**. Edges aún no emiten per-property
 streams (RFC-002 §3.2.7 follow-up). Sin streams separados no hay
 granularidad de proyección.
- **NodeById**. Decodifica un row group con max 1 row; el ahorro de
 IO es marginal y el overhead de la projection mask para un
 point-lookup no se justifica v0.
- **Overflow column elision**. Cuando la query NO referencia
 propiedades no-declaradas, podríamos omitir `__overflow_json`.
 v0 lo mantiene siempre (defensivo).
- **Projection pushdown dentro de Project**. Cuando un `Project` deja
 bindings vivas (`discard_input_bindings: false`), todas las
 columnas downstream son potencialmente referenciadas. v0 trata
 Project no-discarding como barrera.
- **Pruning de schema-version / lsn columns**. Solo de propiedades
 declaradas. Las engine columns son baratas (UInt64 chunks
 RLE-comprimidos) y removerlas rompería la semántica de
 tombstone/winner.

## Motivation

Plan ejemplo pre-rewrite:

```
Project [a.firstName] (est=3000000)
 NodeScan label=Person alias=a predicates=[] (est=3000000)
```

`NodeScan` decodifica `prop_firstName`, `prop_lastName`,
`prop_birthday`, `prop_creationDate`, `prop_locationIP`,
`prop_browserUsed`, `prop_gender`, `prop_email`, `prop_speaks`, …
12+ columns Parquet. El executor luego accede solo
`row[a].get("firstName")`.

Con projection pushdown:

```
Project [a.firstName] (est=3000000)
 NodeScan label=Person alias=a projection=[firstName] predicates=[] (est=3000000)
```

`NodeSstReader::scan_with_predicates_and_projection` construye un
`ProjectionMask::leaves(schema, &[firstName_leaf])` y Parquet sólo
lee las column pages relevantes. **Reducción de bytes leídos: ~10×
sobre Person SF1**. La mejora se acumula con el parquet predicate
pushdown (ya descartó row-groups; ahora descartamos columnas dentro
de los row-groups que sobreviven).

## Design

### 1. IR change

```rust
NodeScan {
 label: String,
 alias: String,
 predicates: Vec<ScanPredicate>,
 /// Optional projection: only these property columns are
 /// materialised. `None` = include every declared property
 /// (back-compat). The rewriter populates this from analysis;
 /// lowering emits `None`.
 projection: Option<Vec<String>>,
}
```

`PartialEq`, `Clone`, `Debug` derive over the new field. All existing
constructions of `NodeScan` upgrade to `projection: None`.

Two NodeScans with different projection are considered different
plans (matters for the `optimize` fixpoint termination check).

### 2. Analysis

Walk the plan TOP-DOWN with a `RequiredSet`:

```rust
#[derive(Default, Clone)]
struct RequiredSet {
 /// Properties accessed for each alias still in scope.
 by_alias: BTreeMap<String, RequiredProps>,
}

#[derive(Default, Clone)]
enum RequiredProps {
 /// A specific set of properties.
 Set(BTreeSet<String>),
 /// At least one expression accessed the binding as a whole
 /// (`Variable(alias)`) — we don't know which properties it
 /// references, so all of them must survive.
 All,
}
```

Algorithm (`compute_required(plan: &LogicalPlan)`):

- Start with `RequiredSet::default()` at the root (no projections
 referenced yet).
- For each operator visited top-down, the operator's *output* may be
 referenced by the parent. Compute the required set *of the
 operator's output*, then determine what the operator's inputs must
 produce:
 - **Project**: items contribute references; outputs are the project's
 aliases. If `discard_input_bindings: true`, only items'
 references survive; else inherit parent's set + items'.
 - **Filter / TopN / Distinct**: predicate / keys contribute
 references on top of parent's.
 - **Expand**: introduces target_alias / rel_alias. Their
 requirements are sourced by reading the target NodeView /
 EdgeView. Removed downstream when the Expand's input is
 computed.
 - **NodeScan**: leaf. Its `alias`'s required set IS the projection
 we set on the NodeScan.
- Each expression contributes via `collect_property_refs(expr)` which
 walks the AST and emits `(alias, key)` pairs from PropertyAccess
 nodes, plus `(alias, ALL)` from bare `Variable(alias)` (e.g.
 `RETURN a` requires all columns).
- Predicates already pushed into `NodeScan.predicates` MUST also
 contribute — they reference column names via `ScanPredicate.column()`.

### 3. Rewriter

`apply_projection_pushdown(plan: LogicalPlan) -> LogicalPlan`:

1. Compute the required set once (a single top-down pass).
2. Walk the plan bottom-up. For each NodeScan, look up its alias in
 the required set:
 - `Some(RequiredProps::Set(cols))` → set
 `projection = Some(cols.into_iter().collect())`. Sort
 alphabetically for determinism.
 - `Some(RequiredProps::All)` or `None` → leave `projection = None`
 (read everything).

Idempotent: re-running on a plan that already has projections is a
no-op because the analysis discovers exactly the same set.

The rewriter is integrated into `optimize::optimize` as the LAST
step of each fixpoint round (after predicate pushdown, normalize,
HashJoin conversion, decorrelation). Putting it last ensures it sees
the FINAL plan shape, including any predicates absorbed into NodeScan
y any nodes the rewriters introduced.

### 4. Storage

```rust
impl NodeSstReader {
 pub fn scan_with_predicates_and_projection(
 &self,
 predicates: &[ScanPredicate],
 projection: Option<&[String]>,
 ) -> Result<Vec<RecordBatch>>;
}
```

Algorithm:

1. If `projection.is_none()` → fall through to
 `scan_with_predicates(predicates)` (no extra projection mask).
2. Build a `ProjectionMask::leaves(schema_descr, &leaf_indices)`
 where `leaf_indices` includes:
 - Engine columns: `node_id`, `tombstone`, `lsn`,
 `__schema_version`, `__overflow_json`. Always.
 - For each property name in `projection`, locate
 `prop_<name>` in the Parquet schema and add its leaf index.
 Defensive: if a property is not in the schema (label evolution
 edge case), skip it.
3. Apply both row-group pruning AND the projection mask via
 `ParquetRecordBatchReaderBuilder::with_projection`.
4. The decoded `RecordBatch`es have a SCHEMA that includes only the
 selected columns. The reader returns them as-is; the caller
 (`Snapshot::scan_label_*`) is already defensive when looking up
 columns by name — missing columns map to `None` properties.

Optimization note: Parquet's `ProjectionMask` avoids decoding the
column pages NOT in the projection. Combined with row-group skipping
(parquet predicate pushdown) the cold read on S3 goes from
`O(R * C)` page reads to `O(R_kept * C_proj)` where `R_kept ≪ R` and
`C_proj ≪ C` (with projection pushdown).

### 5. Snapshot reader adaptation

```rust
impl Snapshot<'_> {
 pub async fn scan_label_with_predicates_and_projection(
 &self,
 label: &str,
 predicates: &[ScanPredicate],
 projection: Option<&[String]>,
 ) -> Result<Vec<NodeView>>;
}
```

Memtable handling: the `properties` BTreeMap is *constructed* in
memory anyway (no IO to save). For consistency we filter the in-mem
properties to only include the projected names — keeps `NodeView`
shape uniform between memtable-sourced and SST-sourced rows. Cheap.

`scan_label_with_predicates(label, predicates)` becomes a wrapper of
`scan_label_with_predicates_and_projection(label, predicates, None)`.
`scan_label(label)` continues to wrap `(label, &[], None)`.

### 6. EXPLAIN VERBOSE

```
NodeScan label=Person alias=a projection=[firstName] predicates=[a.age > 30]
```

When `projection.is_none()` we omit the field (default behaviour).
When `projection.is_some()`, sort alphabetically for stable output.

### 7. Cardinality

No change — the row count out of a projected NodeScan is identical
to the un-projected one (same rows, fewer columns). The cost model
doesn't track byte-level costs in v0; that lives behind a follow-up
(cuando agreguemos CPU-weighted cost).

## Alternatives considered

### A. Project pushdown inside the executor

Skip the storage layer adaptation; let the executor build the
`NodeView` with all columns and discard unused ones. **Rejected**:
defeats the entire purpose. The win is in the IO path (S3 reads
fewer column pages).

### B. Per-property bloom filter

Already deferred from parquet predicate pushdown. Not relevant to projection.

### C. Schema-aware col pruning at the manifest level

The manifest already records `PropertyColumnStats` per column. We
could elide columns whose stats are all-null (the column is missing
in every SST). **Rejected**: dynamic — the schema may evolve;
defensive read of "missing column" returns None and is cheap.

### D. Just rely on Parquet's RLE for unused columns

Parquet's run-length encoding makes unused-column reads cheap if the
column is mostly null or constant. **Rejected**: still pays the
metadata cost (column index fetches) plus the page-header round-trip
per column. The projection mask is strictly better.

## Drawbacks

1. **Missing projection ⇒ no win**. When the query references the
 bare alias (`RETURN a`), the analysis falls back to
 `RequiredProps::All` and we read every column. Same as today.

2. **PropertyAccess inside subqueries**. The analysis descends into
 subplans of SemiApply / HashSemiJoin / PatternList — IF the
 subplan introduces a NodeScan, the NodeScan inside the subplan
 gets its own projection. The decorrelated inner reads `a.id` plus
 whatever the subplan body references. Often just `id` ⇒ massive win.

3. **Schema evolution**. If a writer landed a SST with column X that
 the current schema doesn't declare (extra prop), the projection
 mask might exclude it and accidentally drop a usable column. v0
 only projects from declared properties (`label_def.properties`),
 so extra columns are simply not requested — same as no-projection
 behaviour for those columns.

4. **Composite types**. `FloatVector` and `Json` columns are
 projected as any other property. `Json` may carry overflow
 properties we don't want to drop — but `__overflow_json` is
 always in the engine-columns list, separate from `prop_*` json
 columns. Documented inline.

5. **Cardinality doesn't reflect IO savings**. Two NodeScans with
 identical row counts but different projection cost differently
 in bytes. v0 EXPLAIN VERBOSE shows `est=N` rows for both; un future
 PROFILE surface bytes.

## Open questions

- **OQ1**. Should NodeById get projection too? Each NodeById decodes
 exactly one row group (≤ 1 row); the metadata overhead of building
 a projection mask may dominate the win. Defer.

- **OQ2**. How does projection interact with `compact.rs` (LSM
 compactions)? Compactions read full rows to merge winners. They
 don't go through `scan_label`. Unaffected.

## References

- DuckDB's *push-based execution* projection rewrites (Raasveldt 2022).
- Parquet's `ProjectionMask::leaves` API.
- `docs/rfc/008-logical-plan-ir.md` — IR this RFC extends.
- `docs/rfc/013-parquet-predicate-pushdown.md` — IO-pushdown this
 RFC composes with.

## Plan de implementación

1. **`crates/namidb-storage/src/sst/nodes.rs`** (~60 LoC + 4 tests):
 - `NodeSstReader::scan_with_predicates_and_projection` extending
 `scan_with_predicates` with a `ProjectionMask`. Engine columns
 always included.
 - Unit tests: projection includes engine + named columns; missing
 property is silently skipped; projection=None falls through.

2. **`crates/namidb-storage/src/read.rs`** (~50 LoC + 2 tests):
 - `Snapshot::scan_label_with_predicates_and_projection`.
 - Memtable view filtering: only declared+projected properties
 survive.

3. **`crates/namidb-query/src/plan/logical.rs`** (~10 LoC + 1 test):
 - `NodeScan` adds `projection: Option<Vec<String>>`. Update all
 constructions.

4. **`crates/namidb-query/src/optimize/projection_pushdown.rs`** (~250 LoC + 8 tests):
 - `apply_projection_pushdown(plan)`.
 - `compute_required(plan) -> RequiredSet`.
 - `collect_property_refs(expr, out)` AST walker.

5. **`crates/namidb-query/src/optimize/mod.rs`** (~5 LoC):
 - `pub mod projection_pushdown` + run as last step of pipeline.

6. **`crates/namidb-query/src/exec/walker.rs`** (~5 LoC):
 - NodeScan callsite passes `projection.as_deref()` to snapshot.

7. **`crates/namidb-query/src/plan/explain.rs`** (~10 LoC):
 - Render `projection=[col1, col2]` when present.

8. **`crates/namidb-query/tests/cost_smoke.rs`** (+5 integration tests):
 - `projection_pushdown_extracts_referenced_columns`,
 - `projection_pushdown_handles_bare_variable_as_all`,
 - `projection_pushdown_includes_predicate_columns`,
 - `projection_pushdown_executes_with_parity`,
 - `projection_pushdown_renders_in_explain`.

Snapshot esperado:
- `cargo test --workspace --exclude namidb-py`: 612 → ~640 passed.
- `cargo clippy --workspace --all-targets -- -D warnings`: clean.
- `cargo fmt --all -- --check`: clean.
- LoC nuevo: ~400 src + ~250 tests + ~600 RFC.
