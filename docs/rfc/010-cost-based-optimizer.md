# RFC 010: Cost-Based Optimizer — Foundation

**Status:** draft
**Author(s):** Matías Fonseca <info@namidb.com>
**Supersedes:** —

## Summary

Fija la base del cost-based optimizer (CBO) que cierra el gate de
LDBC SNB Interactive dentro de 2× de Kuzu. El alcance de **esta** RFC es
solamente la **fundación**: un catálogo de estadísticas derivado del
`Manifest`, una rutina de estimación de cardinalidad por operador, una
rutina de estimación de selectividad de predicates, y `EXPLAIN VERBOSE`
con números. Los rewrites estructurales (predicate pushdown, join reorder,
hash-conversión de `SemiApply`/`CrossProduct`) son out-of-scope explícito
de esta RFC; se encadenan en RFCs siguientes sobre esta base.

La RFC se publica como **draft pre-implementation** para alinear shape y
fórmulas antes de quemar decisiones de rewrite. Las cifras concretas
salen del fixture LDBC SNB micro-graph (6 Person / 8 Message / 4 Comment)
y de las estructuras `PropertyColumnStats` / `DegreeHistogram` que ya
viven en cada `SstDescriptor`.

Out-of-scope explícito de esta RFC:

- Predicate pushdown / filter merging.
- Join-order DP/greedy sobre `Expand` chains y `CrossProduct`.
- Conversión `SemiApply` → `HashSemiJoin` y `CrossProduct` con shared
 bindings → `HashJoin` (ver RFC-011).
- Histogramas equi-depth o quantiles para selectividad de rangos
 precisa.
- HyperLogLog actually populated por el writer (hoy `ndv_estimate` es
 siempre `None` — la plomería existe, el cómputo todavía no).
- Adaptive / runtime cost feedback.
- PROFILE con observed cardinality post-ejecución.
- Cost model multi-namespace / partition-aware.

## Motivation

El executor naïve inicial ejecuta cualquier `LogicalPlan` válido
correcta y deterministamente. El problema visible es:

1. **Multi-pattern MATCH naïve.** `MATCH (a:Person {id: $x}),
 (b:Message {id: $y})` se baja a `CrossProduct { NodeById, NodeById }`.
 Sin reorder el outer puede ser el lado pesado; con `SemiApply` el outer
 nested-loop reejecuta el subplan |outer| veces sin cache.

2. **EXPLAIN sin números.** El árbol del plan es indentado pero no dice
 cuántas filas espera procesar cada operador. Sin números, ningún
 rewrite tiene base para decidir "este Expand explota a 10 K rows,
 conviene pushear el Filter antes".

3. **`PropertyColumnStats` + `DegreeHistogram` sin consumer.** Las dos
 estructuras viven en cada `SstDescriptor`. El writer las puebla con
 `min`/`max`/`null_count` (HLL todavía no), pero ningún consumer las
 lee — son data dormida.

El costo de no hacerlo ahora es:

- Los rewrites posteriores tendrían que inventar su propio cost model
 inline, contaminando cada paso con lookups de stats.
- Las decisiones de pushdown/reorder se tomarían a ciegas (heurísticas
 sin números), reproduciendo el problema "Cypher.runtime=slotted" de
 Neo4j: optimizaciones que parecen razonables pero pierden en queries
 reales.
- LDBC SNB SF1 (gate) no se puede preparar sin una baseline
 numérica que diga *dónde* el plan actual gasta tiempo.

Hacerlo ahora cuesta ~1 500 LoC (módulo `cost::`, EXPLAIN VERBOSE, smoke
tests) y abre la puerta a rewrites sin refactor.

## Design

### 1. Catálogo de estadísticas (`StatsCatalog`)

```rust
// crates/namidb-query/src/cost/stats.rs

pub struct StatsCatalog {
 labels: BTreeMap<String, LabelStats>,
 edge_types: BTreeMap<String, EdgeTypeStats>,
 /// Total nodes across all labels — usado como denominador para
 /// estimaciones de patrones anónimos (label desconocido).
 total_nodes: u64,
 /// Total edges across all edge types — análogo para edges
 /// anónimos.
 total_edges: u64,
}

pub struct LabelStats {
 pub name: String,
 /// Σ row_count - tombstone_count sobre SSTs del label (no incluye
 /// memtable: el catálogo se construye desde Manifest committed).
 pub node_count: u64,
 /// Propiedad → estadísticas por columna. Se mergean per-name a
 /// través de todos los SSTs del label.
 pub properties: BTreeMap<String, PropStats>,
}

pub struct PropStats {
 pub null_count: u64,
 pub non_null_count: u64,
 pub min: Option<StatScalar>, // reusado de storage::sst::stats
 pub max: Option<StatScalar>,
 /// NDV decodificado del HLL fused; `None` cuando el writer no
 /// pobló el sketch (caso default en v0).
 pub ndv: Option<u64>,
}

pub struct EdgeTypeStats {
 pub name: String,
 /// Σ row_count - tombstone_count sobre SSTs `EdgesFwd` del tipo.
 pub edge_count: u64,
 /// avg_degree para src → dst, derivado de degree_histogram fused.
 /// Si no hay SST `EdgesFwd`, es 0.
 pub avg_out_degree: f64,
 pub max_out_degree: u64,
 /// idem para EdgesInv (dst → src).
 pub avg_in_degree: f64,
 pub max_in_degree: u64,
 /// Schema-declared endpoints. `None` cuando no hay schema explícito
 /// (caso típico hoy: las queries inferieron edge_type del pattern).
 pub src_label: Option<String>,
 pub dst_label: Option<String>,
}
```

**Construcción:**

```rust
impl StatsCatalog {
 pub fn from_manifest(m: &Manifest) -> Self;
 pub fn empty() -> Self; // fallback cuando el query corre sin Snapshot
 pub fn label(&self, name: &str) -> Option<&LabelStats>;
 pub fn edge_type(&self, name: &str) -> Option<&EdgeTypeStats>;
 pub fn total_nodes(&self) -> u64;
 pub fn total_edges(&self) -> u64;
}
```

**Merge de stats per-label**: itera `m.ssts` filtrando por
`kind == SstKind::Nodes && scope == label`. Para cada `SstDescriptor`:

- `node_count += row_count - tombstone_count` (`KindSpecificStats::Nodes`).
- Para cada `PropertyColumnStats`:
 - `null_count += sst.null_count`.
 - `non_null_count += (row_count - tombstone_count - null_count)`.
 - `min = stat_min(self.min, sst.min)` (lex-order según tipo).
 - `max = stat_max(self.max, sst.max)`.
 - `ndv`: cuando los SSTs traen HLL (v1 follow-up), fuse; v0 → `None`.

**Merge de stats per-edge_type**: itera `m.ssts` filtrando por
`(EdgesFwd, edge_type)` y `(EdgesInv, edge_type)`:

- `edge_count = Σ row_count(EdgesFwd) - Σ tombstone_count(EdgesFwd)`.
- `avg_out_degree = sum_degree(EdgesFwd) / key_count(EdgesFwd)` (Σ y Σ).
- `max_out_degree = max(max_degree(EdgesFwd))` across SSTs.
- idem para `EdgesInv` → `avg_in_degree`, `max_in_degree`.
- `src_label` / `dst_label`: lookup `m.schema.edge_type(name)`; si no
 hay declaración, `None`.

**Coste de construcción**: O(|ssts|). En un manifest real típico
(1 M nodos / 1 M edges sobre R2) son ~10² SSTs — micro-segundos.
Para SF1 LDBC (~3 M nodes / 17 M edges) serán ~10³ SSTs, sigue siendo
sub-milisegundo. El catálogo se construye **una vez por `Snapshot`** y
se reutiliza para todas las optimizaciones del plan; no es hot-path.

**Edge case — schema vacío + zero SSTs (CLI ephemeral `namidb run`):**
el catálogo retorna `LabelStats::empty()` para cualquier label
solicitado. La cardinalidad cae al fallback default (ver §3.4) y EXPLAIN
VERBOSE marca el nodo con `(no stats)`. Esto permite que `namidb
explain --verbose` funcione sin datos cargados, útil para debugging del
plan shape.

### 2. Selectividad de predicates (`cost::selectivity`)

Función pura: dado un `Expression`, un `LabelStats` (o tabla de
`LabelStats` por alias) y un mapa de tipos opcional, retorna la
fracción esperada de filas que satisface el predicate.

```rust
pub fn selectivity(
 expr: &Expression,
 bindings: &BindingStats,
) -> f64;

pub struct BindingStats<'a> {
 /// alias → LabelStats. None cuando el alias no está bound a un
 /// label conocido (Argument / Project synthetic / etc).
 pub by_alias: BTreeMap<String, &'a LabelStats>,
}
```

**Reglas (v0):**

| Predicate | Estimación |
|----------------------------------------|----------------------------------------------------------------|
| `prop = literal` | `1 / ndv(prop)` si hay HLL; `0.1` fallback (10 %). |
| `prop <> literal` | `1 - eq_sel(prop, literal)`. |
| `prop < literal` | rango sobre `[min, max]` si min/max + tipo numérico; `0.33`. |
| `prop <= literal` / `prop > literal` / `prop >= literal` | mismo trato que `<`. |
| `prop BETWEEN low AND high` | rango bilateral; `0.25` fallback. |
| `prop IN [list]` | `min(1, len(list) / ndv)`; `min(1, len(list) * 0.1)` fallback.|
| `prop IS NULL` | `null_count / (null_count + non_null_count)`; `0.05` fallback.|
| `prop IS NOT NULL` | `1 - is_null_sel`. |
| `prop STARTS WITH 'p'` | `0.1` (sin tries / sin index). |
| `prop CONTAINS 'p'` / `ENDS WITH 'p'` | `0.1`. |
| `prop LIKE 'pattern'` (no soportado) | n/a. |
| `__label_eq(alias, L)` | fold pre-Filter — siempre `1.0` (el operador ya garantiza). |
| `AND` | producto: `sel(left) * sel(right)`. Asume independencia. |
| `OR` | unión: `sel(left) + sel(right) - sel(left)*sel(right)`. |
| `NOT (pred)` | `1 - sel(pred)`. |
| `XOR` | `sel(left) + sel(right) - 2*sel(left)*sel(right)`. |
| Cualquier otro caso | `0.5` (unknown). |

**Independencia**: asumimos columnas independientes — clásico Selinger
'79. Es un fallback, no un teorema; selectividades correlacionadas
quedan para más adelante (multi-column histograms).

**Rangos**: para `prop < lit` y un `PropStats { min, max }` numérico,
`sel = clamp01((lit - min) / (max - min))`. Si `min == max`, retorna
`1.0` cuando `min < lit` y `0.0` otherwise (degenerate column).

**Tipos no comparables** (e.g. `min: Utf8`, `lit: Int64`): el selector
cae al fallback `0.33`. La selectividad nunca propaga errores —
robustez sobre exactitud.

**Tabla rationale**: los defaults de la columna derecha siguen el
folklore PostgreSQL `default_statistics_target=100` calibrado para
queries OLTP, no porque sean "verdad", sino porque son el menor mal en
ausencia de stats reales. En particular el `0.1` para `eq` es el
"selectividad agresiva" que prefiere planes index-friendly cuando hay
duda. Se documentan acá para auditarlas después.

### 3. Estimación de cardinalidad por operador

Función pura sobre el árbol:

```rust
pub fn estimate(plan: &LogicalPlan, catalog: &StatsCatalog) -> Cardinality;

pub struct Cardinality {
 /// Filas estimadas que emite este nodo.
 pub rows: f64,
 /// Cardinalidad de los inputs, en mismo orden que `plan.children()`.
 pub children: Vec<Cardinality>,
 /// Bindings que el operador deja "vivos" downstream, junto con la
 /// `LabelStats` asociada cuando se conoce. Heredado por el padre.
 pub bindings: BTreeMap<String, BindingMeta>,
}

pub struct BindingMeta {
 /// Cuando el binding está bound a un nodo de un label conocido,
 /// referenciamos esa LabelStats por nombre. (No anidamos el
 /// borrow porque `Cardinality` es owned.)
 pub label: Option<String>,
 /// Cuando el binding es de un edge.
 pub edge_type: Option<String>,
}
```

#### 3.1 Operadores leaf

| Operador | Cardinalidad |
|---------------------------|---------------------------------------------------------------------------|
| `Empty` | `1.0` (single driver row; consistente con `RETURN 1+1` retornando 1). |
| `Argument { bindings }` | `1.0` (placeholder de outer; siempre exactamente una fila). |
| `NodeScan { label }` | `catalog.label(label).node_count` (0 si no hay stats). |

#### 3.2 Operadores con un input

| Operador | Cardinalidad |
|------------------------|------------------------------------------------------------------------------------------------------------------|
| `NodeById { input, .. }` | `min(input.rows, 1.0)` cuando `input.rows >= 1`. Si `input` es `Empty`, `1.0`. Punto-lookup, asume hit típico. |
| `Expand { input, edge_type, direction, optional }` | `input.rows * branch_factor(edge_type, direction)` + `if optional && branch == 0 { input.rows }`. |
| `Filter { input, predicate }` | `input.rows * selectivity(predicate, bindings)`. |
| `Project { input, distinct: false }` | `input.rows` (projection no cambia cardinalidad). |
| `Project { input, distinct: true }` | `dedup_estimate(input)` — `min(input.rows, Π ndv(item))` cuando los items son props con NDV; fallback `input.rows^0.7`. |
| `Distinct { input }` | `dedup_estimate(input)`. |
| `Aggregate { input, group_by, .. }` | si `group_by.is_empty()`: `1.0`. Si no: `Π ndv(group_by_i)` truncado a `input.rows`; fallback `input.rows ^ 0.5`. |
| `TopN { input, skip, limit, .. }` | `min(input.rows - skip, limit)` clamp a `[0, input.rows]`. |
| `Unwind { input, list }` | `input.rows * avg_list_length(list)` — para `list = Literal::List(xs)` usamos `xs.len()`; para `Parameter` o `Variable` usamos default `5.0`. |
| `PatternList { input, subplan, .. }` | `input.rows` (emite una row por outer; la lista es value, no rows). |
| `Argument`-like wrappers | identidad. |

#### 3.3 Operadores con dos inputs

| Operador | Cardinalidad |
|-----------------------------|--------------------------------------------------------------------------------------------------------------|
| `CrossProduct { left, right }` | `left.rows * right.rows`. Si comparten un binding, un rewrite posterior lo convierte a `HashJoin` y la fórmula cambia. |
| `Union { left, right, all: true }` | `left.rows + right.rows`. |
| `Union { left, right, all: false }` | `dedup_estimate(left + right)` aproximado como `max(left.rows, right.rows) + 0.5 * min(...)`. |
| `SemiApply { input, subplan, negated: false }` | `input.rows * min(1.0, subplan.rows)` — naïve probabilidad de match. |
| `SemiApply { input, subplan, negated: true }` | `input.rows * max(0.0, 1.0 - subplan.rows)`. |

#### 3.4 `branch_factor(edge_type, direction)` para `Expand`

Cuando `edge_type` está declarado:

```
if direction == Right (out):
 branch = catalog.edge_type(et).avg_out_degree
elif direction == Left (in):
 branch = catalog.edge_type(et).avg_in_degree
elif direction == Both:
 branch = avg_out_degree + avg_in_degree
```

Cuando `edge_type` es `None` (anonymous `-[]-`): suma sobre todos los
edge_types `Σ avg_*_degree`. Fallback default `2.0` cuando no hay stats.

Para `Expand { length: Some(l) }` (variable-length): `branch ^ l.max`
hasta cap `MAX_VARLEN_BRANCH = 10_000` (para que `*1..6` no exploten el
estimate a infinito en grafos densos). Esta es la fórmula naive de
DuckDB-graph; mejora con Markov / random-walk va a futuro con WCOJ.

#### 3.5 Operadores write

Los `Create/Merge/Set/Remove/Delete` retornan `0.0` rows (el executor
no emite tuplas; emite `WriteOutcome`). Su `input` mantiene su
cardinalidad para EXPLAIN VERBOSE pero el operador write en sí es
"sink".

#### 3.6 Bindings y heredado

- `NodeScan { label, alias }` introduce `alias → BindingMeta { label, .. }`.
- `NodeById` idem.
- `Expand { target_alias, target_label, .. }` introduce `target_alias`
 con `label = target_label` cuando el lowering lo declaró.
- `Project { distinct, items, discard_input_bindings: true }` reemplaza
 el set de bindings con los aliases del proyectado (sin LabelStats
 asociada, salvo que el item sea `Variable(x)` con `x` un alias
 pre-existente).
- `Project { discard_input_bindings: false }` (WITH) merge: agrega los
 aliases de los items sobre los heredados.
- Los demás operadores (`Filter/TopN/Distinct/Unwind/...`) heredan
 bindings sin modificar.

### 4. `EXPLAIN VERBOSE`

Nueva función `explain_verbose(plan, catalog) -> String` que extiende
`explain(plan)` con cardinalidad estimada por nodo y costo total.

**Formato:**

```
TopN keys=[m.creationDate DESC, m.id ASC] limit=20 (est=20)
 Project [...] (est=20)
 Expand source=p edge_type=KNOWS dir=-> target=friend (est=180)
 NodeById label=Person id=$personId (est=1)
 Empty (est=1)
```

**Convenciones:**

- `(est=N)` redondea `f64` a entero positivo (ceil cuando `0 < x < 1`,
 para no mostrar "est=0" a un operador que sí emite filas).
- Para nodos cuyo `LabelStats` no existe en el catálogo, se agrega
 ` (no stats)` después del `(est=...)`.
- El header del root incluye total: `# Estimated rows: N` antes del
 árbol.

**Total cost (informativo):** Σ rows sobre todos los nodos. No es un
cost en sentido fuerte (no factoriza CPU vs IO), es una baseline para
comparar plans pre y post-rewrite. Futuras iteraciones refinarán si
necesario.

**Parser**: `EXPLAIN VERBOSE <query>`. Sintaxis:

- `EXPLAIN` sin VERBOSE: comportamiento actual (sin números).
- `EXPLAIN VERBOSE`: agrega `Query.explain_verbose: bool` flag (además
 del `explain: bool` existente). `Display for Query` round-trips.
- `EXPLAIN VERBOSE` exige stats; cuando se invoca sin Snapshot (CLI
 ephemeral), usa `StatsCatalog::empty()` y todos los nodos se marcan
 `(no stats)`. No es error.

### 5. Integración con CLI

`namidb explain --verbose <cypher>` activa el flag. La query string
tampoco necesita el `VERBOSE` prefix (`--verbose` lo inyecta).

`namidb run <cypher>` sigue siendo read/write como hoy; el cost
model **no** afecta la ejecución en esta versión (no hay rewrites
todavía).

### 6. API pública del crate

```rust
// crates/namidb-query/src/cost/mod.rs
pub mod stats;
pub mod selectivity;
pub mod cardinality;

pub use stats::{StatsCatalog, LabelStats, EdgeTypeStats, PropStats};
pub use selectivity::selectivity;
pub use cardinality::{estimate, Cardinality, BindingMeta};

// Re-exports desde lib.rs
pub use crate::cost::{StatsCatalog, estimate};
pub use crate::plan::explain_verbose;
```

## Alternativas consideradas

### A. Inferir stats del primer `scan_label` (lazy)

Levantar el catálogo cada vez que el optimizer toca un operador con
label desconocido. Rechazado: triple-pago de IO si dos ramas del plan
hablan del mismo label, y rompe el invariante "todo el plan se optimiza
antes de empezar a ejecutar" (necesario para correctness de pushdown).

### B. Cost en BigDecimal / fixed-point

`f64` puede acumular error de redondeo en plans de 10+ operadores.
Rechazado: el error relativo de un f64 sobre 10 operaciones está en
~10^-13, varios órdenes de magnitud por debajo del error de modelo
(asumir independencia ya introduce 10–50 %). El folklore PostgreSQL /
DuckDB usa f64; no inventemos un problema que no existe.

### C. Sketch-only (sin min/max), HLL-everywhere

Hace los rangos imposibles. Rechazado: rangos numéricos sobre
`creationDate` aparecen en 7/14 LDBC IC; sin min/max el estimate del
filter colapsa al fallback 0.33 y matamos el optimizer en queries
date-bounded.

### D. Cost model basado en bytes (DuckDB-style "rows × width")

Multiplicar `rows` × `avg_row_bytes` para tener algo cercano a IO.
Rechazado por ahora: el executor naïve mantiene todo en memoria;
no hay disco-spill ni vectorización donde el ancho importe. Con
morsels y Arrow vectorization sí, y ahí refinamos.

### E. Manifest-side reporta `StatsCatalog` ya armado

Mover `from_manifest` al crate `namidb-storage`. Rechazado: el
catálogo lo consume el query layer; mantenerlo en `namidb-query`
preserva separation of concerns y permite que el storage lib quede
agnóstico de PropStats con NDV (que es concepto de query). El storage
expone `Manifest`, `SstDescriptor`, `PropertyColumnStats`,
`DegreeHistogram` — primitivas, no agregados.

### F. Pre-construir el catálogo cuando el manifest se carga

El `Snapshot::new` podría construir `StatsCatalog` y exponerlo via
`Snapshot::stats()`. Considerado, **deferido**: requeriría exportar el
tipo cross-crate. Por ahora el caller (executor o CLI) construye el
catálogo a partir de `snapshot.manifest().manifest`. La API
`from_manifest(&Manifest)` queda pura.

## Drawbacks

1. **HLL no poblado → eq selectivity siempre 0.1.** Hoy el writer no
 emite sketches, así que para `prop = literal` el optimizer usa
 fallback aunque haya min/max. Es aceptable v0; HLL real va en
 follow-up (writer side ~200 LoC, cost-side cero).

2. **`avg_degree` es promedio, no mediana.** Distribuciones power-law
 (típicas de social graphs: LDBC SNB tiene exponente ~2.3) hacen que
 el promedio sea engañoso — un fan-out de 100 K en un super-nodo
 eleva el avg sin que la mayoría de nodos lo cumpla. Hoy
 `degree_histogram` está disponible pero no lo usamos en la fórmula
 (los buckets log₂ están ahí para join-order percentile-based futuro).
 Documentado.

3. **Selectividad asume independencia entre columnas.** En LDBC SNB,
 `Person.firstName` y `Person.lastName` son altamente correlacionados
 con `id`; un `WHERE firstName='Alice' AND lastName='Smith'` puede
 ser mucho más selectivo que el producto. A futuro introducimos
 multi-column stats.

4. **No hay sample-based cardinality.** PostgreSQL y CockroachDB hacen
 sampling para columnas con histogramas. Acá no — el writer no
 muestrea y el cost path no lo invoca. Llega a futuro con el morsel
 executor donde sampling es ~free.

5. **Stats viven en el manifest committed → no incluye memtable.** Las
 queries que corren contra una `Snapshot` con memtable activo (caso
 normal de single-writer) usan estimates del manifest sin contar las
 filas no-flushed. Cuando el writer está callado, es ~OK; cuando hay
 ingest activo, el catálogo subestima. Aceptable v0: el writer
 flush-cadence típico es ≤1 GB de memtable, así que el under-estimate
 está acotado. A futuro el vectorized executor agregará
 `memtable_stats` live.

6. **`Cardinality` paraleliza el árbol del plan.** En vez de mutar
 `LogicalPlan` con annotations inline, retornamos un árbol paralelo
 `Cardinality`. Es ~2× memoria del plan pero mantiene `LogicalPlan`
 inmutable (otros consumers — EXPLAIN, executor, future PROFILE — no
 tienen que filtrar las annotations). Trade-off explícito.

## Open questions

- **OQ1.** ¿Selectividad debe ser `f64` o `Probability` (tipo wrapper
 con clamp a [0,1])? Hoy es `f64`; v1 considera wrapper si vemos un
 bug por overflow.

- **OQ2.** ¿`StatsCatalog::from_manifest` debe ser `async` (por si en
 el futuro lee sketches HLL desde un side-car)? Por ahora se mantiene
 síncrono — todo lo que necesita está in-line en el manifest. Si HLL
 side-car aterriza, se rompe la API y lo trabajamos.

- **OQ3.** ¿Cost total debe ser `Σ rows` (estimate-based) o `Σ rows ×
 per-operator-weight` (CPU model)? Por ahora usamos el primero. El
 segundo llega cuando midamos costo real por operador en el morsel
 executor.

- **OQ4.** Cómo expresamos "shared bindings entre lados de CrossProduct"
 en el modelo. Hoy `CrossProduct` cardinality es `L × R`. Cuando se
 introduzca `HashJoin`, queremos algo como
 `(L × R) / max(ndv(shared_key, L), ndv(shared_key, R))`. La
 estructura de `BindingMeta` ya carga el alias; falta agregar acceso a
 PropStats del binding desde Cardinality.

## References

- Selinger et al., *Access Path Selection in a Relational Database
 Management System* (SIGMOD '79) — origen del cost-based optimizer y
 del fallback 0.1.
- Heimel et al., *Hardware-Oblivious Parallelism for In-Memory
 Column-Stores* — defaults modernos para selectividad sin index.
- PostgreSQL `default_statistics_target` documentation — fuente de los
 fallbacks numéricos.
- Kuzu paper (Mhedhbi & Salihoglu, SIGMOD '23) — cardinality estimation
 para graph join enumeration via WCOJ; referencia para trabajo futuro.
- DuckDB CBO blog series (Mark Raasveldt 2023) — uso de stats
 inline-en-Parquet para skipping y join-order; mismo patrón que acá.
- HyperLogLog++ paper (Heule et al., EDBT '13) — formato del sketch
 cuando aterrice el writer.
- `docs/rfc/008-logical-plan-ir.md` — operadores que esta RFC anota.
- `docs/rfc/009-write-clauses.md` — write ops que retornan 0 rows.

## Plan de implementación

1. Crate `namidb-query`:
 - `src/cost/mod.rs` — re-exports.
 - `src/cost/stats.rs` — `StatsCatalog`, `LabelStats`,
 `EdgeTypeStats`, `PropStats` + `from_manifest`. ~300 LoC + 6-8
 unit tests.
 - `src/cost/selectivity.rs` — `selectivity(expr, bindings) -> f64`.
 ~250 LoC + 10-12 unit tests (eq/range/IN/AND/OR/NOT/IS NULL/
 STARTS WITH/fallback).
 - `src/cost/cardinality.rs` — `estimate(plan, catalog) -> Cardinality`.
 ~350 LoC + 8-10 unit tests cubriendo cada operator.

2. `src/plan/explain.rs`:
 - `explain_verbose(plan, catalog) -> String`. ~80 LoC + 5 tests.

3. `src/parser/grammar.rs`:
 - Reconocer `VERBOSE` como soft keyword después de `EXPLAIN`.
 - `Query.explain_verbose: bool`. Display round-trips.
 - ~30 LoC + 3 tests.

4. CLI:
 - `namidb explain --verbose <cypher>`. ~15 LoC.

5. Tests integration:
 - `tests/cost_smoke.rs` — micro-graph → `StatsCatalog::from_manifest`,
 `estimate(plan)` vs `execute(plan).len()`. Documentar gap. 6-8
 tests cubriendo IC2/IC7/IC8/IC9 + filter selectivity sweep.

Snapshot esperado:
- `cargo test --workspace --exclude namidb-py`: 348 → ~390 passed.
- `cargo clippy --workspace --all-targets -- -D warnings`: clean.
- `cargo fmt --all -- --check`: clean.
- LoC nuevo: ~1 500 src + ~500 tests.
- Sin cambios en `namidb-storage` (consumer-only).
