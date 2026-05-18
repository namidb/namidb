# RFC 008: Logical Plan IR

**Status:** draft
**Author(s):** Matías Fonseca <info@namidb.com>
**Supersedes:** —

## Summary

Define la representación intermedia (IR) que el query engine usa entre el
AST de Cypher (RFC-004) y el executor. El IR es un árbol de operadores
relacionales extendidos para grafos — el shape estándar de los DBMS modernos
(DuckDB, DataFusion, Materialize, Kùzu) — adaptado al modelo
property-graph. Esta RFC fija los operadores, su semántica, las reglas de
lowering desde Cypher, el tipo runtime `RuntimeValue` y la API del executor
naïve inicial (tree-walking, eager `Vec<Row>`).

Out-of-scope explícito en la versión inicial: streaming/morsel-driven
execution, cost-based optimizer, WCOJ planner, parallelism, distribución
multi-namespace, query result caching.

## Motivation

Sin un IR estable, lowering, optimizer y executor terminan acoplados a la
forma del AST. Esto duele en tres dimensiones:

1. **Optimizer imposible de injertar.** Para rewrite predicate
 pushdown / join-order / projection-elimination necesitamos un árbol que
 sepa hablar de `Filter(input, pred)` y `Project(input, items)` como
 operadores independientes, no como cláusulas anidadas.

2. **Executor morsel-driven no puede compartir código.** El executor
 vectorizado va a operar sobre el mismo árbol de operadores que el naïve;
 solo cambia la representación de filas (Arrow `RecordBatch` vs `Vec<Row>`)
 y la estrategia de scheduling. Si el IR es estable, la versión
 vectorizada reemplaza solo la implementación de `Operator::execute`.

3. **EXPLAIN/PROFILE necesitan algo qué imprimir.** Sin IR, el `EXPLAIN`
 tendría que recorrer el AST y traducirlo on-the-fly cada vez. Con IR
 imprimimos el árbol directo.

El costo de hacerlo de entrada (vs diferirlo) es ~700–1 000 LoC y una
iteración de design. El costo de diferirlo es refactor obligatorio cuando
entren optimizer y executor vectorizado — peor.

## Design

### Tipo runtime: `RuntimeValue`

`namidb-core::Value` cubre los escalares (`Null/Bool/I64/F64/Str/Bytes/Vec<f32>`)
pero le faltan los compuestos que Cypher necesita: `LIST`, `MAP`, `NODE`,
`RELATIONSHIP`. Definimos `RuntimeValue` standalone en `namidb-query`
para mantener `core` agnóstico del query layer:

```rust
pub enum RuntimeValue {
 Null,
 Bool(bool),
 Integer(i64),
 Float(f64),
 String(String),
 List(Vec<RuntimeValue>),
 Map(BTreeMap<String, RuntimeValue>),
 Node(Box<NodeValue>),
 Rel(Box<RelValue>),
 // Date / DateTime / Duration: stubs iniciales; semantics completas más adelante.
 Date(i32), // days since 1970-01-01
 DateTime(i64), // microseconds since 1970-01-01T00:00:00Z
}
```

`NodeValue` y `RelValue` envuelven `NodeView` / `EdgeView` del storage:

```rust
pub struct NodeValue {
 pub id: NodeId,
 pub label: String,
 pub properties: BTreeMap<String, RuntimeValue>,
}

pub struct RelValue {
 pub edge_type: String,
 pub src: NodeId,
 pub dst: NodeId,
 pub properties: BTreeMap<String, RuntimeValue>,
}
```

Conversiones `From<NodeView>` y `From<EdgeView>` mapean
`core::Value → RuntimeValue` row-wise; esto introduce una copia pero es
aceptable en la versión inicial (la versión vectorizada futura va a operar
directo sobre Arrow batches sin esta conversión).

### Tipo runtime: `Row`

```rust
pub struct Row {
 pub bindings: BTreeMap<String, RuntimeValue>,
}
```

Una `Row` es el estado completo de un binding scope en el current scope.
`MATCH (a)-[r]->(b) RETURN a.name, r.weight, b.id` produce rows con tres
bindings vivos (`a`, `r`, `b`) hasta el `RETURN`, que projecta a una nueva
row con solo `a.name`, `r.weight`, `b.id`.

Decisión `BTreeMap` (no `HashMap`): determinismo en orden de iteración
para tests y `EXPLAIN` output. Lookup `O(log k)` con `k = #bindings` —
inmaterial vs el costo de IO.

### Operadores del IR

Cada operador es una variante de `LogicalPlan`. El árbol es child-pointer
single-input excepto `Union` (dos inputs). Aristas implícitas: cada
operador "produce rows" para su parent.

```rust
pub enum LogicalPlan {
 /// Producer de rows: scan completo de todos los nodes con `label`.
 /// `alias` es el binding que cada NodeValue ocupa en la row de salida.
 NodeScan {
 label: String,
 alias: String,
 },

 /// Variante O(1): scan de un único node por id. Usado cuando el AST
 /// llega con `(p:Person {id: $personId})` — lowering detecta el filtro
 /// trivial y lo convierte en `NodeById` en vez de `Filter(NodeScan, ...)`.
 NodeById {
 label: String,
 alias: String,
 id: Expression, // typically Parameter("personId") or Literal(NodeId)
 },

 /// Toma rows del `input`, expande la binding `source` por sus edges
 /// `direction`/`edge_type`, materializa el destino bajo `target_alias`
 /// y opcionalmente bind la rel en `rel_alias`.
 Expand {
 input: Box<LogicalPlan>,
 source: String,
 edge_type: Option<String>,
 direction: RelationshipDirection,
 rel_alias: Option<String>,
 target_alias: String,
 /// Cuando el AST trae variable-length `*min..max`, este campo
 /// guarda los bounds; lowering decide si genera un único `Expand`
 /// con length o (a futuro) un sub-plan recursivo.
 length: Option<RelationshipLength>,
 },

 /// Selecciona rows que satisfacen `predicate`.
 Filter {
 input: Box<LogicalPlan>,
 predicate: Expression,
 },

 /// Reemplaza el row con una nueva proyección. Mantiene scope abierto
 /// vía la lista de items (cada item es expression + optional alias).
 /// Si `discard_input_bindings = true`, las bindings no proyectadas
 /// se borran (RETURN-style). Si `false`, se conservan (WITH-style).
 Project {
 input: Box<LogicalPlan>,
 items: Vec<ProjectionItem>,
 distinct: bool,
 discard_input_bindings: bool,
 },

 /// Agrupa por `group_by` y aplica las funciones aggregate.
 Aggregate {
 input: Box<LogicalPlan>,
 group_by: Vec<(Expression, String)>, // (key expression, output alias)
 aggregations: Vec<(String, AggregateExpr)>, // (output alias, agg)
 },

 /// Sort + skip + limit fundidos. Si solo hay sort, `skip = 0`,
 /// `limit = u64::MAX`. Si solo hay limit, `keys` es vacío.
 TopN {
 input: Box<LogicalPlan>,
 keys: Vec<OrderKey>,
 skip: u64,
 limit: u64,
 },

 /// Distinct sobre el set completo de columnas visibles.
 Distinct {
 input: Box<LogicalPlan>,
 },

 /// UNION o UNION ALL.
 Union {
 left: Box<LogicalPlan>,
 right: Box<LogicalPlan>,
 all: bool,
 },

 /// Expande una expression-list a multiple rows, una por elemento.
 Unwind {
 input: Box<LogicalPlan>,
 list: Expression,
 alias: String,
 },

 /// Driver inicial sin filas — produce exactamente un row vacío.
 /// Necesario para queries que abren con UNWIND o WITH literal, ni
 /// para subqueries que arrancan independientes.
 Empty,
}

pub struct ProjectionItem {
 pub expression: Expression,
 pub alias: String,
}

pub struct OrderKey {
 pub expression: Expression,
 pub direction: OrderDirection,
}

pub enum AggregateExpr {
 Count { arg: Option<Expression>, distinct: bool },
 Sum { arg: Expression, distinct: bool },
 Avg { arg: Expression, distinct: bool },
 Min { arg: Expression },
 Max { arg: Expression },
 Collect { arg: Expression, distinct: bool },
}
```

### Semántica NULL (three-valued logic)

Misma que Cypher 25 / GQL:

- `NULL OP NULL = NULL` para todo `OP ∈ {=, <>, <, >, ...}`.
- `NULL AND false = false`, `NULL AND true = NULL`, `NULL AND NULL = NULL`.
- `NULL OR true = true`, `NULL OR false = NULL`, `NULL OR NULL = NULL`.
- `Filter(predicate)` descarta rows cuyo predicate evalúa a `NULL` (igual
 que `false`).
- `IS NULL` / `IS NOT NULL` son los **únicos** operadores que devuelven
 `Bool` para input `NULL`.
- Aggregate functions (excepto `count(*)`) **ignoran NULL** en sus inputs.
- Comparison entre tipos incompatibles (e.g. `1 = "x"`) → `NULL` (no error).
- Division by zero entre enteros → error runtime. Entre floats → `NaN`
 (siguiendo IEEE 754; downstream `<` con `NaN` retorna `NULL`).

### Semántica de scope

Cada clause `MATCH`/`OPTIONAL MATCH`/`UNWIND`/`WITH`/`CREATE`/`MERGE`
extiende el scope con nuevas bindings.

- `WITH` **cierra** el scope: bindings no proyectadas se descartan. Es
 el único punto de re-arranque limpio. Cypher fuerza un `WITH` entre dos
 `MATCH` que comparten bindings — esto se controla en el AST, no en el
 IR.
- `OPTIONAL MATCH` propaga `NULL` en todas las bindings cuando el match
 no tiene resultado. Implementado como `Filter` + outer-join semantics
 a futuro — inicialmente se baja a `Expand` con flag `optional` que
 produce rows con bindings `NULL` cuando no encuentra targets.
- Las bindings de una `OrderBy` clausula siguiente a `RETURN` (o `WITH`)
 son las de la proyección, no las pre-proyección. Eso obliga a lower
 `RETURN ... ORDER BY` como `Project + TopN`, no `TopN + Project`.

### Evaluation order garantizado

El executor ejecuta el árbol bottom-up, depth-first. Si un operador tiene
dos entradas (`Union`) ejecuta `left` antes que `right`. Side-effects en
el executor están prohibidos inicialmente (no hay `SET` / `CREATE` /
`DELETE` todavía); cuando lleguen van a operadores dedicados
(`SetProperty`, `CreateNode`, `DeleteNode`) que ejecutan strictly after
todos los reads de la query (o lazy según RFC futuro).

### Lowering rules

Para cada cláusula Cypher del subset RFC-004:

| Cypher | LogicalPlan |
|---|---|
| `MATCH (a:L)` (no patterns más) | `NodeScan { label: "L", alias: "a" }` |
| `MATCH (a:L {id: $x})` (igualdad sobre id) | `NodeById { label: "L", alias: "a", id: Parameter("x") }` |
| `MATCH (a:L {id: $x})` (igualdad sobre otra prop) | `Filter(NodeScan, a.prop = $x)` |
| `MATCH (a)-[r:R]->(b)` | `Expand { input: <prev>, source: a, edge_type: R, dir: Right, rel_alias: r, target_alias: b }` |
| `MATCH (a) WHERE p` | `Filter(<scan>, p)` |
| `RETURN x, y AS z` | `Project { items: [x, z=y], discard_input=true }` |
| `RETURN DISTINCT x` | `Project { distinct: true, ... }` |
| `WITH x, y AS z` | `Project { items: [x, z=y], discard_input=true }` (mismo que RETURN — diferencia es solo si hay clauses siguientes) |
| `WITH x WHERE p` | `Filter(Project(...), p)` |
| `ORDER BY k1, k2 SKIP s LIMIT l` (después de Project) | `TopN { keys: [k1, k2], skip: s, limit: l }` |
| `UNION ALL` | `Union { all: true }` |
| `UNION` | `Distinct(Union { all: false })` |
| `UNWIND list AS x` | `Unwind { input: <prev or Empty>, list, alias: x }` |
| `MATCH a, b` (multiple patterns, mismo `MATCH`) | Cross product: lower `b` con `input = lowered(a)` y sin shared bindings. |

La regla específica para `OPTIONAL MATCH`:

- `OPTIONAL MATCH (a)-[r]->(b)` con `a` ya bindeada del scope anterior:
 lower como `Expand { ..., optional: true }`. Si no hay match, produce
 un row con `r = NULL` y `b = NULL` (preserva el row input).
- Sin variable-length permitido (parser ya lo rechaza, ver
 `RFC-004 §Drawbacks 5`).

### EXPLAIN format

```
Project [name=a.firstName, age=a.age]
 TopN keys=[a.age DESC] skip=0 limit=10
 Filter (a.age > 18)
 Expand source=a edge_type=KNOWS dir=-> target=b
 NodeScan label=Person alias=a
```

Cada operador se imprime en una línea con el nombre del operador, sus
parámetros entre `[...]` o `nombre=value`, y los hijos indentados con
dos espacios. `EXPLAIN` produce esto; `PROFILE` (a futuro) lo decora con
runtime stats (`rows_out`, `time_ms`, `bytes_read`).

### API del executor

```rust
pub async fn execute(
 plan: &LogicalPlan,
 snapshot: &Snapshot<'_>,
 params: &BTreeMap<String, RuntimeValue>,
) -> Result<Vec<Row>, ExecError>;
```

Trae todo a memoria. Eager. Single-thread (tokio current_thread).

`ExecError` cubre: binding not found, type error, parameter not provided,
storage error.

## Alternatives considered

### A. AST → directamente executor (no IR)

**Pro:** menos código, menos boilerplate.
**Con:** acopla executor a AST. Optimizer requeriría refactor
masivo. EXPLAIN tendría que reconstruir el plan en string-time.
**Veredicto:** rechazado. La inversión IR-first es ~300 LoC extra que
ahorra >1000 LoC más adelante.

### B. Push-based dataflow (Materialize-style)

**Pro:** modelo dataflow nativo, encaja con streaming y continuous queries.
**Con:** mucho más complejo. Cada operador es un actor con state +
input/output channels. Overhead alto para queries one-shot. Diferencial
solo aparece en multi-query / streaming scenarios.
**Veredicto:** rechazado; potencial RFC futuro si entramos a
streaming/CDC.

### C. Volcano-style iterator (`trait Operator { fn next(); }`)

**Pro:** estándar en DBMS clásicos (Postgres, MySQL pre-pipelined).
Lazy, low-memory per operator. Streaming natural.
**Con:** sin parallelism. Function-call overhead por row. La industria
moderna (DuckDB, Velox) lo abandonó.
**Veredicto:** rechazado. Inicialmente eager `Vec<Row>` es más simple y
suficiente; a futuro vamos directo a morsel-driven, no Volcano.

### D. DataFusion como IR

**Pro:** maduro, optimizer "para free", compatibilidad con Arrow.
**Con:** DataFusion es relacional, no graph-shaped. Adaptar `Expand`,
multi-hop, WCOJ a DataFusion es trabajo grande y nunca natural.
**Veredicto:** rechazado como **IR único**. A futuro lo cableamos como
**bridge para SQL surface paralelo** (graph queries en nuestro IR,
SQL surface en DataFusion, mismo executor).

### E. Single-input vs multi-input operators

Decisión: single-input excepto `Union`. `Join` (Hash, NL, LFTJ) es
explícito multi-input pero **no aparece inicialmente** (lowering produce
`Expand` chain, no joins). Joins entran cuando el optimizer re-ordene.

## Drawbacks

1. **`RuntimeValue` introduce conversión row-by-row vs Arrow.** Aceptable
 inicialmente (correctness-first); la versión vectorizada elimina la
 conversión midiendo sobre `RecordBatch` directo. Mientras tanto, hot
 loops convierten `BTreeMap<String, core::Value> → BTreeMap<String,
 RuntimeValue>` por cada NodeView accedida.

2. **`Empty` operator + `NodeById` son corner cases.** Podrían vivir como
 casos especiales del `NodeScan`, pero declararlos explícitos en el IR
 los hace inspeccionables en `EXPLAIN` y trivial de optimizar después.

3. **OPTIONAL MATCH como flag en `Expand`** mezcla orthogonality (left
 outer join semantics) con sintaxis (cypher-specific clause). A futuro
 probablemente lo refactorizamos a `LeftOuterExpand` o un explicit
 `LeftJoin` operator cuando el optimizer lo necesite.

4. **`Distinct` sobre el row entero** no permite optimizar `DISTINCT col`
 donde solo necesitamos uniqueness de una columna. Optimización
 diferida.

5. **Lowering errors no son recuperables** — un solo `BindingNotFound`
 aborta el plan. En contraste, parser tiene multi-error recovery.
 Aceptable: semantic errors son menos frecuentes que typos sintácticos
 y queremos fail-fast.

## Addendum — `SemiApply`, `Argument`, `PatternList`

Tres operadores adicionales al IR para soportar pattern predicates,
pattern comprehensions y back-references a outer scope:

- **`Argument { bindings: Vec<String> }`** — single-row placeholder
 cuyas bindings se cargan desde el outer scope. Aparece como leaf de
 subplans dentro de `SemiApply` o `PatternList`. El executor materializa
 `vec![row]` donde `row` copia las bindings nombradas desde el outer.

- **`SemiApply { input, subplan, negated }`** — semi-join existencial.
 Para cada row producida por `input`, ejecuta `subplan` parametrizado
 por el row (vía `outer_row`); mantiene la row iff el subplan emitió
 ≥1 (positivo) ó =0 (negated). Reemplaza la semántica `Filter(Exists(...))`
 con un operador dedicado. Pendiente: convertir nested-loop semi-apply
 a hash-semijoin cuando hay >N rows.

- **`PatternList { input, subplan, projection, alias }`** — materializa
 una `RuntimeValue::List` por outer row. Para cada row, ejecuta
 `subplan` parametrizado por la row, evalúa `projection` sobre cada
 inner row, colecta a una lista y bindea a `alias` en la row outer.
 Es el lowering de `[(pattern) WHERE p | proj]` cuando aparece como
 top-level projection item.

### Lowering rules adicionales

- **WHERE con EXISTS**: descompone el AND-tree del predicate. Cada
 término que es `Exists(pattern)` o `NOT Exists(pattern)` se extrae
 a un `SemiApply` chained sobre el input plan; los residuos se
 reconstruyen como `Filter` encima de la chain. Casos no soportados
 en v0: `Exists` dentro de `OR`, `CASE`, doble negación, etc. →
 `UnsupportedFeature`.

- **Pattern comprehension top-level**: hoist a `PatternList` con alias
 sintético `__pcN`, substitute la comprehension expression por
 `Variable(__pcN)` en el item de la projection.

- **Aggregate nesting** (e.g. `head(collect(x))`): el lowering walk
 recursivo cada item expression, hoist cada aggregate function call a
 un alias sintético `__aggN` con la `AggregateExpr` correspondiente,
 substituye la call por `Variable(__aggN)`. Group keys = items que no
 contienen ningún `__aggN`. Items con agg-nesting se evalúan sobre la
 row post-Aggregate.

- **`RETURN *` / `WITH *`**: expande `ExpressionKind::Star` a una
 projection item por cada binding nombrada visible en `LowerCtx`
 (skip `__anon*`). Cierra RFC-004 Q1.

- **Back-reference de head pattern**: cuando `(a)` reutiliza una
 binding ya en scope y no hay input plan, emite `Argument { bindings: [a] }`
 en vez de `Empty`. Esto permite que un subplan de `SemiApply`/
 `PatternList` reciba la binding outer al ejecutarse.

### Out-of-scope todavía (pendiente para versiones futuras)

- Pattern comprehensions nested dentro de scalar functions
 (`size([(a)-[]->(b)|b.name])`).
- `EXISTS` fuera del AND-root del WHERE (dentro de OR/CASE/etc).
- Path bindings (`p = (a)-[*]->(b)`) + path materialization.
- Write clauses (CREATE/MERGE/SET/REMOVE/DELETE).

## Open questions

- **Q1: ~~Pattern predicates como sub-plans.~~** ✅ Cerrada vía
 `SemiApply` + `Argument`. La optimización a hash-semijoin queda pendiente.

- **Q2: Variable-length patterns sin variable-length operator.**
 Inicialmente podemos pasar `length: Option<RelationshipLength>` al
 `Expand` y dejar que el executor itere `length.min..=length.max`
 iterations. Eso funciona pero no escala. ¿Variable-length explícito
 como operador separado (`Traverse`) a futuro con WCOJ? Probable sí.

- **Q3: Materialización de paths.** `MATCH p = (a)-[*]->(b)` requiere
 que `p` sea materializable como List. Diferido.

- **Q4: ~~`WITH *` y `RETURN *`.~~** ✅ Cerrada vía
 `expand_star_items` en el lowering.

- **Q5: Hoist de pattern comprehensions nested.** Hoy solo top-level
 en projection items. Hoist nested requiere planning de orden de
 evaluación y bookkeeping de scopes intermedios. Diferido.

## References

- DuckDB logical/physical plans —
 https://duckdb.org/docs/sql/query_syntax/select.html (architecture
 notes en el repo de DuckDB).
- Kuzu morsel-driven execution — Boncz et al., CIDR 2024 paper
.
- Materialize/Differential Dataflow operators — McSherry et al., 2013.
- Volcano model — Goetz Graefe, "Volcano—An Extensible and Parallel
 Query Evaluation System", IEEE TKDE 1994.
- Cypher openCypher 9 §Section 3 (Linear queries semantics).
- GQL ISO/IEC 39075:2024 §17 (Linear queries) y §18 (Composite queries).
