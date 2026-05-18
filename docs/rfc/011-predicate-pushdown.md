# RFC 011: Predicate Pushdown + Filter Normalization

**Status:** draft
**Author(s):** Matías Fonseca <info@namidb.com>
AND split + literal folding + `__label_eq` elimination)
**Builds on:** RFC-010 (cost model foundation)
**Supersedes:** —

## Summary

Primer rewrite estructural del optimizer: empuja cada predicate del
`LogicalPlan` lo más cerca posible de los operadores leaf (NodeScan,
NodeById, Argument, Empty), reduciendo la cardinalidad que los
operadores caros (`Expand`, `CrossProduct`, `SemiApply`,
`PatternList`) procesan. El alcance es **solamente el predicate
pushdown a nivel `LogicalPlan`**: el rewriter manipula la estructura
del árbol; **no** baja predicates al storage layer (Parquet predicate
pushdown queda diferido), **no** reordena joins, **no** convierte
`SemiApply`/`CrossProduct` a hash joins.

Acompañando el pushdown, incluye tres normalizaciones del Filter
tree que el lowering deja sub-óptimo y que el pushdown necesita para
funcionar:

1. **AND-split** — `Filter(a AND b AND c)` se descompone en tres
 conjuntos pushables independientemente.
2. **Adjacent merge** — dos `Filter` consecutivos post-pushdown se
 fusionan en uno con AND, para minimizar nodos en EXPLAIN.
3. **Literal fold** — `Filter(true)` se elimina (`input` directo);
 `Filter(false)` se preserva (no se sustituye por `Empty` porque el
 plan podría seguir requiriendo bindings sin filas; el executor maneja
 3VL).
4. **`__label_eq` cleanup** — el `Filter(__label_eq(target, L))` que el
 lowering inyecta defensivamente arriba de un `Expand` con
 `target_label=Some(L)` se elimina (el operador ya garantiza el label
 en la capa storage).

El contrato público cambia: `lower(query)` sigue siendo puro
(unchanged), pero **`execute` / `execute_write` ahora aplican
`optimize` por default**. EXPLAIN VERBOSE muestra el plan optimizado;
EXPLAIN RAW (nueva sintaxis) muestra el plan literal del lowering.

Out-of-scope explícito:

- Parquet predicate pushdown al storage layer.
- Join-order DP/greedy sobre `Expand` chains y `CrossProduct`.
- Conversión `SemiApply`/`CrossProduct` con shared bindings → HashJoin
 (ver RFC-012).
- Projection pushdown / column pruning.
- Boolean simplification más allá de `true`/`false` literales (De
 Morgan, Karnaugh, common-subexpression).
- HLL populated por el writer (RFC-010 §"Drawbacks 1").

## Motivation

La **fundación** del optimizer (catálogo de stats real, selectividad,
cardinalidad, EXPLAIN VERBOSE) ya está. El gap visible es: el plan que
el lowering produce hoy es estructuralmente naïve y deja trabajo
grueso sobre la mesa.

Ejemplo concreto, consulta LDBC SNB IC-shape:

```cypher
MATCH (a:Person)-[:KNOWS]->(b:Person)
WHERE a.age > 30 AND b.firstName = 'Alice'
RETURN b.id
```

Lowering produce:

```
Project [b.id]
 Filter (a.age > 30 AND b.firstName = 'Alice')
 Filter (__label_eq(b, "Person"))
 Expand source=a edge_type=KNOWS dir=-> target=b
 NodeScan label=Person alias=a
```

Sobre el micro-graph LDBC (6 Person, avg_degree=1, age:[25,40]), eso
expande 6 Person × 1 = 6 pairs antes de filtrar — pequeño, pero la
forma es la misma sobre SF1 (3 M Person, avg_degree≈30): 90 M pairs
materializados antes de filtrar al ~17 % (`age > 30 ⇒ sel≈0.67`,
`firstName='Alice' ⇒ sel≈0.10⇒0.067` total). Con pushdown:

```
Project [b.id]
 Expand source=a edge_type=KNOWS dir=-> target=b
 Filter (a.age > 30)
 NodeScan label=Person alias=a
 (Filter b.firstName = 'Alice' queda arriba — refiere target_alias)
```

Filter sobre `a` baja por debajo del Expand (1 M Person → 670 k); el
filter sobre `b` queda por estructura (refiere el alias introducido por
el Expand). El Expand procesa 670 k × 30 = 20 M pairs en lugar de
90 M. **4.5× menos trabajo procesado**, sin tocar storage.

El costo de no hacerlo:

- Cada query LDBC con WHERE compuesto paga el costo de un plan
 estructuralmente subóptimo. SF1 gate inalcanzable.
- Join reorder y hash conversión operan sobre el plan
 optimizado de filters; sin pushdown previo, los algoritmos de
 reorder ven `CrossProduct + Filter combinado` en lugar de
 `Filter ⇒ Subtree`, lo cual oscurece el grafo de joins.
- EXPLAIN VERBOSE hoy muestra `(est=N)` que es matemáticamente
 correcto pero no refleja el plan que el motor podría correr — el
 número es engañoso porque el plan está mal estructurado.

Hacerlo ahora cuesta ~1 500 LoC src + ~700 LoC tests y desbloquea
join reorder y hash conversion.

## Design

### 1. API pública

```rust
// crates/namidb-query/src/optimize/mod.rs

/// Apply the full optimizer pipeline to `plan`. Idempotent — calling
/// `optimize(optimize(p, c), c)` returns a structurally identical plan.
///
/// Today the pipeline consists of `predicate_pushdown` followed by
/// `normalize_filters` (AND-split, adjacent-merge, literal fold,
/// `__label_eq` elimination) iterated to fixpoint (cap 8 rounds).
pub fn optimize(plan: LogicalPlan, catalog: &StatsCatalog) -> LogicalPlan;

/// Push every `Filter` predicate as close to the leaves as possible.
/// Splits AND-conjunctions and dispatches each conjunct independently
/// based on the aliases it references.
pub fn predicate_pushdown(plan: LogicalPlan) -> LogicalPlan;

/// Tidy the Filter tree: merge adjacent Filters into a single AND,
/// fold `Filter(true)` away, drop the `__label_eq` defensive filter
/// when the immediate child is an Expand already constraining the
/// target label.
pub fn normalize_filters(plan: LogicalPlan) -> LogicalPlan;
```

`StatsCatalog` se acepta para que el pipeline pueda usar estimates
para decisiones futuras (`predicate_pushdown` actual no lo necesita —
es estructural — pero ya queda en la firma para evitar romper el
contrato cuando un rewrite futuro lo necesite).

```rust
// crates/namidb-query/src/lib.rs (nuevo)

pub use optimize::{optimize, predicate_pushdown, normalize_filters};

/// Convenience: lower + optimize. Used by the executor and EXPLAIN
/// VERBOSE by default. Tests that want the raw lowering should call
/// `lower(query)` directly.
pub fn plan(query: &Query, catalog: &StatsCatalog) -> Result<LogicalPlan, LowerError> {
 Ok(optimize(lower(query)?, catalog))
}
```

`execute(plan, snapshot, params)` y `execute_write(plan, writer, params)`
no cambian — siguen aceptando un `LogicalPlan` listo. El cambio es en
los **call sites** (CLI, walker bench, tests integration): donde
antes hacían `let p = lower(&query)?;`, ahora hacen
`let p = plan(&query, &catalog)?;`. Tests internos que prueban
operadores específicos (lowering tests, executor unit tests) siguen
usando `lower(query)` directamente.

### 2. Algoritmo `predicate_pushdown`

Single-pass top-down con accumulator. Cada llamada recursiva pasa un
`Vec<Expression>` de predicados pendientes que el caller quiere
empujar hacia abajo. Cada nodo del plan decide cuáles puede absorber
y cuáles devuelve a su parent vía un `Filter` materializado encima.

```rust
fn pushdown_at(plan: LogicalPlan, pending: Vec<Expression>) -> LogicalPlan {
 match plan {
 // Leaf nodes: aplicar pending arriba y terminar.
 LogicalPlan::Empty
 | LogicalPlan::Argument { .. }
 | LogicalPlan::NodeScan { .. } => apply_filters(plan, pending),

 // Filter node: descomponer y propagar.
 LogicalPlan::Filter { input, predicate } => {
 let mut acc = pending;
 for term in split_and_terms(&predicate) {
 acc.push(term);
 }
 pushdown_at(*input, acc)
 }

 // Operadores que introducen aliases — particionar pending por
 // alias-set y propagar lo pushable; el resto queda arriba.
 LogicalPlan::Expand { /* ... */ } => { /* §2.1 */ }
 LogicalPlan::NodeById { /* ... */ } => { /* §2.2 */ }
 LogicalPlan::CrossProduct { /* ... */ } => { /* §2.3 */ }
 LogicalPlan::Project { /* ... */ } => { /* §2.4 */ }
 LogicalPlan::Aggregate { /* ... */ } => { /* §2.5 */ }
 LogicalPlan::Union { /* ... */ } => { /* §2.6 */ }
 LogicalPlan::Unwind { /* ... */ } => { /* §2.7 */ }
 LogicalPlan::SemiApply { /* ... */ } => { /* §2.8 */ }
 LogicalPlan::PatternList { /* ... */ } => { /* §2.8 */ }

 // Barreras — Distinct / TopN no se cruzan porque cambian
 // cardinalidad de forma que el filter pre/post no es semánticamente
 // equivalente.
 LogicalPlan::TopN { /* ... */ }
 | LogicalPlan::Distinct { /* ... */ } => { /* §2.9 */ }

 // Writes son barreras: pending queda arriba, recurse solo en su
 // input con pending vacío.
 LogicalPlan::Create { /* ... */ }
 | LogicalPlan::Merge { /* ... */ }
 | LogicalPlan::Set { /* ... */ }
 | LogicalPlan::Remove { /* ... */ }
 | LogicalPlan::Delete { /* ... */ } => { /* §2.10 */ }
 }
}
```

#### 2.1 `Expand { source, target_alias, rel_alias, target_label, optional, .. }`

El `Expand` introduce `target_alias` y opcionalmente `rel_alias`. Un
predicate puede empujarse al input sii **no referencia ningún alias
introducido por el Expand**. La distinción entre `optional` y
non-optional **no afecta la pushability**: el rule es estructural.
Lo que sí cambia bajo `optional` es la **forma del Filter que queda
arriba** — un predicate sobre `target_alias` post-OPTIONAL ya está
evaluando 3VL contra `NULL` correctamente; no necesitamos invertir su
semántica. (El lowering, además, folds los property/label filters
INSIDE el Expand cuando `optional=true`, así que la situación clásica
"`Filter(b.x > 0) ⇒ OptionalExpand(target=b)`" sólo ocurre con WHERE
explícito del usuario, que es el caso 3VL correcto.)

```rust
let introduced: BTreeSet<String> = {
 let mut s = BTreeSet::new();
 s.insert(target_alias.clone());
 if let Some(r) = &rel_alias { s.insert(r.clone()); }
 s
};
let (pushable, stay) = pending.into_iter()
 .partition(|e| expression_aliases(e).is_disjoint(&introduced));
let new_input = pushdown_at(*input, pushable);
let new_expand = LogicalPlan::Expand {
 input: Box::new(new_input), source, edge_type, direction, rel_alias,
 target_alias, target_label, length, optional,
};
apply_filters(new_expand, stay)
```

#### 2.2 `NodeById { input, alias, .. }`

Introduce `alias`. Idéntico a `Expand` pero con un set de un elemento.

#### 2.3 `CrossProduct { left, right }`

Cada conjunct puede ir a `left`, `right`, o quedarse arriba:

```rust
let left_aliases = produced_aliases(&left);
let right_aliases = produced_aliases(&right);
let mut to_left = Vec::new();
let mut to_right = Vec::new();
let mut keep_top = Vec::new();
for term in pending {
 let refs = expression_aliases(&term);
 let hits_left = !refs.is_disjoint(&left_aliases);
 let hits_right = !refs.is_disjoint(&right_aliases);
 match (hits_left, hits_right) {
 (true, false) => to_left.push(term),
 (false, true) => to_right.push(term),
 (true, true) => keep_top.push(term),
 (false, false) => keep_top.push(term), // constant — safe to keep up
 }
}
```

**Mixed-side equality** (e.g. `a.x = b.y` con `a∈left, b∈right`) queda
en `keep_top`. La inspección de `keep_top` para detectar
**join-candidate** queda como hint visual en EXPLAIN VERBOSE — no
modificamos el IR ni introducimos un `HashJoin` (queda diferido). La
detección es:

```rust
fn is_join_candidate(expr: &Expression, left: &BTreeSet<String>, right: &BTreeSet<String>) -> bool {
 if let ExpressionKind::Binary { op: BinaryOp::Eq, left: l, right: r } = &expr.kind {
 let la = expression_aliases(l);
 let ra = expression_aliases(r);
 let l_side = la.is_subset(left) && ra.is_subset(right);
 let r_side = la.is_subset(right) && ra.is_subset(left);
 return l_side || r_side;
 }
 false
}
```

EXPLAIN VERBOSE anota cada Filter inmediatamente sobre un CrossProduct
con `[join candidate]` cuando `is_join_candidate` true.

#### 2.4 `Project { items, distinct, discard_input_bindings }`

Un alias del input sobrevive arriba del Project **sii** algún
`items[i]` tiene la forma `Variable(x)` con `items[i].alias == x`
(identity projection sin renaming). Si el predicate refiere solo
aliases identidad-proyectados, podemos bajarlo. Si refiere un alias
introducido por la projection (e.g. `expr AS y`), queda arriba —
debajo del Project el alias `y` no existe.

```rust
let preserved: BTreeSet<String> = items.iter().filter_map(|it| {
 if let ExpressionKind::Variable(id) = &it.expression.kind {
 if id.name == it.alias { return Some(id.name.clone()); }
 }
 None
}).collect();
let (pushable, stay) = pending.into_iter()
 .partition(|e| expression_aliases(e).is_subset(&preserved));
```

WITH * (a futuro) podrá relajar esto. Hoy es conservador.

#### 2.5 `Aggregate { group_by, aggregations }`

Análogo a Project, pero con una distinción: predicates que refieren
**aliases de agregaciones** son HAVING semánticos y nunca bajan. Para
group_by keys que son identity (`Variable(x)` con alias `x`),
pushdown OK como pre-aggregate filter.

```rust
let preserved: BTreeSet<String> = group_by.iter().filter_map(|(e, alias)| {
 if let ExpressionKind::Variable(id) = &e.kind {
 if id.name == *alias { return Some(id.name.clone()); }
 }
 None
}).collect();
let agg_aliases: BTreeSet<String> = aggregations.iter().map(|(a, _)| a.clone()).collect();
let (pushable, stay) = pending.into_iter().partition(|e| {
 let refs = expression_aliases(e);
 refs.is_subset(&preserved) && refs.is_disjoint(&agg_aliases)
});
```

#### 2.6 `Union { left, right, all }`

Pushable a ambos lados sii **todos los aliases referenciados existen
en ambos**. Caso típico: post-Union los dos lados proyectan el mismo
schema, así que un Filter sobre la projection sale a ambos sin
ambigüedad. Si un alias falta en un lado, queda arriba.

```rust
let l_aliases = produced_aliases(&left);
let r_aliases = produced_aliases(&right);
let (pushable, stay) = pending.into_iter().partition(|e| {
 let refs = expression_aliases(e);
 refs.is_subset(&l_aliases) && refs.is_subset(&r_aliases)
});
let new_left = pushdown_at(*left, pushable.clone());
let new_right = pushdown_at(*right, pushable);
```

(Cloning pushable es OK — predicates suelen ser pequeños.)

#### 2.7 `Unwind { list, alias }`

Introduce `alias`. Predicate sobre `alias` queda arriba; otros bajan
al input.

#### 2.8 `SemiApply` / `PatternList`

Ambos toman un `input` (outer) y un `subplan` (inner, parametrizado
por la row outer). El **subplan nunca recibe pushdown** del rewriter
— son scopes nested y el pushdown cross-scope requiere correlation
analysis (decorrelation), out-of-scope.

- `SemiApply`: no introduce nuevos aliases visibles arriba (es un
 semi-join, no proyecta). Pending fluye entero a `input`. Subplan
 intacto.
- `PatternList`: introduce `alias` (el valor list). Predicates sobre
 `alias` quedan arriba; otros bajan a `input`.

#### 2.9 `TopN` / `Distinct` (barreras de cardinalidad)

NO se cruzan. Razones:

- `TopN limit=L`: `Filter(p) ⇒ TopN(L)` retorna ≤ L filas filtradas;
 `TopN(L) ⇒ Filter(p)` retorna L filas pre-filter y luego filtra.
 Cardinalidades distintas; rows distintas.
- `Distinct`: para predicates puros (deterministas, sin side-effects)
 el resultado **set** es el mismo, pero permitir el cruce nos obliga
 a verificar la pureza de cada subexpresión. Más seguro mantener
 como barrera v0.

```rust
LogicalPlan::TopN { input, keys, skip, limit } => {
 let new_input = pushdown_at(*input, vec![]);
 let new = LogicalPlan::TopN { input: Box::new(new_input), keys, skip, limit };
 apply_filters(new, pending)
}
```

#### 2.10 Write ops (`Create / Merge / Set / Remove / Delete`)

Barreras. Pending queda arriba (en la práctica el lowering nunca
emite un `Filter` encima de un write — el patrón `MATCH ... WHERE ...
SET` produce `Set { input: Filter { input: ... } }`, no
`Filter { input: Set { ... } }`. La barrera es defensiva).

### 3. Algoritmo `normalize_filters`

Bottom-up. Cuatro reglas, aplicadas en orden:

1. **Recursividad sobre children primero** (post-order).
2. **`Filter { input: Filter { input: x, predicate: p1 }, predicate: p2 }`** →
 `Filter { input: x, predicate: p1 AND p2 }`.
3. **`Filter { input, predicate: Literal::Boolean(true) }`** → `input`.
4. **`Filter { input: Expand { ..., target_alias=A, target_label=Some(L) }, predicate: __label_eq(A, L) }`** →
 `Expand { ... }` (el filter se elimina).

La regla 4 también aplica recursivamente: si después de eliminar el
filter, hay otro `__label_eq` apilado abajo, se elimina. La regla 2
fusiona las cláusulas que el split en pushdown dejó separadas.

`Filter(false)` queda como está — el executor evalúa el predicate
literal y descarta cada row; el optimizer no convierte a `Empty`
porque eso requiere reasoning sobre los bindings que el plan necesita
introducir (e.g. para un downstream Aggregate count(*) = 0).

### 4. Helpers

```rust
/// Set of aliases (Variable identifiers) referenced anywhere in `expr`.
/// Property accesses contribute their target alias. Pattern subqueries
/// (`Exists`, `PatternComprehension`) and list comprehensions are
/// treated as opaque — we return ALL bindings they could possibly
/// reference, by collecting free variables in the inner expression
/// without descending into nested patterns. Conservative: when in
/// doubt, the alias set is wider, so the predicate stays higher up.
fn expression_aliases(expr: &Expression) -> BTreeSet<String>;

/// Set of aliases that `plan` makes visible to its parent.
fn produced_aliases(plan: &LogicalPlan) -> BTreeSet<String>;

/// AND-flatten: `a AND b AND c` → vec![a, b, c]. Used by pushdown to
/// split a compound predicate.
fn split_and_terms(expr: &Expression) -> Vec<Expression>;

/// Concatenate `terms` with binary AND, preserving source order.
/// Returns None if `terms` is empty.
fn and_chain(terms: Vec<Expression>) -> Option<Expression>;

/// If `terms` non-empty, wrap `plan` in a `Filter(AND(terms))`.
/// Otherwise return `plan` unchanged.
fn apply_filters(plan: LogicalPlan, terms: Vec<Expression>) -> LogicalPlan;
```

`produced_aliases` enumera los aliases por tipo de operador:

| Operador | Produce |
|---------------------|-----------------------------------------------------|
| `NodeScan/NodeById` | `{alias}` |
| `Argument` | bindings literales |
| `Expand` | `produced(input) ∪ {target_alias, rel_alias?}` |
| `Filter` | `produced(input)` |
| `Project` | `items.iter().map(|i| i.alias).collect()` |
| `Aggregate` | `group_by.aliases ∪ aggregations.aliases` |
| `TopN`/`Distinct` | `produced(input)` |
| `Union` | `produced(left) ∩ produced(right)` (schema-aware) |
| `Unwind` | `produced(input) ∪ {alias}` |
| `Empty` | `∅` |
| `CrossProduct` | `produced(left) ∪ produced(right)` |
| `SemiApply` | `produced(input)` |
| `PatternList` | `produced(input) ∪ {alias}` |
| Writes | `produced(input) ∪ alias(elements)` |

### 5. Fixpoint

`optimize` corre `predicate_pushdown` + `normalize_filters` en loop
hasta que dos iteraciones consecutivas producen árboles idénticos
(`PartialEq` already derived on `LogicalPlan`). Cap en 8 rondas para
prevenir loops infinitos en caso de bug (cada ronda debería
estrictamente reducir la altura del Filter tree o ser idempotente,
así que >2 rondas indicaría error). Cap se loggea pero no panic.

```rust
pub fn optimize(plan: LogicalPlan, _catalog: &StatsCatalog) -> LogicalPlan {
 let mut current = plan;
 for _ in 0..8 {
 let next = normalize_filters(predicate_pushdown(current.clone()));
 if next == current { return next; }
 current = next;
 }
 current
}
```

### 6. EXPLAIN integration

#### 6.1 EXPLAIN VERBOSE muestra el plan optimizado

`explain_query_verbose(query, catalog)` ahora llama `plan(query,
catalog)` internamente y renderiza el árbol post-optimize. El total
estimate (header `# Estimated rows`) y per-node `(est=…)` reflejan el
plan que el motor realmente correría. Esto cambia el contrato previo
donde EXPLAIN VERBOSE mostraba el lowering crudo — los tests
existentes que dependían de esa forma específica se actualizan.

#### 6.2 EXPLAIN RAW (nueva sintaxis)

`EXPLAIN RAW <query>` y `EXPLAIN RAW VERBOSE <query>` muestran el plan
sin optimizar. Útil para debugging del lowering y para verificar que
el optimizer hizo algo:

```
> EXPLAIN VERBOSE MATCH (a:Person) WHERE a.age > 30 RETURN a
# Estimated rows: 2
Project [a=a] (est=2)
 Filter (a.age > 30) (est=2)
 NodeScan label=Person alias=a (est=6)

> EXPLAIN RAW VERBOSE MATCH (a:Person) WHERE a.age > 30 RETURN a
# Estimated rows: 2
Project [a=a] (est=6)
 Filter (a.age > 30) (est=2)
 NodeScan label=Person alias=a (est=6)
```

En el RAW (lowering crudo) el Filter está bajo el Project, y la
estimación del Project asume que el Filter ya filtró — pero el
operador Project itera 6 rows con el Filter arriba siendo evaluado
después, lo cual es exactamente lo que muestra el árbol. En el
optimizado el Filter está debajo del Project, así el Project itera 2.

#### 6.3 Join-candidate annotation

Cuando un Filter inmediato sobre un CrossProduct contiene una
igualdad cross-side, EXPLAIN VERBOSE agrega `[join candidate]` al
final de la línea del Filter:

```
Filter (a.name = b.name) [join candidate] (est=...)
 CrossProduct (est=...)
 Filter (a.age > 30) (est=...)
 NodeScan label=Person alias=a (est=...)
 Filter (b.age < 50) (est=...)
 NodeScan label=Person alias=b (est=...)
```

Un rewrite posterior detecta el flag y convierte a HashJoin.

### 7. Parser

```text
EXPLAIN [RAW] [VERBOSE] <query>
```

- `EXPLAIN <query>` — lowering crudo, sin estimates.
- `EXPLAIN VERBOSE <query>` — **optimizado**, con estimates.
- `EXPLAIN RAW <query>` — lowering crudo, sin estimates (alias
 explícito del comportamiento legacy).
- `EXPLAIN RAW VERBOSE <query>` — lowering crudo, con estimates.

`RAW` es un soft-keyword reconocido sólo entre `EXPLAIN` y `VERBOSE`
(o `EXPLAIN` y el inicio de la query). No es token reservado, no
rompe queries con una variable llamada `raw`.

`Query.explain_raw: bool` se agrega al AST junto al `explain_verbose:
bool` existente.

### 8. CLI

```bash
namidb explain [--verbose] [--raw] <cypher>
```

- `--raw`: alias de `EXPLAIN RAW` (skip optimize).
- `--verbose`: ya existe, agrega VERBOSE.

Si la query string ya contiene los prefixes, se respeta la mezcla
(flag + prefix son OR'eados).

## Alternativas consideradas

### A. Selinger-style cost-based exhaustivo

Enumerar todas las posiciones donde el Filter puede ir y elegir la de
menor costo. Rechazado: para un plan con N operadores el espacio es
O(N) posiciones por predicate; con K predicates eso es O(N×K). Para
LDBC IC con N≈10, K≈5 son ~50 posiciones — tractable, pero la mejor
posición siempre es "lo más bajo posible" para predicates puros
(propiedad bien conocida: predicate pushdown commutes con cardinalidad
reduction). El cost-based enum sólo aporta cuando los predicates
tienen side effects (no en SQL/Cypher) o cuando hay correlations que
podrían favorecer NO bajar (cross-column correlation — out of scope
hasta que aterricen multi-column histograms).

### B. Rewrite-rule engine genérico (egg / datalog)

Codificar las reglas como rewrites declarativos y dejar que un engine
los aplique a fixpoint. Rechazado para v0: el catálogo inicial de
reglas es 4 normalizaciones + 1 algoritmo (pushdown). Un engine
genérico cuesta ~2 000 LoC de infra para 4 reglas; manual rewrite es
~500 LoC. Cuando lleguemos a 20+ reglas, evaluamos egg.

### C. Pushdown integrado en `lower`

Hacer que el lowering produzca directamente el plan optimizado.
Rechazado: el lowering tiene una responsabilidad clara (AST →
LogicalPlan correcto). Mezclar optimizer rompe testabilidad
unitaria y oculta bugs de lowering detrás de bugs de pushdown.

### D. Filter pushdown solo en `WhereClause`

Procesar el WHERE en `attach_where` y bajar ahí mismo, antes de
generar el Filter. Rechazado: solo cubre el WHERE explícito. Los
property filters (`{key: value}` inline en patterns) producen Filters
**arriba** del Expand también, y el pushdown necesita verlos todos
uniformemente. Además, join reorder opera sobre el plan
post-pushdown — necesita el árbol normalizado.

### E. Bajar al storage layer ahora (Parquet predicate pushdown)

`scan_label(label, predicates: Vec<...>)` lee los row groups del
Parquet con stats min/max + Bloom + Bitmap pushdown. Rechazado por
ahora: requiere extender la API de `Snapshot` con un tipo
`ScanPredicate` neutral, hacer el match en `parquet_loader.rs`, y
agregar tests storage-side. ~800 LoC adicionales que duplican el costo
y son ortogonales al rewrite estructural. Queda diferido, con esta RFC
como pre-requisito (los predicates ya están en su posición ideal
cuando un rewrite futuro los traduzca a Parquet).

## Drawbacks

1. **Cambio de contrato silencioso para callers existentes.** Los
 tests integration que comparan `lower(query)` con un árbol esperado
 siguen funcionando. Los tests que comparan `execute(plan, snapshot)`
 también — el plan optimizado produce el mismo set de rows. Pero
 tests que comparan EXPLAIN VERBOSE output cambian (el plan
 optimizado tiene forma diferente). Mitigación: snapshot tests
 existentes en `explain.rs` se actualizan.

2. **`expression_aliases` es conservador con subqueries.** Para
 `Filter(EXISTS((a)-[]-(b)) AND x > 0)`, el predicate `EXISTS(...)`
 contribuye al alias set TODOS los aliases que la pattern podría
 referenciar. Si el predicate tiene un sub-EXISTS sobre `a` y un
 `x > 0` sobre `a.x` (distintos a y x), el pushdown podría ser
 más fino — hoy es conservador, queda como mejora futura.

3. **No bajamos a través de `TopN` / `Distinct`.** Es una decisión
 conservadora; el caso "filter sobre TopN" con predicate puro es
 pushable, pero queremos primero verificar pureza. Trabajo trivial,
 queda diferido.

4. **Adjacent merge fusiona Filters con spans inconsistentes.** El
 nuevo `Filter` con AND-chain tiene `span` cuya extension cubre los
 spans originales (lo que ya hace `rebuild_and_chain` en
 `lower.rs`). Para error messages downstream (e.g. error en el
 executor), el span podría apuntar a una región más amplia que el
 conjunct específico que falló. Mitigación: el span de cada
 sub-Expression dentro del AND-tree se preserva — el reporter usa
 ese span, no el del Filter root.

5. **El optimizer corre sobre TODOS los queries, incluyendo write.**
 Los write ops son barreras (predicates no bajan a través de
 ellas), pero el rewriter sigue visitándolos para procesar su
 `input`. Costo: ~O(operadores). Para queries grandes (~100 nodos
 del plan), eso es <1 ms — negligible vs la query execution time.

6. **No hay way de skip optimizer en `execute`.** Si un caller
 necesita evitar el optimizer (e.g. para reproducir un bug del
 lowering), debe llamar `lower(query)?` directamente y luego
 `execute(plan, ...)`. La función `plan(query, catalog)` es el
 atajo conveniente, no el único path.

## Open questions

- **OQ1.** ¿Debería `optimize` aceptar un `OptimizerSettings` con
 flags individuales (`enable_pushdown`, `enable_normalize`,
 `enable_label_eq_cleanup`)? Hoy no — un único toggle "todo o nada"
 vía si se llama `optimize` o `lower` directamente. Cuando agreguemos
 más rewrites, evaluamos un settings struct.

- **OQ2.** ¿`__label_eq` cleanup también debería eliminar el filter
 cuando el predicate target está bound por un `NodeScan` con label
 declarado? Hoy sí — el operador ya garantiza el label vía
 `scan_label(L)`. La regla extendida cubre ambos casos sin riesgo.

- **OQ3.** Pure-predicate detection para abrir TopN/Distinct: los
 predicates Cypher son siempre side-effect-free en v0 (sin funciones
 externas). Podríamos bajarlos sin verificar. Decisión: ser
 conservadores hasta que aterricen funciones externas (RFC futuro).

## References

- Mumick & Pirahesh, *Implementation of Magic-sets in a Relational
 Database System* (1994) — origen de las técnicas de pushdown.
- Selinger et al., *Access Path Selection in a Relational Database
 Management System* (SIGMOD '79) — cost-based optimizer fundacional.
- DuckDB *Predicate Pushdown Through Joins* (Mark Raasveldt, 2022) —
 caso moderno de pushdown sobre join trees vectorizados.
- CockroachDB optimizer notes (Andy Kimball, 2018) — pushdown a través
 de Cypher-shaped query trees.
- `docs/rfc/010-cost-based-optimizer.md` — fundación que esta RFC usa.
- `docs/rfc/008-logical-plan-ir.md` — operadores que esta RFC rewritea.

## Plan de implementación

1. **Crate `namidb-query`**:
 - `src/optimize/mod.rs` — re-exports + `optimize(plan, catalog)`.
 - `src/optimize/pushdown.rs` — `predicate_pushdown` + helpers
 (`expression_aliases`, `produced_aliases`, `split_and_terms`,
 `and_chain`, `apply_filters`). ~800 LoC + 20-25 unit tests.
 - `src/optimize/normalize.rs` — `normalize_filters` con las 4
 reglas. ~250 LoC + 8-10 unit tests.
 - `src/lib.rs` — re-export `plan(query, catalog)`.

2. **`src/parser/grammar.rs`**:
 - Reconocer `RAW` como soft keyword entre `EXPLAIN` y
 `VERBOSE`/query body.
 - `Query.explain_raw: bool`. Display round-trips.
 - ~25 LoC + 3 tests.

3. **`src/plan/explain.rs`**:
 - `explain_query_verbose(query, catalog)` aplica `optimize` antes de
 renderizar.
 - Nueva función `explain_query_raw(query)` para `EXPLAIN RAW`.
 - Helper `is_join_candidate` y annotación inline.
 - ~80 LoC + 5 tests.

4. **CLI** (`namidb-cli/src/main.rs`):
 - Flag `--raw`; pasar a través el flag del query string.
 - ~15 LoC.

5. **Executor wiring** (`src/exec/walker.rs`, `src/exec/writer.rs`):
 - Cualquier call site externo que llamaba `lower(&query)?` antes
 de `execute(...)` ahora llama `plan(&query, &catalog)?`. CLI y
 integration tests son los call sites principales.

6. **Tests integration** (`tests/cost_smoke.rs` + nuevo
 `tests/optimize_smoke.rs`):
 - Filter sobre source baja debajo de Expand (LDBC IC2 micro).
 - Filter sobre target NO baja debajo de Expand.
 - Filter sobre OPTIONAL target NO baja.
 - CrossProduct: predicates split a left / right / top.
 - `__label_eq` cleanup verifiable en EXPLAIN output.
 - Plan optimizado y plan crudo producen el mismo result set.
 - Plan optimizado tiene `estimate(...) ≤ estimate(crudo)`.

Snapshot esperado:
- `cargo test --workspace --exclude namidb-py`: 413 → ~445 passed.
- `cargo clippy --workspace --all-targets -- -D warnings`: clean.
- `cargo fmt --all -- --check`: clean.
- LoC nuevo: ~1 100 src + ~600 tests.
- Sin cambios en `namidb-storage`.
