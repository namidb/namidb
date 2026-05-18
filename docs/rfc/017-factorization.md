# RFC 017: Factorized intermediate results

**Status:** draft
**Author(s):** Matías Fonseca <info@namidb.com>
**Builds on:** RFC-008 (LogicalPlan IR), RFC-012 (HashJoin), RFC-015 (projection pushdown), RFC-016 (join reorder)
**Supersedes:** —

## Summary

El executor actual es Volcano-eager con un único tipo de
intermediate result, `Vec<Row>` donde `Row = BTreeMap<String,
RuntimeValue>` (`exec/row.rs:11`). Cada operator materializa
completamente su salida antes de pasarla al siguiente, y `Expand`
(`exec/walker.rs:484`) clona el `BTreeMap` por cada edge expandido
(`new_row.clone()` × 2 en walker.rs:544/554). Esto produce un
blow-up cartesian explícito en multi-hop patterns: para
`(p)-[:KNOWS]->(f)-[:KNOWS]->(fof)<-[:HAS_CREATOR]-(msg)` con
fan-out de ~10 en cada hop, el executor materializa ~1500 `Row` con
3-4 bindings cada uno antes del `LIMIT 20`.

Esta RFC introduce **factorized intermediate results**, una
representación en la que los outputs de Expand / HashJoin /
CrossProduct son cadenas de `FactorNode { parent: Option<NodeIdx>,
binding: Slot }` apoyadas sobre una `FactorArena`. Cada nuevo binding
agrega un único nodo al arena en vez de clonar el BTreeMap completo.
La materialización a `Vec<Row>` se difiere hasta el operador que la
requiere (TopN / Aggregate / Project final), y cuando se hace, solo se
aplastan las chains alcanzables por el `LIMIT` / projection.

Para IC09 esto reduce el footprint de O(fanout³) `Row`s a O(fanout³)
`FactorNode`s de ~16 bytes cada uno, y la materialización final cae a
O(LIMIT × profundidad) — exactamente la cota teórica de Olteanu (2015)
y la representación f-rep que Kùzu (CIDR 2023) usa internamente.

### Alcance v0

- **Pointer-based, arena-allocated factorization**. Trie-based (Olteanu)
 queda como referencia teórica; el shape concreto es un DAG de
 `FactorNode`s con índices `usize` al arena (ver Design §2).
- **Operators reescritos**: `Expand`, `CrossProduct`, `HashJoin`
 (build + probe en F-rep), `Filter`, `Project` intermedio.
- **Materialización en sinks**: `TopN`, `Aggregate`, `Distinct`,
 `Project` final (RETURN), `Unwind`, `Union`, `PatternList`. Sinks
 consumen `FactorRowSet` y emiten `Vec<Row>`.
- **Backwards compat por feature flag**: variable de entorno
 `NAMIDB_FACTORIZE=0` (default = on una vez estabilizado) restaura
 el path `Vec<Row>` para regresión semántica. La SemanticParity test
 suite compara outputs (row-set equality) entre ambos paths.
- **Row parity en `exec_ldbc_snb.rs`**: 100% mantenido. Los tests
 Cypher e2e validan equivalence, no internal representation.

### Out-of-scope v0

- **WCOJ (Worst-Case Optimal Joins)**. RFC-009 (en draft) introduce
 leapfrog triejoin para queries cíclicas. WCOJ se compone con
 factorization (ambos operan sobre f-rep), pero la implementación
 del operator queda diferida.
- **Operators columnar (Arrow-native)**. Mantenemos `RuntimeValue`
 por binding (un value individual). Arrow-vectorized batches quedan
 para una iteración futura (morsel-driven).
- **Spilling a disco**. La FactorArena vive en memoria. Si el dataset
 excede RAM, fallback es flat path con stream spill — fuera de v0.
- **DAG-level reuse** (CSE). Si dos branches del plan comparten un
 prefijo, no detectamos ni compartimos sub-arenas. Selinger ya elige
 un orden global; CSE-on-F-rep es follow-up.

## Motivation

**Bench (smoke gate scale=0.1) revela el costo del path actual:**

| Query | NamiDB p50 | Kùzu p50 | Ratio |
|---|---|---|---|
| IC02 (KNOWS + HAS_CREATOR) | 62 ms | 1.04 ms | **60×** |
| IC07 (HAS_CREATOR + LIKES) | 7 ms | 0.97 ms | 7× |
| IC08 (HAS_CREATOR + REPLY_OF) | 7 ms | 1.10 ms | 6× |
| IC09 (KNOWS·KNOWS + HAS_CREATOR) | **624 ms** | 1.64 ms | **382×** |

Row parity es 100% (compare.py confirma idéntico count y mismas
filas) — la divergencia es puramente de motor. Kùzu mantiene
factorized intermediate (Jin et al., CIDR 2023 §4.2) y emite plans que
defer la materialización al `LIMIT`.

**Plan IC09 actual:**

```
TopN(20, msg.creationDate DESC)
└─ Project [fof.firstName, fof.lastName, msg.content, msg.creationDate]
 └─ Expand HAS_CREATOR (msg ← post.has_creator)
 └─ Expand KNOWS (fof, hop 1..1)
 └─ Expand KNOWS (friend, hop 1..1)
 └─ NodeById Person p {id: $personId}
```

**Footprint en cada nivel (fanout ≈10 para KNOWS, ≈15 para HAS_CREATOR):**

| Operator | Rows | Bindings × Row | Bytes (BTreeMap + Node clones) |
|---|---|---|---|
| NodeById | 1 | 1 (p) | ~200 B |
| Expand friend | 10 | 2 (p, f) | ~4 KB |
| Expand fof | 100 | 3 (p, f, fof) | ~60 KB |
| Expand msg | 1500 | 4 (p, f, fof, msg) | ~1.2 MB |
| TopN(20) | 20 | 4 | (descarta 1480) |

La columna "Bytes" cuenta `Box<NodeValue>` + `BTreeMap` allocs +
`Arc<String>` shared del binding name. Los 1.2 MB en Expand msg son
~80% allocator + ~20% clone CPU. **`new_row.clone()` en walker.rs:544
se invoca 1500 veces en este path**, cada clone copiando 3 entries
previos del BTreeMap.

**Plan IC09 con factorization:**

```
TopN(20) ← materialize() aquí, solo 20 chains finales
└─ Project ← pass-through factorizado (no allocates rows)
 └─ ExpandF HAS_CREATOR → FactorArena nodes for {msg}
 └─ ExpandF KNOWS (fof) → FactorArena nodes for {fof}, parent=friend_node
 └─ ExpandF KNOWS → FactorArena nodes for {friend}, parent=p_node
 └─ NodeById → 1 FactorNode root with {p}
```

| Operator | FactorNodes | Bytes/node | Total |
|---|---|---|---|
| NodeById | 1 | 24 (parent + Slot) | 24 B |
| ExpandF friend | 10 | 24 | 240 B |
| ExpandF fof | 100 | 24 | 2.4 KB |
| ExpandF msg | 1500 | 24 | 36 KB |
| TopN(20) materialize() | 20 × 4 bindings = 80 BTreeMap entries | flat | ~6 KB |

**~36 KB vs ~1.2 MB = 33× menos memoria intermediate.** El CPU
ahorro es similar (no más BTreeMap clones; arena push es ~10 ns vs
clone ~500 ns).

## Design

### 1. Tipos de datos

Nuevo módulo `crates/namidb-query/src/exec/factor.rs`:

```rust
/// Index into FactorArena. usize to keep arena traversal cache-friendly.
pub type FactorIdx = u32;
pub const FACTOR_ROOT: FactorIdx = 0;

/// Single binding introduced by an operator: (name, value). Names are
/// `Arc<str>` so siblings share without allocating.
#[derive(Debug, Clone)]
pub struct Slot {
 pub name: Arc<str>,
 pub value: RuntimeValue,
}

/// One factorized output node. `parent` chains upward to inherited
/// bindings; `slot` is what THIS operator added. The root node
/// (FACTOR_ROOT) has parent=None and an empty Slot vec.
#[derive(Debug)]
pub struct FactorNode {
 pub parent: Option<FactorIdx>,
 /// Bindings added at this level. Usually 1 (Expand adds {target_alias},
 /// HashJoin adds the probe-side bindings) but can be N for CrossProduct
 /// or HashJoin output that emits multiple bindings at once.
 pub slots: SmallVec<[Slot; 2]>,
}

/// Arena of all factor nodes for one query execution. Grows monotonically;
/// no reuse, no GC. Dropped at end of execute().
#[derive(Debug, Default)]
pub struct FactorArena {
 nodes: Vec<FactorNode>,
}

impl FactorArena {
 pub fn new() -> Self {
 let mut a = Self::default();
 a.nodes.push(FactorNode { parent: None, slots: SmallVec::new() });
 debug_assert_eq!(a.nodes.len(), 1, "root is at FACTOR_ROOT");
 a
 }

 pub fn push(&mut self, parent: FactorIdx, slots: SmallVec<[Slot; 2]>) -> FactorIdx {
 let idx = self.nodes.len() as FactorIdx;
 self.nodes.push(FactorNode { parent: Some(parent), slots });
 idx
 }

 /// Walk parent chain and accumulate bindings into a flat Row. Used
 /// only at materialization points.
 pub fn materialize(&self, leaf: FactorIdx, projection: Option<&[&str]>) -> Row {
 let mut row = Row::new();
 let mut cur = Some(leaf);
 while let Some(idx) = cur {
 let node = &self.nodes[idx as usize];
 for slot in node.slots.iter().rev() {
 if let Some(p) = projection {
 if !p.iter().any(|w| **w == *slot.name) {
 continue;
 }
 }
 // First occurrence wins (shadowing — child overrides parent).
 row.bindings.entry(slot.name.to_string())
 .or_insert_with(|| slot.value.clone());
 }
 cur = node.parent;
 }
 row
 }
}

/// What each operator passes to its parent. Replaces `Vec<Row>` as
/// the intermediate type once factorization is enabled.
pub struct FactorRowSet {
 pub arena: Arc<RefCell<FactorArena>>,
 pub leaves: Vec<FactorIdx>,
}
```

**Decisión `usize` vs `u32`:** `u32` para mantener `FactorIdx` denso
(4 bytes vs 8). Cap 4G nodes per query — más que suficiente.

**Decisión `Arc<str>` para `Slot.name`:** Los binding names son de
~10 chars promedio y se repiten en CADA nivel del DAG. Inline string
costaría ~16 B/binding × millones de bindings = MBs desperdiciados.
`Arc<str>` shared = ~10 B/string + 8 B/Arc clone (ref count atomic).

**Decisión `SmallVec<[Slot; 2]>`:** La mayoría de Expand añaden 1
binding (target). HashJoin añade los probe-side bindings (3-5
típicos). `SmallVec` inline 2 evita el alloc del 80% de casos sin
heap-allocar para los menos.

### 2. Operators reescritos

#### 2.1 `execute_expand` (walker.rs:484)

**Antes:**

```rust
async fn execute_expand(rows: Vec<Row>, ...) -> Result<Vec<Row>> {
 let mut out = Vec::new();
 for row in rows {
 let mut frontier = vec![Step { tail, row: row.clone() }];
 for hop in 1..=max {
 for step in frontier.drain(..) {
 for edge in neighbours {
 let mut new_row = step.row.clone(); // ← clone #1
 new_row.set(target_alias, value);
 next_frontier.push(Step { row: new_row.clone() }); // ← clone #2
 if hop >= min { out.push(new_row); }
 }
 }
 }
 }
 Ok(out)
}
```

**Después:**

```rust
async fn execute_expand_factor(
 input: FactorRowSet,
 target_alias: Arc<str>,
 rel_alias: Option<Arc<str>>,
 ...
) -> Result<FactorRowSet> {
 let arena = input.arena.clone();
 let mut out_leaves = Vec::new();
 for leaf in input.leaves {
 // Find tail node id by walking up to the binding `source`.
 let tail = arena.borrow().lookup_binding(leaf, source)?;
 let mut frontier = vec![(leaf, tail)];
 for hop in 1..=max {
 let mut next_frontier = Vec::new();
 for (parent_idx, tail_id) in frontier.drain(..) {
 for edge in neighbours_of(snapshot, edge_type, dir, tail_id).await? {
 let target_id = partner_id(&edge, dir, tail_id);
 let target_view = lookup(...).await?;
 let mut slots = SmallVec::new();
 if let Some(name) = &rel_alias {
 slots.push(Slot { name: name.clone(), value: RuntimeValue::Rel(...) });
 }
 slots.push(Slot {
 name: target_alias.clone(),
 value: RuntimeValue::Node(Box::new(NodeValue::from(target_view))),
 });
 let new_idx = arena.borrow_mut().push(parent_idx, slots);
 next_frontier.push((new_idx, target_id));
 if hop >= min {
 out_leaves.push(new_idx);
 }
 }
 }
 frontier = next_frontier;
 }
 }
 Ok(FactorRowSet { arena, leaves: out_leaves })
}
```

**Clave:** ninguna clonación de Row. El `parent_idx` ya inherita
todos los bindings ancestrales; solo se push un `FactorNode` con el
nuevo binding.

#### 2.2 `cross_product` (walker.rs:693)

**Antes:**

```rust
fn cross_product(left: Vec<Row>, right: Vec<Row>) -> Vec<Row> {
 let mut out = Vec::with_capacity(left.len() * right.len());
 for l in &left {
 for r in &right {
 let mut merged = l.clone(); // ← clone left
 for (k, v) in &r.bindings { merged.set(...); } // ← copy entries
 out.push(merged);
 }
 }
 out
}
```

**Después:**

```rust
fn cross_product_factor(left: FactorRowSet, right: FactorRowSet) -> FactorRowSet {
 // Splice right's chains onto left's leaves. The arena must be merged
 // (offset right's indices). For v0 we copy right's nodes into left's
 // arena (O(|right.nodes|), one-time).
 let arena = left.arena;
 let right_offset = arena.borrow().nodes.len() as FactorIdx;
 arena.borrow_mut().splice_from(&right.arena.borrow());
 let mut out_leaves = Vec::with_capacity(left.leaves.len() * right.leaves.len());
 for &l in &left.leaves {
 for &r in &right.leaves {
 // Reparent right's chain from FACTOR_ROOT to l.
 let r_offset = r + right_offset;
 let bridge = arena.borrow_mut().splice_under(l, r_offset);
 out_leaves.push(bridge);
 }
 }
 FactorRowSet { arena, leaves: out_leaves }
}
```

**`splice_under(parent, foreign_idx)`** reroutea la cadena del nodo
foreign para que su root apunte al `parent`. Es O(altura(foreign_idx))
worst case, pero típico altura ≤ 5 en LDBC.

**Trade-off:** v0 hace `splice_from` (copia los nodos del right en
el left). Alternative: dos arenas separadas + `MergedArenaView` que
los presenta como uno solo. Más eficiente para outputs grandes pero
complica la API de `materialize`. Defer.

#### 2.3 `HashJoin` (walker.rs::execute_hash_join)

**Build side** (la rama "build" de un HashJoin): materializa a
`Vec<Row>` ahora porque necesita ser indexable por las claves.
Mantenemos eso. La build side ya se aplasta — esa parte no cambia.

**Probe side:** se mantiene como `FactorRowSet`. Para cada
`probe.leaf`:

1. Look up `probe.lookup_binding(leaf, probe_key) → val`.
2. Hash table lookup → `Vec<&BuildRow>` (build side rows que matchean).
3. Por cada `BuildRow`, push un `FactorNode` con los bindings de
 build como slots, parent=`probe.leaf`. → un nuevo leaf en arena.

Output es `FactorRowSet` cuyas leaves son los productos
probe×build.

**No reorder semantics**: HashSemiJoin sigue sin swap (RFC-016).

#### 2.4 Sinks (materialization)

`TopN`, `Aggregate`, `Distinct`, `Project` final, `Union`,
`PatternList`, `Unwind` consumen `FactorRowSet` y emiten `Vec<Row>`:

```rust
fn materialize_for_topn(set: FactorRowSet, n: usize, order_key: &str)
 -> Vec<Row>
{
 // 1. Top-N by order_key value WITHOUT materializing — we only need
 // arena.lookup_binding(leaf, order_key) for the heap key.
 let mut heap = BinaryHeap::with_capacity(n + 1);
 for leaf in &set.leaves {
 let key = set.arena.borrow().lookup_binding(*leaf, order_key)?;
 heap.push((Reverse(key), *leaf));
 if heap.len() > n { heap.pop(); }
 }
 // 2. Materialize only the N survivors.
 heap.into_iter()
 .map(|(_, leaf)| set.arena.borrow().materialize(leaf, None))
 .collect()
}
```

Para `Project` final (RETURN columns): materialize con projection
`&[col_names]` para evitar copiar bindings que no se devuelven.
Combina con RFC-015 (projection pushdown ya emite las columnas que
necesita el RETURN).

### 3. Wiring en el optimizer y executor

#### 3.1 Sin cambios en LogicalPlan

`LogicalPlan` se mantiene igual (RFC-008). Factorization es un detalle
del executor — el plan sigue siendo `Expand`, `HashJoin`, etc.

#### 3.2 `execute()` toma una decisión arriba

`execute(plan, snapshot, params)` decide entre dos paths:

```rust
pub async fn execute(plan: &LogicalPlan, snapshot: &Snapshot, params: &Params)
 -> Result<Vec<Row>, ExecError>
{
 if factorize_enabled() {
 let set = execute_factor(plan, snapshot, params).await?;
 Ok(materialize_top(set)) // root materialization
 } else {
 execute_flat(plan, snapshot, params).await
 }
}
```

`factorize_enabled()` lee `NAMIDB_FACTORIZE` (default `1` una vez
estabilizado, `0` durante el desarrollo).

`execute_factor` y `execute_flat` son funciones paralelas. `execute_flat`
es el path actual (renombrado). `execute_factor` es el nuevo path.

**No share parcial:** intentamos mantenerlos como dos paths
independientes para evitar regresiones. Cuando el path factorizado se
estabilice, deprecate `execute_flat` con un `#[deprecated]` y
remove en una iteración posterior (no v0).

#### 3.3 Write operators

`CREATE`, `MERGE`, `SET`, `REMOVE`, `DELETE` consumen el output de
read clauses. v0: materializan F-rep al input de cada write — los
writes ya son row-oriented y la cadena no se beneficia.

### 4. Tests

#### 4.1 Unit tests

`exec/factor.rs::tests`:

- `arena_root_is_empty` — `materialize(FACTOR_ROOT)` returns empty Row.
- `single_push_then_materialize` — push 1 slot, materialize == single binding.
- `chain_inherits_parent` — push A then B, materialize(B) has both A and B.
- `materialize_with_projection` — projection filter hides slots.
- `child_shadows_parent` — same name, child value wins.
- `splice_under_reparent` — splice respects topology.

#### 4.2 Operator parity tests

Cada operator que toca factorization tiene un test que ejecuta el
MISMO plan con `NAMIDB_FACTORIZE=0` y `=1` y compara outputs por
`HashSet<Row>` equality (orden no garantizado en ambos):

```rust
#[tokio::test]
async fn expand_factor_matches_flat() {
 let (flat, fact) = run_both_paths(plan, snapshot, params).await;
 assert_eq!(row_set(&flat), row_set(&fact), "Expand parity failed");
}
```

`row_set(rows) -> BTreeSet<Row>` para ignorar orden, mantener
multiplicidad.

#### 4.3 Integration tests

`crates/namidb-query/tests/exec_ldbc_snb.rs` se ejecuta dos
veces (build matrix con feature flag) — todos los tests existentes
deben pasar en ambos paths.

#### 4.4 Bench

Re-correr el harness gate (`bench/README.md`). Comparar ratios
pre- y post- factorization. Threshold de éxito v0:

- IC09: < 50× Kùzu (era 382×). 8× mejora absoluta.
- IC02: < 10× Kùzu (era 60×).
- IC07/IC08: < 5× Kùzu (eran 6-7×).

Si IC09 < 2× (gate smoke), avance a SF1 real LDBC. Si
no, evaluar morsel-driven execution y/o WCOJ como siguientes.

### 5. Plan de implementación

| Fase | Entregable |
|---|---|
| Diseño | Este documento |
| Tipos base | `factor.rs` + 6 unit tests |
| Expand | `execute_expand_factor` + parity test |
| Joins | `cross_product_factor`, `hash_join_factor` |
| Sinks | Sinks + workspace integration tests verdes |
| Validación | Re-bench gate |

El alcance amplio justifica un RFC explícito antes de tocar walker.rs.

## Alternatives considered

### A1. Trie-based factorization (Olteanu 2015)

F-trie nodes con shape `{level: usize, children: HashMap<RuntimeValue,
FtrieNode>}`. Más cerca del paper, expresividad superior para WCOJ.

**Rechazado v0** porque (a) requiere hash-keyed children → cuesta
HashMap allocs por nivel; (b) la traversal pattern de NamiDB
(walker.rs) es naturalmente pointer-up (cada step inherita parent),
no key-down. El trade-off de Olteanu (memoria mínima asintóticamente)
no compensa en datasets < 1B nodes donde RAM no es el bound.

### A2. Columnar vector batches (Arrow-native, à la DuckDB)

Pasa `RecordBatch` entre operators, no `Vec<Row>`. Combina factorization
+ vectorization en un solo paso.

**Rechazado v0** porque (a) requiere reescribir TODO el executor para
trabajar en Arrow batches en vez de RuntimeValue por binding; (b) la
ruta morsel-driven ya va por ese camino. Pre-condition para Arrow
batches es resolver el factorization shape primero — si los outputs
intermedios son flat tuples Cartesian-blown-up, los batches no
ayudan. Hacemos factorization primero, vectorization después.

### A3. Just batch `Vec<Row>` reuse + clone-on-write

Reemplazar `Row { bindings: BTreeMap }` con `Row { bindings: Arc<BTreeMap> }`
y mutaciones via `Arc::make_mut`. Reduce clone cost pero no elimina
cartesian materialization en operators.

**Rechazado v0** porque ataca el síntoma (clone cost) no la causa
(N×M rows allocated). Para IC09 el problema son las 1500 filas, no
el cost de clonar cada una. Arc-on-Row ayudaría ~3-5× pero no los
~382× requeridos.

### A4. Sin feature flag, full migration directa

Reemplazar `Vec<Row>` con `FactorRowSet` en todos los operators
de una vez.

**Rechazado v0** porque (a) imposibilita el SemanticParity test
suite (no hay path de referencia); (b) bug fix workflow más
peligroso; (c) revert difficult si IC*'s row counts difieren tras
materialize() en algún edge case.

## Drawbacks

1. **Complejidad del executor.** Dos paths paralelos
 (`execute_flat` / `execute_factor`) durante el período de
 feature flag. Mitigado con strict parity tests.

2. **Materialize a flat puede regresar performance en queries que
 YA son flat-friendly.** Ej: `MATCH (a) RETURN a` con 1M nodes,
 F-rep allocates 1M `FactorNode` y luego aplasta a 1M `Row` —
 peor que el path actual.

 Mitigación: el sink reconoce "single-binding sets" y short-circuits
 . Si el set tiene un solo Slot por leaf, no allocate F-rep
 intermedio.

3. **RefCell en FactorArena.** Sharing entre operators implica
 interior mutability. Tokio + RefCell requiere `!Send` discipline
 o `Mutex`. v0 elige `RefCell` (single-threaded executor); cuando
 aterrice morsel parallelism, swap a `Arc<RwLock<FactorArena>>`
 o partitioned arenas.

4. **`lookup_binding(leaf, name)`** es O(depth). Para depth ≤ 5
 (LDBC pattern típico) eso es ~100 ns por lookup. Si una query
 necesita el mismo binding muchas veces, mejor cachear en el operator
 o expandir el slot al arena root.

5. **Memory profile diferente.** Spike acumulativo en el arena hasta
 el sink, vs spike continuo en flat. Para queries largas (no LIMIT)
 el arena puede crecer mucho. Mitigación follow-up: stream sinks
 que drenan el arena progresivamente.

## Open questions

- **¿`Arc<RefCell<FactorArena>>` o pass-by-value?** v0 propone
 `Arc<RefCell<…>>` para que `cross_product` pueda compartir el arena.
 Alternativa: cada operator construye su propio arena y `splice_from`
 los del input. Más allocations, menos contention. Decidir durante
 implementación cuando se vea el patrón real.

- **¿Cómo manejar `OPTIONAL MATCH`?** Cuando un Expand opcional no
 encuentra neighbours, hoy emite el row sin el binding (NULL semantics).
 En F-rep: push un FactorNode con un Slot `{name, RuntimeValue::Null}`?
 ¿O dejar al sink que detecte "missing binding" → null? Decidir
 durante implementación.

- **`Distinct` post-F-rep.** Hashing `FactorIdx` no funciona —
 Distinct compara por valor, no por identidad arena. Materialize-then-
 distinct o introducir un hash sobre la materialización del row?
 Probable: materialize-first para v0.

- **Threshold para feature flag default.** ¿Encender cuando todos los
 parity tests pasen o esperar a bench results? Propuesta: encender
 con flag override disponible y flip default tras bench validation.

## References

- Olteanu, Závodný (2015) — **Size Bounds for Factorised Representations
 of Query Results.** ACM TODS 40(1).
- Jin, Mhedhbi, Lu, Sequoda (2023) — **Kùzu Graph Database Management
 System.** CIDR 2023. §4.2 describes pointer-based factorization.
- Bakibayev, Olteanu (2012) — **FDB: A Query Engine for Factorised
 Relational Databases.** PVLDB 5(11).
- Aberger et al. (2017) — **EmptyHeaded: A Relational Engine for Graph
 Processing.** SIGMOD. §3.1 motivates factorization in graph context.
- Leis et al. (2014) — **Morsel-Driven Parallelism: A NUMA-Aware Query
 Evaluation Framework for the Many-Core Age.** SIGMOD. Composing
 factorization with morsel-driven is a future follow-up.
- `crates/namidb-query/src/exec/walker.rs:484` — current
 `execute_expand` blow-up point.
- `crates/namidb-query/src/exec/walker.rs:693` — current
 `cross_product` blow-up point.
- `crates/namidb-query/src/exec/row.rs:11` — current `Row` type.
- RFC-008 (LogicalPlan IR), RFC-012 (HashJoin), RFC-015 (projection
 pushdown), RFC-016 (join reorder) — operators y plan shape que
 esta RFC reescribe a F-rep.
