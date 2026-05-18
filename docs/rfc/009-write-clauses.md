# RFC 009: Write clauses + execution model

**Status:** accepted
**Author(s):** Matías Fonseca <info@namidb.com>
**Supersedes:** —

## Summary

El read-path está completo: `parse → lower → execute` contra
`Snapshot` (read-only) corre MATCH / Expand / Filter / Project /
TopN / Aggregate / Distinct / Union / Unwind / SemiApply / PatternList /
list & pattern comprehensions / `RETURN *` / `EXPLAIN`, y los 4 IC
representativos del LDBC SNB Interactive (IC2/7/8/9) producen resultados
correctos sobre un mini-graph.

Este RFC extiende el query engine al **write-path**: las cláusulas
`CREATE`, `MERGE`, `SET`, `REMOVE`, `DELETE` (y `DETACH DELETE`)
parsean y producen AST válido, pero el lowering reporta
`UnsupportedFeature`. Esta RFC cierra ese gap.

## Motivation

Sin write-path, el subset Cypher de namidb no es completo: cualquier
usuario que quiera cargar datos debe hacerlo via la API Rust
`WriterSession::upsert_node/upsert_edge` directamente. Eso rompe el pitch
"developer-first universal con embed + Cypher" y bloquea de plano:

- LDBC SNB **Update queries** (IU1 insertPerson, IU2 addPostLike,
 IU3 addCommentLike, IU4 addForum, IU5 addForumMembership,
 IU6 addPost, IU7 addComment, IU8 addFriendship). Sin estas no hay
 pipeline LDBC end-to-end (load → run → measure).
- Quickstart docs ("crea un nodo, agrega una arista, lee de vuelta") —
 hoy requieren un `Cargo.toml` + `tokio::main` boilerplate.
- Loading desde scripts Cypher (.cypher files con CREATE chains que
 Neo4j / Kuzu aceptan de fábrica).

Costo de no hacerlo ahora: cada query LDBC IU permanece como dead-letter;
los benchmarks de siguen necesitando harnesses ad-hoc que
escriben via API Rust en vez de via la abstracción Cypher idiomática;
ningún consumidor externo puede probar namidb sin escribir Rust.

## Design

### Operadores nuevos en `LogicalPlan`

```rust
pub enum LogicalPlan {
 // ... read operators ...

 Create {
 input: Box<LogicalPlan>,
 elements: Vec<CreateElement>,
 },

 Merge {
 input: Box<LogicalPlan>,
 pattern: CreateElement,
 on_match_sets: Vec<SetOp>,
 on_create_sets: Vec<SetOp>,
 },

 Set {
 input: Box<LogicalPlan>,
 items: Vec<SetOp>,
 },

 Remove {
 input: Box<LogicalPlan>,
 items: Vec<RemoveOp>,
 },

 Delete {
 input: Box<LogicalPlan>,
 targets: Vec<Expression>,
 detach: bool,
 },
}
```

Helpers:

```rust
pub enum CreateElement {
 Node {
 alias: String,
 label: String,
 properties: Vec<(String, Expression)>,
 },
 Rel {
 alias: Option<String>,
 edge_type: String,
 source_alias: String,
 target_alias: String,
 direction: RelationshipDirection,
 properties: Vec<(String, Expression)>,
 },
}

pub enum SetOp {
 Property { target_alias: String, key: String, value: Expression },
 Replace { target_alias: String, value: Expression }, // a = {...}
 Merge { target_alias: String, value: Expression }, // a += {...}
 Labels { target_alias: String, labels: Vec<String> }, // a:Label[:Label]
}

pub enum RemoveOp {
 Property { target_alias: String, key: String },
 Labels { target_alias: String, labels: Vec<String> },
}
```

`children()` retorna `[input]` para los 5 nuevos. `operator_name()`
retorna `"Create"`, `"Merge"`, `"Set"`, `"Remove"`, `"Delete"`
(prefijado por `Detach` cuando aplica).

### Lowering rules

- **CREATE** clause sin MATCH previo: `Empty → Create`. Cuando hay MATCH
 previo, `Create` se encadena: `... → Create { input, elements }`. Las
 bindings nuevas (node aliases + rel aliases) se introducen en
 `LowerCtx` antes del próximo clause.
- **MERGE** clause: solo una pattern part en v0. Se baja a
 `Merge { input, pattern, on_match_sets, on_create_sets }`. Las
 bindings del pattern se introducen en `LowerCtx`.
- **SET**: cada item se traduce a un `SetOp`; el operador `Set` lee el
 binding del row y muta.
- **REMOVE**: similar a SET; cada `RemoveOp` se aplica.
- **DELETE / DETACH DELETE**: las expressions de `targets` se evalúan
 per-row para producir Node/Rel/Path; el operador lo tombstones.
- Una query solo-write (sin MATCH) arranca con `LogicalPlan::Empty` para
 proveer una "single driver row". Esto reusa el patrón ya usado por
 UNWIND.

Bindings de salida: al final del query, las bindings del último write
clause + las del último read clause permanecen visibles si hay un RETURN
posterior (Cypher 25 permite `CREATE (a:Person {name: 'Ada'}) RETURN a`).

### Executor split: read vs write

Mantengo dos entry points distintos:

```rust
// Read-only path
pub async fn execute(
 plan: &LogicalPlan,
 snapshot: &Snapshot<'_>,
 params: &Params,
) -> Result<Vec<Row>, ExecError>;

// Write-aware path
pub async fn execute_write(
 plan: &LogicalPlan,
 writer: &mut WriterSession,
 params: &Params,
) -> Result<WriteOutcome, ExecError>;

pub struct WriteOutcome {
 pub rows: Vec<Row>,
 pub nodes_created: u64,
 pub edges_created: u64,
 pub nodes_deleted: u64,
 pub edges_deleted: u64,
 pub properties_set: u64,
}
```

`execute_write`:

1. Walk down the plan. Read operators (NodeScan/Expand/Filter/...) usan
 `writer.snapshot()` interno (re-pinned por clause).
2. Write operators (Create/Merge/Set/Remove/Delete) llaman
 `writer.upsert_node/upsert_edge/tombstone_node/tombstone_edge` per-row.
3. Al final, **auto-commit**: `writer.commit_batch().await` antes de
 retornar `WriteOutcome`. Garantiza durabilidad de toda la query como
 unidad.

`execute_write` queda separado de `execute` por dos razones:

- Type safety — `&mut WriterSession` vs `&Snapshot<'_>` no son
 intercambiables.
- Permite que `execute` se siga ejecutando contra snapshots persistidos
 (read-replicas) en SaaS sin acoplar el writer side.

### Read-your-own-writes: NO en v0

Una query como:

```cypher
CREATE (a:Person {name: 'Ada'})
MATCH (p:Person) RETURN p.name
```

verá rows = whatever existía pre-CREATE. La nueva Ada **no**
está visible al MATCH. Razón:

- Implementar visibility intra-query require overlay sobre Snapshot
 (memtable+SST+pending_payloads). El WriterSession actual ya tiene
 `pending_payloads` pero solo se aplican al memtable post-`commit_batch`.
- La complejidad de read-your-own-writes choca con la semántica de
 cluster-distributed eventual consistency que querremos en SaaS.
- La gran mayoría de queries write-then-read son separadas por commits
 (sesiones interactivas). LDBC IU queries son monolíticas pero
 write-only.

Mitigación: una vez se introduzca transactional consistency real,
overlay la memtable + pending → read-your-own-writes "just works".
Hasta entonces, error explícito si detectamos write-then-read en el
mismo plan tree (advisor warning, no hard fail).

### MERGE semantics

```
MERGE (n:Label {key: value})
 ON MATCH SET n.lastSeen = $now
 ON CREATE SET n.firstSeen = $now, n.lastSeen = $now
```

Ejecución:

1. Intenta matchear el pattern (igual que MATCH). Si encuentra ≥1 row:
 - Para cada row matched, aplica `on_match_sets`.
 - Output rows reflejan los matches.
2. Si encuentra 0 rows:
 - Genera el pattern (igual que CREATE).
 - Aplica `on_create_sets` al row del CREATE.
 - Output rows reflejan la creación.

Limitaciones v0:
- Solo una pattern part por MERGE (no multi-element). RFC-004 ya
 rechazaba multi-label en parser.
- No locks/serializability. Una MERGE concurrente con otra writer puede
 crear duplicados — esto queda para una RFC futura.

### DETACH DELETE semantics

```
MATCH (a:Person {id: $id}) DETACH DELETE a
```

Para cada `a` matched, antes de tombstone el node, enumera todas las
edges incidentes vía `out_edges(*, a.id) + in_edges(*, a.id)` para CADA
edge_type declarado en el manifest schema, y las tombstones primero.
Luego tombstone el node.

DELETE sin DETACH falla con `ExecError::Mutation` si el node tiene
edges (mensaje explícito sugiriendo DETACH).

### Path binding (caso simple)

```rust
pub enum RuntimeValue {
 // ...
 Path(Vec<RuntimeValue>), // alternating Node, Rel, Node, Rel, ..., Node
}
```

Para `MATCH p = (a)-[r]->(b) RETURN p`:

- `PatternPart.binding = Some(p)` se baja a `Expand { ..., path_binding:
 Some("p") }`.
- El executor, al producir cada row, materializa `[a_value, r_value,
 b_value]` y bindea a `p`.
- Para chains más largos `p = (a)-[r1]->(b)-[r2]->(c)`, el executor
 acumula a través del Expand chain.

Variable-length paths (`p = (a)-[*1..3]->(b)`) requieren materializar
listas de longitud variable y quedan diferidos.

`fingerprint_value` se extiende con un caso `Path(items)` para que
Distinct + collect distinct funcionen sobre paths.

## Alternatives considered

**A. Single executor entry que toma `&mut WriterSession` siempre.**
Rechazada: Snapshot read path es claramente diferente del write path
(no mutación, lifetime más corto, posible read-only replica). Forzar
WriterSession en TODOS los reads acopla los SaaS paths.

**B. Lazy commit (caller decides cuándo flush).** Rechazada para v0:
hace que `execute_write` retorne un handle a un "pending transaction"
y requiere transaction API formal. La sentencia "una query es una
transacción" es predecible y suficiente para LDBC IU + quickstart.

**C. Read-your-own-writes via overlay.** Considerada pero deferida: el
overlay sobre Snapshot requiere mantener un view temporal "memtable +
pending_payloads + el plan write effects acumulados hasta ahora". Es
~300 LoC y complica el reasoning sobre snapshot lifetimes. Vuelve a
futuro con el transactional model.

**D. MERGE con locks.** Considerada y rechazada para v0: requiere
coordinación a nivel WriterSession (single-writer per namespace ya
nos da serialización a nivel de namespace, pero MERGE necesita
serialization local entre clauses). Vive bien con LWW pero introduce
flakiness en tests si dos writers race. Mientras tenga
single-writer-per-namespace (que tiene), MERGE es safe.

**E. Mantener Create/Merge/Set/Remove/Delete como UnsupportedFeature.**
Rechazada: bloquea LDBC IU y quickstart developer experience
indefinidamente. El opportunity-cost de no tenerlos es mayor que la
complejidad de implementarlos ahora.

**F. Soportar variable-length path bindings de entrada.** Rechazada:
materializar lista de longitud variable + interaccionar con `Expand`
multi-hop es ~150 LoC más y un test surface considerable. El caso
simple cubre la mayoría de quickstart docs; var-len queda diferido.

## Drawbacks

1. **No read-your-own-writes** rompe expectativas de usuarios que
 vienen de Neo4j / Kuzu. Mitigación: documentar explícitamente en
 README + retornar warning en `WriteOutcome` si se detectó el
 pattern; cerrar a futuro.

2. **Auto-commit per query** no permite multi-statement transactions.
 Para LDBC IU es suficiente (cada IU es atomic by design); para
 workloads ETL más complejos no. Mitigación: a futuro se introduce
 explicit `BEGIN TRANSACTION ... COMMIT` clauses con session API.

3. **MERGE sin locks** depende del single-writer-per-namespace
 invariant. Si en multi-tenant SaaS hacemos multi-writer
 sharded namespaces, MERGE necesita revisitarse. Documentado.

4. **DETACH DELETE enumeration is O(edge_types × incident_edges).** Para
 nodos high-degree (super-nodes) puede ser caro. Acceptable para
 v0; optimización vive junto con el catálogo de edge_types
 activos.

5. **`WriteOutcome` counters son aproximados.** Counters incrementan
 por cada operación del executor, no por cada cambio real de estado
 (e.g. SET de la misma propiedad al mismo valor cuenta como 1
 property_set aunque sea no-op). Documentado.

## Open questions

- **Q1: WriteOutcome.rows.** ¿Una query write-only (CREATE sin RETURN)
 retorna `Vec<Row>` vacío? Cypher dice sí. ¿Y con RETURN?
 `RETURN a` después de CREATE retorna el row con `a` bound. Implementar
 igual que un Project encima del Create.

- **Q2: Schema discovery via CREATE.** Si CREATE introduce una label
 o edge_type nueva, ¿se autopopula el schema en el manifest? RFC-002
 permite schema implícita via property names. Sí — el executor
 introspecciona la label + edge_type y los agrega si no existen.
 Requiere que `WriterSession` exponga un schema extension API; hoy
 el commit_batch no toca schema. Pieza adicional.

- **Q3: Multi-statement Cypher.** `CREATE (a) ; CREATE (b)` (con
 semicolon). Hoy parser lo acepta como query terminator pero no
 como separator entre statements. ¿Statement separator es necesario
 para Cypher scripts? Diferido.

## References

- openCypher 9 §6 (Write clauses), §7 (Reading + writing clauses).
- GQL ISO/IEC 39075:2024 §19 (Linear data modifications).
- Neo4j MERGE semantics: https://neo4j.com/docs/cypher-manual/current/clauses/merge/
- Kuzu storage write path: kuzudb/kuzu README §"Bulk loading + transactions".
- DuckDB inserts as plans: https://duckdb.org/docs/sql/statements/insert.html
- RFC-008 (Logical Plan IR + addendum).
- RFC-002 (SST format) — schema introspection at storage layer.
