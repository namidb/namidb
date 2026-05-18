# RFC 004: Cypher subset compatibility scope

**Status:** draft
**Author(s):** Matías Fonseca <info@namidb.com>
**Supersedes:** —

## Summary

Declara el subconjunto exacto de Cypher 25 / openCypher / GQL ISO/IEC 39075:2024
que el parser `namidb-query` acepta en la primera iteración del query engine
(v0). La meta es **parsear sin error las 12 queries de LDBC SNB Interactive
Complex que no dependen de `shortestPath`/`allShortestPaths`**, dejando IC13 e
IC14 explícitamente fuera de scope hasta RFC-009 (WCOJ + recursive patterns) y
cerrando 100 % de la superficie cubierta con tests que viven con el código.

El subset es **deliberadamente menor** que el de Neo4j Community 5.x y que el de
Kùzu: privilegiamos compatibilidad estricta sobre features, evitamos
APOC, evitamos subqueries `CALL` y evitamos `FOREACH`. El compromiso es:
*"lo que el parser acepta corre o devuelve un error tipado claro — nunca un
warning silencioso que cambia la semántica"*.

## Motivation

Cypher es un lenguaje grande. El Cypher 25 specification (mayo 2025) define
unas 80 cláusulas/expresiones de primer nivel, openCypher TCK suma ~10 000
casos de test. Una implementación completa toma 18+ meses (Memgraph tardó ~2
años en cubrir el 80 % útil; Kuzu nunca llegó al 100 % antes del archive).

Sin un scope explícito el parser se vuelve un agujero negro de tiempo:
- Cada feature nuevo demanda decisiones de semántica (e.g. `MERGE` con
 multi-label, `OPTIONAL MATCH` con left-anti-join, `WITH *`).
- Cada feature nuevo demanda tests, error messages, lowering al IR del
 logical plan.
- Sin un gate de "qué está adentro y qué afuera" no podemos honestamente
 comunicar al usuario qué funciona.

**Referencia de scope:** LDBC SNB Interactive Complex Q1–Q14. Cubrir 12 de
las 14 queries en el parser deja scope ejecutable para las etapas siguientes
(lowering, optimizer, executor) sin abrir nuevos frentes de compatibilidad.

## Design

### Versión declarada del estándar

- **Base normativa:** GQL ISO/IEC 39075:2024 (publicado 11 abril 2024) +
 openCypher 9 (el último cuya specification es libre de patentes).
- **Cypher 25 (Neo4j):** trataremos como referencia de naming y syntax pero
 **no** implementaremos nada exclusivo de Neo4j (e.g. `db.*` functions, APOC).
- **Cuando hay conflicto** entre GQL y openCypher, **GQL wins**. Razón:
 evitar lock-in vendor-specific, posicionarnos junto a la dirección que
 Memgraph, RisingWave y la comunidad académica están tomando.

### Subconjunto v0 (in-scope)

#### Clauses

| Clause | v0 | Notas |
|---|---|---|
| `MATCH` | ✅ | Patrón fijo o variable-length `*n..m` con bounds finitos. |
| `OPTIONAL MATCH` | ✅ | Semantics left-outer-join. |
| `WHERE` | ✅ | Predicados arbitrarios sobre el scope visible. |
| `RETURN` | ✅ | Projection list con aliases (`AS`). `DISTINCT` soportado. `*` no soportado en v0 (se exige projection explícita). |
| `WITH` | ✅ | Pipe que reinicia el scope. Soporta `WHERE` interior y aliases. |
| `ORDER BY` | ✅ | Multi-key `ASC`/`DESC`. |
| `SKIP` / `LIMIT` | ✅ | Solo literales o `$param`. Sin expresiones. |
| `UNWIND` | ✅ | Lista → rows. |
| `CREATE` | ✅ | Nodes y edges con properties literales o `$param`. |
| `MERGE` | ✅ | `MERGE... ON CREATE SET... ON MATCH SET...`. |
| `SET` | ✅ | Property assign, label add. |
| `DELETE` / `DETACH DELETE` | ✅ | Single binding por delete. |
| `REMOVE` | ✅ | Property remove, label remove. |
| `UNION` / `UNION ALL` | ✅ | Mismo arity y mismos aliases. |

#### Patterns

| Element | v0 | Notas |
|---|---|---|
| Node pattern `(a:Label {prop: val})` | ✅ | Multi-label `(a:A:B)`. Map property filter inline. |
| Relationship pattern `-[r:TYPE]->` | ✅ | Direction `-->`, `<--`, `--`. |
| Relationship type alternation `-[r:TYPE_A\|TYPE_B]->` | ✅ | |
| Variable-length `-[r:KNOWS*1..3]->` | ✅ | Bounds finitos requeridos. `*` solo o `*n..` (sin upper bound) → error explícito. |
| Pattern chain `(a)-[]-(b)-[]-(c)` | ✅ | |
| Pattern de múltiples partes `MATCH (a), (b)` | ✅ | |
| Anonymous variable ``, `[]` | ✅ | |

#### Expressions

| Categoría | v0 |
|---|---|
| Literals: int, float, string, bool, null, list `[1,2,3]`, map `{k: v}` | ✅ |
| Parameters `$name` | ✅ |
| Variable reference `a`, property access `a.prop` | ✅ |
| Operators arith `+ - * / % ^` | ✅ |
| Operators string `+` (concat), `=~` (regex) | ✅ |
| Operators bool `AND OR NOT XOR` | ✅ |
| Comparison `= <> < > <= >=` | ✅ |
| `IS NULL` / `IS NOT NULL` | ✅ |
| `IN` (membership lista) | ✅ |
| `STARTS WITH`, `ENDS WITH`, `CONTAINS` | ✅ |
| Function call `length(x)`, `count(a)`, `collect(a.prop)` | ✅ (built-ins listados abajo) |
| `CASE WHEN... THEN... ELSE END` | ✅ (forma simple y forma multi-branch) |
| List comprehension `[x IN list WHERE pred \| expr]` | ✅ |
| Pattern comprehension `[(a)-[]->(b) \| b.name]` | ✅ |
| Pattern predicates `WHERE (a)-[]->(b)` | ✅ |

#### Built-in functions (mínimas para Q1–Q12)

**Aggregations:** `count(*)`, `count(x)`, `count(DISTINCT x)`, `sum`, `avg`,
`min`, `max`, `collect`, `collect(DISTINCT x)`.

**Scalar:** `id(n)`, `labels(n)`, `type(r)`, `keys(n)`, `properties(n)`,
`length(p)`, `size(coll)`, `head(coll)`, `last(coll)`, `tail(coll)`,
`coalesce(x, y,...)`.

**String:** `toLower`, `toUpper`, `trim`, `substring`, `replace`, `split`,
`toString`, `toInteger`, `toFloat`.

**Numeric:** `abs`, `ceil`, `floor`, `round`, `rand`, `sign`.

**Temporal:** `date`, `datetime`, `duration` (forma constructor solo con
ISO 8601 strings; no la álgebra completa todavía).

**Pattern:** `exists(pattern)`, `nodes(path)`, `relationships(path)`.

#### Tipos

`INTEGER` (64-bit signed), `FLOAT` (64-bit), `STRING`, `BOOLEAN`, `NULL`,
`LIST<T>` (heterogénea permitida — typecheck en runtime), `MAP<STRING, T>`,
`NODE`, `RELATIONSHIP`, `PATH`, `DATE`, `DATETIME` (sin timezone),
`DURATION`.

Out-of-scope v0: `BYTES`, `POINT`, `LOCALDATETIME`, `ZONEDDATETIME`,
`LOCALTIME`, `TIME`.

#### Semántica de NULL

Three-valued logic estándar Cypher:
- `NULL = NULL` → `NULL` (no `true`).
- `NULL AND false` → `false`, `NULL AND true` → `NULL`.
- `WHERE` filter rechaza rows con predicado `NULL` (como `false`).
- `IS NULL` / `IS NOT NULL` son las únicas formas de testear NULL.

#### Error model

`ParseError { code: ErrorCode, message: String, span: SourceSpan, help: Option<String> }`
donde `ErrorCode` es un enum exhaustivo (`E001_UnexpectedToken`,
`E002_UnboundedVariableLength`, `E003_ReservedKeyword`,...). Mensaje sigue el
formato de `ariadne` con caret highlighting y `help:` opcional. Múltiples
errores se reportan en la misma pasada via `chumsky::recovery`.

### Out-of-scope explícito v0

Lista exhaustiva — cualquier feature que NO esté aquí ni en el subset
in-scope falla con error de "feature no soportada" + número de RFC futuro
donde aterriza.

| Feature | Por qué afuera | Aterriza en |
|---|---|---|
| `shortestPath(...)` | Recursive pattern matching. Requiere WCOJ + planner especial. | RFC-009 |
| `allShortestPaths(...)` | Idem. | RFC-009 |
| `CALL {... }` (subqueries) | Subquery scoping rules son sutiles, no necesarias para LDBC SNB Interactive. | RFC futuro |
| `CALL procedure.name(...)` | No tenemos procedure registry. APOC explícitamente out. | RFC futuro |
| `FOREACH` | Imperativo, raramente útil. | RFC futuro |
| `USE database` | Cross-database queries. Single namespace por sesión en. | RFC-010 (cloud) |
| `LOAD CSV` | Bulk ingest path es `WriterSession`. | Nunca; usar el ingest API. |
| `CREATE INDEX` / `CREATE CONSTRAINT` | DDL fuera de Cypher; lo manejará el schema API directo. | RFC futuro |
| `EXPLAIN` / `PROFILE` | Pendiente pero ya con scope: vienen una vez exista LogicalPlan. | RFC futuro |
| Transacciones explícitas (`BEGIN`/`COMMIT`/`ROLLBACK` Cypher-level) | El cliente las maneja externamente via `WriterSession.commit_batch`. | Nunca via Cypher en v0. |
| Variable-length sin upper bound (`*1..`) | Sin upper bound el optimizador no puede limitar el blowup. | Posible relajación con WCOJ. |
| Pattern de longitud cero (`*0..n`) | Trivial pero abre dudas semánticas (auto-loops). | RFC futuro. |
| `MATCH p = (a)-[*]->(b) RETURN p` (paths como first-class) | Requiere materialización del path; útil pero no crítico para Q1–Q12. | RFC futuro. |
| Tipos `POINT`, `TIME`, `ZONEDDATETIME` | Sin uso en LDBC SNB Interactive. | RFC futuro cuando aterricen verticales geo / time-series. |
| `db.*` / `apoc.*` namespaces | Vendor-specific Neo4j; no portables. | Nunca. |

### Mapping a LDBC SNB Interactive Complex Q1–Q14

Cada query se evalúa contra el subset y se marca `IN` (parsea en v0) o
`OUT` (queda excluida hasta el RFC indicado).

| Query | Features requeridas | v0 |
|---|---|---|
| **IC1** — Friends by name (transitive) | `MATCH... *1..3... WHERE... ORDER BY... LIMIT` | ✅ IN |
| **IC2** — Recent messages by friends | `MATCH 2-hop... WHERE timestamp <... ORDER BY... LIMIT` | ✅ IN |
| **IC3** — Friends in two countries | `MATCH... WHERE country IN [...]` | ✅ IN |
| **IC4** — New topics on friend posts | `MATCH 2-hop + WITH + collect + UNWIND + WHERE NOT IN` | ✅ IN |
| **IC5** — New groups (membership count) | `MATCH... WITH... count + ORDER BY` | ✅ IN |
| **IC6** — Tag co-occurrence | `MATCH 2-hop... WITH tag, count... ORDER BY` | ✅ IN |
| **IC7** — Recent likers | `MATCH... ORDER BY... LIMIT` | ✅ IN |
| **IC8** — Recent replies | `MATCH... ORDER BY... LIMIT` | ✅ IN |
| **IC9** — Recent messages by friends-of-friends | `MATCH *2..2... WHERE... ORDER BY... LIMIT` | ✅ IN |
| **IC10** — Friend recommendation | `MATCH 2-hop... WITH common_count... ORDER BY` | ✅ IN |
| **IC11** — Job referral | `MATCH... WHERE... ORDER BY` | ✅ IN |
| **IC12** — Expert search by tag class | `MATCH 2-hop + tag class hierarchy + count + ORDER BY` | ✅ IN |
| **IC13** — Single shortest path | `shortestPath((a)-[*]-(b)` | ❌ OUT — RFC-009 |
| **IC14** — All shortest paths weighted | `allShortestPaths` + weight calc | ❌ OUT — RFC-009 |

**Cobertura v0:** 12/14 (85.7 %). IC13–IC14 son los únicos excluidos y ambos
requieren recursive pattern matching que el WCOJ planner desbloquea.

### Estructura del crate `namidb-query`

```
crates/namidb-query/src/
├── lib.rs # reexports públicos
├── parser/
│ ├── mod.rs # entry point: parse(&str) -> Result<Query, Vec<ParseError>>
│ ├── lexer.rs # &str → Vec<(Token, SourceSpan)>
│ ├── ast.rs # tipos AST (Query, Clause, Pattern, Expression,...)
│ ├── grammar.rs # chumsky combinators
│ ├── display.rs # Display impl canonical (round-trip)
│ └── error.rs # ParseError, ErrorCode, SourceSpan
└── tests/ # integration tests parser
```

LogicalPlan, optimizer y executor viven en módulos hermanos cubiertos
por RFCs hermanas — quedan fuera del scope de RFC-004.

### Dependencias agregadas

| Dep | Versión | Por qué |
|---|---|---|
| `chumsky` | 0.10 | Parser combinators con error recovery y AST-friendly. Justificado en §Alternativas. |
| `ariadne` | 0.5 | Pretty error messages (caret, span highlight, multi-error). |

No agregamos `nom`, `pest`, `lalrpop`, ni `antlr-rs`. Justificación en
§Alternativas.

## Alternatives considered

### A. Hand-written recursive descent parser

**Pro:** máxima velocidad de parsing, control absoluto de error messages,
sin dependency tree.
**Con:** ~3 000–5 000 LoC para cubrir el subset declarado, ~30–50 % del
tiempo se va en boilerplate de precedence + error recovery, refactor caro
cuando agregamos features.
**Veredicto:** Rechazado. Es la opción "Postgres" — válida cuando el parser
es el producto principal. Para nosotros el producto es el storage + executor,
el parser es overhead.

### B. `nom` parser combinators

**Pro:** maduro (~10 yrs), rápido, gran comunidad Rust.
**Con:** error messages requieren mucha plumbing manual (`VerboseError` ayuda
pero queda lejos de `ariadne`), no tiene recovery built-in, tipo de
combinators byte-stream-first (no token-stream-first) — friction natural con
un lexer tokenizado separado.
**Veredicto:** Rechazado. Es la mejor opción si el parser fuera la única
prioridad pero el dev experience de errores es inferior a chumsky.

### C. `chumsky` 0.10+

**Pro:** parser combinators con error recovery first-class
(`recovery::skip_then_retry_until`, `nested_delimiters`), AST-friendly
(retorna `Result<T, Vec<E>>` con todos los errores no solo el primero),
buena integración con `ariadne` para pretty errors, version 1.0 cerca.
**Con:** versión 0.10 cambió API significativamente vs 0.9 — un breaking
change vertical futuro probable. Slower que `nom` en benchmarks micro (~2×).
**Veredicto:** **Aceptado**. Velocidad de parsing es irrelevante en nuestro
workload (la query string viene del usuario una vez, se parsea, se cachea).
Error quality es lo que importa.

### D. ANTLR4 + generador de parser Rust (antlr-rust)

**Pro:** openCypher distribuye una gramática ANTLR oficial; reusarla evita
re-litigar precedencia y syntax edge cases; cobertura del estándar "para
free".
**Con:** `antlr-rust` no está bien mantenido (último release 2022), la
gramática openCypher cubre features que están out-of-scope (`shortestPath`,
`CALL`, `FOREACH`,...) y filtrarlos post-parse es más caro que parsear el
subset directo. ANTLR genera código que es pesado de leer; el debugging cuando
algo sale mal es difícil.
**Veredicto:** Rechazado. Reusar la gramática openCypher como referencia
informal — sí. Generar Rust desde ella — no.

### E. LALRPOP (LR(1) generator)

**Pro:** maduro, rápido, parser determinístico.
**Con:** Cypher no es LR(1) limpio (ambigüedad pattern vs expression dentro
de `WHERE` clauses), forzar grammar a LALR causa hacks. Error recovery en
LR(1) es notoriamente difícil.
**Veredicto:** Rechazado. LR genera grammars rígidas; quereremos evolucionar
rápido (futuro).

### F. Lexer separado vs lexer inline en chumsky

chumsky soporta parsear directo desde `&str` sin lexer (es lo idiomático en
muchos ejemplos). Decisión: **lexer separado**.

**Razones:**
- Comments (`//`, `/* */`) y whitespace son más limpios de manejar en lexer.
- Keyword vs identifier es ambigüedad léxica (`COUNT` puede ser función o
 identifier en algunos contextos) — resolverlo a nivel de token simplifica
 el grammar.
- Spans más precisos: cada token lleva su span; el parser solo conecta
 tokens, no recomputa offsets.
- Test independiente: el lexer puede testearse sin tocar el parser, y
 viceversa.

Costo: ~150 LoC extra de lexer. Aceptable.

## Drawbacks

1. **Subset muy chico** comparado con Neo4j (5 % de la superficie) — early
 adopters que vienen de Neo4j chocarán con "feature not supported" en cada
 feature avanzado. Mitigación: error message indica qué RFC futuro lo
 cubre, link a roadmap público.

2. **Cypher 25 está evolucionando**: GQL ISO/IEC 39075 puede ganar
 ammendments. Mitigación: rebase del subset cada release; RFC-004 se trata
 como living document (Status puede pasar a `superseded` cuando aparezca
 RFC-004.1 o RFC-004 v1).

3. **`chumsky` 0.10 → 1.0** breaking change esperado en próximos meses.
 Mitigación: encapsular el uso detrás de `parser::grammar::*` privado,
 refactor confinado a un módulo.

4. **`MERGE` con multi-label patterns** tiene semantics ambiguas (Neo4j y
 Memgraph difieren). Decisión: en v0 `MERGE` requiere exactamente un label
 por node pattern. `MERGE (a:A:B)` retorna error parser-level. Documentado
 en error code `E007_MergeMultiLabel`.

5. **`OPTIONAL MATCH` con variable-length** no está bien definida en el
 estándar (¿qué pasa con OPTIONAL en `*0..n`?). Decisión v0: rechazar la
 combinación en el parser. `E008_OptionalVariableLength`. Aterriza con
 RFC-009.

## Open questions

- **Q1: `RETURN *`** — en v0 no se soporta. ¿Lo agregamos más adelante cuando
 llegue el binding scope resolver? Likely sí, es feature high-value
 low-cost.

- **Q2: `WITH *`** — idem. Decisión deferida.

- **Q3: User-defined functions** — el plan §13.2 menciona RFC futura pero no
 está numerada. ¿`namidb.fn.*` namespace? ¿WASM sandbox? Out of scope
 v0; lo deciden.

- **Q4: `LOAD CSV`** — fuera explícitamente; pero usuarios que vienen de
 Neo4j lo van a buscar. ¿Documentamos un equivalente "`namidb-cli
 ingest --csv...`" o lo dejamos al SDK Python? Decisión separada de este
 RFC.

- **Q5: Identifiers con backticks** — `MATCH (a:`Foo Bar`)`. openCypher
 los permite, GQL los exige para identifiers con espacios o reserved
 words. Decisión: **soportar siempre** (mejor superset de standards).

## References

- GQL ISO/IEC 39075:2024 — https://www.iso.org/standard/76120.html
- openCypher 9 specification — https://opencypher.org/resources/
- Cypher 25 (Neo4j) — https://neo4j.com/docs/cypher-manual/current/
- LDBC SNB Interactive Workload, v0.4 — Erling et al., SIGMOD 2015.
- Memgraph Cypher subset — https://memgraph.com/docs/cypher-manual
- Kuzu Cypher compatibility — https://docs.kuzudb.com/cypher/ (snapshot
 pre-archive oct 2025).
- chumsky 0.10 documentation — https://docs.rs/chumsky/0.10
- ariadne — https://docs.rs/ariadne
- `recursive-descent` vs `combinators` discussion in Rust DBMS
 community — Niko Matsakis, "Why I built lalrpop" (2017); Geal
 blogposts on nom.
