# RFC 033: Rich nested/array properties and named/sparse/multi-vector indexes

**Status:** draft
**Author(s):** NamiDB team
**Created:** 2026-06-26
**Updated:** 2026-06-26
**Implements:** (none yet — design only)
**Relates-to:** RFC-013 (Parquet predicate pushdown), RFC-025 (per-label
property statistics), RFC-030 (DiskANN/Vamana vector index)

## Summary

Two feature requests recur in the same shape: "I want to store an
`acl: string[]` and filter on membership without JSON-string round-tripping",
and "I want more than one vector per node — named, sparse, or multi-vector".
Both are routinely described as *missing*, and both are partly **already
implemented**. This RFC first establishes, against the code, exactly what works
today and what does not, then designs the genuinely missing pieces as
independently-shippable increments with effort estimates.

The honest framing is: the "already works" surface is large enough that the
remaining work is smaller than it looks, but it is still real engineering. The
runtime value model (`namidb-core::Value` / `RuntimeValue`) already carries
lists and maps and the executor already evaluates every list/map operator
Cypher exposes; what is missing for **part A** is purely *typed columnar
storage, predicate pushdown, and indexing* over those values. For **part B**,
"named vectors" already exist as multiple `(label, property)` indexes; what is
missing is *sparse* vectors and *multi-vector / late-interaction* scoring,
which are new index kinds that do not reuse the dense Vamana path from RFC-030.

## Motivation

The pain the gap analysis surfaced was real but its diagnosis was wrong in a
way that matters for scoping. The original limitation note claimed users must
"serialize a list to a JSON string and re-parse it to query it." That is not
true today: a bare list round-trips and is queryable. But the *queryable* path
is an in-memory `Filter` over every candidate row with a JSON re-parse per row —
it never prunes a row group and never uses an index. So the user-visible
symptom (list filters are slow, nested-key filters are slow) is genuine even
though the stated cause (no list storage at all) is not.

Getting the framing right avoids two failure modes: re-implementing value
types that already exist (waste), and under-scoping the storage/index work that
is the actual cost (surprise). The cost of doing nothing is that the engine
keeps advertising list/map/vector properties it stores but cannot accelerate,
so any workload that filters on an `acl`/`tags`/`metadata.*` field or wants
hybrid sparse+dense retrieval degrades to a full label scan.

## Design

The design splits cleanly into two parts that share no code. Each subsection is
marked **Implemented (v1.4)** or **Proposed**, with `file:symbol` references so
the boundary is checkable.

---

### Part A — first-class list / nested-map properties

#### A.0 What is already implemented (v1.4)

The runtime and ingest layers already model lists and maps as first-class
values; only the storage/index layers treat them as opaque.

- **Value model.** `namidb-core::value::Value` carries `List(Vec<Value>)` and
  `Map(BTreeMap<String, Value>)` variants (`crates/namidb-core/src/value.rs`),
  with `$list` / `$map` serde tags (`TAG_LIST`, `TAG_MAP`) so an array shape is
  not silently re-decoded as a `Vec<f32>` vector. `RuntimeValue::List` /
  `RuntimeValue::Map` mirror them (`crates/namidb-query/src/exec/value.rs`), and
  `impl From<CoreValue> for RuntimeValue` maps both through. **No new value
  variant is needed.**
- **Cypher evaluation.** The executor already evaluates every list/map
  operator: `IN` (`ExpressionKind::In { item, list }`), subscript and slicing
  (`Index` / `Range`), list comprehensions (`ListComprehension`), and the four
  list quantifiers `any` / `all` / `none` / `single`
  (`QuantifierKind::{Any, All, None, Single}` in
  `crates/namidb-query/src/parser/ast.rs`, evaluated in
  `crates/namidb-query/src/exec/expr.rs`). So `CREATE (a {tags:['rust','ssh']})`
  followed by `WHERE 'rust' IN a.tags` or
  `WHERE any(t IN a.tags WHERE t = 'rust')` already returns the right rows.
- **Persistence.** `runtime_to_core` (`crates/namidb-query/src/exec/writer.rs`)
  routes `RuntimeValue::List` / `Map` into `CoreValue::List` / `Map`, which the
  flush path serialises into the single `__overflow_json` Utf8 column
  (`OVERFLOW_JSON` in `crates/namidb-storage/src/sst/nodes.rs`). The values
  round-trip; they are simply opaque on disk.

The load-bearing comment is in `runtime_to_core`:

```rust
// Lists store through the `__overflow_json` stream as a
// tagged JSON object; the writer cannot route them into a
// declared columnar property yet.
```

That "yet" is exactly the gap part A closes.

#### A.1 What is genuinely missing

1. **No list/map `DataType`.** `namidb-core::schema::DataType`
   (`crates/namidb-core/src/schema.rs`) has only scalars plus `FloatVector`,
   `Int8Vector`, and `Json`. There is no `DataType::List` / `DataType::Map`, so
   a declared list property can never become a typed `prop_<name>` Arrow
   column — it falls into `__overflow_json`.
2. **No list/nested predicate pushdown.** `ScanPredicate`
   (`crates/namidb-storage/src/sst/predicates.rs`) has only
   `Eq/Lt/LtEq/Gt/GtEq/IsNull/IsNotNull/In{column, scalar-values}`. There is no
   list-contains or nested-key variant. In `parquet_pushdown.rs`,
   `try_into_scan_predicate` recognises `IN` only when the *item* is a property
   and the *list* is literal (`property_column_for_alias(item)`); for
   `'x' IN n.tags` the item is a literal so it returns `None` and the conjunct
   stays a residual in-memory `Filter`. Nested-key access (`n.meta.k = v`) is
   likewise residual.
3. **No stats / index over list elements or map keys.**
   `hll_supported_for_datatype` (`crates/namidb-storage/src/sst/nodes.rs`)
   excludes `Json` and vectors, and there is no secondary index over list
   elements. Filters on a list/map property therefore never prune a row group
   and never resolve through an index.

Net: lists and maps are *storable and filterable* but never *accelerated*.

#### A.2 Proposed: typed homogeneous list columns

Add `DataType::List(Box<DataType>)` to
`namidb-core::schema::DataType`, mapping in `DataType::to_arrow` to Arrow
`List<T>`. The columnar machinery already exists and is exercised: the node SST
schema emits `__labels` as `List<UInt32>` via `ListBuilder<UInt32Builder>`
(`COL_LABELS` in `crates/namidb-storage/src/sst/nodes.rs`), and the read path
already handles a Parquet list leaf whose path is `<col>.list.element` (the
`root_of` projection helper in `NodeSstReader::scan_with_predicates_and_projection`).

A declared `acl: List<Utf8>` then materialises as a real `prop_acl` Arrow
`List<Utf8>` column instead of overflow JSON, and the flush path gains a
`ListBuilder` branch parallel to the existing scalar builders. Constraints to
keep the type honest:

- **Homogeneous, declared element type only.** `List(Box<DataType>)` where the
  inner type is a scalar (`Utf8`, `Int64`, …). Nested vectors-in-lists are
  rejected at `PropertyDef` construction, the same place the reserved-name
  checks live.
- **Heterogeneous lists and string-keyed maps stay in `__overflow_json`**
  (status quo, opaque) unless declared. This keeps the change additive: every
  existing manifest deserialises unchanged because the new variant is additive
  and undeclared properties keep their overflow path.

This is the prerequisite for A.3 and A.4 — without a typed element column there
is nothing for Parquet stats or an inverted index to key on.

#### A.3 Proposed: list-membership predicate pushdown

Add `ScanPredicate::ListContains { column, value: StatScalar }` to
`crates/namidb-storage/src/sst/predicates.rs`, with an `eval_row_group` verdict
that returns `Absent` only when the element column's Parquet min/max bracket
*provably* excludes `value`, and `MaybePresent` otherwise. This matches the
existing conservatism contract verbatim (`Absent` only on proof; anything
inconclusive decodes and lets the executor `Filter` apply 3VL).

In `parquet_pushdown.rs`, extend `try_into_scan_predicate` to recognise the two
membership shapes the executor already evaluates:

- `literal IN <alias>.<listprop>` — add a mirror branch to the existing `In`
  handling that treats the *list operand* as the alias property when the *item*
  is a literal (today `property_column_for_alias(item)` returns `None` here).
- `any(x IN <alias>.<listprop> WHERE x = literal)` — lower the `Quantifier`
  (`QuantifierKind::Any` with an equality body) to the same `ListContains`.

Everything not in these shapes stays residual, which is correctness-safe by the
module's existing contract. NULL / empty-list / 3VL semantics defer to the
residual `Filter` exactly as scalar `IN` already does (a NULL-bearing `IN` list
drops out of pushdown today).

#### A.4 Proposed: inverted secondary index for list membership

This is the real performance win for the `acl: string[]` case. Extend the
equality-index sidecar machinery — `EqualityIndexDescriptor`
(`crates/namidb-storage/src/manifest.rs`), whose body is already a
`BTreeMap<value, Vec<NodeId>>` posting list — so that for an `indexed` list
column the flush path emits one posting per *distinct element*
(`element_value -> node_ids`). `MATCH (d:Doc) WHERE 'u1' IN d.acl` then resolves
through the inverted index like a scalar equality index, instead of a label
scan + per-row JSON re-parse.

Two invariants carry over from existing index work and must be re-asserted in
tests:

- **Freshness.** The SST-backed inverted index must return exactly what the
  flat scan would. As with every other index, the read path unions the
  memtable/overlay delta (or gates and falls back) so a just-written `acl`
  element is findable immediately. A reachability test must assert the lowered
  plan *uses* the inverted index, not merely that results are equal — a flat
  fallback makes an equal-results assertion pass trivially.
- **Migration.** Existing list data lives in `__overflow_json`; a declared-list
  column and its index apply only to newly written / compacted SSTs, so during
  the transition the read path unions typed-column lists with overflow-JSON
  lists.

#### A.5 Proposed (optional): nested-map key access

Full `DataType::Map(K, V)` Arrow Map columns are heavy and only help
homogeneous maps; most `metadata: {...}` payloads are heterogeneous. The
pragmatic design keeps heterogeneous maps as opaque `Json`/overflow and adds:

- a JSON-path scalar accessor in `WHERE` (already half-present via `Index`
  subscript over a `Map` runtime value), and
- an optional **functional equality index** over an extracted path,
  `INDEX ON n.meta.source`, storing `extract(meta, 'source') -> node_ids` using
  the same posting-list sidecar as A.4.

This gives queryable nested keys without a full Map column. `DataType::Map` is
noted as a future option for declared homogeneous maps but is explicitly out of
scope here.

---

### Part B — named, sparse, and multi-vector indexes

#### B.0 Named vectors are already implemented (v1.4)

"Named vectors" in the Qdrant sense — several independently-searchable vectors
attached to one point — already exist in NamiDB as **multiple vector indexes on
the same label keyed by different properties**. The evidence:

- `register_vector_index` (`crates/namidb-storage/src/ingest.rs`) dedups on
  `name` **or** `(label, property, metric)` via `VectorIndexDescriptor::matches`
  (`crates/namidb-storage/src/manifest.rs`). Two `CREATE VECTOR INDEX`
  statements on the same label but different properties (or different metrics on
  one property) both register, each materialising its own `.vg` body.
- The KNN rewrite and `CALL` resolution select an index by name → resolving to
  `(label, property, metric)` (`crates/namidb-query/src/exec/walker.rs`,
  `optimize/vector_search.rs`).

So a NamiDB "named vector" is just *a property plus its index*. A node can
carry `title_emb` and `body_emb`, each with its own `CREATE VECTOR INDEX`, and
each KNN query names the property/index it wants. The only ergonomic gap versus
Qdrant is addressing-by-vector-name-within-a-point vs.
addressing-by-property-within-a-node.

**Proposed (thin):** an optional `CREATE VECTOR INDEX … AS <vectorName>` alias
that maps a friendly name to a `(label, property)`. Structurally a no-op over
the existing `VectorIndexDescriptor` (add an optional `alias` field, resolve it
in the same place `name` resolves). It must reuse the existing
`(label, property, metric)` dedup so two aliases cannot collide on one property
with conflicting metrics. This is documentation-plus-sugar, shippable now.

#### B.1 Proposed: sparse vectors (new index kind)

Sparse vectors (SPLADE / BM25-style term-weight vectors) are genuinely missing.
The dense path is fixed-dimension: `DataType::FloatVector { dim }` /
`Int8Vector { dim }`, a `VectorGraphBody` holding one dense vector per node
(`VectorStorage::F32` / `Int8` in `crates/namidb-storage/src/sst/vector.rs`),
and the `namidb-ann::VectorSpace` trait whose `dim()` and `query_distance`
assume a single fixed-width dense vector (`crates/namidb-ann/src/space.rs`).
None of this represents a high-dimensional mostly-zero vector.

Design as a **separate index kind**, not a modification of the dense path:

- **Type.** `DataType::SparseVector` represented as two parallel columns
  (`indices: List<UInt32>`, `values: List<Float32>`) or one Struct column.
- **Body.** A new `SstKind::SparseVectorIndex` whose body is an **inverted
  index** (`term -> postings of (node_id, weight)`), not the dense Vamana graph.
- **Descriptor.** A `SparseVectorIndexDescriptor` in the manifest, registered
  through a `register_vector_index`-parallel path.
- **Scorer.** A sparse dot-product top-k (WAND / block-max-WAND) over the
  postings. This is a new scorer, *not* a `VectorSpace` impl — `VectorSpace` is
  fixed-dim dense and must stay that way.
- **Surface.** A `CALL` procedure (resolved in `walker.rs`) or a sparse-KNN
  rewrite, parallel to the dense rewrite.

The dense `namidb-ann` crate is untouched. The main risk is scope, not
regression, because nothing is shared with the Vamana path.

#### B.2 Proposed: multi-vector / late interaction (max-sim)

Multi-vector / late-interaction retrieval (ColBERT-style) attaches *N* vectors
to one node and scores a query against a node by aggregating per-token
similarities (max-sim). This is the largest change because both the build
pairing and the scoring in RFC-030 assume exactly one vector per node:

- `VectorGraphBody` (`crates/namidb-storage/src/sst/vector.rs`) stores `ids:
  Vec<[u8;16]>` parallel to `storage` and the graph adjacency — one `NodeId`
  per graph node, one vector per `NodeId`.
- the build pairs `node_id -> single vector`, and search produces one score per
  stored vector.

Design:

- **Body layout.** Map `node_id -> [ordinal_range]` so a node owns *N* vectors;
  store vectors per ordinal; the Vamana graph indexes ordinals, not nodes.
- **Scoring.** Add a max-sim aggregation over the per-token scores at search
  time (for each query token take the best matching node token, sum over query
  tokens).
- **Gating.** A `multi_vector: bool` (or `aggregation: MaxSim`) flag on
  `VectorIndexDescriptor`. The `.vg` body layout changes, so it needs a body
  version bump — `VectorGraphBody` is already versioned — and the reader must
  reject or upgrade older single-vector bodies.

Because this rewrites the body layout *and* the scoring loop, it is the highest
cost of the three vector increments and should ship last.

---

### Effort and sequencing

Estimates are engineer-weeks for one engineer familiar with the codebase,
including tests and the freshness/parity invariants.

| Increment | Scope | Independently shippable | Effort |
|-----------|-------|--------------------------|--------|
| A.0 (frame) | Document list/map already store + evaluate | yes | docs only |
| A.1 typed list columns | `DataType::List`, Arrow `List<T>`, flush builder | yes | 1–2 wk |
| A.2 list pushdown | `ScanPredicate::ListContains`, pushdown branches | needs A.1 | 1 wk |
| A.3 inverted list index | element postings on the equality sidecar | needs A.1 | 1.5–2.5 wk |
| A.5 nested-map functional index | JSON-path accessor + functional eq index | independent of A.1 | 1.5–2 wk |
| B.0 named-vector alias | optional `AS <name>` over existing descriptor | yes (now) | a few days |
| B.1 sparse vectors | new `DataType`, inverted body, sparse scorer | yes | 3–4 wk |
| B.2 multi-vector / max-sim | per-node ordinals + max-sim, body version bump | yes | 4–6 wk |

High-value path: **A.1 → A.2 → A.3** covers the `acl: string[]` use case end to
end (typed storage, pushdown, inverted index). **B.0** is shippable immediately
as documentation-plus-sugar. **B.1** and **B.2** are independent new index kinds
that can land in either order. A.5 (nested maps) is lower priority and can ship
as functional indexes whenever a workload needs it.

## Alternatives considered

- **Add `DataType::Map(K, V)` as a full Arrow Map column.** Rejected as the
  default for nested maps: most `metadata` payloads are heterogeneous, the Map
  column is heavy, and a functional index over an extracted path (A.5) gives the
  queryable-key benefit without it. Kept as a future option for declared
  homogeneous maps.
- **Force users to keep serialising lists to JSON strings.** This is the status
  quo *minus* the typed columns, and it is what the gap analysis assumed was
  already required. Rejected because the runtime already stores and evaluates
  lists natively; doubling down on stringly-typed storage would throw away the
  `$list`/`$map`-tagged round-trip that already works.
- **Model named vectors as multiple vectors inside one point (Qdrant shape).**
  Rejected because the property-per-vector model already delivers the capability
  through the existing `(label, property, metric)` descriptor, and a per-point
  multi-named container would duplicate that machinery. The thin `AS <name>`
  alias (B.0) closes the only real ergonomic gap.
- **Bolt sparse / multi-vector onto the dense Vamana `VectorSpace`.** Rejected:
  `VectorSpace` is fixed-dimension dense by contract (`dim()`,
  `query_distance` over a single dense member). Sparse is an inverted index with
  a sparse-dot scorer; multi-vector needs per-node ordinals and max-sim. Forcing
  either through the dense trait would corrupt the recall-golden dense path.

## Drawbacks

- **A.1 introduces a typed/overflow split for the same logical property.**
  Until old SSTs compact, a declared list lives partly in `prop_<name>` and
  partly in `__overflow_json`, so the read path must union both during the
  transition. This is the same migration shape declared scalar properties
  already have, but it must be tested (a declared-list reachability test that
  asserts the index is used, not just that results match).
- **List-element Parquet stats are weak for high-cardinality string lists.** The
  min/max bracket rarely prunes, so A.2's row-group pruning is best-effort; the
  inverted index (A.3) is the real win and A.2 is mostly a correctness-safe
  fast path.
- **B.1 and B.2 are net-new index kinds.** They add storage formats, manifest
  descriptors, and scorers that the dense path does not share. The cost is
  scope and surface area, not regression risk to existing vector search.
- **B.2 changes the `.vg` body layout.** It requires a body version bump and a
  reader that rejects or upgrades single-vector bodies; a botched migration
  would wedge compaction for the namespace (the descriptor lives in the
  manifest), so the same fail-fast register-time validation RFC-030 uses for
  int8/cosine must apply to any new modality flag.

## Open questions

- Should the inverted list index (A.3) and the functional map index (A.5) share
  one posting-list sidecar format, or stay distinct descriptors? Sharing reduces
  code but couples their evolution.
- For sparse vectors (B.1), do we expose a first-class `DataType::SparseVector`
  or piggyback on two declared list columns plus a sparse-index descriptor that
  names them? The latter is lighter but less ergonomic.
- For multi-vector (B.2), is max-sim the only aggregation we commit to, or do we
  leave room for sum/mean by making `aggregation` an enum from day one?
- Whether the `AS <vectorName>` alias (B.0) is worth the grammar surface, given
  property-per-vector already works and is arguably clearer.

## References

- RFC-030 — DiskANN/Vamana vector index (the dense path this RFC builds beside,
  not on top of).
- RFC-013 — Parquet predicate pushdown (the `ScanPredicate` / `eval_row_group`
  contract A.2 extends).
- RFC-025 — per-label property statistics (the HLL / min-max stats A.2/A.3
  would extend to list elements).
- Santhanam et al., "ColBERTv2: Effective and Efficient Retrieval via
  Lightweight Late Interaction" (max-sim late interaction).
- Formal et al., "SPLADE: Sparse Lexical and Expansion Model for First Stage
  Ranking" (learned sparse vectors).
- Broder et al. / Ding & Suel, WAND and block-max-WAND (sparse top-k scoring).
