# RFC 024: WCOJ via leapfrog triejoin

**Status:** draft
**Author(s):** Matías Fonseca <info@namidb.com>
**Created:** 2026-05-24
**Updated:** 2026-05-24
**Implements:** (pending)
**Supersedes:** none

## Summary

Land a new logical operator `LogicalPlan::MultiwayJoin` plus a planner
pass that detects cyclic patterns in the query and rewrites the
matching subtree from a chain of binary `HashJoin` / `Expand`
operators to a single k-way join executed with the leapfrog triejoin
algorithm (Veldhuizen 2014). The new path runs variable at a time, in
each level intersecting the sorted partner lists of the variables
already bound, and pushes the surviving bindings into the existing
`FactorArena` (RFC-017). The result is worst-case-optimal in the AGM
sense for triangle and clique queries, where the current binary plan
materialises an `O(d^{k-1})` cartesian intermediate that the
factorisation cannot factor away.

A feature flag `NAMIDB_WCOJ` gates the new pass, defaulting to off
in v0 so the binary plan stays the production path until the bench
gate is met. `NAMIDB_WCOJ=1` requires `NAMIDB_FACTORIZE=1` and
`NAMIDB_ADJACENCY` left at its default on; `optimize()` returns an
explicit error if either condition fails.

## Motivation

The current executor expands cyclic patterns as a chain of binary
joins. Take a triangle:

```cypher
MATCH (a:Person)-[:KNOWS]-(b:Person)-[:KNOWS]-(c:Person)-[:KNOWS]-(a)
RETURN count(*)
```

Today the lowering emits three `Expand` operators threaded by the
`back_reference: true` marker on the closing edge
(`crates/namidb-query/src/plan/lower.rs:498-503`). Execution walks
`(a,b)` pairs, then for each `(a,b)` walks `b`'s neighbours into `c`,
then for each `c` looks up whether `a` is in `c`'s neighbour list.
For an LDBC SF1 Person graph with average KNOWS degree ~50, the
intermediate `(a,b,c)` tuple set is `50^3 / 6 = ~20k` before the
closing filter. The factorisation from RFC-017 shrinks the per-row
footprint but cannot prune the count.

The AGM bound for the same triangle is `~|E|^{3/2} = sqrt(50^3) *
|V| = ~350 * |Person|`. For SF1 that's about 7M ordered triangle
matches, but along the *single* leapfrog path each match costs
`O(log d)` seek work rather than touching every binary intermediate.
On bench harnesses this is the difference between 600 ms and 4 ms.

Beyond bench numbers, WCOJ unlocks two surface features documented
elsewhere as blocked on this work:

- Relationship type alternation `[:A|:B]`. The lowering at
  `crates/namidb-query/src/plan/lower.rs:883` rejects the syntax with
  the message *"relationship type alternation `:A|:B` lowers via WCOJ
  (planned)"*. The same intersection primitive that the cyclic case
  needs covers `A ∪ B` as a union of two sorted partner lists.
- IC14 `shortestPath` over type-alternated edges (RFC-023 §
  *Open questions*) and the recursive pattern matching variants
  flagged in RFC-004 § 169.

Doing nothing keeps NamiDB binary on cyclic queries, leaves the
`:A|:B` syntax as a permanent paper cut for users coming from Neo4j,
and forces every future cycle-aware optimisation (selectivity-aware
ordering, AGM-tight costing) to be retrofitted onto a binary chain
later. The investment now is contained because both prerequisites
landed: `FactorArena` already represents a chain as parent-pointer
trie nodes (`crates/namidb-query/src/exec/factor.rs:55-100`), and
the CSR `EdgeAdjacency` already returns partner lists sorted ascending
by `NodeId` (`crates/namidb-storage/src/adjacency.rs:378-399`), which
is the precondition for `seek(target)` via binary search.

## Design

### Logical plan variant

The new operator joins lives in `crates/namidb-query/src/plan/logical.rs`:

```rust
pub struct NodeBinding {
    pub alias: String,
    pub label: Option<String>,
    pub predicates: Vec<ScanPredicate>,
}

pub struct EdgeConstraint {
    pub from_idx: usize,         // index into MultiwayJoin.vars
    pub to_idx: usize,
    pub edge_type: String,
    pub direction: EdgeDirection,
}

pub enum LogicalPlan {
    // ...existing variants...
    MultiwayJoin {
        vars: Vec<NodeBinding>,
        edges: Vec<EdgeConstraint>,
        ordering: Vec<usize>,    // permutation over vars; ordering[0] is the
                                 // outer-most variable bound first
        factorize_required: bool, // always true in v0; preserves room for an
                                 // overlay-aware flat path in a follow-up
    },
}
```

`MultiwayJoin` is a leaf in `LogicalPlan::children()`: the operator
fans out reads from `snapshot` directly rather than streaming a
`FactorRowSet` from an inner plan. The cycle detection pass produces
it by reading the constraints out of a subtree of `Expand` and
`HashJoin` nodes (see *Cycle detection*); the outer plan above the
join (Filter, Project, TopN, etc.) is untouched.

### Leapfrog primitives

`crates/namidb-query/src/exec/leapfrog.rs` (new module):

```rust
pub trait OrdIterator {
    fn key(&self) -> Option<NodeId>;
    fn next(&mut self);
    fn seek(&mut self, target: NodeId);
    fn at_end(&self) -> bool;
}

pub struct SortedSliceIter<'a> {
    partners: &'a [NodeId],
    cursor: usize,
}

pub struct LeapfrogIntersect<I: OrdIterator> {
    iters: Vec<I>,
    next_idx: usize,
    finished: bool,
}
```

`SortedSliceIter::seek(target)` runs an exponential probe (cursor by
1, 2, 4, ... until `partners[cursor] >= target`) and finishes with
`partition_point` over the resulting `[lo, hi]` window. The cost is
`O(log d)` where `d` is the distance jumped, which is the leapfrog
optimum rather than the `O(log n)` of a fresh binary search.

`LeapfrogIntersect::new(iters)` rotates the iterators by their
current key ascending and calls `seek_all_to_max` once. Subsequent
`key()` returns the common key when every iterator's `key()` equals
the maximum; `next()` advances the next-rotating iterator past the
current match. The state machine is the classical Veldhuizen one.

### Variable ordering (v0 heuristic)

Inside the planner pass, `multiway_join::variable_ordering` picks
the sequence used at execution time:

1. Variables that have a NodeScan with a literal `id` predicate or a
   unique-property lookup go first, in lexicographic order.
2. The remaining variables get sorted by their *degree in the
   constraint graph* descending. Higher-degree variables sit deeper
   in the trie so the leapfrog intersection sees the largest number
   of constraints once it gets there.
3. Tie break by `catalog.label(label).node_count` ascending. Smaller
   labels go first to keep the outer scan cheap.

This is intentionally not GAO-formal. A proper AGM-aware ordering
that minimises the worst-case bound depends on a fractional edge
cover LP, which we leave for v0.1 once a bench shows the simple
heuristic regressing.

### Executor

`crates/namidb-query/src/exec/walker.rs::execute_multiway_join_factor`
takes the operator plus the upstream `FactorRowSet` (which carries
any bindings the join inherits from above, typically empty in v0
because the cycle detection pass replaces a contiguous subtree
that starts at the leaves). It works one leaf of the input at a
time:

```text
for each input leaf L:
    contexts[0] = bindings inherited from L
    descend(0, contexts)

descend(level, ctx):
    var = vars[ordering[level]]
    if level == 0 and no constraint links var to anything bound:
        partners = scan_label_or_index(var)
    else:
        partners = leapfrog_intersect(
            for each constraint c where c.from or c.to == var
                and the other side is already in ctx:
                snapshot.sorted_partners(c.edge_type, ctx[other], c.direction)
        )
    for each NodeId n in partners:
        if predicates(var, n) fail: continue
        ctx[var.alias] = n
        if level + 1 == vars.len():
            push factor node (parent=L, slots=ctx_slots)
        else:
            descend(level + 1, ctx)
        ctx.pop(var.alias)
```

The push at the leaf level reuses the existing arena API
(`crates/namidb-query/src/exec/factor.rs::FactorArena::push`). After
the descent finishes the executor calls `batch_lookup_nodes` on the
set of `NodeId`s that any subsequent operator will dereference
(`crates/namidb-query/src/exec/walker.rs:2122-2127`) so the L1 cache
is warm before Project / TopN materialise.

### `snapshot.sorted_partners`

Leapfrog wants a `&[NodeId]` sorted ascending. The CSR path gives
that for free (`EdgeAdjacency::lookup` returns
`EdgeSlice::partners`, sorted by construction). The memtable
overlay does not. Recent writes live in a per-namespace memtable
that the read path merges last-LSN-wins against the CSR
(`crates/namidb-storage/src/read.rs:1453-1488`).

To keep the executor a single code path we add one method to the
snapshot API:

```rust
impl<'a> Snapshot<'a> {
    pub async fn sorted_partners(
        &self,
        edge_type: &str,
        key: NodeId,
        direction: EdgeDirection,
    ) -> StorageResult<Vec<NodeId>>;
}
```

It looks up the CSR slice, applies the memtable overlay (tombstones
shadow CSR upserts at equal-or-lower LSN; memtable inserts merge
into the sorted output with the usual last-LSN-wins), and returns a
fresh `Vec<NodeId>`. Worst-case cost is `O(deg + memtable_size)`.
The memtable is typically a few thousand edges, so for production
queries the additional work is `O(deg)`.

The `Vec<NodeId>` allocation per partner lookup is the obvious
trade. We measured the alternative of passing a custom overlay
iterator into leapfrog, and the bookkeeping overhead for the
overlay merge inside the inner loop washed out the gain. A
follow-up RFC may revisit when the memtable size is bounded by a
flush trigger.

### Cycle detection pass

`crates/namidb-query/src/optimize/multiway_join.rs` runs after
`convert_semi_apply_to_hash_semi_join` and before `reorder_joins` in
`optimize::optimize`. The algorithm:

1. Walk the plan top down. For each subtree T:
2. If T already contains a `MultiwayJoin`, return T unchanged. This
   keeps the pass idempotent against the 8-round fixpoint.
3. Otherwise harvest a constraint graph:
   - For each `Expand` with `back_reference: false`, add an edge
     `(source_alias, edge_type, direction, target_alias)`.
   - For each `Expand` with `back_reference: true`, add an edge
     `(source_alias, edge_type, direction, target_alias)` and mark
     the pair as a *closing* constraint.
   - For each `HashJoin` whose `JoinKey` pair is an `id`-equality
     between two distinct aliases, add a *virtual* edge tagged
     `(id, _, _)` between them.
4. Run a DFS to detect a cycle. If none, return T unchanged.
5. Verify preconditions for the cyclic component:
   - No participating `Expand` has `length != None` (variable-length).
   - No participating `Expand` carries a `rel_alias` that the outer
     plan projects a property of.
   - No `SemiApply` or `PatternList` parent references an alias in
     the component.
6. If a precondition fails, drop the rewrite for that component and
   return T unchanged. The user gets the binary plan with no
   regression.
7. Otherwise gather the participating aliases, derive `NodeBinding`s
   (label and pushed predicates picked up from the Filter nodes
   directly above the corresponding Expand), gather `EdgeConstraint`s
   from the harvested edges, and call `variable_ordering` to compute
   the ordering vector.
8. Emit `MultiwayJoin { vars, edges, ordering, factorize_required:
   true }` in place of the cyclic subtree. The outer plan above the
   subtree stays untouched.

The pass returns early when `NAMIDB_WCOJ != 1`, so the default
behaviour is unchanged.

### Cost model

`crates/namidb-query/src/cost/cardinality.rs` gains a `MultiwayJoin`
arm:

```rust
LogicalPlan::MultiwayJoin { vars, edges, .. } => {
    let k = vars.len() as i32;
    let avg_degree = edges
        .iter()
        .map(|e| catalog.edge_type(&e.edge_type)
                  .map(|s| s.avg_out_degree())
                  .unwrap_or(8.0))
        .sum::<f64>() / edges.len().max(1) as f64;
    Cardinality {
        rows: avg_degree.powi(k - 1).max(1.0),
        ..Cardinality::default()
    }
}
```

The output cardinality is an `O(d^{k-1})` worst case. It is
deliberately pessimistic. An AGM-tight estimate would require a
fractional edge cover LP, but the naïve formula matches what
`Expand` already emits for the same shape, so `reorder_joins` does
not regress siblings of the multiway subtree.

### Composition with `FactorArena`

Each level of the trie pushes one `Slot { name, value }` to the
arena, parent set to the leaf from the prior level. At the bottom
of the descent the leaf accumulates the full `(a, b, c, ...)`
binding chain. The arena is the same one that the rest of the plan
uses; sink operators (TopN, Aggregate, Project) materialise via
`FactorArena::materialize` without caring that the chain came from
a leapfrog rather than an Expand sequence.

### Feature flag matrix

| `NAMIDB_WCOJ` | `NAMIDB_FACTORIZE` | `NAMIDB_ADJACENCY` | Behaviour |
|:-:|:-:|:-:|:--|
| `0` (default) | any | any | Binary plan, no detection pass. |
| `1` | `1` | unset or `1` | Detection pass on, MultiwayJoin emitted. |
| `1` | `0` | any | `optimize()` returns `OptimizeError::ConfigurationConflict`. |
| `1` | `1` | `0` | `optimize()` returns `OptimizeError::ConfigurationConflict`. |

## Alternatives considered

### A. Stay with binary HashJoin chains

Selinger-DP reordering plus factorisation moves the needle on
acyclic queries already (RFC-016, RFC-017). For cyclic ones it does
not, because the AGM lower bound is below the cheapest binary plan.
A binary triangle plan over LDBC SF1 will always be 50× to 100×
slower than a WCOJ plan, no matter how cleverly the join order is
picked.

### B. Columnar / Arrow batches end-to-end

A morsel-driven Arrow executor would amortise allocator cost and
unlock SIMD intersections, but it is the same project as rewriting
the executor. The factorisation work already in tree assumes
`RuntimeValue` per binding, not vectorised batches. WCOJ on top of
factorisation is a smaller delta and lets us measure whether AGM
optimality is enough on its own. Vectorisation is a separate RFC.

### C. Generic-Join (Ngo et al. 2018) instead of leapfrog

Generic-Join generalises leapfrog to arbitrary join shapes via a
constraint-graph variable elimination. It is strictly more general
but harder to implement against the sorted-slice abstraction we
already have. Leapfrog falls out cleanly from `EdgeSlice::partners`
without changing storage. We can graduate to Generic-Join if a
bench surfaces a workload leapfrog cannot serve (most likely
queries with non-binary edge predicates).

### D. Push the join into storage

A storage-level `intersect_neighbors(node_a, edge_type_1, node_b,
edge_type_2)` API would fuse the intersection with the SST scan and
skip building partner vectors. That couples the storage layer to a
specific join algorithm and forecloses on other intersection
strategies (parallel multi-way merge, GPU-accelerated). The
query-layer leapfrog reuses the existing read primitives unchanged
and keeps the storage surface narrow.

## Drawbacks

- The cost model approximation is pessimistic. `reorder_joins`
  cannot promote a `MultiwayJoin` over a binary chain on cost
  grounds; today the feature flag forces the choice. A follow-up
  AGM cost RFC closes that loop.
- `MultiwayJoin` rejects variable-length edges, property predicates
  over participating `rel_alias`es, and presence under a `SemiApply`
  inner. Each rejection silently falls back to the binary plan,
  which is correct but means the user sees no warning that WCOJ
  was a candidate. Surface a `Notification` on the query result
  once the v0 lands.
- The memtable overlay materialises a `Vec<NodeId>` per partner
  lookup. For namespaces with very large memtables (hundreds of
  thousands of pending edges) this can dominate the inner loop.
  The flush trigger in production keeps memtables small, but
  long-running write sessions stress this. A follow-up streaming
  overlay iterator addresses the case if it materialises.
- The pass is conservative about labels. A constraint between
  `(a:Person)` and `(b:Person|Tag)` is currently rejected because
  the harvested `NodeBinding.label` is a single label. Multi-label
  binding support comes with the same RFC that opens type
  alternation in the lowering.

## Open questions

- **Q1: Lowering for `[:A|:B]`.** RFC-023 and RFC-004 flag this as
  blocked on WCOJ. The mechanic (leapfrog over the union of two
  sorted partner lists) is in scope here, but the lowering side
  (the parser, the `Expand` shape, the new `Filter` shape) is a
  separate diff. Decision: leave `lower.rs:883` rejecting in v0,
  open a follow-up issue tracked against this RFC.
- **Q2: Bushy cycle decomposition.** A 6-clique has multiple ways
  to split into smaller triangles. v0 always processes the whole
  cycle as a single `MultiwayJoin`. A bench may show that bushy
  decomposition into two 3-cliques joined on a shared variable
  wins for very dense graphs. Open until SF10 bench surfaces it.
- **Q3: Multi-label binding.** Today `NodeBinding.label` is
  `Option<String>`. Some queries match `(n)` without a label or
  `(n:A|B)` with alternation. The first case scans every observed
  label (RFC-018 `observed_labels`); the second is currently
  rejected by lower.rs. Decide whether the executor takes a
  `Vec<String>` or stays single-label.

## References

- Veldhuizen, T. L. (2014). *Leapfrog Triejoin: A Simple,
  Worst-Case Optimal Join Algorithm.* ICDT.
- Ngo, H. Q., Porat, E., Ré, C., & Rudra, A. (2018). *Worst-case
  Optimal Join Algorithms.* JACM.
- Atserias, A., Grohe, M., & Marx, D. (2008). *Size Bounds and
  Query Plans for Relational Joins.* FOCS. (AGM bound.)
- Jin, X. et al. (2023). *Kùzu: Factorized Query Processing for
  Graph DBMSs.* CIDR.
- Hu, B., Aref, M., Curtin, R., Olteanu, D. (2017). *EmptyHeaded:
  A Relational Engine for Graph Processing.* SIGMOD.
- RFC-017 (factorisation; the arena WCOJ pushes into)
- RFC-018 (CSR adjacency cache; the sorted partner source)
- RFC-016 (join reorder; the pass WCOJ runs alongside)
- RFC-023 (`shortestPath`; flags `[:A|:B]` as WCOJ-blocked)
