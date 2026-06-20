//! Logical plan IR.
//!
//! See [`docs/rfc/008-logical-plan-ir.md`](../../../../docs/rfc/008-logical-plan-ir.md).

use namidb_storage::sst::predicates::ScanPredicate;
use serde::{Deserialize, Serialize};

use crate::parser::{Expression, OrderDirection, RelationshipDirection, RelationshipLength};

/// Shortest-path variant attached to [`LogicalPlan::Expand`]. See
/// RFC-023.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum ShortestMode {
    /// Regular variable-length traversal — emit every reachable
    /// row.
    #[default]
    None,
    /// `shortestPath((a)-[*..N]-(b))` — at most one row per
    /// (source, target) pair, emitted at the first hop the target
    /// appears.
    First,
    /// `allShortestPaths(...)` — every distinct path of the minimum
    /// length.
    All,
}

/// Vector distance metric for [`LogicalPlan::VectorSearch`] (RFC-030). Mirrors
/// the builtins `cosine_similarity` / `dot_product` / `euclidean_distance` —
/// the optimizer matches a query's distance function against this to decide
/// whether a `VectorIndexDescriptor` can serve the lookup.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum VectorDistance {
    Cosine,
    Dot,
    Euclidean,
}

impl VectorDistance {
    /// The builtin Cypher function name that computes this metric.
    pub fn builtin_name(self) -> &'static str {
        match self {
            VectorDistance::Cosine => "cosine_similarity",
            VectorDistance::Dot => "dot_product",
            VectorDistance::Euclidean => "euclidean_distance",
        }
    }
}

/// Tree of relational/graph operators produced by lowering the AST and
/// consumed by the executor. See RFC-008.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum LogicalPlan {
    /// Scan all nodes carrying `label`. `predicates` is the list of
    /// single-column conjunctive predicates pushed by `optimize`
    /// (RFC-013) — empty when lowering, populated when a `Filter`
    /// directly above carried conjuncts the storage layer can use for
    /// row-group pruning. `projection` (RFC-015) is `None` when the
    /// query references the binding as a whole or no projection
    /// analysis ran; `Some(cols)` ⇒ storage decodes only those
    /// property columns plus the engine columns.
    NodeScan {
        /// `Some(label)` restricts the scan to one label; `None`
        /// fans out across every label observable through the
        /// snapshot (`Snapshot::observed_labels`). Lowering produces
        /// `None` for `MATCH (n)` without a label predicate.
        label: Option<String>,
        alias: String,
        predicates: Vec<ScanPredicate>,
        projection: Option<Vec<String>>,
    },

    /// Point-lookup by id. Lowering emits this for an inline `{_id: <expr>}`
    /// filter on a node pattern, and `optimize::unique_lookup` emits it for
    /// `WHERE elementId(n) = <expr>` / `WHERE id(n) = <expr>`. The id
    /// expression is evaluated against each row from `input` (typically
    /// `Empty`). `label = Some(L)` scopes the lookup to one column family;
    /// `label = None` (the unlabelled `MATCH (n) WHERE elementId(n) = ...`
    /// shape a GUI fetch uses) fans out across every observed label.
    NodeById {
        input: Box<LogicalPlan>,
        label: Option<String>,
        alias: String,
        id: Expression,
    },

    /// Point-lookup by a *unique* user property (RFC-pending). Lowering
    /// emits this when the AST contains an inline filter
    /// `{<unique_prop>: <expr>}` on a node pattern AND the property is
    /// declared `unique` in the schema. The executor calls
    /// `Snapshot::lookup_node_by_property(label, property, value)` —
    /// O(log n) or better depending on the storage-side index. Falls
    /// back to NodeScan + Filter when no unique property is matched on
    /// the pattern.
    NodeByPropertyValue {
        input: Box<LogicalPlan>,
        label: String,
        alias: String,
        property: String,
        value: Expression,
        /// `false`: a *unique* property — at most one match, resolved
        /// through the unique sidecar (point lookup). `true`: a non-unique
        /// `indexed` property — fan out one row per match, resolved through
        /// the equality posting-list sidecar.
        multi: bool,
    },

    /// Expand `source` across an edge to produce `target_alias`.
    Expand {
        input: Box<LogicalPlan>,
        source: String,
        /// Edge type filter. `None` matches any observed type
        /// (`MATCH (a)-[]->(b)`); a non-empty `Some(vec)` restricts the
        /// traversal to the listed types. The executor unions the
        /// partner lists across all listed types. `Some(vec![])` is
        /// invariant-violating — the lowering refuses to produce it and
        /// callers may panic on `vec.first().unwrap()` if they get one.
        ///
        /// Cypher `[:A]` lowers to `Some(vec!["A"])`; alternation
        /// `[:A|:B|:C]` lowers to `Some(vec!["A","B","C"])`.
        edge_type: Option<Vec<String>>,
        direction: RelationshipDirection,
        rel_alias: Option<String>,
        target_alias: String,
        /// Labels declared on the target node pattern. Empty means no label
        /// constraint; a non-empty set is CONJUNCTIVE — the matched neighbour
        /// must carry every listed label (`(a)-[:R]->(b:A:B)`). The executor
        /// uses the first as the `lookup_node` / CF scan hint and then confirms
        /// the full set, which lets OPTIONAL MATCH preserve a NULL row when a
        /// neighbour carries only some of the labels.
        target_labels: Vec<String>,
        length: Option<RelationshipLength>,
        /// `true` for `OPTIONAL MATCH`: emit a row with `target=NULL`
        /// when no neighbour matches.
        optional: bool,
        /// `true` when `target_alias` was already bound in scope before
        /// this Expand (LDBC IC01-shaped path-existence patterns:
        /// `MATCH (p), (f) ... MATCH (p)-[:KNOWS*1..3]-(f)`). The
        /// executor must NOT overwrite the existing binding; instead,
        /// at every emission point it asserts the discovered tail node
        /// is identical to the bound target. Variable-length traversals
        /// still explore the frontier freely, but only paths that
        /// terminate at the existing target survive.
        back_reference: bool,
        /// `None` for an ordinary `[*min..max]` traversal. `Some(...)`
        /// for `shortestPath((a)-[*..N]-(b))` (`First`) or
        /// `allShortestPaths(...)` (`All`); the executor terminates
        /// the BFS at the hop that first reaches `target_alias`. See
        /// RFC-023.
        shortest: ShortestMode,
        /// When `shortest != None` and the user bound the path
        /// (`MATCH p = shortestPath(...)`), the executor materialises
        /// the alternating Node / Rel trail into the named runtime
        /// variable so `length(p)` etc. work. `None` for ordinary
        /// expands.
        path_binding: Option<String>,
    },

    /// Selection predicate.
    Filter {
        input: Box<LogicalPlan>,
        predicate: Expression,
    },

    /// Projection (RETURN / WITH).
    Project {
        input: Box<LogicalPlan>,
        items: Vec<ProjectionItem>,
        distinct: bool,
        /// `true` for RETURN — drops every non-projected binding. `false`
        /// for WITH (keeps bindings live so the next clause can reference
        /// them, even if the parser doesn't materialise that yet).
        discard_input_bindings: bool,
    },

    /// Aggregate (group_by + aggregation expressions).
    Aggregate {
        input: Box<LogicalPlan>,
        group_by: Vec<(Expression, String)>,
        aggregations: Vec<(String, AggregateExpr)>,
    },

    /// Sort + skip + limit fused. Pure sort: `skip=Const(0),
    /// limit=Const(u64::MAX)`. Pure limit: `keys.is_empty()`.
    TopN {
        input: Box<LogicalPlan>,
        keys: Vec<OrderKey>,
        skip: RowCount,
        limit: RowCount,
    },

    /// Distinct across every visible binding.
    Distinct { input: Box<LogicalPlan> },

    /// UNION (set) or UNION ALL (multiset).
    Union {
        left: Box<LogicalPlan>,
        right: Box<LogicalPlan>,
        all: bool,
    },

    /// Expand a list expression to one row per element.
    Unwind {
        input: Box<LogicalPlan>,
        list: Expression,
        alias: String,
    },

    /// Single empty driver row — used when the query starts with `UNWIND`
    /// or with a leading `WITH` literal.
    Empty,

    /// Cartesian product of `left` × `right`. Used to combine pattern
    /// parts inside a multi-pattern `MATCH ... , ... ` clause where the
    /// parts share no binding. Naïve nested-loop in v0; hash join when
    /// shared bindings exist arrives.
    CrossProduct {
        left: Box<LogicalPlan>,
        right: Box<LogicalPlan>,
    },

    /// Inner hash equi-join (RFC-012). Build a hash table over
    /// `build`'s rows keyed by each `JoinKey::build_side`; stream
    /// `probe` and look up matches via `JoinKey::probe_side`. Emit one
    /// row per match with bindings from both sides combined.
    ///
    /// `residual` is any non-equi predicate left over from the
    /// pre-conversion Filter (e.g. `a.z >= b.w` in
    /// `WHERE a.x = b.y AND a.z >= b.w`). Evaluated 3VL on the joined
    /// row; False / NULL drop the row.
    HashJoin {
        build: Box<LogicalPlan>,
        probe: Box<LogicalPlan>,
        on: Vec<JoinKey>,
        residual: Option<Expression>,
    },

    /// Decorrelated semi-join (RFC-014). Functionally
    /// equivalent to `SemiApply` but executes the subplan as `inner`
    /// ONCE (not per outer row) and builds a key set; the outer
    /// (`outer`) is probed against the set.
    ///
    /// `outer` bindings flow through to the output; `inner` bindings
    /// are dropped (semi-join semantics — only the existence of a
    /// match matters).
    ///
    /// `negated = true` ⇒ keep outer rows that have NO inner match
    /// (`NOT EXISTS`).
    HashSemiJoin {
        outer: Box<LogicalPlan>,
        inner: Box<LogicalPlan>,
        on: Vec<JoinKey>,
        negated: bool,
        /// Non-equi predicate evaluated on the joined row (outer ∪
        /// build's full row recovered from a side map). Optional —
        /// the rewriter ships `None` in v0.
        residual: Option<Expression>,
    },

    /// Placeholder for "the outer scope provides these bindings". Used
    /// as the leaf of a SemiApply/PatternList subplan when the inner
    /// pattern references variables bound in the outer query. Executor
    /// materialises a single row containing the outer bindings.
    Argument { bindings: Vec<String> },

    /// Existential / anti-existential semi-join. For each row produced
    /// by `input`, evaluate `subplan` parametrised by the row's
    /// bindings; keep the row iff the subplan yields at least one row
    /// (`negated=false`) or zero rows (`negated=true`).
    ///
    /// Lowering rule: `Exists(pattern)` predicates that appear at the
    /// AND-root of a WHERE expression are extracted to a chain of
    /// SemiApply operators; the residual predicates remain in `Filter`.
    SemiApply {
        input: Box<LogicalPlan>,
        subplan: Box<LogicalPlan>,
        negated: bool,
    },

    /// Materialise a pattern comprehension `[(a)-[]->(b) WHERE p | proj]`
    /// into a list-valued binding. For each row from `input`, execute
    /// `subplan` parametrised by the row, evaluate `projection` on each
    /// inner row, and bind the resulting list to `alias` on the outer row.
    PatternList {
        input: Box<LogicalPlan>,
        subplan: Box<LogicalPlan>,
        projection: Expression,
        alias: String,
    },

    // ─── Write-path operators (RFC-009) ────────────────────────
    /// `CREATE (a:Person {name: 'Ada'})-[r:KNOWS]->(b:Person {...})` —
    /// drives `input` (often `Empty` when standalone) and for each row
    /// materialises the listed elements in order. Newly created nodes
    /// and relationships are bound on the row under their aliases.
    Create {
        input: Box<LogicalPlan>,
        elements: Vec<CreateElement>,
    },

    /// `MERGE (pattern) [ON MATCH SET ...] [ON CREATE SET ...]` — try to
    /// MATCH the pattern; if at least one row matches, apply
    /// `on_match_sets`; otherwise CREATE the pattern and apply
    /// `on_create_sets`. The single pattern variant of v0 contains a
    /// single node or one node + one rel + one node chain.
    Merge {
        input: Box<LogicalPlan>,
        pattern: Vec<CreateElement>,
        on_match_sets: Vec<SetOp>,
        on_create_sets: Vec<SetOp>,
    },

    /// `SET a.prop = value` / `SET a = {...}` / `SET a += {...}` /
    /// `SET a:Label`. For each row from `input`, apply every `SetOp`.
    Set {
        input: Box<LogicalPlan>,
        items: Vec<SetOp>,
    },

    /// `REMOVE a.prop` / `REMOVE a:Label`. For each row, drop the named
    /// property or label from the bound node/rel.
    Remove {
        input: Box<LogicalPlan>,
        items: Vec<RemoveOp>,
    },

    /// `DELETE a, b` / `DETACH DELETE a`. For each row, tombstone the
    /// referenced node(s)/rel(s). With `detach=true`, nodes are first
    /// stripped of their incident edges across every edge type known
    /// to the manifest schema.
    Delete {
        input: Box<LogicalPlan>,
        targets: Vec<Expression>,
        detach: bool,
    },

    /// Worst-case optimal multiway join (RFC-024). Emitted by the
    /// `multiway_join` optimiser pass when it detects a cyclic
    /// constraint graph in a subtree of `Expand` / `HashJoin`
    /// operators. The executor walks `ordering` left to right; at each
    /// level it intersects, via leapfrog triejoin, the
    /// `sorted_partners` lists of every `EdgeConstraint` that ties the
    /// current variable to one already bound.
    ///
    /// `MultiwayJoin` is a leaf in [`Self::children`]: the operator
    /// drives its own reads from the snapshot rather than streaming a
    /// `FactorRowSet` through an inner plan. Inherited bindings would
    /// flow through `factorize_required = true` plus the upstream
    /// arena once we land non-leaf MultiwayJoin shapes; the v0 pass
    /// only rewrites contiguous leaf subtrees.
    MultiwayJoin {
        vars: Vec<NodeBinding>,
        edges: Vec<EdgeConstraint>,
        /// Permutation over `vars`. `ordering[0]` is the variable
        /// bound first (outer-most level of the trie); subsequent
        /// entries are the trie descent order.
        ordering: Vec<usize>,
        /// Always `true` in v0; the executor refuses to run when
        /// `NAMIDB_FACTORIZE=0` (the binary plan stays the
        /// `Vec<Row>`-shaped fallback). Reserved for a follow-up
        /// flat-path WCOJ.
        factorize_required: bool,
    },

    /// Direct global count of edges by type, bypassing `NodeScan +
    /// Expand`. Emitted by the `edge_count_pushdown` optimiser pass when
    /// it detects a global `count(*)` / `count(r)` (no GROUP BY) over a
    /// directed, single-hop, unfiltered `Expand` of one or more edge
    /// types whose source is an unfiltered `NodeScan`. The executor sums
    /// [`namidb_storage::read::Snapshot::count_edge_type`] over
    /// `edge_types` (each edge belongs to exactly one type, so the
    /// per-type counts are disjoint) and emits a single row binding
    /// `output` to the total.
    ///
    /// A leaf in [`Self::children`]: it drives its own reads.
    EdgeTypeCount {
        /// One or more edge types to count (alternation `[:A|:B]` lists
        /// every branch). Never empty.
        edge_types: Vec<String>,
        /// Output column the count is bound to — the alias of the
        /// aggregation this pass replaced (`count(r) AS n` ⇒ `"n"`).
        output: String,
    },

    /// Approximate-nearest-neighbour lookup over a node-label embedding
    /// property (RFC-030, `vector-index`). A leaf operator: it drives its own
    /// reads — the flat path (`NodeScan` of `label`, per-row distance, sort,
    /// take `k`) when no backing `VectorIndexDescriptor` exists, or the
    /// DiskANN/Vamana graph when one does. Emits `k` rows binding `alias` to
    /// each hit NodeId and `score_alias` to the (lower-is-closer) distance to
    /// the query. The optimizer rewrites a matching KNN shape into this only
    /// when an index is available; otherwise the flat shape already ranks
    /// correctly, so this variant is produced solely by the `vector_search`
    /// rewrite (feature-gated call site).
    VectorSearch {
        /// `Some(label)` scopes the scan/index to one label; `None` fans out
        /// across every label carrying `property` (flat path only — an index is
        /// per-label, so the rewrite requires a label).
        label: Option<String>,
        /// Binding for the hit node id in each output row.
        alias: String,
        /// Embedding property holding the vector.
        property: String,
        /// Query vector expression (evaluated to a FloatVector at run time).
        query: Expression,
        /// How many nearest rows to emit.
        k: RowCount,
        /// Distance metric / builtin to rank by.
        distance: VectorDistance,
        /// Output column for the (lower-is-closer) distance score.
        score_alias: String,
    },
    /// `CALL <ns>.<name>([args]) [YIELD …]` — invoke a built-in graph
    /// procedure (RFC-008 PR1). A source leaf: no input, yields one row per
    /// procedure output record (e.g. one row per node for `algo.wcc` /
    /// `algo.pagerank`). The executor dispatches on `(namespace, name)`.
    CallProcedure {
        namespace: Option<String>,
        name: String,
        args: Vec<Expression>,
        /// `(source_column, binding_name)` pairs; empty → emit the procedure's
        /// canonical columns verbatim.
        yield_items: Vec<(String, String)>,
    },
}

/// One participating variable in a [`LogicalPlan::MultiwayJoin`]
/// (RFC-024). Carries the alias the executor binds, the optional
/// label that scopes its NodeScan, and any predicates harvested from
/// `Filter` nodes the detection pass folded into the join.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct NodeBinding {
    pub alias: String,
    pub label: Option<String>,
    pub predicates: Vec<ScanPredicate>,
}

/// One edge constraint in a [`LogicalPlan::MultiwayJoin`]
/// (RFC-024). `from_idx` and `to_idx` index into
/// `MultiwayJoin.vars`; for each listed `edge_type` the executor reads
/// `Snapshot::sorted_partners(edge_type, vars[from_idx], direction)`
/// and merge-unions the lists before feeding the result into the
/// leapfrog intersection with the other constraints at this trie level.
///
/// `edge_types` is always non-empty; the detection pass refuses
/// untyped edges (`MATCH (a)-[]-(b)`) because the resulting union
/// would be `O(observed_types * deg)` and the cost model has no way
/// to bound it. Cypher alternation `[:A|:B|:C]` populates the vector
/// with all listed types in source order.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct EdgeConstraint {
    pub from_idx: usize,
    pub to_idx: usize,
    pub edge_types: Vec<String>,
    pub direction: RelationshipDirection,
}

impl LogicalPlan {
    /// Iterate over the direct child plans (0 to 2). Useful for
    /// rewriters and EXPLAIN walkers.
    pub fn children(&self) -> Vec<&LogicalPlan> {
        match self {
            LogicalPlan::NodeScan { .. }
            | LogicalPlan::Empty
            | LogicalPlan::Argument { .. }
            | LogicalPlan::MultiwayJoin { .. }
            | LogicalPlan::EdgeTypeCount { .. }
            | LogicalPlan::VectorSearch { .. }
            | LogicalPlan::CallProcedure { .. } => {
                vec![]
            }
            LogicalPlan::NodeById { input, .. }
            | LogicalPlan::NodeByPropertyValue { input, .. }
            | LogicalPlan::Expand { input, .. }
            | LogicalPlan::Filter { input, .. }
            | LogicalPlan::Project { input, .. }
            | LogicalPlan::Aggregate { input, .. }
            | LogicalPlan::TopN { input, .. }
            | LogicalPlan::Distinct { input }
            | LogicalPlan::Unwind { input, .. } => vec![input.as_ref()],
            LogicalPlan::Union { left, right, .. } | LogicalPlan::CrossProduct { left, right } => {
                vec![left.as_ref(), right.as_ref()]
            }
            LogicalPlan::HashJoin { build, probe, .. } => {
                vec![build.as_ref(), probe.as_ref()]
            }
            LogicalPlan::HashSemiJoin { outer, inner, .. } => {
                vec![outer.as_ref(), inner.as_ref()]
            }
            LogicalPlan::SemiApply { input, subplan, .. } => {
                vec![input.as_ref(), subplan.as_ref()]
            }
            LogicalPlan::PatternList { input, subplan, .. } => {
                vec![input.as_ref(), subplan.as_ref()]
            }
            LogicalPlan::Create { input, .. }
            | LogicalPlan::Merge { input, .. }
            | LogicalPlan::Set { input, .. }
            | LogicalPlan::Remove { input, .. }
            | LogicalPlan::Delete { input, .. } => vec![input.as_ref()],
        }
    }

    /// True if `self` (or any descendant) is a write-side operator.
    /// CLI dispatch and tooling use this to decide between
    /// `execute(plan, &Snapshot, _)` and
    /// `execute_write(plan, &mut WriterSession, _)`.
    pub fn contains_write(&self) -> bool {
        matches!(
            self,
            LogicalPlan::Create { .. }
                | LogicalPlan::Merge { .. }
                | LogicalPlan::Set { .. }
                | LogicalPlan::Remove { .. }
                | LogicalPlan::Delete { .. }
        ) || self.children().iter().any(|c| c.contains_write())
    }

    /// Short operator name used by EXPLAIN headers.
    pub fn operator_name(&self) -> &'static str {
        match self {
            LogicalPlan::NodeScan { .. } => "NodeScan",
            LogicalPlan::NodeById { .. } => "NodeById",
            LogicalPlan::NodeByPropertyValue { .. } => "NodeByPropertyValue",
            LogicalPlan::Expand { optional, .. } => {
                if *optional {
                    "OptionalExpand"
                } else {
                    "Expand"
                }
            }
            LogicalPlan::Filter { .. } => "Filter",
            LogicalPlan::Project { distinct, .. } => {
                if *distinct {
                    "ProjectDistinct"
                } else {
                    "Project"
                }
            }
            LogicalPlan::Aggregate { .. } => "Aggregate",
            LogicalPlan::TopN { .. } => "TopN",
            LogicalPlan::Distinct { .. } => "Distinct",
            LogicalPlan::Union { all, .. } => {
                if *all {
                    "UnionAll"
                } else {
                    "Union"
                }
            }
            LogicalPlan::Unwind { .. } => "Unwind",
            LogicalPlan::Empty => "Empty",
            LogicalPlan::CrossProduct { .. } => "CrossProduct",
            LogicalPlan::HashJoin { .. } => "HashJoin",
            LogicalPlan::HashSemiJoin { negated, .. } => {
                if *negated {
                    "AntiHashSemiJoin"
                } else {
                    "HashSemiJoin"
                }
            }
            LogicalPlan::Argument { .. } => "Argument",
            LogicalPlan::SemiApply { negated, .. } => {
                if *negated {
                    "AntiSemiApply"
                } else {
                    "SemiApply"
                }
            }
            LogicalPlan::PatternList { .. } => "PatternList",
            LogicalPlan::Create { .. } => "Create",
            LogicalPlan::Merge { .. } => "Merge",
            LogicalPlan::Set { .. } => "Set",
            LogicalPlan::Remove { .. } => "Remove",
            LogicalPlan::Delete { detach, .. } => {
                if *detach {
                    "DetachDelete"
                } else {
                    "Delete"
                }
            }
            LogicalPlan::MultiwayJoin { .. } => "MultiwayJoin",
            LogicalPlan::EdgeTypeCount { .. } => "EdgeTypeCount",
            LogicalPlan::VectorSearch { .. } => "VectorSearch",
            LogicalPlan::CallProcedure { .. } => "CallProcedure",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ProjectionItem {
    pub expression: Expression,
    pub alias: String,
}

/// One equi-join pair for [`LogicalPlan::HashJoin`].
///
/// `build_side` is evaluated on each row of the build subtree to
/// compute the hash key; `probe_side` is evaluated on each probe row
/// to look up matches. Each expression references only aliases
/// produced by its respective subtree.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct JoinKey {
    pub build_side: Expression,
    pub probe_side: Expression,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct OrderKey {
    pub expression: Expression,
    pub direction: OrderDirection,
}

/// A `SKIP` / `LIMIT` row count: a plan-time constant, or a `$param`
/// resolved at execution time. The param *name* is part of the plan (not
/// its value), so a cached plan is reused across parameter sets and only
/// the executor reads the bound value. Optimizations that need the concrete
/// count (e.g. limit pushdown) must treat `Param` conservatively as
/// "unknown" via [`RowCount::as_const`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum RowCount {
    Const(u64),
    Param(String),
}

impl RowCount {
    /// The count when it is known at plan time. `Param` returns `None`, so a
    /// caller that needs a numeric bound falls back to a conservative
    /// default (`0` for a skip, "unbounded" for a limit).
    pub fn as_const(&self) -> Option<u64> {
        match self {
            RowCount::Const(n) => Some(*n),
            RowCount::Param(_) => None,
        }
    }
}

/// Aggregate function applied inside an `Aggregate` operator.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum AggregateExpr {
    /// `count(*)` if `arg = None`; `count(expr)` otherwise.
    Count {
        arg: Option<Expression>,
        distinct: bool,
    },
    Sum {
        arg: Expression,
        distinct: bool,
    },
    Avg {
        arg: Expression,
        distinct: bool,
    },
    Min {
        arg: Expression,
    },
    Max {
        arg: Expression,
    },
    Collect {
        arg: Expression,
        distinct: bool,
    },
}

/// One element of a `CREATE` / `MERGE` pattern.
///
/// `Node` introduces a fresh node bound under `alias`; `Rel` connects
/// two already-bound aliases via an edge of the given type and
/// direction. RFC-009 §"Operadores nuevos".
///
/// `properties_spread` carries an optional `$param` reference for the
/// `CREATE (n:L $params)` bulk-insert idiom. When `Some`, the runtime
/// evaluates the expression to a map and merges its entries into the
/// node / edge being created. Explicit `properties` win on key
/// collisions, matching the conventional spread semantics.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum CreateElement {
    Node {
        alias: String,
        /// Full label set to stamp on the new node (`CREATE (n:A:B)`).
        /// Non-empty; written in one `upsert_node_with_labels`.
        labels: Vec<String>,
        properties: Vec<(String, Expression)>,
        properties_spread: Option<Expression>,
    },
    Rel {
        alias: Option<String>,
        edge_type: String,
        source_alias: String,
        target_alias: String,
        direction: RelationshipDirection,
        properties: Vec<(String, Expression)>,
        properties_spread: Option<Expression>,
    },
}

impl CreateElement {
    pub fn alias(&self) -> Option<&str> {
        match self {
            CreateElement::Node { alias, .. } => Some(alias.as_str()),
            CreateElement::Rel { alias, .. } => alias.as_deref(),
        }
    }
}

/// One `SET` operation. RFC-009 §"Operadores nuevos".
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum SetOp {
    Property {
        target_alias: String,
        key: String,
        value: Expression,
    },
    Replace {
        target_alias: String,
        value: Expression,
    },
    Merge {
        target_alias: String,
        value: Expression,
    },
    Labels {
        target_alias: String,
        labels: Vec<String>,
    },
}

impl SetOp {
    pub fn target_alias(&self) -> &str {
        match self {
            SetOp::Property { target_alias, .. }
            | SetOp::Replace { target_alias, .. }
            | SetOp::Merge { target_alias, .. }
            | SetOp::Labels { target_alias, .. } => target_alias,
        }
    }
}

/// One `REMOVE` operation. RFC-009 §"Operadores nuevos".
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum RemoveOp {
    Property {
        target_alias: String,
        key: String,
    },
    Labels {
        target_alias: String,
        labels: Vec<String>,
    },
}

impl AggregateExpr {
    /// Canonical name of the aggregate function (`"count"`, `"sum"`, ...).
    pub fn function_name(&self) -> &'static str {
        match self {
            AggregateExpr::Count { .. } => "count",
            AggregateExpr::Sum { .. } => "sum",
            AggregateExpr::Avg { .. } => "avg",
            AggregateExpr::Min { .. } => "min",
            AggregateExpr::Max { .. } => "max",
            AggregateExpr::Collect { .. } => "collect",
        }
    }

    pub fn distinct(&self) -> bool {
        match self {
            AggregateExpr::Count { distinct, .. }
            | AggregateExpr::Sum { distinct, .. }
            | AggregateExpr::Avg { distinct, .. }
            | AggregateExpr::Collect { distinct, .. } => *distinct,
            AggregateExpr::Min { .. } | AggregateExpr::Max { .. } => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::{ExpressionKind, Identifier, Literal, SourceSpan};

    fn lit(n: i64) -> Expression {
        Expression {
            kind: ExpressionKind::Literal(Literal::Integer(n)),
            span: SourceSpan::point(0),
        }
    }

    #[test]
    fn operator_names_match_explain_expectation() {
        let scan = LogicalPlan::NodeScan {
            label: Some("Person".into()),
            alias: "a".into(),
            predicates: vec![],
            projection: None,
        };
        assert_eq!(scan.operator_name(), "NodeScan");
        assert_eq!(scan.children().len(), 0);
    }

    #[test]
    fn filter_exposes_input_as_child() {
        let scan = LogicalPlan::NodeScan {
            label: Some("Person".into()),
            alias: "a".into(),
            predicates: vec![],
            projection: None,
        };
        let filter = LogicalPlan::Filter {
            input: Box::new(scan.clone()),
            predicate: lit(1),
        };
        assert_eq!(filter.operator_name(), "Filter");
        assert_eq!(filter.children().len(), 1);
        assert_eq!(filter.children()[0], &scan);
    }

    #[test]
    fn union_exposes_both_children() {
        let s = LogicalPlan::NodeScan {
            label: Some("L".into()),
            alias: "a".into(),
            predicates: vec![],
            projection: None,
        };
        let u = LogicalPlan::Union {
            left: Box::new(s.clone()),
            right: Box::new(s.clone()),
            all: true,
        };
        assert_eq!(u.operator_name(), "UnionAll");
        assert_eq!(u.children().len(), 2);
    }

    #[test]
    fn aggregate_expr_helpers() {
        let count_star = AggregateExpr::Count {
            arg: None,
            distinct: false,
        };
        let count_distinct = AggregateExpr::Count {
            arg: Some(lit(1)),
            distinct: true,
        };
        let min = AggregateExpr::Min { arg: lit(2) };
        assert_eq!(count_star.function_name(), "count");
        assert!(!count_star.distinct());
        assert!(count_distinct.distinct());
        assert_eq!(min.function_name(), "min");
        assert!(!min.distinct());
    }

    #[test]
    fn write_operator_names_and_children() {
        let dummy_input = LogicalPlan::Empty;
        let create = LogicalPlan::Create {
            input: Box::new(dummy_input.clone()),
            elements: vec![],
        };
        assert_eq!(create.operator_name(), "Create");
        assert_eq!(create.children().len(), 1);

        let delete = LogicalPlan::Delete {
            input: Box::new(dummy_input.clone()),
            targets: vec![],
            detach: false,
        };
        assert_eq!(delete.operator_name(), "Delete");
        let detach = LogicalPlan::Delete {
            input: Box::new(dummy_input.clone()),
            targets: vec![],
            detach: true,
        };
        assert_eq!(detach.operator_name(), "DetachDelete");

        let set = LogicalPlan::Set {
            input: Box::new(dummy_input.clone()),
            items: vec![],
        };
        assert_eq!(set.operator_name(), "Set");
        assert_eq!(set.children().len(), 1);
    }

    #[test]
    fn create_element_alias_helper() {
        let node = CreateElement::Node {
            alias: "a".into(),
            labels: vec!["Person".into()],
            properties: vec![],
            properties_spread: None,
        };
        assert_eq!(node.alias(), Some("a"));
        let rel = CreateElement::Rel {
            alias: None,
            edge_type: "KNOWS".into(),
            source_alias: "a".into(),
            target_alias: "b".into(),
            direction: crate::parser::RelationshipDirection::Right,
            properties: vec![],
            properties_spread: None,
        };
        assert_eq!(rel.alias(), None);
    }

    #[test]
    fn set_op_target_alias_helper() {
        let p = SetOp::Property {
            target_alias: "a".into(),
            key: "name".into(),
            value: lit(1),
        };
        assert_eq!(p.target_alias(), "a");
        let l = SetOp::Labels {
            target_alias: "b".into(),
            labels: vec!["X".into()],
        };
        assert_eq!(l.target_alias(), "b");
    }

    #[test]
    fn hash_join_exposes_build_and_probe_as_children() {
        let l = LogicalPlan::NodeScan {
            label: Some("Person".into()),
            alias: "a".into(),
            predicates: vec![],
            projection: None,
        };
        let r = LogicalPlan::NodeScan {
            label: Some("Person".into()),
            alias: "b".into(),
            predicates: vec![],
            projection: None,
        };
        let plan = LogicalPlan::HashJoin {
            build: Box::new(l.clone()),
            probe: Box::new(r.clone()),
            on: vec![JoinKey {
                build_side: lit(1),
                probe_side: lit(2),
            }],
            residual: None,
        };
        assert_eq!(plan.operator_name(), "HashJoin");
        assert_eq!(plan.children().len(), 2);
        assert_eq!(plan.children()[0], &l);
        assert_eq!(plan.children()[1], &r);
        assert!(!plan.contains_write());
    }

    #[test]
    fn projection_item_holds_alias() {
        let item = ProjectionItem {
            expression: Expression {
                kind: ExpressionKind::Variable(Identifier::new("a", SourceSpan::point(0))),
                span: SourceSpan::point(0),
            },
            alias: "n".into(),
        };
        assert_eq!(item.alias, "n");
    }
}
