//! Cypher AST types.
//!
//! Every node carries a `span: SourceSpan` so downstream layers (logical
//! plan, executor, EXPLAIN) can reach back to source
//! positions for error reporting.
//!
//! The shapes follow openCypher 9 with the GQL ISO/IEC 39075:2024 naming
//! where the two diverge. The v0 subset is declared in RFC-004.

use serde::{Deserialize, Serialize};

use super::error::SourceSpan;

// ────────────────────────────────────────────────────────────────────
// Top-level
// ────────────────────────────────────────────────────────────────────

/// A whole query — one or more single-queries joined by `UNION` / `UNION ALL`.
///
/// - `explain`: user wrote `EXPLAIN <query>`. The executor honours it by
/// returning the plan tree instead of executing.
/// - `explain_verbose`: user wrote `EXPLAIN VERBOSE <query>`. Implies
/// `explain`. The plan is rendered with per-operator cardinality
/// estimates from the [`StatsCatalog`] (RFC-010).
///
/// [`StatsCatalog`]: crate::cost::StatsCatalog
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Query {
    pub head: SingleQuery,
    pub tail: Vec<UnionPart>,
    pub explain: bool,
    pub explain_verbose: bool,
    /// `EXPLAIN RAW [VERBOSE]` — skips the optimizer pipeline and
    /// renders the plan exactly as the lowering produced it. Useful for
    /// debugging the lowering and for verifying that the optimizer did
    /// something (RFC-011 §6.2).
    pub explain_raw: bool,
    pub span: SourceSpan,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct UnionPart {
    pub all: bool,
    pub query: SingleQuery,
    pub span: SourceSpan,
}

/// A linear sequence of clauses sharing one scope chain.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SingleQuery {
    pub clauses: Vec<Clause>,
    pub span: SourceSpan,
}

// ────────────────────────────────────────────────────────────────────
// Clauses
// ────────────────────────────────────────────────────────────────────

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum Clause {
    Match(MatchClause),
    Return(ReturnClause),
    With(WithClause),
    Where(WhereClause),
    Unwind(UnwindClause),
    Create(CreateClause),
    Merge(MergeClause),
    Set(SetClause),
    Remove(RemoveClause),
    Delete(DeleteClause),
    /// `CREATE VECTOR INDEX` — schema DDL (RFC-030). Always present as a
    /// variant (per the codebase convention); the parse arm and the
    /// out-of-band server execution are gated behind the `vector-index`
    /// Cargo feature. Unlike the read/write clauses it never lowers to a
    /// `LogicalPlan` — the server intercepts it via
    /// [`Query::as_create_vector_index`].
    CreateVectorIndex(CreateVectorIndexClause),
    /// `CREATE FULLTEXT INDEX` — schema DDL for the persistent BM25 index.
    /// Always present as a variant; the parse arm and the out-of-band server
    /// execution are gated behind the `text-index` Cargo feature. Like
    /// `CreateVectorIndex` it never lowers to a `LogicalPlan` — the server
    /// intercepts it via [`Query::as_create_fulltext_index`].
    CreateFulltextIndex(CreateFulltextIndexClause),
    /// `CREATE CONSTRAINT … IS UNIQUE` — schema DDL declaring a uniqueness
    /// constraint. Always a variant; intercepted out-of-band by the server via
    /// [`Query::as_create_constraint`] (never lowered).
    CreateConstraint(CreateConstraintClause),
    /// `CREATE INDEX … ON …` — schema DDL declaring a secondary (equality)
    /// index. Intercepted via [`Query::as_create_index`].
    CreateIndex(CreateIndexClause),
    /// `CALL <ns>.<name>([args]) [YIELD …]` — invoke a built-in procedure
    /// (RFC-008 PR1). A leading source clause: it introduces bindings (the
    /// YIELD columns) like `MATCH` does. Always-on (graph algorithms are
    /// core, not experimental), so unlike `CreateVectorIndex` this one
    /// lowers to a `LogicalPlan::CallProcedure` and the executor runs it.
    Call(CallClause),
}

impl Clause {
    pub fn span(&self) -> SourceSpan {
        match self {
            Clause::Match(c) => c.span,
            Clause::Return(c) => c.span,
            Clause::With(c) => c.span,
            Clause::Where(c) => c.span,
            Clause::Unwind(c) => c.span,
            Clause::Create(c) => c.span,
            Clause::Merge(c) => c.span,
            Clause::Set(c) => c.span,
            Clause::Remove(c) => c.span,
            Clause::Delete(c) => c.span,
            Clause::CreateVectorIndex(c) => c.span,
            Clause::CreateFulltextIndex(c) => c.span,
            Clause::CreateConstraint(c) => c.span,
            Clause::CreateIndex(c) => c.span,
            Clause::Call(c) => c.span,
        }
    }
}

impl Query {
    /// If this query is a standalone `CREATE VECTOR INDEX` statement — the
    /// sole clause, no `UNION`, no `EXPLAIN` prefix — return it.
    ///
    /// `CREATE VECTOR INDEX` is schema DDL, not a read or a row write, so the
    /// server intercepts it *before* planning (it never builds a
    /// `LogicalPlan`). This accessor is the hook that lets the server tell a
    /// DDL query apart from an ordinary one without re-walking the AST. Any
    /// other shape (a `RETURN` after it, a `UNION`, an `EXPLAIN` prefix)
    /// falls through to the normal plan path, where the lowerer rejects it.
    pub fn as_create_vector_index(&self) -> Option<&CreateVectorIndexClause> {
        if self.tail.is_empty() && !self.explain && self.head.clauses.len() == 1 {
            if let Clause::CreateVectorIndex(c) = &self.head.clauses[0] {
                return Some(c);
            }
        }
        None
    }

    /// `CREATE FULLTEXT INDEX` interception hook (mirrors
    /// [`as_create_vector_index`](Self::as_create_vector_index)): `Some` only
    /// when the DDL is the sole statement (no `RETURN`/`UNION`/`EXPLAIN`).
    pub fn as_create_fulltext_index(&self) -> Option<&CreateFulltextIndexClause> {
        if self.tail.is_empty() && !self.explain && self.head.clauses.len() == 1 {
            if let Clause::CreateFulltextIndex(c) = &self.head.clauses[0] {
                return Some(c);
            }
        }
        None
    }

    /// `CREATE CONSTRAINT … IS UNIQUE` interception hook: `Some` only when the
    /// DDL is the sole statement.
    pub fn as_create_constraint(&self) -> Option<&CreateConstraintClause> {
        if self.tail.is_empty() && !self.explain && self.head.clauses.len() == 1 {
            if let Clause::CreateConstraint(c) = &self.head.clauses[0] {
                return Some(c);
            }
        }
        None
    }

    /// `CREATE INDEX … ON …` interception hook: `Some` only when the DDL is the
    /// sole statement.
    pub fn as_create_index(&self) -> Option<&CreateIndexClause> {
        if self.tail.is_empty() && !self.explain && self.head.clauses.len() == 1 {
            if let Clause::CreateIndex(c) = &self.head.clauses[0] {
                return Some(c);
            }
        }
        None
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MatchClause {
    pub optional: bool,
    pub patterns: Vec<PatternPart>,
    pub where_: Option<Expression>,
    pub span: SourceSpan,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ReturnClause {
    pub distinct: bool,
    pub items: Vec<ProjectionItem>,
    pub order_by: Vec<OrderItem>,
    pub skip: Option<Expression>,
    pub limit: Option<Expression>,
    pub span: SourceSpan,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct WithClause {
    pub distinct: bool,
    pub items: Vec<ProjectionItem>,
    pub order_by: Vec<OrderItem>,
    pub skip: Option<Expression>,
    pub limit: Option<Expression>,
    pub where_: Option<Expression>,
    pub span: SourceSpan,
}

/// Free-standing `WHERE` is illegal in Cypher — `WHERE` lives inside `MATCH`
/// or `WITH`. We keep `WhereClause` only to centralise the lowered form once
/// hits; the parser never emits it directly.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct WhereClause {
    pub predicate: Expression,
    pub span: SourceSpan,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct UnwindClause {
    pub list: Expression,
    pub alias: Identifier,
    pub span: SourceSpan,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CreateClause {
    pub patterns: Vec<PatternPart>,
    pub span: SourceSpan,
}

/// Distance metric named in `CREATE VECTOR INDEX ... METRIC <m>` (RFC-030).
///
/// This is the parser's own vocabulary so the AST stays free of storage
/// types; the server converts it to `namidb_storage::manifest::VectorMetric`
/// when it builds the [`VectorIndexDescriptor`]. The three values mirror the
/// vector-distance builtins (`cosine_similarity`, `dot_product`,
/// `euclidean_distance`).
///
/// [`VectorIndexDescriptor`]: namidb_storage::manifest::VectorIndexDescriptor
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum VectorMetric {
    Cosine,
    Dot,
    Euclidean,
}

impl VectorMetric {
    /// `cosine` / `dot` / `euclidean` (case-insensitive) → the metric, else `None`.
    pub fn from_keyword(word: &str) -> Option<Self> {
        match word.to_ascii_lowercase().as_str() {
            "cosine" => Some(VectorMetric::Cosine),
            "dot" => Some(VectorMetric::Dot),
            "euclidean" => Some(VectorMetric::Euclidean),
            _ => None,
        }
    }

    /// Canonical source spelling (for [`fmt::Display`](std::fmt::Display)).
    pub fn as_keyword(self) -> &'static str {
        match self {
            VectorMetric::Cosine => "cosine",
            VectorMetric::Dot => "dot",
            VectorMetric::Euclidean => "euclidean",
        }
    }
}

/// `CREATE VECTOR INDEX <name> FOR (<alias>:<Label>) ON <alias>.<property>
/// METRIC <m> DIMENSION <n> [WITH {r, l_build, alpha}]` (RFC-030).
///
/// A standalone schema command: the parser only emits it as the sole clause
/// of a query, and the server executes it out-of-band (see
/// [`Query::as_create_vector_index`]) — it never becomes a `LogicalPlan`. The
/// Vamana build parameters (`r`, `l_build`, `alpha`) are optional overrides;
/// `None` means "use the engine defaults".
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CreateVectorIndexClause {
    pub name: Identifier,
    pub label: Identifier,
    pub property: Identifier,
    pub dim: u32,
    pub metric: VectorMetric,
    /// Optional `WITH {r: …}` override of the Vamana max out-degree.
    pub r: Option<usize>,
    /// Optional `WITH {l_build: …}` override of the Vamana build beam.
    pub l_build: Option<usize>,
    /// Optional `WITH {alpha: …}` override of the Vamana diversification.
    pub alpha: Option<f32>,
    pub span: SourceSpan,
}

/// `CREATE FULLTEXT INDEX <name> ON :<Label>(<prop1>[, <prop2>, …])`.
///
/// A standalone schema command for the persistent BM25 index: the parser only
/// emits it as the sole clause of a query, and the server executes it
/// out-of-band (see [`Query::as_create_fulltext_index`]) — it never becomes a
/// `LogicalPlan`. The listed properties are concatenated per document at build
/// time; order does not affect BM25 scores.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CreateFulltextIndexClause {
    pub name: Identifier,
    pub label: Identifier,
    pub properties: Vec<Identifier>,
    pub span: SourceSpan,
}

/// `CREATE CONSTRAINT [name] FOR (n:Label) REQUIRE n.prop IS UNIQUE` (and the
/// legacy `ON (n:Label) ASSERT …` form). A standalone schema command executed
/// out-of-band by the server (see [`Query::as_create_constraint`]).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CreateConstraintClause {
    pub name: Option<Identifier>,
    pub label: Identifier,
    pub property: Identifier,
    pub span: SourceSpan,
}

/// `CREATE INDEX [name] FOR (n:Label) ON (n.prop)` (and the legacy
/// `ON :Label(prop)` form). A standalone schema command executed out-of-band by
/// the server (see [`Query::as_create_index`]).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CreateIndexClause {
    pub name: Option<Identifier>,
    pub label: Identifier,
    pub property: Identifier,
    pub span: SourceSpan,
}

/// `CALL <namespace>.<name>([args]) [YIELD <items>]` (RFC-008 PR1) — invoke a
/// built-in procedure (`algo.wcc`, `algo.pagerank`, …) and yield its result
/// rows. A leading source clause: it introduces bindings (the YIELD columns)
/// like `MATCH`. `YIELD` is optional — without it the procedure's canonical
/// output columns are emitted.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CallClause {
    /// Procedure namespace (before the dot), e.g. `algo`. `None` for an
    /// unqualified call.
    pub namespace: Option<String>,
    /// Procedure name (after the dot, or the whole call).
    pub name: String,
    /// Positional argument expressions (may be empty).
    pub args: Vec<Expression>,
    /// `YIELD` projection items; empty when `YIELD` was omitted.
    pub yield_items: Vec<YieldItem>,
    pub span: SourceSpan,
}

/// One `YIELD` column: a procedure output name, optionally renamed with `AS`.
/// Unlike a projection item, YIELD columns are plain names (no expressions).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct YieldItem {
    /// The procedure's output column name being selected.
    pub name: Identifier,
    /// Optional `AS <alias>` rename.
    pub alias: Option<Identifier>,
    pub span: SourceSpan,
}

impl YieldItem {
    /// The binding name this item produces in downstream scope: the alias if
    /// given, else the column name itself.
    pub fn binding_name(&self) -> &str {
        self.alias
            .as_ref()
            .map(|a| a.name.as_str())
            .unwrap_or(&self.name.name)
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MergeClause {
    pub pattern: PatternPart,
    pub actions: Vec<MergeAction>,
    pub span: SourceSpan,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MergeAction {
    pub on: MergeTrigger,
    pub sets: Vec<SetItem>,
    pub span: SourceSpan,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum MergeTrigger {
    Match,
    Create,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SetClause {
    pub items: Vec<SetItem>,
    pub span: SourceSpan,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum SetItem {
    /// `a.prop = value`
    Property {
        target: PropertyAccess,
        value: Expression,
        span: SourceSpan,
    },
    /// `a = {prop: value, ...}` — replace all properties.
    Replace {
        target: Identifier,
        value: Expression,
        span: SourceSpan,
    },
    /// `a += {prop: value, ...}` — merge.
    Merge {
        target: Identifier,
        value: Expression,
        span: SourceSpan,
    },
    /// `a:Label[:Label...]` — add labels.
    Labels {
        target: Identifier,
        labels: Vec<Identifier>,
        span: SourceSpan,
    },
}

impl SetItem {
    pub fn span(&self) -> SourceSpan {
        match self {
            SetItem::Property { span, .. }
            | SetItem::Replace { span, .. }
            | SetItem::Merge { span, .. }
            | SetItem::Labels { span, .. } => *span,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RemoveClause {
    pub items: Vec<RemoveItem>,
    pub span: SourceSpan,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum RemoveItem {
    Property(PropertyAccess),
    Labels {
        target: Identifier,
        labels: Vec<Identifier>,
        span: SourceSpan,
    },
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DeleteClause {
    pub detach: bool,
    pub targets: Vec<Expression>,
    pub span: SourceSpan,
}

// ────────────────────────────────────────────────────────────────────
// Projection
// ────────────────────────────────────────────────────────────────────

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ProjectionItem {
    pub expression: Expression,
    pub alias: Option<Identifier>,
    pub span: SourceSpan,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct OrderItem {
    pub expression: Expression,
    pub direction: OrderDirection,
    pub span: SourceSpan,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum OrderDirection {
    Asc,
    Desc,
}

// ────────────────────────────────────────────────────────────────────
// Patterns
// ────────────────────────────────────────────────────────────────────

/// A pattern part is one chain `(a)-[r]->(b)-[s]->(c) ...` optionally bound
/// to a path variable: `p = (a)-[r]->(b)`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PatternPart {
    pub binding: Option<Identifier>,
    pub element: PatternElement,
    pub span: SourceSpan,
    /// `Some(...)` when the pattern part is wrapped in
    /// `shortestPath(...)` or `allShortestPaths(...)`. Lower
    /// translates this into the `shortest` field on
    /// [`crate::plan::LogicalPlan::Expand`]. See RFC-023.
    pub shortest_path: Option<ShortestPathMode>,
}

/// Shortest-path variant a [`PatternPart`] was wrapped in.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ShortestPathMode {
    /// `shortestPath(...)` — one path per (source, target) pair.
    First,
    /// `allShortestPaths(...)` — every path of the minimum length.
    All,
}

/// A pattern element starts with a node and alternates relationship→node.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PatternElement {
    pub head: NodePattern,
    pub chain: Vec<(RelationshipPattern, NodePattern)>,
    pub span: SourceSpan,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct NodePattern {
    pub binding: Option<Identifier>,
    pub labels: Vec<Identifier>,
    pub properties: Option<PatternProperties>,
    pub span: SourceSpan,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RelationshipPattern {
    pub direction: RelationshipDirection,
    pub binding: Option<Identifier>,
    /// Type alternation, e.g. `:KNOWS|:LIKES`. Empty = no type filter.
    pub types: Vec<Identifier>,
    pub length: Option<RelationshipLength>,
    pub properties: Option<PatternProperties>,
    pub span: SourceSpan,
}

/// Properties section of a [`NodePattern`] / [`RelationshipPattern`].
///
/// The classic Cypher form is an inline map: `(n:Person {name: 'a'})`.
/// We also accept a single `$param` reference, e.g. `(n:Person $props)`,
/// which is the standard bulk-insert idiom. The runtime is expected to
/// supply a map value for the parameter; at lower time we cannot know
/// the keys, so the executor expands the map into properties when it
/// sees the parameter spread.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum PatternProperties {
    Literal(MapLiteral),
    Parameter { name: String, span: SourceSpan },
}

impl PatternProperties {
    pub fn span(&self) -> SourceSpan {
        match self {
            PatternProperties::Literal(m) => m.span,
            PatternProperties::Parameter { span, .. } => *span,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum RelationshipDirection {
    /// `-->`
    Right,
    /// `<--`
    Left,
    /// `--`
    Both,
}

/// `*1..3` — variable-length range. Bounds are inclusive; both required by
/// RFC-004 (no unbounded `*` or `*1..`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RelationshipLength {
    pub min: u32,
    pub max: u32,
}

// ────────────────────────────────────────────────────────────────────
// Expressions
// ────────────────────────────────────────────────────────────────────

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Expression {
    pub kind: ExpressionKind,
    pub span: SourceSpan,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum ExpressionKind {
    Literal(Literal),
    Variable(Identifier),
    Parameter(String),
    Property(Box<PropertyAccess>),
    Index {
        target: Box<Expression>,
        index: Box<Expression>,
    },
    Range {
        target: Box<Expression>,
        from: Option<Box<Expression>>,
        to: Option<Box<Expression>>,
    },
    Unary {
        op: UnaryOp,
        expr: Box<Expression>,
    },
    Binary {
        op: BinaryOp,
        left: Box<Expression>,
        right: Box<Expression>,
    },
    In {
        item: Box<Expression>,
        list: Box<Expression>,
    },
    StringTest {
        op: StringOp,
        target: Box<Expression>,
        pattern: Box<Expression>,
    },
    IsNull {
        expr: Box<Expression>,
        negated: bool,
    },
    FunctionCall {
        name: QualifiedName,
        args: Vec<Expression>,
        distinct: bool,
    },
    Case {
        scrutinee: Option<Box<Expression>>,
        branches: Vec<CaseBranch>,
        otherwise: Option<Box<Expression>>,
    },
    Exists(Box<PatternElement>),
    List(Vec<Expression>),
    Map(MapLiteral),
    ListComprehension(Box<ListComprehension>),
    PatternComprehension(Box<PatternComprehension>),
    /// List quantifier predicate: `all(x IN list WHERE pred)` and its
    /// `any`/`none`/`single` siblings. Returns a boolean.
    Quantifier(Box<Quantifier>),
    /// `*` projection placeholder. Reserved — RFC-004 §Open question Q1.
    Star,
}

/// Which list quantifier a [`Quantifier`] expresses.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum QuantifierKind {
    /// `all(x IN list WHERE pred)` — true iff every element satisfies `pred`.
    All,
    /// `any(x IN list WHERE pred)` — true iff at least one does.
    Any,
    /// `none(x IN list WHERE pred)` — true iff no element does.
    None,
    /// `single(x IN list WHERE pred)` — true iff exactly one does.
    Single,
}

/// `<kind>(<variable> IN <list> WHERE <predicate>)`. Binds `variable` over each
/// element of `list` and evaluates `predicate`, aggregating per [`QuantifierKind`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Quantifier {
    pub kind: QuantifierKind,
    pub variable: Identifier,
    pub list: Expression,
    pub predicate: Expression,
    pub span: SourceSpan,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PropertyAccess {
    pub target: Expression,
    pub key: Identifier,
    pub span: SourceSpan,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CaseBranch {
    pub when: Expression,
    pub then: Expression,
    pub span: SourceSpan,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ListComprehension {
    pub variable: Identifier,
    pub list: Expression,
    pub predicate: Option<Expression>,
    pub projection: Option<Expression>,
    pub span: SourceSpan,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PatternComprehension {
    pub binding: Option<Identifier>,
    pub pattern: PatternElement,
    pub predicate: Option<Expression>,
    pub projection: Expression,
    pub span: SourceSpan,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum Literal {
    Integer(i64),
    Float(f64),
    String(String),
    Boolean(bool),
    Null,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MapLiteral {
    pub entries: Vec<(Identifier, Expression)>,
    pub span: SourceSpan,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum UnaryOp {
    Neg,
    Not,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    Pow,
    And,
    Or,
    Xor,
    Eq,
    Ne,
    Lt,
    Gt,
    Le,
    Ge,
    RegexMatch,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum StringOp {
    StartsWith,
    EndsWith,
    Contains,
}

// ────────────────────────────────────────────────────────────────────
// Identifier
// ────────────────────────────────────────────────────────────────────

#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Identifier {
    pub name: String,
    pub span: SourceSpan,
    /// True iff the source text used backticks.
    pub quoted: bool,
}

impl Identifier {
    pub fn new(name: impl Into<String>, span: SourceSpan) -> Self {
        Self {
            name: name.into(),
            span,
            quoted: false,
        }
    }

    pub fn quoted(name: impl Into<String>, span: SourceSpan) -> Self {
        Self {
            name: name.into(),
            span,
            quoted: true,
        }
    }
}

/// A name with optional namespace, e.g. `count`, `date.truncate`. Used for
/// function calls. v0 keeps the namespace open-ended; the grammar only
/// recognises `name` and `name.name`.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct QualifiedName {
    pub segments: Vec<Identifier>,
    pub span: SourceSpan,
}

impl QualifiedName {
    pub fn single(id: Identifier) -> Self {
        let span = id.span;
        Self {
            segments: vec![id],
            span,
        }
    }

    /// Joins the segments into the source representation `a.b.c`.
    pub fn joined(&self) -> String {
        self.segments
            .iter()
            .map(|s| s.name.as_str())
            .collect::<Vec<_>>()
            .join(".")
    }
}
