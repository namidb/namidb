//! Cypher AST types.
//!
//! Every node carries a `span: SourceSpan` so downstream layers (logical
//! plan, executor, EXPLAIN) can reach back to source
//! positions for error reporting.
//!
//! The shapes follow openCypher 9 with the GQL ISO/IEC 39075:2024 naming
//! where the two diverge. The v0 subset is declared in RFC-004.

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
#[derive(Clone, Debug, PartialEq)]
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

#[derive(Clone, Debug, PartialEq)]
pub struct UnionPart {
    pub all: bool,
    pub query: SingleQuery,
    pub span: SourceSpan,
}

/// A linear sequence of clauses sharing one scope chain.
#[derive(Clone, Debug, PartialEq)]
pub struct SingleQuery {
    pub clauses: Vec<Clause>,
    pub span: SourceSpan,
}

// ────────────────────────────────────────────────────────────────────
// Clauses
// ────────────────────────────────────────────────────────────────────

#[derive(Clone, Debug, PartialEq)]
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
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct MatchClause {
    pub optional: bool,
    pub patterns: Vec<PatternPart>,
    pub where_: Option<Expression>,
    pub span: SourceSpan,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ReturnClause {
    pub distinct: bool,
    pub items: Vec<ProjectionItem>,
    pub order_by: Vec<OrderItem>,
    pub skip: Option<Expression>,
    pub limit: Option<Expression>,
    pub span: SourceSpan,
}

#[derive(Clone, Debug, PartialEq)]
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
#[derive(Clone, Debug, PartialEq)]
pub struct WhereClause {
    pub predicate: Expression,
    pub span: SourceSpan,
}

#[derive(Clone, Debug, PartialEq)]
pub struct UnwindClause {
    pub list: Expression,
    pub alias: Identifier,
    pub span: SourceSpan,
}

#[derive(Clone, Debug, PartialEq)]
pub struct CreateClause {
    pub patterns: Vec<PatternPart>,
    pub span: SourceSpan,
}

#[derive(Clone, Debug, PartialEq)]
pub struct MergeClause {
    pub pattern: PatternPart,
    pub actions: Vec<MergeAction>,
    pub span: SourceSpan,
}

#[derive(Clone, Debug, PartialEq)]
pub struct MergeAction {
    pub on: MergeTrigger,
    pub sets: Vec<SetItem>,
    pub span: SourceSpan,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MergeTrigger {
    Match,
    Create,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SetClause {
    pub items: Vec<SetItem>,
    pub span: SourceSpan,
}

#[derive(Clone, Debug, PartialEq)]
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

#[derive(Clone, Debug, PartialEq)]
pub struct RemoveClause {
    pub items: Vec<RemoveItem>,
    pub span: SourceSpan,
}

#[derive(Clone, Debug, PartialEq)]
pub enum RemoveItem {
    Property(PropertyAccess),
    Labels {
        target: Identifier,
        labels: Vec<Identifier>,
        span: SourceSpan,
    },
}

#[derive(Clone, Debug, PartialEq)]
pub struct DeleteClause {
    pub detach: bool,
    pub targets: Vec<Expression>,
    pub span: SourceSpan,
}

// ────────────────────────────────────────────────────────────────────
// Projection
// ────────────────────────────────────────────────────────────────────

#[derive(Clone, Debug, PartialEq)]
pub struct ProjectionItem {
    pub expression: Expression,
    pub alias: Option<Identifier>,
    pub span: SourceSpan,
}

#[derive(Clone, Debug, PartialEq)]
pub struct OrderItem {
    pub expression: Expression,
    pub direction: OrderDirection,
    pub span: SourceSpan,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OrderDirection {
    Asc,
    Desc,
}

// ────────────────────────────────────────────────────────────────────
// Patterns
// ────────────────────────────────────────────────────────────────────

/// A pattern part is one chain `(a)-[r]->(b)-[s]->(c) ...` optionally bound
/// to a path variable: `p = (a)-[r]->(b)`.
#[derive(Clone, Debug, PartialEq)]
pub struct PatternPart {
    pub binding: Option<Identifier>,
    pub element: PatternElement,
    pub span: SourceSpan,
}

/// A pattern element starts with a node and alternates relationship→node.
#[derive(Clone, Debug, PartialEq)]
pub struct PatternElement {
    pub head: NodePattern,
    pub chain: Vec<(RelationshipPattern, NodePattern)>,
    pub span: SourceSpan,
}

#[derive(Clone, Debug, PartialEq)]
pub struct NodePattern {
    pub binding: Option<Identifier>,
    pub labels: Vec<Identifier>,
    pub properties: Option<MapLiteral>,
    pub span: SourceSpan,
}

#[derive(Clone, Debug, PartialEq)]
pub struct RelationshipPattern {
    pub direction: RelationshipDirection,
    pub binding: Option<Identifier>,
    /// Type alternation, e.g. `:KNOWS|:LIKES`. Empty = no type filter.
    pub types: Vec<Identifier>,
    pub length: Option<RelationshipLength>,
    pub properties: Option<MapLiteral>,
    pub span: SourceSpan,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
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
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RelationshipLength {
    pub min: u32,
    pub max: u32,
}

// ────────────────────────────────────────────────────────────────────
// Expressions
// ────────────────────────────────────────────────────────────────────

#[derive(Clone, Debug, PartialEq)]
pub struct Expression {
    pub kind: ExpressionKind,
    pub span: SourceSpan,
}

#[derive(Clone, Debug, PartialEq)]
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
    Between {
        target: Box<Expression>,
        low: Box<Expression>,
        high: Box<Expression>,
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
    /// `*` projection placeholder. Reserved — RFC-004 §Open question Q1.
    Star,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PropertyAccess {
    pub target: Expression,
    pub key: Identifier,
    pub span: SourceSpan,
}

#[derive(Clone, Debug, PartialEq)]
pub struct CaseBranch {
    pub when: Expression,
    pub then: Expression,
    pub span: SourceSpan,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ListComprehension {
    pub variable: Identifier,
    pub list: Expression,
    pub predicate: Option<Expression>,
    pub projection: Option<Expression>,
    pub span: SourceSpan,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PatternComprehension {
    pub binding: Option<Identifier>,
    pub pattern: PatternElement,
    pub predicate: Option<Expression>,
    pub projection: Expression,
    pub span: SourceSpan,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Literal {
    Integer(i64),
    Float(f64),
    String(String),
    Boolean(bool),
    Null,
}

#[derive(Clone, Debug, PartialEq)]
pub struct MapLiteral {
    pub entries: Vec<(Identifier, Expression)>,
    pub span: SourceSpan,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UnaryOp {
    Neg,
    Not,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StringOp {
    StartsWith,
    EndsWith,
    Contains,
}

// ────────────────────────────────────────────────────────────────────
// Identifier
// ────────────────────────────────────────────────────────────────────

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
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
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
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
