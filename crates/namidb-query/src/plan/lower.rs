//! Lowering AST → [`LogicalPlan`].
//!
//! See RFC-008 §"Lowering rules".
//!
//! Scope tracking is minimal: we keep a `BTreeSet<String>` of
//! visible bindings and emit `LowerError::BindingNotFound` when an
//! expression references an unknown name. A full type-checking semantic
//! analyzer arrives alongside aggregate detection / WITH `*`.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

use super::logical::{
    AggregateExpr, CreateElement, LogicalPlan, OrderKey, ProjectionItem, RemoveOp, RowCount, SetOp,
    ShortestMode,
};
use crate::parser::{
    self as ast, BinaryOp, Clause, Expression, ExpressionKind, Literal, MatchClause, NodePattern,
    PatternElement, PatternPart, PatternProperties, ProjectionItem as AstProjectionItem,
    QualifiedName, Query, RelationshipPattern, ReturnClause, SingleQuery, SourceSpan, UnaryOp,
    UnwindClause, WithClause,
};

/// Error returned by [`lower`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LowerError {
    pub kind: LowerErrorKind,
    pub message: String,
    pub span: SourceSpan,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LowerErrorKind {
    /// Reference to a binding that is not in scope.
    BindingNotFound,
    /// Binding declared twice in the same scope.
    DoubleBinding,
    /// Feature recognised by the parser but not yet implementable.
    UnsupportedFeature,
    /// Pattern shape the lowerer cannot translate yet.
    InvalidPattern,
    /// Top-level union mixed with non-union clauses in unsupported way.
    InvalidUnion,
}

impl LowerError {
    fn new(kind: LowerErrorKind, msg: impl Into<String>, span: SourceSpan) -> Self {
        Self {
            kind,
            message: msg.into(),
            span,
        }
    }
}

impl fmt::Display for LowerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}: {} at {}", self.kind, self.message, self.span)
    }
}

impl std::error::Error for LowerError {}

/// Lower a parsed Query into a `LogicalPlan` tree.
pub fn lower(query: &Query) -> Result<LogicalPlan, LowerError> {
    let head = lower_single_query(&query.head)?;
    if query.tail.is_empty() {
        return Ok(head);
    }
    let mut acc = head;
    for part in &query.tail {
        let right = lower_single_query(&part.query)?;
        acc = LogicalPlan::Union {
            left: Box::new(acc),
            right: Box::new(right),
            all: part.all,
        };
    }
    if query.tail.iter().any(|p| !p.all) {
        // UNION (non-ALL) implies a final Distinct over the combined output.
        acc = LogicalPlan::Distinct {
            input: Box::new(acc),
        };
    }
    Ok(acc)
}

fn lower_single_query(query: &SingleQuery) -> Result<LogicalPlan, LowerError> {
    let mut ctx = LowerCtx::new();
    let mut plan: Option<LogicalPlan> = None;

    for clause in &query.clauses {
        match clause {
            Clause::Match(m) => {
                let p = lower_match(m, plan, &mut ctx)?;
                plan = Some(p);
            }
            Clause::Where(_) => {
                return Err(LowerError::new(
                    LowerErrorKind::InvalidPattern,
                    "free-standing WHERE clause should be attached to MATCH or WITH",
                    clause.span(),
                ));
            }
            Clause::Return(r) => {
                let base = plan.take().unwrap_or(LogicalPlan::Empty);
                plan = Some(lower_return(r, base, &mut ctx)?);
            }
            Clause::With(w) => {
                let base = plan.take().unwrap_or(LogicalPlan::Empty);
                plan = Some(lower_with(w, base, &mut ctx)?);
            }
            Clause::Unwind(u) => {
                let base = plan.take().unwrap_or(LogicalPlan::Empty);
                plan = Some(lower_unwind(u, base, &mut ctx)?);
            }
            Clause::Create(c) => {
                let base = plan.take().unwrap_or(LogicalPlan::Empty);
                plan = Some(lower_create(c, base, &mut ctx)?);
            }
            Clause::Merge(m) => {
                let base = plan.take().unwrap_or(LogicalPlan::Empty);
                plan = Some(lower_merge(m, base, &mut ctx)?);
            }
            Clause::Set(s) => {
                let base = require_input(plan.take(), clause.span())?;
                plan = Some(lower_set(s, base, &mut ctx)?);
            }
            Clause::Remove(r) => {
                let base = require_input(plan.take(), clause.span())?;
                plan = Some(lower_remove(r, base, &mut ctx)?);
            }
            Clause::Delete(d) => {
                let base = require_input(plan.take(), clause.span())?;
                plan = Some(lower_delete(d, base, &mut ctx)?);
            }
        }
    }

    plan.ok_or_else(|| {
        LowerError::new(
            LowerErrorKind::InvalidPattern,
            "query produced no operators",
            query.span,
        )
    })
}

fn require_input(p: Option<LogicalPlan>, span: SourceSpan) -> Result<LogicalPlan, LowerError> {
    p.ok_or_else(|| {
        LowerError::new(
            LowerErrorKind::InvalidPattern,
            "this clause requires a preceding MATCH/WITH/UNWIND",
            span,
        )
    })
}

// ─────────────────────────── lowering ctx ────────────────────────────

struct LowerCtx {
    bindings: BTreeSet<String>,
    /// Monotonic source of internal binding names for anonymous path
    /// elements (see [`LowerCtx::fill_anonymous_path_bindings`]). Never
    /// reset, so names stay unique across clauses.
    anon_counter: usize,
}

impl LowerCtx {
    fn new() -> Self {
        Self {
            bindings: BTreeSet::new(),
            anon_counter: 0,
        }
    }

    /// A fresh internal binding name. The leading space can never appear
    /// in a user-written identifier, so it cannot collide with a real
    /// binding (and reads as internal if it ever surfaces).
    fn fresh_anon(&mut self, span: SourceSpan) -> ast::Identifier {
        let n = self.anon_counter;
        self.anon_counter += 1;
        ast::Identifier::new(format!(" anon{n}"), span)
    }

    /// Return a copy of `elem` with every anonymous head node,
    /// relationship, and target node given a fresh internal binding.
    ///
    /// A *bound* path (`p = ...`) must address each element so
    /// [`build_path_constructor`] can assemble it. Clients like gdotv
    /// write path bindings with anonymous elements — `p = ()-[]->()` for
    /// the default graph view, `p = ()-[r]-()` for edge expansion — which
    /// would otherwise be rejected. Synthesising bindings makes those
    /// behave exactly like the explicitly-aliased form.
    fn fill_anonymous_path_bindings(&mut self, elem: &PatternElement) -> PatternElement {
        let mut out = elem.clone();
        if out.head.binding.is_none() {
            out.head.binding = Some(self.fresh_anon(out.head.span));
        }
        for (rel, target) in out.chain.iter_mut() {
            if rel.binding.is_none() {
                rel.binding = Some(self.fresh_anon(rel.span));
            }
            if target.binding.is_none() {
                target.binding = Some(self.fresh_anon(target.span));
            }
        }
        out
    }

    fn introduce(&mut self, name: &str, span: SourceSpan) -> Result<(), LowerError> {
        if !self.bindings.insert(name.to_string()) {
            return Err(LowerError::new(
                LowerErrorKind::DoubleBinding,
                format!("binding `{}` is already in scope", name),
                span,
            ));
        }
        Ok(())
    }

    /// `MATCH ... (a) ...` allows reusing an already-bound name as a back-
    /// reference (no new binding, just constraint). Used by patterns that
    /// reference previously bound variables.
    fn introduce_or_reuse(&mut self, name: &str) {
        self.bindings.insert(name.to_string());
    }

    fn ensure(&self, name: &str, span: SourceSpan) -> Result<(), LowerError> {
        if self.bindings.contains(name) {
            Ok(())
        } else {
            Err(LowerError::new(
                LowerErrorKind::BindingNotFound,
                format!("binding `{}` is not in scope", name),
                span,
            ))
        }
    }

    fn reset_to(&mut self, names: impl IntoIterator<Item = String>) {
        self.bindings = names.into_iter().collect();
    }
}

// ─────────────────────────── MATCH ───────────────────────────────────

fn lower_match(
    m: &MatchClause,
    input: Option<LogicalPlan>,
    ctx: &mut LowerCtx,
) -> Result<LogicalPlan, LowerError> {
    // The first part consumes the existing input plan (e.g. from WITH).
    // Subsequent parts have no shared input and are combined via
    // CrossProduct. If a later part back-references a binding from an
    // earlier one, lowering reuses that binding (no cross-product needed
    // because the back-reference threading on Expand stitches them).
    let mut plan = input;
    let mut first = true;
    for part in &m.patterns {
        if first {
            plan = Some(lower_pattern_part(part, plan, m.optional, ctx)?);
            first = false;
        } else if pattern_part_back_references(part, ctx) {
            // Threaded continuation — same scope, no cross product.
            plan = Some(lower_pattern_part(part, plan, m.optional, ctx)?);
        } else {
            let right = lower_pattern_part(part, None, m.optional, ctx)?;
            let left = plan.take().expect("MATCH must have ≥ 1 pattern part");
            plan = Some(LogicalPlan::CrossProduct {
                left: Box::new(left),
                right: Box::new(right),
            });
        }
    }
    let mut plan = plan.expect("MATCH must have ≥ 1 pattern part");
    if let Some(pred) = &m.where_ {
        check_expression_bindings(pred, ctx)?;
        plan = attach_where(plan, pred, ctx)?;
    }
    Ok(plan)
}

/// Split a WHERE predicate into a SemiApply chain (for top-level
/// `EXISTS(pattern)` / `NOT EXISTS(pattern)` terms in the AND-tree) plus
/// a residual Filter for the rest. The order of SemiApply nodes follows
/// the order the terms appear in the source.
fn attach_where(
    input: LogicalPlan,
    pred: &Expression,
    ctx: &mut LowerCtx,
) -> Result<LogicalPlan, LowerError> {
    let mut plan = input;
    let mut residuals: Vec<Expression> = Vec::new();
    for term in collect_and_terms(pred) {
        match classify_exists_term(&term) {
            Some((pattern, negated)) => {
                let subplan = lower_exists_subplan(&pattern, ctx)?;
                plan = LogicalPlan::SemiApply {
                    input: Box::new(plan),
                    subplan: Box::new(subplan),
                    negated,
                };
            }
            None => residuals.push(term),
        }
    }
    if let Some(residual) = rebuild_and_chain(residuals) {
        plan = LogicalPlan::Filter {
            input: Box::new(plan),
            predicate: residual,
        };
    }
    Ok(plan)
}

fn collect_and_terms(expr: &Expression) -> Vec<Expression> {
    match &expr.kind {
        ExpressionKind::Binary {
            op: BinaryOp::And,
            left,
            right,
        } => {
            let mut v = collect_and_terms(left);
            v.extend(collect_and_terms(right));
            v
        }
        _ => vec![expr.clone()],
    }
}

fn classify_exists_term(term: &Expression) -> Option<(PatternElement, bool)> {
    match &term.kind {
        ExpressionKind::Exists(p) => Some((p.as_ref().clone(), false)),
        ExpressionKind::Unary {
            op: UnaryOp::Not,
            expr,
        } => match &expr.kind {
            ExpressionKind::Exists(p) => Some((p.as_ref().clone(), true)),
            _ => None,
        },
        _ => None,
    }
}

fn rebuild_and_chain(terms: Vec<Expression>) -> Option<Expression> {
    let mut iter = terms.into_iter();
    let first = iter.next()?;
    Some(iter.fold(first, |acc, next| {
        let span = SourceSpan::new(acc.span.start, next.span.end);
        Expression {
            kind: ExpressionKind::Binary {
                op: BinaryOp::And,
                left: Box::new(acc),
                right: Box::new(next),
            },
            span,
        }
    }))
}

/// Lower an `EXISTS(pattern)` subplan. The pattern's back-references to
/// outer bindings are resolved against `outer_ctx`; new bindings
/// introduced by the pattern remain local to the subplan.
fn lower_exists_subplan(
    elem: &PatternElement,
    outer_ctx: &LowerCtx,
) -> Result<LogicalPlan, LowerError> {
    let mut sub = LowerCtx::new();
    sub.bindings = outer_ctx.bindings.clone();
    lower_pattern_element(elem, None, false, ShortestMode::None, &mut sub)
}

/// Lower the subplan of a pattern comprehension. Like `lower_exists_subplan`
/// but also folds the comprehension's `predicate` as a Filter on top so
/// the projection step only sees rows that satisfy it.
fn lower_pattern_comprehension_subplan(
    pc: &ast::PatternComprehension,
    outer_ctx: &LowerCtx,
) -> Result<LogicalPlan, LowerError> {
    let mut sub = LowerCtx::new();
    sub.bindings = outer_ctx.bindings.clone();
    let mut plan = lower_pattern_element(&pc.pattern, None, false, ShortestMode::None, &mut sub)?;
    if let Some(pred) = &pc.predicate {
        plan = LogicalPlan::Filter {
            input: Box::new(plan),
            predicate: pred.clone(),
        };
    }
    Ok(plan)
}

/// Hoist top-level `PatternComprehension` items into a chain of
/// `PatternList` operators wrapping `input`. Each hoisted comprehension
/// is replaced in the projection by a synthetic `__pcN` variable. Nested
/// `PatternComprehension` (e.g. as an operand inside a binary expression)
/// is rejected with `UnsupportedFeature` — RFC-008 hoisting only covers
/// top-level projection items in v0.
fn hoist_pattern_comprehensions(
    items: Vec<AstProjectionItem>,
    mut plan: LogicalPlan,
    ctx: &mut LowerCtx,
) -> Result<(Vec<AstProjectionItem>, LogicalPlan), LowerError> {
    let mut new_items = Vec::with_capacity(items.len());
    let mut counter = ctx
        .bindings
        .iter()
        .filter(|b| b.starts_with("__pc"))
        .count();
    for item in items {
        if let ExpressionKind::PatternComprehension(pc) = &item.expression.kind {
            let alias = format!("__pc{}", counter);
            counter += 1;
            let subplan = lower_pattern_comprehension_subplan(pc, ctx)?;
            plan = LogicalPlan::PatternList {
                input: Box::new(plan),
                subplan: Box::new(subplan),
                projection: pc.projection.clone(),
                alias: alias.clone(),
            };
            ctx.bindings.insert(alias.clone());
            let span = item.expression.span;
            new_items.push(AstProjectionItem {
                expression: Expression {
                    kind: ExpressionKind::Variable(ast::Identifier::new(alias, span)),
                    span,
                },
                alias: item.alias,
                span: item.span,
            });
        } else {
            reject_nested_pattern_comprehension(&item.expression)?;
            new_items.push(item);
        }
    }
    Ok((new_items, plan))
}

fn reject_nested_pattern_comprehension(expr: &Expression) -> Result<(), LowerError> {
    match &expr.kind {
        ExpressionKind::PatternComprehension(_) => Err(LowerError::new(
            LowerErrorKind::UnsupportedFeature,
            "pattern comprehensions are only supported as top-level projection items in v0",
            expr.span,
        )),
        ExpressionKind::Property(p) => reject_nested_pattern_comprehension(&p.target),
        ExpressionKind::Index { target, index } => {
            reject_nested_pattern_comprehension(target)?;
            reject_nested_pattern_comprehension(index)
        }
        ExpressionKind::Range { target, from, to } => {
            reject_nested_pattern_comprehension(target)?;
            if let Some(e) = from {
                reject_nested_pattern_comprehension(e)?;
            }
            if let Some(e) = to {
                reject_nested_pattern_comprehension(e)?;
            }
            Ok(())
        }
        ExpressionKind::Unary { expr, .. } => reject_nested_pattern_comprehension(expr),
        ExpressionKind::Binary { left, right, .. } => {
            reject_nested_pattern_comprehension(left)?;
            reject_nested_pattern_comprehension(right)
        }
        ExpressionKind::In { item, list } => {
            reject_nested_pattern_comprehension(item)?;
            reject_nested_pattern_comprehension(list)
        }
        ExpressionKind::StringTest {
            target, pattern, ..
        } => {
            reject_nested_pattern_comprehension(target)?;
            reject_nested_pattern_comprehension(pattern)
        }
        ExpressionKind::IsNull { expr, .. } => reject_nested_pattern_comprehension(expr),
        ExpressionKind::FunctionCall { args, .. } => {
            for a in args {
                reject_nested_pattern_comprehension(a)?;
            }
            Ok(())
        }
        ExpressionKind::Case {
            scrutinee,
            branches,
            otherwise,
        } => {
            if let Some(s) = scrutinee {
                reject_nested_pattern_comprehension(s)?;
            }
            for b in branches {
                reject_nested_pattern_comprehension(&b.when)?;
                reject_nested_pattern_comprehension(&b.then)?;
            }
            if let Some(e) = otherwise {
                reject_nested_pattern_comprehension(e)?;
            }
            Ok(())
        }
        ExpressionKind::List(items) => {
            for it in items {
                reject_nested_pattern_comprehension(it)?;
            }
            Ok(())
        }
        ExpressionKind::Map(m) => {
            for (_, v) in &m.entries {
                reject_nested_pattern_comprehension(v)?;
            }
            Ok(())
        }
        ExpressionKind::ListComprehension(lc) => {
            reject_nested_pattern_comprehension(&lc.list)?;
            if let Some(p) = &lc.predicate {
                reject_nested_pattern_comprehension(p)?;
            }
            if let Some(p) = &lc.projection {
                reject_nested_pattern_comprehension(p)?;
            }
            Ok(())
        }
        ExpressionKind::Exists(_)
        | ExpressionKind::Star
        | ExpressionKind::Variable(_)
        | ExpressionKind::Parameter(_)
        | ExpressionKind::Literal(_) => Ok(()),
    }
}

/// True if `part`'s head binding is already in scope — the lowering
/// treats this as "continue from the existing binding" instead of
/// emitting a CrossProduct between two independent pattern parts.
fn pattern_part_back_references(part: &PatternPart, ctx: &LowerCtx) -> bool {
    match &part.element.head.binding {
        Some(b) => ctx.bindings.contains(&b.name),
        None => false,
    }
}

fn lower_pattern_part(
    part: &PatternPart,
    input: Option<LogicalPlan>,
    optional: bool,
    ctx: &mut LowerCtx,
) -> Result<LogicalPlan, LowerError> {
    let shortest = part
        .shortest_path
        .map(|m| match m {
            ast::ShortestPathMode::First => ShortestMode::First,
            ast::ShortestPathMode::All => ShortestMode::All,
        })
        .unwrap_or(ShortestMode::None);
    if shortest != ShortestMode::None {
        validate_shortest_path_pattern_v0(part, &part.element)?;
        // shortestPath: the executor materialises the path trail
        // directly into the named binding, so the lower does NOT
        // emit a Project + build_path_constructor (which would
        // produce a static node list). We thread the path name
        // through to the Expand so the walker can populate it.
        if let Some(path_bind) = &part.binding {
            ctx.introduce(&path_bind.name, path_bind.span)?;
            let plan = lower_pattern_element(&part.element, input, optional, shortest, ctx)?;
            return Ok(attach_path_binding(plan, &path_bind.name));
        }
        return lower_pattern_element(&part.element, input, optional, shortest, ctx);
    }
    if let Some(path_bind) = &part.binding {
        // Anonymous path elements (`p = ()-[]->()`, `p = ()-[r]-()`) get
        // fresh internal bindings so the path can be assembled; without
        // this the lower rejects them. See `fill_anonymous_path_bindings`.
        let element = ctx.fill_anonymous_path_bindings(&part.element);
        validate_path_pattern_v0(&element)?;
        let plan = lower_pattern_element(&element, input, optional, shortest, ctx)?;
        ctx.introduce(&path_bind.name, path_bind.span)?;
        let path_expr = build_path_constructor(&element, path_bind.span);
        return Ok(LogicalPlan::Project {
            input: Box::new(plan),
            items: vec![ProjectionItem {
                expression: path_expr,
                alias: path_bind.name.clone(),
            }],
            distinct: false,
            discard_input_bindings: false,
        });
    }
    lower_pattern_element(&part.element, input, optional, shortest, ctx)
}

/// Walk the plan top-down until the first `Expand` and set its
/// `path_binding`. Used by `lower_pattern_part` to communicate the
/// path variable name to the executor for shortestPath.
fn attach_path_binding(plan: LogicalPlan, name: &str) -> LogicalPlan {
    match plan {
        LogicalPlan::Expand {
            input,
            source,
            edge_type,
            direction,
            rel_alias,
            target_alias,
            target_label,
            length,
            optional,
            back_reference,
            shortest,
            path_binding: _,
        } => LogicalPlan::Expand {
            input,
            source,
            edge_type,
            direction,
            rel_alias,
            target_alias,
            target_label,
            length,
            optional,
            back_reference,
            shortest,
            path_binding: Some(name.to_string()),
        },
        LogicalPlan::Filter { input, predicate } => LogicalPlan::Filter {
            input: Box::new(attach_path_binding(*input, name)),
            predicate,
        },
        other => other,
    }
}

fn validate_shortest_path_pattern_v0(
    part: &PatternPart,
    elem: &PatternElement,
) -> Result<(), LowerError> {
    // 1. path binding required.
    if part.binding.is_none() {
        return Err(LowerError::new(
            LowerErrorKind::UnsupportedFeature,
            "shortestPath / allShortestPaths require a path variable \
             (e.g. `MATCH p = shortestPath(...)`)",
            part.span,
        ));
    }
    // 2. Single relationship hop.
    if elem.chain.len() != 1 {
        return Err(LowerError::new(
            LowerErrorKind::UnsupportedFeature,
            "shortestPath accepts a single relationship hop; \
             split multi-leg paths across separate MATCH clauses",
            part.span,
        ));
    }
    let (rel, target) = &elem.chain[0];
    // 3. Finite upper bound — `*..N` or `*min..N` or fixed `*N`.
    match rel.length {
        Some(crate::parser::RelationshipLength { max, .. }) if max < u32::MAX => {}
        _ => {
            return Err(LowerError::new(
                LowerErrorKind::UnsupportedFeature,
                "shortestPath requires a finite upper bound \
                 (e.g. `*..15` or `*1..5`); unbounded `*` is rejected",
                rel.span,
            ));
        }
    }
    // 4. Both endpoints named, so the executor's back-reference path
    // identifies the targets. Their values come from a prior MATCH
    // clause in the query.
    if elem.head.binding.is_none() {
        return Err(LowerError::new(
            LowerErrorKind::UnsupportedFeature,
            "shortestPath source must reference a previously bound node \
             (e.g. `MATCH (a:Person ...) MATCH p = shortestPath((a)-[*..15]-(b))`)",
            elem.head.span,
        ));
    }
    if target.binding.is_none() {
        return Err(LowerError::new(
            LowerErrorKind::UnsupportedFeature,
            "shortestPath target must reference a previously bound node",
            target.span,
        ));
    }
    Ok(())
}

fn validate_path_pattern_v0(elem: &PatternElement) -> Result<(), LowerError> {
    // Anonymous elements are filled with internal bindings before this
    // runs (see `fill_anonymous_path_bindings`), so the only path-binding
    // shape still unsupported in v0 is variable-length.
    for (rel, _target) in &elem.chain {
        if rel.length.is_some() {
            return Err(LowerError::new(
                LowerErrorKind::UnsupportedFeature,
                "variable-length path bindings (e.g. `p = (a)-[*1..3]->(b)`) are not yet supported",
                rel.span,
            ));
        }
    }
    Ok(())
}

fn build_path_constructor(elem: &PatternElement, span: SourceSpan) -> Expression {
    let mut args: Vec<Expression> = Vec::with_capacity(1 + 2 * elem.chain.len());
    let head_name = elem.head.binding.as_ref().expect("validated").name.clone();
    args.push(var_expression(&head_name, span));
    for (rel, target) in &elem.chain {
        // For shortestPath patterns (RFC-023) the rel / target may
        // lack an explicit binding (LDBC IC13 writes
        // `shortestPath((a)-[:KNOWS*..15]-(b))`). Use a null literal
        // for the missing element so `length(p)` still works; the
        // `__path` builtin tolerates null fillers and counts only
        // non-null relationships.
        if let Some(b) = &rel.binding {
            args.push(var_expression(&b.name, span));
        } else {
            args.push(null_expression(span));
        }
        if let Some(b) = &target.binding {
            args.push(var_expression(&b.name, span));
        } else {
            args.push(null_expression(span));
        }
    }
    Expression {
        kind: ExpressionKind::FunctionCall {
            name: QualifiedName::single(ast::Identifier::new("__path", span)),
            args,
            distinct: false,
        },
        span,
    }
}

fn null_expression(span: SourceSpan) -> Expression {
    Expression {
        kind: ExpressionKind::Literal(crate::parser::Literal::Null),
        span,
    }
}

fn var_expression(name: &str, span: SourceSpan) -> Expression {
    Expression {
        kind: ExpressionKind::Variable(ast::Identifier::new(name, span)),
        span,
    }
}

fn lower_pattern_element(
    elem: &PatternElement,
    input: Option<LogicalPlan>,
    optional: bool,
    shortest: ShortestMode,
    ctx: &mut LowerCtx,
) -> Result<LogicalPlan, LowerError> {
    let mut plan = lower_node_pattern_head(&elem.head, input, optional, ctx)?;
    // Track the source binding for the next relationship explicitly. The
    // head alias is the source for the first hop; each subsequent target
    // becomes the source for the hop after it. This works even when the
    // accumulated plan top is a Project / TopN / Aggregate from a
    // preceding WITH clause — `previous_source` only sees the plan
    // shape and would otherwise reject those operators.
    let mut current_source = elem.head.binding.as_ref().map(|b| b.name.clone());
    for (rel, target) in &elem.chain {
        let target_alias_for_next = target.binding.as_ref().map(|b| b.name.clone());
        plan = lower_rel_node(
            plan,
            current_source.as_deref(),
            &elem.head,
            rel,
            target,
            optional,
            shortest,
            ctx,
        )?;
        // After this rel, the target becomes the source for the next.
        // Anonymous targets are named inside lower_rel_node; we don't
        // need to track them here because variable-length chains
        // terminate at named bindings in v0 LDBC queries.
        current_source = target_alias_for_next.or(current_source);
    }
    Ok(plan)
}

fn lower_node_pattern_head(
    head: &NodePattern,
    input: Option<LogicalPlan>,
    optional: bool,
    ctx: &mut LowerCtx,
) -> Result<LogicalPlan, LowerError> {
    // Back-reference path: if `(a)` re-uses an already bound `a`, no new
    // scan is emitted and a label is not required.
    if let Some(binding) = &head.binding {
        if ctx.bindings.contains(&binding.name) {
            return Ok(input.unwrap_or_else(|| LogicalPlan::Argument {
                bindings: vec![binding.name.clone()],
            }));
        }
    }
    let label = optional_primary_label(head);
    let extra_labels = pattern_extra_labels(head);
    let alias = head
        .binding
        .as_ref()
        .map(|b| b.name.clone())
        .unwrap_or_else(|| anonymous_alias(ctx));
    ctx.introduce(&alias, head.span)?;

    // Detect inline `{_id: $param}` filter and lower to NodeById.
    // The `id` key is reserved for user properties; only `_id` (the
    // explicit internal-NodeId sigil) triggers the fast point lookup.
    if let Some(PatternProperties::Parameter { span, .. }) = &head.properties {
        return Err(LowerError::new(
            LowerErrorKind::UnsupportedFeature,
            "$params spread in a MATCH pattern is not supported yet; list the keys explicitly in WHERE",
            *span,
        ));
    }
    if let Some(PatternProperties::Literal(map)) = &head.properties {
        if let Some(id_expr) = map
            .entries
            .iter()
            .find(|(k, _)| k.name == "_id")
            .map(|(_, v)| v.clone())
        {
            // NodeById requires a label (single-label CF lookup). Reject
            // typeless `MATCH ({_id: $x})` here — the user should write
            // `MATCH (n:Label {_id: $x})` to use the fast point lookup.
            let label = label.ok_or_else(|| {
                LowerError::new(
                    LowerErrorKind::UnsupportedFeature,
                    "_id-lookup requires an explicit label (e.g. `MATCH (n:Label {_id: $x})`)",
                    head.span,
                )
            })?;
            // Map must contain ONLY `_id` — any extra props become Filter.
            let extra_filters: Vec<_> = map
                .entries
                .iter()
                .filter(|(k, _)| k.name != "_id")
                .cloned()
                .collect();
            let inner_input = input.unwrap_or(LogicalPlan::Empty);
            let mut plan = LogicalPlan::NodeById {
                input: Box::new(inner_input),
                label: Some(label.to_string()),
                alias: alias.clone(),
                id: id_expr,
            };
            for (key, val) in extra_filters {
                let pred = build_eq(&alias, &key.name, val, head.span);
                plan = LogicalPlan::Filter {
                    input: Box::new(plan),
                    predicate: pred,
                };
            }
            let _ = optional; // OPTIONAL on NodeById not meaningful in v0.
            return Ok(wrap_extra_label_filters(
                plan,
                &alias,
                &extra_labels,
                head.span,
            ));
        }
        // Map without `_id`: build NodeScan, join with the carried-in
        // input (e.g. an UNWIND that introduces outer-row bindings the
        // map filter references) and then layer the per-prop Filters on
        // top of the joined plan. Doing the Filters BELOW the join would
        // hide the outer bindings (the right side of CrossProduct sees
        // only its own subtree), which is the root cause of B1.
        //
        // Filter pushdown is then free to re-sink predicates that don't
        // reference outer bindings — RFC-014 §"Predicate pushdown".
        let scan = LogicalPlan::NodeScan {
            label: label.map(str::to_string),
            alias: alias.clone(),
            predicates: vec![],
            projection: None,
        };
        let mut plan = combine(input, scan);
        for (key, val) in map.entries.iter() {
            let pred = build_eq(&alias, &key.name, val.clone(), head.span);
            plan = LogicalPlan::Filter {
                input: Box::new(plan),
                predicate: pred,
            };
        }
        return Ok(wrap_extra_label_filters(
            plan,
            &alias,
            &extra_labels,
            head.span,
        ));
    }

    let scan = LogicalPlan::NodeScan {
        label: label.map(str::to_string),
        alias: alias.clone(),
        predicates: vec![],
        projection: None,
    };
    Ok(wrap_extra_label_filters(
        combine(input, scan),
        &alias,
        &extra_labels,
        head.span,
    ))
}

#[allow(clippy::too_many_arguments)]
fn lower_rel_node(
    input: LogicalPlan,
    explicit_source: Option<&str>,
    _head: &NodePattern,
    rel: &RelationshipPattern,
    target: &NodePattern,
    optional: bool,
    shortest: ShortestMode,
    ctx: &mut LowerCtx,
) -> Result<LogicalPlan, LowerError> {
    // Prefer the explicit source (passed by `lower_pattern_element`,
    // which tracks the chain head + each target as the next source).
    // Fall back to inspecting the plan shape only if the chain didn't
    // name a head — e.g. anonymous mid-chain nodes left as fallback.
    let source = match explicit_source {
        Some(name) if ctx.bindings.contains(name) => name.to_string(),
        _ => previous_source(&input)?,
    };
    let _ = ctx; // ctx no longer needed; reserved for label resolution.
                 // `[:A|:B|:C]` lowers to a non-empty Vec; `[]` (untyped) lowers to
                 // None. The executor unions the partner lists across listed types
                 // (RFC-024 §"Open questions" Q1).
    let edge_type: Option<Vec<String>> = match rel.types.as_slice() {
        [] => None,
        types => Some(types.iter().map(|t| t.name.clone()).collect()),
    };
    let rel_alias = rel.binding.as_ref().map(|b| b.name.clone());
    if let Some(name) = &rel_alias {
        ctx.introduce(name, rel.span)?;
    }
    let target_alias = target
        .binding
        .as_ref()
        .map(|b| b.name.clone())
        .unwrap_or_else(|| anonymous_alias(ctx));
    let target_already_bound = ctx.bindings.contains(&target_alias);
    if !target_already_bound {
        ctx.introduce(&target_alias, target.span)?;
    } else {
        // Back-reference; reusing existing binding. Lowering still emits
        // an Expand but downstream the executor checks identity.
        ctx.introduce_or_reuse(&target_alias);
    }

    // A multi-label relationship target needs a conjunctive label check on the
    // matched node. For a non-OPTIONAL expand we emit that as post-expand
    // `__label_eq` filters (below). OPTIONAL is different: the check must live
    // *inside* the expand so a target carrying only some of the labels still
    // yields a NULL row, but `Expand` only carries the primary label today
    // (target_labels is a future change threaded through the WCOJ optimizer).
    // Rather than silently match on the primary label alone, reject it.
    if optional && target.labels.len() > 1 {
        return Err(LowerError::new(
            LowerErrorKind::UnsupportedFeature,
            "OPTIONAL MATCH with a multi-label relationship target is not yet supported",
            target.span,
        ));
    }
    let target_label = target.labels.first().map(|l| l.name.clone());
    let mut plan = LogicalPlan::Expand {
        input: Box::new(input),
        source,
        edge_type,
        direction: rel.direction,
        rel_alias,
        target_alias: target_alias.clone(),
        target_label: target_label.clone(),
        length: rel.length,
        optional,
        back_reference: target_already_bound,
        shortest,
        path_binding: None,
    };
    // OPTIONAL MATCH must preserve rows where the target is NULL. Label
    // and property checks are issued via the executor's `target_label`
    // hint and are folded into the Expand itself when optional.
    if !optional {
        // One label filter per target label: `(a)-[:R]->(b:A:B)` requires the
        // target to carry every listed label. The optimizer folds the primary
        // one into the Expand's `target_label` hint; any extras stay as
        // post-expand `__label_eq` filters.
        for l in target.labels.iter().map(|l| l.name.as_str()) {
            let pred = build_label_eq(&target_alias, l, target.span);
            plan = LogicalPlan::Filter {
                input: Box::new(plan),
                predicate: pred,
            };
        }
        match &target.properties {
            None => {}
            Some(PatternProperties::Literal(map)) => {
                for (key, val) in map.entries.iter() {
                    let pred = build_eq(&target_alias, &key.name, val.clone(), target.span);
                    plan = LogicalPlan::Filter {
                        input: Box::new(plan),
                        predicate: pred,
                    };
                }
            }
            Some(PatternProperties::Parameter { span, .. }) => {
                return Err(LowerError::new(
                    LowerErrorKind::UnsupportedFeature,
                    "$params spread in a MATCH pattern is not supported yet; list the keys explicitly in WHERE",
                    *span,
                ));
            }
        }
    }
    Ok(plan)
}

fn previous_source(plan: &LogicalPlan) -> Result<String, LowerError> {
    // The "source" of an Expand is the most recently introduced binding
    // that exists in scope. We compute it by inspecting the plan shape.
    match plan {
        LogicalPlan::NodeScan { alias, .. } | LogicalPlan::NodeById { alias, .. } => {
            Ok(alias.clone())
        }
        LogicalPlan::Expand { target_alias, .. } => Ok(target_alias.clone()),
        LogicalPlan::Filter { input, .. } | LogicalPlan::SemiApply { input, .. } => {
            previous_source(input)
        }
        LogicalPlan::Argument { bindings } if !bindings.is_empty() => {
            Ok(bindings.last().unwrap().clone())
        }
        LogicalPlan::CrossProduct { right, .. } => previous_source(right),
        _ => Err(LowerError::new(
            LowerErrorKind::InvalidPattern,
            "cannot identify source binding for relationship expansion",
            SourceSpan::point(0),
        )),
    }
}

/// The pattern's primary label (the first), used as the scan / CF hint.
/// `MATCH (n:A:B)` scans by `A`; the remaining labels become conjunctive
/// `__label_eq` filters (see [`wrap_extra_label_filters`]). `None` for an
/// unlabelled pattern.
fn optional_primary_label(node: &NodePattern) -> Option<&str> {
    node.labels.first().map(|l| l.name.as_str())
}

/// Labels beyond the primary one. `MATCH (n:A:B)` yields `["B"]`; these are
/// required on top of the primary-label scan (conjunctive set semantics).
fn pattern_extra_labels(node: &NodePattern) -> Vec<String> {
    node.labels.iter().skip(1).map(|l| l.name.clone()).collect()
}

/// Wrap `plan` in a `__label_eq(alias, label)` Filter for each extra label so
/// a multi-label pattern only keeps nodes carrying ALL of them.
fn wrap_extra_label_filters(
    plan: LogicalPlan,
    alias: &str,
    extra_labels: &[String],
    span: SourceSpan,
) -> LogicalPlan {
    extra_labels
        .iter()
        .fold(plan, |acc, label| LogicalPlan::Filter {
            input: Box::new(acc),
            predicate: build_label_eq(alias, label, span),
        })
}

fn anonymous_alias(ctx: &LowerCtx) -> String {
    // Returns the first unused `__anon<N>` name without inserting it. The
    // caller is responsible for `ctx.introduce(&alias, span)?` immediately
    // afterwards, which makes the name visible for subsequent calls.
    for i in 0..usize::MAX {
        let candidate = format!("__anon{}", i);
        if !ctx.bindings.contains(&candidate) {
            return candidate;
        }
    }
    unreachable!("astronomical alias exhaustion")
}

// ─────────────────────────── RETURN / WITH ───────────────────────────

fn lower_return(
    r: &ReturnClause,
    input: LogicalPlan,
    ctx: &mut LowerCtx,
) -> Result<LogicalPlan, LowerError> {
    lower_projection(
        &r.items,
        r.distinct,
        &r.order_by,
        &r.skip,
        &r.limit,
        None,
        input,
        ctx,
        /*discard*/ true,
    )
}

fn lower_with(
    w: &WithClause,
    input: LogicalPlan,
    ctx: &mut LowerCtx,
) -> Result<LogicalPlan, LowerError> {
    lower_projection(
        &w.items,
        w.distinct,
        &w.order_by,
        &w.skip,
        &w.limit,
        w.where_.as_ref(),
        input,
        ctx,
        /*discard*/ true,
    )
}

#[allow(clippy::too_many_arguments)]
fn lower_projection(
    items: &[AstProjectionItem],
    distinct: bool,
    order_by: &[ast::OrderItem],
    skip: &Option<Expression>,
    limit: &Option<Expression>,
    where_: Option<&Expression>,
    input: LogicalPlan,
    ctx: &mut LowerCtx,
    discard_input_bindings: bool,
) -> Result<LogicalPlan, LowerError> {
    let items_owned = expand_star_items(items, ctx)?;
    let (items_owned, input) = hoist_pattern_comprehensions(items_owned, input, ctx)?;
    let items: &[AstProjectionItem] = &items_owned;

    // Validate item expressions reference live bindings. The WHERE
    // attached to a WITH (`WITH ... AS countries WHERE size(countries)
    // = 2`) is validated AFTER the projection aliases land in scope —
    // see the post-`reset_to` check further down.
    for item in items {
        check_expression_bindings(&item.expression, ctx)?;
    }

    // Resolve aliases per item up-front so the order can reference either
    // the original expression or its alias.
    let aliases: Vec<String> = items
        .iter()
        .map(|i| {
            i.alias
                .as_ref()
                .map(|a| a.name.clone())
                .unwrap_or_else(|| canonical_alias(&i.expression))
        })
        .collect();

    // Hoist every aggregate call (including those nested inside scalar
    // functions like `head(collect(x))`) into a synthetic `__aggN`
    // alias. The rewritten item expression references the synthetic
    // alias by variable, and the Aggregate operator materialises it.
    let (extracted_aggs, items_rewritten) = extract_aggregates_in_items(items)?;
    let has_aggs = !extracted_aggs.is_empty();

    // After hoisting, an item is a *group key* iff its rewritten
    // expression contains no `__aggN` reference. Group keys carry their
    // alias onto the row produced by Aggregate.
    let group_keys: GroupKeys = if has_aggs {
        items_rewritten
            .iter()
            .zip(aliases.iter())
            .filter(|(item, _)| !contains_agg_variable(&item.expression))
            .map(|(item, alias)| (item.expression.clone(), alias.clone()))
            .collect()
    } else {
        Vec::new()
    };

    let mut plan = input;
    if has_aggs {
        plan = LogicalPlan::Aggregate {
            input: Box::new(plan),
            group_by: group_keys.clone(),
            aggregations: extracted_aggs.clone(),
        };
    }

    // ORDER BY substitution: when a key references an alias defined in
    // this projection, swap it for the rewritten item expression so
    // TopN can be evaluated on the row that exists *before* the final
    // Project (pre-Project for non-agg queries; post-Aggregate, pre-
    // Project for agg queries — in both cases the row carries the
    // rewritten expression's free variables, not the alias itself).
    let alias_map: BTreeMap<String, Expression> = items_rewritten
        .iter()
        .zip(aliases.iter())
        .map(|(item, alias)| (alias.clone(), item.expression.clone()))
        .collect();
    // `keep_as_var` lists aliases that already live on the row reaching
    // TopN, i.e. group keys of the Aggregate. Everything else gets
    // substituted back to its rewritten expression so TopN can evaluate
    // it against the pre-Project row.
    let keep_as_var: BTreeSet<String> = if has_aggs {
        items_rewritten
            .iter()
            .zip(aliases.iter())
            .filter(|(item, _)| !contains_agg_variable(&item.expression))
            .map(|(_, alias)| alias.clone())
            .collect()
    } else {
        BTreeSet::new()
    };

    let order_keys: Vec<OrderKey> = order_by
        .iter()
        .map(|k| OrderKey {
            expression: substitute_aliases(&k.expression, &alias_map, &keep_as_var),
            direction: k.direction,
        })
        .collect();
    let skip_rc = optional_row_count(skip)?.unwrap_or(RowCount::Const(0));
    let limit_rc = optional_row_count(limit)?.unwrap_or(RowCount::Const(u64::MAX));

    // Build the Project items. For group-key items under aggregation
    // the alias already lives on the row, so we project `Variable(alias)`.
    // For agg-containing items (e.g. `head(__agg0)`) we keep the
    // rewritten expression so it is evaluated on the post-aggregate row.
    // For non-agg queries we project the rewritten expression directly.
    let projection_items: Vec<ProjectionItem> = items_rewritten
        .iter()
        .zip(aliases.iter())
        .map(|(item, alias)| {
            let agg_inside = contains_agg_variable(&item.expression);
            if has_aggs && !agg_inside {
                ProjectionItem {
                    expression: Expression {
                        kind: ExpressionKind::Variable(ast::Identifier::new(
                            alias.clone(),
                            item.expression.span,
                        )),
                        span: item.expression.span,
                    },
                    alias: alias.clone(),
                }
            } else {
                ProjectionItem {
                    expression: item.expression.clone(),
                    alias: alias.clone(),
                }
            }
        })
        .collect();

    if !order_keys.is_empty()
        || skip_rc != RowCount::Const(0)
        || limit_rc != RowCount::Const(u64::MAX)
    {
        plan = LogicalPlan::TopN {
            input: Box::new(plan),
            keys: order_keys,
            skip: skip_rc,
            limit: limit_rc,
        };
    }

    plan = LogicalPlan::Project {
        input: Box::new(plan),
        items: projection_items,
        distinct,
        discard_input_bindings,
    };

    if discard_input_bindings {
        ctx.reset_to(aliases.iter().cloned());
    } else {
        for name in &aliases {
            ctx.introduce_or_reuse(name);
        }
    }

    if let Some(pred) = where_ {
        // Validate after the projection aliases are visible — IC03 et
        // al. reference WITH-introduced names from the trailing WHERE.
        check_expression_bindings(pred, ctx)?;
        plan = LogicalPlan::Filter {
            input: Box::new(plan),
            predicate: pred.clone(),
        };
    }

    Ok(plan)
}

/// Substitute references to projection aliases for the original (or
/// rewritten) item expression. `keep_as_var` lists alias names that
/// must be left as plain variables — typically because they are
/// already materialised on the row (e.g. group keys after Aggregate).
fn substitute_aliases(
    expr: &Expression,
    alias_map: &BTreeMap<String, Expression>,
    keep_as_var: &BTreeSet<String>,
) -> Expression {
    use ExpressionKind::*;
    let span = expr.span;
    let new_kind = match &expr.kind {
        Variable(id) => {
            if keep_as_var.contains(&id.name) {
                return expr.clone();
            }
            if let Some(orig) = alias_map.get(&id.name) {
                let mut cloned = orig.clone();
                cloned.span = span;
                return cloned;
            }
            return expr.clone();
        }
        Property(p) => Property(Box::new(ast::PropertyAccess {
            target: substitute_aliases(&p.target, alias_map, keep_as_var),
            key: p.key.clone(),
            span: p.span,
        })),
        Unary { op, expr: inner } => Unary {
            op: *op,
            expr: Box::new(substitute_aliases(inner, alias_map, keep_as_var)),
        },
        Binary { op, left, right } => Binary {
            op: *op,
            left: Box::new(substitute_aliases(left, alias_map, keep_as_var)),
            right: Box::new(substitute_aliases(right, alias_map, keep_as_var)),
        },
        IsNull {
            expr: inner,
            negated,
        } => IsNull {
            expr: Box::new(substitute_aliases(inner, alias_map, keep_as_var)),
            negated: *negated,
        },
        In { item, list } => In {
            item: Box::new(substitute_aliases(item, alias_map, keep_as_var)),
            list: Box::new(substitute_aliases(list, alias_map, keep_as_var)),
        },
        FunctionCall {
            name,
            args,
            distinct,
        } => FunctionCall {
            name: name.clone(),
            args: args
                .iter()
                .map(|a| substitute_aliases(a, alias_map, keep_as_var))
                .collect(),
            distinct: *distinct,
        },
        _ => return expr.clone(),
    };
    Expression {
        kind: new_kind,
        span,
    }
}

/// Walk projection items and hoist every aggregate function call into
/// a synthetic `__aggN` alias. Returns the list of (alias, AggregateExpr)
/// extractions plus the rewritten items whose aggregate sub-expressions
/// have been replaced by `Variable(__aggN)`.
fn extract_aggregates_in_items(
    items: &[AstProjectionItem],
) -> Result<(Aggregations, Vec<AstProjectionItem>), LowerError> {
    let mut aggs: Aggregations = Vec::new();
    let mut out = Vec::with_capacity(items.len());
    for item in items {
        let new_expr = walk_extract_agg(&item.expression, &mut aggs)?;
        out.push(AstProjectionItem {
            expression: new_expr,
            alias: item.alias.clone(),
            span: item.span,
        });
    }
    Ok((aggs, out))
}

fn walk_extract_agg(expr: &Expression, aggs: &mut Aggregations) -> Result<Expression, LowerError> {
    // If this node itself is an aggregate call, hoist it.
    if let Some(agg) = try_aggregate(expr)? {
        let alias = format!("__agg{}", aggs.len());
        let span = expr.span;
        aggs.push((alias.clone(), agg));
        return Ok(Expression {
            kind: ExpressionKind::Variable(ast::Identifier::new(alias, span)),
            span,
        });
    }
    // Otherwise recurse into children.
    use ExpressionKind::*;
    let span = expr.span;
    let new_kind = match &expr.kind {
        Property(p) => Property(Box::new(ast::PropertyAccess {
            target: walk_extract_agg(&p.target, aggs)?,
            key: p.key.clone(),
            span: p.span,
        })),
        Index { target, index } => Index {
            target: Box::new(walk_extract_agg(target, aggs)?),
            index: Box::new(walk_extract_agg(index, aggs)?),
        },
        Range { target, from, to } => Range {
            target: Box::new(walk_extract_agg(target, aggs)?),
            from: match from {
                Some(e) => Some(Box::new(walk_extract_agg(e, aggs)?)),
                None => None,
            },
            to: match to {
                Some(e) => Some(Box::new(walk_extract_agg(e, aggs)?)),
                None => None,
            },
        },
        Unary { op, expr: inner } => Unary {
            op: *op,
            expr: Box::new(walk_extract_agg(inner, aggs)?),
        },
        Binary { op, left, right } => Binary {
            op: *op,
            left: Box::new(walk_extract_agg(left, aggs)?),
            right: Box::new(walk_extract_agg(right, aggs)?),
        },
        In { item, list } => In {
            item: Box::new(walk_extract_agg(item, aggs)?),
            list: Box::new(walk_extract_agg(list, aggs)?),
        },
        StringTest {
            op,
            target,
            pattern,
        } => StringTest {
            op: *op,
            target: Box::new(walk_extract_agg(target, aggs)?),
            pattern: Box::new(walk_extract_agg(pattern, aggs)?),
        },
        IsNull {
            expr: inner,
            negated,
        } => IsNull {
            expr: Box::new(walk_extract_agg(inner, aggs)?),
            negated: *negated,
        },
        FunctionCall {
            name,
            args,
            distinct,
        } => FunctionCall {
            name: name.clone(),
            args: args
                .iter()
                .map(|a| walk_extract_agg(a, aggs))
                .collect::<Result<_, _>>()?,
            distinct: *distinct,
        },
        Case {
            scrutinee,
            branches,
            otherwise,
        } => Case {
            scrutinee: match scrutinee {
                Some(s) => Some(Box::new(walk_extract_agg(s, aggs)?)),
                None => None,
            },
            branches: branches
                .iter()
                .map(|b| {
                    Ok::<_, LowerError>(ast::CaseBranch {
                        when: walk_extract_agg(&b.when, aggs)?,
                        then: walk_extract_agg(&b.then, aggs)?,
                        span: b.span,
                    })
                })
                .collect::<Result<_, _>>()?,
            otherwise: match otherwise {
                Some(e) => Some(Box::new(walk_extract_agg(e, aggs)?)),
                None => None,
            },
        },
        List(items) => List(
            items
                .iter()
                .map(|it| walk_extract_agg(it, aggs))
                .collect::<Result<_, _>>()?,
        ),
        _ => return Ok(expr.clone()),
    };
    Ok(Expression {
        kind: new_kind,
        span,
    })
}

fn contains_agg_variable(expr: &Expression) -> bool {
    match &expr.kind {
        ExpressionKind::Variable(id) => id.name.starts_with("__agg"),
        ExpressionKind::Property(p) => contains_agg_variable(&p.target),
        ExpressionKind::Index { target, index } => {
            contains_agg_variable(target) || contains_agg_variable(index)
        }
        ExpressionKind::Range { target, from, to } => {
            contains_agg_variable(target)
                || from.as_ref().is_some_and(|e| contains_agg_variable(e))
                || to.as_ref().is_some_and(|e| contains_agg_variable(e))
        }
        ExpressionKind::Unary { expr, .. } => contains_agg_variable(expr),
        ExpressionKind::Binary { left, right, .. } => {
            contains_agg_variable(left) || contains_agg_variable(right)
        }
        ExpressionKind::In { item, list } => {
            contains_agg_variable(item) || contains_agg_variable(list)
        }
        ExpressionKind::StringTest {
            target, pattern, ..
        } => contains_agg_variable(target) || contains_agg_variable(pattern),
        ExpressionKind::IsNull { expr, .. } => contains_agg_variable(expr),
        ExpressionKind::FunctionCall { args, .. } => args.iter().any(contains_agg_variable),
        ExpressionKind::Case {
            scrutinee,
            branches,
            otherwise,
        } => {
            scrutinee.as_ref().is_some_and(|e| contains_agg_variable(e))
                || branches
                    .iter()
                    .any(|b| contains_agg_variable(&b.when) || contains_agg_variable(&b.then))
                || otherwise.as_ref().is_some_and(|e| contains_agg_variable(e))
        }
        ExpressionKind::List(items) => items.iter().any(contains_agg_variable),
        ExpressionKind::Map(m) => m.entries.iter().any(|(_, v)| contains_agg_variable(v)),
        _ => false,
    }
}

fn canonical_alias(expr: &Expression) -> String {
    // Strip the canonical Display form down to a single line and use as
    // alias for anonymous projections.
    expr.to_string()
}

/// Expand any `*` projection item into one item per visible, named
/// binding in the current scope. Anonymous bindings (`__anon*`,
/// introduced by patterns without an explicit variable) are skipped —
/// they are not user-addressable. Non-`*` items are preserved verbatim.
///
/// RFC-004 §Open question Q1 — `RETURN *` resolution for v0.
fn expand_star_items(
    items: &[AstProjectionItem],
    ctx: &LowerCtx,
) -> Result<Vec<AstProjectionItem>, LowerError> {
    if !items
        .iter()
        .any(|i| matches!(i.expression.kind, ExpressionKind::Star))
    {
        return Ok(items.to_vec());
    }
    let mut out = Vec::with_capacity(items.len());
    for item in items {
        if matches!(item.expression.kind, ExpressionKind::Star) {
            if item.alias.is_some() {
                return Err(LowerError::new(
                    LowerErrorKind::InvalidPattern,
                    "`*` projection cannot carry an alias",
                    item.expression.span,
                ));
            }
            let span = item.expression.span;
            let visible: Vec<&String> = ctx
                .bindings
                .iter()
                .filter(|b| !b.starts_with("__anon"))
                .collect();
            if visible.is_empty() {
                return Err(LowerError::new(
                    LowerErrorKind::InvalidPattern,
                    "`*` projection requires at least one named binding in scope",
                    span,
                ));
            }
            for name in visible {
                let id = ast::Identifier::new(name.clone(), span);
                out.push(AstProjectionItem {
                    expression: Expression {
                        kind: ExpressionKind::Variable(id.clone()),
                        span,
                    },
                    alias: Some(id),
                    span,
                });
            }
        } else {
            out.push(item.clone());
        }
    }
    Ok(out)
}

fn optional_row_count(expr: &Option<Expression>) -> Result<Option<RowCount>, LowerError> {
    match expr {
        None => Ok(None),
        Some(e) => match &e.kind {
            ExpressionKind::Literal(Literal::Integer(n)) if *n >= 0 => {
                Ok(Some(RowCount::Const(*n as u64)))
            }
            ExpressionKind::Literal(Literal::Integer(_)) => Err(LowerError::new(
                LowerErrorKind::InvalidPattern,
                "SKIP / LIMIT must be a non-negative integer literal or a $parameter",
                e.span,
            )),
            // A `$param` is carried into the plan by name and resolved at
            // execution time, so the cached plan is reused across params.
            ExpressionKind::Parameter(name) => Ok(Some(RowCount::Param(name.clone()))),
            _ => Err(LowerError::new(
                LowerErrorKind::InvalidPattern,
                "SKIP / LIMIT must be a non-negative integer literal or a $parameter",
                e.span,
            )),
        },
    }
}

type GroupKeys = Vec<(Expression, String)>;
type Aggregations = Vec<(String, AggregateExpr)>;

fn try_aggregate(expr: &Expression) -> Result<Option<AggregateExpr>, LowerError> {
    let (name, args, distinct) = match &expr.kind {
        ExpressionKind::FunctionCall {
            name,
            args,
            distinct,
        } => (name, args.clone(), *distinct),
        _ => return Ok(None),
    };
    let canonical = name.joined().to_ascii_lowercase();
    match canonical.as_str() {
        "count" => match args.as_slice() {
            [] => Ok(Some(AggregateExpr::Count {
                arg: None,
                distinct,
            })),
            [single] if matches!(single.kind, ExpressionKind::Star) => {
                Ok(Some(AggregateExpr::Count {
                    arg: None,
                    distinct,
                }))
            }
            [single] => Ok(Some(AggregateExpr::Count {
                arg: Some(single.clone()),
                distinct,
            })),
            _ => Err(LowerError::new(
                LowerErrorKind::InvalidPattern,
                "count takes 0 or 1 argument",
                expr.span,
            )),
        },
        "sum" => single_arg(name, &args, expr.span).map(|arg| {
            Some(AggregateExpr::Sum {
                arg: arg.clone(),
                distinct,
            })
        }),
        "avg" => single_arg(name, &args, expr.span).map(|arg| {
            Some(AggregateExpr::Avg {
                arg: arg.clone(),
                distinct,
            })
        }),
        "min" => single_arg(name, &args, expr.span)
            .map(|arg| Some(AggregateExpr::Min { arg: arg.clone() })),
        "max" => single_arg(name, &args, expr.span)
            .map(|arg| Some(AggregateExpr::Max { arg: arg.clone() })),
        "collect" => single_arg(name, &args, expr.span).map(|arg| {
            Some(AggregateExpr::Collect {
                arg: arg.clone(),
                distinct,
            })
        }),
        _ => Ok(None),
    }
}

fn single_arg<'a>(
    name: &QualifiedName,
    args: &'a [Expression],
    span: SourceSpan,
) -> Result<&'a Expression, LowerError> {
    match args {
        [single] => Ok(single),
        _ => Err(LowerError::new(
            LowerErrorKind::InvalidPattern,
            format!("{} takes exactly 1 argument", name.joined()),
            span,
        )),
    }
}

// ─────────────────────────── UNWIND ──────────────────────────────────

fn lower_unwind(
    u: &UnwindClause,
    input: LogicalPlan,
    ctx: &mut LowerCtx,
) -> Result<LogicalPlan, LowerError> {
    check_expression_bindings(&u.list, ctx)?;
    ctx.introduce(&u.alias.name, u.alias.span)?;
    Ok(LogicalPlan::Unwind {
        input: Box::new(input),
        list: u.list.clone(),
        alias: u.alias.name.clone(),
    })
}

// ─────────────────────────── CREATE / MERGE ──────────────────────────

fn lower_create(
    c: &ast::CreateClause,
    input: LogicalPlan,
    ctx: &mut LowerCtx,
) -> Result<LogicalPlan, LowerError> {
    let mut elements = Vec::new();
    for part in &c.patterns {
        if part.binding.is_some() {
            return Err(LowerError::new(
                LowerErrorKind::UnsupportedFeature,
                "path bindings inside CREATE are not yet supported",
                part.span,
            ));
        }
        lower_create_pattern_element(&part.element, &mut elements, ctx)?;
    }
    Ok(LogicalPlan::Create {
        input: Box::new(input),
        elements,
    })
}

fn lower_create_pattern_element(
    elem: &PatternElement,
    out: &mut Vec<CreateElement>,
    ctx: &mut LowerCtx,
) -> Result<(), LowerError> {
    let head_alias = lower_create_node(&elem.head, out, ctx)?;
    let mut source = head_alias;
    for (rel, target) in &elem.chain {
        let target_alias = lower_create_node(target, out, ctx)?;
        let alias = match &rel.binding {
            Some(b) => {
                ctx.introduce(&b.name, b.span)?;
                Some(b.name.clone())
            }
            None => None,
        };
        let edge_type = match rel.types.as_slice() {
            [single] => single.name.clone(),
            _ => {
                return Err(LowerError::new(
                    LowerErrorKind::InvalidPattern,
                    "CREATE relationship must have exactly one type",
                    rel.span,
                ));
            }
        };
        if matches!(rel.direction, ast::RelationshipDirection::Both) {
            return Err(LowerError::new(
                LowerErrorKind::InvalidPattern,
                "CREATE relationship must be directed (use `->` or `<-`)",
                rel.span,
            ));
        }
        if rel.length.is_some() {
            return Err(LowerError::new(
                LowerErrorKind::InvalidPattern,
                "CREATE does not accept variable-length patterns",
                rel.span,
            ));
        }
        let (properties, properties_spread) = lower_pattern_properties(&rel.properties);
        out.push(CreateElement::Rel {
            alias,
            edge_type,
            source_alias: source.clone(),
            target_alias: target_alias.clone(),
            direction: rel.direction,
            properties,
            properties_spread,
        });
        source = target_alias;
    }
    Ok(())
}

/// Convert the parser's optional [`PatternProperties`] into the
/// `(properties, properties_spread)` pair that `CreateElement` stores.
/// A literal map fans out into key/value entries; a `$param` reference
/// becomes the spread expression and entries stay empty. Used by both
/// node and relationship CREATE paths so the executor sees one shape.
fn lower_pattern_properties(
    props: &Option<ast::PatternProperties>,
) -> (Vec<(String, ast::Expression)>, Option<ast::Expression>) {
    match props {
        None => (Vec::new(), None),
        Some(ast::PatternProperties::Literal(map)) => (
            map.entries
                .iter()
                .map(|(k, v)| (k.name.clone(), v.clone()))
                .collect(),
            None,
        ),
        Some(ast::PatternProperties::Parameter { name, span }) => (
            Vec::new(),
            Some(ast::Expression {
                kind: ast::ExpressionKind::Parameter(name.clone()),
                span: *span,
            }),
        ),
    }
}

fn lower_create_node(
    node: &NodePattern,
    out: &mut Vec<CreateElement>,
    ctx: &mut LowerCtx,
) -> Result<String, LowerError> {
    // Back-reference: reuse an already-bound alias instead of producing a new node.
    if let Some(binding) = &node.binding {
        if ctx.bindings.contains(&binding.name) {
            return Ok(binding.name.clone());
        }
    }
    let alias = match &node.binding {
        Some(b) => {
            ctx.introduce(&b.name, b.span)?;
            b.name.clone()
        }
        None => {
            let candidate = anonymous_alias(ctx);
            ctx.introduce(&candidate, node.span)?;
            candidate
        }
    };
    let labels: Vec<String> = match node.labels.as_slice() {
        [] => {
            return Err(LowerError::new(
                LowerErrorKind::InvalidPattern,
                "CREATE node must carry at least one label",
                node.span,
            ));
        }
        ls => ls.iter().map(|l| l.name.clone()).collect(),
    };
    let (properties, properties_spread) = lower_pattern_properties(&node.properties);
    out.push(CreateElement::Node {
        alias: alias.clone(),
        labels,
        properties,
        properties_spread,
    });
    Ok(alias)
}

fn lower_merge(
    m: &ast::MergeClause,
    input: LogicalPlan,
    ctx: &mut LowerCtx,
) -> Result<LogicalPlan, LowerError> {
    if m.pattern.binding.is_some() {
        return Err(LowerError::new(
            LowerErrorKind::UnsupportedFeature,
            "path bindings inside MERGE are not yet supported",
            m.pattern.span,
        ));
    }
    let mut elements = Vec::new();
    lower_create_pattern_element(&m.pattern.element, &mut elements, ctx)?;
    let mut on_match_sets = Vec::new();
    let mut on_create_sets = Vec::new();
    for action in &m.actions {
        for item in &action.sets {
            let op = lower_set_item(item, ctx)?;
            match action.on {
                ast::MergeTrigger::Match => on_match_sets.push(op),
                ast::MergeTrigger::Create => on_create_sets.push(op),
            }
        }
    }
    Ok(LogicalPlan::Merge {
        input: Box::new(input),
        pattern: elements,
        on_match_sets,
        on_create_sets,
    })
}

// ─────────────────────────── SET / REMOVE / DELETE ────────────────────

fn lower_set(
    s: &ast::SetClause,
    input: LogicalPlan,
    ctx: &mut LowerCtx,
) -> Result<LogicalPlan, LowerError> {
    let mut items = Vec::with_capacity(s.items.len());
    for item in &s.items {
        items.push(lower_set_item(item, ctx)?);
    }
    Ok(LogicalPlan::Set {
        input: Box::new(input),
        items,
    })
}

fn lower_set_item(item: &ast::SetItem, ctx: &LowerCtx) -> Result<SetOp, LowerError> {
    match item {
        ast::SetItem::Property { target, value, .. } => {
            let alias = property_root_alias(target)?;
            ctx.ensure(&alias, target.target.span)?;
            check_expression_bindings(value, ctx)?;
            Ok(SetOp::Property {
                target_alias: alias,
                key: target.key.name.clone(),
                value: value.clone(),
            })
        }
        ast::SetItem::Replace { target, value, .. } => {
            ctx.ensure(&target.name, target.span)?;
            check_expression_bindings(value, ctx)?;
            Ok(SetOp::Replace {
                target_alias: target.name.clone(),
                value: value.clone(),
            })
        }
        ast::SetItem::Merge { target, value, .. } => {
            ctx.ensure(&target.name, target.span)?;
            check_expression_bindings(value, ctx)?;
            Ok(SetOp::Merge {
                target_alias: target.name.clone(),
                value: value.clone(),
            })
        }
        ast::SetItem::Labels { target, labels, .. } => {
            ctx.ensure(&target.name, target.span)?;
            Ok(SetOp::Labels {
                target_alias: target.name.clone(),
                labels: labels.iter().map(|l| l.name.clone()).collect(),
            })
        }
    }
}

fn property_root_alias(access: &ast::PropertyAccess) -> Result<String, LowerError> {
    match &access.target.kind {
        ExpressionKind::Variable(id) => Ok(id.name.clone()),
        _ => Err(LowerError::new(
            LowerErrorKind::UnsupportedFeature,
            "SET target must be a direct binding property (e.g. `a.prop = ...`)",
            access.span,
        )),
    }
}

fn lower_remove(
    r: &ast::RemoveClause,
    input: LogicalPlan,
    ctx: &mut LowerCtx,
) -> Result<LogicalPlan, LowerError> {
    let mut items = Vec::with_capacity(r.items.len());
    for item in &r.items {
        items.push(match item {
            ast::RemoveItem::Property(access) => {
                let alias = property_root_alias(access)?;
                ctx.ensure(&alias, access.target.span)?;
                RemoveOp::Property {
                    target_alias: alias,
                    key: access.key.name.clone(),
                }
            }
            ast::RemoveItem::Labels { target, labels, .. } => {
                ctx.ensure(&target.name, target.span)?;
                RemoveOp::Labels {
                    target_alias: target.name.clone(),
                    labels: labels.iter().map(|l| l.name.clone()).collect(),
                }
            }
        });
    }
    Ok(LogicalPlan::Remove {
        input: Box::new(input),
        items,
    })
}

fn lower_delete(
    d: &ast::DeleteClause,
    input: LogicalPlan,
    ctx: &mut LowerCtx,
) -> Result<LogicalPlan, LowerError> {
    for target in &d.targets {
        check_expression_bindings(target, ctx)?;
    }
    Ok(LogicalPlan::Delete {
        input: Box::new(input),
        targets: d.targets.clone(),
        detach: d.detach,
    })
}

// ─────────────────────────── helpers ─────────────────────────────────

fn check_expression_bindings(expr: &Expression, ctx: &LowerCtx) -> Result<(), LowerError> {
    match &expr.kind {
        ExpressionKind::Variable(id) => ctx.ensure(&id.name, id.span),
        ExpressionKind::Property(p) => check_expression_bindings(&p.target, ctx),
        ExpressionKind::Parameter(_) | ExpressionKind::Literal(_) | ExpressionKind::Star => Ok(()),
        ExpressionKind::Index { target, index } => {
            check_expression_bindings(target, ctx)?;
            check_expression_bindings(index, ctx)
        }
        ExpressionKind::Range { target, from, to } => {
            check_expression_bindings(target, ctx)?;
            if let Some(e) = from {
                check_expression_bindings(e, ctx)?;
            }
            if let Some(e) = to {
                check_expression_bindings(e, ctx)?;
            }
            Ok(())
        }
        ExpressionKind::Unary { expr, .. } => check_expression_bindings(expr, ctx),
        ExpressionKind::Binary { left, right, .. } => {
            check_expression_bindings(left, ctx)?;
            check_expression_bindings(right, ctx)
        }
        ExpressionKind::In { item, list } => {
            check_expression_bindings(item, ctx)?;
            check_expression_bindings(list, ctx)
        }
        ExpressionKind::StringTest {
            target, pattern, ..
        } => {
            check_expression_bindings(target, ctx)?;
            check_expression_bindings(pattern, ctx)
        }
        ExpressionKind::IsNull { expr, .. } => check_expression_bindings(expr, ctx),
        ExpressionKind::FunctionCall { args, .. } => {
            for a in args {
                check_expression_bindings(a, ctx)?;
            }
            Ok(())
        }
        ExpressionKind::Case {
            scrutinee,
            branches,
            otherwise,
        } => {
            if let Some(s) = scrutinee {
                check_expression_bindings(s, ctx)?;
            }
            for b in branches {
                check_expression_bindings(&b.when, ctx)?;
                check_expression_bindings(&b.then, ctx)?;
            }
            if let Some(e) = otherwise {
                check_expression_bindings(e, ctx)?;
            }
            Ok(())
        }
        ExpressionKind::Exists(_) => {
            // Pattern predicates are validated semantically (they
            // need their own scope visit). For now we accept them blind.
            Ok(())
        }
        ExpressionKind::List(items) => {
            for it in items {
                check_expression_bindings(it, ctx)?;
            }
            Ok(())
        }
        ExpressionKind::Map(m) => {
            for (_, v) in &m.entries {
                check_expression_bindings(v, ctx)?;
            }
            Ok(())
        }
        ExpressionKind::ListComprehension(lc) => {
            check_expression_bindings(&lc.list, ctx)?;
            // The bound variable is local — we don't add it to ctx because
            // the caller's scope shouldn't see it.
            Ok(())
        }
        ExpressionKind::PatternComprehension(_) => Ok(()),
    }
}

fn build_eq(alias: &str, key: &str, value: Expression, span: SourceSpan) -> Expression {
    let target = Expression {
        kind: ExpressionKind::Variable(ast::Identifier::new(alias, span)),
        span,
    };
    let property = Expression {
        kind: ExpressionKind::Property(Box::new(ast::PropertyAccess {
            target,
            key: ast::Identifier::new(key, span),
            span,
        })),
        span,
    };
    Expression {
        kind: ExpressionKind::Binary {
            op: ast::BinaryOp::Eq,
            left: Box::new(property),
            right: Box::new(value),
        },
        span,
    }
}

fn build_label_eq(alias: &str, label: &str, span: SourceSpan) -> Expression {
    // `labels(alias) STARTS WITH label`? No — semantics want label
    // membership. We encode as a pseudo-call `__label_eq(alias, "Label")`
    // and let the executor short-circuit. Future RFC may add an explicit
    // `labels(x)` runtime function.
    let target = Expression {
        kind: ExpressionKind::Variable(ast::Identifier::new(alias, span)),
        span,
    };
    let lit = Expression {
        kind: ExpressionKind::Literal(Literal::String(label.to_string())),
        span,
    };
    Expression {
        kind: ExpressionKind::FunctionCall {
            name: QualifiedName::single(ast::Identifier::new("__label_eq", span)),
            args: vec![target, lit],
            distinct: false,
        },
        span,
    }
}

fn combine(prev: Option<LogicalPlan>, current: LogicalPlan) -> LogicalPlan {
    match prev {
        None => current,
        Some(prev_plan) => match (prev_plan, current) {
            (LogicalPlan::Empty, c) => c,
            (p, LogicalPlan::Empty) => p,
            (p, c) => LogicalPlan::CrossProduct {
                left: Box::new(p),
                right: Box::new(c),
            },
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::parse;

    fn lp(src: &str) -> LogicalPlan {
        let q = parse(src).expect("parse");
        lower(&q).expect("lower")
    }

    fn err(src: &str) -> LowerErrorKind {
        let q = parse(src).expect("parse");
        lower(&q).expect_err("expected error").kind
    }

    #[test]
    fn match_return_lowers_to_project_over_nodescan() {
        let p = lp("MATCH (a:Person) RETURN a");
        match p {
            LogicalPlan::Project { input, items, .. } => {
                assert_eq!(items.len(), 1);
                assert_eq!(items[0].alias, "a");
                assert!(matches!(*input, LogicalPlan::NodeScan { .. }));
            }
            other => panic!("expected Project, got {:?}", other),
        }
    }

    #[test]
    fn match_with_underscore_id_lookup_lowers_to_node_by_id() {
        // The internal NodeId sigil is `_id` (Bug #1 rename). The legacy
        // plain `id` key is no longer reserved.
        let p = lp("MATCH (a:Person {_id: $personId}) RETURN a");
        match p {
            LogicalPlan::Project { input, .. } => match *input {
                LogicalPlan::NodeById { label, alias, .. } => {
                    assert_eq!(label.as_deref(), Some("Person"));
                    assert_eq!(alias, "a");
                }
                other => panic!("expected NodeById, got {:?}", other),
            },
            _ => panic!("expected Project"),
        }
    }

    #[test]
    fn match_with_plain_id_property_lowers_to_filter_not_node_by_id() {
        // Regression for Bug #1: `id` is a user property now. `{id: ...}`
        // must NOT trigger a NodeById point-lookup; it should fall through
        // to NodeScan + Filter so the engine treats it like any other prop.
        let p = lp("MATCH (a:Person {id: 'external-42'}) RETURN a");
        fn has_node_by_id(plan: &LogicalPlan) -> bool {
            match plan {
                LogicalPlan::NodeById { .. } => true,
                _ => plan.children().iter().any(|c| has_node_by_id(c)),
            }
        }
        assert!(
            !has_node_by_id(&p),
            "expected no NodeById in plan for plain `id` property, got {:?}",
            p,
        );
    }

    #[test]
    fn return_with_order_and_limit_lowers_to_project_over_top_n() {
        // Plan order: NodeScan → TopN(keys, limit=10) → Project(items)
        // — see RFC-008 §"Lowering rules". TopN evalúa keys contra
        // bindings originales (pre-Project) o aliases sustituidos.
        let p = lp("MATCH (a:Person) RETURN a.name AS n ORDER BY a.age DESC LIMIT 10");
        match p {
            LogicalPlan::Project { input, .. } => match *input {
                LogicalPlan::TopN {
                    keys, skip, limit, ..
                } => {
                    assert_eq!(skip, RowCount::Const(0));
                    assert_eq!(limit, RowCount::Const(10));
                    assert_eq!(keys.len(), 1);
                }
                other => panic!("expected TopN under Project, got {:?}", other),
            },
            other => panic!("expected Project, got {:?}", other),
        }
    }

    #[test]
    fn skip_limit_parameters_lower_to_row_count_param() {
        // `SKIP $s LIMIT $l` carries the param names into the plan (resolved
        // at execution time), instead of being rejected as it was in v0.
        let p = lp("MATCH (a:Person) RETURN a SKIP $s LIMIT $l");
        match p {
            LogicalPlan::Project { input, .. } => match *input {
                LogicalPlan::TopN { skip, limit, .. } => {
                    assert_eq!(skip, RowCount::Param("s".into()));
                    assert_eq!(limit, RowCount::Param("l".into()));
                }
                other => panic!("expected TopN under Project, got {:?}", other),
            },
            other => panic!("expected Project, got {:?}", other),
        }
    }

    #[test]
    fn where_lowers_to_filter() {
        let p = lp("MATCH (a:Person) WHERE a.age > 18 RETURN a");
        match p {
            LogicalPlan::Project { input, .. } => match *input {
                LogicalPlan::Filter { .. } => {}
                other => panic!("expected Filter under Project, got {:?}", other),
            },
            _ => panic!(),
        }
    }

    #[test]
    fn expand_chain_two_hops() {
        let p = lp("MATCH (a:Person)-[:KNOWS]->(b:Person)-[:LIKES]->(c:Person) RETURN c");
        // Project → TopN absent → Project → Expand(b→c) → Expand(a→b) → NodeScan(a)
        match p {
            LogicalPlan::Project { input, .. } => match *input {
                LogicalPlan::Filter { input, .. } => match *input {
                    LogicalPlan::Expand { target_alias, .. } => assert_eq!(target_alias, "c"),
                    other => panic!("expected Expand under Filter, got {:?}", other),
                },
                LogicalPlan::Expand { target_alias, .. } => assert_eq!(target_alias, "c"),
                other => panic!("expected Expand/Filter under Project, got {:?}", other),
            },
            _ => panic!(),
        }
    }

    #[test]
    fn binding_not_in_scope_errors() {
        let kind = err("MATCH (a:Person) RETURN b");
        assert_eq!(kind, LowerErrorKind::BindingNotFound);
    }

    #[test]
    fn create_clause_lowers_to_create_operator() {
        let p = lp("CREATE (a:Person {name: 'Ada'}) RETURN a");
        // Top: Project, then Create over Empty.
        match p {
            LogicalPlan::Project { input, .. } => match *input {
                LogicalPlan::Create { input, elements } => {
                    assert!(matches!(*input, LogicalPlan::Empty));
                    assert_eq!(elements.len(), 1);
                    match &elements[0] {
                        CreateElement::Node {
                            alias,
                            labels,
                            properties,
                            properties_spread,
                        } => {
                            assert_eq!(alias, "a");
                            assert_eq!(labels.first().map(String::as_str), Some("Person"));
                            assert_eq!(properties.len(), 1);
                            assert_eq!(properties[0].0, "name");
                            assert!(properties_spread.is_none());
                        }
                        other => panic!("expected Node, got {:?}", other),
                    }
                }
                other => panic!("expected Create, got {:?}", other),
            },
            other => panic!("expected Project, got {:?}", other),
        }
    }

    #[test]
    fn match_then_create_chains_plans() {
        let p = lp("MATCH (a:Person {id: $aid}), (b:Person {id: $bid}) \
 CREATE (a)-[:KNOWS]->(b)");
        // Walk the tree; the last operator should be Create.
        fn walk_to_create(plan: &LogicalPlan) -> Option<&LogicalPlan> {
            if matches!(plan, LogicalPlan::Create { .. }) {
                Some(plan)
            } else {
                plan.children().iter().find_map(|c| walk_to_create(c))
            }
        }
        let create = walk_to_create(&p).expect("Create operator");
        match create {
            LogicalPlan::Create { elements, .. } => {
                // Two back-ref nodes (a, b) reused → no new Node elements
                // for them. The chain emits the Rel element only.
                assert!(
                    elements
                        .iter()
                        .any(|e| matches!(e, CreateElement::Rel { .. })),
                    "expected a Rel CreateElement: {:?}",
                    elements
                );
            }
            _ => unreachable!(),
        }
    }

    #[test]
    fn set_clause_lowers_to_set_op() {
        let p = lp("MATCH (a:Person) SET a.age = 36");
        match &p {
            LogicalPlan::Set { items, .. } => {
                assert_eq!(items.len(), 1);
                match &items[0] {
                    SetOp::Property {
                        target_alias, key, ..
                    } => {
                        assert_eq!(target_alias, "a");
                        assert_eq!(key, "age");
                    }
                    other => panic!("expected SetOp::Property, got {:?}", other),
                }
            }
            other => panic!("expected Set, got {:?}", other),
        }
    }

    #[test]
    fn detach_delete_clause_lowers_with_flag() {
        let p = lp("MATCH (a:Person) DETACH DELETE a");
        match &p {
            LogicalPlan::Delete {
                detach, targets, ..
            } => {
                assert!(detach);
                assert_eq!(targets.len(), 1);
            }
            other => panic!("expected Delete, got {:?}", other),
        }
    }

    #[test]
    fn remove_clause_lowers_to_remove_op() {
        let p = lp("MATCH (a:Person) REMOVE a.age");
        match &p {
            LogicalPlan::Remove { items, .. } => {
                assert_eq!(items.len(), 1);
                assert!(matches!(items[0], RemoveOp::Property { .. }));
            }
            other => panic!("expected Remove, got {:?}", other),
        }
    }

    #[test]
    fn merge_clause_lowers_with_actions() {
        let p = lp("MERGE (a:Person {id: $id}) \
 ON CREATE SET a.firstSeen = 1 \
 ON MATCH SET a.lastSeen = 1");
        match &p {
            LogicalPlan::Merge {
                pattern,
                on_create_sets,
                on_match_sets,
                ..
            } => {
                assert_eq!(pattern.len(), 1);
                assert_eq!(on_create_sets.len(), 1);
                assert_eq!(on_match_sets.len(), 1);
            }
            other => panic!("expected Merge, got {:?}", other),
        }
    }

    #[test]
    fn merge_multi_hop_lowers_with_all_endpoints_and_rels() {
        // B2: a MERGE pattern with multiple hops must lower every node
        // and every relationship in the chain.
        let p = lp(
            "MERGE (a:Person {externalId: 1})-[r1:KNOWS]->(b:Person {externalId: 2})\
             -[r2:KNOWS]->(c:Person {externalId: 3})",
        );
        let pattern = match &p {
            LogicalPlan::Merge { pattern, .. } => pattern,
            other => panic!("expected Merge, got {:?}", other),
        };
        let mut node_aliases: Vec<&str> = pattern
            .iter()
            .filter_map(|e| match e {
                CreateElement::Node { alias, .. } => Some(alias.as_str()),
                _ => None,
            })
            .collect();
        node_aliases.sort();
        assert_eq!(node_aliases, vec!["a", "b", "c"]);
        let mut rel_aliases: Vec<&str> = pattern
            .iter()
            .filter_map(|e| match e {
                CreateElement::Rel { alias: Some(a), .. } => Some(a.as_str()),
                _ => None,
            })
            .collect();
        rel_aliases.sort();
        assert_eq!(rel_aliases, vec!["r1", "r2"]);
    }

    #[test]
    fn merge_with_relationship_lowers_to_node_rel_node_set() {
        // Regression: MERGE (a)-[r]->(b) must lower a triple containing
        // a head Node, a tail Node, and a Rel linking their aliases.
        // The order inside `pattern` follows the CREATE-friendly layout
        // (so apply_create can resolve endpoints sequentially); the
        // MERGE executor reads by alias, not by position.
        let p = lp("MERGE (a:Person {externalId: 1})-[r:KNOWS]->(b:Person {externalId: 2})");
        let pattern = match &p {
            LogicalPlan::Merge { pattern, .. } => pattern,
            other => panic!("expected Merge, got {:?}", other),
        };
        assert_eq!(pattern.len(), 3, "pattern: {:?}", pattern);
        let mut node_aliases: Vec<&str> = pattern
            .iter()
            .filter_map(|e| match e {
                CreateElement::Node { alias, .. } => Some(alias.as_str()),
                _ => None,
            })
            .collect();
        node_aliases.sort();
        assert_eq!(node_aliases, vec!["a", "b"]);
        let rel = pattern
            .iter()
            .find_map(|e| match e {
                CreateElement::Rel {
                    source_alias,
                    target_alias,
                    edge_type,
                    ..
                } => Some((
                    source_alias.as_str(),
                    target_alias.as_str(),
                    edge_type.as_str(),
                )),
                _ => None,
            })
            .expect("Rel element present");
        assert_eq!(rel, ("a", "b", "KNOWS"));
    }

    #[test]
    fn unwind_introduces_alias() {
        let p = lp("UNWIND [1, 2, 3] AS x RETURN x");
        match p {
            LogicalPlan::Project { input, .. } => {
                assert!(matches!(*input, LogicalPlan::Unwind { .. }));
            }
            _ => panic!(),
        }
    }

    #[test]
    fn unwind_alias_binds_in_following_match_pattern() {
        // B1: `UNWIND … AS x MATCH (n {id: x})` — the unwind alias must
        // be visible to the property filter on the following MATCH; if it
        // isn't, lowering fails with "unknown identifier `x`".
        let _p = lp("UNWIND ['a', 'b'] AS uid MATCH (n:Person {id: uid}) RETURN n");
    }

    #[test]
    fn unwind_alias_binds_in_match_with_where() {
        let _p = lp("UNWIND ['a', 'b'] AS uid MATCH (n:Person) WHERE n.id = uid RETURN n");
    }

    #[test]
    fn unwind_alias_binds_in_match_with_id_filter() {
        let _p = lp("UNWIND [$ids] AS uid MATCH (n:Person {_id: uid}) RETURN n");
    }

    #[test]
    fn unwind_alias_binds_in_chained_match_expand() {
        let _p = lp("UNWIND ['a'] AS uid MATCH (n:Person {id: uid})-[:KNOWS]->(m:Person) RETURN m");
    }

    #[test]
    fn aggregate_count_inserts_aggregate_node() {
        let p = lp("MATCH (a:Person)-[:KNOWS]->(b:Person) RETURN a, count(b) AS friends");
        match p {
            LogicalPlan::Project { input, items, .. } => {
                // Top-level Project carries the user-visible alias.
                let aliases: Vec<&str> = items.iter().map(|i| i.alias.as_str()).collect();
                assert!(aliases.contains(&"friends"));
                match *input {
                    LogicalPlan::Aggregate { aggregations, .. } => {
                        assert_eq!(aggregations.len(), 1);
                        // The synthetic alias name is an implementation detail.
                        assert!(aggregations[0].0.starts_with("__agg"));
                        assert!(matches!(aggregations[0].1, AggregateExpr::Count { .. }));
                    }
                    other => panic!("expected Aggregate, got {:?}", other),
                }
            }
            _ => panic!(),
        }
    }

    #[test]
    fn pattern_comprehension_hoists_to_pattern_list() {
        let p = lp("MATCH (a:Person) \
 RETURN a, [(a)-[:KNOWS]->(b:Person) | b.name] AS friends");
        // Top: Project; below: PatternList wrapping NodeScan(a).
        let inner = match p {
            LogicalPlan::Project { input, items, .. } => {
                let aliases: Vec<&str> = items.iter().map(|i| i.alias.as_str()).collect();
                assert!(aliases.contains(&"friends"));
                *input
            }
            other => panic!("expected Project, got {:?}", other),
        };
        match inner {
            LogicalPlan::PatternList {
                alias,
                input,
                subplan,
                ..
            } => {
                assert!(alias.starts_with("__pc"));
                assert!(matches!(*input, LogicalPlan::NodeScan { .. }));
                fn has_argument(plan: &LogicalPlan) -> bool {
                    matches!(plan, LogicalPlan::Argument { .. })
                        || plan.children().iter().any(|c| has_argument(c))
                }
                assert!(has_argument(&subplan));
            }
            other => panic!("expected PatternList, got {:?}", other),
        }
    }

    #[test]
    fn pattern_comprehension_with_predicate_lowers_with_filter() {
        let p = lp("MATCH (a:Person) \
 RETURN a, [(a)-[:KNOWS]->(b:Person) WHERE b.age > 30 | b.name] AS f");
        let inner = match p {
            LogicalPlan::Project { input, .. } => *input,
            _ => panic!(),
        };
        match inner {
            LogicalPlan::PatternList { subplan, .. } => {
                fn has_filter(plan: &LogicalPlan) -> bool {
                    matches!(plan, LogicalPlan::Filter { .. })
                        || plan.children().iter().any(|c| has_filter(c))
                }
                assert!(has_filter(&subplan));
            }
            _ => panic!(),
        }
    }

    #[test]
    fn nested_pattern_comprehension_errors() {
        let kind = err("MATCH (a:Person) \
 RETURN size([(a)-[:KNOWS]->(b:Person) | b.name]) AS friend_count");
        assert_eq!(kind, LowerErrorKind::UnsupportedFeature);
    }

    #[test]
    fn exists_predicate_lowers_to_semiapply() {
        let p = lp("MATCH (a:Person) WHERE EXISTS((a)-[:KNOWS]->(b:Person)) RETURN a");
        // Project -> SemiApply{!neg} -> NodeScan(a)
        let inner = match p {
            LogicalPlan::Project { input, .. } => *input,
            other => panic!("expected Project, got {:?}", other),
        };
        match inner {
            LogicalPlan::SemiApply {
                input,
                subplan,
                negated,
            } => {
                assert!(!negated);
                assert!(matches!(*input, LogicalPlan::NodeScan { .. }));
                // The subplan should start (somewhere) with Argument(a).
                fn has_argument_with(plan: &LogicalPlan, want: &str) -> bool {
                    match plan {
                        LogicalPlan::Argument { bindings } => bindings.iter().any(|b| b == want),
                        _ => plan.children().iter().any(|c| has_argument_with(c, want)),
                    }
                }
                assert!(has_argument_with(&subplan, "a"));
            }
            other => panic!("expected SemiApply, got {:?}", other),
        }
    }

    #[test]
    fn not_exists_predicate_lowers_to_anti_semiapply() {
        let p = lp("MATCH (a:Person) WHERE NOT EXISTS((a)-[:KNOWS]->(b:Person)) RETURN a");
        let inner = match p {
            LogicalPlan::Project { input, .. } => *input,
            other => panic!("expected Project, got {:?}", other),
        };
        match inner {
            LogicalPlan::SemiApply { negated, .. } => assert!(negated),
            other => panic!("expected SemiApply, got {:?}", other),
        }
    }

    #[test]
    fn exists_and_residual_predicate_lower_to_semiapply_plus_filter() {
        let p =
            lp("MATCH (a:Person) WHERE EXISTS((a)-[:KNOWS]->(b:Person)) AND a.age > 18 RETURN a");
        // Expected order: Project -> Filter(a.age > 18) -> SemiApply -> NodeScan(a)
        let inner = match p {
            LogicalPlan::Project { input, .. } => *input,
            other => panic!("expected Project, got {:?}", other),
        };
        match inner {
            LogicalPlan::Filter { input, .. } => match *input {
                LogicalPlan::SemiApply { .. } => {}
                other => panic!("expected SemiApply under Filter, got {:?}", other),
            },
            other => panic!("expected Filter on top, got {:?}", other),
        }
    }

    #[test]
    fn multiple_exists_terms_chain_semiapplies() {
        let p = lp("MATCH (a:Person) \
 WHERE EXISTS((a)-[:KNOWS]->(b:Person)) AND EXISTS((a)-[:LIKES]->(c:Person)) \
 RETURN a");
        let inner = match p {
            LogicalPlan::Project { input, .. } => *input,
            other => panic!("expected Project, got {:?}", other),
        };
        match inner {
            LogicalPlan::SemiApply { input, .. } => match *input {
                LogicalPlan::SemiApply { .. } => {}
                other => panic!("expected nested SemiApply, got {:?}", other),
            },
            other => panic!("expected outer SemiApply, got {:?}", other),
        }
    }

    #[test]
    fn return_star_expands_to_all_visible_bindings() {
        let p = lp("MATCH (a:Person)-[:KNOWS]->(b:Person) RETURN *");
        match p {
            LogicalPlan::Project { items, .. } => {
                let aliases: Vec<&str> = items.iter().map(|i| i.alias.as_str()).collect();
                assert!(aliases.contains(&"a"), "missing a in {:?}", aliases);
                assert!(aliases.contains(&"b"), "missing b in {:?}", aliases);
                assert_eq!(items.len(), 2);
            }
            other => panic!("expected Project, got {:?}", other),
        }
    }

    #[test]
    fn return_star_skips_anonymous_bindings() {
        let p = lp("MATCH (a:Person)-[:KNOWS]->(:Person) RETURN *");
        match p {
            LogicalPlan::Project { items, .. } => {
                let aliases: Vec<&str> = items.iter().map(|i| i.alias.as_str()).collect();
                assert_eq!(aliases, vec!["a"]);
            }
            other => panic!("expected Project, got {:?}", other),
        }
    }

    #[test]
    fn return_star_alone_with_no_visible_bindings_errors() {
        let kind = err("MATCH (:Person) RETURN *");
        assert_eq!(kind, LowerErrorKind::InvalidPattern);
    }

    #[test]
    fn return_star_mixed_with_explicit_item() {
        let p = lp("MATCH (a:Person)-[:KNOWS]->(b:Person) RETURN *, a.name AS n");
        match p {
            LogicalPlan::Project { items, .. } => {
                let aliases: Vec<&str> = items.iter().map(|i| i.alias.as_str()).collect();
                assert!(aliases.contains(&"a"));
                assert!(aliases.contains(&"b"));
                assert!(aliases.contains(&"n"));
                assert_eq!(items.len(), 3);
            }
            _ => panic!(),
        }
    }

    #[test]
    fn optional_match_marks_expand_optional() {
        let p = lp("MATCH (a:Person) OPTIONAL MATCH (a)-[:KNOWS]->(b:Person) RETURN a, b");
        // top is Project; under it is OptionalExpand or Filter wrapping it.
        let project_input = match p {
            LogicalPlan::Project { input, .. } => *input,
            other => panic!("expected Project, got {:?}", other),
        };
        fn has_optional_expand(plan: &LogicalPlan) -> bool {
            match plan {
                LogicalPlan::Expand { optional: true, .. } => true,
                _ => plan.children().iter().any(|c| has_optional_expand(c)),
            }
        }
        assert!(has_optional_expand(&project_input));
    }

    #[test]
    fn match_without_label_lowers_to_typeless_node_scan() {
        // Regression for Bug #3: `MATCH (n)` (no label predicate) used to
        // be rejected by `require_single_label`. Lowering must now produce
        // a `NodeScan { label: None, ... }` so the executor can fan out
        // across observed labels.
        let p = lp("MATCH (n) RETURN n");
        let project_input = match p {
            LogicalPlan::Project { input, .. } => *input,
            other => panic!("expected Project, got {:?}", other),
        };
        match project_input {
            LogicalPlan::NodeScan { label, alias, .. } => {
                assert!(
                    label.is_none(),
                    "expected typeless NodeScan, got label={:?}",
                    label
                );
                assert_eq!(alias, "n");
            }
            other => panic!("expected NodeScan under Project, got {:?}", other),
        }
    }

    #[test]
    fn two_match_clauses_lower_to_cross_product() {
        // Two separate MATCH clauses with no shared binding must join via
        // CrossProduct so downstream clauses (CREATE/RETURN) see both
        // sides of bindings. Regression: `combine` used to drop `prev`.
        let p = lp("MATCH (a:Person) MATCH (b:Person) RETURN a, b");
        let project_input = match p {
            LogicalPlan::Project { input, .. } => *input,
            other => panic!("expected Project, got {:?}", other),
        };
        fn has_cross_product(plan: &LogicalPlan) -> bool {
            match plan {
                LogicalPlan::CrossProduct { .. } => true,
                _ => plan.children().iter().any(|c| has_cross_product(c)),
            }
        }
        assert!(
            has_cross_product(&project_input),
            "expected CrossProduct under Project, got {:?}",
            project_input,
        );
    }
}
