//! Single-line canonical printer for the Cypher AST.
//!
//! `format!("{}", query)` produces a Cypher source equivalent to the parsed
//! query. The round-trip `parse → format → parse` must yield the same AST
//! modulo spans. Indented pretty-printing is deferred (will plug into
//! `EXPLAIN`).

use std::fmt;

use super::ast::*;

impl fmt::Display for Query {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.explain {
            f.write_str("EXPLAIN ")?;
            if self.explain_raw {
                f.write_str("RAW ")?;
            }
            if self.explain_verbose {
                f.write_str("VERBOSE ")?;
            }
        }
        write!(f, "{}", self.head)?;
        for part in &self.tail {
            if part.all {
                write!(f, " UNION ALL {}", part.query)?;
            } else {
                write!(f, " UNION {}", part.query)?;
            }
        }
        Ok(())
    }
}

impl fmt::Display for SingleQuery {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (i, clause) in self.clauses.iter().enumerate() {
            if i > 0 {
                f.write_str(" ")?;
            }
            write!(f, "{}", clause)?;
        }
        Ok(())
    }
}

impl fmt::Display for Clause {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Clause::Match(m) => fmt::Display::fmt(m, f),
            Clause::Return(r) => fmt::Display::fmt(r, f),
            Clause::With(w) => fmt::Display::fmt(w, f),
            Clause::Where(w) => write!(f, "WHERE {}", w.predicate),
            Clause::Unwind(u) => fmt::Display::fmt(u, f),
            Clause::Create(c) => fmt::Display::fmt(c, f),
            Clause::Merge(m) => fmt::Display::fmt(m, f),
            Clause::Set(s) => fmt::Display::fmt(s, f),
            Clause::Remove(r) => fmt::Display::fmt(r, f),
            Clause::Delete(d) => fmt::Display::fmt(d, f),
            Clause::CreateVectorIndex(c) => fmt::Display::fmt(c, f),
            Clause::CreateFulltextIndex(c) => fmt::Display::fmt(c, f),
            Clause::CreateConstraint(c) => fmt::Display::fmt(c, f),
            Clause::CreateIndex(c) => fmt::Display::fmt(c, f),
            Clause::Foreach(c) => fmt::Display::fmt(c, f),
            Clause::Call(c) => fmt::Display::fmt(c, f),
        }
    }
}

impl fmt::Display for MatchClause {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.optional {
            f.write_str("OPTIONAL MATCH ")?;
        } else {
            f.write_str("MATCH ")?;
        }
        write_list(f, &self.patterns, ", ")?;
        if let Some(w) = &self.where_ {
            write!(f, " WHERE {}", w)?;
        }
        Ok(())
    }
}

impl fmt::Display for ReturnClause {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("RETURN ")?;
        if self.distinct {
            f.write_str("DISTINCT ")?;
        }
        write_list(f, &self.items, ", ")?;
        write_return_tail(f, &self.order_by, &self.skip, &self.limit)
    }
}

impl fmt::Display for WithClause {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("WITH ")?;
        if self.distinct {
            f.write_str("DISTINCT ")?;
        }
        write_list(f, &self.items, ", ")?;
        write_return_tail(f, &self.order_by, &self.skip, &self.limit)?;
        if let Some(w) = &self.where_ {
            write!(f, " WHERE {}", w)?;
        }
        Ok(())
    }
}

impl fmt::Display for UnwindClause {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "UNWIND {} AS {}", self.list, self.alias)
    }
}

impl fmt::Display for CreateClause {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("CREATE ")?;
        write_list(f, &self.patterns, ", ")
    }
}

impl fmt::Display for VectorMetric {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_keyword())
    }
}

impl fmt::Display for CreateVectorIndexClause {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "CREATE VECTOR INDEX {} ON :{}({}) METRIC {} DIMENSION {}",
            self.name, self.label, self.property, self.metric, self.dim
        )?;
        // Render WITH only when at least one build override is present, so a
        // defaults-only index round-trips without a trailing `WITH {}`.
        if self.r.is_some() || self.l_build.is_some() || self.alpha.is_some() {
            f.write_str(" WITH {")?;
            let mut parts: Vec<String> = Vec::new();
            if let Some(v) = self.r {
                parts.push(format!("r: {v}"));
            }
            if let Some(v) = self.l_build {
                parts.push(format!("l_build: {v}"));
            }
            if let Some(v) = self.alpha {
                parts.push(format!("alpha: {v}"));
            }
            write!(f, "{}", parts.join(", "))?;
            f.write_str("}")?;
        }
        Ok(())
    }
}

impl fmt::Display for CreateFulltextIndexClause {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let props = self
            .properties
            .iter()
            .map(|p| p.to_string())
            .collect::<Vec<_>>()
            .join(", ");
        write!(
            f,
            "CREATE FULLTEXT INDEX {} ON :{}({})",
            self.name, self.label, props
        )
    }
}

impl fmt::Display for CreateConstraintClause {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("CREATE CONSTRAINT ")?;
        if let Some(n) = &self.name {
            write!(f, "{n} ")?;
        }
        write!(
            f,
            "FOR (n:{}) REQUIRE n.{} IS UNIQUE",
            self.label, self.property
        )
    }
}

impl fmt::Display for CreateIndexClause {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("CREATE INDEX ")?;
        if let Some(n) = &self.name {
            write!(f, "{n} ")?;
        }
        write!(f, "FOR (n:{}) ON (n.{})", self.label, self.property)
    }
}

impl fmt::Display for ForeachClause {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "FOREACH ({} IN {} | ", self.variable, self.list)?;
        for (i, c) in self.body.iter().enumerate() {
            if i > 0 {
                f.write_str(" ")?;
            }
            write!(f, "{c}")?;
        }
        f.write_str(")")
    }
}

impl fmt::Display for CallClause {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("CALL ")?;
        if let Some(ns) = &self.namespace {
            write!(f, "{ns}.")?;
        }
        write!(f, "{}(", self.name)?;
        for (i, a) in self.args.iter().enumerate() {
            if i > 0 {
                f.write_str(", ")?;
            }
            write!(f, "{a}")?;
        }
        f.write_str(")")?;
        if !self.yield_items.is_empty() {
            f.write_str(" YIELD ")?;
            for (i, it) in self.yield_items.iter().enumerate() {
                if i > 0 {
                    f.write_str(", ")?;
                }
                write!(f, "{}", it.name)?;
                if let Some(a) = &it.alias {
                    write!(f, " AS {a}")?;
                }
            }
        }
        Ok(())
    }
}

impl fmt::Display for MergeClause {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "MERGE {}", self.pattern)?;
        for action in &self.actions {
            write!(f, " {}", action)?;
        }
        Ok(())
    }
}

impl fmt::Display for MergeAction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let trigger = match self.on {
            MergeTrigger::Create => "ON CREATE",
            MergeTrigger::Match => "ON MATCH",
        };
        write!(f, "{} SET ", trigger)?;
        write_list(f, &self.sets, ", ")
    }
}

impl fmt::Display for SetClause {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("SET ")?;
        write_list(f, &self.items, ", ")
    }
}

impl fmt::Display for SetItem {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SetItem::Property { target, value, .. } => write!(f, "{} = {}", target, value),
            SetItem::Replace { target, value, .. } => write!(f, "{} = {}", target, value),
            SetItem::Merge { target, value, .. } => write!(f, "{} += {}", target, value),
            SetItem::Labels { target, labels, .. } => {
                write!(f, "{}", target)?;
                for l in labels {
                    write!(f, ":{}", l)?;
                }
                Ok(())
            }
        }
    }
}

impl fmt::Display for RemoveClause {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("REMOVE ")?;
        for (i, item) in self.items.iter().enumerate() {
            if i > 0 {
                f.write_str(", ")?;
            }
            match item {
                RemoveItem::Property(p) => write!(f, "{}", p)?,
                RemoveItem::Labels { target, labels, .. } => {
                    write!(f, "{}", target)?;
                    for l in labels {
                        write!(f, ":{}", l)?;
                    }
                }
            }
        }
        Ok(())
    }
}

impl fmt::Display for DeleteClause {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.detach {
            f.write_str("DETACH DELETE ")?;
        } else {
            f.write_str("DELETE ")?;
        }
        write_list(f, &self.targets, ", ")
    }
}

impl fmt::Display for ProjectionItem {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.expression)?;
        if let Some(alias) = &self.alias {
            write!(f, " AS {}", alias)?;
        }
        Ok(())
    }
}

impl fmt::Display for OrderItem {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.expression)?;
        match self.direction {
            OrderDirection::Asc => Ok(()),
            OrderDirection::Desc => f.write_str(" DESC"),
        }
    }
}

impl fmt::Display for PatternPart {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(b) = &self.binding {
            write!(f, "{} = ", b)?;
        }
        write!(f, "{}", self.element)
    }
}

impl fmt::Display for PatternElement {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.head)?;
        for (rel, node) in &self.chain {
            write!(f, "{}{}", rel, node)?;
        }
        Ok(())
    }
}

impl fmt::Display for NodePattern {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("(")?;
        if let Some(b) = &self.binding {
            write!(f, "{}", b)?;
        }
        for l in &self.labels {
            write!(f, ":{}", l)?;
        }
        if let Some(p) = &self.properties {
            write!(f, " {}", p)?;
        }
        f.write_str(")")
    }
}

impl fmt::Display for PatternProperties {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PatternProperties::Literal(m) => write!(f, "{}", m),
            PatternProperties::Parameter { name, .. } => write!(f, "${}", name),
        }
    }
}

impl fmt::Display for RelationshipPattern {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let lead = match self.direction {
            RelationshipDirection::Left => "<-",
            _ => "-",
        };
        let trail = match self.direction {
            RelationshipDirection::Right => "->",
            _ => "-",
        };
        f.write_str(lead)?;
        let has_detail = self.binding.is_some()
            || !self.types.is_empty()
            || self.length.is_some()
            || self.properties.is_some();
        if has_detail {
            f.write_str("[")?;
            if let Some(b) = &self.binding {
                write!(f, "{}", b)?;
            }
            for (i, t) in self.types.iter().enumerate() {
                if i == 0 {
                    f.write_str(":")?;
                } else {
                    f.write_str("|")?;
                }
                write!(f, "{}", t)?;
            }
            if let Some(len) = self.length {
                if len.min == len.max {
                    write!(f, "*{}", len.min)?;
                } else {
                    write!(f, "*{}..{}", len.min, len.max)?;
                }
            }
            if let Some(p) = &self.properties {
                write!(f, " {}", p)?;
            }
            f.write_str("]")?;
        }
        f.write_str(trail)
    }
}

impl fmt::Display for Expression {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.kind {
            ExpressionKind::Literal(l) => write!(f, "{}", l),
            ExpressionKind::Variable(id) => write!(f, "{}", id),
            ExpressionKind::Parameter(name) => write!(f, "${}", name),
            ExpressionKind::Property(p) => write!(f, "{}", p),
            ExpressionKind::Index { target, index } => write!(f, "{}[{}]", target, index),
            ExpressionKind::Range { target, from, to } => {
                write!(f, "{}[", target)?;
                if let Some(a) = from {
                    write!(f, "{}", a)?;
                }
                f.write_str("..")?;
                if let Some(b) = to {
                    write!(f, "{}", b)?;
                }
                f.write_str("]")
            }
            ExpressionKind::Unary { op, expr } => match op {
                UnaryOp::Neg => write!(f, "-({})", expr),
                UnaryOp::Not => write!(f, "NOT ({})", expr),
            },
            ExpressionKind::Binary { op, left, right } => {
                write!(f, "({} {} {})", left, binary_op_symbol(*op), right)
            }
            ExpressionKind::In { item, list } => write!(f, "({} IN {})", item, list),
            ExpressionKind::StringTest {
                op,
                target,
                pattern,
            } => {
                let kw = match op {
                    StringOp::StartsWith => "STARTS WITH",
                    StringOp::EndsWith => "ENDS WITH",
                    StringOp::Contains => "CONTAINS",
                };
                write!(f, "({} {} {})", target, kw, pattern)
            }
            ExpressionKind::IsNull { expr, negated } => {
                if *negated {
                    write!(f, "({} IS NOT NULL)", expr)
                } else {
                    write!(f, "({} IS NULL)", expr)
                }
            }
            ExpressionKind::FunctionCall {
                name,
                args,
                distinct,
            } => {
                write!(f, "{}(", name.joined())?;
                if *distinct {
                    f.write_str("DISTINCT ")?;
                }
                write_list(f, args, ", ")?;
                f.write_str(")")
            }
            ExpressionKind::Case {
                scrutinee,
                branches,
                otherwise,
            } => {
                f.write_str("CASE")?;
                if let Some(s) = scrutinee {
                    write!(f, " {}", s)?;
                }
                for b in branches {
                    write!(f, " WHEN {} THEN {}", b.when, b.then)?;
                }
                if let Some(e) = otherwise {
                    write!(f, " ELSE {}", e)?;
                }
                f.write_str(" END")
            }
            ExpressionKind::Exists(p) => write!(f, "exists({})", p),
            ExpressionKind::ExistsSubquery(mc) => write!(f, "EXISTS {{ {mc} }}"),
            ExpressionKind::List(items) => {
                f.write_str("[")?;
                write_list(f, items, ", ")?;
                f.write_str("]")
            }
            ExpressionKind::Map(m) => write!(f, "{}", m),
            ExpressionKind::ListComprehension(lc) => {
                write!(f, "[{} IN {}", lc.variable, lc.list)?;
                if let Some(p) = &lc.predicate {
                    write!(f, " WHERE {}", p)?;
                }
                if let Some(p) = &lc.projection {
                    write!(f, " | {}", p)?;
                }
                f.write_str("]")
            }
            ExpressionKind::PatternComprehension(pc) => {
                f.write_str("[")?;
                write!(f, "{}", pc.pattern)?;
                if let Some(p) = &pc.predicate {
                    write!(f, " WHERE {}", p)?;
                }
                write!(f, " | {}]", pc.projection)
            }
            ExpressionKind::Quantifier(q) => {
                let kw = match q.kind {
                    QuantifierKind::All => "all",
                    QuantifierKind::Any => "any",
                    QuantifierKind::None => "none",
                    QuantifierKind::Single => "single",
                };
                write!(
                    f,
                    "{}({} IN {} WHERE {})",
                    kw, q.variable, q.list, q.predicate
                )
            }
            ExpressionKind::Star => f.write_str("*"),
        }
    }
}

impl fmt::Display for PropertyAccess {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}.{}", self.target, self.key)
    }
}

impl fmt::Display for MapLiteral {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("{")?;
        for (i, (k, v)) in self.entries.iter().enumerate() {
            if i > 0 {
                f.write_str(", ")?;
            }
            write!(f, "{}: {}", k, v)?;
        }
        f.write_str("}")
    }
}

impl fmt::Display for Identifier {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.quoted {
            f.write_str("`")?;
            for c in self.name.chars() {
                if c == '`' {
                    f.write_str("``")?;
                } else {
                    f.write_str(&c.to_string())?;
                }
            }
            f.write_str("`")
        } else {
            f.write_str(&self.name)
        }
    }
}

impl fmt::Display for Literal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Literal::Integer(n) => write!(f, "{}", n),
            Literal::Float(x) => {
                if x.fract() == 0.0 && x.is_finite() {
                    write!(f, "{}.0", *x as i64)
                } else {
                    write!(f, "{}", x)
                }
            }
            Literal::String(s) => {
                f.write_str("'")?;
                for c in s.chars() {
                    match c {
                        '\\' => f.write_str("\\\\")?,
                        '\'' => f.write_str("\\'")?,
                        '\n' => f.write_str("\\n")?,
                        '\r' => f.write_str("\\r")?,
                        '\t' => f.write_str("\\t")?,
                        _ => write!(f, "{}", c)?,
                    }
                }
                f.write_str("'")
            }
            Literal::Boolean(true) => f.write_str("TRUE"),
            Literal::Boolean(false) => f.write_str("FALSE"),
            Literal::Null => f.write_str("NULL"),
        }
    }
}

fn binary_op_symbol(op: BinaryOp) -> &'static str {
    match op {
        BinaryOp::Add => "+",
        BinaryOp::Sub => "-",
        BinaryOp::Mul => "*",
        BinaryOp::Div => "/",
        BinaryOp::Mod => "%",
        BinaryOp::Pow => "^",
        BinaryOp::And => "AND",
        BinaryOp::Or => "OR",
        BinaryOp::Xor => "XOR",
        BinaryOp::Eq => "=",
        BinaryOp::Ne => "<>",
        BinaryOp::Lt => "<",
        BinaryOp::Gt => ">",
        BinaryOp::Le => "<=",
        BinaryOp::Ge => ">=",
        BinaryOp::RegexMatch => "=~",
    }
}

fn write_return_tail(
    f: &mut fmt::Formatter<'_>,
    order_by: &[OrderItem],
    skip: &Option<Expression>,
    limit: &Option<Expression>,
) -> fmt::Result {
    if !order_by.is_empty() {
        f.write_str(" ORDER BY ")?;
        write_list(f, order_by, ", ")?;
    }
    if let Some(s) = skip {
        write!(f, " SKIP {}", s)?;
    }
    if let Some(l) = limit {
        write!(f, " LIMIT {}", l)?;
    }
    Ok(())
}

fn write_list<T: fmt::Display>(f: &mut fmt::Formatter<'_>, items: &[T], sep: &str) -> fmt::Result {
    for (i, item) in items.iter().enumerate() {
        if i > 0 {
            f.write_str(sep)?;
        }
        write!(f, "{}", item)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::super::parse;

    /// Round-trip: parse → format → parse must yield the same AST modulo
    /// spans. We compare normalised display strings.
    fn round_trip(src: &str) {
        let first = parse(src).expect("first parse");
        let formatted = first.to_string();
        let second = parse(&formatted).unwrap_or_else(|_| panic!("re-parse failed: {}", formatted));
        let re_formatted = second.to_string();
        assert_eq!(formatted, re_formatted, "second format diverged");
    }

    #[test]
    fn match_return_round_trip() {
        round_trip("MATCH (a:Person) RETURN a");
    }

    #[test]
    fn match_chain_round_trip() {
        round_trip("MATCH (a)-[:KNOWS]->(b)-[:LIKES]->(c) RETURN c");
    }

    #[test]
    fn variable_length_round_trip() {
        round_trip("MATCH (a)-[r:KNOWS*1..3]->(b) RETURN b");
    }

    #[test]
    fn where_order_limit_round_trip() {
        round_trip(
            "MATCH (a:Person) WHERE a.age > 18 RETURN a.name AS n ORDER BY a.age DESC LIMIT 10",
        );
    }

    #[test]
    fn aggregations_round_trip() {
        round_trip("MATCH (a) RETURN count(DISTINCT a.id), count(*)");
    }

    #[test]
    fn list_comprehension_round_trip() {
        round_trip("RETURN [x IN [1, 2, 3] WHERE x > 1 | x * 2] AS doubled");
    }

    #[test]
    fn pattern_comprehension_round_trip() {
        round_trip("MATCH (a) RETURN [(a)-[:KNOWS]->(b) | b.name] AS friends");
    }

    #[test]
    fn union_all_round_trip() {
        round_trip("MATCH (a) RETURN a UNION ALL MATCH (b) RETURN b");
    }

    #[test]
    fn unwind_round_trip() {
        round_trip("UNWIND [1, 2, 3] AS x RETURN x");
    }

    #[test]
    fn case_round_trip() {
        round_trip("MATCH (a) RETURN CASE WHEN a.age >= 18 THEN 'adult' ELSE 'minor' END AS kind");
    }
}
