//! Cypher grammar — hand-written recursive descent with a Pratt parser for
//! expressions. Targets the v0 subset declared in RFC-004.

use super::ast::*;
use super::error::{ErrorCode, ParseError, ParseResult, SourceSpan};
use super::lexer::{Spanned, Token};

pub fn parse_query(src: &str, tokens: Vec<Spanned<Token>>) -> ParseResult<Query> {
    let mut p = Parser::new(src, tokens);
    let q = p.parse_query()?;
    p.expect_eof()?;
    Ok(q)
}

struct Parser<'src> {
    src: &'src str,
    tokens: Vec<Spanned<Token>>,
    pos: usize,
}

type ReturnTail = (Vec<OrderItem>, Option<Expression>, Option<Expression>);

impl<'src> Parser<'src> {
    fn new(src: &'src str, tokens: Vec<Spanned<Token>>) -> Self {
        Self {
            src,
            tokens,
            pos: 0,
        }
    }

    // ──────────────────────────── helpers ────────────────────────────

    fn peek(&self) -> Option<&Token> {
        self.tokens.get(self.pos).map(|s| &s.value)
    }

    fn peek_at(&self, offset: usize) -> Option<&Token> {
        self.tokens.get(self.pos + offset).map(|s| &s.value)
    }

    fn peek_span(&self) -> SourceSpan {
        self.tokens
            .get(self.pos)
            .map(|s| s.span)
            .unwrap_or_else(|| SourceSpan::point(self.src.len()))
    }

    fn bump(&mut self) -> Option<Spanned<Token>> {
        let t = self.tokens.get(self.pos).cloned();
        if t.is_some() {
            self.pos += 1;
        }
        t
    }

    fn check(&self, expected: &Token) -> bool {
        matches!(self.peek(), Some(t) if discriminant_eq(t, expected))
    }

    fn eat(&mut self, expected: &Token) -> Option<Spanned<Token>> {
        if self.check(expected) {
            self.bump()
        } else {
            None
        }
    }

    fn expect(&mut self, expected: &Token) -> Result<Spanned<Token>, ParseError> {
        match self.eat(expected) {
            Some(t) => Ok(t),
            None => {
                let actual = self
                    .peek()
                    .map(|t| t.label().to_string())
                    .unwrap_or_else(|| "<eof>".to_string());
                Err(ParseError::new(
                    if self.peek().is_none() {
                        ErrorCode::UnexpectedEof
                    } else {
                        ErrorCode::UnexpectedToken
                    },
                    format!("expected `{}`, found `{}`", expected.label(), actual),
                    self.peek_span(),
                ))
            }
        }
    }

    fn expect_eof(&mut self) -> Result<(), Vec<ParseError>> {
        if let Some(tok) = self.peek() {
            let span = self.peek_span();
            return Err(vec![ParseError::new(
                ErrorCode::UnexpectedToken,
                format!("expected end of input, found `{}`", tok.label()),
                span,
            )]);
        }
        Ok(())
    }

    fn at_eof(&self) -> bool {
        self.pos >= self.tokens.len()
    }

    /// `true` when the next token could legitimately start an identifier —
    /// either a plain or backtick-quoted ident, or a reserved keyword (which
    /// is forwarded to [`expect_identifier`] so it surfaces the dedicated
    /// "reserved keyword" diagnostic instead of a downstream "expected `)`").
    fn peek_is_identifier_slot(&self) -> bool {
        match self.peek() {
            Some(Token::Ident(_)) | Some(Token::QuotedIdent(_)) => true,
            Some(tok) => tok.is_reserved_keyword(),
            None => false,
        }
    }

    fn expect_identifier(&mut self) -> Result<Identifier, ParseError> {
        let next = self.bump().ok_or_else(|| {
            ParseError::new(
                ErrorCode::UnexpectedEof,
                "expected identifier, found end of input",
                SourceSpan::point(self.src.len()),
            )
        })?;
        match next.value {
            Token::Ident(name) => Ok(Identifier::new(name, next.span)),
            Token::QuotedIdent(name) => Ok(Identifier::quoted(name, next.span)),
            other if other.is_reserved_keyword() => {
                let label = other.label();
                Err(ParseError::new(
                    ErrorCode::ReservedKeyword,
                    format!(
                        "`{label}` is a reserved Cypher keyword and cannot be used \
                         as an identifier here"
                    ),
                    next.span,
                )
                .with_help(format!(
                    "quote it as `` `{label}` `` to use it as a label or variable name"
                )))
            }
            other => Err(ParseError::new(
                ErrorCode::UnexpectedToken,
                format!("expected identifier, found `{}`", other.label()),
                next.span,
            )),
        }
    }

    // ─────────────────────────── productions ─────────────────────────

    fn parse_query(&mut self) -> Result<Query, Vec<ParseError>> {
        if self.at_eof() {
            return Err(vec![ParseError::new(
                ErrorCode::UnexpectedEof,
                "query is empty",
                SourceSpan::point(0),
            )]);
        }

        // `EXPLAIN` is parsed as a soft keyword prefix — only recognised
        // when it is the very first token of the query, and not followed
        // by `(` (which would make it a function call). `RAW` (RFC-011
        // §6.2) and `VERBOSE` (RFC-010 §5) are optional follow-up soft
        // keywords. Both are surfaced by the lexer as `Ident`.
        let explain_start = self.peek_span().start;
        let explain = matches!(
        self.peek(),
        Some(Token::Ident(name)) if name.eq_ignore_ascii_case("EXPLAIN")
        ) && !matches!(self.peek_at(1), Some(Token::LParen));
        if explain {
            self.bump();
        }
        let explain_raw = explain
            && matches!(
            self.peek(),
            Some(Token::Ident(name)) if name.eq_ignore_ascii_case("RAW")
            );
        if explain_raw {
            self.bump();
        }
        let explain_verbose = explain
            && matches!(
            self.peek(),
            Some(Token::Ident(name)) if name.eq_ignore_ascii_case("VERBOSE")
            );
        if explain_verbose {
            self.bump();
        }

        let head = self.parse_single_query().map_err(|e| vec![e])?;
        let start = if explain {
            explain_start
        } else {
            head.span.start
        };
        let mut end = head.span.end;
        let mut tail = Vec::new();
        while let Some(Token::Union) = self.peek() {
            let union_start = self.peek_span().start;
            self.bump();
            let all = self.eat(&Token::All).is_some();
            let part = self.parse_single_query().map_err(|e| vec![e])?;
            end = part.span.end;
            tail.push(UnionPart {
                all,
                query: part,
                span: SourceSpan::new(union_start, end),
            });
        }
        Ok(Query {
            head,
            tail,
            explain,
            explain_verbose,
            explain_raw,
            span: SourceSpan::new(start, end),
        })
    }

    fn parse_single_query(&mut self) -> Result<SingleQuery, ParseError> {
        let mut clauses = Vec::new();
        let start = self.peek_span().start;
        let mut end = start;
        while let Some(tok) = self.peek() {
            if matches!(tok, Token::Union | Token::Semicolon) {
                break;
            }
            let clause = self.parse_clause()?;
            end = clause.span().end;
            clauses.push(clause);
        }
        if clauses.is_empty() {
            return Err(ParseError::new(
                ErrorCode::UnexpectedEof,
                "query has no clauses",
                SourceSpan::new(start, end),
            ));
        }
        // Eat trailing semicolons (Cypher allows `query;`).
        while self.eat(&Token::Semicolon).is_some() {}
        Ok(SingleQuery {
            clauses,
            span: SourceSpan::new(start, end),
        })
    }

    fn parse_clause(&mut self) -> Result<Clause, ParseError> {
        match self.peek() {
            Some(Token::Match) => self.parse_match_clause(false).map(Clause::Match),
            Some(Token::Optional) => {
                let opt_span = self.peek_span();
                self.bump();
                if !self.check(&Token::Match) {
                    return Err(ParseError::new(
                        ErrorCode::UnexpectedToken,
                        "expected `MATCH` after `OPTIONAL`",
                        self.peek_span(),
                    ));
                }
                let mut clause = self.parse_match_clause(true)?;
                clause.span = SourceSpan::new(opt_span.start, clause.span.end);
                Ok(Clause::Match(clause))
            }
            Some(Token::Return) => self.parse_return_clause().map(Clause::Return),
            Some(Token::With) => self.parse_with_clause().map(Clause::With),
            Some(Token::Unwind) => self.parse_unwind_clause().map(Clause::Unwind),
            Some(Token::Create) => self.parse_create_clause().map(Clause::Create),
            Some(Token::Merge) => self.parse_merge_clause().map(Clause::Merge),
            Some(Token::Set) => self.parse_set_clause().map(Clause::Set),
            Some(Token::Remove) => self.parse_remove_clause().map(Clause::Remove),
            Some(Token::Delete) | Some(Token::Detach) => {
                self.parse_delete_clause().map(Clause::Delete)
            }
            Some(other) => Err(ParseError::new(
                ErrorCode::UnexpectedToken,
                format!("expected a clause keyword, found `{}`", other.label()),
                self.peek_span(),
            )),
            None => Err(ParseError::new(
                ErrorCode::UnexpectedEof,
                "expected a clause keyword",
                SourceSpan::point(self.src.len()),
            )),
        }
    }

    fn parse_match_clause(&mut self, optional: bool) -> Result<MatchClause, ParseError> {
        let start = self.peek_span().start;
        self.expect(&Token::Match)?;
        let mut patterns = Vec::new();
        patterns.push(self.parse_pattern_part()?);
        while self.eat(&Token::Comma).is_some() {
            patterns.push(self.parse_pattern_part()?);
        }

        if optional {
            for part in &patterns {
                if has_variable_length(&part.element) {
                    return Err(ParseError::new(
                        ErrorCode::OptionalVariableLength,
                        "OPTIONAL MATCH cannot use variable-length patterns in v0",
                        part.span,
                    )
                    .with_help("see RFC-004 §Drawbacks 5"));
                }
            }
        }

        let where_ = if self.eat(&Token::Where).is_some() {
            Some(self.parse_expression()?)
        } else {
            None
        };

        let end = where_
            .as_ref()
            .map(|e| e.span.end)
            .unwrap_or_else(|| patterns.last().map(|p| p.span.end).unwrap_or(start));

        Ok(MatchClause {
            optional,
            patterns,
            where_,
            span: SourceSpan::new(start, end),
        })
    }

    fn parse_return_clause(&mut self) -> Result<ReturnClause, ParseError> {
        let start = self.peek_span().start;
        self.expect(&Token::Return)?;
        let distinct = self.eat(&Token::Distinct).is_some();
        let items = self.parse_projection_list()?;
        let (order_by, skip, limit) = self.parse_return_tail()?;
        let end = limit
            .as_ref()
            .map(|e| e.span.end)
            .or_else(|| skip.as_ref().map(|e| e.span.end))
            .or_else(|| order_by.last().map(|o| o.span.end))
            .unwrap_or_else(|| items.last().map(|i| i.span.end).unwrap_or(start));
        Ok(ReturnClause {
            distinct,
            items,
            order_by,
            skip,
            limit,
            span: SourceSpan::new(start, end),
        })
    }

    fn parse_with_clause(&mut self) -> Result<WithClause, ParseError> {
        let start = self.peek_span().start;
        self.expect(&Token::With)?;
        let distinct = self.eat(&Token::Distinct).is_some();
        let items = self.parse_projection_list()?;
        let (order_by, skip, limit) = self.parse_return_tail()?;
        let where_ = if self.eat(&Token::Where).is_some() {
            Some(self.parse_expression()?)
        } else {
            None
        };
        let end = where_
            .as_ref()
            .map(|e| e.span.end)
            .or_else(|| limit.as_ref().map(|e| e.span.end))
            .or_else(|| skip.as_ref().map(|e| e.span.end))
            .or_else(|| order_by.last().map(|o| o.span.end))
            .unwrap_or_else(|| items.last().map(|i| i.span.end).unwrap_or(start));
        Ok(WithClause {
            distinct,
            items,
            order_by,
            skip,
            limit,
            where_,
            span: SourceSpan::new(start, end),
        })
    }

    fn parse_return_tail(&mut self) -> Result<ReturnTail, ParseError> {
        let mut order_by = Vec::new();
        if self.eat(&Token::Order).is_some() {
            self.expect(&Token::By)?;
            order_by.push(self.parse_order_item()?);
            while self.eat(&Token::Comma).is_some() {
                order_by.push(self.parse_order_item()?);
            }
        }
        let skip = if self.eat(&Token::Skip).is_some() {
            Some(self.parse_expression()?)
        } else {
            None
        };
        let limit = if self.eat(&Token::Limit).is_some() {
            Some(self.parse_expression()?)
        } else {
            None
        };
        Ok((order_by, skip, limit))
    }

    fn parse_projection_list(&mut self) -> Result<Vec<ProjectionItem>, ParseError> {
        let mut items = Vec::new();
        items.push(self.parse_projection_item()?);
        while self.eat(&Token::Comma).is_some() {
            items.push(self.parse_projection_item()?);
        }
        Ok(items)
    }

    fn parse_projection_item(&mut self) -> Result<ProjectionItem, ParseError> {
        let expr = self.parse_expression()?;
        let alias = if self.eat(&Token::As).is_some() {
            Some(self.expect_identifier()?)
        } else {
            None
        };
        let end = alias.as_ref().map(|a| a.span.end).unwrap_or(expr.span.end);
        let span = SourceSpan::new(expr.span.start, end);
        Ok(ProjectionItem {
            expression: expr,
            alias,
            span,
        })
    }

    fn parse_order_item(&mut self) -> Result<OrderItem, ParseError> {
        let expr = self.parse_expression()?;
        let (direction, dir_end) = if self.eat(&Token::Asc).is_some() {
            (OrderDirection::Asc, self.tokens[self.pos - 1].span.end)
        } else if self.eat(&Token::Desc).is_some() {
            (OrderDirection::Desc, self.tokens[self.pos - 1].span.end)
        } else {
            (OrderDirection::Asc, expr.span.end)
        };
        let span = SourceSpan::new(expr.span.start, dir_end);
        Ok(OrderItem {
            expression: expr,
            direction,
            span,
        })
    }

    fn parse_unwind_clause(&mut self) -> Result<UnwindClause, ParseError> {
        let start = self.peek_span().start;
        self.expect(&Token::Unwind)?;
        let list = self.parse_expression()?;
        self.expect(&Token::As)?;
        let alias = self.expect_identifier()?;
        let end = alias.span.end;
        Ok(UnwindClause {
            list,
            alias,
            span: SourceSpan::new(start, end),
        })
    }

    fn parse_create_clause(&mut self) -> Result<CreateClause, ParseError> {
        let start = self.peek_span().start;
        self.expect(&Token::Create)?;
        let mut patterns = Vec::new();
        patterns.push(self.parse_pattern_part()?);
        while self.eat(&Token::Comma).is_some() {
            patterns.push(self.parse_pattern_part()?);
        }
        let end = patterns.last().map(|p| p.span.end).unwrap_or(start);
        Ok(CreateClause {
            patterns,
            span: SourceSpan::new(start, end),
        })
    }

    fn parse_merge_clause(&mut self) -> Result<MergeClause, ParseError> {
        let start = self.peek_span().start;
        self.expect(&Token::Merge)?;
        let pattern = self.parse_pattern_part()?;
        // RFC-004 drawback 4: MERGE node patterns must carry exactly one label.
        if let Some(head_labels) = first_node_labels(&pattern.element) {
            if head_labels > 1 {
                return Err(ParseError::new(
                    ErrorCode::MergeMultiLabel,
                    "MERGE node patterns must have at most one label in v0",
                    pattern.span,
                )
                .with_help("see RFC-004 §Drawbacks 4"));
            }
        }
        let mut actions = Vec::new();
        let mut end = pattern.span.end;
        while self.eat(&Token::On).is_some() {
            let trigger_span = self.peek_span();
            let on = match self.peek() {
                Some(Token::Create) => {
                    self.bump();
                    MergeTrigger::Create
                }
                Some(Token::Match) => {
                    self.bump();
                    MergeTrigger::Match
                }
                _ => {
                    return Err(ParseError::new(
                        ErrorCode::UnexpectedToken,
                        "expected `MATCH` or `CREATE` after `ON`",
                        trigger_span,
                    ));
                }
            };
            self.expect(&Token::Set)?;
            let items = self.parse_set_items()?;
            let action_end = items
                .last()
                .map(|i| i.span().end)
                .unwrap_or(trigger_span.end);
            end = action_end;
            actions.push(MergeAction {
                on,
                sets: items,
                span: SourceSpan::new(trigger_span.start, action_end),
            });
        }
        Ok(MergeClause {
            pattern,
            actions,
            span: SourceSpan::new(start, end),
        })
    }

    fn parse_set_clause(&mut self) -> Result<SetClause, ParseError> {
        let start = self.peek_span().start;
        self.expect(&Token::Set)?;
        let items = self.parse_set_items()?;
        let end = items.last().map(|i| i.span().end).unwrap_or(start);
        Ok(SetClause {
            items,
            span: SourceSpan::new(start, end),
        })
    }

    fn parse_set_items(&mut self) -> Result<Vec<SetItem>, ParseError> {
        let mut items = Vec::new();
        items.push(self.parse_set_item()?);
        while self.eat(&Token::Comma).is_some() {
            items.push(self.parse_set_item()?);
        }
        Ok(items)
    }

    fn parse_set_item(&mut self) -> Result<SetItem, ParseError> {
        let id = self.expect_identifier()?;
        let start = id.span.start;
        match self.peek() {
            Some(Token::Dot) => {
                // a.prop = value
                self.bump();
                let key = self.expect_identifier()?;
                let prop_span = SourceSpan::new(start, key.span.end);
                self.expect(&Token::Eq)?;
                let value = self.parse_expression()?;
                let end = value.span.end;
                Ok(SetItem::Property {
                    target: PropertyAccess {
                        target: Expression {
                            kind: ExpressionKind::Variable(id),
                            span: SourceSpan::new(start, prop_span.end),
                        },
                        key,
                        span: prop_span,
                    },
                    value,
                    span: SourceSpan::new(start, end),
                })
            }
            Some(Token::Eq) => {
                self.bump();
                let value = self.parse_expression()?;
                let end = value.span.end;
                Ok(SetItem::Replace {
                    target: id,
                    value,
                    span: SourceSpan::new(start, end),
                })
            }
            Some(Token::Plus) if matches!(self.peek_at(1), Some(Token::Eq)) => {
                self.bump();
                self.bump();
                let value = self.parse_expression()?;
                let end = value.span.end;
                Ok(SetItem::Merge {
                    target: id,
                    value,
                    span: SourceSpan::new(start, end),
                })
            }
            Some(Token::Colon) => {
                let mut labels = Vec::new();
                while self.eat(&Token::Colon).is_some() {
                    labels.push(self.expect_identifier()?);
                }
                let end = labels.last().map(|l| l.span.end).unwrap_or(id.span.end);
                Ok(SetItem::Labels {
                    target: id,
                    labels,
                    span: SourceSpan::new(start, end),
                })
            }
            _ => Err(ParseError::new(
                ErrorCode::UnexpectedToken,
                "expected `.`, `=`, `+=`, or `:` in SET item",
                self.peek_span(),
            )),
        }
    }

    fn parse_remove_clause(&mut self) -> Result<RemoveClause, ParseError> {
        let start = self.peek_span().start;
        self.expect(&Token::Remove)?;
        let mut items = Vec::new();
        items.push(self.parse_remove_item()?);
        while self.eat(&Token::Comma).is_some() {
            items.push(self.parse_remove_item()?);
        }
        let end = match items.last() {
            Some(RemoveItem::Property(p)) => p.span.end,
            Some(RemoveItem::Labels { span, .. }) => span.end,
            None => start,
        };
        Ok(RemoveClause {
            items,
            span: SourceSpan::new(start, end),
        })
    }

    fn parse_remove_item(&mut self) -> Result<RemoveItem, ParseError> {
        let id = self.expect_identifier()?;
        let start = id.span.start;
        match self.peek() {
            Some(Token::Dot) => {
                self.bump();
                let key = self.expect_identifier()?;
                let span = SourceSpan::new(start, key.span.end);
                Ok(RemoveItem::Property(PropertyAccess {
                    target: Expression {
                        kind: ExpressionKind::Variable(id),
                        span,
                    },
                    key,
                    span,
                }))
            }
            Some(Token::Colon) => {
                let mut labels = Vec::new();
                while self.eat(&Token::Colon).is_some() {
                    labels.push(self.expect_identifier()?);
                }
                let end = labels.last().map(|l| l.span.end).unwrap_or(id.span.end);
                Ok(RemoveItem::Labels {
                    target: id,
                    labels,
                    span: SourceSpan::new(start, end),
                })
            }
            _ => Err(ParseError::new(
                ErrorCode::UnexpectedToken,
                "expected `.` or `:` in REMOVE item",
                self.peek_span(),
            )),
        }
    }

    fn parse_delete_clause(&mut self) -> Result<DeleteClause, ParseError> {
        let start = self.peek_span().start;
        let detach = self.eat(&Token::Detach).is_some();
        self.expect(&Token::Delete)?;
        let mut targets = Vec::new();
        targets.push(self.parse_expression()?);
        while self.eat(&Token::Comma).is_some() {
            targets.push(self.parse_expression()?);
        }
        let end = targets.last().map(|t| t.span.end).unwrap_or(start);
        Ok(DeleteClause {
            detach,
            targets,
            span: SourceSpan::new(start, end),
        })
    }

    // ───────────────────────── patterns ──────────────────────────────

    fn parse_pattern_part(&mut self) -> Result<PatternPart, ParseError> {
        let start = self.peek_span().start;
        let binding = if matches!(
            self.peek(),
            Some(Token::Ident(_)) | Some(Token::QuotedIdent(_))
        ) && matches!(self.peek_at(1), Some(Token::Eq))
        {
            let id = self.expect_identifier()?;
            self.expect(&Token::Eq)?;
            Some(id)
        } else {
            None
        };
        // RFC-023: shortestPath / allShortestPaths wrap a single
        // pattern. The function name is parser-special so we don't
        // confuse it with a user-defined function call.
        let shortest_path = if let Some(Token::Ident(name)) = self.peek() {
            if name.eq_ignore_ascii_case("shortestPath") {
                self.bump();
                self.expect(&Token::LParen)?;
                Some(ShortestPathMode::First)
            } else if name.eq_ignore_ascii_case("allShortestPaths") {
                self.bump();
                self.expect(&Token::LParen)?;
                Some(ShortestPathMode::All)
            } else {
                None
            }
        } else {
            None
        };
        let element = self.parse_pattern_element()?;
        if shortest_path.is_some() {
            self.expect(&Token::RParen)?;
        }
        let end = element.span.end;
        Ok(PatternPart {
            binding,
            element,
            span: SourceSpan::new(start, end),
            shortest_path,
        })
    }

    /// Heuristic lookahead used by `parse_primary` to distinguish a paren
    /// expression `(a + b)` from a pattern predicate `(a)-[:KNOWS]-(b)`.
    /// Assumes `peek()` is `LParen`.
    fn starts_pattern_node(&self) -> bool {
        let t1 = self.peek_at(1);
        let t2 = self.peek_at(2);
        let t3 = self.peek_at(3);
        match (t1, t2) {
            // (:Label ...)
            (Some(Token::Colon), _) => true,
            // (a:Label ...)
            (Some(Token::Ident(_)) | Some(Token::QuotedIdent(_)), Some(Token::Colon)) => true,
            // (a {props})
            (Some(Token::Ident(_)) | Some(Token::QuotedIdent(_)), Some(Token::LBrace)) => true,
            // (a) — only a pattern if followed by a relationship arrow.
            (Some(Token::Ident(_)) | Some(Token::QuotedIdent(_)), Some(Token::RParen)) => {
                matches!(t3, Some(Token::Minus) | Some(Token::ArrowLeft))
            }
            // () — anonymous; same rule.
            (Some(Token::RParen), _) => {
                matches!(t2, Some(Token::Minus) | Some(Token::ArrowLeft))
            }
            _ => false,
        }
    }

    fn parse_pattern_element(&mut self) -> Result<PatternElement, ParseError> {
        let head = self.parse_node_pattern()?;
        let start = head.span.start;
        let mut end = head.span.end;
        let mut chain = Vec::new();
        loop {
            if !matches!(self.peek(), Some(Token::Minus) | Some(Token::ArrowLeft)) {
                break;
            }
            let rel = self.parse_relationship_pattern()?;
            let node = self.parse_node_pattern()?;
            end = node.span.end;
            chain.push((rel, node));
        }
        Ok(PatternElement {
            head,
            chain,
            span: SourceSpan::new(start, end),
        })
    }

    fn parse_node_pattern(&mut self) -> Result<NodePattern, ParseError> {
        let lparen = self.expect(&Token::LParen)?;
        let start = lparen.span.start;
        // Anything that *could* be the binding identifier slot is forwarded
        // to `expect_identifier`. Reserved keywords are routed there too so
        // they trip E003 with a quoting hint, instead of producing a generic
        // "expected `)`" further down the line.
        let binding = if self.peek_is_identifier_slot() {
            Some(self.expect_identifier()?)
        } else {
            None
        };
        let labels = self.parse_label_list()?;
        let properties = if self.check(&Token::LBrace) {
            Some(self.parse_map_literal()?)
        } else {
            None
        };
        let rparen = self.expect(&Token::RParen)?;
        let end = rparen.span.end;
        Ok(NodePattern {
            binding,
            labels,
            properties,
            span: SourceSpan::new(start, end),
        })
    }

    fn parse_label_list(&mut self) -> Result<Vec<Identifier>, ParseError> {
        let mut labels = Vec::new();
        while self.eat(&Token::Colon).is_some() {
            labels.push(self.expect_identifier()?);
        }
        Ok(labels)
    }

    fn parse_relationship_pattern(&mut self) -> Result<RelationshipPattern, ParseError> {
        // Forms accepted:
        // -[r:T*1..3 {p:v}]-> (full)
        // -- (anonymous undirected)
        // <-- (anonymous left)
        // --> (anonymous right)
        // -[r]- (no direction, with binding/types)
        let start = self.peek_span().start;
        let lead_left = self.eat(&Token::ArrowLeft);
        let lead_minus = if lead_left.is_none() {
            Some(self.expect(&Token::Minus)?)
        } else {
            None
        };

        let mut binding = None;
        let mut types = Vec::new();
        let mut length = None;
        let mut properties = None;

        if self.eat(&Token::LBracket).is_some() {
            if self.peek_is_identifier_slot() {
                binding = Some(self.expect_identifier()?);
            }
            if self.eat(&Token::Colon).is_some() {
                types.push(self.expect_identifier()?);
                while self.eat(&Token::Pipe).is_some() {
                    // Optional `:` after `|`.
                    self.eat(&Token::Colon);
                    types.push(self.expect_identifier()?);
                }
            }
            if let Some(star) = self.eat(&Token::Star) {
                length = Some(self.parse_relationship_length(star.span)?);
            }
            if self.check(&Token::LBrace) {
                properties = Some(self.parse_map_literal()?);
            }
            self.expect(&Token::RBracket)?;
        }

        let trail_arrow = self.eat(&Token::Arrow);
        let trail_minus = if trail_arrow.is_none() {
            Some(self.expect(&Token::Minus)?)
        } else {
            None
        };

        let direction = match (lead_left.is_some(), trail_arrow.is_some()) {
            (true, true) => {
                return Err(ParseError::new(
                    ErrorCode::UnexpectedToken,
                    "relationship cannot be both `<-` and `->`",
                    SourceSpan::new(start, trail_arrow.unwrap().span.end),
                ));
            }
            (true, false) => RelationshipDirection::Left,
            (false, true) => RelationshipDirection::Right,
            (false, false) => RelationshipDirection::Both,
        };

        let end = trail_arrow
            .as_ref()
            .map(|t| t.span.end)
            .or_else(|| trail_minus.as_ref().map(|t| t.span.end))
            .unwrap_or_else(|| {
                lead_minus
                    .as_ref()
                    .map(|t| t.span.end)
                    .unwrap_or_else(|| lead_left.as_ref().map(|t| t.span.end).unwrap_or(start))
            });

        Ok(RelationshipPattern {
            direction,
            binding,
            types,
            length,
            properties,
            span: SourceSpan::new(start, end),
        })
    }

    fn parse_relationship_length(
        &mut self,
        star_span: SourceSpan,
    ) -> Result<RelationshipLength, ParseError> {
        // Forms:
        // * (forbidden — RFC-004)
        // *N (exact)
        // *N..M (range)
        // *..M (forbidden)
        // *N.. (forbidden — no upper bound)
        let min_lit = self.eat_integer();
        let min = match min_lit {
            Some((n, _)) => n,
            None => {
                // `*..M` form: no min, just an upper bound. The next
                // token must be `..`; otherwise it's a bare `*` and
                // we still reject.
                if matches!(self.peek(), Some(Token::Range)) {
                    1
                } else {
                    return Err(ParseError::new(
                        ErrorCode::UnboundedVariableLength,
                        "variable-length pattern requires explicit min..max bounds in v0",
                        star_span,
                    )
                    .with_help("see RFC-004 §Out-of-scope: variable-length without upper bound"));
                }
            }
        };
        let max = if self.eat(&Token::Range).is_some() {
            match self.eat_integer() {
                Some((n, _)) => n,
                None => {
                    return Err(ParseError::new(
                        ErrorCode::UnboundedVariableLength,
                        "variable-length upper bound is required in v0",
                        self.peek_span(),
                    )
                    .with_help("write `*N..M` with finite M"));
                }
            }
        } else {
            min
        };
        if min < 0 || max < 0 || max < min {
            return Err(ParseError::new(
                ErrorCode::InvalidNumber,
                format!("invalid variable-length range *{}..{}", min, max),
                star_span,
            ));
        }
        // `min == 0` (zero-length patterns) is syntactically accepted; the
        // semantic check is deferred to lowering (RFC-004 §Out-of-scope).
        Ok(RelationshipLength {
            min: min as u32,
            max: max as u32,
        })
    }

    fn eat_integer(&mut self) -> Option<(i64, SourceSpan)> {
        match self.peek() {
            Some(Token::Integer(_)) => {
                let t = self.bump().unwrap();
                if let Token::Integer(n) = t.value {
                    Some((n, t.span))
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    fn parse_map_literal(&mut self) -> Result<MapLiteral, ParseError> {
        let lbrace = self.expect(&Token::LBrace)?;
        let start = lbrace.span.start;
        let mut entries = Vec::new();
        if !self.check(&Token::RBrace) {
            entries.push(self.parse_map_entry()?);
            while self.eat(&Token::Comma).is_some() {
                entries.push(self.parse_map_entry()?);
            }
        }
        let rbrace = self.expect(&Token::RBrace)?;
        Ok(MapLiteral {
            entries,
            span: SourceSpan::new(start, rbrace.span.end),
        })
    }

    fn parse_map_entry(&mut self) -> Result<(Identifier, Expression), ParseError> {
        let key = self.expect_identifier()?;
        self.expect(&Token::Colon)?;
        let value = self.parse_expression()?;
        Ok((key, value))
    }

    // ───────────────────── expressions (Pratt) ───────────────────────

    fn parse_expression(&mut self) -> Result<Expression, ParseError> {
        self.parse_expr_bp(0)
    }

    fn parse_expr_bp(&mut self, min_bp: u8) -> Result<Expression, ParseError> {
        let mut lhs = self.parse_unary()?;

        loop {
            let op_info = match self.peek() {
                Some(Token::Or) => Some((1, 2, BinOpKind::Bin(BinaryOp::Or))),
                Some(Token::Xor) => Some((3, 4, BinOpKind::Bin(BinaryOp::Xor))),
                Some(Token::And) => Some((5, 6, BinOpKind::Bin(BinaryOp::And))),
                Some(Token::Eq) => Some((7, 8, BinOpKind::Bin(BinaryOp::Eq))),
                Some(Token::Ne) => Some((7, 8, BinOpKind::Bin(BinaryOp::Ne))),
                Some(Token::Lt) => Some((7, 8, BinOpKind::Bin(BinaryOp::Lt))),
                Some(Token::Gt) => Some((7, 8, BinOpKind::Bin(BinaryOp::Gt))),
                Some(Token::Le) => Some((7, 8, BinOpKind::Bin(BinaryOp::Le))),
                Some(Token::Ge) => Some((7, 8, BinOpKind::Bin(BinaryOp::Ge))),
                Some(Token::RegexMatch) => Some((7, 8, BinOpKind::Bin(BinaryOp::RegexMatch))),
                Some(Token::In) => Some((7, 8, BinOpKind::In)),
                Some(Token::Is) => Some((7, 8, BinOpKind::IsNull)),
                Some(Token::StartsKw) => Some((7, 8, BinOpKind::StringOp(StringOp::StartsWith))),
                Some(Token::EndsKw) => Some((7, 8, BinOpKind::StringOp(StringOp::EndsWith))),
                Some(Token::Contains) => Some((7, 8, BinOpKind::StringOp(StringOp::Contains))),
                Some(Token::Plus) => Some((9, 10, BinOpKind::Bin(BinaryOp::Add))),
                Some(Token::Minus) => Some((9, 10, BinOpKind::Bin(BinaryOp::Sub))),
                Some(Token::Star) => Some((11, 12, BinOpKind::Bin(BinaryOp::Mul))),
                Some(Token::Slash) => Some((11, 12, BinOpKind::Bin(BinaryOp::Div))),
                Some(Token::Percent) => Some((11, 12, BinOpKind::Bin(BinaryOp::Mod))),
                Some(Token::Caret) => Some((14, 13, BinOpKind::Bin(BinaryOp::Pow))),
                _ => None,
            };

            let (lbp, rbp, kind) = match op_info {
                Some(triple) => triple,
                None => break,
            };
            if lbp < min_bp {
                break;
            }

            // Consume the operator token (special cases handle their own tokens).
            match kind {
                BinOpKind::Bin(op) => {
                    self.bump();
                    let rhs = self.parse_expr_bp(rbp)?;
                    let span = lhs.span.merge(rhs.span);
                    lhs = Expression {
                        kind: ExpressionKind::Binary {
                            op,
                            left: Box::new(lhs),
                            right: Box::new(rhs),
                        },
                        span,
                    };
                }
                BinOpKind::In => {
                    self.bump(); // IN
                    let rhs = self.parse_expr_bp(rbp)?;
                    let span = lhs.span.merge(rhs.span);
                    lhs = Expression {
                        kind: ExpressionKind::In {
                            item: Box::new(lhs),
                            list: Box::new(rhs),
                        },
                        span,
                    };
                }
                BinOpKind::IsNull => {
                    self.bump(); // IS
                    let negated = self.eat(&Token::Not).is_some();
                    self.expect(&Token::Null)?;
                    let end = self.tokens[self.pos - 1].span.end;
                    let span = SourceSpan::new(lhs.span.start, end);
                    lhs = Expression {
                        kind: ExpressionKind::IsNull {
                            expr: Box::new(lhs),
                            negated,
                        },
                        span,
                    };
                }
                BinOpKind::StringOp(op) => {
                    self.bump(); // STARTS/ENDS/CONTAINS
                    if matches!(op, StringOp::StartsWith | StringOp::EndsWith) {
                        self.expect(&Token::With)?;
                    }
                    let rhs = self.parse_expr_bp(rbp)?;
                    let span = lhs.span.merge(rhs.span);
                    lhs = Expression {
                        kind: ExpressionKind::StringTest {
                            op,
                            target: Box::new(lhs),
                            pattern: Box::new(rhs),
                        },
                        span,
                    };
                }
            }
        }
        Ok(lhs)
    }

    fn parse_unary(&mut self) -> Result<Expression, ParseError> {
        match self.peek() {
            Some(Token::Not) => {
                let start = self.peek_span().start;
                self.bump();
                let inner = self.parse_expr_bp(5)?;
                let end = inner.span.end;
                Ok(Expression {
                    kind: ExpressionKind::Unary {
                        op: UnaryOp::Not,
                        expr: Box::new(inner),
                    },
                    span: SourceSpan::new(start, end),
                })
            }
            Some(Token::Minus) => {
                let start = self.peek_span().start;
                self.bump();
                let inner = self.parse_expr_bp(13)?;
                let end = inner.span.end;
                Ok(Expression {
                    kind: ExpressionKind::Unary {
                        op: UnaryOp::Neg,
                        expr: Box::new(inner),
                    },
                    span: SourceSpan::new(start, end),
                })
            }
            _ => self.parse_postfix(),
        }
    }

    fn parse_postfix(&mut self) -> Result<Expression, ParseError> {
        let mut expr = self.parse_primary()?;
        loop {
            match self.peek() {
                Some(Token::Dot) => {
                    self.bump();
                    let key = self.expect_identifier()?;
                    let span = SourceSpan::new(expr.span.start, key.span.end);
                    expr = Expression {
                        kind: ExpressionKind::Property(Box::new(PropertyAccess {
                            target: expr,
                            key,
                            span,
                        })),
                        span,
                    };
                }
                Some(Token::LBracket) => {
                    self.bump();
                    // index or range
                    let from = if self.check(&Token::Range) {
                        None
                    } else {
                        Some(Box::new(self.parse_expression()?))
                    };
                    if self.eat(&Token::Range).is_some() {
                        let to = if self.check(&Token::RBracket) {
                            None
                        } else {
                            Some(Box::new(self.parse_expression()?))
                        };
                        let rbracket = self.expect(&Token::RBracket)?;
                        let span = SourceSpan::new(expr.span.start, rbracket.span.end);
                        expr = Expression {
                            kind: ExpressionKind::Range {
                                target: Box::new(expr),
                                from,
                                to,
                            },
                            span,
                        };
                    } else {
                        let index = from.ok_or_else(|| {
                            ParseError::new(
                                ErrorCode::UnexpectedToken,
                                "expected index expression",
                                self.peek_span(),
                            )
                        })?;
                        let rbracket = self.expect(&Token::RBracket)?;
                        let span = SourceSpan::new(expr.span.start, rbracket.span.end);
                        expr = Expression {
                            kind: ExpressionKind::Index {
                                target: Box::new(expr),
                                index,
                            },
                            span,
                        };
                    }
                }
                _ => break,
            }
        }
        Ok(expr)
    }

    fn parse_primary(&mut self) -> Result<Expression, ParseError> {
        let start_span = self.peek_span();
        let next = self.peek().cloned();
        match next {
            Some(Token::Integer(n)) => {
                let t = self.bump().unwrap();
                Ok(Expression {
                    kind: ExpressionKind::Literal(Literal::Integer(n)),
                    span: t.span,
                })
            }
            Some(Token::Float(f)) => {
                let t = self.bump().unwrap();
                Ok(Expression {
                    kind: ExpressionKind::Literal(Literal::Float(f)),
                    span: t.span,
                })
            }
            Some(Token::String(s)) => {
                let t = self.bump().unwrap();
                Ok(Expression {
                    kind: ExpressionKind::Literal(Literal::String(s)),
                    span: t.span,
                })
            }
            Some(Token::True) => {
                let t = self.bump().unwrap();
                Ok(Expression {
                    kind: ExpressionKind::Literal(Literal::Boolean(true)),
                    span: t.span,
                })
            }
            Some(Token::False) => {
                let t = self.bump().unwrap();
                Ok(Expression {
                    kind: ExpressionKind::Literal(Literal::Boolean(false)),
                    span: t.span,
                })
            }
            Some(Token::Null) => {
                let t = self.bump().unwrap();
                Ok(Expression {
                    kind: ExpressionKind::Literal(Literal::Null),
                    span: t.span,
                })
            }
            Some(Token::Parameter(name)) => {
                let t = self.bump().unwrap();
                Ok(Expression {
                    kind: ExpressionKind::Parameter(name),
                    span: t.span,
                })
            }
            Some(Token::Star) => {
                let t = self.bump().unwrap();
                Ok(Expression {
                    kind: ExpressionKind::Star,
                    span: t.span,
                })
            }
            Some(Token::LParen) => {
                if self.starts_pattern_node() {
                    let element = self.parse_pattern_element()?;
                    let span = element.span;
                    Ok(Expression {
                        kind: ExpressionKind::Exists(Box::new(element)),
                        span,
                    })
                } else {
                    self.bump();
                    let inner = self.parse_expression()?;
                    self.expect(&Token::RParen)?;
                    Ok(inner)
                }
            }
            Some(Token::LBracket) => self.parse_list_or_list_comprehension(start_span),
            Some(Token::LBrace) => {
                let map = self.parse_map_literal()?;
                Ok(Expression {
                    span: map.span,
                    kind: ExpressionKind::Map(map),
                })
            }
            Some(Token::Case) => self.parse_case_expression(start_span),
            Some(Token::Ident(_)) | Some(Token::QuotedIdent(_)) => {
                // Could be: function call, variable, qualified name, EXISTS, NULL/TRUE/FALSE handled earlier.
                let id = self.expect_identifier()?;
                if id.name.eq_ignore_ascii_case("exists") && self.check(&Token::LParen) {
                    return self.parse_exists_call(id);
                }
                self.parse_after_identifier(id)
            }
            Some(other) => Err(ParseError::new(
                ErrorCode::UnexpectedToken,
                format!("expected expression, found `{}`", other.label()),
                start_span,
            )),
            None => Err(ParseError::new(
                ErrorCode::UnexpectedEof,
                "expected expression",
                SourceSpan::point(self.src.len()),
            )),
        }
    }

    fn parse_after_identifier(&mut self, head: Identifier) -> Result<Expression, ParseError> {
        // Function call: head ( ... )
        // Qualified function: head.head2 ( ... )
        // Variable: head
        let mut segments = vec![head];
        while self.eat(&Token::Dot).is_some() {
            // Could be property access on a variable expression. Distinguish:
            // - if followed by `(`, treat the whole thing as qualified function call.
            // - else it's a property access; we backtrack-by-construction.
            // We just gobble the segments and let the caller decide.
            // To avoid ambiguity, we only take the dot-form when peek_at(1) is `(`.
            let next_id = self.expect_identifier()?;
            segments.push(next_id);
            if !self.check(&Token::LParen) && !self.check(&Token::Dot) {
                break;
            }
        }

        if self.eat(&Token::LParen).is_some() {
            let span_start = segments[0].span.start;
            let qname = QualifiedName {
                span: SourceSpan::new(segments[0].span.start, segments.last().unwrap().span.end),
                segments,
            };
            let distinct = self.eat(&Token::Distinct).is_some();
            let mut args = Vec::new();
            if !self.check(&Token::RParen) {
                // Special case: `count(*)`
                if matches!(self.peek(), Some(Token::Star))
                    && matches!(self.peek_at(1), Some(Token::RParen))
                {
                    let star = self.bump().unwrap();
                    args.push(Expression {
                        kind: ExpressionKind::Star,
                        span: star.span,
                    });
                } else {
                    args.push(self.parse_expression()?);
                    while self.eat(&Token::Comma).is_some() {
                        args.push(self.parse_expression()?);
                    }
                }
            }
            let rparen = self.expect(&Token::RParen)?;
            let span = SourceSpan::new(span_start, rparen.span.end);
            return Ok(Expression {
                kind: ExpressionKind::FunctionCall {
                    name: qname,
                    args,
                    distinct,
                },
                span,
            });
        }

        // Multiple segments and no `(` → first segment is variable, remaining
        // become property accesses chained. We reverse the build.
        let first = segments.remove(0);
        let mut expr = Expression {
            kind: ExpressionKind::Variable(first.clone()),
            span: first.span,
        };
        for seg in segments {
            let span = SourceSpan::new(expr.span.start, seg.span.end);
            expr = Expression {
                kind: ExpressionKind::Property(Box::new(PropertyAccess {
                    target: expr,
                    key: seg,
                    span,
                })),
                span,
            };
        }
        Ok(expr)
    }

    fn parse_list_or_list_comprehension(
        &mut self,
        lbracket: SourceSpan,
    ) -> Result<Expression, ParseError> {
        self.bump(); // [
                     // Distinguish list-comprehension `[x IN list ... | proj]` from
                     // pattern-comprehension `[(a)-[]-(b) | proj]` from list literal.
        if self.check(&Token::LParen) {
            // Pattern comprehension.
            let element = self.parse_pattern_element()?;
            let predicate = if self.eat(&Token::Where).is_some() {
                Some(self.parse_expression()?)
            } else {
                None
            };
            self.expect(&Token::Pipe)?;
            let projection = self.parse_expression()?;
            let rbracket = self.expect(&Token::RBracket)?;
            let span = SourceSpan::new(lbracket.start, rbracket.span.end);
            return Ok(Expression {
                kind: ExpressionKind::PatternComprehension(Box::new(PatternComprehension {
                    binding: None,
                    pattern: element,
                    predicate,
                    projection,
                    span,
                })),
                span,
            });
        }
        if matches!(
            self.peek(),
            Some(Token::Ident(_)) | Some(Token::QuotedIdent(_))
        ) && matches!(self.peek_at(1), Some(Token::In))
        {
            // List comprehension.
            let var = self.expect_identifier()?;
            self.expect(&Token::In)?;
            let list = self.parse_expression()?;
            let predicate = if self.eat(&Token::Where).is_some() {
                Some(self.parse_expression()?)
            } else {
                None
            };
            let projection = if self.eat(&Token::Pipe).is_some() {
                Some(self.parse_expression()?)
            } else {
                None
            };
            let rbracket = self.expect(&Token::RBracket)?;
            let span = SourceSpan::new(lbracket.start, rbracket.span.end);
            return Ok(Expression {
                kind: ExpressionKind::ListComprehension(Box::new(ListComprehension {
                    variable: var,
                    list,
                    predicate,
                    projection,
                    span,
                })),
                span,
            });
        }
        // List literal.
        let mut items = Vec::new();
        if !self.check(&Token::RBracket) {
            items.push(self.parse_expression()?);
            while self.eat(&Token::Comma).is_some() {
                items.push(self.parse_expression()?);
            }
        }
        let rbracket = self.expect(&Token::RBracket)?;
        let span = SourceSpan::new(lbracket.start, rbracket.span.end);
        Ok(Expression {
            kind: ExpressionKind::List(items),
            span,
        })
    }

    fn parse_case_expression(&mut self, case_span: SourceSpan) -> Result<Expression, ParseError> {
        self.bump(); // CASE
        let scrutinee = if !self.check(&Token::When) {
            Some(Box::new(self.parse_expression()?))
        } else {
            None
        };
        let mut branches = Vec::new();
        while self.eat(&Token::When).is_some() {
            let when = self.parse_expression()?;
            self.expect(&Token::Then)?;
            let then = self.parse_expression()?;
            let span = SourceSpan::new(when.span.start, then.span.end);
            branches.push(CaseBranch { when, then, span });
        }
        if branches.is_empty() {
            return Err(ParseError::new(
                ErrorCode::UnexpectedToken,
                "CASE must have at least one WHEN branch",
                case_span,
            ));
        }
        let otherwise = if self.eat(&Token::Else).is_some() {
            Some(Box::new(self.parse_expression()?))
        } else {
            None
        };
        let end_tok = self.expect(&Token::End)?;
        let span = SourceSpan::new(case_span.start, end_tok.span.end);
        Ok(Expression {
            kind: ExpressionKind::Case {
                scrutinee,
                branches,
                otherwise,
            },
            span,
        })
    }

    fn parse_exists_call(&mut self, exists_id: Identifier) -> Result<Expression, ParseError> {
        self.expect(&Token::LParen)?;
        let element = self.parse_pattern_element()?;
        let rparen = self.expect(&Token::RParen)?;
        let span = SourceSpan::new(exists_id.span.start, rparen.span.end);
        Ok(Expression {
            kind: ExpressionKind::Exists(Box::new(element)),
            span,
        })
    }
}

enum BinOpKind {
    Bin(BinaryOp),
    In,
    IsNull,
    StringOp(StringOp),
}

fn has_variable_length(element: &PatternElement) -> bool {
    element.chain.iter().any(|(r, _)| r.length.is_some())
}

fn first_node_labels(element: &PatternElement) -> Option<usize> {
    Some(element.head.labels.len())
}

/// Variant-only equality (ignores embedded data like `Token::Ident("a") == Token::Ident("b")`).
fn discriminant_eq(a: &Token, b: &Token) -> bool {
    std::mem::discriminant(a) == std::mem::discriminant(b)
}

#[cfg(test)]
mod tests {
    use super::super::parse;
    use super::*;

    fn ok(src: &str) -> Query {
        match parse(src) {
            Ok(q) => q,
            Err(errs) => panic!("expected ok, got errors: {:?}\nsrc: {}", errs, src),
        }
    }

    fn err_code(src: &str) -> ErrorCode {
        let errs = parse(src).expect_err("expected error");
        errs[0].code
    }

    #[test]
    fn simplest_match_return() {
        let q = ok("MATCH (a:Person) RETURN a");
        assert_eq!(q.head.clauses.len(), 2);
        match &q.head.clauses[0] {
            Clause::Match(m) => {
                assert!(!m.optional);
                assert_eq!(m.patterns.len(), 1);
                let head = &m.patterns[0].element.head;
                assert_eq!(head.labels.len(), 1);
                assert_eq!(head.labels[0].name, "Person");
                assert_eq!(head.binding.as_ref().unwrap().name, "a");
            }
            _ => panic!("expected MATCH"),
        }
    }

    #[test]
    fn match_chain_two_hops() {
        let q = ok("MATCH (a)-[:KNOWS]->(b)-[:LIKES]->(c) RETURN c");
        match &q.head.clauses[0] {
            Clause::Match(m) => {
                let elem = &m.patterns[0].element;
                assert_eq!(elem.chain.len(), 2);
                assert!(matches!(
                    elem.chain[0].0.direction,
                    RelationshipDirection::Right
                ));
                assert_eq!(elem.chain[0].0.types[0].name, "KNOWS");
            }
            _ => panic!(),
        }
    }

    #[test]
    fn variable_length_pattern() {
        let q = ok("MATCH (a)-[:KNOWS*1..3]->(b) RETURN b");
        match &q.head.clauses[0] {
            Clause::Match(m) => {
                let rel = &m.patterns[0].element.chain[0].0;
                assert_eq!(rel.length.unwrap().min, 1);
                assert_eq!(rel.length.unwrap().max, 3);
            }
            _ => panic!(),
        }
    }

    #[test]
    fn variable_length_unbounded_is_error() {
        let code = err_code("MATCH (a)-[:KNOWS*]->(b) RETURN b");
        assert_eq!(code, ErrorCode::UnboundedVariableLength);
    }

    #[test]
    fn where_and_order_and_limit() {
        let q = ok("MATCH (a:Person) WHERE a.age > 18 AND a.name <> 'Bob' \
 RETURN a.name AS n ORDER BY a.age DESC LIMIT 10");
        match &q.head.clauses[1] {
            Clause::Return(r) => {
                assert_eq!(r.items.len(), 1);
                assert_eq!(r.items[0].alias.as_ref().unwrap().name, "n");
                assert_eq!(r.order_by.len(), 1);
                assert!(matches!(r.order_by[0].direction, OrderDirection::Desc));
                match &r.limit.as_ref().unwrap().kind {
                    ExpressionKind::Literal(Literal::Integer(10)) => {}
                    other => panic!("expected LIMIT 10, got {:?}", other),
                }
            }
            _ => panic!(),
        }
    }

    #[test]
    fn with_pipes_and_filters() {
        let q = ok("MATCH (a:Person)-[:KNOWS]->(b) \
 WITH a, count(b) AS friends \
 WHERE friends > 5 \
 RETURN a.name, friends ORDER BY friends DESC");
        assert_eq!(q.head.clauses.len(), 3);
        assert!(matches!(q.head.clauses[1], Clause::With(_)));
    }

    #[test]
    fn unwind_clause() {
        let q = ok("UNWIND [1, 2, 3] AS x RETURN x");
        match &q.head.clauses[0] {
            Clause::Unwind(u) => assert_eq!(u.alias.name, "x"),
            _ => panic!(),
        }
    }

    #[test]
    fn aggregations_count_distinct() {
        let q = ok("MATCH (a) RETURN count(DISTINCT a.id), count(*)");
        match &q.head.clauses[1] {
            Clause::Return(r) => {
                assert_eq!(r.items.len(), 2);
                match &r.items[0].expression.kind {
                    ExpressionKind::FunctionCall { name, distinct, .. } => {
                        assert_eq!(name.joined(), "count");
                        assert!(*distinct);
                    }
                    _ => panic!("expected count(DISTINCT ...)"),
                }
                match &r.items[1].expression.kind {
                    ExpressionKind::FunctionCall { name, args, .. } => {
                        assert_eq!(name.joined(), "count");
                        assert!(matches!(args[0].kind, ExpressionKind::Star));
                    }
                    _ => panic!("expected count(*)"),
                }
            }
            _ => panic!(),
        }
    }

    #[test]
    fn list_comprehension() {
        let q = ok("RETURN [x IN [1,2,3] WHERE x > 1 | x * 2] AS doubled");
        match &q.head.clauses[0] {
            Clause::Return(r) => match &r.items[0].expression.kind {
                ExpressionKind::ListComprehension(lc) => {
                    assert_eq!(lc.variable.name, "x");
                    assert!(lc.predicate.is_some());
                    assert!(lc.projection.is_some());
                }
                other => panic!("expected list comprehension, got {:?}", other),
            },
            _ => panic!(),
        }
    }

    #[test]
    fn pattern_comprehension() {
        let q = ok("MATCH (a) RETURN [(a)-[:KNOWS]->(b) | b.name] AS friends");
        match &q.head.clauses[1] {
            Clause::Return(r) => match &r.items[0].expression.kind {
                ExpressionKind::PatternComprehension(_) => {}
                other => panic!("expected pattern comprehension, got {:?}", other),
            },
            _ => panic!(),
        }
    }

    #[test]
    fn in_list_predicate() {
        let q = ok("MATCH (a) WHERE a.country IN ['AR', 'EC', 'MX'] RETURN a");
        match &q.head.clauses[0] {
            Clause::Match(m) => {
                let pred = m.where_.as_ref().expect("where present");
                assert!(matches!(pred.kind, ExpressionKind::In { .. }));
            }
            _ => panic!(),
        }
    }

    #[test]
    fn is_null_predicate() {
        let q = ok("MATCH (a) WHERE a.name IS NOT NULL RETURN a");
        match &q.head.clauses[0] {
            Clause::Match(m) => {
                let pred = m.where_.as_ref().unwrap();
                match &pred.kind {
                    ExpressionKind::IsNull { negated, .. } => assert!(*negated),
                    _ => panic!(),
                }
            }
            _ => panic!(),
        }
    }

    #[test]
    fn string_predicates() {
        let q = ok("MATCH (a) WHERE a.name STARTS WITH 'A' AND a.bio CONTAINS 'rust' RETURN a");
        match &q.head.clauses[0] {
            Clause::Match(m) => {
                assert!(m.where_.is_some());
            }
            _ => panic!(),
        }
    }

    #[test]
    fn case_expression() {
        let q = ok("MATCH (a) RETURN CASE WHEN a.age >= 18 THEN 'adult' ELSE 'minor' END AS kind");
        match &q.head.clauses[1] {
            Clause::Return(r) => match &r.items[0].expression.kind {
                ExpressionKind::Case {
                    branches,
                    otherwise,
                    ..
                } => {
                    assert_eq!(branches.len(), 1);
                    assert!(otherwise.is_some());
                }
                other => panic!("expected CASE, got {:?}", other),
            },
            _ => panic!(),
        }
    }

    #[test]
    fn optional_match_no_variable_length() {
        let code = err_code("OPTIONAL MATCH (a)-[:KNOWS*1..3]->(b) RETURN b");
        assert_eq!(code, ErrorCode::OptionalVariableLength);
    }

    #[test]
    fn create_and_set() {
        let q = ok("CREATE (a:Person {name: 'Ada'}) SET a.age = 36 RETURN a");
        assert_eq!(q.head.clauses.len(), 3);
        assert!(matches!(q.head.clauses[0], Clause::Create(_)));
        assert!(matches!(q.head.clauses[1], Clause::Set(_)));
    }

    #[test]
    fn merge_multi_label_rejected() {
        let code = err_code("MERGE (a:A:B) RETURN a");
        assert_eq!(code, ErrorCode::MergeMultiLabel);
    }

    #[test]
    fn delete_detach() {
        let q = ok("MATCH (a:Person) DETACH DELETE a");
        match &q.head.clauses[1] {
            Clause::Delete(d) => assert!(d.detach),
            _ => panic!(),
        }
    }

    #[test]
    fn union_all() {
        let q = ok("MATCH (a) RETURN a UNION ALL MATCH (b) RETURN b");
        assert_eq!(q.tail.len(), 1);
        assert!(q.tail[0].all);
    }

    #[test]
    fn parameter_in_where() {
        let q = ok("MATCH (a:Person) WHERE a.id = $personId RETURN a");
        match &q.head.clauses[0] {
            Clause::Match(m) => assert!(m.where_.is_some()),
            _ => panic!(),
        }
    }

    #[test]
    fn precedence_or_and_not() {
        let q = ok("RETURN NOT a OR b AND c");
        // Expected: (NOT a) OR (b AND c)
        match &q.head.clauses[0] {
            Clause::Return(r) => match &r.items[0].expression.kind {
                ExpressionKind::Binary {
                    op: BinaryOp::Or,
                    left,
                    right,
                } => {
                    assert!(matches!(
                        left.kind,
                        ExpressionKind::Unary {
                            op: UnaryOp::Not,
                            ..
                        }
                    ));
                    assert!(matches!(
                        right.kind,
                        ExpressionKind::Binary {
                            op: BinaryOp::And,
                            ..
                        }
                    ));
                }
                other => panic!("expected OR, got {:?}", other),
            },
            _ => panic!(),
        }
    }

    #[test]
    fn property_access_chain() {
        let q = ok("RETURN a.b.c");
        match &q.head.clauses[0] {
            Clause::Return(r) => match &r.items[0].expression.kind {
                ExpressionKind::Property(p) => {
                    assert_eq!(p.key.name, "c");
                    assert!(matches!(p.target.kind, ExpressionKind::Property(_)));
                }
                _ => panic!(),
            },
            _ => panic!(),
        }
    }

    #[test]
    fn empty_query_errors() {
        let code = err_code("");
        assert_eq!(code, ErrorCode::UnexpectedEof);
    }

    #[test]
    fn semicolon_terminator_ok() {
        let q = ok("MATCH (a) RETURN a;");
        assert_eq!(q.head.clauses.len(), 2);
    }

    #[test]
    fn explain_prefix_is_recognised() {
        let q = ok("EXPLAIN MATCH (a:Person) RETURN a");
        assert!(q.explain);
        assert_eq!(q.head.clauses.len(), 2);
    }

    #[test]
    fn explain_prefix_round_trips_through_display() {
        let q = ok("EXPLAIN MATCH (a:Person) RETURN a");
        let rendered = q.to_string();
        assert!(rendered.starts_with("EXPLAIN "));
        let q2 = ok(&rendered);
        assert!(q2.explain);
    }

    #[test]
    fn explain_lowercase_also_recognised() {
        let q = ok("explain MATCH (a:Person) RETURN a");
        assert!(q.explain);
    }

    #[test]
    fn explain_used_as_function_name_is_not_a_prefix() {
        // `explain(x)` would be a function call, not a prefix. We can't
        // really test that scenario at top level because the grammar
        // wouldn't accept it as a query, but ensure that a query whose
        // first identifier is something else doesn't flip explain on.
        let q = ok("MATCH (a:Person) RETURN a");
        assert!(!q.explain);
        assert!(!q.explain_verbose);
    }

    #[test]
    fn explain_verbose_prefix_is_recognised() {
        let q = ok("EXPLAIN VERBOSE MATCH (a:Person) RETURN a");
        assert!(q.explain);
        assert!(q.explain_verbose);
    }

    #[test]
    fn explain_verbose_round_trips_through_display() {
        let q = ok("EXPLAIN VERBOSE MATCH (a:Person) RETURN a");
        let rendered = q.to_string();
        assert!(rendered.starts_with("EXPLAIN VERBOSE "));
        let q2 = ok(&rendered);
        assert!(q2.explain);
        assert!(q2.explain_verbose);
    }

    #[test]
    fn explain_without_verbose_does_not_set_flag() {
        let q = ok("EXPLAIN MATCH (a:Person) RETURN a");
        assert!(q.explain);
        assert!(!q.explain_verbose);
    }

    #[test]
    fn verbose_alone_is_treated_as_identifier() {
        // Without a preceding EXPLAIN, VERBOSE is just an identifier and
        // the parser falls back to its normal clause expectations.
        let _ = err_code("VERBOSE MATCH (a) RETURN a");
    }

    #[test]
    fn explain_raw_prefix_is_recognised() {
        let q = ok("EXPLAIN RAW MATCH (a:Person) RETURN a");
        assert!(q.explain);
        assert!(q.explain_raw);
        assert!(!q.explain_verbose);
    }

    #[test]
    fn explain_raw_verbose_prefix_combines() {
        let q = ok("EXPLAIN RAW VERBOSE MATCH (a:Person) RETURN a");
        assert!(q.explain);
        assert!(q.explain_raw);
        assert!(q.explain_verbose);
    }

    #[test]
    fn explain_raw_round_trips_through_display() {
        let q = ok("EXPLAIN RAW MATCH (a:Person) RETURN a");
        let rendered = q.to_string();
        assert!(rendered.starts_with("EXPLAIN RAW "), "got `{}`", rendered);
        let q2 = ok(&rendered);
        assert!(q2.explain);
        assert!(q2.explain_raw);
    }

    #[test]
    fn raw_alone_is_treated_as_identifier() {
        // RAW without EXPLAIN must NOT flip the flag. The token then
        // ends up as a clause keyword candidate, which fails the grammar.
        let _ = err_code("RAW MATCH (a) RETURN a");
    }

    // ─── reserved-keyword diagnostics (E003) ─────────────────────────────

    fn first_err(src: &str) -> ParseError {
        parse(src).expect_err("expected error").remove(0)
    }

    #[test]
    fn reserved_keyword_as_label_reports_e003() {
        let err = first_err("MATCH (n:MATCH) RETURN n");
        assert_eq!(err.code, ErrorCode::ReservedKeyword);
        assert!(
            err.message.contains("`MATCH`") && err.message.contains("reserved"),
            "message was: {}",
            err.message
        );
        let help = err.help.as_deref().expect("expected help text");
        assert!(help.contains('`'), "help should suggest backtick quoting");
    }

    #[test]
    fn reserved_keyword_as_variable_reports_e003() {
        let err = first_err("MATCH (RETURN) RETURN x");
        assert_eq!(err.code, ErrorCode::ReservedKeyword);
        assert!(err.message.contains("`RETURN`"));
    }

    #[test]
    fn reserved_keyword_as_property_key_reports_e003() {
        let err = first_err("MATCH (n {WHERE: 1}) RETURN n");
        assert_eq!(err.code, ErrorCode::ReservedKeyword);
    }

    #[test]
    fn backtick_quoted_reserved_keyword_is_accepted() {
        // The error path advertises backtick quoting as the escape hatch;
        // make sure that path actually parses.
        let q = ok("MATCH (n:`MATCH`) RETURN n");
        assert_eq!(q.head.clauses.len(), 2);
    }
}
