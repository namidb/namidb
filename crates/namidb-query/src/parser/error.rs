//! Error model for the Cypher parser.
//!
//! See RFC-004 §"Error model".

use std::fmt;

use serde::{Deserialize, Serialize};

/// Byte-range in the source string. Half-open `[start, end)`.
///
/// Carries `Serialize` + `Deserialize` so the lowered `LogicalPlan`,
/// which threads `SourceSpan` through every expression for diagnostic
/// spans, can round-trip through a cross-process plan cache without
/// stripping the source coordinates.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SourceSpan {
    pub start: usize,
    pub end: usize,
}

impl SourceSpan {
    pub const fn new(start: usize, end: usize) -> Self {
        Self { start, end }
    }

    pub const fn point(at: usize) -> Self {
        Self { start: at, end: at }
    }

    pub fn len(&self) -> usize {
        self.end - self.start
    }

    pub fn is_empty(&self) -> bool {
        self.start == self.end
    }

    pub fn merge(self, other: SourceSpan) -> SourceSpan {
        SourceSpan {
            start: self.start.min(other.start),
            end: self.end.max(other.end),
        }
    }

    pub fn slice<'src>(&self, src: &'src str) -> &'src str {
        &src[self.start..self.end]
    }
}

impl fmt::Display for SourceSpan {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{}..{})", self.start, self.end)
    }
}

/// Stable error codes — referenced by `ParseError`. Each variant has a
/// canonical short string `Exxx` used in CLI / SDK error reporting.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ErrorCode {
    /// E001 — token did not match the expected production.
    UnexpectedToken,
    /// E002 — variable-length pattern without finite upper bound.
    UnboundedVariableLength,
    /// E003 — identifier collides with a reserved keyword.
    ReservedKeyword,
    /// E004 — unterminated string literal.
    UnterminatedString,
    /// E005 — number literal cannot be parsed.
    InvalidNumber,
    /// E006 — feature deferred to a future RFC.
    UnsupportedFeature,
    /// E007 — MERGE pattern node carries multiple labels (RFC-004 drawback 4).
    MergeMultiLabel,
    /// E008 — OPTIONAL MATCH combined with variable-length (RFC-004 drawback 5).
    OptionalVariableLength,
    /// E009 — invalid escape sequence inside a string literal.
    InvalidEscape,
    /// E010 — input ended while a production was still open.
    UnexpectedEof,
    /// E011 — duplicate aliasing within the same projection scope.
    DuplicateAlias,
}

impl ErrorCode {
    /// Short canonical string (`"E001"`, `"E002"`...). Stable across releases.
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::UnexpectedToken => "E001",
            Self::UnboundedVariableLength => "E002",
            Self::ReservedKeyword => "E003",
            Self::UnterminatedString => "E004",
            Self::InvalidNumber => "E005",
            Self::UnsupportedFeature => "E006",
            Self::MergeMultiLabel => "E007",
            Self::OptionalVariableLength => "E008",
            Self::InvalidEscape => "E009",
            Self::UnexpectedEof => "E010",
            Self::DuplicateAlias => "E011",
        }
    }
}

impl fmt::Display for ErrorCode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// A single parser error. Multiple errors may be returned per `parse` call
/// when the recursive-descent grammar can resynchronise on a clause boundary.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ParseError {
    pub code: ErrorCode,
    pub message: String,
    pub span: SourceSpan,
    pub help: Option<String>,
}

impl ParseError {
    pub fn new(code: ErrorCode, message: impl Into<String>, span: SourceSpan) -> Self {
        Self {
            code,
            message: message.into(),
            span,
            help: None,
        }
    }

    pub fn with_help(mut self, help: impl Into<String>) -> Self {
        self.help = Some(help.into());
        self
    }
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {} at {}", self.code, self.message, self.span)?;
        if let Some(help) = &self.help {
            write!(f, " — help: {}", help)?;
        }
        Ok(())
    }
}

impl std::error::Error for ParseError {}

/// Result returned by `parse`. Carries either a successful AST or a non-empty
/// list of errors (recovery makes "no errors but failed" impossible).
pub type ParseResult<T> = Result<T, Vec<ParseError>>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn span_merge_extends_both_ends() {
        let a = SourceSpan::new(2, 5);
        let b = SourceSpan::new(4, 9);
        let m = a.merge(b);
        assert_eq!(m, SourceSpan::new(2, 9));
    }

    #[test]
    fn span_slice_returns_substring() {
        let span = SourceSpan::new(7, 13);
        assert_eq!(span.slice("MATCH (a:Person) RETURN a"), "a:Pers");
    }

    #[test]
    fn error_code_strings_are_stable() {
        assert_eq!(ErrorCode::UnexpectedToken.as_str(), "E001");
        assert_eq!(ErrorCode::UnsupportedFeature.as_str(), "E006");
        assert_eq!(ErrorCode::DuplicateAlias.as_str(), "E011");
    }

    #[test]
    fn error_display_carries_help() {
        let err = ParseError::new(
            ErrorCode::UnexpectedToken,
            "expected MATCH",
            SourceSpan::new(0, 5),
        )
        .with_help("did you mean `MATCH`?");
        let s = err.to_string();
        assert!(s.contains("E001"));
        assert!(s.contains("expected MATCH"));
        assert!(s.contains("did you mean"));
    }
}
