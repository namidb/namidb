//! Cypher parser.
//!
//! The entry point is [`parse`], which takes a Cypher string and returns
//! either a fully resolved [`ast::Query`] or a non-empty list of
//! [`error::ParseError`]s.
//!
//! The pipeline is:
//!
//! ```text
//! &str ──lex──▶ Vec<Spanned<Token>> ──parse──▶ Query
//! ```
//!
//! See [`docs/rfc/004-cypher-subset.md`](../../../../docs/rfc/004-cypher-subset.md)
//! for the exact subset of Cypher 25 / GQL that is accepted.

pub mod ast;
pub mod display;
pub mod error;
pub mod grammar;
pub mod lexer;

pub use ast::*;
pub use error::{ErrorCode, ParseError, ParseResult, SourceSpan};
pub use lexer::{lex, Spanned, Token};

/// Parse a Cypher source string into a [`Query`] AST.
///
/// On success returns `Ok(Query)`. On failure returns `Err(Vec<ParseError>)`
/// where the vector is guaranteed to be non-empty.
pub fn parse(src: &str) -> ParseResult<Query> {
 let tokens = lex(src).map_err(|e| vec![e])?;
 grammar::parse_query(src, tokens)
}

#[cfg(test)]
mod tests {
 use super::*;

 #[test]
 fn empty_input_is_an_error() {
 let result = parse("");
 assert!(result.is_err());
 }
}
