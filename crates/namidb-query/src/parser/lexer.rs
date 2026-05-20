//! Cypher lexer.
//!
//! Byte-by-byte hand-written tokeniser. Produces a `Vec<Spanned<Token>>` with
//! comments and whitespace stripped. Strings, identifiers, and numbers are
//! materialised into owned `String` (we don't try to be zero-copy at this
//! layer — the parser cost is negligible vs the executor cost).
//!
//! See RFC-004 §"Alternatives F" for why the lexer is a separate pass.

use std::fmt;

use super::error::{ErrorCode, ParseError, SourceSpan};

/// All token kinds the grammar inspects.
#[derive(Clone, Debug, PartialEq)]
pub enum Token {
    // ----- Literals -----
    Integer(i64),
    Float(f64),
    String(String),
    True,
    False,
    Null,

    // ----- Identifiers / parameters -----
    /// Plain identifier (`Person`, `a`, `name`).
    Ident(String),
    /// Backtick-quoted identifier; quotes are stripped, escapes resolved.
    QuotedIdent(String),
    /// `$name` parameter reference.
    Parameter(String),

    // ----- Keywords (case-insensitive in source, normalised here) -----
    Match,
    Optional,
    Where,
    Return,
    With,
    Order,
    By,
    Asc,
    Desc,
    Skip,
    Limit,
    Distinct,
    As,
    Unwind,
    Create,
    Merge,
    On,
    Set,
    Delete,
    Detach,
    Remove,
    Union,
    All,
    And,
    Or,
    Xor,
    Not,
    In,
    Is,
    StartsKw,
    EndsKw,
    Contains,
    Case,
    When,
    Then,
    Else,
    End,

    // ----- Operators / punctuation -----
    /// `+`
    Plus,
    /// `-`
    Minus,
    /// `*`
    Star,
    /// `/`
    Slash,
    /// `%`
    Percent,
    /// `^`
    Caret,
    /// `=`
    Eq,
    /// `<>`
    Ne,
    /// `<`
    Lt,
    /// `>`
    Gt,
    /// `<=`
    Le,
    /// `>=`
    Ge,
    /// `=~`
    RegexMatch,
    /// `.`
    Dot,
    /// `,`
    Comma,
    /// `;`
    Semicolon,
    /// `:`
    Colon,
    /// `::` (type cast)
    DoubleColon,
    /// `|`
    Pipe,
    /// `(`
    LParen,
    /// `)`
    RParen,
    /// `[`
    LBracket,
    /// `]`
    RBracket,
    /// `{`
    LBrace,
    /// `}`
    RBrace,
    /// `->`
    Arrow,
    /// `<-`
    ArrowLeft,
    /// `..` (variable-length range separator)
    Range,
}

impl Token {
    /// Human-readable name used in error messages ("integer", "MATCH", "(").
    pub fn label(&self) -> &'static str {
        match self {
            Token::Integer(_) => "integer literal",
            Token::Float(_) => "float literal",
            Token::String(_) => "string literal",
            Token::True => "TRUE",
            Token::False => "FALSE",
            Token::Null => "NULL",
            Token::Ident(_) => "identifier",
            Token::QuotedIdent(_) => "identifier",
            Token::Parameter(_) => "parameter",
            Token::Match => "MATCH",
            Token::Optional => "OPTIONAL",
            Token::Where => "WHERE",
            Token::Return => "RETURN",
            Token::With => "WITH",
            Token::Order => "ORDER",
            Token::By => "BY",
            Token::Asc => "ASC",
            Token::Desc => "DESC",
            Token::Skip => "SKIP",
            Token::Limit => "LIMIT",
            Token::Distinct => "DISTINCT",
            Token::As => "AS",
            Token::Unwind => "UNWIND",
            Token::Create => "CREATE",
            Token::Merge => "MERGE",
            Token::On => "ON",
            Token::Set => "SET",
            Token::Delete => "DELETE",
            Token::Detach => "DETACH",
            Token::Remove => "REMOVE",
            Token::Union => "UNION",
            Token::All => "ALL",
            Token::And => "AND",
            Token::Or => "OR",
            Token::Xor => "XOR",
            Token::Not => "NOT",
            Token::In => "IN",
            Token::Is => "IS",
            Token::StartsKw => "STARTS",
            Token::EndsKw => "ENDS",
            Token::Contains => "CONTAINS",
            Token::Case => "CASE",
            Token::When => "WHEN",
            Token::Then => "THEN",
            Token::Else => "ELSE",
            Token::End => "END",
            Token::Plus => "+",
            Token::Minus => "-",
            Token::Star => "*",
            Token::Slash => "/",
            Token::Percent => "%",
            Token::Caret => "^",
            Token::Eq => "=",
            Token::Ne => "<>",
            Token::Lt => "<",
            Token::Gt => ">",
            Token::Le => "<=",
            Token::Ge => ">=",
            Token::RegexMatch => "=~",
            Token::Dot => ".",
            Token::Comma => ",",
            Token::Semicolon => ";",
            Token::Colon => ":",
            Token::DoubleColon => "::",
            Token::Pipe => "|",
            Token::LParen => "(",
            Token::RParen => ")",
            Token::LBracket => "[",
            Token::RBracket => "]",
            Token::LBrace => "{",
            Token::RBrace => "}",
            Token::Arrow => "->",
            Token::ArrowLeft => "<-",
            Token::Range => "..",
        }
    }
}

impl fmt::Display for Token {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.label())
    }
}

/// A token paired with the source span it covers.
#[derive(Clone, Debug, PartialEq)]
pub struct Spanned<T> {
    pub value: T,
    pub span: SourceSpan,
}

impl<T> Spanned<T> {
    pub fn new(value: T, span: SourceSpan) -> Self {
        Self { value, span }
    }
}

/// Tokenise the entire source string. Returns either the full token stream
/// or the first lexer-level error (lexer doesn't recover; the parser is the
/// layer that can recover from token-level mistakes).
pub fn lex(src: &str) -> Result<Vec<Spanned<Token>>, ParseError> {
    let mut lx = Lexer::new(src);
    let mut tokens = Vec::with_capacity(src.len() / 6);
    loop {
        lx.skip_trivia()?;
        if lx.is_eof() {
            return Ok(tokens);
        }
        let tok = lx.next_token()?;
        tokens.push(tok);
    }
}

struct Lexer<'src> {
    src: &'src str,
    bytes: &'src [u8],
    pos: usize,
}

impl<'src> Lexer<'src> {
    fn new(src: &'src str) -> Self {
        Self {
            src,
            bytes: src.as_bytes(),
            pos: 0,
        }
    }

    fn is_eof(&self) -> bool {
        self.pos >= self.bytes.len()
    }

    fn peek(&self) -> Option<u8> {
        self.bytes.get(self.pos).copied()
    }

    fn peek_at(&self, offset: usize) -> Option<u8> {
        self.bytes.get(self.pos + offset).copied()
    }

    fn skip_trivia(&mut self) -> Result<(), ParseError> {
        loop {
            match self.peek() {
                Some(b) if b.is_ascii_whitespace() => {
                    self.pos += 1;
                }
                Some(b'/') if self.peek_at(1) == Some(b'/') => {
                    self.pos += 2;
                    while let Some(b) = self.peek() {
                        self.pos += 1;
                        if b == b'\n' {
                            break;
                        }
                    }
                }
                Some(b'/') if self.peek_at(1) == Some(b'*') => {
                    let start = self.pos;
                    self.pos += 2;
                    loop {
                        match self.peek() {
                            None => {
                                return Err(ParseError::new(
                                    ErrorCode::UnexpectedEof,
                                    "unterminated block comment",
                                    SourceSpan::new(start, self.pos),
                                ));
                            }
                            Some(b'*') if self.peek_at(1) == Some(b'/') => {
                                self.pos += 2;
                                break;
                            }
                            Some(_) => self.pos += 1,
                        }
                    }
                }
                _ => return Ok(()),
            }
        }
    }

    fn next_token(&mut self) -> Result<Spanned<Token>, ParseError> {
        let start = self.pos;
        let b = self.peek().ok_or_else(|| {
            ParseError::new(
                ErrorCode::UnexpectedEof,
                "unexpected end of input",
                SourceSpan::point(self.pos),
            )
        })?;

        let token = match b {
            b'(' => {
                self.pos += 1;
                Token::LParen
            }
            b')' => {
                self.pos += 1;
                Token::RParen
            }
            b'[' => {
                self.pos += 1;
                Token::LBracket
            }
            b']' => {
                self.pos += 1;
                Token::RBracket
            }
            b'{' => {
                self.pos += 1;
                Token::LBrace
            }
            b'}' => {
                self.pos += 1;
                Token::RBrace
            }
            b',' => {
                self.pos += 1;
                Token::Comma
            }
            b';' => {
                self.pos += 1;
                Token::Semicolon
            }
            b'+' => {
                self.pos += 1;
                Token::Plus
            }
            b'*' => {
                self.pos += 1;
                Token::Star
            }
            b'/' => {
                self.pos += 1;
                Token::Slash
            }
            b'%' => {
                self.pos += 1;
                Token::Percent
            }
            b'^' => {
                self.pos += 1;
                Token::Caret
            }
            b'|' => {
                self.pos += 1;
                Token::Pipe
            }
            b':' => {
                if self.peek_at(1) == Some(b':') {
                    self.pos += 2;
                    Token::DoubleColon
                } else {
                    self.pos += 1;
                    Token::Colon
                }
            }
            b'.' => {
                if self.peek_at(1) == Some(b'.') {
                    self.pos += 2;
                    Token::Range
                } else if matches!(self.peek_at(1), Some(d) if d.is_ascii_digit()) {
                    self.read_number()?
                } else {
                    self.pos += 1;
                    Token::Dot
                }
            }
            b'-' => {
                if self.peek_at(1) == Some(b'>') {
                    self.pos += 2;
                    Token::Arrow
                } else {
                    self.pos += 1;
                    Token::Minus
                }
            }
            b'<' => match self.peek_at(1) {
                Some(b'=') => {
                    self.pos += 2;
                    Token::Le
                }
                Some(b'>') => {
                    self.pos += 2;
                    Token::Ne
                }
                Some(b'-') => {
                    self.pos += 2;
                    Token::ArrowLeft
                }
                _ => {
                    self.pos += 1;
                    Token::Lt
                }
            },
            b'>' => {
                if self.peek_at(1) == Some(b'=') {
                    self.pos += 2;
                    Token::Ge
                } else {
                    self.pos += 1;
                    Token::Gt
                }
            }
            b'=' => {
                if self.peek_at(1) == Some(b'~') {
                    self.pos += 2;
                    Token::RegexMatch
                } else {
                    self.pos += 1;
                    Token::Eq
                }
            }
            b'$' => {
                self.pos += 1;
                let name_start = self.pos;
                while matches!(self.peek(), Some(c) if is_ident_continue(c)) {
                    self.pos += 1;
                }
                if name_start == self.pos {
                    return Err(ParseError::new(
                        ErrorCode::UnexpectedToken,
                        "empty parameter name after `$`",
                        SourceSpan::new(start, self.pos),
                    ));
                }
                Token::Parameter(self.src[name_start..self.pos].to_string())
            }
            b'\'' | b'"' => self.read_string(b)?,
            b'`' => self.read_quoted_ident()?,
            b'0'..=b'9' => self.read_number()?,
            c if is_ident_start(c) => self.read_word(),
            other => {
                let span = SourceSpan::new(start, start + 1);
                self.pos += 1;
                return Err(ParseError::new(
                    ErrorCode::UnexpectedToken,
                    format!("unexpected character `{}`", other as char),
                    span,
                ));
            }
        };

        Ok(Spanned::new(token, SourceSpan::new(start, self.pos)))
    }

    fn read_number(&mut self) -> Result<Token, ParseError> {
        let start = self.pos;
        let mut is_float = false;

        if self.peek() == Some(b'.') {
            is_float = true;
            self.pos += 1;
        }

        while matches!(self.peek(), Some(c) if c.is_ascii_digit()) {
            self.pos += 1;
        }

        if !is_float
            && self.peek() == Some(b'.')
            && matches!(self.peek_at(1), Some(c) if c.is_ascii_digit())
        {
            is_float = true;
            self.pos += 1;
            while matches!(self.peek(), Some(c) if c.is_ascii_digit()) {
                self.pos += 1;
            }
        }

        if matches!(self.peek(), Some(b'e') | Some(b'E')) {
            is_float = true;
            self.pos += 1;
            if matches!(self.peek(), Some(b'+') | Some(b'-')) {
                self.pos += 1;
            }
            let exp_digits_start = self.pos;
            while matches!(self.peek(), Some(c) if c.is_ascii_digit()) {
                self.pos += 1;
            }
            if exp_digits_start == self.pos {
                return Err(ParseError::new(
                    ErrorCode::InvalidNumber,
                    "exponent has no digits",
                    SourceSpan::new(start, self.pos),
                ));
            }
        }

        let text = &self.src[start..self.pos];
        if is_float {
            text.parse::<f64>().map(Token::Float).map_err(|_| {
                ParseError::new(
                    ErrorCode::InvalidNumber,
                    format!("`{}` is not a valid float", text),
                    SourceSpan::new(start, self.pos),
                )
            })
        } else {
            text.parse::<i64>().map(Token::Integer).map_err(|_| {
                ParseError::new(
                    ErrorCode::InvalidNumber,
                    format!("`{}` overflows i64 or is not an integer", text),
                    SourceSpan::new(start, self.pos),
                )
            })
        }
    }

    fn read_string(&mut self, quote: u8) -> Result<Token, ParseError> {
        let start = self.pos;
        self.pos += 1; // opening quote
        let mut out = String::new();
        loop {
            let b = self.peek().ok_or_else(|| {
                ParseError::new(
                    ErrorCode::UnterminatedString,
                    "string literal is not closed",
                    SourceSpan::new(start, self.pos),
                )
            })?;
            match b {
                b if b == quote => {
                    self.pos += 1;
                    return Ok(Token::String(out));
                }
                b'\\' => {
                    let esc_start = self.pos;
                    self.pos += 1;
                    let esc = self.peek().ok_or_else(|| {
                        ParseError::new(
                            ErrorCode::UnterminatedString,
                            "string literal is not closed",
                            SourceSpan::new(start, self.pos),
                        )
                    })?;
                    let resolved = match esc {
                        b'\\' => '\\',
                        b'\'' => '\'',
                        b'"' => '"',
                        b'`' => '`',
                        b'n' => '\n',
                        b'r' => '\r',
                        b't' => '\t',
                        b'0' => '\0',
                        other => {
                            return Err(ParseError::new(
                                ErrorCode::InvalidEscape,
                                format!("unknown escape `\\{}`", other as char),
                                SourceSpan::new(esc_start, self.pos + 1),
                            ));
                        }
                    };
                    out.push(resolved);
                    self.pos += 1;
                }
                _ => {
                    let ch_start = self.pos;
                    let ch = utf8_char(self.bytes, self.pos).ok_or_else(|| {
                        ParseError::new(
                            ErrorCode::UnexpectedToken,
                            "invalid UTF-8 inside string literal",
                            SourceSpan::new(ch_start, ch_start + 1),
                        )
                    })?;
                    out.push(ch);
                    self.pos += ch.len_utf8();
                }
            }
        }
    }

    fn read_quoted_ident(&mut self) -> Result<Token, ParseError> {
        let start = self.pos;
        self.pos += 1; // opening backtick
        let mut out = String::new();
        loop {
            let b = self.peek().ok_or_else(|| {
                ParseError::new(
                    ErrorCode::UnterminatedString,
                    "backtick-quoted identifier is not closed",
                    SourceSpan::new(start, self.pos),
                )
            })?;
            match b {
                b'`' => {
                    // Either close, or doubled backtick = literal backtick.
                    if self.peek_at(1) == Some(b'`') {
                        out.push('`');
                        self.pos += 2;
                    } else {
                        self.pos += 1;
                        if out.is_empty() {
                            return Err(ParseError::new(
                                ErrorCode::UnexpectedToken,
                                "empty backtick identifier",
                                SourceSpan::new(start, self.pos),
                            ));
                        }
                        return Ok(Token::QuotedIdent(out));
                    }
                }
                _ => {
                    let ch_start = self.pos;
                    let ch = utf8_char(self.bytes, self.pos).ok_or_else(|| {
                        ParseError::new(
                            ErrorCode::UnexpectedToken,
                            "invalid UTF-8 inside identifier",
                            SourceSpan::new(ch_start, ch_start + 1),
                        )
                    })?;
                    out.push(ch);
                    self.pos += ch.len_utf8();
                }
            }
        }
    }

    fn read_word(&mut self) -> Token {
        let start = self.pos;
        while matches!(self.peek(), Some(c) if is_ident_continue(c)) {
            self.pos += 1;
        }
        let text = &self.src[start..self.pos];
        keyword_or_ident(text)
    }
}

fn is_ident_start(b: u8) -> bool {
    b.is_ascii_alphabetic() || b == b'_'
}

fn is_ident_continue(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_'
}

fn keyword_or_ident(text: &str) -> Token {
    let upper = text.to_ascii_uppercase();
    match upper.as_str() {
        "MATCH" => Token::Match,
        "OPTIONAL" => Token::Optional,
        "WHERE" => Token::Where,
        "RETURN" => Token::Return,
        "WITH" => Token::With,
        "ORDER" => Token::Order,
        "BY" => Token::By,
        "ASC" | "ASCENDING" => Token::Asc,
        "DESC" | "DESCENDING" => Token::Desc,
        "SKIP" => Token::Skip,
        "LIMIT" => Token::Limit,
        "DISTINCT" => Token::Distinct,
        "AS" => Token::As,
        "UNWIND" => Token::Unwind,
        "CREATE" => Token::Create,
        "MERGE" => Token::Merge,
        "ON" => Token::On,
        "SET" => Token::Set,
        "DELETE" => Token::Delete,
        "DETACH" => Token::Detach,
        "REMOVE" => Token::Remove,
        "UNION" => Token::Union,
        "ALL" => Token::All,
        "AND" => Token::And,
        "OR" => Token::Or,
        "XOR" => Token::Xor,
        "NOT" => Token::Not,
        "IN" => Token::In,
        "IS" => Token::Is,
        "STARTS" => Token::StartsKw,
        "ENDS" => Token::EndsKw,
        "CONTAINS" => Token::Contains,
        "CASE" => Token::Case,
        "WHEN" => Token::When,
        "THEN" => Token::Then,
        "ELSE" => Token::Else,
        "END" => Token::End,
        "TRUE" => Token::True,
        "FALSE" => Token::False,
        "NULL" => Token::Null,
        _ => Token::Ident(text.to_string()),
    }
}

fn utf8_char(bytes: &[u8], pos: usize) -> Option<char> {
    let first = *bytes.get(pos)?;
    let width = if first < 0x80 {
        1
    } else if first < 0xC0 {
        return None;
    } else if first < 0xE0 {
        2
    } else if first < 0xF0 {
        3
    } else {
        4
    };
    if pos + width > bytes.len() {
        return None;
    }
    std::str::from_utf8(&bytes[pos..pos + width])
        .ok()
        .and_then(|s| s.chars().next())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn toks(src: &str) -> Vec<Token> {
        lex(src).unwrap().into_iter().map(|s| s.value).collect()
    }

    #[test]
    fn empty_source_yields_no_tokens() {
        assert_eq!(toks(""), vec![]);
        assert_eq!(toks(" \n\t "), vec![]);
    }

    #[test]
    fn keywords_are_case_insensitive() {
        let kws = ["match", "MATCH", "Match", "mAtCh"];
        for k in kws {
            assert_eq!(toks(k), vec![Token::Match], "failed for `{}`", k);
        }
    }

    #[test]
    fn match_pattern_tokenises() {
        let got = toks("MATCH (a:Person)-[r:KNOWS]->(b) RETURN a");
        assert_eq!(
            got,
            vec![
                Token::Match,
                Token::LParen,
                Token::Ident("a".into()),
                Token::Colon,
                Token::Ident("Person".into()),
                Token::RParen,
                Token::Minus,
                Token::LBracket,
                Token::Ident("r".into()),
                Token::Colon,
                Token::Ident("KNOWS".into()),
                Token::RBracket,
                Token::Arrow,
                Token::LParen,
                Token::Ident("b".into()),
                Token::RParen,
                Token::Return,
                Token::Ident("a".into()),
            ]
        );
    }

    #[test]
    fn arrow_left_and_undirected() {
        let got = toks("(a)<-[r]-(b)");
        assert_eq!(
            got,
            vec![
                Token::LParen,
                Token::Ident("a".into()),
                Token::RParen,
                Token::ArrowLeft,
                Token::LBracket,
                Token::Ident("r".into()),
                Token::RBracket,
                Token::Minus,
                Token::LParen,
                Token::Ident("b".into()),
                Token::RParen,
            ]
        );
    }

    #[test]
    fn integers_and_floats() {
        let got = toks("0 42 -7 2.5 .5 6.022e23 1E-9");
        assert_eq!(
            got,
            vec![
                Token::Integer(0),
                Token::Integer(42),
                Token::Minus,
                Token::Integer(7),
                Token::Float(2.5),
                Token::Float(0.5),
                Token::Float(6.022e23),
                Token::Float(1e-9),
            ]
        );
    }

    #[test]
    fn strings_with_escapes() {
        let got = toks(r#" 'hello' "world" 'line\nbreak' "quote\"in" "#);
        assert_eq!(
            got,
            vec![
                Token::String("hello".into()),
                Token::String("world".into()),
                Token::String("line\nbreak".into()),
                Token::String("quote\"in".into()),
            ]
        );
    }

    #[test]
    fn unterminated_string_errors_with_code_e004() {
        let err = lex("'no closing").unwrap_err();
        assert_eq!(err.code, ErrorCode::UnterminatedString);
    }

    #[test]
    fn backtick_identifier_with_space() {
        let got = toks("MATCH (a:`Foo Bar`)");
        // Tokens: MATCH, (, a, :, `Foo Bar`, )
        assert!(matches!(got[4], Token::QuotedIdent(ref s) if s == "Foo Bar"));
    }

    #[test]
    fn parameter_with_name() {
        let got = toks("$personId");
        assert_eq!(got, vec![Token::Parameter("personId".into())]);
    }

    #[test]
    fn comparison_operators() {
        let got = toks("= <> < > <= >= =~");
        assert_eq!(
            got,
            vec![
                Token::Eq,
                Token::Ne,
                Token::Lt,
                Token::Gt,
                Token::Le,
                Token::Ge,
                Token::RegexMatch,
            ]
        );
    }

    #[test]
    fn variable_length_range() {
        let got = toks("*1..3");
        assert_eq!(
            got,
            vec![
                Token::Star,
                Token::Integer(1),
                Token::Range,
                Token::Integer(3)
            ]
        );
    }

    #[test]
    fn line_comment_is_stripped() {
        let got = toks("MATCH // comment until eol\n (a) RETURN a");
        assert!(!got
            .iter()
            .any(|t| matches!(t, Token::Ident(s) if s == "comment")));
        assert_eq!(got.first(), Some(&Token::Match));
    }

    #[test]
    fn block_comment_is_stripped() {
        let got = toks("MATCH /* multi\n line */ (a)");
        assert_eq!(
            got,
            vec![
                Token::Match,
                Token::LParen,
                Token::Ident("a".into()),
                Token::RParen
            ]
        );
    }

    #[test]
    fn block_comment_unterminated_errors() {
        let err = lex("MATCH /* unclosed").unwrap_err();
        assert_eq!(err.code, ErrorCode::UnexpectedEof);
    }

    #[test]
    fn span_points_to_token_bytes() {
        let toks = lex("MATCH (a)").unwrap();
        assert_eq!(toks[0].span, SourceSpan::new(0, 5));
        assert_eq!(toks[1].span, SourceSpan::new(6, 7));
        assert_eq!(toks[2].span, SourceSpan::new(7, 8));
        assert_eq!(toks[3].span, SourceSpan::new(8, 9));
    }

    #[test]
    fn double_colon_is_one_token() {
        let got = toks("a::INT");
        assert_eq!(
            got,
            vec![
                Token::Ident("a".into()),
                Token::DoubleColon,
                Token::Ident("INT".into()),
            ]
        );
    }

    #[test]
    fn range_dotdot_is_one_token() {
        let got = toks("1..10");
        assert_eq!(
            got,
            vec![Token::Integer(1), Token::Range, Token::Integer(10)]
        );
    }
}
