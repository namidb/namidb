//! Crate-local error enum.
//!
//! Codec errors surface to the session, which translates them into
//! `FAILURE { code, message }` responses with Neo4j-style dotted codes.
//! See RFC-022 §Errors for the mapping table.

use std::io;

use thiserror::Error;

#[derive(Debug, Error)]
pub enum BoltError {
    #[error("io: {0}")]
    Io(#[from] io::Error),

    #[error("unexpected end of input while decoding {what}")]
    UnexpectedEof { what: &'static str },

    #[error("invalid marker byte 0x{byte:02X} for {expected}")]
    InvalidMarker { byte: u8, expected: &'static str },

    #[error("invalid utf-8 in string: {0}")]
    InvalidUtf8(#[from] std::str::Utf8Error),

    #[error("value too large: {what} length {len} exceeds maximum {max}")]
    TooLarge {
        what: &'static str,
        len: usize,
        max: usize,
    },

    #[error("value nesting too deep: exceeds maximum depth {max}")]
    NestingTooDeep { max: usize },

    #[error("unsupported struct tag 0x{tag:02X}")]
    UnsupportedStruct { tag: u8 },

    #[error("malformed struct {struct_name}: {detail}")]
    MalformedStruct {
        struct_name: &'static str,
        detail: String,
    },

    #[error("handshake failed: {0}")]
    Handshake(String),

    #[error("protocol error: {0}")]
    Protocol(String),

    #[error("client sent message {message} in state {state:?} which does not accept it")]
    InvalidState {
        message: &'static str,
        state: crate::state::State,
    },
}

pub type Result<T> = std::result::Result<T, BoltError>;
