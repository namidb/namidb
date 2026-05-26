//! Opaque cursor helpers for offset-style pagination.
//!
//! v0 is deliberately small: a cursor carries the next `skip` value
//! plus a version tag. Callers (the cloud worker, the CLI, embedded
//! SDKs) round-trip the encoded string verbatim — they do not parse
//! it themselves. That keeps the door open for adding a stable-key
//! component (`WHERE _id > value`) later without breaking clients.
//!
//! Why offset cursors instead of going straight to keyset:
//!
//! * The executor is already `Vec<Row>`-eager and supports
//!   `SKIP` / `LIMIT` natively. Wrapping that in a token shape gives
//!   us paginated APIs today.
//! * Keyset cursors need a stable sort key and a streaming executor.
//!   Both are real follow-ups, but neither is needed to unblock the
//!   dashboard's paginated tables right now.
//!
//! If you need stable pagination under concurrent inserts, ORDER BY
//! a unique key and translate the cursor server-side. This module
//! does the offset half; the keyset half lands when the executor
//! becomes streaming.

use crate::plan::LogicalPlan;
use crate::plan::OrderKey;

/// Current wire-format version. Bumped if the encoded shape changes.
const CURSOR_PREFIX: &str = "v1:";

/// Opaque pagination cursor. Encoded as a short ASCII string; the
/// engine treats it as a black box on input and only emits valid
/// tokens on output. Future versions can extend [`Cursor`] with extra
/// fields and a higher prefix without rejecting old tokens at decode
/// time (each prefix has its own parser).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Cursor {
    /// Number of rows to skip before returning the next page. Combined
    /// with the caller-supplied page size, this becomes the
    /// `TopN.skip` / `TopN.limit` of the executed plan.
    pub skip: u64,
}

/// Errors surfaced by [`Cursor::decode`].
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum CursorError {
    #[error("unknown cursor version (expected `{CURSOR_PREFIX}<n>`, got `{0}`)")]
    UnknownVersion(String),
    #[error("cursor skip value is not a non-negative integer: `{0}`")]
    InvalidSkip(String),
}

impl Cursor {
    /// Build a cursor that asks the executor to skip `skip` rows
    /// before the next page.
    pub fn from_skip(skip: u64) -> Self {
        Self { skip }
    }

    /// Encode the cursor as a short ASCII string. The shape is
    /// `v1:<decimal-skip>`; it stays printable so callers can include
    /// it in URL query strings or response JSON without extra
    /// escaping.
    pub fn encode(&self) -> String {
        format!("{CURSOR_PREFIX}{}", self.skip)
    }

    /// Parse an encoded cursor. Rejects unknown prefixes (so a stale
    /// token from a future engine version surfaces a clear error
    /// instead of silently turning into `skip = 0`).
    pub fn decode(s: &str) -> Result<Self, CursorError> {
        let rest = s
            .strip_prefix(CURSOR_PREFIX)
            .ok_or_else(|| CursorError::UnknownVersion(s.to_string()))?;
        let skip = rest
            .parse::<u64>()
            .map_err(|_| CursorError::InvalidSkip(rest.to_string()))?;
        Ok(Self { skip })
    }
}

/// Produce a plan that paginates `plan` against the supplied cursor.
///
/// * `cursor = None` → start from the beginning (skip = 0).
/// * `cursor = Some(c)` → resume at `c.skip`.
///
/// The resulting plan wraps the input in a `TopN` so the executor
/// applies `skip` and `limit` even when the input had no order or
/// pre-existing pagination. If `plan` is *already* a `TopN`, its
/// `skip` / `limit` are replaced — the cursor wins.
///
/// `page_size` of zero is treated as "no limit" so the executor
/// returns every remaining row past `cursor.skip`. Callers should
/// pick a sensible default (the dashboard uses 500); this just keeps
/// the cap explicit.
pub fn paginate_plan(plan: LogicalPlan, cursor: Option<&Cursor>, page_size: u64) -> LogicalPlan {
    let skip = cursor.map(|c| c.skip).unwrap_or(0);
    let limit = if page_size == 0 { u64::MAX } else { page_size };
    match plan {
        LogicalPlan::TopN { input, keys, .. } => LogicalPlan::TopN {
            input,
            keys,
            skip,
            limit,
        },
        other => LogicalPlan::TopN {
            input: Box::new(other),
            keys: Vec::<OrderKey>::new(),
            skip,
            limit,
        },
    }
}

/// Build the cursor the caller should hand back to the executor for
/// the next page. Returns `None` when the just-served page contained
/// fewer than `page_size` rows — that signals the caller "you have
/// reached the end".
pub fn next_cursor(
    current: Option<&Cursor>,
    returned_rows: usize,
    page_size: u64,
) -> Option<Cursor> {
    if page_size == 0 || (returned_rows as u64) < page_size {
        return None;
    }
    let prev_skip = current.map(|c| c.skip).unwrap_or(0);
    Some(Cursor::from_skip(prev_skip + page_size))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plan::LogicalPlan;

    fn empty_plan() -> LogicalPlan {
        LogicalPlan::Empty
    }

    #[test]
    fn cursor_encode_decode_round_trip() {
        let c = Cursor::from_skip(0);
        assert_eq!(c.encode(), "v1:0");
        assert_eq!(Cursor::decode("v1:0").unwrap(), c);

        let c = Cursor::from_skip(12_345);
        assert_eq!(c.encode(), "v1:12345");
        assert_eq!(Cursor::decode("v1:12345").unwrap(), c);
    }

    #[test]
    fn cursor_decode_rejects_unknown_prefix() {
        let err = Cursor::decode("v2:500").unwrap_err();
        assert!(matches!(err, CursorError::UnknownVersion(_)));
    }

    #[test]
    fn cursor_decode_rejects_non_numeric_skip() {
        let err = Cursor::decode("v1:nope").unwrap_err();
        assert!(matches!(err, CursorError::InvalidSkip(_)));
    }

    #[test]
    fn paginate_plan_wraps_plain_input_in_topn() {
        let plan = paginate_plan(empty_plan(), Some(&Cursor::from_skip(7)), 50);
        match plan {
            LogicalPlan::TopN {
                skip, limit, keys, ..
            } => {
                assert_eq!(skip, 7);
                assert_eq!(limit, 50);
                assert!(keys.is_empty());
            }
            other => panic!("expected TopN, got {:?}", other),
        }
    }

    #[test]
    fn paginate_plan_overrides_existing_topn() {
        let inner = LogicalPlan::TopN {
            input: Box::new(LogicalPlan::Empty),
            keys: vec![],
            skip: 999,
            limit: 999,
        };
        let plan = paginate_plan(inner, Some(&Cursor::from_skip(10)), 20);
        match plan {
            LogicalPlan::TopN { skip, limit, .. } => {
                assert_eq!(skip, 10);
                assert_eq!(limit, 20);
            }
            other => panic!("expected TopN, got {:?}", other),
        }
    }

    #[test]
    fn next_cursor_signals_end_when_page_short() {
        // Full page → more to fetch.
        assert_eq!(
            next_cursor(Some(&Cursor::from_skip(50)), 50, 50),
            Some(Cursor::from_skip(100))
        );
        // Short page → reached the end.
        assert_eq!(next_cursor(Some(&Cursor::from_skip(50)), 17, 50), None);
        // Zero page size means "no limit", so there is never a next.
        assert_eq!(next_cursor(None, 1_000_000, 0), None);
    }
}
