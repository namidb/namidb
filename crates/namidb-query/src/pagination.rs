//! Opaque cursor helpers for paginated queries.
//!
//! Two shapes ship side by side:
//!
//! * **Offset (`v1`)** — [`Cursor`] + [`paginate_plan`] + [`next_cursor`].
//!   Wraps the plan in a `TopN { skip, limit }` and the cursor carries
//!   the next `skip` value. Simple, works for any plan, but the cost
//!   degrades linearly with the offset (the executor still walks the
//!   skipped rows).
//!
//! * **Keyset (`v2`)** — [`CursorKeyset`] + [`paginate_plan_keyset`] +
//!   [`next_cursor_keyset`]. Carries the last `_id` returned plus a
//!   plan hash, and rewrites the plan into
//!   `WHERE alias._id > cursor.last_id ORDER BY alias._id ASC LIMIT page_size`.
//!   The cost stays flat across deep pages because the executor only
//!   ever touches `page_size` rows. The caller is responsible for
//!   picking the alias whose `_id` defines the keyset ordering.
//!
//! The wire shapes share a `v<N>:` prefix so a stale or unknown
//! variant surfaces a clear `CursorError::UnknownVersion` instead of
//! silently turning into a different page.

use crate::parser::{
    BinaryOp, Expression, ExpressionKind, Identifier, Literal, OrderDirection, PropertyAccess,
    SourceSpan,
};
use crate::plan::{LogicalPlan, OrderKey};

/// Wire-format version for the offset cursor.
const CURSOR_PREFIX: &str = "v1:";
/// Wire-format version for the keyset cursor.
const KEYSET_PREFIX: &str = "v2:";

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

/// Errors surfaced by [`Cursor::decode`] / [`CursorKeyset::decode`].
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum CursorError {
    #[error("unknown cursor version (expected `{CURSOR_PREFIX}<n>` or `{KEYSET_PREFIX}<hash>:<id>`, got `{0}`)")]
    UnknownVersion(String),
    #[error("cursor skip value is not a non-negative integer: `{0}`")]
    InvalidSkip(String),
    #[error("keyset cursor payload is malformed (expected `<hash_hex>:<last_id>`, got `{0}`)")]
    InvalidKeyset(String),
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

/// Keyset pagination cursor (RFC-pending).
///
/// Carries the last `_id` the previous page returned plus a hash of
/// the plan it was issued against. Callers should reject a cursor
/// whose `plan_hash` does not match the current
/// [`query_text_hash`](crate::query_text_hash) of the request —
/// resuming on a different plan would silently skip or duplicate
/// rows.
///
/// The wire shape is `v2:<plan_hash_hex>:<last_id>`, where
/// `plan_hash` is the same `u64` xxh3 fingerprint used by the
/// plan cache and `last_id` is the alias-resolved `_id` of the last
/// row of the previous page. `last_id` is opaque to this module;
/// callers convert through [`NodeId`](namidb_core::id::NodeId)::to_string
/// before encoding.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CursorKeyset {
    pub plan_hash: u64,
    pub last_id: String,
}

impl CursorKeyset {
    pub fn new(plan_hash: u64, last_id: impl Into<String>) -> Self {
        Self {
            plan_hash,
            last_id: last_id.into(),
        }
    }

    /// Encode as `v2:<plan_hash_hex>:<last_id>`.
    pub fn encode(&self) -> String {
        format!("{}{:016x}:{}", KEYSET_PREFIX, self.plan_hash, self.last_id)
    }

    /// Parse a `v2:`-prefixed token. Rejects unknown prefixes and
    /// malformed hash bytes; `last_id` is taken verbatim (the engine
    /// does not validate it is a NodeId because the alias might point
    /// at any string-keyed entity in the future).
    pub fn decode(s: &str) -> Result<Self, CursorError> {
        let rest = s
            .strip_prefix(KEYSET_PREFIX)
            .ok_or_else(|| CursorError::UnknownVersion(s.to_string()))?;
        let (hash_part, last_id) = rest
            .split_once(':')
            .ok_or_else(|| CursorError::InvalidKeyset(rest.to_string()))?;
        let plan_hash = u64::from_str_radix(hash_part, 16)
            .map_err(|_| CursorError::InvalidKeyset(hash_part.to_string()))?;
        if last_id.is_empty() {
            return Err(CursorError::InvalidKeyset(rest.to_string()));
        }
        Ok(Self {
            plan_hash,
            last_id: last_id.to_string(),
        })
    }
}

/// Wrap `plan` with a keyset filter / sort / limit so the executor
/// returns at most `page_size` rows that come *after*
/// `cursor.last_id` in `<key_alias>._id` order.
///
/// * `cursor = None` → no filter, just `ORDER BY <alias>._id ASC LIMIT N`.
/// * `cursor = Some(c)` → `WHERE <alias>._id > c.last_id` precedes
///   the order + limit.
///
/// The caller is responsible for verifying `cursor.plan_hash` matches
/// the current query before passing the cursor in. Mismatched plans
/// must reject the request with a fresh page-1 response.
///
/// `page_size = 0` means "no limit" so the keyset trims everything
/// at and below `last_id` but otherwise returns all remaining rows.
pub fn paginate_plan_keyset(
    plan: LogicalPlan,
    cursor: Option<&CursorKeyset>,
    page_size: u64,
    key_alias: &str,
) -> LogicalPlan {
    let span = SourceSpan::point(0);
    let limit = if page_size == 0 { u64::MAX } else { page_size };

    let plan = if let Some(c) = cursor {
        let id_access = property_access(key_alias, "_id", span);
        let last_id_lit = Expression {
            kind: ExpressionKind::Literal(Literal::String(c.last_id.clone())),
            span,
        };
        let predicate = Expression {
            kind: ExpressionKind::Binary {
                op: BinaryOp::Gt,
                left: Box::new(id_access),
                right: Box::new(last_id_lit),
            },
            span,
        };
        LogicalPlan::Filter {
            input: Box::new(plan),
            predicate,
        }
    } else {
        plan
    };

    let order_key = OrderKey {
        expression: property_access(key_alias, "_id", span),
        direction: OrderDirection::Asc,
    };

    LogicalPlan::TopN {
        input: Box::new(plan),
        keys: vec![order_key],
        skip: 0,
        limit,
    }
}

/// Build a [`CursorKeyset`] from the last row the executor handed
/// back. Returns `None` when the page came back short of `page_size`
/// (end of stream) or when `page_size` is zero (no limit means there
/// is no next page).
///
/// `last_id` is opaque: the caller extracts it from the bound row
/// (`row.get(key_alias)`'s NodeId → string) and passes it verbatim.
pub fn next_cursor_keyset(
    plan_hash: u64,
    returned_rows: usize,
    page_size: u64,
    last_id: Option<&str>,
) -> Option<CursorKeyset> {
    if page_size == 0 || (returned_rows as u64) < page_size {
        return None;
    }
    let last_id = last_id?.to_string();
    Some(CursorKeyset { plan_hash, last_id })
}

fn property_access(alias: &str, key: &str, span: SourceSpan) -> Expression {
    let target = Expression {
        kind: ExpressionKind::Variable(Identifier::new(alias, span)),
        span,
    };
    Expression {
        kind: ExpressionKind::Property(Box::new(PropertyAccess {
            target,
            key: Identifier::new(key, span),
            span,
        })),
        span,
    }
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

    #[test]
    fn cursor_keyset_encode_decode_round_trip() {
        let c = CursorKeyset::new(0xDEAD_BEEF_DEAD_BEEF, "01J5XY7K");
        let s = c.encode();
        assert!(
            s.starts_with("v2:deadbeefdeadbeef:"),
            "unexpected wire shape: {s}"
        );
        let back = CursorKeyset::decode(&s).expect("round-trips");
        assert_eq!(back, c);
    }

    #[test]
    fn cursor_keyset_decode_rejects_offset_prefix() {
        let err = CursorKeyset::decode("v1:0").expect_err("offset cursor must not decode here");
        assert!(matches!(err, CursorError::UnknownVersion(_)));
    }

    #[test]
    fn cursor_keyset_decode_rejects_missing_payload() {
        let err = CursorKeyset::decode("v2:nope").expect_err("malformed payload");
        assert!(matches!(err, CursorError::InvalidKeyset(_)));
    }

    #[test]
    fn paginate_plan_keyset_wraps_in_filter_and_topn() {
        let plan = paginate_plan_keyset(
            LogicalPlan::Empty,
            Some(&CursorKeyset::new(42, "01J5XY7K")),
            25,
            "a",
        );
        let LogicalPlan::TopN {
            input,
            keys,
            skip,
            limit,
        } = plan
        else {
            panic!("expected TopN at the root");
        };
        assert_eq!(skip, 0);
        assert_eq!(limit, 25);
        assert_eq!(keys.len(), 1);
        assert!(matches!(keys[0].direction, OrderDirection::Asc));
        // Inside the TopN we expect Filter(<a._id > "01J5XY7K"> over input).
        match input.as_ref() {
            LogicalPlan::Filter { .. } => {}
            other => panic!("expected Filter under TopN, got {:?}", other),
        }
    }

    #[test]
    fn paginate_plan_keyset_without_cursor_only_adds_order_and_limit() {
        let plan = paginate_plan_keyset(LogicalPlan::Empty, None, 50, "n");
        let LogicalPlan::TopN { input, limit, .. } = plan else {
            panic!("expected TopN at the root");
        };
        assert_eq!(limit, 50);
        // No filter when there is no cursor — input should be the
        // original Empty.
        assert!(matches!(input.as_ref(), LogicalPlan::Empty));
    }

    #[test]
    fn next_cursor_keyset_returns_none_on_short_page() {
        let next = next_cursor_keyset(123, 5, 25, Some("01ABC"));
        assert!(
            next.is_none(),
            "short page must signal end of stream, got {:?}",
            next
        );
    }

    #[test]
    fn next_cursor_keyset_returns_some_on_full_page() {
        let next =
            next_cursor_keyset(123, 25, 25, Some("01ABC")).expect("full page must produce cursor");
        assert_eq!(next.plan_hash, 123);
        assert_eq!(next.last_id, "01ABC");
    }
}
