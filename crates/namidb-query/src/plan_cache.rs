//! Helpers for caching `LogicalPlan`s across queries.
//!
//! The cloud worker wants to skip parse + lower + optimize when the
//! same Cypher text shows up again. This module owns the two pieces
//! the cache strategy needs to be uniform:
//!
//! * [`query_text_hash`] produces a stable 64-bit fingerprint of a
//!   query string. Whitespace is normalised (leading/trailing trimmed,
//!   internal runs collapsed to single spaces) so cosmetic
//!   reformatting still hits the cache. The hasher is `xxh3-64`, fast
//!   and stable across runs of the same engine version.
//! * [`parse_lower_optimize`] is the one-call entry point the cache
//!   wraps. It parses, lowers, and runs the optimiser; on success the
//!   caller can store the returned `LogicalPlan` against the hash.
//!
//! ## What this does NOT do (yet)
//!
//! `LogicalPlan` is currently not `Serialize` / `Deserialize`. An
//! in-process cache that owns the `LogicalPlan` outright works fine
//! today (and that is the common case — one worker process per
//! namespace). Cross-process caches (Redis, R2-backed key/value) need
//! the plan IR to grow serde derives, which is a larger piece of work
//! tracked separately. The hash returned here is still useful as the
//! cache key because the IR can be regenerated cheaply on a miss.
//!
//! Bumping the engine's parser / lowerer / optimiser changes the
//! plan shape for the same input. Callers must include the engine's
//! own version in the cache key (e.g. `format!("{engine_version}:{hash}")`)
//! to avoid serving a plan from an older optimiser.

use xxhash_rust::xxh3::xxh3_64;

use crate::cost::StatsCatalog;
use crate::optimize::optimize;
use crate::parser::{parse, ParseError};
use crate::plan::{lower, LogicalPlan, LowerError};

/// Stable 64-bit fingerprint of a Cypher query string.
///
/// Two query texts that differ only in surrounding or interior
/// whitespace hash to the same value. Everything else (comments,
/// keyword casing, parameter ordering) is intentionally *not*
/// normalised — those can change the parsed query and must miss the
/// cache.
pub fn query_text_hash(query: &str) -> u64 {
    let normalised = normalise_query_whitespace(query);
    xxh3_64(normalised.as_bytes())
}

fn normalise_query_whitespace(query: &str) -> String {
    let mut out = String::with_capacity(query.len());
    let mut last_was_space = true; // suppress leading whitespace
    for ch in query.chars() {
        if ch.is_whitespace() {
            if !last_was_space {
                out.push(' ');
                last_was_space = true;
            }
        } else {
            out.push(ch);
            last_was_space = false;
        }
    }
    if out.ends_with(' ') {
        out.pop();
    }
    out
}

/// Errors surfaced by [`parse_lower_optimize`].
#[derive(Debug, thiserror::Error)]
pub enum PlanError {
    #[error("parse failed: {} error(s)", .0.len())]
    Parse(Vec<ParseError>),
    #[error("lowering failed: {0}")]
    Lower(#[from] LowerError),
}

impl PlanError {
    /// First parse error, if any. Convenience for callers that want
    /// to surface a single string to users.
    pub fn first_parse_error(&self) -> Option<&ParseError> {
        match self {
            PlanError::Parse(errs) => errs.first(),
            PlanError::Lower(_) => None,
        }
    }
}

impl From<Vec<ParseError>> for PlanError {
    fn from(errs: Vec<ParseError>) -> Self {
        PlanError::Parse(errs)
    }
}

/// Parse, lower, and optimise `query`. Returns the optimised plan
/// that the executor would consume.
///
/// Cache layout the caller is expected to wire up:
///
/// ```ignore
/// let key = format!("{ENGINE_VERSION}:{}", query_text_hash(text));
/// if let Some(plan) = cache.get(&key) { return execute(plan, ...); }
/// let plan = parse_lower_optimize(text, &catalog)?;
/// cache.insert(key, plan.clone());
/// execute(&plan, ...);
/// ```
pub fn parse_lower_optimize(query: &str, catalog: &StatsCatalog) -> Result<LogicalPlan, PlanError> {
    let parsed = parse(query)?;
    let lowered = lower(&parsed)?;
    Ok(optimize(lowered, catalog))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hash_is_stable_for_identical_input() {
        let q = "MATCH (a:Person) RETURN a";
        assert_eq!(query_text_hash(q), query_text_hash(q));
    }

    #[test]
    fn hash_ignores_surrounding_whitespace() {
        let bare = "MATCH (a:Person) RETURN a";
        let padded = "  MATCH (a:Person) RETURN a   ";
        assert_eq!(query_text_hash(bare), query_text_hash(padded));
    }

    #[test]
    fn hash_collapses_internal_whitespace_runs() {
        let one_space = "MATCH (a:Person) RETURN a";
        let many_spaces = "MATCH   (a:Person)\tRETURN\na";
        assert_eq!(query_text_hash(one_space), query_text_hash(many_spaces));
    }

    #[test]
    fn hash_distinguishes_different_queries() {
        let a = "MATCH (a:Person) RETURN a";
        let b = "MATCH (a:Person) RETURN a.name";
        assert_ne!(query_text_hash(a), query_text_hash(b));
    }

    #[test]
    fn hash_is_case_sensitive() {
        // Keyword casing matters: `MATCH` and `match` parse identically
        // but cache hits should be conservative — let the user opt
        // into case-insensitive matching by normalising upstream.
        let upper = "MATCH (a) RETURN a";
        let lower = "match (a) return a";
        assert_ne!(query_text_hash(upper), query_text_hash(lower));
    }

    #[test]
    fn parse_lower_optimize_returns_a_plan() {
        let plan = parse_lower_optimize("MATCH (a:Person) RETURN a", &StatsCatalog::empty())
            .expect("a simple match must plan");
        // Sanity: the resulting plan is non-empty (project over a
        // scan, at minimum).
        assert!(
            !matches!(plan, LogicalPlan::Empty),
            "plan must not be Empty for a valid query"
        );
    }

    #[test]
    fn parse_lower_optimize_propagates_parse_errors() {
        let err = parse_lower_optimize("THIS IS NOT CYPHER", &StatsCatalog::empty())
            .expect_err("garbage must not plan");
        assert!(matches!(err, PlanError::Parse(_)));
    }

    #[test]
    fn logical_plan_round_trips_through_serde_json() {
        // The new `Serialize` / `Deserialize` derives on LogicalPlan
        // (and every type it transitively contains) let the cloud
        // worker stash a plan in a cross-process cache (Redis, R2,
        // Supabase) and recover it bit-for-bit on a hit. This test
        // pins that contract.
        let plan = parse_lower_optimize(
            "MATCH (a:Person) WHERE a.age >= 18 RETURN a.name AS name ORDER BY name LIMIT 10",
            &StatsCatalog::empty(),
        )
        .expect("a representative plan");
        let json = serde_json::to_string(&plan).expect("plan serializes");
        let back: LogicalPlan = serde_json::from_str(&json).expect("plan deserializes");
        assert_eq!(plan, back);
    }

    #[test]
    fn logical_plan_round_trips_with_create_clause() {
        // CREATE plans carry CreateElement / SetOp / RemoveOp; the
        // round-trip must cover those variants too, not just the
        // read-only operators above.
        let plan = parse_lower_optimize(
            "CREATE (a:Person {name: 'Ada'}) RETURN a",
            &StatsCatalog::empty(),
        )
        .expect("a create plan");
        let json = serde_json::to_string(&plan).expect("create plan serializes");
        let back: LogicalPlan = serde_json::from_str(&json).expect("create plan deserializes");
        assert_eq!(plan, back);
    }
}
