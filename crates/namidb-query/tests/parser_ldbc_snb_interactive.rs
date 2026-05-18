//! Integration tests against the LDBC SNB Interactive Complex queries.
//!
//! Each `tests/fixtures/ic*.cypher` is parsed and checked according to
//! RFC-004:
//!
//! - **IC1–IC12** must parse successfully and round-trip
//! (`parse → display → parse` produces an identical canonical string).
//! - **IC13–IC14** are out-of-scope v0 (they need `shortestPath` /
//! `allShortestPaths`, whose canonical Cypher forms use unbounded
//! `*` patterns) and must fail with `ErrorCode::UnboundedVariableLength`.

use std::fs;
use std::path::PathBuf;

use namidb_query::parser::{parse, ErrorCode};

fn fixture(name: &str) -> String {
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.push("tests");
    path.push("fixtures");
    path.push(name);
    fs::read_to_string(&path).unwrap_or_else(|e| panic!("failed to read {:?}: {}", path, e))
}

/// Parse the fixture, then re-parse its canonical Display form and ensure the
/// two display strings are identical.
fn parse_and_roundtrip(name: &str) {
    let src = fixture(name);
    let parsed = parse(&src).unwrap_or_else(|errs| {
        panic!(
            "fixture `{}` failed to parse:\n--- src ---\n{}\n--- errors ---\n{:#?}",
            name, src, errs
        )
    });
    let formatted = parsed.to_string();
    let reparsed = parse(&formatted).unwrap_or_else(|errs| {
        panic!(
            "fixture `{}` re-parse failed:\n--- formatted ---\n{}\n--- errors ---\n{:#?}",
            name, formatted, errs
        )
    });
    let reformatted = reparsed.to_string();
    assert_eq!(formatted, reformatted, "round-trip mismatch for `{}`", name);
}

// ────────────────── IN-SCOPE queries (must parse) ──────────────────

#[test]
fn ic01_friends_by_name() {
    parse_and_roundtrip("ic01_friends_by_name.cypher");
}

#[test]
fn ic02_recent_messages_by_friends() {
    parse_and_roundtrip("ic02_recent_messages_by_friends.cypher");
}

#[test]
fn ic03_friends_in_two_countries() {
    parse_and_roundtrip("ic03_friends_in_two_countries.cypher");
}

#[test]
fn ic04_new_topics() {
    parse_and_roundtrip("ic04_new_topics.cypher");
}

#[test]
fn ic05_new_groups() {
    parse_and_roundtrip("ic05_new_groups.cypher");
}

#[test]
fn ic06_tag_cooccurrence() {
    parse_and_roundtrip("ic06_tag_cooccurrence.cypher");
}

#[test]
fn ic07_recent_likers() {
    parse_and_roundtrip("ic07_recent_likers.cypher");
}

#[test]
fn ic08_recent_replies() {
    parse_and_roundtrip("ic08_recent_replies.cypher");
}

#[test]
fn ic09_friends_of_friends_messages() {
    parse_and_roundtrip("ic09_friends_of_friends_messages.cypher");
}

#[test]
fn ic10_friend_recommendation() {
    parse_and_roundtrip("ic10_friend_recommendation.cypher");
}

#[test]
fn ic11_job_referral() {
    parse_and_roundtrip("ic11_job_referral.cypher");
}

#[test]
fn ic12_expert_search() {
    parse_and_roundtrip("ic12_expert_search.cypher");
}

// ────────────────── OUT-OF-SCOPE queries (must reject) ─────────────

#[test]
fn ic13_shortest_path_is_out_of_scope_v0() {
    let src = fixture("ic13_shortest_path.cypher");
    let errs = parse(&src).expect_err("IC13 must fail in v0");
    assert!(
        errs.iter().any(|e| e.code == ErrorCode::UnsupportedFeature),
        "expected UnsupportedFeature, got {:?}",
        errs
    );
}

#[test]
fn ic14_all_shortest_paths_is_out_of_scope_v0() {
    let src = fixture("ic14_all_shortest_paths.cypher");
    let errs = parse(&src).expect_err("IC14 must fail in v0");
    assert!(
        errs.iter().any(|e| e.code == ErrorCode::UnsupportedFeature),
        "expected UnsupportedFeature, got {:?}",
        errs
    );
}

// ────────────────── meta-check: every fixture is exercised ─────────

#[test]
fn every_fixture_has_a_test() {
    let mut dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    dir.push("tests");
    dir.push("fixtures");
    let mut names: Vec<String> = fs::read_dir(&dir)
        .expect("fixtures dir")
        .filter_map(|e| {
            let e = e.ok()?;
            let name = e.file_name().to_string_lossy().to_string();
            if name.ends_with(".cypher") {
                Some(name)
            } else {
                None
            }
        })
        .collect();
    names.sort();
    let expected: Vec<&str> = vec![
        "ic01_friends_by_name.cypher",
        "ic02_recent_messages_by_friends.cypher",
        "ic03_friends_in_two_countries.cypher",
        "ic04_new_topics.cypher",
        "ic05_new_groups.cypher",
        "ic06_tag_cooccurrence.cypher",
        "ic07_recent_likers.cypher",
        "ic08_recent_replies.cypher",
        "ic09_friends_of_friends_messages.cypher",
        "ic10_friend_recommendation.cypher",
        "ic11_job_referral.cypher",
        "ic12_expert_search.cypher",
        "ic13_shortest_path.cypher",
        "ic14_all_shortest_paths.cypher",
        // Update queries (executor coverage lives in
        // `tests/exec_ldbc_snb_updates.rs`; this suite just tracks the
        // existence of the fixture files).
        "iu01_insert_person.cypher",
        "iu02_add_post_like.cypher",
        "iu06_add_post.cypher",
        "iu08_add_friendship.cypher",
    ];
    assert_eq!(names, expected, "fixtures drift — add or remove test");
}
