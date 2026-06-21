//! BM25 lexical relevance scoring (hybrid search, Item 13).
//!
//! Two surfaces share the same tokenizer and TF/length-normalization math:
//!
//! 1. **`bm25(document, query)` scalar builtin** — a per-row, corpus-free
//!    score (IDF treated as 1.0). It needs no global state, so it composes
//!    anywhere in a projection, but it cannot weight rare terms above common
//!    ones. See [`bm25_score`].
//! 2. **`CALL search.bm25(...)` procedure** — full BM25 with **real IDF** and a
//!    corpus-derived average document length. The procedure scans a label's
//!    text property, builds corpus statistics (document count, average length,
//!    per-term document frequency) in one pass, then scores every candidate
//!    document with [`bm25_term_score`]. A registered full-text index
//!    (`text-index`) serves the same scores without the scan. This is the IDF
//!    the scalar omits.
//!
//! **The BM25 model.** For query term `t` in document `d`:
//!
//! ```text
//! score(d, t) = idf(t) · tf(t,d)·(k1+1) / (tf(t,d) + k1·(1 - b + b·|d|/avgdl))
//! idf(t)      = ln(1 + (N - df(t) + 0.5) / (df(t) + 0.5))
//! ```
//!
//! with `k1 = 1.5`, `b = 0.75`, `N` the document count, `df(t)` the number of
//! documents containing `t`, and `avgdl` the average document length. The `+1`
//! inside the IDF log is the Lucene form that keeps IDF non-negative (a term in
//! more than half the corpus never goes negative). The scalar builtin uses the
//! same expression with `idf = 1` and a fixed reference `avgdl`.
//!
//! Tokenization is ASCII-lowercase, split on non-alphanumeric runs (no
//! stemming, no stopwords in v0).
//!
//! The shared, corpus-aware primitives ([`tokenize_counts`], [`bm25_idf`],
//! [`bm25_term_score`]) live in [`namidb_storage::text`] so the persistent
//! full-text index and this query-time path score identically; they are
//! re-exported here for the procedure and the scalar below.

pub use namidb_storage::text::{bm25_idf, bm25_term_score, tokenize, tokenize_counts, B, K1};

/// Reference average document length (in tokens) for the corpus-free scalar
/// builtin, where the true corpus average is unavailable. A document of this
/// length gets a neutral length factor. The `CALL search.bm25` procedure uses
/// the real corpus average instead.
const AVG_LEN: f64 = 120.0;

/// BM25 relevance of `document` for `query` (see the module docs). Returns
/// `0.0` when the query has no terms or none of them occur in the document.
pub fn bm25_score(document: &str, query: &str) -> f64 {
    let doc_terms = tokenize(document);
    if doc_terms.is_empty() {
        return 0.0;
    }
    let query_terms = tokenize(query);
    if query_terms.is_empty() {
        return 0.0;
    }
    let doc_len = doc_terms.len() as f64;
    // Length normalization denominator (constant across query terms).
    let norm = 1.0 - B + B * (doc_len / AVG_LEN);

    // Count occurrences of each document term once.
    let mut counts: std::collections::HashMap<&str, u32> = std::collections::HashMap::new();
    for t in &doc_terms {
        *counts.entry(t.as_str()).or_insert(0) += 1;
    }

    // Score each DISTINCT query term once (a query term repeated doesn't
    // double-count), summing its saturated, length-normalized contribution.
    let mut scored: std::collections::HashSet<&str> = std::collections::HashSet::new();
    let mut score = 0.0;
    for qt in &query_terms {
        if !scored.insert(qt.as_str()) {
            continue;
        }
        let tf = counts.get(qt.as_str()).copied().unwrap_or(0) as f64;
        if tf > 0.0 {
            // IDF treated as 1.0 (no corpus document-frequency available).
            score += (tf * (K1 + 1.0)) / (tf + K1 * norm);
        }
    }
    score
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_match_is_zero() {
        assert_eq!(bm25_score("the quick brown fox", "elephant"), 0.0);
        assert_eq!(bm25_score("", "fox"), 0.0);
        assert_eq!(bm25_score("fox", ""), 0.0);
    }

    #[test]
    fn a_match_scores_positive() {
        assert!(bm25_score("the quick brown fox", "fox") > 0.0);
    }

    #[test]
    fn more_query_terms_matched_scores_higher() {
        let one = bm25_score("the quick brown fox jumps", "fox");
        let two = bm25_score("the quick brown fox jumps", "quick fox");
        assert!(
            two > one,
            "two matched terms ({two}) should beat one ({one})"
        );
    }

    #[test]
    fn term_frequency_saturates() {
        // Going from 1 to 2 occurrences helps more than 9 to 10 (saturation).
        let doc1 = "fox bird bird bird bird bird bird bird bird bird";
        let doc2 = "fox fox bird bird bird bird bird bird bird bird";
        let gain_low = bm25_score(doc2, "fox") - bm25_score(doc1, "fox");
        let doc9 = "fox fox fox fox fox fox fox fox fox bird";
        let doc10 = "fox fox fox fox fox fox fox fox fox fox";
        let gain_high = bm25_score(doc10, "fox") - bm25_score(doc9, "fox");
        assert!(
            gain_low > gain_high,
            "early occurrences should help more (low {gain_low} > high {gain_high})"
        );
    }

    #[test]
    fn shorter_document_scores_higher_for_same_tf() {
        // Same single occurrence; the shorter document is more "about" the term.
        let short = bm25_score("fox", "fox");
        let long = bm25_score(
            "fox lorem ipsum dolor sit amet consectetur adipiscing elit sed do",
            "fox",
        );
        assert!(short > long, "short {short} should beat long {long}");
    }

    #[test]
    fn tokenization_is_case_and_punctuation_insensitive() {
        assert!(bm25_score("The Quick, Brown FOX!", "fox") > 0.0);
        assert!(bm25_score("e-mail server", "email").abs() < f64::EPSILON);
        // hyphen splits into two tokens; "mail" matches.
        assert!(bm25_score("e-mail server", "mail") > 0.0);
    }

    // Corpus-aware primitives (bm25_idf / bm25_term_score / tokenize_counts) are
    // tested in `namidb_storage::text`, their home module.
}
