//! BM25 lexical relevance scoring (hybrid search, Item 13 Layer B).
//!
//! A `bm25(document, query)` scalar builtin scores how well a document text
//! matches a query, giving the hybrid-search lexical channel a real relevance
//! signal instead of alphabetical order.
//!
//! **Scope — single-document BM25.** Classic BM25 needs corpus statistics
//! (per-term document frequency for IDF, and the corpus average document
//! length). NamiDB's `StatsCatalog` carries no text statistics today, so this
//! is the pragmatic subset that needs none:
//!
//! - **TF saturation** — Σ over query terms of `tf·(k1+1) / (tf + k1)`, the
//!   BM25 term-frequency saturation with `k1 = 1.5`. Repeated matches help
//!   with diminishing returns, exactly as full BM25.
//! - **Field-length normalization** — divided by
//!   `1 - b + b·(len/avg_len)` with `b = 0.75` and a fixed reference
//!   `avg_len`, so a term in a short document outweighs the same term in a
//!   long one.
//! - **IDF omitted** (treated as 1.0) — without corpus document-frequency we
//!   cannot weight rare terms above common ones. This is the one place this
//!   subset diverges from full BM25; it is a deliberate, documented tradeoff
//!   (a future inverted index + term stats would supply real IDF).
//!
//! Tokenization is ASCII-lowercase, split on non-alphanumeric runs (no
//! stemming, no stopwords in v0). Scoring is deterministic and
//! corpus-independent, so it composes cleanly with the RRF fusion that ranks
//! the two channels.

/// BM25 `k1` — term-frequency saturation point.
const K1: f64 = 1.5;
/// BM25 `b` — field-length normalization strength.
const B: f64 = 0.75;
/// Reference average document length (in tokens) used for length
/// normalization, since the true corpus average is not available without text
/// statistics. A document of this length gets a neutral length factor.
const AVG_LEN: f64 = 120.0;

/// Tokenize text into lowercased alphanumeric terms.
fn tokenize(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|s| !s.is_empty())
        .map(|s| s.to_ascii_lowercase())
        .collect()
}

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
        assert!(two > one, "two matched terms ({two}) should beat one ({one})");
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
}
