//! Shared BM25 text-scoring primitives.
//!
//! These pure functions are the common ground between two consumers that must
//! agree exactly on tokenization and the scoring formula:
//!
//! - the `bm25` Cypher scalar and the `CALL search.bm25` procedure in
//!   `namidb-query`, which scan a label's text property at query time; and
//! - the persistent full-text index (`sst::text`, feature `text-index`), which
//!   precomputes postings + corpus statistics during compaction and scores
//!   queries against them.
//!
//! Keeping the math here (rather than duplicated across the storage/query
//! boundary) guarantees the index and the flat scan return identical scores for
//! the same corpus.
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
//! inside the IDF log is the Lucene form that keeps IDF non-negative.

use std::collections::HashMap;

/// BM25 `k1` — term-frequency saturation point.
pub const K1: f64 = 1.5;
/// BM25 `b` — field-length normalization strength.
pub const B: f64 = 0.75;

/// Tokenize text into lowercased alphanumeric terms (split on non-alphanumeric
/// runs; no stemming, no stopwords).
pub fn tokenize(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|s| !s.is_empty())
        .map(|s| s.to_ascii_lowercase())
        .collect()
}

/// Tokenize `text` into a term→count map plus the total token count (document
/// length). The basis for corpus-aware BM25 over a document.
pub fn tokenize_counts(text: &str) -> (HashMap<String, u32>, usize) {
    let terms = tokenize(text);
    let len = terms.len();
    let mut counts: HashMap<String, u32> = HashMap::new();
    for t in terms {
        *counts.entry(t).or_insert(0) += 1;
    }
    (counts, len)
}

/// BM25 inverse document frequency, Lucene form:
/// `ln(1 + (N - df + 0.5) / (df + 0.5))`.
///
/// The `+1` keeps it non-negative even for a term present in most documents
/// (classic Robertson–Spärck Jones IDF goes negative past N/2). A rarer term
/// (smaller `df`) gets a larger weight.
pub fn bm25_idf(n_docs: usize, doc_freq: usize) -> f64 {
    let n = n_docs as f64;
    let df = doc_freq as f64;
    (1.0 + (n - df + 0.5) / (df + 0.5)).ln()
}

/// One query term's BM25 contribution to a document: the saturated,
/// length-normalized term score scaled by its IDF weight. `tf == 0` → 0.
/// `avg_len` is clamped to ≥ 1.0 so the length factor never divides by zero.
pub fn bm25_term_score(idf: f64, tf: u32, doc_len: usize, avg_len: f64) -> f64 {
    if tf == 0 {
        return 0.0;
    }
    let tf = tf as f64;
    let norm = 1.0 - B + B * (doc_len as f64 / avg_len.max(1.0));
    idf * (tf * (K1 + 1.0)) / (tf + K1 * norm)
}

/// Average document length from a corpus total, with the empty-corpus guard.
pub fn avg_len(total_len: u64, n_docs: usize) -> f64 {
    if n_docs > 0 {
        total_len as f64 / n_docs as f64
    } else {
        1.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn idf_rewards_rarer_terms_and_stays_non_negative() {
        let rare = bm25_idf(100, 1);
        let common = bm25_idf(100, 50);
        assert!(rare > common);
        assert!(bm25_idf(100, 100) >= 0.0);
        assert!(common > 0.0);
    }

    #[test]
    fn term_score_zero_when_absent() {
        assert_eq!(bm25_term_score(2.0, 0, 50, 120.0), 0.0);
    }

    #[test]
    fn term_score_scales_with_idf() {
        let low = bm25_term_score(0.5, 3, 100, 120.0);
        let high = bm25_term_score(2.0, 3, 100, 120.0);
        assert!(high > low);
        assert!((high / low - 4.0).abs() < 1e-9);
    }

    #[test]
    fn term_score_saturates_and_normalizes_length() {
        let g_low = bm25_term_score(1.0, 2, 50, 50.0) - bm25_term_score(1.0, 1, 50, 50.0);
        let g_high = bm25_term_score(1.0, 10, 50, 50.0) - bm25_term_score(1.0, 9, 50, 50.0);
        assert!(g_low > g_high);
        let short = bm25_term_score(1.0, 1, 10, 100.0);
        let long = bm25_term_score(1.0, 1, 200, 100.0);
        assert!(short > long);
    }

    #[test]
    fn tokenize_counts_groups_and_measures_length() {
        let (counts, len) = tokenize_counts("Fox fox, FOX! bird");
        assert_eq!(len, 4);
        assert_eq!(counts.get("fox").copied(), Some(3));
        assert_eq!(counts.get("bird").copied(), Some(1));
    }

    #[test]
    fn avg_len_guards_empty_corpus() {
        assert_eq!(avg_len(0, 0), 1.0);
        assert_eq!(avg_len(300, 3), 100.0);
    }
}
