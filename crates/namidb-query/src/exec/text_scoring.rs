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
//!    text property, builds corpus statistics ([`Bm25Stats`] + per-term
//!    document frequency) in one pass, then scores every candidate document
//!    with [`bm25_term_score`]. This is "Layer C": the IDF the scalar omits.
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

/// BM25 `k1` — term-frequency saturation point.
const K1: f64 = 1.5;
/// BM25 `b` — field-length normalization strength.
const B: f64 = 0.75;
/// Reference average document length (in tokens) for the corpus-free scalar
/// builtin, where the true corpus average is unavailable. A document of this
/// length gets a neutral length factor. The `CALL search.bm25` procedure uses
/// the real corpus average instead.
const AVG_LEN: f64 = 120.0;

/// Tokenize text into lowercased alphanumeric terms.
pub(crate) fn tokenize(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|s| !s.is_empty())
        .map(|s| s.to_ascii_lowercase())
        .collect()
}

/// Tokenize `text` into a term→count map plus the total token count (document
/// length). The basis for corpus-aware BM25 over a scanned document.
pub fn tokenize_counts(text: &str) -> (std::collections::HashMap<String, u32>, usize) {
    let terms = tokenize(text);
    let len = terms.len();
    let mut counts: std::collections::HashMap<String, u32> = std::collections::HashMap::new();
    for t in terms {
        *counts.entry(t).or_insert(0) += 1;
    }
    (counts, len)
}

/// Corpus-level statistics needed for true BM25.
#[derive(Debug, Clone, Copy)]
pub struct Bm25Stats {
    /// Number of documents in the corpus (the label's docs with the text field).
    pub n_docs: usize,
    /// Average document length in tokens. Falls back to a neutral 1.0 when the
    /// corpus is empty so the length factor never divides by zero.
    pub avg_len: f64,
}

/// BM25 inverse document frequency, Lucene form:
/// `ln(1 + (N - df + 0.5) / (df + 0.5))`.
///
/// The `+1` keeps it non-negative even for a term present in most documents
/// (classic Robertson–Spärck Jones IDF goes negative past N/2). A rarer term
/// (smaller `df`) gets a larger weight. `df = 0` yields the maximum weight but
/// callers only score terms that occur, so it is harmless.
pub fn bm25_idf(n_docs: usize, doc_freq: usize) -> f64 {
    let n = n_docs as f64;
    let df = doc_freq as f64;
    (1.0 + (n - df + 0.5) / (df + 0.5)).ln()
}

/// One query term's BM25 contribution to a document: the saturated,
/// length-normalized term score scaled by its IDF weight. `tf == 0` → 0.
pub fn bm25_term_score(idf: f64, tf: u32, doc_len: usize, avg_len: f64) -> f64 {
    if tf == 0 {
        return 0.0;
    }
    let tf = tf as f64;
    let norm = 1.0 - B + B * (doc_len as f64 / avg_len.max(1.0));
    idf * (tf * (K1 + 1.0)) / (tf + K1 * norm)
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

    // --- corpus-aware BM25 (real IDF) ---------------------------------------

    #[test]
    fn idf_rewards_rarer_terms_and_stays_non_negative() {
        // In a 100-doc corpus, a term in 1 doc must outweigh one in 50.
        let rare = bm25_idf(100, 1);
        let common = bm25_idf(100, 50);
        assert!(rare > common, "rare ({rare}) should outweigh common ({common})");
        // Even a term present in every document is non-negative (Lucene form).
        assert!(bm25_idf(100, 100) >= 0.0);
        assert!(common > 0.0);
    }

    #[test]
    fn term_score_zero_when_absent() {
        assert_eq!(bm25_term_score(2.0, 0, 50, 120.0), 0.0);
    }

    #[test]
    fn term_score_scales_with_idf() {
        // Same tf/length, higher idf → higher contribution.
        let low = bm25_term_score(0.5, 3, 100, 120.0);
        let high = bm25_term_score(2.0, 3, 100, 120.0);
        assert!(high > low);
        // Proportional to idf for fixed tf/len.
        assert!((high / low - 4.0).abs() < 1e-9);
    }

    #[test]
    fn term_score_saturates_and_normalizes_length() {
        // Saturation: 1→2 helps more than 9→10.
        let g_low = bm25_term_score(1.0, 2, 50, 50.0) - bm25_term_score(1.0, 1, 50, 50.0);
        let g_high = bm25_term_score(1.0, 10, 50, 50.0) - bm25_term_score(1.0, 9, 50, 50.0);
        assert!(g_low > g_high);
        // Length norm: same tf, shorter doc scores higher.
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
}
