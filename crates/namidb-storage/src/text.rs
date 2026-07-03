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
//! the same corpus. The same goes for the query **syntax**: [`parse_query`]
//! turns a raw query string into a [`TextQuery`] — quoted phrases (adjacency
//! constraints), trailing-`*` prefixes (bounded vocabulary expansion), plain
//! bag-of-words terms — that both consumers interpret with the same rules.
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

use std::collections::{BTreeSet, HashMap};

/// BM25 `k1` — term-frequency saturation point.
pub const K1: f64 = 1.5;
/// BM25 `b` — field-length normalization strength.
pub const B: f64 = 0.75;

/// Cap on how many vocabulary terms one `foo*` prefix pattern may expand to:
/// the lexicographically-first N matching terms, picked identically on the
/// index path (postings `BTreeMap` range) and the flat scan (sorted corpus
/// vocabulary), so both paths score the same expansion. Bounded so a short
/// prefix over a large vocabulary cannot blow one query up into thousands of
/// scored terms.
pub const PREFIX_EXPANSION_LIMIT: usize = 64;

/// A parsed full-text query: quoted phrases (position-adjacency required),
/// trailing-`*` prefixes (expanded over the corpus vocabulary), and plain
/// bag-of-words terms. Produced by [`parse_query`], consumed by both the
/// persistent index and the flat-scan fallback so their semantics agree.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct TextQuery {
    /// Distinct plain terms, sorted — ordinary bag-of-words BM25.
    pub terms: Vec<String>,
    /// Quoted phrases as token sequences. A document must contain every
    /// phrase's tokens at adjacent token positions (a hard constraint on
    /// candidacy); a passing document then scores the phrase's tokens as
    /// ordinary BM25 terms — adjacency gates candidacy, it does not change
    /// the scoring formula.
    pub phrases: Vec<Vec<String>>,
    /// Distinct lowercased prefixes from trailing-`*` patterns, sorted. Each
    /// expands to at most [`PREFIX_EXPANSION_LIMIT`] vocabulary terms scored
    /// normally.
    pub prefixes: Vec<String>,
}

impl TextQuery {
    /// A plain bag-of-words query over pre-tokenized terms (no phrases or
    /// prefixes; duplicates collapse) — the historical search entry point.
    pub fn from_terms(terms: &[String]) -> Self {
        let set: BTreeSet<String> = terms.iter().cloned().collect();
        Self {
            terms: set.into_iter().collect(),
            ..Default::default()
        }
    }

    /// `true` when nothing was parsed — there is nothing to search for.
    pub fn is_empty(&self) -> bool {
        self.terms.is_empty() && self.phrases.is_empty() && self.prefixes.is_empty()
    }

    /// Distinct sorted scored terms before prefix expansion: the plain terms
    /// plus every phrase token. Sorted so both consumers sum per-term score
    /// contributions in the same order (bit-identical floats).
    pub fn base_terms(&self) -> BTreeSet<&str> {
        let mut out: BTreeSet<&str> = self.terms.iter().map(String::as_str).collect();
        for phrase in &self.phrases {
            for t in phrase {
                out.insert(t);
            }
        }
        out
    }
}

/// Parse a raw query string into [`TextQuery`] syntax:
///
/// - `"..."` — a quoted span is a **phrase**: its tokens (via [`tokenize`],
///   so a CJK span becomes adjacent bigrams) must appear at adjacent token
///   positions in a document. A single-token phrase degrades to a required
///   containment constraint. `*` inside quotes is ordinary punctuation. An
///   unclosed quote runs to the end of the string.
/// - `foo*` — a `*` immediately after an alphanumeric run marks the run's
///   **last** emitted token as a prefix pattern (for a Latin word, the word
///   itself; earlier tokens of the run stay plain terms). A `*` not preceded
///   by an alphanumeric character is ignored.
/// - everything else — plain bag-of-words terms.
///
/// A query with no `"` and no `*` parses to plain terms only, keeping the
/// historical bag-of-words behaviour untouched.
pub fn parse_query(query: &str) -> TextQuery {
    let mut terms: BTreeSet<String> = BTreeSet::new();
    let mut phrases: Vec<Vec<String>> = Vec::new();
    let mut prefixes: BTreeSet<String> = BTreeSet::new();

    let mut in_quotes = false;
    for span in query.split('"') {
        if in_quotes {
            let tokens = tokenize(span);
            if !tokens.is_empty() {
                phrases.push(tokens);
            }
        } else {
            parse_unquoted_span(span, &mut terms, &mut prefixes);
        }
        in_quotes = !in_quotes;
    }

    TextQuery {
        terms: terms.into_iter().collect(),
        phrases,
        prefixes: prefixes.into_iter().collect(),
    }
}

/// Split one unquoted span into plain terms and trailing-`*` prefixes.
fn parse_unquoted_span(span: &str, terms: &mut BTreeSet<String>, prefixes: &mut BTreeSet<String>) {
    let mut flush = |run: &mut String, starred: bool| {
        if run.is_empty() {
            return;
        }
        let mut tokens = Vec::new();
        emit_segment_tokens(run, &mut tokens);
        run.clear();
        if starred {
            if let Some(last) = tokens.pop() {
                prefixes.insert(last);
            }
        }
        terms.extend(tokens);
    };
    let mut run = String::new();
    for c in span.chars() {
        if c.is_alphanumeric() {
            run.push(c);
        } else {
            flush(&mut run, c == '*');
        }
    }
    flush(&mut run, false);
}

/// `true` when `phrase` (a non-empty token sequence) occurs at adjacent token
/// positions in `tokens` — the flat-scan side of the phrase constraint. The
/// persistent index answers the same question from stored posting positions;
/// both operate on the offsets of [`tokenize`]-emitted tokens (CJK bigrams
/// included), so they agree exactly.
pub fn contains_phrase(tokens: &[String], phrase: &[String]) -> bool {
    if phrase.is_empty() {
        return true;
    }
    if tokens.len() < phrase.len() {
        return false;
    }
    tokens.windows(phrase.len()).any(|w| w == phrase)
}

/// `true` for scripts written without spaces between words (CJK ideographs,
/// kana, Hangul), where whitespace/punctuation splitting yields one giant token.
/// Such runs are indexed as overlapping bigrams instead (the Lucene CJKAnalyzer
/// approach: dictionary-free, and symmetric between index and query).
fn is_cjk(c: char) -> bool {
    matches!(c as u32,
        0x3040..=0x30FF   // Hiragana + Katakana
        | 0x3400..=0x4DBF // CJK Unified Ideographs Extension A
        | 0x4E00..=0x9FFF // CJK Unified Ideographs
        | 0xF900..=0xFAFF // CJK Compatibility Ideographs
        | 0xAC00..=0xD7AF // Hangul syllables
        | 0x20000..=0x2A6DF // CJK Unified Ideographs Extension B
    )
}

/// Tokenize text into lowercased terms (split on non-alphanumeric runs; no
/// stemming, no stopwords).
///
/// Case folding is Unicode-aware (`to_lowercase`), so `CAFÉ`, `ÜBER`, `ПРИВЕТ`
/// fold to `café`, `über`, `привет` and match their lowercase forms. Using
/// `to_ascii_lowercase` left every non-ASCII capital un-folded, so
/// case-insensitive full-text search silently failed for all non-English text.
///
/// CJK/Hangul runs (which carry no word separators) are emitted as overlapping
/// bigrams — `東京大学` → `東京`, `京大`, `大学` — so a query like `東京` matches;
/// otherwise the whole run became one token that no realistic query ever typed.
/// A length-1 CJK run is emitted as a unigram.
///
/// The index and the flat `bm25` scalar both call this, so they stay in exact
/// agreement (an index built by an older binary reindexes on the next
/// authoritative compaction).
pub fn tokenize(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    for segment in text.split(|c: char| !c.is_alphanumeric()) {
        if segment.is_empty() {
            continue;
        }
        emit_segment_tokens(segment, &mut out);
    }
    out
}

/// Emit tokens for one maximal alphanumeric segment: non-CJK subruns become one
/// lowercased word each; CJK subruns become overlapping lowercased bigrams.
fn emit_segment_tokens(segment: &str, out: &mut Vec<String>) {
    let chars: Vec<char> = segment.chars().collect();
    let mut i = 0;
    while i < chars.len() {
        if is_cjk(chars[i]) {
            // Consume the maximal CJK run and emit overlapping bigrams.
            let start = i;
            while i < chars.len() && is_cjk(chars[i]) {
                i += 1;
            }
            let run = &chars[start..i];
            if run.len() == 1 {
                out.push(run[0].to_lowercase().to_string());
            } else {
                for w in run.windows(2) {
                    out.push(w.iter().collect::<String>().to_lowercase());
                }
            }
        } else {
            // Consume the maximal non-CJK run as one word.
            let start = i;
            while i < chars.len() && !is_cjk(chars[i]) {
                i += 1;
            }
            out.push(chars[start..i].iter().collect::<String>().to_lowercase());
        }
    }
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
    fn tokenize_case_folds_non_ascii_letters() {
        // Non-ASCII capitals must fold so case-insensitive search works for
        // non-English text (the ASCII-only fold left É/Ü/П un-folded).
        assert_eq!(tokenize("CAFÉ"), vec!["café"]);
        assert_eq!(tokenize("ÜBER Über über"), vec!["über", "über", "über"]);
        assert_eq!(tokenize("ПРИВЕТ Привет"), vec!["привет", "привет"]);
    }

    #[test]
    fn tokenize_cjk_runs_into_overlapping_bigrams() {
        // A CJK run must become overlapping bigrams so a shorter query matches.
        assert_eq!(tokenize("東京大学"), vec!["東京", "京大", "大学"]);
        // A query token is tokenized the same way, so `東京` is findable.
        assert_eq!(tokenize("東京"), vec!["東京"]);
        // A length-1 run stays a unigram.
        assert_eq!(tokenize("犬"), vec!["犬"]);
        // Mixed Latin + CJK in one segment: Latin word + CJK bigrams.
        assert_eq!(tokenize("iPhone東京"), vec!["iphone", "東京"]);
        // Latin words are unaffected.
        assert_eq!(tokenize("hello world"), vec!["hello", "world"]);
    }

    #[test]
    fn avg_len_guards_empty_corpus() {
        assert_eq!(avg_len(0, 0), 1.0);
        assert_eq!(avg_len(300, 3), 100.0);
    }

    fn strs(items: &[&str]) -> Vec<String> {
        items.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn parse_query_plain_matches_tokenize() {
        // No quotes/asterisks → distinct sorted terms, nothing else: the
        // historical bag-of-words path.
        let q = parse_query("Fox common, fox!");
        assert_eq!(q.terms, strs(&["common", "fox"]));
        assert!(q.phrases.is_empty());
        assert!(q.prefixes.is_empty());
        assert!(!q.is_empty());
    }

    #[test]
    fn parse_query_extracts_quoted_phrases() {
        let q = parse_query("\"Graph Database\" fast");
        assert_eq!(q.phrases, vec![strs(&["graph", "database"])]);
        assert_eq!(q.terms, strs(&["fast"]));

        // Unclosed quote runs to the end; empty quotes are ignored.
        let q = parse_query("\"graph database");
        assert_eq!(q.phrases, vec![strs(&["graph", "database"])]);
        assert!(parse_query("\"\"").is_empty());

        // `*` inside quotes is ordinary punctuation, not prefix syntax.
        let q = parse_query("\"data* base\"");
        assert_eq!(q.phrases, vec![strs(&["data", "base"])]);
        assert!(q.prefixes.is_empty());
    }

    #[test]
    fn parse_query_extracts_trailing_star_prefixes() {
        let q = parse_query("data* base");
        assert_eq!(q.prefixes, strs(&["data"]));
        assert_eq!(q.terms, strs(&["base"]));

        // The `*` binds to the last token of the run before it; a bare `*`
        // (no preceding alphanumeric) is ignored.
        let q = parse_query("e-mail* *");
        assert_eq!(q.prefixes, strs(&["mail"]));
        assert_eq!(q.terms, strs(&["e"]));
        assert!(parse_query("*").is_empty());

        // A CJK run before `*`: the last emitted bigram is the prefix.
        let q = parse_query("東京大*");
        assert_eq!(q.prefixes, strs(&["京大"]));
        assert_eq!(q.terms, strs(&["東京"]));
    }

    #[test]
    fn contains_phrase_requires_adjacency() {
        let doc = tokenize("a database of graph paper");
        assert!(!contains_phrase(&doc, &strs(&["graph", "database"])));
        let doc = tokenize("graph database systems");
        assert!(contains_phrase(&doc, &strs(&["graph", "database"])));
        // Single-token phrase = containment; longer-than-doc phrase never hits.
        assert!(contains_phrase(&doc, &strs(&["systems"])));
        assert!(!contains_phrase(
            &tokenize("graph"),
            &strs(&["graph", "database"])
        ));
        // CJK: positions are per emitted bigram, so `東京大` (→ 東京, 京大)
        // matches the contiguous run but not the particle-split one.
        let phrase = tokenize("東京大");
        assert!(contains_phrase(&tokenize("東京大学"), &phrase));
        assert!(!contains_phrase(&tokenize("東京の大学"), &phrase));
    }
}
