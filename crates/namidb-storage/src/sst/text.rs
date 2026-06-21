//! Full-text inverted-index SST body (`text-index` feature).
//!
//! A `.ft` body is self-contained: it carries an inverted index (term →
//! postings) plus the corpus statistics BM25 needs (document count, per-document
//! length, and — implicitly, as posting-list length — per-term document
//! frequency), so a query answers a top-k by touching only the documents that
//! contain the query terms, never re-scanning the whole label. The format is an
//! 8-byte magic + a bincode-serialised [`TextIndexBody`]. Built during
//! compaction from the merged node rows ([`build_body`]); searched by decoding
//! into a [`TextIndex`] and calling [`TextIndex::search`].
//!
//! The scoring math is shared with the query-time flat scan via
//! [`crate::text`], so the index and the scan return identical BM25 scores for
//! the same corpus. Like the vector index, a `.ft` body reflects the **compacted**
//! corpus as of the last compaction; documents written since are served by the
//! flat-scan fallback, not this index.

use std::collections::{BTreeMap, HashMap, HashSet};

use bytes::Bytes;
use serde::{Deserialize, Serialize};

use crate::error::Error;
use crate::text::{avg_len, bm25_idf, bm25_term_score, tokenize_counts};

/// On-disk magic + format major version (`NAMI` `FT` `01`). Bumped on any
/// incompatible layout change so a reader never silently misparses a file.
const MAGIC: &[u8; 8] = b"NAMIFT01";

/// The body of a `SstKind::TextIndex` SST, bincode-serialised after [`MAGIC`].
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TextIndexBody {
    /// Number of documents indexed.
    pub n_docs: u32,
    /// Sum of all document lengths in tokens (→ average document length).
    pub total_len: u64,
    /// `NodeId` per document index `i` (parallel to `doc_lens`).
    pub doc_ids: Vec<[u8; 16]>,
    /// Token count per document index `i`.
    pub doc_lens: Vec<u32>,
    /// Inverted index: term → postings of `(document index, term frequency)`,
    /// ascending by document index. `postings[t].len()` is `df(t)`.
    pub postings: BTreeMap<String, Vec<(u32, u32)>>,
}

/// Stats harvested at build time, mirrored into
/// [`crate::manifest::KindSpecificStats::TextIndex`].
#[derive(Debug, Clone)]
pub struct TextIndexBuildStats {
    pub doc_count: u64,
    pub term_count: u64,
    pub total_len: u64,
}

/// Build a text-index body from `(NodeId, document text)` pairs. The text is the
/// already-concatenated value of the indexed properties for one document.
/// Returns `Ok(None)` when there are no documents (nothing to index → the caller
/// keeps the flat-scan fallback).
pub fn build_body(
    members: Vec<([u8; 16], String)>,
) -> Result<Option<(Bytes, TextIndexBuildStats)>, Error> {
    if members.is_empty() {
        return Ok(None);
    }
    let mut body = TextIndexBody::default();
    for (id, text) in members {
        let (counts, len) = tokenize_counts(&text);
        // A document with no tokens still counts toward N and average length
        // (it is a document); it simply contributes no postings.
        let di = body.doc_ids.len() as u32;
        body.doc_ids.push(id);
        body.doc_lens.push(len as u32);
        body.n_docs += 1;
        body.total_len += len as u64;
        for (term, tf) in counts {
            body.postings.entry(term).or_default().push((di, tf));
        }
    }
    // We pushed postings in ascending document order, so each list is already
    // sorted by document index — deterministic and ready for scoring.
    let stats = TextIndexBuildStats {
        doc_count: body.n_docs as u64,
        term_count: body.postings.len() as u64,
        total_len: body.total_len,
    };
    let payload = bincode::serialize(&body)
        .map_err(|e| Error::invariant(format!("text index encode failed: {e}")))?;
    let mut bytes = MAGIC.to_vec();
    bytes.extend_from_slice(&payload);
    Ok(Some((Bytes::from(bytes), stats)))
}

/// A decoded, searchable text index.
#[derive(Debug)]
pub struct TextIndex {
    body: TextIndexBody,
}

impl TextIndex {
    /// Decode a `.ft` body (magic + bincode). Errors on a truncated/foreign file
    /// or a magic mismatch.
    pub fn decode(bytes: &[u8]) -> Result<Self, Error> {
        if bytes.len() < MAGIC.len() {
            return Err(Error::invariant("text index body too short for magic"));
        }
        let (magic, rest) = bytes.split_at(MAGIC.len());
        if magic != MAGIC {
            return Err(Error::invariant(format!(
                "text index magic mismatch: {magic:?}"
            )));
        }
        let body: TextIndexBody = bincode::deserialize(rest)
            .map_err(|e| Error::invariant(format!("text index decode failed: {e}")))?;
        Ok(Self { body })
    }

    /// Number of documents indexed.
    pub fn doc_count(&self) -> u64 {
        self.body.n_docs as u64
    }

    /// Full BM25 top-`k` for `query_terms` (already tokenized + lowercased;
    /// duplicates are scored once). Returns `(NodeId, score)` best-first, with a
    /// node-id tie-break for determinism. Only documents in the postings of a
    /// query term are scored — the rest of the corpus is never touched. `k =
    /// None` returns every matching document.
    pub fn search(&self, query_terms: &[String], k: Option<usize>) -> Vec<([u8; 16], f64)> {
        let n = self.body.n_docs as usize;
        let avgdl = avg_len(self.body.total_len, n);

        let mut seen: HashSet<&str> = HashSet::new();
        let mut scores: HashMap<u32, f64> = HashMap::new();
        for term in query_terms {
            if !seen.insert(term.as_str()) {
                continue;
            }
            let Some(postings) = self.body.postings.get(term) else {
                continue;
            };
            let idf = bm25_idf(n, postings.len());
            for &(di, tf) in postings {
                let len = self.body.doc_lens[di as usize] as usize;
                *scores.entry(di).or_insert(0.0) += bm25_term_score(idf, tf, len, avgdl);
            }
        }

        let mut scored: Vec<([u8; 16], f64)> = scores
            .into_iter()
            .map(|(di, s)| (self.body.doc_ids[di as usize], s))
            .collect();
        scored.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.0.cmp(&b.0))
        });
        if let Some(k) = k {
            scored.truncate(k);
        }
        scored
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn id(b: u8) -> [u8; 16] {
        let mut a = [0u8; 16];
        a[15] = b;
        a
    }

    fn build(docs: &[(u8, &str)]) -> TextIndex {
        let members: Vec<([u8; 16], String)> =
            docs.iter().map(|(b, t)| (id(*b), t.to_string())).collect();
        let (bytes, _stats) = build_body(members).unwrap().unwrap();
        TextIndex::decode(&bytes).unwrap()
    }

    fn terms(s: &str) -> Vec<String> {
        crate::text::tokenize(s)
    }

    #[test]
    fn empty_corpus_builds_nothing() {
        assert!(build_body(Vec::new()).unwrap().is_none());
    }

    #[test]
    fn rare_term_outranks_common_term() {
        // "fox" in 1 doc (rare), "common" in 4 (common). Query both → the
        // rare-term doc must rank first via real IDF.
        let idx = build(&[
            (1, "fox the cat"),
            (2, "common the cat"),
            (3, "common the dog"),
            (4, "common the bird"),
            (5, "common the lizard"),
        ]);
        assert_eq!(idx.doc_count(), 5);
        let hits = idx.search(&terms("fox common"), None);
        assert_eq!(hits.len(), 5, "all docs match a query term");
        assert_eq!(hits[0].0, id(1), "the rare-term doc ranks first");
        assert!(hits[0].1 > hits[1].1);
    }

    #[test]
    fn only_matching_docs_are_returned() {
        let idx = build(&[(1, "alpha beta"), (2, "gamma delta"), (3, "alpha gamma")]);
        let hits = idx.search(&terms("alpha"), None);
        let ids: Vec<[u8; 16]> = hits.iter().map(|(i, _)| *i).collect();
        assert_eq!(hits.len(), 2);
        assert!(ids.contains(&id(1)) && ids.contains(&id(3)));
        assert!(!ids.contains(&id(2)));
    }

    #[test]
    fn k_truncates_to_top_results() {
        let idx = build(&[(1, "x x x"), (2, "x x"), (3, "x")]);
        let hits = idx.search(&terms("x"), Some(2));
        assert_eq!(hits.len(), 2);
    }

    #[test]
    fn no_query_match_is_empty() {
        let idx = build(&[(1, "alpha"), (2, "beta")]);
        assert!(idx.search(&terms("zeta"), None).is_empty());
    }

    #[test]
    fn decode_rejects_bad_magic() {
        assert!(TextIndex::decode(b"XXXXXXXXjunk").is_err());
        assert!(TextIndex::decode(b"short").is_err());
    }

    #[test]
    fn ranking_matches_a_flat_bm25_scan() {
        // The index and a manual flat BM25 over the same corpus must agree on
        // the score (same shared math), so swapping in the index is invisible.
        let docs = [
            (1u8, "the quick brown fox"),
            (2, "the lazy dog sleeps"),
            (3, "quick fox quick fox"),
        ];
        let idx = build(&docs);
        let q = terms("quick fox");
        let hits = idx.search(&q, None);

        // Flat recompute.
        let n = docs.len();
        let total_len: u64 = docs.iter().map(|(_, t)| terms(t).len() as u64).sum();
        let avgdl = avg_len(total_len, n);
        let df = |term: &str| {
            docs.iter()
                .filter(|(_, t)| terms(t).iter().any(|w| w == term))
                .count()
        };
        let mut expect: Vec<([u8; 16], f64)> = docs
            .iter()
            .map(|(b, t)| {
                let (counts, len) = tokenize_counts(t);
                let mut s = 0.0;
                for term in ["quick", "fox"] {
                    let tf = counts.get(term).copied().unwrap_or(0);
                    s += bm25_term_score(bm25_idf(n, df(term)), tf, len, avgdl);
                }
                (id(*b), s)
            })
            .filter(|(_, s)| *s > 0.0)
            .collect();
        expect.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap().then_with(|| a.0.cmp(&b.0)));

        assert_eq!(hits.len(), expect.len());
        for (h, e) in hits.iter().zip(expect.iter()) {
            assert_eq!(h.0, e.0);
            assert!((h.1 - e.1).abs() < 1e-9, "score {} vs {}", h.1, e.1);
        }
    }
}
