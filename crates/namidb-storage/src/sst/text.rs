//! Full-text inverted-index SST body (`text-index` feature).
//!
//! A `.ft` body is self-contained: it carries an inverted index (term →
//! postings of document, term frequency and token positions) plus the corpus
//! statistics BM25 needs. [`build_body`] emits `NAMIFT03`, whose footer,
//! sparse dictionary, independently-compressed posting blocks and fixed-width
//! document table can be queried through [`TextIndexV3Reader`] using byte-range
//! GETs. The legacy [`TextIndex`] full-body API reads both `NAMIFT02` and
//! `NAMIFT03`, which keeps existing snapshots and the engine integration
//! backward-compatible while the query path migrates to ranged I/O.
//!
//! The scoring math and the query syntax ([`crate::text::parse_query`]) are
//! shared with the query-time flat scan via [`crate::text`], so the index and
//! the scan return identical results for the same corpus. Like the vector
//! index, a `.ft` body reflects the **compacted** corpus as of the last
//! compaction; documents written since are served by the flat-scan fallback,
//! not this index.

use std::cmp::Ordering;
use std::collections::{BTreeMap, BinaryHeap, HashMap, HashSet};
use std::ops::Bound;

use bytes::Bytes;
use serde::{Deserialize, Serialize};

use crate::error::Error;
use crate::text::{avg_len, bm25_idf, bm25_term_score, TextQuery, PREFIX_EXPANSION_LIMIT};

mod v3;
pub mod v4;

pub use v3::{
    ExternalTextIndexBuildMetrics, ExternalTextIndexBuildOptions, TextIndexExternalBuilder,
    TextIndexFileArtifact, TextIndexRangeSource, TextIndexV3Reader, COMPACTION_SPOOL_DIR_ENV,
    INDEX_BUILD_MEMORY_ENV,
};

/// Magic prefix used to select the range-readable reader without downloading
/// the complete optional accelerator.
pub const RANGE_READABLE_MAGIC: &[u8; 8] = v3::MAGIC_V3;
/// Legacy full-body magic retained only for backward-compatible reads.
pub const LEGACY_MONOLITHIC_MAGIC: &[u8; 8] = MAGIC_V2;

/// Legacy monolithic format. V2 added token positions to postings.
const MAGIC_V2: &[u8; 8] = b"NAMIFT02";

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
    /// Inverted index: term → postings of `(document index, term frequency,
    /// ascending token offsets of the term in the document)`, ascending by
    /// document index. `postings[t].len()` is `df(t)`; the offsets are
    /// [`tokenize`] emission positions, which is what makes quoted-phrase
    /// adjacency answerable from the index alone.
    pub postings: BTreeMap<String, Vec<(u32, u32, Vec<u32>)>>,
}

/// Stats harvested at build time, mirrored into
/// [`crate::manifest::KindSpecificStats::TextIndex`].
#[derive(Debug, Clone)]
pub struct TextIndexBuildStats {
    pub doc_count: u64,
    pub term_count: u64,
    pub total_len: u64,
    /// Exact NodeId bounds of the indexed document corpus. These become the
    /// TextIndex SST key range used by the persisted freshness gate.
    pub min_node_id: [u8; 16],
    pub max_node_id: [u8; 16],
}

/// Build a text-index body from `(NodeId, document text)` pairs. The text is the
/// already-concatenated value of the indexed properties for one document.
/// Returns `Ok(None)` when there are no documents (nothing to index → the caller
/// keeps the flat-scan fallback).
pub fn build_body(
    members: Vec<([u8; 16], String)>,
) -> Result<Option<(Bytes, TextIndexBuildStats)>, Error> {
    v3::build(members)
}

/// A decoded, searchable text index.
#[derive(Debug)]
pub struct TextIndex {
    body: TextIndexBody,
    /// Sorted copy of `doc_ids` for `O(log n)` membership probes
    /// ([`Self::contains_doc`] — the label-scoped freshness gate asks whether a
    /// dirty memtable id is one of the indexed documents).
    sorted_ids: Vec<[u8; 16]>,
}

#[derive(Debug)]
struct LegacyRankedHit {
    id: [u8; 16],
    score: f64,
}

impl PartialEq for LegacyRankedHit {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl Eq for LegacyRankedHit {}

impl PartialOrd for LegacyRankedHit {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// The heap root is the worst retained hit: lower score, then larger id.
impl Ord for LegacyRankedHit {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .score
            .total_cmp(&self.score)
            .then_with(|| self.id.cmp(&other.id))
    }
}

impl TextIndex {
    /// Decode a complete v2 or v3 `.ft` body. This compatibility API
    /// materializes the index; new object-store read paths should use
    /// [`TextIndexV3Reader`] so only query-relevant ranges are decoded.
    ///
    /// A `NAMIFT01` body carries no positions and is rejected. The engine maps
    /// unknown/corrupt formats to "index absent", preserving the flat-scan
    /// correctness fallback.
    pub fn decode(bytes: &[u8]) -> Result<Self, Error> {
        if bytes.len() < MAGIC_V2.len() {
            return Err(Error::invariant("text index body too short for magic"));
        }
        let (magic, rest) = bytes.split_at(MAGIC_V2.len());
        let body = if magic == MAGIC_V2 {
            bincode::deserialize(rest)
                .map_err(|e| Error::invariant(format!("text index v2 decode failed: {e}")))?
        } else if magic == v3::MAGIC_V3 {
            v3::decode_whole(bytes)?
        } else {
            return Err(Error::invariant(format!(
                "text index magic mismatch: {magic:?}"
            )));
        };
        let mut sorted_ids = body.doc_ids.clone();
        sorted_ids.sort_unstable();
        Ok(Self { body, sorted_ids })
    }

    /// Number of documents indexed.
    pub fn doc_count(&self) -> u64 {
        self.body.n_docs as u64
    }

    /// `true` when `id` is one of the indexed documents.
    pub fn contains_doc(&self, id: &[u8; 16]) -> bool {
        self.sorted_ids.binary_search(id).is_ok()
    }

    /// Full BM25 top-`k` for pre-tokenized bag-of-words `query_terms`
    /// (duplicates are scored once) — equivalent to [`Self::search_query`]
    /// with no phrase or prefix syntax.
    pub fn search(&self, query_terms: &[String], k: Option<usize>) -> Vec<([u8; 16], f64)> {
        self.search_query(&TextQuery::from_terms(query_terms), k)
    }

    /// Full BM25 top-`k` for a parsed [`TextQuery`]. Returns `(NodeId, score)`
    /// best-first, with a node-id tie-break for determinism; `k = None`
    /// returns every matching document. Only documents in the postings of a
    /// scored term are touched.
    ///
    /// Semantics (shared verbatim with the flat-scan fallback):
    /// - every quoted phrase is a hard candidacy constraint — a document must
    ///   contain the phrase's tokens at adjacent positions, else it is
    ///   excluded even when it matches other query terms; passing documents
    ///   score the phrase's tokens as ordinary BM25 terms (adjacency gates
    ///   candidacy, it does not change the formula);
    /// - each prefix expands to the lexicographically-first
    ///   [`PREFIX_EXPANSION_LIMIT`] vocabulary terms carrying it, scored as
    ///   ordinary terms;
    /// - plain terms are bag-of-words BM25, each distinct term scored once.
    pub fn search_query(&self, query: &TextQuery, k: Option<usize>) -> Vec<([u8; 16], f64)> {
        let n = self.body.n_docs as usize;
        let avgdl = avg_len(self.body.total_len, n);

        // Scored terms: plain + phrase tokens + prefix expansions, distinct
        // and sorted so both paths sum per-term contributions in one order.
        let mut scored_terms = query.base_terms();
        for prefix in &query.prefixes {
            scored_terms.extend(
                self.body
                    .postings
                    .range::<str, _>((Bound::Included(prefix.as_str()), Bound::Unbounded))
                    .take_while(|(t, _)| t.starts_with(prefix.as_str()))
                    .take(PREFIX_EXPANSION_LIMIT)
                    .map(|(t, _)| t.as_str()),
            );
        }

        // Phrase hard constraint: intersect the per-phrase adjacency-passing
        // document sets. `Some(set)` restricts scoring to `set`.
        let mut allowed: Option<HashSet<u32>> = None;
        for phrase in &query.phrases {
            let docs = self.phrase_docs(phrase);
            allowed = Some(match allowed {
                None => docs,
                Some(acc) => acc.intersection(&docs).copied().collect(),
            });
            if allowed.as_ref().is_some_and(HashSet::is_empty) {
                return Vec::new();
            }
        }

        let mut scores: HashMap<u32, f64> = HashMap::new();
        for term in scored_terms {
            let Some(postings) = self.body.postings.get(term) else {
                continue;
            };
            let idf = bm25_idf(n, postings.len());
            for (di, tf, _positions) in postings {
                if allowed.as_ref().is_some_and(|a| !a.contains(di)) {
                    continue;
                }
                let len = self.body.doc_lens[*di as usize] as usize;
                *scores.entry(*di).or_insert(0.0) += bm25_term_score(idf, *tf, len, avgdl);
            }
        }

        let mut scored: Vec<([u8; 16], f64)> = match k {
            None => scores
                .into_iter()
                .map(|(di, score)| (self.body.doc_ids[di as usize], score))
                .collect(),
            Some(0) => Vec::new(),
            Some(limit) => {
                let limit = limit.min(scores.len());
                let mut heap = BinaryHeap::<LegacyRankedHit>::with_capacity(limit);
                for (di, score) in scores {
                    let candidate = LegacyRankedHit {
                        id: self.body.doc_ids[di as usize],
                        score,
                    };
                    if heap.len() < limit {
                        heap.push(candidate);
                        continue;
                    }
                    let Some(worst) = heap.peek() else {
                        continue;
                    };
                    let better = candidate
                        .score
                        .total_cmp(&worst.score)
                        .then_with(|| worst.id.cmp(&candidate.id))
                        == Ordering::Greater;
                    if better {
                        let _ = heap.pop();
                        heap.push(candidate);
                    }
                }
                heap.into_iter().map(|hit| (hit.id, hit.score)).collect()
            }
        };
        scored.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.0.cmp(&b.0))
        });
        scored
    }

    /// Document indexes containing `phrase`'s tokens at adjacent token
    /// positions: for some start offset `p`, token `i` of the phrase occurs
    /// at `p + i`. A phrase token absent from the vocabulary matches nothing.
    fn phrase_docs(&self, phrase: &[String]) -> HashSet<u32> {
        let Some(lists) = phrase
            .iter()
            .map(|t| self.body.postings.get(t))
            .collect::<Option<Vec<_>>>()
        else {
            return HashSet::new();
        };
        let mut out = HashSet::new();
        // Postings are ascending by document index and positions are
        // ascending offsets (build order), so both probes binary-search.
        'docs: for (di, _tf, first) in lists[0] {
            let mut rest: Vec<&[u32]> = Vec::with_capacity(lists.len() - 1);
            for list in &lists[1..] {
                match list.binary_search_by_key(di, |(d, _, _)| *d) {
                    Ok(ix) => rest.push(&list[ix].2),
                    Err(_) => continue 'docs,
                }
            }
            for &p in first {
                if rest
                    .iter()
                    .enumerate()
                    .all(|(j, ps)| ps.binary_search(&(p + 1 + j as u32)).is_ok())
                {
                    out.insert(*di);
                    continue 'docs;
                }
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::text::{parse_query, tokenize_counts};
    use std::ops::Range;
    use std::sync::{Arc, Mutex};

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
    fn builder_emits_range_readable_v3() {
        let (bytes, stats) = build_body(vec![(id(1), "legal production corpus".into())])
            .unwrap()
            .unwrap();
        assert_eq!(&bytes[..8], b"NAMIFT03");
        assert_eq!(stats.doc_count, 1);
        assert_eq!(stats.term_count, 3);
        // The compatibility decoder must continue serving the existing
        // full-body engine path while ranged integration lands.
        let idx = TextIndex::decode(&bytes).unwrap();
        assert_eq!(idx.search(&terms("legal"), None)[0].0, id(1));
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
    fn bounded_legacy_heap_matches_full_stable_ranking_for_every_k() {
        let idx = build(&[
            (1, "alpha alpha beta"),
            (2, "alpha beta"),
            (3, "alpha beta"),
            (4, "beta"),
            (5, "alpha alpha alpha"),
        ]);
        let query = parse_query("alpha beta");
        let all = idx.search_query(&query, None);
        for k in 0..=all.len() + 2 {
            let mut expected = all.clone();
            expected.truncate(k);
            assert_eq!(idx.search_query(&query, Some(k)), expected, "k={k}");
        }
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
    fn decode_rejects_legacy_v1_body() {
        // A well-formed NAMIFT01 body (position-less postings) must fail
        // decode rather than silently misparse — the read path then treats
        // the index as absent and the flat scan serves.
        #[derive(Serialize)]
        struct V1Body {
            n_docs: u32,
            total_len: u64,
            doc_ids: Vec<[u8; 16]>,
            doc_lens: Vec<u32>,
            postings: BTreeMap<String, Vec<(u32, u32)>>,
        }
        let mut postings = BTreeMap::new();
        postings.insert("fox".to_string(), vec![(0u32, 1u32)]);
        let v1 = V1Body {
            n_docs: 1,
            total_len: 1,
            doc_ids: vec![id(1)],
            doc_lens: vec![1],
            postings,
        };
        let mut bytes = b"NAMIFT01".to_vec();
        bytes.extend_from_slice(&bincode::serialize(&v1).unwrap());
        assert!(TextIndex::decode(&bytes).is_err());
    }

    #[test]
    fn decode_accepts_legacy_v2_body() {
        let mut postings = BTreeMap::new();
        postings.insert("fox".to_string(), vec![(0u32, 1u32, vec![0u32])]);
        let body = TextIndexBody {
            n_docs: 1,
            total_len: 1,
            doc_ids: vec![id(1)],
            doc_lens: vec![1],
            postings,
        };
        let mut bytes = b"NAMIFT02".to_vec();
        bytes.extend_from_slice(&bincode::serialize(&body).unwrap());
        let idx = TextIndex::decode(&bytes).unwrap();
        let hits = idx.search(&terms("fox"), None);
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].0, id(1));
    }

    #[test]
    fn phrase_query_requires_adjacency() {
        // Both docs contain "graph" AND "database"; only doc 1 has them
        // adjacent, so the quoted query must return doc 1 alone.
        let idx = build(&[
            (1, "graph database systems"),
            (2, "a database of graph paper"),
        ]);
        let q = parse_query("\"graph database\"");
        let hits = idx.search_query(&q, None);
        assert_eq!(hits.len(), 1, "adjacency must exclude doc 2: {hits:?}");
        assert_eq!(hits[0].0, id(1));

        // Once adjacency passes, the phrase's tokens score as plain terms:
        // doc 1's score equals its bag-of-words score for the same terms.
        let bag = idx.search(&terms("graph database"), None);
        let doc1_bag = bag.iter().find(|(i, _)| *i == id(1)).unwrap().1;
        assert!((hits[0].1 - doc1_bag).abs() < 1e-12);
    }

    #[test]
    fn quoted_single_token_is_a_containment_constraint() {
        // Doc 2 matches "beta" but lacks the required quoted "alpha".
        let idx = build(&[(1, "alpha beta"), (2, "gamma beta")]);
        let hits = idx.search_query(&parse_query("\"alpha\" beta"), None);
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].0, id(1));
    }

    #[test]
    fn cjk_phrase_requires_adjacent_bigrams() {
        // `"東京大"` tokenizes to the bigrams [東京, 京大]; only the
        // contiguous run 東京大学 carries them at adjacent positions.
        let idx = build(&[(1, "東京大学"), (2, "東京の大学")]);
        let hits = idx.search_query(&parse_query("\"東京大\""), None);
        assert_eq!(hits.len(), 1, "{hits:?}");
        assert_eq!(hits[0].0, id(1));
    }

    #[test]
    fn prefix_query_expands_over_the_vocabulary() {
        let idx = build(&[
            (1, "database systems"),
            (2, "dataset curation"),
            (3, "cat dog"),
        ]);
        let hits = idx.search_query(&parse_query("data*"), None);
        let ids: Vec<[u8; 16]> = hits.iter().map(|(i, _)| *i).collect();
        assert_eq!(hits.len(), 2, "{hits:?}");
        assert!(ids.contains(&id(1)) && ids.contains(&id(2)));
    }

    #[test]
    fn prefix_expansion_cap_is_deterministic() {
        // 80 single-term docs t00..t79: `t*` must expand to exactly the
        // lexicographically-first PREFIX_EXPANSION_LIMIT terms (t00..t63).
        let bodies: Vec<String> = (0..80).map(|i| format!("t{i:02}")).collect();
        let docs: Vec<(u8, &str)> = bodies
            .iter()
            .enumerate()
            .map(|(i, b)| (i as u8, b.as_str()))
            .collect();
        let idx = build(&docs);
        let hits = idx.search_query(&parse_query("t*"), None);
        assert_eq!(hits.len(), PREFIX_EXPANSION_LIMIT);
        let got: HashSet<[u8; 16]> = hits.iter().map(|(i, _)| *i).collect();
        let want: HashSet<[u8; 16]> = (0..PREFIX_EXPANSION_LIMIT as u8).map(id).collect();
        assert_eq!(got, want, "expansion must pick the lexicographic head");
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

    #[derive(Debug)]
    struct TrackingRangeSource {
        body: Bytes,
        ranges: Mutex<Vec<Range<u64>>>,
    }

    impl TrackingRangeSource {
        fn new(body: Bytes) -> Self {
            Self {
                body,
                ranges: Mutex::new(Vec::new()),
            }
        }

        fn ranges(&self) -> Vec<Range<u64>> {
            self.ranges.lock().unwrap().clone()
        }
    }

    #[async_trait::async_trait]
    impl TextIndexRangeSource for TrackingRangeSource {
        async fn read_range(&self, range: Range<u64>) -> Result<Bytes, Error> {
            if range.start > range.end || range.end > self.body.len() as u64 {
                return Err(Error::invariant("test range is outside the body"));
            }
            self.ranges.lock().unwrap().push(range.clone());
            Ok(self.body.slice(range.start as usize..range.end as usize))
        }

        async fn read_ranges(&self, ranges: &[Range<u64>]) -> Result<Vec<Bytes>, Error> {
            let mut out = Vec::with_capacity(ranges.len());
            for range in ranges {
                out.push(self.read_range(range.clone()).await?);
            }
            Ok(out)
        }
    }

    #[tokio::test]
    async fn v3_range_reader_does_not_fetch_or_decode_the_corpus() {
        // More than four dictionary blocks, and a sizeable irrelevant
        // document table/posting corpus. One rare-term lookup should touch one
        // dictionary block, one posting block and one document record.
        let docs: Vec<([u8; 16], String)> = (0..700u16)
            .map(|i| {
                let text = if i == 431 {
                    format!("term{i:04} needle needle")
                } else {
                    format!("term{i:04} filler")
                };
                let mut node = [0u8; 16];
                node[14..].copy_from_slice(&i.to_be_bytes());
                (node, text)
            })
            .collect();
        let (body, _stats) = build_body(docs).unwrap().unwrap();
        let source = Arc::new(TrackingRangeSource::new(body.clone()));
        let reader = TextIndexV3Reader::open(source.clone(), body.len() as u64)
            .await
            .unwrap();
        assert_eq!(reader.doc_count(), 700);
        let hits = reader.search(&terms("needle"), Some(5)).await.unwrap();
        assert_eq!(hits.len(), 1);
        let mut expected = [0u8; 16];
        expected[14..].copy_from_slice(&431u16.to_be_bytes());
        assert_eq!(hits[0].0, expected);

        let ranges = source.ranges();
        assert!(
            ranges
                .iter()
                .all(|range| !(range.start == 0 && range.end == body.len() as u64)),
            "a ranged query must never fetch the whole object: {ranges:?}"
        );
        let bytes_read: u64 = ranges.iter().map(|range| range.end - range.start).sum();
        assert!(
            bytes_read < body.len() as u64 / 2,
            "rare-term query read {bytes_read} of {} bytes: {ranges:?}",
            body.len()
        );
    }

    #[tokio::test]
    async fn v3_range_search_matches_full_decode_for_phrase_prefix_and_top_k() {
        let mut docs = vec![
            (id(1), "graph database dataflow".to_string()),
            (id(2), "database for graph datasets".to_string()),
            (id(3), "graph database dataset dataset".to_string()),
            (id(4), "unrelated legal text".to_string()),
        ];
        // Force several dictionary blocks without making any of those terms
        // relevant to the query.
        for i in 0..400u16 {
            let mut node = [0u8; 16];
            node[13..].copy_from_slice(&(10_000u32 + i as u32).to_be_bytes()[1..]);
            docs.push((node, format!("vocabulary{i:04}")));
        }
        let (body, _stats) = build_body(docs).unwrap().unwrap();
        let full = TextIndex::decode(&body).unwrap();
        let source = Arc::new(TrackingRangeSource::new(body.clone()));
        let ranged = TextIndexV3Reader::open(source, body.len() as u64)
            .await
            .unwrap();
        let query = parse_query("\"graph database\" data*");
        let expected = full.search_query(&query, Some(2));
        let actual = ranged.search_query(&query, Some(2)).await.unwrap();
        assert_eq!(actual.len(), expected.len());
        for (actual, expected) in actual.iter().zip(expected.iter()) {
            assert_eq!(actual.0, expected.0);
            assert!(
                (actual.1 - expected.1).abs() < 1e-12,
                "{} != {}",
                actual.1,
                expected.1
            );
        }
    }

    #[tokio::test]
    async fn v3_membership_probe_binary_searches_the_document_table() {
        let docs = (0..2_048u16)
            .map(|ordinal| {
                let mut node = [0u8; 16];
                node[14..].copy_from_slice(&ordinal.to_be_bytes());
                (node, format!("document {ordinal}"))
            })
            .collect();
        let (body, _stats) = build_body(docs).unwrap().unwrap();
        let source = Arc::new(TrackingRangeSource::new(body.clone()));
        let reader = TextIndexV3Reader::open(source.clone(), body.len() as u64)
            .await
            .unwrap();

        let mut present = [0u8; 16];
        present[14..].copy_from_slice(&1_733u16.to_be_bytes());
        let mut absent = [0u8; 16];
        absent[13..].copy_from_slice(&[1, 0, 0]);
        assert!(reader.contains_any_doc(&[absent, present]).await.unwrap());
        assert!(!reader.contains_any_doc(&[absent]).await.unwrap());

        let ranges = source.ranges();
        let membership_bytes: u64 = ranges
            .iter()
            .filter(|range| range.end - range.start == 16)
            .map(|range| range.end - range.start)
            .sum();
        assert!(
            membership_bytes < 16 * 64,
            "batched binary search read too many document IDs: {ranges:?}"
        );
    }
}
