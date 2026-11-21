//! The committed-memtable tail of a vector index, kept searchable at commit.
//!
//! A vector query must see every committed write, so it merges the persisted
//! index with the rows still in the memtable. Decoding every memtable record
//! on every query made that merge cost O(memtable bytes): under an embedding
//! backfill the memtable holds tens of thousands of embedding writes, and each
//! query decoded all of them (labels, text and the 4 KB vectors) to score them.
//!
//! [`VectorTail`] holds, per `(label, property)` vector index and committed
//! memtable generation, the winning version of every node id the memtable
//! touched: either a suppression (deleted, relabelled, embedding removed) or
//! an int8 code of the live embedding. It is built once per memtable
//! generation and then carried incrementally by each commit from only the
//! rows it changed (`MemtableClaimantCell`, the same pinning as the equality
//! claimant maps), so a query never decodes the memtable.
//!
//! Search is exact. Each code carries the norm of its quantisation residual,
//! so a query computes, per row, an interval that provably contains the exact
//! f32 inner product (Cauchy-Schwarz on the residuals of both the row and the
//! int8 query). Only rows whose upper bound reaches the k-th best lower bound
//! can be in the exact top-k; those few are decoded from the memtable and
//! scored with the exact metric by the caller. Rows whose code cannot bound
//! the score (zero vectors under cosine, a dimension other than the query's)
//! are always handed to the caller, which applies the flat scan's rules.

use std::sync::Arc;

use namidb_core::id::NodeId;

use crate::manifest::VectorMetric;

/// int8 code of one embedding (normalised first under cosine).
#[derive(Debug)]
pub(crate) struct TailCode {
    codes: Box<[i8]>,
    /// `x̂ = scale * codes`.
    scale: f32,
    /// `‖x − x̂‖` of the vector the code approximates.
    residual: f32,
    /// `‖x̂‖`.
    hat_norm: f32,
    /// `‖x‖` of the embedding as written (before cosine normalisation).
    norm: f32,
}

/// Winning memtable version of one node, for one vector index.
#[derive(Clone, Debug)]
pub(crate) enum TailEntry {
    /// The id's persisted index row (if any) is superseded and must not be
    /// returned: deleted, relabelled, or its embedding removed.
    Suppress,
    /// A live embedding. `None` when the code cannot bound the score; the
    /// caller always re-scores such rows exactly.
    Live(Option<Arc<TailCode>>),
}

/// `NodeId → (winning LSN, entry)`. Persistent, so carrying it across a
/// commit copies only the paths of the changed ids.
pub(crate) type VectorTail = imbl::OrdMap<NodeId, (u64, TailEntry)>;

/// Encode an embedding for the tail of an index with `metric`.
pub(crate) fn encode(embedding: &[f32], metric: VectorMetric) -> TailEntry {
    TailEntry::Live(encode_code(embedding, metric).map(Arc::new))
}

/// The int8 code of `embedding` for `metric`, or `None` when no code can
/// bound its score (non-finite values, a zero vector under cosine).
pub(crate) fn encode_code(embedding: &[f32], metric: VectorMetric) -> Option<TailCode> {
    let norm = l2(embedding);
    if !norm.is_finite() || embedding.iter().any(|x| !x.is_finite()) {
        return None;
    }
    let unit;
    let source: &[f32] = if metric == VectorMetric::Cosine {
        if norm == 0.0 {
            return None;
        }
        unit = embedding.iter().map(|x| x / norm).collect::<Vec<_>>();
        &unit
    } else {
        embedding
    };
    let (codes, scale, residual, hat_norm) = quantize(source);
    Some(TailCode {
        codes,
        scale,
        residual,
        hat_norm,
        norm,
    })
}

impl TailCode {
    /// Resident bytes of this code.
    pub(crate) fn bytes(&self) -> usize {
        self.codes.len() + std::mem::size_of::<Self>()
    }
}

fn l2(v: &[f32]) -> f32 {
    v.iter()
        .map(|x| (*x as f64) * (*x as f64))
        .sum::<f64>()
        .sqrt() as f32
}

/// Symmetric int8 quantisation with the exact residual norm.
fn quantize(v: &[f32]) -> (Box<[i8]>, f32, f32, f32) {
    let max = v.iter().fold(0.0_f32, |m, x| m.max(x.abs()));
    if max == 0.0 {
        return (vec![0; v.len()].into_boxed_slice(), 0.0, 0.0, 0.0);
    }
    let scale = max / 127.0;
    let mut residual = 0.0_f64;
    let mut hat = 0.0_f64;
    let codes = v
        .iter()
        .map(|x| {
            let code = (x / scale).round().clamp(-127.0, 127.0) as i8;
            let approx = code as f64 * scale as f64;
            residual += (*x as f64 - approx).powi(2);
            hat += approx * approx;
            code
        })
        .collect::<Vec<_>>()
        .into_boxed_slice();
    (codes, scale, residual.sqrt() as f32, hat.sqrt() as f32)
}

fn dot_i8(a: &[i8], b: &[i8]) -> i64 {
    // Chunks of 64 stay well inside i32 (64 * 127 * 127 < 2^31) and let the
    // compiler vectorise the inner loop.
    let mut total = 0_i64;
    for (ca, cb) in a.chunks(64).zip(b.chunks(64)) {
        let mut acc = 0_i32;
        for (x, y) in ca.iter().zip(cb) {
            acc += *x as i32 * *y as i32;
        }
        total += acc as i64;
    }
    total
}

/// Relative slack added to every bound for the f32 arithmetic around it.
const BOUND_SLACK: f64 = 1e-4;

/// A query prepared once for many [`TailCode`] bounds.
pub(crate) struct PreparedQuery {
    metric: VectorMetric,
    dim: usize,
    norm: f64,
    /// `‖q‖` in the space the codes live in (1 under cosine).
    len: f64,
    codes: Box<[i8]>,
    scale: f32,
    residual: f32,
    valid: bool,
}

impl PreparedQuery {
    pub(crate) fn new(query: &[f32], metric: VectorMetric) -> Self {
        let norm = l2(query) as f64;
        let cosine = metric == VectorMetric::Cosine && norm > 0.0;
        let unit;
        let source: &[f32] = if cosine {
            unit = query
                .iter()
                .map(|x| (*x as f64 / norm) as f32)
                .collect::<Vec<_>>();
            &unit
        } else {
            query
        };
        let (codes, scale, residual, _) = quantize(source);
        Self {
            metric,
            dim: query.len(),
            norm,
            len: if cosine { 1.0 } else { norm },
            codes,
            scale,
            residual,
            valid: norm.is_finite() && query.iter().all(|x| x.is_finite()),
        }
    }

    /// An interval, in higher-is-better orientation, that contains the exact
    /// score of the row `code` approximates (`-‖q − x‖²` for euclidean).
    /// `None` when the code cannot bound it.
    pub(crate) fn bounds(&self, code: &TailCode) -> Option<(f64, f64)> {
        if !self.valid || code.codes.len() != self.dim {
            return None;
        }
        let estimate =
            dot_i8(&self.codes, &code.codes) as f64 * self.scale as f64 * code.scale as f64;
        let bound = self.len * code.residual as f64
            + self.residual as f64 * code.hat_norm as f64
            + BOUND_SLACK * (self.len * code.hat_norm as f64 + 1e-9);
        Some(match self.metric {
            VectorMetric::Cosine | VectorMetric::Dot => (estimate - bound, estimate + bound),
            VectorMetric::Euclidean => {
                let fixed = self.norm * self.norm + (code.norm as f64) * (code.norm as f64);
                (
                    2.0 * (estimate - bound) - fixed,
                    2.0 * (estimate + bound) - fixed,
                )
            }
        })
    }
}

/// Keep, of `bounded` `(lower, upper, item)` triples, those whose upper bound
/// reaches the `k`-th largest lower bound: the only items that can be in the
/// exact top-`k`.
pub(crate) fn retain_possible_top_k<T>(bounded: &mut Vec<(f64, f64, T)>, k: usize) {
    if k == 0 {
        bounded.clear();
        return;
    }
    if bounded.len() <= k {
        return;
    }
    let mut lowers: Vec<f64> = bounded.iter().map(|(lower, _, _)| *lower).collect();
    let (_, kth, _) = lowers.select_nth_unstable_by(k - 1, |a, b| b.total_cmp(a));
    let threshold = *kth;
    bounded.retain(|(_, upper, _)| *upper >= threshold);
}

/// Ids of `tail` that may belong to the exact top-`k` of its live rows
/// under `metric` for `query`, plus every row whose code cannot bound its
/// score. The caller scores these exactly; every other live row provably
/// scores below the `k`-th of them.
pub(crate) fn candidates(
    tail: &VectorTail,
    query: &[f32],
    metric: VectorMetric,
    k: usize,
) -> Vec<NodeId> {
    let prepared = PreparedQuery::new(query, metric);
    let mut always = Vec::new();
    let mut bounded: Vec<(f64, f64, NodeId)> = Vec::new();
    for (id, (_, entry)) in tail.iter() {
        let TailEntry::Live(code) = entry else {
            continue;
        };
        match code.as_deref().and_then(|code| prepared.bounds(code)) {
            Some((lower, upper)) => bounded.push((lower, upper, *id)),
            None => always.push(*id),
        }
    }
    retain_possible_top_k(&mut bounded, k);
    always.extend(bounded.into_iter().map(|(_, _, id)| id));
    always
}

#[cfg(test)]
mod tests {
    use super::*;
    use uuid::Uuid;

    /// [`dot_i8`] equals the exact integer dot on every length class, over
    /// the whole `i8` range, and scores the common prefix of buffers of
    /// different lengths.
    #[test]
    fn kernel_dot_i8_matches_the_exact_integer_dot_on_every_length() {
        use crate::kernel_test_support::{random_i8, test_lengths, test_rng};
        use rand::Rng;
        let mut rng = test_rng("vector_tail dot_i8");
        for len in test_lengths(&mut rng) {
            let other = if rng.gen_ratio(1, 4) {
                rng.gen_range(0..=2100)
            } else {
                len
            };
            let (a, b) = (random_i8(&mut rng, len), random_i8(&mut rng, other));
            let exact: i64 = a.iter().zip(&b).map(|(x, y)| *x as i64 * *y as i64).sum();
            assert_eq!(dot_i8(&a, &b), exact, "{len}/{other}");
            assert_eq!(dot_i8(&b, &a), exact, "{other}/{len}");
        }
        // Worst case per 64-wide chunk: every product 128 * 128.
        let extreme = vec![i8::MIN; 4096];
        assert_eq!(dot_i8(&extreme, &extreme), 4096 * 128 * 128);
    }

    fn id(n: u64) -> NodeId {
        NodeId::from_uuid(Uuid::from_u64_pair(0, n))
    }

    fn exact(metric: VectorMetric, q: &[f32], x: &[f32]) -> f64 {
        let dot: f64 = q.iter().zip(x).map(|(a, b)| *a as f64 * *b as f64).sum();
        match metric {
            VectorMetric::Dot => dot,
            VectorMetric::Cosine => dot / (l2(q) as f64 * l2(x) as f64),
            VectorMetric::Euclidean => -q
                .iter()
                .zip(x)
                .map(|(a, b)| (*a as f64 - *b as f64).powi(2))
                .sum::<f64>(),
        }
    }

    fn lcg(seed: &mut u64) -> f32 {
        *seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*seed >> 33) as f32 / (1u64 << 31) as f32) - 0.5
    }

    #[test]
    fn candidates_always_contain_the_exact_top_k() {
        for metric in [
            VectorMetric::Cosine,
            VectorMetric::Dot,
            VectorMetric::Euclidean,
        ] {
            let mut seed = 42_u64;
            let dim = 64;
            let mut tail = VectorTail::new();
            let mut rows = Vec::new();
            for n in 0..2_000u64 {
                // Clustered data makes near-ties common.
                let centre = (n % 7) as f32;
                let v: Vec<f32> = (0..dim).map(|_| centre * 0.1 + lcg(&mut seed)).collect();
                tail.insert(id(n), (n, encode(&v, metric)));
                rows.push((id(n), v));
            }
            // Suppressed and unbounded rows.
            tail.insert(id(9_000), (1, TailEntry::Suppress));
            tail.insert(
                id(9_001),
                (1, encode(&vec![0.0; dim], VectorMetric::Cosine)),
            );
            for round in 0..20 {
                let q: Vec<f32> = (0..dim)
                    .map(|_| lcg(&mut seed) * (round as f32 + 1.0))
                    .collect();
                for k in [1, 10, 37] {
                    let found = candidates(&tail, &q, metric, k);
                    let mut scored: Vec<(f64, NodeId)> = rows
                        .iter()
                        .map(|(id, v)| (exact(metric, &q, v), *id))
                        .collect();
                    scored.sort_by(|a, b| b.0.total_cmp(&a.0));
                    for (_, want) in scored.iter().take(k) {
                        assert!(found.contains(want), "{metric:?} k={k}: exact hit missing");
                    }
                    assert!(
                        found.contains(&id(9_001)),
                        "an unbounded row is always reranked"
                    );
                    assert!(!found.contains(&id(9_000)));
                    assert!(
                        found.len() < rows.len() / 4,
                        "{metric:?} k={k}: the bound should prune ({} of {})",
                        found.len(),
                        rows.len()
                    );
                }
            }
        }
    }

    #[test]
    fn a_wrong_dimension_is_left_to_the_exact_scorer() {
        let mut tail = VectorTail::new();
        tail.insert(id(1), (1, encode(&[1.0, 2.0, 3.0], VectorMetric::Dot)));
        assert_eq!(
            candidates(&tail, &[1.0, 2.0], VectorMetric::Dot, 1),
            vec![id(1)]
        );
    }
}
