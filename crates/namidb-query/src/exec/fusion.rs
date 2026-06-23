//! Hybrid-search rank fusion: combine a dense (vector KNN) and a sparse (BM25)
//! ranked result set into a single ranking.
//!
//! Two strategies, matching what Elasticsearch / Weaviate / Qdrant / pgvector
//! ship:
//!
//! - **Reciprocal Rank Fusion (RRF)** — the default. Each leg contributes
//!   `1 / (rrf_k + rank)` (rank 1-based) for the nodes it ranks; the fused score
//!   is the sum across legs. It is purely rank-based, so it needs no calibration
//!   between the cosine-similarity scale of the dense leg and the unbounded BM25
//!   scale of the sparse leg — which is exactly why it is everyone's default.
//! - **Weighted linear** — min-max-normalise each leg's scores into `[0, 1]`
//!   and take a weighted sum. More tunable, but sensitive to score outliers and
//!   the calibration RRF avoids, so it is opt-in.
//!
//! Both are pure functions over best-first `(NodeId, score)` legs and produce a
//! deterministic ranking (fused score desc, `NodeId` asc on ties), so a hybrid
//! query is reproducible and its parity is testable.

use std::collections::BTreeMap;

use namidb_core::id::NodeId;

/// The standard RRF constant (k = 60), as used by the major engines.
pub const DEFAULT_RRF_K: f64 = 60.0;

/// Reciprocal Rank Fusion over best-first `legs`. A node's fused score is
/// `Σ 1 / (rrf_k + rank)` over the legs that rank it (rank is 1-based position).
/// Returns `(node, fused_score)` sorted by score desc, `NodeId` asc on ties.
///
/// Rank-based, so the legs' score *scales* are irrelevant — only their order
/// matters; each leg must already be sorted best-first (which the vector and
/// BM25 retrievers guarantee). A node absent from a leg simply contributes
/// nothing from it.
pub fn rrf(legs: &[&[(NodeId, f64)]], rrf_k: f64) -> Vec<(NodeId, f64)> {
    let mut acc: BTreeMap<NodeId, f64> = BTreeMap::new();
    for leg in legs {
        for (rank, (id, _score)) in leg.iter().enumerate() {
            *acc.entry(*id).or_insert(0.0) += 1.0 / (rrf_k + (rank as f64 + 1.0));
        }
    }
    finalize(acc)
}

/// Weighted-linear fusion. Each leg's scores are min-max-normalised into
/// `[0, 1]` using the leg's best-first order (best → 1, worst → 0), so the
/// normalisation is orientation-agnostic (it works for a lower-is-closer
/// euclidean leg as well as higher-is-closer cosine/BM25). The fused score is
/// `Σ wᵢ · normᵢ`; a node absent from a leg contributes 0 there. A leg whose
/// hits all share one score (zero range) — or a single-hit leg — normalises its
/// present hits to 1.0.
pub fn linear(legs: &[&[(NodeId, f64)]], weights: &[f64]) -> Vec<(NodeId, f64)> {
    let mut acc: BTreeMap<NodeId, f64> = BTreeMap::new();
    for (li, leg) in legs.iter().enumerate() {
        let w = weights.get(li).copied().unwrap_or(0.0);
        if leg.is_empty() || w == 0.0 {
            continue;
        }
        // Best-first: first element is the closest, last the farthest. Mapping
        // (score − worst) / (best − worst) sends best → 1, worst → 0 for either
        // orientation (the sign of `range` cancels).
        let best = leg.first().unwrap().1;
        let worst = leg.last().unwrap().1;
        let range = best - worst;
        for (id, score) in leg.iter() {
            let norm = if range == 0.0 {
                1.0
            } else {
                (score - worst) / range
            };
            *acc.entry(*id).or_insert(0.0) += w * norm;
        }
    }
    finalize(acc)
}

/// Sort the accumulated scores best-first with a deterministic tie-break. The
/// `BTreeMap` yields `NodeId`-ascending, and the stable sort by score descending
/// preserves that order on ties.
fn finalize(acc: BTreeMap<NodeId, f64>) -> Vec<(NodeId, f64)> {
    let mut out: Vec<(NodeId, f64)> = acc.into_iter().collect();
    out.sort_by(|a, b| b.1.total_cmp(&a.1));
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use uuid::Uuid;

    fn nid(b: u8) -> NodeId {
        NodeId(Uuid::from_bytes([b; 16]))
    }

    #[test]
    fn rrf_rewards_agreement_across_legs() {
        // Node 1 is top of the dense leg and mid of the sparse leg; node 2 is top
        // of sparse only. With agreement, node 1 should win.
        let dense = [(nid(1), 0.99), (nid(3), 0.80), (nid(2), 0.10)];
        let sparse = [(nid(2), 12.0), (nid(1), 9.0), (nid(4), 1.0)];
        let fused = rrf(&[&dense, &sparse], DEFAULT_RRF_K);
        assert_eq!(fused[0].0, nid(1), "agreeing node ranks first: {fused:?}");
        // Every node from either leg appears once.
        assert_eq!(fused.len(), 4);
    }

    #[test]
    fn rrf_single_leg_preserves_order() {
        let dense = [(nid(5), 0.9), (nid(6), 0.5), (nid(7), 0.1)];
        let fused = rrf(&[&dense, &[]], DEFAULT_RRF_K);
        assert_eq!(
            fused.iter().map(|(id, _)| *id).collect::<Vec<_>>(),
            vec![nid(5), nid(6), nid(7)]
        );
    }

    #[test]
    fn rrf_is_deterministic_on_ties() {
        // Two nodes each appear once at rank 1 of a different leg → equal RRF
        // score; the tie-break is NodeId asc.
        let a = [(nid(2), 1.0)];
        let b = [(nid(1), 1.0)];
        let fused = rrf(&[&a, &b], DEFAULT_RRF_K);
        assert_eq!(fused[0].0, nid(1));
        assert_eq!(fused[1].0, nid(2));
    }

    #[test]
    fn linear_blends_normalised_scores() {
        // alpha on dense = 1.0, sparse weight 0 → pure dense order.
        let dense = [(nid(1), 0.9), (nid(2), 0.1)];
        let sparse = [(nid(2), 100.0), (nid(1), 1.0)];
        let pure_dense = linear(&[&dense, &sparse], &[1.0, 0.0]);
        assert_eq!(pure_dense[0].0, nid(1));
        // Equal weights: node 2 is worst in dense (norm 0) but best in sparse
        // (norm 1); node 1 is best in dense (norm 1) and worst in sparse (norm 0).
        // Tie at 0.5 each → NodeId asc.
        let blended = linear(&[&dense, &sparse], &[0.5, 0.5]);
        assert!((blended[0].1 - 0.5).abs() < 1e-9);
        assert_eq!(blended[0].0, nid(1));
    }

    #[test]
    fn linear_zero_range_leg_normalises_to_one() {
        // A leg whose hits all share a score must not divide by zero.
        let flat = [(nid(1), 5.0), (nid(2), 5.0)];
        let out = linear(&[&flat], &[1.0]);
        assert!(out.iter().all(|(_, s)| (*s - 1.0).abs() < 1e-9));
    }

    #[test]
    fn linear_handles_euclidean_orientation() {
        // A lower-is-closer leg (euclidean distances): best is the smallest.
        // best-first order already encodes that, so norm sends the first → 1.
        let dist = [(nid(1), 0.2), (nid(2), 0.5), (nid(3), 0.9)];
        let out = linear(&[&dist], &[1.0]);
        assert_eq!(out[0].0, nid(1));
        assert!((out[0].1 - 1.0).abs() < 1e-9);
    }
}
