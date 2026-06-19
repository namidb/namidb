//! Vamana graph build ([`build`]) + the α-robust prune ([`robust_prune`]).
//!
//! Vamana (the DiskANN graph algorithm) builds a bounded-degree search graph in
//! one pass: for each point, find its `L_build` nearest members by a greedy
//! search over the graph-so-far, α-prune that candidate set down to `R`
//! diverse neighbours, write those as the point's out-edges, and add the
//! reverse edges (re-pruning the recipient if it overflows `R`). A larger `α`
//! prunes less aggressively → more diverse neighbours → better recall.
//!
//! References: Subramanya et al., "DiskANN: Fast Accurate Billion-point
//! Nearest Neighbor Search on a Single Node" (NeurIPS 2019), Algorithm 1
//! (RobustPrune) + Algorithm 2 (Vamana Index).

use rand::seq::{IteratorRandom, SliceRandom};
use rand::Rng;
use rand_chacha::ChaCha8Rng;
use rand::SeedableRng;

use crate::graph::VamanaGraph;
use crate::search::beam_search;
use crate::space::VectorSpace;

/// Build knobs. Defaults follow DiskANN's small/medium-set recommendations:
/// `R = 64`, `L_build = 128`, `α = 1.2`.
#[derive(Clone, Copy, Debug)]
pub struct BuildParams {
    /// Max out-degree (`R`). Larger → better recall, more RAM/edge storage.
    pub r: usize,
    /// Build-time search beam (`L_build`). Must be `≥ r`; larger → better
    /// neighbour candidates → better recall, slower build.
    pub l_build: usize,
    /// Robust-prune diversification (`α`). `1.0` = standard greedy NN; `> 1.0`
    /// (e.g. 1.2) keeps more diverse neighbours and improves recall.
    pub alpha: f32,
    /// Initial neighbour-list strategy before the main refine loop.
    pub init: InitStrategy,
}

impl Default for BuildParams {
    fn default() -> Self {
        Self {
            r: 64,
            l_build: 128,
            alpha: 1.2,
            init: InitStrategy::Auto,
        }
    }
}

/// How neighbour lists are seeded before the main refine pass.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InitStrategy {
    /// `R` random members each — fast, the refine loop does the real work.
    Random,
    /// `R` true nearest members (brute-force `O(N²)` distances) — highest
    /// quality, only for modest `N`.
    BruteForce,
    /// BruteForce below [`AUTO_BRUTEFORCE_MAX`], Random above.
    Auto,
}

/// Below this set size the build uses brute-force init (better starting graph);
/// above it, random init (avoids the `O(N²)` cost). SST-sized sets usually land
/// under this, but a wide L1 merge can exceed it.
pub const AUTO_BRUTEFORCE_MAX: usize = 4_000;

/// `(distance-from-anchor, candidate-id)` pair — the working type for prune.
type Cand = (f32, u32);

/// α-robust prune (DiskANN RobustPrune). Given the candidate set of neighbour
/// ids (with their distance to `anchor`), select at most `r` that cover the
/// directions away from `anchor`. Removes a candidate `p''` when it lies in the
/// "shadow" of an already-kept `p'` — i.e. `α · d(p', p'') ≤ d(anchor, p'')` —
/// since `p'` then reaches that region more directly. Larger `α` removes fewer
/// candidates → more diverse, higher-recall (but denser) neighbour lists.
///
/// `anchor` itself is excluded if present; duplicate ids collapse to the
/// closest. The returned ids are NOT guaranteed sorted.
pub fn robust_prune<S: VectorSpace>(
    space: &S,
    anchor: u32,
    mut candidates: Vec<Cand>,
    alpha: f32,
    r: usize,
) -> Vec<u32> {
    if candidates.is_empty() || r == 0 {
        return Vec::new();
    }
    // Drop the anchor, sort ascending by distance to it (tie-break by id),
    // dedupe by id keeping the closest.
    candidates.retain(|(_, id)| *id != anchor);
    candidates.sort_unstable_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
    candidates.dedup_by(|a, b| a.1 == b.1);

    let mut out = Vec::with_capacity(r.min(candidates.len()));
    let mut i = 0;
    while i < candidates.len() {
        if out.len() == r {
            break;
        }
        let (_, p_star) = candidates[i];
        out.push(p_star);
        // In-place filter of candidates[i+1..]: drop p'' iff
        // α·d(p_star, p'') ≤ d(anchor, p'') (== candidates[..].0).
        let mut write = i + 1;
        for j in (i + 1)..candidates.len() {
            let (d_anchor_pp, p_pp) = candidates[j];
            let d_star_pp = space.pair_distance(p_star, p_pp);
            if alpha * d_star_pp <= d_anchor_pp {
                // redundant — drop (don't copy forward)
            } else {
                candidates[write] = candidates[j];
                write += 1;
            }
        }
        candidates.truncate(write);
        i += 1;
    }
    out
}

/// Build a [`VamanaGraph`] over `space`. The RNG seeds the entry-point sample,
/// the random-permutation order, and (for `Random` init) the seed neighbours;
/// pass a fixed seed (`rand_chacha::ChaCha8Rng::seed_from_u64`) for a
/// deterministic build. Returns a graph indexing all `space.len()` members.
pub fn build<S: VectorSpace, R: Rng>(
    space: &S,
    params: BuildParams,
    rng: &mut R,
) -> VamanaGraph {
    let n = space.len();
    if n == 0 {
        return VamanaGraph::new(Vec::new(), 0);
    }
    if n == 1 {
        return VamanaGraph::new(vec![Vec::new()], 0);
    }
    // l_build must be ≥ r to give prune enough candidates.
    let params = BuildParams {
        l_build: params.l_build.max(params.r + 1),
        ..params
    };

    let entry = approximate_medoid(space, rng);

    // 1. Seed adjacency.
    let mut adj: Vec<Vec<u32>> = match params.init {
        InitStrategy::Random => random_init(n, params.r, rng),
        InitStrategy::BruteForce => brute_force_init(space, params.r),
        InitStrategy::Auto if n <= AUTO_BRUTEFORCE_MAX => brute_force_init(space, params.r),
        InitStrategy::Auto => random_init(n, params.r, rng),
    };

    // 2. Refine pass over a random permutation.
    let mut order: Vec<u32> = (0..n as u32).collect();
    order.shuffle(rng);
    let l_build = params.l_build;
    let alpha = params.alpha;
    let r = params.r;

    for &i in &order {
        // Find l_build nearest members to i over the graph-so-far.
        let found = beam_search(
            &adj,
            n,
            entry,
            l_build,
            l_build,
            |id| space.pair_distance(i, id),
        );
        // Candidate set = found neighbours (dist to i), excluding i.
        let cands: Vec<Cand> = found.into_iter().filter(|nb| nb.id != i).map(|nb| (nb.dist, nb.id)).collect();
        let new_n = robust_prune(space, i, cands, alpha, r);
        // Write i's pruned out-list.
        adj[i as usize] = new_n.clone();
        // Back-edges; re-prune recipients that overflow r.
        for &j in &new_n {
            let list = &mut adj[j as usize];
            if !list.contains(&i) {
                list.push(i);
            }
            if list.len() > r {
                let cj: Vec<Cand> = list
                    .iter()
                    .map(|&nb| (space.pair_distance(j, nb), nb))
                    .collect();
                // Rebuild after computing all distances (borrow ends above).
                let pruned = robust_prune(space, j, cj, alpha, r);
                adj[j as usize] = pruned;
            }
        }
    }

    VamanaGraph::new(adj, entry)
}

/// Build with a deterministic `ChaCha8Rng` seeded from `seed`. Convenience for
/// callers (storage compaction, the recall harness) that want reproducible
/// builds without constructing an RNG themselves — same `(data, params, seed)`
/// always yields the same graph.
pub fn build_with_seed<S: VectorSpace>(
    space: &S,
    params: BuildParams,
    seed: u64,
) -> VamanaGraph {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    build(space, params, &mut rng)
}

/// Approximate medoid: the sample member minimizing total distance to the
/// sample. For `n ≤ MEDOID_SAMPLE` this is the exact medoid.
fn approximate_medoid<S: VectorSpace, R: Rng>(space: &S, rng: &mut R) -> u32 {
    const MEDOID_SAMPLE: usize = 256;
    let n = space.len();
    let sample: Vec<u32> = if n <= MEDOID_SAMPLE {
        (0..n as u32).collect()
    } else {
        (0..n as u32).choose_multiple(rng, MEDOID_SAMPLE)
    };
    let mut best = sample[0];
    let mut best_sum = f32::INFINITY;
    for &cand in &sample {
        let sum: f32 = sample.iter().map(|&s| space.pair_distance(cand, s)).sum();
        if sum < best_sum {
            best_sum = sum;
            best = cand;
        }
    }
    best
}

/// Brute-force `R`-NN init: for each member, its `R` closest by true distance.
fn brute_force_init<S: VectorSpace>(space: &S, r: usize) -> Vec<Vec<u32>> {
    let n = space.len();
    (0..n as u32)
        .map(|i| {
            let mut scored: Vec<Cand> = (0..n as u32)
                .filter(|&j| j != i)
                .map(|j| (space.pair_distance(i, j), j))
                .collect();
            scored.sort_unstable_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
            scored.into_iter().take(r).map(|(_, id)| id).collect()
        })
        .collect()
}

/// Random init: `min(r, n-1)` distinct random neighbours per member.
fn random_init<R: Rng>(n: usize, r: usize, rng: &mut R) -> Vec<Vec<u32>> {
    let take = r.min(n.saturating_sub(1));
    (0..n as u32)
        .map(|_| {
            let mut picks: Vec<u32> = (0..n as u32).collect();
            picks.shuffle(rng);
            picks.into_iter().take(take).collect()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::space::{F32CosineSpace, Int8Space};
    use crate::search::search;
    use rand_chacha::ChaCha8Rng;
    use rand::SeedableRng;

    /// Clustered unit vectors: `clusters` centroids, members perturbed around
    /// them — the regime where ANN recall is meaningful (true NN well-separated).
    fn clustered(cosine: bool, n: usize, clusters: usize, dim: usize, seed: u64) -> F32CosineSpace {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let centroids: Vec<Vec<f32>> = (0..clusters)
            .map(|_| random_unit(&mut rng, dim))
            .collect();
        let vecs: Vec<Vec<f32>> = (0..n)
            .map(|i| perturbed(&mut rng, &centroids[i % clusters], 0.15))
            .collect();
        let _ = cosine;
        F32CosineSpace::new(vecs)
    }

    fn gaussian(rng: &mut ChaCha8Rng, dim: usize) -> Vec<f32> {
        let mut v = Vec::with_capacity(dim);
        while v.len() < dim {
            let u1: f32 = rng.gen::<f32>().max(1e-12);
            let u2: f32 = rng.gen::<f32>();
            let r = (-2.0 * u1.ln()).sqrt();
            let (s, c) = (std::f32::consts::TAU * u2).sin_cos();
            v.push(r * c);
            if v.len() < dim { v.push(r * s); }
        }
        v
    }
    fn normalize(v: &mut [f32]) {
        let nrm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if nrm > 0.0 { for x in v { *x /= nrm; } }
    }
    fn random_unit(rng: &mut ChaCha8Rng, dim: usize) -> Vec<f32> {
        let mut v = gaussian(rng, dim);
        normalize(&mut v);
        v
    }
    fn perturbed(rng: &mut ChaCha8Rng, base: &[f32], spread: f32) -> Vec<f32> {
        let noise = gaussian(rng, base.len());
        let mut v: Vec<f32> = base.iter().zip(&noise).map(|(b, n)| b + spread * n).collect();
        normalize(&mut v);
        v
    }

    /// Brute-force exact top-k cosine ids (the recall ground truth).
    fn exact_topk(space: &F32CosineSpace, query: &[f32], k: usize) -> Vec<u32> {
        let mut scored: Vec<(f32, u32)> = (0..space.len() as u32)
            .map(|i| (space.query_distance(query, i), i))
            .collect();
        scored.sort_unstable_by(|a, b| a.0.total_cmp(&b.0));
        scored.into_iter().take(k).map(|(_, i)| i).collect()
    }

    fn recall(approx: &[u32], truth: &[u32]) -> f64 {
        let t: std::collections::HashSet<u32> = truth.iter().copied().collect();
        let hits = approx.iter().filter(|i| t.contains(i)).count();
        hits as f64 / truth.len().max(1) as f64
    }

    #[test]
    fn build_empty_and_single() {
        let mut rng = ChaCha8Rng::seed_from_u64(1);
        let empty = F32CosineSpace::new(vec![]);
        let g = build(&empty, BuildParams::default(), &mut rng);
        assert!(g.is_empty());

        let one = F32CosineSpace::new(vec![vec![1.0, 0.0]]);
        let g = build(&one, BuildParams::default(), &mut rng);
        assert_eq!(g.len(), 1);
        assert_eq!(g.max_degree(), 0);
    }

    #[test]
    fn degree_bounded_by_r_plus_overshoot() {
        // After build, no node exceeds R out-neighbours (back-edge prune caps it).
        let space = clustered(true, 200, 10, 32, 7);
        let mut rng = ChaCha8Rng::seed_from_u64(3);
        let params = BuildParams { r: 16, l_build: 32, alpha: 1.2, init: InitStrategy::BruteForce };
        let g = build(&space, params, &mut rng);
        assert!(
            g.max_degree() <= 16,
            "degree {} exceeded R=16",
            g.max_degree()
        );
    }

    #[test]
    fn recall_on_clustered_data_f32() {
        // With well-separated clusters, recall@10 should be near-perfect.
        let n = 500;
        let clusters = 20;
        let dim = 48;
        let space = clustered(true, n, clusters, dim, 42);
        let mut rng = ChaCha8Rng::seed_from_u64(99);
        let params = BuildParams { r: 32, l_build: 64, alpha: 1.2, init: InitStrategy::Auto };
        let g = build(&space, params, &mut rng);

        let k = 10;
        let ef = 64;
        let mut total_recall = 0.0;
        let mut rng = ChaCha8Rng::seed_from_u64(5);
        for i in 0..50 {
            let q = perturbed(&mut rng, &space.vector((i as u32) % (clusters as u32)), 0.1);
            let truth = exact_topk(&space, &q, k);
            let approx: Vec<u32> = search(&space, &g, &q, k, ef).into_iter().map(|n| n.id).collect();
            total_recall += recall(&approx, &truth);
        }
        let avg = total_recall / 50.0;
        assert!(
            avg >= 0.90,
            "recall@{k} = {avg:.3}, expected ≥ 0.90 on clustered data"
        );
    }

    #[test]
    fn recall_int8_tracks_f32() {
        // int8 quantization costs some recall but should stay high on clusters.
        let n = 400;
        let clusters = 16;
        let dim = 64;
        let space = clustered(true, n, clusters, dim, 17);
        // Build the int8 space from the same vectors.
        let members: Vec<(Vec<i8>, f32)> = (0..space.len())
            .map(|i| namidb_core::quantize::quantize_i8(space.vector(i as u32)))
            .collect();
        let i8space = Int8Space::new(members);

        let mut rng = ChaCha8Rng::seed_from_u64(8);
        let params = BuildParams { r: 32, l_build: 64, alpha: 1.2, init: InitStrategy::Auto };
        let g = build(&i8space, params, &mut rng);

        let k = 10;
        let ef = 64;
        let mut total_recall = 0.0;
        let mut rng = ChaCha8Rng::seed_from_u64(5);
        for i in 0..50 {
            // Use the f32 ground truth, search over the int8 graph/space.
            let q = perturbed(&mut rng, &space.vector((i as u32) % (clusters as u32)), 0.1);
            let truth = exact_topk(&space, &q, k);
            let approx: Vec<u32> = search(&i8space, &g, &q, k, ef).into_iter().map(|n| n.id).collect();
            total_recall += recall(&approx, &truth);
        }
        let avg = total_recall / 50.0;
        assert!(
            avg >= 0.80,
            "int8 recall@{k} = {avg:.3}, expected ≥ 0.80 (quantization floor)"
        );
    }

    #[test]
    fn robust_prune_caps_and_excludes_anchor() {
        // 5 candidates all equidistant-ish; prune to r=2, anchor excluded.
        let space = F32CosineSpace::new(vec![
            vec![1.0, 0.0],   // 0 = anchor
            vec![0.9, 0.1],   // 1 near anchor
            vec![0.8, 0.2],   // 2
            vec![0.1, 0.9],   // 3 far / different dir
            vec![-1.0, 0.0],  // 4 opposite
        ]);
        let cands = vec![
            (space.pair_distance(0, 1), 1),
            (space.pair_distance(0, 2), 2),
            (space.pair_distance(0, 3), 3),
            (space.pair_distance(0, 4), 4),
        ];
        let out = robust_prune(&space, 0, cands, 1.0, 2);
        assert!(out.len() <= 2);
        assert!(!out.contains(&0), "anchor must be excluded");
    }

    #[test]
    fn build_is_deterministic_for_fixed_seed() {
        let space = clustered(true, 100, 5, 16, 31);
        let p = BuildParams::default();
        let g1 = build(&space, p, &mut ChaCha8Rng::seed_from_u64(1));
        let g2 = build(&space, p, &mut ChaCha8Rng::seed_from_u64(1));
        assert_eq!(g1.adjacency, g2.adjacency, "same seed → same graph");
        assert_eq!(g1.entry, g2.entry);
    }
}
