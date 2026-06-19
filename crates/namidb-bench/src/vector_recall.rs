//! `vector-recall` — measure int8 quantization recall@k + latency vs exact
//! f32 on synthetic unit-norm vectors that mimic embeddings.
//!
//! int8 quantization is lossy, so before the storage layer commits to it we
//! measure the recall it costs. The harness quantizes the stored vectors, runs
//! brute-force top-k with the asymmetric scorer (f32 query × int8 stored, the
//! exact arithmetic the engine will use), and reports recall@k against the
//! exact f32 ranking plus the latency and bytes/vector change. It reports BOTH
//! the per-vector max-abs scale (what the engine ships) and a naive fixed-127
//! scale, so the data justifies the choice.
//!
//! Two workloads: `--clusters 0` is pure uniform-on-sphere (a pessimistic
//! floor — random vectors have no meaningful neighbours, so any noise reshuffles
//! near-tied ranks); `--clusters N` draws vectors around N centroids so the true
//! top-k are well separated, like real embeddings. Generation is deterministic
//! from `--seed`.

use std::collections::HashSet;
use std::time::Instant;

use namidb_core::quantize::{dot_i8_asymmetric, quantize_i8};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use serde::Serialize;

/// JSON report emitted to stdout.
#[derive(Debug, Serialize)]
pub struct VectorRecallReport {
    pub dim: usize,
    pub num_vectors: usize,
    pub num_queries: usize,
    pub k: usize,
    pub clusters: usize,
    /// Recall@k with per-vector max-abs scale (the scheme the engine ships).
    pub recall_at_k: f64,
    /// Recall@k with a naive fixed-127 scale, for comparison.
    pub recall_at_k_fixed_scale: f64,
    pub exact_p50_us: u128,
    pub exact_p99_us: u128,
    pub int8_p50_us: u128,
    pub int8_p99_us: u128,
    pub bytes_per_vector_f32: usize,
    /// int8 codes (dim) + the per-vector f32 scale (4 bytes).
    pub bytes_per_vector_int8: usize,
    pub compression_ratio: f64,
}

/// Fill `dim` Gaussian components via Box-Muller (uniform direction on the
/// sphere once normalized).
fn gaussian(rng: &mut ChaCha8Rng, dim: usize) -> Vec<f32> {
    let mut v = Vec::with_capacity(dim);
    while v.len() < dim {
        let u1: f32 = rng.gen::<f32>().max(1e-12);
        let u2: f32 = rng.gen::<f32>();
        let r = (-2.0 * u1.ln()).sqrt();
        let (s, c) = (std::f32::consts::TAU * u2).sin_cos();
        v.push(r * c);
        if v.len() < dim {
            v.push(r * s);
        }
    }
    v
}

fn normalize(v: &mut [f32]) {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in v {
            *x /= norm;
        }
    }
}

fn random_unit_vector(rng: &mut ChaCha8Rng, dim: usize) -> Vec<f32> {
    let mut v = gaussian(rng, dim);
    normalize(&mut v);
    v
}

/// `base + spread * gaussian`, normalized — a vector near `base`'s direction.
fn perturbed_unit_vector(rng: &mut ChaCha8Rng, base: &[f32], spread: f32) -> Vec<f32> {
    let noise = gaussian(rng, base.len());
    let mut v: Vec<f32> = base
        .iter()
        .zip(&noise)
        .map(|(b, n)| b + spread * n)
        .collect();
    normalize(&mut v);
    v
}

fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Naive fixed-127 quantize + score, for the comparison column.
fn quantize_fixed(v: &[f32]) -> Vec<i8> {
    v.iter()
        .map(|&x| (x * 127.0).round().clamp(-127.0, 127.0) as i8)
        .collect()
}
fn dot_f32_i8_fixed(q: &[f32], codes: &[i8]) -> f32 {
    q.iter()
        .zip(codes)
        .map(|(x, &c)| x * (c as f32 / 127.0))
        .sum()
}

fn top_k(scores: &[f32], k: usize) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..scores.len()).collect();
    idx.sort_by(|&a, &b| {
        scores[b]
            .partial_cmp(&scores[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    idx.truncate(k);
    idx
}

fn percentile(sorted_us: &[u128], p: f64) -> u128 {
    if sorted_us.is_empty() {
        return 0;
    }
    let rank = ((p / 100.0) * (sorted_us.len() as f64 - 1.0)).round() as usize;
    sorted_us[rank.min(sorted_us.len() - 1)]
}

fn recall(int8_top: &[usize], exact_set: &HashSet<usize>) -> usize {
    int8_top.iter().filter(|i| exact_set.contains(i)).count()
}

/// Run the harness. `clusters == 0` is uniform random; `clusters > 0` draws
/// stored vectors and queries around that many centroids (`spread` controls
/// cluster tightness).
pub fn run(
    dim: usize,
    num_vectors: usize,
    num_queries: usize,
    k: usize,
    clusters: usize,
    spread: f32,
    seed: u64,
) -> VectorRecallReport {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);

    let centroids: Vec<Vec<f32>> = (0..clusters)
        .map(|_| random_unit_vector(&mut rng, dim))
        .collect();

    let stored_f32: Vec<Vec<f32>> = (0..num_vectors)
        .map(|i| {
            if clusters == 0 {
                random_unit_vector(&mut rng, dim)
            } else {
                let base = &centroids[i % clusters];
                perturbed_unit_vector(&mut rng, base, spread)
            }
        })
        .collect();

    let stored_scaled: Vec<(Vec<i8>, f32)> = stored_f32.iter().map(|v| quantize_i8(v)).collect();
    let stored_fixed: Vec<Vec<i8>> = stored_f32.iter().map(|v| quantize_fixed(v)).collect();

    let queries: Vec<Vec<f32>> = (0..num_queries)
        .map(|i| {
            if clusters == 0 {
                random_unit_vector(&mut rng, dim)
            } else {
                // A query near a centroid: its true neighbours are that
                // cluster's members, so the top-k are well separated.
                let base = &centroids[i % clusters];
                perturbed_unit_vector(&mut rng, base, spread * 0.5)
            }
        })
        .collect();

    let mut hits_scaled = 0usize;
    let mut hits_fixed = 0usize;
    let mut total = 0usize;
    let mut exact_us: Vec<u128> = Vec::with_capacity(num_queries);
    let mut int8_us: Vec<u128> = Vec::with_capacity(num_queries);

    for q in &queries {
        let t = Instant::now();
        let exact_scores: Vec<f32> = stored_f32.iter().map(|s| dot_f32(q, s)).collect();
        let exact_top = top_k(&exact_scores, k);
        exact_us.push(t.elapsed().as_micros());

        let t = Instant::now();
        let scaled_scores: Vec<f32> = stored_scaled
            .iter()
            .map(|(codes, scale)| dot_i8_asymmetric(q, codes, *scale))
            .collect();
        let scaled_top = top_k(&scaled_scores, k);
        int8_us.push(t.elapsed().as_micros());

        let fixed_scores: Vec<f32> = stored_fixed
            .iter()
            .map(|c| dot_f32_i8_fixed(q, c))
            .collect();
        let fixed_top = top_k(&fixed_scores, k);

        let exact_set: HashSet<usize> = exact_top.iter().copied().collect();
        hits_scaled += recall(&scaled_top, &exact_set);
        hits_fixed += recall(&fixed_top, &exact_set);
        total += exact_top.len();
    }

    exact_us.sort_unstable();
    int8_us.sort_unstable();
    let div = |h: usize| {
        if total > 0 {
            h as f64 / total as f64
        } else {
            0.0
        }
    };

    VectorRecallReport {
        dim,
        num_vectors,
        num_queries,
        k,
        clusters,
        recall_at_k: div(hits_scaled),
        recall_at_k_fixed_scale: div(hits_fixed),
        exact_p50_us: percentile(&exact_us, 50.0),
        exact_p99_us: percentile(&exact_us, 99.0),
        int8_p50_us: percentile(&int8_us, 50.0),
        int8_p99_us: percentile(&int8_us, 99.0),
        bytes_per_vector_f32: dim * 4,
        bytes_per_vector_int8: dim + 4,
        compression_ratio: (dim * 4) as f64 / (dim + 4) as f64,
    }
}
