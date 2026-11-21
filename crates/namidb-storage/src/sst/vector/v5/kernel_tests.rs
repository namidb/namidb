//! The int8 navigation kernels against plain scalar references, on every
//! length class (empty, below one lane group, ragged tails, the common
//! embedding widths and odd ones) and random contents.
//!
//! The kernels are portable Rust with no architecture-specific code; the
//! compiler vectorises their fixed-lane loops for each target (SSE2 on
//! x86-64, NEON on aarch64). These tests pin the compiled result to the
//! lane-order scalar arithmetic bit for bit, so a codegen or tail-handling
//! difference on one architecture fails here rather than as a wrong ranking
//! or a panic in production. CI also runs them at `opt-level = 3` on amd64,
//! where the vectoriser actually runs (`kernels (amd64, opt-level 3)`).

use rand::Rng;

use super::{approximate_score, lane_norm, VectorMetric};
use crate::kernel_test_support::{random_f32, random_i8, test_lengths, test_rng};

/// [`lane_norm`]'s arithmetic, one element at a time.
fn lane_norm_reference(query: &[f32]) -> f32 {
    let split = query.len() / 8 * 8;
    let mut lanes = [0.0f32; 8];
    for (index, q) in query[..split].iter().enumerate() {
        lanes[index % 8] += q * q;
    }
    let mut sum = lanes.iter().sum::<f32>();
    for q in &query[split..] {
        sum += q * q;
    }
    sum.sqrt()
}

/// [`approximate_score`]'s arithmetic, one element at a time.
fn approximate_score_reference(
    metric: VectorMetric,
    query: &[f32],
    query_norm: f32,
    codes: &[i8],
    scale: f32,
) -> f32 {
    let len = query.len().min(codes.len());
    let split = len / 8 * 8;
    match metric {
        VectorMetric::Dot => {
            let mut dot = [0.0f32; 8];
            for index in 0..split {
                dot[index % 8] += query[index] * codes[index] as f32;
            }
            let mut sum = dot.iter().sum::<f32>();
            for index in split..len {
                sum += query[index] * codes[index] as f32;
            }
            sum * scale
        }
        VectorMetric::Cosine => {
            let mut dot = [0.0f32; 8];
            let mut norm_sum = 0i64;
            for index in 0..split {
                dot[index % 8] += query[index] * codes[index] as f32;
                norm_sum += codes[index] as i64 * codes[index] as i64;
            }
            let mut dot_sum = dot.iter().sum::<f32>();
            for index in split..len {
                dot_sum += query[index] * codes[index] as f32;
                norm_sum += codes[index] as i64 * codes[index] as i64;
            }
            if query_norm == 0.0 || norm_sum == 0 || scale == 0.0 {
                0.0
            } else {
                dot_sum / (query_norm * (norm_sum as f32).sqrt())
            }
        }
        VectorMetric::Euclidean => {
            let mut sum = [0.0f32; 8];
            for index in 0..split {
                let delta = query[index] - codes[index] as f32 * scale;
                sum[index % 8] += delta * delta;
            }
            let mut total = sum.iter().sum::<f32>();
            for index in split..len {
                let delta = query[index] - codes[index] as f32 * scale;
                total += delta * delta;
            }
            total.sqrt()
        }
    }
}

/// The exact value in f64, for a tolerance check independent of lane order.
fn approximate_score_f64(
    metric: VectorMetric,
    query: &[f32],
    codes: &[i8],
    scale: f32,
) -> (f64, f64) {
    let len = query.len().min(codes.len());
    let (query, codes) = (&query[..len], &codes[..len]);
    let scale = scale as f64;
    let dot: f64 = query
        .iter()
        .zip(codes)
        .map(|(q, c)| *q as f64 * *c as f64)
        .sum();
    // Sum of absolute terms: the scale of the f32 rounding error.
    let magnitude: f64 = query
        .iter()
        .zip(codes)
        .map(|(q, c)| (*q as f64 * *c as f64).abs())
        .sum();
    match metric {
        VectorMetric::Dot => (dot * scale, magnitude * scale),
        VectorMetric::Cosine => {
            let q_norm = query
                .iter()
                .map(|q| *q as f64 * *q as f64)
                .sum::<f64>()
                .sqrt();
            let c_norm = codes
                .iter()
                .map(|c| *c as f64 * *c as f64)
                .sum::<f64>()
                .sqrt();
            if q_norm == 0.0 || c_norm == 0.0 || scale == 0.0 {
                (0.0, 0.0)
            } else {
                (dot / (q_norm * c_norm), magnitude / (q_norm * c_norm))
            }
        }
        VectorMetric::Euclidean => {
            let sq: f64 = query
                .iter()
                .zip(codes)
                .map(|(q, c)| {
                    let delta = *q as f64 - *c as f64 * scale;
                    delta * delta
                })
                .sum();
            (sq.sqrt(), sq.sqrt())
        }
    }
}

const METRICS: [VectorMetric; 3] = [
    VectorMetric::Dot,
    VectorMetric::Cosine,
    VectorMetric::Euclidean,
];

#[test]
fn lane_norm_matches_the_scalar_lane_order_on_every_length() {
    let mut rng = test_rng("lane_norm");
    for len in test_lengths(&mut rng) {
        let query = random_f32(&mut rng, len);
        let got = lane_norm(&query);
        assert_eq!(
            got.to_bits(),
            lane_norm_reference(&query).to_bits(),
            "len {len}"
        );
        let exact = query
            .iter()
            .map(|q| *q as f64 * *q as f64)
            .sum::<f64>()
            .sqrt();
        assert!(
            (got as f64 - exact).abs() <= 1e-5 * exact + 1e-30,
            "len {len}: {got} vs {exact}"
        );
    }
}

#[test]
fn approximate_score_matches_the_scalar_references_on_every_length() {
    let mut rng = test_rng("approximate_score");
    for len in test_lengths(&mut rng) {
        let query = random_f32(&mut rng, len);
        let codes = random_i8(&mut rng, len);
        let scale = if rng.gen_ratio(1, 8) {
            0.0
        } else {
            rng.gen_range(1e-4f32..1.0)
        };
        let query_norm = lane_norm(&query);
        for metric in METRICS {
            let got = approximate_score(metric, &query, query_norm, &codes, scale);
            let lane = approximate_score_reference(metric, &query, query_norm, &codes, scale);
            assert_eq!(
                got.to_bits(),
                lane.to_bits(),
                "{metric:?} len {len}: {got} vs lane-order {lane}"
            );
            let (exact, magnitude) = approximate_score_f64(metric, &query, &codes, scale);
            assert!(got.is_finite(), "{metric:?} len {len}: {got}");
            assert!(
                (got as f64 - exact).abs() <= 1e-4 * magnitude + 1e-6,
                "{metric:?} len {len}: {got} vs exact {exact}"
            );
        }
    }
}

/// Query and code buffers of different lengths score their common prefix
/// and never read past the shorter one.
#[test]
fn approximate_score_scores_the_common_prefix_of_mismatched_lengths() {
    let mut rng = test_rng("approximate_score mismatched");
    for len in test_lengths(&mut rng) {
        let other = rng.gen_range(0..=2100);
        let query = random_f32(&mut rng, len);
        let codes = random_i8(&mut rng, other);
        let common = len.min(other);
        let scale = rng.gen_range(1e-4f32..1.0);
        for metric in METRICS {
            let prefix_norm = lane_norm(&query[..common]);
            let got = approximate_score(metric, &query, prefix_norm, &codes, scale);
            let prefix = approximate_score(
                metric,
                &query[..common],
                prefix_norm,
                &codes[..common],
                scale,
            );
            assert_eq!(got.to_bits(), prefix.to_bits(), "{metric:?} {len}/{other}");
        }
    }
}

/// Slices that start at every offset of a lane group (unaligned to the
/// vector width) give the same bits as an aligned copy.
#[test]
fn approximate_score_is_alignment_independent() {
    let mut rng = test_rng("approximate_score alignment");
    let query = random_f32(&mut rng, 1100);
    let codes = random_i8(&mut rng, 1100);
    for offset in 0..16 {
        for len in [0, 1, 7, 8, 33, 1024, 1025] {
            let (q, c) = (&query[offset..offset + len], &codes[offset..offset + len]);
            let (q_copy, c_copy) = (q.to_vec(), c.to_vec());
            let norm = lane_norm(q);
            assert_eq!(norm.to_bits(), lane_norm(&q_copy).to_bits());
            for metric in METRICS {
                assert_eq!(
                    approximate_score(metric, q, norm, c, 0.25).to_bits(),
                    approximate_score(metric, &q_copy, norm, &c_copy, 0.25).to_bits(),
                    "{metric:?} offset {offset} len {len}"
                );
            }
        }
    }
}

/// The cosine kernel's per-lane `i32` code norm holds at the largest
/// dimension the format accepts (`MAX_DIM`), with every code at `-128`.
#[test]
fn cosine_code_norm_lanes_do_not_overflow_at_max_dim() {
    let dim = super::MAX_DIM as usize;
    let codes = vec![i8::MIN; dim];
    let query = vec![1.0f32; dim];
    let norm = lane_norm(&query);
    let score = approximate_score(VectorMetric::Cosine, &query, norm, &codes, 1.0);
    assert!((score + 1.0).abs() < 1e-3, "{score}");
}

/// Which x86 features this CPU reports. Nothing in the engine dispatches on
/// them (the kernels are portable and built for the x86-64 baseline); the
/// list is printed so a CI or emulator log shows what the tests ran on.
#[cfg(target_arch = "x86_64")]
#[test]
fn report_x86_features() {
    let features = [
        ("sse2", std::arch::is_x86_feature_detected!("sse2")),
        ("sse4.2", std::arch::is_x86_feature_detected!("sse4.2")),
        ("popcnt", std::arch::is_x86_feature_detected!("popcnt")),
        ("avx", std::arch::is_x86_feature_detected!("avx")),
        ("avx2", std::arch::is_x86_feature_detected!("avx2")),
        ("fma", std::arch::is_x86_feature_detected!("fma")),
        ("bmi2", std::arch::is_x86_feature_detected!("bmi2")),
        ("avx512f", std::arch::is_x86_feature_detected!("avx512f")),
        ("avx512bw", std::arch::is_x86_feature_detected!("avx512bw")),
        (
            "avx512vpopcntdq",
            std::arch::is_x86_feature_detected!("avx512vpopcntdq"),
        ),
    ];
    let line = features
        .iter()
        .map(|(name, on)| format!("{name}={}", if *on { "yes" } else { "no" }))
        .collect::<Vec<_>>()
        .join(" ");
    eprintln!("x86 features: {line}");
    assert!(features[0].1, "x86-64 always has SSE2");
}
