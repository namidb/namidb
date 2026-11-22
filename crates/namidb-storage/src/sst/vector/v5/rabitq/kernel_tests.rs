//! The RaBitQ 1-bit kernels (rotation, sign coding, lookup-table code dot)
//! against scalar references on every dimension class and random contents.
//! See `v5::kernel_tests` for the length classes and why the comparisons
//! are bitwise where the arithmetic order is fixed.

use rand::Rng;

use super::{QueryScorer, Rotation, VectorMetric, PAD_MULTIPLE, ROTATION_ROUNDS, ROTATION_SEED};
use crate::kernel_test_support::{random_f32, test_lengths, test_rng};

fn rotation(dim: usize) -> Rotation {
    Rotation::new(dim, ROTATION_SEED, ROTATION_ROUNDS).expect("rotation")
}

fn dot_f64(a: &[f32], b: &[f32]) -> f64 {
    a.iter().zip(b).map(|(x, y)| *x as f64 * *y as f64).sum()
}

#[test]
fn rotation_rejects_dimension_zero_and_pads_every_other_length() {
    assert!(Rotation::new(0, ROTATION_SEED, ROTATION_ROUNDS).is_err());
    let mut rng = test_rng("rotation shape");
    for dim in test_lengths(&mut rng).into_iter().filter(|dim| *dim > 0) {
        let rotation = rotation(dim);
        assert_eq!(rotation.padded % PAD_MULTIPLE, 0, "dim {dim}");
        assert!(rotation.padded >= dim && rotation.padded < dim + PAD_MULTIPLE);
        assert_eq!(rotation.padded % rotation.block, 0, "dim {dim}");
        assert!(rotation.block.is_power_of_two());
    }
}

/// The rotation is orthogonal on every dimension: it preserves norms and
/// inner products (up to f32 rounding), whatever the padding.
#[test]
fn rotation_preserves_inner_products_on_every_length() {
    let mut rng = test_rng("rotation inner products");
    for dim in test_lengths(&mut rng).into_iter().filter(|dim| *dim > 0) {
        let rotation = rotation(dim);
        let (x, y) = (random_f32(&mut rng, dim), random_f32(&mut rng, dim));
        let mut rx = vec![f32::NAN; rotation.padded];
        let mut ry = vec![f32::NAN; rotation.padded];
        let mut scratch = vec![f32::NAN; rotation.padded];
        rotation.apply(&x, &mut rx, &mut scratch);
        rotation.apply(&y, &mut ry, &mut scratch);
        let scale = (dot_f64(&x, &x) * dot_f64(&y, &y)).sqrt();
        for (got, want) in [
            (dot_f64(&rx, &rx), dot_f64(&x, &x)),
            (dot_f64(&ry, &ry), dot_f64(&y, &y)),
            (dot_f64(&rx, &ry), dot_f64(&x, &y)),
        ] {
            assert!(
                (got - want).abs() <= 1e-4 * scale.max(want.abs()) + 1e-12,
                "dim {dim}: {got} vs {want}"
            );
        }
    }
}

/// [`QueryScorer::code_dot`]'s arithmetic, one byte at a time in its lane
/// order.
fn code_dot_lane_reference(scorer: &QueryScorer, code: &[u8]) -> f32 {
    let split = code.len() / 4 * 4;
    let mut sums = [0.0f32; 4];
    for (index, byte) in code[..split].iter().enumerate() {
        sums[index % 4] += scorer.lut[index * 256 + *byte as usize];
    }
    let mut total = sums[0] + sums[1] + sums[2] + sums[3];
    for (index, byte) in code.iter().enumerate().skip(split) {
        total += scorer.lut[index * 256 + *byte as usize];
    }
    (2.0 * total - scorer.sum_rotated) * scorer.inv_sqrt_d
}

/// `<ō, P(q - c)>` from the bits directly: `Σ ±rotated_i / sqrt(D)`.
fn code_dot_sign_reference(scorer: &QueryScorer, code: &[u8]) -> (f64, f64) {
    let mut dot = 0.0f64;
    let mut magnitude = 0.0f64;
    for (index, value) in scorer.rotated.iter().enumerate() {
        let set = code[index / 8] >> (index % 8) & 1 == 1;
        let value = *value as f64;
        dot += if set { value } else { -value };
        magnitude += value.abs();
    }
    let inv = 1.0 / (scorer.rotated.len() as f64).sqrt();
    (dot * inv, magnitude * inv)
}

/// The lookup-table code dot equals the plain sign dot on every dimension,
/// for random codes and for the codes the encoder writes.
#[test]
fn code_dot_matches_the_sign_dot_on_every_length() {
    let mut rng = test_rng("rabitq code_dot");
    for dim in test_lengths(&mut rng).into_iter().filter(|dim| *dim > 0) {
        let rotation = rotation(dim);
        let code_bytes = super::code_bytes(&rotation);
        assert_eq!(code_bytes * 8, rotation.padded);
        for metric in [
            VectorMetric::Dot,
            VectorMetric::Cosine,
            VectorMetric::Euclidean,
        ] {
            let query = random_f32(&mut rng, dim);
            let center = random_f32(&mut rng, dim);
            let mut scorer = QueryScorer::new(metric, &query, rotation.padded);
            assert_eq!(scorer.lut.len(), code_bytes * 256);
            scorer.prepare_page(&rotation, &center);

            let mut encoder = super::Encoder::new(metric, dim).expect("encoder");
            let row = random_f32(&mut rng, dim);
            let (mut codes, mut factors) = (Vec::new(), Vec::new());
            let encoded = encoder
                .encode_row(&row, &center, &mut codes, &mut factors)
                .is_ok();
            let random: Vec<u8> = (0..code_bytes).map(|_| rng.gen()).collect();
            let mut cases = vec![random, vec![0u8; code_bytes], vec![0xff; code_bytes]];
            if encoded {
                assert_eq!(codes.len(), code_bytes, "dim {dim}");
                assert_eq!(factors.len(), 3, "dim {dim}");
                cases.push(codes);
            }
            for code in cases {
                let got = scorer.code_dot(&code);
                assert_eq!(
                    got.to_bits(),
                    code_dot_lane_reference(&scorer, &code).to_bits(),
                    "{metric:?} dim {dim}"
                );
                let (exact, magnitude) = code_dot_sign_reference(&scorer, &code);
                assert!(
                    (got as f64 - exact).abs() <= 1e-4 * magnitude + 1e-6,
                    "{metric:?} dim {dim}: {got} vs {exact}"
                );
            }
        }
    }
}

/// The encoder's bits are the signs of the rotated residual, packed least
/// significant bit first, with the padding coordinates (zero) coded as 0.
#[test]
fn encoder_packs_the_rotated_signs_on_every_length() {
    let mut rng = test_rng("rabitq encode");
    for dim in test_lengths(&mut rng).into_iter().filter(|dim| *dim > 0) {
        let metric = VectorMetric::Euclidean;
        let mut encoder = super::Encoder::new(metric, dim).expect("encoder");
        let (row, center) = (random_f32(&mut rng, dim), random_f32(&mut rng, dim));
        let (mut codes, mut factors) = (Vec::new(), Vec::new());
        if encoder
            .encode_row(&row, &center, &mut codes, &mut factors)
            .is_err()
        {
            // A residual with no alignment (all-zero row and center) is refused.
            continue;
        }
        let residual: Vec<f32> = row.iter().zip(&center).map(|(x, c)| x - c).collect();
        let rotation = rotation(dim);
        let mut rotated = vec![0.0; rotation.padded];
        let mut scratch = vec![0.0; rotation.padded];
        rotation.apply(&residual, &mut rotated, &mut scratch);
        for (index, value) in rotated.iter().enumerate() {
            let bit = codes[index / 8] >> (index % 8) & 1;
            assert_eq!(bit == 1, *value > 0.0, "dim {dim} coordinate {index}");
        }
        assert!(factors.iter().all(|value| value.is_finite()));
    }
}

/// An all-zero residual codes to all-zero bytes of the full code width.
#[test]
fn a_zero_residual_codes_to_zero_bytes_on_every_length() {
    let mut rng = test_rng("rabitq zero residual");
    for dim in test_lengths(&mut rng).into_iter().filter(|dim| *dim > 0) {
        let mut encoder = super::Encoder::new(VectorMetric::Dot, dim).expect("encoder");
        let row = random_f32(&mut rng, dim);
        let (mut codes, mut factors) = (vec![0xaa], Vec::new());
        encoder
            .encode_row(&row, &row, &mut codes, &mut factors)
            .expect("zero residual");
        assert_eq!(codes.len(), 1 + encoder.code_bytes(), "dim {dim}");
        assert!(codes[1..].iter().all(|byte| *byte == 0));
        assert_eq!(factors[1..], [0.0, 0.0]);
    }
}
