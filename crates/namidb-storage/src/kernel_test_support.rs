//! Shared inputs of the kernel tests (`sst::vector::v5::kernel_tests`, the
//! RaBitQ kernel tests, `vector_tail`): length classes and random contents.

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

/// Lengths every kernel test covers: the edge classes below, then random
/// ones up to 2100. `NAMIDB_KERNEL_TEST_SEED` picks other random lengths and
/// contents (the seed in use is printed).
pub(crate) fn test_lengths(rng: &mut ChaCha8Rng) -> Vec<usize> {
    let mut lengths = vec![
        0, 1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 256,
        257, 383, 384, 512, 768, 1023, 1024, 1025, 1536, 2047, 2048, 2049, 2100,
    ];
    lengths.extend((0..32).map(|_| rng.gen_range(0..=2100)));
    lengths
}

pub(crate) fn test_rng(name: &str) -> ChaCha8Rng {
    let seed = std::env::var("NAMIDB_KERNEL_TEST_SEED")
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or(0x4b45_524e_454c_5331);
    eprintln!("{name}: NAMIDB_KERNEL_TEST_SEED={seed}");
    ChaCha8Rng::seed_from_u64(seed)
}

/// Random floats over several magnitudes, with exact zeros mixed in.
// Only the vector kernels use floats; `vector_tail` scores int8 codes.
#[cfg_attr(not(feature = "vector-index"), allow(dead_code))]
pub(crate) fn random_f32(rng: &mut ChaCha8Rng, len: usize) -> Vec<f32> {
    let magnitude = [1e-3f32, 1.0, 37.0][rng.gen_range(0..3)];
    (0..len)
        .map(|_| {
            if rng.gen_ratio(1, 16) {
                0.0
            } else {
                rng.gen_range(-1.0f32..1.0) * magnitude
            }
        })
        .collect()
}

/// Random int8 codes over the whole `i8` range: decoded pages are not
/// limited to the `±127` the quantizer writes.
pub(crate) fn random_i8(rng: &mut ChaCha8Rng, len: usize) -> Vec<i8> {
    (0..len).map(|_| rng.gen::<i8>()).collect()
}
