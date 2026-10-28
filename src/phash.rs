use image::{DynamicImage, imageops::FilterType};

/// A 64-bit perceptual hash.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PerceptualHash(pub u64);

impl std::fmt::Display for PerceptualHash {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:016x}", self.0)
    }
}

/// Calculate a perceptual hash (pHash) using a 32x32 grayscale image
/// and the low-frequency coefficients of its 2D DCT.
pub fn perceptual_hash(image: &DynamicImage) -> PerceptualHash {
    const SIZE: usize = 32;
    const HASH_SIZE: usize = 8;

    let gray = image
        .resize_exact(SIZE as u32, SIZE as u32, FilterType::Lanczos3)
        .to_luma8();

    let mut pixels = vec![0.0_f64; SIZE * SIZE];

    for y in 0..SIZE {
        for x in 0..SIZE {
            pixels[y * SIZE + x] = f64::from(gray.get_pixel(x as u32, y as u32)[0]);
        }
    }

    // Compute the 2D DCT-II by applying the 1D DCT to each row,
    // then to each column.
    for y in 0..SIZE {
        let row: Vec<f64> = (0..SIZE).map(|x| pixels[y * SIZE + x]).collect();

        let transformed = dct(&row);

        for x in 0..SIZE {
            pixels[y * SIZE + x] = transformed[x];
        }
    }

    for x in 0..SIZE {
        let column: Vec<f64> = (0..SIZE).map(|y| pixels[y * SIZE + x]).collect();

        let transformed = dct(&column);

        for y in 0..SIZE {
            pixels[y * SIZE + x] = transformed[y];
        }
    }

    // Skip the DC coefficient at (0, 0), take 8x8 low-frequency
    // coefficients, and compare each coefficient to their median.
    let mut coefficients = Vec::with_capacity(HASH_SIZE * HASH_SIZE);

    for y in 0..HASH_SIZE {
        for x in 0..HASH_SIZE {
            if x == 0 && y == 0 {
                continue;
            }
            coefficients.push(pixels[y * SIZE + x]);
        }
    }

    let median = median_f64(&mut coefficients.clone()).expect("Invalid or empty DCT coefficients");

    let mut hash = 0_u64;

    for &coefficient in &coefficients {
        hash = (hash << 1) | u64::from(coefficient > median);
    }

    PerceptualHash(hash)
}

#[inline]
fn median_f64(v: &mut [f64]) -> Option<f64> {
    let len = v.len();
    if len == 0 || v.iter().any(|x| !x.is_finite()) {
        return None;
    }

    let mid = len / 2;

    let cmp = f64::total_cmp;

    if len % 2 == 1 {
        // Odd number of elements: the middle element
        v.select_nth_unstable_by(mid, cmp);
        Some(v[mid])
    } else {
        // Even number of elements: average of the two middle elements
        v.select_nth_unstable_by(mid, cmp);
        let upper = v[mid];
        // Find the max of the lower half slice [0..mid]
        let lower = v[..mid].iter().cloned().max_by(cmp)?;
        Some(upper / 2.0 + lower / 2.0)
    }
}

#[inline]
fn dct(input: &[f64]) -> Vec<f64> {
    let n = input.len();

    (0..n)
        .map(|k| {
            let sum: f64 = input
                .iter()
                .enumerate()
                .map(|(i, xi)| {
                    xi * (std::f64::consts::PI * k as f64 / n as f64 * (i as f64 + 0.5)).cos()
                })
                .sum();

            if k == 0 {
                sum / (n as f64).sqrt()
            } else {
                sum * (2.0 / n as f64).sqrt()
            }
        })
        .collect()
}

/// Number of differing bits between two perceptual hashes.
/// Smaller distances indicate more similar hashes.
pub fn hamming_distance(a: PerceptualHash, b: PerceptualHash) -> u32 {
    (a.0 ^ b.0).count_ones()
}
