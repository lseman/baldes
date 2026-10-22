use super::*;

fn reference(
    tables: &[Vec<f64>],
    increased: &[Vec<u64>],
    decreased: &[Vec<u64>],
    nbytes: usize,
    max_rhs: i32,
) -> f64 {
    let mut gain = 0.;
    for byte in 0..nbytes {
        if max_rhs > 0 {
            for level in 1..=max_rhs as usize {
                let shift = 8 * (byte % 8);
                let coefficient = level as f64;
                let a = ((increased[level][byte / 8] >> shift) & 255) as usize;
                let b = ((decreased[level][byte / 8] >> shift) & 255) as usize;
                let x1 = tables[byte][a] * coefficient;
                let x2 = coefficient * tables[byte][b];
                gain = gain + x1;
                gain = gain - x2;
            }
        }
    }
    gain
}

#[test]
fn specialized_and_fallback_scores_match_scalar_rounding_and_partial_words() {
    let values = [
        0.,
        -0.,
        1.,
        -1.,
        1e16,
        -1e16,
        f64::from_bits(1),
        -f64::from_bits(1),
        f64::MIN_POSITIVE,
        -f64::MIN_POSITIVE,
        1.25,
        -0.75,
        1e-6,
        -1e-6,
    ];
    for nbytes in [0, 1, 7, 8, 9, 21, 22, 24, 25, 65] {
        let nwords = nbytes / 8 + usize::from(nbytes % 8 != 0);
        let tables: Vec<Vec<f64>> = (0..nbytes + 2)
            .map(|byte| {
                (0..256)
                    .map(|i| values[(i * 7 + byte) % values.len()])
                    .collect()
            })
            .collect();
        let make = |salt: u64| -> Vec<Vec<u64>> {
            (0..5)
                .map(|level| {
                    (0..nwords + 1)
                        .map(|word| {
                            (word as u64)
                                .wrapping_mul(0x9e3779b97f4a7c15)
                                .wrapping_add(salt)
                                .rotate_left(level * 13)
                        })
                        .collect()
                })
                .collect()
        };
        for salt in [0, 1, u64::MAX, 0x0123456789abcdef] {
            let increased = make(salt);
            let decreased = make(!salt);
            for levels in [-1, 0, 1, 2, 3, 4] {
                let expected = reference(&tables, &increased, &decreased, nbytes, levels);
                let actual = score(&tables, &increased, &decreased, nbytes, levels);
                assert_eq!(
                    actual.to_bits(),
                    expected.to_bits(),
                    "bytes={nbytes}, levels={levels}, salt={salt}"
                );
            }
        }
    }
}

#[test]
fn two_level_scoring_keeps_add_subtract_order_during_cancellation() {
    let mut table = vec![0.; 256];
    table[1] = 1e16;
    table[2] = 1e16;
    table[3] = 0.5;
    let increased = vec![vec![0], vec![1], vec![3]];
    let decreased = vec![vec![0], vec![2], vec![0]];
    // The first level cancels before adding the second level's contribution.
    // Grouping positive terms first would round away that contribution.
    assert_eq!(score(&[table], &increased, &decreased, 1, 2), 1.);
}
