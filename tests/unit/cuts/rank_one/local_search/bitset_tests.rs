use super::*;

#[test]
fn dispatched_accumulation_matches_word_reference_and_preserves_tails() {
    for n in [0, 1, 3, 4, 7, 8, 15, 16, 17, 31, 32, 33, 65] {
        let bits: Vec<u64> = (0..n)
            .map(|i| (i as u64).wrapping_mul(0x9e3779b97f4a7c15))
            .collect();
        let states: Vec<Vec<u64>> = (0..3)
            .map(|s| (0..n + 3).map(|i| !(i as u64).rotate_left(s * 7)).collect())
            .collect();
        for changes in [vec![], vec![(0, 1)], vec![(2, 0), (1, 1), (0, 1), (2, 0)]] {
            let mut actual = vec![vec![0x5555555555555555; n + 5]; 2];
            let mut expected = actual.clone();
            for &(state, level) in &changes {
                for i in 0..n {
                    expected[level as usize][i] |= bits[i] & states[state as usize][i];
                }
            }
            accumulate(&bits, &changes, &states, &mut actual);
            assert_eq!(actual, expected, "words={n}");
        }
    }
}

#[test]
#[ignore = "manual integer-kernel timing experiment"]
fn measure_accumulation_dispatch() {
    use std::hint::black_box;
    use std::time::Instant;
    for n in [4, 16, 64, 256] {
        let bits = vec![0x123456789abcdef0; n];
        let states = vec![vec![0xfedcba9876543210; n]; 8];
        let changes: Vec<_> = (0..8).map(|i| (i, i % 3)).collect();
        let measure = |kernel: fn(&[u64], &[(i32, i32)], &[Vec<u64>], &mut [Vec<u64>])| {
            let mut target = vec![vec![0; n]; 3];
            let started = Instant::now();
            for _ in 0..100_000 {
                kernel(
                    black_box(&bits),
                    black_box(&changes),
                    black_box(&states),
                    black_box(&mut target),
                );
            }
            black_box(target);
            started.elapsed()
        };
        println!(
            "words={n} scalar={:?} dispatched={:?}",
            measure(accumulate_scalar),
            measure(accumulate)
        );
    }
}
