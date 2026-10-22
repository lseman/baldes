//! Shared deterministic random number generation for BALDES.
//! Each consumer owns its state; there is no process-global RNG.

/// SplitMix64 step; advances `x` like the C++ reference parameter.
fn splitmix64(x: &mut u64) -> u64 {
    *x = x.wrapping_add(0x9e3779b97f4a7c15);
    let mut z = *x;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
    z ^ (z >> 31)
}

/// Xoroshiro128+ with SplitMix64 seeding, as BALDES's C++ `math/RNG.h`.
#[derive(Clone, Debug)]
pub struct Xoroshiro128Plus {
    s0: u64,
    s1: u64,
}

impl Xoroshiro128Plus {
    /// Seed as the C++ constructor, which converts the int seed to uint64.
    pub fn new(seed: i32) -> Self {
        Self::from_seed(seed as i64 as u64)
    }

    /// Initialize from the full unsigned 64-bit seed used by the C++ RNG.
    pub fn from_seed(seed: u64) -> Self {
        let mut x = seed;
        let mut s0 = splitmix64(&mut x);
        // SplitMix64 takes its argument by reference, so seeding the second
        // word also advances the first one.
        let s1 = splitmix64(&mut s0);
        Self { s0, s1 }
    }

    /// Generate the next random u64.
    #[inline]
    pub fn next(&mut self) -> u64 {
        let s0 = self.s0;
        let mut s1 = self.s1;
        let result = s0.wrapping_add(s1);
        s1 ^= s0;
        self.s0 = s0.rotate_left(55) ^ s1 ^ (s1 << 14);
        self.s1 = s1.rotate_left(36);
        result
    }

    /// `rng() % max`, as HGS draws bounded values (no rejection sampling).
    /// Panics if max is zero.
    #[inline]
    pub fn next_u64(&mut self, max: u64) -> u64 {
        assert!(max > 0, "random bound must be positive");
        self.next() % max
    }

    /// Generate the next random u32 from the upper half of a draw.
    #[inline]
    pub fn next_u32(&mut self) -> u32 {
        (self.next() >> 32) as u32
    }

    /// Uniform integer in the inclusive interval [low, high].
    ///
    /// Panics if low is greater than high.
    pub fn uniform_i32(&mut self, low: i32, high: i32) -> i32 {
        assert!(low <= high, "random interval must be ordered");
        let range = (high as i64 - low as i64 + 1) as u64;
        (low as i64 + self.uniform_below(range) as i64) as i32
    }

    /// Random f64 in [0, 1).
    #[inline]
    pub fn next_f64(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Pick a random element from a slice.
    pub fn choose<'a, T>(&mut self, items: &'a [T]) -> &'a T {
        if items.is_empty() {
            panic!("cannot choose from empty slice");
        }
        let idx = self.next_u64(items.len() as u64) as usize;
        &items[idx]
    }

    /// Uniform value in [0, range) as libstdc++'s uniform_int_distribution
    /// with a 64-bit generator (Lemire's nearly divisionless method).
    /// Panics if range is zero.
    pub fn uniform_below(&mut self, range: u64) -> u64 {
        assert!(range > 0, "random bound must be positive");
        let mut product = self.next() as u128 * range as u128;
        if (product as u64) < range {
            let threshold = range.wrapping_neg() % range;
            while (product as u64) < threshold {
                product = self.next() as u128 * range as u128;
            }
        }
        (product >> 64) as u64
    }

    /// Shuffle a slice in place as libstdc++'s std::shuffle: two swap
    /// positions per draw when the generator range allows it.
    pub fn shuffle<T>(&mut self, items: &mut [T]) {
        let n = items.len() as u64;
        if n == 0 {
            return;
        }
        if u64::MAX / n >= n {
            let mut i = 1;
            if n % 2 == 0 {
                items.swap(1, self.uniform_below(2) as usize);
                i = 2;
            }
            while i < n {
                let swap_range = i + 1;
                let x = self.uniform_below(swap_range * (swap_range + 1));
                items.swap(i as usize, (x / (swap_range + 1)) as usize);
                items.swap(i as usize + 1, (x % (swap_range + 1)) as usize);
                i += 2;
            }
        } else {
            for i in 1..n {
                items.swap(i as usize, self.uniform_below(i + 1) as usize);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rng_matches_cpp_reference() {
        // Values from BALDES's C++ math/RNG.h and libstdc++ 16 std::shuffle.
        let mut rng = Xoroshiro128Plus::new(0);
        assert_eq!(
            [rng.next(), rng.next(), rng.next()],
            [
                2836985337983649843,
                17383111449422238058,
                15852657771511707500
            ]
        );
        assert_eq!(rng.next_u64(100), 86);
        let mut shuffled: Vec<Vec<usize>> = [1, 2, 10].iter().map(|&n| (0..n).collect()).collect();
        for v in &mut shuffled {
            rng.shuffle(v);
        }
        assert_eq!(
            shuffled,
            [vec![0], vec![1, 0], vec![7, 5, 9, 2, 3, 4, 1, 0, 8, 6]]
        );
    }
    #[test]
    fn full_width_seeds_match_archived_cpp() {
        for (seed, expected) in [
            (
                1_u64,
                [
                    10197084890663893492,
                    16183632380530289291,
                    17854549295427883785,
                ],
            ),
            (
                42_u64,
                [
                    12966046507894987950,
                    3238562683063015607,
                    4709601321616383589,
                ],
            ),
            (
                9223372036854775808_u64,
                [
                    7868295714955407760,
                    1665575697381515169,
                    10600772145784059665,
                ],
            ),
            (
                18446744073709551615_u64,
                [
                    16200281118964062572,
                    18254272431674763336,
                    1028928092969063525,
                ],
            ),
        ] {
            let mut rng = Xoroshiro128Plus::from_seed(seed);
            assert_eq!([rng.next(), rng.next(), rng.next()], expected);
        }
        for seed in [i32::MIN, -1, 0, 1, i32::MAX] {
            let mut signed = Xoroshiro128Plus::new(seed);
            let mut unsigned = Xoroshiro128Plus::from_seed(seed as i64 as u64);
            for _ in 0..100 {
                assert_eq!(signed.next(), unsigned.next());
            }
        }
    }

    #[test]
    fn distributions_handle_extreme_bounds() {
        let mut rng = Xoroshiro128Plus::new(42);
        for _ in 0..1000 {
            for bound in [1, 2, 3, (1 << 63) + 1, u64::MAX] {
                assert!(rng.uniform_below(bound) < bound);
                assert!(rng.next_u64(bound) < bound);
            }
            for (low, high) in [
                (i32::MIN, i32::MAX),
                (i32::MIN, i32::MIN),
                (i32::MAX, i32::MAX),
                (-5, 5),
            ] {
                assert!((low..=high).contains(&rng.uniform_i32(low, high)));
            }
            assert!((0.0..1.0).contains(&rng.next_f64()));
        }
    }

    #[test]
    #[should_panic(expected = "random bound must be positive")]
    fn uniform_zero_bound_is_rejected() {
        Xoroshiro128Plus::new(0).uniform_below(0);
    }

    #[test]
    #[should_panic(expected = "random bound must be positive")]
    fn modulo_zero_bound_is_rejected() {
        Xoroshiro128Plus::new(0).next_u64(0);
    }

    #[test]
    #[should_panic(expected = "random interval must be ordered")]
    fn reversed_interval_is_rejected() {
        Xoroshiro128Plus::new(0).uniform_i32(1, -1);
    }

    #[test]
    fn shuffle_handles_empty_singleton_odd_and_even_lengths() {
        for n in [0, 1, 2, 3, 10, 31] {
            let mut rng = Xoroshiro128Plus::new(42);
            let mut values: Vec<_> = (0..n).collect();
            rng.shuffle(&mut values);
            values.sort_unstable();
            assert_eq!(values, (0..n).collect::<Vec<_>>());
        }
    }
}
