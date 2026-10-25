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
#[path = "../tests/unit/rng/tests.rs"]
mod tests;
