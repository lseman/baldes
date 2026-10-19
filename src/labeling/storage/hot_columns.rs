use super::*;

impl HotColumns {
    #[inline(always)]
    pub fn get(&self, i: usize) -> HotLabel {
        HotLabel {
            key: self.key[i],
            cost: self.cost[i],
            additional_cost: self.additional_cost[i],
            resources: [self.resources[0][i], self.resources[1][i]],
            hops: self.hops[i],
            ng: self.ng[i],
        }
    }

    pub fn push(&mut self, h: HotLabel) {
        self.key.push(h.key);
        self.cost.push(h.cost);
        self.additional_cost.push(h.additional_cost);
        self.resources[0].push(h.resources[0]);
        self.resources[1].push(h.resources[1]);
        self.hops.push(h.hops);
        self.ng.push(h.ng);
    }

    pub(super) fn insert(&mut self, i: usize, h: HotLabel) {
        self.key.insert(i, h.key);
        self.cost.insert(i, h.cost);
        self.additional_cost.insert(i, h.additional_cost);
        self.resources[0].insert(i, h.resources[0]);
        self.resources[1].insert(i, h.resources[1]);
        self.hops.insert(i, h.hops);
        self.ng.insert(i, h.ng);
    }

    /// Moves entries `from` down to start at `to`.
    #[inline]
    pub(super) fn move_run(&mut self, from: std::ops::Range<usize>, to: usize) {
        self.key.copy_within(from.clone(), to);
        self.cost.copy_within(from.clone(), to);
        self.additional_cost.copy_within(from.clone(), to);
        self.resources[0].copy_within(from.clone(), to);
        self.resources[1].copy_within(from.clone(), to);
        self.hops.copy_within(from.clone(), to);
        self.ng.copy_within(from, to);
    }

    pub fn truncate(&mut self, n: usize) {
        self.key.truncate(n);
        self.cost.truncate(n);
        self.additional_cost.truncate(n);
        self.resources[0].truncate(n);
        self.resources[1].truncate(n);
        self.hops.truncate(n);
        self.ng.truncate(n);
    }

    /// First index at or after `from` that passes dominance's resource and
    /// NG tests against `c`, or `len` when none does. In the `PREFIX` scan the
    /// entries are the dominating side and a key above `threshold` also
    /// stops; otherwise `c` dominates and `threshold` is unused.
    #[inline(always)]
    pub(super) fn next_candidate<
        const PREFIX: bool,
        const FORWARD: bool,
        const RESOURCES: usize,
        const NG: bool,
    >(
        &self,
        from: usize,
        threshold: f64,
        candidate_r0: f64,
        candidate_r1: f64,
        candidate_ng: u16,
    ) -> usize {
        let n = self.key.len();
        // One label's test on values, so the 8-wide block below reads
        // fixed-size arrays without bounds checks and vectorizes.
        let test = |key: f64, r0: f64, r1: f64, ng: u16| -> bool {
            let mut pass = true;
            for (r, x) in [r0, r1].into_iter().enumerate().take(RESOURCES) {
                let candidate = if r == 0 { candidate_r0 } else { candidate_r1 };
                let (a, b) = if PREFIX {
                    (x, candidate)
                } else {
                    (candidate, x)
                };
                pass &= if FORWARD {
                    a <= b + EPSILON
                } else {
                    b - EPSILON <= a
                };
            }
            if NG {
                let (a, b) = if PREFIX {
                    (ng, candidate_ng)
                } else {
                    (candidate_ng, ng)
                };
                pass &= a & !b == 0;
            }
            if PREFIX {
                (key > threshold) | pass
            } else {
                pass
            }
        };
        let hit = |i: usize| {
            test(
                self.key[i],
                self.resources[0][i],
                self.resources[1][i],
                self.ng[i],
            )
        };
        let mut i = from;
        while i + 8 <= n {
            let key: &[f64; 8] = self.key[i..i + 8].try_into().unwrap();
            let r0: &[f64; 8] = self.resources[0][i..i + 8].try_into().unwrap();
            let r1: &[f64; 8] = self.resources[1][i..i + 8].try_into().unwrap();
            let ng: &[u16; 8] = self.ng[i..i + 8].try_into().unwrap();
            let mut mask = 0u32;
            for j in 0..8 {
                mask |= (test(key[j], r0[j], r1[j], ng[j]) as u32) << j;
            }
            if mask != 0 {
                return i + mask.trailing_zeros() as usize;
            }
            i += 8;
        }
        while i < n && !hit(i) {
            i += 1;
        }
        i
    }
}
