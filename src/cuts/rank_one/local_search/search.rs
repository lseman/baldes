use super::*;
use rayon::prelude::*;

impl<const COVERING: bool> LocalSearch<COVERING> {
    pub(super) fn counts(c: &ColumnInfo) -> &Vec<i32> {
        if COVERING {
            &c.covering_set_count
        } else {
            &c.packing_set_count
        }
    }

    /// The count-difference bitset of rows (from, to) at difference c.
    #[inline]
    pub(super) fn difference(&self, from: usize, to: usize, c: usize) -> &[u64] {
        let rows = self.num_rows.max(0) as usize;
        let slots = (self.max_count.max(0) + 1) as usize;
        let start = ((from * rows + to) * slots + c) * self.nwords;
        &self.count_difference_bits[start..start + self.nwords]
    }

    pub(super) fn divide(a: i32, d: i32) -> i32 {
        if COVERING {
            a / d + (a % d != 0) as i32
        } else {
            a / d
        }
    }

    /// Largest weighted row count over the columns.
    pub(super) fn compute_max_state(
        &self,
        columns: &[ColumnInfo],
        k: usize,
        numerators: &[i32],
    ) -> i32 {
        let mut best = 0;
        let mut m = numerators.to_vec();
        m.sort();
        let mut top = vec![0i32; k];
        for &c in &self.column_indices {
            top.fill(0);
            let cnt = Self::counts(&columns[c as usize]);
            for r in 0..self.num_rows as usize {
                let v = cnt[r];
                if v > top[0] {
                    top[0] = v;
                    top.sort();
                }
            }
            let mut s: u32 = 0;
            for i in 0..k {
                s = s.wrapping_add((top[i] as u32).wrapping_mul(m[i] as u32));
            }
            let si = s as i32;
            best = if best >= si { best } else { si };
        }
        best
    }

    /// Best-improvement multiplier swaps; returns the cut violation.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn run_local_search(
        &self,
        parent: &Separation,
        columns: &[ColumnInfo],
        rows: &mut Vec<i32>,
        multipliers: &mut Vec<i32>,
        denominator: i32,
        rhs: f64,
        max_state: i32,
    ) -> f64 {
        let max_rhs = Self::divide(max_state, denominator);
        let n = self.column_indices.len();
        let nbytes = n / 8 + (n & 7 != 0) as usize;
        let nwords = nbytes / 8 + (nbytes & 7 != 0) as usize;
        let row_count = self.num_rows.max(0) as usize;
        let max_multiplier = *multipliers.iter().max().unwrap();
        let max_change = self.max_count * max_multiplier;
        let mut increases: Vec<Vec<(i32, i32)>> = vec![vec![]; (max_change + 1) as usize];
        for d in 1..=max_change {
            for s in 0..=max_state - d {
                let (a, b) = (
                    Self::divide(s, denominator),
                    Self::divide(d + s, denominator),
                );
                if a < b {
                    increases[d as usize].push((s, b - a));
                }
            }
        }
        let mut decreases: Vec<Vec<(i32, i32)>> = vec![vec![]; (max_change + 1) as usize];
        for d in 1..=max_change {
            for s in d..=max_state {
                let (a, b) = (
                    Self::divide(s, denominator),
                    Self::divide(s - d, denominator),
                );
                if a > b {
                    decreases[d as usize].push((s, a - b));
                }
            }
        }
        let mut row_multipliers = vec![0i32; self.num_rows as usize];
        for (i, &r) in rows.iter().enumerate() {
            row_multipliers[r as usize] = multipliers[i];
        }
        let mut states = vec![];
        compute_state_bitmap::<COVERING>(
            &mut states,
            max_state,
            &row_multipliers,
            columns,
            &self.column_indices,
        );
        let byte_sum = |byte: usize, word: u64, j: usize| {
            self.byte_value_sums[byte][((word >> (8 * j)) & 0xff) as usize]
        };
        let mut lhs = 0.0f64;
        for s in 1..=max_state {
            let coefficient = Self::divide(s, denominator) as f64;
            let mut byte = 0;
            for w in 0..nwords {
                if byte >= nbytes {
                    continue;
                }
                let word = states[s as usize][w];
                let mut j = 0;
                while j < 8 && byte < nbytes {
                    lhs = byte_sum(byte, word, j) * coefficient + lhs;
                    j += 1;
                    byte += 1;
                }
            }
        }
        let use_neighbours = !COVERING && parent.packing_distances_defined;
        let mut increased = vec![vec![0u64; nwords]; (max_rhs.max(0) + 1) as usize];
        let mut decreased = increased.clone();
        loop {
            let in_cut: Vec<usize> = (0..self.num_rows as usize)
                .filter(|&r| row_multipliers[r] > 0)
                .collect();
            if in_cut.is_empty() {
                break;
            }
            let mut best = 0.0f64;
            let (mut best_in, mut best_out) = (-1i32, -1i32);
            for &row_out in &in_cut {
                let m_out = row_multipliers[row_out];
                let mut mask: Bits = [0; 16];
                if use_neighbours {
                    mask = [u64::MAX; 16];
                    for &r in &in_cut {
                        if r != row_out {
                            for (m, n) in mask.iter_mut().zip(&parent.neighbours_208[r]) {
                                *m &= n;
                            }
                        }
                    }
                }
                for row_in in 0..self.num_rows as usize {
                    let m_in = row_multipliers[row_in];
                    if m_in == m_out || (m_in > 0 && row_out > row_in) {
                        continue;
                    }
                    if use_neighbours && !bit(&mask, row_in) {
                        continue;
                    }
                    for l in 0..=max_rhs.max(0) as usize {
                        increased[l].iter_mut().for_each(|w| *w = 0);
                        decreased[l].iter_mut().for_each(|w| *w = 0);
                    }
                    let delta = m_out - m_in;
                    let accumulate =
                        |from: usize,
                         to: usize,
                         step: i32,
                         changes: &Vec<Vec<(i32, i32)>>,
                         target: &mut Vec<Vec<u64>>| {
                            let cnt = self.row_max_count[from];
                            // An all-zero difference ORs nothing in.
                            if cnt <= 0 || !self.difference_nonzero[from * row_count + to] {
                                return;
                            }
                            for c in 1..=cnt {
                                let bits = self.difference(from, to, c as usize);
                                super::bitset::accumulate(
                                    bits,
                                    &changes[(c * step) as usize],
                                    &states,
                                    target,
                                );
                            }
                        };
                    if delta > 0 {
                        accumulate(row_out, row_in, delta, &decreases, &mut decreased);
                        accumulate(row_in, row_out, delta, &increases, &mut increased);
                    } else {
                        accumulate(row_out, row_in, -delta, &increases, &mut increased);
                        accumulate(row_in, row_out, -delta, &decreases, &mut decreased);
                    }
                    let gain = super::gain::score(
                        &self.byte_value_sums,
                        &increased,
                        &decreased,
                        nbytes,
                        max_rhs,
                    );
                    let better = if COVERING {
                        best - 1e-6 > gain
                    } else {
                        gain > 1e-6 + best
                    };
                    if better {
                        best = gain;
                        best_in = row_in as i32;
                        best_out = row_out as i32;
                    }
                }
            }
            let stop = if COVERING { best > -1e-6 } else { 1e-6 > best };
            if stop {
                break;
            }
            row_multipliers.swap(best_out as usize, best_in as usize);
            compute_state_bitmap::<COVERING>(
                &mut states,
                max_state,
                &row_multipliers,
                columns,
                &self.column_indices,
            );
            lhs = lhs + best;
        }
        rows.clear();
        multipliers.clear();
        for r in 0..self.num_rows as usize {
            if row_multipliers[r] > 0 {
                rows.push(r as i32);
                multipliers.push(row_multipliers[r]);
            }
        }
        if COVERING {
            rhs - lhs
        } else {
            lhs - rhs
        }
    }

    /// `num_iterations` randomized starts, each improved by local search.
    #[allow(clippy::too_many_arguments)]
    pub fn run(
        &mut self,
        parent: &mut Separation,
        columns: &[ColumnInfo],
        cuts: &mut CandidateSet,
        num_rows_cut: i32,
        rhs: f64,
        numerators: &[i32],
        denominator: i32,
        class: i32,
    ) {
        if self.num_rows < num_rows_cut {
            return;
        }
        let mut seen: HashSet<Vec<i32>> = HashSet::new();
        let parallel = (4..=5).contains(&num_rows_cut)
            && self.num_iterations >= 32
            && rayon::current_num_threads() > 1;
        if parallel {
            // Bound memory independently of the configured restart count. The
            // RNG is used only while building starts, in the original order.
            let batch_size = (rayon::current_num_threads() * 4).clamp(32, 256);
            let mut max_state = None;
            for first in (0..self.num_iterations as usize).step_by(batch_size) {
                let count = batch_size.min(self.num_iterations as usize - first);
                let starts: Vec<_> = (0..count)
                    .map(|_| self.build_start(parent, columns, numerators, denominator))
                    .collect();
                let max_state = *max_state.get_or_insert_with(|| {
                    self.compute_max_state(
                        columns,
                        starts[0].len(),
                        &numerators[..num_rows_cut as usize],
                    )
                });
                let results: Vec<_> = starts
                    .into_par_iter()
                    .map(|mut rows| {
                        let mut multipliers = numerators[..num_rows_cut as usize].to_vec();
                        let violation = self.run_local_search(
                            parent,
                            columns,
                            &mut rows,
                            &mut multipliers,
                            denominator,
                            rhs,
                            max_state,
                        );
                        (rows, multipliers, violation)
                    })
                    .collect();
                // Indexed collection retains restart order, including across
                // batches. This preserves duplicate winners and score ties.
                for (rows, multipliers, violation) in results {
                    self.insert_result(
                        parent,
                        columns,
                        cuts,
                        &mut seen,
                        rows,
                        multipliers,
                        violation,
                        num_rows_cut,
                        rhs,
                        denominator,
                        class,
                    );
                }
            }
            return;
        }
        // The bound depends on the column counts and multiplier multiset,
        // which stay fixed across restarts of this class. Initialize lazily
        // so a search with no starts still performs no bound calculation.
        let mut max_state = None;
        for _ in 0..self.num_iterations {
            let mut rows = self.build_start(parent, columns, numerators, denominator);
            let mut multipliers: Vec<i32> = numerators[..num_rows_cut as usize].to_vec();
            let max_state = *max_state
                .get_or_insert_with(|| self.compute_max_state(columns, rows.len(), &multipliers));
            let violation = self.run_local_search(
                parent,
                columns,
                &mut rows,
                &mut multipliers,
                denominator,
                rhs,
                max_state,
            );
            self.insert_result(
                parent,
                columns,
                cuts,
                &mut seen,
                rows,
                multipliers,
                violation,
                num_rows_cut,
                rhs,
                denominator,
                class,
            );
        }
    }

    fn build_start(
        &mut self,
        parent: &Separation,
        columns: &[ColumnInfo],
        numerators: &[i32],
        denominator: i32,
    ) -> Vec<i32> {
        let mut rows = Vec::new();
        if COVERING {
            self.build_covering_row_set(columns, &mut rows, numerators, 0.5);
        } else {
            self.build_packing_row_set(parent, columns, &mut rows, numerators, denominator, 0.5);
        }
        rows
    }

    /// Apply deduplication, sub-cut checks and IDs in restart order. Workers
    /// never mutate the candidate tree or the separator's arena.
    #[allow(clippy::too_many_arguments)]
    fn insert_result(
        &self,
        parent: &mut Separation,
        columns: &[ColumnInfo],
        cuts: &mut CandidateSet,
        seen: &mut HashSet<Vec<i32>>,
        rows: Vec<i32>,
        mut multipliers: Vec<i32>,
        violation: f64,
        num_rows_cut: i32,
        rhs: f64,
        denominator: i32,
        class: i32,
    ) {
        let mut key = rows.clone();
        key.sort();
        if seen.contains(&key) {
            return;
        }
        let ratio = violation / rhs;
        if !(ratio > self.violation_tolerance) {
            return;
        }
        if num_rows_cut == 4 {
            if !self.skip_subcut_checks {
                let violated = if COVERING {
                    subcuts::three_rows_cov_violated(
                        parent,
                        columns,
                        &rows,
                        &parent.permutations_4rows,
                    )
                } else {
                    subcuts::three_rows_pack_violated(
                        parent,
                        columns,
                        &rows,
                        &parent.permutations_4rows,
                    )
                };
                if violated {
                    return;
                }
            }
        } else if num_rows_cut == 5 && !self.skip_subcut_checks {
            let violated = if COVERING {
                subcuts::three_rows_cov_violated(parent, columns, &rows, &parent.permutations_5rows)
                    || subcuts::four_rows_cov_violated(parent, columns, &rows, &self.column_indices)
            } else {
                subcuts::three_rows_pack_violated(
                    parent,
                    columns,
                    &rows,
                    &parent.permutations_5rows,
                ) || subcuts::four_rows_pack_violated(parent, columns, &rows, &self.column_indices)
            };
            if violated {
                return;
            }
        }
        if COVERING {
            for m in multipliers.iter_mut().take(num_rows_cut as usize) {
                *m = -*m;
            }
        }
        let id = parent.round_cut_counter;
        parent.round_cut_counter += 1;
        let cut = parent.push_candidate(NativeCut {
            id,
            rhs,
            kind: COVERING as i32,
            num_rows: num_rows_cut,
            class,
            denominator: if COVERING { -denominator } else { denominator },
            numerators: multipliers,
            set_ids: rows,
            memory_type: self.cut_memory_type,
            memory: BTreeMap::new(),
        });
        parent.insert_candidate(cuts, (0.5 - 1e8 * ratio).floor() / 1e8, cut);
        seen.insert(key);
    }
}
