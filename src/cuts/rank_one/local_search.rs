//! Local-search rank-one separation (packing and covering), from the C++
//! reference `native/separation/LocalSearchRankOneSeparation.h`. Packing row
//! sets use the separator's own std::minstd_rand0 (seed 1); covering row sets
//! use the C library rand().
use crate::cuts::shared::rbtree::CandidateSet;
use super::separation::{bit, Bits, ColumnInfo, NativeCut, Separation};
use super::subcuts;
use std::collections::{BTreeMap, HashSet};

extern "C" {
    fn rand() -> std::os::raw::c_int;
}

/// std::minstd_rand0 with libstdc++'s unsigned long state.
#[derive(Clone, Debug)]
struct MinStd(u64);
impl MinStd {
    fn next(&mut self) -> u64 {
        self.0 = self.0 * 16807 % 2147483647;
        self.0
    }
    /// libstdc++ uniform_int_distribution<unsigned long> on this engine.
    fn uniform(&mut self, a: u64, b: u64) -> u64 {
        const URNGMIN: u64 = 1;
        const URNGRANGE: u64 = 2147483645;
        let urange = b - a;
        let ret;
        if URNGRANGE > urange {
            let uerange = urange + 1;
            let scaling = URNGRANGE / uerange;
            let past = uerange * scaling;
            let mut r;
            loop {
                r = self.next() - URNGMIN;
                if r < past {
                    break;
                }
            }
            ret = r / scaling;
        } else if URNGRANGE < urange {
            let mut r;
            loop {
                let uerngrange = URNGRANGE + 1;
                let tmp = uerngrange * self.uniform(0, urange / uerngrange);
                r = tmp + (self.next() - URNGMIN);
                if !(r > urange || r < tmp) {
                    break;
                }
            }
            ret = r;
        } else {
            ret = self.next() - URNGMIN;
        }
        ret + a
    }
}

/// Per state (weighted row count), the bitset of columns in that state.
fn compute_state_bitmap<const COVERING: bool>(
    bitmaps: &mut Vec<Vec<u64>>,
    max_state: i32,
    row_multipliers: &[i32],
    columns: &[ColumnInfo],
    column_indices: &[i32],
) {
    let n = column_indices.len();
    let nbytes = n / 8 + (n & 7 != 0) as usize;
    let nwords = nbytes / 8 + (nbytes & 7 != 0) as usize;
    bitmaps.clear();
    bitmaps.resize((max_state + 1) as usize, vec![0; nwords]);
    for k in 0..n {
        let col = &columns[column_indices[k] as usize];
        let count = if COVERING {
            &col.covering_set_count
        } else {
            &col.packing_set_count
        };
        let mut state: u32 = 0;
        for r in 0..row_multipliers.len() {
            state = state.wrapping_add((count[r] as u32).wrapping_mul(row_multipliers[r] as u32));
        }
        bitmaps[state as i32 as usize][k >> 6] |= 1u64 << (k & 63);
    }
}

pub struct LocalSearch<const COVERING: bool> {
    column_indices: Vec<i32>,
    num_rows: i32,
    violation_tolerance: f64,
    cut_memory_type: i32,
    skip_subcut_checks: bool,
    num_iterations: i32,
    byte_value_sums: Vec<Vec<f64>>,
    row_values: Vec<f64>,
    /// Columns whose count in row i exceeds that in row j by c, as nwords
    /// words at ((i * rows + j) * (max_count + 1) + c) * nwords.
    count_difference_bits: Vec<u64>,
    /// Whether any bit of row pair (i, j) is set.
    difference_nonzero: Vec<bool>,
    nwords: usize,
    max_count: i32,
    row_max_count: Vec<i32>,
    rng: MinStd,
}

impl<const COVERING: bool> LocalSearch<COVERING> {
    fn counts(c: &ColumnInfo) -> &Vec<i32> {
        if COVERING {
            &c.covering_set_count
        } else {
            &c.packing_set_count
        }
    }
    /// The count-difference bitset of rows (from, to) at difference c.
    #[inline]
    fn difference(&self, from: usize, to: usize, c: usize) -> &[u64] {
        let rows = self.num_rows.max(0) as usize;
        let slots = (self.max_count.max(0) + 1) as usize;
        let start = ((from * rows + to) * slots + c) * self.nwords;
        &self.count_difference_bits[start..start + self.nwords]
    }
    fn divide(a: i32, d: i32) -> i32 {
        if COVERING {
            a / d + (a % d != 0) as i32
        } else {
            a / d
        }
    }
    /// Per-byte value sums, row values and count-difference bitsets.
    pub fn new(parent: &Separation, cols: &[ColumnInfo], elementary: &[i32]) -> Self {
        let num_rows = if COVERING {
            parent.num_covering_sets
        } else {
            parent.num_packing_sets
        };
        let column_indices: Vec<i32> = if COVERING {
            (0..cols.len() as i32).collect()
        } else {
            elementary.to_vec()
        };
        let n = column_indices.len();
        let nbytes = n / 8 + (n & 7 != 0) as usize;
        let nwords = nbytes / 8 + (nbytes & 7 != 0) as usize;
        let mut byte_value_sums = vec![vec![0.0f64; 256]; nbytes];
        for g in 0..nbytes {
            for mask in 0..256usize {
                let mut v = 0.0;
                for b in 0..8 {
                    if n <= 8 * g + b {
                        break;
                    }
                    if mask & (1 << b) != 0 {
                        v += cols[column_indices[8 * g + b] as usize].value;
                    }
                }
                byte_value_sums[g][mask] = v;
            }
        }
        let rows = num_rows.max(0) as usize;
        let mut row_values = vec![0.0f64; rows];
        if num_rows > 0 && !column_indices.is_empty() {
            for (i, value) in row_values.iter_mut().enumerate() {
                for &k in &column_indices {
                    let c = &cols[k as usize];
                    if Self::counts(c)[i] != 0 {
                        *value += c.value;
                    }
                }
            }
        }
        let mut row_max_count = vec![0; rows];
        let mut max_count = 0;
        for (i, m) in row_max_count.iter_mut().enumerate() {
            for &k in &column_indices {
                let c = Self::counts(&cols[k as usize])[i];
                if *m < c {
                    *m = c;
                }
            }
            if max_count < *m {
                max_count = *m;
            }
        }
        let slots = (max_count.max(0) + 1) as usize;
        let mut count_difference_bits = vec![0u64; rows * rows * slots * nwords];
        let mut difference_nonzero = vec![false; rows * rows];
        for k in 0..n {
            let cnt = Self::counts(&cols[column_indices[k] as usize]);
            for i in 0..rows {
                for j in 0..rows {
                    let (ci, cj) = (cnt[i], cnt[j]);
                    if i != j && cj < ci {
                        let pair = i * rows + j;
                        count_difference_bits[(pair * slots + (ci - cj) as usize) * nwords + (k >> 6)] |=
                            1u64 << (k & 63);
                        difference_nonzero[pair] = true;
                    }
                }
            }
        }
        Self {
            column_indices,
            num_rows,
            violation_tolerance: 0.02,
            cut_memory_type: parent.cut_memory_type,
            skip_subcut_checks: parent.params.skip_subcut_checks,
            num_iterations: parent.params.ls_num_iterations,
            byte_value_sums,
            row_values,
            count_difference_bits,
            difference_nonzero,
            nwords,
            max_count,
            row_max_count,
            rng: MinStd(1),
        }
    }
    /// Greedy randomized packing row set (alpha-restricted candidates).
    fn build_packing_row_set(
        &mut self,
        parent: &Separation,
        columns: &[ColumnInfo],
        row_set: &mut Vec<i32>,
        numerators: &[i32],
        denominator: i32,
        alpha: f64,
    ) {
        let use_neighbours = parent.packing_distances_defined;
        let mut accumulated = vec![0i32; self.column_indices.len()];
        let mut values = self.row_values.clone();
        let mut candidates: Vec<i32> = Vec::with_capacity(self.num_rows.max(0) as usize);
        let mut mask: Bits = [0; 16];
        if use_neighbours {
            mask = [u64::MAX; 16];
        }
        for t in 0..numerators.len() {
            candidates.clear();
            let collect = |masked: bool, candidates: &mut Vec<i32>, values: &[f64], mask: &Bits| {
                let (mut min_value, mut max_value) = (f64::MAX, f64::MIN_POSITIVE);
                for r in 0..self.num_rows as usize {
                    let v = values[r];
                    if v >= -1e-6 && (!masked || bit(mask, r)) {
                        max_value = if v > max_value { v } else { max_value };
                        min_value = if v < min_value { v } else { min_value };
                    }
                }
                let threshold = max_value - (max_value - min_value) * alpha - 1e-6;
                for r in 0..self.num_rows as usize {
                    if values[r] >= threshold && (!masked || bit(mask, r)) {
                        candidates.push(r as i32);
                    }
                }
            };
            collect(use_neighbours, &mut candidates, &values, &mask);
            if candidates.is_empty() {
                collect(false, &mut candidates, &values, &mask);
            }
            if candidates.is_empty() {
                eprintln!("RCSP error: construction of rank-1 cut rowset is buggy! {}", use_neighbours as i32);
                std::process::exit(1);
            }
            let row = candidates[self.rng.uniform(0, candidates.len() as u64 - 1) as usize];
            row_set.push(row);
            values[row as usize] = -1.0;
            if use_neighbours {
                for (m, n) in mask.iter_mut().zip(&parent.neighbours_208[row as usize]) {
                    *m &= n;
                }
            }
            for k in 0..self.column_indices.len() {
                let col = &columns[self.column_indices[k] as usize];
                let cnt = col.packing_set_count[row as usize];
                if cnt <= 0 {
                    continue;
                }
                let a = numerators[t] * cnt + accumulated[k];
                accumulated[k] = a;
                if a >= denominator && a - cnt * numerators[t] < denominator {
                    for &s in &col.packing_set_sequence {
                        values[s as usize] -= col.value;
                    }
                }
            }
        }
    }
    /// Greedy randomized covering row set.
    fn build_covering_row_set(&mut self, columns: &[ColumnInfo], row_set: &mut Vec<i32>, numerators: &[i32], alpha: f64) {
        let mut accumulated = vec![0i32; self.column_indices.len()];
        let mut values = self.row_values.clone();
        let mut candidates: Vec<i32> = vec![];
        for t in 0..numerators.len() {
            candidates.clear();
            let (mut min_value, mut max_value) = (f64::MAX, f64::MIN_POSITIVE);
            for r in 0..self.num_rows as usize {
                let v = values[r];
                if v >= -1e-6 {
                    max_value = if v > max_value { v } else { max_value };
                    min_value = if v < min_value { v } else { min_value };
                }
            }
            let threshold = (max_value - min_value) * alpha + min_value - 1e-6;
            for r in 0..self.num_rows as usize {
                if values[r] >= threshold {
                    candidates.push(r as i32);
                }
            }
            if candidates.is_empty() {
                eprintln!("RCSP error: construction of rank-1 cut rowset is buggy! ");
                std::process::exit(1);
            }
            let rnd = unsafe { rand() } as i64 as u64;
            let row = candidates[(rnd % candidates.len() as u64) as usize];
            row_set.push(row);
            values[row as usize] = -1.0;
            for k in 0..self.column_indices.len() {
                let col = &columns[self.column_indices[k] as usize];
                let cnt = &col.covering_set_count;
                if cnt[row as usize] == 0 {
                    continue;
                }
                let a = accumulated[k];
                if a == 0 && self.num_rows > 0 {
                    for r in 0..self.num_rows as usize {
                        if cnt[r] != 0 {
                            values[r] -= col.value;
                        }
                    }
                }
                accumulated[k] = a + numerators[t];
            }
        }
    }
    /// Largest weighted row count over the columns.
    fn compute_max_state(&self, columns: &[ColumnInfo], k: usize, numerators: &[i32]) -> i32 {
        let mut best = 0;
        let mut m = numerators.to_vec();
        m.sort();
        for &c in &self.column_indices {
            let mut top = vec![0i32; k];
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
    fn run_local_search(
        &self,
        parent: &Separation,
        columns: &[ColumnInfo],
        rows: &mut Vec<i32>,
        multipliers: &mut Vec<i32>,
        denominator: i32,
        rhs: f64,
    ) -> f64 {
        let max_state = self.compute_max_state(columns, rows.len(), multipliers);
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
                let (a, b) = (Self::divide(s, denominator), Self::divide(d + s, denominator));
                if a < b {
                    increases[d as usize].push((s, b - a));
                }
            }
        }
        let mut decreases: Vec<Vec<(i32, i32)>> = vec![vec![]; (max_change + 1) as usize];
        for d in 1..=max_change {
            for s in d..=max_state {
                let (a, b) = (Self::divide(s, denominator), Self::divide(s - d, denominator));
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
        compute_state_bitmap::<COVERING>(&mut states, max_state, &row_multipliers, columns, &self.column_indices);
        let byte_sum = |byte: usize, word: u64, j: usize| self.byte_value_sums[byte][((word >> (8 * j)) & 0xff) as usize];
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
            let in_cut: Vec<usize> = (0..self.num_rows as usize).filter(|&r| row_multipliers[r] > 0).collect();
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
                    let accumulate = |from: usize,
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
                            for &(s, l) in &changes[(c * step) as usize] {
                                let st = &states[s as usize][..nwords];
                                let tg = &mut target[l as usize][..nwords];
                                for ((t, &b), &s) in tg.iter_mut().zip(bits).zip(st) {
                                    *t |= b & s;
                                }
                            }
                        }
                    };
                    if delta > 0 {
                        accumulate(row_out, row_in, delta, &decreases, &mut decreased);
                        accumulate(row_in, row_out, delta, &increases, &mut increased);
                    } else {
                        accumulate(row_out, row_in, -delta, &increases, &mut increased);
                        accumulate(row_in, row_out, -delta, &decreases, &mut decreased);
                    }
                    let mut gain = 0.0f64;
                    let mut byte = 0;
                    for w in 0..nwords {
                        if byte >= nbytes {
                            continue;
                        }
                        let mut j = 0;
                        while j < 8 && byte < nbytes {
                            if max_rhs > 0 {
                                for l in 1..=max_rhs as usize {
                                    let ld = l as f64;
                                    let x1 = byte_sum(byte, increased[l][w], j) * ld;
                                    let x2 = ld * byte_sum(byte, decreased[l][w], j);
                                    gain = gain + x1;
                                    gain = gain - x2;
                                }
                            }
                            j += 1;
                            byte += 1;
                        }
                    }
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
            compute_state_bitmap::<COVERING>(&mut states, max_state, &row_multipliers, columns, &self.column_indices);
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
        for _ in 0..self.num_iterations {
            let mut rows = vec![];
            if COVERING {
                self.build_covering_row_set(columns, &mut rows, numerators, 0.5);
            } else {
                self.build_packing_row_set(parent, columns, &mut rows, numerators, denominator, 0.5);
            }
            let mut multipliers: Vec<i32> = numerators[..num_rows_cut as usize].to_vec();
            let violation = self.run_local_search(parent, columns, &mut rows, &mut multipliers, denominator, rhs);
            let mut key = rows.clone();
            key.sort();
            if seen.contains(&key) {
                continue;
            }
            let ratio = violation / rhs;
            if !(ratio > self.violation_tolerance) {
                continue;
            }
            if num_rows_cut == 4 {
                if !self.skip_subcut_checks {
                    let violated = if COVERING {
                        subcuts::three_rows_cov_violated(parent, columns, &rows, &parent.permutations_4rows)
                    } else {
                        subcuts::three_rows_pack_violated(parent, columns, &rows, &parent.permutations_4rows)
                    };
                    if violated {
                        continue;
                    }
                }
            } else if num_rows_cut == 5 && !self.skip_subcut_checks {
                let violated = if COVERING {
                    subcuts::three_rows_cov_violated(parent, columns, &rows, &parent.permutations_5rows)
                        || subcuts::four_rows_cov_violated(parent, columns, &rows, &self.column_indices)
                } else {
                    subcuts::three_rows_pack_violated(parent, columns, &rows, &parent.permutations_5rows)
                        || subcuts::four_rows_pack_violated(parent, columns, &rows, &self.column_indices)
                };
                if violated {
                    continue;
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
}
