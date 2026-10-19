use super::*;

impl<const COVERING: bool> LocalSearch<COVERING> {
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
                        count_difference_bits
                            [(pair * slots + (ci - cj) as usize) * nwords + (k >> 6)] |=
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
    pub(super) fn build_packing_row_set(
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
                eprintln!(
                    "RCSP error: construction of rank-1 cut rowset is buggy! {}",
                    use_neighbours as i32
                );
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
    pub(super) fn build_covering_row_set(
        &mut self,
        columns: &[ColumnInfo],
        row_set: &mut Vec<i32>,
        numerators: &[i32],
        alpha: f64,
    ) {
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
}
