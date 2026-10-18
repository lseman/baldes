//! Tuple-based 4- and 5-row rank-one separation, from the C++ reference `native/separation/TupleRankOneSeparation.h`. Its hash
//! sets are used for membership only; `std::sort` ties follow libstdc++.
use crate::cuts::shared::rbtree::CandidateSet;
use super::separation::{ColumnInfo, NativeCut, Separation};
use crate::cuts::shared::stdsort;
use std::collections::{BTreeMap, HashMap, HashSet};

/// A row tuple and its value; ordering is by value descending.
#[derive(Clone, Debug)]
struct Tuple {
    ids: Vec<i32>,
    value: f64,
}
fn maxsd(a: f64, b: f64) -> f64 {
    if a > b {
        a
    } else {
        b
    }
}
fn violation_key(violation: f64) -> f64 {
    (0.5 - violation * 100000000.0).floor() / 100000000.0
}

pub struct TupleSeparation {
    column_ids: Vec<i32>,
    num_packing_sets: i32,
    violation_tolerance: f64,
    max_num_triplets: i32,
    cut_memory_type: i32,
    skip_subcut_checks: bool,
    three_row_subsets: Vec<Vec<i32>>,
    four_row_subsets: Vec<Vec<i32>>,
    one_row_value: Vec<f64>,
    two_row_value: Vec<Vec<f64>>,
    selected: Vec<Tuple>,
    unviolated: Vec<Tuple>,
    triplets: HashMap<Vec<i32>, f64>,
    quadruples: HashMap<Vec<i32>, f64>,
    quintuples: HashMap<Vec<i32>, f64>,
    /// Columns of the current separate call (all of them).
    columns: Vec<ColumnInfo>,
}

impl TupleSeparation {
    /// One- and two-row sums, triplets, quadruples and quintuples.
    pub fn new(parent: &Separation, columns: &[ColumnInfo], column_ids: &[i32]) -> Self {
        let n = parent.num_packing_sets;
        let mut s = Self {
            column_ids: column_ids.to_vec(),
            num_packing_sets: n,
            violation_tolerance: parent.params.violation_tolerance,
            max_num_triplets: parent.params.max_num_triplets,
            cut_memory_type: parent.cut_memory_type,
            skip_subcut_checks: parent.params.skip_subcut_checks,
            three_row_subsets: vec![],
            four_row_subsets: vec![],
            one_row_value: vec![0.0; (n + 1) as usize],
            two_row_value: vec![vec![0.0; (n + 1) as usize]; (n + 1) as usize],
            selected: vec![],
            unviolated: vec![],
            triplets: HashMap::new(),
            quadruples: HashMap::new(),
            quintuples: HashMap::new(),
            columns: columns.to_vec(),
        };
        if !s.skip_subcut_checks {
            s.three_row_subsets = parent.permutations_5rows.clone();
            s.four_row_subsets = parent.permutations_190.clone();
        }
        let num_columns = column_ids.len();
        if n > 0 && num_columns != 0 {
            for i in 0..n as usize {
                for &idx in column_ids {
                    let col = &columns[idx as usize];
                    let cnt = &col.packing_set_count;
                    if cnt[i] == 0 {
                        continue;
                    }
                    s.one_row_value[i] += col.value;
                    for j in i + 1..n as usize {
                        if cnt[j] == 0 {
                            continue;
                        }
                        s.two_row_value[i][j] += col.value;
                        s.two_row_value[j][i] += col.value;
                    }
                }
            }
        }
        let mut all = vec![];
        for i in 0..n {
            for j in i + 1..n {
                if 0.0 >= s.two_row_value[i as usize][j as usize] {
                    continue;
                }
                for k in j + 1..n {
                    if num_columns == 0 {
                        continue;
                    }
                    let (mut at_least_two, mut all_three) = (0.0f64, 0.0f64);
                    for &idx in column_ids {
                        let col = &columns[idx as usize];
                        let cnt = &col.packing_set_count;
                        let c = cnt[j as usize] + cnt[i as usize] + cnt[k as usize];
                        if c > 1 {
                            at_least_two += col.value;
                            if c == 3 {
                                all_three += col.value;
                            }
                        }
                    }
                    if !(all_three > 0.0) {
                        continue;
                    }
                    let t = Tuple {
                        ids: vec![i, j, k],
                        value: all_three,
                    };
                    s.triplets.entry(t.ids.clone()).or_insert(all_three);
                    all.push(t.clone());
                    if s.violation_tolerance >= at_least_two - 1.0 {
                        s.unviolated.push(t);
                    }
                }
            }
        }
        stdsort::sort(&mut s.unviolated, |a, b| a.value > b.value);
        if parent.params.max_num_rows > 4 {
            for t in &all {
                let (a, b, c) = (t.ids[0] as usize, t.ids[1] as usize, t.ids[2] as usize);
                for l in t.ids[2] + 1..n {
                    if num_columns == 0 {
                        continue;
                    }
                    let mut value = 0.0f64;
                    let mut visiting = vec![];
                    for q in 0..num_columns {
                        let cnt = &columns[column_ids[q] as usize].packing_set_count;
                        if cnt[a] != 0 && cnt[b] != 0 && cnt[c] != 0 && cnt[l as usize] != 0 {
                            value += columns[column_ids[q] as usize].value;
                            visiting.push(q);
                        }
                    }
                    if !(value > 0.0) {
                        continue;
                    }
                    s.quadruples
                        .entry(vec![a as i32, b as i32, c as i32, l])
                        .or_insert(value);
                    for m in l + 1..n {
                        let mut five = value;
                        for &q in &visiting {
                            let col = &columns[column_ids[q] as usize];
                            if col.packing_set_count[m as usize] == 0 {
                                five -= col.value;
                            }
                        }
                        if five > 0.0 {
                            s.quintuples
                                .entry(vec![a as i32, b as i32, c as i32, l, m])
                                .or_insert(five);
                        }
                    }
                }
            }
        }
        s
    }
    fn lookup(set: &HashMap<Vec<i32>, f64>, mut ids: Vec<i32>, sort_ids: bool) -> f64 {
        if sort_ids {
            ids.sort();
        }
        set.get(&ids).copied().unwrap_or(0.0)
    }
    fn get3(&self, a: i32, b: i32, c: i32, sort_ids: bool) -> f64 {
        Self::lookup(&self.triplets, vec![a, b, c], sort_ids)
    }
    fn get4(&self, a: i32, b: i32, c: i32, d: i32) -> f64 {
        Self::lookup(&self.quadruples, vec![a, b, c, d], false)
    }
    fn get5(&self, ids: &[i32]) -> f64 {
        Self::lookup(&self.quintuples, ids.to_vec(), false)
    }
    /// Whether a 3-row sub-cut is violated (all columns).
    fn three_rows_violated(&self, ids: &[i32]) -> bool {
        let mut sums = vec![0.0f64; self.three_row_subsets.len()];
        for col in &self.columns {
            let cnt = &col.packing_set_count;
            for (t, p) in self.three_row_subsets.iter().enumerate() {
                let c = (cnt[ids[p[0] as usize] as usize] != 0) as i32
                    + (cnt[ids[p[1] as usize] as usize] != 0) as i32
                    + (cnt[ids[p[2] as usize] as usize] != 0) as i32;
                if c > 1 {
                    sums[t] += col.value;
                    if sums[t] - 1.0 > self.violation_tolerance {
                        return true;
                    }
                }
            }
        }
        false
    }
    /// Whether a 4-row sub-cut is violated (listed columns).
    fn four_rows_violated(&self, ids: &[i32]) -> bool {
        let mut sums = vec![0.0f64; self.four_row_subsets.len()];
        for &idx in &self.column_ids {
            let cnt = &self.columns[idx as usize].packing_set_count;
            let value = self.columns[idx as usize].value;
            for (t, p) in self.four_row_subsets.iter().enumerate() {
                let c = if cnt[ids[p[0] as usize] as usize] != 0 { 2 } else { 0 }
                    + (cnt[ids[p[1] as usize] as usize] != 0) as i32
                    + (cnt[ids[p[2] as usize] as usize] != 0) as i32
                    + (cnt[ids[p[3] as usize] as usize] != 0) as i32;
                if c > 2 {
                    sums[t] += value;
                    if sums[t] - 1.0 > self.violation_tolerance {
                        return true;
                    }
                }
            }
        }
        false
    }
    /// Selects the first `max_num_triplets` unviolated triplets.
    fn prepare_four_rows(&mut self) {
        self.selected = self
            .unviolated
            .iter()
            .take(self.max_num_triplets.max(0) as usize)
            .cloned()
            .collect();
    }
    /// Whether row `l` can extend triplet `t`.
    fn is_neighbor(&self, t: &Tuple, l: i32) -> bool {
        let limit = 1.0 + self.violation_tolerance;
        let ll = l as usize;
        if limit > self.one_row_value[ll] + t.value {
            return false;
        }
        let (a, b, c) = (t.ids[0], t.ids[1], t.ids[2]);
        let (aa, bb, cc) = (a as usize, b as usize, c as usize);
        let pa = self.two_row_value[aa][ll];
        if pa == 0.0 {
            return false;
        }
        let pb = self.two_row_value[bb][ll];
        if pb == 0.0 {
            return false;
        }
        let pc = self.two_row_value[cc][ll];
        if pc == 0.0 {
            return false;
        }
        if (pa + pb) + self.two_row_value[aa][bb] - 2.0 * self.get3(a, b, l, true) > limit {
            return false;
        }
        if (pb + pc) + self.two_row_value[bb][cc] - 2.0 * self.get3(b, c, l, true) > limit {
            return false;
        }
        !((pa + pc) + self.two_row_value[aa][cc] - 2.0 * self.get3(a, c, l, true) > limit)
    }
    /// Violation of the 1/2 cut on five rows, from stored tuple values.
    fn five_rows_type0(&self, ids: &[i32]) -> f64 {
        let p = |x: usize, y: usize| self.two_row_value[ids[x] as usize][ids[y] as usize];
        let mut pairs = 0.0;
        for (x, y) in [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)] {
            pairs += p(x, y);
        }
        let excess = pairs - 2.0;
        if !(excess > 0.0) {
            return 0.0;
        }
        let mut v = excess + 2.0;
        if !self.three_row_subsets.is_empty() {
            let mut t3 = 0.0;
            for s in &self.three_row_subsets {
                t3 = t3 + self.get3(ids[s[0] as usize], ids[s[1] as usize], ids[s[2] as usize], false);
            }
            v = v - (t3 + t3);
        }
        let mut q = 0.0;
        q = q + self.get4(ids[1], ids[2], ids[3], ids[4]);
        q = q + self.get4(ids[0], ids[2], ids[3], ids[4]);
        q = q + self.get4(ids[0], ids[1], ids[3], ids[4]);
        q = q + self.get4(ids[0], ids[1], ids[2], ids[4]);
        let r = (q + self.get4(ids[0], ids[1], ids[2], ids[3])) * 4.0 + v;
        let f = self.get5(ids);
        maxsd((r - f * 8.0) - 2.0, 0.0)
    }
    /// Violation of the 1/3 cut on five rows, from stored tuple values.
    fn five_rows_type1(&self, ids: &[i32]) -> f64 {
        if self.three_row_subsets.is_empty() {
            return 0.0;
        }
        let mut t = 0.0;
        for s in &self.three_row_subsets {
            t = t + self.get3(ids[s[0] as usize], ids[s[1] as usize], ids[s[2] as usize], false);
        }
        let excess = t - 1.0;
        if !(excess > 0.0) {
            return 0.0;
        }
        let mut v = excess + 1.0;
        v = v - 3.0 * self.get4(ids[1], ids[2], ids[3], ids[4]);
        v = v - 3.0 * self.get4(ids[0], ids[2], ids[3], ids[4]);
        v = v - 3.0 * self.get4(ids[0], ids[1], ids[3], ids[4]);
        v = v - 3.0 * self.get4(ids[0], ids[1], ids[2], ids[4]);
        v = v - 3.0 * self.get4(ids[0], ids[1], ids[2], ids[3]);
        let f = self.get5(ids);
        maxsd((f * 6.0 + v) - 1.0, 0.0)
    }
    /// Four-row (1,1,1,2)/3 cuts extending the selected triplets.
    pub fn four_rows_pack(&mut self, parent: &mut Separation, cuts: &mut CandidateSet) {
        self.prepare_four_rows();
        let mut generated = HashSet::new();
        for t in self.selected.clone() {
            let mut best_row = -1;
            let mut best_violation = 0.0f64;
            for l in 0..self.num_packing_sets {
                if !self.is_neighbor(&t, l) {
                    continue;
                }
                let (a, b, c) = (t.ids[0], t.ids[1], t.ids[2]);
                let pl = &self.two_row_value[l as usize];
                let mut w = t.value;
                w += pl[a as usize];
                w += pl[b as usize];
                w += pl[c as usize];
                let mut x = -self.get3(l, b, c, true);
                x = x - self.get3(a, l, c, true);
                x = x - self.get3(a, b, l, true);
                let violation = maxsd((x + w) - 1.0, 0.0);
                if violation > best_violation {
                    best_violation = violation;
                    best_row = l;
                }
            }
            if best_row == -1 || !(best_violation > self.violation_tolerance) {
                continue;
            }
            let mut rows = t.ids.clone();
            rows.push(best_row);
            let mut key = rows.clone();
            key.sort();
            if generated.contains(&key) {
                continue;
            }
            let id = parent.round_cut_counter;
            parent.round_cut_counter += 1;
            let cut = parent.push_candidate(NativeCut {
                id,
                rhs: 1.0,
                kind: 0,
                num_rows: 4,
                class: -1,
                denominator: 3,
                numerators: vec![1, 1, 1, 2],
                set_ids: rows,
                memory_type: self.cut_memory_type,
                memory: BTreeMap::new(),
            });
            parent.insert_candidate(cuts, violation_key(best_violation), cut);
            generated.insert(key);
        }
    }
    /// Rescores and selects triplets for type-0 five-row cuts; rescoring is
    /// written back into the unviolated triplets.
    fn prepare_five_rows_type0(&mut self) {
        let mut candidates = vec![];
        for t in &mut self.unviolated {
            let (a, b, c) = (t.ids[0] as usize, t.ids[1] as usize, t.ids[2] as usize);
            let mut score = self.two_row_value[a][b];
            score += self.two_row_value[a][c];
            score += self.two_row_value[b][c];
            score -= t.value + t.value;
            if score > 0.0 {
                t.value = score;
                candidates.push(t.clone());
            }
        }
        stdsort::sort(&mut candidates, |a, b| a.value > b.value);
        candidates.truncate(self.max_num_triplets.max(0) as usize);
        self.selected = candidates;
    }
    /// Five-row cuts from pairs of selected triplets sharing one row.
    pub fn five_rows_pack(
        &mut self,
        parent: &mut Separation,
        cuts: &mut CandidateSet,
        rhs: f64,
        numerators: &[i32],
        denominator: i32,
        kind: i32,
    ) {
        match kind {
            0 => self.prepare_five_rows_type0(),
            1 => self.prepare_four_rows(),
            _ => return,
        }
        let mut generated = HashSet::new();
        let selected = self.selected.clone();
        for (i, t1) in selected.iter().enumerate() {
            for t2 in &selected[i + 1..] {
                let common: usize = t1
                    .ids
                    .iter()
                    .map(|x| t2.ids.iter().filter(|y| *y == x).count())
                    .sum();
                if common != 1 {
                    continue;
                }
                let mut rows = t1.ids.clone();
                for &y in &t2.ids {
                    if !rows.contains(&y) {
                        rows.push(y);
                    }
                }
                let mut key = rows.clone();
                key.sort();
                if generated.contains(&key) {
                    continue;
                }
                let raw = if kind == 0 {
                    self.five_rows_type0(&key)
                } else {
                    self.five_rows_type1(&key)
                };
                let violation = raw / rhs;
                if !(violation > self.violation_tolerance) {
                    continue;
                }
                if !self.skip_subcut_checks && (self.three_rows_violated(&rows) || self.four_rows_violated(&rows)) {
                    continue;
                }
                let id = parent.round_cut_counter;
                parent.round_cut_counter += 1;
                let cut = parent.push_candidate(NativeCut {
                    id,
                    rhs,
                    kind: 0,
                    num_rows: 5,
                    class: kind,
                    denominator,
                    numerators: numerators.to_vec(),
                    set_ids: rows,
                    memory_type: self.cut_memory_type,
                    memory: BTreeMap::new(),
                });
                parent.insert_candidate(cuts, violation_key(violation), cut);
                generated.insert(key);
            }
        }
    }
}
