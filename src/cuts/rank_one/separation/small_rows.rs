use super::*;

impl Separation {
    /// One-row packing cuts.
    pub(super) fn one_row_pack(&mut self, columns: &[ColumnInfo], cuts: &mut CandidateSet) {
        let cov = self.params.use_cov_sets_for_one_row_packing;
        let n = if cov {
            self.num_covering_sets
        } else {
            self.num_packing_sets
        };
        let kind = if cov { 2 } else { 0 };
        for set in 0..n {
            let mut lhs = 0.0;
            for col in columns {
                let c = if cov {
                    col.covering_set_count[set as usize]
                } else {
                    col.packing_set_count[set as usize]
                };
                if c > 1 {
                    lhs = ((c >> 1) as f64) * col.value + lhs;
                }
            }
            if !(lhs > self.params.violation_tolerance) {
                continue;
            }
            let cut = self.candidate(0.0, kind, 1, -1, 2, vec![1], vec![set]);
            self.insert_candidate(cuts, rounded_score(0.5 - lhs * 1e8), cut);
        }
    }

    /// One-row covering cuts.
    pub(super) fn one_row_cov(&mut self, columns: &[ColumnInfo], cuts: &mut CandidateSet) {
        for set in 0..self.num_covering_sets {
            let mut lhs = 0.0;
            for col in columns {
                let c = col.covering_set_count[set as usize];
                if c > 0 {
                    lhs = (((c + 1) >> 1) as f64) * col.value + lhs;
                }
            }
            if !(1.0 - lhs > self.params.violation_tolerance) {
                continue;
            }
            let cut = self.candidate(1.0, 1, 1, -1, -2, vec![-1], vec![set]);
            self.insert_candidate(cuts, rounded_score((lhs - 1.0) * 1e8 + 0.5), cut);
        }
    }

    /// Three-row packing cuts.
    pub(super) fn three_rows_pack(&mut self, columns: &[ColumnInfo], cuts: &mut CandidateSet) {
        let n = self.num_packing_sets.max(0) as usize;
        let mut pv = vec![vec![0.0f64; n]; n];
        for col in columns {
            if n == 0 {
                continue;
            }
            let present: Vec<usize> = (0..n).filter(|&p| col.packing_set_count[p] != 0).collect();
            for a in 1..present.len() {
                for b in 0..a {
                    let (pa, pb) = (present[a], present[b]);
                    pv[pb][pa] += col.value;
                    pv[pa][pb] += col.value;
                }
            }
        }
        for i in 0..n {
            for j in i + 1..n {
                let vij = pv[i][j];
                if vij == 0.0 {
                    continue;
                }
                for k in j + 1..n {
                    let approx = vij + pv[i][k] + pv[j][k];
                    if !(approx - 1.0 > self.params.violation_tolerance) {
                        continue;
                    }
                    let mut lhs = 0.0;
                    for col in columns {
                        let pc = &col.packing_set_count;
                        let (a, b, c) = (pc[i] != 0, pc[j] != 0, pc[k] != 0);
                        if if a { b || c } else { b && c } {
                            lhs += col.value;
                        }
                    }
                    if !(lhs - 1.0 > self.params.violation_tolerance) {
                        continue;
                    }
                    let cut = self.candidate(
                        1.0,
                        0,
                        3,
                        -1,
                        2,
                        vec![1, 1, 1],
                        vec![i as i32, j as i32, k as i32],
                    );
                    self.insert_candidate(cuts, rounded_score((1.0 - lhs) * 1e8 + 0.5), cut);
                }
            }
        }
    }

    /// Three-row covering cuts.
    pub(super) fn three_rows_cov(&mut self, columns: &[ColumnInfo], cuts: &mut CandidateSet) {
        let n = self.num_covering_sets.max(0) as usize;
        for i in 0..n {
            for j in i + 1..n {
                for k in j + 1..n {
                    let mut lhs = 0.0;
                    for col in columns {
                        let cc = &col.covering_set_count;
                        let sum = cc[j] + cc[i] + cc[k] + 1;
                        lhs = ((sum / 2) as f64) * col.value + lhs;
                    }
                    if !(2.0 - lhs > self.params.violation_tolerance) {
                        continue;
                    }
                    let cut = self.candidate(
                        2.0,
                        1,
                        3,
                        -1,
                        -2,
                        vec![-1, -1, -1],
                        vec![i as i32, j as i32, k as i32],
                    );
                    self.insert_candidate(cuts, rounded_score((lhs - 2.0) * 1e8 + 0.5), cut);
                }
            }
        }
    }
}
