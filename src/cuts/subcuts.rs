//! Sub-cut violation checks of rank-one separation, from the C++ reference
//! `native/separation/RankOneSubCutChecks.h`.
use super::separation::{ColumnInfo, Separation};

/// Whether some 3-row packing sub-cut of `sets` is violated.
pub fn three_rows_pack_violated(s: &Separation, columns: &[ColumnInfo], sets: &[i32], perms: &[Vec<i32>]) -> bool {
    let mut lhs = vec![0.0f64; perms.len()];
    for col in columns {
        for (i, p) in perms.iter().enumerate() {
            let pc = &col.packing_set_count;
            let at = |q: usize| pc[sets[p[q] as usize] as usize] != 0;
            let cnt = at(1) as i32 + at(0) as i32 + at(2) as i32;
            if cnt > 1 {
                lhs[i] += col.value;
                if lhs[i] - 1.0 > s.params.violation_tolerance {
                    return true;
                }
            }
        }
    }
    false
}
/// Whether some 4-row packing sub-cut (weights 2,1,1,1) of `sets` is violated.
pub fn four_rows_pack_violated(s: &Separation, columns: &[ColumnInfo], sets: &[i32], column_ids: &[i32]) -> bool {
    let perms = &s.permutations_190;
    let mut lhs = vec![0.0f64; perms.len()];
    for &ci in column_ids {
        let col = &columns[ci as usize];
        let pc = &col.packing_set_count;
        for (i, p) in perms.iter().enumerate() {
            let at = |q: usize| pc[sets[p[q] as usize] as usize] != 0;
            let w = if at(0) { 2 } else { 0 } + at(1) as i32 + at(2) as i32 + at(3) as i32;
            if w > 2 {
                lhs[i] += col.value;
                if lhs[i] - 1.0 > s.params.violation_tolerance {
                    return true;
                }
            }
        }
    }
    false
}
/// Whether some 3-row covering sub-cut of `sets` is violated.
pub fn three_rows_cov_violated(s: &Separation, columns: &[ColumnInfo], sets: &[i32], perms: &[Vec<i32>]) -> bool {
    let mut lhs = vec![0.0f64; perms.len()];
    for col in columns {
        for (i, p) in perms.iter().enumerate() {
            let cc = &col.covering_set_count;
            let at = |q: usize| cc[sets[p[q] as usize] as usize] != 0;
            let cnt = at(1) as i32 + at(0) as i32 + at(2) as i32;
            if cnt == 0 {
                continue;
            }
            lhs[i] = (((cnt >> 1) + (cnt & 1)) as f64) * col.value + lhs[i];
            if s.params.violation_tolerance > lhs[i] - 2.0 {
                return true;
            }
        }
    }
    false
}
/// Whether some 4-row covering sub-cut of `sets` is violated.
pub fn four_rows_cov_violated(s: &Separation, columns: &[ColumnInfo], sets: &[i32], column_ids: &[i32]) -> bool {
    let perms_a = &s.permutations_5rows_b;
    let perms_b = &s.covering_permutations;
    let mut lhs_a = vec![0.0f64; perms_a.len()];
    let mut lhs_b = vec![0.0f64; perms_b.len()];
    let ceil3 = |c: u32| {
        let q = c / 3;
        q + (c != q * 3) as u32
    };
    for &ci in column_ids {
        let col = &columns[ci as usize];
        let cc = &col.covering_set_count;
        let present = |p: &[i32], q: usize| cc[sets[p[q] as usize] as usize] != 0;
        for (i, p) in perms_a.iter().enumerate() {
            let c = present(p, 1) as u32 + present(p, 0) as u32 + present(p, 2) as u32 + present(p, 3) as u32;
            if c == 0 {
                continue;
            }
            lhs_a[i] = (ceil3(c) as f64) * col.value + lhs_a[i];
            if s.params.violation_tolerance > lhs_a[i] - 2.0 {
                return true;
            }
        }
        for (i, p) in perms_b.iter().enumerate() {
            let w = present(p, 0) as u32 + if present(p, 1) { 2 } else { 0 } + if present(p, 2) { 4 } else { 0 };
            if w == 0 {
                continue;
            }
            lhs_b[i] = (ceil3(w) as f64) * col.value + lhs_b[i];
            if s.params.violation_tolerance > lhs_b[i] - 3.0 {
                return true;
            }
        }
    }
    false
}
