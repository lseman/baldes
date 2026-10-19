//! Local-search rank-one separation (packing and covering), from the C++
//! reference `native/separation/LocalSearchRankOneSeparation.h`. Packing row
//! sets use the separator's own std::minstd_rand0 (seed 1); covering row sets
//! use the C library rand().
use super::separation::{bit, Bits, ColumnInfo, NativeCut, Separation};
use super::subcuts;
use crate::cuts::shared::rbtree::CandidateSet;
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

mod construction;
mod search;
