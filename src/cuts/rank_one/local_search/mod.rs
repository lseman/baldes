//! Local-search rank-one separation (packing and covering).
//! Randomized starts use an owned, deterministically seeded shared RNG.
use super::separation::{bit, Bits, ColumnInfo, NativeCut, Separation};
use super::subcuts;
use crate::cuts::shared::rbtree::CandidateSet;
use crate::rng::Xoroshiro128Plus;
use std::collections::{BTreeMap, HashSet};

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
    // Accepted swaps rebuild the same state matrix. Keep its row allocations
    // and reset every word so states from the preceding assignment disappear.
    bitmaps.resize_with((max_state + 1) as usize, Vec::new);
    for bitmap in bitmaps.iter_mut() {
        bitmap.resize(nwords, 0);
        bitmap.fill(0);
    }
    // Cuts use only a few rows. Select their nonzero multipliers once rather
    // than scanning the entire row universe for every column after each swap.
    let active: Vec<_> = row_multipliers
        .iter()
        .enumerate()
        .filter(|(_, multiplier)| **multiplier != 0)
        .map(|(row, &multiplier)| (row, multiplier as u32))
        .collect();
    for k in 0..n {
        let col = &columns[column_indices[k] as usize];
        let count = if COVERING {
            &col.covering_set_count
        } else {
            &col.packing_set_count
        };
        let mut state: u32 = 0;
        for &(row, multiplier) in &active {
            state = state.wrapping_add((count[row] as u32).wrapping_mul(multiplier));
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
    rng: Xoroshiro128Plus,
}

mod bitset;
mod construction;
mod gain;
mod search;

#[cfg(test)]
#[path = "../../../../tests/unit/cuts/rank_one/local_search/tests.rs"]
mod tests;
