//! Task-local buffers retained across randomized local-search starts.
use super::LocalSearch;

#[derive(Default)]
pub(super) struct SearchWorkspace {
    key: Option<(bool, i32, i32, i32)>,
    pub(super) increases: Vec<Vec<(i32, i32)>>,
    pub(super) decreases: Vec<Vec<(i32, i32)>>,
    pub(super) row_multipliers: Vec<i32>,
    pub(super) states: Vec<Vec<u64>>,
    pub(super) increased: Vec<Vec<u64>>,
    pub(super) decreased: Vec<Vec<u64>>,
    pub(super) in_cut: Vec<usize>,
    pub(super) active_rows: Vec<(usize, u32)>,
}

impl SearchWorkspace {
    pub(super) fn prepare<const COVERING: bool>(
        &mut self,
        max_state: i32,
        max_change: i32,
        denominator: i32,
        rows: usize,
        words: usize,
    ) {
        let key = (COVERING, max_state, max_change, denominator);
        if self.key != Some(key) {
            self.increases
                .resize_with((max_change + 1) as usize, Vec::new);
            self.decreases
                .resize_with((max_change + 1) as usize, Vec::new);
            for changes in self.increases.iter_mut().chain(&mut self.decreases) {
                changes.clear();
            }
            for d in 1..=max_change {
                for s in 0..=max_state - d {
                    let a = LocalSearch::<COVERING>::divide(s, denominator);
                    let b = LocalSearch::<COVERING>::divide(d + s, denominator);
                    if a < b {
                        self.increases[d as usize].push((s, b - a));
                    }
                }
                for s in d..=max_state {
                    let a = LocalSearch::<COVERING>::divide(s, denominator);
                    let b = LocalSearch::<COVERING>::divide(s - d, denominator);
                    if a > b {
                        self.decreases[d as usize].push((s, a - b));
                    }
                }
            }
            self.key = Some(key);
        }
        self.row_multipliers.resize(rows, 0);
        self.row_multipliers.fill(0);
        let states = (LocalSearch::<COVERING>::divide(max_state, denominator).max(0) + 1) as usize;
        for matrix in [&mut self.increased, &mut self.decreased] {
            matrix.resize_with(states, Vec::new);
            for bitmap in matrix {
                bitmap.resize(words, 0);
                bitmap.fill(0);
            }
        }
    }
}
