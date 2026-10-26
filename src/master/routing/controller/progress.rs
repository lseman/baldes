//! Counts of cut rows currently installed in the master, grouped by enabled family.
use super::*;
use crate::cuts::{capacity::CapacityKind, knapsack::KnapsackKind};

impl CutState {
    pub(super) fn progress_counts(
        &self,
        knapsack_cuts: &[crate::cuts::knapsack::ResourceKnapsackCut],
    ) -> Vec<(&'static str, usize)> {
        let mut counts = vec![];
        let kpath_indices: HashSet<_> = self.kpath_indices.iter().copied().collect();
        if self.options.enabled {
            let rank_one = self
                .rows
                .iter()
                .enumerate()
                .filter(|(index, row)| row.is_some() && !kpath_indices.contains(index))
                .count();
            counts.push(("Rank1", rank_one));
        }
        if let Some(separator) = &self.capacity_separator {
            if separator.parameters.enabled {
                let mut capacity = 0;
                let mut two_path = 0;
                for (cut, row) in self.capacity_cuts.iter().zip(&self.capacity_rows) {
                    if row.is_some() {
                        if cut.kind == CapacityKind::TwoPath {
                            two_path += 1;
                        } else {
                            capacity += 1;
                        }
                    }
                }
                counts.push(("Capacity", capacity));
                if separator.parameters.use_two_path {
                    counts.push(("2Path", two_path));
                }
            }
        }
        if self.kpath_separator.is_some() {
            let kpath = self
                .rows
                .iter()
                .enumerate()
                .filter(|(index, row)| row.is_some() && kpath_indices.contains(index))
                .count();
            counts.push(("KPath", kpath));
        }
        if let Some(separator) = &self.knapsack_separator {
            let mut knapsack = 0;
            let mut fenchel = 0;
            for (cut, row) in knapsack_cuts.iter().zip(&self.knapsack_rows) {
                if row.is_some() {
                    if cut.cut.kind == KnapsackKind::Fenchel {
                        fenchel += 1;
                    } else {
                        knapsack += 1;
                    }
                }
            }
            if separator.parameters.use_rounding || separator.parameters.max_one_k > 0 {
                counts.push(("Knapsack", knapsack));
            }
            if separator.parameters.fenchel.is_some() {
                counts.push(("Fenchel", fenchel));
            }
        }
        counts
    }
}

#[cfg(test)]
#[path = "../../../../tests/unit/master/routing/controller/progress/tests.rs"]
mod tests;
