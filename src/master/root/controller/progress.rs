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
mod tests {
    use super::*;
    use crate::cuts::knapsack::{FenchelParameters, RouteLoadKnapsackCut};

    fn empty_cut_state() -> CutState {
        let mut graph = Graph::<2>::new(3, 1).unwrap();
        for vertex in 0..3 {
            graph.set_bounds(vertex, 0, 0., 10.).unwrap();
        }
        graph.add_arc(0, 1, 1.).unwrap();
        graph.add_arc(1, 2, 1.).unwrap();
        let problem = RootProblem::set_partitioning(graph, 1.).unwrap();
        let mut options = RootOptions::default();
        options.capture_state = true;
        options.enumerate = false;
        super::super::super::solve(problem, options)
            .unwrap()
            .snapshot
            .unwrap()
            .cut
            .clone()
    }

    #[test]
    fn enabled_families_count_live_rows_without_double_counting() {
        let mut state = empty_cut_state();
        assert!(state.progress_counts(&[]).is_empty());
        state.options.enabled = true;
        state.rows = vec![Some(0), None, Some(1), None];
        state.kpath_indices = vec![2, 3];
        state.kpath_separator = Some(RootKPathSeparator::Standalone(
            KPathSeparator::new(vec![0., 1.], 2.).unwrap(),
        ));
        let mut capacity = CapacitySeparator::new(vec![0., 1.], 2.).unwrap();
        capacity.parameters.use_two_path = true;
        state.capacity_separator = Some(capacity);
        state.capacity_cuts = [
            CapacityKind::Capacity,
            CapacityKind::Complement,
            CapacityKind::TwoPath,
            CapacityKind::Capacity,
        ]
        .iter()
        .enumerate()
        .map(|(id, &kind)| CapacityCut {
            id,
            kind,
            customers: vec![],
            other: vec![],
            rhs: 0,
            k: 0,
            demand: 0.,
            violation: 0.,
            edges: Default::default(),
        })
        .collect();
        state.capacity_rows = vec![Some(2), Some(3), Some(4), None];
        let mut separator = ResourceKnapsackSeparator::default();
        separator.parameters.fenchel = Some(FenchelParameters::default());
        state.knapsack_separator = Some(separator);
        let knapsack: Vec<_> = [
            KnapsackKind::Steps,
            KnapsackKind::Fenchel,
            KnapsackKind::Fenchel,
        ]
        .iter()
        .map(|&kind| crate::cuts::knapsack::ResourceKnapsackCut {
            id: 0,
            row: 0,
            cut_class: 0,
            graph_resources: Default::default(),
            dual: 0.,
            cut: RouteLoadKnapsackCut {
                kind,
                numerator: 1,
                denominator: 1,
                rhs: 0,
                lhs: 0.,
                violation: 0.,
                thresholds: vec![],
                coefficients: Default::default(),
            },
        })
        .collect();
        state.knapsack_rows = vec![Some(5), Some(6), None];
        assert_eq!(
            state.progress_counts(&knapsack),
            vec![
                ("Rank1", 1),
                ("Capacity", 2),
                ("2Path", 1),
                ("KPath", 1),
                ("Knapsack", 1),
                ("Fenchel", 1),
            ]
        );
        state.options.enabled = false;
        let capacity = state.capacity_separator.as_mut().unwrap();
        capacity.parameters.enabled = false;
        state.kpath_separator = None;
        state
            .knapsack_separator
            .as_mut()
            .unwrap()
            .parameters
            .max_one_k = 0;
        assert_eq!(state.progress_counts(&knapsack), vec![("Fenchel", 1)]);
        state
            .knapsack_separator
            .as_mut()
            .unwrap()
            .parameters
            .fenchel = None;
        assert!(state.progress_counts(&knapsack).is_empty());
    }
}
