use super::*;

impl CutState {
    /// Installed separators by decreasing priority. On a tie the relaxation
    /// improvement, which has no name, sorts after the cut families.
    pub(super) fn separators(&self) -> Vec<(f64, Separator)> {
        let mut list = vec![];
        if self.capacity_separator.is_some() {
            list.push((self.capacity_priority, Separator::Capacity));
        }
        if self.kpath_separator.is_some() {
            list.push((RANK_ONE_PRIORITY, Separator::KPath));
        }
        if self.knapsack_separator.is_some() {
            list.push((RANK_ONE_PRIORITY, Separator::Knapsack));
        }
        if self.options.enabled {
            list.push((RANK_ONE_PRIORITY, Separator::RankOne));
        }
        if self.relaxation_priority > 0. {
            list.push((self.relaxation_priority, Separator::Relaxation));
        }
        list.sort_by(|a, b| {
            b.0.total_cmp(&a.0)
                .then((a.1 == Separator::Relaxation).cmp(&(b.1 == Separator::Relaxation)))
        });
        list
    }
    /// Whether rank-one cuts are at or above the current level.
    pub(super) fn rank_one_active(&self) -> bool {
        (self.options.enabled || self.kpath_separator.is_some()) && self.level <= RANK_ONE_PRIORITY
    }
    /// The next lower priority level, if any.
    pub(super) fn next_level(&self) -> Option<f64> {
        self.separators()
            .into_iter()
            .map(|(p, _)| p)
            .find(|&p| p < self.level)
    }
}

impl CutState {
    pub(super) fn entries<const D: usize>(
        &self,
        solver: &Solver<D>,
        route: &Route,
        enumerated: bool,
    ) -> Result<Vec<(usize, f64)>> {
        let mut entries = vec![];
        let customers = State::<D>::customers(route);
        for (cut, row) in self.capacity_cuts.iter().zip(&self.capacity_rows) {
            if let Some(row) = row {
                let coefficient = cut.coefficient(&customers);
                if coefficient != 0 {
                    entries.push((*row, coefficient as f64));
                }
            }
        }
        for (cut, row) in solver.cuts.knapsack_cuts.iter().zip(&self.knapsack_rows) {
            if let Some(row) = row {
                let coefficient = solver.knapsack_coefficient(cut, &route.resources);
                if coefficient != 0 {
                    entries.push((*row, coefficient as f64));
                }
            }
        }
        if self.rows.iter().all(Option::is_none) {
            return Ok(entries);
        }
        let path = solver
            .cuts
            .route(&solver.graph, &State::<D>::customers(route))?;
        for (index, row) in self.rows.iter().enumerate() {
            let Some(row) = *row else {
                continue;
            };
            let c = solver
                .cuts
                .route_coefficient(&solver.graph, index, &path, enumerated)?;
            if c != 0 {
                entries.push((row, c as f64));
            }
        }
        Ok(entries)
    }
}
