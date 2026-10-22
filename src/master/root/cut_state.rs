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
        self.entries_with_capacity(solver, route, enumerated, None)
    }

    pub(super) fn cached_entries<const D: usize>(
        &self,
        problem: &RootProblem<D>,
        solver: &Solver<D>,
        route: &Route,
        enumerated: bool,
    ) -> Result<Vec<(usize, f64)>> {
        let capacity = capacity_entries_from_arcs(
            problem,
            route,
            &self.capacity_rows,
            &self.capacity_arc_entries,
        );
        self.entries_with_capacity(solver, route, enumerated, capacity)
    }

    fn entries_with_capacity<const D: usize>(
        &self,
        solver: &Solver<D>,
        route: &Route,
        enumerated: bool,
        capacity: Option<Vec<(usize, f64)>>,
    ) -> Result<Vec<(usize, f64)>> {
        let mut entries = if let Some(capacity) = capacity {
            capacity
        } else {
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
            entries
        };
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

/// Accumulate prepared physical-arc weights in route order, then round once per
/// cut, just like CapacityCut::coefficient. Unsupported route representations
/// return None so the customer-sequence calculation remains the fallback.
fn capacity_entries_from_arcs<const D: usize>(
    problem: &RootProblem<D>,
    route: &Route,
    rows: &[Option<usize>],
    arc_entries: &[Vec<(usize, f64)>],
) -> Option<Vec<(usize, f64)>> {
    let Some(last_row) = rows.iter().flatten().max() else {
        return Some(vec![]);
    };
    let graph = &problem.graph;
    if graph.source != 0
        || route.nodes.len() < 2
        || route.arcs.len() != route.nodes.len() - 1
        || route.nodes.first().copied() != Some(0)
        || route.nodes.last().copied() != Some(graph.sink as i32)
        || route.nodes[1..route.nodes.len() - 1].contains(&(graph.sink as i32))
    {
        return None;
    }
    let mut sums = vec![0.; last_row.checked_add(1)?];
    for (edge, &id) in route.arcs.iter().enumerate() {
        let index = problem.arc_position(id)?;
        let arc = &graph.arcs[index];
        if arc.tail as i32 != route.nodes[edge] || arc.head as i32 != route.nodes[edge + 1] {
            return None;
        }
        for &(row, weight) in arc_entries.get(index)? {
            *sums.get_mut(row)? += weight;
        }
    }
    let mut entries = vec![];
    for &row in rows.iter().flatten() {
        let coefficient = sums[row].round() as i32;
        if coefficient != 0 {
            entries.push((row, coefficient as f64));
        }
    }
    Some(entries)
}

#[cfg(test)]
#[path = "../../../tests/unit/master/root/cut_state/tests.rs"]
mod tests;
