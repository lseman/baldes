use super::*;

impl<const D: usize> State<D> {
    pub(super) fn remaining_time(&self) -> Result<std::time::Duration> {
        match self.deadline {
            Some(end) => end
                .checked_duration_since(std::time::Instant::now())
                .filter(|d| !d.is_zero())
                .ok_or_else(|| crate::incomplete("node time limit")),
            None => Ok(std::time::Duration::MAX),
        }
    }
    pub(super) fn customers(route: &Route) -> Vec<usize> {
        route.nodes[1..route.nodes.len() - 1]
            .iter()
            .map(|&v| v as usize)
            .collect()
    }
    pub(super) fn is_enumerated(&self, route: &Route) -> bool {
        if self.cut.enumerated.is_empty() {
            return false;
        }
        let mut arcs = route.arcs.clone();
        arcs.sort_unstable();
        self.cut.enumerated.contains(&arcs)
    }
    pub(super) fn cut_entries(
        &self,
        problem: &RoutingContext<D>,
        route: &Route,
    ) -> Result<Vec<(usize, f64)>> {
        self.cut
            .cached_entries(problem, &self.solver, route, self.is_enumerated(route))
    }
    pub(super) fn column(
        &self,
        problem: &RoutingContext<D>,
        route: &Route,
        duals: &[f64],
        tag: i32,
    ) -> Result<Column> {
        let mut entries = problem.base_entries(route)?;
        entries.extend(self.cut_entries(problem, route)?);
        entries.sort_by_key(|&(row, _)| row);
        let (cost, _) = problem.route_cost(route)?;
        let mut reduced_cost = cost;
        for &(r, a) in &entries {
            reduced_cost -= duals.get(r).copied().unwrap_or(0.) * a;
        }
        Ok(Column {
            cost,
            reduced_cost,
            entries,
            acceptable: true,
            tag,
            // Registration assigns a stable node-local tag using the full
            // ordered arc path and resource schedule. Pool membership needs
            // only that interned identity, not another copy of the schedule.
            identity: format!("route:{tag}"),
        })
    }
    /// cache_costs: arc reduced costs and the rounded cut duals, active cuts
    /// ordered by rounded dual magnitude then index.
    pub(super) fn reprice(&mut self, problem: &RoutingContext<D>, duals: &[f64]) -> Result<f64> {
        let rounded = self.rounded;
        let offset = problem.reprice(
            &mut self.solver,
            duals,
            rounded,
            &self.cut.capacity_arc_entries,
        )?;
        for (cut, row) in self
            .solver
            .cuts
            .knapsack_cuts
            .iter_mut()
            .zip(&self.cut.knapsack_rows)
        {
            cut.dual = row.map_or(0., |r| duals[r]);
        }
        if !self.cut.rows.is_empty() {
            // Strong k-path cuts are not sorted: they keep registry order.
            let mut order: Vec<usize> = (0..self.cut.rows.len())
                .filter(|&i| {
                    !self.solver.cuts.cuts[i].visit_once
                        && self.cut.rows[i].is_some_and(|r| duals[r].abs() > 1e-6)
                })
                .collect();
            sort_rank_one_pricing_order(&mut order, |i| -duals[self.cut.rows[i].unwrap()]);
            self.solver.cuts.pricing_order = order;
            // Keep raw duals: native retrieval filters them before the 1e-8
            // rounding, so a dual just above 1e-6 stays active.
            self.solver.cuts.round_duals = true;
            for (i, row) in self.cut.rows.iter().enumerate() {
                self.solver.cuts.cuts[i].dual = row.map_or(0., |r| duals[r]);
            }
        }
        Ok(offset)
    }
}

/// Native rank-one retrieval stable-sorts the unrounded pricing values,
/// largest first; values within 1e-6 keep cut-id order. The tolerant
/// comparator is not transitive, so use the libstdc++ algorithm.
pub(super) fn sort_rank_one_pricing_order(order: &mut [usize], value: impl Fn(usize) -> f64) {
    crate::cuts::shared::stdsort::stable_sort(order, |&a, &b| {
        let (va, vb) = (value(a), value(b));
        va > vb + 1e-6 || (!(vb - 1e-6 > va) && a < b)
    });
}

#[cfg(test)]
#[path = "../../../tests/unit/master/routing/state/tests.rs"]
mod tests;
