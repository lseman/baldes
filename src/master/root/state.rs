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
        problem: &RootProblem<D>,
        route: &Route,
    ) -> Result<Vec<(usize, f64)>> {
        self.cut
            .cached_entries(problem, &self.solver, route, self.is_enumerated(route))
    }
    pub(super) fn column(
        &self,
        problem: &RootProblem<D>,
        route: &Route,
        duals: &[f64],
        tag: i32,
    ) -> Result<Column> {
        let mut entries = problem.base_entries(route)?;
        entries.extend(self.cut_entries(problem, route)?);
        entries.sort_by_key(|&(row, _)| row);
        let (cost, _) = problem.route_cost(route);
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
    pub(super) fn reprice(&mut self, problem: &RootProblem<D>, duals: &[f64]) -> Result<f64> {
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
            let scaled = |row: usize| (-duals[row] * 1e8 + 0.5).floor();
            let mut order: Vec<usize> = (0..self.cut.rows.len())
                .filter(|&i| self.cut.rows[i].is_some_and(|r| duals[r].abs() > 1e-6))
                .collect();
            order.sort_by(|&a, &b| {
                let ka = -scaled(self.cut.rows[a].unwrap()).abs();
                let kb = -scaled(self.cut.rows[b].unwrap()).abs();
                ka.total_cmp(&kb).then(a.cmp(&b))
            });
            self.solver.cuts.pricing_order = order;
            for (i, row) in self.cut.rows.iter().enumerate() {
                self.solver.cuts.cuts[i].dual = match row {
                    Some(r) => -scaled(*r) / 1e8,
                    None => 0.,
                };
            }
        }
        Ok(offset)
    }
}
