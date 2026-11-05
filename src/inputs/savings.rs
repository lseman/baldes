//! Clarke–Wright parallel savings initialization for routing masters.
use super::{HeuristicRouteMap, Instance, Rounding};
use crate::{invalid, Graph, Result, Route};

/// Graph-validated initial columns and, when available, a fleet-feasible plan.
#[derive(Clone, Debug)]
pub struct SavingsSeed {
    pub routes: Vec<Route>,
    /// A complete partition within the instance fleet, in original cost units.
    pub incumbent: Option<f64>,
}

impl Instance {
    /// Build parallel Clarke–Wright savings routes. Capacity, time windows,
    /// missing arcs and additional graph resources are checked on every merge.
    /// Infeasible singleton routes are omitted; artificial variables can cover
    /// customers for which this constructor cannot supply a feasible column.
    pub fn seed_savings<const D: usize>(
        &self,
        graph: &Graph<D>,
        rounding: Rounding,
    ) -> Result<SavingsSeed> {
        let n = self.customers.len().saturating_sub(1);
        if n == 0 || graph.vertices.len() != n + 2 || graph.source != 0 || graph.sink != n + 1 {
            return Err(invalid(
                "Savings seeding requires the instance's split-depot graph",
            ));
        }
        graph.validate()?;
        let map = HeuristicRouteMap::new(graph);
        let mut plans: Vec<Vec<usize>> = (0..=n).map(|i| vec![i]).collect();
        let mut routes: Vec<_> = (0..=n)
            .map(|i| {
                if i == 0 {
                    None
                } else {
                    map.convert(graph, &[i])
                }
            })
            .collect();
        let mut owner: Vec<_> = (0..=n).collect();
        let mut savings = Vec::new();
        for i in 1..=n {
            for j in 1..=n {
                if i == j {
                    continue;
                }
                let saving = self.distance(i, 0, rounding) + self.distance(0, j, rounding)
                    - self.distance(i, j, rounding);
                if saving > 0. && saving.is_finite() {
                    savings.push((saving, i, j));
                }
            }
        }
        savings
            .sort_unstable_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)).then(a.2.cmp(&b.2)));
        for (_, i, j) in savings {
            let (a, b) = (owner[i], owner[j]);
            if a == b || routes[a].is_none() || routes[b].is_none() {
                continue;
            }
            let (left, right) = (&plans[a], &plans[b]);
            if !(left.first() == Some(&i) || left.last() == Some(&i))
                || !(right.first() == Some(&j) || right.last() == Some(&j))
            {
                continue;
            }
            let mut merged = left.clone();
            if merged.last() != Some(&i) {
                merged.reverse();
            }
            if right.first() == Some(&j) {
                merged.extend(right);
            } else {
                merged.extend(right.iter().rev());
            }
            let Some(route) = map.convert(graph, &merged) else {
                continue;
            };
            // Reversal can change directed costs; require an actual saving.
            if route.original_cost
                >= routes[a].as_ref().unwrap().original_cost
                    + routes[b].as_ref().unwrap().original_cost
            {
                continue;
            }
            for &customer in &merged {
                owner[customer] = a;
            }
            plans[a] = merged;
            plans[b].clear();
            routes[a] = Some(route);
            routes[b] = None;
        }
        let routes: Vec<_> = routes.into_iter().flatten().collect();
        let covered: usize = routes.iter().map(|r| r.nodes.len() - 2).sum();
        let incumbent = (covered == n && routes.len() <= self.vehicles)
            .then(|| routes.iter().map(|r| r.original_cost).sum());
        Ok(SavingsSeed { routes, incumbent })
    }
}

#[cfg(test)]
#[path = "../../tests/unit/inputs/savings.rs"]
mod tests;
