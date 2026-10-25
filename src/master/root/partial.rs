//! Node-local fixed routes and exact residual row contributions.
use super::*;

/// Fix an additional integer quantity of a physical elementary route.
/// Arc IDs are in traversal order; costs/resources are reconstructed from the graph.
#[derive(Clone, Debug)]
pub struct FixedRoute {
    pub arcs: Vec<i32>,
    pub quantity: u32,
}

impl<const D: usize> RootProblem<D> {
    pub(super) fn fixed_cost(&self) -> f64 {
        self.fixed_routes
            .iter()
            .map(|(r, q)| r.original_cost * f64::from(*q))
            .sum()
    }
    pub(super) fn fixed_sum(
        &self,
        mut coefficient: impl FnMut(&Route) -> Result<f64>,
    ) -> Result<f64> {
        let mut sum = 0.;
        for (route, quantity) in &self.fixed_routes {
            sum += coefficient(route)? * f64::from(*quantity);
        }
        if !sum.is_finite() {
            return Err(invalid("fixed-route contribution overflow"));
        }
        Ok(sum)
    }
}
impl<const D: usize> RootOutput<D> {
    /// Current route values including the fixed part. Positive artificials must
    /// still be ruled out before treating the current LP values as a witness.
    pub fn route_values(&self) -> Vec<(&Route, f64)> {
        self.fixed_routes
            .iter()
            .map(|(r, q)| (r, *q))
            .chain(
                self.master
                    .column_tags
                    .iter()
                    .zip(&self.values)
                    .filter_map(|(&tag, &value)| {
                        (tag >= 0 && value > 0.).then(|| (&self.routes[tag as usize], value))
                    }),
            )
            .collect()
    }
}
impl<const D: usize> NodeSnapshot<D> {
    pub fn fixed_routes(&self) -> &[(Route, u32)] {
        &self.problem.fixed_routes
    }
    pub fn fixed_cost(&self) -> f64 {
        self.problem.fixed_cost()
    }
}

/// Restrict a solved node by fixing additional elementary routes. Quantities
/// are additive across calls. The parent, its basis and siblings are unchanged.
/// All returned objective values include the fixed contribution.
pub fn inherit_fixed_routes<const D: usize>(
    parent: &NodeSnapshot<D>,
    fixed: &[FixedRoute],
    mut options: RootOptions,
) -> Result<RootOutput<D>> {
    let (problem, master, warm) = residual_fork(parent, fixed, options.cg.artificial_cost)?;
    // This is the remaining fleet bound, not the total pre-fixing multiplicity.
    options.cg.multiplicity = problem.rows[0].rhs.max(0.);
    options.node_depth = options.node_depth.max(parent.branch_depth());
    solve_seeded(problem, options, Some((parent, master, warm)))
}

fn residual_fork<const D: usize>(
    parent: &NodeSnapshot<D>,
    fixed: &[FixedRoute],
    penalty: f64,
) -> Result<(RootProblem<D>, ClpMaster, cg::WarmStart)> {
    let (mut problem, mut master, mut warm) = parent.fork(&[], penalty)?;
    let mut checked = vec![];
    for entry in fixed {
        if entry.quantity == 0 {
            return Err(invalid("positive fixed-route quantity required"));
        }
        let route = problem
            .elementary_route(&entry.arcs)
            .ok_or_else(|| invalid("fixed route must be a nonempty feasible elementary path"))?;
        if !parent
            .pricing
            .route_satisfies_relaxation(&route.nodes, &route.arcs)?
        {
            return Err(invalid("fixed route violates the node pricing constraints"));
        }
        checked.push((route, entry.quantity));
    }
    let mut rhs = master.rhs.clone();
    for (route, quantity) in &checked {
        let mut entries = problem.base_entries(route)?;
        // Fixed elementary routes always use exact full-memory coefficients.
        // Remaining LP routes may keep valid weaker limited-memory coefficients.
        entries.extend(parent.cut.entries(&parent.pricing, route, true)?);
        for (row, coefficient) in entries {
            rhs[row] -= coefficient * f64::from(*quantity);
        }
    }
    master.set_rhs(&rhs)?;
    for (logical, &physical) in problem.row_map.borrow().iter().enumerate() {
        problem.rows[logical].rhs = rhs[physical];
    }
    problem.fixed_routes.extend(checked);
    let offset = problem.fixed_cost();
    if !offset.is_finite() {
        return Err(invalid("fixed-route objective overflow"));
    }
    master.objective_offset = offset;
    if !fixed.is_empty() {
        warm.clear_proximal();
    }
    Ok((problem, master, warm))
}

#[cfg(test)]
#[path = "../../../tests/unit/master/root/partial/tests.rs"]
mod tests;
