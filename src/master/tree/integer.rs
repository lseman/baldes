//! Optional end-of-node HiGHS solve over a complete enumerated universe.
use super::*;
use crate::master::solvers::mip::MipStatus;

pub(super) fn run<const D: usize>(
    snapshot: Option<&Rc<root::NodeSnapshot<D>>>,
    problem: &RootProblem<D>,
    node: &mut Node<D>,
    options: &TreeOptions,
    output: &mut TreeOutput,
    deadline: Instant,
) -> Result<()> {
    let (Some(snapshot), Some(configured)) = (snapshot, options.enumerated_mip.as_ref()) else {
        return Ok(());
    };
    let mut configured = configured.clone();
    configured.time_limit = root::EnumerationMipPolicy::time_limit(
        deadline.saturating_duration_since(Instant::now()),
        configured.time_limit,
    );
    configured.objective_cutoff = configured.objective_cutoff.min(output.incumbent);
    let result = if let Some(fixing) = options
        .deluxing
        .as_ref()
        .filter(|_| configured.objective_cutoff.is_finite())
    {
        snapshot.solve_enumerated_mip_with_deluxing(&configured, fixing)?
    } else {
        snapshot.solve_enumerated_mip(&configured)?
    };
    let Some(result) = result else {
        return Ok(());
    };
    output.enumerated_mip_calls += 1;
    output.enumerated_mip_nodes += result.mip.nodes;
    output.mip_nodes += result.mip.nodes;
    if let Some(solution) = &result.mip.solution {
        let mut routes = vec![];
        let mut cover = vec![0; problem.graph.vertices.len()];
        let mut cost = 0.;
        for (route, value) in &result.route_values {
            // The tree requires unit customer coverage, hence nonempty routes
            // cannot have multiplicity greater than one in a feasible solution.
            if *value != 1. {
                return Err(invalid("enumerated MIP routing multiplicity"));
            }
            let checked = feasible_route(problem, &route.arcs)
                .ok_or_else(|| invalid("enumerated MIP route feasibility"))?;
            for &v in &checked.nodes[1..checked.nodes.len() - 1] {
                cover[v as usize] += 1;
            }
            cost += checked.original_cost;
            routes.push(checked);
        }
        if !problem.customers.iter().all(|&v| cover[v] == 1)
            || routes.len() as f64 > problem.rows[0].rhs + 1e-7
            || (cost - solution.objective).abs() > 1e-6 + 1e-10 * cost.abs()
        {
            return Err(invalid("enumerated MIP routing witness"));
        }
        if let Some(upper) = problem.unit_route_incumbent(&routes) {
            if upper < output.incumbent {
                output.incumbent = upper;
                output.routes = routes;
            }
        }
    }
    if matches!(
        result.mip.status,
        MipStatus::Optimal | MipStatus::Infeasible | MipStatus::ObjectiveCutoff
    ) {
        // Any omitted route universe covers no improving solution below its
        // saved incumbent. A restricted MIP proof therefore yields only the
        // minimum of its bound and that coverage cutoff for this routing node.
        let bound = result.mip.bound.min(result.coverage_cutoff);
        if bound.is_finite() {
            node.bound = node.bound.max(bound);
        }
    }
    Ok(())
}
