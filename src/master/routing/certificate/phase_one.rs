//! Complete unit-cover path proof for cost-suppressed coupled phase I.
use super::*;

/// Coupled replay may remove only independently certified infeasible paths.
/// Unknown paths remain in the relaxation. Dynamic cut rows are omitted: this
/// proof uses the original path-defined rows and the unit fleet upper bound.
pub(in crate::master::routing) fn zero_objective_lower_bound<const D: usize>(
    problem: &RoutingContext<D>,
    master: &ClpMaster,
    duals: &[f64],
    deadline: Instant,
) -> Option<f64> {
    let graph = &problem.model.networks[0].graph;
    let kernel = graph.coupled_kernel()?;
    if kernel.has_costs() || graph.arcs.iter().any(|arc| arc.cost != 0.) {
        return None;
    }
    let mut budget = Budget {
        deadline: deadline.min(Instant::now().checked_add(Duration::from_millis(250))?),
        work: 10_000_000,
        entries: 1_000_000,
    };
    budget.spend(0)?;
    let map = problem.row_map.borrow();
    if map.len() != problem.model.rows.len() || duals.len() != master.rhs.len() {
        return None;
    }
    budget.spend(master.rhs.len().checked_add(map.len())?)?;
    let mut used = vec![false; master.rhs.len()];
    let mut rhs = Vec::with_capacity(map.len());
    let mut senses = Vec::with_capacity(map.len());
    let mut projected = Vec::with_capacity(map.len());
    for (row, &physical) in problem.model.rows.iter().zip(map.iter()) {
        budget.spend(1)?;
        if physical >= used.len()
            || used[physical]
            || row.rhs != master.rhs[physical]
            || row.sense != master.senses[physical]
        {
            return None;
        }
        used[physical] = true;
        rhs.push(row.rhs);
        senses.push(row.sense);
        projected.push(duals[physical]);
    }
    // Integer unit-cover rows justify physical-simple paths. Scalar resource
    // intervals remain conservative, while removing coupled restrictions
    // first gives a complete candidate superset for independent replay.
    budget.spend(graph.arcs.len().checked_add(graph.vertices.len())?)?;
    let mut relaxed = problem.clone();
    relaxed.model.networks[0]
        .graph
        .set_coupled_resources(&[])
        .ok()?;
    let domain = ConservativeCoverRelaxation::collect(
        &relaxed,
        PathDomainLimits {
            max_paths: 10_000,
            max_stored_arcs: budget.entries,
            max_work: budget.work,
            deadline: Some(budget.deadline),
        },
    )
    .ok()?;
    budget.spend(domain.work_used())?;
    let mut entries = Vec::new();
    for path in domain.paths().paths() {
        let arcs = path.arc_positions();
        budget.store(arcs.len().checked_add(1)?)?;
        if kernel
            .certify_with_budget(arcs, &mut budget.work, Some(budget.deadline))
            .is_some_and(|(feasible, _)| !feasible)
        {
            continue;
        }
        budget.spend(0)?;
        let mut route = Route {
            coupled_values: vec![],
            nodes: vec![i32::try_from(graph.source).ok()?],
            arcs: vec![],
            cost: 0.,
            original_cost: 0.,
            resources: vec![],
            resource_profile: vec![],
        };
        let mut column = Vec::new();
        for &position in arcs {
            let arc = &graph.arcs[position];
            route.nodes.push(i32::try_from(arc.head).ok()?);
            route.arcs.push(arc.id);
            let coefficients = &problem.model.networks[0].arc_entries[position];
            budget.store(coefficients.len())?;
            column.extend_from_slice(coefficients);
        }
        // Empty terminal paths are excluded by the original routing contract.
        if route.nodes.len() == 2 {
            continue;
        }
        budget.store(problem.model.networks[0].route_entries.len())?;
        column.extend_from_slice(&problem.model.networks[0].route_entries);
        for (row, constraint) in &problem.branch_constraints {
            budget.spend(1)?;
            if matches!(constraint, BranchConstraint::RyanFoster(_)) {
                budget.store(1)?;
                column.push((*row, constraint.coefficient(problem, &route)));
            }
        }
        entries.push(column);
    }
    let columns = entries
        .iter()
        .map(|entries| ProofColumn {
            cost_terms: &[],
            entries,
        })
        .collect::<Vec<_>>();
    FiniteMaster {
        rhs: &rhs,
        senses: &senses,
        columns: &columns,
        offset_terms: &[],
        multiplicity_row: 0,
    }
    .lower_bound_checked(&projected, || budget.spend(1))
}

#[cfg(test)]
#[path = "../../../../tests/unit/master/routing/certificate/phase_one_tests.rs"]
mod tests;
