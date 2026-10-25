//! Integer-valid disjunctions on original quantities, never on artificial LPs.
use super::*;
use crate::master::preprocessing::Bounds;

pub(super) fn fractional(value: f64, tolerance: f64) -> bool {
    value.is_finite()
        && value.abs() < 9_007_199_254_740_992.
        && (value - value.round()).abs() > tolerance
}

pub(super) fn children<const D: usize>(
    output: &Output<D>,
    tolerance: f64,
    policy: NetworkBranchingPolicy,
    windows: &[super::time_windows::NetworkWindows],
    variables: &[MappedBranchVariable],
) -> Result<Option<Vec<NodeSnapshot<D>>>> {
    let mut children = children_inner(output, tolerance, policy, windows, variables)?;
    if let Some(children) = &mut children {
        for child in children {
            child.state.node_depth = output.snapshot.state.node_depth.saturating_add(1);
        }
    }
    Ok(children)
}

fn children_inner<const D: usize>(
    output: &Output<D>,
    tolerance: f64,
    policy: NetworkBranchingPolicy,
    windows: &[super::time_windows::NetworkWindows],
    variables: &[MappedBranchVariable],
) -> Result<Option<Vec<NodeSnapshot<D>>>> {
    if policy == NetworkBranchingPolicy::MultiplicityFirst {
        if let Some(children) = multiplicity_children(output, tolerance)? {
            return Ok(Some(children));
        }
    }
    let mut children = if let Some(mapped) = model_mapping::children(output, variables, tolerance)?
    {
        Some(mapped)
    } else {
        resource_or_assignment_children(output, tolerance, windows)?
    };
    if policy == NetworkBranchingPolicy::MultiplicityFirst {
        if let Some(children) = &mut children {
            children.reverse();
        }
    }
    Ok(children)
}

pub(super) fn multiplicity_children<const D: usize>(
    output: &Output<D>,
    tolerance: f64,
) -> Result<Option<Vec<NodeSnapshot<D>>>> {
    let node = &output.snapshot;
    for oracle in &node.state.oracles {
        let value = output
            .routes
            .iter()
            .zip(&output.values)
            .filter(|(route, _)| route.network == oracle.network.id)
            .map(|(_, &quantity)| quantity)
            .chain(
                node.state
                    .fixed_routes
                    .iter()
                    .filter(|fixed| fixed.route.network == oracle.network.id)
                    .map(|fixed| fixed.quantity),
            )
            .sum::<f64>();
        if value >= 0. && fractional(value, tolerance) {
            return Ok(Some(vec![
                node.with_multiplicity(
                    oracle.network.id,
                    Bounds {
                        lower: value.ceil(),
                        upper: f64::INFINITY,
                    },
                )?,
                node.with_multiplicity(
                    oracle.network.id,
                    Bounds {
                        lower: 0.,
                        upper: value.floor(),
                    },
                )?,
            ]));
        }
    }
    Ok(None)
}

fn resource_or_assignment_children<const D: usize>(
    output: &Output<D>,
    tolerance: f64,
    windows: &[super::time_windows::NetworkWindows],
) -> Result<Option<Vec<NodeSnapshot<D>>>> {
    let node = &output.snapshot;
    let mut resources = node.resource_group_candidates(tolerance)?;
    resources.sort_by(|a, b| (a.value - 0.5).abs().total_cmp(&(b.value - 0.5).abs()));
    for candidate in resources {
        // Candidate generation is heuristic. If a child's exact schedule cannot
        // be reconstructed, neither child may prune: use another disjunction.
        // Attachment works on private forks and leaves this parent unchanged.
        if let Ok(down) = candidate.apply(node, false) {
            if let Ok(up) = candidate.apply(node, true) {
                return Ok(Some(vec![down, up]));
            }
        }
    }
    let assignments = node.fractional_assignments(tolerance)?;
    if let Some(c) = assignments
        .iter()
        .min_by(|a, b| (a.value - 0.5).abs().total_cmp(&(b.value - 0.5).abs()))
    {
        return Ok(Some(vec![
            node.branch_assignment(c.shared_row, c.network, false)?,
            node.branch_assignment(c.shared_row, c.network, true)?,
        ]));
    }
    for (variable, &(_, value)) in node
        .state
        .master_variables
        .iter()
        .zip(&output.master_values)
    {
        if !variable.integer {
            continue;
        }
        if fractional(value, tolerance) {
            return Ok(Some(vec![
                node.with_master_variable_bounds(
                    variable.id,
                    Bounds {
                        lower: f64::NEG_INFINITY,
                        upper: value.floor(),
                    },
                )?,
                node.with_master_variable_bounds(
                    variable.id,
                    Bounds {
                        lower: value.ceil(),
                        upper: f64::INFINITY,
                    },
                )?,
            ]));
        }
    }
    let positions: HashMap<_, _> = node
        .state
        .oracles
        .iter()
        .enumerate()
        .map(|(i, o)| (o.network.id, i))
        .collect();
    let mut multiplicities = vec![0.; node.state.oracles.len()];
    let mut activities: Vec<Vec<f64>> = node
        .state
        .oracles
        .iter()
        .map(|o| vec![0.; o.network.graph.arcs.len()])
        .collect();
    let mut add = |route: &NetworkRoute, quantity: f64| {
        if quantity == 0. {
            return;
        }
        let i = positions[&route.network];
        multiplicities[i] += quantity;
        for id in &route.route.arcs {
            activities[i][node.state.oracles[i].index[id]] += quantity;
        }
    };
    for (route, &quantity) in output.routes.iter().zip(&output.values) {
        add(route, quantity);
    }
    for fixed in &node.state.fixed_routes {
        add(&fixed.route, fixed.quantity);
    }
    for (i, oracle) in node.state.oracles.iter().enumerate() {
        let value = multiplicities[i];
        if value >= 0. && fractional(value, tolerance) {
            return Ok(Some(vec![
                node.with_multiplicity(
                    oracle.network.id,
                    Bounds {
                        lower: 0.,
                        upper: value.floor(),
                    },
                )?,
                node.with_multiplicity(
                    oracle.network.id,
                    Bounds {
                        lower: value.ceil(),
                        upper: f64::INFINITY,
                    },
                )?,
            ]));
        }
    }
    // Time-window selection counts are integer activities over window copies.
    if !windows.is_empty() {
        let flows: HashMap<_, _> = node
            .state
            .oracles
            .iter()
            .zip(&activities)
            .map(|(o, a)| (o.network.id, a.clone()))
            .collect();
        if let Some(children) =
            super::time_windows::selection_children(node, windows, &flows, tolerance)?
        {
            return Ok(Some(children));
        }
    }
    // Integer route quantities imply integer aggregate physical-arc activity,
    // including repeated traversals. Each split covers every integer value,
    // irrespective of numerical error in the LP point used to choose it.
    for (i, oracle) in node.state.oracles.iter().enumerate() {
        for (a, &value) in activities[i].iter().enumerate() {
            if value >= 0. && fractional(value, tolerance) {
                let mut arcs = vec![0.; oracle.network.graph.arcs.len()];
                arcs[a] = 1.;
                let coefficients = [RowCoefficients {
                    network: oracle.network.id,
                    arcs,
                    per_route: 0.,
                }];
                return Ok(Some(vec![
                    node.with_activity_bounds(
                        Bounds {
                            lower: 0.,
                            upper: value.floor(),
                        },
                        &coefficients,
                    )?,
                    node.with_activity_bounds(
                        Bounds {
                            lower: value.ceil(),
                            upper: f64::INFINITY,
                        },
                        &coefficients,
                    )?,
                ]));
            }
        }
    }
    if let Some(children) = super::time_windows::synchronized_children(output, windows, tolerance)?
    {
        return Ok(Some(children));
    }
    synchronization::children(output, tolerance)
}

#[cfg(test)]
#[path = "../../../../tests/unit/master/mp/network_branching_tests.rs"]
mod tests;
