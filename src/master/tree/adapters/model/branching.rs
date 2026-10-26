//! Integer-valid disjunctions on original quantities, never on artificial LPs.
use super::*;
use crate::master::preprocessing::Bounds;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum ChildOrder {
    LowerThenUpper,
    UpperThenLower,
    Custom,
}

struct BranchChildren<const D: usize> {
    nodes: Vec<NodeSnapshot<D>>,
    order: ChildOrder,
}

fn floor_ceiling<const D: usize>(nodes: Vec<NodeSnapshot<D>>) -> BranchChildren<D> {
    BranchChildren {
        nodes,
        order: ChildOrder::LowerThenUpper,
    }
}

fn custom<const D: usize>(nodes: Vec<NodeSnapshot<D>>) -> BranchChildren<D> {
    BranchChildren {
        nodes,
        order: ChildOrder::Custom,
    }
}

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
    let mut branches = children_inner(output, tolerance, policy, windows, variables)?;
    if let Some(branches) = &mut branches {
        for child in &mut branches.nodes {
            child.state.node_depth = output.snapshot.state.node_depth.saturating_add(1);
        }
    }
    Ok(branches.map(|branches| branches.nodes))
}

/// Generate the ordinary fallback disjunction and retain enough information
/// to reproduce native upper-first admission for conventional floor/ceiling
/// splits. Custom disjunction order is left to its generator.
pub(super) fn phased_children<const D: usize>(
    output: &Output<D>,
    tolerance: f64,
    windows: &[super::time_windows::NetworkWindows],
) -> Result<Option<(Vec<NodeSnapshot<D>>, ChildOrder)>> {
    let branches = children_inner(
        output,
        tolerance,
        NetworkBranchingPolicy::ResourceFirst,
        windows,
        &[],
    )?;
    Ok(branches.map(|mut branches| {
        for child in &mut branches.nodes {
            child.state.node_depth = output.snapshot.state.node_depth.saturating_add(1);
        }
        (branches.nodes, branches.order)
    }))
}

fn children_inner<const D: usize>(
    output: &Output<D>,
    tolerance: f64,
    policy: NetworkBranchingPolicy,
    windows: &[super::time_windows::NetworkWindows],
    variables: &[MappedBranchVariable],
) -> Result<Option<BranchChildren<D>>> {
    if policy == NetworkBranchingPolicy::MultiplicityFirst {
        if let Some(children) = multiplicity_children(output, tolerance)? {
            return Ok(Some(BranchChildren {
                nodes: children,
                order: ChildOrder::UpperThenLower,
            }));
        }
    }
    let mut branches = if let Some(mapped) = model_mapping::children(output, variables, tolerance)?
    {
        Some(BranchChildren {
            nodes: mapped,
            order: ChildOrder::LowerThenUpper,
        })
    } else {
        resource_or_assignment_children(output, tolerance, windows)?
    };
    if policy == NetworkBranchingPolicy::MultiplicityFirst {
        if let Some(branches) = &mut branches {
            branches.nodes.reverse();
            branches.order = match branches.order {
                ChildOrder::LowerThenUpper => ChildOrder::UpperThenLower,
                ChildOrder::UpperThenLower => ChildOrder::LowerThenUpper,
                ChildOrder::Custom => ChildOrder::Custom,
            };
        }
    }
    Ok(branches)
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
) -> Result<Option<BranchChildren<D>>> {
    let node = &output.snapshot;
    let mut resources = node.resource_group_candidates(tolerance)?;
    resources.sort_by(|a, b| (a.value - 0.5).abs().total_cmp(&(b.value - 0.5).abs()));
    for candidate in resources {
        // Candidate generation is heuristic. If a child's exact schedule cannot
        // be reconstructed, neither child may prune: use another disjunction.
        // Attachment works on private forks and leaves this parent unchanged.
        if let Ok(down) = candidate.apply(node, false) {
            if let Ok(up) = candidate.apply(node, true) {
                return Ok(Some(floor_ceiling(vec![down, up])));
            }
        }
    }
    let assignments = node.fractional_assignments(tolerance)?;
    if let Some(c) = assignments
        .iter()
        .min_by(|a, b| (a.value - 0.5).abs().total_cmp(&(b.value - 0.5).abs()))
    {
        return Ok(Some(floor_ceiling(vec![
            node.branch_assignment(c.shared_row, c.network, false)?,
            node.branch_assignment(c.shared_row, c.network, true)?,
        ])));
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
            return Ok(Some(floor_ceiling(vec![
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
            ])));
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
            return Ok(Some(floor_ceiling(vec![
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
            ])));
        }
    }
    // On window-split networks, arc flows are branched on summed over each
    // input arc's copies, then on window selection counts (Li et al., 2020).
    let flows: HashMap<_, _> = node
        .state
        .oracles
        .iter()
        .zip(&activities)
        .map(|(o, a)| (o.network.id, a.clone()))
        .collect();
    if let Some(children) = super::time_windows::arc_children(node, windows, &flows, tolerance)? {
        return Ok(Some(floor_ceiling(children)));
    }
    let copied: HashSet<(i32, i32)> = windows
        .iter()
        .flat_map(|w| {
            w.split
                .arcs
                .iter()
                .filter(|a| {
                    w.split
                        .arcs
                        .iter()
                        .filter(|b| b.original == a.original)
                        .count()
                        > 1
                })
                .map(move |a| (w.network, a.arc))
        })
        .collect();
    // Integer route quantities imply integer aggregate physical-arc activity,
    // including repeated traversals. Each split covers every integer value,
    // irrespective of numerical error in the LP point used to choose it.
    for (i, oracle) in node.state.oracles.iter().enumerate() {
        for (a, &value) in activities[i].iter().enumerate() {
            if copied.contains(&(oracle.network.id, oracle.network.graph.arcs[a].id)) {
                continue;
            }
            if value >= 0. && fractional(value, tolerance) {
                let mut arcs = vec![0.; oracle.network.graph.arcs.len()];
                arcs[a] = 1.;
                let coefficients = [RowCoefficients {
                    network: oracle.network.id,
                    arcs,
                    per_route: 0.,
                }];
                return Ok(Some(floor_ceiling(vec![
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
                ])));
            }
        }
    }
    if let Some(children) =
        super::time_windows::selection_children(node, windows, &flows, tolerance)?
    {
        return Ok(Some(floor_ceiling(children)));
    }
    if let Some(children) =
        super::time_windows::multiple_window_children(output, windows, tolerance)?
    {
        return Ok(Some(custom(children)));
    }
    if let Some(children) = super::time_windows::conflict_children(output, windows, tolerance)? {
        return Ok(Some(custom(children)));
    }
    Ok(synchronization::children(output, tolerance)?.map(custom))
}

#[cfg(test)]
#[path = "../../../../../tests/unit/master/model/network_branching_tests.rs"]
mod tests;
