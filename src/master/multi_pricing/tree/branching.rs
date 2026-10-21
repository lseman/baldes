//! Integer-valid disjunctions on original quantities, never on artificial LPs.
use super::*;
use crate::master::preprocessing::Bounds;

fn fractional(value: f64, tolerance: f64) -> bool {
    value.is_finite()
        && value.abs() < 9_007_199_254_740_992.
        && (value - value.round()).abs() > tolerance
}

pub(super) fn children<const D: usize>(
    output: &Output<D>,
    tolerance: f64,
) -> Result<Option<Vec<NodeSnapshot<D>>>> {
    let node = &output.snapshot;
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
    Ok(None)
}
