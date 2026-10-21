//! Stable original-coordinate, qualified-column and physical-routing splits.
use super::*;
fn fractional(value: f64, tolerance: f64) -> bool {
    value.is_finite()
        && value.abs() < 9_007_199_254_740_992.
        && (value - value.round()).abs() > tolerance
}
fn split(value: f64, nonnegative: bool) -> [Bounds; 2] {
    [
        Bounds {
            lower: if nonnegative { 0. } else { f64::NEG_INFINITY },
            upper: value.floor(),
        },
        Bounds {
            lower: value.ceil(),
            upper: f64::INFINITY,
        },
    ]
}
pub(super) fn children<const D: usize>(
    session: &MultiImport<D>,
    output: &Output<D>,
    tolerance: f64,
) -> Result<Option<Vec<MultiImport<D>>>> {
    let compiled = lifecycle::compile_definition(&session.definition, Some(&session.context))?;
    let values = compiled.registry.values(&output.master_values)?;
    for &(id, value) in &values.original {
        if session
            .definition
            .variables
            .iter()
            .any(|v| v.id == id && v.integer)
            && fractional(value, tolerance)
        {
            return split(value, false)
                .into_iter()
                .map(|b| session.with_variable_bounds(id, b))
                .collect::<Result<Vec<_>>>()
                .map(Some);
        }
    }
    for value in &values.columns {
        let group = session
            .definition
            .subproblems
            .iter()
            .find(|g| g.id == value.subproblem)
            .unwrap();
        if group
            .columns
            .iter()
            .any(|c| c.id == value.column && c.integer)
            && fractional(value.value, tolerance)
        {
            return split(value.value, true)
                .into_iter()
                .map(|b| session.with_column_bounds(value.subproblem, value.column, b))
                .collect::<Result<Vec<_>>>()
                .map(Some);
        }
    }
    for network in &session.definition.problem.networks {
        let mut multiplicity = 0.;
        let mut activities = vec![0.; network.graph.arcs.len()];
        let index: HashMap<_, _> = network
            .graph
            .arcs
            .iter()
            .enumerate()
            .map(|(i, a)| (a.id, i))
            .collect();
        for (route, quantity) in output
            .routes
            .iter()
            .zip(&output.values)
            .chain(output.fixed_routes.iter().map(|f| (&f.route, &f.quantity)))
        {
            if route.network != network.id {
                continue;
            }
            multiplicity += quantity;
            for id in &route.route.arcs {
                activities[index[id]] += quantity;
            }
        }
        if multiplicity >= 0. && fractional(multiplicity, tolerance) {
            return split(multiplicity, true)
                .into_iter()
                .map(|b| session.with_routing_multiplicity(network.id, b))
                .collect::<Result<Vec<_>>>()
                .map(Some);
        }
        for (a, &value) in activities.iter().enumerate() {
            if value >= 0. && fractional(value, tolerance) {
                let mut arcs = vec![0.; activities.len()];
                arcs[a] = 1.;
                let c = [RowCoefficients {
                    network: network.id,
                    arcs,
                    per_route: 0.,
                }];
                return Ok(Some(vec![
                    session.with_row(
                        Row {
                            sense: RowSense::LessEqual,
                            rhs: value.floor(),
                        },
                        &c,
                        &[],
                    )?,
                    session.with_row(
                        Row {
                            sense: RowSense::GreaterEqual,
                            rhs: value.ceil(),
                        },
                        &c,
                        &[],
                    )?,
                ]));
            }
        }
    }
    Ok(None)
}
