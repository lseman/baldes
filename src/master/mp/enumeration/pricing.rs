//! Price the installed finite route universe.
use super::*;

pub(in crate::master::mp) fn price_universe<const D: usize>(
    state: &mut State<D>,
    duals: &[f64],
) -> Result<cg::Answer> {
    let mut columns = vec![];
    let mut minimum = f64::INFINITY;
    let mut correction = 0.;
    let mut gradient = state.rhs.clone();
    for oracle in &mut state.oracles {
        if oracle.multiplicity_upper() == 0. {
            continue;
        }
        oracle.calls += 1;

        for &tag in state.universe.as_ref().unwrap() {
            let route = &state.routes[tag];
            if route.network != oracle.network.id
                || route.route.arcs.iter().any(|id| {
                    oracle
                        .proved_zero_arcs
                        .get(oracle.index[id])
                        .copied()
                        .unwrap_or(false)
                })
            {
                continue;
            }
            let cap = state
                .route_upper_bounds
                .get(&tag)
                .copied()
                .unwrap_or(oracle.multiplicity_upper())
                .min(oracle.multiplicity_upper());
            if cap == 0. {
                continue;
            }
            let coefficients = entries(oracle, &route.route, state.shared_offset)?;
            let reduced = coefficients
                .iter()
                .fold(route.route.original_cost, |v, &(r, a)| v - duals[r] * a);
            if !reduced.is_finite() {
                return Err(invalid("enumerated reduced cost overflow"));
            }
            minimum = minimum.min(reduced);
            if reduced < 0. {
                correction += reduced * cap;
                for &(r, a) in &coefficients {
                    gradient[r] -= cap * a;
                }
            }
            columns.push(cg::Column {
                cost: route.route.original_cost,
                reduced_cost: reduced,
                entries: coefficients,
                acceptable: true,
                tag: tag as i32,
                identity: format!("{}:{}", route.network, route.route.identity()),
            });
        }
    }
    if let Some((static_correction, static_gradient)) =
        super::super::master_variables::correction(state, duals)
    {
        correction += static_correction;
        for (row, value) in static_gradient {
            gradient[row] -= value;
        }
        if correction.is_finite() && gradient.iter().all(|g| g.is_finite()) {
            state.correction = Some(correction);
            state.gradient = Some(gradient);
        } else {
            state.correction = None;
            state.gradient = None;
        }
    } else {
        state.correction = None;
        state.gradient = None;
    }
    Ok(cg::Answer {
        columns,
        minimum: if minimum == f64::INFINITY {
            0.
        } else {
            minimum
        },
        exact: true,
        bound_minimum: None,
    })
}
