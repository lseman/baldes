//! Complete-domain column caps and their Lagrangian compensation.
use super::*;
use crate::core::rounding::{add_down, multiply_down};

pub(super) fn upper<const D: usize>(state: &State<D>, tag: usize) -> Option<f64> {
    let route = state.routes.get(tag)?;
    let oracle = state
        .oracles
        .iter()
        .find(|o| o.network.id == route.network)?;
    let value = state
        .route_upper_bounds
        .get(&tag)
        .copied()
        .unwrap_or(oracle.multiplicity_upper())
        .min(oracle.multiplicity_upper());
    (!value.is_nan() && value >= 0.).then_some(value)
}

fn reduced<const D: usize>(
    state: &State<D>,
    oracle: &Oracle<D>,
    route: &Route,
    duals: &[f64],
) -> Option<f64> {
    if oracle.network.graph.has_coupled_resources() {
        return None;
    }
    let mut value = 0.;
    for id in &route.arcs {
        let index = oracle.index[id];
        value = add_down(value, oracle.network.graph.arcs[index].cost)?;
        for &(r, a) in &oracle.network.arc_entries[index] {
            value = add_down(value, multiply_down(-duals[state.shared_offset + r], a)?)?;
        }
    }
    for &(r, a) in &oracle.route_entries {
        value = add_down(value, multiply_down(-duals[r], a)?)?;
    }
    for &(r, a) in oracle.network.route_entries.iter() {
        value = add_down(value, multiply_down(-duals[state.shared_offset + r], a)?)?;
    }
    for (r, a) in super::nonlinear::entries(oracle, route).ok()? {
        value = add_down(value, multiply_down(-duals[state.shared_offset + r], a)?)?;
    }
    Some(value)
}

/// min rc_j*x_j over each independent box [0,u_j]. All original rows,
/// including multiplicity rows, are dualized by the caller. The relaxation
/// need not impose network total multiplicity a second time. At a bounded LP
/// optimum this includes upper-active columns with negative reduced costs.
pub(super) fn certificate<const D: usize>(state: &State<D>, duals: &[f64]) -> Option<f64> {
    let universe = state.universe.as_ref()?;
    let mut correction = 0.;
    for oracle in &state.oracles {
        let mut available = false;
        for &tag in universe {
            let route = &state.routes[tag];
            if route.network != oracle.network.id {
                continue;
            }
            if Instant::now() >= state.deadline {
                return None;
            }
            let cap = upper(state, tag)?;
            if cap == 0.
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
            available = true;
            let value = reduced(state, oracle, &route.route, duals)?;
            if value < 0. {
                correction = add_down(correction, multiply_down(value, cap)?)?;
            }
        }
        if !available && oracle.network.multiplicity.lower > 0. {
            return None;
        }
    }
    Some(correction)
}

#[cfg(test)]
#[path = "../../../tests/unit/master/mp/route_bounds/tests.rs"]
mod tests;
