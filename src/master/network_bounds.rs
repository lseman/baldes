//! Conservative original-graph walk relaxation for a network pricing batch.
//! Negative cycles that can reach the sink leave the certificate unknown.
use super::*;

use crate::master::rounding::{add_down as add, multiply_down as multiply};

fn subtract_product(value: f64, a: f64, b: f64) -> Option<f64> {
    // Negate an input so the subtraction itself is rounded downward.
    add(value, multiply(-a, b)?)
}

fn network_minimum<const D: usize>(
    oracle: &Oracle<D>,
    offset: usize,
    duals: &[f64],
) -> Option<Option<f64>> {
    let graph = &oracle.network.graph;
    // Only cycles on a source-to-sink walk can affect the relaxation.
    let mut reaches_sink = vec![false; graph.vertices.len()];
    reaches_sink[graph.sink] = true;
    for _ in 0..graph.vertices.len() {
        let mut changed = false;
        for arc in &graph.arcs {
            if reaches_sink[arc.head] && !reaches_sink[arc.tail] {
                reaches_sink[arc.tail] = true;
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    let mut weights = vec![];
    for (arc, coefficients) in graph.arcs.iter().zip(&oracle.network.arc_entries) {
        let mut weight = arc.cost;
        for &(row, coefficient) in coefficients {
            weight = subtract_product(weight, duals[offset + row], coefficient)?;
        }
        weights.push(weight);
    }
    let mut distances = vec![None; graph.vertices.len()];
    distances[graph.source] = Some(0.);
    for pass in 0..graph.vertices.len() {
        let mut changed = false;
        for (arc, &weight) in graph.arcs.iter().zip(&weights) {
            if !reaches_sink[arc.head] {
                continue;
            }
            if let Some(tail) = distances[arc.tail] {
                let candidate = add(tail, weight)?;
                if distances[arc.head].map_or(true, |previous| candidate < previous) {
                    distances[arc.head] = Some(candidate);
                    changed = true;
                }
            }
        }
        if !changed {
            break;
        }
        if pass + 1 == graph.vertices.len() {
            return None;
        }
    }
    let mut minimum = match distances[graph.sink] {
        Some(value) => value,
        None => return Some(None),
    };
    for &(row, coefficient) in &oracle.route_entries {
        minimum = subtract_product(minimum, duals[row], coefficient)?;
    }
    for &(row, coefficient) in &oracle.network.route_entries {
        minimum = subtract_product(minimum, duals[offset + row], coefficient)?;
    }
    Some(Some(minimum))
}

pub(super) fn certificate<const D: usize>(
    state: &State<D>,
    master: &ClpMaster,
    duals: &[f64],
) -> Option<f64> {
    if !exact_fixed_reduction(state, master).unwrap_or(false) {
        return None;
    }
    if duals.len() != state.rhs.len() || duals.iter().any(|value| !value.is_finite()) {
        return None;
    }
    let mut value = master.objective_offset;
    for ((&dual, &rhs), sense) in duals.iter().zip(&state.rhs).zip(&master.senses) {
        if (*sense == RowSense::LessEqual && dual > 0.)
            || (*sense == RowSense::GreaterEqual && dual < 0.)
        {
            return None;
        }
        value = add(value, multiply(dual, rhs)?)?;
    }
    for oracle in &state.oracles {
        let limits = oracle.network.multiplicity;
        if limits.upper == 0. {
            continue;
        }
        match network_minimum(oracle, state.shared_offset, duals)? {
            None if limits.lower > 0. => return None,
            None => {}
            Some(minimum) => {
                let quantity = if minimum < 0. {
                    limits.upper
                } else {
                    limits.lower
                };
                value = add(value, multiply(minimum, quantity)?)?;
            }
        }
    }
    Some(value)
}

// Full original-model contributions shared by witness replay and validation of
// fixed-route reductions. Never trust cached rounded path costs or resources.
type Contributions = (
    Vec<crate::master::root::ExactSum>,
    crate::master::root::ExactSum,
    Vec<crate::master::root::ExactSum>,
);
pub(super) fn exact_contributions<const D: usize>(
    state: &State<D>,
    routes: &[FixedRoute],
) -> Result<Option<Contributions>> {
    use crate::master::root::ExactSum;
    let mut rows = vec![ExactSum::default(); state.shared_rows.len()];
    let mut objective = ExactSum::default();
    let mut counts = vec![ExactSum::default(); state.oracles.len()];
    for fixed in routes {
        let index = state
            .oracles
            .iter()
            .position(|o| o.network.id == fixed.route.network)
            .ok_or_else(|| invalid("unknown witness network"))?;
        let oracle = &state.oracles[index];
        if !super::lifecycle::integer_route_feasible(oracle, &fixed.route.route)? {
            return Ok(None);
        }
        if counts[index].add(fixed.quantity).is_none() {
            return Ok(None);
        }
        for &(row, coefficient) in &oracle.network.route_entries {
            if rows[row].product(coefficient, fixed.quantity).is_none() {
                return Ok(None);
            }
        }
        for id in &fixed.route.route.arcs {
            let arc = oracle.index[id];
            if objective
                .product(oracle.network.graph.arcs[arc].cost, fixed.quantity)
                .is_none()
            {
                return Ok(None);
            }
            for &(row, coefficient) in &oracle.network.arc_entries[arc] {
                if rows[row].product(coefficient, fixed.quantity).is_none() {
                    return Ok(None);
                }
            }
        }
    }
    Ok(Some((rows, objective, counts)))
}

// A rounded RHS or offset after fixing is not an exact model reduction. Keep
// its original-domain certificate unknown instead of certifying the rounded LP.
fn exact_fixed_reduction<const D: usize>(state: &State<D>, master: &ClpMaster) -> Result<bool> {
    if state.fixed_routes.is_empty() {
        return Ok(true);
    }
    let Some((rows, mut objective, counts)) = exact_contributions(state, &state.fixed_routes)?
    else {
        return Ok(false);
    };
    if objective.add(-master.objective_offset).is_none() || objective.sign() != 0. {
        return Ok(false);
    }
    for (i, mut activity) in rows.into_iter().enumerate() {
        if activity.add(state.rhs[state.shared_offset + i]).is_none()
            || activity.add(-state.shared_rows[i].rhs).is_none()
            || activity.sign() != 0.
        {
            return Ok(false);
        }
    }
    for (count, oracle) in counts.iter().zip(&state.oracles) {
        for &(row, _) in &oracle.route_entries {
            let original = if row == oracle.route_entries[0].0 {
                oracle.total_multiplicity.upper
            } else {
                oracle.total_multiplicity.lower
            };
            let mut difference = count.clone();
            if difference.add(state.rhs[row]).is_none()
                || difference.add(-original).is_none()
                || difference.sign() != 0.
            {
                return Ok(false);
            }
        }
        let mut upper = count.clone();
        if upper.add(oracle.network.multiplicity.upper).is_none()
            || upper.add(-oracle.total_multiplicity.upper).is_none()
            || upper.sign() != 0.
        {
            return Ok(false);
        }
        let mut lower = count.clone();
        if lower.add(oracle.network.multiplicity.lower).is_none()
            || lower.add(-oracle.total_multiplicity.lower).is_none()
        {
            return Ok(false);
        }
        if oracle.network.multiplicity.lower > 0. && lower.sign() != 0.
            || oracle.network.multiplicity.lower == 0. && lower.sign() < 0.
        {
            return Ok(false);
        }
    }
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rounded_fixed_rhs_cannot_supply_a_new_certificate() {
        let mut graph = Graph::<2>::new(3, 1).unwrap();
        graph.add_arc(0, 1, 1.).unwrap();
        graph.add_arc(1, 2, 0.).unwrap();
        let problem = Problem {
            rows: vec![Row {
                sense: RowSense::Equal,
                rhs: 1.,
            }],
            networks: vec![Network {
                id: 1,
                graph,
                multiplicity: Window::new(0., 20.).unwrap(),
                arc_entries: vec![vec![(0, 0.1)], vec![]],
                route_entries: vec![],
            }],
        };
        let mut options = Options::default();
        options.cg.stabilize = false;
        options.cg.artificial_cost = 100.;
        let root = solve(problem, options).unwrap();
        assert_eq!(root.master.status, cg::Status::Optimal);
        let node = root.snapshot.fix_route(0, 1.).unwrap();
        // f64(0.9) + f64(0.1) differs from one in exact arithmetic.
        assert!(!exact_fixed_reduction(&node.state, &node.master).unwrap());
        assert!(certificate(&node.state, &node.master, &root.duals).is_none());
    }
}
