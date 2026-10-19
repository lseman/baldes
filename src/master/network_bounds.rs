//! Conservative original-graph walk relaxation for a network pricing batch.
//! Negative cycles that can reach the sink leave the certificate unknown.
use super::*;

use crate::master::rounding::{add_down as add, multiply_down as multiply};

fn subtract_product(value: f64, a: f64, b: f64) -> Option<f64> {
    // Negate an input so the subtraction itself is rounded downward.
    add(value, multiply(-a, b)?)
}

fn network_minimum(oracle: &Oracle, offset: usize, duals: &[f64]) -> Option<Option<f64>> {
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

pub(super) fn certificate(state: &State, master: &ClpMaster, duals: &[f64]) -> Option<f64> {
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
