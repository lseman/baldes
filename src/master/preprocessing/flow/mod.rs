//! Canonical arc-flow construction shared by all master input registrations.
use super::{Constraint, RowSense};
use crate::Graph;
use std::collections::BTreeMap;

/// Coalesce only exact coefficients; rounded cancellation cannot forbid an arc.
pub(crate) fn row(entries: Vec<(usize, f64)>, sense: RowSense, rhs: f64) -> Option<Constraint> {
    let mut sums = BTreeMap::<usize, crate::master::numerics::ExactSum>::new();
    for (variable, coefficient) in entries {
        sums.entry(variable).or_default().add(coefficient)?;
    }
    let mut entries = vec![];
    for (variable, mut sum) in sums {
        let coefficient = sum.upper()?;
        sum.add(-coefficient)?;
        if sum.sign() != 0. {
            return None;
        }
        if coefficient != 0. {
            entries.push((variable, coefficient));
        }
    }
    Some(Constraint {
        entries,
        sense,
        rhs,
    })
}

/// Outgoing minus incoming flow. A route-count variable balances source/sink;
/// without one, only interior conservation rows are needed. Self loops cancel.
pub(crate) fn conservation<const D: usize>(
    graph: &Graph<D>,
    start: usize,
    quantity: Option<usize>,
) -> Vec<Constraint> {
    let mut flows = vec![vec![]; graph.vertices.len()];
    for (position, arc) in graph.arcs.iter().enumerate() {
        flows[arc.tail].push((start + position, 1.));
        flows[arc.head].push((start + position, -1.));
    }
    flows
        .into_iter()
        .enumerate()
        .filter_map(|(vertex, mut terms)| {
            if let Some(quantity) = quantity {
                if vertex == graph.source {
                    terms.push((quantity, -1.));
                }
                if vertex == graph.sink {
                    terms.push((quantity, 1.));
                }
            } else if vertex == graph.source || vertex == graph.sink {
                return None;
            }
            row(terms, RowSense::Equal, 0.)
        })
        .collect()
}

#[cfg(test)]
#[path = "../../../../tests/unit/master/preprocessing/flow_tests.rs"]
mod tests;
