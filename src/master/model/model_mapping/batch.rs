//! Share exact unit-arc activity evaluation across registered variables.
use super::{MappedBranchVariable, Output};
use crate::master::numerics::ExactSum;
use std::collections::{HashMap, HashSet};

fn unit_arc(variable: &MappedBranchVariable) -> Option<(i32, i32)> {
    if !variable.masters.is_empty() {
        return None;
    }
    let [network] = variable.networks.as_slice() else {
        return None;
    };
    let [(arc, coefficient)] = network.arcs.as_slice() else {
        return None;
    };
    (network.per_route == 0. && network.activities.is_empty() && *coefficient == 1.)
        .then_some((network.network, *arc))
}

fn scalar_values<const D: usize>(
    variables: &[MappedBranchVariable],
    output: &Output<D>,
    skip: &impl Fn(&MappedBranchVariable) -> bool,
) -> Vec<Option<f64>> {
    variables
        .iter()
        .map(|variable| {
            if skip(variable) {
                None
            } else {
                variable.value(output)
            }
        })
        .collect()
}

/// Keep one exact expansion per distinct registered unit arc. Each expansion
/// receives the same route products, in the same route/fixed-copy order, as the
/// scalar evaluator. General expressions retain their original evaluation.
pub(in crate::master::model) fn values<const D: usize>(
    variables: &[MappedBranchVariable],
    output: &Output<D>,
) -> Vec<Option<f64>> {
    values_except(variables, output, |_| false)
}

/// Evaluate registered mappings except those replaced by a separate projector.
/// Skipped entries retain their position and return `None` without evaluation.
pub(in crate::master::model) fn values_except<const D: usize>(
    variables: &[MappedBranchVariable],
    output: &Output<D>,
    skip: impl Fn(&MappedBranchVariable) -> bool,
) -> Vec<Option<f64>> {
    // Tiny registries/points do not amortize the shared lookup structures.
    if variables.len() < 8 || output.routes.len() + output.snapshot.state.fixed_routes.len() < 16 {
        return scalar_values(variables, output, &skip);
    }
    let mut slots = HashMap::new();
    let mut keys = vec![];
    let destinations: Vec<_> = variables
        .iter()
        .map(|variable| {
            if skip(variable) {
                return None;
            }
            unit_arc(variable).map(|key| {
                let next = slots.len();
                *slots.entry(key).or_insert_with(|| {
                    keys.push(key);
                    next
                })
            })
        })
        .collect();
    if slots.is_empty() {
        return scalar_values(variables, output, &skip);
    }
    let mut sums: Vec<_> = (0..slots.len())
        .map(|_| Some(ExactSum::default()))
        .collect();
    let mut counts = vec![0_usize; slots.len()];
    let mut activated = vec![false; slots.len()];
    let mut active = vec![];
    let mut invalid_networks = HashSet::new();
    for (route, quantity) in output
        .routes
        .iter()
        .zip(&output.values)
        .map(|(route, &quantity)| (route, quantity))
        .chain(
            output
                .snapshot
                .state
                .fixed_routes
                .iter()
                .map(|fixed| (&fixed.route, fixed.quantity)),
        )
    {
        if quantity == 0. {
            continue;
        }
        // Even an absent unit arc contributes product(nonfinite, 0), which
        // declines the scalar value. Invalidate every unit arc of this network.
        if !quantity.is_finite() {
            invalid_networks.insert(route.network);
            continue;
        }
        for arc in &route.route.arcs {
            if let Some(&slot) = slots.get(&(route.network, *arc)) {
                if !activated[slot] {
                    activated[slot] = true;
                    active.push(slot);
                }
                counts[slot] = counts[slot].saturating_add(1);
            }
        }
        for &slot in &active {
            if keys[slot].0 != route.network {
                continue;
            }
            let count = counts[slot];
            counts[slot] = 0;
            if count as u128 > 9_007_199_254_740_992_u128 {
                sums[slot] = None;
            } else if let Some(sum) = &mut sums[slot] {
                // Before activation, finite zero-coefficient products leave an
                // empty expansion unchanged. After activation preserve every
                // product, including zero, and the scalar collapse semantics.
                if sum.product(quantity, count as f64).is_none() {
                    sums[slot] = None;
                }
            }
        }
    }
    let values: Vec<_> = sums
        .into_iter()
        .enumerate()
        .map(|(slot, sum)| {
            if invalid_networks.contains(&keys[slot].0) {
                None
            } else {
                sum.and_then(|sum| sum.upper())
            }
        })
        .collect();
    variables
        .iter()
        .zip(destinations)
        .map(|(variable, destination)| {
            if skip(variable) {
                return None;
            }
            match destination {
                Some(slot) => values[slot],
                None => variable.value(output),
            }
        })
        .collect()
}

#[cfg(test)]
#[path = "../../../../tests/unit/master/model/mapped_batch_tests.rs"]
mod tests;
