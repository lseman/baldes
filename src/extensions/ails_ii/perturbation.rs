use crate::{extensions::common::validate_route, inputs::Instance, rng::Xoroshiro128Plus};
use std::time::Instant;

/// Batch removal and feasible reinsertion, preserving the original on failure.
pub(super) fn perturb(
    instance: &Instance,
    routes: &mut Vec<Vec<usize>>,
    rng: &mut Xoroshiro128Plus,
    strength: usize,
    sequence: bool,
    deadline: Instant,
) {
    let original = routes.clone();
    let nonempty: Vec<_> = routes
        .iter()
        .enumerate()
        .filter_map(|(i, r)| (!r.is_empty()).then_some(i))
        .collect();
    if nonempty.is_empty() {
        return;
    }
    let route_index = nonempty[rng.next() as usize % nonempty.len()];
    let position = rng.next() as usize % routes[route_index].len();
    let center = routes[route_index][position];
    let count = strength.max(1).min(routes.iter().map(Vec::len).sum());
    let mut removed = Vec::with_capacity(count);
    if sequence {
        while removed.len() < count && position < routes[route_index].len() {
            removed.push(routes[route_index].remove(position));
        }
    } else {
        let mut ranked: Vec<_> = routes.iter().flatten().copied().collect();
        ranked.sort_by(|&a, &b| {
            instance
                .distance(center, a, instance.rounding)
                .total_cmp(&instance.distance(center, b, instance.rounding))
        });
        for customer in ranked.into_iter().take(count) {
            for route in routes.iter_mut() {
                if let Some(pos) = route.iter().position(|&c| c == customer) {
                    route.remove(pos);
                    break;
                }
            }
            removed.push(customer);
        }
    }
    let by_distance = rng.next() & 1 == 0;
    for customer in removed {
        let original_gap = original.iter().enumerate().find_map(|(r, route)| {
            route.iter().position(|&v| v == customer).map(|pos| {
                (
                    r,
                    if pos == 0 { 0 } else { route[pos - 1] },
                    route.get(pos + 1).copied().unwrap_or(0),
                )
            })
        });
        let mut best: Option<(usize, usize, bool, f64)> = None;
        let mut fallback: Option<(usize, usize, f64)> = None;
        for (r, route) in routes.iter().enumerate() {
            for pos in 0..=route.len() {
                if Instant::now() >= deadline {
                    *routes = original;
                    return;
                }
                let mut candidate = route.clone();
                candidate.insert(pos, customer);
                let before = if pos == 0 { 0 } else { route[pos - 1] };
                let after = route.get(pos).copied().unwrap_or(0);
                let d = |a, b| instance.distance(a, b, instance.rounding);
                let score = if by_distance {
                    d(before, customer).min(d(customer, after))
                } else {
                    d(before, customer) + d(customer, after) - d(before, after)
                };
                if fallback.map_or(true, |(_, _, value)| score < value) {
                    fallback = Some((r, pos, score));
                }
                if validate_route(instance, &candidate).is_err() {
                    continue;
                }
                let unchanged = original_gap == Some((r, before, after));
                if best.map_or(true, |(_, _, old_unchanged, value)| {
                    (!unchanged && old_unchanged) || (unchanged == old_unchanged && score < value)
                }) {
                    best = Some((r, pos, unchanged, score));
                }
            }
        }
        if let Some((r, pos, _, _)) = best {
            routes[r].insert(pos, customer);
        } else if let Some((r, pos, _)) = fallback {
            routes[r].insert(pos, customer);
        } else {
            *routes = original;
            return;
        }
    }
    if crate::extensions::common::validate_plan(instance, routes).is_err()
        && !super::repair::repair_plan(instance, routes, deadline)
    {
        *routes = original;
    }
}
