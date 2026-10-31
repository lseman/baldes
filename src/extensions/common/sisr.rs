//! Spatially adjacent string removals and feasible greedy recreation.
use super::{validate_plan, validate_route};
use crate::{inputs::Instance, rng::Xoroshiro128Plus};
use std::time::Instant;

/// Ruin spatially close strings from distinct routes, then reinsert customers.
/// Returns false and restores the input if recreation fails or the deadline expires.
pub fn ruin_recreate(
    instance: &Instance,
    routes: &mut Vec<Vec<usize>>,
    rng: &mut Xoroshiro128Plus,
    strength: usize,
    deadline: Instant,
) -> bool {
    if validate_plan(instance, routes).is_err() {
        return false;
    }
    let original = routes.clone();
    let mut owner = vec![None; instance.customers.len()];
    for (route, customers) in routes.iter().enumerate() {
        for (position, &customer) in customers.iter().enumerate() {
            owner[customer] = Some((route, position));
        }
    }
    let all: Vec<_> = routes.iter().flatten().copied().collect();
    if all.is_empty() {
        return false;
    }
    let seed = all[rng.next() as usize % all.len()];
    let mut nearby = all;
    nearby.sort_by(|&a, &b| {
        instance
            .distance(seed, a, instance.rounding)
            .total_cmp(&instance.distance(seed, b, instance.rounding))
    });
    let mut ruined = vec![false; routes.len()];
    let mut absent = Vec::new();
    let target = strength.max(1).min(nearby.len());
    for customer in nearby {
        if Instant::now() >= deadline {
            *routes = original;
            return false;
        }
        if absent.len() >= target {
            break;
        }
        let Some((index, pos)) = owner[customer] else {
            continue;
        };
        if ruined[index] {
            continue;
        }
        let route = &mut routes[index];
        let max_len = route.len().min(target - absent.len()).min(10);
        let length = 1 + rng.next() as usize % max_len;
        let split = length >= 2 && length < route.len() && rng.next() % 2 == 0;
        let gap = if split {
            let mut gap = 1;
            let max_gap = route.len() - length;
            while gap < max_gap && rng.next() % 100 != 0 {
                gap += 1;
            }
            gap
        } else {
            0
        };
        let span = length + gap;
        let earliest = pos.saturating_add(1).saturating_sub(span);
        let latest = pos.min(route.len() - span);
        let start = earliest + rng.next() as usize % (latest - earliest + 1);
        if split {
            let left = 1 + rng.next() as usize % (length - 1);
            absent.extend(route.drain(start + left + gap..start + span));
            absent.extend(route.drain(start..start + left));
        } else {
            absent.extend(route.drain(start..start + length));
        }
        ruined[index] = true;
    }
    match rng.next() % 11 {
        0..=3 => rng.shuffle(&mut absent),
        4..=7 => absent.sort_by(|&a, &b| {
            instance.customers[b]
                .demand
                .total_cmp(&instance.customers[a].demand)
        }),
        8..=9 => absent.sort_by(|&a, &b| {
            instance
                .distance(0, b, instance.rounding)
                .total_cmp(&instance.distance(0, a, instance.rounding))
        }),
        _ => absent.sort_by(|&a, &b| {
            instance
                .distance(0, a, instance.rounding)
                .total_cmp(&instance.distance(0, b, instance.rounding))
        }),
    }
    // Blinks occasionally skip a feasible insertion. Always retain a fallback
    // so a customer is not stranded solely by random skipping.
    for customer in absent {
        let mut best: Option<(usize, usize, f64)> = None;
        let mut fallback: Option<(usize, usize, f64)> = None;
        let mut route_order: Vec<_> = (0..routes.len()).collect();
        rng.shuffle(&mut route_order);
        for r in route_order {
            let route = &routes[r];
            for pos in 0..=route.len() {
                if Instant::now() >= deadline {
                    *routes = original;
                    return false;
                }
                let mut trial = route.clone();
                trial.insert(pos, customer);
                if validate_route(instance, &trial).is_err() {
                    continue;
                }
                let previous = if pos == 0 { 0 } else { route[pos - 1] };
                let next = route.get(pos).copied().unwrap_or(0);
                let d = |a, b| instance.distance(a, b, instance.rounding);
                let delta = d(previous, customer) + d(customer, next) - d(previous, next);
                if fallback.map_or(true, |(_, _, value)| delta < value) {
                    fallback = Some((r, pos, delta));
                }
                if rng.next() % 100 >= 1 && best.map_or(true, |(_, _, value)| delta < value) {
                    best = Some((r, pos, delta));
                }
            }
        }
        if let Some((route, pos, _)) = best.or(fallback) {
            routes[route].insert(pos, customer);
        } else if (instance.vehicles == 0
            || routes.iter().filter(|route| !route.is_empty()).count() < instance.vehicles)
            && validate_route(instance, &[customer]).is_ok()
        {
            routes.push(vec![customer]);
        } else {
            *routes = original;
            return false;
        }
    }
    if validate_plan(instance, routes).is_err() {
        *routes = original;
        return false;
    }
    true
}
