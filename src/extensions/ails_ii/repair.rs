//! Feasibility repair for complete plans after perturbation.
use crate::{extensions::common::validate_plan, inputs::Instance};
use std::time::Instant;

fn measures(instance: &Instance, routes: &[Vec<usize>]) -> (f64, f64) {
    let depot = &instance.customers[0];
    let mut violation = 0.0;
    let mut cost = 0.0;
    for route in routes.iter().filter(|r| !r.is_empty()) {
        let mut load = 0.0;
        let mut time = depot.ready.max(0.0) + depot.service;
        let mut prev = 0;
        for &customer in route {
            let c = &instance.customers[customer];
            let d = instance.distance(prev, customer, instance.rounding);
            cost += d;
            time = (time + d).max(c.ready);
            violation += (time - c.due).max(0.0);
            time += c.service;
            load += c.demand;
            prev = customer;
        }
        let d = instance.distance(prev, 0, instance.rounding);
        cost += d;
        violation += (time + d - depot.due).max(0.0);
        violation += (load - instance.capacity).max(0.0);
    }
    (violation, cost)
}

/// Reduce route violations with relocation and swap moves. On failure or
/// deadline expiry, restore the supplied plan and return false.
pub fn repair_plan(instance: &Instance, routes: &mut Vec<Vec<usize>>, deadline: Instant) -> bool {
    let n = instance.customers.len();
    if n < 2
        || routes.is_empty()
        || !instance.capacity.is_finite()
        || instance.capacity <= 0.0
        || instance
            .distances
            .as_ref()
            .is_some_and(|m| m.len() != n || m.iter().any(|row| row.len() != n))
        || (instance.vehicles > 0
            && routes.iter().filter(|r| !r.is_empty()).count() > instance.vehicles)
    {
        return false;
    }
    let mut seen = vec![false; n];
    for &c in routes.iter().flatten() {
        if c == 0 || c >= n || seen[c] {
            return false;
        }
        seen[c] = true;
    }
    if seen[1..].iter().any(|&v| !v) {
        return false;
    }
    let original = routes.clone();
    loop {
        if validate_plan(instance, routes).is_ok() {
            return true;
        }
        if Instant::now() >= deadline {
            *routes = original;
            return false;
        }
        let (current_violation, current_cost) = measures(instance, routes);
        if !current_violation.is_finite() || !current_cost.is_finite() || current_violation <= 1e-9
        {
            *routes = original;
            return false;
        }
        let mut best: Option<(f64, Vec<Vec<usize>>)> = None;
        for a in 0..routes.len() {
            for i in 0..routes[a].len() {
                for b in 0..routes.len() {
                    for j in 0..=routes[b].len() {
                        if a == b && (j == i || j == i + 1) {
                            continue;
                        }
                        if Instant::now() >= deadline {
                            *routes = original;
                            return false;
                        }
                        let mut candidate = routes.clone();
                        let customer = candidate[a].remove(i);
                        let pos = if a == b && j > i { j - 1 } else { j };
                        candidate[b].insert(pos, customer);
                        let (v, c) = measures(instance, &candidate);
                        let gain = current_violation - v;
                        if gain > 1e-9 {
                            let score = (c - current_cost) / gain;
                            if score.is_finite()
                                && best.as_ref().map_or(true, |(old, _)| score < *old)
                            {
                                best = Some((score, candidate));
                            }
                        }
                        if a != b && j < routes[b].len() {
                            let mut candidate = routes.clone();
                            let other = candidate[b][j];
                            candidate[b][j] = candidate[a][i];
                            candidate[a][i] = other;
                            let (v, c) = measures(instance, &candidate);
                            let gain = current_violation - v;
                            if gain > 1e-9 {
                                let score = (c - current_cost) / gain;
                                if score.is_finite()
                                    && best.as_ref().map_or(true, |(old, _)| score < *old)
                                {
                                    best = Some((score, candidate));
                                }
                            }
                        }
                    }
                }
            }
        }
        match best {
            Some((_, next)) => *routes = next,
            None => {
                *routes = original;
                return false;
            }
        }
    }
}
