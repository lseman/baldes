//! Route-level perturbation and improvement operators shared by heuristics.
use crate::{inputs::Instance, rng::DefaultRng};
mod local_search;
pub(crate) use local_search::LocalSearch;
pub mod sisr;
pub mod solution;
#[cfg(test)]
#[path = "../../../tests/unit/extensions/common.rs"]
mod tests;

/// Replay a complete solution and return its distance in instance units.
pub fn validate_plan(instance: &Instance, routes: &[Vec<usize>]) -> Result<f64, String> {
    validate_plan_with_fleet(instance, routes, true)
}

/// Build a dense customer subproblem from disjoint routes. The caller validates
/// the input routes; `map[local]` restores the original customer ID.
pub(crate) fn route_subproblem<'a>(
    instance: &Instance,
    routes: impl IntoIterator<Item = &'a [usize]>,
) -> (Instance, Vec<usize>, Vec<Vec<usize>>) {
    let mut map = vec![0];
    let mut seed = Vec::new();
    for customers in routes {
        let mut route = Vec::with_capacity(customers.len());
        for &customer in customers {
            route.push(map.len());
            map.push(customer);
        }
        seed.push(route);
    }
    let sub = Instance {
        name: format!("{}-subset", instance.name),
        vehicles: seed.len(),
        capacity: instance.capacity,
        customers: map.iter().map(|&i| instance.customers[i].clone()).collect(),
        rounding: instance.rounding,
        distances: instance.distances.as_ref().map(|matrix| {
            map.iter()
                .map(|&i| map.iter().map(|&j| matrix[i][j]).collect())
                .collect()
        }),
    };
    (sub, map, seed)
}

pub(crate) fn validate_plan_with_fleet(
    instance: &Instance,
    routes: &[Vec<usize>],
    enforce_fleet: bool,
) -> Result<f64, String> {
    if instance.customers.len() < 2 || routes.is_empty() {
        return Err("a solution needs customers and routes".into());
    }
    validate_instance(instance)?;
    plan_cost(instance, routes, enforce_fleet)
}

/// Replay a plan after immutable instance data has been validated at the search
/// boundary. Coverage, fleet, resource feasibility and costs are still checked.
pub(crate) fn plan_cost(
    instance: &Instance,
    routes: &[Vec<usize>],
    enforce_fleet: bool,
) -> Result<f64, String> {
    let n = instance.customers.len();
    if enforce_fleet
        && instance.vehicles > 0
        && routes.iter().filter(|r| !r.is_empty()).count() > instance.vehicles
    {
        return Err("solution exceeds the available fleet".into());
    }
    let depot = &instance.customers[0];
    let mut seen = vec![false; n];
    let mut cost = 0.0;
    for route in routes.iter().filter(|r| !r.is_empty()) {
        let mut time = depot.ready.max(0.0) + depot.service;
        let mut load = 0.0;
        let mut previous = 0;
        for &customer in route {
            if customer == 0 || customer >= n || seen[customer] {
                return Err(format!(
                    "duplicate, depot or out-of-range customer: {customer}"
                ));
            }
            seen[customer] = true;
            let c = &instance.customers[customer];
            let travel = instance.distance(previous, customer, instance.rounding);
            if !travel.is_finite() || travel < 0.0 {
                return Err("invalid travel distance".into());
            }
            cost += travel;
            time = (time + travel).max(c.ready);
            if time > c.due + 1e-6 {
                return Err(format!("time window violated at customer {customer}"));
            }
            time += c.service;
            load += c.demand;
            previous = customer;
        }
        let travel = instance.distance(previous, 0, instance.rounding);
        if !travel.is_finite() || travel < 0.0 {
            return Err("invalid depot return distance".into());
        }
        cost += travel;
        if time + travel > depot.due + 1e-6 || load > instance.capacity + 1e-6 {
            return Err("route violates capacity or depot return window".into());
        }
    }
    if seen[1..].iter().any(|&s| !s) || !cost.is_finite() {
        return Err("solution omits customers or has a non-finite cost".into());
    }
    Ok(cost)
}

pub(crate) fn validate_instance(instance: &Instance) -> Result<(), String> {
    let n = instance.customers.len();
    if n < 2 {
        return Err("an instance needs a depot and customers".into());
    }
    if !instance.capacity.is_finite()
        || instance.capacity <= 0.0
        || instance.customers.iter().any(|c| {
            !c.x.is_finite()
                || !c.y.is_finite()
                || !c.demand.is_finite()
                || c.demand < 0.0
                || !c.ready.is_finite()
                || c.due.is_nan()
                || c.due < c.ready
                || !c.service.is_finite()
                || c.service < 0.0
        })
        || instance
            .distances
            .as_ref()
            .is_some_and(|m| m.len() != n || m.iter().any(|r| r.len() != n))
    {
        return Err("invalid instance data or distance matrix dimensions".into());
    }
    Ok(())
}

/// Build a feasible starting plan by cheapest feasible insertion, opening a
/// new singleton route only when existing routes cannot accept the customer.
pub fn construct_feasible(
    instance: &Instance,
    deadline: std::time::Instant,
) -> Result<Vec<Vec<usize>>, String> {
    if instance.customers.len() < 2 {
        return Err("construction needs at least one customer".into());
    }
    let n = instance.customers.len();
    if instance
        .distances
        .as_ref()
        .is_some_and(|m| m.len() != n || m.iter().any(|r| r.len() != n))
    {
        return Err("invalid distance matrix dimensions".into());
    }
    let mut routes: Vec<Vec<usize>> = Vec::new();
    let mut order: Vec<_> = (1..instance.customers.len()).collect();
    order.sort_by(|&a, &b| {
        instance.customers[a]
            .due
            .total_cmp(&instance.customers[b].due)
            .then_with(|| {
                instance.customers[b]
                    .demand
                    .total_cmp(&instance.customers[a].demand)
            })
            .then(a.cmp(&b))
    });
    for customer in order.iter().copied() {
        if std::time::Instant::now() >= deadline {
            return Err("time limit reached during AILS-II construction".into());
        }
        let mut best: Option<(usize, usize, f64)> = None;
        for (r, route) in routes.iter().enumerate() {
            for pos in 0..=route.len() {
                if std::time::Instant::now() >= deadline {
                    return Err("time limit reached during AILS-II construction".into());
                }
                let mut trial = route.clone();
                trial.insert(pos, customer);
                if validate_route(instance, &trial).is_err() {
                    continue;
                }
                let prev = if pos == 0 { 0 } else { route[pos - 1] };
                let next = route.get(pos).copied().unwrap_or(0);
                let delta = instance.distance(prev, customer, instance.rounding)
                    + instance.distance(customer, next, instance.rounding)
                    - instance.distance(prev, next, instance.rounding);
                if best.map_or(true, |(_, _, value)| delta < value) {
                    best = Some((r, pos, delta));
                }
            }
        }
        if let Some((r, pos, _)) = best {
            routes[r].insert(pos, customer);
        } else {
            if instance.vehicles > 0 && routes.len() >= instance.vehicles {
                if order.len() <= 20 {
                    return construct_backtracking(instance, &order, deadline);
                }
                return Err(format!(
                    "could not construct feasible route plan at customer {customer}"
                ));
            }
            let singleton = vec![customer];
            if validate_route(instance, &singleton).is_err() {
                return Err(format!("customer {customer} cannot be served alone"));
            }
            routes.push(singleton);
        }
    }
    validate_plan(instance, &routes)?;
    Ok(routes)
}

fn construct_backtracking(
    instance: &Instance,
    order: &[usize],
    deadline: std::time::Instant,
) -> Result<Vec<Vec<usize>>, String> {
    fn visit(
        instance: &Instance,
        order: &[usize],
        index: usize,
        routes: &mut Vec<Vec<usize>>,
        deadline: std::time::Instant,
    ) -> bool {
        if std::time::Instant::now() >= deadline {
            return false;
        }
        if index == order.len() {
            return true;
        }
        let customer = order[index];
        for r in 0..routes.len() {
            for pos in 0..=routes[r].len() {
                routes[r].insert(pos, customer);
                if validate_route(instance, &routes[r]).is_ok()
                    && visit(instance, order, index + 1, routes, deadline)
                {
                    return true;
                }
                routes[r].remove(pos);
            }
        }
        if (instance.vehicles == 0 || routes.len() < instance.vehicles)
            && validate_route(instance, &[customer]).is_ok()
        {
            routes.push(vec![customer]);
            if visit(instance, order, index + 1, routes, deadline) {
                return true;
            }
            routes.pop();
        }
        false
    }
    let mut routes = Vec::new();
    if visit(instance, order, 0, &mut routes, deadline) {
        validate_plan(instance, &routes)?;
        Ok(routes)
    } else {
        Err("could not construct a feasible plan within the fleet and time limit".into())
    }
}

pub(crate) fn validate_route(instance: &Instance, route: &[usize]) -> Result<(), String> {
    let mut time = instance.customers[0].ready.max(0.0) + instance.customers[0].service;
    let mut load = 0.0;
    let mut previous = 0;
    for &customer in route {
        if customer == 0 || customer >= instance.customers.len() {
            return Err("invalid customer".into());
        }
        let c = &instance.customers[customer];
        let travel = instance.distance(previous, customer, instance.rounding);
        if !travel.is_finite() || travel < 0.0 {
            return Err("invalid travel distance".into());
        }
        time = (time + travel).max(c.ready);
        if time > c.due + 1e-6 {
            return Err("time window".into());
        }
        time += c.service;
        load += c.demand;
        previous = customer;
    }
    let travel = instance.distance(previous, 0, instance.rounding);
    if !travel.is_finite() || travel < 0.0 {
        return Err("invalid depot return distance".into());
    }
    time += travel;
    if load > instance.capacity + 1e-6 || time > instance.customers[0].due + 1e-6 {
        return Err("capacity or depot window".into());
    }
    Ok(())
}

/// Remove a spatially coherent string from one route and greedily reinsert it.
pub fn sisr_kick(
    instance: &Instance,
    routes: &mut [Vec<usize>],
    rng: &mut DefaultRng,
    strength: usize,
) {
    let eligible: Vec<_> = routes
        .iter()
        .enumerate()
        .filter(|(_, r)| !r.is_empty())
        .map(|(i, _)| i)
        .collect();
    if eligible.is_empty() {
        return;
    }
    let route = &mut routes[eligible[rng.next() as usize % eligible.len()]];
    let mut removed = vec![route.remove(rng.next() as usize % route.len())];
    while removed.len() < strength.clamp(1, removed.len() + route.len()) && !route.is_empty() {
        let anchor = removed[rng.next() as usize % removed.len()];
        let pos = route
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| {
                instance
                    .distance(anchor, **a, instance.rounding)
                    .total_cmp(&instance.distance(anchor, **b, instance.rounding))
            })
            .map(|(i, _)| i)
            .unwrap();
        removed.push(route.remove(pos));
    }
    for customer in removed {
        let mut best = None;
        for (r, route) in routes.iter().enumerate() {
            for pos in 0..=route.len() {
                let prev = if pos == 0 { 0 } else { route[pos - 1] };
                let next = route.get(pos).copied().unwrap_or(0);
                let delta = instance.distance(prev, customer, instance.rounding)
                    + instance.distance(customer, next, instance.rounding)
                    - instance.distance(prev, next, instance.rounding);
                if best.map_or(true, |(_, _, cost): (usize, usize, f64)| delta < cost) {
                    best = Some((r, pos, delta));
                }
            }
        }
        if let Some((r, pos, _)) = best {
            routes[r].insert(pos, customer);
        }
    }
}

/// Apply improving route moves, accepting only fully feasible plans.
pub fn improve_plan(
    instance: &Instance,
    routes: &mut Vec<Vec<usize>>,
    deadline: std::time::Instant,
) {
    let changed = vec![true; routes.len()];
    improve_modified_routes(instance, routes, deadline, &changed);
}

/// Improve moves touching at least one modified route. Accepted moves extend
/// the set of routes considered modified for the rest of the pass.
pub(crate) fn improve_modified_routes(
    instance: &Instance,
    routes: &mut Vec<Vec<usize>>,
    deadline: std::time::Instant,
    initial_changed: &[bool],
) {
    local_search::improve(instance, routes, deadline, initial_changed);
}

fn nearest_positions(
    instance: &Instance,
    anchor: usize,
    route: &[usize],
    limit: usize,
) -> Vec<usize> {
    let mut neighbors: Vec<_> = (0..route.len()).collect();
    let compare = |&left: &usize, &right: &usize| {
        let distance = |index| {
            let customer = route[index];
            instance
                .distance(anchor, customer, instance.rounding)
                .min(instance.distance(customer, anchor, instance.rounding))
        };
        distance(left)
            .total_cmp(&distance(right))
            .then(left.cmp(&right))
    };
    if neighbors.len() > limit {
        neighbors.select_nth_unstable_by(limit, compare);
        neighbors.truncate(limit);
    }
    neighbors.sort_unstable_by(compare);
    neighbors
}
