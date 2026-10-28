//! CVRPLib route files and independent feasibility replay in instance units.
use crate::inputs::Instance;

/// Read `Route #1: 1 2 ...` records. Cost metadata is deliberately ignored:
/// objective values must be recomputed using the selected instance metric.
pub fn parse_solution(text: &str) -> Result<Vec<Vec<usize>>, String> {
    let mut routes = Vec::new();
    for line in text.lines() {
        let line = line.trim();
        if !line.to_ascii_lowercase().starts_with("route ") {
            continue;
        }
        let (_, stops) = line.split_once(':').ok_or("route record is missing ':'")?;
        let route = stops
            .split_whitespace()
            .map(|s| {
                s.parse::<usize>()
                    .map_err(|_| format!("invalid customer index: {s}"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        if route.is_empty() {
            return Err("empty route record".into());
        }
        routes.push(route);
    }
    if routes.is_empty() {
        return Err("no Route records found in solution".into());
    }
    Ok(routes)
}

/// Check coverage, fleet, capacity, service-start windows and depot return;
/// return the recomputed distance. No rounding is applied to the total.
pub fn validate_plan(instance: &Instance, routes: &[Vec<usize>]) -> Result<f64, String> {
    validate_plan_with_fleet(instance, routes, true)
}

/// Replay all constraints except the fleet bound while repairing a clustered plan.
pub(super) fn validate_plan_relaxed_fleet(
    instance: &Instance,
    routes: &[Vec<usize>],
) -> Result<f64, String> {
    validate_plan_with_fleet(instance, routes, false)
}

fn validate_plan_with_fleet(
    instance: &Instance,
    routes: &[Vec<usize>],
    enforce_fleet: bool,
) -> Result<f64, String> {
    if instance.customers.len() < 2 || routes.is_empty() {
        return Err("a solution needs customers and routes".into());
    }
    let n = instance.customers.len();
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
    if enforce_fleet
        && instance.vehicles > 0
        && routes.iter().filter(|r| !r.is_empty()).count() > instance.vehicles
    {
        return Err("solution exceeds the available fleet".into());
    }
    let depot = &instance.customers[0];
    let mut seen = vec![false; instance.customers.len()];
    let mut cost = 0.0;
    for route in routes.iter().filter(|r| !r.is_empty()) {
        let mut time = depot.ready.max(0.0) + depot.service;
        let mut load = 0.0;
        let mut previous = 0;
        for &customer in route {
            if customer == 0 || customer >= seen.len() || seen[customer] {
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

/// Serialize a validated plan in CVRPLib format (internal customer indices).
pub fn format_solution(routes: &[Vec<usize>], cost: f64) -> String {
    use std::fmt::Write;
    let mut text = String::new();
    for (i, route) in routes.iter().filter(|r| !r.is_empty()).enumerate() {
        let _ = write!(text, "Route #{}:", i + 1);
        for customer in route {
            let _ = write!(text, " {customer}");
        }
        text.push('\n');
    }
    let _ = writeln!(text, "Cost {cost:.10}");
    text
}
