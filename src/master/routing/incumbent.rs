//! Validate a rounded integer witness before publishing its objective.
use super::*;

/// Exact-real scalar feasibility of a nonempty physically elementary path.
/// Stored floating resource profiles are not used as feasibility proofs.
pub(crate) fn exact_route_feasible<const D: usize>(
    problem: &RoutingContext<D>,
    route: &Route,
) -> Option<()> {
    exact_route_feasibility(problem, route)?.then_some(())
}

/// Distinguish a proved invalid path from unsupported arithmetic/resource rules.
/// `None` must not be used to discard paths from an original-domain proof.
pub(crate) fn exact_route_feasibility<const D: usize>(
    problem: &RoutingContext<D>,
    route: &Route,
) -> Option<bool> {
    exact_route_feasibility_using(problem, route, |kernel, arcs| {
        kernel.certify(arcs).map(|result| result.0)
    })
}

/// The same original resource check under a caller-owned proof budget.
pub(crate) fn exact_route_feasibility_with_budget<const D: usize>(
    problem: &RoutingContext<D>,
    route: &Route,
    work: &mut usize,
    deadline: std::time::Instant,
) -> Option<bool> {
    if std::time::Instant::now() >= deadline {
        return None;
    }
    let result = exact_route_feasibility_using(problem, route, |kernel, arcs| {
        kernel
            .certify_with_budget(arcs, work, Some(deadline))
            .map(|result| result.0)
    });
    if std::time::Instant::now() >= deadline {
        None
    } else {
        result
    }
}

fn exact_route_feasibility_using<const D: usize>(
    problem: &RoutingContext<D>,
    route: &Route,
    certify: impl FnOnce(&crate::graph::coupled::CoupledKernel, &[usize]) -> Option<bool>,
) -> Option<bool> {
    use crate::graph::ResourceKind;
    use std::cmp::Ordering::{Greater, Less};
    let g = &problem.model.networks[0].graph;
    if g.resources.len() > D {
        return None;
    }
    if g.source == g.sink
        || g.sink >= g.vertices.len()
        || route.arcs.len() > g.original_max_arcs()
        || route.nodes.len() != route.arcs.len().checked_add(1)?
        || route.nodes.first().copied() != Some(i32::try_from(g.source).ok()?)
    {
        return Some(false);
    }
    let source = g.vertices.get(g.source)?;
    let mut resources =
        crate::graph::ScalarValues::<_, D>::filled(g.resources.len(), ExactSum::default());
    let valid_window =
        |w: crate::graph::Window| w.lower.is_finite() && w.upper.is_finite() && w.lower <= w.upper;
    for (r, value) in resources.iter_mut().enumerate().take(g.resources.len()) {
        let bounds = *source.bounds.get(r)?;
        if !valid_window(bounds) {
            return None;
        }
        let Some(initial) = g.root_resource(r, true) else {
            return Some(false);
        };
        value.add(initial)?;
        if value.compare(bounds.lower)? == Less || value.compare(bounds.upper)? == Greater {
            return Some(false);
        }
    }
    let mut at = g.source;
    let mut visited = HashSet::new();
    let mut groups = HashSet::new();
    if g.has_explicit_elementarity_groups() {
        groups.extend(g.vertices[g.source].elements.iter().copied());
    }
    for (position, &id) in route.arcs.iter().enumerate() {
        let Some(index) = problem.arc_position(id) else {
            return Some(false);
        };
        let arc = g.arcs.get(index)?;
        if arc.id != id
            || arc.tail != at
            || at == g.sink
            || arc.head == g.source
            || route.nodes[position + 1] != i32::try_from(arc.head).ok()?
        {
            return Some(false);
        }
        if g.has_explicit_elementarity_groups() {
            let mut transition_elements = HashSet::new();
            for &element in arc.elements.iter().chain(&g.vertices[arc.head].elements) {
                if transition_elements.insert(element) && !groups.insert(element) {
                    return Some(false);
                }
            }
        }
        let head = g.vertices.get(arc.head)?;
        if arc.head != g.sink && !visited.insert(arc.head) && g.max_route_arcs().is_none() {
            return Some(false);
        }
        for (r, value) in resources.iter_mut().enumerate().take(g.resources.len()) {
            let mut bounds = *head.bounds.get(r)?;
            if !valid_window(bounds) {
                return None;
            }
            if let Some(window) = *arc.bounds.get(r)? {
                if !valid_window(window) {
                    return None;
                }
                bounds.lower = bounds.lower.max(window.lower);
                bounds.upper = bounds.upper.min(window.upper);
            }
            let consumption = *arc.consumption.get(r)?;
            if g.resources[r] == ResourceKind::Disposable && consumption < 0. {
                return None;
            }
            value.add(consumption)?;
            #[cfg(feature = "multi-time-windows")]
            if !g.time_windows(arc.head, r).is_empty() {
                if g.resources[r] != ResourceKind::Disposable {
                    return None;
                }
                let mut feasible = false;
                for window in g.time_windows(arc.head, r) {
                    let lower = window.lower.max(bounds.lower);
                    let upper = window.upper.min(bounds.upper);
                    if lower <= upper && value.compare(upper)? != Greater {
                        if value.compare(lower)? == Less {
                            *value = ExactSum::default();
                            value.add(lower)?;
                        }
                        feasible = true;
                        break;
                    }
                }
                if !feasible {
                    return Some(false);
                }
                continue;
            }
            if value.compare(bounds.lower)? == Less {
                if g.resources[r] == ResourceKind::Equality {
                    return Some(false);
                }
                *value = ExactSum::default();
                value.add(bounds.lower)?;
            }
            if value.compare(bounds.upper)? == Greater {
                return Some(false);
            }
        }
        at = arc.head;
    }
    if let Some(kernel) = g.coupled_kernel() {
        let arcs = route
            .arcs
            .iter()
            .map(|id| problem.arc_position(*id))
            .collect::<Option<Vec<_>>>()?;
        if !certify(kernel, &arcs)? {
            return Some(false);
        }
    }
    Some(at == g.sink)
}

pub(super) fn validated_upper<const D: usize>(
    problem: &RoutingContext<D>,
    routes: &[Route],
    tags: &[i32],
    values: &[f64],
) -> Option<f64> {
    if tags.len() != values.len() {
        return None;
    }
    // Most LP solutions are fractional. Reject the whole vector before
    // allocating exact row sums or validating any of its integral routes.
    if values.iter().any(|&value| {
        let quantity = value.round();
        !value.is_finite()
            || (value - quantity).abs() > 1e-6
            || quantity < 0.
            || quantity > 9_007_199_254_740_992.
    }) {
        return None;
    }
    let mut rows: Vec<_> = problem
        .model
        .rows
        .iter()
        .map(|_| ExactSum::default())
        .collect();
    let mut objective = ExactSum::default();
    for (&tag, &value) in tags.iter().zip(values) {
        let quantity = value.round();
        if quantity == 0. {
            continue;
        }
        let route = routes.get(usize::try_from(tag).ok()?)?;
        exact_route_feasible(problem, route)?;
        if let Some(kernel) = problem.model.networks[0].graph.coupled_kernel() {
            let arcs = route
                .arcs
                .iter()
                .map(|id| problem.arc_position(*id))
                .collect::<Option<Vec<_>>>()?;
            let (valid, cost) = kernel.certify(&arcs)?;
            if !valid {
                return None;
            }
            objective.product(cost[1], quantity)?;
        }
        for &(row, coefficient) in &problem.model.networks[0].route_entries {
            rows.get_mut(row)?.product(coefficient, quantity)?;
        }
        for &id in &route.arcs {
            let arc = problem.arc_position(id)?;
            objective.product(problem.model.networks[0].graph.arcs[arc].cost, quantity)?;
            for &(row, coefficient) in &problem.model.networks[0].arc_entries[arc] {
                rows.get_mut(row)?.product(coefficient, quantity)?;
            }
        }
        for (row, constraint) in &problem.branch_constraints {
            if constraint.nonlinear() {
                let coefficient = constraint.coefficient(problem, route);
                if coefficient != 0. {
                    return None;
                }
                rows.get_mut(*row)?.product(coefficient, quantity)?;
            }
        }
    }
    for (mut activity, row) in rows.into_iter().zip(&problem.model.rows) {
        activity.add(-row.rhs)?;
        let sign = activity.sign();
        if match row.sense {
            RowSense::Equal => sign != 0.,
            RowSense::LessEqual => sign > 0.,
            RowSense::GreaterEqual => sign < 0.,
        } {
            return None;
        }
    }
    for (route, quantity) in &problem.fixed_routes {
        if *quantity == 0 {
            return None;
        }
        exact_route_feasible(problem, route)?;
        if let Some(kernel) = problem.model.networks[0].graph.coupled_kernel() {
            let arcs = route
                .arcs
                .iter()
                .map(|id| problem.arc_position(*id))
                .collect::<Option<Vec<_>>>()?;
            let (valid, cost) = kernel.certify(&arcs)?;
            if !valid {
                return None;
            }
            objective.product(cost[1], f64::from(*quantity))?;
        }
        for &id in &route.arcs {
            objective.product(
                problem.model.networks[0].graph.arcs[problem.arc_position(id)?].cost,
                f64::from(*quantity),
            )?;
        }
    }
    objective.upper()
}

impl<const D: usize> RoutingContext<D> {
    pub(crate) fn unit_route_incumbent(&self, routes: &[Route]) -> Option<f64> {
        let tags = (0..routes.len())
            .map(i32::try_from)
            .collect::<std::result::Result<Vec<_>, _>>()
            .ok()?;
        validated_upper(self, routes, &tags, &vec![1.; routes.len()])
    }
}

#[cfg(test)]
#[path = "../../../tests/unit/master/routing/incumbent/tests.rs"]
mod tests;
