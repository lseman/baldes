//! Packing-group resource candidates with feasible suffix intervals.
use super::*;
use crate::graph::{Arc, ResourceKind, Window};
use crate::master::branching::{
    resource_intervals::{resource_branch_fraction, select_resource_threshold, ResourceInterval},
    AccumulatedResourceGroupBranch,
};

pub(in crate::master::tree) fn candidates<const D: usize>(
    problem: &RootProblem<D>,
    routes: &[(&crate::Route, f64)],
    history: &[BranchConstraint],
) -> Vec<BranchCandidate> {
    let axes: Vec<_> = problem
        .graph
        .resources
        .iter()
        .enumerate()
        .filter_map(|(r, kind)| (*kind == ResourceKind::Disposable).then_some(r))
        .collect();
    let Some(&first_resource) = axes.first() else {
        return vec![];
    };
    let mut incoming = vec![vec![]; problem.graph.vertices.len()];
    for arc in &problem.graph.arcs {
        incoming[arc.head].push(arc.id);
    }
    let mut packing_rows = vec![None; problem.graph.vertices.len()];
    for &vertex in &problem.customers {
        let Some(&first) = incoming[vertex].first() else {
            continue;
        };
        packing_rows[vertex] = problem.arc_entries[problem.arc_position(first).unwrap()]
            .iter()
            .find_map(|&(row, coefficient)| {
                if coefficient < 1. {
                    return None;
                }
                let group = AccumulatedResourceGroupBranch::new(
                    row,
                    incoming[vertex].clone(),
                    first_resource,
                    0.,
                    false,
                )
                .ok()?;
                group.validate(problem).ok().map(|_| row)
            });
    }
    // Several physical heads can represent the same original packing set.
    // Aggregate by the proved row, rather than splitting that set by vertex.
    let mut groups = BTreeMap::<usize, Vec<i32>>::new();
    for (vertex, row) in packing_rows.iter().enumerate() {
        if let Some(row) = row {
            groups
                .entry(*row)
                .or_default()
                .extend_from_slice(&incoming[vertex]);
        }
    }
    let mut observations = BTreeMap::<(usize, usize), Vec<ResourceInterval>>::new();
    let mut invalid = HashSet::new();
    let mut bounds = [0.; D];
    for &resource in &axes {
        bounds[resource] = problem
            .graph
            .vertices
            .iter()
            .map(|v| v.bounds[resource].upper)
            .fold(f64::NEG_INFINITY, f64::max)
            + 2e-6;
        for &(route, weight) in routes {
            // Replay a suffix once per route/resource, not once per group.
            let uppers = suffix_upper(problem, route, resource, history);
            for (i, &id) in route.arcs.iter().enumerate().rev() {
                let Some(position) = problem.arc_position(id) else {
                    continue;
                };
                let vertex = problem.graph.arcs[position].head;
                let Some(packing_row) = packing_rows[vertex] else {
                    continue;
                };
                let lower = route
                    .resource_profile
                    .get(i + 1)
                    .and_then(|q| q.get(resource))
                    .copied();
                match (lower, uppers.as_ref()) {
                    (Some(lower), Some(uppers)) if lower.is_finite() && lower <= uppers[i] => {
                        observations
                            .entry((packing_row, resource))
                            .or_default()
                            .push(ResourceInterval {
                                lower,
                                upper: uppers[i],
                                weight,
                            });
                    }
                    _ => {
                        invalid.insert((packing_row, resource));
                    }
                }
            }
        }
    }
    let mut candidates = vec![];
    for (packing_row, arcs) in groups {
        for &resource in &axes {
            if invalid.contains(&(packing_row, resource)) {
                continue;
            }
            let intervals = observations
                .remove(&(packing_row, resource))
                .unwrap_or_default();
            if intervals.iter().map(|i| i.weight).sum::<f64>() > 1. + 1e-6 {
                continue;
            }
            let Ok(Some(threshold)) = select_resource_threshold(&intervals, bounds[resource])
            else {
                continue;
            };
            let Ok(down) = AccumulatedResourceGroupBranch::new(
                packing_row,
                arcs.clone(),
                resource,
                threshold,
                false,
            ) else {
                continue;
            };
            if history.iter().any(|c|matches!(c,BranchConstraint::AccumulatedResourceGroup(old) if old.same_split(&down))) {continue;}
            let value = resource_branch_fraction(&intervals, threshold);
            if value > 1e-6 && value < 1. - 1e-6 {
                candidates.push(BranchCandidate {
                    up: BranchConstraint::AccumulatedResourceGroup(down.with_upper(true)),
                    down: BranchConstraint::AccumulatedResourceGroup(down),
                    value,
                });
            }
        }
    }
    candidates
}

fn head_window<const D: usize>(
    problem: &RootProblem<D>,
    arc: &Arc<D>,
    resource: usize,
    history: &[BranchConstraint],
) -> Option<Window> {
    let mut window = problem.graph.vertices[arc.head].bounds[resource];
    if let Some(bound) = arc.bounds[resource] {
        window.lower = window.lower.max(bound.lower);
        window.upper = window.upper.min(bound.upper);
    }
    for constraint in history {
        let restriction = match constraint {
            BranchConstraint::AccumulatedResource(c)
                if c.arc == arc.id && c.resource == resource =>
            {
                Some((c.upper, c.boundary()))
            }
            BranchConstraint::AccumulatedResourceGroup(c)
                if c.resource() == resource && c.arcs().binary_search(&arc.id).is_ok() =>
            {
                Some((c.upper(), c.boundary()))
            }
            _ => None,
        };
        if let Some((upper, bound)) = restriction {
            if upper {
                window.lower = window.lower.max(bound)
            } else {
                window.upper = window.upper.min(bound)
            }
        }
    }
    (window.lower <= window.upper).then_some(window)
}

fn suffix_upper<const D: usize>(
    problem: &RootProblem<D>,
    route: &crate::Route,
    resource: usize,
    history: &[BranchConstraint],
) -> Option<Vec<f64>> {
    crate::master::branching::resource_intervals::suffix_upper(
        &problem.graph,
        route,
        resource,
        |id| problem.arc_position(id),
        |arc| head_window(problem, arc, resource, history),
    )
}

#[cfg(test)]
#[path = "../../../../tests/unit/master/tree/candidates/resource_groups/tests.rs"]
pub(crate) mod tests;
