//! Bounded exact-resource path proof for acyclic grouped-branch domains.
use crate::{
    core::rounding::add_down,
    enumeration::{CompleteResourceRelaxation, PathDomainLimits},
    master::routing::{exact_route_feasibility, RoutingContext},
    Graph, Route,
};
use std::time::Instant;

/// A DAG has no repeated walks, so a complete conservative simple-path scan
/// covers its entire walk domain. Cyclic graphs decline this stronger proof.
/// The previous unrestricted walk relaxation remains available on any failure.
pub(super) fn minimum<const D: usize>(
    graph: &Graph<D>,
    weights: &[f64],
    work: &mut usize,
    deadline: Instant,
) -> Option<Option<f64>> {
    let allowance = (*work / 2).min(100_000);
    *work -= allowance;
    let mut left = allowance;
    let topology_work = graph
        .vertices
        .len()
        .checked_mul(3)?
        .checked_add(graph.arcs.len().checked_mul(2)?)?;
    left = left.checked_sub(topology_work)?;
    if Instant::now() >= deadline
        || graph.source == graph.sink
        || graph
            .arcs
            .iter()
            .any(|a| a.tail == graph.source && a.head == graph.sink)
    {
        return None;
    }
    let mut degree = vec![0usize; graph.vertices.len()];
    let mut outgoing = vec![vec![]; graph.vertices.len()];
    for arc in &graph.arcs {
        degree[arc.head] += 1;
        outgoing[arc.tail].push(arc.head);
    }
    let mut queue: Vec<_> = degree
        .iter()
        .enumerate()
        .filter_map(|(i, &d)| (d == 0).then_some(i))
        .collect();
    let mut visited = 0;
    while let Some(v) = queue.pop() {
        visited += 1;
        for &head in &outgoing[v] {
            degree[head] -= 1;
            if degree[head] == 0 {
                queue.push(head);
            }
        }
    }
    if visited != graph.vertices.len() {
        return None;
    }
    let mut relaxed = graph.clone();
    relaxed
        .resources
        .fill(crate::graph::ResourceKind::Disposable);
    let paths = CompleteResourceRelaxation::collect(
        &relaxed,
        PathDomainLimits {
            max_paths: 1024,
            max_work: left,
            max_stored_arcs: left,
            deadline: Some(deadline),
        },
    )
    .ok()?;
    left = left.checked_sub(paths.work_used())?;
    let original = RoutingContext::set_partitioning(graph.clone(), 1.).ok()?;
    let mut minimum: Option<f64> = None;
    for path in paths.paths() {
        if Instant::now() >= deadline {
            return None;
        }
        let length = path.arc_positions().len().checked_add(1)?;
        // Exact expansions/comparisons can grow with the path length.
        left = left.checked_sub(
            length
                .checked_mul(length)?
                .checked_mul(graph.resources.len().checked_mul(8)?.checked_add(4)?)?,
        )?;
        let route = Route {
            coupled_controls: Vec::new(),
            coupled_values: Vec::new(),
            nodes: std::iter::once(graph.source as i32)
                .chain(
                    path.arc_positions()
                        .iter()
                        .map(|&a| graph.arcs[a].head as i32),
                )
                .collect(),
            arcs: path
                .arc_positions()
                .iter()
                .map(|&a| graph.arcs[a].id)
                .collect(),
            cost: 0.,
            original_cost: 0.,
            resources: vec![],
            resource_profile: vec![],
        };
        match exact_route_feasibility(&original, &route) {
            Some(false) => continue,
            Some(true) => {}
            None => return None,
        }
        let mut cost = 0.;
        for &a in path.arc_positions() {
            cost = add_down(cost, weights[a])?;
        }
        minimum = Some(minimum.map_or(cost, |old| old.min(cost)));
    }
    Some(minimum)
}

#[cfg(test)]
#[path = "../../../../tests/unit/master/model/bounds/resource_groups/tests.rs"]
mod tests;
