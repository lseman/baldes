//! Bounded exact-resource path proof for acyclic grouped-branch domains.
use crate::{
    enumeration::{CompleteResourceRelaxation, PathDomainLimits},
    master::{
        root::{exact_route_feasibility, RootProblem},
        rounding::add_down,
    },
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
    let original = RootProblem::set_partitioning(graph.clone(), 1.).ok()?;
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
mod tests {
    use super::*;
    use crate::graph::ResourceKind;
    #[test]
    fn acyclic_resource_minima_match_independent_quarter_grid_paths() {
        let mut compared = 0;
        for kind in [ResourceKind::Disposable, ResourceKind::Equality] {
            for capacity in [4, 6] {
                for first in [1, 2, 4] {
                    for second in [1, 2, 4] {
                        for direct in [1, 2, 4] {
                            for costs in [[-3., 2., 1., 4., -2.], [2., -4., 3., -1., 5.]] {
                                let mut graph = Graph::<5>::new(4, 5).unwrap();
                                graph.resources[4] = kind;
                                for v in 0..4 {
                                    for r in 0..5 {
                                        graph.set_bounds(v, r, 0., capacity as f64 / 4.).unwrap();
                                    }
                                }
                                graph.set_bounds(1, 4, 0.75, capacity as f64 / 4.).unwrap();
                                for (tail, head, q) in [
                                    (0, 1, first),
                                    (1, 2, second),
                                    (2, 3, 0),
                                    (1, 3, 0),
                                    (0, 2, direct),
                                ] {
                                    let arc = graph.add_arc(tail, head, 0.).unwrap();
                                    graph.arc_mut(arc).unwrap().consumption[4] = q as f64 / 4.;
                                }
                                let mut expected: Option<f64> = None;
                                for path in [vec![0, 1, 2], vec![0, 3], vec![4, 2]] {
                                    let mut q = 0;
                                    let mut feasible = true;
                                    for &a in &path {
                                        q += [first, second, 0, 0, direct][a];
                                        let lower = if graph.arcs[a].head == 1 { 3 } else { 0 };
                                        if q < lower {
                                            if kind == ResourceKind::Equality {
                                                feasible = false;
                                                break;
                                            }
                                            q = lower;
                                        }
                                        if q > capacity {
                                            feasible = false;
                                            break;
                                        }
                                    }
                                    if feasible {
                                        let cost = path.iter().map(|&a| costs[a]).sum::<f64>();
                                        expected = Some(expected.map_or(cost, |v| v.min(cost)));
                                    }
                                }
                                let mut work = 1_000_000;
                                assert_eq!(
                                    minimum(
                                        &graph,
                                        &costs,
                                        &mut work,
                                        Instant::now() + std::time::Duration::from_secs(10)
                                    ),
                                    Some(expected)
                                );
                                compared += 1;
                            }
                        }
                    }
                }
            }
        }
        assert_eq!(compared, 216);
    }
    #[test]
    fn cycles_unsupported_resources_and_budgets_never_return_partial_path_proofs() {
        let mut graph = Graph::<5>::new(3, 5).unwrap();
        for v in 0..3 {
            for r in 0..5 {
                graph.set_bounds(v, r, 0., 3.).unwrap();
            }
        }
        graph.add_arc(0, 1, 0.).unwrap();
        graph.add_arc(1, 1, -1.).unwrap();
        graph.add_arc(1, 2, 0.).unwrap();
        graph.arcs[1].consumption[4] = 1.;
        let end = Instant::now() + std::time::Duration::from_secs(10);
        let mut work = 1_000_000;
        assert_eq!(minimum(&graph, &[0., -1., 0.], &mut work, end), None);
        graph.arcs.remove(1);
        graph.arcs[0].consumption[4] = -1.;
        let mut work = 1_000_000;
        assert_eq!(minimum(&graph, &[0., 0.], &mut work, end), None);
        graph.arcs[0].consumption[4] = 1.;
        let mut work = 0;
        assert_eq!(minimum(&graph, &[0., 0.], &mut work, end), None);
        let mut work = 1_000_000;
        assert_eq!(minimum(&graph, &[0., 0.], &mut work, Instant::now()), None);
    }
}
