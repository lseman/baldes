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
    let mut result = vec![0.; route.arcs.len()];
    let mut upper = problem.graph.vertices[problem.graph.sink].bounds[resource].upper;
    let mut expected_head = problem.graph.sink;
    for (i, &id) in route.arcs.iter().enumerate().rev() {
        let arc = &problem.graph.arcs[problem.arc_position(id)?];
        if arc.head != expected_head {
            return None;
        }
        let window = head_window(problem, arc, resource, history)?;
        upper = upper.min(window.upper);
        if upper < window.lower {
            return None;
        }
        result[i] = upper;
        upper = (upper - arc.consumption[resource])
            .min(problem.graph.vertices[arc.tail].bounds[resource].upper);
        if !upper.is_finite() || upper < problem.graph.vertices[arc.tail].bounds[resource].lower {
            return None;
        }
        expected_head = arc.tail;
    }
    (expected_head == problem.graph.source).then_some(result)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    #[test]
    fn one_packing_row_combines_distinct_physical_heads() {
        let mut graph = crate::Graph::<5>::new(4, 5).unwrap();
        for vertex in 0..4 {
            for resource in 0..5 {
                graph.set_bounds(vertex, resource, 0., 3.).unwrap();
            }
        }
        graph.set_bounds(1, 4, 0., 0.5).unwrap();
        graph.set_bounds(2, 4, 0., 2.5).unwrap();
        let low = graph.add_arc(0, 1, 1.).unwrap();
        graph.arc_mut(low).unwrap().consumption[4] = 0.5;
        let low_end = graph.add_arc(1, 3, 0.).unwrap();
        let high = graph.add_arc(0, 2, 1.).unwrap();
        graph.arc_mut(high).unwrap().consumption[4] = 2.5;
        let high_end = graph.add_arc(2, 3, 0.).unwrap();
        let mut problem = RootProblem::set_partitioning(graph, 1.).unwrap();
        // The two physical heads are alternatives for one original unit cover.
        let high_position = problem.arc_position(high).unwrap();
        problem.arc_entries[high_position] = vec![(1, 1.)];
        problem.rows[2].sense = RowSense::LessEqual;
        let low_route = problem.elementary_route(&[low, low_end]).unwrap();
        let high_route = problem.elementary_route(&[high, high_end]).unwrap();
        let routes = [(&low_route, 0.25), (&high_route, 0.75)];
        let choices = candidates(&problem, &routes, &[]);
        assert_eq!(choices.len(), 1);
        let BranchConstraint::AccumulatedResourceGroup(group) = &choices[0].down else {
            panic!()
        };
        assert_eq!(group.packing_row(), 1);
        assert_eq!(group.arcs(), [low, high]);
        assert_eq!(group.resource(), 4);
        assert_eq!(group.threshold(), 2.5);
        assert_eq!(choices[0].value, 0.25);
        assert!(candidates(&problem, &routes, &[choices[0].down.clone()]).is_empty());
        for (index, row) in problem.rows.iter().enumerate() {
            let activity: f64 = routes
                .iter()
                .map(|(route, weight)| {
                    problem
                        .base_entries(route)
                        .unwrap()
                        .iter()
                        .filter(|(r, _)| *r == index)
                        .map(|(_, a)| a)
                        .sum::<f64>()
                        * weight
                })
                .sum();
            match row.sense {
                RowSense::Equal => assert_eq!(activity, row.rhs),
                RowSense::LessEqual => assert!(activity <= row.rhs),
                _ => panic!(),
            }
        }
        for choice in [choices[0].down.clone(), choices[0].up.clone()] {
            let mut child = problem.clone();
            child.branch(choice).unwrap();
            let feasible: Vec<_> = [[low, low_end], [high, high_end]]
                .iter()
                .filter_map(|path| child.elementary_route(path))
                .filter(|route| {
                    let entries = child.base_entries(route).unwrap();
                    child.rows.iter().enumerate().all(|(row, constraint)| {
                        let activity: f64 = entries
                            .iter()
                            .filter(|(r, _)| *r == row)
                            .map(|(_, a)| a)
                            .sum();
                        match constraint.sense {
                            RowSense::Equal => activity == constraint.rhs,
                            RowSense::LessEqual => activity <= constraint.rhs,
                            RowSense::GreaterEqual => activity >= constraint.rhs,
                        }
                    })
                })
                .collect();
            assert_eq!(feasible.len(), 1);
        }
    }
    pub(crate) fn fixture() -> (RootProblem<5>, Vec<(crate::Route, f64)>, Vec<i32>) {
        let mut graph = crate::Graph::<5>::new(6, 5).unwrap();
        for v in 0..6 {
            for r in 0..5 {
                graph.set_bounds(v, r, 0., 10.).unwrap();
            }
        }
        let a = graph.add_arc(0, 1, 1.).unwrap();
        graph.arc_mut(a).unwrap().consumption[4] = 0.5;
        let b = graph.add_arc(0, 2, 1.).unwrap();
        graph.arc_mut(b).unwrap().consumption[4] = 2.5;
        let c = graph.add_arc(0, 4, 1.).unwrap();
        let low = graph.add_arc(1, 3, 1.).unwrap();
        graph.arc_mut(low).unwrap().bounds[4] = Some(Window {
            lower: 0.,
            upper: 0.5,
        });
        let high = graph.add_arc(2, 3, 1.).unwrap();
        graph.arc_mut(high).unwrap().bounds[4] = Some(Window {
            lower: 0.,
            upper: 2.5,
        });
        let end = graph.add_arc(3, 5, 1.).unwrap();
        let e1 = graph.add_arc(1, 5, 1.).unwrap();
        let e2 = graph.add_arc(2, 5, 1.).unwrap();
        let e4 = graph.add_arc(4, 5, 1.).unwrap();
        let problem = RootProblem::set_partitioning(graph, 3.).unwrap();
        let paths = [
            (vec![a, low, end], 0.25),
            (vec![b, high, end], 0.75),
            (vec![a, e1], 0.75),
            (vec![b, e2], 0.25),
            (vec![c, e4], 1.),
        ];
        let routes = paths
            .into_iter()
            .map(|(ids, w)| (problem.elementary_route(&ids).unwrap(), w))
            .collect();
        (problem, routes, vec![low, high])
    }
    #[test]
    fn inherited_suffix_window_limits_intervals_before_selection() {
        let (mut problem, routes, arcs) = fixture();
        let route = &routes[0].0;
        for arc in arcs {
            problem.graph.arc_mut(arc).unwrap().bounds[4] = Some(Window {
                lower: 0.,
                upper: 3.,
            });
        }
        let suffix = *route.arcs.last().unwrap();
        let rule = crate::pricing::restrictions::AccumulatedResourceBranch {
            arc: suffix,
            resource: 4,
            threshold: 1. + 1e-6,
            upper: false,
        };
        let boundary = rule.boundary();
        let restriction = BranchConstraint::AccumulatedResource(rule);
        assert_eq!(
            suffix_upper(&problem, route, 4, &[restriction]).unwrap()[1],
            boundary
        );
    }
}
