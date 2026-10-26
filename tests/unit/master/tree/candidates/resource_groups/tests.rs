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
    let mut problem = RoutingContext::set_partitioning(graph, 1.).unwrap();
    // The two physical heads are alternatives for one original unit cover.
    let high_position = problem.arc_position(high).unwrap();
    problem.model.networks[0].arc_entries[high_position] = vec![(1, 1.)];
    problem.model.rows[2].sense = RowSense::LessEqual;
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
    for (index, row) in problem.model.rows.iter().enumerate() {
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
                child
                    .model
                    .rows
                    .iter()
                    .enumerate()
                    .all(|(row, constraint)| {
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
pub(crate) fn fixture() -> (RoutingContext<5>, Vec<(crate::Route, f64)>, Vec<i32>) {
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
    let problem = RoutingContext::set_partitioning(graph, 3.).unwrap();
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
        problem.model.networks[0].graph.arc_mut(arc).unwrap().bounds[4] = Some(Window {
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
