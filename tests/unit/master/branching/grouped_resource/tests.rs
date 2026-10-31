use super::*;
use crate::master::model::Problem;

fn fixture() -> (Problem<5>, Vec<Route>, Vec<i32>) {
    let mut graph = Graph::<5>::new(6, 5).unwrap();
    for vertex in 0..6 {
        for resource in 0..5 {
            graph.set_bounds(vertex, resource, 0., 10.).unwrap();
        }
    }
    let a = graph.add_arc(0, 1, 1.).unwrap();
    graph.arc_mut(a).unwrap().consumption[4] = 0.5;
    let b = graph.add_arc(0, 2, 1.).unwrap();
    graph.arc_mut(b).unwrap().consumption[4] = 2.5;
    let c = graph.add_arc(0, 4, 1.).unwrap();
    let low = graph.add_arc(1, 3, 1.).unwrap();
    graph.arc_mut(low).unwrap().bounds[4] = Some(crate::graph::Window {
        lower: 0.,
        upper: 0.5,
    });
    let high = graph.add_arc(2, 3, 1.).unwrap();
    graph.arc_mut(high).unwrap().bounds[4] = Some(crate::graph::Window {
        lower: 0.,
        upper: 2.5,
    });
    let end = graph.add_arc(3, 5, 1.).unwrap();
    let e1 = graph.add_arc(1, 5, 1.).unwrap();
    let e2 = graph.add_arc(2, 5, 1.).unwrap();
    let e4 = graph.add_arc(4, 5, 1.).unwrap();
    let routes = [
        vec![a, low, end],
        vec![b, high, end],
        vec![a, e1],
        vec![b, e2],
        vec![c, e4],
    ]
    .into_iter()
    .map(|path| graph.elementary_route(&path).unwrap().unwrap())
    .collect();
    (
        Problem::set_partitioning(graph, 3.).unwrap(),
        routes,
        vec![low, high],
    )
}

#[test]
fn blocked_members_remain_optional_and_old_row_decompositions_are_frozen() {
    let (problem, routes, arcs) = fixture();
    let network = &problem.networks[0];
    let packing = network.arc_entries[arcs[0] as usize][0].0;
    let upper = AccumulatedResourceGroupBranch::new(packing, arcs.clone(), 4, 1. + 1e-6, true)
        .unwrap()
        .prepare(&network.graph)
        .unwrap();
    assert_eq!(upper.coefficient(&routes[0], false), 1.);
    assert_eq!(upper.coefficient(&routes[1], false), 0.);
    let lower = AccumulatedResourceGroupBranch::new(packing, arcs, 4, 2. + 1e-6, false)
        .unwrap()
        .prepare(&network.graph)
        .unwrap();
    assert_eq!(upper.coefficient(&routes[0], false), 1.);
    assert_eq!(upper.coefficient(&routes[1], false), 0.);
    assert_eq!(lower.coefficient(&routes[1], false), 1.);
}

#[test]
fn group_disjunction_requires_an_actual_unit_packing_proof() {
    let (mut problem, _, arcs) = fixture();
    let network = &problem.networks[0];
    let packing = network.arc_entries[arcs[0] as usize][0].0;
    let group = AccumulatedResourceGroupBranch::new(packing, arcs, 4, 1., false).unwrap();
    let validate = |p: &Problem<5>| {
        group.validate_support(
            &p.networks[0].graph,
            &p.rows,
            &p.networks[0].arc_entries,
            &p.networks[0].route_entries,
        )
    };
    validate(&problem).unwrap();
    problem.rows[packing].rhs = 2.;
    assert!(validate(&problem).is_err());
    problem.rows[packing].rhs = 1.;
    problem.networks[0].arc_entries[0].push((packing, -0.25));
    assert!(validate(&problem).is_err());
    assert!(AccumulatedResourceGroupBranch::new(0, vec![1, 1], 0, 1., false).is_err());
}

#[test]
fn repeated_group_violations_have_one_coefficient_including_blocked_arcs() {
    let (problem, routes, arcs) = fixture();
    let network = &problem.networks[0];
    let packing = network.arc_entries[arcs[0] as usize][0].0;
    let group = AccumulatedResourceGroupBranch::new(packing, arcs, 4, 1. + 1e-6, true)
        .unwrap()
        .prepare(&network.graph)
        .unwrap();
    let mut repeated = routes[0].clone();
    repeated.arcs.extend_from_within(..);
    let heads = repeated.resource_profile[1..].to_vec();
    repeated.resource_profile.extend(heads);
    assert_eq!(group.coefficient(&repeated, false), 1.);
    if super::super::ACCUMULATED_RESOURCE_IN_PRICING {
        let additive: f64 = repeated.arcs.iter().map(|&arc| group.additive(arc)).sum();
        assert_eq!(additive + group.coefficient(&repeated, true), 1.);
    }
}

#[test]
fn empty_single_arc_window_is_detected_without_changing_the_graph() {
    let (problem, _, arcs) = fixture();
    let network = &problem.networks[0];
    let packing = network.arc_entries[arcs[0] as usize][0].0;
    let group = AccumulatedResourceGroupBranch::new(packing, vec![arcs[0]], 4, 100., true)
        .unwrap()
        .prepare(&network.graph)
        .unwrap();
    assert_eq!(group.additive(arcs[0]), 1.);
    assert_eq!(
        network.graph.arcs[arcs[0] as usize].bounds[4]
            .unwrap()
            .upper,
        0.5
    );
}
