use super::*;
use crate::graph::RouteDomain;

#[test]
fn repeated_arc_costs_use_grouped_original_variable_order() {
    let mut graph = Graph::<2>::new(3, 0).unwrap();
    graph.add_arc(0, 1, 0.1).unwrap();
    graph.add_arc(1, 1, 0.1).unwrap();
    graph.add_arc(1, 2, 0.2).unwrap();
    graph
        .set_route_domain(RouteDomain::BoundedWalks { max_arcs: 8 })
        .unwrap();
    let problem = Problem {
        rows: vec![Row {
            sense: RowSense::Equal,
            rhs: 1.,
        }],
        networks: vec![Network {
            id: 0,
            graph,
            multiplicity: Window::new(0., 1.).unwrap(),
            arc_entries: vec![vec![(0, 1.)], vec![], vec![]],
            route_entries: vec![],
            activity_entries: vec![],
        }],
    };
    let output = solve(problem, Options::default()).unwrap();
    let route = Route {
        nodes: vec![0, 1, 1, 1, 1, 1, 1, 1, 2],
        arcs: vec![0, 1, 1, 1, 1, 1, 1, 2],
        cost: 0.,
        original_cost: 0.,
        resources: vec![],
        resource_profile: vec![],
        coupled_values: vec![],
    };
    let expected = 0.1_f64 + 0.1_f64 * 6. + 0.2_f64;
    let mut repeated_addition = 0.1_f64;
    for _ in 0..6 {
        repeated_addition += 0.1;
    }
    repeated_addition += 0.2;
    assert_ne!(expected.to_bits(), repeated_addition.to_bits());
    assert_eq!(
        route_cost(&output.snapshot.state.oracles[0], &route)
            .unwrap()
            .to_bits(),
        expected.to_bits()
    );
}
