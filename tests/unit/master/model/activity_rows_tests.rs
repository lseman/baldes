use super::*;
use crate::graph::coupled::CoupledResource;
use crate::master::formulation::Row;

#[test]
fn uncertain_intrinsic_rows_do_not_authorize_exact_fixed_or_integer_contributions() {
    let mut graph = Graph::<2>::new(4, 0).unwrap();
    graph.elementary = true;
    for (tail, head, cost) in [(0, 1, -1.), (1, 2, 0.), (2, 3, 0.)] {
        graph.add_arc(tail, head, cost).unwrap();
    }
    graph
        .set_coupled_resources(&[CoupledResource::CustomResource {
            increments: vec![1., 2f64.powi(-53), -1.],
            weights: vec![0., 0., 1.],
            forward_bound: None,
            backward_bound: None,
        }])
        .unwrap();
    graph.suppress_coupled_costs();
    let problem = Problem {
        rows: vec![],
        networks: vec![Network {
            id: 42,
            graph,
            multiplicity: Window::new(0., 1.).unwrap(),
            arc_entries: vec![vec![]; 3],
            route_entries: vec![],
            activity_entries: vec![],
        }],
    };
    let mut options = Options::default();
    options.cg.stabilize = false;
    let root = crate::master::model::solve(problem, options).unwrap();
    let node = root
        .snapshot
        .with_accumulated_activity_row(
            Row {
                sense: RowSense::LessEqual,
                rhs: 0.,
            },
            &[AccumulatedActivityCoefficients {
                network: 42,
                resource: 0,
                coefficient: 1.,
            }],
            &[],
        )
        .unwrap();
    let oracle = &node.state.oracles[0];
    let kernel = oracle.network.graph.coupled_kernel().unwrap();
    assert_eq!(kernel.accumulated_activity(0, &[0, 1, 2]).unwrap(), 0.);
    let (valid, enclosure) = kernel
        .certify_accumulated_activity(0, &[0, 1, 2], &mut 100_000, None)
        .unwrap();
    assert!(valid && enclosure[0] <= 2f64.powi(-53) && enclosure[1] >= 2f64.powi(-53));
    assert_ne!(enclosure[0], enclosure[1]);
    let routes = vec![FixedRoute {
        route: root.routes[0].clone(),
        quantity: 1.,
    }];
    assert!(
        crate::master::model::bounds::exact_contributions(&node.state, &routes)
            .unwrap()
            .is_none()
    );
    assert!(super::minimum(
        oracle,
        node.state.shared_offset,
        &[0.; 2],
        &mut 0,
        node.state.deadline
    )
    .is_none());
    assert!(super::minimum(
        oracle,
        node.state.shared_offset,
        &[0.; 2],
        &mut 100_000,
        Instant::now()
    )
    .is_none());
}
