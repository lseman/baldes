use super::*;

#[test]
fn inequality_duals_are_projected_before_recomputing_the_complete_certificate() {
    let mut graph = Graph::<2>::new(2, 0).unwrap();
    graph.add_arc(0, 1, 2.).unwrap();
    let problem = Problem {
        rows: vec![
            Row {
                sense: RowSense::GreaterEqual,
                rhs: 1.,
            },
            Row {
                sense: RowSense::LessEqual,
                rhs: 2.,
            },
        ],
        networks: vec![Network {
            id: 1,
            graph,
            multiplicity: Window::new(0., 3.).unwrap(),
            arc_entries: vec![vec![]],
            route_entries: vec![(0, 1.), (1, 1.)],
            activity_entries: vec![],
        }],
    };
    let root = solve(problem, Options::default()).unwrap();
    // Physical row zero is the <= multiplicity row. Project all three
    // wrong-sign multipliers, including large violations, to the zero point.
    // Its independently recomputed bound is zero, below the optimum of two.
    for magnitude in [1e-11, 100.] {
        let duals = [magnitude, -magnitude, magnitude];
        let state = &root.snapshot.state;
        let master = &root.snapshot.master;
        assert_eq!(certificate(state, master, &duals), Some(0.));
        assert_eq!(extension_certificate(state, master, &duals), Some(0.));
        assert_eq!(duals, [magnitude, -magnitude, magnitude]);
    }
    for invalid in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert!(certificate(
            &root.snapshot.state,
            &root.snapshot.master,
            &[invalid, 0., 0.]
        )
        .is_none());
    }
}

#[test]
fn rounded_fixed_rhs_cannot_supply_a_new_certificate() {
    let mut graph = Graph::<2>::new(3, 1).unwrap();
    graph.add_arc(0, 1, 1.).unwrap();
    graph.add_arc(1, 2, 0.).unwrap();
    let problem = Problem {
        rows: vec![Row {
            sense: RowSense::Equal,
            rhs: 1.,
        }],
        networks: vec![Network {
            id: 1,
            graph,
            multiplicity: Window::new(0., 20.).unwrap(),
            arc_entries: vec![vec![(0, 0.1)], vec![]],
            route_entries: vec![],
            activity_entries: vec![],
        }],
    };
    let mut options = Options::default();
    options.cg.stabilize = false;
    options.cg.artificial_cost = 100.;
    let root = solve(problem, options).unwrap();
    assert_eq!(root.master.status, cg::Status::Optimal);
    let node = root.snapshot.fix_route(0, 1.).unwrap();
    // f64(0.9) + f64(0.1) differs from one in exact arithmetic.
    assert!(!exact_fixed_reduction(&node.state, &node.master).unwrap());
    assert!(certificate(&node.state, &node.master, &root.duals).is_none());
}
