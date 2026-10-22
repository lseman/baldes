use super::*;

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
