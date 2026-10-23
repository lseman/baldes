use super::*;
#[test]
fn adverse_unbounded_route_duals_keep_bounds_unknown_before_and_after_enumeration() {
    let mut problem: Problem<2> = format::Model::from_json(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/examples/networks/triangle.json"
    )))
    .unwrap()
    .problem()
    .unwrap();
    for n in &mut problem.networks {
        n.multiplicity.upper = f64::INFINITY;
    }
    let mut options = Options::default();
    options.cg.stabilize = false;
    options.cg.artificial_cost = 100.;
    options.preprocessing.enabled = false;
    let root = solve(problem, options.clone()).unwrap();
    for enumerated in [false, true] {
        let mut node = if enumerated {
            root.snapshot
                .enumerate(&EnumerationOptions::default())
                .unwrap()
        } else {
            root.snapshot.fork().unwrap()
        };
        let duals = vec![1.; node.state.rhs.len()];
        let phase = node.state.oracles[0]
            .solver
            .phases
            .iter()
            .position(|p| p.exact())
            .unwrap();
        let answer = node.state.price(phase, &duals).unwrap();
        assert!(answer.exact);
        assert!(!answer.columns.is_empty());
        assert!(node.state.correction.is_none());
        assert!(node.state.gradient.is_none());
        assert!(bounds::certificate(&node.state, &node.master, &duals).is_none());
        assert!(!node
            .state
            .route_upper_bounds
            .values()
            .any(|&u| u == f64::MAX));
        if enumerated {
            let tag = node
                .state
                .routes
                .iter()
                .position(|r| r.network == 7 && r.route.arcs == [0, 1])
                .unwrap();
            let fixed = node.fix_route(tag, 1.).unwrap();
            let solved = fixed.solve(options.clone()).unwrap();
            assert_eq!(solved.master.status, cg::Status::Optimal);
            assert!((solved.master.lp - 2.).abs() < 1e-8);
        }
    }
}
