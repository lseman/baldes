use super::*;

fn fixture() -> Output<2> {
    let mut graph = Graph::<2>::new(4, 0).unwrap();
    for (tail, head, cost) in [(0, 1, 1.), (1, 3, 0.), (0, 2, 50.), (2, 3, 0.)] {
        graph.add_arc(tail, head, cost).unwrap();
    }
    let problem = Problem {
        rows: vec![Row {
            sense: RowSense::Equal,
            rhs: 1.,
        }],
        networks: vec![Network {
            id: 0,
            graph,
            multiplicity: Window::new(0., 1.).unwrap(),
            arc_entries: vec![vec![(0, 1.)], vec![], vec![(0, 1.)], vec![]],
            route_entries: vec![],
            activity_entries: vec![],
        }],
    };
    let mut options = Options::default();
    options.integer_primal_termination = true;
    options.gap_enumeration = false;
    options.cg.stabilize = false;
    options.preprocessing.enabled = false;
    solve(problem, options).unwrap()
}

#[test]
fn fixing_uses_a_current_dual_bound_and_keeps_the_incumbent_path() {
    let mut output = fixture();
    assert_eq!(output.master.status, cg::Status::Optimal);
    assert_eq!(output.master.lp, 1.);
    assert!(output.snapshot.state.pricing_witness.is_some());
    let parent = output.snapshot.fork().unwrap();
    let duals = vec![0.; output.snapshot.master.rhs.len()];
    apply(&mut output.snapshot.state, &output.snapshot.master, &duals).unwrap();
    assert!(output.snapshot.state.previous_fixing_gap < 1e12);
    let oracle = &mut output.snapshot.state.oracles[0];
    let routes = oracle.solver.enumerate(100.).unwrap();
    assert!(routes.iter().any(|route| route.arcs == [0, 1]));
    assert!(routes.iter().all(|route| route.arcs != [2, 3]));
    let mut untouched = parent;
    let routes = untouched.state.oracles[0].solver.enumerate(100.).unwrap();
    assert!(routes.iter().any(|route| route.arcs == [2, 3]));
}

#[test]
fn fixing_requires_an_integer_witness_and_a_positive_time_budget() {
    let mut output = fixture();
    let duals = vec![0.; output.snapshot.master.rhs.len()];
    output.snapshot.state.pricing_witness = None;
    apply(&mut output.snapshot.state, &output.snapshot.master, &duals).unwrap();
    assert_eq!(output.snapshot.state.previous_fixing_gap, 1e12);
    let mut output = fixture();
    output.snapshot.state.oracles[0]
        .solver
        .options
        .set("BALDESredCostFixingTimeThreshold", 0.)
        .unwrap();
    apply(&mut output.snapshot.state, &output.snapshot.master, &duals).unwrap();
    assert_eq!(output.snapshot.state.previous_fixing_gap, 1e12);
}

#[test]
fn mandatory_network_quantity_is_not_fixed_by_the_optional_quantity_cutoff() {
    let mut output = fixture();
    output.snapshot.state.oracles[0].network.multiplicity.lower = 1.;
    let duals = vec![0.; output.snapshot.master.rhs.len()];
    apply(&mut output.snapshot.state, &output.snapshot.master, &duals).unwrap();
    assert_eq!(output.snapshot.state.previous_fixing_gap, 1e12);
}

#[test]
fn controller_runs_fixing_only_for_numerical_non_enumerating_solves() {
    use crate::master::cg::Controller;
    for (numerical, enumeration, expected) in [
        (true, false, true),
        (false, false, false),
        (true, true, false),
    ] {
        let output = fixture();
        let NodeSnapshot { master, state, .. } = output.snapshot;
        let state = Rc::new(RefCell::new(state));
        let mut controller = Coordinator(
            state.clone(),
            false,
            false,
            None,
            enumeration.then(EnumerationOptions::default),
            None,
            Vec::new(),
            false,
            numerical,
            super::CheckpointStore::default(),
        );
        let duals = vec![0.; master.rhs.len()];
        controller.fix_pricing(&master, &duals).unwrap();
        assert_eq!(state.borrow().previous_fixing_gap < 1e12, expected);
    }
}

#[test]
fn independent_child_restores_paths_fixed_by_its_parents_incumbent() {
    use crate::master::preprocessing::Bounds;
    let mut output = fixture();
    let duals = vec![0.; output.snapshot.master.rhs.len()];
    apply(&mut output.snapshot.state, &output.snapshot.master, &duals).unwrap();
    assert!(output.snapshot.state.objective_fixed);
    let child = output
        .snapshot
        .with_activity_bounds(
            Bounds {
                lower: 0.,
                upper: 0.,
            },
            &[RowCoefficients {
                network: 0,
                arcs: vec![1., 0., 0., 0.],
                per_route: 0.,
            }],
        )
        .unwrap();
    assert!(!child.state.objective_fixed);
    assert_eq!(child.warm.bound, f64::NEG_INFINITY);
    let mut options = Options::default();
    options.preprocessing.enabled = false;
    options.cg.stabilize = false;
    let child = child.solve(options).unwrap();
    assert_eq!(
        child.master.status,
        cg::Status::Optimal,
        "{:?}",
        child.master
    );
    assert!((child.master.lp - 50.).abs() < 1e-8);
    assert!(child
        .routes
        .iter()
        .zip(&child.values)
        .any(|(route, &value)| route.route.arcs == [2, 3] && value > 0.5));
}

#[test]
fn tree_incumbent_fixes_without_a_node_local_witness() {
    let mut output = fixture();
    output.snapshot.state.pricing_witness = None;
    output.snapshot.state.tree_incumbent = 1.;
    let duals = vec![0.; output.snapshot.master.rhs.len()];
    apply(&mut output.snapshot.state, &output.snapshot.master, &duals).unwrap();
    assert!(output.snapshot.state.previous_fixing_gap < 1e12);
    assert!(output.snapshot.state.pricing_witness.is_none());
    let routes = output.snapshot.state.oracles[0]
        .solver
        .enumerate(100.)
        .unwrap();
    assert!(routes.iter().any(|route| route.arcs == [0, 1]));
    assert!(routes.iter().all(|route| route.arcs != [2, 3]));
}
