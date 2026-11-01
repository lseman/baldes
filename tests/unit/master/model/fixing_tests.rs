use super::*;

fn fixture() -> Output<2> {
    fixture_with_cost(1.)
}

fn fixture_with_cost(cheap_cost: f64) -> Output<2> {
    let mut graph = Graph::<2>::new(4, 0).unwrap();
    for (tail, head, cost) in [(0, 1, cheap_cost), (1, 3, 0.), (0, 2, 50.), (2, 3, 0.)] {
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
fn cancelling_complement_and_route_offset_keep_an_improving_route() {
    let incumbent = 1.;
    let other = 1e16;
    let offset = -1e16;
    let margin = 2e-6;
    let rounded_ceiling = incumbent - other + margin - offset;
    assert_eq!(rounded_ceiling, 0.);
    let ceiling = pricing_ceiling(incumbent, other, margin, offset).unwrap();
    assert!(ceiling >= incumbent + margin);
    let mut output = fixture_with_cost(0.5);
    let solver = &mut output.snapshot.state.oracles[0].solver;
    solver.run_fixing_enumeration(-1, 0., ceiling).unwrap();
    let routes = solver.enumerate(100.).unwrap();
    assert!(routes.iter().any(|route| route.arcs == [0, 1]));
}

#[test]
fn pricing_cutoff_declines_overflow_instead_of_fixing() {
    assert_eq!(pricing_ceiling(f64::MAX, -f64::MAX, 0., 0.), None);
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
fn mandatory_network_quantity_with_a_closed_bound_needs_no_fixing() {
    let mut output = fixture();
    output.snapshot.state.oracles[0].network.multiplicity.lower = 1.;
    let duals = vec![0.; output.snapshot.master.rhs.len()];
    apply(&mut output.snapshot.state, &output.snapshot.master, &duals).unwrap();
    assert_eq!(output.snapshot.state.previous_fixing_gap, 1e12);
    let routes = output.snapshot.state.oracles[0]
        .solver
        .enumerate(100.)
        .unwrap();
    assert!(routes.iter().any(|route| route.arcs == [0, 1]));
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
        let mut controller = ColumnGenerationController {
            state: state.clone(),
            request_initial_enumeration: false,
            search_inactive_columns: false,
            artificial_penalty: None,
            enumeration_options: enumeration.then(EnumerationOptions::default),
            enumeration_checkpoint: None,
            removed_columns: Vec::new(),
            trust_preprocessing_infeasibility: false,
            allow_objective_fixing: numerical,
            checkpoints: super::CheckpointStore::default(),
        };
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
