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
fn gap_enumeration_rollback_restores_routes_tags_and_pricing_domain() {
    let mut output = fixture();
    let state = &mut output.snapshot.state;
    state.gap_enumeration = true;
    state.tree_incumbent = 2.;
    let count = state.routes.len();
    let tags = state.tags.clone();
    let duals = vec![0.; output.snapshot.master.rhs.len()];
    assert!(begin(state, 0.).unwrap());
    let columns = universe(state, &duals, 0.).unwrap().unwrap();
    assert!(columns.iter().any(|column| column.cost == 1.));
    assert!(columns.iter().all(|column| column.cost < 2.));
    assert!(state.oracles[0].solver.enumeration.is_some());
    finish(state, false);
    assert_eq!(state.routes.len(), count);
    assert_eq!(state.tags, tags);
    assert_eq!(state.previous_fixing_gap, 1e12);
    assert!(!state.objective_fixed);
    assert!(state.oracles[0].enumerated.is_empty());
    assert!(state.oracles[0].solver.enumeration.is_none());
}

#[test]
fn committed_gap_pool_is_dropped_on_an_independent_node_fork() {
    let mut output = fixture();
    let state = &mut output.snapshot.state;
    state.gap_enumeration = true;
    state.tree_incumbent = 2.;
    let duals = vec![0.; output.snapshot.master.rhs.len()];
    assert!(begin(state, 0.).unwrap());
    assert!(universe(state, &duals, 0.).unwrap().is_some());
    finish(state, true);
    assert!(state.objective_fixed);
    let child = output.snapshot.fork().unwrap();
    assert!(child.state.oracles[0].solver.enumeration.is_none());
    assert!(output.snapshot.state.oracles[0]
        .solver
        .enumeration
        .is_some());
    assert_eq!(child.state.tree_incumbent, f64::INFINITY);
}

#[test]
fn lp_only_and_mandatory_quantity_models_decline_gap_enumeration() {
    let mut output = fixture();
    let state = &mut output.snapshot.state;
    state.gap_enumeration = true;
    state.integer_primal_termination = false;
    assert!(!begin(state, 0.).unwrap());
    state.tree_incumbent = 2.;
    state.oracles[0].network.multiplicity.lower = 1.;
    assert!(!begin(state, 0.).unwrap());
}

#[test]
fn retained_gap_pricing_uses_registered_pool_without_whole_domain_enumeration() {
    use crate::master::pricing::Networks;
    let mut output = fixture();
    let state = &mut output.snapshot.state;
    state.gap_enumeration = true;
    state.tree_incumbent = 2.;
    let duals = vec![0.; output.snapshot.master.rhs.len()];
    assert!(begin(state, 0.).unwrap());
    assert!(universe(state, &duals, 0.).unwrap().is_some());
    finish(state, true);
    assert!(state.universe.is_none());
    assert!(!state.enumeration_report.complete);
    let tolerance = state.scheduler.tolerance;
    let retained = adapter::NetworkAdapter { state, tolerance }
        .retained(0, &duals)
        .unwrap()
        .unwrap();
    assert!(!retained.columns.is_empty());
    assert!(retained.columns.iter().all(|(column, _)| column.cost < 2.));
}

#[test]
fn gap_schedule_requires_a_strict_ten_percent_reduction_and_rejects_nested_transactions() {
    let mut output = fixture();
    let state = &mut output.snapshot.state;
    state.gap_enumeration = true;
    state.tree_incumbent = 100.;
    state.previous_fixing_gap = 100.;
    assert!(!begin(state, 10.).unwrap());
    assert!(begin(state, 11.).unwrap());
    assert!(begin(state, 11.).is_err());
    finish(state, false);
    assert_eq!(state.previous_fixing_gap, 100.);
}
