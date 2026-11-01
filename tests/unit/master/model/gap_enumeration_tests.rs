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
    state.pricing_witness = None;
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
    state.pricing_witness = None;
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
    state.pricing_witness = None;
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
    state.pricing_witness = None;
    let duals = vec![0.; output.snapshot.master.rhs.len()];
    assert!(begin(state, 0.).unwrap());
    assert!(universe(state, &duals, 0.).unwrap().is_some());
    finish(state, true);
    assert!(state.universe.is_none());
    assert!(!state.enumeration_report.complete);
    let tolerance = state.scheduler.tolerance;
    let retained = pricing::NetworkPricingAdapter { state, tolerance }
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
    state.pricing_witness = None;
    state.previous_fixing_gap = 100.;
    assert!(!begin(state, 10.).unwrap());
    assert!(begin(state, 11.).unwrap());
    assert!(begin(state, 11.).is_err());
    finish(state, false);
    assert_eq!(state.previous_fixing_gap, 100.);
}

#[test]
fn resumed_solve_receives_incumbent_authority_only_from_current_options() {
    let output = fixture();
    let mut configured = Options::default();
    configured.integer_primal_termination = true;
    configured.gap_enumeration = false;
    configured.fixing_upper_bound = Some(2.);
    configured.cg.stabilize = false;
    configured.preprocessing.enabled = false;
    let solved = output.snapshot.solve(configured).unwrap();
    assert_eq!(solved.snapshot.state.tree_incumbent, 2.);
    let mut lp = Options::default();
    lp.gap_enumeration = false;
    lp.preprocessing.enabled = false;
    let resumed = solved.snapshot.solve(lp).unwrap();
    assert_eq!(resumed.snapshot.state.tree_incumbent, f64::INFINITY);
    assert!(!resumed.snapshot.state.objective_fixed);
}

fn multi_network_fixture() -> Output<2> {
    let source = fixture();
    let original = &source.snapshot.state.oracles[0].network;
    let network = Network {
        id: 0,
        graph: (*original.graph).clone(),
        multiplicity: original.multiplicity,
        arc_entries: (*original.arc_entries).clone(),
        route_entries: vec![],
        activity_entries: vec![],
    };
    let mut second = network.clone();
    second.id = 7;
    let problem = Problem {
        rows: vec![Row {
            sense: RowSense::Equal,
            rhs: 1.,
        }],
        networks: vec![network, second],
    };
    let variable = MasterVariable {
        id: 99,
        cost: 100.,
        integer: true,
        bounds: crate::master::preprocessing::Bounds {
            lower: 0.,
            upper: 1.,
        },
        entries: vec![(0, 1.)],
    };
    let mut options = Options::default();
    options.integer_primal_termination = true;
    options.gap_enumeration = false;
    options.cg.stabilize = false;
    options.preprocessing.enabled = false;
    solve_with_master_variables(problem, vec![variable], options).unwrap()
}

#[test]
fn multi_network_gap_plan_includes_master_variables_and_restores_all_oracles() {
    let mut output = multi_network_fixture();
    let state = &mut output.snapshot.state;
    state.gap_enumeration = true;
    state.tree_incumbent = 2.;
    state.pricing_witness = None;
    let duals = vec![0.; output.snapshot.master.rhs.len()];
    let mut work = 1_000_000;
    let plan = bounds::cutoff_plan(
        state,
        &output.snapshot.master,
        &duals,
        2.,
        &mut work,
        state.deadline,
    )
    .unwrap();
    state.fixing_cutoffs = Some(
        plan.into_iter()
            .map(|cutoff| match cutoff {
                bounds::NetworkCutoff::Empty { .. } => None,
                bounds::NetworkCutoff::Cost {
                    other_contributions,
                    ..
                } => Some(2. - other_contributions),
            })
            .collect(),
    );
    assert!(begin(state, 0.).unwrap());
    let columns = universe(state, &duals, 0.).unwrap().unwrap();
    for network in [0, 7] {
        assert!(columns.iter().any(
            |column| state.routes[column.tag as usize].network == network && column.cost == 1.
        ));
    }
    assert_eq!(state.gap_pool_cutoff, Some(2.));
    finish(state, false);
    assert!(state
        .oracles
        .iter()
        .all(|oracle| oracle.solver.enumeration.is_none() && oracle.enumerated.is_empty()));
    assert!(state.gap_pool_cutoff.is_none());
}

#[cfg(feature = "highs")]
#[test]
fn retained_multi_network_pool_hands_off_to_mip_with_finite_coverage() {
    let mut output = multi_network_fixture();
    let state = &mut output.snapshot.state;
    state.gap_enumeration = true;
    state.tree_incumbent = 2.;
    state.pricing_witness = None;
    state.fixing_cutoffs = Some(vec![Some(2.), Some(2.)]);
    let duals = vec![0.; output.snapshot.master.rhs.len()];
    assert!(begin(state, 0.).unwrap());
    assert!(universe(state, &duals, 0.).unwrap().is_some());
    finish(state, true);
    let mip = output
        .snapshot
        .solve_enumerated_mip(&Default::default())
        .unwrap()
        .unwrap();
    assert_eq!(mip.coverage_cutoff, 2.);
    assert_eq!(mip.mip.solution.unwrap().objective, 1.);
    assert!(mip.mip.bound <= 2.);
    assert!(output.snapshot.state.universe.is_none());
}

#[cfg(feature = "highs")]
#[test]
fn cg_post_install_handoff_retains_a_validated_original_incumbent() {
    use crate::master::cg::Controller;
    let mut output = multi_network_fixture();
    output.snapshot.state.gap_enumeration = true;
    output.snapshot.state.tree_incumbent = 2.;
    output.snapshot.state.pricing_witness = None;
    output.snapshot.state.fixing_cutoffs = Some(vec![Some(2.), Some(2.)]);
    output.snapshot.state.gap_mip = Some(Default::default());
    let duals = vec![0.; output.snapshot.master.rhs.len()];
    assert!(begin(&mut output.snapshot.state, 0.).unwrap());
    universe(&mut output.snapshot.state, &duals, 0.)
        .unwrap()
        .unwrap();
    finish(&mut output.snapshot.state, true);
    let NodeSnapshot {
        master,
        warm,
        state,
    } = output.snapshot;
    let state = Rc::new(RefCell::new(state));
    let mut controller = ColumnGenerationController {
        state: state.clone(),
        request_initial_enumeration: false,
        search_inactive_columns: false,
        artificial_penalty: None,
        enumeration_options: None,
        enumeration_checkpoint: None,
        removed_columns: vec![],
        trust_preprocessing_infeasibility: false,
        allow_objective_fixing: true,
        checkpoints: CheckpointStore::default(),
    };
    controller
        .enumeration_installed(&master, &warm.columns, &warm.artificials)
        .unwrap();
    assert_eq!(state.borrow().pricing_witness.as_ref().unwrap().0, 1.);
    for column in warm
        .columns
        .iter()
        .flatten()
        .filter(|column| master_variables::decode(column.tag).is_some())
    {
        assert!(controller.preserve_enumeration_column(column));
    }
}

#[test]
fn tree_rejects_an_initial_objective_that_understates_its_route_witness() {
    let output = fixture();
    let routes = output
        .snapshot
        .state
        .pricing_witness
        .as_ref()
        .unwrap()
        .1
        .clone();
    let mut options = tree_adapter::ModelTreeOptions::default();
    options.node.gap_enumeration = false;
    options.initial_solution = Some((0., routes));
    let error = output.snapshot.solve_tree(options).unwrap_err();
    assert!(error.message.contains("understates"), "{error}");
}

#[test]
fn native_false_gap_shrink_is_recorded_as_narrower_coverage() {
    let mut output = fixture();
    let state = &mut output.snapshot.state;
    state.gap_enumeration = true;
    state.tree_incumbent = 2.;
    state.pricing_witness = None;
    state.oracles[0]
        .solver
        .options
        .set("BALDESredCostFixingFalseGap", 2.)
        .unwrap();
    let duals = vec![0.; output.snapshot.master.rhs.len()];
    assert!(begin(state, 0.).unwrap());
    universe(state, &duals, 0.).unwrap().unwrap();
    finish(state, true);
    assert!(state.gap_pool_cutoff.unwrap() <= 1.);
    assert!(state.gap_pool_cutoff.unwrap() > 0.99);
}

#[test]
fn fixing_preparation_refreshes_raw_duals_with_a_retained_pool() {
    let mut output = fixture();
    let state = &mut output.snapshot.state;
    state.gap_enumeration = true;
    state.tree_incumbent = 2.;
    state.pricing_witness = None;
    let mut duals = vec![0.; output.snapshot.master.rhs.len()];
    assert!(begin(state, 0.).unwrap());
    universe(state, &duals, 0.).unwrap().unwrap();
    finish(state, true);
    let coverage = state.gap_pool_cutoff;
    duals[state.shared_offset] = 0.25;
    pricing::cache_duals(state, 0, &duals).unwrap();
    assert_eq!(state.oracles[0].costs[0], 0.75);
    assert!(state.oracles[0].solver.enumeration.is_some());
    assert_eq!(state.gap_pool_cutoff, coverage);
}

#[test]
fn fixing_only_mode_commits_without_requiring_an_enumerated_universe() {
    let mut output = fixture();
    let state = &mut output.snapshot.state;
    state.gap_enumeration = true;
    state.tree_incumbent = 2.;
    state.pricing_witness = None;
    let duals = vec![0.; output.snapshot.master.rhs.len()];
    assert!(begin(state, -10.).unwrap());
    assert!(universe(state, &duals, -10.).unwrap().is_none());
    finish(state, true);
    assert_eq!(state.previous_fixing_gap, 12.);
    assert!(state.gap_pool_cutoff.is_none());
    assert!(state.oracles[0].solver.enumeration.is_none());
}

#[test]
fn regeneration_keeps_ng_transitions_network_local_and_refreshes_between_calls() {
    let mut graph = Graph::<2>::new(4, 0).unwrap();
    for (tail, head) in [(0, 1), (1, 2), (2, 1), (1, 3)] {
        graph.add_arc(tail, head, 1.).unwrap();
    }
    let mut networks = Vec::new();
    for (id, memory) in [(0, vec![1, 2]), (7, vec![2])] {
        let mut graph = graph.clone();
        graph.set_ng(2, &memory).unwrap();
        networks.push(Network {
            id,
            graph,
            multiplicity: Window::new(0., 1.).unwrap(),
            arc_entries: vec![vec![]; 4],
            route_entries: vec![],
            activity_entries: vec![],
        });
    }
    let (master, mut state, _) = solve::prepare_with_seed_routes(
        Problem {
            rows: vec![],
            networks,
        },
        vec![],
        Options::default(),
        &[],
        &[],
        &[],
    )
    .unwrap();
    state.tree_incumbent = 100.;
    let route = |cycle: bool| Route {
        coupled_values: vec![],
        nodes: if cycle {
            vec![0, 1, 2, 1, 3]
        } else {
            vec![0, 1, 3]
        },
        arcs: if cycle { vec![0, 1, 2, 3] } else { vec![0, 3] },
        cost: if cycle { 4. } else { 2. },
        original_cost: if cycle { 4. } else { 2. },
        resources: vec![],
        resource_profile: vec![vec![]; if cycle { 5 } else { 3 }],
    };
    let tags: Vec<_> = [(0, true), (7, true), (0, false), (7, false)]
        .into_iter()
        .map(|(id, cycle)| register(&mut state, id, route(cycle)).unwrap())
        .collect();
    let duals = vec![0.; master.rhs.len()];
    let pool: Vec<_> = tags
        .iter()
        .chain(&tags)
        .map(|&tag| column(&state, tag, &duals).unwrap())
        .collect();
    let accepted = |state: &mut State<2>| {
        regenerate(state, &pool, &duals, 0.)
            .unwrap()
            .into_iter()
            .map(|c| c.tag)
            .collect::<Vec<_>>()
    };
    assert_eq!(accepted(&mut state), [tags[1], tags[2], tags[3]].repeat(2));
    state.oracles[0].solver.graph.set_ng(2, &[2]).unwrap();
    state.oracles[1].solver.graph.set_ng(2, &[1, 2]).unwrap();
    assert_eq!(accepted(&mut state), [tags[0], tags[2], tags[3]].repeat(2));
    state.oracles[0].solver.forbidden_arcs.resize(4, false);
    state.oracles[0].solver.forbidden_arcs[1] = true;
    assert_eq!(accepted(&mut state), [tags[2], tags[3]].repeat(2));
}

#[test]
fn gap_rollback_without_registration_keeps_shared_registries() {
    let mut output = fixture();
    let sibling = output.snapshot.fork().unwrap();
    let state = &mut output.snapshot.state;
    state.gap_enumeration = true;
    state.tree_incumbent = 2.;
    state.pricing_witness = None;
    assert!(begin(state, 0.).unwrap());
    finish(state, false);
    assert!(Arc::ptr_eq(&state.routes, &sibling.state.routes));
    assert!(Arc::ptr_eq(&state.tags, &sibling.state.tags));
}
