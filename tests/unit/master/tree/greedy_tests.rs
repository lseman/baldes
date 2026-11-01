use super::*;

#[test]
fn shifts_positive_zero_and_negative_costs_with_recovered_margin() {
    assert_eq!(shift([1., 3., 2.]), 3. * 1.1);
    assert_eq!(shift([0., 0.]), 1.);
    assert_eq!(shift([-3., -2.]), 0.);
    assert_eq!(shift([5e-7]), 1.);
}

fn fixture() -> Problem<2> {
    let mut graph = Graph::<2>::new(4, 0).unwrap();
    for (tail, head, cost) in [(0, 1, 0.), (1, 3, 1.), (0, 2, 0.), (2, 3, 1.), (1, 2, 0.)] {
        graph.add_arc(tail, head, cost).unwrap();
    }
    Problem::set_partitioning(graph, 2.).unwrap()
}

#[test]
fn residual_greedy_builds_and_replays_the_pair_before_root_column_generation() {
    let mut options = ModelTreeOptions::default();
    options.greedy = true;
    options.node.cg.max_iterations = 1;
    let output = crate::master::tree::solve(fixture(), options).unwrap();
    assert_eq!(output.incumbent, 1., "{output:?}");
    assert_eq!(output.routes.len(), 1);
    assert_eq!(output.routes[0].quantity, 1.);
    assert!(output.greedy_nodes >= 2);
    assert!(output.iterations <= 1);
    assert!(output.bound <= 1.);
}

#[test]
fn greedy_private_fixing_keeps_original_root_multiplicity_and_rows() {
    let options = Options::default();
    let (master, state, configured) =
        solve::prepare_with_seed_routes(fixture(), vec![], options, &[], &[], &[]).unwrap();
    let rhs = state.rhs.clone();
    let mut tree = ModelTreeOptions::default();
    tree.greedy = true;
    tree.node.cg.max_iterations = 1;
    // Exercise the public snapshot resume too; the base root remains reusable.
    let mut configured = configured;
    let base = snapshot(master, state, &mut configured);
    let output = base.solve_tree(tree).unwrap();
    assert_eq!(output.incumbent, 1.);
    assert_eq!(base.state.rhs, rhs);
    assert_eq!(base.state.oracles[0].network.multiplicity.upper, 2.);
    assert!(base.state.fixed_routes.is_empty());
}

#[test]
fn registered_variable_shift_follows_associations_instead_of_physical_arc_count() {
    let problem = fixture();
    let network = problem.networks[0].id;
    let variables = vec![OriginalPricingVariable {
        network,
        id: 17,
        cost: 3.,
        entries: vec![],
        arcs: vec![(0, 2.), (1, 1.)],
        accumulated_resource: None,
        integer: false,
        priority: 0.,
    }];
    let (problem, registry) = compile_pricing_variables(problem, &variables).unwrap();
    let mut configured = Options::default();
    registry.configure_pricing(&mut configured);
    let (master, state, mut configured) =
        solve::prepare_with_seed_routes(problem, vec![], configured, &[], &[], &[]).unwrap();
    let node = snapshot(master, state, &mut configured);
    let shifted = costs(&node, 0).unwrap();
    let graph = &node.state.oracles[0].network.graph;
    assert!((shifted.arcs[0] - (graph.arcs[0].cost - 2. * 3.3)).abs() < 1e-12);
    assert!((shifted.arcs[1] - (graph.arcs[1].cost - 3.3)).abs() < 1e-12);
    assert_eq!(shifted.arcs[2], graph.arcs[2].cost);
}

#[test]
fn zero_node_budget_does_not_start_the_initial_heuristic() {
    let mut options = ModelTreeOptions::default();
    options.greedy = true;
    options.node_limit = 0;
    let output = crate::master::tree::solve(fixture(), options).unwrap();
    assert_eq!(output.greedy_nodes, 0);
    assert_eq!(output.pricing_calls, 0);
}

#[test]
fn greedy_does_not_hide_an_invalid_caller_initial_solution() {
    let mut options = ModelTreeOptions::default();
    options.greedy = true;
    options.initial_solution = Some((0., vec![]));
    assert!(crate::master::tree::solve(fixture(), options)
        .unwrap_err()
        .message
        .contains("initial tree solution"));
}

#[cfg(feature = "highs")]
#[test]
fn greedy_mip_stage_can_complete_original_static_variables_without_networks() {
    let problem = Problem::<2> {
        rows: vec![],
        networks: vec![],
    };
    let variables = vec![MasterVariable {
        id: 9,
        cost: 3.,
        bounds: crate::master::preprocessing::Bounds {
            lower: 2.,
            upper: 2.,
        },
        integer: true,
        entries: vec![],
    }];
    let mut options = ModelTreeOptions::default();
    options.greedy = true;
    options.node.cg.max_iterations = 1;
    options.restricted_master_mip = Some(RestrictedMasterMip::default());
    let output = crate::master::tree::solve((problem, variables), options).unwrap();
    assert_eq!(output.incumbent, 6.);
    assert_eq!(output.master_values, vec![(9, 2.)]);
    assert_eq!(output.greedy_nodes, 1);
    assert!(output.restricted_mip_calls >= 1);
}

#[test]
fn direct_initial_construction_uses_no_master_lp_iterations() {
    let problem = fixture();
    let options = ModelTreeOptions::default();
    let mut zero = options.clone();
    zero.node_limit = 0;
    let mut result = crate::master::tree::solve(problem.clone(), zero).unwrap();
    let (master, state, configured) =
        solve::prepare_with_seed_routes(problem, vec![], options.node.clone(), &[], &[], &[])
            .unwrap();
    run(
        &master,
        &state,
        &configured,
        None,
        &options,
        &mut result,
        Instant::now() + Duration::from_secs(10),
    )
    .unwrap();
    assert_eq!(result.incumbent, 1.);
    assert_eq!(result.iterations, 0);
    assert_eq!(result.greedy_nodes, 2);
    assert_eq!(master.simplex_iterations, 0);
}

#[test]
fn greedy_resumed_enumerated_snapshot_uses_shifted_costs_without_old_cut_prices() {
    let output = crate::master::model::solve(fixture(), Options::default()).unwrap();
    let node = output
        .snapshot
        .with_rank_one_row(
            Row {
                sense: RowSense::LessEqual,
                rhs: 3.,
            },
            &[RankOneCoefficients {
                network: 0,
                weights: vec![0, 1, 1, 0],
                denominator: 2,
            }],
        )
        .unwrap();
    let mut node = node.enumerate(&EnumerationOptions::default()).unwrap();
    assert!(node.state.universe.is_some());
    node.state.oracles[0].solver.cuts.cuts[0].dual = -100.;
    let options = ModelTreeOptions::default();
    let mut zero = options.clone();
    zero.node_limit = 0;
    let mut result = crate::master::tree::solve(fixture(), zero).unwrap();
    run(
        &node.master,
        &node.state,
        &options.node,
        Some(&node.warm),
        &options,
        &mut result,
        Instant::now() + Duration::from_secs(10),
    )
    .unwrap();
    assert_eq!(result.incumbent, 1.);
    assert_eq!(result.greedy_nodes, 2);
    assert_eq!(node.state.oracles[0].solver.cuts.cuts[0].dual, -100.);
    assert!(node.state.universe.is_some());
}
