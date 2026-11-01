use super::*;

fn triangle() -> Problem<2> {
    format::Model::from_json(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/examples/networks/triangle.json"
    )))
    .unwrap()
    .problem()
    .unwrap()
}

fn node_options() -> Options {
    let mut options = Options::default();
    options.cg.stabilize = false;
    options.cg.artificial_cost = 100.;
    options
}

#[test]
fn local_search_defaults_keep_the_disabled_depth_gate_and_effective_controls() {
    let default = LocalSearchOptions::default();
    assert!(!default.due(0));
    assert_eq!(default.fixed_column_ratio, 0.8);
    assert_eq!(default.max_iterations, 3);
    let enabled = LocalSearchOptions {
        depth_limit: 2,
        ..default.clone()
    };
    assert!(enabled.due(0));
    assert!(enabled.due(2));
    assert!(!enabled.due(3));
    for ratio in [-0.1, 1.1, f64::NAN] {
        assert!(LocalSearchOptions {
            fixed_column_ratio: ratio,
            ..default.clone()
        }
        .validate()
        .is_err());
    }
    let mut overridden = default;
    overridden.dive.lds_depth = 2;
    assert!(overridden.validate().is_err());
}

#[test]
fn partial_selection_matches_modulo_draws_with_duplicate_rejection() {
    let mut random = GnuRand::new(1);
    assert_eq!(
        selection(10, 0.8, &mut random),
        Some(vec![3, 6, 7, 5, 2, 9, 1, 0])
    );
    assert_eq!(selected_count(10, 0.10000005), Some(1));
    assert_eq!(selected_count(10, 0.1000002), Some(2));
    assert_eq!(selected_count(2, 0.8), None);
    assert_eq!(selected_count(0, 0.), None);
    let mut untouched = random.clone();
    assert_eq!(selection(10, 1., &mut random), None);
    assert_eq!(random.next(), untouched.next());
    assert_eq!(selection(10, 0., &mut random), Some(vec![]));
}

#[test]
fn local_search_runs_child_dives_and_retains_only_replayed_incumbents() {
    let root = super::super::super::solve(triangle(), node_options()).unwrap();
    assert!((root.master.lp - 1.5).abs() < 1e-8);
    let tree = ModelTreeOptions {
        node: node_options(),
        ..Default::default()
    };
    let config = LocalSearchOptions {
        depth_limit: 0,
        fixed_column_ratio: 0.5,
        ..Default::default()
    };
    let result = run(
        &root,
        &tree,
        &config,
        None,
        &mut GnuRand::new(1),
        Instant::now() + Duration::from_secs(10),
    )
    .unwrap();
    let (objective, routes, mut values) = result.incumbent.unwrap();
    assert_eq!(objective, 2.);
    assert!(result.nodes >= config.max_iterations);
    assert_eq!(
        super::super::validate_routes(&root.snapshot.state, &routes, &mut values).unwrap(),
        Some(2.)
    );
    assert_eq!(root.snapshot.master.objective(), root.master.lp);
    assert!(root.snapshot.state.fixed_routes.is_empty());
}

#[test]
fn local_search_fixed_static_contributions_leave_the_remaining_domain_open() {
    let variable = MasterVariable {
        id: 100,
        cost: 1.,
        bounds: crate::master::preprocessing::Bounds {
            lower: 0.,
            upper: 10.,
        },
        integer: true,
        entries: vec![],
    };
    let root = super::super::super::solve_with_master_variables(
        triangle(),
        vec![variable],
        node_options(),
    )
    .unwrap();
    let child = fix_entry(&root.snapshot, &IncumbentEntry::Master(0, 3.)).unwrap();
    let solved = child.solve(node_options()).unwrap();
    assert_eq!(solved.master_values, vec![(100, 3.)]);
    assert_eq!(solved.snapshot.state.master_variables[0].bounds.upper, 10.);
    let additional = child
        .with_master_variable_bounds(
            100,
            crate::master::preprocessing::Bounds {
                lower: 4.,
                upper: 10.,
            },
        )
        .unwrap()
        .solve(node_options())
        .unwrap();
    assert_eq!(additional.master_values, vec![(100, 4.)]);
    assert_eq!(root.snapshot.state.master_variables[0].bounds.lower, 0.);
    let mut signed = root.snapshot.fork().unwrap();
    signed.state.master_variables[0].bounds = crate::master::preprocessing::Bounds {
        lower: -10.,
        upper: 10.,
    };
    assert!(fix_entry(&signed, &IncumbentEntry::Master(0, 3.)).is_err());
}

#[test]
fn enabled_local_search_preserves_static_incumbent_contributions() {
    let mut problem = triangle();
    let row = problem.rows.len();
    problem.rows.push(crate::master::formulation::Row {
        sense: RowSense::GreaterEqual,
        rhs: 3.,
    });
    let variable = MasterVariable {
        id: 100,
        cost: 1.,
        bounds: crate::master::preprocessing::Bounds {
            lower: 0.,
            upper: 10.,
        },
        integer: true,
        entries: vec![(row, 1.)],
    };
    let root =
        super::super::super::solve_with_master_variables(problem, vec![variable], node_options())
            .unwrap();
    let original_lower = root.snapshot.state.master_variables[0].bounds.lower;
    let tree = ModelTreeOptions {
        node: node_options(),
        ..Default::default()
    };
    let config = LocalSearchOptions {
        depth_limit: 0,
        fixed_column_ratio: 0.5,
        ..Default::default()
    };
    let result = run(
        &root,
        &tree,
        &config,
        None,
        &mut GnuRand::new(1),
        Instant::now() + Duration::from_secs(10),
    )
    .unwrap();
    let (objective, _, values) = result.incumbent.unwrap();
    assert_eq!(objective, 5.);
    assert_eq!(values, vec![(100, 3.)]);
    assert!(result.nodes >= config.max_iterations);
    assert_eq!(
        root.snapshot.state.master_variables[0].bounds.lower,
        original_lower
    );
}

#[test]
fn local_search_zero_budget_skips_dive_and_random_work() {
    let root = super::super::super::solve(triangle(), node_options()).unwrap();
    let tree = ModelTreeOptions {
        node: node_options(),
        ..Default::default()
    };
    let mut random = GnuRand::new(1);
    let mut untouched = random.clone();
    let result = run(
        &root,
        &tree,
        &LocalSearchOptions::default(),
        None,
        &mut random,
        Instant::now(),
    )
    .unwrap();
    assert_eq!(result.nodes, 0);
    assert!(result.incumbent.is_none());
    assert_eq!(random.next(), untouched.next());
}

#[test]
fn zero_local_iterations_still_evaluate_an_isolated_initial_child() {
    let mut cold = node_options();
    cold.cg.time_limit = Duration::ZERO;
    let root = super::super::super::solve(triangle(), cold).unwrap();
    assert!(!root.snapshot.master.solved());
    let columns = root.snapshot.master.costs.len();
    let tree = ModelTreeOptions {
        node: node_options(),
        ..Default::default()
    };
    let mut config = LocalSearchOptions {
        depth_limit: 0,
        max_iterations: 0,
        ..Default::default()
    };
    config.dive.node_limit = 1;
    let mut random = GnuRand::new(1);
    let mut untouched = random.clone();
    let result = run(
        &root,
        &tree,
        &config,
        None,
        &mut random,
        Instant::now() + Duration::from_secs(10),
    )
    .unwrap();
    assert_eq!(result.nodes, 1);
    assert!(result.iterations > 0);
    assert!(result.pricing_calls > 0);
    assert!(!root.snapshot.master.solved());
    assert_eq!(root.snapshot.master.costs.len(), columns);
    assert!(root.snapshot.state.routes.is_empty());
    assert!(root.snapshot.state.fixed_routes.is_empty());
    assert_eq!(random.next(), untouched.next());
}

#[test]
fn each_initial_and_partial_child_consumes_one_node_from_its_own_dive_budget() {
    let mut graph = Graph::<2>::new(3, 0).unwrap();
    graph.add_arc(0, 1, 1.).unwrap();
    graph.add_arc(1, 2, 1.).unwrap();
    let root = super::super::super::solve(
        Problem::set_partitioning(graph, 1.).unwrap(),
        node_options(),
    )
    .unwrap();
    assert_eq!(root.master.lp, 2.);
    let tree = ModelTreeOptions {
        node: node_options(),
        ..Default::default()
    };
    let mut config = LocalSearchOptions {
        depth_limit: 0,
        fixed_column_ratio: 0.,
        max_iterations: 1,
        ..Default::default()
    };
    config.dive.node_limit = 1;
    let result = run(
        &root,
        &tree,
        &config,
        None,
        &mut GnuRand::new(1),
        Instant::now() + Duration::from_secs(10),
    )
    .unwrap();
    assert_eq!(result.nodes, 2);
    assert_eq!(result.incumbent.unwrap().0, 2.);
    assert_eq!(root.snapshot.master.objective(), 2.);
    assert!(root.snapshot.state.fixed_routes.is_empty());
}
