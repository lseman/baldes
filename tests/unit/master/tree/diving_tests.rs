use super::*;
use crate::inputs::{Instance, Rounding};

fn rc101(customers: usize) -> Problem {
    let root = env!("CARGO_MANIFEST_DIR");
    let mut instance = Instance::read(format!("{root}/instances/Solomon/RC101.txt")).unwrap();
    instance.truncate(customers).unwrap();
    let graph = instance.graph(8, Rounding::None).unwrap();
    Problem::set_partitioning(graph, customers as f64).unwrap()
}

#[test]
fn dives_follow_the_native_depth_limit_and_frequency() {
    let root_only = DivingOptions::default();
    assert!(root_only.due(0, 1));
    assert!(!root_only.due(1, 2));
    let periodic = DivingOptions {
        depth_limit: 3,
        frequency: 4,
        ..Default::default()
    };
    assert!(periodic.due(0, 1));
    assert!(!periodic.due(2, 3));
    assert!(periodic.due(2, 5));
    assert!(!periodic.due(4, 5));
    assert!(DivingOptions {
        strong_candidates: 0,
        ..Default::default()
    }
    .validate()
    .is_err());
}

fn root_tree(diving: Option<DivingOptions>) -> ModelTreeOutput {
    let problem = rc101(25);
    let mut configured = ModelTreeOptions::default();
    configured.mapped_variables = problem.arc_flow_variables(1.);
    configured.mapped_strong_branching = None;
    configured.node_limit = 1;
    configured.diving = diving;
    crate::master::tree::solve(problem, configured).unwrap()
}

#[test]
fn root_dive_supplies_a_valid_incumbent_without_changing_the_bound() {
    let plain = root_tree(None);
    assert!(
        plain.incumbent.is_infinite(),
        "the RC101 root is fractional"
    );
    for diving in [
        DivingOptions::default(),
        DivingOptions {
            strong_candidates: 3,
            lds_depth: 2,
            max_discrepancy: 1,
            ..Default::default()
        },
    ] {
        let dived = root_tree(Some(diving));
        assert!(dived.diving_nodes > 0);
        assert!(dived.incumbent.is_finite(), "{dived:?}");
        assert_eq!(dived.bound, plain.bound);
        assert_eq!(dived.nodes, plain.nodes);
        let replayed: f64 = dived
            .routes
            .iter()
            .map(|r| r.quantity * r.route.route.original_cost)
            .sum();
        assert!((replayed - dived.incumbent).abs() < 1e-6);
    }
}

#[test]
fn diving_tree_reaches_the_default_optimum() {
    let solve = |diving: Option<DivingOptions>| {
        let problem = rc101(25);
        let mut configured = ModelTreeOptions::default();
        configured.mapped_variables = problem.arc_flow_variables(1.);
        configured.node_limit = 2000;
        configured.diving = diving;
        crate::master::tree::solve(problem, configured).unwrap()
    };
    let plain = solve(None);
    let dived = solve(Some(DivingOptions {
        depth_limit: usize::MAX,
        frequency: 5,
        strong_candidates: 2,
        lds_depth: 3,
        max_discrepancy: 1,
        stop_after_first_solution: true,
        ..Default::default()
    }));
    assert_eq!(plain.status, crate::master::tree::TreeStatus::Optimal);
    assert_eq!(dived.status, crate::master::tree::TreeStatus::Optimal);
    assert!(dived.diving_nodes > 0);
    assert!(
        (dived.incumbent - plain.incumbent).abs() < 1e-6,
        "{} {}",
        dived.incumbent,
        plain.incumbent
    );
    // Root dive incumbents never undercut the proved optimum.
    let root = root_tree(Some(DivingOptions::default()));
    assert!(root.incumbent >= plain.incumbent - 1e-6);
}

fn static_relaxation(
    lower: f64,
    upper: f64,
    rhs: f64,
    cost: f64,
    sense: RowSense,
    integer: bool,
) -> Output<2> {
    let mut options = Options::default();
    options.cg.stabilize = false;
    super::super::super::solve_with_master_variables(
        Problem {
            rows: vec![crate::master::formulation::Row { sense, rhs }],
            networks: vec![],
        },
        vec![MasterVariable {
            id: 77,
            cost,
            integer,
            entries: vec![(0, 1.)],
            bounds: crate::master::preprocessing::Bounds { lower, upper },
        }],
        options,
    )
    .unwrap()
}

#[test]
fn static_diving_rounds_positive_and_negative_integer_variables_with_residual_sign_clamps() {
    for (lower, upper, rhs, cost, sense, expected) in [
        (0., 5., 1.5, 1., RowSense::GreaterEqual, 2.),
        (-5., 0., -1.5, -1., RowSense::LessEqual, -2.),
    ] {
        let root = static_relaxation(lower, upper, rhs, cost, sense, true);
        let options = ModelTreeOptions::default();
        let mut config = DivingOptions::default();
        config.column_selection_criteria = vec![4];
        let result = run(
            &root,
            &options,
            &config,
            f64::INFINITY,
            Instant::now() + Duration::from_secs(10),
        )
        .unwrap();
        let witness = result.incumbent.unwrap();
        assert_eq!(witness.2, vec![(77, expected)]);
        assert_eq!(witness.0, 2.);
        assert!(result.nodes > 0);
        assert_eq!(root.master_values, vec![(77, rhs)]);
    }
}

#[test]
fn static_diving_skips_continuous_variables_and_keeps_declared_free_sign_after_inference() {
    let continuous = static_relaxation(0., 5., 1.5, 1., RowSense::GreaterEqual, false);
    assert!(route_columns(&continuous, 1e-6, &Residuals::new()).is_empty());
    let root = static_relaxation(-5., 5., 1.5, 1., RowSense::GreaterEqual, true);
    assert_eq!(root.snapshot.state.master_variable_signs, vec![0]);
    let mut residuals = Residuals::new();
    let child = fix_variable(&root.snapshot, Variable::Master(0), 2., &mut residuals).unwrap();
    assert_eq!(residuals[&0], 2.);
    let solved = child.solve(Options::default()).unwrap();
    assert_eq!(solved.master_values, vec![(77, 1.5)]);
    assert_eq!(
        route_columns(&solved, 1e-6, &residuals),
        vec![(Variable::Master(0), -0.5)]
    );
    assert_eq!(root.master_values, vec![(77, 1.5)]);
    assert!(fix_variable(&root.snapshot, Variable::Master(0), 0., &mut residuals).is_err());
}

#[test]
fn static_diving_priority_metadata_selects_registered_ids_and_rejects_unknown_ids() {
    let root = static_relaxation(0., 5., 1.5, 1., RowSense::GreaterEqual, true);
    let mut config = DivingOptions::default();
    config.column_selection_criteria = vec![2];
    config.master_variable_priorities.insert(77, 8.);
    assert_eq!(
        selection::rank(
            &root.snapshot,
            vec![(Variable::Master(0), 1.5)],
            &config,
            None,
            &Residuals::new()
        )
        .unwrap(),
        vec![(Variable::Master(0), 1.)]
    );
    config.master_variable_priorities.insert(999, 1.);
    assert!(run(
        &root,
        &ModelTreeOptions::default(),
        &config,
        f64::INFINITY,
        Instant::now() + Duration::from_secs(1)
    )
    .is_err());
    config.master_variable_priorities.insert(77, f64::NAN);
    assert!(config.validate().is_err());
}

#[test]
fn greedy_column_selection_scores_the_residual_after_integer_route_fixing() {
    let network = |id, cost, lower, row| {
        let mut graph = Graph::<2>::new(2, 0).unwrap();
        graph.add_arc(0, 1, cost).unwrap();
        Network {
            id,
            graph,
            multiplicity: crate::graph::Window::new(lower, 3.).unwrap(),
            arc_entries: vec![vec![(row, 1.)]],
            route_entries: vec![],
            activity_entries: vec![],
        }
    };
    let mut first = network(0, 0., 1., 0);
    first.multiplicity.upper = 1.;
    let problem = Problem {
        rows: vec![
            crate::master::formulation::Row {
                sense: RowSense::GreaterEqual,
                rhs: 1.5,
            },
            crate::master::formulation::Row {
                sense: RowSense::GreaterEqual,
                rhs: 0.5,
            },
        ],
        networks: vec![first, network(1, 1., 0., 0), network(2, 0.75, 0., 1)],
    };
    let mut options = ModelTreeOptions::default();
    options.node.cg.stabilize = false;
    let root = super::super::super::solve(problem, options.node.clone()).unwrap();
    let mut config = DivingOptions::default();
    config.column_selection_criteria = vec![11];
    let selected = children(
        &root,
        &options,
        &config,
        f64::INFINITY,
        None,
        0,
        0,
        &[],
        &Residuals::new(),
        Instant::now() + Duration::from_secs(10),
        &mut Dive::default(),
    )
    .unwrap();
    let Pending::Fixed(child) = &selected[0].0 else {
        panic!("greedy selection must be unevaluated");
    };
    let networks: Vec<_> = child
        .state
        .fixed_routes
        .iter()
        .map(|r| r.route.network)
        .collect();
    // Before integer network0 is fixed, network1 scores1 and network2 scores1.5.
    // Afterwards network1's remaining coverage is0.5, making its score2;
    // network2 must therefore be selected after the integer contribution.
    assert_eq!(networks, vec![0, 2]);
    assert!(root.snapshot.state.fixed_routes.is_empty());
}
