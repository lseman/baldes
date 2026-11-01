use super::*;
use crate::cuts::soft_conflicts::{Mode, PresenceRow};
use std::sync::atomic::{AtomicUsize, Ordering};

fn options() -> Options {
    let mut options = Options::default();
    options.cg.stabilize = false;
    options.cg.artificial_cost = 100.;
    options
}

fn problem() -> Problem<2> {
    let network = |id| {
        let mut graph = Graph::<2>::new(4, 0).unwrap();
        for (u, v, cost) in [(0, 1, 1.), (1, 3, 0.), (0, 2, 2.), (2, 3, 0.)] {
            graph.add_arc(u, v, cost).unwrap();
        }
        Network {
            id,
            graph,
            multiplicity: crate::graph::Window::new(1., 1.).unwrap(),
            arc_entries: vec![vec![], vec![], vec![], vec![]],
            route_entries: vec![],
            activity_entries: vec![],
        }
    };
    Problem {
        rows: vec![],
        networks: vec![network(0), network(1)],
    }
}

fn fixed() -> NodeSnapshot<2> {
    let root = super::super::super::solve(problem(), options()).unwrap();
    let tag = root
        .snapshot
        .state
        .routes
        .iter()
        .position(|r| r.network == 0 && r.route.arcs == [0, 1])
        .unwrap();
    root.snapshot.fix_route(tag, 1.).unwrap()
}

#[test]
fn fixed_solution_cut_input_erases_fixed_identity_instead_of_subtracting_its_quantity() {
    let mut model = problem();
    model.networks[0].multiplicity = crate::graph::Window::new(2., 2.).unwrap();
    let root = super::super::super::solve_with_master_variables(
        model,
        vec![MasterVariable {
            id: 77,
            cost: 0.,
            integer: true,
            entries: vec![],
            bounds: crate::master::preprocessing::Bounds {
                lower: 3.,
                upper: 3.,
            },
        }],
        options(),
    )
    .unwrap();
    let tag = root
        .snapshot
        .state
        .routes
        .iter()
        .position(|r| r.network == 0 && r.route.arcs == [0, 1])
        .unwrap();
    let child = root.snapshot.fix_route(tag, 1.).unwrap();
    let captured = input(&child);
    assert_eq!(captured.fixed_routes.len(), 1);
    assert_eq!(captured.fixed_routes[0].quantity, 1.);
    assert!(captured
        .residual_routes
        .iter()
        .all(|r| r.route.network != 0));
    assert_eq!(captured.residual_routes.len(), 1);
    assert_eq!(captured.residual_routes[0].route.network, 1);
    assert_eq!(captured.residual_routes[0].quantity, 1.);
    let child = child.fix_route(tag, 1.).unwrap();
    let captured = input(&child);
    assert_eq!(captured.fixed_routes.len(), 1);
    assert_eq!(captured.fixed_routes[0].quantity, 2.);
    assert!(root.snapshot.state.fixed_routes.is_empty());
}

fn conflict() -> FixedSolutionCut {
    FixedSolutionCut {
        priority: 1.,
        row: Row {
            sense: RowSense::LessEqual,
            rhs: 1.,
        },
        coefficients: [0, 1]
            .into_iter()
            .map(|network| PresenceCoefficients {
                network,
                row: PresenceRow::new(2, vec![(0, 1)], Mode::AnyPair).unwrap(),
                arc_events: vec![vec![0], vec![1], vec![], vec![]],
            })
            .collect(),
        master_coefficients: vec![],
    }
}

#[test]
fn fixed_solution_cut_rows_shift_fixed_rhs_and_price_future_columns_on_an_isolated_child() {
    let node = fixed();
    let before = node.state.shared_rows.len();
    let mut tree = ModelTreeOptions::default();
    tree.fixed_solution_separators
        .push(FixedSolutionSeparator::new("conflicts", |input| {
            assert_eq!(input.fixed_routes.len(), 1);
            assert_eq!(input.fixed_routes[0].route.network, 0);
            assert!(input.residual_routes.iter().all(|r| r.route.network == 1));
            Ok(vec![conflict()])
        }));
    let child = prepare(&node, &tree, false).unwrap().unwrap();
    assert_eq!(child.state.shared_rows.len(), before + 1);
    assert_eq!(*child.state.rhs.last().unwrap(), 0.);
    let solved = child.solve(options()).unwrap();
    assert_eq!(solved.master.lp, 3.);
    assert!(solved
        .routes
        .iter()
        .zip(&solved.values)
        .any(|(r, q)| r.network == 1 && *q == 1. && r.route.arcs == [2, 3]));
    assert_eq!(node.state.shared_rows.len(), before);
    assert_eq!(node.solve(options()).unwrap().master.lp, 2.);
}

#[test]
fn fixed_solution_cut_gate_includes_static_only_fixing_and_default_remains_empty() {
    let root = super::super::super::solve(problem(), options()).unwrap();
    let calls = Arc::new(AtomicUsize::new(0));
    let called = calls.clone();
    let mut tree = ModelTreeOptions::default();
    tree.fixed_solution_separators
        .push(FixedSolutionSeparator::new("capture", move |input| {
            called.fetch_add(1, Ordering::Relaxed);
            assert!(input.fixed_routes.is_empty());
            assert_eq!(input.residual_routes.len(), 2);
            Ok(vec![])
        }));
    prepare(&root.snapshot, &tree, false).unwrap();
    assert_eq!(calls.load(Ordering::Relaxed), 0);
    prepare(&root.snapshot, &tree, true).unwrap();
    assert_eq!(calls.load(Ordering::Relaxed), 1);
    assert!(ModelTreeOptions::default()
        .fixed_solution_separators
        .is_empty());
}

#[test]
fn fixed_solution_cut_invalid_descriptors_are_rejected_without_mutating_parent() {
    let node = fixed();
    let before = node.state.shared_rows.len();
    for bad_priority in [false, true] {
        let mut tree = ModelTreeOptions::default();
        tree.fixed_solution_separators
            .push(FixedSolutionSeparator::new("invalid", move |_| {
                let mut cut = conflict();
                if bad_priority {
                    cut.priority = f64::NAN;
                } else {
                    cut.coefficients[0].arc_events.clear();
                }
                Ok(vec![cut])
            }));
        assert!(prepare(&node, &tree, false).is_err());
        assert_eq!(node.state.shared_rows.len(), before);
    }
}

#[test]
fn fixed_solution_cut_priorities_order_rows_and_static_coordinates_validate() {
    let node = fixed();
    let before = node.state.shared_rows.len();
    let mut tree = ModelTreeOptions::default();
    tree.fixed_solution_separators
        .push(FixedSolutionSeparator::new("order", |_| {
            let mut low = conflict();
            low.priority = 0.;
            low.row.rhs = 2.;
            let mut high = conflict();
            high.priority = 10.;
            high.row.rhs = 1.;
            Ok(vec![low, high])
        }));
    let child = prepare(&node, &tree, false).unwrap().unwrap();
    assert_eq!(child.state.shared_rows[before].rhs, 1.);
    assert_eq!(child.state.shared_rows[before + 1].rhs, 2.);
    for (priority, difference) in [(1., 5e-7), (1e8, 0.005)] {
        tree.fixed_solution_separators = vec![FixedSolutionSeparator::new("near-tie", move |_| {
            let mut first = conflict();
            first.priority = priority;
            first.row.rhs = 2.;
            let mut later = conflict();
            later.priority = priority + difference;
            later.row.rhs = 1.;
            Ok(vec![first, later])
        })];
        let child = prepare(&node, &tree, false).unwrap().unwrap();
        assert_eq!(child.state.shared_rows[before].rhs, 2.);
        assert_eq!(child.state.shared_rows[before + 1].rhs, 1.);
    }
    tree.fixed_solution_separators = vec![FixedSolutionSeparator::new("unknown-static", |_| {
        let mut cut = conflict();
        cut.master_coefficients = vec![(999, 1.)];
        Ok(vec![cut])
    })];
    assert!(prepare(&node, &tree, false).is_err());
    assert_eq!(node.state.shared_rows.len(), before);
}

#[test]
fn fixed_solution_cut_hook_runs_before_ordinary_and_local_search_child_cg() {
    let triangle: Problem<2> = format::Model::from_json(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/examples/networks/triangle.json"
    )))
    .unwrap()
    .problem()
    .unwrap();
    for local in [false, true] {
        let root = super::super::super::solve(triangle.clone(), options()).unwrap();
        let calls = Arc::new(AtomicUsize::new(0));
        let called = calls.clone();
        let mut tree = ModelTreeOptions::default();
        tree.node = options();
        tree.fixed_solution_separators
            .push(FixedSolutionSeparator::new("pre-cg", move |input| {
                called.fetch_add(1, Ordering::Relaxed);
                assert!(!input.fixed_routes.is_empty());
                // This is the inherited fractional LP map, before the child is solved.
                assert!(input.residual_routes.iter().all(|r| r.quantity == 0.5));
                for fixed in &input.fixed_routes {
                    assert!(!input
                        .residual_routes
                        .iter()
                        .any(|r| r.route.network == fixed.route.network
                            && r.route.route.identity() == fixed.route.route.identity()));
                }
                Ok(vec![])
            }));
        let deadline = Instant::now() + Duration::from_secs(10);
        let result = if local {
            super::super::local_search::run(
                &root,
                &tree,
                &LocalSearchOptions {
                    depth_limit: 0,
                    fixed_column_ratio: 0.5,
                    max_iterations: 1,
                    ..Default::default()
                },
                None,
                &mut crate::rng::GnuRand::new(1),
                deadline,
            )
            .unwrap()
        } else {
            super::super::diving::run(
                &root,
                &tree,
                &DivingOptions::default(),
                f64::INFINITY,
                deadline,
            )
            .unwrap()
        };
        assert_eq!(result.incumbent.as_ref().map(|w| w.0), Some(2.));
        assert!(calls.load(Ordering::Relaxed) >= if local { 2 } else { 1 });
        assert!(root.snapshot.state.fixed_routes.is_empty());
    }
}

#[test]
fn fixed_solution_cut_hook_is_not_called_by_the_greedy_heuristic() {
    let calls = Arc::new(AtomicUsize::new(0));
    let called = calls.clone();
    let mut tree = ModelTreeOptions::default();
    tree.greedy = true;
    tree.fixed_solution_separators
        .push(FixedSolutionSeparator::new("not-greedy", move |_| {
            called.fetch_add(1, Ordering::Relaxed);
            Ok(vec![])
        }));
    let result = crate::master::tree::solve(problem(), tree).unwrap();
    assert_eq!(result.incumbent, 2.);
    assert!(result.greedy_nodes > 0);
    assert_eq!(calls.load(Ordering::Relaxed), 0);
}

#[test]
fn fixed_solution_cut_hook_runs_on_the_initial_local_child_with_zero_local_iterations() {
    let root = fixed().solve(options()).unwrap();
    let calls = Arc::new(AtomicUsize::new(0));
    let called = calls.clone();
    let mut tree = ModelTreeOptions::default();
    tree.node = options();
    tree.fixed_solution_separators
        .push(FixedSolutionSeparator::new("initial-local", move |input| {
            called.fetch_add(1, Ordering::Relaxed);
            assert_eq!(input.fixed_routes.len(), 1);
            assert_eq!(input.residual_routes.len(), 1);
            Ok(vec![])
        }));
    let mut config = LocalSearchOptions {
        depth_limit: 0,
        max_iterations: 0,
        ..Default::default()
    };
    config.dive.node_limit = 1;
    let result = super::super::local_search::run(
        &root,
        &tree,
        &config,
        None,
        &mut crate::rng::GnuRand::new(1),
        Instant::now() + Duration::from_secs(10),
    )
    .unwrap();
    assert_eq!(calls.load(Ordering::Relaxed), 1);
    assert_eq!(result.nodes, 1);
    assert_eq!(result.incumbent.as_ref().map(|w| w.0), Some(2.));
}
