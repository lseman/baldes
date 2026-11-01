use super::*;
use std::sync::atomic::{AtomicUsize, Ordering};

fn fixture() -> Problem<2> {
    let mut graph = Graph::<2>::new(3, 0).unwrap();
    graph.add_arc(0, 1, 2.).unwrap();
    graph.add_arc(1, 2, 3.).unwrap();
    Problem::set_partitioning(graph, 1.).unwrap()
}

#[test]
fn fractional_callback_receives_original_point_and_replays_cached_costs() {
    let calls = Arc::new(AtomicUsize::new(0));
    let called = calls.clone();
    let mut options = ModelTreeOptions::default();
    options
        .primal_heuristics
        .push(PrimalHeuristic::fractional("application", move |input| {
            called.fetch_add(1, Ordering::Relaxed);
            assert!(input.patterns.is_none());
            assert_eq!(input.depth, 0);
            let mut candidate = input.fractional.clone();
            for route in &mut candidate.routes {
                route.route.route.original_cost = -1e100;
                route.route.route.nodes = vec![-999];
            }
            Ok(Some(candidate))
        }));
    let result = crate::master::tree::solve(fixture(), options).unwrap();
    assert_eq!(calls.load(Ordering::Relaxed), 1);
    assert_eq!(result.incumbent, 5.);
    assert_eq!(result.routes[0].route.route.original_cost, 5.);
    assert_eq!(result.routes[0].route.route.nodes, vec![0, 1, 2]);
}

#[test]
fn fractional_quantities_and_unknown_master_ids_are_rejected() {
    let output = crate::master::model::solve(fixture(), Options::default()).unwrap();
    let mut candidate = fractional(&output);
    candidate.routes[0].quantity = -1.;
    assert!(validate(&output.snapshot, candidate).unwrap().is_none());
    let mut candidate = fractional(&output);
    candidate.routes[0].quantity = 0.5;
    assert!(validate(&output.snapshot, candidate).unwrap().is_none());
    let mut candidate = fractional(&output);
    candidate.master_values.push((999, 1.));
    assert!(validate(&output.snapshot, candidate).unwrap().is_none());
}

#[test]
fn enumerated_callbacks_require_a_retained_pattern_domain() {
    let calls = Arc::new(AtomicUsize::new(0));
    let called = calls.clone();
    let mut options = ModelTreeOptions::default();
    options
        .primal_heuristics
        .push(PrimalHeuristic::enumerated("patterns", move |input| {
            called.fetch_add(1, Ordering::Relaxed);
            let patterns = input.patterns.as_ref().unwrap();
            assert!(!patterns.is_empty());
            assert_eq!(input.coverage_cutoff, None);
            Ok(Some(PrimalHeuristicSolution {
                routes: vec![FixedRoute {
                    route: patterns[0].clone(),
                    quantity: 1.,
                }],
                master_values: vec![],
            }))
        }));
    let result = crate::master::tree::solve(fixture(), options.clone()).unwrap();
    assert_eq!(calls.load(Ordering::Relaxed), 0);
    assert_eq!(result.incumbent, 5.);
    // An integer root can close before a requested initial enumeration runs.
    // Install the complete domain explicitly, then resume its tree.
    let root = crate::master::model::solve(fixture(), Options::default()).unwrap();
    let complete = root
        .snapshot
        .enumerate(&EnumerationOptions::default())
        .unwrap();
    assert!(complete.enumeration_report().complete);
    let result = complete.solve_tree(options).unwrap();
    assert_eq!(calls.load(Ordering::Relaxed), 1);
    assert_eq!(result.incumbent, 5.);
}

#[test]
fn duplicate_static_values_and_noninteger_original_variables_are_rejected() {
    let problem = Problem::<2> {
        rows: vec![],
        networks: vec![],
    };
    let variables = vec![MasterVariable {
        id: 4,
        cost: 1.,
        bounds: crate::master::preprocessing::Bounds {
            lower: 0.,
            upper: 10.,
        },
        integer: true,
        entries: vec![],
    }];
    let output = solve_with_master_variables(problem, variables, Options::default()).unwrap();
    for values in [vec![(4, 1.), (4, 1.)], vec![(4, 0.5)]] {
        assert!(validate(
            &output.snapshot,
            PrimalHeuristicSolution {
                routes: vec![],
                master_values: values
            }
        )
        .unwrap()
        .is_none());
    }
}

#[test]
fn enumerated_callback_keeps_finite_gap_coverage_without_claiming_full_domain() {
    let mut output = crate::master::model::solve(fixture(), Options::default()).unwrap();
    assert!(output.snapshot.state.universe.is_none());
    let mut arcs = output.routes[0].route.arcs.clone();
    arcs.sort_unstable();
    output.snapshot.state.oracles[0].enumerated = Arc::new([arcs].into_iter().collect());
    output.snapshot.state.gap_pool_cutoff = Some(6.);
    let mut options = ModelTreeOptions::default();
    let calls = Arc::new(AtomicUsize::new(0));
    let called = calls.clone();
    options
        .primal_heuristics
        .push(PrimalHeuristic::enumerated("cutoff", move |input| {
            called.fetch_add(1, Ordering::Relaxed);
            assert_eq!(input.coverage_cutoff, Some(6.));
            assert_eq!(input.patterns.as_ref().unwrap().len(), 1);
            Ok(None)
        }));
    let mut zero = options.clone();
    zero.node_limit = 0;
    let mut result = crate::master::tree::solve(fixture(), zero).unwrap();
    run(
        &output,
        &options,
        &mut result,
        Instant::now() + Duration::from_secs(5),
    )
    .unwrap();
    assert_eq!(calls.load(Ordering::Relaxed), 1);
    assert_eq!(result.bound, f64::NEG_INFINITY);
    assert_eq!(result.incumbent, f64::INFINITY);
    assert!(!output.snapshot.enumeration_report().complete);
}
