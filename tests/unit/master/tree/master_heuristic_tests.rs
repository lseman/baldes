use super::*;
use std::sync::{Arc, Mutex};

fn options() -> Options {
    let mut options = Options::default();
    options.cg.stabilize = false;
    options.cg.artificial_cost = 100.;
    options
        .pricing_options
        .push(("BALDESmaxNumOfColsPerIteration".into(), 1.));
    options
}

fn fixture() -> Output<2> {
    fixture_with_variable(crate::master::preprocessing::Bounds {
        lower: 0.,
        upper: 8.,
    })
}

fn fixture_with_variable(bounds: crate::master::preprocessing::Bounds) -> Output<2> {
    let mut networks = vec![];
    for id in 0..2 {
        let mut graph = Graph::<2>::new(4, 0).unwrap();
        for (u, v, cost) in [(0, 1, 1.), (1, 3, 0.), (0, 2, 2.), (2, 3, 0.)] {
            graph.add_arc(u, v, cost).unwrap();
        }
        networks.push(Network {
            id,
            graph,
            multiplicity: crate::graph::Window::new(
                if id == 0 { 2. } else { 1. },
                if id == 0 { 2. } else { 1. },
            )
            .unwrap(),
            arc_entries: vec![vec![]; 4],
            route_entries: vec![],
            activity_entries: vec![],
        });
    }
    let mut root = super::super::super::solve_with_master_variables(
        Problem {
            rows: vec![],
            networks,
        },
        vec![MasterVariable {
            id: 77,
            cost: 2.,
            bounds,
            integer: true,
            entries: vec![],
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
    root.snapshot = root.snapshot.fix_route(tag, 1.).unwrap();
    root
}

fn route(root: &Output<2>, network: i32, alternate: bool, quantity: f64) -> FixedRoute {
    let mut route = root
        .snapshot
        .state
        .routes
        .iter()
        .find(|r| r.network == network && r.route.arcs == [0, 1])
        .unwrap()
        .as_ref()
        .clone();
    if alternate {
        route.route.arcs = vec![2, 3];
    }
    route.route.original_cost = -999.;
    route.route.cost = -999.;
    route.route.nodes = vec![];
    FixedRoute { route, quantity }
}

#[test]
fn master_heuristic_replays_residual_statics_and_original_costs_on_an_isolated_child() {
    let root = fixture();
    let node = root
        .snapshot
        .with_master_variable_bounds(
            77,
            crate::master::preprocessing::Bounds {
                lower: 3.,
                upper: 8.,
            },
        )
        .unwrap();
    let candidate = ResidualMasterSolution {
        routes: vec![route(&root, 0, false, 1.), route(&root, 1, false, 1.)],
        master_values: vec![(77, 2.)],
    };
    let mut tree = ModelTreeOptions::default();
    tree.master_heuristic = Some(MasterHeuristic::new("residual", move |input| {
        assert_eq!(input.routes.len(), 2);
        assert!(input
            .routes
            .iter()
            .any(|r| r.route.network == 0 && r.quantity == 2.));
        Ok(Some(candidate.clone()))
    }));
    let prepared = master_heuristic::prepare(&node, &tree, &HashMap::from([(0, 3.)])).unwrap();
    let child = prepared.node.unwrap();
    let witness = child.state.pricing_witness.as_ref().unwrap();
    assert_eq!(witness.0, 13.);
    assert_eq!(witness.2, vec![(77, 5.)]);
    assert!(witness.1.iter().all(|r| r.route.route.original_cost == 1.));
    assert!(child.state.master_variables[0].integer);
    assert!(node.state.pricing_witness.is_none());
    assert_eq!(node.master.values(), root.snapshot.master.values());
    assert_eq!(prepared.routes.unwrap().len(), 2);
}

#[test]
fn master_heuristic_fractional_routes_and_statics_replace_cut_input_without_becoming_incumbents() {
    let root = fixture();
    let candidate = ResidualMasterSolution {
        routes: vec![
            route(&root, 0, false, 1.),
            route(&root, 1, false, 0.25),
            route(&root, 1, false, 0.25),
            route(&root, 1, true, 0.5),
        ],
        master_values: vec![(77, 0.5)],
    };
    let mut tree = ModelTreeOptions::default();
    tree.master_heuristic = Some(MasterHeuristic::new("fractional", move |_| {
        Ok(Some(candidate.clone()))
    }));
    tree.fixed_solution_separators
        .push(FixedSolutionSeparator::new("observe", |input| {
            assert_eq!(input.fixed_routes.len(), 1);
            assert_eq!(input.residual_routes.len(), 2);
            assert!(input
                .residual_routes
                .iter()
                .all(|r| r.route.network == 1 && r.quantity == 0.5));
            assert_eq!(input.residual_routes[0].route.route.original_cost, 1.);
            assert_eq!(input.residual_routes[1].route.route.original_cost, 2.);
            Ok(vec![])
        }));
    let prepared = master_heuristic::prepare(&root.snapshot, &tree, &HashMap::new()).unwrap();
    let child = prepared.node.unwrap();
    assert!(child.state.pricing_witness.is_none());
    assert!(child.state.master_variables[0].integer);
    fixed_solution_cuts::prepare_with_routes(&child, &tree, false, prepared.routes.as_deref())
        .unwrap();
    assert!(root.snapshot.state.pricing_witness.is_none());
}

#[test]
fn master_heuristic_invalid_or_declined_outputs_leave_an_empty_cut_map_and_parent_unchanged() {
    let root = fixture();
    let valid = ResidualMasterSolution {
        routes: vec![route(&root, 0, false, 1.), route(&root, 1, false, 1.)],
        master_values: vec![],
    };
    let mut bad_quantity = valid.clone();
    bad_quantity.routes[0].quantity = -1.;
    let mut bad_static = valid.clone();
    bad_static.master_values = vec![(77, 9.)];
    let mut unknown_static = valid.clone();
    unknown_static.master_values = vec![(999, 0.)];
    let mut missing_route = valid.clone();
    missing_route.routes.pop();
    let mut duplicate_static = valid;
    duplicate_static.master_values = vec![(77, 0.), (77, 1.)];
    for candidate in [
        None,
        Some(bad_quantity),
        Some(bad_static),
        Some(unknown_static),
        Some(missing_route),
        Some(duplicate_static),
    ] {
        let mut tree = ModelTreeOptions::default();
        tree.master_heuristic = Some(MasterHeuristic::new("reject", move |_| {
            Ok(candidate.clone())
        }));
        let prepared = master_heuristic::prepare(&root.snapshot, &tree, &HashMap::new()).unwrap();
        assert!(prepared.node.is_none());
        assert!(prepared.routes.unwrap().is_empty());
        assert!(root.snapshot.state.pricing_witness.is_none());
    }
    let prepared = master_heuristic::prepare(
        &root.snapshot,
        &ModelTreeOptions::default(),
        &HashMap::new(),
    )
    .unwrap();
    assert!(prepared.node.is_none() && prepared.routes.is_none());
}

#[test]
fn master_heuristic_runs_before_cuts_and_zero_iteration_cg_in_ordinary_and_initial_local_children()
{
    let root = fixture();
    let events = Arc::new(Mutex::new(vec![]));
    let mut tree = ModelTreeOptions::default();
    tree.node = options();
    let callback_events = events.clone();
    let candidate = ResidualMasterSolution {
        routes: vec![route(&root, 0, false, 1.), route(&root, 1, false, 1.)],
        master_values: vec![],
    };
    tree.master_heuristic = Some(MasterHeuristic::new("early", move |_| {
        callback_events.lock().unwrap().push("master");
        Ok(Some(candidate.clone()))
    }));
    let cut_events = events.clone();
    tree.fixed_solution_separators
        .push(FixedSolutionSeparator::new("next", move |input| {
            cut_events.lock().unwrap().push("cuts");
            assert_eq!(input.residual_routes.len(), 1);
            Ok(vec![])
        }));
    let mut config = DivingOptions::default();
    config.node_limit = 1;
    tree.node.cg.time_limit = Duration::ZERO;
    let end = Instant::now() + Duration::from_secs(30);
    let mut early = None;
    let solved = solve_child(
        &root.snapshot,
        &tree,
        &config,
        f64::INFINITY,
        end,
        &HashMap::new(),
        &mut early,
    )
    .unwrap();
    assert_eq!(solved.master.iterations, 0);
    // A time-limited CG snapshot need not retain an incumbent. The evaluator
    // transfers this already-replayed candidate separately before CG starts.
    assert_eq!(early.unwrap().0, 3.);
    assert_eq!(*events.lock().unwrap(), vec!["master", "cuts"]);
    events.lock().unwrap().clear();
    let local = LocalSearchOptions {
        max_iterations: 0,
        dive: config,
        ..Default::default()
    };
    let dive = local_search::run(
        &root,
        &tree,
        &local,
        None,
        &mut crate::rng::GnuRand::new(1),
        end,
    )
    .unwrap();
    assert_eq!(dive.nodes, 1);
    assert_eq!(dive.incumbent.unwrap().0, 3.);
    assert_eq!(*events.lock().unwrap(), vec!["master", "cuts"]);
    assert!(root.snapshot.state.pricing_witness.is_none());
}

#[test]
fn master_heuristic_is_not_scheduled_by_greedy_or_ordinary_tree_evaluation() {
    let root = fixture();
    let mut tree = ModelTreeOptions::default();
    tree.node = options();
    tree.greedy = true;
    tree.master_heuristic = Some(MasterHeuristic::new("dive-only", |_| {
        panic!("not a dive child")
    }));
    let result = root.snapshot.solve_tree(tree).unwrap();
    assert_eq!(result.incumbent, 3.);
    assert!(result.greedy_nodes > 0);
}

#[test]
fn master_heuristic_retains_an_integer_candidate_even_when_child_cg_has_no_time_left() {
    let root = fixture();
    let candidate = ResidualMasterSolution {
        routes: vec![route(&root, 0, false, 1.), route(&root, 1, false, 1.)],
        master_values: vec![],
    };
    let mut tree = ModelTreeOptions::default();
    tree.node = options();
    tree.master_heuristic = Some(MasterHeuristic::new("early", move |_| {
        Ok(Some(candidate.clone()))
    }));
    let mut early = None;
    let evaluated = solve_child(
        &root.snapshot,
        &tree,
        &DivingOptions::default(),
        f64::INFINITY,
        Instant::now() - Duration::from_secs(1),
        &HashMap::new(),
        &mut early,
    );
    if let Err(error) = evaluated {
        assert!(budget_error(&error), "{error}");
    }
    assert_eq!(early.unwrap().0, 3.);
    assert!(root.snapshot.state.pricing_witness.is_none());
}

#[test]
fn master_heuristic_loads_new_fractional_routes_before_cg_with_private_original_metadata() {
    let mut root = fixture();
    let alternate = route(&root, 1, true, 0.5);
    let key = (1, alternate.route.route.identity());
    // Pricing may have retained this nonbasic path. Remove its zero-quantity
    // physical column using the backend's supported mapping, retaining its
    // registry/pool identity as an inactive route. The parent point is intact.
    let known_tag = root.snapshot.state.tags.get(&key).copied();
    let removed: Vec<_> = root
        .snapshot
        .warm
        .columns
        .iter()
        .enumerate()
        .filter_map(|(i, c)| {
            (c.as_ref().map(|c| c.tag) == known_tag && known_tag.is_some()).then_some(i)
        })
        .collect();
    for &column in &removed {
        assert_eq!(root.snapshot.master.values()[column], 0.);
    }
    let (_, mapping) = root.snapshot.master.remove(&[], &removed).unwrap();
    root.snapshot.warm.columns = std::mem::take(&mut root.snapshot.warm.columns)
        .into_iter()
        .enumerate()
        .filter(|(i, _)| mapping[*i].is_some())
        .map(|(_, c)| c)
        .collect();
    assert!(!root
        .snapshot
        .warm
        .columns
        .iter()
        .flatten()
        .any(|c| Some(c.tag) == known_tag));
    let parent_registry = root.snapshot.state.routes.len();
    let parent_columns = root.snapshot.master.costs.len();
    let candidate = ResidualMasterSolution {
        routes: vec![
            route(&root, 0, false, 1.),
            route(&root, 1, false, 0.5),
            alternate,
        ],
        master_values: vec![],
    };
    let mut tree = ModelTreeOptions::default();
    tree.master_heuristic = Some(MasterHeuristic::new("load", move |_| {
        Ok(Some(candidate.clone()))
    }));
    let prepared = master_heuristic::prepare(&root.snapshot, &tree, &HashMap::new()).unwrap();
    let child = prepared.node.unwrap();
    let tag = child.state.tags[&key];
    let column = child
        .warm
        .columns
        .iter()
        .position(|c| c.as_ref().is_some_and(|c| c.tag == tag))
        .unwrap();
    assert_eq!(child.master.costs.len(), parent_columns + 1);
    assert_eq!(child.master.costs[column], 2.);
    assert_eq!(child.state.routes[tag as usize].route.original_cost, 2.);
    assert_eq!(child.master.upper[column], 1.);
    assert!(child
        .warm
        .generated_pool
        .iter()
        .any(|c| c.tag == tag && c.cost == 2.));
    assert!(child.state.pricing_witness.is_none());
    assert_eq!(root.snapshot.state.tags.get(&key).copied(), known_tag);
    assert_eq!(root.snapshot.state.routes.len(), parent_registry);
    assert_eq!(root.snapshot.master.costs.len(), parent_columns);
}

#[test]
fn master_heuristic_observes_signed_static_offsets_and_fixed_routes_to_complete_residual_solution()
{
    let root = fixture_with_variable(crate::master::preprocessing::Bounds {
        lower: -8.,
        upper: 8.,
    });
    let mut residuals = HashMap::new();
    let node = fix_variable(&root.snapshot, Variable::Master(0), -3., &mut residuals).unwrap();
    let old_original = root
        .master_values
        .iter()
        .find(|(id, _)| *id == 77)
        .unwrap()
        .1;
    let routes = vec![route(&root, 0, false, 1.), route(&root, 1, false, 1.)];
    let mut tree = ModelTreeOptions::default();
    tree.master_heuristic = Some(MasterHeuristic::new("signed-context", move |input| {
        assert_eq!(input.fixed_routes.len(), 1);
        assert_eq!(input.fixed_routes[0].route.network, 0);
        assert_eq!(input.fixed_routes[0].quantity, 1.);
        let coordinate = &input.master_variables[0];
        assert_eq!(coordinate.id, 77);
        assert_eq!(coordinate.cost, 2.);
        assert!(coordinate.integer);
        assert_eq!(coordinate.offset, -3.);
        assert_eq!(
            (coordinate.bounds.lower, coordinate.bounds.upper),
            (-5., 11.)
        );
        assert_eq!(coordinate.value + coordinate.offset, old_original);
        let residual = -2. - coordinate.offset;
        assert!(residual >= coordinate.bounds.lower && residual <= coordinate.bounds.upper);
        Ok(Some(ResidualMasterSolution {
            routes: routes.clone(),
            master_values: vec![(coordinate.id, residual)],
        }))
    }));
    let prepared = master_heuristic::prepare(&node, &tree, &residuals).unwrap();
    let child = prepared.node.unwrap();
    let witness = child.state.pricing_witness.as_ref().unwrap();
    assert_eq!(witness.2, vec![(77, -2.)]);
    assert_eq!(witness.0, -1.);
    assert_eq!(child.state.master_variable_signs[0], 0);
    assert_eq!(node.state.master_variables[0].bounds.lower, -8.);
    assert_eq!(node.state.master_variables[0].bounds.upper, 8.);
    assert!(node.state.pricing_witness.is_none());
}
