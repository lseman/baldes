use super::*;
use crate::master::preprocessing::Bounds;

fn triangle() -> Problem<2> {
    format::Model::from_json(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/examples/networks/triangle.json"
    )))
    .unwrap()
    .problem()
    .unwrap()
}
fn options() -> Options {
    let mut options = Options::default();
    options.cg.stabilize = false;
    options.cg.max_penalty_updates = 0;
    options
}
fn total() -> MappedBranchVariable {
    MappedBranchVariable {
        name: "total".into(),
        priority: 1.,
        networks: vec![
            MappedNetworkVariable {
                network: 7,
                activities: vec![],
                arcs: vec![],
                per_route: 1.,
            },
            MappedNetworkVariable {
                network: 42,
                activities: vec![],
                arcs: vec![],
                per_route: 1.,
            },
        ],
        masters: vec![],
    }
}

#[test]
fn directed_network_phase_one_closes_cross_network_count_contradiction() {
    let root = solve_with_master_variables(triangle(), vec![], options()).unwrap();
    let original_costs = root.snapshot.master.costs.clone();
    let child = total()
        .apply(
            &root.snapshot,
            Bounds {
                lower: f64::NEG_INFINITY,
                upper: 1.,
            },
        )
        .unwrap();
    let output = child.solve(options()).unwrap();
    assert_eq!(output.master.status, cg::Status::Infeasible);
    assert!(output
        .master
        .feasibility
        .iter()
        .any(|r| r.certified_infeasible && r.lower_bound.is_some_and(|b| b > 0.)));
    assert_eq!(root.snapshot.solve(options()).unwrap().master.lp, 1.5);
    assert_eq!(root.snapshot.state.routes[0].route.original_cost, 1.);
    assert_eq!(root.snapshot.master.costs, original_costs);
}

#[test]
fn zero_objective_feasibility_prices_restore_scalar_coupled_and_static_costs() {
    for coupled in [false, true] {
        let mut problem = triangle();
        for network in &mut problem.networks {
            for arc in &mut network.graph.arcs {
                arc.cost *= 1000.;
            }
            if coupled {
                let increments = network
                    .graph
                    .arcs
                    .iter()
                    .map(|a| {
                        if a.tail == network.graph.source {
                            1.
                        } else {
                            0.
                        }
                    })
                    .collect();
                let weights = network
                    .graph
                    .arcs
                    .iter()
                    .map(|a| {
                        if a.tail == network.graph.source {
                            100.
                        } else {
                            0.
                        }
                    })
                    .collect();
                network
                    .graph
                    .set_coupled_resources(&[
                        crate::graph::coupled::CoupledResource::CumulativeCost {
                            initial: 0.,
                            increments,
                            weights,
                            sink_upper: None,
                            source_upper: None,
                        },
                    ])
                    .unwrap();
            }
        }
        let variables = if coupled {
            vec![MasterVariable {
                id: 55,
                cost: 37.,
                bounds: Bounds {
                    lower: 1.,
                    upper: 1.,
                },
                integer: true,
                entries: vec![],
            }]
        } else {
            vec![]
        };
        let output = solve_with_master_variables(problem, variables, options()).unwrap();
        assert_eq!(output.master.status, cg::Status::Optimal);
        let expected = if coupled { 1687. } else { 1500. };
        assert!(
            (output.master.lp - expected).abs() < 1e-7,
            "{:?}",
            output.master
        );
        assert!(output
            .master
            .feasibility
            .iter()
            .any(|r| r.status == FeasibilityStatus::Feasible));
        let cost = if coupled { 1100. } else { 1000. };
        for (route, &value) in output.routes.iter().zip(&output.values) {
            if value > 0. {
                assert_eq!(route.route.original_cost, cost);
            }
        }
        assert!(output.snapshot.master.costs.contains(&cost));
        if coupled {
            assert_eq!(output.master_values, vec![(55, 1.)]);
        }
    }
}

#[test]
fn mapped_total_route_count_tree_closes_infeasible_lower_child_with_and_without_trials() {
    for strong in [true, false] {
        let mut controls = ModelTreeOptions::default();
        controls.node = options();
        controls.mapped_variables = vec![total()];
        controls.branching = NetworkBranchingPolicy::ResourceFirst;
        if !strong {
            controls.mapped_strong_branching = None;
        }
        let output = crate::master::tree::solve(triangle(), controls).unwrap();
        assert_eq!(output.status, crate::master::tree::TreeStatus::Optimal);
        assert_eq!(output.incumbent, 2.);
    }
}

fn without_route_columns(root: &Output<2>) -> (ClpMaster, cg::WarmStart) {
    let mut master = root.snapshot.master.try_clone().unwrap();
    let mut warm = root.snapshot.warm.clone();
    let removed: Vec<_> = warm
        .columns
        .iter()
        .enumerate()
        .filter_map(|(i, c)| c.as_ref().filter(|c| c.tag >= 0).map(|_| i))
        .collect();
    let (_, map) = master.remove_cuts(&[], &removed).unwrap();
    warm.columns = warm
        .columns
        .into_iter()
        .zip(&map)
        .filter_map(|(c, m)| m.map(|_| c))
        .collect();
    warm.artificials = warm.artificials.iter().filter_map(|&c| map[c]).collect();
    warm.cut_artificials = warm
        .cut_artificials
        .iter()
        .filter_map(|&(r, c)| Some((r, map[c]?)))
        .collect();
    (master, warm)
}

#[test]
fn network_phase_one_recovers_existing_inactive_route_registry_tags() {
    let root = solve_with_master_variables(triangle(), vec![], options()).unwrap();
    let (master, warm) = without_route_columns(&root);
    let count = root.snapshot.state.routes.len();
    let state = Rc::new(RefCell::new(root.snapshot.state.fork_state()));
    let recovered = recover(
        &state,
        &master,
        &warm,
        Duration::from_secs(5),
        100,
        1e-6,
        cg::BoundMode::Certified,
    )
    .unwrap();
    assert_eq!(recovered.report.status, FeasibilityStatus::Feasible);
    assert!(!recovered.columns.is_empty());
    assert!(recovered
        .columns
        .iter()
        .all(|c| c.tag >= 0 && (c.tag as usize) < count && c.cost == 1.));
    assert_eq!(state.borrow().routes.len(), count);
}

#[test]
fn failed_phase_one_original_cost_replay_leaves_live_registry_unchanged() {
    let root = solve_with_master_variables(triangle(), vec![], options()).unwrap();
    let (master, warm) = without_route_columns(&root);
    let mut original = root.snapshot.state.fork_state();
    for oracle in &mut original.oracles {
        for arc in &mut Arc::make_mut(&mut oracle.network.graph).arcs {
            arc.cost = f64::MAX;
        }
    }
    let before_tags = original.tags.clone();
    let before_routes = original.routes.len();
    let before_calls: Vec<_> = original.oracles.iter().map(|o| o.calls).collect();
    let state = Rc::new(RefCell::new(original));
    assert!(recover(
        &state,
        &master,
        &warm,
        Duration::from_secs(5),
        100,
        1e-6,
        cg::BoundMode::Certified
    )
    .is_err());
    assert_eq!(state.borrow().tags, before_tags);
    assert_eq!(state.borrow().routes.len(), before_routes);
    assert_eq!(
        state
            .borrow()
            .oracles
            .iter()
            .map(|o| o.calls)
            .collect::<Vec<_>>(),
        before_calls
    );
}

#[test]
fn zero_network_phase_one_budgets_remain_unknown_and_publish_nothing() {
    let root = solve_with_master_variables(triangle(), vec![], options()).unwrap();
    for (remaining, iterations) in [(Duration::ZERO, 100), (Duration::from_secs(5), 0)] {
        let state = Rc::new(RefCell::new(root.snapshot.state.fork_state()));
        let tags = state.borrow().tags.clone();
        let recovery = recover(
            &state,
            &root.snapshot.master,
            &root.snapshot.warm,
            remaining,
            iterations,
            1e-6,
            cg::BoundMode::Certified,
        )
        .unwrap();
        assert!(!recovery.report.certified_infeasible);
        assert!(recovery.report.lower_bound.is_none());
        assert!(recovery.columns.is_empty());
        assert_eq!(recovery.report.iterations, 0);
        assert_eq!(state.borrow().tags, tags);
        assert!(state.borrow().oracles.iter().all(|o| o.calls == 0));
    }
}

#[test]
fn directed_infeasible_snapshots_resume_and_restrict_without_forging_warm_metadata() {
    let root = solve_with_master_variables(triangle(), vec![], options()).unwrap();
    let child = total()
        .apply(
            &root.snapshot,
            Bounds {
                lower: f64::NEG_INFINITY,
                upper: 1.,
            },
        )
        .unwrap()
        .solve(options())
        .unwrap();
    assert_eq!(child.master.status, cg::Status::Infeasible);
    assert!(child.snapshot.state.phase_one_infeasible);
    assert_eq!(
        child.snapshot.solve(options()).unwrap().master.status,
        cg::Status::Infeasible
    );
    let restricted = child
        .snapshot
        .with_row(
            Row {
                sense: RowSense::LessEqual,
                rhs: 5.,
            },
            &[],
        )
        .unwrap();
    assert_eq!(
        restricted.solve(options()).unwrap().master.status,
        cg::Status::Infeasible
    );
    let mut unrelated = root.snapshot.fork().unwrap();
    unrelated.warm.bound = f64::INFINITY;
    unrelated.warm.numerical_bound = f64::INFINITY;
    assert!(unrelated
        .solve(options())
        .unwrap_err()
        .message
        .contains("warm master metadata"));
}

#[test]
fn master_only_directed_infeasibility_resumes_through_tree_adapter() {
    let output = solve_master(
        vec![Row {
            sense: RowSense::Equal,
            rhs: 1.,
        }],
        vec![],
        options(),
    )
    .unwrap();
    assert_eq!(output.master.status, cg::Status::Infeasible);
    let repeated = output.snapshot.solve(options()).unwrap();
    assert_eq!(repeated.master.status, cg::Status::Infeasible);
    let tree = repeated
        .snapshot
        .solve_tree(ModelTreeOptions::default())
        .unwrap();
    assert_eq!(tree.status, crate::master::tree::TreeStatus::Infeasible);
    assert_eq!(tree.open_nodes, 0);
}

#[test]
fn machine_front_phase_one_preserves_an_explicit_rounded_program_domain() {
    use crate::graph::coupled::{CoupledResource, FrontOperationPolicy, PreparedSlopeFunction};
    let function = PreparedSlopeFunction::constant_point(3., 0.30000000000000004).unwrap();
    // The declared IEEE program translates .1 by .2 to this exact singleton.
    // A source-control outward enclosure straddles the singleton and is a
    // different domain contract; it cannot establish original infeasibility.
    let mut graph = Graph::<2>::new(2, 0).unwrap();
    graph.add_arc(0, 1, 1000.).unwrap();
    graph
        .set_coupled_resources(&[CoupledResource::SlopeIntervalCost {
            initial: Window::new(0.1, 0.1).unwrap(),
            bounds: Window::new(0., 1.).unwrap(),
            increments: vec![0.2],
            functions: vec![function],
            policy: FrontOperationPolicy::FrontAlgebra,
        }])
        .unwrap();
    graph.set_front_program_domain(true).unwrap();
    assert_eq!(
        graph.coupled_kernel().unwrap().certify(&[0]),
        Some((true, [3., 3.]))
    );
    let problem = Problem {
        rows: vec![Row {
            sense: RowSense::Equal,
            rhs: 1.,
        }],
        networks: vec![Network {
            id: 7,
            graph,
            multiplicity: Window::new(0., 1.).unwrap(),
            arc_entries: vec![vec![]],
            route_entries: vec![(0, 1.)],
            activity_entries: vec![],
        }],
    };
    let output = solve_with_master_variables(problem, vec![], options()).unwrap();
    assert_eq!(output.master.status, cg::Status::Optimal);
    assert_eq!(output.master.lp, 1003.);
    assert!(output.snapshot.state.oracles[0]
        .network
        .graph
        .front_program_domain());
    assert!(output
        .master
        .feasibility
        .iter()
        .any(|r| r.status == FeasibilityStatus::Feasible));
    assert!(output
        .master
        .feasibility
        .iter()
        .all(|r| !r.certified_infeasible));
}

#[test]
fn a_relaxation_column_excluded_by_the_original_front_keeps_recovery_unknown() {
    use crate::graph::coupled::{CoupledResource, FrontOperationPolicy, PreparedSlopeFunction};
    let root = solve_with_master_variables(triangle(), vec![], options()).unwrap();
    let mut state = root.snapshot.state.fork_state();
    let oracle = &mut state.oracles[0];
    let graph = Arc::make_mut(&mut oracle.network.graph);
    let functions = graph
        .arcs
        .iter()
        .map(|_| PreparedSlopeFunction::from_prepared_pairs(vec![(0., 1.), (0., 2.)]).unwrap())
        .collect();
    graph
        .set_coupled_resources(&[CoupledResource::SlopeIntervalCost {
            initial: Window::new(0., 0.).unwrap(),
            bounds: Window::new(0., 2.).unwrap(),
            increments: vec![0.; graph.arcs.len()],
            functions,
            policy: FrontOperationPolicy::FrontAlgebra,
        }])
        .unwrap();
    graph.set_front_program_domain(true).unwrap();
    let route = root.routes.iter().find(|r| r.network == 7).unwrap();
    // The proposed path is physically/scalar feasible, but this legitimate
    // original front has an empty intersection at its first charge domain.
    assert_eq!(
        graph
            .coupled_path_cost(
                &route
                    .route
                    .arcs
                    .iter()
                    .map(|id| oracle.index[id])
                    .collect::<Vec<_>>()
            )
            .unwrap(),
        None
    );
    let mut report = recover(
        &Rc::new(RefCell::new(root.snapshot.state.fork_state())),
        &root.snapshot.master,
        &root.snapshot.warm,
        Duration::ZERO,
        0,
        1e-6,
        cg::BoundMode::Certified,
    )
    .unwrap()
    .report;
    report.status = FeasibilityStatus::Feasible;
    assert_eq!(
        original_cost_for_recovery(oracle, &route.route, &mut report).unwrap(),
        None
    );
    assert_eq!(report.status, FeasibilityStatus::Stalled);
    assert!(!report.certified_infeasible);
}

#[test]
fn scalar_dag_empty_pricing_domain_closes_the_certified_zero_selection_bound() {
    let mut graph = Graph::<2>::new(3, 2).unwrap();
    for v in 0..3 {
        graph.set_bounds(v, 1, 0., 1.).unwrap();
    }
    for v in 0..2 {
        let arc = graph.add_arc(v, v + 1, -1.).unwrap();
        graph.arc_mut(arc).unwrap().consumption[0] = 1.;
        graph.arc_mut(arc).unwrap().consumption[1] = 1.;
    }
    let problem = Problem {
        rows: vec![],
        networks: vec![Network {
            id: 7,
            graph,
            multiplicity: Window {
                lower: 0.,
                upper: 1.,
            },
            arc_entries: vec![vec![]; 2],
            route_entries: vec![],
            activity_entries: vec![],
        }],
    };
    let mut options = Options::default();
    options.cg.stabilize = false;
    options.cg.bound_mode = cg::BoundMode::Certified;
    options.preprocessing.enabled = false;
    let output = solve(problem, options).unwrap();
    assert_eq!(output.master.status, cg::Status::Optimal);
    assert_eq!(output.master.lp, 0.);
    assert_eq!(output.master.certified_bound, 0.);
}
