use super::*;
use crate::graph::RouteDomain;
use crate::master::{preprocessing::Bounds, tree::TreeStatus};

fn problem(max_arcs: usize, groups: bool) -> Problem<2> {
    let mut graph = Graph::<2>::new(4, 1).unwrap();
    graph.main_resource = Some(0);
    graph.initial[0] = Some(0.);
    for vertex in &mut graph.vertices {
        vertex.bounds[0] = Window::new(0., 8.).unwrap();
    }
    for (tail, head, cost) in [(0, 1, 0.), (1, 1, -2.), (1, 2, 0.), (2, 3, 0.)] {
        graph.add_arc(tail, head, cost).unwrap();
    }
    if groups {
        graph.add_arc(2, 1, -100.).unwrap();
        graph.add_elementarity_set(&[2]).unwrap();
    }
    graph
        .set_route_domain(RouteDomain::BoundedWalks { max_arcs })
        .unwrap();
    Problem {
        rows: vec![Row {
            sense: RowSense::Equal,
            rhs: 1.,
        }],
        networks: vec![Network {
            id: 7,
            graph,
            multiplicity: Window::new(0., 1.).unwrap(),
            arc_entries: vec![vec![]; if groups { 5 } else { 4 }],
            route_entries: vec![(0, 1.)],
        }],
    }
}
fn options(mode: cg::BoundMode) -> Options {
    let mut options = Options::default();
    options.cg.stabilize = false;
    options.cg.bound_mode = mode;
    options
}

#[test]
fn bounded_walk_labeling_enforces_hops_in_both_directions_without_physical_ng() {
    use crate::labeling::{
        engine::{EngineControls, LabelingEngine},
        Controls,
    };
    for max_arcs in [3, 4, 5, 7] {
        let graph = problem(max_arcs, false).networks.remove(0).graph;
        let expected = -2. * (max_arcs - 3) as f64;
        let reference = crate::labeling::reference_price(&graph, &Controls::default()).unwrap();
        assert_eq!(reference.minimum_reduced_cost, expected);
        assert!(reference.columns.iter().all(|r| r.arcs.len() <= max_arcs));
        for bidirectional in [false, true] {
            let mut engine = LabelingEngine::prepare(
                &graph,
                EngineControls {
                    bidirectional,
                    ..Default::default()
                },
                crate::pricing::phases::Phase {
                    check_ng: true,
                    jump_arcs: true,
                    labels_per_bucket: i64::MAX,
                },
            )
            .unwrap();
            if bidirectional {
                assert!(!engine.grid.backward.buckets.is_empty());
            }
            let answer = engine.solve(&graph, &Controls::default()).unwrap();
            assert!(answer.exact);
            assert_eq!(answer.minimum_reduced_cost, expected);
            assert!(answer.columns.iter().all(|r| r.arcs.len() <= max_arcs));
        }
        assert!(crate::labeling::price(
            &graph,
            &Controls {
                max_route_arcs: Some(max_arcs - 1),
                ..Default::default()
            }
        )
        .is_err());
    }
}

#[test]
fn bounded_walk_arc_elementarity_groups_remain_hard_in_pricing_enumeration_and_witnesses() {
    let mut graph = Graph::<2>::new(5, 0).unwrap();
    for (tail, head, cost) in [(0, 1, 0.), (1, 2, -1.), (2, 4, 0.), (0, 3, 0.), (3, 4, 0.)] {
        graph.add_arc(tail, head, cost).unwrap();
    }
    let group = graph.add_arc_elementarity_set(&[1, 2, 1]).unwrap();
    assert_eq!(group, 0);
    graph
        .set_route_domain(RouteDomain::BoundedWalks { max_arcs: 4 })
        .unwrap();

    let priced =
        crate::labeling::reference_price(&graph, &crate::labeling::Controls::default()).unwrap();
    assert!(priced.exact);
    assert_eq!(priced.minimum_reduced_cost, 0.);
    assert!(priced
        .columns
        .iter()
        .all(|route| !(route.arcs.contains(&1) && route.arcs.contains(&2))));

    let complete = crate::enumeration::CompleteCanonicalPaths::collect(
        &graph,
        crate::enumeration::PathDomainLimits::default(),
    )
    .unwrap();
    assert_eq!(complete.paths().len(), 1);
    assert_eq!(complete.paths()[0].arc_positions(), [3, 4]);

    let root = crate::master::root::RootProblem::set_partitioning(graph, 1.).unwrap();
    assert!(root.elementary_route(&[0, 1, 2]).is_none());
    assert!(root.elementary_route(&[3, 4]).is_some());
}

#[test]
fn bounded_walk_root_master_guards_repeated_walks_but_allows_proven_simple_routes() {
    let mut graph = Graph::<2>::new(3, 0).unwrap();
    graph.add_arc(0, 1, 0.).unwrap();
    graph.add_arc(1, 1, -10.).unwrap();
    graph.add_arc(1, 2, 0.).unwrap();
    graph
        .set_route_domain(RouteDomain::BoundedWalks { max_arcs: 3 })
        .unwrap();

    let problem = crate::master::root::RootProblem::set_partitioning(graph, 1.).unwrap();
    let mut options = crate::master::root::RootOptions::default();
    options.enumerate = false;
    options.cg.stabilize = false;
    let output = crate::master::root::solve(problem.clone(), options.clone()).unwrap();
    assert_eq!(output.master.status, crate::master::cg::Status::Optimal);
    assert!((output.master.lp + 5.).abs() < 1e-6);
    assert!(output.routes.iter().any(|route| route.arcs == [0, 1, 2]));

    let mut ng_options = options.clone();
    ng_options.relaxation_improvement_priority = 1.;
    let error = crate::master::root::solve(problem.clone(), ng_options).unwrap_err();
    assert!(error
        .to_string()
        .contains("dynamic NG strengthening is unsupported"));

    options.cuts.enabled = true;
    let error = crate::master::root::solve(problem, options.clone()).unwrap_err();
    assert!(error.to_string().contains("hard elementarity groups"));

    let mut elementary = Graph::<2>::new(3, 0).unwrap();
    elementary.add_arc(0, 1, 1.).unwrap();
    elementary.add_arc(1, 1, -10.).unwrap();
    elementary.add_arc(1, 2, 0.).unwrap();
    elementary.add_elementarity_set(&[1]).unwrap();
    elementary
        .set_route_domain(RouteDomain::BoundedWalks { max_arcs: 3 })
        .unwrap();
    let mut pricing = crate::pricing::Solver::<2>::new(3, 0).unwrap();
    pricing.graph = elementary.clone();
    let ng = pricing
        .strengthen_ng(
            &[crate::graph::relaxation::FractionalPath {
                arcs: vec![0, 2],
                value: 1.,
            }],
            &crate::graph::relaxation::Controls::default(),
        )
        .unwrap();
    assert_eq!(ng.columns_forbidden, 0);
    let problem = crate::master::root::RootProblem::set_partitioning(elementary, 1.).unwrap();
    options.cuts.enabled = true;
    options.relaxation_improvement_priority = 1.;
    let output = crate::master::root::solve(problem, options.clone()).unwrap();
    assert_eq!(output.master.status, crate::master::cg::Status::Optimal);
    assert!((output.master.lp - 1.).abs() < 1e-6);

    let mut dag = Graph::<2>::new(3, 0).unwrap();
    dag.add_arc(0, 1, 0.).unwrap();
    dag.add_arc(1, 2, 0.).unwrap();
    dag.set_route_domain(RouteDomain::BoundedWalks { max_arcs: 4 })
        .unwrap();
    let mut pricing = crate::pricing::Solver::<2>::new(3, 0).unwrap();
    pricing.graph = dag.clone();
    assert_eq!(
        pricing
            .strengthen_ng(
                &[crate::graph::relaxation::FractionalPath {
                    arcs: vec![0, 1],
                    value: 1.,
                }],
                &crate::graph::relaxation::Controls::default(),
            )
            .unwrap()
            .columns_forbidden,
        0
    );
    let problem = crate::master::root::RootProblem::set_partitioning(dag, 1.).unwrap();
    let output = crate::master::root::solve(problem, options).unwrap();
    assert_eq!(output.master.status, crate::master::cg::Status::Optimal);
}

#[test]
fn bounded_walk_forks_fixing_enumeration_and_integer_witnesses_keep_the_domain() {
    for mode in [cg::BoundMode::Numerical, cg::BoundMode::Certified] {
        let root = solve_with_master_variables(problem(5, false), vec![], options(mode)).unwrap();
        assert_eq!(root.master.status, cg::Status::Optimal);
        assert_eq!(root.master.lp, -4.);
        assert!(root.master.bound <= -4.);
        let tag = root
            .routes
            .iter()
            .position(|r| r.route.arcs == [0, 1, 1, 2, 3])
            .unwrap();
        let fixed = root
            .snapshot
            .fix_route(tag, 1.)
            .unwrap()
            .solve(options(mode))
            .unwrap();
        assert_eq!(fixed.master.lp, -4.);
        let full = root
            .snapshot
            .enumerate(&EnumerationOptions::default())
            .unwrap();
        assert!(full.enumeration_report().complete);
        assert_eq!(full.state.universe.as_ref().unwrap().len(), 3);
        let cutoff = root
            .snapshot
            .enumerate_below(-3., &EnumerationOptions::default())
            .unwrap();
        assert_eq!(cutoff.report().routes, 1);
        let partial = root
            .snapshot
            .enumerate(&EnumerationOptions {
                max_routes: 2,
                ..Default::default()
            })
            .unwrap();
        assert!(!partial.enumeration_report().complete);
        assert!(partial.state.universe.is_none());
        let child = root
            .snapshot
            .with_activity_bounds(
                Bounds {
                    lower: 0.,
                    upper: 1.,
                },
                &[RowCoefficients {
                    network: 7,
                    arcs: vec![0., 1., 0., 0.],
                    per_route: 0.,
                }],
            )
            .unwrap()
            .solve(options(mode))
            .unwrap();
        assert_eq!(child.master.lp, -2.);
        assert_eq!(
            child.snapshot.state.oracles[0].network.graph.route_domain(),
            RouteDomain::BoundedWalks { max_arcs: 5 }
        );
        let tree = root
            .snapshot
            .solve_assignment_tree(AssignmentTreeOptions {
                node: options(mode),
                ..Default::default()
            })
            .unwrap();
        assert_eq!(tree.status, TreeStatus::Optimal);
        assert_eq!(tree.incumbent, -4.);
        #[cfg(feature = "highs")]
        {
            let mip = full
                .solve_enumerated_mip(&crate::master::solvers::mip::MipOptions::default())
                .unwrap()
                .unwrap();
            assert_eq!(mip.mip.solution.unwrap().objective, -4.);
        }
    }
}

#[test]
fn declared_customer_groups_remain_hard_while_stations_can_repeat() {
    let root =
        solve_with_master_variables(problem(5, true), vec![], options(cg::BoundMode::Certified))
            .unwrap();
    assert_eq!(root.master.lp, -4.);
    for (route, &value) in root.routes.iter().zip(&root.values) {
        if value > 0. {
            assert_eq!(route.route.nodes.iter().filter(|&&v| v == 2).count(), 1);
        }
    }
    let full = root
        .snapshot
        .enumerate(&EnumerationOptions::default())
        .unwrap();
    assert!(full.enumeration_report().complete);
    assert_eq!(full.state.universe.as_ref().unwrap().len(), 3);
    let witness = root.snapshot.integer_witness(1e-6).unwrap().unwrap();
    assert_eq!(witness.0, -4.);
}

#[test]
fn coupled_negative_cycle_bounds_cover_the_declared_walk_length() {
    use crate::graph::coupled::CoupledResource;
    let mut problem = problem(5, false);
    let graph = &mut problem.networks[0].graph;
    for arc in &mut graph.arcs {
        arc.cost = 0.;
        arc.reduced_cost = 0.;
    }
    graph
        .set_coupled_resources(&[CoupledResource::CumulativeCost {
            initial: 0.,
            increments: vec![0., 1., 0., 0.],
            weights: vec![0., -1., 0., 0.],
            sink_upper: None,
            source_upper: None,
        }])
        .unwrap();
    let root =
        solve_with_master_variables(problem, vec![], options(cg::BoundMode::Certified)).unwrap();
    assert_eq!(root.master.status, cg::Status::Optimal);
    assert_eq!(root.master.lp, -3.);
    assert!(root.master.certified_bound <= -3.);
    let full = root
        .snapshot
        .enumerate(&EnumerationOptions::default())
        .unwrap();
    assert_eq!(full.state.universe.as_ref().unwrap().len(), 3);
    assert_eq!(
        full.solve(options(cg::BoundMode::Certified))
            .unwrap()
            .master
            .lp,
        -3.
    );
}

#[test]
fn default_simple_contract_and_walk_domain_guards_are_explicit() {
    let mut graph = problem(5, false).networks.remove(0).graph;
    let simple = {
        let mut g = graph.clone();
        g.set_route_domain(RouteDomain::PhysicalSimple).unwrap();
        g
    };
    assert!(!graph.same_topology(&simple));
    let original = crate::master::root::RootProblem::set_partitioning(simple, 1.).unwrap();
    assert!(original.elementary_route(&[0, 1, 2, 3]).is_none());
    assert!(graph
        .set_route_domain(RouteDomain::BoundedWalks { max_arcs: 0 })
        .is_err());
    let root = crate::master::root::RootProblem::set_partitioning(graph.clone(), 1.).unwrap();
    let output = crate::master::root::solve(root, Default::default()).unwrap();
    assert_eq!(output.master.status, crate::master::cg::Status::Optimal);
    assert!(output.routes.iter().all(|route| route.arcs.len() <= 5));
    graph.elementary = true;
    assert!(graph
        .set_route_domain(RouteDomain::BoundedWalks { max_arcs: 5 })
        .is_err());
    let mut direct = Graph::<2>::new(2, 0).unwrap();
    direct.add_arc(0, 1, 3.).unwrap();
    direct
        .set_route_domain(RouteDomain::BoundedWalks { max_arcs: 1 })
        .unwrap();
    let single_network =
        crate::master::root::RootProblem::set_partitioning(direct.clone(), 1.).unwrap();
    assert!(single_network.elementary_route(&[0]).is_some());
    let direct_problem = Problem {
        rows: vec![Row {
            sense: RowSense::Equal,
            rhs: 1.,
        }],
        networks: vec![Network {
            id: 17,
            graph: direct,
            multiplicity: Window::new(0., 1.).unwrap(),
            arc_entries: vec![vec![(0, 1.)]],
            route_entries: vec![],
        }],
    };
    let solved =
        solve_with_master_variables(direct_problem, vec![], options(cg::BoundMode::Certified))
            .unwrap();
    assert_eq!(solved.master.status, cg::Status::Optimal);
    assert_eq!(solved.master.lp, 3.);
    assert!(solved.routes.iter().any(|route| route.route.arcs == [0]));
    let enumerated = solved
        .snapshot
        .enumerate(&EnumerationOptions::default())
        .unwrap();
    assert!(enumerated.enumeration_report().complete);
    assert_eq!(enumerated.state.universe.as_ref().unwrap().len(), 1);
}

#[test]
#[cfg(target_pointer_width = "64")]
fn integer_mapping_rejects_walk_coefficients_beyond_binary64_exact_integer_range() {
    let mut problem = problem(5, false);
    problem.networks[0]
        .graph
        .set_route_domain(RouteDomain::BoundedWalks {
            max_arcs: 3_002_399_751_580_331,
        })
        .unwrap();
    let controls = AssignmentTreeOptions {
        node_limit: 0,
        mapped_variables: vec![MappedBranchVariable {
            name: "unsafe".into(),
            priority: 1.,
            networks: vec![MappedNetworkVariable {
                network: 7,
                arcs: vec![(1, 3.)],
                per_route: 0.,
            }],
            masters: vec![],
        }],
        ..Default::default()
    };
    assert!(solve_assignment_tree(problem, controls).is_err());
}

#[test]
fn a_declared_walk_cap_below_every_terminal_path_has_a_directed_infeasibility_proof() {
    let output =
        solve_with_master_variables(problem(2, false), vec![], options(cg::BoundMode::Certified))
            .unwrap();
    assert_eq!(output.master.status, cg::Status::Infeasible);
    assert!(output
        .master
        .feasibility
        .iter()
        .any(|r| r.certified_infeasible));
}

#[test]
fn synchronized_walk_service_counts_distinct_copies_instead_of_repeated_visits() {
    let service = SynchronizedService {
        id: 8,
        resource: 0,
        vehicles_required: 2,
        reward: 0.,
        start: [0, 0],
        members: vec![SynchronizedServiceMember {
            network: 7,
            vertex: 1,
        }],
    };
    let mut controls = options(cg::BoundMode::Numerical);
    controls.synchronized_services = vec![service.clone()];
    assert!(
        solve_with_master_variables(problem(5, false), vec![], controls.clone())
            .unwrap_err()
            .message
            .contains("vertex elementarity group")
    );
    let mut declared = problem(5, false);
    declared.networks[0]
        .graph
        .add_elementarity_set(&[1])
        .unwrap();
    declared.networks[0].multiplicity = Window::new(0., 2.).unwrap();
    declared.rows[0].rhs = 2.;
    let output = solve_with_master_variables(declared, vec![], controls).unwrap();
    assert_eq!(output.master.status, cg::Status::Optimal);
    assert_eq!(output.master.lp, 0.);
    assert_eq!(output.values.iter().sum::<f64>(), 2.);
    assert!(output
        .routes
        .iter()
        .zip(&output.values)
        .filter(|(_, value)| **value > 0.)
        .all(|(route, _)| route.route.nodes.iter().filter(|&&v| v == 1).count() == 1));
}
