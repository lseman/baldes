use baldes::{
    graph::{Graph, Window},
    master::{
        cg::{BoundMode, Status},
        model::{self, Network, Options, Output, Problem, RowCoefficients},
        routing::Row,
        RowSense,
    },
};

#[test]
fn root_and_one_network_adapters_share_raw_pricing_results() {
    use baldes::master::routing::{self, RootOptions, RoutingContext};
    for mode in [BoundMode::Numerical, BoundMode::Certified] {
        let mut graph = Graph::<2>::new(4, 1).unwrap();
        for v in 0..4 {
            graph.set_bounds(v, 0, 0., 10.).unwrap();
        }
        for (tail, head, cost) in [
            (0, 1, 1.000000004),
            (1, 3, 0.),
            (0, 2, 2.),
            (2, 3, 0.),
            (1, 2, 0.5),
        ] {
            let id = graph.add_arc(tail, head, cost).unwrap();
            graph.arc_mut(id).unwrap().consumption[0] = 1.;
        }
        let routing = RoutingContext::set_partitioning(graph.clone(), 2.).unwrap();
        let rows = routing.model.rows[1..].to_vec();
        let arc_entries = routing.model.networks[0]
            .arc_entries
            .iter()
            .map(|entries| {
                entries
                    .iter()
                    .filter_map(|&(r, a)| (r > 0).then(|| (r - 1, a)))
                    .collect()
            })
            .collect();
        let mut options = Options::default();
        options.cg.stabilize = false;
        options.cg.bound_mode = mode;
        options.preprocessing.enabled = false;
        let mut root_options = RootOptions::default();
        root_options.cg = options.cg.clone();
        root_options.bound_mode = mode;
        root_options.round_reduced_costs = true;
        root_options.enumerate = false;
        root_options.clp_options = Default::default();
        let single = routing::solve(routing, root_options).unwrap();
        let multiple = model::solve(
            Problem {
                rows,
                networks: vec![Network {
                    id: 7,
                    graph,
                    multiplicity: Window::new(0., 2.).unwrap(),
                    arc_entries,
                    route_entries: vec![],
                }],
            },
            options,
        )
        .unwrap();
        assert_eq!(single.master.status, Status::Optimal);
        assert_eq!(multiple.master.status, Status::Optimal);
        assert_eq!(single.master.lp, 1.500000004);
        assert_eq!(single.master.lp, multiple.master.lp);
        assert!(single.oracle_calls > 0);
        // The routing fleet row and network multiplicity have different LP
        // representations. Degenerate optimal dual bases can therefore admit
        // different numbers of oracle calls through the same coordinator.
        assert!(multiple.oracle_calls[0].1 > 0);
        assert!((single.master.bound - multiple.master.bound).abs() < 1e-12);
        let mut root_paths: Vec<_> = single.routes.iter().map(|r| r.arcs.clone()).collect();
        let mut network_paths: Vec<_> = multiple
            .routes
            .iter()
            .map(|r| r.route.arcs.clone())
            .collect();
        root_paths.sort();
        root_paths.dedup();
        network_paths.sort();
        network_paths.dedup();
        assert_eq!(root_paths, network_paths);
    }
}

#[test]
fn one_and_two_network_collections_preserve_quantity_and_shared_row_bounds() {
    for mode in [BoundMode::Numerical, BoundMode::Certified] {
        let mut combined = Graph::<2>::new(4, 0).unwrap();
        for (tail, head, cost) in [(0, 1, 1.), (1, 3, 0.), (0, 2, 2.), (2, 3, 0.)] {
            combined.add_arc(tail, head, cost).unwrap();
        }
        let rows = vec![
            Row {
                sense: RowSense::Equal,
                rhs: 1.
            };
            2
        ];
        let single = Problem {
            rows: rows.clone(),
            networks: vec![Network {
                id: 7,
                graph: combined,
                multiplicity: Window::new(0., 2.).unwrap(),
                arc_entries: vec![vec![(0, 1.)], vec![], vec![(1, 1.)], vec![]],
                route_entries: vec![],
            }],
        };
        let networks = (0..2)
            .map(|row| {
                let mut graph = Graph::<2>::new(3, 0).unwrap();
                graph.add_arc(0, 1, 1. + row as f64).unwrap();
                graph.add_arc(1, 2, 0.).unwrap();
                Network {
                    id: 7 + row as i32,
                    graph,
                    multiplicity: Window::new(
                        if row == 0 { 1. } else { 0. },
                        if row == 0 { 1. } else { 2. },
                    )
                    .unwrap(),
                    arc_entries: vec![vec![(row, 1.)], vec![]],
                    route_entries: vec![],
                }
            })
            .collect();
        let mut options = Options::default();
        options.cg.stabilize = false;
        options.cg.bound_mode = mode;
        let one = model::solve(single, options.clone()).unwrap();
        let two = model::solve(Problem { rows, networks }, options).unwrap();
        assert_eq!(one.master.lp, 3.);
        assert_eq!(one.master.bound, 3.);
        assert_eq!(two.master.lp, one.master.lp);
        assert_eq!(two.master.bound, one.master.bound);
        assert!(two.oracle_calls.iter().all(|(_, calls)| *calls > 0));
    }
}

fn network(id: i32, routes: &[(f64, &[usize])], row_count: usize) -> Network<2> {
    let mut graph = Graph::<2>::new(routes.len() + 2, 1).unwrap();
    let sink = routes.len() + 1;
    let mut entries = vec![];
    for (i, &(cost, customers)) in routes.iter().enumerate() {
        graph.add_arc(0, i + 1, cost).unwrap();
        graph.add_arc(i + 1, sink, 0.).unwrap();
        assert!(customers.iter().all(|&row| row < row_count));
        entries.push(customers.iter().map(|&row| (row, 1.)).collect());
        entries.push(vec![]);
    }
    Network {
        id,
        graph,
        multiplicity: Window::new(0., 3.).unwrap(),
        arc_entries: entries,
        route_entries: vec![],
    }
}

fn triangle() -> Problem<2> {
    Problem {
        rows: (0..3)
            .map(|_| Row {
                sense: RowSense::Equal,
                rhs: 1.,
            })
            .collect(),
        networks: vec![
            network(
                7,
                &[
                    (1., &[0, 1]),
                    (1., &[1, 2]),
                    (1., &[0]),
                    (1., &[1]),
                    (1., &[2]),
                ],
                3,
            ),
            network(42, &[(1., &[0, 2]), (1., &[0]), (1., &[1]), (1., &[2])], 3),
        ],
    }
}

fn options() -> Options {
    let mut options = Options::default();
    options.cg.stabilize = false;
    options.cg.artificial_cost = 100.;
    options.cg.max_iterations = 100;
    options
}

#[path = "networks/scheduling.rs"]
mod scheduling_tests;

fn optimal<const D: usize>(output: &Output<D>, expected: f64) {
    assert_eq!(output.master.status, Status::Optimal, "{:?}", output.master);
    assert!(
        (output.master.lp - expected).abs() < 1e-7,
        "{} != {expected}",
        output.master.lp
    );
    assert!(output.master.bound <= expected + 1e-7);
    assert!(output.master.bound >= expected - 1e-6);
    assert_eq!(output.routes.len(), output.values.len());
}

#[test]
fn signed_secondary_disposable_resource_waits_from_a_nonzero_start() {
    let mut graph = Graph::<2>::new(3, 2).unwrap();
    graph.initial[1] = Some(5.);
    for vertex in 0..3 {
        graph.set_bounds(vertex, 0, 0., 2.).unwrap();
    }
    graph.set_bounds(0, 1, 5., 10.).unwrap();
    graph.set_bounds(1, 1, 3., 3.).unwrap();
    graph.set_bounds(2, 1, 0., 7.).unwrap();
    let first = graph.add_arc(0, 1, -1.).unwrap();
    let second = graph.add_arc(1, 2, 0.).unwrap();
    graph.arc_mut(first).unwrap().consumption[0] = 1.;
    graph.arc_mut(first).unwrap().consumption[1] = -3.;
    graph.arc_mut(second).unwrap().consumption[0] = 1.;
    graph.arc_mut(second).unwrap().consumption[1] = 4.;
    let output = model::solve(
        Problem {
            rows: vec![],
            networks: vec![Network {
                id: 7,
                graph,
                multiplicity: Window::new(0., 1.).unwrap(),
                arc_entries: vec![vec![], vec![]],
                route_entries: vec![],
            }],
        },
        options(),
    )
    .unwrap();
    optimal(&output, -1.);
    assert_eq!(output.oracle_calls, vec![(7, 3)]);
    assert_eq!(output.routes[0].route.arcs, [first, second]);
    assert_eq!(
        output.routes[0].route.resource_profile,
        [vec![0., 5.], vec![1., 3.], vec![2., 7.]]
    );
}

#[test]
fn signed_equality_resource_returns_to_its_fixed_sink_value() {
    for mode in [BoundMode::Numerical, BoundMode::Certified] {
        let problem = model::format::Model::from_json(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/multi_network_native_equality_signed_transitions.json"
        )))
        .unwrap()
        .problem::<2>()
        .unwrap();
        let mut options = options();
        options.cg.bound_mode = mode;
        let output = model::solve(problem, options).unwrap();

        optimal(&output, -1.);
        assert_eq!(output.master.bound, -1.);
        assert_eq!(output.oracle_calls, vec![(7, 3)]);
        assert_eq!(output.routes.len(), 1);
        assert_eq!(output.values, [1.]);
        assert_eq!(output.routes[0].route.arcs, [0, 1]);
        assert_eq!(
            output.routes[0].route.resource_profile,
            [vec![0., 0.], vec![1., 1.], vec![2., 0.]]
        );
    }
}

fn assignment_mass(output: &Output, problem: &Problem<2>, network: i32, row: usize) -> f64 {
    let coefficient = |route: &model::NetworkRoute| {
        if route.network != network {
            return 0.;
        }
        let oracle = problem.networks.iter().find(|n| n.id == network).unwrap();
        let mut value: f64 = oracle
            .route_entries
            .iter()
            .filter(|&&(r, _)| r == row)
            .map(|&(_, v)| v)
            .sum();
        for id in &route.route.arcs {
            let index = oracle.graph.arcs.iter().position(|a| a.id == *id).unwrap();
            value += oracle.arc_entries[index]
                .iter()
                .filter(|&&(r, _)| r == row)
                .map(|&(_, v)| v)
                .sum::<f64>();
        }
        value
    };
    output
        .routes
        .iter()
        .zip(&output.values)
        .map(|(r, &v)| coefficient(r) * v)
        .sum::<f64>()
        + output
            .fixed_routes
            .iter()
            .map(|f| coefficient(&f.route) * f.quantity)
            .sum::<f64>()
}

#[test]
fn shared_root_prices_both_networks_with_overlapping_arc_ids() {
    let problem = triangle();
    let output = model::solve(problem.clone(), options()).unwrap();
    // Three pair routes, each with quantity 1/2. Summing coverage gives
    // 2*sum(pair quantities)+sum(singletons)=3, proving LP >= 3/2.
    optimal(&output, 1.5);
    assert_eq!(
        output
            .oracle_calls
            .iter()
            .map(|&(id, _)| id)
            .collect::<Vec<_>>(),
        [7, 42]
    );
    assert!(output.oracle_calls.iter().all(|&(_, calls)| calls > 0));
    assert_eq!(
        output
            .routes
            .iter()
            .filter(|r| r.route.arcs == [0, 1])
            .count(),
        2
    );
    assert!((assignment_mass(&output, &problem, 7, 0) - 0.5).abs() < 1e-8);
    let candidates = output.snapshot.fractional_assignments(1e-6).unwrap();
    assert!(candidates
        .iter()
        .any(|c| c.shared_row == 0 && c.network == 7 && (c.value - 0.5).abs() < 1e-8));
}

#[test]
fn assignment_children_and_sibling_restore_preserve_all_oracles() {
    let problem = triangle();
    let parent = model::solve(problem.clone(), options()).unwrap();
    let left = parent.snapshot.branch_assignment(0, 7, true).unwrap();
    let right = parent.snapshot.branch_assignment(0, 7, false).unwrap();
    // Every integer cover costs at least ceil(3/2)=2. Both sides have
    // explicit pair+singleton covers of cost two.
    for stabilized in [false, true] {
        let mut node_options = options();
        node_options.cg.stabilize = stabilized;
        let excluded = right.solve(node_options.clone()).unwrap();
        let assigned = left.solve(node_options).unwrap();
        optimal(&excluded, 2.);
        optimal(&assigned, 2.);
        assert!(assignment_mass(&excluded, &problem, 7, 0).abs() < 1e-8);
        assert!((assignment_mass(&assigned, &problem, 7, 0) - 1.).abs() < 1e-8);
        for result in [&excluded, &assigned] {
            assert_eq!(
                result
                    .oracle_calls
                    .iter()
                    .map(|&(id, _)| id)
                    .collect::<Vec<_>>(),
                [7, 42]
            );
            if stabilized {
                assert!(result.oracle_calls.iter().all(|&(_, calls)| calls > 0));
            }
        }
        optimal(&parent.snapshot.solve(options()).unwrap(), 1.5);
    }
}

#[test]
fn additive_cut_and_fixed_route_survive_descendants() {
    let problem = triangle();
    let parent = model::solve(problem.clone(), options()).unwrap();
    // A globally valid integer-cover cut: at least two routes.
    let coefficients: Vec<_> = problem
        .networks
        .iter()
        .map(|n| RowCoefficients {
            network: n.id,
            arcs: vec![0.; n.graph.arcs.len()],
            per_route: 1.,
        })
        .collect();
    let cut = parent
        .snapshot
        .with_row(
            Row {
                sense: RowSense::GreaterEqual,
                rhs: 2.,
            },
            &coefficients,
        )
        .unwrap();
    let with_cut = cut.solve(options()).unwrap();
    optimal(&with_cut, 2.);
    let pair = with_cut
        .routes
        .iter()
        .position(|r| r.network == 42 && r.route.arcs == [0, 1])
        .unwrap();
    let fixed = with_cut.snapshot.fix_route(pair, 1.).unwrap();
    let descendant = fixed
        .branch_assignment(0, 7, false)
        .unwrap()
        .solve(options())
        .unwrap();
    optimal(&descendant, 2.);
    assert_eq!(descendant.fixed_routes.len(), 1);
    for row in 0..3 {
        let total: f64 = [7, 42]
            .into_iter()
            .map(|n| assignment_mass(&descendant, &problem, n, row))
            .sum();
        assert!((total - 1.).abs() < 1e-8);
    }
    assert!(descendant
        .snapshot
        .fractional_assignments(1e-6)
        .unwrap()
        .is_empty());
    optimal(&parent.snapshot.solve(options()).unwrap(), 1.5);
}

#[test]
fn new_row_after_fixing_subtracts_fixed_contributions() {
    let problem = triangle();
    let parent = model::solve(problem.clone(), options()).unwrap();
    let pair = parent
        .routes
        .iter()
        .position(|r| r.network == 42 && r.route.arcs == [0, 1])
        .unwrap();
    let fixed = parent.snapshot.fix_route(pair, 1.).unwrap();
    let coeffs = problem
        .networks
        .iter()
        .map(|n| RowCoefficients {
            network: n.id,
            arcs: vec![0.; n.graph.arcs.len()],
            per_route: 1.,
        })
        .collect::<Vec<_>>();
    let child = fixed
        .with_row(
            Row {
                sense: RowSense::Equal,
                rhs: 2.,
            },
            &coeffs,
        )
        .unwrap()
        .solve(options())
        .unwrap();
    optimal(&child, 2.);
    assert!((child.values.iter().sum::<f64>() - 1.).abs() < 1e-8);
}

#[test]
fn required_fractional_and_disabled_multiplicities() {
    for bound_mode in [BoundMode::Numerical, BoundMode::Certified] {
        let mut problem = Problem {
            rows: vec![Row {
                sense: RowSense::Equal,
                rhs: 1.,
            }],
            networks: vec![
                network(3, &[(1., &[0])], 1),
                network(9, &[(3., &[0])], 1),
                network(11, &[(0., &[0])], 1),
            ],
        };
        problem.networks[0].multiplicity = Window::new(0.5, 0.5).unwrap();
        problem.networks[1].multiplicity = Window::new(0.5, 1.).unwrap();
        problem.networks[2].multiplicity = Window::new(0., 0.).unwrap();
        let mut opts = options();
        opts.cg.bound_mode = bound_mode;
        let output = model::solve(problem.clone(), opts).unwrap();
        optimal(&output, 2.);
        assert!((assignment_mass(&output, &problem, 3, 0) - 0.5).abs() < 1e-8);
        assert!((assignment_mass(&output, &problem, 9, 0) - 0.5).abs() < 1e-8);
        assert_eq!(output.oracle_calls[2], (11, 0));
    }
}

#[test]
fn once_per_route_coefficients_and_invalid_node_requests() {
    let mut problem = triangle();
    problem.rows.push(Row {
        sense: RowSense::LessEqual,
        rhs: 3.,
    });
    for n in &mut problem.networks {
        n.route_entries.push((3, 1.));
    }
    let parent = model::solve(problem.clone(), options()).unwrap();
    optimal(&parent, 1.5);
    assert!(parent.snapshot.branch_assignment(3, 7, true).is_err());
    assert!(parent.snapshot.branch_assignment(0, 99, true).is_err());
    assert!(parent.snapshot.fix_route(usize::MAX, 1.).is_err());
    assert!(parent.snapshot.fix_route(0, 0.5).is_err());
    assert!(parent.snapshot.fix_route(0, 4.).is_err());
    assert!(parent.snapshot.fractional_assignments(0.).is_err());
    assert!(parent
        .snapshot
        .with_row(
            Row {
                sense: RowSense::Equal,
                rhs: 1.
            },
            &[RowCoefficients {
                network: 7,
                arcs: vec![],
                per_route: 1.,
            }]
        )
        .is_err());
    let mut opts = options();
    opts.cg.known = parent.master.warm_start.generated_pool.clone();
    assert!(parent.snapshot.solve(opts).is_err());
    optimal(&parent.snapshot.solve(options()).unwrap(), 1.5);
}

#[test]
fn assignment_tree_matches_exhaustive_integer_covers() {
    use baldes::master::tree::TreeStatus;
    let mut opts = baldes::master::tree::ModelTreeOptions::default();
    opts.node = options();
    let tree = baldes::master::tree::solve(triangle(), opts.clone()).unwrap();
    // Independently enumerate the nine binary route choices. No optimum uses
    // repeated copies because every route covers a unit-demand customer.
    let masks = [3u8, 6, 1, 2, 4, 5, 1, 2, 4];
    let mut expected = f64::INFINITY;
    for selected in 0..(1usize << masks.len()) {
        let mut counts = [0; 3];
        let mut cost = 0.;
        for (i, &mask) in masks.iter().enumerate() {
            if selected & (1 << i) == 0 {
                continue;
            }
            cost += 1.;
            for (customer, count) in counts.iter_mut().enumerate() {
                *count += usize::from(mask & (1 << customer) != 0);
            }
        }
        if counts == [1, 1, 1] {
            expected = expected.min(cost);
        }
    }
    assert_eq!(tree.status, TreeStatus::Optimal, "{tree:?}");
    assert_eq!(tree.incumbent, expected);
    assert_eq!(tree.bound, expected);
    assert_eq!(tree.open_nodes, 0);
    // The explicitly supported direct source/sink column participates in the
    // modern route universe and changes this fixture's tree trajectory.
    assert_eq!(tree.nodes, 6);
    assert!(tree.routes.iter().all(|r| r.quantity.fract() == 0.));

    opts.node_limit = 1;
    let stopped = baldes::master::tree::solve(triangle(), opts.clone()).unwrap();
    assert_eq!(stopped.status, TreeStatus::NodeLimit);
    assert_eq!(stopped.open_nodes, 2);
    assert!((stopped.bound - 1.5).abs() < 1e-8);
    assert!(!stopped.incumbent.is_finite());
    opts.time_limit = std::time::Duration::ZERO;
    assert_eq!(
        baldes::master::tree::solve(triangle(), opts)
            .unwrap()
            .status,
        TreeStatus::TimeLimit
    );
}

#[test]
fn phase_one_closes_infeasible_multiplicity_children_and_original_empty_rows() {
    use baldes::master::tree::TreeStatus;
    let mut problem = triangle();
    // Put every pair in one network: row assignments are integral, although
    // route quantities are fractional. Every route covers two rows, so integer
    // quantities cannot cover exactly three rows. General phase I now proves
    // the combined-row contradictions in both multiplicity children.
    problem.networks = vec![network(
        7,
        &[(1., &[0, 1]), (1., &[1, 2]), (1., &[0, 2])],
        3,
    )];
    let mut opts = baldes::master::tree::ModelTreeOptions::default();
    opts.node = options();
    let result = baldes::master::tree::solve(problem, opts.clone()).unwrap();
    assert_eq!(result.status, TreeStatus::Infeasible);
    assert_eq!(result.open_nodes, 0);
    assert!(!result.incumbent.is_finite());

    let impossible = Problem {
        rows: vec![Row {
            sense: RowSense::Equal,
            rhs: 1.,
        }],
        networks: vec![network(7, &[(1., &[])], 1)],
    };
    let incomplete = baldes::master::tree::solve(impossible, opts).unwrap();
    assert_eq!(incomplete.status, TreeStatus::Infeasible);
    assert_eq!(incomplete.open_nodes, 0);
    assert!(!incomplete.incumbent.is_finite());
}

#[test]
fn child_row_prices_a_route_absent_from_the_parent_registry() {
    let covers = vec![(1., &[0][..]); 200];
    let mut network = network(7, &covers, 1);
    for (index, arc) in network.graph.arcs.iter_mut().enumerate() {
        if index % 2 == 0 {
            arc.cost = 1. + (index / 2) as f64;
            arc.reduced_cost = arc.cost;
        }
    }
    let parent = model::solve(
        Problem {
            rows: vec![Row {
                sense: RowSense::Equal,
                rhs: 1.,
            }],
            networks: vec![network.clone()],
        },
        options(),
    )
    .unwrap();
    optimal(&parent, 1.);
    let missing = (0..200)
        .find(|&i| {
            !parent
                .routes
                .iter()
                .any(|r| r.route.arcs == [2 * i, 2 * i + 1])
        })
        .expect("bounded pricing pool");
    let mut arcs = vec![0.; network.graph.arcs.len()];
    arcs[2 * missing as usize] = 1.;
    // Every route contributes once, and the selected path contributes once
    // more. Unit coverage plus RHS two forces that previously unseen path.
    let child = parent
        .snapshot
        .with_row(
            Row {
                sense: RowSense::Equal,
                rhs: 2.,
            },
            &[RowCoefficients {
                network: 7,
                arcs,
                per_route: 1.,
            }],
        )
        .unwrap()
        .solve(options())
        .unwrap();
    optimal(&child, 1. + missing as f64);
    assert!(child
        .routes
        .iter()
        .zip(&child.values)
        .any(|(r, &v)| r.route.arcs == [2 * missing, 2 * missing + 1] && (v - 1.).abs() < 1e-8));
    optimal(&parent.snapshot.solve(options()).unwrap(), 1.);
}

#[test]
fn resource_aware_bounds_certify_bounded_cycles_and_keep_unrestricted_cycles_unknown() {
    for bounded in [false, true] {
        let mut graph = Graph::<2>::new(4, 1).unwrap();
        // Pricing is finite in the control case; the certificate's unrestricted
        // walk relaxation must still decline its zero-consumption negative cycle.
        graph.elementary = true;
        for vertex in 0..4 {
            graph.set_bounds(vertex, 0, 0., 2.).unwrap();
        }
        for (tail, head, cost) in [(0, 1, 0.), (1, 2, -1.), (2, 1, 0.), (1, 3, 0.)] {
            let arc = graph.add_arc(tail, head, cost).unwrap();
            graph.arc_mut(arc).unwrap().consumption[0] = if bounded && head != 3 { 1. } else { 0. };
        }
        let problem = Problem {
            rows: vec![Row {
                sense: RowSense::Equal,
                rhs: 1.,
            }],
            networks: vec![Network {
                id: 7,
                graph,
                multiplicity: Window::new(0., 1.).unwrap(),
                arc_entries: vec![vec![]; 4],
                route_entries: vec![(0, 1.)],
            }],
        };
        let mut opts = options();
        opts.cg.bound_mode = BoundMode::Certified;
        let output = model::solve(problem, opts.clone()).unwrap();
        let resumed = output.snapshot.solve(opts).unwrap();
        for result in [&output, &resumed] {
            if bounded {
                // The only elementary route has cost zero. Resource-derived
                // hop limits may retain extra infeasible walks, so a lower
                // certificate is valid; it need not close the root gap.
                assert!(
                    result.master.certified_bound.is_finite(),
                    "{:?}",
                    result.master
                );
                assert!(result.master.certified_bound <= 0.);
                assert_eq!(result.master.bound, result.master.certified_bound);
                assert!((result.master.lp - 0.).abs() < 1e-8);
            } else {
                assert_eq!(result.master.status, Status::Stalled);
                assert_eq!(result.master.certified_bound, f64::NEG_INFINITY);
                assert_eq!(result.master.bound, f64::NEG_INFINITY);
            }
        }
    }
}

#[test]
fn stabilized_children_keep_column_cleanup_metadata_consistent() {
    let mut opts = options();
    opts.cg.stabilize = true;
    opts.cg.column_cleanup_threshold = 3.;
    opts.cg.column_cleanup_ratio = 0.66;
    let parent = model::solve(triangle(), opts.clone()).unwrap();
    optimal(&parent, 1.5);
    let branch = parent.snapshot.branch_assignment(0, 7, false).unwrap();
    optimal(&branch.solve(opts.clone()).unwrap(), 2.);
    optimal(&parent.snapshot.solve(opts).unwrap(), 1.5);
}

#[test]
fn nondefault_resource_dimension_survives_node_forks() {
    let mut graph = Graph::<3>::new(3, 3).unwrap();
    graph.add_arc(0, 1, 1.).unwrap();
    graph.add_arc(1, 2, 1.).unwrap();
    let problem = Problem {
        rows: vec![Row {
            sense: RowSense::Equal,
            rhs: 1.,
        }],
        networks: vec![Network {
            id: 19,
            graph,
            multiplicity: Window::new(1., 2.).unwrap(),
            arc_entries: vec![vec![(0, 1.)], vec![]],
            route_entries: vec![],
        }],
    };
    let parent = model::solve(problem, options()).unwrap();
    assert_eq!(parent.master.status, Status::Optimal);
    assert!((parent.master.lp - 2.).abs() < 1e-8);
    let child = parent
        .snapshot
        .branch_assignment(0, 19, true)
        .unwrap()
        .solve(options())
        .unwrap();
    assert_eq!(child.master.status, Status::Optimal);
    assert!((child.master.lp - 2.).abs() < 1e-8);
}

#[test]
fn rounded_resource_profiles_do_not_authorize_fixed_routes_or_incumbents() {
    use baldes::master::tree::TreeStatus;
    let mut graph = Graph::<2>::new(3, 1).unwrap();
    for vertex in 0..3 {
        graph.set_bounds(vertex, 0, 1., 1.).unwrap();
    }
    let arc = graph.add_arc(0, 1, 1.).unwrap();
    graph.arc_mut(arc).unwrap().consumption[0] = f64::EPSILON * 0.5;
    graph.add_arc(1, 2, 0.).unwrap();
    let problem = Problem {
        rows: vec![Row {
            sense: RowSense::Equal,
            rhs: 1.,
        }],
        networks: vec![Network {
            id: 7,
            graph,
            multiplicity: Window::new(0., 1.).unwrap(),
            arc_entries: vec![vec![(0, 1.)], vec![]],
            route_entries: vec![],
        }],
    };
    let root = model::solve(problem.clone(), options()).unwrap();
    assert!(!root.routes.is_empty());
    assert!(root.snapshot.fix_route(0, 1.).is_err());
    let mut opts = baldes::master::tree::ModelTreeOptions::default();
    opts.node = options();
    let result = baldes::master::tree::solve(problem, opts).unwrap();
    assert_eq!(result.status, TreeStatus::Incomplete);
    assert_eq!(result.open_nodes, 1);
    assert!(!result.incumbent.is_finite());
}

fn nonlinear_network(id: i32, first_row: usize) -> Network<2> {
    let mut graph = Graph::<2>::new(4, 1).unwrap();
    for (tail, head, cost) in [(0, 1, 1.), (1, 2, 0.), (2, 3, 1.), (1, 3, 1.), (0, 2, 1.)] {
        graph.add_arc(tail, head, cost).unwrap();
    }
    let arc_entries = graph
        .arcs
        .iter()
        .map(|a| match a.head {
            1 => vec![(first_row, 1.)],
            2 => vec![(first_row + 1, 1.)],
            _ => vec![],
        })
        .collect();
    Network {
        id,
        graph,
        multiplicity: Window::new(0., 3.).unwrap(),
        arc_entries,
        route_entries: vec![],
    }
}

#[test]
fn nonlinear_shared_rows_propagate_fixed_contributions_across_networks_and_siblings() {
    use model::{EnumerationOptions, RankOneCoefficients};
    let problem = Problem {
        rows: (0..4)
            .map(|_| Row {
                sense: RowSense::Equal,
                rhs: 1.,
            })
            .collect(),
        networks: vec![nonlinear_network(7, 0), nonlinear_network(42, 2)],
    };
    let root = model::solve(problem, options()).unwrap();
    optimal(&root, 4.);
    let coefficients = [7, 42].map(|network| RankOneCoefficients {
        network,
        weights: vec![0, 1, 1, 0],
        denominator: 2,
    });
    let pair = root
        .routes
        .iter()
        .position(|r| r.network == 7 && r.route.arcs.len() == 3)
        .unwrap();
    let fixed = root.snapshot.fix_route(pair, 1.).unwrap();
    let child = fixed
        .with_rank_one_row(
            Row {
                sense: RowSense::LessEqual,
                rhs: 1.,
            },
            &coefficients,
        )
        .unwrap();
    let solved = child.solve(options()).unwrap();
    // Network 7's fixed pair consumes the nonlinear row's entire RHS. Network
    // 42 must supply two singletons; summing vertex weights instead would not
    // implement floor(sum/2) and would give different residual coefficients.
    optimal(&solved, 6.);
    assert_eq!(solved.fixed_routes.len(), 1);
    assert!(solved
        .routes
        .iter()
        .zip(&solved.values)
        .filter(|(r, v)| r.network == 42 && **v > 1e-8)
        .all(|(r, _)| r.route.arcs.len() == 2));
    let enumerated = child.enumerate(&EnumerationOptions::default()).unwrap();
    assert!(enumerated.enumeration_report().complete);
    let preprocessed = enumerated.preprocess().unwrap();
    assert!(preprocessed.preprocessing_report().complete_domain);
    assert!(preprocessed.preprocessing_report().fixed_columns > 0);
    let automatic = enumerated.solve(options()).unwrap();
    assert!(automatic.snapshot.preprocessing_report().complete_domain);
    assert!(automatic.snapshot.preprocessing_report().fixed_columns > 0);
    optimal(&automatic, 6.);
    #[cfg(feature = "highs")]
    {
        let mip = preprocessed
            .solve_enumerated_mip(&Default::default())
            .unwrap()
            .unwrap();
        assert!((mip.mip.solution.unwrap().objective - 6.).abs() < 1e-8);
    }
    assert!(
        !root
            .snapshot
            .preprocess()
            .unwrap()
            .preprocessing_report()
            .complete_domain
    );
    // Installing the nonlinear row before fixing gives the same model.
    let reordered = root
        .snapshot
        .with_rank_one_row(
            Row {
                sense: RowSense::LessEqual,
                rhs: 1.,
            },
            &coefficients,
        )
        .unwrap()
        .fix_route(pair, 1.)
        .unwrap();
    optimal(&reordered.solve(options()).unwrap(), 6.);
    optimal(&root.snapshot.solve(options()).unwrap(), 4.);
    optimal(&fixed.solve(options()).unwrap(), 4.);
    let mut invalid = coefficients[0].clone();
    invalid.weights[0] = 1;
    assert!(root
        .snapshot
        .with_rank_one_row(
            Row {
                sense: RowSense::LessEqual,
                rhs: 1.
            },
            &[invalid]
        )
        .is_err());
}

#[test]
fn original_activity_fixing_and_multiplicity_links_survive_future_pricing() {
    use baldes::master::preprocessing::Bounds;
    let problem = triangle();
    let root = model::solve(problem.clone(), options()).unwrap();
    let coefficients: Vec<_> = problem
        .networks
        .iter()
        .map(|n| RowCoefficients {
            network: n.id,
            arcs: n
                .arc_entries
                .iter()
                .map(|e| {
                    e.iter()
                        .filter(|&&(r, _)| r == if n.id == 7 { 0 } else { 1 })
                        .map(|&(_, a)| a)
                        .sum()
                })
                .collect(),
            per_route: 0.,
        })
        .collect();
    let fixed = root
        .snapshot
        .with_activity_bounds(
            Bounds {
                lower: 0.,
                upper: 0.,
            },
            &coefficients,
        )
        .unwrap();
    optimal(&fixed.solve(options()).unwrap(), 2.);
    let child = root
        .snapshot
        .with_multiplicity(
            42,
            Bounds {
                lower: 0.,
                upper: 0.,
            },
        )
        .unwrap();
    let solved = child.solve(options()).unwrap();
    optimal(&solved, 2.);
    assert!(solved
        .routes
        .iter()
        .zip(&solved.values)
        .filter(|(r, _)| r.network == 42)
        .all(|(_, v)| v.abs() < 1e-8));
    optimal(&root.snapshot.solve(options()).unwrap(), 1.5);
    assert!(root
        .snapshot
        .with_activity_bounds(
            Bounds {
                lower: 2.,
                upper: 1.
            },
            &coefficients
        )
        .is_err());
}

#[test]
fn fixed_original_variable_links_two_networks_and_nonlinear_row() {
    use baldes::master::preprocessing::Bounds;
    use model::{EnumerationOptions, PreprocessingOptions, RankOneCoefficients};
    let problem = Problem {
        rows: (0..4)
            .map(|_| Row {
                sense: RowSense::Equal,
                rhs: 1.,
            })
            .collect(),
        networks: vec![nonlinear_network(7, 0), nonlinear_network(42, 2)],
    };
    let root = model::solve(problem, options()).unwrap();
    let coefficients = [7, 42].map(|network| RowCoefficients {
        network,
        arcs: vec![0., 1., 0., 0., 0.],
        per_route: 0.,
    });
    let nonlinear = [7, 42].map(|network| RankOneCoefficients {
        network,
        weights: vec![0, 1, 1, 0],
        denominator: 2,
    });
    let fixed = root
        .snapshot
        .with_activity_bounds(
            Bounds {
                lower: 1.,
                upper: 1.,
            },
            &coefficients,
        )
        .unwrap()
        .with_rank_one_row(
            Row {
                sense: RowSense::LessEqual,
                rhs: 1.,
            },
            &nonlinear,
        )
        .unwrap();
    // Independently enumerate pair/split choices in the two networks. Exactly
    // one middle-arc traversal permits (pair,split) or (split,pair), both cost 6.
    let exact = (0..=1)
        .flat_map(|a| (0..=1).map(move |b| (a, b)))
        .filter(|(a, b)| a + b == 1)
        .map(|(a, b)| if a == 1 { 2. } else { 4. } + if b == 1 { 2. } else { 4. })
        .fold(f64::INFINITY, f64::min);
    assert_eq!(exact, 6.);
    optimal(&fixed.solve(options()).unwrap(), exact);
    let one_side = fixed
        .with_multiplicity(
            7,
            Bounds {
                lower: 2.,
                upper: 2.,
            },
        )
        .unwrap()
        .solve(options())
        .unwrap();
    optimal(&one_side, exact);
    assert!(one_side.snapshot.preprocessing_report().forbidden_arcs > 0);
    let mut off = options();
    off.preprocessing.enabled = false;
    optimal(&fixed.solve(off).unwrap(), exact);
    let zero_work = fixed
        .preprocess_with_options(&PreprocessingOptions {
            enabled: true,
            work_limit: 0,
            ..Default::default()
        })
        .unwrap();
    assert!(!zero_work.preprocessing_report().propagation.complete);
    assert_eq!(zero_work.preprocessing_report().forbidden_arcs, 0);
    let complete = fixed
        .enumerate(&EnumerationOptions::default())
        .unwrap()
        .preprocess()
        .unwrap();
    optimal(&complete.solve(options()).unwrap(), exact);
    optimal(&root.snapshot.solve(options()).unwrap(), 4.);
}

#[test]
fn rounded_original_coefficients_do_not_justify_automatic_fixing() {
    let mut graph = Graph::<2>::new(3, 1).unwrap();
    graph.add_arc(0, 1, 1.).unwrap();
    graph.add_arc(1, 2, 0.).unwrap();
    let problem = Problem {
        rows: vec![Row {
            sense: RowSense::LessEqual,
            rhs: 1.,
        }],
        networks: vec![Network {
            id: 7,
            graph,
            multiplicity: Window::new(0., 2.).unwrap(),
            arc_entries: vec![vec![(0, 0.9), (0, 0.1)], vec![]],
            route_entries: vec![],
        }],
    };
    let root = model::solve(problem, options()).unwrap();
    assert_eq!(
        root.snapshot.preprocessing_report().skipped.as_deref(),
        Some("RoundedFlowCoefficients")
    );
    assert_eq!(root.snapshot.preprocessing_report().forbidden_arcs, 0);
    let child = root
        .snapshot
        .enumerate(&model::EnumerationOptions::default())
        .unwrap()
        .preprocess()
        .unwrap();
    assert_eq!(
        child.preprocessing_report().skipped.as_deref(),
        Some("RoundedFlowCoefficients")
    );
    assert_eq!(child.preprocessing_report().fixed_columns, 0);
}

#[test]
fn original_variable_and_multiplicity_fixtures_match_exact_cover_domains() {
    use baldes::master::preprocessing::Bounds;
    let root = model::solve(triangle(), options()).unwrap();
    // The first source arc selects exactly the first pair route in network 7.
    let mut arcs = vec![0.; 10];
    arcs[0] = 1.;
    let coefficients = [RowCoefficients {
        network: 7,
        arcs,
        per_route: 0.,
    }];
    for value in [0., 1.] {
        let child = root
            .snapshot
            .with_activity_bounds(
                Bounds {
                    lower: value,
                    upper: value,
                },
                &coefficients,
            )
            .unwrap();
        optimal(&child.solve(options()).unwrap(), 2.);
    }
    for value in [0., 2.] {
        let child = root
            .snapshot
            .with_multiplicity(
                7,
                Bounds {
                    lower: value,
                    upper: value,
                },
            )
            .unwrap();
        optimal(&child.solve(options()).unwrap(), 2.);
    }
    optimal(&root.snapshot.solve(options()).unwrap(), 1.5);
}

#[path = "networks/master_variables.rs"]
mod master_variable_tests;

#[path = "networks/coupled_objective.rs"]
mod coupled_objective_tests;

#[path = "networks/general_domains.rs"]
mod general_domain_tests;

#[path = "networks/registered_cuts.rs"]
mod registered_cut_tests;

#[path = "networks/resource_rows.rs"]
mod resource_row_tests;

#[path = "networks/generic_tree.rs"]
mod generic_tree_tests;

#[path = "networks/unbounded_multiplicity.rs"]
mod unbounded_multiplicity_tests;

#[path = "networks/presence_rows.rs"]
mod presence_row_tests;

#[path = "networks/master_only.rs"]
mod master_only_tests;

#[path = "networks/imported.rs"]
mod imported_tests;

#[path = "networks/imported_extension.rs"]
mod imported_extension_tests;

#[cfg(feature = "highs")]
#[path = "networks/imported_oracle.rs"]
mod imported_oracle_tests;

#[path = "networks/imported_multiple.rs"]
mod imported_multiple_tests;

#[path = "networks/imported_tree.rs"]
mod imported_tree_tests;

#[path = "networks/bound_domains.rs"]
mod bound_domain_tests;

#[path = "networks/oracle_replies.rs"]
mod oracle_reply_tests;

#[path = "networks/generic_domains.rs"]
mod generic_domain_oracle_tests;

#[path = "networks/imported_nonlinear.rs"]
mod imported_nonlinear_tests;

#[path = "networks/resource_fractional.rs"]
mod fractional_resource_tests;

#[path = "networks/resource_groups.rs"]
mod resource_group_tests;

#[path = "networks/registered_propagation.rs"]
mod registered_propagation_tests;

#[test]
fn native_half_incident_mapping_matches_physical_pricing_entries() {
    // Pinned native function0/rule0, phase-two candidate limit1: CG10,
    // processed3, six trial evaluations, eight actual MetaSolver entries.
    // Its published oracle counter20 also counts bound-only admissions.
    for (phase_two_candidates, iterations, calls) in [(1, 10, 7), (10, 18, 15)] {
        for mode in [BoundMode::Numerical, BoundMode::Certified] {
            let problem = model::format::Model::from_json(include_str!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/examples/networks/triangle.json"
            )))
            .unwrap()
            .problem::<2>()
            .unwrap();
            let mut options = baldes::master::tree::ModelTreeOptions::default();
            options.node.cg.stabilize = false;
            options.node.cg.bound_mode = mode;
            options.branching = baldes::master::tree::NetworkBranchingPolicy::ResourceFirst;
            options.mapped_variables = serde_json::from_str(include_str!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/tests/fixtures/multi_network_native_half_arc_mappings.json"
            )))
            .unwrap();
            options.mapped_strong_branching.as_mut().unwrap().phases[1].candidates =
                phase_two_candidates;
            let result = baldes::master::tree::solve(problem, options).unwrap();
            assert_eq!(result.status, baldes::master::tree::TreeStatus::Optimal);
            assert_eq!(result.incumbent, 2.);
            assert_eq!(result.bound, 2.);
            assert_eq!(result.processed_nodes, 3);
            assert_eq!(result.strong_branching_trials, 6);
            assert_eq!(result.iterations, iterations);
            assert_eq!(result.oracle_calls, vec![(7, calls), (42, 1)]);
        }
    }
}

#[test]
fn native_plain_integral_child_closes_before_extra_pricing_phases() {
    for mode in [BoundMode::Numerical, BoundMode::Certified] {
        let problem = model::format::Model::from_json(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/multi_network_native_tree.json"
        )))
        .unwrap()
        .problem::<2>()
        .unwrap();
        let mut options = baldes::master::tree::ModelTreeOptions::default();
        options.node.cg.stabilize = false;
        options.node.cg.bound_mode = mode;
        let result = baldes::master::tree::solve(problem, options).unwrap();
        assert_eq!(result.status, baldes::master::tree::TreeStatus::Optimal);
        assert_eq!(result.incumbent, 2.);
        assert_eq!(result.bound, 2.);
        assert_eq!(result.processed_nodes, 7);
        assert_eq!(result.nodes, 6);
        // Native prints 17 master rows, including the integral child's single
        // row, but records only 16 CG/pricing rounds. Neither is the count of
        // physical MetaSolver entries, which are tracked separately.
        assert_eq!(result.iterations, 17);
        assert_eq!(result.pricing_calls, 16);
        assert_eq!(result.oracle_calls, vec![(7, 4), (42, 7)]);
    }
}

#[test]
fn native_packing_registration_branches_without_manual_expression_mappings() {
    for mode in [BoundMode::Numerical, BoundMode::Certified] {
        let problem = model::format::Model::from_json(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/multi_network_native_tree.json"
        )))
        .unwrap()
        .problem::<2>()
        .unwrap();
        let mut options = baldes::master::tree::ModelTreeOptions::default();
        options.node.cg.stabilize = false;
        options.node.cg.bound_mode = mode;
        options.packing = Some(
            serde_json::from_str(include_str!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/tests/fixtures/multi_network_native_packing.json"
            )))
            .unwrap(),
        );
        let result = baldes::master::tree::solve(problem, options).unwrap();
        assert_eq!(result.status, baldes::master::tree::TreeStatus::Optimal);
        assert_eq!(result.incumbent, 2.);
        assert_eq!(result.bound, 2.);
        assert_eq!(result.processed_nodes, 3);
        assert_eq!(result.strong_branching_trials, 6);
        assert!(result
            .mapped_branch_evaluations
            .iter()
            .all(|e| e.name.starts_with("ESA_")));
    }
}

#[test]
fn native_packing_separator_is_bypassed_without_strong_branching() {
    for mode in [BoundMode::Numerical, BoundMode::Certified] {
        let problem = model::format::Model::from_json(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/multi_network_native_tree.json"
        )))
        .unwrap()
        .problem::<2>()
        .unwrap();
        let mut options = baldes::master::tree::ModelTreeOptions::default();
        options.node.cg.stabilize = false;
        options.node.cg.bound_mode = mode;
        options.packing = Some(
            serde_json::from_str(include_str!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/tests/fixtures/multi_network_native_packing.json"
            )))
            .unwrap(),
        );
        options.mapped_strong_branching = None;
        let result = baldes::master::tree::solve(problem, options).unwrap();
        assert_eq!(result.status, baldes::master::tree::TreeStatus::Optimal);
        assert_eq!(result.incumbent, 2.);
        assert_eq!(result.bound, 2.);
        assert_eq!(result.processed_nodes, 7);
        assert_eq!(result.strong_branching_trials, 0);
        assert!(result.mapped_branch_evaluations.is_empty());
    }
}

#[cfg(feature = "multi-time-windows")]
#[test]
fn native_packing_registration_survives_window_splitting_and_restores_input_arcs() {
    let mut problem = model::format::Model::from_json(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/multi_network_native_tree.json"
    )))
    .unwrap()
    .problem::<2>()
    .unwrap();
    for network in &mut problem.networks {
        for vertex in 0..network.graph.vertices.len() {
            network.graph.set_bounds(vertex, 0, 0., 20.).unwrap();
        }
        network
            .graph
            .set_time_windows(
                1,
                0,
                &[Window::new(0., 2.).unwrap(), Window::new(8., 10.).unwrap()],
            )
            .unwrap();
    }
    let mut options = baldes::master::tree::ModelTreeOptions::default();
    options.node.cg.stabilize = false;
    options.time_window_selection = Some(0);
    options.packing = Some(
        serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/multi_network_native_packing.json"
        )))
        .unwrap(),
    );
    let result = baldes::master::tree::solve(problem, options).unwrap();
    assert_eq!(result.status, baldes::master::tree::TreeStatus::Optimal);
    assert_eq!(result.incumbent, 2.);
    assert_eq!(result.bound, 2.);
    for fixed in result.routes {
        let upper = if fixed.route.network == 7 { 10 } else { 8 };
        assert!(fixed
            .route
            .route
            .arcs
            .iter()
            .all(|id| (0..upper).contains(id)));
    }
}

#[test]
fn native_packing_registration_can_start_from_a_saved_root_snapshot() {
    let problem = model::format::Model::from_json(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/multi_network_native_tree.json"
    )))
    .unwrap()
    .problem::<2>()
    .unwrap();
    let mut options = baldes::master::tree::ModelTreeOptions::default();
    options.node.cg.stabilize = false;
    let output = model::solve(problem, options.node.clone()).unwrap();
    options.packing = Some(
        serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/multi_network_native_packing.json"
        )))
        .unwrap(),
    );
    let result = output.snapshot.solve_tree(options).unwrap();
    assert_eq!(result.status, baldes::master::tree::TreeStatus::Optimal);
    assert_eq!(result.incumbent, 2.);
    assert_eq!(result.bound, 2.);
    assert_eq!(output.master.lp, 1.5);
}

#[test]
fn finite_variable_admission_matches_native_negative_path_single_solver_entry() {
    for mode in [BoundMode::Numerical, BoundMode::Certified] {
        let problem = model::format::Model::from_json(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/multi_network_native_negative_path.json"
        )))
        .unwrap()
        .problem::<2>()
        .unwrap();
        let mut options = Options::default();
        options.cg.stabilize = false;
        options.variable_boxes = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/multi_network_native_negative_boxes.json"
        )))
        .unwrap();
        options.cg.bound_mode = mode;
        let output = model::solve(problem, options).unwrap();
        assert_eq!(output.master.status, Status::Optimal);
        assert_eq!(output.master.lp, -1.);
        assert_eq!(output.master.bound, -1.);
        assert_eq!(output.oracle_calls, vec![(7, 1)]);
        let mut options = Options::default();
        options.cg.stabilize = false;
        options.cg.bound_mode = mode;
        let problem = model::format::Model::from_json(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/multi_network_native_negative_path.json"
        )))
        .unwrap()
        .problem::<2>()
        .unwrap();
        let unbounded = model::solve(problem, options).unwrap();
        assert_eq!(unbounded.master.bound, -1.);
        assert_eq!(unbounded.oracle_calls, vec![(7, 3)]);
    }
}

#[test]
fn positive_arc_lower_and_cyclic_repeat_boxes_match_native_solver_entries() {
    for mode in [BoundMode::Numerical, BoundMode::Certified] {
        for (model_text, box_text, objective, calls) in [
            (
                include_str!("../fixtures/multi_network_native_positive_lower.json"),
                include_str!("../fixtures/multi_network_native_positive_boxes.json"),
                0.,
                0,
            ),
            (
                include_str!("../fixtures/multi_network_native_positive_lower.json"),
                include_str!("../fixtures/multi_network_native_negative_boxes.json"),
                0.,
                3,
            ),
            (
                include_str!("../fixtures/multi_network_native_cyclic_box.json"),
                include_str!("../fixtures/multi_network_native_cyclic_boxes.json"),
                -2.,
                1,
            ),
        ] {
            let problem = model::format::Model::from_json(model_text)
                .unwrap()
                .problem::<2>()
                .unwrap();
            let mut options = Options::default();
            options.cg.stabilize = false;
            options.cg.bound_mode = mode;
            options.variable_boxes = serde_json::from_str(box_text).unwrap();
            let output = model::solve(problem, options).unwrap();
            assert_eq!(output.master.status, Status::Optimal);
            assert_eq!(output.master.lp, objective);
            assert_eq!(output.master.bound, objective);
            assert_eq!(output.oracle_calls, vec![(7, calls)]);
        }
    }
}

#[test]
fn required_network_quantity_gets_a_real_witness_before_box_admission() {
    for mode in [BoundMode::Numerical, BoundMode::Certified] {
        for registered in [false, true] {
            let mut problem = model::format::Model::from_json(include_str!(
                "../fixtures/multi_network_native_negative_path.json"
            ))
            .unwrap()
            .problem::<2>()
            .unwrap();
            problem.networks[0].multiplicity.lower = 1.;
            for arc in &mut problem.networks[0].graph.arcs {
                arc.cost = 0.;
            }
            let mut options = Options::default();
            options.cg.stabilize = false;
            options.cg.bound_mode = mode;
            if registered {
                options.variable_boxes = serde_json::from_str(include_str!(
                    "../fixtures/multi_network_native_negative_boxes.json"
                ))
                .unwrap();
            }
            let output = model::solve(problem, options).unwrap();
            assert_eq!(output.master.status, Status::Optimal);
            assert_eq!(output.master.lp, 0.);
            assert_eq!(output.master.bound, 0.);
            assert_eq!(output.oracle_calls, vec![(7, 1)]);
            assert_eq!(output.values.iter().sum::<f64>(), 1.);
            assert!(output.routes.iter().any(|r| r.route.arcs == vec![0, 1]));
        }
    }
}

#[cfg(feature = "multi-time-windows")]
#[test]
fn original_variable_registry_keeps_window_copy_costs_and_restored_activities() {
    use baldes::master::model::{
        self,
        format::{Model, SolveResult},
    };
    use baldes::master::tree::ModelTreeOptions;
    let mut data: serde_json::Value = serde_json::from_str(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/multi_network_original_variables.json"
    )))
    .unwrap();
    data["networks"][0]["resources"] = serde_json::json!([{
        "kind":"disposable", "bounds":[[0.,0.],[0.,4.],[0.,4.],[0.,10.]],
        "time_windows":[null,[[1.,1.],[2.,2.]],null,null]
    }]);
    for arc in data["networks"][0]["arcs"].as_array_mut().unwrap() {
        arc["consumption"] = serde_json::json!([1.]);
    }
    let (problem, registry) = Model::from_json(&data.to_string())
        .unwrap()
        .compile::<2>()
        .unwrap();
    let mut options = ModelTreeOptions::default();
    options.node.cg.stabilize = false;
    options.time_window_selection = Some(0);
    registry.configure_tree(&mut options).unwrap();
    registry
        .configure_boxes(
            &mut options.node,
            &[model::OriginalVariableBound {
                network: 7,
                id: 0,
                lower: 0.,
                upper: Some(1.),
            }],
        )
        .unwrap();
    let output = baldes::master::tree::solve(problem, options).unwrap();
    assert_eq!(output.incumbent, -1.);
    assert!(output
        .routes
        .iter()
        .all(|r| r.route.route.arcs.iter().all(|&a| a < 4)));
    let report = SolveResult::tree(&output, 0.)
        .with_original_variables(&registry)
        .unwrap();
    assert_eq!(report.original_variables[0].lower, 1.);
    assert_eq!(report.original_variables[0].upper, 1.);
}

#[test]
fn signed_original_boxes_keep_negative_activity_with_optional_networks_and_preprocessing() {
    use baldes::master::model::{format::Model, OriginalVariableBound};
    for serial in [false, true] {
        for coefficient in [-0.5, -1.] {
            for mode in [BoundMode::Numerical, BoundMode::Certified] {
                for preprocessing in [false, true] {
                    for upper in [0., 4.] {
                        let arcs = if serial {
                            vec![(0, 1, true), (1, 2, true)]
                        } else {
                            vec![(0, 1, true), (1, 3, false), (0, 2, true), (2, 3, false)]
                        };
                        let n = if serial { 3 } else { 4 };
                        let data = serde_json::json!({
                            "rows": [],
                            "networks": [{"id":7, "vertices":n, "multiplicity":[0,1],
                                "resources":[
                                    {"kind":"disposable", "bounds":vec![[0.,8.];n]},
                                    {"kind":"disposable", "bounds":vec![[0.,8.];n]}],
                                "arcs": arcs.iter().map(|&(tail, head, attached)| {
                                    serde_json::json!({"tail":tail, "head":head, "cost":0.,
                                        "consumption":[1.,if attached {coefficient} else {0.}]})
                                }).collect::<Vec<_>>() }],
                            "pricing_variables":[{"network":7, "id":0, "cost":1.,
                                "arcs":arcs.iter().enumerate().filter(|(_,a)|a.2)
                                    .map(|(p,_)|(p,coefficient)).collect::<Vec<_>>() }]
                        });
                        let (problem, registry) = Model::from_json(&data.to_string())
                            .unwrap()
                            .compile::<2>()
                            .unwrap();
                        let mut options = Options::default();
                        options.cg.stabilize = false;
                        options.cg.bound_mode = mode;
                        options.preprocessing.enabled = preprocessing;
                        registry
                            .configure_boxes(
                                &mut options,
                                &[OriginalVariableBound {
                                    network: 7,
                                    id: 0,
                                    lower: -4.,
                                    upper: Some(upper),
                                }],
                            )
                            .unwrap();
                        let output = model::solve(problem, options).unwrap();
                        let expected = coefficient * if serial { 2. } else { 1. };
                        assert_eq!(output.master.status, Status::Optimal);
                        assert_eq!(output.master.lp, expected);
                        assert_eq!(output.master.bound, expected);
                        let activity = registry.activity(7, 0, &output).unwrap();
                        assert_eq!(activity.lower, expected);
                        assert_eq!(activity.upper, expected);
                        assert_eq!(output.oracle_calls, [(7, 3)]);
                    }
                }
            }
        }
    }
}

#[test]
fn certified_signed_resource_dag_closes_the_feasible_path_gap() {
    let input = model::format::Model::from_json(include_str!(
        "../fixtures/multi_network_native_signed_dag_bound.json"
    ))
    .unwrap();
    for mode in [BoundMode::Numerical, BoundMode::Certified] {
        let (problem, _) = input.clone().compile::<2>().unwrap();
        let mut opts = Options::default();
        opts.cg.stabilize = false;
        opts.cg.bound_mode = mode;
        let output = model::solve(problem, opts).unwrap();
        assert_eq!(output.master.status, Status::Optimal);
        assert_eq!(output.master.lp, -6.5);
        assert_eq!(output.master.bound, -6.5);
        assert_eq!(output.routes[0].route.nodes, vec![0, 1, 3, 4]);
    }
}
