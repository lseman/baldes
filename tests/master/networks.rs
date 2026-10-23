use baldes::{
    graph::{Graph, Window},
    master::{
        cg::{BoundMode, Status},
        mp::{self, Network, Options, Output, Problem, RowCoefficients},
        root::Row,
        RowSense,
    },
};

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

fn assignment_mass(output: &Output, problem: &Problem<2>, network: i32, row: usize) -> f64 {
    let coefficient = |route: &mp::NetworkRoute| {
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
    let output = mp::solve(problem.clone(), options()).unwrap();
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
    let parent = mp::solve(problem.clone(), options()).unwrap();
    let left = parent.snapshot.branch_assignment(0, 7, true).unwrap();
    let right = parent.snapshot.branch_assignment(0, 7, false).unwrap();
    // Every integer cover costs at least ceil(3/2)=2. Both sides have
    // explicit pair+singleton covers of cost two.
    for _ in 0..2 {
        let excluded = right.solve(options()).unwrap();
        let assigned = left.solve(options()).unwrap();
        optimal(&excluded, 2.);
        optimal(&assigned, 2.);
        assert!(assignment_mass(&excluded, &problem, 7, 0).abs() < 1e-8);
        assert!((assignment_mass(&assigned, &problem, 7, 0) - 1.).abs() < 1e-8);
        assert!(excluded.oracle_calls.iter().all(|&(_, calls)| calls > 0));
        assert!(assigned.oracle_calls.iter().all(|&(_, calls)| calls > 0));
        optimal(&parent.snapshot.solve(options()).unwrap(), 1.5);
    }
}

#[test]
fn additive_cut_and_fixed_route_survive_descendants() {
    let problem = triangle();
    let parent = mp::solve(problem.clone(), options()).unwrap();
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
    let parent = mp::solve(problem.clone(), options()).unwrap();
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
        let output = mp::solve(problem.clone(), opts).unwrap();
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
    let parent = mp::solve(problem.clone(), options()).unwrap();
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
    let mut opts = mp::AssignmentTreeOptions::default();
    opts.node = options();
    let tree = mp::solve_assignment_tree(triangle(), opts.clone()).unwrap();
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
    assert_eq!(tree.nodes, 3);
    assert!(tree.routes.iter().all(|r| r.quantity.fract() == 0.));

    opts.node_limit = 1;
    let stopped = mp::solve_assignment_tree(triangle(), opts.clone()).unwrap();
    assert_eq!(stopped.status, TreeStatus::NodeLimit);
    assert_eq!(stopped.open_nodes, 2);
    assert!((stopped.bound - 1.5).abs() < 1e-8);
    assert!(!stopped.incumbent.is_finite());
    opts.time_limit = std::time::Duration::ZERO;
    assert_eq!(
        mp::solve_assignment_tree(triangle(), opts).unwrap().status,
        TreeStatus::TimeLimit
    );
}

#[test]
fn tree_keeps_unproved_infeasibility_open_and_closes_exact_original_contradictions() {
    use baldes::master::tree::TreeStatus;
    let mut problem = triangle();
    // Put every pair in one network: row assignments are integral, although
    // route quantities are fractional. Multiplicity branching is valid, but
    // these empty child LPs lack a whole-domain infeasibility proof.
    problem.networks = vec![network(
        7,
        &[(1., &[0, 1]), (1., &[1, 2]), (1., &[0, 2])],
        3,
    )];
    let mut opts = mp::AssignmentTreeOptions::default();
    opts.node = options();
    let result = mp::solve_assignment_tree(problem, opts.clone()).unwrap();
    assert_eq!(result.status, TreeStatus::Incomplete);
    assert_eq!(result.open_nodes, 2);
    assert!((result.bound - 1.5).abs() < 1e-8);
    assert!(!result.incumbent.is_finite());

    let impossible = Problem {
        rows: vec![Row {
            sense: RowSense::Equal,
            rhs: 1.,
        }],
        networks: vec![network(7, &[(1., &[])], 1)],
    };
    let incomplete = mp::solve_assignment_tree(impossible, opts).unwrap();
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
    let parent = mp::solve(
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
        let output = mp::solve(problem, opts.clone()).unwrap();
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
    let parent = mp::solve(triangle(), opts.clone()).unwrap();
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
    let parent = mp::solve(problem, options()).unwrap();
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
    let root = mp::solve(problem.clone(), options()).unwrap();
    assert!(!root.routes.is_empty());
    assert!(root.snapshot.fix_route(0, 1.).is_err());
    let mut opts = mp::AssignmentTreeOptions::default();
    opts.node = options();
    let result = mp::solve_assignment_tree(problem, opts).unwrap();
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
    use mp::{EnumerationOptions, RankOneCoefficients};
    let problem = Problem {
        rows: (0..4)
            .map(|_| Row {
                sense: RowSense::Equal,
                rhs: 1.,
            })
            .collect(),
        networks: vec![nonlinear_network(7, 0), nonlinear_network(42, 2)],
    };
    let root = mp::solve(problem, options()).unwrap();
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
    let root = mp::solve(problem.clone(), options()).unwrap();
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
    use mp::{EnumerationOptions, PreprocessingOptions, RankOneCoefficients};
    let problem = Problem {
        rows: (0..4)
            .map(|_| Row {
                sense: RowSense::Equal,
                rhs: 1.,
            })
            .collect(),
        networks: vec![nonlinear_network(7, 0), nonlinear_network(42, 2)],
    };
    let root = mp::solve(problem, options()).unwrap();
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
    let root = mp::solve(problem, options()).unwrap();
    assert_eq!(
        root.snapshot.preprocessing_report().skipped.as_deref(),
        Some("RoundedFlowCoefficients")
    );
    assert_eq!(root.snapshot.preprocessing_report().forbidden_arcs, 0);
    let child = root
        .snapshot
        .enumerate(&mp::EnumerationOptions::default())
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
    let root = mp::solve(triangle(), options()).unwrap();
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
