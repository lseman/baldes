use baldes::{
    graph::{Graph, Window},
    master::{
        column_generation::{BoundMode, Status},
        networks::{self, Network, Options, Output, Problem, RowCoefficients},
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

fn optimal(output: &Output, expected: f64) {
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
    let coefficient = |route: &networks::NetworkRoute| {
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
    let output = networks::solve(problem.clone(), options()).unwrap();
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
    let parent = networks::solve(problem.clone(), options()).unwrap();
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
    let parent = networks::solve(problem.clone(), options()).unwrap();
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
    let parent = networks::solve(problem.clone(), options()).unwrap();
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
        let output = networks::solve(problem.clone(), opts).unwrap();
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
    let parent = networks::solve(problem.clone(), options()).unwrap();
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
    let mut opts = networks::AssignmentTreeOptions::default();
    opts.node = options();
    let tree = networks::solve_assignment_tree(triangle(), opts.clone()).unwrap();
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
    let stopped = networks::solve_assignment_tree(triangle(), opts.clone()).unwrap();
    assert_eq!(stopped.status, TreeStatus::NodeLimit);
    assert_eq!(stopped.open_nodes, 2);
    assert!((stopped.bound - 1.5).abs() < 1e-8);
    assert!(!stopped.incumbent.is_finite());
    opts.time_limit = std::time::Duration::ZERO;
    assert_eq!(
        networks::solve_assignment_tree(triangle(), opts)
            .unwrap()
            .status,
        TreeStatus::TimeLimit
    );
}

#[test]
fn assignment_tree_leaves_unsupported_or_artificial_nodes_open() {
    use baldes::master::tree::TreeStatus;
    let mut problem = triangle();
    // Put every pair in one network: row assignments are integral, although
    // route quantities are fractional. Another branching family is required.
    problem.networks = vec![network(
        7,
        &[(1., &[0, 1]), (1., &[1, 2]), (1., &[0, 2])],
        3,
    )];
    let mut opts = networks::AssignmentTreeOptions::default();
    opts.node = options();
    let result = networks::solve_assignment_tree(problem, opts.clone()).unwrap();
    assert_eq!(result.status, TreeStatus::Incomplete);
    assert_eq!(result.open_nodes, 1);
    assert!((result.bound - 1.5).abs() < 1e-8);
    assert!(!result.incumbent.is_finite());

    let impossible = Problem {
        rows: vec![Row {
            sense: RowSense::Equal,
            rhs: 1.,
        }],
        networks: vec![network(7, &[(1., &[])], 1)],
    };
    let incomplete = networks::solve_assignment_tree(impossible, opts).unwrap();
    assert_eq!(incomplete.status, TreeStatus::Incomplete);
    assert_eq!(incomplete.open_nodes, 1);
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
    let parent = networks::solve(
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
fn unknown_original_domain_certificate_stays_unknown() {
    let mut graph = Graph::<2>::new(4, 1).unwrap();
    for vertex in 0..4 {
        graph.set_bounds(vertex, 0, 0., 2.).unwrap();
    }
    for (tail, head, cost) in [(0, 1, 0.), (1, 2, -1.), (2, 1, 0.), (1, 3, 0.)] {
        let arc = graph.add_arc(tail, head, cost).unwrap();
        graph.arc_mut(arc).unwrap().consumption[0] = if head == 3 { 0. } else { 1. };
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
    let output = networks::solve(problem, opts.clone()).unwrap();
    assert_eq!(output.master.status, Status::Stalled);
    assert_eq!(output.master.certified_bound, f64::NEG_INFINITY);
    assert_eq!(output.master.bound, f64::NEG_INFINITY);
    let resumed = output.snapshot.solve(opts).unwrap();
    assert_eq!(resumed.master.status, Status::Stalled);
    assert_eq!(resumed.master.bound, f64::NEG_INFINITY);
}

#[test]
fn stabilized_children_keep_column_cleanup_metadata_consistent() {
    let mut opts = options();
    opts.cg.stabilize = true;
    opts.cg.column_cleanup_threshold = 3.;
    opts.cg.column_cleanup_ratio = 0.66;
    let parent = networks::solve(triangle(), opts.clone()).unwrap();
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
    let parent = networks::solve(problem, options()).unwrap();
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
    let root = networks::solve(problem.clone(), options()).unwrap();
    assert!(!root.routes.is_empty());
    assert!(root.snapshot.fix_route(0, 1.).is_err());
    let mut opts = networks::AssignmentTreeOptions::default();
    opts.node = options();
    let result = networks::solve_assignment_tree(problem, opts).unwrap();
    assert_eq!(result.status, TreeStatus::Incomplete);
    assert_eq!(result.open_nodes, 1);
    assert!(!result.incumbent.is_finite());
}
