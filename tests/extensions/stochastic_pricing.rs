use baldes::extensions::stochastic::{Demand, Policy, Problem};

fn problem() -> Problem {
    let demands = (2..=5)
        .map(|d| Demand::new(vec![(0, 0.65), (d, 0.2), (10, 0.15)]).unwrap())
        .collect();
    let costs = (0..5)
        .map(|i| (0..5).map(|j| ((j + 5 - i) % 5) as f64).collect())
        .collect();
    Problem::new(10, demands, costs).unwrap()
}
fn close(a: f64, b: f64) {
    assert!(
        (a - b).abs() <= 1e-6 * (1. + a.abs().max(b.abs())),
        "{a} != {b}"
    );
}

#[test]
fn theta_graph_reweights_negative_rounded_arcs_without_changing_route_costs() {
    let p = Problem::new(
        10,
        vec![Demand::new(vec![(1, 1.)]).unwrap(); 3],
        vec![
            vec![0., 1., 1., 1.],
            vec![1., 0., 1., 1.],
            vec![1., 1., 0., 1.],
            vec![1., 1., 3., 0.],
        ],
    )
    .unwrap();
    let graph = p.theta_cut_graph(3).unwrap();
    assert!(graph.arcs.iter().all(|arc| arc.cost >= 0.));
    for route in [vec![1, 2, 3], vec![3, 2, 1], vec![2], vec![1, 3]] {
        let mut from = graph.source;
        let mut cost = 0.;
        for to in route.iter().copied().chain(std::iter::once(graph.sink)) {
            let arc = graph
                .arcs
                .iter()
                .find(|a| a.tail == from && a.head == to)
                .unwrap();
            cost += arc.cost;
            from = to;
        }
        close(cost, p.relaxed_route_cost(&route));
    }
}

#[cfg(feature = "clp")]
fn capacity_tight_problem() -> Problem {
    let demands = (0..7)
        .map(|k| Demand::new(vec![(3 + k % 3, 0.6), (6, 0.4)]).unwrap())
        .collect();
    let position: [f64; 8] = [6., 0., 3., 7., 12., 5., 9., 14.];
    let costs = position
        .iter()
        .map(|a| {
            position
                .iter()
                .map(|b| if a == b { 0. } else { (a - b).abs() + 1. })
                .collect()
        })
        .collect();
    Problem::new(10, demands, costs).unwrap()
}

#[cfg(feature = "clp")]
#[test]
fn routing_tree_with_theta_cuts_matches_the_tiny_oracle() {
    use baldes::extensions::stochastic::TinyOptions;
    use baldes::master::tree::{TreeOptions, TreeStatus};
    for (p, fleets) in [
        (problem(), vec![1, 2, 4]),
        (capacity_tight_problem(), vec![4, 5]),
    ] {
        for policy in [Policy::Classical, Policy::OptimalRestocking] {
            for &vehicles in &fleets {
                let oracle = p
                    .solve_tiny(
                        policy,
                        TinyOptions {
                            max_vehicles: vehicles,
                            route_budget: 100_000,
                        },
                    )
                    .unwrap()
                    .unwrap();
                for (threads, dynamic_ng, cut_limit, completion) in [
                    (1, None, None, true),
                    (4, None, None, true),
                    (1, Some(0.), None, true),
                    (1, Some(1.), Some(3.), true),
                    (4, Some(1.), Some(12.), true),
                    (1, Some(-1.), Some(8.), true),
                    (1, None, None, false),
                    (4, Some(1.), Some(8.), false),
                ] {
                    let mut tree = TreeOptions {
                        threads,
                        ..Default::default()
                    };
                    // Leave CutOptions at its generic default: solve_theta_cuts
                    // must enable both rank-one and mean-capacity separation.
                    if let Some(limit) = cut_limit {
                        tree.root
                            .pricing_options
                            .push(("BALDESrankOneCutsMaxNumPerRound".into(), limit));
                    }
                    if !completion {
                        tree.root
                            .pricing_options
                            .push(("BALDESloadCompletionResource".into(), -1.));
                    }
                    if let Some(mode) = dynamic_ng {
                        tree.root
                            .pricing_options
                            .push(("BALDESdynamicNGmode".into(), mode));
                    }
                    let answer = p.solve_theta_cuts(policy, vehicles, 2, tree).unwrap();
                    assert_eq!(answer.status, TreeStatus::Optimal, "{policy:?} {vehicles}");
                    close(answer.incumbent, oracle.total_cost);
                    assert!(answer.bound <= oracle.total_cost + 1e-6);
                    let replayed: f64 = answer
                        .routes
                        .iter()
                        .map(|r| {
                            let customers: Vec<usize> = r.nodes[1..r.nodes.len() - 1]
                                .iter()
                                .map(|&v| v as usize)
                                .collect();
                            p.evaluate_route(&customers, policy).unwrap().total_cost
                        })
                        .sum();
                    close(replayed, oracle.total_cost);
                }
            }
        }
    }
}

#[cfg(feature = "clp")]
#[test]
fn theta_recourse_runs_bucket_fixing_without_enumerating_routes() {
    use baldes::extensions::stochastic::TinyOptions;
    use baldes::master::{cg, routing};
    let p = Problem::new(
        2,
        vec![Demand::new(vec![(0, 0.5), (2, 0.5)]).unwrap(); 3],
        (0..4)
            .map(|i| (0..4).map(|j| if i == j { 0. } else { 1. }).collect())
            .collect(),
    )
    .unwrap();
    for policy in [Policy::Classical, Policy::OptimalRestocking] {
        let oracle = p
            .solve_tiny(
                policy,
                TinyOptions {
                    max_vehicles: 2,
                    route_budget: 1000,
                },
            )
            .unwrap()
            .unwrap();
        let mut objectives = vec![];
        for enabled in [false, true] {
            let mut options = routing::RootOptions::default();
            // A fractional pair cover leaves an open gap. Theta recourse
            // disables enumeration even though RootOptions defaults it on.
            options.incumbent = oracle.total_cost;
            options.pricing_options.push((
                "BALDESapplyReducedCostFixing".into(),
                if enabled { 1. } else { 0. },
            ));
            let answer =
                routing::solve(p.theta_cut_context(policy, 2, 3).unwrap(), options).unwrap();
            assert_eq!(answer.master.status, cg::Status::Optimal);
            assert_eq!(answer.enumerated_routes, 0);
            assert_eq!(answer.fixing_calls > 0, enabled);
            assert!(answer.master.bound <= oracle.total_cost + 1e-6);
            objectives.push(answer.master.lp);
        }
        close(objectives[0], objectives[1]);
    }
}

#[cfg(feature = "clp")]
#[test]
fn hgs_plan_seeds_theta_tree_without_pool_search_or_node_work() {
    use baldes::extensions::stochastic::{HgsWarmStart, SolveOptions};
    use baldes::master::tree::TreeOptions;
    use std::time::Duration;
    let p = problem();
    for policy in [Policy::Classical, Policy::OptimalRestocking] {
        let answer = p
            .solve(
                policy,
                2,
                SolveOptions {
                    tree: TreeOptions {
                        node_limit: 0,
                        mip_node_limit: 0,
                        ..Default::default()
                    },
                    hgs_warm_start: Some(HgsWarmStart {
                        time_limit: Duration::from_millis(30),
                        seed: 3,
                    }),
                    ..Default::default()
                },
            )
            .unwrap();
        let summary = answer.hgs_warm_start.unwrap();
        assert!(summary.accepted_columns > 0);
        assert!(summary.accepted_plan);
        assert_eq!(answer.tree.nodes, 0);
        assert_eq!(answer.tree.mip_nodes, 0);
        let mut covered = Vec::<usize>::new();
        let replay: f64 = answer
            .tree
            .routes
            .iter()
            .map(|route| {
                let customers: Vec<_> = route.nodes[1..route.nodes.len() - 1]
                    .iter()
                    .map(|&v| v as usize)
                    .collect();
                covered.extend(&customers);
                p.evaluate_route(&customers, policy).unwrap().total_cost
            })
            .sum();
        covered.sort_unstable();
        assert_eq!(covered, vec![1, 2, 3, 4]);
        assert!(answer.tree.routes.len() <= 2);
        close(answer.tree.incumbent, replay);
        close(summary.initial_objective.unwrap(), replay);
        assert!(answer.tree.bound <= replay + 1e-6);

        // Imported columns follow the Theta variables. Their route metadata
        // must not be attached to the auxiliary columns (which would also
        // apply binary route caps to unbounded Theta variables).
        let mut root = baldes::master::routing::RootOptions {
            initial_routes: answer.tree.routes.clone(),
            ..Default::default()
        };
        root.cg.max_iterations = 1;
        let seeded =
            baldes::master::routing::solve(p.theta_cut_context(policy, 2, 2).unwrap(), root)
                .unwrap();
        assert!(seeded.master.column_tags[..p.demands().len()]
            .iter()
            .all(|&tag| tag == -1));
        assert!(seeded.master.column_tags.iter().any(|&tag| tag >= 0));

        // The tree must reconstruct the stochastic objective, not trust route
        // metadata or require enough time to rediscover this plan in its pool.
        let mut routes = answer.tree.routes;
        for route in &mut routes {
            route.cost = -999.;
            route.original_cost = -999.;
            route.nodes.clear();
        }
        let context = p.theta_cut_context(policy, 2, 2).unwrap();
        let out = baldes::master::tree::solve(
            context,
            TreeOptions {
                initial_solution: routes.clone(),
                time_limit: Duration::ZERO,
                mip_node_limit: 0,
                ..Default::default()
            },
        )
        .unwrap();
        close(out.incumbent, replay);
        assert_eq!(out.nodes, 0);
        assert!(!out.routes[0].nodes.is_empty());

        let mut repeated = routes.clone();
        repeated.push(routes[0].clone());
        assert!(baldes::master::tree::solve(
            p.theta_cut_context(policy, 2, 2).unwrap(),
            TreeOptions {
                initial_solution: repeated,
                ..Default::default()
            }
        )
        .is_err());
        routes[0].arcs[0] = i32::MAX;
        assert!(baldes::master::tree::solve(
            p.theta_cut_context(policy, 2, 2).unwrap(),
            TreeOptions {
                initial_solution: routes,
                ..Default::default()
            }
        )
        .is_err());
    }
}

#[cfg(feature = "clp")]
#[test]
fn hgs_seeded_theta_solve_matches_tiny_oracle() {
    use baldes::extensions::stochastic::{HgsWarmStart, SolveOptions, TinyOptions};
    use baldes::master::tree::TreeStatus;
    use std::time::Duration;
    let p = problem();
    for policy in [Policy::Classical, Policy::OptimalRestocking] {
        let oracle = p
            .solve_tiny(
                policy,
                TinyOptions {
                    max_vehicles: 2,
                    route_budget: 1000,
                },
            )
            .unwrap()
            .unwrap();
        let answer = p
            .solve(
                policy,
                2,
                SolveOptions {
                    hgs_warm_start: Some(HgsWarmStart {
                        time_limit: Duration::from_millis(30),
                        seed: 3,
                    }),
                    ..Default::default()
                },
            )
            .unwrap();
        assert_eq!(answer.tree.status, TreeStatus::Optimal);
        close(answer.tree.incumbent, oracle.total_cost);
        assert!(answer.tree.bound <= oracle.total_cost + 1e-6);
    }
}

#[cfg(feature = "clp")]
#[test]
fn hgs_budget_is_charged_to_theta_solve_and_zero_budget_skips_hgs() {
    use baldes::extensions::stochastic::{HgsWarmStart, SolveOptions};
    use baldes::master::tree::TreeOptions;
    use std::time::Duration;
    let p = problem();
    let answer = p
        .solve(
            Policy::OptimalRestocking,
            2,
            SolveOptions {
                tree: TreeOptions {
                    time_limit: Duration::from_millis(30),
                    mip_node_limit: 0,
                    ..Default::default()
                },
                hgs_warm_start: Some(HgsWarmStart {
                    time_limit: Duration::from_secs(10),
                    seed: 3,
                }),
                ..Default::default()
            },
        )
        .unwrap();
    assert!(answer.hgs_warm_start.unwrap().accepted_plan);
    assert!(answer.tree.incumbent.is_finite());
    assert_eq!(answer.tree.nodes, 0);

    let answer = p
        .solve(
            Policy::OptimalRestocking,
            2,
            SolveOptions {
                tree: TreeOptions {
                    time_limit: Duration::ZERO,
                    ..Default::default()
                },
                hgs_warm_start: Some(HgsWarmStart {
                    time_limit: Duration::from_secs(10),
                    seed: 3,
                }),
                ..Default::default()
            },
        )
        .unwrap();
    let summary = answer.hgs_warm_start.unwrap();
    assert_eq!(summary.elapsed_seconds, 0.);
    assert_eq!(summary.accepted_columns, 0);
    assert!(!summary.accepted_plan);
    assert!(!answer.tree.incumbent.is_finite());
    assert_eq!(answer.tree.nodes, 0);
}
