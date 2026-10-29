use baldes::extensions::stochastic::{
    ArcBranch, Demand, Policy, PricingOptions, PricingStatus, Problem,
};

#[cfg(feature = "clp")]
#[test]
fn hgs_warm_start_replays_routes_and_preserves_a_valid_bound() {
    use baldes::extensions::stochastic::{HgsWarmStart, SolveOptions};
    let p = problem();
    for policy in [Policy::Classical, Policy::OptimalRestocking] {
        let answer = p
            .solve(
                policy,
                2,
                SolveOptions {
                    max_nodes: 0,
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
        let solution = answer.solution.unwrap();
        let mut covered: Vec<_> = solution.routes.iter().flatten().copied().collect();
        covered.sort_unstable();
        assert_eq!(covered, vec![1, 2, 3, 4]);
        assert!(solution.routes.len() <= 2);
        let replay: f64 = solution
            .routes
            .iter()
            .map(|r| p.evaluate_route(r, policy).unwrap().total_cost)
            .sum();
        close(solution.total_cost, replay);
        assert!(answer.lower_bound <= replay);
    }
}

#[cfg(feature = "clp")]
#[test]
fn adaptive_ng_schedule_matches_tiny_oracle() {
    use baldes::extensions::stochastic::{
        ColumnGenerationOptions, SolveOptions, SolveStatus, TinyOptions,
    };
    let demands = (0..6)
        .map(|j| Demand::new(vec![(0, 0.4), (2 + j % 2, 0.6)]).unwrap())
        .collect();
    let costs = (0usize..7)
        .map(|i| (0usize..7).map(|j| i.abs_diff(j) as f64).collect())
        .collect();
    let p = Problem::new(5, demands, costs).unwrap();
    for policy in [Policy::Classical, Policy::OptimalRestocking] {
        let oracle = p
            .solve_tiny(
                policy,
                TinyOptions {
                    max_vehicles: 3,
                    route_budget: 100_000,
                },
            )
            .unwrap()
            .unwrap();
        let answer = p
            .solve(
                policy,
                3,
                SolveOptions {
                    column_generation: ColumnGenerationOptions {
                        ng_size: 5,
                        ..Default::default()
                    },
                    ..Default::default()
                },
            )
            .unwrap();
        assert_eq!(answer.status, SolveStatus::Optimal);
        close(answer.solution.unwrap().total_cost, oracle.total_cost);
        assert!(answer.lower_bound <= oracle.total_cost + 1e-6);
    }
}
use std::time::Duration;

fn problem() -> Problem {
    let demands = (2..=5)
        .map(|d| Demand::new(vec![(0, 0.65), (d, 0.2), (10, 0.15)]).unwrap())
        .collect();
    let costs = (0..5)
        .map(|i| (0..5).map(|j| ((j + 5 - i) % 5) as f64).collect())
        .collect();
    Problem::new(10, demands, costs).unwrap()
}
fn uses(route: &[usize], b: ArcBranch) -> bool {
    let mut from = 0;
    for to in route.iter().copied().chain(std::iter::once(0)) {
        if from == b.from && to == b.to {
            return true;
        }
        from = to;
    }
    false
}
fn enumerate(p: &Problem, policy: Policy, route: &mut Vec<usize>, out: &mut Vec<Vec<usize>>) {
    for j in 1..=p.demands().len() {
        if route.contains(&j) {
            continue;
        }
        route.push(j);
        if p.evaluate_route(route, policy).is_ok() {
            out.push(route.clone());
            enumerate(p, policy, route, out);
        }
        route.pop();
    }
}
fn close(a: f64, b: f64) {
    assert!((a - b).abs() <= 1e-7 * (1. + a.abs()), "{a} != {b}");
}

#[test]
fn baldes_prices_true_recourse_in_both_directions_against_exhaustive_orders() {
    let p = problem();
    for policy in [Policy::Classical, Policy::OptimalRestocking] {
        let mut orders = vec![];
        enumerate(&p, policy, &mut vec![], &mut orders);
        for duals in [vec![0.; 4], vec![5., 8., 2., 4.], vec![-2., 7., 1., 12.]] {
            for bidirectional in [false, true] {
                let fleet = -0.7;
                let expected = orders
                    .iter()
                    .map(|r| {
                        p.evaluate_route(r, policy).unwrap().total_cost
                            - r.iter().map(|&j| duals[j - 1]).sum::<f64>()
                            - fleet
                    })
                    .fold(f64::INFINITY, f64::min);
                let priced = p
                    .price(
                        policy,
                        &duals,
                        fleet,
                        &[],
                        &[],
                        false,
                        PricingOptions {
                            max_columns: 1,
                            bidirectional,
                            ..Default::default()
                        },
                    )
                    .unwrap();
                assert_eq!(priced.status, PricingStatus::Complete);
                assert!(priced.statistics.generated > 0);
                close(priced.minimum_reduced_cost.unwrap(), expected);
                assert!(priced.lower_bound.unwrap() <= expected + 1e-7);
            }
        }
    }
}

#[test]
fn physical_graph_replay_preserves_original_and_priced_objectives() {
    let p = problem();
    for policy in [Policy::Classical, Policy::OptimalRestocking] {
        let graph = p
            .pricing_graph(policy, &[1., 2., 3., 4.], -2., &[], &[], false)
            .unwrap();
        let mut orders = vec![];
        enumerate(&p, policy, &mut vec![], &mut orders);
        for route in orders {
            let mut arcs = vec![];
            let mut i = 0;
            for j in route.iter().copied().chain(std::iter::once(5)) {
                arcs.push(
                    graph
                        .arcs
                        .iter()
                        .position(|a| a.tail == i && a.head == j)
                        .unwrap(),
                );
                i = j;
            }
            let breakdown = graph
                .coupled_kernel()
                .unwrap()
                .replay_cost_components(&arcs)
                .unwrap()
                .unwrap();
            let evaluation = p.evaluate_route(&route, policy).unwrap();
            close(breakdown.total, evaluation.expected_recourse_cost);
            close(breakdown.resources[0], evaluation.expected_recourse_cost);
            let original = arcs.iter().map(|&a| graph.arcs[a].cost).sum::<f64>() + breakdown.total;
            close(original, evaluation.total_cost);
        }
    }
}

#[test]
fn branch_duals_and_forbidden_arcs_are_enforced_during_pricing() {
    let p = problem();
    let branches = [
        ArcBranch {
            from: 1,
            to: 2,
            required: false,
        },
        ArcBranch {
            from: 0,
            to: 3,
            required: true,
        },
    ];
    let duals = [10.; 4];
    for phase_one in [true, false] {
        for bidirectional in [true, false] {
            let mut orders = vec![];
            enumerate(&p, Policy::OptimalRestocking, &mut vec![], &mut orders);
            let expected = orders
                .iter()
                .filter(|r| !uses(r, branches[0]))
                .map(|r| {
                    (if phase_one {
                        0.
                    } else {
                        p.evaluate_route(r, Policy::OptimalRestocking)
                            .unwrap()
                            .total_cost
                    }) - 10. * r.len() as f64
                        - if uses(r, branches[1]) { 3. } else { 0. }
                })
                .fold(f64::INFINITY, f64::min);
            let priced = p
                .price(
                    Policy::OptimalRestocking,
                    &duals,
                    0.,
                    &branches,
                    &[0., 3.],
                    phase_one,
                    PricingOptions {
                        bidirectional,
                        ..Default::default()
                    },
                )
                .unwrap();
            assert_eq!(priced.status, PricingStatus::Complete);
            close(priced.minimum_reduced_cost.unwrap(), expected);
            assert!(priced.lower_bound.unwrap() <= expected + 1e-7);
            assert!(priced
                .routes
                .iter()
                .all(|r| !uses(&r.customers, branches[0])));
        }
    }
}

#[test]
fn pricing_limits_never_claim_completion() {
    let p = problem();
    for options in [
        PricingOptions {
            max_labels: 0,
            ..Default::default()
        },
        PricingOptions {
            max_labels: 1,
            ..Default::default()
        },
        PricingOptions {
            time_limit: Duration::ZERO,
            ..Default::default()
        },
    ] {
        let result = p
            .price(Policy::Classical, &[10.; 4], 0., &[], &[], false, options)
            .unwrap();
        assert_ne!(result.status, PricingStatus::Complete);
    }
}

#[cfg(feature = "clp")]
mod master {
    use super::*;
    use baldes::extensions::stochastic::{
        ColumnGenerationOptions, ColumnGenerationStatus, Master, SolveOptions, SolveStatus,
        TinyOptions,
    };
    use baldes::master::clp::ClpMaster;

    #[test]
    fn column_generation_converges_to_full_route_lp_with_recourse() {
        let p = problem();
        for policy in [Policy::Classical, Policy::OptimalRestocking] {
            let mut master =
                Master::new(p.clone(), policy, 4, vec![], ClpMaster::new().unwrap()).unwrap();
            let solved = master.optimize(ColumnGenerationOptions::default()).unwrap();
            assert_eq!(solved.status, ColumnGenerationStatus::Converged);
            let oracle = p
                .solve_tiny(
                    policy,
                    TinyOptions {
                        max_vehicles: 4,
                        route_budget: 1000,
                    },
                )
                .unwrap()
                .unwrap();
            assert!(solved.lower_bound.unwrap() <= oracle.total_cost + 1e-6);
            assert!(solved.pricing_calls > 0);
            assert!(master.columns().iter().any(|c| c.customers.len() > 1));
            // Compare with the same master seeded with every elementary route.
            let mut all = vec![];
            enumerate(&p, policy, &mut vec![], &mut all);
            let mut full =
                Master::new(p.clone(), policy, 4, vec![], ClpMaster::new().unwrap()).unwrap();
            for route in all {
                full.add_route(&route).unwrap();
            }
            let full_result = full.optimize(ColumnGenerationOptions::default()).unwrap();
            assert_eq!(full_result.status, ColumnGenerationStatus::Converged);
            close(
                solved.snapshot.unwrap().objective,
                full_result.snapshot.unwrap().objective,
            );
        }
    }

    #[test]
    fn branch_and_price_matches_tiny_oracle_for_both_policies() {
        let p = problem();
        for policy in [Policy::Classical, Policy::OptimalRestocking] {
            for vehicles in [1, 2, 4] {
                let answer = p.solve(policy, vehicles, SolveOptions::default()).unwrap();
                assert_eq!(answer.status, SolveStatus::Optimal);
                let oracle = p
                    .solve_tiny(
                        policy,
                        TinyOptions {
                            max_vehicles: vehicles,
                            route_budget: 1000,
                        },
                    )
                    .unwrap()
                    .unwrap();
                close(answer.solution.unwrap().total_cost, oracle.total_cost);
                assert!(answer.lower_bound <= oracle.total_cost + 1e-6);
                assert!(answer.pricing_calls > 0);
            }
        }
    }

    fn fractional() -> Problem {
        Problem::new(
            2,
            vec![Demand::new(vec![(1, 1.)]).unwrap(); 3],
            vec![
                vec![0., 1., 1., 1.],
                vec![1., 0., 1., 1.],
                vec![1., 1., 0., 1.],
                vec![1., 1., 1., 0.],
            ],
        )
        .unwrap()
    }

    #[test]
    fn fractional_root_requires_arc_branching_and_child_pricing() {
        let p = fractional();
        let answer = p
            .solve(Policy::Classical, 2, SolveOptions::default())
            .unwrap();
        assert_eq!(answer.status, SolveStatus::Optimal);
        assert!(
            answer.nodes > 1,
            "must branch on fractional pair-route cover"
        );
        close(answer.solution.unwrap().total_cost, 5.);
        assert!(answer.lower_bound > 4.9);
        let limited = p
            .solve(
                Policy::Classical,
                2,
                SolveOptions {
                    max_nodes: 1,
                    ..Default::default()
                },
            )
            .unwrap();
        assert_eq!(limited.status, SolveStatus::NodeLimit);
        assert!(limited.lower_bound <= 4.5 + 1e-6);
    }

    #[test]
    fn phase_one_proves_fleet_and_required_arc_infeasibility() {
        let p = fractional();
        assert_eq!(
            p.solve(Policy::Classical, 1, SolveOptions::default())
                .unwrap()
                .status,
            SolveStatus::Infeasible
        );
        let branches = vec![
            ArcBranch {
                from: 1,
                to: 2,
                required: true,
            },
            ArcBranch {
                from: 1,
                to: 2,
                required: false,
            },
        ];
        let mut master =
            Master::new(p, Policy::Classical, 2, branches, ClpMaster::new().unwrap()).unwrap();
        assert_eq!(
            master
                .optimize(ColumnGenerationOptions::default())
                .unwrap()
                .status,
            ColumnGenerationStatus::Infeasible
        );
    }

    #[test]
    fn interrupted_pricing_retains_an_incumbent_and_conservative_bound() {
        let p = problem();
        let options = SolveOptions {
            column_generation: ColumnGenerationOptions {
                pricing: PricingOptions {
                    max_labels: 1,
                    ..Default::default()
                },
                ..Default::default()
            },
            ..Default::default()
        };
        let answer = p.solve(Policy::Classical, 1, options).unwrap();
        assert!(matches!(answer.status, SolveStatus::NodeIncomplete(_)));
        let oracle = p
            .solve_tiny(
                Policy::Classical,
                TinyOptions {
                    max_vehicles: 1,
                    route_budget: 100_000,
                },
            )
            .unwrap()
            .unwrap();
        assert!(answer.solution.is_some());
        assert!(answer.lower_bound > 0.);
        assert!(answer.lower_bound <= oracle.total_cost + 1e-6);
        assert!(answer.solution.unwrap().total_cost >= oracle.total_cost - 1e-6);
    }
}

#[cfg(feature = "clp")]
#[test]
fn solver_runs_beyond_the_tiny_enumeration_limit() {
    use baldes::extensions::stochastic::{SolveOptions, SolveStatus, TinyOptions};
    let n = 10;
    let costs = (0..=n)
        .map(|i| (0..=n).map(|j| if i == j { 0. } else { 1. }).collect())
        .collect();
    let p = Problem::new(1, vec![Demand::new(vec![(1, 1.)]).unwrap(); n], costs).unwrap();
    assert!(p
        .solve_tiny(
            Policy::Classical,
            TinyOptions {
                max_vehicles: n,
                route_budget: 100
            }
        )
        .is_err());
    let answer = p
        .solve(Policy::Classical, n, SolveOptions::default())
        .unwrap();
    assert_eq!(answer.status, SolveStatus::Optimal);
    close(answer.solution.unwrap().total_cost, 20.);
}

#[cfg(feature = "clp")]
#[test]
fn cost_plugin_never_reuses_feasibility_certificate_as_a_zero_cost_proof() {
    let p = problem();
    let g = p
        .pricing_graph(Policy::Classical, &[0.; 4], 0., &[], &[], false)
        .unwrap();
    let nodes = [0, 1, 2, 5];
    let positions: Vec<_> = nodes
        .windows(2)
        .map(|w| {
            g.arcs
                .iter()
                .position(|a| a.tail == w[0] && a.head == w[1])
                .unwrap()
        })
        .collect();
    assert!(g
        .coupled_kernel()
        .unwrap()
        .certify_with_controls(&positions, &[], &mut 1_000_000, None)
        .is_none());
}

#[cfg(feature = "clp")]
#[test]
fn empty_instance_and_zero_fleet_have_distinct_statuses() {
    use baldes::extensions::stochastic::{SolveOptions, SolveStatus};
    let empty = Problem::new(1, vec![], vec![vec![0.]]).unwrap();
    let answer = empty
        .solve(Policy::Classical, 0, SolveOptions::default())
        .unwrap();
    assert_eq!(answer.status, SolveStatus::Optimal);
    close(answer.solution.unwrap().total_cost, 0.);
    assert_eq!(
        problem()
            .solve(Policy::Classical, 0, SolveOptions::default())
            .unwrap()
            .status,
        SolveStatus::Infeasible
    );
}

#[cfg(feature = "clp")]
#[test]
fn shared_routing_master_suppresses_and_restores_stochastic_plugin_costs() {
    use baldes::master::{
        cg::Status,
        routing::{self, RootOptions, RoutingContext},
    };
    let p = Problem::new(
        10,
        vec![
            Demand::new(vec![(4, 0.5), (8, 0.5)]).unwrap(),
            Demand::new(vec![(4, 1.)]).unwrap(),
        ],
        vec![vec![0., 1., 10.], vec![1., 0., 10.], vec![10., 10., 0.]],
    )
    .unwrap();
    let graph = p
        .pricing_graph(Policy::OptimalRestocking, &[0.; 2], 0., &[], &[], false)
        .unwrap();
    let mut options = RootOptions {
        enumerate: false,
        round_reduced_costs: false,
        ..Default::default()
    };
    options.cg.artificial_cost = 1.;
    options.cg.max_penalty_updates = 0;
    options.cg.insert_non_negative = false;
    options.cg.stabilize = false;
    let answer = routing::solve(
        RoutingContext::set_partitioning(graph, 1.).unwrap(),
        options,
    )
    .unwrap();
    assert_eq!(answer.master.status, Status::Optimal, "{answer:?}");
    close(answer.master.lp, 21.5);
}

#[test]
fn interrupted_sweeps_keep_completed_routes_statistics_and_a_global_bound() {
    let p = problem();
    let duals = [10.; 4];
    for policy in [Policy::Classical, Policy::OptimalRestocking] {
        let full = p
            .price(
                policy,
                &duals,
                -1.,
                &[],
                &[],
                false,
                PricingOptions::default(),
            )
            .unwrap();
        let partial = p
            .price(
                policy,
                &duals,
                -1.,
                &[],
                &[],
                false,
                PricingOptions {
                    max_labels: 25,
                    ..Default::default()
                },
            )
            .unwrap();
        assert_eq!(partial.status, PricingStatus::WorkLimit);
        assert!(partial.statistics.generated > 0);
        assert!(!partial.routes.is_empty());
        assert!(partial.minimum_reduced_cost.is_none());
        assert!(partial.lower_bound.unwrap() <= full.minimum_reduced_cost.unwrap() + 1e-7);
        for route in partial.routes {
            close(
                route.evaluation.total_cost,
                p.evaluate_route(&route.customers, policy)
                    .unwrap()
                    .total_cost,
            );
        }
    }
}

#[cfg(feature = "clp")]
#[test]
fn zero_time_and_iteration_limits_preserve_a_feasible_start_and_valid_bound() {
    use baldes::extensions::stochastic::{
        ColumnGenerationOptions, SolveOptions, SolveStatus, TinyOptions,
    };
    let p = problem();
    for policy in [Policy::Classical, Policy::OptimalRestocking] {
        let oracle = p
            .solve_tiny(
                policy,
                TinyOptions {
                    max_vehicles: 2,
                    route_budget: 100_000,
                },
            )
            .unwrap()
            .unwrap();
        for options in [
            SolveOptions {
                time_limit: Duration::ZERO,
                ..Default::default()
            },
            SolveOptions {
                max_nodes: 0,
                ..Default::default()
            },
            SolveOptions {
                column_generation: ColumnGenerationOptions {
                    max_iterations: 0,
                    ..Default::default()
                },
                ..Default::default()
            },
        ] {
            let answer = p.solve(policy, 2, options).unwrap();
            assert_ne!(answer.status, SolveStatus::Optimal);
            let solution = answer.solution.unwrap();
            assert!(solution.routes.len() <= 2);
            let mut customers: Vec<_> = solution.routes.iter().flatten().copied().collect();
            customers.sort();
            assert_eq!(customers, vec![1, 2, 3, 4]);
            close(
                solution.total_cost,
                solution
                    .routes
                    .iter()
                    .map(|r| p.evaluate_route(r, policy).unwrap().total_cost)
                    .sum(),
            );
            assert!(answer.lower_bound <= oracle.total_cost + 1e-7);
            assert!(solution.total_cost >= oracle.total_cost - 1e-7);
        }
    }
}

#[test]
fn continuation_dominance_and_walk_bounds_match_nonmetric_exhaustive_pricing() {
    let mut random = 17u64;
    for _ in 0..4 {
        let mut costs = vec![vec![0.; 6]; 6];
        for (i, row) in costs.iter_mut().enumerate() {
            for (j, value) in row.iter_mut().enumerate() {
                random = random.wrapping_mul(6364136223846793005).wrapping_add(1);
                if i != j {
                    *value = (random >> 32) as u32 as f64 % 31.;
                }
            }
        }
        let p = Problem::new(
            8,
            vec![Demand::new(vec![(0, 0.35), (4, 0.65)]).unwrap(); 5],
            costs,
        )
        .unwrap();
        for policy in [Policy::Classical, Policy::OptimalRestocking] {
            let mut routes = vec![];
            enumerate(&p, policy, &mut vec![], &mut routes);
            let duals = [13., -2., 27., 4., 16.];
            let expected = routes
                .iter()
                .map(|r| {
                    p.evaluate_route(r, policy).unwrap().total_cost
                        - r.iter().map(|&j| duals[j - 1]).sum::<f64>()
                        + 1.
                })
                .fold(f64::INFINITY, f64::min);
            for bidirectional in [false, true] {
                let answer = p
                    .price(
                        policy,
                        &duals,
                        -1.,
                        &[],
                        &[],
                        false,
                        PricingOptions {
                            bidirectional,
                            ..Default::default()
                        },
                    )
                    .unwrap();
                assert_eq!(answer.status, PricingStatus::Complete);
                close(answer.minimum_reduced_cost.unwrap(), expected);
                assert!(answer.lower_bound.unwrap() <= expected + 1e-7);
            }
        }
    }
}

#[test]
fn ng_scalar_bounds_cover_true_recourse_and_only_emit_elementary_columns() {
    let p = problem();
    let branches = [
        ArcBranch {
            from: 1,
            to: 2,
            required: false,
        },
        ArcBranch {
            from: 3,
            to: 4,
            required: true,
        },
    ];
    for policy in [Policy::Classical, Policy::OptimalRestocking] {
        let mut routes = vec![];
        enumerate(&p, policy, &mut vec![], &mut routes);
        for phase_one in [false, true] {
            let expected = routes
                .iter()
                .filter(|r| !uses(r, branches[0]))
                .map(|r| {
                    (if phase_one {
                        0.
                    } else {
                        p.evaluate_route(r, policy).unwrap().total_cost
                    }) - r.iter().map(|&j| [13., 8., 5., 2.][j - 1]).sum::<f64>()
                        + 1.
                        - if uses(r, branches[1]) { 3. } else { 0. }
                })
                .fold(f64::INFINITY, f64::min);
            let result = p
                .price_ng(
                    policy,
                    &[13., 8., 5., 2.],
                    -1.,
                    &branches,
                    &[0., 3.],
                    phase_one,
                    2,
                    PricingOptions::default(),
                )
                .unwrap();
            assert_eq!(result.status, PricingStatus::Complete);
            assert!(result.lower_bound.unwrap() <= expected + 1e-7);
            for route in result.routes {
                assert!(!uses(&route.customers, branches[0]));
                close(
                    route.evaluation.total_cost,
                    p.evaluate_route(&route.customers, policy)
                        .unwrap()
                        .total_cost,
                );
                assert!(route.reduced_cost < 0.);
            }
        }
    }
}

#[test]
fn ng_refines_repeated_customer_cycles_and_never_uses_partial_minima_as_bounds() {
    let p = Problem::new(
        1,
        vec![Demand::new(vec![(0, 1.)]).unwrap(); 4],
        (0..5)
            .map(|i| (0..5).map(|j| if i == j { 0. } else { 1. }).collect())
            .collect(),
    )
    .unwrap();
    let full = p
        .price_ng(
            Policy::Classical,
            &[100., 0., 0., 0.],
            0.,
            &[],
            &[],
            false,
            1,
            PricingOptions {
                max_columns: 1,
                ..Default::default()
            },
        )
        .unwrap();
    assert!(full.refinements > 0);
    assert!(full.discarded_walks > 0);
    assert!(!full.routes.is_empty());
    assert!(full.lower_bound.unwrap() <= -98. + 1e-7);
    for route in full.routes {
        p.evaluate_route(&route.customers, Policy::Classical)
            .unwrap();
    }
    for options in [
        PricingOptions {
            max_labels: 1,
            ..Default::default()
        },
        PricingOptions {
            time_limit: Duration::ZERO,
            ..Default::default()
        },
    ] {
        let partial = p
            .price_ng(
                Policy::Classical,
                &[100., 0., 0., 0.],
                0.,
                &[],
                &[],
                false,
                1,
                options,
            )
            .unwrap();
        assert_ne!(partial.status, PricingStatus::Complete);
        assert!(partial.lower_bound.is_none());
    }
}

#[cfg(feature = "clp")]
#[test]
fn ng_prepricing_master_and_tree_match_the_tiny_stochastic_oracle() {
    use baldes::extensions::stochastic::{
        ColumnGenerationOptions, SolveOptions, SolveStatus, TinyOptions,
    };
    let p = problem();
    for policy in [Policy::Classical, Policy::OptimalRestocking] {
        for vehicles in [1, 2, 4] {
            let oracle = p
                .solve_tiny(
                    policy,
                    TinyOptions {
                        max_vehicles: vehicles,
                        route_budget: 100000,
                    },
                )
                .unwrap()
                .unwrap();
            let answer = p
                .solve(
                    policy,
                    vehicles,
                    SolveOptions {
                        column_generation: ColumnGenerationOptions {
                            ng_size: 2,
                            ..Default::default()
                        },
                        ..Default::default()
                    },
                )
                .unwrap();
            assert_eq!(answer.status, SolveStatus::Optimal);
            close(answer.solution.unwrap().total_cost, oracle.total_cost);
            assert!(answer.lower_bound <= oracle.total_cost + 1e-6);
        }
    }
}

#[test]
fn ng_bound_includes_negative_nonmetric_recourse() {
    let p = Problem::new(
        3,
        vec![Demand::new(vec![(1, 1.)]).unwrap(); 3],
        vec![
            vec![0., 1., 1., 1.],
            vec![1., 0., 10., 3.],
            vec![1., 10., 0., 4.],
            vec![1., 3., 4., 0.],
        ],
    )
    .unwrap();
    let mut routes = vec![];
    enumerate(&p, Policy::OptimalRestocking, &mut vec![], &mut routes);
    let expected = routes
        .iter()
        .map(|r| {
            p.evaluate_route(r, Policy::OptimalRestocking)
                .unwrap()
                .total_cost
                - 10. * r.len() as f64
        })
        .fold(f64::INFINITY, f64::min);
    close(expected, -24.);
    let ng = p
        .price_ng(
            Policy::OptimalRestocking,
            &[10.; 3],
            0.,
            &[],
            &[],
            false,
            2,
            PricingOptions::default(),
        )
        .unwrap();
    assert!(ng.lower_bound.unwrap() <= expected + 1e-7);
    for r in ng.routes {
        close(
            r.evaluation.total_cost,
            p.evaluate_route(&r.customers, Policy::OptimalRestocking)
                .unwrap()
                .total_cost,
        );
    }
}

#[test]
fn bounded_ng_column_search_does_not_claim_a_relaxation_bound() {
    let n = 10;
    let p = Problem::new(
        10,
        vec![Demand::new(vec![(1, 1.)]).unwrap(); n],
        (0..=n)
            .map(|i| (0..=n).map(|j| if i == j { 0. } else { 1. }).collect())
            .collect(),
    )
    .unwrap();
    let ng = p
        .price_ng(
            Policy::Classical,
            &[10.; 10],
            0.,
            &[],
            &[],
            false,
            10,
            PricingOptions::default(),
        )
        .unwrap();
    assert_eq!(ng.status, PricingStatus::Heuristic);
    assert!(!ng.routes.is_empty());
    assert!(ng.lower_bound.is_none());
    for route in ng.routes {
        p.evaluate_route(&route.customers, Policy::Classical)
            .unwrap();
    }
}
