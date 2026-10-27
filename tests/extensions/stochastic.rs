use baldes::extensions::stochastic::{Demand, Policy, Problem, TinyOptions};

fn demand(values: &[(usize, f64)]) -> Demand {
    Demand::new(values.to_vec()).unwrap()
}
fn fixture() -> Problem {
    Problem::new(
        10,
        vec![demand(&[(5, 0.5), (10, 0.5)]), demand(&[(2, 1.0)])],
        vec![vec![0., 1., 10.], vec![1., 0., 10.], vec![10., 10., 0.]],
    )
    .unwrap()
}
fn close(a: f64, b: f64) {
    assert!((a - b).abs() < 1e-10, "{a} != {b}");
}

#[test]
fn empty_after_service_replenishes_before_next_customer() {
    let p = fixture();
    let result = p.evaluate_route(&[1, 2], Policy::Classical).unwrap();
    close(result.expected_demand, 9.5);
    close(result.travel_cost, 21.0);
    // Half the time customer 1 empties the truck: detour costs 1+10-10.
    close(result.expected_recourse_cost, 0.5);
    close(result.total_cost, 21.5);
    close(
        p.evaluate_route(&[1], Policy::Classical)
            .unwrap()
            .expected_recourse_cost,
        0.0,
    );
}

#[test]
fn preventive_restocking_avoids_expensive_failure() {
    let p = Problem::new(
        10,
        vec![demand(&[(4, 0.5), (8, 0.5)]), demand(&[(4, 1.)])],
        vec![vec![0., 1., 10.], vec![1., 0., 10.], vec![10., 10., 0.]],
    )
    .unwrap();
    close(
        p.evaluate_route(&[1, 2], Policy::Classical)
            .unwrap()
            .expected_recourse_cost,
        10.0,
    );
    close(
        p.evaluate_route(&[1, 2], Policy::OptimalRestocking)
            .unwrap()
            .expected_recourse_cost,
        0.5,
    );
}

// Independent scenario tree: enumerate demand realizations and decisions along
// the route, rather than computing a backward vector over all load states.
fn scenario(p: &Problem, route: &[usize], t: usize, q: usize, policy: Policy) -> f64 {
    if t == route.len() {
        return 0.;
    }
    let i = if t == 0 { 0 } else { route[t - 1] };
    let j = route[t];
    let expected = |load: usize| -> f64 {
        p.demands()[j - 1]
            .outcomes()
            .iter()
            .map(|&(d, prob)| {
                let (remaining, failure) = if d > load {
                    (p.capacity() - (d - load), p.costs()[j][0] + p.costs()[0][j])
                } else {
                    (load - d, 0.)
                };
                prob * (failure + scenario(p, route, t + 1, remaining, policy))
            })
            .sum()
    };
    let refill = p.costs()[i][0] + p.costs()[0][j] - p.costs()[i][j] + expected(p.capacity());
    match policy {
        Policy::Classical if q == 0 => refill,
        Policy::Classical => expected(q),
        Policy::OptimalRestocking => refill.min(expected(q)),
    }
}

#[test]
fn backward_evaluator_matches_scenario_tree_for_both_orders_and_policies() {
    let p = fixture();
    for route in [vec![], vec![1], vec![2], vec![1, 2], vec![2, 1]] {
        for policy in [Policy::Classical, Policy::OptimalRestocking] {
            close(
                p.evaluate_route(&route, policy)
                    .unwrap()
                    .expected_recourse_cost,
                scenario(&p, &route, 0, p.capacity(), policy),
            );
        }
    }
}

#[test]
fn tiny_solver_optimizes_order_and_honors_vehicle_limit() {
    let p = fixture();
    let solution = p
        .solve_tiny(
            Policy::Classical,
            TinyOptions {
                max_vehicles: 1,
                route_budget: 10,
            },
        )
        .unwrap()
        .unwrap();
    assert_eq!(solution.routes, vec![vec![1, 2]]);
    close(solution.total_cost, 21.5);
    assert_eq!(solution.enumerated_routes, 4);
    assert!(p
        .solve_tiny(
            Policy::Classical,
            TinyOptions {
                max_vehicles: 0,
                route_budget: 10
            }
        )
        .unwrap()
        .is_none());
    assert!(p
        .solve_tiny(
            Policy::Classical,
            TinyOptions {
                max_vehicles: 1,
                route_budget: 1
            }
        )
        .is_err());
}

#[test]
fn mean_capacity_requires_multiple_routes_and_zero_demand_is_served() {
    let p = Problem::new(
        5,
        vec![demand(&[(5, 1.)]), demand(&[(5, 1.)]), demand(&[(0, 1.)])],
        vec![
            vec![0., 1., 1., 1.],
            vec![1., 0., 1., 1.],
            vec![1., 1., 0., 1.],
            vec![1., 1., 1., 0.],
        ],
    )
    .unwrap();
    assert!(p.evaluate_route(&[1, 2], Policy::Classical).is_err());
    assert!(p
        .solve_tiny(
            Policy::OptimalRestocking,
            TinyOptions {
                max_vehicles: 1,
                route_budget: 100
            }
        )
        .unwrap()
        .is_none());
    let solution = p
        .solve_tiny(
            Policy::OptimalRestocking,
            TinyOptions {
                max_vehicles: 2,
                route_budget: 100,
            },
        )
        .unwrap()
        .unwrap();
    close(solution.total_cost, 5.);
    assert_eq!(solution.routes.iter().map(Vec::len).sum::<usize>(), 3);
}

#[test]
fn invalid_data_and_routes_are_rejected() {
    assert!(Demand::new(vec![]).is_err());
    assert!(Demand::new(vec![(1, f64::NAN)]).is_err());
    assert!(Demand::new(vec![(1, -1.), (2, 2.)]).is_err());
    assert!(Demand::new(vec![(1, 0.2)]).is_err());
    assert!(Demand::new(vec![(1, 0.5), (1, 0.5)]).is_err());
    assert!(Problem::new(
        1,
        vec![demand(&[(2, 1.)])],
        vec![vec![0., 1.], vec![1., 0.]]
    )
    .is_err());
    assert!(Problem::new(1, vec![], vec![vec![]]).is_err());
    let p = fixture();
    for route in [vec![0], vec![3], vec![1, 1]] {
        assert!(p.evaluate_route(&route, Policy::Classical).is_err());
    }
    close(
        p.evaluate_route(&[], Policy::Classical).unwrap().total_cost,
        0.,
    );
}

#[test]
fn asymmetric_multistop_costs_match_scenarios_including_zero_demands() {
    let demands = (2..=5)
        .map(|d| demand(&[(0, 0.65), (d, 0.2), (10, 0.15)]))
        .collect();
    // Shortest-path distances on a directed cycle, hence asymmetric and metric.
    let costs = (0..5)
        .map(|i| (0..5).map(|j| ((j + 5 - i) % 5) as f64).collect())
        .collect();
    let p = Problem::new(10, demands, costs).unwrap();
    for a in 1..=4 {
        for b in 1..=4 {
            for c in 1..=4 {
                for d in 1..=4 {
                    let route = [a, b, c, d];
                    if (0..4).any(|i| (i + 1..4).any(|j| route[i] == route[j])) {
                        continue;
                    }
                    let mut previous = f64::INFINITY;
                    for policy in [Policy::Classical, Policy::OptimalRestocking] {
                        let evaluated = p.evaluate_route(&route, policy).unwrap();
                        close(
                            evaluated.expected_recourse_cost,
                            scenario(&p, &route, 0, 10, policy),
                        );
                        assert!(evaluated.expected_recourse_cost <= previous + 1e-10);
                        previous = evaluated.expected_recourse_cost;
                    }
                }
            }
        }
    }
}

#[test]
fn tiny_optimum_changes_order_with_recourse_policy() {
    let p = Problem::new(
        10,
        vec![demand(&[(4, 0.5), (8, 0.5)]), demand(&[(4, 1.)])],
        vec![vec![0., 1., 10.], vec![1., 0., 10.], vec![10., 10., 0.]],
    )
    .unwrap();
    let options = TinyOptions {
        max_vehicles: 2,
        route_budget: 100,
    };
    let classical = p.solve_tiny(Policy::Classical, options).unwrap().unwrap();
    close(classical.total_cost, 22.);
    let optimal = p
        .solve_tiny(Policy::OptimalRestocking, options)
        .unwrap()
        .unwrap();
    close(optimal.total_cost, 21.5);
    assert_eq!(optimal.routes, vec![vec![1, 2]]);
}

#[test]
fn empty_problem_and_customer_limit_have_explicit_results() {
    let p = Problem::new(1, vec![], vec![vec![0.]]).unwrap();
    let options = TinyOptions {
        max_vehicles: 0,
        route_budget: 0,
    };
    let result = p.solve_tiny(Policy::Classical, options).unwrap().unwrap();
    assert!(result.routes.is_empty());
    close(result.total_cost, 0.);
    let p = Problem::new(1, vec![demand(&[(0, 1.)]); 10], vec![vec![0.; 11]; 11]).unwrap();
    assert!(p.solve_tiny(Policy::Classical, options).is_err());
}

#[test]
fn partition_overflow_is_an_error_rather_than_fleet_infeasibility() {
    let p = Problem::new(
        1,
        vec![demand(&[(1, 1.)]); 2],
        vec![
            vec![0., 6e307, 6e307],
            vec![6e307, 0., 6e307],
            vec![6e307, 6e307, 0.],
        ],
    )
    .unwrap();
    assert!(p
        .solve_tiny(
            Policy::Classical,
            TinyOptions {
                max_vehicles: 2,
                route_budget: 10
            }
        )
        .is_err());
}
