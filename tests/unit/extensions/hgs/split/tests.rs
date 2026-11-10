use super::*;
use crate::extensions::hgs::Params;
use crate::inputs::{Customer, Instance};

fn make_test_instance() -> Instance {
    Instance {
        name: "test-5".to_string(),
        vehicles: 3,
        capacity: 100.0,
        customers: vec![
            Customer {
                x: 0.0,
                y: 0.0,
                demand: 0.0,
                ready: 0.0,
                due: i32::MAX as f64,
                service: 0.0,
            },
            Customer {
                x: 10.0,
                y: 0.0,
                demand: 10.0,
                ready: 0.0,
                due: 1000.0,
                service: 5.0,
            },
            Customer {
                x: 20.0,
                y: 0.0,
                demand: 20.0,
                ready: 0.0,
                due: 1000.0,
                service: 10.0,
            },
            Customer {
                x: 30.0,
                y: 0.0,
                demand: 15.0,
                ready: 0.0,
                due: 1000.0,
                service: 5.0,
            },
            Customer {
                x: 40.0,
                y: 0.0,
                demand: 25.0,
                ready: 0.0,
                due: 1000.0,
                service: 10.0,
            },
        ],
        rounding: crate::inputs::Rounding::None,
        distances: None,
    }
}

#[test]
fn test_split_basic() {
    let instance = make_test_instance();
    let params = Params::from_instance(&instance);
    let mut indiv = Individual::new_random(&params);

    let mut split = Split::new(&params);
    split.general_split(&mut indiv, 3);

    // After split, routes should be populated.
    assert!(indiv.chrom_r.iter().any(|r| !r.is_empty()));
}

#[test]
fn test_split_preserves_all_customers() {
    let instance = make_test_instance();
    let params = Params::from_instance(&instance);
    let mut indiv = Individual::new_random(&params);

    let mut split = Split::new(&params);
    split.general_split(&mut indiv, 3);

    // All customers should appear exactly once across all routes.
    let mut all_customers: Vec<usize> = Vec::new();
    for route in &indiv.chrom_r {
        all_customers.extend(route);
    }
    all_customers.sort();
    assert_eq!(all_customers, vec![1, 2, 3, 4]);
}

#[test]
fn test_trivial_deque() {
    let mut dq = TrivialDeque::new(10, 0);
    assert_eq!(dq.size(), 1);
    assert_eq!(dq.get_front(), 0);

    dq.push_back(1);
    dq.push_back(2);
    assert_eq!(dq.size(), 3);
    assert_eq!(dq.get_front(), 0);
    assert_eq!(dq.get_back(), 2);

    dq.pop_front();
    assert_eq!(dq.size(), 2);
    assert_eq!(dq.get_front(), 1);

    dq.pop_back();
    assert_eq!(dq.size(), 1);
    assert_eq!(dq.get_back(), 1);
}

/// Enumerate every contiguous partition and score it through the independent
/// forward route evaluator, rather than Split's sequence concatenation.
fn assert_matches_partitions(params: &Params, tour: &[usize], fleet: usize) {
    let mut individual = Individual::new_empty(params);
    individual.chrom_t = tour.to_vec();
    let mut split = Split::new(params);
    split.general_split(&mut individual, fleet);
    let cap = split.max_vehicles.min(tour.len());
    let mut expected = f64::INFINITY;
    let mut candidate = Individual::new_empty(params);
    for mask in 0usize..1 << (tour.len() - 1) {
        if mask.count_ones() as usize >= cap {
            continue;
        }
        for route in &mut candidate.chrom_r {
            route.clear();
        }
        let mut route = 0;
        for (i, &client) in tour.iter().enumerate() {
            candidate.chrom_r[route].push(client);
            if mask & (1 << i) != 0 {
                route += 1;
            }
        }
        candidate.evaluate_complete_cost();
        expected = expected.min(candidate.my_cost_sol.penalized_cost);
    }
    assert!(
        (individual.my_cost_sol.penalized_cost - expected).abs() < 1e-7,
        "tour={tour:?}, fleet={fleet}: got {}, expected {expected}",
        individual.my_cost_sol.penalized_cost
    );
    let flattened: Vec<_> = individual.chrom_r.iter().flatten().copied().collect();
    assert_eq!(flattened, tour);
    assert!(individual.my_cost_sol.nb_routes <= cap);
}

#[test]
fn time_windows_change_the_optimal_route_boundaries_and_refresh_penalties() {
    let mut params = Params::from_instance(&make_test_instance());
    params.vehicle_capacity = 1000;
    // Customer 1's late opening makes 1 -> 2 infeasible, but both singleton
    // routes are feasible. Keep fractional values throughout the scoring.
    params.cli[1].earliest_arrival = 400.25;
    params.cli[1].latest_arrival = 500.75;
    params.cli[2].latest_arrival = 250.5;
    let mut individual = Individual::new_empty(&params);
    individual.chrom_t = vec![1, 2, 3, 4];
    let mut split = Split::new(&params);
    params.penalty_time_warp.set(0.);
    split.general_split(&mut individual, 3);
    assert_eq!(individual.my_cost_sol.nb_routes, 1);
    params.penalty_time_warp.set(10.25);
    split.general_split(&mut individual, 3);
    assert_eq!(individual.my_cost_sol.nb_routes, 2);
    assert_eq!(individual.my_cost_sol.time_warp, 0.);
    assert_matches_partitions(&params, &individual.chrom_t, 3);
    params.penalty_time_warp.set(0.);
    split.general_split(&mut individual, 3);
    assert_eq!(individual.my_cost_sol.nb_routes, 1);
    assert!(individual.chrom_r[1..].iter().all(Vec::is_empty));
}

#[test]
fn time_window_split_matches_exhaustive_partitions_with_fractional_asymmetric_costs() {
    let mut params = Params::from_instance(&make_test_instance());
    for i in 0..5 {
        for j in 0..5 {
            params.time_cost.set(
                i,
                j,
                if i == j {
                    0.
                } else {
                    ((i * 17 + j * 11) % 23) as f64 + 0.375
                },
            );
        }
    }
    params.cli[0].latest_arrival = 85.125;
    params.penalty_capacity.set(2.75);
    params.penalty_time_warp.set(3.125);
    for case in 0..12 {
        for i in 1..5 {
            params.cli[i].earliest_arrival = ((case * 19 + i * 7) % 60) as f64 + 0.125;
            params.cli[i].latest_arrival = params.cli[i].earliest_arrival + 8.75;
            params.cli[i].service_duration = i as f64 + 0.625;
            params.cli[i].release_time = ((case * i * 3) % 17) as f64 + 0.25;
        }
        for a in 1..5 {
            for b in 1..5 {
                if b == a {
                    continue;
                }
                for c in 1..5 {
                    if c == a || c == b {
                        continue;
                    }
                    let d = 10 - a - b - c;
                    for fleet in 1..=3 {
                        assert_matches_partitions(&params, &[a, b, c, d], fleet);
                    }
                }
            }
        }
    }
}

#[test]
fn time_window_split_scores_multiple_windows_and_waiting_like_route_evaluation() {
    use crate::extensions::hgs::params::TimeWindow;
    let mut params = Params::from_instance(&make_test_instance());
    params.cli[2].n_tw = 2;
    params.cli[2].time_windows = vec![TimeWindow::new(0., 180.25), TimeWindow::new(700.5, 800.75)];
    params.cli[1].earliest_arrival = 400.125;
    params.cli[3].release_time = 150.75;
    params.penalty_wait_time = 0.375;
    for fleet in 1..=3 {
        assert_matches_partitions(&params, &[1, 2, 3, 4], fleet);
        assert_matches_partitions(&params, &[4, 3, 2, 1], fleet);
    }
    params.cli[2].n_tw = 0;
    for fleet in 1..=3 {
        assert_matches_partitions(&params, &[1, 2, 3, 4], fleet);
    }
}

#[test]
fn time_window_split_handles_a_single_customer_and_late_depot_return() {
    let mut instance = make_test_instance();
    instance.customers.truncate(2);
    instance.customers[0].due = 15.125;
    instance.customers[1].due = 50.;
    let params = Params::from_instance(&instance);
    assert_matches_partitions(&params, &[1], 1);
}

#[test]
fn capacity_only_split_still_matches_exhaustive_partitions() {
    let mut instance = make_test_instance();
    for client in &mut instance.customers {
        client.due = f64::INFINITY;
    }
    let params = Params::from_instance(&instance);
    assert!(!params.is_time_window_constrained);
    for fleet in 1..=3 {
        assert_matches_partitions(&params, &[1, 3, 2, 4], fleet);
    }
}

#[test]
fn time_window_split_keeps_overloaded_routes_when_the_fleet_is_tight() {
    let mut params = Params::from_instance(&make_test_instance());
    // The existing bin-packing bound is capped by the available fleet.
    params.vehicle_capacity = 10;
    for fleet in 1..=3 {
        assert_matches_partitions(&params, &[1, 3, 2, 4], fleet);
    }
}

/// Check every prefix and fleet layer, so an incorrect predecessor cannot hide
/// behind a final partition that happens to have the same objective.
fn assert_queue_matches_exhaustive(params: &Params, tour: &[usize]) {
    let mut individual = Individual::new_empty(params);
    individual.chrom_t = tour.to_vec();
    let mut queue = Split::new(params);
    let mut exhaustive = Split::new(params);
    assert!(queue.time_windows.prepare(params, tour));
    for fleet in [None, Some(1), Some(2), Some(3)] {
        queue.time_window_potentials(&individual, fleet, true);
        exhaustive.time_window_potentials(&individual, fleet, false);
        for layer in 0..=fleet.unwrap_or(0) {
            for end in 0..=tour.len() {
                let actual = queue.potential[layer][end];
                let expected = exhaustive.potential[layer][end];
                assert!(
                    actual == expected || (actual - expected).abs() <= 1e-8 * expected.abs().max(1.),
                    "layer={layer}, end={end}, fleet={fleet:?}, tour={tour:?}: {actual} != {expected}"
                );
            }
        }
        assert!(queue.time_windows.operations <= 20 * tour.len() * fleet.unwrap_or(1));
    }
    for fleet in 1..=3 {
        assert_matches_partitions(params, tour, fleet);
    }
}

#[test]
fn predecessor_queue_matches_all_prefixes_and_fleet_layers_on_fractional_metric_instances() {
    let mut state = 7_u64;
    let mut next = || {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        (state >> 32) as usize
    };
    for _case in 0..500 {
        let mut instance = make_test_instance();
        instance.customers.truncate(1);
        instance.customers[0].x = 0.;
        instance.customers[0].y = 0.;
        instance.customers[0].due = 90.;
        instance.capacity = (10 + next() % 30) as f64;
        for _ in 0..6 {
            let x = (next() % 1000) as f64 / 100.;
            let y = (next() % 1000) as f64 / 100.;
            let ready = x.hypot(y) + (next() % 4500) as f64 / 100.;
            instance.customers.push(Customer {
                x,
                y,
                demand: (next() % 9) as f64,
                ready,
                due: ready + (next() % 4000) as f64 / 100.,
                service: (next() % 300) as f64 / 100.,
            });
        }
        let params = Params::from_instance(&instance);
        params.penalty_capacity.set((next() % 100) as f64 / 10.);
        params.penalty_time_warp.set((next() % 100) as f64 / 10.);
        let mut tour: Vec<_> = (1..=6).collect();
        for i in (1..tour.len()).rev() {
            tour.swap(i, next() % (i + 1));
        }
        assert_queue_matches_exhaustive(&params, &tour);
    }
}

#[test]
fn predecessor_queue_rechecks_penalized_feasible_border_without_cursor_movement() {
    // A penalized candidate's score can overtake the first feasible candidate
    // while the capacity/time-warp frontiers remain at the same positions.
    let data = [
        (0., 0., 0., 90., 0., 0.),
        (3.04, 7.56, 31.488324981246144, 38.988324981246144, 2.06, 3.),
        (9.97, 6.24, 47.92173881702871, 52.50173881702871, 1.2, 6.),
        (1.18, 5.8, 33.52881744945728, 66.44881744945728, 2.94, 1.),
        (9.14, 4.67, 50.593941737948434, 81.35394173794843, 2.73, 2.),
        (8.68, 6.73, 11.503410217232169, 45.213410217232166, 1.08, 0.),
        (3.66, 5.68, 47.01707037110018, 55.77707037110018, 2.27, 6.),
        (9.73, 4.75, 45.297529727504795, 49.2875297275048, 1.01, 8.),
        (4.16, 0.41, 39.35015549949999, 49.05015549949999, 0.51, 1.),
        (4.26, 7.92, 16.232997275658434, 34.78299727565843, 2.45, 1.),
        (7.09, 0.96, 42.51469775741785, 63.79469775741785, 0.61, 0.),
        (1.71, 9.46, 22.353308483555494, 28.783308483555494, 1.03, 7.),
    ];
    let mut instance = make_test_instance();
    instance.capacity = 14.;
    instance.customers = data
        .into_iter()
        .map(|(x, y, ready, due, service, demand)| Customer {
            x,
            y,
            ready,
            due,
            service,
            demand,
        })
        .collect();
    let params = Params::from_instance(&instance);
    params.penalty_capacity.set(9.4);
    params.penalty_time_warp.set(8.3);
    assert_queue_matches_exhaustive(&params, &[3, 6, 9, 8, 7, 10, 2, 5, 11, 1, 4]);
}

#[test]
fn predecessor_queue_retains_exact_fallback_for_nonmetric_and_extended_schedules() {
    let mut params = Params::from_instance(&make_test_instance());
    let tour = [1, 2, 3, 4];
    let mut queue = time_windows::TimeWindowSplit::new(4);
    assert!(queue.prepare(&params, &tour));
    params.cli[1].release_time = 1.;
    assert!(!queue.prepare(&params, &tour));
    assert_matches_partitions(&params, &tour, 3);
    params.cli[1].release_time = 0.;
    params.time_cost.set(0, 2, 1000.);
    assert!(!queue.prepare(&params, &tour));
    assert_matches_partitions(&params, &tour, 3);
}

#[test]
#[ignore = "manual Split benchmark; run with --release --ignored --nocapture"]
fn benchmark_predecessor_queue_against_exhaustive_split() {
    use std::time::Instant;
    for name in ["Solomon/R201", "Homberger/C1_10_1"] {
        let instance = Instance::read(format!(
            "{}/instances/{name}.txt",
            env!("CARGO_MANIFEST_DIR")
        ))
        .unwrap();
        let params = Params::from_instance(&instance);
        let mut individual = Individual::new_random(&params);
        // Use a fixed permutation, independent of the genetic search path.
        individual.chrom_t = (1..=params.nb_clients as usize).collect();
        let mut split = Split::new(&params);
        assert!(split.time_windows.prepare(&params, &individual.chrom_t));
        let repeats = if params.nb_clients < 1000 { 1000 } else { 100 };
        for fleet in [None, Some(10)] {
            for optimized in [false, true] {
                let start = Instant::now();
                for _ in 0..repeats {
                    split.time_window_potentials(
                        std::hint::black_box(&individual),
                        fleet,
                        optimized,
                    );
                    std::hint::black_box(&split.potential);
                }
                eprintln!(
                    "{name},fleet={fleet:?},optimized={optimized},repeats={repeats},seconds={:.6}",
                    start.elapsed().as_secs_f64()
                );
            }
        }
    }
}
