use super::*;
use crate::extensions::hgs::solution::{format_solution, parse_solution};
use crate::inputs::{Customer, Rounding};

fn instance() -> Instance {
    Instance {
        name: "decomposition".into(),
        vehicles: 4,
        capacity: 2.0,
        customers: [0.0, 1.0, 2.0, 9.0, 10.0]
            .iter()
            .enumerate()
            .map(|(i, &x)| Customer {
                x,
                y: 0.0,
                demand: if i == 0 { 0.0 } else { 1.0 },
                ready: 0.0,
                due: f64::INFINITY,
                service: 0.0,
            })
            .collect(),
        rounding: Rounding::None,
        distances: None,
    }
}

fn options() -> IntensifyOptions {
    IntensifyOptions {
        time_limit: 0.0,
        rounds: 8,
        max_routes: 4,
        max_customers: 4,
        subproblem: HgsOptions {
            time_limit: 0.0,
            nb_iter: 10,
            population_size: 3,
            seed: 7,
            verbose: false,
        },
        restart_after: 2,
        ..IntensifyOptions::default()
    }
}

#[test]
fn seeded_subproblem_search_improves_and_checkpoints_complete_plans() {
    let instance = instance();
    let seed = vec![vec![1, 3], vec![2, 4]];
    let mut checkpoints = Vec::new();
    let result = intensify_with_callback(&instance, &seed, &options(), |r, c| {
        assert_eq!(validate_plan(&instance, r)?, c);
        checkpoints.push(c);
        Ok(())
    })
    .unwrap();
    assert!(result.best_cost < result.initial_cost);
    assert_eq!(result.best_cost, 24.0);
    assert_eq!(result.rounds, 8);
    assert!(checkpoints.windows(2).all(|c| c[1] < c[0]));
    assert_eq!(
        validate_plan(&instance, &result.routes).unwrap(),
        result.best_cost
    );
    let again = intensify(&instance, &seed, &options()).unwrap();
    assert_eq!(result.routes, again.routes);
    assert_eq!(result.generations, again.generations);
}

#[test]
fn reduced_fleet_vrptw_search_keeps_windows_and_service_feasible() {
    let mut instance = instance();
    instance.capacity = 4.0;
    for c in &mut instance.customers {
        c.due = 100.0;
        c.service = 1.0;
    }
    let seed = vec![vec![1], vec![2], vec![3], vec![4]];
    let opts = IntensifyOptions {
        objective: Objective::VehiclesThenDistance,
        ..options()
    };
    let result = intensify(&instance, &seed, &opts).unwrap();
    assert!(result.routes.len() < seed.len());
    assert_eq!(
        validate_plan(&instance, &result.routes).unwrap(),
        result.best_cost
    );
}

#[test]
fn route_file_roundtrip_and_invalid_seeds_are_rejected() {
    let mut instance = instance();
    let seed = vec![vec![1, 2], vec![3, 4]];
    let cost = validate_plan(&instance, &seed).unwrap();
    assert_eq!(parse_solution(&format_solution(&seed, cost)).unwrap(), seed);
    assert!(parse_solution("<html>not a solution</html>").is_err());
    for bad in [
        vec![vec![1, 1, 2, 3, 4]],
        vec![vec![1, 2]],
        vec![vec![0, 1, 2, 3, 4]],
        vec![vec![1, 2, 3, 4]],
        vec![vec![1, 2, 3, 5]],
    ] {
        assert!(validate_plan(&instance, &bad).is_err());
        assert!(run_hgs_seeded(&instance, &options().subproblem, &[bad]).is_err());
    }
    instance.customers[4].due = 1.0;
    assert!(validate_plan(&instance, &seed).is_err());
    instance.customers[4].due = 100.0;
    instance.customers[0].due = 5.0;
    assert!(validate_plan(&instance, &seed).is_err());
}

#[test]
fn subproblem_preserves_asymmetric_matrix_and_customer_mapping() {
    let mut instance = instance();
    instance.distances = Some(
        (0..5)
            .map(|i| {
                (0..5)
                    .map(|j| if i == j { 0.0 } else { (10 * i + j) as f64 })
                    .collect()
            })
            .collect(),
    );
    let routes = vec![vec![4, 2], vec![1, 3]];
    let plan = Plan::new(routes.clone(), validate_plan(&instance, &routes).unwrap());
    let (sub, map, seed) = subproblem(&instance, &plan, &[1]);
    assert_eq!(map, vec![0, 1, 3]);
    assert_eq!(seed, vec![vec![1, 2]]);
    assert_eq!(sub.distance(1, 2, sub.rounding), 13.0);
    assert_eq!(sub.distance(2, 1, sub.rounding), 31.0);
    assert_eq!(validate_plan(&sub, &seed).unwrap(), 44.0);
}

#[test]
fn threshold_allows_distance_escape_but_never_more_vehicles() {
    let current = Plan::new(vec![vec![1, 2]], 100.0);
    let worse = Plan::new(vec![vec![2, 1]], 100.05);
    assert!(accepts(&worse, &current, Objective::Distance, 0.001));
    assert!(!accepts(&worse, &current, Objective::Distance, 0.0));
    let more_routes = Plan::new(vec![vec![1], vec![2]], 90.0);
    assert!(!accepts(
        &more_routes,
        &current,
        Objective::VehiclesThenDistance,
        1.0
    ));
    assert!(better(
        &current,
        &more_routes,
        Objective::VehiclesThenDistance
    ));
}

#[test]
fn elite_pool_ignores_route_order_and_keeps_diverse_candidates() {
    let first = Plan::new(vec![vec![1, 2], vec![3, 4]], 24.0);
    let permuted = Plan::new(vec![vec![3, 4], vec![1, 2]], 24.0);
    let different = Plan::new(vec![vec![1, 3], vec![2, 4]], 38.0);
    let mut pool = vec![first];
    add_elite(&mut pool, &permuted, &options());
    assert_eq!(pool.len(), 1);
    add_elite(&mut pool, &different, &options());
    assert_eq!(pool.len(), 2);
    assert_eq!(pool[0].cost, 24.0);
}

#[test]
fn neighborhood_size_and_time_limits_preserve_seed() {
    let instance = instance();
    let seed = vec![vec![1, 3], vec![2, 4]];
    let opts = IntensifyOptions {
        time_limit: 1e-12,
        ..options()
    };
    let result = intensify(&instance, &seed, &opts).unwrap();
    assert_eq!(result.routes, seed);
    assert_eq!(result.rounds, 0);
    let plan = Plan::new(seed.clone(), result.initial_cost);
    let opts = IntensifyOptions {
        max_customers: 2,
        ..options()
    };
    let mut rng = Xoroshiro128Plus::new(1);
    let selected = select_routes(&instance, &plan, &opts, 0, 20, &mut rng, &BTreeSet::new());
    assert!(
        selected
            .iter()
            .map(|&i| plan.routes[i].len())
            .sum::<usize>()
            <= 2
    );
    let result = run_hgs_seeded(
        &instance,
        &HgsOptions {
            time_limit: 1e-12,
            ..options().subproblem
        },
        &[seed],
    )
    .unwrap();
    assert_eq!(result.best_cost, Some(38.0));
}

#[test]
fn invalid_options_and_failed_checkpoints_stop_search() {
    let instance = instance();
    let seed = vec![vec![1, 2], vec![3, 4]];
    let opts = IntensifyOptions {
        threshold: f64::NAN,
        ..options()
    };
    assert!(intensify(&instance, &seed, &opts).is_err());
    assert_eq!(
        intensify_with_callback(&instance, &seed, &options(), |_, _| Err("disk full".into()))
            .unwrap_err(),
        "disk full"
    );
}

fn independent_jobs() -> Vec<Job> {
    let instance = instance();
    let routes = vec![vec![1], vec![2], vec![3], vec![4]];
    let plan = Plan::new(routes.clone(), validate_plan(&instance, &routes).unwrap());
    [vec![0, 1], vec![2, 3]]
        .into_iter()
        .map(|selected| {
            let (sub, map, seed) = subproblem(&instance, &plan, &selected);
            Job {
                selected,
                instance: sub,
                map,
                seed: Some(seed),
                options: options().subproblem,
                deadline: None,
            }
        })
        .collect()
}

#[test]
fn parallel_jobs_match_serial_results_in_input_order() {
    let serial = solve_jobs(independent_jobs(), None).unwrap();
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(2)
        .build()
        .unwrap();
    let parallel = solve_jobs(independent_jobs(), Some(&pool)).unwrap();
    assert_eq!(serial, parallel);
    assert_eq!(parallel[0].selected, vec![0, 1]);
    assert_eq!(parallel[1].selected, vec![2, 3]);
    let routes: Vec<_> = parallel
        .into_iter()
        .flat_map(|r| r.routes.unwrap())
        .collect();
    assert!(validate_plan(&instance(), &routes).is_ok());
}

#[test]
fn fixed_batches_produce_identical_search_across_worker_counts() {
    let instance = instance();
    let seed = vec![vec![1], vec![2], vec![3], vec![4]];
    let serial_opts = IntensifyOptions {
        batch_size: 2,
        ..options()
    };
    let parallel_opts = IntensifyOptions {
        workers: 2,
        ..serial_opts.clone()
    };
    let serial = intensify(&instance, &seed, &serial_opts).unwrap();
    let parallel = intensify(&instance, &seed, &parallel_opts).unwrap();
    assert_eq!(serial.routes, parallel.routes);
    assert_eq!(serial.best_cost, parallel.best_cost);
    assert_eq!(serial.generations, parallel.generations);
    assert_eq!(serial.rounds, parallel.rounds);
    assert_eq!(serial.batches, parallel.batches);
    assert!(parallel.batches < parallel.rounds);
    assert_eq!(
        validate_plan(&instance, &parallel.routes).unwrap(),
        parallel.best_cost
    );
}

#[test]
fn parallel_selection_excludes_already_assigned_routes() {
    let instance = instance();
    let routes = vec![vec![1], vec![2], vec![3], vec![4]];
    let plan = Plan::new(routes.clone(), validate_plan(&instance, &routes).unwrap());
    let mut rng = Xoroshiro128Plus::new(2);
    let first = select_routes(
        &instance,
        &plan,
        &options(),
        0,
        0,
        &mut rng,
        &BTreeSet::new(),
    );
    let excluded: BTreeSet<_> = first.iter().copied().collect();
    let second = select_routes(&instance, &plan, &options(), 1, 0, &mut rng, &excluded);
    assert!(!first.is_empty() && !second.is_empty());
    assert!(second.iter().all(|i| !excluded.contains(i)));
}

#[test]
fn queued_jobs_respect_deadlines_and_zero_workers_are_rejected() {
    let mut jobs = independent_jobs();
    for job in &mut jobs {
        job.deadline = Some(Instant::now());
    }
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(2)
        .build()
        .unwrap();
    for result in solve_jobs(jobs, Some(&pool)).unwrap() {
        assert!(result.routes.is_none());
        assert_eq!(result.generations, 0);
    }
    let opts = IntensifyOptions {
        workers: 0,
        ..options()
    };
    assert!(intensify(&instance(), &[vec![1, 2], vec![3, 4]], &opts).is_err());
}
