use super::*;
use crate::inputs::{Customer, Rounding};
fn instance() -> Instance {
    Instance {
        name: "route-groups".into(),
        vehicles: 4,
        capacity: 3.0,
        customers: [0.0, 1.0, 2.0, 9.0, 10.0, 20.0, 21.0]
            .iter()
            .enumerate()
            .map(|(i, &x)| Customer {
                x,
                y: 0.0,
                demand: if i == 0 { 0.0 } else { 1.0 },
                ready: 0.0,
                due: 1000.0,
                service: 0.0,
            })
            .collect(),
        rounding: Rounding::None,
        distances: None,
    }
}
fn options() -> RouteDecompositionOptions {
    RouteDecompositionOptions {
        time_limit: 0.08,
        target_customers: 4,
        phases: 2,
        hgs: HgsOptions {
            nb_iter: 5,
            population_size: 3,
            ..Default::default()
        },
        ..Default::default()
    }
}
#[test]
fn grouping_keeps_whole_routes_and_regroups_reproducibly() {
    let data = instance();
    let routes = vec![vec![1, 2], vec![3, 4], vec![5, 6]];
    let first = route_groups(&data, &routes, 4, 0, 7);
    assert_eq!(first, route_groups(&data, &routes, 4, 0, 7));
    let mut ids: Vec<_> = first.iter().flatten().copied().collect();
    ids.sort_unstable();
    assert_eq!(ids, vec![0, 1, 2]);
    assert!(first
        .iter()
        .all(|g| g.iter().map(|&r| routes[r].len()).sum::<usize>() <= 4));
    assert!((1..10).any(|phase| route_groups(&data, &routes, 4, phase, 7) != first));
    let oversized = route_groups(&data, &routes, 1, 0, 7);
    assert_eq!(oversized.len(), 3);
    assert!(oversized.iter().all(|g| g.len() == 1));
}
#[test]
fn projection_preserves_directed_fractional_arcs_and_tight_windows() {
    let mut data = instance();
    let n = data.customers.len();
    let mut matrix = vec![vec![90.125; n]; n];
    for (i, row) in matrix.iter_mut().enumerate() {
        row[i] = 0.0;
    }
    matrix[0][5] = 1.125;
    matrix[5][2] = 1.25;
    matrix[2][0] = 1.375;
    data.distances = Some(matrix);
    data.customers[5].due = 1.125;
    data.customers[2].due = 2.375;
    let original = vec![vec![5, 2]];
    let (sub, map, projected) = subproblem(&data, &original, &[5, 2], 1);
    assert_eq!(map, vec![0, 5, 2]);
    assert_eq!(projected, vec![vec![1, 2]]);
    assert_eq!(sub.vehicles, 1);
    assert_eq!(validate_plan(&sub, &projected).unwrap(), 3.75);
    assert_eq!(sub.distance(1, 2, sub.rounding), 1.25);
    assert_eq!(sub.distance(2, 1, sub.rounding), 90.125);
    assert_eq!(sub.customers[1].due, 1.125);
}
#[test]
fn progressive_rebuild_retains_all_unsolved_routes() {
    let base = vec![vec![1, 4], vec![2, 3], vec![5, 6]];
    let groups = vec![vec![0, 1], vec![2]];
    let replacement = vec![Some(vec![vec![1, 2], vec![3, 4]]), None];
    assert_eq!(
        rebuild(&base, &groups, &replacement),
        vec![vec![1, 2], vec![3, 4], vec![5, 6]]
    );
}
#[test]
fn expired_budget_returns_exact_seed_without_launching_jobs() {
    let data = instance();
    let seed = vec![vec![1, 2], vec![3, 4], vec![5, 6]];
    let result = run_hgs_decomposed(
        &data,
        Some(&seed),
        &RouteDecompositionOptions {
            time_limit: f64::MIN_POSITIVE,
            ..options()
        },
    )
    .unwrap();
    assert_eq!(result.solution.routes, seed);
    assert_eq!(result.decomposition.subproblems, 0);
    assert_eq!(
        result.solution.best_cost,
        Some(validate_plan(&data, &seed).unwrap())
    );
}
#[test]
fn seeded_phases_preserve_feasibility_and_never_worsen_the_incumbent() {
    let mut data = instance();
    data.vehicles = 3;
    let seed = vec![vec![1, 4], vec![2, 3], vec![5, 6]];
    let result = run_hgs_decomposed(&data, Some(&seed), &options()).unwrap();
    assert!(result.decomposition.subproblems > 0);
    assert!(result.decomposition.phases_completed > 0);
    let cost = validate_plan(&data, &result.solution.routes).unwrap();
    assert!(cost <= validate_plan(&data, &seed).unwrap());
    assert_eq!(result.solution.best_cost, Some(cost));
    assert!(result.solution.routes.len() <= 3);
}
#[test]
fn invalid_options_inputs_and_seeds_return_errors() {
    let mut data = instance();
    let seed = vec![vec![1, 2], vec![3, 4], vec![5, 6]];
    assert!(run_hgs_decomposed(
        &data,
        Some(&seed),
        &RouteDecompositionOptions {
            phases: 0,
            ..options()
        }
    )
    .is_err());
    assert!(run_hgs_decomposed(
        &data,
        Some(&seed),
        &RouteDecompositionOptions {
            time_limit: f64::NAN,
            ..options()
        }
    )
    .is_err());
    assert!(run_hgs_decomposed(&data, Some(&[vec![1, 1]]), &options()).is_err());
    data.distances = Some(vec![vec![0.0]]);
    assert!(run_hgs_decomposed(&data, Some(&seed), &options()).is_err());
    data.distances = None;
    data.customers[1].demand = 0.5;
    assert!(run_hgs_decomposed(&data, Some(&seed), &options()).is_err());
}
#[test]
fn unspecified_fleet_and_automatic_initialization_are_supported() {
    let mut data = instance();
    data.vehicles = 0;
    let result = run_hgs_decomposed(
        &data,
        None,
        &RouteDecompositionOptions {
            time_limit: 0.2,
            ..options()
        },
    )
    .unwrap();
    assert!(result.solution.best_cost.is_some());
    data.vehicles = 6;
    assert_eq!(
        result.solution.best_cost,
        Some(validate_plan(&data, &result.solution.routes).unwrap())
    );
}

#[test]
fn decomposition_runs_compressed_delayed_seed_subproblems_with_original_ids() {
    let data = instance();
    let seed = vec![vec![1, 4], vec![2, 3], vec![5, 6]];
    let mut opts = options();
    opts.time_limit = 1.;
    opts.hgs.compression = Some(crate::extensions::hgs::CompressionOptions { interval: 1 });
    opts.hgs.delay_seed = true;
    let result = run_hgs_decomposed(&data, Some(&seed), &opts).unwrap();
    assert!(result.decomposition.subproblems > 0);
    let cost = validate_plan(&data, &result.solution.routes).unwrap();
    assert_eq!(result.solution.best_cost, Some(cost));
    assert!(cost <= validate_plan(&data, &seed).unwrap());
    opts.hgs.compression.as_mut().unwrap().interval = 0;
    assert!(run_hgs_decomposed(&data, Some(&seed), &opts).is_err());
}

#[test]
fn progressive_phases_reach_each_depth_without_resetting_user_generation_caps() {
    let data = instance();
    let seed = vec![vec![1, 2], vec![3, 4], vec![5, 6]];
    let mut opts = options();
    opts.time_limit = 2.;
    opts.phases = 2;
    opts.exploration_levels = 3;
    opts.hgs.generation_limit = Some(3);
    opts.master_local_search = true;
    opts.hgs.local_search.best_move = true;
    let result = run_hgs_decomposed(&data, Some(&seed), &opts).unwrap();
    assert_eq!(
        result.decomposition.exploration_schedule,
        vec![1, 1, 2, 2, 3, 3]
    );
    assert_eq!(result.decomposition.phases_completed, 6);
    assert_eq!(result.decomposition.master_passes, 6);
    for level in 1..=6 {
        let controls = exploration_options(&opts.hgs, level);
        assert!(controls.generation_limit.unwrap() <= 3);
        assert!(controls.population_size <= opts.hgs.population_size);
        assert!(controls.nb_iter <= opts.hgs.nb_iter);
    }
    assert!(
        validate_plan(&data, &result.solution.routes).unwrap()
            <= validate_plan(&data, &seed).unwrap()
    );
}

#[test]
fn master_education_improves_across_groups_that_cannot_exchange_customers() {
    let mut data = instance();
    data.vehicles = 3;
    data.capacity = 2.;
    let seed = vec![vec![1, 4], vec![2, 3], vec![5, 6]];
    let mut opts = options();
    opts.time_limit = 1.;
    opts.phases = 1;
    opts.target_customers = 1;
    opts.hgs.generation_limit = Some(0);
    opts.hgs.local_search.best_move = true;
    let plain = run_hgs_decomposed(&data, Some(&seed), &opts).unwrap();
    assert_eq!(
        plain.solution.best_cost,
        Some(validate_plan(&data, &seed).unwrap())
    );
    opts.master_local_search = true;
    let polished = run_hgs_decomposed(&data, Some(&seed), &opts).unwrap();
    assert_eq!(polished.decomposition.master_passes, 1);
    assert_eq!(polished.decomposition.master_improvements, 1);
    assert!(polished.solution.best_cost < plain.solution.best_cost);
    assert_eq!(
        validate_plan(&data, &polished.solution.routes).unwrap(),
        polished.solution.best_cost.unwrap()
    );
    let expired = educate_master(
        &data,
        &seed,
        &HgsOptions {
            time_limit: f64::MIN_POSITIVE,
            ..opts.hgs
        },
    )
    .unwrap();
    assert_eq!(expired, seed);
}

#[test]
fn vehicles_first_acceptance_rejects_cheaper_plans_with_more_vehicles() {
    assert!(!improves(
        (4, 10.),
        (3, 20.),
        Objective::VehiclesThenDistance
    ));
    assert!(improves(
        (2, 30.),
        (3, 20.),
        Objective::VehiclesThenDistance
    ));
    assert!(improves((4, 10.), (3, 20.), Objective::Distance));
    let data = instance();
    let seed = vec![vec![1, 2], vec![3, 4], vec![5, 6]];
    let mut opts = options();
    opts.time_limit = 1.;
    opts.hgs.generation_limit = Some(2);
    opts.objective = Objective::VehiclesThenDistance;
    opts.exploration_levels = 2;
    opts.master_local_search = true;
    opts.hgs.local_search.best_move = true;
    let result = run_hgs_decomposed(&data, Some(&seed), &opts).unwrap();
    assert!(result.solution.routes.len() <= seed.len());
    assert!(run_hgs_decomposed(
        &data,
        Some(&seed),
        &RouteDecompositionOptions {
            exploration_levels: 7,
            ..opts
        }
    )
    .is_err());
}
