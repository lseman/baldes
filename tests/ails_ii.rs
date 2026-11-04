use baldes::extensions::ails_ii::{run_ails_ii, AilsIiOptions};
use baldes::extensions::common::improve_plan;
use baldes::extensions::common::sisr_kick;
use baldes::extensions::common::{construct_feasible, validate_plan};
use baldes::inputs::{Customer, Instance, Rounding};
use baldes::rng::DefaultRng;
use std::time::{Duration, Instant};

fn instance() -> Instance {
    let customers = [(0., 0.), (1., 0.), (0., 1.), (1., 1.), (2., 0.)]
        .into_iter()
        .enumerate()
        .map(|(i, (x, y))| Customer {
            x,
            y,
            demand: if i == 0 { 0. } else { 1. },
            ready: 0.,
            due: 100.,
            service: 0.,
        })
        .collect();
    Instance {
        name: "small".into(),
        vehicles: 2,
        capacity: 3.,
        customers,
        rounding: Rounding::None,
        distances: None,
    }
}

#[test]
fn standalone_search_preserves_feasibility_and_incumbent() {
    let instance = instance();
    let seed = construct_feasible(&instance, Instant::now() + Duration::from_secs(5)).unwrap();
    let initial = validate_plan(&instance, &seed).unwrap();
    let options = AilsIiOptions {
        time_limit: 0.,
        iterations: 10,
        phase_length: 2,
        ..Default::default()
    };
    let result = run_ails_ii(&instance, &seed, &options).unwrap();
    assert_eq!(result.iterations, 10);
    assert!(result.best_cost <= initial + 1e-8);
    assert!((validate_plan(&instance, &result.routes).unwrap() - result.best_cost).abs() < 1e-8);
}

#[test]
fn standalone_search_handles_a_single_customer() {
    let mut instance = instance();
    instance.customers.truncate(2);
    instance.vehicles = 1;
    let options = AilsIiOptions {
        time_limit: 0.,
        iterations: 6,
        phase_length: 1,
        ..Default::default()
    };
    let result = run_ails_ii(&instance, &[vec![1]], &options).unwrap();
    assert_eq!(result.routes, vec![vec![1]]);
    assert_eq!(result.iterations, 6);
    assert_eq!(
        result.best_cost,
        validate_plan(&instance, &result.routes).unwrap()
    );
}

#[test]
fn constructor_rejects_malformed_distance_matrix() {
    let mut instance = instance();
    instance.distances = Some(vec![vec![0.]]);
    assert!(construct_feasible(&instance, Instant::now() + Duration::from_secs(1)).is_err());
}

#[test]
fn repair_moves_customers_from_an_overloaded_route() {
    let instance = instance();
    let mut routes = vec![vec![1, 2, 3, 4], vec![]];
    assert!(baldes::extensions::ails_ii::repair::repair_plan(
        &instance,
        &mut routes,
        Instant::now() + Duration::from_secs(5)
    ));
    assert!(validate_plan(&instance, &routes).is_ok());
}

#[test]
fn repair_restores_plan_if_deadline_expires() {
    let instance = instance();
    let original = vec![vec![1, 2, 3, 4], vec![]];
    let mut routes = original.clone();
    assert!(!baldes::extensions::ails_ii::repair::repair_plan(
        &instance,
        &mut routes,
        Instant::now()
    ));
    assert_eq!(routes, original);
}

#[test]
fn repair_can_swap_when_no_single_relocation_reduces_overload() {
    let mut instance = instance();
    instance.capacity = 3.;
    instance.customers[1].demand = 2.;
    instance.customers[2].demand = 2.;
    let mut routes = vec![vec![1, 2], vec![3, 4]];
    assert!(baldes::extensions::ails_ii::repair::repair_plan(
        &instance,
        &mut routes,
        Instant::now() + Duration::from_secs(5),
    ));
    validate_plan(&instance, &routes).unwrap();
}

#[test]
fn repair_reorders_one_route_to_respect_time_windows() {
    let mut instance = instance();
    instance.capacity = 4.;
    instance.customers[1].due = 1.1;
    let mut routes = vec![vec![2, 1, 3, 4]];
    assert!(baldes::extensions::ails_ii::repair::repair_plan(
        &instance,
        &mut routes,
        Instant::now() + Duration::from_secs(5)
    ));
    validate_plan(&instance, &routes).unwrap();
}

#[test]
fn repair_rejects_invalid_coverage_and_matrix_without_mutation() {
    let mut instance = instance();
    let original = vec![vec![1, 1, 2], vec![3, 4]];
    let mut routes = original.clone();
    assert!(!baldes::extensions::ails_ii::repair::repair_plan(
        &instance,
        &mut routes,
        Instant::now() + Duration::from_secs(5)
    ));
    assert_eq!(routes, original);
    instance.distances = Some(vec![vec![0.]]);
    let original = vec![vec![1, 2, 3, 4], vec![]];
    let mut routes = original.clone();
    assert!(!baldes::extensions::ails_ii::repair::repair_plan(
        &instance,
        &mut routes,
        Instant::now() + Duration::from_secs(5)
    ));
    assert_eq!(routes, original);
}

#[test]
fn repair_restores_plan_when_fleet_capacity_is_insufficient() {
    let mut instance = instance();
    instance.capacity = 1.;
    let original = vec![vec![1, 2], vec![3, 4]];
    let mut routes = original.clone();
    assert!(!baldes::extensions::ails_ii::repair::repair_plan(
        &instance,
        &mut routes,
        Instant::now() + Duration::from_secs(5)
    ));
    assert_eq!(routes, original);
}

#[test]
fn oversized_time_limit_is_reported_instead_of_panicking() {
    let instance = instance();
    let seed = vec![vec![1, 2, 3], vec![4]];
    let options = AilsIiOptions {
        time_limit: f64::MAX,
        ..Default::default()
    };
    assert!(run_ails_ii(&instance, &seed, &options).is_err());
}

#[test]
fn local_search_exchanges_customers_between_full_routes() {
    let mut instance = instance();
    instance.capacity = 2.;
    for (customer, x) in instance
        .customers
        .iter_mut()
        .skip(1)
        .zip([1., 10., 2., 11.])
    {
        customer.x = x;
        customer.y = 0.;
    }
    let mut routes = vec![vec![1, 2], vec![3, 4]];
    let before = validate_plan(&instance, &routes).unwrap();
    improve_plan(
        &instance,
        &mut routes,
        Instant::now() + Duration::from_secs(5),
    );
    let after = validate_plan(&instance, &routes).unwrap();
    assert!(
        after < before - 1.0,
        "exchange did not improve {before} -> {after}"
    );
}

#[test]
fn expired_local_search_does_not_mutate_the_plan() {
    let instance = instance();
    let original = vec![vec![1, 2, 3], vec![4]];
    let mut routes = original.clone();
    improve_plan(&instance, &mut routes, Instant::now());
    assert_eq!(routes, original);
}

#[test]
fn relocation_handles_asymmetric_costs_and_nonzero_depot_diagonal() {
    let mut instance = instance();
    instance.capacity = 4.;
    let mut matrix = vec![vec![10.; 5]; 5];
    for (i, row) in matrix.iter_mut().enumerate() {
        row[i] = 0.;
    }
    matrix[0][0] = 100.;
    matrix[0][1] = 5.;
    matrix[1][0] = 5.;
    matrix[1][2] = 1.;
    instance.distances = Some(matrix);
    let mut routes = vec![vec![1], vec![2, 3, 4]];
    let before = validate_plan(&instance, &routes).unwrap();
    improve_plan(
        &instance,
        &mut routes,
        Instant::now() + Duration::from_secs(5),
    );
    let after = validate_plan(&instance, &routes).unwrap();
    assert!(
        after < before - 1.0,
        "relocation did not improve {before} -> {after}"
    );
}

#[test]
fn local_search_preserves_feasibility_for_varied_directed_matrices() {
    for seed in 0..24 {
        let mut instance = instance();
        let mut rng = DefaultRng::new(seed);
        let mut matrix = vec![vec![0.; 5]; 5];
        for row in &mut matrix {
            for value in row {
                *value = 1. + (rng.next() % 50) as f64;
            }
        }
        instance.distances = Some(matrix);
        instance
            .customers
            .iter_mut()
            .for_each(|c| c.due = f64::INFINITY);
        let mut routes = vec![vec![1, 2, 3], vec![4]];
        let before = validate_plan(&instance, &routes).unwrap();
        improve_plan(
            &instance,
            &mut routes,
            Instant::now() + Duration::from_secs(5),
        );
        let after = validate_plan(&instance, &routes).unwrap();
        assert!(after <= before + 1e-8, "seed {seed}: {before} -> {after}");
    }
}

#[test]
fn infeasible_seed_is_rejected_before_search() {
    let instance = instance();
    let seed = vec![vec![1, 2, 3, 4]];
    let options = AilsIiOptions {
        time_limit: 0.,
        iterations: 1,
        ..Default::default()
    };
    assert!(run_ails_ii(&instance, &seed, &options).is_err());
}

#[test]
fn constructor_respects_customer_and_depot_windows() {
    let mut instance = instance();
    instance.customers[1].due = 1.1;
    instance.customers[0].due = 8.;
    let routes = construct_feasible(&instance, Instant::now() + Duration::from_secs(5)).unwrap();
    assert!(validate_plan(&instance, &routes).is_ok());
}

#[test]
fn constructor_places_larger_demands_first_under_a_tight_fleet() {
    let mut instance = instance();
    let template = instance.customers[1].clone();
    instance.customers.resize(7, template);
    instance.vehicles = 2;
    instance.capacity = 7.;
    for (customer, demand) in instance
        .customers
        .iter_mut()
        .skip(1)
        .zip([2., 2., 2., 2., 3., 3.])
    {
        customer.demand = demand;
    }
    let routes = construct_feasible(&instance, Instant::now() + Duration::from_secs(5)).unwrap();
    assert_eq!(routes.iter().filter(|r| !r.is_empty()).count(), 2);
    validate_plan(&instance, &routes).unwrap();
}

#[test]
fn fixed_iteration_search_is_reproducible() {
    let instance = instance();
    let seed = vec![vec![1, 2, 3], vec![4]];
    let options = AilsIiOptions {
        time_limit: 0.,
        iterations: 12,
        phase_length: 3,
        seed: 17,
        ..Default::default()
    };
    let first = run_ails_ii(&instance, &seed, &options).unwrap();
    let second = run_ails_ii(&instance, &seed, &options).unwrap();
    assert_eq!(first.routes, second.routes);
    assert_eq!(first.best_cost, second.best_cost);
    assert_eq!(first.iterations, second.iterations);
}

#[test]
fn sisr_kick_preserves_the_customer_set() {
    let instance = instance();
    let mut routes = vec![vec![1, 2, 3], vec![4]];
    sisr_kick(&instance, &mut routes, &mut DefaultRng::new(12), 3);
    let mut customers: Vec<_> = routes.into_iter().flatten().collect();
    customers.sort_unstable();
    assert_eq!(customers, vec![1, 2, 3, 4]);
}

#[test]
fn sisr_recreation_preserves_feasibility_and_restores_on_timeout() {
    let instance = instance();
    let original = vec![vec![1, 2, 3], vec![4]];
    let mut routes = original.clone();
    let mut rng = DefaultRng::new(8);
    assert!(!baldes::extensions::common::sisr::ruin_recreate(
        &instance,
        &mut routes,
        &mut rng,
        4,
        Instant::now()
    ));
    assert_eq!(routes, original);
    let _ = baldes::extensions::common::sisr::ruin_recreate(
        &instance,
        &mut routes,
        &mut rng,
        4,
        Instant::now() + Duration::from_secs(5),
    );
    assert!(validate_plan(&instance, &routes).is_ok());
}

#[test]
fn sisr_keeps_capacity_and_coverage_across_seeds() {
    let instance = instance();
    let initial = vec![vec![1, 2, 3], vec![4]];
    let mut completed = 0;
    let mut changed = 0;
    for seed in 0..50 {
        let mut routes = initial.clone();
        let mut rng = DefaultRng::new(seed);
        if baldes::extensions::common::sisr::ruin_recreate(
            &instance,
            &mut routes,
            &mut rng,
            4,
            Instant::now() + Duration::from_secs(5),
        ) {
            completed += 1;
            changed += usize::from(routes != initial);
        }
        validate_plan(&instance, &routes).unwrap();
    }
    assert!(completed > 0, "SISR never completed recreation");
    assert!(changed > 0, "SISR never changed the plan");
}

#[test]
fn sisr_preserves_time_windows_across_seeds() {
    let mut instance = instance();
    instance.customers[1].due = 1.1;
    let initial = vec![vec![1, 2, 3], vec![4]];
    validate_plan(&instance, &initial).unwrap();
    for seed in 0..50 {
        let mut routes = initial.clone();
        let mut rng = DefaultRng::new(seed);
        let _ = baldes::extensions::common::sisr::ruin_recreate(
            &instance,
            &mut routes,
            &mut rng,
            4,
            Instant::now() + Duration::from_secs(5),
        );
        validate_plan(&instance, &routes).unwrap();
    }
}

#[cfg(feature = "clp")]
#[test]
fn cli_exposes_separate_command_and_rejects_hgs_flag() {
    let binary = env!("CARGO_BIN_EXE_baldes");
    let help = std::process::Command::new(binary)
        .args(["ails-ii", "--help"])
        .output()
        .unwrap();
    assert!(help.status.success());
    let invalid = std::process::Command::new(binary)
        .args(["hgs", "unused.vrp", "--ails-ii"])
        .output()
        .unwrap();
    assert!(!invalid.status.success());
    assert!(String::from_utf8_lossy(&invalid.stderr).contains("unknown argument: --ails-ii"));
    let invalid_ails = std::process::Command::new(binary)
        .args(["ails-ii", "unused.vrp", "--population-size", "10"])
        .output()
        .unwrap();
    assert!(!invalid_ails.status.success());
    assert!(String::from_utf8_lossy(&invalid_ails.stderr).contains("do not apply to ails-ii"));
}

#[cfg(feature = "clp")]
#[test]
fn standalone_cli_runs_requested_iterations() {
    let path = std::env::temp_dir().join(format!("baldes-ails-{}.vrp", std::process::id()));
    std::fs::write(&path, "NAME : tiny\nTYPE : CVRP\nDIMENSION : 4\nEDGE_WEIGHT_TYPE : EUC_2D\nCAPACITY : 4\nNODE_COORD_SECTION\n1 0 0\n2 1 0\n3 0 1\n4 1 1\nDEMAND_SECTION\n1 0\n2 1\n3 1\n4 1\nDEPOT_SECTION\n1\n-1\nEOF\n").unwrap();
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_baldes"))
        .arg("ails-ii")
        .arg(&path)
        .args(["--time", "5", "--iterations", "3"])
        .output()
        .unwrap();
    let invalid = std::process::Command::new(env!("CARGO_BIN_EXE_baldes"))
        .arg("ails-ii")
        .arg(&path)
        .args(["--iterations", "0"])
        .output()
        .unwrap();
    let _ = std::fs::remove_file(&path);
    assert!(!invalid.status.success());
    assert!(String::from_utf8_lossy(&invalid.stderr).contains("iterations must be positive"));
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(String::from_utf8_lossy(&output.stdout).contains("iterations=3"));
}
