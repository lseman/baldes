use baldes::extensions::hgs::driver::{run_hgs, HgsOptions};
use baldes::inputs::{Customer, Instance};

/// Create a small test instance (CVRP, 10 customers + depot).
fn make_test_instance() -> Instance {
    let customers: Vec<Customer> = vec![
        Customer {
            x: 0.0,
            y: 0.0,
            demand: 0.0,
            ready: 0.0,
            due: 100000.0,
            service: 0.0,
        }, // depot
        Customer {
            x: 10.0,
            y: 0.0,
            demand: 10.0,
            ready: 0.0,
            due: 100000.0,
            service: 0.0,
        },
        Customer {
            x: 20.0,
            y: 0.0,
            demand: 15.0,
            ready: 0.0,
            due: 100000.0,
            service: 0.0,
        },
        Customer {
            x: 0.0,
            y: 10.0,
            demand: 12.0,
            ready: 0.0,
            due: 100000.0,
            service: 0.0,
        },
        Customer {
            x: 10.0,
            y: 10.0,
            demand: 8.0,
            ready: 0.0,
            due: 100000.0,
            service: 0.0,
        },
        Customer {
            x: 20.0,
            y: 10.0,
            demand: 14.0,
            ready: 0.0,
            due: 100000.0,
            service: 0.0,
        },
        Customer {
            x: 0.0,
            y: 20.0,
            demand: 11.0,
            ready: 0.0,
            due: 100000.0,
            service: 0.0,
        },
        Customer {
            x: 10.0,
            y: 20.0,
            demand: 9.0,
            ready: 0.0,
            due: 100000.0,
            service: 0.0,
        },
        Customer {
            x: 20.0,
            y: 20.0,
            demand: 13.0,
            ready: 0.0,
            due: 100000.0,
            service: 0.0,
        },
        Customer {
            x: 15.0,
            y: 15.0,
            demand: 7.0,
            ready: 0.0,
            due: 100000.0,
            service: 0.0,
        },
        Customer {
            x: 5.0,
            y: 5.0,
            demand: 6.0,
            ready: 0.0,
            due: 100000.0,
            service: 0.0,
        },
    ];

    Instance {
        name: "test-cvrp".to_string(),
        vehicles: 3,
        capacity: 50.0,
        customers,
        rounding: baldes::inputs::Rounding::Nearest,
        distances: None,
    }
}

#[test]
fn test_hgs_cvrp_runs() {
    let instance = make_test_instance();

    let options = HgsOptions {
        time_limit: 2.0,
        nb_iter: 500,
        population_size: 20,
        ..Default::default()
    };

    let result = run_hgs(&instance, &options);
    assert!(
        result.is_ok(),
        "HGS should run without error: {:?}",
        result.err()
    );

    let result = result.unwrap();
    assert!(
        result.elapsed_secs < 10.0,
        "Should complete within time limit"
    );
    assert!(
        result.generations > 0,
        "Should execute at least one generation"
    );
    eprintln!(
        "RESULT: best_cost={:?}, generations={}, elapsed={:.3}s",
        result.best_cost, result.generations, result.elapsed_secs
    );
    eprintln!("RESULT: routes={:?}", result.routes);

    // HGS should find a feasible solution for this small instance.
    assert!(
        result.best_cost.is_some(),
        "Should find a feasible solution"
    );
    assert!(result.best_cost.unwrap() > 0.0, "Cost should be positive");

    // Routes should be non-empty if feasible.
    if result.routes.is_empty() {
        assert!(result.best_cost.is_none(), "Empty routes means infeasible");
    }
}

fn fractional_instance() -> Instance {
    Instance {
        name: "fractional".into(),
        vehicles: 1,
        capacity: 1.,
        customers: vec![
            Customer {
                x: 0.,
                y: 0.,
                demand: 0.,
                ready: 0.,
                due: 10.,
                service: 0.,
            },
            Customer {
                x: 1.,
                y: 1.,
                demand: 1.,
                ready: 0.,
                due: 1.41,
                service: 0.013,
            },
        ],
        rounding: baldes::inputs::Rounding::None,
        distances: None,
    }
}

#[test]
fn hgs_uses_unrounded_distances_unless_tenths_are_requested() {
    use baldes::extensions::hgs::Params;
    use baldes::inputs::Rounding;
    let mut instance = fractional_instance();
    let params = Params::from_instance(&instance);
    assert!((params.time_cost.get(0, 1) / 10. - 2_f64.sqrt()).abs() < 1e-12);
    assert!((params.cli[1].latest_arrival / 10. - 1.41).abs() < 1e-12);
    assert!((params.cli[1].service_duration / 10. - 0.013).abs() < 1e-12);
    let options = HgsOptions {
        time_limit: 0.,
        nb_iter: 1,
        population_size: 2,
        ..Default::default()
    };
    // 1.414... cannot meet a 1.41 deadline. The old implicit tenths metric
    // accepted this route while reporting its unrounded distance afterward.
    assert!(run_hgs(&instance, &options).unwrap().best_cost.is_none());
    instance.rounding = Rounding::Tenth;
    let result = run_hgs(&instance, &options).unwrap();
    assert_eq!(result.routes, vec![vec![1]]);
    assert!((result.best_cost.unwrap() - 2.8).abs() < 1e-12);
}

#[test]
fn hgs_retains_fractional_explicit_matrix_costs_and_search_costs() {
    use baldes::extensions::hgs::{Individual, Params};
    let mut instance = fractional_instance();
    instance.customers[1].due = 5.;
    // Explicit matrices define the metric even if rounding is requested.
    instance.rounding = baldes::inputs::Rounding::Tenth;
    instance.distances = Some(vec![vec![0., 1.23456], vec![2.34567, 0.]]);
    let params = Params::from_instance(&instance);
    assert!((params.time_cost.get(0, 1) / 10. - 1.23456).abs() < 1e-12);
    assert!((params.time_cost.get(1, 0) / 10. - 2.34567).abs() < 1e-12);
    let mut individual = Individual::new_empty(&params);
    individual.chrom_r[0] = vec![1];
    individual.evaluate_complete_cost();
    assert!(individual.is_feasible);
    let expected = 1.23456 + 2.34567;
    assert!((individual.my_cost_sol.distance / 10. - expected).abs() < 1e-12);
    let result = run_hgs(
        &instance,
        &HgsOptions {
            time_limit: 0.,
            nb_iter: 1,
            population_size: 2,
            ..Default::default()
        },
    )
    .unwrap();
    assert!((result.best_cost.unwrap() - expected).abs() < 1e-12);
}
