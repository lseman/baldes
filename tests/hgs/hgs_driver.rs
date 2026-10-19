use baldes::hgs::driver::{run_hgs, HgsOptions};
use baldes::inputs::{Instance, Customer};

/// Create a small test instance (CVRP, 10 customers + depot).
fn make_test_instance() -> Instance {
    let customers: Vec<Customer> = vec![
        Customer { x: 0.0, y: 0.0, demand: 0.0, ready: 0.0, due: 100000.0, service: 0.0 }, // depot
        Customer { x: 10.0, y: 0.0, demand: 10.0, ready: 0.0, due: 100000.0, service: 0.0 },
        Customer { x: 20.0, y: 0.0, demand: 15.0, ready: 0.0, due: 100000.0, service: 0.0 },
        Customer { x: 0.0, y: 10.0, demand: 12.0, ready: 0.0, due: 100000.0, service: 0.0 },
        Customer { x: 10.0, y: 10.0, demand: 8.0, ready: 0.0, due: 100000.0, service: 0.0 },
        Customer { x: 20.0, y: 10.0, demand: 14.0, ready: 0.0, due: 100000.0, service: 0.0 },
        Customer { x: 0.0, y: 20.0, demand: 11.0, ready: 0.0, due: 100000.0, service: 0.0 },
        Customer { x: 10.0, y: 20.0, demand: 9.0, ready: 0.0, due: 100000.0, service: 0.0 },
        Customer { x: 20.0, y: 20.0, demand: 13.0, ready: 0.0, due: 100000.0, service: 0.0 },
        Customer { x: 15.0, y: 15.0, demand: 7.0, ready: 0.0, due: 100000.0, service: 0.0 },
        Customer { x: 5.0, y: 5.0, demand: 6.0, ready: 0.0, due: 100000.0, service: 0.0 },
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
    assert!(result.is_ok(), "HGS should run without error: {:?}", result.err());

    let result = result.unwrap();
    assert!(result.elapsed_secs < 10.0, "Should complete within time limit");
    assert!(result.generations > 0, "Should execute at least one generation");
    eprintln!("RESULT: best_cost={:?}, generations={}, elapsed={:.3}s", result.best_cost, result.generations, result.elapsed_secs);
    eprintln!("RESULT: routes={:?}", result.routes);

    // HGS should find a feasible solution for this small instance.
    assert!(result.best_cost.is_some(), "Should find a feasible solution");
    assert!(result.best_cost.unwrap() > 0.0, "Cost should be positive");

    // Routes should be non-empty if feasible.
    if result.routes.is_empty() {
        assert!(result.best_cost.is_none(), "Empty routes means infeasible");
    }
}
