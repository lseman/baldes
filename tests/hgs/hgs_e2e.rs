use baldes::hgs::driver::{run_hgs, HgsOptions};
use baldes::inputs::Instance;

/// Check the HGS routes: every customer once, capacity, and time windows in
/// HGS's metric (coordinates and times scaled by 10, distances truncated).
fn assert_feasible(instance: &Instance, routes: &[Vec<usize>]) {
    let customers = &instance.customers;
    let mut visits = vec![0; customers.len()];
    let scaled = |v: f64| (v * 10.0) as i64;
    let travel = |a: usize, b: usize| {
        let (dx, dy) = (customers[a].x * 10.0 - customers[b].x * 10.0, customers[a].y * 10.0 - customers[b].y * 10.0);
        (dx * dx + dy * dy).sqrt() as i64
    };
    assert!(routes.len() <= instance.vehicles, "{} routes", routes.len());
    for route in routes {
        let load: f64 = route.iter().map(|&c| customers[c].demand).sum();
        assert!(load <= instance.capacity, "route {route:?} carries {load}");
        let (mut time, mut previous) = (0, 0);
        for &c in route.iter().chain(std::iter::once(&0)) {
            visits[c] += 1;
            time = (time + travel(previous, c)).max(scaled(customers[c].ready));
            assert!(time <= scaled(customers[c].due), "late at {c} in {route:?}");
            time += scaled(customers[c].service);
            previous = c;
        }
    }
    assert!(visits[1..].iter().all(|&v| v == 1), "customers not visited exactly once");
}

#[test]
fn hgs_solves_solomon_instances() {
    // Best-known solutions (truncated-tenth metric) at real distances.
    for (name, best_known) in [("C201", 591.56), ("R201", 1147.82)] {
        let instance = Instance::read(format!("{}/instances/{name}.txt", env!("CARGO_MANIFEST_DIR"))).unwrap();
        let options = HgsOptions { time_limit: 2.0, nb_iter: 1000, ..Default::default() };
        let result = run_hgs(&instance, &options).expect("HGS run failed");
        let cost = result.best_cost.expect("no feasible solution");
        assert_feasible(&instance, &result.routes);
        // Within 2% of the best-known distance after two seconds.
        assert!(cost < best_known * 1.02, "{name}: {cost} vs best known {best_known}");
    }
}

#[test]
fn hgs_is_deterministic_without_time_limit() {
    let instance = Instance::read(concat!(env!("CARGO_MANIFEST_DIR"), "/instances/C101.txt")).unwrap();
    let options = HgsOptions { time_limit: 0.0, nb_iter: 300, seed: 7, ..Default::default() };
    let first = run_hgs(&instance, &options).unwrap();
    let second = run_hgs(&instance, &options).unwrap();
    assert_feasible(&instance, &first.routes);
    assert_eq!(first.routes, second.routes);
    assert_eq!(first.generations, second.generations);
}
