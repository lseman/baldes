use baldes::extensions::hgs::driver::{run_hgs, HgsOptions};
use baldes::inputs::Instance;

/// Check every customer once, capacity and time windows in the selected metric.
fn assert_feasible(instance: &Instance, routes: &[Vec<usize>]) {
    let customers = &instance.customers;
    let mut visits = vec![0; customers.len()];
    assert!(routes.len() <= instance.vehicles, "{} routes", routes.len());
    for route in routes {
        let load: f64 = route.iter().map(|&c| customers[c].demand).sum();
        assert!(load <= instance.capacity, "route {route:?} carries {load}");
        let (mut time, mut previous) = (0_f64, 0);
        for &c in route.iter().chain(std::iter::once(&0)) {
            visits[c] += 1;
            time =
                (time + instance.distance(previous, c, instance.rounding)).max(customers[c].ready);
            assert!(time <= customers[c].due + 1e-6, "late at {c} in {route:?}");
            time += customers[c].service;
            previous = c;
        }
    }
    assert!(
        visits[1..].iter().all(|&v| v == 1),
        "customers not visited exactly once"
    );
}

#[test]
fn hgs_solves_solomon_instances() {
    // Best-known solutions (truncated-tenth metric) at real distances.
    for (name, best_known) in [("C201", 591.56), ("R201", 1147.82)] {
        let instance = Instance::read(format!(
            "{}/instances/Solomon/{name}.txt",
            env!("CARGO_MANIFEST_DIR")
        ))
        .unwrap();
        let options = HgsOptions {
            time_limit: 2.0,
            nb_iter: 1000,
            ..Default::default()
        };
        let result = run_hgs(&instance, &options).expect("HGS run failed");
        let cost = result.best_cost.expect("no feasible solution");
        assert_feasible(&instance, &result.routes);
        // Within 2% of the best-known distance after two seconds.
        assert!(
            cost < best_known * 1.02,
            "{name}: {cost} vs best known {best_known}"
        );
    }
}

#[test]
fn hgs_is_deterministic_without_time_limit() {
    let instance = Instance::read(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/instances/Solomon/C101.txt"
    ))
    .unwrap();
    let options = HgsOptions {
        time_limit: 0.0,
        nb_iter: 300,
        seed: 7,
        ..Default::default()
    };
    let first = run_hgs(&instance, &options).unwrap();
    let second = run_hgs(&instance, &options).unwrap();
    assert_feasible(&instance, &first.routes);
    assert_eq!(first.routes, second.routes);
    assert_eq!(first.generations, second.generations);
}

#[test]
fn hgs_fixed_work_matches_broad_swap_star_baselines() {
    // Broad search remains the default. Candidate filtering has separate
    // explicit snapshots because it changes the heuristic trajectory.
    for (name, seed, generations, expected_cost) in [
        ("C201", 0, 300, 591.5565566715014),
        ("R201", 0, 1200, 1148.094027615398),
        ("R201", 1, 847, 1147.803772621077),
        ("R201", 2, 967, 1148.165514469302),
    ] {
        let instance = Instance::read(format!(
            "{}/instances/Solomon/{name}.txt",
            env!("CARGO_MANIFEST_DIR")
        ))
        .unwrap();
        let options = HgsOptions {
            time_limit: 0.0,
            nb_iter: 300,
            seed,
            ..Default::default()
        };
        let result = run_hgs(&instance, &options).unwrap();
        assert_feasible(&instance, &result.routes);
        assert_eq!(result.generations, generations, "{name}, seed {seed}");
        assert!(
            (result.best_cost.unwrap() - expected_cost).abs() < 1e-8,
            "{name}, seed {seed}"
        );
    }
}

fn configured_fixed_work(
    instance: &Instance,
    seed: i32,
    neighbors: usize,
    interval: usize,
) -> (u32, f64) {
    use baldes::extensions::hgs::{
        genetic::Genetic, local_search::HGSLocalSearch, Params, Population, Split,
    };
    let mut params = Params::from_instance(instance);
    params.config.swap_star_neighbors = neighbors;
    params.config.swap_star_full_pass_interval = interval;
    params.config.nb_iter = 300;
    params.verbose = false;
    params.set_seed(seed);
    let mut split = Split::new(&params);
    let mut local_search = HGSLocalSearch::new(&params);
    let mut population = Population::new(&params);
    population.generate_population(&mut split, &mut local_search);
    Genetic::new(&params, &mut split, &mut local_search).run(&mut population, 300);
    let best = population.get_best_found().unwrap();
    assert!(best.is_feasible);
    let routes: Vec<_> = best
        .chrom_r
        .iter()
        .filter(|r| !r.is_empty())
        .cloned()
        .collect();
    assert_feasible(instance, &routes);
    (population.nb_generations, best.my_cost_sol.distance / 10.)
}

#[test]
fn disabling_swap_filter_preserves_broad_search_baselines() {
    let instance = Instance::read(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/instances/Solomon/R201.txt"
    ))
    .unwrap();
    for (neighbors, interval) in [(0, 5), (35, 1)] {
        for (seed, generations, expected_cost) in [
            (0, 1200, 1148.094027615398),
            (1, 847, 1147.803772621077),
            (2, 967, 1148.165514469302),
        ] {
            let (actual_generations, cost) =
                configured_fixed_work(&instance, seed, neighbors, interval);
            assert_eq!(actual_generations, generations);
            assert!((cost - expected_cost).abs() < 1e-8);
        }
    }
}

#[test]
fn filtered_swap_star_has_deterministic_fixed_work_baselines() {
    let instance = Instance::read(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/instances/Solomon/R201.txt"
    ))
    .unwrap();
    for (seed, generations, expected_cost) in [
        (0, 961, 1148.094027615398),
        (1, 504, 1152.7129469609317),
        (2, 967, 1148.165514469302),
    ] {
        let (actual_generations, cost) = configured_fixed_work(&instance, seed, 35, 5);
        assert_eq!(actual_generations, generations);
        assert!((cost - expected_cost).abs() < 1e-8);
    }
}

#[test]
fn swap_star_policies_solve_a_cvrplib_capacity_only_instance() {
    let instance = Instance::read(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/instances/cvrplib/A-n32-k5.vrp"
    ))
    .unwrap();
    for neighbors in [0, 35] {
        let (_, cost) = configured_fixed_work(&instance, 1, neighbors, 5);
        assert!(cost > 0.);
    }
}

#[test]
fn consensus_compression_shrinks_solomon_graph_and_replays_original_customers() {
    use baldes::extensions::hgs::{run_hgs_compressed, CompressionOptions};
    let instance = Instance::read(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/instances/Solomon/C201.txt"
    ))
    .unwrap();
    let result = run_hgs_compressed(
        &instance,
        &HgsOptions {
            time_limit: 3.,
            nb_iter: 300,
            seed: 0,
            ..Default::default()
        },
        &CompressionOptions::default(),
    )
    .unwrap();
    assert!(result.compression.contractions > 0);
    assert!(result.compression.final_customers < result.compression.initial_customers);
    assert!(result.compression.frozen_routes > 0);
    assert_feasible(&instance, &result.solution.routes);
    assert!((result.solution.best_cost.unwrap() - 591.5565566715014).abs() < 1e-6);
}
