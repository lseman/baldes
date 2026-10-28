//! High-level driver for running HGS on VRPTW/CVRP instances.

use crate::extensions::hgs::local_search::HGSLocalSearch;
use crate::extensions::hgs::split::Split;
use crate::extensions::hgs::{Genetic, Params, Population};
use crate::inputs::Instance;

/// HGS solver options.
#[derive(Clone, Debug)]
pub struct HgsOptions {
    /// Time limit in seconds (0 = no limit). With a limit, HGS restarts its
    /// population after `nb_iter` non-improving iterations instead of stopping.
    pub time_limit: f64,
    /// Number of iterations without improvement before termination or restart.
    pub nb_iter: u32,
    /// Minimum population size (mu).
    pub population_size: u32,
    /// Random seed.
    pub seed: i32,
    /// Print the population state periodically.
    pub verbose: bool,
}

impl Default for HgsOptions {
    fn default() -> Self {
        Self {
            time_limit: 60.0,
            nb_iter: 20000,
            population_size: 25,
            seed: 0,
            verbose: false,
        }
    }
}

/// Result of running HGS.
#[derive(Clone, Debug)]
pub struct HgsResult {
    /// Best feasible route plan (excluding depot 0).
    pub routes: Vec<Vec<usize>>,
    /// Distinct encountered routes, populated only by `run_hgs_collecting_routes`.
    /// These candidates still require validation against the target graph.
    pub columns: Vec<Vec<usize>>,
    /// Distance of `routes` with the instance's own rounding (None if no
    /// feasible solution was found).
    pub best_cost: Option<f64>,
    /// Number of generations executed.
    pub generations: u32,
    /// Execution time in seconds.
    pub elapsed_secs: f64,
}

/// Run HGS on a VRPTW/CVRP instance and return the best solution found.
pub fn run_hgs(
    instance: &Instance,
    options: &HgsOptions,
) -> std::result::Result<HgsResult, String> {
    run(instance, options, false, &[])
}

/// Populate HGS with validated complete route plans before generating other
/// individuals. Customer indices exclude depot 0. Seeds remain eligible as
/// the best result even when initialization consumes the search budget.
pub fn run_hgs_seeded(
    instance: &Instance,
    options: &HgsOptions,
    seeds: &[Vec<Vec<usize>>],
) -> std::result::Result<HgsResult, String> {
    run(instance, options, false, seeds)
}

/// Run HGS and retain all distinct routes passed through population insertion,
/// including routes from individuals later discarded or replaced at restart.
pub fn run_hgs_collecting_routes(
    instance: &Instance,
    options: &HgsOptions,
) -> std::result::Result<HgsResult, String> {
    run(instance, options, true, &[])
}

fn run(
    instance: &Instance,
    options: &HgsOptions,
    collect: bool,
    seeds: &[Vec<Vec<usize>>],
) -> std::result::Result<HgsResult, String> {
    if !options.time_limit.is_finite()
        || options.time_limit < 0.
        || options.population_size == 0
        || options.population_size > i32::MAX as u32
        || options.nb_iter == 0
        || options.nb_iter > i32::MAX as u32
    {
        return Err("invalid HGS time, population size or iteration limit".into());
    }
    if instance.customers.len() < 2 {
        return Err("HGS needs at least one customer".into());
    }
    for seed in seeds {
        super::solution::validate_plan(instance, seed)?;
    }
    let mut params = Params::from_instance(instance);
    if seeds
        .iter()
        .any(|seed| seed.iter().filter(|r| !r.is_empty()).count() > params.nb_vehicles as usize)
    {
        return Err("seed exceeds the HGS fleet; set instance.vehicles explicitly".into());
    }
    params.config.time_limit = options.time_limit;
    params.config.nb_iter = options.nb_iter as i32;
    params.config.minimum_population_size = options.population_size as i32;
    params.set_seed(options.seed);
    params.verbose = options.verbose;
    params.reset_clock();
    let params = params;

    if options.verbose {
        let _ = crate::logging::banner(&mut std::io::stderr().lock(), "HYBRID GENETIC SEARCH");
        crate::log_err!(
            "  Instance: {} | Customers: {} | Population: {} | Seed: {}",
            instance.name,
            instance.customers.len() - 1,
            options.population_size,
            options.seed
        );
        crate::log_err!("  Distance rounding: {:?}", instance.rounding);
        crate::log_err!(
            "  Time limit: {}s | Stall limit: {}",
            options.time_limit,
            options.nb_iter
        );
        crate::log_err!("{}", "─".repeat(96));
        crate::log_err!(
            "{:>6} {:>7} {:>13} {:>6} {:>6} {:>6} {:>13} {:>13} {:>8}",
            "Iter",
            "Stall",
            "Incumbent",
            "Routes",
            "Feas",
            "Infeas",
            "Best feas",
            "Best infeas",
            "Time(s)"
        );
        crate::log_err!("{}", "─".repeat(96));
    }
    let mut split = Split::new(&params);
    let mut local_search = HGSLocalSearch::new(&params);
    let mut population = Population::new(&params);
    if collect {
        population.collected_routes = Some(Default::default());
    }
    for seed in seeds {
        let mut individual = super::Individual::new_empty(&params);
        let mut routes: Vec<_> = seed.iter().filter(|r| !r.is_empty()).cloned().collect();
        routes.resize(params.nb_vehicles as usize, Vec::new());
        individual.chrom_t = routes.iter().flatten().copied().collect();
        individual.chrom_r = routes;
        individual.evaluate_complete_cost();
        population.add_individual(&individual, true);
        if !params.is_time_limit_exceeded() {
            population.do_hgs_local_search_and_add_individual(&mut individual, &mut local_search);
        }
    }
    population.generate_population(&mut split, &mut local_search);
    Genetic::new(&params, &mut split, &mut local_search)
        .run(&mut population, options.nb_iter as usize);

    let (routes, best_cost) = match population
        .get_best_found()
        .filter(|indiv| indiv.is_feasible)
    {
        Some(best) => {
            let routes: Vec<Vec<usize>> = best
                .chrom_r
                .iter()
                .filter(|r| !r.is_empty())
                .cloned()
                .collect();
            let cost = routes
                .iter()
                .map(|route| {
                    let stops = std::iter::once(0)
                        .chain(route.iter().copied())
                        .chain(std::iter::once(0));
                    stops
                        .clone()
                        .zip(stops.skip(1))
                        .map(|(a, b)| instance.distance(a, b, instance.rounding))
                        .sum::<f64>()
                })
                .sum();
            (routes, Some(cost))
        }
        None => (Vec::new(), None),
    };
    if options.verbose {
        population.print_state(population.nb_generations as i32, -1);
        crate::log_err!("{}", "─".repeat(96));
        crate::log_err!(
            "  HGS {} | {} generations | {} routes | Cost {} | {:.2}s\n",
            if best_cost.is_some() {
                "Feasible"
            } else {
                "No feasible solution"
            },
            population.nb_generations,
            routes.len(),
            best_cost.map_or_else(|| "--".into(), |c: f64| format!("{c:.5}")),
            params.elapsed_secs()
        );
    }
    Ok(HgsResult {
        routes,
        columns: population
            .collected_routes
            .take()
            .map_or_else(Vec::new, |r| r.into_iter().collect()),
        best_cost,
        generations: population.nb_generations,
        elapsed_secs: params.elapsed_secs(),
    })
}
