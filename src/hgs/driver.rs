//! High-level driver for running HGS on VRPTW/CVRP instances.

use crate::hgs::local_search::HGSLocalSearch;
use crate::hgs::split::Split;
use crate::hgs::{Genetic, Params, Population};
use crate::vrptw::Instance;

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
        Self { time_limit: 60.0, nb_iter: 20000, population_size: 25, seed: 0, verbose: false }
    }
}

/// Result of running HGS.
#[derive(Clone, Debug)]
pub struct HgsResult {
    /// Best feasible route plan (excluding depot 0).
    pub routes: Vec<Vec<usize>>,
    /// Distance of `routes` with the instance's own rounding (None if no
    /// feasible solution was found).
    pub best_cost: Option<f64>,
    /// Number of generations executed.
    pub generations: u32,
    /// Execution time in seconds.
    pub elapsed_secs: f64,
}

/// Run HGS on a VRPTW/CVRP instance and return the best solution found.
pub fn run_hgs(instance: &Instance, options: &HgsOptions) -> std::result::Result<HgsResult, String> {
    if instance.customers.len() < 2 {
        return Err("HGS needs at least one customer".into());
    }
    let mut params = Params::from_instance(instance);
    params.config.time_limit = options.time_limit;
    params.config.nb_iter = options.nb_iter as i32;
    params.config.minimum_population_size = options.population_size as i32;
    params.set_seed(options.seed);
    params.verbose = options.verbose;
    params.reset_clock();
    let params = params;

    let mut split = Split::new(&params);
    let mut local_search = HGSLocalSearch::new(&params);
    let mut population = Population::new(&params);
    population.generate_population(&mut split, &mut local_search);
    Genetic::new(&params, &mut split, &mut local_search).run(&mut population, options.nb_iter as usize);

    let (routes, best_cost) = match population.get_best_found().filter(|indiv| indiv.is_feasible) {
        Some(best) => {
            let routes: Vec<Vec<usize>> = best.chrom_r.iter().filter(|r| !r.is_empty()).cloned().collect();
            let cost = routes
                .iter()
                .map(|route| {
                    let stops = std::iter::once(0).chain(route.iter().copied()).chain(std::iter::once(0));
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
    Ok(HgsResult {
        routes,
        best_cost,
        generations: population.nb_generations,
        elapsed_secs: params.elapsed_secs(),
    })
}
