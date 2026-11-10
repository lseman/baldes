//! Consensus-chain compression for HGS. Active graphs shrink between search
//! stages; every returned witness is expanded and replayed on the original input.

use super::{
    genetic::{EvolutionState, Genetic},
    local_search::HGSLocalSearch,
    population::Checkpoint,
    HgsOptions, HgsResult, Params, Population, Split,
};
use crate::inputs::Instance;
use std::time::Instant;
pub(crate) mod model;
pub(crate) use model::Consensus;
use model::{Expansion, Routes};

#[derive(Clone, Debug)]
pub struct CompressionOptions {
    /// Evolutionary generations between consensus checks (must be nonzero).
    pub interval: usize,
}
impl Default for CompressionOptions {
    fn default() -> Self {
        Self { interval: 75 }
    }
}

#[derive(Clone, Debug, Default)]
pub struct CompressionStats {
    pub checks: usize,
    pub contractions: usize,
    pub initial_customers: usize,
    pub final_customers: usize,
    pub frozen_routes: usize,
}
#[derive(Clone, Debug)]
pub struct CompressedHgsResult {
    pub solution: HgsResult,
    pub compression: CompressionStats,
}

/// Run the HGS with periodic consensus compression. Only arcs shared across all
/// feasible observations of an epoch AND the protected population are fixed.
/// Compression restricts the heuristic search space; it is opt-in. The original
/// incumbent, RNG stream, adaptive penalties and time/stagnation budgets survive.
pub fn run_hgs_compressed(
    instance: &Instance,
    options: &HgsOptions,
    compression: &CompressionOptions,
) -> Result<CompressedHgsResult, String> {
    run_seeded(instance, options, compression, &[], false)
}

pub(crate) fn run_seeded(
    instance: &Instance,
    options: &HgsOptions,
    compression: &CompressionOptions,
    seeds: &[Routes],
    collect: bool,
) -> Result<CompressedHgsResult, String> {
    crate::extensions::common::validate_instance(instance)?;
    if !options.local_search.valid()
        || instance.customers.len() < 2
        || compression.interval == 0
        || !options.time_limit.is_finite()
        || options.time_limit < 0.
        || options.population_size == 0
        || options.population_size > i32::MAX as u32
        || options.nb_iter == 0
        || options.nb_iter > i32::MAX as u32
    {
        return Err("invalid HGS compression options".into());
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
    params.config.local_search = options.local_search.clone();
    params.config.time_limit = options.time_limit;
    params.config.nb_iter = options.nb_iter as i32;
    params.config.generation_limit = options.generation_limit;
    params.config.minimum_population_size = options.population_size as i32;
    params.set_seed(options.seed);
    params.verbose = options.verbose;
    params.reset_clock();
    let clock = Instant::now();
    let n = params.nb_clients as usize;
    let mut stats = CompressionStats {
        initial_customers: n,
        final_customers: n,
        ..Default::default()
    };
    let mut expansion = Expansion::new(n);
    let mut saved = None;
    let mut state = EvolutionState::default();
    let mut incumbent: Option<(f64, Routes)> = None;
    let mut columns = std::collections::BTreeSet::new();
    let controls = LevelControls {
        stagnation: options.nb_iter as usize,
        interval: compression.interval,
        delay_seed: options.delay_seed,
        collect,
    };
    let mut generations: u32;
    loop {
        let level = run_level(params, saved, seeds, &controls, &mut state, &mut stats);
        generations = level.generations;
        if collect {
            columns.extend(expansion.expand(&level.columns));
        }
        if let Some(routes) = level.best.as_ref() {
            let expanded = expansion.expand(routes);
            let cost = super::solution::validate_plan(instance, &expanded)?;
            if incumbent.as_ref().is_none_or(|(best, _)| cost < *best) {
                incumbent = Some((cost, expanded));
            }
        }
        let Some(transition) = level.next else {
            break;
        };
        let (plan, stage) = match transition {
            Transition::Terminal(plan) => (plan, None),
            Transition::Continue(stage, plan) => (plan, Some(*stage)),
        };
        expansion = expansion.contract(&plan);
        stats.contractions += 1;
        stats.final_customers = plan.chains.len();
        stats.frozen_routes = expansion.fixed.len();
        if options.verbose {
            crate::log_err!(
                "  HGS compression | {} active customers | {} frozen routes | generation {}",
                stats.final_customers,
                stats.frozen_routes,
                generations
            );
        }
        if plan.chains.is_empty() {
            let routes = expansion.fixed.clone();
            let cost = super::solution::validate_plan(instance, &routes)?;
            if incumbent.as_ref().is_none_or(|(best, _)| cost < *best) {
                incumbent = Some((cost, routes));
            }
            break;
        }
        let Some(stage) = stage else {
            break;
        };
        if stage.params.is_time_limit_exceeded() {
            break;
        }
        params = stage.params;
        saved = Some(stage.population);
    }
    let (best_cost, routes) = incumbent.map_or((None, Vec::new()), |(c, r)| (Some(c), r));
    Ok(CompressedHgsResult {
        solution: HgsResult {
            routes,
            best_cost,
            columns: columns.into_iter().collect(),
            generations,
            elapsed_secs: clock.elapsed().as_secs_f64(),
        },
        compression: stats,
    })
}

struct SearchStage {
    params: Params,
    population: Checkpoint,
}
enum Transition {
    Terminal(model::Plan),
    Continue(Box<SearchStage>, model::Plan),
}
struct Level {
    columns: Routes,
    best: Option<Routes>,
    generations: u32,
    next: Option<Transition>,
}
struct LevelControls {
    stagnation: usize,
    interval: usize,
    delay_seed: bool,
    collect: bool,
}
fn run_level(
    params: Params,
    saved: Option<Checkpoint>,
    seeds: &[Routes],
    controls: &LevelControls,
    state: &mut EvolutionState,
    stats: &mut CompressionStats,
) -> Level {
    let mut split = Split::new(&params);
    let mut local = HGSLocalSearch::new(&params);
    if let Some(granular) = state.granular.filter(|&g| g != params.config.nb_granular) {
        local.set_granular(granular);
    }
    let initialize = saved.is_none();
    let mut population = if let Some(saved) = saved {
        saved.restore(&params)
    } else {
        let mut population = Population::new(&params);
        if controls.collect {
            population.collected_routes = Some(Default::default());
        }
        population.initialize_seeds(seeds, &mut local, controls.delay_seed);
        population
    };
    if controls.collect && population.collected_routes.is_none() {
        population.collected_routes = Some(Default::default());
    }
    if initialize || (population.feasible.is_empty() && population.infeasible.is_empty()) {
        population.generate_population(&mut split, &mut local);
    }
    population.consensus = Some(Consensus::new(params.nb_clients as usize));
    let mut genetic = Genetic::new(&params, &mut split, &mut local);
    loop {
        let finished = genetic.run_epoch(
            &mut population,
            controls.stagnation,
            state,
            controls.interval,
        );
        let best = population
            .get_best_found()
            .filter(|i| i.is_feasible)
            .map(|i| {
                i.chrom_r
                    .iter()
                    .filter(|r| !r.is_empty())
                    .cloned()
                    .collect()
            });
        let generations = population.nb_generations;
        if finished {
            return Level {
                columns: population
                    .collected_routes
                    .take()
                    .unwrap_or_default()
                    .into_iter()
                    .collect(),
                best,
                generations,
                next: None,
            };
        }
        stats.checks += 1;
        if let Some(plan) = population.compression_plan() {
            if plan.chains.is_empty() {
                return Level {
                    columns: population
                        .collected_routes
                        .take()
                        .unwrap_or_default()
                        .into_iter()
                        .collect(),
                    best,
                    generations,
                    next: Some(Transition::Terminal(plan)),
                };
            }
            if let Some(next) = plan.parameters(&params) {
                if let Some(saved) = population.checkpoint().project(&next, &plan) {
                    return Level {
                        columns: population
                            .collected_routes
                            .take()
                            .unwrap_or_default()
                            .into_iter()
                            .collect(),
                        best,
                        generations,
                        next: Some(Transition::Continue(
                            Box::new(SearchStage {
                                params: next,
                                population: saved,
                            }),
                            plan,
                        )),
                    };
                }
            }
        }
    }
}

#[cfg(test)]
#[path = "../../../../tests/unit/extensions/hgs/compression.rs"]
mod tests;
