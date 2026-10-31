//! Adaptive iterated local search (AILS-II) over complete CVRP solutions.
//!
//! The first phase follows an acceptance-selected reference. After a bounded
//! number of non-improving iterations, the second phase selects references
//! from the diverse elite set. Each reference is educated by a seeded HGS run.
use super::driver::{run_hgs_seeded, HgsOptions, HgsResult};
use super::solution::validate_plan;
use crate::inputs::Instance;

#[derive(Clone, Debug)]
pub struct AilsIiOptions {
    pub time_limit: f64,
    pub iterations: usize,
    pub phase_length: usize,
    pub elite_size: usize,
    pub threshold: f64,
    pub hgs: HgsOptions,
}

impl Default for AilsIiOptions {
    fn default() -> Self {
        Self {
            time_limit: 300.0,
            iterations: 100,
            phase_length: 20,
            elite_size: 8,
            threshold: 0.001,
            hgs: HgsOptions::default(),
        }
    }
}

/// Run AILS-II from one complete feasible plan. The returned solution is
/// independently replayed before it can become the incumbent or an elite.
pub fn run_ails_ii(
    instance: &Instance,
    seed: &[Vec<usize>],
    options: &AilsIiOptions,
) -> Result<HgsResult, String> {
    if options.iterations == 0
        || options.phase_length == 0
        || options.elite_size == 0
        || !options.time_limit.is_finite()
        || options.time_limit < 0.0
        || !options.threshold.is_finite()
        || options.threshold < 0.0
    {
        return Err("invalid AILS-II iteration, phase, elite or threshold setting".into());
    }
    let initial_cost = validate_plan(instance, seed)?;
    let mut best = seed.to_vec();
    let mut best_cost = initial_cost;
    let mut current = best.clone();
    let mut current_cost = best_cost;
    let mut elite = vec![(best_cost, best.clone())];
    let started = std::time::Instant::now();
    let mut total_generations: u32 = 0;
    let mut no_improve = 0usize;
    let mut rng = super::params::Xoroshiro128Plus::new(options.hgs.seed);
    // Whole-instance HGS spends meaningful time building its initial
    // population. Avoid slicing an XL budget into sub-second calls that never
    // reach a generation; with a wall limit, use at most one call per 15 sec.
    let iterations = if options.time_limit > 0.0 {
        options
            .iterations
            .min((options.time_limit / 15.0).floor().max(1.0) as usize)
    } else {
        options.iterations
    };
    for iteration in 0..iterations {
        if options.time_limit > 0.0 && started.elapsed().as_secs_f64() >= options.time_limit {
            break;
        }
        let phase_two = (iteration / options.phase_length) % 2 == 1;
        let mut reference = if phase_two {
            elite[rng.next() as usize % elite.len()].1.clone()
        } else {
            current.clone()
        };
        let original = reference.clone();
        super::intensify::sisr_kick(
            instance,
            &mut reference,
            &mut rng,
            1 + (no_improve / 2).min(12),
        );
        if validate_plan(instance, &reference).is_err() {
            reference = original;
        }
        let mut hgs = options.hgs.clone();
        hgs.seed = rng.next() as i32;
        if options.time_limit > 0.0 {
            let remaining = (options.time_limit - started.elapsed().as_secs_f64()).max(0.0);
            let slice = remaining / (iterations - iteration) as f64;
            hgs.time_limit = if hgs.time_limit > 0.0 {
                hgs.time_limit.min(slice)
            } else {
                slice
            };
        }
        let result = run_hgs_seeded(instance, &hgs, &[reference])?;
        total_generations = total_generations.saturating_add(result.generations);
        let Some(cost) = result.best_cost else {
            continue;
        };
        validate_plan(instance, &result.routes)?;
        if cost < best_cost - 1e-7 {
            best_cost = cost;
            best.clone_from(&result.routes);
            no_improve = 0;
        } else {
            no_improve += 1;
        }
        let duplicate = elite.iter().any(|(_, routes)| routes == &result.routes);
        if !duplicate {
            elite.push((cost, result.routes.clone()));
            elite.sort_by(|a, b| a.0.total_cmp(&b.0));
            elite.truncate(options.elite_size);
        }
        let allowance = options.threshold * current_cost.abs();
        if cost <= current_cost + allowance || no_improve >= options.phase_length {
            current = result.routes;
            current_cost = cost;
            no_improve = 0;
        }
    }
    Ok(HgsResult {
        routes: best,
        columns: Vec::new(),
        best_cost: Some(best_cost),
        generations: total_generations,
        elapsed_secs: started.elapsed().as_secs_f64(),
    })
}
