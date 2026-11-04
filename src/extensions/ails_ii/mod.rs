//! Adaptive iterated local search over complete CVRP/VRPTW solutions.
//!
//! The search uses a two-phase reference policy and adaptive diversification
//! inspired by AILS-II. Feasible insertion and complete-plan validation support
//! time windows; the published CVRP feasibility-repair procedure and granular
//! SWAP* neighborhood are not yet implemented.
mod perturbation;
pub mod repair;
#[cfg(test)]
#[path = "../../../tests/unit/extensions/ails_ii.rs"]
mod tests;
use crate::{
    extensions::common::{self, validate_plan},
    inputs::Instance,
    rng::DefaultRng,
};

#[derive(Clone, Debug)]
pub struct AilsIiOptions {
    /// Search budget in seconds; zero runs the configured number of iterations.
    pub time_limit: f64,
    /// Maximum number of perturbation and local-search iterations.
    pub iterations: usize,
    /// Number of iterations per reference-selection phase and adaptation window.
    pub phase_length: usize,
    /// Maximum number of distinct elite plans retained.
    pub elite_size: usize,
    /// Final acceptance factor between zero and one.
    pub threshold: f64,
    /// Deterministic random seed.
    pub seed: i32,
}
#[derive(Clone, Debug)]
pub struct AilsIiResult {
    /// Best feasible route plan; depot vertices are omitted.
    pub routes: Vec<Vec<usize>>,
    /// Distance of the returned plan under the instance metric.
    pub best_cost: f64,
    /// Completed perturbation and local-search iterations.
    pub iterations: usize,
    /// Elapsed search time in seconds.
    pub elapsed_secs: f64,
}
impl Default for AilsIiOptions {
    fn default() -> Self {
        Self {
            time_limit: 300.,
            iterations: 1000,
            phase_length: 20,
            elite_size: 8,
            threshold: 0.01,
            seed: 0,
        }
    }
}

/// Improve a complete feasible seed, validating the returned plan.
pub fn run_ails_ii(
    instance: &Instance,
    seed: &[Vec<usize>],
    options: &AilsIiOptions,
) -> Result<AilsIiResult, String> {
    if options.iterations == 0
        || options.phase_length == 0
        || options.elite_size == 0
        || !options.time_limit.is_finite()
        || options.time_limit < 0.
        || !options.threshold.is_finite()
        || !(0.0..=1.0).contains(&options.threshold)
    {
        return Err("invalid AILS-II iteration, phase, elite or threshold setting".into());
    }
    validate_plan(instance, seed)?;
    let started = std::time::Instant::now();
    let deadline = if options.time_limit == 0. {
        started + std::time::Duration::from_secs(365 * 24 * 3600)
    } else {
        let duration = std::time::Duration::try_from_secs_f64(options.time_limit)
            .map_err(|_| "AILS-II time limit is too large")?;
        started
            .checked_add(duration)
            .ok_or("AILS-II time limit is too large")?
    };
    let mut rng = DefaultRng::new(options.seed);
    let mut best = seed.to_vec();
    common::improve_plan(
        instance,
        &mut best,
        pass_deadline(deadline, options.time_limit),
    );
    let mut best_cost = validate_plan(instance, &best)?;
    let mut current = best.clone();
    let mut elite = vec![(best_cost, best.clone())];
    let mut iterations = 0usize;
    let mut recent = std::collections::VecDeque::new();
    let max_strength = instance.customers.len().saturating_sub(1).max(1) as f64;
    let mut strength = [30.0_f64.min(max_strength); 3];
    let mut distance_sum = [0.0; 3];
    let mut distance_count = [0usize; 3];
    for iteration in 0..options.iterations {
        if std::time::Instant::now() >= deadline {
            break;
        }
        let phase_two = (iteration / options.phase_length) % 2 == 1;
        let mut candidate = if phase_two {
            elite[rng.next() as usize % elite.len()].1.clone()
        } else {
            current.clone()
        };
        let original = candidate.clone();
        let operator = (rng.next() % 3) as usize;
        if operator == 0 {
            common::sisr::ruin_recreate(
                instance,
                &mut candidate,
                &mut rng,
                strength[operator].round() as usize,
                deadline,
            );
        } else {
            perturbation::perturb(
                instance,
                &mut candidate,
                &mut rng,
                strength[operator].round() as usize,
                operator == 1,
                deadline,
            );
        }
        if validate_plan(instance, &candidate).is_err() {
            candidate = original.clone();
        }
        let changed: Vec<_> = candidate
            .iter()
            .enumerate()
            .map(|(i, route)| original.get(i) != Some(route))
            .collect();
        common::improve_modified_routes(
            instance,
            &mut candidate,
            pass_deadline(deadline, options.time_limit),
            &changed,
        );
        let cost = validate_plan(instance, &candidate)?;
        distance_sum[operator] += edge_distance(&original, &candidate) as f64;
        distance_count[operator] += 1;
        if distance_count[operator] == options.phase_length {
            let progress = search_progress(started, options, iteration);
            let target = 30.0 * (0.5_f64).powf(progress);
            let observed = distance_sum[operator] / distance_count[operator] as f64;
            strength[operator] = if observed > 0.0 {
                strength[operator] * target / observed
            } else {
                max_strength
            };
            strength[operator] = strength[operator].clamp(1.0, max_strength);
            distance_sum[operator] = 0.0;
            distance_count[operator] = 0;
        }
        iterations += 1;
        if cost < best_cost - 1e-7 {
            best = candidate.clone();
            best_cost = cost;
        }
        let candidate_key = canonical_routes(&candidate);
        if !elite
            .iter()
            .any(|(_, routes)| canonical_routes(routes) == candidate_key)
        {
            elite.push((cost, candidate.clone()));
            elite.sort_by(|a, b| a.0.total_cmp(&b.0));
            elite.truncate(options.elite_size);
        }
        recent.push_back(cost);
        if recent.len() > options.phase_length {
            recent.pop_front();
        }
        let min = recent.iter().copied().fold(f64::INFINITY, f64::min);
        let mean = recent.iter().sum::<f64>() / recent.len() as f64;
        let progress = search_progress(started, options, iteration + 1);
        let eta = options.threshold.max(f64::EPSILON).powf(progress);
        if cost <= min + eta * (mean - min) {
            current = candidate;
        }
    }
    Ok(AilsIiResult {
        routes: best,
        best_cost,
        iterations,
        elapsed_secs: started.elapsed().as_secs_f64(),
    })
}

fn pass_deadline(overall: std::time::Instant, time_limit: f64) -> std::time::Instant {
    if time_limit == 0.0 {
        return overall;
    }
    let now = std::time::Instant::now();
    let budget = std::time::Duration::from_secs_f64((time_limit * 0.1).max(0.001));
    now.checked_add(budget)
        .map_or(overall, |end| end.min(overall))
}

fn canonical_routes(routes: &[Vec<usize>]) -> Vec<Vec<usize>> {
    let mut result: Vec<_> = routes
        .iter()
        .filter(|route| !route.is_empty())
        .cloned()
        .collect();
    result.sort_unstable();
    result
}

fn search_progress(started: std::time::Instant, options: &AilsIiOptions, iterations: usize) -> f64 {
    let iteration_progress = iterations as f64 / options.iterations as f64;
    if options.time_limit > 0.0 {
        iteration_progress
            .max(started.elapsed().as_secs_f64() / options.time_limit)
            .min(1.0)
    } else {
        iteration_progress.min(1.0)
    }
}

fn edge_distance(a: &[Vec<usize>], b: &[Vec<usize>]) -> usize {
    fn edges(routes: &[Vec<usize>]) -> std::collections::HashMap<(usize, usize), usize> {
        let mut counts = std::collections::HashMap::new();
        for edge in routes.iter().filter(|r| !r.is_empty()).flat_map(|route| {
            let nodes = std::iter::once(0)
                .chain(route.iter().copied())
                .chain(std::iter::once(0))
                .collect::<Vec<_>>();
            nodes
                .windows(2)
                .map(|pair| {
                    if pair[0] < pair[1] {
                        (pair[0], pair[1])
                    } else {
                        (pair[1], pair[0])
                    }
                })
                .collect::<Vec<_>>()
        }) {
            *counts.entry(edge).or_default() += 1;
        }
        counts
    }
    let other = edges(b);
    edges(a)
        .into_iter()
        .map(|(edge, count)| count.saturating_sub(other.get(&edge).copied().unwrap_or(0)))
        .sum()
}
