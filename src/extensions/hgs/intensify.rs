//! Seeded route decomposition, threshold acceptance and diverse elite restarts.
//!
//! Only selected routes enter a fresh HGS population, keeping the quadratic HGS
//! structures bounded by the subproblem size rather than the full instance.
use rayon::prelude::*;
use std::collections::BTreeSet;
use std::time::Instant;

use super::solution::validate_plan;
use super::{run_hgs, run_hgs_seeded, HgsOptions, Xoroshiro128Plus};
use crate::inputs::Instance;

/// Outer-search objective. HGS subproblems minimize distance at a fixed fleet
/// cap; periodic smaller-fleet solves implement route elimination.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Objective {
    Distance,
    VehiclesThenDistance,
}

#[derive(Clone, Debug)]
pub struct IntensifyOptions {
    /// Wall seconds for search, including subproblem setup; 0 uses rounds only.
    pub time_limit: f64,
    pub rounds: usize,
    /// Independent, disjoint route-subset solves per batch. One preserves the
    /// sequential search. Each worker owns its HGS population and RNG.
    pub workers: usize,
    /// Jobs selected from each solution snapshot; 0 selects `workers`. Set a
    /// fixed width when comparing serial and parallel execution of equal work.
    pub batch_size: usize,
    /// Rank nearby routes using the paper's spatial-temporal-demand score.
    pub std_similarity: bool,
    /// Rank route neighbors by Euclidean distance between customer barycenters.
    pub barycenter: bool,
    /// Maximum routes selected per round (at least two).
    pub max_routes: usize,
    /// Maximum selected customers. A single oversized seed route is skipped.
    pub max_customers: usize,
    pub subproblem: HgsOptions,
    /// Initial relative distance deterioration allowed; cools linearly to zero
    /// over each restart interval. Never permits an increase in vehicle count
    /// under VehiclesThenDistance.
    pub threshold: f64,
    pub elite_size: usize,
    /// Minimum fraction of differing directed arcs for another elite entry.
    pub elite_diversity: f64,
    pub restart_after: usize,
    pub objective: Objective,
}

impl Default for IntensifyOptions {
    fn default() -> Self {
        Self {
            time_limit: 300.0,
            rounds: 1000,
            workers: 1,
            batch_size: 0,
            std_similarity: false,
            barycenter: false,
            max_routes: 8,
            max_customers: 300,
            subproblem: HgsOptions {
                time_limit: 2.0,
                nb_iter: 500,
                population_size: 12,
                ..HgsOptions::default()
            },
            threshold: 0.001,
            elite_size: 8,
            elite_diversity: 0.05,
            restart_after: 30,
            objective: Objective::Distance,
        }
    }
}

#[derive(Clone, Debug)]
pub struct IntensifyResult {
    pub routes: Vec<Vec<usize>>,
    pub initial_cost: f64,
    pub best_cost: f64,
    pub rounds: usize,
    pub batches: usize,
    pub accepted: usize,
    pub improvements: usize,
    pub restarts: usize,
    pub elite_count: usize,
    pub generations: u32,
    pub elapsed_secs: f64,
}

#[derive(Clone)]
struct Plan {
    routes: Vec<Vec<usize>>,
    cost: f64,
    arcs: BTreeSet<(usize, usize)>,
}

impl Plan {
    fn new(routes: Vec<Vec<usize>>, cost: f64) -> Self {
        let arcs = routes
            .iter()
            .flat_map(|r| {
                let stops: Vec<_> = std::iter::once(0)
                    .chain(r.iter().copied())
                    .chain(std::iter::once(0))
                    .collect();
                stops.windows(2).map(|s| (s[0], s[1])).collect::<Vec<_>>()
            })
            .collect();
        Self { routes, cost, arcs }
    }
}

fn better(a: &Plan, b: &Plan, objective: Objective) -> bool {
    if objective == Objective::VehiclesThenDistance && a.routes.len() != b.routes.len() {
        return a.routes.len() < b.routes.len();
    }
    a.cost < b.cost - 1e-7
}

fn accepts(candidate: &Plan, current: &Plan, objective: Objective, threshold: f64) -> bool {
    if objective == Objective::VehiclesThenDistance
        && candidate.routes.len() != current.routes.len()
    {
        return candidate.routes.len() < current.routes.len();
    }
    candidate.cost <= current.cost + threshold * current.cost.abs() + 1e-7
}

fn add_elite(pool: &mut Vec<Plan>, candidate: &Plan, options: &IntensifyOptions) {
    // Route order does not create diversity; directed arcs do. A better member
    // can replace its similar neighbor, but a duplicate cannot fill the pool.
    let similar: Vec<_> = pool
        .iter()
        .enumerate()
        .filter_map(|(i, p)| {
            let common = p.arcs.intersection(&candidate.arcs).count();
            let union = p.arcs.len() + candidate.arcs.len() - common;
            let close = p.arcs == candidate.arcs
                || 1.0 - common as f64 / (union.max(1) as f64) < options.elite_diversity;
            close.then_some(i)
        })
        .collect();
    if similar
        .iter()
        .any(|&i| !better(candidate, &pool[i], options.objective))
    {
        return;
    }
    for &i in similar.iter().rev() {
        pool.remove(i);
    }
    pool.push(candidate.clone());
    pool.sort_by(|a, b| {
        if options.objective == Objective::VehiclesThenDistance {
            a.routes
                .len()
                .cmp(&b.routes.len())
                .then(a.cost.total_cmp(&b.cost))
        } else {
            a.cost.total_cmp(&b.cost)
        }
    });
    pool.truncate(options.elite_size);
}

/// Pick an anchor route randomly, then nearby routes by minimum cross-route
/// distance from up to four anchor customers or route barycenters. Every fifth round uses random
/// neighbors; stalled phases expand the number of routes selected.
fn select_routes(
    instance: &Instance,
    plan: &Plan,
    options: &IntensifyOptions,
    round: usize,
    stall: usize,
    rng: &mut Xoroshiro128Plus,
    excluded: &BTreeSet<usize>,
) -> Vec<usize> {
    let mut eligible: Vec<_> = (0..plan.routes.len())
        .filter(|&i| !excluded.contains(&i) && plan.routes[i].len() <= options.max_customers)
        .collect();
    if eligible.is_empty() {
        return Vec::new();
    }
    let anchor_index = rng.next() as usize % eligible.len();
    let anchor = eligible.remove(anchor_index);
    let route = &plan.routes[anchor];
    let mut neighbors = eligible;
    if round % 5 == 4 {
        rng.shuffle(&mut neighbors);
    } else {
        let center = |r: &[usize]| {
            let (x, y) = r.iter().fold((0.0, 0.0), |(x, y), &c| {
                (x + instance.customers[c].x, y + instance.customers[c].y)
            });
            (x / r.len() as f64, y / r.len() as f64)
        };
        let anchor_center = options.barycenter.then(|| center(route));
        let mut ranked: Vec<_> = neighbors
            .iter()
            .copied()
            .map(|i| {
                let distance = if options.barycenter {
                    let (ax, ay) = anchor_center.unwrap();
                    let (bx, by) = center(&plan.routes[i]);
                    (ax - bx).hypot(ay - by)
                } else {
                    (0..route.len().min(4))
                        .flat_map(|k| {
                            let a = route[k * route.len() / route.len().min(4)];
                            plan.routes[i].iter().map(move |&b| {
                                if options.std_similarity {
                                    std_dissimilarity(instance, a, b)
                                } else {
                                    instance
                                        .distance(a, b, instance.rounding)
                                        .min(instance.distance(b, a, instance.rounding))
                                }
                            })
                        })
                        .fold(f64::INFINITY, f64::min)
                };
                (i, distance)
            })
            .collect();
        ranked.sort_by(|a, b| a.1.total_cmp(&b.1).then(a.0.cmp(&b.0)));
        neighbors.clear();
        neighbors.extend(ranked.into_iter().map(|(i, _)| i));
    }
    let count = (2 + stall / 5).min(options.max_routes);
    let mut selected = vec![anchor];
    let mut customers = route.len();
    for i in neighbors {
        if selected.len() >= count {
            break;
        }
        if customers + plan.routes[i].len() <= options.max_customers {
            customers += plan.routes[i].len();
            selected.push(i);
        }
    }
    selected.sort_unstable();
    selected
}

// Equations 11–14, using travel cost as the spatial metric. Unconstrained
// instances use the limiting temporal factor of one.
pub(super) fn std_dissimilarity(instance: &Instance, a: usize, b: usize) -> f64 {
    let depot = &instance.customers[0];
    let horizon = depot.due - depot.ready;
    let directed = |i: usize, j: usize| {
        let from = &instance.customers[i];
        let to = &instance.customers[j];
        let distance = instance.distance(i, j, instance.rounding);
        let temporal =
            if horizon.is_finite() && horizon > 0.0 && from.due.is_finite() && to.due.is_finite() {
                let flexibility = to.due - (from.ready + from.service + distance);
                let waiting = (to.ready - (from.due + from.service + distance)).max(0.0);
                2.0 - (flexibility - waiting) / horizon
            } else {
                1.0
            };
        distance * (temporal + (from.demand + to.demand) / instance.capacity)
    };
    directed(a, b).min(directed(b, a))
}

fn subproblem(
    instance: &Instance,
    plan: &Plan,
    selected: &[usize],
) -> (Instance, Vec<usize>, Vec<Vec<usize>>) {
    let mut map = vec![0];
    let mut seed = Vec::new();
    for &i in selected {
        let mut route = Vec::new();
        for &customer in &plan.routes[i] {
            route.push(map.len());
            map.push(customer);
        }
        seed.push(route);
    }
    let sub = Instance {
        name: format!("{}-subset", instance.name),
        vehicles: selected.len(),
        capacity: instance.capacity,
        customers: map.iter().map(|&i| instance.customers[i].clone()).collect(),
        rounding: instance.rounding,
        distances: instance.distances.as_ref().map(|matrix| {
            map.iter()
                .map(|&i| map.iter().map(|&j| matrix[i][j]).collect())
                .collect()
        }),
    };
    (sub, map, seed)
}

fn perturb_seed(instance: &Instance, seed: &mut Vec<Vec<usize>>, rng: &mut Xoroshiro128Plus) {
    // Feasible kicks make threshold acceptance meaningful: retaining the exact
    // unperturbed seed in every inner solve would make every proposal monotone.
    for _ in 0..24 {
        let mut kicked = seed.clone();
        let a = rng.next() as usize % kicked.len();
        let b = rng.next() as usize % kicked.len();
        let i = rng.next() as usize % kicked[a].len();
        let j = rng.next() as usize % kicked[b].len();
        let value = kicked[a][i];
        kicked[a][i] = kicked[b][j];
        kicked[b][j] = value;
        if kicked != *seed && validate_plan(instance, &kicked).is_ok() {
            *seed = kicked;
            return;
        }
    }
}

struct Job {
    selected: Vec<usize>,
    instance: Instance,
    map: Vec<usize>,
    seed: Option<Vec<Vec<usize>>>,
    options: HgsOptions,
    deadline: Option<Instant>,
}

#[derive(Debug, PartialEq)]
struct JobResult {
    selected: Vec<usize>,
    routes: Option<Vec<Vec<usize>>>,
    generations: u32,
}

fn solve_job(mut job: Job) -> Result<JobResult, String> {
    if let Some(deadline) = job.deadline {
        let remaining = deadline
            .saturating_duration_since(Instant::now())
            .as_secs_f64();
        if remaining <= 0.0 {
            return Ok(JobResult {
                selected: job.selected,
                routes: None,
                generations: 0,
            });
        }
        if job.options.time_limit > 0.0 {
            job.options.time_limit = job.options.time_limit.min(remaining);
        }
    }
    let solved = if let Some(seed) = job.seed {
        run_hgs_seeded(&job.instance, &job.options, &[seed])?
    } else {
        run_hgs(&job.instance, &job.options)?
    };
    let routes =
        if solved.best_cost.is_some() && validate_plan(&job.instance, &solved.routes).is_ok() {
            Some(
                solved
                    .routes
                    .iter()
                    .map(|r| r.iter().map(|&c| job.map[c]).collect())
                    .collect(),
            )
        } else {
            None
        };
    Ok(JobResult {
        selected: job.selected,
        routes,
        generations: solved.generations,
    })
}

fn solve_jobs(jobs: Vec<Job>, pool: Option<&rayon::ThreadPool>) -> Result<Vec<JobResult>, String> {
    if let Some(pool) = pool {
        // Indexed parallel collection preserves input order. Parameters and RNG
        // are constructed inside each worker; no HGS state crosses threads.
        pool.install(|| jobs.into_par_iter().map(solve_job).collect())
    } else {
        jobs.into_iter().map(solve_job).collect()
    }
}

/// Improve a complete feasible seed, preserving the best independently replayed
/// plan. This does not fetch network data; the companion Python runner downloads
/// BKS files before invoking the executable.
pub fn intensify(
    instance: &Instance,
    seed: &[Vec<usize>],
    options: &IntensifyOptions,
) -> Result<IntensifyResult, String> {
    intensify_with_callback(instance, seed, options, |_, _| Ok(()))
}

/// As [`intensify`], calling `on_best` for the initial plan and each strict
/// improvement. A checkpoint error stops search and is returned to the caller.
pub fn intensify_with_callback(
    instance: &Instance,
    seed: &[Vec<usize>],
    options: &IntensifyOptions,
    mut on_best: impl FnMut(&[Vec<usize>], f64) -> Result<(), String>,
) -> Result<IntensifyResult, String> {
    let start = Instant::now();
    if !options.time_limit.is_finite()
        || options.time_limit < 0.0
        || options.rounds == 0
        || options.workers == 0
        || options.max_routes < 2
        || options.max_customers == 0
        || options.elite_size == 0
        || options.restart_after == 0
        || !options.threshold.is_finite()
        || options.threshold < 0.0
        || !options.elite_diversity.is_finite()
        || !(0.0..=1.0).contains(&options.elite_diversity)
        || !options.subproblem.time_limit.is_finite()
        || options.subproblem.time_limit < 0.0
        || options.subproblem.nb_iter == 0
        || options.subproblem.nb_iter > i32::MAX as u32
        || options.subproblem.population_size == 0
        || options.subproblem.population_size > i32::MAX as u32
    {
        return Err("invalid intensification limits, threshold or elite settings".into());
    }
    // Current HGS stores capacity and demands as i32; reject loss of precision.
    if !instance.capacity.is_finite()
        || instance.capacity <= 0.0
        || instance.capacity.fract() != 0.0
        || instance.capacity > i32::MAX as f64
        || instance
            .customers
            .iter()
            .any(|c| c.demand.fract() != 0.0 || c.demand > i32::MAX as f64)
        || instance.customers.iter().map(|c| c.demand).sum::<f64>() > i32::MAX as f64
    {
        return Err("HGS requires integral capacity/demands with totals fitting i32".into());
    }
    let initial_cost = validate_plan(instance, seed)?;
    let mut current = Plan::new(
        seed.iter().filter(|r| !r.is_empty()).cloned().collect(),
        initial_cost,
    );
    let mut best = current.clone();
    on_best(&best.routes, best.cost)?;
    let mut pool = vec![current.clone()];
    let workers = if options.workers > 1 {
        Some(
            rayon::ThreadPoolBuilder::new()
                .num_threads(options.workers)
                .thread_name(|i| format!("hgs-subproblem-{i}"))
                .build()
                .map_err(|e| format!("cannot create HGS workers: {e}"))?,
        )
    } else {
        None
    };
    let mut rng = Xoroshiro128Plus::new(options.subproblem.seed);
    let mut stall = 0;
    let mut accepted = 0;
    let mut improvements = 0;
    let mut restarts = 0;
    let mut rounds = 0;
    let mut batches = 0;
    let mut generations = 0u32;
    let batch_size = if options.batch_size == 0 {
        options.workers
    } else {
        options.batch_size
    };
    let deadline = if options.time_limit > 0.0 {
        Some(
            start
                .checked_add(
                    std::time::Duration::try_from_secs_f64(options.time_limit)
                        .map_err(|_| "intensification time limit is out of range")?,
                )
                .ok_or("intensification time limit is out of range")?,
        )
    } else {
        None
    };
    while rounds < options.rounds {
        if options.time_limit > 0.0 && start.elapsed().as_secs_f64() >= options.time_limit {
            break;
        }
        let mut jobs = Vec::new();
        let mut excluded = BTreeSet::new();
        for slot in 0..batch_size.min(options.rounds - rounds) {
            let round = rounds + slot;
            let selected = select_routes(
                instance, &current, options, round, stall, &mut rng, &excluded,
            );
            if selected.is_empty() {
                break;
            }
            excluded.extend(selected.iter().copied());
            let (mut sub, map, mut sub_seed) = subproblem(instance, &current, &selected);
            let mut hgs = options.subproblem.clone();
            hgs.seed = rng.next() as i32;
            if options.time_limit > 0.0 {
                let remaining = options.time_limit - start.elapsed().as_secs_f64();
                if remaining <= 0.0 {
                    break;
                }
                if hgs.time_limit > 0.0 {
                    hgs.time_limit = hgs.time_limit.min(remaining);
                }
            }
            let reduce_fleet = options.objective == Objective::VehiclesThenDistance
                && selected.len() > 1
                && round % 4 == 3
                && sub.customers.iter().map(|c| c.demand).sum::<f64>()
                    <= (sub.vehicles - 1) as f64 * sub.capacity;
            let seed = if reduce_fleet {
                sub.vehicles -= 1;
                None
            } else {
                if round % 2 == 1 {
                    perturb_seed(&sub, &mut sub_seed, &mut rng);
                }
                Some(sub_seed)
            };
            jobs.push(Job {
                selected,
                instance: sub,
                map,
                seed,
                options: hgs,
                deadline,
            });
        }
        if jobs.is_empty() {
            break;
        }
        let results = solve_jobs(jobs, workers.as_ref())?;
        rounds += results.len();
        batches += 1;
        stall += results.len();
        let mut replaced = BTreeSet::new();
        let mut replacements = Vec::new();
        for result in results {
            generations = generations.saturating_add(result.generations);
            if let Some(routes) = result.routes {
                replaced.extend(result.selected);
                replacements.extend(routes);
            }
        }
        if !replaced.is_empty() {
            let mut routes: Vec<_> = current
                .routes
                .iter()
                .enumerate()
                .filter(|(i, _)| !replaced.contains(i))
                .map(|(_, r)| r.clone())
                .collect();
            routes.extend(replacements);
            if let Ok(cost) = validate_plan(instance, &routes) {
                let candidate = Plan::new(routes, cost);
                add_elite(&mut pool, &candidate, options);
                if better(&candidate, &best, options.objective) {
                    best = candidate.clone();
                    if options.objective == Objective::VehiclesThenDistance {
                        pool.retain(|p| p.routes.len() <= best.routes.len());
                    }
                    improvements += 1;
                    on_best(&best.routes, best.cost)?;
                    stall = 0;
                    if options.subproblem.verbose {
                        crate::log_err!(
                            "Intensify round {}: {} routes, cost {:.6}",
                            rounds,
                            best.routes.len(),
                            best.cost
                        );
                    }
                }
                let cooling =
                    1.0 - stall.min(options.restart_after) as f64 / options.restart_after as f64;
                if accepts(
                    &candidate,
                    &current,
                    options.objective,
                    options.threshold * cooling,
                ) {
                    current = candidate;
                    accepted += 1;
                }
            }
        }
        if stall >= options.restart_after {
            current = pool[rng.next() as usize % pool.len()].clone();
            stall = 0;
            restarts += 1;
        }
    }
    Ok(IntensifyResult {
        routes: best.routes,
        initial_cost,
        best_cost: best.cost,
        rounds,
        batches,
        accepted,
        improvements,
        restarts,
        elite_count: pool.len(),
        generations,
        elapsed_secs: start.elapsed().as_secs_f64(),
    })
}

#[cfg(test)]
#[path = "../../../tests/unit/extensions/hgs/intensify.rs"]
mod tests;
