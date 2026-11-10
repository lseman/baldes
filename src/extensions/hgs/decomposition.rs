//! Repeated spatial decomposition of complete incumbent routes.
use super::intensify::Objective;
use super::{run_hgs, run_hgs_seeded, solution::validate_plan, HgsOptions, HgsResult};
use crate::inputs::Instance;
use std::time::{Duration, Instant};

/// Options for seeded, route-preserving decomposition. Ordinary HGS is unchanged.
#[derive(Clone, Debug)]
pub struct RouteDecompositionOptions {
    /// Shared wall-clock budget, including initialization and every subproblem.
    pub time_limit: f64,
    /// Soft customer limit per group: a complete route is never split.
    pub target_customers: usize,
    /// Maximum number of regrouping phases.
    pub phases: usize,
    /// Zero uses constant depth; otherwise run `phases` at each level 1..=N (N <= 6).
    pub exploration_levels: usize,
    /// Educate the whole merged plan after each completed phase.
    pub master_local_search: bool,
    /// Fraction of each phase share reserved for master education (0 < x < 1).
    pub master_fraction: f64,
    pub objective: Objective,
    /// Fraction of the budget used to find an initial plan when no seed is given.
    pub initialization_fraction: f64,
    /// Inner HGS controls. Its time limit is replaced by each job's budget.
    pub hgs: HgsOptions,
}
impl Default for RouteDecompositionOptions {
    fn default() -> Self {
        Self {
            time_limit: 60.0,
            target_customers: 200,
            phases: 6,
            exploration_levels: 0,
            master_local_search: false,
            master_fraction: 0.15,
            objective: Objective::Distance,
            initialization_fraction: 0.2,
            hgs: HgsOptions::default(),
        }
    }
}

/// Observations from the decomposition run; group sizes can exceed the target.
#[derive(Clone, Debug, Default)]
pub struct RouteDecompositionStats {
    pub phases_started: usize,
    pub phases_completed: usize,
    pub subproblems: usize,
    pub improvements: usize,
    pub largest_group: usize,
    pub exploration_schedule: Vec<usize>,
    pub master_passes: usize,
    pub master_improvements: usize,
}
#[derive(Clone, Debug)]
pub struct RouteDecompositionResult {
    pub solution: HgsResult,
    pub decomposition: RouteDecompositionStats,
}

/// Improve a complete feasible plan by repeatedly regrouping whole routes.
///
/// With no seed, ordinary HGS receives `initialization_fraction` of the shared
/// budget. Each group inherits its original routes as a feasible fallback and
/// may use the fleet slots left free by the rest of the incumbent. Only strictly
/// cheaper, fully replayed global plans are accepted. The objective is distance,
/// not a vehicles-first objective. A zero instance fleet means unrestricted
/// fleet (at most one vehicle per customer). Encountered columns are not collected.
///
/// The deadline is checked between jobs; an indivisible inner HGS operation or
/// input preparation may overrun it. A supplied seed survives budget exhaustion.
pub fn run_hgs_decomposed(
    instance: &Instance,
    seed: Option<&[Vec<usize>]>,
    options: &RouteDecompositionOptions,
) -> Result<RouteDecompositionResult, String> {
    let start = Instant::now();
    validate_input(instance, options)?;
    let deadline = start
        .checked_add(
            Duration::try_from_secs_f64(options.time_limit)
                .map_err(|_| "invalid decomposition time limit")?,
        )
        .ok_or("invalid decomposition time limit")?;
    let mut free_fleet;
    let instance = if instance.vehicles == 0 {
        free_fleet = instance.clone();
        free_fleet.vehicles = instance.customers.len() - 1;
        &free_fleet
    } else {
        instance
    };
    let mut generations = 0u32;
    let mut routes = if let Some(seed) = seed {
        validate_plan(instance, seed)?;
        seed.iter().filter(|r| !r.is_empty()).cloned().collect()
    } else {
        let mut hgs = options.hgs.clone();
        hgs.time_limit =
            remaining(deadline).min(options.time_limit * options.initialization_fraction);
        if hgs.time_limit <= 0.0 {
            return Err("decomposition budget expired before initialization".into());
        }
        let initial = run_hgs(instance, &hgs)?;
        generations = initial.generations;
        if initial.best_cost.is_none() {
            return Err("decomposition initialization found no feasible plan; supply a seed or increase the budget".into());
        }
        initial.routes
    };
    let mut cost = validate_plan(instance, &routes)?;
    let mut stats = RouteDecompositionStats::default();
    let levels = options.exploration_levels.max(1);
    let total_phases = options
        .phases
        .checked_mul(levels)
        .ok_or("too many decomposition phases")?;
    for phase in 0..total_phases {
        if remaining(deadline) <= 0.0 {
            break;
        }
        let base = routes.clone();
        let groups = route_groups(
            instance,
            &base,
            options.target_customers,
            phase,
            options.hgs.seed,
        );
        stats.phases_started += 1;
        let level = if options.exploration_levels == 0 {
            0
        } else {
            phase / options.phases + 1
        };
        stats.exploration_schedule.push(level);
        let master_budget = if options.master_local_search {
            remaining(deadline) / (total_phases - phase) as f64 * options.master_fraction
        } else {
            0.
        };
        // Every accepted partial rebuild is a complete incumbent; unsolved groups
        // retain their exact routes, even for non-metric directed matrices.
        let mut replacements: Vec<Option<Vec<Vec<usize>>>> = vec![None; groups.len()];
        let mut complete = true;
        for (index, group) in groups.iter().enumerate() {
            let left = remaining(deadline);
            if left <= 0.0 {
                complete = false;
                break;
            }
            let customers: Vec<_> = group
                .iter()
                .flat_map(|&r| base[r].iter().copied())
                .collect();
            stats.largest_group = stats.largest_group.max(customers.len());
            let original: Vec<_> = group.iter().map(|&r| base[r].clone()).collect();
            let fleet = if options.objective == Objective::VehiclesThenDistance {
                original.len()
            } else {
                instance.vehicles - (routes.len() - original.len())
            };
            let (sub, map, projected) = subproblem(instance, &original, &customers, fleet);
            let mut hgs = exploration_options(&options.hgs, level);
            // Reserve a fair share for every remaining job and phase. No job
            // receives a fresh copy of the outer time budget.
            let jobs = (groups.len() - index)
                .saturating_add(groups.len().saturating_mul(total_phases - phase - 1));
            let available = (remaining(deadline) - master_budget).max(0.);
            if available <= 0.0 {
                complete = false;
                break;
            }
            hgs.time_limit = (available.min(left) / jobs as f64).max(f64::MIN_POSITIVE);
            hgs.seed = hgs
                .seed
                .wrapping_add(phase as i32)
                .wrapping_mul(31)
                .wrapping_add(index as i32);
            let result = run_hgs_seeded(&sub, &hgs, &[projected])?;
            stats.subproblems += 1;
            generations = generations.saturating_add(result.generations);
            if result.best_cost.is_none() || validate_plan(&sub, &result.routes).is_err() {
                continue;
            }
            let mapped: Vec<Vec<usize>> = result
                .routes
                .iter()
                .map(|r| r.iter().map(|&i| map[i]).collect())
                .collect();
            replacements[index] = Some(mapped);
            let candidate = rebuild(&base, &groups, &replacements);
            match validate_plan(instance, &candidate) {
                // Ignore summation-order noise when regrouping the same arcs.
                Ok(next)
                    if improves(
                        (candidate.len(), next),
                        (routes.len(), cost),
                        options.objective,
                    ) =>
                {
                    cost = next;
                    routes = candidate;
                    stats.improvements += 1;
                }
                _ => {
                    replacements[index] = None;
                }
            }
        }
        if complete {
            stats.phases_completed += 1;
        }
        let budget = master_budget.min(remaining(deadline));
        if options.master_local_search && budget > 0. {
            let mut controls = options.hgs.clone();
            controls.time_limit = budget;
            controls.seed = controls.seed.wrapping_add(phase as i32);
            // Vehicles-first education cannot use additional vehicles. Limit
            // the active fleet before allocating the master insertion caches.
            let mut constrained;
            let master = if options.objective == Objective::VehiclesThenDistance {
                constrained = instance.clone();
                constrained.vehicles = routes.len();
                &constrained
            } else {
                instance
            };
            let candidate = educate_master(master, &routes, &controls)?;
            stats.master_passes += 1;
            let next = validate_plan(instance, &candidate)?;
            if improves(
                (candidate.len(), next),
                (routes.len(), cost),
                options.objective,
            ) {
                routes = candidate;
                cost = next;
                stats.master_improvements += 1;
                stats.improvements += 1;
            }
        }
    }
    // Independently replay the original customer IDs before returning.
    let best_cost = Some(validate_plan(instance, &routes)?);
    Ok(RouteDecompositionResult {
        solution: HgsResult {
            routes,
            columns: Vec::new(),
            best_cost,
            generations,
            elapsed_secs: start.elapsed().as_secs_f64(),
        },
        decomposition: stats,
    })
}
fn improves(new: (usize, f64), old: (usize, f64), objective: Objective) -> bool {
    if objective == Objective::VehiclesThenDistance && new.0 != old.0 {
        return new.0 < old.0;
    }
    new.1 < old.1 - 1e-10 * old.1.abs().max(new.1.abs()).max(1.)
}

fn exploration_options(base: &HgsOptions, level: usize) -> HgsOptions {
    let mut hgs = base.clone();
    if level > 0 {
        let (population, stagnation, total) = [
            (2, 1, 0),
            (3, 40, 150),
            (7, 100, 400),
            (15, 200, 2000),
            (25, 2000, 20000),
            (25, 5000, 50000),
        ][level - 1];
        hgs.population_size = base.population_size.min(population);
        hgs.nb_iter = base.nb_iter.min(stagnation);
        hgs.generation_limit = Some(base.generation_limit.map_or(total, |cap| cap.min(total)));
    }
    hgs
}

/// Master education has no population. Roll back infeasible or worse proposals.
fn educate_master(
    instance: &Instance,
    routes: &[Vec<usize>],
    options: &HgsOptions,
) -> Result<Vec<Vec<usize>>, String> {
    use super::{local_search::HGSLocalSearch, Individual, Params};
    let mut params = Params::from_instance(instance);
    params.config.time_limit = options.time_limit;
    params.config.local_search = options.local_search.clone();
    params.set_seed(options.seed);
    params.reset_clock();
    let mut individual = Individual::new_empty(&params);
    for (target, source) in individual.chrom_r.iter_mut().zip(routes) {
        target.clone_from(source);
    }
    individual.chrom_t = routes.iter().flatten().copied().collect();
    individual.evaluate_complete_cost();
    let mut local = HGSLocalSearch::new(&params);
    for factor in [1., 5., 20., 100.] {
        if params.is_time_limit_exceeded() {
            break;
        }
        local.run(
            &mut individual,
            params.penalty_capacity.get() * factor,
            params.penalty_time_warp.get() * factor,
        );
        if individual.is_feasible {
            let candidate: Vec<_> = individual
                .chrom_r
                .iter()
                .filter(|r| !r.is_empty())
                .cloned()
                .collect();
            if validate_plan(instance, &candidate).is_ok() {
                return Ok(candidate);
            }
        }
    }
    Ok(routes.to_vec())
}

fn remaining(deadline: Instant) -> f64 {
    deadline
        .saturating_duration_since(Instant::now())
        .as_secs_f64()
}

fn validate_input(instance: &Instance, options: &RouteDecompositionOptions) -> Result<(), String> {
    let hgs = &options.hgs;
    if !options.time_limit.is_finite()
        || options.time_limit <= 0.0
        || options.phases == 0
        || options.exploration_levels > 6
        || options
            .phases
            .checked_mul(options.exploration_levels.max(1))
            .is_none()
        || !options.master_fraction.is_finite()
        || options.master_fraction <= 0.
        || options.master_fraction >= 1.
        || options.target_customers == 0
        || !options.initialization_fraction.is_finite()
        || options.initialization_fraction <= 0.0
        || options.initialization_fraction >= 1.0
        || !hgs.local_search.valid()
        || hgs.nb_iter == 0
        || hgs.nb_iter > i32::MAX as u32
        || hgs.population_size == 0
        || hgs.population_size > i32::MAX as u32
        || hgs.compression.as_ref().is_some_and(|c| c.interval == 0)
    {
        return Err("invalid route decomposition options".into());
    }
    let n = instance.customers.len();
    if n < 2
        || n > i32::MAX as usize
        || instance.vehicles > i32::MAX as usize
        || !instance.capacity.is_finite()
        || instance.capacity <= 0.0
        || instance.capacity.fract() != 0.0
        || instance.capacity > i32::MAX as f64
        || instance.customers.iter().any(|c| {
            !c.x.is_finite()
                || !c.y.is_finite()
                || !c.ready.is_finite()
                || c.due.is_nan()
                || c.due < c.ready
                || !c.service.is_finite()
                || c.service < 0.0
                || !c.demand.is_finite()
                || c.demand < 0.0
                || c.demand.fract() != 0.0
                || c.demand > i32::MAX as f64
        })
        || instance.customers.iter().map(|c| c.demand).sum::<f64>() > i32::MAX as f64
        || instance.distances.as_ref().is_some_and(|m| {
            m.len() != n
                || m.iter()
                    .any(|r| r.len() != n || r.iter().any(|d| !d.is_finite() || *d < 0.0))
        })
    {
        return Err("route decomposition requires valid integral demands/capacity and a square distance matrix".into());
    }
    Ok(())
}

// Reference policy: shuffled route anchors, then let the smallest group
// take its nearest unassigned route. The target controls average group size;
// individual routes and the final assignment can exceed it.
fn route_groups(
    instance: &Instance,
    routes: &[Vec<usize>],
    target: usize,
    phase: usize,
    seed: i32,
) -> Vec<Vec<usize>> {
    if routes.is_empty() {
        return Vec::new();
    }
    let centers: Vec<(f64, f64)> = routes
        .iter()
        .map(|r| {
            r.iter().fold((0.0, 0.0), |(x, y), &c| {
                (
                    x + instance.customers[c].x / r.len() as f64,
                    y + instance.customers[c].y / r.len() as f64,
                )
            })
        })
        .collect();
    let count = ((routes.iter().map(Vec::len).sum::<usize>() as f64 / target as f64).round()
        as usize)
        .clamp(1, routes.len());
    let mut order: Vec<_> = (0..routes.len()).collect();
    let mut rng = super::DefaultRng::new(seed.wrapping_add(phase as i32));
    for i in (1..order.len()).rev() {
        order.swap(i, (rng.next() % (i + 1) as u64) as usize);
    }
    let mut used = vec![false; routes.len()];
    let mut groups: Vec<Vec<usize>> = order[..count]
        .iter()
        .map(|&r| {
            used[r] = true;
            vec![r]
        })
        .collect();
    let mut sizes: Vec<_> = groups.iter().map(|g| routes[g[0]].len()).collect();
    let candidates: Vec<Vec<usize>> = groups
        .iter()
        .map(|g| {
            let anchor = g[0];
            let mut nearest: Vec<_> = (0..routes.len()).filter(|&r| r != anchor).collect();
            let distance = |r: usize| {
                (centers[r].0 - centers[anchor].0).hypot(centers[r].1 - centers[anchor].1)
            };
            nearest.sort_by(|&a, &b| distance(a).total_cmp(&distance(b)).then(a.cmp(&b)));
            nearest
        })
        .collect();
    let mut positions = vec![0; count];
    for _ in count..routes.len() {
        let group = (0..count).min_by_key(|&g| (sizes[g], g)).unwrap();
        while used[candidates[group][positions[group]]] {
            positions[group] += 1;
        }
        let route = candidates[group][positions[group]];
        used[route] = true;
        groups[group].push(route);
        sizes[group] += routes[route].len();
    }
    groups
}

fn subproblem(
    instance: &Instance,
    routes: &[Vec<usize>],
    customers: &[usize],
    fleet: usize,
) -> (Instance, Vec<usize>, Vec<Vec<usize>>) {
    let (mut sub, map, projected) =
        crate::extensions::common::route_subproblem(instance, routes.iter().map(Vec::as_slice));
    sub.vehicles = fleet.min(customers.len());
    (sub, map, projected)
}

fn rebuild(
    base: &[Vec<usize>],
    groups: &[Vec<usize>],
    replacements: &[Option<Vec<Vec<usize>>>],
) -> Vec<Vec<usize>> {
    groups
        .iter()
        .zip(replacements)
        .flat_map(|(group, replacement)| {
            replacement
                .clone()
                .unwrap_or_else(|| group.iter().map(|&r| base[r].clone()).collect())
        })
        .collect()
}

#[cfg(test)]
#[path = "../../../tests/unit/extensions/hgs/decomposition.rs"]
mod tests;
