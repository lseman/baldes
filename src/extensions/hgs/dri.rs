//! Customer clustering and cross-cluster improvement for large routing instances.
//!
//! The clustering uses the symmetric spatial-temporal-demand dissimilarity from
//! Kerscher and Minner (2025), equations 11–14. The bounded fuzzy c-medoids variant and
//! cross-cluster improvement are described in the extension guide.
use super::intensify::std_dissimilarity;
use super::solution::{validate_plan, validate_plan_relaxed_fleet};
use super::{run_hgs, run_hgs_seeded, HgsOptions, HgsResult};
use crate::inputs::Instance;
use rayon::prelude::*;
use std::time::{Duration, Instant};

/// Options for customer clustering and routing. Each cluster contains at most
/// `cluster_size` customers, bounding the quadratic HGS data structures.
#[derive(Clone, Debug)]
pub struct DriOptions {
    pub time_limit: f64,
    pub cluster_size: usize,
    pub workers: usize,
    pub vicinity: usize,
    pub customer_neighbors: usize,
    pub clustering_iterations: usize,
    /// Fuzziness exponent κ; must be finite and greater than one.
    pub fuzziness: f64,
    pub routing_fraction: f64,
    /// Customers with maximum fuzzy membership at most this threshold may move.
    pub fuzzy_threshold: f64,
    pub hgs: HgsOptions,
}
impl Default for DriOptions {
    fn default() -> Self {
        Self {
            time_limit: 60.0,
            cluster_size: 200,
            workers: 1,
            vicinity: 3,
            customer_neighbors: 20,
            clustering_iterations: 5,
            fuzziness: 2.0,
            routing_fraction: 0.8,
            fuzzy_threshold: 0.8,
            hgs: HgsOptions::default(),
        }
    }
}

struct Clustering {
    medoids: Vec<usize>,
    clusters: Vec<Vec<usize>>,
    assignment: Vec<usize>,
    memberships: Vec<Vec<f64>>,
}

type RoutedCluster = (Vec<Vec<usize>>, u32);

// Fuzzy c-medoids with kappa=2 (paper equation 18). Farthest-first
// initialization and bounded candidate sampling make the method predictable
// and limit work; weighted medoid updates use the fuzzy objective (19).
fn cluster(
    instance: &Instance,
    limit: usize,
    iterations: usize,
    fuzziness: f64,
    deadline: Instant,
) -> Result<Clustering, String> {
    let n = instance.customers.len() - 1;
    let k = n / limit + usize::from(n % limit != 0);
    if (n + 1)
        .checked_mul(k)
        .map_or(true, |size| size > 50_000_000)
    {
        return Err("DRI membership matrix is too large; increase --subproblem-customers".into());
    }
    let mut medoids = vec![1];
    let mut nearest = vec![f64::INFINITY; n + 1];
    while medoids.len() < k {
        if Instant::now() >= deadline {
            return Err("DRI clustering time limit exceeded".into());
        }
        let last = *medoids.last().unwrap();
        for (i, value) in nearest.iter_mut().enumerate().skip(1) {
            *value = value.min(std_dissimilarity(instance, i, last).max(0.0));
        }
        let next = (1..=n)
            .filter(|i| !medoids.contains(i))
            .max_by(|&a, &b| nearest[a].total_cmp(&nearest[b]).then_with(|| b.cmp(&a)))
            .unwrap();
        medoids.push(next);
    }
    let mut memberships = vec![vec![0.0; k]; n + 1];
    let update_memberships =
        |medoids: &[usize], memberships: &mut Vec<Vec<f64>>| -> Result<(), String> {
            for (i, row) in memberships.iter_mut().enumerate().skip(1) {
                if i % 256 == 0 && Instant::now() >= deadline {
                    return Err("DRI clustering time limit exceeded".into());
                }
                let distances: Vec<_> = medoids
                    .iter()
                    .map(|&m| std_dissimilarity(instance, i, m))
                    .collect();
                if distances.iter().any(|d| !d.is_finite()) {
                    return Err("DRI similarity produced a non-finite value".into());
                }
                let distances: Vec<_> = distances.into_iter().map(|d| d.max(0.0)).collect();
                let zeros = distances.iter().filter(|&&d| d <= 1e-12).count();
                if zeros > 0 {
                    for (u, &d) in row.iter_mut().zip(&distances) {
                        *u = if d <= 1e-12 { 1.0 / zeros as f64 } else { 0.0 };
                    }
                } else {
                    let min = distances.iter().copied().fold(f64::INFINITY, f64::min);
                    let weights: Vec<_> = distances
                        .iter()
                        .map(|&d| (min / d).powf(2.0 / (fuzziness - 1.0)))
                        .collect();
                    let sum: f64 = weights.iter().sum();
                    for (u, &w) in row.iter_mut().zip(&weights) {
                        *u = w / sum;
                    }
                }
            }
            Ok(())
        };
    let mut assignment = vec![usize::MAX; n + 1];
    let mut clusters = vec![Vec::new(); k];
    for _ in 0..iterations {
        update_memberships(&medoids, &mut memberships)?;
        clusters.iter_mut().for_each(Vec::clear);
        for (p, &m) in medoids.iter().enumerate() {
            clusters[p].push(m);
            assignment[m] = p;
        }
        for i in 1..=n {
            if medoids.contains(&i) {
                continue;
            }
            let p = (0..k)
                .filter(|&p| clusters[p].len() < limit)
                .max_by(|&a, &b| {
                    memberships[i][a]
                        .total_cmp(&memberships[i][b])
                        .then_with(|| b.cmp(&a))
                })
                .unwrap();
            clusters[p].push(i);
            assignment[i] = p;
        }
        let mut next = Vec::with_capacity(k);
        for (p, candidates) in clusters.iter().enumerate() {
            let mut best = (f64::INFINITY, medoids[p]);
            // Deterministic spread of at most 16 candidates in each cluster.
            for j in 0..candidates.len().min(16) {
                if Instant::now() >= deadline {
                    return Err("DRI clustering time limit exceeded".into());
                }
                let candidate = candidates[j * candidates.len() / candidates.len().min(16)];
                let score = (1..=n)
                    .map(|i| {
                        memberships[i][p].powf(fuzziness)
                            * std_dissimilarity(instance, i, candidate).max(0.0)
                    })
                    .sum::<f64>();
                if score.total_cmp(&best.0).is_lt() || (score == best.0 && candidate < best.1) {
                    best = (score, candidate);
                }
            }
            next.push(best.1);
        }
        if next == medoids {
            break;
        }
        medoids = next;
    }
    // Recompute memberships and the hard partition against the final medoids.
    update_memberships(&medoids, &mut memberships)?;
    clusters.iter_mut().for_each(Vec::clear);
    for (p, &m) in medoids.iter().enumerate() {
        clusters[p].push(m);
        assignment[m] = p;
    }
    for i in 1..=n {
        if i % 256 == 0 && Instant::now() >= deadline {
            return Err("DRI clustering time limit exceeded".into());
        }
        if medoids.contains(&i) {
            continue;
        }
        let p = (0..k)
            .filter(|&p| clusters[p].len() < limit)
            .max_by(|&a, &b| {
                memberships[i][a]
                    .total_cmp(&memberships[i][b])
                    .then_with(|| b.cmp(&a))
            })
            .unwrap();
        clusters[p].push(i);
        assignment[i] = p;
    }
    Ok(Clustering {
        medoids,
        clusters,
        assignment,
        memberships,
    })
}

fn subset(instance: &Instance, customers: &[usize]) -> (Instance, Vec<usize>) {
    let map: Vec<_> = std::iter::once(0)
        .chain(customers.iter().copied())
        .collect();
    let sub = Instance {
        name: format!("{}-cluster", instance.name),
        vehicles: customers.len(),
        capacity: instance.capacity,
        customers: map.iter().map(|&i| instance.customers[i].clone()).collect(),
        rounding: instance.rounding,
        distances: instance.distances.as_ref().map(|m| {
            map.iter()
                .map(|&i| map.iter().map(|&j| m[i][j]).collect())
                .collect()
        }),
    };
    (sub, map)
}

/// Cluster customers, route subproblems independently, and improve the merged
/// plan using cross-cluster relocations. A supplied incumbent is preserved when
/// clustering cannot produce a feasible plan or improve it.
pub fn run_dri(
    instance: &Instance,
    seed: Option<&[Vec<usize>]>,
    options: &DriOptions,
) -> Result<HgsResult, String> {
    let start = Instant::now();
    if !options.time_limit.is_finite()
        || options.time_limit <= 0.0
        || options.cluster_size == 0
        || options.workers == 0
        || options.clustering_iterations == 0
        || !options.fuzziness.is_finite()
        || options.fuzziness <= 1.0
        || options.hgs.nb_iter == 0
        || options.hgs.nb_iter > i32::MAX as u32
        || options.hgs.population_size == 0
        || options.hgs.population_size > i32::MAX as u32
        || options.vicinity == 0
        || options.customer_neighbors == 0
        || !options.fuzzy_threshold.is_finite()
        || !(0.0..=1.0).contains(&options.fuzzy_threshold)
        || !options.routing_fraction.is_finite()
        || options.routing_fraction <= 0.0
        || !(0.0..1.0).contains(&options.routing_fraction)
    {
        return Err("invalid DRI options".into());
    }
    let mut best = if let Some(seed) = seed {
        Some((
            validate_plan(instance, seed)?,
            seed.iter()
                .filter(|r| !r.is_empty())
                .cloned()
                .collect::<Vec<_>>(),
        ))
    } else {
        None
    };
    // Validate basic input even when no full incumbent is supplied.
    if instance.customers.len() < 2
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
            m.len() != instance.customers.len()
                || m.iter()
                    .any(|r| r.len() != m.len() || r.iter().any(|d| !d.is_finite() || *d < 0.0))
        })
    {
        return Err(
            "DRI requires valid integral capacity and demand and a square distance matrix".into(),
        );
    }
    let deadline = start
        .checked_add(
            Duration::try_from_secs_f64(options.time_limit)
                .map_err(|_| "invalid DRI time limit")?,
        )
        .ok_or("invalid DRI time limit")?;
    let clusters = match cluster(
        instance,
        options.cluster_size,
        options.clustering_iterations,
        options.fuzziness,
        deadline,
    ) {
        Ok(value) => value,
        Err(error) => {
            let (cost, routes) = best.ok_or(error)?;
            return Ok(HgsResult {
                routes,
                columns: Vec::new(),
                best_cost: Some(cost),
                generations: 0,
                elapsed_secs: start.elapsed().as_secs_f64(),
            });
        }
    };
    let remaining = deadline
        .saturating_duration_since(Instant::now())
        .as_secs_f64();
    let routing_deadline = Instant::now()
        .checked_add(Duration::from_secs_f64(
            remaining * options.routing_fraction,
        ))
        .unwrap_or(deadline);
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(options.workers)
        .build()
        .map_err(|e| e.to_string())?;
    let results: Vec<Result<RoutedCluster, String>> = pool.install(|| {
        clusters
            .clusters
            .par_iter()
            .enumerate()
            .map(|(index, customers)| {
                let left = routing_deadline
                    .saturating_duration_since(Instant::now())
                    .as_secs_f64();
                if left <= 0.0 {
                    return Err("DRI routing budget expired before every cluster was solved".into());
                }
                let (sub, map) = subset(instance, customers);
                let mut hgs = options.hgs.clone();
                // Jobs are dispatched to a bounded worker pool; cap each job by the
                // remaining wall budget rather than allocating a full budget per job.
                hgs.time_limit = left
                    .min(
                        remaining * options.routing_fraction * customers.len() as f64
                            / (instance.customers.len() - 1) as f64
                            * options.workers as f64,
                    )
                    .max(1e-9);
                hgs.seed = options.hgs.seed.wrapping_add(index as i32);
                // Project the supplied incumbent onto this customer cluster.
                // Removing stops can violate time windows for non-metric travel
                // matrices, so only seed HGS after replaying the projection.
                let projected = seed.and_then(|plan| {
                    let mut inverse = vec![0usize; instance.customers.len()];
                    for (local, &global) in map.iter().enumerate().skip(1) {
                        inverse[global] = local;
                    }
                    let routes: Vec<Vec<usize>> = plan
                        .iter()
                        .map(|r| {
                            r.iter()
                                .filter_map(|&c| {
                                    let local = inverse[c];
                                    (local != 0).then_some(local)
                                })
                                .collect::<Vec<_>>()
                        })
                        .filter(|r| !r.is_empty())
                        .collect();
                    validate_plan(&sub, &routes).ok().map(|_| routes)
                });
                let result = match projected {
                    Some(routes) => run_hgs_seeded(&sub, &hgs, &[routes])?,
                    None => run_hgs(&sub, &hgs)?,
                };
                validate_plan(&sub, &result.routes)?;
                Ok((
                    result
                        .routes
                        .iter()
                        .map(|r| r.iter().map(|&i| map[i]).collect())
                        .collect(),
                    result.generations,
                ))
            })
            .collect()
    });
    let mut routes = Vec::new();
    let mut generations = 0u32;
    let mut failed = false;
    for result in results {
        match result {
            Ok((r, g)) => {
                routes.extend(r);
                generations = generations.saturating_add(g);
            }
            Err(_) => failed = true,
        }
    }
    if !failed {
        if let Ok(cost) = validate_plan(instance, &routes) {
            if best.as_ref().map_or(true, |(c, _)| cost < *c - 1e-7) {
                best = Some((cost, routes.clone()));
            }
        }
        // Improve even when independently solved clusters use too many
        // vehicles; route elimination can restore feasibility before merging.
        improve(instance, &clusters, &mut routes, options, deadline);
        if let Ok(cost) = validate_plan(instance, &routes) {
            if best.as_ref().map_or(true, |(c, _)| cost < *c - 1e-7) {
                best = Some((cost, routes));
            }
        }
    }
    let (cost, routes) = best.ok_or(
        "DRI could not produce a feasible complete solution within the fleet and time limits",
    )?;
    Ok(HgsResult {
        routes,
        columns: Vec::new(),
        best_cost: Some(cost),
        generations,
        elapsed_secs: start.elapsed().as_secs_f64(),
    })
}

#[cfg(test)]
#[path = "../../../tests/unit/extensions/hgs/dri.rs"]
mod tests;

// First-descent relocations restricted to nearest cluster medoids and nearest
// customer vertices. Replaying complete candidates enforces the original fleet,
// capacity, windows, service and exact distance convention.
fn improve(
    instance: &Instance,
    clusters: &Clustering,
    routes: &mut Vec<Vec<usize>>,
    options: &DriOptions,
    deadline: Instant,
) {
    let k = clusters.medoids.len();
    let mut vicinity = vec![Vec::new(); k];
    for (p, list) in vicinity.iter_mut().enumerate() {
        *list = (0..k).filter(|&q| q != p).collect();
        list.sort_by(|&a, &b| {
            std_dissimilarity(instance, clusters.medoids[p], clusters.medoids[a])
                .total_cmp(&std_dissimilarity(
                    instance,
                    clusters.medoids[p],
                    clusters.medoids[b],
                ))
                .then(a.cmp(&b))
        });
        list.truncate(options.vicinity);
    }
    let mut cost = match validate_plan_relaxed_fleet(instance, routes) {
        Ok(c) => c,
        Err(_) => return,
    };
    loop {
        let mut accepted = false;
        'search: for a in 0..routes.len() {
            for pos in 0..routes[a].len() {
                if Instant::now() >= deadline {
                    return;
                }
                let customer = routes[a][pos];
                let source = clusters.assignment[customer];
                if clusters.memberships[customer]
                    .iter()
                    .copied()
                    .fold(0.0, f64::max)
                    > options.fuzzy_threshold
                {
                    continue;
                }
                // Single-route 2-opt complements the inter-route tail exchange.
                for end in (pos + 1)..routes[a].len() {
                    if Instant::now() >= deadline {
                        return;
                    }
                    let mut candidate = routes.clone();
                    candidate[a][pos..=end].reverse();
                    if let Ok(next) = validate_plan_relaxed_fleet(instance, &candidate) {
                        if improves_repair(instance, routes, cost, &candidate, next) {
                            *routes = candidate;
                            cost = next;
                            accepted = true;
                            break 'search;
                        }
                    }
                }
                let mut neighbors: Vec<_> = (1..instance.customers.len())
                    .filter(|&i| i != customer)
                    .collect();
                neighbors.sort_by(|&x, &y| {
                    std_dissimilarity(instance, customer, x)
                        .total_cmp(&std_dissimilarity(instance, customer, y))
                        .then(x.cmp(&y))
                });
                neighbors.truncate(options.customer_neighbors);
                for b in 0..routes.len() {
                    if a == b {
                        continue;
                    }
                    for target_pos in 0..routes[b].len() {
                        if Instant::now() >= deadline {
                            return;
                        }
                        let target = routes[b][target_pos];
                        if !vicinity[source].contains(&clusters.assignment[target])
                            || !neighbors.contains(&target)
                        {
                            continue;
                        }
                        for insertion in [target_pos, target_pos + 1] {
                            let mut candidate = routes.clone();
                            candidate[a].remove(pos);
                            candidate[b].insert(insertion, customer);
                            candidate.retain(|r| !r.is_empty());
                            if let Ok(next) = validate_plan_relaxed_fleet(instance, &candidate) {
                                if improves_repair(instance, routes, cost, &candidate, next) {
                                    *routes = candidate;
                                    cost = next;
                                    accepted = true;
                                    break 'search;
                                }
                            }
                        }
                        let mut candidate = routes.clone();
                        candidate[a][pos] = target;
                        candidate[b][target_pos] = customer;
                        if let Ok(next) = validate_plan_relaxed_fleet(instance, &candidate) {
                            if improves_repair(instance, routes, cost, &candidate, next) {
                                *routes = candidate;
                                cost = next;
                                accepted = true;
                                break 'search;
                            }
                        }
                        let mut candidate = routes.clone();
                        let tail_a = candidate[a].split_off(pos + 1);
                        let tail_b = candidate[b].split_off(target_pos + 1);
                        candidate[a].extend(tail_b);
                        candidate[b].extend(tail_a);
                        candidate.retain(|r| !r.is_empty());
                        if let Ok(next) = validate_plan_relaxed_fleet(instance, &candidate) {
                            if improves_repair(instance, routes, cost, &candidate, next) {
                                *routes = candidate;
                                cost = next;
                                accepted = true;
                                break 'search;
                            }
                        }
                    }
                }
            }
        }
        if !accepted || Instant::now() >= deadline {
            break;
        }
    }
}

fn improves_repair(
    instance: &Instance,
    current: &[Vec<usize>],
    cost: f64,
    candidate: &[Vec<usize>],
    next: f64,
) -> bool {
    let excess = |routes: &[Vec<usize>]| {
        if instance.vehicles == 0 {
            0
        } else {
            routes.len().saturating_sub(instance.vehicles)
        }
    };
    let (before, after) = (excess(current), excess(candidate));
    after < before || (after == before && next < cost - 1e-7)
}
