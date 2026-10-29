//! Robust insertion heuristic: cheapest pair insertion, pair relocation,
//! route elimination, and ruin-and-recreate iterations.
//!
//! Each route caches its forward worst-case starts, its backward start
//! thresholds (as in pricing joins), and per-position on-board load data.
//! Inserting request r with its pickup after position i and its delivery
//! after position j ≥ i then costs O((j − i + 1)(Γt + 1)) time work and O(1)
//! capacity work per position. A move is accepted only after
//! [`Problem::replay`] confirms the new route.
use super::Problem;
use crate::rng::Xoroshiro128Plus;
use std::time::{Duration, Instant};

mod genetic;
mod operators;

/// Optional ILS operations; the default preserves the original trajectory.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct IlsExtensions {
    /// Regret-2 repair over distinct receiving routes, including singletons.
    pub regret_repair: bool,
    /// Mix uniform ruin with removals at tight load cuts or time segments.
    pub critical_removal: bool,
    /// Remove requests close to a randomly chosen request in space and time.
    pub related_removal: bool,
    /// Prefer requests whose removal saves the most route travel.
    pub expensive_removal: bool,
    /// Exchange two complete requests between routes as one transaction.
    pub pair_exchange: bool,
    /// Swap tails only at cuts with no open requests.
    pub tail_exchange: bool,
    /// Move a request while ejecting and repairing one other request.
    pub ejection_chain: bool,
}

impl IlsExtensions {
    /// Enable every optional operation.
    pub fn all() -> Self {
        Self {
            regret_repair: true,
            critical_removal: true,
            related_removal: true,
            expensive_removal: true,
            pair_exchange: true,
            tail_exchange: true,
            ejection_chain: true,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum HeuristicMethod {
    /// One trajectory of ruin-and-recreate with local search.
    #[default]
    IteratedLocalSearch,
    /// ILS with explicitly selected optional operations.
    ExtendedIteratedLocalSearch(IlsExtensions),
    /// HGS-style population with route-based crossover and diversity control.
    Genetic,
}

#[derive(Clone, Copy, Debug)]
pub struct HeuristicOptions {
    pub time_limit: Duration,
    /// Ruin-and-recreate iterations, or offspring for the genetic method,
    /// after the first local optimum.
    pub iterations: usize,
    pub seed: u64,
    pub method: HeuristicMethod,
}
impl Default for HeuristicOptions {
    fn default() -> Self {
        Self {
            time_limit: Duration::from_secs(1),
            iterations: 2_000,
            seed: 1,
            method: HeuristicMethod::default(),
        }
    }
}

/// Routes use the original node ids, from depot 0 to depot 2n+1.
#[derive(Clone, Debug)]
pub struct HeuristicSolution {
    pub routes: Vec<Vec<usize>>,
    /// Travel cost plus one vehicle cost per route.
    pub cost: f64,
}

/// One robust-feasible route with the data that insertion checks need.
#[derive(Clone, Debug)]
struct Route {
    nodes: Vec<usize>,
    /// Worst service start at each position for at most g delayed arcs.
    forward: Vec<Vec<f64>>,
    /// Latest start at each position allowing at most g delays after it,
    /// excluding that position's own window.
    backward: Vec<Vec<f64>>,
    /// After service at each position: nominal load, sum of the Γq largest
    /// on-board deviations, and the deviation a newcomer must exceed.
    load: Vec<[f64; 3]>,
    travel: f64,
}

#[derive(Clone, Copy, Debug)]
struct Insertion {
    route: Option<usize>,
    pickup_after: usize,
    delivery_after: usize,
    delta: f64,
}

struct Search<'a> {
    problem: &'a Problem,
    extensions: IlsExtensions,
    n: usize,
    budget: usize,
    rng: Xoroshiro128Plus,
    deadline: Instant,
}

impl Problem {
    /// Robust-feasible routes covering every request once, or `None` when
    /// some request cannot be served even by a route of its own.
    pub fn heuristic(&self, options: HeuristicOptions) -> Option<HeuristicSolution> {
        let n = self.instance.requests;
        let mut search = Search {
            problem: self,
            extensions: match options.method {
                HeuristicMethod::ExtendedIteratedLocalSearch(extensions) => extensions,
                _ => IlsExtensions::default(),
            },
            n,
            budget: self.parameters.time_budget.min(2 * n + 1),
            rng: Xoroshiro128Plus::from_seed(options.seed),
            deadline: Instant::now() + options.time_limit,
        };
        if (0..n).any(|r| search.alone(r).is_none()) {
            return None;
        }
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&a, &b| {
            let open = |r: usize| self.instance.nodes[r + 1].open;
            open(a).total_cmp(&open(b)).then(a.cmp(&b))
        });
        let mut current = search.recreate(Vec::new(), &order)?;
        search.local_search(&mut current);
        let mut best = current.clone();
        // The genetic method first spends 40% of the time on this trajectory,
        // which reduces the fleet more reliably, then evolves from its best.
        let phase_end = match options.method {
            HeuristicMethod::IteratedLocalSearch
            | HeuristicMethod::ExtendedIteratedLocalSearch(_) => search.deadline,
            HeuristicMethod::Genetic => Instant::now() + options.time_limit.mul_f64(0.4),
        };
        for _ in 0..options.iterations {
            if Instant::now() >= phase_end.min(search.deadline) {
                break;
            }
            let mut candidate = current.clone();
            let removed = search.ruin(&mut candidate);
            let Some(mut candidate) = search.recreate(candidate, &removed) else {
                continue;
            };
            search.local_search(&mut candidate);
            let (value, reference) = (search.cost(&candidate), search.cost(&current));
            if value < reference - 1e-9 || value <= search.cost(&best) * 1.002 {
                current = candidate;
            }
            if search.cost(&current) < search.cost(&best) - 1e-9 {
                best = current.clone();
            }
        }
        if options.method == HeuristicMethod::Genetic {
            best = genetic::run(&mut search, best, options.iterations);
        }
        let routes: Vec<Vec<usize>> = best.into_iter().map(|r| r.nodes).collect();
        let cost = routes
            .iter()
            .map(|r| self.replay(r).map(|s| s.cost))
            .sum::<crate::Result<f64>>()
            .ok()?;
        Some(HeuristicSolution { routes, cost })
    }
}

impl Search<'_> {
    fn window(&self, v: usize) -> (f64, f64) {
        let node = &self.problem.instance.nodes[v];
        (node.open, node.close)
    }
    fn travel(&self, u: usize, v: usize) -> f64 {
        self.problem.travel[u][v]
    }
    /// The forward recurrence of `Problem::replay`, with the same arithmetic.
    fn step(&self, start: &[f64], u: usize, v: usize) -> Option<Vec<f64>> {
        let duration = self.problem.instance.nodes[u].service + self.travel(u, v);
        let deviation = (self.problem.parameters.time_deviation * self.travel(u, v)).ceil();
        let (open, close) = self.window(v);
        let mut next = Vec::with_capacity(start.len());
        for (g, &b) in start.iter().enumerate() {
            let mut value = open.max(b + duration);
            if g > 0 {
                value = value.max(start[g - 1] + duration + deviation);
            }
            if !value.is_finite() || value > close {
                return None;
            }
            next.push(value);
        }
        Some(next)
    }
    fn joins(&self, start: &[f64], threshold: &[f64]) -> bool {
        (0..=self.budget).all(|g| start[g] <= threshold[self.budget - g])
    }
    /// Worst load after adding request r to the on-board set summarized by
    /// `load`, without rescanning that set.
    fn load_with(&self, load: &[f64; 3], r: usize) -> f64 {
        let extra = if self.problem.parameters.demand_budget > 0 {
            (self.problem.deviations[r] - load[2]).max(0.)
        } else {
            0.
        };
        load[0] + self.problem.demands[r] + load[1] + extra
    }

    fn build(&self, nodes: Vec<usize>) -> Option<Route> {
        let schedule = self.problem.replay(&nodes).ok()?;
        let m = nodes.len() - 1;
        let mut backward = vec![vec![f64::INFINITY; self.budget + 1]; m + 1];
        for k in (0..m).rev() {
            let (u, v) = (nodes[k], nodes[k + 1]);
            let duration = self.problem.instance.nodes[u].service + self.travel(u, v);
            let deviation = (self.problem.parameters.time_deviation * self.travel(u, v)).ceil();
            let close = self.window(v).1;
            let clipped: Vec<f64> = backward[k + 1].iter().map(|b| b.min(close)).collect();
            backward[k][0] = clipped[0] - duration;
            for g in 1..=self.budget {
                backward[k][g] = (clipped[g] - duration).min(clipped[g - 1] - duration - deviation);
            }
        }
        let gamma = self.problem.parameters.demand_budget;
        let mut open = vec![false; self.n];
        let mut load = Vec::with_capacity(m + 1);
        for &v in &nodes {
            if (1..=self.n).contains(&v) {
                open[v - 1] = true;
            } else if (self.n + 1..=2 * self.n).contains(&v) {
                open[v - self.n - 1] = false;
            }
            let mut nominal = 0.;
            let mut deviations = vec![];
            for (r, _) in open.iter().enumerate().filter(|(_, o)| **o) {
                nominal += self.problem.demands[r];
                deviations.push(self.problem.deviations[r]);
            }
            deviations.sort_by(|a, b| b.total_cmp(a));
            let top: f64 = deviations.iter().take(gamma).sum();
            let threshold = if gamma > 0 && deviations.len() >= gamma {
                deviations[gamma - 1]
            } else {
                0.
            };
            load.push([nominal, top, threshold]);
        }
        Some(Route {
            nodes,
            forward: schedule.starts,
            backward,
            load,
            travel: schedule.cost - self.problem.parameters.vehicle_cost,
        })
    }

    /// Travel cost of serving request r alone, if robust-feasible.
    fn alone(&self, r: usize) -> Option<f64> {
        let (p, d) = (r + 1, r + 1 + self.n);
        self.build(vec![0, p, d, 2 * self.n + 1])
            .map(|route| route.travel)
    }

    fn cost(&self, routes: &[Route]) -> f64 {
        routes
            .iter()
            .map(|r| r.travel + self.problem.parameters.vehicle_cost)
            .sum()
    }

    /// Every robust-feasible placement of request r in `route`, by the cached
    /// forward vectors, backward thresholds and load summaries.
    fn placements(&self, route: &Route, index: usize, r: usize, out: &mut Vec<Insertion>) {
        let capacity = self.problem.instance.capacity;
        let (p, d) = (r + 1, r + 1 + self.n);
        let nodes = &route.nodes;
        let m = nodes.len() - 1;
        let arc = |u: usize, v: usize| self.travel(u, v);
        for i in 0..m {
            if self.load_with(&route.load[i], r) > capacity {
                continue;
            }
            let Some(mut current) = self.step(&route.forward[i], nodes[i], p) else {
                continue;
            };
            let pickup_delta = arc(nodes[i], p);
            let mut previous = p;
            for j in i..m {
                // Delivery right after `previous`, then the rest of the route.
                if let Some(at_delivery) = self.step(&current, previous, d) {
                    if let Some(next) = self.step(&at_delivery, d, nodes[j + 1]) {
                        if self.joins(&next, &route.backward[j + 1]) {
                            let delta = if j == i {
                                pickup_delta + arc(p, d) + arc(d, nodes[i + 1])
                                    - arc(nodes[i], nodes[i + 1])
                            } else {
                                pickup_delta + arc(p, nodes[i + 1]) - arc(nodes[i], nodes[i + 1])
                                    + arc(nodes[j], d)
                                    + arc(d, nodes[j + 1])
                                    - arc(nodes[j], nodes[j + 1])
                            };
                            out.push(Insertion {
                                route: Some(index),
                                pickup_after: i,
                                delivery_after: j,
                                delta,
                            });
                        }
                    }
                }
                // Carry request r on board past the next original node.
                if j + 1 == m || self.load_with(&route.load[j + 1], r) > capacity {
                    break;
                }
                let Some(next) = self.step(&current, previous, nodes[j + 1]) else {
                    break;
                };
                current = next;
                previous = nodes[j + 1];
            }
        }
    }

    fn insert_nodes(&self, nodes: &[usize], r: usize, i: usize, j: usize) -> Vec<usize> {
        let (p, d) = (r + 1, r + 1 + self.n);
        let mut next = Vec::with_capacity(nodes.len() + 2);
        next.extend_from_slice(&nodes[..=i]);
        next.push(p);
        next.extend_from_slice(&nodes[i + 1..=j]);
        next.push(d);
        next.extend_from_slice(&nodes[j + 1..]);
        next
    }

    /// Apply the cheapest verified insertion of r; `allow_new` permits a new
    /// route. Returns the change in total cost.
    fn insert(&self, routes: &mut Vec<Route>, r: usize, allow_new: bool) -> Option<f64> {
        let mut candidates = Vec::new();
        for (index, route) in routes.iter().enumerate() {
            self.placements(route, index, r, &mut candidates);
        }
        if allow_new {
            if let Some(travel) = self.alone(r) {
                candidates.push(Insertion {
                    route: None,
                    pickup_after: 0,
                    delivery_after: 0,
                    delta: travel + self.problem.parameters.vehicle_cost,
                });
            }
        }
        candidates.sort_by(|a, b| a.delta.total_cmp(&b.delta));
        for c in candidates {
            match c.route {
                None => {
                    let (p, d) = (r + 1, r + 1 + self.n);
                    routes.push(self.build(vec![0, p, d, 2 * self.n + 1])?);
                    return Some(c.delta);
                }
                Some(index) => {
                    let nodes = self.insert_nodes(
                        &routes[index].nodes,
                        r,
                        c.pickup_after,
                        c.delivery_after,
                    );
                    let before = routes[index].travel;
                    // Floating-point boundary disagreements fall back to the
                    // next candidate rather than accepting an unchecked route.
                    if let Some(route) = self.build(nodes) {
                        let delta = route.travel - before;
                        routes[index] = route;
                        return Some(delta);
                    }
                }
            }
        }
        None
    }

    /// Remove request r from its route; `None` if the shortened route is not
    /// robust-feasible (without the triangle inequality it need not be).
    fn remove(&self, routes: &mut Vec<Route>, r: usize) -> Option<f64> {
        let (p, d) = (r + 1, r + 1 + self.n);
        let index = routes.iter().position(|route| route.nodes.contains(&p))?;
        let nodes: Vec<usize> = routes[index]
            .nodes
            .iter()
            .copied()
            .filter(|&v| v != p && v != d)
            .collect();
        let before = routes[index].travel + self.problem.parameters.vehicle_cost;
        if nodes.len() == 2 {
            routes.swap_remove(index);
            return Some(-before);
        }
        let route = self.build(nodes)?;
        let after = route.travel + self.problem.parameters.vehicle_cost;
        routes[index] = route;
        Some(after - before)
    }

    fn recreate(&mut self, mut routes: Vec<Route>, requests: &[usize]) -> Option<Vec<Route>> {
        if self.extensions.regret_repair {
            return self.regret_recreate(routes, requests);
        }
        for &r in requests {
            self.insert(&mut routes, r, true)?;
        }
        Some(routes)
    }

    /// Remove a random 15–50% of the requests, at least two when possible.
    fn ruin(&mut self, routes: &mut Vec<Route>) -> Vec<usize> {
        let share = 0.15 + 0.35 * self.rng.next_f64();
        let count = ((share * self.n as f64).round() as usize).clamp(self.n.min(2), self.n);
        let mut requests: Vec<usize> = (0..self.n).collect();
        self.rng.shuffle(&mut requests);
        let mut strategies = Vec::new();
        if self.extensions.critical_removal {
            strategies.push(0);
        }
        if self.extensions.related_removal {
            strategies.push(1);
        }
        if self.extensions.expensive_removal {
            strategies.push(2);
        }
        if !strategies.is_empty() && self.rng.next_f64() < 0.5 {
            let choice = (self.rng.next_f64() * strategies.len() as f64) as usize;
            match strategies[choice.min(strategies.len() - 1)] {
                0 => self.critical_order(routes, &mut requests),
                1 => self.related_order(&mut requests),
                _ => self.expensive_order(routes, &mut requests),
            }
        }
        let mut removed = vec![];
        for r in requests.into_iter().take(count) {
            if self.remove(routes, r).is_some() {
                removed.push(r);
            }
        }
        self.rng.shuffle(&mut removed);
        removed
    }

    fn local_search(&mut self, routes: &mut Vec<Route>) {
        loop {
            if Instant::now() >= self.deadline {
                return;
            }
            let mut improved = self.eliminate_route(routes) | self.relocate_pairs(routes);
            if self.extensions.pair_exchange && Instant::now() < self.deadline {
                improved |= self.exchange_pairs(routes);
            }
            if self.extensions.tail_exchange && Instant::now() < self.deadline {
                improved |= self.exchange_tails(routes);
            }
            if self.extensions.ejection_chain && Instant::now() < self.deadline {
                improved |= self.ejection_chain(routes);
            }
            if !improved {
                return;
            }
        }
    }

    /// Move each request to its cheapest verified position when that pays.
    fn relocate_pairs(&mut self, routes: &mut Vec<Route>) -> bool {
        let mut improved = false;
        let mut requests: Vec<usize> = (0..self.n).collect();
        self.rng.shuffle(&mut requests);
        for r in requests {
            let backup = routes.clone();
            let Some(saving) = self.remove(routes, r) else {
                continue;
            };
            match self.insert(routes, r, true) {
                Some(delta) if saving + delta < -1e-7 => improved = true,
                _ => *routes = backup,
            }
        }
        improved
    }

    /// Try to empty the route with the fewest requests into the others.
    fn eliminate_route(&mut self, routes: &mut Vec<Route>) -> bool {
        if routes.len() < 2 {
            return false;
        }
        let mut order: Vec<usize> = (0..routes.len()).collect();
        order.sort_by_key(|&i| routes[i].nodes.len());
        for index in order {
            let backup = routes.clone();
            let requests: Vec<usize> = routes[index].nodes[1..routes[index].nodes.len() - 1]
                .iter()
                .filter(|&&v| v <= self.n)
                .map(|&v| v - 1)
                .collect();
            let before = self.cost(routes);
            routes.remove(index);
            if requests
                .iter()
                .all(|&r| self.insert(routes, r, false).is_some())
                && self.cost(routes) < before - 1e-7
            {
                return true;
            }
            *routes = backup;
        }
        false
    }
}

#[cfg(test)]
#[path = "../../../tests/unit/extensions/robust_pd/heuristic_tests.rs"]
mod tests;
