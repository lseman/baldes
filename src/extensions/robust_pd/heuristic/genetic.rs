//! Hybrid genetic search over robust PD routes (after Vidal's HGS).
//!
//! Offspring come from route-based crossover, which keeps pickup–delivery
//! pairs intact: copy some routes of one parent, add the other parent's
//! routes without the already covered requests, then reinsert the missing
//! requests. Each child is educated by the shared local search. Parents
//! are drawn by binary tournament on HGS biased fitness, which ranks cost
//! and the average broken-pairs distance to the closest individuals.
use super::{Route, Search};
use std::time::Instant;

/// Survivors kept after selection.
const MU: usize = 25;
/// Offspring added before the next selection.
const LAMBDA: usize = 40;
/// Individuals whose cost rank is protected from diversity pressure.
const ELITE: usize = 4;
/// Closest individuals averaged in the diversity contribution.
const CLOSE: usize = 5;
/// Offspring without a new best before the population restarts.
const RESTART: usize = 2_000;
/// Probability that a child is ruined and recreated before education.
const MUTATION: f64 = 0.3;

#[derive(Clone)]
struct Individual {
    routes: Vec<Route>,
    cost: f64,
    /// Successor of each customer node; route ends map to the sink.
    successor: Vec<usize>,
}

impl Search<'_> {
    fn individual(&self, routes: Vec<Route>) -> Individual {
        let mut successor = vec![0; 2 * self.n + 2];
        for route in &routes {
            for arc in route.nodes.windows(2) {
                successor[arc[0]] = arc[1];
            }
        }
        Individual {
            cost: self.cost(&routes),
            routes,
            successor,
        }
    }

    /// Share of customer nodes whose successors differ (broken pairs).
    fn distance(&self, a: &Individual, b: &Individual) -> f64 {
        let differing = (1..=2 * self.n)
            .filter(|&v| a.successor[v] != b.successor[v])
            .count();
        differing as f64 / (2 * self.n) as f64
    }

    /// HGS biased fitness: cost rank plus diversity rank, lower is better.
    fn biased_fitness(&self, population: &[Individual]) -> Vec<f64> {
        let size = population.len();
        if size < 2 {
            return vec![0.; size];
        }
        let diversity: Vec<f64> = (0..size)
            .map(|i| {
                let mut distances: Vec<f64> = (0..size)
                    .filter(|&j| j != i)
                    .map(|j| self.distance(&population[i], &population[j]))
                    .collect();
                distances.sort_by(f64::total_cmp);
                let close = CLOSE.min(distances.len());
                distances[..close].iter().sum::<f64>() / close as f64
            })
            .collect();
        let rank = |key: &dyn Fn(usize) -> f64| {
            let mut order: Vec<usize> = (0..size).collect();
            order.sort_by(|&a, &b| key(a).total_cmp(&key(b)));
            let mut ranks = vec![0.; size];
            for (position, i) in order.into_iter().enumerate() {
                ranks[i] = position as f64 / (size - 1) as f64;
            }
            ranks
        };
        let cost = rank(&|i| population[i].cost);
        let spread = rank(&|i| -diversity[i]);
        let weight = 1. - (ELITE as f64 / size as f64).min(1.);
        (0..size).map(|i| cost[i] + weight * spread[i]).collect()
    }

    fn tournament(&mut self, fitness: &[f64]) -> usize {
        let size = fitness.len() as u64;
        let a = self.rng.uniform_below(size) as usize;
        let b = self.rng.uniform_below(size) as usize;
        if fitness[a] <= fitness[b] {
            a
        } else {
            b
        }
    }

    /// Drop clones first, then the worst biased fitness, down to MU.
    fn select_survivors(&self, population: &mut Vec<Individual>) {
        while population.len() > MU {
            let clone = (0..population.len()).find(|&i| {
                (0..population.len()).any(|j| {
                    j != i
                        && population[j].cost <= population[i].cost
                        && self.distance(&population[i], &population[j]) == 0.
                })
            });
            let worst = clone.unwrap_or_else(|| {
                let fitness = self.biased_fitness(population);
                (0..population.len())
                    .max_by(|&a, &b| fitness[a].total_cmp(&fitness[b]))
                    .unwrap()
            });
            population.swap_remove(worst);
        }
    }

    fn requests(&self, route: &Route) -> impl Iterator<Item = usize> + '_ {
        let n = self.n;
        route
            .nodes
            .iter()
            .filter(move |&&v| (1..=n).contains(&v))
            .map(|&v| v - 1)
            .collect::<Vec<_>>()
            .into_iter()
    }

    /// Route-based crossover; `None` when a missing request cannot be placed.
    fn crossover(&mut self, a: &Individual, b: &Individual) -> Option<Vec<Route>> {
        let mut chosen: Vec<usize> = (0..a.routes.len()).collect();
        self.rng.shuffle(&mut chosen);
        let take = 1 + self.rng.uniform_below(a.routes.len() as u64) as usize;
        let mut child: Vec<Route> = chosen[..take]
            .iter()
            .map(|&i| a.routes[i].clone())
            .collect();
        let mut covered = vec![false; self.n];
        for route in &child {
            for r in self.requests(route) {
                covered[r] = true;
            }
        }
        let mut others: Vec<usize> = (0..b.routes.len()).collect();
        self.rng.shuffle(&mut others);
        for index in others {
            let route = &b.routes[index];
            let requests: Vec<usize> = self.requests(route).collect();
            let kept: Vec<usize> = requests.iter().copied().filter(|&r| !covered[r]).collect();
            if kept.is_empty() {
                continue;
            }
            let rebuilt = if kept.len() == requests.len() {
                Some(route.clone())
            } else {
                let nodes = route
                    .nodes
                    .iter()
                    .copied()
                    .filter(|&v| v == 0 || v > 2 * self.n || kept.contains(&((v - 1) % self.n)))
                    .collect();
                self.build(nodes)
            };
            if let Some(route) = rebuilt {
                for &r in &kept {
                    covered[r] = true;
                }
                child.push(route);
            }
        }
        let mut missing: Vec<usize> = (0..self.n).filter(|&r| !covered[r]).collect();
        self.rng.shuffle(&mut missing);
        self.recreate(child, &missing)
    }

    /// Ruin-and-recreate a child; half of the time, dissolve one whole route
    /// and first try to reinsert its requests without opening a new one,
    /// since a vehicle outweighs most travel savings.
    fn mutate(&mut self, routes: &mut Vec<Route>) {
        if routes.len() > 1 && self.rng.next_f64() < 0.5 {
            let index = self.rng.uniform_below(routes.len() as u64) as usize;
            let mut removed: Vec<usize> = self.requests(&routes[index]).collect();
            routes.swap_remove(index);
            removed.extend(self.ruin(routes));
            self.rng.shuffle(&mut removed);
            let mut packed = routes.clone();
            if removed
                .iter()
                .all(|&r| self.insert(&mut packed, r, false).is_some())
            {
                *routes = packed;
                return;
            }
            if let Some(rebuilt) = self.recreate(routes.clone(), &removed) {
                *routes = rebuilt;
            }
            return;
        }
        let removed = self.ruin(routes);
        if let Some(rebuilt) = self.recreate(routes.clone(), &removed) {
            *routes = rebuilt;
        }
    }

    /// A locally optimal solution from a random insertion order.
    fn random_individual(&mut self) -> Option<Individual> {
        let mut order: Vec<usize> = (0..self.n).collect();
        self.rng.shuffle(&mut order);
        let mut routes = self.recreate(Vec::new(), &order)?;
        self.local_search(&mut routes);
        Some(self.individual(routes))
    }
}

/// Evolve from `start` until the deadline or `offspring` children; returns
/// the best routes found, never worse than `start`.
pub(super) fn run(search: &mut Search<'_>, start: Vec<Route>, offspring: usize) -> Vec<Route> {
    let mut best = search.individual(start);
    let mut population = vec![best.clone()];
    let fill = |search: &mut Search<'_>, population: &mut Vec<Individual>| {
        while population.len() < 4 * MU && Instant::now() < search.deadline {
            if let Some(individual) = search.random_individual() {
                population.push(individual);
            }
        }
    };
    fill(search, &mut population);
    let mut stalled = 0;
    for _ in 0..offspring {
        if Instant::now() >= search.deadline {
            break;
        }
        let fitness = search.biased_fitness(&population);
        let first = search.tournament(&fitness);
        let second = search.tournament(&fitness);
        let (a, b) = (population[first].clone(), population[second].clone());
        let Some(mut routes) = search.crossover(&a, &b) else {
            continue;
        };
        if search.rng.next_f64() < MUTATION {
            search.mutate(&mut routes);
        }
        search.local_search(&mut routes);
        let child = search.individual(routes);
        stalled += 1;
        if child.cost < best.cost - 1e-9 {
            best = child.clone();
            stalled = 0;
        }
        population.push(child);
        if population.len() >= MU + LAMBDA {
            search.select_survivors(&mut population);
        }
        if stalled >= RESTART {
            population = vec![best.clone()];
            fill(search, &mut population);
            stalled = 0;
        }
    }
    best.routes
}
