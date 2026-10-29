//! Route local search on true stochastic reduced costs.
//!
//! Scalar NG pricing ignores the recourse that separates good candidates from
//! poor ones, and exact Bellman pricing is expensive. This heuristic improves
//! seed routes (basic master columns and NG candidates) with insertion,
//! removal, replacement, relocation and reversal moves. Moves are ranked by
//! their scalar reduced cost and only the best are replayed with the full
//! Bellman program. It supplies columns only and never establishes a bound.
use super::pricing::route_uses;
use super::{ArcBranch, Policy, PricedRoute, Problem};
use crate::Result;
use std::collections::BTreeSet;
use std::time::{Duration, Instant};

/// Moves replayed with full recourse per local-search step.
const REPLAYS_PER_STEP: usize = 48;
const MAX_STEPS: usize = 64;

pub(super) struct ReducedCosts<'a> {
    pub policy: Policy,
    pub customer_duals: &'a [f64],
    pub fleet_dual: f64,
    pub branches: &'a [ArcBranch],
    pub branch_duals: &'a [f64],
    pub phase_one: bool,
}

impl Problem {
    /// Improve each seed by local search and return distinct routes whose
    /// replayed reduced cost is below `-tolerance`, best first. Stops when
    /// `budget` elapses; seeds that are infeasible or forbidden are skipped.
    pub(super) fn search_routes(
        &self,
        prices: &ReducedCosts,
        seeds: &[Vec<usize>],
        tolerance: f64,
        budget: Duration,
    ) -> Result<Vec<PricedRoute>> {
        let started = Instant::now();
        let mut found = BTreeSet::new();
        let mut routes = Vec::new();
        let mut visited = BTreeSet::new();
        for seed in seeds {
            if started.elapsed() >= budget {
                break;
            }
            if !visited.insert(seed.clone()) || !self.admissible(prices, seed) {
                continue;
            }
            let mut current = seed.clone();
            let mut value = self.reduced_cost(prices, &current)?.0;
            for _ in 0..MAX_STEPS {
                if started.elapsed() >= budget {
                    break;
                }
                let mut candidates: Vec<_> = self
                    .neighbors(&current)
                    .into_iter()
                    .filter(|r| self.admissible(prices, r))
                    .map(|r| (self.scalar_reduced_cost(prices, &r), r))
                    .collect();
                candidates.sort_by(|a, b| a.0.total_cmp(&b.0));
                let mut best: Option<(f64, Vec<usize>)> = None;
                for (_, route) in candidates.into_iter().take(REPLAYS_PER_STEP) {
                    if !visited.insert(route.clone()) {
                        continue;
                    }
                    let (rc, evaluation) = self.reduced_cost(prices, &route)?;
                    if rc < -tolerance && found.insert(route.clone()) {
                        routes.push(PricedRoute {
                            customers: route.clone(),
                            evaluation,
                            reduced_cost: rc,
                        });
                    }
                    if best.as_ref().map_or(true, |(b, _)| rc < *b) {
                        best = Some((rc, route));
                    }
                }
                match best {
                    Some((rc, route)) if rc < value - tolerance => {
                        value = rc;
                        current = route;
                    }
                    _ => break,
                }
            }
        }
        routes.sort_by(|a, b| a.reduced_cost.total_cmp(&b.reduced_cost));
        Ok(routes)
    }

    fn admissible(&self, prices: &ReducedCosts, route: &[usize]) -> bool {
        let n = self.demands.len();
        let mut seen = vec![false; n + 1];
        let mut load = 0.;
        for &j in route {
            if j == 0 || j > n || seen[j] {
                return false;
            }
            seen[j] = true;
            load += self.demands[j - 1].mean();
        }
        !route.is_empty()
            && load <= self.capacity as f64
            && !prices
                .branches
                .iter()
                .any(|b| !b.required && route_uses(route, *b))
    }

    fn dual_value(&self, prices: &ReducedCosts, route: &[usize]) -> f64 {
        let mut value = prices.fleet_dual;
        for &j in route {
            value += prices.customer_duals[j - 1];
        }
        for (&b, &dual) in prices.branches.iter().zip(prices.branch_duals) {
            if route_uses(route, b) {
                value += dual;
            }
        }
        value
    }

    fn scalar_reduced_cost(&self, prices: &ReducedCosts, route: &[usize]) -> f64 {
        let mut travel = 0.;
        if !prices.phase_one {
            let mut i = 0;
            for &j in route.iter().chain(std::iter::once(&0)) {
                travel += self.costs[i][j];
                i = j;
            }
        }
        travel - self.dual_value(prices, route)
    }

    fn reduced_cost(
        &self,
        prices: &ReducedCosts,
        route: &[usize],
    ) -> Result<(f64, super::RouteEvaluation)> {
        let evaluation = self.evaluate_route(route, prices.policy)?;
        let cost = if prices.phase_one {
            0.
        } else {
            evaluation.total_cost
        };
        Ok((cost - self.dual_value(prices, route), evaluation))
    }

    fn neighbors(&self, route: &[usize]) -> Vec<Vec<usize>> {
        let n = self.demands.len();
        let mut inside = vec![false; n + 1];
        for &j in route {
            inside[j] = true;
        }
        let outside: Vec<_> = (1..=n).filter(|&k| !inside[k]).collect();
        let len = route.len();
        let mut out = Vec::new();
        for &k in &outside {
            for p in 0..=len {
                let mut r = route.to_vec();
                r.insert(p, k);
                out.push(r);
            }
        }
        for p in 0..len {
            if len > 1 {
                let mut r = route.to_vec();
                r.remove(p);
                out.push(r);
            }
            for &k in &outside {
                let mut r = route.to_vec();
                r[p] = k;
                out.push(r);
            }
            for t in (0..len).filter(|&t| t != p) {
                let mut r = route.to_vec();
                let j = r.remove(p);
                r.insert(t, j);
                out.push(r);
            }
            for q in p + 2..=len {
                let mut r = route.to_vec();
                r[p..q].reverse();
                out.push(r);
            }
        }
        out
    }
}
