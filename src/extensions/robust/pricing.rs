//! Reference forward, elementary labeling for the robust RCESPP.
//! Labels carry γ-indexed worst-case loads and start times (4.5)-(4.8) and
//! are pruned by Proposition 4.1: cost, visited subset and both vectors
//! componentwise. Bidirectional search, ng-routes and cuts are not yet used.

use super::{Problem, RobustState};
use crate::{invalid, Result};

#[derive(Clone, Copy, Debug)]
pub struct PricingOptions {
    /// Total labels created, including the root.
    pub max_labels: usize,
    /// Routes returned, in increasing reduced cost.
    pub max_routes: usize,
    pub dominance: bool,
}

impl Default for PricingOptions {
    fn default() -> Self {
        Self {
            max_labels: 2_000_000,
            max_routes: 16,
            dominance: true,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PricingStatus {
    /// Every robust feasible elementary route was covered, in f64 arithmetic.
    Complete,
    /// The label budget ran out; returned routes are valid but not proven best.
    LabelLimit,
}

#[derive(Clone, Debug)]
pub struct PricedRoute {
    /// Closed route `depot, ..., depot`, replay-validated.
    pub route: Vec<usize>,
    pub cost: f64,
    pub reduced_cost: f64,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct PricingStatistics {
    pub labels: usize,
    pub dominance_checks: usize,
    pub dominated: usize,
}

#[derive(Clone, Debug)]
pub struct PricingResult {
    pub status: PricingStatus,
    /// Lowest reduced-cost routes found, whether or not they are negative.
    /// An empty list has proof meaning only with Complete.
    pub routes: Vec<PricedRoute>,
    pub statistics: PricingStatistics,
}

struct Label {
    node: usize,
    parent: Option<usize>,
    visited: Vec<u64>,
    reduced: f64,
    state: RobustState,
    active: bool,
}

fn subset(a: &[u64], b: &[u64]) -> bool {
    a.iter().zip(b).all(|(x, y)| x & !y == 0)
}

impl Problem {
    /// Price elementary robust routes under customer duals (one per node,
    /// zero at the depot) and a fleet dual subtracted once per route.
    pub fn price(
        &self,
        duals: &[f64],
        fleet_dual: f64,
        options: PricingOptions,
    ) -> Result<PricingResult> {
        self.validate()?;
        if duals.len() != self.nodes.len()
            || duals.iter().any(|d| !d.is_finite())
            || duals[self.depot] != 0.0
            || !fleet_dual.is_finite()
            || options.max_labels == 0
        {
            return Err(invalid("invalid robust pricing duals or options"));
        }
        let n = self.nodes.len();
        let words = (n + 63) / 64;
        let mut stats = PricingStatistics {
            labels: 1,
            ..Default::default()
        };
        let mut status = PricingStatus::Complete;
        let mut labels = vec![Label {
            node: self.depot,
            parent: None,
            visited: vec![0; words],
            reduced: -fleet_dual,
            state: self.initial_state(),
            active: true,
        }];
        let mut buckets = vec![Vec::<usize>::new(); n];
        let mut completions = Vec::<(f64, usize)>::new();
        let mut cursor = 0;
        'search: while cursor < labels.len() {
            if !labels[cursor].active {
                cursor += 1;
                continue;
            }
            let at = labels[cursor].node;
            if at != self.depot {
                if let Some(arc) = self.travel[at][self.depot] {
                    if self.extend(&labels[cursor].state, at, self.depot).is_some() {
                        completions.push((labels[cursor].reduced + arc.cost, cursor));
                    }
                }
            }
            for next in 0..n {
                if next == self.depot
                    || next == at
                    || labels[cursor].visited[next / 64] >> (next % 64) & 1 == 1
                {
                    continue;
                }
                let Some(arc) = self.travel[at][next] else {
                    continue;
                };
                let Some(state) = self.extend(&labels[cursor].state, at, next) else {
                    continue;
                };
                if labels.len() >= options.max_labels {
                    status = PricingStatus::LabelLimit;
                    break 'search;
                }
                let mut visited = labels[cursor].visited.clone();
                visited[next / 64] |= 1 << (next % 64);
                let mut new = Label {
                    node: next,
                    parent: Some(cursor),
                    visited,
                    reduced: labels[cursor].reduced + arc.cost - duals[next],
                    state,
                    active: true,
                };
                if options.dominance {
                    for &old in &buckets[next] {
                        let old = &mut labels[old];
                        if !old.active {
                            continue;
                        }
                        stats.dominance_checks += 1;
                        if old.reduced <= new.reduced
                            && subset(&old.visited, &new.visited)
                            && old.state.dominates(&new.state)
                        {
                            new.active = false;
                            break;
                        }
                        if new.reduced <= old.reduced
                            && subset(&new.visited, &old.visited)
                            && new.state.dominates(&old.state)
                        {
                            old.active = false;
                            stats.dominated += 1;
                        }
                    }
                }
                if new.active {
                    buckets[next].push(labels.len());
                    labels.push(new);
                    stats.labels += 1;
                } else {
                    stats.dominated += 1;
                }
            }
            cursor += 1;
        }
        completions.sort_by(|a, b| a.0.total_cmp(&b.0));
        let mut routes = Vec::new();
        for &(_, last) in completions.iter().take(options.max_routes) {
            let mut route = vec![self.depot];
            let mut id = Some(last);
            while let Some(i) = id {
                route.push(labels[i].node);
                id = labels[i].parent;
            }
            route.reverse();
            // Recompute from original data rather than accumulated label sums.
            let schedule = self.replay(&route)?;
            let reduced_cost = route
                .iter()
                .fold(schedule.cost - fleet_dual, |c, &v| c - duals[v]);
            routes.push(PricedRoute {
                route,
                cost: schedule.cost,
                reduced_cost,
            });
        }
        routes.sort_by(|a, b| a.reduced_cost.total_cmp(&b.reduced_cost));
        Ok(PricingResult {
            status,
            routes,
            statistics: stats,
        })
    }
}
