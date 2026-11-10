//! Bounded initial-plan improvement with true recourse, independent of pricing.
use super::{warm_start::SeedSolution, Policy, Problem};
use crate::{invalid, Result};
use std::time::Instant;

const REPLAYS_PER_STEP: usize = 48;
const MAX_STEPS: usize = 64;

/// Work and independently replayed objectives of initial-plan refinement.
#[derive(Clone, Copy, Debug)]
pub struct IncumbentSearchSummary {
    pub initial_objective: f64,
    pub final_objective: f64,
    pub evaluated_moves: usize,
    pub accepted_moves: usize,
    pub elapsed_seconds: f64,
}

struct Move {
    a: usize,
    route_a: Vec<usize>,
    b: Option<usize>,
    route_b: Vec<usize>,
    travel: f64,
}

fn travel(problem: &Problem, route: &[usize]) -> f64 {
    if route.is_empty() {
        return 0.;
    }
    let mut from = 0;
    let mut cost = 0.;
    for &to in route.iter().chain(std::iter::once(&0)) {
        cost += problem.costs()[from][to];
        from = to;
    }
    cost
}

fn retain_best(moves: &mut Vec<Move>) {
    moves.sort_by(|a, b| a.travel.total_cmp(&b.travel));
    moves.truncate(REPLAYS_PER_STEP);
}

fn neighbors(problem: &Problem, routes: &[Vec<usize>], deadline: Instant) -> Vec<Move> {
    let baseline: Vec<_> = routes.iter().map(|r| travel(problem, r)).collect();
    let mut moves = Vec::new();
    let mut add = |a: usize, route_a: Vec<usize>, b: Option<usize>, route_b: Vec<usize>| {
        if Instant::now() >= deadline {
            return false;
        }
        let score = travel(problem, &route_a) - baseline[a]
            + b.map_or(0., |b| travel(problem, &route_b) - baseline[b]);
        if score.is_finite() {
            moves.push(Move {
                a,
                route_a,
                b,
                route_b,
                travel: score,
            });
            // Bound retained allocation even for large plans. Stable sorting
            // preserves generation order among equal-travel alternatives.
            if moves.len() >= 2 * REPLAYS_PER_STEP {
                retain_best(&mut moves);
            }
        }
        true
    };
    for (a, route) in routes.iter().enumerate() {
        for from in 0..route.len() {
            for to in 0..route.len() {
                if from == to {
                    continue;
                }
                let mut next = route.clone();
                let customer = next.remove(from);
                next.insert(to, customer);
                if !add(a, next, None, vec![]) {
                    retain_best(&mut moves);
                    return moves;
                }
            }
            for end in from + 1..route.len() {
                let mut next = route.clone();
                next[from..=end].reverse();
                if !add(a, next, None, vec![]) {
                    retain_best(&mut moves);
                    return moves;
                }
            }
            for (b, other) in routes.iter().enumerate() {
                if a == b {
                    continue;
                }
                for to in 0..=other.len() {
                    let mut left = route.clone();
                    let customer = left.remove(from);
                    let mut right = other.clone();
                    right.insert(to, customer);
                    if !add(a, left, Some(b), right) {
                        retain_best(&mut moves);
                        return moves;
                    }
                }
                if a < b {
                    for to in 0..other.len() {
                        let mut left = route.clone();
                        let mut right = other.clone();
                        std::mem::swap(&mut left[from], &mut right[to]);
                        if !add(a, left, Some(b), right) {
                            retain_best(&mut moves);
                            return moves;
                        }
                    }
                }
            }
        }
    }
    retain_best(&mut moves);
    moves
}

impl Problem {
    pub(super) fn search_incumbent(
        &self,
        policy: Policy,
        vehicles: usize,
        mut routes: Vec<Vec<usize>>,
        deadline: Instant,
    ) -> Result<(SeedSolution, IncumbentSearchSummary)> {
        let started = Instant::now();
        if vehicles == 0 || routes.len() > vehicles || routes.iter().any(Vec::is_empty) {
            return Err(invalid(
                "incumbent search needs a nonempty-route plan within the fleet",
            ));
        }
        let mut sorted: Vec<_> = routes.iter().flatten().copied().collect();
        sorted.sort_unstable();
        if sorted != (1..=self.demands().len()).collect::<Vec<_>>() {
            return Err(invalid("incumbent search needs exact customer coverage"));
        }
        let mut costs: Vec<_> = routes
            .iter()
            .map(|r| self.evaluate_route(r, policy).map(|e| e.total_cost))
            .collect::<Result<_>>()?;
        let mut total: f64 = costs.iter().sum();
        if !total.is_finite() {
            return Err(invalid("incumbent search objective overflow"));
        }
        let mut summary = IncumbentSearchSummary {
            initial_objective: total,
            final_objective: total,
            evaluated_moves: 0,
            accepted_moves: 0,
            elapsed_seconds: 0.,
        };
        for _ in 0..MAX_STEPS {
            if Instant::now() >= deadline {
                break;
            }
            let candidates = neighbors(self, &routes, deadline);
            let mut best: Option<(f64, Move)> = None;
            for candidate in candidates {
                if Instant::now() >= deadline {
                    break;
                }
                summary.evaluated_moves += 1;
                let Ok(left) = self.evaluate_route(&candidate.route_a, policy) else {
                    continue;
                };
                let mut cost = total - costs[candidate.a] + left.total_cost;
                if let Some(b) = candidate.b {
                    let Ok(right) = self.evaluate_route(&candidate.route_b, policy) else {
                        continue;
                    };
                    cost += right.total_cost - costs[b];
                }
                if cost < total - 1e-9 * (1. + total.abs())
                    && best.as_ref().is_none_or(|(old, _)| cost < *old)
                {
                    best = Some((cost, candidate));
                }
            }
            let Some((_, candidate)) = best else {
                break;
            };
            let mut next = routes.clone();
            next[candidate.a] = candidate.route_a;
            if let Some(b) = candidate.b {
                next[b] = candidate.route_b;
            }
            next.retain(|r| !r.is_empty());
            // Accept using a canonical whole-plan sum, not cancellation in a
            // move delta. Every move preserves coverage and never adds a route.
            let next_costs = next
                .iter()
                .map(|r| self.evaluate_route(r, policy).map(|e| e.total_cost))
                .collect::<Result<Vec<_>>>()?;
            let next_total: f64 = next_costs.iter().sum();
            if !next_total.is_finite() || next_total >= total - 1e-9 * (1. + total.abs()) {
                break;
            }
            routes = next;
            costs = next_costs;
            total = next_total;
            summary.accepted_moves += 1;
        }
        summary.final_objective = total;
        summary.elapsed_seconds = started.elapsed().as_secs_f64();
        Ok((
            SeedSolution {
                routes,
                total_cost: total,
            },
            summary,
        ))
    }
}

#[cfg(test)]
#[path = "../../../tests/unit/extensions/stochastic/incumbent_search.rs"]
mod tests;
