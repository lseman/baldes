use super::{Policy, Problem, RouteEvaluation};
use crate::{invalid, Result};

/// Exhaustive baseline only: at most nine customers and a bounded number of
/// enumerated routes. Budget exhaustion returns an error, never an optimal claim.
#[derive(Clone, Copy, Debug)]
pub struct TinyOptions {
    pub max_vehicles: usize,
    pub route_budget: usize,
}

#[derive(Clone, Debug)]
pub struct ExactSolution {
    pub routes: Vec<Vec<usize>>,
    pub evaluations: Vec<RouteEvaluation>,
    pub total_cost: f64,
    pub enumerated_routes: usize,
}

impl Problem {
    /// Enumerate all mean-capacity-feasible elementary routes, keep the cheapest
    /// order for each customer subset, then solve set partitioning by subset DP.
    /// Returns None only after exhaustive proof of fleet infeasibility.
    /// Optimality is combinatorial, subject to ordinary f64 cost comparisons.
    pub fn solve_tiny(
        &self,
        policy: Policy,
        options: TinyOptions,
    ) -> Result<Option<ExactSolution>> {
        let n = self.demands.len();
        if n > 9 {
            return Err(invalid(
                "tiny stochastic solver supports at most nine customers",
            ));
        }
        let size = 1usize << n;
        let mut search = Search {
            problem: self,
            policy,
            budget: options.route_budget,
            count: 0,
            best: vec![None; size],
        };
        search.visit(&mut Vec::new(), 0, 0.0)?;
        let vehicles = options.max_vehicles.min(n);
        let mut dp = vec![vec![f64::INFINITY; size]; vehicles + 1];
        let mut choice = vec![vec![0; size]; vehicles + 1];
        dp[0][0] = 0.0;
        for k in 1..=vehicles {
            dp[k][0] = 0.0;
            for mask in 1..size {
                let first = 1usize << mask.trailing_zeros();
                let mut subset = mask;
                while subset != 0 {
                    if subset & first != 0 {
                        if let Some((_, evaluation)) = &search.best[subset] {
                            let cost = evaluation.total_cost + dp[k - 1][mask ^ subset];
                            if dp[k - 1][mask ^ subset].is_finite() && !cost.is_finite() {
                                return Err(invalid("stochastic partition cost overflow"));
                            }
                            if cost < dp[k][mask] {
                                dp[k][mask] = cost;
                                choice[k][mask] = subset;
                            }
                        }
                    }
                    subset = (subset - 1) & mask;
                }
            }
        }
        let mut mask = size - 1;
        if !dp[vehicles][mask].is_finite() {
            return Ok(None);
        }
        let mut solution = ExactSolution {
            routes: Vec::new(),
            evaluations: Vec::new(),
            total_cost: dp[vehicles][mask],
            enumerated_routes: search.count,
        };
        let mut k = vehicles;
        while mask != 0 {
            let subset = choice[k][mask];
            let (route, evaluation) = search.best[subset]
                .as_ref()
                .ok_or_else(|| invalid("missing stochastic partition witness"))?;
            solution.routes.push(route.clone());
            solution.evaluations.push(*evaluation);
            mask ^= subset;
            k -= 1;
        }
        Ok(Some(solution))
    }
}

struct Search<'a> {
    problem: &'a Problem,
    policy: Policy,
    budget: usize,
    count: usize,
    best: Vec<Option<(Vec<usize>, RouteEvaluation)>>,
}

impl Search<'_> {
    fn visit(&mut self, route: &mut Vec<usize>, mask: usize, mean: f64) -> Result<()> {
        for j in 1..=self.problem.demands.len() {
            let bit = 1 << (j - 1);
            if mask & bit != 0 {
                continue;
            }
            let next_mean = mean + self.problem.demands[j - 1].mean();
            if next_mean > self.problem.capacity as f64 {
                continue;
            }
            if self.count == self.budget {
                return Err(invalid("tiny stochastic route budget exhausted"));
            }
            self.count += 1;
            route.push(j);
            let evaluation = self.problem.evaluate_route(route, self.policy)?;
            let next_mask = mask | bit;
            if self.best[next_mask]
                .as_ref()
                .map_or(true, |(_, old)| evaluation.total_cost < old.total_cost)
            {
                self.best[next_mask] = Some((route.clone(), evaluation));
            }
            self.visit(route, next_mask, next_mean)?;
            route.pop();
        }
        Ok(())
    }
}
