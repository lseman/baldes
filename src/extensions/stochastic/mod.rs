//! Independent discrete demands with a priori routes and replenishment recourse.
//!
//! Implements policies P1 and P3, equations (9) and (11), of Tomazella,
//! Munari and Morabito (2026). Customer order is fixed during execution.
//! Capacity constrains expected route demand; realized overload triggers recourse.
//! Uses the conventional routing branch-cut-and-price tree with Theta cuts
//! for expected recourse. Tiny enumeration remains an independent oracle.

use crate::{invalid, Result};

mod solve;
mod theta;
mod tiny;
mod warm_start;
pub use solve::{SolveOptions, SolveResult};
pub use tiny::{ExactSolution, TinyOptions};
pub use warm_start::{HgsWarmStart, HgsWarmStartSummary};

/// Finite integer support, with probabilities normalized on construction.
#[derive(Clone, Debug)]
pub struct Demand {
    outcomes: Vec<(usize, f64)>,
}

impl Demand {
    /// Support must be nonempty and strictly increasing. Probabilities must be
    /// finite and nonnegative, summing to one within 1e-12 before normalization.
    pub fn new(mut outcomes: Vec<(usize, f64)>) -> Result<Self> {
        let sum: f64 = outcomes.iter().map(|&(_, p)| p).sum();
        if outcomes.is_empty()
            || outcomes.windows(2).any(|w| w[0].0 >= w[1].0)
            || outcomes.iter().any(|&(_, p)| !p.is_finite() || p < 0.0)
            || !sum.is_finite()
            || (sum - 1.0).abs() > 1e-12
        {
            return Err(invalid("invalid discrete demand distribution"));
        }
        for (_, p) in &mut outcomes {
            *p /= sum;
        }
        Ok(Self { outcomes })
    }

    pub fn outcomes(&self) -> &[(usize, f64)] {
        &self.outcomes
    }

    pub fn mean(&self) -> f64 {
        self.outcomes.iter().map(|&(d, p)| d as f64 * p).sum()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Policy {
    /// Replenish on failure; if empty, replenish before the next customer.
    Classical,
    /// Choose between preventive replenishment and direct travel at each state.
    OptimalRestocking,
}

/// Depot is vertex 0; demand entry i belongs to customer vertex i + 1.
/// Directed costs must be finite, nonnegative and have zero diagonal.
/// As in the paper, callers should supply a metric (triangle-inequality) matrix.
#[derive(Clone, Debug)]
pub struct Problem {
    capacity: usize,
    demands: Vec<Demand>,
    costs: Vec<Vec<f64>>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RouteEvaluation {
    pub travel_cost: f64,
    pub expected_recourse_cost: f64,
    pub total_cost: f64,
    pub expected_demand: f64,
}

impl Problem {
    pub fn new(capacity: usize, demands: Vec<Demand>, costs: Vec<Vec<f64>>) -> Result<Self> {
        let vertices = demands
            .len()
            .checked_add(1)
            .ok_or_else(|| invalid("too many customers"))?;
        if capacity == 0
            || capacity.checked_add(1).is_none()
            || costs.len() != vertices
            || costs.iter().enumerate().any(|(i, row)| {
                row.len() != vertices
                    || row.iter().any(|c| !c.is_finite() || *c < 0.0)
                    || row[i] != 0.0
            })
            || demands
                .iter()
                .any(|d| d.outcomes.iter().any(|&(v, _)| v > capacity))
        {
            return Err(invalid(
                "invalid stochastic capacity, support, or cost matrix",
            ));
        }
        Ok(Self {
            capacity,
            demands,
            costs,
        })
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }
    pub fn demands(&self) -> &[Demand] {
        &self.demands
    }
    pub fn costs(&self) -> &[Vec<f64>] {
        &self.costs
    }

    /// Evaluate an elementary sequence of customer IDs, excluding depot endpoints.
    /// Rejects routes whose mean demand exceeds capacity. Empty routes cost zero.
    /// Work is O(route length * (Q+1) * support size), memory O(Q).
    pub fn evaluate_route(&self, route: &[usize], policy: Policy) -> Result<RouteEvaluation> {
        let mut seen = vec![false; self.demands.len()];
        for &j in route {
            if j == 0 || j > self.demands.len() || seen[j - 1] {
                return Err(invalid("route must contain distinct customer IDs in 1..=n"));
            }
            seen[j - 1] = true;
        }
        // Capacity depends on the customer set, independently of visit order.
        let expected_demand: f64 = seen
            .iter()
            .zip(&self.demands)
            .filter(|(present, _)| **present)
            .map(|(_, d)| d.mean())
            .sum();
        if expected_demand > self.capacity as f64 {
            return Err(invalid("route expected demand exceeds capacity"));
        }
        if route.is_empty() {
            return Ok(RouteEvaluation {
                travel_cost: 0.0,
                expected_recourse_cost: 0.0,
                total_cost: 0.0,
                expected_demand,
            });
        }
        let states = self.capacity + 1;
        let allocate = || -> Result<Vec<f64>> {
            let mut v = Vec::new();
            v.try_reserve_exact(states)
                .map_err(|_| invalid("recourse state allocation failed"))?;
            v.resize(states, 0.0);
            Ok(v)
        };
        let mut next = allocate()?;
        let mut current = allocate()?;
        // next[q] is the recourse after servicing j with residual q.
        // Terminal values are zero even when the last customer empties the truck.
        for t in (0..route.len()).rev() {
            let j = route[t];
            let i = if t == 0 { 0 } else { route[t - 1] };
            self.prepend_recourse(i, j, &next, &mut current, policy)?;
            std::mem::swap(&mut current, &mut next);
        }
        let mut travel_cost = 0.0;
        let mut i = 0;
        for &j in route {
            travel_cost += self.costs[i][j];
            i = j;
        }
        travel_cost += self.costs[i][0];
        let expected_recourse_cost = next[self.capacity];
        let total_cost = travel_cost + expected_recourse_cost;
        if !total_cost.is_finite() || !expected_recourse_cost.is_finite() {
            return Err(invalid("stochastic route cost overflow"));
        }
        Ok(RouteEvaluation {
            travel_cost,
            expected_recourse_cost,
            total_cost,
            expected_demand,
        })
    }
    /// Bellman transition from after i through customer j into a fixed suffix.
    fn prepend_recourse(
        &self,
        i: usize,
        j: usize,
        next: &[f64],
        current: &mut [f64],
        policy: Policy,
    ) -> Result<()> {
        let demand = &self.demands[j - 1];
        let capacity = self.capacity;
        let failure = self.costs[j][0] + self.costs[0][j];
        let mut refill = self.costs[i][0] + self.costs[0][j] - self.costs[i][j];
        current.fill(0.);
        // Accumulate the direct expectation one outcome at a time, so each
        // pass is a contiguous slice operation over residual loads.
        for &(d, p) in &demand.outcomes {
            refill += p * next[capacity - d];
            let (short, served) = current.split_at_mut(d);
            for (value, &after) in served.iter_mut().zip(&next[..=capacity - d]) {
                *value += p * after;
            }
            for (value, &after) in short.iter_mut().zip(&next[capacity - d..]) {
                *value += p * (failure + after);
            }
        }
        match policy {
            Policy::Classical => current[0] = refill,
            Policy::OptimalRestocking => {
                for value in current.iter_mut() {
                    *value = value.min(refill);
                }
            }
        }
        if current.iter().any(|v| !v.is_finite()) {
            return Err(invalid("stochastic continuation cost overflow"));
        }
        Ok(())
    }
}
