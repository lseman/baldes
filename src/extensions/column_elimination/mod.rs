//! Column elimination through decision-diagram refinement for CVRP/VRPTW.
//!
//! Inspired by Karahalios and van Hoeve's column elimination and the
//! `amkarahalios/dd_vrptw` reference solver. This independent implementation
//! solves a continuous arc-flow relaxation, splits infeasible root prefixes,
//! and preserves every feasible elementary route. Convergence gives the
//! set-partitioning route LP bound, subject to backend numerical tolerances;
//! it does not establish an integer optimum.

use crate::{invalid, Result};
use std::time::Duration;

mod diagram;
mod solve;
pub use solve::{RouteFlow, SolveResult, Status};

/// Start depot 0, customers 1..=n, end depot n+1. Costs and travel times
/// are separate directed (n+2)-square matrices; `None` forbids an arc.
/// Windows bound service start, including at both depots. Waiting is allowed.
#[derive(Clone, Debug)]
pub struct Problem {
    pub demands: Vec<f64>,
    pub capacity: f64,
    pub costs: Vec<Vec<Option<f64>>>,
    pub travel_times: Vec<Vec<f64>>,
    pub windows: Vec<(f64, f64)>,
    pub service_times: Vec<f64>,
    /// Optional upper bound on the number of routes, possibly fractional.
    pub max_vehicles: Option<f64>,
}

#[derive(Clone, Debug)]
pub struct Options {
    pub max_iterations: usize,
    pub max_nodes: usize,
    pub max_arcs: usize,
    pub time_limit: Duration,
    pub tolerance: f64,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            max_iterations: 1000,
            max_nodes: 100_000,
            max_arcs: 1_000_000,
            time_limit: Duration::from_secs(300),
            tolerance: 1e-7,
        }
    }
}

impl Problem {
    /// Copy a parsed Solomon/CVRPLIB instance, respecting its distance
    /// convention and declared fleet size. Set `max_vehicles = None` on the
    /// result for an unlimited fleet. Travel times stay in instance units.
    pub fn from_instance(instance: &crate::inputs::Instance) -> Result<Self> {
        let count = instance.customers.len();
        if count < 2
            || instance
                .distances
                .as_ref()
                .is_some_and(|m| m.len() != count || m.iter().any(|r| r.len() != count))
        {
            return Err(invalid(
                "column elimination needs a depot, customers and a valid distance matrix",
            ));
        }
        let end = count;
        let original = |i| if i == end { 0 } else { i };
        let times: Vec<Vec<_>> = (0..=end)
            .map(|i| {
                (0..=end)
                    .map(|j| instance.distance(original(i), original(j), instance.rounding))
                    .collect()
            })
            .collect();
        let problem = Self {
            demands: instance.customers[1..].iter().map(|c| c.demand).collect(),
            capacity: instance.capacity,
            costs: times
                .iter()
                .map(|r| r.iter().copied().map(Some).collect())
                .collect(),
            travel_times: times,
            windows: (0..=end)
                .map(|i| {
                    let c = &instance.customers[original(i)];
                    (c.ready, c.due)
                })
                .collect(),
            service_times: (0..=end)
                .map(|i| instance.customers[original(i)].service)
                .collect(),
            max_vehicles: Some(instance.vehicles as f64),
        };
        problem.validate()?;
        Ok(problem)
    }

    pub fn validate(&self) -> Result<()> {
        let size = self
            .demands
            .len()
            .checked_add(2)
            .ok_or_else(|| invalid("column elimination instance too large"))?;
        if self.demands.is_empty()
            || !self.capacity.is_finite()
            || self.capacity <= 0.0
            || self.demands.iter().any(|q| !q.is_finite() || *q < 0.0)
            || self.costs.len() != size
            || self.costs.iter().any(|r| r.len() != size)
            || self
                .costs
                .iter()
                .flatten()
                .flatten()
                .any(|c| !c.is_finite() || *c < 0.0)
            || self.travel_times.len() != size
            || self.travel_times.iter().any(|r| r.len() != size)
            || self
                .travel_times
                .iter()
                .flatten()
                .any(|t| !t.is_finite() || *t < 0.0)
            || self.windows.len() != size
            || self
                .windows
                .iter()
                .any(|(a, b)| !a.is_finite() || b.is_nan() || a > b)
            || self.service_times.len() != size
            || self
                .service_times
                .iter()
                .any(|t| !t.is_finite() || *t < 0.0)
            || self.max_vehicles.is_some_and(|k| !k.is_finite() || k < 0.0)
        {
            return Err(invalid("invalid column elimination routing data"));
        }
        Ok(())
    }

    /// First infeasible transition (zero-based) in a depot-to-depot path.
    /// Exact resource comparisons deliberately avoid relaxing feasibility by
    /// the LP tolerance. Zero-demand customers still obey elementarity.
    fn conflict(&self, path: &[usize]) -> Option<usize> {
        let n = self.demands.len();
        let mut seen = vec![false; n + 2];
        let mut load = 0.0;
        let mut time = self.windows[0].0;
        for (step, pair) in path.windows(2).enumerate() {
            let (from, to) = (pair[0], pair[1]);
            if self.costs[from][to].is_none() {
                return Some(step);
            }
            time = (time + self.service_times[from] + self.travel_times[from][to])
                .max(self.windows[to].0);
            if to <= n {
                if seen[to] {
                    return Some(step);
                }
                seen[to] = true;
                load += self.demands[to - 1];
            }
            if !time.is_finite()
                || time > self.windows[to].1
                || !load.is_finite()
                || load > self.capacity
            {
                return Some(step);
            }
        }
        None
    }
}

#[cfg(test)]
mod tests;
