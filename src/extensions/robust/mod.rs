//! Robust VRPTW under budgeted demand and travel-time uncertainty.
//!
//! Demands q̃_i ∈ [q_i, q_i + q̂_i] and travel times t̃_ij ∈ [t_ij, t_ij + t̂_ij]
//! range over budgeted polytopes: at most Γq demands and Γt travel times
//! take their worst case per route. A route is robust feasible when it meets
//! capacity and every time window for all such realizations.
//!
//! Robustness is checked by the dynamic programs of Munari et al. (2018):
//! the worst-case load and service start are kept for each γ = 0..=Γ. Time
//! cannot use the Γ largest deviations, because waiting absorbs some of them.
//!
//! Production pricing uses [`Problem::pricing_graph`] with the standard
//! `crate::master::routing::solve` adapter (Clp feature), or CLI `--robust`.
//! The independent forward labeler in [`reference`](mod@reference) is for validation only.

use crate::{invalid, Result};

mod physical_graph;
pub mod reference;
mod solomon;
#[doc(hidden)]
#[deprecated(note = "validation-only API; use robust::reference explicitly")]
pub use reference::{PricedRoute, PricingOptions, PricingResult, PricingStatistics, PricingStatus};
#[doc(hidden)]
#[deprecated(
    note = "validation-only API; use robust::reference explicitly, or pricing_graph with the standard routing solver for production"
)]
pub use reference::{
    RobustColumn, RobustColumnGenerationOptions, RobustColumnGenerationResult,
    RobustColumnGenerationStatus, RobustIteration, RobustMaster,
};
pub use solomon::Uncertainty;

fn nonnegative(x: f64) -> bool {
    x.is_finite() && x >= 0.0
}

#[derive(Clone, Copy, Debug)]
pub struct Node {
    /// Nominal demand q_i. Zero at the depot.
    pub demand: f64,
    /// Worst-case demand increase q̂_i. Zero at the depot.
    pub demand_deviation: f64,
    /// Service must start within this inclusive window.
    pub window: (f64, f64),
    pub service: f64,
}

#[derive(Clone, Copy, Debug)]
pub struct Travel {
    /// Nominal travel time t_ij.
    pub time: f64,
    /// Worst-case travel-time increase t̂_ij.
    pub time_deviation: f64,
    pub cost: f64,
}

/// Single depot, homogeneous unlimited fleet. Routes leave and return to
/// `depot`; the depot window bounds both departure and return.
#[derive(Clone, Debug)]
pub struct Problem {
    pub nodes: Vec<Node>,
    /// Directed row-major matrix; None denotes an unavailable arc.
    pub travel: Vec<Vec<Option<Travel>>>,
    pub depot: usize,
    pub capacity: f64,
    /// Γq: demands that may simultaneously take their worst case.
    pub demand_budget: usize,
    /// Γt: travel times that may simultaneously take their worst case.
    pub time_budget: usize,
}

/// Worst-case resources at a node: `load[γ]` and `start[γ]` hold the values
/// when at most γ demands or travel times deviate (equations 2.3 and 2.5).
/// Both vectors are nondecreasing in γ, so the last entry is binding.
#[derive(Clone, Debug, PartialEq)]
pub struct RobustState {
    pub load: Vec<f64>,
    pub start: Vec<f64>,
}

impl RobustState {
    /// Componentwise ≤ on both vectors: hypotheses (iii)-(iv) of Prop. 4.1.
    pub fn dominates(&self, other: &RobustState) -> bool {
        self.load.iter().zip(&other.load).all(|(a, b)| a <= b)
            && self.start.iter().zip(&other.start).all(|(a, b)| a <= b)
    }
}

#[derive(Clone, Debug)]
pub struct RobustSchedule {
    /// One state per route position, depot at both ends.
    pub states: Vec<RobustState>,
    pub cost: f64,
}

impl Problem {
    /// Compatibility entry point for the validation-only forward labeler.
    /// Production pricing uses [`Self::pricing_graph`] and the standard solver.
    #[deprecated(note = "use reference::price for validation, or pricing_graph for production")]
    pub fn price(
        &self,
        duals: &[f64],
        fleet_dual: f64,
        options: reference::PricingOptions,
    ) -> Result<reference::PricingResult> {
        reference::price(self, duals, fleet_dual, options)
    }

    pub fn validate(&self) -> Result<()> {
        let n = self.nodes.len();
        if self.depot >= n
            || !nonnegative(self.capacity)
            || self.travel.len() != n
            || self.travel.iter().any(|r| r.len() != n)
        {
            return Err(invalid(
                "invalid robust VRPTW dimensions, depot or capacity",
            ));
        }
        for (i, node) in self.nodes.iter().enumerate() {
            let (open, close) = node.window;
            if !nonnegative(node.demand)
                || !nonnegative(node.demand_deviation)
                || !nonnegative(node.service)
                || !open.is_finite()
                || !close.is_finite()
                || open > close
                || (i == self.depot && (node.demand != 0.0 || node.demand_deviation != 0.0))
            {
                return Err(invalid("invalid robust VRPTW node data"));
            }
        }
        if self
            .travel
            .iter()
            .flatten()
            .flatten()
            .any(|a| !nonnegative(a.time) || !nonnegative(a.time_deviation) || !a.cost.is_finite())
        {
            return Err(invalid("invalid robust VRPTW travel data"));
        }
        Ok(())
    }

    /// Paper's benchmark deviations: q̂_i = trunc(αq·q_i) and
    /// t̂_ij = 0.1·trunc(10·αt·t_ij), from nominal data already in `self`.
    pub fn with_relative_deviations(mut self, alpha_demand: f64, alpha_time: f64) -> Result<Self> {
        if !nonnegative(alpha_demand) || !nonnegative(alpha_time) {
            return Err(invalid(
                "relative deviations must be finite and nonnegative",
            ));
        }
        for (i, node) in self.nodes.iter_mut().enumerate() {
            node.demand_deviation = if i == self.depot {
                0.0
            } else {
                (alpha_demand * node.demand).trunc()
            };
        }
        for arc in self.travel.iter_mut().flatten().flatten() {
            arc.time_deviation = 0.1 * (10.0 * alpha_time * arc.time).trunc();
        }
        Ok(self)
    }

    /// State at a route's start: zero load, service at the depot opening.
    pub fn initial_state(&self) -> RobustState {
        RobustState {
            load: vec![0.0; self.demand_budget + 1],
            start: vec![self.nodes[self.depot].window.0; self.time_budget + 1],
        }
    }

    /// Extend along (from, to) by equations (4.7)-(4.8). None if the head's
    /// capacity or window is violated for some admissible realization.
    pub fn extend(&self, state: &RobustState, from: usize, to: usize) -> Option<RobustState> {
        let arc = self.travel[from][to]?;
        let head = &self.nodes[to];
        let (q, dq) = (head.demand, head.demand_deviation);
        let mut load = Vec::with_capacity(state.load.len());
        load.push(state.load[0] + q);
        for g in 1..state.load.len() {
            load.push((state.load[g] + q).max(state.load[g - 1] + q + dq));
        }
        let shift = self.nodes[from].service + arc.time;
        let mut start = Vec::with_capacity(state.start.len());
        start.push(head.window.0.max(state.start[0] + shift));
        for g in 1..state.start.len() {
            start.push(
                head.window
                    .0
                    .max(state.start[g] + shift)
                    .max(state.start[g - 1] + shift + arc.time_deviation),
            );
        }
        let feasible =
            *load.last().unwrap() <= self.capacity && *start.last().unwrap() <= head.window.1;
        feasible.then_some(RobustState { load, start })
    }

    /// Independently replay a closed route `depot, ..., depot`. Customers may
    /// occur at most once. Comparisons are strict f64 comparisons.
    pub fn replay(&self, route: &[usize]) -> Result<RobustSchedule> {
        self.validate()?;
        if route.len() < 2 || route[0] != self.depot || route[route.len() - 1] != self.depot {
            return Err(invalid("robust route must start and end at its depot"));
        }
        let mut seen = vec![false; self.nodes.len()];
        let mut states = vec![self.initial_state()];
        let mut cost = 0.0;
        for w in route.windows(2) {
            let (from, to) = (w[0], w[1]);
            if to >= self.nodes.len() {
                return Err(invalid("unknown robust route node"));
            }
            if to != self.depot {
                if seen[to] {
                    return Err(invalid("robust route repeats a customer"));
                }
                seen[to] = true;
            } else if states.len() != route.len() - 1 {
                return Err(invalid("intermediate depot visit unsupported"));
            }
            let arc = self.travel[from][to].ok_or_else(|| invalid("unavailable robust arc"))?;
            cost += arc.cost;
            let state = self
                .extend(states.last().unwrap(), from, to)
                .ok_or_else(|| invalid("route is not robust feasible"))?;
            states.push(state);
        }
        Ok(RobustSchedule { states, cost })
    }
}

#[cfg(all(test, feature = "clp"))]
#[path = "../../../tests/unit/extensions/robust/master_tests.rs"]
mod master_tests;
#[cfg(test)]
#[path = "../../../tests/unit/extensions/robust/native_tests.rs"]
mod native_tests;
#[cfg(test)]
#[path = "../../../tests/unit/extensions/robust/tests.rs"]
mod tests;
