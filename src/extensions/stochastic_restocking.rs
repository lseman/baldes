//! Independent stochastic demands priced with optimal-restocking Bellman states.
//!
//! Unlike [`super::stochastic`]'s Theta formulation, this resource adds expected
//! recourse directly to the pricing objective. Attach it through
//! `CoupledResource::Plugin(PluginResource::new(resource))`. Physical arc costs
//! must contain planned travel costs (and any reduced-cost adjustments), not
//! expected recourse. Depot copies both map to customer ID zero.
//!
//! Backward extensions update the Q+1 continuation values in one Bellman step.
//! Forward extensions rebuild the continuation of the prefix; joins use the
//! kernel's canonical forward replay. Backward dominance compares independently
//! enclosed continuation vectors and visited-set inclusion. Independent finite demands, support at most Q,
//! and expected route demand at most Q follow the shared `Problem` contract.
//! Restocking travel changes cost only: time-window recourse is not modeled.
//! Complete routes have an independent outward recourse-cost certificate.
//! Global pricing bounds may still be numerical or unavailable.

pub use super::stochastic::{Demand, Problem, RouteEvaluation};
use crate::graph::coupled::{CoupledPlugin, PluginState};
use crate::{invalid, Result};

/// Data for Bellman pricing, indexed by physical arc position, not arc ID.
#[derive(Clone, Debug)]
pub struct RestockingResource {
    problem: Problem,
    arcs: Vec<(usize, usize)>,
    pricing: Option<std::sync::Arc<completion::Completion>>,
    prefix_mass: Option<[f64; 2]>,
    prefix_error: Option<f64>,
    reversed: bool,
}

/// A label's continuation costs after serving its first customer, for each
/// residual load. Customer order excludes the depot. Values exclude planned
/// suffix travel. A closed backward state includes its first demand transition.
#[derive(Clone, Debug, PartialEq)]
pub struct BellmanState {
    route: Vec<usize>,
    values: Vec<f64>,
    recourse_cost: f64,
    endpoint: usize,
    closed: bool,
    forward: bool,
    reduced: f64,
    suffix_travel: f64,
    visited: u64,
}
impl BellmanState {
    pub fn customers(&self) -> &[usize] {
        &self.route
    }
    pub fn continuation_values(&self) -> &[f64] {
        &self.values
    }
    pub fn expected_recourse_cost(&self) -> f64 {
        self.recourse_cost
    }
}
impl RestockingResource {
    /// Arc endpoints use the shared problem's IDs: 0 is depot, 1..=n customers.
    /// Both source and sink copies must map to 0. Rebuild after reordering arcs.
    pub fn new(problem: Problem, arcs: Vec<(usize, usize)>) -> Result<Self> {
        let n = problem.demands().len();
        if arcs.iter().any(|&(i, j)| i > n || j > n) {
            return Err(invalid("restocking arc endpoint outside demand model"));
        }
        let prefix_mass = certificate::prefix_mass(&problem);
        let mut resource = Self {
            prefix_mass,
            prefix_error: None,
            reversed: false,
            problem,
            arcs,
            pricing: None,
        };
        resource.prefix_error = certificate::dominance_roundoff(&resource);
        Ok(resource)
    }
    pub(crate) fn with_reversed_pricing(mut self) -> Self {
        self.reversed = true;
        self
    }
    pub fn problem(&self) -> &Problem {
        &self.problem
    }

    fn allocate(&self, value: f64) -> Result<Vec<f64>> {
        let count = self.problem.capacity() + 1;
        let mut values = Vec::new();
        values
            .try_reserve_exact(count)
            .map_err(|_| invalid("Bellman state allocation failed"))?;
        values.resize(count, value);
        Ok(values)
    }
    /// Outcome-major Bellman update recovered from the earlier pricer. The
    /// suffix's planned travel is removed from every value. Literal PMF mass
    /// need not be exactly one, so retain its travel correction explicitly.
    fn prepend(&self, i: usize, j: usize, next: &[f64], travel: f64) -> Result<Vec<f64>> {
        let c = self.problem.costs();
        let capacity = self.problem.capacity();
        let failure = c[j][0] + c[0][j];
        let mut refill = c[i][0] + c[0][j] - c[i][j];
        let mut values = self.allocate(0.)?;
        let mut mass = 0.;
        for &(d, probability) in self.problem.demands()[j - 1].outcomes() {
            mass += probability;
            refill += probability * next[capacity - d];
            let (short, served) = values.split_at_mut(d);
            for (value, &after) in served.iter_mut().zip(&next[..=capacity - d]) {
                *value += probability * after;
            }
            for (value, &after) in short.iter_mut().zip(&next[capacity - d..]) {
                *value += probability * (failure + after);
            }
        }
        let correction = (mass - 1.) * travel;
        for value in &mut values {
            *value = value.min(refill) + correction;
        }
        Ok(values)
    }
    fn advance(
        &self,
        old: &BellmanState,
        arc: usize,
        forward: bool,
    ) -> Result<Option<BellmanState>> {
        if old.forward != forward {
            return Err(invalid("Bellman label direction mismatch"));
        }
        let &(i, j) = self
            .arcs
            .get(arc)
            .ok_or_else(|| invalid("unknown restocking physical arc"))?;
        let (from, to) = if forward { (i, j) } else { (j, i) };
        if old.closed || old.endpoint != from {
            return Ok(None);
        }
        // Prune on masks/load/completion bounds before cloning the route or
        // allocating the Q+1 values, as in the historical resource.
        let compact = self.problem.demands().len() <= 63;
        let store_route = forward || self.pricing.is_none();
        let mut state = BellmanState {
            route: if compact { vec![] } else { old.route.clone() },
            values: vec![],
            recourse_cost: old.recourse_cost,
            endpoint: old.endpoint,
            closed: old.closed,
            forward: old.forward,
            reduced: old.reduced,
            suffix_travel: old.suffix_travel,
            visited: old.visited,
        };
        if let Some(pricing) = &self.pricing {
            state.reduced += pricing.costs[arc];
        }
        if !forward {
            state.suffix_travel += self.problem.costs()[i][j];
        }
        state.endpoint = to;
        state.closed = to == 0;
        if to != 0 {
            if if self.problem.demands().len() <= 63 {
                state.visited & (1u64 << (to - 1)) != 0
            } else {
                state.route.contains(&to)
            } {
                return Ok(None);
            }
            if to <= 63 {
                state.visited |= 1u64 << (to - 1);
            }
            if !compact {
                if forward {
                    state.route.push(to);
                } else {
                    state.route.insert(0, to);
                }
            }
        }
        let mean: f64 = self
            .problem
            .demands()
            .iter()
            .enumerate()
            .filter(|(k, _)| {
                if self.problem.demands().len() <= 63 {
                    state.visited & (1u64 << k) != 0
                } else {
                    state.route.contains(&(k + 1))
                }
            })
            .map(|(_, d)| d.mean())
            .sum();
        if mean > self.problem.capacity() as f64 {
            return Ok(None);
        }
        if !forward && to != 0 {
            if let Some(pricing) = &self.pricing {
                if pricing.rejects(&self.problem, &state, old, i, j, mean, false) {
                    return Ok(None);
                }
            }
        }
        if compact && store_route {
            state.route = old.route.clone();
            if to != 0 {
                if forward {
                    state.route.push(to);
                } else {
                    state.route.insert(0, to);
                }
            }
        }
        if state.visited != 0 || !state.route.is_empty() {
            let costs = self.problem.costs();
            if forward {
                let last = *state.route.last().unwrap();
                state.suffix_travel = costs[last][0];
                state.values = self.allocate(0.)?;
                for pair in state.route.windows(2).rev() {
                    state.values =
                        self.prepend(pair[0], pair[1], &state.values, state.suffix_travel)?;
                    state.suffix_travel += costs[pair[0]][pair[1]];
                }
                let first = state.route[0];
                state.recourse_cost = self.prepend(0, first, &state.values, state.suffix_travel)?
                    [self.problem.capacity()];
            } else {
                state.values = if j == 0 {
                    self.allocate(0.)?
                } else {
                    self.prepend(i, j, &old.values, old.suffix_travel)?
                };
                // Intermediate objectives use the continuation at full load;
                // closure prepends the first demand and gives the route cost.
                state.recourse_cost = state.values[self.problem.capacity()];
            }
            if !forward && to != 0 {
                if let Some(pricing) = &self.pricing {
                    if pricing.rejects(&self.problem, &state, old, i, j, mean, true) {
                        return Ok(None);
                    }
                }
            }
            if state.closed && store_route {
                // Canonical complete-route replay preserves the public model's
                // floating evaluation and independent incumbent certificates.
                state.recourse_cost = self
                    .problem
                    .evaluate_route(&state.route, super::stochastic::Policy::OptimalRestocking)?
                    .expected_recourse_cost;
            }
            if !state.recourse_cost.is_finite() || state.values.iter().any(|x| !x.is_finite()) {
                return Err(invalid("Bellman restocking cost overflow"));
            }
        }
        Ok(Some(state))
    }
}
impl CoupledPlugin for RestockingResource {
    fn backward_pricing(&self) -> bool {
        !self.reversed
    }
    fn staged_pricing(&self) -> bool {
        true
    }
    fn scalar_pricing_charges(&self) -> Option<Vec<f64>> {
        let lower = certificate::objective_bounds(
            self,
            self.problem.demands().len() + 1,
            &mut 1_000_000,
            None,
        )?[0];
        Some(
            self.arcs
                .iter()
                .map(|&(i, _)| if i == 0 { lower } else { 0. })
                .collect(),
        )
    }
    fn supports_cost_dominance(&self) -> bool {
        true
    }
    fn dominates(&self, a: &PluginState, b: &PluginState, forward: bool) -> bool {
        let forward = forward != self.reversed;
        if a == b {
            return true;
        }
        let (Some(a), Some(b)) = (a.get::<BellmanState>(), b.get::<BellmanState>()) else {
            return false;
        };
        if forward
            || a.forward
            || b.forward
            || a.closed
            || b.closed
            || a.endpoint != b.endpoint
            || a.values.len() != b.values.len()
            || !a.route.iter().all(|v| b.route.contains(v))
        {
            return false;
        }
        let Some(mass) = self.prefix_mass else {
            return false;
        };
        // Cheap proposal only. The independent outward continuation replay
        // below decides dominance, including planned travel and literal PMF mass.
        let delta = a
            .values
            .iter()
            .zip(&b.values)
            .map(|(a, b)| a - b)
            .fold(f64::NEG_INFINITY, f64::max)
            + a.suffix_travel
            - b.suffix_travel;
        if delta * mass[usize::from(delta >= 0.)] > a.suffix_travel - b.suffix_travel {
            return false;
        }
        match (
            certificate::continuation(self, &a.route),
            certificate::continuation(self, &b.route),
        ) {
            (Some(left), Some(right)) => left.dominates(&right, mass),
            _ => false,
        }
    }
    fn cost_dominance_penalty(
        &self,
        a: &PluginState,
        b: &PluginState,
        forward: bool,
    ) -> Option<f64> {
        let forward = forward != self.reversed;
        use crate::core::rounding::add_up;
        if a == b {
            return Some(0.);
        }
        let (a, b) = (a.get::<BellmanState>()?, b.get::<BellmanState>()?);
        if forward
            || a.forward
            || b.forward
            || a.closed
            || b.closed
            || a.endpoint != b.endpoint
            || a.values.len() != b.values.len()
            || if self.problem.demands().len() <= 63 {
                a.visited & b.visited != a.visited
            } else {
                !a.route.iter().all(|v| b.route.contains(v))
            }
            || a.values
                .iter()
                .zip(&b.values)
                .any(|(a, b)| a > &(b + 1e-9 * (1. + b.abs())))
        {
            return None;
        }
        let excess = certificate::recourse_excess(a, b, self.prefix_mass?, self.prefix_error?)?;
        // Current resource objectives are already included in the hot key.
        Some(add_up(add_up(excess, b.recourse_cost)?, -a.recourse_cost)?.max(0.))
    }

    fn prepare_pricing(
        &self,
        costs: &[f64],
        threshold: f64,
        deadline: std::time::Instant,
    ) -> Option<crate::graph::coupled::PluginResource> {
        let completion = completion::Completion::new(self, costs, threshold, deadline)?;
        let mut resource = self.clone();
        resource.pricing = Some(std::sync::Arc::new(completion));
        Some(crate::graph::coupled::PluginResource::new(resource))
    }
    fn certify_pricing_minimum(
        &self,
        costs: &[f64],
        work: &mut usize,
        deadline: std::time::Instant,
    ) -> Option<f64> {
        let base = completion::minimum(self, costs, work, deadline)?;
        Some(ng::minimum(self, costs, work, deadline).map_or(base, |bound| base.max(bound)))
    }
    fn objective_bounds(
        &self,
        max_arcs: usize,
        work: &mut usize,
        deadline: Option<std::time::Instant>,
    ) -> Option<[f64; 2]> {
        certificate::objective_bounds(self, max_arcs, work, deadline)
    }
    fn certify_objective(
        &self,
        arcs: &[usize],
        work: &mut usize,
        deadline: Option<std::time::Instant>,
    ) -> Option<(bool, [f64; 2])> {
        if self.reversed {
            certificate::certify(
                self,
                &arcs.iter().copied().rev().collect::<Vec<_>>(),
                work,
                deadline,
            )
        } else {
            certificate::certify(self, arcs, work, deadline)
        }
    }
    fn has_costs(&self) -> bool {
        true
    }
    fn validate(&self, arcs: usize) -> Result<()> {
        if arcs != self.arcs.len() {
            return Err(invalid("restocking physical arc count mismatch"));
        }
        Ok(())
    }
    fn root(&self, forward: bool) -> PluginState {
        let forward = forward != self.reversed;
        PluginState::new(BellmanState {
            route: vec![],
            values: vec![],
            recourse_cost: 0.,
            endpoint: 0,
            closed: false,
            forward,
            reduced: 0.,
            suffix_travel: 0.,
            visited: 0,
        })
    }
    fn objective(&self, state: &PluginState, _forward: bool) -> Result<f64> {
        Ok(state
            .get::<BellmanState>()
            .ok_or_else(|| invalid("expected Bellman label"))?
            .recourse_cost)
    }
    fn extend(
        &self,
        state: &PluginState,
        arc: usize,
        forward: bool,
    ) -> Result<Option<PluginState>> {
        let old = state
            .get::<BellmanState>()
            .ok_or_else(|| invalid("expected Bellman label"))?;
        Ok(self
            .advance(old, arc, forward != self.reversed)?
            .map(PluginState::new))
    }
    fn join(&self, state: &PluginState, suffix: &mut dyn Iterator<Item = usize>) -> Result<bool> {
        let mut current = state.clone();
        for arc in suffix {
            match self.extend(&current, arc, true)? {
                Some(next) => current = next,
                None => return Ok(false),
            }
        }
        Ok(true)
    }
}

/// Solve the shared stochastic model with Bellman pricing instead of Theta cuts.
/// Uses the same warm-start and tree controls as the cut-based formulation.
/// Inspect tree status and bound mode rather than treating an unfinished solve
/// or a numerical lower bound as a directed proof.
#[cfg(feature = "clp")]
pub fn solve(
    problem: &Problem,
    vehicles: usize,
    options: super::stochastic::SolveOptions,
) -> Result<super::stochastic::SolveResult> {
    let started = std::time::Instant::now();
    let bound = certificate::routing_lower_bound(problem, vehicles).unwrap_or(0.);
    let mut options = options;
    options.tree.time_limit = options.tree.time_limit.saturating_sub(started.elapsed());
    let mut result = problem.solve_formulation(
        super::stochastic::Policy::OptimalRestocking,
        vehicles,
        options,
        true,
    )?;
    if bound > result.tree.bound {
        result.tree.bound = bound;
        result.lower_bound_source = "RestockingAssignmentRelaxation";
    }
    Ok(result)
}

mod bounds;
#[path = "stochastic_restocking/certificate.rs"]
mod certificate;
mod completion;
mod ng;

#[cfg(test)]
#[path = "../../tests/unit/extensions/restocking_reversed.rs"]
mod reversed_tests;
