//! Stochastic recourse on BALDES's physical graph and ordinary pricing engine.
use super::bounds::{LoadWalkBounds, SubsetWalkBounds};
use super::{Policy, Problem, RouteEvaluation};
use crate::graph::coupled::{CoupledPlugin, CoupledResource, PluginResource, PluginState};
use crate::{invalid, Graph, Result};
use rayon::prelude::*;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// `2^M` walk tables are built per exact pricing call.
const MAX_COMPLETION_SUBSET_SIZE: usize = 8;

#[derive(Clone, Copy, Debug)]
pub struct PricingOptions {
    pub max_labels: usize,
    pub max_columns: usize,
    pub time_limit: Duration,
    pub bidirectional: bool,
    /// `Some(t)` with `t <= 0` prunes reverse Bellman labels whose completion
    /// bound (suffix reduced cost, cheapest continuation value and a
    /// load-indexed prefix walk bound) is at least `t`. A Complete result then
    /// certifies only that no route has reduced cost below `t`:
    /// `minimum_reduced_cost` is the observed minimum capped at `t`, a lower
    /// bound on the true minimum. Ignored by the bidirectional adapter.
    pub completion_threshold: Option<f64>,
    /// Size of the customer set `M` whose subsets get restricted completion
    /// walk tables (`2^M` tables); zero keeps the unrestricted bound only.
    pub completion_subset_size: usize,
    /// Add the elementary knapsack completion bound to reverse pruning.
    pub completion_knapsack: bool,
    /// NG prepricing replays routes at `relaxed_route_cost` (travel plus
    /// per-customer recourse lower bounds) instead of exact expected cost.
    pub relaxed_route_costs: bool,
    /// Run NG pricing with BALDES bidirectional bucket labeling (forward and
    /// backward sweeps on two threads) instead of one forward sweep.
    pub ng_bidirectional: bool,
}
impl Default for PricingOptions {
    fn default() -> Self {
        Self {
            max_labels: 1_000_000,
            max_columns: 30,
            time_limit: Duration::from_secs(60),
            bidirectional: false,
            completion_threshold: None,
            completion_subset_size: 4,
            completion_knapsack: true,
            relaxed_route_costs: false,
            ng_bidirectional: false,
        }
    }
}

#[derive(Clone, Debug)]
pub struct PricedRoute {
    pub customers: Vec<usize>,
    pub evaluation: RouteEvaluation,
    pub reduced_cost: f64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PricingStatus {
    Complete,
    /// Useful replayed columns from NG prepricing; exact recourse search has
    /// not run. This status never by itself establishes LP convergence.
    Heuristic,
    WorkLimit,
    TimeLimit,
}

#[derive(Clone, Debug)]
pub struct PricingResult {
    pub status: PricingStatus,
    pub routes: Vec<PricedRoute>,
    /// Minimum over all feasible routes only when status is Complete.
    /// With Complete, None denotes an empty route domain. Incomplete results
    /// always use None; their observed routes cannot establish the minimum.
    pub minimum_reduced_cost: Option<f64>,
    /// Numerical global reduced-cost lower bound from a hop-bounded walk
    /// relaxation, available even when labeling stops. Unlike an observed route
    /// minimum, this includes every elementary route in its domain.
    pub lower_bound: Option<f64>,
    pub statistics: crate::labeling::Statistics,
}

/// Customer IDs use the public problem numbering; vertex 0 also represents the
/// duplicated sink when a branch names a depot-return arc.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ArcBranch {
    pub from: usize,
    pub to: usize,
    /// false forbids this arc; true requires the solution's total arc flow = 1.
    pub required: bool,
}

#[derive(Clone, Debug)]
struct RecourseResource {
    problem: Problem,
    policy: Policy,
    arcs: Vec<(usize, usize)>,
    enabled: bool,
    reversed: bool,
    /// Reduced cost of each graph arc, aligned with `arcs`.
    reduced: Vec<f64>,
    /// Prefix bounds and threshold for completion pruning of reverse labels.
    prefixes: Option<(Arc<SubsetWalkBounds>, f64)>,
}
#[derive(Clone, Debug, PartialEq)]
enum State {
    // For optimal recourse a forward prefix must retain its decision history:
    // decisions depend on the as-yet unknown continuation. Re-evaluate the
    // prefix Bellman program when extended; unequal histories never dominate.
    Forward {
        route: Vec<usize>,
        cost: f64,
    },
    // Expected continuation cost for every remaining load at the current node.
    Backward(Vec<f64>),
    // `reduced` repeats the label's arc reduced cost for completion bounds.
    Continuation {
        visited: u64,
        values: Vec<f64>,
        reduced: f64,
    },
}
impl CoupledPlugin for RecourseResource {
    fn supports_cost_dominance(&self) -> bool {
        self.reversed
    }
    fn dominates(&self, a: &PluginState, b: &PluginState, forward: bool) -> bool {
        match (a.get::<State>(), b.get::<State>(), forward) {
            (
                Some(State::Continuation {
                    visited: a,
                    values: av,
                    ..
                }),
                Some(State::Continuation {
                    visited: b,
                    values: bv,
                    ..
                }),
                true,
            ) => {
                // Fewer visited customers leave every common completion
                // feasible, with no larger canonical mean load. Bellman
                // expectation/min operators preserve componentwise order.
                (*a & *b) == *a && av.iter().zip(bv).all(|(a, b)| a <= b)
            }
            _ => a == b,
        }
    }
    fn has_costs(&self) -> bool {
        self.enabled
    }
    fn objective(&self, state: &PluginState, forward: bool) -> Result<f64> {
        if !self.enabled {
            return Ok(0.);
        }
        match (state.get::<State>(), forward) {
            (Some(State::Continuation { values, .. }), true) => Ok(values[self.problem.capacity]),
            (Some(State::Forward { cost, .. }), true) => Ok(*cost),
            (Some(State::Backward(values)), false) => Ok(values[self.problem.capacity]),
            _ => Err(invalid("stochastic plugin direction mismatch")),
        }
    }
    fn validate(&self, arcs: usize) -> Result<()> {
        if arcs != self.arcs.len() {
            return Err(invalid("stochastic arc shape mismatch"));
        }
        Ok(())
    }
    fn root(&self, forward: bool) -> PluginState {
        if self.reversed && forward {
            return PluginState::new(State::Continuation {
                visited: 0,
                values: vec![0.; self.problem.capacity + 1],
                reduced: 0.,
            });
        }
        PluginState::new(if forward {
            State::Forward {
                route: vec![],
                cost: 0.,
            }
        } else {
            State::Backward(vec![0.; self.problem.capacity + 1])
        })
    }
    fn extend(
        &self,
        state: &PluginState,
        arc: usize,
        forward: bool,
    ) -> Result<Option<PluginState>> {
        let &(i, j) = self
            .arcs
            .get(arc)
            .ok_or_else(|| invalid("unknown stochastic arc"))?;
        match (state.get::<State>(), forward) {
            (
                Some(State::Continuation {
                    visited,
                    values: next,
                    reduced,
                }),
                true,
            ) => {
                // Stored arcs are in the original problem's direction. Search
                // traverses them backwards, prepending one Bellman operator.
                let mut visited = *visited;
                if i != 0 {
                    let bit = 1u64 << (i - 1);
                    if visited & bit != 0 {
                        return Ok(None);
                    }
                    visited |= bit;
                }
                let mean: f64 = self
                    .problem
                    .demands
                    .iter()
                    .enumerate()
                    .filter(|(k, _)| visited & (1u64 << k) != 0)
                    .map(|(_, d)| d.mean())
                    .sum();
                if mean > self.problem.capacity as f64 {
                    return Ok(None);
                }
                let reduced = reduced + self.reduced[arc];
                // Every completion is a depot walk entering i whose load
                // excludes the rest of the suffix; its recourse is at least
                // the walk's detour charges plus a continuation value.
                let prefix = match (&self.prefixes, i != 0) {
                    (Some((prefixes, _)), true) => {
                        let load = self.problem.capacity as f64 - mean
                            + self.problem.demands[i - 1].mean()
                            + 1e-9 * (1. + self.problem.capacity as f64);
                        Some(prefixes.prefix(i, j, load, visited))
                    }
                    _ => None,
                };
                let threshold = self.prefixes.as_ref().map_or(0., |(_, t)| *t);
                let prunes = |continuation: f64, prefix: f64| {
                    let bound = reduced + continuation + prefix;
                    bound
                        >= threshold
                            + 1e-9 * (1. + reduced.abs() + continuation.abs() + prefix.abs())
                };
                if let (Some(prefix), true) = (prefix, j != 0 && self.enabled) {
                    // Failures cost nothing negative and a refill before j
                    // costs at least its detour, so this bounds the Bellman
                    // minimum below without evaluating it.
                    let detour = self.problem.costs[i][0] + self.problem.costs[0][j]
                        - self.problem.costs[i][j];
                    let floor = next.iter().copied().fold(f64::INFINITY, f64::min) + detour.min(0.);
                    if prunes(floor, prefix) {
                        return Ok(None);
                    }
                }
                let mut values = vec![0.; self.problem.capacity + 1];
                if j != 0 && self.enabled {
                    self.problem
                        .prepend_recourse(i, j, next, &mut values, self.policy)?;
                }
                if let Some(prefix) = prefix {
                    let continuation = values.iter().copied().fold(f64::INFINITY, f64::min);
                    if prunes(continuation, prefix) {
                        return Ok(None);
                    }
                }
                Ok(Some(PluginState::new(State::Continuation {
                    visited,
                    values,
                    reduced,
                })))
            }
            (Some(State::Forward { route, .. }), true) => {
                let mut route = route.clone();
                if j != 0 {
                    if route.contains(&j) {
                        return Ok(None);
                    }
                    route.push(j);
                }
                let mean: f64 = self
                    .problem
                    .demands
                    .iter()
                    .enumerate()
                    .filter(|(k, _)| route.contains(&(k + 1)))
                    .map(|(_, d)| d.mean())
                    .sum();
                if mean > self.problem.capacity as f64 {
                    return Ok(None);
                }
                let cost = if self.enabled {
                    self.problem
                        .evaluate_route(&route, self.policy)?
                        .expected_recourse_cost
                } else {
                    0.
                };
                Ok(Some(PluginState::new(State::Forward { route, cost })))
            }
            (Some(State::Backward(next)), false) => {
                let mut values = vec![0.; self.problem.capacity + 1];
                if j != 0 && self.enabled {
                    self.problem
                        .prepend_recourse(i, j, next, &mut values, self.policy)?;
                }
                Ok(Some(PluginState::new(State::Backward(values))))
            }
            _ => Err(invalid("stochastic plugin direction mismatch")),
        }
    }
    fn join(&self, state: &PluginState, suffix: &mut dyn Iterator<Item = usize>) -> Result<bool> {
        let mut current = state.clone();
        for arc in suffix {
            let Some(next) = self.extend(&current, arc, true)? else {
                return Ok(false);
            };
            current = next;
        }
        Ok(true)
    }
}

impl Problem {
    pub(super) fn validate_branches(&self, branches: &[ArcBranch]) -> Result<()> {
        let n = self.demands.len();
        for b in branches {
            if b.from > n || b.to > n || b.from == b.to {
                return Err(invalid(
                    "stochastic branch must name a distinct pair of vertices in 0..=n",
                ));
            }
        }
        Ok(())
    }

    /// Compile stochastic recourse into the ordinary BALDES physical graph.
    /// No deterministic-cost surrogate is used. This adapter uses full NG
    /// elementarity and supports NG_MEMORY_CAPACITY customers (15 by default).
    /// Branch duals align with `branches`; customer duals exclude the depot.
    #[allow(clippy::too_many_arguments)]
    pub fn pricing_graph(
        &self,
        policy: Policy,
        customer_duals: &[f64],
        fleet_dual: f64,
        branches: &[ArcBranch],
        branch_duals: &[f64],
        phase_one: bool,
    ) -> Result<Graph<3>> {
        self.build_pricing_graph(
            policy,
            customer_duals,
            fleet_dual,
            branches,
            branch_duals,
            phase_one,
            false,
            None,
        )
        .map(|(graph, _)| graph)
    }

    #[allow(clippy::too_many_arguments)]
    fn build_pricing_graph(
        &self,
        policy: Policy,
        customer_duals: &[f64],
        fleet_dual: f64,
        branches: &[ArcBranch],
        branch_duals: &[f64],
        phase_one: bool,
        reversed: bool,
        // Pruning threshold and size of the completion subset set `M`.
        completion: Option<(f64, usize, bool)>,
    ) -> Result<(Graph<3>, LoadWalkBounds)> {
        self.validate_branches(branches)?;
        let n = self.demands.len();
        if n > crate::graph::ng::NG_MEMORY_CAPACITY
            || customer_duals.len() != n
            || branch_duals.len() != branches.len()
            || customer_duals
                .iter()
                .chain(branch_duals)
                .any(|d| !d.is_finite())
            || !fleet_dual.is_finite()
        {
            return Err(invalid(
                "invalid stochastic pricing dimensions, duals, or NG customer limit",
            ));
        }
        let sink = n + 1;
        let mut g = Graph::<3>::new(n + 2, 2)?;
        g.source = 0;
        g.sink = sink;
        g.elementary = true;
        g.main_resource = Some(0);
        g.element_count = 0;
        for vertex in &mut g.vertices {
            vertex.elements.clear();
            vertex.memory.clear();
        }
        for j in 1..=n {
            g.add_elementarity_set(&[j])?;
        }
        for v in 0..=sink {
            g.set_bounds(v, 0, 0., (n + 1) as f64)?;
            // Scalar arithmetic is only a necessary filter. The plugin checks
            // expected capacity by canonical forward route replay at joins.
            g.set_bounds(
                v,
                1,
                0.,
                self.capacity as f64 + 1e-6 * (1. + self.capacity as f64),
            )?;
        }
        for r in 0..2 {
            g.initial[r] = Some(0.);
            g.set_bounds(0, r, 0., 0.)?;
        }
        let mut arcs = Vec::new();
        for i in 0..=n {
            for j in 0..=n {
                let (from, to) = if reversed { (j, i) } else { (i, j) };
                if i == j
                    || branches
                        .iter()
                        .any(|b| !b.required && b.from == from && b.to == to)
                {
                    continue;
                }
                let id = g.add_arc(
                    i,
                    if j == 0 { sink } else { j },
                    if phase_one { 0. } else { self.costs[from][to] },
                )?;
                let arc = g.arc_mut(id)?;
                arc.reduced_cost = arc.cost
                    - if j == 0 { 0. } else { customer_duals[j - 1] }
                    - if i == 0 { fleet_dual } else { 0. };
                for (b, dual) in branches.iter().zip(branch_duals) {
                    if b.from == from && b.to == to {
                        arc.reduced_cost -= dual;
                    }
                }
                arc.consumption[0] = 1.;
                arc.consumption[1] = if j == 0 {
                    0.
                } else {
                    self.demands[j - 1].mean()
                };
                arcs.push((from, to));
            }
        }
        let scalar = g
            .arcs
            .iter()
            .map(|a| a.reduced_cost.abs())
            .fold(0., f64::max);
        let travel = self.costs.iter().flatten().copied().fold(0., f64::max);
        let magnitude = (n + 1) as f64 * scalar + 3. * n as f64 * travel;
        if !magnitude.is_finite() || magnitude >= crate::labeling::NO_ROUTE_COST / 4. {
            return Err(invalid(
                "stochastic pricing cost scale exceeds BALDES range; rescale costs",
            ));
        }
        let reduced: Vec<_> = g.arcs.iter().map(|a| a.reduced_cost).collect();
        let charges = if phase_one {
            vec![0.; n]
        } else {
            self.recourse_customer_lower_bounds()
        };
        let weighted: Vec<_> = arcs
            .iter()
            .zip(&reduced)
            .map(|(&(from, to), &r)| (from, to, r))
            .collect();
        let (bounds, prefixes) = match completion.filter(|_| reversed) {
            Some((threshold, size, knapsack)) => {
                // M: customers with the largest positive dual per unit of
                // mean demand (Florio et al. 2023).
                let mut members: Vec<_> = (1..=n).filter(|&j| customer_duals[j - 1] > 0.).collect();
                let ratio = |j: usize| customer_duals[j - 1] / self.demands[j - 1].mean().max(1e-9);
                members.sort_by(|&a, &b| ratio(b).total_cmp(&ratio(a)).then(a.cmp(&b)));
                members.truncate(size.min(MAX_COMPLETION_SUBSET_SIZE));
                let subsets = SubsetWalkBounds::new(self, &weighted, &charges, members, knapsack);
                let bounds = subsets.unrestricted().clone();
                (bounds, Some((Arc::new(subsets), threshold)))
            }
            None => (LoadWalkBounds::new(self, &weighted, &charges), None),
        };
        g.set_coupled_resources(&[CoupledResource::Plugin(PluginResource::new(
            RecourseResource {
                problem: self.clone(),
                policy,
                arcs,
                enabled: !phase_one,
                reversed,
                reduced,
                prefixes,
            },
        ))])?;
        Ok((g, bounds))
    }

    /// Load-indexed walk (q-route) lower bound on every elementary route's
    /// stochastic reduced cost, valid after any pricing interruption.
    pub(super) fn route_relaxation_bound(
        &self,
        policy: Policy,
        customer_duals: &[f64],
        fleet_dual: f64,
        branches: &[ArcBranch],
        branch_duals: &[f64],
        phase_one: bool,
    ) -> Result<Option<f64>> {
        let (_, bounds) = self.build_pricing_graph(
            policy,
            customer_duals,
            fleet_dual,
            branches,
            branch_duals,
            phase_one,
            true,
            None,
        )?;
        Ok(bounds.route().map(|b| b - 1e-9 * (1. + b.abs())))
    }

    // A customer can incur at most one negative preventive detour per visit.
    // Charging each customer's own bound is tighter than n times the worst
    // detour, and remains safe for rounded/nonmetric distance matrices.
    pub(super) fn recourse_customer_lower_bounds(&self) -> Vec<f64> {
        (1..=self.demands.len())
            .map(|j| {
                (0..=self.demands.len())
                    .map(|i| self.costs[i][0] + self.costs[0][j] - self.costs[i][j])
                    .fold(0., f64::min)
                    * (1. + 1e-10)
            })
            .collect()
    }

    /// Travel cost plus per-customer recourse lower bounds: a lower bound on
    /// the expected cost of the route under either policy.
    pub fn relaxed_route_cost(&self, route: &[usize]) -> f64 {
        let charges = self.recourse_customer_lower_bounds();
        let mut from = 0;
        let mut cost = 0.;
        for &to in route.iter().chain(std::iter::once(&0)) {
            cost += self.costs[from][to];
            if to != 0 {
                cost += charges[to - 1];
            }
            from = to;
        }
        cost
    }

    pub(super) fn recourse_lower_bound(&self) -> f64 {
        self.recourse_customer_lower_bounds().iter().sum()
    }

    fn walk_pricing_bound(&self, graph: &Graph<3>, phase_one: bool) -> Option<f64> {
        use crate::core::rounding::add_down;
        let mut distances = vec![f64::INFINITY; graph.vertices.len()];
        distances[graph.source] = 0.;
        let mut best = f64::INFINITY;
        for _ in 0..=self.demands.len() {
            let mut next = vec![f64::INFINITY; distances.len()];
            for arc in &graph.arcs {
                let from = arc.tail;
                let to = arc.head;
                if distances[from].is_finite() {
                    next[to] = next[to].min(add_down(distances[from], arc.reduced_cost)?);
                }
            }
            best = best.min(next[graph.sink]);
            distances = next;
        }
        if !best.is_finite() {
            return None;
        }
        let bound = add_down(
            best,
            if phase_one {
                0.
            } else {
                self.recourse_lower_bound()
            },
        )?;
        Some(bound - 1e-9 * (1. + bound.abs()))
    }

    /// Solve stochastic reduced-cost pricing using BALDES labeling. Limits
    /// return an incomplete status and never authorize LP convergence.
    #[allow(clippy::too_many_arguments)]
    pub fn price(
        &self,
        policy: Policy,
        customer_duals: &[f64],
        fleet_dual: f64,
        branches: &[ArcBranch],
        branch_duals: &[f64],
        phase_one: bool,
        options: PricingOptions,
    ) -> Result<PricingResult> {
        let started = Instant::now();
        let pruning = options
            .completion_threshold
            .filter(|t| !options.bidirectional && t.is_finite())
            .map(|t| t.min(0.));
        let (graph, bounds) = self.build_pricing_graph(
            policy,
            customer_duals,
            fleet_dual,
            branches,
            branch_duals,
            phase_one,
            !options.bidirectional,
            pruning.map(|t| {
                (
                    t,
                    options.completion_subset_size,
                    options.completion_knapsack,
                )
            }),
        )?;
        let walk = self.walk_pricing_bound(&graph, phase_one);
        let loaded = bounds.route().map(|b| b - 1e-9 * (1. + b.abs()));
        let mut result = PricingResult {
            status: PricingStatus::WorkLimit,
            routes: vec![],
            minimum_reduced_cost: None,
            lower_bound: match (walk, loaded) {
                (Some(a), Some(b)) => Some(a.max(b)),
                (a, b) => a.or(b),
            },
            statistics: Default::default(),
        };
        if self.demands.is_empty() {
            result.status = PricingStatus::Complete;
            return Ok(result);
        }
        if options.max_labels == 0 || options.max_columns == 0 {
            return Ok(result);
        }
        if started.elapsed() >= options.time_limit {
            result.status = PricingStatus::TimeLimit;
            return Ok(result);
        }
        let mut solver = crate::pricing::Solver::<3>::new(graph.vertices.len(), 2)?;
        solver.graph = graph;
        solver.controls.max_labels = options.max_labels;
        solver.controls.max_route_arcs = Some(self.demands.len() + 1);
        solver.controls.time_limit = options.time_limit.saturating_sub(started.elapsed());
        solver.set_option(
            "BALDESuseBidirectionalSearch",
            if options.bidirectional { 1. } else { 0. },
        )?;
        let priced = match solver.price_phase_partial(0, options.max_columns, false) {
            Ok(p) => p,
            Err(e) if e.message.contains("time limit") => {
                result.status = PricingStatus::TimeLimit;
                return Ok(result);
            }
            Err(e) if e.message.contains("limit") => return Ok(result),
            Err(e) => return Err(e),
        };
        result.statistics = priced.statistics;
        for r in priced.columns {
            let mut customers: Vec<_> = r
                .nodes
                .iter()
                .skip(1)
                .take(r.nodes.len() - 2)
                .map(|&v| v as usize)
                .collect();
            if !options.bidirectional {
                customers.reverse();
            }
            let evaluation = self.evaluate_route(&customers, policy)?;
            let mut rc = if phase_one { 0. } else { evaluation.total_cost } - fleet_dual;
            for &j in &customers {
                rc -= customer_duals[j - 1];
            }
            for (b, dual) in branches.iter().zip(branch_duals) {
                if route_uses(&customers, *b) {
                    rc -= dual;
                }
            }
            if !rc.is_finite() || (rc - r.cost).abs() > 1e-7 * (1. + rc.abs()) {
                return Err(invalid("stochastic pricing objective replay mismatch"));
            }
            result.routes.push(PricedRoute {
                customers,
                evaluation,
                reduced_cost: rc,
            });
        }
        result
            .routes
            .sort_by(|a, b| a.reduced_cost.total_cmp(&b.reduced_cost));
        result.minimum_reduced_cost = result.routes.first().map(|r| r.reduced_cost);
        result.status = if started.elapsed() >= options.time_limit {
            PricingStatus::TimeLimit
        } else if priced.exact {
            PricingStatus::Complete
        } else {
            PricingStatus::WorkLimit
        };
        if result.status != PricingStatus::Complete {
            result.minimum_reduced_cost = None;
        } else if let Some(threshold) = pruning {
            result.minimum_reduced_cost = Some(
                result
                    .minimum_reduced_cost
                    .map_or(threshold, |m| m.min(threshold)),
            );
        }
        Ok(result)
    }

    /// Split pricing by each route's last customer. Part `g` forbids the
    /// depot arcs of every customer outside it, so the parts are disjoint and
    /// together contain every route. Returns `None` when there is no split.
    pub(super) fn last_customer_parts(
        &self,
        branches: &[ArcBranch],
        branch_duals: &[f64],
        parts: usize,
    ) -> Option<Vec<(Vec<ArcBranch>, Vec<f64>)>> {
        let n = self.demands.len();
        let parts = parts.min(n);
        if parts < 2 {
            return None;
        }
        Some(
            (0..parts)
                .map(|g| {
                    let mut part_branches = branches.to_vec();
                    let mut part_duals = branch_duals.to_vec();
                    for j in (1..=n).filter(|j| (j - 1) % parts != g) {
                        part_branches.push(ArcBranch {
                            from: j,
                            to: 0,
                            required: false,
                        });
                        part_duals.push(0.);
                    }
                    (part_branches, part_duals)
                })
                .collect(),
        )
    }

    /// Exact pricing over `parts` disjoint last-customer parts, each a full
    /// reverse labeling run on the ambient rayon pool. Minima and bounds
    /// combine by minimum; the result is Complete only if every part is.
    #[allow(clippy::too_many_arguments)]
    pub fn price_partitioned(
        &self,
        policy: Policy,
        customer_duals: &[f64],
        fleet_dual: f64,
        branches: &[ArcBranch],
        branch_duals: &[f64],
        phase_one: bool,
        options: PricingOptions,
        parts: usize,
    ) -> Result<PricingResult> {
        let Some(split) = self
            .last_customer_parts(branches, branch_duals, parts)
            .filter(|_| !options.bidirectional)
        else {
            return self.price(
                policy,
                customer_duals,
                fleet_dual,
                branches,
                branch_duals,
                phase_one,
                options,
            );
        };
        // Parts share the label budget, so peak label memory stays that of
        // one serial call.
        let mut options = options;
        options.max_labels = options.max_labels.div_ceil(split.len());
        let results = split
            .par_iter()
            .map(|(part_branches, part_duals)| {
                self.price(
                    policy,
                    customer_duals,
                    fleet_dual,
                    part_branches,
                    part_duals,
                    phase_one,
                    options,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(combine_parts(results))
    }

    /// Exact pricing over last-customer parts that retries only the parts
    /// stopped by their work limit. Each retry doubles their label budget
    /// while the retried parts together stay within `label_cap` labels, and
    /// stops once any part finds a route below `-tolerance`. Completed parts
    /// keep their results, so hard parts search deeper in the same memory.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn price_adaptive(
        &self,
        policy: Policy,
        customer_duals: &[f64],
        fleet_dual: f64,
        branches: &[ArcBranch],
        branch_duals: &[f64],
        phase_one: bool,
        options: PricingOptions,
        parts: usize,
        label_cap: usize,
        tolerance: f64,
    ) -> Result<PricingResult> {
        let started = Instant::now();
        let split = self
            .last_customer_parts(branches, branch_duals, parts)
            .filter(|_| !options.bidirectional)
            .unwrap_or_else(|| vec![(branches.to_vec(), branch_duals.to_vec())]);
        let mut budget = options.max_labels.div_ceil(split.len()).max(1);
        let mut results: Vec<Option<PricingResult>> = vec![None; split.len()];
        let mut statistics = crate::labeling::Statistics::default();
        let mut pending: Vec<usize> = (0..split.len()).collect();
        loop {
            let mut part_options = options;
            part_options.max_labels = budget;
            part_options.time_limit = options.time_limit.saturating_sub(started.elapsed());
            let priced = pending
                .par_iter()
                .map(|&k| {
                    let (part_branches, part_duals) = &split[k];
                    self.price(
                        policy,
                        customer_duals,
                        fleet_dual,
                        part_branches,
                        part_duals,
                        phase_one,
                        part_options,
                    )
                })
                .collect::<Result<Vec<_>>>()?;
            for (&k, part) in pending.iter().zip(priced) {
                statistics.absorb(&part.statistics);
                results[k] = Some(part);
            }
            let improving = results
                .iter()
                .flatten()
                .any(|r| r.routes.iter().any(|route| route.reduced_cost < -tolerance));
            pending.retain(|&k| {
                results[k]
                    .as_ref()
                    .is_some_and(|r| r.status == PricingStatus::WorkLimit)
            });
            let next = budget
                .saturating_mul(2)
                .min(label_cap / pending.len().max(1));
            if improving
                || pending.is_empty()
                || next <= budget
                || started.elapsed() >= options.time_limit
            {
                break;
            }
            budget = next;
        }
        let mut combined = combine_parts(results.into_iter().flatten().collect());
        combined.statistics = statistics;
        Ok(combined)
    }
}

/// Worst status of disjoint pricing parts.
pub(super) fn combined_status(statuses: impl IntoIterator<Item = PricingStatus>) -> PricingStatus {
    let rank = |s: PricingStatus| match s {
        PricingStatus::Complete => 0,
        PricingStatus::Heuristic => 1,
        PricingStatus::WorkLimit => 2,
        PricingStatus::TimeLimit => 3,
    };
    statuses
        .into_iter()
        .max_by_key(|&s| rank(s))
        .unwrap_or(PricingStatus::Complete)
}

/// A global bound exists only if every part supplies one.
pub(super) fn combined_bound(bounds: impl IntoIterator<Item = Option<f64>>) -> Option<f64> {
    bounds
        .into_iter()
        .try_fold(f64::INFINITY, |acc, b| b.map(|b| acc.min(b)))
        .filter(|b| b.is_finite())
}

fn combine_parts(parts: Vec<PricingResult>) -> PricingResult {
    let status = combined_status(parts.iter().map(|p| p.status));
    let lower_bound = combined_bound(parts.iter().map(|p| p.lower_bound));
    // A complete part with no minimum has an empty route domain.
    let minimum_reduced_cost = (status == PricingStatus::Complete)
        .then(|| {
            parts
                .iter()
                .filter_map(|p| p.minimum_reduced_cost)
                .fold(None, |m: Option<f64>, x| Some(m.map_or(x, |m| m.min(x))))
        })
        .flatten();
    let mut result = PricingResult {
        status,
        routes: vec![],
        minimum_reduced_cost,
        lower_bound,
        statistics: Default::default(),
    };
    for mut part in parts {
        result.statistics.absorb(&part.statistics);
        result.routes.append(&mut part.routes);
    }
    result
        .routes
        .sort_by(|a, b| a.reduced_cost.total_cmp(&b.reduced_cost));
    result
}

pub(super) fn route_uses(route: &[usize], branch: ArcBranch) -> bool {
    let mut i = 0;
    for j in route.iter().copied().chain(std::iter::once(0)) {
        if i == branch.from && j == branch.to {
            return true;
        }
        i = j;
    }
    false
}
