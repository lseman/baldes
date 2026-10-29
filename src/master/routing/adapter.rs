//! Routing-model adapter to the shared network pricing coordinator.
use super::*;
use crate::master::pricing::{
    Additive, Collected, Context, Envelopes, NetworkPricingSchedule, Networks, Request, Scheduler,
};

// Refresh an exact pricing bound periodically while using discovery passes.
const EXACT_BOUND_REFRESH_INTERVAL: usize = 16;

pub(super) struct RootPricing {
    scheduler: Scheduler,
    pub master: Option<(f64, f64)>,
    pub rhs: Vec<f64>,
    pub correction: Option<f64>,
    pub gradient: Option<Vec<f64>>,
    gap: (f64, f64),
    admission: bool,
    pub attempts: usize,
    pub oracle_calls: usize,
    pub node_depth: usize,
    pub columns_only: bool,
    pub force_exact: bool,
    multiplicity: f64,
}
impl Default for RootPricing {
    fn default() -> Self {
        Self::new(&cg::Options::default())
    }
}
impl RootPricing {
    pub fn new(options: &cg::Options) -> Self {
        Self {
            scheduler: Scheduler::new(NetworkPricingSchedule::FullSweep, 1, options.tolerance),
            master: None,
            rhs: vec![],
            correction: None,
            gradient: None,
            gap: (options.gap_absolute, options.gap_relative),
            admission: !options.stabilize && options.proximal.is_none(),
            attempts: 0,
            oracle_calls: 0,
            node_depth: options.node_depth,
            columns_only: false,
            force_exact: false,
            multiplicity: options.multiplicity,
        }
    }
    pub fn complete(&self) -> bool {
        self.scheduler.complete
    }
    fn quantity_upper<const D: usize>(&self, problem: &RoutingContext<D>) -> f64 {
        // Recognize the model's fleet row structurally. Other routing-row
        // layouts retain the caller's explicit CG multiplicity contract.
        let coefficient = |entries: &[(usize, f64)], expected: f64| {
            let mut seen = false;
            for &(row, a) in entries {
                if row != 0 || a == 0. {
                    continue;
                }
                if seen || a != expected {
                    return false;
                }
                seen = true;
            }
            seen == (expected != 0.)
        };
        let fleet = problem
            .model
            .rows
            .first()
            .is_some_and(|row| matches!(row.sense, RowSense::LessEqual | RowSense::Equal));
        let on_arcs = coefficient(&problem.model.networks[0].route_entries, 0.)
            && problem.model.networks[0]
                .graph
                .arcs
                .iter()
                .zip(&problem.model.networks[0].arc_entries)
                .all(|(arc, entries)| {
                    coefficient(
                        entries,
                        if arc.head == problem.model.networks[0].graph.sink {
                            1.
                        } else {
                            0.
                        },
                    )
                });
        let on_routes = coefficient(&problem.model.networks[0].route_entries, 1.)
            && problem.model.networks[0]
                .arc_entries
                .iter()
                .all(|e| coefficient(e, 0.));
        if fleet && (on_arcs || on_routes) {
            problem.model.rows[0].rhs.max(0.)
        } else {
            self.multiplicity
        }
    }
}
impl<const D: usize> State<D> {
    pub(super) fn price(
        &mut self,
        problem: &RoutingContext<D>,
        phase: usize,
        duals: &[f64],
        floor: usize,
    ) -> Result<Answer> {
        let started = std::time::Instant::now();
        self.pricing.attempts += 1;
        self.pricing.correction = None;
        self.pricing.gradient = None;
        self.pricing.columns_only = false;
        let force_exact = std::mem::take(&mut self.pricing.force_exact);
        let solver_phase = phase + floor;
        let mut pricing_deadline = self.deadline;
        if !force_exact
            && self.pricing.attempts % EXACT_BOUND_REFRESH_INTERVAL != 0
            && self
                .solver
                .phases
                .get(solver_phase)
                .is_some_and(|p| p.exact())
            && !self.solver.cuts.cuts.is_empty()
            && !self
                .solver
                .enumeration
                .as_ref()
                .is_some_and(|e| e.completed)
        {
            let mut adapter = RoutingAdapter {
                state: self,
                problem,
                floor,
            };
            // Candidate generation needs only the current cached objective.
            // The full raw arc vector is built if exact pricing is needed.
            let offset = adapter.state.reprice(problem, duals)?;
            let remaining = adapter
                .state
                .deadline
                .map_or(std::time::Duration::MAX, |end| {
                    end.saturating_duration_since(std::time::Instant::now())
                });
            if remaining.is_zero() {
                return Err(crate::incomplete("network pricing time limit"));
            }
            let solver = adapter.enter(0);
            let saved_limit = solver.controls.time_limit;
            if let Some(end) = started.checked_add(saved_limit) {
                pricing_deadline = Some(pricing_deadline.map_or(end, |old| old.min(end)));
            }
            let remaining = pricing_deadline.map_or(remaining, |end| {
                remaining.min(end.saturating_duration_since(std::time::Instant::now()))
            });
            if remaining.is_zero() {
                return Err(crate::incomplete("network pricing time limit"));
            }
            solver.controls.time_limit = remaining;
            let result = solver.price_phase(solver_phase, solver.controls.max_columns, false);
            solver.controls.time_limit = saved_limit;
            let result = result?;
            let minimum = result.minimum_reduced_cost + offset;
            let collected = adapter.collect(0, result.columns, offset, duals)?;
            if collected.fresh > 0 {
                adapter.state.pricing.columns_only = true;
                adapter.state.pricing_seconds += started.elapsed().as_secs_f64();
                return Ok(Answer {
                    columns: collected.columns,
                    minimum,
                    exact: false,
                    bound_minimum: None,
                });
            }
        }
        let rhs = self.pricing.rhs.clone();
        let context = Context {
            phase,
            duals,
            rhs: &rhs,
            deadline: pricing_deadline,
            admission: self.pricing.admission,
            master: self.pricing.master,
            gap: self.pricing.gap,
        };
        let fresh = self.pricing.scheduler.fresh();
        let mut scheduler = std::mem::replace(&mut self.pricing.scheduler, fresh);
        let result = crate::master::pricing::price(
            &mut RoutingAdapter {
                state: self,
                problem,
                floor,
            },
            &mut scheduler,
            context,
        );
        self.pricing.scheduler = scheduler;
        self.pricing_seconds += started.elapsed().as_secs_f64();
        let batch = result?;
        self.pricing.correction = batch.correction;
        self.pricing.gradient = batch.gradient;
        Ok(batch.answer)
    }
}
struct RoutingAdapter<'a, const D: usize> {
    state: &'a mut State<D>,
    problem: &'a RoutingContext<D>,
    floor: usize,
}
impl<const D: usize> Networks<D> for RoutingAdapter<'_, D> {
    fn active(&self) -> Vec<usize> {
        if self.state.pricing.quantity_upper(self.problem) > 0. {
            vec![0]
        } else {
            vec![]
        }
    }
    fn prepare(
        &mut self,
        _index: usize,
        phase: usize,
        duals: &[f64],
        admission: bool,
    ) -> Result<Request> {
        let offset = self.state.reprice(self.problem, duals)?;
        let map = self.problem.row_map.borrow();
        let raw: Vec<_> = self.problem.model.networks[0]
            .graph
            .arcs
            .iter()
            .enumerate()
            .map(|(p, arc)| {
                let mut cost = arc.cost;
                for &(row, a) in &self.problem.model.networks[0].arc_entries[p] {
                    cost -= duals[map[row]] * a;
                }
                for &(row, a) in &self.state.cut.capacity_arc_entries[p] {
                    cost -= duals[row] * a;
                }
                cost
            })
            .collect();
        let cached: Vec<_> = self
            .state
            .solver
            .graph
            .arcs
            .iter()
            .map(|a| a.reduced_cost)
            .collect();
        self.state.solver.cache_arc_costs(&raw, &cached)?;
        let lower = 0.;
        let upper = self.state.pricing.quantity_upper(self.problem);
        let supported = self.state.solver.cuts.cuts.iter().all(|cut| cut.dual == 0.)
            && self
                .state
                .solver
                .cuts
                .knapsack_cuts
                .iter()
                .all(|cut| cut.dual == 0.)
            && self.state.solver.presence_rows().is_empty();
        let domain = Additive {
            graph: &self.state.solver.graph,
            costs: &raw,
            lower,
            upper,
            supported,
        };
        let retained = self
            .state
            .solver
            .enumeration
            .as_ref()
            .is_some_and(|e| e.completed);
        let envelopes = if admission && !retained {
            Envelopes {
                nonnegative: domain.nonnegative(offset),
                finite: domain.required(offset),
                near_zero: domain.near_zero(offset, self.state.pricing.scheduler.tolerance),
                original: false,
            }
        } else {
            Envelopes::default()
        };
        let phase = phase + self.floor;
        Ok(Request {
            phase,
            exact: self.state.solver.phases[phase].exact(),
            offset,
            lower,
            upper,
            envelopes,
        })
    }
    fn enter(&mut self, _index: usize) -> &mut Solver<D> {
        self.state.pricing.oracle_calls += 1;
        &mut self.state.solver
    }
    fn retained(
        &mut self,
        _index: usize,
        duals: &[f64],
    ) -> Option<Result<crate::master::pricing::Retained>> {
        if !self
            .state
            .solver
            .enumeration
            .as_ref()
            .is_some_and(|e| e.completed)
        {
            return None;
        }
        Some((|| {
            self.state.pricing.oracle_calls += 1;
            let routes = self
                .state
                .solver
                .inspect_enumeration(f64::INFINITY)?
                .unwrap_or_default();
            let mut columns = vec![];
            let upper = self.state.pricing.quantity_upper(self.problem);
            for mut route in routes {
                if route.arcs.iter().any(|id| {
                    self.problem.arc_position(*id).is_some_and(|p| {
                        self.state
                            .solver
                            .forbidden_arcs
                            .get(p)
                            .copied()
                            .unwrap_or(false)
                    })
                }) {
                    continue;
                }
                if !self
                    .state
                    .solver
                    .route_satisfies_relaxation(&route.nodes, &route.arcs)?
                {
                    continue;
                }
                let tag = register(self.state, &route)?;
                let column = self.state.column(self.problem, &route, duals, tag)?;
                route.cost = column.reduced_cost;
                self.state.routes[tag as usize].cost = route.cost;
                columns.push((column, upper));
            }
            Ok(crate::master::pricing::Retained {
                minimum_bound: None,
                columns,
                independent_caps: false,
                maximum: self.state.solver.controls.max_columns,
            })
        })())
    }
    fn collect(
        &mut self,
        _index: usize,
        routes: Vec<Route>,
        offset: f64,
        duals: &[f64],
    ) -> Result<Collected> {
        let mut columns = vec![];
        let mut best = None;
        let mut fresh = 0;
        for mut route in routes {
            let is_new = !self.state.route_tags.contains_key(&route.identity());
            route.cost += offset;
            let tag = register(self.state, &route)?;
            let column = self.state.column(self.problem, &route, duals, tag)?;
            if best
                .as_ref()
                .map_or(true, |(value, _)| column.reduced_cost < *value)
            {
                best = Some((column.reduced_cost, column.entries.clone()));
            }
            fresh += usize::from(
                is_new && column.reduced_cost < -self.state.pricing.scheduler.tolerance,
            );
            columns.push(column);
        }
        Ok(Collected {
            columns,
            fresh,
            best_entries: best.map(|(_, entries)| entries),
        })
    }
}

#[cfg(test)]
#[path = "../../../tests/unit/master/routing/pricing_tests.rs"]
mod tests;
