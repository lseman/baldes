//! Shared EV set-partitioning master on the existing LP backend interface.
//! Phase I minimizes artificial coverage; phase II minimizes travel cost.
//! The two pricing backends differ only in how they find a complete route.

use super::{
    nonnegative, DirectOptions, EvPricingStatistics, EvPricingStatus, LocationKind, Problem,
    Schedule, SubpathOptions, Visit,
};
use crate::master::{LpMaster, LpSolveOutcome, RowSense};
use crate::{invalid, Result};
use std::collections::BTreeMap;
use std::time::{Duration, Instant};

#[derive(Clone, Copy, Debug)]
pub enum EvPricingBackend {
    Direct(DirectOptions),
    Subpath(SubpathOptions),
}

#[derive(Clone, Copy, Debug)]
pub struct EvColumnGenerationOptions {
    pub backend: EvPricingBackend,
    /// Maximum additional LP solves, across both phases.
    pub max_iterations: usize,
    /// LP solves honor the remaining budget. Pricing is bounded by its work
    /// limits; elapsed time is checked before and after each pricing call.
    pub time_limit: Duration,
    pub reduced_cost_tolerance: f64,
    pub lp_tolerance: f64,
}

impl Default for EvColumnGenerationOptions {
    fn default() -> Self {
        Self {
            backend: EvPricingBackend::Direct(DirectOptions::default()),
            max_iterations: 1000,
            time_limit: Duration::from_secs(3600),
            reduced_cost_tolerance: 1e-7,
            lp_tolerance: 1e-7,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EvMasterPhase {
    Feasibility,
    TravelCost,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EvColumnGenerationStatus {
    /// Full EV LP convergence at the supplied tolerances, in f64 arithmetic.
    /// This is neither an integer routing optimum nor a rigorous bound.
    Converged,
    /// Positive phase-I optimum after complete pricing, at the LP tolerances.
    Infeasible,
    PricingIncomplete(EvPricingStatus),
    IterationLimit,
    TimeLimit,
    LpStopped(LpSolveOutcome),
    NumericalFailure,
}

#[derive(Clone, Debug)]
pub struct EvColumn {
    pub visits: Vec<Visit>,
    pub schedule: Schedule,
    pub customers: Vec<usize>,
    pub lp_column: usize,
}

#[derive(Clone, Debug)]
pub struct EvMasterSnapshot {
    pub phase: EvMasterPhase,
    pub lp_objective: f64,
    pub artificial_usage: f64,
    /// Aligned to EvMaster::columns() at the time of the solve. A column added
    /// afterward has implicit value zero in this previous feasible point.
    pub route_values: Vec<f64>,
    /// Travel objective only when artificial coverage is zero to tolerance.
    /// A restricted-pool feasible objective is not a full-model lower bound.
    pub travel_objective: Option<f64>,
}

#[derive(Clone, Debug)]
pub struct EvIteration {
    pub phase: EvMasterPhase,
    pub lp_objective: f64,
    pub artificial_usage: f64,
    pub pricing_status: EvPricingStatus,
    /// Best observed reduced cost. It has global minimum meaning only when
    /// pricing_status is Complete, within the floating-point pricing contract.
    pub minimum_reduced_cost: Option<f64>,
    pub pricing_statistics: EvPricingStatistics,
}

#[derive(Clone, Debug)]
pub struct EvColumnGenerationResult {
    pub status: EvColumnGenerationStatus,
    pub snapshot: Option<EvMasterSnapshot>,
    pub iterations: Vec<EvIteration>,
}

/// Owns a single-vehicle-type EV problem and an initially empty LP backend.
/// Customer rows partition customers exactly once; an optional fleet row bounds
/// total route weight. No integer, cut or branching claims are made here.
pub struct EvMaster<M: LpMaster> {
    problem: Problem,
    phase_one_problem: Problem,
    lp: M,
    departure: f64,
    customer_rows: Vec<Option<usize>>,
    fleet_row: Option<usize>,
    artificials: Vec<usize>,
    columns: Vec<EvColumn>,
    orders: BTreeMap<Vec<usize>, usize>,
    phase: EvMasterPhase,
}

impl<M: LpMaster> EvMaster<M> {
    pub fn new(
        problem: Problem,
        mut lp: M,
        departure: f64,
        fleet_limit: Option<f64>,
    ) -> Result<Self> {
        problem.validate()?;
        if !nonnegative(departure)
            || fleet_limit.map_or(false, |v| !nonnegative(v))
            || !lp.rhs().is_empty()
            || !lp.costs().is_empty()
            || lp.objective_offset() != 0.0
        {
            return Err(invalid(
                "EV master requires an empty LP, finite departure and nonnegative fleet limit",
            ));
        }
        let mut customer_rows = vec![None; problem.locations.len()];
        let mut artificials = Vec::new();
        for (v, location) in problem.locations.iter().enumerate() {
            if matches!(location.kind, LocationKind::Customer { .. }) {
                let row = lp.add_row(RowSense::Equal, 1.0, &[])?;
                customer_rows[v] = Some(row);
                artificials.push(lp.add_column(1.0, &[(row, 1.0)])?);
            }
        }
        let fleet_row = fleet_limit
            .map(|limit| lp.add_row(RowSense::LessEqual, limit, &[]))
            .transpose()?;
        let mut phase_one_problem = problem.clone();
        for arc in phase_one_problem.travel.iter_mut().flatten().flatten() {
            arc.cost = 0.0;
        }
        Ok(Self {
            problem,
            phase_one_problem,
            lp,
            departure,
            customer_rows,
            fleet_row,
            artificials,
            columns: Vec::new(),
            orders: BTreeMap::new(),
            phase: EvMasterPhase::Feasibility,
        })
    }

    pub fn problem(&self) -> &Problem {
        &self.problem
    }
    pub fn columns(&self) -> &[EvColumn] {
        &self.columns
    }
    pub fn lp(&self) -> &M {
        &self.lp
    }
    pub fn phase(&self) -> EvMasterPhase {
        self.phase
    }

    /// Replay and add a seed/generated complete route. Different charging
    /// amounts on the same stop order describe the same travel-cost column.
    /// Returns its LP index, reusing an existing column for a duplicate order.
    pub fn add_route(&mut self, visits: &[Visit]) -> Result<usize> {
        let schedule = self.problem.replay(visits, self.departure)?;
        let order = visits.iter().map(|v| v.location).collect::<Vec<_>>();
        if let Some(&column) = self.orders.get(&order) {
            return Ok(column);
        }
        let mut customers = visits
            .iter()
            .filter_map(|v| self.customer_rows[v.location].map(|_| v.location))
            .collect::<Vec<_>>();
        if customers.is_empty() {
            return Err(invalid("EV master column must serve at least one customer"));
        }
        customers.sort_unstable();
        let mut entries = customers
            .iter()
            .map(|&v| (self.customer_rows[v].unwrap(), 1.0))
            .collect::<Vec<_>>();
        if let Some(row) = self.fleet_row {
            entries.push((row, 1.0));
        }
        entries.sort_by_key(|e| e.0);
        let cost = if self.phase == EvMasterPhase::TravelCost {
            schedule.travel_cost
        } else {
            0.0
        };
        let column = self.lp.add_column(cost, &entries)?;
        self.orders.insert(order, column);
        self.columns.push(EvColumn {
            visits: visits.to_vec(),
            schedule,
            customers,
            lp_column: column,
        });
        Ok(column)
    }

    fn snapshot(&self, tolerance: f64) -> Result<Option<EvMasterSnapshot>> {
        if !self.lp.solved()
            || self.lp.values().len() != self.lp.costs().len()
            || self.lp.duals().len() != self.lp.rhs().len()
        {
            return Ok(None);
        }
        let mut activity = vec![0.0; self.lp.rhs().len()];
        let mut objective = self.lp.objective_offset();
        for (column, &value) in self.lp.values().iter().enumerate() {
            if !value.is_finite()
                || value < -tolerance
                || value > self.lp.upper()[column] + tolerance
            {
                return Ok(None);
            }
            objective += value * self.lp.costs()[column];
            let mut reduced = self.lp.costs()[column];
            for (row, a) in self.lp.column_entries(column)? {
                activity[row] += a * value;
                reduced -= a * self.lp.duals()[row];
            }
            if !reduced.is_finite() || (self.lp.upper()[column] > 0.0 && reduced < -tolerance) {
                return Ok(None);
            }
        }
        for (row, &a) in activity.iter().enumerate() {
            let (rhs, dual) = (self.lp.rhs()[row], self.lp.duals()[row]);
            if !a.is_finite() || !dual.is_finite() {
                return Ok(None);
            }
            let valid = match self.lp.senses()[row] {
                RowSense::Equal => (a - rhs).abs() <= tolerance,
                RowSense::LessEqual => a <= rhs + tolerance && dual <= tolerance,
                RowSense::GreaterEqual => a >= rhs - tolerance && dual >= -tolerance,
            };
            if !valid {
                return Ok(None);
            }
        }
        let dual_objective = self
            .lp
            .rhs()
            .iter()
            .zip(self.lp.duals())
            .map(|(b, d)| b * d)
            .sum::<f64>();
        if !objective.is_finite()
            || !self.lp.objective().is_finite()
            || !dual_objective.is_finite()
            || (objective - self.lp.objective()).abs() > tolerance * (1.0 + objective.abs())
            || (objective - dual_objective).abs() > tolerance * (1.0 + objective.abs())
        {
            return Ok(None);
        }
        let artificial_usage = self
            .artificials
            .iter()
            .map(|&c| self.lp.values()[c].max(0.0))
            .sum::<f64>();
        let route_values = self
            .columns
            .iter()
            .map(|c| self.lp.values()[c.lp_column])
            .collect::<Vec<_>>();
        let travel = route_values
            .iter()
            .zip(&self.columns)
            .map(|(v, c)| v * c.schedule.travel_cost)
            .sum::<f64>();
        if !travel.is_finite() {
            return Ok(None);
        }
        Ok(Some(EvMasterSnapshot {
            phase: self.phase,
            lp_objective: objective,
            artificial_usage,
            route_values,
            travel_objective: (artificial_usage <= tolerance).then_some(travel),
        }))
    }

    pub fn optimize(
        &mut self,
        options: EvColumnGenerationOptions,
    ) -> Result<EvColumnGenerationResult> {
        if !options.lp_tolerance.is_finite()
            || options.lp_tolerance <= 0.0
            || !options.reduced_cost_tolerance.is_finite()
            || options.reduced_cost_tolerance < options.lp_tolerance
        {
            return Err(invalid(
                "EV CG needs positive LP tolerance and reduced-cost tolerance at least as large",
            ));
        }
        let zero_duals = vec![0.0; self.problem.locations.len()];
        let pricing_options = match options.backend {
            EvPricingBackend::Direct(o) => o,
            EvPricingBackend::Subpath(o) => o.pricing,
        };
        self.problem
            .validate_pricing(&zero_duals, 0.0, self.departure, pricing_options)?;
        self.lp
            .set_tolerances(options.lp_tolerance, options.lp_tolerance)?;
        let started = Instant::now();
        let mut result = EvColumnGenerationResult {
            status: EvColumnGenerationStatus::IterationLimit,
            snapshot: None,
            iterations: Vec::new(),
        };
        for _ in 0..options.max_iterations {
            let remaining = options.time_limit.saturating_sub(started.elapsed());
            if remaining.is_zero() {
                result.status = EvColumnGenerationStatus::TimeLimit;
                break;
            }
            let outcome = self.lp.solve_with_time_limit(remaining)?;
            if outcome != LpSolveOutcome::Optimal {
                result.status = if outcome == LpSolveOutcome::TimeLimit {
                    EvColumnGenerationStatus::TimeLimit
                } else {
                    EvColumnGenerationStatus::LpStopped(outcome)
                };
                break;
            }
            let Some(snapshot) = self.snapshot(options.lp_tolerance)? else {
                result.status = EvColumnGenerationStatus::NumericalFailure;
                break;
            };
            let artificial_usage = snapshot.artificial_usage;
            result.snapshot = Some(snapshot);
            if self.phase == EvMasterPhase::Feasibility && artificial_usage <= options.lp_tolerance
            {
                for &c in &self.artificials {
                    self.lp.set_upper(c, 0.0)?;
                    self.lp.set_cost(c, 0.0)?;
                }
                for column in &self.columns {
                    self.lp
                        .set_cost(column.lp_column, column.schedule.travel_cost)?;
                }
                self.phase = EvMasterPhase::TravelCost;
                continue;
            }
            if started.elapsed() >= options.time_limit {
                result.status = EvColumnGenerationStatus::TimeLimit;
                break;
            }
            let duals = self
                .customer_rows
                .iter()
                .map(|row| row.map_or(0.0, |r| self.lp.duals()[r]))
                .collect::<Vec<_>>();
            let fleet_dual = self.fleet_row.map_or(0.0, |r| self.lp.duals()[r]);
            let problem = if self.phase == EvMasterPhase::Feasibility {
                &self.phase_one_problem
            } else {
                &self.problem
            };
            let pricing = match options.backend {
                EvPricingBackend::Direct(o) => {
                    problem.price_direct(&duals, fleet_dual, self.departure, o)?
                }
                EvPricingBackend::Subpath(o) => {
                    problem.price_subpaths(&duals, fleet_dual, self.departure, o)?
                }
            };
            // With complete pricing, a small negative reduced cost can be
            // eliminated by decreasing every customer dual by its magnitude:
            // every admitted route serves at least one customer. Cap artificial
            // duals at one and project a positive fleet dual to zero as well.
            // This prevents a tolerance-sized phase-I residue from being called
            // infeasible solely because the restricted primal is positive.
            let phase_one_dual_bound = if self.phase == EvMasterPhase::Feasibility {
                let correction = pricing
                    .best_route
                    .as_ref()
                    .map_or(0.0, |r| (-r.reduced_cost).max(0.0));
                let mut bound = 0.0;
                for (v, row) in self.customer_rows.iter().enumerate() {
                    if row.is_some() {
                        bound += (duals[v] - correction).min(1.0);
                    }
                }
                if let Some(row) = self.fleet_row {
                    bound += fleet_dual.min(0.0) * self.lp.rhs()[row];
                }
                Some(bound)
            } else {
                None
            };
            result.iterations.push(EvIteration {
                phase: self.phase,
                lp_objective: self.lp.objective(),
                artificial_usage,
                pricing_status: pricing.status,
                minimum_reduced_cost: pricing.best_route.as_ref().map(|r| r.reduced_cost),
                pricing_statistics: pricing.statistics,
            });
            if started.elapsed() >= options.time_limit {
                result.status = EvColumnGenerationStatus::TimeLimit;
                break;
            }
            if let Some(route) = pricing.best_route {
                if route.reduced_cost < -options.reduced_cost_tolerance {
                    let before = self.columns.len();
                    self.add_route(&route.charging.visits)?;
                    if self.columns.len() == before {
                        result.status = EvColumnGenerationStatus::NumericalFailure;
                        break;
                    }
                    continue;
                }
            }
            result.status = if pricing.status != EvPricingStatus::Complete {
                EvColumnGenerationStatus::PricingIncomplete(pricing.status)
            } else if self.phase == EvMasterPhase::Feasibility {
                if phase_one_dual_bound.map_or(false, |bound| {
                    bound.is_finite() && bound > options.lp_tolerance
                }) {
                    EvColumnGenerationStatus::Infeasible
                } else {
                    EvColumnGenerationStatus::NumericalFailure
                }
            } else {
                EvColumnGenerationStatus::Converged
            };
            break;
        }
        Ok(result)
    }
}
