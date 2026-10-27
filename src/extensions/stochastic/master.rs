//! Two-phase set-partitioning column generation over BALDES stochastic pricing.
use super::pricing::route_uses;
use super::{ArcBranch, Policy, PricingOptions, PricingStatus, Problem, RouteEvaluation};
use crate::master::{LpMaster, LpSolveOutcome, RowSense};
use crate::{invalid, Result};
use std::collections::BTreeMap;
use std::time::{Duration, Instant};

/// Adaptive pricing retries may grow the per-call label cap, but remain bounded
/// independently of the caller's wall-clock limit.
const MAX_ADAPTIVE_PRICING_LABELS: usize = 16_000_000;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MasterPhase {
    Feasibility,
    ExpectedCost,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ColumnGenerationStatus {
    /// Numerical full-route LP convergence; not an integer optimum or a directed certificate.
    Converged,
    Infeasible,
    PricingIncomplete(PricingStatus),
    IterationLimit,
    TimeLimit,
    LpStopped(LpSolveOutcome),
    NumericalFailure,
}
#[derive(Clone, Copy, Debug)]
pub struct ColumnGenerationOptions {
    pub pricing: PricingOptions,
    /// Nearest-customer NG neighborhood for scalar prepricing; zero disables.
    /// Returned columns always receive full stochastic recourse replay.
    pub ng_size: usize,
    pub max_iterations: usize,
    pub time_limit: Duration,
    pub lp_tolerance: f64,
    pub reduced_cost_tolerance: f64,
}
impl Default for ColumnGenerationOptions {
    fn default() -> Self {
        Self {
            pricing: PricingOptions::default(),
            ng_size: 8,
            max_iterations: 1000,
            time_limit: Duration::from_secs(3600),
            lp_tolerance: 1e-7,
            reduced_cost_tolerance: 1e-7,
        }
    }
}
impl ColumnGenerationOptions {
    pub(super) fn validate(self) -> Result<()> {
        if !self.lp_tolerance.is_finite()
            || self.lp_tolerance <= 0.
            || !self.reduced_cost_tolerance.is_finite()
            || self.reduced_cost_tolerance < self.lp_tolerance
        {
            return Err(invalid("stochastic CG requires positive LP tolerance and reduced-cost tolerance >= LP tolerance"));
        }
        Ok(())
    }
}
#[derive(Clone, Debug)]
pub struct Column {
    pub customers: Vec<usize>,
    pub evaluation: RouteEvaluation,
    pub lp_column: usize,
}
#[derive(Clone, Debug)]
pub struct MasterSnapshot {
    pub phase: MasterPhase,
    pub objective: f64,
    pub artificial_usage: f64,
    pub route_values: Vec<f64>,
}
#[derive(Clone, Debug)]
pub struct ColumnGenerationResult {
    pub status: ColumnGenerationStatus,
    pub snapshot: Option<MasterSnapshot>,
    /// Full-master numerical bound from complete pricing or a global walk
    /// relaxation. Adjusted for reduced costs and LP tolerances; not a directed
    /// arithmetic certificate. Never the uncorrected restricted LP objective.
    pub lower_bound: Option<f64>,
    pub iterations: usize,
    pub pricing_calls: usize,
    pub generated_labels: usize,
}

/// Generic over the existing Clp, HiGHS, Gurobi or other LpMaster backend.
/// Required arcs are equality rows; forbidden arcs are removed from pricing.
/// Artificial coverage and required-arc variables provide a genuine phase I.
pub struct Master<M: LpMaster> {
    problem: Problem,
    policy: Policy,
    vehicles: usize,
    branches: Vec<ArcBranch>,
    branch_rows: Vec<Option<usize>>,
    customer_rows: Vec<usize>,
    fleet_row: usize,
    artificials: Vec<usize>,
    columns: Vec<Column>,
    orders: BTreeMap<Vec<usize>, usize>,
    phase: MasterPhase,
    lp: M,
}
impl<M: LpMaster> Master<M> {
    pub fn new(
        problem: Problem,
        policy: Policy,
        vehicles: usize,
        branches: Vec<ArcBranch>,
        mut lp: M,
    ) -> Result<Self> {
        problem.validate_branches(&branches)?;
        if !lp.rhs().is_empty() || !lp.costs().is_empty() || lp.objective_offset() != 0. {
            return Err(invalid("stochastic master requires an empty LP backend"));
        }
        // Validate the engine's supported graph domain before solving an LP.
        problem.pricing_graph(
            policy,
            &vec![0.; problem.demands.len()],
            0.,
            &branches,
            &vec![0.; branches.len()],
            true,
        )?;
        let mut customer_rows = vec![];
        let mut artificials = vec![];
        for _ in &problem.demands {
            let row = lp.add_row(RowSense::Equal, 1., &[])?;
            customer_rows.push(row);
            artificials.push(lp.add_column(1., &[(row, 1.)])?);
        }
        let fleet_row = lp.add_row(
            RowSense::LessEqual,
            vehicles.min(problem.demands.len()) as f64,
            &[],
        )?;
        let mut branch_rows = vec![];
        for branch in &branches {
            let row = if branch.required {
                let row = lp.add_row(RowSense::Equal, 1., &[])?;
                artificials.push(lp.add_column(1., &[(row, 1.)])?);
                Some(row)
            } else {
                None
            };
            branch_rows.push(row);
        }
        Ok(Self {
            problem,
            policy,
            vehicles,
            branches,
            branch_rows,
            customer_rows,
            fleet_row,
            artificials,
            columns: vec![],
            orders: BTreeMap::new(),
            phase: MasterPhase::Feasibility,
            lp,
        })
    }
    pub fn problem(&self) -> &Problem {
        &self.problem
    }
    pub fn columns(&self) -> &[Column] {
        &self.columns
    }
    pub fn lp(&self) -> &M {
        &self.lp
    }
    pub fn phase(&self) -> MasterPhase {
        self.phase
    }

    /// Independently evaluate every seed or generated route at original costs.
    pub fn add_route(&mut self, route: &[usize]) -> Result<usize> {
        if route.is_empty()
            || self
                .branches
                .iter()
                .any(|b| !b.required && route_uses(route, *b))
        {
            return Err(invalid("empty or forbidden stochastic master route"));
        }
        let evaluation = self.problem.evaluate_route(route, self.policy)?;
        if let Some(&column) = self.orders.get(route) {
            return Ok(column);
        }
        let mut entries: Vec<_> = route
            .iter()
            .map(|&j| (self.customer_rows[j - 1], 1.))
            .collect();
        entries.push((self.fleet_row, 1.));
        for (b, row) in self.branches.iter().zip(&self.branch_rows) {
            if route_uses(route, *b) {
                if let Some(row) = row {
                    entries.push((*row, 1.));
                }
            }
        }
        entries.sort_by_key(|&(r, _)| r);
        let cost = if self.phase == MasterPhase::Feasibility {
            0.
        } else {
            evaluation.total_cost
        };
        let lp_column = self.lp.add_column(cost, &entries)?;
        self.orders.insert(route.to_vec(), lp_column);
        self.columns.push(Column {
            customers: route.to_vec(),
            evaluation,
            lp_column,
        });
        Ok(lp_column)
    }

    fn snapshot(&self, tolerance: f64) -> Result<Option<MasterSnapshot>> {
        if !self.lp.solved()
            || self.lp.values().len() != self.lp.costs().len()
            || self.lp.upper().len() != self.lp.costs().len()
            || self.lp.duals().len() != self.lp.rhs().len()
        {
            return Ok(None);
        }
        let mut activity = vec![0.; self.lp.rhs().len()];
        let mut objective = 0.;
        for (c, &x) in self.lp.values().iter().enumerate() {
            if !x.is_finite() || x < -tolerance || x > self.lp.upper()[c] + tolerance {
                return Ok(None);
            }
            objective += x * self.lp.costs()[c];
            let mut rc = self.lp.costs()[c];
            for (row, a) in self.lp.column_entries(c)? {
                activity[row] += x * a;
                rc -= self.lp.duals()[row] * a;
            }
            if !rc.is_finite() || (self.lp.upper()[c] > 0. && rc < -tolerance) {
                return Ok(None);
            }
        }
        for (r, &a) in activity.iter().enumerate() {
            let (rhs, dual) = (self.lp.rhs()[r], self.lp.duals()[r]);
            if !a.is_finite() || !dual.is_finite() {
                return Ok(None);
            }
            let valid = match self.lp.senses()[r] {
                RowSense::Equal => (a - rhs).abs() <= tolerance,
                RowSense::LessEqual => a <= rhs + tolerance && dual <= tolerance,
                RowSense::GreaterEqual => a >= rhs - tolerance && dual >= -tolerance,
            };
            if !valid {
                return Ok(None);
            }
        }
        let dual_objective: f64 = self
            .lp
            .rhs()
            .iter()
            .zip(self.lp.duals())
            .map(|(b, d)| b * d)
            .sum();
        if !objective.is_finite()
            || !self.lp.objective().is_finite()
            || !dual_objective.is_finite()
            || (objective - self.lp.objective()).abs() > tolerance * (1. + objective.abs())
            || (objective - dual_objective).abs() > tolerance * (1. + objective.abs())
        {
            return Ok(None);
        }
        Ok(Some(MasterSnapshot {
            phase: self.phase,
            objective,
            artificial_usage: self
                .artificials
                .iter()
                .map(|&c| self.lp.values()[c].max(0.))
                .sum(),
            route_values: self
                .columns
                .iter()
                .map(|c| self.lp.values()[c.lp_column])
                .collect(),
        }))
    }

    pub fn optimize(&mut self, options: ColumnGenerationOptions) -> Result<ColumnGenerationResult> {
        self.optimize_with_observer(options, |_, _| Ok(()))
    }

    pub(super) fn optimize_with_observer(
        &mut self,
        options: ColumnGenerationOptions,
        mut observer: impl FnMut(&[Column], &MasterSnapshot) -> Result<()>,
    ) -> Result<ColumnGenerationResult> {
        options.validate()?;
        if self.problem.demands.is_empty() {
            self.phase = MasterPhase::ExpectedCost;
            return Ok(ColumnGenerationResult {
                status: ColumnGenerationStatus::Converged,
                snapshot: Some(MasterSnapshot {
                    phase: self.phase,
                    objective: 0.,
                    artificial_usage: 0.,
                    route_values: vec![],
                }),
                lower_bound: Some(0.),
                iterations: 0,
                pricing_calls: 0,
                generated_labels: 0,
            });
        }
        self.lp
            .set_tolerances(options.lp_tolerance, options.lp_tolerance)?;
        let started = Instant::now();
        let mut result = ColumnGenerationResult {
            status: ColumnGenerationStatus::IterationLimit,
            snapshot: None,
            lower_bound: None,
            iterations: 0,
            pricing_calls: 0,
            generated_labels: 0,
        };
        for _ in 0..options.max_iterations {
            let remaining = options.time_limit.saturating_sub(started.elapsed());
            if remaining.is_zero() {
                result.status = ColumnGenerationStatus::TimeLimit;
                break;
            }
            let status = self.lp.solve_with_time_limit(remaining)?;
            result.iterations += 1;
            if status != LpSolveOutcome::Optimal {
                result.status = if status == LpSolveOutcome::TimeLimit {
                    ColumnGenerationStatus::TimeLimit
                } else {
                    ColumnGenerationStatus::LpStopped(status)
                };
                break;
            }
            let Some(snapshot) = self.snapshot(options.lp_tolerance)? else {
                result.status = ColumnGenerationStatus::NumericalFailure;
                break;
            };
            observer(&self.columns, &snapshot)?;
            let artificial = snapshot.artificial_usage;
            result.snapshot = Some(snapshot);
            if self.phase == MasterPhase::Feasibility && artificial <= options.lp_tolerance {
                for &c in &self.artificials {
                    self.lp.set_upper(c, 0.)?;
                    self.lp.set_cost(c, 0.)?;
                }
                for c in &self.columns {
                    self.lp.set_cost(c.lp_column, c.evaluation.total_cost)?;
                }
                self.phase = MasterPhase::ExpectedCost;
                continue;
            }
            let duals: Vec<_> = self
                .customer_rows
                .iter()
                .map(|&r| self.lp.duals()[r])
                .collect();
            let branch_duals: Vec<_> = self
                .branch_rows
                .iter()
                .map(|r| r.map_or(0., |r| self.lp.duals()[r]))
                .collect();
            let fleet_dual = self.lp.duals()[self.fleet_row];
            let mut pricing_options = options.pricing;
            pricing_options.time_limit = pricing_options
                .time_limit
                .min(options.time_limit.saturating_sub(started.elapsed()));
            let pricing_deadline = pricing_options.time_limit;
            let pricing_started = Instant::now();
            let relaxed = if options.ng_size > 0 && self.problem.demands.len() > options.ng_size {
                let mut ng_options = pricing_options;
                ng_options.max_labels = ng_options.max_labels.min(100_000);
                ng_options.time_limit = ng_options.time_limit.min(Duration::from_secs(2));
                result.pricing_calls += 1;
                Some(self.problem.price_ng(
                    self.policy,
                    &duals,
                    fleet_dual,
                    &self.branches,
                    &branch_duals,
                    self.phase == MasterPhase::Feasibility,
                    options.ng_size,
                    ng_options,
                )?)
            } else {
                None
            };
            let ng_proves_convergence = relaxed
                .as_ref()
                .and_then(|r| r.lower_bound)
                .is_some_and(|b| b >= -options.reduced_cost_tolerance);
            let ng_improves = relaxed.as_ref().is_some_and(|r| {
                r.routes
                    .iter()
                    .any(|r| r.reduced_cost < -options.reduced_cost_tolerance)
            });
            let pricing = if ng_proves_convergence || ng_improves {
                let relaxed = relaxed.expect("NG result present");
                super::PricingResult {
                    status: if ng_proves_convergence {
                        PricingStatus::Complete
                    } else {
                        PricingStatus::Heuristic
                    },
                    routes: relaxed.routes,
                    // A global lower bound suffices to prove absence of
                    // negative elementary columns; it need not be attained.
                    minimum_reduced_cost: if ng_proves_convergence {
                        relaxed.lower_bound
                    } else {
                        None
                    },
                    lower_bound: relaxed.lower_bound,
                    statistics: relaxed.statistics,
                }
            } else {
                pricing_options.time_limit = pricing_options
                    .time_limit
                    .saturating_sub(pricing_started.elapsed());
                result.pricing_calls += 1;
                let mut attempts = crate::labeling::Statistics::default();
                let mut candidates = Vec::new();
                let mut walk_bound: Option<f64> = None;
                let mut exact = loop {
                    pricing_options.time_limit =
                        pricing_deadline.saturating_sub(pricing_started.elapsed());
                    if pricing_options.time_limit.is_zero() {
                        break super::PricingResult {
                            status: PricingStatus::TimeLimit,
                            routes: candidates,
                            minimum_reduced_cost: None,
                            lower_bound: walk_bound,
                            statistics: attempts,
                        };
                    }
                    let mut priced = self.problem.price(
                        self.policy,
                        &duals,
                        fleet_dual,
                        &self.branches,
                        &branch_duals,
                        self.phase == MasterPhase::Feasibility,
                        pricing_options,
                    )?;
                    attempts.absorb(&priced.statistics);
                    if let Some(bound) = priced.lower_bound {
                        walk_bound = Some(walk_bound.map_or(bound, |current| current.max(bound)));
                    }
                    candidates.append(&mut priced.routes);
                    candidates.sort_by(|a, b| a.reduced_cost.total_cmp(&b.reduced_cost));
                    candidates.dedup_by(|a, b| a.customers == b.customers);
                    let has_improving_route = candidates
                        .iter()
                        .any(|route| route.reduced_cost < -options.reduced_cost_tolerance);
                    let can_retry = priced.status == PricingStatus::WorkLimit
                        && !has_improving_route
                        && pricing_options.max_labels < MAX_ADAPTIVE_PRICING_LABELS
                        && pricing_started.elapsed() < pricing_deadline;
                    if can_retry {
                        pricing_options.max_labels = pricing_options
                            .max_labels
                            .saturating_mul(2)
                            .min(MAX_ADAPTIVE_PRICING_LABELS);
                        continue;
                    }
                    priced.routes = candidates;
                    priced.statistics = attempts;
                    priced.lower_bound = walk_bound;
                    break priced;
                };
                if let Some(relaxed) = relaxed {
                    if let Some(bound) = relaxed.lower_bound {
                        exact.lower_bound = Some(exact.lower_bound.map_or(bound, |b| b.max(bound)));
                    }
                    exact.statistics.absorb(&relaxed.statistics);
                }
                exact
            };
            result.generated_labels = result
                .generated_labels
                .saturating_add(pricing.statistics.generated);
            // Shift the fleet dual by a lower bound on every route's reduced
            // cost. This gives a full-master dual bound even with interrupted
            // pricing; the restricted objective alone cannot do that.
            if self.phase == MasterPhase::ExpectedCost {
                if let Some(minimum) = pricing.lower_bound {
                    let bound = duals.iter().sum::<f64>()
                        + branch_duals.iter().sum::<f64>()
                        + self.vehicles.min(self.problem.demands.len()) as f64
                            * (fleet_dual.min(0.) + minimum.min(0.));
                    if bound.is_finite() {
                        let bound = bound - options.lp_tolerance * (1. + bound.abs());
                        result.lower_bound =
                            Some(result.lower_bound.map_or(bound, |b| b.max(bound)));
                    }
                }
            }
            let mut added = 0;
            for route in &pricing.routes {
                if route.reduced_cost < -options.reduced_cost_tolerance {
                    let before = self.columns.len();
                    self.add_route(&route.customers)?;
                    added += self.columns.len() - before;
                }
            }
            if started.elapsed() >= options.time_limit {
                result.status = ColumnGenerationStatus::TimeLimit;
                break;
            }
            if pricing.status != PricingStatus::Complete {
                // Keep improving the restricted pool if the incomplete sweep
                // found useful columns. Only complete pricing can converge.
                if added > 0 {
                    continue;
                }
                result.status = ColumnGenerationStatus::PricingIncomplete(pricing.status);
                break;
            }
            if added > 0 {
                continue;
            }
            let minimum = pricing.minimum_reduced_cost.unwrap_or(0.);
            if minimum < -options.reduced_cost_tolerance {
                result.status = ColumnGenerationStatus::NumericalFailure;
                break;
            }
            // Decrease each customer dual by the negative pricing residual.
            // Every route is nonempty. Project the <= fleet dual to <= 0.
            let correction = (-minimum).max(0.);
            let mut bound: f64 = duals
                .iter()
                .map(|d| {
                    let d = d - correction;
                    if self.phase == MasterPhase::Feasibility {
                        d.min(1.)
                    } else {
                        d
                    }
                })
                .sum();
            bound += fleet_dual.min(0.) * self.vehicles.min(self.problem.demands.len()) as f64;
            bound += branch_duals
                .iter()
                .map(|&d| {
                    if self.phase == MasterPhase::Feasibility {
                        d.min(1.)
                    } else {
                        d
                    }
                })
                .sum::<f64>();
            if !bound.is_finite() {
                result.status = ColumnGenerationStatus::NumericalFailure;
                break;
            }
            if self.phase == MasterPhase::Feasibility {
                result.status = if bound > options.lp_tolerance {
                    ColumnGenerationStatus::Infeasible
                } else {
                    ColumnGenerationStatus::NumericalFailure
                };
            } else {
                // Leave a tolerance margin for the checked restricted LP.
                result.lower_bound = Some(bound - options.lp_tolerance * (1. + bound.abs()));
                result.status = ColumnGenerationStatus::Converged;
            }
            break;
        }
        Ok(result)
    }
}
