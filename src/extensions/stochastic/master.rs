//! Two-phase set-partitioning column generation over BALDES stochastic pricing.
use super::pricing::route_uses;
use super::{ArcBranch, Policy, PricingOptions, PricingStatus, Problem, RouteEvaluation};
use crate::cuts::capacity::{CapacityCut, CapacitySeparator};
use crate::master::stabilization::{DualPriceSmoothing, Feedback};
use crate::master::{LpMaster, LpSolveOutcome, RowSense};
use crate::{invalid, Result};
use std::collections::BTreeMap;
use std::time::{Duration, Instant};

/// Adaptive pricing retries may grow the per-call label cap, but remain bounded
/// independently of the caller's wall-clock limit.
const MAX_ADAPTIVE_PRICING_LABELS: usize = 16_000_000;
// Periodically spend the NG budget on a complete scalar relaxation, even
// when heuristic sweeps repeatedly find new columns. Only complete pricing
// can turn this attempt into a bound.
const NG_PROOF_SWEEP_INTERVAL: usize = 8;
// Wall-clock budget for true-recourse local search before exact pricing.
const ROUTE_SEARCH_BUDGET: Duration = Duration::from_millis(250);

// Each completed scalar relaxation bounds the same elementary pricing problem.
// An interrupted later stage cannot invalidate an earlier complete bound.
fn merge_ng_results(aggregate: &mut super::NgPricingResult, mut stage: super::NgPricingResult) {
    if let Some(bound) = stage.lower_bound {
        aggregate.lower_bound = Some(aggregate.lower_bound.map_or(bound, |old| old.max(bound)));
    }
    aggregate.statistics.absorb(&stage.statistics);
    aggregate.refinements = aggregate.refinements.saturating_add(stage.refinements);
    aggregate.discarded_walks = aggregate
        .discarded_walks
        .saturating_add(stage.discarded_walks);
    aggregate.routes.append(&mut stage.routes);
    aggregate
        .routes
        .sort_by(|a, b| a.reduced_cost.total_cmp(&b.reduced_cost));
    aggregate.routes.dedup_by(|a, b| a.customers == b.customers);
    aggregate.status = stage.status;
}

#[cfg(test)]
#[path = "../../../tests/unit/extensions/stochastic/ng_schedule.rs"]
mod ng_schedule_tests;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MasterPhase {
    Feasibility,
    ExpectedCost,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ColumnGenerationStatus {
    /// Numerical full-route LP convergence; not an integer optimum or a directed certificate.
    Converged,
    /// The full-master numerical bound reached `ColumnGenerationOptions::cutoff`;
    /// the restricted LP need not be converged.
    BoundCutoff,
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
    /// Maximum nearest-customer NG neighborhood for scalar prepricing; zero disables.
    /// Sizes above four first try a four-customer neighborhood.
    /// Returned columns always receive full stochastic recourse replay.
    pub ng_size: usize,
    pub max_iterations: usize,
    pub time_limit: Duration,
    pub lp_tolerance: f64,
    pub reduced_cost_tolerance: f64,
    /// Objective of a known feasible solution. Once the full-master bound
    /// reaches it, column generation stops with `BoundCutoff`. While the
    /// restricted LP exceeds it, exact pricing only seeks routes whose
    /// reduced cost is low enough to bring the LP below the cutoff.
    pub cutoff: Option<f64>,
    /// Rounds of rounded capacity cuts (mean demands) separated after the
    /// expected-cost LP converges; zero disables them.
    pub capacity_cut_rounds: usize,
    /// Disjoint last-customer pricing parts priced concurrently; zero uses
    /// the ambient rayon thread count and one prices serially.
    pub pricing_parts: usize,
    /// Smooth the duals of NG prepricing in the expected-cost phase with
    /// BALDES dual price smoothing. Exact pricing and every convergence
    /// proof keep the restricted LP duals.
    pub stabilize: bool,
    /// Bytes of reverse-label values (one per load level) that adaptive
    /// exact pricing may hold at once; it bounds the retried label budget.
    pub pricing_label_memory: usize,
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
            cutoff: None,
            capacity_cut_rounds: 20,
            pricing_parts: 0,
            stabilize: true,
            pricing_label_memory: 4 << 30,
        }
    }
}
impl ColumnGenerationOptions {
    pub(super) fn validate(self) -> Result<()> {
        if !self.lp_tolerance.is_finite()
            || self.lp_tolerance <= 0.
            || !self.reduced_cost_tolerance.is_finite()
            || self.reduced_cost_tolerance < self.lp_tolerance
            || self.cutoff.is_some_and(f64::is_nan)
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
    /// Rounded capacity cuts, their >= rows and penalized artificials.
    cuts: Vec<CapacityCut>,
    cut_rows: Vec<usize>,
    cut_artificials: Vec<usize>,
    separator: Option<CapacitySeparator>,
    smoothing: Option<DualPriceSmoothing>,
    /// Negative reduced-cost target for the next staged exact search: a
    /// quarter of the best reduced cost exact pricing found last.
    exact_target: Option<f64>,
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
            cuts: vec![],
            cut_rows: vec![],
            cut_artificials: vec![],
            separator: None,
            smoothing: None,
            exact_target: None,
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
    pub fn capacity_cuts(&self) -> &[CapacityCut] {
        &self.cuts
    }
    /// Phase-two cost of a cut artificial: more than serving every customer
    /// on its own route, so it only remains when the restricted LP needs it.
    fn cut_penalty(&self) -> f64 {
        let n = self.problem.demands.len();
        (1..=n)
            .map(|j| self.problem.costs[0][j] + self.problem.costs[j][0])
            .sum::<f64>()
            .max(1.)
    }

    /// Add a valid `>=` rounded capacity cut over mean demands. Every column
    /// has mean load at most the capacity, so the cut holds for every integer
    /// solution. Returns false for complement-form or duplicate cuts.
    pub fn add_capacity_cut(&mut self, cut: CapacityCut) -> Result<bool> {
        if !cut.greater_equal() || self.cuts.iter().any(|c| c.customers == cut.customers) {
            return Ok(false);
        }
        let entries: Vec<_> = self
            .columns
            .iter()
            .filter_map(|c| {
                let a = cut.coefficient(&c.customers);
                (a != 0).then_some((c.lp_column, a as f64))
            })
            .collect();
        let row = self
            .lp
            .add_row(RowSense::GreaterEqual, cut.rhs as f64, &entries)?;
        let cost = if self.phase == MasterPhase::Feasibility {
            1.
        } else {
            self.cut_penalty()
        };
        self.cut_artificials
            .push(self.lp.add_column(cost, &[(row, 1.)])?);
        self.cut_rows.push(row);
        self.cuts.push(cut);
        Ok(true)
    }

    /// LP rows and coefficients of a route column, sorted by row.
    fn route_entries(&self, route: &[usize]) -> Vec<(usize, f64)> {
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
        for (cut, &row) in self.cuts.iter().zip(&self.cut_rows) {
            let a = cut.coefficient(route);
            if a != 0 {
                entries.push((row, a as f64));
            }
        }
        entries.sort_by_key(|&(r, _)| r);
        entries
    }

    /// Customer, fleet, branch and cut duals of a full row dual vector.
    #[allow(clippy::type_complexity)]
    fn split_duals(&self, point: &[f64]) -> (Vec<f64>, f64, Vec<f64>, Vec<f64>) {
        (
            self.customer_rows.iter().map(|&r| point[r]).collect(),
            point[self.fleet_row],
            self.branch_rows
                .iter()
                .map(|r| r.map_or(0., |r| point[r]))
                .collect(),
            self.cut_rows.iter().map(|&r| point[r]).collect(),
        )
    }

    /// Price NG relaxations of growing neighborhood at the given duals until
    /// one proves no improving route or finds one.
    #[allow(clippy::too_many_arguments)]
    fn ng_prepricing(
        &self,
        duals: &[f64],
        fleet_dual: f64,
        branches: &[ArcBranch],
        branch_duals: &[f64],
        pricing_options: PricingOptions,
        options: &ColumnGenerationOptions,
        complete_first: bool,
        parts: usize,
        deadline: Instant,
        calls: &mut usize,
    ) -> Result<Option<super::NgPricingResult>> {
        let mut relaxed: Option<super::NgPricingResult> = None;
        let n = self.problem.demands.len();
        if options.ng_size == 0 || n <= options.ng_size {
            return Ok(None);
        }
        let size = options.ng_size;
        let initial = size.min(4);
        let neighborhoods = [Some(initial), (initial != size).then_some(size)];
        for neighborhood in neighborhoods.into_iter().flatten() {
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                break;
            }
            let mut ng_options = pricing_options;
            ng_options.max_labels = ng_options.max_labels.min(100_000);
            ng_options.time_limit = remaining.min(Duration::from_secs(2));
            *calls += 1;
            let stage = self.problem.price_ng_partitioned(
                self.policy,
                duals,
                fleet_dual,
                branches,
                branch_duals,
                self.phase == MasterPhase::Feasibility,
                neighborhood,
                ng_options,
                complete_first,
                parts,
            )?;
            if let Some(previous) = &mut relaxed {
                merge_ng_results(previous, stage);
            } else {
                relaxed = Some(stage);
            }
            let combined = relaxed.as_ref().expect("NG result present");
            let proves = combined
                .lower_bound
                .is_some_and(|b| b >= -options.reduced_cost_tolerance);
            let improves = combined.routes.iter().any(|r| {
                r.reduced_cost < -options.reduced_cost_tolerance
                    && !self.orders.contains_key(&r.customers)
            });
            if proves || improves {
                break;
            }
        }
        Ok(relaxed)
    }

    /// Separate violated cuts from the restricted LP solution and add them.
    fn separate_capacity_cuts(&mut self, route_values: &[f64]) -> Result<usize> {
        if self.separator.is_none() {
            let mut demands = vec![0.];
            demands.extend(self.problem.demands.iter().map(|d| d.mean()));
            // Pad the capacity so float mean sums never round k(S) up.
            let mut separator =
                CapacitySeparator::new(demands, self.problem.capacity as f64 * (1. + 1e-9))?;
            separator.parameters.use_two_path = false;
            separator.parameters.complement_form = false;
            separator.parameters.max_cuts_per_round = 20;
            self.separator = Some(separator);
        }
        let support: Vec<_> = self
            .columns
            .iter()
            .zip(route_values)
            .filter(|(_, &x)| x > 1e-6)
            .map(|(c, &x)| (c.customers.clone(), x))
            .collect();
        let cuts = self
            .separator
            .as_mut()
            .expect("separator present")
            .separate(&support)?;
        let mut added = 0;
        for cut in cuts {
            added += usize::from(self.add_capacity_cut(cut)?);
        }
        Ok(added)
    }

    /// Branch and cut duals as per-arc pricing duals. Capacity cuts count
    /// undirected edges, so each crossing edge charges both arc directions.
    fn pricing_arcs(&self, branch_duals: &[f64], cut_duals: &[f64]) -> (Vec<ArcBranch>, Vec<f64>) {
        let mut branches = self.branches.clone();
        let mut duals = branch_duals.to_vec();
        if self.cuts.is_empty() {
            return (branches, duals);
        }
        let n = self.problem.demands.len();
        let mut arc = vec![0.; (n + 1) * (n + 1)];
        for (cut, &dual) in self.cuts.iter().zip(cut_duals) {
            if dual == 0. {
                continue;
            }
            for (&(a, b), &weight) in &cut.edges {
                arc[a * (n + 1) + b] += weight * dual;
                arc[b * (n + 1) + a] += weight * dual;
            }
        }
        for from in 0..=n {
            for to in 0..=n {
                let dual = arc[from * (n + 1) + to];
                if from != to && dual != 0. {
                    branches.push(ArcBranch {
                        from,
                        to,
                        required: true,
                    });
                    duals.push(dual);
                }
            }
        }
        (branches, duals)
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
        let entries = self.route_entries(route);
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
            // Clp checks dual feasibility on its scaled model; accept the
            // matching unscaled violation relative to the column cost.
            let rc_tolerance = tolerance * (1. + self.lp.costs()[c].abs());
            if !rc.is_finite() || (self.lp.upper()[c] > 0. && rc < -rc_tolerance) {
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
                .chain(&self.cut_artificials)
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
        let mut cut_rounds = 0;
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
                let penalty = self.cut_penalty();
                for &c in &self.cut_artificials {
                    self.lp.set_cost(c, penalty)?;
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
            let cut_duals: Vec<_> = self.cut_rows.iter().map(|&r| self.lp.duals()[r]).collect();
            let cut_value: f64 = self
                .cuts
                .iter()
                .zip(&cut_duals)
                .map(|(c, &d)| c.rhs as f64 * d)
                .sum::<f64>();
            let (pricing_branches, pricing_duals) = self.pricing_arcs(&branch_duals, &cut_duals);
            let parts = match options.pricing_parts {
                0 => rayon::current_num_threads(),
                parts => parts,
            };
            let fleet_dual = self.lp.duals()[self.fleet_row];
            let mut pricing_options = options.pricing;
            pricing_options.relaxed_route_costs = false;
            // Column generation needs only improving routes, so exact pricing
            // may prune labels whose completion bound is nonnegative. While
            // the LP exceeds the cutoff, at most K routes can lower it, so a
            // route matters only below (cutoff - LP) / K.
            let fleet = self.vehicles.min(self.problem.demands.len()).max(1) as f64;
            let dual_value = duals.iter().sum::<f64>()
                + branch_duals.iter().sum::<f64>()
                + cut_value
                + fleet * fleet_dual.min(0.);
            let threshold = match options.cutoff {
                Some(cutoff) if self.phase == MasterPhase::ExpectedCost && cutoff.is_finite() => {
                    let margin = 2. * options.lp_tolerance * (1. + cutoff.abs() + dual_value.abs());
                    ((cutoff + margin - dual_value) / fleet).min(0.)
                }
                _ => 0.,
            };
            pricing_options.completion_threshold = Some(threshold);
            pricing_options.time_limit = pricing_options
                .time_limit
                .min(options.time_limit.saturating_sub(started.elapsed()));
            let pricing_deadline = pricing_options.time_limit;
            let pricing_started = Instant::now();
            let complete_first = self.phase == MasterPhase::ExpectedCost
                && result.iterations % NG_PROOF_SWEEP_INTERVAL == 0;
            let deadline =
                Instant::now() + pricing_deadline.saturating_sub(pricing_started.elapsed());
            if options.stabilize
                && self.phase == MasterPhase::ExpectedCost
                && self.smoothing.is_none()
            {
                self.smoothing = Some(DualPriceSmoothing::default());
            }
            let mut relaxed = None;
            if let Some(mut smoothing) = self
                .smoothing
                .take()
                .filter(|_| self.phase == MasterPhase::ExpectedCost)
            {
                // Wentges smoothing: price NG between the stability center and
                // the LP duals. Columns must improve at the LP duals; on a
                // misprice the point moves toward them. Only an NG pass at the
                // LP duals keeps its bound as a convergence proof.
                let raw = self.lp.duals().to_vec();
                let first = smoothing.center.is_empty();
                smoothing.after_master(&raw, self.lp.senses(), 0., 0, first);
                let rhs = self.lp.rhs().to_vec();
                let fleet = self.vehicles.min(self.problem.demands.len()) as f64;
                loop {
                    let at_raw = !smoothing.active;
                    let point = smoothing.pricing.clone();
                    let (d, f, bd, cd) = self.split_duals(&point);
                    let (pb, pd) = self.pricing_arcs(&bd, &cd);
                    let Some(mut stage) = self.ng_prepricing(
                        &d,
                        f,
                        &pb,
                        &pd,
                        pricing_options,
                        &options,
                        complete_first,
                        parts,
                        deadline,
                        &mut result.pricing_calls,
                    )?
                    else {
                        break;
                    };
                    // Lagrangian bound at the pricing point: a valid master bound.
                    let walk =
                        self.problem
                            .route_relaxation_bound(self.policy, &d, f, &pb, &pd, false)?;
                    let minimum = match (walk, stage.lower_bound) {
                        (Some(a), Some(b)) => Some(a.max(b)),
                        (a, b) => a.or(b),
                    };
                    let lagrangian = minimum.map_or(f64::NEG_INFINITY, |m| {
                        let bound = rhs.iter().zip(&point).map(|(b, p)| b * p).sum::<f64>()
                            + fleet * m.min(0.);
                        bound - options.lp_tolerance * (1. + bound.abs())
                    });
                    if lagrangian.is_finite() {
                        result.lower_bound =
                            Some(result.lower_bound.map_or(lagrangian, |b| b.max(lagrangian)));
                    }
                    let mut subgradient = rhs.clone();
                    if let Some(best) = stage.routes.first() {
                        for (row, a) in self.route_entries(&best.customers) {
                            subgradient[row] -= fleet * a;
                        }
                    }
                    if !at_raw {
                        // Keep only routes that improve at the LP duals.
                        stage.lower_bound = None;
                        stage.routes.retain_mut(|r| {
                            r.reduced_cost = r.evaluation.total_cost
                                - self
                                    .route_entries(&r.customers)
                                    .iter()
                                    .map(|&(row, a)| raw[row] * a)
                                    .sum::<f64>();
                            r.reduced_cost < -options.reduced_cost_tolerance
                                && !self.orders.contains_key(&r.customers)
                        });
                    }
                    let fresh = stage
                        .routes
                        .iter()
                        .filter(|r| {
                            r.reduced_cost < -options.reduced_cost_tolerance
                                && !self.orders.contains_key(&r.customers)
                        })
                        .count();
                    let again = smoothing.after_pricing(&Feedback {
                        negative_columns: fresh,
                        subgradient_available: !stage.routes.is_empty() || at_raw,
                        subgradient,
                        lagrangian,
                        phase: 0,
                    });
                    relaxed = Some(stage);
                    if at_raw || !again || fresh > 0 {
                        break;
                    }
                }
                smoothing.after_iteration();
                self.smoothing = Some(smoothing);
            } else {
                relaxed = self.ng_prepricing(
                    &duals,
                    fleet_dual,
                    &pricing_branches,
                    &pricing_duals,
                    pricing_options,
                    &options,
                    complete_first,
                    parts,
                    deadline,
                    &mut result.pricing_calls,
                )?;
            }
            let ng_proves_convergence = relaxed
                .as_ref()
                .and_then(|r| r.lower_bound)
                .is_some_and(|b| b >= -options.reduced_cost_tolerance);
            let ng_improves = relaxed.as_ref().is_some_and(|r| {
                r.routes.iter().any(|r| {
                    r.reduced_cost < -options.reduced_cost_tolerance
                        && !self.orders.contains_key(&r.customers)
                })
            });
            // Before exact pricing, improve basic columns and NG candidates
            // on true reduced costs. Found columns never certify convergence.
            let searched = if ng_proves_convergence || ng_improves {
                vec![]
            } else {
                let mut seeds: Vec<(f64, Vec<usize>)> = result
                    .snapshot
                    .as_ref()
                    .map(|s| {
                        s.route_values
                            .iter()
                            .zip(&self.columns)
                            .filter(|(&x, _)| x > options.lp_tolerance)
                            .map(|(&x, c)| (-x, c.customers.clone()))
                            .collect()
                    })
                    .unwrap_or_default();
                seeds.sort_by(|a, b| a.0.total_cmp(&b.0));
                let mut seeds: Vec<_> = seeds.into_iter().map(|(_, r)| r).collect();
                if let Some(relaxed) = &relaxed {
                    seeds.extend(relaxed.routes.iter().map(|r| r.customers.clone()));
                }
                let prices = super::search::ReducedCosts {
                    policy: self.policy,
                    customer_duals: &duals,
                    fleet_dual,
                    branches: &pricing_branches,
                    branch_duals: &pricing_duals,
                    phase_one: self.phase == MasterPhase::Feasibility,
                };
                let budget = ROUTE_SEARCH_BUDGET
                    .min(pricing_deadline.saturating_sub(pricing_started.elapsed()));
                let mut routes = self.problem.search_routes(
                    &prices,
                    &seeds,
                    options.reduced_cost_tolerance,
                    budget,
                )?;
                routes.retain(|r| !self.orders.contains_key(&r.customers));
                routes.truncate(options.pricing.max_columns.max(1));
                routes
            };
            let mut pricing = if !searched.is_empty() {
                let relaxed = relaxed.take();
                super::PricingResult {
                    status: PricingStatus::Heuristic,
                    routes: searched,
                    minimum_reduced_cost: None,
                    lower_bound: relaxed.as_ref().and_then(|r| r.lower_bound),
                    statistics: relaxed.map(|r| r.statistics).unwrap_or_default(),
                }
            } else if ng_proves_convergence || ng_improves {
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
                let label_bytes = 8 * (self.problem.capacity + 1) + 256;
                let label_cap = MAX_ADAPTIVE_PRICING_LABELS
                    .min(options.pricing_label_memory / label_bytes)
                    .max(options.pricing.max_labels);
                // Staged search (as in Florio et al.'s code): a complete
                // search below a negative target prunes far more labels.
                // Its routes feed the LP without certifying convergence;
                // only a stage at the full threshold can prove it.
                let target = self.exact_target.or_else(|| {
                    relaxed
                        .as_ref()
                        .and_then(|r| r.lower_bound)
                        .filter(|&b| b < 0.)
                        .map(|b| b / 4.)
                });
                let margin = options.reduced_cost_tolerance * (1. + threshold.abs());
                let mut stages: Vec<f64> = target
                    .into_iter()
                    .flat_map(|t| [t, t / 8.])
                    .filter(|&t| t < threshold - margin)
                    .collect();
                stages.push(threshold);
                let mut statistics = crate::labeling::Statistics::default();
                let mut staged_bound: Option<f64> = None;
                let mut exact = None;
                for &stage_threshold in &stages {
                    let mut stage_options = pricing_options;
                    stage_options.completion_threshold = Some(stage_threshold);
                    stage_options.time_limit =
                        pricing_deadline.saturating_sub(pricing_started.elapsed());
                    let mut stage = self.problem.price_adaptive(
                        self.policy,
                        &duals,
                        fleet_dual,
                        &pricing_branches,
                        &pricing_duals,
                        self.phase == MasterPhase::Feasibility,
                        stage_options,
                        parts,
                        label_cap,
                        options.reduced_cost_tolerance,
                    )?;
                    statistics.absorb(&stage.statistics);
                    let best = stage
                        .routes
                        .first()
                        .map(|r| r.reduced_cost)
                        .filter(|&rc| rc < -options.reduced_cost_tolerance);
                    if let Some(rc) = best {
                        self.exact_target = Some(rc / 4.);
                    }
                    if stage_threshold >= threshold {
                        exact = Some(stage);
                        break;
                    }
                    // A complete stage proves every route costs at least
                    // min(found, threshold): a valid reduced-cost bound.
                    if stage.status == PricingStatus::Complete {
                        let bound = stage.minimum_reduced_cost.unwrap_or(stage_threshold);
                        staged_bound = Some(staged_bound.map_or(bound, |b: f64| b.max(bound)));
                    }
                    if best.is_some() || stage.status != PricingStatus::Complete {
                        if stage.status == PricingStatus::Complete {
                            stage.status = PricingStatus::Heuristic;
                        }
                        stage.minimum_reduced_cost = None;
                        exact = Some(stage);
                        break;
                    }
                }
                let mut exact = exact.expect("the full-threshold stage always runs");
                exact.statistics = statistics;
                if let Some(bound) = staged_bound {
                    exact.lower_bound = Some(exact.lower_bound.map_or(bound, |b| b.max(bound)));
                }
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
            if let Some(bound) = self.problem.route_relaxation_bound(
                self.policy,
                &duals,
                fleet_dual,
                &pricing_branches,
                &pricing_duals,
                self.phase == MasterPhase::Feasibility,
            )? {
                pricing.lower_bound = Some(pricing.lower_bound.map_or(bound, |b| b.max(bound)));
            }
            // Shift the fleet dual by a lower bound on every route's reduced
            // cost. This gives a full-master dual bound even with interrupted
            // pricing; the restricted objective alone cannot do that.
            if pricing.status == PricingStatus::Complete {
                if let Some(minimum) = pricing.minimum_reduced_cost {
                    pricing.lower_bound =
                        Some(pricing.lower_bound.map_or(minimum, |b| b.max(minimum)));
                }
            }
            if self.phase == MasterPhase::ExpectedCost {
                if let Some(minimum) = pricing.lower_bound {
                    let bound = duals.iter().sum::<f64>()
                        + branch_duals.iter().sum::<f64>()
                        + cut_value
                        + self.vehicles.min(self.problem.demands.len()) as f64
                            * (fleet_dual.min(0.) + minimum.min(0.));
                    if bound.is_finite() {
                        let bound = bound - options.lp_tolerance * (1. + bound.abs());
                        result.lower_bound =
                            Some(result.lower_bound.map_or(bound, |b| b.max(bound)));
                    }
                }
            }
            if self.phase == MasterPhase::ExpectedCost
                && options
                    .cutoff
                    .zip(result.lower_bound)
                    .is_some_and(|(cutoff, bound)| bound >= cutoff)
            {
                result.status = ColumnGenerationStatus::BoundCutoff;
                break;
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
                // A cutoff threshold that did not certify the cutoff bound
                // leaves the LP unconverged; anything else is numerical.
                result.status = if threshold < 0. {
                    ColumnGenerationStatus::PricingIncomplete(PricingStatus::Complete)
                } else {
                    ColumnGenerationStatus::NumericalFailure
                };
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
            bound += self
                .cuts
                .iter()
                .zip(&cut_duals)
                .map(|(c, &d)| {
                    c.rhs as f64
                        * if self.phase == MasterPhase::Feasibility {
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
                let bound = bound - options.lp_tolerance * (1. + bound.abs());
                result.lower_bound = Some(result.lower_bound.map_or(bound, |b| b.max(bound)));
                if cut_rounds < options.capacity_cut_rounds {
                    let values = result
                        .snapshot
                        .as_ref()
                        .map(|s| s.route_values.clone())
                        .unwrap_or_default();
                    if self.separate_capacity_cuts(&values)? > 0 {
                        cut_rounds += 1;
                        continue;
                    }
                }
                result.status = ColumnGenerationStatus::Converged;
            }
            break;
        }
        Ok(result)
    }
}
