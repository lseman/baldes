//! Two-phase column generation; LP stays on the calling thread.
use super::{step_entries, Formulation, PricingMethod, PricingOptions, Problem, Step};
use crate::master::{LpMaster, LpSolveOutcome};
use crate::{incomplete, invalid, Result};
use std::collections::BTreeSet;
use std::time::{Duration, Instant};

#[derive(Clone, Debug)]
pub struct ColumnGenerationOptions {
    pub pricing: PricingOptions,
    pub max_iterations: usize,
    /// Overall LP + pricing wall budget. A pricing limit returns an error.
    pub time_limit: Duration,
    pub tolerance: f64,
}

impl Default for ColumnGenerationOptions {
    fn default() -> Self {
        Self {
            pricing: PricingOptions::default(),
            max_iterations: 1000,
            time_limit: Duration::from_secs(300),
            tolerance: 1e-6,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ColumnGenerationStatus {
    Converged,
    Infeasible,
    IterationLimit,
    TimeLimit,
    LpStopped(LpSolveOutcome),
}

#[derive(Clone, Debug)]
pub struct ColumnGenerationResult {
    pub status: ColumnGenerationStatus,
    /// Full p-step LP objective only after successful travel-cost convergence.
    /// Floating-point solver tolerances apply; this is not an integer optimum.
    pub lp_bound: Option<f64>,
    /// Latest travel-cost restricted-master objective, potentially above the
    /// full LP optimum. Never use this alone as a lower bound.
    pub restricted_objective: Option<f64>,
    pub iterations: usize,
    pub columns: usize,
    pub pricing_jobs: usize,
    pub pricing_workers: usize,
    pub generated_labels: usize,
    pub enumerated_extensions: usize,
    pub pricing_method: Option<PricingMethod>,
    pub lp_seconds: f64,
    pub pricing_seconds: f64,
    pub elapsed_seconds: f64,
}

pub(super) trait Model {
    type Column;
    fn validate(&self) -> Result<()>;
    fn customers(&self) -> usize;
    fn row_count(&self) -> usize;
    fn empty_master<M: LpMaster>(&self, lp: M) -> Result<M>;
    fn price(
        &self,
        duals: &[f64],
        weight: f64,
        options: &PricingOptions,
    ) -> Result<super::PricingRound<Self::Column>>;
    fn entries(&self, column: &Self::Column) -> Vec<(usize, f64)>;
    fn cost(&self, column: &Self::Column) -> f64;
    fn identity(&self, column: &Self::Column) -> (Vec<usize>, Vec<u64>);
}
impl Model for Problem {
    type Column = Step;
    fn validate(&self) -> Result<()> {
        Problem::validate(self)
    }
    fn customers(&self) -> usize {
        self.demands.len()
    }
    fn row_count(&self) -> usize {
        3 * self.demands.len()
    }
    fn empty_master<M: LpMaster>(&self, lp: M) -> Result<M> {
        Formulation {
            customers: self.demands.len(),
            demands: self.demands.clone(),
            steps: vec![],
        }
        .into_master(lp)
    }
    fn price(
        &self,
        duals: &[f64],
        weight: f64,
        options: &PricingOptions,
    ) -> Result<super::PricingRound> {
        self.price_steps(duals, weight, options)
    }
    fn entries(&self, step: &Step) -> Vec<(usize, f64)> {
        step_entries(&self.demands, step)
    }
    fn cost(&self, step: &Step) -> f64 {
        step.cost
    }
    fn identity(&self, step: &Step) -> (Vec<usize>, Vec<u64>) {
        (step.path.clone(), vec![step.prior_demand.to_bits()])
    }
}
impl Problem {
    /// Solve the unlimited-fleet CVRP p-step LP using an empty LP backend.
    pub fn solve_column_generation<M: LpMaster>(
        &self,
        lp: M,
        options: ColumnGenerationOptions,
    ) -> Result<ColumnGenerationResult> {
        solve(self, lp, options)
    }
}
pub(super) fn solve<P: Model, M: LpMaster>(
    model: &P,
    lp: M,
    options: ColumnGenerationOptions,
) -> Result<ColumnGenerationResult> {
    model.validate()?;
    if !options.tolerance.is_finite()
        || options.tolerance < 1e-7
        || options.pricing.p == 0
        || options.pricing.threads == 0
        || options.pricing.max_labels_per_job == 0
        || options.pricing.max_subproblems_per_job == 0
    {
        return Err(invalid("invalid p-step column generation options"));
    }
    let n = model.customers();
    let mut lp = model.empty_master(lp)?;
    lp.set_tolerances(options.tolerance * 0.1, options.tolerance * 0.1)?;
    // Nonnegative degree artificials provide a feasible phase-I start.
    // Endpoint rows are zero, load rows are >=0, hence need no artificials.
    let artificials: Vec<_> = (0..n)
        .map(|r| lp.add_column(1.0, &[(r, 1.0)]))
        .collect::<Result<_>>()?;
    let mut columns: Vec<(usize, P::Column)> = Vec::new();
    let mut identities = BTreeSet::new();
    let mut phase_one = true;
    let started = Instant::now();
    let mut result = ColumnGenerationResult {
        status: ColumnGenerationStatus::IterationLimit,
        lp_bound: None,
        restricted_objective: None,
        iterations: 0,
        columns: 0,
        pricing_jobs: 0,
        pricing_workers: 0,
        generated_labels: 0,
        enumerated_extensions: 0,
        pricing_method: None,
        lp_seconds: 0.0,
        pricing_seconds: 0.0,
        elapsed_seconds: 0.0,
    };
    for _ in 0..options.max_iterations {
        let remaining = options.time_limit.saturating_sub(started.elapsed());
        if remaining.is_zero() {
            result.status = ColumnGenerationStatus::TimeLimit;
            break;
        }
        let tick = Instant::now();
        let outcome = lp.solve_with_time_limit(remaining)?;
        result.lp_seconds += tick.elapsed().as_secs_f64();
        result.iterations += 1;
        if outcome != LpSolveOutcome::Optimal {
            result.status = ColumnGenerationStatus::LpStopped(outcome);
            break;
        }
        if lp.duals().len() != model.row_count()
            || lp.values().len() != lp.costs().len()
            || !lp.objective().is_finite()
            || lp.duals().iter().chain(lp.values()).any(|v| !v.is_finite())
        {
            return Err(incomplete("invalid p-step LP solution"));
        }
        let artificial_usage: f64 = artificials.iter().map(|&c| lp.values()[c].max(0.0)).sum();
        if phase_one && artificial_usage <= options.tolerance {
            for &column in &artificials {
                lp.set_upper(column, 0.0)?;
                lp.set_cost(column, 0.0)?;
            }
            for (column, step) in &columns {
                lp.set_cost(*column, model.cost(step))?;
            }
            phase_one = false;
            continue;
        }
        if !phase_one {
            result.restricted_objective = Some(lp.objective());
        }
        let mut pricing = options.pricing.clone();
        pricing.time_limit = pricing
            .time_limit
            .min(options.time_limit.saturating_sub(started.elapsed()));
        let tick = Instant::now();
        let duals = lp.duals().to_vec();
        let round = model.price(&duals, if phase_one { 0.0 } else { 1.0 }, &pricing)?;
        result.pricing_seconds += tick.elapsed().as_secs_f64();
        result.pricing_jobs += round.jobs;
        result.pricing_workers = result.pricing_workers.max(round.workers);
        result.generated_labels += round.generated_labels;
        result.enumerated_extensions += round.enumerated_extensions;
        result.pricing_method = Some(round.method);
        if started.elapsed() >= options.time_limit {
            result.status = ColumnGenerationStatus::TimeLimit;
            break;
        }
        let mut pending = Vec::new();
        for step in round.steps {
            let entries = model.entries(&step);
            let cost = if phase_one { 0.0 } else { model.cost(&step) };
            let reduced = cost - entries.iter().map(|&(r, a)| a * duals[r]).sum::<f64>();
            if reduced >= -options.tolerance {
                continue;
            }
            // Zero and maximum coincide when path demand equals capacity.
            let identity = model.identity(&step);
            if !identities.insert(identity) {
                continue;
            }
            pending.push((cost, entries, step));
        }
        if pending.is_empty() {
            if round
                .minimum_reduced_cost
                .is_some_and(|rc| rc < -options.tolerance)
            {
                return Err(incomplete(
                    "existing p-step column has negative reduced cost",
                ));
            }
            if phase_one {
                // Every path has degree >=1. Lower all degree duals by
                // -min(rc) to obtain a feasible full-pool phase-I dual.
                let correction = round.minimum_reduced_cost.map_or(0.0, |r| (-r).max(0.0));
                let dual_bound = lp
                    .rhs()
                    .iter()
                    .zip(lp.duals())
                    .map(|(b, d)| b * d)
                    .sum::<f64>()
                    - 2.0 * n as f64 * correction;
                if dual_bound <= options.tolerance {
                    return Err(incomplete("p-step phase I could not certify infeasibility"));
                }
                result.status = ColumnGenerationStatus::Infeasible;
            } else {
                result.status = ColumnGenerationStatus::Converged;
                result.lp_bound = Some(lp.objective());
            }
            break;
        }
        let input: Vec<_> = pending.iter().map(|(c, a, _)| (*c, a.as_slice())).collect();
        let added = lp.add_columns(&input)?;
        for (column, (_, _, step)) in added.zip(pending) {
            // Loose bound avoids bound duals absent from pricing.
            if lp.upper()[column] != 1e12 {
                lp.set_upper(column, 1e12)?;
            }
            columns.push((column, step));
        }
    }
    result.columns = columns.len();
    result.elapsed_seconds = started.elapsed().as_secs_f64();
    Ok(result)
}
