//! Validation-only robust set-partitioning master on the LP backend interface.
//! Each customer row has an artificial column with cost `penalty`; positive
//! artificial usage at convergence means the penalty was too small or the
//! instance has no robust feasible cover. No integer or branching claims.

use super::{PricingOptions, PricingStatistics, PricingStatus, Problem};
use crate::master::{LpMaster, LpSolveOutcome, RowSense};
use crate::{invalid, Result};
use std::collections::BTreeSet;
use std::time::{Duration, Instant};

#[derive(Clone, Copy, Debug)]
pub struct RobustColumnGenerationOptions {
    pub pricing: PricingOptions,
    pub max_iterations: usize,
    pub time_limit: Duration,
    pub reduced_cost_tolerance: f64,
    pub lp_tolerance: f64,
}

impl Default for RobustColumnGenerationOptions {
    fn default() -> Self {
        Self {
            pricing: PricingOptions::default(),
            max_iterations: 10_000,
            time_limit: Duration::from_secs(3600),
            reduced_cost_tolerance: 1e-7,
            lp_tolerance: 1e-9,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RobustColumnGenerationStatus {
    /// Complete pricing found no negative reduced cost at the tolerances.
    Converged,
    PricingIncomplete(PricingStatus),
    IterationLimit,
    TimeLimit,
    LpStopped(LpSolveOutcome),
    NumericalFailure,
}

#[derive(Clone, Debug)]
pub struct RobustColumn {
    pub route: Vec<usize>,
    pub cost: f64,
    pub lp_column: usize,
}

#[derive(Clone, Debug)]
pub struct RobustIteration {
    pub lp_objective: f64,
    pub minimum_reduced_cost: Option<f64>,
    pub columns_added: usize,
    pub pricing_status: PricingStatus,
    pub pricing_statistics: PricingStatistics,
}

#[derive(Clone, Debug)]
pub struct RobustColumnGenerationResult {
    pub status: RobustColumnGenerationStatus,
    pub lp_objective: Option<f64>,
    /// Total artificial value in the last LP solution.
    pub artificial_usage: f64,
    /// Aligned to RobustMaster::columns().
    pub route_values: Vec<f64>,
    pub iterations: Vec<RobustIteration>,
}

pub struct RobustMaster<M: LpMaster> {
    problem: Problem,
    lp: M,
    customer_rows: Vec<Option<usize>>,
    artificials: Vec<usize>,
    columns: Vec<RobustColumn>,
    routes: BTreeSet<Vec<usize>>,
}

impl<M: LpMaster> RobustMaster<M> {
    pub fn new(problem: Problem, mut lp: M, penalty: f64) -> Result<Self> {
        problem.validate()?;
        if !penalty.is_finite() || penalty <= 0.0 || !lp.rhs().is_empty() || !lp.costs().is_empty()
        {
            return Err(invalid(
                "robust master requires an empty LP and a positive penalty",
            ));
        }
        let mut customer_rows = vec![None; problem.nodes.len()];
        let mut artificials = Vec::new();
        for v in (0..problem.nodes.len()).filter(|&v| v != problem.depot) {
            let row = lp.add_row(RowSense::Equal, 1.0, &[])?;
            customer_rows[v] = Some(row);
            artificials.push(lp.add_column(penalty, &[(row, 1.0)])?);
        }
        Ok(Self {
            problem,
            lp,
            customer_rows,
            artificials,
            columns: Vec::new(),
            routes: BTreeSet::new(),
        })
    }

    pub fn problem(&self) -> &Problem {
        &self.problem
    }
    pub fn columns(&self) -> &[RobustColumn] {
        &self.columns
    }
    pub fn lp(&self) -> &M {
        &self.lp
    }

    /// Replay and add a robust route. Returns false for a duplicate.
    pub fn add_route(&mut self, route: &[usize]) -> Result<bool> {
        if self.routes.contains(route) {
            return Ok(false);
        }
        let schedule = self.problem.replay(route)?;
        let mut entries = route
            .iter()
            .filter_map(|&v| self.customer_rows[v].map(|r| (r, 1.0)))
            .collect::<Vec<_>>();
        if entries.is_empty() {
            return Err(invalid("robust column must serve at least one customer"));
        }
        entries.sort_by_key(|e| e.0);
        let lp_column = self.lp.add_column(schedule.cost, &entries)?;
        self.routes.insert(route.to_vec());
        self.columns.push(RobustColumn {
            route: route.to_vec(),
            cost: schedule.cost,
            lp_column,
        });
        Ok(true)
    }

    pub fn optimize(
        &mut self,
        options: RobustColumnGenerationOptions,
    ) -> Result<RobustColumnGenerationResult> {
        if !options.lp_tolerance.is_finite()
            || options.lp_tolerance <= 0.0
            || !options.reduced_cost_tolerance.is_finite()
            || options.reduced_cost_tolerance < options.lp_tolerance
        {
            return Err(invalid(
                "robust CG needs positive LP tolerance and reduced-cost tolerance at least as large",
            ));
        }
        self.lp
            .set_tolerances(options.lp_tolerance, options.lp_tolerance)?;
        let started = Instant::now();
        let mut result = RobustColumnGenerationResult {
            status: RobustColumnGenerationStatus::IterationLimit,
            lp_objective: None,
            artificial_usage: 0.0,
            route_values: Vec::new(),
            iterations: Vec::new(),
        };
        for _ in 0..options.max_iterations {
            let remaining = options.time_limit.saturating_sub(started.elapsed());
            if remaining.is_zero() {
                result.status = RobustColumnGenerationStatus::TimeLimit;
                break;
            }
            let outcome = self.lp.solve_with_time_limit(remaining)?;
            if outcome != LpSolveOutcome::Optimal {
                result.status = match outcome {
                    LpSolveOutcome::TimeLimit => RobustColumnGenerationStatus::TimeLimit,
                    other => RobustColumnGenerationStatus::LpStopped(other),
                };
                break;
            }
            result.lp_objective = Some(self.lp.objective());
            result.artificial_usage = self
                .artificials
                .iter()
                .map(|&c| self.lp.values()[c].max(0.0))
                .sum();
            result.route_values = self
                .columns
                .iter()
                .map(|c| self.lp.values()[c.lp_column])
                .collect();
            let duals = self
                .customer_rows
                .iter()
                .map(|row| row.map_or(0.0, |r| self.lp.duals()[r]))
                .collect::<Vec<_>>();
            let pricing = super::price(&self.problem, &duals, 0.0, options.pricing)?;
            let mut added = 0;
            for route in &pricing.routes {
                if route.reduced_cost < -options.reduced_cost_tolerance
                    && self.add_route(&route.route)?
                {
                    added += 1;
                }
            }
            result.iterations.push(RobustIteration {
                lp_objective: self.lp.objective(),
                minimum_reduced_cost: pricing.routes.first().map(|r| r.reduced_cost),
                columns_added: added,
                pricing_status: pricing.status,
                pricing_statistics: pricing.statistics,
            });
            if added > 0 {
                continue;
            }
            // A negative route already in the LP means duals and pricing disagree.
            let negative = pricing
                .routes
                .first()
                .is_some_and(|r| r.reduced_cost < -options.reduced_cost_tolerance);
            result.status = match pricing.status {
                _ if negative => RobustColumnGenerationStatus::NumericalFailure,
                PricingStatus::Complete => RobustColumnGenerationStatus::Converged,
                s => RobustColumnGenerationStatus::PricingIncomplete(s),
            };
            break;
        }
        Ok(result)
    }
}
