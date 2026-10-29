//! Mean-demand HGS proposals, independently checked against stochastic recourse.
use super::{Policy, Problem, Solution};
use crate::extensions::hgs::driver::{run_hgs_collecting_routes, HgsOptions, HgsResult};
use crate::inputs::{Customer, Instance, Rounding};
use crate::{invalid, Result};
use std::collections::BTreeSet;
use std::time::Duration;

/// Controls the optional HGS proposal search. Its time is part of the solve limit.
#[derive(Clone, Copy, Debug)]
pub struct HgsWarmStart {
    pub time_limit: Duration,
    pub seed: i32,
}

/// HGS candidate counts after independent stochastic feasibility checks.
#[derive(Clone, Copy, Debug)]
pub struct HgsWarmStartSummary {
    pub accepted_columns: usize,
    pub rejected_columns: usize,
    pub accepted_plan: bool,
    pub elapsed_seconds: f64,
}

impl Problem {
    pub(super) fn hgs_seed(
        &self,
        policy: Policy,
        vehicles: usize,
        warm: HgsWarmStart,
        remaining: Duration,
    ) -> Result<(BTreeSet<Vec<usize>>, Option<Solution>, HgsWarmStartSummary)> {
        if warm.time_limit.is_zero() || vehicles > i32::MAX as usize {
            return Err(invalid(
                "HGS warm start requires a positive time limit and representable fleet",
            ));
        }
        // HGS uses i32 loads. Scale fractional means, rounding down to avoid
        // excluding routes near capacity; exact stochastic validation follows.
        // Leave headroom for intermediate load additions in HGS.
        let total: f64 = self.demands.iter().map(|d| d.mean()).sum();
        let max_load = total.max(self.capacity as f64);
        let scale = (1_000_000_000. / max_load).min(1000.).floor();
        if !scale.is_finite() || scale < 1. {
            return Err(invalid(
                "stochastic demands are too large for HGS integer loads",
            ));
        }
        let customer = |demand| Customer {
            x: 0.,
            y: 0.,
            demand,
            ready: 0.,
            due: f64::INFINITY,
            service: 0.,
        };
        let mut customers = vec![customer(0.)];
        customers.extend(
            self.demands
                .iter()
                .map(|d| customer((d.mean() * scale).floor())),
        );
        let instance = Instance {
            name: "stochastic mean demand".into(),
            vehicles,
            capacity: (self.capacity as f64 * scale).floor(),
            customers,
            rounding: Rounding::None,
            distances: Some(self.costs.clone()),
        };
        let options = HgsOptions {
            time_limit: warm.time_limit.min(remaining).as_secs_f64(),
            seed: warm.seed,
            ..Default::default()
        };
        let found = run_hgs_collecting_routes(&instance, &options).map_err(invalid)?;
        Ok(self.validate_hgs_result(policy, vehicles, found))
    }

    fn validate_hgs_result(
        &self,
        policy: Policy,
        vehicles: usize,
        found: HgsResult,
    ) -> (BTreeSet<Vec<usize>>, Option<Solution>, HgsWarmStartSummary) {
        let mut columns = BTreeSet::new();
        let mut rejected = 0;
        for route in found.columns.iter().chain(&found.routes) {
            if route.is_empty() || self.evaluate_route(route, policy).is_err() {
                rejected += 1;
            } else {
                columns.insert(route.clone());
            }
        }
        let mut covered = vec![false; self.demands.len()];
        let mut evaluations = Vec::new();
        let mut valid = found.best_cost.is_some() && found.routes.len() <= vehicles;
        for route in &found.routes {
            match self.evaluate_route(route, policy) {
                Ok(evaluation) if !route.is_empty() => evaluations.push(evaluation),
                _ => {
                    valid = false;
                    continue;
                }
            }
            for &j in route {
                if std::mem::replace(&mut covered[j - 1], true) {
                    valid = false;
                }
            }
        }
        valid &= covered.iter().all(|&v| v);
        let solution = valid.then(|| Solution {
            routes: found.routes,
            total_cost: evaluations.iter().map(|e| e.total_cost).sum(),
            evaluations,
        });
        let summary = HgsWarmStartSummary {
            accepted_columns: columns.len(),
            rejected_columns: rejected,
            accepted_plan: solution.is_some(),
            elapsed_seconds: found.elapsed_secs,
        };
        (columns, solution, summary)
    }
}

#[cfg(test)]
#[path = "../../../tests/unit/extensions/stochastic/warm_start.rs"]
mod tests;
