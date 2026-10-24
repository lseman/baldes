//! Robust Solomon instances with the deviations of Munari et al. (2019).
use super::{Node, Problem, Travel};
use crate::inputs::{Instance, Rounding};
use crate::{invalid, Result};

/// Budgets Γq, Γt and relative deviations αq, αt.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Uncertainty {
    pub demand_budget: usize,
    pub time_budget: usize,
    pub demand_alpha: f64,
    pub time_alpha: f64,
}

impl Uncertainty {
    /// Parse `GQ,GT,AQ,AT`, e.g. `5,5,0.1,0.25`.
    pub fn parse(text: &str) -> Result<Self> {
        let parts: Vec<_> = text.split(',').map(str::trim).collect();
        let error = || invalid("robust uncertainty must be GQ,GT,AQ,AT");
        let [gq, gt, aq, at] = parts[..] else {
            return Err(error());
        };
        let value = Self {
            demand_budget: gq.parse().map_err(|_| error())?,
            time_budget: gt.parse().map_err(|_| error())?,
            demand_alpha: aq.parse().map_err(|_| error())?,
            time_alpha: at.parse().map_err(|_| error())?,
        };
        if !(value.demand_alpha.is_finite() && value.demand_alpha >= 0.)
            || !(value.time_alpha.is_finite() && value.time_alpha >= 0.)
        {
            return Err(error());
        }
        Ok(value)
    }
}

impl Problem {
    /// Solomon VRPTW in the solver's tenth units: travel times are distances
    /// truncated to one decimal and scaled by ten, so the paper's
    /// t̂ = 0.1·trunc(10·αt·t) is exactly trunc(αt·T) on the integer T = 10t.
    /// Demand deviations are q̂ = trunc(αq·q). Costs are unscaled distances.
    /// Node ids match `Instance::graph`, with the depot at zero.
    pub fn from_instance(
        instance: &Instance,
        rounding: Rounding,
        uncertainty: Uncertainty,
    ) -> Result<Self> {
        if rounding != Rounding::Tenth || !instance.has_time_windows() {
            return Err(invalid(
                "robust Solomon instances need time windows and tenth rounding",
            ));
        }
        let scale = rounding.scale();
        let n = instance.customers.len();
        let nodes = instance
            .customers
            .iter()
            .enumerate()
            .map(|(i, c)| Node {
                demand: if i == 0 { 0. } else { c.demand },
                demand_deviation: if i == 0 {
                    0.
                } else {
                    (uncertainty.demand_alpha * c.demand).trunc()
                },
                window: (c.ready * scale, c.due * scale),
                service: c.service * scale,
            })
            .collect();
        let travel = (0..n)
            .map(|i| {
                (0..n)
                    .map(|j| {
                        (i != j).then(|| {
                            let time = instance.graph_distance(i, j, rounding);
                            Travel {
                                time,
                                time_deviation: (uncertainty.time_alpha * time).trunc(),
                                cost: time / scale,
                            }
                        })
                    })
                    .collect()
            })
            .collect();
        let problem = Problem {
            nodes,
            travel,
            depot: 0,
            capacity: instance.capacity,
            demand_budget: uncertainty.demand_budget,
            time_budget: uncertainty.time_budget,
        };
        problem.validate()?;
        Ok(problem)
    }
}
