//! Compact p-step CVRP LP of Dollevoet, Munari and Spliet (EI-2020-01),
//! Proposition 1: degree, endpoint flow and cumulative-load inequalities.
//!
//! Solve by exact parallel column generation or build the full pool
//! by enumeration for validation. Both compute the LP relaxation. Integer
//! optimization requires binary **arc flows**, not binary extreme columns.

use crate::master::{LpMaster, RowSense};
use crate::{incomplete, invalid, Result};

mod master;
mod pricing;
pub use master::{ColumnGenerationOptions, ColumnGenerationResult, ColumnGenerationStatus};
pub use pricing::{PricingMethod, PricingOptions, PricingRound};

/// Directed CVRP with start depot 0, customers 1..=n and end depot n+1.
/// `None` denotes a missing arc; diagonal and depot-invalid arcs are ignored.
#[derive(Clone, Debug)]
pub struct Problem {
    pub demands: Vec<f64>,
    pub capacity: f64,
    pub costs: Vec<Vec<Option<f64>>>,
}

/// Explicit bounds on enumeration work; exceeding either returns an error,
/// never a partial formulation that could be mistaken for the full LP.
#[derive(Clone, Copy, Debug)]
pub struct EnumerationOptions {
    pub p: usize,
    pub max_columns: usize,
    pub max_extensions: usize,
}

impl Default for EnumerationOptions {
    fn default() -> Self {
        Self {
            p: 2,
            max_columns: 1_000_000,
            max_extensions: 10_000_000,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Step {
    pub path: Vec<usize>,
    pub prior_demand: f64,
    pub demand: f64,
    pub cost: f64,
}

/// Complete compact LP column pool. Row order: n degree rows (=2), n
/// endpoint-flow rows (=0), n load rows (>=0). Nonnegative continuous columns.
#[derive(Clone, Debug)]
pub struct Formulation {
    customers: usize,
    demands: Vec<f64>,
    steps: Vec<Step>,
}

impl Problem {
    /// Copy a parsed CVRPLIB instance using its declared distance convention.
    /// The conventional fleet size is not imposed by this unlimited-fleet model.
    pub fn from_instance(instance: &crate::inputs::Instance) -> Result<Self> {
        let n = instance
            .customers
            .len()
            .checked_sub(1)
            .ok_or_else(|| invalid("p-step instance needs a depot"))?;
        let depot = n + 1;
        let problem = Self {
            demands: instance.customers[1..].iter().map(|c| c.demand).collect(),
            capacity: instance.capacity,
            costs: (0..n + 2)
                .map(|i| {
                    (0..n + 2)
                        .map(|j| {
                            Some(instance.distance(
                                if i == depot { 0 } else { i },
                                if j == depot { 0 } else { j },
                                instance.rounding,
                            ))
                        })
                        .collect()
                })
                .collect(),
        };
        problem.validate()?;
        Ok(problem)
    }

    pub fn validate(&self) -> Result<()> {
        let n = self.demands.len();
        if n == 0
            || !self.capacity.is_finite()
            || self.capacity <= 0.0
            || self
                .demands
                .iter()
                .any(|&q| !q.is_finite() || q <= 0.0 || q > self.capacity)
            || self.costs.len() != n + 2
            || self.costs.iter().any(|row| row.len() != n + 2)
            || self
                .costs
                .iter()
                .flatten()
                .flatten()
                .any(|&c| !c.is_finite() || c < 0.0)
        {
            return Err(invalid("p-step requires positive customer demands within capacity and a finite nonnegative (n+2)-square cost matrix"));
        }
        Ok(())
    }

    pub fn enumerate(&self, options: EnumerationOptions) -> Result<Formulation> {
        self.validate()?;
        if options.p == 0 || options.max_columns == 0 || options.max_extensions == 0 {
            return Err(invalid("p-step p and enumeration limits must be positive"));
        }
        let mut search = Search {
            problem: self,
            options,
            pool: Formulation {
                customers: self.demands.len(),
                demands: self.demands.clone(),
                steps: Vec::new(),
            },
            extensions: 0,
        };
        for start in 0..=self.demands.len() {
            let demand = if start == 0 {
                0.0
            } else {
                self.demands[start - 1]
            };
            search.visit(&mut vec![start], demand, 0.0)?;
        }
        Ok(search.pool)
    }
}

struct Search<'a> {
    problem: &'a Problem,
    options: EnumerationOptions,
    pool: Formulation,
    extensions: usize,
}

impl Search<'_> {
    fn visit(&mut self, path: &mut Vec<usize>, demand: f64, cost: f64) -> Result<()> {
        let end = self.problem.demands.len() + 1;
        let at = *path.last().unwrap();
        let arcs = path.len() - 1;
        if arcs > 0 && (arcs == self.options.p || path[0] == 0) {
            let slack = self.problem.capacity - demand;
            // Proposition 1: depot-start -> zero; depot-end -> maximum;
            // internal -> both extremes (coincident extremes only once).
            let priors = if path[0] == 0 {
                vec![0.0]
            } else if at == end {
                vec![slack]
            } else if slack == 0.0 {
                vec![0.0]
            } else {
                vec![0.0, slack]
            };
            for prior_demand in priors {
                if self.pool.steps.len() >= self.options.max_columns {
                    return Err(incomplete("p-step column enumeration limit reached"));
                }
                self.pool.steps.push(Step {
                    path: path.clone(),
                    prior_demand,
                    demand,
                    cost,
                });
            }
        }
        if at == end || arcs == self.options.p {
            return Ok(());
        }
        for next in 1..=end {
            if self.extensions >= self.options.max_extensions {
                return Err(incomplete("p-step extension enumeration limit reached"));
            }
            self.extensions += 1;
            if path.contains(&next) || (path.len() == 1 && at == 0 && next == end) {
                continue;
            }
            let Some(arc) = self.problem.costs[at][next] else {
                continue;
            };
            let load = demand
                + if next == end {
                    0.0
                } else {
                    self.problem.demands[next - 1]
                };
            if load > self.problem.capacity {
                continue;
            }
            let total = cost + arc;
            if !total.is_finite() {
                return Err(invalid("p-step path cost overflow"));
            }
            path.push(next);
            self.visit(path, load, total)?;
            path.pop();
        }
        Ok(())
    }
}

impl Formulation {
    pub fn steps(&self) -> &[Step] {
        &self.steps
    }
    pub fn customers(&self) -> usize {
        self.customers
    }

    /// Sparse column for the documented row order, also usable for reduced
    /// cost evaluation c_r - sum_i a_ir * dual_i.
    pub fn entries(&self, column: usize) -> Result<Vec<(usize, f64)>> {
        let step = self
            .steps
            .get(column)
            .ok_or_else(|| invalid("invalid p-step column"))?;
        Ok(step_entries(&self.demands, step))
    }

    /// Populate an empty backend with the full continuous relaxation.
    /// Use the backend's solve/status API; only an optimal full-pool solve
    /// provides the p-step LP bound (subject to numerical tolerances).
    pub fn into_master<M: LpMaster>(&self, mut lp: M) -> Result<M> {
        if !lp.rhs().is_empty() || !lp.costs().is_empty() || lp.objective_offset() != 0.0 {
            return Err(invalid("p-step requires an empty LP backend"));
        }
        for (sense, rhs) in [
            (RowSense::Equal, 2.0),
            (RowSense::Equal, 0.0),
            (RowSense::GreaterEqual, 0.0),
        ] {
            for _ in 0..self.customers {
                lp.add_row(sense, rhs, &[])?;
            }
        }
        let entries: Vec<_> = self
            .steps
            .iter()
            .map(|s| step_entries(&self.demands, s))
            .collect();
        let columns: Vec<_> = self
            .steps
            .iter()
            .zip(&entries)
            .map(|(s, a)| (s.cost, a.as_slice()))
            .collect();
        let added = lp.add_columns(&columns)?;
        for column in added {
            // x<=1 is implied; leave a backend's looser default intact.
            if lp.upper()[column] < 1.0 {
                lp.set_upper(column, 1.0)?;
            }
        }
        Ok(lp)
    }
}

fn step_entries(demands: &[f64], step: &Step) -> Vec<(usize, f64)> {
    let n = demands.len();
    let last = step.path.len() - 1;
    let mut entries = Vec::new();
    for (pos, &v) in step.path.iter().enumerate() {
        if v == 0 || v == n + 1 {
            continue;
        }
        let endpoint = pos == 0 || pos == last;
        entries.push((v - 1, if endpoint { 1.0 } else { 2.0 }));
        if endpoint {
            let sign = if pos == 0 { 1.0 } else { -1.0 };
            entries.push((n + v - 1, sign));
            let load = if pos == 0 {
                demands[v - 1]
            } else {
                step.demand
            };
            let coefficient = sign * (step.prior_demand + load);
            if coefficient != 0.0 {
                entries.push((2 * n + v - 1, coefficient));
            }
        }
    }
    entries.sort_unstable_by_key(|&(row, _)| row);
    entries
}

#[cfg(test)]
mod tests;
