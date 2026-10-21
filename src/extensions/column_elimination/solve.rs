use super::{diagram::Diagram, Options, Problem};
use crate::master::{LpMaster, LpSolveOutcome, RowSense};
use crate::{incomplete, invalid, Result};
use std::time::Instant;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Status {
    Converged,
    Infeasible,
    IterationLimit,
    TimeLimit,
    DiagramLimit,
    LpStopped(LpSolveOutcome),
}

#[derive(Clone, Debug)]
pub struct RouteFlow {
    /// Includes both depot indices.
    pub path: Vec<usize>,
    pub flow: f64,
    pub cost: f64,
}

#[derive(Clone, Debug)]
pub struct SolveResult {
    pub status: Status,
    /// Last optimal relaxed-diagram objective. A valid lower bound even before
    /// convergence, subject to numerical tolerances. None before a solved LP.
    pub lower_bound: Option<f64>,
    /// Feasible route decomposition, populated only at convergence. Fractional
    /// flows are allowed: these routes are not an integer incumbent.
    pub routes: Vec<RouteFlow>,
    pub iterations: usize,
    pub refinements: usize,
    pub nodes: usize,
    /// Includes disabled arcs, which count toward the memory budget.
    pub arcs: usize,
}

impl Problem {
    /// Solve the unlimited- or bounded-fleet set-partitioning route LP by
    /// column elimination. `make_lp` must supply a fresh empty backend each
    /// iteration; rebuilding avoids backend-dependent basis mutation.
    ///
    /// The initial diagram allows repeated visits and ignores resources.
    /// Refinement deletes only prefixes proven infeasible by exact replay.
    /// The initial diagram exceeding its budget returns an error; subsequent
    /// budgets return an explicit non-converged status and the last bound.
    pub fn solve_column_elimination<M: LpMaster>(
        &self,
        mut make_lp: impl FnMut() -> Result<M>,
        options: Options,
    ) -> Result<SolveResult> {
        self.validate()?;
        if options.max_nodes < 2
            || options.log_interval == Some(0)
            || options.max_arcs == 0
            || !options.tolerance.is_finite()
            || options.tolerance < 1e-9
            || options.tolerance > 1e-4
        {
            return Err(invalid("invalid column elimination options"));
        }
        let started = Instant::now();
        let mut dd = Diagram::new(self, &options)?;
        let mut progress = super::progress::ProgressLog::new(options.log_interval, started);
        let mut result = SolveResult {
            status: Status::IterationLimit,
            lower_bound: None,
            routes: vec![],
            iterations: 0,
            refinements: 0,
            nodes: dd.nodes.len(),
            arcs: dd.arcs.len(),
        };
        for _ in 0..options.max_iterations {
            if started.elapsed() >= options.time_limit {
                result.status = Status::TimeLimit;
                break;
            }
            let mut lp = dd.master(self, make_lp()?, options.tolerance)?;
            let outcome =
                lp.solve_with_time_limit(options.time_limit.saturating_sub(started.elapsed()))?;
            result.iterations += 1;
            if outcome != LpSolveOutcome::Optimal {
                result.status = if outcome == LpSolveOutcome::PrimalInfeasible {
                    Status::Infeasible
                } else if outcome == LpSolveOutcome::TimeLimit {
                    Status::TimeLimit
                } else {
                    Status::LpStopped(outcome)
                };
                break;
            }
            if !lp.solved() || !lp.objective().is_finite() {
                return Err(incomplete(
                    "column elimination backend returned invalid optimal solution",
                ));
            }
            let paths = dd.decompose(lp.values(), options.tolerance)?;
            result.lower_bound = Some(lp.objective());
            let mut conflict = None;
            let mut infeasible = 0;
            for (ids, route) in &paths {
                if let Some(i) = self.conflict(&route.path) {
                    infeasible += 1;
                    if conflict.is_none() {
                        conflict = Some(ids[..=i].to_vec());
                    }
                }
            }
            progress.iteration(&result, paths.len(), infeasible);
            if let Some(prefix) = conflict {
                if started.elapsed() >= options.time_limit {
                    result.status = Status::TimeLimit;
                    break;
                }
                if !dd.separate(&prefix, &options)? {
                    result.status = Status::DiagramLimit;
                    break;
                }
                result.refinements += 1;
                result.nodes = dd.nodes.len();
                result.arcs = dd.arcs.len();
            } else {
                let routes: Vec<_> = paths.into_iter().map(|(_, r)| r).collect();
                // Reject appreciable residual losses instead of claiming convergence
                // from an incomplete numerical decomposition.
                let mut visits = vec![0.0; self.demands.len()];
                let mut cost = 0.0;
                for r in &routes {
                    cost += r.flow * r.cost;
                    for &v in &r.path[1..r.path.len() - 1] {
                        visits[v - 1] += r.flow;
                    }
                }
                if visits
                    .iter()
                    .any(|v| (v - 1.0).abs() > options.tolerance * 10.0)
                    || (cost - lp.objective()).abs() > options.tolerance * 10.0 * (1.0 + cost.abs())
                    || self.max_vehicles.is_some_and(|k| {
                        routes.iter().map(|r| r.flow).sum::<f64>() > k + options.tolerance * 10.0
                    })
                {
                    return Err(incomplete(
                        "column elimination flow decomposition failed numerical checks",
                    ));
                }
                result.routes = routes;
                result.status = Status::Converged;
                break;
            }
        }
        progress.finish(&result);
        Ok(result)
    }
}

impl Diagram {
    fn master<M: LpMaster>(&self, problem: &Problem, mut lp: M, tolerance: f64) -> Result<M> {
        if !lp.rhs().is_empty() || !lp.costs().is_empty() || lp.objective_offset() != 0.0 {
            return Err(invalid("column elimination requires an empty LP backend"));
        }
        lp.set_tolerances(tolerance * 0.1, tolerance * 0.1)?;
        let n = problem.demands.len();
        for _ in 0..n {
            lp.add_row(RowSense::Equal, 1.0, &[])?;
        }
        for _ in 2..self.nodes.len() {
            lp.add_row(RowSense::Equal, 0.0, &[])?;
        }
        let fleet_row = if let Some(k) = problem.max_vehicles {
            Some(lp.add_row(RowSense::LessEqual, k, &[])?)
        } else {
            None
        };
        // Keep a one-to-one mapping between arc IDs and LP columns, including
        // disabled arcs fixed at zero. No flow row is needed at root or sink.
        for arc in &self.arcs {
            let mut entries = Vec::with_capacity(4);
            let location = self.nodes[arc.to].location;
            if location <= n {
                entries.push((location - 1, 1.0));
            }
            if arc.from >= 2 {
                entries.push((n + arc.from - 2, -1.0));
            }
            if arc.to >= 2 {
                entries.push((n + arc.to - 2, 1.0));
            }
            if arc.from == 0 {
                if let Some(row) = fleet_row {
                    entries.push((row, 1.0));
                }
            }
            entries.sort_unstable_by_key(|e| e.0);
            let column = lp.add_column(arc.cost, &entries)?;
            // Degree rows imply each nonnegative arc flow <=1.
            lp.set_upper(column, if arc.active { 1.0 } else { 0.0 })?;
        }
        Ok(lp)
    }

    fn decompose(&self, values: &[f64], tolerance: f64) -> Result<Vec<(Vec<usize>, RouteFlow)>> {
        if values.len() != self.arcs.len()
            || values.iter().any(|x| !x.is_finite() || *x < -tolerance)
        {
            return Err(incomplete("column elimination invalid arc flow vector"));
        }
        let mut residual = values.to_vec();
        let mut routes = vec![];
        // Each decomposition exhausts at least one arc. An acyclic diagram
        // has no positive circulation; any significant residue is an error.
        loop {
            let mut at = 0;
            let mut ids = vec![];
            let mut path = vec![0];
            let mut flow = f64::INFINITY;
            let mut cost = 0.0;
            while at != 1 {
                let next = self.nodes[at]
                    .outgoing
                    .iter()
                    .copied()
                    .find(|&a| self.arcs[a].active && residual[a] > tolerance * 0.01);
                let Some(id) = next else {
                    if at == 0 {
                        break;
                    }
                    return Err(incomplete(
                        "column elimination positive flow reaches a dead end",
                    ));
                };
                let arc = &self.arcs[id];
                flow = flow.min(residual[id]);
                cost += arc.cost;
                ids.push(id);
                at = arc.to;
                path.push(self.nodes[at].location);
            }
            if ids.is_empty() {
                break;
            }
            if !cost.is_finite() {
                return Err(invalid("column elimination route cost overflow"));
            }
            for &id in &ids {
                residual[id] -= flow;
            }
            routes.push((ids, RouteFlow { path, flow, cost }));
        }
        if residual.iter().any(|x| *x > tolerance) {
            return Err(incomplete("column elimination undecomposed residual flow"));
        }
        Ok(routes)
    }
}
