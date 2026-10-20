//! Assignment branching over shared unit-cover rows. Models requiring another
//! branching family return Incomplete rather than claiming integer optimality.
use super::*;
use crate::master::{tree::TreeStatus, LpMaster};
use std::time::Duration;

#[derive(Clone, Debug)]
pub struct AssignmentTreeOptions {
    pub node: Options,
    pub node_limit: usize,
    pub time_limit: Duration,
    pub tolerance: f64,
}
impl Default for AssignmentTreeOptions {
    fn default() -> Self {
        Self {
            node: Options::default(),
            node_limit: 10000,
            time_limit: Duration::from_secs(120),
            tolerance: 1e-6,
        }
    }
}

#[derive(Debug)]
pub struct AssignmentTreeOutput {
    pub status: TreeStatus,
    pub bound_mode: cg::BoundMode,
    pub bound: f64,
    pub incumbent: f64,
    /// Integer route quantities, retaining their network identities.
    pub routes: Vec<FixedRoute>,
    pub nodes: usize,
    pub open_nodes: usize,
    pub iterations: usize,
    pub pricing_calls: usize,
    pub oracle_calls: Vec<(i32, usize)>,
}

impl<const D: usize> NodeSnapshot<D> {
    // Replay rounded columns and fixed routes in the original graph and rows,
    // using exact sums and an upward objective bound.
    fn integer_witness(&self, tolerance: f64) -> Result<Option<(f64, Vec<FixedRoute>)>> {
        if !self.master.solved() {
            return Ok(None);
        }
        let mut routes = self.state.fixed_routes.clone();
        for (column, &value) in self.master.values().iter().enumerate() {
            let quantity = value.round();
            if !value.is_finite()
                || (value - quantity).abs() > tolerance
                || quantity < 0.
                || quantity > 9_007_199_254_740_992.
            {
                return Ok(None);
            }
            if quantity == 0. {
                continue;
            }
            if quantity > self.master.upper[column] {
                return Ok(None);
            }
            let Some(metadata) = &self.warm.columns[column] else {
                return Ok(None);
            };
            routes.push(FixedRoute {
                route: self.state.routes[metadata.tag as usize].clone(),
                quantity,
            });
        }
        let Some((mut rows, objective, counts)) =
            bounds::exact_contributions(&self.state, &routes)?
        else {
            return Ok(None);
        };
        for (activity, row) in rows.iter_mut().zip(&self.state.shared_rows) {
            if activity.add(-row.rhs).is_none() {
                return Ok(None);
            }
            let sign = activity.sign();
            if match row.sense {
                RowSense::Equal => sign != 0.,
                RowSense::LessEqual => sign > 0.,
                RowSense::GreaterEqual => sign < 0.,
            } {
                return Ok(None);
            }
        }
        for (count, oracle) in counts.iter().zip(&self.state.oracles) {
            let mut lower = count.clone();
            let mut upper = count.clone();
            if lower.add(-oracle.total_multiplicity.lower).is_none()
                || upper.add(-oracle.total_multiplicity.upper).is_none()
                || lower.sign() < 0.
                || upper.sign() > 0.
            {
                return Ok(None);
            }
        }
        let Some(objective) = objective.upper() else {
            return Ok(None);
        };
        Ok(Some((objective, routes)))
    }
}

/// Best-bound assignment tree. Each child prices all its networks and carries
/// inherited rows/fixed routes. Incomplete pricing or lack of an assignment
/// candidate leaves an open node; finite artificial penalties never prune it.
pub fn solve_assignment_tree<const D: usize>(
    problem: Problem<D>,
    options: AssignmentTreeOptions,
) -> Result<AssignmentTreeOutput> {
    validate_options(&options.node)?;
    problem.validate()?;
    if !options.tolerance.is_finite() || options.tolerance <= 0. || options.tolerance >= 0.5 {
        return Err(invalid(
            "assignment tree tolerance must be between zero and one half",
        ));
    }
    let deadline = Instant::now()
        .checked_add(options.time_limit)
        .ok_or_else(|| invalid("assignment tree time limit"))?;
    let mut result = AssignmentTreeOutput {
        status: TreeStatus::Incomplete,
        bound_mode: options.node.cg.bound_mode,
        bound: f64::NEG_INFINITY,
        incumbent: f64::INFINITY,
        routes: vec![],
        nodes: 0,
        open_nodes: 1,
        iterations: 0,
        pricing_calls: 0,
        oracle_calls: problem.networks.iter().map(|n| (n.id, 0)).collect(),
    };
    if options.node_limit == 0 {
        result.status = TreeStatus::NodeLimit;
        return Ok(result);
    }
    if options.time_limit.is_zero() {
        result.status = TreeStatus::TimeLimit;
        return Ok(result);
    }
    let node_options = |remaining: Duration| {
        let mut node = options.node.clone();
        node.cg.time_limit = node.cg.time_limit.min(remaining);
        node
    };
    let root = solve(
        problem,
        node_options(deadline.saturating_duration_since(Instant::now())),
    )?;
    let mut current = Some(root);
    let mut pending: Vec<NodeSnapshot<D>> = vec![];
    let mut unresolved = None;
    loop {
        let output = if let Some(output) = current.take() {
            output
        } else {
            if pending.is_empty() {
                result.status = TreeStatus::Optimal;
                break;
            }
            if result.nodes >= options.node_limit {
                result.status = TreeStatus::NodeLimit;
                break;
            }
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                result.status = TreeStatus::TimeLimit;
                break;
            }
            let best = (0..pending.len())
                .min_by(|&a, &b| pending[a].warm.bound.total_cmp(&pending[b].warm.bound))
                .unwrap();
            let node = pending.swap_remove(best);
            if node.warm.bound >= result.incumbent - options.tolerance {
                continue;
            }
            node.solve(node_options(remaining))?
        };
        result.nodes += 1;
        result.iterations += output.master.iterations;
        result.pricing_calls += output.master.pricing_calls;
        for &(id, calls) in &output.oracle_calls {
            if let Some((_, total)) = result
                .oracle_calls
                .iter_mut()
                .find(|(network, _)| *network == id)
            {
                *total += calls;
            }
        }
        let bound = output.master.bound;
        if output.master.status != cg::Status::Optimal {
            result.status = if output.master.status == cg::Status::TimeLimit {
                TreeStatus::TimeLimit
            } else {
                TreeStatus::Incomplete
            };
            unresolved = Some(bound);
            break;
        }
        if bound >= result.incumbent - options.tolerance {
            continue;
        }
        if let Some((objective, routes)) = output.snapshot.integer_witness(options.tolerance)? {
            if objective < result.incumbent {
                result.incumbent = objective;
                result.routes = routes;
            }
            if bound >= result.incumbent - options.tolerance {
                continue;
            }
        }
        let candidates = output.snapshot.fractional_assignments(options.tolerance)?;
        let Some(candidate) = candidates
            .iter()
            .min_by(|a, b| (a.value - 0.5).abs().total_cmp(&(b.value - 0.5).abs()))
        else {
            unresolved = Some(bound);
            break;
        };
        pending.push(output.snapshot.branch_assignment(
            candidate.shared_row,
            candidate.network,
            false,
        )?);
        pending.push(output.snapshot.branch_assignment(
            candidate.shared_row,
            candidate.network,
            true,
        )?);
    }
    // A frontier node already covered by the incumbent's bound is closed even
    // if its LP was not re-solved. Unknown bounds remain negative infinity.
    pending.retain(|node| node.warm.bound < result.incumbent - options.tolerance);
    result.open_nodes = pending.len() + usize::from(unresolved.is_some());
    result.bound = pending
        .iter()
        .map(|node| node.warm.bound)
        .chain(unresolved)
        .fold(result.incumbent, f64::min);
    if result.open_nodes == 0 && result.incumbent.is_finite() {
        result.status = TreeStatus::Optimal;
    }
    Ok(result)
}
