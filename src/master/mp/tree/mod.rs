//! Shared-network tree over assignment, integer master and original activities.
//! Unsupported fractional domains remain open rather than claiming optimality.
use super::*;
mod branching;
mod time_windows;
use crate::master::{tree::TreeStatus, LpMaster};
use std::time::Duration;

/// Family/child ordering only; it never changes a disjunction or a proof bound.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Deserialize, serde::Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum NetworkBranchingPolicy {
    /// Existing shared-resource/assignment-first policy with lower children first.
    #[default]
    ResourceFirst,
    /// Network route counts first, upper children first, as in the recovered
    /// public native controls. Other model/backend settings can change the tree.
    MultiplicityFirst,
}

impl NetworkBranchingPolicy {
    pub fn parse(value: &str) -> Result<Self> {
        match value {
            "resource-first" => Ok(Self::ResourceFirst),
            "multiplicity-first" => Ok(Self::MultiplicityFirst),
            _ => Err(invalid(
                "network branching must be resource-first or multiplicity-first",
            )),
        }
    }
}

#[derive(Clone, Debug)]
pub struct AssignmentTreeOptions {
    pub node: Options,
    pub branching: NetworkBranchingPolicy,
    /// Explicit original integer variables spanning networks and static masters.
    pub mapped_variables: Vec<MappedBranchVariable>,
    /// Phased trials rank explicit mapped generators; trial values never prune.
    pub mapped_strong_branching: Option<crate::master::tree::StrongBranchingOptions>,
    pub node_limit: usize,
    pub time_limit: Duration,
    pub tolerance: f64,
    pub enumerated_mip: Option<crate::master::solvers::mip::MipOptions>,
    /// Branch on how often each time window of a network vertex is selected,
    /// on this disposable resource (multi-time-windows builds, fresh problems
    /// only). Arcs into multi-window vertices are copied once per window;
    /// reported routes use the input arc IDs.
    pub time_window_selection: Option<usize>,
}
impl Default for AssignmentTreeOptions {
    fn default() -> Self {
        Self {
            node: Options::default(),
            branching: NetworkBranchingPolicy::default(),
            mapped_variables: vec![],
            mapped_strong_branching: Some(crate::master::tree::StrongBranchingOptions::default()),
            node_limit: 10000,
            time_limit: Duration::from_secs(120),
            tolerance: 1e-6,
            enumerated_mip: None,
            time_window_selection: None,
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
    pub master_values: Vec<(i32, f64)>,
    pub nodes: usize,
    pub strong_branching_trials: usize,
    pub mapped_history_ranked: usize,
    pub mapped_branch_evaluations: Vec<MappedBranchEvaluation>,
    pub open_nodes: usize,
    pub iterations: usize,
    pub pricing_calls: usize,
    pub oracle_calls: Vec<(i32, usize)>,
    pub enumerated_mip_calls: usize,
    pub enumerated_mip_nodes: usize,
    pub enumerated_mip_failures: usize,
    pub enumeration_reports: Vec<EnumerationReport>,
}

impl<const D: usize> NodeSnapshot<D> {
    // Replay rounded columns and fixed routes in the original graph and rows,
    // using exact sums and an upward objective bound.
    pub(super) fn integer_witness(
        &self,
        tolerance: f64,
    ) -> Result<Option<(f64, Vec<FixedRoute>, Vec<(i32, f64)>)>> {
        if !self.master.solved() {
            return Ok(None);
        }
        let mut routes = self.state.fixed_routes.clone();
        let mut master_values: Vec<_> = self
            .state
            .master_variables
            .iter()
            .map(|v| (v.id, 0.))
            .collect();
        for (column, &value) in self.master.values().iter().enumerate() {
            let static_variable = self.warm.columns[column]
                .as_ref()
                .and_then(|m| master_variables::decode(m.tag));
            let continuous =
                static_variable.is_some_and(|(i, _)| !self.state.master_variables[i].integer);
            let quantity = if continuous { value } else { value.round() };
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
            if let Some((index, sign)) = static_variable {
                master_values[index].1 += sign * quantity;
                continue;
            }
            routes.push(FixedRoute {
                route: self.state.routes[metadata.tag as usize].as_ref().clone(),
                quantity,
            });
        }
        if let Some(scheduled) = synchronization::witness(&self.state, &routes)? {
            routes = scheduled;
        } else {
            return Ok(None);
        }
        Ok(validate_routes(&self.state, &routes, &master_values)?
            .map(|objective| (objective, routes, master_values)))
    }
}

pub(super) fn validate_routes<const D: usize>(
    state: &State<D>,
    routes: &[FixedRoute],
    master_values: &[(i32, f64)],
) -> Result<Option<f64>> {
    if synchronization::witness(state, routes)?.is_none() {
        return Ok(None);
    }
    let Some((mut rows, mut objective, counts)) = bounds::exact_contributions(state, routes)?
    else {
        return Ok(None);
    };
    if !master_variables::witness(state, master_values, &mut rows, &mut objective) {
        return Ok(None);
    }
    for (activity, row) in rows.iter_mut().zip(&state.shared_rows) {
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
    for (count, oracle) in counts.iter().zip(&state.oracles) {
        let mut lower = count.clone();
        let mut upper = count.clone();
        if lower.add(-oracle.total_multiplicity.lower).is_none()
            || lower.sign() < 0.
            || oracle.total_multiplicity.upper.is_finite()
                && (upper.add(-oracle.total_multiplicity.upper).is_none() || upper.sign() > 0.)
        {
            return Ok(None);
        }
    }
    let Some(objective) = objective.upper() else {
        return Ok(None);
    };
    Ok(Some(objective))
}

/// Best-bound assignment tree. Each child prices all its networks and carries
/// inherited rows/fixed routes. Incomplete pricing or lack of an assignment
/// candidate leaves an open node; finite artificial penalties never prune it.
pub fn solve_assignment_tree<const D: usize>(
    problem: Problem<D>,
    options: AssignmentTreeOptions,
) -> Result<AssignmentTreeOutput> {
    solve_assignment_tree_with_master_variables(problem, vec![], options)
}

pub fn solve_assignment_tree_with_master_variables<const D: usize>(
    problem: Problem<D>,
    variables: Vec<MasterVariable>,
    options: AssignmentTreeOptions,
) -> Result<AssignmentTreeOutput> {
    solve(Start::Fresh(problem, variables), options)
}

impl<const D: usize> NodeSnapshot<D> {
    /// Run a tree from this restricted model, preserving shared cuts, fixed
    /// routes, static variables and complete universes. The parent is reusable.
    pub fn solve_assignment_tree(
        &self,
        options: AssignmentTreeOptions,
    ) -> Result<AssignmentTreeOutput> {
        solve(Start::Snapshot(self.fork()?), options)
    }
}

enum Start<const D: usize> {
    Fresh(Problem<D>, Vec<MasterVariable>),
    Snapshot(NodeSnapshot<D>),
}

fn solve<const D: usize>(
    start: Start<D>,
    mut options: AssignmentTreeOptions,
) -> Result<AssignmentTreeOutput> {
    let Some(resource) = options.time_window_selection else {
        return solve_split(start, options, &[]);
    };
    let Start::Fresh(mut problem, variables) = start else {
        return Err(invalid(
            "time-window selection branching requires a fresh network problem",
        ));
    };
    #[cfg(not(feature = "multi-time-windows"))]
    {
        let _ = (resource, &mut problem, variables);
        Err(invalid(
            "time-window selection branching requires multi-time-windows",
        ))
    }
    #[cfg(feature = "multi-time-windows")]
    {
        problem.validate()?;
        let windows = time_windows::split(&mut problem, resource)?;
        for variable in &mut options.mapped_variables {
            for contribution in &mut variable.networks {
                if let Some(w) = windows.iter().find(|w| w.network == contribution.network) {
                    let original = contribution.arcs.clone();
                    for copy in w.split.arcs.iter().filter(|a| a.arc != a.original) {
                        if let Some(&(_, coefficient)) =
                            original.iter().find(|(id, _)| *id == copy.original)
                        {
                            contribution.arcs.push((copy.arc, coefficient));
                        }
                    }
                }
            }
        }
        let mut output = solve_split(Start::Fresh(problem, variables), options, &windows)?;
        for fixed in &mut output.routes {
            time_windows::restore(&windows, &mut fixed.route);
        }
        Ok(output)
    }
}

fn solve_split<const D: usize>(
    start: Start<D>,
    mut options: AssignmentTreeOptions,
    windows: &[time_windows::NetworkWindows],
) -> Result<AssignmentTreeOutput> {
    validate_options(&options.node)?;
    if let Some(strong) = &options.mapped_strong_branching {
        model_strong::validate(strong)?;
    }
    if options.enumerated_mip.is_some() {
        if !cfg!(feature = "highs") {
            return Err(invalid("enumerated MIP requires the highs feature"));
        }
        if options.node.enumeration.is_none() {
            options.node.enumeration = Some(EnumerationOptions::default());
        }
    }
    match &start {
        Start::Fresh(problem, variables) => model_mapping::validate::<D>(
            &options.mapped_variables,
            problem
                .networks
                .iter()
                .map(|n| (n.id, n.graph.arcs.iter().map(|a| a.id).collect())),
            variables,
        )?,
        Start::Snapshot(node) => model_mapping::validate::<D>(
            &options.mapped_variables,
            node.state.oracles.iter().map(|o| {
                (
                    o.network.id,
                    o.network.graph.arcs.iter().map(|a| a.id).collect(),
                )
            }),
            &node.state.master_variables,
        )?,
    }
    let networks: Vec<_> = match &start {
        Start::Fresh(problem, variables) => {
            problem.validate()?;
            if options.node_limit == 0 || options.time_limit.is_zero() {
                // Zero-budget runs still reject malformed synchronization input.
                synchronization::prepare(
                    &mut problem.clone(),
                    &mut variables.clone(),
                    options.node.synchronized_services.clone(),
                )?;
            }
            problem.networks.iter().map(|n| n.id).collect()
        }
        Start::Snapshot(node) => node.state.oracles.iter().map(|o| o.network.id).collect(),
    };
    let initial_bound = match &start {
        Start::Fresh(..) => f64::NEG_INFINITY,
        Start::Snapshot(node) => match options.node.cg.bound_mode {
            cg::BoundMode::Numerical => node.warm.numerical_bound,
            cg::BoundMode::Certified => node.warm.certified_bound,
        },
    };
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
        bound: initial_bound,
        incumbent: f64::INFINITY,
        routes: vec![],
        master_values: vec![],
        nodes: 0,
        strong_branching_trials: 0,
        mapped_history_ranked: 0,
        mapped_branch_evaluations: vec![],
        open_nodes: 1,
        iterations: 0,
        pricing_calls: 0,
        oracle_calls: networks.into_iter().map(|id| (id, 0)).collect(),
        enumerated_mip_calls: 0,
        enumerated_mip_nodes: 0,
        enumerated_mip_failures: 0,
        enumeration_reports: vec![],
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
    let configured = node_options(deadline.saturating_duration_since(Instant::now()));
    let root = match start {
        Start::Fresh(problem, variables) => {
            solve_with_master_variables(problem, variables, configured)?
        }
        Start::Snapshot(node) => node.solve(configured)?,
    };
    let mut current = Some(root);
    let mut pending: Vec<NodeSnapshot<D>> = vec![];
    let mut unresolved = None;
    let mut mapped_history = model_strong::History::default();
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
        result
            .enumeration_reports
            .push(output.snapshot.enumeration_report().clone());
        let bound = output.master.bound;
        // The whole original-flow relaxation proves LP infeasibility without
        // integer rounding. Complete-master integer deductions retain their
        // explicitly requested integer-domain policy.
        let preprocessing = output.snapshot.preprocessing_report();
        if preprocessing.propagation.infeasible
            && (!preprocessing.complete_domain || options.node.preprocessing.integer_domains)
        {
            continue;
        }
        if let Some(configured) = &options.enumerated_mip {
            let mut configured = configured.clone();
            configured.time_limit = configured
                .time_limit
                .min(deadline.saturating_duration_since(Instant::now()));
            configured.objective_cutoff = configured.objective_cutoff.min(result.incumbent);
            let attempt = if output.snapshot.enumeration_report().complete {
                Some(output.snapshot.solve_enumerated_mip(&configured))
            } else if configured.objective_cutoff.is_finite() {
                let mut limits = options.node.enumeration.clone().unwrap_or_default();
                limits.time_limit = limits
                    .time_limit
                    .min(deadline.saturating_duration_since(Instant::now()));
                match output
                    .snapshot
                    .enumerate_below(configured.objective_cutoff, &limits)
                {
                    Ok(view) => {
                        result.enumeration_reports.push(view.report().clone());
                        configured.time_limit = configured
                            .time_limit
                            .min(deadline.saturating_duration_since(Instant::now()));
                        Some(view.solve_mip(&configured).map(Some))
                    }
                    Err(error) => {
                        result.enumeration_reports.push(EnumerationReport {
                            attempts: 1,
                            coverage_cutoff: Some(configured.objective_cutoff),
                            failure: Some(error.to_string()),
                            ..Default::default()
                        });
                        None
                    }
                }
            } else {
                None
            };
            if let Some(attempt) = attempt {
                result.enumerated_mip_calls += 1;
                match attempt {
                    Ok(Some(mip)) => {
                        use crate::master::solvers::mip::MipStatus;
                        result.enumerated_mip_nodes += mip.mip.nodes;
                        if let Some(solution) = &mip.mip.solution {
                            if solution.objective < result.incumbent {
                                result.incumbent = solution.objective;
                                result.routes = mip.route_values;
                                result.master_values = mip.master_values;
                            }
                        }
                        // HiGHS supplies a numerical integer bound; it never
                        // becomes a directed-arithmetic publication certificate.
                        if options.node.cg.bound_mode == cg::BoundMode::Numerical {
                            if mip.mip.status == MipStatus::Infeasible {
                                continue;
                            }
                            if matches!(
                                mip.mip.status,
                                MipStatus::Optimal | MipStatus::ObjectiveCutoff
                            ) && result.incumbent.is_finite()
                                && mip.mip.bound >= result.incumbent - options.tolerance
                            {
                                continue;
                            }
                        }
                    }
                    Err(_) => {
                        result.enumerated_mip_failures += 1;
                    }
                    Ok(None) => {}
                }
            }
        }
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
        if let Some((objective, routes, master_values)) =
            output.snapshot.integer_witness(options.tolerance)?
        {
            if objective < result.incumbent {
                result.incumbent = objective;
                result.routes = routes;
                result.master_values = master_values;
            }
            if bound >= result.incumbent - options.tolerance {
                continue;
            }
        }
        let count_children = if options.branching == NetworkBranchingPolicy::MultiplicityFirst {
            branching::multiplicity_children(&output, options.tolerance)?
        } else {
            None
        };
        let children = if count_children.is_some() {
            count_children
        } else if let Some(children) = model_strong::choose(
            &output,
            &options,
            &mut mapped_history,
            result.incumbent,
            deadline,
        )? {
            Some(children)
        } else {
            branching::children(&output, options.tolerance, options.branching, windows, &[])?
        };
        if let Some(mut children) = children {
            for child in &mut children {
                child.state.node_depth = output.snapshot.state.node_depth.saturating_add(1);
            }
            pending.extend(children);
        } else {
            unresolved = Some(bound);
            break;
        }
    }
    result.strong_branching_trials = mapped_history.trials;
    result.mapped_history_ranked = mapped_history.ranked;
    result.mapped_branch_evaluations = mapped_history.evaluations;
    // A frontier node already covered by the incumbent's bound is closed even
    // if its LP was not re-solved. Unknown bounds remain negative infinity.
    pending.retain(|node| node.warm.bound < result.incumbent - options.tolerance);
    result.open_nodes = pending.len() + usize::from(unresolved.is_some());
    result.bound = pending
        .iter()
        .map(|node| node.warm.bound)
        .chain(unresolved)
        .fold(result.incumbent, f64::min);
    if result.open_nodes == 0 {
        result.status = if result.incumbent.is_finite() {
            TreeStatus::Optimal
        } else {
            TreeStatus::Infeasible
        };
    }
    Ok(result)
}
