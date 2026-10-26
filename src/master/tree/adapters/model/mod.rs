//! Shared-network tree over assignment, integer master and original activities.
//! Unsupported fractional domains remain open rather than claiming optimality.
use super::*;
mod adapter;
mod branching;
pub(super) mod proposals;
mod time_windows;
use crate::master::{tree::TreeStatus, LpMaster};
use std::time::Duration;

const DEFAULT_OPEN_NODE_LIMIT: usize = 1000;

/// Family/child ordering only; it never changes a disjunction or a proof bound.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Deserialize, serde::Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum NetworkBranchingPolicy {
    /// Prefer fractional route counts, then use shared-resource/assignment fallback;
    /// commit count children upward first, matching the recovered native control.
    #[default]
    MultiplicityFirst,
    /// Existing shared-resource/assignment-first policy with lower children first.
    ResourceFirst,
    /// Evaluate registered original generators and fallback families together
    /// through the recovered phased scoring/history mechanisms.
    Phased,
}

impl NetworkBranchingPolicy {
    pub fn parse(value: &str) -> Result<Self> {
        match value {
            "resource-first" => Ok(Self::ResourceFirst),
            "multiplicity-first" => Ok(Self::MultiplicityFirst),
            "phased" => Ok(Self::Phased),
            _ => Err(invalid(
                "network branching must be resource-first, multiplicity-first or phased",
            )),
        }
    }
}

pub use crate::master::tree::{IntegerBoundRounding, TreeSearchPolicy};
/// Explicit model-family registration priorities. Defaults reuse the routing
/// adapter preset; native models can register other priorities.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct NetworkBranchPriorities {
    pub multiplicity: f64,
    pub resource: f64,
    pub assignment: f64,
    pub master_variable: f64,
    pub arc: f64,
    pub window: f64,
}
impl Default for NetworkBranchPriorities {
    fn default() -> Self {
        Self {
            multiplicity: 11.,
            resource: 10.,
            assignment: 3.,
            master_variable: 1.5,
            arc: 1.,
            window: 1.5,
        }
    }
}
impl NetworkBranchPriorities {
    fn validate(&self) -> Result<()> {
        if [
            self.multiplicity,
            self.resource,
            self.assignment,
            self.master_variable,
            self.arc,
            self.window,
        ]
        .iter()
        .all(|v| v.is_finite())
        {
            Ok(())
        } else {
            Err(invalid("finite network branch family priorities required"))
        }
    }
}

/// Constructor/trial order for a named integer branch generator. This does
/// not override the frontier's subsequent bound/depth comparison.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum NetworkBranchDirection {
    UpperFirst,
    LowerFirst,
}

impl NetworkBranchDirection {
    pub(super) fn sides(self) -> [usize; 2] {
        match self {
            Self::UpperFirst => [1, 0],
            Self::LowerFirst => [0, 1],
        }
    }
}

#[derive(Clone, Debug)]
pub struct ModelTreeOptions {
    pub node: Options,
    pub branching: NetworkBranchingPolicy,
    pub tree_search: TreeSearchPolicy,
    /// Rounding mode for strategy-2's integer-valued dual-bound queue key.
    pub strategy2_bound_rounding: IntegerBoundRounding,
    /// Number of nodes held in the native primary frontier before overflow.
    pub open_node_limit: usize,
    /// Native absolute optimality-gap control for continuous-objective branch
    /// scoring. This margin does not change published proof bounds.
    pub scoring_gap_tolerance: f64,
    pub branch_priorities: NetworkBranchPriorities,
    /// Named generator directions, requiring Phased (also selected by
    /// native packing). Side zero remains the lower bound side in history
    /// and proof storage, independently of trial/admission order.
    pub branch_directions: BTreeMap<String, NetworkBranchDirection>,
    /// Explicit original integer variables spanning networks and static masters.
    pub mapped_variables: Vec<MappedBranchVariable>,
    pub packing: Option<PackingBranching>,
    /// Reprice paths discovered in other tree nodes/trials in this recipient.
    /// Disabled by default; the retained native inactive-search gate is zero.
    pub share_inactive_columns: bool,
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
impl Default for ModelTreeOptions {
    fn default() -> Self {
        Self {
            node: Options::default(),
            branching: NetworkBranchingPolicy::default(),
            tree_search: TreeSearchPolicy::default(),
            strategy2_bound_rounding: IntegerBoundRounding::default(),
            open_node_limit: DEFAULT_OPEN_NODE_LIMIT,
            scoring_gap_tolerance: 1e-6,
            branch_priorities: NetworkBranchPriorities::default(),
            branch_directions: BTreeMap::new(),
            mapped_variables: vec![],
            packing: None,
            share_inactive_columns: false,
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
pub struct ModelTreeOutput {
    pub status: TreeStatus,
    pub bound_mode: cg::BoundMode,
    pub bound: f64,
    pub incumbent: f64,
    /// Integer route quantities, retaining their network identities.
    pub routes: Vec<FixedRoute>,
    pub master_values: Vec<(i32, f64)>,
    pub nodes: usize,
    /// Fresh node work, including exact-pricing trials and bound-pruned fresh nodes.
    /// Reusing a completed exact trial does not count it a second time.
    pub processed_nodes: usize,
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

type TreeFrontier<const D: usize> = crate::master::tree::TreeFrontier<NodeSnapshot<D>, Output<D>>;
pub(super) use crate::master::tree::queue_bound;
#[cfg(test)]
use crate::master::tree::{
    bound_priority_bound, bound_priority_less, bound_priority_values, BoundPriorityKey,
};
impl<const D: usize> crate::master::tree::NodeInfo for NodeSnapshot<D> {
    fn depth(&self) -> usize {
        self.state.node_depth
    }
    fn numerical_bound(&self) -> f64 {
        self.warm.numerical_bound
    }
}
impl<const D: usize> crate::master::tree::Evaluation for Output<D> {
    fn proven_infeasible(&self) -> bool {
        model_strong::proven_infeasible(self)
    }
}

impl<const D: usize> NodeSnapshot<D> {
    // Replay rounded columns and fixed routes in the original graph and rows,
    // using exact sums and an upward objective bound.
    pub(super) fn integer_witness(&self, tolerance: f64) -> Result<Option<IntegerWitness>> {
        let current = replay_integer_witness(
            &self.state,
            &self.master,
            |column| {
                self.warm
                    .columns
                    .get(column)
                    .and_then(|m| m.as_ref())
                    .map(|m| m.tag)
            },
            tolerance,
        )?;
        let retained = match &self.state.pricing_witness {
            Some(witness) => validate_routes(&self.state, &witness.1, &witness.2)?
                .map(|objective| (objective, witness.1.clone(), witness.2.clone())),
            None => None,
        };
        Ok(match (current, retained) {
            (Some(a), Some(b)) => Some(if a.0 <= b.0 { a } else { b }),
            (a, b) => a.or(b),
        })
    }
}

pub(super) fn replay_integer_witness<const D: usize>(
    state: &State<D>,
    master: &ClpMaster,
    metadata: impl Fn(usize) -> Option<i32>,
    tolerance: f64,
) -> Result<Option<IntegerWitness>> {
    if !master.solved() {
        return Ok(None);
    }
    let mut routes = state.fixed_routes.clone();
    let mut master_values: Vec<_> = state.master_variables.iter().map(|v| (v.id, 0.)).collect();
    for (column, &value) in master.values().iter().enumerate() {
        let tag = metadata(column);
        let static_variable = tag.and_then(master_variables::decode);
        let continuous = match static_variable {
            Some((index, _)) => match state.master_variables.get(index) {
                Some(variable) => !variable.integer,
                None => return Ok(None),
            },
            None => false,
        };
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
        if quantity > master.upper[column] {
            return Ok(None);
        }
        let Some(tag) = tag.filter(|tag| *tag != -1) else {
            return Ok(None);
        };
        if let Some((index, sign)) = static_variable {
            master_values[index].1 += sign * quantity;
            continue;
        }
        let Some(route) = usize::try_from(tag)
            .ok()
            .and_then(|tag| state.routes.get(tag))
        else {
            return Ok(None);
        };
        routes.push(FixedRoute {
            route: route.as_ref().clone(),
            quantity,
        });
    }
    if let Some(scheduled) = synchronization::witness(state, &routes)? {
        routes = scheduled;
    } else {
        return Ok(None);
    }
    Ok(validate_routes(state, &routes, &master_values)?
        .map(|objective| (objective, routes, master_values)))
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
/// inherited rows/fixed routes. A selected lower bound that rules out improving
/// an original-model incumbent closes the node even if its LP is unfinished.
/// Otherwise incomplete pricing or lack of an assignment candidate leaves an
/// open node; finite artificial penalties alone never prove infeasibility.
fn evaluate_model<const D: usize>(
    problem: Problem<D>,
    options: ModelTreeOptions,
) -> Result<ModelTreeOutput> {
    evaluate_with_variables(problem, vec![], options)
}

fn evaluate_with_variables<const D: usize>(
    problem: Problem<D>,
    variables: Vec<MasterVariable>,
    options: ModelTreeOptions,
) -> Result<ModelTreeOutput> {
    solve(Start::Fresh(problem, variables), options)
}

impl<const D: usize> NodeSnapshot<D> {
    /// Run a tree from this restricted model, preserving shared cuts, fixed
    /// routes, static variables and complete universes. The parent is reusable.
    pub fn solve_tree(&self, options: ModelTreeOptions) -> Result<ModelTreeOutput> {
        crate::master::tree::solve(self, options)
    }
}

enum Start<const D: usize> {
    Fresh(Problem<D>, Vec<MasterVariable>),
    Snapshot(NodeSnapshot<D>),
}

fn phased_fallback_selection<const D: usize>(
    children: Option<(Vec<NodeSnapshot<D>>, branching::ChildOrder)>,
) -> Option<model_strong::Selection<D>> {
    children.map(|(nodes, order)| {
        let mut selection = model_strong::Selection::fresh(nodes);
        if order == branching::ChildOrder::LowerThenUpper {
            selection.reverse();
        }
        selection
    })
}

fn solve<const D: usize>(
    start: Start<D>,
    mut options: ModelTreeOptions,
) -> Result<ModelTreeOutput> {
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
        for model in Arc::make_mut(&mut options.node.original_variable_costs) {
            if let Some(w) = windows.iter().find(|w| w.network == model.network) {
                model.copy_arcs(
                    w.split
                        .arcs
                        .iter()
                        .filter(|a| a.arc != a.original)
                        .map(|a| (a.arc, a.original)),
                );
            }
        }
        for group in &mut options.node.grouped_variable_boxes {
            if let Some(w) = windows.iter().find(|w| w.network == group.variable.network) {
                let original = group.variable.arcs.clone();
                for copy in w.split.arcs.iter().filter(|a| a.arc != a.original) {
                    if let Some(&(_, coefficient)) =
                        original.iter().find(|(id, _)| *id == copy.original)
                    {
                        group.variable.arcs.push((copy.arc, coefficient));
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
    mut options: ModelTreeOptions,
    windows: &[time_windows::NetworkWindows],
) -> Result<ModelTreeOutput> {
    validate_options(&options.node)?;
    options.node.integer_primal_termination = true;
    options.branch_priorities.validate()?;
    if options
        .branch_directions
        .keys()
        .any(|name| name.trim().is_empty())
    {
        return Err(invalid(
            "branch direction requires a nonempty generator name",
        ));
    }
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
        Start::Fresh(problem, _) => variable_boxes::prepare(
            &options.node.variable_boxes,
            problem.networks.iter().map(|n| (n.id, &n.graph, &[][..])),
        )?,
        Start::Snapshot(node) => variable_boxes::prepare(
            &options.node.variable_boxes,
            node.state.oracles.iter().map(|o| {
                (
                    o.network.id,
                    o.network.graph.as_ref(),
                    o.solver.forbidden_arcs.as_slice(),
                )
            }),
        )?,
    };
    match &start {
        Start::Fresh(problem, _) => grouped_boxes::prepare(
            &options.node.grouped_variable_boxes,
            problem.rows.len(),
            problem.networks.iter().map(|n| (n.id, &n.graph, &[][..])),
        )?,
        Start::Snapshot(node) => grouped_boxes::prepare(
            &options.node.grouped_variable_boxes,
            node.state.shared_rows.len(),
            node.state.oracles.iter().map(|o| {
                (
                    o.network.id,
                    o.network.graph.as_ref(),
                    o.solver.forbidden_arcs.as_slice(),
                )
            }),
        )?,
    };
    if let Some(packing) = &mut options.packing {
        match &start {
            Start::Fresh(problem, _) => packing.prepare(
                problem.networks.iter().map(|n| (n.id, &n.graph)),
                &options.mapped_variables,
            )?,
            Start::Snapshot(node) => packing.prepare(
                node.state
                    .oracles
                    .iter()
                    .map(|o| (o.network.id, o.network.graph.as_ref())),
                &options.mapped_variables,
            )?,
        }
        options.branching = NetworkBranchingPolicy::Phased;
    }
    if !options.branch_directions.is_empty() && options.branching != NetworkBranchingPolicy::Phased
    {
        return Err(invalid("named branch directions require phased branching"));
    }
    match &start {
        Start::Fresh(problem, variables) => model_mapping::validate::<D>(
            &options.mapped_variables,
            problem.networks.iter().map(|n| (n.id, &n.graph)),
            variables,
        )?,
        Start::Snapshot(node) => model_mapping::validate::<D>(
            &options.mapped_variables,
            node.state
                .oracles
                .iter()
                .map(|o| (o.network.id, o.network.graph.as_ref())),
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
    if !options.scoring_gap_tolerance.is_finite() || options.scoring_gap_tolerance < 0. {
        return Err(invalid("scoring gap tolerance"));
    }
    if !options.tolerance.is_finite() || options.tolerance <= 0. || options.tolerance >= 0.5 {
        return Err(invalid(
            "assignment tree tolerance must be between zero and one half",
        ));
    }
    let deadline = Instant::now()
        .checked_add(options.time_limit)
        .ok_or_else(|| invalid("assignment tree time limit"))?;
    let mut result = ModelTreeOutput {
        status: TreeStatus::Incomplete,
        bound_mode: options.node.cg.bound_mode,
        bound: initial_bound,
        incumbent: f64::INFINITY,
        routes: vec![],
        master_values: vec![],
        nodes: 0,
        processed_nodes: 0,
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
        node.search_inactive_columns |= options.share_inactive_columns;
        node.cg.time_limit = node.cg.time_limit.min(remaining);
        node
    };
    let configured = node_options(deadline.saturating_duration_since(Instant::now()));
    let root = match start {
        Start::Fresh(problem, variables) => {
            solve_with_master_variables(problem, variables, configured)?
        }
        Start::Snapshot(mut node) => {
            node.state.mapped_lineage = 0;
            node.solve(configured)?
        }
    };
    let base_depth = root.snapshot.state.node_depth;
    let mut subtree_history = options
        .mapped_strong_branching
        .as_ref()
        .filter(|s| s.use_history_subtree_size)
        .map(|_| crate::master::tree::strong::SubtreeHistory::new());
    let mut pending = TreeFrontier::new(
        options.tree_search,
        options.open_node_limit,
        options.strategy2_bound_rounding,
    );
    let mut unresolved = None;
    let mut mapped_history = model_strong::History::default();
    let mut selected_nodes = 0;
    let stop = crate::master::tree::run(
        &mut adapter::NetworkSearch {
            options: &options,
            result: &mut result,
            deadline,
            windows,
            base_depth,
            subtree_history: &mut subtree_history,
            mapped_history: &mut mapped_history,
            selected_nodes: &mut selected_nodes,
            unresolved: &mut unresolved,
        },
        &mut pending,
        Some(root),
    )?;
    match stop {
        crate::master::tree::Stop::Exhausted => result.status = TreeStatus::Optimal,
        crate::master::tree::Stop::OverflowOnly | crate::master::tree::Stop::Incomplete => {
            result.status = TreeStatus::Incomplete
        }
        crate::master::tree::Stop::NodeLimit => result.status = TreeStatus::NodeLimit,
        crate::master::tree::Stop::TimeLimit => result.status = TreeStatus::TimeLimit,
        crate::master::tree::Stop::Requested => {}
    }
    result.pricing_calls += mapped_history.pricing_calls;
    result.iterations += mapped_history.iterations;
    for (id, calls) in mapped_history.oracle_calls {
        if let Some((_, total)) = result
            .oracle_calls
            .iter_mut()
            .find(|(network, _)| *network == id)
        {
            *total += calls;
        }
    }
    result.strong_branching_trials = mapped_history.trials;
    result.mapped_history_ranked = mapped_history.ranked;
    result.mapped_branch_evaluations = mapped_history.evaluations;
    // A frontier node already covered by the incumbent's bound is closed even
    // if its LP was not re-solved. Unknown bounds remain negative infinity.
    let mut pending = pending.into_nodes();
    pending.retain(|node| {
        let keep = !node.proven_infeasible()
            && node.node.warm.bound < result.incumbent - options.tolerance;
        if !keep {
            if let Some(history) = &mut subtree_history {
                history.close(node.node.state.mapped_lineage);
            }
        }
        keep
    });
    result.open_nodes = pending.len() + usize::from(unresolved.is_some());
    result.bound = pending
        .iter()
        .map(|node| node.node.warm.bound)
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

#[cfg(test)]
#[path = "../../../../../tests/unit/master/tree/model_tests.rs"]
mod tree_search_tests;

#[cfg(test)]
#[path = "../../../../../tests/unit/master/model/stalled_bound_tests.rs"]
mod stalled_bound_tests;

impl<const D: usize> crate::master::tree::TreeInput for Problem<D> {
    type Options = ModelTreeOptions;
    type Output = ModelTreeOutput;
    fn run_tree(self, options: Self::Options) -> Result<Self::Output> {
        evaluate_model(self, options)
    }
}
impl<const D: usize> crate::master::tree::TreeInput for (Problem<D>, Vec<MasterVariable>) {
    type Options = ModelTreeOptions;
    type Output = ModelTreeOutput;
    fn run_tree(self, options: Self::Options) -> Result<Self::Output> {
        evaluate_with_variables(self.0, self.1, options)
    }
}
impl<const D: usize> crate::master::tree::TreeInput for &NodeSnapshot<D> {
    type Options = ModelTreeOptions;
    type Output = ModelTreeOutput;
    fn run_tree(self, options: Self::Options) -> Result<Self::Output> {
        solve(Start::Snapshot(self.fork()?), options)
    }
}
