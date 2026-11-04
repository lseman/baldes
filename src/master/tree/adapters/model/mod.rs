//! Shared-network tree over assignment, integer master and original activities.
//! Unsupported fractional domains remain open rather than claiming optimality.
use super::*;
mod adapter;
mod branching;
mod diving;
mod fixed_solution_cuts;
mod greedy;
mod integer_objective;
mod local_search;
mod master_heuristic;
mod parallel;
mod primal_callbacks;
pub use diving::DivingOptions;
pub use fixed_solution_cuts::{FixedSolutionCut, FixedSolutionInput, FixedSolutionSeparator};
pub use local_search::LocalSearchOptions;
pub use master_heuristic::{
    MasterHeuristic, MasterHeuristicInput, ResidualMasterCoordinate, ResidualMasterSolution,
};
pub use primal_callbacks::{
    PrimalHeuristic, PrimalHeuristicInput, PrimalHeuristicSolution, PrimalHeuristicSource,
};
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
    /// Unified configuration and registration of all branching families.
    pub branching_registry: super::BranchingRegistry,
    pub node: Options,
    pub tree_search: TreeSearchPolicy,
    /// Rounding mode for strategy-2's integer-valued dual-bound queue key.
    pub strategy2_bound_rounding: IntegerBoundRounding,
    /// Number of nodes held in the native primary frontier before overflow.
    pub open_node_limit: usize,
    /// Native absolute optimality-gap control for continuous-objective branch
    /// scoring. This margin does not change published proof bounds.
    pub scoring_gap_tolerance: f64,
    /// Diving primal heuristic; disabled by default, as in the native solver.
    pub diving: Option<DivingOptions>,
    /// Construct an initial residual solution by shifted-cost network pricing.
    pub greedy: bool,
    /// Application-defined primal candidates, replayed before acceptance.
    pub primal_heuristics: Vec<PrimalHeuristic>,
    /// Residual master candidate before dive child cuts and column generation.
    pub master_heuristic: Option<MasterHeuristic>,
    /// User presence cuts separated before a dive child evaluates its fixed solution.
    pub fixed_solution_separators: Vec<FixedSolutionSeparator>,
    /// Random incumbent partial-column fixing and child diving; disabled by default.
    pub local_search: Option<LocalSearchOptions>,
    /// Restricted-master integer heuristic; disabled by default, as in native.
    pub restricted_master_mip: Option<RestrictedMasterMip>,
    /// Native `optimalityGapTolerance`: a node closes when its bound is within
    /// this absolute gap of the incumbent.
    pub absolute_gap: f64,
    /// Native `relOptimalityGapTolerance`, relative to the incumbent.
    pub relative_gap: f64,
    /// Every integer solution has an integer objective, so node bounds are
    /// rounded up before pruning (native `integerValuedBound`). Validated.
    pub integer_objective: bool,
    /// Worker threads evaluating a batch of open nodes concurrently. One keeps
    /// the sequential native search; more change the trajectory, never proofs.
    pub threads: usize,
    /// Reprice paths discovered in other tree nodes/trials in this recipient.
    /// Disabled by default; the retained native inactive-search gate is zero.
    pub share_inactive_columns: bool,
    pub node_limit: usize,
    pub time_limit: Duration,
    pub tolerance: f64,
    pub enumerated_mip: Option<crate::master::solvers::mip::MipOptions>,
    /// Branch on how often each time window of a network vertex is selected,
    /// on this disposable resource (multi-time-windows builds, fresh problems
    /// only). Arcs into multi-window vertices are copied once per window;
    /// reported routes use the input arc IDs.
    pub time_window_selection: Option<usize>,
    /// Caller-validated integer solution (for example a heuristic one) and
    /// its full objective, including any recourse paid by master variables.
    /// The objective is the initial incumbent; the routes are its witness.
    pub initial_solution: Option<(f64, Vec<FixedRoute>)>,
}
/// Native `RestrictedMasterIpHeuristic` controls.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct RestrictedMasterMip {
    /// Run at every `frequency`-th selected node; zero runs at the root only.
    pub frequency: usize,
    /// Seconds per MIP solve.
    pub time_limit: f64,
    pub node_limit: usize,
}
impl Default for RestrictedMasterMip {
    fn default() -> Self {
        Self {
            frequency: 0,
            time_limit: 10.,
            node_limit: 100_000,
        }
    }
}
impl RestrictedMasterMip {
    pub(super) fn due(&self, depth: usize, selected: usize) -> bool {
        if self.frequency == 0 {
            depth == 0
        } else {
            selected.saturating_sub(1) % self.frequency == 0
        }
    }
}

impl ModelTreeOptions {
    /// Whether a node bound closes it against the incumbent: integer-valued
    /// objectives round the bound up first, and the gap is the largest of
    /// the tree tolerance and the absolute and relative gaps.
    pub(super) fn closes(&self, bound: f64, incumbent: f64) -> bool {
        let bound = if self.integer_objective && bound.is_finite() {
            (bound - (1e-6 + 1e-10 * bound.abs())).ceil()
        } else {
            bound
        };
        let gap = if incumbent.is_finite() {
            self.tolerance
                .max(self.absolute_gap)
                .max(self.relative_gap * incumbent.abs())
        } else {
            self.tolerance
        };
        bound >= incumbent - gap
    }
}

impl Default for ModelTreeOptions {
    fn default() -> Self {
        Self {
            branching_registry: super::BranchingRegistry::default(),
            node: Options::default(),
            tree_search: TreeSearchPolicy::default(),
            strategy2_bound_rounding: IntegerBoundRounding::default(),
            open_node_limit: DEFAULT_OPEN_NODE_LIMIT,
            scoring_gap_tolerance: 1e-6,
            diving: None,
            greedy: false,
            primal_heuristics: vec![],
            master_heuristic: None,
            fixed_solution_separators: vec![],
            local_search: None,
            restricted_master_mip: None,
            absolute_gap: 1e-6,
            relative_gap: 1e-9,
            integer_objective: false,
            threads: 1,
            share_inactive_columns: false,
            node_limit: 10000,
            time_limit: Duration::from_secs(120),
            tolerance: 1e-6,
            enumerated_mip: None,
            time_window_selection: None,
            initial_solution: None,
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
    /// Dispatches containing two or more concurrently evaluated fresh nodes.
    /// Prepared results, singleton dispatches and heuristic solves are excluded.
    pub parallel_batches: usize,
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
    /// Column-generation solves inside diving heuristics (not tree nodes).
    pub diving_nodes: usize,
    pub greedy_nodes: usize,
    /// Restricted-master integer heuristic solves.
    pub restricted_mip_calls: usize,
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
    /// Bound used by the tree to prune and report an open node. A fork can
    /// drop incumbent-justified warm bounds; the tree floor keeps
    /// min(parent bound, incumbent), valid for every solution of the child.
    pub(super) fn tree_bound(&self) -> f64 {
        self.warm.bound.max(self.state.tree_bound_floor)
    }
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
            Some(witness) => {
                let mut values = witness.2.clone();
                validate_routes(&self.state, &witness.1, &mut values)?
                    .map(|objective| (objective, witness.1.clone(), values))
            }
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
    replay_rounded_witness(state, master, metadata, tolerance)
}

fn replay_rounded_witness<const D: usize>(
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
    Ok(validate_routes(state, &routes, &mut master_values)?
        .map(|objective| (objective, routes, master_values)))
}

pub(super) fn validate_routes<const D: usize>(
    state: &State<D>,
    routes: &[FixedRoute],
    master_values: &mut [(i32, f64)],
) -> Result<Option<f64>> {
    if synchronization::witness(state, routes)?.is_none() {
        return Ok(None);
    }
    let Some((mut rows, mut objective, counts)) = bounds::exact_contributions(state, routes)?
    else {
        return Ok(None);
    };
    // Theta variables only reflect the optimality cuts separated so far. An
    // integer route set must pay its exact recourse: raise each anchor's
    // Theta to at least its routes' exact excess, so the cut rows hold and the
    // claimed objective bounds the true cost.
    if let Some(theta) = separation::recourse_theta(state, routes)? {
        for (id, excess) in theta {
            match master_values.iter_mut().find(|(v, _)| *v == id) {
                Some((_, value)) => *value = value.max(excess),
                None => return Ok(None),
            }
        }
    }
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
    // Rebuild lookup tables after caller edits or a change of problem.
    options.invalidate_branching_cache();
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
        let mut options = options;
        problem.validate()?;
        let component_positions: HashMap<_, HashMap<_, _>> = problem
            .networks
            .iter()
            .map(|n| {
                (
                    n.id,
                    n.graph
                        .arcs
                        .iter()
                        .enumerate()
                        .map(|(i, a)| (a.id, i))
                        .collect(),
                )
            })
            .collect();
        let windows = time_windows::split(&mut problem, resource)?;
        for class in &mut options.branching_registry.component_classes {
            if let Some(w) = windows.iter().find(|w| w.network == class.network) {
                let positions = &component_positions[&class.network];
                for coordinate in &mut class.coordinates {
                    if coordinate.arc_values.len() != positions.len() {
                        return Err(invalid("component coordinate mapping"));
                    }
                    let original = coordinate.arc_values.clone();
                    for copy in w.split.arcs.iter().filter(|a| a.arc != a.original) {
                        coordinate
                            .arc_values
                            .push(original[positions[&copy.original]]);
                    }
                }
            }
        }
        for generator in &mut options.branching_registry.component_generators {
            if let Some(w) = windows.iter().find(|w| w.network == generator.network) {
                let positions = &component_positions[&generator.network];
                let copies: Vec<_> = w
                    .split
                    .arcs
                    .iter()
                    .filter(|a| a.arc != a.original)
                    .map(|a| (positions.len(), positions[&a.original]))
                    .enumerate()
                    .map(|(i, (base, original))| (base + i, original))
                    .collect();
                generator.remap_arcs(&copies)?;
            }
        }
        for variable in &mut options.branching_registry.mapped_variables {
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
    options.branching_registry.branch_priorities.validate()?;
    if options
        .master_heuristic
        .as_ref()
        .is_some_and(|h| h.name.trim().is_empty())
    {
        return Err(invalid("master heuristic name must be nonempty"));
    }
    if options
        .primal_heuristics
        .iter()
        .any(|h| h.name.trim().is_empty())
    {
        return Err(invalid("primal heuristic name must be nonempty"));
    }
    match &start {
        Start::Fresh(problem, _) => super::path_counts::validate(
            &options.branching_registry.paths_per_network,
            problem
                .networks
                .iter()
                .map(|network| (network.id, &network.graph)),
        )?,
        Start::Snapshot(node) => super::path_counts::validate(
            &options.branching_registry.paths_per_network,
            node.state
                .oracles
                .iter()
                .map(|oracle| (oracle.network.id, oracle.network.graph.as_ref())),
        )?,
    }
    if let Some(diving) = &options.diving {
        diving.validate()?;
    }
    if let Some(local_search) = &options.local_search {
        local_search.validate()?;
    }
    if !options.absolute_gap.is_finite()
        || options.absolute_gap < 0.
        || !options.relative_gap.is_finite()
        || options.relative_gap < 0.
        || options.threads == 0
    {
        return Err(invalid(
            "tree gaps must be finite and nonnegative and threads positive",
        ));
    }
    if let Some(mip) = &options.restricted_master_mip {
        if !cfg!(feature = "highs") {
            return Err(invalid("restricted master MIP requires the highs feature"));
        }
        if !mip.time_limit.is_finite() || mip.time_limit < 0. {
            return Err(invalid("restricted master MIP time limit"));
        }
    }
    if options.integer_objective {
        integer_objective::validate(&start, &options)?;
    }
    options.node.gap_mip = options.enumerated_mip.clone();
    if options.enumerated_mip.is_some() {
        if !cfg!(feature = "highs") {
            return Err(invalid("enumerated MIP requires the highs feature"));
        }
        if !options.node.gap_enumeration && options.node.enumeration.is_none() {
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
    if let Some(packing) = &mut options.branching_registry.packing {
        match &start {
            Start::Fresh(problem, _) => packing.prepare(
                problem.networks.iter().map(|n| (n.id, &n.graph)),
                &options.branching_registry.mapped_variables,
            )?,
            Start::Snapshot(node) => packing.prepare(
                node.state
                    .oracles
                    .iter()
                    .map(|o| (o.network.id, o.network.graph.as_ref())),
                &options.branching_registry.mapped_variables,
            )?,
        }
        options
            .branching_registry
            .register(super::BranchingRegistration::Policy(
                NetworkBranchingPolicy::Phased,
            ))?;
    }
    for clusters in &mut options.branching_registry.clusters {
        match &start {
            Start::Fresh(problem, _) => {
                clusters.prepare(problem.networks.iter().map(|n| (n.id, &n.graph)))?
            }
            Start::Snapshot(node) => clusters.prepare(
                node.state
                    .oracles
                    .iter()
                    .map(|o| (o.network.id, o.network.graph.as_ref())),
            )?,
        }
    }
    super::family_registration::validate_selection(&options)?;
    if let Some(ryan_foster) = &mut options.branching_registry.ryan_foster {
        match &start {
            Start::Fresh(problem, _) => ryan_foster.prepare(
                problem.networks.iter().map(|n| (n.id, &n.graph)),
                options.branching_registry.packing.as_ref(),
                options.node.separation.rank_one_incidence.as_ref(),
            )?,
            Start::Snapshot(node) => ryan_foster.prepare(
                node.state
                    .oracles
                    .iter()
                    .map(|o| (o.network.id, o.network.graph.as_ref())),
                options.branching_registry.packing.as_ref(),
                options
                    .node
                    .separation
                    .rank_one_incidence
                    .as_ref()
                    .or_else(|| {
                        node.state
                            .cuts
                            .as_ref()
                            .and_then(|cuts| cuts.incidence_config())
                    }),
            )?,
        }
    }
    match &start {
        Start::Fresh(problem, variables) => model_mapping::validate::<D>(
            &options.branching_registry.mapped_variables,
            problem.networks.iter().map(|n| (n.id, &n.graph)),
            variables,
        )?,
        Start::Snapshot(node) => model_mapping::validate::<D>(
            &options.branching_registry.mapped_variables,
            node.state
                .oracles
                .iter()
                .map(|o| (o.network.id, o.network.graph.as_ref())),
            &node.state.master_variables,
        )?,
    }
    if options
        .fixed_solution_separators
        .iter()
        .any(|s| s.name.trim().is_empty())
    {
        return Err(invalid("fixed-solution separator name must be nonempty"));
    }
    match &start {
        Start::Fresh(problem, _) => {
            super::family_registration::validate_components(&options, |id| {
                problem
                    .networks
                    .iter()
                    .find(|n| n.id == id)
                    .map(|n| &n.graph)
            })?
        }
        Start::Snapshot(node) => super::family_registration::validate_components(&options, |id| {
            node.state
                .oracles
                .iter()
                .find(|o| o.network.id == id)
                .map(|o| o.network.graph.as_ref())
        })?,
    }
    options.compile_branching_registry()?;
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
        Start::Snapshot(node) => {
            let selected = match options.node.cg.bound_mode {
                cg::BoundMode::Numerical => node.warm.numerical_bound,
                cg::BoundMode::Certified => node.warm.certified_bound,
            };
            if node.warm.bound_mode == options.node.cg.bound_mode {
                selected.max(node.state.tree_bound_floor)
            } else {
                selected
            }
        }
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
        parallel_batches: 0,
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
        diving_nodes: 0,
        greedy_nodes: 0,
        restricted_mip_calls: 0,
    };
    if let Some((cost, routes)) = &options.initial_solution {
        if !cost.is_finite() {
            return Err(invalid("initial solution cost"));
        }
        result.incumbent = *cost;
        // Rebuild each supplied route's visits from its arcs, so reported
        // routes never carry caller node lists; unknown arcs are rejected.
        // Original-model feasibility is validated after the root solve.
        result.routes = routes
            .iter()
            .map(|fixed| {
                let graph = match &start {
                    Start::Fresh(problem, _) => problem
                        .networks
                        .iter()
                        .find(|n| n.id == fixed.route.network)
                        .map(|n| &n.graph),
                    Start::Snapshot(node) => node
                        .state
                        .oracles
                        .iter()
                        .find(|o| o.network.id == fixed.route.network)
                        .map(|o| &*o.network.graph),
                }
                .ok_or_else(|| invalid("initial solution network"))?;
                let mut nodes = vec![graph.source as i32];
                for id in &fixed.route.route.arcs {
                    let arc = graph
                        .arcs
                        .iter()
                        .find(|a| a.id == *id)
                        .ok_or_else(|| invalid("initial solution arc"))?;
                    nodes.push(arc.head as i32);
                }
                let mut replayed = fixed.clone();
                replayed.route.route.nodes = nodes;
                Ok(replayed)
            })
            .collect::<Result<Vec<_>>>()?;
    }
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
        // Caller-supplied initial routes are validated against the prepared
        // root before their objective authorizes any pricing exclusions.
        node.fixing_upper_bound = None;
        node.cg.time_limit = node.cg.time_limit.min(remaining);
        node
    };
    let initial_routes = result.routes.clone();
    let configured = node_options(deadline.saturating_duration_since(Instant::now()));
    let root = match start {
        Start::Fresh(problem, variables) => {
            if options.greedy {
                let (master, state, configured) =
                    solve::prepare_with_seed_routes(problem, variables, configured, &[], &[], &[])?;
                greedy::run(
                    &master,
                    &state,
                    &configured,
                    None,
                    &options,
                    &mut result,
                    deadline,
                )?;
                solve::run(master, state, configured)?
            } else {
                solve_with_master_variables(problem, variables, configured)?
            }
        }
        Start::Snapshot(mut node) => {
            node.state.mapped_lineage = 0;
            if options.greedy {
                let prepared = configured.clone();
                greedy::run(
                    &node.master,
                    &node.state,
                    &prepared,
                    Some(&node.warm),
                    &options,
                    &mut result,
                    deadline,
                )?;
            }
            node.solve(configured)?
        }
    };
    if let Some((supplied_cost, _)) = &options.initial_solution {
        let mut zero_master: Vec<_> = root
            .snapshot
            .state
            .master_variables
            .iter()
            .map(|variable| (variable.id, 0.))
            .collect();
        let validated = validate_routes(&root.snapshot.state, &initial_routes, &mut zero_master)?
            .ok_or_else(|| {
            invalid("initial tree solution is not feasible in the original model")
        })?;
        if validated > supplied_cost + 1e-6 + 1e-10 * validated.abs().max(supplied_cost.abs()) {
            return Err(invalid(
                "initial tree solution objective understates its replayed cost",
            ));
        }
        if validated <= result.incumbent {
            result.incumbent = validated;
            result.routes = initial_routes;
            result.master_values = zero_master;
        }
    }
    let base_depth = root.snapshot.state.node_depth;
    let mut subtree_history = options
        .branching_registry()
        .strong_branching
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
    mapped_history.incidence_history = root.snapshot.incidence_separator_history();
    let mut selected_nodes = 0;
    let workers = parallel::workers(options.threads)?;
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
            workers: workers.as_ref(),
            local_search_rng: crate::rng::GnuRand::new(
                options
                    .local_search
                    .as_ref()
                    .map_or(1, |config| config.seed),
            ),
        },
        &mut pending,
        Some(root),
    )?;
    match stop {
        crate::master::tree::Stop::Exhausted => result.status = TreeStatus::Optimal,
        crate::master::tree::Stop::OverflowOnly => result.status = TreeStatus::Incomplete,
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
        let keep =
            !node.proven_infeasible() && !options.closes(node.node.tree_bound(), result.incumbent);
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
        .map(|node| node.node.tree_bound())
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
