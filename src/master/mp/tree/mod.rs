//! Shared-network tree over assignment, integer master and original activities.
//! Unsupported fractional domains remain open rather than claiming optimality.
use super::*;
mod branching;
pub(super) mod proposals;
mod time_windows;
use crate::master::{tree::TreeStatus, LpMaster};
use std::time::Duration;

const NATIVE_OPEN_NODE_LIMIT: usize = 1000;

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
    NativePhased,
}

impl NetworkBranchingPolicy {
    pub fn parse(value: &str) -> Result<Self> {
        match value {
            "resource-first" => Ok(Self::ResourceFirst),
            "multiplicity-first" => Ok(Self::MultiplicityFirst),
            "native-phased" => Ok(Self::NativePhased),
            _ => Err(invalid(
                "network branching must be resource-first, multiplicity-first or native-phased",
            )),
        }
    }
}

/// Assignment-tree frontier order, independent of which branch family is used.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Deserialize, serde::Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum NetworkTreeSearchPolicy {
    /// Native strategy 1: depth, then bound, then stable creation order.
    #[default]
    NativeDefault,
    /// Process the open node with the lowest inherited bound first.
    BestBound,
    /// Native strategy 2: inherited integer-rounded numerical dual bound,
    /// queued-child primal sentinels, depth, parent branch-and-price order.
    #[serde(rename = "native-strategy-2")]
    NativeStrategy2,
}

impl NetworkTreeSearchPolicy {
    pub fn parse(value: &str) -> Result<Self> {
        match value {
            "native-default" => Ok(Self::NativeDefault),
            "best-bound" => Ok(Self::BestBound),
            "native-strategy-2" => Ok(Self::NativeStrategy2),
            _ => Err(invalid(
                "network tree search must be native-default, best-bound or native-strategy-2",
            )),
        }
    }
}

/// Integer conversion used for native strategy-2 dual-bound ranking.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Deserialize, serde::Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum NativeIntegerBoundRounding {
    /// Native nonpositive control with a continuous-objective bound tag.
    #[default]
    Unrounded,
    /// Native nonpositive control with an integer-objective bound tag.
    GuardedTolerance,
    /// Native positive rounding control: ceil the numerical bound directly.
    ExactCeil,
}

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

#[derive(Clone, Debug)]
pub struct AssignmentTreeOptions {
    pub node: Options,
    pub branching: NetworkBranchingPolicy,
    pub tree_search: NetworkTreeSearchPolicy,
    /// Rounding mode for strategy-2's integer-valued dual-bound queue key.
    pub strategy2_bound_rounding: NativeIntegerBoundRounding,
    /// Number of nodes held in the native primary frontier before overflow.
    pub native_open_node_limit: usize,
    pub branch_priorities: NetworkBranchPriorities,
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
            tree_search: NetworkTreeSearchPolicy::default(),
            strategy2_bound_rounding: NativeIntegerBoundRounding::default(),
            native_open_node_limit: NATIVE_OPEN_NODE_LIMIT,
            branch_priorities: NetworkBranchPriorities::default(),
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
    /// Nodes selected from the frontier, including nodes closed by bound before an LP solve.
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

struct QueuedNode<const D: usize> {
    node: NodeSnapshot<D>,
    serial: u64,
    branch_order: i32,
    parent_branch_order: i32,
    strategy2_dual_bound: f64,
}

#[derive(Clone, Copy)]
enum HeapOrder {
    NativePrimary,
    NativeSecondary,
    NativeStrategy2,
    BestBound,
}

fn native_queue_bound(value: f64) -> f64 {
    if value == f64::NEG_INFINITY {
        -1e12
    } else if value == f64::INFINITY {
        1e12
    } else {
        value
    }
}

fn native_bound_less(left: f64, right: f64) -> bool {
    let left = native_queue_bound(left);
    let right = native_queue_bound(right);
    let epsilon = 1e-6 + 1e-10 * left.abs().max(right.abs());
    left < right - epsilon
}

/// Strategy 2 uses the native node's secondary dual bound, whose rounding
/// depends on the declared objective contract.
/// BALDES keeps the underlying numerical LP bound separately from its
/// controller-certified bound, so the native queue key is derived from the
/// numerical channel and never from certification metadata.
fn native_strategy2_bound<const D: usize>(
    node: &NodeSnapshot<D>,
    rounding: NativeIntegerBoundRounding,
) -> f64 {
    native_strategy2_bound_values(node.warm.numerical_bound, node.warm.bound, rounding)
}

fn native_strategy2_bound_values(
    numerical_bound: f64,
    selected_bound: f64,
    rounding: NativeIntegerBoundRounding,
) -> f64 {
    let bound = if numerical_bound.is_finite() {
        numerical_bound
    } else {
        selected_bound
    };
    if !bound.is_finite() {
        return native_queue_bound(bound);
    }
    if bound.abs() >= 4_503_599_627_370_496. {
        return bound;
    }
    match rounding {
        NativeIntegerBoundRounding::Unrounded => bound,
        NativeIntegerBoundRounding::GuardedTolerance => {
            let epsilon = 1e-6 + 1e-10 * bound.abs();
            (bound - epsilon).ceil()
        }
        NativeIntegerBoundRounding::ExactCeil => bound.ceil(),
    }
}

#[derive(Clone, Copy)]
struct NativeStrategy2Key {
    dual_bound: f64,
    depth: usize,
    parent_branch_order: i32,
    serial: u64,
}

/// Native operator< reports lower queue priority first.
fn native_strategy2_less(left: NativeStrategy2Key, right: NativeStrategy2Key) -> bool {
    if native_bound_less(left.dual_bound, right.dual_bound) {
        return false;
    }
    if native_bound_less(right.dual_bound, left.dual_bound) {
        return true;
    }
    if left.depth != right.depth {
        return left.depth < right.depth;
    }
    if left.parent_branch_order != right.parent_branch_order {
        return left.parent_branch_order > right.parent_branch_order;
    }
    left.serial >= right.serial
}

struct NativeHeap<const D: usize> {
    nodes: Vec<QueuedNode<D>>,
    order: HeapOrder,
}

impl<const D: usize> NativeHeap<D> {
    fn new(order: HeapOrder) -> Self {
        Self {
            nodes: Vec::new(),
            order,
        }
    }

    fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Returns true when `left` has lower queue priority than `right`, exactly
    /// following the native pairwise bound predicate and short-circuit order.
    fn less(&self, left: &QueuedNode<D>, right: &QueuedNode<D>) -> bool {
        let left_depth = left.node.state.node_depth;
        let right_depth = right.node.state.node_depth;
        if matches!(self.order, HeapOrder::NativeStrategy2) {
            // A native child initializes both primal-bound tie fields to the
            // objective-sense sentinel and they stay equal until evaluation.
            return native_strategy2_less(
                NativeStrategy2Key {
                    dual_bound: left.strategy2_dual_bound,
                    depth: left_depth,
                    parent_branch_order: left.parent_branch_order,
                    serial: left.serial,
                },
                NativeStrategy2Key {
                    dual_bound: right.strategy2_dual_bound,
                    depth: right_depth,
                    parent_branch_order: right.parent_branch_order,
                    serial: right.serial,
                },
            );
        }
        if !matches!(self.order, HeapOrder::BestBound) && left_depth != right_depth {
            return left_depth < right_depth;
        }
        let left_bound = left.node.warm.bound;
        let right_bound = right.node.warm.bound;
        if native_bound_less(left_bound, right_bound) {
            return false;
        }
        if native_bound_less(right_bound, left_bound) {
            return true;
        }
        if matches!(self.order, HeapOrder::BestBound) && left_depth != right_depth {
            return left_depth < right_depth;
        }
        match self.order {
            HeapOrder::NativeSecondary => left.serial > right.serial,
            HeapOrder::NativePrimary | HeapOrder::NativeStrategy2 | HeapOrder::BestBound => {
                left.serial >= right.serial
            }
        }
    }

    fn push(&mut self, node: QueuedNode<D>) {
        self.nodes.push(node);
        let mut hole = self.nodes.len() - 1;
        while hole > 0 {
            let parent = (hole - 1) / 2;
            if !self.less(&self.nodes[parent], &self.nodes[hole]) {
                break;
            }
            self.nodes.swap(parent, hole);
            hole = parent;
        }
    }

    fn pop(&mut self) -> Option<QueuedNode<D>> {
        if self.nodes.is_empty() {
            return None;
        }
        let result = self.nodes.swap_remove(0);
        let len = self.nodes.len();
        if len == 0 {
            return Some(result);
        }
        let mut hole = 0;
        while 2 * hole + 2 < len {
            let right = 2 * hole + 2;
            let left = right - 1;
            let child = if self.less(&self.nodes[right], &self.nodes[left]) {
                left
            } else {
                right
            };
            self.nodes.swap(hole, child);
            hole = child;
        }
        if 2 * hole + 1 < len {
            let child = 2 * hole + 1;
            self.nodes.swap(hole, child);
            hole = child;
        }
        while hole > 0 {
            let parent = (hole - 1) / 2;
            if !self.less(&self.nodes[parent], &self.nodes[hole]) {
                break;
            }
            self.nodes.swap(parent, hole);
            hole = parent;
        }
        Some(result)
    }
}

enum TreeFrontier<const D: usize> {
    BestBound {
        nodes: NativeHeap<D>,
        overflow: NativeHeap<D>,
        next_serial: u64,
        next_branch_order: i32,
        open_node_limit: usize,
        strategy2_bound_rounding: NativeIntegerBoundRounding,
    },
    Native {
        primary: NativeHeap<D>,
        overflow: NativeHeap<D>,
        next_serial: u64,
        next_branch_order: i32,
        open_node_limit: usize,
        strategy2_bound_rounding: NativeIntegerBoundRounding,
    },
}

impl<const D: usize> TreeFrontier<D> {
    fn new(
        policy: NetworkTreeSearchPolicy,
        open_node_limit: usize,
        strategy2_bound_rounding: NativeIntegerBoundRounding,
    ) -> Self {
        match policy {
            NetworkTreeSearchPolicy::BestBound => Self::BestBound {
                nodes: NativeHeap::new(HeapOrder::BestBound),
                overflow: NativeHeap::new(HeapOrder::NativeSecondary),
                next_serial: 0,
                next_branch_order: 1,
                open_node_limit,
                strategy2_bound_rounding,
            },
            NetworkTreeSearchPolicy::NativeStrategy2 => Self::Native {
                primary: NativeHeap::new(HeapOrder::NativeStrategy2),
                overflow: NativeHeap::new(HeapOrder::NativeSecondary),
                next_serial: 0,
                next_branch_order: 1,
                open_node_limit,
                strategy2_bound_rounding,
            },
            NetworkTreeSearchPolicy::NativeDefault => Self::Native {
                primary: NativeHeap::new(HeapOrder::NativePrimary),
                overflow: NativeHeap::new(HeapOrder::NativeSecondary),
                next_serial: 0,
                next_branch_order: 1,
                open_node_limit,
                strategy2_bound_rounding,
            },
        }
    }

    fn is_empty(&self) -> bool {
        match self {
            Self::BestBound {
                nodes, overflow, ..
            } => nodes.is_empty() && overflow.is_empty(),
            Self::Native {
                primary, overflow, ..
            } => primary.is_empty() && overflow.is_empty(),
        }
    }

    fn push(&mut self, node: NodeSnapshot<D>, parent_branch_order: i32) -> Result<()> {
        let strategy2_bound_rounding = match self {
            Self::BestBound {
                strategy2_bound_rounding,
                ..
            }
            | Self::Native {
                strategy2_bound_rounding,
                ..
            } => *strategy2_bound_rounding,
        };
        let strategy2_dual_bound = native_strategy2_bound(&node, strategy2_bound_rounding);
        match self {
            Self::BestBound {
                nodes,
                overflow,
                next_serial,
                next_branch_order,
                open_node_limit,
                ..
            } => {
                let branch_order = *next_branch_order;
                *next_branch_order = next_branch_order
                    .checked_add(1)
                    .ok_or_else(|| invalid("native branch-and-price order overflow"))?;
                let queued = QueuedNode {
                    node,
                    serial: *next_serial,
                    branch_order,
                    parent_branch_order,
                    strategy2_dual_bound,
                };
                *next_serial = next_serial
                    .checked_add(1)
                    .ok_or_else(|| invalid("network tree creation serial overflow"))?;
                if nodes.nodes.len() < *open_node_limit {
                    nodes.push(queued);
                } else {
                    overflow.push(queued);
                }
            }
            Self::Native {
                primary,
                overflow,
                next_serial,
                next_branch_order,
                open_node_limit,
                ..
            } => {
                let branch_order = *next_branch_order;
                *next_branch_order = next_branch_order
                    .checked_add(1)
                    .ok_or_else(|| invalid("native branch-and-price order overflow"))?;
                let queued = QueuedNode {
                    node,
                    serial: *next_serial,
                    branch_order,
                    parent_branch_order,
                    strategy2_dual_bound,
                };
                *next_serial = next_serial
                    .checked_add(1)
                    .ok_or_else(|| invalid("native tree creation serial overflow"))?;
                if primary.nodes.len() < *open_node_limit {
                    primary.push(queued);
                } else {
                    overflow.push(queued);
                }
            }
        }
        Ok(())
    }

    fn pop(&mut self) -> Option<QueuedNode<D>> {
        match self {
            Self::BestBound {
                nodes, overflow, ..
            } => overflow.pop().or_else(|| nodes.pop()),
            Self::Native {
                primary, overflow, ..
            } => overflow.pop().or_else(|| primary.pop()),
        }
    }

    fn into_nodes(self) -> Vec<NodeSnapshot<D>> {
        match self {
            Self::BestBound {
                nodes, overflow, ..
            } => nodes
                .nodes
                .into_iter()
                .chain(overflow.nodes)
                .map(|queued| queued.node)
                .collect(),
            Self::Native {
                primary, overflow, ..
            } => primary
                .nodes
                .into_iter()
                .chain(overflow.nodes)
                .map(|queued| queued.node)
                .collect(),
        }
    }
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
    options.branch_priorities.validate()?;
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
            problem.networks.iter().map(|n| {
                (
                    n.id,
                    n.graph.arcs.iter().map(|a| a.id).collect(),
                    n.graph.original_max_arcs(),
                )
            }),
            variables,
        )?,
        Start::Snapshot(node) => model_mapping::validate::<D>(
            &options.mapped_variables,
            node.state.oracles.iter().map(|o| {
                (
                    o.network.id,
                    o.network.graph.arcs.iter().map(|a| a.id).collect(),
                    o.network.graph.original_max_arcs(),
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
    let mut current = Some(root);
    let mut pending = TreeFrontier::new(
        options.tree_search,
        options.native_open_node_limit,
        options.strategy2_bound_rounding,
    );
    let mut current_branch_order = -1;
    let mut unresolved = None;
    let mut mapped_history = model_strong::History::default();
    loop {
        let output = if let Some(output) = current.take() {
            result.processed_nodes += 1;
            output
        } else {
            if pending.is_empty() {
                result.status = TreeStatus::Optimal;
                break;
            }
            let limited_nodes = match options.tree_search {
                NetworkTreeSearchPolicy::BestBound => result.nodes,
                NetworkTreeSearchPolicy::NativeDefault
                | NetworkTreeSearchPolicy::NativeStrategy2 => result.processed_nodes,
            };
            if limited_nodes >= options.node_limit {
                result.status = TreeStatus::NodeLimit;
                break;
            }
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                result.status = TreeStatus::TimeLimit;
                break;
            }
            let queued = pending.pop().expect("nonempty frontier was checked");
            current_branch_order = queued.branch_order;
            result.processed_nodes += 1;
            if queued.node.warm.bound >= result.incumbent - options.tolerance {
                if let Some(history) = &mut subtree_history {
                    history.close(queued.node.state.mapped_lineage);
                }
                continue;
            }
            queued.node.solve(node_options(remaining))?
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
        let lineage = output.snapshot.state.mapped_lineage;
        mapped_history.completed_estimate = subtree_history
            .as_ref()
            .and_then(|h| h.mean(output.snapshot.state.node_depth.saturating_sub(base_depth)));
        // The whole original-flow relaxation proves LP infeasibility without
        // integer rounding. Complete-master integer deductions retain their
        // explicitly requested integer-domain policy.
        let preprocessing = output.snapshot.preprocessing_report();
        if preprocessing.propagation.infeasible
            && (!preprocessing.complete_domain || options.node.preprocessing.integer_domains)
        {
            if let Some(history) = &mut subtree_history {
                history.close(lineage);
            }
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
                                if let Some(history) = &mut subtree_history {
                                    history.close(lineage);
                                }
                                continue;
                            }
                            if matches!(
                                mip.mip.status,
                                MipStatus::Optimal | MipStatus::ObjectiveCutoff
                            ) && result.incumbent.is_finite()
                                && mip.mip.bound >= result.incumbent - options.tolerance
                            {
                                if let Some(history) = &mut subtree_history {
                                    history.close(lineage);
                                }
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
        if output.master.status == cg::Status::Infeasible
            && output
                .master
                .feasibility
                .iter()
                .any(|r| r.certified_infeasible)
        {
            if let Some(history) = &mut subtree_history {
                history.close(lineage);
            }
            continue;
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
            if let Some(history) = &mut subtree_history {
                history.close(lineage);
            }
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
                if let Some(history) = &mut subtree_history {
                    history.close(lineage);
                }
                continue;
            }
        }
        let count_children = if options.branching == NetworkBranchingPolicy::MultiplicityFirst {
            branching::multiplicity_children(&output, options.tolerance)?
        } else {
            None
        };
        let children = if options.branching == NetworkBranchingPolicy::NativePhased {
            let proposals = proposals::collect(&output, &options, windows)?;
            if let Some(mut children) = model_strong::choose_proposals(
                &output,
                &options,
                &mut mapped_history,
                result.incumbent,
                deadline,
                proposals,
            )? {
                children.reverse();
                Some(children)
            } else {
                branching::children(
                    &output,
                    options.tolerance,
                    NetworkBranchingPolicy::ResourceFirst,
                    windows,
                    &[],
                )?
            }
        } else if count_children.is_some() {
            count_children
        } else if let Some(mut children) = model_strong::choose(
            &output,
            &options,
            &mut mapped_history,
            result.incumbent,
            deadline,
        )? {
            if options.branching == NetworkBranchingPolicy::MultiplicityFirst {
                children.reverse();
            }
            Some(children)
        } else {
            branching::children(&output, options.tolerance, options.branching, windows, &[])?
        };
        if let Some(mut children) = children {
            if children.len() != 2 {
                return Err(invalid("network branch proposal must have two children"));
            }
            let lineages = subtree_history
                .as_mut()
                .map(|h| h.split(lineage))
                .unwrap_or([0; 2]);
            for (side, child) in children.iter_mut().enumerate() {
                child.state.node_depth = output.snapshot.state.node_depth.saturating_add(1);
                child.state.mapped_lineage = lineages[side];
            }
            for child in children {
                pending.push(child, current_branch_order)?;
            }
        } else {
            unresolved = Some(bound);
            break;
        }
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
        let keep = node.warm.bound < result.incumbent - options.tolerance;
        if !keep {
            if let Some(history) = &mut subtree_history {
                history.close(node.state.mapped_lineage);
            }
        }
        keep
    });
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

#[cfg(test)]
#[path = "../../../../tests/unit/master/mp/tree_search_tests.rs"]
mod tree_search_tests;
