//! Best-bound branch-and-price for a routing set-partitioning master.
//!
//! Vehicle counts, packing-set assignments and directed arc-flow rows are
//! priced through `RootProblem::arc_entries`. Ryan–Foster rows additionally
//! install native binary resources in labeling and filter enumeration.
//! Each child forks the parent's solved Clp model (with cuts intact) and
//! carries the cut registry forward; feasible elementary columns are shared
//! between nodes. Numerical bounds use exact pricing with solver tolerances;
//! certified mode requires independent bound validation. Restricted-master
//! heuristic objectives never supply node lower bounds.
//! Phased strong branching and diving
//! use isolated node evaluations via fork-and-carry.
use super::{
    cg::Status,
    root::{self, RootOptions, RootProblem},
    RowSense,
};
use crate::{invalid, Result, Route};
use std::collections::{BTreeMap, HashSet};
use std::rc::Rc;
use std::time::{Duration, Instant};
pub mod candidates;
pub mod clusters;
pub mod history;
pub use history::{BranchEvaluationHistory, BranchHistoryOptions, BranchHistoryPolicy};
mod diving;
#[cfg(feature = "highs")]
mod integer;
mod objective_lattice;
pub use super::branching::{BranchConstraint, BranchingOptions};
pub mod strong;
pub use diving::DivingOptions;
pub use strong::{StrongBranchingOptions, StrongPhase};

#[derive(Clone, Debug)]
pub struct TreeOptions {
    /// Concurrent node evaluations. One preserves sequential best-bound search.
    pub threads: usize,
    pub root: RootOptions,
    pub branching: BranchingOptions,
    pub node_limit: usize,
    pub time_limit: Duration,
    /// Work budget of the binary restricted-master set-partitioning search.
    pub mip_node_limit: usize,
    pub mip_time_limit: Duration,
    /// Optional HiGHS MIP over complete enumeration (requires `highs`).
    /// Independent of the restricted-column heuristic and disabled by default.
    pub enumerated_mip: Option<super::solvers::mip::MipOptions>,
    /// Optional DeLuxing filtering before complete enumerated MIP solves.
    /// Requires `enumerated_mip`; used when a finite cutoff is available.
    pub deluxing: Option<super::deluxing::DeluxingOptions>,
    pub absolute_gap: f64,
    pub relative_gap: f64,
    pub strong_branching: Option<StrongBranchingOptions>,
    pub branch_history: Option<BranchHistoryOptions>,
    pub diving: Option<DivingOptions>,
}
impl Default for TreeOptions {
    fn default() -> Self {
        Self {
            threads: 1,
            root: RootOptions::default(),
            branching: BranchingOptions::default(),
            node_limit: 10000,
            time_limit: Duration::from_secs(120),
            mip_node_limit: 10000,
            mip_time_limit: Duration::from_millis(100),
            enumerated_mip: None,
            deluxing: None,
            absolute_gap: 1e-6,
            relative_gap: 1e-9,
            strong_branching: Some(StrongBranchingOptions::default()),
            branch_history: Some(BranchHistoryOptions::default()),
            diving: Some(DivingOptions::default()),
        }
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TreeStatus {
    Optimal,
    Infeasible,
    TimeLimit,
    NodeLimit,
    Incomplete,
}
#[derive(Debug)]
pub struct TreeOutput {
    /// Defines the guarantee of `bound` and any gap-based termination.
    pub bound_mode: root::BoundMode,
    pub status: TreeStatus,
    pub incumbent: f64,
    pub bound: f64,
    pub routes: Vec<Route>,
    pub nodes: usize,
    /// Number of batches with two or more concurrently evaluated nodes.
    pub parallel_batches: usize,
    pub open_nodes: usize,
    pub pricing_calls: usize,
    /// Fixing/inspection calls across normal nodes, strong trials and dives.
    pub fixing_calls: usize,
    /// Sum of retained route counts reported by those evaluations.
    pub enumerated_routes: usize,
    pub mip_nodes: usize,
    pub enumerated_mip_calls: usize,
    pub enumerated_mip_nodes: usize,
    pub strong_branching_trials: usize,
    pub branch_history: BranchEvaluationHistory,
    pub diving_nodes: usize,
    pub seconds: f64,
}
impl TreeOutput {
    pub fn to_json(&self, name: &str) -> String {
        fn number(x: f64) -> String {
            if x.is_finite() {
                x.to_string()
            } else {
                "null".into()
            }
        }
        let routes: Vec<_> = self
            .routes
            .iter()
            .map(|r| {
                format!(
                    "{{\"nodes\":{:?},\"arcs\":{:?},\"cost\":{}}}",
                    r.nodes,
                    r.arcs,
                    number(r.original_cost)
                )
            })
            .collect();
        format!("{{\"instance\":{:?},\"status\":\"{:?}\",\"incumbent\":{},\"bound\":{},\"bound_mode\":\"{}\",\"numerical_bound\":{},\"certified_bound\":{},\"nodes\":{},\"parallel_batches\":{},\"open_nodes\":{},\"pricing_calls\":{},\"fixing_calls\":{},\"enumerated_routes\":{},\"mip_nodes\":{},\"enumerated_mip_calls\":{},\"enumerated_mip_nodes\":{},\"strong_branching_trials\":{},\"diving_nodes\":{},\"branch_history_ranked\":{},\"branch_evaluations\":[{}],\"seconds\":{},\"routes\":[{}]}}\n",
            name, self.status, number(self.incumbent), number(self.bound), self.bound_mode.as_str(),
            if self.bound_mode == root::BoundMode::Numerical { number(self.bound) } else { "null".into() },
            if self.bound_mode == root::BoundMode::Certified { number(self.bound) } else { "null".into() }, self.nodes,
            self.parallel_batches, self.open_nodes, self.pricing_calls, self.fixing_calls, self.enumerated_routes, self.mip_nodes, self.enumerated_mip_calls, self.enumerated_mip_nodes, self.strong_branching_trials, self.diving_nodes, self.branch_history.ranked_candidates, self.branch_history.to_json(), self.seconds, routes.join(","))
    }
}
type Branch = BranchConstraint;
struct Node<const D: usize = { crate::DEFAULT_RESOURCE_DIMENSIONS }> {
    lineage: usize,
    subtree: strong::SubtreeEstimate,
    observation: Option<(candidates::BranchCandidate, usize, f64)>,
    branches: Vec<Branch>,
    bound: f64,
    parent: Option<Rc<root::NodeSnapshot<D>>>,
}
struct Pool {
    routes: Vec<Route>,
    customers: Vec<Vec<usize>>,
    seen: HashSet<Vec<i32>>,
}
impl Pool {
    fn add<const D: usize>(&mut self, problem: &RootProblem<D>, arcs: &[i32]) {
        if self.seen.contains(arcs) {
            return;
        }
        if let Some(route) = feasible_route(problem, arcs) {
            let customers = route.nodes[1..route.nodes.len() - 1]
                .iter()
                .map(|&v| v as usize)
                .collect();
            self.seen.insert(arcs.to_vec());
            self.routes.push(route);
            self.customers.push(customers);
        }
    }
}
/// Reconstruct and check every incumbent route, including resources and
/// repeated customers. Never trust a supplied cost, node list or LP tag alone.
fn feasible_route<const D: usize>(problem: &RootProblem<D>, ids: &[i32]) -> Option<Route> {
    problem.elementary_route(ids)
}
fn closed(bound: f64, incumbent: f64, options: &TreeOptions) -> bool {
    bound == f64::INFINITY
        || (incumbent.is_finite()
            && bound
                >= incumbent
                    - options.absolute_gap
                    - options.relative_gap * incumbent.abs().max(bound.abs()))
}
fn branch_problem<const D: usize>(
    problem: &RootProblem<D>,
    branches: &[Branch],
) -> Result<RootProblem<D>> {
    let mut child = problem.clone();
    for branch in branches {
        child.branch(branch.clone())?;
    }
    Ok(child)
}
fn budget_error(error: &crate::core::error::OrError) -> bool {
    error.message.contains("time limit") || error.message.contains("label limit")
}
fn evaluate<const D: usize>(
    parent: Option<&Rc<root::NodeSnapshot<D>>>,
    previous: Option<&root::NodeSnapshot<D>>,
    problem: &RootProblem<D>,
    branches: &[Branch],
    _pool: &Pool,
    options: &TreeOptions,
    incumbent: f64,
    deadline: Instant,
    max_iterations: usize,
    floor: usize,
    capture_state: bool,
) -> Result<root::RootOutput<D>> {
    let mut controls = options.root.clone();
    controls.capture_state = capture_state;
    controls.node_depth = branches.len();
    controls.pricing_phase_floor = floor;
    controls.cg.max_iterations = max_iterations;
    controls.cg.time_limit = deadline.saturating_duration_since(Instant::now());
    controls.cg.multiplicity = problem.customers.len() as f64;
    // Explicit seeds enter the root. Child masters inherit their own snapshot;
    // the global integer pool does not preload routes from other evaluations.
    controls.initial_routes = if parent.is_some() {
        vec![]
    } else {
        options.root.initial_routes.clone()
    };
    controls.incumbent = incumbent;
    controls.enumerate = options.root.enumerate && floor == 0;
    // Bound trial pricing by its own remaining time as well as the CG limit.
    let remaining = controls.cg.time_limit.as_secs_f64().max(1e-6);
    if let Some((_, value)) = controls
        .pricing_options
        .iter_mut()
        .find(|(name, _)| name == "BALDEShardTimeThresholdInPricing")
    {
        if *value <= 0. || *value > remaining {
            *value = remaining;
        }
    } else {
        controls
            .pricing_options
            .push(("BALDEShardTimeThresholdInPricing".into(), remaining));
    }
    if let Some(parent) = parent {
        let depth = parent.branch_depth();
        if depth > branches.len() {
            return Err(invalid("parent branch depth exceeds child branches"));
        }
        root::inherit_with_history(parent, &branches[depth..], previous, controls)
    } else {
        root::solve(branch_problem(problem, branches)?, controls)
    }
}
/// Solve a standard routing set-partitioning problem. `root.initial_routes`
/// may seed incumbents; the cost-only `root.incumbent` is not used without a
/// feasible witness. Limits preserve the bound of every unfinished node.
#[cfg(test)]
#[path = "../../../tests/unit/master/tree/pool_tests.rs"]
mod pool_tests;

mod heuristic;
use heuristic::integer_heuristic;

mod parallel;
mod search;
pub use search::solve;
