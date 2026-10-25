//! Root master integrating the native Rust pricing engine, Clp and
//! limited-memory rank-one cuts. Cut handling follows the C++ reference
//! driver `_solve_root_clp` of `python/src/openrscp/vrptw.py`.
use super::branching::BranchConstraint;
use super::{
    cg::{self, Answer, Column, Controller, LpMaster, Options, Output},
    clp::ClpMaster,
    RowSense,
};
use crate::cuts::capacity::{CapacityCut, CapacityParameters, CapacitySeparator, TimeWindows};
use crate::cuts::knapsack::{
    KnapsackColumn, KnapsackParameters, KnapsackRow, ResourceKnapsackSeparator,
};
use crate::cuts::kpath::{KPathGraph, KPathParameters, KPathSeparator, NativeKPathSeparator};
use crate::cuts::FractionalColumn;
pub use crate::master::solvers::clp::{ClpOptions, ClpScaling, MasterSolver};
use crate::pricing::Solver;
use crate::{invalid, Graph, Result, Route};
use std::cell::RefCell;
use std::collections::HashSet;
use std::rc::Rc;

mod certificate;
mod controller;
mod domain;
mod restoration;
pub use domain::{CanonicalCoverRelaxation, ConservativeCoverRelaxation, CoverDomainFailure};
mod cover_rows;
pub use cover_rows::{
    CoverRowAudit, CoverRowFailure, CoverRowLimits, CoverRowRange, WeightedCoverRowAudit,
};
mod incumbent;
pub(crate) use incumbent::{exact_route_feasibility, exact_route_feasible, ExactSum};
mod integer;
pub use integer::{EnumeratedMipOutput, EnumerationMipPolicy};
mod options;
mod partial;
mod preprocessing;
mod problem;
pub use cg::BoundMode;
use controller::RootController;
use options::RootKPathSeparator;
pub use options::{
    CapacityCutConfig, CutOptions, KPathCutConfig, KnapsackCutConfig, NativeKPathCutConfig,
    RootOptions,
};
pub use partial::{inherit_fixed_routes, FixedRoute};
pub use problem::{RootProblem, Row};

#[derive(Debug)]
pub struct RootOutput<const D: usize = { crate::DEFAULT_RESOURCE_DIMENSIONS }> {
    /// Routes removed into the fixed part, separate from residual LP columns.
    pub fixed_routes: Vec<(Route, f64)>,
    pub fixed_cost: f64,
    pub preprocessing: crate::master::preprocessing::Report,
    pub snapshot: Option<Rc<NodeSnapshot<D>>>,
    pub simplex_iterations: usize,
    pub ipm_iterations: usize,
    pub inherited_cuts: usize,
    pub master: Output,
    pub routes: Vec<Route>,
    pub values: Vec<f64>,
    pub duals: Vec<f64>,
    pub incumbent: f64,
    pub pricing_seconds: f64,
    pub solver_seconds: f64,
    pub cuts: usize,
    pub lp_before_cuts: f64,
    pub fixing_calls: usize,
    pub enumerated_routes: usize,
}
/// Immutable solved-node state, shared by siblings and copied only when a
/// descendant or a trial is evaluated. No sibling mutates this model.
pub struct NodeSnapshot<const D: usize = { crate::DEFAULT_RESOURCE_DIMENSIONS }> {
    master: ClpMaster,
    problem: RootProblem<D>,
    cut: CutState,
    pricing: Solver<D>,
    incumbent: f64,
    routes: Vec<Route>,
    warm: cg::WarmStart,
}

#[derive(Clone)]
struct CutState {
    options: CutOptions,
    /// Master row of each registered cut (None once removed).
    rows: Vec<Option<usize>>,
    /// Cut memory when its coefficients were last written.
    memories: Vec<Vec<bool>>,
    /// Sorted arc IDs of the routes of the last successful enumeration.
    enumerated: HashSet<Vec<i32>>,
    lp_before_cuts: f64,
    separation_lp: f64,
    rounds: usize,
    tailing: usize,
    capacity_separator: Option<CapacitySeparator>,
    capacity_priority: f64,
    capacity_cuts: Vec<CapacityCut>,
    capacity_rows: Vec<Option<usize>>,
    capacity_arc_entries: Vec<Vec<(usize, f64)>>,
    kpath_separator: Option<RootKPathSeparator>,
    kpath_indices: Vec<usize>,
    knapsack_separator: Option<ResourceKnapsackSeparator>,
    knapsack_rows: Vec<Option<usize>>,
    /// Current separator priority level: separators at or above it run
    /// every round, lower ones only while nothing at a higher one succeeds.
    level: f64,
    /// Dynamic NG priority, 0 when disabled.
    relaxation_priority: f64,
    /// Master columns to delete after a relaxation strengthening.
    removed_columns: Vec<usize>,
}
/// Cut rows of one separation round, added to the master together.
struct PendingRows {
    base: usize,
    rows: Vec<(RowSense, f64, Vec<(usize, f64)>)>,
}
impl PendingRows {
    /// Queue a row; returns the master index it will have.
    fn push(&mut self, sense: RowSense, rhs: f64, entries: Vec<(usize, f64)>) -> usize {
        self.rows.push((sense, rhs, entries));
        self.base + self.rows.len() - 1
    }
}
/// Root separator priorities: capacity cuts, then rank-one cuts.
const CAPACITY_PRIORITY: f64 = 3.;
const RANK_ONE_PRIORITY: f64 = 1.;
#[derive(Clone, Copy, PartialEq)]
enum Separator {
    Capacity,
    KPath,
    Knapsack,
    RankOne,
    Relaxation,
}

struct State<const D: usize = { crate::DEFAULT_RESOURCE_DIMENSIONS }> {
    solver: Solver<D>,
    routes: Vec<Route>,
    /// Arc sequence -> tag of each entry in `routes`.
    route_tags: std::collections::HashMap<String, i32>,
    incumbent: f64,
    enumerate: bool,
    rounded: bool,
    previous_fixing_gap: f64,
    deadline: Option<std::time::Instant>,
    pricing_seconds: f64,
    cut: CutState,
    fixing_calls: usize,
    enumerated_routes: usize,
}

/// Stable tag of the first route with this ordered arc path and resource schedule.
fn register<const D: usize>(state: &mut State<D>, route: &Route) -> Result<i32> {
    let identity = route.identity();
    if let Some(&tag) = state.route_tags.get(&identity) {
        return Ok(tag);
    }
    let id = i32::try_from(state.routes.len()).map_err(|_| invalid("route tag overflow"))?;
    state.routes.push(route.clone());
    state.route_tags.insert(identity, id);
    Ok(id)
}

mod cut_state;
mod snapshot;
mod state;
mod transfer;

mod solve;
pub(crate) use solve::inherit_with_history;
pub use solve::{inherit, inherit_branches, solve};

use solve::solve_seeded;
