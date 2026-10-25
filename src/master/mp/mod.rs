//! Shared master with independently priced networks and immutable node snapshots.
//!
//! Arc entries refer to shared rows. Each network also receives its own
//! multiplicity rows. Additive rows and assignment branches survive node forks.
//! Complete all-network enumeration can feed an optional HiGHS integer copy.
mod pricing;
mod scheduling;
pub use scheduling::NetworkPricingSchedule;
mod solve;
use super::{cg, clp::ClpMaster, root::Row, RowSense};
use crate::{graph::Window, incomplete, invalid, pricing::Solver, Graph, Result, Route};
use std::{
    cell::RefCell,
    collections::{BTreeMap, HashMap, HashSet},
    rc::Rc,
    sync::Arc,
    time::Instant,
};

mod bounds;
mod enumeration;
pub mod format;
pub mod imported;
mod lifecycle;
mod nonlinear;
mod presence_rows;
mod resource_groups;
mod synchronization;
pub use synchronization::{SynchronizedService, SynchronizedServiceMember};
mod resource_rows;
pub use presence_rows::PresenceCoefficients;
pub use resource_groups::{ResourceGroupCandidate, ResourceGroupMember};
pub use resource_rows::ResourceKnapsackCoefficients;
mod preprocessing;
mod preprocessing_flow;
mod route_bounds;
pub use preprocessing::{NetworkActivityBounds, PreprocessingOptions, PreprocessingReport};
mod master_variables;
mod model_mapping;
mod model_strong;
mod variables;
pub use enumeration::{
    CutoffCostBound, CutoffEnumeration, EnumeratedMipOutput, EnumerationOptions, EnumerationReport,
};
pub use master_variables::MasterVariable;
pub use model_mapping::{MappedBranchVariable, MappedNetworkVariable};
pub use model_strong::{MappedBranchEvaluation, MappedEvaluationKind};
pub use nonlinear::{RankOneCoefficients, RegisteredCutCoefficients};
mod tree;
pub use lifecycle::{Assignment, FixedRoute, NodeSnapshot, RowCoefficients};
pub use tree::{
    solve_assignment_tree, solve_assignment_tree_with_master_variables, AssignmentTreeOptions,
    AssignmentTreeOutput, NetworkBranchingPolicy,
};

#[derive(Clone, Debug)]
pub struct Network<const D: usize = { crate::DEFAULT_RESOURCE_DIMENSIONS }> {
    pub id: i32,
    pub graph: Graph<D>,
    pub multiplicity: Window,
    /// Shared-row coefficients, aligned with `graph.arcs`.
    pub arc_entries: Vec<Vec<(usize, f64)>>,
    /// Shared-row coefficients charged once per route.
    pub route_entries: Vec<(usize, f64)>,
}

#[derive(Clone, Debug)]
pub struct Problem<const D: usize = { crate::DEFAULT_RESOURCE_DIMENSIONS }> {
    pub rows: Vec<Row>,
    pub networks: Vec<Network<D>>,
}

impl<const D: usize> Problem<D> {
    /// Validate every network and shared coefficient, even before a zero-budget run.
    pub fn validate(&self) -> Result<()> {
        let mut ids = HashSet::new();
        for network in &self.networks {
            network.graph.validate()?;
            if network.id < 0
                || !ids.insert(network.id)
                || network.arc_entries.len() != network.graph.arcs.len()
                || !network.multiplicity.lower.is_finite()
                || network.multiplicity.upper.is_nan()
                || network.multiplicity.upper == f64::NEG_INFINITY
                || network.multiplicity.lower < 0.
                || network.multiplicity.upper < network.multiplicity.lower
                || network
                    .arc_entries
                    .iter()
                    .flatten()
                    .chain(&network.route_entries)
                    .any(|&(row, value)| row >= self.rows.len() || !value.is_finite())
            {
                return Err(invalid("network identity, multiplicity or coefficients"));
            }
        }
        if self.rows.iter().any(|row| !row.rhs.is_finite()) {
            return Err(invalid("network master row RHS"));
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct Options {
    pub network_pricing_schedule: NetworkPricingSchedule,
    /// Budget for unpromising networks in native queue scheduling modes.
    pub network_unpromising_limit: usize,
    /// Native promising-status visit budget (default one million).
    pub network_promising_limit: usize,
    /// Optional integer-time, simultaneous service by distinct route copies.
    pub synchronized_services: Vec<SynchronizedService>,
    pub cg: cg::Options,
    pub pricing_options: Vec<(String, f64)>,
    pub enumeration: Option<EnumerationOptions>,
    pub preprocessing: PreprocessingOptions,
}
impl Default for Options {
    fn default() -> Self {
        let mut cg = cg::Options::default();
        cg.artificial_cost = 2.76;
        Self {
            network_pricing_schedule: NetworkPricingSchedule::default(),
            network_unpromising_limit: 1,
            network_promising_limit: 1_000_000,
            synchronized_services: vec![],
            cg,
            pricing_options: vec![],
            enumeration: None,
            preprocessing: PreprocessingOptions::default(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct NetworkRoute {
    pub network: i32,
    pub route: Route,
}

#[derive(Debug)]
pub struct Output<const D: usize = { crate::DEFAULT_RESOURCE_DIMENSIONS }> {
    pub master: cg::Output,
    pub snapshot: NodeSnapshot<D>,
    /// Fixed contribution, separate from the remaining master quantities.
    pub fixed_routes: Vec<FixedRoute>,
    pub routes: Vec<NetworkRoute>,
    /// Restricted-master quantity for each registered route.
    pub values: Vec<f64>,
    pub duals: Vec<f64>,
    pub master_values: Vec<(i32, f64)>,
    /// Actual network pricing calls, rather than aggregate pricing batches.
    pub oracle_calls: Vec<(i32, usize)>,
}

struct Oracle<const D: usize = { crate::DEFAULT_RESOURCE_DIMENSIONS }> {
    network: OracleNetwork<D>,
    total_multiplicity: Window,
    solver: Solver<D>,
    index: Arc<HashMap<i32, usize>>,
    costs: Vec<f64>,
    route_entries: Vec<(usize, f64)>,
    calls: usize,
    nonlinear_rows: Vec<(usize, usize)>,
    knapsack_rows: Vec<(usize, usize)>,
    presence_rows: Vec<(usize, usize)>,
    resource_groups: Vec<(usize, super::branching::AccumulatedResourceGroupBranch)>,
    synchronized_windows: Vec<(usize, usize, Window)>,
    proved_zero_arcs: Vec<bool>,
    proved_multiplicity_upper: Option<f64>,
}
/// Immutable input topology and row definitions are shared across node forks.
/// Row edits copy only the coefficient arrays; multiplicity is node-local.
#[derive(Clone)]
struct OracleNetwork<const D: usize> {
    id: i32,
    graph: Arc<Graph<D>>,
    multiplicity: Window,
    arc_entries: Arc<Vec<Vec<(usize, f64)>>>,
    route_entries: Arc<Vec<(usize, f64)>>,
}
impl<const D: usize> From<Network<D>> for OracleNetwork<D> {
    fn from(network: Network<D>) -> Self {
        Self {
            id: network.id,
            graph: Arc::new(network.graph),
            multiplicity: network.multiplicity,
            arc_entries: Arc::new(network.arc_entries),
            route_entries: Arc::new(network.route_entries),
        }
    }
}
impl<const D: usize> Oracle<D> {
    fn multiplicity_upper(&self) -> f64 {
        self.network
            .multiplicity
            .upper
            .min(self.proved_multiplicity_upper.unwrap_or(f64::INFINITY))
    }
}
struct State<const D: usize = { crate::DEFAULT_RESOURCE_DIMENSIONS }> {
    mapped_subtree_size: f64,
    node_depth: usize,
    scheduler: scheduling::Scheduler,
    synchronized_services: Vec<synchronization::Service>,
    oracles: Vec<Oracle<D>>,
    shared_offset: usize,
    routes: Vec<Arc<NetworkRoute>>,
    tags: HashMap<(i32, String), i32>,
    correction: Option<f64>,
    gradient: Option<Vec<f64>>,
    rhs: Vec<f64>,
    deadline: Instant,
    shared_rows: Vec<Row>,
    curvature_rows: Vec<cg::CurvatureRow>,
    fixed_routes: Vec<FixedRoute>,
    universe: Option<Vec<usize>>,
    enumeration_report: EnumerationReport,
    preprocessing: PreprocessingReport,
    master_variables: Vec<MasterVariable>,
    route_upper_bounds: HashMap<usize, f64>,
}
struct Coordinator<const D: usize = { crate::DEFAULT_RESOURCE_DIMENSIONS }>(
    Rc<RefCell<State<D>>>,
    bool,
);

fn entries<const D: usize>(
    oracle: &Oracle<D>,
    route: &Route,
    offset: usize,
) -> Result<Vec<(usize, f64)>> {
    let mut values = BTreeMap::new();
    for &(row, value) in &oracle.route_entries {
        *values.entry(row).or_insert(0.) += value;
    }
    for &(row, value) in oracle.network.route_entries.iter() {
        *values.entry(offset + row).or_insert(0.) += value;
    }
    for id in &route.arcs {
        let index = *oracle
            .index
            .get(id)
            .ok_or_else(|| invalid("unknown network route arc"))?;
        for &(row, value) in &oracle.network.arc_entries[index] {
            *values.entry(offset + row).or_insert(0.) += value;
        }
    }
    if values.values().any(|value: &f64| !value.is_finite()) {
        return Err(invalid("network route coefficient overflow"));
    }
    for (row, coefficient) in nonlinear::entries(oracle, route)? {
        *values.entry(offset + row).or_insert(0.) += coefficient;
    }
    Ok(values
        .into_iter()
        .filter(|(_, value)| *value != 0.)
        .collect())
}

#[cfg(test)]
#[path = "../../../tests/unit/master/mp/multiplicity_tests.rs"]
mod multiplicity_tests;

#[cfg(test)]
#[path = "../../../tests/unit/master/mp/sharing_tests.rs"]
mod sharing_tests;

use solve::{run, solve_with_fixed_routes, solve_with_seed_routes, validate_options};
pub use solve::{solve, solve_master, solve_with_master_variables};
