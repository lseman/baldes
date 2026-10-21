//! Shared master with independently priced networks and immutable node snapshots.
//!
//! Arc entries refer to shared rows. Each network also receives its own
//! multiplicity rows. Additive rows and assignment branches survive node forks.
//! Complete all-network enumeration can feed an optional HiGHS integer copy.
use super::{clp::ClpMaster, column_generation as cg, root::Row, RowSense};
use crate::{graph::Window, incomplete, invalid, pricing::Solver, Graph, Result, Route};
use std::{
    cell::RefCell,
    collections::{BTreeMap, HashMap, HashSet},
    rc::Rc,
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
mod resource_rows;
pub use presence_rows::PresenceCoefficients;
pub use resource_groups::ResourceGroupMember;
pub use resource_rows::ResourceKnapsackCoefficients;
mod preprocessing;
mod preprocessing_flow;
mod route_bounds;
pub use preprocessing::{NetworkActivityBounds, PreprocessingOptions, PreprocessingReport};
mod master_variables;
mod variables;
pub use enumeration::{
    CutoffCostBound, CutoffEnumeration, EnumeratedMipOutput, EnumerationOptions, EnumerationReport,
};
pub use master_variables::MasterVariable;
pub use nonlinear::{RankOneCoefficients, RegisteredCutCoefficients};
mod tree;
pub use lifecycle::{Assignment, FixedRoute, NodeSnapshot, RowCoefficients};
pub use tree::{
    solve_assignment_tree, solve_assignment_tree_with_master_variables, AssignmentTreeOptions,
    AssignmentTreeOutput,
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
    network: Network<D>,
    total_multiplicity: Window,
    solver: Solver<D>,
    index: HashMap<i32, usize>,
    route_entries: Vec<(usize, f64)>,
    calls: usize,
    nonlinear_rows: Vec<(usize, usize)>,
    knapsack_rows: Vec<(usize, usize)>,
    presence_rows: Vec<(usize, usize)>,
    resource_groups: Vec<(usize, super::branching::AccumulatedResourceGroupBranch)>,
    proved_zero_arcs: Vec<bool>,
    proved_multiplicity_upper: Option<f64>,
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
    oracles: Vec<Oracle<D>>,
    shared_offset: usize,
    routes: Vec<NetworkRoute>,
    tags: HashMap<(i32, String), i32>,
    correction: Option<f64>,
    gradient: Option<Vec<f64>>,
    rhs: Vec<f64>,
    deadline: Instant,
    shared_rows: Vec<Row>,
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
impl<const D: usize> cg::Controller<ClpMaster> for Coordinator<D> {
    fn request_enumeration_checkpoint(&mut self, _master: &ClpMaster) -> Result<bool> {
        let requested = self.1 && self.0.borrow().universe.is_none();
        // A failed attempt must resume ordinary pricing, not request another
        // checkpoint at the same point. A later solve may retry with new limits.
        self.1 = false;
        Ok(requested)
    }
    fn column_upper_bound(&self, tag: i32) -> Option<f64> {
        let state = self.0.borrow();
        if tag >= 0 {
            enumeration::route_upper(&state, tag as usize).filter(|u| u.is_finite())
        } else if let Some((index, sign)) = master_variables::decode(tag) {
            state
                .master_variables
                .get(index)
                .map(|v| master_variables::upper(v, sign))
                .filter(|u| u.is_finite())
        } else {
            None
        }
    }
    fn certified_pricing_bound(&self, master: &ClpMaster, duals: &[f64]) -> Result<Option<f64>> {
        Ok(Some(
            bounds::certificate(&self.0.borrow(), master, duals).unwrap_or(f64::NEG_INFINITY),
        ))
    }
    fn pricing_bound(&self, master: &ClpMaster, duals: &[f64]) -> Result<Option<f64>> {
        let state = self.0.borrow();
        Ok(Some(match state.correction {
            Some(correction) => {
                duals
                    .iter()
                    .zip(&state.rhs)
                    .fold(master.objective_offset, |value, (dual, rhs)| {
                        value + dual * rhs
                    })
                    + correction
            }
            None => f64::NEG_INFINITY,
        }))
    }
    fn pricing_subgradient(&self, _rhs: &[f64]) -> Option<Vec<f64>> {
        self.0.borrow().gradient.clone()
    }
}

fn entries<const D: usize>(
    oracle: &Oracle<D>,
    route: &Route,
    offset: usize,
) -> Result<Vec<(usize, f64)>> {
    let mut values = BTreeMap::new();
    for &(row, value) in &oracle.route_entries {
        *values.entry(row).or_insert(0.) += value;
    }
    for &(row, value) in &oracle.network.route_entries {
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

impl<const D: usize> State<D> {
    fn price(&mut self, phase: usize, duals: &[f64]) -> Result<cg::Answer> {
        if self.universe.is_some() {
            return enumeration::price(self, duals);
        }
        let mut columns = vec![];
        let mut minimum = f64::INFINITY;
        let mut exact = true;
        let mut correction = 0.;
        let mut gradient = self.rhs.clone();
        for oracle in &mut self.oracles {
            if oracle.multiplicity_upper() == 0. {
                continue;
            }
            let remaining = self.deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                return Err(incomplete("multiple-network pricing time limit"));
            }
            oracle.solver.controls.time_limit = remaining;
            let costs: Vec<_> = oracle
                .network
                .graph
                .arcs
                .iter()
                .zip(&oracle.network.arc_entries)
                .map(|(arc, coefficients)| {
                    coefficients.iter().fold(arc.cost, |cost, &(row, value)| {
                        cost - duals[self.shared_offset + row] * value
                    })
                })
                .collect();
            for &(cut, row) in &oracle.nonlinear_rows {
                oracle.solver.cuts.cuts[cut].dual = duals[self.shared_offset + row];
            }
            for &(cut, row) in &oracle.knapsack_rows {
                oracle.solver.cuts.knapsack_cuts[cut].dual = duals[self.shared_offset + row];
            }
            if !oracle.presence_rows.is_empty() {
                let mut inputs = oracle.solver.presence_rows().to_vec();
                for &(input, row) in &oracle.presence_rows {
                    inputs[input].dual = duals[self.shared_offset + row];
                }
                oracle.solver.set_presence_rows(inputs)?;
            }
            oracle.solver.cache_variable_costs(&costs)?;
            let offset = oracle
                .route_entries
                .iter()
                .map(|&(row, value)| -duals[row] * value)
                .sum::<f64>()
                - oracle
                    .network
                    .route_entries
                    .iter()
                    .map(|&(row, value)| duals[self.shared_offset + row] * value)
                    .sum::<f64>();
            oracle.calls += 1;
            let result = oracle
                .solver
                .price_functor_phase(phase, oracle.solver.controls.max_columns)?;
            if oracle.solver.phases[phase].exact() && !result.exact {
                return Err(incomplete("network exact pricing did not finish"));
            }
            exact &= result.exact;
            let reduced = result.minimum_reduced_cost + offset;
            if !reduced.is_finite() {
                return Err(incomplete("network minimum is not finite"));
            }
            if result.columns.is_empty() && (reduced < 0. || oracle.network.multiplicity.lower > 0.)
            {
                return Err(incomplete("required network pricing witness unavailable"));
            }
            minimum = minimum.min(reduced);
            let quantity = if reduced > 0. {
                oracle.network.multiplicity.lower
            } else {
                let upper = oracle.multiplicity_upper();
                if reduced == 0. && !upper.is_finite() {
                    oracle.network.multiplicity.lower
                } else {
                    upper
                }
            };
            correction += quantity * reduced;
            let mut best: Option<(f64, Vec<(usize, f64)>)> = None;
            for mut route in result.columns {
                let coefficients = entries(oracle, &route, self.shared_offset)?;
                let mut ids = route.arcs.clone();
                ids.sort_unstable();
                let cost = ids
                    .iter()
                    .map(|id| oracle.network.graph.arcs[oracle.index[id]].cost)
                    .sum::<f64>();
                let reduced_cost = coefficients
                    .iter()
                    .fold(cost, |value, &(row, coefficient)| {
                        value - duals[row] * coefficient
                    });
                if !cost.is_finite() || !reduced_cost.is_finite() {
                    return Err(invalid("network route cost overflow"));
                }
                if best
                    .as_ref()
                    .map_or(true, |(value, _)| reduced_cost < *value)
                {
                    best = Some((reduced_cost, coefficients.clone()));
                }
                route.original_cost = cost;
                route.cost += offset;
                let identity = route.identity();
                let key = (oracle.network.id, identity.clone());
                let tag = if let Some(&tag) = self.tags.get(&key) {
                    tag
                } else {
                    let tag = i32::try_from(self.routes.len())
                        .map_err(|_| invalid("network route tag overflow"))?;
                    self.tags.insert(key, tag);
                    self.routes.push(NetworkRoute {
                        network: oracle.network.id,
                        route,
                    });
                    tag
                };
                columns.push(cg::Column {
                    cost,
                    reduced_cost,
                    entries: coefficients,
                    acceptable: true,
                    tag,
                    identity: format!("{}:{identity}", oracle.network.id),
                });
            }
            if let Some((_, coefficients)) = best {
                for (row, value) in coefficients {
                    gradient[row] -= quantity * value;
                }
            }
        }
        if minimum == f64::INFINITY {
            minimum = 0.;
        }
        // Heuristic batches also supply the phase estimate used by smoothing.
        // The CG driver publishes it as a bound only for an exact batch.
        if let Some((static_correction, static_gradient)) =
            master_variables::correction(self, duals)
        {
            correction += static_correction;
            for (row, value) in static_gradient {
                gradient[row] -= value;
            }
            if correction.is_finite() && gradient.iter().all(|g| g.is_finite()) {
                self.correction = Some(correction);
                self.gradient = Some(gradient);
            } else {
                self.correction = None;
                self.gradient = None;
            }
        } else {
            // A negative reduced cost on an unbounded static component leaves
            // this dual point's estimate unknown. Still return priced routes.
            self.correction = None;
            self.gradient = None;
        }
        Ok(cg::Answer {
            columns,
            minimum,
            exact,
            bound_minimum: None,
        })
    }
}

fn validate_options(options: &Options) -> Result<()> {
    if options.cg.warm_start.is_some() || !options.cg.known.is_empty() {
        return Err(invalid("network nodes own their column registry"));
    }
    if !options.cg.artificial_cost.is_finite()
        || options.cg.artificial_cost <= 0.
        || !options.cg.tolerance.is_finite()
        || options.cg.tolerance < 0.
        || !options.cg.gap_absolute.is_finite()
        || options.cg.gap_absolute < 0.
        || !options.cg.gap_relative.is_finite()
        || options.cg.gap_relative < 0.
    {
        return Err(invalid("network root artificial cost or tolerance"));
    }
    Ok(())
}

/// Solve a fresh shared master. Each oracle retains its own pricing state.
pub fn solve<const D: usize>(problem: Problem<D>, options: Options) -> Result<Output<D>> {
    solve_with_master_variables(problem, vec![], options)
}

/// Solve the LP relaxation for a snapshot containing only original master
/// variables. Trees or MIP enforce declared integrality. The empty pricing family
/// is complete, so preprocessing and optional MIP need no enumeration step.
/// Rust callers needing a particular snapshot dimension can equivalently use
/// `solve_with_master_variables` with an empty `Problem::<D>::networks`.
pub fn solve_master(
    rows: Vec<Row>,
    variables: Vec<MasterVariable>,
    options: Options,
) -> Result<Output> {
    solve_with_master_variables(
        Problem {
            rows,
            networks: vec![],
        },
        variables,
        options,
    )
}

pub fn solve_with_master_variables<const D: usize>(
    problem: Problem<D>,
    variables: Vec<MasterVariable>,
    options: Options,
) -> Result<Output<D>> {
    solve_with_fixed_routes(problem, variables, options, &[], &[])
}

fn solve_with_fixed_routes<const D: usize>(
    problem: Problem<D>,
    variables: Vec<MasterVariable>,
    options: Options,
    fixed: &[FixedRoute],
    nonlinear: &[(usize, nonlinear::RowKernels)],
) -> Result<Output<D>> {
    solve_with_seed_routes(problem, variables, options, fixed, nonlinear, &[])
}

fn solve_with_seed_routes<const D: usize>(
    mut problem: Problem<D>,
    mut variables: Vec<MasterVariable>,
    mut options: Options,
    fixed: &[FixedRoute],
    nonlinear: &[(usize, nonlinear::RowKernels)],
    seeds: &[NetworkRoute],
) -> Result<Output<D>> {
    problem.validate()?;
    master_variables::install(&mut problem, &mut variables)?;
    if options.cg.warm_start.is_some() || !options.cg.known.is_empty() {
        return Err(invalid("shared root requires a fresh column registry"));
    }
    validate_options(&options)?;
    problem.validate()?;
    let deadline = Instant::now()
        .checked_add(options.cg.time_limit)
        .ok_or_else(|| invalid("network time limit"))?;
    let mut master = ClpMaster::new()?;
    master.set_column_default_upper(f64::MAX)?;
    let mut network_rows = vec![vec![]; problem.networks.len()];
    for (index, network) in problem.networks.iter().enumerate() {
        if network.multiplicity.upper.is_finite() {
            let row = master.add_row(RowSense::LessEqual, network.multiplicity.upper, &[])?;
            network_rows[index].push((row, 1.));
        }
        if network.multiplicity.lower > 0. {
            let row = master.add_row(RowSense::GreaterEqual, network.multiplicity.lower, &[])?;
            network_rows[index].push((row, 1.));
        }
    }
    let shared_offset = master.rhs.len();
    let shared_rows = problem.rows.clone();
    for row in problem.rows {
        master.add_row(row.sense, row.rhs, &[])?;
    }
    let mut oracles = vec![];
    for (network, route_entries) in problem.networks.into_iter().zip(network_rows) {
        let mut solver = Solver::new(network.graph.vertices.len(), network.graph.resources.len())?;
        solver.graph = network.graph.clone();
        solver.graph.graph_id = network.id;
        for (index, arc) in solver.graph.arcs.iter_mut().enumerate() {
            arc.cost = 0.;
            arc.variables = vec![(index, 1.)];
        }
        let index = network
            .graph
            .arcs
            .iter()
            .enumerate()
            .map(|(i, arc)| (arc.id, i))
            .collect();
        let total_multiplicity = network.multiplicity;
        oracles.push(Oracle {
            total_multiplicity,
            network,
            solver,
            index,
            route_entries,
            calls: 0,
            nonlinear_rows: vec![],
            knapsack_rows: vec![],
            presence_rows: vec![],
            resource_groups: vec![],
            proved_zero_arcs: vec![],
            proved_multiplicity_upper: None,
        });
    }
    let master_only = oracles.is_empty();
    let mut state = State {
        oracles,
        shared_offset,
        routes: vec![],
        tags: HashMap::new(),
        correction: None,
        gradient: None,
        rhs: master.rhs.clone(),
        deadline,
        shared_rows,
        fixed_routes: vec![],
        // With no pricing subproblems every original variable is already
        // loaded. Completeness requires no synthetic graph or enumeration.
        universe: master_only.then(Vec::new),
        enumeration_report: EnumerationReport {
            complete: master_only,
            ..Default::default()
        },
        preprocessing: PreprocessingReport::default(),
        master_variables: variables,
        route_upper_bounds: HashMap::new(),
    };
    for (shared, kernels) in nonlinear {
        kernels.install(&mut state, *shared)?;
    }
    lifecycle::install_fixed_routes(&mut master, &mut state, fixed)?;
    options.cg.known = master_variables::load(&mut master, &state)?;
    lifecycle::install_seed_routes(&mut master, &mut state, &mut options.cg.known, seeds)?;
    run(master, state, options)
}

fn run<const D: usize>(
    mut master: ClpMaster,
    mut state: State<D>,
    mut options: Options,
) -> Result<Output<D>> {
    options.cg.time_limit = options
        .cg
        .time_limit
        .min(state.deadline.saturating_duration_since(Instant::now()));
    for oracle in &mut state.oracles {
        oracle.calls = 0;
        for (name, value) in &options.pricing_options {
            oracle.solver.set_option(name, *value)?;
        }
        let equality_sink_is_free =
            oracle
                .network
                .graph
                .resources
                .iter()
                .enumerate()
                .any(|(r, kind)| {
                    let sink = oracle.network.graph.vertices[oracle.network.graph.sink].bounds[r];
                    *kind == crate::graph::ResourceKind::Equality && sink.lower != sink.upper
                });
        if resource_rows::requires_forward(oracle) || equality_sink_is_free {
            // Backward equality needs a fixed terminal value. Resource-row
            // complements also cannot reproduce general waiting/initial usage.
            // Apply these model contracts after caller direction options.
            oracle
                .solver
                .set_option("BALDESuseBidirectionalSearch", 0.)?;
        }
    }
    let loaded: Vec<_> = if let Some(warm) = &options.cg.warm_start {
        warm.columns
            .iter()
            .enumerate()
            .filter_map(|(c, m)| {
                m.as_ref()
                    .filter(|m| m.tag >= 0)
                    .map(|m| (c, m.tag as usize))
            })
            .collect()
    } else {
        options
            .cg
            .known
            .iter()
            .enumerate()
            .filter(|(_, m)| m.tag >= 0)
            .map(|(c, m)| (c, m.tag as usize))
            .collect()
    };
    let previous_bounds: Vec<_> = state.master_variables.iter().map(|v| v.bounds).collect();
    preprocessing::apply(&mut master, &mut state, &loaded, &options.preprocessing)?;
    master_variables::publish::bounds(
        &mut master,
        &mut state,
        &previous_bounds,
        &mut options.cg.known,
        options.cg.warm_start.as_mut(),
    )?;
    let schedule = state.oracles.first().map_or_else(
        || {
            vec![crate::pricing::phases::Phase {
                check_ng: true,
                jump_arcs: true,
                labels_per_bucket: i64::MAX,
            }]
        },
        |oracle| oracle.solver.phases.clone(),
    );
    let state = Rc::new(RefCell::new(state));
    let mut controller = Coordinator(state.clone(), options.enumeration.is_some());
    let gap = (options.cg.gap_absolute, options.cg.gap_relative);
    let started = Instant::now();
    let time_limit = options.cg.time_limit;
    let max_iterations = options.cg.max_iterations;
    let mut previous: Option<cg::Output> = None;
    let mut result = loop {
        let spent = previous.as_ref().map_or(0, |out| out.iterations);
        options.cg.max_iterations = max_iterations - spent;
        options.cg.time_limit = time_limit.saturating_sub(started.elapsed()).min(
            state
                .borrow()
                .deadline
                .saturating_duration_since(Instant::now()),
        );
        let mut current = cg::optimize(
            &mut master,
            &schedule,
            |phase, duals| state.borrow_mut().price(phase, duals),
            &mut controller,
            options.cg.clone(),
        )?;
        if let Some(mut prior) = previous.take() {
            if current.enumeration_failure.is_none() {
                current.enumeration_failure = prior.enumeration_failure.take();
            }
            for row in &mut current.history {
                row.iteration += prior.iterations;
                row.pricing_calls += prior.pricing_calls;
            }
            current.iterations += prior.iterations;
            current.pricing_calls += prior.pricing_calls;
            current.penalty_updates += prior.penalty_updates;
            current.cut_rounds += prior.cut_rounds;
            prior.history.append(&mut current.history);
            current.history = prior.history;
            prior.inserted_tags.append(&mut current.inserted_tags);
            current.inserted_tags = prior.inserted_tags;
        }
        if !current.enumeration_requested {
            break current;
        }
        current.enumeration_requested = false;
        if current.iterations >= max_iterations {
            break current;
        }
        if started.elapsed() >= time_limit || Instant::now() >= state.borrow().deadline {
            current.status = cg::Status::TimeLimit;
            break current;
        }
        options.cg.warm_start = Some(current.warm_start.clone());
        // transfer stages every model/registry change on private copies. On a
        // backend failure the checkpoint is still valid for ordinary pricing.
        if let Err(error) = enumeration::transfer(
            &mut master,
            &mut state.borrow_mut(),
            &mut options.cg,
            options.enumeration.as_ref().unwrap(),
        ) {
            state.borrow_mut().enumeration_report.failure = Some(error.to_string());
        }
        if state.borrow().universe.is_some() {
            // Complete-domain deductions need the newly registered routes.
            // They previously ran after the eager transfer; preserve that
            // ordering here before any resumed pricing or bound publication.
            let mut state = state.borrow_mut();
            let loaded: Vec<_> = options
                .cg
                .warm_start
                .as_ref()
                .unwrap()
                .columns
                .iter()
                .enumerate()
                .filter_map(|(column, metadata)| {
                    metadata
                        .as_ref()
                        .filter(|metadata| metadata.tag >= 0)
                        .map(|metadata| (column, metadata.tag as usize))
                })
                .collect();
            let previous_bounds: Vec<_> = state.master_variables.iter().map(|v| v.bounds).collect();
            preprocessing::apply(&mut master, &mut state, &loaded, &options.preprocessing)?;
            master_variables::publish::bounds(
                &mut master,
                &mut state,
                &previous_bounds,
                &mut options.cg.known,
                options.cg.warm_start.as_mut(),
            )?;
        }
        previous = Some(current);
    };
    // A finite artificial penalty cannot prove original-model infeasibility.
    if result.status == cg::Status::Infeasible {
        result.status = cg::Status::Stalled;
        result.lp = master.objective;
    }
    if result.status == cg::Status::Optimal
        && (!result.bound.is_finite()
            || !cg::gap_smaller_than_tolerance(result.bound, result.lp, gap.0, gap.1))
    {
        result.status = cg::Status::Stalled;
    }
    drop(controller);
    let state = Rc::try_unwrap(state)
        .map_err(|_| invalid("network state still borrowed"))?
        .into_inner();
    let mut values = vec![0.; state.routes.len()];
    let mut master_values: Vec<_> = state.master_variables.iter().map(|v| (v.id, 0.)).collect();
    for (&tag, &value) in result.column_tags.iter().zip(master.values().iter()) {
        if tag >= 0 {
            values[tag as usize] += value;
        } else if let Some((index, sign)) = master_variables::decode(tag) {
            master_values[index].1 += sign * value;
        }
    }
    Ok(Output {
        master_values,
        routes: state.routes.clone(),
        fixed_routes: state.fixed_routes.clone(),
        values,
        duals: master.duals(),
        oracle_calls: state
            .oracles
            .iter()
            .map(|oracle| (oracle.network.id, oracle.calls))
            .collect(),
        snapshot: NodeSnapshot {
            master,
            warm: result.warm_start.clone(),
            state,
        },
        master: result,
    })
}

#[cfg(test)]
mod multiplicity_tests;
