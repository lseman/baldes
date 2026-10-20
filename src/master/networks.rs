//! Shared master with independently priced networks and immutable node snapshots.
//!
//! Arc entries refer to shared rows. Each network also receives its own
//! multiplicity rows. Additive rows and assignment branches survive node forks.
//! Nonlinear cut separation and enumeration/MIP transfer remain separate workflows.
use super::{clp::ClpMaster, column_generation as cg, root::Row, RowSense};
use crate::{graph::Window, incomplete, invalid, pricing::Solver, Graph, Result, Route};
use std::{
    cell::RefCell,
    collections::{BTreeMap, HashMap, HashSet},
    rc::Rc,
    time::Instant,
};

#[path = "network_bounds.rs"]
mod bounds;
mod lifecycle;
mod tree;
pub use lifecycle::{Assignment, FixedRoute, NodeSnapshot, RowCoefficients};
pub use tree::{solve_assignment_tree, AssignmentTreeOptions, AssignmentTreeOutput};

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

#[derive(Clone, Debug)]
pub struct Options {
    pub cg: cg::Options,
    pub pricing_options: Vec<(String, f64)>,
}
impl Default for Options {
    fn default() -> Self {
        let mut cg = cg::Options::default();
        cg.artificial_cost = 2.76;
        Self {
            cg,
            pricing_options: vec![],
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
}
struct Coordinator<const D: usize = { crate::DEFAULT_RESOURCE_DIMENSIONS }>(Rc<RefCell<State<D>>>);
impl<const D: usize> cg::Controller<ClpMaster> for Coordinator<D> {
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
    Ok(values
        .into_iter()
        .filter(|(_, value)| *value != 0.)
        .collect())
}

impl<const D: usize> State<D> {
    fn price(&mut self, phase: usize, duals: &[f64]) -> Result<cg::Answer> {
        let mut columns = vec![];
        let mut minimum = f64::INFINITY;
        let mut exact = true;
        let mut correction = 0.;
        let mut gradient = self.rhs.clone();
        for oracle in &mut self.oracles {
            if oracle.network.multiplicity.upper == 0. {
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
                oracle.network.multiplicity.upper
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
        if !correction.is_finite() {
            return Err(invalid("network bound overflow"));
        }
        // Heuristic batches also supply the phase estimate used by smoothing.
        // The CG driver publishes it as a bound only for an exact batch.
        self.correction = Some(correction);
        self.gradient = Some(gradient);
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
    if problem.networks.is_empty()
        || options.cg.warm_start.is_some()
        || !options.cg.known.is_empty()
    {
        return Err(invalid(
            "network root requires networks and a fresh column registry",
        ));
    }
    validate_options(&options)?;
    let mut ids = HashSet::new();
    for network in &problem.networks {
        network.graph.validate()?;
        if network.id < 0
            || !ids.insert(network.id)
            || network.arc_entries.len() != network.graph.arcs.len()
            || !network.multiplicity.lower.is_finite()
            || !network.multiplicity.upper.is_finite()
            || network.multiplicity.lower < 0.
            || network.multiplicity.upper < network.multiplicity.lower
            || network
                .arc_entries
                .iter()
                .flatten()
                .chain(&network.route_entries)
                .any(|&(row, value)| row >= problem.rows.len() || !value.is_finite())
        {
            return Err(invalid("network identity, multiplicity or coefficients"));
        }
    }
    if problem.rows.iter().any(|row| !row.rhs.is_finite()) {
        return Err(invalid("network master row RHS"));
    }
    let deadline = Instant::now()
        .checked_add(options.cg.time_limit)
        .ok_or_else(|| invalid("network time limit"))?;
    let mut master = ClpMaster::new()?;
    let mut network_rows = vec![vec![]; problem.networks.len()];
    for (index, network) in problem.networks.iter().enumerate() {
        let row = master.add_row(RowSense::LessEqual, network.multiplicity.upper, &[])?;
        network_rows[index].push((row, 1.));
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
        });
    }
    let state = State {
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
    };
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
    }
    let schedule = state.oracles[0].solver.phases.clone();
    let state = Rc::new(RefCell::new(state));
    let mut controller = Coordinator(state.clone());
    let gap = (options.cg.gap_absolute, options.cg.gap_relative);
    let mut result = cg::optimize(
        &mut master,
        &schedule,
        |phase, duals| state.borrow_mut().price(phase, duals),
        &mut controller,
        options.cg,
    )?;
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
    for (&tag, &value) in result.column_tags.iter().zip(master.values().iter()) {
        if tag >= 0 {
            values[tag as usize] += value;
        }
    }
    Ok(Output {
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
