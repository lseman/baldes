//! Root master integrating the native Rust pricing engine, Clp and
//! limited-memory rank-one cuts. Cut handling follows the C++ reference
//! driver `_solve_root_clp` of `python/src/openrscp/vrptw.py`.
use super::branching::BranchConstraint;
use super::{
    clp::ClpMaster,
    column_generation::{self, Answer, Column, Controller, LpMaster, Options, Output},
    RowSense,
};
use crate::cuts::capacity::{CapacityCut, CapacityParameters, CapacitySeparator, TimeWindows};
use crate::cuts::knapsack::{
    KnapsackColumn, KnapsackParameters, KnapsackRow, ResourceKnapsackSeparator,
};
use crate::cuts::kpath::{KPathGraph, KPathParameters, KPathSeparator, NativeKPathSeparator};
use crate::cuts::FractionalColumn;
use crate::pricing::Solver;
use crate::{invalid, Graph, Result, Route};
use std::cell::RefCell;
use std::collections::HashSet;
use std::rc::Rc;

mod controller;
mod integer;
pub use integer::{EnumeratedMipOutput, EnumerationMipPolicy};
mod options;
mod problem;
mod preprocessing;
use controller::RootController;
use options::RootKPathSeparator;
pub use options::{
    CapacityCutConfig, CutOptions, KPathCutConfig, KnapsackCutConfig, NativeKPathCutConfig,
    RootOptions,
};
pub use problem::{RootProblem, Row};

#[derive(Debug)]
pub struct RootOutput {
    pub preprocessing: crate::master::preprocessing::Report,
    pub snapshot: Option<Rc<NodeSnapshot>>,
    pub simplex_iterations: usize,
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
pub struct NodeSnapshot {
    master: ClpMaster,
    problem: RootProblem,
    cut: CutState,
    pricing: Solver,
    incumbent: f64,
    routes: Vec<Route>,
    warm: column_generation::WarmStart,
}
impl std::fmt::Debug for NodeSnapshot {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NodeSnapshot")
            .field("rows", &self.master.rhs.len())
            .field("columns", &self.master.costs.len())
            .field("cuts", &self.active_cuts())
            .finish()
    }
}
impl NodeSnapshot {
    pub fn branch_depth(&self) -> usize {
        self.problem.branch_constraints.len()
    }
    pub fn active_cuts(&self) -> usize {
        self.cut.rows.iter().filter(|r| r.is_some()).count()
            + self
                .cut
                .capacity_rows
                .iter()
                .filter(|r| r.is_some())
                .count()
            + self
                .cut
                .knapsack_rows
                .iter()
                .filter(|r| r.is_some())
                .count()
    }
    pub fn enumerated_routes(&self) -> Option<usize> {
        self.pricing.enumeration.as_ref().map(|e| e.paths.len)
    }
    fn fork(
        &self,
        branches: &[BranchConstraint],
        penalty: f64,
    ) -> Result<(RootProblem, ClpMaster, column_generation::WarmStart)> {
        let mut master = self.master.try_clone()?;
        let mut problem = self.problem.clone();
        let mut warm = self.warm.clone();
        let penalty = warm
            .artificials
            .iter()
            .map(|&c| master.costs[c])
            .fold(penalty, f64::max);
        for constraint in branches {
            constraint.validate(&problem)?;
            let (sense, rhs) = constraint.row();
            let mut entries = vec![];
            for (c, column) in warm.columns.iter_mut().enumerate() {
                let Some(column) = column else {
                    continue;
                };
                let route = &self.routes[column.tag as usize];
                // Native RF binary resources also make each pair endpoint
                // elementary. Retire old NG cycles outside the new relaxation.
                if let BranchConstraint::RyanFoster(rule) = constraint {
                    if !rule.elementary_on_pair(route.nodes.iter().map(|&v| v as usize)) {
                        master.set_upper(c, 0.)?;
                    }
                }
                let coefficient = constraint.coefficient(&problem, route);
                if coefficient != 0. {
                    entries.push((c, coefficient));
                    column.entries.push((master.rhs.len(), coefficient));
                }
            }
            let row = master.add_row(sense, rhs, &entries)?;
            problem.append_constraint(constraint.clone(), row)?;
            if penalty > 0. {
                for sign in [1., -1.] {
                    if (sign > 0. && sense == RowSense::LessEqual)
                        || (sign < 0. && sense == RowSense::GreaterEqual)
                    {
                        continue;
                    }
                    let c = master.add_column(penalty, &[(row, sign)])?;
                    warm.columns.push(None);
                    warm.artificials.push(c);
                }
            }
        }
        Ok((problem, master, warm))
    }
    pub(crate) fn restricted_value(
        &self,
        branches: &[BranchConstraint],
        penalty: f64,
    ) -> Result<(f64, usize)> {
        let (_, mut master, _) = self.fork(branches, penalty)?;
        let value = if master.solve() {
            master.objective
        } else {
            1e12
        };
        Ok((value, master.simplex_iterations))
    }
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
impl CutState {
    /// Installed separators by decreasing priority. On a tie the relaxation
    /// improvement, which has no name, sorts after the cut families.
    fn separators(&self) -> Vec<(f64, Separator)> {
        let mut list = vec![];
        if self.capacity_separator.is_some() {
            list.push((self.capacity_priority, Separator::Capacity));
        }
        if self.kpath_separator.is_some() {
            list.push((RANK_ONE_PRIORITY, Separator::KPath));
        }
        if self.knapsack_separator.is_some() {
            list.push((RANK_ONE_PRIORITY, Separator::Knapsack));
        }
        if self.options.enabled {
            list.push((RANK_ONE_PRIORITY, Separator::RankOne));
        }
        if self.relaxation_priority > 0. {
            list.push((self.relaxation_priority, Separator::Relaxation));
        }
        list.sort_by(|a, b| {
            b.0.total_cmp(&a.0)
                .then((a.1 == Separator::Relaxation).cmp(&(b.1 == Separator::Relaxation)))
        });
        list
    }
    /// Whether rank-one cuts are at or above the current level.
    fn rank_one_active(&self) -> bool {
        (self.options.enabled || self.kpath_separator.is_some()) && self.level <= RANK_ONE_PRIORITY
    }
    /// The next lower priority level, if any.
    fn next_level(&self) -> Option<f64> {
        self.separators()
            .into_iter()
            .map(|(p, _)| p)
            .find(|&p| p < self.level)
    }
}
struct State {
    solver: Solver,
    routes: Vec<Route>,
    /// Arc sequence -> tag of each entry in `routes`.
    route_tags: std::collections::HashMap<Vec<i32>, i32>,
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
impl State {
    fn remaining_time(&self) -> Result<std::time::Duration> {
        match self.deadline {
            Some(end) => end
                .checked_duration_since(std::time::Instant::now())
                .filter(|d| !d.is_zero())
                .ok_or_else(|| crate::incomplete("node time limit")),
            None => Ok(std::time::Duration::MAX),
        }
    }
    fn customers(route: &Route) -> Vec<usize> {
        route.nodes[1..route.nodes.len() - 1]
            .iter()
            .map(|&v| v as usize)
            .collect()
    }
    fn is_enumerated(&self, route: &Route) -> bool {
        if self.cut.enumerated.is_empty() {
            return false;
        }
        let mut arcs = route.arcs.clone();
        arcs.sort_unstable();
        self.cut.enumerated.contains(&arcs)
    }
    fn cut_entries(&self, route: &Route) -> Result<Vec<(usize, f64)>> {
        self.cut
            .entries(&self.solver, route, self.is_enumerated(route))
    }
    fn column(
        &self,
        problem: &RootProblem,
        route: &Route,
        duals: &[f64],
        tag: i32,
    ) -> Result<Column> {
        let mut entries = problem.base_entries(route)?;
        entries.extend(self.cut_entries(route)?);
        entries.sort_by_key(|&(row, _)| row);
        let (cost, _) = problem.route_cost(route);
        let mut reduced_cost = cost;
        for &(r, a) in &entries {
            reduced_cost -= duals.get(r).copied().unwrap_or(0.) * a;
        }
        Ok(Column {
            cost,
            reduced_cost,
            entries,
            acceptable: true,
            tag,
            // The route's arcs in path order: NG routes with the same arcs in
            // another order (a cycle traversed in reverse) are distinct columns,
            // and can have different rank-one coefficients.
            identity: route
                .arcs
                .iter()
                .map(i32::to_string)
                .collect::<Vec<_>>()
                .join(","),
        })
    }
    /// cache_costs: arc reduced costs and the rounded cut duals, active cuts
    /// ordered by rounded dual magnitude then index.
    fn reprice(&mut self, problem: &RootProblem, duals: &[f64]) -> Result<f64> {
        let rounded = self.rounded;
        let offset = problem.reprice(
            &mut self.solver,
            duals,
            rounded,
            &self.cut.capacity_arc_entries,
        )?;
        for (cut, row) in self
            .solver
            .cuts
            .knapsack_cuts
            .iter_mut()
            .zip(&self.cut.knapsack_rows)
        {
            cut.dual = row.map_or(0., |r| duals[r]);
        }
        if !self.cut.rows.is_empty() {
            let scaled = |row: usize| (-duals[row] * 1e8 + 0.5).floor();
            let mut order: Vec<usize> = (0..self.cut.rows.len())
                .filter(|&i| self.cut.rows[i].is_some_and(|r| duals[r].abs() > 1e-6))
                .collect();
            order.sort_by(|&a, &b| {
                let ka = -scaled(self.cut.rows[a].unwrap()).abs();
                let kb = -scaled(self.cut.rows[b].unwrap()).abs();
                ka.total_cmp(&kb).then(a.cmp(&b))
            });
            self.solver.cuts.pricing_order = order;
            for (i, row) in self.cut.rows.iter().enumerate() {
                self.solver.cuts.cuts[i].dual = match row {
                    Some(r) => -scaled(*r) / 1e8,
                    None => 0.,
                };
            }
        }
        Ok(offset)
    }
}
/// Tag of `route`, the index of the first registered route with its arcs.
fn register(state: &mut State, route: &Route) -> Result<i32> {
    if let Some(&tag) = state.route_tags.get(&route.arcs) {
        return Ok(tag);
    }
    let id = i32::try_from(state.routes.len()).map_err(|_| invalid("route tag overflow"))?;
    state.routes.push(route.clone());
    state.route_tags.insert(route.arcs.clone(), id);
    Ok(id)
}
pub fn solve(problem: RootProblem, mut options: RootOptions) -> Result<RootOutput> {
    options.cg.warm_start = None;
    solve_seeded(problem, options, None)
}
pub fn inherit(
    parent: &NodeSnapshot,
    branches: &[(usize, RowSense, f64)],
    options: RootOptions,
) -> Result<RootOutput> {
    let branches: Vec<_> = branches
        .iter()
        .map(|&(arc, sense, rhs)| BranchConstraint::ArcFlow { arc, sense, rhs })
        .collect();
    inherit_branches(parent, &branches, options)
}
pub fn inherit_branches(
    parent: &NodeSnapshot,
    branches: &[BranchConstraint],
    mut options: RootOptions,
) -> Result<RootOutput> {
    options.node_depth = options
        .node_depth
        .max(parent.branch_depth() + branches.len());
    let (problem, master, warm) = parent.fork(branches, options.cg.artificial_cost)?;
    solve_seeded(problem, options, Some((parent, master, warm)))
}
fn solve_seeded(
    problem: RootProblem,
    mut options: RootOptions,
    seed: Option<(&NodeSnapshot, ClpMaster, column_generation::WarmStart)>,
) -> Result<RootOutput> {
    options.normalize_pricing_options()?;
    let solve_started = std::time::Instant::now();
    problem.graph.validate()?;
    let (preprocessing, arc_bounds) = problem.preprocess()?;
    if preprocessing.infeasible {
        return Ok(preprocessing::infeasible_output(preprocessing, solve_started));
    }
    let (saved, mut master) = if let Some((saved, master, warm)) = seed {
        options.cg.warm_start = Some(warm);
        (Some(saved), master)
    } else {
        (None, ClpMaster::new()?)
    };
    // Keep Clp's default 1e-7 primal and dual tolerances.
    master.set_tolerances(1e-7, 1e-7)?;
    master.zero_tolerance = 1e-7;
    if saved.is_none() {
        for row in &problem.rows {
            master.add_row(row.sense, row.rhs, &[])?;
        }
    }
    let mut solver = if let Some(saved) = saved {
        saved.pricing.fork_for_node()
    } else {
        let mut solver = Solver::new(problem.graph.vertices.len(), problem.graph.resources.len())?;
        solver.graph = problem.graph.clone();
        solver
    };
    // A snapshot may later branch on any arc; same-customer-set orderings
    // must remain distinct in its retained enumeration.
    solver.enumeration_preserve_arc_paths =
        options.capture_state || problem.rows.len() > problem.customers.len() + 1;
    for (name, value) in &options.pricing_options {
        solver.set_option(name, *value)?;
    }
    solver.set_ryan_foster(
        problem
            .branch_constraints
            .iter()
            .filter_map(|(_, c)| {
                if let BranchConstraint::RyanFoster(c) = c {
                    Some(*c)
                } else {
                    None
                }
            })
            .collect(),
    )?;
    solver.set_forbidden_arcs(arc_bounds.iter().enumerate()
        .filter_map(|(a, b)| (b.upper <= 0.).then_some(a)).collect())?;
    if let Some(warm) = options.cg.warm_start.as_ref() {
        if let Some(saved) = saved {
            for (c, column) in warm.columns.iter().enumerate() {
                if let Some(column) = column {
                    let route = &saved.routes[column.tag as usize];
                    if route.arcs.iter().any(|&id| problem.arc_position(id)
                        .is_some_and(|a| arc_bounds[a].upper <= 0.)) {
                        master.set_upper(c, 0.)?;
                    }
                }
            }
        }
    }
    let phase_floor = options.pricing_phase_floor;
    if phase_floor >= solver.phases.len() {
        return Err(invalid("pricing phase floor outside schedule"));
    }
    let schedule = solver.phases[phase_floor..].to_vec();
    let capacity_separator = if saved.is_some() {
        None
    } else if let Some(config) = options.capacity_cuts.take() {
        if problem.graph.source != 0
            || problem.graph.sink != config.demands.len()
            || problem.graph.vertices.len() != config.demands.len() + 1
        {
            return Err(invalid(
                "capacity cuts require depot 0, customers 1..n, sink n+1",
            ));
        }
        let mut separator = CapacitySeparator::new(config.demands, config.capacity)?;
        separator.parameters = config.parameters;
        if let Some(windows) = config.time_windows {
            separator.set_time_windows(windows)?;
        }
        Some(separator)
    } else {
        None
    };
    if saved.is_none() && options.kpath_cuts.is_some() && options.native_kpath_cuts.is_some() {
        return Err(invalid("select standalone or native k-path configuration"));
    }
    let kpath_separator =
        if saved.is_some() {
            None
        } else if let Some(config) = options.native_kpath_cuts.take() {
            let graph_id = problem.graph.graph_id.max(0);
            let input = config
                .graphs
                .iter()
                .find(|g| g.graph.id == graph_id)
                .ok_or_else(|| invalid("native k-path configuration has no pricing graph"))?;
            if input.graph.source != problem.graph.source as i32
                || input.graph.sink != problem.graph.sink as i32
                || input
                    .graph
                    .vertices
                    .iter()
                    .any(|v| v.id < 0 || v.id as usize >= problem.graph.vertices.len())
            {
                return Err(invalid("native k-path pricing vertices/terminals differ"));
            }
            for arc in &problem.graph.arcs {
                if !input.graph.arcs.iter().any(|a| {
                    a.id == arc.id && a.tail == arc.tail as i32 && a.head == arc.head as i32
                }) {
                    return Err(invalid("native k-path pricing arc incidence differs"));
                }
            }
            let mut separator = NativeKPathSeparator::new(config.use_packing_sets);
            separator.parameters = config.parameters;
            separator.prepare(
                &config.graphs,
                &config.demands,
                config.capacity,
                config.two_path_resource,
            )?;
            Some(RootKPathSeparator::Native {
                separator,
                graphs: config.graphs,
            })
        } else if let Some(config) = options.kpath_cuts.take() {
            if problem.graph.source != 0
                || problem.graph.sink != config.demands.len()
                || problem.graph.vertices.len() != config.demands.len() + 1
            {
                return Err(invalid(
                    "k-path cuts require depot 0, customers 1..n, sink n+1",
                ));
            }
            let mut separator = KPathSeparator::new(config.demands, config.capacity)?;
            separator.parameters = config.parameters;
            if let Some(windows) = config.time_windows {
                separator.set_time_windows(windows)?;
            }
            Some(RootKPathSeparator::Standalone(separator))
        } else {
            None
        };
    let knapsack_separator = if saved.is_some() {
        None
    } else if let Some(mut config) = options.knapsack_cuts.take() {
        let graph = problem.graph.graph_id.max(0);
        solver.knapsack_resource_indices = config.resource_indices;
        let indices = solver.knapsack_indices();
        if indices
            .values()
            .any(|&r| r >= problem.graph.resources.len())
        {
            return Err(invalid(
                "knapsack resource mapping exceeds solver coordinates",
            ));
        }
        for row in &config.rows {
            let Some(&(resource, scale)) = row.graph_resources.get(&graph) else {
                return Err(invalid(
                    "knapsack original row has no resource for pricing graph",
                ));
            };
            if !indices.contains_key(&resource) || !scale.is_finite() || scale <= 0. {
                return Err(invalid("knapsack original row resource/scaling"));
            }
            // Apply scale to separation/master coefficients, while
            // active-dual retrieval retains the integer cut breakpoints.
        }
        for (name, value) in &options.pricing_options {
            match name.as_str() {
                "BALDESresConsKnapsackCutsMode" => {
                    config.parameters.use_rounding = *value > 0.;
                    config.parameters.max_one_k = value.abs() as i32;
                }
                "BALDESresConsKnapsackCutsMaxNumPerRound" => {
                    config.parameters.max_cuts_per_round = *value as usize
                }
                _ => {}
            }
        }
        crate::cuts::knapsack::KnapsackSeparator::with_parameters(config.parameters.clone())?;
        Some(ResourceKnapsackSeparator {
            rows: config.rows,
            parameters: config.parameters,
            ..Default::default()
        })
    } else {
        None
    };
    let arc_count = problem.graph.arcs.len();
    let state = Rc::new(RefCell::new(State {
        solver,
        routes: saved.map_or_else(Vec::new, |s| s.routes.clone()),
        route_tags: saved.map_or_else(Default::default, |s| {
            s.routes
                .iter()
                .enumerate()
                .map(|(i, r)| (r.arcs.clone(), i as i32))
                .collect()
        }),
        incumbent: saved.map_or(options.incumbent, |s| options.incumbent.min(s.incumbent)),
        enumerate: options.enumerate,
        rounded: options.round_reduced_costs,
        previous_fixing_gap: 1e12,
        deadline: solve_started.checked_add(options.cg.time_limit),
        pricing_seconds: 0.,
        cut: CutState {
            options: options.cuts.clone(),
            rows: vec![],
            memories: vec![],
            enumerated: HashSet::new(),
            lp_before_cuts: f64::NAN,
            separation_lp: f64::NAN,
            rounds: 0,
            tailing: 0,
            capacity_separator,
            capacity_priority: if options.node_depth == 0 {
                CAPACITY_PRIORITY
            } else {
                1.
            },
            capacity_cuts: vec![],
            capacity_rows: vec![],
            capacity_arc_entries: vec![vec![]; arc_count],
            kpath_separator,
            kpath_indices: vec![],
            knapsack_separator,
            knapsack_rows: vec![],
            level: f64::INFINITY,
            relaxation_priority: options.relaxation_improvement_priority as f64,
            removed_columns: vec![],
        },
        fixing_calls: 0,
        enumerated_routes: 0,
    }));
    {
        // eval starts at the highest separator priority.
        let mut s = state.borrow_mut();
        if let Some(saved) = saved {
            s.cut = saved.cut.clone();
            s.cut.options = options.cuts.clone();
            s.cut.capacity_priority = if options.node_depth == 0 {
                CAPACITY_PRIORITY
            } else {
                1.
            };
            s.cut.relaxation_priority = options.relaxation_improvement_priority as f64;
            s.cut.rounds = 0;
            s.cut.tailing = 0;
            s.cut.separation_lp = f64::NAN;
            s.cut.lp_before_cuts = f64::NAN;
            s.cut.removed_columns.clear();
        }
        // Apply explicit controls after cloning inherited separator state.
        // Unspecified controls retain the caller's or parent's configuration.
        if let Some(separator) = s.cut.capacity_separator.as_mut() {
            for (name, value) in &options.pricing_options {
                match name.as_str() {
                    "BALDESuseCapacityCuts" => separator.parameters.enabled = *value != 0.,
                    "BALDEScapCutsMaxNumPerRound" => {
                        separator.parameters.max_cuts_per_round = *value as usize
                    }
                    _ => {}
                }
            }
        }
        s.cut.level = s
            .cut
            .separators()
            .first()
            .map_or(0., |&(priority, _)| priority);
    }
    let zeros = vec![0.; problem.rows.len()];
    options.cg.known.clear();
    let mut live: HashSet<i32> = options
        .cg
        .warm_start
        .as_ref()
        .map_or_else(HashSet::new, |w| {
            w.columns.iter().flatten().map(|c| c.tag).collect()
        });
    for route in &options.initial_routes {
        if route.arcs.iter().any(|&id| problem.arc_position(id)
            .is_some_and(|a| arc_bounds[a].upper <= 0.)) { continue; }
        let mut s = state.borrow_mut();
        if s.solver
            .ryan_foster
            .iter()
            .any(|c| !c.elementary_on_pair(route.nodes.iter().map(|&v| v as usize)))
        {
            continue;
        }
        let tag = register(&mut s, route)?;
        if !live.insert(tag) {
            continue;
        }
        let column = s.column(&problem, route, &zeros, tag)?;
        master.add_column(column.cost, &column.entries)?;
        if let Some(warm) = options.cg.warm_start.as_mut() {
            warm.columns.push(Some(column));
        } else {
            options.cg.known.push(column);
        }
    }
    let problem = Rc::new(problem);
    let mut controller = RootController {
        problem: problem.clone(),
        state: state.clone(),
    };
    let initial_rows = options
        .cg
        .warm_start
        .as_ref()
        .map_or(problem.rows.len(), |w| w.initial_rows);
    let capture_state = options.capture_state;
    let inherited_cuts = saved.map_or(0, |s| s.active_cuts());
    let mut output = column_generation::optimize::<ClpMaster, _, _>(
        &mut master,
        &schedule,
        |phase, duals| {
            let pricing_started = std::time::Instant::now();
            let mut state = state.borrow_mut();
            let rounded = state.rounded;
            let offset = state.reprice(&problem, duals)?;
            let maximum = state.solver.controls.max_columns;
            let remaining = state.remaining_time()?;
            let saved_time = state.solver.controls.time_limit;
            state.solver.controls.time_limit = saved_time.min(remaining);
            let result = state
                .solver
                .price_phase(phase + phase_floor, maximum, false);
            state.solver.controls.time_limit = saved_time;
            let result = result?;
            let mut columns = vec![];
            let mut minimum = result.minimum_reduced_cost + offset;
            let mut corrected = 0.;
            for route in result.columns {
                let tag = register(&mut state, &route)?;
                let mut column = state.column(&problem, &route, duals, tag)?;
                if columns.is_empty() {
                    corrected = route.cost + offset;
                    let mut counts = std::collections::BTreeMap::new();
                    for &arc in &route.arcs {
                        *counts.entry(arc).or_insert(0usize) += 1;
                    }
                    if rounded {
                        for (id, count) in counts {
                            let index = problem.arc_position(id).unwrap();
                            let mut raw = problem.graph.arcs[index].cost;
                            for &(r, a) in &problem.arc_entries[index] {
                                raw -= duals[problem.row_map.borrow()[r]] * a;
                            }
                            for &(r, a) in &state.cut.capacity_arc_entries[index] {
                                raw -= duals[r] * a;
                            }
                            let cached = (raw * 1e8 + 0.5).floor() / 1e8;
                            corrected += (raw - cached) * count as f64;
                        }
                    }
                }
                column.reduced_cost = route.cost + offset;
                minimum = minimum.min(column.reduced_cost);
                columns.push(column);
            }
            state.pricing_seconds += pricing_started.elapsed().as_secs_f64();
            if std::env::var_os("BALDES_DEBUG_COLUMNS").is_some() {
                eprintln!("price phase={phase} columns={}", columns.len());
                for c in &columns {
                    eprintln!("  {:.17e} {}", c.reduced_cost, c.identity);
                }
            }
            Ok(Answer {
                columns,
                minimum,
                exact: result.exact,
                bound_minimum: Some(corrected),
            })
        },
        &mut controller,
        options.cg,
    )?;
    output.warm_start.initial_rows = initial_rows;
    let state = state.borrow();
    // Extract values, duals and simplex iterations before moving master into snapshot.
    let values = master.values().to_vec();
    let duals = master.duals().to_vec();
    let simplex_iterations = master.simplex_iterations;
    let snapshot = capture_state.then(|| {
        Rc::new(NodeSnapshot {
            master,
            problem: (*problem).clone(),
            cut: state.cut.clone(),
            pricing: state.solver.fork_for_node(),
            incumbent: state.incumbent,
            routes: state.routes.clone(),
            warm: output.warm_start.clone(),
        })
    });
    Ok(RootOutput {
        preprocessing,
        snapshot,
        simplex_iterations,
        inherited_cuts,
        master: output,
        routes: state.routes.clone(),
        values,
        duals,
        incumbent: state.incumbent,
        pricing_seconds: state.pricing_seconds,
        solver_seconds: solve_started.elapsed().as_secs_f64(),
        cuts: state.solver.cuts.cuts.len()
            + state.cut.capacity_cuts.len()
            + state.solver.cuts.knapsack_cuts.len(),
        lp_before_cuts: state.cut.lp_before_cuts,
        fixing_calls: state.fixing_calls,
        enumerated_routes: state.enumerated_routes,
    })
}

impl CutState {
    fn entries(
        &self,
        solver: &Solver,
        route: &Route,
        enumerated: bool,
    ) -> Result<Vec<(usize, f64)>> {
        let mut entries = vec![];
        let customers = State::customers(route);
        for (cut, row) in self.capacity_cuts.iter().zip(&self.capacity_rows) {
            if let Some(row) = row {
                let coefficient = cut.coefficient(&customers);
                if coefficient != 0 {
                    entries.push((*row, coefficient as f64));
                }
            }
        }
        for (cut, row) in solver.cuts.knapsack_cuts.iter().zip(&self.knapsack_rows) {
            if let Some(row) = row {
                let coefficient = solver.knapsack_coefficient(cut, &route.resources);
                if coefficient != 0 {
                    entries.push((*row, coefficient as f64));
                }
            }
        }
        if self.rows.iter().all(Option::is_none) {
            return Ok(entries);
        }
        let path = solver.cuts.route(&solver.graph, &State::customers(route))?;
        for (index, row) in self.rows.iter().enumerate() {
            let Some(row) = *row else {
                continue;
            };
            let c = solver
                .cuts
                .route_coefficient(&solver.graph, index, &path, enumerated)?;
            if c != 0 {
                entries.push((row, c as f64));
            }
        }
        Ok(entries)
    }
}
