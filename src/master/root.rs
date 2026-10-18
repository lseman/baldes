//! Root master integrating the native Rust pricing engine, Clp and
//! limited-memory rank-one cuts. Cut handling follows the C++ reference
//! driver `_solve_root_clp` of `python/src/openrscp/vrptw.py`.
use super::{
    clp::ClpMaster,
    column_generation::{self, Answer, Column, Controller, LpMaster, Options, Output},
    RowSense,
};
use crate::cuts::FractionalColumn;
use crate::cuts::capacity::{CapacityCut, CapacityParameters, CapacitySeparator, TimeWindows};
use crate::cuts::kpath::{KPathGraph, KPathParameters, KPathSeparator, NativeKPathSeparator};
use crate::cuts::knapsack::{KnapsackColumn, KnapsackParameters, KnapsackRow, ResourceKnapsackSeparator};
use crate::pricing::Solver;
use crate::{invalid, Graph, Result, Route};
use std::cell::RefCell;
use std::collections::HashSet;
use std::rc::Rc;
#[derive(Clone, Debug)]
pub struct Row {
    pub sense: RowSense,
    pub rhs: f64,
}
#[derive(Clone, Debug)]
pub struct RootProblem {
    pub graph: Graph,
    pub rows: Vec<Row>,
    pub arc_entries: Vec<Vec<(usize, f64)>>,
    pub route_entries: Vec<(usize, f64)>,
    /// Vertices with a unit covering row: the rank-one cut customers.
    pub customers: Vec<usize>,
    /// Arc id -> first index in `graph.arcs`.
    arc_index: std::collections::HashMap<i32, usize>,
    /// Logical base/branch row -> physical master row. Branches appended
    /// after inherited cuts stay valid when those cuts are retired.
    row_map: RefCell<Vec<usize>>,
}
impl RootProblem {
    /// Cover each nonterminal vertex and bound the number of vehicle routes.
    /// Arc entries charge the head vertex, retaining repeated-visit coefficients.
    pub fn set_partitioning(graph: Graph, vehicles: f64) -> Result<Self> {
        if !vehicles.is_finite() || vehicles <= 0. {
            return Err(invalid("positive finite vehicle multiplicity required"));
        }
        let mut rows = vec![Row {
            sense: RowSense::LessEqual,
            rhs: vehicles,
        }];
        let mut customer_row = vec![None; graph.vertices.len()];
        let mut customers = vec![];
        for v in 0..graph.vertices.len() {
            if v != graph.source && v != graph.sink {
                customer_row[v] = Some(rows.len());
                customers.push(v);
                rows.push(Row {
                    sense: RowSense::Equal,
                    rhs: 1.,
                });
            }
        }
        let fleet = 0;
        let arc_entries = graph
            .arcs
            .iter()
            .map(|a| {
                if a.head == graph.sink {
                    vec![(fleet, 1.)]
                } else {
                    customer_row[a.head].map_or_else(Vec::new, |r| vec![(r, 1.)])
                }
            })
            .collect();
        let mut arc_index = std::collections::HashMap::with_capacity(graph.arcs.len());
        for (index, a) in graph.arcs.iter().enumerate() {
            arc_index.entry(a.id).or_insert(index);
        }
        let row_map = RefCell::new((0..rows.len()).collect());
        Ok(Self {
            graph,
            rows,
            arc_entries,
            route_entries: vec![],
            customers,
            arc_index,
            row_map,
        })
    }
    pub fn arc_position(&self, id: i32) -> Option<usize> {
        self.arc_index.get(&id).copied()
    }
    pub(crate) fn append_branch(&mut self, arc: usize, sense: RowSense, rhs: f64, physical: usize) {
        let logical = self.rows.len();
        self.rows.push(Row { sense, rhs });
        self.row_map.borrow_mut().push(physical);
        self.arc_entries[arc].push((logical, 1.));
    }
    fn base_entries(&self, route: &Route) -> Result<Vec<(usize, f64)>> {
        let map = self.row_map.borrow();
        let mut coefficients = std::collections::BTreeMap::new();
        for &(r, a) in &self.route_entries {
            *coefficients.entry(map[r]).or_insert(0.) += a;
        }
        for id in &route.arcs {
            let physical = self
                .arc_position(*id)
                .ok_or_else(|| invalid("unknown priced route arc"))?;
            for &(r, a) in &self.arc_entries[physical] {
                *coefficients.entry(map[r]).or_insert(0.) += a;
            }
        }
        Ok(coefficients.into_iter().filter(|(_, a)| *a != 0.).collect())
    }
    fn route_cost(&self, route: &Route) -> (f64, Vec<i32>) {
        let mut identities = route.arcs.clone();
        identities.sort_unstable();
        let mut cost = 0.;
        let mut counts = std::collections::BTreeMap::new();
        for &id in &identities {
            *counts.entry(id).or_insert(0usize) += 1;
        }
        for (id, count) in counts {
            cost += self.graph.arcs[self.arc_position(id).unwrap()].cost * count as f64;
        }
        (cost, identities)
    }
    fn reprice(&self, solver: &mut Solver, duals: &[f64], round: bool,
        capacity_entries: &[Vec<(usize, f64)>]) -> Result<f64> {
        let map = self.row_map.borrow();
        if duals.len() < self.rows.len() || self.arc_entries.len() != self.graph.arcs.len() {
            return Err(invalid("root master dimensions"));
        }
        for (index, arc) in solver.graph.arcs.iter_mut().enumerate() {
            let mut cost = self.graph.arcs[index].cost;
            for &(r, a) in &self.arc_entries[index] {
                cost -= duals[map[r]] * a;
            }
            for &(r, a) in &capacity_entries[index] {
                cost -= duals[r] * a;
            }
            arc.reduced_cost = if round {
                (cost * 1e8 + 0.5).floor() / 1e8
            } else {
                cost
            };
        }
        Ok(-self
            .route_entries
            .iter()
            .map(|&(r, a)| duals[map[r]] * a)
            .sum::<f64>())
    }
}
/// Rank-one cut controls of the root driver.
#[derive(Clone, Debug)]
pub struct CutOptions {
    pub enabled: bool,
    pub tailing_off_threshold: f64,
    pub tailing_off_counter: usize,
    pub min_rounds_before_stop_by_sp: usize,
}
impl Default for CutOptions {
    fn default() -> Self {
        Self {
            enabled: false,
            tailing_off_threshold: 0.02,
            tailing_off_counter: 3,
            min_rounds_before_stop_by_sp: 1,
        }
    }
}
#[derive(Clone, Debug)]
pub struct CapacityCutConfig {
    /// Depot at index 0, customers at indices 1..n; sink is n+1.
    pub demands: Vec<f64>,
    pub capacity: f64,
    pub parameters: CapacityParameters,
    pub time_windows: Option<TimeWindows>,
}
impl CapacityCutConfig {
    /// Attach the time resource used by Kohl 2-path separator.
    /// These cuts share the capacity family's separation priority and pool.
    pub fn with_time_windows(mut self, windows: TimeWindows) -> Self {
        self.parameters.use_two_path = true;
        self.time_windows = Some(windows);
        self
    }
    pub fn new(demands: Vec<f64>, capacity: f64) -> Self {
        let mut parameters = CapacityParameters::default();
        // GenericRCSPCapacityCutConstr's packing form, no Kohl resource.
        parameters.complement_form = true;
        parameters.use_two_path = false;
        Self { demands, capacity, parameters, time_windows: None }
    }
}
#[derive(Clone, Debug)]
pub struct KPathCutConfig {
    /// Depot at index 0, customers at indices 1..n; sink is n+1.
    pub demands: Vec<f64>,
    pub capacity: f64,
    pub parameters: KPathParameters,
    pub time_windows: Option<TimeWindows>,
}
impl KPathCutConfig {
    pub fn new(demands: Vec<f64>, capacity: f64) -> Self {
        Self {
            demands,
            capacity,
            parameters: KPathParameters::default(),
            time_windows: None,
        }
    }
    pub fn with_time_windows(mut self, windows: TimeWindows) -> Self {
        self.time_windows = Some(windows);
        self
    }
}
/// General native strong k-path separation on original graph/set IDs.
#[derive(Clone, Debug)]
pub struct NativeKPathCutConfig {
    pub graphs: Vec<KPathGraph>,
    /// Original zero-based packing/covering set order; no depot entry.
    pub demands: Vec<f64>,
    pub capacity: f64,
    pub use_packing_sets: bool,
    pub two_path_resource: i32,
    pub parameters: KPathParameters,
}
impl NativeKPathCutConfig {
    pub fn new(graphs: Vec<KPathGraph>, demands: Vec<f64>, capacity: f64, use_packing_sets: bool) -> Self {
        Self { graphs, demands, capacity, use_packing_sets, two_path_resource: -1, parameters: Default::default() }
    }
}
#[derive(Clone, Debug)]
enum RootKPathSeparator {
    Standalone(KPathSeparator),
    Native { separator: NativeKPathSeparator, graphs: Vec<KPathGraph> },
}
/// Original integer resource rows used by RouteLoadKnapsackCutSeparation.
#[derive(Clone, Debug, Default)]
pub struct KnapsackCutConfig {
    pub rows: Vec<KnapsackRow>,
    pub parameters: KnapsackParameters,
}
impl KnapsackCutConfig {
    pub fn new(rows: Vec<KnapsackRow>) -> Self { Self { rows, parameters: Default::default() } }
}
#[derive(Clone, Debug)]
pub struct RootOptions {
    /// Retain an independently forkable model and active cuts for descendants.
    pub capture_state: bool,
    /// Node depth: capacity/Kohl priority is 3 at root and 1 below it,
    /// as in generic capacity-cut registration.
    pub node_depth: usize,
    /// Lowest pricing phase evaluated (0 for a certified node solve).
    /// Strong-branching heuristic trials may stop above the exact phase.
    pub pricing_phase_floor: usize,
    pub cg: Options,
    pub pricing_options: Vec<(String, f64)>,
    pub initial_routes: Vec<Route>,
    pub incumbent: f64,
    pub enumerate: bool,
    pub round_reduced_costs: bool,
    pub cuts: CutOptions,
    pub capacity_cuts: Option<CapacityCutConfig>,
    pub kpath_cuts: Option<KPathCutConfig>,
    pub native_kpath_cuts: Option<NativeKPathCutConfig>,
    pub knapsack_cuts: Option<KnapsackCutConfig>,
    /// ColGenSpRelaxationImprovementPriority: separator priority of dynamic
    /// NG strengthening (0 disables it, default). Its mode and caps
    /// are the pricing options RCSPdynamicNGmode, RCSPmaxNGneighbourhoodSize,
    /// RCSPmaxNGaverNeighbourhoodSize, RCSPmaxNumOfColumnsInDynamicNG and
    /// RCSPmaxCycleSizeInDynamicNG.
    pub relaxation_improvement_priority: f32,
}
impl Default for RootOptions {
    fn default() -> Self {
        let mut cg = Options::default();
        cg.artificial_cost = 2.76;
        // master column clean-up: threshold 10000, keep ratio 0.66.
        cg.column_cleanup_threshold = 10000.;
        cg.column_cleanup_ratio = 0.66;
        Self {
            capture_state: false,
            node_depth: 0,
            pricing_phase_floor: 0,
            cg,
            pricing_options: vec![],
            initial_routes: vec![],
            incumbent: f64::INFINITY,
            enumerate: true,
            round_reduced_costs: true,
            cuts: CutOptions::default(),
            capacity_cuts: None,
            kpath_cuts: None,
            native_kpath_cuts: None,
            knapsack_cuts: None,
            relaxation_improvement_priority: 0.,
        }
    }
}
impl RootOptions {
    pub fn with_capacity_cuts(mut self, config: CapacityCutConfig) -> Self {
        self.capacity_cuts = Some(config);
        self.cg.remove_basic_cuts = true;
        self.cg.cut_artificial_cost = if self.cg.artificial_cost > 0. { self.cg.artificial_cost / 1.2 } else { 0. };
        self.cg.tailing_off_counter = 1_000_000;
        self
    }
    pub fn with_kpath_cuts(mut self, config: KPathCutConfig) -> Self {
        self.kpath_cuts = Some(config);
        self.native_kpath_cuts = None;
        self.cg.remove_basic_cuts = true;
        self.cg.cut_artificial_cost = if self.cg.artificial_cost > 0. {
            self.cg.artificial_cost / 1.2
        } else {
            0.
        };
        self.cg.tailing_off_counter = 1_000_000;
        self
    }
    pub fn with_native_kpath_cuts(mut self, config: NativeKPathCutConfig) -> Self {
        self.native_kpath_cuts = Some(config);
        self.kpath_cuts = None;
        self.cg.remove_basic_cuts = true;
        self.cg.cut_artificial_cost = if self.cg.artificial_cost > 0. { self.cg.artificial_cost / 1.2 } else { 0. };
        self.cg.tailing_off_counter = 1_000_000;
        self
    }
    pub fn with_knapsack_cuts(mut self, config: KnapsackCutConfig) -> Self {
        self.knapsack_cuts = Some(config);
        self.cg.remove_basic_cuts = true;
        self.cg.cut_artificial_cost = if self.cg.artificial_cost > 0. { self.cg.artificial_cost / 1.2 } else { 0. };
        self.cg.tailing_off_counter = 1_000_000;
        self
    }
    /// The C++ driver's master options with rank-one separation: basic cut
    /// rows are removed, cut artificials cost artificial_cost / 1.2, and the
    /// separator owns tailing off.
    pub fn with_rank_one_cuts(mut self) -> Self {
        self.cuts.enabled = true;
        self.cg.remove_basic_cuts = true;
        self.cg.cut_artificial_cost = if self.cg.artificial_cost > 0. {
            self.cg.artificial_cost / 1.2
        } else {
            0.
        };
        self.cg.tailing_off_counter = 1_000_000;
        self
    }
}
#[derive(Debug)]
pub struct RootOutput {
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
    registry: crate::cuts::CutRegistry,
    pricing_graph: Graph,
    routes: Vec<Route>,
    warm: column_generation::WarmStart,
}
impl std::fmt::Debug for NodeSnapshot {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NodeSnapshot").field("rows", &self.master.rhs.len())
            .field("columns", &self.master.costs.len()).field("cuts", &self.active_cuts()).finish()
    }
}
impl NodeSnapshot {
    pub fn branch_depth(&self) -> usize { self.problem.rows.len()-self.problem.customers.len()-1 }
    pub fn active_cuts(&self) -> usize {
        self.cut.rows.iter().filter(|r|r.is_some()).count()
            + self.cut.capacity_rows.iter().filter(|r|r.is_some()).count()
            + self.cut.knapsack_rows.iter().filter(|r|r.is_some()).count()
    }
    fn fork(&self, branches: &[(usize,RowSense,f64)], penalty: f64)
        -> Result<(RootProblem,ClpMaster,column_generation::WarmStart)> {
        let mut master=self.master.try_clone()?;
        let mut problem=self.problem.clone();
        let mut warm=self.warm.clone();
        let penalty=warm.artificials.iter().map(|&c|master.costs[c]).fold(penalty,f64::max);
        for &(arc,sense,rhs) in branches {
            if arc>=problem.graph.arcs.len() || !rhs.is_finite() {return Err(invalid("inherited branch row"));}
            let id=problem.graph.arcs[arc].id;
            let mut entries=vec![];
            for (c,column) in warm.columns.iter_mut().enumerate() {
                let Some(column)=column else {continue;};
                let coefficient=self.routes[column.tag as usize].arcs.iter().filter(|&&a|a==id).count() as f64;
                if coefficient!=0. {entries.push((c,coefficient));column.entries.push((master.rhs.len(),coefficient));}
            }
            let row=master.add_row(sense,rhs,&entries)?;
            problem.append_branch(arc,sense,rhs,row);
            if penalty>0. {
                for sign in [1.,-1.] {
                    if (sign>0. && sense==RowSense::LessEqual) || (sign<0. && sense==RowSense::GreaterEqual) {continue;}
                    let c=master.add_column(penalty,&[(row,sign)])?;
                    warm.columns.push(None);warm.artificials.push(c);
                }
            }
        }
        Ok((problem,master,warm))
    }
    pub(crate) fn restricted_value(&self, branches: &[(usize,RowSense,f64)], penalty:f64) -> Result<(f64,usize)> {
        let (_,mut master,_)=self.fork(branches,penalty)?;
        let value=if master.solve() {master.objective} else {1e12};
        Ok((value,master.simplex_iterations))
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
        list.sort_by(|a, b| b.0.total_cmp(&a.0).then((a.1 == Separator::Relaxation).cmp(&(b.1 == Separator::Relaxation))));
        list
    }
    /// Whether rank-one cuts are at or above the current level.
    fn rank_one_active(&self) -> bool {
        (self.options.enabled || self.kpath_separator.is_some()) && self.level <= RANK_ONE_PRIORITY
    }
    /// The next lower priority level, if any.
    fn next_level(&self) -> Option<f64> {
        self.separators().into_iter().map(|(p, _)| p).find(|&p| p < self.level)
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
    pricing_seconds: f64,
    cut: CutState,
    fixing_calls: usize,
    enumerated_routes: usize,
}
impl State {
    fn customers(route: &Route) -> Vec<usize> {
        route.nodes[1..route.nodes.len() - 1].iter().map(|&v| v as usize).collect()
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
        let mut entries = vec![];
        let customers = Self::customers(route);
        for (cut, row) in self.cut.capacity_cuts.iter().zip(&self.cut.capacity_rows) {
            if let Some(row) = row {
                let coefficient = cut.coefficient(&customers);
                if coefficient != 0 { entries.push((*row, coefficient as f64)); }
            }
        }
        for (cut, row) in self.solver.cuts.knapsack_cuts.iter().zip(&self.cut.knapsack_rows) {
            if let Some(row) = row {
                let coefficient = cut.coefficient(self.solver.graph.graph_id.max(0), &route.resources);
                if coefficient != 0 { entries.push((*row, coefficient as f64)); }
            }
        }
        if self.cut.rows.iter().all(Option::is_none) {
            return Ok(entries);
        }
        let enumerated = self.is_enumerated(route);
        let path = self.solver.cuts.route(&self.solver.graph, &Self::customers(route))?;
        for (index, row) in self.cut.rows.iter().enumerate() {
            let Some(row) = *row else {
                continue;
            };
            let c = self.solver.cuts.route_coefficient(&self.solver.graph, index, &path, enumerated)?;
            if c != 0 {
                entries.push((row, c as f64));
            }
        }
        Ok(entries)
    }
    fn column(&self, problem: &RootProblem, route: &Route, duals: &[f64], tag: i32) -> Result<Column> {
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
            identity: route.arcs.iter().map(i32::to_string).collect::<Vec<_>>().join(","),
        })
    }
    /// cache_costs: arc reduced costs and the rounded cut duals, active cuts
    /// ordered by rounded dual magnitude then index.
    fn reprice(&mut self, problem: &RootProblem, duals: &[f64]) -> Result<f64> {
        let rounded = self.rounded;
        let offset = problem.reprice(&mut self.solver, duals, rounded, &self.cut.capacity_arc_entries)?;
        for (cut, row) in self.solver.cuts.knapsack_cuts.iter_mut().zip(&self.cut.knapsack_rows) {
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
struct RootController {
    problem: Rc<RootProblem>,
    state: Rc<RefCell<State>>,
}
impl<M: LpMaster> Controller<M> for RootController {
    fn protected_rows(&self) -> Vec<usize> { self.problem.row_map.borrow().clone() }
    fn master_solved(
        &mut self,
        master: &M,
        tags: &[i32],
        artificials: &[usize],
    ) -> Result<()> {
        if artificials.iter().any(|&i| master.values()[i] > 1e-6) {
            return Ok(());
        }
        if master.values().iter().all(|x| (*x - x.round()).abs() <= 1e-6)
            && tags
                .iter()
                .zip(master.values())
                .all(|(&tag, &x)| tag >= 0 || x <= 1e-6)
        {
            let mut state = self.state.borrow_mut();
            for (&tag, &value) in tags.iter().zip(master.values()) {
                if tag < 0 || value <= 1e-6 {
                    continue;
                }
                let route = &state.routes[tag as usize];
                let customers = &route.nodes[1..route.nodes.len() - 1];
                if customers
                    .iter()
                    .copied()
                    .collect::<std::collections::BTreeSet<_>>()
                    .len()
                    != customers.len()
                {
                    return Ok(());
                }
            }
            state.incumbent = state.incumbent.min(master.objective());
        }
        Ok(())
    }
    fn incumbent(&self) -> f64 {
        self.state.borrow().incumbent
    }
    fn enumerated_universe(&mut self, duals: &[f64], bound: f64) -> Result<Option<Vec<Column>>> {
        let mut state = self.state.borrow_mut();
        if !state.enumerate || !bound.is_finite() {
            return Ok(None);
        }
        let upper = if state.incumbent.is_finite() {
            state.incumbent
        } else {
            1e12
        };
        let gap = upper - bound;
        let ratio = gap / state.previous_fixing_gap;
        let fixing_ratio = 0.9f32 as f64;
        if !(ratio < fixing_ratio - (fixing_ratio.abs().max(ratio.abs()) * 1e-10 + 1e-6)) {
            return Ok(None);
        }
        state.previous_fixing_gap = gap;
        let relative = gap / upper.abs();
        let mode = if relative - (1f64.max(relative.abs()) * 1e-10 + 1e-6) > 1. {
            -1
        } else {
            0
        };
        let offset = state.reprice(&self.problem, duals)?;
        let cutoff = gap.max(0.) - offset;
        let fixing_started = std::time::Instant::now();
        let fixed = state.solver.run_fixing_enumeration(mode, 0., cutoff)?;
        state.pricing_seconds += fixing_started.elapsed().as_secs_f64();
        state.fixing_calls += 1;
        if fixed.state != 2 && fixed.state != 3 {
            return Ok(None);
        }
        let routes = state
            .solver
            .enumeration
            .as_ref()
            .unwrap()
            .paths
            .routes(&state.solver.graph)?;
        state.enumerated_routes = routes.len();
        state.cut.enumerated = routes
            .iter()
            .map(|r| {
                let mut arcs = r.arcs.clone();
                arcs.sort_unstable();
                arcs
            })
            .collect();
        let mut columns = vec![];
        for route in routes {
            let tag = register(&mut state, &route)?;
            columns.push(state.column(&self.problem, &route, duals, tag)?);
        }
        Ok(Some(columns))
    }
    fn remap_rows(&mut self, rows: &[Option<usize>]) -> Result<()> {
        for row in self.problem.row_map.borrow_mut().iter_mut() {
            *row = rows[*row].ok_or_else(|| invalid("attempt to remove a base or branch row"))?;
        }
        let mut state = self.state.borrow_mut();
        for index in 0..state.cut.knapsack_rows.len() {
            if let Some(row) = state.cut.knapsack_rows[index] {
                state.cut.knapsack_rows[index] = rows[row];
                if rows[row].is_none() { state.solver.cuts.knapsack_cuts[index].dual = 0.; }
            }
        }
        for row in &mut state.cut.capacity_rows {
            *row = row.and_then(|r| rows[r]);
        }
        for entries in &mut state.cut.capacity_arc_entries {
            *entries = entries.iter().filter_map(|&(r, a)| rows[r].map(|row| (row, a))).collect();
        }
        for index in 0..state.cut.rows.len() {
            let Some(row) = state.cut.rows[index] else {
                continue;
            };
            state.cut.rows[index] = rows[row];
            if rows[row].is_none() {
                state.solver.cuts.retire(index);
            }
        }
        Ok(())
    }
    fn separates(&self) -> bool {
        !self.state.borrow().cut.separators().is_empty()
    }
    fn separate(&mut self, master: &mut M, tags: &[i32]) -> Result<usize> {
        let mut state = self.state.borrow_mut();
        let state = &mut *state;
        let value = master.objective();
        // Every converged column generation, round 0 included, reports to the
        let stop = state.solver.column_generation_terminated(0, 0)?;
        if state.cut.rounds == 0 {
            state.cut.lp_before_cuts = value;
        } else {
            // Alg4EvalByColAndCutGen::eval: the gain is relative to the gap to
            // the incumbent (an integral master solution) when that gap is at
            // most a tenth of the bound, else to a tenth of the bound.
            let previous = state.cut.separation_lp;
            let upper = if state.incumbent.is_finite() { state.incumbent } else { 1e12 };
            let gap = upper - previous;
            let tenth = (0.1 * previous.abs()).max(1e-9);
            let denominator = if gap - (gap.abs().max(tenth) * 1e-10 + 1e-6) > tenth { tenth } else { gap };
            let gain = (value - previous) / denominator;
            let rank_active = state.cut.rank_one_active();
            let min_rounds = state.cut.options.min_rounds_before_stop_by_sp;
            if rank_active && stop && state.cut.rounds >= min_rounds {
                return Ok(0);
            }
            // the solver stores the threshold as a float.
            let threshold = state.cut.options.tailing_off_threshold as f32 as f64;
            if gain < threshold - (threshold.abs().max(gain.abs()) * 1e-10 + 1e-6) {
                if state.cut.options.enabled
                    && state.cut.level <= RANK_ONE_PRIORITY
                    && !state.solver.cuts.separation_phase(&state.solver.options)?.1
                {
                    let options = state.solver.options.clone();
                    state.solver.cuts.increase_phase(&options)?;
                } else {
                    state.cut.tailing += 1;
                }
                // Tailing off at a level moves to the next lower priority, and
                // stops cut generation at the lowest one.
                if state.cut.tailing >= state.cut.options.tailing_off_counter && state.cut.rounds >= min_rounds {
                    match state.cut.next_level() {
                        Some(level) => {
                            state.cut.level = level;
                            state.cut.tailing = 0;
                        }
                        None => return Ok(0),
                    }
                }
            }
        }
        state.cut.separation_lp = value;
        // Each cut round first checks gapSmallerThanTol: separation stops once
        // the bound reaches the incumbent (absolute 1e-6, relative 1e-9).
        if state.incumbent.is_finite() {
            let gap = state.incumbent - value;
            if gap <= value.abs().max(state.incumbent.abs()) * 1e-10 + 1e-6 || gap / state.incumbent.abs() <= 1e-9 {
                return Ok(0);
            }
        }
        let values = master.values().to_vec();
        let tagged: Vec<(usize, usize)> = tags
            .iter()
            .enumerate()
            .filter(|(_, &t)| t >= 0)
            .map(|(i, &t)| (i, t as usize))
            .collect();
        let mut fractional = vec![];
        for &(index, tag) in &tagged {
            if values[index] > 1e-9 {
                let route = &state.routes[tag];
                fractional.push(FractionalColumn {
                    vertices: State::customers(route),
                    value: values[index],
                    graph_id: -1,
                    enumerated: state.is_enumerated(route),
                });
            }
        }
        // Alg4EvalOfNode::addCutToMaster: separators by decreasing priority.
        // Those at or above the level run; a lower one runs, and becomes the
        // level, only while nothing has been added this round.
        let mut level = state.cut.level;
        let mut rows = PendingRows { base: master.rhs().len(), rows: vec![] };
        // Cuts separated this round reach the formulation after the loop.
        let rank_one_in_master = state.cut.rows.iter().any(Option::is_some);
        let mut added = 0;
        let mut success = false;
        for (priority, separator) in state.cut.separators() {
            if priority < level {
                if success {
                    break;
                }
                level = priority;
            }
            let (count, changed) = match separator {
                Separator::Capacity => (self.separate_capacity(state, &mut rows, &tagged, &fractional)?, false),
                Separator::KPath => self.separate_kpath(state, master, &mut rows, &tagged, &fractional)?,
                Separator::Knapsack => (self.separate_knapsack(state, &mut rows, &tagged, &values)?, false),
                Separator::RankOne => self.separate_rank_one(state, master, &mut rows, &tagged, &fractional)?,
                Separator::Relaxation => (0, self.improve_relaxation(state, rank_one_in_master, &tagged, &values)?),
            };
            added += count;
            success |= count > 0 || changed;
        }
        state.cut.level = level;
        // the solver adds the round's cuts to Clp in one call.
        let batch: Vec<_> = rows.rows.iter().map(|(sense, rhs, entries)| (*sense, *rhs, entries.as_slice())).collect();
        master.add_rows(&batch)?;
        if success {
            state.cut.rounds += 1;
        }
        Ok(if added > 0 { added } else { success as usize })
    }
    fn take_removed_columns(&mut self) -> Vec<usize> {
        std::mem::take(&mut self.state.borrow_mut().cut.removed_columns)
    }
}
impl RootController {
    fn separate_capacity(
        &self,
        state: &mut State,
        rows: &mut PendingRows,
        tagged: &[(usize, usize)],
        fractional: &[FractionalColumn],
    ) -> Result<usize> {
        let Some(separator) = state.cut.capacity_separator.as_mut() else {
            return Ok(0);
        };
        let columns: Vec<_> = fractional.iter().map(|c| (c.vertices.clone(), c.value)).collect();
        let mut added = 0;
        for cut in separator.separate(&columns)? {
            let entries: Vec<_> = tagged.iter().filter_map(|&(column, tag)| {
                let coefficient = cut.coefficient(&State::customers(&state.routes[tag]));
                (coefficient != 0).then_some((column, coefficient as f64))
            }).collect();
            let sense = if cut.greater_equal() { RowSense::GreaterEqual } else { RowSense::LessEqual };
            let row = rows.push(sense, cut.rhs as f64, entries);
            for (index, arc) in self.problem.graph.arcs.iter().enumerate() {
                let u = if arc.tail == self.problem.graph.sink { 0 } else { arc.tail };
                let v = if arc.head == self.problem.graph.sink { 0 } else { arc.head };
                let coefficient = cut.edge_coefficient(u, v);
                if coefficient != 0. { state.cut.capacity_arc_entries[index].push((row, coefficient)); }
            }
            state.cut.capacity_rows.push(Some(row));
            state.cut.capacity_cuts.push(cut);
            added += 1;
        }
        Ok(added)
    }
    fn separate_knapsack(
        &self, state: &mut State, rows: &mut PendingRows,
        tagged: &[(usize, usize)], values: &[f64],
    ) -> Result<usize> {
        let graph = state.solver.graph.graph_id.max(0);
        let columns: Vec<_> = tagged.iter().filter(|&&(column, _)| values[column] > 1e-9)
            .map(|&(column, tag)| KnapsackColumn { graph_id: graph,
                resources: state.routes[tag].resources.clone(), value: values[column] }).collect();
        let Some(separator) = &mut state.cut.knapsack_separator else { return Ok(0) };
        let cuts = separator.separate(&columns)?;
        let count = cuts.len();
        for cut in cuts {
            let entries = tagged.iter().filter_map(|&(column, tag)| {
                let coefficient = cut.coefficient(graph, &state.routes[tag].resources);
                (coefficient != 0).then_some((column, coefficient as f64))
            }).collect();
            let row = rows.push(RowSense::LessEqual, cut.cut.rhs as f64, entries);
            state.cut.knapsack_rows.push(Some(row));
            state.solver.cuts.knapsack_cuts.push(cut);
        }
        Ok(count)
    }
    fn separate_kpath<M: LpMaster>(
        &self,
        state: &mut State,
        master: &mut M,
        rows: &mut PendingRows,
        tagged: &[(usize, usize)],
        fractional: &[FractionalColumn],
    ) -> Result<(usize, bool)> {
        let graph_id = state.solver.graph.graph_id.max(0);
        let mut solutions = vec![];
        let mut values = vec![];
        if matches!(state.cut.kpath_separator, Some(RootKPathSeparator::Native { .. })) {
            for &(column, tag) in tagged {
                if master.values()[column] <= 1e-9 { continue; }
                let route = &state.routes[tag];
                solutions.push(crate::cuts::separation::Solution {
                    graph_id, enumerated: state.is_enumerated(route), arc_ids: route.arcs.clone(),
                });
                values.push(master.values()[column]);
            }
        }
        let Some(separator) = state.cut.kpath_separator.as_mut() else { return Ok((0, false)); };
        let before = state.solver.cuts.cuts.len();
        match separator {
            RootKPathSeparator::Standalone(separator) => {
                let vertices = state.solver.graph.vertices.len();
                for cut in separator.separate_fractional(fractional)? {
                    let index = state.solver.cuts.register_strong_k_path_cut(vertices, graph_id, &cut, 0.)?;
                    state.cut.kpath_indices.push(index);
                }
            }
            RootKPathSeparator::Native { separator, graphs } => {
                for cut in separator.separate(&solutions, &values)? {
                    let index = state.solver.cuts.register_native_k_path_cut(graphs, separator.preparation.use_packing_sets, &cut, 0.)?;
                    state.cut.kpath_indices.push(index);
                }
            }
        }
        let after = state.solver.cuts.cuts.len();
        let graph = state.solver.graph.clone();
        let mut paths = vec![];
        for &(_, tag) in tagged {
            let route = &state.routes[tag];
            paths.push((
                state.solver.cuts.route(&graph, &State::customers(route))?,
                state.is_enumerated(route),
            ));
        }
        for index in before..after {
            let mut entries = vec![];
            for (&(column, _), (path, enumerated)) in tagged.iter().zip(&paths) {
                let c = state.solver.cuts.route_coefficient(&graph, index, path, *enumerated)?;
                if c != 0 {
                    entries.push((column, c as f64));
                }
            }
            let cut = &state.solver.cuts.cuts[index];
            let row = rows.push(RowSense::GreaterEqual, cut.rhs as f64, entries);
            state.cut.rows.push(Some(row));
            state.cut.memories.push(cut.memory_for_graph(graph_id));
        }
        let mut changed = false;
        for index in 0..before {
            let Some(row) = state.cut.rows[index] else {
                continue;
            };
            let memory = state.solver.cuts.cuts[index].memory_for_graph(graph_id);
            if memory == state.cut.memories[index] {
                continue;
            }
            changed = true;
            state.cut.memories[index] = memory;
            for (&(column, _), (path, enumerated)) in tagged.iter().zip(&paths) {
                let c = state.solver.cuts.route_coefficient(&graph, index, path, *enumerated)?;
                master.set_coefficient(row, column, c as f64)?;
            }
        }
        Ok((after - before, changed))
    }
    /// New rank-one cuts, and whether existing cut memories changed.
    fn separate_rank_one<M: LpMaster>(
        &self,
        state: &mut State,
        master: &mut M,
        rows: &mut PendingRows,
        tagged: &[(usize, usize)],
        fractional: &[FractionalColumn],
    ) -> Result<(usize, bool)> {
        let before = state.solver.cuts.cuts.len();
        let options = state.solver.options.clone();
        let graph = state.solver.graph.clone();
        state
            .solver
            .cuts
            .separate_default(&graph, &options, &self.problem.customers, fractional)?;
        let after = state.solver.cuts.cuts.len();
        let mut paths = vec![];
        for &(_, tag) in tagged {
            let route = &state.routes[tag];
            paths.push((state.solver.cuts.route(&graph, &State::customers(route))?, state.is_enumerated(route)));
        }
        for index in before..after {
            let mut entries = vec![];
            for (&(column, _), (path, enumerated)) in tagged.iter().zip(&paths) {
                let c = state.solver.cuts.route_coefficient(&graph, index, path, *enumerated)?;
                if c != 0 {
                    entries.push((column, c as f64));
                }
            }
            let cut = &state.solver.cuts.cuts[index];
            let sense = if state.solver.cuts.is_covering(index) {
                RowSense::GreaterEqual
            } else {
                RowSense::LessEqual
            };
            let row = rows.push(sense, cut.rhs as f64, entries);
            state.cut.rows.push(Some(row));
            state.cut.memories.push(cut.memory_for_graph(graph.graph_id.max(0)));
        }
        let mut changed = false;
        for index in 0..before {
            let Some(row) = state.cut.rows[index] else {
                continue;
            };
            let memory = state.solver.cuts.cuts[index].memory_for_graph(graph.graph_id.max(0));
            if memory == state.cut.memories[index] {
                continue;
            }
            changed = true;
            state.cut.memories[index] = memory;
            for (&(column, _), (path, enumerated)) in tagged.iter().zip(&paths) {
                let c = state.solver.cuts.route_coefficient(&graph, index, path, *enumerated)?;
                master.set_coefficient(row, column, c as f64)?;
            }
        }
        Ok((after - before, changed))
    }
    /// Dynamic NG (runColGenSpRelaxationImprovement and
    /// BcRCSPFunctor::improveCurrentSpRelaxation): strengthen the NG memories
    /// from the cycles of the routes in the LP solution, then mark the master
    /// columns the stronger relaxation forbids for removal. A negative
    /// RCSPdynamicNGmode only strengthens while no rank-one cut is in the
    /// master (cuts of the current round do not count yet).
    fn improve_relaxation(
        &self,
        state: &mut State,
        rank_one_in_master: bool,
        tagged: &[(usize, usize)],
        values: &[f64],
    ) -> Result<bool> {
        let option = |name: &str| state.solver.options.get(name);
        let mode = option("RCSPdynamicNGmode")? as i32;
        if mode == 0 || (mode < 0 && rank_one_in_master) {
            return Ok(false);
        }
        let controls = crate::graph::relaxation::Controls {
            mode: mode.abs(),
            max_vertex_size: option("RCSPmaxNGneighbourhoodSize")?.max(0.) as usize,
            max_average_size: option("RCSPmaxNGaverNeighbourhoodSize")?.max(0.) as usize,
            max_columns: option("RCSPmaxNumOfColumnsInDynamicNG")?.max(0.) as usize,
            max_cycle_size: option("RCSPmaxCycleSizeInDynamicNG")? as i32,
            mirror: false,
        };
        let paths: Vec<_> = tagged
            .iter()
            .filter(|&&(column, _)| values[column] > 1e-9)
            .map(|&(column, tag)| crate::graph::relaxation::FractionalPath {
                arcs: state.routes[tag].arcs.clone(),
                value: values[column],
            })
            .collect();
        let counts = state.solver.strengthen_ng(&paths, &controls)?;
        if std::env::var_os("BALDES_DEBUG_NG").is_some() {
            eprintln!("dynamic NG: columns={} paths={} forbidden={} steps={:?}", values.len(), paths.len(), counts.columns_forbidden,
                state.solver.engine.as_ref().map(|e| e.grid.step));
        }
        if counts.columns_forbidden == 0 {
            return Ok(false);
        }
        let mut removed = vec![];
        for &(column, tag) in tagged {
            let route = &state.routes[tag];
            if !state.solver.route_satisfies_relaxation(&route.nodes, &route.arcs)? {
                removed.push(column);
            }
        }
        state.cut.removed_columns = removed;
        Ok(true)
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
pub fn inherit(parent:&NodeSnapshot, branches:&[(usize,RowSense,f64)], options:RootOptions) -> Result<RootOutput> {
    let (problem,master,warm)=parent.fork(branches,options.cg.artificial_cost)?;
    solve_seeded(problem,options,Some((parent,master,warm)))
}
fn solve_seeded(problem: RootProblem, mut options: RootOptions,
    seed: Option<(&NodeSnapshot,ClpMaster,column_generation::WarmStart)>) -> Result<RootOutput> {
    let solve_started = std::time::Instant::now();
    problem.graph.validate()?;
    let (saved,mut master)=if let Some((saved,master,warm))=seed {
        options.cg.warm_start=Some(warm);(Some(saved),master)
    } else {(None,ClpMaster::new()?)};
    // the solver leaves Clp's default 1e-7 primal and dual tolerances in place.
    master.set_tolerances(1e-7, 1e-7)?;
    master.zero_tolerance = 1e-7;
    if saved.is_none() {
        for row in &problem.rows { master.add_row(row.sense, row.rhs, &[])?; }
    }
    let mut solver = Solver::new(problem.graph.vertices.len(), problem.graph.resources.len())?;
    solver.graph = saved.map_or_else(||problem.graph.clone(),|s|s.pricing_graph.clone());
    if let Some(saved)=saved { solver.cuts=saved.registry.clone(); }
    for (name, value) in &options.pricing_options {
        solver.set_option(name, *value)?;
    }
    let phase_floor = options.pricing_phase_floor;
    if phase_floor >= solver.phases.len() { return Err(invalid("pricing phase floor outside schedule")); }
    let schedule = solver.phases[phase_floor..].to_vec();
    let capacity_separator = if saved.is_some() { None } else if let Some(config) = options.capacity_cuts.take() {
        if problem.graph.source != 0 || problem.graph.sink != config.demands.len()
            || problem.graph.vertices.len() != config.demands.len() + 1 {
            return Err(invalid("capacity cuts require depot 0, customers 1..n, sink n+1"));
        }
        let mut separator = CapacitySeparator::new(config.demands, config.capacity)?;
        separator.parameters = config.parameters;
        if let Some(windows) = config.time_windows { separator.set_time_windows(windows)?; }
        Some(separator)
    } else { None };
    if saved.is_none() && options.kpath_cuts.is_some() && options.native_kpath_cuts.is_some() {
        return Err(invalid("select standalone or native k-path configuration"));
    }
    let kpath_separator = if saved.is_some() { None } else if let Some(config) = options.native_kpath_cuts.take() {
        let graph_id = problem.graph.graph_id.max(0);
        let input = config.graphs.iter().find(|g| g.graph.id == graph_id)
            .ok_or_else(|| invalid("native k-path configuration has no pricing graph"))?;
        if input.graph.source != problem.graph.source as i32 || input.graph.sink != problem.graph.sink as i32
            || input.graph.vertices.iter().any(|v| v.id < 0 || v.id as usize >= problem.graph.vertices.len()) {
            return Err(invalid("native k-path pricing vertices/terminals differ"));
        }
        for arc in &problem.graph.arcs {
            if !input.graph.arcs.iter().any(|a| a.id == arc.id && a.tail == arc.tail as i32 && a.head == arc.head as i32) {
                return Err(invalid("native k-path pricing arc incidence differs"));
            }
        }
        let mut separator = NativeKPathSeparator::new(config.use_packing_sets);
        separator.parameters = config.parameters;
        separator.prepare(&config.graphs, &config.demands, config.capacity, config.two_path_resource)?;
        Some(RootKPathSeparator::Native { separator, graphs: config.graphs })
    } else if let Some(config) = options.kpath_cuts.take() {
        if problem.graph.source != 0 || problem.graph.sink != config.demands.len()
            || problem.graph.vertices.len() != config.demands.len() + 1 {
            return Err(invalid("k-path cuts require depot 0, customers 1..n, sink n+1"));
        }
        let mut separator = KPathSeparator::new(config.demands, config.capacity)?;
        separator.parameters = config.parameters;
        if let Some(windows) = config.time_windows { separator.set_time_windows(windows)?; }
        Some(RootKPathSeparator::Standalone(separator))
    } else { None };
    let knapsack_separator = if saved.is_some() { None } else if let Some(mut config) = options.knapsack_cuts.take() {
        let graph = problem.graph.graph_id.max(0);
        for row in &config.rows {
            let Some(&(resource, scale)) = row.graph_resources.get(&graph) else {
                return Err(invalid("knapsack original row has no resource for pricing graph"));
            };
            if resource >= problem.graph.resources.len() || !scale.is_finite() || scale <= 0. {
                return Err(invalid("knapsack original row resource/scaling"));
            }
            // the solver applies scale to separation/master coefficients, while
            // active-dual retrieval retains the integer cut breakpoints.
        }
        for (name, value) in &options.pricing_options {
            match name.as_str() {
                "RCSPresConsKnapsackCutsMode" => { config.parameters.use_rounding = *value > 0.; config.parameters.max_one_k = value.abs() as i32; }
                "RCSPresConsKnapsackCutsMaxNumPerRound" => config.parameters.max_cuts_per_round = *value as usize,
                _ => {}
            }
        }
        crate::cuts::knapsack::KnapsackSeparator::with_parameters(config.parameters.clone())?;
        Some(ResourceKnapsackSeparator { rows: config.rows, parameters: config.parameters, ..Default::default() })
    } else { None };
    let arc_count = problem.graph.arcs.len();
    let state = Rc::new(RefCell::new(State {
        solver,
        routes: saved.map_or_else(Vec::new,|s|s.routes.clone()),
        route_tags: saved.map_or_else(Default::default,|s|s.routes.iter().enumerate().map(|(i,r)|(r.arcs.clone(),i as i32)).collect()),
        incumbent: options.incumbent,
        enumerate: options.enumerate,
        rounded: options.round_reduced_costs,
        previous_fixing_gap: 1e12,
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
            capacity_priority: if options.node_depth == 0 { CAPACITY_PRIORITY } else { 1. },
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
        if let Some(saved)=saved {
            s.cut=saved.cut.clone();
            s.cut.options=options.cuts.clone();
            s.cut.capacity_priority=if options.node_depth==0 {CAPACITY_PRIORITY} else {1.};
            s.cut.relaxation_priority=options.relaxation_improvement_priority as f64;
            s.cut.rounds=0;s.cut.tailing=0;s.cut.separation_lp=f64::NAN;
            s.cut.lp_before_cuts=f64::NAN;s.cut.removed_columns.clear();
        }
        s.cut.level = s.cut.separators().first().map_or(0., |&(priority, _)| priority);
    }
    let zeros = vec![0.; problem.rows.len()];
    options.cg.known.clear();
    let mut live:HashSet<i32>=options.cg.warm_start.as_ref().map_or_else(HashSet::new,|w|
        w.columns.iter().flatten().map(|c|c.tag).collect());
    for route in &options.initial_routes {
        let mut s = state.borrow_mut();
        let tag = register(&mut s, route)?;
        if !live.insert(tag) {continue;}
        let column = s.column(&problem, route, &zeros, tag)?;
        master.add_column(column.cost, &column.entries)?;
        if let Some(warm)=options.cg.warm_start.as_mut() {warm.columns.push(Some(column));}
        else {options.cg.known.push(column);}
    }
    let problem = Rc::new(problem);
    let mut controller = RootController {
        problem: problem.clone(),
        state: state.clone(),
    };
    let initial_rows = options.cg.warm_start.as_ref().map_or(problem.rows.len(), |w| w.initial_rows);
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
            let result = state.solver.price_phase(phase + phase_floor, maximum, false)?;
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
            registry: state.solver.cuts.clone(),
            pricing_graph: state.solver.graph.clone(),
            routes: state.routes.clone(),
            warm: output.warm_start.clone(),
        })
    });
    Ok(RootOutput {
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
        cuts: state.solver.cuts.cuts.len() + state.cut.capacity_cuts.len() + state.solver.cuts.knapsack_cuts.len(),
        lp_before_cuts: state.cut.lp_before_cuts,
        fixing_calls: state.fixing_calls,
        enumerated_routes: state.enumerated_routes,
    })
}
