//! Best-bound branch-and-price for a routing set-partitioning master.
//!
//! Vehicle counts, packing-set assignments and directed arc-flow rows are
//! priced through `RootProblem::arc_entries`. Ryan–Foster rows additionally
//! install native binary resources in labeling and filter enumeration.
//! Each child forks the parent's solved Clp model (with cuts intact) and
//! carries the cut registry forward; feasible elementary columns are shared
//! between nodes. Bounds require exact pricing or a proved MIP over complete
//! enumeration; restricted-master heuristic objectives do not certify nodes.
//! Phased strong branching and diving
//! use isolated node evaluations via fork-and-carry.
use super::{
    column_generation::Status,
    root::{self, RootOptions, RootProblem},
    RowSense,
};
use crate::{invalid, Result, Route};
use std::collections::{BTreeMap, HashSet};
use std::rc::Rc;
use std::time::{Duration, Instant};
pub mod candidates;
mod diving;
#[cfg(feature = "highs")]
mod integer;
pub use super::branching::{BranchConstraint, BranchingOptions};
pub mod strong;
pub use diving::DivingOptions;
pub use strong::{StrongBranchingOptions, StrongPhase};

#[derive(Clone, Debug)]
pub struct TreeOptions {
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
    pub absolute_gap: f64,
    pub relative_gap: f64,
    pub strong_branching: Option<StrongBranchingOptions>,
    pub diving: Option<DivingOptions>,
}
impl Default for TreeOptions {
    fn default() -> Self {
        Self {
            root: RootOptions::default(),
            branching: BranchingOptions::default(),
            node_limit: 10000,
            time_limit: Duration::from_secs(120),
            mip_node_limit: 10000,
            mip_time_limit: Duration::from_millis(100),
            enumerated_mip: None,
            absolute_gap: 1e-6,
            relative_gap: 1e-9,
            strong_branching: Some(StrongBranchingOptions::default()),
            diving: Some(DivingOptions::default()),
        }
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TreeStatus {
    Optimal,
    TimeLimit,
    NodeLimit,
    Incomplete,
}
#[derive(Debug)]
pub struct TreeOutput {
    pub status: TreeStatus,
    pub incumbent: f64,
    pub bound: f64,
    pub routes: Vec<Route>,
    pub nodes: usize,
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
        format!("{{\"instance\":{:?},\"status\":\"{:?}\",\"incumbent\":{},\"bound\":{},\"nodes\":{},\"open_nodes\":{},\"pricing_calls\":{},\"fixing_calls\":{},\"enumerated_routes\":{},\"mip_nodes\":{},\"enumerated_mip_calls\":{},\"enumerated_mip_nodes\":{},\"strong_branching_trials\":{},\"diving_nodes\":{},\"seconds\":{},\"routes\":[{}]}}\n",
            name, self.status, number(self.incumbent), number(self.bound), self.nodes,
            self.open_nodes, self.pricing_calls, self.fixing_calls, self.enumerated_routes, self.mip_nodes, self.enumerated_mip_calls, self.enumerated_mip_nodes, self.strong_branching_trials, self.diving_nodes, self.seconds, routes.join(","))
    }
}
type Branch = BranchConstraint;
struct Node {
    branches: Vec<Branch>,
    bound: f64,
    parent: Option<Rc<root::NodeSnapshot>>,
}
struct Pool {
    routes: Vec<Route>,
    customers: Vec<Vec<usize>>,
    seen: HashSet<Vec<i32>>,
}
impl Pool {
    fn add(&mut self, problem: &RootProblem, arcs: &[i32]) {
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
fn feasible_route(problem: &RootProblem, ids: &[i32]) -> Option<Route> {
    problem.elementary_route(ids)
}
struct IntegerSearch<'a> {
    pool: &'a Pool,
    by_customer: Vec<Vec<usize>>,
    customers: &'a [usize],
    fleet: usize,
    used: Vec<bool>,
    chosen: Vec<usize>,
    best: &'a mut f64,
    solution: &'a mut Vec<Route>,
    nodes: usize,
    limit: usize,
    deadline: Instant,
}
impl IntegerSearch<'_> {
    fn search(&mut self, cost: f64) {
        if self.nodes >= self.limit || Instant::now() >= self.deadline || cost >= *self.best {
            return;
        }
        self.nodes += 1;
        let mut selected = None;
        let mut count = usize::MAX;
        for &v in self.customers {
            if self.used[v] {
                continue;
            }
            let available = self.by_customer[v]
                .iter()
                .filter(|&&r| self.pool.customers[r].iter().all(|&c| !self.used[c]))
                .count();
            if available == 0 {
                return;
            }
            if available < count {
                count = available;
                selected = Some(v);
            }
        }
        let Some(v) = selected else {
            *self.best = cost;
            *self.solution = self
                .chosen
                .iter()
                .map(|&r| self.pool.routes[r].clone())
                .collect();
            return;
        };
        if self.chosen.len() >= self.fleet {
            return;
        }
        // Binary set partitioning: branch on the route covering the most
        // constrained uncovered customer. This enumeration is only a heuristic
        // for the full problem because its column universe is restricted.
        for i in 0..self.by_customer[v].len() {
            let r = self.by_customer[v][i];
            if self.pool.customers[r].iter().any(|&c| self.used[c]) {
                continue;
            }
            for &c in &self.pool.customers[r] {
                self.used[c] = true;
            }
            self.chosen.push(r);
            self.search(cost + self.pool.routes[r].original_cost);
            self.chosen.pop();
            for &c in &self.pool.customers[r] {
                self.used[c] = false;
            }
            if self.nodes >= self.limit || Instant::now() >= self.deadline {
                break;
            }
        }
    }
}
fn integer_heuristic(
    problem: &RootProblem,
    pool: &Pool,
    best: &mut f64,
    solution: &mut Vec<Route>,
    options: &TreeOptions,
    deadline: Instant,
) -> usize {
    if options.mip_node_limit == 0 {
        return 0;
    }
    let mut by_customer = vec![vec![]; problem.graph.vertices.len()];
    for (r, customers) in pool.customers.iter().enumerate() {
        for &v in customers {
            by_customer[v].push(r);
        }
    }
    for rows in &mut by_customer {
        rows.sort_by(|&a, &b| {
            (pool.routes[a].original_cost / pool.customers[a].len() as f64)
                .total_cmp(&(pool.routes[b].original_cost / pool.customers[b].len() as f64))
                .then(a.cmp(&b))
        });
    }
    let mut search = IntegerSearch {
        pool,
        by_customer,
        customers: &problem.customers,
        fleet: problem.rows[0].rhs.floor() as usize,
        used: vec![false; problem.graph.vertices.len()],
        chosen: vec![],
        best,
        solution,
        nodes: 0,
        limit: options.mip_node_limit,
        deadline: deadline.min(Instant::now() + options.mip_time_limit),
    };
    search.search(0.);
    search.nodes
}
fn closed(bound: f64, incumbent: f64, options: &TreeOptions) -> bool {
    incumbent.is_finite()
        && bound
            >= incumbent
                - options.absolute_gap
                - options.relative_gap * incumbent.abs().max(bound.abs())
}
fn branch_problem(problem: &RootProblem, branches: &[Branch]) -> Result<RootProblem> {
    let mut child = problem.clone();
    for branch in branches {
        child.append_constraint(branch.clone(), child.rows.len())?;
    }
    Ok(child)
}
fn budget_error(error: &crate::core::error::OrError) -> bool {
    error.message.contains("time limit") || error.message.contains("label limit")
}
fn evaluate(
    parent: Option<&Rc<root::NodeSnapshot>>,
    problem: &RootProblem,
    branches: &[Branch],
    pool: &Pool,
    options: &TreeOptions,
    incumbent: f64,
    deadline: Instant,
    max_iterations: usize,
    floor: usize,
    capture_state: bool,
) -> Result<root::RootOutput> {
    let mut controls = options.root.clone();
    controls.capture_state = capture_state;
    controls.node_depth = branches.len();
    controls.pricing_phase_floor = floor;
    controls.cg.max_iterations = max_iterations;
    controls.cg.time_limit = deadline.saturating_duration_since(Instant::now());
    controls.cg.multiplicity = problem.customers.len() as f64;
    controls.initial_routes = pool.routes.clone();
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
        root::inherit_branches(parent, &branches[depth..], controls)
    } else {
        root::solve(branch_problem(problem, branches)?, controls)
    }
}
/// Solve a standard routing set-partitioning problem. `root.initial_routes`
/// may seed incumbents; the cost-only `root.incumbent` is not used without a
/// feasible witness. Limits preserve the bound of every unfinished node.
pub fn solve(problem: RootProblem, mut options: TreeOptions) -> Result<TreeOutput> {
    #[cfg(not(feature = "highs"))]
    if options.enumerated_mip.is_some() {
        return Err(invalid("enumerated MIP requires the highs feature"));
    }
    options.root.normalize_pricing_options()?;
    let started = Instant::now();
    if !options.absolute_gap.is_finite()
        || options.absolute_gap < 0.
        || !options.relative_gap.is_finite()
        || options.relative_gap < 0.
    {
        return Err(invalid(
            "tree gap tolerances must be finite and nonnegative",
        ));
    }
    problem.graph.validate()?;
    let standard = RootProblem::set_partitioning(
        problem.graph.clone(),
        problem
            .rows
            .first()
            .ok_or_else(|| invalid("tree requires set partitioning rows"))?
            .rhs,
    )?;
    if problem.rows.len() != standard.rows.len()
        || problem
            .rows
            .iter()
            .zip(&standard.rows)
            .any(|(a, b)| a.sense != b.sense || a.rhs != b.rhs)
        || problem.arc_entries != standard.arc_entries
        || problem.route_entries != standard.route_entries
        || problem.customers != standard.customers
        || problem.graph.arcs.iter().any(|a| a.cost < 0.)
        || problem
            .graph
            .arcs
            .iter()
            .map(|a| a.id)
            .collect::<HashSet<_>>()
            .len()
            != problem.graph.arcs.len()
    {
        return Err(invalid("tree requires a routing set-partitioning master with nonnegative costs and unique arc ids"));
    }
    let deadline = started + options.time_limit;
    let mut output = TreeOutput {
        status: TreeStatus::NodeLimit,
        incumbent: f64::INFINITY,
        bound: 0.,
        routes: vec![],
        nodes: 0,
        open_nodes: 1,
        pricing_calls: 0,
        fixing_calls: 0,
        enumerated_routes: 0,
        mip_nodes: 0,
        enumerated_mip_calls: 0,
        enumerated_mip_nodes: 0,
        strong_branching_trials: 0,
        diving_nodes: 0,
        seconds: 0.,
    };
    let mut pool = Pool {
        routes: vec![],
        customers: vec![],
        seen: HashSet::new(),
    };
    for route in &options.root.initial_routes {
        pool.add(&problem, &route.arcs);
    }
    for &v in &problem.customers {
        let a = problem
            .graph
            .arcs
            .iter()
            .find(|a| a.tail == problem.graph.source && a.head == v);
        let b = problem
            .graph
            .arcs
            .iter()
            .find(|a| a.tail == v && a.head == problem.graph.sink);
        if let (Some(a), Some(b)) = (a, b) {
            pool.add(&problem, &[a.id, b.id]);
        }
    }
    output.mip_nodes += integer_heuristic(
        &problem,
        &pool,
        &mut output.incumbent,
        &mut output.routes,
        &options,
        deadline,
    );
    let mut open = vec![Node {
        branches: vec![],
        bound: 0.,
        parent: None,
    }];
    let mut unresolved = vec![];
    while !open.is_empty() {
        // Discard only nodes whose certified bound closes the global gap.
        open.retain(|n| !closed(n.bound, output.incumbent, &options));
        if open.is_empty() {
            break;
        }
        if Instant::now() >= deadline {
            output.status = TreeStatus::TimeLimit;
            break;
        }
        if output.nodes >= options.node_limit {
            output.status = TreeStatus::NodeLimit;
            break;
        }
        let index = (0..open.len())
            .min_by(|&a, &b| open[a].bound.total_cmp(&open[b].bound))
            .unwrap();
        let mut node = open.remove(index);
        let parent_ref = node.parent.as_ref();
        let result = match evaluate(
            parent_ref,
            &problem,
            &node.branches,
            &pool,
            &options,
            output.incumbent,
            deadline,
            options.root.cg.max_iterations,
            0,
            true,
        ) {
            Ok(result) => result,
            Err(error) if budget_error(&error) => {
                open.push(node);
                output.status = if Instant::now() >= deadline {
                    TreeStatus::TimeLimit
                } else {
                    TreeStatus::Incomplete
                };
                break;
            }
            Err(error) => return Err(error),
        };
        output.nodes += 1;
        output.pricing_calls += result.master.pricing_calls;
        output.fixing_calls += result.fixing_calls;
        output.enumerated_routes += result.enumerated_routes;
        // The LP objective may include artificials or come from a heuristic
        // pricing phase. Only the pricing certificate is a node lower bound.
        if result.master.bound.is_finite() {
            node.bound = node.bound.max(result.master.bound);
        }
        for route in &result.routes {
            pool.add(&problem, &route.arcs);
        }
        diving::record_solution(&problem, &result, &mut output);
        output.mip_nodes += integer_heuristic(
            &problem,
            &pool,
            &mut output.incumbent,
            &mut output.routes,
            &options,
            deadline,
        );
        #[cfg(feature = "highs")]
        integer::run(
            result.snapshot.as_ref(),
            &problem,
            &mut node,
            &options,
            &mut output,
            deadline,
        )?;
        if closed(node.bound, output.incumbent, &options) {
            continue;
        }
        if result.master.status != Status::Optimal
            || result
                .master
                .artificials
                .iter()
                .any(|&c| result.values.get(c).copied().unwrap_or(0.) > 1e-6)
        {
            // A finite artificial penalty does not prove true infeasibility.
            unresolved.push(node);
            continue;
        }
        let mut flows = BTreeMap::<usize, f64>::new();
        for (&tag, &value) in result.master.column_tags.iter().zip(&result.values) {
            if tag < 0 || value <= 0. {
                continue;
            }
            for &id in &result.routes[tag as usize].arcs {
                *flows.entry(problem.arc_position(id).unwrap()).or_default() += value;
            }
        }
        let candidates = candidates::candidates_with_history(
            &problem,
            &result,
            &options.branching,
            &node.branches,
        );
        // Capture this node's solved snapshot before moving result into diving.
        let parent_snapshot = result.snapshot.clone();
        diving::run(
            &problem,
            &node,
            result,
            &mut pool,
            &options,
            &mut output,
            deadline,
        )?;
        if closed(node.bound, output.incumbent, &options) {
            continue;
        }
        let branch = strong::choose(
            parent_snapshot.as_ref(),
            &problem,
            &node,
            candidates,
            &mut pool,
            &options,
            &mut output,
            deadline,
        )?;
        if let Some((candidate, bounds)) = branch {
            for (side, constraint) in [candidate.down, candidate.up].into_iter().enumerate() {
                let mut branches = node.branches.clone();
                branches.push(constraint);
                open.push(Node {
                    branches,
                    bound: bounds[side],
                    parent: parent_snapshot.clone(),
                });
            }
        } else {
            // Integral aggregate arcs identify depot-to-depot paths even if
            // identical route variables split their weights fractionally.
            let mut used = HashSet::new();
            let mut routes = vec![];
            for (&a, &x) in &flows {
                if x < 0.5 || problem.graph.arcs[a].tail != problem.graph.source {
                    continue;
                }
                let mut path = vec![];
                let mut next = a;
                loop {
                    if !used.insert(next) {
                        break;
                    }
                    path.push(problem.graph.arcs[next].id);
                    let head = problem.graph.arcs[next].head;
                    if head == problem.graph.sink {
                        break;
                    }
                    let Some((&b, _)) = flows
                        .iter()
                        .find(|(b, y)| **y > 0.5 && problem.graph.arcs[**b].tail == head)
                    else {
                        break;
                    };
                    next = b;
                }
                if let Some(route) = feasible_route(&problem, &path) {
                    routes.push(route);
                }
            }
            let mut cover = vec![0; problem.graph.vertices.len()];
            for route in &routes {
                for &v in &route.nodes[1..route.nodes.len() - 1] {
                    cover[v as usize] += 1;
                }
            }
            let cost = routes.iter().map(|r| r.original_cost).sum::<f64>();
            if problem.customers.iter().all(|&v| cover[v] == 1)
                && routes.len() as f64 <= problem.rows[0].rhs
                && cost < output.incumbent
            {
                output.incumbent = cost;
                output.routes = routes;
            }
            if !closed(node.bound, output.incumbent, &options) {
                unresolved.push(node);
            }
        }
    }
    unresolved.retain(|n| !closed(n.bound, output.incumbent, &options));
    if open.is_empty() && !unresolved.is_empty() {
        output.status = TreeStatus::Incomplete;
    }
    open.extend(unresolved);
    output.open_nodes = open.len();
    if open.is_empty() && output.incumbent.is_finite() {
        output.status = TreeStatus::Optimal;
        output.bound = output.incumbent;
    } else {
        output.bound = open
            .iter()
            .map(|n| n.bound)
            .fold(output.incumbent, f64::min);
    }
    output.seconds = started.elapsed().as_secs_f64();
    Ok(output)
}
