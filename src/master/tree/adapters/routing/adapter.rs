//! Routing node evaluation and branch hooks for the shared tree driver.
use super::*;
use crate::master::tree::{Decision, Limits, Model, QueuedNode, Stop, TreeFrontier};

pub(super) struct RoutingSearch<'a, const D: usize> {
    pub problem: &'a RoutingContext<D>,
    pub options: &'a TreeOptions,
    pub output: &'a mut TreeOutput,
    pub pool: &'a mut Pool,
    pub workers: Option<&'a rayon::ThreadPool>,
    pub deadline: Instant,
    pub previous: &'a mut Option<Rc<routing::NodeSnapshot<D>>>,
    pub subtree_history: &'a mut Option<strong::SubtreeHistory>,
    pub objective_lattice: &'a Option<objective_lattice::ObjectiveLattice>,
    pub closed_bound: &'a mut f64,
    pub unresolved: &'a mut Vec<Node<D>>,
}
impl<const D: usize> Model for RoutingSearch<'_, D> {
    type Node = Node<D>;
    type Output = routing::RootOutput<D>;
    fn limits(&self) -> Limits {
        Limits {
            deadline: self.deadline,
            remaining_nodes: self.options.node_limit.saturating_sub(self.output.nodes),
            batch_size: self.options.threads,
            time_first: true,
        }
    }
    fn prune(&mut self, frontier: &mut TreeFrontier<Self::Node, Self::Output>) {
        frontier.retain(|node| {
            if closed(node.bound, self.output.incumbent, self.options) {
                if let Some(history) = self.subtree_history.as_mut() {
                    history.close(node.lineage);
                }
                *self.closed_bound = self.closed_bound.min(node.bound);
                false
            } else {
                true
            }
        });
    }
    fn evaluate(
        &mut self,
        entries: &mut [QueuedNode<Self::Node, Self::Output>],
        _frontier: &mut crate::master::tree::TreeFrontier<Self::Node, Self::Output>,
    ) -> Result<Vec<Result<Self::Output>>> {
        if let Some(workers) = self.workers.filter(|_| entries.len() > 1) {
            self.output.parallel_batches += 1;
            // Worker transfer owns a batch; reconstruct it without cloning
            // solver state by borrowing the queued node payloads below.
            parallel::evaluate_entries(
                workers,
                self.problem,
                entries,
                self.options,
                self.output.incumbent,
                self.deadline,
            )
        } else {
            let node = &entries[0].node;
            Ok(vec![evaluate(
                node.parent.as_ref(),
                self.previous.as_deref(),
                self.problem,
                &node.branches,
                self.pool,
                self.options,
                self.output.incumbent,
                self.deadline,
                self.options.root.cg.max_iterations,
                0,
                true,
            )])
        }
    }
    fn consume(
        &mut self,
        entry: Option<QueuedNode<Self::Node, Self::Output>>,
        result: Result<Self::Output>,
        _reused: bool,
        open: &mut TreeFrontier<Self::Node, Self::Output>,
    ) -> Result<Decision> {
        let mut node = entry
            .ok_or_else(|| invalid("routing tree requires a node payload"))?
            .node;
        let result = match result {
            Ok(result) => result,
            Err(error) if budget_error(&error) => {
                open.push_evaluated(node, 0, None)?;
                return Ok(Decision::Stop {
                    reason: if Instant::now() >= self.deadline {
                        Stop::TimeLimit
                    } else {
                        Stop::Incomplete
                    },
                    restore_ready: true,
                });
            }
            Err(error) => return Err(error),
        };
        let problem = self.problem;
        let options = self.options;
        let mut output = &mut *self.output;
        let mut pool = &mut *self.pool;
        let previous = &mut *self.previous;
        let mut subtree_history = &mut *self.subtree_history;
        let objective_lattice = self.objective_lattice;
        let closed_bound = &mut *self.closed_bound;
        let unresolved = &mut *self.unresolved;
        let deadline = self.deadline;
        output.nodes += 1;
        if options.branch_history.is_some()
            && result.master.status == Status::Optimal
            && !result.master.pruned_by_bound
            && !result.preprocessing.infeasible
            && result
                .master
                .artificials
                .iter()
                .all(|&c| result.values.get(c).copied().unwrap_or(0.) <= 1e-6)
        {
            if let Some((candidate, side, parent_bound)) = &node.observation {
                output.branch_history.record_at_stage(
                    candidate,
                    *side,
                    node.branches.len().saturating_sub(1),
                    history::EvaluationKind::NodePricing,
                    None,
                    *parent_bound,
                    result.master.lp,
                    result.solver_seconds,
                );
            }
        }
        if let Some(snapshot) = &result.snapshot {
            *previous = Some(snapshot.clone());
        }
        output.pricing_calls += result.master.pricing_calls;
        output.fixing_calls += result.fixing_calls;
        output.enumerated_routes += result.enumerated_routes;
        if result.preprocessing.infeasible
            || (result.master.status == Status::Infeasible && result.master.bound == f64::INFINITY)
        {
            if let Some(history) = &mut subtree_history {
                history.close(node.lineage);
            }
            return Ok(Decision::Continue);
        }
        // The LP objective may include artificials or come from a heuristic
        // pricing phase. Only the selected bound is a node lower bound:
        // numerical estimates never enter certified-mode node evaluations.
        if result.master.bound.is_finite() {
            node.bound = node.bound.max(result.master.bound);
            if let Some(lattice) = &objective_lattice {
                node.bound = lattice.strengthen(node.bound);
            }
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
            if let Some(history) = &mut subtree_history {
                history.close(node.lineage);
            }
            *closed_bound = closed_bound.min(node.bound);
            return Ok(Decision::Continue);
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
            return Ok(Decision::Continue);
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
        let legacy = options
            .branch_history
            .as_ref()
            .is_some_and(|h| h.policy == BranchHistoryPolicy::Reliable);
        let mut quota = options
            .strong_branching
            .as_ref()
            .map_or(usize::MAX, |s| s.phases.first().map_or(0, |p| p.candidates));
        let mut budget = options
            .strong_branching
            .as_ref()
            .filter(|_| !legacy)
            .map(|strong| {
                let average = subtree_history
                    .as_ref()
                    .and_then(|history| history.mean(node.branches.len()));
                let mut estimate = node.subtree;
                if let Some(size) = average {
                    estimate.size = size;
                }
                let mut budget = strong::CandidateBudget::new(
                    quota,
                    strong
                        .phases
                        .first()
                        .map_or(0., |phase| phase.subtree_ratio),
                    estimate,
                );
                if average.is_some() {
                    budget.use_history();
                }
                budget
            });
        if let Some(budget) = &budget {
            quota = budget.quota;
        }
        let mut candidates = if legacy {
            candidates::legacy_candidates_with_history(
                &problem,
                &result,
                &options.branching,
                &node.branches,
            )
        } else {
            candidates::fresh_candidates_with_budget(
                &problem,
                &result,
                &options.branching,
                &node.branches,
                quota,
                budget.as_mut(),
            )
        };
        if let Some(budget) = &budget {
            quota = budget.quota;
            node.subtree = budget.estimate;
        }
        let cutoff = BranchEvaluationHistory::cutoff(&candidates, quota, &options.branching);
        let fresh_count = candidates.len();
        if !node.branches.is_empty()
            && quota >= 2
            && options
                .branch_history
                .as_ref()
                .is_some_and(|history| history.policy == BranchHistoryPolicy::Default)
        {
            output.branch_history.refresh_generators(
                &problem,
                &result.route_values(),
                &node.branches,
                &options.branching,
                cutoff,
                &mut candidates,
            );
        }
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
            if let Some(history) = &mut subtree_history {
                history.close(node.lineage);
            }
            *closed_bound = closed_bound.min(node.bound);
            return Ok(Decision::Continue);
        }
        let branch = strong::choose(
            parent_snapshot.as_ref(),
            &problem,
            &node,
            candidates,
            fresh_count,
            quota,
            cutoff,
            &mut pool,
            &options,
            &mut output,
            deadline,
        )?;
        if let Some((candidate, bounds, subtree)) = branch {
            let lineage = subtree_history
                .as_mut()
                .map(|history| history.split(node.lineage))
                .unwrap_or([0, 0]);
            for (side, constraint) in [(1, candidate.up.clone()), (0, candidate.down.clone())] {
                let mut branches = node.branches.clone();
                branches.push(constraint);
                open.push_evaluated(
                    Node {
                        lineage: lineage[side],
                        subtree,
                        observation: options
                            .branch_history
                            .as_ref()
                            .map(|_| (candidate.clone(), side, node.bound)),
                        branches,
                        bound: bounds[side],
                        parent: parent_snapshot.clone(),
                    },
                    0,
                    None,
                )?;
            }
        } else {
            // Integral aggregate arcs identify depot-to-depot paths even if
            // identical route variables split their weights fractionally.
            let mut used = HashSet::new();
            let mut routes = vec![];
            for (&a, &x) in &flows {
                if x < 0.5
                    || problem.model.networks[0].graph.arcs[a].tail
                        != problem.model.networks[0].graph.source
                {
                    continue;
                }
                let mut path = vec![];
                let mut next = a;
                loop {
                    if !used.insert(next) {
                        break;
                    }
                    path.push(problem.model.networks[0].graph.arcs[next].id);
                    let head = problem.model.networks[0].graph.arcs[next].head;
                    if head == problem.model.networks[0].graph.sink {
                        break;
                    }
                    let Some((&b, _)) = flows.iter().find(|(b, y)| {
                        **y > 0.5 && problem.model.networks[0].graph.arcs[**b].tail == head
                    }) else {
                        break;
                    };
                    next = b;
                }
                if let Some(route) = feasible_route(&problem, &path) {
                    routes.push(route);
                }
            }
            let mut cover = vec![0; problem.model.networks[0].graph.vertices.len()];
            for route in &routes {
                for &v in &route.nodes[1..route.nodes.len() - 1] {
                    cover[v as usize] += 1;
                }
            }
            let cost = problem
                .unit_route_incumbent(&routes)
                .unwrap_or(f64::INFINITY);
            if problem.customers.iter().all(|&v| cover[v] == 1)
                && routes.len() as f64 <= problem.model.rows[0].rhs
                && cost < output.incumbent
            {
                output.incumbent = cost;
                output.routes = routes;
            }
            if !closed(node.bound, output.incumbent, &options) {
                unresolved.push(node);
            } else {
                if let Some(history) = &mut subtree_history {
                    history.close(node.lineage);
                }
                *closed_bound = closed_bound.min(node.bound);
            }
        }
        Ok(Decision::Continue)
    }
}
