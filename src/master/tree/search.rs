use super::*;

pub fn solve<const D: usize>(
    problem: RootProblem<D>,
    mut options: TreeOptions,
) -> Result<TreeOutput> {
    #[cfg(not(feature = "highs"))]
    if options.enumerated_mip.is_some() {
        return Err(invalid("enumerated MIP requires the highs feature"));
    }
    options.root.master_solver.ensure_available()?;
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
    // Resolve and validate the partition once for the whole tree.
    let mut options = options;
    if let Some(config) = &mut options.branching.clusters {
        config.partition = Some(clusters::partition(&problem, config)?);
    }
    if options
        .branch_history
        .as_ref()
        .is_some_and(|c| c.minimum_observations == 0)
    {
        return Err(invalid(
            "branch history requires positive observation count",
        ));
    }
    let deadline = started + options.time_limit;
    let mut output = TreeOutput {
        bound_mode: options.root.bound_mode,
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
        branch_history: Default::default(),
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
        observation: None,
        branches: vec![],
        bound: 0.,
        parent: None,
    }];
    let mut unresolved = vec![];
    // Tolerance closure does not strengthen a node's lower certificate to
    // the incumbent. These leaves still contribute to the global minimum.
    let mut closed_bound = f64::INFINITY;
    let mut previous = None::<Rc<root::NodeSnapshot<D>>>;
    while !open.is_empty() {
        // Discard only nodes whose selected bound closes the global gap.
        open.retain(|n| {
            if closed(n.bound, output.incumbent, &options) {
                closed_bound = closed_bound.min(n.bound);
                false
            } else {
                true
            }
        });
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
            previous.as_deref(),
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
                output.branch_history.record(
                    candidate,
                    *side,
                    node.branches.len().saturating_sub(1),
                    history::EvaluationKind::NodePricing,
                    *parent_bound,
                    result.master.lp,
                    result.solver_seconds,
                );
            }
        }
        if let Some(snapshot) = &result.snapshot {
            previous = Some(snapshot.clone());
        }
        output.pricing_calls += result.master.pricing_calls;
        output.fixing_calls += result.fixing_calls;
        output.enumerated_routes += result.enumerated_routes;
        if result.preprocessing.infeasible {
            continue;
        }
        // The LP objective may include artificials or come from a heuristic
        // pricing phase. Only the selected bound is a node lower bound:
        // numerical estimates never enter certified-mode node evaluations.
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
            closed_bound = closed_bound.min(node.bound);
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
            closed_bound = closed_bound.min(node.bound);
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
            for (side, constraint) in [(1, candidate.up.clone()), (0, candidate.down.clone())] {
                let mut branches = node.branches.clone();
                branches.push(constraint);
                open.push(Node {
                    observation: options
                        .branch_history
                        .as_ref()
                        .map(|_| (candidate.clone(), side, node.bound)),
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
            let cost = problem
                .unit_route_incumbent(&routes)
                .unwrap_or(f64::INFINITY);
            if problem.customers.iter().all(|&v| cover[v] == 1)
                && routes.len() as f64 <= problem.rows[0].rhs
                && cost < output.incumbent
            {
                output.incumbent = cost;
                output.routes = routes;
            }
            if !closed(node.bound, output.incumbent, &options) {
                unresolved.push(node);
            } else {
                closed_bound = closed_bound.min(node.bound);
            }
        }
    }
    unresolved.retain(|n| {
        if closed(n.bound, output.incumbent, &options) {
            closed_bound = closed_bound.min(n.bound);
            false
        } else {
            true
        }
    });
    if open.is_empty() && !unresolved.is_empty() {
        output.status = TreeStatus::Incomplete;
    }
    open.extend(unresolved);
    output.open_nodes = open.len();
    if open.is_empty() && output.incumbent.is_finite() {
        output.bound = if closed_bound.is_finite() {
            closed_bound.min(output.incumbent)
        } else {
            // Routing costs are nonnegative; no finite leaf certificate was
            // retained. An incumbent alone cannot strengthen this baseline.
            0.
        };
        output.status = if closed(output.bound, output.incumbent, &options) {
            TreeStatus::Optimal
        } else {
            TreeStatus::Incomplete
        };
    } else if open.is_empty() {
        output.status = TreeStatus::Infeasible;
        output.bound = f64::INFINITY;
    } else {
        output.bound = open
            .iter()
            .map(|n| n.bound)
            .fold(closed_bound.min(output.incumbent), f64::min);
    }
    output.seconds = started.elapsed().as_secs_f64();
    Ok(output)
}
