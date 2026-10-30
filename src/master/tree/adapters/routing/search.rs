use super::*;

/// Best-bound branch-and-price. With `branching.time_window_selection`, the
/// search runs on window copies of the arcs into multi-window vertices,
/// branches on window selection counts and reports routes in input arc IDs.
pub fn solve<const D: usize>(
    problem: RoutingContext<D>,
    options: TreeOptions,
) -> Result<TreeOutput> {
    let Some(resource) = options.branching.time_window_selection else {
        return solve_split(problem, options);
    };
    #[cfg(not(feature = "multi-time-windows"))]
    {
        let _ = resource;
        Err(invalid(
            "time-window selection branching requires multi-time-windows",
        ))
    }
    #[cfg(feature = "multi-time-windows")]
    {
        let mut options = options;
        let vehicles = problem
            .model
            .rows
            .first()
            .ok_or_else(|| invalid("tree requires set partitioning rows"))?
            .rhs;
        let mut graph = problem.model.networks[0].graph.clone();
        let split = graph.split_time_window_arcs(resource)?;
        options.root.initial_routes = std::mem::take(&mut options.root.initial_routes)
            .into_iter()
            .filter_map(|mut route| {
                route.arcs = graph.route_on_window_arcs(&split, &route.arcs)?;
                Some(route)
            })
            .collect();
        options.branching.window_arcs = Some(split.clone());
        options.initial_solution = std::mem::take(&mut options.initial_solution)
            .into_iter()
            .map(|mut route| {
                route.arcs = graph
                    .route_on_window_arcs(&split, &route.arcs)
                    .ok_or_else(|| invalid("initial solution has no feasible window arcs"))?;
                Ok(route)
            })
            .collect::<Result<Vec<_>>>()?;
        let mut output = solve_split(RoutingContext::set_partitioning(graph, vehicles)?, options)?;
        for route in &mut output.routes {
            split.restore(route);
        }
        Ok(output)
    }
}

fn solve_split<const D: usize>(
    problem: RoutingContext<D>,
    mut options: TreeOptions,
) -> Result<TreeOutput> {
    if options.threads == 0 {
        return Err(invalid("tree threads must be positive"));
    }
    if options.deluxing.is_some() && options.enumerated_mip.is_none() {
        return Err(invalid("DeLuxing requires enumerated MIP"));
    }
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
    problem.model.networks[0].graph.validate()?;
    options.branching.priorities.validate()?;
    if options.strong_branching.as_ref().is_some_and(|strong| {
        strong
            .phases
            .iter()
            .any(|phase| !phase.subtree_ratio.is_finite() || phase.subtree_ratio < 0.)
    }) {
        return Err(invalid(
            "strong branching subtree ratios must be finite and nonnegative",
        ));
    }
    let standard = RoutingContext::set_partitioning(
        problem.model.networks[0].graph.clone(),
        problem
            .model
            .rows
            .first()
            .ok_or_else(|| invalid("tree requires set partitioning rows"))?
            .rhs,
    )?;
    if problem.model.rows.len() != standard.model.rows.len()
        || problem
            .model
            .rows
            .iter()
            .zip(&standard.model.rows)
            .any(|(a, b)| a.sense != b.sense || a.rhs != b.rhs)
        || problem.model.networks[0].arc_entries != standard.model.networks[0].arc_entries
        || problem.model.networks[0].route_entries != standard.model.networks[0].route_entries
        || problem.customers != standard.customers
        || problem.model.networks[0]
            .graph
            .arcs
            .iter()
            .any(|a| a.cost < 0.)
        || problem.model.networks[0]
            .graph
            .arcs
            .iter()
            .map(|a| a.id)
            .collect::<HashSet<_>>()
            .len()
            != problem.model.networks[0].graph.arcs.len()
    {
        return Err(invalid("tree requires a routing set-partitioning master with nonnegative costs and unique arc ids"));
    }
    let objective_lattice = objective_lattice::ObjectiveLattice::prepare(&problem);
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
    let workers = if options.threads > 1 {
        Some(
            rayon::ThreadPoolBuilder::new()
                .num_threads(options.threads)
                .thread_name(|i| format!("baldes-tree-{i}"))
                .build()
                .map_err(|e| invalid(format!("tree worker pool: {e}")))?,
        )
    } else {
        None
    };
    let deadline = started + options.time_limit;
    let mut output = TreeOutput {
        bound_mode: options.root.bound_mode,
        status: TreeStatus::NodeLimit,
        incumbent: f64::INFINITY,
        bound: 0.,
        routes: vec![],
        nodes: 0,
        parallel_batches: 0,
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
    if !options.initial_solution.is_empty() {
        let routes = options
            .initial_solution
            .iter()
            .map(|route| {
                feasible_route(&problem, &route.arcs)
                    .ok_or_else(|| invalid("invalid initial solution route"))
            })
            .collect::<Result<Vec<_>>>()?;
        output.incumbent = problem.unit_route_incumbent(&routes).ok_or_else(|| {
            invalid("initial solution must cover each customer once within the fleet limit")
        })?;
        output.routes = routes;
        for route in &output.routes {
            pool.add(&problem, &route.arcs);
        }
        options
            .root
            .initial_routes
            .extend(output.routes.iter().cloned());
    }
    for route in &options.root.initial_routes {
        pool.add(&problem, &route.arcs);
    }
    for &v in &problem.customers {
        let a = problem.model.networks[0]
            .graph
            .arcs
            .iter()
            .find(|a| a.tail == problem.model.networks[0].graph.source && a.head == v);
        let b = problem.model.networks[0]
            .graph
            .arcs
            .iter()
            .find(|a| a.tail == v && a.head == problem.model.networks[0].graph.sink);
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
    let mut open =
        crate::master::tree::TreeFrontier::<Node<D>, routing::RootOutput<D>>::exact_best_bound();
    open.push_evaluated(
        Node {
            lineage: 0,
            subtree: Default::default(),
            observation: None,
            branches: vec![],
            bound: 0.,
            parent: None,
        },
        0,
        None,
    )?;
    let mut unresolved = vec![];
    let mut subtree_history = options
        .strong_branching
        .as_ref()
        .filter(|strong| {
            strong.use_history_subtree_size
                && !options
                    .branch_history
                    .as_ref()
                    .is_some_and(|h| h.policy == BranchHistoryPolicy::Reliable)
        })
        .map(|_| strong::SubtreeHistory::new());
    // Tolerance closure does not strengthen a node's lower certificate to
    // the incumbent. These leaves still contribute to the global minimum.
    let mut closed_bound = f64::INFINITY;
    let mut previous = None::<Rc<routing::NodeSnapshot<D>>>;
    let stop = crate::master::tree::run(
        &mut super::adapter::RoutingSearch {
            problem: &problem,
            options: &options,
            output: &mut output,
            pool: &mut pool,
            workers: workers.as_ref(),
            deadline,
            previous: &mut previous,
            subtree_history: &mut subtree_history,
            objective_lattice: &objective_lattice,
            closed_bound: &mut closed_bound,
            unresolved: &mut unresolved,
        },
        &mut open,
        None,
    )?;
    match stop {
        crate::master::tree::Stop::TimeLimit => output.status = TreeStatus::TimeLimit,
        crate::master::tree::Stop::NodeLimit => output.status = TreeStatus::NodeLimit,
        crate::master::tree::Stop::Incomplete | crate::master::tree::Stop::OverflowOnly => {
            output.status = TreeStatus::Incomplete
        }
        _ => {}
    }
    unresolved.retain(|n| {
        if closed(n.bound, output.incumbent, &options) {
            if let Some(history) = &mut subtree_history {
                history.close(n.lineage);
            }
            closed_bound = closed_bound.min(n.bound);
            false
        } else {
            true
        }
    });
    if open.is_empty() && !unresolved.is_empty() {
        output.status = TreeStatus::Incomplete;
    }
    for node in unresolved {
        open.push_evaluated(node, 0, None)?;
    }
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
            .map(|n| n.node.bound)
            .fold(closed_bound.min(output.incumbent), f64::min);
    }
    output.seconds = started.elapsed().as_secs_f64();
    Ok(output)
}
