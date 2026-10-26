use super::*;
use crate::master::branching::enforcement;

pub fn solve<const D: usize>(
    problem: RoutingContext<D>,
    mut options: RootOptions,
) -> Result<RootOutput<D>> {
    options.cg.warm_start = None;
    solve_seeded(problem, options, None)
}
pub fn inherit<const D: usize>(
    parent: &NodeSnapshot<D>,
    branches: &[(usize, RowSense, f64)],
    options: RootOptions,
) -> Result<RootOutput<D>> {
    let branches: Vec<_> = branches
        .iter()
        .map(|&(arc, sense, rhs)| BranchConstraint::ArcFlow { arc, sense, rhs })
        .collect();
    inherit_branches(parent, &branches, options)
}
pub fn inherit_branches<const D: usize>(
    parent: &NodeSnapshot<D>,
    branches: &[BranchConstraint],
    options: RootOptions,
) -> Result<RootOutput<D>> {
    inherit_with_history(parent, branches, None, options)
}
pub(crate) fn inherit_with_history<const D: usize>(
    parent: &NodeSnapshot<D>,
    branches: &[BranchConstraint],
    previous: Option<&NodeSnapshot<D>>,
    mut options: RootOptions,
) -> Result<RootOutput<D>> {
    options.node_depth = options
        .node_depth
        .max(parent.branch_depth() + branches.len());
    let (problem, mut master, warm) = parent.fork(branches, options.cg.artificial_cost)?;
    // Penalty escalation belongs to an evaluation, not to the inherited node.
    // Start each child with the configured penalty, including cut artificials.
    for &column in &warm.artificials {
        master.set_cost(column, options.cg.artificial_cost)?;
    }
    if let Some(previous) = previous {
        // Restore costs on a copy before checking compatibility, retaining the
        // preceding sibling's simplex state when only its penalties changed.
        let mut continued = previous.master.try_clone()?;
        for &column in &warm.artificials {
            if column < continued.costs.len() {
                continued.set_cost(column, master.costs[column])?;
            }
        }
        master.continue_from(&continued)?;
    }
    solve_seeded(problem, options, Some((parent, master, warm)))
}
pub(super) fn solve_seeded<const D: usize>(
    problem: RoutingContext<D>,
    mut options: RootOptions,
    seed: Option<(&NodeSnapshot<D>, ClpMaster, cg::WarmStart)>,
) -> Result<RootOutput<D>> {
    problem.model.validate()?;
    if problem.model.networks.len() != 1 {
        return Err(invalid(
            "routing registration must identify its compiled pricing network",
        ));
    }
    let configured_root_separator = options.cuts.enabled
        || options.capacity_cuts.is_some()
        || options.kpath_cuts.is_some()
        || options.general_kpath_cuts.is_some()
        || options.knapsack_cuts.is_some();
    if problem.model.networks[0].graph.max_route_arcs().is_some()
        && !problem.model.networks[0]
            .graph
            .bounded_walks_are_vertex_simple()
        && options.relaxation_improvement_priority > 0.
    {
        return Err(invalid(
            "dynamic NG strengthening is unsupported for bounded-walk root solves",
        ));
    }
    if configured_root_separator
        && !problem.model.networks[0]
            .graph
            .bounded_walks_are_vertex_simple()
        && problem.model.networks[0].graph.max_route_arcs().is_some()
    {
        return Err(invalid(
            "bounded-walk root separators require hard elementarity groups on every interior vertex",
        ));
    }

    options.cg.bound_mode = options.bound_mode;
    options.cg.node_depth = options.node_depth;
    options.master_solver.ensure_available()?;
    options.normalize_pricing_options()?;
    if !options.bucket_steps.is_empty()
        && (options.bucket_steps.len() != problem.model.networks[0].graph.resources.len()
            || options
                .bucket_steps
                .iter()
                .any(|s| !s.is_finite() || *s < 0.))
    {
        return Err(invalid(
            "bucket steps must be finite, nonnegative widths in resource order",
        ));
    }
    if options.master_solver == MasterSolver::Ipm {
        options.cg.stabilize = false;
        options.cg.smoothing.initial_alpha = 0.;
    }
    if !problem.fixed_routes.is_empty()
        || problem.branch_constraints.iter().any(|(row, branch)| {
            *row == 0
                && matches!(
                    branch,
                    BranchConstraint::NumberOfVehicles {
                        sense: RowSense::LessEqual,
                        ..
                    }
                )
        })
    {
        options.cg.multiplicity = problem.model.rows[0].rhs.max(0.);
    }
    let solve_started = std::time::Instant::now();
    problem.model.networks[0].graph.validate()?;
    let cut_rows = match seed.as_ref() {
        Some((saved, master, _)) => saved
            .cut
            .preprocessing_rows(&problem.model.networks[0].graph, master)?,
        None => vec![],
    };
    let new_fixed = seed.as_ref().is_some_and(|(saved, _, _)| {
        problem.fixed_routes.len() != saved.problem.fixed_routes.len()
    });
    let (preprocessing, arc_bounds) = problem.preprocess(&cut_rows)?;
    if preprocessing.infeasible {
        let mut output = preprocessing_adapter::infeasible_output(
            preprocessing,
            solve_started,
            options.bound_mode,
        );
        output.fixed_cost = problem.fixed_cost();
        output.fixed_routes = problem
            .fixed_routes
            .iter()
            .map(|(r, q)| (r.clone(), f64::from(*q)))
            .collect();
        output.inherited_cuts = seed.as_ref().map_or(0, |(saved, _, _)| saved.active_cuts());
        return Ok(output);
    }
    let backend = crate::master::formulation::Backend {
        solver: options.master_solver,
        threads: options.master_threads,
        clp: options.clp_options,
        zero_tolerance: 1e-7,
        // Retain the routing adapter's existing route-domain cap.
        column_upper: 1e12,
    };
    let (saved, mut master) = if let Some((saved, mut master, warm)) = seed {
        options.cg.warm_start = Some(warm);
        backend.configure(&mut master)?;
        (Some(saved), master)
    } else {
        let built = crate::master::formulation::build_model(&problem.model, &backend)?;
        (None, built.master)
    };
    let log_context = if options.cg.log_interval.is_some() {
        let mut context = format!(
            "  Master: {:?} | Node depth: {} | Stabilization: {} | Bound mode: {}",
            options.master_solver,
            options.node_depth,
            if options.cg.stabilize && options.cg.smoothing.alpha_factor > 0. {
                "on"
            } else {
                "off"
            },
            options.bound_mode.as_str()
        );
        if matches!(
            options.master_solver,
            MasterSolver::Highs | MasterSolver::Gurobi
        ) {
            let workers = if options.master_threads == 0 {
                std::thread::available_parallelism().map_or(1, usize::from)
            } else {
                options.master_threads
            };
            context.push_str(&format!(" | Workers: {workers}"));
        }
        Some(context)
    } else {
        None
    };
    let mut solver = if let Some(saved) = saved {
        saved.pricing.fork_for_node()
    } else {
        let mut solver = Solver::new(
            problem.model.networks[0].graph.vertices.len(),
            problem.model.networks[0].graph.resources.len(),
        )?;
        solver.graph = problem.model.networks[0].graph.clone();
        for (slot, &step) in options.bucket_steps.iter().enumerate() {
            solver.bucket_steps[slot] = step;
        }
        solver
    };
    // Resource branches change the physical head windows. Rebuild labels and
    // enumeration before using inherited cost-dependent state in that domain.
    let mut resource_domain_changed = false;
    for (arc, original) in solver
        .graph
        .arcs
        .iter_mut()
        .zip(&problem.model.networks[0].graph.arcs)
    {
        if arc.bounds != original.bounds {
            arc.bounds = original.bounds;
            resource_domain_changed = true;
        }
    }
    if resource_domain_changed {
        solver.restore_arcs();
    }
    solver.rollback_point_saved = false;
    solver.requires_rollback = false;
    if new_fixed {
        // Parent fixing/enumeration may cover only improvements below its old
        // incumbent. A fixed-node solve starts a fresh complete pricing domain.
        solver.restore_arcs();
    }
    // A snapshot may later branch on any arc; same-customer-set orderings
    // must remain distinct in its retained enumeration.
    solver.enumeration_preserve_arc_paths =
        options.capture_state || problem.model.rows.len() > problem.customers.len() + 1;
    for (name, value) in &options.pricing_options {
        solver.set_option(name, *value)?;
    }
    solver.set_ryan_foster(enforcement::pricing_ryan_foster(&problem))?;
    solver.set_forbidden_arcs(
        arc_bounds
            .iter()
            .enumerate()
            .filter_map(|(a, b)| (b.upper <= 0.).then_some(a))
            .collect(),
    )?;
    if let Some(warm) = options.cg.warm_start.as_ref() {
        if let Some(saved) = saved {
            for (c, column) in warm.columns.iter().enumerate() {
                if let Some(column) = column {
                    let route = &saved.routes[column.tag as usize];
                    if route.arcs.iter().any(|&id| {
                        problem
                            .arc_position(id)
                            .is_some_and(|a| arc_bounds[a].upper <= 0.)
                    }) {
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
        if problem.model.networks[0].graph.source != 0
            || problem.model.networks[0].graph.sink != config.demands.len()
            || problem.model.networks[0].graph.vertices.len() != config.demands.len() + 1
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
    if saved.is_none() && options.kpath_cuts.is_some() && options.general_kpath_cuts.is_some() {
        return Err(invalid("select standalone or native k-path configuration"));
    }
    let kpath_separator =
        if saved.is_some() {
            None
        } else if let Some(config) = options.general_kpath_cuts.take() {
            let graph_id = problem.model.networks[0].graph.graph_id.max(0);
            let input = config
                .graphs
                .iter()
                .find(|g| g.graph.id == graph_id)
                .ok_or_else(|| invalid("native k-path configuration has no pricing graph"))?;
            if input.graph.source != problem.model.networks[0].graph.source as i32
                || input.graph.sink != problem.model.networks[0].graph.sink as i32
                || input.graph.vertices.iter().any(|v| {
                    v.id < 0 || v.id as usize >= problem.model.networks[0].graph.vertices.len()
                })
            {
                return Err(invalid("native k-path pricing vertices/terminals differ"));
            }
            for arc in &problem.model.networks[0].graph.arcs {
                if !input.graph.arcs.iter().any(|a| {
                    a.id == arc.id && a.tail == arc.tail as i32 && a.head == arc.head as i32
                }) {
                    return Err(invalid("native k-path pricing arc incidence differs"));
                }
            }
            let mut separator = GeneralKPathSeparator::new(config.use_packing_sets);
            separator.parameters = config.parameters;
            separator.prepare(
                &config.graphs,
                &config.demands,
                config.capacity,
                config.two_path_resource,
            )?;
            Some(RootKPathSeparator::General {
                separator,
                graphs: config.graphs,
            })
        } else if let Some(config) = options.kpath_cuts.take() {
            if problem.model.networks[0].graph.source != 0
                || problem.model.networks[0].graph.sink != config.demands.len()
                || problem.model.networks[0].graph.vertices.len() != config.demands.len() + 1
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
        let graph = problem.model.networks[0].graph.graph_id.max(0);
        if config.parameters.fenchel.is_some() {
            solver.knapsack_exact_coefficients = true;
        }
        solver.knapsack_resource_indices = config.resource_indices;
        let indices = solver.knapsack_indices();
        if indices
            .values()
            .any(|&r| r >= problem.model.networks[0].graph.resources.len())
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
    let arc_count = problem.model.networks[0].graph.arcs.len();
    let state = Rc::new(RefCell::new(State {
        solver,
        routes: saved.map_or_else(Vec::new, |s| s.routes.clone()),
        route_tags: saved.map_or_else(Default::default, |s| {
            s.routes
                .iter()
                .enumerate()
                .map(|(i, r)| (r.identity(), i as i32))
                .collect()
        }),
        incumbent: if new_fixed {
            options.incumbent
        } else {
            saved.map_or(options.incumbent, |s| options.incumbent.min(s.incumbent))
        },
        enumerate: options.enumerate,
        rounded: options.round_reduced_costs,
        pricing: adapter::RootPricing::new(&options.cg),
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
    let zeros = vec![0.; problem.model.rows.len()];
    options.cg.known.clear();
    let mut live: HashSet<i32> = options
        .cg
        .warm_start
        .as_ref()
        .map_or_else(HashSet::new, |w| {
            w.columns.iter().flatten().map(|c| c.tag).collect()
        });
    for route in &options.initial_routes {
        if route.arcs.iter().any(|&id| {
            problem
                .arc_position(id)
                .is_some_and(|a| arc_bounds[a].upper <= 0.)
        }) {
            continue;
        }
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
        checkpoint: None,
        enumeration_checkpoint: None,
        certificate_search: certificate::SearchCache::default(),
    };
    let initial_rows = options
        .cg
        .warm_start
        .as_ref()
        .map_or(problem.model.rows.len(), |w| w.initial_rows);
    let capture_state = options.capture_state;
    let inherited_cuts = saved.map_or(0, |s| s.active_cuts());
    if let Some(deadline) = state.borrow().deadline {
        // Preparation has already consumed part of the root budget. CG and
        // each backend solve must share the pricing callback's same deadline.
        options.cg.time_limit = options
            .cg
            .time_limit
            .min(deadline.saturating_duration_since(std::time::Instant::now()));
    }
    let optimized = cg::optimize_with_log_context::<ClpMaster, _, _>(
        &mut master,
        &schedule,
        |phase, duals| {
            state
                .borrow_mut()
                .price(&problem, phase, duals, phase_floor)
        },
        &mut controller,
        options.cg,
        log_context.as_deref(),
    );
    let mut output = match optimized {
        Ok(output) => output,
        Err(error) => {
            let (restored, mut output) = controller.recover_pricing_timeout(error)?;
            master = restored;
            output.pricing_calls = state.borrow().pricing.attempts;
            output
        }
    };
    enforcement::close_at_master_objective(&problem, phase_floor, &master.values, &mut output);
    output.warm_start.initial_rows = initial_rows;
    let state = state.borrow();
    // Extract values, duals and simplex iterations before moving master into snapshot.
    let values = master.values();
    let duals = master.duals();
    let simplex_iterations = output
        .feasibility
        .iter()
        .fold(master.simplex_iterations, |sum, round| {
            sum.saturating_add(round.simplex_iterations)
        });
    let ipm_iterations = output
        .feasibility
        .iter()
        .fold(master.ipm_iterations, |sum, round| {
            sum.saturating_add(round.ipm_iterations)
        });
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
        fixed_cost: problem.fixed_cost(),
        fixed_routes: problem
            .fixed_routes
            .iter()
            .map(|(r, q)| (r.clone(), f64::from(*q)))
            .collect(),
        preprocessing,
        snapshot,
        simplex_iterations,
        ipm_iterations,
        inherited_cuts,
        master: output,
        routes: state.routes.clone(),
        values,
        duals,
        incumbent: state.incumbent,
        pricing_seconds: state.pricing_seconds,
        oracle_calls: state.pricing.oracle_calls,
        solver_seconds: solve_started.elapsed().as_secs_f64(),
        cuts: state.solver.cuts.cuts.len()
            + state.cut.capacity_cuts.len()
            + state.solver.cuts.knapsack_cuts.len(),
        lp_before_cuts: state.cut.lp_before_cuts,
        fixing_calls: state.fixing_calls,
        enumerated_routes: state.enumerated_routes,
    })
}
