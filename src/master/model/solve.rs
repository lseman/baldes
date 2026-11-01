//! Shared-master initialization and column-generation solve workflow.
use super::*;

pub(super) fn validate_options(options: &Options) -> Result<()> {
    if options.gap_mip.is_some() && !cfg!(feature = "highs") {
        return Err(invalid("gap enumeration MIP requires the highs feature"));
    }
    if options
        .fixing_upper_bound
        .is_some_and(|bound| !bound.is_finite())
    {
        return Err(invalid("fixing upper bound must be finite"));
    }
    if options.network_promising_limit == 0 {
        return Err(invalid("network_promising_limit must be positive"));
    }
    if options.network_unpromising_limit == 0 {
        return Err(invalid("network_unpromising_limit must be positive"));
    }
    if let Some(proximal) = &options.cg.proximal {
        proximal.validate()?;
    }
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

pub(super) fn solve_with_fixed_routes<const D: usize>(
    problem: Problem<D>,
    variables: Vec<MasterVariable>,
    options: Options,
    fixed: &[FixedRoute],
    nonlinear: &[(usize, nonlinear::RowKernels)],
) -> Result<Output<D>> {
    solve_with_seed_routes(problem, variables, options, fixed, nonlinear, &[])
}

pub(super) fn solve_with_seed_routes<const D: usize>(
    problem: Problem<D>,
    variables: Vec<MasterVariable>,
    options: Options,
    fixed: &[FixedRoute],
    nonlinear: &[(usize, nonlinear::RowKernels)],
    seeds: &[NetworkRoute],
) -> Result<Output<D>> {
    let (master, state, options) =
        prepare_with_seed_routes(problem, variables, options, fixed, nonlinear, seeds)?;
    run(master, state, options)
}

/// Prepare an owned original model without running column generation.
pub(super) fn prepare_with_seed_routes<const D: usize>(
    mut problem: Problem<D>,
    mut variables: Vec<MasterVariable>,
    mut options: Options,
    fixed: &[FixedRoute],
    nonlinear: &[(usize, nonlinear::RowKernels)],
    seeds: &[NetworkRoute],
) -> Result<(ClpMaster, State<D>, Options)> {
    problem.validate()?;
    let synchronized_services = synchronization::prepare(
        &mut problem,
        &mut variables,
        std::mem::take(&mut options.synchronized_services),
    )?;
    if options.separation.route_recourse.is_some() {
        variables.extend(separation::theta_variables(&problem)?);
    }
    let first_static_bound_row = problem.rows.len();
    master_variables::install(&mut problem, &mut variables)?;
    let static_bound_rows = first_static_bound_row..problem.rows.len();
    if options.cg.warm_start.is_some() || !options.cg.known.is_empty() {
        return Err(invalid("shared root requires a fresh column registry"));
    }
    validate_options(&options)?;
    problem.validate()?;
    let deadline = Instant::now()
        .checked_add(options.cg.time_limit)
        .ok_or_else(|| invalid("network time limit"))?;
    if options.master_solver == crate::master::clp::MasterSolver::Ipm {
        options.cg.stabilize = false;
        options.cg.smoothing.initial_alpha = 0.;
    }
    if options
        .bucket_steps
        .iter()
        .any(|step| !step.is_finite() || *step < 0.)
    {
        return Err(invalid("bucket steps must be finite, nonnegative widths"));
    }
    let built = crate::master::formulation::build_model(
        &problem,
        &crate::master::formulation::Backend {
            solver: options.master_solver,
            threads: options.master_threads,
            clp: options.clp_options,
            ..Default::default()
        },
    )?;
    let mut master = built.master;
    let network_rows = built.quantity_rows;
    let shared_offset = built.shared_offset;
    let shared_rows = problem.rows;
    let mut oracles = vec![];
    for (network, route_entries) in problem.networks.into_iter().zip(network_rows) {
        let activity_rows = network.activity_entries.clone();
        let mut solver = Solver::new(network.graph.vertices.len(), network.graph.resources.len())?;
        solver.graph = network.graph.clone();
        solver.graph.graph_id = network.id;
        if !options.bucket_steps.is_empty() {
            let implicit_paired = match network.graph.coupled_kernel() {
                Some(kernel) => {
                    kernel.bound_paired_base_resource(&network.graph).is_none()
                        && kernel.paired_bucket_domains(&network.graph)?.is_some()
                }
                None => false,
            };
            let paired_width = implicit_paired && options.bucket_steps.len() == 1;
            if !paired_width && options.bucket_steps.len() != network.graph.resources.len() {
                return Err(invalid(
                    "bucket steps must list every network resource or the implicit paired base",
                ));
            }
            for (slot, &step) in options.bucket_steps.iter().enumerate() {
                solver.bucket_steps[slot] = step;
            }
        }
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
            network: network.into(),
            solver,
            index: Arc::new(index),
            costs: vec![],
            route_entries,
            calls: 0,
            enumerated: Arc::new(HashSet::new()),
            nonlinear_rows: vec![],
            knapsack_rows: vec![],
            presence_rows: vec![],
            activity_rows,
            resource_groups: vec![],
            ryan_foster_rows: vec![],
            synchronized_windows: vec![],
            proved_zero_arcs: vec![],
            proved_multiplicity_upper: None,
        });
    }
    let master_only = oracles.is_empty();
    let mut state = State {
        component_branches: vec![],
        pricing_witness: None,
        tree_incumbent: options.fixing_upper_bound.unwrap_or(f64::INFINITY),
        integer_primal_termination: options.integer_primal_termination,
        witness_tolerance: options.cg.tolerance,
        box_pricing_admission: false,
        variable_boxes: HashMap::new(),
        grouped_variable_boxes: HashMap::new(),
        original_variable_costs: options.original_variable_costs.clone(),
        pricing_master: None,
        pricing_gap: (options.cg.gap_absolute, options.cg.gap_relative),
        previous_fixing_gap: 1e12,
        gap_enumeration: options.gap_enumeration,
        tree_bound_floor: f64::NEG_INFINITY,
        fixing_bound: None,
        fixing_cutoffs: None,
        gap_checkpoint: None,
        enumerated_routes: 0,
        gap_pool_cutoff: None,
        gap_mip: options.gap_mip.clone(),
        fixing_calls: 0,
        objective_fixed: false,
        phase_one_infeasible: false,
        mapped_subtree_size: 1e12,
        mapped_subtree_priority: 1e12,
        mapped_lineage: 0,
        node_depth: 0,
        scheduler: Scheduler::new(
            options.network_pricing_schedule,
            oracles.len(),
            options.cg.tolerance,
        ),
        synchronized_services,
        oracles,
        shared_offset,
        routes: vec![],
        tags: HashMap::new(),
        correction: None,
        gradient: None,
        rhs: master.rhs.clone(),
        curvature_rows: (0..master.rhs.len())
            .map(|row| cg::CurvatureRow {
                class: if row < shared_offset {
                    cg::CurvatureRowClass::Excluded
                } else {
                    cg::CurvatureRowClass::Ordinary
                },
                active: true,
            })
            .collect(),
        deadline,
        shared_rows,
        retired_rows: HashSet::new(),
        fixed_routes: vec![],
        // With no pricing subproblems every original variable is already
        // loaded. Completeness requires no synthetic graph or enumeration.
        universe: master_only.then(Vec::new),
        enumeration_report: EnumerationReport {
            complete: master_only,
            ..Default::default()
        },
        preprocessing: PreprocessingReport::default(),
        master_variable_signs: variables
            .iter()
            .map(|v| {
                if v.bounds.lower >= 0. {
                    1
                } else if v.bounds.upper <= 0. {
                    -1
                } else {
                    0
                }
            })
            .collect(),
        master_variable_bound_rows: static_bound_rows.map(|r| shared_offset + r).collect(),
        master_variables: variables,
        route_upper_bounds: HashMap::new(),
        cuts: None,
    };
    for (shared, kernels) in nonlinear {
        kernels.install(&mut state, *shared)?;
    }
    lifecycle::install_fixed_routes(&mut master, &mut state, fixed)?;
    options.cg.known = master_variables::load(&mut master, &state)?;
    lifecycle::install_seed_routes(&mut master, &mut state, &mut options.cg.known, seeds)?;
    let caller_seeds = std::mem::take(&mut options.seed_routes);
    lifecycle::install_seed_routes(
        &mut master,
        &mut state,
        &mut options.cg.known,
        &caller_seeds,
    )?;
    Ok((master, state, options))
}

pub(super) fn run<const D: usize>(
    mut master: ClpMaster,
    mut state: State<D>,
    mut options: Options,
) -> Result<Output<D>> {
    let eval_started = Instant::now();
    // A finished directed infeasibility report publishes +infinity, whereas
    // the generic CG warm-start contract only accepts finite/-infinity bounds.
    // An owned proven-infeasible snapshot may safely discard its cached bounds
    // before re-solving; do not normalize arbitrary caller metadata this way.
    if state.phase_one_infeasible {
        if let Some(warm) = &mut options.cg.warm_start {
            if warm.numerical_bound == f64::INFINITY {
                warm.numerical_bound = f64::NEG_INFINITY;
            }
            if warm.certified_bound == f64::INFINITY {
                warm.certified_bound = f64::NEG_INFINITY;
            }
            if warm.bound == f64::INFINITY {
                warm.bound = match warm.bound_mode {
                    cg::BoundMode::Numerical => warm.numerical_bound,
                    cg::BoundMode::Certified => warm.certified_bound,
                };
            }
        }
    }
    if !options.original_variable_costs.is_empty() {
        state
            .original_variable_costs
            .clone_from(&options.original_variable_costs);
    }
    state.variable_boxes = variable_boxes::prepare(
        &options.variable_boxes,
        state.oracles.iter().map(|o| {
            (
                o.network.id,
                o.network.graph.as_ref(),
                o.solver.forbidden_arcs.as_slice(),
            )
        }),
    )?;
    // Explicit stabilization bypasses variable-bound admission. Its caller's
    // envelopes still require validation, but implicit proof work is unused.
    state.box_pricing_admission = !options.cg.stabilize && options.cg.proximal.is_none();
    let implicit = if state.box_pricing_admission {
        state.original_variable_costs.as_slice()
    } else {
        &[]
    };
    state.grouped_variable_boxes = grouped_boxes::prepare_registered(
        &options.grouped_variable_boxes,
        implicit,
        state.shared_rows.len(),
        state.oracles.iter().map(|o| {
            (
                o.network.id,
                o.network.graph.as_ref(),
                o.solver.forbidden_arcs.as_slice(),
            )
        }),
    )?;
    state.tree_incumbent = options.fixing_upper_bound.unwrap_or(f64::INFINITY);
    state.gap_enumeration = options.gap_enumeration;
    state.gap_mip = options.gap_mip.clone();
    state.pricing_witness = None;
    state.pricing_master = None;
    state.pricing_gap = (options.cg.gap_absolute, options.cg.gap_relative);
    state.integer_primal_termination = options.integer_primal_termination;
    state.witness_tolerance = options.cg.tolerance;
    // A resumed or modified formulation starts a fresh scan. Cached scores
    // from another node never supply coverage or a dual-bound contribution.
    state.scheduler = Scheduler::new(
        options.network_pricing_schedule,
        state.oracles.len(),
        options.cg.tolerance,
    );
    state.scheduler.max_unpromising = options.network_unpromising_limit;
    if options.separation.route_recourse.is_some() {
        // Enumerated routes would be selected by their arc costs alone; with
        // route recourse a node may only close through Theta-cut separation.
        options.enumeration = None;
        state.gap_enumeration = false;
        // Certificates rebuild the master from route columns alone.
        if options.cg.bound_mode == cg::BoundMode::Certified {
            return Err(invalid("certified bounds do not support route recourse"));
        }
    }
    separation::prepare(&mut state, &options)?;
    options.separation.configure(&mut options.cg);
    if options.cg.warm_start.is_none() && separation::supports_row_cleanup(&state) {
        options.cg.remove_basic_cuts = true;
    }
    state.scheduler.max_promising = options.network_promising_limit;
    options.cg.node_depth = state.node_depth;
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
    if let Some(memory) = state.cuts.as_ref().and_then(|c| c.resolved_memory_type) {
        for oracle in &mut state.oracles {
            oracle
                .solver
                .set_option("BALDESrankOneCutsMemoryType", (memory as i32) as f64)?;
        }
    }
    let auto_search = if state
        .cuts
        .as_ref()
        .is_some_and(|c| c.options.enabled && !state.oracles.is_empty())
    {
        let policy = crate::cuts::memory_policy(&state.oracles[0].solver.options)?;
        state.node_depth == 0
            && options.cg.warm_start.is_none()
            && state
                .cuts
                .as_ref()
                .is_some_and(|c| c.resolved_memory_type.is_none())
            && policy.memory_type == crate::cuts::MemoryType::Automatic
    } else {
        false
    };
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
    preprocessing_adapter::apply(&mut master, &mut state, &loaded, &options.preprocessing)?;
    master_variables::publish::bounds(
        &mut master,
        &mut state,
        &previous_bounds,
        &mut options.cg.known,
        options.cg.warm_start.as_mut(),
    )?;
    if options.preprocessing.enabled {
        retirement::apply(
            &mut master,
            &mut state,
            &mut options.cg,
            options.preprocessing.work_limit,
        )?;
    }
    if !auto_search {
        let (output, _) = execute_node_cg(master, state, options, None, false)?;
        return Ok(output);
    }
    let mut pass2_options = options.clone();
    state.cuts.as_mut().unwrap().resolved_memory_type = Some(crate::cuts::MemoryType::Vertex);
    for oracle in &mut state.oracles {
        oracle.solver.set_option(
            "BALDESrankOneCutsMemoryType",
            (crate::cuts::MemoryType::Vertex as i32) as f64,
        )?;
    }
    let (mut pass1, capture) = execute_node_cg(master, state, options, None, true)?;
    let upper_bound = pass2_options
        .separation
        .upper_bound
        .unwrap_or(f64::INFINITY);
    let pass1_incumbent = pass1
        .snapshot
        .state
        .pricing_witness
        .as_ref()
        .map_or(f64::INFINITY, |w| w.0)
        .min(upper_bound);
    let pass1_closed = pass1.snapshot.state.phase_one_infeasible
        || separation::separation_gap_closed(pass1.master.bound, pass1_incumbent);
    let pass1_tailed_off = pass1
        .snapshot
        .state
        .cuts
        .as_ref()
        .is_some_and(|c| c.stopped_by_tailing_off);
    let Some(auto) = capture.filter(|auto| {
        pass1.master.status == cg::Status::Optimal
            && !pass1_closed
            && !pass1_tailed_off
            && Instant::now() < pass1.snapshot.state.deadline
            && auto.pre_rank_one.is_some()
    }) else {
        return Ok(pass1);
    };
    let node_elapsed_secs = auto.node_elapsed.as_secs_f64();
    let SavedNodePoint {
        master: ckpt_master,
        output: mut ckpt_output,
        state: mut ckpt_state,
    } = auto.pre_rank_one.unwrap();
    if let Some(witness) = &pass1.snapshot.state.pricing_witness {
        if ckpt_state
            .pricing_witness
            .as_ref()
            .map_or(true, |old| witness.0 < old.0)
        {
            ckpt_state.pricing_witness = Some(witness.clone());
        }
    }
    if let Some(cuts) = ckpt_state.cuts.as_mut() {
        cuts.resolved_memory_type = Some(crate::cuts::MemoryType::Arc);
        cuts.stopped_by_tailing_off = false;
    }
    for oracle in &mut ckpt_state.oracles {
        oracle.solver.set_option(
            "BALDESrankOneCutsMemoryType",
            (crate::cuts::MemoryType::Arc as i32) as f64,
        )?;
    }
    let pre_cut_rounds = ckpt_output.cut_rounds;
    pass2_options.cg.known.clear();
    pass2_options.cg.warm_start = Some(std::mem::replace(
        &mut ckpt_output.warm_start,
        cg::WarmStart::empty(ckpt_output.bound_mode),
    ));
    pass2_options.cg.max_cut_rounds = pass2_options
        .cg
        .max_cut_rounds
        .saturating_sub(pre_cut_rounds);
    if ckpt_state.universe.is_some() {
        pass2_options.enumeration = None;
    }
    let (pass2, _) = execute_node_cg(
        ckpt_master,
        ckpt_state,
        pass2_options,
        Some(ckpt_output),
        false,
    )?;
    let arc_elapsed_secs = eval_started.elapsed().as_secs_f64();
    if let Some(witness) = &pass2.snapshot.state.pricing_witness {
        if pass1
            .snapshot
            .state
            .pricing_witness
            .as_ref()
            .map_or(true, |old| witness.0 < old.0)
        {
            pass1.snapshot.state.pricing_witness = Some(witness.clone());
        }
    }
    if pass2.snapshot.state.phase_one_infeasible {
        return Ok(pass2);
    }
    if pass2.master.status != cg::Status::Optimal {
        return Ok(pass1);
    }
    let incumbent = pass1
        .snapshot
        .state
        .pricing_witness
        .as_ref()
        .map_or(f64::INFINITY, |w| w.0)
        .min(upper_bound);
    if separation::separation_gap_closed(pass2.master.bound, incumbent) {
        return Ok(pass2);
    }
    let (prefer_arc, _score) = separation::prefer_arc_memory(
        pass1.master.bound,
        pass2.master.bound,
        incumbent,
        node_elapsed_secs,
        arc_elapsed_secs,
    );
    if prefer_arc {
        Ok(pass2)
    } else {
        Ok(pass1)
    }
}

pub(super) fn prepend_cg_output(prior: &cg::Output, current: &mut cg::Output) {
    if current.enumeration_failure.is_none() {
        current
            .enumeration_failure
            .clone_from(&prior.enumeration_failure);
    }
    for row in &mut current.history {
        row.iteration += prior.iterations;
        row.pricing_calls += prior.pricing_calls;
    }
    current.iterations += prior.iterations;
    current.pricing_calls += prior.pricing_calls;
    current.penalty_updates += prior.penalty_updates;
    current.cut_rounds += prior.cut_rounds;
    let mut history = prior.history.clone();
    history.append(&mut current.history);
    current.history = history;
    let mut inserted_tags = prior.inserted_tags.clone();
    inserted_tags.append(&mut current.inserted_tags);
    current.inserted_tags = inserted_tags;
}

fn execute_node_cg<const D: usize>(
    mut master: ClpMaster,
    mut state: State<D>,
    mut options: Options,
    mut previous: Option<cg::Output>,
    capture_auto_rank_one: bool,
) -> Result<(Output<D>, Option<AutoRankOneCapture<D>>)> {
    for oracle in &mut state.oracles {
        oracle.solver.rollback_point_saved = false;
        oracle.solver.requires_rollback = false;
    }
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
    let request_initial_enumeration = options.enumeration.is_some() && previous.is_none();
    let state = Rc::new(RefCell::new(state));
    let mut controller = Coordinator(
        state.clone(),
        request_initial_enumeration,
        options.search_inactive_columns,
        None,
        options.enumeration.clone(),
        None,
        Vec::new(),
        !options.preprocessing.integer_domains,
        options.cg.bound_mode == cg::BoundMode::Numerical,
        CheckpointStore {
            prior_cg: None,
            rollback: None,
            auto_rank_one: capture_auto_rank_one.then(|| AutoRankOneCapture {
                pre_rank_one: None,
                rank_one_started_at: Instant::now(),
                node_elapsed: std::time::Duration::ZERO,
            }),
        },
    );
    let log_context = options.cg.log_interval.is_some().then(|| {
        let mut context = format!(
            "  Master: {:?} | Node depth: {} | Stabilization: {} | Bound mode: {}",
            options.master_solver,
            state.borrow().node_depth,
            if options.cg.stabilize && options.cg.smoothing.alpha_factor > 0. {
                "on"
            } else {
                "off"
            },
            options.cg.bound_mode.as_str()
        );
        if matches!(
            options.master_solver,
            crate::master::clp::MasterSolver::Highs | crate::master::clp::MasterSolver::Gurobi
        ) {
            let workers = if options.master_threads == 0 {
                std::thread::available_parallelism().map_or(1, usize::from)
            } else {
                options.master_threads
            };
            context.push_str(&format!(" | Workers: {workers}"));
        }
        context
    });
    let gap = (options.cg.gap_absolute, options.cg.gap_relative);
    let started = Instant::now();
    let time_limit = options.cg.time_limit;
    let max_iterations = options.cg.max_iterations;
    let mut result = loop {
        let spent = previous.as_ref().map_or(0, |out| out.iterations);
        options.cg.max_iterations = max_iterations.saturating_sub(spent);
        options.cg.time_limit = time_limit.saturating_sub(started.elapsed()).min(
            state
                .borrow()
                .deadline
                .saturating_duration_since(Instant::now()),
        );
        controller.9.prior_cg = previous.clone();
        let optimized = cg::optimize_with_log_context(
            &mut master,
            &schedule,
            |phase, duals| state.borrow_mut().price(phase, duals),
            &mut controller,
            options.cg.take_continuation(),
            log_context.as_deref(),
        );
        let mut current = match optimized {
            Ok(current) => current,
            Err(error) => {
                let (restored_master, restored_output) =
                    controller.recover_pricing_timeout(error)?;
                master = restored_master;
                break restored_output;
            }
        };
        if let Some(prior) = previous.take() {
            prepend_cg_output(&prior, &mut current);
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
        let mode = current.bound_mode;
        options.cg.warm_start = Some(std::mem::replace(
            &mut current.warm_start,
            cg::WarmStart::empty(mode),
        ));
        // transfer stages every model/registry change on private copies. On a
        // backend failure the checkpoint is still valid for ordinary pricing.
        let (routes, counts, deadline) = controller
            .5
            .take()
            .ok_or_else(|| invalid("missing enumeration checkpoint"))?;
        if let Err(error) = enumeration::install_collected(
            &mut master,
            &mut state.borrow_mut(),
            &mut options.cg,
            routes,
            counts,
            deadline,
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
            preprocessing_adapter::apply(&mut master, &mut state, &loaded, &options.preprocessing)?;
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
    if result.status == cg::Status::Infeasible
        && !(state.borrow().preprocessing.propagation.infeasible
            && !options.preprocessing.integer_domains)
        && !result.feasibility.iter().any(|r| {
            r.certified_infeasible
                || (result.bound_mode == cg::BoundMode::Numerical
                    && r.status == cg::FeasibilityStatus::Infeasible)
        })
    {
        result.status = cg::Status::Stalled;
        result.lp = master.objective;
    }
    if result.status == cg::Status::Optimal
        && (!result.bound.is_finite()
            || !cg::gap_smaller_than_tolerance(result.bound, result.lp, gap.0, gap.1))
    {
        result.status = cg::Status::Stalled;
    }
    if result.status == cg::Status::Stalled
        && state.borrow().has_unpriced_paired_terms(&master.duals())
    {
        eprintln!("BALDES warning: paired pricing retains the unit resource objective; the associated owner objective or active row dual differs. The restricted master value is available, but raw pricing cannot certify this owner optimum (status Stalled).");
    }
    let preprocessing_proof = result.status == cg::Status::Infeasible
        && state.borrow().preprocessing.propagation.infeasible
        && !options.preprocessing.integer_domains;
    if preprocessing_proof || result.feasibility.iter().any(|r| r.certified_infeasible) {
        state.borrow_mut().phase_one_infeasible = true;
    }
    let auto_rank_one = controller.9.auto_rank_one.take();
    drop(controller);
    let state = Rc::try_unwrap(state)
        .map_err(|_| invalid("network state still borrowed"))?
        .into_inner();
    let mut values = vec![0.; state.routes.len()];
    let mut master_values: Vec<_> = state.master_variables.iter().map(|v| (v.id, 0.)).collect();
    if !preprocessing_proof {
        for (&tag, &value) in result.column_tags.iter().zip(master.values().iter()) {
            if tag >= 0 {
                values[tag as usize] += value;
            } else if let Some((index, sign)) = master_variables::decode(tag) {
                master_values[index].1 += sign * value;
            }
        }
    }
    let (cuts, lp_before_cuts) = state.cuts.as_ref().map_or((0, f64::NAN), |cuts| {
        (
            cuts.cut_count()
                + state
                    .oracles
                    .first()
                    .map_or(0, |o| o.solver.cuts.knapsack_cuts.len()),
            cuts.lp_before_cuts,
        )
    });
    Ok((
        Output {
            cuts,
            lp_before_cuts,
            master_values,
            routes: state
                .routes
                .iter()
                .map(|route| route.as_ref().clone())
                .collect(),
            fixed_routes: state.fixed_routes.clone(),
            values,
            duals: if preprocessing_proof {
                vec![0.; master.rhs.len()]
            } else {
                retirement::duals(&state, &master.duals())
            },
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
        },
        auto_rank_one,
    ))
}
