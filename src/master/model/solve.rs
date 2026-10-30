//! Shared-master initialization and column-generation solve workflow.
use super::*;

pub(super) fn validate_options(options: &Options) -> Result<()> {
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
    mut problem: Problem<D>,
    mut variables: Vec<MasterVariable>,
    mut options: Options,
    fixed: &[FixedRoute],
    nonlinear: &[(usize, nonlinear::RowKernels)],
    seeds: &[NetworkRoute],
) -> Result<Output<D>> {
    problem.validate()?;
    let synchronized_services = synchronization::prepare(
        &mut problem,
        &mut variables,
        std::mem::take(&mut options.synchronized_services),
    )?;
    master_variables::install(&mut problem, &mut variables)?;
    if options.cg.warm_start.is_some() || !options.cg.known.is_empty() {
        return Err(invalid("shared root requires a fresh column registry"));
    }
    validate_options(&options)?;
    problem.validate()?;
    let deadline = Instant::now()
        .checked_add(options.cg.time_limit)
        .ok_or_else(|| invalid("network time limit"))?;
    let built = crate::master::formulation::build_model(
        &problem,
        &crate::master::formulation::Backend::default(),
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
            nonlinear_rows: vec![],
            knapsack_rows: vec![],
            presence_rows: vec![],
            activity_rows,
            resource_groups: vec![],
            synchronized_windows: vec![],
            proved_zero_arcs: vec![],
            proved_multiplicity_upper: None,
        });
    }
    let master_only = oracles.is_empty();
    let mut state = State {
        pricing_witness: None,
        integer_primal_termination: options.integer_primal_termination,
        witness_tolerance: options.cg.tolerance,
        box_pricing_admission: false,
        variable_boxes: HashMap::new(),
        grouped_variable_boxes: HashMap::new(),
        original_variable_costs: Arc::new(vec![]),
        pricing_master: None,
        pricing_gap: (options.cg.gap_absolute, options.cg.gap_relative),
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

pub(super) fn run<const D: usize>(
    mut master: ClpMaster,
    mut state: State<D>,
    mut options: Options,
) -> Result<Output<D>> {
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
    let mut controller = Coordinator(
        state.clone(),
        options.enumeration.is_some(),
        options.search_inactive_columns,
        None,
    );
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
            options.cg.take_continuation(),
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
        let mode = current.bound_mode;
        options.cg.warm_start = Some(std::mem::replace(
            &mut current.warm_start,
            cg::WarmStart::empty(mode),
        ));
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
        && !result.feasibility.iter().any(|r| r.certified_infeasible)
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
    if result.feasibility.iter().any(|r| r.certified_infeasible) {
        state.borrow_mut().phase_one_infeasible = true;
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
        routes: state
            .routes
            .iter()
            .map(|route| route.as_ref().clone())
            .collect(),
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
