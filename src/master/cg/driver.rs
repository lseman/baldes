use super::controller::apply_column_cap;
use super::*;

/// Munari--Gondzio, Algorithm 1: bounds for the continuous master, distinct
/// from integer incumbents and certificates for stronger pricing domains.
struct IpmAccuracy {
    lower: f64,
    upper: f64,
}

impl IpmAccuracy {
    fn new() -> Self {
        Self {
            lower: f64::NEG_INFINITY,
            upper: f64::INFINITY,
        }
    }

    fn tolerance(&self) -> f64 {
        if !self.lower.is_finite() || !self.upper.is_finite() {
            return 0.5;
        }
        // epsilon_max = 1, degree of optimality D = 10. A numerical floor
        // avoids requesting a gap below the accurate master's precision.
        (((self.upper - self.lower).max(0.) / (1. + self.upper.abs())) / 10.).clamp(1e-9, 1.)
    }
}

/// Generic column generation loop that works with any `LpMaster` backend.
pub fn optimize<M: LpMaster, P: FnMut(usize, &[f64]) -> Result<Answer>, C: Controller<M>>(
    master: &mut M,
    supplied: &[Phase],
    price: P,
    controller: &mut C,
    options: Options,
) -> Result<Output> {
    optimize_with_log_context(master, supplied, price, controller, options, None)
}

pub(crate) fn optimize_with_log_context<
    M: LpMaster,
    P: FnMut(usize, &[f64]) -> Result<Answer>,
    C: Controller<M>,
>(
    master: &mut M,
    supplied: &[Phase],
    price: P,
    controller: &mut C,
    options: Options,
    log_context: Option<&str>,
) -> Result<Output> {
    if options.log_interval == Some(0) {
        return Err(invalid("CG log interval must be positive"));
    }
    let logging = options.log_interval.is_some();
    let counts = if logging {
        controller.cut_counts()
    } else {
        vec![]
    };
    let mut progress =
        crate::master::progress::ProgressLog::new(options.log_interval, counts, log_context);
    let result = optimize_inner(master, supplied, price, controller, options, &mut progress);
    if logging {
        progress.finish(
            &result,
            controller.incumbent(),
            controller.cut_counts(),
            controller.bucket_steps(),
        );
    }
    result
}

fn optimize_inner<M: LpMaster, P: FnMut(usize, &[f64]) -> Result<Answer>, C: Controller<M>>(
    master: &mut M,
    supplied: &[Phase],
    mut price: P,
    controller: &mut C,
    mut options: Options,
    progress: &mut crate::master::progress::ProgressLog,
) -> Result<Output> {
    let schedule = phases::initialize(0, supplied)?;
    setup::validate_options(&options)?;
    let started = Instant::now();
    // A zero alpha factor disables the reference's smoothing workflow, not
    // only publication of smoothed duals.
    let mut smoothing = DualPriceSmoothing::new(options.smoothing.clone());
    let configured_proximal = options.proximal.as_ref().filter(|_| options.stabilize);
    let depth_active = options.stabilization_active_at_depth();
    let proximal = configured_proximal.filter(|_| depth_active);
    let mut proximal_state = crate::master::stabilization::proximal::ProximalState::default();
    let retain_smoothing = options.stabilize
        && configured_proximal.is_none()
        && smoothing.parameters.alpha_factor > 0.;
    let stabilize = retain_smoothing && depth_active;
    let mut warm = options.warm_start.take();
    let is_resume = warm.is_some();
    let inherited_center = warm.as_ref().and_then(|warm| warm.stabilization.as_ref());
    if let Some(previous) = inherited_center {
        smoothing.inherit_center(previous, (schedule.len() - 1) as i32);
    }
    let has_inherited_center = inherited_center.is_some();
    if !depth_active {
        smoothing.deactivate();
    }
    smoothing.exclude_rows(controller.unstabilized_rows());
    setup::validate_warm_start(master, warm.as_ref())?;
    if let (Some(parameters), Some(previous)) = (
        configured_proximal,
        warm.as_ref().and_then(|w| w.proximal.as_ref()),
    ) {
        if let Some(snapshot) =
            previous.restore(master, controller, parameters, options.node_depth)?
        {
            proximal_state.inherit_for_solve(&snapshot, schedule.len() - 1)?;
        }
    }
    let inherited_numerical = warm
        .as_ref()
        .map_or(f64::NEG_INFINITY, |w| w.numerical_bound);
    let inherited_certified = warm
        .as_ref()
        .map_or(f64::NEG_INFINITY, |w| w.certified_bound);
    let inherited_bound = match options.bound_mode {
        BoundMode::Numerical => inherited_numerical,
        BoundMode::Certified => inherited_certified,
    };
    // The published certificate may cover the original integer domain through
    // a stronger relaxation than the pricing LP. Keep its convergence estimate
    // separate: an integer bound cannot prove that LP pricing is exhausted.
    let mut pricing_estimate = f64::NEG_INFINITY;
    // A successful universe handoff applies once to this formulation. A new
    // row, coefficient/domain change or remapping makes it eligible again.
    let mut enumeration_transferred = false;
    let mut feasibility_attempted = false;
    let mut ipm_accuracy = IpmAccuracy::new();
    let mut accurate_next = false;
    let mut round_lower_bound = if inherited_bound.is_finite() {
        inherited_bound
    } else {
        -1e12
    };
    let mut round_primal_bound = 1e12_f64;
    let mut loaded: Vec<Option<Column>> = match warm.as_mut() {
        Some(warm) => std::mem::take(&mut warm.columns),
        None => std::mem::take(&mut options.known)
            .into_iter()
            .map(Some)
            .collect(),
    };
    loaded.resize(master.costs().len(), None);
    for (index, column) in loaded.iter().enumerate() {
        if let Some(column) = column {
            apply_column_cap(master, controller, index, column.tag)?;
        }
    }
    let mut seen: HashSet<_> = loaded.iter().flatten().map(Column::signature).collect();
    // Generated columns exist independently of membership in the restricted
    // master. Columns omitted by a pricing insertion test can become
    // eligible under the strict regeneration cutoff.
    let mut pool: Vec<Column> = warm
        .as_mut()
        .map_or_else(Vec::new, |warm| std::mem::take(&mut warm.generated_pool));
    let mut pool_seen: HashSet<_> = pool.iter().map(Column::signature).collect();
    // Include newly supplied initial routes without losing same-batch
    // duplicates when the caller starts with only a populated master.
    let missing: Vec<_> = loaded
        .iter()
        .flatten()
        .filter(|column| !pool_seen.contains(&column.signature()))
        .cloned()
        .collect();
    pool_seen.extend(missing.iter().map(Column::signature));
    pool.extend(missing);
    let mut out = Output {
        status: Status::IterationLimit,
        feasibility: vec![],
        enumeration_requested: false,
        enumeration_failure: None,
        lp: 0.,
        bound: inherited_bound,
        bound_mode: options.bound_mode,
        numerical_bound: inherited_numerical,
        certified_bound: inherited_certified,
        iterations: 0,
        pricing_calls: 0,
        columns: master.costs().len(),
        history: vec![],
        column_tags: loaded
            .iter()
            .map(|c| c.as_ref().map_or(-1, |c| c.tag))
            .collect(),
        inserted_tags: vec![],
        artificials: warm
            .as_mut()
            .map_or_else(Vec::new, |w| std::mem::take(&mut w.artificials)),
        penalty_updates: 0,
        pruned_by_bound: false,
        cut_rounds: 0,
        warm_start: WarmStart {
            bound: f64::NEG_INFINITY,
            bound_mode: options.bound_mode,
            numerical_bound: f64::NEG_INFINITY,
            certified_bound: f64::NEG_INFINITY,
            stabilization: None,
            proximal: None,
            generated_pool: vec![],
            columns: vec![],
            artificials: vec![],
            cut_artificials: vec![],
            initial_rows: 0,
        },
    };
    out.column_tags.resize(master.costs().len(), -1);
    if options.artificial_cost > 0. && !is_resume {
        let rows = 0..master.rhs().len();
        for (_, column) in append_artificials(master, rows, options.artificial_cost)? {
            out.artificials.push(column);
            loaded.push(None);
            out.column_tags.push(-1);
        }
    }
    let mut stage = schedule.len() - 1;
    let mut pass_bound: f64;
    // Phase-local stopping
    // value, including heuristic passes; never expose it as a global bound.
    let mut stage_bound = -1e12_f64;
    let mut round_penalty_updates = 0;
    let mut tailing = 0;
    let mut lp_at_separation = 0.0f64;
    let mut separated_once = false;
    let initial_rows = warm.as_ref().map_or(master.rhs().len(), |w| w.initial_rows);
    let mut cut_artificials: Vec<(usize, usize)> = warm
        .as_mut()
        .map_or_else(Vec::new, |w| std::mem::take(&mut w.cut_artificials));
    drop(warm);
    let finish = |mut out: Output,
                  loaded: Vec<Option<Column>>,
                  cut_artificials: Vec<(usize, usize)>,
                  pool: Vec<Column>,
                  smoothing: DualPriceSmoothing| {
        out.warm_start = WarmStart {
            bound: out.bound,
            bound_mode: out.bound_mode,
            numerical_bound: out.numerical_bound,
            certified_bound: out.certified_bound,
            stabilization: retain_smoothing.then_some(smoothing),
            proximal: out.warm_start.proximal.take(),
            generated_pool: pool,
            columns: loaded,
            artificials: out.artificials.clone(),
            cut_artificials,
            initial_rows,
        };
        Ok(out)
    };
    if let Some(parameters) = configured_proximal {
        out.warm_start.proximal = ProximalWarmStart::capture(
            &proximal_state,
            master,
            controller,
            parameters,
            &out.artificials,
            &cut_artificials,
            options.node_depth,
        )?;
    }
    let mut proximal_retry = false;
    for _ in 0..options.max_iterations {
        if out.iterations >= options.max_iterations {
            out.status = Status::IterationLimit;
            break;
        }
        let iteration = out.iterations + 1;
        if started.elapsed() >= options.time_limit {
            out.status = Status::TimeLimit;
            return finish(out, loaded, cut_artificials, pool, smoothing);
        }
        if controller.refresh_master_columns(master, &mut loaded)? {
            controller.synchronize_generated_columns(&mut pool, &loaded)?;
            seen = loaded.iter().flatten().map(Column::signature).collect();
            pool_seen = pool.iter().map(Column::signature).collect();
            // Integer-domain bounds survive, but the new coefficient contract
            // needs its own LP convergence estimate.
            pricing_estimate = f64::NEG_INFINITY;
            ipm_accuracy = IpmAccuracy::new();
            accurate_next = false;
            enumeration_transferred = false;
            round_primal_bound = 1e12;
            smoothing.restart_schedule((schedule.len() - 1) as i32);
            proximal_state.reset();
            out.warm_start.proximal = None;
            stage = schedule.len() - 1;
            stage_bound = -1e12;
            round_penalty_updates = 0;
        }
        let requested_gap = (!accurate_next).then(|| ipm_accuracy.tolerance());
        let outcome = master.solve_for_column_generation(
            options.time_limit.saturating_sub(started.elapsed()),
            requested_gap,
        )?;
        let accurate = outcome == LpSolveOutcome::Optimal;
        if !accurate && outcome != LpSolveOutcome::Suboptimal {
            out.iterations = iteration;
            if outcome == LpSolveOutcome::PrimalInfeasible && !feasibility_attempted {
                feasibility_attempted = true;
                let warm = finish(
                    out.clone(),
                    loaded.clone(),
                    cut_artificials.clone(),
                    pool.clone(),
                    smoothing.clone(),
                )?
                .warm_start;
                if let Some(recovery) = controller.recover_feasibility(
                    master,
                    &warm,
                    options.time_limit.saturating_sub(started.elapsed()),
                    options.max_iterations.saturating_sub(out.iterations),
                    options.tolerance,
                    options.bound_mode,
                )? {
                    let resume = super::feasibility::install(
                        master,
                        controller,
                        recovery,
                        &mut out,
                        super::feasibility::Columns {
                            loaded: &mut loaded,
                            seen: &mut seen,
                            pool: &mut pool,
                            pool_seen: &mut pool_seen,
                        },
                    )?;
                    if resume {
                        pricing_estimate = f64::NEG_INFINITY;
                        out.numerical_bound = inherited_numerical;
                        out.certified_bound = inherited_certified;
                        out.bound = inherited_bound;
                        ipm_accuracy = IpmAccuracy::new();
                        accurate_next = false;
                        enumeration_transferred = false;
                        smoothing = DualPriceSmoothing::new(options.smoothing.clone());
                        proximal_state.reset();
                        out.warm_start.proximal = None;
                        smoothing.exclude_rows(controller.unstabilized_rows());
                        round_lower_bound = if inherited_bound.is_finite() {
                            inherited_bound
                        } else {
                            -1e12
                        };
                        round_primal_bound = 1e12;
                        stage = schedule.len() - 1;
                        stage_bound = -1e12;
                        round_penalty_updates = 0;
                        continue;
                    }
                    return finish(out, loaded, cut_artificials, pool, smoothing);
                }
            }
            out.status = match outcome {
                LpSolveOutcome::PrimalInfeasible => Status::Infeasible,
                LpSolveOutcome::DualInfeasible => Status::Unbounded,
                LpSolveOutcome::TimeLimit => Status::TimeLimit,
                LpSolveOutcome::IterationLimit => Status::IterationLimit,
                _ => Status::Stalled,
            };
            if out.status == Status::Infeasible {
                out.lp = 1e12;
            }
            if out.status == Status::Unbounded {
                out.lp = f64::NEG_INFINITY;
            }
            out.iterations = iteration;
            return finish(out, loaded, cut_artificials, pool, smoothing);
        }
        out.lp = master.objective();
        out.iterations = iteration;
        let raw = master.duals().to_vec();
        let positive = out.artificials.iter().any(|&i| master.values()[i] > 1e-6);
        if !positive {
            ipm_accuracy.upper = ipm_accuracy.upper.min(out.lp);
        }
        controller.master_solved(master, &out.column_tags, &out.artificials)?;
        if accurate
            && !positive
            && controller.original_witness_available()
            && out.bound.is_finite()
            && controller.incumbent().is_finite()
            && gap_smaller_than_tolerance(
                out.bound,
                controller.incumbent(),
                options.gap_absolute,
                options.gap_relative,
            )
        {
            // Original integer-domain closure is separate from LP pricing
            // exhaustion. Publish the retained primal and inherited proof.
            out.status = Status::Optimal;
            out.pruned_by_bound = true;
            out.bound = out.bound.min(controller.incumbent());
            match out.bound_mode {
                BoundMode::Numerical => out.numerical_bound = out.bound,
                BoundMode::Certified => out.certified_bound = out.bound,
            }
            return finish(out, loaded, cut_artificials, pool, smoothing);
        }

        // Stabilization compares this round's inherited lower bound with
        // its preceding artificial-free restricted-LP objective. The global
        // integer incumbent is a separate stopping/enumeration control.
        let gap = optimality_gap(round_lower_bound, round_primal_bound);
        let primal_slack = round_primal_bound.abs().max(out.lp.abs()) * 1e-10 + options.tolerance;
        if !positive && round_primal_bound - primal_slack > out.lp {
            round_primal_bound = out.lp;
        }
        if stabilize {
            smoothing.after_master(
                &raw,
                master.senses(),
                gap,
                stage as i32,
                iteration == 1 && !has_inherited_center,
            );
        }
        // earlyCGtermType1 uses the phase value for stage > 0
        // and the LP pricing estimate for stage zero, before another call.
        let stopping_bound = if stage > 0 {
            stage_bound
        } else {
            pricing_estimate
        };
        let converged = !proximal_retry
            && accurate
            && !positive
            && stopping_bound.is_finite()
            && gap_smaller_than_tolerance(
                stopping_bound,
                out.lp,
                options.gap_absolute,
                options.gap_relative,
            );
        let mut minimum = 0.;
        let mut fresh = 0;
        let mut exact = stage == 0;
        let mut answer_phase = stage;
        let mut inserted = 0;
        let mut first_angle = 0.;
        let mut advanced = false;
        let mut penalties = false;
        if converged && stage > 0 {
            stage -= 1;
            // eval clears both values on each phase change.
            stage_bound = -1e12;
            // Each phase is a new solveMastLpPrimPh1A2 call,
            // whose penalty-update count starts at zero.
            round_penalty_updates = 0;
            advanced = true;
            if stabilize {
                smoothing.after_iteration();
            }
        }
        if !converged {
            if options.column_cleanup_threshold > 0. {
                let removed = cleanup_columns(master, &out.column_tags, &options);
                if !removed.is_empty() {
                    for &i in &removed {
                        if let Some(column) = &loaded[i] {
                            seen.remove(&column.signature());
                        }
                    }
                    let (_, map) = master.remove_cuts(&[], &removed)?;
                    loaded = loaded
                        .into_iter()
                        .zip(&map)
                        .filter_map(|(c, m)| m.map(|_| c))
                        .collect();
                    out.column_tags = out
                        .column_tags
                        .iter()
                        .zip(&map)
                        .filter_map(|(&t, m)| m.map(|_| t))
                        .collect();
                    out.artificials = out.artificials.iter().filter_map(|&i| map[i]).collect();
                    cut_artificials = cut_artificials
                        .iter()
                        .filter_map(|&(r, c)| Some((r, map[c]?)))
                        .collect();
                    controller.remap_columns(&map)?;
                }
            }
            if accurate {
                let first = loaded.len();
                if let Some(minimum) = super::pool::restore(
                    master,
                    controller,
                    &raw,
                    &pool,
                    &mut loaded,
                    &mut seen,
                    &mut out,
                    options.tolerance,
                    started.checked_add(options.time_limit),
                )? {
                    // Only the restricted master improved. No pricing minimum,
                    // phase completion or new global bound follows from reuse.
                    if stabilize {
                        smoothing.after_iteration();
                    }
                    out.history.push(Iteration {
                        iteration,
                        lp: out.lp,
                        minimum,
                        new_columns: loaded.len() - first,
                        columns: out.columns,
                        exact: false,
                        alpha: smoothing.alpha,
                        effective_alpha: smoothing.effective_alpha,
                        angle: 0.,
                        misprices: smoothing.misprices,
                        phase: stage,
                        pricing_calls: out.pricing_calls,
                        stabilization_gap: gap,
                        master_gap_tolerance: None,
                    });
                    if options.log_interval.is_some() {
                        progress.iteration(
                            out.history.last().unwrap(),
                            out.bound,
                            controller.incumbent(),
                            controller.cut_counts(),
                            controller.bucket_steps(),
                        );
                    }
                    continue;
                }
            }
            if started.elapsed() >= options.time_limit {
                out.status = Status::TimeLimit;
                return finish(out, loaded, cut_artificials, pool, smoothing);
            }
            proximal_retry = false;
            let mut current = if let Some(parameters) = proximal {
                let row_roles = proximal::roles(controller, raw.len())?;
                let ordinary: Vec<_> = row_roles
                    .iter()
                    .enumerate()
                    .filter_map(|(r, role)| {
                        (role.active && role.class == CurvatureRowClass::Ordinary).then_some(r)
                    })
                    .collect();
                let mut excluded = controller.unstabilized_rows();
                if parameters.curvature.is_some()
                    || parameters.explicit.is_some()
                    || parameters.rhs_row_scaling
                {
                    excluded.extend(row_roles.iter().enumerate().filter_map(|(row, role)| {
                        (!role.active || role.class == CurvatureRowClass::Excluded).then_some(row)
                    }));
                    excluded.sort_unstable();
                    excluded.dedup();
                }
                if row_roles
                    .iter()
                    .any(|role| role.active && role.class != CurvatureRowClass::Excluded)
                {
                    proximal_state.begin_iteration(&raw, &ordinary, stage, parameters)?;
                }
                proximal_state.pricing_point(
                    master,
                    &raw,
                    stage,
                    parameters,
                    &excluded,
                    options.time_limit.saturating_sub(started.elapsed()),
                )?
            } else if stabilize {
                smoothing.pricing.clone()
            } else {
                raw.clone()
            };
            let mut pending = vec![];
            loop {
                if started.elapsed() >= options.time_limit {
                    out.status = Status::TimeLimit;
                    return finish(out, loaded, cut_artificials, pool, smoothing);
                }
                let mut answer = price(stage, &current)?;
                // One exact key per returned column is reused for the pool,
                // improvement test and insertion instead of reformatting it.
                let keys: Vec<_> = answer.columns.iter().map(Column::signature).collect();
                let mut created = vec![];
                for (column, identity) in answer.columns.iter().zip(&keys) {
                    if column.acceptable && !pool_seen.contains(identity) {
                        pool.push(column.clone());
                        created.push(identity.clone());
                    }
                }
                // Preserve duplicates produced within one oracle batch, as
                // with restricted-master insertion below.
                pool_seen.extend(created);
                answer.exact =
                    controller.pricing_complete() && (answer.exact || schedule[stage].exact());
                out.pricing_calls += 1;
                fresh = answer
                    .columns
                    .iter()
                    .zip(&keys)
                    .filter(|(c, identity)| {
                        c.acceptable
                            && master_reduced_cost(c, &raw) < -options.tolerance
                            && !seen.contains(*identity)
                    })
                    .count();
                minimum = answer
                    .columns
                    .iter()
                    .filter(|c| c.acceptable)
                    .fold(answer.minimum, |best, c| best.min(c.reduced_cost));
                let corrected = answer.bound_minimum.unwrap_or(minimum);
                pass_bound = controller
                    .pricing_bound(master, &current)?
                    .unwrap_or_else(|| {
                        lagrangian(master, &current, corrected, options.multiplicity)
                    });
                if !pass_bound.is_finite() && pass_bound != f64::NEG_INFINITY {
                    return Err(invalid("invalid pricing lower bound"));
                }
                let slack = stage_bound.abs().max(pass_bound.abs()) * 1e-10 + 1e-6;
                if !(stage_bound - slack > pass_bound) {
                    stage_bound = pass_bound;
                }
                if answer.exact {
                    pricing_estimate = pricing_estimate.max(pass_bound);
                    ipm_accuracy.lower = ipm_accuracy.lower.max(pass_bound);
                    out.numerical_bound = out.numerical_bound.max(pass_bound);
                    if options.bound_mode == BoundMode::Certified {
                        let incumbent = controller.incumbent();
                        // Certify a closing pricing pass, or when the current
                        // master point suggests the incumbent gap can close.
                        // A numerical bound only schedules this attempt; it
                        // never authorizes pruning in certified mode.
                        if fresh == 0
                            || (incumbent.is_finite()
                                && gap_smaller_than_tolerance(
                                    out.lp,
                                    incumbent,
                                    options.gap_absolute,
                                    options.gap_relative,
                                ))
                        {
                            publish_certificate(
                                &mut out,
                                controller.certified_pricing_bound(master, &current)?,
                            )?;
                        }
                    }
                    out.bound = match options.bound_mode {
                        BoundMode::Numerical => out.numerical_bound,
                        BoundMode::Certified => out.certified_bound,
                    };
                }
                exact = answer.exact;
                answer_phase = stage;
                if proximal.is_some() {
                    proximal_state.observe(&current, pass_bound, exact);
                    pending.push(keys.into_iter().zip(answer.columns).collect::<Vec<_>>());
                    // A proximal null step cannot finish a phase or raise
                    // penalties. Complete ordinary pricing at the original LP
                    // dual before consuming any no-column result.
                    if fresh == 0 && current != raw {
                        current = raw.clone();
                        continue;
                    }
                    break;
                }
                if !stabilize {
                    pending.push(keys.into_iter().zip(answer.columns).collect::<Vec<_>>());
                    break;
                }
                let best = answer
                    .columns
                    .iter()
                    .filter(|c| c.acceptable)
                    .min_by(|a, b| a.reduced_cost.total_cmp(&b.reduced_cost));
                let supplied_gradient = controller.take_pricing_subgradient(master.rhs());
                let controller_has_gradient = supplied_gradient.is_some();
                let subgradient_available = controller_has_gradient || best.is_some();
                let mut gradient = if let Some(supplied) = supplied_gradient {
                    if supplied.len() != master.rhs().len()
                        || supplied.iter().any(|g| !g.is_finite())
                    {
                        return Err(invalid("invalid pricing subgradient"));
                    }
                    supplied
                } else {
                    master.rhs().to_vec()
                };
                if let Some(best) = best.filter(|_| !controller_has_gradient) {
                    // A returned pricing route contributes to the subgradient
                    // even when its rounded reduced cost is zero. Filtering it
                    // by that cost changes the reference's angle inputs.
                    for &(r, a) in &best.entries {
                        if let Some(g) = gradient.get_mut(r) {
                            *g -= options.multiplicity * a;
                        }
                    }
                }
                pending.push(keys.into_iter().zip(answer.columns).collect::<Vec<_>>());
                let first = smoothing.misprices == 0;
                let again = smoothing.after_pricing(&Feedback {
                    negative_columns: fresh,
                    subgradient_available,
                    subgradient: gradient,
                    lagrangian: pass_bound,
                    phase: stage as i32,
                });
                if first {
                    first_angle = smoothing.angle;
                }
                if !again {
                    break;
                }
                current = smoothing.pricing.clone();
            }
            if stabilize {
                smoothing.after_iteration();
            }
            if let Some(parameters) = proximal {
                proximal_state.after_iteration(parameters)?;
                let active = proximal_state
                    .take_artificial_values()
                    .is_some_and(|values| values.iter().any(|v| v.abs() > 1e-6));
                let activity_due = round_penalty_updates < options.max_penalty_updates
                    && !(stage > 0
                        && controller.incumbent().is_finite()
                        && pass_bound.is_finite()
                        && gap_smaller_than_tolerance(
                            pass_bound,
                            controller.incumbent(),
                            options.gap_absolute,
                            options.gap_relative,
                        ));
                if fresh == 0 && activity_due {
                    proximal_retry = proximal_state.artificial_activity(active, parameters)?;
                    if proximal_retry {
                        round_penalty_updates += 1;
                        out.penalty_updates += 1;
                    }
                }
                out.warm_start.proximal = ProximalWarmStart::capture(
                    &proximal_state,
                    master,
                    controller,
                    parameters,
                    &out.artificials,
                    &cut_artificials,
                    options.node_depth,
                )?;
            }
            let due = round_penalty_updates < options.max_penalty_updates && positive;
            penalties = fresh == 0
                && due
                && !(stage > 0
                    && controller.incumbent().is_finite()
                    && pass_bound.is_finite()
                    && gap_smaller_than_tolerance(
                        pass_bound,
                        controller.incumbent(),
                        options.gap_absolute,
                        options.gap_relative,
                    ));
            advanced = fresh == 0 && !exact && !penalties && !proximal_retry && stage > 0;
            if advanced {
                stage -= 1;
                stage_bound = -1e12;
                round_penalty_updates = 0;
            }
            let first = loaded.len();
            for batch in pending {
                let mut identities = vec![];
                for (identity, column) in batch {
                    if !column.acceptable
                        || (!options.insert_non_negative
                            && master_reduced_cost(&column, &raw) >= -options.tolerance)
                    {
                        continue;
                    }
                    if seen.contains(&identity) {
                        continue;
                    }
                    out.column_tags.push(column.tag);
                    out.inserted_tags.push(column.tag);
                    loaded.push(Some(column));
                    identities.push(identity);
                    inserted += 1;
                }
                seen.extend(identities);
            }
            // One master append for the round, in insertion order.
            let added: Vec<(f64, &[(usize, f64)])> = loaded[first..]
                .iter()
                .map(|c| c.as_ref().map(|c| (c.cost, c.entries.as_slice())).unwrap())
                .collect();
            let added_indices = master.add_columns(&added)?;
            for index in added_indices {
                apply_column_cap(
                    master,
                    controller,
                    index,
                    loaded[index].as_ref().unwrap().tag,
                )?;
            }
        }
        out.columns = master.costs().len();
        out.history.push(Iteration {
            iteration,
            lp: out.lp,
            minimum,
            new_columns: inserted,
            columns: out.columns,
            exact,
            alpha: smoothing.alpha,
            effective_alpha: smoothing.effective_alpha,
            angle: first_angle,
            misprices: smoothing.misprices,
            phase: answer_phase,
            pricing_calls: out.pricing_calls,
            stabilization_gap: gap,
            master_gap_tolerance: if accurate { None } else { requested_gap },
        });
        // A completed auxiliary-activity update is a null step of the
        // automatic controller, not termination of the original pricing loop.
        if proximal_retry {
            continue;
        }
        // A previous exact bound can close this LP without another oracle
        // call. Certified mode must validate this closing point as well.
        if converged && exact && options.bound_mode == BoundMode::Certified {
            publish_certificate(&mut out, controller.certified_pricing_bound(master, &raw)?)?;
        }
        if options.log_interval.is_some() {
            progress.iteration(
                out.history.last().unwrap(),
                out.bound,
                controller.incumbent(),
                controller.cut_counts(),
                controller.bucket_steps(),
            );
        }
        // An exact bound can close the incumbent gap even without a fresh
        // column. Stop before strict-improvement enumeration discards every
        // real route and replaces this feasible master with artificials.
        if !converged
            && !positive
            && (out.bound_mode == BoundMode::Certified || fresh > 0 || exact)
            && controller.incumbent().is_finite()
            && out.bound.is_finite()
            && gap_smaller_than_tolerance(
                out.bound,
                controller.incumbent(),
                options.gap_absolute,
                options.gap_relative,
            )
        {
            if !accurate {
                // The integer certificate may close the node while this
                // interior point is still suboptimal. Finalize its LP point
                // before publishing a completed node or capturing its state.
                accurate_next = true;
                continue;
            }
            out.status = Status::Optimal;
            out.pruned_by_bound = true;
            // Fixing/enumeration can retain only routes capable of improving
            // the incumbent. A bound on that restricted problem may exceed
            // the incumbent; the original problem's global bound cannot.
            out.bound = out.bound.min(controller.incumbent());
            match out.bound_mode {
                BoundMode::Numerical => out.numerical_bound = out.bound,
                BoundMode::Certified => out.certified_bound = out.bound,
            }
            return finish(out, loaded, cut_artificials, pool, smoothing);
        }
        if penalties {
            proximal_state.restart_schedule();
            for &id in &out.artificials {
                master.set_cost(id, master.costs()[id] * options.penalty_factor)?;
            }
            out.penalty_updates += 1;
            round_penalty_updates += 1;
            ipm_accuracy = IpmAccuracy::new();
            continue;
        }
        if fresh == 0 && exact && !accurate {
            // No improving column at a suboptimal RMP point is not an LP
            // optimality proof. Re-solve accurately and repeat exact pricing
            // before separation, snapshots, enumeration, or termination.
            accurate_next = true;
            continue;
        }
        accurate_next = false;
        if fresh == 0 && exact {
            if positive && !feasibility_attempted {
                feasibility_attempted = true;
                let warm = finish(
                    out.clone(),
                    loaded.clone(),
                    cut_artificials.clone(),
                    pool.clone(),
                    smoothing.clone(),
                )?
                .warm_start;
                if let Some(recovery) = controller.recover_feasibility(
                    master,
                    &warm,
                    options.time_limit.saturating_sub(started.elapsed()),
                    options.max_iterations.saturating_sub(out.iterations),
                    options.tolerance,
                    options.bound_mode,
                )? {
                    let resume = super::feasibility::install(
                        master,
                        controller,
                        recovery,
                        &mut out,
                        super::feasibility::Columns {
                            loaded: &mut loaded,
                            seen: &mut seen,
                            pool: &mut pool,
                            pool_seen: &mut pool_seen,
                        },
                    )?;
                    if resume {
                        pricing_estimate = f64::NEG_INFINITY;
                        // A feasibility minimum cannot refresh an objective bound.
                        out.numerical_bound = inherited_numerical;
                        out.certified_bound = inherited_certified;
                        out.bound = inherited_bound;
                        ipm_accuracy = IpmAccuracy::new();
                        accurate_next = false;
                        enumeration_transferred = false;
                        smoothing = DualPriceSmoothing::new(options.smoothing.clone());
                        proximal_state.reset();
                        out.warm_start.proximal = None;
                        smoothing.exclude_rows(controller.unstabilized_rows());
                        round_lower_bound = if inherited_bound.is_finite() {
                            inherited_bound
                        } else {
                            -1e12
                        };
                        round_primal_bound = 1e12;
                        stage = schedule.len() - 1;
                        stage_bound = -1e12;
                        round_penalty_updates = 0;
                        continue;
                    }
                    return finish(out, loaded, cut_artificials, pool, smoothing);
                }
            }
            if !positive && controller.request_enumeration_checkpoint(master)? {
                out.enumeration_requested = true;
                return finish(out, loaded, cut_artificials, pool, smoothing);
            }
            if !positive && !converged && !enumeration_transferred {
                let transferred = match super::enumeration::transfer(
                    master,
                    controller,
                    &raw,
                    &out,
                    &loaded,
                    &cut_artificials,
                    &pool,
                    &pool_seen,
                    options.enumerated_columns_added,
                ) {
                    Ok(transferred) => transferred,
                    Err(error) => {
                        // The optional handoff restored pricing state and left
                        // the live LP untouched. Continue ordinary CG/separation.
                        out.enumeration_failure = Some(error.to_string());
                        None
                    }
                };
                if let Some(next) = transferred {
                    enumeration_transferred = true;
                    *master = next.master;
                    out = next.output;
                    loaded = next.loaded;
                    cut_artificials = next.cut_artificials;
                    seen = next.seen;
                    pool = next.pool;
                    pool_seen = next.pool_seen;
                    // eval restarts the phase walk whether or not the master changed.
                    smoothing.restart_schedule((schedule.len() - 1) as i32);
                    proximal_state.restart_schedule();
                    if pricing_estimate.is_finite() {
                        round_lower_bound = pricing_estimate.max(round_lower_bound);
                        // A completed CG round accepts its LP objective at the
                        // configured convergence tolerance for gap bookkeeping.
                        // The published pricing certificate stays unchanged.
                        if !positive
                            && gap_smaller_than_tolerance(
                                pricing_estimate,
                                out.lp,
                                options.gap_absolute,
                                options.gap_relative,
                            )
                        {
                            round_lower_bound = round_lower_bound.max(out.lp);
                        }
                    }
                    round_primal_bound = 1e12;
                    ipm_accuracy = IpmAccuracy::new();
                    stage = schedule.len() - 1;
                    stage_bound = -1e12;
                    round_penalty_updates = 0;
                    continue;
                }
            }
            if !positive && controller.separates() && controller.needs_pricing_checkpoint(master) {
                let mut checkpoint = finish(
                    out.clone(),
                    loaded.clone(),
                    cut_artificials.clone(),
                    pool.clone(),
                    smoothing.clone(),
                )?;
                checkpoint.status = Status::Optimal;
                controller.store_pricing_checkpoint(master, checkpoint)?;
            }
            let mut stop_cuts = !controller.separates() || out.cut_rounds >= options.max_cut_rounds;
            if !stop_cuts && separated_once {
                let denominator = (0.1 * lp_at_separation.abs()).max(1e-9);
                let gain = (out.lp - lp_at_separation) / denominator;
                if gain < options.tailing_off_threshold - 1e-6 {
                    tailing += 1;
                } else {
                    tailing = 0;
                }
                if tailing >= options.tailing_off_counter {
                    stop_cuts = true;
                }
            }
            lp_at_separation = out.lp;
            if !stop_cuts {
                let mut rows_remapped = false;
                // A pass closed by its existing bound has not run pricing.
                // Keep its solved model; cleanup is enabled by a completed
                // exact pricing pass, not merely by entering separation.
                if options.remove_basic_cuts && separated_once && !converged {
                    let protected = controller.protected_rows();
                    let rows: Vec<_> = master
                        .cleanup_cut_rows(initial_rows)?
                        .into_iter()
                        .filter(|r| !protected.contains(r))
                        .collect();
                    let columns: Vec<usize> = cut_artificials
                        .iter()
                        .filter(|(r, _)| rows.binary_search(r).is_ok())
                        .map(|&(_, c)| c)
                        .collect();
                    if !rows.is_empty() {
                        rows_remapped = true;
                        let (row_map, map) = master.remove_cuts(&rows, &columns)?;
                        loaded = loaded
                            .into_iter()
                            .zip(&map)
                            .filter_map(|(c, m)| m.map(|_| c))
                            .collect();
                        out.column_tags = out
                            .column_tags
                            .iter()
                            .zip(&map)
                            .filter_map(|(&t, m)| m.map(|_| t))
                            .collect();
                        out.artificials = out.artificials.iter().filter_map(|&i| map[i]).collect();
                        for column in loaded.iter_mut().flatten().chain(&mut pool) {
                            column.entries = column
                                .entries
                                .iter()
                                .filter_map(|&(row, value)| row_map[row].map(|r| (r, value)))
                                .collect();
                        }
                        smoothing.remap_rows(&row_map);
                        proximal_state.reset();
                        out.warm_start.proximal = None;
                        cut_artificials = cut_artificials
                            .iter()
                            .filter_map(|&(r, c)| Some((row_map[r]?, map[c]?)))
                            .collect();
                        controller.remap_columns(&map)?;
                        controller.remap_rows(&row_map)?;
                    }
                }
                if !master.solved() && !master.solve() {
                    return Err(invalid(
                        "master could not restore its solution before separation",
                    ));
                }
                let previous_rows = master.rhs().len();
                let added = controller.separate(master, &out.column_tags)?;
                let coefficients_changed =
                    added > 0 || master.rhs().len() != previous_rows || !master.solved();
                // Add the cut artificials in one call, then delete
                // the columns a strengthened relaxation forbids.
                if added > 0 && options.cut_artificial_cost > 0. {
                    let rows = previous_rows..master.rhs().len();
                    for (row, column) in
                        append_artificials(master, rows, options.cut_artificial_cost)?
                    {
                        out.artificials.push(column);
                        out.column_tags.push(-1);
                        loaded.push(None);
                        cut_artificials.push((row, column));
                    }
                }
                let removed = controller.take_removed_columns();
                if !removed.is_empty() {
                    let (_, map) = master.remove_cuts(&[], &removed)?;
                    loaded = loaded
                        .into_iter()
                        .zip(&map)
                        .filter_map(|(c, m)| m.map(|_| c))
                        .collect();
                    out.column_tags = out
                        .column_tags
                        .iter()
                        .zip(&map)
                        .filter_map(|(&t, m)| m.map(|_| t))
                        .collect();
                    out.artificials = out.artificials.iter().filter_map(|&i| map[i]).collect();
                    cut_artificials = cut_artificials
                        .iter()
                        .filter_map(|&(r, c)| Some((r, map[c]?)))
                        .collect();
                    controller.remap_columns(&map)?;
                }
                // Separation may add rows or change existing cut memories.
                // Keep warm metadata synchronized with the actual LP, and
                // let the controller rebuild coefficients for unloaded paths.
                for (index, column) in loaded.iter_mut().enumerate() {
                    if let Some(column) = column {
                        column.entries = master.column_entries(index)?;
                    }
                }
                if coefficients_changed {
                    controller.synchronize_generated_columns(&mut pool, &loaded)?;
                }
                if coefficients_changed
                    || rows_remapped
                    || !removed.is_empty()
                    || loaded
                        .iter()
                        .flatten()
                        .any(|column| column.identity.is_empty())
                {
                    if coefficients_changed || rows_remapped || !removed.is_empty() {
                        enumeration_transferred = false;
                    }
                    seen = loaded.iter().flatten().map(Column::signature).collect();
                }
                if coefficients_changed || rows_remapped {
                    pool_seen = pool.iter().map(Column::signature).collect();
                }
                if coefficients_changed || rows_remapped || !removed.is_empty() {
                    proximal_state.reset();
                    out.warm_start.proximal = None;
                }
                if added > 0 {
                    feasibility_attempted = false;
                    out.cut_rounds += 1;
                    round_penalty_updates = 0;
                    smoothing.restart_schedule((schedule.len() - 1) as i32);
                    proximal_state.reset();
                    out.warm_start.proximal = None;
                    if pricing_estimate.is_finite() {
                        round_lower_bound = pricing_estimate.max(round_lower_bound);
                        // A completed CG round accepts its LP objective at the
                        // configured convergence tolerance for gap bookkeeping.
                        // The published pricing certificate stays unchanged.
                        if !positive
                            && gap_smaller_than_tolerance(
                                pricing_estimate,
                                out.lp,
                                options.gap_absolute,
                                options.gap_relative,
                            )
                        {
                            round_lower_bound = round_lower_bound.max(out.lp);
                        }
                    }
                    round_primal_bound = 1e12;
                    // New rows invalidate the old primal upper bound.
                    ipm_accuracy = IpmAccuracy::new();
                    separated_once = true;
                    stage = schedule.len() - 1;
                    stage_bound = -1e12;
                    out.columns = master.costs().len();
                    continue;
                }
            }
            if positive {
                let mut present: HashSet<_> =
                    loaded.iter().flatten().map(Column::signature).collect();
                let restored = controller.restore_feasible_columns(&raw)?;
                let mut inserted = 0;
                for column in restored {
                    if !column.acceptable {
                        continue;
                    }
                    let identity = column.signature();
                    if !present.insert(identity.clone()) {
                        continue;
                    }
                    let index = master.add_column(column.cost, &column.entries)?;
                    apply_column_cap(master, controller, index, column.tag)?;
                    seen.insert(identity.clone());
                    if pool_seen.insert(identity) {
                        pool.push(column.clone());
                    }
                    out.column_tags.push(column.tag);
                    out.inserted_tags.push(column.tag);
                    loaded.push(Some(column));
                    inserted += 1;
                }
                if inserted > 0 {
                    // The original domain can contain feasible columns omitted
                    // by a restricted pricing cache. Re-solve before accepting
                    // an LP point or interpreting exhausted artificial penalties.
                    out.columns = master.costs().len();
                    pricing_estimate = f64::NEG_INFINITY;
                    out.numerical_bound = inherited_numerical;
                    if out.bound_mode == BoundMode::Numerical {
                        out.bound = out.numerical_bound;
                    }
                    ipm_accuracy = IpmAccuracy::new();
                    smoothing.restart_schedule((schedule.len() - 1) as i32);
                    proximal_state.reset();
                    out.warm_start.proximal = None;
                    stage = schedule.len() - 1;
                    stage_bound = -1e12;
                    round_penalty_updates = 0;
                    continue;
                }
                if started.elapsed() >= options.time_limit {
                    out.status = Status::TimeLimit;
                    return finish(out, loaded, cut_artificials, pool, smoothing);
                }
            }
            out.status = if positive {
                // Exact pricing proves optimality for these finite artificial
                // penalties, not infeasibility after removing the artificials.
                // A feasible real column can simply cost more than the last
                // allowed penalty. No phase-I/Farkas certificate exists here.
                Status::Stalled
            } else {
                Status::Optimal
            };
            if positive {
                out.lp = 1e12;
            }
            return finish(out, loaded, cut_artificials, pool, smoothing);
        }
        if advanced {
            continue;
        }
        if fresh == 0 {
            out.status = Status::Stalled;
            return finish(out, loaded, cut_artificials, pool, smoothing);
        }
    }
    finish(out, loaded, cut_artificials, pool, smoothing)
}
