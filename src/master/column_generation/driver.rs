use super::controller::apply_column_cap;
use super::*;

/// Generic column generation loop that works with any `LpMaster` backend.
pub fn optimize<M: LpMaster, P: FnMut(usize, &[f64]) -> Result<Answer>, C: Controller<M>>(
    master: &mut M,
    supplied: &[Phase],
    price: P,
    controller: &mut C,
    options: Options,
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
    let mut progress = crate::master::progress::ProgressLog::new(options.log_interval, counts);
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
    options: Options,
    progress: &mut crate::master::progress::ProgressLog,
) -> Result<Output> {
    let schedule = phases::initialize(0, supplied)?;
    if !options.multiplicity.is_finite() || options.multiplicity < 0. || options.max_iterations == 0
    {
        return Err(invalid("column generation options"));
    }
    let started = Instant::now();
    // A zero alpha factor disables the reference's smoothing workflow, not
    // only publication of smoothed duals.
    let mut smoothing = DualPriceSmoothing::new(options.smoothing.clone());
    let stabilize = options.stabilize && smoothing.parameters.alpha_factor > 0.;
    let inherited_center = options
        .warm_start
        .as_ref()
        .and_then(|warm| warm.stabilization.as_ref());
    if let Some(previous) = inherited_center {
        smoothing.inherit_center(previous, (schedule.len() - 1) as i32);
    }
    smoothing.exclude_rows(controller.unstabilized_rows());
    if let Some(warm) = &options.warm_start {
        if (!warm.bound.is_finite() && warm.bound != f64::NEG_INFINITY)
            || [warm.numerical_bound, warm.certified_bound]
                .iter()
                .any(|b| !b.is_finite() && *b != f64::NEG_INFINITY)
            || warm.bound
                != match warm.bound_mode {
                    BoundMode::Numerical => warm.numerical_bound,
                    BoundMode::Certified => warm.certified_bound,
                }
            || warm.columns.len() != master.costs().len()
            || warm.initial_rows > master.rhs().len()
            || warm.artificials.iter().any(|&c| c >= master.costs().len())
            || warm
                .cut_artificials
                .iter()
                .any(|&(r, c)| r >= master.rhs().len() || c >= master.costs().len())
            || warm
                .columns
                .iter()
                .flatten()
                .chain(&warm.generated_pool)
                .any(|column| column.entries.iter().any(|&(r, _)| r >= master.rhs().len()))
        {
            return Err(invalid("warm master metadata dimensions"));
        }
    }
    let inherited_numerical = options
        .warm_start
        .as_ref()
        .map_or(f64::NEG_INFINITY, |w| w.numerical_bound);
    let inherited_certified = options
        .warm_start
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
    let mut loaded: Vec<Option<Column>> = options.warm_start.as_ref().map_or_else(
        || options.known.iter().cloned().map(Some).collect(),
        |w| w.columns.clone(),
    );
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
    let mut pool: Vec<Column> = options
        .warm_start
        .as_ref()
        .map_or_else(Vec::new, |warm| warm.generated_pool.clone());
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
        artificials: options
            .warm_start
            .as_ref()
            .map_or_else(Vec::new, |w| w.artificials.clone()),
        penalty_updates: 0,
        pruned_by_bound: false,
        cut_rounds: 0,
        warm_start: WarmStart {
            bound: f64::NEG_INFINITY,
            bound_mode: options.bound_mode,
            numerical_bound: f64::NEG_INFINITY,
            certified_bound: f64::NEG_INFINITY,
            stabilization: None,
            generated_pool: vec![],
            columns: vec![],
            artificials: vec![],
            cut_artificials: vec![],
            initial_rows: 0,
        },
    };
    out.column_tags.resize(master.costs().len(), -1);
    if options.artificial_cost > 0. && options.warm_start.is_none() {
        for r in 0..master.rhs().len() {
            if master.senses()[r] != RowSense::LessEqual {
                out.artificials
                    .push(master.add_column(options.artificial_cost, &[(r, 1.)])?);
                loaded.push(None);
                out.column_tags.push(-1);
            }
            if master.senses()[r] != RowSense::GreaterEqual {
                out.artificials
                    .push(master.add_column(options.artificial_cost, &[(r, -1.)])?);
                loaded.push(None);
                out.column_tags.push(-1);
            }
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
    let initial_rows = options
        .warm_start
        .as_ref()
        .map_or(master.rhs().len(), |w| w.initial_rows);
    let mut cut_artificials: Vec<(usize, usize)> = options
        .warm_start
        .as_ref()
        .map_or_else(Vec::new, |w| w.cut_artificials.clone());
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
            stabilization: stabilize.then_some(smoothing),
            generated_pool: pool,
            columns: loaded,
            artificials: out.artificials.clone(),
            cut_artificials,
            initial_rows,
        };
        Ok(out)
    };
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
                iteration == 1 && inherited_center.is_none(),
            );
        }
        // earlyCGtermType1 uses the phase value for stage > 0
        // and the LP pricing estimate for stage zero, before another call.
        let stopping_bound = if stage > 0 {
            stage_bound
        } else {
            pricing_estimate
        };
        let converged = accurate
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
            let mut current = if stabilize {
                smoothing.pricing.clone()
            } else {
                raw.clone()
            };
            let mut pending = vec![];
            loop {
                let mut answer = price(stage, &current)?;
                let mut created = vec![];
                for column in &answer.columns {
                    let identity = column.signature();
                    if column.acceptable && !pool_seen.contains(&identity) {
                        pool.push(column.clone());
                        created.push(identity);
                    }
                }
                // Preserve duplicates produced within one oracle batch, as
                // with restricted-master insertion below.
                pool_seen.extend(created);
                answer.exact |= schedule[stage].exact();
                out.pricing_calls += 1;
                fresh = answer
                    .columns
                    .iter()
                    .filter(|c| {
                        c.acceptable
                            && master_reduced_cost(c, &raw) < -options.tolerance
                            && !seen.contains(&c.signature())
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
                let mut gradient = master.rhs().to_vec();
                let best = answer
                    .columns
                    .iter()
                    .filter(|c| c.acceptable)
                    .min_by(|a, b| a.reduced_cost.total_cmp(&b.reduced_cost));
                let supplied_gradient = controller.pricing_subgradient(master.rhs());
                let subgradient_available = supplied_gradient.is_some() || best.is_some();
                if let Some(supplied) = supplied_gradient {
                    if supplied.len() != gradient.len() || supplied.iter().any(|g| !g.is_finite()) {
                        return Err(invalid("invalid pricing subgradient"));
                    }
                    gradient = supplied;
                } else if let Some(best) = best {
                    // A returned pricing route contributes to the subgradient
                    // even when its rounded reduced cost is zero. Filtering it
                    // by that cost changes the reference's angle inputs.
                    for &(r, a) in &best.entries {
                        if let Some(g) = gradient.get_mut(r) {
                            *g -= options.multiplicity * a;
                        }
                    }
                }
                pending.push(answer.columns);
                if !stabilize {
                    break;
                }
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
            advanced = fresh == 0 && !exact && !penalties && stage > 0;
            if advanced {
                stage -= 1;
                stage_bound = -1e12;
                round_penalty_updates = 0;
            }
            let first = loaded.len();
            for batch in pending {
                let mut identities = vec![];
                for column in batch {
                    if !column.acceptable
                        || (!options.insert_non_negative
                            && master_reduced_cost(&column, &raw) >= -options.tolerance)
                    {
                        continue;
                    }
                    let identity = column.signature();
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
                    &seen,
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
            if !positive && controller.separates() {
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
                    let mut artificials: Vec<(usize, f64)> = vec![];
                    for row in previous_rows..master.rhs().len() {
                        if master.senses()[row] != RowSense::LessEqual {
                            artificials.push((row, 1.));
                        }
                        if master.senses()[row] != RowSense::GreaterEqual {
                            artificials.push((row, -1.));
                        }
                    }
                    let entries: Vec<[(usize, f64); 1]> =
                        artificials.iter().map(|&e| [e]).collect();
                    let batch: Vec<_> = entries
                        .iter()
                        .map(|e| (options.cut_artificial_cost, e.as_slice()))
                        .collect();
                    let columns = master.add_columns(&batch)?;
                    for (c, &(row, _)) in columns.zip(&artificials) {
                        out.artificials.push(c);
                        out.column_tags.push(-1);
                        loaded.push(None);
                        cut_artificials.push((row, c));
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
                if added > 0 {
                    feasibility_attempted = false;
                    out.cut_rounds += 1;
                    round_penalty_updates = 0;
                    smoothing.restart_schedule((schedule.len() - 1) as i32);
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
