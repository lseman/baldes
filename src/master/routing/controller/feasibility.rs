//! Independent routing phase I, preserving original rows and route identities.
use super::*;
use crate::master::cg::{FeasibilityRecovery, FeasibilityReport, FeasibilityStatus, WarmStart};
use std::time::{Duration, Instant};

impl<const D: usize> RootController<D> {
    pub(super) fn recover_phase_one(
        &mut self,
        master: &ClpMaster,
        warm: &WarmStart,
        remaining: Duration,
        iterations: usize,
        tolerance: f64,
        bound_mode: BoundMode,
    ) -> Result<Option<FeasibilityRecovery>> {
        if !tolerance.is_finite() || tolerance < 0. {
            return Err(invalid("phase-I tolerance"));
        }
        // Every real variable must have a routing identity. An unknown static
        // variable needs its own pricing/feasibility contract.
        let artificial: HashSet<_> = warm.artificials.iter().copied().collect();
        if warm.columns.len() != master.costs.len()
            || warm.artificials.iter().any(|&c| c >= master.costs.len())
            || warm
                .columns
                .iter()
                .enumerate()
                .any(|(c, column)| column.is_none() && !artificial.contains(&c))
        {
            return Ok(None);
        }
        let started = Instant::now();
        let Some(deadline) = started.checked_add(remaining) else {
            return Err(invalid("phase-I deadline"));
        };
        let mut report = FeasibilityReport {
            status: FeasibilityStatus::IterationLimit,
            iterations: 0,
            pricing_calls: 0,
            simplex_iterations: 0,
            ipm_iterations: 0,
            seconds: 0.,
            artificial_activity: None,
            lower_bound: None,
            certified_infeasible: false,
        };
        if remaining.is_zero() || iterations == 0 {
            if remaining.is_zero() {
                report.status = FeasibilityStatus::TimeLimit;
            }
            return Ok(Some(FeasibilityRecovery {
                columns: vec![],
                report,
            }));
        }
        let mut phase_problem = (*self.problem).clone();
        phase_problem.model.networks[0]
            .graph
            .suppress_coupled_costs();
        for arc in &mut phase_problem.model.networks[0].graph.arcs {
            arc.cost = 0.;
            arc.reduced_cost = 0.;
        }
        let mut phase_master = master.try_clone()?;
        phase_master.objective_offset = 0.;
        for column in 0..phase_master.costs.len() {
            phase_master.set_cost(column, 0.)?;
            if artificial.contains(&column) {
                phase_master.set_upper(column, 0.)?;
            }
        }
        // Every row, including inherited cuts and late branch rows, receives
        // its own unrestricted violation columns. These never leave phase I.
        let mut violations = vec![];
        for row in 0..phase_master.rhs.len() {
            for sign in [1., -1.] {
                if (sign > 0. && phase_master.senses[row] == RowSense::LessEqual)
                    || (sign < 0. && phase_master.senses[row] == RowSense::GreaterEqual)
                {
                    continue;
                }
                let column = phase_master.add_column(1., &[(row, sign)])?;
                phase_master.set_upper(column, f64::MAX)?;
                violations.push(column);
            }
        }
        let original = self.state.borrow();
        let initial_route_count = original.routes.len();
        let mut state = State {
            solver: original.solver.fork_for_node(),
            routes: original.routes.clone(),
            route_tags: original.route_tags.clone(),
            incumbent: f64::INFINITY,
            enumerate: false,
            rounded: false,
            pricing: adapter::RootPricing::default(),
            previous_fixing_gap: 1e12,
            deadline: Some(original.deadline.map_or(deadline, |end| end.min(deadline))),
            pricing_seconds: 0.,
            cut: original.cut.clone(),
            fixing_calls: 0,
            enumerated_routes: 0,
        };
        drop(original);
        // Cost-dependent fixing, completion bounds and cutoff enumeration from
        // the original objective cannot restrict feasibility pricing.
        state.solver.restore_arcs();
        let mut phase_columns = warm.columns.clone();
        let mut coefficients_changed = false;
        state.solver.rollback_point_saved = false;
        state.solver.graph.suppress_coupled_costs();
        state.solver.requires_rollback = false;
        for arc in &mut state.solver.graph.arcs {
            arc.cost = 0.;
            arc.reduced_cost = 0.;
        }
        state.solver.knapsack_exact_coefficients = true;
        state.solver.exact_cut_duals = true;
        state.solver.phases = vec![crate::pricing::phases::Phase {
            check_ng: true,
            jump_arcs: true,
            labels_per_bucket: i64::MAX,
        }];
        let mut seen: HashSet<_> = warm.columns.iter().flatten().map(|c| c.tag).collect();
        let mut recovered = vec![];
        let proof_cache = super::super::certificate::SearchCache::default();
        for _ in 0..iterations {
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                report.status = FeasibilityStatus::TimeLimit;
                break;
            }
            report.iterations += 1;
            let outcome = phase_master.solve_for_column_generation(remaining, None)?;
            if outcome != crate::master::LpSolveOutcome::Optimal {
                report.status = match outcome {
                    crate::master::LpSolveOutcome::TimeLimit => FeasibilityStatus::TimeLimit,
                    crate::master::LpSolveOutcome::IterationLimit => {
                        FeasibilityStatus::IterationLimit
                    }
                    // Even a restricted elastic LP failure is not an
                    // original-domain infeasibility result.
                    _ => FeasibilityStatus::Stalled,
                };
                break;
            }
            let violation = violations.iter().try_fold(0., |sum, &c| {
                let value = phase_master.values[c];
                (value.is_finite() && value >= -tolerance).then_some(sum + value.max(0.))
            });
            let Some(violation) = violation.filter(|value| value.is_finite()) else {
                report.status = FeasibilityStatus::Stalled;
                break;
            };
            report.artificial_activity = Some(violation);
            if violation <= tolerance && real_point_feasible(&phase_master, &violations, tolerance)?
            {
                report.status = FeasibilityStatus::Feasible;
                break;
            }
            // First test the actual inherited matrix. A feasible full-memory
            // universe needs no coefficient transition or pricing. Only an
            // infeasible restricted master needs ordinary labeling recovery.
            if !state.cut.enumerated.is_empty() {
                super::pool::restore_limited_memory_columns(
                    &self.problem,
                    &mut state,
                    &mut phase_master,
                    &mut phase_columns,
                )?;
                coefficients_changed = phase_columns
                    .iter()
                    .flatten()
                    .zip(warm.columns.iter().flatten())
                    .any(|(phase, original)| phase.entries != original.entries);
                if coefficients_changed {
                    // The previous duals and artificial activity describe the
                    // old coefficients. Solve the new auxiliary LP first.
                    continue;
                }
            }
            // Project to row signs and the unit violation-column dual bounds.
            // Price at this exact point, rather than mixing projected duals
            // with a minimum obtained at a different point.
            let duals: Vec<_> = phase_master
                .duals
                .iter()
                .zip(&phase_master.senses)
                .map(|(&dual, sense)| match sense {
                    RowSense::LessEqual => dual.clamp(-1., 0.),
                    RowSense::GreaterEqual => dual.clamp(0., 1.),
                    RowSense::Equal => dual.clamp(-1., 1.),
                })
                .collect();
            if duals.iter().any(|dual| !dual.is_finite()) {
                report.status = FeasibilityStatus::Stalled;
                break;
            }
            let maximum = state.solver.controls.max_columns.max(1);
            state.solver.controls.max_columns = maximum;
            state.pricing.rhs = phase_master.rhs.clone();
            report.pricing_calls += 1;
            let priced = state.price(&phase_problem, 0, &duals, 0);
            let result = match priced {
                Ok(result) => result,
                Err(error)
                    if error.kind == crate::core::error::ErrorKind::Runtime
                        && error.message.contains("time limit") =>
                {
                    report.status = FeasibilityStatus::TimeLimit;
                    break;
                }
                Err(error) => return Err(error),
            };
            let mut minimum = result.minimum;
            let returned = result.columns.len();
            let mut inserted = 0;
            for phase_column in result.columns {
                let tag = phase_column.tag;
                let mut route = state.routes[tag as usize].clone();
                let (original_cost, _) = self.problem.route_cost(&route);
                route.original_cost = original_cost;
                state.routes[tag as usize].original_cost = original_cost;
                minimum = minimum.min(phase_column.reduced_cost);
                if phase_column.reduced_cost >= -tolerance || !seen.insert(tag) {
                    continue;
                }
                phase_master.add_column(0., &phase_column.entries)?;
                // Coefficients and original costs are reconstructed through
                // the same current row mapping and cut replay as ordinary CG.
                recovered.push(state.column(&self.problem, &route, &duals, tag)?);
                inserted += 1;
            }
            if result.exact && minimum.is_finite() {
                report.lower_bound =
                    routing_feasibility_bound(&phase_problem, &phase_master, &duals, minimum);
                if report.lower_bound.is_some_and(|bound| bound > tolerance) {
                    // Floating resource extension can reject a boundary path
                    // admitted by an exact-real conservative relaxation. Even
                    // numerical objective policy must not promote that oracle
                    // observation to original-domain infeasibility.
                    report.certified_infeasible = super::super::certificate::lower_bound(
                        &phase_problem,
                        &state,
                        &phase_master,
                        &proof_cache,
                    )
                    .or_else(|| {
                        super::super::certificate::zero_objective_lower_bound(
                            &phase_problem,
                            &phase_master,
                            &duals,
                            deadline,
                        )
                    })
                    .is_some_and(|bound| bound > tolerance);
                    if report.certified_infeasible {
                        report.status = FeasibilityStatus::Infeasible;
                        break;
                    }
                }
            }
            if inserted == 0 {
                // A bounded batch can contain only already loaded routes
                // whose quantity caps are active. Ask for more paths before
                // declaring recovery stalled; the shared budgets still apply.
                if result.exact && minimum < -tolerance && returned == maximum {
                    if let Some(larger) = maximum.checked_mul(2) {
                        state.solver.controls.max_columns = larger;
                        continue;
                    }
                }
                if let Some((status, certified)) =
                    crate::master::branching::enforcement::phase_one_stall(
                        &self.problem,
                        result.exact,
                        bound_mode,
                        || {
                            super::super::certificate::lower_bound(
                                &phase_problem,
                                &state,
                                &phase_master,
                                &proof_cache,
                            )
                            .or_else(|| {
                                super::super::certificate::zero_objective_lower_bound(
                                    &phase_problem,
                                    &phase_master,
                                    &duals,
                                    deadline,
                                )
                            })
                            .is_some_and(|bound| bound > tolerance)
                        },
                    )
                {
                    report.status = status;
                    report.certified_infeasible = certified;
                    break;
                }
                report.status = FeasibilityStatus::Stalled;
                break;
            }
        }
        // Publish append-only route identities after the private operation has
        // completed. Neither original objective/pricing state nor parent/sibling
        // snapshots has been mutated while solving the auxiliary problem.
        let mut original = self.state.borrow_mut();
        for route in state.routes.iter().skip(initial_route_count) {
            register(&mut original, route)?;
        }
        original.pricing_seconds += state.pricing_seconds;
        original.pricing.oracle_calls += state.pricing.oracle_calls;
        // Recovery may have replaced enumeration's full-memory coefficients
        // with ordinary limited-memory coefficients. In that case the resumed
        // master must make the same transition, even without a pricing call;
        // otherwise its unchanged matrix contradicts the auxiliary witness.
        // Preserve a complete universe when neither coefficients nor pricing
        // changed, as a feasibility check alone does not invalidate coverage.
        if !recovered.is_empty()
            || (report.status == FeasibilityStatus::Feasible
                && (report.pricing_calls > 0 || coefficients_changed))
        {
            original.solver.restore_arcs();
        }
        report.simplex_iterations = phase_master.simplex_iterations;
        report.ipm_iterations = phase_master.ipm_iterations;
        report.seconds = started.elapsed().as_secs_f64();
        Ok(Some(FeasibilityRecovery {
            columns: recovered,
            report,
        }))
    }
}

/// A feasible auxiliary point must remain feasible when violation columns are
/// discarded, within the configured numerical LP tolerance.
fn real_point_feasible(master: &ClpMaster, violations: &[usize], tolerance: f64) -> Result<bool> {
    let artificial: HashSet<_> = violations.iter().copied().collect();
    let mut activity = vec![0.; master.rhs.len()];
    for (column, &value) in master.values.iter().enumerate() {
        if artificial.contains(&column) {
            continue;
        }
        if !value.is_finite() || value < -tolerance || value > master.upper[column] + tolerance {
            return Ok(false);
        }
        for (row, coefficient) in master.column_entries(column)? {
            activity[row] += coefficient * value;
        }
    }
    Ok(activity
        .iter()
        .zip(&master.rhs)
        .zip(&master.senses)
        .all(|((&value, &rhs), sense)| {
            value.is_finite()
                && match sense {
                    RowSense::LessEqual => value <= rhs + tolerance,
                    RowSense::GreaterEqual => value >= rhs - tolerance,
                    RowSense::Equal => (value - rhs).abs() <= tolerance,
                }
        }))
}

/// Numerical Lagrangian bound on total artificial activity. Prove the total
/// route quantity from the original fleet row; no backend infinity or guessed
/// oracle multiplicity becomes a bound. Unit violation columns have nonnegative
/// reduced costs at the projected row prices.
fn routing_feasibility_bound<const D: usize>(
    problem: &RoutingContext<D>,
    master: &ClpMaster,
    duals: &[f64],
    minimum: f64,
) -> Option<f64> {
    let row = *problem.row_map.borrow().first()?;
    if master.senses.get(row) != Some(&RowSense::LessEqual)
        || !master.rhs[row].is_finite()
        || master.rhs[row] < 0.
        || problem.model.networks[0]
            .route_entries
            .iter()
            .any(|&(r, a)| r == 0 && a != 0.)
    {
        return None;
    }
    for (arc, entries) in problem.model.networks[0]
        .graph
        .arcs
        .iter()
        .zip(&problem.model.networks[0].arc_entries)
    {
        let coefficient: f64 = entries
            .iter()
            .filter(|&&(r, _)| r == 0)
            .map(|&(_, a)| a)
            .sum();
        if coefficient
            != if arc.head == problem.model.networks[0].graph.sink {
                1.
            } else {
                0.
            }
        {
            return None;
        }
    }
    let bound = duals
        .iter()
        .zip(&master.rhs)
        .map(|(y, b)| y * b)
        .sum::<f64>()
        + master.rhs[row] * minimum.min(0.);
    bound.is_finite().then_some(bound)
}

#[cfg(test)]
#[path = "../../../../tests/unit/master/routing/controller/feasibility_tests.rs"]
mod tests;
