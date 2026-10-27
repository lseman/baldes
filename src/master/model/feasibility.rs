//! Private all-network phase I with independently directed infeasibility proof.
//! Unit violations and zero objective costs never enter the original master.
use super::*;
use crate::master::{
    cg::{FeasibilityRecovery, FeasibilityReport, FeasibilityStatus},
    LpMaster, LpSolveOutcome,
};
use std::time::Duration;

pub(super) fn recover<const D: usize>(
    original: &Rc<RefCell<State<D>>>,
    master: &ClpMaster,
    warm: &cg::WarmStart,
    remaining: Duration,
    iterations: usize,
    tolerance: f64,
    _bound_mode: cg::BoundMode,
) -> Result<FeasibilityRecovery> {
    let started = Instant::now();
    let deadline = started
        .checked_add(remaining)
        .ok_or_else(|| invalid("network phase-I deadline"))?;
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
        return Ok(FeasibilityRecovery {
            columns: vec![],
            report,
        });
    }
    let mut state = original.borrow().fork_state();
    let machine_domain = state
        .oracles
        .iter()
        .any(|o| o.network.graph.front_program_domain());
    state.deadline = state.deadline.min(deadline);
    state.scheduler = Scheduler::new(
        NetworkPricingSchedule::FullSweep,
        state.oracles.len(),
        tolerance,
    );
    if state.enumeration_report.coverage_cutoff.is_some() {
        report.status = FeasibilityStatus::Stalled;
        return Ok(FeasibilityRecovery {
            columns: vec![],
            report,
        });
    }
    for oracle in &mut state.oracles {
        Arc::make_mut(&mut oracle.network.graph).suppress_coupled_costs();
        for arc in &mut Arc::make_mut(&mut oracle.network.graph).arcs {
            arc.cost = 0.;
            arc.reduced_cost = 0.;
        }
        oracle.solver.restore_arcs();
        oracle.solver.rollback_point_saved = false;
        oracle.solver.requires_rollback = false;
        oracle.solver.cutoff = f64::INFINITY;
        oracle.solver.graph.suppress_coupled_costs();
        for arc in &mut oracle.solver.graph.arcs {
            arc.cost = 0.;
            arc.reduced_cost = 0.;
        }
        oracle.solver.knapsack_exact_coefficients = true;
        oracle.solver.exact_cut_duals = true;
        oracle.solver.phases = vec![crate::pricing::phases::Phase {
            check_ng: true,
            jump_arcs: true,
            labels_per_bucket: i64::MAX,
        }];
    }
    for variable in &mut state.master_variables {
        variable.cost = 0.;
    }
    for route in &mut state.routes {
        let route = Arc::make_mut(route);
        route.route.original_cost = 0.;
        route.route.cost = 0.;
    }
    for fixed in &mut state.fixed_routes {
        fixed.route.route.original_cost = 0.;
        fixed.route.route.cost = 0.;
    }
    // Suppressing charges can change the IEEE instruction sequence of a
    // machine-front feasibility model. Its Farkas proof must therefore use a
    // separate enlarged physical/scalar domain, never assume that the zero-cost
    // front program has exactly the original machine program's accepted paths.
    let proof_relaxation = if machine_domain {
        let mut proof = state.fork_state();
        proof.universe = None;
        proof.route_upper_bounds.clear();
        for oracle in &mut proof.oracles {
            Arc::make_mut(&mut oracle.network.graph).set_coupled_resources(&[])?;
            oracle.solver.graph.set_coupled_resources(&[])?;
        }
        Some(proof)
    } else {
        None
    };
    let mut private = master.try_clone()?;
    private.objective_offset = 0.;
    let old_artificial: HashSet<_> = warm
        .artificials
        .iter()
        .copied()
        .chain(warm.cut_artificials.iter().map(|&(_, c)| c))
        .collect();
    for column in 0..private.costs.len() {
        private.set_cost(column, 0.)?;
        if old_artificial.contains(&column) {
            private.set_upper(column, 0.)?;
        }
    }
    let mut violations = vec![];
    for row in 0..private.rhs.len() {
        for sign in [1., -1.] {
            if (sign > 0. && private.senses[row] == RowSense::LessEqual)
                || (sign < 0. && private.senses[row] == RowSense::GreaterEqual)
            {
                continue;
            }
            let column = private.add_column(1., &[(row, sign)])?;
            private.set_upper(column, f64::MAX)?;
            violations.push(column);
        }
    }
    let mut recovered_tags = vec![];
    let mut seen: HashSet<_> = warm.columns.iter().flatten().map(|c| c.tag).collect();
    // Known but currently inactive routes remain part of a complete universe.
    // A capped route has its semantic cap restored when inserted here.
    for _ in 0..iterations {
        let remaining = state.deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            report.status = FeasibilityStatus::TimeLimit;
            break;
        }
        report.iterations += 1;
        let outcome = private.solve_for_column_generation(remaining, None)?;
        if outcome != LpSolveOutcome::Optimal {
            report.status = match outcome {
                LpSolveOutcome::TimeLimit => FeasibilityStatus::TimeLimit,
                LpSolveOutcome::IterationLimit => FeasibilityStatus::IterationLimit,
                _ => FeasibilityStatus::Stalled,
            };
            break;
        }
        let activity = violations.iter().try_fold(0., |sum, &c| {
            let x = private.values[c];
            (x.is_finite() && x >= -tolerance).then_some(sum + x.max(0.))
        });
        report.artificial_activity = activity.filter(|x| x.is_finite());
        if report.artificial_activity.is_some_and(|x| x <= tolerance)
            && real_point_feasible(&private, &violations, tolerance)?
        {
            report.status = FeasibilityStatus::Feasible;
            break;
        }
        let duals: Vec<_> = private
            .duals
            .iter()
            .zip(&private.senses)
            .map(|(&p, sense)| match sense {
                RowSense::LessEqual => p.clamp(-1., 0.),
                RowSense::GreaterEqual => p.clamp(0., 1.),
                RowSense::Equal => p.clamp(-1., 1.),
            })
            .collect();
        if duals.iter().any(|p| !p.is_finite()) {
            report.status = FeasibilityStatus::Stalled;
            break;
        }
        report.pricing_calls += 1;
        let answer = match state.price(0, &duals) {
            Ok(answer) => answer,
            Err(error)
                if error.message.contains("time limit")
                    || error.message.contains("label limit") =>
            {
                report.status = FeasibilityStatus::TimeLimit;
                break;
            }
            Err(error) => return Err(error),
        };
        // Every enabled network must complete exact feasibility pricing. The
        // independent directed bound additionally covers the original domain,
        // including resources/cost suppression and semantic variable caps.
        if answer.exact && state.scheduler.complete {
            report.lower_bound = bounds::certificate(
                proof_relaxation.as_ref().unwrap_or(&state),
                &private,
                &duals,
            );
            if report.lower_bound.is_some_and(|bound| bound > tolerance) {
                report.certified_infeasible = true;
                report.status = FeasibilityStatus::Infeasible;
                break;
            }
        }
        let mut inserted = 0;
        for column in answer.columns {
            if !column.acceptable || column.reduced_cost >= -tolerance || !seen.insert(column.tag) {
                continue;
            }
            let c = private.add_column(0., &column.entries)?;
            if let Ok(tag) = usize::try_from(column.tag) {
                recovered_tags.push(tag);
            }
            let cap = if let Some(tag) = usize::try_from(column.tag).ok() {
                enumeration::route_upper(&state, tag)
            } else if let Some((index, sign)) = master_variables::decode(column.tag) {
                Some(master_variables::upper(
                    &state.master_variables[index],
                    sign,
                ))
            } else {
                None
            };
            private.set_upper(c, cap.filter(|x| x.is_finite()).unwrap_or(f64::MAX))?;
            inserted += 1;
        }
        if inserted == 0 {
            report.status = FeasibilityStatus::Stalled;
            break;
        }
    }
    let mut columns = vec![];
    let origin = original.borrow();
    let mut tags = origin.tags.clone();
    let mut routes = origin.routes.clone();
    if report.status != FeasibilityStatus::Infeasible {
        for &candidate_tag in &recovered_tags {
            let candidate = &state.routes[candidate_tag];
            let oracle = origin
                .oracles
                .iter()
                .find(|o| o.network.id == candidate.network)
                .unwrap();
            let Some(cost) = original_cost_for_recovery(oracle, &candidate.route, &mut report)?
            else {
                continue;
            };
            let coefficients = entries(oracle, &candidate.route, origin.shared_offset)?;
            let mut route = candidate.as_ref().clone();
            route.route.original_cost = cost;
            route.route.cost = cost;
            let identity = route.route.identity();
            let key = (route.network, identity.clone());
            let tag = if let Some(&tag) = tags.get(&key) {
                tag
            } else {
                let tag = i32::try_from(routes.len())
                    .map_err(|_| invalid("phase-I network route tag overflow"))?;
                tags.insert(key, tag);
                routes.push(Arc::new(route));
                tag
            };
            columns.push(cg::Column {
                cost,
                reduced_cost: -1.,
                entries: coefficients,
                acceptable: true,
                tag,
                identity: format!("{}:{identity}", candidate.network),
            });
        }
    }
    drop(origin);
    // Publish original-cost metadata only after every fallible replay succeeds.
    let mut origin = original.borrow_mut();
    origin.tags = tags;
    origin.routes = routes;
    for (old, phase) in origin.oracles.iter_mut().zip(&state.oracles) {
        old.calls += phase.calls;
    }
    report.seconds = started.elapsed().as_secs_f64();
    Ok(FeasibilityRecovery { columns, report })
}

/// A private zero-cost program can accept more paths than the original
/// machine-front program. Such a candidate is neither an original feasible
/// column nor a modelling error: omit it and keep the recovery outcome unknown.
fn original_cost_for_recovery<const D: usize>(
    oracle: &Oracle<D>,
    route: &Route,
    report: &mut FeasibilityReport,
) -> Result<Option<f64>> {
    let path = route
        .arcs
        .iter()
        .map(|id| oracle.index[id])
        .collect::<Vec<_>>();
    let Some(coupled) = route.coupled_cost(&oracle.network.graph, &path)? else {
        if report.status == FeasibilityStatus::Feasible {
            report.status = FeasibilityStatus::Stalled;
        }
        return Ok(None);
    };
    let mut ids = route.arcs.clone();
    ids.sort_unstable();
    let cost = ids
        .iter()
        .map(|id| oracle.network.graph.arcs[oracle.index[id]].cost)
        .sum::<f64>()
        + coupled;
    if !cost.is_finite() {
        return Err(invalid("recovered network phase-I original cost overflow"));
    }
    Ok(Some(cost))
}

fn real_point_feasible(master: &ClpMaster, violations: &[usize], tolerance: f64) -> Result<bool> {
    let mut activity = vec![0.; master.rhs.len()];
    for (column, &value) in master.values.iter().enumerate() {
        if violations.contains(&column) {
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
        .all(|((&x, &rhs), sense)| {
            x.is_finite()
                && match sense {
                    RowSense::Equal => (x - rhs).abs() <= tolerance,
                    RowSense::LessEqual => x <= rhs + tolerance,
                    RowSense::GreaterEqual => x >= rhs - tolerance,
                }
        }))
}

#[cfg(test)]
#[path = "../../../tests/unit/master/model/feasibility_tests.rs"]
mod tests;
