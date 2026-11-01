//! Incumbent-based reduced-cost bucket-arc fixing in the shared model.
use super::*;
use std::time::Duration;

/// Convert the necessary whole-solution cutoff to additive pricing costs.
/// Every operation must round upward: large complementary contributions and
/// an opposite route offset can otherwise cancel away an improving route.
fn pricing_ceiling(incumbent: f64, other: f64, margin: f64, offset: f64) -> Option<f64> {
    use crate::core::rounding::add_up;
    add_up(add_up(add_up(incumbent, -other)?, margin)?, -offset)
}

/// The caller has completed accurate pricing without improving columns. A
/// bound computed for a smoothed pricing point cannot be paired with the raw
/// duals supplied here, so derive a directed bound for these duals instead.
pub(super) fn apply<const D: usize>(
    state: &mut State<D>,
    master: &ClpMaster,
    duals: &[f64],
) -> Result<()> {
    if !state.integer_primal_termination {
        return Ok(());
    }
    let incumbent = state.tree_incumbent.min(
        state
            .pricing_witness
            .as_ref()
            .filter(|_| state.integer_primal_termination)
            .map_or(f64::INFINITY, |witness| witness.0),
    );
    if !incumbent.is_finite() {
        return Ok(());
    }
    if state.oracles.is_empty()
        || state.universe.is_some()
        || !state.original_variable_costs.is_empty()
    {
        return Ok(());
    }
    let Some(bound) = bounds::certificate(state, master, duals).filter(|value| value.is_finite())
    else {
        return Ok(());
    };
    let gap = incumbent - bound;
    let ratio = gap / state.previous_fixing_gap;
    let threshold = f64::from(0.9_f32);
    if gap <= 0.
        || !gap.is_finite()
        || ratio >= threshold - (threshold.abs().max(ratio.abs()) * 1e-10 + 1e-6)
    {
        return Ok(());
    }
    let mut work = 10_000_000;
    let Some(plan) =
        bounds::cutoff_plan(state, master, duals, incumbent, &mut work, state.deadline)
    else {
        return Ok(());
    };
    // Stage the entire pricing collection: one failed network must not leave
    // the others with exclusions from an uncommitted attempt.
    let mut staged = state.fork_state();
    let mut changed = false;
    let mut attempted = false;
    for (index, cutoff) in plan.into_iter().enumerate() {
        let bounds::NetworkCutoff::Cost {
            other_contributions,
            ..
        } = cutoff
        else {
            continue;
        };
        let solver = &staged.oracles[index].solver;
        if solver.options.get("BALDESapplyReducedCostFixing")? == 0. {
            continue;
        }
        let seconds = solver.options.get("BALDESredCostFixingTimeThreshold")?;
        let budget = staged
            .deadline
            .saturating_duration_since(Instant::now())
            .min(solver.controls.time_limit)
            .min(
                Duration::try_from_secs_f64(seconds.max(0.))
                    .map_err(|_| invalid("fixing time threshold"))?,
            );
        if budget.is_zero() {
            continue;
        }
        let offset = adapter::cache_duals(&mut staged, index, duals)?;
        let margin = 1e-6 * (1. + incumbent.abs() + bound.abs());
        let Some(ceiling) = pricing_ceiling(incumbent, other_contributions, margin, offset) else {
            continue;
        };
        let solver = &mut staged.oracles[index].solver;
        let saved_limit = solver.controls.time_limit;
        let modes = (
            solver.cuts.round_duals,
            solver.exact_cut_duals,
            solver.knapsack_exact_coefficients,
        );
        solver.controls.time_limit = budget;
        solver.cuts.round_duals = false;
        solver.exact_cut_duals = true;
        solver.knapsack_exact_coefficients = true;
        attempted = true;
        let report = solver.run_fixing_enumeration(-1, 0., ceiling);
        solver.controls.time_limit = saved_limit;
        solver.cuts.round_duals = modes.0;
        solver.exact_cut_duals = modes.1;
        solver.knapsack_exact_coefficients = modes.2;
        match report {
            Ok(report) => {
                changed |= report.eliminated > 0;
            }
            Err(error)
                if error.message.contains("time limit")
                    || error.message.contains("label limit")
                    || error.message.contains("budget") =>
            {
                return Ok(())
            }
            Err(error) => return Err(error),
        }
    }
    if !attempted {
        return Ok(());
    }
    for (live, staged) in state.oracles.iter_mut().zip(staged.oracles) {
        live.solver = staged.solver;
        live.costs = staged.costs;
    }
    state.previous_fixing_gap = gap;
    state.objective_fixed |= changed;
    Ok(())
}

#[cfg(test)]
#[path = "../../../tests/unit/master/model/fixing_tests.rs"]
mod tests;
