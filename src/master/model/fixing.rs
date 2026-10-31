//! Incumbent-based reduced-cost bucket-arc fixing in the shared model.
use super::*;
use crate::master::pricing::Networks;
use std::time::Duration;

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
    if state.oracles.len() != 1
        || state.universe.is_some()
        || !state.master_variables.is_empty()
        || !state.original_variable_costs.is_empty()
        // A positive mandatory quantity contributes a positive minimum route
        // cost to the bound. The routing cutoff below assumes an optional
        // network and cannot subtract that mandatory contribution safely.
        || state.oracles[0].network.multiplicity.lower != 0.
        || state.oracles[0]
            .solver
            .options
            .get("BALDESapplyReducedCostFixing")?
            == 0.
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
    let solver = &state.oracles[0].solver;
    let seconds = solver.options.get("BALDESredCostFixingTimeThreshold")?;
    let budget = state
        .deadline
        .saturating_duration_since(Instant::now())
        .min(solver.controls.time_limit)
        .min(
            Duration::try_from_secs_f64(seconds.max(0.))
                .map_err(|_| invalid("fixing time threshold"))?,
        );
    if budget.is_zero() {
        return Ok(());
    }
    let margin = 1e-6 * (1. + incumbent.abs() + bound.abs());
    if !margin.is_finite() || !(gap + margin).is_finite() {
        return Ok(());
    }

    // Reprice and fix only a private solver. Keep all live labels, exclusions,
    // and cut inputs intact if preparation or the fixing pass is interrupted.
    let started = Instant::now();
    let fork = state.oracles[0].solver.fork_for_node();
    let original = std::mem::replace(&mut state.oracles[0].solver, fork);
    let prepared = (|| {
        let tolerance = state.scheduler.tolerance;
        let request = adapter::NetworkAdapter { state, tolerance }.prepare(0, 0, duals, false)?;
        let cutoff = gap + margin - request.offset;
        if !cutoff.is_finite() {
            return Ok(None);
        }
        let oracle = &mut state.oracles[0];
        oracle
            .solver
            .cache_physical_costs(&oracle.costs, &oracle.costs)?;
        let remaining = budget.saturating_sub(started.elapsed());
        if remaining.is_zero() {
            return Ok(None);
        }
        let solver = &mut oracle.solver;
        let saved_limit = solver.controls.time_limit;
        let saved_modes = (
            solver.cuts.round_duals,
            solver.exact_cut_duals,
            solver.knapsack_exact_coefficients,
        );
        solver.controls.time_limit = remaining;
        solver.cuts.round_duals = false;
        solver.exact_cut_duals = true;
        solver.knapsack_exact_coefficients = true;
        let result = solver.run_fixing_enumeration(-1, 0., cutoff);
        solver.controls.time_limit = saved_limit;
        solver.cuts.round_duals = saved_modes.0;
        solver.exact_cut_duals = saved_modes.1;
        solver.knapsack_exact_coefficients = saved_modes.2;
        result.map(Some)
    })();
    match prepared {
        Ok(Some(report)) => {
            state.previous_fixing_gap = gap;
            state.objective_fixed |= report.eliminated > 0;
            Ok(())
        }
        Ok(None) => {
            state.oracles[0].solver = original;
            Ok(())
        }
        Err(error)
            if error.message.contains("time limit")
                || error.message.contains("label limit")
                || error.message.contains("budget") =>
        {
            state.oracles[0].solver = original;
            state.previous_fixing_gap = gap;
            Ok(())
        }
        Err(error) => {
            state.oracles[0].solver = original;
            Err(error)
        }
    }
}

#[cfg(test)]
#[path = "../../../tests/unit/master/model/fixing_tests.rs"]
mod tests;
