//! Reduced-cost bucket-arc fixing without a route-universe handoff.
use super::*;

impl<const D: usize> RootController<D> {
    pub(super) fn fix_without_enumeration(&mut self, duals: &[f64], bound: f64) -> Result<()> {
        let mut state = self.state.borrow_mut();
        // Enumeration already performs its own fixing. Without enumeration,
        // require an actual incumbent and a bound for these pricing duals.
        if state.enumerate
            || !bound.is_finite()
            || !state.incumbent.is_finite()
            || state.solver.options.get("BALDESapplyReducedCostFixing")? == 0.
        {
            return Ok(());
        }
        let gap = state.incumbent - bound;
        let ratio = gap / state.previous_fixing_gap;
        let threshold = 0.9f32 as f64;
        if gap <= 0.
            || !gap.is_finite()
            || ratio >= threshold - (threshold.abs().max(ratio.abs()) * 1e-10 + 1e-6)
        {
            return Ok(());
        }
        let remaining = state.remaining_time()?;
        let limit = state
            .solver
            .options
            .get("BALDESredCostFixingTimeThreshold")?;
        let budget = remaining.min(state.solver.controls.time_limit).min(
            std::time::Duration::try_from_secs_f64(limit.max(0.))
                .map_err(|_| invalid("fixing time threshold"))?,
        );
        if budget.is_zero() {
            return Ok(());
        }
        let started = std::time::Instant::now();
        // Keep live exclusions and caches intact on errors or timeouts. The
        // next ordinary pricing call refreshes this fork at its new duals.
        let mut solver = state.solver.fork_for_node();
        let saved_limit = solver.controls.time_limit;
        solver.controls.time_limit = budget;
        let offset =
            self.problem
                .reprice(&mut solver, duals, false, &state.cut.capacity_arc_entries)?;
        // Fix at the raw objective, including the active cut duals. Cached
        // rounded arc costs must not authorize permanent exclusions.
        let saved_modes = (
            solver.cuts.round_duals,
            solver.exact_cut_duals,
            solver.knapsack_exact_coefficients,
        );
        solver.cuts.round_duals = false;
        solver.exact_cut_duals = true;
        solver.knapsack_exact_coefficients = true;
        let margin = 1e-6 * (1. + state.incumbent.abs() + bound.abs());
        let cutoff = gap + margin - offset;
        let result = solver.run_fixing_enumeration(-1, 0., cutoff);
        state.pricing_seconds += started.elapsed().as_secs_f64();
        state.fixing_calls += 1;
        match result {
            Ok(_) => {
                solver.controls.time_limit = saved_limit;
                solver.cuts.round_duals = saved_modes.0;
                solver.exact_cut_duals = saved_modes.1;
                solver.knapsack_exact_coefficients = saved_modes.2;
                state.solver = solver;
                state.previous_fixing_gap = gap;
                Ok(())
            }
            // An optional fixing pass may exhaust its own budget. Ordinary
            // pricing continues from the untouched live state.
            Err(error)
                if error.message.contains("time limit")
                    || error.message.contains("label limit")
                    || error.message.contains("budget") =>
            {
                state.previous_fixing_gap = gap;
                Ok(())
            }
            Err(error) => Err(error),
        }
    }
}
