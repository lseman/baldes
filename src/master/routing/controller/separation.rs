//! Cut-round priority scheduling, tailing off and batched insertion.
use super::*;

pub(super) fn separation_gap_closed(value: f64, incumbent: f64) -> bool {
    if !incumbent.is_finite() {
        return false;
    }
    let gap = incumbent - value;
    gap <= value.abs().max(incumbent.abs()) * 1e-10 + 1e-6 || gap / incumbent.abs() <= 1e-9
}

impl<const D: usize> RootController<D> {
    pub(super) fn separate_round(&mut self, master: &mut ClpMaster, tags: &[i32]) -> Result<usize> {
        let mut state = self.state.borrow_mut();
        let state = &mut *state;
        let value = master.objective();
        // A solution using an improper route of a relaxed selection never
        // ends the node: tighten (or drop stale columns) before any stop test.
        if self.problem.model.networks[0].graph.selection_relaxed() {
            let values = master.values().to_vec();
            let tagged: Vec<(usize, usize)> = tags
                .iter()
                .enumerate()
                .filter(|(_, &t)| t >= 0)
                .map(|(i, &t)| (i, t as usize))
                .collect();
            if self.uses_improper_selection(state, &tagged, &values) {
                let rank_one_in_master = state.cut.rows.iter().any(Option::is_some);
                if !self.improve_relaxation(state, rank_one_in_master, &tagged, &values)? {
                    // The memory already excludes them; remove the stale columns.
                    let mut removed = vec![];
                    for &(column, tag) in &tagged {
                        let route = &state.routes[tag];
                        if !state
                            .solver
                            .route_satisfies_relaxation(&route.nodes, &route.arcs)?
                        {
                            removed.push(column);
                        }
                    }
                    state.cut.removed_columns = removed;
                }
                return Ok(1);
            }
        }
        // Every converged column generation, round 0 included, reports its
        // node depth. Bucket-step adjustment is a root-only decision.
        let depth = i32::try_from(state.pricing.node_depth).unwrap_or(i32::MAX);
        let stop = state.solver.column_generation_terminated(0, depth)?;
        if state.cut.rounds == 0 {
            state.cut.lp_before_cuts = value;
        } else {
            // Alg4EvalByColAndCutGen::eval: the gain is relative to the gap to
            // the incumbent (an integral master solution) when that gap is at
            // most a tenth of the bound, else to a tenth of the bound.
            let previous = state.cut.separation_lp;
            let upper = if state.incumbent.is_finite() {
                state.incumbent
            } else {
                1e12
            };
            let gap = upper - previous;
            let tenth = (0.1 * previous.abs()).max(1e-9);
            let denominator = if gap - (gap.abs().max(tenth) * 1e-10 + 1e-6) > tenth {
                tenth
            } else {
                gap
            };
            let gain = (value - previous) / denominator;
            let rank_active = state.cut.rank_one_active();
            let min_rounds = state.cut.options.min_rounds_before_stop_by_sp;
            if rank_active && stop && state.cut.rounds >= min_rounds {
                return Ok(0);
            }
            // Store the threshold as a float.
            let threshold = state.cut.options.tailing_off_threshold as f32 as f64;
            if gain < threshold - (threshold.abs().max(gain.abs()) * 1e-10 + 1e-6) {
                if state.cut.options.enabled
                    && state.cut.level <= RANK_ONE_PRIORITY
                    && !state.solver.cuts.separation_phase(&state.solver.options)?.1
                {
                    let options = &state.solver.options;
                    state.solver.cuts.increase_phase(options)?;
                } else {
                    state.cut.tailing += 1;
                }
                // Tailing off at a level moves to the next lower priority, and
                // stops cut generation at the lowest one.
                if state.cut.tailing >= state.cut.options.tailing_off_counter
                    && state.cut.rounds >= min_rounds
                {
                    match state.cut.next_level() {
                        Some(level) => {
                            state.cut.level = level;
                            state.cut.tailing = 0;
                        }
                        None => return Ok(0),
                    }
                }
            }
        }
        state.cut.separation_lp = value;
        // Each cut round first checks gapSmallerThanTol: separation stops once
        // the bound reaches the incumbent (absolute 1e-6, relative 1e-9).
        if separation_gap_closed(value, state.incumbent) {
            return Ok(0);
        }
        let values = master.values().to_vec();
        let tagged: Vec<(usize, usize)> = tags
            .iter()
            .enumerate()
            .filter(|(_, &t)| t >= 0)
            .map(|(i, &t)| (i, t as usize))
            .collect();
        let mut fractional = vec![];
        for &(index, tag) in &tagged {
            if values[index] > 1e-9 {
                let route = &state.routes[tag];
                fractional.push(FractionalColumn {
                    vertices: State::<D>::customers(route),
                    value: values[index],
                    graph_id: -1,
                    enumerated: state.is_enumerated(route),
                });
            }
        }
        // Separation sees the complete solution, including fixed routes.
        // New rows are then shifted once for the residual master below.
        for (route, quantity) in &self.problem.fixed_routes {
            fractional.push(FractionalColumn {
                vertices: State::<D>::customers(route),
                value: f64::from(*quantity),
                graph_id: -1,
                enumerated: true,
            });
        }
        // Alg4EvalOfNode::addCutToMaster: separators by decreasing priority.
        // Those at or above the level run; a lower one runs, and becomes the
        // level, only while nothing has been added this round.
        let mut level = state.cut.level;
        let mut rows = PendingRows::new(master.rhs().len());
        // Cuts separated this round reach the formulation after the loop.
        let rank_one_in_master = state.cut.rows.iter().any(Option::is_some);
        let mut added = 0;
        let mut success = false;
        for (priority, separator) in state.cut.separators() {
            if priority < level {
                if success {
                    break;
                }
                level = priority;
            }
            let (count, changed) = match separator {
                Separator::Capacity => (
                    self.separate_capacity(state, &mut rows, &tagged, &fractional)?,
                    false,
                ),
                Separator::KPath => {
                    self.separate_kpath(state, master, &mut rows, &tagged, &fractional)?
                }
                Separator::Knapsack => (
                    self.separate_knapsack(state, &mut rows, &tagged, &values)?,
                    false,
                ),
                Separator::RankOne => {
                    self.separate_rank_one(state, master, &mut rows, &tagged, &fractional)?
                }
                Separator::Relaxation => (
                    0,
                    self.improve_relaxation(state, rank_one_in_master, &tagged, &values)?,
                ),
            };
            added += count;
            success |= count > 0 || changed;
        }
        state.cut.level = level;
        // Add the round's cuts to Clp in one call.
        rows.install(master)?;
        if success {
            state.cut.rounds += 1;
        }
        Ok(if added > 0 { added } else { success as usize })
    }
}
