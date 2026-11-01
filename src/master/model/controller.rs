//! Model adapter for column generation, separation and pricing checkpoints.
use super::*;
use crate::master::LpMaster;

pub(super) struct ColumnGenerationController<
    const D: usize = { crate::DEFAULT_RESOURCE_DIMENSIONS },
> {
    pub(super) state: Rc<RefCell<State<D>>>,
    pub(super) request_initial_enumeration: bool,
    pub(super) search_inactive_columns: bool,
    pub(super) artificial_penalty: Option<f64>,
    pub(super) enumeration_options: Option<EnumerationOptions>,
    pub(super) enumeration_checkpoint: Option<(Vec<NetworkRoute>, Vec<(i32, usize)>, Instant)>,
    pub(super) removed_columns: Vec<usize>,
    pub(super) trust_preprocessing_infeasibility: bool,
    pub(super) allow_objective_fixing: bool,
    pub(super) checkpoints: CheckpointStore<D>,
}

impl<const D: usize> cg::Controller<ClpMaster> for ColumnGenerationController<D> {
    fn proven_infeasible(&self) -> bool {
        self.trust_preprocessing_infeasibility
            && self.state.borrow().preprocessing.propagation.infeasible
    }
    fn master_solved(
        &mut self,
        master: &ClpMaster,
        tags: &[i32],
        artificials: &[usize],
    ) -> Result<()> {
        self.artificial_penalty = artificials
            .iter()
            .try_fold(f64::INFINITY, |minimum, &index| {
                let cost = *master.costs().get(index)?;
                (cost.is_finite() && cost > 0.).then_some(minimum.min(cost))
            });
        let mut state = self.state.borrow_mut();
        let objective = master.objective();
        state.pricing_master =
            (master.solved() && objective.is_finite() && master.objective_offset.is_finite())
                .then_some((objective, master.objective_offset));
        if !state.integer_primal_termination && !separation::tracks_integer_witness(&state) {
            return Ok(());
        }
        if let Some(witness) = tree_adapter::replay_integer_witness(
            &state,
            master,
            |column| tags.get(column).copied(),
            state.witness_tolerance,
        )? {
            if state
                .pricing_witness
                .as_ref()
                .map_or(true, |old| witness.0 < old.0)
            {
                state.pricing_witness = Some(witness);
            }
        }
        Ok(())
    }
    fn incumbent(&self) -> f64 {
        let state = self.state.borrow();
        if state.integer_primal_termination {
            state
                .pricing_witness
                .as_ref()
                .map_or(f64::INFINITY, |w| w.0)
        } else {
            f64::INFINITY
        }
    }
    fn separates(&self) -> bool {
        super::separation::separates(&self.state.borrow())
    }
    fn needs_pricing_checkpoint(&self, _master: &ClpMaster) -> bool {
        self.state.borrow().cuts.is_some()
    }
    fn store_pricing_checkpoint(
        &mut self,
        master: &ClpMaster,
        mut output: cg::Output,
    ) -> Result<()> {
        if let Some(prior) = &self.checkpoints.prior_cg {
            solve::prepend_cg_output(prior, &mut output);
        }
        let mut state = self.state.borrow_mut();
        let rank_one_present = state
            .cuts
            .as_ref()
            .is_some_and(|cuts| cuts.rank_one_cuts_present());
        if let Some(auto) = self.checkpoints.auto_rank_one.as_mut() {
            if rank_one_present {
                auto.node_elapsed = auto.rank_one_started_at.elapsed();
            }
        }
        let incumbent = state
            .pricing_witness
            .as_ref()
            .map_or(f64::INFINITY, |w| w.0)
            .min(state.cuts.as_ref().map_or(f64::INFINITY, |c| c.upper_bound));
        if self.checkpoints.rollback.is_some()
            && separation::separation_gap_closed(master.objective(), incumbent)
        {
            return Ok(());
        }
        let saved_state = state.snapshot_for_checkpoint();
        if let Some(auto) = self.checkpoints.auto_rank_one.as_mut() {
            if !rank_one_present {
                auto.pre_rank_one = Some(SavedNodePoint {
                    master: master.try_clone()?,
                    output: output.clone(),
                    state: saved_state.snapshot_for_checkpoint(),
                });
                auto.rank_one_started_at = Instant::now();
                auto.node_elapsed = std::time::Duration::ZERO;
            }
        }
        self.checkpoints.rollback = Some(SavedNodePoint {
            master: master.try_clone()?,
            output,
            state: saved_state,
        });
        for oracle in &mut state.oracles {
            oracle.solver.rollback_point_saved = true;
        }
        Ok(())
    }
    fn inactive_rows(&self) -> Vec<usize> {
        self.state.borrow().retired_rows.iter().copied().collect()
    }
    fn unstabilized_rows(&self) -> Vec<usize> {
        self.inactive_rows()
    }
    fn protected_rows(&self) -> Vec<usize> {
        separation::protected_rows(&self.state.borrow())
    }
    fn remap_rows(&mut self, rows: &[Option<usize>]) -> Result<()> {
        separation::remap_rows(&mut self.state.borrow_mut(), rows)
    }
    fn requires_separation(&self, master: &ClpMaster, tags: &[i32]) -> Result<bool> {
        super::separation::requires_separation(&mut self.state.borrow_mut(), master, tags)
    }
    fn separate(&mut self, master: &mut ClpMaster, tags: &[i32]) -> Result<usize> {
        let mut state = self.state.borrow_mut();
        if state.cuts.is_none() {
            let (changed, removed) = separation::tighten_selections(&mut state, master, tags)?;
            self.removed_columns = removed;
            Ok(changed)
        } else {
            separation::separate(&mut state, master, tags)
        }
    }
    fn take_removed_columns(&mut self) -> Vec<usize> {
        let mut removed = std::mem::take(&mut self.removed_columns);
        removed.extend(
            self.state
                .borrow_mut()
                .cuts
                .as_mut()
                .map_or_else(Vec::new, |cuts| std::mem::take(&mut cuts.removed_columns)),
        );
        removed.sort_unstable();
        removed.dedup();
        removed
    }
    fn cut_counts(&self) -> Vec<(&'static str, usize)> {
        super::separation::cut_counts(&self.state.borrow())
    }
    fn bucket_steps(&self) -> Vec<f64> {
        let state = self.state.borrow();
        state.oracles.first().map_or_else(Vec::new, |oracle| {
            oracle
                .solver
                .pricing_engine()
                .map_or_else(Vec::new, |engine| {
                    engine.grid.step[..engine.grid.dimensions].to_vec()
                })
        })
    }
    /// Rebuild cached generated columns after cut rows were added or cut
    /// memories changed; master-variable and artificial columns are kept.
    fn refresh_generated_columns(&mut self, pool: &mut [cg::Column]) -> Result<()> {
        let state = self.state.borrow();
        for column in pool {
            let Some(route) = usize::try_from(column.tag)
                .ok()
                .and_then(|tag| state.routes.get(tag))
            else {
                continue;
            };
            let oracle = state
                .oracles
                .iter()
                .find(|o| o.network.id == route.network)
                .ok_or_else(|| invalid("cached route network"))?;
            column.entries = retirement::active_entries(&state, oracle, &route.route)?;
        }
        Ok(())
    }
    fn synchronize_generated_columns(
        &mut self,
        pool: &mut [cg::Column],
        loaded: &[Option<cg::Column>],
    ) -> Result<()> {
        let state = self.state.borrow();
        if !separation::supports_row_cleanup(&state) {
            drop(state);
            return self.refresh_generated_columns(pool);
        }
        let dynamic = separation::dynamic_rows(&state);
        let mut by_tag = vec![None; state.routes.len()];
        for column in loaded.iter().flatten() {
            if let Some(slot) = usize::try_from(column.tag)
                .ok()
                .and_then(|tag| by_tag.get_mut(tag))
            {
                *slot = Some(column);
            }
        }
        for column in pool {
            let current = usize::try_from(column.tag)
                .ok()
                .and_then(|tag| by_tag.get(tag))
                .and_then(|column| *column)
                .filter(|current| {
                    current.identity == column.identity && current.cost == column.cost
                });
            if let Some(current) = current {
                column.entries.retain(|(row, _)| !dynamic.contains(row));
                column.entries.extend(
                    current
                        .entries
                        .iter()
                        .copied()
                        .filter(|(row, value)| dynamic.contains(row) && *value != 0.),
                );
                column.entries.sort_by_key(|&(row, _)| row);
            } else if let Some(route) = usize::try_from(column.tag)
                .ok()
                .and_then(|tag| state.routes.get(tag))
            {
                let oracle = state
                    .oracles
                    .iter()
                    .find(|oracle| oracle.network.id == route.network)
                    .ok_or_else(|| invalid("cached route network"))?;
                column.entries = retirement::active_entries(&state, oracle, &route.route)?;
            }
        }
        Ok(())
    }
    fn original_witness_available(&self) -> bool {
        let state = self.state.borrow();
        state.integer_primal_termination && state.pricing_witness.is_some()
    }
    fn curvature_rows(&self, row_count: usize) -> Vec<cg::CurvatureRow> {
        let state = self.state.borrow();
        (0..row_count)
            .map(|row| {
                state
                    .curvature_rows
                    .get(row)
                    .copied()
                    .unwrap_or(cg::CurvatureRow {
                        class: cg::CurvatureRowClass::Excluded,
                        active: false,
                    })
            })
            .collect()
    }
    fn recover_feasibility(
        &mut self,
        master: &ClpMaster,
        warm: &cg::WarmStart,
        remaining: std::time::Duration,
        iterations: usize,
        tolerance: f64,
        bound_mode: cg::BoundMode,
    ) -> Result<Option<cg::FeasibilityRecovery>> {
        super::feasibility::recover(
            &self.state,
            master,
            warm,
            remaining,
            iterations,
            tolerance,
            bound_mode,
        )
        .map(Some)
    }
    fn fix_pricing(&mut self, master: &ClpMaster, duals: &[f64]) -> Result<()> {
        self.state.borrow_mut().fixing_bound = None;
        self.state.borrow_mut().fixing_cutoffs = None;
        if self.enumeration_options.is_none() && self.allow_objective_fixing {
            let bound = bounds::certificate(&self.state.borrow(), master, duals);
            self.state.borrow_mut().fixing_bound = bound.filter(|value| value.is_finite());
            if self.state.borrow().gap_enumeration {
                let state = self.state.borrow();
                let upper = state.tree_incumbent.min(
                    state
                        .pricing_witness
                        .as_ref()
                        .map_or(f64::INFINITY, |w| w.0),
                );
                let mut work = 10_000_000;
                let plan =
                    bounds::cutoff_plan(&state, master, duals, upper, &mut work, state.deadline);
                drop(state);
                self.state.borrow_mut().fixing_cutoffs = plan.map(|plan| {
                    plan.into_iter()
                        .map(|cutoff| match cutoff {
                            bounds::NetworkCutoff::Empty { .. } => None,
                            bounds::NetworkCutoff::Cost {
                                other_contributions,
                                ..
                            } => crate::core::rounding::add_up(upper, -other_contributions),
                        })
                        .collect()
                });
            } else {
                fixing::apply(&mut self.state.borrow_mut(), master, duals)?;
            }
        }
        Ok(())
    }
    fn preserve_enumeration_column(&self, column: &cg::Column) -> bool {
        master_variables::decode(column.tag).is_some()
    }
    fn begin_enumeration_transaction(&mut self, _bound: f64) -> Result<bool> {
        if !self.allow_objective_fixing || self.enumeration_options.is_some() {
            return Ok(false);
        }
        let Some(bound) = self.state.borrow().fixing_bound else {
            return Ok(false);
        };
        gap_enumeration::begin(&mut self.state.borrow_mut(), bound)
    }
    fn finish_enumeration_transaction(&mut self, commit: bool) {
        gap_enumeration::finish(&mut self.state.borrow_mut(), commit);
    }
    fn enumerated_universe(
        &mut self,
        duals: &[f64],
        bound: f64,
    ) -> Result<Option<Vec<cg::Column>>> {
        let bound = self.state.borrow().fixing_bound.unwrap_or(bound);
        gap_enumeration::universe(&mut self.state.borrow_mut(), duals, bound)
    }
    fn regenerate_enumerated_columns(
        &mut self,
        pool: &[cg::Column],
        duals: &[f64],
        bound: f64,
        _universe: &[cg::Column],
    ) -> Result<Vec<cg::Column>> {
        let bound = self.state.borrow().fixing_bound.unwrap_or(bound);
        gap_enumeration::regenerate(&mut self.state.borrow_mut(), pool, duals, bound)
    }
    fn refresh_master_columns(
        &mut self,
        master: &mut ClpMaster,
        loaded: &mut [Option<cg::Column>],
    ) -> Result<bool> {
        gap_enumeration::restore_limited_memory(&mut self.state.borrow_mut(), master, loaded)
    }
    fn enumeration_installed(
        &mut self,
        master: &ClpMaster,
        loaded: &[Option<cg::Column>],
        artificials: &[usize],
    ) -> Result<()> {
        let state = self.state.borrow();
        let Some(mut options) = state.gap_mip.clone() else {
            return Ok(());
        };
        if state.gap_pool_cutoff.is_none() {
            return Ok(());
        }
        #[cfg(feature = "highs")]
        if let Some(oracle) = state.oracles.first() {
            let limit = oracle
                .solver
                .options
                .get("BALDESmaxNumOfEnumSolutionsForMIP")?
                .max(0.) as usize;
            if state.enumerated_routes > limit {
                return Ok(());
            }
        }
        let mut warm = cg::WarmStart::empty(cg::BoundMode::Numerical);
        warm.columns = loaded.to_vec();
        warm.artificials = artificials.to_vec();
        options.time_limit = options
            .time_limit
            .min(state.deadline.saturating_duration_since(Instant::now()));
        options.objective_cutoff = options.objective_cutoff.min(
            state.tree_incumbent.min(
                state
                    .pricing_witness
                    .as_ref()
                    .map_or(f64::INFINITY, |w| w.0),
            ),
        );
        let snapshot = NodeSnapshot {
            master: master.try_clone()?,
            warm: warm.into(),
            state: state.snapshot_for_checkpoint(),
        };
        drop(state);
        if let Some(output) = snapshot.solve_enumerated_mip(&options)? {
            if let Some(solution) = output.mip.solution {
                let mut state = self.state.borrow_mut();
                if state
                    .pricing_witness
                    .as_ref()
                    .map_or(true, |old| solution.objective < old.0)
                {
                    state.pricing_witness = Some((
                        solution.objective,
                        output.route_values,
                        output.master_values,
                    ));
                }
            }
        }
        Ok(())
    }
    fn pricing_complete(&self) -> bool {
        self.state.borrow().scheduler.complete
    }
    fn request_enumeration_checkpoint(&mut self, _master: &ClpMaster) -> Result<bool> {
        let requested = self.request_initial_enumeration && self.state.borrow().universe.is_none();
        // A failed attempt must resume ordinary pricing, not request another
        // checkpoint at the same point. A later solve may retry with new limits.
        self.request_initial_enumeration = false;
        if !requested {
            return Ok(false);
        }
        let options = self
            .enumeration_options
            .as_ref()
            .ok_or_else(|| invalid("missing enumeration options"))?;
        let mut state = self.state.borrow_mut();
        state.enumeration_report.attempts += 1;
        let deadline = Instant::now()
            .checked_add(options.time_limit)
            .ok_or_else(|| invalid("enumeration time limit"))?
            .min(state.deadline);
        match enumeration::collect_routes(&state, options) {
            Ok((routes, counts)) => {
                self.enumeration_checkpoint = Some((routes, counts, deadline));
                Ok(true)
            }
            Err(reason) => {
                state.enumeration_report.failure = Some(reason);
                Ok(false)
            }
        }
    }
    fn column_upper_bound(&self, tag: i32) -> Option<f64> {
        let state = self.state.borrow();
        if tag >= 0 {
            // Ordinary generated columns share the network multiplicity row.
            // Its auxiliary violation must cost at least as much as every
            // uncapped column can improve the objective; otherwise preserve
            // the cap until penalties have increased enough.
            // Repeating that limit on every column is redundant and can move
            // the convexity dual onto bound-active columns, leaving pricing
            // with an artificially weak network offset. Explicit restrictions
            // and finite retained-universe boxes still need their own caps.
            // Keep the multi-network policy until its bound compensation and
            // certified tree behavior are validated for uncapped columns.
            let covered = self.artificial_penalty.is_some_and(|penalty| {
                penalty == f64::INFINITY
                    || state.routes.get(tag as usize).is_some_and(|route| {
                        crate::core::rounding::add_down(route.route.original_cost, penalty)
                            .is_some_and(|sum| sum >= 0.)
                    })
            });
            if covered
                && state.oracles.len() == 1
                && state.master_variables.is_empty()
                && state.variable_boxes.is_empty()
                && state.grouped_variable_boxes.is_empty()
                && state.synchronized_services.is_empty()
                && state.universe.is_none()
                && !state.route_upper_bounds.contains_key(&(tag as usize))
            {
                return None;
            }
            enumeration::route_upper(&state, tag as usize).filter(|u| u.is_finite())
        } else if let Some((index, sign)) = master_variables::decode(tag) {
            state
                .master_variables
                .get(index)
                .map(|v| master_variables::upper(v, sign))
                .filter(|u| u.is_finite())
        } else {
            None
        }
    }
    fn certified_pricing_bound(&self, master: &ClpMaster, duals: &[f64]) -> Result<Option<f64>> {
        Ok(Some(
            bounds::certificate(&self.state.borrow(), master, duals).unwrap_or(f64::NEG_INFINITY),
        ))
    }
    fn pricing_bound(&self, master: &ClpMaster, duals: &[f64]) -> Result<Option<f64>> {
        let state = self.state.borrow();
        let bound = match state.correction {
            Some(correction) => {
                duals
                    .iter()
                    .zip(&state.rhs)
                    .fold(master.objective_offset, |value, (dual, rhs)| {
                        value + dual * rhs
                    })
                    + correction
            }
            None => f64::NEG_INFINITY,
        };
        // Objective-dependent exclusions preserve improving integer solutions,
        // not the unrestricted LP optimum. Never publish a bound above the
        // feasible objective that authorized these exclusions.
        let bound = if state.objective_fixed {
            bound
                .min(state.gap_pool_cutoff.unwrap_or(f64::INFINITY))
                .min(
                    state.tree_incumbent.min(
                        state
                            .pricing_witness
                            .as_ref()
                            .map_or(f64::INFINITY, |w| w.0),
                    ),
                )
        } else {
            bound
        };
        Ok(Some(bound))
    }
    fn prepare_pricing_duals(&self, duals: &mut [f64]) {
        let state = self.state.borrow();
        for &row in &state.retired_rows {
            if let Some(value) = duals.get_mut(row) {
                *value = 0.;
            }
        }
        separation::repair_theta_duals(&state, duals);
    }
    fn pricing_subgradient(&self, _rhs: &[f64]) -> Option<Vec<f64>> {
        self.state.borrow().gradient.clone()
    }
    fn take_pricing_subgradient(&mut self, _rhs: &[f64]) -> Option<Vec<f64>> {
        self.state.borrow_mut().gradient.take()
    }
    fn uses_inactive_column_pool(&self) -> bool {
        self.search_inactive_columns
    }
    fn reprice_inactive_column(
        &mut self,
        column: &cg::Column,
        duals: &[f64],
    ) -> Result<Option<cg::Column>> {
        let state = self.state.borrow();
        let Some(tag) = usize::try_from(column.tag).ok() else {
            return Ok(None);
        };
        let Some(route) = state.routes.get(tag) else {
            return Ok(None);
        };
        let Some(oracle) = state.oracles.iter().find(|o| o.network.id == route.network) else {
            return Ok(None);
        };
        if oracle.multiplicity_upper() == 0.
            || (oracle.solver.enumeration.is_some() && !oracle.is_enumerated(&route.route))
            || state
                .universe
                .as_ref()
                .is_some_and(|universe| !universe.contains(&tag))
            || route.route.arcs.iter().any(|id| {
                oracle
                    .proved_zero_arcs
                    .get(oracle.index[id])
                    .copied()
                    .unwrap_or(false)
            })
            || !lifecycle::integer_route_feasible(oracle, &route.route).unwrap_or(false)
        {
            return Ok(None);
        }
        let entries = retirement::active_entries(&state, oracle, &route.route)?;
        let cost = route.route.original_cost;
        let reduced_cost = entries
            .iter()
            .fold(cost, |value, &(row, a)| value - duals[row] * a);
        Ok(Some(cg::Column {
            cost,
            reduced_cost,
            entries,
            acceptable: true,
            tag: column.tag,
            identity: column.identity.clone(),
        }))
    }
}

impl<const D: usize> ColumnGenerationController<D> {
    pub(super) fn recover_pricing_timeout(
        &mut self,
        error: crate::core::error::OrError,
    ) -> Result<(ClpMaster, cg::Output)> {
        if error.kind != crate::core::error::ErrorKind::Runtime
            || !error.message.contains("time limit")
            || !self
                .state
                .borrow()
                .oracles
                .iter()
                .any(|o| o.solver.requires_rollback)
        {
            return Err(error);
        }
        let Some(saved) = self.checkpoints.rollback.as_ref() else {
            return Err(error);
        };
        let restored_master = saved.master.try_clone()?;
        let mut restored_state = saved.state.snapshot_for_checkpoint();
        {
            let live = self.state.borrow();
            if let Some(witness) = &live.pricing_witness {
                if restored_state
                    .pricing_witness
                    .as_ref()
                    .map_or(true, |old| witness.0 < old.0)
                {
                    restored_state.pricing_witness = Some(witness.clone());
                }
            }
            for (dst, src) in restored_state.oracles.iter_mut().zip(&live.oracles) {
                dst.calls = dst.calls.max(src.calls);
            }
        }
        *self.state.borrow_mut() = restored_state;
        Ok((restored_master, saved.output.clone()))
    }
}
