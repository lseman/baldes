//! Per-network pricing and shared-master controller callbacks.
use super::*;
use crate::master::LpMaster;

impl<const D: usize> cg::Controller<ClpMaster> for Coordinator<D> {
    fn proven_infeasible(&self) -> bool {
        self.7 && self.0.borrow().preprocessing.propagation.infeasible
    }
    fn master_solved(
        &mut self,
        master: &ClpMaster,
        tags: &[i32],
        artificials: &[usize],
    ) -> Result<()> {
        self.3 = artificials
            .iter()
            .try_fold(f64::INFINITY, |minimum, &index| {
                let cost = *master.costs().get(index)?;
                (cost.is_finite() && cost > 0.).then_some(minimum.min(cost))
            });
        let mut state = self.0.borrow_mut();
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
        let state = self.0.borrow();
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
        super::separation::separates(&self.0.borrow())
    }
    fn needs_pricing_checkpoint(&self, _master: &ClpMaster) -> bool {
        self.0.borrow().cuts.is_some()
    }
    fn store_pricing_checkpoint(
        &mut self,
        master: &ClpMaster,
        mut output: cg::Output,
    ) -> Result<()> {
        if let Some(prior) = &self.9.prior_cg {
            solve::prepend_cg_output(prior, &mut output);
        }
        let mut state = self.0.borrow_mut();
        let rank_one_present = state
            .cuts
            .as_ref()
            .is_some_and(|cuts| cuts.rank_one_cuts_present());
        if let Some(auto) = self.9.auto_rank_one.as_mut() {
            if rank_one_present {
                auto.node_elapsed = auto.rank_one_started_at.elapsed();
            }
        }
        let incumbent = state
            .pricing_witness
            .as_ref()
            .map_or(f64::INFINITY, |w| w.0)
            .min(state.cuts.as_ref().map_or(f64::INFINITY, |c| c.upper_bound));
        if self.9.rollback.is_some()
            && separation::separation_gap_closed(master.objective(), incumbent)
        {
            return Ok(());
        }
        let saved_state = state.snapshot_for_checkpoint();
        if let Some(auto) = self.9.auto_rank_one.as_mut() {
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
        self.9.rollback = Some(SavedNodePoint {
            master: master.try_clone()?,
            output,
            state: saved_state,
        });
        for oracle in &mut state.oracles {
            oracle.solver.rollback_point_saved = true;
        }
        Ok(())
    }
    fn protected_rows(&self) -> Vec<usize> {
        separation::protected_rows(&self.0.borrow())
    }
    fn remap_rows(&mut self, rows: &[Option<usize>]) -> Result<()> {
        separation::remap_rows(&mut self.0.borrow_mut(), rows)
    }
    fn requires_separation(&self, master: &ClpMaster, tags: &[i32]) -> Result<bool> {
        super::separation::requires_separation(&mut self.0.borrow_mut(), master, tags)
    }
    fn separate(&mut self, master: &mut ClpMaster, tags: &[i32]) -> Result<usize> {
        let mut state = self.0.borrow_mut();
        if state.cuts.is_none() {
            let (changed, removed) = separation::tighten_selections(&mut state, master, tags)?;
            self.6 = removed;
            Ok(changed)
        } else {
            separation::separate(&mut state, master, tags)
        }
    }
    fn take_removed_columns(&mut self) -> Vec<usize> {
        let mut removed = std::mem::take(&mut self.6);
        removed.extend(
            self.0
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
        super::separation::cut_counts(&self.0.borrow())
    }
    fn bucket_steps(&self) -> Vec<f64> {
        let state = self.0.borrow();
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
        let state = self.0.borrow();
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
            column.entries = entries(oracle, &route.route, state.shared_offset)?;
        }
        Ok(())
    }
    fn synchronize_generated_columns(
        &mut self,
        pool: &mut [cg::Column],
        loaded: &[Option<cg::Column>],
    ) -> Result<()> {
        let state = self.0.borrow();
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
                column.entries = entries(oracle, &route.route, state.shared_offset)?;
            }
        }
        Ok(())
    }
    fn original_witness_available(&self) -> bool {
        let state = self.0.borrow();
        state.integer_primal_termination && state.pricing_witness.is_some()
    }
    fn curvature_rows(&self, row_count: usize) -> Vec<cg::CurvatureRow> {
        let state = self.0.borrow();
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
            &self.0, master, warm, remaining, iterations, tolerance, bound_mode,
        )
        .map(Some)
    }
    fn fix_pricing(&mut self, master: &ClpMaster, duals: &[f64]) -> Result<()> {
        self.0.borrow_mut().fixing_bound = None;
        self.0.borrow_mut().fixing_cutoffs = None;
        if self.4.is_none() && self.8 {
            let bound = bounds::certificate(&self.0.borrow(), master, duals);
            self.0.borrow_mut().fixing_bound = bound.filter(|value| value.is_finite());
            if self.0.borrow().gap_enumeration {
                let state = self.0.borrow();
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
                self.0.borrow_mut().fixing_cutoffs = plan.map(|plan| {
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
                fixing::apply(&mut self.0.borrow_mut(), master, duals)?;
            }
        }
        Ok(())
    }
    fn preserve_enumeration_column(&self, column: &cg::Column) -> bool {
        master_variables::decode(column.tag).is_some()
    }
    fn begin_enumeration_transaction(&mut self, _bound: f64) -> Result<bool> {
        if !self.8 || self.4.is_some() {
            return Ok(false);
        }
        let Some(bound) = self.0.borrow().fixing_bound else {
            return Ok(false);
        };
        gap_enumeration::begin(&mut self.0.borrow_mut(), bound)
    }
    fn finish_enumeration_transaction(&mut self, commit: bool) {
        gap_enumeration::finish(&mut self.0.borrow_mut(), commit);
    }
    fn enumerated_universe(
        &mut self,
        duals: &[f64],
        bound: f64,
    ) -> Result<Option<Vec<cg::Column>>> {
        let bound = self.0.borrow().fixing_bound.unwrap_or(bound);
        gap_enumeration::universe(&mut self.0.borrow_mut(), duals, bound)
    }
    fn regenerate_enumerated_columns(
        &mut self,
        pool: &[cg::Column],
        duals: &[f64],
        bound: f64,
        _universe: &[cg::Column],
    ) -> Result<Vec<cg::Column>> {
        let bound = self.0.borrow().fixing_bound.unwrap_or(bound);
        gap_enumeration::regenerate(&mut self.0.borrow_mut(), pool, duals, bound)
    }
    fn refresh_master_columns(
        &mut self,
        master: &mut ClpMaster,
        loaded: &mut [Option<cg::Column>],
    ) -> Result<bool> {
        gap_enumeration::restore_limited_memory(&mut self.0.borrow_mut(), master, loaded)
    }
    fn enumeration_installed(
        &mut self,
        master: &ClpMaster,
        loaded: &[Option<cg::Column>],
        artificials: &[usize],
    ) -> Result<()> {
        let state = self.0.borrow();
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
            warm,
            state: state.snapshot_for_checkpoint(),
        };
        drop(state);
        if let Some(output) = snapshot.solve_enumerated_mip(&options)? {
            if let Some(solution) = output.mip.solution {
                let mut state = self.0.borrow_mut();
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
        self.0.borrow().scheduler.complete
    }
    fn request_enumeration_checkpoint(&mut self, _master: &ClpMaster) -> Result<bool> {
        let requested = self.1 && self.0.borrow().universe.is_none();
        // A failed attempt must resume ordinary pricing, not request another
        // checkpoint at the same point. A later solve may retry with new limits.
        self.1 = false;
        if !requested {
            return Ok(false);
        }
        let options = self
            .4
            .as_ref()
            .ok_or_else(|| invalid("missing enumeration options"))?;
        let mut state = self.0.borrow_mut();
        state.enumeration_report.attempts += 1;
        let deadline = Instant::now()
            .checked_add(options.time_limit)
            .ok_or_else(|| invalid("enumeration time limit"))?
            .min(state.deadline);
        match enumeration::collect_routes(&state, options) {
            Ok((routes, counts)) => {
                self.5 = Some((routes, counts, deadline));
                Ok(true)
            }
            Err(reason) => {
                state.enumeration_report.failure = Some(reason);
                Ok(false)
            }
        }
    }
    fn column_upper_bound(&self, tag: i32) -> Option<f64> {
        let state = self.0.borrow();
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
            let covered = self.3.is_some_and(|penalty| {
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
            bounds::certificate(&self.0.borrow(), master, duals).unwrap_or(f64::NEG_INFINITY),
        ))
    }
    fn pricing_bound(&self, master: &ClpMaster, duals: &[f64]) -> Result<Option<f64>> {
        let state = self.0.borrow();
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
        separation::repair_theta_duals(&self.0.borrow(), duals);
    }
    fn pricing_subgradient(&self, _rhs: &[f64]) -> Option<Vec<f64>> {
        self.0.borrow().gradient.clone()
    }
    fn take_pricing_subgradient(&mut self, _rhs: &[f64]) -> Option<Vec<f64>> {
        self.0.borrow_mut().gradient.take()
    }
    fn uses_inactive_column_pool(&self) -> bool {
        self.2
    }
    fn reprice_inactive_column(
        &mut self,
        column: &cg::Column,
        duals: &[f64],
    ) -> Result<Option<cg::Column>> {
        let state = self.0.borrow();
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
        let entries = entries(oracle, &route.route, state.shared_offset)?;
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

impl<const D: usize> Coordinator<D> {
    pub(super) fn recover_pricing_timeout(
        &mut self,
        error: crate::core::error::OrError,
    ) -> Result<(ClpMaster, cg::Output)> {
        if error.kind != crate::core::error::ErrorKind::Runtime
            || !error.message.contains("time limit")
            || !self
                .0
                .borrow()
                .oracles
                .iter()
                .any(|o| o.solver.requires_rollback)
        {
            return Err(error);
        }
        let Some(saved) = self.9.rollback.as_ref() else {
            return Err(error);
        };
        let restored_master = saved.master.try_clone()?;
        let mut restored_state = saved.state.snapshot_for_checkpoint();
        {
            let live = self.0.borrow();
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
        *self.0.borrow_mut() = restored_state;
        Ok((restored_master, saved.output.clone()))
    }
}

impl<const D: usize> State<D> {
    pub(super) fn price(&mut self, phase: usize, duals: &[f64]) -> Result<cg::Answer> {
        for &(network, resource, bounds) in &self.preprocessing.paired_owner_bounds {
            if self
                .oracles
                .iter()
                .find(|oracle| oracle.network.id == network)
                .is_some_and(|oracle| oracle.multiplicity_upper() == 0.)
            {
                continue;
            }
            if crate::graph::coupled::CoupledKernel::owner_registration_bound_is_unsupported([
                bounds.lower,
                bounds.upper,
            ]) {
                return Err(invalid(format!("non-zero lower bounds on variables associated to arcs are not supported (network {network}, paired resource {resource}, propagated owner bounds [{}, {}])", bounds.lower, bounds.upper)));
            }
        }
        // Explicit registration bounds remain relevant when propagation was
        // disabled or stopped before it could publish an owner coordinate.
        for oracle in &self.oracles {
            if oracle.multiplicity_upper() == 0. {
                continue;
            }
            if let Some(kernel) = oracle.network.graph.coupled_kernel() {
                for &(resource, bounds) in kernel.paired_owner_registration_bounds() {
                    if self
                        .preprocessing
                        .paired_owner_bounds
                        .iter()
                        .any(|&(network, index, _)| {
                            network == oracle.network.id && index == resource
                        })
                    {
                        continue;
                    }
                    if crate::graph::coupled::CoupledKernel::owner_registration_bound_is_unsupported(
                        bounds,
                    ) {
                        return Err(invalid(format!("non-zero lower bounds on variables associated to arcs are not supported (network {}, paired resource {resource}, input owner bounds {:?})", oracle.network.id, bounds)));
                    }
                }
            }
        }
        self.correction = None;
        self.gradient = None;
        self.scheduler.complete = false;
        let rhs = self.rhs.clone();
        let context = crate::master::pricing::Context {
            phase,
            duals,
            rhs: &rhs,
            deadline: Some(self.deadline),
            admission: self.box_pricing_admission,
            master: self.pricing_master,
            gap: self.pricing_gap,
        };
        let tolerance = self.scheduler.tolerance;
        let fresh = self.scheduler.fresh();
        let mut scheduler = std::mem::replace(&mut self.scheduler, fresh);
        let result = crate::master::pricing::price(
            &mut NetworkAdapter {
                state: self,
                tolerance,
            },
            &mut scheduler,
            context,
        );
        self.scheduler = scheduler;
        let mut batch = result?;
        if self.has_unpriced_paired_terms(duals) {
            // The raw operator completed, but its unit-charge minimum is not
            // a lower bound for the associated owner expression. Keep the raw
            // completion signal (also used to request explicit enumeration)
            // without publishing a numerical or certified owner bound.
            batch.correction = None;
            batch.gradient = None;
        }
        self.correction = batch.correction;
        self.gradient = batch.gradient;
        Ok(batch.answer)
    }
}

pub(super) struct NetworkAdapter<'a, const D: usize> {
    pub(super) state: &'a mut State<D>,
    pub(super) tolerance: f64,
}
impl<const D: usize> crate::master::pricing::Networks<D> for NetworkAdapter<'_, D> {
    fn active(&self) -> Vec<usize> {
        self.state
            .oracles
            .iter()
            .enumerate()
            .filter_map(|(p, o)| (o.multiplicity_upper() != 0.).then_some(p))
            .collect()
    }
    fn prepare(
        &mut self,
        index: usize,
        phase: usize,
        duals: &[f64],
        admission: bool,
    ) -> Result<crate::master::pricing::Request> {
        use crate::master::pricing::{Envelopes, Request};
        let oracle = &mut self.state.oracles[index];
        if self.state.universe.is_some()
            || self.state.gap_pool_cutoff.is_some()
            || oracle.solver.enumeration.is_some()
        {
            return Ok(Request {
                phase,
                exact: true,
                offset: 0.,
                lower: oracle.network.multiplicity.lower,
                upper: oracle.multiplicity_upper(),
                envelopes: Envelopes::default(),
            });
        }
        oracle.costs.clear();
        oracle.costs.extend(
            oracle
                .network
                .graph
                .arcs
                .iter()
                .zip(oracle.network.arc_entries.iter())
                .map(|(arc, coefficients)| {
                    coefficients.iter().fold(arc.cost, |cost, &(row, value)| {
                        cost - duals[self.state.shared_offset + row] * value
                    })
                }),
        );
        if !oracle.activity_rows.is_empty() {
            super::activity_rows::price(oracle, duals, self.state.shared_offset)?;
        }
        for &(cut, row) in &oracle.nonlinear_rows {
            oracle.solver.cuts.cuts[cut].dual = duals[self.state.shared_offset + row];
        }
        if !oracle.nonlinear_rows.is_empty() {
            // Modern rank-one retrieval filters and orders raw prices before
            // rounding. Keep k-path rows in registry order, as the routing
            // adapter does. Exact pricing uses separate raw cut inputs.
            let registry = &mut oracle.solver.cuts;
            let mut order: Vec<usize> = (0..registry.cuts.len())
                .filter(|&index| {
                    !registry.retired.contains(&index)
                        && !registry.cuts[index].visit_once
                        && registry.cuts[index].dual.abs() > 1e-6
                })
                .collect();
            crate::cuts::shared::stdsort::stable_sort(&mut order, |&a, &b| {
                let (va, vb) = (-registry.cuts[a].dual, -registry.cuts[b].dual);
                va > vb + 1e-6 || (!(vb - 1e-6 > va) && a < b)
            });
            registry.pricing_order = order;
            registry.round_duals = true;
        }
        for &(cut, row) in &oracle.knapsack_rows {
            oracle.solver.cuts.knapsack_cuts[cut].dual = duals[self.state.shared_offset + row];
        }
        if !oracle.presence_rows.is_empty() {
            let mut inputs = oracle.solver.presence_rows().to_vec();
            for &(input, row) in &oracle.presence_rows {
                inputs[input].dual = duals[self.state.shared_offset + row];
            }
            oracle.solver.set_presence_rows(inputs)?;
        }
        let cached = self
            .state
            .original_variable_costs
            .iter()
            .find(|model| model.network == oracle.network.id)
            .and_then(|model| {
                model.cached(
                    &oracle.network,
                    &oracle.index,
                    duals,
                    self.state.shared_offset,
                )
            });
        if let Some(cached) = cached {
            oracle.solver.cache_physical_costs(&oracle.costs, &cached)?;
        } else {
            oracle.solver.cache_variable_costs(&oracle.costs)?;
        }
        let offset = oracle
            .route_entries
            .iter()
            .map(|&(row, value)| -duals[row] * value)
            .sum::<f64>()
            - oracle
                .network
                .route_entries
                .iter()
                .map(|&(row, value)| duals[self.state.shared_offset + row] * value)
                .sum::<f64>();

        let original_box = self.state.grouped_variable_boxes.get(&oracle.network.id);
        let envelopes = if admission {
            let arc_box = self.state.variable_boxes.get(&oracle.network.id);
            let finite = if let Some(groups) = original_box {
                grouped_variable_box(
                    oracle,
                    offset,
                    groups,
                    arc_box.map(Vec::as_slice),
                    duals,
                    self.state.shared_offset,
                )
            } else {
                arc_box.map_or_else(
                    || required_nonnegative_box(oracle, offset),
                    |bounds| finite_variable_box(oracle, offset, bounds),
                )
            };
            Envelopes {
                nonnegative: nonnegative_box(oracle, offset),
                finite,
                near_zero: near_zero_box(oracle, offset, self.tolerance).map(
                    |(minimum, correction)| BoxAdmission {
                        minimum,
                        correction,
                        zero_quantity: minimum >= 0.,
                    },
                ),
                original: original_box.is_some(),
            }
        } else {
            Envelopes::default()
        };
        // Pricing omits the route-constant dual contribution from arc costs.
        // Its zero threshold therefore has the opposite sign to that offset.
        oracle.solver.controls.zero_threshold = -offset;
        Ok(Request {
            phase,
            exact: oracle.solver.phases[phase].exact(),
            offset,
            lower: oracle.network.multiplicity.lower,
            upper: oracle.multiplicity_upper(),
            envelopes,
        })
    }
    fn enter(&mut self, index: usize) -> &mut Solver<D> {
        let oracle = &mut self.state.oracles[index];
        oracle.calls += 1;
        &mut oracle.solver
    }
    fn retained(
        &mut self,
        index: usize,
        duals: &[f64],
    ) -> Option<Result<crate::master::pricing::Retained>> {
        let gap_universe;
        let universe = if let Some(universe) = self.state.universe.as_ref() {
            universe
        } else if self.state.gap_pool_cutoff.is_some()
            || self.state.oracles[index].solver.enumeration.is_some()
        {
            let oracle = &self.state.oracles[index];
            gap_universe = self
                .state
                .routes
                .iter()
                .enumerate()
                .filter(|(_, route)| {
                    route.network == oracle.network.id && oracle.is_enumerated(&route.route)
                })
                .map(|(tag, _)| tag)
                .collect::<Vec<_>>();
            &gap_universe
        } else {
            return None;
        };
        Some((|| {
            let oracle = &mut self.state.oracles[index];
            oracle.calls += 1;
            let mut columns = vec![];
            for &tag in universe {
                let route = &self.state.routes[tag];
                if route.network != oracle.network.id
                    || route.route.arcs.iter().any(|id| {
                        oracle
                            .proved_zero_arcs
                            .get(oracle.index[id])
                            .copied()
                            .unwrap_or(false)
                    })
                {
                    continue;
                }
                let cap = self
                    .state
                    .route_upper_bounds
                    .get(&tag)
                    .copied()
                    .unwrap_or(oracle.multiplicity_upper())
                    .min(oracle.multiplicity_upper());
                let coefficients = entries(oracle, &route.route, self.state.shared_offset)?;
                let reduced = coefficients
                    .iter()
                    .fold(route.route.original_cost, |v, &(r, a)| v - duals[r] * a);
                columns.push((
                    cg::Column {
                        cost: route.route.original_cost,
                        reduced_cost: reduced,
                        entries: coefficients,
                        acceptable: true,
                        tag: tag as i32,
                        identity: format!("{}:{}", route.network, route.route.identity()),
                    },
                    cap,
                ));
            }
            Ok(crate::master::pricing::Retained {
                minimum_bound: None,
                columns,
                independent_caps: true,
                maximum: usize::MAX,
            })
        })())
    }
    fn collect(
        &mut self,
        index: usize,
        routes: Vec<Route>,
        offset: f64,
        duals: &[f64],
    ) -> Result<crate::master::pricing::Collected> {
        let oracle = &mut self.state.oracles[index];
        let mut columns = vec![];
        let mut new_improving_columns = 0;
        let mut best: Option<(f64, Vec<(usize, f64)>, Route)> = None;
        for mut route in routes {
            let coefficients = entries(oracle, &route, self.state.shared_offset)?;
            let cost = original_costs::route_cost(oracle, &route)?;
            let reduced_cost = coefficients
                .iter()
                .fold(cost, |value, &(row, coefficient)| {
                    value - duals[row] * coefficient
                });
            if !cost.is_finite() || !reduced_cost.is_finite() {
                return Err(invalid("network route cost overflow"));
            }
            if best
                .as_ref()
                .map_or(true, |(value, _, _)| reduced_cost < *value)
            {
                best = Some((reduced_cost, coefficients.clone(), route.clone()));
            }
            route.original_cost = cost;
            route.cost += offset;
            let identity = route.identity();
            let key = (oracle.network.id, identity.clone());
            let tag = if let Some(&tag) = self.state.tags.get(&key) {
                tag
            } else {
                new_improving_columns += usize::from(reduced_cost < -self.tolerance);
                let tag = i32::try_from(self.state.routes.len())
                    .map_err(|_| invalid("network route tag overflow"))?;
                self.state.tags.insert(key, tag);
                self.state.routes.push(Arc::new(NetworkRoute {
                    network: oracle.network.id,
                    route,
                }));
                tag
            };
            columns.push(cg::Column {
                cost,
                reduced_cost,
                entries: coefficients,
                acceptable: true,
                tag,
                identity: format!("{}:{identity}", oracle.network.id),
            });
        }

        Ok(crate::master::pricing::Collected {
            columns,
            fresh: new_improving_columns,
            best_route: best.as_ref().map(|(_, _, route)| route.clone()),
            best_entries: best.map(|(_, entries, _)| entries),
        })
    }
    fn static_contribution(&self, duals: &[f64]) -> Option<(f64, Vec<(usize, f64)>)> {
        master_variables::correction(self.state, duals)
    }
}

#[cfg(test)]
#[path = "../../../tests/unit/master/model/native_contribution_tests.rs"]
mod native_contribution_tests;

/// Nonnegative orthant case of the native variable-bound pricing test.
/// No unit arc cap is assumed: non-elementary walks can repeat an arc.
fn additive<const D: usize>(oracle: &Oracle<D>) -> crate::master::pricing::Additive<'_, D> {
    crate::master::pricing::Additive {
        graph: &oracle.network.graph,
        costs: &oracle.costs,
        lower: oracle.network.multiplicity.lower,
        upper: oracle.multiplicity_upper(),
        supported: oracle.nonlinear_rows.is_empty()
            && oracle.knapsack_rows.is_empty()
            && oracle.presence_rows.is_empty()
            && oracle.resource_groups.is_empty()
            && oracle.ryan_foster_rows.is_empty()
            && oracle.activity_rows.is_empty(),
    }
}
fn nonnegative_box<const D: usize>(oracle: &Oracle<D>, offset: f64) -> bool {
    additive(oracle).nonnegative(offset)
}
type BoxAdmission = crate::master::pricing::Estimate;

fn finite_variable_box<const D: usize>(
    oracle: &Oracle<D>,
    offset: f64,
    bounds: &[crate::master::preprocessing::Bounds],
) -> Option<BoxAdmission> {
    if !offset.is_finite()
        || bounds.len() != oracle.costs.len()
        || oracle.costs.iter().any(|cost| !cost.is_finite())
        || !oracle.nonlinear_rows.is_empty()
        || !oracle.knapsack_rows.is_empty()
        || !oracle.presence_rows.is_empty()
        || !oracle.resource_groups.is_empty()
        || !oracle.ryan_foster_rows.is_empty()
        || !oracle.activity_rows.is_empty()
        || oracle.network.graph.has_coupled_resources()
    {
        return None;
    }
    let mut negative = crate::master::numerics::ExactSum::default();
    for (arc, (&cost, bounds)) in oracle.costs.iter().zip(bounds).enumerate() {
        if cost == 0. || oracle.solver.forbidden_arcs.get(arc) == Some(&true) {
            continue;
        }
        let selected = if cost < 0. {
            bounds.upper
        } else {
            bounds.lower
        };
        if !selected.is_finite() {
            return None;
        }
        negative.product_upper(-cost, selected)?;
    }
    negative.add(-offset)?;
    // Preserve negative remainders even when naive cancellation rounds them away.
    let lower = -negative.upper()?;
    box_correction(oracle, lower)
}

fn grouped_variable_box<const D: usize>(
    oracle: &Oracle<D>,
    offset: f64,
    groups: &grouped_boxes::PreparedBox,
    arc_bounds: Option<&[crate::master::preprocessing::Bounds]>,
    duals: &[f64],
    shared_offset: usize,
) -> Option<BoxAdmission> {
    if oracle.network.graph.has_coupled_resources() {
        return grouped_boxes::coupled_admission(oracle, groups, duals, shared_offset);
    }
    if !offset.is_finite()
        || oracle.costs.len() != groups.caps.len()
        || groups
            .variables
            .iter()
            .any(|v| v.accumulated_resource.is_some())
        || oracle.costs.iter().any(|c| !c.is_finite())
        || !oracle.nonlinear_rows.is_empty()
        || !oracle.knapsack_rows.is_empty()
        || !oracle.presence_rows.is_empty()
        || !oracle.resource_groups.is_empty()
        || !oracle.ryan_foster_rows.is_empty()
        || !oracle.activity_rows.is_empty()
        || oracle.network.graph.has_coupled_resources()
    {
        return None;
    }
    let mut residual: Vec<_> = oracle
        .costs
        .iter()
        .map(|&cost| {
            let mut value = crate::master::numerics::ExactSum::default();
            value.add(-cost)?;
            Some(value)
        })
        .collect::<Option<_>>()?;
    let mut negative = crate::master::numerics::ExactSum::default();
    for variable in &groups.variables {
        let mut reduced = variable.cost;
        for &(row, a) in &variable.entries {
            reduced -= *duals.get(shared_offset.checked_add(row)?)? * a;
        }
        if !reduced.is_finite() {
            return None;
        }
        if reduced != 0. {
            let endpoint = if reduced < 0. {
                variable.bounds.upper
            } else {
                variable.bounds.lower
            };
            if !endpoint.is_finite() {
                return None;
            }
            negative.product_upper(-reduced, endpoint)?;
        }
        for &(arc, a) in &variable.arcs {
            residual[arc].product_upper(reduced, a)?;
        }
    }
    for (arc, value) in residual.iter().enumerate() {
        if oracle.solver.forbidden_arcs.get(arc) == Some(&true) {
            continue;
        }
        let cost = -value.upper()?;
        if cost == 0. {
            continue;
        }
        let lower = arc_bounds.map_or(0., |b| b[arc].lower.max(0.));
        let upper = arc_bounds.map_or(groups.caps[arc], |b| b[arc].upper.min(groups.caps[arc]));
        let endpoint = if cost < 0. { upper } else { lower };
        if !endpoint.is_finite() {
            return None;
        }
        negative.product_upper(-cost, endpoint)?;
    }
    negative.add(-offset)?;
    box_correction(oracle, -negative.upper()?)
}

/// Native's implicit [0,infinity] variable box gives a lower estimate,
/// including the route offset and required quantity. A first feasible route cannot be replaced
/// by zero selection; every such estimate goes through the master-gap gate.
fn required_nonnegative_box<const D: usize>(
    oracle: &Oracle<D>,
    offset: f64,
) -> Option<BoxAdmission> {
    additive(oracle).required(offset)
}
fn box_correction<const D: usize>(oracle: &Oracle<D>, lower: f64) -> Option<BoxAdmission> {
    BoxAdmission::contribution(
        lower,
        oracle.network.multiplicity.lower,
        oracle.multiplicity_upper(),
    )
}

#[cfg(test)]
#[path = "../../../tests/unit/master/model/box_pricing_tests.rs"]
mod box_pricing_tests;

#[cfg(test)]
#[path = "../../../tests/unit/master/model/early_witness_tests.rs"]
mod early_witness_tests;

/// Native cost sign precision is 1e-6. Unlike snapping those costs to zero,
/// retain a lower estimate covering every walk in the actual pricing domain.
fn near_zero_box<const D: usize>(
    oracle: &Oracle<D>,
    offset: f64,
    tolerance: f64,
) -> Option<(f64, f64)> {
    additive(oracle)
        .near_zero(offset, tolerance)
        .map(|bound| (bound.minimum, bound.correction))
}

#[cfg(test)]
#[path = "../../../tests/unit/master/model/grouped_box_tests.rs"]
mod grouped_box_tests;

#[cfg(test)]
#[path = "../../../tests/unit/master/model/coupled_admission_tests.rs"]
mod coupled_admission_tests;

/// Prepare the shared oracle at the exact dual point used by fixing.
pub(super) fn cache_duals<const D: usize>(
    state: &mut State<D>,
    index: usize,
    duals: &[f64],
) -> Result<f64> {
    use crate::master::pricing::Networks;
    let tolerance = state.scheduler.tolerance;
    // Retained-pool pricing normally skips labeling preparation. Fixing and
    // inspection must instead refresh costs at these raw duals, even when a
    // preceding gap pass left a complete retained pool.
    let retained = state.oracles[index].solver.enumeration.take();
    let coverage = state.gap_pool_cutoff.take();
    let prepared = NetworkAdapter { state, tolerance }.prepare(index, 0, duals, false);
    state.oracles[index].solver.enumeration = retained;
    state.gap_pool_cutoff = coverage;
    let request = prepared?;
    let oracle = &mut state.oracles[index];
    oracle
        .solver
        .cache_physical_costs(&oracle.costs, &oracle.costs)?;
    Ok(request.offset)
}
