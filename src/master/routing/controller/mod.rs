//! Column-generation callback adapter; implementations are grouped by responsibility.
use super::*;
mod capacity;
mod checkpoint;
mod enumeration;
mod feasibility;
mod knapsack;
mod kpath;
mod pool;
mod progress;
mod rank_one;
mod relaxation;
mod separation;

pub(super) use checkpoint::PricingCheckpoint;
#[cfg(test)]
use pool::synchronize_pool;
use separation::separation_gap_closed;
#[cfg(test)]
#[path = "../../../../tests/unit/master/routing/controller/tests.rs"]
mod synchronization_tests;

pub(super) struct RootController<const D: usize = { crate::DEFAULT_RESOURCE_DIMENSIONS }> {
    pub(super) problem: Rc<RoutingContext<D>>,
    pub(super) state: Rc<RefCell<State<D>>>,
    pub(super) checkpoint: Option<PricingCheckpoint<D>>,
    pub(super) enumeration_checkpoint: Option<enumeration::EnumerationCheckpoint<D>>,
    pub(super) certificate_search: super::certificate::SearchCache,
}

impl<const D: usize> Controller<ClpMaster> for RootController<D> {
    fn recover_feasibility(
        &mut self,
        master: &ClpMaster,
        warm: &cg::WarmStart,
        remaining: std::time::Duration,
        iterations: usize,
        tolerance: f64,
        bound_mode: BoundMode,
    ) -> Result<Option<cg::FeasibilityRecovery>> {
        self.recover_phase_one(master, warm, remaining, iterations, tolerance, bound_mode)
    }
    fn begin_enumeration_transaction(&mut self, bound: f64) -> Result<bool> {
        self.begin_enumeration(bound)
    }

    fn finish_enumeration_transaction(&mut self, commit: bool) {
        self.finish_enumeration(commit);
    }
    fn restore_feasible_columns(&mut self, duals: &[f64]) -> Result<Vec<Column>> {
        super::restoration::columns(&self.problem, &mut self.state.borrow_mut(), duals)
    }

    fn certified_pricing_bound(&self, master: &ClpMaster, _duals: &[f64]) -> Result<Option<f64>> {
        let bound = super::certificate::lower_bound(
            &self.problem,
            &self.state.borrow(),
            master,
            &self.certificate_search,
        );
        Ok(Some(bound.unwrap_or(f64::NEG_INFINITY)))
    }

    fn save_pricing_checkpoint(&mut self, master: &ClpMaster, output: &Output) -> Result<()> {
        self.store_pricing_checkpoint(master, output.clone())
    }

    fn needs_pricing_checkpoint(&self, master: &ClpMaster) -> bool {
        self.checkpoint.is_none()
            || !separation_gap_closed(master.objective(), self.state.borrow().incumbent)
    }

    fn store_pricing_checkpoint(&mut self, master: &ClpMaster, output: Output) -> Result<()> {
        self.store_checkpoint(master, output)
    }

    fn curvature_rows(&self, row_count: usize) -> Vec<cg::CurvatureRow> {
        let ordinary = self.problem.row_map.borrow();
        let excluded = self.unstabilized_rows();
        (0..row_count)
            .map(|row| cg::CurvatureRow {
                class: if excluded.contains(&row) {
                    cg::CurvatureRowClass::Excluded
                } else if ordinary.contains(&row) {
                    cg::CurvatureRowClass::Ordinary
                } else {
                    cg::CurvatureRowClass::Dynamic
                },
                active: true,
            })
            .collect()
    }

    fn unstabilized_rows(&self) -> Vec<usize> {
        let map = self.problem.row_map.borrow();
        std::iter::once(map[0])
            .chain(
                self.problem
                    .branch_constraints
                    .iter()
                    .filter_map(|(row, constraint)| {
                        matches!(constraint, BranchConstraint::NumberOfVehicles { .. })
                            .then_some(map[*row])
                    }),
            )
            .collect()
    }

    fn protected_rows(&self) -> Vec<usize> {
        self.problem.row_map.borrow().clone()
    }

    fn master_solved(
        &mut self,
        master: &ClpMaster,
        tags: &[i32],
        artificials: &[usize],
    ) -> Result<()> {
        {
            let mut state = self.state.borrow_mut();
            state.pricing.rhs = master.rhs.clone();
            state.pricing.master = (master.solved()
                && master.objective().is_finite()
                && master.objective_offset.is_finite())
            .then_some((master.objective(), master.objective_offset));
        }
        if artificials.iter().any(|&i| master.values()[i] > 1e-6) {
            return Ok(());
        }
        let mut state = self.state.borrow_mut();
        if let Some(upper) =
            super::incumbent::validated_upper(&self.problem, &state.routes, tags, &master.values())
        {
            state.incumbent = state.incumbent.min(upper);
        }
        Ok(())
    }

    fn pricing_complete(&self) -> bool {
        self.state.borrow().pricing.complete()
    }
    fn pricing_bound(&self, master: &ClpMaster, duals: &[f64]) -> Result<Option<f64>> {
        let state = self.state.borrow();
        Ok(Some(state.pricing.correction.map_or(
            f64::NEG_INFINITY,
            |correction| {
                duals
                    .iter()
                    .zip(&master.rhs)
                    .fold(master.objective_offset, |v, (d, r)| v + d * r)
                    + correction
            },
        )))
    }
    fn pricing_subgradient(&self, _rhs: &[f64]) -> Option<Vec<f64>> {
        self.state.borrow().pricing.gradient.clone()
    }
    fn take_pricing_subgradient(&mut self, _rhs: &[f64]) -> Option<Vec<f64>> {
        self.state.borrow_mut().pricing.gradient.take()
    }

    fn incumbent(&self) -> f64 {
        self.state.borrow().incumbent
    }

    fn cut_counts(&self) -> Vec<(&'static str, usize)> {
        let state = self.state.borrow();
        state.cut.progress_counts(&state.solver.cuts.knapsack_cuts)
    }

    fn bucket_steps(&self) -> Vec<f64> {
        let state = self.state.borrow();
        state
            .solver
            .pricing_engine()
            .map_or_else(Vec::new, |engine| {
                engine.grid.step[..engine.grid.dimensions].to_vec()
            })
    }

    fn refresh_generated_columns(&mut self, pool: &mut [Column]) -> Result<()> {
        self.refresh_pool(pool)
    }

    fn refresh_master_columns(
        &mut self,
        master: &mut ClpMaster,
        loaded: &mut [Option<Column>],
    ) -> Result<bool> {
        self.restore_limited_memory_columns(master, loaded)
    }

    fn uses_inactive_column_pool(&self) -> bool {
        let state = self.state.borrow();
        // A finite pricing universe and its full-memory cut contract own
        // their own reuse rules. Likewise, a graph narrowed by resource
        // branching needs current-domain replay, not original-graph replay.
        state.solver.enumeration.is_none()
            && state.cut.enumerated.is_empty()
            && state
                .solver
                .graph
                .same_topology(&self.problem.model.networks[0].graph)
    }

    fn reprice_inactive_column(
        &mut self,
        column: &Column,
        duals: &[f64],
    ) -> Result<Option<Column>> {
        let state = self.state.borrow();
        let Some(route) = usize::try_from(column.tag)
            .ok()
            .and_then(|tag| state.routes.get(tag))
        else {
            return Ok(None);
        };
        // Only reuse an independently replayed original path. Relaxed-only
        // paths remain available to normal pricing, so this optional filter
        // cannot strengthen the domain or certify exhaustion.
        if super::incumbent::exact_route_feasible(&self.problem, route).is_none()
            || !state
                .solver
                .route_satisfies_relaxation(&route.nodes, &route.arcs)
                .unwrap_or(false)
            || route.arcs.iter().any(|id| {
                state
                    .solver
                    .graph
                    .arcs
                    .iter()
                    .position(|a| a.id == *id)
                    .map_or(true, |arc| {
                        state
                            .solver
                            .forbidden_arcs
                            .get(arc)
                            .copied()
                            .unwrap_or(false)
                    })
            })
        {
            return Ok(None);
        }
        Ok(Some(state.column(
            &self.problem,
            route,
            duals,
            column.tag,
        )?))
    }

    fn synchronize_generated_columns(
        &mut self,
        pool: &mut [Column],
        loaded: &[Option<Column>],
    ) -> Result<()> {
        self.synchronize_pool_entries(pool, loaded)
    }

    fn enumerated_universe(&mut self, duals: &[f64], bound: f64) -> Result<Option<Vec<Column>>> {
        self.enumerate_columns(duals, bound)
    }

    fn regenerate_enumerated_columns(
        &mut self,
        pool: &[Column],
        duals: &[f64],
        bound: f64,
        universe: &[Column],
    ) -> Result<Vec<Column>> {
        self.regenerate_pool(pool, duals, bound, universe)
    }

    fn remap_rows(&mut self, rows: &[Option<usize>]) -> Result<()> {
        self.remap_pool_rows(rows)
    }

    fn separates(&self) -> bool {
        !self.state.borrow().cut.separators().is_empty()
            || self.problem.model.networks[0].graph.selection_relaxed()
    }

    fn separate(&mut self, master: &mut ClpMaster, tags: &[i32]) -> Result<usize> {
        self.separate_round(master, tags)
    }

    fn take_removed_columns(&mut self) -> Vec<usize> {
        std::mem::take(&mut self.state.borrow_mut().cut.removed_columns)
    }
}
