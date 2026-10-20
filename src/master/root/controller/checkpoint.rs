//! Completed pricing checkpoints and timeout recovery.
use super::*;

pub(in crate::master::root) struct PricingCheckpoint<
    const D: usize = { crate::DEFAULT_RESOURCE_DIMENSIONS },
> {
    pub(super) master: ClpMaster,
    pub(super) row_map: Vec<usize>,
    pub(super) cut: CutState,
    pub(super) pricing: Solver<D>,
    pub(super) incumbent: f64,
    pub(super) route_count: usize,
    pub(super) output: Output,
    pub(super) previous_fixing_gap: f64,
    pub(super) enumerated_routes: usize,
}

impl<const D: usize> RootController<D> {
    pub(super) fn store_checkpoint(&mut self, master: &ClpMaster, output: Output) -> Result<()> {
        let mut state = self.state.borrow_mut();
        // Separation returns without changing the formulation at a closed
        // incumbent gap. Keep the previous rollback point; no new pricing
        // transaction can start from this terminal point.
        if self.checkpoint.is_some() && separation_gap_closed(master.objective(), state.incumbent) {
            return Ok(());
        }
        let mut pricing = state.solver.fork_for_node();
        pricing.rollback_point_saved = false;
        pricing.requires_rollback = false;
        self.checkpoint = Some(PricingCheckpoint {
            master: master.try_clone()?,
            row_map: self.problem.row_map.borrow().clone(),
            cut: state.cut.clone(),
            pricing,
            incumbent: state.incumbent,
            // Routes are immutable after registration and only appended.
            // Rollback therefore needs a prefix length, not a deep copy of
            // every resource schedule in the route store.
            route_count: state.routes.len(),
            output,
            previous_fixing_gap: state.previous_fixing_gap,
            enumerated_routes: state.enumerated_routes,
        });
        state.solver.rollback_point_saved = true;
        Ok(())
    }

    /// Recover only an engine timeout that explicitly requests saved-point rollback.
    pub(in crate::master::root) fn recover_pricing_timeout(
        &mut self,
        error: crate::core::error::OrError,
    ) -> Result<(ClpMaster, Output)> {
        if error.kind != crate::core::error::ErrorKind::Runtime
            || !error.message.contains("time limit")
            || !self.state.borrow().solver.requires_rollback
        {
            return Err(error);
        }
        let Some(checkpoint) = &self.checkpoint else {
            return Err(error);
        };
        if self.state.borrow().routes.len() < checkpoint.route_count {
            return Err(error);
        }
        // Clone before replacing any state so a failed allocation cannot partially restore it.
        let master = checkpoint.master.try_clone()?;
        let pricing = checkpoint.pricing.fork_for_node();
        let mut state = self.state.borrow_mut();
        state.incumbent = state.incumbent.min(checkpoint.incumbent);
        state.cut = checkpoint.cut.clone();
        state.solver = pricing;
        state.routes.truncate(checkpoint.route_count);
        state.previous_fixing_gap = checkpoint.previous_fixing_gap;
        state.enumerated_routes = checkpoint.enumerated_routes;
        state
            .route_tags
            .retain(|_, tag| *tag >= 0 && (*tag as usize) < checkpoint.route_count);
        *self.problem.row_map.borrow_mut() = checkpoint.row_map.clone();
        Ok((master, checkpoint.output.clone()))
    }
}
