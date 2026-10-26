//! Round completion and invalidation after graph changes.
use super::*;

impl<const D: usize> Solver<D> {
    /// Geometry of the objective used by the most recent functor search.
    pub(crate) fn pricing_engine(&self) -> Option<&LabelingEngine<D>> {
        self.raw_pricing
            .as_ref()
            .and_then(|raw| raw.solver.engine.as_ref())
            .or(self.engine.as_ref())
    }
    /// columnGenerationTerminated as Alg4EvalByLagrangianDuality
    /// calls it after each converged column generation: the statistics form,
    /// then the end-of-round form that applies a pending bucket-step decrease.
    /// Returns the stop-cut-generation flag of the statistics form.
    pub fn column_generation_terminated(&mut self, flag3: i32, flag5: i32) -> Result<bool> {
        let enumerated = self.enumeration.is_some();
        // Borrow the geometry and round state independently.
        let grid = if self.raw_pricing.is_some() {
            self.raw_pricing
                .as_ref()
                .and_then(|raw| raw.solver.engine.as_ref())
                .map(|e| &e.grid)
        } else {
            self.engine.as_ref().map(|e| &e.grid)
        };
        let (stop, _) = self
            .round
            .action(&self.options, grid, enumerated, false, flag3, flag5)?;
        let (_, decrease) =
            self.round
                .action(&self.options, grid, enumerated, true, flag3, flag5)?;
        if decrease {
            let engine = if let Some(raw) = &mut self.raw_pricing {
                raw.solver.engine.as_mut()
            } else {
                self.engine.as_mut()
            };
            if let Some(engine) = engine {
                // "cannot decrease bucket steps" is reported and ignored.
                let _ = engine.grid.decrease_steps(&self.graph);
                for (axis, &slot) in engine.grid.resource_slots.iter().enumerate() {
                    self.bucket_steps[slot] = engine.grid.step[axis];
                }
            }
        }
        Ok(stop)
    }
}
impl<const D: usize> Solver<D> {
    /// Resource and topology edits invalidate both completion labels and paths.
    /// Reduced costs are deliberately excluded by `same_topology`.
    pub(super) fn invalidate_changed_graph(&mut self) {
        if self
            .prepared
            .as_ref()
            .into_iter()
            .chain(self.enumeration_snapshot.as_ref())
            .any(|snapshot| !snapshot.same_topology(&self.graph))
        {
            self.restore_arcs();
        }
    }
    pub(super) fn topology(&self) -> Arc<Graph<D>> {
        let mut graph = self.graph.clone();
        for a in &mut graph.arcs {
            a.reduced_cost = 0.;
        }
        Arc::new(graph)
    }
}
