//! Solver-owned presence rows and cache invalidation.
use super::*;
use crate::cuts::soft_conflicts::{PresencePricing, PricingInput};

impl<const D: usize> Solver<D> {
    /// Replace the original-variable presence rows used by pricing. Validation
    /// is atomic. Identical inputs retain working state; changed inputs discard
    /// cost-dependent bucket fixing, labels and retained enumeration. A dual-only
    /// update can retain the private raw workspace's objective-independent grid.
    /// Logical graph, NG, branch constraints and physical exclusions remain in force.
    pub fn set_presence_rows(&mut self, rows: Vec<PricingInput>) -> Result<()> {
        let prepared = PresencePricing::new(self.graph.arcs.len(), rows)?;
        prepared.validate_engine(self.graph.arcs.len())?;
        if self.presence_pricing.inputs() == prepared.inputs() {
            return Ok(());
        }
        let same_structure = self.presence_pricing.inputs().len() == prepared.inputs().len()
            && self
                .presence_pricing
                .inputs()
                .iter()
                .zip(prepared.inputs())
                .all(|(old, new)| {
                    let PricingInput {
                        row,
                        arc_events,
                        dual: _,
                    } = old;
                    row == &new.row && arc_events == &new.arc_events
                });
        let mut raw = same_structure.then(|| self.raw_pricing.take()).flatten();
        if let Some(cache) = &mut raw {
            if cache.solver.graph.same_topology(&self.graph)
                && cache.solver.presence_pricing.inputs() == self.presence_pricing.inputs()
            {
                // This private solver never performs objective-dependent arc
                // fixing or enumeration. Refresh its row prices and invalidate
                // every label/completion bound before retaining preparation.
                if let Some(engine) = &mut cache.solver.engine {
                    engine.load_presence(&cache.solver.graph, prepared.inputs().to_vec())?;
                    engine.release_raw_labels();
                }
                cache.solver.presence_pricing = prepared.clone();
            } else {
                raw = None;
            }
        }
        self.presence_pricing = prepared;
        self.restore_arcs();
        self.raw_pricing = raw;
        self.round = controller::RoundState::default();
        Ok(())
    }

    #[cfg(feature = "clp")]
    pub(crate) fn presence_route_coefficients(&self, arcs: &[usize]) -> Result<Vec<usize>> {
        self.presence_pricing.route_coefficients(arcs)
    }

    /// Read the owned descriptors. Update through `set_presence_rows` so dual
    /// and event edits cannot leave labels or a route universe at stale costs.
    pub fn presence_rows(&self) -> &[PricingInput] {
        self.presence_pricing.inputs()
    }
}
