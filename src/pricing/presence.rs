//! Solver-owned presence rows and cache invalidation.
use super::*;
use crate::cuts::soft_conflicts::{PresencePricing, PricingInput};

impl<const D: usize> Solver<D> {
    /// Replace the original-variable presence rows used by pricing. Validation
    /// is atomic. Identical inputs retain working state; changed inputs discard
    /// cost-dependent bucket fixing, labels and retained enumeration. Logical
    /// graph, NG, branch constraints and physical exclusions remain in force.
    pub fn set_presence_rows(&mut self, rows: Vec<PricingInput>) -> Result<()> {
        let prepared = PresencePricing::new(self.graph.arcs.len(), rows)?;
        prepared.validate_engine(self.graph.arcs.len())?;
        if self.presence_pricing.inputs() == prepared.inputs() {
            return Ok(());
        }
        self.presence_pricing = prepared;
        self.restore_arcs();
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
