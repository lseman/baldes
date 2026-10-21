//! Presence pricing setup and conservative completion-key adjustment.
use super::*;
use crate::cuts::soft_conflicts::{LabelState, PresencePricing, PricingInput};

impl<const D: usize> LabelingEngine<D> {
    /// Install physical-arc presence rows and invalidate all dual-dependent
    /// labels. Invalid input leaves the previous preparation unchanged.
    /// Symmetric prefix reuse falls back to ordinary bidirectional execution,
    /// which preserves the actual orientation of each original-variable event.
    pub fn load_presence(&mut self, g: &Graph<D>, rows: Vec<PricingInput>) -> Result<()> {
        let prepared = PresencePricing::new(g.arcs.len(), rows)?;
        prepared.validate_engine(g.arcs.len())?;
        self.cuts.presence = prepared;
        self.half_ready = false;
        self.full_ready = false;
        Ok(())
    }
}

impl<const D: usize> Context<'_, D> {
    pub(super) fn presence_completion_key(&self, key: f64, state: &LabelState) -> f64 {
        if !self.cuts.presence.active() {
            return key;
        }
        self.cuts
            .presence
            .join_correction_lower_bound(state)
            .ok()
            .flatten()
            .and_then(|bound| crate::core::rounding::add_down(key, bound))
            .unwrap_or(f64::NEG_INFINITY)
    }

    pub(super) fn completion_sum(&self, a: f64, b: f64) -> f64 {
        if !self.cuts.presence.active() {
            return a + b;
        }
        crate::core::rounding::add_down(a, b).unwrap_or(f64::NEG_INFINITY)
    }
}
