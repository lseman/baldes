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
        self.coupled_completion = std::array::from_fn(|_| Default::default());
        self.half_ready = false;
        self.full_ready = false;
        Ok(())
    }
}

impl<const D: usize> Context<'_, D> {
    fn coupled_cache(&self, g: &Graph<D>, forward: bool) -> Option<&CoupledCompletion> {
        let cache = &self.coupled_completion[usize::from(forward)];
        let token = cache.identity.as_ref()?;
        std::sync::Arc::ptr_eq(token, g.coupled_kernel()?.cache_identity()).then_some(cache)
    }

    /// Cached hull-pair rejection is an additional necessary check. A matching
    /// query that is not rejected still uses the original point-query bound,
    /// so coarse pair hulls cannot weaken the existing pruning.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn bucket_completion(
        &self,
        g: &Graph<D>,
        forward: bool,
        label: LabelRef<'_, D>,
        key: f64,
        target: BucketId,
        first_key: Option<f64>,
        threshold: f64,
    ) -> Option<f64> {
        if let Some(kernel) = g.coupled_kernel() {
            let first_key = first_key?;
            if let Some(cache) = self.coupled_cache(g, forward) {
                let own = label.bucket();
                if own.vertex == target.vertex {
                    let hull = cache
                        .queries
                        .get(own.vertex)
                        .and_then(|r| r.get(own.bucket));
                    let pair = cache
                        .pairs
                        .get(own.vertex)
                        .and_then(|r| r.get(own.bucket))
                        .and_then(Option::as_ref)
                        .and_then(|r| r.get(target.bucket));
                    if let (Some(hull), Some(pair)) = (hull, pair) {
                        // Membership is needed only to authorize a rejection.
                        // Most coarse hull pairs survive; do not rescan their
                        // query vectors and nonlinear envelopes unnecessarily.
                        let rejects = !pair.possible
                            || (kernel.has_costs() && {
                                let cached_key = crate::core::rounding::add_down(
                                    self.presence_completion_key(key, label.presence_state()),
                                    pair.correction_lower,
                                )
                                .unwrap_or(f64::NEG_INFINITY);
                                self.completion_sum(cached_key, first_key) >= threshold
                            });
                        if rejects && hull.contains(label.coupled_state()) {
                            return None;
                        }
                    }
                }
            }
        }
        if !self.bucket_completion_feasible(
            g,
            forward,
            label.coupled_state(),
            target.vertex,
            target.bucket,
        ) {
            return None;
        }
        Some(self.resource_bucket_key(
            g,
            forward,
            label.coupled_state(),
            label.presence_state(),
            key,
            target.vertex,
            target.bucket,
        ))
    }

    pub(super) fn resource_completion_key(
        &self,
        g: &Graph<D>,
        forward: bool,
        coupled: Option<&crate::graph::coupled::CoupledLabelState>,
        presence: &LabelState,
        key: f64,
    ) -> f64 {
        let key = self.presence_completion_key(key, presence);
        let Some(kernel) = g.coupled_kernel().filter(|k| k.has_costs()) else {
            return key;
        };
        let Some(cache) = self.coupled_cache(g, forward) else {
            return f64::NEG_INFINITY;
        };
        let correction = kernel.completion_correction_lower(forward, coupled, &cache.global);
        crate::core::rounding::add_down(key, correction).unwrap_or(f64::NEG_INFINITY)
    }
    pub(super) fn bucket_completion_feasible(
        &self,
        g: &Graph<D>,
        forward: bool,
        coupled: Option<&crate::graph::coupled::CoupledLabelState>,
        vertex: usize,
        bucket: usize,
    ) -> bool {
        let Some(kernel) = g.coupled_kernel() else {
            return true;
        };
        let Some(cache) = self.coupled_cache(g, forward) else {
            return true;
        };
        let bounds = cache
            .buckets
            .get(vertex)
            .and_then(|row| row.get(bucket))
            .unwrap_or(&cache.global);
        kernel.completion_feasible(forward, coupled, bounds)
    }
    // A bucket's own active labels need a narrower hull than its inherited
    // completion minimum. Keep this key scoped to that row's scan only.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn resource_bucket_key(
        &self,
        g: &Graph<D>,
        forward: bool,
        coupled: Option<&crate::graph::coupled::CoupledLabelState>,
        presence: &LabelState,
        key: f64,
        vertex: usize,
        bucket: usize,
    ) -> f64 {
        let Some(kernel) = g.coupled_kernel().filter(|k| k.has_costs()) else {
            return self.presence_completion_key(key, presence);
        };
        let Some(cache) = self.coupled_cache(g, forward) else {
            return f64::NEG_INFINITY;
        };
        let bounds = cache
            .buckets
            .get(vertex)
            .and_then(|row| row.get(bucket))
            .unwrap_or(&cache.global);
        let correction = kernel.completion_correction_lower(forward, coupled, bounds);
        crate::core::rounding::add_down(self.presence_completion_key(key, presence), correction)
            .unwrap_or(f64::NEG_INFINITY)
    }
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
        // Keys already include the resource correction lower bound. Unknown
        // state hulls leave a -infinity key and cannot authorize pruning.
        if !self.cuts.presence.active() && !self.coupled_costs {
            return a + b;
        }
        crate::core::rounding::add_down(a, b).unwrap_or(f64::NEG_INFINITY)
    }
}
