//! Preserve scalar arc sums independently of resource-function minima.
use super::*;

/// Literal physical-prefix sums. Never reconstruct these by subtracting a
/// resource minimum from a mixed rounded label key: that loses information.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(super) struct ScalarCosts {
    pub(super) reduced: f64,
    pub(super) original: f64,
}
impl CoupledKernel {
    /// Rounded resource objectives are separate components: mixed-key
    /// margins alone cannot finance cut-state cost tradeoffs.
    pub(crate) fn retains_cost_components(&self) -> bool {
        self.has_costs()
            || !self.front_indices.is_empty()
            || self.has_slope_fronts()
            || !self.relation_indices.is_empty()
    }
}

impl CoupledLabelState {
    pub(crate) fn scalar_reduced_cost(&self) -> Option<f64> {
        self.accounting.map(|costs| costs.reduced)
    }
    pub(crate) fn scalar_original_cost(&self) -> Option<f64> {
        self.accounting.map(|costs| costs.original)
    }

    /// Attach one physical arc's scalar costs to a freshly extended state.
    /// Cut accumulators that are stored separately by the caller remain
    /// separate; including them here as well would charge them twice.
    pub(crate) fn account_extension(
        &mut self,
        previous: &Self,
        reduced_increment: f64,
        original_increment: f64,
    ) -> Result<(f64, f64)> {
        let previous = previous.accounting.ok_or_else(|| {
            invalid("coupled label is missing its physical scalar cost accounting")
        })?;
        let costs = ScalarCosts {
            reduced: finite(previous.reduced + finite(reduced_increment)?)?,
            original: finite(previous.original + finite(original_increment)?)?,
        };
        let reduced = finite(costs.reduced + self.resource_objective)?;
        let original = finite(costs.original + self.resource_objective)?;
        self.accounting = Some(costs);
        Ok((reduced, original))
    }
}

#[cfg(test)]
#[path = "../../../../tests/unit/graph/coupled/kernel/accounting/tests.rs"]
mod tests;
