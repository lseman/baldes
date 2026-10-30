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
        self.has_costs() || !self.front_indices.is_empty() || self.has_slope_fronts()
    }
}

impl CoupledLabelState {
    fn paired_key_components(&self) -> Option<(f64, f64)> {
        if !self.paired_ordered_key {
            return None;
        }
        let [label] = self.slope_relations.as_slice() else {
            return None;
        };
        let history = label.machine_history.as_ref()?;
        let key = match history.last() {
            descriptor::MachinePathSnapshot::Root(root) => root.key(),
            descriptor::MachinePathSnapshot::Forward { state, .. } => state.key(),
            descriptor::MachinePathSnapshot::Backward { key, .. } => *key,
        };
        let scalar = self.accounting?.reduced;
        Some((key, scalar))
    }

    /// Ordered completion key for an unscaled paired front and scalar label cost.
    pub(crate) fn paired_completion_key(&self, half_potential: f64, cut_cost: f64) -> Option<f64> {
        let (key, scalar) = self.paired_key_components()?;
        Some((half_potential + cut_cost) + ((0. + key) + scalar))
    }

    /// Candidate key used after the first element of a completion scan.
    pub(crate) fn paired_completion_scan_key(
        &self,
        half_potential: f64,
        cut_cost: f64,
    ) -> Option<f64> {
        let (key, scalar) = self.paired_key_components()?;
        Some(((0. + key) + scalar) + (cut_cost + half_potential))
    }

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
        let reduced = finite(costs.reduced + self.pricing_objective()?)?;
        let original = finite(costs.original + self.resource_objective)?;
        self.accounting = Some(costs);
        Ok((reduced, original))
    }
}

#[cfg(test)]
#[path = "../../../../tests/unit/graph/coupled/kernel/accounting/tests.rs"]
mod tests;
