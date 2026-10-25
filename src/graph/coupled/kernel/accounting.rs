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
mod tests {
    use super::*;
    use crate::graph::coupled::PiecewiseLinearFunction;

    #[test]
    fn scalar_bases_survive_resource_cancellation_and_stay_independent() {
        let large = 9_007_199_254_740_994.;
        let mut graph = Graph::<2>::new(3, 0).unwrap();
        graph.add_arc(0, 1, -large).unwrap();
        graph.add_arc(1, 2, 0.).unwrap();
        let constant =
            |value| PiecewiseLinearFunction::new(vec![[0., value], [1., value]]).unwrap();
        graph
            .set_coupled_resources(&[CoupledResource::IntervalCost {
                initial: Window::new(0., 1.).unwrap(),
                bounds: Window::new(0., 1.).unwrap(),
                increments: vec![0.; 2],
                functions: vec![constant(1.), constant(large)],
            }])
            .unwrap();
        let kernel = graph.coupled_kernel().unwrap();
        let root = kernel.root(true);
        let (mut first, _) = kernel.extend(&root, 0, true).unwrap().unwrap();
        first.account_extension(&root, -large, -large + 2.).unwrap();
        let (mut last, _) = kernel.extend(&first, 1, true).unwrap().unwrap();
        let costs = last.account_extension(&first, 0., 0.).unwrap();
        assert_eq!(costs, (2., 4.));
        assert_eq!(last.accounting.unwrap().reduced, -large);
        assert_eq!(last.accounting.unwrap().original, -large + 2.);
        // Interleaving deltas would incorrectly produce four instead of the canonical two.
        assert_eq!((-large + 1.) + ((1. + large) - 1.), 4.);
    }

    #[test]
    fn missing_prefix_accounting_declines_instead_of_inventing_a_scalar_base() {
        let mut graph = Graph::<2>::new(2, 0).unwrap();
        graph.add_arc(0, 1, 0.).unwrap();
        graph
            .set_coupled_resources(&[CoupledResource::Selection {
                initial: 1,
                allowed: vec![1],
                vertices: vec![],
                relaxed: false,
            }])
            .unwrap();
        let kernel = graph.coupled_kernel().unwrap();
        let root = kernel.root(true);
        let (mut next, _) = kernel.extend(&root, 0, true).unwrap().unwrap();
        let untracked = next.clone();
        assert!(next.account_extension(&untracked, 1., 2.).is_err());
    }
}
