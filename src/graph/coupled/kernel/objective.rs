//! Absolute model coefficients, separate from intrinsic resource activities.
use super::*;

impl CoupledKernel {
    /// Minimum-front labeling does not optimize a negative paired objective.
    /// Such models need explicit source controls and a maximum-value bound.
    pub(crate) fn has_negative_paired_objectives(&self) -> bool {
        self.negative_paired_objective
    }

    /// Model coefficient of this descriptor's accumulated charge. Other
    /// descriptor families retain their declared objective contract.
    #[cfg(feature = "clp")]
    pub(crate) fn effective_objective_coefficient(&self, resource: usize) -> Option<f64> {
        self.objective_coefficients
            .get(resource)
            .and_then(|value| finite(self.cost_scale * value).ok())
    }
    pub(crate) fn objective_scale(&self) -> f64 {
        self.cost_scale
    }
    pub fn objective_coefficient(&self, resource: usize) -> Option<f64> {
        self.objective_coefficients.get(resource).copied()
    }
}

impl<const D: usize> Graph<D> {
    /// Replace absolute objective coefficients for custom/cumulative or
    /// cost-bearing paired resources. Negative paired coefficients require the
    /// Clp-enabled source-control master and a source model. Both absolute
    /// and final-minus-initial normalization retain their declared meaning.
    /// Ordinary minimum-front labeling cannot optimize negative coefficients.
    /// Omitted descriptors have coefficient one. Intrinsic activities, numeric
    /// dependencies and feasibility remain unchanged. This is not an additive
    /// q-1 rewrite, which would erase sufficiently small nonzero coefficients.
    pub fn set_coupled_objective_coefficients(
        &mut self,
        coefficients: &[(usize, f64)],
    ) -> Result<()> {
        let Some(original) = self.coupled_kernel() else {
            return if coefficients.is_empty() {
                Ok(())
            } else {
                Err(invalid("objective coefficient requires coupled resources"))
            };
        };
        let mut values = vec![1.; original.definitions.len()];
        let mut seen = std::collections::HashSet::new();
        for &(resource, coefficient) in coefficients {
            if !coefficient.is_finite()
                || !seen.insert(resource)
                || !match original.definitions.get(resource) {
                    Some(
                        CoupledResource::CustomResource { .. }
                        | CoupledResource::CumulativeCost { .. },
                    ) => true,
                    Some(
                        CoupledResource::PairedPiecewiseLinear { mode, .. }
                        | CoupledResource::SlopePaired { mode, .. },
                    ) => {
                        *mode != PairedResourceMode::Feasibility
                            && (coefficient >= 0.
                                || (cfg!(feature = "clp") && !original.front_program_domain))
                    }
                    _ => false,
                }
            {
                return Err(invalid("invalid accumulated objective coefficient"));
            }
            values[resource] = coefficient;
        }
        if values.as_slice() == original.objective_coefficients.as_ref() {
            return Ok(());
        }
        let mut kernel = original.clone();
        kernel.identity = std::sync::Arc::new(());
        kernel.weighted_objective = values.iter().any(|&value| value != 1.);
        kernel.negative_paired_objective =
            original
                .definitions
                .iter()
                .enumerate()
                .any(|(r, definition)| {
                    values[r] < 0.
                        && matches!(
                            definition,
                            CoupledResource::SlopePaired { .. }
                                | CoupledResource::PairedPiecewiseLinear { .. }
                        )
                });
        kernel.objective_coefficients = values.into();
        for (&resource, prepared) in kernel.relation_indices.iter().zip(&mut kernel.paired) {
            prepared.objective_coefficient = kernel.objective_coefficients[resource];
        }
        for (resource, prepared) in kernel
            .definitions
            .iter()
            .enumerate()
            .filter(|(_, definition)| matches!(definition, CoupledResource::SlopePaired { .. }))
            .map(|(resource, _)| resource)
            .zip(&mut kernel.slope_paired)
        {
            prepared.objective_coefficient = kernel.objective_coefficients[resource];
        }
        kernel.transitions = PreparedTransitions::new(
            &kernel.definitions,
            &kernel.slots,
            kernel.initial.len(),
            kernel.arc_count,
            kernel.cost_scale,
            &kernel.objective_coefficients,
        );
        self.coupled = Some(Box::new(kernel));
        Ok(())
    }
}

#[cfg(test)]
#[path = "../../../../tests/unit/graph/coupled/kernel/activity_objective_tests.rs"]
mod tests;
