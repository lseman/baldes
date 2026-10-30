//! Absolute model coefficients, separate from intrinsic resource activities.
use super::*;

impl CoupledKernel {
    /// Whether raw resource pricing contains a negative paired front charge.
    /// Owner-associated coefficients are separate and never change that charge.
    pub(crate) fn has_negative_paired_objectives(&self) -> bool {
        self.negative_paired_objective
    }

    pub fn objective_coefficient(&self, resource: usize) -> Option<f64> {
        self.paired_associated_costs
            .iter()
            .find(|&&(index, _)| index == resource)
            .map(|&(_, coefficient)| coefficient)
            .or_else(|| self.objective_coefficients.get(resource).copied())
    }
}

impl<const D: usize> Graph<D> {
    /// Replace model objective coefficients for accumulated resources.
    /// For paired fronts these register the original associated-variable charge;
    /// raw label pricing and selection retain their unit paired resource cost.
    /// All finite signs are accepted without negating or maximizing a front.
    /// Omitted descriptors have coefficient one. Intrinsic activities, numeric
    /// dependencies and feasibility remain unchanged.
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
        let mut paired_costs = Vec::new();
        let mut seen = std::collections::HashSet::new();
        for &(resource, coefficient) in coefficients {
            if !coefficient.is_finite()
                || !seen.insert(resource)
                || !match original.definitions.get(resource) {
                    Some(
                        CoupledResource::CustomResource { .. }
                        | CoupledResource::CumulativeCost { .. },
                    ) => true,
                    Some(CoupledResource::PairedFront { mode, .. }) => {
                        *mode != PairedResourceMode::Feasibility
                    }
                    _ => false,
                }
            {
                return Err(invalid("invalid accumulated objective coefficient"));
            }
            if matches!(
                original.definitions[resource],
                CoupledResource::PairedFront { .. }
            ) {
                paired_costs.push((resource, coefficient));
            } else {
                values[resource] = coefficient;
            }
        }
        paired_costs.sort_by_key(|&(resource, _)| resource);
        if values.as_slice() == original.objective_coefficients.as_ref()
            && paired_costs.as_slice() == original.paired_associated_costs.as_ref()
        {
            return Ok(());
        }
        let mut kernel = original.clone();
        kernel.identity = std::sync::Arc::new(());
        kernel.weighted_objective = values.iter().any(|&value| value != 1.);
        // Backward paired summaries prove a componentwise final-value order.
        // Multiplication by the same finite nonnegative coefficient preserves
        // that order even when the product rounds. Arbitrarily weighted
        // numeric transition programs still require their existing proof.
        kernel.nonnegative_paired_weights_only =
            values
                .iter()
                .zip(kernel.definitions.iter())
                .all(|(&value, definition)| {
                    value == 1.
                        || (value >= 0.
                            && matches!(definition, CoupledResource::PairedFront { .. }))
                });
        kernel.negative_paired_objective =
            original
                .definitions
                .iter()
                .enumerate()
                .any(|(r, definition)| {
                    values[r] < 0. && matches!(definition, CoupledResource::PairedFront { .. })
                });
        kernel.objective_coefficients = values.into();
        kernel.paired_associated_costs = paired_costs.into();
        for (resource, prepared) in kernel
            .definitions
            .iter()
            .enumerate()
            .filter(|(_, definition)| matches!(definition, CoupledResource::PairedFront { .. }))
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

impl CoupledKernel {
    pub(crate) fn paired_associated_costs(&self) -> &[(usize, f64)] {
        &self.paired_associated_costs
    }
}

impl<const D: usize> Graph<D> {
    /// Retain original associated-variable coefficients without modifying the
    /// paired machine program's unit resource charge.
    pub(crate) fn set_paired_associated_costs(&mut self, costs: &[(usize, f64)]) -> Result<()> {
        let kernel = self
            .coupled
            .as_mut()
            .ok_or_else(|| invalid("paired association requires coupled resources"))?;
        let mut costs = costs.to_vec();
        costs.sort_by_key(|&(resource, _)| resource);
        for (index, &(resource, coefficient)) in costs.iter().enumerate() {
            if !coefficient.is_finite()
                || index > 0 && costs[index - 1].0 == resource
                || !matches!(
                    kernel.definitions.get(resource),
                    Some(CoupledResource::PairedFront {
                        mode: PairedResourceMode::CostBounded | PairedResourceMode::CostUnbounded
                    })
                )
                || kernel.objective_coefficients[resource] != 1.
            {
                return Err(invalid("invalid paired associated-variable cost"));
            }
        }
        if costs.as_slice() != kernel.paired_associated_costs.as_ref() {
            kernel.paired_associated_costs = costs.into();
            kernel.identity = std::sync::Arc::new(());
        }
        Ok(())
    }
}

impl CoupledKernel {
    pub(crate) fn owner_registration_bound_is_unsupported([lower, upper]: [f64; 2]) -> bool {
        let active = upper == f64::INFINITY || upper - (1e-10 * upper.abs() + 1e-6) > 0.;
        active && lower - (1e-10 * lower.abs().max(1e-6) + 1e-6) > 1e-6
    }

    /// Explicit owner registration bounds, separate from graph resource windows.
    pub fn paired_owner_registration_bounds(&self) -> &[(usize, [f64; 2])] {
        &self.paired_owner_registration_bounds
    }
}
impl<const D: usize> Graph<D> {
    /// Retain owner-variable input bounds for preprocessing and callback admission.
    /// These do not truncate individual paired paths or add shared master rows.
    /// Missing registrations have an unbounded owner relaxation. Infinite ends
    /// are accepted; bounds from verified graph envelopes are not substituted.
    pub fn set_paired_owner_registration_bounds(
        &mut self,
        bounds: &[(usize, [f64; 2])],
    ) -> Result<()> {
        let original = self
            .coupled_kernel()
            .ok_or_else(|| invalid("owner registration bounds require paired resources"))?;
        let mut values = bounds.to_vec();
        values.sort_by_key(|&(resource, _)| resource);
        for (position, &(resource, [lower, upper])) in values.iter().enumerate() {
            if lower.is_nan()
                || upper.is_nan()
                || lower == f64::INFINITY
                || upper == f64::NEG_INFINITY
                || lower > upper
                || position > 0 && values[position - 1].0 == resource
                || !matches!(
                    original.definitions.get(resource),
                    Some(CoupledResource::PairedFront {
                        mode: PairedResourceMode::CostBounded | PairedResourceMode::CostUnbounded
                    })
                )
            {
                return Err(invalid("invalid paired owner registration bounds"));
            }
        }
        let mut kernel = original.clone();
        kernel.identity = std::sync::Arc::new(());
        kernel.paired_owner_registration_bounds = values.into();
        self.coupled = Some(Box::new(kernel));
        Ok(())
    }
}
