//! Descriptor-scoped charges, retaining the shared transition program.
use super::*;

/// Numerical resource charges on one replayed physical path.
///
/// `resources[i]` belongs to descriptor `i`; feasibility-only resources have
/// zero charge. The total retains the existing combined labeling accumulation
/// order. Summing components can round differently and must not replace that
/// total. These values are numerical activities, not directed certificates.
#[derive(Clone, Debug, PartialEq)]
pub struct CoupledPathCosts {
    pub total: f64,
    pub resources: Vec<f64>,
}

impl CoupledKernel {
    /// Selected physical dependent values for cost-bearing paired descriptors.
    /// Values are descriptor-indexed and independent of cost suppression and
    /// objective scaling. This replays the current minimum-front choice; it is
    /// not a control trajectory or an independent feasibility certificate.
    pub fn selected_paired_values(&self, arcs: &[usize]) -> Result<Option<Vec<(usize, f64)>>> {
        if !self.slope_paired.iter().any(|p| p.has_costs()) {
            return Ok(Some(Vec::new()));
        }
        if self.machine_paired_join_complete() {
            let Some((_, selected)) = self.joined_machine_path(arcs)? else {
                return Ok(None);
            };
            let resource = self
                .definitions
                .iter()
                .position(|definition| matches!(definition, CoupledResource::PairedFront { .. }))
                .ok_or_else(|| invalid("paired native resource is unavailable"))?;
            return Ok(Some(vec![(
                resource,
                selected.ok_or_else(|| {
                    crate::incomplete("paired native selected value is unavailable")
                })?,
            )]));
        }
        let Some((state, _)) = self.replay_state(arcs)? else {
            return Ok(None);
        };
        let mut values = self.selected_values_from_state(&state)?;
        if self.front_program_domain
            && self
                .slope_paired
                .iter()
                .any(|paired| paired.has_machine_program_input())
        {
            let root = self.root(false);
            if self.join_original_objective(&state, &root)?.is_none() {
                return Ok(None);
            }
            for (resource, value) in &mut values {
                if !matches!(
                    self.definitions.get(*resource),
                    Some(CoupledResource::PairedFront { .. })
                ) {
                    continue;
                }
                match self.paired_machine_resource_join(
                    *resource,
                    &state,
                    &root,
                    descriptor::FrontierBudget::default(),
                ) {
                    descriptor::FrontierOutcome::Complete(selection) => {
                        *value = selection.dependent.ok_or_else(|| {
                            crate::incomplete("paired native selected value is unavailable")
                        })?;
                    }
                    descriptor::FrontierOutcome::Empty => return Ok(None),
                    descriptor::FrontierOutcome::Unknown(reason) => {
                        return Err(crate::incomplete(format!(
                            "paired native selection is unavailable: {reason:?}"
                        )));
                    }
                }
            }
        }
        Ok(Some(values))
    }
    fn selected_values_from_state(&self, state: &CoupledLabelState) -> Result<Vec<(usize, f64)>> {
        let mut values = Vec::new();
        let mut slopes = state.slope_relations.iter().zip(&self.slope_paired);
        for (resource, definition) in self.definitions.iter().enumerate() {
            match definition {
                CoupledResource::PairedFront { .. } => {
                    let (label, prepared) = slopes.next().expect("prepared paired shape");
                    if prepared.has_costs() {
                        values.push((resource, prepared.selected_value(label, self.cost_scale)?));
                    }
                }
                _ => {}
            }
        }
        Ok(values)
    }
    /// Intrinsic unit-charge activity of one custom/cumulative descriptor.
    /// Model-cost suppression and pricing-only adjustments do not change this
    /// activity. This evaluates the ordered numeric program, not feasibility
    /// of the physical path or the other resources. Use the independent
    /// activity certificate when a source-domain proof is needed.
    pub fn accumulated_activity(&self, resource: usize, arcs: &[usize]) -> Result<f64> {
        let (initial, increments, weights) = match self.definitions.get(resource) {
            Some(CoupledResource::CustomResource {
                increments,
                weights,
                ..
            }) => (0., increments, weights),
            Some(CoupledResource::CumulativeCost {
                initial,
                increments,
                weights,
                ..
            }) => (*initial, increments, weights),
            _ => return Err(invalid("activity requires a custom/cumulative descriptor")),
        };
        let mut numeric = [initial];
        let mut cost = 0.;
        for &arc in arcs {
            let operation = transitions::NumericOperation::Cumulative {
                slot: 0,
                increment: *increments
                    .get(arc)
                    .ok_or_else(|| invalid("unknown activity arc"))?,
                weight: weights[arc],
                cost_coefficient: 1.,
            };
            operation.apply(&mut numeric, 1., &mut cost)?;
        }
        Ok(cost)
    }

    /// Replay the current resource program and retain each descriptor's charge.
    ///
    /// Arguments are physical arc positions, as for `replay`. The same numerical
    /// constraints remain active, including cost-function domains when costs
    /// are suppressed. Returns None for a numerically infeasible path; errors
    /// and unsupported operations remain errors. A free-initial-value front
    /// contributes its current minimum.
    /// This does not assert a source-domain or exact-real feasibility proof.
    pub fn replay_cost_components(&self, arcs: &[usize]) -> Result<Option<CoupledPathCosts>> {
        if !arcs.is_empty()
            && self.machine_paired_join_complete()
            && self.paired_path_is_complete(arcs)?
        {
            let Some((total, selected)) =
                self.joined_associated_machine_path_matching(arcs, None)?
            else {
                return Ok(None);
            };
            let resource = self
                .retained_paired_machine_resource()
                .ok_or_else(|| invalid("paired resource is unavailable"))?;
            let mut resources = vec![0.; self.definitions.len()];
            resources[resource] = if self.slope_paired[0].machine_mode() == 4 {
                0.
            } else {
                finite(
                    self.cost_scale
                        * finite(
                            self.paired_associated_coefficient()
                                * selected
                                    .ok_or_else(|| invalid("paired selected value unavailable"))?,
                        )?,
                )?
            };
            return Ok(Some(CoupledPathCosts { total, resources }));
        }
        let mut state = self.root(true);
        let mut workspace = CoupledWorkspace::default();
        let mut resources = vec![0.; self.definitions.len()];
        for &arc in arcs {
            let Some((next, _)) =
                self.extend_internal(&state, arc, true, false, false, &mut workspace)?
            else {
                return Ok(None);
            };
            // Execute the same ordered scalar operations, retaining charges by
            // their original descriptor rather than reconstructing an activity
            // by subtracting mixed, already-rounded objective totals.
            let mut numeric = state.numeric.clone();
            let program = &self.transitions.arcs[arc];
            if program.operations.len() != self.transitions.operation_resources.len() {
                return Err(invalid("coupled operation descriptor shape mismatch"));
            }
            for (operation, &resource) in program
                .operations
                .iter()
                .zip(self.transitions.operation_resources.iter())
            {
                if !operation.apply(&mut numeric, self.cost_scale, &mut resources[resource])? {
                    return Ok(None);
                }
            }
            state = next;
        }
        if arcs.is_empty() {
            return Ok(Some(CoupledPathCosts {
                total: 0.,
                resources,
            }));
        }
        for (&resource, front) in self.front_indices.iter().zip(&state.fronts) {
            resources[resource] = front.minimum();
        }
        for (&resource, front) in self.slope_indices.iter().zip(&state.slope_fronts) {
            resources[resource] = front.minimum;
        }
        let mut slope = state.slope_relations.iter();
        for (resource, definition) in self.definitions.iter().enumerate() {
            if matches!(definition, CoupledResource::PairedFront { .. }) {
                resources[resource] = slope.next().expect("prepared slope resource shape").minimum;
            }
        }
        let mut plugins = state.plugins.iter();
        for (resource, definition) in self.definitions.iter().enumerate() {
            if let CoupledResource::Plugin(plugin) = definition {
                let label = plugins.next().expect("prepared plugin resource shape");
                if plugin.0.has_costs() && self.cost_scale != 0. {
                    resources[resource] =
                        finite(self.cost_scale * plugin.0.objective(label, true)?)?;
                }
            }
        }
        Ok(Some(CoupledPathCosts {
            total: state.resource_objective,
            resources,
        }))
    }
}

impl<const D: usize> Graph<D> {
    /// Selected paired values carried by a reconstructed route.
    pub fn coupled_paired_values(&self, arcs: &[usize]) -> Result<Vec<(usize, f64)>> {
        match self.coupled_kernel() {
            Some(kernel) => kernel
                .selected_paired_values(arcs)?
                .ok_or_else(|| invalid("cannot reconstruct paired values for an infeasible path")),
            None => Ok(Vec::new()),
        }
    }
}

#[cfg(test)]
#[path = "../../../../tests/unit/graph/coupled/kernel/activity_tests.rs"]
mod tests;
