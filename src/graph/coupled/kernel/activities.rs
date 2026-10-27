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

/// Explicit base coordinates for one paired descriptor on a physical arc path.
/// Coordinates include the initial base and one head coordinate per arc.
/// They select a source control, including choices discarded by minimum-front
/// optimization. The descriptor index is scoped to the recipient kernel.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "clp", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "clp", serde(deny_unknown_fields))]
pub struct PairedControl {
    pub resource: usize,
    pub base: Vec<f64>,
}

/// Original resource charge and selected physical paired values for a path.
#[derive(Clone, Debug, PartialEq)]
pub struct CoupledPathValues {
    pub cost: f64,
    pub paired_values: Vec<(usize, f64)>,
}

impl CoupledKernel {
    /// Replay supplied source controls and retain their selected physical values.
    /// Controls are strictly descriptor-ordered and require singleton independent
    /// source certificates. Machine-front model controls remain unsupported.
    /// Empty controls preserve ordinary minimum-front replay without proof work.
    pub fn replay_with_controls(
        &self,
        arcs: &[usize],
        controls: &[PairedControl],
    ) -> Result<Option<CoupledPathValues>> {
        self.replay_with_controls_using(arcs, controls, &mut 1_000_000, None)
    }

    pub(crate) fn replay_with_controls_using(
        &self,
        arcs: &[usize],
        controls: &[PairedControl],
        _work: &mut usize,
        _deadline: Option<std::time::Instant>,
    ) -> Result<Option<CoupledPathValues>> {
        if !controls.is_empty() {
            let Some(remaining) = arcs
                .len()
                .checked_add(1)
                .and_then(|width| width.checked_mul(self.definitions.len().max(1)))
                .and_then(|required| _work.checked_sub(required))
            else {
                return Ok(None);
            };
            *_work = remaining;
        }
        if _deadline.is_some_and(|end| std::time::Instant::now() >= end) {
            return Ok(None);
        }
        let Some((state, cost)) = self.replay_state(arcs)? else {
            return Ok(None);
        };
        let values = self.selected_values_from_state(&state)?;
        if controls.is_empty() {
            return Ok(Some(CoupledPathValues {
                cost,
                paired_values: values,
            }));
        }
        #[cfg(not(feature = "clp"))]
        {
            return Err(invalid(
                "explicit paired controls require source certification with clp",
            ));
        }
        #[cfg(feature = "clp")]
        {
            if self.front_program_domain {
                return Err(invalid(
                    "source controls cannot replace machine-front model choices",
                ));
            }
            let mut values = values;
            let mut previous = None;
            let mut selected = Vec::with_capacity(controls.len());
            for control in controls {
                if previous.is_some_and(|index| index >= control.resource) {
                    return Err(invalid(
                        "paired controls must be unique and descriptor ordered",
                    ));
                }
                previous = Some(control.resource);
                let Some(certificate) =
                    self.certify_paired_control(arcs, control, _work, _deadline)?
                else {
                    return Ok(None);
                };
                let value = certificate.dependent[0];
                if !certificate.certifies_value(value) {
                    return Ok(None);
                }
                if let Some((_, saved)) = values
                    .iter_mut()
                    .find(|(index, _)| *index == control.resource)
                {
                    *saved = value;
                }
                selected.push((control.resource, value));
            }
            let cost = if arcs.is_empty() {
                0.
            } else {
                self.final_resource_objective(&state, &selected)?
            };
            if _deadline.is_some_and(|end| std::time::Instant::now() >= end) {
                return Ok(None);
            }
            Ok(Some(CoupledPathValues {
                cost,
                paired_values: values,
            }))
        }
    }

    /// Selected physical dependent values for cost-bearing paired descriptors.
    /// Values are descriptor-indexed and independent of cost suppression and
    /// normalization. This replays the current minimum-front choice; it is
    /// not a control trajectory or an independent feasibility certificate.
    pub fn selected_paired_values(&self, arcs: &[usize]) -> Result<Option<Vec<(usize, f64)>>> {
        if !self.slope_paired.iter().any(|p| p.has_costs())
            && !self.paired.iter().any(|p| p.has_costs())
        {
            return Ok(Some(Vec::new()));
        }
        let Some((state, _)) = self.replay_state(arcs)? else {
            return Ok(None);
        };
        Ok(Some(self.selected_values_from_state(&state)?))
    }
    fn selected_values_from_state(&self, state: &CoupledLabelState) -> Result<Vec<(usize, f64)>> {
        let mut values = Vec::new();
        let mut slopes = state.slope_relations.iter().zip(&self.slope_paired);
        let mut relations = state.relations.iter().zip(&self.paired);
        for (resource, definition) in self.definitions.iter().enumerate() {
            match definition {
                CoupledResource::SlopePaired { .. } => {
                    let (label, prepared) = slopes.next().expect("prepared paired shape");
                    if prepared.has_costs() {
                        values.push((resource, prepared.selected_value(label, self.cost_scale)?));
                    }
                }
                CoupledResource::PairedPiecewiseLinear { .. } => {
                    let (label, prepared) = relations.next().expect("prepared paired shape");
                    if prepared.has_costs() {
                        values.push((
                            resource,
                            finite(
                                label
                                    .forward()
                                    .ok_or_else(|| invalid("missing paired front"))?
                                    .minimum(),
                            )?,
                        ));
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
    /// contributes its current minimum, with the descriptor's normalization.
    /// This does not assert a source-domain or exact-real feasibility proof.
    pub fn replay_cost_components(&self, arcs: &[usize]) -> Result<Option<CoupledPathCosts>> {
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
            if matches!(definition, CoupledResource::SlopePaired { .. }) {
                resources[resource] = slope.next().expect("prepared slope resource shape").minimum;
            }
        }
        for ((&resource, prepared), label) in self
            .relation_indices
            .iter()
            .zip(&self.paired)
            .zip(&state.relations)
        {
            resources[resource] = prepared.objective(label, self.cost_scale)?;
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
