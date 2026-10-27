//! Pricing-only accumulated activities; original resource programs stay intact.
use super::*;

impl CoupledLabelState {
    pub(super) fn pricing_objective(&self) -> Result<f64> {
        self.pricing_activities
            .as_ref()
            .map_or(Ok(self.resource_objective), |states| {
                states
                    .iter()
                    .try_fold(self.resource_objective, |cost, state| {
                        finite(cost + state.resource_objective)
                    })
            })
    }
}

impl CoupledKernel {
    /// Original resource charge plus explicit pricing-only activity adjustments.
    /// Unlike `replay`, this value is not the original model objective.
    pub fn replay_pricing(&self, arcs: &[usize]) -> Result<Option<f64>> {
        self.replay_state(arcs)?
            .map(|(state, _)| state.pricing_objective())
            .transpose()
    }

    pub fn join(&self, f: &CoupledLabelState, b: &CoupledLabelState) -> Result<Option<f64>> {
        let Some(mut cost) = self.join_original(f, b)? else {
            return Ok(None);
        };
        if let (Some(front), Some(back)) = (&f.pricing_activities, &b.pricing_activities) {
            for ((kernel, front), back) in self
                .pricing_activities
                .iter()
                .zip(front.iter())
                .zip(back.iter())
            {
                let Some(extra) = kernel.join_original(front, back)? else {
                    return Ok(None);
                };
                cost = finite(cost + extra)?;
            }
        }
        Ok(Some(cost))
    }

    pub(crate) fn join_resource_objective(
        &self,
        f: &CoupledLabelState,
        b: &CoupledLabelState,
    ) -> Result<Option<f64>> {
        Ok(self.join_objective_costs(f, b)?.map(|(pricing, _)| pricing))
    }

    pub(crate) fn join_objective_costs(
        &self,
        f: &CoupledLabelState,
        b: &CoupledLabelState,
    ) -> Result<Option<(f64, f64)>> {
        let Some(original) = self.join_original_objective(f, b)? else {
            return Ok(None);
        };
        let mut pricing = original;
        if let (Some(front), Some(back)) = (&f.pricing_activities, &b.pricing_activities) {
            for ((kernel, front), back) in self
                .pricing_activities
                .iter()
                .zip(front.iter())
                .zip(back.iter())
            {
                let Some(extra) = kernel.join_original_objective(front, back)? else {
                    return Ok(None);
                };
                pricing = finite(pricing + extra)?;
            }
        }
        Ok(Some((pricing, original)))
    }

    /// Enclose the pricing objective separately from the original objective.
    /// Every component uses directed replay and the same cooperative budget.
    #[cfg(feature = "clp")]
    pub fn certify_pricing_with_budget(
        &self,
        arcs: &[usize],
        work: &mut usize,
        deadline: Option<std::time::Instant>,
    ) -> Option<(bool, [f64; 2])> {
        let (valid, mut cost) = self.certify_with_budget(arcs, work, deadline)?;
        if !valid {
            return Some((false, [0.; 2]));
        }
        for &(resource, coefficient) in self.pricing_adjustments.iter() {
            let (valid, activity) =
                self.certify_accumulated_activity(resource, arcs, work, deadline)?;
            if !valid {
                return Some((false, [0.; 2]));
            }
            *work = work.checked_sub(1)?;
            let extra = bounds::Range {
                lo: activity[0],
                hi: activity[1],
            }
            .multiply(bounds::Range::point(coefficient));
            cost = [
                crate::core::rounding::add_down(cost[0], extra.lo)?,
                crate::core::rounding::add_up(cost[1], extra.hi)?,
            ];
        }
        if deadline.is_some_and(|limit| std::time::Instant::now() >= limit) {
            None
        } else {
            Some((true, cost))
        }
    }
}

impl<const D: usize> Graph<D> {
    /// Replace pricing-only charges on deterministic custom/cumulative activities.
    /// Each pair is (original descriptor index, additive pricing coefficient).
    /// Original charges, feasibility and descriptor identities remain unchanged.
    /// Zero coefficients clear an adjustment. Duplicate indices and other
    /// resource families are errors; failure leaves the graph untouched.
    ///
    /// This is a numerical pricing objective. Multiplication and accumulation
    /// retain ordered binary64 arithmetic; the separate pricing certificate
    /// encloses the corresponding source expression. Updating these charges
    /// invalidates old prepared pricing states, including cached raw forks.
    pub fn set_coupled_pricing_adjustments(&mut self, adjustments: &[(usize, f64)]) -> Result<()> {
        let Some(original) = self.coupled_kernel() else {
            return if adjustments.is_empty() {
                Ok(())
            } else {
                Err(invalid("pricing activity requires coupled resources"))
            };
        };
        let mut ordered = adjustments.to_vec();
        ordered.sort_by_key(|&(resource, _)| resource);
        let mut resources = Vec::with_capacity(ordered.len());
        let mut previous = None;
        for &(resource, coefficient) in &ordered {
            if previous == Some(resource) || !coefficient.is_finite() {
                return Err(invalid("invalid or duplicate pricing activity"));
            }
            previous = Some(resource);
            let definition = original
                .definitions
                .get(resource)
                .ok_or_else(|| invalid("unknown pricing activity descriptor"))?;
            let definition = match definition {
                CoupledResource::CustomResource {
                    increments,
                    weights,
                    ..
                } => CoupledResource::CustomResource {
                    increments: increments.clone(),
                    weights: weights.clone(),
                    forward_bound: None,
                    backward_bound: None,
                },
                CoupledResource::CumulativeCost {
                    initial,
                    increments,
                    weights,
                    ..
                } => CoupledResource::CumulativeCost {
                    initial: *initial,
                    increments: increments.clone(),
                    weights: weights.clone(),
                    sink_upper: None,
                    source_upper: None,
                },
                _ => {
                    return Err(invalid(
                        "pricing activity requires a deterministic accumulated resource",
                    ))
                }
            };
            if coefficient == 0. {
                continue;
            }
            let mut activity = CoupledKernel::prepare(self, &[definition])?;
            activity.cost_scale = coefficient;
            activity.transitions = PreparedTransitions::new(
                &activity.definitions,
                &activity.slots,
                activity.initial.len(),
                activity.arc_count,
                coefficient,
                &activity.objective_coefficients,
            );
            resources.push(activity);
        }
        ordered.retain(|(_, coefficient)| *coefficient != 0.);
        if original.pricing_adjustments.as_ref() == ordered {
            return Ok(());
        }
        let mut kernel = original.clone();
        kernel.identity = std::sync::Arc::new(());
        kernel.pricing_adjustments = ordered.into();
        kernel.pricing_activities = resources.into();
        self.coupled = Some(Box::new(kernel));
        Ok(())
    }

    /// Replay the pricing resource objective; original charges use
    /// `coupled_path_cost`. Arguments are physical arc positions.
    pub fn coupled_pricing_path_cost(&self, arcs: &[usize]) -> Result<Option<f64>> {
        self.coupled_kernel()
            .map_or(Ok(Some(0.)), |kernel| kernel.replay_pricing(arcs))
    }
}

#[cfg(test)]
#[path = "../../../../tests/unit/graph/coupled/kernel/activity_pricing_tests.rs"]
mod tests;
