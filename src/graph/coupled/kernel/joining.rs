//! Forward/backward label joins and joined resource objectives.
use super::*;
use crate::incomplete;

impl CoupledKernel {
    pub(super) fn paired_path_is_complete(&self, arcs: &[usize]) -> Result<bool> {
        let topology = self
            .slope_paired
            .first()
            .and_then(|paired| paired.machine_topology())
            .ok_or_else(|| incomplete("paired machine topology is unavailable"))?;
        let mut vertex = topology.source;
        for &index in arcs {
            let &(_, tail, head) = topology
                .arcs
                .get(index)
                .ok_or_else(|| invalid("unknown paired path arc"))?;
            if tail != vertex {
                return Ok(false);
            }
            vertex = head;
        }
        Ok(vertex == topology.sink)
    }
    pub(crate) fn has_single_custom_machine_resource(&self) -> bool {
        self.custom_indices.len() == 1
    }
    pub(crate) fn has_custom_machine_resource(&self) -> bool {
        !self.custom_indices.is_empty()
    }
    pub(crate) fn retained_paired_machine_resource(&self) -> Option<usize> {
        if !self.front_program_domain {
            return None;
        }
        let mut paired = 0;
        self.definitions.iter().position(|resource| {
            if !matches!(resource, CoupledResource::PairedFront { .. }) {
                return false;
            }
            let installed = self.slope_paired[paired].has_machine_program_input();
            paired += 1;
            installed
        })
    }
    /// Replay the native custom solution recorder over a physical arc path
    /// and compare it with the supplied joined label states. The retained
    /// suffix program and both raw callback states must match the path.
    /// The returned value is raw machine output, not a source certificate.
    pub fn record_custom_machine_path(
        &self,
        arcs: &[usize],
        forward: &CoupledLabelState,
        backward: &CoupledLabelState,
    ) -> Result<Option<f64>> {
        let [resource] = self.custom_indices.as_slice() else {
            return Err(invalid(
                "custom machine recorder requires one custom resource",
            ));
        };
        self.record_custom_machine_resource_path(*resource, arcs, forward, backward)
    }

    /// Replay one custom-resource recorder, selected by coupled-resource index.
    pub fn record_custom_machine_resource_path(
        &self,
        resource: usize,
        arcs: &[usize],
        forward: &CoupledLabelState,
        backward: &CoupledLabelState,
    ) -> Result<Option<f64>> {
        self.check_state(forward)?;
        self.check_state(backward)?;
        if forward.suffix.is_some() || backward.suffix.is_none() {
            return Err(invalid("custom machine recorder direction mismatch"));
        }
        let index = self
            .custom_indices
            .iter()
            .position(|&entry| entry == resource)
            .ok_or_else(|| invalid("custom machine resource unavailable"))?;
        let (left, right) = (
            &forward.custom_machine[index],
            &backward.custom_machine[index],
        );
        if !left.accepted || !right.accepted {
            return Ok(None);
        }
        let Some(CoupledResource::CustomResource {
            increments,
            weights,
            ..
        }) = self.definitions.get(resource)
        else {
            return Err(invalid("custom machine resource unavailable"));
        };
        if arcs.iter().any(|&arc| arc >= increments.len()) {
            return Err(invalid("unknown custom machine path arc"));
        }
        let suffix: Vec<_> = backward.suffix.as_ref().unwrap().program.iter().collect();
        if !arcs.ends_with(&suffix) {
            return Ok(None);
        }
        let prefix_len = arcs.len() - suffix.len();
        let mut prefix_state = crate::graph::coupled::CustomMachineState::default();
        for &arc in &arcs[..prefix_len] {
            prefix_state.extend_forward(increments[arc], weights[arc]);
        }
        let mut suffix_state = crate::graph::coupled::CustomMachineState::default();
        for &arc in suffix.iter().rev() {
            suffix_state.extend_backward(increments[arc], weights[arc]);
        }
        let same_state = |a: &crate::graph::coupled::CustomMachineState,
                          b: &crate::graph::coupled::CustomMachineState| {
            a.cost.to_bits() == b.cost.to_bits() && a.value.to_bits() == b.value.to_bits()
        };
        if !same_state(&prefix_state, &left.state) || !same_state(&suffix_state, &right.state) {
            return Ok(None);
        }
        let replay = |backward: bool| {
            let mut state = crate::graph::coupled::CustomMachineState::default();
            let mut total = 0.;
            let mut step = |arc: usize| {
                let value = if backward {
                    state.extend_backward(increments[arc], weights[arc])
                } else {
                    state.extend_forward(increments[arc], weights[arc])
                };
                total = if value == 1e12 { 1e12 } else { total + value };
                total = state.extend_vertex() + total;
            };
            if backward {
                arcs.iter().rev().copied().for_each(&mut step);
            } else {
                arcs.iter().copied().for_each(&mut step);
            }
            total
        };
        let forward_total = replay(false);
        let backward_total = replay(true);
        if (forward_total - backward_total).abs() > 1e-6 {
            return Ok(None);
        }
        let joined =
            (left.state.concatenate(&right.state) + left.wrapper_cost) + right.wrapper_cost;
        if (joined - forward_total).abs() > 1e-6 {
            return Ok(None);
        }
        Ok(Some(forward_total))
    }

    /// Evaluate the recovered single-custom-resource wrapper for two retained
    /// labels. This raw machine cost is not a source-model join certificate.
    pub fn custom_machine_join(
        &self,
        forward: &CoupledLabelState,
        backward: &CoupledLabelState,
        previous: f64,
    ) -> Result<Option<f64>> {
        let [resource] = self.custom_indices.as_slice() else {
            return Err(invalid("custom machine join requires one custom resource"));
        };
        self.custom_machine_resource_join(*resource, forward, backward, previous)
    }

    /// Evaluate one custom callback and wrapper by coupled-resource index.
    pub fn custom_machine_resource_join(
        &self,
        resource: usize,
        forward: &CoupledLabelState,
        backward: &CoupledLabelState,
        previous: f64,
    ) -> Result<Option<f64>> {
        self.check_state(forward)?;
        self.check_state(backward)?;
        if forward.suffix.is_some() || backward.suffix.is_none() {
            return Err(invalid("custom machine join direction mismatch"));
        }
        let index = self
            .custom_indices
            .iter()
            .position(|&entry| entry == resource)
            .ok_or_else(|| invalid("custom machine resource unavailable"))?;
        let (left, right) = (
            &forward.custom_machine[index],
            &backward.custom_machine[index],
        );
        if !left.accepted || !right.accepted {
            return Ok(None);
        }
        let cross = left.state.concatenate(&right.state);
        Ok(
            crate::graph::coupled::CustomMachineState::contribution_accepted(cross)
                .then(|| ((cross + left.wrapper_cost) + right.wrapper_cost) + previous),
        )
    }

    /// Evaluate the forward/forward custom wrapper for a selected resource.
    /// The callback returns zero; its ordered additions remain observable.
    pub fn custom_machine_resource_forward_join(
        &self,
        resource: usize,
        first: &CoupledLabelState,
        second: &CoupledLabelState,
        previous: f64,
    ) -> Result<Option<f64>> {
        self.check_state(first)?;
        self.check_state(second)?;
        if first.suffix.is_some() || second.suffix.is_some() {
            return Err(invalid("custom machine forward join direction mismatch"));
        }
        let index = self
            .custom_indices
            .iter()
            .position(|&entry| entry == resource)
            .ok_or_else(|| invalid("custom machine resource unavailable"))?;
        let (left, right) = (&first.custom_machine[index], &second.custom_machine[index]);
        if !left.accepted || !right.accepted {
            return Ok(None);
        }
        Ok(Some(
            ((0.0 + left.wrapper_cost) + right.wrapper_cost) + previous,
        ))
    }

    /// Whether retained machine contributions are the complete resource
    /// charge for inspection: one feasibility-only paired program contributes
    /// zero and has no pricing-only activity adjustments.
    pub(crate) fn uses_zero_paired_inspection_cost(&self) -> bool {
        self.machine_feasibility_join_complete() && self.pricing_activities.is_empty()
    }

    /// All non-paired resources here are checked above the machine scan by
    /// their selection, load, or budgeted join. None contributes an objective.
    pub(crate) fn machine_feasibility_join_complete(&self) -> bool {
        self.machine_paired_join_complete()
            && self.paired_objective_is_only_charge()
            && self.slope_paired[0].machine_mode() == 4
    }

    pub(crate) fn machine_paired_join_complete(&self) -> bool {
        self.front_program_domain
            && self.definitions.len() == 1
            && self.slope_paired.len() == 1
            && self.slope_paired[0].has_machine_program_input()
    }

    /// Paired-only keys omit every independently charged resource.
    pub(crate) fn paired_objective_is_only_charge(&self) -> bool {
        self.initial.is_empty()
            && self.front_indices.is_empty()
            && self.slope_indices.is_empty()
            && self.plugins.iter().all(|plugin| !plugin.0.has_costs())
    }

    pub(crate) fn paired_associated_coefficient(&self) -> f64 {
        self.retained_paired_machine_resource()
            .and_then(|resource| {
                self.paired_associated_costs
                    .iter()
                    .find(|&&(index, _)| index == resource)
                    .map(|&(_, coefficient)| coefficient)
            })
            .unwrap_or(1.)
    }

    pub(crate) fn joined_associated_machine_path_matching(
        &self,
        arcs: &[usize],
        selected: Option<f64>,
    ) -> Result<Option<(f64, Option<f64>)>> {
        self.joined_machine_path_matching_with_paired_coefficient(
            arcs,
            selected,
            self.paired_associated_coefficient(),
        )
    }

    pub(crate) fn join_associated_objective(
        &self,
        forward: &CoupledLabelState,
        backward: &CoupledLabelState,
    ) -> Result<Option<f64>> {
        self.join_original_objective_using(
            forward,
            backward,
            Some(self.paired_associated_coefficient()),
        )
    }

    /// Convert a retained paired join to the configured route contribution.
    pub(crate) fn paired_retained_cost(&self, value: f64) -> Result<f64> {
        if !self.machine_paired_join_complete() {
            return Err(invalid("paired retained cost is unavailable"));
        }
        self.slope_paired[0].final_objective(value, self.cost_scale)
    }

    /// Reconstruct a paired join of a physical path.
    /// A route exported from enumeration no longer carries its split states,
    /// so check every split using the same resource extensions and join.
    pub(super) fn joined_machine_path(&self, arcs: &[usize]) -> Result<Option<(f64, Option<f64>)>> {
        self.joined_machine_path_matching(arcs, None)
    }

    /// Enclose the paired contribution of all successful split joins of a
    /// physical path. Other resource charges are certified independently.
    pub(super) fn joined_machine_path_bounds(
        &self,
        arcs: &[usize],
        work: &mut usize,
        deadline: Option<std::time::Instant>,
    ) -> Option<Option<[f64; 2]>> {
        if !self.machine_paired_join_complete() {
            return None;
        }
        let mut take = |amount: usize| {
            if deadline.is_some_and(|end| std::time::Instant::now() >= end) {
                return None;
            }
            *work = work.checked_sub(amount)?;
            Some(())
        };
        take(arcs.len().checked_add(1)?)?;
        if !self.paired_path_is_complete(arcs).ok()? {
            return Some(None);
        }
        let mut suffixes = vec![None; arcs.len() + 1];
        suffixes[arcs.len()] = Some(self.root(false));
        let mut workspace = CoupledWorkspace::default();
        for index in (0..arcs.len()).rev() {
            take(1)?;
            let Some(suffix) = suffixes[index + 1].as_deref() else {
                break;
            };
            let Some((next, _)) = self
                .extend_internal(suffix, arcs[index], false, false, false, &mut workspace)
                .ok()?
            else {
                break;
            };
            suffixes[index] = Some(next);
        }
        let mut prefix = self.root(true);
        let mut bounds: Option<[f64; 2]> = None;
        for index in 0..=arcs.len() {
            take(1)?;
            if let Some(suffix) = suffixes[index].as_deref() {
                if self
                    .join_original_objective(&prefix, suffix)
                    .ok()?
                    .is_some()
                {
                    let descriptor::FrontierOutcome::Complete(selection) = self
                        .paired_machine_join_checked(
                            0,
                            &prefix,
                            suffix,
                            descriptor::FrontierBudget::default(),
                        )
                    else {
                        return None;
                    };
                    let cost = finite(
                        self.cost_scale
                            * finite(
                                self.paired_associated_coefficient() * selection.accumulate(0.),
                            )
                            .ok()?,
                    )
                    .ok()?;
                    if !cost.is_finite() {
                        return None;
                    }
                    match &mut bounds {
                        Some([lower, upper]) => {
                            *lower = lower.min(cost);
                            *upper = upper.max(cost);
                        }
                        None => bounds = Some([cost, cost]),
                    }
                }
            }
            if index == arcs.len() {
                break;
            }
            let Some((next, _)) = self
                .extend_internal(&prefix, arcs[index], true, false, false, &mut workspace)
                .ok()?
            else {
                break;
            };
            prefix = next;
        }
        take(0)?;
        Some(bounds)
    }

    pub(crate) fn joined_machine_path_matching(
        &self,
        arcs: &[usize],
        selected: Option<f64>,
    ) -> Result<Option<(f64, Option<f64>)>> {
        self.joined_machine_path_matching_using(arcs, selected, None)
    }

    /// Recompose the original resource charge with an owner-associated paired
    /// coefficient, without subtracting an already rounded mixed total.
    pub(crate) fn joined_machine_path_matching_with_paired_coefficient(
        &self,
        arcs: &[usize],
        selected: Option<f64>,
        coefficient: f64,
    ) -> Result<Option<(f64, Option<f64>)>> {
        if !coefficient.is_finite() {
            return Err(invalid("nonfinite paired associated objective coefficient"));
        }
        self.joined_machine_path_matching_using(arcs, selected, Some(coefficient))
    }

    fn joined_machine_path_matching_using(
        &self,
        arcs: &[usize],
        selected: Option<f64>,
        coefficient: Option<f64>,
    ) -> Result<Option<(f64, Option<f64>)>> {
        if !self.machine_paired_join_complete() {
            return Err(invalid("paired machine path unavailable"));
        }
        if !self.paired_path_is_complete(arcs)? {
            return Ok(None);
        }
        let mut suffixes = vec![None; arcs.len() + 1];
        suffixes[arcs.len()] = Some(self.root(false));
        let mut workspace = CoupledWorkspace::default();
        let mut unresolved = None;
        for index in (0..arcs.len()).rev() {
            let Some(suffix) = suffixes[index + 1].as_deref() else {
                break;
            };
            let next = match self.extend_internal(
                suffix,
                arcs[index],
                false,
                false,
                false,
                &mut workspace,
            ) {
                Ok(Some((next, _))) => next,
                Ok(None) => break,
                Err(error) => {
                    unresolved = Some(error);
                    break;
                }
            };
            suffixes[index] = Some(next);
        }
        let mut prefix = self.root(true);
        for index in 0..=arcs.len() {
            if let Some(suffix) = suffixes[index].as_deref() {
                match self.join_original_objective_using(&prefix, suffix, coefficient) {
                    Ok(Some(cost)) => match self.paired_machine_join_checked(
                        0,
                        &prefix,
                        suffix,
                        descriptor::FrontierBudget::default(),
                    ) {
                        descriptor::FrontierOutcome::Complete(selection) => {
                            if selected.is_none_or(|value| selection.dependent == Some(value)) {
                                return Ok(Some((cost, selection.dependent)));
                            }
                        }
                        descriptor::FrontierOutcome::Empty => {}
                        descriptor::FrontierOutcome::Unknown(reason) => {
                            if unresolved.is_none() {
                                unresolved = Some(incomplete(format!(
                                    "paired machine selection unavailable: {reason:?}"
                                )));
                            }
                        }
                    },
                    Ok(None) => {}
                    Err(error) => {
                        if unresolved.is_none() {
                            unresolved = Some(error);
                        }
                    }
                }
            }
            if index == arcs.len() {
                break;
            }
            let next = match self.extend_internal(
                &prefix,
                arcs[index],
                true,
                false,
                false,
                &mut workspace,
            ) {
                Ok(Some((next, _))) => next,
                Ok(None) => break,
                Err(error) => {
                    if unresolved.is_none() {
                        unresolved = Some(error);
                    }
                    break;
                }
            };
            prefix = next;
        }
        if let Some(error) = unresolved {
            return Err(error);
        }
        Ok(None)
    }

    /// Whether a retained scalar contains every coupled resource charge used
    /// by raw inspection under the current pricing configuration.
    pub(crate) fn has_complete_machine_inspection_contribution(&self) -> bool {
        if !self.pricing_activities.is_empty() {
            return false;
        }
        let mut charged = 0;
        for resource in self.definitions.iter() {
            match resource {
                CoupledResource::Selection { .. }
                | CoupledResource::SimultaneousPickupDelivery { .. }
                | CoupledResource::BudgetedLoad { .. }
                | CoupledResource::BudgetedTime { .. } => {}
                CoupledResource::CustomResource { .. } => charged += 1,
                CoupledResource::PairedFront { .. } if self.front_program_domain => charged += 1,
                _ => return false,
            }
        }
        charged == 1
            && (self.custom_indices.len() == 1
                || (self.front_program_domain && self.slope_paired.len() == 1))
    }

    /// Join the retained machine fronts of one paired resource. The resource
    /// index is in coupled-descriptor order. This raw selection is independent
    /// of other resource charges and is not a source-model certificate.
    pub fn paired_machine_resource_join(
        &self,
        resource: usize,
        forward: &CoupledLabelState,
        backward: &CoupledLabelState,
        budget: descriptor::FrontierBudget,
    ) -> descriptor::FrontierOutcome<descriptor::MachineConcatenation> {
        use descriptor::{FrontierOutcome, FrontierUnknown};
        let started = std::time::Instant::now();
        if budget.max_elapsed.is_some_and(|limit| limit.is_zero()) {
            return FrontierOutcome::Unknown(FrontierUnknown::Deadline);
        }
        let validation = self
            .definitions
            .len()
            .saturating_mul(2)
            .saturating_add(resource)
            .saturating_add(
                forward
                    .suffix
                    .as_ref()
                    .map_or(0, |suffix| suffix.bounds.len()),
            )
            .saturating_add(
                backward
                    .suffix
                    .as_ref()
                    .map_or(0, |suffix| suffix.bounds.len()),
            )
            .saturating_add(
                forward
                    .reversal_peaks
                    .as_ref()
                    .map_or(0, |peaks| peaks.len()),
            )
            .saturating_add(
                backward
                    .reversal_peaks
                    .as_ref()
                    .map_or(0, |peaks| peaks.len()),
            );
        let Some(remaining_work) = budget.max_operations.checked_sub(validation) else {
            return FrontierOutcome::Unknown(FrontierUnknown::WorkLimit);
        };
        if !matches!(
            self.definitions.get(resource),
            Some(CoupledResource::PairedFront { .. })
        ) || self.check_state(forward).is_err()
            || self.check_state(backward).is_err()
            || forward.suffix.is_some()
            || backward.suffix.is_none()
        {
            return FrontierOutcome::Unknown(FrontierUnknown::Unsupported(
                "paired machine join resource or direction",
            ));
        }
        let index = self.definitions[..resource]
            .iter()
            .filter(|definition| matches!(definition, CoupledResource::PairedFront { .. }))
            .count();
        let remaining_time = budget
            .max_elapsed
            .map(|limit| limit.saturating_sub(started.elapsed()));
        if remaining_time.is_some_and(|time| time.is_zero()) {
            return FrontierOutcome::Unknown(FrontierUnknown::Deadline);
        }
        self.paired_machine_join_checked(
            index,
            forward,
            backward,
            descriptor::FrontierBudget {
                max_operations: remaining_work,
                max_elapsed: remaining_time,
                ..budget
            },
        )
    }

    // The caller validates state shapes and the paired descriptor index.
    fn paired_machine_join_checked(
        &self,
        index: usize,
        forward: &CoupledLabelState,
        backward: &CoupledLabelState,
        budget: descriptor::FrontierBudget,
    ) -> descriptor::FrontierOutcome<descriptor::MachineConcatenation> {
        use descriptor::{FrontierOutcome, FrontierUnknown};
        let prepared = &self.slope_paired[index];
        let (Some(left), Some(right)) = (
            forward.slope_relations[index].machine_history.as_deref(),
            backward.slope_relations[index].machine_history.as_deref(),
        ) else {
            return FrontierOutcome::Unknown(FrontierUnknown::Unsupported(
                "paired machine join histories unavailable",
            ));
        };
        if !prepared.matches_forward_root(left) || !prepared.matches_backward_root(right) {
            return FrontierOutcome::Unknown(FrontierUnknown::Unsupported(
                "paired machine join histories are stale",
            ));
        }
        let ([left], [right_front], Some(ceiling)) = (
            left.last().front().components(),
            right.last().front().components(),
            right.backward_ceiling(),
        ) else {
            return FrontierOutcome::Unknown(FrontierUnknown::Unsupported(
                "paired machine join front or ceiling unavailable",
            ));
        };
        descriptor::PairedFrontOperations::machine_concatenation(
            left,
            right_front,
            ceiling,
            prepared.machine_mode(),
            budget,
        )
    }

    /// Raw resource value stored by the machine's enumeration candidate. It is
    /// kept separate from normalized source objectives and is available only
    /// when both labels carry the installed machine histories.
    pub(crate) fn machine_join_contribution(
        &self,
        forward: &CoupledLabelState,
        backward: &CoupledLabelState,
    ) -> Option<f64> {
        if !self.has_complete_machine_inspection_contribution()
            || self.check_state(forward).is_err()
            || self.check_state(backward).is_err()
            || forward.suffix.is_some()
            || backward.suffix.is_none()
        {
            return None;
        }
        if self.custom_indices.len() == 1 {
            return self
                .custom_machine_resource_join(self.custom_indices[0], forward, backward, 0.)
                .ok()
                .flatten();
        }
        if !self.front_program_domain || self.slope_paired.len() != 1 {
            return None;
        }
        match self.paired_machine_join_checked(
            0,
            forward,
            backward,
            descriptor::FrontierBudget::default(),
        ) {
            descriptor::FrontierOutcome::Complete(value) => Some(value.accumulate(0.)),
            descriptor::FrontierOutcome::Empty | descriptor::FrontierOutcome::Unknown(_) => None,
        }
    }

    pub(super) fn join_original(
        &self,
        f: &CoupledLabelState,
        b: &CoupledLabelState,
    ) -> Result<Option<f64>> {
        self.check_state(f)?;
        self.check_state(b)?;
        if f.suffix.is_some() || b.suffix.is_none() {
            return Err(invalid("coupled join direction mismatch"));
        }
        if self.plugins.iter().any(|p| p.0.has_costs()) || self.has_slope_fronts() {
            let Some(total) = self.join_original_objective(f, b)? else {
                return Ok(None);
            };
            return Ok(Some(finite(
                finite(total - f.resource_objective)? - b.resource_objective,
            )?));
        }
        if f.selection
            .iter()
            .zip(&b.selection)
            .any(|(a, b)| a & b == 0)
        {
            return Ok(None);
        }
        for ((&left, &right), &capacity) in f.loads.iter().zip(&b.loads).zip(&self.load_capacities)
        {
            let load = combine_load(left, right)?;
            if finite(load[0] + load[2])? > capacity {
                return Ok(None);
            }
        }
        if !self.budgets_join(f, b)? {
            return Ok(None);
        }
        let mut front_correction = 0.;
        for (left, right) in f.fronts.iter().zip(&b.fronts) {
            let Some(joined) = left.joined_minimum(right)? else {
                return Ok(None);
            };
            front_correction = finite(
                front_correction + finite(finite(joined - left.minimum())? - right.minimum())?,
            )?;
        }
        let suffix = b.suffix.as_ref().unwrap();
        for (front, plugin) in f.plugins.iter().zip(&self.plugins) {
            if !plugin.0.join(front, &mut suffix.program.iter())? {
                return Ok(None);
            }
        }
        let mut state_cost = 0.;
        if self.transitions.deferred {
            let mut numeric = f.numeric.clone();

            for arc in suffix.program.iter() {
                for operation in &self.transitions.arcs[arc].operations {
                    let scale =
                        if matches!(operation, transitions::NumericOperation::StateCost { .. }) {
                            self.cost_scale
                        } else {
                            0.
                        };
                    if !operation.apply(&mut numeric, scale, &mut state_cost)?
                        || !operation.feasible(&numeric)
                    {
                        return Ok(None);
                    }
                }
            }
        }
        for bound in &suffix.bounds {
            let value = finite(bound.offset + dot(&bound.coefficients, &f.numeric)?)?;
            if value < bound.bounds.lower || value > bound.bounds.upper {
                return Ok(None);
            }
        }
        let entry_cost = dot(&suffix.coefficients, &f.numeric)?;
        let baseline = dot(&suffix.coefficients, &self.initial)?;
        Ok(Some(finite(
            finite(finite(entry_cost - baseline)? + state_cost)? + front_correction,
        )?))
    }
    /// Physical suffix arcs in forward traversal order. Charged resources
    /// retain this chain even when their numeric transition is affine.
    pub(crate) fn suffix_arcs<'a>(
        &self,
        b: &'a CoupledLabelState,
    ) -> impl Iterator<Item = usize> + 'a {
        b.suffix
            .as_ref()
            .into_iter()
            .flat_map(|suffix| suffix.program.iter())
    }
    /// Return the complete joined resource objective directly. Reconstructing
    /// it from two rounded minima plus a correction loses scalar information.
    pub(super) fn join_original_objective(
        &self,
        f: &CoupledLabelState,
        b: &CoupledLabelState,
    ) -> Result<Option<f64>> {
        self.join_original_objective_using(f, b, None)
    }

    fn join_original_objective_using(
        &self,
        f: &CoupledLabelState,
        b: &CoupledLabelState,
        paired_coefficient: Option<f64>,
    ) -> Result<Option<f64>> {
        self.check_state(f)?;
        self.check_state(b)?;
        if f.suffix.is_some() || b.suffix.is_none() {
            return Err(invalid("coupled join direction mismatch"));
        }
        // A complete join participates in pricing and completion decisions.
        // Shape-compatible labels from an earlier machine configuration must
        // not be accepted merely because their retained fronts still join.
        if self.front_program_domain
            && (!self.completion_state_matches(f) || !self.completion_state_matches(b))
        {
            return Err(incomplete("paired machine join state is stale"));
        }
        if f.selection
            .iter()
            .zip(&b.selection)
            .any(|(a, b)| a & b == 0)
        {
            return Ok(None);
        }
        for ((&left, &right), &capacity) in f.loads.iter().zip(&b.loads).zip(&self.load_capacities)
        {
            let load = combine_load(left, right)?;
            if finite(load[0] + load[2])? > capacity {
                return Ok(None);
            }
        }
        if !self.budgets_join(f, b)? {
            return Ok(None);
        }
        // The explicitly selected machine-front domain uses the reference
        // paired-label scan for feasibility. Cost modes still need their
        // source-objective mapping. An unavailable machine scan cannot be
        // replaced by a different feasibility decision.
        if self.front_program_domain {
            for (index, prepared) in self.slope_paired.iter().enumerate() {
                let mode = prepared.machine_mode();
                if !prepared.has_machine_program_input() {
                    continue;
                }
                match self.paired_machine_join_checked(
                    index,
                    f,
                    b,
                    descriptor::FrontierBudget::default(),
                ) {
                    descriptor::FrontierOutcome::Complete(selection)
                        if self.machine_paired_join_complete() =>
                    {
                        return if mode == 4 {
                            Ok(Some(0.))
                        } else {
                            selection
                                .dependent
                                .ok_or_else(|| incomplete("paired join cost unavailable"))?;
                            let paired = match paired_coefficient {
                                Some(coefficient) => finite(
                                    self.cost_scale
                                        * finite(coefficient * selection.accumulate(0.))?,
                                )?,
                                None => prepared
                                    .final_objective(selection.accumulate(0.), self.cost_scale)?,
                            };
                            Ok(Some(paired))
                        };
                    }
                    descriptor::FrontierOutcome::Complete(_) => {
                        return Err(incomplete(
                            "paired join with additional charged resources is unavailable",
                        ));
                    }
                    descriptor::FrontierOutcome::Empty => return Ok(None),
                    descriptor::FrontierOutcome::Unknown(reason) => {
                        return Err(incomplete(format!(
                            "paired machine concatenation unavailable: {reason:?}"
                        )))
                    }
                }
            }
        }
        // Numeric-only canonical joins retain one state vector and literal
        // cost accumulator. Rebuilding a boxed physical label per suffix arc
        // would only copy already-checked selection and load summaries.
        if self.front_indices.is_empty() && !self.has_slope_fronts() && self.plugins.is_empty() {
            let mut numeric = f.numeric.clone();
            let mut cost = f.numeric_cost;
            for arc in self.suffix_arcs(b) {
                let prepared = &self.transitions.arcs[arc];
                if prepared.impossible {
                    return Ok(None);
                }
                for operation in &prepared.operations {
                    if !operation.apply(&mut numeric, self.cost_scale, &mut cost)?
                        || !operation.feasible(&numeric)
                    {
                        return Ok(None);
                    }
                }
            }
            return Ok(Some(cost));
        }
        let mut workspace = CoupledWorkspace::default();
        let mut current: Option<Box<CoupledLabelState>> = None;
        for arc in self.suffix_arcs(b) {
            let input = current.as_deref().unwrap_or(f);
            let Some((next, _)) =
                self.extend_internal(input, arc, true, false, false, &mut workspace)?
            else {
                return Ok(None);
            };
            current = Some(next);
        }
        Ok(Some(current.as_deref().unwrap_or(f).resource_objective))
    }
}
