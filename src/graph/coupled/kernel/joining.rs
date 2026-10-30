//! Forward/backward label joins and joined resource objectives.
use super::*;
use crate::incomplete;

impl CoupledKernel {
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
            if !matches!(resource, CoupledResource::SlopePaired { .. }) {
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
            return Err(invalid("custom machine recorder requires one custom resource"));
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
        let index = self.custom_indices.iter().position(|&entry| entry == resource)
            .ok_or_else(|| invalid("custom machine resource unavailable"))?;
        let (left, right) = (&forward.custom_machine[index], &backward.custom_machine[index]);
        if !left.accepted || !right.accepted {
            return Ok(None);
        }
        let Some(CoupledResource::CustomResource { increments, weights, .. }) = self.definitions.get(resource) else {
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
        let same_state = |a: &crate::graph::coupled::CustomMachineState, b: &crate::graph::coupled::CustomMachineState| {
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
            if backward { arcs.iter().rev().copied().for_each(&mut step); }
            else { arcs.iter().copied().for_each(&mut step); }
            total
        };
        let forward_total = replay(false);
        let backward_total = replay(true);
        if (forward_total - backward_total).abs() > 1e-6 {
            return Ok(None);
        }
        let joined = (left.state.concatenate(&right.state) + left.wrapper_cost) + right.wrapper_cost;
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
        let index = self.custom_indices.iter().position(|&entry| entry == resource)
            .ok_or_else(|| invalid("custom machine resource unavailable"))?;
        let (left, right) = (&forward.custom_machine[index], &backward.custom_machine[index]);
        if !left.accepted || !right.accepted {
            return Ok(None);
        }
        let cross = left.state.concatenate(&right.state);
        Ok(crate::graph::coupled::CustomMachineState::contribution_accepted(cross)
            .then(|| ((cross + left.wrapper_cost) + right.wrapper_cost) + previous))
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
        let index = self.custom_indices.iter().position(|&entry| entry == resource)
            .ok_or_else(|| invalid("custom machine resource unavailable"))?;
        let (left, right) = (&first.custom_machine[index], &second.custom_machine[index]);
        if !left.accepted || !right.accepted {
            return Ok(None);
        }
        Ok(Some(((0.0 + left.wrapper_cost) + right.wrapper_cost) + previous))
    }

    /// Whether retained machine contributions are the complete resource
    /// charge for inspection: one feasibility-only paired program contributes
    /// zero and has no pricing-only activity adjustments.
    pub(crate) fn uses_zero_paired_inspection_cost(&self) -> bool {
        self.front_program_domain
            && self.definitions.len() == 1
            && self.slope_paired.len() == 1
            && self.slope_paired[0].machine_mode() == 4
            && self.slope_paired[0].has_machine_program_input()
            && self.pricing_activities.is_empty()
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
                CoupledResource::SlopePaired { .. } if self.front_program_domain => charged += 1,
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
        let validation = self.definitions.len().saturating_mul(2).saturating_add(resource)
            .saturating_add(forward.suffix.as_ref().map_or(0, |suffix| suffix.bounds.len()))
            .saturating_add(backward.suffix.as_ref().map_or(0, |suffix| suffix.bounds.len()))
            .saturating_add(forward.reversal_peaks.as_ref().map_or(0, |peaks| peaks.len()))
            .saturating_add(backward.reversal_peaks.as_ref().map_or(0, |peaks| peaks.len()));
        let Some(remaining_work) = budget.max_operations.checked_sub(validation) else {
            return FrontierOutcome::Unknown(FrontierUnknown::WorkLimit);
        };
        if !matches!(self.definitions.get(resource), Some(CoupledResource::SlopePaired { .. }))
            || self.check_state(forward).is_err()
            || self.check_state(backward).is_err()
            || forward.suffix.is_some()
            || backward.suffix.is_none()
        {
            return FrontierOutcome::Unknown(FrontierUnknown::Unsupported("paired machine join resource or direction"));
        }
        let index = self.definitions[..resource].iter()
            .filter(|definition| matches!(definition, CoupledResource::SlopePaired { .. })).count();
        let remaining_time = budget.max_elapsed.map(|limit| limit.saturating_sub(started.elapsed()));
        if remaining_time.is_some_and(|time| time.is_zero()) {
            return FrontierOutcome::Unknown(FrontierUnknown::Deadline);
        }
        self.paired_machine_join_checked(index, forward, backward, descriptor::FrontierBudget {
            max_operations: remaining_work,
            max_elapsed: remaining_time,
            ..budget
        })
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
            return FrontierOutcome::Unknown(FrontierUnknown::Unsupported("paired machine join histories unavailable"));
        };
        if !prepared.matches_forward_root(left) || !prepared.matches_backward_root(right) {
            return FrontierOutcome::Unknown(FrontierUnknown::Unsupported("paired machine join histories are stale"));
        }
        let ([left], [right_front], Some(ceiling)) = (
            left.last().front().components(),
            right.last().front().components(),
            right.backward_ceiling(),
        ) else {
            return FrontierOutcome::Unknown(FrontierUnknown::Unsupported("paired machine join front or ceiling unavailable"));
        };
        descriptor::PreparedSlopeRelationArc::machine_concatenation(
            left, right_front, ceiling, prepared.machine_mode(), budget,
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
            return self.custom_machine_resource_join(self.custom_indices[0], forward, backward, 0.).ok().flatten();
        }
        if !self.front_program_domain || self.slope_paired.len() != 1 {
            return None;
        }
        match self.paired_machine_join_checked(0, forward, backward, descriptor::FrontierBudget::default()) {
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
        if self.plugins.iter().any(|p| p.0.has_costs())
            || self.has_slope_fronts()
            || self.paired.iter().any(|prepared| prepared.has_costs())
        {
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
        let mut workspace = CoupledWorkspace::default();
        for ((left, right), prepared) in f.relations.iter().zip(&b.relations).zip(&self.paired) {
            if !paired::domains_may_intersect(left, right, self.front_bucket_policy) {
                return Ok(None);
            }
            let scale = finite(self.cost_scale * prepared.objective_coefficient)?;
            let cached = if scale >= 0. {
                prepared.cached_join(left, right)
            } else {
                None
            };
            match cached {
                Some(Some(value)) => value,
                Some(None) => return Ok(None),
                None => {
                    let Some(value) = prepared.replay_suffix(
                        left,
                        &b.suffix.as_ref().unwrap().program,
                        &mut workspace.relation,
                        scale < 0.,
                    )?
                    else {
                        return Ok(None);
                    };
                    value
                }
            };
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
        self.check_state(f)?;
        self.check_state(b)?;
        if f.suffix.is_some() || b.suffix.is_none() {
            return Err(invalid("coupled join direction mismatch"));
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
                match self.paired_machine_join_checked(index, f, b, descriptor::FrontierBudget::default()) {
                    descriptor::FrontierOutcome::Complete(_)
                        if mode == 4 && self.definitions.len() == 1 => return Ok(Some(0.)),
                    descriptor::FrontierOutcome::Complete(_) => {}
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
        if self.front_indices.is_empty()
            && !self.has_slope_fronts()
            && self.relation_indices.is_empty()
            && self.plugins.is_empty()
        {
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
        // In the restricted exact bounded-cost specialization the cached
        // final dependent minimum is already canonical; avoid suffix replay.
        if self.front_indices.is_empty()
            && self.slope_indices.is_empty()
            && self.initial.is_empty()
            && (!self.paired.is_empty() || !self.slope_paired.is_empty())
            && self.plugins.is_empty()
            && !self.has_negative_paired_objectives()
        {
            let mut total = 0.;
            let mut known = true;
            // Match final_resource_objective's family order. The cache owns
            // its exact dependent-unit conversion; raw machine programs and
            // nonconstant source prefixes retain full suffix replay.
            for ((left, right), prepared) in f
                .slope_relations
                .iter()
                .zip(&b.slope_relations)
                .zip(&self.slope_paired)
            {
                match prepared.cached_join(left, right) {
                    Some(Some(final_minimum)) => {
                        total = finite(
                            total + prepared.final_objective(final_minimum, self.cost_scale)?,
                        )?;
                    }
                    Some(None) => return Ok(None),
                    None => {
                        known = false;
                        break;
                    }
                }
            }
            for ((left, right), prepared) in f.relations.iter().zip(&b.relations).zip(&self.paired)
            {
                if !known {
                    break;
                }
                match prepared.cached_join(left, right) {
                    Some(Some(final_minimum)) => {
                        total = finite(
                            total + prepared.final_objective(final_minimum, self.cost_scale)?,
                        )?
                    }
                    Some(None) => return Ok(None),
                    None => {
                        known = false;
                        break;
                    }
                }
            }
            if known {
                return Ok(Some(total));
            }
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
