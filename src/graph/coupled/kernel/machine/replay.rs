use super::*;

impl<const D: usize> Graph<D> {
    /// Replay an explicit prepared-machine paired resource from a caller-
    /// supplied machine root. `path` contains physical arc-vector indices,
    /// starts at the graph source and may stop at any vertex. The `mode` is
    /// the reference resource mode integer, supplied explicitly by the caller.
    ///
    /// This bounded diagnostic does not prove source feasibility, validate a
    /// route witness, or affect pricing. Missing sidecar data, stale physical
    /// topology, noncontiguous paths and unsupported machine stages return
    /// unknown. Changing the graph requires reinstalling the sidecar.
    pub fn replay_paired_machine_path(
        &self,
        resource: usize,
        root: &descriptor::MachineRootState,
        path: &[usize],
        mode: i32,
        max_states: usize,
        budget: descriptor::FrontierBudget,
    ) -> descriptor::FrontierOutcome<descriptor::MachinePathHistory> {
        let Some(kernel) = self.coupled.as_ref() else {
            return descriptor::FrontierOutcome::Unknown(descriptor::FrontierUnknown::Unsupported(
                "no coupled resources",
            ));
        };
        if !matches!(
            kernel.definitions.get(resource),
            Some(CoupledResource::PairedFront { .. })
        ) {
            return descriptor::FrontierOutcome::Unknown(descriptor::FrontierUnknown::Unsupported(
                "machine replay requires a paired slope resource",
            ));
        }
        let index = kernel.definitions[..resource]
            .iter()
            .filter(|r| matches!(r, CoupledResource::PairedFront { .. }))
            .count();
        kernel.slope_paired[index].machine_replay(self, root, path, mode, max_states, budget)
    }

    /// Replay and bind a diagnostic history to the exact installed machine
    /// inputs. A rejected machine transition returns `Empty`. This does not
    /// validate the supplied root or certify a route.
    pub fn replay_bound_paired_machine_path(
        &self,
        resource: usize,
        root: &descriptor::MachineRootState,
        path: &[usize],
        mode: i32,
        max_states: usize,
        budget: descriptor::FrontierBudget,
    ) -> descriptor::FrontierOutcome<BoundMachinePathHistory> {
        use descriptor::{FrontierBudget, FrontierOutcome, FrontierUnknown};
        let started = std::time::Instant::now();
        if budget.max_elapsed.is_some_and(|limit| limit.is_zero()) {
            return FrontierOutcome::Unknown(FrontierUnknown::Deadline);
        }
        let Some(kernel) = self.coupled.as_ref() else {
            return FrontierOutcome::Unknown(FrontierUnknown::Unsupported("no coupled resources"));
        };
        if !matches!(
            kernel.definitions.get(resource),
            Some(CoupledResource::PairedFront { .. })
        ) {
            return FrontierOutcome::Unknown(FrontierUnknown::Unsupported(
                "machine replay requires a paired slope resource",
            ));
        }
        let Some(wrapper_work) = resource.checked_add(path.len()) else {
            return FrontierOutcome::Unknown(FrontierUnknown::WorkLimit);
        };
        let Some(remaining_work) = budget.max_operations.checked_sub(wrapper_work) else {
            return FrontierOutcome::Unknown(FrontierUnknown::WorkLimit);
        };
        let mut index = 0;
        for definition in &kernel.definitions[..resource] {
            if budget
                .max_elapsed
                .is_some_and(|limit| started.elapsed() >= limit)
            {
                return FrontierOutcome::Unknown(FrontierUnknown::Deadline);
            }
            if matches!(definition, CoupledResource::PairedFront { .. }) {
                index += 1;
            }
        }
        let Some(inputs) = kernel.slope_paired[index].machine_input_handle() else {
            return FrontierOutcome::Unknown(FrontierUnknown::Unsupported(
                "paired machine inputs unavailable",
            ));
        };
        let remaining_time = budget
            .max_elapsed
            .map(|limit| limit.saturating_sub(started.elapsed()));
        if remaining_time.is_some_and(|remaining| remaining.is_zero()) {
            return FrontierOutcome::Unknown(FrontierUnknown::Deadline);
        }
        let history = match kernel.slope_paired[index].machine_replay(
            self,
            root,
            path,
            mode,
            max_states,
            FrontierBudget {
                max_operations: remaining_work,
                max_elapsed: remaining_time,
                ..budget
            },
        ) {
            FrontierOutcome::Complete(value) => value,
            FrontierOutcome::Empty => return FrontierOutcome::Empty,
            FrontierOutcome::Unknown(reason) => return FrontierOutcome::Unknown(reason),
        };
        if budget
            .max_elapsed
            .is_some_and(|limit| started.elapsed() >= limit)
        {
            return FrontierOutcome::Unknown(FrontierUnknown::Deadline);
        }
        let path = path.to_vec();
        if budget
            .max_elapsed
            .is_some_and(|limit| started.elapsed() >= limit)
        {
            return FrontierOutcome::Unknown(FrontierUnknown::Deadline);
        }
        FrontierOutcome::Complete(BoundMachinePathHistory {
            history,
            resource,
            path,
            inputs,
            identity: kernel.machine_identity.clone(),
        })
    }

    /// Backtrack a sidecar-bound history using its installed inverse vectors.
    /// `root_is_entry_zero` explicitly selects the unproved correspondence
    /// between this history's root and reference path entry zero. This is a
    /// diagnostic, not a route, objective, or source-feasibility certificate.
    pub fn backtrack_bound_paired_machine_path(
        &self,
        history: &BoundMachinePathHistory,
        terminal: f64,
        root_is_entry_zero: bool,
        budget: descriptor::FrontierBudget,
    ) -> descriptor::FrontierOutcome<Vec<f64>> {
        use descriptor::{FrontierBudget, FrontierOutcome, FrontierUnknown};
        let started = std::time::Instant::now();
        if budget.max_elapsed.is_some_and(|limit| limit.is_zero()) {
            return FrontierOutcome::Unknown(FrontierUnknown::Deadline);
        }
        let Some(kernel) = self.coupled.as_ref() else {
            return FrontierOutcome::Unknown(FrontierUnknown::Unsupported("no coupled resources"));
        };
        if !std::sync::Arc::ptr_eq(&kernel.machine_identity, &history.identity)
            || !matches!(
                kernel.definitions.get(history.resource),
                Some(CoupledResource::PairedFront { .. })
            )
        {
            return FrontierOutcome::Unknown(FrontierUnknown::Unsupported(
                "paired machine program identity changed",
            ));
        }
        let Some(remaining_work) = budget.max_operations.checked_sub(history.resource) else {
            return FrontierOutcome::Unknown(FrontierUnknown::WorkLimit);
        };
        let mut index = 0;
        for definition in &kernel.definitions[..history.resource] {
            if budget
                .max_elapsed
                .is_some_and(|limit| started.elapsed() >= limit)
            {
                return FrontierOutcome::Unknown(FrontierUnknown::Deadline);
            }
            if matches!(definition, CoupledResource::PairedFront { .. }) {
                index += 1;
            }
        }
        let remaining_time = budget
            .max_elapsed
            .map(|limit| limit.saturating_sub(started.elapsed()));
        if remaining_time.is_some_and(|remaining| remaining.is_zero()) {
            return FrontierOutcome::Unknown(FrontierUnknown::Deadline);
        }
        let result = kernel.slope_paired[index].machine_backtrack_bound(
            self,
            history,
            terminal,
            root_is_entry_zero,
            FrontierBudget {
                max_operations: remaining_work,
                max_elapsed: remaining_time,
                ..budget
            },
        );
        if budget
            .max_elapsed
            .is_some_and(|limit| started.elapsed() >= limit)
        {
            return FrontierOutcome::Unknown(FrontierUnknown::Deadline);
        }
        result
    }
    /// Reconstruct a retained backward label's machine suffix from explicitly
    /// supplied joined values. The installed terminal root and arc operands
    /// must still match the history. This returns machine values, not a source
    /// feasibility certificate or a normalized route objective.
    pub fn reconstruct_paired_machine_suffix(
        &self,
        resource: usize,
        backward: &CoupledLabelState,
        initial_base: f64,
        initial_dependent: f64,
        budget: descriptor::FrontierBudget,
    ) -> descriptor::FrontierOutcome<descriptor::MachinePathReconstruction> {
        use descriptor::{FrontierBudget, FrontierOutcome, FrontierUnknown, PairedFrontOperations};
        let started = std::time::Instant::now();
        if budget.max_elapsed.is_some_and(|limit| limit.is_zero()) {
            return FrontierOutcome::Unknown(FrontierUnknown::Deadline);
        }
        let Some(kernel) = self.coupled.as_ref() else {
            return FrontierOutcome::Unknown(FrontierUnknown::Unsupported("no coupled resources"));
        };
        if kernel.check_state(backward).is_err()
            || !matches!(
                kernel.definitions.get(resource),
                Some(CoupledResource::PairedFront { .. })
            )
        {
            return FrontierOutcome::Unknown(FrontierUnknown::Unsupported(
                "machine suffix resource or state",
            ));
        }
        let Some(suffix) = backward.suffix.as_ref() else {
            return FrontierOutcome::Unknown(FrontierUnknown::Unsupported(
                "machine suffix requires a backward label",
            ));
        };
        let index = kernel.definitions[..resource]
            .iter()
            .filter(|definition| matches!(definition, CoupledResource::PairedFront { .. }))
            .count();
        let prepared = &kernel.slope_paired[index];
        let Some(history) = backward.slope_relations[index].machine_history.as_ref() else {
            return FrontierOutcome::Unknown(FrontierUnknown::Unsupported(
                "machine suffix history unavailable",
            ));
        };
        let validation_work = resource
            .saturating_add(self.arcs.len())
            .saturating_add(suffix.program.len().saturating_mul(4))
            .saturating_add(history.len())
            .saturating_add(1);
        let Some(remaining_work) = budget.max_operations.checked_sub(validation_work) else {
            return FrontierOutcome::Unknown(FrontierUnknown::WorkLimit);
        };
        let path: Vec<_> = suffix.program.iter().collect();
        if !prepared.matches_backward_root(history) {
            return FrontierOutcome::Unknown(FrontierUnknown::Unsupported(
                "machine suffix terminal root changed",
            ));
        }
        let (Some(inputs), Some(topology)) =
            (prepared.machine_input_handle(), prepared.machine_topology())
        else {
            return FrontierOutcome::Unknown(FrontierUnknown::Unsupported(
                "machine suffix operands unavailable",
            ));
        };
        if self.source != topology.source
            || self.sink != topology.sink
            || self.vertices.len() != topology.vertex_count
            || self.arcs.len() != topology.arcs.len()
            || self.arcs.len() != inputs.len()
            || history.len().checked_sub(1) != Some(path.len())
        {
            return FrontierOutcome::Unknown(FrontierUnknown::Unsupported(
                "machine suffix topology changed",
            ));
        }
        for (arc, &(id, tail, head)) in self.arcs.iter().zip(topology.arcs.iter()) {
            if budget
                .max_elapsed
                .is_some_and(|limit| started.elapsed() >= limit)
            {
                return FrontierOutcome::Unknown(FrontierUnknown::Deadline);
            }
            if (arc.id, arc.tail, arc.head) != (id, tail, head)
                || tail >= self.vertices.len()
                || head >= self.vertices.len()
            {
                return FrontierOutcome::Unknown(FrontierUnknown::Unsupported(
                    "machine suffix topology changed",
                ));
            }
        }
        let mut vertex = path
            .first()
            .and_then(|&index| self.arcs.get(index))
            .map_or(self.sink, |arc| arc.tail);
        let mut steps = Vec::with_capacity(path.len());
        for &arc in &path {
            if budget
                .max_elapsed
                .is_some_and(|limit| started.elapsed() >= limit)
            {
                return FrontierOutcome::Unknown(FrontierUnknown::Deadline);
            }
            let Some(physical) = self.arcs.get(arc).filter(|arc| arc.tail == vertex) else {
                return FrontierOutcome::Unknown(FrontierUnknown::Unsupported(
                    "machine suffix path continuity",
                ));
            };
            let Some(input) = inputs[arc].backward_inputs() else {
                return FrontierOutcome::Unknown(FrontierUnknown::Unsupported(
                    "machine suffix backward operands unavailable",
                ));
            };
            vertex = physical.head;
            steps.push((arc, input));
        }
        if vertex != self.sink {
            return FrontierOutcome::Unknown(FrontierUnknown::Unsupported(
                "machine suffix does not reach sink",
            ));
        }
        let remaining_time = budget
            .max_elapsed
            .map(|limit| limit.saturating_sub(started.elapsed()));
        if remaining_time.is_some_and(|limit| limit.is_zero()) {
            return FrontierOutcome::Unknown(FrontierUnknown::Deadline);
        }
        let waiting_rates = prepared
            .machine_waiting_rates()
            .map(|rates| path.iter().map(|&arc| rates[arc]).collect::<Vec<_>>());
        PairedFrontOperations::machine_reconstruct_backward_path_with_rates(
            history,
            &steps,
            initial_base,
            initial_dependent,
            prepared.machine_mode(),
            waiting_rates.as_deref(),
            FrontierBudget {
                max_operations: remaining_work,
                max_elapsed: remaining_time,
                ..budget
            },
        )
    }
}
