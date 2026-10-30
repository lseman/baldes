//! Recovered paired recorder wired to the installed machine labels/operands.
use super::*;
use descriptor::{
    FrontierBudget, FrontierOutcome, FrontierUnknown, MachineJoinedPathReconstruction,
    MachinePathSnapshot, PreparedSlopeRelationArc,
};
use std::time::Instant;

impl<const D: usize> Graph<D> {
    /// Select a paired join only when the saved histories are the requested
    /// physical route. This checks the handoff without running the recorder.
    pub(crate) fn paired_machine_selection_for_path(
        &self,
        resource: usize,
        forward: &CoupledLabelState,
        backward: &CoupledLabelState,
        arcs: &[usize],
        budget: FrontierBudget,
    ) -> FrontierOutcome<descriptor::MachineConcatenation> {
        let started = Instant::now();
        let Some(kernel) = self.coupled.as_ref() else {
            return FrontierOutcome::Unknown(FrontierUnknown::Unsupported("machine join has no coupled resources"));
        };
        if !matches!(kernel.definitions.get(resource), Some(CoupledResource::SlopePaired { .. }))
            || kernel.check_state(forward).is_err() || kernel.check_state(backward).is_err()
            || forward.suffix.is_some() || backward.suffix.is_none()
        {
            return FrontierOutcome::Unknown(FrontierUnknown::Unsupported("machine join resource or label direction"));
        }
        let index = kernel.definitions[..resource].iter()
            .filter(|definition| matches!(definition, CoupledResource::SlopePaired { .. })).count();
        let Some(history) = forward.slope_relations[index].machine_history.as_ref() else {
            return FrontierOutcome::Unknown(FrontierUnknown::Unsupported("machine join forward history unavailable"));
        };
        let Some(backward_history) = backward.slope_relations[index].machine_history.as_ref() else {
            return FrontierOutcome::Unknown(FrontierUnknown::Unsupported("machine join backward history unavailable"));
        };
        let suffix = &backward.suffix.as_ref().unwrap().program;
        if backward_history.len().checked_sub(1) != Some(suffix.len()) {
            return FrontierOutcome::Unknown(FrontierUnknown::Unsupported("machine join backward history shape"));
        }
        let validation = resource.saturating_add(kernel.definitions.len().saturating_mul(2))
            .saturating_add(history.len().saturating_mul(2))
            .saturating_add(backward_history.len().saturating_mul(2))
            .saturating_add(suffix.len()).saturating_add(arcs.len());
        let Some(work) = budget.max_operations.checked_sub(validation) else {
            return FrontierOutcome::Unknown(FrontierUnknown::WorkLimit);
        };
        let remaining = budget.max_elapsed.map(|limit| limit.saturating_sub(started.elapsed()));
        if remaining.is_some_and(|time| time.is_zero()) {
            return FrontierOutcome::Unknown(FrontierUnknown::Deadline);
        }
        let snapshots = match history.snapshots(FrontierBudget {
            max_operations: history.len(), max_elapsed: remaining, ..budget
        }) {
            FrontierOutcome::Complete(snapshots) => snapshots,
            FrontierOutcome::Empty => return FrontierOutcome::Unknown(FrontierUnknown::Unsupported("machine join forward history empty")),
            FrontierOutcome::Unknown(reason) => return FrontierOutcome::Unknown(reason),
        };
        let remaining = budget.max_elapsed.map(|limit| limit.saturating_sub(started.elapsed()));
        if remaining.is_some_and(|time| time.is_zero()) {
            return FrontierOutcome::Unknown(FrontierUnknown::Deadline);
        }
        let backward_snapshots = match backward_history.snapshots(FrontierBudget {
            max_operations: backward_history.len(), max_elapsed: remaining, ..budget
        }) {
            FrontierOutcome::Complete(snapshots) => snapshots,
            FrontierOutcome::Empty => return FrontierOutcome::Unknown(FrontierUnknown::Unsupported("machine join backward history empty")),
            FrontierOutcome::Unknown(reason) => return FrontierOutcome::Unknown(reason),
        };
        if !matches!(snapshots.first(), Some(MachinePathSnapshot::Root(_)))
            || !matches!(backward_snapshots.first(), Some(MachinePathSnapshot::Root(_)))
            || !backward_snapshots.iter().skip(1).rev().zip(suffix.iter()).all(|(snapshot, arc)| {
                matches!(snapshot, MachinePathSnapshot::Backward { arc: saved, .. } if *saved == arc)
            })
        {
            return FrontierOutcome::Unknown(FrontierUnknown::Unsupported("machine join backward history path mismatch"));
        }
        let mut path = Vec::with_capacity(arcs.len());
        for snapshot in snapshots.iter().skip(1) {
            let MachinePathSnapshot::Forward { arc, .. } = snapshot else {
                return FrontierOutcome::Unknown(FrontierUnknown::Unsupported("machine join forward history direction"));
            };
            path.push(*arc);
        }
        path.extend(suffix.iter());
        if path != arcs {
            return FrontierOutcome::Unknown(FrontierUnknown::Unsupported("machine join path does not match retained route"));
        }
        let remaining = budget.max_elapsed.map(|limit| limit.saturating_sub(started.elapsed()));
        if remaining.is_some_and(|time| time.is_zero()) {
            return FrontierOutcome::Unknown(FrontierUnknown::Deadline);
        }
        kernel.paired_machine_resource_join(resource, forward, backward, FrontierBudget {
            max_operations: work, max_elapsed: remaining, ..budget
        })
    }

    /// Select and reconstruct a joined pair of retained machine labels.
    /// Installed roots, physical arc identities, path continuity and prepared
    /// operands must match. This returns raw machine consumptions, not a source
    /// certificate or a normalized pricing objective.
    pub fn reconstruct_paired_machine_join(
        &self,
        resource: usize,
        forward: &CoupledLabelState,
        backward: &CoupledLabelState,
        budget: FrontierBudget,
    ) -> FrontierOutcome<MachineJoinedPathReconstruction> {
        self.reconstruct_paired_machine_join_checked(resource, forward, backward, None, 0, budget)
    }

    /// Reconstruct using the specified native recorder diagnostic level.
    pub fn reconstruct_paired_machine_join_with_debug(
        &self,
        resource: usize,
        forward: &CoupledLabelState,
        backward: &CoupledLabelState,
        debug_level: i32,
        budget: FrontierBudget,
    ) -> FrontierOutcome<MachineJoinedPathReconstruction> {
        self.reconstruct_paired_machine_join_checked(resource, forward, backward, None, debug_level, budget)
    }

    pub(crate) fn reconstruct_paired_machine_join_for_path(
        &self,
        resource: usize,
        forward: &CoupledLabelState,
        backward: &CoupledLabelState,
        arcs: &[usize],
        debug_level: i32,
        budget: FrontierBudget,
    ) -> FrontierOutcome<MachineJoinedPathReconstruction> {
        self.reconstruct_paired_machine_join_checked(resource, forward, backward, Some(arcs), debug_level, budget)
    }

    fn reconstruct_paired_machine_join_checked(
        &self,
        resource: usize,
        forward: &CoupledLabelState,
        backward: &CoupledLabelState,
        expected_arcs: Option<&[usize]>,
        debug_level: i32,
        budget: FrontierBudget,
    ) -> FrontierOutcome<MachineJoinedPathReconstruction> {
        let started = Instant::now();
        let compute =
            || -> std::result::Result<MachineJoinedPathReconstruction, FrontierUnknown> {
                let remaining = |work| -> std::result::Result<FrontierBudget, FrontierUnknown> {
                    let max_elapsed = budget
                        .max_elapsed
                        .map(|limit| limit.saturating_sub(started.elapsed()));
                    if max_elapsed.is_some_and(|limit| limit.is_zero()) {
                        return Err(FrontierUnknown::Deadline);
                    }
                    Ok(FrontierBudget {
                        max_operations: work,
                        max_elapsed,
                        ..budget
                    })
                };
                remaining(0)?;
                let kernel = self.coupled.as_ref().ok_or(FrontierUnknown::Unsupported(
                    "machine join has no coupled resources",
                ))?;
                // Resource-index scanning is included before its first traversal.
                let mut work = budget
                    .max_operations
                    .checked_sub(resource)
                    .ok_or(FrontierUnknown::WorkLimit)?;
                if !matches!(
                    kernel.definitions.get(resource),
                    Some(CoupledResource::SlopePaired { .. })
                ) || kernel.check_state(forward).is_err()
                    || kernel.check_state(backward).is_err()
                    || forward.suffix.is_some()
                {
                    return Err(FrontierUnknown::Unsupported(
                        "machine join resource or label direction",
                    ));
                }
                let suffix = backward
                    .suffix
                    .as_ref()
                    .ok_or(FrontierUnknown::Unsupported(
                        "machine join requires a backward label",
                    ))?;
                let index = kernel.definitions[..resource]
                    .iter()
                    .filter(|definition| matches!(definition, CoupledResource::SlopePaired { .. }))
                    .count();
                let prepared = &kernel.slope_paired[index];
                let forward = forward.slope_relations[index]
                    .machine_history
                    .as_ref()
                    .ok_or(FrontierUnknown::Unsupported(
                        "machine join forward history unavailable",
                    ))?;
                let backward = backward.slope_relations[index]
                    .machine_history
                    .as_ref()
                    .ok_or(FrontierUnknown::Unsupported(
                        "machine join backward history unavailable",
                    ))?;
                let validation = forward
                    .len()
                    .saturating_mul(3)
                    .saturating_add(backward.len())
                    .saturating_add(suffix.program.len().saturating_mul(3))
                    .saturating_add(self.arcs.len())
                    .saturating_add(expected_arcs.map_or(0, |arcs| arcs.len()))
                    .saturating_add(1);
                work = work
                    .checked_sub(validation)
                    .ok_or(FrontierUnknown::WorkLimit)?;
                remaining(work)?;
                if !prepared.matches_forward_root(forward)
                    || !prepared.matches_backward_root(backward)
                {
                    return Err(FrontierUnknown::Unsupported(
                        "machine join installed root changed",
                    ));
                }
                let inputs =
                    prepared
                        .machine_input_handle()
                        .ok_or(FrontierUnknown::Unsupported(
                            "machine join operands unavailable",
                        ))?;
                let topology = prepared
                    .machine_topology()
                    .ok_or(FrontierUnknown::Unsupported(
                        "machine join topology unavailable",
                    ))?;
                if self.source != topology.source
                    || self.sink != topology.sink
                    || self.source >= self.vertices.len()
                    || self.sink >= self.vertices.len()
                    || self.vertices.len() != topology.vertex_count
                    || self.arcs.len() != topology.arcs.len()
                    || self.arcs.len() != inputs.len()
                    || backward.len().checked_sub(1) != Some(suffix.program.len())
                {
                    return Err(FrontierUnknown::Unsupported(
                        "machine join topology or suffix changed",
                    ));
                }
                for (arc, &(id, tail, head)) in self.arcs.iter().zip(topology.arcs.iter()) {
                    remaining(work)?;
                    if (arc.id, arc.tail, arc.head) != (id, tail, head)
                        || tail >= self.vertices.len()
                        || head >= self.vertices.len()
                    {
                        return Err(FrontierUnknown::Unsupported(
                            "machine join physical topology changed",
                        ));
                    }
                }
                let snapshots = match forward.snapshots(remaining(forward.len())?) {
                    FrontierOutcome::Complete(value) => value,
                    FrontierOutcome::Empty => {
                        return Err(FrontierUnknown::Unsupported(
                            "machine join forward history empty",
                        ))
                    }
                    FrontierOutcome::Unknown(reason) => return Err(reason),
                };
                if !matches!(snapshots.first(), Some(MachinePathSnapshot::Root(_))) {
                    return Err(FrontierUnknown::Unsupported(
                        "machine join forward root unavailable",
                    ));
                }
                let mut vertex = self.source;
                let mut inverses = Vec::with_capacity(snapshots.len().saturating_sub(1));
                for snapshot in &snapshots[1..] {
                    remaining(work)?;
                    let MachinePathSnapshot::Forward { arc, .. } = snapshot else {
                        return Err(FrontierUnknown::Unsupported(
                            "machine join forward history direction",
                        ));
                    };
                    let physical = self.arcs.get(*arc).filter(|arc| arc.tail == vertex).ok_or(
                        FrontierUnknown::Unsupported("machine join prefix continuity"),
                    )?;
                    let input = inputs
                        .get(*arc)
                        .ok_or(FrontierUnknown::Unsupported("machine join prefix operands"))?;
                    vertex = physical.head;
                    inverses.push((*arc, &input.inverse));
                }
                let mut steps = Vec::with_capacity(suffix.program.len());
                for arc in suffix.program.iter() {
                    remaining(work)?;
                    let physical = self.arcs.get(arc).filter(|arc| arc.tail == vertex).ok_or(
                        FrontierUnknown::Unsupported("machine join suffix continuity"),
                    )?;
                    let input = inputs
                        .get(arc)
                        .and_then(PreparedMachineArc::backward_inputs)
                        .ok_or(FrontierUnknown::Unsupported("machine join suffix operands"))?;
                    vertex = physical.head;
                    steps.push((arc, input));
                }
                if vertex != self.sink {
                    return Err(FrontierUnknown::Unsupported(
                        "machine join path does not reach sink",
                    ));
                }
                if expected_arcs.is_some_and(|arcs| {
                    !arcs.iter().copied().eq(inverses.iter().map(|(arc, _)| *arc).chain(steps.iter().map(|(arc, _)| *arc)))
                }) {
                    return Err(FrontierUnknown::Unsupported("machine join path does not match retained route"));
                }
                let waiting_rates = prepared.machine_waiting_rates().map(|rates| {
                    steps.iter().map(|(arc, _)| rates[*arc]).collect::<Vec<_>>()
                });
                let result = PreparedSlopeRelationArc::machine_reconstruct_joined_path_with_debug(
                    forward,
                    backward,
                    &inverses,
                    &steps,
                    prepared.machine_mode(),
                    waiting_rates.as_deref(),
                    debug_level,
                    remaining(work)?,
                );
                remaining(0)?;
                match result {
                    FrontierOutcome::Complete(value) => Ok(value),
                    FrontierOutcome::Empty => {
                        Err(FrontierUnknown::Unsupported("machine join recorder empty"))
                    }
                    FrontierOutcome::Unknown(reason) => Err(reason),
                }
            };
        match compute() {
            Ok(value) => FrontierOutcome::Complete(value),
            Err(reason) => FrontierOutcome::Unknown(reason),
        }
    }
}
