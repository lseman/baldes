//! Paired raw slope relations: physical dependent state and objective stay separate.
use super::*;
use descriptor::{
    FrontierBudget, FrontierOutcome, FrontierUnknown, MachinePathHistory, MachineRootState,
    PairedFrontOperations, SlopeFront,
};
use slope_interval::{outcome, SlopeLabel};
use std::sync::Arc;

mod inputs;

use inputs::machine_inputs_equal;
pub use inputs::{PreparedMachineArc, PreparedMachineBackwardInputs};

/// Physical identities captured when caller-supplied operands are installed.
/// Public graph fields can be changed independently of a prepared kernel.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct MachineTopology {
    pub source: usize,
    pub sink: usize,
    pub vertex_count: usize,
    pub arcs: Arc<[(i32, usize, usize)]>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct BaseWindows {
    pub resource: usize,
    pub vertices: Vec<(u64, u64)>,
    pub arcs: Vec<Option<(u64, u64)>>,
}

#[derive(Clone, Debug)]
pub(super) struct PreparedSlopePaired {
    pub(super) objective_coefficient: f64,
    arc_count: usize,
    waiting_windows: Option<std::sync::Arc<[Window]>>,
    machine_inputs: Option<Arc<[PreparedMachineArc]>>,
    machine_waiting_rates: Option<Arc<[f64]>>,
    machine_root_input: Option<PreparedSlopeFunction>,
    machine_root_history: Option<Arc<MachinePathHistory>>,
    machine_backward_root: Option<Arc<MachinePathHistory>>,
    machine_topology: Option<MachineTopology>,
    base_windows: Option<BaseWindows>,
    mode: PairedResourceMode,
}
impl PartialEq for PreparedSlopePaired {
    fn eq(&self, b: &Self) -> bool {
        self.objective_coefficient == b.objective_coefficient
            && self.arc_count == b.arc_count
            && waiting_windows_equal(
                self.waiting_windows.as_deref(),
                b.waiting_windows.as_deref(),
            )
            && machine_inputs_equal(self.machine_inputs.as_deref(), b.machine_inputs.as_deref())
            && match (&self.machine_waiting_rates, &b.machine_waiting_rates) {
                (None, None) => true,
                (Some(a), Some(b)) => {
                    a.len() == b.len()
                        && a.iter()
                            .zip(b.iter())
                            .all(|(x, y)| x.to_bits() == y.to_bits())
                }
                _ => false,
            }
            && match (&self.machine_backward_root, &b.machine_backward_root) {
                (None, None) => true,
                (Some(a), Some(b)) => Arc::ptr_eq(a, b),
                _ => false,
            }
            && match (&self.machine_root_input, &b.machine_root_input) {
                (Some(a), Some(b)) => {
                    a.has_machine_only_geometry() == b.has_machine_only_geometry()
                        && a.pairs().len() == b.pairs().len()
                        && a.pairs().iter().zip(b.pairs()).all(|(a, b)| {
                            a.0.to_bits() == b.0.to_bits() && a.1.to_bits() == b.1.to_bits()
                        })
                }
                (None, None) => true,
                _ => false,
            }
            && self.machine_topology == b.machine_topology
            && self.base_windows == b.base_windows
            && self.mode == b.mode
    }
}
fn waiting_windows_equal(a: Option<&[Window]>, b: Option<&[Window]>) -> bool {
    match (a, b) {
        (None, None) => true,
        (Some(a), Some(b)) => {
            a.len() == b.len()
                && a.iter().zip(b).all(|(a, b)| {
                    a.lower.to_bits() == b.lower.to_bits() && a.upper.to_bits() == b.upper.to_bits()
                })
        }
        _ => false,
    }
}
impl PreparedSlopePaired {
    pub(super) fn machine_topology_matches<const D: usize>(&self, graph: &Graph<D>) -> bool {
        match (&self.machine_inputs, &self.machine_topology) {
            (None, None) => true,
            (Some(inputs), Some(topology)) => {
                graph.source == topology.source
                    && graph.sink == topology.sink
                    && graph.vertices.len() == topology.vertex_count
                    && graph.arcs.len() == inputs.len()
                    && graph.arcs.len() == topology.arcs.len()
                    && self.base_windows.as_ref().is_none_or(|snapshot| {
                        let resource = snapshot.resource;
                        #[cfg(feature = "multi-time-windows")]
                        if graph
                            .time_windows
                            .keys()
                            .any(|&(_, index)| index == resource)
                        {
                            return false;
                        }
                        graph.vertices.len() == snapshot.vertices.len()
                            && graph.arcs.len() == snapshot.arcs.len()
                            && graph.vertices.iter().zip(&snapshot.vertices).all(
                                |(vertex, &expected)| {
                                    vertex.bounds.get(resource).is_some_and(|window| {
                                        (window.lower.to_bits(), window.upper.to_bits()) == expected
                                    })
                                },
                            )
                            && graph
                                .arcs
                                .iter()
                                .zip(&snapshot.arcs)
                                .all(|(arc, expected)| {
                                    arc.bounds.get(resource).is_some_and(|window| {
                                        window.map(|w| (w.lower.to_bits(), w.upper.to_bits()))
                                            == *expected
                                    })
                                })
                    })
                    && graph
                        .arcs
                        .iter()
                        .zip(topology.arcs.iter())
                        .all(|(arc, expected)| (arc.id, arc.tail, arc.head) == *expected)
            }
            _ => false,
        }
    }
    pub(super) fn machine_input_handle(&self) -> Option<Arc<[PreparedMachineArc]>> {
        self.machine_inputs.clone()
    }
    pub(super) fn machine_waiting_rates(&self) -> Option<&[f64]> {
        self.machine_waiting_rates.as_deref()
    }
    pub(super) fn set_machine_waiting_rates(&mut self, rates: &[f64]) -> Result<bool> {
        let inputs = self
            .machine_inputs
            .as_deref()
            .ok_or_else(|| invalid("paired machine inputs unavailable"))?;
        if rates.len() != inputs.len()
            || rates.iter().zip(inputs).any(|(rate, input)| {
                !rate.is_finite() || rate.trunc() != f64::from(input.waiting_rate)
            })
        {
            return Err(invalid(
                "paired machine waiting rates do not match transition operands",
            ));
        }
        if self.machine_waiting_rates.as_ref().is_some_and(|old| {
            old.iter()
                .zip(rates)
                .all(|(a, b)| a.to_bits() == b.to_bits())
        }) {
            return Ok(false);
        }
        self.machine_waiting_rates = Some(Arc::from(rates));
        Ok(true)
    }
    pub(super) fn machine_topology(&self) -> Option<&MachineTopology> {
        self.machine_topology.as_ref()
    }
    pub(super) fn matches_backward_root(&self, history: &MachinePathHistory) -> bool {
        self.machine_backward_root
            .as_ref()
            .is_some_and(|root| history.shares_root(root))
    }
    pub(super) fn matches_forward_root(&self, history: &MachinePathHistory) -> bool {
        self.machine_root_history
            .as_ref()
            .is_some_and(|root| history.shares_root(root))
    }
    pub(super) fn set_machine_backward_root(
        &mut self,
        root: MachineRootState,
        budget: FrontierBudget,
    ) -> FrontierOutcome<()> {
        match MachinePathHistory::from_root(&root, usize::MAX, budget) {
            FrontierOutcome::Complete(history) => {
                self.machine_backward_root = Some(Arc::new(history));
                FrontierOutcome::Complete(())
            }
            FrontierOutcome::Empty => FrontierOutcome::Empty,
            FrontierOutcome::Unknown(reason) => FrontierOutcome::Unknown(reason),
        }
    }
    pub(super) fn has_machine_program_input(&self) -> bool {
        self.machine_inputs.is_some() || self.machine_root_input.is_some()
    }
    pub(super) fn machine_program_inputs_complete(&self) -> bool {
        self.machine_inputs.is_some()
            && self.machine_root_history.is_some()
            && self.machine_backward_root.is_some()
    }

    pub(super) fn waiting_windows_match_inputs(&self) -> bool {
        match (&self.waiting_windows, &self.machine_inputs) {
            (Some(windows), Some(inputs)) => {
                windows.len() == inputs.len()
                    && windows.iter().zip(inputs.iter()).all(|(window, input)| {
                        window.lower.to_bits() == input.waiting_lower.to_bits()
                            && window.upper.to_bits() == input.waiting_upper.to_bits()
                    })
            }
            _ => true,
        }
    }

    pub(super) fn machine_backtrack_bound<const D: usize>(
        &self,
        graph: &Graph<D>,
        bound: &BoundMachinePathHistory,
        terminal: f64,
        root_is_entry_zero: bool,
        budget: FrontierBudget,
    ) -> FrontierOutcome<Vec<f64>> {
        let started = std::time::Instant::now();
        if !root_is_entry_zero {
            return FrontierOutcome::Unknown(FrontierUnknown::Unsupported(
                "machine backtrack requires explicit root-entry assumption",
            ));
        }
        let Some(inputs) = self.machine_inputs.as_ref() else {
            return FrontierOutcome::Unknown(FrontierUnknown::Unsupported(
                "paired machine inputs unavailable",
            ));
        };
        let Some(topology) = self.machine_topology.as_ref() else {
            return FrontierOutcome::Unknown(FrontierUnknown::Unsupported(
                "paired machine topology unavailable",
            ));
        };
        if !self.machine_topology_matches(graph)
            || !Arc::ptr_eq(inputs, &bound.inputs)
            || graph.source != topology.source
            || graph.sink != topology.sink
            || graph.vertices.len() != topology.vertex_count
            || graph.source >= graph.vertices.len()
            || graph.arcs.len() != topology.arcs.len()
            || graph.arcs.len() != inputs.len()
            || graph.arcs.len() != self.arc_count
            || bound.history.len().checked_sub(1) != Some(bound.path.len())
        {
            return FrontierOutcome::Unknown(FrontierUnknown::Unsupported(
                "paired machine history or topology changed",
            ));
        }
        // Reserve graph validation, history traversal, path validation and
        // inverse collection before running the independent backtrack.
        let Some(validation_work) = bound
            .path
            .len()
            .checked_mul(3)
            .and_then(|work| work.checked_add(graph.arcs.len()))
            .and_then(|work| work.checked_add(1))
        else {
            return FrontierOutcome::Unknown(FrontierUnknown::WorkLimit);
        };
        let Some(remaining_work) = budget.max_operations.checked_sub(validation_work) else {
            return FrontierOutcome::Unknown(FrontierUnknown::WorkLimit);
        };
        for (arc, &(id, tail, head)) in graph.arcs.iter().zip(topology.arcs.iter()) {
            if budget
                .max_elapsed
                .is_some_and(|limit| started.elapsed() >= limit)
            {
                return FrontierOutcome::Unknown(FrontierUnknown::Deadline);
            }
            if (arc.id, arc.tail, arc.head) != (id, tail, head)
                || tail >= graph.vertices.len()
                || head >= graph.vertices.len()
            {
                return FrontierOutcome::Unknown(FrontierUnknown::Unsupported(
                    "paired machine graph topology changed",
                ));
            }
        }
        let remaining = budget
            .max_elapsed
            .map(|limit| limit.saturating_sub(started.elapsed()));
        if remaining.is_some_and(|limit| limit.is_zero()) {
            return FrontierOutcome::Unknown(FrontierUnknown::Deadline);
        }
        let snapshots = match bound.history.snapshots(FrontierBudget {
            max_operations: bound.history.len(),
            max_elapsed: remaining,
            ..budget
        }) {
            FrontierOutcome::Complete(value) => value,
            FrontierOutcome::Empty => {
                return FrontierOutcome::Unknown(FrontierUnknown::Unsupported(
                    "paired machine history unavailable",
                ))
            }
            FrontierOutcome::Unknown(reason) => return FrontierOutcome::Unknown(reason),
        };
        if !matches!(
            snapshots.first(),
            Some(descriptor::MachinePathSnapshot::Root(_))
        ) {
            return FrontierOutcome::Unknown(FrontierUnknown::Unsupported(
                "paired machine history has no root",
            ));
        }
        let mut vertex = graph.source;
        let mut inverses = Vec::with_capacity(bound.path.len());
        for (position, &index) in bound.path.iter().enumerate() {
            if budget
                .max_elapsed
                .is_some_and(|limit| started.elapsed() >= limit)
            {
                return FrontierOutcome::Unknown(FrontierUnknown::Deadline);
            }
            let Some(arc) = graph.arcs.get(index) else {
                return FrontierOutcome::Unknown(FrontierUnknown::Unsupported(
                    "paired machine path arc index",
                ));
            };
            if arc.tail != vertex
                || !matches!(snapshots.get(position + 1), Some(descriptor::MachinePathSnapshot::Forward { arc, .. }) if *arc == index)
            {
                return FrontierOutcome::Unknown(FrontierUnknown::Unsupported(
                    "paired machine history path mismatch",
                ));
            }
            vertex = arc.head;
            inverses.push((index, &inputs[index].inverse));
        }
        let remaining = budget
            .max_elapsed
            .map(|limit| limit.saturating_sub(started.elapsed()));
        if remaining.is_some_and(|limit| limit.is_zero()) {
            return FrontierOutcome::Unknown(FrontierUnknown::Deadline);
        }
        PairedFrontOperations::machine_backtrack_coordinates(
            &bound.history,
            terminal,
            &inverses,
            FrontierBudget {
                max_operations: remaining_work,
                max_elapsed: remaining,
                ..budget
            },
        )
    }

    pub(super) fn set_machine_inputs(
        &mut self,
        inputs: Option<&[PreparedMachineArc]>,
        topology: Option<MachineTopology>,
    ) -> Result<bool> {
        if inputs.is_some_and(|inputs| {
            inputs.len() != self.arc_count || inputs.iter().any(|arc| !arc.valid())
        }) {
            return Err(invalid(
                "paired machine inputs must be finite, ordered, and match the arc count",
            ));
        }
        if machine_inputs_equal(self.machine_inputs.as_deref(), inputs)
            && self.machine_topology == topology
        {
            return Ok(false);
        }
        // Existing path histories were extended with the old arc operands.
        // Give the unchanged forward root a fresh identity before publishing
        // replacement operands so those histories cannot be joined later.
        let root_history = if let Some(root) = &self.machine_root_input {
            let budget = FrontierBudget::default();
            let state = outcome(MachineRootState::from_prepared(root, budget))?
                .ok_or_else(|| invalid("paired machine root is empty"))?;
            Some(Arc::new(
                outcome(MachinePathHistory::from_root(&state, usize::MAX, budget))?
                    .ok_or_else(|| invalid("paired machine root history is empty"))?,
            ))
        } else {
            None
        };
        self.machine_inputs = inputs.map(|inputs| Arc::from(inputs.to_vec()));
        self.machine_waiting_rates = None;
        self.machine_topology = topology;
        self.machine_root_history = root_history;
        self.machine_backward_root = None;
        Ok(true)
    }

    pub(super) fn set_machine_root_input(
        &mut self,
        root: Option<&PreparedSlopeFunction>,
    ) -> Result<bool> {
        if root.is_some_and(|root| {
            root.pairs().is_empty()
                || root.pairs().len() > 12
                || root
                    .pairs()
                    .iter()
                    .any(|(a, b)| !a.is_finite() || !b.is_finite())
        }) {
            return Err(invalid(
                "paired machine root must contain finite prepared pairs",
            ));
        }
        let same = match (&self.machine_root_input, root) {
            (Some(a), Some(b)) => {
                a.has_machine_only_geometry() == b.has_machine_only_geometry()
                    && a.pairs().len() == b.pairs().len()
                    && a.pairs().iter().zip(b.pairs()).all(|(a, b)| {
                        a.0.to_bits() == b.0.to_bits() && a.1.to_bits() == b.1.to_bits()
                    })
            }
            (None, None) => true,
            _ => false,
        };
        if same {
            return Ok(false);
        }
        let history = match root {
            Some(root) => {
                let budget = FrontierBudget::default();
                let state = outcome(MachineRootState::from_prepared(root, budget))?
                    .ok_or_else(|| invalid("paired machine root is empty"))?;
                Some(Arc::new(
                    outcome(MachinePathHistory::from_root(&state, usize::MAX, budget))?
                        .ok_or_else(|| invalid("paired machine root history is empty"))?,
                ))
            }
            None => None,
        };
        self.machine_root_input = root.cloned();
        self.machine_root_history = history;
        Ok(true)
    }

    pub(super) fn machine_replay<const D: usize>(
        &self,
        graph: &Graph<D>,
        root: &MachineRootState,
        path: &[usize],
        mode: i32,
        max_states: usize,
        budget: FrontierBudget,
    ) -> FrontierOutcome<MachinePathHistory> {
        let started = std::time::Instant::now();
        let Some(inputs) = self.machine_inputs.as_deref() else {
            return FrontierOutcome::Unknown(FrontierUnknown::Unsupported(
                "paired machine inputs unavailable",
            ));
        };
        let Some(topology) = self.machine_topology.as_ref() else {
            return FrontierOutcome::Unknown(FrontierUnknown::Unsupported(
                "paired machine topology unavailable",
            ));
        };
        if path
            .len()
            .checked_add(1)
            .map_or(true, |count| count > max_states)
        {
            return FrontierOutcome::Unknown(FrontierUnknown::ComponentLimit);
        }
        let Some(scans) = graph.arcs.len().checked_add(path.len()) else {
            return FrontierOutcome::Unknown(FrontierUnknown::WorkLimit);
        };
        let Some(remaining_work) = budget.max_operations.checked_sub(scans) else {
            return FrontierOutcome::Unknown(FrontierUnknown::WorkLimit);
        };
        if !self.machine_topology_matches(graph)
            || graph.source != topology.source
            || graph.sink != topology.sink
            || graph.vertices.len() != topology.vertex_count
            || graph.source >= graph.vertices.len()
            || graph.arcs.len() != topology.arcs.len()
            || graph.arcs.len() != inputs.len()
            || graph.arcs.len() != self.arc_count
        {
            return FrontierOutcome::Unknown(FrontierUnknown::Unsupported(
                "paired machine graph topology changed",
            ));
        }
        for (arc, &(id, tail, head)) in graph.arcs.iter().zip(topology.arcs.iter()) {
            if budget
                .max_elapsed
                .is_some_and(|limit| started.elapsed() >= limit)
            {
                return FrontierOutcome::Unknown(FrontierUnknown::Deadline);
            }
            if (arc.id, arc.tail, arc.head) != (id, tail, head)
                || tail >= graph.vertices.len()
                || head >= graph.vertices.len()
            {
                return FrontierOutcome::Unknown(FrontierUnknown::Unsupported(
                    "paired machine graph topology changed",
                ));
            }
        }
        let mut vertex = graph.source;
        let mut steps = Vec::with_capacity(path.len());
        for &index in path {
            if budget
                .max_elapsed
                .is_some_and(|limit| started.elapsed() >= limit)
            {
                return FrontierOutcome::Unknown(FrontierUnknown::Deadline);
            }
            let Some(arc) = graph.arcs.get(index) else {
                return FrontierOutcome::Unknown(FrontierUnknown::Unsupported(
                    "paired machine path arc index",
                ));
            };
            if arc.tail != vertex {
                return FrontierOutcome::Unknown(FrontierUnknown::Unsupported(
                    "paired machine path is not contiguous",
                ));
            }
            vertex = arc.head;
            steps.push((index, inputs[index].continuation_inputs()));
        }
        let remaining_time = budget
            .max_elapsed
            .map(|limit| limit.saturating_sub(started.elapsed()));
        if remaining_time.is_some_and(|limit| limit.is_zero()) {
            return FrontierOutcome::Unknown(FrontierUnknown::Deadline);
        }
        PairedFrontOperations::machine_replay_path(
            root,
            &steps,
            mode,
            max_states,
            FrontierBudget {
                max_operations: remaining_work,
                max_elapsed: remaining_time,
                ..budget
            },
        )
    }
    pub(super) fn set_waiting_windows(&mut self, windows: Option<&[Window]>) -> Result<bool> {
        if let Some(windows) = windows {
            if windows.len() != self.arc_count
                || windows
                    .iter()
                    .any(|w| !w.lower.is_finite() || !w.upper.is_finite() || w.lower > w.upper)
            {
                return Err(invalid(
                    "paired waiting windows must be finite, ordered, and match the arc count",
                ));
            }
        }
        if waiting_windows_equal(self.waiting_windows.as_deref(), windows) {
            return Ok(false);
        }
        self.waiting_windows = windows.map(|w| std::sync::Arc::from(w.to_vec()));
        Ok(true)
    }

    pub(super) fn new(arc_count: usize, mode: PairedResourceMode) -> Result<Self> {
        Ok(Self {
            objective_coefficient: 1.,
            arc_count,
            waiting_windows: None,
            machine_inputs: None,
            machine_waiting_rates: None,
            machine_root_input: None,
            machine_root_history: None,
            machine_backward_root: None,
            machine_topology: None,
            base_windows: None,
            mode,
        })
    }
    pub(super) fn has_costs(&self) -> bool {
        self.mode != PairedResourceMode::Feasibility
    }
    pub(super) fn uses_unscaled_final_value(&self) -> bool {
        self.has_costs() && self.objective_coefficient == 1.
    }
    pub(super) fn base_window_snapshot(&self) -> Option<&BaseWindows> {
        self.base_windows.as_ref()
    }

    pub(super) fn set_base_windows(&mut self, windows: BaseWindows) -> bool {
        if self.base_windows.as_ref() == Some(&windows) {
            return false;
        }
        self.base_windows = Some(windows);
        true
    }
    pub(super) fn clear_base_windows(&mut self) -> bool {
        self.base_windows.take().is_some()
    }
    pub(super) fn same_arc_operation(&self, a: usize, b: usize) -> bool {
        self.machine_inputs.as_ref().is_some_and(|inputs| {
            a < inputs.len()
                && b < inputs.len()
                && machine_inputs_equal(Some(&inputs[a..a + 1]), Some(&inputs[b..b + 1]))
        }) && self.machine_waiting_rates.as_ref().is_none_or(|rates| {
            rates
                .get(a)
                .zip(rates.get(b))
                .is_some_and(|(a, b)| a.to_bits() == b.to_bits())
        })
    }
    pub(super) fn selected_value(&self, label: &SlopeLabel, _scale: f64) -> Result<f64> {
        let history = label
            .machine_history
            .as_ref()
            .ok_or_else(|| crate::incomplete("paired native history is unavailable"))?;
        match history.last() {
            descriptor::MachinePathSnapshot::Forward { state, .. } => finite(state.key()),
            descriptor::MachinePathSnapshot::Root(_) => self.initial_front_minimum(),
            descriptor::MachinePathSnapshot::Backward { .. } => {
                Err(invalid("paired selected value requires a forward state"))
            }
        }
    }
    pub(super) fn root(&self, forward: bool) -> SlopeLabel {
        let history = if forward {
            self.machine_root_history.clone()
        } else {
            self.machine_backward_root.clone()
        };
        SlopeLabel {
            front: if forward {
                history.as_ref().map(|root| root.last().front().clone())
            } else {
                None
            },
            upper_negative: None,
            minimum: 0.,
            machine_history: history,
        }
    }

    pub(super) fn machine_mode(&self) -> i32 {
        match self.mode {
            PairedResourceMode::Feasibility => 4,
            PairedResourceMode::CostUnbounded => 5,
            PairedResourceMode::CostBounded => 6,
        }
    }

    fn extend_machine(
        &self,
        old: &SlopeLabel,
        arc: usize,
        forward: bool,
        scale: f64,
        source_scale: f64,
    ) -> Result<Option<SlopeLabel>> {
        let input = self
            .machine_inputs
            .as_ref()
            .and_then(|inputs| inputs.get(arc))
            .ok_or_else(|| crate::incomplete("paired machine arc inputs are unavailable"))?;
        let budget = FrontierBudget::default();
        if !forward {
            let history = old
                .machine_history
                .as_ref()
                .ok_or_else(|| crate::incomplete("paired machine backward root is unavailable"))?;
            if !self.matches_backward_root(history) {
                return Err(crate::incomplete(
                    "paired machine backward history is stale",
                ));
            }
            let ceiling = history.backward_ceiling().ok_or_else(|| {
                crate::incomplete("paired machine backward history is unavailable")
            })?;
            let backward = input.backward_inputs().ok_or_else(|| {
                crate::incomplete("paired machine backward arc inputs are unavailable")
            })?;
            let Some((front, bits, key)) =
                outcome(PairedFrontOperations::machine_backward_transition(
                    history.last().front(),
                    &backward,
                    ceiling,
                    self.machine_mode(),
                    budget,
                ))?
            else {
                return Ok(None);
            };
            let next_history = outcome(history.append_backward(arc, front, bits, key, budget))?
                .ok_or_else(|| crate::incomplete("paired machine backward history rejected"))?;
            return Ok(Some(SlopeLabel {
                front: None,
                upper_negative: None,
                minimum: if self.mode == PairedResourceMode::Feasibility || scale == 0. {
                    0.
                } else {
                    finite(key * scale)?
                },
                machine_history: Some(Arc::new(next_history)),
            }));
        }
        let history = old
            .machine_history
            .as_ref()
            .ok_or_else(|| crate::incomplete("paired machine forward history is unavailable"))?;
        if !self.matches_forward_root(history) {
            return Err(crate::incomplete("paired machine forward history is stale"));
        }
        let Some((reduced, next)) = outcome(PairedFrontOperations::machine_transition(
            history.last().front(),
            &input.continuation_inputs(),
            self.machine_mode(),
            budget,
        ))?
        else {
            return Ok(None);
        };
        let front = next.front().clone();
        let next_history = outcome(history.append(arc, &reduced, &next, budget))?
            .ok_or_else(|| crate::incomplete("paired machine history rejected"))?;
        let objective = if self.mode == PairedResourceMode::Feasibility || scale == 0. {
            0.
        } else {
            self.final_objective(next.key(), source_scale)?
        };
        Ok(Some(SlopeLabel {
            front: Some(front),
            upper_negative: None,
            minimum: finite(objective)?,
            machine_history: Some(Arc::new(next_history)),
        }))
    }

    fn initial_front_minimum(&self) -> Result<f64> {
        if let Some(root) = &self.machine_root_input {
            let pairs = root.pairs();
            let mut value = pairs[0].0;
            let mut minimum = value;
            for pair in pairs.windows(2) {
                value = finite(value + finite(finite(pair[1].1 - pair[0].1)? * pair[1].0)?)?;
                if value < minimum {
                    minimum = value;
                }
            }
            Ok(minimum)
        } else {
            Err(crate::incomplete("paired native root is unavailable"))
        }
    }

    pub(super) fn final_objective(&self, value: f64, scale: f64) -> Result<f64> {
        if !self.has_costs() || scale == 0. || self.objective_coefficient == 0. {
            return Ok(0.);
        }
        finite(value * finite(scale * self.objective_coefficient)?)
    }

    pub(super) fn extend(
        &self,
        old: &SlopeLabel,
        arc: usize,
        forward: bool,
        _first_backward: bool,
        scale: f64,
        machine_domain: bool,
    ) -> Result<Option<SlopeLabel>> {
        if !machine_domain
            || !self.machine_program_inputs_complete()
            || !self.has_machine_program_input()
        {
            return Err(crate::incomplete(
                "paired native front program is not prepared",
            ));
        }
        let weighted = finite(scale * self.objective_coefficient)?;
        self.extend_machine(old, arc, forward, weighted, scale)
    }
}
impl CoupledLabelState {
    /// Backward native fronts in paired-resource order.
    pub fn slope_relation_backward_fronts(&self) -> impl Iterator<Item = Option<&SlopeFront>> {
        self.slope_relations.iter().map(|label| {
            if self.suffix.is_some() {
                label
                    .machine_history
                    .as_ref()
                    .map(|history| history.last().front())
            } else {
                None
            }
        })
    }
    /// Computed forward paired slope fronts. Backward states retain the operator
    /// arc program plus a feasibility upper or exact bounded-cost transfer
    /// when available; they expose no invented forward frontier.
    pub fn slope_relation_fronts(&self) -> impl Iterator<Item = Option<&SlopeFront>> {
        self.slope_relations
            .iter()
            .map(|label| label.front.as_ref())
    }
}

#[cfg(test)]
#[path = "../../../../tests/unit/graph/coupled/kernel/slope_paired/tests.rs"]
mod tests;
