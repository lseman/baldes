//! Paired raw slope relations: physical dependent state and objective stay separate.
use super::bounds::Range;
use super::*;
use descriptor::{
    FrontierBudget, FrontierOutcome, FrontierUnknown, MachinePathHistory, MachineRootState,
    PairedCostNormalization, PreparedSlopeAdjointArc, PreparedSlopeRelationArc, SlopeFront,
    SlopeRelationArc,
};
use relation::{
    BoundedCostFront, PreparedCostLattice, PreparedRelationArc, RelationFront, RelationPiece,
};
use slope_interval::{outcome, SlopeLabel};
use std::sync::{Arc, OnceLock};

/// Persistent suffix program. Numerical fronts are diagnostic, so label
/// extension records the program and only evaluates it when inspected.
#[derive(Debug)]
pub(super) struct RawAdjoint {
    pub(super) depth: usize,
    arc: Arc<PreparedSlopeAdjointArc>,
    previous: Option<Arc<RawAdjoint>>,
    terminal: bool,
    outcome: OnceLock<FrontierOutcome<SlopeFront>>,
}
impl Drop for RawAdjoint {
    fn drop(&mut self) {
        // A uniquely owned long suffix would otherwise recursively drop one
        // Arc per arc. Stop at a shared node, whose remaining owners keep it
        // alive, and drain uniquely owned predecessors iteratively.
        let mut previous = self.previous.take();
        while let Some(link) = previous {
            match Arc::try_unwrap(link) {
                Ok(mut node) => previous = node.previous.take(),
                Err(_) => break,
            }
        }
    }
}
impl RawAdjoint {
    fn recompute(&self, budget: FrontierBudget) -> FrontierOutcome<SlopeFront> {
        let started = std::time::Instant::now();
        let mut nodes = Vec::new();
        let mut current = self;
        loop {
            if nodes.len() >= budget.max_operations {
                return FrontierOutcome::Unknown(FrontierUnknown::WorkLimit);
            }
            if budget
                .max_elapsed
                .is_some_and(|limit| started.elapsed() >= limit)
            {
                return FrontierOutcome::Unknown(FrontierUnknown::Deadline);
            }
            nodes.push(current);
            match current.previous.as_deref() {
                Some(previous) => current = previous,
                None => break,
            }
        }
        let per_arc = (budget.max_operations - nodes.len()) / nodes.len();
        if per_arc == 0 {
            return FrontierOutcome::Unknown(FrontierUnknown::WorkLimit);
        }
        let mut result = None;
        for node in nodes.into_iter().rev() {
            let remaining_time = match budget.max_elapsed {
                Some(limit) => match limit.checked_sub(started.elapsed()) {
                    Some(remaining) if !remaining.is_zero() => Some(remaining),
                    _ => return FrontierOutcome::Unknown(FrontierUnknown::Deadline),
                },
                None => None,
            };
            let arc_budget = FrontierBudget {
                max_operations: per_arc,
                max_elapsed: remaining_time,
                ..budget
            };
            result = Some(match result {
                None if node.terminal => node.arc.prepend_terminal(arc_budget),
                None => FrontierOutcome::Unknown(FrontierUnknown::Unsupported(
                    "preceding raw adjoint is unavailable",
                )),
                Some(FrontierOutcome::Complete(front)) => node.arc.prepend(&front, arc_budget),
                Some(other) => other,
            });
        }
        result.expect("a suffix contains at least one arc")
    }

    fn outcome(&self) -> &FrontierOutcome<SlopeFront> {
        // Walk iteratively to avoid recursion on long bounded walks. Shared
        // suffixes retain their computed result and are evaluated at most once
        // in the uncontended case; racing computations have identical inputs.
        let mut pending = Vec::new();
        let mut current = self;
        while current.outcome.get().is_none() {
            pending.push(current);
            match current.previous.as_deref() {
                Some(previous) => current = previous,
                None => break,
            }
        }
        while let Some(node) = pending.pop() {
            node.outcome.get_or_init(|| match node.previous.as_deref() {
                Some(previous) => match previous.outcome.get() {
                    Some(FrontierOutcome::Complete(front)) => {
                        node.arc.prepend(front, FrontierBudget::default())
                    }
                    Some(FrontierOutcome::Empty) => FrontierOutcome::Empty,
                    Some(FrontierOutcome::Unknown(reason)) => {
                        FrontierOutcome::Unknown(reason.clone())
                    }
                    None => FrontierOutcome::Unknown(FrontierUnknown::Unsupported(
                        "preceding raw adjoint is unavailable",
                    )),
                },
                None if node.terminal => node.arc.prepend_terminal(FrontierBudget::default()),
                None => FrontierOutcome::Unknown(FrontierUnknown::Unsupported(
                    "preceding raw adjoint is unavailable",
                )),
            });
        }
        self.outcome.get().expect("raw adjoint outcome initialized")
    }
}

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

#[derive(Clone, Debug)]
pub(super) struct PreparedSlopePaired {
    pub(super) objective_coefficient: f64,
    initial: SlopeLabel,
    upper_initial: Option<SlopeFront>,
    initial_minimum: f64,
    arcs: std::sync::Arc<[PreparedSlopeRelationArc]>,
    upper_arcs: Option<std::sync::Arc<[SlopeFront]>>,
    sources: std::sync::Arc<[SlopeRelationArc]>,
    operator_classes: Arc<[usize]>,
    waiting_windows: Option<std::sync::Arc<[Window]>>,
    machine_inputs: Option<Arc<[PreparedMachineArc]>>,
    machine_root_input: Option<PreparedSlopeFunction>,
    machine_root_history: Option<Arc<MachinePathHistory>>,
    machine_backward_root: Option<Arc<MachinePathHistory>>,
    machine_topology: Option<MachineTopology>,
    backward: std::sync::Arc<[Option<PreparedRelationArc>]>,
    adjoints: Arc<[Arc<PreparedSlopeAdjointArc>]>,
    exact_cost: bool,
    integral_costs: bool,
    monotone_constant_transfers: bool,
    cost_lattice: Option<Arc<PreparedCostLattice>>,
    ceiling: f64,
    mode: PairedResourceMode,
    normalization: PairedCostNormalization,
    policy: FrontOperationPolicy,
    /// Cost-unbounded native mode without a reachable upper bound (the
    /// reference's 1e12 infinity): a higher front may pay its excess.
    shifts: bool,
}
impl PartialEq for PreparedSlopePaired {
    fn eq(&self, b: &Self) -> bool {
        self.initial == b.initial
            && self.objective_coefficient == b.objective_coefficient
            && self.initial_minimum.to_bits() == b.initial_minimum.to_bits()
            && self.sources == b.sources
            && self.operator_classes == b.operator_classes
            && waiting_windows_equal(
                self.waiting_windows.as_deref(),
                b.waiting_windows.as_deref(),
            )
            && machine_inputs_equal(self.machine_inputs.as_deref(), b.machine_inputs.as_deref())
            && match (&self.machine_backward_root, &b.machine_backward_root) {
                (None, None) => true,
                (Some(a), Some(b)) => Arc::ptr_eq(a, b),
                _ => false,
            }
            && match (&self.machine_root_input, &b.machine_root_input) {
                (Some(a), Some(b)) => {
                    a.is_machine_zero_length() == b.is_machine_zero_length()
                        && a.pairs().len() == b.pairs().len()
                        && a.pairs().iter().zip(b.pairs()).all(|(a, b)| {
                            a.0.to_bits() == b.0.to_bits() && a.1.to_bits() == b.1.to_bits()
                        })
                }
                (None, None) => true,
                _ => false,
            }
            && self.machine_topology == b.machine_topology
            && self.mode == b.mode
            && self.integral_costs == b.integral_costs
            && self.monotone_constant_transfers == b.monotone_constant_transfers
            && self.cost_lattice == b.cost_lattice
            && self.normalization == b.normalization
            && self.policy == b.policy
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
fn constant(function: &PreparedSlopeFunction) -> Option<f64> {
    function
        .pairs()
        .iter()
        .skip(1)
        .all(|p| p.0 == 0.)
        .then(|| function.pairs()[0].0)
}
fn negate_function(function: &PreparedSlopeFunction) -> Result<PreparedSlopeFunction> {
    PreparedSlopeFunction::from_prepared_pairs(
        function
            .pairs()
            .iter()
            .map(|&(slope, coordinate)| (-slope, coordinate))
            .collect(),
    )
    .map_err(|error| invalid(error.to_string()))
}
fn negated_upper_on(
    feasible: &SlopeFront,
    negative_upper: &SlopeFront,
    policy: FrontOperationPolicy,
    budget: FrontierBudget,
) -> FrontierOutcome<SlopeFront> {
    let zero_components = match feasible
        .components()
        .iter()
        .map(|component| {
            let (lower, upper) = component.domain();
            if lower == upper {
                PreparedSlopeFunction::constant_point(0., lower)
            } else {
                PreparedSlopeFunction::from_prepared_pairs(vec![(0., lower), (0., upper)])
            }
        })
        .collect::<std::result::Result<Vec<_>, _>>()
    {
        Ok(components) => components,
        Err(error) => return FrontierOutcome::Unknown(FrontierUnknown::Arithmetic(error)),
    };
    match SlopeFront::from_components(zero_components, budget) {
        FrontierOutcome::Complete(mask) => negative_upper.add(&mask, policy, budget),
        FrontierOutcome::Empty => FrontierOutcome::Empty,
        FrontierOutcome::Unknown(reason) => FrontierOutcome::Unknown(reason),
    }
}
// Conversion is advisory. Singleton domains and unrepresentable slopes keep
// suffix replay as the authoritative feasibility check.
fn relation_function(function: &PreparedSlopeFunction) -> Option<PiecewiseLinearFunction> {
    let pairs = function.pairs();
    let mut points = vec![[pairs[0].1, pairs[0].0]];
    for &(slope, end) in &pairs[1..] {
        let [start, value] = *points.last()?;
        let next = value + slope * (end - start);
        if (next - value) / (end - start) != slope {
            return None;
        }
        points.push([end, next]);
    }
    PiecewiseLinearFunction::new(points).ok()
}
fn backward_arc(arc: &SlopeRelationArc) -> Option<PreparedRelationArc> {
    PreparedRelationArc::new(&PairedResourceArc {
        arrival: relation_function(&arc.arrival)?,
        increment: relation_function(&arc.increment)?,
        lower: relation_function(&arc.lower)?,
        upper: relation_function(&arc.upper)?,
        waiting: arc.waiting,
    })
    .ok()
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
        self.machine_inputs.is_some() == self.machine_root_history.is_some()
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
        if !Arc::ptr_eq(inputs, &bound.inputs)
            || graph.source != topology.source
            || graph.sink != topology.sink
            || graph.vertices.len() != topology.vertex_count
            || graph.source >= graph.vertices.len()
            || graph.arcs.len() != topology.arcs.len()
            || graph.arcs.len() != inputs.len()
            || graph.arcs.len() != self.arcs.len()
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
        PreparedSlopeRelationArc::machine_backtrack_coordinates(
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
        if inputs.is_some() && self.policy != FrontOperationPolicy::FrontAlgebra {
            return Err(invalid("paired machine inputs require front algebra"));
        }
        if inputs.is_some_and(|inputs| {
            inputs.len() != self.arcs.len() || inputs.iter().any(|arc| !arc.valid())
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
        self.machine_inputs = inputs.map(|inputs| Arc::from(inputs.to_vec()));
        self.machine_topology = topology;
        self.machine_backward_root = None;
        Ok(true)
    }

    pub(super) fn set_machine_root_input(
        &mut self,
        root: Option<&PreparedSlopeFunction>,
    ) -> Result<bool> {
        if root.is_some() && self.policy != FrontOperationPolicy::FrontAlgebra {
            return Err(invalid("paired machine root requires front algebra"));
        }
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
                a.is_machine_zero_length() == b.is_machine_zero_length()
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
        if graph.source != topology.source
            || graph.sink != topology.sink
            || graph.vertices.len() != topology.vertex_count
            || graph.source >= graph.vertices.len()
            || graph.arcs.len() != topology.arcs.len()
            || graph.arcs.len() != inputs.len()
            || graph.arcs.len() != self.arcs.len()
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
            steps.push((
                index,
                &self.arcs[index],
                inputs[index].continuation_inputs(),
            ));
        }
        let remaining_time = budget
            .max_elapsed
            .map(|limit| limit.saturating_sub(started.elapsed()));
        if remaining_time.is_some_and(|limit| limit.is_zero()) {
            return FrontierOutcome::Unknown(FrontierUnknown::Deadline);
        }
        PreparedSlopeRelationArc::machine_replay_path(
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
        if windows.is_some() && self.policy != FrontOperationPolicy::FrontAlgebra {
            return Err(invalid("paired waiting windows require front algebra"));
        }
        if let Some(windows) = windows {
            if windows.len() != self.arcs.len()
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

    pub(super) fn new(
        initial: &PreparedSlopeFunction,
        base: Window,
        arcs: &[SlopeRelationArc],
        mode: PairedResourceMode,
        normalization: PairedCostNormalization,
        policy: FrontOperationPolicy,
    ) -> Result<Self> {
        let budget = FrontierBudget::default();
        let Some(front) = outcome(
            SlopeFront::from_prepared(initial.clone())
                .restrict(base.lower, base.upper, policy, budget),
        )?
        else {
            return Err(invalid("empty raw paired initial front"));
        };
        let Some(witness) = outcome(front.minimum(budget))? else {
            return Err(invalid("empty raw paired initial front"));
        };
        let upper_initial = negate_function(initial)
            .and_then(|function| {
                Ok(SlopeFront::from_prepared(function)
                    .restrict(base.lower, base.upper, policy, budget))
            })
            .ok()
            .and_then(|front| outcome(front).ok().flatten());
        let upper_arcs = arcs
            .iter()
            .map(|arc| negate_function(&arc.upper).map(SlopeFront::from_prepared))
            .collect::<Result<Vec<_>>>()
            .ok()
            .map(std::sync::Arc::from);
        let mut shifts = false;
        if mode != PairedResourceMode::Feasibility {
            let first = arcs
                .first()
                .ok_or_else(|| invalid("raw paired cost requires physical arcs"))?;
            let lower = constant(&first.lower).ok_or_else(|| {
                invalid("raw paired cost requires shared constant dependent bounds")
            })?;
            let upper = constant(&first.upper).ok_or_else(|| {
                invalid("raw paired cost requires shared constant dependent bounds")
            })?;
            if lower > upper
                || arcs.iter().any(|arc| {
                    constant(&arc.lower) != Some(lower) || constant(&arc.upper) != Some(upper)
                })
            {
                return Err(invalid(
                    "raw paired cost requires shared ordered constant dependent bounds",
                ));
            }
            shifts = mode == PairedResourceMode::CostUnbounded
                && policy == FrontOperationPolicy::FrontAlgebra
                && upper >= 1e12;
            if mode == PairedResourceMode::CostBounded {
                if witness.value < lower || arcs.iter().any(|arc| arc.waiting < 0) {
                    return Err(invalid("bounded raw paired cost requires nonnegative waiting and initial minimum above the lower bound"));
                }
                for arc in arcs {
                    let Some(minimum) =
                        outcome(SlopeFront::from_prepared(arc.increment.clone()).minimum(budget))?
                    else {
                        return Err(invalid("empty raw paired increment"));
                    };
                    if minimum.value <= 0. {
                        return Err(invalid(
                            "bounded raw paired cost requires positive increment functions",
                        ));
                    }
                }
            }
        }
        let prepared = arcs
            .iter()
            .map(|arc| {
                PreparedSlopeRelationArc::new(arc, policy)
                    .map_err(|error| invalid(error.to_string()))
            })
            .collect::<Result<Vec<_>>>()?;
        let operator_classes = continuation::operator_classes(arcs.iter().map(|arc| {
            let mut key = vec![arc.waiting as i64 as u64];
            for function in [&arc.arrival, &arc.increment, &arc.lower, &arc.upper] {
                key.push(u64::from(function.is_machine_zero_length()));
                key.push(function.pairs().len() as u64);
                for &(slope, coordinate) in function.pairs() {
                    key.push(slope.to_bits());
                    key.push(coordinate.to_bits());
                }
            }
            key
        }));
        let adjoints = arcs
            .iter()
            .map(|arc| {
                PreparedSlopeAdjointArc::new(arc, policy)
                    .map(Arc::new)
                    .map_err(|error| invalid(error.to_string()))
            })
            .collect::<Result<Vec<_>>>()?;
        let backward: std::sync::Arc<[Option<PreparedRelationArc>]> = arcs
            .iter()
            .map(|arc| {
                (policy == FrontOperationPolicy::StrictDomain)
                    .then(|| backward_arc(arc))
                    .flatten()
            })
            .collect::<Vec<_>>()
            .into();
        let integer = |x: f64| x.is_finite() && x.fract() == 0. && x.abs() <= 1_099_511_627_776.;
        // The raw/knot conversion preserves every stored segment slope. Only
        // the existing exact translation/nonnegative-increment contract may use
        // the converted cache for cost dominance; general raw programs replay.
        let exact_cost = mode != PairedResourceMode::Feasibility
            && integer(base.lower)
            && integer(base.upper)
            && constant(initial).is_some_and(integer)
            && arcs.first().is_some_and(|arc| {
                constant(&arc.lower).is_some_and(|lower| witness.value >= lower)
            })
            && initial
                .pairs()
                .iter()
                .all(|pair| integer(pair.0) && integer(pair.1))
            && backward.iter().all(|arc| {
                arc.as_ref()
                    .is_some_and(PreparedRelationArc::exact_cache_eligible)
            });
        let ceiling = arcs
            .first()
            .and_then(|arc| constant(&arc.upper))
            .unwrap_or(0.);
        let integral_costs = exact_cost && arcs.iter().all(|arc| arc.waiting == 0);
        // Identity base maps and identical constant, zero-waiting transfers
        // admit control deletion: remove extra intermediate base coordinates
        // from a longer suffix. The remaining coordinates still satisfy the
        // same domains and identity arrivals. T(y)=max(y+c, lower), c>=0,
        // is monotone and T(y)>=y under binary64 arithmetic; any upper bound
        // satisfied by the longer suffix also holds for the shorter one.
        let monotone_constant_transfers = mode != PairedResourceMode::Feasibility
            && constant(initial).is_some()
            && arcs.first().is_some_and(|first| {
                let domain = initial.domain();
                domain.0 == 0.
                    && domain.1 > 0.
                    && first.arrival.pairs() == [(0., 0.), (1., domain.1)]
                    && [&first.increment, &first.lower, &first.upper]
                        .iter()
                        .all(|f| {
                            f.domain() == domain && f.pairs().len() == 2 && constant(f).is_some()
                        })
                    && constant(&first.increment).is_some_and(|value| value >= 0.)
                    && arcs.iter().all(|arc| arc.waiting == 0 && arc == first)
            });
        let cost_lattice = if mode != PairedResourceMode::Feasibility
            && !integral_costs
            && policy == FrontOperationPolicy::StrictDomain
        {
            constant(initial).and_then(|value| {
                let arcs = backward.iter().cloned().collect::<Option<Vec<_>>>()?;
                PreparedCostLattice::new(value, &arcs).map(Arc::new)
            })
        } else {
            None
        };
        Ok(Self {
            objective_coefficient: 1.,
            initial: SlopeLabel::root(&front, true),
            upper_initial,
            initial_minimum: finite(witness.value)?,
            arcs: prepared.into(),
            upper_arcs,
            sources: arcs.to_vec().into(),
            operator_classes,
            waiting_windows: None,
            machine_inputs: None,
            machine_root_input: None,
            machine_root_history: None,
            machine_backward_root: None,
            machine_topology: None,
            backward,
            adjoints: adjoints.into(),
            exact_cost,
            integral_costs,
            monotone_constant_transfers,
            cost_lattice,
            ceiling,
            mode,
            normalization,
            policy,
            shifts,
        })
    }
    /// Forward fronts cover and stay below under this resource's policy.
    /// Backward feasibility uppers cover and stay above; exact bounded-cost
    /// transfers additionally preserve the incremental suffix objective.
    /// Unknowns require replay. A shifting forward front may instead exceed by
    /// delta: every future step is max(value + increment, lower), which keeps
    /// the pointwise gap, so a's terminal minimum exceeds b's by at most delta.
    /// The objective is scale times that minimum plus a common baseline, so
    /// the penalty is scale * delta + objective_b - objective_a. Returns the
    /// penalty, or None.
    pub(super) fn penalty(
        &self,
        a: &SlopeLabel,
        b: &SlopeLabel,
        forward: bool,
        scale: f64,
    ) -> Option<f64> {
        let scale = if self.has_costs() {
            scale * self.objective_coefficient
        } else {
            0.
        };
        if !forward && self.monotone_constant_suffix_dominates(a, b) {
            return (scale.is_finite() && scale >= 0.).then_some(0.);
        }
        if a == b {
            return Some(0.);
        }
        // Machine histories carry distinct raw arithmetic and waiting bits.
        // Until a matching continuation proof exists, never compare two
        // different machine programs by their source-front envelopes.
        if a.machine_history.is_some() || b.machine_history.is_some() {
            return None;
        }
        if !forward {
            if self.has_costs() {
                return (scale.is_finite() && scale >= 0.)
                    .then_some(())
                    .and_then(|()| match (&a.bounded_cost, &b.bounded_cost) {
                        (Some(a), Some(b)) => {
                            let dominates = if self.has_exact_cost_continuations() {
                                a.dominates_exact_lattice(b)
                            } else {
                                a.dominates(b)
                            };
                            dominates.then_some(0.)
                        }
                        _ => None,
                    });
            }
            return match (&a.upper, &b.upper) {
                (Some(a), Some(b))
                    if match self.policy {
                        FrontOperationPolicy::StrictDomain => a.covers_above_strict(b),
                        FrontOperationPolicy::FrontAlgebra => a.covers_above(b),
                    } =>
                {
                    Some(0.)
                }
                _ => None,
            };
        }
        if scale < 0. {
            if self.shifts || self.policy != FrontOperationPolicy::StrictDomain {
                return None;
            }
            let (Some(front_a), Some(front_b)) = (&a.upper_negative, &b.upper_negative) else {
                return None;
            };
            let (Some(lower_a), Some(lower_b)) = (&a.front, &b.front) else {
                return None;
            };
            return (lower_a.covers_below(lower_b, self.policy)
                && front_a.covers_below(front_b, self.policy))
            .then_some(0.);
        }
        let (Some(front_a), Some(front_b)) = (&a.front, &b.front) else {
            return None;
        };
        if !(self.shifts && scale != 0.) {
            return front_a.covers_below(front_b, self.policy).then_some(0.);
        }
        let excess = front_a.excess(front_b, self.policy)?;
        let gap = Range::point(excess)
            .multiply(Range::point(scale))
            .add(Range::point(b.minimum))
            .subtract(Range::point(a.minimum))
            .hi;
        gap.is_finite().then_some(gap.max(0.))
    }
    pub(super) fn has_costs(&self) -> bool {
        self.mode != PairedResourceMode::Feasibility
    }
    pub(super) fn monotone_constant_suffix_dominates(
        &self,
        a: &SlopeLabel,
        b: &SlopeLabel,
    ) -> bool {
        self.monotone_constant_transfers
            && self.waiting_windows.is_none()
            && self.machine_inputs.is_none()
            && a.machine_history.is_none()
            && b.machine_history.is_none()
            && matches!((&a.raw_upper, &b.raw_upper), (Some(a), Some(b)) if a.depth <= b.depth)
    }
    pub(super) fn has_exact_cost_continuations(&self) -> bool {
        self.integral_costs || self.cost_lattice.is_some()
    }
    pub(super) fn same_arc_operation(&self, a: usize, b: usize) -> bool {
        self.operator_classes[a] == self.operator_classes[b]
            && self.waiting_windows.as_ref().is_none_or(|windows| {
                windows[a].lower.to_bits() == windows[b].lower.to_bits()
                    && windows[a].upper.to_bits() == windows[b].upper.to_bits()
            })
            && self.machine_inputs.as_ref().is_none_or(|inputs| {
                machine_inputs_equal(Some(&inputs[a..a + 1]), Some(&inputs[b..b + 1]))
            })
    }
    pub(super) fn uses_strict_front_policy(&self) -> bool {
        self.policy == FrontOperationPolicy::StrictDomain
    }
    pub(super) fn cached_join(
        &self,
        forward: &SlopeLabel,
        backward: &SlopeLabel,
    ) -> Option<Option<f64>> {
        if !self.has_exact_cost_continuations()
            || !self.uses_strict_front_policy()
            || forward.machine_history.is_some()
            || backward.machine_history.is_some()
        {
            return None;
        }
        let cache = backward.bounded_cost.as_deref()?;
        let front = forward.front.as_ref()?;
        let mut pieces: Vec<RelationPiece> = Vec::with_capacity(front.components().len());
        for component in front.components() {
            let pairs = component.pairs();
            let value = pairs.first()?.0;
            if component.is_machine_zero_length()
                || !value.is_finite()
                || pairs.iter().skip(1).any(|pair| pair.0 != 0.)
                || (value == 0.
                    && value.is_sign_negative()
                    && pairs
                        .iter()
                        .skip(1)
                        .any(|pair| pair.0.to_bits() != (-0.0f64).to_bits()))
            {
                return None;
            }
            if pieces
                .first()
                .is_some_and(|first| first.left[1].to_bits() != value.to_bits())
            {
                return None;
            }
            let (lower, upper) = component.domain();
            pieces.push(RelationPiece::constant(Window { lower, upper }, value)?);
        }
        if pieces.is_empty() {
            return None;
        }
        // A raw front may represent overlapping component domains. The join
        // stream needs ordered, nonoverlapping pieces. Their common constant
        // value makes the exact domain union sufficient here.
        pieces.sort_by(|a, b| {
            a.left[0]
                .total_cmp(&b.left[0])
                .then_with(|| a.right[0].total_cmp(&b.right[0]))
        });
        let mut union: Vec<RelationPiece> = Vec::with_capacity(pieces.len());
        for piece in pieces {
            if let Some(last) = union.last_mut() {
                if piece.left[0] <= last.right[0] {
                    last.right[0] = last.right[0].max(piece.right[0]);
                    continue;
                }
            }
            union.push(piece);
        }
        cache.checked_compatible_join(&RelationFront::from_pieces(union))
    }
    fn machine_constant_value(front: &SlopeFront) -> Option<f64> {
        let [function] = front.components() else {
            return None;
        };
        let pairs = function.pairs();
        let value = pairs.first()?.0;
        let stable_zero = value != 0.
            || !value.is_sign_negative()
            || pairs
                .iter()
                .skip(1)
                .all(|pair| pair.0.to_bits() == (-0.0f64).to_bits());
        (value.is_finite()
            && stable_zero
            && value != 1e12
            && !function.is_machine_zero_length()
            && pairs.iter().skip(1).all(|pair| pair.0 == 0.))
        .then_some(value)
    }
    pub(super) fn selected_value(&self, label: &SlopeLabel, scale: f64) -> Result<f64> {
        if let Some(history) = &label.machine_history {
            // Use only constants whose literal accumulation preserves their
            // bits at every coordinate, including signed-zero behavior.
            if let descriptor::MachinePathSnapshot::Forward { state, .. } = history.last() {
                if let Some(value) = Self::machine_constant_value(state.front()) {
                    return Ok(value);
                }
            }
            return Err(crate::incomplete(
                "paired machine selected value is unavailable",
            ));
        }
        if self.has_costs() && finite(scale * self.objective_coefficient)? < 0. {
            let upper = label.upper_negative.as_ref().ok_or_else(|| {
                crate::incomplete("signed raw paired upper frontier is unavailable")
            })?;
            let witness = outcome(upper.minimum(FrontierBudget::default()))?
                .ok_or_else(|| crate::incomplete("empty signed raw paired upper frontier"))?;
            finite(-witness.value)
        } else {
            let front = label
                .front
                .as_ref()
                .ok_or_else(|| invalid("missing raw paired front"))?;
            let witness = outcome(front.minimum(FrontierBudget::default()))?
                .ok_or_else(|| invalid("empty raw paired front"))?;
            finite(witness.value)
        }
    }
    pub(super) fn root(&self, forward: bool, machine_domain: bool) -> SlopeLabel {
        if forward {
            let mut root = self.initial.clone();
            if machine_domain {
                if let Some(history) = &self.machine_root_history {
                    root.front = Some(history.last().front().clone());
                    root.upper_negative = None;
                    root.minimum = 0.;
                    root.machine_history = Some(history.clone());
                    return root;
                }
            }
            root.upper_negative = self.upper_initial.clone();
            root
        } else {
            SlopeLabel {
                front: None,
                upper_negative: None,
                minimum: 0.,
                upper: None,
                bounded_cost: None,
                raw_upper: None,
                machine_history: machine_domain
                    .then(|| self.machine_backward_root.clone())
                    .flatten(),
            }
        }
    }

    fn machine_root(&self, budget: FrontierBudget) -> FrontierOutcome<MachineRootState> {
        let Some(root) = self.machine_root_input.as_ref() else {
            return FrontierOutcome::Unknown(FrontierUnknown::Unsupported(
                "explicit paired machine root is unavailable",
            ));
        };
        MachineRootState::from_prepared(root, budget)
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
        let prepared = self
            .arcs
            .get(arc)
            .ok_or_else(|| crate::incomplete("paired machine arc is unavailable"))?;
        let budget = FrontierBudget::default();
        if !forward {
            let history = old
                .machine_history
                .as_ref()
                .ok_or_else(|| crate::incomplete("paired machine backward root is unavailable"))?;
            let ceiling = history.backward_ceiling().ok_or_else(|| {
                crate::incomplete("paired machine backward history is unavailable")
            })?;
            let backward = input.backward_inputs().ok_or_else(|| {
                crate::incomplete("paired machine backward arc inputs are unavailable")
            })?;
            let (front, bits, key) =
                outcome(PreparedSlopeRelationArc::machine_backward_transition(
                    history.last().front(),
                    &backward,
                    ceiling,
                    self.machine_mode(),
                    budget,
                ))?
                .ok_or_else(|| crate::incomplete("paired machine backward transition rejected"))?;
            let mut source = None;
            if self.mode != PairedResourceMode::Feasibility && scale != 0. {
                let unavailable = || {
                    crate::incomplete(
                        "paired machine backward objective normalization is unavailable",
                    )
                };
                if scale < 0.
                    || !self.monotone_constant_transfers
                    || self.waiting_windows.is_some()
                    || ceiling.to_bits() != self.ceiling.to_bits()
                    || history.len().saturating_mul(128) > budget.max_operations
                {
                    return Err(unavailable());
                }
                let mut previous = old.clone();
                previous.machine_history = None;
                let next = self
                    .extend(
                        &previous,
                        arc,
                        false,
                        history.len() == 1,
                        source_scale,
                        false,
                    )?
                    .ok_or_else(unavailable)?;
                let descriptor::FrontierOutcome::Complete(expected) =
                    next.raw_upper.as_deref().ok_or_else(unavailable)?.outcome()
                else {
                    return Err(unavailable());
                };
                if !Self::same_machine_constant_front(
                    expected,
                    &SlopeFront::from_prepared(front.clone()),
                ) {
                    return Err(unavailable());
                }
                source = Some(next);
            }
            let next_history = outcome(history.append_backward(arc, front, bits, key, budget))?
                .ok_or_else(|| crate::incomplete("paired machine backward history rejected"))?;
            if let Some(mut source) = source {
                source.machine_history = Some(Arc::new(next_history));
                return Ok(Some(source));
            }
            return Ok(Some(SlopeLabel {
                front: None,
                upper_negative: None,
                minimum: 0.,
                upper: None,
                bounded_cost: None,
                raw_upper: None,
                machine_history: Some(Arc::new(next_history)),
            }));
        }
        let history = if let Some(history) = &old.machine_history {
            history.clone()
        } else {
            let root = outcome(self.machine_root(budget))?
                .ok_or_else(|| crate::incomplete("paired machine root is empty"))?;
            Arc::new(
                outcome(MachinePathHistory::from_root(&root, usize::MAX, budget))?
                    .ok_or_else(|| crate::incomplete("paired machine root history is empty"))?,
            )
        };
        let (reduced, next) = outcome(prepared.machine_transition(
            history.last().front(),
            &input.continuation_inputs(),
            self.machine_mode(),
            budget,
        ))?
        .ok_or_else(|| crate::incomplete("paired machine transition rejected"))?;
        let front = next.front().clone();
        let next_history = outcome(history.append(arc, &reduced, &next, budget))?
            .ok_or_else(|| crate::incomplete("paired machine history rejected"))?;
        let objective = if self.mode == PairedResourceMode::Feasibility || scale == 0. {
            0.
        } else {
            self.mapped_machine_objective(&next_history, source_scale, scale, budget)?
        };
        Ok(Some(SlopeLabel {
            front: Some(front),
            upper_negative: None,
            minimum: finite(objective)?,
            upper: None,
            bounded_cost: None,
            raw_upper: None,
            machine_history: Some(Arc::new(next_history)),
        }))
    }

    fn same_machine_constant_front(source: &SlopeFront, machine: &SlopeFront) -> bool {
        Self::machine_constant_value(source).is_some()
            && Self::machine_constant_value(machine).is_some()
            && source.components().len() == machine.components().len()
            && source
                .components()
                .iter()
                .zip(machine.components())
                .all(|(a, b)| {
                    a.pairs().len() == b.pairs().len()
                        && a.pairs().iter().zip(b.pairs()).all(|(a, b)| {
                            a.0.to_bits() == b.0.to_bits() && a.1.to_bits() == b.1.to_bits()
                        })
                })
    }

    /// Establish compatibility rather than interpreting a raw key as an
    /// objective. Replay the declared source program and require identical
    /// constant fronts at every retained machine state, including the root.
    fn mapped_machine_objective(
        &self,
        history: &MachinePathHistory,
        source_scale: f64,
        weighted_scale: f64,
        budget: FrontierBudget,
    ) -> Result<f64> {
        let unavailable =
            || crate::incomplete("paired machine objective normalization is unavailable");
        if weighted_scale < 0.
            || !self.monotone_constant_transfers
            || self.waiting_windows.is_some()
        {
            return Err(unavailable());
        }
        let started = std::time::Instant::now();
        let snapshots = outcome(history.snapshots(budget))?.ok_or_else(unavailable)?;
        if snapshots.len().saturating_mul(128) > budget.max_operations {
            return Err(crate::incomplete(
                "paired machine objective replay work limit",
            ));
        }
        let same = |source: &SlopeLabel, machine: &SlopeFront| -> bool {
            let Some(source) = &source.front else {
                return false;
            };
            Self::same_machine_constant_front(source, machine)
        };
        let mut source = self.root(true, false);
        for (index, snapshot) in snapshots.into_iter().enumerate() {
            if budget
                .max_elapsed
                .is_some_and(|limit| started.elapsed() >= limit)
            {
                return Err(crate::incomplete(
                    "paired machine objective replay deadline",
                ));
            }
            match snapshot {
                descriptor::MachinePathSnapshot::Root(_) if index == 0 => {}
                descriptor::MachinePathSnapshot::Forward { arc, .. } if index != 0 => {
                    source = self
                        .extend(&source, *arc, true, false, source_scale, false)?
                        .ok_or_else(unavailable)?;
                }
                _ => return Err(unavailable()),
            }
            if !same(&source, snapshot.front()) {
                return Err(unavailable());
            }
        }
        finite(source.minimum)
    }
    pub(super) fn initial_minimum(&self) -> f64 {
        self.initial_minimum
    }
    pub(super) fn final_objective(&self, value: f64, scale: f64) -> Result<f64> {
        if !self.has_costs() {
            return Ok(0.);
        }
        self.normalization
            .charge_selected(
                self.initial_minimum,
                value,
                finite(scale * self.objective_coefficient)?,
            )
            .map_err(|error| invalid(error.to_string()))
    }

    pub(super) fn extend(
        &self,
        old: &SlopeLabel,
        arc: usize,
        forward: bool,
        first_backward: bool,
        scale: f64,
        machine_domain: bool,
    ) -> Result<Option<SlopeLabel>> {
        let source_scale = scale;
        let scale = finite(scale * self.objective_coefficient)?;
        let signed = self.has_costs() && scale < 0.;
        if machine_domain && self.has_machine_program_input() {
            return self.extend_machine(old, arc, forward, scale, source_scale);
        }
        if !forward {
            let mut next = old.clone();
            // A numerical adjoint never rejects a suffix or certifies a join.
            // Keep the independently checked cache and ordered replay guards.
            next.raw_upper = Some(Arc::new(RawAdjoint {
                depth: old
                    .raw_upper
                    .as_ref()
                    .map_or(Some(1), |old| old.depth.checked_add(1))
                    .ok_or_else(|| invalid("paired suffix length overflow"))?,
                arc: self.adjoints[arc].clone(),
                previous: old.raw_upper.clone(),
                terminal: first_backward,
                outcome: OnceLock::new(),
            }));
            next.bounded_cost = (if let Some(lattice) = &self.cost_lattice {
                lattice.prepend(old.bounded_cost.as_deref(), arc, first_backward)
            } else if self.exact_cost {
                self.backward[arc].as_ref().and_then(|prepared| {
                    if let Some(old) = &old.bounded_cost {
                        old.checked_prepend(prepared)
                    } else if first_backward {
                        BoundedCostFront::from_terminal_arc(prepared, self.ceiling)
                    } else {
                        None
                    }
                })
            } else {
                None
            })
            .map(Box::new);
            // The numerical suffix minimum does not certify a lower bound
            // for arbitrary fractional prefixes under literal rounding.
            next.minimum = 0.;
            next.upper = (self.mode == PairedResourceMode::Feasibility)
                .then(|| self.backward[arc].as_ref())
                .flatten()
                .and_then(|prepared| {
                    let root;
                    let upper = if let Some(upper) = old.upper.as_ref() {
                        upper
                    } else if first_backward {
                        let arc = prepared.arc();
                        let lo = arc.lower.prepared_domain();
                        let hi = arc.upper.prepared_domain();
                        let (lower, upper) = (lo.lower.max(hi.lower), lo.upper.min(hi.upper));
                        if lower > upper || arc.upper.points.iter().any(|p| p[1] > 1e12) {
                            return None;
                        }
                        root = RelationFront::unbounded_upper(lower, upper);
                        &root
                    } else {
                        return None;
                    };
                    upper
                        .extend_backward_upper_prepared(prepared)
                        .ok()
                        .flatten()
                });
            return Ok(Some(next));
        }
        let front = old
            .front
            .as_ref()
            .ok_or_else(|| invalid("missing raw paired forward state"))?;
        let extension = if let Some(windows) = &self.waiting_windows {
            let window = windows[arc];
            self.arcs[arc].extend_with_machine_waiting_bounds(
                front,
                window.lower,
                window.upper,
                FrontierBudget::default(),
            )
        } else {
            self.arcs[arc].extend(front, FrontierBudget::default())
        };
        let Some(front) = outcome(extension)? else {
            return Ok(None);
        };
        let Some(witness) = outcome(front.minimum(FrontierBudget::default()))? else {
            return Ok(None);
        };
        let upper_negative = self.upper_arcs.as_ref().and_then(|arcs| {
            match negated_upper_on(&front, &arcs[arc], self.policy, FrontierBudget::default()) {
                FrontierOutcome::Complete(front) => Some(front),
                FrontierOutcome::Empty | FrontierOutcome::Unknown(_) => None,
            }
        });
        if signed && upper_negative.is_none() {
            return Err(crate::incomplete(
                "signed raw paired upper frontier is unavailable",
            ));
        }
        let selected = if signed {
            let witness = outcome(
                upper_negative
                    .as_ref()
                    .expect("checked signed envelope")
                    .minimum(FrontierBudget::default()),
            )?
            .ok_or_else(|| invalid("empty signed raw paired upper frontier"))?;
            -witness.value
        } else {
            witness.value
        };
        let objective = if self.mode == PairedResourceMode::Feasibility {
            0.
        } else {
            self.normalization
                .charge_selected(self.initial_minimum, selected, scale)
                .map_err(|error| invalid(error.to_string()))?
        };
        Ok(Some(SlopeLabel {
            front: Some(front),
            upper_negative,
            minimum: finite(objective)?,
            upper: None,
            bounded_cost: None,
            raw_upper: None,
            machine_history: None,
        }))
    }
}
impl CoupledLabelState {
    /// General numerical backward upper fronts in raw paired declaration
    /// order. None denotes a forward state or unknown/empty adjoint, not a
    /// source infeasibility proof. Joins retain their ordered replay contract.
    pub fn slope_relation_backward_fronts(&self) -> impl Iterator<Item = Option<&SlopeFront>> {
        self.slope_relations.iter().map(|label| {
            match label.raw_upper.as_deref().map(RawAdjoint::outcome) {
                Some(FrontierOutcome::Complete(front)) => Some(front),
                _ => None,
            }
        })
    }
    /// Numerical backward outcomes, including an empty front or the reason a
    /// bounded computation could not complete. These are diagnostic results,
    /// not certificates of source infeasibility or safe dominance.
    pub fn slope_relation_backward_outcomes(
        &self,
    ) -> impl Iterator<Item = Option<&FrontierOutcome<SlopeFront>>> {
        self.slope_relations
            .iter()
            .map(|label| label.raw_upper.as_deref().map(RawAdjoint::outcome))
    }
    /// Recompute each numerical backward outcome using a caller-supplied
    /// budget, without reusing a previously cached unknown. Each relation
    /// receives its own budget, shared conservatively across its suffix arcs.
    /// These results remain advisory and do not certify source feasibility.
    pub fn slope_relation_backward_outcomes_with_budget(
        &self,
        budget: FrontierBudget,
    ) -> impl Iterator<Item = Option<FrontierOutcome<SlopeFront>>> + '_ {
        self.slope_relations.iter().map(move |label| {
            label
                .raw_upper
                .as_deref()
                .map(|upper| upper.recompute(budget))
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
