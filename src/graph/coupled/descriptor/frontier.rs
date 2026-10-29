//! Slope FRONT algebra; deliberately distinct from source-expression replay.
use super::{
    EvaluationPolicy, PreparedSlopeFunction, PreparedSlopeRelationArc, RawAffine, RawFunctionError,
};
use std::{
    sync::Arc,
    time::{Duration, Instant},
};

/// Source evaluation policy does not implicitly select an operation policy.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "clp", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "clp", serde(rename_all = "snake_case"))]
pub enum FrontOperationPolicy {
    /// Exact domain intersection and exact coordinate comparisons.
    StrictDomain,
    /// Native left-coordinate choices at near-aligned segment ends and starts.
    FrontAlgebra,
}
#[derive(Clone, Copy, Debug)]
pub struct FrontierBudget {
    pub max_operations: usize,
    pub max_components: usize,
    pub max_pairs: usize,
    pub max_elapsed: Option<Duration>,
}
impl Default for FrontierBudget {
    fn default() -> Self {
        Self {
            max_operations: 1_000_000,
            max_components: 4096,
            max_pairs: 4096,
            max_elapsed: None,
        }
    }
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FrontierUnknown {
    WorkLimit,
    ComponentLimit,
    PairLimit,
    Deadline,
    Arithmetic(RawFunctionError),
    Unsupported(&'static str),
}
#[derive(Clone, Debug)]
pub enum FrontierOutcome<T> {
    Complete(T),
    Empty,
    Unknown(FrontierUnknown),
}
/// Persistent provenance of performed front operations, not literal replay.
#[derive(Clone, Debug)]
pub enum FrontierProvenance {
    /// Copied machine root array. This is not a source-control witness.
    MachineRoot {
        input: Arc<PreparedSlopeFunction>,
    },
    /// Numerical backward paired adjoint, retaining its evaluated input
    /// frontier rather than a recursive suffix-history chain. This is not a
    /// source witness or a forward machine-program certificate.
    PairedAdjoint {
        input: Arc<[PreparedSlopeFunction]>,
        arc: Arc<super::SlopeRelationArc>,
        policy: FrontOperationPolicy,
    },
    /// Bounded machine waiting path, retaining the reference predecessor bit.
    /// This is not a source-control witness or directed certificate.
    PairedMachineWaiting {
        input: Arc<Self>,
        arc: Arc<super::SlopeRelationArc>,
        predecessor: Option<f64>,
        waiting_bits: u64,
    },
    /// Machine-only reduction before continuation. The input
    /// array and provenance remain available for later reconstruction; this
    /// operation is not a source-control equivalence certificate.
    MachineReduction {
        input: Arc<Self>,
        input_front: Arc<[PreparedSlopeFunction]>,
        merged: bool,
    },
    ClampFilter {
        input: Arc<Self>,
        lower: Arc<PreparedSlopeFunction>,
        upper: Arc<PreparedSlopeFunction>,
        policy: FrontOperationPolicy,
    },
    /// Paired transition retains the original raw descriptors and waiting rate.
    Paired {
        input: Arc<Self>,
        arc: Arc<super::SlopeRelationArc>,
        policy: FrontOperationPolicy,
        intermediate_tolerant_alignment: bool,
    },
    Source(Arc<[PreparedSlopeFunction]>),
    Addition {
        left: Arc<Self>,
        right: Arc<Self>,
        policy: FrontOperationPolicy,
    },
    Translation {
        input: Arc<Self>,
        delta: f64,
    },
    Restriction {
        input: Arc<Self>,
        lower: f64,
        upper: f64,
        policy: FrontOperationPolicy,
    },
}
#[derive(Clone, Debug)]
pub struct SlopeFront {
    components: Arc<[PreparedSlopeFunction]>,
    provenance: Arc<FrontierProvenance>,
    tolerant_alignment: bool,
}

/// Initial state of the paired machine front. Its key and predecessor are the
/// literal root fields, before any arc-extension reset or minimum scan.
/// It does not establish source feasibility or a pruning certificate.
#[derive(Clone, Debug)]
pub struct MachineRootState {
    front: SlopeFront,
    waiting_bits: u64,
    predecessor: f64,
    key: f64,
}

/// Explicit encoding of the dependent input to the machine root preparation.
/// A singleton pair is affine parameters, whereas a vector is an already
/// encoded slope function; the two encodings must not be inferred from one
/// another.
#[derive(Clone, Copy, Debug)]
pub enum MachineRootInput<'a> {
    /// The singleton `(intercept, slope)` encoding.
    Affine(RawAffine),
    /// The two-to-eleven-pair slope-vector encoding.
    Vector(&'a PreparedSlopeFunction),
}

impl MachineRootState {
    /// Prepare the backward machine root from already prepared upper vectors
    /// of selected sink arcs and explicitly supplied terminal bounds. This
    /// does not verify arc selection, source feasibility, or the reference
    /// initializer's append destination.
    pub fn from_backward_inputs(
        upper_vectors: &[&PreparedSlopeFunction],
        lower: f64,
        upper: f64,
        budget: FrontierBudget,
    ) -> FrontierOutcome<Self> {
        let started = Instant::now();
        if budget.max_components == 0 {
            return FrontierOutcome::Unknown(FrontierUnknown::ComponentLimit);
        }
        if budget.max_pairs < 2 {
            return FrontierOutcome::Unknown(FrontierUnknown::PairLimit);
        }
        if !lower.is_finite() || !upper.is_finite() || lower > upper {
            return FrontierOutcome::Unknown(FrontierUnknown::Unsupported(
                "machine backward root bounds",
            ));
        }
        let mut work = Work::new(budget);
        let mut ceiling = -1e12;
        for function in upper_vectors {
            let pairs = function.pairs();
            if pairs.is_empty() || pairs.len() > 12 {
                return FrontierOutcome::Unknown(FrontierUnknown::Unsupported(
                    "machine backward root upper vector",
                ));
            }
            if pairs.len() > budget.max_pairs {
                return FrontierOutcome::Unknown(FrontierUnknown::PairLimit);
            }
            if let Err(reason) = work.charge(pairs.len().saturating_mul(4)) {
                return FrontierOutcome::Unknown(reason);
            }
            let (mut value, mut coordinate) = pairs[0];
            let mut maximum = value;
            for &(slope, end) in &pairs[1..] {
                value = match checked(end - coordinate, "backward root span")
                    .and_then(|span| checked(span * slope, "backward root multiplication"))
                    .and_then(|product| checked(value + product, "backward root accumulation"))
                {
                    Ok(value) => value,
                    Err(error) => {
                        return FrontierOutcome::Unknown(FrontierUnknown::Arithmetic(error))
                    }
                };
                if value > maximum {
                    maximum = value;
                }
                coordinate = end;
            }
            if maximum > ceiling {
                ceiling = maximum;
            }
        }
        let vector =
            match PreparedSlopeFunction::from_machine_pairs(vec![(ceiling, lower), (0., upper)]) {
                Ok(vector) => vector,
                Err(error) => return FrontierOutcome::Unknown(FrontierUnknown::Arithmetic(error)),
            };
        let remaining_work = match budget.max_operations.checked_sub(work.operations) {
            Some(remaining) => remaining,
            None => return FrontierOutcome::Unknown(FrontierUnknown::WorkLimit),
        };
        let remaining_time = budget
            .max_elapsed
            .map(|limit| limit.saturating_sub(started.elapsed()));
        if remaining_time.is_some_and(|limit| limit.is_zero()) {
            return FrontierOutcome::Unknown(FrontierUnknown::Deadline);
        }
        match Self::from_prepared(
            &vector,
            FrontierBudget {
                max_operations: remaining_work,
                max_elapsed: remaining_time,
                ..budget
            },
        ) {
            FrontierOutcome::Complete(mut state) => {
                state.waiting_bits = 0;
                state.predecessor = 1e12;
                state.key = ceiling;
                FrontierOutcome::Complete(state)
            }
            FrontierOutcome::Empty => FrontierOutcome::Empty,
            FrontierOutcome::Unknown(reason) => FrontierOutcome::Unknown(reason),
        }
    }

    /// Prepare an explicit machine root input in an initially empty output.
    ///
    /// The supplied bounds and input correspond to the reference initializer's
    /// arguments; their graph origin, the subsequent domain check, and source
    /// feasibility are not established by this operation. It returns unknown
    /// for unsupported truncation and arithmetic, not source infeasibility.
    /// The initializer's failure-side effects for a nonempty output are not
    /// modeled. Work and elapsed limits are shared by preparation and copying.
    pub fn from_input(
        input: MachineRootInput<'_>,
        lower: f64,
        upper: f64,
        budget: FrontierBudget,
    ) -> FrontierOutcome<Self> {
        let started = Instant::now();
        if budget.max_components == 0 {
            return FrontierOutcome::Unknown(FrontierUnknown::ComponentLimit);
        }
        let (count, work) = match input {
            MachineRootInput::Affine(_) => (2, Some(8usize)),
            MachineRootInput::Vector(vector) => {
                let count = vector.pairs().len();
                if !(2..=11).contains(&count) {
                    return FrontierOutcome::Unknown(FrontierUnknown::Unsupported(
                        "machine root source vector pair count",
                    ));
                }
                (
                    count,
                    count.checked_mul(16).and_then(|work| work.checked_add(32)),
                )
            }
        };
        if count > budget.max_pairs {
            return FrontierOutcome::Unknown(FrontierUnknown::PairLimit);
        }
        let Some(work) = work else {
            return FrontierOutcome::Unknown(FrontierUnknown::WorkLimit);
        };
        let Some(remaining_work) = budget.max_operations.checked_sub(work) else {
            return FrontierOutcome::Unknown(FrontierUnknown::WorkLimit);
        };
        if budget
            .max_elapsed
            .is_some_and(|limit| started.elapsed() >= limit)
        {
            return FrontierOutcome::Unknown(FrontierUnknown::Deadline);
        }
        let prepared = match input {
            MachineRootInput::Affine(affine) => affine.prepare_machine(lower, upper),
            MachineRootInput::Vector(vector) => vector.machine_truncate(lower, upper),
        };
        let prepared = match prepared {
            Ok(prepared) => prepared,
            Err(error) => return FrontierOutcome::Unknown(FrontierUnknown::Arithmetic(error)),
        };
        let remaining_time = budget
            .max_elapsed
            .map(|limit| limit.saturating_sub(started.elapsed()));
        if remaining_time.is_some_and(|time| time.is_zero()) {
            return FrontierOutcome::Unknown(FrontierUnknown::Deadline);
        }
        Self::from_prepared(
            &prepared,
            FrontierBudget {
                max_operations: remaining_work,
                max_elapsed: remaining_time,
                ..budget
            },
        )
    }

    /// Copy one finite prepared vector into the bounded machine root state.
    /// This does not construct the reference's upstream initial vector.
    pub fn from_prepared(
        input: &PreparedSlopeFunction,
        budget: FrontierBudget,
    ) -> FrontierOutcome<Self> {
        let mut work = Work::new(budget);
        let count = input.pairs().len();
        if count == 0 || count > 12 {
            return FrontierOutcome::Unknown(FrontierUnknown::Unsupported(
                "machine root pair count",
            ));
        }
        if budget.max_components == 0 {
            return FrontierOutcome::Unknown(FrontierUnknown::ComponentLimit);
        }
        if count > budget.max_pairs {
            return FrontierOutcome::Unknown(FrontierUnknown::PairLimit);
        }
        if let Err(reason) = work.charge(count) {
            return FrontierOutcome::Unknown(reason);
        }
        if input
            .pairs()
            .iter()
            .any(|&(a, b)| !a.is_finite() || !b.is_finite())
        {
            return FrontierOutcome::Unknown(FrontierUnknown::Unsupported(
                "machine root nonfinite pair",
            ));
        }
        let input = Arc::new(input.clone());
        let segments = count - 1;
        let waiting_bits = ((1u64 << (segments + 1)) - 1) & !1;
        let front = SlopeFront {
            components: vec![input.as_ref().clone()].into(),
            provenance: Arc::new(FrontierProvenance::MachineRoot { input }),
            tolerant_alignment: false,
        };
        if let Err(reason) = work.charge(0) {
            return FrontierOutcome::Unknown(reason);
        }
        FrontierOutcome::Complete(Self {
            front,
            waiting_bits,
            predecessor: 0.0,
            key: 0.0,
        })
    }

    /// Copied machine front and its machine-only provenance.
    pub fn front(&self) -> &SlopeFront {
        &self.front
    }
    /// Forward roots set bits 1 through the segment count; backward roots
    /// start with all waiting bits clear.
    pub fn waiting_bits(&self) -> u64 {
        self.waiting_bits
    }
    /// Initial predecessor: positive zero forward, the sentinel backward.
    pub fn predecessor(&self) -> f64 {
        self.predecessor
    }
    /// Initial key: positive zero forward, the terminal ceiling backward.
    pub fn key(&self) -> f64 {
        self.key
    }
}
/// Result of the front algebra's ordered endpoint scan. Re-evaluation under
/// a source function's tolerance/snap policy is a different operation and is
/// not asserted to reproduce this value or certify a source-program minimum.
#[derive(Clone, Debug)]
pub struct FrontAlgebraWitness {
    pub coordinate: f64,
    pub value: f64,
    pub component: usize,
    pub provenance: Arc<FrontierProvenance>,
}

/// Recovered intermediate machine waiting output. It has not passed the
/// subsequent bit-aware maximum and upper-bound operations.
#[derive(Clone, Debug)]
pub struct MachineWaitingEnvelope {
    function: PreparedSlopeFunction,
    predecessor: Option<f64>,
    waiting_bits: u64,
}

/// A bounded forward machine state after a successful paired transition.
/// Its raw key and provenance do not certify source-control feasibility or
/// supply a directed pricing bound.
#[derive(Clone, Debug)]
pub struct MachineForwardState {
    front: SlopeFront,
    waiting_bits: u64,
    predecessor: Option<f64>,
    key: f64,
}

/// A copied machine state before the next arc's in-place reduction. These
/// snapshots do not certify source feasibility or a route.
#[derive(Clone, Debug)]
pub enum MachinePathSnapshot {
    Root(MachineRootState),
    Forward {
        arc: usize,
        state: MachineForwardState,
    },
    Backward {
        arc: usize,
        front: SlopeFront,
        waiting_bits: u64,
        key: f64,
        ceiling: f64,
    },
}

impl MachinePathSnapshot {
    /// Unreduced front stored at this position in the history.
    pub fn front(&self) -> &SlopeFront {
        match self {
            Self::Root(root) => root.front(),
            Self::Forward { state, .. } => state.front(),
            Self::Backward { front, .. } => front,
        }
    }
}

#[derive(Clone, Debug)]
struct MachinePathNode {
    previous: Option<Arc<MachinePathNode>>,
    snapshot: MachinePathSnapshot,
}

impl Drop for MachinePathNode {
    fn drop(&mut self) {
        // Release uniquely owned prefixes iteratively, without recursion for
        // long bounded walks. A shared prefix stays alive for its other owner.
        let mut previous = self.previous.take();
        while let Some(mut node) = previous {
            previous = match Arc::get_mut(&mut node) {
                Some(unique) => unique.previous.take(),
                None => break,
            };
        }
    }
}

/// Persistent bounded machine snapshots. Cloned histories share immutable
/// nodes, so a later reduction cannot modify an earlier state. This is not a
/// pricing label or route certificate.
#[derive(Clone, Debug)]
pub struct MachinePathHistory {
    head: Arc<MachinePathNode>,
    len: usize,
    max_states: usize,
}

impl MachinePathHistory {
    /// Compare retained backward feasibility fronts under machine tolerance.
    /// Acceptance is not strict source-domain containment.
    pub fn backward_feasibility_comparison(
        &self,
        other: &Self,
        budget: FrontierBudget,
    ) -> FrontierOutcome<()> {
        if self.backward_ceiling().is_none() || other.backward_ceiling().is_none() {
            return FrontierOutcome::Unknown(FrontierUnknown::Unsupported(
                "backward comparison requires backward histories",
            ));
        }
        let left = self.last().front().components();
        let right = other.last().front().components();
        if left.len() != 1 || right.len() != 1 {
            return FrontierOutcome::Unknown(FrontierUnknown::Unsupported(
                "backward comparison requires single machine fronts",
            ));
        }
        PreparedSlopeRelationArc::machine_backward_feasibility_dominance(
            &left[0], &right[0], budget,
        )
    }

    /// Compare the retained backward fronts using the ordered machine scan.
    /// Returns the signed updated accumulator, rejection, or unknown. This
    /// does not certify source feasibility or normalized route-cost order.
    pub fn backward_cost_comparison(
        &self,
        other: &Self,
        mode: i32,
        accumulated: f64,
        budget: FrontierBudget,
    ) -> FrontierOutcome<f64> {
        if self.backward_ceiling().is_none() || other.backward_ceiling().is_none() {
            return FrontierOutcome::Unknown(FrontierUnknown::Unsupported(
                "backward comparison requires backward histories",
            ));
        }
        let left = self.last().front().components();
        let right = other.last().front().components();
        if left.len() != 1 || right.len() != 1 {
            return FrontierOutcome::Unknown(FrontierUnknown::Unsupported(
                "backward comparison requires single machine fronts",
            ));
        }
        PreparedSlopeRelationArc::machine_backward_cost_dominance(
            &left[0],
            &right[0],
            mode,
            accumulated,
            budget,
        )
    }

    /// Immutable root identity, used to reject histories from replaced arc
    /// inputs or a different installed terminal root. Caller budgets traversal.
    pub(crate) fn shares_root(&self, root: &Self) -> bool {
        if root.len != 1 {
            return false;
        }
        let mut node = &self.head;
        while let Some(previous) = &node.previous {
            node = previous;
        }
        Arc::ptr_eq(node, &root.head)
    }

    /// Terminal ceiling retained by a backward machine history.
    pub(crate) fn backward_ceiling(&self) -> Option<f64> {
        match &self.head.snapshot {
            MachinePathSnapshot::Root(root)
                if root.predecessor().to_bits() == 1e12f64.to_bits() =>
            {
                Some(root.key())
            }
            MachinePathSnapshot::Root(_) => None,
            MachinePathSnapshot::Backward { ceiling, .. } => Some(*ceiling),
            MachinePathSnapshot::Forward { .. } => None,
        }
    }

    /// Append a machine-only backward transition. This internal operation
    /// carries no source-domain feasibility or objective certificate.
    pub(crate) fn append_backward(
        &self,
        arc: usize,
        front: PreparedSlopeFunction,
        waiting_bits: u64,
        key: f64,
        budget: FrontierBudget,
    ) -> FrontierOutcome<Self> {
        let Some(ceiling) = self.backward_ceiling() else {
            return FrontierOutcome::Unknown(FrontierUnknown::Unsupported(
                "forward history cannot append backward state",
            ));
        };
        let Some(next_len) = self.len.checked_add(1) else {
            return FrontierOutcome::Unknown(FrontierUnknown::ComponentLimit);
        };
        if next_len > self.max_states || budget.max_components == 0 {
            return FrontierOutcome::Unknown(FrontierUnknown::ComponentLimit);
        }
        if front.pairs().len() > budget.max_pairs {
            return FrontierOutcome::Unknown(FrontierUnknown::PairLimit);
        }
        let mut work = Work::new(budget);
        if let Err(reason) = work.charge(front.pairs().len().saturating_add(1)) {
            return FrontierOutcome::Unknown(reason);
        }
        if !key.is_finite() {
            return FrontierOutcome::Unknown(FrontierUnknown::Unsupported(
                "nonfinite backward machine key",
            ));
        }
        FrontierOutcome::Complete(Self {
            head: Arc::new(MachinePathNode {
                previous: Some(self.head.clone()),
                snapshot: MachinePathSnapshot::Backward {
                    arc,
                    front: SlopeFront::from_prepared(front),
                    waiting_bits,
                    key,
                    ceiling,
                },
            }),
            len: next_len,
            max_states: self.max_states,
        })
    }
    /// Start with an unreduced root. The explicit limit includes the root.
    pub fn from_root(
        root: &MachineRootState,
        max_states: usize,
        budget: FrontierBudget,
    ) -> FrontierOutcome<Self> {
        if max_states == 0 || budget.max_components == 0 {
            return FrontierOutcome::Unknown(FrontierUnknown::ComponentLimit);
        }
        let pairs = root
            .front
            .components
            .iter()
            .map(|f| f.pairs().len())
            .sum::<usize>();
        if pairs > budget.max_pairs {
            return FrontierOutcome::Unknown(FrontierUnknown::PairLimit);
        }
        let mut work = Work::new(budget);
        if let Err(reason) = work.charge(pairs.saturating_add(1)) {
            return FrontierOutcome::Unknown(reason);
        }
        FrontierOutcome::Complete(Self {
            head: Arc::new(MachinePathNode {
                previous: None,
                snapshot: MachinePathSnapshot::Root(root.clone()),
            }),
            len: 1,
            max_states,
        })
    }

    /// Append a successful transition. The reduction must come from the
    /// latest snapshot and the new state must reference that reduction. Arc
    /// identity is caller supplied and is not independently certified.
    pub fn append(
        &self,
        arc: usize,
        reduced: &SlopeFront,
        state: &MachineForwardState,
        budget: FrontierBudget,
    ) -> FrontierOutcome<Self> {
        let Some(next_len) = self.len.checked_add(1) else {
            return FrontierOutcome::Unknown(FrontierUnknown::ComponentLimit);
        };
        if next_len > self.max_states || budget.max_components == 0 {
            return FrontierOutcome::Unknown(FrontierUnknown::ComponentLimit);
        }
        let previous = self.head.snapshot.front();
        match reduced.provenance.as_ref() {
            FrontierProvenance::MachineReduction {
                input, input_front, ..
            } if Arc::ptr_eq(input, &previous.provenance)
                && Arc::ptr_eq(input_front, &previous.components) => {}
            _ => {
                return FrontierOutcome::Unknown(FrontierUnknown::Unsupported(
                    "machine history reduction does not match prior state",
                ))
            }
        }
        match state.front.provenance.as_ref() {
            FrontierProvenance::PairedMachineWaiting { input, .. }
                if Arc::ptr_eq(input, &reduced.provenance) => {}
            _ => {
                return FrontierOutcome::Unknown(FrontierUnknown::Unsupported(
                    "machine history state does not match reduction",
                ))
            }
        }
        let pairs = state
            .front
            .components
            .iter()
            .map(|f| f.pairs().len())
            .sum::<usize>();
        if pairs > budget.max_pairs {
            return FrontierOutcome::Unknown(FrontierUnknown::PairLimit);
        }
        let mut work = Work::new(budget);
        if let Err(reason) = work.charge(pairs.saturating_add(1)) {
            return FrontierOutcome::Unknown(reason);
        }
        FrontierOutcome::Complete(Self {
            head: Arc::new(MachinePathNode {
                previous: Some(self.head.clone()),
                snapshot: MachinePathSnapshot::Forward {
                    arc,
                    state: state.clone(),
                },
            }),
            len: next_len,
            max_states: self.max_states,
        })
    }

    /// Number of stored snapshots, including the root.
    pub fn len(&self) -> usize {
        self.len
    }
    /// Histories always contain their root.
    pub fn is_empty(&self) -> bool {
        false
    }
    /// Latest snapshot, before any subsequent reduction.
    pub fn last(&self) -> &MachinePathSnapshot {
        &self.head.snapshot
    }

    /// Materialize borrowed snapshots in root-to-last order. The output
    /// component and work limits apply to the whole traversal.
    pub fn snapshots(&self, budget: FrontierBudget) -> FrontierOutcome<Vec<&MachinePathSnapshot>> {
        if self.len > budget.max_components {
            return FrontierOutcome::Unknown(FrontierUnknown::ComponentLimit);
        }
        let mut work = Work::new(budget);
        let mut result = Vec::with_capacity(self.len);
        let mut node = Some(self.head.as_ref());
        while let Some(current) = node {
            if let Err(reason) = work.charge(1) {
                return FrontierOutcome::Unknown(reason);
            }
            result.push(&current.snapshot);
            node = current.previous.as_deref();
        }
        result.reverse();
        if let Err(reason) = work.charge(0) {
            return FrontierOutcome::Unknown(reason);
        }
        FrontierOutcome::Complete(result)
    }
}

impl MachineForwardState {
    /// Construct a machine state from a successful, already filtered waiting
    /// envelope. `reduced` retains the input front's machine-reduction history
    /// and `arc` identifies the transition. The caller must supply an envelope
    /// produced by that transition after its lower and upper filters succeed.
    /// This operation does not verify the envelope's association with the arc.
    /// A missing predecessor means the recurrence did not write that field;
    /// the modern wrapper initializes it to `1e12` before the recurrence.
    pub fn from_filtered_waiting(
        reduced: &SlopeFront,
        envelope: &MachineWaitingEnvelope,
        arc: Arc<super::SlopeRelationArc>,
        mode: i32,
        budget: FrontierBudget,
    ) -> FrontierOutcome<Self> {
        let mut work = Work::new(budget);
        if !matches!(
            reduced.provenance().as_ref(),
            FrontierProvenance::MachineReduction { .. }
        ) {
            return FrontierOutcome::Unknown(FrontierUnknown::Unsupported(
                "machine forward state requires reduction provenance",
            ));
        }
        let function = &envelope.function;
        let pairs = function.pairs();
        // The wrapper scans at most eleven segments in its twelve-pair
        // fixed array. Ordinary prepared functions have strictly increasing
        // coordinates; machine-only duplicates retain their narrower proven
        // representations. This does not validate the preceding filters.
        let ordinary = !function.is_machine_zero_length() && (2..=12).contains(&pairs.len());
        let duplicate = pairs.len() == 2 && function.is_machine_zero_length()
            || pairs.len() == 3 && function.has_internal_zero_length();
        if !ordinary && !duplicate {
            return FrontierOutcome::Unknown(FrontierUnknown::Unsupported(
                "unrecovered machine forward state geometry",
            ));
        }
        if budget.max_components == 0 {
            return FrontierOutcome::Unknown(FrontierUnknown::ComponentLimit);
        }
        if pairs.len() > budget.max_pairs {
            return FrontierOutcome::Unknown(FrontierUnknown::PairLimit);
        }
        let valid_bits = ((1u64 << pairs.len()) - 1) & !1;
        if envelope.waiting_bits & !valid_bits != 0
            || envelope.predecessor.is_some_and(|value| !value.is_finite())
            || pairs.iter().any(|&(a, b)| !a.is_finite() || !b.is_finite())
        {
            return FrontierOutcome::Unknown(FrontierUnknown::Unsupported(
                "invalid machine forward state fields",
            ));
        }
        let operations = if mode == 4 {
            1
        } else {
            1 + 3 * (pairs.len() - 1)
        };
        if let Err(reason) = work.charge(operations) {
            return FrontierOutcome::Unknown(reason);
        }
        let key = if mode == 4 {
            1e12
        } else {
            let (mut value, mut coordinate) = pairs[0];
            let mut minimum = value;
            for &(slope, endpoint) in &pairs[1..] {
                value = match checked(endpoint - coordinate, "machine key span")
                    .and_then(|span| checked(span * slope, "machine key multiplication"))
                    .and_then(|product| checked(value + product, "machine key accumulation"))
                {
                    Ok(value) => value,
                    Err(error) => {
                        return FrontierOutcome::Unknown(FrontierUnknown::Arithmetic(error))
                    }
                };
                // MINSD selects the old minimum on equality, including zeros.
                if value < minimum {
                    minimum = value;
                }
                coordinate = endpoint;
            }
            minimum
        };
        if let Err(reason) = work.charge(0) {
            return FrontierOutcome::Unknown(reason);
        }
        let front = SlopeFront {
            components: vec![function.clone()].into(),
            provenance: Arc::new(FrontierProvenance::PairedMachineWaiting {
                input: reduced.provenance().clone(),
                arc,
                predecessor: envelope.predecessor,
                waiting_bits: envelope.waiting_bits,
            }),
            tolerant_alignment: true,
        };
        FrontierOutcome::Complete(Self {
            front,
            waiting_bits: envelope.waiting_bits,
            predecessor: envelope.predecessor,
            key,
        })
    }

    /// Resulting machine front, retaining the reduction history.
    pub fn front(&self) -> &SlopeFront {
        &self.front
    }
    /// Reference waiting provenance bitset.
    pub fn waiting_bits(&self) -> u64 {
        self.waiting_bits
    }
    /// The predecessor written by the recurrence, if any.
    pub fn predecessor(&self) -> Option<f64> {
        self.predecessor
    }
    /// Raw machine state key, without objective normalization.
    pub fn key(&self) -> f64 {
        self.key
    }

    /// Recover `(waiting, carried_coordinate)` for a strictly interior point
    /// of a finite ordinary multi-segment machine front, away from every knot's
    /// machine tolerance band. The carried coordinate is the value passed to
    /// the later arc's inverse; it is not reconstructed by subtracting the
    /// rounded waiting amount. This is not a source or route certificate.
    pub fn machine_multisegment_waiting_at(
        &self,
        coordinate: f64,
        budget: FrontierBudget,
    ) -> FrontierOutcome<(f64, f64)> {
        let mut work = Work::new(budget);
        if budget.max_components == 0 {
            return FrontierOutcome::Unknown(FrontierUnknown::ComponentLimit);
        }
        if self.front.components.len() != 1 || !coordinate.is_finite() {
            return FrontierOutcome::Unknown(FrontierUnknown::Unsupported(
                "machine multisegment waiting inputs",
            ));
        }
        let function = &self.front.components[0];
        let pairs = function.pairs();
        if function.is_machine_zero_length() || !(3..=12).contains(&pairs.len()) {
            return FrontierOutcome::Unknown(FrontierUnknown::Unsupported(
                "machine multisegment waiting geometry",
            ));
        }
        if pairs.len() > budget.max_pairs {
            return FrontierOutcome::Unknown(FrontierUnknown::PairLimit);
        }
        let Some(operations) = pairs.len().checked_mul(8).and_then(|n| n.checked_add(8)) else {
            return FrontierOutcome::Unknown(FrontierUnknown::WorkLimit);
        };
        if let Err(reason) = work.charge(operations) {
            return FrontierOutcome::Unknown(reason);
        }
        if coordinate <= pairs[0].1 || coordinate >= pairs[pairs.len() - 1].1 {
            return FrontierOutcome::Unknown(FrontierUnknown::Unsupported(
                "machine multisegment waiting coordinate",
            ));
        }
        for &(_, knot) in pairs {
            match near(coordinate, knot, FrontOperationPolicy::FrontAlgebra) {
                Ok(false) => {}
                Ok(true) => {
                    return FrontierOutcome::Unknown(FrontierUnknown::Unsupported(
                        "machine multisegment waiting knot alignment",
                    ));
                }
                Err(reason) => return FrontierOutcome::Unknown(reason),
            }
        }
        let valid_bits = ((1u64 << pairs.len()) - 1) & !1;
        if self.waiting_bits & !valid_bits != 0 {
            return FrontierOutcome::Unknown(FrontierUnknown::Unsupported(
                "machine multisegment waiting bits",
            ));
        }
        let segment = pairs.partition_point(|pair| pair.1 < coordinate);
        // The strict interior and not-near guards imply 1 <= segment < len.
        match function.evaluate(coordinate, EvaluationPolicy::EndpointTolerance) {
            Ok(Some(value)) if value != 1e12 => {}
            Ok(_) => {
                return FrontierOutcome::Unknown(FrontierUnknown::Unsupported(
                    "machine multisegment waiting lookup sentinel",
                ));
            }
            Err(error) => return FrontierOutcome::Unknown(FrontierUnknown::Arithmetic(error)),
        }
        let (waiting, carried) = if self.waiting_bits & (1u64 << segment) == 0 {
            (0.0, coordinate)
        } else {
            let preceding = (1..segment)
                .rev()
                .find(|&index| self.waiting_bits & (1u64 << index) == 0);
            let predecessor = match preceding {
                Some(index) => pairs[index].1,
                None => match self.predecessor {
                    Some(value) => value,
                    None => {
                        return FrontierOutcome::Unknown(FrontierUnknown::Unsupported(
                            "machine multisegment predecessor unavailable",
                        ));
                    }
                },
            };
            if !predecessor.is_finite() || predecessor == 1e12 || predecessor < 0.0 {
                return FrontierOutcome::Unknown(FrontierUnknown::Unsupported(
                    "machine multisegment predecessor rejected",
                ));
            }
            let waiting = match checked(coordinate - predecessor, "machine waiting subtraction") {
                Ok(value) => value,
                Err(error) => return FrontierOutcome::Unknown(FrontierUnknown::Arithmetic(error)),
            };
            (waiting, predecessor)
        };
        if let Err(reason) = work.charge(0) {
            return FrontierOutcome::Unknown(reason);
        }
        FrontierOutcome::Complete((waiting, carried))
    }

    /// Recover waiting for the bounded internal-duplicate front with bit 1
    /// clear and bit 2 set. This is a machine result, not a source witness.
    /// Near the first knot takes precedence even when it is also near the end.
    pub fn machine_internal_duplicate_waiting_at(
        &self,
        coordinate: f64,
        budget: FrontierBudget,
    ) -> FrontierOutcome<f64> {
        let mut work = Work::new(budget);
        if budget.max_components == 0 {
            return FrontierOutcome::Unknown(FrontierUnknown::ComponentLimit);
        }
        if budget.max_pairs < 3 {
            return FrontierOutcome::Unknown(FrontierUnknown::PairLimit);
        }
        if self.front.components.len() != 1 || !coordinate.is_finite() {
            return FrontierOutcome::Unknown(FrontierUnknown::Unsupported(
                "machine duplicate waiting reconstruction inputs",
            ));
        }
        let function = &self.front.components[0];
        let pairs = function.pairs();
        if !function.has_internal_zero_length() || pairs.len() != 3 || self.waiting_bits != 1 << 2 {
            return FrontierOutcome::Unknown(FrontierUnknown::Unsupported(
                "machine duplicate waiting reconstruction geometry",
            ));
        }
        if let Err(reason) = work.charge(10) {
            return FrontierOutcome::Unknown(reason);
        }
        let (anchor, first) = pairs[0];
        let (slope, duplicate) = pairs[1];
        let (waiting_slope, end) = pairs[2];
        let first_near = match near(coordinate, first, FrontOperationPolicy::FrontAlgebra) {
            Ok(value) => value,
            Err(reason) => return FrontierOutcome::Unknown(reason),
        };
        let (value, waiting) = if first_near {
            (anchor, 0.0)
        } else {
            if coordinate <= first {
                return FrontierOutcome::Unknown(FrontierUnknown::Unsupported(
                    "machine duplicate waiting coordinate below start",
                ));
            }
            if duplicate == 1e12 || duplicate < 0.0 {
                return FrontierOutcome::Unknown(FrontierUnknown::Unsupported(
                    "machine duplicate waiting predecessor rejected",
                ));
            }
            let endpoint = if coordinate > end {
                match near(coordinate, end, FrontOperationPolicy::FrontAlgebra) {
                    Ok(true) => end,
                    Ok(false) => {
                        return FrontierOutcome::Unknown(FrontierUnknown::Unsupported(
                            "machine duplicate waiting coordinate above end",
                        ));
                    }
                    Err(reason) => return FrontierOutcome::Unknown(reason),
                }
            } else {
                coordinate
            };
            let evaluated = checked(duplicate - first, "machine duplicate waiting span")
                .and_then(|span| checked(span * slope, "machine duplicate waiting multiplication"))
                .and_then(|product| {
                    checked(anchor + product, "machine duplicate waiting accumulation")
                })
                .and_then(|value| {
                    checked(endpoint - duplicate, "machine duplicate waiting span")
                        .and_then(|span| {
                            checked(
                                span * waiting_slope,
                                "machine duplicate waiting multiplication",
                            )
                        })
                        .and_then(|product| {
                            checked(value + product, "machine duplicate waiting accumulation")
                        })
                });
            let value = match evaluated {
                Ok(value) => value,
                Err(error) => return FrontierOutcome::Unknown(FrontierUnknown::Arithmetic(error)),
            };
            let waiting = match checked(
                coordinate - duplicate,
                "machine duplicate waiting subtraction",
            ) {
                Ok(value) => value,
                Err(error) => return FrontierOutcome::Unknown(FrontierUnknown::Arithmetic(error)),
            };
            (value, waiting)
        };
        if value == 1e12 {
            return FrontierOutcome::Unknown(FrontierUnknown::Unsupported(
                "machine duplicate waiting lookup sentinel",
            ));
        }
        if let Err(reason) = work.charge(0) {
            return FrontierOutcome::Unknown(reason);
        }
        FrontierOutcome::Complete(waiting)
    }

    /// Return the directly selected predecessor for the bounded duplicate
    /// waiting case, without recovering it by subtracting rounded waiting.
    pub fn machine_internal_duplicate_predecessor_at(
        &self,
        coordinate: f64,
        budget: FrontierBudget,
    ) -> FrontierOutcome<f64> {
        let Some(remaining) = budget.max_operations.checked_sub(1) else {
            return FrontierOutcome::Unknown(FrontierUnknown::WorkLimit);
        };
        let started = Instant::now();
        match self.machine_internal_duplicate_waiting_at(
            coordinate,
            FrontierBudget {
                max_operations: remaining,
                ..budget
            },
        ) {
            FrontierOutcome::Complete(_) => {}
            FrontierOutcome::Empty => return FrontierOutcome::Empty,
            FrontierOutcome::Unknown(reason) => return FrontierOutcome::Unknown(reason),
        }
        if budget
            .max_elapsed
            .is_some_and(|limit| started.elapsed() >= limit)
        {
            return FrontierOutcome::Unknown(FrontierUnknown::Deadline);
        }
        let pairs = self.front.components[0].pairs();
        match near(coordinate, pairs[0].1, FrontOperationPolicy::FrontAlgebra) {
            Ok(true) => FrontierOutcome::Complete(coordinate),
            Ok(false) => FrontierOutcome::Complete(pairs[1].1),
            Err(reason) => FrontierOutcome::Unknown(reason),
        }
    }

    /// Recover waiting for a coordinate selected within a finite nonzero-width
    /// one-segment machine front. This is not a source-control witness.
    pub fn machine_single_segment_waiting_at(
        &self,
        coordinate: f64,
        budget: FrontierBudget,
    ) -> FrontierOutcome<f64> {
        let mut work = Work::new(budget);
        if budget.max_components == 0 {
            return FrontierOutcome::Unknown(FrontierUnknown::ComponentLimit);
        }
        if budget.max_pairs < 2 {
            return FrontierOutcome::Unknown(FrontierUnknown::PairLimit);
        }
        if self.front.components.len() != 1 || !coordinate.is_finite() {
            return FrontierOutcome::Unknown(FrontierUnknown::Unsupported(
                "machine waiting reconstruction inputs",
            ));
        }
        let function = &self.front.components[0];
        let pairs = function.pairs();
        if function.is_machine_zero_length()
            || pairs.len() != 2
            || coordinate < pairs[0].1
            || coordinate > pairs[1].1
            || self.waiting_bits & !(1 << 1) != 0
        {
            return FrontierOutcome::Unknown(FrontierUnknown::Unsupported(
                "machine waiting reconstruction geometry",
            ));
        }
        if let Err(reason) = work.charge(5) {
            return FrontierOutcome::Unknown(reason);
        }
        // The caller's lookup rejects its 1e12 sentinel before using the bit.
        match function.evaluate(coordinate, EvaluationPolicy::EndpointTolerance) {
            Ok(Some(value)) if value != 1e12 => {}
            Ok(_) => {
                return FrontierOutcome::Unknown(FrontierUnknown::Unsupported(
                    "machine waiting reconstruction lookup sentinel",
                ));
            }
            Err(error) => return FrontierOutcome::Unknown(FrontierUnknown::Arithmetic(error)),
        }
        let waiting = if self.waiting_bits & (1 << 1) == 0 {
            0.0
        } else {
            let Some(predecessor) = self.predecessor else {
                return FrontierOutcome::Unknown(FrontierUnknown::Unsupported(
                    "machine waiting reconstruction predecessor unavailable",
                ));
            };
            if predecessor == 1e12 || predecessor < 0.0 {
                return FrontierOutcome::Unknown(FrontierUnknown::Unsupported(
                    "machine waiting reconstruction predecessor rejected",
                ));
            }
            match checked(
                coordinate - predecessor,
                "machine waiting reconstruction subtraction",
            ) {
                Ok(value) => value,
                Err(error) => return FrontierOutcome::Unknown(FrontierUnknown::Arithmetic(error)),
            }
        };
        if let Err(reason) = work.charge(0) {
            return FrontierOutcome::Unknown(reason);
        }
        FrontierOutcome::Complete(waiting)
    }

    /// Return the coordinate passed to the current arc's inverse while
    /// reconstructing a finite one-segment machine state. When waiting bit 1
    /// is clear this is the selected coordinate; when it is set the reference
    /// carries the stored predecessor directly. In particular, do not recover
    /// it as `coordinate - waiting`: that would introduce another rounded
    /// subtraction. The caller must still evaluate the correct arc inverse.
    /// This operation is not a source-control witness or route certificate.
    pub fn machine_single_segment_predecessor_at(
        &self,
        coordinate: f64,
        budget: FrontierBudget,
    ) -> FrontierOutcome<f64> {
        match self.machine_single_segment_waiting_at(coordinate, budget) {
            FrontierOutcome::Complete(_) => {
                if self.waiting_bits & (1 << 1) == 0 {
                    FrontierOutcome::Complete(coordinate)
                } else {
                    match self.predecessor {
                        Some(predecessor) => FrontierOutcome::Complete(predecessor),
                        None => FrontierOutcome::Unknown(FrontierUnknown::Unsupported(
                            "machine waiting reconstruction predecessor unavailable",
                        )),
                    }
                }
            }
            FrontierOutcome::Empty => FrontierOutcome::Empty,
            FrontierOutcome::Unknown(reason) => FrontierOutcome::Unknown(reason),
        }
    }
}

impl MachineWaitingEnvelope {
    /// The intermediate slope function, before dependent bounds are applied.
    pub fn function(&self) -> &PreparedSlopeFunction {
        &self.function
    }
    /// Coordinate written by the recovered branch, or None if it leaves the
    /// predecessor field unchanged and its prior value is unavailable.
    pub fn predecessor(&self) -> Option<f64> {
        self.predecessor
    }
    /// Waiting provenance bits retained by the reference branch.
    pub fn waiting_bits(&self) -> u64 {
        self.waiting_bits
    }

    /// Compatibility entry point for the recovered lower-bound maximum.
    pub fn pass_constant_lower(
        &self,
        lower: &PreparedSlopeFunction,
        budget: FrontierBudget,
    ) -> FrontierOutcome<Self> {
        self.apply_lower_bound(lower, budget)
    }

    /// Ordered maximum with a covering dependent lower vector. Each emitted
    /// piece carries the active fixed segment's waiting bit, including pieces
    /// selected from the bound. Merging requires equal bits and near slopes.
    pub fn apply_lower_bound(
        &self,
        lower: &PreparedSlopeFunction,
        budget: FrontierBudget,
    ) -> FrontierOutcome<Self> {
        let started = Instant::now();
        let compute = || -> FrontResult<Self> {
            let deadline = || -> FrontResult<()> {
                if budget
                    .max_elapsed
                    .is_some_and(|limit| started.elapsed() >= limit)
                {
                    return Err(FrontierUnknown::Deadline);
                }
                Ok(())
            };
            deadline()?;
            if budget.max_components == 0 {
                return Err(FrontierUnknown::ComponentLimit);
            }
            let a = self.function.pairs();
            let b = lower.pairs();
            if a.len() > budget.max_pairs || b.len() > budget.max_pairs {
                return Err(FrontierUnknown::PairLimit);
            }
            if a.len() > 12 {
                return Err(FrontierUnknown::Unsupported(
                    "machine lower maximum array size",
                ));
            }
            if budget.max_operations < a.len().saturating_add(b.len()).saturating_mul(96) {
                return Err(FrontierUnknown::WorkLimit);
            }
            let close = |a, b| near(a, b, FrontOperationPolicy::FrontAlgebra);
            let finite = |value| arithmetic(value, "machine lower maximum");
            if (a[0].1 < b[0].1 && !close(a[0].1, b[0].1)?)
                || (a.last().unwrap().1 > b.last().unwrap().1
                    && !close(a.last().unwrap().1, b.last().unwrap().1)?)
            {
                return Err(FrontierUnknown::Unsupported("machine lower maximum domain"));
            }
            let (mut i, mut j) = (1usize, 1usize);
            let (mut av, mut bv) = (a[0].0, b[0].0);
            let (mut ax, mut bx) = (a[0].1, b[0].1);
            let b_start = if close(ax, bx)? {
                bv
            } else {
                while j < b.len() && b[j].1 < ax {
                    deadline()?;
                    bv = finite(bv + finite(finite(b[j].1 - bx)? * b[j].0)?)?;
                    bx = b[j].1;
                    j += 1;
                }
                let &(slope, end) = b.get(j).ok_or(FrontierUnknown::Unsupported(
                    "machine lower maximum start lookup",
                ))?;
                if close(end, ax)? {
                    bv = finite(bv + finite(finite(end - bx)? * slope)?)?;
                    bx = end;
                    j += 1;
                    bv
                } else {
                    finite(finite(slope * finite(ax - bx)?)? + bv)?
                }
            };
            let mut chosen_a = b_start <= av;
            let mut previous_tie = close(av, b_start)?;
            let mut output = vec![(if chosen_a { av } else { b_start }, ax)];
            let mut output_bits = 0u64;
            let append = |output: &mut Vec<(f64, f64)>,
                          bits: &mut u64,
                          slope: f64,
                          end: f64,
                          waits: bool|
             -> FrontResult<()> {
                let previous = output.len() - 1;
                let merge = previous > 0
                    && (((*bits >> previous) & 1 != 0) == waits)
                    && close(output[previous].0, slope)?;
                let index = if merge { previous } else { output.len() };
                if index >= 12 || index >= budget.max_pairs {
                    return Err(FrontierUnknown::PairLimit);
                }
                if merge {
                    output[index] = (slope, end);
                } else {
                    output.push((slope, end));
                }
                let bit = 1u64 << index;
                if waits {
                    *bits |= bit;
                } else {
                    *bits &= !bit;
                }
                Ok(())
            };
            while i < a.len() && j < b.len() {
                deadline()?;
                let (ae, be) = (a[i].1, b[j].1);
                let aend = finite(finite(finite(ae - ax)? * a[i].0)? + av)?;
                let bend = finite(finite(finite(be - bx)? * b[j].0)? + bv)?;
                let order = if close(ae, be)? {
                    0
                } else if be <= ae {
                    1
                } else {
                    -1
                };
                let (coordinate, ay, by) = match order {
                    1 => (be, finite(finite(finite(be - ax)? * a[i].0)? + av)?, bend),
                    -1 => (ae, aend, finite(finite(finite(ae - bx)? * b[j].0)? + bv)?),
                    _ => (ae, aend, bend),
                };
                let tie = close(ay, by)?;
                let next_a = if tie { chosen_a } else { by <= ay };
                let waits = (self.waiting_bits >> i) & 1 != 0;
                if chosen_a != next_a {
                    // A preceding near tie switches operands without an
                    // artificial zero-width crossing, irrespective of which
                    // original segment supplied that previous endpoint.
                    if !previous_tie {
                        let bi = finite(bend - finite(be * b[j].0)?)?;
                        let ai = finite(aend - finite(ae * a[i].0)?)?;
                        let crossing = finite(finite(bi - ai)? / finite(a[i].0 - b[j].0)?)?;
                        append(
                            &mut output,
                            &mut output_bits,
                            if chosen_a { a[i].0 } else { b[j].0 },
                            crossing,
                            waits,
                        )?;
                    }
                    append(
                        &mut output,
                        &mut output_bits,
                        if next_a { a[i].0 } else { b[j].0 },
                        coordinate,
                        waits,
                    )?;
                } else if (chosen_a && order != 1) || (!chosen_a && order != -1) {
                    append(
                        &mut output,
                        &mut output_bits,
                        if chosen_a { a[i].0 } else { b[j].0 },
                        coordinate,
                        waits,
                    )?;
                }
                chosen_a = next_a;
                previous_tie = tie;
                if order != 1 {
                    av = aend;
                    ax = ae;
                    i += 1;
                }
                if order != -1 {
                    bv = bend;
                    bx = be;
                    j += 1;
                }
            }
            if output.len() == 1 {
                if budget.max_pairs < 2 {
                    return Err(FrontierUnknown::PairLimit);
                }
                output.push((0., output[0].1));
            }
            let function = PreparedSlopeFunction::from_machine_pairs(output)
                .map_err(FrontierUnknown::Arithmetic)?;
            deadline()?;
            Ok(Self {
                function,
                waiting_bits: output_bits,
                predecessor: self.predecessor,
            })
        };
        match compute() {
            Ok(value) => FrontierOutcome::Complete(value),
            Err(reason) => FrontierOutcome::Unknown(reason),
        }
    }

    /// Compatibility entry point for the recovered connected upper filter.
    pub fn pass_constant_upper(
        &self,
        upper: &PreparedSlopeFunction,
        budget: FrontierBudget,
    ) -> FrontierOutcome<Self> {
        self.apply_upper_bound(upper, budget)
    }

    /// Ordered connected-region upper filter. A later re-entry is the
    /// machine helper's status 2 and remains unknown. The reference caller
    /// retains its waiting bits and predecessor after this in-place filter.
    pub fn apply_upper_bound(
        &self,
        upper: &PreparedSlopeFunction,
        budget: FrontierBudget,
    ) -> FrontierOutcome<Self> {
        let started = Instant::now();
        let compute = || -> FrontResult<Self> {
            let deadline = || -> FrontResult<()> {
                if budget
                    .max_elapsed
                    .is_some_and(|limit| started.elapsed() >= limit)
                {
                    return Err(FrontierUnknown::Deadline);
                }
                Ok(())
            };
            deadline()?;
            if budget.max_components == 0 {
                return Err(FrontierUnknown::ComponentLimit);
            }
            let a = self.function.pairs();
            let b = upper.pairs();
            if a.len() > 12 {
                return Err(FrontierUnknown::Unsupported(
                    "machine upper filter array size",
                ));
            }
            if a.len() > budget.max_pairs || b.len() > budget.max_pairs {
                return Err(FrontierUnknown::PairLimit);
            }
            if budget.max_operations < a.len().saturating_add(b.len()).saturating_mul(64) {
                return Err(FrontierUnknown::WorkLimit);
            }
            let close = |a, b| near(a, b, FrontOperationPolicy::FrontAlgebra);
            let finite = |value| arithmetic(value, "machine upper filter");
            if (a[0].1 < b[0].1 && !close(a[0].1, b[0].1)?)
                || (a.last().unwrap().1 > b.last().unwrap().1
                    && !close(a.last().unwrap().1, b.last().unwrap().1)?)
            {
                return Err(FrontierUnknown::Unsupported(
                    "machine upper filter domain status",
                ));
            }
            let (mut i, mut j) = (1usize, 1usize);
            let (mut av, mut bv) = (a[0].0, b[0].0);
            let (mut ax, mut bx) = (a[0].1, b[0].1);
            let b_start = if close(ax, bx)? {
                bv
            } else {
                while j < b.len() && b[j].1 < ax {
                    deadline()?;
                    bv = finite(bv + finite(finite(b[j].1 - bx)? * b[j].0)?)?;
                    bx = b[j].1;
                    j += 1;
                }
                let &(slope, end) = b.get(j).ok_or(FrontierUnknown::Unsupported(
                    "machine upper filter start lookup",
                ))?;
                if close(end, ax)? {
                    bv = finite(bv + finite(finite(end - bx)? * slope)?)?;
                    bx = end;
                    j += 1;
                    bv
                } else {
                    finite(finite(finite(ax - bx)? * slope)? + bv)?
                }
            };
            let mut active = av < b_start || close(av, b_start)?;
            let mut output = if active { vec![a[0]] } else { Vec::new() };
            let append = |output: &mut Vec<(f64, f64)>, pair| -> FrontResult<()> {
                if output.len() >= 12 || output.len() >= budget.max_pairs {
                    return Err(FrontierUnknown::PairLimit);
                }
                output.push(pair);
                Ok(())
            };
            while i < a.len() && j < b.len() {
                deadline()?;
                let (ae, be) = (a[i].1, b[j].1);
                let aend = finite(finite(finite(ae - ax)? * a[i].0)? + av)?;
                let bend = finite(finite(finite(be - bx)? * b[j].0)? + bv)?;
                let order = if close(ae, be)? {
                    0
                } else if be <= ae {
                    1
                } else {
                    -1
                };
                let (coordinate, ay, by) = match order {
                    1 => (be, finite(finite(finite(be - ax)? * a[i].0)? + av)?, bend),
                    -1 => (ae, aend, finite(finite(finite(ae - bx)? * b[j].0)? + bv)?),
                    _ => (ae, aend, bend),
                };
                let tie = close(ay, by)?;
                if active {
                    if tie || ay <= by {
                        append(&mut output, (a[i].0, coordinate))?;
                    } else {
                        let ai = finite(aend - finite(ae * a[i].0)?)?;
                        let bi = finite(bend - finite(be * b[j].0)?)?;
                        let crossing = finite(finite(bi - ai)? / finite(a[i].0 - b[j].0)?)?;
                        if !close(crossing, output.last().unwrap().1)? {
                            append(&mut output, (a[i].0, crossing))?;
                        }
                        active = false;
                    }
                } else if tie {
                    if !output.is_empty() {
                        return Err(FrontierUnknown::Unsupported(
                            "machine upper filter reentry status",
                        ));
                    }
                    append(&mut output, (ay, coordinate))?;
                    active = true;
                } else if ay < by {
                    if !output.is_empty() {
                        return Err(FrontierUnknown::Unsupported(
                            "machine upper filter reentry status",
                        ));
                    }
                    let ai = finite(aend - finite(ae * a[i].0)?)?;
                    let bi = finite(bend - finite(be * b[j].0)?)?;
                    let crossing = finite(finite(bi - ai)? / finite(a[i].0 - b[j].0)?)?;
                    let anchor = finite(finite(crossing * a[i].0)? + ai)?;
                    append(&mut output, (anchor, crossing))?;
                    // The entry branch copies the original fixed endpoint,
                    // including when only the vector endpoint was consumed.
                    append(&mut output, a[i])?;
                    active = true;
                }
                if order != 1 {
                    av = aend;
                    ax = ae;
                    i += 1;
                }
                if order != -1 {
                    bv = bend;
                    bx = be;
                    j += 1;
                }
            }
            if output.is_empty() {
                return Err(FrontierUnknown::Unsupported(
                    "machine upper filter invalid empty count",
                ));
            }
            if output.len() == 1 {
                let coordinate = output[0].1;
                append(&mut output, (0., coordinate))?;
            }
            let function = PreparedSlopeFunction::from_machine_pairs(output)
                .map_err(FrontierUnknown::Arithmetic)?;
            deadline()?;
            Ok(Self {
                function,
                waiting_bits: self.waiting_bits,
                predecessor: self.predecessor,
            })
        };
        match compute() {
            Ok(value) => FrontierOutcome::Complete(value),
            Err(reason) => FrontierOutcome::Unknown(reason),
        }
    }
}
/// Verified ordered charges evaluated at a shared argument. This is a distinct
/// witness scope; its value can differ from the added front's value/minimum.
#[derive(Clone, Debug)]
pub struct SourceReplayWitness {
    argument: f64,
    value: f64,
    charges: Arc<[f64]>,
}
impl SourceReplayWitness {
    pub fn evaluate_shared_argument(
        functions: &[PreparedSlopeFunction],
        argument: f64,
        policy: EvaluationPolicy,
    ) -> Result<Option<Self>, RawFunctionError> {
        if !argument.is_finite() {
            return Err(RawFunctionError::NonFinite("argument"));
        }
        let mut value = 0.;
        let mut charges = Vec::with_capacity(functions.len());
        for function in functions {
            if function.is_machine_zero_length() {
                return Err(RawFunctionError::Unsupported(
                    "machine zero-length segment is not a source descriptor",
                ));
            }
            let Some(charge) = function.evaluate(argument, policy)? else {
                return Ok(None);
            };
            value = checked(value + charge, "source replay accumulation")?;
            charges.push(charge);
        }
        Ok(Some(Self {
            argument,
            value,
            charges: charges.into(),
        }))
    }
    pub fn argument(&self) -> f64 {
        self.argument
    }
    pub fn value(&self) -> f64 {
        self.value
    }
    pub fn charge_values(&self) -> &[f64] {
        &self.charges
    }
}
impl SlopeFront {
    /// Compatibility entry point for the recovered forward waiting scan.
    pub fn machine_single_segment_waiting(
        &self,
        lower: f64,
        upper: f64,
        waiting: i32,
        budget: FrontierBudget,
    ) -> FrontierOutcome<MachineWaitingEnvelope> {
        self.machine_waiting(lower, upper, waiting, budget)
    }

    /// Literal forward-extension waiting scan, retaining the original source
    /// value anchor across runs and the recorder's predecessor and waiting
    /// bits. This is an intermediate machine array, before dependent bounds.
    pub fn machine_waiting(
        &self,
        lower: f64,
        upper: f64,
        waiting: i32,
        budget: FrontierBudget,
    ) -> FrontierOutcome<MachineWaitingEnvelope> {
        let started = Instant::now();
        let compute = || -> FrontResult<MachineWaitingEnvelope> {
            let deadline = || -> FrontResult<()> {
                if budget
                    .max_elapsed
                    .is_some_and(|limit| started.elapsed() >= limit)
                {
                    return Err(FrontierUnknown::Deadline);
                }
                Ok(())
            };
            deadline()?;
            if !lower.is_finite() || !upper.is_finite() || lower > upper {
                return Err(FrontierUnknown::Unsupported(
                    "machine waiting needs finite ordered bounds",
                ));
            }
            if budget.max_components == 0 {
                return Err(FrontierUnknown::ComponentLimit);
            }
            let [function] = self.components() else {
                return Err(FrontierUnknown::Unsupported(
                    "machine waiting requires one component",
                ));
            };
            let pairs = function.pairs();
            if !(2..=12).contains(&pairs.len()) || pairs[0].0 == 1e12 {
                return Err(FrontierUnknown::Unsupported("machine waiting input array"));
            }
            if pairs.len() > budget.max_pairs {
                return Err(FrontierUnknown::PairLimit);
            }
            let required = pairs
                .len()
                .saturating_mul(pairs.len())
                .saturating_mul(32)
                .saturating_add(32);
            if required > budget.max_operations {
                return Err(FrontierUnknown::WorkLimit);
            }
            let close = |a, b| near(a, b, FrontOperationPolicy::FrontAlgebra);
            let finite = |value| arithmetic(value, "machine forward waiting scan");
            fn emit(
                out: &mut Vec<(f64, f64)>,
                bits: &mut u64,
                slope: f64,
                end: f64,
                waits: bool,
                budget: FrontierBudget,
            ) -> FrontResult<()> {
                if out.len() >= 12 || out.len() >= budget.max_pairs {
                    return Err(FrontierUnknown::PairLimit);
                }
                let bit = 1u64 << out.len();
                if waits {
                    *bits |= bit;
                } else {
                    *bits &= !bit;
                }
                out.push((slope, end));
                Ok(())
            }
            // The wrapper resets bits and predecessor before extension.
            let mut output = vec![(0., 0.)];
            let mut bits = 0;
            let mut predecessor = None;
            let rate = f64::from(waiting);
            let mut value = pairs[0].0;
            let last = pairs.len() - 1;
            let mut i = 1;
            while i <= last {
                deadline()?;
                let (slope, end) = pairs[i];
                let previous = pairs[i - 1].1;
                if slope <= rate {
                    if lower < end && !close(lower, end)? {
                        if output.len() == 1 {
                            let start = if previous <= lower { lower } else { previous };
                            output[0] = (
                                finite(finite(finite(start - previous)? * slope)? + value)?,
                                start,
                            );
                        }
                        if upper <= end || close(upper, end)? {
                            emit(&mut output, &mut bits, slope, upper, false, budget)?;
                            break;
                        }
                        emit(&mut output, &mut bits, slope, end, false, budget)?;
                    }
                    value = finite(value + finite(slope * finite(end - previous)?)?)?;
                    if i < last {
                        i += 1;
                        continue;
                    }
                    if end < upper && !close(end, upper)? {
                        if output.len() == 1 {
                            output[0] =
                                (finite(value + finite(finite(lower - end)? * rate)?)?, lower);
                            predecessor = Some(end);
                        }
                        emit(&mut output, &mut bits, rate, upper, true, budget)?;
                    }
                    break;
                }
                // Follow a waiting line from the beginning of this run until
                // the scanned source endpoint reaches it or the source ends.
                let anchor_coordinate = previous;
                let mut scanned = finite(finite(finite(end - previous)? * slope)? + value)?;
                let mut j = i + 1;
                if j <= last {
                    let mut wait_value =
                        finite(finite(finite(pairs[j].1 - anchor_coordinate)? * rate)? + value)?;
                    scanned = finite(scanned + finite(pairs[j].0 * finite(pairs[j].1 - end)?)?)?;
                    while scanned > wait_value && !close(scanned, wait_value)? {
                        deadline()?;
                        j += 1;
                        if j > last {
                            break;
                        }
                        wait_value = finite(
                            finite(finite(pairs[j].1 - anchor_coordinate)? * rate)? + value,
                        )?;
                        scanned = finite(
                            scanned + finite(finite(pairs[j].1 - pairs[j - 1].1)? * pairs[j].0)?,
                        )?;
                    }
                }
                if j > last {
                    if output.len() == 1 {
                        let start = if anchor_coordinate <= lower {
                            lower
                        } else {
                            anchor_coordinate
                        };
                        // The exhausted branch uses the original run slope
                        // for its anchor, even though the emitted slope waits.
                        output[0] = (
                            finite(value + finite(slope * finite(start - anchor_coordinate)?)?)?,
                            start,
                        );
                        predecessor = Some(anchor_coordinate);
                    }
                    let finish = if pairs[last].1 <= upper {
                        upper
                    } else {
                        pairs[last].1
                    };
                    emit(&mut output, &mut bits, rate, finish, true, budget)?;
                    break;
                }
                let (next_slope, next_end) = pairs[j];
                if next_end <= lower || close(next_end, lower)? {
                    // Retain the value anchor, as the ordered recorder does;
                    // replacing it with the scanned endpoint changes later runs.
                    i = j + 1;
                    continue;
                }
                let before_next = pairs[j - 1].1;
                let crossing = finite(
                    finite(
                        finite(finite(scanned - value)? - finite(before_next * next_slope)?)?
                            + finite(anchor_coordinate * rate)?,
                    )? / finite(rate - next_slope)?,
                )?;
                if crossing <= lower || close(crossing, lower)? {
                    if output.len() == 1 {
                        let start = if anchor_coordinate <= lower {
                            lower
                        } else {
                            anchor_coordinate
                        };
                        output[0] = (
                            finite(scanned + finite(finite(start - before_next)? * next_slope)?)?,
                            start,
                        );
                    }
                } else {
                    if output.len() == 1 {
                        let start = if anchor_coordinate <= lower {
                            lower
                        } else {
                            anchor_coordinate
                        };
                        output[0] = (
                            finite(finite(finite(start - anchor_coordinate)? * rate)? + value)?,
                            start,
                        );
                        predecessor = Some(anchor_coordinate);
                    }
                    if upper <= crossing || close(upper, crossing)? {
                        emit(&mut output, &mut bits, rate, upper, true, budget)?;
                        break;
                    }
                    emit(&mut output, &mut bits, rate, crossing, true, budget)?;
                }
                if upper <= next_end || close(upper, next_end)? {
                    emit(&mut output, &mut bits, next_slope, upper, false, budget)?;
                    break;
                }
                emit(&mut output, &mut bits, next_slope, next_end, false, budget)?;
                if j == last {
                    if next_end < upper && !close(next_end, upper)? {
                        emit(&mut output, &mut bits, rate, upper, true, budget)?;
                    }
                    break;
                }
                i = j + 1;
            }
            if output.len() == 1 {
                return Err(FrontierUnknown::Unsupported(
                    "machine waiting produced no segments",
                ));
            }
            let function = PreparedSlopeFunction::from_machine_pairs(output)
                .map_err(FrontierUnknown::Arithmetic)?;
            deadline()?;
            Ok(MachineWaitingEnvelope {
                function,
                predecessor,
                waiting_bits: bits,
            })
        };
        match compute() {
            Ok(value) => FrontierOutcome::Complete(value),
            Err(reason) => FrontierOutcome::Unknown(reason),
        }
    }

    /// Reduce consecutive near slopes in the fixed machine array. The first
    /// slope is retained on a merge; the original snapshots remain available
    /// for reconstruction. Zero-width first segments follow the same scan.
    pub fn machine_reduce_for_continuation(&self, budget: FrontierBudget) -> FrontierOutcome<Self> {
        let mut work = Work::new(budget);
        let mut compute = || -> FrontResult<Self> {
            if self.components.len() != 1 {
                return Err(FrontierUnknown::Unsupported(
                    "machine reduction requires one component",
                ));
            }
            if budget.max_components == 0 {
                return Err(FrontierUnknown::ComponentLimit);
            }
            let pairs = self.components[0].pairs();
            if !(2..=12).contains(&pairs.len()) {
                return Err(FrontierUnknown::Unsupported("machine reduction array size"));
            }
            if pairs.len() > budget.max_pairs {
                return Err(FrontierUnknown::PairLimit);
            }
            work.charge(pairs.len().saturating_mul(8).saturating_add(4))?;
            let mut reduced = vec![pairs[0], pairs[1]];
            let mut merged = false;
            for &(slope, end) in &pairs[2..] {
                work.charge(0)?;
                let last = reduced.len() - 1;
                if near(reduced[last].0, slope, FrontOperationPolicy::FrontAlgebra)? {
                    reduced[last].1 = end;
                    merged = true;
                } else {
                    reduced.push((slope, end));
                }
            }
            // The reducer checks consecutive emitted segment endpoints after
            // merging. The first anchor/endpoint pair is exempt.
            for index in 2..reduced.len() {
                work.charge(0)?;
                if near(
                    reduced[index - 1].1,
                    reduced[index].1,
                    FrontOperationPolicy::FrontAlgebra,
                )? {
                    return Err(FrontierUnknown::Unsupported(
                        "machine reduction consecutive endpoints",
                    ));
                }
            }
            let components = if merged {
                vec![PreparedSlopeFunction::from_machine_pairs(reduced)
                    .map_err(FrontierUnknown::Arithmetic)?]
                .into()
            } else {
                self.components.clone()
            };
            work.charge(0)?;
            Ok(Self {
                components,
                provenance: Arc::new(FrontierProvenance::MachineReduction {
                    input: self.provenance.clone(),
                    input_front: self.components.clone(),
                    merged,
                }),
                tolerant_alignment: self.tolerant_alignment || merged,
            })
        };
        match compute() {
            Ok(value) => FrontierOutcome::Complete(value),
            Err(reason) => FrontierOutcome::Unknown(reason),
        }
    }

    /// Recovered machine waiting steps for a single zero-length segment.
    /// The lower-slope branch accepts a lower bound at or above the point,
    /// or a lower bound below it within the machine tolerance, and requires
    /// a distinctly later upper bound. `lower` and `upper` are the
    /// reference helper's second and third floating-point arguments. The
    /// returned array and predecessor bit have not passed subsequent native
    /// maximum or upper-bound operations and are not a feasibility result.
    pub fn machine_zero_length_waiting(
        &self,
        lower: f64,
        upper: f64,
        waiting: i32,
        budget: FrontierBudget,
    ) -> FrontierOutcome<MachineWaitingEnvelope> {
        let started = Instant::now();
        if !lower.is_finite() || !upper.is_finite() || lower > upper {
            return FrontierOutcome::Unknown(FrontierUnknown::Unsupported(
                "machine waiting needs finite ordered bounds",
            ));
        }
        if budget.max_components == 0 {
            return FrontierOutcome::Unknown(FrontierUnknown::ComponentLimit);
        }
        if budget.max_pairs < 2 {
            return FrontierOutcome::Unknown(FrontierUnknown::PairLimit);
        }
        if budget.max_operations < 4 {
            return FrontierOutcome::Unknown(FrontierUnknown::WorkLimit);
        }
        if budget
            .max_elapsed
            .is_some_and(|limit| started.elapsed() >= limit)
        {
            return FrontierOutcome::Unknown(FrontierUnknown::Deadline);
        }
        if self.components.len() != 1
            || !self.components[0].is_machine_zero_length()
            || self.components[0].has_internal_zero_length()
        {
            return FrontierOutcome::Unknown(FrontierUnknown::Unsupported(
                "machine waiting requires one zero-length segment",
            ));
        }
        let pairs = self.components[0].pairs();
        let (value, predecessor) = pairs[0];
        let (slope, end) = pairs[1];
        let wait_slope = f64::from(waiting);
        if value == 1e12 {
            return FrontierOutcome::Unknown(FrontierUnknown::Unsupported(
                "unrecovered machine waiting branch",
            ));
        }
        if slope <= wait_slope {
            if lower >= upper || upper <= predecessor {
                return FrontierOutcome::Unknown(FrontierUnknown::Unsupported(
                    "unrecovered lower-slope machine waiting bounds",
                ));
            }
            if lower < predecessor {
                match near(lower, predecessor, FrontOperationPolicy::FrontAlgebra) {
                    Ok(true) => {}
                    Ok(false) => {
                        let upper_near =
                            match near(upper, predecessor, FrontOperationPolicy::FrontAlgebra) {
                                Ok(value) => value,
                                Err(error) => return FrontierOutcome::Unknown(error),
                            };
                        let anchor = match checked(end - predecessor, "machine waiting span")
                            .and_then(|span| {
                                checked(span * slope, "machine waiting multiplication")
                            })
                            .and_then(|product| {
                                checked(value + product, "machine waiting accumulation")
                            }) {
                            Ok(value) => value,
                            Err(error) => {
                                return FrontierOutcome::Unknown(FrontierUnknown::Arithmetic(error))
                            }
                        };
                        if !upper_near && budget.max_pairs < 3 {
                            return FrontierOutcome::Unknown(FrontierUnknown::PairLimit);
                        }
                        if budget
                            .max_elapsed
                            .is_some_and(|limit| started.elapsed() >= limit)
                        {
                            return FrontierOutcome::Unknown(FrontierUnknown::Deadline);
                        }
                        let function = if upper_near {
                            PreparedSlopeFunction::from_prepared_pairs(vec![
                                (anchor, predecessor),
                                (slope, upper),
                            ])
                        } else {
                            PreparedSlopeFunction::machine_internal_zero_length(
                                anchor,
                                slope,
                                predecessor,
                                wait_slope,
                                upper,
                            )
                        };
                        return match function {
                            Ok(function) => FrontierOutcome::Complete(MachineWaitingEnvelope {
                                function,
                                predecessor: None,
                                waiting_bits: if upper_near { 0 } else { 1 << 2 },
                            }),
                            Err(error) => {
                                FrontierOutcome::Unknown(FrontierUnknown::Arithmetic(error))
                            }
                        };
                    }
                    Err(error) => return FrontierOutcome::Unknown(error),
                }
            }
            match near(upper, predecessor, FrontOperationPolicy::FrontAlgebra) {
                Ok(false) => {}
                Ok(true) => {
                    return FrontierOutcome::Unknown(FrontierUnknown::Unsupported(
                        "unrecovered near-end machine waiting branch",
                    ))
                }
                Err(error) => return FrontierOutcome::Unknown(error),
            }
            let anchor = checked(end - predecessor, "machine waiting span")
                .and_then(|span| checked(span * slope, "machine waiting multiplication"))
                .and_then(|product| checked(value + product, "machine waiting accumulation"))
                .and_then(|v| {
                    checked(lower - end, "machine waiting lower span")
                        .and_then(|span| {
                            checked(span * wait_slope, "machine waiting lower multiplication")
                        })
                        .and_then(|product| {
                            checked(v + product, "machine waiting lower accumulation")
                        })
                });
            let anchor = match anchor {
                Ok(value) => value,
                Err(error) => return FrontierOutcome::Unknown(FrontierUnknown::Arithmetic(error)),
            };
            if budget.max_operations < 6 {
                return FrontierOutcome::Unknown(FrontierUnknown::WorkLimit);
            }
            if budget
                .max_elapsed
                .is_some_and(|limit| started.elapsed() >= limit)
            {
                return FrontierOutcome::Unknown(FrontierUnknown::Deadline);
            }
            return match PreparedSlopeFunction::from_prepared_pairs(vec![
                (anchor, lower),
                (wait_slope, upper),
            ]) {
                Ok(function) => FrontierOutcome::Complete(MachineWaitingEnvelope {
                    function,
                    predecessor: Some(predecessor),
                    waiting_bits: 1 << 1,
                }),
                Err(error) => FrontierOutcome::Unknown(FrontierUnknown::Arithmetic(error)),
            };
        }
        // MAXSD selects its second operand on equality, including signed
        // zero. Preserve that order for both observed maximum operations.
        let start = if predecessor > lower {
            predecessor
        } else {
            lower
        };
        let finish = if end > upper { end } else { upper };
        let anchor = match checked(start - predecessor, "machine waiting span")
            .and_then(|span| checked(slope * span, "machine waiting multiplication"))
            .and_then(|product| checked(value + product, "machine waiting accumulation"))
        {
            Ok(value) => value,
            Err(error) => return FrontierOutcome::Unknown(FrontierUnknown::Arithmetic(error)),
        };
        let function = if start == finish {
            PreparedSlopeFunction::machine_zero_length(anchor, wait_slope, start, finish)
        } else {
            PreparedSlopeFunction::from_prepared_pairs(vec![(anchor, start), (wait_slope, finish)])
        };
        if budget
            .max_elapsed
            .is_some_and(|limit| started.elapsed() >= limit)
        {
            return FrontierOutcome::Unknown(FrontierUnknown::Deadline);
        }
        match function {
            Ok(function) => FrontierOutcome::Complete(MachineWaitingEnvelope {
                function,
                predecessor: Some(predecessor),
                waiting_bits: 1 << 1,
            }),
            Err(error) => FrontierOutcome::Unknown(FrontierUnknown::Arithmetic(error)),
        }
    }

    pub(super) fn with_provenance(
        mut self,
        provenance: FrontierProvenance,
        tolerant_alignment: bool,
    ) -> Self {
        self.provenance = Arc::new(provenance);
        self.tolerant_alignment |= tolerant_alignment;
        self
    }
    pub(super) fn paired_result(
        mut self,
        input: &Self,
        arc: Arc<super::SlopeRelationArc>,
        policy: FrontOperationPolicy,
        intermediate_tolerant_alignment: bool,
    ) -> Self {
        self.provenance = Arc::new(FrontierProvenance::Paired {
            input: input.provenance.clone(),
            arc,
            policy,
            intermediate_tolerant_alignment,
        });
        self.tolerant_alignment |= input.tolerant_alignment || intermediate_tolerant_alignment;
        self
    }
    pub(super) fn paired_adjoint_result(
        self,
        input: &Self,
        arc: Arc<super::SlopeRelationArc>,
        policy: FrontOperationPolicy,
        intermediate_tolerant_alignment: bool,
    ) -> Self {
        self.with_provenance(
            FrontierProvenance::PairedAdjoint {
                input: input.components.clone(),
                arc,
                policy,
            },
            input.tolerant_alignment
                || intermediate_tolerant_alignment
                || policy == FrontOperationPolicy::FrontAlgebra,
        )
    }
    pub fn zero(lower: f64, upper: f64) -> Result<Self, RawFunctionError> {
        let function = super::RawAffine::new(0., 0.)?.prepare(lower, upper)?;
        Ok(Self::from_prepared(function))
    }

    /// Translate every retained coordinate with a literal addition. Slopes
    /// and anchor values remain unchanged. A rounded coordinate collapse is
    /// unsupported, rather than a reason to drop one of the source pieces.
    pub fn translate(&self, delta: f64, budget: FrontierBudget) -> FrontierOutcome<Self> {
        if self.has_machine_zero_length() {
            return FrontierOutcome::Unknown(FrontierUnknown::Unsupported(
                "machine zero-length segment translation",
            ));
        }
        if !delta.is_finite() {
            return FrontierOutcome::Unknown(FrontierUnknown::Arithmetic(
                RawFunctionError::NonFinite("translation"),
            ));
        }
        let mut work = Work::new(budget);
        if self.components.len() > budget.max_components {
            return FrontierOutcome::Unknown(FrontierUnknown::ComponentLimit);
        }
        let mut components = Vec::with_capacity(self.components.len());
        for component in self.components.iter() {
            let mut pairs = Vec::with_capacity(component.pairs().len());
            let mut previous = None;
            for &(value_or_slope, coordinate) in component.pairs() {
                if let Err(reason) = work.pair() {
                    return FrontierOutcome::Unknown(reason);
                }
                let translated =
                    match arithmetic(coordinate + delta, "front coordinate translation") {
                        Ok(value) => value,
                        Err(reason) => return FrontierOutcome::Unknown(reason),
                    };
                if previous.is_some_and(|previous| translated <= previous) {
                    return FrontierOutcome::Unknown(FrontierUnknown::Unsupported(
                        "translation collapsed consecutive coordinates",
                    ));
                }
                pairs.push((value_or_slope, translated));
                previous = Some(translated);
            }
            match PreparedSlopeFunction::from_prepared_pairs(pairs) {
                Ok(component) => {
                    if components.len() >= budget.max_components {
                        return FrontierOutcome::Unknown(FrontierUnknown::ComponentLimit);
                    }
                    components.push(component);
                }
                Err(error) => return FrontierOutcome::Unknown(FrontierUnknown::Arithmetic(error)),
            }
        }
        if let Err(reason) = validate(&components, &mut work) {
            return FrontierOutcome::Unknown(reason);
        }
        FrontierOutcome::Complete(Self {
            components: components.into(),
            provenance: Arc::new(FrontierProvenance::Translation {
                input: self.provenance.clone(),
                delta,
            }),
            tolerant_alignment: self.tolerant_alignment,
        })
    }

    /// Intersect the component union with the exact requested closed window.
    /// Re-anchor at its lower coordinate and retain native slopes; no secant
    /// fitting or bridging a gap occurs. Unrecovered near-boundary native
    /// truncation cases remain Unknown instead of guessing their topology.
    pub fn restrict(
        &self,
        lower: f64,
        upper: f64,
        policy: FrontOperationPolicy,
        budget: FrontierBudget,
    ) -> FrontierOutcome<Self> {
        if self.has_machine_zero_length() {
            return FrontierOutcome::Unknown(FrontierUnknown::Unsupported(
                "machine zero-length segment restriction",
            ));
        }
        if !lower.is_finite() || !upper.is_finite() {
            return FrontierOutcome::Unknown(FrontierUnknown::Arithmetic(
                RawFunctionError::NonFinite("restriction window"),
            ));
        }
        if lower > upper {
            return FrontierOutcome::Unknown(FrontierUnknown::Arithmetic(
                RawFunctionError::InvalidDomain,
            ));
        }
        let mut work = Work::new(budget);
        let mut components = Vec::new();
        for function in self.components.iter() {
            if let Err(reason) = work.charge(function.pairs().len()) {
                return FrontierOutcome::Unknown(reason);
            }
            let (start, end) = function.domain();
            let lo = start.max(lower);
            let hi = end.min(upper);
            if lo > hi {
                continue;
            }
            if policy == FrontOperationPolicy::FrontAlgebra {
                for &(_, boundary) in function.pairs() {
                    for coordinate in [lo, hi] {
                        if coordinate != boundary {
                            match near(coordinate, boundary, policy) {
                                Ok(true) => {
                                    return FrontierOutcome::Unknown(FrontierUnknown::Unsupported(
                                        "nonexact near-boundary truncation needs a topology proof",
                                    ))
                                }
                                Err(reason) => return FrontierOutcome::Unknown(reason),
                                Ok(false) => (),
                            }
                        }
                    }
                }
            }
            let anchor = match value_at(function, lo, policy) {
                Ok(value) => value,
                Err(reason) => return FrontierOutcome::Unknown(reason),
            };
            if let Err(reason) = work.pair() {
                return FrontierOutcome::Unknown(reason);
            }
            let mut pairs = vec![(anchor, lo)];
            if lo != hi {
                for &(slope, segment_end) in &function.pairs()[1..] {
                    if segment_end <= lo {
                        continue;
                    }
                    if let Err(reason) = work.pair() {
                        return FrontierOutcome::Unknown(reason);
                    }
                    pairs.push((slope, segment_end.min(hi)));
                    if segment_end >= hi {
                        break;
                    }
                }
            }
            match PreparedSlopeFunction::from_prepared_pairs(pairs) {
                Ok(component) => {
                    if components.len() >= budget.max_components {
                        return FrontierOutcome::Unknown(FrontierUnknown::ComponentLimit);
                    }
                    components.push(component);
                }
                Err(error) => return FrontierOutcome::Unknown(FrontierUnknown::Arithmetic(error)),
            }
        }
        if components.is_empty() {
            return FrontierOutcome::Empty;
        }
        if let Err(reason) = validate(&components, &mut work) {
            return FrontierOutcome::Unknown(reason);
        }
        FrontierOutcome::Complete(Self {
            components: components.into(),
            provenance: Arc::new(FrontierProvenance::Restriction {
                input: self.provenance.clone(),
                lower,
                upper,
                policy,
            }),
            tolerant_alignment: self.tolerant_alignment,
        })
    }
    pub fn from_prepared(function: PreparedSlopeFunction) -> Self {
        let tolerant_alignment = function.is_machine_zero_length();
        let components: Arc<[PreparedSlopeFunction]> = vec![function].into();
        Self {
            provenance: Arc::new(FrontierProvenance::Source(components.clone())),
            components,
            tolerant_alignment,
        }
    }
    /// Components must be strictly disjoint and sorted. Overlaps/touching
    /// components need an envelope, which this addition primitive does not fake.
    pub fn from_components(
        components: Vec<PreparedSlopeFunction>,
        budget: FrontierBudget,
    ) -> FrontierOutcome<Self> {
        if components.is_empty() {
            return FrontierOutcome::Empty;
        }
        let mut work = Work::new(budget);
        if let Err(reason) = validate(&components, &mut work) {
            return FrontierOutcome::Unknown(reason);
        }
        let tolerant_alignment = components
            .iter()
            .any(PreparedSlopeFunction::is_machine_zero_length);
        let components: Arc<[PreparedSlopeFunction]> = components.into();
        FrontierOutcome::Complete(Self {
            provenance: Arc::new(FrontierProvenance::Source(components.clone())),
            components,
            tolerant_alignment,
        })
    }
    pub fn components(&self) -> &[PreparedSlopeFunction] {
        &self.components
    }
    pub(super) fn has_machine_zero_length(&self) -> bool {
        self.components
            .iter()
            .any(PreparedSlopeFunction::is_machine_zero_length)
    }
    pub fn provenance(&self) -> &Arc<FrontierProvenance> {
        &self.provenance
    }
    pub fn used_tolerant_alignment(&self) -> bool {
        self.tolerant_alignment
    }
    /// Add active slopes directly, without fitting rounded endpoint values.
    pub fn add(
        &self,
        other: &Self,
        policy: FrontOperationPolicy,
        budget: FrontierBudget,
    ) -> FrontierOutcome<Self> {
        if self
            .components
            .iter()
            .any(PreparedSlopeFunction::has_internal_zero_length)
            || other
                .components
                .iter()
                .any(PreparedSlopeFunction::has_internal_zero_length)
        {
            return FrontierOutcome::Unknown(FrontierUnknown::Unsupported(
                "machine internal zero-length segment addition",
            ));
        }
        if (self.has_machine_zero_length() || other.has_machine_zero_length())
            && (policy != FrontOperationPolicy::FrontAlgebra
                || self.components.len() != 1
                || other.components.len() != 1
                || !self.components[0].is_machine_zero_length())
        {
            return FrontierOutcome::Unknown(FrontierUnknown::Unsupported(
                "machine zero-length segment addition",
            ));
        }
        let mut work = Work::new(budget);
        let mut output = Vec::new();
        let mut tolerant = self.tolerant_alignment || other.tolerant_alignment;
        let (mut i, mut j) = (0, 0);
        while i < self.components.len() && j < other.components.len() {
            if let Err(reason) = work.charge(1) {
                return FrontierOutcome::Unknown(reason);
            }
            let a = &self.components[i];
            let b = &other.components[j];
            match add_component(a, b, policy, &mut work) {
                Ok(Some((component, aligned))) => {
                    if output.len() >= budget.max_components {
                        return FrontierOutcome::Unknown(FrontierUnknown::ComponentLimit);
                    }
                    output.push(component);
                    tolerant |= aligned;
                }
                Ok(None) => (),
                Err(reason) => return FrontierOutcome::Unknown(reason),
            }
            let ae = a.domain().1;
            let be = b.domain().1;
            if ae <= be {
                i += 1;
            }
            if be <= ae {
                j += 1;
            }
        }
        if output.is_empty() {
            return FrontierOutcome::Empty;
        }
        if let Err(reason) = validate(&output, &mut work) {
            return FrontierOutcome::Unknown(reason);
        }
        FrontierOutcome::Complete(Self {
            components: output.into(),
            provenance: Arc::new(FrontierProvenance::Addition {
                left: self.provenance.clone(),
                right: other.provenance.clone(),
                policy,
            }),
            tolerant_alignment: tolerant,
        })
    }
    /// This front is defined wherever `other` is and is no higher there.
    /// A front's value is the minimum over its components. Each of
    /// `other`'s components is checked at its knots merged with this
    /// front's knots inside its domain; one component of this front must
    /// cover each sub-interval by itself (components are continuous, so its
    /// endpoint values bound it). Under the native policy, domains snap and
    /// values compare as the reference does (1e-4, or relative 2.22e-10);
    /// under the strict policy both are exact. Arithmetic failure rejects.
    pub(crate) fn covers_below(&self, other: &Self, policy: FrontOperationPolicy) -> bool {
        self.excess(other, policy) == Some(0.)
    }

    /// The largest amount by which this front exceeds `other` on `other`'s
    /// domain, or None when it does not cover that domain. Checked knot-wise
    /// exactly as `covers_below`; a value no higher than (or, under the
    /// native policy, near) `other` contributes zero.
    pub(crate) fn excess(&self, other: &Self, policy: FrontOperationPolicy) -> Option<f64> {
        if self.has_machine_zero_length() || other.has_machine_zero_length() {
            return None;
        }
        let evaluation = match policy {
            FrontOperationPolicy::StrictDomain => EvaluationPolicy::StrictDomain,
            FrontOperationPolicy::FrontAlgebra => EvaluationPolicy::EndpointTolerance,
        };
        let value = |f: &PreparedSlopeFunction, x: f64| f.evaluate(x, evaluation).ok().flatten();
        let gap = |a: f64, b: f64| {
            if a <= b || near(a, b, policy).unwrap_or(false) {
                Some(0.)
            } else {
                Some(a - b).filter(|gap| gap.is_finite())
            }
        };
        let mut knots = Vec::new();
        let mut worst = 0f64;
        for b in other.components.iter() {
            let (lo, hi) = b.domain();
            knots.clear();
            knots.extend(b.pairs().iter().map(|p| p.1));
            knots.extend(
                self.components
                    .iter()
                    .flat_map(|a| a.pairs().iter().map(|p| p.1))
                    .filter(|&x| lo < x && x < hi),
            );
            knots.sort_by(f64::total_cmp);
            knots.dedup();
            // The least excess of any single component covering [l, h];
            // components are continuous, so endpoint gaps bound it.
            let covers = |l: f64, h: f64| {
                let (bl, bh) = (value(b, l)?, value(b, h)?);
                self.components
                    .iter()
                    .filter_map(|a| {
                        let (al, ah) = (value(a, l)?, value(a, h)?);
                        Some(gap(al, bl)?.max(gap(ah, bh)?))
                    })
                    .min_by(f64::total_cmp)
            };
            if knots.len() == 1 {
                worst = worst.max(covers(knots[0], knots[0])?);
                continue;
            }
            for w in knots.windows(2) {
                worst = worst.max(covers(w[0], w[1])?);
            }
        }
        Some(worst)
    }

    /// Compute the backward wrapper's key after a successful machine
    /// transition. This key is a machine quantity, not a source-domain
    /// objective bound. Mode 4 leaves the wrapper's sentinel in place.
    pub fn machine_backward_key(
        &self,
        mode: i32,
        terminal_ceiling: f64,
        budget: FrontierBudget,
    ) -> FrontierOutcome<f64> {
        let mut work = Work::new(budget);
        if self.components.len() != 1 {
            return FrontierOutcome::Unknown(FrontierUnknown::Unsupported(
                "machine backward key requires one front",
            ));
        }
        if budget.max_components == 0 {
            return FrontierOutcome::Unknown(FrontierUnknown::ComponentLimit);
        }
        let pairs = self.components[0].pairs();
        if pairs.len() > 12 {
            return FrontierOutcome::Unknown(FrontierUnknown::Unsupported(
                "machine backward key pair count",
            ));
        }
        if pairs.len() > budget.max_pairs {
            return FrontierOutcome::Unknown(FrontierUnknown::PairLimit);
        }
        if let Err(reason) = work.charge(1) {
            return FrontierOutcome::Unknown(reason);
        }
        if mode == 4 {
            return FrontierOutcome::Complete(1e12);
        }
        if !terminal_ceiling.is_finite() {
            return FrontierOutcome::Unknown(FrontierUnknown::Unsupported(
                "machine backward key ceiling",
            ));
        }
        let (mut value, mut coordinate) = pairs[0];
        let mut maximum = value;
        for &(slope, end) in &pairs[1..] {
            if let Err(reason) = work.charge(1) {
                return FrontierOutcome::Unknown(reason);
            }
            value = match checked(end - coordinate, "machine backward key span")
                .and_then(|span| checked(span * slope, "machine backward key multiplication"))
                .and_then(|product| checked(value + product, "machine backward key accumulation"))
            {
                Ok(value) => value,
                Err(error) => return FrontierOutcome::Unknown(FrontierUnknown::Arithmetic(error)),
            };
            // MAXSD takes the preceding maximum on an equal comparison.
            if value > maximum {
                maximum = value;
            }
            coordinate = end;
        }
        if let Err(reason) = work.charge(0) {
            return FrontierOutcome::Unknown(reason);
        }
        match checked(
            terminal_ceiling - maximum,
            "machine backward key subtraction",
        ) {
            Ok(key) => FrontierOutcome::Complete(key),
            Err(error) => FrontierOutcome::Unknown(FrontierUnknown::Arithmetic(error)),
        }
    }

    /// Reproduce the stored forward machine key for the recovered single
    /// internal-zero-width front after a successful extension. `mode` is the
    /// reference resource mode integer; mode 4 retains the sentinel. This
    /// value is diagnostic and is not a source-control bound or a witness.
    pub fn machine_internal_forward_key(
        &self,
        mode: i32,
        budget: FrontierBudget,
    ) -> FrontierOutcome<f64> {
        if mode != 4 {
            return match self.machine_internal_forward_minimum(budget) {
                FrontierOutcome::Complete(witness) => FrontierOutcome::Complete(witness.value),
                FrontierOutcome::Empty => FrontierOutcome::Empty,
                FrontierOutcome::Unknown(reason) => FrontierOutcome::Unknown(reason),
            };
        }
        let mut work = Work::new(budget);
        if self.components.len() != 1 || !self.components[0].has_internal_zero_length() {
            return FrontierOutcome::Unknown(FrontierUnknown::Unsupported(
                "machine forward key requires one internal zero-length segment",
            ));
        }
        if budget.max_components < 1 {
            return FrontierOutcome::Unknown(FrontierUnknown::ComponentLimit);
        }
        if budget.max_pairs < 3 {
            return FrontierOutcome::Unknown(FrontierUnknown::PairLimit);
        }
        if let Err(reason) = work.charge(1) {
            return FrontierOutcome::Unknown(reason);
        }
        FrontierOutcome::Complete(1e12)
    }

    /// Ordered endpoint minimum for the recovered single internal-zero-width
    /// machine front. The coordinate identifies the first scanned endpoint
    /// attaining the returned value; it is not a source-control witness, and
    /// endpoint-tolerant reevaluation may select a different value. This is
    /// diagnostic and is not a directed lower bound or route certificate.
    pub fn machine_internal_forward_minimum(
        &self,
        budget: FrontierBudget,
    ) -> FrontierOutcome<FrontAlgebraWitness> {
        let mut work = Work::new(budget);
        if self.components.len() != 1 || !self.components[0].has_internal_zero_length() {
            return FrontierOutcome::Unknown(FrontierUnknown::Unsupported(
                "machine forward minimum requires one internal zero-length segment",
            ));
        }
        if budget.max_components < 1 {
            return FrontierOutcome::Unknown(FrontierUnknown::ComponentLimit);
        }
        if budget.max_pairs < 3 {
            return FrontierOutcome::Unknown(FrontierUnknown::PairLimit);
        }
        if let Err(reason) = work.charge(7) {
            return FrontierOutcome::Unknown(reason);
        }
        let pairs = self.components[0].pairs();
        let (mut value, mut coordinate) = pairs[0];
        let mut minimum = value;
        let mut minimum_coordinate = coordinate;
        for &(slope, end) in &pairs[1..] {
            value = match checked(end - coordinate, "machine key span")
                .and_then(|span| checked(span * slope, "machine key multiplication"))
                .and_then(|product| checked(value + product, "machine key accumulation"))
            {
                Ok(value) => value,
                Err(error) => return FrontierOutcome::Unknown(FrontierUnknown::Arithmetic(error)),
            };
            // MINSD uses the previous minimum as its second operand, so it
            // retains that value on equality, including signed-zero ties.
            if value < minimum {
                minimum = value;
                minimum_coordinate = end;
            }
            coordinate = end;
        }
        if let Err(reason) = work.charge(0) {
            return FrontierOutcome::Unknown(reason);
        }
        FrontierOutcome::Complete(FrontAlgebraWitness {
            coordinate: minimum_coordinate,
            value: minimum,
            component: 0,
            provenance: self.provenance.clone(),
        })
    }

    /// Minimum of the computed slope front, not of a separate source program.
    /// Ordered endpoint accumulation suffices for each monotone slope segment.
    pub fn minimum(&self, budget: FrontierBudget) -> FrontierOutcome<FrontAlgebraWitness> {
        if self.has_machine_zero_length() {
            return FrontierOutcome::Unknown(FrontierUnknown::Unsupported(
                "machine zero-length segment minimum",
            ));
        }
        let mut work = Work::new(budget);
        let mut best: Option<(f64, f64, usize)> = None;
        for (component, function) in self.components.iter().enumerate() {
            let (mut value, mut x) = function.pairs()[0];
            for (index, &(slope, end)) in function.pairs().iter().enumerate() {
                if let Err(reason) = work.charge(1) {
                    return FrontierOutcome::Unknown(reason);
                }
                if index > 0 {
                    let next = checked(end - x, "minimum span")
                        .and_then(|span| checked(slope * span, "minimum multiplication"))
                        .and_then(|product| checked(value + product, "minimum accumulation"));
                    match next {
                        Ok(next) => value = next,
                        Err(error) => {
                            return FrontierOutcome::Unknown(FrontierUnknown::Arithmetic(error))
                        }
                    }
                    x = end;
                }
                if best.as_ref().map_or(true, |old| value < old.1) {
                    best = Some((x, value, component));
                }
            }
        }
        match best {
            Some((coordinate, value, component)) => {
                FrontierOutcome::Complete(FrontAlgebraWitness {
                    coordinate,
                    value,
                    component,
                    provenance: self.provenance.clone(),
                })
            }
            None => FrontierOutcome::Empty,
        }
    }
}
type FrontResult<T> = Result<T, FrontierUnknown>;
struct Work {
    budget: FrontierBudget,
    operations: usize,
    pairs: usize,
    timer: Option<(Instant, Duration)>,
}
impl Work {
    fn new(budget: FrontierBudget) -> Self {
        Self {
            budget,
            operations: 0,
            pairs: 0,
            timer: budget.max_elapsed.map(|limit| (Instant::now(), limit)),
        }
    }
    fn charge(&mut self, count: usize) -> FrontResult<()> {
        self.operations = self
            .operations
            .checked_add(count)
            .ok_or(FrontierUnknown::WorkLimit)?;
        if self.operations > self.budget.max_operations {
            return Err(FrontierUnknown::WorkLimit);
        }
        if self
            .timer
            .is_some_and(|(start, limit)| start.elapsed() >= limit)
        {
            return Err(FrontierUnknown::Deadline);
        }
        Ok(())
    }
    fn pair(&mut self) -> FrontResult<()> {
        self.charge(1)?;
        self.pairs = self
            .pairs
            .checked_add(1)
            .ok_or(FrontierUnknown::PairLimit)?;
        if self.pairs > self.budget.max_pairs {
            Err(FrontierUnknown::PairLimit)
        } else {
            Ok(())
        }
    }
}
fn validate(components: &[PreparedSlopeFunction], work: &mut Work) -> FrontResult<()> {
    if components.len() > work.budget.max_components {
        return Err(FrontierUnknown::ComponentLimit);
    }
    let mut end = None;
    let mut count = 0usize;
    for component in components {
        work.charge(component.pairs().len())?;
        count = count
            .checked_add(component.pairs().len())
            .ok_or(FrontierUnknown::PairLimit)?;
        if count > work.budget.max_pairs {
            return Err(FrontierUnknown::PairLimit);
        }
        if end.is_some_and(|end| component.domain().0 <= end) {
            return Err(FrontierUnknown::Unsupported(
                "overlapping or unordered components require an envelope",
            ));
        }
        end = Some(component.domain().1);
    }
    Ok(())
}
fn checked(value: f64, operation: &'static str) -> Result<f64, RawFunctionError> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err(RawFunctionError::ArithmeticOverflow(operation))
    }
}
fn arithmetic(value: f64, operation: &'static str) -> FrontResult<f64> {
    checked(value, operation).map_err(FrontierUnknown::Arithmetic)
}
fn near(a: f64, b: f64, policy: FrontOperationPolicy) -> FrontResult<bool> {
    if a == b {
        return Ok(true);
    }
    if policy == FrontOperationPolicy::StrictDomain {
        return Ok(false);
    }
    let difference = arithmetic(a - b, "front alignment difference")?.abs();
    if difference < 0.0001 {
        return Ok(true);
    }
    let sum = arithmetic(a + b, "front alignment sum")?.abs();
    Ok(difference < arithmetic(2.220446049250313e-10 * sum, "front alignment band")?)
}
fn value_at(
    function: &PreparedSlopeFunction,
    x: f64,
    policy: FrontOperationPolicy,
) -> FrontResult<f64> {
    let policy = match policy {
        FrontOperationPolicy::StrictDomain => EvaluationPolicy::StrictDomain,
        FrontOperationPolicy::FrontAlgebra => EvaluationPolicy::EndpointTolerance,
    };
    function
        .evaluate(x, policy)
        .map_err(FrontierUnknown::Arithmetic)?
        .ok_or(FrontierUnknown::Unsupported(
            "alignment escaped a source domain",
        ))
}
fn add_component(
    a: &PreparedSlopeFunction,
    b: &PreparedSlopeFunction,
    policy: FrontOperationPolicy,
    work: &mut Work,
) -> FrontResult<Option<(PreparedSlopeFunction, bool)>> {
    let input_pairs = a
        .pairs()
        .len()
        .checked_add(b.pairs().len())
        .ok_or(FrontierUnknown::PairLimit)?;
    if input_pairs > work.budget.max_pairs {
        return Err(FrontierUnknown::PairLimit);
    }
    work.charge(input_pairs)?;
    if a.is_machine_zero_length() {
        return add_machine_zero_length(a, b, policy, work);
    }
    if b.is_machine_zero_length() {
        return Err(FrontierUnknown::Unsupported(
            "machine zero-length segment as vector addition input",
        ));
    }
    let (as_, ae) = a.domain();
    let (bs, be) = b.domain();
    if as_ > be || bs > ae {
        if policy == FrontOperationPolicy::FrontAlgebra
            && (near(as_, be, policy)? || near(bs, ae, policy)?)
        {
            return Err(FrontierUnknown::Unsupported(
                "near-disjoint domains need a topology proof",
            ));
        }
        return Ok(None);
    }
    let same = near(as_, bs, policy)?;
    let start = if same { as_ } else { as_.max(bs) };
    let mut tolerant = same && as_ != bs;
    let (av, bv) = if same {
        (a.pairs()[0].0, b.pairs()[0].0)
    } else {
        (value_at(a, start, policy)?, value_at(b, start, policy)?)
    };
    work.pair()?;
    let mut pairs = vec![(arithmetic(av + bv, "front initial addition")?, start)];
    if as_ == ae || bs == be || start == ae.min(be) {
        if start != ae.min(be) {
            return Err(FrontierUnknown::Unsupported(
                "tolerant singleton alignment is unresolved",
            ));
        }
        return PreparedSlopeFunction::from_prepared_pairs(pairs)
            .map(|f| Some((f, tolerant)))
            .map_err(FrontierUnknown::Arithmetic);
    }
    let (mut i, mut j) = (1, 1);
    while i < a.pairs().len() && a.pairs()[i].1 <= start {
        i += 1;
    }
    while j < b.pairs().len() && b.pairs()[j].1 <= start {
        j += 1;
    }
    let mut previous = start;
    while i < a.pairs().len() && j < b.pairs().len() {
        work.charge(1)?;
        let (sa, xa) = a.pairs()[i];
        let (sb, xb) = b.pairs()[j];
        let aligned = near(xa, xb, policy)?;
        let end = if aligned { xa } else { xa.min(xb) };
        if end <= previous {
            return Err(FrontierUnknown::Unsupported(
                "alignment reversed consecutive coordinates",
            ));
        }
        work.pair()?;
        let slope = arithmetic(sa + sb, "front slope addition")?;
        if policy == FrontOperationPolicy::FrontAlgebra
            && pairs.len() > 1
            && near(slope, pairs[pairs.len() - 1].0, policy)?
        {
            // Native addition keeps the FIRST near-equal summed slope and
            // extends its endpoint. Replacing it by the later slope changes
            // ordered endpoint accumulation, including its last-bit minimum.
            pairs.last_mut().unwrap().1 = end;
        } else {
            pairs.push((slope, end));
        }
        previous = end;
        if aligned {
            tolerant |= xa != xb;
            i += 1;
            j += 1;
        } else if xa < xb {
            i += 1;
        } else {
            j += 1;
        }
    }
    PreparedSlopeFunction::from_prepared_pairs(pairs)
        .map(|f| Some((f, tolerant)))
        .map_err(FrontierUnknown::Arithmetic)
}

/// Recovered fixed-array + vector sum when the fixed input has one zero-width
/// segment. Keep the native segment count even when its output has zero span.
fn add_machine_zero_length(
    fixed: &PreparedSlopeFunction,
    vector: &PreparedSlopeFunction,
    policy: FrontOperationPolicy,
    work: &mut Work,
) -> FrontResult<Option<(PreparedSlopeFunction, bool)>> {
    if policy != FrontOperationPolicy::FrontAlgebra {
        return Err(FrontierUnknown::Unsupported(
            "machine zero-length segment outside front algebra",
        ));
    }
    let x = fixed.pairs()[0].1;
    let (start, end) = vector.domain();
    if (x < start && !near(x, start, policy)?) || (x > end && !near(x, end, policy)?) {
        return Ok(None);
    }
    let same_start = near(x, start, policy)?;
    let (value, segment) = if same_start {
        (vector.pairs()[0].0, 1)
    } else {
        if vector.is_machine_zero_length() {
            return Err(FrontierUnknown::Unsupported(
                "unresolved machine zero-length vector alignment",
            ));
        }
        let value = value_at(vector, x, policy)?;
        let mut index = vector.pairs().partition_point(|pair| pair.1 < x);
        if index < vector.pairs().len() && near(x, vector.pairs()[index].1, policy)? {
            index += 1;
        }
        (value, index)
    };
    let anchor = arithmetic(fixed.pairs()[0].0 + value, "front initial addition")?;
    let slope = if let Some(&(slope, _)) = vector.pairs().get(segment) {
        arithmetic(fixed.pairs()[1].0 + slope, "front slope addition")?
    } else {
        // The native fallback writes a zero slope when the vector is
        // exhausted, regardless of the fixed segment's stored slope.
        0.
    };
    work.pair()?;
    work.pair()?;
    let output = PreparedSlopeFunction::machine_zero_length(anchor, slope, x, fixed.pairs()[1].1)
        .map_err(FrontierUnknown::Arithmetic)?;
    Ok(Some((output, x != start && same_start)))
}

#[cfg(test)]
#[path = "../../../../tests/unit/graph/coupled/descriptor/frontier/tests.rs"]
mod tests;
