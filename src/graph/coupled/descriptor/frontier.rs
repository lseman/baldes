//! Slope FRONT algebra; deliberately distinct from source-expression replay.
use super::{
    EvaluationPolicy, PairedFrontOperations, PreparedSlopeFunction, RawAffine, RawFunctionError,
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
    /// Bounded machine waiting path, retaining the reference predecessor bit.
    /// This is not a source-control witness or directed certificate.
    PairedMachineWaiting {
        input: Arc<Self>,
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

mod root;
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
    root: Arc<MachinePathNode>,
    len: usize,
    max_states: usize,
}

mod history;

mod forward;

mod waiting;
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
            if function.has_machine_only_geometry() {
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
mod algebra;
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
    if a.has_machine_only_geometry() {
        return add_machine_zero_length(a, b, policy, work);
    }
    if b.has_machine_only_geometry() {
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
    if fixed.pairs().len() != 2 || fixed.pairs()[0].1 != fixed.pairs()[1].1 {
        return Err(FrontierUnknown::Unsupported(
            "machine zero-length addition requires a single point segment",
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
        if vector.has_machine_only_geometry() {
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
