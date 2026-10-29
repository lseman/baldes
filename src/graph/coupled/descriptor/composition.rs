//! Ordered slope-function inverse and secant composition.
//!
//! These operations construct a front algebra, not a directed certificate for
//! every sequential floating-point evaluation of the input functions.
use super::{
    FrontOperationPolicy, FrontierBudget, FrontierOutcome, FrontierUnknown, PreparedSlopeFunction,
    RawFunctionError,
};
use std::{
    borrow::Cow,
    sync::Arc,
    time::{Duration, Instant},
};

type Result<T> = std::result::Result<T, FrontierUnknown>;
mod machine;

struct Work {
    budget: FrontierBudget,
    operations: usize,
    timer: Option<(Instant, Duration)>,
}
impl Work {
    fn new(budget: FrontierBudget) -> Self {
        Self {
            budget,
            operations: 0,
            timer: budget.max_elapsed.map(|d| (Instant::now(), d)),
        }
    }
    fn charge(&mut self, count: usize) -> Result<()> {
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
    fn pairs(&mut self, count: usize) -> Result<()> {
        self.charge(1)?;
        if self.budget.max_components == 0 {
            return Err(FrontierUnknown::ComponentLimit);
        }
        if count > self.budget.max_pairs {
            return Err(FrontierUnknown::PairLimit);
        }
        Ok(())
    }
}
fn checked(value: f64, name: &'static str) -> Result<f64> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err(FrontierUnknown::Arithmetic(
            RawFunctionError::ArithmeticOverflow(name),
        ))
    }
}
fn unsupported(message: &'static str) -> FrontierUnknown {
    FrontierUnknown::Unsupported(message)
}
fn finish(value: Result<Option<PreparedSlopeFunction>>) -> FrontierOutcome<PreparedSlopeFunction> {
    match value {
        Ok(Some(f)) => FrontierOutcome::Complete(f),
        Ok(None) => FrontierOutcome::Empty,
        Err(e) => FrontierOutcome::Unknown(e),
    }
}
fn prepared(pairs: Vec<(f64, f64)>) -> Result<PreparedSlopeFunction> {
    PreparedSlopeFunction::from_prepared_pairs(pairs).map_err(FrontierUnknown::Arithmetic)
}

/// Invert an injective increasing prepared map. Reciprocal slopes are retained;
/// they are not fitted from rounded endpoint differences. A point maps to a point.
pub fn inverse_increasing(
    function: &PreparedSlopeFunction,
    budget: FrontierBudget,
) -> FrontierOutcome<PreparedSlopeFunction> {
    finish((|| {
        if function.is_machine_zero_length() {
            return Err(unsupported("machine zero-length segment inversion"));
        }
        let mut work = Work::new(budget);
        work.pairs(function.pairs().len())?;
        work.charge(function.pairs().len())?;
        let (mut y, mut x) = function.pairs()[0];
        let mut output = Vec::with_capacity(function.pairs().len());
        output.push((x, y));
        for &(slope, end) in &function.pairs()[1..] {
            work.charge(1)?;
            if slope <= 0. {
                return Err(unsupported("inverse requires positive slopes"));
            }
            let span = checked(end - x, "inverse span")?;
            let product = checked(span * slope, "inverse multiplication")?;
            let mapped = checked(y + product, "inverse endpoint")?;
            if mapped <= y {
                return Err(unsupported("inverse mapped coordinates collapsed"));
            }
            let reciprocal = checked(1. / slope, "inverse reciprocal")?;
            output.push((reciprocal, mapped));
            x = end;
            y = mapped;
        }
        Ok(Some(prepared(output)?))
    })())
}

/// Literal vector inversion used by machine arc initialization. Unlike source
/// inversion, this retains representable collapsed endpoint coordinates. A
/// nonfinite reciprocal or an unrepresentable array remains unknown.
pub fn machine_inverse(
    function: &PreparedSlopeFunction,
    budget: FrontierBudget,
) -> FrontierOutcome<PreparedSlopeFunction> {
    finish((|| {
        let mut work = Work::new(budget);
        work.pairs(function.pairs().len())?;
        work.charge(function.pairs().len().saturating_mul(6))?;
        let (mut value, mut coordinate) = function.pairs()[0];
        let mut output = Vec::with_capacity(function.pairs().len());
        output.push((coordinate, value));
        for &(slope, end) in &function.pairs()[1..] {
            work.charge(0)?;
            let span = checked(end - coordinate, "machine inverse span")?;
            let product = checked(span * slope, "machine inverse multiplication")?;
            value = checked(value + product, "machine inverse endpoint")?;
            let reciprocal = checked(1. / slope, "machine inverse reciprocal")?;
            output.push((reciprocal, value));
            coordinate = end;
        }
        let result = PreparedSlopeFunction::from_machine_pairs(output)
            .map_err(FrontierUnknown::Arithmetic)?;
        work.charge(0)?;
        Ok(Some(result))
    })())
}

/// The three distinct machine-program vectors prepared for one paired arc.
/// These are intermediate arrays, not a source-control feasibility certificate.
#[derive(Clone, Debug)]
pub struct MachinePreparedInputs {
    pub forward: PreparedSlopeFunction,
    pub inverse: PreparedSlopeFunction,
    pub sum: PreparedSlopeFunction,
}

/// Bounded machine values computed while initializing one paired arc.
/// These values are not source-domain feasibility or pricing certificates.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MachineInitialLimits {
    /// The prepared forward lookup at the tail lower bound.
    pub forward_at_tail_lower: f64,
    /// The corrected first scalar used by forward continuation.
    pub tail_limit: f64,
    /// The corrected mapped upper value retained by the initializer.
    pub mapped_upper: f64,
}

/// Prepared arc vectors and finite initialization limits. These are machine
/// program inputs, not a source-control or pricing certificate.
#[derive(Clone, Debug)]
pub struct MachineArcPreparation {
    pub inputs: MachinePreparedInputs,
    pub limits: MachineInitialLimits,
}

/// Prepare machine vectors from explicitly supplied, already prepared inputs
/// and compute the reference initializer's bounded limit branch. This does not
/// infer truncation, dependent map entries, resource identity, or graph bounds.
/// Both stages share one elapsed deadline and conservatively split work.
pub fn prepare_machine_arc(
    truncated: &PreparedSlopeFunction,
    dependent: &PreparedSlopeFunction,
    tail_lower: f64,
    tail_upper: f64,
    head_upper: f64,
    budget: FrontierBudget,
) -> FrontierOutcome<MachineArcPreparation> {
    let started = Instant::now();
    let per_stage = budget.max_operations / 2;
    if per_stage == 0 {
        return FrontierOutcome::Unknown(FrontierUnknown::WorkLimit);
    }
    let stage_budget = || -> Result<FrontierBudget> {
        let remaining = budget
            .max_elapsed
            .map(|limit| limit.saturating_sub(started.elapsed()));
        if remaining.is_some_and(|limit| limit.is_zero()) {
            return Err(FrontierUnknown::Deadline);
        }
        Ok(FrontierBudget {
            max_operations: per_stage,
            max_elapsed: remaining,
            ..budget
        })
    };
    let first_budget = match stage_budget() {
        Ok(value) => value,
        Err(reason) => return FrontierOutcome::Unknown(reason),
    };
    let inputs = match prepare_machine_inputs(truncated, dependent, first_budget) {
        FrontierOutcome::Complete(value) => value,
        FrontierOutcome::Empty => {
            return FrontierOutcome::Unknown(unsupported("machine arc preparation empty"));
        }
        FrontierOutcome::Unknown(reason) => return FrontierOutcome::Unknown(reason),
    };
    let second_budget = match stage_budget() {
        Ok(value) => value,
        Err(reason) => return FrontierOutcome::Unknown(reason),
    };
    let limits = match machine_initial_limits(
        &inputs.forward,
        &inputs.inverse,
        tail_lower,
        tail_upper,
        head_upper,
        second_budget,
    ) {
        FrontierOutcome::Complete(value) => value,
        FrontierOutcome::Empty => {
            return FrontierOutcome::Unknown(unsupported("machine arc limits empty"));
        }
        FrontierOutcome::Unknown(reason) => return FrontierOutcome::Unknown(reason),
    };
    if budget
        .max_elapsed
        .is_some_and(|limit| started.elapsed() >= limit)
    {
        return FrontierOutcome::Unknown(FrontierUnknown::Deadline);
    }
    FrontierOutcome::Complete(MachineArcPreparation { inputs, limits })
}

/// Compute the recovered finite initial-limit branch using separately
/// prepared forward and inverse vectors. Lookup sentinel `1e12` is treated
/// exactly like an out-of-domain return, including a genuine value of `1e12`.
/// Unknown includes the reference's diagnostic branches and does not assert
/// source infeasibility. This does not install inputs or invoke pricing.
pub fn machine_initial_limits(
    forward: &PreparedSlopeFunction,
    inverse: &PreparedSlopeFunction,
    tail_lower: f64,
    tail_upper: f64,
    head_upper: f64,
    budget: FrontierBudget,
) -> FrontierOutcome<MachineInitialLimits> {
    let result = (|| {
        if !tail_lower.is_finite() || !tail_upper.is_finite() || !head_upper.is_finite() {
            return Err(unsupported("machine initial limit inputs"));
        }
        let mut work = Work::new(budget);
        work.pairs(forward.pairs().len())?;
        work.pairs(inverse.pairs().len())?;
        // Reserve both forward lookups, the inverse lookup and branch work.
        let operations = forward
            .pairs()
            .len()
            .checked_mul(2)
            .and_then(|n| n.checked_add(inverse.pairs().len()))
            .and_then(|n| n.checked_add(4))
            .ok_or(FrontierUnknown::WorkLimit)?;
        work.charge(operations)?;
        let lookup = |function: &PreparedSlopeFunction, coordinate: f64| -> Result<Option<f64>> {
            function
                .machine_value_and_segment(coordinate)
                .map(|value| value.map(|(value, _)| value).filter(|&value| value != 1e12))
                .map_err(FrontierUnknown::Arithmetic)
        };
        let first = lookup(forward, tail_lower)?
            .ok_or_else(|| unsupported("machine initial lower lookup sentinel"))?;
        let inverse_upper = lookup(inverse, head_upper)?;
        let forward_upper = lookup(forward, tail_upper)?;
        // MINSD takes its second operand on equality, including signed zero.
        let min_second = |first: f64, second: f64| {
            if first < second {
                first
            } else {
                second
            }
        };
        let (tail_limit, mapped_upper) = match (inverse_upper, forward_upper) {
            (Some(z), Some(y)) => (min_second(tail_upper, z), min_second(head_upper, y)),
            (None, Some(y)) => (tail_upper, y),
            (Some(z), None) => (z, head_upper),
            (None, None) => return Err(unsupported("machine initial upper lookup sentinels")),
        };
        work.charge(0)?;
        Ok(MachineInitialLimits {
            forward_at_tail_lower: first,
            tail_limit,
            mapped_upper,
        })
    })();
    match result {
        Ok(limits) => FrontierOutcome::Complete(limits),
        Err(reason) => FrontierOutcome::Unknown(reason),
    }
}

/// Prepare the recovered finite machine vectors from an already truncated
/// base vector and an already prepared dependent vector. This does not perform
/// the reference's truncation, map lookup or graph initialization. Its bounded
/// inverse and composition operations can return unknown for machine behavior
/// that the prepared representation cannot express.
pub fn prepare_machine_inputs(
    truncated: &PreparedSlopeFunction,
    dependent: &PreparedSlopeFunction,
    budget: FrontierBudget,
) -> FrontierOutcome<MachinePreparedInputs> {
    let started = Instant::now();
    let per_stage = budget.max_operations / 4;
    let forward_work = match truncated.pairs().len().checked_mul(2) {
        Some(value) => value,
        None => return FrontierOutcome::Unknown(FrontierUnknown::WorkLimit),
    };
    if per_stage < forward_work {
        return FrontierOutcome::Unknown(FrontierUnknown::WorkLimit);
    }
    if budget.max_components == 0 {
        return FrontierOutcome::Unknown(FrontierUnknown::ComponentLimit);
    }
    if truncated.pairs().len() > budget.max_pairs || dependent.pairs().len() > budget.max_pairs {
        return FrontierOutcome::Unknown(FrontierUnknown::PairLimit);
    }
    let remaining = || -> std::result::Result<FrontierBudget, FrontierUnknown> {
        let remaining = budget
            .max_elapsed
            .map(|limit| limit.saturating_sub(started.elapsed()));
        if remaining.is_some_and(|limit| limit.is_zero()) {
            return Err(FrontierUnknown::Deadline);
        }
        Ok(FrontierBudget {
            max_operations: per_stage,
            max_elapsed: remaining,
            ..budget
        })
    };
    if let Err(reason) = remaining() {
        return FrontierOutcome::Unknown(reason);
    }
    let forward = match truncated.machine_forward_from_truncated() {
        Ok(value) => value,
        Err(error) => return FrontierOutcome::Unknown(FrontierUnknown::Arithmetic(error)),
    };
    let inverse_budget = match remaining() {
        Ok(value) => value,
        Err(reason) => return FrontierOutcome::Unknown(reason),
    };
    let inverse = match machine_inverse(&forward, inverse_budget) {
        FrontierOutcome::Complete(value) => value,
        FrontierOutcome::Empty => {
            return FrontierOutcome::Unknown(unsupported("machine inverse preparation empty"))
        }
        FrontierOutcome::Unknown(reason) => return FrontierOutcome::Unknown(reason),
    };
    let (lower, upper) = inverse.domain();
    let composition_budget = match remaining() {
        Ok(value) => value,
        Err(reason) => return FrontierOutcome::Unknown(reason),
    };
    let sum = match compose_increasing(
        dependent,
        &inverse,
        lower,
        upper,
        FrontOperationPolicy::FrontAlgebra,
        composition_budget,
    ) {
        FrontierOutcome::Complete(value) => value,
        FrontierOutcome::Empty => {
            return FrontierOutcome::Unknown(unsupported("machine sum preparation empty"))
        }
        FrontierOutcome::Unknown(reason) => return FrontierOutcome::Unknown(reason),
    };
    // The initializer checks the composed fixed array before consuming it.
    // Its first segment is exempt from coordinate-separation checks; later
    // coordinates must strictly increase outside the machine tolerance band.
    if !(2..=12).contains(&sum.pairs().len()) || sum.pairs()[0].0 == 1e12 {
        return FrontierOutcome::Unknown(unsupported("machine prepared sum output check"));
    }
    let check_budget = match remaining() {
        Ok(value) => value,
        Err(reason) => return FrontierOutcome::Unknown(reason),
    };
    let mut work = Work::new(check_budget);
    if let Err(reason) = work.charge(sum.pairs().len().saturating_mul(6)) {
        return FrontierOutcome::Unknown(reason);
    }
    for index in 2..sum.pairs().len() {
        let current = sum.pairs()[index].1;
        let previous = sum.pairs()[index - 1].1;
        match near(current, previous) {
            Ok(false) if current > previous => {}
            Ok(_) => {
                return FrontierOutcome::Unknown(unsupported(
                    "machine prepared sum consecutive coordinates",
                ))
            }
            Err(reason) => return FrontierOutcome::Unknown(reason),
        }
        if let Err(reason) = work.charge(0) {
            return FrontierOutcome::Unknown(reason);
        }
    }
    if let Err(reason) = remaining() {
        return FrontierOutcome::Unknown(reason);
    }
    FrontierOutcome::Complete(MachinePreparedInputs {
        forward,
        inverse,
        sum,
    })
}

struct Cached<'a> {
    function: &'a PreparedSlopeFunction,
    knots: Cow<'a, [(f64, f64)]>, // coordinate, ordered accumulated value
}
impl<'a> Cached<'a> {
    fn new(function: &'a PreparedSlopeFunction, increasing: bool, work: &mut Work) -> Result<Self> {
        if function.is_machine_zero_length() {
            return Err(unsupported("machine zero-length segment composition"));
        }
        work.pairs(function.pairs().len())?;
        work.charge(function.pairs().len())?;
        let (mut y, mut x) = function.pairs()[0];
        let mut knots = Vec::with_capacity(function.pairs().len());
        knots.push((x, y));
        for &(slope, end) in &function.pairs()[1..] {
            if increasing && slope <= 0. {
                return Err(unsupported(
                    "composition inner map requires positive slopes",
                ));
            }
            let span = checked(end - x, "composition source span")?;
            let next = checked(
                y + checked(slope * span, "composition source multiplication")?,
                "composition source endpoint",
            )?;
            if increasing && next <= y {
                return Err(unsupported("composition inner coordinates collapsed"));
            }
            knots.push((end, next));
            x = end;
            y = next;
        }
        Ok(Self {
            function,
            knots: knots.into(),
        })
    }
    fn value(&self, argument: f64, policy: FrontOperationPolicy, work: &mut Work) -> Result<f64> {
        work.charge(1 + self.knots.len().ilog2() as usize)?;
        let (lower, upper) = self.function.domain();
        if argument < lower || argument > upper {
            return Err(unsupported("composition endpoint outside outer domain"));
        }
        if argument == lower
            || (policy == FrontOperationPolicy::FrontAlgebra && near(argument, lower)?)
        {
            return Ok(self.knots[0].1);
        }
        let index = self.knots.partition_point(|&(x, _)| x < argument);
        if self.knots[index].0 == argument {
            return Ok(self.knots[index].1);
        }
        let (x, y) = self.knots[index - 1];
        let slope = self.function.pairs()[index].0;
        checked(
            y + checked(
                slope * checked(argument - x, "composition evaluation span")?,
                "composition evaluation multiplication",
            )?,
            "composition evaluation addition",
        )
    }
}
fn near(a: f64, b: f64) -> Result<bool> {
    let difference = checked(a - b, "composition comparison difference")?.abs();
    if difference < 0.0001 {
        return Ok(true);
    }
    Ok(difference
        < checked(
            2.220446049250313e-10 * checked(a + b, "composition comparison sum")?.abs(),
            "composition comparison band",
        )?)
}

/// Construct `outer(inner(x))` using the selected domain policy.
/// Strict composition intersects both domains and requires an increasing inner
/// map. Front algebra follows the ordered machine scan, including constant and
/// degenerate inner maps. Output slopes use ordered endpoint secants.
/// Raw front arithmetic is not a strict source-function certificate.
pub fn compose_increasing(
    outer: &PreparedSlopeFunction,
    inner: &PreparedSlopeFunction,
    lower: f64,
    upper: f64,
    policy: FrontOperationPolicy,
    budget: FrontierBudget,
) -> FrontierOutcome<PreparedSlopeFunction> {
    finish(compose_checked(outer, inner, lower, upper, policy, budget))
}

/// Immutable inner map and its ordered endpoint values, prepared once per arc.
/// Clones share both arrays; composing a label does not rescan the inner map.
#[derive(Clone, Debug)]
pub struct PreparedCompositionMap {
    function: Arc<PreparedSlopeFunction>,
    knots: Arc<[(f64, f64)]>,
}
impl PreparedCompositionMap {
    pub fn new(inner: &PreparedSlopeFunction, budget: FrontierBudget) -> FrontierOutcome<Self> {
        let mut work = Work::new(budget);
        match Cached::new(inner, true, &mut work) {
            Ok(cached) => FrontierOutcome::Complete(Self {
                function: Arc::new(inner.clone()),
                knots: cached.knots.into_owned().into(),
            }),
            Err(reason) => FrontierOutcome::Unknown(reason),
        }
    }
    pub fn compose(
        &self,
        outer: &PreparedSlopeFunction,
        lower: f64,
        upper: f64,
        policy: FrontOperationPolicy,
        budget: FrontierBudget,
    ) -> FrontierOutcome<PreparedSlopeFunction> {
        self.compose_with_clipping(outer, lower, upper, policy, false, budget)
    }

    /// Paired relations explicitly intersect a component's feasible preimage.
    /// Preserve their ordered native arithmetic while permitting that clipping;
    /// this is a relation operation, not a standalone native-helper witness.
    pub(super) fn compose_clipped(
        &self,
        outer: &PreparedSlopeFunction,
        lower: f64,
        upper: f64,
        policy: FrontOperationPolicy,
        budget: FrontierBudget,
    ) -> FrontierOutcome<PreparedSlopeFunction> {
        self.compose_with_clipping(outer, lower, upper, policy, true, budget)
    }

    fn compose_with_clipping(
        &self,
        outer: &PreparedSlopeFunction,
        lower: f64,
        upper: f64,
        policy: FrontOperationPolicy,
        allow_outer_clipping: bool,
        budget: FrontierBudget,
    ) -> FrontierOutcome<PreparedSlopeFunction> {
        let mut work = Work::new(budget);
        if policy == FrontOperationPolicy::FrontAlgebra && !allow_outer_clipping {
            return finish(machine::compose(
                outer,
                &self.function,
                lower,
                upper,
                &mut work,
            ));
        }
        if let Err(reason) = work.pairs(self.knots.len()) {
            return FrontierOutcome::Unknown(reason);
        }
        let inner = Cached {
            function: &self.function,
            knots: Cow::Borrowed(&self.knots),
        };
        finish(compose_cached(
            outer,
            &inner,
            lower,
            upper,
            policy,
            allow_outer_clipping,
            &mut work,
        ))
    }
}

fn compose_checked(
    outer: &PreparedSlopeFunction,
    inner: &PreparedSlopeFunction,
    lower: f64,
    upper: f64,
    policy: FrontOperationPolicy,
    budget: FrontierBudget,
) -> Result<Option<PreparedSlopeFunction>> {
    let mut work = Work::new(budget);
    if policy == FrontOperationPolicy::FrontAlgebra {
        return machine::compose(outer, inner, lower, upper, &mut work);
    }
    let inner = Cached::new(inner, true, &mut work)?;
    compose_cached(outer, &inner, lower, upper, policy, false, &mut work)
}

/// The short-interval branch performs only two lookups. In particular, it
/// never walks or divides a zero-length segment retained by machine output.
fn short_composition(
    outer: &PreparedSlopeFunction,
    inner: &PreparedSlopeFunction,
    lower: f64,
    upper: f64,
    work: &mut Work,
) -> Result<PreparedSlopeFunction> {
    work.pairs(inner.pairs().len())?;
    work.pairs(outer.pairs().len())?;
    work.charge(inner.pairs().len() + outer.pairs().len())?;
    let inner_value = inner
        .evaluate(lower, super::EvaluationPolicy::EndpointTolerance)
        .map_err(FrontierUnknown::Arithmetic)?
        .ok_or_else(|| unsupported("native short inner evaluation outside domain"))?;
    if inner_value == 1e12 {
        return Err(unsupported("native composition inner sentinel"));
    }
    let value = outer
        .evaluate(inner_value, super::EvaluationPolicy::EndpointTolerance)
        .map_err(FrontierUnknown::Arithmetic)?
        .ok_or_else(|| unsupported("native short outer evaluation outside domain"))?;
    if value == 1e12 {
        return Err(unsupported("native composition outer sentinel"));
    }
    work.pairs(2)?;
    if lower == upper {
        PreparedSlopeFunction::machine_zero_length(value, 0., lower, upper)
            .map_err(FrontierUnknown::Arithmetic)
    } else {
        prepared(vec![(value, lower), (0., upper)])
    }
}
fn compose_cached(
    outer: &PreparedSlopeFunction,
    inner: &Cached<'_>,
    lower: f64,
    upper: f64,
    policy: FrontOperationPolicy,
    allow_outer_clipping: bool,
    work: &mut Work,
) -> Result<Option<PreparedSlopeFunction>> {
    if !lower.is_finite() || !upper.is_finite() || lower > upper {
        return Err(unsupported("composition needs a finite ordered interval"));
    }
    // The machine helper selects this branch from the requested endpoints,
    // before clipping either endpoint to the inner domain.
    if policy == FrontOperationPolicy::FrontAlgebra && near(lower, upper)? {
        return short_composition(outer, inner.function, lower, upper, work).map(Some);
    }
    let outer = Cached::new(outer, false, work)?;
    if policy == FrontOperationPolicy::FrontAlgebra
        && inner.knots.len() == 1
        && near(lower, inner.knots[0].0)?
        && (upper >= inner.knots[0].0 || near(upper, inner.knots[0].0)?)
    {
        // A point map exhausts its segment stream immediately after the
        // initial lookup. The long-interval branch retains the requested
        // start coordinate; it does not extend a constant to the requested
        // end or move the point to the inner domain's coordinate.
        work.charge(outer.function.pairs().len())?;
        let value = outer
            .function
            .evaluate(inner.knots[0].1, super::EvaluationPolicy::EndpointTolerance)
            .map_err(FrontierUnknown::Arithmetic)?
            .ok_or_else(|| unsupported("machine point composition outside outer domain"))?;
        if value == 1e12 {
            return Err(unsupported("native composition outer sentinel"));
        }
        work.pairs(1)?;
        return Ok(Some(prepared(vec![(value, lower)])?));
    }
    if policy == FrontOperationPolicy::FrontAlgebra
        && inner.knots.len() > 1
        && upper >= inner.function.domain().1
        && lower > inner.knots[inner.knots.len() - 2].0
        && near(lower, inner.function.domain().1)?
        && !near(lower, inner.function.domain().0)?
    {
        work.charge(inner.function.pairs().len() + outer.function.pairs().len())?;
        let inner_value = inner
            .function
            .evaluate(lower, super::EvaluationPolicy::EndpointTolerance)
            .map_err(FrontierUnknown::Arithmetic)?
            .ok_or_else(|| unsupported("native snapped upper inner evaluation"))?;
        // On this long path the reference passes the inner value through to
        // the outer helper, including a literal 1e12; only the outer result is
        // treated as a sentinel here.
        let value = outer
            .function
            .evaluate(inner_value, super::EvaluationPolicy::EndpointTolerance)
            .map_err(FrontierUnknown::Arithmetic)?
            .ok_or_else(|| unsupported("native snapped upper outer evaluation"))?;
        if value == 1e12 {
            return Err(unsupported("native composition outer sentinel"));
        }
        work.pairs(1)?;
        return Ok(Some(prepared(vec![(value, lower)])?));
    }
    let snapped_start = policy == FrontOperationPolicy::FrontAlgebra
        && lower < inner.function.domain().0
        && near(lower, inner.function.domain().0)?;
    if snapped_start
        && (inner.knots.len() == 1
            || (upper < inner.function.domain().0 && !near(upper, inner.function.domain().0)?))
    {
        return Err(unsupported("native near-disjoint composition interval"));
    }
    // The front-algebra helper preserves the requested coordinate while
    // taking the first inner value as the anchor for a near-lower start.
    let lo = if snapped_start {
        lower
    } else {
        lower.max(inner.function.domain().0)
    };
    let hi = upper.min(inner.function.domain().1);
    if lo > hi {
        if policy == FrontOperationPolicy::FrontAlgebra && near(lo, hi)? {
            return Err(unsupported("native near-disjoint composition interval"));
        }
        return Ok(None);
    }
    // A clipped short interval does not imply that the requested interval
    // took the native short branch; its remaining path is not yet recovered.
    if policy == FrontOperationPolicy::FrontAlgebra && near(lo, hi)? {
        return Err(unsupported("native clipped short composition interval"));
    }
    let (outer_lo, outer_hi) = outer.function.domain();
    if inner.knots.len() == 1 {
        let (x, y) = inner.knots[0];
        if y < outer_lo || y > outer_hi {
            if policy == FrontOperationPolicy::FrontAlgebra
                && (near(y, outer_lo)? || near(y, outer_hi)?)
            {
                return Err(unsupported("native near-disjoint point composition"));
            }
            return Ok(None);
        }
        return Ok(Some(prepared(vec![(outer.value(y, policy, work)?, x)])?));
    }
    let mut output = Vec::new();
    let mut last: Option<(f64, f64)> = None;
    let mut inner_endpoint = None;
    let mut outer_anchor = None;
    let mut outer_next = None;
    for index in 1..inner.knots.len() {
        work.charge(1)?;
        let (gx, gy) = inner.knots[index - 1];
        let (ge, _) = inner.knots[index];
        let slope = inner.function.pairs()[index].0;
        let mut x0 = if snapped_start && index == 1 {
            lo
        } else {
            gx.max(lo)
        };
        let mut x1 = ge.min(hi);
        if x0 > x1 {
            continue;
        }
        // Native front algebra advances from the clipped interval start,
        // carrying the rounded endpoint into the following segment. Reusing
        // the original knot anchor is mathematically equivalent, but can
        // change both a crossing and the final secant by a binary64 bit.
        let (anchor_x, anchor_y) = if policy == FrontOperationPolicy::FrontAlgebra {
            let y = match inner_endpoint {
                Some(y) => y,
                None if near(lo, inner.function.domain().0)? => inner.knots[0].1,
                None => segment_value(gx, gy, slope, x0)?,
            };
            (x0, y)
        } else {
            (gx, gy)
        };
        let mut y0 = segment_value(anchor_x, anchor_y, slope, x0)?;
        let mut y1 = segment_value(anchor_x, anchor_y, slope, x1)?;
        inner_endpoint = Some(y1);
        if y0 > outer_hi || y1 < outer_lo {
            if policy == FrontOperationPolicy::FrontAlgebra
                && (near(y0, outer_hi)? || near(y1, outer_lo)?)
            {
                return Err(unsupported("native near-disjoint composition image"));
            }
            continue;
        }
        if policy == FrontOperationPolicy::FrontAlgebra
            && !allow_outer_clipping
            && (y0 < outer_lo || y1 > outer_hi)
        {
            // The native helper can reject an exhausted outer domain even
            // when a nonempty clipped composition exists. Its false result
            // does not certify infeasibility. Keep strict-domain clipping
            // below, but do not publish a native witness for this case.
            return Err(unsupported("native composition image outside outer domain"));
        }
        if y0 < outer_lo {
            x0 = crossing(anchor_x, anchor_y, slope, outer_lo)?;
            y0 = outer_lo;
        }
        if y1 > outer_hi {
            x1 = crossing(anchor_x, anchor_y, slope, outer_hi)?;
            y1 = outer_hi;
        }
        if x0 < lo || x1 > hi || x0 > x1 || (x0 == x1 && y0 != y1) {
            return Err(unsupported("rounded composition domain crossing"));
        }
        let native = policy == FrontOperationPolicy::FrontAlgebra;
        let f0 = if native {
            match last {
                // A skipped short inner remainder does not advance the
                // output anchor before the next inner segment starts.
                Some((x, value)) if x == x0 || (x < x0 && near(x, x0)?) => value,
                _ => outer.value(y0, policy, work)?,
            }
        } else {
            outer.value(y0, policy, work)?
        };
        if native && last.is_none() && f0 == 1e12 {
            return Err(unsupported("native composition outer sentinel"));
        }
        if native && outer_anchor.is_none() {
            outer_anchor = Some((y0, f0));
        }
        match last {
            None => {
                work.pairs(1)?;
                output.push((f0, x0));
                last = Some((x0, f0));
            }
            Some((x, y)) if y == f0 && (x == x0 || (native && x < x0 && near(x, x0)?)) => (),
            _ => {
                return Err(unsupported(
                    "composition components do not meet continuously",
                ))
            }
        }
        let mut next = if native {
            match outer_next {
                Some(next) => next,
                None => {
                    // getYvalueAndSegId advances past a near upper knot but
                    // returns the value at the actual argument. Its first
                    // coordinate instead snaps both the value and segment.
                    let at = outer.knots.partition_point(|&(x, _)| x < y0);
                    if near(y0, outer_lo)? {
                        1
                    } else if at < outer.knots.len() && near(outer.knots[at].0, y0)? {
                        at + 1
                    } else {
                        at
                    }
                }
            }
        } else {
            outer.knots.partition_point(|&(x, _)| x <= y0)
        };
        while next < outer.knots.len()
            && (outer.knots[next].0 < y1 || (native && near(outer.knots[next].0, y1)?))
        {
            let (target, cached_value) = outer.knots[next];
            let value = if let Some((y, value)) = outer_anchor {
                let value = segment_value(y, value, outer.function.pairs()[next].0, target)?;
                outer_anchor = Some((target, value));
                value
            } else {
                cached_value
            };
            let x = if native && near(target, y1)? {
                x1
            } else {
                crossing(anchor_x, anchor_y, slope, target)?
            };
            if x < x0 || x > x1 {
                return Err(unsupported("rounded composition knot crossing"));
            }
            emit(&mut output, &mut last, x, value, policy, false, work)?;
            next += 1;
        }
        let end = if native && last.is_some_and(|(x, _)| x == x1) {
            // A snapped outer knot already emitted this endpoint. The native
            // inner-end branch skips it and retains the knot's accumulated value.
            last.unwrap().1
        } else if let Some((y, value)) = outer_anchor {
            // At an exact final knot `next` still names its incoming segment.
            // Once it was consumed, use the following segment's slope.
            if y1 == y {
                value
            } else {
                let slope = outer
                    .function
                    .pairs()
                    .get(next)
                    .ok_or_else(|| unsupported("native composition outside outer domain"))?
                    .0;
                segment_value(y, value, slope, y1)?
            }
        } else {
            outer.value(y1, policy, work)?
        };
        if native && outer.knots.get(next).is_some_and(|&(knot, _)| knot == y1) {
            outer_anchor = Some((y1, end));
        }
        if native {
            outer_next = Some(next);
        }
        let inner_endpoint =
            !(native && outer.knots.get(next).is_some_and(|&(knot, _)| knot == y1));
        emit(
            &mut output,
            &mut last,
            x1,
            end,
            policy,
            inner_endpoint,
            work,
        )?;
        // A tolerance-sized inner remainder emits no endpoint. The next
        // segment keeps that earlier output anchor; once it is near the
        // requested end, the helper stops without manufacturing a last piece.
        if native {
            if let Some((x, _)) = last {
                if near(x, hi)? {
                    break;
                }
            }
        }
    }
    if output.is_empty() {
        Ok(None)
    } else {
        Ok(Some(prepared(output)?))
    }
}
fn segment_value(x: f64, y: f64, slope: f64, argument: f64) -> Result<f64> {
    if argument == x {
        return Ok(y);
    }
    let span = checked(argument - x, "composition inner span")?;
    let product = checked(span * slope, "composition inner product")?;
    checked(y + product, "composition inner value")
}
fn crossing(x: f64, y: f64, slope: f64, target: f64) -> Result<f64> {
    let difference = checked(target - y, "composition crossing difference")?;
    let quotient = checked(difference / slope, "composition crossing division")?;
    checked(x + quotient, "composition crossing addition")
}
fn emit(
    output: &mut Vec<(f64, f64)>,
    last: &mut Option<(f64, f64)>,
    x: f64,
    value: f64,
    policy: FrontOperationPolicy,
    inner_endpoint: bool,
    work: &mut Work,
) -> Result<()> {
    work.charge(1)?;
    let (previous_x, previous_value) = last.unwrap();
    if x == previous_x && value == previous_value {
        return Ok(());
    }
    // Unlike an outer-knot crossing, a nearby inner endpoint emits no piece
    // in the native helper. Keep the previous value/coordinate as its anchor.
    if policy == FrontOperationPolicy::FrontAlgebra && inner_endpoint && near(x, previous_x)? {
        return Ok(());
    }
    if x <= previous_x {
        return Err(unsupported("composition output coordinates collapsed"));
    }
    let span = checked(x - previous_x, "composition secant span")?;
    let slope = checked(
        checked(value - previous_value, "composition secant difference")? / span,
        "composition secant division",
    )?;
    let merge = if policy == FrontOperationPolicy::FrontAlgebra && output.len() > 1 {
        let old = output.last().unwrap().0;
        near(old, slope)?
    } else {
        false
    };
    if merge {
        let previous = output.last_mut().unwrap();
        // The original helper keeps the prior slope at an outer knot, but
        // replaces it with the new secant at an inner-segment endpoint.
        if inner_endpoint {
            previous.0 = slope;
        }
        previous.1 = x;
    } else {
        work.pairs(output.len() + 1)?;
        output.push((slope, x));
    }
    *last = Some((x, value));
    Ok(())
}

#[cfg(test)]
#[path = "../../../../tests/unit/graph/coupled/descriptor/composition/tests.rs"]
mod tests;
