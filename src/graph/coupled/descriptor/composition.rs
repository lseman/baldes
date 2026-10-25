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

struct Cached<'a> {
    function: &'a PreparedSlopeFunction,
    knots: Cow<'a, [(f64, f64)]>, // coordinate, ordered accumulated value
}
impl<'a> Cached<'a> {
    fn new(function: &'a PreparedSlopeFunction, increasing: bool, work: &mut Work) -> Result<Self> {
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
            || (policy == FrontOperationPolicy::NativeFrontAlgebra && near(argument, lower)?)
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

/// Construct `outer(inner(x))`, intersecting the requested interval with both
/// function domains. The inner map must be strictly increasing (or a point).
/// Output slopes use ordered endpoint secants, rather than slope products.
/// Unrecovered native near-coordinate/slope cases return Unknown explicitly.
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
    let inner = Cached::new(inner, true, &mut work)?;
    compose_cached(outer, &inner, lower, upper, policy, false, &mut work)
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
    let outer = Cached::new(outer, false, work)?;
    if policy == FrontOperationPolicy::NativeFrontAlgebra
        && lower < inner.function.domain().0
        && near(lower, inner.function.domain().0)?
    {
        return Err(unsupported("native snapped composition start"));
    }
    let lo = lower.max(inner.function.domain().0);
    let hi = upper.min(inner.function.domain().1);
    if lo > hi {
        if policy == FrontOperationPolicy::NativeFrontAlgebra && near(lo, hi)? {
            return Err(unsupported("native near-disjoint composition interval"));
        }
        return Ok(None);
    }
    // The native helper treats a tolerance-sized interval as a constant
    // front evaluated at its requested start, rather than fitting a secant.
    // Keep exact domain intersection above: tolerance must not invent a domain.
    if policy == FrontOperationPolicy::NativeFrontAlgebra && near(lo, hi)? {
        work.charge(inner.function.pairs().len() + outer.function.pairs().len())?;
        let inner_value = inner
            .function
            .evaluate(lo, super::EvaluationPolicy::NativeTolerance)
            .map_err(FrontierUnknown::Arithmetic)?
            .ok_or_else(|| unsupported("native singleton inner evaluation outside domain"))?;
        let value = outer
            .function
            .evaluate(inner_value, super::EvaluationPolicy::NativeTolerance)
            .map_err(FrontierUnknown::Arithmetic)?
            .ok_or_else(|| unsupported("native singleton outer evaluation outside domain"))?;
        let pairs = if lo == hi {
            vec![(value, lo)]
        } else {
            vec![(value, lo), (0., hi)]
        };
        work.pairs(pairs.len())?;
        return Ok(Some(prepared(pairs)?));
    }
    let (outer_lo, outer_hi) = outer.function.domain();
    if inner.knots.len() == 1 {
        let (x, y) = inner.knots[0];
        if y < outer_lo || y > outer_hi {
            if policy == FrontOperationPolicy::NativeFrontAlgebra
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
    let mut native_inner_end = None;
    let mut native_outer_anchor = None;
    let mut native_outer_next = None;
    for index in 1..inner.knots.len() {
        work.charge(1)?;
        let (gx, gy) = inner.knots[index - 1];
        let (ge, _) = inner.knots[index];
        let slope = inner.function.pairs()[index].0;
        let mut x0 = gx.max(lo);
        let mut x1 = ge.min(hi);
        if x0 > x1 {
            continue;
        }
        // Native front algebra advances from the clipped interval start,
        // carrying the rounded endpoint into the following segment. Reusing
        // the original knot anchor is mathematically equivalent, but can
        // change both a crossing and the final secant by a binary64 bit.
        let (anchor_x, anchor_y) = if policy == FrontOperationPolicy::NativeFrontAlgebra {
            let y = match native_inner_end {
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
        native_inner_end = Some(y1);
        if y0 > outer_hi || y1 < outer_lo {
            if policy == FrontOperationPolicy::NativeFrontAlgebra
                && (near(y0, outer_hi)? || near(y1, outer_lo)?)
            {
                return Err(unsupported("native near-disjoint composition image"));
            }
            continue;
        }
        if policy == FrontOperationPolicy::NativeFrontAlgebra
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
        let native = policy == FrontOperationPolicy::NativeFrontAlgebra;
        let f0 = if native {
            match last {
                Some((x, value)) if x == x0 => value,
                _ => outer.value(y0, policy, work)?,
            }
        } else {
            outer.value(y0, policy, work)?
        };
        if native && native_outer_anchor.is_none() {
            native_outer_anchor = Some((y0, f0));
        }
        match last {
            None => {
                work.pairs(1)?;
                output.push((f0, x0));
                last = Some((x0, f0));
            }
            Some((x, y)) if x == x0 && y == f0 => (),
            _ => {
                return Err(unsupported(
                    "composition components do not meet continuously",
                ))
            }
        }
        let mut next = if native {
            match native_outer_next {
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
            let value = if let Some((y, value)) = native_outer_anchor {
                let value = segment_value(y, value, outer.function.pairs()[next].0, target)?;
                native_outer_anchor = Some((target, value));
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
        } else if let Some((y, value)) = native_outer_anchor {
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
            native_outer_anchor = Some((y1, end));
        }
        if native {
            native_outer_next = Some(next);
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
    if policy == FrontOperationPolicy::NativeFrontAlgebra && inner_endpoint && near(x, previous_x)?
    {
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
    let merge = if policy == FrontOperationPolicy::NativeFrontAlgebra && output.len() > 1 {
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
