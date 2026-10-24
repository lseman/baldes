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
mod tests {
    use super::*;
    use crate::graph::coupled::descriptor::EvaluationPolicy;
    fn f(pairs: &[(f64, f64)]) -> PreparedSlopeFunction {
        PreparedSlopeFunction::from_prepared_pairs(pairs.to_vec()).unwrap()
    }
    fn complete(result: FrontierOutcome<PreparedSlopeFunction>) -> PreparedSlopeFunction {
        match result {
            FrontierOutcome::Complete(f) => f,
            other => panic!("unexpected {other:?}"),
        }
    }
    #[test]
    fn native_affine_and_outer_knot_secants() {
        let budget = FrontierBudget::default();
        let affine = complete(compose_increasing(
            &f(&[(1., 0.), (2., 10.)]),
            &f(&[(2., 0.), (3., 2.)]),
            0.,
            2.,
            FrontOperationPolicy::NativeFrontAlgebra,
            budget,
        ));
        assert_eq!(affine.pairs(), &[(5., 0.), (6., 2.)]);
        let crossed = complete(compose_increasing(
            &f(&[(1., 0.), (2., 2.), (-1., 6.)]),
            &f(&[(0., 0.), (2., 3.)]),
            0.,
            3.,
            FrontOperationPolicy::NativeFrontAlgebra,
            budget,
        ));
        assert_eq!(crossed.pairs(), &[(1., 0.), (4., 1.), (-2., 3.)]);
    }
    #[test]
    fn inverse_retains_reciprocal_and_rejects_collapsed_images() {
        let inverse = complete(inverse_increasing(
            &f(&[(2., 1.), (3., 4.)]),
            FrontierBudget::default(),
        ));
        assert_eq!(inverse.pairs(), &[(1., 2.), (1. / 3., 11.)]);
        assert!(matches!(
            inverse_increasing(&f(&[(2., 0.), (0., 1.)]), FrontierBudget::default()),
            FrontierOutcome::Unknown(_)
        ));
        assert!(matches!(
            inverse_increasing(&f(&[(1e30, 0.), (1., 1.)]), FrontierBudget::default()),
            FrontierOutcome::Unknown(_)
        ));
    }
    #[test]
    fn native_fractional_secants_match_retained_helper_observation() {
        // Pinned binary helper fixture, not idealized slope multiplication.
        let map = match PreparedCompositionMap::new(
            &f(&[(0.05, 0.), (0.7, 2.)]),
            FrontierBudget::default(),
        ) {
            FrontierOutcome::Complete(map) => map,
            other => panic!("unexpected prepared map {other:?}"),
        };
        let actual = complete(map.compose(
            &f(&[(0.1, 0.), (0.3, 0.7), (-0.2, 2.)]),
            0.1,
            1.7,
            FrontOperationPolicy::NativeFrontAlgebra,
            FrontierBudget::default(),
        ));
        assert_eq!(
            actual.pairs(),
            &[
                (0.136, 0.1),
                (0.21, 0.9285714285714285),
                (-0.13999999999999999, 1.7)
            ]
        );
    }
    #[test]
    fn clipped_native_and_strict_composition_preserve_their_arithmetic() {
        let outer = f(&[(0.1, 0.), (0.3, 0.7), (-0.2, 2.)]);
        let inner = f(&[(0.05, 0.), (0.7, 2.)]);
        let strict = complete(compose_increasing(
            &outer,
            &inner,
            0.1,
            1.7,
            FrontOperationPolicy::StrictDomain,
            FrontierBudget::default(),
        ));
        let native = complete(compose_increasing(
            &outer,
            &inner,
            0.1,
            1.7,
            FrontOperationPolicy::NativeFrontAlgebra,
            FrontierBudget::default(),
        ));
        assert_eq!(strict.pairs()[2].0.to_bits(), (-0.14_f64).to_bits());
        assert_eq!(native.pairs()[2].0.to_bits(), 0xbfc1_eb85_1eb8_51eb);
    }
    #[test]
    fn native_composition_carries_an_outer_knot_at_an_inner_segment_end() {
        let actual = complete(compose_increasing(
            &f(&[(0.16, 0.), (0.75, 0.7), (-0.79, 2.)]),
            &f(&[(0.034, 0.), (0.74, 0.9), (0.629, 2.)]),
            0.167,
            1.593,
            FrontOperationPolicy::NativeFrontAlgebra,
            FrontierBudget::default(),
        ));
        let bits: Vec<_> = actual
            .pairs()
            .iter()
            .map(|(a, b)| (a.to_bits(), b.to_bits()))
            .collect();
        assert_eq!(
            bits,
            vec![
                (0x3fd1_cdc8_754f_3776, 0x3fc5_6041_8937_4bc7),
                (0x3fe1_c28f_5c28_f5c2, 0x3fec_cccc_cccc_cccd),
                (0xbfdf_cd5f_99c3_8b05, 0x3ff9_7ced_9168_72b0),
            ]
        );
    }
    #[test]
    fn clipping_point_and_budget_are_not_guessed_infeasible() {
        let outer = f(&[(7., 2.)]);
        let inner = f(&[(0., 0.), (2., 2.)]);
        let point = complete(compose_increasing(
            &outer,
            &inner,
            0.,
            2.,
            FrontOperationPolicy::StrictDomain,
            FrontierBudget::default(),
        ));
        assert_eq!(point.pairs(), &[(7., 1.)]);
        assert!(matches!(
            compose_increasing(
                &outer,
                &inner,
                0.,
                2.,
                FrontOperationPolicy::StrictDomain,
                FrontierBudget {
                    max_pairs: 0,
                    ..FrontierBudget::default()
                }
            ),
            FrontierOutcome::Unknown(FrontierUnknown::PairLimit)
        ));
        assert!(matches!(
            compose_increasing(
                &f(&[(0., 10.), (1., 11.)]),
                &inner,
                0.,
                2.,
                FrontOperationPolicy::StrictDomain,
                FrontierBudget::default()
            ),
            FrontierOutcome::Empty
        ));
    }
    #[test]
    fn strict_composition_preserves_negative_outer_slopes_and_domains() {
        let composed = complete(compose_increasing(
            &f(&[(4., 1.), (-2., 3.)]),
            &f(&[(0., 0.), (2., 2.)]),
            0.,
            2.,
            FrontOperationPolicy::StrictDomain,
            FrontierBudget::default(),
        ));
        assert_eq!(composed.pairs(), &[(4., 0.5), (-4., 1.5)]);
        assert_eq!(
            composed
                .evaluate(1., EvaluationPolicy::StrictDomain)
                .unwrap(),
            Some(2.)
        );
    }

    #[test]
    fn native_exhausted_outer_domain_is_unknown_while_strict_composition_clips() {
        let outer = f(&[(0.25, 0.), (-0.5, 0.7), (0.75, 2.)]);
        let inner = f(&[(0.7, 0.), (0.75, 0.9), (0.75, 2.)]);
        for lower in [0.9, 1., 1.5] {
            for upper in [1.8, 2.] {
                assert!(matches!(
                    compose_increasing(
                        &outer,
                        &inner,
                        lower,
                        upper,
                        FrontOperationPolicy::NativeFrontAlgebra,
                        FrontierBudget::default()
                    ),
                    FrontierOutcome::Unknown(FrontierUnknown::Unsupported(
                        "native composition image outside outer domain"
                    ))
                ));
                let clipped = complete(compose_increasing(
                    &outer,
                    &inner,
                    lower,
                    upper,
                    FrontOperationPolicy::StrictDomain,
                    FrontierBudget::default(),
                ));
                assert_eq!(clipped.domain().0, lower);
                assert!(clipped.domain().1 < upper);
            }
            assert!(matches!(
                compose_increasing(
                    &outer,
                    &inner,
                    lower,
                    1.6,
                    FrontOperationPolicy::NativeFrontAlgebra,
                    FrontierBudget::default()
                ),
                FrontierOutcome::Complete(_)
            ));
        }
    }
    #[test]
    fn native_snaps_and_near_slope_merges_match_original_helper_bits() {
        // Original ELF SHA256 103caf4a...15889d6, helper 0x649680.
        // Fixed-array(12)/vector ABI checked against the complete assembly;
        // fixtures were executed in a separate original-library probe process.
        type Fixture<'a> = (
            &'a [(f64, f64)],
            &'a [(f64, f64)],
            f64,
            f64,
            &'a [(u64, u64)],
        );
        let cases: &[Fixture<'_>] = &[
            // snap-start
            (
                &[(1.0, 0.0), (2.0, 10.0)],
                &[(2.0, 0.0), (3.0, 2.0)],
                1e-05,
                2.0,
                &[
                    (0x4014000000000000, 0x3ee4f8b588e368f1),
                    (0x4017ffffffffffff, 0x4000000000000000),
                ],
            ),
            // tiny-interval
            (
                &[(1.0, 0.0), (2.0, 10.0)],
                &[(2.0, 0.0), (3.0, 2.0)],
                0.5,
                0.50001,
                &[
                    (0x4020000000000000, 0x3fe0000000000000),
                    (0x0000000000000000, 0x3fe00014f8b588e3),
                ],
            ),
            // near-endpoint-slopes
            (
                &[(0.0, 0.0), (1.0, 1.0), (1.00001, 2.0)],
                &[(0.0, 0.0), (1.0, 1.0), (1.0, 2.0)],
                0.0,
                2.0,
                &[
                    (0x0000000000000000, 0x0000000000000000),
                    (0x3ff0000000000000, 0x4000000000000000),
                ],
            ),
            // near-knot-slopes
            (
                &[(0.0, 0.0), (1.0, 1.0), (1.00001, 2.0)],
                &[(0.0, 0.0), (1.0, 2.0)],
                0.0,
                2.0,
                &[
                    (0x0000000000000000, 0x0000000000000000),
                    (0x3ff0000000000000, 0x4000000000000000),
                ],
            ),
            // near-inner-end-slopes
            (
                &[(0.0, 0.0), (1.0, 1.0), (1.00001, 4.0)],
                &[(0.0, 0.0), (1.0, 2.0)],
                0.0,
                2.0,
                &[
                    (0x0000000000000000, 0x0000000000000000),
                    (0x3ff0000a7c5ac472, 0x4000000000000000),
                ],
            ),
        ];
        for &(outer, inner, lo, hi, expected) in cases {
            let outer = f(outer);
            let inner = f(inner);
            let actual = complete(compose_increasing(
                &outer,
                &inner,
                lo,
                hi,
                FrontOperationPolicy::NativeFrontAlgebra,
                FrontierBudget::default(),
            ));
            let bits: Vec<_> = actual
                .pairs()
                .iter()
                .map(|&(a, b)| (a.to_bits(), b.to_bits()))
                .collect();
            assert_eq!(bits, expected, "domain={lo:?}..{hi:?}");
            let prepared = match PreparedCompositionMap::new(&inner, FrontierBudget::default()) {
                FrontierOutcome::Complete(map) => map,
                other => panic!("{other:?}"),
            };
            let cached = complete(prepared.compose(
                &outer,
                lo,
                hi,
                FrontOperationPolicy::NativeFrontAlgebra,
                FrontierBudget::default(),
            ));
            assert_eq!(cached.pairs(), actual.pairs());
        }
    }

    #[test]
    fn native_constant_short_front_does_not_change_strict_composition() {
        let outer = f(&[(1., 0.), (2., 10.)]);
        let inner = f(&[(2., 0.), (3., 2.)]);
        let strict = complete(compose_increasing(
            &outer,
            &inner,
            0.5,
            0.50001,
            FrontOperationPolicy::StrictDomain,
            FrontierBudget::default(),
        ));
        assert!((strict.pairs()[1].0 - 6.).abs() < 1e-8);
        let native = complete(compose_increasing(
            &outer,
            &inner,
            0.5,
            0.50001,
            FrontOperationPolicy::NativeFrontAlgebra,
            FrontierBudget::default(),
        ));
        assert_eq!(native.pairs()[1].0, 0.);
        assert!(matches!(
            compose_increasing(
                &outer,
                &inner,
                0.5,
                0.50001,
                FrontOperationPolicy::NativeFrontAlgebra,
                FrontierBudget {
                    max_pairs: 0,
                    ..FrontierBudget::default()
                }
            ),
            FrontierOutcome::Unknown(FrontierUnknown::PairLimit)
        ));
        assert!(matches!(
            compose_increasing(
                &outer,
                &inner,
                -0.00001,
                1.,
                FrontOperationPolicy::NativeFrontAlgebra,
                FrontierBudget::default()
            ),
            FrontierOutcome::Unknown(_)
        ));
    }
    #[test]
    fn native_near_knots_preserve_segment_advancement_and_endpoint_snaps() {
        // Original helper fixture: near-knot-end.
        let actual = complete(compose_increasing(
            &f(&[(-0.88, 0.0), (0.15, 0.7), (-0.57, 2.0)]),
            &f(&[(0.037, 0.0), (0.387, 0.9), (0.456, 2.0)]),
            0.251,
            1.59,
            FrontOperationPolicy::NativeFrontAlgebra,
            FrontierBudget::default(),
        ));
        assert_eq!(
            actual
                .pairs()
                .iter()
                .map(|&(a, b)| (a.to_bits(), b.to_bits()))
                .collect::<Vec<_>>(),
            vec![
                (0xbfeb8421e8881436, 0x3fd010624dd2f1aa),
                (0x3fadb8bac710cb2f, 0x3feccccccccccccd),
                (0x3fb18384685be48d, 0x3ff970a3d70a3d71)
            ]
        );
        // Original helper fixture: near-knot-start.
        let actual = complete(compose_increasing(
            &f(&[(0.0, 0.0), (1.0, 1.0), (2.0, 4.0)]),
            &f(&[(0.99999, 0.0), (1.0, 2.0)]),
            0.0,
            1.0,
            FrontOperationPolicy::NativeFrontAlgebra,
            FrontierBudget::default(),
        ));
        assert_eq!(
            actual
                .pairs()
                .iter()
                .map(|&(a, b)| (a.to_bits(), b.to_bits()))
                .collect::<Vec<_>>(),
            vec![
                (0x3fefffeb074a771d, 0x0000000000000000),
                (0x4000000000000000, 0x3ff0000000000000)
            ]
        );
        // Original helper fixture: near-knot-end-carried.
        let actual = complete(compose_increasing(
            &f(&[(0.0, 0.0), (1.0, 1.0), (2.0, 4.0)]),
            &f(&[(0.0, 0.0), (0.99999, 1.0), (1.0, 2.0)]),
            0.0,
            2.0,
            FrontOperationPolicy::NativeFrontAlgebra,
            FrontierBudget::default(),
        ));
        assert_eq!(
            actual
                .pairs()
                .iter()
                .map(|&(a, b)| (a.to_bits(), b.to_bits()))
                .collect::<Vec<_>>(),
            vec![
                (0x0000000000000000, 0x0000000000000000),
                (0x3ff0000000000000, 0x3ff0000000000000),
                (0x3fffffeb074a771c, 0x4000000000000000)
            ]
        );
    }
}
