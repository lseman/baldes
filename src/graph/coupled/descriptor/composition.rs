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
        let mut work = Work::new(budget);
        if let Err(reason) = work.pairs(self.knots.len()) {
            return FrontierOutcome::Unknown(reason);
        }
        let inner = Cached {
            function: &self.function,
            knots: Cow::Borrowed(&self.knots),
        };
        finish(compose_cached(
            outer, &inner, lower, upper, policy, &mut work,
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
    compose_cached(outer, &inner, lower, upper, policy, &mut work)
}
fn compose_cached(
    outer: &PreparedSlopeFunction,
    inner: &Cached<'_>,
    lower: f64,
    upper: f64,
    policy: FrontOperationPolicy,
    work: &mut Work,
) -> Result<Option<PreparedSlopeFunction>> {
    if !lower.is_finite() || !upper.is_finite() || lower > upper {
        return Err(unsupported("composition needs a finite ordered interval"));
    }
    let outer = Cached::new(outer, false, work)?;
    if policy == FrontOperationPolicy::NativeFrontAlgebra
        && lower != inner.function.domain().0
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
        let mut y0 = segment_value(gx, gy, slope, x0)?;
        let mut y1 = segment_value(gx, gy, slope, x1)?;
        if y0 > outer_hi || y1 < outer_lo {
            if policy == FrontOperationPolicy::NativeFrontAlgebra
                && (near(y0, outer_hi)? || near(y1, outer_lo)?)
            {
                return Err(unsupported("native near-disjoint composition image"));
            }
            continue;
        }
        if y0 < outer_lo {
            x0 = crossing(gx, gy, slope, outer_lo)?;
            y0 = outer_lo;
        }
        if y1 > outer_hi {
            x1 = crossing(gx, gy, slope, outer_hi)?;
            y1 = outer_hi;
        }
        if x0 < lo || x1 > hi || x0 > x1 || (x0 == x1 && y0 != y1) {
            return Err(unsupported("rounded composition domain crossing"));
        }
        let f0 = outer.value(y0, policy, work)?;
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
        let mut next = outer.knots.partition_point(|&(x, _)| x <= y0);
        while next < outer.knots.len() && outer.knots[next].0 < y1 {
            let (target, value) = outer.knots[next];
            let x = crossing(gx, gy, slope, target)?;
            if x < x0 || x > x1 {
                return Err(unsupported("rounded composition knot crossing"));
            }
            emit(&mut output, &mut last, x, value, policy, work)?;
            next += 1;
        }
        let end = outer.value(y1, policy, work)?;
        emit(&mut output, &mut last, x1, end, policy, work)?;
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
    work: &mut Work,
) -> Result<()> {
    work.charge(1)?;
    let (previous_x, previous_value) = last.unwrap();
    if x == previous_x && value == previous_value {
        return Ok(());
    }
    if x <= previous_x {
        return Err(unsupported("composition output coordinates collapsed"));
    }
    let span = checked(x - previous_x, "composition secant span")?;
    if policy == FrontOperationPolicy::NativeFrontAlgebra && near(x, previous_x)? {
        return Err(unsupported("native near-singleton composition"));
    }
    let slope = checked(
        checked(value - previous_value, "composition secant difference")? / span,
        "composition secant division",
    )?;
    let merge = if policy == FrontOperationPolicy::NativeFrontAlgebra && output.len() > 1 {
        let old = output.last().unwrap().0;
        if old != slope && near(old, slope)? {
            return Err(unsupported("native near-slope composition merge"));
        }
        old == slope
    } else {
        false
    };
    if merge {
        output.last_mut().unwrap().1 = x;
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
}
