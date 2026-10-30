//! Directed enclosures for paired terminal charge functions.
use super::*;

/// Directed enclosure of every value on a prepared raw-slope function.
/// Each segment is affine, so its extrema occur at its endpoints; interval
/// arithmetic also encloses the function's sequential binary64 evaluation.
pub(in super::super) fn slope_function_hull(
    function: &crate::graph::coupled::PreparedSlopeFunction,
    work: &mut usize,
    deadline: Instant,
) -> Option<Interval> {
    let pairs = function.pairs();
    let &(initial, _) = pairs.first()?;
    let mut value = Interval::point(initial);
    let mut lower = initial;
    let mut upper = initial;
    let mut previous = pairs.first()?.1;
    for &(slope, end) in pairs.iter().skip(1) {
        *work = work.checked_sub(1)?;
        if Instant::now() >= deadline {
            return None;
        }
        let span = end - previous;
        if !span.is_finite() || span <= 0. {
            return None;
        }
        let delta = Interval::point(slope).multiply(Interval::point(span))?;
        value = value.add(delta)?;
        lower = lower.min(value.lo);
        upper = upper.max(value.hi);
        previous = end;
    }
    Some(Interval {
        lo: lower,
        hi: upper,
    })
}

pub(in super::super) fn slope_charge_hull(
    functions: &[crate::graph::coupled::PreparedSlopeFunction],
    work: &mut usize,
    deadline: Instant,
) -> Option<Interval> {
    let mut hull = Interval {
        lo: f64::INFINITY,
        hi: f64::NEG_INFINITY,
    };
    for function in functions {
        let range = slope_function_hull(function, work, deadline)?;
        hull.lo = hull.lo.min(range.lo);
        hull.hi = hull.hi.max(range.hi);
    }
    if functions.is_empty() {
        Some(Interval::point(0.))
    } else {
        Some(hull)
    }
}

/// Directed hull for encoded state costs. Knot functions clamp interpolation
/// to their segment endpoints; slope functions retain their prepared
/// sequential arithmetic, which `slope_function_hull` encloses outward.
pub(in super::super) fn encoded_charge_hull(
    functions: &[crate::graph::coupled::EncodedFunction],
    work: &mut usize,
    deadline: Instant,
) -> Option<Interval> {
    let mut hull = Interval {
        lo: f64::INFINITY,
        hi: f64::NEG_INFINITY,
    };
    for function in functions {
        let range = match function {
            crate::graph::coupled::EncodedFunction::Knots { function } => {
                piecewise_charge_hull(std::slice::from_ref(function), work, deadline)?
            }
            crate::graph::coupled::EncodedFunction::SlopeSegments { function, .. } => {
                slope_function_hull(function, work, deadline)?
            }
        };
        hull.lo = hull.lo.min(range.lo);
        hull.hi = hull.hi.max(range.hi);
    }
    if functions.is_empty() {
        Some(Interval::point(0.))
    } else {
        Some(hull)
    }
}

/// Piecewise-linear source evaluation clamps interpolation to its segment's
/// endpoint values, so the global knot hull encloses every rounded charge.
pub(in super::super) fn piecewise_charge_hull(
    functions: &[crate::graph::coupled::PiecewiseLinearFunction],
    work: &mut usize,
    deadline: Instant,
) -> Option<Interval> {
    let mut hull = Interval {
        lo: f64::INFINITY,
        hi: f64::NEG_INFINITY,
    };
    for function in functions {
        let values = piecewise_value_hull(function, work, deadline)?;
        hull.lo = hull.lo.min(values.lo);
        hull.hi = hull.hi.max(values.hi);
    }
    if functions.is_empty() {
        Some(Interval::point(0.))
    } else {
        Some(hull)
    }
}

pub(in super::super) fn piecewise_value_hull(
    function: &crate::graph::coupled::PiecewiseLinearFunction,
    work: &mut usize,
    deadline: Instant,
) -> Option<Interval> {
    let mut hull = Interval {
        lo: f64::INFINITY,
        hi: f64::NEG_INFINITY,
    };
    for point in &function.points {
        *work = work.checked_sub(1)?;
        if Instant::now() >= deadline || !point[1].is_finite() {
            return None;
        }
        hull.lo = hull.lo.min(point[1]);
        hull.hi = hull.hi.max(point[1]);
    }
    (hull.lo.is_finite() && hull.hi.is_finite()).then_some(hull)
}

/// Sum a global per-arc charge range over all path lengths up to `hops`.
/// Zero is included at each position to relax routes that end earlier.
pub(in super::super) fn repeated_charge_hull(
    per_arc: Interval,
    hops: usize,
    work: &mut usize,
    deadline: Instant,
) -> Option<Interval> {
    let term = Interval {
        lo: per_arc.lo.min(0.),
        hi: per_arc.hi.max(0.),
    };
    let mut total = Interval::point(0.);
    for _ in 0..hops {
        *work = work.checked_sub(1)?;
        if Instant::now() >= deadline {
            return None;
        }
        let previous = total;
        total = total.add(term)?;
        // The term is invariant, so a bit-identical interval is a fixed
        // point of every remaining directed addition. This commonly skips
        // long cyclic walk caps when the per-arc charge is zero.
        if total.lo.to_bits() == previous.lo.to_bits()
            && total.hi.to_bits() == previous.hi.to_bits()
        {
            break;
        }
    }
    Some(total)
}

/// Outward lower enclosure of a cumulative/custom cost over every walk of at
/// most `hops` arcs. Increment and weight choices are relaxed independently,
/// while interval addition encloses the source program's per-arc rounding.
pub(in super::super) fn accumulated_cost_hull(
    initial: f64,
    increments: &[f64],
    weights: &[f64],
    hops: usize,
    work: &mut usize,
    deadline: Instant,
) -> Option<Interval> {
    if increments.len() != weights.len() {
        return None;
    }
    if hops == 0 {
        return Some(Interval::point(0.));
    }
    let mut increment_hull = Interval {
        lo: f64::INFINITY,
        hi: f64::NEG_INFINITY,
    };
    let mut weight_hull = increment_hull;
    for (&increment, &weight) in increments.iter().zip(weights) {
        *work = work.checked_sub(1)?;
        if Instant::now() >= deadline {
            return None;
        }
        if !increment.is_finite() || !weight.is_finite() {
            return None;
        }
        increment_hull.lo = increment_hull.lo.min(increment);
        increment_hull.hi = increment_hull.hi.max(increment);
        weight_hull.lo = weight_hull.lo.min(weight);
        weight_hull.hi = weight_hull.hi.max(weight);
    }
    if increments.is_empty() {
        return Some(Interval::point(0.));
    }
    let mut state = Interval::point(initial);
    let mut cost = Interval::point(0.);
    for _ in 0..hops {
        *work = work.checked_sub(1)?;
        if Instant::now() >= deadline {
            return None;
        }
        let previous_state = state;
        let previous_cost = cost;
        state = state.add(increment_hull)?;
        let term = state.multiply(weight_hull)?;
        let term = Interval {
            lo: term.lo.min(0.),
            hi: term.hi.max(0.),
        };
        cost = cost.add(term)?;
        // Both the state and accumulated-cost intervals participate in
        // the next recurrence. Stop only if both are bit-identical, so no
        // future directed result or overflow can differ from this step.
        if state.lo.to_bits() == previous_state.lo.to_bits()
            && state.hi.to_bits() == previous_state.hi.to_bits()
            && cost.lo.to_bits() == previous_cost.lo.to_bits()
            && cost.hi.to_bits() == previous_cost.hi.to_bits()
        {
            break;
        }
    }
    Some(cost)
}
