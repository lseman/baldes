//! Outward images of literal encoded numeric functions, including proved
//! native endpoint snapping. A partly accepted interval remains unknown.
use super::*;
use crate::graph::coupled::{EncodedFunction, EvaluationPolicy};

fn include(a: bounds::Range, b: bounds::Range) -> bounds::Range {
    bounds::Range {
        lo: a.lo.min(b.lo),
        hi: a.hi.max(b.hi),
    }
}

fn all_near(argument: bounds::Range, boundary: f64) -> bool {
    let difference = argument.subtract(bounds::Range::point(boundary));
    let sum = argument.add(bounds::Range::point(boundary));
    if !difference.finite() || !sum.finite() {
        return false;
    }
    let maximum = difference.lo.abs().max(difference.hi.abs());
    if maximum < 0.0001 {
        return true;
    }
    let minimum = if sum.lo <= 0. && sum.hi >= 0. {
        0.
    } else {
        sum.lo.abs().min(sum.hi.abs())
    };
    let relative =
        bounds::Range::point(minimum).multiply(bounds::Range::point(2.220446049250313e-10));
    relative.finite() && maximum < relative.lo
}

fn may_near(argument: bounds::Range, boundary: f64) -> bool {
    let difference = argument.subtract(bounds::Range::point(boundary));
    let sum = argument.add(bounds::Range::point(boundary));
    if !difference.finite() || !sum.finite() {
        return true;
    }
    let minimum = if difference.lo <= 0. && difference.hi >= 0. {
        0.
    } else {
        difference.lo.abs().min(difference.hi.abs())
    };
    if minimum < 0.0001 {
        return true;
    }
    let maximum = sum.lo.abs().max(sum.hi.abs());
    let relative =
        bounds::Range::point(maximum).multiply(bounds::Range::point(2.220446049250313e-10));
    !relative.finite() || minimum < relative.hi
}

pub(super) fn image(
    function: &EncodedFunction,
    argument: bounds::Range,
    work: &mut usize,
    deadline: Option<Instant>,
) -> Option<bounds::Range> {
    match function {
        EncodedFunction::Knots { function } => {
            debit(work, function.points.len(), deadline)?;
            let value = function.range(argument)?;
            value.finite().then_some(value)
        }
        EncodedFunction::SlopeSegments {
            function,
            policy: EvaluationPolicy::StrictDomain,
        } => slope_function_range(function, argument, work, deadline),
        EncodedFunction::SlopeSegments {
            function,
            policy: EvaluationPolicy::EndpointTolerance,
        } => {
            debit(work, 1, deadline)?;
            let (lower, upper) = function.domain();
            let initial = bounds::Range::point(function.pairs()[0].0);
            if all_near(argument, lower) {
                return Some(initial);
            }
            if argument.lo < lower
                && !all_near(
                    bounds::Range {
                        lo: argument.lo,
                        hi: argument.hi.min(lower),
                    },
                    lower,
                )
            {
                return None;
            }
            if argument.hi > upper
                && !all_near(
                    bounds::Range {
                        lo: argument.lo.max(upper),
                        hi: argument.hi,
                    },
                    upper,
                )
            {
                return None;
            }
            let mut result = None;
            let accepted = bounds::Range {
                lo: argument.lo.max(lower),
                hi: argument.hi.min(upper),
            };
            if accepted.lo <= accepted.hi {
                result = Some(slope_function_range(function, accepted, work, deadline)?);
            }
            // The first-coordinate snap can take precedence inside the
            // literal domain. Including its value is conservative even when
            // only part of the interval satisfies the snap predicate.
            if argument.lo <= upper && may_near(argument, lower) {
                result = Some(result.map_or(initial, |r| include(r, initial)));
            }
            if argument.hi > upper {
                let last =
                    slope_function_range(function, bounds::Range::point(upper), work, deadline)?;
                result = Some(result.map_or(last, |r| include(r, last)));
            }
            result
        }
    }
}
