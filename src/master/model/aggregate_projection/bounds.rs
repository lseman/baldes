/// Canonical lower-child upper bound and upper-child lower bound.
///
/// Preserve the signed relative shifts and operation order. Selection activity
/// is numerical data; these endpoints describe the aggregate disjunction.
pub(super) fn split_bounds(value: f64) -> Option<(f64, f64)> {
    if !value.is_finite() || value.abs() >= 9_007_199_254_740_992. {
        return None;
    }
    let precision = 1e-6;
    let relative = value * 1e-10;
    let conversion_limit = 4_503_599_627_370_496.;

    let mut lower = (value + relative) + precision;
    if lower.abs() < conversion_limit {
        lower = lower.floor();
    }
    if ((value - 1.) + relative) + precision > lower {
        lower += 1.;
    }
    if (-precision..=precision).contains(&lower) {
        lower = 0.;
    }

    let mut upper = (value - relative) - precision;
    if upper.abs() < conversion_limit {
        upper = upper.ceil();
    }
    if upper >= value + 1. {
        upper -= 1.;
    }
    if (-precision..=precision).contains(&upper) {
        upper = 0.;
    }
    Some((lower, upper))
}

/// Apply recovered rounding only when the children cover every integer value.
/// Large signed relative shifts can otherwise leave a gap wider than one.
pub(super) fn covering_split_bounds(value: f64) -> Option<(f64, f64)> {
    let (lower, upper) = split_bounds(value)?;
    if upper - lower > 1. {
        Some((value.floor(), value.ceil()))
    } else {
        Some((lower, upper))
    }
}

/// Independently rounded distances to the canonical lower and upper children.
/// Near an integer both can be zero; the upper distance is not generally the
/// complement of the lower distance.
pub(super) fn fractional_distances(value: f64) -> Option<(f64, f64)> {
    let (lower, upper) = split_bounds(value)?;
    let normalize = |distance: f64| {
        if (-1e-6..=1e-6).contains(&distance) {
            0.
        } else {
            distance
        }
    };
    Some((normalize(value - lower), normalize(upper - value)))
}

#[cfg(test)]
#[path = "../../../../tests/unit/master/model/aggregate_bounds_tests.rs"]
mod tests;
