/// Integer endpoints retained by a component sequence.
/// Empty and nonempty sequences deliberately round in different directions.
pub(in crate::master::model) fn split_bounds(value: f64, nonempty: bool) -> Option<(f64, f64)> {
    if !value.is_finite() || value.abs() >= 9_007_199_254_740_992. {
        return None;
    }
    let mut upper = if nonempty {
        let shifted = value + 1.;
        let relative = shifted * 1e-10;
        let mut bound = (shifted + relative) + 1e-6;
        if bound.abs() < 4_503_599_627_370_496. {
            bound = bound.floor();
        }
        if ((shifted - 1.) + relative) + 1e-6 > bound {
            bound += 1.;
        }
        bound
    } else {
        let relative = value * 1e-10;
        let mut bound = (value - relative) - 1e-6;
        if bound.abs() < 4_503_599_627_370_496. {
            bound = bound.ceil();
        }
        if bound >= value + 1. {
            bound -= 1.;
        }
        bound
    };
    if (-1e-6..=1e-6).contains(&upper) {
        upper = 0.;
    }
    Some((upper - 1., upper))
}

/// Keep the child disjunction exhaustive even when endpoints exceed exact
/// binary64 integer spacing after the relative shift.
pub(in crate::master::model) fn covering_split_bounds(
    value: f64,
    nonempty: bool,
) -> Option<(f64, f64)> {
    let (lower, upper) = split_bounds(value, nonempty)?;
    if upper - lower > 1. {
        Some((value.floor(), value.ceil()))
    } else {
        Some((lower, upper))
    }
}

#[cfg(test)]
#[path = "../../../../tests/unit/master/model/component_history_rounding_tests.rs"]
mod tests;
