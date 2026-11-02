/// Whether a finite saved master quantity enters fractional-solution projection.
///
/// The signed relative shifts are intentional: large negative quantities can
/// pass even when their floating-point representation is an exact integer.
pub(super) fn fractional_quantity(q: f64) -> bool {
    if !q.is_finite() {
        return false;
    }
    let precision = 1e-6;
    let relative = q * 1e-10;
    let conversion_limit = 4_503_599_627_370_496.;

    let mut lower = (q + relative) + precision;
    if lower.abs() < conversion_limit {
        lower = lower.floor();
    }
    if ((q - 1.) + relative) + precision > lower {
        lower += 1.;
    }
    let mut down = if lower > precision || lower < -precision {
        q - lower
    } else {
        q
    };
    if down >= -precision && down <= precision {
        down = 0.;
    }

    let mut upper = (q - relative) - precision;
    if upper.abs() < conversion_limit {
        upper = upper.ceil();
    }
    if upper >= q + 1. {
        upper -= 1.;
    }
    if upper >= -precision && upper <= precision {
        upper = 0.;
    }
    let distance = down.min(upper - q);
    let cutoff = precision * 1e-4;
    distance - (distance.abs().max(cutoff) * 1e-10 + precision) > cutoff
}

#[cfg(test)]
#[path = "../../../../tests/unit/master/model/aggregate_quantity_tests.rs"]
mod tests;
