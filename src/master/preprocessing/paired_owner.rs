//! Local callback bounds and aggregate owner-variable propagation are distinct.
use super::Bounds;

#[derive(Clone, Copy)]
pub(crate) struct OwnerBounds {
    pub local: Bounds,
    pub aggregate: Bounds,
    registered: bool,
}
fn product(a: f64, b: f64) -> f64 {
    if a == 0. || b == 0. {
        0.
    } else {
        a * b
    }
}
fn tighter_lower(value: f64, previous: f64) -> bool {
    value.is_finite()
        && (previous == f64::NEG_INFINITY
            || value - (1e-10 * value.abs().max(previous.abs()) + 1e-6) > previous)
}
fn tighter_upper(value: f64, previous: f64) -> bool {
    value.is_finite()
        && (previous == f64::INFINITY
            || value + (1e-10 * value.abs().max(previous.abs()) + 1e-6) < previous)
}
impl OwnerBounds {
    pub fn new(input: Option<[f64; 2]>, multiplicity: Bounds, fixed: [f64; 2]) -> Self {
        let local = input
            .map(|[lower, upper]| Bounds { lower, upper })
            .unwrap_or(Bounds {
                lower: f64::NEG_INFINITY,
                upper: f64::INFINITY,
            });
        // Explicit registration initializes a nonnegative aggregate variable;
        // the caller's possibly signed input lower belongs to the local copy.
        // Absent registration input leaves this relaxation unbounded.
        let mut aggregate = Bounds {
            lower: if input.is_some() {
                -fixed[1]
            } else {
                f64::NEG_INFINITY
            },
            upper: local.upper - fixed[0],
        };
        let lower = product(multiplicity.lower, local.lower);
        let upper = product(multiplicity.upper, local.upper);
        if input.is_some() && lower.is_finite() {
            aggregate.lower = aggregate.lower.max(lower);
        }
        if input.is_some() && upper.is_finite() {
            aggregate.upper = aggregate.upper.min(upper);
        }
        Self {
            local,
            aggregate,
            registered: input.is_some(),
        }
    }
    /// Apply aggregate-to-local residual conversion, then local-to-aggregate
    /// multiplicity products. Local bounds never become paired path windows.
    pub fn update(&mut self, aggregate: Bounds, multiplicity: Bounds) -> bool {
        self.aggregate = aggregate;
        if multiplicity.upper > 1e-6 && multiplicity.upper.is_finite() {
            let lower = aggregate.lower - product(multiplicity.upper - 1., self.local.upper);
            if tighter_lower(lower, self.local.lower) {
                self.local.lower = lower;
            }
        }
        if multiplicity.upper >= 1. - (1e-6 + 1e-10 * multiplicity.upper.abs()) {
            let minimum = if multiplicity.lower.abs() <= 1e-6 * (1. + multiplicity.lower.abs()) {
                1.
            } else {
                multiplicity.lower
            };
            let upper = aggregate.upper - product(minimum - 1., self.local.lower);
            if tighter_upper(upper, self.local.upper) {
                self.local.upper = upper;
            }
        }
        if !self.registered {
            return false;
        }
        let lower = product(multiplicity.lower, self.local.lower);
        let upper = product(multiplicity.upper, self.local.upper);
        let mut changed = false;
        if tighter_lower(lower, self.aggregate.lower) {
            self.aggregate.lower = lower;
            changed = true;
        }
        if tighter_upper(upper, self.aggregate.upper) {
            self.aggregate.upper = upper;
            changed = true;
        }
        changed
    }
}

#[cfg(test)]
#[path = "../../../tests/unit/master/preprocessing/paired_owner_tests.rs"]
mod tests;
