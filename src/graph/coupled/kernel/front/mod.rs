//! Cost functions over a free entry interval, retained through physical labels.
use super::*;

mod dominance;
mod storage;
use storage::FrontPoints;

#[derive(Clone, Debug, PartialEq)]
pub(super) struct CostFront {
    // A singleton is a feasible fixed value, not an empty piecewise function.
    points: FrontPoints,
}
impl CostFront {
    pub(super) fn zero(domain: Window) -> Self {
        let points = if domain.upper > domain.lower {
            FrontPoints::Two([[domain.lower, 0.], [domain.upper, 0.]])
        } else {
            FrontPoints::One([domain.lower, 0.])
        };
        Self { points }
    }
    pub(super) fn points(&self) -> &[[f64; 2]] {
        &self.points
    }
    pub(super) fn domain(&self) -> Window {
        Window {
            lower: self.points[0][0],
            upper: self.points[self.points.len() - 1][0],
        }
    }
    #[cfg(test)]
    pub(super) fn value(&self, x: f64) -> Result<f64> {
        let domain = self.domain();
        if x < domain.lower || x > domain.upper {
            return Err(invalid("interval-front coordinate escaped its domain"));
        }
        let i = self.points.partition_point(|p| p[0] < x);
        if self.points[i][0] == x {
            return Ok(self.points[i][1]);
        }
        let [a, b] = self.points[i - 1];
        let [c, d] = self.points[i];
        let t = finite(finite(x - a)? / finite(c - a)?)?;
        Ok(finite(b + finite(t * finite(d - b)?)?)?.clamp(b.min(d), b.max(d)))
    }
    pub(super) fn minimum(&self) -> f64 {
        self.points
            .iter()
            .map(|p| p[1])
            .fold(f64::INFINITY, f64::min)
    }
    pub(super) fn extend(
        &self,
        increment: f64,
        charge: &PiecewiseLinearFunction,
        bounds: Window,
        forward: bool,
        scale: f64,
    ) -> Result<Option<Self>> {
        let old = self.domain();
        let domain = charge.prepared_domain();
        let shift = if forward { increment } else { -increment };
        // A rounded translation that merges distinct old knots cannot be
        // represented as a single-valued frontier: different prefix costs
        // would share one coordinate. Report unsupported arithmetic rather
        // than silently dropping a feasible cheaper initial choice.
        let check_translation = |points: &[[f64; 2]], delta: f64| -> Result<()> {
            for pair in points.windows(2) {
                if finite(pair[0][0] + delta)? >= finite(pair[1][0] + delta)? {
                    return Err(invalid("interval-front translation merges distinct knots"));
                }
            }
            Ok(())
        };
        check_translation(&self.points, shift)?;
        if !forward {
            check_translation(&charge.points, -increment)?;
        }

        let lower = finite(old.lower + shift)?
            .max(bounds.lower)
            .max(if forward {
                domain.lower
            } else {
                finite(domain.lower - increment)?
            });
        let upper = finite(old.upper + shift)?
            .min(bounds.upper)
            .min(if forward {
                domain.upper
            } else {
                finite(domain.upper - increment)?
            });
        if lower > upper {
            return Ok(None);
        }
        // Shared knot grids are common. Reserving the sum retains nearly twice
        // the live front on every label in that case. Start with the larger
        // stream; disjoint grids still grow dynamically when needed.
        let mut points = FrontPoints::with_capacity(self.points.len().max(charge.points.len()) + 2);
        let mut previous_cursor = SegmentCursor::new(&self.points);
        let mut charge_cursor = SegmentCursor::new(&charge.points);
        merged_coordinates(
            &self.points,
            shift,
            &charge.points,
            if forward { 0. } else { -increment },
            Window { lower, upper },
            |x| {
                // Preserve the literal inverse-translation and endpoint clamps;
                // translated knots need not round-trip to their source knots.
                let previous = finite(x - shift)?.clamp(old.lower, old.upper);
                let argument = if forward { x } else { finite(x + increment)? }
                    .clamp(domain.lower, domain.upper);
                let value = finite(
                    previous_cursor.value(previous)?
                        + finite(scale * charge_cursor.value(argument)?)?,
                )?;
                push_exact_plateau(&mut points, [x, value]);
                Ok(())
            },
        )?;
        Ok(Some(Self { points }))
    }
    pub(super) fn joined_minimum(&self, other: &Self) -> Result<Option<f64>> {
        let a = self.domain();
        let b = other.domain();
        let lower = a.lower.max(b.lower);
        let upper = a.upper.min(b.upper);
        if lower > upper {
            return Ok(None);
        }
        let mut left = SegmentCursor::new(&self.points);
        let mut right = SegmentCursor::new(&other.points);
        let mut best = f64::INFINITY;
        merged_coordinates(
            &self.points,
            0.,
            &other.points,
            0.,
            Window { lower, upper },
            |x| {
                best = best.min(finite(left.value(x)? + right.value(x)?)?);
                Ok(())
            },
        )?;
        Ok(Some(best))
    }
}

/// Evaluate a sorted stream without repeating binary searches. Arithmetic and
/// exact-knot branches match the public piecewise-linear evaluator.
struct SegmentCursor<'a> {
    points: &'a [[f64; 2]],
    upper: usize,
}
impl<'a> SegmentCursor<'a> {
    fn new(points: &'a [[f64; 2]]) -> Self {
        Self { points, upper: 0 }
    }
    fn value(&mut self, x: f64) -> Result<f64> {
        while self.upper < self.points.len() && self.points[self.upper][0] < x {
            self.upper += 1;
        }
        let [c, d] = self.points[self.upper];
        if c == x {
            return Ok(d);
        }
        let [a, b] = self.points[self.upper - 1];
        let t = finite(finite(x - a)? / finite(c - a)?)?;
        Ok(finite(b + finite(t * finite(d - b)?)?)?.clamp(b.min(d), b.max(d)))
    }
}

/// Merge strictly ordered translated streams, clipping them to the intersection.
/// No temporary knot vector, sorting, or fixed piece limit is needed.
fn merged_coordinates(
    left: &[[f64; 2]],
    left_shift: f64,
    right: &[[f64; 2]],
    right_shift: f64,
    domain: Window,
    mut visit: impl FnMut(f64) -> Result<()>,
) -> Result<()> {
    visit(domain.lower)?;
    if domain.lower == domain.upper {
        return Ok(());
    }
    let (mut i, mut j) = (0, 0);
    loop {
        let x = if i < left.len() {
            finite(left[i][0] + left_shift)?
        } else {
            f64::INFINITY
        };
        let y = if j < right.len() {
            finite(right[j][0] + right_shift)?
        } else {
            f64::INFINITY
        };
        let next = x.min(y);
        if next >= domain.upper {
            break;
        }
        if x == next {
            i += 1;
        }
        if y == next {
            j += 1;
        }
        if next > domain.lower {
            visit(next)?;
        }
    }
    visit(domain.upper)
}

/// Removing general collinear knots changes floating-point interpolation.
/// Constant plateaus are the exact safe subset: keep endpoints and preserve
/// signed-zero branches and finite interpolation spans, without an epsilon.
fn push_exact_plateau(points: &mut FrontPoints, point: [f64; 2]) {
    if points.len() >= 2 {
        let a = points[points.len() - 2];
        let b = points[points.len() - 1];
        let bits = point[1].to_bits();
        if bits != (-0.0_f64).to_bits()
            && a[1].to_bits() == bits
            && b[1].to_bits() == bits
            && (point[0] - a[0]).is_finite()
        {
            *points.last_mut().unwrap() = point;
            return;
        }
    }
    points.push(point);
}

#[cfg(test)]
#[path = "../../../../../tests/unit/graph/coupled/kernel/front/tests.rs"]
mod tests;

#[cfg(all(test, feature = "clp"))]
#[path = "../../../../../tests/unit/graph/coupled/kernel/front/certificate_tests.rs"]
mod certificate_tests;
