//! Conservative continuation compensation for complete interval cost fronts.
use super::*;
use crate::graph::coupled::kernel::bounds::Range;

impl CostFront {
    /// Snapshot-only upper bound on `(self - min(self)) - (other - min(other))` over
    /// `other`'s domain. Domain inclusion is the feasibility requirement.
    /// This is NOT a continuation-cost certificate: common future rounded
    /// additions can amplify its difference. Kernel dominance uses a separate
    /// identity/zero-cost/frozen-future proof instead.
    ///
    /// Each merged cell lies in a single segment of both fronts. Literal
    /// interpolation is monotone there, including its exact-knot branches
    /// and endpoint clamp; its endpoint values therefore enclose every
    /// interior evaluation. Outward subtraction encloses normalization and
    /// cost compensation without a tolerance or assumed equal slopes.
    /// Correlation between two changing fronts is deliberately not assumed:
    /// such cells can yield an overestimate and retain additional labels.
    #[allow(dead_code)] // Retained snapshot comparison; not a continuation certificate.
    pub(in crate::graph::coupled::kernel) fn dominance_penalty(&self, other: &Self) -> Option<f64> {
        if self == other {
            return Some(0.);
        }
        // When both minima are literally zero and the interpolation grids
        // agree, compare the ordered arithmetic operands, not sampled slopes.
        // Both evaluators use the same nonnegative t. Ordered bases, rounded
        // deltas, and endpoint clamps imply pointwise literal value order.
        // This retains correlation that independent interval hulls lose.
        if self.minimum() == 0.
            && other.minimum() == 0.
            && self.points.len() == other.points.len()
            && self
                .points
                .iter()
                .zip(&other.points)
                .all(|(a, b)| a[0] == b[0] && a[1] <= b[1])
            && self
                .points
                .windows(2)
                .zip(other.points.windows(2))
                .all(|(a, b)| {
                    let da = a[1][1] - a[0][1];
                    let db = b[1][1] - b[0][1];
                    (a[1][0] - a[0][0]).is_finite() && da.is_finite() && db.is_finite() && da <= db
                })
        {
            return Some(0.);
        }
        let a = self.domain();
        let domain = other.domain();
        if a.lower > domain.lower || a.upper < domain.upper {
            return None;
        }
        // Exact endpoints can conceal a segment whose difference overflows.
        // Those fronts do not establish a finite interior evaluation contract.
        for front in [self, other] {
            for pair in front.points.windows(2) {
                if pair[0][0] < domain.upper
                    && pair[1][0] > domain.lower
                    && (!(pair[1][0] - pair[0][0]).is_finite()
                        || !(pair[1][1] - pair[0][1]).is_finite())
                {
                    return None;
                }
            }
        }
        let (a_min, b_min) = (self.minimum(), other.minimum());
        let mut left = SegmentCursor::new(&self.points);
        let mut right = SegmentCursor::new(&other.points);
        let mut previous: Option<[f64; 2]> = None;
        let mut penalty: f64 = 0.;
        merged_coordinates(&self.points, 0., &other.points, 0., domain, |x| {
            let values = [left.value(x)?, right.value(x)?];
            let before = previous.unwrap_or(values);
            let normalized = |i: usize, minimum: f64| {
                let mut range = Range {
                    lo: before[i].min(values[i]),
                    hi: before[i].max(values[i]),
                }
                .subtract(Range::point(minimum));
                // Every literal interpolation is clamped to its segment's
                // knot values, so global-minimum normalization is nonnegative.
                range.lo = range.lo.max(0.);
                range
            };
            let bound = normalized(0, a_min).subtract(normalized(1, b_min));
            if !bound.finite() {
                return Err(invalid("interval-front dominance bound overflow"));
            }
            penalty = penalty.max(bound.hi);
            previous = Some(values);
            Ok(())
        })
        .ok()?;
        Some(penalty)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn front(points: &[[f64; 2]]) -> CostFront {
        CostFront {
            points: points.to_vec().into(),
        }
    }
    #[test]
    fn normalized_constant_and_rising_fronts_use_directional_compensation() {
        let constant = front(&[[0., 8.], [4., 8.]]);
        let rising = front(&[[0., -2.], [4., 2.]]);
        assert_eq!(constant.dominance_penalty(&rising), Some(0.));
        assert_eq!(rising.dominance_penalty(&constant), Some(4.));
        assert_eq!(rising.dominance_penalty(&rising), Some(0.));
        assert_eq!(
            constant.dominance_penalty(&front(&[[-1., 0.], [4., 0.]])),
            None
        );
    }
    #[test]
    fn correlated_snapshot_orders_shared_literal_interpolation_operands() {
        let a = front(&[[0., 0.], [4., 4.]]);
        let b = front(&[[0., 0.], [4., 8.]]);
        assert_eq!(a.dominance_penalty(&b), Some(0.));
        // Future rounded additions remain outside this snapshot guarantee.
        let current = front(&[[0., 0.], [1., 1.]]);
        let zero = front(&[[0., 0.], [1., 0.]]);
        assert_eq!(current.dominance_penalty(&zero), Some(1.));
    }
    #[test]
    fn complete_front_bound_encloses_every_sample_and_retains_interior_knots() {
        let fronts = [
            front(&[[-2., 6.], [0., 0.], [1., 7.], [4., 2.], [6., -3.]]),
            front(&[[0., 4.], [0.5, -2.], [2., 5.], [4., 0.]]),
            front(&[[0., 0.], [4., 0.]]),
            front(&[[0., -1e12], [1., 1e12], [4., -1e12]]),
            front(&[[2., 3.]]),
        ];
        for a in &fronts {
            for b in &fronts {
                let domain = b.domain();
                let Some(penalty) = a.dominance_penalty(b) else {
                    continue;
                };
                assert!(penalty >= 0. && penalty.is_finite());
                for i in 0..=512 {
                    let x = domain.lower + (domain.upper - domain.lower) * (i as f64 / 512.);
                    // Outward comparisons also protect the test's subtraction
                    // from rounding in the opposite direction to the proof.
                    let a_value =
                        Range::point(a.value(x).unwrap()).subtract(Range::point(a.minimum()));
                    let b_value =
                        Range::point(b.value(x).unwrap()).subtract(Range::point(b.minimum()));
                    assert!(a_value.lo - b_value.hi <= penalty);
                }
            }
        }
        let peak = front(&[[0., 0.], [1., 9.], [2., 0.]]);
        let flat = front(&[[0., 0.], [2., 0.]]);
        assert_eq!(peak.dominance_penalty(&flat), Some(9.));
    }
    #[test]
    fn compensation_declines_overflow_but_supports_fixed_domains() {
        let huge = front(&[[0., -1e308], [1., 1e308]]);
        let zero = front(&[[0., 0.], [1., 0.]]);
        assert_eq!(huge.dominance_penalty(&zero), None);
        let fixed = front(&[[0.5, 2.]]);
        assert_eq!(zero.dominance_penalty(&fixed), Some(0.));
        assert_eq!(fixed.dominance_penalty(&zero), None);
    }
}
