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
mod tests {
    use super::*;
    fn window(lower: f64, upper: f64) -> Window {
        Window { lower, upper }
    }
    // Deliberately use concatenation/sorting and scalar evaluation as an
    // independent implementation of the previous front operation contract.
    fn reference_extension(
        source: &CostFront,
        increment: f64,
        charge: &PiecewiseLinearFunction,
        bounds: Window,
        forward: bool,
        scale: f64,
    ) -> CostFront {
        let old = source.domain();
        let domain = charge.prepared_domain();
        let shift = if forward { increment } else { -increment };
        let charge_shift = if forward { 0. } else { -increment };
        let lower = (old.lower + shift)
            .max(bounds.lower)
            .max(domain.lower + charge_shift);
        let upper = (old.upper + shift)
            .min(bounds.upper)
            .min(domain.upper + charge_shift);
        let mut knots = vec![lower, upper];
        knots.extend(source.points.iter().map(|p| p[0] + shift));
        knots.extend(charge.points.iter().map(|p| p[0] + charge_shift));
        knots.retain(|x| lower <= *x && *x <= upper);
        knots.sort_by(f64::total_cmp);
        knots.dedup();
        CostFront {
            points: knots
                .into_iter()
                .map(|x| {
                    let previous = (x - shift).clamp(old.lower, old.upper);
                    let argument =
                        if forward { x } else { x + increment }.clamp(domain.lower, domain.upper);
                    [
                        x,
                        source.value(previous).unwrap()
                            + scale * charge.evaluate(argument).unwrap().unwrap(),
                    ]
                })
                .collect(),
        }
    }
    #[test]
    fn linear_merge_matches_literal_reference_for_wide_and_translated_fronts() {
        // More than twelve pieces, alternating slopes, distinct/intersecting
        // knots, and non-dyadic translation exercise the dynamic fallback.
        let source = CostFront {
            points: (0..=96)
                .map(|i| [i as f64 * 0.125, ((i * 17) % 23) as f64 - 11.])
                .collect(),
        };
        let charge = PiecewiseLinearFunction::new(
            (0..=80)
                .map(|i| [i as f64 * 0.15, ((i * 11) % 19) as f64 - 9.])
                .collect(),
        )
        .unwrap();
        for forward in [false, true] {
            for increment in [-0.1, 0., 0.25, 0.3] {
                for scale in [-1., 0., 1., 0.125] {
                    let bounds = window(0.5, 11.5);
                    let expected =
                        reference_extension(&source, increment, &charge, bounds, forward, scale);
                    let actual = source
                        .extend(increment, &charge, bounds, forward, scale)
                        .unwrap()
                        .unwrap();
                    assert_eq!(actual.domain(), expected.domain());
                    for p in &expected.points {
                        assert_eq!(actual.value(p[0]).unwrap().to_bits(), p[1].to_bits());
                    }
                    let domain = actual.domain();
                    for i in 0..=257 {
                        let x = domain.lower + (domain.upper - domain.lower) * (i as f64 / 257.);
                        assert_eq!(
                            actual.value(x).unwrap().to_bits(),
                            expected.value(x).unwrap().to_bits()
                        );
                    }
                    let mut expected_join = f64::INFINITY;
                    let lower = actual.domain().lower.max(source.domain().lower);
                    let upper = actual.domain().upper.min(source.domain().upper);
                    for x in actual
                        .points
                        .iter()
                        .chain(&source.points)
                        .map(|p| p[0])
                        .chain([lower, upper])
                        .filter(|x| lower <= *x && *x <= upper)
                    {
                        expected_join =
                            expected_join.min(actual.value(x).unwrap() + source.value(x).unwrap());
                    }
                    assert_eq!(actual.joined_minimum(&source).unwrap(), Some(expected_join));
                }
            }
        }
    }
    #[test]
    fn exact_plateau_reduction_retains_signed_zero_and_finite_spans() {
        let mut constant = FrontPoints::Empty;
        for i in 0..=100 {
            push_exact_plateau(&mut constant, [i as f64, 4.]);
        }
        assert_eq!(constant, vec![[0., 4.], [100., 4.]]);
        let mut negative_zero = FrontPoints::Empty;
        for x in [0., 1., 2.] {
            push_exact_plateau(&mut negative_zero, [x, -0.]);
        }
        assert_eq!(negative_zero.len(), 3);
        let mut wide = FrontPoints::Empty;
        for x in [-1e308, 0., 1e308] {
            push_exact_plateau(&mut wide, [x, 1.]);
        }
        assert_eq!(wide.len(), 3);
        let source = CostFront {
            points: vec![[0., 1.], [1., 1.00001], [2., 4.00001]].into(),
        };
        let zero = PiecewiseLinearFunction::new(vec![[0., 0.], [2., 0.]]).unwrap();
        let actual = source
            .extend(0., &zero, window(0., 2.), true, 1.)
            .unwrap()
            .unwrap();
        assert_eq!(actual.points, source.points);
    }
    #[test]
    fn join_optimizes_one_shared_value_instead_of_independent_half_minima() {
        let forward = CostFront {
            points: vec![[0., 0.], [4., 4.]].into(),
        };
        let backward = CostFront {
            points: vec![[0., 0.], [4., -4.]].into(),
        };
        assert_eq!(forward.minimum() + backward.minimum(), -4.);
        assert_eq!(forward.joined_minimum(&backward).unwrap(), Some(0.));
        let restricted = CostFront::zero(window(4., 4.));
        assert_eq!(forward.joined_minimum(&restricted).unwrap(), Some(4.));
        assert_eq!(
            forward
                .joined_minimum(&CostFront::zero(window(5., 6.)))
                .unwrap(),
            None
        );
    }
    #[test]
    fn collapsed_translation_is_an_error_instead_of_losing_a_cheaper_choice() {
        let front = CostFront {
            points: vec![[1., 2.], [2., -1.]].into(),
        };
        let charge = PiecewiseLinearFunction::new(vec![[0., 0.], [2e20, 0.]]).unwrap();
        assert!(front
            .extend(1e20, &charge, window(0., 2e20), true, 1.)
            .is_err());
    }
    #[test]
    fn extension_keeps_interior_optimum_and_constraints_when_costs_are_suppressed() {
        let zero = CostFront::zero(window(0., 4.));
        let charge = PiecewiseLinearFunction::new(vec![[1., 3.], [3., -2.], [5., 3.]]).unwrap();
        let front = zero
            .extend(1., &charge, window(0., 6.), true, 1.)
            .unwrap()
            .unwrap();
        assert_eq!(front.minimum(), -2.);
        assert_eq!(front.domain(), window(1., 5.));
        let suppressed = zero
            .extend(1., &charge, window(0., 6.), true, 0.)
            .unwrap()
            .unwrap();
        assert_eq!(suppressed.minimum(), 0.);
        assert_eq!(suppressed.domain(), front.domain());
        assert!(zero
            .extend(10., &charge, window(0., 20.), true, 0.)
            .unwrap()
            .is_none());
    }
}

#[cfg(all(test, feature = "clp"))]
mod certificate_tests {
    use super::*;
    #[test]
    fn interval_front_witnesses_bound_opposing_costs_and_decline_fractional_translations() {
        let mut graph = Graph::<2>::new(3, 0).unwrap();
        graph.add_arc(0, 1, 0.).unwrap();
        graph.add_arc(1, 2, 0.).unwrap();
        let resource = |increment| CoupledResource::IntervalCost {
            initial: Window {
                lower: 0.,
                upper: 4.,
            },
            bounds: Window {
                lower: -4.,
                upper: 8.,
            },
            increments: vec![increment, -increment],
            functions: vec![
                PiecewiseLinearFunction::new(vec![[-4., -4.], [8., 8.]]).unwrap(),
                PiecewiseLinearFunction::new(vec![[-4., 4.], [8., -8.]]).unwrap(),
            ],
        };
        let mut kernel = CoupledKernel::prepare(&graph, &[resource(0.)]).unwrap();
        let (valid, [lo, hi]) = kernel.certify(&[0, 1]).unwrap();
        assert!(valid && lo <= 0. && (0. ..1e-10).contains(&hi));
        kernel.suppress_costs();
        let (valid, [lo, hi]) = kernel.certify(&[0, 1]).unwrap();
        assert!(valid && lo <= 0. && hi >= 0.);
        assert_eq!(
            CoupledKernel::prepare(&graph, &[resource(0.25)])
                .unwrap()
                .certify(&[0, 1]),
            None
        );
    }
}
