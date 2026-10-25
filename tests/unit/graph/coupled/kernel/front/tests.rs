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
