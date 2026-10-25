use super::*;
fn cumulative(increments: Vec<f64>, weights: Vec<f64>) -> CoupledResource {
    CoupledResource::CumulativeCost {
        initial: 0.,
        increments,
        weights,
        sink_upper: None,
        source_upper: None,
    }
}
#[test]
fn accepted_positive_and_negative_cones_preserve_literal_costs_over_all_small_words() {
    for sign in [-1., 1.] {
        let increments = [sign, 2. * sign];
        let weights = [sign, 3. * sign];
        let order = NumericCostOrder::new(
            &[cumulative(increments.to_vec(), weights.to_vec())],
            &[Some(0)],
            1,
        );
        for smaller in 0..=3 {
            for larger in smaller..=3 {
                let a0 = sign * f64::from(smaller);
                let b0 = sign * f64::from(larger);
                for ac in 0..=3 {
                    for bc in ac..=3 {
                        assert!(order.preserves(&[a0], &[b0], f64::from(ac), f64::from(bc)));
                        for length in 0..=4 {
                            for word in 0..(1usize << length) {
                                let (mut a, mut b) = (a0, b0);
                                let (mut ca, mut cb) = (f64::from(ac), f64::from(bc));
                                // Independent physical recurrence; this
                                // does not call NumericOperation::apply.
                                for step in 0..length {
                                    let arc = (word >> step) & 1;
                                    a += increments[arc];
                                    b += increments[arc];
                                    ca += a * weights[arc];
                                    cb += b * weights[arc];
                                    assert!(ca.is_finite() && cb.is_finite() && ca <= cb);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
#[test]
fn sink_bound_lets_a_larger_state_pay_for_every_short_completion() {
    let (increments, weights) = ([1., 2.], [1., 3.]);
    let bounded = |sink_upper| {
        NumericCostOrder::new(
            &[CoupledResource::CumulativeCost {
                initial: 0.,
                increments: increments.to_vec(),
                weights: weights.to_vec(),
                sink_upper,
                source_upper: None,
            }],
            &[Some(0)],
            1,
        )
    };
    // At most four future arcs, each weighing at most 3.
    let order = bounded(Some(12.));
    assert!(order.preserves(&[3.], &[0.], 0., 36.));
    assert!(!order.preserves(&[3.], &[0.], 0., 35.));
    assert!(!bounded(None).preserves(&[3.], &[0.], 0., 36.));
    let mut accepted = 0;
    for a0 in 0..=3 {
        for b0 in 0..=3 {
            for ac in 0..=40 {
                for bc in ac..=40 {
                    let (a0, b0) = (f64::from(a0), f64::from(b0));
                    if !order.preserves(&[a0], &[b0], f64::from(ac), f64::from(bc)) {
                        continue;
                    }
                    accepted += usize::from(a0 > b0);
                    for length in 0..=4 {
                        for word in 0..(1usize << length) {
                            let (mut a, mut b) = (a0, b0);
                            let (mut ca, mut cb) = (f64::from(ac), f64::from(bc));
                            for step in 0..length {
                                let arc = (word >> step) & 1;
                                a += increments[arc];
                                b += increments[arc];
                                ca += a * weights[arc];
                                cb += b * weights[arc];
                            }
                            assert!(ca <= cb, "{a0} {b0} {ac} {bc} {word}");
                        }
                    }
                }
            }
        }
    }
    assert!(accepted > 0);
}
#[test]
fn signed_shared_charges_require_identical_cost_inputs_and_prefix_costs() {
    let definitions = [
        cumulative(vec![0., 1.], vec![-1., 1.]),
        cumulative(vec![0., 0.], vec![0., 0.]),
    ];
    let order = NumericCostOrder::new(&definitions, &[Some(0), Some(1)], 2);
    assert!(order.preserves(&[2., -7.], &[2., 9.], 3., 3.));
    assert!(!order.preserves(&[1., -7.], &[2., 9.], 3., 3.));
    assert!(!order.preserves(&[2., -7.], &[2., 9.], 2., 3.));
    for length in 0..=5 {
        for word in 0..(1usize << length) {
            let (mut a, mut b) = (2., 2.);
            let (mut ca, mut cb) = (3., 3.);
            for step in 0..length {
                let arc = (word >> step) & 1;
                a += [0., 1.][arc];
                b += [0., 1.][arc];
                ca += a * [-1., 1.][arc];
                cb += b * [-1., 1.][arc];
                assert_eq!(ca, cb);
            }
        }
    }
}
#[test]
fn large_finite_products_and_zero_weight_state_updates_keep_overflow_guards() {
    let order = NumericCostOrder::new(&[cumulative(vec![0.], vec![f64::MAX])], &[Some(0)], 1);
    // The wrong-sign dominator product overflows while B stays finite.
    let a: f64 = -2.;
    let b: f64 = 0.;
    assert!(!(a * f64::MAX).is_finite());
    assert!((b * f64::MAX).is_finite());
    assert!(!order.preserves(&[a], &[b], 0., 0.));
    assert!(order.preserves(&[0.], &[0.5], 0., 0.));
    let zero_weight = NumericCostOrder::new(&[cumulative(vec![f64::MAX], vec![0.])], &[Some(0)], 1);
    let a = f64::MAX;
    assert!(!(a + f64::MAX).is_finite());
    assert!((0. + f64::MAX).is_finite());
    assert!(!zero_weight.preserves(&[a], &[0.], 0., 0.));
    assert!(zero_weight.preserves(&[0.], &[a], 0., 0.));
}
#[test]
fn signed_zero_and_reachable_nonnegative_prefix_policy_are_explicit() {
    let order = NumericCostOrder::new(&[cumulative(vec![-0., 1.], vec![1., 2.])], &[Some(0)], 1);
    assert!(order.preserves(&[-0.], &[0.], -0., 0.));
    assert!(order.preserves(&[0.], &[f64::from_bits(1)], 0., f64::from_bits(1)));
    assert!(!order.preserves(&[0.], &[1.], -1., 0.));
    let signed = NumericCostOrder::new(&[cumulative(vec![0.], vec![-1.])], &[Some(0)], 1);
    // Prefix equality prevents a shared negative charge from overflowing
    // only the putative dominator's already very negative cost.
    assert!(!signed.preserves(&[2.], &[2.], -f64::MAX, 0.));
}
