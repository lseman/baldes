use super::*;

#[test]
fn native_prepared_slope_values_and_left_knot_are_preserved() {
    let function =
        PreparedSlopeFunction::from_prepared_pairs(vec![(2., 0.), (3., 5.), (-1., 10.)]).unwrap();
    let actual: Vec<_> = [0., 2., 5., 7., 10.]
        .into_iter()
        .map(|x| {
            function
                .evaluate(x, EvaluationPolicy::StrictDomain)
                .unwrap()
                .unwrap()
        })
        .collect();
    assert_eq!(actual, [2., 8., 17., 15., 12.]);
    let before = f64::from_bits(5_f64.to_bits() - 1);
    assert_eq!(
        function
            .evaluate(before, EvaluationPolicy::StrictDomain)
            .unwrap(),
        Some(16.999999999999996)
    );
    assert_eq!(
        function
            .evaluate(
                f64::from_bits(5_f64.to_bits() + 1),
                EvaluationPolicy::StrictDomain
            )
            .unwrap(),
        Some(17.)
    );
}

#[test]
fn fractional_native_witness_is_not_endpoint_knot_interpolation() {
    let x0 = 61.42565465487604;
    let y0 = 7.245618290940143;
    let slope = 9.462315279587411;
    let x1 = 68.72567520394728;
    let argument = 64.1889633870269;
    let function = PreparedSlopeFunction::from_prepared_pairs(vec![(y0, x0), (slope, x1)]).unwrap();
    let native = function
        .evaluate(argument, EvaluationPolicy::StrictDomain)
        .unwrap()
        .unwrap();
    let endpoint = y0 + slope * (x1 - x0);
    let knots = y0 + ((argument - x0) / (x1 - x0)) * (endpoint - y0);
    assert_eq!(native, 33.39291672938855);
    assert_eq!(knots, 33.39291672938856);
    assert_ne!(native.to_bits(), knots.to_bits());
}

#[test]
fn explicit_boundary_policy_matches_native_start_and_end_observations() {
    let function =
        PreparedSlopeFunction::from_prepared_pairs(vec![(2., 0.), (3., 5.), (-1., 10.)]).unwrap();
    assert_eq!(
        function
            .evaluate(-0.00001, EvaluationPolicy::StrictDomain)
            .unwrap(),
        None
    );
    assert_eq!(
        function
            .evaluate(-0.00001, EvaluationPolicy::NativeTolerance)
            .unwrap(),
        Some(2.)
    );
    assert_eq!(
        function
            .evaluate(0.00001, EvaluationPolicy::NativeTolerance)
            .unwrap(),
        Some(2.)
    );
    assert_eq!(
        function
            .evaluate(-0.0001, EvaluationPolicy::NativeTolerance)
            .unwrap(),
        None
    );
    assert_eq!(
        function
            .evaluate(0.0001, EvaluationPolicy::NativeTolerance)
            .unwrap(),
        Some(2.0003)
    );
    assert_eq!(
        function
            .evaluate(10.00001, EvaluationPolicy::NativeTolerance)
            .unwrap(),
        Some(12.)
    );
    assert_eq!(
        function
            .evaluate(10.001, EvaluationPolicy::NativeTolerance)
            .unwrap(),
        None
    );
    let large =
        PreparedSlopeFunction::from_prepared_pairs(vec![(2., 1e8), (3., 1e8 + 10.)]).unwrap();
    assert_eq!(
        large
            .evaluate(1e8 - 0.01, EvaluationPolicy::NativeTolerance)
            .unwrap(),
        Some(2.)
    );
    assert_eq!(
        large
            .evaluate(1e8 + 0.01, EvaluationPolicy::NativeTolerance)
            .unwrap(),
        Some(2.)
    );
    assert_eq!(
        large
            .evaluate(1e8 + 10.01, EvaluationPolicy::NativeTolerance)
            .unwrap(),
        Some(32.)
    );
}

#[test]
fn affine_pair_and_prepared_point_have_distinct_meanings() {
    let affine = RawAffine::from_pair((2., 3.)).unwrap();
    assert_eq!(affine.evaluate(10.).unwrap(), 32.);
    assert_eq!(
        affine.prepare(0., 10.).unwrap().pairs(),
        &[(2., 0.), (3., 10.)]
    );
    let point = PreparedSlopeFunction::constant_point(2., 3.).unwrap();
    assert_eq!(point.domain(), (3., 3.));
    assert_eq!(
        point.evaluate(3., EvaluationPolicy::StrictDomain).unwrap(),
        Some(2.)
    );
    assert_eq!(
        point.evaluate(10., EvaluationPolicy::StrictDomain).unwrap(),
        None
    );
    assert_eq!(affine.prepare(3., 3.).unwrap().pairs(), &[(11., 3.)]);
}

#[test]
fn genuine_sentinel_value_is_not_misclassified_as_outside() {
    let function = PreparedSlopeFunction::from_prepared_pairs(vec![(1e12, 0.), (0., 1.)]).unwrap();
    assert_eq!(
        function
            .evaluate(0.5, EvaluationPolicy::StrictDomain)
            .unwrap(),
        Some(1e12)
    );
    assert_eq!(
        function
            .evaluate(2., EvaluationPolicy::StrictDomain)
            .unwrap(),
        None
    );
}

#[test]
fn validation_and_arithmetic_failures_are_explicit() {
    assert!(PreparedSlopeFunction::from_prepared_pairs(vec![]).is_err());
    assert!(PreparedSlopeFunction::from_prepared_pairs(vec![(0., 0.), (1., 0.)]).is_err());
    assert!(PreparedSlopeFunction::from_prepared_pairs(vec![(0., 1.), (1., 0.)]).is_err());
    assert!(PreparedSlopeFunction::from_prepared_pairs(vec![(f64::NAN, 0.)]).is_err());
    assert!(PreparedSlopeFunction::from_prepared_pairs(vec![(0., 0.), (f64::MAX, 2.)]).is_err());
    assert!(RawAffine::new(f64::MAX, 1.)
        .unwrap()
        .evaluate(f64::MAX)
        .is_err());
    let point = PreparedSlopeFunction::constant_point(0., 1e308).unwrap();
    assert_eq!(
        point
            .evaluate(1e308, EvaluationPolicy::NativeTolerance)
            .unwrap(),
        Some(0.)
    );
    assert!(point
        .evaluate(1e308 - 1e300, EvaluationPolicy::NativeTolerance)
        .is_err());
    assert!(point
        .evaluate(f64::INFINITY, EvaluationPolicy::StrictDomain)
        .is_err());
}

#[test]
fn signed_zero_at_anchor_is_preserved_without_a_knot_conversion() {
    let point = PreparedSlopeFunction::constant_point(-0., 0.).unwrap();
    assert_eq!(
        point
            .evaluate(0., EvaluationPolicy::StrictDomain)
            .unwrap()
            .unwrap()
            .to_bits(),
        (-0_f64).to_bits()
    );
    let function = PreparedSlopeFunction::from_prepared_pairs(vec![(-0., 0.), (0., 1.)]).unwrap();
    assert_eq!(
        function
            .evaluate(0., EvaluationPolicy::StrictDomain)
            .unwrap()
            .unwrap()
            .to_bits(),
        (-0_f64).to_bits()
    );
    assert_eq!(
        function
            .evaluate(0., EvaluationPolicy::NativeTolerance)
            .unwrap()
            .unwrap()
            .to_bits(),
        (-0_f64).to_bits()
    );
    assert_eq!(
        function
            .evaluate(0.5, EvaluationPolicy::StrictDomain)
            .unwrap()
            .unwrap()
            .to_bits(),
        0_f64.to_bits()
    );
}

#[cfg(feature = "clp")]
#[test]
fn serde_round_trip_cannot_bypass_prepared_validation() {
    let function = PreparedSlopeFunction::from_prepared_pairs(vec![(2., 0.), (3., 5.)]).unwrap();
    let wire = serde_json::to_string(&function).unwrap();
    assert_eq!(
        serde_json::from_str::<PreparedSlopeFunction>(&wire).unwrap(),
        function
    );
    assert!(serde_json::from_str::<PreparedSlopeFunction>(r#"{"pairs":[[0,1],[2,0]]}"#).is_err());
    assert!(serde_json::from_str::<PreparedSlopeFunction>(r#"{"pairs":[]}"#).is_err());
    assert!(serde_json::from_str::<RawAffine>(r#"{"intercept":0,"slope":1,"extra":3}"#).is_err());
}
