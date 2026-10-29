use super::*;

#[test]
fn machine_truncation_preserves_near_knot_scan_order_and_one_sided_windows() {
    // Static vector-truncator evidence: REA
    // ev_6f512e19d40c4389f120392c4fd0b76ee90eaf56b60fe097f5ef1448d9b86f91;
    // ELF 6068e0-606941 consumes near lower knots, 6069f3-606a19
    // reanchors, and 606a24-606b59 copies the selected slopes.
    let function =
        PreparedSlopeFunction::from_prepared_pairs(vec![(2., 0.), (3., 1.), (-2., 2.), (4., 3.)])
            .unwrap();
    assert_eq!(
        function.machine_truncate(0., 1.5).unwrap().pairs(),
        &[(2., 0.), (3., 1.), (-2., 1.5)]
    );
    assert_eq!(
        function.machine_truncate(1.5, 4.).unwrap().pairs(),
        &[(4., 1.5), (-2., 2.), (4., 3.)]
    );
    assert_eq!(
        function.machine_truncate(1., 2.).unwrap().pairs(),
        &[(5., 1.), (-2., 2.)]
    );
    let lower = 0.99999;
    let anchor = 5. + (lower - 1.) * -2.;
    assert_eq!(
        function.machine_truncate(lower, 2.5).unwrap().pairs(),
        &[(anchor, lower), (-2., 2.), (4., 2.5)]
    );
    // A near upper knot retains the preceding active slope at the requested
    // coordinate. It is not a point-evaluated clip of the next segment.
    assert_eq!(
        function.machine_truncate(0.5, 1.00001).unwrap().pairs(),
        &[(3.5, 0.5), (3., 1.00001)]
    );
    assert_eq!(
        function.machine_truncate(0.5, 0.50001).unwrap().pairs(),
        &[(3.5, 0.5)]
    );
    let exhausted = function.machine_truncate(2.99999, 4.).unwrap();
    assert_eq!(exhausted.pairs(), &[(7., 2.99999), (0., 2.99999)]);
    assert!(exhausted.is_machine_zero_length());
    assert!(function.machine_truncate(4., 5.).is_err());
    assert!(function.machine_truncate(-2., -1.).is_err());
    assert!(function.machine_truncate(2., 1.).is_err());
}

#[test]
fn machine_truncation_retains_covering_bits_and_duplicate_segment_shapes() {
    let machine =
        PreparedSlopeFunction::from_machine_pairs(vec![(2., -0.0), (3., 0.0), (4., 1.)]).unwrap();
    let covered = machine.machine_truncate(-1., 2.).unwrap();
    for (original, retained) in machine.pairs().iter().zip(covered.pairs()) {
        assert_eq!(original.0.to_bits(), retained.0.to_bits());
        assert_eq!(original.1.to_bits(), retained.1.to_bits());
    }
    let clipped = machine.machine_truncate(-1., 0.5).unwrap();
    assert_eq!(clipped.pairs(), &[(2., -0.0), (3., 0.0), (4., 0.5)]);
    assert!(clipped.has_internal_zero_length());
    let point = PreparedSlopeFunction::from_machine_pairs(vec![(5., 2.), (3., 2.)]).unwrap();
    let covered = point.machine_truncate(1.99999, 2.00001).unwrap();
    assert_eq!(covered.pairs(), point.pairs());
    assert!(covered.is_machine_zero_length());
}

#[test]
fn vector_and_fixed_truncation_keep_their_distinct_short_and_upper_contracts() {
    // The fixed-array helper (REA
    // ev_cead0cf252057112ad1de41b9391ad5c894e82fdb776de22b0e0597bc9686f55)
    // does not cap the requested upper coordinate before copying segments.
    // ELF 723298-7232ec also forces one zero-width output segment when no
    // ordinary segment survives. The vector helper can retain just a point.
    let function =
        PreparedSlopeFunction::from_prepared_pairs(vec![(0., 0.), (1., 1.), (2., 1.00001)])
            .unwrap();
    assert_eq!(
        function.machine_truncate(0.5, 2.).unwrap().pairs(),
        &[(0.5, 0.5), (1., 1.00001)]
    );
    assert_eq!(
        function.machine_truncate_fixed(0.5, 2.).unwrap().pairs(),
        &[(0.5, 0.5), (1., 1.), (2., 1.00001)]
    );
    let vector = function.machine_truncate(0.5, 0.50001).unwrap();
    let fixed = function.machine_truncate_fixed(0.5, 0.50001).unwrap();
    assert_eq!(vector.pairs(), &[(0.5, 0.5)]);
    assert!(!vector.is_machine_zero_length());
    assert_eq!(fixed.pairs(), &[(0.5, 0.5), (0., 0.5)]);
    assert!(fixed.is_machine_zero_length());
}

#[test]
fn machine_pairs_preserve_zero_width_coordinate_bits_without_relaxing_source_input() {
    let pairs = vec![(2., -0.0), (3., 0.0), (4., 1.)];
    assert!(PreparedSlopeFunction::from_prepared_pairs(pairs.clone()).is_err());
    let machine = PreparedSlopeFunction::from_machine_pairs(pairs).unwrap();
    assert!(machine.is_machine_zero_length());
    assert!(machine.has_internal_zero_length());
    assert_eq!(machine.pairs()[0].1.to_bits(), (-0.0f64).to_bits());
    assert_eq!(machine.pairs()[1].1.to_bits(), 0.0f64.to_bits());
    assert!(PreparedSlopeFunction::from_machine_pairs(vec![(1., 0.), (2., 0.), (3., 0.)]).is_err());
}

#[cfg(feature = "clp")]
#[test]
fn machine_function_wire_round_trips_zero_width_but_source_decoder_rejects_it() {
    #[derive(serde::Deserialize)]
    struct Machine(
        #[serde(deserialize_with = "super::deserialize_machine_function")] PreparedSlopeFunction,
    );
    let machine = PreparedSlopeFunction::from_machine_pairs(vec![(1., -0.0), (2., 0.0)]).unwrap();
    let wire = serde_json::to_string(&machine).unwrap();
    assert!(!wire.contains("machine_zero_length"));
    assert!(!wire.contains("machine_internal_zero_length"));
    assert!(serde_json::from_str::<PreparedSlopeFunction>(&wire).is_err());
    let decoded: Machine = serde_json::from_str(&wire).unwrap();
    assert_eq!(decoded.0.pairs()[0].1.to_bits(), (-0.0f64).to_bits());
    assert_eq!(decoded.0.pairs()[1].1.to_bits(), 0.0f64.to_bits());
}

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
            .evaluate(-0.00001, EvaluationPolicy::EndpointTolerance)
            .unwrap(),
        Some(2.)
    );
    assert_eq!(
        function
            .evaluate(0.00001, EvaluationPolicy::EndpointTolerance)
            .unwrap(),
        Some(2.)
    );
    assert_eq!(
        function
            .evaluate(-0.0001, EvaluationPolicy::EndpointTolerance)
            .unwrap(),
        None
    );
    assert_eq!(
        function
            .evaluate(0.0001, EvaluationPolicy::EndpointTolerance)
            .unwrap(),
        Some(2.0003)
    );
    assert_eq!(
        function
            .evaluate(10.00001, EvaluationPolicy::EndpointTolerance)
            .unwrap(),
        Some(12.)
    );
    assert_eq!(
        function
            .evaluate(10.001, EvaluationPolicy::EndpointTolerance)
            .unwrap(),
        None
    );
    let large =
        PreparedSlopeFunction::from_prepared_pairs(vec![(2., 1e8), (3., 1e8 + 10.)]).unwrap();
    assert_eq!(
        large
            .evaluate(1e8 - 0.01, EvaluationPolicy::EndpointTolerance)
            .unwrap(),
        Some(2.)
    );
    assert_eq!(
        large
            .evaluate(1e8 + 0.01, EvaluationPolicy::EndpointTolerance)
            .unwrap(),
        Some(2.)
    );
    assert_eq!(
        large
            .evaluate(1e8 + 10.01, EvaluationPolicy::EndpointTolerance)
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
            .evaluate(1e308, EvaluationPolicy::EndpointTolerance)
            .unwrap(),
        Some(0.)
    );
    assert!(point
        .evaluate(1e308 - 1e300, EvaluationPolicy::EndpointTolerance)
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
            .evaluate(0., EvaluationPolicy::EndpointTolerance)
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
