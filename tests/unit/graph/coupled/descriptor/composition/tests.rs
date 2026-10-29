use super::*;
use crate::graph::coupled::descriptor::EvaluationPolicy;
fn f(pairs: &[(f64, f64)]) -> PreparedSlopeFunction {
    PreparedSlopeFunction::from_prepared_pairs(pairs.to_vec()).unwrap()
}
fn complete(result: FrontierOutcome<PreparedSlopeFunction>) -> PreparedSlopeFunction {
    match result {
        FrontierOutcome::Complete(f) => f,
        other => panic!("unexpected {other:?}"),
    }
}

#[test]
fn machine_point_inner_long_interval_retains_requested_start_and_no_segment() {
    // Static reference branch: ELF 649853 initializes inner segment id 1;
    // 6498ed-64990a stores the initial pair and skips the stream for size 1.
    // REA ev_7b76290689145ad474cf1a87289746495fd1c35128c387dc2c3e41fc6524bcf5.
    // This fixture is not an executed reference observation.
    let outer = f(&[(1., 0.), (2., 4.)]);
    let inner = f(&[(2., 1.)]);
    for lower in [1., 0.99999, 1.00001] {
        let actual = complete(compose_increasing(
            &outer,
            &inner,
            lower,
            2.,
            FrontOperationPolicy::FrontAlgebra,
            FrontierBudget::default(),
        ));
        assert_eq!(actual.pairs(), &[(5., lower)]);
        assert!(!actual.is_machine_zero_length());
    }
    let strict = complete(compose_increasing(
        &outer,
        &inner,
        0.99999,
        2.,
        FrontOperationPolicy::StrictDomain,
        FrontierBudget::default(),
    ));
    assert_eq!(strict.pairs(), &[(5., 1.)]);
    assert!(matches!(
        compose_increasing(
            &outer,
            &inner,
            1.00001,
            2.,
            FrontOperationPolicy::StrictDomain,
            FrontierBudget::default(),
        ),
        FrontierOutcome::Empty
    ));
    assert!(matches!(
        compose_increasing(
            &f(&[(1e12, 2.)]),
            &inner,
            1.,
            2.,
            FrontOperationPolicy::FrontAlgebra,
            FrontierBudget::default(),
        ),
        FrontierOutcome::Unknown(_)
    ));
    assert!(matches!(
        compose_increasing(
            &outer,
            &inner,
            1.,
            2.,
            FrontOperationPolicy::FrontAlgebra,
            FrontierBudget {
                max_pairs: 0,
                ..FrontierBudget::default()
            },
        ),
        FrontierOutcome::Unknown(FrontierUnknown::PairLimit)
    ));
}

#[test]
fn short_machine_composition_consumes_retained_zero_length_segments_by_lookup() {
    // The recovered short branch at ELF 649757-6497bf performs the two
    // lookups before any segment-stream operations. Its duplicate output at
    // 649ae0 is preserved across another short composition.
    let budget = FrontierBudget::default();
    let identity = f(&[(0., 0.), (1., 4.)]);
    let duplicate = complete(compose_increasing(
        &identity,
        &identity,
        1.,
        1.,
        FrontOperationPolicy::FrontAlgebra,
        budget,
    ));
    assert!(duplicate.is_machine_zero_length());
    for (outer, inner) in [
        (&identity, &duplicate),
        (&duplicate, &identity),
        (&duplicate, &duplicate),
    ] {
        let output = complete(compose_increasing(
            outer,
            inner,
            1.,
            1.,
            FrontOperationPolicy::FrontAlgebra,
            budget,
        ));
        assert_eq!(output.pairs(), &[(1., 1.), (0., 1.)]);
        assert!(output.is_machine_zero_length());
        let short = complete(compose_increasing(
            outer,
            inner,
            1.,
            1.00001,
            FrontOperationPolicy::FrontAlgebra,
            budget,
        ));
        assert_eq!(short.pairs(), &[(1., 1.), (0., 1.00001)]);
        assert!(!short.is_machine_zero_length());
        assert!(matches!(
            compose_increasing(
                outer,
                inner,
                1.,
                1.,
                FrontOperationPolicy::StrictDomain,
                budget,
            ),
            FrontierOutcome::Unknown(_)
        ));
        let long = compose_increasing(
            outer,
            inner,
            1.,
            2.,
            FrontOperationPolicy::FrontAlgebra,
            budget,
        );
        if !outer.is_machine_zero_length() {
            // A zero-width vector exhausts after the initial lookup; the
            // pinned helper returns this finite point on its long branch.
            assert_eq!(complete(long).pairs(), &[(1., 1.)]);
        } else {
            // The native duplicate outer can instead produce a NaN secant.
            assert!(matches!(long, FrontierOutcome::Unknown(_)));
        }
    }
    let prepared = match PreparedCompositionMap::new(&identity, budget) {
        FrontierOutcome::Complete(map) => map,
        other => panic!("unexpected prepared map: {other:?}"),
    };
    let output = complete(prepared.compose(
        &duplicate,
        1.,
        1.,
        FrontOperationPolicy::FrontAlgebra,
        budget,
    ));
    assert_eq!(output.pairs(), &[(1., 1.), (0., 1.)]);
    assert!(output.is_machine_zero_length());
}
#[test]
fn native_affine_and_outer_knot_secants() {
    let budget = FrontierBudget::default();
    let affine = complete(compose_increasing(
        &f(&[(1., 0.), (2., 10.)]),
        &f(&[(2., 0.), (3., 2.)]),
        0.,
        2.,
        FrontOperationPolicy::FrontAlgebra,
        budget,
    ));
    assert_eq!(affine.pairs(), &[(5., 0.), (6., 2.)]);
    let crossed = complete(compose_increasing(
        &f(&[(1., 0.), (2., 2.), (-1., 6.)]),
        &f(&[(0., 0.), (2., 3.)]),
        0.,
        3.,
        FrontOperationPolicy::FrontAlgebra,
        budget,
    ));
    assert_eq!(crossed.pairs(), &[(1., 0.), (4., 1.), (-2., 3.)]);
}
#[test]
fn inverse_retains_reciprocal_and_rejects_collapsed_images() {
    let inverse = complete(inverse_increasing(
        &f(&[(2., 1.), (3., 4.)]),
        FrontierBudget::default(),
    ));
    assert_eq!(inverse.pairs(), &[(1., 2.), (1. / 3., 11.)]);
    assert!(matches!(
        inverse_increasing(&f(&[(2., 0.), (0., 1.)]), FrontierBudget::default()),
        FrontierOutcome::Unknown(_)
    ));
    assert!(matches!(
        inverse_increasing(&f(&[(1e30, 0.), (1., 1.)]), FrontierBudget::default()),
        FrontierOutcome::Unknown(_)
    ));
}
#[test]
fn native_fractional_secants_match_retained_helper_observation() {
    // Pinned binary helper fixture, not idealized slope multiplication.
    let map = match PreparedCompositionMap::new(
        &f(&[(0.05, 0.), (0.7, 2.)]),
        FrontierBudget::default(),
    ) {
        FrontierOutcome::Complete(map) => map,
        other => panic!("unexpected prepared map {other:?}"),
    };
    let actual = complete(map.compose(
        &f(&[(0.1, 0.), (0.3, 0.7), (-0.2, 2.)]),
        0.1,
        1.7,
        FrontOperationPolicy::FrontAlgebra,
        FrontierBudget::default(),
    ));
    assert_eq!(
        actual.pairs(),
        &[
            (0.136, 0.1),
            (0.21, 0.9285714285714285),
            (-0.13999999999999999, 1.7)
        ]
    );
}
#[test]
fn clipped_native_and_strict_composition_preserve_their_arithmetic() {
    let outer = f(&[(0.1, 0.), (0.3, 0.7), (-0.2, 2.)]);
    let inner = f(&[(0.05, 0.), (0.7, 2.)]);
    let strict = complete(compose_increasing(
        &outer,
        &inner,
        0.1,
        1.7,
        FrontOperationPolicy::StrictDomain,
        FrontierBudget::default(),
    ));
    let native = complete(compose_increasing(
        &outer,
        &inner,
        0.1,
        1.7,
        FrontOperationPolicy::FrontAlgebra,
        FrontierBudget::default(),
    ));
    assert_eq!(strict.pairs()[2].0.to_bits(), (-0.14_f64).to_bits());
    assert_eq!(native.pairs()[2].0.to_bits(), 0xbfc1_eb85_1eb8_51eb);
}
#[test]
fn native_composition_carries_an_outer_knot_at_an_inner_segment_end() {
    let actual = complete(compose_increasing(
        &f(&[(0.16, 0.), (0.75, 0.7), (-0.79, 2.)]),
        &f(&[(0.034, 0.), (0.74, 0.9), (0.629, 2.)]),
        0.167,
        1.593,
        FrontOperationPolicy::FrontAlgebra,
        FrontierBudget::default(),
    ));
    let bits: Vec<_> = actual
        .pairs()
        .iter()
        .map(|(a, b)| (a.to_bits(), b.to_bits()))
        .collect();
    assert_eq!(
        bits,
        vec![
            (0x3fd1_cdc8_754f_3776, 0x3fc5_6041_8937_4bc7),
            (0x3fe1_c28f_5c28_f5c2, 0x3fec_cccc_cccc_cccd),
            (0xbfdf_cd5f_99c3_8b05, 0x3ff9_7ced_9168_72b0),
        ]
    );
}
#[test]
fn clipping_point_and_budget_are_not_guessed_infeasible() {
    let outer = f(&[(7., 2.)]);
    let inner = f(&[(0., 0.), (2., 2.)]);
    let point = complete(compose_increasing(
        &outer,
        &inner,
        0.,
        2.,
        FrontOperationPolicy::StrictDomain,
        FrontierBudget::default(),
    ));
    assert_eq!(point.pairs(), &[(7., 1.)]);
    assert!(matches!(
        compose_increasing(
            &outer,
            &inner,
            0.,
            2.,
            FrontOperationPolicy::StrictDomain,
            FrontierBudget {
                max_pairs: 0,
                ..FrontierBudget::default()
            }
        ),
        FrontierOutcome::Unknown(FrontierUnknown::PairLimit)
    ));
    assert!(matches!(
        compose_increasing(
            &f(&[(0., 10.), (1., 11.)]),
            &inner,
            0.,
            2.,
            FrontOperationPolicy::StrictDomain,
            FrontierBudget::default()
        ),
        FrontierOutcome::Empty
    ));
}
#[test]
fn strict_composition_preserves_negative_outer_slopes_and_domains() {
    let composed = complete(compose_increasing(
        &f(&[(4., 1.), (-2., 3.)]),
        &f(&[(0., 0.), (2., 2.)]),
        0.,
        2.,
        FrontOperationPolicy::StrictDomain,
        FrontierBudget::default(),
    ));
    assert_eq!(composed.pairs(), &[(4., 0.5), (-4., 1.5)]);
    assert_eq!(
        composed
            .evaluate(1., EvaluationPolicy::StrictDomain)
            .unwrap(),
        Some(2.)
    );
}

#[test]
fn native_exhausted_outer_domain_is_unknown_while_strict_composition_clips() {
    let outer = f(&[(0.25, 0.), (-0.5, 0.7), (0.75, 2.)]);
    let inner = f(&[(0.7, 0.), (0.75, 0.9), (0.75, 2.)]);
    for lower in [0.9, 1., 1.5] {
        for upper in [1.8, 2.] {
            assert!(matches!(
                compose_increasing(
                    &outer,
                    &inner,
                    lower,
                    upper,
                    FrontOperationPolicy::FrontAlgebra,
                    FrontierBudget::default()
                ),
                FrontierOutcome::Unknown(FrontierUnknown::Unsupported(_))
            ));
            let clipped = complete(compose_increasing(
                &outer,
                &inner,
                lower,
                upper,
                FrontOperationPolicy::StrictDomain,
                FrontierBudget::default(),
            ));
            assert_eq!(clipped.domain().0, lower);
            assert!(clipped.domain().1 < upper);
        }
        assert!(matches!(
            compose_increasing(
                &outer,
                &inner,
                lower,
                1.6,
                FrontOperationPolicy::FrontAlgebra,
                FrontierBudget::default()
            ),
            FrontierOutcome::Complete(_)
        ));
    }
}
#[test]
fn native_snaps_and_near_slope_merges_match_original_helper_bits() {
    // Original ELF SHA256 103caf4a...15889d6, helper 0x649680.
    // Fixed-array(12)/vector ABI checked against the complete assembly;
    // fixtures were executed in a separate original-library probe process.
    type Fixture<'a> = (
        &'a [(f64, f64)],
        &'a [(f64, f64)],
        f64,
        f64,
        &'a [(u64, u64)],
    );
    let cases: &[Fixture<'_>] = &[
        // snap-start
        (
            &[(1.0, 0.0), (2.0, 10.0)],
            &[(2.0, 0.0), (3.0, 2.0)],
            1e-05,
            2.0,
            &[
                (0x4014000000000000, 0x3ee4f8b588e368f1),
                (0x4017ffffffffffff, 0x4000000000000000),
            ],
        ),
        // tiny-interval
        (
            &[(1.0, 0.0), (2.0, 10.0)],
            &[(2.0, 0.0), (3.0, 2.0)],
            0.5,
            0.50001,
            &[
                (0x4020000000000000, 0x3fe0000000000000),
                (0x0000000000000000, 0x3fe00014f8b588e3),
            ],
        ),
        // near-endpoint-slopes
        (
            &[(0.0, 0.0), (1.0, 1.0), (1.00001, 2.0)],
            &[(0.0, 0.0), (1.0, 1.0), (1.0, 2.0)],
            0.0,
            2.0,
            &[
                (0x0000000000000000, 0x0000000000000000),
                (0x3ff0000000000000, 0x4000000000000000),
            ],
        ),
        // near-knot-slopes
        (
            &[(0.0, 0.0), (1.0, 1.0), (1.00001, 2.0)],
            &[(0.0, 0.0), (1.0, 2.0)],
            0.0,
            2.0,
            &[
                (0x0000000000000000, 0x0000000000000000),
                (0x3ff0000000000000, 0x4000000000000000),
            ],
        ),
        // near-inner-end-slopes
        (
            &[(0.0, 0.0), (1.0, 1.0), (1.00001, 4.0)],
            &[(0.0, 0.0), (1.0, 2.0)],
            0.0,
            2.0,
            &[
                (0x0000000000000000, 0x0000000000000000),
                (0x3ff0000a7c5ac472, 0x4000000000000000),
            ],
        ),
    ];
    for &(outer, inner, lo, hi, expected) in cases {
        let outer = f(outer);
        let inner = f(inner);
        let actual = complete(compose_increasing(
            &outer,
            &inner,
            lo,
            hi,
            FrontOperationPolicy::FrontAlgebra,
            FrontierBudget::default(),
        ));
        let bits: Vec<_> = actual
            .pairs()
            .iter()
            .map(|&(a, b)| (a.to_bits(), b.to_bits()))
            .collect();
        assert_eq!(bits, expected, "domain={lo:?}..{hi:?}");
        let prepared = match PreparedCompositionMap::new(&inner, FrontierBudget::default()) {
            FrontierOutcome::Complete(map) => map,
            other => panic!("{other:?}"),
        };
        let cached = complete(prepared.compose(
            &outer,
            lo,
            hi,
            FrontOperationPolicy::FrontAlgebra,
            FrontierBudget::default(),
        ));
        assert_eq!(cached.pairs(), actual.pairs());
    }
}

#[test]
fn native_constant_short_front_does_not_change_strict_composition() {
    let outer = f(&[(1., 0.), (2., 10.)]);
    let inner = f(&[(2., 0.), (3., 2.)]);
    let strict = complete(compose_increasing(
        &outer,
        &inner,
        0.5,
        0.50001,
        FrontOperationPolicy::StrictDomain,
        FrontierBudget::default(),
    ));
    assert!((strict.pairs()[1].0 - 6.).abs() < 1e-8);
    let native = complete(compose_increasing(
        &outer,
        &inner,
        0.5,
        0.50001,
        FrontOperationPolicy::FrontAlgebra,
        FrontierBudget::default(),
    ));
    assert_eq!(native.pairs()[1].0, 0.);
    assert!(matches!(
        compose_increasing(
            &outer,
            &inner,
            0.5,
            0.50001,
            FrontOperationPolicy::FrontAlgebra,
            FrontierBudget {
                max_pairs: 0,
                ..FrontierBudget::default()
            }
        ),
        FrontierOutcome::Unknown(FrontierUnknown::PairLimit)
    ));
    // Pinned reference composition at ELF 649680: the requested near-start
    // coordinate is retained, and the secant rounds above six.
    let near_start = complete(compose_increasing(
        &outer,
        &inner,
        -0.00001,
        1.,
        FrontOperationPolicy::FrontAlgebra,
        FrontierBudget::default(),
    ));
    assert_eq!(
        near_start.pairs(),
        &[(5., -0.00001), (6.000000000000001, 1.)]
    );
}
#[test]
fn native_near_knots_preserve_segment_advancement_and_endpoint_snaps() {
    // Original helper fixture: near-knot-end.
    let actual = complete(compose_increasing(
        &f(&[(-0.88, 0.0), (0.15, 0.7), (-0.57, 2.0)]),
        &f(&[(0.037, 0.0), (0.387, 0.9), (0.456, 2.0)]),
        0.251,
        1.59,
        FrontOperationPolicy::FrontAlgebra,
        FrontierBudget::default(),
    ));
    assert_eq!(
        actual
            .pairs()
            .iter()
            .map(|&(a, b)| (a.to_bits(), b.to_bits()))
            .collect::<Vec<_>>(),
        vec![
            (0xbfeb8421e8881436, 0x3fd010624dd2f1aa),
            (0x3fadb8bac710cb2f, 0x3feccccccccccccd),
            (0x3fb18384685be48d, 0x3ff970a3d70a3d71)
        ]
    );
    // Original helper fixture: near-knot-start.
    let actual = complete(compose_increasing(
        &f(&[(0.0, 0.0), (1.0, 1.0), (2.0, 4.0)]),
        &f(&[(0.99999, 0.0), (1.0, 2.0)]),
        0.0,
        1.0,
        FrontOperationPolicy::FrontAlgebra,
        FrontierBudget::default(),
    ));
    assert_eq!(
        actual
            .pairs()
            .iter()
            .map(|&(a, b)| (a.to_bits(), b.to_bits()))
            .collect::<Vec<_>>(),
        vec![
            (0x3fefffeb074a771d, 0x0000000000000000),
            (0x4000000000000000, 0x3ff0000000000000)
        ]
    );
    // Original helper fixture: near-knot-end-carried.
    let actual = complete(compose_increasing(
        &f(&[(0.0, 0.0), (1.0, 1.0), (2.0, 4.0)]),
        &f(&[(0.0, 0.0), (0.99999, 1.0), (1.0, 2.0)]),
        0.0,
        2.0,
        FrontOperationPolicy::FrontAlgebra,
        FrontierBudget::default(),
    ));
    assert_eq!(
        actual
            .pairs()
            .iter()
            .map(|&(a, b)| (a.to_bits(), b.to_bits()))
            .collect::<Vec<_>>(),
        vec![
            (0x0000000000000000, 0x0000000000000000),
            (0x3ff0000000000000, 0x3ff0000000000000),
            (0x3fffffeb074a771c, 0x4000000000000000)
        ]
    );
}
#[test]
fn short_inner_remainders_preserve_the_composition_anchor_and_stop_coordinate() {
    let outer = f(&[(0., 0.), (1., 1.), (2., 4.)]);
    let inner = f(&[(0., 0.), (1.00010001, 1.), (1., 2.)]);
    for (upper, expected) in [
        (1., vec![(0, 0), (0x3ff00068de3ade4d, 0x3fefff2e48e8ca4d)]),
        (
            1.5,
            vec![
                (0, 0),
                (0x3ff00068de3ade4d, 0x3fefff2e48e8ca4d),
                (0x40000000055e408a, 0x3ff8000000000000),
            ],
        ),
    ] {
        let result = complete(compose_increasing(
            &outer,
            &inner,
            0.,
            upper,
            FrontOperationPolicy::FrontAlgebra,
            FrontierBudget::default(),
        ));
        assert_eq!(
            result
                .pairs()
                .iter()
                .map(|&(a, b)| (a.to_bits(), b.to_bits()))
                .collect::<Vec<_>>(),
            expected,
        );
        let map = match PreparedCompositionMap::new(&inner, FrontierBudget::default()) {
            FrontierOutcome::Complete(map) => map,
            other => panic!("{other:?}"),
        };
        assert_eq!(
            complete(map.compose(
                &outer,
                0.,
                upper,
                FrontOperationPolicy::FrontAlgebra,
                FrontierBudget::default()
            ))
            .pairs(),
            result.pairs()
        );
        // Exact-domain composition still reaches the requested endpoint.
        let strict = complete(compose_increasing(
            &outer,
            &inner,
            0.,
            upper,
            FrontOperationPolicy::StrictDomain,
            FrontierBudget::default(),
        ));
        assert_eq!(strict.domain().1, upper);
    }
}

#[test]
fn composition_matches_retained_finite_and_tolerance_boundary_controls() {
    for line in
        include_str!("../../../../../fixtures/coupled/metasolver-composition-boundaries.txt")
            .lines()
            .filter(|line| !line.starts_with('#'))
    {
        let (input, reference) = line.split_once('|').unwrap();
        let mut input = input.split_whitespace();
        let name = input.next().unwrap();
        let outer_count: usize = input.next().unwrap().parse().unwrap();
        let inner_count: usize = input.next().unwrap().parse().unwrap();
        let lo: f64 = input.next().unwrap().parse().unwrap();
        let hi: f64 = input.next().unwrap().parse().unwrap();
        assert_eq!(input.next(), Some("0"));
        let mut pairs = |count| {
            (0..=count)
                .map(|_| {
                    (
                        input.next().unwrap().parse::<f64>().unwrap(),
                        input.next().unwrap().parse::<f64>().unwrap(),
                    )
                })
                .collect::<Vec<_>>()
        };
        let outer = f(&pairs(outer_count));
        let inner = f(&pairs(inner_count));
        let mut reference = reference.split_whitespace();
        assert_eq!(reference.next(), Some(name));
        assert_eq!(reference.next(), Some("1"));
        let count: usize = reference.next().unwrap().parse().unwrap();
        assert_eq!(reference.next(), Some("0"));
        let expected: Vec<_> = (0..=count)
            .map(|_| {
                (
                    u64::from_str_radix(reference.next().unwrap(), 16).unwrap(),
                    u64::from_str_radix(reference.next().unwrap(), 16).unwrap(),
                )
            })
            .collect();
        let actual = complete(compose_increasing(
            &outer,
            &inner,
            lo,
            hi,
            FrontOperationPolicy::FrontAlgebra,
            FrontierBudget::default(),
        ));
        assert_eq!(
            actual
                .pairs()
                .iter()
                .map(|&(a, b)| (a.to_bits(), b.to_bits()))
                .collect::<Vec<_>>(),
            expected,
            "{name}"
        );
    }
}

#[cfg(feature = "clp")]
#[test]
fn machine_composition_endpoint_and_degenerate_branches_match_recorded_binary_bits() {
    let fixtures: serde_json::Value = serde_json::from_str(include_str!(
        "../../../../../fixtures/coupled/metasolver-composition-operators.json"
    ))
    .unwrap();
    for fixture in fixtures.as_array().unwrap() {
        let function = |key: &str| {
            let pairs: Vec<(f64, f64)> = serde_json::from_value(fixture[key].clone()).unwrap();
            PreparedSlopeFunction::from_machine_pairs(pairs).unwrap()
        };
        let result = compose_increasing(
            &function("outer"),
            &function("inner"),
            fixture["lower"].as_f64().unwrap(),
            fixture["upper"].as_f64().unwrap(),
            FrontOperationPolicy::FrontAlgebra,
            FrontierBudget::default(),
        );
        if fixture["status"] == true && fixture["finite"] == true {
            let output = complete(result);
            let expected: Vec<(u64, u64)> = fixture["output_bits"]
                .as_array()
                .unwrap()
                .iter()
                .map(|pair| {
                    (
                        u64::from_str_radix(pair[0].as_str().unwrap(), 16).unwrap(),
                        u64::from_str_radix(pair[1].as_str().unwrap(), 16).unwrap(),
                    )
                })
                .collect();
            let actual: Vec<_> = output
                .pairs()
                .iter()
                .map(|pair| (pair.0.to_bits(), pair.1.to_bits()))
                .collect();
            assert_eq!(actual, expected, "{fixture}");
        } else {
            assert!(
                matches!(result, FrontierOutcome::Unknown(_)),
                "{fixture}: {result:?}"
            );
        }
    }
}

#[test]
fn machine_near_knot_composition_is_not_a_strict_source_function_certificate() {
    let outer = f(&[(1., 0.), (2., 0.0001)]);
    let inner = f(&[(0., 0.), (1., 0.00005)]);
    let raw = complete(compose_increasing(
        &outer,
        &inner,
        0.,
        1.,
        FrontOperationPolicy::FrontAlgebra,
        FrontierBudget::default(),
    ));
    let strict = complete(compose_increasing(
        &outer,
        &inner,
        0.,
        1.,
        FrontOperationPolicy::StrictDomain,
        FrontierBudget::default(),
    ));
    let source = 1. + 2. * 0.00005;
    assert_eq!(
        strict
            .evaluate(0.00005, EvaluationPolicy::StrictDomain)
            .unwrap(),
        Some(source)
    );
    assert_eq!(
        raw.evaluate(0.00005, EvaluationPolicy::StrictDomain)
            .unwrap(),
        Some(1.0002)
    );
    assert_ne!(
        Some(source),
        raw.evaluate(0.00005, EvaluationPolicy::StrictDomain)
            .unwrap()
    );
}
