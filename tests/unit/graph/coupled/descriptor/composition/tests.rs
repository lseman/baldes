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
                FrontierOutcome::Unknown(FrontierUnknown::Unsupported(
                    "native composition image outside outer domain"
                ))
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
    assert!(matches!(
        compose_increasing(
            &outer,
            &inner,
            -0.00001,
            1.,
            FrontOperationPolicy::FrontAlgebra,
            FrontierBudget::default()
        ),
        FrontierOutcome::Unknown(_)
    ));
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
