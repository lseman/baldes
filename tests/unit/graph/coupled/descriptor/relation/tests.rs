use super::*;

#[cfg(feature = "clp")]
#[test]
fn general_machine_minimum_matches_binary_scan() {
    let rows: serde_json::Value = serde_json::from_str(include_str!(
        "../../../../../fixtures/coupled/metasolver-minimum.json"
    ))
    .unwrap();
    for (index, row) in rows.as_array().unwrap().iter().enumerate() {
        let decode = |name: &str| {
            function(
                &row[name]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|p| (p[0].as_f64().unwrap(), p[1].as_f64().unwrap()))
                    .collect::<Vec<_>>(),
            )
        };
        let result = PairedFrontOperations::machine_minimum_streams(
            &decode("left"),
            &decode("right"),
            FrontierBudget::default(),
        );
        if !row["accepted"].as_bool().unwrap() {
            assert!(
                matches!(result, FrontierOutcome::Empty),
                "case {index}: {result:?}"
            );
        } else if row["finite"].as_bool().unwrap() {
            let FrontierOutcome::Complete(front) = result else {
                panic!("case {index}: {result:?}");
            };
            let bits: Vec<Vec<String>> = front
                .pairs()
                .iter()
                .map(|p| {
                    [p.0, p.1]
                        .into_iter()
                        .map(|v| v.to_le_bytes().iter().map(|b| format!("{b:02x}")).collect())
                        .collect()
                })
                .collect();
            assert_eq!(
                serde_json::to_value(bits).unwrap(),
                row["bits"],
                "case {index}"
            );
        } else {
            assert!(
                matches!(result, FrontierOutcome::Unknown(_)),
                "case {index}: {result:?}"
            );
        }
    }
}

#[cfg(feature = "clp")]
#[test]
fn machine_backward_cost_comparison_matches_ordered_binary_scan() {
    let rows: serde_json::Value = serde_json::from_str(include_str!(
        "../../../../../fixtures/coupled/metasolver-backward-dominance.json"
    ))
    .unwrap();
    for (index, row) in rows.as_array().unwrap().iter().enumerate() {
        let decode = |name: &str| {
            function(
                &row[name]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|p| (p[0].as_f64().unwrap(), p[1].as_f64().unwrap()))
                    .collect::<Vec<_>>(),
            )
        };
        let result = if row["mode"].as_i64().unwrap() == 4 {
            match PairedFrontOperations::machine_backward_feasibility_dominance(
                &decode("left"),
                &decode("right"),
                FrontierBudget::default(),
            ) {
                FrontierOutcome::Complete(()) => {
                    FrontierOutcome::Complete(row["accumulated"].as_f64().unwrap())
                }
                FrontierOutcome::Empty => FrontierOutcome::Empty,
                FrontierOutcome::Unknown(reason) => FrontierOutcome::Unknown(reason),
            }
        } else {
            PairedFrontOperations::machine_backward_cost_dominance(
                &decode("left"),
                &decode("right"),
                row["mode"].as_i64().unwrap() as i32,
                row["accumulated"].as_f64().unwrap(),
                FrontierBudget::default(),
            )
        };
        if row["accepted"].as_bool().unwrap() {
            let FrontierOutcome::Complete(value) = result else {
                panic!("case {index}: {result:?}");
            };
            let bits: String = value
                .to_le_bytes()
                .iter()
                .map(|b| format!("{b:02x}"))
                .collect();
            assert_eq!(bits, row["output_bits"].as_str().unwrap(), "case {index}");
        } else {
            assert!(
                matches!(result, FrontierOutcome::Empty),
                "case {index}: {result:?}"
            );
        }
    }
}

#[test]
fn machine_lower_streams_keep_a_prefix_across_strictly_rejected_trailing_segments() {
    // REA ev_5534b90c3f5bf51193b5fdecc544e7f1a58f217883c90ec674d8cd6947f9ff1f.
    // At ELF 7237e0 the inactive/non-near/below case advances the streams;
    // 7237eb and 723522 reach status 2 only on re-entry or a later near tie.
    let lower = function(&[(2., 0.), (0., 3.)]);
    let input = SlopeFront::from_prepared(function(&[(3., 0.), (-2., 1.), (-1., 3.)]));
    let FrontierOutcome::Complete(result) = PairedFrontOperations::machine_backward_lower_streams(
        &input,
        &lower,
        FrontierBudget::default(),
    ) else {
        panic!("strictly rejected trailing segments should retain the prefix");
    };
    assert_eq!(result.components()[0].pairs(), &[(3., 0.), (-2., 0.5)]);
    for end_slope in [2., 0.5] {
        // The first alternative re-enters above; the second reaches a tie.
        let input = SlopeFront::from_prepared(function(&[(3., 0.), (-2., 1.), (end_slope, 3.)]));
        assert!(matches!(
            PairedFrontOperations::machine_backward_lower_streams(
                &input,
                &lower,
                FrontierBudget::default(),
            ),
            FrontierOutcome::Unknown(FrontierUnknown::Unsupported(
                "machine lower stream reentry status"
            ))
        ));
    }
}

#[test]
fn machine_lower_streams_preserve_near_ties_and_point_outputs_without_source_claims() {
    let lower = function(&[(2.00001, 0.), (0., 1.)]);
    let input = SlopeFront::from_prepared(function(&[(2., 0.), (0., 1.)]));
    let FrontierOutcome::Complete(result) =
        PairedFrontOperations::machine_backward_constant_lower_pass(
            &input,
            &lower,
            FrontierBudget::default(),
        )
    else {
        panic!("machine near tie should retain the original front");
    };
    assert_eq!(result.components()[0].pairs(), &[(2., 0.), (0., 1.)]);
    // This accepted machine front is strictly below the declared source
    // bound, so it must never be promoted to a source feasibility certificate.
    assert!(result.components()[0].pairs()[0].0 < lower.pairs()[0].0);
    let lower = function(&[(1., 0.), (0., 2.)]);
    let input = SlopeFront::from_prepared(function(&[(1., 0.), (-1., 2.)]));
    let FrontierOutcome::Complete(point) = PairedFrontOperations::machine_backward_lower_streams(
        &input,
        &lower,
        FrontierBudget::default(),
    ) else {
        panic!("a suppressed near-anchor crossing should retain a machine point");
    };
    assert_eq!(point.components()[0].pairs(), &[(1., 0.), (0., 0.)]);
    assert!(point.components()[0].is_machine_zero_length());
    let input = SlopeFront::from_prepared(function(&[(0., 0.), (0., 2.)]));
    assert!(matches!(
        PairedFrontOperations::machine_backward_lower_streams(
            &input,
            &lower,
            FrontierBudget::default(),
        ),
        // The pinned helper returns status zero with cleared count -1.
        // The standalone filter exposes no retained front; its backward
        // caller handles the later minimum stage separately.
        FrontierOutcome::Empty
    ));
}

#[test]
fn machine_lower_streams_begin_at_a_tied_endpoint_then_retain_original_slopes() {
    let lower = function(&[(1., 0.), (0., 2.)]);
    let input = SlopeFront::from_prepared(function(&[(0., 0.), (1., 1.), (1., 2.)]));
    let FrontierOutcome::Complete(result) = PairedFrontOperations::machine_backward_lower_streams(
        &input,
        &lower,
        FrontierBudget::default(),
    ) else {
        panic!("first tied endpoint should start the retained region");
    };
    assert_eq!(result.components()[0].pairs(), &[(1., 1.), (1., 2.)]);
}

#[test]
fn machine_minimum_streams_follow_near_anchor_endpoint_and_lookup_branches() {
    // Static minimum-helper evidence: REA
    // ev_aa05370a3124e437626f860d6fc7e2b44212ebf6b04a9f08b667bec61fed98f8,
    // ELF 723b10. These are instruction-derived fixtures, not executed probes.
    let minimum = |a: &[(f64, f64)], b: &[(f64, f64)]| {
        let a = PreparedSlopeFunction::from_prepared_pairs(a.to_vec()).unwrap();
        let b = PreparedSlopeFunction::from_prepared_pairs(b.to_vec()).unwrap();
        match PairedFrontOperations::machine_minimum_streams(&a, &b, FrontierBudget::default()) {
            FrontierOutcome::Complete(value) => value,
            other => panic!("unexpected minimum: {other:?}"),
        }
    };
    // A near initial tie switches slopes without emitting an interior
    // crossing. The retained anchor therefore differs from an ideal envelope.
    let result = minimum(&[(0., 0.), (1., 1.)], &[(0.00001, 0.), (0., 1.)]);
    assert_eq!(result.pairs(), &[(0., 0.), (0., 1.)]);
    assert_eq!(
        result
            .evaluate(0.5, EvaluationPolicy::StrictDomain)
            .unwrap(),
        Some(0.)
    );
    assert!(0.00001_f64.min(0.5) > 0.);
    // Near segment ends consume both streams and retain the fixed endpoint.
    assert_eq!(
        minimum(&[(0., 0.), (1., 1.)], &[(-10., 0.), (0., 1.00001)]).pairs(),
        &[(-10., 0.), (0., 1.)]
    );
    // Near starts use the original vector anchor at the fixed coordinate.
    assert_eq!(
        minimum(&[(2., 0.00001), (0., 1.)], &[(1., 0.), (0., 1.)]).pairs(),
        &[(1., 0.00001), (0., 1.)]
    );
    // A non-start lookup near a vector knot snaps to that knot's full value
    // and advances its segment id before the next comparison.
    assert_eq!(
        minimum(&[(4., 0.99999), (0., 2.)], &[(1., 0.), (2., 1.), (0., 2.)]).pairs(),
        &[(3., 0.99999), (0., 2.)]
    );
}

#[test]
fn machine_minimum_streams_retain_point_and_internal_duplicate_outputs() {
    let a = PreparedSlopeFunction::constant_point(2., 0.).unwrap();
    let b = PreparedSlopeFunction::from_prepared_pairs(vec![(1., 0.), (0., 1.)]).unwrap();
    let FrontierOutcome::Complete(point) =
        PairedFrontOperations::machine_minimum_streams(&a, &b, FrontierBudget::default())
    else {
        panic!("point input should preserve the machine segment count");
    };
    assert_eq!(point.pairs(), &[(1., 0.), (0., 0.)]);
    assert!(point.is_machine_zero_length());
    let a = PreparedSlopeFunction::from_machine_pairs(vec![(2., 0.), (3., 0.), (4., 1.)]).unwrap();
    let b = PreparedSlopeFunction::from_prepared_pairs(vec![(100., 0.), (0., 1.)]).unwrap();
    let FrontierOutcome::Complete(result) =
        PairedFrontOperations::machine_minimum_streams(&a, &b, FrontierBudget::default())
    else {
        panic!("minimum should retain the selected duplicate segment");
    };
    assert_eq!(result.pairs(), a.pairs());
    assert!(result.has_internal_zero_length());
    assert!(matches!(
        PairedFrontOperations::machine_minimum_streams(
            &a,
            &b,
            FrontierBudget {
                max_operations: 0,
                ..FrontierBudget::default()
            },
        ),
        FrontierOutcome::Unknown(FrontierUnknown::WorkLimit)
    ));
    assert!(matches!(
        PairedFrontOperations::machine_minimum_streams(
            &a,
            &b,
            FrontierBudget {
                max_elapsed: Some(std::time::Duration::ZERO),
                ..FrontierBudget::default()
            },
        ),
        FrontierOutcome::Unknown(FrontierUnknown::Deadline)
    ));
}

#[test]
fn machine_minimum_streams_cross_with_ordered_intercepts_and_merge_selected_slopes() {
    let a =
        PreparedSlopeFunction::from_prepared_pairs(vec![(0., 0.), (2., 2.), (-2., 4.)]).unwrap();
    let b = PreparedSlopeFunction::from_prepared_pairs(vec![(1., 0.), (0., 1.), (0., 4.)]).unwrap();
    let FrontierOutcome::Complete(result) =
        PairedFrontOperations::machine_minimum_streams(&a, &b, FrontierBudget::default())
    else {
        panic!("independent streams should complete");
    };
    assert_eq!(result.pairs(), &[(0., 0.), (2., 0.5), (0., 3.5), (-2., 4.)]);
    // Here the machine agrees with the exact mathematical envelope: all
    // anchors, crossings and segment arithmetic are dyadic.
    for x in [0., 0.25, 0.5, 1., 2., 3., 3.5, 3.75, 4.] {
        let av = a
            .evaluate(x, EvaluationPolicy::StrictDomain)
            .unwrap()
            .unwrap();
        let bv = b
            .evaluate(x, EvaluationPolicy::StrictDomain)
            .unwrap()
            .unwrap();
        assert_eq!(
            result.evaluate(x, EvaluationPolicy::StrictDomain).unwrap(),
            Some(av.min(bv))
        );
    }
}

#[test]
fn backward_initial_bounds_cross_a_covering_constant_upper() {
    let front = SlopeFront::from_prepared(
        PreparedSlopeFunction::from_prepared_pairs(vec![(0., 0.), (1., 2.)]).unwrap(),
    );
    let lower = PreparedSlopeFunction::from_prepared_pairs(vec![(-100., -1.), (0., 3.)]).unwrap();
    let upper = PreparedSlopeFunction::from_prepared_pairs(vec![(1., -1.), (0., 3.)]).unwrap();
    let result = PairedFrontOperations::machine_backward_initial_bounds(
        &front,
        0.,
        2.,
        0.,
        2.,
        &lower,
        &upper,
        FrontierBudget::default(),
    );
    let FrontierOutcome::Complete((result, limit)) = result else {
        panic!("unexpected covering upper minimum");
    };
    assert_eq!(result.pairs(), &[(0., 0.), (1., 1.), (0., 2.)]);
    assert_eq!(limit, 2.);
}

#[test]
fn backward_lower_filter_clips_single_front_against_covering_bound() {
    let lower = PreparedSlopeFunction::from_prepared_pairs(vec![(1., -1.), (0., 4.)]).unwrap();
    let front = SlopeFront::from_prepared(
        PreparedSlopeFunction::from_prepared_pairs(vec![(0., 0.), (1., 2.)]).unwrap(),
    );
    let result = PairedFrontOperations::machine_backward_constant_lower_pass(
        &front,
        &lower,
        FrontierBudget::default(),
    );
    let FrontierOutcome::Complete(result) = result else {
        panic!("unexpected covering lower crossing");
    };
    assert_eq!(result.components()[0].pairs(), &[(1., 1.), (1., 2.)]);
}

#[test]
fn backward_lower_filter_clips_multisegment_front_against_zero_bound() {
    let lower = PreparedSlopeFunction::from_prepared_pairs(vec![(0., -1.), (0., 4.)]).unwrap();
    let front = SlopeFront::from_prepared(
        PreparedSlopeFunction::from_prepared_pairs(vec![(-2., 0.), (1., 1.), (2., 2.), (1., 3.)])
            .unwrap(),
    );
    let result = PairedFrontOperations::machine_backward_constant_lower_pass(
        &front,
        &lower,
        FrontierBudget::default(),
    );
    let FrontierOutcome::Complete(result) = result else {
        panic!("unexpected zero lower crossing");
    };
    assert_eq!(
        result.components()[0].pairs(),
        &[(0., 1.5), (2., 2.), (1., 3.)]
    );
}

#[test]
fn backward_transition_caps_multisegment_front_by_constant_upper() {
    let make = |value, slope| {
        PreparedSlopeFunction::from_prepared_pairs(vec![(value, 0.), (slope, 2.)]).unwrap()
    };
    let incoming = SlopeFront::from_prepared(
        PreparedSlopeFunction::from_prepared_pairs(vec![(1., 0.), (2., 1.), (-0.5, 2.)]).unwrap(),
    );
    let mapping = make(0., 1.);
    let lower = make(0., 0.);
    let upper = make(2., 0.);
    let increment = make(0., 0.);
    let input = MachineBackwardInputs {
        waiting_lower: 0.,
        waiting_upper: 2.,
        forward_lower: 0.,
        mapped_upper: 2.,
        tail_lower: 0.,
        tail_upper: 2.,
        waiting_rate: 2,
        mapping: &mapping,
        inverse: &mapping,
        dependent: &increment,
        dependent_lower: &lower,
        dependent_upper: &upper,
    };
    let result = PairedFrontOperations::machine_backward_transition(
        &incoming,
        &input,
        100.,
        5,
        FrontierBudget::default(),
    );
    let FrontierOutcome::Complete((function, bits, key)) = result else {
        panic!("unexpected constant upper crossing transition");
    };
    assert_eq!(function.pairs(), &[(1., 0.), (2., 0.5), (0., 2.)]);
    assert_eq!(bits, 0);
    assert_eq!(key, 98.);
}

#[test]
fn backward_lower_filter_clips_multisegment_front_against_covering_constant() {
    let lower = PreparedSlopeFunction::from_prepared_pairs(vec![(1., -1.), (0., 4.)]).unwrap();
    let front = SlopeFront::from_prepared(
        PreparedSlopeFunction::from_prepared_pairs(vec![(-1., 0.), (1., 1.), (2., 2.), (1., 3.)])
            .unwrap(),
    );
    let result = PairedFrontOperations::machine_backward_constant_lower_pass(
        &front,
        &lower,
        FrontierBudget::default(),
    );
    let FrontierOutcome::Complete(result) = result else {
        panic!("unexpected covering constant lower crossing");
    };
    assert_eq!(
        result.components()[0].pairs(),
        &[(1., 1.5), (2., 2.), (1., 3.)]
    );
}

#[test]
fn backward_lower_filter_clips_one_crossing_in_aligned_streams() {
    let lower =
        PreparedSlopeFunction::from_prepared_pairs(vec![(0., 0.), (0., 1.), (0., 2.), (0., 3.)])
            .unwrap();
    let ascending = SlopeFront::from_prepared(
        PreparedSlopeFunction::from_prepared_pairs(vec![(-2., 0.), (1., 1.), (2., 2.), (1., 3.)])
            .unwrap(),
    );
    let result = PairedFrontOperations::machine_backward_constant_lower_pass(
        &ascending,
        &lower,
        FrontierBudget::default(),
    );
    let FrontierOutcome::Complete(result) = result else {
        panic!("unexpected aligned ascending lower crossing");
    };
    assert_eq!(
        result.components()[0].pairs(),
        &[(0., 1.5), (2., 2.), (1., 3.)]
    );
    let descending = SlopeFront::from_prepared(
        PreparedSlopeFunction::from_prepared_pairs(vec![(2., 0.), (-1., 1.), (-2., 2.), (-1., 3.)])
            .unwrap(),
    );
    let result = PairedFrontOperations::machine_backward_constant_lower_pass(
        &descending,
        &lower,
        FrontierBudget::default(),
    );
    let FrontierOutcome::Complete(result) = result else {
        panic!("strictly rejected trailing region should retain the prefix");
    };
    assert_eq!(
        result.components()[0].pairs(),
        &[(2., 0.), (-1., 1.), (-2., 1.5)]
    );
}

#[test]
fn backward_transition_uses_endpoint_after_lower_bound_clips_front() {
    let make = |value, slope| {
        PreparedSlopeFunction::from_prepared_pairs(vec![(value, 0.), (slope, 2.)]).unwrap()
    };
    let incoming = SlopeFront::from_prepared(
        PreparedSlopeFunction::from_prepared_pairs(vec![(2., 0.), (-1., 2.)]).unwrap(),
    );
    let mapping = make(0., 1.);
    let lower = make(1., 0.);
    let upper = make(100., 0.);
    let increment = make(0., 0.);
    let input = MachineBackwardInputs {
        waiting_lower: 0.,
        waiting_upper: 2.,
        forward_lower: 0.,
        mapped_upper: 2.,
        tail_lower: 0.,
        tail_upper: 2.,
        waiting_rate: 0,
        mapping: &mapping,
        inverse: &mapping,
        dependent: &increment,
        dependent_lower: &lower,
        dependent_upper: &upper,
    };
    let result = PairedFrontOperations::machine_backward_transition(
        &incoming,
        &input,
        100.,
        5,
        FrontierBudget::default(),
    );
    let FrontierOutcome::Complete((function, bits, key)) = result else {
        panic!("unexpected lower-clipped backward transition");
    };
    assert_eq!(function.pairs(), &[(2., 0.), (-1., 1.)]);
    assert_eq!(bits, 0);
    assert_eq!(key, 98.);
}

#[test]
fn backward_lower_filter_ends_at_strict_interior_crossing() {
    let prepared = PreparedSlopeFunction::from_prepared_pairs(vec![(1., 0.), (-1., 2.)]).unwrap();
    let lower = PreparedSlopeFunction::from_prepared_pairs(vec![(0., 0.), (0., 2.)]).unwrap();
    let front = SlopeFront::from_prepared(prepared);
    let result = PairedFrontOperations::machine_backward_constant_lower_pass(
        &front,
        &lower,
        FrontierBudget::default(),
    );
    let FrontierOutcome::Complete(result) = result else {
        panic!("unexpected descending crossing lower bound");
    };
    assert_eq!(result.components()[0].pairs(), &[(1., 0.), (-1., 1.)]);
}

#[test]
fn backward_lower_filter_starts_at_strict_interior_crossing() {
    let prepared = PreparedSlopeFunction::from_prepared_pairs(vec![(-1., 0.), (1., 2.)]).unwrap();
    let lower = PreparedSlopeFunction::from_prepared_pairs(vec![(0., 0.), (0., 2.)]).unwrap();
    let front = SlopeFront::from_prepared(prepared);
    let result = PairedFrontOperations::machine_backward_constant_lower_pass(
        &front,
        &lower,
        FrontierBudget::default(),
    );
    let FrontierOutcome::Complete(result) = result else {
        panic!("unexpected crossing lower bound");
    };
    assert_eq!(result.components()[0].pairs(), &[(0., 1.), (1., 2.)]);
}

#[test]
fn backward_minimum_crosses_inside_an_aligned_multisegment_front() {
    let fixed =
        PreparedSlopeFunction::from_prepared_pairs(vec![(0., 0.), (2., 1.), (0., 2.)]).unwrap();
    let upper =
        PreparedSlopeFunction::from_prepared_pairs(vec![(1., 0.), (0., 1.), (-1., 2.)]).unwrap();
    let result = PairedFrontOperations::machine_minimum_aligned_streams(
        &fixed,
        &upper,
        FrontierBudget::default(),
    );
    let FrontierOutcome::Complete(result) = result else {
        panic!("unexpected aligned crossing minimum");
    };
    assert_eq!(result.pairs(), &[(0., 0.), (2., 0.5), (0., 1.), (-1., 2.)]);
}

#[test]
fn backward_transition_prefixes_and_caps_multiple_steep_segments() {
    let make = |value, slope| {
        PreparedSlopeFunction::from_prepared_pairs(vec![(value, 0.), (slope, 4.)]).unwrap()
    };
    let incoming = SlopeFront::from_prepared(
        PreparedSlopeFunction::from_prepared_pairs(vec![(1., 1.), (2., 2.), (3., 3.), (0., 4.)])
            .unwrap(),
    );
    let mapping = make(0., 1.);
    let lower = make(0., 0.);
    let upper = make(100., 0.);
    let increment = make(0., 0.);
    let mut input = MachineBackwardInputs {
        waiting_lower: 1.,
        waiting_upper: 4.,
        forward_lower: 0.,
        mapped_upper: 3.5,
        tail_lower: 0.,
        tail_upper: 4.,
        waiting_rate: 1,
        mapping: &mapping,
        inverse: &mapping,
        dependent: &increment,
        dependent_lower: &lower,
        dependent_upper: &upper,
    };
    for (upper_limit, expected) in [
        (3.5, vec![(3., 0.), (1., 3.), (0., 3.5)]),
        (4., vec![(3., 0.), (1., 3.), (0., 4.)]),
    ] {
        input.mapped_upper = upper_limit;
        let result = PairedFrontOperations::machine_backward_transition(
            &incoming,
            &input,
            100.,
            5,
            FrontierBudget::default(),
        );
        let FrontierOutcome::Complete((function, bits, key)) = result else {
            panic!("unexpected steep-prefix backward transition");
        };
        assert_eq!(function.pairs(), expected);
        assert_eq!(bits, 1 << 1);
        assert_eq!(key, 94.);
    }
}

#[test]
fn backward_waiting_collapses_multiple_steep_leading_segments() {
    let front =
        PreparedSlopeFunction::from_prepared_pairs(vec![(0., 0.), (2., 1.), (3., 2.), (0., 3.)])
            .unwrap();
    let result = PairedFrontOperations::machine_backward_wait_full_window(
        &front,
        1,
        FrontierBudget::default(),
    );
    let FrontierOutcome::Complete((function, bits)) = result else {
        panic!("unexpected steep-prefix backward waiting envelope");
    };
    assert_eq!(function.pairs(), &[(3., 0.), (1., 2.), (0., 3.)]);
    assert_eq!(bits, 1 << 1);
}

#[test]
fn backward_waiting_intersection_scans_multiple_nonsteep_segments() {
    let front =
        PreparedSlopeFunction::from_prepared_pairs(vec![(0., 0.), (0., 1.), (0.25, 2.), (1.5, 3.)])
            .unwrap();
    let result = PairedFrontOperations::machine_backward_wait_full_window(
        &front,
        1,
        FrontierBudget::default(),
    );
    assert!(matches!(
        result,
        FrontierOutcome::Unknown(FrontierUnknown::Arithmetic(
            RawFunctionError::InvalidOrdering { .. }
        ))
    ));
}

#[test]
fn backward_transition_prefixes_and_caps_intersecting_waiting_envelope() {
    let make = |value, slope| {
        PreparedSlopeFunction::from_prepared_pairs(vec![(value, 0.), (slope, 3.)]).unwrap()
    };
    let incoming = SlopeFront::from_prepared(
        PreparedSlopeFunction::from_prepared_pairs(vec![(1., 1.), (0., 2.), (1.5, 3.)]).unwrap(),
    );
    let mapping = make(0., 1.);
    let lower = make(0., 0.);
    let upper = make(100., 0.);
    let increment = make(0., 0.);
    let input = MachineBackwardInputs {
        waiting_lower: 1.,
        waiting_upper: 3.,
        forward_lower: 0.,
        mapped_upper: 2.5,
        tail_lower: 0.,
        tail_upper: 3.,
        waiting_rate: 1,
        mapping: &mapping,
        inverse: &mapping,
        dependent: &increment,
        dependent_lower: &lower,
        dependent_upper: &upper,
    };
    let result = PairedFrontOperations::machine_backward_transition(
        &incoming,
        &input,
        100.,
        5,
        FrontierBudget::default(),
    );
    let FrontierOutcome::Complete((function, bits, key)) = result else {
        panic!("unexpected prefixed and capped intersecting waiting envelope");
    };
    assert_eq!(
        function.pairs(),
        &[(0., 0.), (1., 1.), (0., 1.5), (1., 2.5)]
    );
    assert_eq!(bits, (1 << 1) | (1 << 3));
    assert_eq!(key, 98.);
}

#[test]
fn backward_waiting_intersects_first_segment_before_steep_suffix() {
    let front =
        PreparedSlopeFunction::from_prepared_pairs(vec![(0., 0.), (0., 1.), (1.5, 2.)]).unwrap();
    let result = PairedFrontOperations::machine_backward_wait_full_window(
        &front,
        1,
        FrontierBudget::default(),
    );
    assert!(matches!(
        result,
        FrontierOutcome::Unknown(FrontierUnknown::Arithmetic(
            RawFunctionError::InvalidOrdering { .. }
        ))
    ));
}

#[test]
fn backward_waiting_dominates_mixed_front_with_steep_first_segment() {
    let front =
        PreparedSlopeFunction::from_prepared_pairs(vec![(0., 0.), (2., 1.), (0., 2.), (4., 3.)])
            .unwrap();
    let result = PairedFrontOperations::machine_backward_wait_full_window(
        &front,
        1,
        FrontierBudget::default(),
    );
    let FrontierOutcome::Complete((function, bits)) = result else {
        panic!("unexpected mixed waiting envelope");
    };
    assert_eq!(function.pairs(), &[(3., 0.), (1., 3.)]);
    assert_eq!(bits, 1 << 1);
}

#[test]
fn prepared_front_rejects_overflow_before_backward_arithmetic() {
    assert!(matches!(
        PreparedSlopeFunction::from_prepared_pairs(vec![(1., 0.), (f64::MAX, 2.), (1., 3.)]),
        Err(RawFunctionError::ArithmeticOverflow(_)),
    ));
}

#[test]
fn backward_bounds_accept_identical_multisegment_fronts() {
    let prepared =
        PreparedSlopeFunction::from_prepared_pairs(vec![(1., 0.), (1., 1.), (2., 2.)]).unwrap();
    let front = SlopeFront::from_prepared(prepared.clone());
    let lower = PairedFrontOperations::machine_backward_constant_lower_pass(
        &front,
        &prepared,
        FrontierBudget::default(),
    );
    let FrontierOutcome::Complete(lower) = lower else {
        panic!("unexpected equal lower bound");
    };
    assert_eq!(lower.components()[0].pairs(), prepared.pairs());
    let upper = PairedFrontOperations::machine_minimum_constant_multisegment(
        &prepared,
        &prepared,
        FrontierBudget::default(),
    );
    let FrontierOutcome::Complete(upper) = upper else {
        panic!("unexpected equal upper bound");
    };
    assert_eq!(upper.pairs(), prepared.pairs());
}

#[test]
fn backward_minimum_keeps_strictly_lower_aligned_piecewise_front() {
    let first =
        PreparedSlopeFunction::from_prepared_pairs(vec![(1., 0.), (1., 1.), (2., 2.)]).unwrap();
    let second =
        PreparedSlopeFunction::from_prepared_pairs(vec![(10., 0.), (-1., 1.), (1., 2.)]).unwrap();
    for (fixed, vector) in [(&first, &second), (&second, &first)] {
        let result = PairedFrontOperations::machine_minimum_constant_multisegment(
            fixed,
            vector,
            FrontierBudget::default(),
        );
        let FrontierOutcome::Complete(result) = result else {
            panic!("unexpected aligned piecewise minimum");
        };
        assert_eq!(result.pairs(), first.pairs());
    }
}

#[test]
fn backward_lower_filter_accepts_strictly_separated_aligned_slopes() {
    let prepared =
        PreparedSlopeFunction::from_prepared_pairs(vec![(5., 0.), (2., 1.), (-1., 2.)]).unwrap();
    let lower =
        PreparedSlopeFunction::from_prepared_pairs(vec![(0., 0.), (1., 1.), (1., 2.)]).unwrap();
    let front = SlopeFront::from_prepared(prepared.clone());
    let result = PairedFrontOperations::machine_backward_constant_lower_pass(
        &front,
        &lower,
        FrontierBudget::default(),
    );
    let FrontierOutcome::Complete(result) = result else {
        panic!("unexpected aligned lower filter");
    };
    assert_eq!(result.components()[0].pairs(), prepared.pairs());
}

#[test]
fn backward_composition_starts_on_inner_and_outer_breakpoints() {
    let outer =
        PreparedSlopeFunction::from_prepared_pairs(vec![(0., 0.), (1., 1.), (2., 3.)]).unwrap();
    let inner =
        PreparedSlopeFunction::from_prepared_pairs(vec![(0., 0.), (1., 1.), (2., 2.)]).unwrap();
    let result = PairedFrontOperations::machine_backward_compose_streams(
        &outer,
        1 << 2,
        &inner,
        1.,
        2.,
        FrontierBudget::default(),
    );
    let FrontierOutcome::Complete((function, bits)) = result else {
        panic!("unexpected breakpoint-start composition");
    };
    assert_eq!(function.pairs(), &[(1., 1.), (4., 2.)]);
    assert_eq!(bits, 1 << 1);
}

#[test]
fn backward_waiting_dominates_mixed_multisegment_front() {
    let front =
        PreparedSlopeFunction::from_prepared_pairs(vec![(0., 0.), (0., 1.), (2., 2.), (4., 3.)])
            .unwrap();
    let result = PairedFrontOperations::machine_backward_wait_full_window(
        &front,
        1,
        FrontierBudget::default(),
    );
    let FrontierOutcome::Complete((function, bits)) = result else {
        panic!("unexpected mixed waiting envelope");
    };
    assert_eq!(function.pairs(), &[(3., 0.), (1., 3.)]);
    assert_eq!(bits, 1 << 1);
}

#[test]
fn backward_multistream_composition_crosses_outer_with_one_mapping_segment() {
    let outer =
        PreparedSlopeFunction::from_prepared_pairs(vec![(0., 0.), (2., 2.), (1., 4.)]).unwrap();
    let inner = PreparedSlopeFunction::from_prepared_pairs(vec![(0., 0.), (1., 3.)]).unwrap();
    let result = PairedFrontOperations::machine_backward_compose_streams(
        &outer,
        1 << 1,
        &inner,
        0.5,
        2.5,
        FrontierBudget::default(),
    );
    let FrontierOutcome::Complete((function, bits)) = result else {
        panic!("unexpected single-mapping-segment composition");
    };
    assert_eq!(function.pairs(), &[(1., 0.5), (2., 2.), (1., 2.5)]);
    assert_eq!(bits, 1 << 1);
}

#[test]
fn backward_multistream_composition_accepts_interior_outer_bounds() {
    let outer =
        PreparedSlopeFunction::from_prepared_pairs(vec![(0., 0.), (2., 2.), (1., 4.)]).unwrap();
    let inner =
        PreparedSlopeFunction::from_prepared_pairs(vec![(0., 0.), (1., 1.), (1., 3.)]).unwrap();
    let result = PairedFrontOperations::machine_backward_compose_streams(
        &outer,
        1 << 1,
        &inner,
        0.5,
        2.5,
        FrontierBudget::default(),
    );
    let FrontierOutcome::Complete((function, bits)) = result else {
        panic!("unexpected interior outer composition");
    };
    assert_eq!(function.pairs(), &[(1., 0.5), (2., 2.), (1., 2.5)]);
    assert_eq!(bits, 1 << 1);
}

#[test]
fn backward_multistream_composition_starts_and_stops_inside_mapping_segments() {
    let outer =
        PreparedSlopeFunction::from_prepared_pairs(vec![(1., 0.5), (2., 2.), (1., 2.5)]).unwrap();
    let inner =
        PreparedSlopeFunction::from_prepared_pairs(vec![(0., 0.), (1., 1.), (1.5, 3.)]).unwrap();
    let result = PairedFrontOperations::machine_backward_compose_streams(
        &outer,
        1 << 1,
        &inner,
        0.5,
        2.,
        FrontierBudget::default(),
    );
    let FrontierOutcome::Complete((function, bits)) = result else {
        panic!("unexpected bounded multi-stream composition");
    };
    let crossing = 1. + 1. / 1.5;
    assert_eq!(
        function.pairs(),
        &[
            (1., 0.5),
            (2., 1.),
            (2. / (crossing - 1.), crossing),
            (0.5 / (2. - crossing), 2.),
        ]
    );
    assert_eq!(bits, (1 << 1) | (1 << 2));
}

#[test]
fn backward_multistream_composition_stops_inside_mapping_segment() {
    let outer =
        PreparedSlopeFunction::from_prepared_pairs(vec![(1., 0.), (2., 2.), (1., 2.5)]).unwrap();
    let inner =
        PreparedSlopeFunction::from_prepared_pairs(vec![(0., 0.), (1., 1.), (1.5, 3.)]).unwrap();
    let result = PairedFrontOperations::machine_backward_compose_streams(
        &outer,
        1 << 1,
        &inner,
        0.,
        2.,
        FrontierBudget::default(),
    );
    let FrontierOutcome::Complete((function, bits)) = result else {
        panic!("unexpected capped multi-stream composition");
    };
    let crossing = 1. + 1. / 1.5;
    assert_eq!(
        function.pairs(),
        &[
            (1., 0.),
            (2., 1.),
            (2. / (crossing - 1.), crossing),
            (0.5 / (2. - crossing), 2.),
        ]
    );
    assert_eq!(bits, (1 << 1) | (1 << 2));
}

#[test]
fn backward_composition_walks_both_multi_segment_streams() {
    let outer =
        PreparedSlopeFunction::from_prepared_pairs(vec![(1., 0.), (2., 2.), (1., 4.)]).unwrap();
    let inner =
        PreparedSlopeFunction::from_prepared_pairs(vec![(0., 0.), (1., 1.), (1.5, 3.)]).unwrap();
    let result = PairedFrontOperations::machine_backward_compose_streams(
        &outer,
        1 << 1,
        &inner,
        0.,
        3.,
        FrontierBudget::default(),
    );
    let FrontierOutcome::Complete((function, bits)) = result else {
        panic!("unexpected multi-stream backward composition");
    };
    let crossing = 1. + 1. / 1.5;
    assert_eq!(
        function.pairs(),
        &[
            (1., 0.),
            (2., 1.),
            (2. / (crossing - 1.), crossing),
            (2. / (3. - crossing), 3.),
        ]
    );
    assert_eq!(bits, (1 << 1) | (1 << 2));
}

#[test]
fn backward_transition_combines_prefix_with_multisegment_cap() {
    let incoming = SlopeFront::from_prepared(
        PreparedSlopeFunction::from_prepared_pairs(vec![(1., 2.), (2., 4.), (0., 8.)]).unwrap(),
    );
    let make = |value, slope| {
        PreparedSlopeFunction::from_prepared_pairs(vec![(value, 0.), (slope, 8.)]).unwrap()
    };
    let mapping = make(0., 1.);
    let lower = make(0., 0.);
    let upper = make(100., 0.);
    let increment = make(0., 0.);
    let input = MachineBackwardInputs {
        waiting_lower: 2.,
        waiting_upper: 8.,
        forward_lower: 0.,
        mapped_upper: 6.,
        tail_lower: 0.,
        tail_upper: 8.,
        waiting_rate: 0,
        mapping: &mapping,
        inverse: &mapping,
        dependent: &increment,
        dependent_lower: &lower,
        dependent_upper: &upper,
    };
    let result = PairedFrontOperations::machine_backward_transition(
        &incoming,
        &input,
        100.,
        5,
        FrontierBudget::default(),
    );
    let FrontierOutcome::Complete((function, bits, key)) = result else {
        panic!("unexpected multisegment prefix and cap");
    };
    assert_eq!(function.pairs(), &[(5., 0.), (0., 4.), (0., 6.)]);
    assert_eq!(bits, 1 << 1);
    assert_eq!(key, 95.);
}

#[test]
fn backward_transition_caps_intersecting_waiting_envelope() {
    let make = |value, slope| {
        PreparedSlopeFunction::from_prepared_pairs(vec![(value, 0.), (slope, 2.)]).unwrap()
    };
    let incoming = SlopeFront::from_prepared(
        PreparedSlopeFunction::from_prepared_pairs(vec![(1., 0.), (0., 1.), (1.5, 2.)]).unwrap(),
    );
    let mapping = make(0., 1.);
    let lower = make(0., 0.);
    let upper = make(100., 0.);
    let increment = make(0., 0.);
    let input = MachineBackwardInputs {
        waiting_lower: 0.,
        waiting_upper: 2.,
        forward_lower: 0.,
        mapped_upper: 1.5,
        tail_lower: 0.,
        tail_upper: 2.,
        waiting_rate: 1,
        mapping: &mapping,
        inverse: &mapping,
        dependent: &increment,
        dependent_lower: &lower,
        dependent_upper: &upper,
    };
    let result = PairedFrontOperations::machine_backward_transition(
        &incoming,
        &input,
        100.,
        5,
        FrontierBudget::default(),
    );
    let FrontierOutcome::Complete((function, bits, key)) = result else {
        panic!("unexpected intersecting waiting cap");
    };
    // Pinned modern backward helper at ELF 7281e0 retains its value anchor.
    assert_eq!(function.pairs(), &[(2.5, 0.), (0., 0.5), (1., 1.5)]);
    assert_eq!(bits, 1 << 2);
    assert_eq!(key, 96.5);
}

#[test]
fn backward_transition_caps_a_multi_segment_waiting_front() {
    let make = |value, slope| {
        PreparedSlopeFunction::from_prepared_pairs(vec![(value, 0.), (slope, 2.)]).unwrap()
    };
    let incoming = SlopeFront::from_prepared(
        PreparedSlopeFunction::from_prepared_pairs(vec![(1., 0.), (2., 1.), (0., 2.)]).unwrap(),
    );
    let mapping = make(0., 1.);
    let lower = make(0., 0.);
    let upper = make(100., 0.);
    let increment = make(0., 0.);
    let mut input = MachineBackwardInputs {
        waiting_lower: 0.,
        waiting_upper: 2.,
        forward_lower: 0.,
        mapped_upper: 1.5,
        tail_lower: 0.,
        tail_upper: 2.,
        waiting_rate: 2,
        mapping: &mapping,
        inverse: &mapping,
        dependent: &increment,
        dependent_lower: &lower,
        dependent_upper: &upper,
    };
    let result = PairedFrontOperations::machine_backward_transition(
        &incoming,
        &input,
        100.,
        5,
        FrontierBudget::default(),
    );
    let FrontierOutcome::Complete((function, bits, key)) = result else {
        panic!("unexpected multi-segment waiting cap");
    };
    assert_eq!(function.pairs(), &[(1., 0.), (2., 1.), (0., 1.5)]);
    assert_eq!(bits, 0);
    assert_eq!(key, 97.);
    input.waiting_rate = 0;
    let result = PairedFrontOperations::machine_backward_transition(
        &incoming,
        &input,
        100.,
        5,
        FrontierBudget::default(),
    );
    let FrontierOutcome::Complete((function, bits, key)) = result else {
        panic!("unexpected mixed-slope waiting cap");
    };
    assert_eq!(function.pairs(), &[(3., 0.), (0., 1.), (0., 1.5)]);
    assert_eq!(bits, 1 << 1);
    assert_eq!(key, 97.);
}

#[test]
fn backward_transition_supports_prefix_and_cap_together() {
    let make = |value, slope| {
        PreparedSlopeFunction::from_prepared_pairs(vec![(value, 0.), (slope, 8.)]).unwrap()
    };
    let mapping = make(0., 1.);
    let lower = make(0., 0.);
    let upper = make(100., 0.);
    let increment = make(0., 0.);
    let input = MachineBackwardInputs {
        waiting_lower: 2.,
        waiting_upper: 8.,
        forward_lower: 0.,
        mapped_upper: 6.,
        tail_lower: 0.,
        tail_upper: 8.,
        waiting_rate: 0,
        mapping: &mapping,
        inverse: &mapping,
        dependent: &increment,
        dependent_lower: &lower,
        dependent_upper: &upper,
    };
    for (slope, expected, bits, key) in [
        (0., vec![(4., 0.), (0., 2.), (0., 6.)], 1 << 1, 96.),
        (1., vec![(10., 0.), (0., 6.)], 1 << 1, 90.),
    ] {
        let front = SlopeFront::from_prepared(
            PreparedSlopeFunction::from_prepared_pairs(vec![(4., 2.), (slope, 8.)]).unwrap(),
        );
        let result = PairedFrontOperations::machine_backward_transition(
            &front,
            &input,
            100.,
            5,
            FrontierBudget::default(),
        );
        let FrontierOutcome::Complete((function, actual_bits, actual_key)) = result else {
            panic!("unexpected prefixed capped backward transition");
        };
        assert_eq!(function.pairs(), expected);
        assert_eq!(actual_bits, bits);
        assert_eq!(actual_key, key);
    }
}

#[test]
fn backward_transition_caps_waiting_at_mapped_upper_limit() {
    let make = |value, slope| {
        PreparedSlopeFunction::from_prepared_pairs(vec![(value, 0.), (slope, 8.)]).unwrap()
    };
    let incoming = SlopeFront::from_prepared(make(4., 1.));
    let mapping = make(0., 1.);
    let lower = make(0., 0.);
    let upper = make(100., 0.);
    let increment = make(0., 0.);
    let input = MachineBackwardInputs {
        waiting_lower: 0.,
        waiting_upper: 8.,
        forward_lower: 0.,
        mapped_upper: 6.,
        tail_lower: 0.,
        tail_upper: 8.,
        waiting_rate: 0,
        mapping: &mapping,
        inverse: &mapping,
        dependent: &increment,
        dependent_lower: &lower,
        dependent_upper: &upper,
    };
    let result = PairedFrontOperations::machine_backward_transition(
        &incoming,
        &input,
        100.,
        5,
        FrontierBudget::default(),
    );
    let FrontierOutcome::Complete((function, bits, key)) = result else {
        panic!("unexpected capped backward transition");
    };
    assert_eq!(function.pairs(), &[(12., 0.), (0., 6.)]);
    assert_eq!(bits, 1 << 1);
    assert_eq!(key, 88.);
}

#[test]
fn backward_transition_extends_waiting_prefix_before_retained_suffix() {
    let incoming = SlopeFront::from_prepared(
        PreparedSlopeFunction::from_prepared_pairs(vec![(1., 2.), (2., 4.), (0., 8.)]).unwrap(),
    );
    let make = |value, slope| {
        PreparedSlopeFunction::from_prepared_pairs(vec![(value, 0.), (slope, 8.)]).unwrap()
    };
    let mapping = make(0., 1.);
    let lower = make(0., 0.);
    let upper = make(100., 0.);
    let increment = make(0., 0.);
    let input = MachineBackwardInputs {
        waiting_lower: 2.,
        waiting_upper: 8.,
        forward_lower: 0.,
        mapped_upper: 8.,
        tail_lower: 0.,
        tail_upper: 8.,
        waiting_rate: 0,
        mapping: &mapping,
        inverse: &mapping,
        dependent: &increment,
        dependent_lower: &lower,
        dependent_upper: &upper,
    };
    let result = PairedFrontOperations::machine_backward_transition(
        &incoming,
        &input,
        100.,
        5,
        FrontierBudget::default(),
    );
    let FrontierOutcome::Complete((function, bits, key)) = result else {
        panic!("unexpected backward waiting prefix and suffix");
    };
    assert_eq!(function.pairs(), &[(5., 0.), (0., 4.), (0., 8.)]);
    assert_eq!(bits, 1 << 1);
    assert_eq!(key, 95.);
}

#[test]
fn backward_transition_extends_an_existing_waiting_segment() {
    let incoming = SlopeFront::from_prepared(
        PreparedSlopeFunction::from_prepared_pairs(vec![(4., 2.), (1., 8.)]).unwrap(),
    );
    let make = |value, slope| {
        PreparedSlopeFunction::from_prepared_pairs(vec![(value, 0.), (slope, 8.)]).unwrap()
    };
    let mapping = make(0., 1.);
    let lower = make(0., 0.);
    let upper = make(100., 0.);
    let increment = make(0., 0.);
    let input = MachineBackwardInputs {
        waiting_lower: 2.,
        waiting_upper: 8.,
        forward_lower: 0.,
        mapped_upper: 8.,
        tail_lower: 0.,
        tail_upper: 8.,
        waiting_rate: 0,
        mapping: &mapping,
        inverse: &mapping,
        dependent: &increment,
        dependent_lower: &lower,
        dependent_upper: &upper,
    };
    let result = PairedFrontOperations::machine_backward_transition(
        &incoming,
        &input,
        100.,
        5,
        FrontierBudget::default(),
    );
    let FrontierOutcome::Complete((function, bits, key)) = result else {
        panic!("unexpected backward waiting extension");
    };
    assert_eq!(function.pairs(), &[(10., 0.), (0., 8.)]);
    assert_eq!(bits, 1 << 1);
    assert_eq!(key, 90.);
}

#[test]
fn backward_transition_extends_waiting_before_bounded_front() {
    let incoming = SlopeFront::from_prepared(
        PreparedSlopeFunction::from_prepared_pairs(vec![(4., 2.), (0., 8.)]).unwrap(),
    );
    let make = |value, slope| {
        PreparedSlopeFunction::from_prepared_pairs(vec![(value, 0.), (slope, 8.)]).unwrap()
    };
    let mapping = make(0., 1.);
    let lower = make(0., 0.);
    let upper = make(100., 0.);
    let increment = make(0., 0.);
    let input = MachineBackwardInputs {
        waiting_lower: 2.,
        waiting_upper: 8.,
        forward_lower: 0.,
        mapped_upper: 8.,
        tail_lower: 0.,
        tail_upper: 8.,
        waiting_rate: 0,
        mapping: &mapping,
        inverse: &mapping,
        dependent: &increment,
        dependent_lower: &lower,
        dependent_upper: &upper,
    };
    let result = PairedFrontOperations::machine_backward_transition(
        &incoming,
        &input,
        100.,
        5,
        FrontierBudget::default(),
    );
    let FrontierOutcome::Complete((function, bits, key)) = result else {
        panic!("unexpected backward waiting prefix");
    };
    assert_eq!(function.pairs(), &[(4., 0.), (0., 2.), (0., 8.)]);
    assert_eq!(bits, 1 << 1);
    assert_eq!(key, 96.);
}

#[test]
fn backward_preflight_truncates_one_side_of_multi_segment_front() {
    let front = SlopeFront::from_prepared(
        PreparedSlopeFunction::from_prepared_pairs(vec![(2., 0.), (1., 2.), (-1., 4.), (2., 6.)])
            .unwrap(),
    );
    for (lower, upper, expected) in [
        (0., 5., vec![(2., 0.), (1., 2.), (-1., 4.), (2., 5.)]),
        (3., 6., vec![(3., 3.), (-1., 4.), (2., 6.)]),
    ] {
        let result = PairedFrontOperations::machine_backward_cover_preflight(
            &front,
            lower,
            upper,
            lower,
            upper,
            FrontierBudget::default(),
        );
        let FrontierOutcome::Complete((reduced, end)) = result else {
            panic!("unexpected multi-segment truncation");
        };
        assert_eq!(reduced.components()[0].pairs(), expected);
        assert_eq!(end, upper);
    }
}

#[test]
fn backward_preflight_truncates_either_side_of_a_segment() {
    let front = SlopeFront::from_prepared(
        PreparedSlopeFunction::from_prepared_pairs(vec![(2., 0.), (1., 10.)]).unwrap(),
    );
    for (lower, upper, expected) in [
        (0., 8., vec![(2., 0.), (1., 8.)]),
        (2., 10., vec![(4., 2.), (1., 10.)]),
    ] {
        let result = PairedFrontOperations::machine_backward_cover_preflight(
            &front,
            lower,
            upper,
            lower,
            upper,
            FrontierBudget::default(),
        );
        let FrontierOutcome::Complete((reduced, end)) = result else {
            panic!("unexpected one-sided truncation");
        };
        assert_eq!(reduced.components()[0].pairs(), expected);
        assert_eq!(end, upper);
    }
}

#[test]
fn backward_transition_handles_clipped_window_and_covering_bounds() {
    let make = |value, slope| {
        PreparedSlopeFunction::from_prepared_pairs(vec![(value, 0.), (slope, 10.)]).unwrap()
    };
    let incoming = SlopeFront::from_prepared(make(4., 0.));
    let mapping = make(0., 1.);
    let lower = make(0., 0.);
    let upper = make(100., 0.);
    let increment = make(0., 0.);
    let input = MachineBackwardInputs {
        waiting_lower: 2.,
        waiting_upper: 8.,
        forward_lower: 2.,
        mapped_upper: 8.,
        tail_lower: 2.,
        tail_upper: 8.,
        waiting_rate: 0,
        mapping: &mapping,
        inverse: &mapping,
        dependent: &increment,
        dependent_lower: &lower,
        dependent_upper: &upper,
    };
    let result = PairedFrontOperations::machine_backward_transition(
        &incoming,
        &input,
        100.,
        5,
        FrontierBudget::default(),
    );
    let FrontierOutcome::Complete((function, bits, key)) = result else {
        panic!("unexpected clipped backward transition");
    };
    assert_eq!(function.pairs(), &[(4., 2.), (0., 8.)]);
    assert_eq!(bits, 0);
    assert_eq!(key, 96.);
}

#[test]
fn backward_preflight_truncates_an_interior_waiting_window() {
    let front = SlopeFront::from_prepared(
        PreparedSlopeFunction::from_prepared_pairs(vec![(2., 0.), (1., 10.)]).unwrap(),
    );
    let result = PairedFrontOperations::machine_backward_cover_preflight(
        &front,
        2.,
        8.,
        2.,
        8.,
        FrontierBudget::default(),
    );
    let FrontierOutcome::Complete((front, end)) = result else {
        panic!("unexpected backward truncation");
    };
    assert_eq!(front.components()[0].pairs(), &[(4., 2.), (1., 8.)]);
    assert_eq!(end, 8.);
}

#[test]
fn backward_lower_filter_accepts_a_wider_constant_domain() {
    let front = SlopeFront::from_prepared(
        PreparedSlopeFunction::from_prepared_pairs(vec![(4., 2.), (1., 8.)]).unwrap(),
    );
    let lower = PreparedSlopeFunction::from_prepared_pairs(vec![(0., 0.), (0., 10.)]).unwrap();
    let result = PairedFrontOperations::machine_backward_constant_lower_pass(
        &front,
        &lower,
        FrontierBudget::default(),
    );
    let FrontierOutcome::Complete(filtered) = result else {
        panic!("unexpected lower filter result");
    };
    assert_eq!(filtered.components()[0].pairs(), &[(4., 2.), (1., 8.)]);
}

#[test]
fn backward_transition_accepts_reduced_multisegment_front_below_constant_bounds() {
    let function = |value, slope| {
        PreparedSlopeFunction::from_prepared_pairs(vec![(value, 0.), (slope, 2.)]).unwrap()
    };
    let incoming = SlopeFront::from_prepared(
        PreparedSlopeFunction::from_prepared_pairs(vec![(1., 0.), (2., 1.), (0., 2.)]).unwrap(),
    );
    let mapping = function(0., 1.);
    let lower = function(0., 0.);
    let upper = function(100., 0.);
    let increment = function(0., 0.);
    let input = MachineBackwardInputs {
        waiting_lower: 0.,
        waiting_upper: 2.,
        forward_lower: 0.,
        mapped_upper: 2.,
        tail_lower: 0.,
        tail_upper: 2.,
        waiting_rate: 2,
        mapping: &mapping,
        inverse: &mapping,
        dependent: &increment,
        dependent_lower: &lower,
        dependent_upper: &upper,
    };
    let result = PairedFrontOperations::machine_backward_transition(
        &incoming,
        &input,
        100.,
        5,
        FrontierBudget::default(),
    );
    let FrontierOutcome::Complete((function, bits, key)) = result else {
        panic!("unexpected multi-segment backward transition");
    };
    assert_eq!(function.pairs(), &[(1., 0.), (2., 1.), (0., 2.)]);
    assert_eq!(bits, 0);
    assert_eq!(key, 97.);
}

#[test]
fn backward_constant_lower_checks_every_multi_segment_endpoint() {
    let lower = PreparedSlopeFunction::from_prepared_pairs(vec![(0., 0.), (0., 3.)]).unwrap();
    let safe =
        PreparedSlopeFunction::from_prepared_pairs(vec![(3., 0.), (-1., 1.), (1., 2.), (-1., 3.)])
            .unwrap();
    let front = SlopeFront::from_prepared(safe.clone());
    let FrontierOutcome::Complete(result) =
        PairedFrontOperations::machine_backward_constant_lower_pass(
            &front,
            &lower,
            FrontierBudget::default(),
        )
    else {
        panic!("unexpected lower filter result");
    };
    assert_eq!(result.components()[0].pairs(), safe.pairs());
    let crossing =
        PreparedSlopeFunction::from_prepared_pairs(vec![(3., 0.), (-4., 1.), (4., 2.), (0., 3.)])
            .unwrap();
    assert!(matches!(
        PairedFrontOperations::machine_backward_constant_lower_pass(
            &SlopeFront::from_prepared(crossing),
            &lower,
            FrontierBudget::default(),
        ),
        FrontierOutcome::Unknown(_)
    ));
}

#[test]
fn backward_cover_preflight_reduces_multi_segment_fronts() {
    let function = PreparedSlopeFunction::from_prepared_pairs(vec![
        (1., 0.),
        (2., 1.),
        (2.00005, 2.),
        (0., 3.),
    ])
    .unwrap();
    let front = SlopeFront::from_prepared(function);
    let result = PairedFrontOperations::machine_backward_cover_preflight(
        &front,
        0.,
        3.,
        0.,
        3.,
        FrontierBudget::default(),
    );
    let FrontierOutcome::Complete((reduced, end)) = result else {
        panic!("unexpected multi-segment preflight");
    };
    assert_eq!(
        reduced.components()[0].pairs(),
        &[(1., 0.), (2., 2.), (0., 3.)]
    );
    assert_eq!(end, 3.);
}

#[test]
fn backward_wait_steep_suffix_can_dominate_the_full_window() {
    let front =
        PreparedSlopeFunction::from_prepared_pairs(vec![(0., 0.), (0., 1.), (2., 2.)]).unwrap();
    let result = PairedFrontOperations::machine_backward_wait_full_window(
        &front,
        0,
        FrontierBudget::default(),
    );
    let FrontierOutcome::Complete((function, bits)) = result else {
        panic!("unexpected steep-suffix waiting envelope");
    };
    assert_eq!(function.pairs(), &[(2., 0.), (0., 2.)]);
    assert_eq!(bits, 1 << 1);
}

#[test]
fn backward_wait_keeps_a_multi_segment_suffix_after_steep_prefix() {
    let front =
        PreparedSlopeFunction::from_prepared_pairs(vec![(1., 0.), (2., 1.), (0., 2.), (-1., 3.)])
            .unwrap();
    let result = PairedFrontOperations::machine_backward_wait_full_window(
        &front,
        0,
        FrontierBudget::default(),
    );
    let FrontierOutcome::Complete((function, bits)) = result else {
        panic!("unexpected multi-segment waiting envelope");
    };
    assert_eq!(function.pairs(), &[(3., 0.), (0., 1.), (0., 2.), (-1., 3.)]);
    assert_eq!(bits, 1 << 1);
}

#[test]
fn backward_wait_replaces_steep_prefix_and_preserves_later_segment() {
    let front =
        PreparedSlopeFunction::from_prepared_pairs(vec![(1., 0.), (2., 1.), (0., 2.)]).unwrap();
    let result = PairedFrontOperations::machine_backward_wait_full_window(
        &front,
        0,
        FrontierBudget::default(),
    );
    let FrontierOutcome::Complete((function, bits)) = result else {
        panic!("unexpected mixed-slope waiting envelope");
    };
    assert_eq!(function.pairs(), &[(3., 0.), (0., 1.), (0., 2.)]);
    assert_eq!(bits, 1 << 1);
}

#[test]
fn backward_wait_full_window_replaces_strictly_steep_segments() {
    let front =
        PreparedSlopeFunction::from_prepared_pairs(vec![(0., 0.), (2., 1.), (3., 2.)]).unwrap();
    let result = PairedFrontOperations::machine_backward_wait_full_window(
        &front,
        1,
        FrontierBudget::default(),
    );
    let FrontierOutcome::Complete((function, bits)) = result else {
        panic!("unexpected steep backward waiting envelope");
    };
    assert_eq!(function.pairs(), &[(3., 0.), (1., 2.)]);
    assert_eq!(bits, 1 << 1);
}

#[test]
fn backward_wait_full_window_walks_each_segment_in_reverse() {
    let front =
        PreparedSlopeFunction::from_prepared_pairs(vec![(0., 0.), (2., 1.5), (0., 2.)]).unwrap();
    let result = PairedFrontOperations::machine_backward_wait_full_window(
        &front,
        2,
        FrontierBudget::default(),
    );
    let FrontierOutcome::Complete((function, bits)) = result else {
        panic!("unexpected backward waiting envelope");
    };
    assert_eq!(function.pairs(), front.pairs());
    assert_eq!(bits, 0);
}

#[test]
fn backward_transition_carries_crossing_upper_through_full_window() {
    let function = |value, slope| {
        PreparedSlopeFunction::from_prepared_pairs(vec![(value, 0.), (slope, 2.)]).unwrap()
    };
    let incoming = SlopeFront::from_prepared(function(1., 2.));
    let mapping = function(0., 1.);
    let lower = function(0., 0.);
    let upper = function(3., 0.);
    let increment = function(0., 0.);
    let mut input = MachineBackwardInputs {
        waiting_lower: 0.,
        waiting_upper: 2.,
        forward_lower: 0.,
        mapped_upper: 2.,
        tail_lower: 0.,
        tail_upper: 2.,
        waiting_rate: 2,
        mapping: &mapping,
        inverse: &mapping,
        dependent: &increment,
        dependent_lower: &lower,
        dependent_upper: &upper,
    };
    let result = PairedFrontOperations::machine_backward_transition(
        &incoming,
        &input,
        10.,
        5,
        FrontierBudget::default(),
    );
    let FrontierOutcome::Complete((function, bits, key)) = result else {
        panic!("unexpected crossing backward transition");
    };
    assert_eq!(function.pairs(), &[(1., 0.), (2., 1.), (0., 2.)]);
    assert_eq!(bits, 0);
    assert_eq!(key, 7.);
    input.waiting_rate = 0;
    let result = PairedFrontOperations::machine_backward_transition(
        &incoming,
        &input,
        10.,
        5,
        FrontierBudget::default(),
    );
    let FrontierOutcome::Complete((function, bits, key)) = result else {
        panic!("unexpected mixed-slope backward transition");
    };
    assert_eq!(function.pairs(), &[(3., 0.), (0., 1.), (0., 2.)]);
    assert_eq!(bits, 1 << 1);
    assert_eq!(key, 7.);
}

#[test]
fn backward_minimum_splits_crossing_segments_at_reference_intersection() {
    let fixed = PreparedSlopeFunction::from_prepared_pairs(vec![(0., 0.), (2., 2.)]).unwrap();
    let upper = PreparedSlopeFunction::from_prepared_pairs(vec![(3., 0.), (0., 2.)]).unwrap();
    let result = PairedFrontOperations::machine_minimum_separated_segments(
        &fixed,
        &upper,
        FrontierBudget::default(),
    );
    let FrontierOutcome::Complete(function) = result else {
        panic!("unexpected crossing minimum");
    };
    assert_eq!(function.pairs(), &[(0., 0.), (2., 1.5), (0., 2.)]);
}

#[test]
fn backward_minimum_keeps_bit_identical_single_segments() {
    let function = PreparedSlopeFunction::from_prepared_pairs(vec![(100., 0.), (0., 2.)]).unwrap();
    let result = PairedFrontOperations::machine_minimum_separated_segments(
        &function,
        &function,
        FrontierBudget::default(),
    );
    let FrontierOutcome::Complete(value) = result else {
        panic!("unexpected identical backward minimum");
    };
    assert_eq!(value.pairs(), function.pairs());
}

#[test]
fn backward_transition_connects_bounds_waiting_composition_and_key() {
    let function = |value, slope| {
        PreparedSlopeFunction::from_prepared_pairs(vec![(value, 0.), (slope, 10.)]).unwrap()
    };
    let incoming = SlopeFront::from_prepared(function(10., 0.));
    let mapping = function(0., 1.);
    let lower = function(0., 0.);
    let upper = function(100., 0.);
    let increment = function(1., 0.);
    let input = MachineBackwardInputs {
        waiting_lower: 0.,
        waiting_upper: 10.,
        forward_lower: 0.,
        mapped_upper: 10.,
        tail_lower: 0.,
        tail_upper: 10.,
        waiting_rate: 0,
        mapping: &mapping,
        inverse: &mapping,
        dependent: &increment,
        dependent_lower: &lower,
        dependent_upper: &upper,
    };
    let result = PairedFrontOperations::machine_backward_transition(
        &incoming,
        &input,
        100.,
        5,
        FrontierBudget::default(),
    );
    let FrontierOutcome::Complete((function, bits, key)) = result else {
        panic!("unexpected backward transition");
    };
    assert_eq!(function.pairs(), &[(9., 0.), (0., 10.)]);
    assert_eq!(bits, 0);
    assert_eq!(key, 91.);
}

#[test]
fn backward_wait_finish_carries_waiting_through_composition_and_difference() {
    let upper = PreparedSlopeFunction::from_prepared_pairs(vec![(4., 0.), (2., 10.)]).unwrap();
    let mapping = PreparedSlopeFunction::from_prepared_pairs(vec![(0., 0.), (1., 10.)]).unwrap();
    let increment = PreparedSlopeFunction::from_prepared_pairs(vec![(1., 0.), (0., 10.)]).unwrap();
    let result = PairedFrontOperations::machine_backward_wait_finish(
        &upper,
        0,
        &mapping,
        &increment,
        0.,
        10.,
        30.,
        5,
        FrontierBudget::default(),
    );
    let FrontierOutcome::Complete((function, bits, key)) = result else {
        panic!("unexpected backward waiting finish");
    };
    assert_eq!(function.pairs(), &[(23., 0.), (0., 10.)]);
    assert_eq!(bits, 1 << 1);
    assert_eq!(key, 7.);
}

#[test]
fn backward_outer_composition_splits_when_waiting_bits_change() {
    let outer =
        PreparedSlopeFunction::from_prepared_pairs(vec![(1., 0.), (2., 2.), (2., 4.)]).unwrap();
    let inner = PreparedSlopeFunction::from_prepared_pairs(vec![(0., 0.), (1., 4.)]).unwrap();
    let result = PairedFrontOperations::machine_backward_compose_outer_stream(
        &outer,
        1 << 1,
        &inner,
        0.,
        4.,
        FrontierBudget::default(),
    );
    let FrontierOutcome::Complete((function, bits)) = result else {
        panic!("unexpected backward outer composition");
    };
    assert_eq!(function.pairs(), &[(1., 0.), (2., 2.), (2., 4.)]);
    assert_eq!(bits, 1 << 1);
}

#[test]
fn backward_composition_stream_preserves_segment_waiting_bits() {
    let outer = PreparedSlopeFunction::from_prepared_pairs(vec![(0., 0.), (2., 10.)]).unwrap();
    let inner =
        PreparedSlopeFunction::from_prepared_pairs(vec![(1., 0.), (1., 2.), (2., 4.)]).unwrap();
    let result = PairedFrontOperations::machine_backward_compose_stream(
        &outer,
        1 << 1,
        &inner,
        0.,
        4.,
        FrontierBudget::default(),
    );
    let FrontierOutcome::Complete((function, bits)) = result else {
        panic!("unexpected backward composition stream");
    };
    assert_eq!(function.pairs(), &[(2., 0.), (2., 2.), (4., 4.)]);
    assert_eq!(bits, (1 << 1) | (1 << 2));
}

#[test]
fn backward_single_outer_stream_accepts_interior_mapping_bounds() {
    let outer = PreparedSlopeFunction::from_prepared_pairs(vec![(0., 0.), (2., 10.)]).unwrap();
    let inner =
        PreparedSlopeFunction::from_prepared_pairs(vec![(1., 0.), (1., 2.), (2., 4.)]).unwrap();
    let result = PairedFrontOperations::machine_backward_compose_stream(
        &outer,
        1 << 1,
        &inner,
        1.,
        3.,
        FrontierBudget::default(),
    );
    let FrontierOutcome::Complete((function, bits)) = result else {
        panic!("unexpected bounded backward composition stream");
    };
    assert_eq!(function.pairs(), &[(4., 1.), (2., 2.), (4., 3.)]);
    assert_eq!(bits, (1 << 1) | (1 << 2));
}

#[test]
fn backward_composition_stream_accepts_exact_full_outer_domain() {
    let outer = PreparedSlopeFunction::from_prepared_pairs(vec![(4., 0.), (2., 4.)]).unwrap();
    let inner =
        PreparedSlopeFunction::from_prepared_pairs(vec![(0., 0.), (1., 2.), (1., 4.)]).unwrap();
    let result = PairedFrontOperations::machine_backward_compose_stream(
        &outer,
        1 << 1,
        &inner,
        0.,
        4.,
        FrontierBudget::default(),
    );
    let FrontierOutcome::Complete((function, bits)) = result else {
        panic!("unexpected backward composition stream");
    };
    assert_eq!(function.pairs(), &[(4., 0.), (2., 4.)]);
    // The merged candidate leaves its scratch waiting bit set in the reference.
    assert_eq!(bits, (1 << 1) | (1 << 2));
}

#[test]
fn backward_finish_composes_subtracts_and_computes_the_key() {
    let envelope = PreparedSlopeFunction::from_prepared_pairs(vec![(4., 0.), (2., 10.)]).unwrap();
    let mapping = PreparedSlopeFunction::from_prepared_pairs(vec![(0., 0.), (1., 10.)]).unwrap();
    let increment = PreparedSlopeFunction::from_prepared_pairs(vec![(1., 0.), (0., 10.)]).unwrap();
    let result = PairedFrontOperations::machine_backward_finish(
        &envelope,
        1 << 1,
        &mapping,
        &increment,
        0.,
        10.,
        30.,
        5,
        FrontierBudget::default(),
    );
    let FrontierOutcome::Complete((function, bits, key)) = result else {
        panic!("unexpected backward finish");
    };
    assert_eq!(function.pairs(), &[(3., 0.), (2., 10.)]);
    assert_eq!(bits, 1 << 1);
    assert_eq!(key, 7.);
}

#[test]
fn backward_composition_uses_secant_and_carries_waiting_bit() {
    let outer = PreparedSlopeFunction::from_prepared_pairs(vec![(0., 0.), (3., 20.)]).unwrap();
    let inner = PreparedSlopeFunction::from_prepared_pairs(vec![(0., 0.), (1., 10.)]).unwrap();
    let result = PairedFrontOperations::machine_backward_compose_single_segment(
        &outer,
        1 << 1,
        &inner,
        2.,
        4.,
        FrontierBudget::default(),
    );
    let FrontierOutcome::Complete((function, bits)) = result else {
        panic!("unexpected backward composition");
    };
    assert_eq!(function.pairs(), &[(6., 2.), (3., 4.)]);
    assert_eq!(bits, 1 << 1);
}

#[test]
fn backward_composition_accepts_full_identity_domain() {
    let outer = PreparedSlopeFunction::from_prepared_pairs(vec![(4., 0.), (2., 10.)]).unwrap();
    let inner = PreparedSlopeFunction::from_prepared_pairs(vec![(0., 0.), (1., 10.)]).unwrap();
    let result = PairedFrontOperations::machine_backward_compose_single_segment(
        &outer,
        1 << 1,
        &inner,
        0.,
        10.,
        FrontierBudget::default(),
    );
    let FrontierOutcome::Complete((function, bits)) = result else {
        panic!("unexpected backward composition");
    };
    assert_eq!(function.pairs(), &[(4., 0.), (2., 10.)]);
    assert_eq!(bits, 1 << 1);
}

#[test]
fn backward_aligned_difference_merges_only_equal_waiting_segments() {
    let first =
        PreparedSlopeFunction::from_prepared_pairs(vec![(10., 0.), (3., 1.), (4., 2.), (4., 3.)])
            .unwrap();
    let second =
        PreparedSlopeFunction::from_prepared_pairs(vec![(2., 0.), (1., 1.), (2., 2.), (2., 3.)])
            .unwrap();
    let result = PairedFrontOperations::machine_backward_difference(
        &first,
        (1 << 1) | (1 << 2),
        &second,
        FrontierBudget::default(),
    );
    let FrontierOutcome::Complete((function, bits)) = result else {
        panic!("unexpected backward difference");
    };
    assert_eq!(function.pairs(), &[(8., 0.), (2., 2.), (2., 3.)]);
    assert_eq!(bits, 1 << 1);
}

#[test]
fn backward_difference_splits_at_both_inputs_and_carries_first_wait_bits() {
    let first =
        PreparedSlopeFunction::from_prepared_pairs(vec![(10., 0.), (3., 2.), (4., 4.)]).unwrap();
    let second =
        PreparedSlopeFunction::from_prepared_pairs(vec![(2., 0.), (1., 1.), (2., 3.), (1., 4.)])
            .unwrap();
    let result = PairedFrontOperations::machine_backward_difference(
        &first,
        1 << 1,
        &second,
        FrontierBudget::default(),
    );
    let FrontierOutcome::Complete((function, bits)) = result else {
        panic!("unexpected backward difference");
    };
    assert_eq!(
        function.pairs(),
        &[(8., 0.), (2., 1.), (1., 2.), (2., 3.), (3., 4.)]
    );
    assert_eq!(bits, (1 << 1) | (1 << 2));
}

#[test]
fn backward_difference_stops_at_the_first_exhausted_input() {
    let first = PreparedSlopeFunction::from_prepared_pairs(vec![(5., 0.), (2., 4.)]).unwrap();
    let second = PreparedSlopeFunction::from_prepared_pairs(vec![(1., 0.), (1., 2.)]).unwrap();
    let result = PairedFrontOperations::machine_backward_difference(
        &first,
        1 << 1,
        &second,
        FrontierBudget::default(),
    );
    let FrontierOutcome::Complete((function, bits)) = result else {
        panic!("unexpected backward difference");
    };
    assert_eq!(function.pairs(), &[(4., 0.), (1., 2.)]);
    assert_eq!(bits, 1 << 1);
}

#[test]
fn backward_difference_reanchors_at_later_domain_start() {
    let first = PreparedSlopeFunction::from_prepared_pairs(vec![(5., 0.), (2., 3.)]).unwrap();
    let second = PreparedSlopeFunction::from_prepared_pairs(vec![(2., 1.), (1., 2.)]).unwrap();
    let result = PairedFrontOperations::machine_backward_difference(
        &first,
        1 << 1,
        &second,
        FrontierBudget::default(),
    );
    let FrontierOutcome::Complete((function, bits)) = result else {
        panic!("unexpected backward difference");
    };
    assert_eq!(function.pairs(), &[(5., 1.), (1., 2.)]);
    assert_eq!(bits, 1 << 1);
}

#[test]
fn backward_difference_uses_first_coordinates_for_near_matches() {
    let first = PreparedSlopeFunction::from_prepared_pairs(vec![(5., 0.), (2., 2.)]).unwrap();
    let second =
        PreparedSlopeFunction::from_prepared_pairs(vec![(2., 0.00005), (1., 2.00005)]).unwrap();
    let result = PairedFrontOperations::machine_backward_difference(
        &first,
        1 << 1,
        &second,
        FrontierBudget::default(),
    );
    let FrontierOutcome::Complete((function, bits)) = result else {
        panic!("unexpected backward difference");
    };
    assert_eq!(function.pairs(), &[(3., 0.), (1., 2.)]);
    assert_eq!(bits, 1 << 1);
}

#[test]
fn backward_difference_advances_segment_for_start_near_breakpoint() {
    let first =
        PreparedSlopeFunction::from_prepared_pairs(vec![(5., 0.), (2., 1.), (4., 3.)]).unwrap();
    let start = 0.99995;
    let second = PreparedSlopeFunction::from_prepared_pairs(vec![(2., start), (1., 2.)]).unwrap();
    let result = PairedFrontOperations::machine_backward_difference(
        &first,
        1 << 2,
        &second,
        FrontierBudget::default(),
    );
    let FrontierOutcome::Complete((function, bits)) = result else {
        panic!("unexpected backward difference");
    };
    assert_eq!(function.pairs(), &[(5. + 2. * start - 2., start), (3., 2.)]);
    assert_eq!(bits, 1 << 1);
}

#[test]
fn backward_difference_accepts_point_front_with_near_start() {
    let first = PreparedSlopeFunction::from_prepared_pairs(vec![(5., 0.)]).unwrap();
    let second = PreparedSlopeFunction::from_prepared_pairs(vec![(2., 0.00005), (1., 2.)]).unwrap();
    let result = PairedFrontOperations::machine_backward_difference(
        &first,
        0,
        &second,
        FrontierBudget::default(),
    );
    let FrontierOutcome::Complete((function, bits)) = result else {
        panic!("unexpected backward difference");
    };
    assert_eq!(function.pairs(), &[(3., 0.)]);
    assert_eq!(bits, 0);
}

#[test]
fn backward_initial_bounds_apply_separated_upper_minimum() {
    let front = SlopeFront::from_prepared(
        PreparedSlopeFunction::from_prepared_pairs(vec![(3., 0.), (0., 2.)]).unwrap(),
    );
    let lower = PreparedSlopeFunction::from_prepared_pairs(vec![(0., 0.), (0., 2.)]).unwrap();
    let upper = PreparedSlopeFunction::from_prepared_pairs(vec![(2., 0.), (0., 2.)]).unwrap();
    match PairedFrontOperations::machine_backward_initial_bounds(
        &front,
        0.,
        2.,
        0.,
        2.,
        &lower,
        &upper,
        FrontierBudget::default(),
    ) {
        FrontierOutcome::Complete((result, limit)) => {
            assert_eq!(result.pairs(), upper.pairs());
            assert_eq!(limit, 2.);
        }
        _ => panic!("separated upper bound should produce the minimum"),
    }
}

#[test]
fn backward_initial_filter_combines_cover_and_lower_bound() {
    let front = SlopeFront::from_prepared(
        PreparedSlopeFunction::from_prepared_pairs(vec![(3., 0.), (-1., 2.)]).unwrap(),
    );
    let lower = PreparedSlopeFunction::from_prepared_pairs(vec![(0., 0.), (0., 2.)]).unwrap();
    match PairedFrontOperations::machine_backward_initial_filter(
        &front,
        0.,
        2.,
        0.,
        2.,
        &lower,
        FrontierBudget::default(),
    ) {
        FrontierOutcome::Complete((result, upper)) => {
            assert_eq!(
                result.components()[0].pairs(),
                front.components()[0].pairs()
            );
            assert_eq!(upper, 2.);
        }
        _ => panic!("covered front above its lower bound should pass"),
    }
}

#[test]
fn backward_lower_filter_retains_the_region_above_a_crossing_bound() {
    let function = PreparedSlopeFunction::from_prepared_pairs(vec![(3., 0.), (-1., 2.)]).unwrap();
    let front = SlopeFront::from_prepared(function.clone());
    let lower = PreparedSlopeFunction::from_prepared_pairs(vec![(0., 0.), (0., 2.)]).unwrap();
    match PairedFrontOperations::machine_backward_constant_lower_pass(
        &front,
        &lower,
        FrontierBudget::default(),
    ) {
        FrontierOutcome::Complete(result) => {
            assert_eq!(result.components()[0].pairs(), function.pairs())
        }
        _ => panic!("strictly lower constant should preserve the front"),
    }
    let crossing = PreparedSlopeFunction::from_prepared_pairs(vec![(2., 0.), (0., 2.)]).unwrap();
    let FrontierOutcome::Complete(clipped) =
        PairedFrontOperations::machine_backward_constant_lower_pass(
            &front,
            &crossing,
            FrontierBudget::default(),
        )
    else {
        panic!("connected retained region should be supported");
    };
    assert_eq!(clipped.components()[0].pairs(), &[(3., 0.), (-1., 1.)]);
}

#[test]
fn backward_cover_preflight_retains_front_and_caps_end() {
    let front = SlopeFront::from_prepared(
        PreparedSlopeFunction::from_prepared_pairs(vec![(2., 1.), (0., 3.)]).unwrap(),
    );
    let result = PairedFrontOperations::machine_backward_cover_preflight(
        &front,
        0.,
        4.,
        0.5,
        2.5,
        FrontierBudget::default(),
    );
    match result {
        FrontierOutcome::Complete((reduced, upper)) => {
            assert_eq!(reduced.components()[0].pairs(), &[(2., 1.), (0., 3.)]);
            assert_eq!(upper, 2.5);
        }
        _ => panic!("covered one-segment front should pass preflight"),
    }
    assert!(matches!(
        PairedFrontOperations::machine_backward_cover_preflight(
            &front,
            2.,
            4.,
            0.5,
            2.5,
            FrontierBudget::default(),
        ),
        FrontierOutcome::Complete(_)
    ));
}

#[test]
fn mode_four_aligned_meeting_handles_initial_overlap_and_crossing() {
    let forward = PreparedSlopeFunction::from_prepared_pairs(vec![(2., 0.), (-1., 2.)]).unwrap();
    let backward = PreparedSlopeFunction::from_prepared_pairs(vec![(0., 0.), (1., 2.)]).unwrap();
    match PairedFrontOperations::machine_mode_four_single_segment_meeting(
        &forward,
        &backward,
        FrontierBudget::default(),
    ) {
        FrontierOutcome::Complete(coordinate) => assert_eq!(coordinate, 1.),
        _ => panic!("aligned crossing should yield a coordinate"),
    }
    match PairedFrontOperations::machine_mode_four_single_segment_meeting(
        &backward,
        &forward,
        FrontierBudget::default(),
    ) {
        FrontierOutcome::Complete(coordinate) => assert_eq!(coordinate, 0.),
        _ => panic!("initial overlap should yield the starting coordinate"),
    }
    let later = PreparedSlopeFunction::from_prepared_pairs(vec![(0., 0.5), (1., 2.)]).unwrap();
    match PairedFrontOperations::machine_mode_four_single_segment_meeting(
        &forward,
        &later,
        FrontierBudget::default(),
    ) {
        FrontierOutcome::Complete(coordinate) => assert_eq!(coordinate, 1.25),
        _ => panic!("different starts should select the crossing"),
    }
}

#[test]
fn machine_minimum_preserves_selected_segments_and_splits_crossings() {
    let a = PreparedSlopeFunction::from_prepared_pairs(vec![(1., 0.), (2., 2.)]).unwrap();
    let b = PreparedSlopeFunction::from_prepared_pairs(vec![(4., 0.), (1., 2.)]).unwrap();
    let minimum = PairedFrontOperations::machine_minimum_separated_segments(
        &a,
        &b,
        FrontierBudget::default(),
    );
    match minimum {
        FrontierOutcome::Complete(front) => assert_eq!(front.pairs(), a.pairs()),
        _ => panic!("separated segments should produce a minimum"),
    }
    let crossing = PreparedSlopeFunction::from_prepared_pairs(vec![(4., 0.), (-1., 2.)]).unwrap();
    let FrontierOutcome::Complete(split) =
        PairedFrontOperations::machine_minimum_separated_segments(
            &a,
            &crossing,
            FrontierBudget::default(),
        )
    else {
        panic!("crossing machine minimum should be supported");
    };
    assert_eq!(split.pairs(), &[(1., 0.), (2., 1.), (-1., 2.)]);
    let short_a = PreparedSlopeFunction::from_prepared_pairs(vec![(1., 0.), (0., 1e-12)]).unwrap();
    let short_b = PreparedSlopeFunction::from_prepared_pairs(vec![(4., 0.), (0., 1e-12)]).unwrap();
    assert!(matches!(
        PairedFrontOperations::machine_minimum_separated_segments(
            &short_a,
            &short_b,
            FrontierBudget::default(),
        ),
        FrontierOutcome::Unknown(_)
    ));
}
fn function(p: &[(f64, f64)]) -> PreparedSlopeFunction {
    PreparedSlopeFunction::from_prepared_pairs(p.to_vec()).unwrap()
}
#[test]
fn machine_sum_and_difference_match_recorded_reference_merges() {
    // Reproduce: python3 internal/reference/metasolver_paired_operators.py.
    // Reference sum ELF 71e840 and difference ELF 727ce0, pinned SHA therein.
    let a = function(&[(2., 0.), (1., 1.), (2., 3.)]);
    let b = function(&[(3., 0.), (4., 2.), (5., 3.)]);
    let FrontierOutcome::Complete(sum) =
        PairedFrontOperations::machine_sum_streams(&a, &b, FrontierBudget::default())
    else {
        panic!("multi-segment machine sum");
    };
    assert_eq!(sum.pairs(), &[(5., 0.), (5., 1.), (6., 2.), (7., 3.)]);
    // Independent dyadic point oracle; these exact segments need no near merge.
    for x in [0., 0.5, 1., 1.5, 2., 2.5, 3.] {
        let expected = a
            .evaluate(x, EvaluationPolicy::StrictDomain)
            .unwrap()
            .unwrap()
            + b.evaluate(x, EvaluationPolicy::StrictDomain)
                .unwrap()
                .unwrap();
        assert_eq!(
            sum.evaluate(x, EvaluationPolicy::StrictDomain).unwrap(),
            Some(expected)
        );
    }
    let near = function(&[(2., 0.), (1., 1.), (1.00001, 3.)]);
    let b = function(&[(3., 0.), (4., 3.)]);
    let FrontierOutcome::Complete((difference, bits)) =
        PairedFrontOperations::machine_backward_difference(&near, 6, &b, FrontierBudget::default())
    else {
        panic!("machine difference merge");
    };
    assert_eq!(difference.pairs(), &[(-1., 0.), (-3., 3.)]);
    assert_eq!(bits, 6); // Includes the reference's scratch bit after merging.
    let point = PreparedSlopeFunction::from_machine_pairs(vec![(2., 0.), (1., 0.)]).unwrap();
    let FrontierOutcome::Complete(sum) =
        PairedFrontOperations::machine_sum_streams(&point, &b, FrontierBudget::default())
    else {
        panic!("retained zero-width machine sum");
    };
    assert_eq!(sum.pairs(), &[(5., 0.), (5., 0.)]);
}

#[test]
fn backward_pipeline_consumes_unordered_intermediate_without_certifying_source_value() {
    let incoming = SlopeFront::from_prepared(function(&[(0., 0.), (0., 1.), (1.5, 2.)]));
    let mapping = function(&[(0., 0.), (1., 4.)]);
    let zero = function(&[(0., 0.), (0., 4.)]);
    let upper = function(&[(100., 0.), (0., 4.)]);
    let input = MachineBackwardInputs {
        waiting_lower: 0.,
        waiting_upper: 2.,
        forward_lower: 0.,
        mapped_upper: 2.,
        tail_lower: 0.,
        tail_upper: 4.,
        waiting_rate: 1,
        mapping: &mapping,
        inverse: &mapping,
        dependent: &zero,
        dependent_lower: &zero,
        dependent_upper: &upper,
    };
    let FrontierOutcome::Complete((result, bits, key)) =
        PairedFrontOperations::machine_backward_transition(
            &incoming,
            &input,
            100.,
            5,
            FrontierBudget::default(),
        )
    else {
        panic!("reference backward pipeline should finish");
    };
    // Reference modern extension ELF 7281e0, directly observed by the probe.
    assert_eq!(result.pairs(), &[(1.5, 0.), (0., 0.5), (1., 2.)]);
    assert_eq!(bits, 4);
    assert_eq!(key, 97.);
    // An independent source adjoint at zero maximizes F(t)-t at its knots.
    // The binary's copied value anchor differs from that source operation.
    let source_value = [0., 1., 2.]
        .into_iter()
        .map(|x| {
            incoming.components()[0]
                .evaluate(x, EvaluationPolicy::StrictDomain)
                .unwrap()
                .unwrap()
                - x
        })
        .fold(f64::NEG_INFINITY, f64::max);
    assert_eq!(source_value, 0.);
    assert_ne!(result.pairs()[0].0, source_value);
}

#[test]
fn waiting_composition_and_difference_retain_nan_candidates_for_every_waiting_mask() {
    let outer =
        PreparedSlopeFunction::from_machine_pairs(vec![(1., 0.), (0., 1.), (0., 1.)]).unwrap();
    let inner = PreparedSlopeFunction::from_machine_pairs(vec![(0., 0.), (2., 2.)]).unwrap();
    let zero = PreparedSlopeFunction::from_machine_pairs(vec![(0., 0.), (0., 2.)]).unwrap();
    for mask in 0..8 {
        let FrontierOutcome::Complete((composed, bits)) =
            PairedFrontOperations::machine_backward_compose_streams(
                &outer,
                mask,
                &inner,
                0.,
                0.5,
                FrontierBudget::default(),
            )
        else {
            panic!("native composition retains the empty NaN candidate for mask {mask}");
        };
        assert_eq!(bits, mask & 6);
        assert_eq!(&composed.pairs()[..2], &[(1., 0.), (0., 0.5)]);
        assert_eq!(composed.pairs().len(), 3);
        assert_eq!(composed.pairs()[2].0.to_bits(), 0xfff8_0000_0000_0000);
        assert_eq!(composed.pairs()[2].1, 0.5);
        let FrontierOutcome::Complete((difference, bits)) =
            PairedFrontOperations::machine_backward_difference(
                &composed,
                bits,
                &zero,
                FrontierBudget::default(),
            )
        else {
            panic!("native difference retains the empty NaN candidate for mask {mask}");
        };
        assert_eq!(bits, mask & 6);
        assert_eq!(difference, composed);
    }
}

#[test]
fn retained_nan_front_survives_preflight_filters_and_reverse_waiting_matches_the_machine() {
    let nan = f64::from_bits(0xfff8_0000_0000_0000);
    let raw =
        PreparedSlopeFunction::from_machine_output_pairs(vec![(0., 0.), (-1., 0.5), (nan, 0.5)])
            .unwrap();
    let front = SlopeFront::from_prepared(raw.clone());
    let FrontierOutcome::Complete(reduced) =
        front.machine_reduce_for_continuation(FrontierBudget::default())
    else {
        panic!("native reduction retains the NaN slope");
    };
    assert_eq!(reduced.components()[0], raw);
    let lower = PreparedSlopeFunction::from_machine_pairs(vec![(-100., 0.), (0., 2.)]).unwrap();
    let upper = PreparedSlopeFunction::from_machine_pairs(vec![(100., 0.), (0., 2.)]).unwrap();
    let FrontierOutcome::Complete(filtered) = PairedFrontOperations::machine_backward_lower_streams(
        &front,
        &lower,
        FrontierBudget::default(),
    ) else {
        panic!("native lower filter retains the NaN slope");
    };
    assert_eq!(filtered.components()[0], raw);
    let FrontierOutcome::Complete(bounded) =
        PairedFrontOperations::machine_minimum_streams(&raw, &upper, FrontierBudget::default())
    else {
        panic!("native minimum retains the NaN slope");
    };
    assert_eq!(bounded, raw);
    let FrontierOutcome::Complete((waiting, bits)) =
        PairedFrontOperations::machine_backward_wait_window(
            &raw,
            0,
            0.,
            0.5,
            FrontierBudget::default(),
        )
    else {
        panic!("native reverse waiting consumes the NaN slope");
    };
    assert_eq!(waiting.pairs(), &[(-0.5, 0.)]);
    assert_eq!(bits, 0);
}

#[test]
fn waiting_composition_retains_an_initial_nan_segment_before_later_finite_segments() {
    let outer =
        PreparedSlopeFunction::from_machine_pairs(vec![(1., 0.), (0., 0.), (0., 1.)]).unwrap();
    let inner = PreparedSlopeFunction::from_machine_pairs(vec![(0., 0.), (1., 1.)]).unwrap();
    let FrontierOutcome::Complete((composed, bits)) =
        PairedFrontOperations::machine_backward_compose_streams(
            &outer,
            0,
            &inner,
            0.,
            1.,
            FrontierBudget::default(),
        )
    else {
        panic!("native composition retains the initial NaN slope");
    };
    assert_eq!(composed.pairs().len(), 3);
    assert_eq!(composed.pairs()[0], (1., 0.));
    assert_eq!(composed.pairs()[1].0.to_bits(), 0xfff8_0000_0000_0000);
    assert_eq!(composed.pairs()[1].1, 0.);
    assert_eq!(composed.pairs()[2], (0., 1.));
    assert_eq!(bits, 0);
}

#[test]
fn nan_lower_filter_uses_ordered_reentry_and_empty_count_is_explicit() {
    let nan = f64::from_bits(0xfff8_0000_0000_0000);
    let raw =
        PreparedSlopeFunction::from_machine_output_pairs(vec![(0., 0.), (-1., 0.5), (nan, 0.5)])
            .unwrap();
    let front = SlopeFront::from_prepared(raw.clone());
    for bound in [-0.5, -0.25, 0., 0.25, 1.] {
        let lower = PreparedSlopeFunction::from_machine_pairs(vec![(bound, 0.), (0., 2.)]).unwrap();
        let result = PairedFrontOperations::machine_backward_lower_streams(
            &front,
            &lower,
            FrontierBudget::default(),
        );
        if bound > 0. {
            assert!(matches!(result, FrontierOutcome::Empty));
            continue;
        }
        let FrontierOutcome::Complete(filtered) = result else {
            panic!("native status zero preserves/truncates the front at bound {bound}");
        };
        if bound == -0.5 {
            assert_eq!(filtered.components()[0], raw);
        } else if bound == -0.25 {
            assert_eq!(filtered.components()[0].pairs(), &[(0., 0.), (-1., 0.25)]);
        } else {
            assert_eq!(filtered.components()[0].pairs(), &[(0., 0.), (0., 0.)]);
        }
    }
}

#[test]
fn backward_caller_continues_with_the_cleared_array_after_a_successful_empty_lower_filter() {
    let nan = f64::from_bits(0xfff8_0000_0000_0000);
    let raw =
        PreparedSlopeFunction::from_machine_output_pairs(vec![(0., 0.), (-1., 0.5), (nan, 0.5)])
            .unwrap();
    let front = SlopeFront::from_prepared(raw);
    let lower = PreparedSlopeFunction::from_machine_pairs(vec![(0.25, 0.), (0., 2.)]).unwrap();
    let upper = PreparedSlopeFunction::from_machine_pairs(vec![(100., 0.), (0., 2.)]).unwrap();
    let FrontierOutcome::Complete((bounded, limit)) =
        PairedFrontOperations::machine_backward_initial_bounds(
            &front,
            0.,
            2.,
            0.,
            2.,
            &lower,
            &upper,
            FrontierBudget::default(),
        )
    else {
        panic!("native minimum consumes the cleared count-minus-one storage");
    };
    assert_eq!(bounded.pairs(), &[(0., 0.), (0., 0.)]);
    assert_eq!(limit, 0.5);
}
