use super::*;

#[cfg(feature = "clp")]
#[test]
fn joined_label_cost_order_and_ng_rejection_match_recorded_binary_outputs() {
    let cases: serde_json::Value = serde_json::from_str(include_str!(
        "../../../../fixtures/coupled/metasolver-label-join-operators.json"
    ))
    .unwrap();
    let forward = PreparedSlopeFunction::from_machine_pairs(vec![(5., 0.), (0., 2.)]).unwrap();
    let backward = PreparedSlopeFunction::from_machine_pairs(vec![(98., 0.), (0., 2.)]).unwrap();
    for case in cases.as_array().unwrap() {
        let FrontierOutcome::Complete(selection) = PairedFrontOperations::machine_concatenation(
            &forward,
            &backward,
            100.,
            case["mode"].as_i64().unwrap() as i32,
            FrontierBudget::default(),
        ) else {
            panic!("recorded paired selection should succeed: {case}");
        };
        let FrontierOutcome::Complete(joined) = selection.join_labels(
            case["forward_cost"].as_f64().unwrap(),
            case["backward_cost"].as_f64().unwrap(),
            case["additional_cost"].as_f64().unwrap(),
            1,
            if case["overlap"].as_bool().unwrap() {
                1
            } else {
                2
            },
        ) else {
            panic!("recorded label join should have finite outputs: {case}");
        };
        assert_eq!(joined.compatible, case["accepted"].as_bool().unwrap());
        assert_eq!(
            joined.dependent_cost,
            case["dependent_cost"].as_f64().unwrap()
        );
        let expected = u64::from_str_radix(case["cost_bits"].as_str().unwrap(), 16).unwrap();
        assert_eq!(joined.reduced_cost.to_bits(), expected, "{case}");
    }
}

#[test]
fn label_join_cost_preserves_raw_ieee_nonfinite_outputs() {
    let selection = MachineConcatenation {
        coordinate: 0.,
        dependent: Some(7.),
    };
    for costs in [
        (f64::MAX, f64::MAX, 0.),
        (0., 0., f64::INFINITY),
        (f64::NAN, 0., 0.),
    ] {
        let FrontierOutcome::Complete(joined) =
            selection.join_labels(costs.0, costs.1, costs.2, 0, 0)
        else {
            panic!("raw label arithmetic should complete");
        };
        let expected = (costs.0 + costs.1) + (7. + costs.2);
        assert!(joined.compatible);
        assert_eq!(joined.dependent_cost, 7.);
        if expected.is_nan() {
            assert!(joined.reduced_cost.is_nan());
        } else {
            assert_eq!(joined.reduced_cost, expected);
        }
    }
}

#[cfg(feature = "clp")]
#[test]
fn paired_completion_joins_match_registered_reference_operands() {
    let capture: serde_json::Value = serde_json::from_str(include_str!(
        "../../../../fixtures/coupled/metasolver-completion-joins.json"
    ))
    .unwrap();
    assert_eq!(
        capture["reference_sha256"],
        "103caf4a9c58d6141e935848294bff16677f400323466dc23cd4b9e5c15889d6"
    );
    let cases = capture["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 46);
    for case in cases {
        let make = |side: &str| {
            let pairs = case[side]["front"]
                .as_array()
                .unwrap()
                .iter()
                .map(|pair| (pair[0].as_f64().unwrap(), pair[1].as_f64().unwrap()))
                .collect();
            PreparedSlopeFunction::from_machine_pairs(pairs).unwrap()
        };
        let left = make("left");
        let right = make("right");
        let FrontierOutcome::Complete(selection) = PairedFrontOperations::machine_concatenation(
            &left,
            &right,
            case["ceiling"].as_f64().unwrap(),
            case["mode"].as_i64().unwrap() as i32,
            FrontierBudget::default(),
        ) else {
            panic!("native join did not reconstruct: {case}")
        };
        let mask = |side: &str| {
            u64::from_str_radix(
                case[side]["mask"]
                    .as_str()
                    .unwrap()
                    .trim_start_matches("0x"),
                16,
            )
            .unwrap()
        };
        let FrontierOutcome::Complete(joined) = selection.join_labels(
            case["left"]["scalar_cost"].as_f64().unwrap(),
            case["right"]["scalar_cost"].as_f64().unwrap(),
            0.,
            mask("left"),
            mask("right"),
        ) else {
            panic!("native label join did not reconstruct: {case}")
        };
        assert_eq!(
            joined.compatible,
            case["returned"].as_bool().unwrap(),
            "{case}"
        );
        assert!(
            (joined.dependent_cost - case["correction"].as_f64().unwrap()).abs() < 1e-12,
            "{case}"
        );
        assert!(
            (joined.reduced_cost - case["cost"].as_f64().unwrap()).abs() < 1e-12,
            "{case}"
        );
    }
}

#[cfg(feature = "clp")]
#[test]
fn completion_join_cutoff_matches_native_equality_and_adjacent_values() {
    let fixture: serde_json::Value = serde_json::from_str(include_str!(
        "../../../../fixtures/coupled/metasolver-completion-cutoff.json"
    ))
    .unwrap();
    let make = |name: &str| {
        PreparedSlopeFunction::from_machine_pairs(
            fixture[name]
                .as_array()
                .unwrap()
                .iter()
                .map(|p| (p[0].as_f64().unwrap(), p[1].as_f64().unwrap()))
                .collect(),
        )
        .unwrap()
    };
    let FrontierOutcome::Complete(selection) = PairedFrontOperations::machine_concatenation(
        &make("forward"),
        &make("backward"),
        fixture["ceiling"].as_f64().unwrap(),
        fixture["mode"].as_i64().unwrap() as i32,
        FrontierBudget::default(),
    ) else {
        panic!("expected a finite native-compatible join")
    };
    let FrontierOutcome::Complete(joined) = selection.join_labels(
        fixture["forward_cost"].as_f64().unwrap(),
        fixture["backward_cost"].as_f64().unwrap(),
        0.,
        1,
        2,
    ) else {
        panic!("expected a finite joined label")
    };
    assert!(joined.compatible);
    assert_eq!(
        joined.reduced_cost,
        fixture["joined_cost"].as_f64().unwrap()
    );
    for case in fixture["cases"].as_array().unwrap() {
        assert_eq!(
            joined.reduced_cost < case["cutoff"].as_f64().unwrap(),
            case["accepted"].as_bool().unwrap()
        );
    }
}

#[test]
fn retained_empty_nan_tail_does_not_discard_a_finite_machine_join() {
    let nan = f64::from_bits(0xfff8_0000_0000_0000);
    let forward = PreparedSlopeFunction::from_machine_pairs(vec![(0., 0.), (0., 2.)]).unwrap();
    let backward =
        PreparedSlopeFunction::from_machine_output_pairs(vec![(0., 0.), (-1., 0.5), (nan, 0.5)])
            .unwrap();
    for mode in [4, 5, 6] {
        let FrontierOutcome::Complete(joined) = PairedFrontOperations::machine_concatenation(
            &forward,
            &backward,
            3.,
            mode,
            FrontierBudget::default(),
        ) else {
            panic!("retained native NaN tail must preserve the earlier feasible join");
        };
        assert_eq!(joined.coordinate, 0.);
        assert_eq!(joined.dependent, (mode != 4).then_some(3.));
    }
}
