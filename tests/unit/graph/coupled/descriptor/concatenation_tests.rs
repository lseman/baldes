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
        let FrontierOutcome::Complete(selection) = PreparedSlopeRelationArc::machine_concatenation(
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
            if case["overlap"].as_bool().unwrap() { 1 } else { 2 },
        ) else {
            panic!("recorded label join should have finite outputs: {case}");
        };
        assert_eq!(joined.compatible, case["accepted"].as_bool().unwrap());
        assert_eq!(joined.dependent_cost, case["dependent_cost"].as_f64().unwrap());
        let expected = u64::from_str_radix(case["cost_bits"].as_str().unwrap(), 16).unwrap();
        assert_eq!(joined.reduced_cost.to_bits(), expected, "{case}");
    }
}

#[test]
fn label_join_cost_keeps_nonfinite_arithmetic_unknown() {
    let selection = MachineConcatenation { coordinate: 0., dependent: Some(7.) };
    for costs in [(f64::MAX, f64::MAX, 0.), (0., 0., f64::INFINITY),
                  (f64::NAN, 0., 0.)] {
        assert!(matches!(selection.join_labels(costs.0, costs.1, costs.2, 0, 0),
                         FrontierOutcome::Unknown(FrontierUnknown::Arithmetic(_))));
    }
}
