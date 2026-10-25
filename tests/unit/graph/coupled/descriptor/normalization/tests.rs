use super::*;

#[test]
fn absolute_native_prefix_charges_include_the_initial_value() {
    let absolute = PairedCostNormalization::AbsoluteFinalValue;
    let relative = PairedCostNormalization::FinalMinusInitialMinimum;
    assert_eq!(absolute.root_charge(), 0.);
    assert_eq!(relative.root_charge(), 0.);
    for (minimum, native_charge, relative_charge) in [(6., 6., 1.), (7., 7., 2.), (8., 8., 3.)] {
        assert_eq!(absolute.charge(5., minimum, 1.).unwrap(), native_charge);
        assert_eq!(relative.charge(5., minimum, 1.).unwrap(), relative_charge);
    }
}

#[test]
fn suppression_does_not_evaluate_an_overflowing_unused_difference() {
    let relative = PairedCostNormalization::FinalMinusInitialMinimum;
    assert_eq!(relative.charge(-f64::MAX, f64::MAX, 0.).unwrap(), 0.);
    assert!(relative.charge(-f64::MAX, f64::MAX, 1.).is_err());
    assert!(relative.charge(0., f64::MAX, 2.).is_err());
    assert!(relative.charge(0., 1., -1.).is_err());
    assert!(relative.charge(f64::NAN, 1., 0.).is_err());
}

#[cfg(feature = "clp")]
#[test]
fn serialized_conventions_remain_explicit_and_distinct() {
    for (convention, wire) in [
        (
            PairedCostNormalization::AbsoluteFinalValue,
            "absolute_final_value",
        ),
        (
            PairedCostNormalization::FinalMinusInitialMinimum,
            "final_minus_initial_minimum",
        ),
    ] {
        let encoded = serde_json::to_string(&convention).unwrap();
        assert_eq!(encoded, format!("\"{wire}\""));
        assert_eq!(
            serde_json::from_str::<PairedCostNormalization>(&encoded).unwrap(),
            convention
        );
    }
}
