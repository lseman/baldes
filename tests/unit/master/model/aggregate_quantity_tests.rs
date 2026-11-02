use super::fractional_quantity;

#[test]
fn quantity_gate_matches_original_producer_finite_cases() {
    let cases: Vec<(f64, bool)> =
        serde_json::from_str(include_str!("aggregate_quantity_fixtures.json")).unwrap();
    for (quantity, expected) in cases {
        assert_eq!(
            fractional_quantity(quantity),
            expected,
            "quantity={quantity:?}"
        );
    }
}

#[test]
fn quantity_gate_preserves_signed_relative_large_values() {
    assert!(fractional_quantity(-1e20));
    assert!(fractional_quantity(-1e308));
    assert!(!fractional_quantity(1e20));
    assert!(!fractional_quantity(1e308));
    assert!(!fractional_quantity(0.));
    assert!(!fractional_quantity(-0.));
    assert!(!fractional_quantity(1e-8));
    assert!(fractional_quantity(-0.25));
}

#[test]
fn quantity_gate_declines_nonfinite_solver_values() {
    for quantity in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert!(!fractional_quantity(quantity));
    }
}
