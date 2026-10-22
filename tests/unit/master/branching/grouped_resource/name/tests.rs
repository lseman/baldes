use super::*;
#[test]
fn display_rounding_preserves_exact_identity_outside_this_module() {
    for (value, expected) in [
        (0., "0"),
        (-0., "-0"),
        (0.0001, "0.0001"),
        (0.00001, "1e-05"),
        (999999.9, "1e+06"),
        (100000., "100000"),
        (1.23456789, "1.23457"),
        (-1.23456789, "-1.23457"),
    ] {
        assert_eq!(number(value), expected);
    }
}
