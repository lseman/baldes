use super::*;
#[test]
fn clones_are_independent_on_both_sides_of_inline_boundary() {
    for len in [0, 1, 4, 5, 20] {
        let original = StateValues::<f64, 4>::filled(len, 3.);
        let mut copy = original.clone();
        assert_eq!(original.as_slice(), vec![3.; len]);
        assert_eq!(matches!(original, StateValues::Inline { .. }), len <= 4);
        if len > 0 {
            copy[0] = 7.;
            assert_eq!(original[0], 3.);
        }
        let rebuilt = StateValues::<f64, 4>::from_slice(&original);
        assert_eq!(original, rebuilt);
    }
}
