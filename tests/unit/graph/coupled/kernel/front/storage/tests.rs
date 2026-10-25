use super::*;
#[test]
fn compact_points_preserve_exact_values_and_independent_wide_clones() {
    for len in [0, 1, 2, 3, 12, 97] {
        let original: FrontPoints = (0..len).map(|i| [i as f64, -(i as f64)]).collect();
        assert_eq!(matches!(original, FrontPoints::Heap(_)), len > 2);
        assert_eq!(original.len(), len);
        let mut copy = original.clone();
        if len > 0 {
            copy[0][1] = 100.;
            assert_eq!(original[0][1].to_bits(), (-0.0_f64).to_bits());
        }
        let rebuilt = FrontPoints::from(original.to_vec());
        assert_eq!(rebuilt, original);
    }
}
