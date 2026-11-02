use super::*;

#[test]
fn combined_tree_metrics_match_original_binary_across_objective_scales() {
    // Actual two-child routine at ELF 0x1183e0 in the supplied reference,
    // SHA256 103caf4a9c58d6141e935848294bff16677f400323466dc23cd4b9e5c15889d6.
    // Fields +0x38/+0x40 hold score and actual subtree size. The latter is
    // independent of the separate depth diagnostic at +0x48.
    let fixtures: Vec<[f64; 6]> =
        serde_json::from_str(include_str!("numerical_metrics_fixtures.json")).unwrap();
    for [parent, cutoff, lower, upper, score, size] in fixtures {
        let metrics = tree_depth_metrics(parent, cutoff, [lower, upper], 1.);
        assert_eq!(
            metrics.0, score,
            "score at {parent}, {cutoff}, [{lower}, {upper}]"
        );
        assert_eq!(
            metrics.1, size,
            "size at {parent}, {cutoff}, [{lower}, {upper}]"
        );
        assert_eq!(tree_depth_score(parent, cutoff, [lower, upper], 1.), score);
    }
}

#[test]
fn score_normalization_leaves_actual_subtree_size_unchanged() {
    let expected = tree_depth_metrics(10., 20., [12., 14.], 1.).1;
    for normalization in [1e-12, 0.1, 10., 1e12] {
        assert_eq!(
            tree_depth_metrics(10., 20., [12., 14.], normalization).1,
            expected
        );
    }
}
