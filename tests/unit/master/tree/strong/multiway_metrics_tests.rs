use super::*;
use serde::Deserialize;
#[derive(Deserialize)]
struct Case {
    parent: u64,
    cutoff: u64,
    children: Vec<u64>,
    product: u64,
    tree_score: u64,
    tree_size: u64,
    processed: usize,
    factor: u64,
}
#[derive(Deserialize)]
struct Fixtures {
    complete: Vec<Case>,
    partial: Vec<Case>,
}
#[test]
fn multiway_completed_scores_match_original_binary_at_arity_two_through_eight() {
    let fixtures: Fixtures =
        serde_json::from_str(include_str!("multiway_metrics_fixtures.json")).unwrap();
    assert_eq!(fixtures.complete.len(), 385);
    for (index, case) in fixtures.complete.into_iter().enumerate() {
        let children: Vec<_> = case.children.into_iter().map(f64::from_bits).collect();
        let parent = f64::from_bits(case.parent);
        let cutoff = f64::from_bits(case.cutoff);
        assert_eq!(
            product_score(parent, cutoff, &children).to_bits(),
            case.product,
            "product case {index}"
        );
        let metrics = tree_depth_metrics(parent, cutoff, &children, 1.);
        assert_eq!(
            metrics.0.to_bits(),
            case.tree_score,
            "tree score case {index}"
        );
        assert_eq!(
            metrics.1.to_bits(),
            case.tree_size,
            "tree size case {index}"
        );
    }
}
#[test]
fn multiway_partial_scores_match_original_processed_prefix_and_pending_scaling() {
    let fixtures: Fixtures =
        serde_json::from_str(include_str!("multiway_metrics_fixtures.json")).unwrap();
    assert_eq!(fixtures.partial.len(), 405);
    for (index, case) in fixtures.partial.into_iter().enumerate() {
        let children: Vec<_> = case.children.into_iter().map(f64::from_bits).collect();
        let score = partial_tree_depth_score(
            f64::from_bits(case.parent),
            f64::from_bits(case.cutoff),
            &children,
            case.processed,
            f64::from_bits(case.factor),
        );
        assert_eq!(score.to_bits(), case.tree_score, "partial case {index}");
    }
}

#[test]
fn exhausted_and_singleton_groups_match_original_scores() {
    let fixtures: Fixtures =
        serde_json::from_str(include_str!("low_arity_metrics_fixtures.json")).unwrap();
    assert_eq!(fixtures.complete.len(), 48);
    for (index, case) in fixtures.complete.into_iter().enumerate() {
        let children: Vec<_> = case.children.into_iter().map(f64::from_bits).collect();
        let parent = f64::from_bits(case.parent);
        let cutoff = f64::from_bits(case.cutoff);
        assert_eq!(
            product_score(parent, cutoff, &children).to_bits(),
            case.product,
            "low-arity product {index}"
        );
        let (score, size) = tree_depth_metrics(parent, cutoff, &children, 1.);
        assert_eq!(score.to_bits(), case.tree_score, "low-arity tree {index}");
        assert_eq!(size.to_bits(), case.tree_size, "low-arity size {index}");
    }
    assert_eq!(fixtures.partial.len(), 270);
    for (index, case) in fixtures.partial.into_iter().enumerate() {
        let children: Vec<_> = case.children.into_iter().map(f64::from_bits).collect();
        let actual = partial_tree_depth_score(
            f64::from_bits(case.parent),
            f64::from_bits(case.cutoff),
            &children,
            case.processed,
            f64::from_bits(case.factor),
        );
        assert_eq!(
            actual.to_bits(),
            case.tree_score,
            "low-arity partial {index}"
        );
    }
}
