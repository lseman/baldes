use super::*;

#[test]
fn rank_one_order_sorts_unrounded_values_largest_first() {
    let values = [0.25, 0.75, 0.5];
    let mut order = vec![0, 1, 2];
    sort_rank_one_pricing_order(&mut order, |i| values[i]);
    assert_eq!(order, vec![1, 2, 0]);
}

#[test]
fn rank_one_order_keeps_cut_ids_within_tolerance() {
    // Distinct after 1e-8 rounding, but within the native 1e-6 tie window.
    let values = [0.5, 0.5000004, 0.4999997];
    let mut order = vec![2, 1, 0];
    sort_rank_one_pricing_order(&mut order, |i| values[i]);
    assert_eq!(order, vec![0, 1, 2]);
}

#[test]
fn rank_one_order_is_signed_not_absolute() {
    let values = [-0.75, 0.5];
    let mut order = vec![0, 1];
    sort_rank_one_pricing_order(&mut order, |i| values[i]);
    assert_eq!(order, vec![1, 0]);
}
