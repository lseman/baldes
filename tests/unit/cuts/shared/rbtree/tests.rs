use super::*;
#[test]
fn orders_by_score_then_id() {
    let mut s = CandidateSet::new();
    for (i, score) in [-0.3, -0.5, -0.1, -0.5, -0.2].into_iter().enumerate() {
        assert!(s.insert(score, i as i32, i));
    }
    let order: Vec<_> = s.ordered().into_iter().map(|(_, v)| v).collect();
    assert_eq!(order, vec![1, 3, 0, 4, 2]);
    assert!(!s.insert(-0.5, 1, 9));
}

#[test]
fn caller_ordered_handles_retain_unique_tree_order() {
    let keys = [3, 1, 4, 1, 2];
    let mut set = CandidateSet::new();
    for handle in 0..keys.len() {
        assert_eq!(set.insert_by(handle, |a, b| keys[a] < keys[b]), handle != 3);
    }
    let handles: Vec<_> = set
        .ordered()
        .into_iter()
        .map(|(_, handle)| handle)
        .collect();
    assert_eq!(handles, vec![1, 4, 0, 2]);
    assert_eq!(set.len(), 4);
}

#[test]
fn caller_ordering_uses_balanced_search_for_large_registries() {
    use std::cell::Cell;
    let comparisons = Cell::new(0_usize);
    let mut set = CandidateSet::new();
    let count = 4096;
    for handle in 0..count {
        assert!(set.insert_by(handle, |a, b| {
            comparisons.set(comparisons.get() + 1);
            a > b
        }));
    }
    assert_eq!(
        set.ordered()
            .into_iter()
            .map(|(_, handle)| handle)
            .collect::<Vec<_>>(),
        (0..count).rev().collect::<Vec<_>>()
    );
    // A sorted-vector insertion scan needs millions of comparisons here.
    assert!(comparisons.get() < count * 32, "{}", comparisons.get());
}
