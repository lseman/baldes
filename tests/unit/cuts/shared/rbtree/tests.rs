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
