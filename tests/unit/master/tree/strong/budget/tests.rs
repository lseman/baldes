use super::*;


#[test]
fn observed_depth_history_prevents_the_parent_priority_reset() {
    let mut budget = CandidateBudget::new(
        100,
        f64::from(0.3f32),
        SubtreeEstimate {
            size: 3.236066202157737,
            priority: 2.,
        },
    );
    budget.use_history();
    budget.visit(1.);
    assert_eq!(budget.quota, 1);
    assert_eq!(budget.estimate.size, 3.236066202157737);
}




#[test]
fn lower_priority_resets_the_quota_and_estimate() {
    let mut budget = CandidateBudget::new(
        100,
        f64::from(0.3f32),
        SubtreeEstimate {
            size: 3.236066202157737,
            priority: 2.,
        },
    );
    assert_eq!(budget.quota, 1);
    budget.visit(10.);
    budget.visit(2.);
    assert_eq!(budget.quota, 1);
    budget.visit(1.);
    assert_eq!(budget.quota, 100);
    assert_eq!(budget.estimate.size, 1e12);
}
