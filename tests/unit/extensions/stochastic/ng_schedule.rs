use super::*;

fn result(status: PricingStatus, lower_bound: Option<f64>) -> super::super::NgPricingResult {
    super::super::NgPricingResult {
        status,
        routes: vec![],
        lower_bound,
        statistics: Default::default(),
        refinements: 0,
        discarded_walks: 0,
    }
}

#[test]
fn completed_ng_bound_survives_interrupted_later_stages() {
    let mut aggregate = result(PricingStatus::Complete, Some(-3.));
    merge_ng_results(&mut aggregate, result(PricingStatus::TimeLimit, None));
    assert_eq!(aggregate.lower_bound, Some(-3.));
    assert_eq!(aggregate.status, PricingStatus::TimeLimit);
    merge_ng_results(&mut aggregate, result(PricingStatus::Complete, Some(-2.)));
    assert_eq!(aggregate.lower_bound, Some(-2.));
    merge_ng_results(&mut aggregate, result(PricingStatus::Complete, Some(-4.)));
    assert_eq!(aggregate.lower_bound, Some(-2.));
}
