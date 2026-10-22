use super::*;
#[test]
fn automatic_pricing_bounds_enumeration_work() {
    assert_eq!(
        select_method(31, 3, PricingMethod::Auto),
        PricingMethod::Enumeration
    );
    assert_eq!(
        select_method(21, 4, PricingMethod::Auto),
        PricingMethod::Enumeration
    );
    assert_eq!(
        select_method(64, 4, PricingMethod::Auto),
        PricingMethod::Bucket
    );
    assert_eq!(
        select_method(usize::MAX, 6, PricingMethod::Auto),
        PricingMethod::Bucket
    );
}
