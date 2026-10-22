use super::*;



#[test]
fn no_split_and_invalid_intervals_are_distinct() {
    assert_eq!(select_resource_threshold(&[], 10.).unwrap(), None);
    let i = ResourceInterval {
        lower: 1.,
        upper: 2.,
        weight: 1.,
    };
    assert_eq!(select_resource_threshold(&[i], 10.).unwrap(), None);
    for bad in [
        ResourceInterval {
            lower: f64::NAN,
            ..i
        },
        ResourceInterval {
            upper: f64::INFINITY,
            ..i
        },
        ResourceInterval { lower: 3., ..i },
        ResourceInterval { weight: 0., ..i },
        ResourceInterval { weight: -1., ..i },
    ] {
        assert!(select_resource_threshold(&[bad], 10.).is_err());
    }
    assert!(select_resource_threshold(&[i], 1.).is_err());
    assert!(select_resource_threshold(&[i], f64::INFINITY).is_err());
}
