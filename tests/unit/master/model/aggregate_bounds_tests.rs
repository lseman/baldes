use super::split_bounds;

#[test]
fn aggregate_split_endpoints_match_original_next_node_observations() {
    let fixture: serde_json::Value =
        serde_json::from_str(include_str!("aggregate_split_fixtures.json")).unwrap();
    assert_eq!(
        fixture["binary_sha256"],
        "103caf4a9c58d6141e935848294bff16677f400323466dc23cd4b9e5c15889d6"
    );
    let cases = fixture["cases"].as_array().unwrap();
    assert!(cases.len() >= 40);
    for case in cases {
        let value = case["value"].as_f64().unwrap();
        let expected = (
            case["lower"].as_f64().unwrap(),
            case["upper"].as_f64().unwrap(),
        );
        assert_eq!(
            split_bounds(value),
            Some(expected),
            "full activity {value:?}"
        );
    }
}

#[test]
fn aggregate_split_tolerates_small_activity_error_around_an_integer() {
    for value in [1. - 5e-7, 1., 1. + 5e-7] {
        assert_eq!(split_bounds(value), Some((1., 1.)));
    }
    assert_eq!(split_bounds(1. + 2e-6), Some((1., 2.)));
    assert_eq!(split_bounds(1. - 2e-6), Some((0., 1.)));
    assert_eq!(split_bounds(-1. + 5e-7), Some((-1., -1.)));
    assert_eq!(split_bounds(-1. - 5e-7), Some((-1., -1.)));
    assert_eq!(split_bounds(-0.), Some((0., 0.)));
}

#[test]
fn aggregate_split_declines_nonfinite_or_unsupported_integer_activity() {
    for value in [
        f64::NAN,
        f64::INFINITY,
        f64::NEG_INFINITY,
        9_007_199_254_740_992.,
        -9_007_199_254_740_992.,
    ] {
        assert_eq!(split_bounds(value), None);
    }
}

#[test]
fn aggregate_fractional_distances_match_original_getters_independently() {
    let fixture: serde_json::Value =
        serde_json::from_str(include_str!("aggregate_distance_fixtures.json")).unwrap();
    assert_eq!(
        fixture["binary_sha256"],
        "103caf4a9c58d6141e935848294bff16677f400323466dc23cd4b9e5c15889d6"
    );
    let cases = fixture["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 55);
    for case in cases {
        let value = case["value"].as_f64().unwrap();
        let (lower, upper) = super::fractional_distances(value).unwrap();
        assert_eq!(
            format!("{:016x}", lower.to_bits()),
            case["lower_bits"].as_str().unwrap(),
            "lower distance at {value:?}"
        );
        assert_eq!(
            format!("{:016x}", upper.to_bits()),
            case["upper_bits"].as_str().unwrap(),
            "upper distance at {value:?}"
        );
    }
}

#[test]
fn aggregate_near_integer_distances_are_both_zero_and_are_not_complements() {
    for value in [1. - 5e-7, 1., 1. + 5e-7, -1. - 5e-7, -1., -1. + 5e-7] {
        assert_eq!(super::fractional_distances(value), Some((0., 0.)));
    }
    assert_eq!(super::fractional_distances(1.25), Some((0.25, 0.75)));
    assert_eq!(super::fractional_distances(-1.25), Some((0.75, 0.25)));
    assert_eq!(super::fractional_distances(f64::NAN), None);
}

#[test]
fn applied_split_preserves_integer_coverage_under_large_negative_shifts() {
    for value in [-1e10 + 0.25, -1e12, -1e12 + 0.25, -4_503_599_627_370_496.] {
        let raw = split_bounds(value).unwrap();
        assert!(raw.1 - raw.0 > 1.);
        let applied = super::covering_split_bounds(value).unwrap();
        assert_eq!(applied, (value.floor(), value.ceil()));
        for integer in [
            value.floor() - 1.,
            value.floor(),
            value.ceil(),
            value.ceil() + 1.,
        ] {
            assert!(integer <= applied.0 || integer >= applied.1);
        }
    }
    for value in [0.25, 1. + 5e-7, -1.25, 1e10 + 0.25] {
        assert_eq!(super::covering_split_bounds(value), split_bounds(value));
    }
}
