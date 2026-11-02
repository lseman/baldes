use super::*;

#[derive(serde::Deserialize)]
struct Capture {
    cases: Vec<Case>,
}
#[derive(serde::Deserialize)]
struct Case {
    nonempty: bool,
    value: f64,
    lower: f64,
    upper: f64,
    lower_distance: f64,
    upper_distance: f64,
}

#[test]
fn component_stored_splits_and_distances_match_original_rounding() {
    let capture: Capture =
        serde_json::from_str(include_str!("component_history_rounding_fixtures.json")).unwrap();
    assert_eq!(capture.cases.len(), 1058);
    for case in capture.cases {
        let (lower, upper) = split_bounds(case.value, case.nonempty).unwrap();
        assert_eq!(
            lower.to_bits(),
            case.lower.to_bits(),
            "{} nonempty={}",
            case.value,
            case.nonempty
        );
        assert_eq!(upper.to_bits(), case.upper.to_bits());
        assert_eq!(
            (case.value - lower).to_bits(),
            case.lower_distance.to_bits()
        );
        assert_eq!(
            (upper - case.value).to_bits(),
            case.upper_distance.to_bits()
        );
        assert_eq!(lower.fract(), 0.);
        assert_eq!(upper.fract(), 0.);
    }
    for value in [
        f64::NAN,
        f64::INFINITY,
        f64::NEG_INFINITY,
        9_007_199_254_740_992.,
    ] {
        assert!(split_bounds(value, false).is_none());
        assert!(split_bounds(value, true).is_none());
    }
}

#[test]
fn shifted_component_endpoints_cannot_leave_an_integer_gap() {
    let mut encountered_gap = false;
    for offset in 1..4096 {
        let value = 9_007_199_254_740_992. - offset as f64;
        let (lower, upper) = split_bounds(value, true).unwrap();
        let covered = covering_split_bounds(value, true).unwrap();
        assert!(covered.1 - covered.0 <= 1.);
        if upper - lower > 1. {
            encountered_gap = true;
            assert_eq!(covered, (value.floor(), value.ceil()));
        }
    }
    assert!(encountered_gap);
}
