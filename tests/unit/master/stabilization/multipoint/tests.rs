use super::*;
use crate::master::clp::ClpMaster;
#[test]
fn retention_protects_centers_ages_inactive_rows_and_limits_memory() {
    let make = |id, dual, age, protected| SupportPlane {
        point: vec![id as f64],
        dual_activity: dual,
        zero_age: age,
        protected_center: protected,
        creation_id: id,
    };
    let mut planes = vec![
        make(1, 0., 2, false),
        make(2, 1., 0, false),
        make(3, 0., 10, true),
    ];
    assert!(observe_activity(&mut planes, Some(&[0., 1., 0.])));
    retain_planes(&mut planes, 3).unwrap();
    assert_eq!(
        planes.iter().map(|p| p.creation_id).collect::<Vec<_>>(),
        vec![3, 2]
    );
    retain_planes(&mut planes, 1).unwrap();
    assert_eq!(planes[0].creation_id, 3);
    let before = planes[0].zero_age;
    assert!(!observe_activity(&mut planes, None));
    assert_eq!(planes[0].zero_age, before);
    let mut near = vec![
        make(1, 1., 0, false),
        make(2, 1. + 1e-7, 0, false),
        make(3, 1., 1, false),
    ];
    retain_planes(&mut near, 3).unwrap();
    assert_eq!(
        near.iter().map(|p| p.creation_id).collect::<Vec<_>>(),
        vec![2, 1, 3]
    );
    assert!(retain_planes(&mut near, 0).is_err());
}
#[test]
fn beta_epigraph_matches_two_support_planes_and_allows_negative_beta() {
    for (samples, sign, expected) in [
        (vec![vec![2.], vec![5.]], 1., 5.),
        (vec![vec![2.], vec![5.]], -1., -2.),
    ] {
        let mut private = ClpMaster::new().unwrap();
        let fixed = private.add_row(RowSense::Equal, sign, &[]).unwrap();
        let negative = private.add_column(0., &[(fixed, -1.)]).unwrap();
        let positive = private.add_column(0., &[(fixed, 1.)]).unwrap();
        add_support_planes(&mut private, &[(0, negative, positive)], &samples).unwrap();
        assert!(private.solve());
        assert!((private.objective() - expected).abs() < 1e-7);
    }
}

#[test]
fn support_plane_order_matches_captured_native_comparator() {
    let capture: serde_json::Value =
        serde_json::from_str(include_str!("native_sort_capture.json")).unwrap();
    let cases = capture["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 712);
    let plane = |value: &serde_json::Value| SupportPlane {
        point: vec![],
        dual_activity: f64::from_bits(value[0].as_u64().unwrap()),
        protected_center: value[1].as_bool().unwrap(),
        zero_age: value[2].as_u64().unwrap() as u32,
        creation_id: value[3].as_u64().unwrap(),
    };
    for (index, case) in cases.iter().enumerate() {
        assert_eq!(
            precedes(&plane(&case["b"]), &plane(&case["a"])),
            case["b_precedes_a"].as_bool().unwrap(),
            "case {index}"
        );
    }
}

#[test]
fn full_support_plane_order_matches_native_buffered_sort() {
    let capture: serde_json::Value =
        serde_json::from_str(include_str!("order_capture.json")).unwrap();
    let cases = capture["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 169);
    assert_eq!(capture["insertion_mismatches"], 85);
    for case in cases {
        let mut planes: Vec<_> = case["planes"]
            .as_array()
            .unwrap()
            .iter()
            .map(|row| SupportPlane {
                point: vec![],
                dual_activity: f64::from_bits(row[0].as_u64().unwrap()),
                protected_center: row[1].as_bool().unwrap(),
                zero_age: row[2].as_u64().unwrap() as u32,
                creation_id: row[3].as_u64().unwrap(),
            })
            .collect();
        let capacity = planes.len();
        retain_planes(&mut planes, capacity).unwrap();
        let actual: Vec<_> = planes.iter().map(|p| p.creation_id - 1).collect();
        let expected: Vec<_> = case["order"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap())
            .collect();
        assert_eq!(actual, expected, "{case}");
    }
}
