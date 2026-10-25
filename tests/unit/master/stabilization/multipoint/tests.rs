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
