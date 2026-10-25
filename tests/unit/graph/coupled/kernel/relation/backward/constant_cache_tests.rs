use super::*;
fn f(points: Vec<[f64; 2]>) -> PiecewiseLinearFunction {
    PiecewiseLinearFunction::new(points).unwrap()
}
#[test]
fn constant_upper_fast_transform_matches_general_adjoint_with_forced_waiting() {
    for waiting in 0..=3 {
        let arc = PreparedRelationArc::new(&PairedResourceArc {
            arrival: f(vec![[0., 1.], [8., 9.]]),
            increment: f(vec![[0., 2.], [8., 2.]]),
            lower: f(vec![[3., 0.], [8., 0.]]),
            upper: f(vec![[3., 100.], [8., 100.]]),
            waiting,
        })
        .unwrap();
        let root = BoundedCostFront::terminal(
            Window {
                lower: 3.,
                upper: 8.,
            },
            100.,
        )
        .unwrap();
        let fast = root.checked_prepend(&arc).unwrap();
        let general = root.prepend(&arc).unwrap().unwrap();
        for t in 0..=8 {
            assert_eq!(
                fast.upper.value(f64::from(t)).unwrap(),
                general.upper.value(f64::from(t)).unwrap()
            );
        }
    }
}
