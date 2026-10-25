use super::*;
fn f(points: Vec<[f64; 2]>) -> PiecewiseLinearFunction {
    PiecewiseLinearFunction::new(points).unwrap()
}
#[test]
fn prepared_workspace_reuses_buffers_without_aliasing_retained_fronts() {
    let a = RelationArc {
        arrival: f(vec![[0., 1.], [4., 5.]]),
        increment: f(vec![[0., 2.], [4., 2.]]),
        lower: f(vec![[1., 0.], [6., 0.]]),
        upper: f(vec![[1., 100.], [6., 100.]]),
        waiting: 1,
    };
    let prepared = PreparedRelationArc::new(&a).unwrap();
    assert!(prepared.exact_cache_eligible());
    let front = RelationFront::initial(
        &f(vec![[0., 0.], [4., 0.]]),
        Window {
            lower: 0.,
            upper: 0.,
        },
    )
    .unwrap()
    .unwrap();
    let mut workspace = RelationWorkspace::default();
    let first = front
        .extend_prepared(&prepared, &mut workspace)
        .unwrap()
        .unwrap();
    let saved = first.clone();
    let second = first
        .extend_prepared(&prepared, &mut workspace)
        .unwrap()
        .unwrap();
    assert_eq!(first, saved);
    assert_eq!(second.minimum(), 4.);
    assert!(!workspace.transported.is_empty());
}
#[test]
fn interior_arrival_plateau_is_unknown_instead_of_false_infeasibility() {
    let base = 1e16;
    let a = RelationArc {
        arrival: f(vec![[0., base], [1., base + 2.]]),
        increment: f(vec![[0., 0.], [1., 0.]]),
        lower: f(vec![[base, -10.], [base + 2., -10.]]),
        upper: f(vec![[base, -0.2], [base + 2., -2.]]),
        waiting: 0,
    };
    let initial = f(vec![[0., 0.], [1., -1.]]);
    let t = 0.25;
    let arrival = a.arrival.value(t).unwrap().unwrap();
    assert!(initial.value(t).unwrap().unwrap() <= a.upper.value(arrival).unwrap().unwrap());
    let front = RelationFront::initial(
        &initial,
        Window {
            lower: 0.,
            upper: 1.,
        },
    )
    .unwrap()
    .unwrap();
    assert!(front.extend(&a).is_err());
}
