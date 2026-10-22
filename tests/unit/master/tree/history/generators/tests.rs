use super::*;
#[test]
fn retained_ryan_foster_uses_current_cooccurrence_and_respects_ancestor_resources() {
    let (problem, routes, _) = crate::master::tree::candidates::resource_groups::tests::fixture();
    let rule = crate::pricing::restrictions::RyanFosterConstraint {
        first: 1,
        second: 3,
        together: false,
    };
    let candidate = BranchCandidate {
        down: BranchConstraint::RyanFoster(rule),
        up: BranchConstraint::RyanFoster(crate::pricing::restrictions::RyanFosterConstraint {
            together: true,
            ..rule
        }),
        value: 0.5,
    };
    let mut h = BranchEvaluationHistory::default();
    for side in 0..2 {
        h.record(
            &candidate,
            side,
            0,
            EvaluationKind::ExactPricing,
            0.,
            1.,
            0.,
        );
    }
    let point: Vec<_> = routes.iter().map(|(r, w)| (r, *w)).collect();
    let options = BranchingOptions {
        ryan_foster: true,
        ..BranchingOptions::arc_flow_only()
    };
    let mut supplied = vec![];
    h.refresh_generators(&problem, &point, &[], &options, 2., &mut supplied);
    assert_eq!(supplied.len(), 1);
    assert_eq!(supplied[0].value, 0.25);
    let mut excluded = vec![];
    h.refresh_generators(
        &problem,
        &point,
        &[candidate.up],
        &options,
        2.,
        &mut excluded,
    );
    assert!(excluded.is_empty());
    let ancestors =
        vec![
            BranchConstraint::RyanFoster(crate::pricing::restrictions::RyanFosterConstraint {
                first: 2,
                second: 4,
                together: false
            });
            512
        ];
    h.refresh_generators(&problem, &point, &ancestors, &options, 2., &mut excluded);
    assert!(excluded.is_empty());
}
#[test]
fn retained_arc_activity_refreshes_only_after_priority_admission() {
    let (problem, routes, arcs) =
        crate::master::tree::candidates::resource_groups::tests::fixture();
    let arc = problem.arc_position(arcs[0]).unwrap();
    let c = BranchCandidate {
        down: BranchConstraint::ArcFlow {
            arc,
            sense: RowSense::LessEqual,
            rhs: 0.,
        },
        up: BranchConstraint::ArcFlow {
            arc,
            sense: RowSense::GreaterEqual,
            rhs: 1.,
        },
        value: 0.5,
    };
    let mut h = BranchEvaluationHistory::default();
    for side in 0..2 {
        h.record(&c, side, 0, EvaluationKind::ExactPricing, 0., 1., 0.);
    }
    let point: Vec<_> = routes.iter().map(|(r, w)| (r, *w)).collect();
    let options = BranchingOptions::arc_flow_only();
    let mut supplied = vec![];
    h.refresh_generators(&problem, &point, &[], &options, 2., &mut supplied);
    assert!(supplied.is_empty());
    h.refresh_generators(&problem, &point, &[], &options, 1., &mut supplied);
    assert_eq!(supplied.len(), 1);
    assert_eq!(supplied[0].value, 0.25);
    assert_eq!(supplied[0].down, c.down);
    h.refresh_generators(&problem, &point, &[], &options, 1., &mut supplied);
    assert_eq!(supplied.len(), 1);
    let mut unknown = routes[0].0.clone();
    unknown.arcs = vec![i32::MAX];
    let mut excluded = vec![];
    // This would reach coefficient replay's validated-arc assertion if a
    // below-cutoff generator were refreshed before the admission check.
    h.refresh_generators(
        &problem,
        &[(&unknown, 0.5)],
        &[],
        &options,
        2.,
        &mut excluded,
    );
    assert!(excluded.is_empty());
    h.refresh_generators(
        &problem,
        &point,
        &[c.up.clone()],
        &options,
        1.,
        &mut excluded,
    );
    assert!(excluded.is_empty());
}
