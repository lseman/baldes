use super::*;
fn function(points: &[[f64; 2]]) -> PiecewiseLinearFunction {
    PiecewiseLinearFunction::new(points.to_vec()).unwrap()
}
fn constant(lo: f64, hi: f64, value: f64) -> PiecewiseLinearFunction {
    function(&[[lo, value], [hi, value]])
}
fn arc() -> RelationArc {
    RelationArc {
        arrival: function(&[[0., 1.], [8., 9.]]),
        increment: constant(0., 8., 2.),
        lower: constant(1., 10., 0.),
        upper: constant(1., 10., 100.),
        waiting: 1,
    }
}
fn root() -> RelationFront {
    RelationFront::initial(
        &constant(0., 8., 0.),
        Window {
            lower: 0.,
            upper: 0.,
        },
    )
    .unwrap()
    .unwrap()
}
#[test]
fn waiting_retains_all_choices_and_marks_waiting_pieces() {
    let next = root().extend(&arc()).unwrap().unwrap();
    assert_eq!(next.value(1.).unwrap(), Some(2.));
    assert_eq!(next.value(4.).unwrap(), Some(5.));
    assert!(next
        .pieces()
        .iter()
        .any(|p| p.waiting && p.left[0] <= 4. && p.right[0] >= 4.));
}
#[test]
fn lower_functions_clamp_the_dependent_coordinate_instead_of_adding_cost() {
    let mut a = arc();
    a.lower = function(&[[1., 5.], [5., 1.], [10., 1.]]);
    let next = root().extend(&a).unwrap().unwrap();
    assert_eq!(next.value(1.).unwrap(), Some(5.));
    assert_eq!(next.value(2.).unwrap(), Some(4.));
    assert_eq!(next.value(3.).unwrap(), Some(4.));
    assert_eq!(next.minimum(), 3.5);
}
#[test]
fn upper_function_clipping_preserves_disconnected_domains_and_singletons() {
    let mut a = arc();
    a.arrival = function(&[[0., 0.], [2., 2.]]);
    a.increment = constant(0., 2., 0.);
    a.lower = constant(0., 2., 0.);
    a.upper = function(&[[0., 1.], [1., -1.], [2., 1.]]);
    a.waiting = 0;
    let next = root().extend(&a).unwrap().unwrap();
    assert_eq!(next.value(0.5).unwrap(), Some(0.));
    assert_eq!(next.value(1.).unwrap(), None);
    assert_eq!(next.value(1.5).unwrap(), Some(0.));
    a.upper = function(&[[0., -1.], [1., 0.], [2., -1.]]);
    let singleton = root().extend(&a).unwrap().unwrap();
    assert_eq!(singleton.value(1.).unwrap(), Some(0.));
    assert_eq!(singleton.value(0.5).unwrap(), None);
    assert_eq!(singleton.value(1.5).unwrap(), None);
}
#[test]
fn backward_feasibility_adjoint_accounts_for_forced_wait_and_upper_clamp() {
    let mut a = arc();
    a.upper = constant(3., 8., 7.);
    a.lower = constant(3., 8., 0.);
    let suffix = RelationFront::initial(
        &constant(3., 8., 8.),
        Window {
            lower: 3.,
            upper: 8.,
        },
    )
    .unwrap()
    .unwrap();
    let backward = suffix.extend_backward_upper(&a).unwrap().unwrap();
    // The head upper clamp gives7; departure0 arrives1, waits to3 and
    // spends2 waiting+2 increment, leaving a predecessor allowance3.
    assert_eq!(backward.value(0.).unwrap(), Some(3.));
    assert_eq!(backward.value(2.).unwrap(), Some(5.));
    assert_eq!(backward.value(7.).unwrap(), Some(5.));
    assert_eq!(backward.value(8.).unwrap(), None);
}
#[test]
fn backward_adjoint_preserves_feasibility_at_every_sampled_base_coordinate() {
    let a = arc();
    let suffix = RelationFront::initial(
        &function(&[[1., 0.], [5., 8.], [10., 3.]]),
        Window {
            lower: 1.,
            upper: 10.,
        },
    )
    .unwrap()
    .unwrap();
    let backward = suffix.extend_backward_upper(&a).unwrap().unwrap();
    for departure in 0..=8 {
        let t = f64::from(departure);
        let mut literal = f64::NEG_INFINITY;
        for head in (departure + 1)..=10 {
            let x = f64::from(head);
            let allowance = suffix.value(x).unwrap().unwrap().min(100.);
            if allowance >= 0. {
                literal = literal.max(allowance - 2. - (x - (t + 1.)));
            }
        }
        assert_eq!(backward.value(t).unwrap(), Some(literal));
    }
}
#[test]
fn native_cost_modes_enforce_their_own_restrictions_without_weakening_feasibility() {
    let a = arc();
    validate_cost_mode(std::slice::from_ref(&a), 0., false).unwrap();
    validate_cost_mode(std::slice::from_ref(&a), 0., true).unwrap();
    let mut signed = a.clone();
    signed.waiting = -1;
    signed.increment = constant(0., 8., -2.);
    signed.validate().unwrap();
    validate_cost_mode(std::slice::from_ref(&signed), 0., false).unwrap();
    assert!(validate_cost_mode(std::slice::from_ref(&signed), 0., true).is_err());
    let mut varying = a.clone();
    varying.lower = function(&[[1., 0.], [10., 1.]]);
    varying.validate().unwrap();
    assert!(validate_cost_mode(std::slice::from_ref(&varying), 0., false).is_err());
    let mut unequal = a.clone();
    unequal.upper = constant(1., 10., 99.);
    assert!(validate_cost_mode(&[a.clone(), unequal], 0., false).is_err());
    assert!(validate_cost_mode(&[a], -1., true).is_err());
}
#[test]
fn base_transition_rejects_non_fifo_and_negative_consumption() {
    let mut a = arc();
    a.arrival = function(&[[0., 1.], [1., 1.], [8., 9.]]);
    assert!(a.validate().is_err());
    a.arrival = function(&[[0., -1.], [8., 9.]]);
    assert!(a.validate().is_err());
}
#[test]
fn rounded_crossing_selects_the_interior_branch_and_retains_feasible_continuations() {
    // At the manufactured crossing0.35, literal F=1-2x evaluates to
    //0.30000000000000004, just above G=0.3. Selecting by that endpoint
    // used to retain G throughout the right interval, losing F(0.5)=0.
    let initial = function(&[[-1., 0.3], [0., 1.], [1., -1.]]);
    let front = RelationFront::initial(
        &initial,
        Window {
            lower: -1.,
            upper: 1.,
        },
    )
    .unwrap()
    .unwrap();
    let first = RelationArc {
        arrival: function(&[[-1., -1.], [1., 1.]]),
        increment: constant(-1., 1., 0.),
        lower: constant(0., 1., -10.),
        upper: constant(0., 1., 10.),
        waiting: 0,
    };
    let next = front.extend(&first).unwrap().unwrap();
    assert_eq!(next.value(0.5).unwrap(), Some(0.));
    assert_eq!(next.value(0.6).unwrap(), Some(-0.19999999999999996));
    let second = RelationArc {
        arrival: function(&[[0.5, 0.5], [0.6, 0.6]]),
        increment: constant(0.5, 0.6, 0.),
        lower: constant(0.5, 0.6, -10.),
        upper: constant(0.5, 0.6, 0.),
        waiting: 0,
    };
    let final_front = next.extend(&second).unwrap().unwrap();
    assert_eq!(final_front.minimum(), -0.19999999999999996);
}
#[test]
fn envelope_work_limit_is_an_error_instead_of_an_empty_frontier() {
    let candidate = RelationPiece {
        metadata: 0,
        left: [0., 0.],
        right: [1., 1.],
        waiting: false,
    };
    assert!(lower_envelope(&vec![candidate; 1500]).is_err());
}
