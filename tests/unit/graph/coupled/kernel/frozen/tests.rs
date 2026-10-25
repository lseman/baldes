use super::*;

#[test]
fn preparation_propagates_through_cycles_and_keeps_disconnected_components_separate() {
    let mut graph = Graph::<2>::new(6, 0).unwrap();
    for (tail, head) in [(0, 1), (1, 2), (2, 1), (1, 3), (4, 5)] {
        graph.add_arc(tail, head, 0.).unwrap();
    }
    let zero = || PiecewiseLinearFunction::new(vec![[0., 0.], [4., 0.]]).unwrap();
    let mut definitions = vec![CoupledResource::IntervalCost {
        initial: Window {
            lower: 0.,
            upper: 4.,
        },
        bounds: Window {
            lower: 0.,
            upper: 4.,
        },
        increments: vec![0., 0., 0., 0., 1.],
        functions: vec![
            zero(),
            zero(),
            zero(),
            PiecewiseLinearFunction::new(vec![[1., 1.], [3., 1.]]).unwrap(),
            zero(),
        ],
    }];
    let prepared = PreparedFrozen::new(&graph, &definitions, &[0]);
    let contained = Window {
        lower: 1.,
        upper: 3.,
    };
    for vertex in 0..3 {
        assert!(prepared.contains(0, vertex, contained, 0.));
        assert!(!prepared.contains(0, vertex, contained, 1.));
        assert!(!prepared.contains(
            0,
            vertex,
            Window {
                lower: 0.,
                upper: 4.
            },
            0.
        ));
    }
    assert!(prepared.contains(
        0,
        3,
        Window {
            lower: -100.,
            upper: 100.
        },
        1.
    ));
    assert!(!prepared.contains(0, 4, contained, 0.));
    let CoupledResource::IntervalCost { increments, .. } = &mut definitions[0] else {
        unreachable!()
    };
    increments[2] = 1.;
    let prepared = PreparedFrozen::new(&graph, &definitions, &[0]);
    for vertex in 0..3 {
        assert!(!prepared.contains(0, vertex, contained, 0.));
    }
}

#[test]
fn frozen_domains_and_charge_suppression_are_independent() {
    let prepared = PreparedFrozen {
        fronts: vec![vec![
            Continuation::IDENTITY,
            Continuation {
                zero_increment: true,
                zero_charge: false,
                lower: 0.,
                upper: 4.,
            },
            Continuation::UNKNOWN,
        ]]
        .into(),
    };
    let domain = Window {
        lower: 0.,
        upper: 4.,
    };
    assert!(prepared.contains(0, 0, domain, 1.));
    assert!(!prepared.contains(0, 1, domain, 1.));
    assert!(prepared.contains(0, 1, domain, 0.));
    assert!(!prepared.contains(
        0,
        1,
        Window {
            lower: -1.,
            upper: 4.
        },
        0.
    ));
    assert!(!prepared.contains(0, 2, domain, 0.));
    assert!(!prepared.contains(1, 0, domain, 0.));
    assert!(!prepared.contains(0, 3, domain, 0.));
}

#[test]
fn meet_never_restores_a_disproved_frozen_condition() {
    let a = Continuation {
        zero_increment: true,
        zero_charge: false,
        lower: 1.,
        upper: 8.,
    };
    let b = Continuation {
        zero_increment: false,
        zero_charge: true,
        lower: 2.,
        upper: 6.,
    };
    let next = a.meet(b);
    assert!(!next.zero_increment && !next.zero_charge);
    assert_eq!((next.lower, next.upper), (2., 6.));
    assert_eq!(next.meet(Continuation::IDENTITY), next);
    assert_eq!(next.meet(a), next);
    assert_eq!(next.meet(b), next);
}
