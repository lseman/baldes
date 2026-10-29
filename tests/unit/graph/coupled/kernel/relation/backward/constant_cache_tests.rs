use super::*;
fn f(points: Vec<[f64; 2]>) -> PiecewiseLinearFunction {
    PiecewiseLinearFunction::new(points).unwrap()
}

#[test]
fn zero_increment_suffix_cycles_preserve_exact_cost_and_join() {
    let arc = PreparedRelationArc::new(&PairedResourceArc {
        arrival: f(vec![[0., 0.], [1., 1.]]),
        increment: f(vec![[0., 0.], [1., 0.]]),
        lower: f(vec![[0., 0.], [1., 0.]]),
        upper: f(vec![[0., 16.], [1., 16.]]),
        waiting: 0,
    })
    .unwrap();
    let terminal = BoundedCostFront::terminal(
        Window {
            lower: 0.,
            upper: 1.,
        },
        16.,
    )
    .unwrap();
    let one = terminal
        .checked_prepend(&arc)
        .expect("zero increments preserve the exact lattice");
    let mut repeated = one.clone();
    for _ in 0..64 {
        repeated = repeated.checked_prepend(&arc).unwrap();
    }
    let prefix = RelationFront::initial(
        &f(vec![[0., 3.], [1., 3.]]),
        Window {
            lower: 0.,
            upper: 1.,
        },
    )
    .unwrap()
    .unwrap();
    assert_eq!(repeated.minimum_additional(), Some(0.));
    assert!(one.dominates_exact_lattice(&repeated));
    assert!(repeated.dominates_exact_lattice(&one));
    assert_eq!(repeated.checked_compatible_join(&prefix), Some(Some(3.)));
    let outside = RelationFront::initial(
        &f(vec![[0., 17.], [1., 17.]]),
        Window {
            lower: 0.,
            upper: 1.,
        },
    )
    .unwrap()
    .unwrap();
    assert_eq!(repeated.checked_compatible_join(&outside), Some(None));
}

#[test]
fn dyadic_cost_lattice_retains_zero_arcs_and_accumulates_only_charged_arcs() {
    let arc = |increment| {
        PreparedRelationArc::new(&PairedResourceArc {
            arrival: f(vec![[0., 0.], [1., 1.]]),
            increment: f(vec![[0., increment], [1., increment]]),
            lower: f(vec![[0., 0.], [1., 0.]]),
            upper: f(vec![[0., 8.], [1., 8.]]),
            waiting: 0,
        })
        .unwrap()
    };
    let lattice = PreparedCostLattice::new(0.5, &[arc(0.), arc(0.25)]).unwrap();
    let charged = lattice.prepend(None, 1, true).unwrap();
    let mut mixed = charged.clone();
    for _ in 0..64 {
        mixed = lattice.prepend(Some(&mixed), 0, false).unwrap();
    }
    assert_eq!(mixed.minimum_additional(), Some(0.25));
    assert!(charged.dominates_exact_lattice(&mixed));
    assert!(mixed.dominates_exact_lattice(&charged));
    let prefix = RelationFront::initial(
        &f(vec![[0., 0.5], [1., 0.5]]),
        Window {
            lower: 0.,
            upper: 1.,
        },
    )
    .unwrap()
    .unwrap();
    assert_eq!(mixed.checked_compatible_join(&prefix), Some(Some(0.75)));

    let zero = PreparedCostLattice::new(
        0.,
        &[PreparedRelationArc::new(&PairedResourceArc {
            arrival: f(vec![[0., 0.], [1., 1.]]),
            increment: f(vec![[0., 0.], [1., 0.]]),
            lower: f(vec![[0., 0.], [1., 0.]]),
            upper: f(vec![[0., 0.], [1., 0.]]),
            waiting: 0,
        })
        .unwrap()],
    )
    .expect("all-zero programs use the integer unit");
    assert_eq!(
        zero.prepend(None, 0, true).unwrap().minimum_additional(),
        Some(0.)
    );
    assert!(PreparedCostLattice::new(0.5, &[arc(-0.25)]).is_none());
}
#[test]
fn bounded_cost_cache_accepts_zero_based_identity_with_positive_common_domain() {
    let arc = PreparedRelationArc::new(&PairedResourceArc {
        arrival: f(vec![[0., 0.], [8., 8.]]),
        increment: f(vec![[1., 2.], [8., 2.]]),
        lower: f(vec![[1., 0.], [8., 0.]]),
        upper: f(vec![[1., 100.], [8., 100.]]),
        waiting: 0,
    })
    .unwrap();
    let root = BoundedCostFront::terminal(
        Window {
            lower: 1.,
            upper: 8.,
        },
        100.,
    )
    .unwrap();
    let cached = root
        .checked_prepend(&arc)
        .expect("restricted identity is exact");
    assert_eq!(cached.upper.value(1.5).unwrap(), Some(98.));
}
#[test]
fn bounded_cost_cache_accepts_exact_unit_identity_on_continuous_domain() {
    let arrival = f(vec![[0., 0.], [1., 1.]]);
    let arc = PreparedRelationArc::new(&PairedResourceArc {
        arrival: arrival.clone(),
        increment: f(vec![[0., 2.], [1., 2.]]),
        lower: f(vec![[0., 0.], [1., 0.]]),
        upper: f(vec![[0., 100.], [1., 100.]]),
        waiting: 0,
    })
    .unwrap();
    let root = BoundedCostFront::terminal(
        Window {
            lower: 0.,
            upper: 1.,
        },
        100.,
    )
    .unwrap();
    let cached = root.checked_prepend(&arc).expect("unit identity is exact");
    for x in [0., f64::from_bits(1), 0.25, 0.5, 1.] {
        assert_eq!(evaluate(&arrival, x).unwrap().to_bits(), x.to_bits());
        assert_eq!(cached.upper.value(x).unwrap(), Some(98.));
    }
}

#[test]
fn bounded_cost_cache_accepts_unit_identity_away_from_zero() {
    for (lo, hi, midpoint) in [
        (2., 3., 2.5),
        (-2., -1., -1.5),
        (8., 16., 12.),
        (-16., -8., -12.),
    ] {
        let arrival = f(vec![[lo, lo], [hi, hi]]);
        let arc = PreparedRelationArc::new(&PairedResourceArc {
            arrival: arrival.clone(),
            increment: f(vec![[lo, 2.], [hi, 2.]]),
            lower: f(vec![[lo, 0.], [hi, 0.]]),
            upper: f(vec![[lo, 100.], [hi, 100.]]),
            waiting: 0,
        })
        .unwrap();
        let root = BoundedCostFront::terminal(
            Window {
                lower: lo,
                upper: hi,
            },
            100.,
        )
        .unwrap();
        let cached = root.checked_prepend(&arc).expect("unit identity is exact");
        for x in [lo, midpoint, hi] {
            assert_eq!(evaluate(&arrival, x).unwrap().to_bits(), x.to_bits());
            assert_eq!(cached.upper.value(x).unwrap(), Some(98.));
        }
    }
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
        // x + 1 over this continuous interval has no universal exact-cache
        // proof; compare the numerical shortcut without relaxing that guard.
        assert!(root.checked_prepend(&arc).is_none());
        let fast = root.prepend_constant(&arc).unwrap();
        let general = root.prepend(&arc).unwrap().unwrap();
        for t in 0..=8 {
            assert_eq!(
                fast.upper.value(f64::from(t)).unwrap(),
                general.upper.value(f64::from(t)).unwrap()
            );
        }
    }
}
