//! Independent finite-control oracles for the exact raw operator subsets.
use super::*;
use crate::graph::coupled::descriptor::{
    FrontOperationPolicy, PairedCostNormalization, PreparedSlopeFunction, SlopeRelationArc,
};
use std::collections::BTreeSet;

#[test]
fn native_alignment_spacing_is_a_proof_condition_not_a_feasibility_epsilon() {
    let unit = 1. / 16384.;
    for policy in [
        FrontOperationPolicy::StrictDomain,
        FrontOperationPolicy::NativeFrontAlgebra,
    ] {
        let mut graph = Graph::<2>::new(2, 0).unwrap();
        graph.add_arc(0, 1, 0.).unwrap();
        graph
            .set_coupled_resources(&[CoupledResource::SlopeIntervalCost {
                initial: Window::new(0., 2. * unit).unwrap(),
                bounds: Window::new(0., 2. * unit).unwrap(),
                increments: vec![unit],
                functions: vec![PreparedSlopeFunction::from_prepared_pairs(vec![
                    (0., 0.),
                    (0.5, 2. * unit),
                ])
                .unwrap()],
                policy,
            }])
            .unwrap();
        let expected = if policy == FrontOperationPolicy::StrictDomain {
            Some((true, [unit / 2., unit / 2.]))
        } else {
            None
        };
        assert_eq!(graph.coupled_kernel().unwrap().certify(&[0]), expected);
    }
}

#[test]
fn disconnected_native_paired_fronts_do_not_bypass_the_lattice_invariant() {
    let f = |value, lo, slope, hi| {
        PreparedSlopeFunction::from_prepared_pairs(vec![(value, lo), (slope, hi)]).unwrap()
    };
    let mut graph = Graph::<2>::new(2, 0).unwrap();
    graph.add_arc(0, 1, 0.).unwrap();
    graph
        .set_coupled_resources(&[CoupledResource::SlopePaired {
            initial: PreparedSlopeFunction::from_prepared_pairs(vec![
                (0., 0.),
                (1., 2.),
                (-1., 4.),
            ])
            .unwrap(),
            initial_base: Window::new(0., 4.).unwrap(),
            arcs: vec![SlopeRelationArc {
                arrival: f(1., 0., 1., 5.),
                increment: f(0., 0., 0., 5.),
                lower: f(-10., 1., 0., 5.),
                upper: f(0., 1., 0., 5.),
                waiting: 1,
            }],
            mode: PairedResourceMode::CostUnbounded,
            normalization: PairedCostNormalization::AbsoluteFinalValue,
            policy: FrontOperationPolicy::NativeFrontAlgebra,
        }])
        .unwrap();
    // Only the separated endpoints survive the upper filter. A numerical
    // feasible result is not permission to ignore the certificate invariant.
    let kernel = graph.coupled_kernel().unwrap();
    assert_eq!(kernel.replay(&[0]).unwrap(), Some(0.));
    assert_eq!(kernel.certify(&[0]), None);
}

#[test]
fn dyadic_raw_operator_certificates_match_a_shared_control_oracle() {
    for policy in [
        FrontOperationPolicy::StrictDomain,
        FrontOperationPolicy::NativeFrontAlgebra,
    ] {
        for unit in [0.5, 0.25, 0.125, 1. / 1024.] {
            for sign in [-1., 1.] {
                let mut graph = Graph::<2>::new(3, 0).unwrap();
                graph.add_arc(0, 1, 0.).unwrap();
                graph.add_arc(1, 2, 0.).unwrap();
                let functions = vec![
                    PreparedSlopeFunction::from_prepared_pairs(vec![
                        (2. * unit, -4. * unit),
                        (sign, 4. * unit),
                    ])
                    .unwrap(),
                    PreparedSlopeFunction::from_prepared_pairs(vec![
                        (3. * unit, -4. * unit),
                        (-sign, 4. * unit),
                    ])
                    .unwrap(),
                ];
                graph
                    .set_coupled_resources(&[CoupledResource::SlopeIntervalCost {
                        initial: Window::new(-2. * unit, 2. * unit).unwrap(),
                        bounds: Window::new(-4. * unit, 4. * unit).unwrap(),
                        increments: vec![unit, -2. * unit],
                        functions,
                        policy,
                    }])
                    .unwrap();
                // For the SAME initial x, arguments are x+u and x-u.
                // Their opposite slopes cancel, yielding (5+2*sign)*u.
                let expected = (5. + 2. * sign) * unit;
                let kernel = graph.coupled_kernel().unwrap();
                assert_eq!(kernel.certify(&[0, 1]), Some((true, [expected, expected])));
                assert_eq!(kernel.replay(&[0, 1]).unwrap(), Some(expected));
                assert_eq!(kernel.certify_with_budget(&[0, 1], &mut 0, None), None);
            }
        }
    }
}

#[test]
fn native_nonconstant_paired_fronts_and_signed_waiting_match_control_paths() {
    let function = |value, lo, slope, hi| {
        PreparedSlopeFunction::from_prepared_pairs(vec![(value, lo), (slope, hi)]).unwrap()
    };
    for upper in [0, 4, 50] {
        for head_hi in [4, 12] {
            for waiting in [-1, 0, 1] {
                for length in 1..=3 {
                    for normalization in [
                        PairedCostNormalization::AbsoluteFinalValue,
                        PairedCostNormalization::FinalMinusInitialMinimum,
                    ] {
                        // Independent control-path enumeration. All affine vertices
                        // and bound crossings are integers in this unit-slope class.
                        let mut states: BTreeSet<_> =
                            (0..=4).map(|x: i32| (x, 1 + (x - 2).abs())).collect();
                        for arc in 0..length {
                            let mut next = BTreeSet::new();
                            for &(base, dependent) in &states {
                                for head in 0..=head_hi {
                                    if head >= base + 1 {
                                        let value = (dependent + arc - 1
                                            + waiting * (head - base - 1))
                                            .max(-20);
                                        if value <= upper {
                                            next.insert((head, value));
                                        }
                                    }
                                }
                            }
                            states = next;
                        }
                        let baseline =
                            if normalization == PairedCostNormalization::FinalMinusInitialMinimum {
                                1
                            } else {
                                0
                            };
                        let expected = states.iter().map(|p| p.1 - baseline).min().map(f64::from);
                        let mut graph = Graph::<2>::new(length as usize + 1, 0).unwrap();
                        for arc in 0..length as usize {
                            graph.add_arc(arc, arc + 1, 0.).unwrap();
                        }
                        graph
                            .set_coupled_resources(&[CoupledResource::SlopePaired {
                                initial: PreparedSlopeFunction::from_prepared_pairs(vec![
                                    (3., 0.),
                                    (-1., 2.),
                                    (1., 4.),
                                ])
                                .unwrap(),
                                initial_base: Window::new(0., 4.).unwrap(),
                                arcs: (0..length)
                                    .map(|arc| SlopeRelationArc {
                                        arrival: function(1., 0., 1., 20.),
                                        increment: function(f64::from(arc - 1), 0., 0., 20.),
                                        lower: function(-20., 0., 0., f64::from(head_hi)),
                                        upper: function(
                                            f64::from(upper),
                                            0.,
                                            0.,
                                            f64::from(head_hi),
                                        ),
                                        waiting,
                                    })
                                    .collect(),
                                mode: PairedResourceMode::CostUnbounded,
                                normalization,
                                policy: FrontOperationPolicy::NativeFrontAlgebra,
                            }])
                            .unwrap();
                        let path: Vec<_> = (0..length as usize).collect();
                        let kernel = graph.coupled_kernel().unwrap();
                        let certificate = kernel.certify(&path).unwrap();
                        assert_eq!(kernel.replay(&path).unwrap(), expected);
                        match expected {
                    Some(value) => assert_eq!(
                        certificate,
                        (true, [value, value]),
                        "waiting={waiting} length={length} normalization={normalization:?} upper={upper} head_hi={head_hi}"
                    ),
                    None => assert!(!certificate.0),
                }
                    }
                }
            }
        }
    }
}

#[test]
fn signed_piecewise_slope_operator_certificates_match_integer_control_oracle() {
    for policy in [
        FrontOperationPolicy::StrictDomain,
        FrontOperationPolicy::NativeFrontAlgebra,
    ] {
        for unit in [1., 0.5, 0.125, 1. / 1024.] {
            for value_factor in [0.5, 1., 2.] {
                for seed in 0..12 {
                    for length in 1..=6 {
                        let increments: Vec<_> = (0..length)
                            .map(|i| ((seed + 2 * i) % 3) as i32 - 1)
                            .collect();
                        let charges: Vec<_> = (0..length)
                            .map(|i| {
                                (
                                    ((seed + i) % 7) as i32 - 3,
                                    ((2 * seed + i) % 5) as i32 - 2,
                                    ((seed + 3 * i) % 5) as i32 - 2,
                                )
                            })
                            .collect();
                        let mut expected: Option<i32> = None;
                        // Every feasible-domain boundary and translated charge knot
                        // is integral; one of these integer controls attains the
                        // continuous piecewise-affine minimum.
                        for initial in -2..=4 {
                            let (mut x, mut cost, mut feasible) = (initial, 0, true);
                            for (&increment, &(anchor, first, second)) in
                                increments.iter().zip(&charges)
                            {
                                x += increment;
                                if !(-3..=5).contains(&x) {
                                    feasible = false;
                                    break;
                                }
                                cost += if x <= 0 {
                                    anchor + first * (x + 6)
                                } else {
                                    anchor + first * 6 + second * x
                                };
                            }
                            if feasible {
                                expected = Some(expected.map_or(cost, |old| old.min(cost)));
                            }
                        }
                        let functions = charges
                            .iter()
                            .map(|&(anchor, first, second)| {
                                PreparedSlopeFunction::from_prepared_pairs(vec![
                                    (f64::from(anchor) * unit * value_factor, -6. * unit),
                                    (f64::from(first) * value_factor, 0.),
                                    (f64::from(second) * value_factor, 6. * unit),
                                ])
                                .unwrap()
                            })
                            .collect();
                        let mut graph = Graph::<2>::new(length + 1, 0).unwrap();
                        for i in 0..length {
                            graph.add_arc(i, i + 1, 0.).unwrap();
                        }
                        graph
                            .set_coupled_resources(&[CoupledResource::SlopeIntervalCost {
                                initial: Window::new(-2. * unit, 4. * unit).unwrap(),
                                bounds: Window::new(-3. * unit, 5. * unit).unwrap(),
                                increments: increments
                                    .iter()
                                    .map(|&x| f64::from(x) * unit)
                                    .collect(),
                                functions,
                                policy,
                            }])
                            .unwrap();
                        let path: Vec<_> = (0..length).collect();
                        let certificate = graph.coupled_kernel().unwrap().certify(&path).unwrap();
                        let expected = expected.map(|x| f64::from(x) * unit * value_factor);
                        assert_eq!(
                            graph.coupled_kernel().unwrap().replay(&path).unwrap(),
                            expected
                        );
                        match expected {
                            Some(cost) => assert_eq!(
                                certificate,
                                (true, [cost, cost]),
                                "seed={seed} length={length} policy={policy:?}"
                            ),
                            None => assert!(!certificate.0),
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn native_constant_paired_certificates_match_explicit_integer_state_paths() {
    fn function(value: i32, lo: i32, slope: i32, hi: i32) -> PreparedSlopeFunction {
        PreparedSlopeFunction::from_prepared_pairs(vec![
            (f64::from(value), f64::from(lo)),
            (f64::from(slope), f64::from(hi)),
        ])
        .unwrap()
    }
    for mode in [
        PairedResourceMode::Feasibility,
        PairedResourceMode::CostUnbounded,
        PairedResourceMode::CostBounded,
    ] {
        for normalization in [
            PairedCostNormalization::AbsoluteFinalValue,
            PairedCostNormalization::FinalMinusInitialMinimum,
        ] {
            for seed in 0..8 {
                for length in 1..=4 {
                    let upper = if seed % 3 == 0 { 0 } else { 20 };
                    let specs: Vec<_> = (0..length)
                        .map(|i| {
                            let increment = if mode == PairedResourceMode::CostBounded {
                                1 + (seed + i) % 3
                            } else {
                                (seed + 3 * i) % 5 - 2
                            };
                            (
                                1 + (seed + i) % 2,
                                increment,
                                seed % 2,
                                2 + seed % 4 + 2 * i,
                            )
                        })
                        .collect();
                    let mut states: BTreeSet<_> = (0..=4).map(|base| (base, 2)).collect();
                    for &(delta, increment, left, right) in &specs {
                        let mut next = BTreeSet::new();
                        for &(base, dependent) in &states {
                            if !(0..=30).contains(&base) {
                                continue;
                            }
                            for head in left..=right {
                                let dependent = (dependent + increment).max(-5);
                                if head >= base + delta && dependent <= upper {
                                    next.insert((head, dependent));
                                }
                            }
                        }
                        states = next;
                    }
                    let expected = states.iter().map(|p| p.1).min().map(|value| {
                        if mode == PairedResourceMode::Feasibility {
                            0.
                        } else {
                            f64::from(
                                value
                                    - match normalization {
                                        PairedCostNormalization::AbsoluteFinalValue => 0,
                                        PairedCostNormalization::FinalMinusInitialMinimum => 2,
                                    },
                            )
                        }
                    });
                    let arcs = specs
                        .iter()
                        .map(|&(delta, increment, left, right)| SlopeRelationArc {
                            arrival: function(delta, 0, 1, 30),
                            increment: function(increment, 0, 0, 30),
                            lower: function(-5, left, 0, right),
                            upper: function(upper, left, 0, right),
                            waiting: 0,
                        })
                        .collect();
                    let mut graph = Graph::<2>::new(length as usize + 1, 0).unwrap();
                    for i in 0..length as usize {
                        graph.add_arc(i, i + 1, 0.).unwrap();
                    }
                    graph
                        .set_coupled_resources(&[CoupledResource::SlopePaired {
                            initial: function(2, 0, 0, 4),
                            initial_base: Window::new(0., 4.).unwrap(),
                            arcs,
                            mode,
                            normalization,
                            policy: FrontOperationPolicy::NativeFrontAlgebra,
                        }])
                        .unwrap();
                    let path: Vec<_> = (0..length as usize).collect();
                    let certificate = graph.coupled_kernel().unwrap().certify(&path).unwrap();
                    assert_eq!(
                        graph.coupled_kernel().unwrap().replay(&path).unwrap(),
                        expected
                    );
                    match expected {
                        Some(cost)=>assert_eq!(certificate,(true,[cost,cost]),"mode={mode:?} normalization={normalization:?} seed={seed} length={length}"),
                        None=>assert!(!certificate.0),
                    }
                }
            }
        }
    }
}
