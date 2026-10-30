//! Independent finite-control oracles for the exact raw operator subsets.
use super::*;
use crate::graph::coupled::descriptor::{
    FrontierBudget, FrontierOutcome, MachineRootInput, PreparedSlopeFunction, RawAffine,
};

/// One chain arc: tail and head windows, constant base consumption, and
/// constant dependent increment, lower and upper values.
struct ChainArc {
    tail: Window,
    head: Window,
    consumption: f64,
    increment: f64,
    lower: f64,
    upper: f64,
    waiting: i32,
}

/// Build a chain graph with one explicit paired front program.
fn paired_chain(
    root: MachineRootInput<'_>,
    arcs: &[ChainArc],
    mode: PairedResourceMode,
) -> Graph<2> {
    let mut graph = Graph::<2>::new(arcs.len() + 1, 0).unwrap();
    for arc in 0..arcs.len() {
        graph.add_arc(arc, arc + 1, 0.).unwrap();
    }
    graph
        .set_coupled_resources(&[CoupledResource::PairedFront { mode }])
        .unwrap();
    let affine = |value| MachineRootInput::Affine(RawAffine::new(value, 0.).unwrap());
    let base: Vec<_> = arcs
        .iter()
        .map(|arc| {
            PreparedSlopeFunction::from_prepared_pairs(vec![
                (arc.consumption, arc.tail.lower),
                (0., arc.tail.upper),
            ])
            .unwrap()
        })
        .collect();
    let inputs: Vec<_> = arcs
        .iter()
        .zip(&base)
        .enumerate()
        .map(|(physical_arc, (arc, base))| MachineArcFunctionInput {
            physical_arc,
            tail_bounds: arc.tail,
            waiting_bounds: arc.head,
            waiting_rate: arc.waiting,
            base,
            dependent: affine(arc.increment),
            dependent_lower: affine(arc.lower),
            dependent_upper: affine(arc.upper),
        })
        .collect();
    let result = graph.prepare_paired_front_program(
        0,
        &inputs,
        root,
        arcs[0].tail,
        arcs[arcs.len() - 1].head,
        FrontierBudget::default(),
    );
    assert!(
        matches!(result, FrontierOutcome::Complete(())),
        "{result:?}"
    );
    graph
}

#[test]
fn native_alignment_spacing_is_a_proof_condition_not_a_feasibility_epsilon() {
    let unit = 1. / 16384.;
    for policy in [
        FrontOperationPolicy::StrictDomain,
        FrontOperationPolicy::FrontAlgebra,
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
    let initial =
        PreparedSlopeFunction::from_prepared_pairs(vec![(0., 0.), (1., 2.), (-1., 4.)]).unwrap();
    let graph = paired_chain(
        MachineRootInput::Vector(&initial),
        &[ChainArc {
            tail: Window::new(0., 4.).unwrap(),
            head: Window::new(1., 5.).unwrap(),
            consumption: 1.,
            increment: 0.,
            lower: -10.,
            upper: 0.,
            waiting: 1,
        }],
        PairedResourceMode::CostUnbounded,
    );
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
        FrontOperationPolicy::FrontAlgebra,
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
fn paired_nonconstant_certificates_enclose_split_zero_replay() {
    let initial =
        PreparedSlopeFunction::from_prepared_pairs(vec![(3., 0.), (-1., 2.), (1., 4.)]).unwrap();
    for upper in [0, 4, 50] {
        for head_hi in [4, 12] {
            for waiting in [-1, 0, 1] {
                for length in 1..=3 {
                    let head = Window::new(0., f64::from(head_hi)).unwrap();
                    let arcs: Vec<_> = (0..length)
                        .map(|arc| ChainArc {
                            tail: if arc == 0 {
                                Window::new(0., 4.).unwrap()
                            } else {
                                head
                            },
                            head,
                            consumption: 1.,
                            increment: f64::from(arc - 1),
                            lower: -20.,
                            upper: f64::from(upper),
                            waiting,
                        })
                        .collect();
                    let graph = paired_chain(
                        MachineRootInput::Vector(&initial),
                        &arcs,
                        PairedResourceMode::CostUnbounded,
                    );
                    let path: Vec<_> = (0..length as usize).collect();
                    let kernel = graph.coupled_kernel().unwrap();
                    assert_paired_certificate_encloses_replay(kernel, &path);
                }
            }
        }
    }
}

#[test]
fn signed_piecewise_slope_operator_certificates_match_integer_control_oracle() {
    for policy in [
        FrontOperationPolicy::StrictDomain,
        FrontOperationPolicy::FrontAlgebra,
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
fn paired_constant_certificates_enclose_split_zero_replay() {
    for mode in [
        PairedResourceMode::Feasibility,
        PairedResourceMode::CostUnbounded,
        PairedResourceMode::CostBounded,
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
                let mut tail = Window::new(0., 4.).unwrap();
                let arcs: Vec<_> = specs
                    .iter()
                    .map(|&(delta, increment, left, right)| {
                        let head = Window::new(f64::from(left), f64::from(right)).unwrap();
                        let arc = ChainArc {
                            tail,
                            head,
                            consumption: f64::from(delta),
                            increment: f64::from(increment),
                            lower: -5.,
                            upper: f64::from(upper),
                            waiting: 0,
                        };
                        tail = head;
                        arc
                    })
                    .collect();
                let graph = paired_chain(
                    MachineRootInput::Affine(RawAffine::new(2., 0.).unwrap()),
                    &arcs,
                    mode,
                );
                let path: Vec<_> = (0..length as usize).collect();
                assert_paired_certificate_encloses_replay(graph.coupled_kernel().unwrap(), &path);
            }
        }
    }
}

// Certificates enclose every successful split, while replay selects split zero.
// Native split values are checked independently by the captured chain corpus.
fn assert_paired_certificate_encloses_replay(kernel: &CoupledKernel, path: &[usize]) {
    let certificate = kernel.certify(path);
    if certificate.is_none() {
        return; // An unavailable split or arithmetic proof remains unknown.
    }
    let replay = kernel.replay(path).unwrap();
    match certificate {
        Some((true, [lower, upper])) => {
            assert!(lower <= upper);
            if let Some(value) = replay {
                assert!(
                    lower <= value && value <= upper,
                    "{path:?}: {value} outside [{lower}, {upper}]"
                );
            }
        }
        Some((false, _)) => assert_eq!(replay, None),
        None => {} // An unavailable split or arithmetic proof remains unknown.
    }
}
