//! Independent finite-control oracles for the exact raw operator subsets.
use super::*;
use crate::graph::coupled::descriptor::{
    FrontOperationPolicy, PairedCostNormalization, PreparedSlopeFunction, SlopeRelationArc,
};
use std::collections::BTreeSet;

#[test]
fn signed_piecewise_slope_operator_certificates_match_integer_control_oracle() {
    for policy in [
        FrontOperationPolicy::StrictDomain,
        FrontOperationPolicy::NativeFrontAlgebra,
    ] {
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
                    for (&increment, &(anchor, first, second)) in increments.iter().zip(&charges) {
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
                            (f64::from(anchor), -6.),
                            (f64::from(first), 0.),
                            (f64::from(second), 6.),
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
                        initial: Window::new(-2., 4.).unwrap(),
                        bounds: Window::new(-3., 5.).unwrap(),
                        increments: increments.iter().map(|&x| f64::from(x)).collect(),
                        functions,
                        policy,
                    }])
                    .unwrap();
                let path: Vec<_> = (0..length).collect();
                let certificate = graph.coupled_kernel().unwrap().certify(&path).unwrap();
                let expected = expected.map(|x| f64::from(x));
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
