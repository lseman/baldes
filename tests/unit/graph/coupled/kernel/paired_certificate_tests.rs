use super::*;

fn constant(lo: f64, hi: f64, value: f64) -> PiecewiseLinearFunction {
    PiecewiseLinearFunction::new(vec![[lo, value], [hi, value]]).unwrap()
}

pub(super) fn graph(mode: PairedResourceMode, waiting: [i32; 2]) -> Graph<2> {
    let mut graph = Graph::<2>::new(3, 0).unwrap();
    graph.add_arc(0, 1, 0.).unwrap();
    graph.add_arc(1, 2, 0.).unwrap();
    graph
        .set_coupled_resources(&[CoupledResource::PairedPiecewiseLinear {
            normalization: crate::graph::coupled::PairedCostNormalization::FinalMinusInitialMinimum,
            initial: constant(0., 1., 10.),
            initial_base: Window::new(0., 0.).unwrap(),
            arcs: [(3., 8., 2.), (6., 10., 1.)]
                .into_iter()
                .enumerate()
                .map(|(i, (lo, hi, increment))| PairedResourceArc {
                    arrival: PiecewiseLinearFunction::new(vec![[0., 1.], [10., 11.]]).unwrap(),
                    increment: constant(0., 10., increment),
                    lower: constant(lo, hi, 0.),
                    upper: constant(lo, hi, 100.),
                    waiting: waiting[i],
                })
                .collect(),
            mode,
        }])
        .unwrap();
    graph
}

#[test]
fn explicit_waiting_controls_bound_the_independent_optimum_in_all_modes() {
    for mode in [
        PairedResourceMode::Feasibility,
        PairedResourceMode::CostUnbounded,
        PairedResourceMode::CostBounded,
    ] {
        let graph = graph(mode, [1, 2]);
        let kernel = graph.coupled_kernel().unwrap();
        let (feasible, interval) = kernel.certify(&[0, 1]).unwrap();
        assert!(feasible);
        // Integer vertices of this two-control waiting polygon include its
        // continuous optimum. This oracle uses no production frontier code.
        let optimum = if mode == PairedResourceMode::Feasibility {
            0.
        } else {
            (3..=8)
                .flat_map(|first| {
                    ((first + 1).max(6)..=10).map(move |second| {
                        2. + f64::from(first - 1) + 1. + 2. * f64::from(second - first - 1)
                    })
                })
                .reduce(f64::min)
                .unwrap()
        };
        assert!(
            interval[0] <= optimum && optimum <= interval[1],
            "{interval:?}"
        );
        assert!((interval[1] - optimum).abs() < 1e-10);
        assert_eq!(kernel.certify(&[]), Some((true, [0., 0.])));
    }
}

#[test]
fn constant_charge_has_an_exact_certificate_and_phase_one_retains_feasibility() {
    let graph = graph(PairedResourceMode::CostBounded, [0, 0]);
    assert_eq!(
        graph.coupled_kernel().unwrap().certify(&[0, 1]),
        Some((true, [3., 3.]))
    );
    let mut definitions = graph.coupled_kernel().unwrap().definitions().to_vec();
    let CoupledResource::PairedPiecewiseLinear { arcs, .. } = &mut definitions[0] else {
        unreachable!()
    };
    arcs[1].upper = constant(6., 10., 12.);
    arcs[0].upper = constant(3., 8., 12.);
    let mut infeasible = graph.clone();
    infeasible.set_coupled_resources(&definitions).unwrap();
    assert_eq!(
        infeasible.coupled_kernel().unwrap().certify(&[0, 1]),
        Some((false, [0., 0.]))
    );
}

#[test]
fn piecewise_and_signed_waiting_witnesses_do_not_assume_constant_increments() {
    let mut graph = Graph::<2>::new(3, 0).unwrap();
    graph.add_arc(0, 1, 0.).unwrap();
    graph.add_arc(1, 2, 0.).unwrap();
    graph
        .set_coupled_resources(&[CoupledResource::PairedPiecewiseLinear {
            normalization: crate::graph::coupled::PairedCostNormalization::FinalMinusInitialMinimum,
            initial: PiecewiseLinearFunction::new(vec![[0., 4.], [2., 2.]]).unwrap(),
            initial_base: Window::new(0., 2.).unwrap(),
            arcs: vec![
                PairedResourceArc {
                    arrival: PiecewiseLinearFunction::new(vec![[0., 1.], [2., 4.]]).unwrap(),
                    increment: PiecewiseLinearFunction::new(vec![[0., 1.], [1., -2.], [2., 1.]])
                        .unwrap(),
                    lower: PiecewiseLinearFunction::new(vec![[1., -5.], [3., 0.], [5., -5.]])
                        .unwrap(),
                    upper: constant(1., 5., 20.),
                    waiting: -1,
                },
                PairedResourceArc {
                    arrival: PiecewiseLinearFunction::new(vec![[1., 2.], [5., 6.]]).unwrap(),
                    increment: PiecewiseLinearFunction::new(vec![[1., 2.], [3., -1.], [5., 2.]])
                        .unwrap(),
                    lower: constant(2., 8., -10.),
                    upper: constant(2., 8., 20.),
                    waiting: 1,
                },
            ],
            mode: PairedResourceMode::Feasibility,
        }])
        .unwrap();
    assert_eq!(
        graph.coupled_kernel().unwrap().certify(&[0, 1]),
        Some((true, [0., 0.]))
    );
}

#[test]
fn uncertain_rounding_boundary_does_not_certify_a_numerical_route() {
    let mut graph = graph(PairedResourceMode::CostUnbounded, [0, 0]);
    let mut definitions = graph.coupled_kernel().unwrap().definitions().to_vec();
    let CoupledResource::PairedPiecewiseLinear { initial, arcs, .. } = &mut definitions[0] else {
        unreachable!()
    };
    *initial = constant(0., 1., 0.);
    for (i, arc) in arcs.iter_mut().enumerate() {
        arc.increment = constant(0., 10., if i == 0 { 0.1 } else { 0.2 });
        arc.upper = constant(
            if i == 0 { 3. } else { 6. },
            if i == 0 { 8. } else { 10. },
            0.3,
        );
    }
    graph.set_coupled_resources(&definitions).unwrap();
    assert_eq!(graph.coupled_kernel().unwrap().certify(&[0, 1]), None);
}

#[test]
fn strict_slope_pairs_and_exact_native_subset_keep_ordered_encoding() {
    use crate::graph::coupled::descriptor::{PairedCostNormalization, SlopeRelationArc};
    let f = |value, start, slope, end| {
        PreparedSlopeFunction::from_prepared_pairs(vec![(value, start), (slope, end)]).unwrap()
    };
    for normalization in [
        PairedCostNormalization::AbsoluteFinalValue,
        PairedCostNormalization::FinalMinusInitialMinimum,
    ] {
        let mut graph = Graph::<2>::new(2, 0).unwrap();
        graph.add_arc(0, 1, 0.).unwrap();
        let mut definition = CoupledResource::SlopePaired {
            initial: f(10., 0., 0., 1.),
            initial_base: Window::new(0., 0.).unwrap(),
            arcs: vec![SlopeRelationArc {
                arrival: f(1., 0., 1., 2.),
                increment: f(2., 0., 0., 2.),
                lower: f(0., 1., 0., 3.),
                upper: f(100., 1., 0., 3.),
                waiting: 0,
            }],
            mode: PairedResourceMode::CostBounded,
            normalization,
            policy: FrontOperationPolicy::StrictDomain,
        };
        graph.set_coupled_resources(&[definition.clone()]).unwrap();
        let expected = match normalization {
            PairedCostNormalization::AbsoluteFinalValue => 12.,
            PairedCostNormalization::FinalMinusInitialMinimum => 2.,
        };
        assert_eq!(
            graph.coupled_kernel().unwrap().certify(&[0]),
            Some((true, [expected, expected]))
        );
        let CoupledResource::SlopePaired { policy, .. } = &mut definition else {
            unreachable!()
        };
        *policy = FrontOperationPolicy::FrontAlgebra;
        graph.set_coupled_resources(&[definition.clone()]).unwrap();
        assert_eq!(
            graph.coupled_kernel().unwrap().certify(&[0]),
            Some((true, [expected, expected]))
        );
        let CoupledResource::SlopePaired { arcs, .. } = &mut definition else {
            unreachable!()
        };
        arcs[0].waiting = 1;
        graph.set_coupled_resources(&[definition]).unwrap();
        assert_eq!(
            graph.coupled_kernel().unwrap().certify(&[0]),
            Some((true, [expected, expected]))
        );
    }
}

#[test]
fn a_witness_budget_failure_does_not_prove_infeasibility() {
    let graph = graph(PairedResourceMode::CostBounded, [1, 2]);
    let definition = &graph.coupled_kernel().unwrap().definitions()[0];
    let CoupledResource::PairedPiecewiseLinear {
        initial,
        initial_base,
        arcs,
        mode,
        ..
    } = definition
    else {
        unreachable!()
    };
    let functions: Vec<_> = arcs
        .iter()
        .map(|a| ArcFunctions {
            arrival: Function::Knots(&a.arrival),
            increment: Function::Knots(&a.increment),
            lower: Function::Knots(&a.lower),
            upper: Function::Knots(&a.upper),
            waiting: a.waiting,
        })
        .collect();
    assert_eq!(
        certify(
            Function::Knots(initial),
            *initial_base,
            &functions,
            *mode,
            descriptor::PairedCostNormalization::FinalMinusInitialMinimum,
            1.,
            &mut Work(1, None)
        ),
        None
    );
}

#[test]
fn every_float_control_in_small_domains_agrees_with_the_certificate() {
    // There are exactly 257 binary64 controls in each domain. Enumerating
    // every pair, rather than a real-valued grid, is an independent oracle for
    // the literal interpolation/arrival/waiting program on this finite domain.
    let source = |i| f64::from_bits(1f64.to_bits() + i);
    let head = |i| f64::from_bits(2f64.to_bits() + i);
    let initial = vec![[source(0), 3.], [source(128), 1.], [source(256), 2.]];
    let arrival = vec![[source(0), head(0)], [source(256), head(256)]];
    let increment = vec![[source(0), -1.], [source(128), 3.], [source(256), -2.]];
    // Deliberately do not call the production evaluator or relation code.
    let evaluate = |points: &[[f64; 2]], x: f64| {
        if let Some(p) = points.iter().find(|p| p[0] == x) {
            return p[1];
        }
        let p = points
            .windows(2)
            .find(|p| p[0][0] < x && x < p[1][0])
            .unwrap();
        let t = (x - p[0][0]) / (p[1][0] - p[0][0]);
        (p[0][1] + t * (p[1][1] - p[0][1])).clamp(p[0][1].min(p[1][1]), p[0][1].max(p[1][1]))
    };
    let baseline = (0..=256)
        .map(|i| evaluate(&initial, source(i)))
        .reduce(f64::min)
        .unwrap();
    for waiting in [-2, -1, 0, 1, 2] {
        for upper in [-3., 10.] {
            let mut optimum: Option<f64> = None;
            for i in 0..=256 {
                let t = source(i);
                let y = evaluate(&arrival, t);
                for j in 0..=256 {
                    let x = head(j);
                    if x < y {
                        continue;
                    }
                    let d = (evaluate(&initial, t)
                        + evaluate(&increment, t)
                        + f64::from(waiting) * (x - y))
                        .max(-4.);
                    if d <= upper {
                        optimum = Some(optimum.map_or(d - baseline, |v| v.min(d - baseline)));
                    }
                }
            }
            let mut graph = Graph::<2>::new(2, 0).unwrap();
            graph.add_arc(0, 1, 0.).unwrap();
            graph
                .set_coupled_resources(&[CoupledResource::PairedPiecewiseLinear {
                    normalization:
                        crate::graph::coupled::PairedCostNormalization::FinalMinusInitialMinimum,
                    initial: PiecewiseLinearFunction::new(initial.clone()).unwrap(),
                    initial_base: Window::new(source(0), source(256)).unwrap(),
                    arcs: vec![PairedResourceArc {
                        arrival: PiecewiseLinearFunction::new(arrival.clone()).unwrap(),
                        increment: PiecewiseLinearFunction::new(increment.clone()).unwrap(),
                        lower: constant(head(0), head(256), -4.),
                        upper: constant(head(0), head(256), upper),
                        waiting,
                    }],
                    mode: PairedResourceMode::CostUnbounded,
                }])
                .unwrap();
            let (feasible, interval) = graph.coupled_kernel().unwrap().certify(&[0]).unwrap();
            match optimum {
                None => assert!(!feasible),
                Some(value) => {
                    assert!(feasible);
                    assert!(
                        interval[0] <= value && value <= interval[1],
                        "waiting={waiting}, upper={upper}, optimum={value}, interval={interval:?}"
                    );
                }
            }
        }
    }
}

#[test]
fn shared_work_and_expired_deadlines_never_publish_a_route_proof() {
    let graph = graph(PairedResourceMode::CostBounded, [0, 0]);
    let kernel = graph.coupled_kernel().unwrap();
    assert_eq!(kernel.certify_with_budget(&[0, 1], &mut 0, None), None);
    let mut work = 1_000_000;
    let expected = kernel.certify_with_budget(&[0, 1], &mut work, None);
    assert_eq!(expected, Some((true, [3., 3.])));
    let spent = 1_000_000 - work;
    assert!(spent > 1);
    let mut shared = 2 * spent - 1;
    assert_eq!(
        kernel.certify_with_budget(&[0, 1], &mut shared, None),
        expected
    );
    assert_eq!(kernel.certify_with_budget(&[0, 1], &mut shared, None), None);
    assert!(shared < spent);
    let expired = Instant::now()
        .checked_sub(std::time::Duration::from_secs(1))
        .unwrap();
    let mut remaining = 1_000_000;
    assert_eq!(
        kernel.certify_with_budget(&[0, 1], &mut remaining, Some(expired)),
        None
    );
    assert_eq!(remaining, 1_000_000);
    assert!(Work(10, Some(expired)).take(1).is_none());
}

#[test]
fn weighted_control_point_paired_costs_keep_physical_values_and_cached_join_semantics() {
    for mode in [
        PairedResourceMode::CostUnbounded,
        PairedResourceMode::CostBounded,
    ] {
        for normalization in [
            PairedCostNormalization::FinalMinusInitialMinimum,
            PairedCostNormalization::AbsoluteFinalValue,
        ] {
            for coefficient in [0., 0.5, 1., 2., 1e-308] {
                let mut graph = graph(mode, [1, 2]);
                let mut definitions = graph.coupled_kernel().unwrap().definitions().to_vec();
                if let CoupledResource::PairedPiecewiseLinear {
                    normalization: saved,
                    ..
                } = &mut definitions[0]
                {
                    *saved = normalization;
                }
                graph.set_coupled_resources(&definitions).unwrap();
                graph
                    .set_coupled_objective_coefficients(&[(0, coefficient)])
                    .unwrap();
                let kernel = graph.coupled_kernel().unwrap();
                let expected = (if normalization == PairedCostNormalization::AbsoluteFinalValue {
                    17.
                } else {
                    7.
                }) * coefficient;
                assert_eq!(kernel.replay(&[]).unwrap(), Some(0.));
                assert_eq!(kernel.replay(&[0, 1]).unwrap(), Some(expected));
                assert_eq!(
                    kernel.selected_paired_values(&[0, 1]).unwrap(),
                    Some(vec![(0, 17.)])
                );
                let (feasible, bound) = kernel.certify(&[0, 1]).unwrap();
                assert!(feasible && bound[0] <= expected && expected <= bound[1]);
                for split in 0..=2 {
                    let mut f = kernel.root(true);
                    for arc in 0..split {
                        f = kernel.extend(&f, arc, true).unwrap().unwrap().0;
                    }
                    let mut b = kernel.root(false);
                    for arc in (split..2).rev() {
                        b = kernel.extend(&b, arc, false).unwrap().unwrap().0;
                    }
                    assert_eq!(
                        kernel.join_original_objective(&f, &b).unwrap(),
                        Some(expected)
                    );
                    let correction = kernel.join(&f, &b).unwrap().unwrap();
                    assert_eq!(
                        f.resource_objective + b.resource_objective + correction,
                        expected
                    );
                }
            }
        }
    }
    assert!(graph(PairedResourceMode::Feasibility, [1, 2])
        .set_coupled_objective_coefficients(&[(0, 0.)])
        .is_err());
}
