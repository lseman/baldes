use super::*;

fn prepared(case: usize, control_points: bool) -> Graph<2> {
    let corpus: serde_json::Value = serde_json::from_str(include_str!(
        "../../../../fixtures/coupled/metasolver-paired-preparation.json"
    ))
    .unwrap();
    let mut problem =
        crate::master::model::format::Model::from_json(&corpus["cases"][case]["model"].to_string())
            .unwrap()
            .problem::<2>()
            .unwrap();
    let mut graph = problem.networks.remove(0).graph;
    if control_points {
        let knots = |f: &PreparedSlopeFunction| {
            let (mut y, mut x) = f.pairs()[0];
            let mut points = vec![[x, y]];
            for &(slope, end) in &f.pairs()[1..] {
                y += slope * (end - x);
                x = end;
                points.push([x, y]);
            }
            PiecewiseLinearFunction::new(points).unwrap()
        };
        let CoupledResource::SlopePaired {
            initial,
            initial_base,
            arcs,
            mode,
            normalization,
            ..
        } = &graph.coupled_kernel().unwrap().definitions()[0]
        else {
            unreachable!()
        };
        let definition = CoupledResource::PairedPiecewiseLinear {
            initial: knots(initial),
            initial_base: *initial_base,
            arcs: arcs
                .iter()
                .map(|a| PairedResourceArc {
                    arrival: knots(&a.arrival),
                    increment: knots(&a.increment),
                    lower: knots(&a.lower),
                    upper: knots(&a.upper),
                    waiting: a.waiting,
                })
                .collect(),
            mode: *mode,
            normalization: *normalization,
        };
        graph.set_coupled_resources(&[definition]).unwrap();
    }
    graph
}

#[test]
fn explicit_paired_controls_certify_the_reference_row_alternative_independently_of_fronts() {
    for case in [0, 18] {
        for control_points in [false, true] {
            for coefficient in [0., 0.5, 2., 1e-308] {
                let mut graph = prepared(case, control_points);
                graph
                    .set_coupled_objective_coefficients(&[(0, coefficient)])
                    .unwrap();
                for suppressed in [false, true] {
                    let mut kernel = graph.coupled_kernel().unwrap().clone();
                    if suppressed {
                        kernel.suppress_costs();
                    }
                    assert_eq!(
                        kernel.selected_paired_values(&[1, 2]).unwrap(),
                        Some(vec![(0, 3.)])
                    );
                    for (base, expected) in [(vec![0., 1., 2.], 3.), (vec![0.5, 1.5, 2.5], 4.)] {
                        let control = PairedControl { resource: 0, base };
                        let encoded = serde_json::to_string(&control).unwrap();
                        assert_eq!(
                            serde_json::from_str::<PairedControl>(&encoded).unwrap(),
                            control
                        );
                        let mut work = 10_000;
                        let certificate = kernel
                            .certify_paired_control(&[1, 2], &control, &mut work, None)
                            .unwrap()
                            .unwrap();
                        assert!(work < 10_000 && certificate.feasible);
                        assert_eq!(certificate.dependent, [expected; 2]);
                        assert!(certificate.certifies_value(expected));
                        assert!(!certificate.certifies_value(expected + 1.));
                    }
                }
            }
        }
    }
}

#[test]
fn explicit_paired_control_rejection_does_not_reject_a_feasible_path() {
    let graph = prepared(0, false);
    let kernel = graph.coupled_kernel().unwrap();
    // Initial domain, earliest arrival, dependent upper bound, head domain.
    for base in [
        vec![-1., 1., 2.],
        vec![0.5, 1., 2.],
        vec![6., 7., 8.],
        vec![0., 1., 9.],
    ] {
        let certificate = kernel
            .certify_paired_control(
                &[1, 2],
                &PairedControl { resource: 0, base },
                &mut 10_000,
                None,
            )
            .unwrap()
            .unwrap();
        assert!(!certificate.feasible);
        assert!(!certificate.certifies_value(0.));
    }
    assert!(kernel.certify(&[1, 2]).unwrap().0);
}

#[test]
fn explicit_paired_control_arguments_and_cooperative_limits_are_not_feasibility_results() {
    let graph = prepared(0, false);
    let kernel = graph.coupled_kernel().unwrap();
    for control in [
        PairedControl {
            resource: 99,
            base: vec![0., 1., 2.],
        },
        PairedControl {
            resource: 0,
            base: vec![],
        },
        PairedControl {
            resource: 0,
            base: vec![0., f64::NAN, 2.],
        },
        PairedControl {
            resource: 0,
            base: vec![0., f64::INFINITY, 2.],
        },
    ] {
        assert!(kernel
            .certify_paired_control(&[1, 2], &control, &mut 10_000, None)
            .is_err());
    }
    let control = PairedControl {
        resource: 0,
        base: vec![0., 1., 2.],
    };
    assert!(kernel
        .certify_paired_control(&[1, 99], &control, &mut 10_000, None)
        .is_err());
    assert!(kernel
        .certify_paired_control(&[1, 2], &control, &mut 0, None)
        .unwrap()
        .is_none());
    assert!(kernel
        .certify_paired_control(&[1, 2], &control, &mut 10_000, Some(Instant::now()))
        .unwrap()
        .is_none());
    let mut partial = 12;
    assert!(kernel
        .certify_paired_control(&[1, 2], &control, &mut partial, None)
        .unwrap()
        .is_none());
    assert!(partial < 12);
    let mut other = graph.clone();
    other
        .set_coupled_resources(&[CoupledResource::CustomResource {
            increments: vec![1.; other.arcs.len()],
            weights: vec![1.; other.arcs.len()],
            forward_bound: None,
            backward_bound: None,
        }])
        .unwrap();
    assert!(other
        .coupled_kernel()
        .unwrap()
        .certify_paired_control(&[1, 2], &control, &mut 10_000, None)
        .is_err());
}

#[test]
fn explicit_paired_waiting_controls_retain_nonminimal_values_in_feasibility_and_cost_modes() {
    for mode in [
        PairedResourceMode::Feasibility,
        PairedResourceMode::CostBounded,
        PairedResourceMode::CostUnbounded,
    ] {
        let graph = super::tests::graph(mode, [1, 2]);
        let kernel = graph.coupled_kernel().unwrap();
        for (base, expected) in [(vec![0., 5., 6.], 17.), (vec![0., 3., 6.], 19.)] {
            assert!(kernel
                .certify_paired_control(
                    &[0, 1],
                    &PairedControl { resource: 0, base },
                    &mut 10_000,
                    None
                )
                .unwrap()
                .unwrap()
                .certifies_value(expected));
        }
        // Empty source control reports physical initial value, not route cost.
        assert!(kernel
            .certify_paired_control(
                &[],
                &PairedControl {
                    resource: 0,
                    base: vec![0.]
                },
                &mut 10_000,
                None
            )
            .unwrap()
            .unwrap()
            .certifies_value(10.));
    }
    let graph = super::tests::graph(PairedResourceMode::CostUnbounded, [-5, -2]);
    assert!(graph
        .coupled_kernel()
        .unwrap()
        .certify_paired_control(
            &[0, 1],
            &PairedControl {
                resource: 0,
                base: vec![0., 8., 10.]
            },
            &mut 10_000,
            None,
        )
        .unwrap()
        .unwrap()
        .certifies_value(0.));
}

fn large_control(initial: f64, increment: f64) -> Graph<2> {
    let mut graph = Graph::new(2, 0).unwrap();
    graph.add_arc(0, 1, 0.).unwrap();
    let constant = |y| PiecewiseLinearFunction::new(vec![[0., y], [2., y]]).unwrap();
    graph
        .set_coupled_resources(&[CoupledResource::PairedPiecewiseLinear {
            initial: constant(initial),
            initial_base: Window::new(0., 0.).unwrap(),
            arcs: vec![PairedResourceArc {
                arrival: PiecewiseLinearFunction::new(vec![[0., 1.], [1., 2.]]).unwrap(),
                increment: constant(increment),
                lower: constant(0.),
                upper: constant(f64::MAX),
                waiting: 0,
            }],
            mode: PairedResourceMode::CostUnbounded,
            normalization: PairedCostNormalization::AbsoluteFinalValue,
        }])
        .unwrap();
    graph
}

#[test]
fn explicit_paired_control_enclosure_does_not_claim_a_rounded_value_is_attainable() {
    let graph = large_control(9_007_199_254_740_992., 1.);
    let certificate = graph
        .coupled_kernel()
        .unwrap()
        .certify_paired_control(
            &[0],
            &PairedControl {
                resource: 0,
                base: vec![0., 1.],
            },
            &mut 10_000,
            None,
        )
        .unwrap()
        .unwrap();
    assert!(certificate.feasible);
    assert_eq!(
        certificate.dependent,
        [9_007_199_254_740_992., 9_007_199_254_740_994.]
    );
    assert!(!certificate.certifies_value(certificate.dependent[0]));
    assert!(!certificate.certifies_value(certificate.dependent[1]));
}

#[test]
fn explicit_paired_control_overflow_is_unknown() {
    let graph = large_control(1e308, 1e308);
    assert!(graph
        .coupled_kernel()
        .unwrap()
        .certify_paired_control(
            &[0],
            &PairedControl {
                resource: 0,
                base: vec![0., 1.]
            },
            &mut 10_000,
            None,
        )
        .unwrap()
        .is_none());
}

#[test]
fn explicit_paired_control_nonrepresentable_slope_keeps_boundary_uncertainty() {
    let mut graph = large_control(0., 1.);
    let mut definitions = graph.coupled_kernel().unwrap().definitions().to_vec();
    let CoupledResource::PairedPiecewiseLinear {
        initial_base, arcs, ..
    } = &mut definitions[0]
    else {
        unreachable!()
    };
    *initial_base = Window::new(1., 1.).unwrap();
    arcs[0].arrival = PiecewiseLinearFunction::new(vec![[0., 1.], [3., 5.]]).unwrap();
    arcs[0].lower = PiecewiseLinearFunction::new(vec![[0., 0.], [6., 0.]]).unwrap();
    arcs[0].upper = PiecewiseLinearFunction::new(vec![[0., f64::MAX], [6., f64::MAX]]).unwrap();
    graph.set_coupled_resources(&definitions).unwrap();
    let boundary = f64::from_bits((7_f64 / 3.).to_bits() - 1);
    let kernel = graph.coupled_kernel().unwrap();
    // This rounded head is below exact 7/3; a straddling enclosure must not
    // authorize it. A head beyond the enclosure is rigorously reachable.
    assert!(kernel
        .certify_paired_control(
            &[0],
            &PairedControl {
                resource: 0,
                base: vec![1., boundary]
            },
            &mut 10_000,
            None
        )
        .unwrap()
        .is_none());
    assert!(kernel
        .certify_paired_control(
            &[0],
            &PairedControl {
                resource: 0,
                base: vec![1., 3.]
            },
            &mut 10_000,
            None
        )
        .unwrap()
        .unwrap()
        .certifies_value(1.));
}

#[test]
fn controlled_replay_and_cost_certificates_keep_other_resource_charges_and_normalization() {
    for case in [0, 18] {
        for coefficient in [0., 0.5, 1., 2., 1e-308] {
            let mut graph = prepared(case, false);
            graph.set_front_program_domain(false).unwrap();
            let mut definitions = graph.coupled_kernel().unwrap().definitions().to_vec();
            if let CoupledResource::SlopePaired { policy, .. } = &mut definitions[0] {
                *policy = FrontOperationPolicy::StrictDomain;
            }
            definitions.push(CoupledResource::CumulativeCost {
                initial: 2.,
                increments: vec![0., 2., 1., 0., 0.],
                weights: vec![0., 1., 2., 0., 0.],
                sink_upper: None,
                source_upper: None,
            });
            graph.set_coupled_resources(&definitions).unwrap();
            graph
                .set_coupled_objective_coefficients(&[(0, coefficient), (1, -1.)])
                .unwrap();
            let controls = [PairedControl {
                resource: 0,
                base: vec![0.5, 1.5, 2.5],
            }];
            for suppressed in [false, true] {
                let mut kernel = graph.coupled_kernel().unwrap().clone();
                if suppressed {
                    kernel.suppress_costs();
                }
                let result = kernel
                    .replay_with_controls(&[1, 2], &controls)
                    .unwrap()
                    .unwrap();
                let expected = if suppressed {
                    0.
                } else {
                    -14. + 4. * coefficient
                };
                assert_eq!(result.cost, expected);
                assert_eq!(result.paired_values, vec![(0, 4.)]);
                let (valid, cost) = kernel
                    .certify_with_controls(&[1, 2], &controls, &mut 1_000_000, None)
                    .unwrap();
                assert!(valid && cost[0] <= expected && expected <= cost[1]);
            }
        }
    }
    let mut graph = super::tests::graph(PairedResourceMode::CostBounded, [1, 2]);
    graph
        .set_coupled_objective_coefficients(&[(0, 2.)])
        .unwrap();
    let kernel = graph.coupled_kernel().unwrap();
    let controls = [PairedControl {
        resource: 0,
        base: vec![0., 3., 6.],
    }];
    let result = kernel
        .replay_with_controls(&[0, 1], &controls)
        .unwrap()
        .unwrap();
    assert_eq!(result.paired_values, vec![(0, 19.)]);
    assert_eq!(result.cost, 18.);
    assert_eq!(
        kernel.certify_with_controls(&[0, 1], &controls, &mut 1_000_000, None),
        Some((true, [18.; 2]))
    );
}

#[test]
fn controlled_costs_require_valid_ordered_controls_and_the_source_model() {
    let mut graph = prepared(0, false);
    let control = PairedControl {
        resource: 0,
        base: vec![0.5, 1.5, 2.5],
    };
    assert!(graph
        .coupled_kernel()
        .unwrap()
        .replay_with_controls(&[1, 2], &[control.clone()])
        .is_err());
    assert!(graph
        .coupled_kernel()
        .unwrap()
        .certify_with_controls(&[1, 2], &[control.clone()], &mut 1_000_000, None)
        .is_none());
    graph.set_front_program_domain(false).unwrap();
    let kernel = graph.coupled_kernel().unwrap();
    assert!(kernel
        .replay_with_controls(&[1, 2], &[control.clone(), control.clone()])
        .is_err());
    assert!(kernel
        .certify_with_controls(
            &[1, 2],
            &[control.clone(), control.clone()],
            &mut 1_000_000,
            None
        )
        .is_none());
    assert!(kernel
        .certify_with_controls(&[1, 2], &[control.clone()], &mut 0, None)
        .is_none());
    let invalid = PairedControl {
        resource: 0,
        base: vec![0.5, 1., 2.],
    };
    assert_eq!(
        kernel.certify_with_controls(&[1, 2], &[invalid], &mut 1_000_000, None),
        Some((false, [0.; 2]))
    );
    assert_eq!(
        kernel.certify_with_controls(&[1, 2], &[], &mut 1_000_000, None),
        kernel.certify(&[1, 2])
    );
}

#[test]
fn controlled_numeric_replay_respects_the_callers_proof_budget_and_deadline() {
    for control_points in [false, true] {
        let mut graph = prepared(0, control_points);
        graph.set_front_program_domain(false).unwrap();
        let kernel = graph.coupled_kernel().unwrap();
        let controls = [PairedControl {
            resource: 0,
            base: vec![0.5, 1.5, 2.5],
        }];
        assert!(kernel
            .replay_with_controls_using(&[1, 2], &controls, &mut 0, None)
            .unwrap()
            .is_none());
        assert!(kernel
            .replay_with_controls_using(
                &[1, 2],
                &controls,
                &mut 1_000_000,
                Some(std::time::Instant::now())
            )
            .unwrap()
            .is_none());
        let mut work = 1_000_000;
        assert!(kernel
            .replay_with_controls_using(&[1, 2], &controls, &mut work, None)
            .unwrap()
            .is_some());
        assert!(work < 1_000_000);
    }
}

#[test]
fn equivalent_control_witnesses_keep_the_ordinary_objective_addition_order() {
    for control_points in [false, true] {
        let mut graph = prepared(0, control_points);
        let first = graph.coupled_kernel().unwrap().definitions()[0].clone();
        let other = prepared(0, !control_points);
        let second = other.coupled_kernel().unwrap().definitions()[0].clone();
        let count = graph.arcs.len();
        graph
            .set_coupled_resources(&[
                CoupledResource::CustomResource {
                    increments: vec![1.; count],
                    weights: (0..count)
                        .map(|arc| if arc == 1 { -1e16 } else { 0. })
                        .collect(),
                    forward_bound: None,
                    backward_bound: None,
                },
                first,
                second,
            ])
            .unwrap();
        graph.set_front_program_domain(false).unwrap();
        graph
            .set_coupled_objective_coefficients(&[(1, 1. / 3.), (2, 1e16 / 3.)])
            .unwrap();
        let kernel = graph.coupled_kernel().unwrap();
        let ordinary = kernel.replay_with_controls(&[1, 2], &[]).unwrap().unwrap();
        assert_eq!(ordinary.paired_values, [(1, 3.), (2, 3.)]);
        // Slope charges precede knot charges. The small term disappears when
        // it is added before cancellation, and survives when added afterward.
        assert_eq!(ordinary.cost, if control_points { 1. } else { 0. });
        for selected in [vec![1], vec![2], vec![1, 2]] {
            let controls: Vec<_> = selected
                .into_iter()
                .map(|resource| PairedControl {
                    resource,
                    base: vec![0., 1., 2.],
                })
                .collect();
            let explicit = kernel
                .replay_with_controls(&[1, 2], &controls)
                .unwrap()
                .unwrap();
            assert_eq!(explicit.paired_values, ordinary.paired_values);
            assert_eq!(
                explicit.cost.to_bits(),
                ordinary.cost.to_bits(),
                "{control_points}: {controls:?}"
            );
        }
    }
}

#[test]
fn normalization_baseline_queries_share_work_deadline_and_cost_suppression() {
    for points in [false, true] {
        let graph = prepared(0, points);
        let kernel = graph.coupled_kernel().unwrap();
        assert!(kernel.paired_initial_minimum(0, &mut 0, None).is_none());
        assert!(kernel
            .paired_initial_minimum(0, &mut 1_000_000, Some(std::time::Instant::now()))
            .is_none());
        let mut work = 1_000_000;
        assert_eq!(
            kernel.paired_initial_minimum(0, &mut work, None),
            Some([0., 0.])
        );
        assert!(work < 1_000_000);
        let mut suppressed = kernel.clone();
        suppressed.suppress_costs();
        assert_eq!(
            suppressed.paired_initial_minimum(0, &mut 1_000_000, None),
            Some([0., 0.])
        );
    }
}
