use super::*;

#[test]
fn compatible_constant_machine_paths_use_source_normalization_and_coefficients() {
    for normalization in [
        PairedCostNormalization::AbsoluteFinalValue,
        PairedCostNormalization::FinalMinusInitialMinimum,
    ] {
        for coefficient in [0., 0.25, 1., 2.] {
            let mut graph = graph(PairedResourceMode::CostUnbounded, normalization);
            let mut definition = graph.coupled_kernel().unwrap().definitions()[0].clone();
            if let CoupledResource::SlopePaired { policy, .. } = &mut definition {
                *policy = FrontOperationPolicy::FrontAlgebra;
            }
            graph.set_coupled_resources(&[definition]).unwrap();
            graph
                .set_coupled_objective_coefficients(&[(0, coefficient)])
                .unwrap();
            let input = PreparedMachineArc {
                tail_limit: 2.,
                head_limit: 2.,
                waiting_lower: 0.,
                waiting_upper: 2.,
                waiting_rate: 0,
                forward_mapping: function(0., 1.),
                inverse: function(0., 1.),
                sum_vector: function(1., 0.),
                dependent_lower: function(0., 0.),
                dependent_upper: function(100., 0.),
                backward: Some(PreparedMachineBackwardInputs {
                    tail_lower: 0.,
                    tail_upper: 2.,
                    forward_at_tail_lower: 0.,
                    mapped_upper: 2.,
                    dependent: function(1., 0.),
                }),
            };
            graph
                .set_paired_machine_inputs(0, Some(&[input.clone(), input]))
                .unwrap();
            graph
                .set_paired_machine_root(0, Some(&function(5., 0.)))
                .unwrap();
            assert!(matches!(
                graph.install_paired_machine_backward_root(
                    0,
                    Window::new(0., 2.).unwrap(),
                    descriptor::FrontierBudget::default()
                ),
                descriptor::FrontierOutcome::Complete(())
            ));
            graph.set_front_program_domain(true).unwrap();
            let kernel = graph.coupled_kernel().unwrap();
            let root = kernel.root(true);
            let (first, delta1) = kernel.extend(&root, 0, true).unwrap().unwrap();
            let (second, delta2) = kernel.extend(&first, 1, true).unwrap().unwrap();
            let expected = coefficient
                * match normalization {
                    PairedCostNormalization::AbsoluteFinalValue => 7.,
                    PairedCostNormalization::FinalMinusInitialMinimum => 2.,
                };
            assert_eq!(second.resource_objective, expected);
            assert_eq!(delta1 + delta2, expected);
            assert!(second.slope_relations[0].machine_history.is_some());
            let backward_root = kernel.root(false);
            let suffix = kernel.extend(&backward_root, 1, false).unwrap().unwrap().0;
            let entire_suffix = kernel.extend(&suffix, 0, false).unwrap().unwrap().0;
            for (forward, backward) in [
                (&root, &entire_suffix),
                (&first, &suffix),
                (&second, &backward_root),
            ] {
                assert_eq!(
                    kernel.join_resource_objective(forward, backward).unwrap(),
                    Some(expected)
                );
            }
            for search in [0., 1., 2.] {
                let mut priced = graph.clone();
                for arc in &mut priced.arcs {
                    arc.cost = -10.;
                    arc.reduced_cost = -10.;
                }
                let mut solver = crate::pricing::Solver::<2>::new(3, 0).unwrap();
                solver.graph = priced;
                solver
                    .set_option("BALDESuseBidirectionalSearch", search)
                    .unwrap();
                let result = solver.price_phase(0, 20, false).unwrap();
                assert_eq!(
                    result.minimum_reduced_cost,
                    -20. + expected,
                    "search={search}, coefficient={coefficient}, normalization={normalization:?}"
                );
                assert!(result
                    .columns
                    .iter()
                    .any(|route| route.arcs == [0, 1] && route.original_cost == -20. + expected));
            }
            if coefficient != 0. {
                graph
                    .set_paired_machine_root(0, Some(&function(4., 0.)))
                    .unwrap();
                let kernel = graph.coupled_kernel().unwrap();
                assert!(kernel.extend(&kernel.root(true), 0, true).is_err());
                let mut inputs = kernel.slope_paired[0]
                    .machine_inputs
                    .as_ref()
                    .unwrap()
                    .to_vec();
                inputs[1].backward.as_mut().unwrap().dependent = function(2., 0.);
                graph.set_paired_machine_inputs(0, Some(&inputs)).unwrap();
                graph
                    .set_paired_machine_root(0, Some(&function(5., 0.)))
                    .unwrap();
                assert!(matches!(
                    graph.install_paired_machine_backward_root(
                        0,
                        Window::new(0., 2.).unwrap(),
                        descriptor::FrontierBudget::default()
                    ),
                    descriptor::FrontierOutcome::Complete(())
                ));
                let kernel = graph.coupled_kernel().unwrap();
                assert!(kernel.extend(&kernel.root(false), 1, false).is_err());
            }
        }
    }
}

#[test]
fn explicit_machine_sidecars_do_not_define_the_declared_source_objective() {
    let mut graph = graph(
        PairedResourceMode::CostUnbounded,
        PairedCostNormalization::AbsoluteFinalValue,
    );
    let mut definition = graph.coupled_kernel().unwrap().definitions()[0].clone();
    if let CoupledResource::SlopePaired { policy, .. } = &mut definition {
        *policy = FrontOperationPolicy::FrontAlgebra;
    }
    graph.set_coupled_resources(&[definition]).unwrap();
    // The declared source recurrence is 5+1+1=7. Explicit sidecar operands
    // deliberately describe a different machine recurrence, 5+2+2=9.
    let input = PreparedMachineArc {
        tail_limit: 2.,
        head_limit: 2.,
        waiting_lower: 0.,
        waiting_upper: 2.,
        waiting_rate: 0,
        forward_mapping: function(0., 1.),
        inverse: function(0., 1.),
        sum_vector: function(2., 0.),
        dependent_lower: function(0., 0.),
        dependent_upper: function(100., 0.),
        backward: None,
    };
    graph
        .set_paired_machine_inputs(0, Some(&[input.clone(), input]))
        .unwrap();
    graph
        .set_paired_machine_root(0, Some(&function(5., 0.)))
        .unwrap();
    assert_eq!(
        graph.coupled_kernel().unwrap().replay(&[0, 1]).unwrap(),
        Some(7.)
    );
    graph.set_front_program_domain(true).unwrap();
    let kernel = graph.coupled_kernel().unwrap();
    assert!(kernel.extend(&kernel.root(true), 0, true).is_err());
    // Raw state inspection can run without asserting that its machine key is
    // a normalized objective of the independently declared source model.
    graph.coupled.as_mut().unwrap().suppress_costs();
    let kernel = graph.coupled_kernel().unwrap();
    let first = kernel
        .extend(&kernel.root(true), 0, true)
        .unwrap()
        .unwrap()
        .0;
    let second = kernel.extend(&first, 1, true).unwrap().unwrap().0;
    let front = second.slope_relation_fronts().next().unwrap().unwrap();
    let descriptor::FrontierOutcome::Complete(minimum) =
        front.minimum(descriptor::FrontierBudget::default())
    else {
        panic!("constant machine front must have an attained minimum");
    };
    assert_eq!(minimum.value, 9.);
    assert_eq!(second.resource_objective, 0.);
}

#[test]
fn machine_tolerance_dominance_cannot_replace_source_suffix_containment() {
    // Exact dyadic recurrence: the common prefix value is half the ceiling.
    // Suffix A adds the ceiling; B adds zero. Only B can complete the prefix.
    let ceiling = 1. / 16384.;
    let initial = ceiling / 2.;
    let mut graph = Graph::<2>::new(2, 0).unwrap();
    graph.add_arc(0, 1, 0.).unwrap();
    graph.add_arc(0, 1, 0.).unwrap();
    graph
        .set_coupled_resources(&[CoupledResource::SlopePaired {
            initial: ranged(initial, 0., 0., 1.),
            initial_base: Window::new(0., 1.).unwrap(),
            arcs: [ceiling, 0.]
                .into_iter()
                .map(|increment| SlopeRelationArc {
                    arrival: ranged(0., 1., 0., 1.),
                    increment: ranged(increment, 0., 0., 1.),
                    lower: ranged(0., 0., 0., 1.),
                    upper: ranged(ceiling, 0., 0., 1.),
                    waiting: 0,
                })
                .collect(),
            mode: PairedResourceMode::Feasibility,
            normalization: PairedCostNormalization::AbsoluteFinalValue,
            policy: FrontOperationPolicy::StrictDomain,
        }])
        .unwrap();
    let kernel = graph.coupled_kernel().unwrap();
    assert!(initial + ceiling > ceiling);
    assert!(initial <= ceiling);
    assert_eq!(kernel.replay(&[0]).unwrap(), None);
    assert_eq!(kernel.replay(&[1]).unwrap(), Some(0.));
    let a = kernel
        .extend(&kernel.root(false), 0, false)
        .unwrap()
        .unwrap()
        .0;
    let b = kernel
        .extend(&kernel.root(false), 1, false)
        .unwrap()
        .unwrap()
        .0;
    assert_eq!(kernel.dominance_penalty(&a, &b, false), None);
    let raw_a = ranged(0., 0., 0., 1.);
    let raw_b = ranged(ceiling, 0., 0., 1.);
    assert!(matches!(
        PreparedSlopeRelationArc::machine_backward_feasibility_dominance(
            &raw_a,
            &raw_b,
            descriptor::FrontierBudget::default()
        ),
        descriptor::FrontierOutcome::Complete(())
    ));
}

#[test]
fn exact_cost_suffix_dominance_and_joins_survive_zero_increment_cycles() {
    for knots in [false, true] {
        for normalization in [
            PairedCostNormalization::AbsoluteFinalValue,
            PairedCostNormalization::FinalMinusInitialMinimum,
        ] {
            for weight in [1., 2.] {
                let mut graph = Graph::<2>::new(3, 0).unwrap();
                for (tail, head) in [(0, 1), (1, 1), (1, 2)] {
                    graph.add_arc(tail, head, 0.).unwrap();
                }
                let definition = if knots {
                    let constant = |value| {
                        PiecewiseLinearFunction::new(vec![[0., value], [1., value]]).unwrap()
                    };
                    CoupledResource::PairedPiecewiseLinear {
                        initial: constant(0.5),
                        initial_base: Window::new(0., 1.).unwrap(),
                        arcs: [0.25, 0., 0.25]
                            .into_iter()
                            .map(|increment| PairedResourceArc {
                                arrival: PiecewiseLinearFunction::new(vec![[0., 0.], [1., 1.]])
                                    .unwrap(),
                                increment: constant(increment),
                                lower: constant(0.),
                                upper: constant(8.),
                                waiting: 0,
                            })
                            .collect(),
                        mode: PairedResourceMode::CostUnbounded,
                        normalization,
                    }
                } else {
                    CoupledResource::SlopePaired {
                        initial: ranged(0.5, 0., 0., 1.),
                        initial_base: Window::new(0., 1.).unwrap(),
                        arcs: [0.25, 0., 0.25]
                            .into_iter()
                            .map(|increment| SlopeRelationArc {
                                arrival: ranged(0., 1., 0., 1.),
                                increment: ranged(increment, 0., 0., 1.),
                                lower: ranged(0., 0., 0., 1.),
                                upper: ranged(8., 0., 0., 1.),
                                waiting: 0,
                            })
                            .collect(),
                        mode: PairedResourceMode::CostUnbounded,
                        normalization,
                        policy: FrontOperationPolicy::StrictDomain,
                    }
                };
                graph.set_coupled_resources(&[definition]).unwrap();
                graph
                    .set_coupled_objective_coefficients(&[(0, weight)])
                    .unwrap();
                let kernel = graph.coupled_kernel().unwrap();
                let prefix = kernel
                    .extend(&kernel.root(true), 0, true)
                    .unwrap()
                    .unwrap()
                    .0;
                let direct = kernel
                    .extend(&kernel.root(false), 2, false)
                    .unwrap()
                    .unwrap()
                    .0;
                let mut cycled = direct.clone();
                for _ in 0..8 {
                    cycled = kernel.extend(&cycled, 1, false).unwrap().unwrap().0;
                }
                assert_eq!(kernel.dominance_penalty(&direct, &cycled, false), Some(0.));
                assert_eq!(kernel.dominance_penalty(&cycled, &direct, false), Some(0.));
                // Independent recurrence: .5 + .25 + eight zeros + .25 = 1.
                let expected = weight
                    * match normalization {
                        PairedCostNormalization::AbsoluteFinalValue => 1.,
                        PairedCostNormalization::FinalMinusInitialMinimum => 0.5,
                    };
                let mut path = vec![0];
                path.extend([1; 8]);
                path.push(2);
                assert_eq!(kernel.replay(&path).unwrap(), Some(expected));
                assert_eq!(
                    kernel.join_resource_objective(&prefix, &direct).unwrap(),
                    Some(expected)
                );
                assert_eq!(
                    kernel.join_resource_objective(&prefix, &cycled).unwrap(),
                    Some(expected)
                );
            }
        }
    }
}

#[test]
fn machine_mode_distinguishes_bounded_and_unbounded_costs() {
    for (mode, expected) in [
        (PairedResourceMode::Feasibility, 4),
        (PairedResourceMode::CostUnbounded, 5),
        (PairedResourceMode::CostBounded, 6),
    ] {
        let graph = graph(mode, PairedCostNormalization::AbsoluteFinalValue);
        assert_eq!(
            graph.coupled_kernel().unwrap().slope_paired[0].machine_mode(),
            expected
        );
    }
}

#[test]
fn backward_machine_labels_extend_from_the_installed_root() {
    let mut graph = Graph::<2>::new(3, 0).unwrap();
    graph.add_arc(0, 1, 0.).unwrap();
    graph.add_arc(1, 2, 0.).unwrap();
    graph.add_arc(1, 2, 0.).unwrap();
    graph
        .set_coupled_resources(&[CoupledResource::SlopePaired {
            initial: function(5., 0.),
            initial_base: Window::new(0., 2.).unwrap(),
            arcs: vec![arc(); 3],
            mode: PairedResourceMode::Feasibility,
            normalization: PairedCostNormalization::AbsoluteFinalValue,
            policy: FrontOperationPolicy::FrontAlgebra,
        }])
        .unwrap();
    let backward = PreparedMachineBackwardInputs {
        tail_lower: 0.,
        tail_upper: 2.,
        forward_at_tail_lower: 0.,
        mapped_upper: 2.,
        dependent: function(1., 0.),
    };
    let input = PreparedMachineArc {
        tail_limit: 2.,
        head_limit: 2.,
        waiting_lower: 0.,
        waiting_upper: 2.,
        waiting_rate: 0,
        forward_mapping: function(0., 1.),
        inverse: function(0., 1.),
        sum_vector: function(0., 0.),
        dependent_lower: function(0., 0.),
        dependent_upper: function(100., 0.),
        backward: Some(backward),
    };
    let other_sink = input.clone();
    graph
        .set_paired_machine_inputs(0, Some(&[input.clone(), input.clone(), other_sink]))
        .unwrap();
    graph
        .set_paired_machine_root(0, Some(&function(5., 0.)))
        .unwrap();
    assert!(matches!(
        graph.install_paired_machine_backward_root(
            0,
            Window::new(0., 2.).unwrap(),
            descriptor::FrontierBudget::default(),
        ),
        descriptor::FrontierOutcome::Complete(())
    ));
    graph.set_front_program_domain(true).unwrap();
    let kernel = graph.coupled_kernel().unwrap();
    let root = kernel.root(false);
    let label = kernel.extend(&root, 1, false).unwrap().unwrap().0;
    let history = label.slope_relations[0].machine_history.as_ref().unwrap();
    let descriptor::MachinePathSnapshot::Backward {
        front,
        waiting_bits,
        key,
        ceiling,
        ..
    } = history.last()
    else {
        panic!("expected backward machine snapshot");
    };
    assert_eq!(front.components()[0].pairs(), &[(99., 0.), (0., 2.)]);
    assert_eq!(*waiting_bits, 0);
    assert_eq!(*ceiling, 100.);
    assert_eq!(*key, 1e12);
    assert_eq!(label.slope_relations[0].minimum, 0.);
    let earlier = kernel.extend(&label, 0, false).unwrap().unwrap().0;
    let history = earlier.slope_relations[0].machine_history.as_ref().unwrap();
    assert_eq!(history.len(), 3);
    let descriptor::MachinePathSnapshot::Backward { front, ceiling, .. } = history.last() else {
        panic!("expected second backward machine snapshot");
    };
    assert_eq!(front.components()[0].pairs(), &[(98., 0.), (0., 2.)]);
    assert_eq!(*ceiling, 100.);

    let mut changed = input.clone();
    changed.dependent_upper = function(110., 0.);
    graph
        .set_paired_machine_inputs(0, Some(&[input.clone(), input, changed]))
        .unwrap();
    let kernel = graph.coupled_kernel().unwrap();
    let root = kernel.root(false);
    assert!(root.slope_relations[0].machine_history.is_none());
    assert!(kernel.extend(&root, 1, false).is_err());
}

#[cfg(feature = "clp")]
#[test]
fn backward_machine_arc_operands_round_trip_in_json() {
    let arc = PreparedMachineArc {
        tail_limit: 2.,
        head_limit: 2.,
        waiting_lower: 0.,
        waiting_upper: 2.,
        waiting_rate: 0,
        forward_mapping: function(0., 1.),
        inverse: function(0., 1.),
        sum_vector: function(0., 0.),
        dependent_lower: function(0., 0.),
        dependent_upper: function(100., 0.),
        backward: Some(PreparedMachineBackwardInputs {
            tail_lower: -0.0,
            tail_upper: 2.,
            forward_at_tail_lower: 0.,
            mapped_upper: 2.,
            dependent: function(1., 0.),
        }),
    };
    let wire = serde_json::to_string(&arc).unwrap();
    let parsed: PreparedMachineArc = serde_json::from_str(&wire).unwrap();
    let original = arc.backward.unwrap();
    let restored = parsed.backward.unwrap();
    assert_eq!(restored.tail_lower.to_bits(), original.tail_lower.to_bits());
    assert_eq!(restored.dependent.pairs(), original.dependent.pairs());
}

#[test]
fn machine_root_without_arc_inputs_does_not_fall_back_to_source_extension() {
    let mut graph = graph(
        PairedResourceMode::Feasibility,
        PairedCostNormalization::AbsoluteFinalValue,
    );
    let mut definitions = graph.coupled_kernel().unwrap().definitions().to_vec();
    let CoupledResource::SlopePaired { policy, .. } = &mut definitions[0] else {
        unreachable!()
    };
    *policy = FrontOperationPolicy::FrontAlgebra;
    graph.set_coupled_resources(&definitions).unwrap();
    graph
        .set_paired_machine_root(0, Some(&function(5., 0.)))
        .unwrap();
    graph.set_front_program_domain(true).unwrap();
    assert!(graph.validate().is_err());
    let kernel = graph.coupled_kernel().unwrap();
    assert!(kernel.extend(&kernel.root(true), 0, true).is_err());
}

#[test]
fn machine_label_root_uses_explicit_prepared_front() {
    let mut graph = graph(
        PairedResourceMode::Feasibility,
        PairedCostNormalization::AbsoluteFinalValue,
    );
    let mut definitions = graph.coupled_kernel().unwrap().definitions().to_vec();
    let CoupledResource::SlopePaired { policy, .. } = &mut definitions[0] else {
        unreachable!()
    };
    *policy = FrontOperationPolicy::FrontAlgebra;
    graph.set_coupled_resources(&definitions).unwrap();
    graph
        .set_paired_machine_root(0, Some(&function(7., 0.)))
        .unwrap();
    graph.set_front_program_domain(true).unwrap();
    let label = graph.coupled_kernel().unwrap().root(true);
    let root = &label.slope_relations[0];
    assert!(root.machine_history.is_some());
    assert_eq!(
        root.front.as_ref().unwrap().components()[0].pairs()[0].0,
        7.
    );
    assert_eq!(root.minimum.to_bits(), 0.0f64.to_bits());
}

#[test]
fn machine_inputs_require_root_and_reject_changed_arc_identity() {
    let mut graph = graph(
        PairedResourceMode::Feasibility,
        PairedCostNormalization::AbsoluteFinalValue,
    );
    let mut definitions = graph.coupled_kernel().unwrap().definitions().to_vec();
    let CoupledResource::SlopePaired { policy, .. } = &mut definitions[0] else {
        unreachable!()
    };
    *policy = FrontOperationPolicy::FrontAlgebra;
    graph.set_coupled_resources(&definitions).unwrap();
    let input = PreparedMachineArc {
        tail_limit: 2.,
        head_limit: 2.,
        waiting_lower: 0.,
        waiting_upper: 2.,
        waiting_rate: 0,
        forward_mapping: function(0., 1.),
        inverse: function(0., 1.),
        sum_vector: function(0., 0.),
        dependent_lower: function(0., 0.),
        dependent_upper: function(100., 0.),
        backward: None,
    };
    let mut nonterminal = input.clone();
    nonterminal.dependent_upper = function(900., 0.);
    graph
        .set_paired_machine_inputs(0, Some(&[nonterminal, input]))
        .unwrap();
    let bounds = Window::new(0., 2.).unwrap();
    let budget = descriptor::FrontierBudget::default();
    let backward = match graph.prepare_paired_machine_backward_root(0, bounds, budget) {
        descriptor::FrontierOutcome::Complete(root) => root,
        other => panic!("unexpected backward root: {other:?}"),
    };
    assert_eq!(backward.key(), 100.);
    assert!(matches!(
        graph.install_paired_machine_backward_root(0, bounds, budget),
        descriptor::FrontierOutcome::Complete(())
    ));
    let paired = &mut graph.coupled.as_mut().unwrap().slope_paired[0];
    let installed = paired.machine_backward_root.as_ref().unwrap().clone();
    assert!(matches!(
        paired.set_machine_backward_root(
            backward,
            descriptor::FrontierBudget {
                max_pairs: 1,
                ..budget
            },
        ),
        descriptor::FrontierOutcome::Unknown(descriptor::FrontierUnknown::PairLimit)
    ));
    assert!(Arc::ptr_eq(
        paired.machine_backward_root.as_ref().unwrap(),
        &installed
    ));
    graph.set_front_program_domain(true).unwrap();
    let label = graph.coupled_kernel().unwrap().root(false);
    let history = label.slope_relations[0].machine_history.as_ref().unwrap();
    let descriptor::MachinePathSnapshot::Root(root) = history.last() else {
        panic!("expected backward root");
    };
    assert_eq!(root.key(), 100.);
    assert_eq!(root.waiting_bits(), 0);
    assert!(matches!(
        history.backward_cost_comparison(history, 6, 0.25, budget),
        descriptor::FrontierOutcome::Complete(0.25)
    ));
    assert!(matches!(
        history.backward_feasibility_comparison(history, budget),
        descriptor::FrontierOutcome::Complete(())
    ));
    assert!(matches!(
        history.backward_cost_comparison(
            history,
            6,
            0.25,
            descriptor::FrontierBudget {
                max_operations: 0,
                ..budget
            }
        ),
        descriptor::FrontierOutcome::Unknown(descriptor::FrontierUnknown::WorkLimit)
    ));
    assert!(graph.validate().is_err());
    graph
        .set_paired_machine_root(0, Some(&function(5., 0.)))
        .unwrap();
    assert!(graph.validate().is_ok());
    graph.sink = 1;
    assert!(matches!(
        graph.prepare_paired_machine_backward_root(0, bounds, budget),
        descriptor::FrontierOutcome::Unknown(_)
    ));
    graph.sink = 2;
    graph.arcs[0].id += 10;
    assert!(graph.validate().is_err());
    assert!(matches!(
        graph.prepare_paired_machine_backward_root(0, bounds, budget),
        descriptor::FrontierOutcome::Unknown(_)
    ));
}
#[test]
fn feasibility_fronts_compare_in_both_directions_and_preserve_prefix_acceptance() {
    let mut graph = graph(
        PairedResourceMode::Feasibility,
        PairedCostNormalization::AbsoluteFinalValue,
    );
    // Alternative suffixes allow dependent values up to 9 and 7.
    let mut definitions = graph.coupled_kernel().unwrap().definitions().to_vec();
    let CoupledResource::SlopePaired { arcs, .. } = &mut definitions[0] else {
        unreachable!()
    };
    arcs[0].upper = function(10., 0.);
    arcs[1].upper = function(8., 0.);
    graph.set_coupled_resources(&definitions).unwrap();
    let kernel = graph.coupled_kernel().unwrap();
    let root = kernel.root(false);
    let wide = kernel.extend(&root, 0, false).unwrap().unwrap().0;
    let narrow = kernel.extend(&root, 1, false).unwrap().unwrap().0;
    assert!(wide.slope_relations[0].upper.is_some());
    assert!(narrow.slope_relations[0].upper.is_some());
    assert_eq!(kernel.dominance_penalty(&wide, &narrow, false), Some(0.));
    assert_eq!(kernel.dominance_penalty(&narrow, &wide, false), None);
    // Check the upper front against literal replay for attainable constant prefixes.
    for value in 0..=12 {
        let mut prefix = kernel.root(true);
        prefix.slope_relations[0] =
            SlopeLabel::root(&SlopeFront::from_prepared(function(value as f64, 0.)), true);
        if kernel.join(&prefix, &narrow).unwrap().is_some() {
            assert!(kernel.join(&prefix, &wide).unwrap().is_some());
        }
    }
    let prepared = &kernel.slope_paired[0];
    let low = SlopeLabel::root(&SlopeFront::from_prepared(function(1., 0.)), true);
    let high = SlopeLabel::root(&SlopeFront::from_prepared(function(2., 0.)), true);
    assert_eq!(prepared.penalty(&low, &high, true, 1.), Some(0.));
    assert_eq!(prepared.penalty(&high, &low, true, 1.), None);
}

#[test]
fn terminal_bounds_above_numerical_infinity_keep_the_adjoint_unknown() {
    let mut graph = graph(
        PairedResourceMode::Feasibility,
        PairedCostNormalization::AbsoluteFinalValue,
    );
    let mut definitions = graph.coupled_kernel().unwrap().definitions().to_vec();
    let CoupledResource::SlopePaired { initial, arcs, .. } = &mut definitions[0] else {
        unreachable!()
    };
    *initial = function(1.5e12, 0.);
    for arc in arcs {
        arc.upper = function(2e12, 0.);
    }
    graph.set_coupled_resources(&definitions).unwrap();
    let kernel = graph.coupled_kernel().unwrap();
    let suffix = kernel
        .extend(&kernel.root(false), 1, false)
        .unwrap()
        .unwrap()
        .0;
    assert!(suffix.slope_relations[0].upper.is_none());
    let prefix = kernel
        .extend(&kernel.root(true), 0, true)
        .unwrap()
        .unwrap()
        .0;
    assert!(kernel.join(&prefix, &suffix).unwrap().is_some());
}

#[test]
fn unknown_backward_conversion_stays_unknown_after_prepend() {
    let mut graph = graph(
        PairedResourceMode::Feasibility,
        PairedCostNormalization::AbsoluteFinalValue,
    );
    let mut definitions = graph.coupled_kernel().unwrap().definitions().to_vec();
    let CoupledResource::SlopePaired { arcs, .. } = &mut definitions[0] else {
        unreachable!()
    };
    // A singleton bound is supported by raw slopes but cannot be converted
    // into a two-knot piecewise-linear function.
    arcs[1].upper = PreparedSlopeFunction::constant_point(100., 1.).unwrap();
    graph.set_coupled_resources(&definitions).unwrap();
    let kernel = graph.coupled_kernel().unwrap();
    let suffix = kernel
        .extend(&kernel.root(false), 1, false)
        .unwrap()
        .unwrap()
        .0;
    assert!(suffix.slope_relations[0].upper.is_none());
    let suffix = kernel.extend(&suffix, 0, false).unwrap().unwrap().0;
    assert!(suffix.slope_relations[0].upper.is_none());
}

fn function(value: f64, slope: f64) -> PreparedSlopeFunction {
    descriptor::RawAffine::new(value, slope)
        .unwrap()
        .prepare(0., 2.)
        .unwrap()
}
fn arc() -> SlopeRelationArc {
    SlopeRelationArc {
        arrival: function(0., 1.),
        increment: function(1., 0.),
        lower: function(0., 0.),
        upper: function(100., 0.),
        waiting: 0,
    }
}
fn graph(mode: PairedResourceMode, normalization: PairedCostNormalization) -> Graph<2> {
    let mut graph = Graph::new(3, 0).unwrap();
    graph.add_arc(0, 1, 0.).unwrap();
    graph.add_arc(1, 2, 0.).unwrap();
    graph
        .set_coupled_resources(&[CoupledResource::SlopePaired {
            initial: function(5., 0.),
            initial_base: Window::new(0., 2.).unwrap(),
            arcs: vec![arc(); 2],
            mode,
            normalization,
            policy: FrontOperationPolicy::StrictDomain,
        }])
        .unwrap();
    graph
}
fn shifted_prefixes(
    policy: FrontOperationPolicy,
    upper: f64,
) -> (Graph<2>, Box<CoupledLabelState>, Box<CoupledLabelState>) {
    let mut graph = Graph::new(3, 0).unwrap();
    for (tail, head) in [(0, 1), (0, 1), (1, 2)] {
        graph.add_arc(tail, head, 0.).unwrap();
    }
    let arc = |increment| SlopeRelationArc {
        increment,
        upper: function(upper, 0.),
        lower: function(0., 0.),
        ..arc()
    };
    graph
        .set_coupled_resources(&[CoupledResource::SlopePaired {
            initial: function(0., 0.),
            initial_base: Window::new(0., 2.).unwrap(),
            // The falling prefix ends at 2 - x, the flat one at 1.
            arcs: vec![
                arc(function(2., -1.)),
                arc(function(1., 0.)),
                arc(function(0., 0.)),
            ],
            mode: PairedResourceMode::CostUnbounded,
            normalization: PairedCostNormalization::AbsoluteFinalValue,
            policy,
        }])
        .unwrap();
    let kernel = graph.coupled_kernel().unwrap();
    let root = kernel.root(true);
    let prefix = |arc| {
        let mut state = kernel.extend(&root, arc, true).unwrap().unwrap().0;
        state.account_extension(&root, 0., 0.).unwrap();
        state
    };
    let (falling, flat) = (prefix(0), prefix(1));
    (graph, falling, flat)
}
#[test]
fn native_unbounded_raw_paired_front_pays_its_excess() {
    let native = FrontOperationPolicy::FrontAlgebra;
    let (graph, falling, flat) = shifted_prefixes(native, 1e12);
    let kernel = graph.coupled_kernel().unwrap();
    // Falling exceeds flat by 1 at x=0: 1 + objective 1 - objective 0.
    let penalty = kernel.dominance_penalty(&falling, &flat, true).unwrap();
    assert!((2. ..=2. + 1e-12).contains(&penalty), "{penalty}");
    // Flat exceeds falling by 1 at x=2, already paid by its objective.
    assert_eq!(kernel.dominance_penalty(&flat, &falling, true), Some(0.));
    // A reachable upper bound or the strict policy keeps exact coverage.
    for (policy, upper) in [(native, 100.), (FrontOperationPolicy::StrictDomain, 1e12)] {
        let (graph, falling, flat) = shifted_prefixes(policy, upper);
        let kernel = graph.coupled_kernel().unwrap();
        assert_eq!(kernel.dominance_penalty(&falling, &flat, true), None);
        assert_eq!(kernel.dominance_penalty(&flat, &falling, true), None);
    }
}
fn minimum(state: &CoupledLabelState) -> f64 {
    match state
        .slope_relation_fronts()
        .next()
        .unwrap()
        .unwrap()
        .minimum(FrontierBudget::default())
    {
        descriptor::FrontierOutcome::Complete(witness) => witness.value,
        _ => panic!("expected complete raw paired front"),
    }
}
#[test]
fn raw_paired_absolute_and_relative_objectives_keep_root_zero_and_state_unscaled() {
    for (normalization, expected) in [
        (PairedCostNormalization::AbsoluteFinalValue, 7.),
        (PairedCostNormalization::FinalMinusInitialMinimum, 2.),
    ] {
        let graph = graph(PairedResourceMode::CostBounded, normalization);
        let kernel = graph.coupled_kernel().unwrap();
        assert_eq!(kernel.replay(&[]).unwrap(), Some(0.));
        let root = kernel.root(true);
        assert_eq!(minimum(&root), 5.);
        assert_eq!(root.resource_objective, 0.);
        let (state, cost) = kernel.replay_state(&[0, 1]).unwrap().unwrap();
        assert_eq!(cost, expected);
        assert_eq!(minimum(&state), 7.);
        for split in 0..=2 {
            let f = kernel.replay_state(&[0, 1][..split]).unwrap().unwrap().0;
            let mut b = kernel.root(false);
            for &a in [0, 1][split..].iter().rev() {
                b = kernel.extend(&b, a, false).unwrap().unwrap().0;
            }
            assert_eq!(
                kernel.join_resource_objective(&f, &b).unwrap(),
                Some(expected)
            );
        }
        assert_eq!(
            graph
                .clone()
                .coupled_kernel()
                .unwrap()
                .replay(&[0, 1])
                .unwrap(),
            Some(expected)
        );
    }
}
#[test]
fn raw_paired_pricing_and_enumeration_charge_absolute_initial_value_in_all_modes() {
    let mut graph = Graph::<2>::new(4, 1).unwrap();
    for v in 0..4 {
        graph.set_bounds(v, 0, 0., 2.).unwrap();
    }
    for (u, v, cost) in [(0, 1, -10.), (1, 3, -10.), (0, 2, -4.), (2, 3, -4.)] {
        let a = graph.add_arc(u, v, cost).unwrap();
        graph.arc_mut(a).unwrap().consumption[0] = 1.;
    }
    graph
        .set_coupled_resources(&[CoupledResource::SlopePaired {
            initial: function(5., 0.),
            initial_base: Window::new(0., 2.).unwrap(),
            arcs: vec![arc(); 4],
            mode: PairedResourceMode::CostBounded,
            normalization: PairedCostNormalization::AbsoluteFinalValue,
            policy: FrontOperationPolicy::StrictDomain,
        }])
        .unwrap();
    for mode in [0., 1., 2.] {
        let mut solver = crate::pricing::Solver::<2>::new(4, 1).unwrap();
        solver.graph = graph.clone();
        solver.enumeration_preserve_arc_paths = true;
        solver
            .set_option("BALDESuseBidirectionalSearch", mode)
            .unwrap();
        assert_eq!(
            solver
                .price_phase(0, 20, false)
                .unwrap()
                .minimum_reduced_cost,
            -13.
        );
        let mut costs: Vec<_> = solver
            .enumerate(0.)
            .unwrap()
            .into_iter()
            .map(|route| {
                assert_eq!(route.cost, route.original_cost);
                route.cost
            })
            .collect();
        costs.sort_by(f64::total_cmp);
        assert_eq!(costs, vec![-13., -1.]);
    }
}
#[test]
fn raw_paired_feasibility_suffix_is_replayed_even_without_resource_objective() {
    let mut graph = graph(
        PairedResourceMode::Feasibility,
        PairedCostNormalization::AbsoluteFinalValue,
    );
    let mut definitions = graph.coupled_kernel().unwrap().definitions().to_vec();
    let CoupledResource::SlopePaired { arcs, .. } = &mut definitions[0] else {
        unreachable!()
    };
    for arc in arcs {
        arc.upper = function(6.5, 0.);
    }
    graph.set_coupled_resources(&definitions).unwrap();
    let kernel = graph.coupled_kernel().unwrap();
    assert!(!kernel.has_costs());
    assert_eq!(kernel.replay(&[0, 1]).unwrap(), None);
    let f = kernel
        .extend(&kernel.root(true), 0, true)
        .unwrap()
        .unwrap()
        .0;
    let b = kernel
        .extend(&kernel.root(false), 1, false)
        .unwrap()
        .unwrap()
        .0;
    assert_eq!(kernel.join(&f, &b).unwrap(), None);
}
#[test]
fn unsafe_native_waiting_crossovers_never_become_underpriced_or_empty_routes() {
    for from_increment in [false, true] {
        let hump = PreparedSlopeFunction::from_prepared_pairs(vec![
            (if from_increment { 1.1 } else { 2.1 }, 0.),
            (0.3, 0.7),
            (-0.2, 2.),
        ])
        .unwrap();
        let mut graph = Graph::<2>::new(3, 0).unwrap();
        graph.add_arc(0, 1, 0.5).unwrap();
        graph.add_arc(1, 2, 0.5).unwrap();
        let initial = if from_increment {
            function(2., 0.)
        } else {
            hump.clone()
        };
        let mut arcs = vec![arc(); 2];
        if from_increment {
            for arc in &mut arcs {
                arc.increment = hump.clone();
            }
        }
        let installed = graph.set_coupled_resources(&[CoupledResource::SlopePaired {
            initial,
            initial_base: Window::new(0., 2.).unwrap(),
            arcs,
            mode: PairedResourceMode::CostBounded,
            normalization: PairedCostNormalization::AbsoluteFinalValue,
            policy: FrontOperationPolicy::StrictDomain,
        }]);
        if let Err(error) = installed {
            let text = error.to_string().to_ascii_lowercase();
            assert!(
                text.contains("unsupported") || text.contains("unknown"),
                "unexpected preparation failure: {text}"
            );
            continue;
        }
        // Independent witness at t=2: ordered endpoint accumulation gives
        // hump minimum2.05 or increment1.05. Waiting cannot improve below
        // these global source minima; both are attained by this witness.
        let expected = if from_increment { 4.1 } else { 4.05 };
        match graph.coupled_kernel().unwrap().replay(&[0, 1]) {
            Ok(Some(value)) => assert!(
                (value - expected).abs() < 1e-12,
                "raw waiting underprice: {value} vs {expected}"
            ),
            Ok(None) => panic!("unsupported waiting cannot prove this feasible route empty"),
            Err(error) => {
                let text = error.to_string().to_ascii_lowercase();
                assert!(
                    text.contains("unsupported") || text.contains("unknown"),
                    "unexpected replay failure: {text}"
                );
            } // Explicit unsupported/unknown is required until proved.
        }
    }
}
#[cfg(feature = "clp")]
#[test]
fn raw_paired_phase_i_preserves_physical_front_and_certifies_strict_controls() {
    let graph = graph(
        PairedResourceMode::CostBounded,
        PairedCostNormalization::AbsoluteFinalValue,
    );
    let mut kernel = graph.coupled_kernel().unwrap().clone();
    kernel.suppress_costs();
    let (state, cost) = kernel.replay_state(&[0, 1]).unwrap().unwrap();
    assert_eq!(cost, 0.);
    assert_eq!(minimum(&state), 7.);
    assert_eq!(kernel.certify(&[0, 1]), Some((true, [0., 0.])));
    let json = serde_json::to_string(kernel.definitions()).unwrap();
    let restored: Vec<CoupledResource> = serde_json::from_str(&json).unwrap();
    assert_eq!(restored, kernel.definitions());
    let missing = json.replace(",\"normalization\":\"absolute_final_value\"", "");
    assert!(serde_json::from_str::<Vec<CoupledResource>>(&missing).is_err());
}

fn ranged(intercept: f64, slope: f64, lower: f64, upper: f64) -> PreparedSlopeFunction {
    descriptor::RawAffine::new(intercept, slope)
        .unwrap()
        .prepare(lower, upper)
        .unwrap()
}
fn physical_graph(vertices: usize, arcs: &[(usize, usize, f64)]) -> Graph<2> {
    let mut graph = Graph::new(vertices, 1).unwrap();
    for vertex in 0..vertices {
        graph.set_bounds(vertex, 0, 0., 2.).unwrap();
    }
    for &(tail, head, cost) in arcs {
        let arc = graph.add_arc(tail, head, cost).unwrap();
        graph.arc_mut(arc).unwrap().consumption[0] = 1.;
    }
    graph
}
#[test]
fn composed_arrivals_variable_increments_and_signed_waiting_match_attained_bounds() {
    for policy in [
        FrontOperationPolicy::StrictDomain,
        FrontOperationPolicy::FrontAlgebra,
    ] {
        for waiting in [-1, 1] {
            for normalization in [
                PairedCostNormalization::AbsoluteFinalValue,
                PairedCostNormalization::FinalMinusInitialMinimum,
            ] {
                let mut graph = physical_graph(3, &[(0, 1, -10.), (1, 2, -10.)]);
                let arc = |lo, hi| SlopeRelationArc {
                    arrival: ranged(0., 2., 0., 4.),
                    increment: ranged(1., 1., 0., 4.),
                    lower: ranged(0., 0., lo, hi),
                    upper: ranged(100., 0., lo, hi),
                    waiting,
                };
                graph
                    .set_coupled_resources(&[CoupledResource::SlopePaired {
                        initial: ranged(5., 2., 0., 2.),
                        initial_base: Window::new(0., 2.).unwrap(),
                        arcs: vec![arc(1., 4.), arc(0., 8.)],
                        mode: if waiting < 0 {
                            PairedResourceMode::CostUnbounded
                        } else {
                            PairedResourceMode::CostBounded
                        },
                        normalization,
                        policy,
                    }])
                    .unwrap();
                // With initial coordinate t, first head x and second head y:
                // k=1 gives final 7+t+y >=9, attained at (t,x,y)=(0,1,2).
                // k=-1 gives final 7+5t+2x-y >=1, attained at (0,1,8).
                // Clamps never bind at those controls. These are independent
                // attained lower bounds, not evaluations of the front helper.
                let absolute = if waiting > 0 { 9. } else { 1. };
                let resource = match normalization {
                    PairedCostNormalization::AbsoluteFinalValue => absolute,
                    PairedCostNormalization::FinalMinusInitialMinimum => absolute - 5.,
                };
                let kernel = graph.coupled_kernel().unwrap();
                assert_eq!(kernel.replay(&[0, 1]).unwrap(), Some(resource));
                for split in 0..=2 {
                    let f = kernel.replay_state(&[0, 1][..split]).unwrap().unwrap().0;
                    let mut b = kernel.root(false);
                    for &a in [0, 1][split..].iter().rev() {
                        b = kernel.extend(&b, a, false).unwrap().unwrap().0;
                    }
                    assert_eq!(
                        kernel.join_resource_objective(&f, &b).unwrap(),
                        Some(resource)
                    );
                }
                for search in [0., 1., 2.] {
                    let mut solver = crate::pricing::Solver::<2>::new(3, 1).unwrap();
                    solver.graph = graph.clone();
                    solver.enumeration_preserve_arc_paths = true;
                    solver
                        .set_option("BALDESuseBidirectionalSearch", search)
                        .unwrap();
                    assert_eq!(
                        solver
                            .price_phase(0, 20, false)
                            .unwrap()
                            .minimum_reduced_cost,
                        resource - 20.
                    );
                    let routes = solver.enumerate(0.).unwrap();
                    assert_eq!(routes.len(), 1);
                    assert_eq!(routes[0].cost, resource - 20.);
                    assert_eq!(routes[0].original_cost, resource - 20.);
                }
            }
        }
    }
}
#[test]
fn composed_function_bounds_retain_a_singleton_and_reject_a_cheaper_infeasible_route() {
    for policy in [
        FrontOperationPolicy::StrictDomain,
        FrontOperationPolicy::FrontAlgebra,
    ] {
        let mut graph = physical_graph(4, &[(0, 1, -3.), (1, 3, -3.), (0, 2, -10.), (2, 3, -10.)]);
        let first = SlopeRelationArc {
            arrival: ranged(0., 2., 0., 4.),
            increment: ranged(1., 1., 0., 4.),
            lower: ranged(0., 1., 1., 4.),
            upper: ranged(100., 0., 1., 4.),
            waiting: 0,
        };
        let second = SlopeRelationArc {
            arrival: ranged(0., 2., 0., 4.),
            increment: ranged(1., 1., 0., 4.),
            lower: ranged(0., 0., 2., 4.),
            upper: ranged(5., -1., 2., 4.),
            waiting: 0,
        };
        let mut impossible = second.clone();
        impossible.increment = ranged(50., 0., 0., 4.);
        graph
            .set_coupled_resources(&[CoupledResource::SlopePaired {
                initial: ranged(0., 0., 0., 2.),
                initial_base: Window::new(0., 2.).unwrap(),
                arcs: vec![first.clone(), second, first, impossible],
                mode: PairedResourceMode::Feasibility,
                normalization: PairedCostNormalization::AbsoluteFinalValue,
                policy,
            }])
            .unwrap();
        // First head x>=1 is clamped to dependent x. At the second head
        // y>=2*x, the least dependent value is 3. Upper=5-y leaves only
        // y=2. The expensive-increment branch has dependent>=51>upper.
        let kernel = graph.coupled_kernel().unwrap();
        let (state, cost) = kernel.replay_state(&[0, 1]).unwrap().unwrap();
        assert_eq!(cost, 0.);
        let front = state.slope_relation_fronts().next().unwrap().unwrap();
        assert_eq!(front.components().len(), 1);
        assert_eq!(front.components()[0].pairs(), &[(3., 2.)]);
        assert_eq!(kernel.replay(&[2, 3]).unwrap(), None);
        for search in [0., 1., 2.] {
            let mut solver = crate::pricing::Solver::<2>::new(4, 1).unwrap();
            solver.graph = graph.clone();
            solver.enumeration_preserve_arc_paths = true;
            solver
                .set_option("BALDESuseBidirectionalSearch", search)
                .unwrap();
            assert_eq!(
                solver
                    .price_phase(0, 20, false)
                    .unwrap()
                    .minimum_reduced_cost,
                -6.
            );
            let routes = solver.enumerate(0.).unwrap();
            assert_eq!(routes.len(), 1);
            assert_eq!(routes[0].cost, -6.);
        }
    }
}

#[test]
fn joined_machine_reconstruction_preserves_consumptions_and_rejects_replaced_roots() {
    use descriptor::{FrontierBudget, FrontierOutcome, FrontierUnknown};
    for mode in [
        PairedResourceMode::Feasibility,
        PairedResourceMode::CostUnbounded,
    ] {
        let mut graph = Graph::<2>::new(3, 0).unwrap();
        graph.add_arc(0, 1, 0.).unwrap();
        graph.add_arc(1, 2, 0.).unwrap();
        graph
            .set_coupled_resources(&[CoupledResource::SlopePaired {
                initial: function(5., 0.),
                initial_base: Window::new(0., 2.).unwrap(),
                arcs: vec![arc(); 2],
                mode,
                normalization: PairedCostNormalization::AbsoluteFinalValue,
                policy: FrontOperationPolicy::FrontAlgebra,
            }])
            .unwrap();
        let input = PreparedMachineArc {
            tail_limit: 2.,
            head_limit: 2.,
            waiting_lower: 0.,
            waiting_upper: 2.,
            waiting_rate: 0,
            forward_mapping: function(0., 1.),
            inverse: function(0., 1.),
            sum_vector: function(1., 0.),
            dependent_lower: function(0., 0.),
            dependent_upper: function(100., 0.),
            backward: Some(PreparedMachineBackwardInputs {
                tail_lower: 0.,
                tail_upper: 2.,
                forward_at_tail_lower: 0.,
                mapped_upper: 2.,
                dependent: function(1., 0.),
            }),
        };
        graph
            .set_paired_machine_inputs(0, Some(&[input.clone(), input]))
            .unwrap();
        graph
            .set_paired_machine_root(0, Some(&function(5., 0.)))
            .unwrap();
        let budget = FrontierBudget::default();
        assert!(matches!(
            graph.install_paired_machine_backward_root(0, Window::new(0., 2.).unwrap(), budget,),
            FrontierOutcome::Complete(())
        ));
        graph.set_front_program_domain(true).unwrap();
        if mode == PairedResourceMode::CostUnbounded {
            let kernel = graph.coupled_kernel().unwrap();
            let first = kernel
                .extend(&kernel.root(true), 0, true)
                .unwrap()
                .unwrap()
                .0;
            assert_eq!(first.resource_objective, 6.);
            let backward = kernel
                .extend(&kernel.root(false), 1, false)
                .unwrap()
                .unwrap()
                .0;
            assert_eq!(backward.resource_objective, 0.);
            assert_eq!(
                kernel.join_resource_objective(&first, &backward).unwrap(),
                Some(7.)
            );
            // This test reconstructs raw consumptions; it must not bypass the
            // still-unproved nonzero backward objective accounting contract.
            graph.coupled.as_mut().unwrap().suppress_costs();
        }
        let kernel = graph.coupled_kernel().unwrap();
        let forward = kernel
            .extend(&kernel.root(true), 0, true)
            .unwrap()
            .unwrap()
            .0;
        let backward = kernel
            .extend(&kernel.root(false), 1, false)
            .unwrap()
            .unwrap()
            .0;
        if mode == PairedResourceMode::Feasibility {
            assert_eq!(kernel.join(&forward, &backward).unwrap(), Some(0.));
        }
        let FrontierOutcome::Complete(joined) =
            graph.reconstruct_paired_machine_join(0, &forward, &backward, budget)
        else {
            panic!("joined machine path should reconstruct");
        };
        // Independent exact recurrence: initial 5 plus one unit on each arc.
        assert_eq!(joined.path.base, vec![0., 0., 0.]);
        assert_eq!(joined.path.dependent, vec![5., 6., 7.]);
        assert_eq!(joined.path.waiting, vec![0., 0.]);
        assert_eq!(joined.initial_waiting, 0.);
        assert_eq!(joined.selection.coordinate, 0.);
        assert_eq!(
            joined.selection.dependent,
            (mode == PairedResourceMode::CostUnbounded).then_some(7.)
        );
        let FrontierOutcome::Complete(label_join) =
            joined.selection.join_labels(-20., 1., 0., 1, 2)
        else {
            panic!("raw reconstructed label costs should combine");
        };
        assert!(label_join.compatible);
        assert_eq!(
            label_join.reduced_cost,
            if mode == PairedResourceMode::CostUnbounded {
                -12.
            } else {
                -19.
            }
        );
        assert!(matches!(
            graph.reconstruct_paired_machine_join(
                0,
                &forward,
                &backward,
                FrontierBudget {
                    max_operations: 0,
                    ..budget
                },
            ),
            FrontierOutcome::Unknown(FrontierUnknown::WorkLimit)
        ));
        assert!(matches!(
            graph.install_paired_machine_backward_root(0, Window::new(0., 1.5).unwrap(), budget,),
            FrontierOutcome::Complete(())
        ));
        assert!(matches!(
            graph.reconstruct_paired_machine_join(0, &forward, &backward, budget,),
            FrontierOutcome::Unknown(_)
        ));
    }
}

#[test]
fn constant_identity_suffixes_discard_positive_cycles_without_bounded_cache() {
    for policy in [
        FrontOperationPolicy::FrontAlgebra,
        FrontOperationPolicy::StrictDomain,
    ] {
        for mode in [
            PairedResourceMode::CostUnbounded,
            PairedResourceMode::CostBounded,
        ] {
            for normalization in [
                PairedCostNormalization::AbsoluteFinalValue,
                PairedCostNormalization::FinalMinusInitialMinimum,
            ] {
                let mut graph = Graph::<2>::new(3, 0).unwrap();
                for (tail, head) in [(0, 1), (1, 1), (1, 2)] {
                    graph.add_arc(tail, head, 0.).unwrap();
                }
                graph
                    .set_coupled_resources(&[CoupledResource::SlopePaired {
                        initial: ranged(0., 0., 0., 8.),
                        initial_base: Window::new(0., 8.).unwrap(),
                        arcs: vec![
                            SlopeRelationArc {
                                arrival: ranged(0., 1., 0., 8.),
                                increment: ranged(1., 0., 0., 8.),
                                lower: ranged(0., 0., 0., 8.),
                                upper: ranged(8., 0., 0., 8.),
                                waiting: 0,
                            };
                            3
                        ],
                        mode,
                        normalization,
                        policy,
                    }])
                    .unwrap();
                let kernel = graph.coupled_kernel().unwrap();
                let direct = kernel
                    .extend(&kernel.root(false), 2, false)
                    .unwrap()
                    .unwrap()
                    .0;
                let mut cycled = direct.clone();
                for _ in 0..3 {
                    cycled = kernel.extend(&cycled, 1, false).unwrap().unwrap().0;
                    assert_eq!(kernel.dominance_penalty(&direct, &cycled, false), Some(0.));
                    assert_eq!(kernel.dominance_penalty(&cycled, &direct, false), None);
                }
                // Independent recurrence for the full paths: each traversed
                // arc adds exactly one to the constant initial value.
                assert_eq!(kernel.replay(&[0, 2]).unwrap().unwrap(), 2.);
                assert_eq!(kernel.replay(&[0, 1, 1, 1, 2]).unwrap().unwrap(), 5.);
                graph
                    .set_coupled_objective_coefficients(&[(0, -1.)])
                    .unwrap();
                let kernel = graph.coupled_kernel().unwrap();
                let direct = kernel
                    .extend(&kernel.root(false), 2, false)
                    .unwrap()
                    .unwrap()
                    .0;
                let cycled = kernel.extend(&direct, 1, false).unwrap().unwrap().0;
                assert_eq!(kernel.dominance_penalty(&direct, &cycled, false), None);
            }
        }
    }
}
