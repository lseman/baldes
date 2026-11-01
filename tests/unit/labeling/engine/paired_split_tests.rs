use super::*;
use crate::graph::coupled::descriptor::{
    FrontierBudget, FrontierOutcome, MachineRootInput, PreparedSlopeFunction, RawAffine,
};
use crate::graph::coupled::{CoupledResource, MachineArcFunctionInput, PairedResourceMode};

fn solver() -> crate::pricing::Solver<2> {
    solver_with_root(None)
}

fn solver_with_root(root: Option<&PreparedSlopeFunction>) -> crate::pricing::Solver<2> {
    solver_with_binding(root, true)
}

fn solver_with_binding(
    root: Option<&PreparedSlopeFunction>,
    binding: bool,
) -> crate::pricing::Solver<2> {
    solver_with_program(root, binding, 100.)
}

fn solver_with_program(
    root: Option<&PreparedSlopeFunction>,
    binding: bool,
    ceiling: f64,
) -> crate::pricing::Solver<2> {
    let mut solver = crate::pricing::Solver::<2>::new(3, usize::from(binding)).unwrap();
    let window = crate::Window::new(0., 4.).unwrap();
    if binding {
        for vertex in 0..3 {
            solver.graph.set_bounds(vertex, 0, 0., 4.).unwrap();
        }
    }
    solver.graph.vertices[0].elements.clear();
    solver.graph.vertices[2].elements.clear();
    solver.graph.add_arc(0, 1, -1.).unwrap();
    solver.graph.add_arc(1, 2, -1.).unwrap();
    solver
        .graph
        .set_coupled_resources(&[CoupledResource::PairedFront {
            mode: PairedResourceMode::CostUnbounded,
        }])
        .unwrap();
    let base = PreparedSlopeFunction::from_prepared_pairs(vec![(0., 0.), (0., 4.)]).unwrap();
    let affine = |value| MachineRootInput::Affine(RawAffine::new(value, 0.).unwrap());
    let inputs: Vec<_> = (0..2)
        .map(|physical_arc| MachineArcFunctionInput {
            physical_arc,
            tail_bounds: window,
            waiting_bounds: window,
            waiting_rate: 0,
            base: &base,
            dependent: affine(0.),
            dependent_lower: affine(-100.),
            dependent_upper: affine(ceiling),
        })
        .collect();
    let prepared = solver.graph.prepare_paired_front_program(
        0,
        &inputs,
        root.map_or_else(|| affine(0.), MachineRootInput::Vector),
        window,
        window,
        FrontierBudget::default(),
    );
    assert!(
        matches!(prepared, FrontierOutcome::Complete(())),
        "{prepared:?}"
    );
    if binding {
        solver
            .graph
            .set_paired_waiting_windows_from_resource(0, 0)
            .unwrap();
        solver.bucket_steps[0] = 1.;
    }
    solver
}

#[test]
fn paired_bucket_pieces_preserve_route_and_cost_and_partial_mode_forces_bucket_split() {
    for bidirectional in [0., 1., 2.] {
        let mut baseline = solver();
        baseline
            .set_option("BALDESuseBidirectionalSearch", bidirectional)
            .unwrap();
        let reference = baseline.price_phase(0, 10, false).unwrap();
        assert!(reference.exact);
        assert_eq!(reference.minimum_reduced_cost, -2.);
        for partial in [false, true] {
            let mut candidate = solver();
            candidate
                .set_option("BALDESuseBidirectionalSearch", bidirectional)
                .unwrap();
            candidate
                .set_option("BALDESlabelSplitStrategy", if partial { 1. } else { 2. })
                .unwrap();
            if partial {
                candidate
                    .set_option("BALDESlabelDominanceStrategy", 1.)
                    .unwrap();
            }
            let result = candidate.price_phase(0, 10, false).unwrap();
            assert!(result.exact);
            assert_eq!(result.minimum_reduced_cost, reference.minimum_reduced_cost);
            assert!(!result.columns.is_empty());
            assert!(result
                .columns
                .iter()
                .all(|route| route.arcs == [0, 1] && route.cost == -2.));
            let engine = candidate.engine.as_ref().unwrap();
            assert_eq!(engine.controls.paired_split_strategy, 2);
            assert_eq!(engine.controls.partial_dominance, partial);
            assert!(result.statistics.generated > reference.statistics.generated);
            let mut child = candidate.fork_for_node();
            let replay = child.price_phase(0, 10, false).unwrap();
            assert!(replay.exact);
            assert_eq!(replay.minimum_reduced_cost, -2.);
        }
    }
}

#[test]
fn engine_direct_partial_configuration_rejects_an_ordinary_resource() {
    let graph = Graph::<2>::new(2, 1).unwrap();
    let error = LabelingEngine::prepare(
        &graph,
        EngineControls {
            partial_dominance: true,
            ..Default::default()
        },
        Phase {
            check_ng: true,
            jump_arcs: true,
            labels_per_bucket: i64::MAX,
        },
    )
    .unwrap_err();
    assert!(error.to_string().contains("paired resource"));
}

#[test]
fn split_strategy_is_inactive_for_ordinary_native_state_types() {
    let graph = Graph::<2>::new(2, 1).unwrap();
    for strategy in [1, 2] {
        let engine = LabelingEngine::prepare(
            &graph,
            EngineControls {
                paired_split_strategy: strategy,
                ..Default::default()
            },
            Phase {
                check_ng: true,
                jump_arcs: true,
                labels_per_bucket: i64::MAX,
            },
        )
        .unwrap();
        assert_eq!(engine.controls.paired_split_strategy, 0);
    }
}

#[test]
fn paired_bucket_splitting_matches_signed_multisegment_split_extension_objective() {
    let root =
        PreparedSlopeFunction::from_prepared_pairs(vec![(5., 0.), (-1., 2.), (-2., 4.)]).unwrap();
    let mut baseline = solver_with_root(Some(&root));
    baseline
        .set_option("BALDESuseBidirectionalSearch", 0.)
        .unwrap();
    let reference = baseline.price_phase(0, 10, false).unwrap();
    assert!(reference.exact);
    assert_eq!(reference.minimum_reduced_cost, -3.);
    let mut split = solver_with_root(Some(&root));
    split
        .set_option("BALDESuseBidirectionalSearch", 0.)
        .unwrap();
    split.set_option("BALDESlabelSplitStrategy", 2.).unwrap();
    let result = split.price_phase(0, 10, false).unwrap();
    assert!(result.exact);
    // The crossing coordinate is shifted by +2e-6 without changing its
    // anchor. Actual reference extension recomputes this literal tail key.
    let tail_key = 3. + (4. - 2.000002) * -2.;
    assert_eq!(result.minimum_reduced_cost, -2. + tail_key);
    let middle_key = 4. + (2.000002 - 1.) * -1.;
    let expected = [tail_key, middle_key, 4.];
    assert_eq!(result.columns.len(), expected.len());
    for (route, dependent) in result.columns.iter().zip(expected) {
        assert_eq!(route.arcs, [0, 1]);
        assert_eq!(route.cost, -2. + dependent);
        assert_eq!(route.original_cost, -2. + dependent);
        assert_eq!(route.coupled_values, [(0, dependent)]);
    }
}

#[test]
fn paired_bucket_manager_indexes_only_base_and_still_checks_standard_resources() {
    let mut candidate = solver();
    candidate
        .graph
        .add_resource(crate::graph::ResourceKind::Disposable)
        .unwrap();
    for vertex in 0..3 {
        candidate.graph.set_bounds(vertex, 1, 0., 4.).unwrap();
    }
    for arc in &mut candidate.graph.arcs {
        arc.consumption[1] = 1.;
    }
    candidate.bucket_steps[1] = 1.;
    candidate
        .set_option("BALDESlabelDominanceStrategy", 1.)
        .unwrap();
    candidate
        .set_option("BALDESuseBidirectionalSearch", 0.)
        .unwrap();
    let result = candidate.price_phase(0, 10, false).unwrap();
    assert!(result.exact);
    assert_eq!(result.minimum_reduced_cost, -2.);
    let grid = &candidate.engine.as_ref().unwrap().grid;
    assert_eq!(grid.dimensions, 1);
    assert_eq!(grid.resource_slots, [0]);
    candidate.graph.set_bounds(2, 1, 0., 1.).unwrap();
    let infeasible = candidate.price_phase(0, 10, false).unwrap();
    assert!(infeasible.exact);
    assert!(infeasible.columns.is_empty());
    assert!(infeasible.minimum_reduced_cost >= NO_ROUTE_COST);
}

#[test]
fn paired_directional_fixing_keeps_the_backward_suffix_and_its_complete_route() {
    let root =
        PreparedSlopeFunction::from_prepared_pairs(vec![(5., 0.), (-1., 2.), (-2., 4.)]).unwrap();
    for mode in [0, 1, 2] {
        let mut candidate = solver_with_root(Some(&root));
        candidate
            .set_option("BALDESlabelSplitStrategy", 2.)
            .unwrap();
        candidate
            .set_option("BALDESuseBidirectionalSearch", 0.)
            .unwrap();
        candidate.price_phase(0, 10, false).unwrap();
        candidate.complete_labels(false).unwrap();
        let report = candidate
            .engine
            .as_mut()
            .unwrap()
            .prepare_and_fix_bucket_arcs(&candidate.graph, &candidate.controls, 0., false, mode)
            .unwrap();
        assert!(report.eliminated_forward > 0, "mode {mode}");
        assert_eq!(report.eliminated_backward, 0, "mode {mode}");
        let engine = candidate.engine.as_mut().unwrap();
        let terminal = 1;
        assert!(engine.physical_arc_survives_fixing(terminal, &[]));
        let all_occurrences: Vec<_> = [true, false]
            .map(|forward| {
                (
                    forward,
                    engine
                        .grid
                        .direction(forward)
                        .buckets
                        .iter()
                        .flatten()
                        .flat_map(|bucket| &bucket.arcs)
                        .filter(|arc| arc.physical == terminal)
                        .map(|arc| arc.id)
                        .collect::<std::collections::BTreeSet<_>>(),
                )
            })
            .into_iter()
            .collect();
        assert!(!engine.physical_arc_survives_fixing(terminal, &all_occurrences));
        // Query the retained directional domain with both directions enabled.
        // Changing the public option normally discards the working grid, which
        // would conceal whether its actual backward suffix still works.
        engine.controls.bidirectional = true;
        engine.controls.bidirectional_mode = 1;
        candidate
            .options
            .set("BALDESuseBidirectionalSearch", 1.)
            .unwrap();
        let mut child = candidate.fork_for_node();
        let repriced = child.price_phase(0, 10, false).unwrap();
        assert!(repriced.exact, "mode {mode}");
        assert!(
            repriced.minimum_reduced_cost < -2.9,
            "mode {mode}: {repriced:?}"
        );
        assert!(repriced.columns.iter().all(|route| route.arcs == [0, 1]));
        let enumerated = child.enumerate(0.).unwrap();
        assert!(!enumerated.is_empty(), "mode {mode}");
        assert!(enumerated.iter().all(|route| route.arcs == [0, 1]));
    }
}

#[test]
fn paired_bucket_arrival_uses_function_and_inverse_in_both_directions() {
    for waiting_gap in [false, true] {
        let (tail, head, consumption, forward_expected, backward_expected) = if waiting_gap {
            (
                crate::Window::new(3., 8.).unwrap(),
                crate::Window::new(6., 10.).unwrap(),
                vec![(1., 3.), (0., 8.)],
                vec![4, 4, 4, 3, 2, 1],
                vec![5, 5, 4, 3, 2],
            )
        } else {
            (
                crate::Window::new(0., 4.).unwrap(),
                crate::Window::new(1., 3.).unwrap(),
                vec![(1., 0.), (-0.5, 4.)],
                vec![2, 1, 1, 0, 0],
                vec![4, 2, 0],
            )
        };
        let mut graph = Graph::<2>::new(2, 1).unwrap();
        graph.set_bounds(0, 0, tail.lower, tail.upper).unwrap();
        graph.set_bounds(1, 0, head.lower, head.upper).unwrap();
        graph.add_arc(0, 1, -1.).unwrap();
        graph
            .set_coupled_resources(&[CoupledResource::PairedFront {
                mode: PairedResourceMode::CostUnbounded,
            }])
            .unwrap();
        let base = PreparedSlopeFunction::from_prepared_pairs(consumption).unwrap();
        let affine = |v| MachineRootInput::Affine(RawAffine::new(v, 0.).unwrap());
        assert!(matches!(
            graph.prepare_paired_front_program(
                0,
                &[MachineArcFunctionInput {
                    physical_arc: 0,
                    tail_bounds: tail,
                    waiting_bounds: head,
                    waiting_rate: 0,
                    base: &base,
                    dependent: affine(0.),
                    dependent_lower: affine(-100.),
                    dependent_upper: affine(100.),
                }],
                affine(0.),
                tail,
                head,
                FrontierBudget::default()
            ),
            FrontierOutcome::Complete(())
        ));
        graph
            .set_paired_waiting_windows_from_resource(0, 0)
            .unwrap();
        let grid = BucketGraph::prepare(&graph, [1.], 5, true).unwrap();
        // Actual paired-manager captures include a backward corner10 outside
        // the inverse's [4,9] domain; its sentinel clips to the tail upper8.
        for (forward, expected) in [(true, forward_expected), (false, backward_expected)] {
            let source = if forward { 0 } else { 1 };
            let target = if forward { 1 } else { 0 };
            for (k, expected) in expected.into_iter().enumerate() {
                let corner = grid.direction(forward).corners[source][k];
                let q = grid.project(&graph, forward, 0, corner).unwrap().unwrap();
                let value = if waiting_gap {
                    if forward {
                        (corner[0] + 1.).max(head.lower)
                    } else {
                        (corner[0] - 1.).min(tail.upper)
                    }
                } else if forward {
                    1. + 0.5 * corner[0]
                } else {
                    2. * corner[0] - 2.
                };
                assert_eq!(q[0], value, "waiting_gap={waiting_gap}, forward={forward}");
                assert_eq!(grid.bucket_id(!forward, target, q), Some(expected));
            }
        }
    }
}

#[test]
fn unbound_paired_program_does_not_project_a_nonexistent_scalar_axis() {
    for bidirectional in [0., 1., 2.] {
        let mut candidate = solver_with_binding(None, false);
        candidate
            .set_option("BALDESuseBidirectionalSearch", bidirectional)
            .unwrap();
        let priced = candidate.price_phase(0, 10, false).unwrap();
        assert!(priced.exact);
        assert_eq!(priced.minimum_reduced_cost, -2.);
        assert!(!priced.columns.is_empty());
        assert!(priced.columns.iter().all(|route| route.arcs == [0, 1]));
        assert!(candidate
            .engine
            .as_ref()
            .unwrap()
            .grid
            .resource_slots
            .is_empty());
    }
}

#[test]
fn signed_paired_split_completion_and_fixing_with_zero_terminal_ceiling_keep_a_negative_route() {
    let root =
        PreparedSlopeFunction::from_prepared_pairs(vec![(-1., 0.), (-1., 2.), (-2., 4.)]).unwrap();
    let mut baseline = solver_with_program(Some(&root), true, 0.);
    baseline
        .set_option("BALDESuseBidirectionalSearch", 0.)
        .unwrap();
    let unsplit = baseline.price_phase(0, 10, false).unwrap();
    assert_eq!(unsplit.minimum_reduced_cost, -9.);
    for mode in [0, 1, 2] {
        let mut candidate = solver_with_program(Some(&root), true, 0.);
        candidate
            .set_option("BALDESlabelSplitStrategy", 2.)
            .unwrap();
        candidate
            .set_option("BALDESuseBidirectionalSearch", 0.)
            .unwrap();
        let split = candidate.price_phase(0, 10, false).unwrap();
        assert!(
            split.statistics.generated > unsplit.statistics.generated,
            "fixture must actually emit split pieces"
        );
        candidate.complete_labels(false).unwrap();
        let report = candidate
            .engine
            .as_mut()
            .unwrap()
            .prepare_and_fix_bucket_arcs(&candidate.graph, &candidate.controls, 0., false, mode)
            .unwrap();
        assert!(report.checked > 0);
        let mut child = candidate.fork_for_node();
        let repriced = child.price_phase(0, 10, false).unwrap();
        assert!(repriced.exact, "mode {mode}");
        assert!(
            repriced.minimum_reduced_cost < -8.9,
            "mode {mode}: {repriced:?}"
        );
        assert!(!repriced.columns.is_empty(), "mode {mode}");
        assert!(repriced.columns.iter().all(|route| route.arcs == [0, 1]));
    }
}

#[test]
fn implicit_paired_axis_requires_a_positive_width_before_publishing_split_options() {
    let mut candidate = solver_with_binding(None, false);
    assert!(candidate
        .set_option("BALDESlabelDominanceStrategy", 1.)
        .unwrap_err()
        .to_string()
        .contains("positive configured paired-axis bucket width"));
    assert_eq!(
        candidate
            .options
            .get("BALDESlabelDominanceStrategy")
            .unwrap(),
        0.
    );
    assert!(candidate.engine.is_none());
    candidate.bucket_steps[0] = 1.;
    candidate
        .set_option("BALDESlabelDominanceStrategy", 1.)
        .unwrap();
    let result = candidate.price_phase(0, 10, false).unwrap();
    assert!(result.exact);
    assert_eq!(result.minimum_reduced_cost, -2.);
    let engine = candidate.engine.as_ref().unwrap();
    assert_eq!(engine.controls.paired_split_strategy, 2);
    assert_eq!(engine.grid.dimensions, 1);
    assert!(engine.grid.resource_slots.is_empty());
    assert_eq!(candidate.graph.resources.len(), 0);
}

#[test]
fn implicit_paired_axis_splits_without_adding_a_scalar_slot_in_each_search_direction() {
    for direction in [0., 1., 2.] {
        let mut baseline = solver_with_binding(None, false);
        baseline.bucket_steps[0] = 1.;
        baseline
            .set_option("BALDESuseBidirectionalSearch", direction)
            .unwrap();
        let reference = baseline.price_phase(0, 10, false).unwrap();
        for strategy in [1., 2.] {
            let mut candidate = solver_with_binding(None, false);
            candidate.bucket_steps[0] = 1.;
            candidate
                .set_option("BALDESuseBidirectionalSearch", direction)
                .unwrap();
            candidate
                .set_option("BALDESlabelSplitStrategy", strategy)
                .unwrap();
            let priced = candidate.price_phase(0, 10, false).unwrap();
            assert!(priced.exact, "direction {direction}, strategy {strategy}");
            assert_eq!(priced.minimum_reduced_cost, reference.minimum_reduced_cost);
            assert!(priced.columns.iter().all(|column| column.arcs == [0, 1]));
            let mut child = candidate.fork_for_node();
            let replay = child.price_phase(0, 10, false).unwrap();
            assert!(replay.exact);
            assert_eq!(replay.minimum_reduced_cost, -2.);
        }
    }
}

#[test]
fn implicit_paired_split_completion_and_fixing_use_the_virtual_base_coordinate() {
    let root =
        PreparedSlopeFunction::from_prepared_pairs(vec![(-1., 0.), (-1., 2.), (-2., 4.)]).unwrap();
    for mode in [0, 1, 2] {
        let mut candidate = solver_with_program(Some(&root), false, 0.);
        candidate.bucket_steps[0] = 1.;
        candidate
            .set_option("BALDESlabelSplitStrategy", 2.)
            .unwrap();
        candidate
            .set_option("BALDESuseBidirectionalSearch", 0.)
            .unwrap();
        let split = candidate.price_phase(0, 10, false).unwrap();
        assert!(split.exact);
        assert!(split.minimum_reduced_cost < -8.9);
        candidate.complete_labels(false).unwrap();
        let report = candidate
            .engine
            .as_mut()
            .unwrap()
            .prepare_and_fix_bucket_arcs(&candidate.graph, &candidate.controls, 0., false, mode)
            .unwrap();
        assert!(report.checked > 0);
        let mut child = candidate.fork_for_node();
        let repriced = child.price_phase(0, 10, false).unwrap();
        assert!(repriced.exact);
        assert!(repriced.minimum_reduced_cost < -8.9, "mode {mode}");
        assert!(child.graph.resources.is_empty());
        let routes = child.enumerate(0.).unwrap();
        assert!(!routes.is_empty(), "mode {mode}");
        assert!(routes
            .iter()
            .all(|route| route.arcs == [0, 1] && route.cost < -8.9));
    }
}

#[test]
fn backward_terminal_suffix_preserves_a_route_after_all_forward_terminal_occurrences_are_removed() {
    let mut candidate = solver_with_binding(None, false);
    candidate.bucket_steps[0] = 1.;
    candidate
        .set_option("BALDESuseBidirectionalSearch", 1.)
        .unwrap();
    let initial = candidate.price_phase(0, 10, false).unwrap();
    assert_eq!(initial.minimum_reduced_cost, -2.);
    let engine = candidate.engine.as_mut().unwrap();
    let terminal = 1;
    let forward_removed = engine
        .grid
        .forward
        .buckets
        .iter()
        .flatten()
        .flat_map(|bucket| &bucket.arcs)
        .filter(|arc| arc.physical == terminal)
        .map(|arc| arc.id)
        .collect::<std::collections::BTreeSet<_>>();
    assert!(!forward_removed.is_empty());
    assert!(engine
        .grid
        .backward
        .buckets
        .iter()
        .flatten()
        .flat_map(|bucket| &bucket.arcs)
        .any(|arc| arc.physical == terminal && arc.jump.is_none()));
    let removals = vec![(true, forward_removed), (false, Default::default())];
    assert!(engine.physical_arc_survives_fixing(terminal, &removals));
    // Publish the one-direction deletion. The next solve rebuilds labels, so
    // this exercises an actual surviving suffix rather than a cached solution.
    for bucket in engine.grid.forward.buckets.iter_mut().flatten() {
        bucket.arcs.retain(|arc| arc.physical != terminal);
    }
    engine.grid.regenerate_jumps(&candidate.graph, true);
    assert!(!engine
        .grid
        .forward
        .buckets
        .iter()
        .flatten()
        .flat_map(|bucket| &bucket.arcs)
        .any(|arc| arc.physical == terminal));
    let mut child = candidate.fork_for_node();
    let repriced = child.price_phase(0, 10, false).unwrap();
    assert!(repriced.exact);
    assert_eq!(repriced.minimum_reduced_cost, -2.);
    assert!(repriced.columns.iter().any(|route| route.arcs == [0, 1]));
    let enumerated = child.enumerate(0.).unwrap();
    assert!(!enumerated.is_empty());
    assert!(enumerated
        .iter()
        .all(|route| route.arcs == [0, 1] && route.cost == -2.));
}
