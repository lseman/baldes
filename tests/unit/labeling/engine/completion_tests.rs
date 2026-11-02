use super::*;
use crate::graph::coupled::CoupledResource;

fn model() -> Graph<2> {
    let mut graph = Graph::new(3, 0).unwrap();
    graph.add_arc(0, 1, -10.).unwrap();
    graph.add_arc(1, 2, -10.).unwrap();
    graph
        .set_coupled_resources(&[CoupledResource::CumulativeCost {
            initial: 0.,
            increments: vec![1., 1.],
            weights: vec![1., -2.],
            sink_upper: None,
            source_upper: None,
        }])
        .unwrap();
    graph
}

#[test]
fn stored_paired_completion_scans_only_after_labels_are_complete() {
    use crate::graph::coupled::descriptor::{
        FrontierBudget, FrontierOutcome, MachineRootInput, PreparedSlopeFunction, RawAffine,
    };
    use crate::graph::coupled::{MachineArcFunctionInput, PairedResourceMode};
    use crate::graph::Window;

    let mut graph = Graph::<2>::new(3, 1).unwrap();
    graph.vertices[0].elements.clear();
    graph.vertices[2].elements.clear();
    for vertex in 0..3 {
        graph.set_bounds(vertex, 0, 0., 4.).unwrap();
    }
    for (tail, head) in [(0, 1), (1, 2)] {
        let id = graph.add_arc(tail, head, -1.).unwrap();
        graph.arc_mut(id).unwrap().consumption[0] = 1.;
    }
    graph
        .set_coupled_resources(&[CoupledResource::PairedFront {
            mode: PairedResourceMode::CostUnbounded,
        }])
        .unwrap();
    let window = Window::new(0., 4.).unwrap();
    let base = PreparedSlopeFunction::from_prepared_pairs(vec![(1., 0.), (0., 4.)]).unwrap();
    let affine = |value| MachineRootInput::Affine(RawAffine::new(value, 0.).unwrap());
    let inputs: Vec<_> = (0..2)
        .map(|physical_arc| MachineArcFunctionInput {
            physical_arc,
            tail_bounds: window,
            waiting_bounds: window,
            waiting_rate: 0,
            base: &base,
            dependent: affine(0.25),
            dependent_lower: affine(0.),
            dependent_upper: affine(8.),
        })
        .collect();
    let result = graph.prepare_paired_front_program(
        0,
        &inputs,
        affine(0.),
        window,
        window,
        FrontierBudget::default(),
    );
    assert!(
        matches!(result, FrontierOutcome::Complete(())),
        "{result:?}"
    );
    graph
        .set_paired_waiting_windows_from_resource(0, 0)
        .unwrap();
    let mut engine = engine(&graph);
    engine
        .prepare_full_labels(&graph, &Controls::default())
        .unwrap();
    let vertex = engine.grid.internal[1];
    let id = engine.forward.buckets[vertex]
        .iter()
        .flat_map(|bucket| &bucket.active)
        .copied()
        .next()
        .unwrap();
    let mut label = engine.forward.labels.view(id).to_owned();
    // A stored label has already visited the meeting vertex. The completion
    // query used during extension runs before that vertex step, since the
    // opposite stored label already contains its elements.
    assert!(!engine.completion_exists(&graph, true, &label, 100., true));
    label.ng &= !engine.ng.vertex_bits[label.vertex];
    assert!(!engine.completion_exists(&graph, true, &label, -100., true));
    assert!(engine.completion_exists(&graph, true, &label, 100., true));
    engine.full_ready = false;
    assert!(engine.completion_exists(&graph, true, &label, -100., true));
}
fn engine(g: &Graph<2>) -> LabelingEngine<2> {
    LabelingEngine::prepare(
        g,
        EngineControls {
            bidirectional: true,
            bidirectional_mode: 1,
            direction_flags: 3,
            use_symmetric_execution: false,
            ..Default::default()
        },
        Phase {
            check_ng: true,
            jump_arcs: true,
            labels_per_bucket: i64::MAX,
        },
    )
    .unwrap()
}
#[test]
fn completion_rebuilds_after_changes_and_does_not_cross_preparations_or_forks() {
    let graph = model();
    let mut engine = engine(&graph);
    engine
        .prepare_full_labels(&graph, &Controls::default())
        .unwrap();
    assert!(engine
        .coupled_completion
        .iter()
        .all(|c| c.identity.is_some()));
    let v = engine.grid.internal[1];
    let id = engine.forward.buckets[v][0].active[0];
    let mut label = engine.forward.labels.view(id).to_owned();
    // A sparse model deliberately has no eager pair table. Install a
    // conservative test hull and a poisoned gate to exercise membership.
    let mut hull = crate::graph::coupled::CoupledStateBounds::default();
    hull.include(label.coupled_state.as_deref());
    engine.coupled_completion[1].queries[v] = vec![hull];
    engine.coupled_completion[1].pairs[v] = vec![Some(vec![
        crate::graph::coupled::CoupledPairSummary::default(),
    ])];
    // A valid standalone resource prefix outside this bucket's retained hull
    // must not reuse even a cached rejection for the bucket pair.
    label.coupled_state = Some(
        graph
            .coupled_kernel()
            .unwrap()
            .replay_state(&[0; 12])
            .unwrap()
            .unwrap()
            .0,
    );
    assert!(!engine.coupled_completion[1].queries[v][0].contains(label.coupled_state.as_deref()));
    engine.coupled_completion[1].pairs[v][0].as_mut().unwrap()[0].possible = false;
    let expected = engine.ctx().resource_bucket_key(
        &graph,
        true,
        label.coupled_state.as_deref(),
        &label.presence_state,
        label.key(),
        v,
        0,
    );
    assert_eq!(
        engine.ctx().bucket_completion(
            &graph,
            true,
            (&label).into(),
            label.key(),
            BucketId {
                vertex: v,
                bucket: 0
            },
            engine.backward.buckets[v][0].hot.key.first().copied(),
            f64::INFINITY
        ),
        Some(expected)
    );
    let child = engine.fork_for_node();
    assert!(child
        .coupled_completion
        .iter()
        .all(|c| c.identity.is_none()));
    let mut reinstalled = graph.clone();
    reinstalled
        .set_coupled_resources(graph.coupled_kernel().unwrap().definitions())
        .unwrap();
    assert_eq!(
        engine.ctx().resource_completion_key(
            &reinstalled,
            true,
            label.coupled_state.as_deref(),
            &label.presence_state,
            0.
        ),
        f64::NEG_INFINITY
    );
    engine.load_presence(&graph, vec![]).unwrap();
    assert!(engine
        .coupled_completion
        .iter()
        .all(|c| c.identity.is_none()));
    engine
        .prepare_full_labels(&graph, &Controls::default())
        .unwrap();
    assert!(engine.load_cuts(&graph, vec![], vec![]));
    assert!(engine
        .coupled_completion
        .iter()
        .all(|c| c.identity.is_none()));
}
#[test]
fn pair_preparation_budget_keeps_uncached_rows_as_unknown() {
    let graph = model();
    let mut engine = engine(&graph);
    for forward in [true, false] {
        let storage = engine.storage_mut(forward);
        *storage = LabelStorage::new([300]);
        for bucket in 0..300 {
            let label = StoredLabel {
                id: 0,
                local_id: 0,
                bucket: BucketId { vertex: 0, bucket },
                vertex: 0,
                cost: 0.,
                original_cost: 0.,
                resources: [0.; 2].into(),
                ng: 0,
                branch_state: None,
                presence_state: None,
                coupled_state: Some(graph.coupled_kernel().unwrap().root(forward)),
                parent: None,
                arc: None,
                hops: 0,
                expanded: false,
                cut_words: [0; LIMITED_WORDS],
                additional_cost: 0.,
                half_potential: 0.,
            };
            let hot = HotLabel::of(&label);
            let id = storage.retain(label);
            for _ in 0..8 {
                storage.buckets[0][bucket].active.push(id);
                storage.buckets[0][bucket].hot.push(hot);
            }
        }
    }
    engine.refresh_coupled_completion(&graph);
    let entries: usize = engine
        .coupled_completion
        .iter()
        .flat_map(|c| c.pairs.iter().flatten())
        .flatten()
        .map(Vec::len)
        .sum();
    assert!(entries > 0 && entries <= 65_536);
    assert!(engine
        .coupled_completion
        .iter()
        .flat_map(|c| c.pairs.iter().flatten())
        .any(Option::is_none));
}

#[test]
fn replacement_resources_rebuild_full_labels_even_at_unchanged_arc_costs() {
    let mut graph = model();
    let mut engine = engine(&graph);
    engine
        .prepare_full_labels(&graph, &Controls::default())
        .unwrap();
    let old = engine.coupled_identity.clone().unwrap();
    let definitions = graph.coupled_kernel().unwrap().definitions().to_vec();
    graph.set_coupled_resources(&definitions).unwrap();
    engine
        .prepare_full_labels(&graph, &Controls::default())
        .unwrap();
    assert!(!std::sync::Arc::ptr_eq(
        &old,
        engine.coupled_identity.as_ref().unwrap()
    ));
    assert!(std::sync::Arc::ptr_eq(
        engine.coupled_identity.as_ref().unwrap(),
        graph.coupled_kernel().unwrap().cache_identity()
    ));
    assert!(engine.full_ready);
}
#[test]
fn sparse_completion_skips_pair_tables_but_keeps_point_bounds() {
    let graph = model();
    let mut engine = engine(&graph);
    engine
        .prepare_full_labels(&graph, &Controls::default())
        .unwrap();
    assert!(engine.coupled_completion.iter().all(|c| c
        .pairs
        .iter()
        .flatten()
        .all(Option::is_none)));
    assert!(engine
        .coupled_completion
        .iter()
        .all(|c| c.queries.iter().all(Vec::is_empty)));
}

#[test]
fn enumeration_projection_borrows_payloads_and_matches_owned_completion() {
    use crate::cuts::soft_conflicts::{Mode, PresenceRow, PricingInput};
    let graph = model();
    let mut engine = engine(&graph);
    engine
        .load_presence(
            &graph,
            vec![PricingInput {
                row: PresenceRow::new(130, vec![(0, 129)], Mode::AnyPair).unwrap(),
                arc_events: vec![vec![0], vec![129]],
                dual: -0.5,
            }],
        )
        .unwrap();
    engine
        .prepare_full_labels(&graph, &Controls::default())
        .unwrap();
    let v = engine.grid.internal[1];
    let id = engine.forward.buckets[v]
        .iter()
        .flat_map(|b| &b.active)
        .copied()
        .next()
        .unwrap();
    let original = engine.forward.labels.view(id).to_owned();
    let label = crate::enumeration::Label {
        id: 0,
        vertex: original.vertex,
        resources: original.resources,
        reduced_cost: original.cost,
        original_cost: original.original_cost,
        visited: [0; 16],
        element_hash: 0,
        parent: None,
        arc: None,
        hops: original.hops,
        cut_words: [0; crate::cuts::state::FULL_WORDS],
        cut_cost: original.additional_cost,
        presence_state: original.presence_state.clone(),
        branch_state: None,
        coupled_state: original.coupled_state.clone(),
    };
    let projected = EnumerationProjection {
        label: &label,
        ng: 0,
        cut_words: [0; LIMITED_WORDS],
        half_potential: 0.,
    };
    assert!(std::ptr::eq(
        projected.presence_state(),
        &label.presence_state
    ));
    assert!(std::ptr::eq(
        projected.coupled_state().unwrap(),
        label.coupled_state.as_deref().unwrap()
    ));
    let owned = StoredLabel {
        bucket: BucketId::default(),
        ng: 0,
        ..original
    };
    for forward in [true, false] {
        for scan in [false, true] {
            for threshold in [-100., 0., 100.] {
                assert_eq!(
                    engine.enumeration_completion(&graph, forward, &label, threshold, scan),
                    engine.completion_exists(&graph, forward, &owned, threshold, scan)
                );
            }
        }
    }
}

#[test]
fn sweeps_reuse_id_scratch_and_do_not_replay_previous_terminals() {
    let graph = model();
    let mut engine = engine(&graph);
    let first = engine.solve(&graph, &Controls::default()).unwrap();
    let capacities = (
        engine.forward_scratch.labels.capacity(),
        engine.backward_scratch.labels.capacity(),
    );
    assert!(capacities.0 > 0 && capacities.1 > 0);
    for _ in 0..3 {
        let next = engine.solve(&graph, &Controls::default()).unwrap();
        assert_eq!(
            next.minimum_reduced_cost.to_bits(),
            first.minimum_reduced_cost.to_bits()
        );
        assert_eq!(next.statistics.generated, first.statistics.generated);
        assert_eq!(
            next.columns.iter().map(|r| &r.arcs).collect::<Vec<_>>(),
            first.columns.iter().map(|r| &r.arcs).collect::<Vec<_>>()
        );
        assert_eq!(
            capacities,
            (
                engine.forward_scratch.labels.capacity(),
                engine.backward_scratch.labels.capacity()
            )
        );
        assert!(
            engine.forward_scratch.pending.is_empty() && engine.backward_scratch.pending.is_empty()
        );
        assert!(
            engine.forward_scratch.terminals.is_empty()
                && engine.backward_scratch.terminals.is_empty()
        );
    }
}

#[test]
fn fixing_completion_modes_match_native_bound_and_ng_scan_cases() {
    let capture: serde_json::Value =
        serde_json::from_str(include_str!("fixing_completion_capture.json")).unwrap();
    assert_eq!(capture["cases"].as_array().unwrap().len(), 12);
    check_native_completion_cases(&capture);
}

#[test]
fn fixing_completion_strict_cutoff_boundaries_match_reference() {
    let capture: serde_json::Value =
        serde_json::from_str(include_str!("completion_boundary_capture.json")).unwrap();
    assert_eq!(capture["cases"].as_array().unwrap().len(), 972);
    check_native_completion_cases(&capture);
}

fn check_native_completion_cases(capture: &serde_json::Value) {
    for case in capture["cases"].as_array().unwrap() {
        let mut graph = Graph::<2>::new(3, 0).unwrap();
        graph.add_arc(0, 1, -2.).unwrap();
        graph.add_arc(1, 2, 0.).unwrap();
        let mut prepared = engine(&graph);
        prepared
            .prepare_full_labels(&graph, &Controls::default())
            .unwrap();
        let vertex = prepared.grid.internal[1];
        let id = prepared.forward.buckets[vertex]
            .iter()
            .flat_map(|b| &b.active)
            .copied()
            .next()
            .unwrap();
        let mut own = prepared.forward.labels.view(id).to_owned();
        own.cost = case["own_cost"].as_f64().unwrap();
        own.ng = case["own_ng"].as_u64().unwrap() as NgMask;
        let id = prepared.backward.buckets[vertex]
            .iter()
            .flat_map(|b| &b.active)
            .copied()
            .next()
            .unwrap();
        let mut other = prepared.backward.labels.view(id).to_owned();
        other.cost = case["other_cost"].as_f64().unwrap();
        other.ng = case["other_ng"].as_u64().unwrap() as NgMask;
        let hot = HotLabel::of(&other);
        let id = prepared.backward.retain(other);
        for row in &mut prepared.backward.buckets[vertex] {
            row.active.clear();
            row.hot = Default::default();
            row.lower_cost = NO_ROUTE_COST;
        }
        let row = &mut prepared.backward.buckets[vertex][0];
        row.active.push(id);
        row.hot.push(hot);
        row.lower_cost = case["lower"].as_f64().unwrap();
        assert_eq!(
            prepared.completion_exists(
                &graph,
                true,
                &own,
                case["threshold"].as_f64().unwrap(),
                case["scan"].as_bool().unwrap()
            ),
            case["exists"].as_bool().unwrap(),
            "{case}"
        );
    }
}

fn fixing_graph() -> Graph<2> {
    let mut graph = Graph::<2>::new(5, 1).unwrap();
    graph.elementary = true;
    for v in 0..5 {
        graph.set_bounds(v, 0, 0., 4.).unwrap();
    }
    for (tail, head, cost, use_) in [
        (0, 1, -5., 1.5),
        (1, 2, 0., 1.5),
        (0, 3, 50., 1.),
        (3, 2, 0., 1.5),
        (2, 4, 0., 0.5),
    ] {
        let arc = graph.add_arc(tail, head, cost).unwrap();
        graph.arc_mut(arc).unwrap().consumption[0] = use_;
    }
    graph
}

#[test]
fn fixing_scans_regenerated_jumps_even_when_the_pricing_phase_disables_them() {
    let graph = fixing_graph();
    let mut prepared = LabelingEngine::prepare(
        &graph,
        EngineControls {
            steps: [0.5, 0.].into(),
            bidirectional: true,
            bidirectional_mode: 1,
            direction_flags: 3,
            use_symmetric_execution: false,
            ..Default::default()
        },
        Phase {
            check_ng: true,
            jump_arcs: true,
            labels_per_bucket: i64::MAX,
        },
    )
    .unwrap();
    prepared
        .prepare_full_labels(&graph, &Controls::default())
        .unwrap();
    prepared.fix_bucket_arcs(&graph, 0.).unwrap();
    prepared
        .prepare_full_labels(&graph, &Controls::default())
        .unwrap();
    prepared.phase.jump_arcs = false;

    assert!([&prepared.grid.forward, &prepared.grid.backward]
        .into_iter()
        .flat_map(|direction| direction.buckets.iter().flatten())
        .flat_map(|bucket| &bucket.arcs)
        .any(|arc| arc.jump.is_some()));
    let expected = [&prepared.grid.forward, &prepared.grid.backward]
        .into_iter()
        .map(|direction| {
            direction
                .components
                .iter()
                .flatten()
                .map(|source| {
                    direction.buckets[source.vertex][source.bucket]
                        .arcs
                        .iter()
                        .filter(|arc| graph.arcs[arc.physical].reduced_cost < NO_ROUTE_COST)
                        .count()
                })
                .sum::<usize>()
        })
        .sum::<usize>();
    let report = prepared.fix_bucket_arcs(&graph, 0.).unwrap();
    assert_eq!(report.checked, expected);
}

#[test]
fn fixing_extends_a_jump_with_waiting_when_pricing_jumps_are_disabled() {
    let mut graph = Graph::<2>::new(3, 1).unwrap();
    for vertex in 0..3 {
        graph.set_bounds(vertex, 0, 0., 4.).unwrap();
    }
    graph.add_arc(0, 1, 0.).unwrap();
    graph.add_arc(1, 2, 0.).unwrap();
    let mut prepared = LabelingEngine::prepare(
        &graph,
        EngineControls {
            steps: [1., 0.].into(),
            use_symmetric_execution: false,
            ..Default::default()
        },
        Phase {
            check_ng: true,
            jump_arcs: false,
            labels_per_bucket: i64::MAX,
        },
    )
    .unwrap();
    prepared
        .prepare_full_labels(&graph, &Controls::default())
        .unwrap();
    let source = prepared.grid.internal[graph.source];
    let mut jump = prepared.grid.forward.buckets[source][0].arcs[0];
    jump.jump = Some(2);
    let root = prepared.forward.root.unwrap();
    let label = prepared.forward.labels.view(root);
    let ordinary = prepared.ctx().extend(&graph, true, &jump, label).unwrap();
    let fixing = prepared
        .extend_for_fixing(&graph, true, &jump, label)
        .unwrap();
    assert_eq!(ordinary.resources[0], 0.);
    assert_eq!(fixing.resources[0], 2.);
}

#[test]
fn fixing_cutoff_labels_never_escape_as_complete_opposite_coverage() {
    let graph = fixing_graph();
    for mode in [0, 1, 2] {
        let mut prepared = engine(&graph);
        prepared.controls.direction_flags = 1;
        let priced = prepared.solve(&graph, &Controls::default()).unwrap();
        assert_eq!(priced.minimum_reduced_cost, -5.);
        assert!(prepared.half_ready);
        let report = prepared
            .prepare_and_fix_bucket_arcs(&graph, &Controls::default(), 0., false, mode)
            .unwrap();
        assert!(report.eliminated_forward + report.eliminated_backward > 0);
        if mode != 0 {
            assert!(prepared.statistics.completion_pruned > 0, "mode {mode}");
            assert!(!prepared.full_ready && !prepared.half_ready);
        } else {
            assert_eq!(prepared.statistics.completion_pruned, 0);
            // Removing bucket arcs invalidates full coverage even without a completion gate.
            assert!(!prepared.full_ready && !prepared.half_ready);
        }
        prepared
            .prepare_full_labels(&graph, &Controls::default())
            .unwrap();
        assert!(prepared.full_ready);
        assert_eq!(
            prepared
                .solve(&graph, &Controls::default())
                .unwrap()
                .minimum_reduced_cost,
            -5.
        );
    }
}

#[test]
fn fixing_mode_options_and_expired_budget_preserve_retry_domain() {
    let mut solver = crate::pricing::Solver::<2>::new(5, 1).unwrap();
    solver.graph = fixing_graph();
    let baseline = solver.enumerate(1000.).unwrap();
    assert_eq!(baseline.len(), 2, "unfixed fixture route domain");
    assert_eq!(baseline.iter().filter(|route| route.cost < 1.).count(), 1);
    solver.restore_arcs();
    assert_eq!(
        solver
            .options
            .get("RCSPuseComplBoundsInRedCostFixing")
            .unwrap(),
        2.
    );
    for mode in [0., 1., 2., -1., 3.] {
        solver.restore_arcs();
        solver
            .set_option("RCSPuseComplBoundsInRedCostFixing", mode)
            .unwrap();
        let report = solver.fix(0.).unwrap();
        assert!(report.checked > 0, "mode {mode}");
        let routes = solver.enumerate(1.).unwrap();
        assert_eq!(
            routes.len(),
            1,
            "mode {mode}, fixing retained no cheap route"
        );
        assert_eq!(routes[0].arcs, [0, 1, 4]);
        solver.restore_arcs();
        assert_eq!(solver.enumerate(1000.).unwrap().len(), 2);
    }
    solver.controls.time_limit = std::time::Duration::ZERO;
    assert!(solver
        .enumerate(1000.)
        .unwrap_err()
        .message
        .contains("time limit"));
    assert!(solver.inspect_enumeration(1000.).unwrap().is_none());
    assert!(solver.fix(0.).unwrap_err().message.contains("time limit"));
    assert!(solver.inspect_enumeration(1000.).unwrap().is_none());
    solver.controls.time_limit = std::time::Duration::from_secs(10);
    assert_eq!(solver.enumerate(1000.).unwrap().len(), 2);
}

#[test]
fn fixing_completion_resource_scan_matches_reference_boundaries() {
    let capture: serde_json::Value =
        serde_json::from_str(include_str!("completion_resource_capture.json")).unwrap();
    let cases = capture["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 140);
    for case in cases {
        let mut graph = Graph::<2>::new(3, 1).unwrap();
        for vertex in 0..3 {
            graph.set_bounds(vertex, 0, 0., 2.).unwrap();
        }
        graph.add_arc(0, 1, -2.).unwrap();
        graph.add_arc(1, 2, 0.).unwrap();
        let mut prepared = LabelingEngine::prepare(
            &graph,
            EngineControls {
                steps: [10., 0.].into(),
                bidirectional: true,
                bidirectional_mode: 1,
                direction_flags: 3,
                use_symmetric_execution: false,
                ..Default::default()
            },
            Phase {
                check_ng: true,
                jump_arcs: true,
                labels_per_bucket: i64::MAX,
            },
        )
        .unwrap();
        prepared
            .prepare_full_labels(&graph, &Controls::default())
            .unwrap();
        let vertex = prepared.grid.internal[1];
        let id = prepared.forward.buckets[vertex]
            .iter()
            .flat_map(|b| &b.active)
            .copied()
            .next()
            .unwrap();
        let mut own = prepared.forward.labels.view(id).to_owned();
        own.cost = -2.;
        own.ng = 1;
        own.resources[0] = case["forward"].as_f64().unwrap();
        let id = prepared.backward.buckets[vertex]
            .iter()
            .flat_map(|b| &b.active)
            .copied()
            .next()
            .unwrap();
        let mut other = prepared.backward.labels.view(id).to_owned();
        other.cost = 0.;
        other.ng = if case["overlap"].as_bool().unwrap() {
            1
        } else {
            2
        };
        other.resources[0] = case["backward"].as_f64().unwrap();
        let hot = HotLabel::of(&other);
        let id = prepared.backward.retain(other);
        for row in &mut prepared.backward.buckets[vertex] {
            row.active.clear();
            row.hot = Default::default();
            row.lower_cost = NO_ROUTE_COST;
        }
        let row = &mut prepared.backward.buckets[vertex][0];
        row.active.push(id);
        row.hot.push(hot);
        row.lower_cost = 0.;
        assert_eq!(
            prepared.completion_exists(&graph, true, &own, 0., case["scan"].as_bool().unwrap()),
            case["exists"].as_bool().unwrap(),
            "{case}"
        );
    }
}

#[test]
fn fixing_completion_two_resource_scan_matches_reference_boundaries() {
    let capture: serde_json::Value =
        serde_json::from_str(include_str!("completion_two_resource_capture.json")).unwrap();
    let cases = capture["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 144);
    for case in cases {
        let mut graph = Graph::<2>::new(3, 2).unwrap();
        for vertex in 0..3 {
            graph.set_bounds(vertex, 0, 0., 2.).unwrap();
            graph.set_bounds(vertex, 1, 0., 1.).unwrap();
        }
        graph.add_arc(0, 1, -2.).unwrap();
        graph.add_arc(1, 2, 0.).unwrap();
        let mut prepared = LabelingEngine::prepare(
            &graph,
            EngineControls {
                steps: [10., 10.].into(),
                bidirectional: true,
                bidirectional_mode: 1,
                direction_flags: 3,
                use_symmetric_execution: false,
                ..Default::default()
            },
            Phase {
                check_ng: true,
                jump_arcs: true,
                labels_per_bucket: i64::MAX,
            },
        )
        .unwrap();
        prepared
            .prepare_full_labels(&graph, &Controls::default())
            .unwrap();
        let vertex = prepared.grid.internal[1];
        let id = prepared.forward.buckets[vertex]
            .iter()
            .flat_map(|b| &b.active)
            .copied()
            .next()
            .unwrap();
        let mut own = prepared.forward.labels.view(id).to_owned();
        own.cost = -2.;
        own.ng = 1;
        for r in 0..2 {
            own.resources[r] = case["forward"][r].as_f64().unwrap();
        }
        let id = prepared.backward.buckets[vertex]
            .iter()
            .flat_map(|b| &b.active)
            .copied()
            .next()
            .unwrap();
        let mut other = prepared.backward.labels.view(id).to_owned();
        other.cost = 0.;
        other.ng = if case["overlap"].as_bool().unwrap() {
            1
        } else {
            2
        };
        for r in 0..2 {
            other.resources[r] = case["backward"][r].as_f64().unwrap();
        }
        let hot = HotLabel::of(&other);
        let id = prepared.backward.retain(other);
        for row in &mut prepared.backward.buckets[vertex] {
            row.active.clear();
            row.hot = Default::default();
            row.lower_cost = NO_ROUTE_COST;
        }
        let row = &mut prepared.backward.buckets[vertex][0];
        row.active.push(id);
        row.hot.push(hot);
        row.lower_cost = 0.;
        assert_eq!(
            prepared.completion_exists(&graph, true, &own, 0., case["scan"].as_bool().unwrap()),
            case["exists"].as_bool().unwrap(),
            "{case}"
        );
    }
}

#[test]
fn fixing_completion_with_packing_cut_matches_reference_boundaries() {
    use crate::cuts::state::LimitedInput;
    use std::collections::{BTreeMap, BTreeSet};

    let capture: serde_json::Value =
        serde_json::from_str(include_str!("completion_cut_capture.json")).unwrap();
    let cases = capture["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 48);
    for case in cases {
        let mut graph = Graph::<2>::new(5, 1).unwrap();
        for vertex in 0..5 {
            graph.set_bounds(vertex, 0, 0., 2.).unwrap();
        }
        graph.add_arc(0, 3, -2.).unwrap();
        graph.add_arc(3, 4, 0.).unwrap();
        let mut prepared = LabelingEngine::prepare(
            &graph,
            EngineControls {
                steps: [10., 0.].into(),
                bidirectional: true,
                bidirectional_mode: 1,
                direction_flags: 3,
                use_symmetric_execution: false,
                ..Default::default()
            },
            Phase {
                check_ng: true,
                jump_arcs: true,
                labels_per_bucket: i64::MAX,
            },
        )
        .unwrap();
        let cut = LimitedInput {
            covering: false,
            type2: false,
            arc_memory: false,
            denominator: 2,
            dual: 1.,
            vertex_coefficient: vec![0, 1, 1, 1, 0],
            arc_coefficient: BTreeMap::new(),
            memory_vertices: vec![1, 2, 3],
            memory_arcs: BTreeSet::new(),
        };
        assert!(prepared.load_cuts(&graph, vec![cut], vec![]));
        prepared
            .prepare_full_labels(&graph, &Controls::default())
            .unwrap();
        let vertex = prepared.grid.internal[3];
        let id = prepared.forward.buckets[vertex]
            .iter()
            .flat_map(|b| &b.active)
            .copied()
            .next()
            .unwrap();
        let mut own = prepared.forward.labels.view(id).to_owned();
        own.cost = -2.;
        own.ng = 1;
        own.resources[0] = 1.;
        own.cut_words.fill(0);
        own.cut_words[0] = case["own"].as_u64().unwrap() as u32;
        own.additional_cost = 0.;
        let id = prepared.backward.buckets[vertex]
            .iter()
            .flat_map(|b| &b.active)
            .copied()
            .next()
            .unwrap();
        let mut other = prepared.backward.labels.view(id).to_owned();
        other.cost = 0.;
        other.ng = 2;
        other.resources[0] = 1.;
        other.cut_words.fill(0);
        other.cut_words[0] = case["other"].as_u64().unwrap() as u32;
        other.additional_cost = 0.;
        let hot = HotLabel::of(&other);
        let id = prepared.backward.retain(other);
        for row in &mut prepared.backward.buckets[vertex] {
            row.active.clear();
            row.hot = Default::default();
            row.lower_cost = NO_ROUTE_COST;
        }
        let row = &mut prepared.backward.buckets[vertex][0];
        row.active.push(id);
        row.hot.push(hot);
        row.lower_cost = -100.;
        assert_eq!(
            prepared.completion_exists(
                &graph,
                true,
                &own,
                case["threshold"].as_f64().unwrap(),
                case["scan"].as_bool().unwrap()
            ),
            case["exists"].as_bool().unwrap(),
            "{case}"
        );
    }
}

#[test]
fn fixing_completion_with_covering_cut_matches_reference_boundaries() {
    use crate::cuts::state::LimitedInput;
    use std::collections::{BTreeMap, BTreeSet};

    let capture: serde_json::Value =
        serde_json::from_str(include_str!("completion_covering_capture.json")).unwrap();
    let cases = capture["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 48);
    for case in cases {
        let mut graph = Graph::<2>::new(5, 1).unwrap();
        for vertex in 0..5 {
            graph.set_bounds(vertex, 0, 0., 2.).unwrap();
        }
        graph.add_arc(0, 3, -2.).unwrap();
        graph.add_arc(3, 4, 0.).unwrap();
        let mut prepared = LabelingEngine::prepare(
            &graph,
            EngineControls {
                steps: [10., 0.].into(),
                bidirectional: true,
                bidirectional_mode: 1,
                direction_flags: 3,
                use_symmetric_execution: false,
                ..Default::default()
            },
            Phase {
                check_ng: true,
                jump_arcs: true,
                labels_per_bucket: i64::MAX,
            },
        )
        .unwrap();
        let cut = LimitedInput {
            covering: true,
            type2: false,
            arc_memory: false,
            denominator: -2,
            dual: -1.,
            vertex_coefficient: vec![0, -1, -1, -1, 0],
            arc_coefficient: BTreeMap::new(),
            memory_vertices: vec![1, 2, 3],
            memory_arcs: BTreeSet::new(),
        };
        assert!(prepared.load_cuts(&graph, vec![cut], vec![]));
        prepared
            .prepare_full_labels(&graph, &Controls::default())
            .unwrap();
        let vertex = prepared.grid.internal[3];
        let id = prepared.forward.buckets[vertex]
            .iter()
            .flat_map(|b| &b.active)
            .copied()
            .next()
            .unwrap();
        let mut own = prepared.forward.labels.view(id).to_owned();
        own.cost = -2.;
        own.ng = 1;
        own.resources[0] = 1.;
        own.cut_words.fill(0);
        own.cut_words[0] = case["own"].as_u64().unwrap() as u32;
        own.additional_cost = 0.;
        let id = prepared.backward.buckets[vertex]
            .iter()
            .flat_map(|b| &b.active)
            .copied()
            .next()
            .unwrap();
        let mut other = prepared.backward.labels.view(id).to_owned();
        other.cost = 0.;
        other.ng = 2;
        other.resources[0] = 1.;
        other.cut_words.fill(0);
        other.cut_words[0] = case["other"].as_u64().unwrap() as u32;
        other.additional_cost = 0.;
        let hot = HotLabel::of(&other);
        let id = prepared.backward.retain(other);
        for row in &mut prepared.backward.buckets[vertex] {
            row.active.clear();
            row.hot = Default::default();
            row.lower_cost = NO_ROUTE_COST;
        }
        let row = &mut prepared.backward.buckets[vertex][0];
        row.active.push(id);
        row.hot.push(hot);
        row.lower_cost = -100.;
        assert_eq!(
            prepared.completion_exists(
                &graph,
                true,
                &own,
                case["threshold"].as_f64().unwrap(),
                case["scan"].as_bool().unwrap()
            ),
            case["exists"].as_bool().unwrap(),
            "{case}"
        );
    }
}

#[test]
fn fixing_completion_equality_resource_matches_reference_boundaries() {
    let capture: serde_json::Value =
        serde_json::from_str(include_str!("completion_equality_capture.json")).unwrap();
    let cases = capture["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 144);
    for case in cases {
        let mut graph = Graph::<2>::new(3, 2).unwrap();
        graph.resources[1] = crate::graph::ResourceKind::Equality;
        for vertex in 0..3 {
            graph.set_bounds(vertex, 0, 0., 2.).unwrap();
            graph
                .set_bounds(vertex, 1, 0., if vertex == 1 { 1. } else { 0. })
                .unwrap();
        }
        graph.add_arc(0, 1, -2.).unwrap();
        graph.add_arc(1, 2, 0.).unwrap();
        let mut prepared = LabelingEngine::prepare(
            &graph,
            EngineControls {
                steps: [10., 10.].into(),
                bidirectional: true,
                bidirectional_mode: 1,
                direction_flags: 3,
                use_symmetric_execution: false,
                ..Default::default()
            },
            Phase {
                check_ng: true,
                jump_arcs: true,
                labels_per_bucket: i64::MAX,
            },
        )
        .unwrap();
        prepared
            .prepare_full_labels(&graph, &Controls::default())
            .unwrap();
        let vertex = prepared.grid.internal[1];
        let id = prepared.forward.buckets[vertex]
            .iter()
            .flat_map(|b| &b.active)
            .copied()
            .next()
            .unwrap();
        let mut own = prepared.forward.labels.view(id).to_owned();
        own.cost = -2.;
        own.ng = 1;
        for r in 0..2 {
            own.resources[r] = case["forward"][r].as_f64().unwrap();
        }
        let id = prepared.backward.buckets[vertex]
            .iter()
            .flat_map(|b| &b.active)
            .copied()
            .next()
            .unwrap();
        let mut other = prepared.backward.labels.view(id).to_owned();
        other.cost = 0.;
        other.ng = if case["overlap"].as_bool().unwrap() {
            1
        } else {
            2
        };
        for r in 0..2 {
            other.resources[r] = case["backward"][r].as_f64().unwrap();
        }
        let hot = HotLabel::of(&other);
        let id = prepared.backward.retain(other);
        for row in &mut prepared.backward.buckets[vertex] {
            row.active.clear();
            row.hot = Default::default();
            row.lower_cost = NO_ROUTE_COST;
        }
        let row = &mut prepared.backward.buckets[vertex][0];
        row.active.push(id);
        row.hot.push(hot);
        row.lower_cost = 0.;
        assert_eq!(
            prepared.completion_exists(&graph, true, &own, 0., case["scan"].as_bool().unwrap()),
            case["exists"].as_bool().unwrap(),
            "{case}"
        );
    }
}

#[test]
fn fixing_completion_with_two_covering_cuts_matches_reference() {
    use crate::cuts::state::LimitedInput;
    use std::collections::{BTreeMap, BTreeSet};

    let capture: serde_json::Value =
        serde_json::from_str(include_str!("completion_two_covering_capture.json")).unwrap();
    let cases = capture["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 224);
    for case in cases {
        let mut graph = Graph::<2>::new(5, 1).unwrap();
        for vertex in 0..5 {
            graph.set_bounds(vertex, 0, 0., 2.).unwrap();
        }
        graph.add_arc(0, 3, -2.).unwrap();
        graph.add_arc(3, 4, 0.).unwrap();
        let mut prepared = LabelingEngine::prepare(
            &graph,
            EngineControls {
                steps: [10., 0.].into(),
                bidirectional: true,
                bidirectional_mode: 1,
                direction_flags: 3,
                use_symmetric_execution: false,
                ..Default::default()
            },
            Phase {
                check_ng: true,
                jump_arcs: true,
                labels_per_bucket: i64::MAX,
            },
        )
        .unwrap();
        let cut = LimitedInput {
            covering: true,
            type2: false,
            arc_memory: false,
            denominator: -2,
            dual: -1.,
            vertex_coefficient: vec![0, -1, -1, 0, 0],
            arc_coefficient: BTreeMap::new(),
            memory_vertices: vec![1, 2, 3],
            memory_arcs: BTreeSet::new(),
        };
        let cut_two = LimitedInput {
            vertex_coefficient: vec![0, 0, -1, -1, 0],
            ..cut.clone()
        };
        assert!(prepared.load_cuts(&graph, vec![cut, cut_two], vec![]));
        prepared
            .prepare_full_labels(&graph, &Controls::default())
            .unwrap();
        let vertex = prepared.grid.internal[3];
        let id = prepared.forward.buckets[vertex]
            .iter()
            .flat_map(|b| &b.active)
            .copied()
            .next()
            .unwrap();
        let mut own = prepared.forward.labels.view(id).to_owned();
        own.cost = -2.;
        own.ng = 1;
        own.resources[0] = 1.;
        own.cut_words.fill(0);
        let own_bits = case["own"].as_u64().unwrap() as u32;
        own.cut_words[0] = (own_bits & 1) | ((own_bits & 2) << 6);
        own.additional_cost = 0.;
        let id = prepared.backward.buckets[vertex]
            .iter()
            .flat_map(|b| &b.active)
            .copied()
            .next()
            .unwrap();
        let mut other = prepared.backward.labels.view(id).to_owned();
        other.cost = 0.;
        other.ng = 2;
        other.resources[0] = 1.;
        other.cut_words.fill(0);
        let other_bits = case["other"].as_u64().unwrap() as u32;
        other.cut_words[0] = (other_bits & 1) | ((other_bits & 2) << 6);
        other.additional_cost = 0.;
        let hot = HotLabel::of(&other);
        let id = prepared.backward.retain(other);
        for row in &mut prepared.backward.buckets[vertex] {
            row.active.clear();
            row.hot = Default::default();
            row.lower_cost = NO_ROUTE_COST;
        }
        let row = &mut prepared.backward.buckets[vertex][0];
        row.active.push(id);
        row.hot.push(hot);
        row.lower_cost = -100.;
        assert_eq!(
            prepared.completion_exists(
                &graph,
                true,
                &own,
                case["threshold"].as_f64().unwrap(),
                case["scan"].as_bool().unwrap()
            ),
            case["exists"].as_bool().unwrap(),
            "{case}"
        );
    }
}

#[test]
fn fixing_completion_predecessor_traversal_matches_reference() {
    use crate::graph::buckets::{BucketId, Predecessor};

    let capture: serde_json::Value =
        serde_json::from_str(include_str!("completion_multibucket_capture.json")).unwrap();
    let cases = capture["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 1152);
    for case in cases {
        let mut graph = Graph::<2>::new(3, 1).unwrap();
        for vertex in 0..3 {
            graph.set_bounds(vertex, 0, 0., 2.).unwrap();
        }
        graph.add_arc(0, 1, -2.).unwrap();
        graph.add_arc(1, 2, 0.).unwrap();
        let mut prepared = LabelingEngine::prepare(
            &graph,
            EngineControls {
                steps: [1., 0.].into(),
                bidirectional: true,
                bidirectional_mode: 1,
                direction_flags: 3,
                use_symmetric_execution: false,
                ..Default::default()
            },
            Phase {
                check_ng: true,
                jump_arcs: true,
                labels_per_bucket: i64::MAX,
            },
        )
        .unwrap();
        prepared
            .prepare_full_labels(&graph, &Controls::default())
            .unwrap();
        let vertex = prepared.grid.internal[1];
        let id = prepared.forward.buckets[vertex]
            .iter()
            .flat_map(|b| &b.active)
            .copied()
            .next()
            .unwrap();
        let mut own = prepared.forward.labels.view(id).to_owned();
        own.cost = -2.;
        own.ng = 1;
        own.resources[0] = 0.;
        let id = prepared.backward.buckets[vertex]
            .iter()
            .flat_map(|b| &b.active)
            .copied()
            .next()
            .unwrap();
        let template = prepared.backward.labels.view(id).to_owned();
        assert_eq!(prepared.backward.buckets[vertex].len(), 3);
        let predecessor = |bucket, minimum| Predecessor {
            bucket: BucketId { vertex, bucket },
            minimum,
        };
        let topology = case["topology"].as_u64().unwrap();
        let buckets = &mut prepared.grid.backward.buckets[vertex];
        buckets[0].predecessors.clear();
        buckets[1].predecessors = vec![predecessor(0, 0)];
        buckets[2].predecessors = vec![predecessor(1, if topology == 0 { 0 } else { 1 })];
        if topology == 2 {
            buckets[2].predecessors.push(predecessor(0, 0));
        }
        let present = case["present"].as_u64().unwrap();
        let high = case["high"].as_u64().unwrap();
        let target = case["target"].as_u64().unwrap() as usize;
        for k in 0..3 {
            let mut other = template.clone();
            other.bucket = BucketId { vertex, bucket: k };
            other.cost = 0.;
            other.resources[0] = 2.;
            other.ng = if k == target { 2 } else { 1 };
            let hot = HotLabel::of(&other);
            let id = prepared.backward.retain(other);
            let row = &mut prepared.backward.buckets[vertex][k];
            row.active.clear();
            row.hot = Default::default();
            row.lower_cost = if high & (1 << k) != 0 { 2. } else { 0. };
            if present & (1 << k) != 0 {
                row.active.push(id);
                row.hot.push(hot);
            }
        }
        assert_eq!(
            prepared.completion_exists(&graph, true, &own, 0., case["scan"].as_bool().unwrap()),
            case["exists"].as_bool().unwrap(),
            "{case}"
        );
    }
}

#[test]
fn custom_label_join_matches_controlled_native_resource_and_overlap_cases() {
    // The native helper used copied labels on a live registered custom owner.
    // Both custom states correspond to one arc of increment .25 and weight .125.
    let mut graph = Graph::<2>::new(3, 1).unwrap();
    for v in 0..3 {
        graph.set_bounds(v, 0, 0., 2.).unwrap();
    }
    graph.add_arc(0, 1, 0.5).unwrap();
    graph.add_arc(1, 2, -0.25).unwrap();
    graph
        .set_coupled_resources(&[CoupledResource::CustomResource {
            increments: vec![0.25, 0.25],
            weights: vec![0.125, 0.125],
            forward_bound: None,
            backward_bound: None,
        }])
        .unwrap();
    let kernel = graph.coupled_kernel().unwrap();
    let root_f = kernel.root(true);
    let root_b = kernel.root(false);
    let mut fstate = kernel.extend(&root_f, 0, true).unwrap().unwrap().0;
    let mut bstate = kernel.extend(&root_b, 1, false).unwrap().unwrap().0;
    fstate.account_extension(&root_f, 0.5, 0.5).unwrap();
    bstate.account_extension(&root_b, -0.25, -0.25).unwrap();
    let make = |state, cost, resource, mask| StoredLabel {
        id: 0,
        local_id: 0,
        bucket: BucketId {
            vertex: 0,
            bucket: 0,
        },
        vertex: 1,
        cost,
        original_cost: cost,
        resources: [resource, 0.].into(),
        ng: mask,
        branch_state: None,
        presence_state: None,
        coupled_state: Some(state),
        parent: None,
        arc: None,
        hops: 1,
        expanded: false,
        cut_words: [0; LIMITED_WORDS],
        additional_cost: 0.,
        half_potential: 0.,
    };
    let engine = engine(&graph);
    for (case, row) in include_str!("custom_join_reference_capture.txt")
        .lines()
        .enumerate()
    {
        let fields: Vec<_> = row.split_whitespace().collect();
        let left = make(
            fstate.clone(),
            0.5,
            fields[0].parse().unwrap(),
            fields[2].parse().unwrap(),
        );
        let right = make(
            bstate.clone(),
            -0.25,
            fields[1].parse().unwrap(),
            fields[3].parse().unwrap(),
        );
        let accepted = fields[4] == "1";
        let expected: u64 = fields[5].parse().unwrap();
        let result = engine.join_cost(&graph, 1, &left, &right);
        assert_eq!(
            result.map(f64::to_bits),
            accepted.then_some(expected),
            "case={case}, row={row}"
        );
    }
}

#[test]
fn custom_label_join_matches_native_with_two_active_packing_cuts() {
    use crate::cuts::state::LimitedInput;
    use std::collections::{BTreeMap, BTreeSet};

    // Native has two active packing fields (word 0, shifts 0 and 1). Its
    // controlled labels have one custom arc per direction and standard .5/1.5.
    let mut graph = Graph::<2>::new(3, 1).unwrap();
    for v in 0..3 {
        graph.set_bounds(v, 0, 0., 2.).unwrap();
    }
    graph.add_arc(0, 1, 0.5).unwrap();
    graph.add_arc(1, 2, -0.25).unwrap();
    graph
        .set_coupled_resources(&[CoupledResource::CustomResource {
            increments: vec![0.25, 0.25],
            weights: vec![0.125, 0.125],
            forward_bound: None,
            backward_bound: None,
        }])
        .unwrap();
    let kernel = graph.coupled_kernel().unwrap();
    let root_f = kernel.root(true);
    let root_b = kernel.root(false);
    let mut fstate = kernel.extend(&root_f, 0, true).unwrap().unwrap().0;
    let mut bstate = kernel.extend(&root_b, 1, false).unwrap().unwrap().0;
    fstate.account_extension(&root_f, 0.5, 0.5).unwrap();
    bstate.account_extension(&root_b, -0.25, -0.25).unwrap();
    let make = |state, cost, resource, mask, word, extra| {
        let mut words = [0; LIMITED_WORDS];
        words[0] = word;
        StoredLabel {
            id: 0,
            local_id: 0,
            bucket: BucketId {
                vertex: 0,
                bucket: 0,
            },
            vertex: 1,
            cost,
            original_cost: cost,
            resources: [resource, 0.].into(),
            ng: mask,
            branch_state: None,
            presence_state: None,
            coupled_state: Some(state),
            parent: None,
            arc: None,
            hops: 1,
            expanded: false,
            cut_words: words,
            additional_cost: extra,
            half_potential: 0.,
        }
    };
    let cut = LimitedInput {
        covering: false,
        type2: false,
        arc_memory: false,
        denominator: 2,
        dual: 1.,
        vertex_coefficient: vec![0, 1, 0],
        arc_coefficient: BTreeMap::new(),
        memory_vertices: vec![0, 1, 2],
        memory_arcs: BTreeSet::new(),
    };
    let mut engine = engine(&graph);
    assert!(engine.load_cuts(&graph, vec![cut.clone(), cut], vec![]));
    let fields = &engine.cuts.info.vertices[1].dominance;
    assert_eq!(fields.len(), 2);
    assert_eq!(
        (
            fields[0].word,
            fields[0].shift,
            fields[0].mask,
            fields[0].denominator,
            fields[0].cost
        ),
        (0, 0, 1, 2, 1.)
    );
    assert_eq!(
        (
            fields[1].word,
            fields[1].shift,
            fields[1].mask,
            fields[1].denominator,
            fields[1].cost
        ),
        (0, 1, 1, 2, 1.)
    );
    assert_eq!(engine.cuts.info.vertex_cut_potential[1], 0.);
    for (case, row) in include_str!("custom_cut_join_reference_capture.txt")
        .lines()
        .enumerate()
    {
        let fields: Vec<_> = row.split_whitespace().collect();
        let left_word = fields[0].parse().unwrap();
        let right_word = fields[1].parse().unwrap();
        let overlap = fields[2] == "1";
        let extra = fields[3].parse().unwrap();
        let accepted = fields[4] == "1";
        let expected: u64 = fields[5].parse().unwrap();
        let left = make(fstate.clone(), 0.5, 0.5, 1, left_word, extra);
        let right = make(
            bstate.clone(),
            -0.25,
            1.5,
            if overlap { 1 } else { 2 },
            right_word,
            -0.125,
        );
        let result = engine.join_cost(&graph, 1, &left, &right);
        assert_eq!(
            result.map(f64::to_bits),
            accepted.then_some(expected),
            "case={case}, row={row}"
        );
    }
}

#[test]
fn custom_label_join_matches_controlled_native_mixed_cut_fields() {
    use crate::cuts::state::LimitedInput;
    use std::collections::{BTreeMap, BTreeSet};

    // Controlled native descriptors have packing and covering fields (shifts 0 and 1). Its
    // controlled labels have one custom arc per direction and standard .5/1.5.
    let mut graph = Graph::<2>::new(3, 1).unwrap();
    for v in 0..3 {
        graph.set_bounds(v, 0, 0., 2.).unwrap();
    }
    graph.add_arc(0, 1, 0.5).unwrap();
    graph.add_arc(1, 2, -0.25).unwrap();
    graph
        .set_coupled_resources(&[CoupledResource::CustomResource {
            increments: vec![0.25, 0.25],
            weights: vec![0.125, 0.125],
            forward_bound: None,
            backward_bound: None,
        }])
        .unwrap();
    let kernel = graph.coupled_kernel().unwrap();
    let root_f = kernel.root(true);
    let root_b = kernel.root(false);
    let mut fstate = kernel.extend(&root_f, 0, true).unwrap().unwrap().0;
    let mut bstate = kernel.extend(&root_b, 1, false).unwrap().unwrap().0;
    fstate.account_extension(&root_f, 0.5, 0.5).unwrap();
    bstate.account_extension(&root_b, -0.25, -0.25).unwrap();
    let make = |state, cost, resource, mask, word, extra| {
        let mut words = [0; LIMITED_WORDS];
        words[0] = word;
        StoredLabel {
            id: 0,
            local_id: 0,
            bucket: BucketId {
                vertex: 0,
                bucket: 0,
            },
            vertex: 1,
            cost,
            original_cost: cost,
            resources: [resource, 0.].into(),
            ng: mask,
            branch_state: None,
            presence_state: None,
            coupled_state: Some(state),
            parent: None,
            arc: None,
            hops: 1,
            expanded: false,
            cut_words: words,
            additional_cost: extra,
            half_potential: 0.,
        }
    };
    let cut = LimitedInput {
        covering: false,
        type2: false,
        arc_memory: false,
        denominator: 2,
        dual: 1.,
        vertex_coefficient: vec![0, 1, 0],
        arc_coefficient: BTreeMap::new(),
        memory_vertices: vec![0, 1, 2],
        memory_arcs: BTreeSet::new(),
    };
    let covering = LimitedInput {
        covering: true,
        denominator: -2,
        dual: -1.,
        vertex_coefficient: vec![0, -1, 0],
        ..cut.clone()
    };
    let mut engine = engine(&graph);
    assert!(engine.load_cuts(&graph, vec![cut, covering], vec![]));
    let fields = &engine.cuts.info.vertices[1].dominance;
    assert_eq!(fields.len(), 2);
    assert_eq!(
        (
            fields[0].word,
            fields[0].shift,
            fields[0].mask,
            fields[0].denominator,
            fields[0].cost
        ),
        (0, 0, 1, 2, 1.)
    );
    assert_eq!(
        (
            fields[1].word,
            fields[1].shift,
            fields[1].mask,
            fields[1].denominator,
            fields[1].cost
        ),
        (0, 1, 1, -2, -1.)
    );
    assert_eq!(engine.cuts.info.vertex_cut_potential[1], 0.);
    for (case, row) in include_str!("custom_mixed_cut_join_reference_capture.txt")
        .lines()
        .enumerate()
    {
        let fields: Vec<_> = row.split_whitespace().collect();
        let left_word = fields[0].parse().unwrap();
        let right_word = fields[1].parse().unwrap();
        let overlap = fields[2] == "1";
        let extra = fields[3].parse().unwrap();
        let accepted = fields[4] == "1";
        let expected: u64 = fields[5].parse().unwrap();
        let left = make(fstate.clone(), 0.5, 0.5, 1, left_word, extra);
        let right = make(
            bstate.clone(),
            -0.25,
            1.5,
            if overlap { 1 } else { 2 },
            right_word,
            -0.125,
        );
        let result = engine.join_cost(&graph, 1, &left, &right);
        assert_eq!(
            result.map(f64::to_bits),
            accepted.then_some(expected),
            "case={case}, row={row}"
        );
    }
}

#[test]
fn custom_label_join_matches_naturally_prepared_native_mixed_cuts() {
    use crate::cuts::state::LimitedInput;
    use std::collections::{BTreeMap, BTreeSet};

    // Native prepared covering and packing fields (shifts 0 and 7). Its
    // controlled labels have one custom arc per direction and standard .5/1.5.
    let mut graph = Graph::<2>::new(3, 1).unwrap();
    for v in 0..3 {
        graph.set_bounds(v, 0, 0., 2.).unwrap();
    }
    graph.add_arc(0, 1, 0.5).unwrap();
    graph.add_arc(1, 2, -0.25).unwrap();
    graph
        .set_coupled_resources(&[CoupledResource::CustomResource {
            increments: vec![0.25, 0.25],
            weights: vec![0.125, 0.125],
            forward_bound: None,
            backward_bound: None,
        }])
        .unwrap();
    let kernel = graph.coupled_kernel().unwrap();
    let root_f = kernel.root(true);
    let root_b = kernel.root(false);
    let mut fstate = kernel.extend(&root_f, 0, true).unwrap().unwrap().0;
    let mut bstate = kernel.extend(&root_b, 1, false).unwrap().unwrap().0;
    fstate.account_extension(&root_f, 0.5, 0.5).unwrap();
    bstate.account_extension(&root_b, -0.25, -0.25).unwrap();
    let make = |state, cost, resource, mask, word, extra| {
        let mut words = [0; LIMITED_WORDS];
        words[0] = word;
        StoredLabel {
            id: 0,
            local_id: 0,
            bucket: BucketId {
                vertex: 0,
                bucket: 0,
            },
            vertex: 1,
            cost,
            original_cost: cost,
            resources: [resource, 0.].into(),
            ng: mask,
            branch_state: None,
            presence_state: None,
            coupled_state: Some(state),
            parent: None,
            arc: None,
            hops: 1,
            expanded: false,
            cut_words: words,
            additional_cost: extra,
            half_potential: 0.,
        }
    };
    let cut = LimitedInput {
        covering: false,
        type2: false,
        arc_memory: false,
        denominator: 2,
        dual: 1.,
        vertex_coefficient: vec![0, 1, 0],
        arc_coefficient: BTreeMap::new(),
        memory_vertices: vec![0, 1, 2],
        memory_arcs: BTreeSet::new(),
    };
    let covering = LimitedInput {
        covering: true,
        denominator: -2,
        dual: -1.,
        vertex_coefficient: vec![0, -1, 0],
        ..cut.clone()
    };
    let mut engine = engine(&graph);
    assert!(engine.load_cuts(&graph, vec![covering, cut], vec![]));
    let fields = &engine.cuts.info.vertices[1].dominance;
    assert_eq!(fields.len(), 2);
    assert_eq!(
        (
            fields[0].word,
            fields[0].shift,
            fields[0].mask,
            fields[0].denominator,
            fields[0].cost
        ),
        (0, 0, 1, -2, -1.)
    );
    assert_eq!(
        (
            fields[1].word,
            fields[1].shift,
            fields[1].mask,
            fields[1].denominator,
            fields[1].cost
        ),
        (0, 7, 1, 2, 1.)
    );
    assert_eq!(engine.cuts.info.vertex_cut_potential[1], 0.);
    for (case, row) in include_str!("custom_mixed_prepared_join_reference_capture.txt")
        .lines()
        .enumerate()
    {
        let fields: Vec<_> = row.split_whitespace().collect();
        let left: u32 = fields[0].parse().unwrap();
        let left_word = (left & 1) | ((left & 2) << 6);
        let right: u32 = fields[1].parse().unwrap();
        let right_word = (right & 1) | ((right & 2) << 6);
        let overlap = fields[2] == "1";
        let extra = fields[3].parse().unwrap();
        let accepted = fields[4] == "1";
        let expected: u64 = fields[5].parse().unwrap();
        let left = make(fstate.clone(), 0.5, 0.5, 1, left_word, extra);
        let right = make(
            bstate.clone(),
            -0.25,
            1.5,
            if overlap { 1 } else { 2 },
            right_word,
            -0.125,
        );
        let result = engine.join_cost(&graph, 1, &left, &right);
        assert_eq!(
            result.map(f64::to_bits),
            accepted.then_some(expected),
            "case={case}, row={row}"
        );
    }
}

#[test]
fn custom_label_join_matches_native_with_two_active_covering_cuts() {
    use crate::cuts::state::LimitedInput;
    use std::collections::{BTreeMap, BTreeSet};

    // Native has two active covering fields (word 0, shifts 0 and 7). Its
    // controlled labels have one custom arc per direction and standard .5/1.5.
    let mut graph = Graph::<2>::new(3, 1).unwrap();
    for v in 0..3 {
        graph.set_bounds(v, 0, 0., 2.).unwrap();
    }
    graph.add_arc(0, 1, 0.5).unwrap();
    graph.add_arc(1, 2, -0.25).unwrap();
    graph
        .set_coupled_resources(&[CoupledResource::CustomResource {
            increments: vec![0.25, 0.25],
            weights: vec![0.125, 0.125],
            forward_bound: None,
            backward_bound: None,
        }])
        .unwrap();
    let kernel = graph.coupled_kernel().unwrap();
    let root_f = kernel.root(true);
    let root_b = kernel.root(false);
    let mut fstate = kernel.extend(&root_f, 0, true).unwrap().unwrap().0;
    let mut bstate = kernel.extend(&root_b, 1, false).unwrap().unwrap().0;
    fstate.account_extension(&root_f, 0.5, 0.5).unwrap();
    bstate.account_extension(&root_b, -0.25, -0.25).unwrap();
    let make = |state, cost, resource, mask, word, extra| {
        let mut words = [0; LIMITED_WORDS];
        words[0] = word;
        StoredLabel {
            id: 0,
            local_id: 0,
            bucket: BucketId {
                vertex: 0,
                bucket: 0,
            },
            vertex: 1,
            cost,
            original_cost: cost,
            resources: [resource, 0.].into(),
            ng: mask,
            branch_state: None,
            presence_state: None,
            coupled_state: Some(state),
            parent: None,
            arc: None,
            hops: 1,
            expanded: false,
            cut_words: words,
            additional_cost: extra,
            half_potential: 0.,
        }
    };
    let cut = LimitedInput {
        covering: true,
        type2: false,
        arc_memory: false,
        denominator: -2,
        dual: -1.,
        vertex_coefficient: vec![0, -1, 0],
        arc_coefficient: BTreeMap::new(),
        memory_vertices: vec![0, 1, 2],
        memory_arcs: BTreeSet::new(),
    };
    let mut engine = engine(&graph);
    assert!(engine.load_cuts(&graph, vec![cut.clone(), cut], vec![]));
    let fields = &engine.cuts.info.vertices[1].dominance;
    assert_eq!(fields.len(), 2);
    assert_eq!(
        (
            fields[0].word,
            fields[0].shift,
            fields[0].mask,
            fields[0].denominator,
            fields[0].cost
        ),
        (0, 0, 1, -2, -1.)
    );
    assert_eq!(
        (
            fields[1].word,
            fields[1].shift,
            fields[1].mask,
            fields[1].denominator,
            fields[1].cost
        ),
        (0, 7, 1, -2, -1.)
    );
    assert_eq!(engine.cuts.info.vertex_cut_potential[1], 0.);
    for (case, row) in include_str!("custom_covering_join_reference_capture.txt")
        .lines()
        .enumerate()
    {
        let fields: Vec<_> = row.split_whitespace().collect();
        let left_bits: u32 = fields[0].parse().unwrap();
        let right_bits: u32 = fields[1].parse().unwrap();
        let left_word = (left_bits & 1) | ((left_bits & 2) << 6);
        let right_word = (right_bits & 1) | ((right_bits & 2) << 6);
        let overlap = fields[2] == "1";
        let extra = fields[3].parse().unwrap();
        let accepted = fields[4] == "1";
        let expected: u64 = fields[5].parse().unwrap();
        let left = make(fstate.clone(), 0.5, 0.5, 1, left_word, extra);
        let right = make(
            bstate.clone(),
            -0.25,
            1.5,
            if overlap { 1 } else { 2 },
            right_word,
            -0.125,
        );
        let result = engine.join_cost(&graph, 1, &left, &right);
        assert_eq!(
            result.map(f64::to_bits),
            accepted.then_some(expected),
            "case={case}, row={row}"
        );
    }
}
