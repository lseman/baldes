//! Captured modern MetaSolver paired enumeration regressions.
use baldes::graph::coupled::CoupledResource;
use baldes::{Graph, Window};

#[cfg(feature = "clp")]
fn install_capture_front(graph: &mut Graph<2>, dependent_increment: f64) {
    use baldes::graph::coupled::descriptor::{
        FrontierBudget, FrontierOutcome, MachineRootInput, PreparedSlopeFunction, RawAffine,
    };
    use baldes::graph::coupled::{MachineArcFunctionInput, PairedResourceMode};
    let window = Window::new(0., 2.5).unwrap();
    let affine = |value| MachineRootInput::Affine(RawAffine::new(value, 0.).unwrap());
    let base: Vec<_> = graph
        .arcs
        .iter()
        .map(|arc| {
            PreparedSlopeFunction::from_prepared_pairs(vec![(arc.consumption[0], 0.), (0., 2.5)])
                .unwrap()
        })
        .collect();
    let inputs: Vec<_> = base
        .iter()
        .enumerate()
        .map(|(physical_arc, base)| MachineArcFunctionInput {
            physical_arc,
            tail_bounds: window,
            waiting_bounds: window,
            waiting_rate: 0,
            base,
            dependent: affine(dependent_increment),
            dependent_lower: affine(0.),
            dependent_upper: affine(8.),
        })
        .collect();
    graph
        .set_coupled_resources(&[CoupledResource::PairedFront {
            mode: PairedResourceMode::CostUnbounded,
        }])
        .unwrap();
    let outcome = graph.prepare_paired_front_program(
        0,
        &inputs,
        affine(0.),
        window,
        window,
        FrontierBudget::default(),
    );
    assert!(
        matches!(outcome, FrontierOutcome::Complete(())),
        "{outcome:?}"
    );
    graph
        .set_paired_waiting_windows_from_resource(0, 0)
        .unwrap();
}

#[cfg(feature = "clp")]
#[test]
fn paired_enumeration_five_customer_native_capture() {
    let captured: serde_json::Value = serde_json::from_str(include_str!(
        "../fixtures/coupled/metasolver-paired-enumeration.json"
    ))
    .unwrap();
    assert_eq!(
        captured["reference_sha256"],
        "103caf4a9c58d6141e935848294bff16677f400323466dc23cd4b9e5c15889d6"
    );
    let costs = captured["physical_arc_costs"].as_array().unwrap();
    let mut solver = baldes::pricing::Solver::<2>::new(7, 1).unwrap();
    solver.graph.vertices[0].elements.clear();
    solver.graph.vertices[6].elements.clear();
    for vertex in 0..7 {
        solver.graph.set_bounds(vertex, 0, 0., 2.5).unwrap();
    }
    for customer in 1..=5 {
        let mut edges = vec![(0, customer), (customer, 6)];
        edges.extend(
            (1..=5)
                .filter(|&other| other != customer)
                .map(|other| (customer, other)),
        );
        for (tail, head) in edges {
            let original_cost = if tail == 0 {
                1.
            } else if (tail, head) == (1, 2) || (tail, head) == (2, 1) {
                0.8
            } else {
                0.
            };
            let id = solver.graph.add_arc(tail, head, original_cost).unwrap();
            solver.graph.arc_mut(id).unwrap().reduced_cost = costs[id as usize].as_f64().unwrap();
            let increment = if head == 6 { 0. } else { 1. };
            solver.graph.arc_mut(id).unwrap().consumption[0] = increment;
        }
    }
    install_capture_front(&mut solver.graph, 0.);
    solver.bucket_steps[0] = 1.;
    let routes = solver
        .enumerate(captured["completion_cutoffs"][0].as_f64().unwrap())
        .unwrap();
    let mut ordered: Vec<Vec<usize>> = routes
        .iter()
        .map(|r| r.arcs.iter().map(|&a| a as usize).collect())
        .collect();
    ordered.sort();
    let mut expected_ordered: Vec<Vec<usize>> = captured["oracle_ordered_arcs"]
        .as_array()
        .unwrap()
        .iter()
        .map(|route| {
            route
                .as_array()
                .unwrap()
                .iter()
                .map(|arc| arc.as_u64().unwrap() as usize)
                .collect()
        })
        .collect();
    expected_ordered.sort();
    // The reference keeps one representative for each customer set. Each
    // retained orientation must still be a feasible ordered oracle route.
    assert_eq!(
        ordered.len(),
        captured["native_customer_sets"].as_array().unwrap().len()
    );
    assert!(ordered
        .iter()
        .all(|route| expected_ordered.binary_search(route).is_ok()));
    let mut sets: Vec<Vec<usize>> = routes
        .iter()
        .map(|r| {
            let mut s: Vec<_> = r
                .arcs
                .iter()
                .map(|&i| solver.graph.arcs[i as usize].head)
                .filter(|&v| v != 6)
                .collect();
            s.sort_unstable();
            s
        })
        .collect();
    sets.sort();
    sets.dedup();
    let expected: Vec<Vec<usize>> = captured["native_customer_sets"]
        .as_array()
        .unwrap()
        .iter()
        .map(|row| {
            row.as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap() as usize)
                .collect()
        })
        .collect();
    assert_eq!(sets, expected);
}

#[cfg(feature = "clp")]
#[test]
fn paired_enumeration_nonzero_capture_respects_native_cutoff() {
    let captured: serde_json::Value = serde_json::from_str(include_str!(
        "../fixtures/coupled/metasolver-paired-enumeration-increment.json"
    ))
    .unwrap();
    assert_eq!(
        captured["reference_sha256"],
        "103caf4a9c58d6141e935848294bff16677f400323466dc23cd4b9e5c15889d6"
    );
    let costs = captured["physical_arc_costs"].as_array().unwrap();
    let mut solver = baldes::pricing::Solver::<2>::new(7, 1).unwrap();
    solver.graph.vertices[0].elements.clear();
    solver.graph.vertices[6].elements.clear();
    for vertex in 0..7 {
        solver.graph.set_bounds(vertex, 0, 0., 2.5).unwrap();
    }
    for customer in 1..=5 {
        let mut edges = vec![(0, customer), (customer, 6)];
        edges.extend(
            (1..=5)
                .filter(|&other| other != customer)
                .map(|other| (customer, other)),
        );
        for (tail, head) in edges {
            let original_cost = if tail == 0 {
                1.
            } else if (tail, head) == (1, 2) || (tail, head) == (2, 1) {
                0.8
            } else {
                0.
            };
            let id = solver.graph.add_arc(tail, head, original_cost).unwrap();
            solver.graph.arc_mut(id).unwrap().reduced_cost = costs[id as usize].as_f64().unwrap();
            let increment = if head == 6 { 0. } else { 1. };
            solver.graph.arc_mut(id).unwrap().consumption[0] = increment;
        }
    }
    install_capture_front(&mut solver.graph, 0.25);
    solver.bucket_steps[0] = 1.;
    // The directional API must honor its caller's completion callback for
    // paired fronts.
    solver.complete_labels(true).unwrap();
    let callbacks = std::cell::Cell::new(0);
    let checked = baldes::enumeration::Enumeration::run_directional(
        &solver.graph,
        &solver.engine.as_ref().unwrap().grid,
        &baldes::enumeration::Controls {
            cutoff: captured["completion_cutoff"].as_f64().unwrap(),
            ..Default::default()
        },
        |_, _| {
            callbacks.set(callbacks.get() + 1);
            false
        },
    )
    .unwrap();
    assert!(callbacks.get() > 0);
    assert!(checked.pruning.completion_rejections > 0);
    let cutoff = captured["completion_cutoff"].as_f64().unwrap();
    let engine = solver.engine.as_ref().unwrap();
    let checked = baldes::enumeration::Enumeration::run_directional(
        &solver.graph,
        &engine.grid,
        &baldes::enumeration::Controls {
            cutoff,
            ..Default::default()
        },
        |forward, label| engine.enumeration_completion(&solver.graph, forward, label, cutoff, true),
    )
    .unwrap();
    assert!(checked.pruning.completion_rejections > 0);
    let fixing = solver.fix(cutoff).unwrap();
    assert!(fixing.checked > 0);
    let routes = solver
        .enumerate(captured["completion_cutoff"].as_f64().unwrap())
        .unwrap();
    assert_eq!(
        routes.len(),
        captured["native_paths"].as_array().unwrap().len()
    );
    for native in captured["native_paths"].as_array().unwrap() {
        let native_arcs: Vec<i32> = native["arcs"]
            .as_array()
            .unwrap()
            .iter()
            .map(|arc| arc.as_i64().unwrap() as i32)
            .collect();
        let native_cost = native_arcs
            .iter()
            .map(|&arc| costs[arc as usize].as_f64().unwrap())
            .sum::<f64>()
            + native["paired_scalar"].as_f64().unwrap();
        let route = routes
            .iter()
            .find(|route| route.arcs == native_arcs)
            .expect("native ordering must be present");
        assert!(
            (route.cost - native_cost).abs() < 1e-12,
            "arcs {native_arcs:?}, actual {}, native {native_cost}",
            route.cost
        );
    }
    let mut sets: Vec<Vec<usize>> = routes
        .iter()
        .map(|r| {
            let mut s: Vec<_> = r
                .arcs
                .iter()
                .map(|&i| solver.graph.arcs[i as usize].head)
                .filter(|&v| v != 6)
                .collect();
            s.sort_unstable();
            s
        })
        .collect();
    sets.sort();
    sets.dedup();
    let expected: Vec<Vec<usize>> = captured["native_customer_sets"]
        .as_array()
        .unwrap()
        .iter()
        .map(|row| {
            row.as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap() as usize)
                .collect()
        })
        .collect();
    // The reference and the front-program storage retain one representative
    // per customer set for this capture.
    assert_eq!(sets, expected);
    // A controlled native replay lowers the enumeration cutoff to zero;
    // all ten completion calls reject and no path survives.
    assert!(solver.enumerate(0.).unwrap().is_empty());
    // The registered native model also reports incomplete enumeration when
    // its concatenation ratio is zero. Reuse the captured graph and cutoff,
    // then verify a later ordinary enumeration can start from clean state.
    solver.restore_arcs();
    solver
        .options
        .set("BALDESapplyReducedCostFixing", 1.)
        .unwrap();
    solver
        .options
        .set("BALDESstopRatioForConcatenationInEnum", 0.)
        .unwrap();
    let interrupted = solver.run_fixing_enumeration(0, 0., cutoff).unwrap();
    assert_eq!((interrupted.state, interrupted.routes), (1, 0));
    assert!(solver.enumeration.is_none());
    // Native keeps its bucket arcs across this interrupted call.
    assert!(solver.engine.is_some());
    let retried = solver.enumerate(cutoff).unwrap();
    for native in captured["native_paths"].as_array().unwrap() {
        let arcs: Vec<i32> = native["arcs"]
            .as_array()
            .unwrap()
            .iter()
            .map(|arc| arc.as_i64().unwrap() as i32)
            .collect();
        assert!(retried.iter().any(|route| route.arcs == arcs));
    }
}

fn install_paired_front(
    graph: &mut Graph<2>,
    upper: f64,
    ceiling: f64,
    transitions: &[(f64, f64, i32)],
) {
    use baldes::graph::coupled::descriptor::{
        FrontierBudget, FrontierOutcome, MachineRootInput, PreparedSlopeFunction, RawAffine,
    };
    use baldes::graph::coupled::{MachineArcFunctionInput, PairedResourceMode};
    let bounds = Window::new(0., upper).unwrap();
    let affine = |value, slope| MachineRootInput::Affine(RawAffine::new(value, slope).unwrap());
    let base = PreparedSlopeFunction::from_prepared_pairs(vec![(1., 0.), (0., upper)]).unwrap();
    let inputs: Vec<_> = transitions
        .iter()
        .enumerate()
        .map(
            |(physical_arc, &(value, slope, waiting_rate))| MachineArcFunctionInput {
                physical_arc,
                tail_bounds: bounds,
                waiting_bounds: bounds,
                waiting_rate,
                base: &base,
                dependent: affine(value, slope),
                dependent_lower: affine(0., 0.),
                dependent_upper: affine(ceiling, 0.),
            },
        )
        .collect();
    graph
        .set_coupled_resources(&[CoupledResource::PairedFront {
            mode: PairedResourceMode::CostUnbounded,
        }])
        .unwrap();
    let result = graph.prepare_paired_front_program(
        0,
        &inputs,
        affine(0., 0.),
        bounds,
        bounds,
        FrontierBudget::default(),
    );
    assert!(
        matches!(result, FrontierOutcome::Complete(())),
        "{result:?}"
    );
    graph
        .set_paired_waiting_windows_from_resource(0, 0)
        .unwrap();
}

#[test]
fn paired_concatenation_timeout_retains_fixed_grid_until_explicit_restoration() {
    let mut solver = baldes::pricing::Solver::<2>::new(3, 1).unwrap();
    solver.graph.vertices[0].elements.clear();
    solver.graph.vertices[2].elements.clear();
    for vertex in 0..3 {
        solver.graph.set_bounds(vertex, 0, 0., 4.).unwrap();
    }
    for (tail, head) in [(0, 1), (1, 2)] {
        let arc = solver.graph.add_arc(tail, head, -1.).unwrap();
        solver.graph.arc_mut(arc).unwrap().consumption[0] = 1.;
    }
    install_paired_front(&mut solver.graph, 4., 8., &[(0.25, 0., 0); 2]);
    let mut baseline = solver.fork_for_node();
    let before = baseline.price_phase(0, 10, false).unwrap();
    assert_eq!(before.minimum_reduced_cost, -1.5, "baseline: {before:?}");
    solver
        .options
        .set("BALDESmaxNumOfLabelsInHeurEnumeration", 10_000.)
        .unwrap();
    solver
        .options
        .set("BALDESapplyReducedCostFixing", 1.)
        .unwrap();
    solver
        .options
        .set("BALDESstopRatioForConcatenationInEnum", 0.)
        .unwrap();
    let report = solver.run_fixing_enumeration(1, 0., 0.).unwrap();
    assert_eq!((report.state, report.routes), (1, 0));
    assert!(report.eliminated > 0);
    assert!(solver.engine.is_some());
    assert!(solver.enumeration.is_none());
    // Native retains the fixed grid after timeout. Its literal prejoin key
    // gate removes this chain's forward terminal arc; no forward label crosses
    // halfway, so subsequent pricing cannot recover it at another split.
    // See internal/migrations/paired-fixing-timeout-repricing.md for the
    // controlled repricing capture on a real registered native owner.
    let retained = solver.price_phase(0, 10, false).unwrap();
    assert!(retained.columns.is_empty());
    assert_eq!(retained.minimum_reduced_cost, 1e12);
    assert!(solver.enumeration.is_none());
    solver.restore_arcs();
    assert_eq!(
        solver
            .price_phase(0, 10, false)
            .unwrap()
            .minimum_reduced_cost,
        -1.5
    );
}

#[test]
fn paired_concatenation_timeout_allows_repricing_through_a_retained_crossing() {
    let mut solver = baldes::pricing::Solver::<2>::new(5, 1).unwrap();
    solver.graph.vertices[0].elements.clear();
    solver.graph.vertices[4].elements.clear();
    for vertex in 0..5 {
        solver.graph.set_bounds(vertex, 0, 0., 4.).unwrap();
    }
    for tail in 0..4 {
        let arc = solver.graph.add_arc(tail, tail + 1, -1.).unwrap();
        solver.graph.arc_mut(arc).unwrap().consumption[0] = 1.;
    }
    install_paired_front(&mut solver.graph, 4., 8., &[(0.25, 0., 0); 4]);
    let before = solver.fork_for_node().price_phase(0, 10, false).unwrap();
    assert_eq!(before.minimum_reduced_cost, -3.);
    solver
        .options
        .set("BALDESmaxNumOfLabelsInHeurEnumeration", 10_000.)
        .unwrap();
    solver
        .options
        .set("BALDESapplyReducedCostFixing", 1.)
        .unwrap();
    solver
        .options
        .set("BALDESstopRatioForConcatenationInEnum", 0.)
        .unwrap();
    let report = solver.run_fixing_enumeration(1, 0., 0.).unwrap();
    assert_eq!((report.state, report.routes), (1, 0));
    assert!(report.eliminated > 0);
    assert!(solver.engine.is_some());
    assert!(solver.enumeration.is_none());
    let repriced = solver.price_phase(0, 10, false).unwrap();
    assert!(repriced.exact);
    assert_eq!(repriced.minimum_reduced_cost, -3.);
    assert_eq!(repriced.columns[0].arcs, vec![0, 1, 2, 3]);
    assert!(repriced.statistics.concatenation_tests > 0);
    assert!(solver.enumeration.is_none());
}

#[test]
fn paired_signed_completion_and_fixing_apply_the_strict_local_split_cutoff() {
    use baldes::enumeration::{Controls, Enumeration};
    let seed = 0;
    let mut solver = baldes::pricing::Solver::<2>::new(5, 1).unwrap();
    solver.graph.vertices[0].elements.clear();
    solver.graph.vertices[4].elements.clear();
    for v in 0..5 {
        solver.graph.set_bounds(v, 0, 0., 8.).unwrap();
    }
    let mut arcs = Vec::new();
    for tail in 0..4 {
        for head in (tail + 1)..5 {
            let id = solver
                .graph
                .add_arc(
                    tail,
                    head,
                    ((seed * 13 + tail * 7 + head * 3) % 13) as f64 / 4. - 1.5,
                )
                .unwrap();
            solver.graph.arc_mut(id).unwrap().consumption[0] = 1.;
            arcs.push((
                ((seed + tail * 3 + head) % 5) as f64 / 4. - 0.5,
                ((seed + tail + head) % 3) as f64 / 8. - 0.125,
                (seed % 2) as i32,
            ));
        }
    }
    install_paired_front(&mut solver.graph, 8., 10., &arcs);
    solver.complete_labels(true).unwrap();
    let engine = solver.engine.as_ref().unwrap();
    let cutoff = -1.;
    let controls = Controls {
        cutoff,
        preserve_arc_paths: true,
        ..Default::default()
    };
    let collect = |universe: &Enumeration<2>| {
        let mut paths: Vec<_> = universe
            .paths
            .partitions
            .iter()
            .flatten()
            .map(|p| p.arcs.clone())
            .collect();
        paths.sort();
        paths
    };
    let reference =
        Enumeration::run_directional(&solver.graph, &engine.grid, &controls, |_, _| true).unwrap();
    assert_eq!(
        collect(&reference),
        vec![vec![0, 4, 8], vec![0, 5, 9], vec![1, 8]]
    );
    // The native resource capture below checks [0,5,9] at every split:
    // its total costs are [-1.5,-1.25,-1,-1]. Local terminal completion
    // and fixing therefore reject it at cutoff -1, although another split
    // admits the same physical path. Cross-split preservation is not asserted.
    let candidate = Enumeration::run_directional(&solver.graph, &engine.grid, &controls, |d, l| {
        engine.enumeration_completion(&solver.graph, d, l, cutoff, true)
    })
    .unwrap();
    let expected = vec![vec![0, 4, 8], vec![1, 8]];
    assert_eq!(collect(&candidate), expected);
    assert!(candidate.pruning.completion_rejections > 0);
    let mut fixed_solver = solver.fork_for_node();
    let fixing = fixed_solver.fix(cutoff).unwrap();
    assert!(fixing.eliminated_forward + fixing.eliminated_backward > 0);
    let fixed = Enumeration::run_directional(
        &fixed_solver.graph,
        &fixed_solver.engine.as_ref().unwrap().grid,
        &controls,
        |_, _| true,
    )
    .unwrap();
    // Fixing also tests the backward [8,4] suffix against the forward [0]
    // prefix. With these raw, unpreprocessed bounds, native extension retains
    // a zero-width backward front at base zero; its join with the prefix at
    // base one fails. Completion can accept another split of [0,4,8], so its
    // result is not an invariant for the locally fixed bucket domain.
    // The exact affine split capture below pins this distinction.
    assert_eq!(collect(&fixed), vec![vec![1, 8]]);
}

#[test]
fn paired_signed_affine_local_splits_match_native_degenerate_front_capture() {
    use baldes::graph::coupled::descriptor::{
        FrontierBudget, FrontierOutcome, PairedFrontOperations, PreparedSlopeFunction,
    };
    let captured: serde_json::Value = serde_json::from_str(include_str!(
        "../fixtures/coupled/metasolver-signed-affine-fixing-splits.json"
    ))
    .unwrap();
    assert_eq!(
        captured["reference_sha256"],
        "103caf4a9c58d6141e935848294bff16677f400323466dc23cd4b9e5c15889d6"
    );
    let mut graph = Graph::<2>::new(4, 1).unwrap();
    for vertex in 0..4 {
        graph.set_bounds(vertex, 0, 0., 8.).unwrap();
    }
    for vertex in 0..3 {
        let arc = graph.add_arc(vertex, vertex + 1, 0.).unwrap();
        graph.arc_mut(arc).unwrap().consumption[0] = 1.;
    }
    let transitions: Vec<_> = captured["case"]["arcs"]
        .as_array()
        .unwrap()
        .iter()
        .map(|arc| {
            (
                arc["increment"][0][0].as_f64().unwrap(),
                arc["increment"][1][0].as_f64().unwrap(),
                arc["waiting"].as_i64().unwrap() as i32,
            )
        })
        .collect();
    install_paired_front(&mut graph, 8., 10., &transitions);
    let kernel = graph.coupled_kernel().unwrap();
    for entry in captured["splits"].as_array().unwrap() {
        let split = entry["split"].as_u64().unwrap() as usize;
        let extend = |forward, arcs: Vec<usize>| {
            let mut state = Some(kernel.root(forward));
            let mut cost = 0.;
            for arc in arcs {
                state = state.and_then(|previous| {
                    kernel
                        .extend(&previous, arc, forward)
                        .unwrap()
                        .map(|(next, delta)| {
                            cost += delta;
                            next
                        })
                });
            }
            (state, cost)
        };
        let (forward, forward_cost) = extend(true, (0..split).collect());
        let (backward, backward_cost) = extend(false, (split..3).rev().collect());
        assert_eq!(forward.is_some(), entry["forward_ok"].as_bool().unwrap());
        assert_eq!(backward.is_some(), entry["backward_ok"].as_bool().unwrap());
        let actual = forward.zip(backward).and_then(|(forward, backward)| {
            kernel
                .join(&forward, &backward)
                .unwrap()
                .map(|cost| forward_cost + backward_cost + cost)
        });
        assert_eq!(actual, entry["dependent"].as_f64(), "split={split}");
        if entry["forward_ok"] == true && entry["backward_ok"] == true {
            let front = |side: &str| {
                PreparedSlopeFunction::from_machine_pairs(
                    entry[side]["pairs"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(|pair| (pair[0].as_f64().unwrap(), pair[1].as_f64().unwrap()))
                        .collect(),
                )
                .unwrap()
            };
            let joined = PairedFrontOperations::machine_concatenation(
                &front("forward"),
                &front("backward"),
                10.,
                5,
                FrontierBudget::default(),
            );
            match joined {
                FrontierOutcome::Complete(selection) => assert_eq!(selection.dependent, actual),
                FrontierOutcome::Empty => assert!(actual.is_none()),
                FrontierOutcome::Unknown(reason) => panic!("split={split}: {reason:?}"),
            }
        }
    }
}

#[test]
fn paired_fixing_preserves_enumeration_with_presence_and_branching() {
    use baldes::cuts::soft_conflicts::{Mode, PresenceRow, PricingInput};
    use baldes::pricing::restrictions::RyanFosterConstraint;

    let mut eliminated = 0;
    for together in [false, true] {
        for presence in [false, true] {
            for cutoff in [-1., 0., 1., 2.] {
                let mut solver = baldes::pricing::Solver::<2>::new(5, 1).unwrap();
                solver.graph.vertices[0].elements.clear();
                solver.graph.vertices[4].elements.clear();
                for v in 0..5 {
                    solver.graph.set_bounds(v, 0, 0., 8.).unwrap();
                }
                let mut arcs = Vec::new();
                let mut events = Vec::new();
                for tail in 0..4 {
                    for head in tail + 1..5 {
                        let id = solver
                            .graph
                            .add_arc(tail, head, ((tail * 7 + head * 3) % 9) as f64 / 4. - 1.)
                            .unwrap();
                        solver.graph.arc_mut(id).unwrap().consumption[0] = 1.;
                        arcs.push((0.25, 0., 0));
                        events.push(if head == 1 {
                            vec![0]
                        } else if head == 3 {
                            vec![1]
                        } else {
                            vec![]
                        });
                    }
                }
                install_paired_front(&mut solver.graph, 8., 10., &arcs);
                solver
                    .set_ryan_foster(vec![RyanFosterConstraint {
                        first: 1,
                        second: 3,
                        together,
                    }])
                    .unwrap();
                if presence {
                    solver
                        .set_presence_rows(vec![PricingInput {
                            row: PresenceRow::new(2, vec![(0, 1)], Mode::AnyPair).unwrap(),
                            arc_events: events,
                            dual: 0.5,
                        }])
                        .unwrap();
                }
                solver.complete_labels(true).unwrap();
                let mut baseline = solver.fork_for_node();
                let mut fixed = solver.fork_for_node();
                let expected = baseline.enumerate(cutoff).unwrap();
                let report = fixed.fix(cutoff).unwrap();
                eliminated += report.eliminated_forward + report.eliminated_backward;
                let actual = fixed.enumerate(cutoff).unwrap();
                let paths = |routes: Vec<baldes::Route>| {
                    let mut paths: Vec<_> = routes.into_iter().map(|route| route.arcs).collect();
                    paths.sort();
                    paths
                };
                assert_eq!(
                    paths(actual),
                    paths(expected),
                    "together={together}, presence={presence}, cutoff={cutoff}, report={report:?}"
                );
            }
        }
    }
    assert!(eliminated > 0);
}

#[test]
fn paired_signed_path_join_values_match_native_at_every_split() {
    let fixture: serde_json::Value = serde_json::from_str(include_str!(
        "../fixtures/coupled/metasolver-signed-fixing-splits.json"
    ))
    .unwrap();
    assert_eq!(
        fixture["reference_sha256"],
        "103caf4a9c58d6141e935848294bff16677f400323466dc23cd4b9e5c15889d6"
    );
    let mut graph = Graph::<2>::new(4, 1).unwrap();
    for vertex in 0..4 {
        graph.set_bounds(vertex, 0, 0., 8.).unwrap();
    }
    for vertex in 0..3 {
        let id = graph.add_arc(vertex, vertex + 1, 0.).unwrap();
        graph.arc_mut(id).unwrap().consumption[0] = 1.;
    }
    install_paired_front(
        &mut graph,
        8.,
        10.,
        &[(-0.25, 0., 0), (-0.25, 0., 0), (0.25, 0., 0)],
    );
    let kernel = graph.coupled_kernel().unwrap();
    for (split, expected) in fixture["split_dependent_values"]
        .as_array()
        .unwrap()
        .iter()
        .enumerate()
    {
        let mut forward = kernel.root(true);
        let mut backward = kernel.root(false);
        let mut cost = 0.;
        for arc in 0..split {
            let (state, delta) = kernel.extend(&forward, arc, true).unwrap().unwrap();
            forward = state;
            cost += delta;
        }
        for arc in (split..3).rev() {
            let (state, delta) = kernel.extend(&backward, arc, false).unwrap().unwrap();
            backward = state;
            cost += delta;
        }
        cost += kernel.join(&forward, &backward).unwrap().unwrap();
        assert_eq!(cost, expected.as_f64().unwrap(), "split={split}");
    }
}

#[test]
fn registered_signed_owner_fixing_joins_match_native_label_operands() {
    use baldes::graph::coupled::descriptor::{
        FrontierBudget, FrontierOutcome, PairedFrontOperations, PreparedSlopeFunction,
    };
    let captured: serde_json::Value = serde_json::from_str(include_str!(
        "../fixtures/coupled/metasolver-signed-owner-fixing.json"
    ))
    .unwrap();
    assert_eq!(
        captured["reference_sha256"],
        "103caf4a9c58d6141e935848294bff16677f400323466dc23cd4b9e5c15889d6"
    );
    let joins = captured["resource_joins"].as_array().unwrap();
    assert_eq!(joins.len(), 24);
    for join in joins {
        let front = |side: &str| {
            PreparedSlopeFunction::from_machine_pairs(
                join[side]["front"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|pair| (pair[0].as_f64().unwrap(), pair[1].as_f64().unwrap()))
                    .collect(),
            )
            .unwrap()
        };
        let left = front("left");
        let right = front("right");
        let result = PairedFrontOperations::machine_concatenation(
            &left,
            &right,
            join["ceiling"].as_f64().unwrap(),
            join["mode"].as_i64().unwrap() as i32,
            FrontierBudget::default(),
        );
        let FrontierOutcome::Complete(selection) = result else {
            panic!("native owner fixing resource join was available: {join}, {result:?}");
        };
        assert!(join["returned"].as_bool().unwrap());
        assert_eq!(
            selection.dependent.unwrap(),
            join["dependent_cost"].as_f64().unwrap()
        );
        // The fixing callback adds its dependent accumulator before the two
        // scalar label costs, then tests strict cutoff and NG compatibility.
        let total = ((selection.accumulate(0.) + 0.)
            + join["left"]["scalar_cost"].as_f64().unwrap())
            + join["right"]["scalar_cost"].as_f64().unwrap();
        assert_eq!(total, join["total"].as_f64().unwrap());
        let mask = |side: &str| {
            u64::from_str_radix(
                join[side]["mask"]
                    .as_str()
                    .unwrap()
                    .trim_start_matches("0x"),
                16,
            )
            .unwrap()
        };
        assert_eq!(
            mask("left") & mask("right") == 0 && total < captured["cutoff"].as_f64().unwrap(),
            join["accepted"].as_bool().unwrap()
        );
    }
}

#[test]
fn registered_signed_owner_fixing_and_enumeration_match_native() {
    use baldes::graph::coupled::descriptor::{
        FrontierBudget, FrontierOutcome, MachineRootInput, PreparedSlopeFunction, RawAffine,
    };
    use baldes::graph::coupled::{MachineArcFunctionInput, PairedResourceMode};
    let captured: serde_json::Value = serde_json::from_str(include_str!(
        "../fixtures/coupled/metasolver-signed-owner-fixing.json"
    ))
    .unwrap();
    let prepared = &captured["preparation"];
    let lower = prepared["vertex_lower"].as_array().unwrap();
    let upper = prepared["vertex_upper"].as_array().unwrap();
    let windows: Vec<_> = lower
        .iter()
        .zip(upper)
        .map(|(lo, hi)| Window::new(lo.as_f64().unwrap(), hi.as_f64().unwrap()).unwrap())
        .collect();
    let mut solver = baldes::pricing::Solver::<2>::new(windows.len(), 1).unwrap();
    let sink = windows.len() - 1;
    solver.graph.vertices[0].elements.clear();
    solver.graph.vertices[sink].elements.clear();
    for (vertex, bounds) in windows.iter().enumerate() {
        solver
            .graph
            .set_bounds(vertex, 0, bounds.lower, bounds.upper)
            .unwrap();
    }
    let arcs = prepared["arcs"].as_array().unwrap();
    let base: Vec<_> = arcs
        .iter()
        .map(|arc| {
            PreparedSlopeFunction::from_prepared_pairs(
                arc["functions"]["0x30"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|pair| (pair[0].as_f64().unwrap(), pair[1].as_f64().unwrap()))
                    .collect(),
            )
            .unwrap()
        })
        .collect();
    for (index, arc) in arcs.iter().enumerate() {
        assert_eq!(arc["id"].as_u64().unwrap() as usize, index);
        let tail = arc["tail"].as_u64().unwrap() as usize;
        let head = arc["head"].as_u64().unwrap() as usize;
        let id = solver
            .graph
            .add_arc(tail, head, if tail == 0 { 1. } else { 0. })
            .unwrap();
        let edge = solver.graph.arc_mut(id).unwrap();
        edge.consumption[0] = base[index].pairs()[0].0;
        edge.reduced_cost = captured["physical_arc_reduced_costs"][index]
            .as_f64()
            .unwrap();
    }
    solver
        .graph
        .set_coupled_resources(&[CoupledResource::PairedFront {
            mode: PairedResourceMode::CostUnbounded,
        }])
        .unwrap();
    let affine = |value| MachineRootInput::Affine(RawAffine::new(value, 0.).unwrap());
    let inputs: Vec<_> = arcs
        .iter()
        .enumerate()
        .map(|(physical_arc, arc)| MachineArcFunctionInput {
            physical_arc,
            tail_bounds: windows[arc["tail"].as_u64().unwrap() as usize],
            waiting_bounds: windows[arc["head"].as_u64().unwrap() as usize],
            waiting_rate: 0,
            base: &base[physical_arc],
            dependent: affine(-0.25),
            dependent_lower: affine(0.),
            dependent_upper: affine(8.),
        })
        .collect();
    let installed = solver.graph.prepare_paired_front_program(
        0,
        &inputs,
        affine(2.),
        windows[0],
        windows[sink],
        FrontierBudget::default(),
    );
    assert!(
        matches!(installed, FrontierOutcome::Complete(())),
        "{installed:?}"
    );
    solver
        .graph
        .set_paired_waiting_windows_from_resource(0, 0)
        .unwrap();
    solver.bucket_steps[0] = 1.;
    solver.complete_labels(true).unwrap();
    let cutoff = captured["cutoff"].as_f64().unwrap();
    let bucket_arcs = |solver: &baldes::pricing::Solver<2>| {
        let grid = &solver.engine.as_ref().unwrap().grid;
        let mut arcs = std::collections::BTreeSet::new();
        for forward in [true, false] {
            for row in &grid.direction(forward).buckets {
                for bucket in row {
                    for arc in bucket.arcs.iter().filter(|arc| arc.jump.is_none()) {
                        arcs.insert((
                            forward,
                            grid.order[arc.from.vertex],
                            arc.from.bucket,
                            arc.physical,
                        ));
                    }
                }
            }
        }
        arcs
    };
    let before = bucket_arcs(&solver);
    let fixing = solver.fix(cutoff).unwrap();
    assert_eq!(
        fixing.checked,
        captured["fixing"]["checked"].as_u64().unwrap() as usize
    );
    assert_eq!(
        fixing.eliminated_forward,
        captured["fixing"]["eliminated_forward"].as_u64().unwrap() as usize
    );
    assert_eq!(
        fixing.eliminated_backward,
        captured["fixing"]["eliminated_backward"].as_u64().unwrap() as usize
    );
    let after = bucket_arcs(&solver);
    let decisions = captured["fixing"]["decisions"].as_array().unwrap();
    assert_eq!(before.len(), decisions.len());
    for decision in decisions {
        let key = (
            decision["forward"].as_bool().unwrap(),
            decision["source"][0].as_u64().unwrap() as usize,
            decision["source"][1].as_u64().unwrap() as usize,
            decision["physical_arc_id"].as_u64().unwrap() as usize,
        );
        assert!(
            before.contains(&key),
            "native checked bucket arc absent before fixing: {key:?}"
        );
        assert_eq!(
            after.contains(&key),
            !decision["eliminated"].as_bool().unwrap(),
            "native bucket arc decision: {key:?}"
        );
    }
    let routes = solver.enumerate(cutoff).unwrap();
    assert_eq!(
        routes.len(),
        captured["enumerated_paths"].as_array().unwrap().len()
    );
    let cover = |path: &[usize]| {
        let mut cover: Vec<_> = path
            .iter()
            .map(|&arc| arcs[arc]["head"].as_u64().unwrap() as usize)
            .filter(|&head| head != sink)
            .collect();
        cover.sort_unstable();
        cover
    };
    let mut expected: Vec<_> = captured["enumerated_paths"]
        .as_array()
        .unwrap()
        .iter()
        .map(|path| {
            let physical: Vec<_> = path["arcs"]
                .as_array()
                .unwrap()
                .iter()
                .map(|a| a.as_u64().unwrap() as usize)
                .collect();
            (cover(&physical), path["paired"].as_f64().unwrap())
        })
        .collect();
    let mut actual: Vec<_> = routes
        .iter()
        .map(|route| {
            let path: Vec<_> = route.arcs.iter().map(|&arc| arc as usize).collect();
            assert_eq!(route.original_cost, 2.25);
            (cover(&path), route.coupled_values[0].1)
        })
        .collect();
    expected.sort_by(|a, b| a.0.cmp(&b.0));
    actual.sort_by(|a, b| a.0.cmp(&b.0));
    assert_eq!(actual, expected);
}

#[cfg(feature = "clp")]
#[test]
fn registered_paired_standard_composite_matches_native_owner_objectives() {
    use baldes::master::model::{self, format};
    let fixture: serde_json::Value = serde_json::from_str(include_str!(
        "../fixtures/coupled/metasolver-paired-standard-dispatch.json"
    ))
    .unwrap();
    assert_eq!(
        fixture["reference_sha256"],
        "103caf4a9c58d6141e935848294bff16677f400323466dc23cd4b9e5c15889d6"
    );
    assert!(fixture["factory"]["returned_nonnull"].as_bool().unwrap());
    assert_eq!(fixture["resource_joins"].as_array().unwrap().len(), 9);
    for case in fixture["cases"].as_array().unwrap() {
        let problem = format::Model::from_json(&case["model"].to_string())
            .unwrap()
            .problem::<2>()
            .unwrap();
        let mut options = model::Options::default();
        options.cg.stabilize = false;
        let output = model::solve(problem, options).unwrap();
        assert_eq!(output.master.status, baldes::master::cg::Status::Optimal);
        assert_eq!(
            output.master.lp,
            case["native_root_lp"].as_f64().unwrap(),
            "{}",
            case["name"]
        );
    }
}

#[cfg(feature = "clp")]
#[path = "paired_standard_matrix.rs"]
mod paired_standard_matrix;

#[path = "paired_signed_affine_capture.rs"]
mod paired_signed_affine_capture;
