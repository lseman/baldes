//! Captured modern MetaSolver paired enumeration regressions.
use baldes::graph::coupled::CoupledResource;
use baldes::{Graph, Window};

#[cfg(feature = "clp")]
fn install_capture_front(graph: &mut Graph<2>, dependent_increment: f64) {
    install_capture_front_with_preprocessing(graph, dependent_increment, false);
}

#[cfg(feature = "clp")]
fn install_capture_front_with_preprocessing(
    graph: &mut Graph<2>,
    dependent_increment: f64,
    preprocess: bool,
) {
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
    if preprocess {
        graph
            .set_paired_waiting_windows_from_resource(0, 0)
            .unwrap();
        let domains =
            match graph.preprocess_paired_base_domains(0, &inputs, FrontierBudget::default()) {
                FrontierOutcome::Complete(value) => value,
                other => panic!("{other:?}"),
            };
        assert!(matches!(
            domains.install_front_program(graph, &inputs, affine(0.), FrontierBudget::default()),
            FrontierOutcome::Complete(())
        ));
        return;
    }
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
fn preprocessed_paired_bucket_components_match_native_model() {
    use baldes::graph::buckets::BucketGraph;
    use std::collections::BTreeMap;

    let capture: serde_json::Value = serde_json::from_str(include_str!(
        "../fixtures/coupled/bucket_tarjan_paired_model.json"
    ))
    .unwrap();
    let mut graph = Graph::<2>::new(11, 1).unwrap();
    for vertex in 0..11 {
        graph.set_bounds(vertex, 0, 0., 2.5).unwrap();
    }
    for arc in capture["arcs"].as_array().unwrap() {
        let tail = arc[0].as_u64().unwrap() as usize;
        let head = arc[1].as_u64().unwrap() as usize;
        let id = graph.add_arc(tail, head, 0.).unwrap();
        graph.arc_mut(id).unwrap().consumption[0] = arc[2].as_f64().unwrap();
    }
    install_capture_front_with_preprocessing(&mut graph, 1., true);
    let cases = std::iter::once((1., &capture["components"], &capture["buckets"])).chain(
        capture["variants"]
            .as_array()
            .unwrap()
            .iter()
            .map(|variant| {
                (
                    variant["step"].as_f64().unwrap(),
                    &variant["components"],
                    &variant["buckets"],
                )
            }),
    );
    for (step, components, buckets) in cases {
        let grid = BucketGraph::prepare(&graph, [step, 0.], 100, true).unwrap();
        for (index, direction) in [&grid.forward, &grid.backward].iter().enumerate() {
            let actual: Vec<_> = direction
                .buckets
                .iter()
                .flatten()
                .map(|bucket| bucket.component.map(|value| value as i32).unwrap_or(-1))
                .collect();
            let expected: Vec<_> = components[index]
                .as_array()
                .unwrap()
                .iter()
                .map(|value| value.as_i64().unwrap() as i32)
                .collect();
            assert_eq!(actual, expected, "step {step}, direction {index}");
            let actual_arcs: BTreeMap<_, Vec<_>> = direction
                .buckets
                .iter()
                .flatten()
                .filter(|bucket| bucket.reachable)
                .map(|bucket| {
                    let arcs = bucket
                        .arcs
                        .iter()
                        .map(|arc| {
                            (
                                graph.arcs[arc.physical].id,
                                arc.jump.map(|id| id as i32).unwrap_or(-1),
                            )
                        })
                        .collect();
                    ((bucket.id.vertex, bucket.id.bucket), arcs)
                })
                .collect();
            let name = if index == 0 { "forward" } else { "backward" };
            let expected_arcs: BTreeMap<_, Vec<_>> = buckets[name]
                .as_array()
                .unwrap()
                .iter()
                .map(|bucket| {
                    let key = (
                        bucket["vertex"].as_u64().unwrap() as usize,
                        bucket["bucket"].as_u64().unwrap() as usize,
                    );
                    let arcs = bucket["arcs"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(|arc| {
                            (
                                arc[0].as_i64().unwrap() as i32,
                                arc[1].as_i64().unwrap() as i32,
                            )
                        })
                        .collect();
                    (key, arcs)
                })
                .collect();
            assert_eq!(actual_arcs, expected_arcs, "step {step}, direction {index}");
        }
    }
}

#[cfg(feature = "clp")]
#[test]
fn paired_jump_regeneration_matches_native_fixed_bucket_arcs() {
    use baldes::graph::buckets::BucketGraph;
    use std::collections::BTreeMap;

    let capture: serde_json::Value = serde_json::from_str(include_str!(
        "../fixtures/coupled/bucket_paired_regeneration_native.json"
    ))
    .unwrap();
    let mut graph = Graph::<2>::new(7, 1).unwrap();
    for vertex in 0..7 {
        graph.set_bounds(vertex, 0, 0., 2.5).unwrap();
    }
    for arc in capture["arcs"].as_array().unwrap() {
        let id = graph
            .add_arc(
                arc[0].as_u64().unwrap() as usize,
                arc[1].as_u64().unwrap() as usize,
                0.,
            )
            .unwrap();
        graph.arc_mut(id).unwrap().consumption[0] = arc[2].as_f64().unwrap();
    }
    install_capture_front_with_preprocessing(&mut graph, 0., true);
    let decode = |buckets: &serde_json::Value| -> BTreeMap<(usize, usize), Vec<(i32, i32)>> {
        buckets
            .as_array()
            .unwrap()
            .iter()
            .map(|bucket| {
                let key = (
                    bucket["vertex"].as_u64().unwrap() as usize,
                    bucket["bucket"].as_u64().unwrap() as usize,
                );
                let arcs = bucket["arcs"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|arc| {
                        (
                            arc[0].as_i64().unwrap() as i32,
                            arc[1].as_i64().unwrap() as i32,
                        )
                    })
                    .collect();
                (key, arcs)
            })
            .collect()
    };
    for case in capture["cases"].as_array().unwrap() {
        let step = case["step"].as_f64().unwrap();
        for captured in case["captures"].as_array().unwrap() {
            let forward = captured["forward"].as_bool().unwrap();
            let mut grid = BucketGraph::prepare(&graph, [step, 0.], 100, true).unwrap();
            let before = decode(&captured["before"]);
            let direction = if forward {
                &mut grid.forward
            } else {
                &mut grid.backward
            };
            for bucket in direction
                .buckets
                .iter_mut()
                .flatten()
                .filter(|b| b.reachable)
            {
                let expected = &before[&(bucket.id.vertex, bucket.id.bucket)];
                bucket
                    .arcs
                    .retain(|arc| expected.contains(&(graph.arcs[arc.physical].id, -1)));
                let actual: Vec<_> = bucket
                    .arcs
                    .iter()
                    .map(|arc| (graph.arcs[arc.physical].id, -1))
                    .collect();
                assert_eq!(&actual, expected, "before step {step}, forward {forward}");
            }
            grid.regenerate_jumps(&graph, forward);
            let direction = if forward {
                &grid.forward
            } else {
                &grid.backward
            };
            let actual: BTreeMap<_, Vec<_>> = direction
                .buckets
                .iter()
                .flatten()
                .filter(|b| b.reachable)
                .map(|bucket| {
                    let arcs = bucket
                        .arcs
                        .iter()
                        .map(|arc| {
                            (
                                graph.arcs[arc.physical].id,
                                arc.jump.map(|v| v as i32).unwrap_or(-1),
                            )
                        })
                        .collect();
                    ((bucket.id.vertex, bucket.id.bucket), arcs)
                })
                .collect();
            assert_eq!(
                actual,
                decode(&captured["buckets"]),
                "after step {step}, forward {forward}"
            );
        }
    }
}

#[cfg(feature = "clp")]
#[test]
fn packing_supported_arc_outside_memory_is_charged_in_pricing() {
    use baldes::cuts::{CutClass, MemoryKind, RankOneCut};
    use std::collections::{BTreeMap, BTreeSet};
    let mut solver = baldes::pricing::Solver::<2>::new(5, 0).unwrap();
    solver.graph.edit().graph_id = 7;
    let supported = solver.graph.add_arc(0, 1, -3.).unwrap();
    let remembered = solver.graph.add_arc(1, 2, 0.).unwrap();
    let terminal = solver.graph.add_arc(2, 4, 0.).unwrap();
    let unreachable = solver.graph.add_arc(3, 1, 0.).unwrap();
    solver.cuts.cuts.push(RankOneCut {
        weights: vec![0; 5],
        graph_support: BTreeMap::new(),
        graph_pricing_support: BTreeMap::new(),
        general_kpath_support: None,
        memory: vec![false; 5],
        graph_dependent_memory: false,
        graph_memories: BTreeMap::new(),
        memory_kind: MemoryKind::Arc,
        arc_memory: BTreeSet::from([remembered, terminal, unreachable]),
        graph_arc_memories: BTreeMap::new(),
        arc_weights: BTreeMap::from([(supported, 1), (remembered, 1)]),
        graph_arc_weights: BTreeMap::new(),
        visit_once: false,
        type2: false,
        denominator: 2,
        dual: -1.,
        class: CutClass::Packing,
        rhs: 1,
        kind: -1,
        id: 0,
        ordered_support: None,
    });
    let priced = solver.price_phase(0, 10, false).unwrap();
    assert_eq!(priced.columns.len(), 1);
    assert_eq!(
        priced.columns[0].arcs,
        vec![supported, remembered, terminal]
    );
    assert_eq!(priced.columns[0].cost, -2.);
    let enumerated = solver.enumerate(100.).unwrap();
    assert_eq!(enumerated.len(), 1);
    assert_eq!(enumerated[0].cost, -2.);
}

#[cfg(feature = "clp")]
#[test]
fn packing_arc_memory_pricing_matches_controlled_native_bound() {
    use baldes::cuts::{CutClass, MemoryKind, RankOneCut};
    use std::collections::{BTreeMap, BTreeSet};
    let cases: serde_json::Value = serde_json::from_str(include_str!(
        "../fixtures/cuts/packing_arc_pricing_native.json"
    ))
    .unwrap();
    for (case_index, case) in cases.as_array().unwrap().iter().enumerate() {
        let mut solver = baldes::pricing::Solver::<2>::new(8, 1).unwrap();
        solver.graph.edit().graph_id = case["graph_id"].as_i64().unwrap() as i32;
        for vertex in 0..8 {
            solver.graph.set_bounds(vertex, 0, 0., 2.).unwrap();
        }
        let mut weights = BTreeMap::new();
        for (index, arc) in case["arcs"].as_array().unwrap().iter().enumerate() {
            let tail = arc["tail"].as_u64().unwrap() as usize;
            let head = arc["head"].as_u64().unwrap() as usize;
            let id = solver.graph.add_arc(tail, head, 0.).unwrap();
            let edge = solver.graph.arc_mut(id).unwrap();
            edge.consumption[0] = if head == 7 { 0. } else { 1. };
            edge.reduced_cost = case["costs"][index].as_f64().unwrap();
            if head == 1 && tail != 4 && tail != 5 {
                weights.insert(id, 1);
            }
        }
        for captured in case["cuts"].as_array().unwrap() {
            let mut vertex_weights = vec![0; 8];
            for set in captured["sets"].as_array().unwrap() {
                let set = set.as_u64().unwrap() as usize;
                if set != 0 {
                    vertex_weights[set + 1] += 1;
                }
            }
            solver.cuts.cuts.push(RankOneCut {
                weights: vertex_weights,
                graph_support: BTreeMap::new(),
                graph_pricing_support: BTreeMap::new(),
                general_kpath_support: None,
                memory: vec![false; 8],
                graph_dependent_memory: false,
                graph_memories: BTreeMap::new(),
                memory_kind: MemoryKind::Arc,
                arc_memory: captured["memory"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_i64().unwrap() as i32)
                    .collect::<BTreeSet<_>>(),
                graph_arc_memories: BTreeMap::new(),
                arc_weights: weights.clone(),
                graph_arc_weights: BTreeMap::new(),
                visit_once: false,
                type2: false,
                denominator: captured["denominator"].as_i64().unwrap() as i32,
                dual: -captured["price"].as_f64().unwrap(),
                class: CutClass::Packing,
                rhs: 1,
                kind: -1,
                id: captured["id"].as_i64().unwrap() as i32,
                ordered_support: None,
            });
        }
        let phase = case["phase"].as_u64().unwrap() as usize;
        let initial_phase_routes: Option<BTreeSet<_>> = if phase == 0 {
            None
        } else {
            Some(
                solver
                    .price_phase(phase, 100, false)
                    .unwrap()
                    .columns
                    .into_iter()
                    .map(|r| r.arcs)
                    .collect(),
            )
        };
        let result = solver.price_phase(0, 100, false).unwrap();
        let rust_routes: BTreeSet<_> = result.columns.iter().map(|r| r.arcs.clone()).collect();
        let native_routes: BTreeSet<Vec<i32>> = case["native_routes"]
            .as_array()
            .unwrap()
            .iter()
            .map(|route| {
                route
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|arc| arc.as_i64().unwrap() as i32)
                    .collect()
            })
            .collect();
        let recorded: BTreeMap<Vec<i32>, f64> = case["recorded_routes"]
            .as_array()
            .unwrap()
            .iter()
            .map(|route| {
                (
                    route["arcs"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(|arc| arc.as_i64().unwrap() as i32)
                        .collect(),
                    route["value"].as_f64().unwrap(),
                )
            })
            .collect();
        assert!(native_routes.is_subset(&rust_routes));
        if case["phase"].as_u64().unwrap() == 0 {
            assert_eq!(native_routes, rust_routes);
            for route in &result.columns {
                assert!((route.cost - recorded[&route.arcs]).abs() < 1e-9);
            }
            let mut expected: Vec<_> = case["candidate_values"]
                .as_array()
                .unwrap()
                .iter()
                .map(|candidate| candidate["value"].as_f64().unwrap())
                .collect();
            let mut actual: Vec<_> = result.columns.iter().map(|route| route.cost).collect();
            expected.sort_by(f64::total_cmp);
            actual.sort_by(f64::total_cmp);
            assert_eq!(expected.len(), actual.len());
            for (a, b) in actual.iter().zip(expected) {
                assert!(
                    (a - b).abs() < 1e-9,
                    "case {case_index}: Rust {a}, native {b}"
                );
            }
            for candidate in case["candidate_values"].as_array().unwrap() {
                let arc = candidate["label_arc"].as_i64().unwrap() as i32;
                let matches: Vec<_> = result
                    .columns
                    .iter()
                    .filter(|r| r.arcs.contains(&arc))
                    .collect();
                let route = if matches.len() == 1 {
                    matches[0]
                } else {
                    matches.iter().copied().find(|r| r.arcs.len() == 2).unwrap()
                };
                assert!(
                    (route.cost - candidate["value"].as_f64().unwrap()).abs() < 1e-9,
                    "case {case_index}, candidate arc {arc}"
                );
            }
        } else {
            let phased = solver.price_phase(phase, 100, false).unwrap();
            let phased_routes: BTreeSet<_> =
                phased.columns.iter().map(|r| r.arcs.clone()).collect();
            assert_eq!(initial_phase_routes.unwrap(), phased_routes);
            assert!((phased.minimum_reduced_cost - case["bound"].as_f64().unwrap()).abs() < 1e-9);
            assert_eq!(native_routes, phased_routes, "case {case_index}");
            for route in &phased.columns {
                assert!((route.cost - recorded[&route.arcs]).abs() < 1e-9);
            }
            let mut expected: Vec<_> = case["candidate_values"]
                .as_array()
                .unwrap()
                .iter()
                .map(|candidate| candidate["value"].as_f64().unwrap())
                .collect();
            let mut actual: Vec<_> = phased.columns.iter().map(|route| route.cost).collect();
            expected.sort_by(f64::total_cmp);
            actual.sort_by(f64::total_cmp);
            assert_eq!(expected.len(), actual.len());
            for (a, b) in actual.iter().zip(expected) {
                assert!(
                    (a - b).abs() < 1e-9,
                    "case {case_index}: Rust {a}, native {b}"
                );
            }
            let mut native_by_arc = BTreeMap::<i32, Vec<f64>>::new();
            for candidate in case["candidate_values"].as_array().unwrap() {
                native_by_arc
                    .entry(candidate["label_arc"].as_i64().unwrap() as i32)
                    .or_default()
                    .push(candidate["value"].as_f64().unwrap());
            }
            let mut rust_by_arc = BTreeMap::<i32, Vec<f64>>::new();
            for route in &phased.columns {
                rust_by_arc
                    .entry(*route.arcs.last().unwrap())
                    .or_default()
                    .push(route.cost);
            }
            assert_eq!(native_by_arc.len(), rust_by_arc.len());
            for (arc, mut expected) in native_by_arc {
                let mut actual = rust_by_arc.remove(&arc).unwrap();
                expected.sort_by(f64::total_cmp);
                actual.sort_by(f64::total_cmp);
                assert_eq!(expected.len(), actual.len());
                for (a, b) in actual.iter().zip(expected) {
                    assert!((a - b).abs() < 1e-9, "case {case_index}, arc {arc}");
                }
            }
            if let Some(zero) = case["native_zero_routes"].as_array() {
                let zero_routes: BTreeSet<Vec<i32>> = zero
                    .iter()
                    .map(|route| {
                        route
                            .as_array()
                            .unwrap()
                            .iter()
                            .map(|v| v.as_i64().unwrap() as i32)
                            .collect()
                    })
                    .collect();
                for cut in &mut solver.cuts.cuts {
                    cut.dual = 0.;
                }
                let zero_priced = solver.price_phase(phase, 100, false).unwrap();
                let rust_zero: BTreeSet<_> =
                    zero_priced.columns.iter().map(|r| r.arcs.clone()).collect();
                assert_eq!(zero_routes, rust_zero);
                if case_index == 6 {
                    solver.cuts.cuts[0].dual = -1.;
                    let incomplete = solver.price_phase(phase, 100, false).unwrap();
                    let incomplete_routes: BTreeSet<_> =
                        incomplete.columns.iter().map(|r| r.arcs.clone()).collect();
                    assert_ne!(native_routes, incomplete_routes);
                }
            }
        }
        if let Some(routes) = case["native_routes_with_phase_flag_2"].as_array() {
            let native_flag_routes: BTreeSet<Vec<i32>> = routes
                .iter()
                .map(|route| {
                    route
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(|arc| arc.as_i64().unwrap() as i32)
                        .collect()
                })
                .collect();
            assert_eq!(native_flag_routes, rust_routes);
        }
        assert!(
            (result.minimum_reduced_cost - case["bound"].as_f64().unwrap()).abs() < 1e-9,
            "graph {}: Rust {}, native {}",
            solver.graph.graph_id,
            result.minimum_reduced_cost,
            case["bound"]
        );
    }
}

#[cfg(feature = "clp")]
#[test]
fn covering_arc_preparation_matches_controlled_native_tables() {
    use baldes::cuts::state::{
        change_along_arc, prepare_arcs, prepare_vertices, CutResourceInfo, LimitedInput,
        LimitedPreparation, LIMITED_WORDS,
    };
    use std::collections::{BTreeMap, BTreeSet};
    let fixture: serde_json::Value = serde_json::from_str(include_str!(
        "../fixtures/cuts/covering_arc_preparation_native.json"
    ))
    .unwrap();
    for graph in fixture["graphs"].as_array().unwrap() {
        let mut solver = baldes::pricing::Solver::<2>::new(8, 1).unwrap();
        let mut coefficients = BTreeMap::new();
        for arc in graph["arcs"].as_array().unwrap() {
            let tail = arc["tail"].as_u64().unwrap() as usize;
            let head = arc["head"].as_u64().unwrap() as usize;
            let id = solver.graph.add_arc(tail, head, 0.).unwrap();
            assert_eq!(id, arc["id"].as_i64().unwrap() as i32);
            if head == 1 && tail != 4 && tail != 5 {
                coefficients.insert(id, graph["arc_increment"].as_i64().unwrap_or(-1) as i32);
            }
        }
        let packing = graph["packing"].as_bool().unwrap_or(false);
        let cut = LimitedInput {
            covering: !packing,
            type2: false,
            arc_memory: graph.get("arc_memory").is_some(),
            denominator: if packing { 2 } else { -2 },
            dual: if packing { 1. } else { -1. },
            vertex_coefficient: if packing {
                vec![0, 0, 0, 0, 1, 0, 1, 0]
            } else {
                vec![0, 0, 0, 0, -1, 0, -1, 0]
            },
            arc_coefficient: coefficients,
            memory_vertices: graph["memory"].as_array().map_or_else(
                || (1..=6).collect(),
                |values| {
                    values
                        .iter()
                        .map(|v| v.as_u64().unwrap() as usize)
                        .collect()
                },
            ),
            memory_arcs: graph["arc_memory"]
                .as_array()
                .map_or_else(BTreeSet::new, |values| {
                    values.iter().map(|v| v.as_i64().unwrap() as i32).collect()
                }),
        };
        let mut info = CutResourceInfo::default();
        let mut prep = LimitedPreparation::default();
        assert!(prepare_vertices(
            &mut info,
            &mut prep,
            &solver.graph,
            std::slice::from_ref(&cut),
            &[]
        ));
        for forward in [true, false] {
            prepare_arcs(
                &mut info,
                &mut prep,
                &solver.graph,
                std::slice::from_ref(&cut),
                &[],
                forward,
            );
            let table = if forward {
                &info.forward
            } else {
                &info.backward
            };
            let mut actual = Vec::new();
            for (arc, row) in table.iter().enumerate() {
                for t in &row.ordinary {
                    actual.push((
                        arc,
                        vec![
                            t.mask as i32,
                            t.denominator,
                            t.source_word,
                            t.source_shift,
                            t.target_word,
                            t.target_shift,
                            t.increment,
                        ],
                        t.covering,
                        t.overflow_cost.to_bits(),
                    ));
                }
            }
            let field = if forward {
                "prepared"
            } else {
                "backward_prepared"
            };
            let mut expected: Vec<_> = graph[field]
                .as_array()
                .unwrap()
                .iter()
                .map(|t| {
                    (
                        t["arc"].as_u64().unwrap() as usize,
                        t["descriptor"]
                            .as_array()
                            .unwrap()
                            .iter()
                            .map(|x| x.as_i64().unwrap() as i32)
                            .collect::<Vec<_>>(),
                        t["covering"].as_bool().unwrap(),
                        t["cost"].as_f64().unwrap().to_bits(),
                    )
                })
                .collect();
            actual.sort();
            expected.sort();
            assert_eq!(actual, expected);
        }
        for (field, table) in [("chain", &info.forward), ("backward_chain", &info.backward)] {
            let Some(chain) = graph[field].as_array() else {
                continue;
            };
            let mut words = [0; LIMITED_WORDS];
            let mut cost = 0.;
            for step in chain {
                change_along_arc(
                    &mut words,
                    &mut cost,
                    &table[step["arc"].as_u64().unwrap() as usize].ordinary,
                );
                let expected: Vec<_> = step["after_words"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_u64().unwrap() as u32)
                    .collect();
                assert_eq!(words.as_slice(), expected);
                assert_eq!(cost.to_bits(), step["cost"].as_f64().unwrap().to_bits());
            }
        }
        for probe in graph["executions"].as_array().unwrap() {
            let mut words = [0; LIMITED_WORDS];
            words[0] = probe["before"].as_u64().unwrap() as u32;
            let mut cost = probe["initial"].as_f64().unwrap();
            let table = if probe["forward"].as_bool().unwrap() {
                &info.forward
            } else {
                &info.backward
            };
            change_along_arc(
                &mut words,
                &mut cost,
                &table[probe["arc"].as_u64().unwrap() as usize].ordinary,
            );
            let expected: Vec<_> = probe["after_words"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap() as u32)
                .collect();
            assert_eq!(words.as_slice(), expected);
            assert_eq!(cost.to_bits(), probe["cost"].as_f64().unwrap().to_bits());
        }
    }
}

#[cfg(feature = "clp")]
#[test]
fn covering_full_memory_support_matches_controlled_native_capture() {
    use baldes::cuts::state::{
        apply_vertex_support, change_full_memory_along_arc, prepare_full_memory, CutResourceInfo,
        FullInput, FullPreparation, FULL_WORDS,
    };
    use std::collections::BTreeMap;
    let probes: serde_json::Value = serde_json::from_str(include_str!(
        "../fixtures/cuts/covering_full_support_native.json"
    ))
    .unwrap();
    let arcs: serde_json::Value = serde_json::from_str(include_str!(
        "../fixtures/cuts/covering_limited_arcs_native.json"
    ))
    .unwrap();
    let mut solver = baldes::pricing::Solver::<2>::new(11, 1).unwrap();
    for arc in arcs["arcs"].as_array().unwrap() {
        solver
            .graph
            .add_arc(
                arc["tail"].as_u64().unwrap() as usize,
                arc["head"].as_u64().unwrap() as usize,
                0.,
            )
            .unwrap();
    }
    let cut = FullInput {
        ordered_support: vec![(1, -1), (2, -1), (3, -1)],
        denominator: -2,
        covering: true,
        dual: -1.,
        vertex_coefficient: vec![0, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0],
        arc_coefficient: BTreeMap::new(),
    };
    let mut info = CutResourceInfo::default();
    let mut prep = FullPreparation::default();
    assert!(prepare_full_memory(
        &mut info,
        &mut prep,
        &solver.graph,
        &[cut],
        &[],
        true
    ));
    for probe in probes.as_array().unwrap() {
        let mut words = [0; FULL_WORDS];
        words[0] = probe["before"].as_u64().unwrap() as u32;
        let mut cost = probe["initial"].as_f64().unwrap();
        apply_vertex_support(
            &mut words,
            &mut cost,
            &info.vertices[probe["vertex"].as_u64().unwrap() as usize].full_memory_support,
        );
        let expected: Vec<_> = probe["after_words"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u32)
            .collect();
        assert_eq!(words.as_slice(), expected);
        assert_eq!(cost.to_bits(), probe["cost"].as_f64().unwrap().to_bits());
    }
    let arc_probes: serde_json::Value = serde_json::from_str(include_str!(
        "../fixtures/cuts/covering_full_arcs_native.json"
    ))
    .unwrap();
    for probe in arc_probes.as_array().unwrap() {
        let mut words = [0; FULL_WORDS];
        words[0] = probe["before"].as_u64().unwrap() as u32;
        let mut cost = probe["initial"].as_f64().unwrap();
        let table = if probe["forward"].as_bool().unwrap() {
            &info.forward
        } else {
            &info.backward
        };
        change_full_memory_along_arc(
            &mut words,
            &mut cost,
            &table[probe["arc"].as_u64().unwrap() as usize].full_memory,
        );
        let expected: Vec<_> = probe["after_words"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u32)
            .collect();
        assert_eq!(words.as_slice(), expected);
        assert_eq!(cost.to_bits(), probe["cost"].as_f64().unwrap().to_bits());
    }
    let cut = FullInput {
        ordered_support: vec![(11, -1)],
        denominator: -2,
        covering: true,
        dual: -1.,
        vertex_coefficient: vec![0; 11],
        arc_coefficient: BTreeMap::from([(0, -1)]),
    };
    let mut info = CutResourceInfo::default();
    let mut prep = FullPreparation::default();
    assert!(prepare_full_memory(
        &mut info,
        &mut prep,
        &solver.graph,
        &[cut],
        &[],
        true
    ));
    let probes: serde_json::Value = serde_json::from_str(include_str!(
        "../fixtures/cuts/covering_full_arc_charge_native.json"
    ))
    .unwrap();
    for probe in probes.as_array().unwrap() {
        let mut words = [0; FULL_WORDS];
        words[0] = probe["before"].as_u64().unwrap() as u32;
        let mut cost = probe["initial"].as_f64().unwrap();
        let table = if probe["forward"].as_bool().unwrap() {
            &info.forward
        } else {
            &info.backward
        };
        change_full_memory_along_arc(
            &mut words,
            &mut cost,
            &table[probe["arc"].as_u64().unwrap() as usize].full_memory,
        );
        let expected: Vec<_> = probe["after_words"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u32)
            .collect();
        assert_eq!(words.as_slice(), expected);
        assert_eq!(cost.to_bits(), probe["cost"].as_f64().unwrap().to_bits());
    }
}

#[cfg(feature = "clp")]
#[test]
fn covering_limited_arc_transitions_match_controlled_native_capture() {
    use baldes::cuts::state::{
        change_along_arc, prepare_arcs, prepare_vertices, ArcTransition, CutResourceInfo,
        LimitedInput, LimitedPreparation, LIMITED_WORDS,
    };
    use std::collections::{BTreeMap, BTreeSet};
    let fixture: serde_json::Value = serde_json::from_str(include_str!(
        "../fixtures/cuts/covering_limited_arcs_native.json"
    ))
    .unwrap();
    let mut solver = baldes::pricing::Solver::<2>::new(11, 1).unwrap();
    for arc in fixture["arcs"].as_array().unwrap() {
        let id = solver
            .graph
            .add_arc(
                arc["tail"].as_u64().unwrap() as usize,
                arc["head"].as_u64().unwrap() as usize,
                0.,
            )
            .unwrap();
        assert_eq!(id, arc["id"].as_i64().unwrap() as i32);
    }
    let cut = LimitedInput {
        covering: true,
        type2: false,
        arc_memory: false,
        denominator: -2,
        dual: -1.,
        vertex_coefficient: vec![0, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0],
        arc_coefficient: BTreeMap::new(),
        memory_vertices: vec![1, 2, 3],
        memory_arcs: BTreeSet::new(),
    };
    let mut info = CutResourceInfo::default();
    let mut prep = LimitedPreparation::default();
    assert!(prepare_vertices(
        &mut info,
        &mut prep,
        &solver.graph,
        std::slice::from_ref(&cut),
        &[]
    ));
    for forward in [true, false] {
        prepare_arcs(
            &mut info,
            &mut prep,
            &solver.graph,
            std::slice::from_ref(&cut),
            &[],
            forward,
        );
    }
    for probe in fixture["probes"].as_array().unwrap() {
        let mut words = [0; LIMITED_WORDS];
        words[0] = probe["before"].as_u64().unwrap() as u32;
        let mut cost = probe["initial"].as_f64().unwrap();
        let table = if probe["forward"].as_bool().unwrap() {
            &info.forward
        } else {
            &info.backward
        };
        change_along_arc(
            &mut words,
            &mut cost,
            &table[probe["arc"].as_u64().unwrap() as usize].ordinary,
        );
        let expected: Vec<_> = probe["after_words"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u32)
            .collect();
        assert_eq!(words.as_slice(), expected);
        assert_eq!(cost.to_bits(), probe["cost"].as_f64().unwrap().to_bits());
    }
    let probes: serde_json::Value = serde_json::from_str(include_str!(
        "../fixtures/cuts/covering_limited_arc_charge_native.json"
    ))
    .unwrap();
    for probe in probes.as_array().unwrap() {
        let mut words = [0; LIMITED_WORDS];
        words[0] = probe["before"].as_u64().unwrap() as u32;
        let mut cost = probe["initial"].as_f64().unwrap();
        let transition = ArcTransition {
            mask: 1,
            denominator: -2,
            source_word: 0,
            source_shift: 0,
            target_word: probe["destination"].as_i64().unwrap() as i32,
            target_shift: 0,
            increment: -1,
            covering: true,
            overflow_cost: -1.,
        };
        change_along_arc(&mut words, &mut cost, &[transition]);
        let expected: Vec<_> = probe["after_words"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u32)
            .collect();
        assert_eq!(words.as_slice(), expected);
        assert_eq!(cost.to_bits(), probe["cost"].as_f64().unwrap().to_bits());
    }
}

#[cfg(feature = "clp")]
#[test]
fn standard_cut_active_enumeration_matches_controlled_native_routes() {
    use baldes::cuts::{CutClass, MemoryKind, RankOneCut};
    use std::collections::{BTreeMap, BTreeSet};

    let graph: serde_json::Value = serde_json::from_str(include_str!(
        "../fixtures/coupled/bucket_paired_regeneration_native.json"
    ))
    .unwrap();
    let cases: serde_json::Value = serde_json::from_str(include_str!(
        "../fixtures/cuts/standard_rank_one_forced_enumeration_native.json"
    ))
    .unwrap();
    for case in cases.as_array().unwrap() {
        let mut solver = baldes::pricing::Solver::<2>::new(7, 1).unwrap();
        for vertex in 0..7 {
            solver
                .graph
                .set_bounds(vertex, 0, 0., case["capacity"].as_f64().unwrap_or(2.))
                .unwrap();
        }
        let membership = (0..7)
            .map(|v| {
                if (1..=5).contains(&v) {
                    vec![v - 1]
                } else {
                    vec![]
                }
            })
            .collect();
        solver
            .graph
            .configure_ng(5, membership, vec![(0..5).collect(); 7])
            .unwrap();
        for (index, arc) in graph["arcs"].as_array().unwrap().iter().enumerate() {
            let id = solver
                .graph
                .add_arc(
                    arc[0].as_u64().unwrap() as usize,
                    arc[1].as_u64().unwrap() as usize,
                    0.,
                )
                .unwrap();
            let edge = solver.graph.arc_mut(id).unwrap();
            edge.consumption[0] = arc[2].as_f64().unwrap();
            edge.reduced_cost = case["costs"][index].as_f64().unwrap();
        }
        let support: Vec<_> = case["support"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as usize)
            .collect();
        let mut weights = vec![0; 7];
        for &v in &support {
            weights[v] = 1;
        }
        let mut memory = vec![false; 7];
        for v in case["memory"].as_array().unwrap() {
            memory[v.as_u64().unwrap() as usize] = true;
        }
        if let Some(transitions) = case["transitions"].as_array() {
            use baldes::cuts::state::{
                apply_vertex_support, change_along_arc, prepare_arcs, prepare_vertices,
                CutResourceInfo, LimitedInput, LimitedPreparation, LIMITED_WORDS,
            };
            let cut = LimitedInput {
                covering: false,
                type2: false,
                arc_memory: false,
                denominator: 2,
                dual: case["price"].as_f64().unwrap(),
                vertex_coefficient: weights.clone(),
                arc_coefficient: BTreeMap::new(),
                memory_vertices: (0..7).filter(|&v| memory[v]).collect(),
                memory_arcs: BTreeSet::new(),
            };
            let mut info = CutResourceInfo::default();
            let mut prep = LimitedPreparation::default();
            assert!(prepare_vertices(
                &mut info,
                &mut prep,
                &solver.graph,
                std::slice::from_ref(&cut),
                &[]
            ));
            for captured in case["support_transitions"].as_array().unwrap() {
                let mut words: Vec<_> = captured["before"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|x| x.as_u64().unwrap() as u32)
                    .collect();
                let expected: Vec<_> = captured["after"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|x| x.as_u64().unwrap() as u32)
                    .collect();
                let mut cost = captured["before_cost"].as_f64().unwrap();
                apply_vertex_support(
                    &mut words,
                    &mut cost,
                    &info.vertices[captured["vertex"].as_u64().unwrap() as usize].support,
                );
                assert_eq!(words, expected);
                assert_eq!(
                    cost.to_bits(),
                    captured["after_cost"].as_f64().unwrap().to_bits()
                );
            }
            let second = LimitedInput {
                covering: false,
                type2: false,
                arc_memory: false,
                denominator: 2,
                dual: 0.25,
                vertex_coefficient: vec![0, 0, 0, 1, 1, 1, 0],
                arc_coefficient: BTreeMap::new(),
                memory_vertices: vec![3, 4, 5],
                memory_arcs: BTreeSet::new(),
            };
            let mut multiple = CutResourceInfo::default();
            let mut multiple_prep = LimitedPreparation::default();
            assert!(prepare_vertices(
                &mut multiple,
                &mut multiple_prep,
                &solver.graph,
                &[cut.clone(), second.clone()],
                &[]
            ));
            for captured in case["multicut_support"].as_array().unwrap() {
                let mut words: Vec<_> = captured["before"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|x| x.as_u64().unwrap() as u32)
                    .collect();
                let expected: Vec<_> = captured["after"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|x| x.as_u64().unwrap() as u32)
                    .collect();
                let mut cost = captured["before_cost"].as_f64().unwrap();
                apply_vertex_support(
                    &mut words,
                    &mut cost,
                    &multiple.vertices[captured["vertex"].as_u64().unwrap() as usize].support,
                );
                assert_eq!(words, expected);
                assert_eq!(
                    cost.to_bits(),
                    captured["after_cost"].as_f64().unwrap().to_bits()
                );
            }
            for forward in [true, false] {
                prepare_arcs(
                    &mut multiple,
                    &mut multiple_prep,
                    &solver.graph,
                    &[cut.clone(), second.clone()],
                    &[],
                    forward,
                );
            }
            for captured in case["multicut_arcs"].as_array().unwrap() {
                let mut words = [0; LIMITED_WORDS];
                for (dst, value) in words.iter_mut().zip(captured["before"].as_array().unwrap()) {
                    *dst = value.as_u64().unwrap() as u32;
                }
                let expected: Vec<_> = captured["after"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|x| x.as_u64().unwrap() as u32)
                    .collect();
                let mut cost = captured["before_cost"].as_f64().unwrap();
                let table = if captured["forward"].as_bool().unwrap() {
                    &multiple.forward
                } else {
                    &multiple.backward
                };
                change_along_arc(
                    &mut words,
                    &mut cost,
                    &table[captured["arc"].as_u64().unwrap() as usize].ordinary,
                );
                assert_eq!(words.as_slice(), expected);
                assert_eq!(
                    cost.to_bits(),
                    captured["after_cost"].as_f64().unwrap().to_bits()
                );
            }
            prepare_arcs(
                &mut info,
                &mut prep,
                &solver.graph,
                std::slice::from_ref(&cut),
                &[],
                true,
            );
            prepare_arcs(&mut info, &mut prep, &solver.graph, &[cut], &[], false);
            for (forward, transitions) in [
                (true, transitions),
                (false, case["backward_transitions"].as_array().unwrap()),
            ] {
                let mut words = [0; LIMITED_WORDS];
                let mut cost = 0.;
                for captured in transitions {
                    let arc = captured["arc"].as_u64().unwrap() as usize;
                    let before: Vec<_> = captured["before"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(|x| x.as_u64().unwrap() as u32)
                        .collect();
                    let after: Vec<_> = captured["after"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(|x| x.as_u64().unwrap() as u32)
                        .collect();
                    assert_eq!(words.as_slice(), before);
                    assert_eq!(cost, captured["before_cost"].as_f64().unwrap());
                    let table = if forward {
                        &info.forward
                    } else {
                        &info.backward
                    };
                    change_along_arc(&mut words, &mut cost, &table[arc].ordinary);
                    assert_eq!(words.as_slice(), after);
                    assert_eq!(cost, captured["after_cost"].as_f64().unwrap());
                    apply_vertex_support(
                        &mut words,
                        &mut cost,
                        &info.vertices[if forward {
                            solver.graph.arcs[arc].head
                        } else {
                            solver.graph.arcs[arc].tail
                        }]
                        .support,
                    );
                }
            }
        }
        solver.cuts.cuts.push(RankOneCut {
            weights,
            graph_support: BTreeMap::new(),
            graph_pricing_support: BTreeMap::new(),
            general_kpath_support: None,
            memory,
            graph_dependent_memory: false,
            graph_memories: BTreeMap::new(),
            memory_kind: MemoryKind::Vertex,
            arc_memory: BTreeSet::new(),
            graph_arc_memories: BTreeMap::new(),
            arc_weights: BTreeMap::new(),
            graph_arc_weights: BTreeMap::new(),
            visit_once: false,
            type2: false,
            denominator: 2,
            dual: -case["price"].as_f64().unwrap(),
            class: CutClass::Packing,
            rhs: (support.len() / 2) as i32,
            kind: -1,
            id: 1,
            ordered_support: Some(support.into_iter().map(|v| (v, 1)).collect()),
        });
        if let Some(second) = case.get("second_cut") {
            let mut cut = solver.cuts.cuts[0].clone();
            cut.weights.fill(0);
            let support: Vec<_> = second["support"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap() as usize)
                .collect();
            for &v in &support {
                cut.weights[v] = 1;
            }
            cut.memory.fill(false);
            for v in second["memory"].as_array().unwrap() {
                cut.memory[v.as_u64().unwrap() as usize] = true;
            }
            cut.dual = -second["price"].as_f64().unwrap();
            cut.id = 2;
            cut.ordered_support = Some(support.into_iter().map(|v| (v, 1)).collect());
            solver.cuts.cuts.push(cut);
        }
        solver.bucket_steps[0] = 0.5;
        let cutoff = case["cutoff"].as_f64().unwrap();
        let mut expected: Vec<Vec<i32>> = case["routes"]
            .as_array()
            .unwrap()
            .iter()
            .map(|route| {
                route
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|id| id.as_i64().unwrap() as i32)
                    .collect()
            })
            .collect();
        expected.sort();
        if case["capacity"].as_f64() == Some(3.)
            && case["price"].as_f64() == Some(0.5)
            && case.get("second_cut").is_none()
        {
            // Visits support vertex 3, forgets at vertex 4, then re-enters at 1.
            assert!(expected.contains(&vec![12, 16, 20, 1]));
        }
        let mut before: Vec<_> = solver
            .enumerate(cutoff)
            .unwrap()
            .into_iter()
            .map(|r| r.arcs)
            .collect();
        before.sort();
        assert_eq!(before, expected, "before fixing, price {}", case["price"]);
        solver.complete_labels(true).unwrap();
        solver.fix(cutoff).unwrap();
        let grid = &solver.engine.as_ref().unwrap().grid;
        for native in case["captures"].as_array().unwrap() {
            let forward = native["forward"].as_bool().unwrap();
            let direction = if forward {
                &grid.forward
            } else {
                &grid.backward
            };
            let actual: BTreeMap<_, Vec<_>> = direction
                .buckets
                .iter()
                .flatten()
                .filter(|b| b.reachable)
                .map(|b| {
                    (
                        (b.id.vertex, b.id.bucket),
                        b.arcs
                            .iter()
                            .map(|a| {
                                (
                                    solver.graph.arcs[a.physical].id,
                                    a.jump.map(|v| v as i32).unwrap_or(-1),
                                )
                            })
                            .collect(),
                    )
                })
                .collect();
            let expected_arcs: BTreeMap<_, Vec<_>> = native["buckets"]
                .as_array()
                .unwrap()
                .iter()
                .map(|b| {
                    (
                        (
                            b["vertex"].as_u64().unwrap() as usize,
                            b["bucket"].as_u64().unwrap() as usize,
                        ),
                        b["arcs"]
                            .as_array()
                            .unwrap()
                            .iter()
                            .map(|a| (a[0].as_i64().unwrap() as i32, a[1].as_i64().unwrap() as i32))
                            .collect(),
                    )
                })
                .collect();
            assert_eq!(
                actual, expected_arcs,
                "price {} cutoff {cutoff} forward {forward}",
                case["price"]
            );
        }
        let mut after: Vec<_> = solver
            .enumerate(cutoff)
            .unwrap()
            .into_iter()
            .map(|r| r.arcs)
            .collect();
        after.sort();
        assert_eq!(after, expected, "after fixing, price {}", case["price"]);
    }
}

#[cfg(feature = "clp")]
#[test]
fn paired_fixing_matches_native_bucket_arc_survivors() {
    use std::collections::BTreeMap;

    let capture: serde_json::Value = serde_json::from_str(include_str!(
        "../fixtures/coupled/bucket_paired_regeneration_native.json"
    ))
    .unwrap();
    for case in capture["cases"].as_array().unwrap() {
        let mut solver = baldes::pricing::Solver::<2>::new(7, 1).unwrap();
        for vertex in 0..7 {
            solver.graph.set_bounds(vertex, 0, 0., 2.5).unwrap();
        }
        let membership = (0..7)
            .map(|v| {
                if (1..=5).contains(&v) {
                    vec![v - 1]
                } else {
                    vec![]
                }
            })
            .collect();
        solver
            .graph
            .configure_ng(5, membership, vec![(0..5).collect(); 7])
            .unwrap();
        let costs = case["fixing"]["costs"].as_array().unwrap();
        for (index, arc) in capture["arcs"].as_array().unwrap().iter().enumerate() {
            let id = solver
                .graph
                .add_arc(
                    arc[0].as_u64().unwrap() as usize,
                    arc[1].as_u64().unwrap() as usize,
                    0.,
                )
                .unwrap();
            let edge = solver.graph.arc_mut(id).unwrap();
            edge.consumption[0] = arc[2].as_f64().unwrap();
            edge.reduced_cost = costs[index].as_f64().unwrap();
        }
        install_capture_front_with_preprocessing(
            &mut solver.graph,
            case["increment"].as_f64().unwrap_or(0.),
            true,
        );
        if let Some(row) = case.get("rank_one") {
            use baldes::cuts::{CutClass, MemoryKind, RankOneCut};
            use std::collections::BTreeSet;
            let mut weights = vec![0; 7];
            let support: Vec<_> = row["sets"]
                .as_array()
                .unwrap()
                .iter()
                .zip(row["numerators"].as_array().unwrap())
                .map(|(set, weight)| {
                    let vertex = set.as_u64().unwrap() as usize + 1;
                    let weight = weight.as_i64().unwrap() as i32;
                    weights[vertex] = weight;
                    (vertex, weight)
                })
                .collect();
            let mut memory = vec![false; 7];
            for vertex in row["memory"][0]["values"].as_array().unwrap() {
                memory[vertex.as_u64().unwrap() as usize] = true;
            }
            solver.cuts.cuts.push(RankOneCut {
                weights,
                graph_support: BTreeMap::new(),
                graph_pricing_support: BTreeMap::new(),
                general_kpath_support: None,
                memory,
                graph_dependent_memory: false,
                graph_memories: BTreeMap::new(),
                memory_kind: MemoryKind::Vertex,
                arc_memory: BTreeSet::new(),
                graph_arc_memories: BTreeMap::new(),
                arc_weights: BTreeMap::new(),
                graph_arc_weights: BTreeMap::new(),
                visit_once: false,
                type2: false,
                denominator: row["denominator"].as_i64().unwrap() as i32,
                dual: -row["dual"].as_f64().unwrap(),
                class: CutClass::Packing,
                rhs: 2,
                kind: -1,
                id: row["id"].as_i64().unwrap() as i32,
                ordered_support: Some(support),
            });
        }
        let step = case["step"].as_f64().unwrap();
        solver.bucket_steps[0] = step;
        solver.complete_labels(true).unwrap();
        if let Some(source_keys) = case["native_source_keys"].as_array() {
            let engine = solver.engine.as_ref().unwrap();
            let mut multiplicities = 0;
            let mut above_cutoff = 0;
            for captured in source_keys {
                let forward = captured["forward"].as_bool().unwrap();
                let storage = if forward {
                    &engine.forward
                } else {
                    &engine.backward
                };
                let vertex = captured["vertex"].as_u64().unwrap() as usize;
                let bucket = captured["bucket"].as_u64().unwrap() as usize;
                let hot = &storage.buckets[vertex][bucket].hot;
                let actual_labels: Vec<_> = hot
                    .key
                    .iter()
                    .copied()
                    .zip(hot.ng.iter().map(|&mask| mask as u64))
                    .collect();
                if let Some(full_ng) = captured["full_ng_labels"].as_array() {
                    let mut expected: Vec<_> = full_ng
                        .iter()
                        .map(|pair| (pair[0].as_f64().unwrap(), pair[1].as_u64().unwrap()))
                        .collect();
                    let mut actual = actual_labels.clone();
                    let order = |a: &(f64, u64), b: &(f64, u64)| {
                        a.0.total_cmp(&b.0).then_with(|| a.1.cmp(&b.1))
                    };
                    expected.sort_by(order);
                    actual.sort_by(order);
                    assert_eq!(actual, expected, "{forward} {vertex} {bucket}");
                }
                for (key, mask) in captured["keys"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .zip(captured["masks"].as_array().unwrap())
                {
                    assert!(
                        actual_labels.contains(&(key.as_f64().unwrap(), mask.as_u64().unwrap()))
                    );
                }
                let mut actual = storage.buckets[vertex][bucket].hot.key.clone();
                let mut native: Vec<_> = captured["keys"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_f64().unwrap())
                    .collect();
                actual.sort_by(f64::total_cmp);
                native.sort_by(f64::total_cmp);
                if actual != native {
                    assert_eq!(actual.len(), 4);
                    let masks: std::collections::BTreeSet<_> = hot.ng.iter().copied().collect();
                    assert_eq!(masks.len(), 4);
                    actual.dedup();
                    if native.is_empty() {
                        assert!(!forward);
                        assert!(actual
                            .iter()
                            .all(|key| *key >= case["fixing"]["cutoff"].as_f64().unwrap()));
                        above_cutoff += 1;
                    } else {
                        assert_eq!(actual, native);
                        multiplicities += 1;
                    }
                }
            }
            if case["name"] == "forced_price_2" {
                assert_eq!((multiplicities, above_cutoff), (5, 5));
            } else {
                assert_eq!((multiplicities, above_cutoff), (10, 0));
            }
        }
        let cutoff = case["fixing"]["cutoff"].as_f64().unwrap();
        let before_routes = if case.get("rank_one").is_some() {
            let mut baseline = solver.fork_for_node();
            Some(baseline.enumerate(cutoff).unwrap())
        } else {
            None
        };
        solver.fix(cutoff).unwrap();
        let grid = &solver.engine.as_ref().unwrap().grid;
        if let Some(ng_captures) = case.get("ng_captures") {
            // The diagnostic native run changes insertion to unlimited,
            // NG-aware dominance while leaving the fixing cutoff unchanged.
            assert_eq!(&case["captures"], ng_captures);
        }
        for native in case["captures"].as_array().unwrap() {
            let forward = native["forward"].as_bool().unwrap();
            let direction = if forward {
                &grid.forward
            } else {
                &grid.backward
            };
            let actual: BTreeMap<_, Vec<_>> = direction
                .buckets
                .iter()
                .flatten()
                .filter(|b| b.reachable)
                .map(|bucket| {
                    let arcs = bucket
                        .arcs
                        .iter()
                        .filter(|a| a.jump.is_none())
                        .map(|arc| solver.graph.arcs[arc.physical].id)
                        .collect();
                    ((bucket.id.vertex, bucket.id.bucket), arcs)
                })
                .collect();
            let expected: BTreeMap<_, Vec<_>> = native["before"]
                .as_array()
                .unwrap()
                .iter()
                .map(|bucket| {
                    let key = (
                        bucket["vertex"].as_u64().unwrap() as usize,
                        bucket["bucket"].as_u64().unwrap() as usize,
                    );
                    let arcs = bucket["arcs"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .filter(|a| a[1].as_i64().unwrap() == -1)
                        .map(|arc| arc[0].as_i64().unwrap() as i32)
                        .collect();
                    (key, arcs)
                })
                .collect();
            if case.get("rank_one").is_some() {
                let mut extras = 0;
                for (key, rust) in &actual {
                    let native = &expected[key];
                    let retained: Vec<_> = rust
                        .iter()
                        .copied()
                        .filter(|a| native.contains(a))
                        .collect();
                    assert_eq!(&retained, native, "native survivor missing: {key:?}");
                    extras += rust.len() - native.len();
                }
                assert_eq!(actual.len(), expected.len());
                let price = case["rank_one"]["dual"].as_f64().unwrap();
                let expected_extra = if let Some(expected) = case["expected_extra"].as_array() {
                    expected[usize::from(!forward)].as_u64().unwrap() as usize
                } else if price >= 2. {
                    if forward {
                        60
                    } else {
                        40
                    }
                } else if forward {
                    20
                } else {
                    0
                };
                assert_eq!(
                    extras, expected_extra,
                    "case {} direction {}",
                    case["name"], forward
                );
            } else {
                assert_eq!(actual, expected, "step {step}, forward {forward}");
            }
            let actual_jumps: BTreeMap<_, Vec<_>> = direction
                .buckets
                .iter()
                .flatten()
                .filter(|b| b.reachable)
                .map(|bucket| {
                    let arcs = bucket
                        .arcs
                        .iter()
                        .map(|arc| {
                            (
                                solver.graph.arcs[arc.physical].id,
                                arc.jump.map(|v| v as i32).unwrap_or(-1),
                            )
                        })
                        .collect();
                    ((bucket.id.vertex, bucket.id.bucket), arcs)
                })
                .collect();
            let expected_jumps: BTreeMap<_, Vec<_>> = native["buckets"]
                .as_array()
                .unwrap()
                .iter()
                .map(|bucket| {
                    let key = (
                        bucket["vertex"].as_u64().unwrap() as usize,
                        bucket["bucket"].as_u64().unwrap() as usize,
                    );
                    let arcs = bucket["arcs"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(|arc| {
                            (
                                arc[0].as_i64().unwrap() as i32,
                                arc[1].as_i64().unwrap() as i32,
                            )
                        })
                        .collect();
                    (key, arcs)
                })
                .collect();
            if case.get("rank_one").is_some() {
                let mut extra = 0;
                let mut missing = 0;
                for (key, native) in &expected_jumps {
                    let rust = &actual_jumps[key];
                    extra += rust.iter().filter(|a| !native.contains(a)).count();
                    missing += native.iter().filter(|a| !rust.contains(a)).count();
                }
                let price = case["rank_one"]["dual"].as_f64().unwrap();
                let expected_extra = if let Some(expected) = case["expected_extra"].as_array() {
                    expected[usize::from(!forward)].as_u64().unwrap() as usize
                } else if price >= 2. {
                    if forward {
                        60
                    } else {
                        40
                    }
                } else if forward {
                    20
                } else {
                    0
                };
                assert_eq!(
                    (extra, missing),
                    (expected_extra, 0),
                    "case {} direction {}",
                    case["name"],
                    forward
                );
            } else {
                assert_eq!(
                    actual_jumps, expected_jumps,
                    "regenerated step {step}, forward {forward}"
                );
            }
        }
        if let Some(before) = before_routes {
            let mut expected: Vec<_> = before.into_iter().map(|route| route.arcs).collect();
            let mut actual: Vec<_> = solver
                .enumerate(cutoff)
                .unwrap()
                .into_iter()
                .map(|route| route.arcs)
                .collect();
            expected.sort();
            actual.sort();
            assert!(!expected.is_empty());
            assert_eq!(actual, expected);
            if let Some(routes) = case["native_routes"].as_array() {
                let mut native: Vec<Vec<i32>> = routes
                    .iter()
                    .map(|route| {
                        route
                            .as_array()
                            .unwrap()
                            .iter()
                            .map(|id| id.as_i64().unwrap() as i32)
                            .collect()
                    })
                    .collect();
                native.sort();
                assert_eq!(actual, native);
            }
        }
    }
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
    solver.graph.edit().vertices[0].elements.clear();
    solver.graph.edit().vertices[6].elements.clear();
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
    solver.graph.edit().vertices[0].elements.clear();
    solver.graph.edit().vertices[6].elements.clear();
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
    solver.graph.edit().vertices[0].elements.clear();
    solver.graph.edit().vertices[2].elements.clear();
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
    solver.graph.edit().vertices[0].elements.clear();
    solver.graph.edit().vertices[4].elements.clear();
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
    solver.graph.edit().vertices[0].elements.clear();
    solver.graph.edit().vertices[4].elements.clear();
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
    let mut regenerated_jumps = 0;
    for together in [false, true] {
        for presence in [false, true] {
            for cutoff in [-1., 0., 1., 2.] {
                let mut solver = baldes::pricing::Solver::<2>::new(5, 1).unwrap();
                solver.graph.edit().vertices[0].elements.clear();
                solver.graph.edit().vertices[4].elements.clear();
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
                let engine = fixed.engine.as_ref().unwrap();
                let jumps = [&engine.grid.forward, &engine.grid.backward]
                    .into_iter()
                    .flat_map(|direction| direction.buckets.iter().flatten())
                    .flat_map(|bucket| &bucket.arcs)
                    .filter(|arc| arc.jump.is_some())
                    .count();
                if jumps != 0 {
                    regenerated_jumps += jumps;
                    fixed.fix(cutoff).unwrap();
                }
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
    assert!(regenerated_jumps > 0);
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
    solver.graph.edit().vertices[0].elements.clear();
    solver.graph.edit().vertices[sink].elements.clear();
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
        edge.bounds[0] = Some(
            Window::new(
                arc["bounds"][0].as_f64().unwrap(),
                arc["bounds"][1].as_f64().unwrap(),
            )
            .unwrap(),
        );
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
            // Prepared arc windows include arrival tightening beyond the
            // vertex hull; use the captured bounds for exact bucket admission.
            waiting_bounds: Window::new(
                arc["bounds"][0].as_f64().unwrap(),
                arc["bounds"][1].as_f64().unwrap(),
            )
            .unwrap(),
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

#[cfg(feature = "clp")]
#[test]
fn custom_enumeration_registered_capture_preserves_both_orientations() {
    let captured: serde_json::Value = serde_json::from_str(include_str!(
        "../fixtures/coupled/custom_enumeration_native.json"
    ))
    .unwrap();
    assert_eq!(
        captured["reference_sha256"],
        "103caf4a9c58d6141e935848294bff16677f400323466dc23cd4b9e5c15889d6"
    );
    let costs = captured["physical_arc_costs"].as_array().unwrap();
    let mut solver = baldes::pricing::Solver::<2>::new(7, 1).unwrap();
    solver.graph.edit().vertices[0].elements.clear();
    solver.graph.edit().vertices[6].elements.clear();
    for vertex in 0..7 {
        solver.graph.set_bounds(vertex, 0, 0., 2.).unwrap();
    }
    for customer in 1..=5 {
        let mut edges = vec![(0, customer), (customer, 6)];
        edges.extend(
            (1..=5)
                .filter(|&other| other != customer)
                .map(|other| (customer, other)),
        );
        for (tail, head) in edges {
            let id = solver
                .graph
                .add_arc(tail, head, if tail == 0 { 1. } else { 0. })
                .unwrap();
            let arc = solver.graph.arc_mut(id).unwrap();
            arc.reduced_cost = costs[id as usize].as_f64().unwrap();
            arc.consumption[0] = if head == 6 { 0. } else { 1. };
        }
    }
    solver
        .graph
        .set_coupled_resources(&[CoupledResource::CustomResource {
            increments: vec![0.25; 30],
            weights: vec![0.125; 30],
            forward_bound: None,
            backward_bound: None,
        }])
        .unwrap();
    solver.bucket_steps[0] = 1.;
    let routes = solver
        .enumerate(captured["cutoff"].as_f64().unwrap())
        .unwrap();
    let mut actual: Vec<Vec<i32>> = routes.iter().map(|r| r.arcs.clone()).collect();
    actual.sort();
    let mut expected: Vec<Vec<i32>> = captured["paths"]
        .as_array()
        .unwrap()
        .iter()
        .map(|row| {
            row["arcs"]
                .as_array()
                .unwrap()
                .iter()
                .map(|arc| arc.as_i64().unwrap() as i32)
                .collect()
        })
        .collect();
    expected.sort();
    // The reference retains one orientation per customer pair. BALDES keeps
    // both because custom states do not establish path coalescing safety.
    assert_eq!(actual.len(), 20);
    assert_eq!(expected.len(), 10);
    assert!(expected
        .iter()
        .all(|path| actual.binary_search(path).is_ok()));
    let sets = |paths: &[Vec<i32>]| {
        let mut sets: Vec<Vec<usize>> = paths
            .iter()
            .map(|path| {
                let mut vertices: Vec<_> = path
                    .iter()
                    .map(|id| solver.graph.arcs[*id as usize].head)
                    .filter(|&head| head != 6)
                    .collect();
                vertices.sort_unstable();
                vertices
            })
            .collect();
        sets.sort();
        sets.dedup();
        sets
    };
    assert_eq!(sets(&actual), sets(&expected));
    for route in routes {
        assert_eq!(
            solver
                .record_enumerated_custom_machine_path(&route.arcs)
                .unwrap(),
            Some(0.1875)
        );
        assert_eq!(route.cost, 0.);
    }
}

#[cfg(feature = "clp")]
#[test]
fn paired_front_dependent_knapsack_cut_reports_missing_resource_mapping() {
    use baldes::cuts::knapsack::{KnapsackKind, ResourceKnapsackCut, RouteLoadKnapsackCut};
    use std::collections::BTreeMap;

    let mut solver = baldes::pricing::Solver::<2>::new(4, 1).unwrap();
    solver.graph.edit().graph_id = 7;
    solver.graph.edit().vertices[0].elements.clear();
    solver.graph.edit().vertices[3].elements.clear();
    for vertex in 0..4 {
        solver.graph.set_bounds(vertex, 0, 0., 2.5).unwrap();
    }
    for (tail, head, consumption) in [(0, 1, 1.), (1, 2, 1.), (2, 3, 0.)] {
        let id = solver.graph.add_arc(tail, head, -1.).unwrap();
        solver.graph.arc_mut(id).unwrap().consumption[0] = consumption;
    }
    install_capture_front(&mut solver.graph, 1.);
    let baseline = solver.price_phase(0, 10, false).unwrap();
    assert!(!baseline.columns.is_empty());
    solver.cuts.knapsack_cuts.push(ResourceKnapsackCut {
        id: 0,
        row: 0,
        cut_class: 0,
        graph_resources: BTreeMap::from([(7, (1, 1.))]),
        cut: RouteLoadKnapsackCut {
            kind: KnapsackKind::Rounding,
            numerator: 1,
            denominator: 3,
            rhs: 4,
            lhs: 0.,
            violation: 0.,
            thresholds: vec![],
            coefficients: BTreeMap::from([(0, 0), (3, 1), (6, 2), (9, 3), (12, 4)]),
        },
        dual: 0.,
    });
    solver.knapsack_exact_coefficients = true;
    let zero_dual = solver.price_phase(0, 10, false).unwrap();
    assert_eq!(
        zero_dual.minimum_reduced_cost,
        baseline.minimum_reduced_cost
    );
    assert_eq!(zero_dual.columns.len(), baseline.columns.len());
    solver.cuts.knapsack_cuts[0].dual = -2.;
    let error = solver.price_phase(0, 10, false).unwrap_err();
    assert!(
        format!("{error:?}").contains("exact knapsack resource mapping"),
        "{error:?}"
    );
    let coefficients = solver.cuts.knapsack_cuts[0].cut.coefficients.clone();
    solver.cuts.knapsack_cuts[0].cut.coefficients = BTreeMap::from([(0, 0), (3, 0)]);
    let zero_coefficient = solver.price_phase(0, 10, false).unwrap();
    assert_eq!(
        zero_coefficient.minimum_reduced_cost,
        baseline.minimum_reduced_cost
    );
    assert_eq!(zero_coefficient.columns.len(), baseline.columns.len());
    assert!(!solver.enumerate(100.).unwrap().is_empty());
    let retained = solver.price_phase(0, 10, false).unwrap();
    assert_eq!(retained.minimum_reduced_cost, baseline.minimum_reduced_cost);
    solver.cuts.knapsack_cuts[0].cut.coefficients = coefficients;
    let error = solver.price_phase(0, 10, false).unwrap_err();
    assert!(
        format!("{error:?}").contains("exact knapsack resource mapping"),
        "{error:?}"
    );
}

#[cfg(feature = "clp")]
#[test]
fn paired_front_scalar_knapsack_charges_the_master_coefficient() {
    use baldes::cuts::knapsack::{KnapsackKind, ResourceKnapsackCut, RouteLoadKnapsackCut};
    use std::collections::BTreeMap;

    // Native paired-plus-scalar calls retained the same bound and paths when
    // the captured cut price was zeroed. Here we verify the distinct Rust
    // coefficient-consistent behavior on a single paired route.
    let native = include_str!("../fixtures/coupled/paired_scalar_knapsack_native.txt");
    assert_eq!(native.lines().count(), 3);
    assert!(native.lines().all(|line| line.starts_with("2.0 -2.0 ")));
    assert_eq!(
        include_str!("../fixtures/coupled/paired_scalar_knapsack_inspection_native.txt"),
        "2.0 -2.0 18\n"
    );
    let mut solver = baldes::pricing::Solver::<2>::new(4, 2).unwrap();
    solver.graph.edit().graph_id = 7;
    solver.graph.edit().vertices[0].elements.clear();
    solver.graph.edit().vertices[3].elements.clear();
    for vertex in 0..4 {
        for resource in 0..2 {
            solver.graph.set_bounds(vertex, resource, 0., 2.5).unwrap();
        }
    }
    for (tail, head, consumption) in [(0, 1, 1.), (1, 2, 1.), (2, 3, 0.)] {
        let id = solver.graph.add_arc(tail, head, -2.).unwrap();
        solver.graph.arc_mut(id).unwrap().consumption[0] = consumption;
        solver.graph.arc_mut(id).unwrap().consumption[1] = consumption;
    }
    install_capture_front(&mut solver.graph, 1.);
    solver.knapsack_resource_indices = BTreeMap::from([(2, 1)]);
    solver.knapsack_exact_coefficients = true;
    solver.cuts.knapsack_cuts.push(ResourceKnapsackCut {
        id: 0,
        row: 0,
        cut_class: 0,
        graph_resources: BTreeMap::from([(7, (2, 1.))]),
        cut: RouteLoadKnapsackCut {
            kind: KnapsackKind::Rounding,
            numerator: 1,
            denominator: 2,
            rhs: 4,
            lhs: 0.,
            violation: 0.,
            thresholds: vec![],
            coefficients: BTreeMap::from([(0, 0), (2, 1), (4, 2), (6, 3), (8, 4)]),
        },
        dual: 0.,
    });
    let without_price = solver.price_phase(0, 10, false).unwrap();
    solver.cuts.knapsack_cuts[0].dual = -2.;
    let with_price = solver.price_phase(0, 10, false).unwrap();
    let before = without_price
        .columns
        .iter()
        .find(|route| route.arcs == [0, 1, 2])
        .unwrap();
    let after = with_price
        .columns
        .iter()
        .find(|route| route.arcs == [0, 1, 2])
        .unwrap();
    assert_eq!(after.resources[1], 2.);
    assert_eq!(
        solver.knapsack_coefficient(&solver.cuts.knapsack_cuts[0], &after.resources),
        1
    );
    assert_eq!(after.cost - before.cost, 2.);

    // Reprice the retained universe after the cut price changes. The native
    // capture reaches inspection with price 2 but does not change its output
    // when that price is zeroed.
    solver.cuts.knapsack_cuts[0].dual = 0.;
    solver.price_phase(0, 10, false).unwrap();
    let retained = solver.enumerate(0.).unwrap();
    assert!(retained.iter().any(|route| route.arcs == [0, 1, 2]));
    let without_price = solver.price_phase(0, 10, false).unwrap();
    solver.cuts.knapsack_cuts[0].dual = -2.;
    let with_price = solver.price_phase(0, 10, false).unwrap();
    let before = without_price
        .columns
        .iter()
        .find(|route| route.arcs == [0, 1, 2])
        .unwrap();
    let after = with_price
        .columns
        .iter()
        .find(|route| route.arcs == [0, 1, 2])
        .unwrap();
    assert_eq!(after.cost - before.cost, 2.);
}
