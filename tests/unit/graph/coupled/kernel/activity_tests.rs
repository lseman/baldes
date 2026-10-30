use super::*;

fn custom(increments: &[f64], weights: &[f64]) -> CoupledResource {
    CoupledResource::CustomResource {
        increments: increments.to_vec(),
        weights: weights.to_vec(),
        forward_bound: None,
        backward_bound: None,
    }
}
fn graph(resources: &[CoupledResource]) -> Graph<2> {
    let mut g = Graph::new(4, 0).unwrap();
    for (tail, head) in [(0, 1), (1, 3), (0, 2), (2, 3)] {
        g.add_arc(tail, head, 0.).unwrap();
    }
    g.set_coupled_resources(resources).unwrap();
    g
}

#[test]
fn registered_custom_and_cumulative_activities_keep_distinct_descriptor_identities() {
    // Activities from the actual two-path registered reference controls.
    let g = graph(&[
        CoupledResource::Selection {
            initial: 3,
            allowed: vec![3; 4],
            vertices: vec![],
            relaxed: false,
        },
        custom(&[1., 1., 1., 1.], &[3., 4., 0., 1.]),
        CoupledResource::CumulativeCost {
            initial: 0.,
            increments: vec![2., 1., 3., 1.],
            weights: vec![2., 2., 1., 1.],
            sink_upper: None,
            source_upper: None,
        },
    ]);
    let k = g.coupled_kernel().unwrap();
    for (path, expected) in [([0, 1], [0., 11., 10.]), ([2, 3], [0., 2., 7.])] {
        let costs = k.replay_cost_components(&path).unwrap().unwrap();
        assert_eq!(costs.resources, expected);
        assert_eq!(costs.total, expected.iter().sum::<f64>());
        assert_eq!(Some(costs.total), k.replay(&path).unwrap());
    }
    assert_eq!(
        k.replay_cost_components(&[]).unwrap().unwrap().resources,
        [0.; 3]
    );
}

#[test]
fn activity_replay_preserves_nonadditivity_and_repeated_arc_occurrences() {
    let mut g = Graph::<2>::new(3, 0).unwrap();
    for (tail, head) in [(0, 1), (0, 1), (1, 2), (1, 2)] {
        g.add_arc(tail, head, 0.).unwrap();
    }
    g.set_coupled_resources(&[custom(&[1., 2., 0., 0.], &[0., 0., 1., 2.])])
        .unwrap();
    let k = g.coupled_kernel().unwrap();
    let values: Vec<_> = [[0, 2], [0, 3], [1, 2], [1, 3]]
        .iter()
        .map(|p| k.replay_cost_components(p).unwrap().unwrap().resources[0])
        .collect();
    assert_eq!(values, [1., 2., 2., 4.]);
    assert_ne!(values[0] + values[3], values[1] + values[2]);
    // As with ordinary replay, this is an arc-position program, not a graph
    // connectivity validator. Repeated occurrences execute repeatedly.
    assert_eq!(
        k.replay_cost_components(&[0, 0, 3])
            .unwrap()
            .unwrap()
            .resources,
        [4.]
    );
}

#[test]
fn numeric_dependencies_and_literal_function_encodings_share_the_existing_program() {
    let identity = PiecewiseLinearFunction::new(vec![[-20., -20.], [20., 20.]]).unwrap();
    let g = graph(&[
        custom(&[1.; 4], &[2.; 4]),
        CoupledResource::Dependent {
            parent: 0,
            initial: 0.,
            bounds: Window {
                lower: -20.,
                upper: 20.,
            },
            offsets: vec![0.; 4],
            factors: vec![1.; 4],
        },
        CoupledResource::PiecewiseLinearCost {
            parent: 1,
            functions: vec![identity.clone(); 4],
        },
        CoupledResource::EncodedPiecewiseLinearCost {
            parent: 0,
            functions: vec![identity.into(); 4],
        },
    ]);
    let k = g.coupled_kernel().unwrap();
    let costs = k.replay_cost_components(&[0, 1]).unwrap().unwrap();
    assert_eq!(costs.resources, [6., 0., 4., 3.]);
    assert_eq!(costs.total, 13.);
    assert_eq!(k.replay(&[0, 1]).unwrap(), Some(13.));
    assert!(k.replay_cost_components(&[4]).is_err());
}

#[test]
fn mixed_accumulation_rounding_is_not_reconstructed_from_component_sums() {
    let g = graph(&[
        custom(&[1., 0., 0., 0.], &[1e16, -1e16, 0., 0.]),
        custom(&[1., 0., 0., 0.], &[1., 1., 0., 0.]),
    ]);
    let k = g.coupled_kernel().unwrap();
    let costs = k.replay_cost_components(&[0, 1]).unwrap().unwrap();
    assert_eq!(costs.resources, [0., 2.]);
    assert_eq!(costs.total, 1.);
    assert_eq!(k.replay(&[0, 1]).unwrap(), Some(costs.total));
}

#[test]
fn infeasibility_and_arithmetic_errors_do_not_publish_partial_activities() {
    let g = graph(&[
        custom(&[1.; 4], &[1.; 4]),
        CoupledResource::Selection {
            initial: 3,
            allowed: vec![1, 2, 3, 3],
            vertices: vec![],
            relaxed: false,
        },
    ]);
    assert_eq!(
        g.coupled_kernel()
            .unwrap()
            .replay_cost_components(&[0, 1])
            .unwrap(),
        None
    );
    let g = graph(&[custom(&[1.; 4], &[f64::MAX; 4])]);
    assert!(g
        .coupled_kernel()
        .unwrap()
        .replay_cost_components(&[0, 1])
        .is_err());
}

#[test]
fn independent_interval_front_families_keep_their_objectives() {
    let window = Window::new(0., 4.).unwrap();
    let line = |y: f64, slope: f64| {
        PiecewiseLinearFunction::new(vec![[0., y], [4., y + 4. * slope]]).unwrap()
    };
    let raw = |y: f64, slope: f64| {
        PreparedSlopeFunction::from_prepared_pairs(vec![(y, 0.), (slope, 4.)]).unwrap()
    };
    let g = graph(&[
        CoupledResource::IntervalCost {
            initial: window,
            bounds: window,
            increments: vec![1., -1., 0., 0.],
            functions: vec![line(0., 1.), line(5., -1.), line(0., 0.), line(0., 0.)],
        },
        CoupledResource::SlopeIntervalCost {
            initial: window,
            bounds: window,
            increments: vec![1., -1., 0., 0.],
            functions: vec![raw(0., 1.), raw(5., -1.), raw(0., 0.), raw(0., 0.)],
            policy: FrontOperationPolicy::StrictDomain,
        },
    ]);
    let k = g.coupled_kernel().unwrap();
    let costs = k.replay_cost_components(&[0, 1]).unwrap().unwrap();
    assert_eq!(costs.resources, [6., 6.]);
    assert_eq!(costs.total, 12.);
    assert_eq!(k.replay(&[0, 1]).unwrap(), Some(12.));
    let empty = k.replay_cost_components(&[]).unwrap().unwrap();
    assert_eq!(empty.resources, [0.; 2]);
    assert_eq!(empty.total, 0.);
}

#[cfg(feature = "clp")]
#[test]
fn suppressed_components_are_zero_and_cost_function_domains_still_restrict_paths() {
    let identity = PiecewiseLinearFunction::new(vec![[0., 0.], [1.5, 1.5]]).unwrap();
    let g = graph(&[
        custom(&[1.; 4], &[2.; 4]),
        CoupledResource::PiecewiseLinearCost {
            parent: 0,
            functions: vec![identity; 4],
        },
    ]);
    let mut k = g.coupled_kernel().unwrap().clone();
    assert_eq!(
        k.replay_cost_components(&[0]).unwrap().unwrap().resources,
        [2., 1.]
    );
    assert_eq!(k.replay_cost_components(&[0, 1]).unwrap(), None);
    k.suppress_costs();
    assert_eq!(
        k.replay_cost_components(&[0]).unwrap().unwrap().resources,
        [0., 0.]
    );
    assert_eq!(k.replay_cost_components(&[0, 1]).unwrap(), None);
}

#[cfg(feature = "clp")]
#[test]
fn actual_registered_preparations_replay_and_certify_individual_activities() {
    let fixture: serde_json::Value = serde_json::from_str(include_str!(
        "../../../../fixtures/coupled/metasolver-preparation-bounds.json"
    ))
    .unwrap();
    let mut checked = 0;
    for case in fixture["cases"].as_array().unwrap() {
        let prepared = &case["prepared"];
        let numbers = |key: &str| {
            prepared[key]
                .as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_f64().unwrap())
                .collect::<Vec<_>>()
        };
        let increments = numbers("increments");
        let weights = numbers("weights");
        let target = if case["input"]["kind"] == "custom" {
            custom(&increments, &weights)
        } else {
            CoupledResource::CumulativeCost {
                initial: 0.,
                increments: increments.clone(),
                weights: weights.clone(),
                sink_upper: None,
                source_upper: None,
            }
        };
        let mut g = Graph::<2>::new(5, 0).unwrap();
        for arc in prepared["prepared_arcs"].as_array().unwrap() {
            g.add_arc(
                arc["tail"].as_u64().unwrap() as usize,
                arc["head"].as_u64().unwrap() as usize,
                0.,
            )
            .unwrap();
        }
        g.set_coupled_resources(&[
            CoupledResource::Selection {
                initial: 1,
                allowed: vec![1; 5],
                vertices: vec![],
                relaxed: false,
            },
            target,
        ])
        .unwrap();
        let k = g.coupled_kernel().unwrap();
        let mut best = f64::INFINITY;
        for path in [[1, 2], [3, 4]] {
            let (mut level, mut expected) = (0., 0.);
            for arc in path {
                level += increments[arc];
                expected += level * weights[arc];
            }
            let costs = k.replay_cost_components(&path).unwrap().unwrap();
            assert_eq!(costs.resources, [0., expected], "{}", case["name"]);
            assert_eq!(costs.total, expected);
            let (valid, bounds) = k
                .certify_accumulated_activity(1, &path, &mut 100_000, None)
                .unwrap();
            assert!(valid && bounds[0] <= expected && expected <= bounds[1]);
            best = best.min(-20. + expected);
            checked += 1;
        }
        assert_eq!(best, case["input"]["bcRecRootLpVal"].as_f64().unwrap());
    }
    assert_eq!(checked, 20);
}

#[cfg(feature = "clp")]
#[test]
fn accumulated_activity_proofs_keep_unknown_budgets_and_source_domain_gates() {
    use std::time::Instant;
    let g = graph(&[
        custom(&[1.; 4], &[1.; 4]),
        CoupledResource::Selection {
            initial: 3,
            allowed: vec![1, 2, 3, 3],
            vertices: vec![],
            relaxed: true,
        },
    ]);
    let k = g.coupled_kernel().unwrap();
    assert_eq!(k.certify_accumulated_activity(0, &[0], &mut 0, None), None);
    assert_eq!(
        k.certify_accumulated_activity(0, &[0], &mut 100_000, Some(Instant::now())),
        None
    );
    assert_eq!(
        k.certify_accumulated_activity(0, &[0, 1], &mut 100_000, None),
        Some((false, [0.; 2]))
    );
    assert_eq!(
        k.certify_accumulated_activity(1, &[0], &mut 100_000, None),
        None
    );
    assert_eq!(
        k.certify_accumulated_activity(3, &[0], &mut 100_000, None),
        None
    );
    let g = graph(&[custom(&[1., 1., 1., 1.], &[1e16, -5e15, 0., 0.])]);
    let (valid, bounds) = g
        .coupled_kernel()
        .unwrap()
        .certify_accumulated_activity(0, &[0, 1], &mut 100_000, None)
        .unwrap();
    assert!(valid && bounds[0] <= 0. && bounds[1] >= 0.);
}

#[cfg(feature = "clp")]
#[test]
fn selected_paired_values_are_physical_and_survive_cost_suppression() {
    let corpus: serde_json::Value = serde_json::from_str(include_str!(
        "../../../../fixtures/coupled/metasolver-paired-preparation.json"
    ))
    .unwrap();
    let problem =
        crate::master::model::format::Model::from_json(&corpus["cases"][36]["model"].to_string())
            .unwrap()
            .problem::<2>()
            .unwrap();
    let graph = &problem.networks[0].graph;
    assert_eq!(graph.coupled_paired_values(&[1, 2]).unwrap(), vec![(0, 5.)]);
    let mut kernel = graph.coupled_kernel().unwrap().clone();
    kernel.suppress_costs();
    assert_eq!(kernel.replay(&[1, 2]).unwrap(), Some(0.));
    assert_eq!(
        kernel.selected_paired_values(&[1, 2]).unwrap(),
        Some(vec![(0, 5.)])
    );
}
