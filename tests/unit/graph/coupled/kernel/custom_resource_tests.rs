use super::*;

fn custom(increments: &[f64], weights: &[f64]) -> CoupledResource {
    CoupledResource::CustomResource {
        increments: increments.to_vec(),
        weights: weights.to_vec(),
        forward_bound: None,
        backward_bound: None,
    }
}

fn chain(definition: CoupledResource, arcs: usize) -> Graph<2> {
    let mut graph = Graph::new(arcs + 1, 0).unwrap();
    for tail in 0..arcs {
        graph.add_arc(tail, tail + 1, 0.).unwrap();
    }
    graph.set_coupled_resources(&[definition]).unwrap();
    graph
}

#[test]
fn custom_signed_recurrence_and_every_join_split_match_independent_sums() {
    for seed in 0..24 {
        let increments: Vec<_> = (0..5).map(|i| ((seed + i * 7) % 9) as f64 - 4.).collect();
        let weights: Vec<_> = (0..5)
            .map(|i| ((seed * 3 + i * 5) % 7) as f64 - 3.)
            .collect();
        let graph = chain(custom(&increments, &weights), 5);
        let kernel = graph.coupled_kernel().unwrap();
        let cumulative_graph = chain(
            CoupledResource::CumulativeCost {
                initial: 0.,
                increments: increments.clone(),
                weights: weights.clone(),
                sink_upper: None,
                source_upper: None,
            },
            5,
        );
        let cumulative = cumulative_graph.coupled_kernel().unwrap();
        assert!(matches!(
            kernel.definitions()[0],
            CoupledResource::CustomResource { .. }
        ));
        assert_eq!(kernel.root(true).numeric_values(), Some(&[0.][..]));
        assert_eq!(kernel.replay(&[]).unwrap(), Some(0.));
        for length in 1..=5 {
            let arcs: Vec<_> = (0..length).collect();
            let (mut level, mut expected) = (0., 0.);
            for i in 0..length {
                level += increments[i];
                expected += level * weights[i];
            }
            let (state, cost) = kernel.replay_state(&arcs).unwrap().unwrap();
            assert_eq!(cost, expected);
            assert_eq!(state.numeric_values(), Some(&[level][..]));
            assert_eq!(cumulative.replay(&arcs).unwrap(), Some(expected));
            for split in 0..=length {
                // Standalone prefixes/suffixes intentionally bypass the full-route filter.
                let f = kernel.replay_state(&arcs[..split]).unwrap().unwrap().0;
                let mut b = kernel.root(false);
                for &arc in arcs[split..].iter().rev() {
                    b = kernel.extend(&b, arc, false).unwrap().unwrap().0;
                }
                assert_eq!(
                    kernel.join_resource_objective(&f, &b).unwrap(),
                    Some(expected)
                );
                let cross = kernel.join(&f, &b).unwrap().unwrap();
                assert_eq!(
                    f.resource_objective + b.resource_objective + cross,
                    expected
                );
            }
        }
    }
}

#[test]
fn custom_forward_and_backward_dominance_bounds_are_independent() {
    let mut graph = Graph::<2>::new(3, 0).unwrap();
    for (tail, head) in [(0, 1), (0, 1), (1, 2)] {
        graph.add_arc(tail, head, 0.).unwrap();
    }
    for forward_bound in [None, Some(101.)] {
        graph
            .set_coupled_resources(&[CoupledResource::CustomResource {
                increments: vec![3., 2., 0.],
                weights: vec![0., 100., 1.],
                forward_bound,
                backward_bound: None,
            }])
            .unwrap();
        let kernel = graph.coupled_kernel().unwrap();
        let root = kernel.root(true);
        let high = kernel.extend(&root, 0, true).unwrap().unwrap().0;
        let low = kernel.extend(&root, 1, true).unwrap().unwrap().0;
        assert_eq!(
            kernel.dominance_penalty(&high, &low, true),
            forward_bound.map(|_| 0.)
        );
    }
    let mut graph = Graph::<2>::new(3, 0).unwrap();
    for (tail, head) in [(0, 1), (1, 2), (1, 2), (2, 1)] {
        graph.add_arc(tail, head, 0.).unwrap();
    }
    for (forward_bound, backward_bound) in [(None, None), (Some(4.), None), (None, Some(5.))] {
        graph
            .set_coupled_resources(&[CoupledResource::CustomResource {
                increments: vec![1., 0., 0., 0.],
                weights: vec![0., 1., 3., 0.],
                forward_bound,
                backward_bound,
            }])
            .unwrap();
        let kernel = graph.coupled_kernel().unwrap();
        let root = kernel.root(false);
        let light = kernel.extend(&root, 1, false).unwrap().unwrap().0;
        let heavy = kernel.extend(&root, 2, false).unwrap().unwrap().0;
        let penalty = kernel.dominance_penalty(&heavy, &light, false);
        if backward_bound.is_some() {
            assert!((10. ..=10. + 1e-9).contains(&penalty.unwrap()));
        } else {
            assert_eq!(penalty, None);
        }
    }
}

#[test]
fn bounded_walk_domain_certifies_fractional_backward_entry_states() {
    let mut graph = Graph::<2>::new(3, 0).unwrap();
    for (tail, head) in [(0, 1), (1, 2), (1, 2)] {
        graph.add_arc(tail, head, 0.).unwrap();
    }
    graph
        .set_route_domain(crate::graph::RouteDomain::BoundedWalks { max_arcs: 2 })
        .unwrap();
    graph
        .set_coupled_resources(&[CoupledResource::CustomResource {
            increments: vec![0.25, 0., 0.],
            weights: vec![0., 1., 3.],
            forward_bound: None,
            backward_bound: None,
        }])
        .unwrap();

    let kernel = graph.coupled_kernel().unwrap();
    let root = kernel.root(false);
    let light = kernel.extend(&root, 1, false).unwrap().unwrap().0;
    let heavy = kernel.extend(&root, 2, false).unwrap().unwrap().0;

    let penalty = kernel.dominance_penalty(&heavy, &light, false).unwrap();
    assert!(penalty >= 0.5);
    assert!(penalty <= 1. + 1e-9);
}

#[test]
fn acyclic_physical_domain_bounds_fractional_backward_entry_states() {
    let mut graph = Graph::<2>::new(3, 0).unwrap();
    for (tail, head) in [(0, 1), (1, 2), (1, 2)] {
        graph.add_arc(tail, head, 0.).unwrap();
    }
    graph
        .set_coupled_resources(&[CoupledResource::CustomResource {
            increments: vec![0.25, 0., 0.],
            weights: vec![0., 1., 3.],
            forward_bound: None,
            backward_bound: None,
        }])
        .unwrap();

    let kernel = graph.coupled_kernel().unwrap();
    let root = kernel.root(false);
    let light = kernel.extend(&root, 1, false).unwrap().unwrap().0;
    let heavy = kernel.extend(&root, 2, false).unwrap().unwrap().0;

    let penalty = kernel.dominance_penalty(&heavy, &light, false).unwrap();
    assert!(penalty >= 0.5);
    assert!(penalty <= 1. + 1e-9);
}

#[test]
fn custom_pricing_and_enumeration_keep_original_cost_in_every_direction() {
    let mut graph = Graph::<2>::new(4, 1).unwrap();
    for vertex in 0..4 {
        graph.set_bounds(vertex, 0, 0., 4.).unwrap();
    }
    for (tail, head, cost) in [(0, 1, -10.), (1, 3, 0.), (0, 2, -9.), (2, 3, 0.)] {
        let arc = graph.add_arc(tail, head, cost).unwrap();
        graph.arc_mut(arc).unwrap().consumption[0] = 1.;
    }
    graph
        .set_coupled_resources(&[custom(&[2., 1., 1., 2.], &[1., 3., 2., 0.])])
        .unwrap();
    // Path 0-1-3: -10 + 2*1 + 3*3 = 1; path 0-2-3: -9 + 1*2 + 3*0 = -7.
    for direction in [0., 1., 2.] {
        let mut solver = crate::pricing::Solver::<2>::new(4, 1).unwrap();
        solver.graph = graph.clone();
        solver.enumeration_preserve_arc_paths = true;
        solver
            .set_option("BALDESuseBidirectionalSearch", direction)
            .unwrap();
        assert_eq!(
            solver
                .price_phase(0, 20, false)
                .unwrap()
                .minimum_reduced_cost,
            -7.
        );
        let mut costs: Vec<_> = solver
            .enumerate(100.)
            .unwrap()
            .into_iter()
            .map(|route| {
                assert_eq!(route.cost, route.original_cost);
                route.original_cost
            })
            .collect();
        costs.sort_by(f64::total_cmp);
        assert_eq!(costs, vec![-7., 1.]);
    }
}

#[test]
fn custom_numeric_parent_keeps_declaration_slots_and_zero_initial_state() {
    let mut graph = chain(custom(&[2., 1.], &[3., 4.]), 2);
    graph
        .set_coupled_resources(&[
            CoupledResource::CumulativeCost {
                initial: 7.,
                increments: vec![0.; 2],
                weights: vec![0.; 2],
                sink_upper: None,
                source_upper: None,
            },
            custom(&[2., 1.], &[3., 4.]),
            CoupledResource::Dependent {
                parent: 1,
                initial: 0.,
                bounds: Window::new(0., 10.).unwrap(),
                offsets: vec![0.; 2],
                factors: vec![1.; 2],
            },
            CoupledResource::PiecewiseLinearCost {
                parent: 2,
                functions: vec![
                    PiecewiseLinearFunction::new(vec![[0., 0.], [10., 10.]]).unwrap();
                    2
                ],
            },
        ])
        .unwrap();
    let kernel = graph.coupled_kernel().unwrap();
    let (state, cost) = kernel.replay_state(&[0, 1]).unwrap().unwrap();
    assert_eq!(state.numeric_values(), Some(&[7., 3., 5.][..]));
    assert_eq!(cost, 25.); // Custom charges 18, dependent state charges 2+5.
    #[cfg(feature = "clp")]
    {
        let (valid, bounds) = kernel.certify(&[0, 1]).unwrap();
        assert!(valid && bounds[0] <= cost && cost <= bounds[1]);
        let mut suppressed = kernel.clone();
        suppressed.suppress_costs();
        let (state, cost) = suppressed.replay_state(&[0, 1]).unwrap().unwrap();
        assert_eq!(cost, 0.);
        assert_eq!(state.numeric_values(), Some(&[7., 3., 5.][..]));
        assert!(suppressed.certify(&[0, 1]).unwrap().0);
    }
}

#[test]
fn custom_rejects_invalid_arc_data_and_dominance_bounds() {
    let mut graph = Graph::<2>::new(2, 0).unwrap();
    graph.add_arc(0, 1, 0.).unwrap();
    for (increments, weights) in [
        (vec![], vec![1.]),
        (vec![1.], vec![]),
        (vec![f64::INFINITY], vec![1.]),
        (vec![1.], vec![f64::NAN]),
    ] {
        assert!(graph
            .set_coupled_resources(&[custom(&increments, &weights)])
            .is_err());
    }
    for invalid in [-1., f64::INFINITY, f64::NAN] {
        for forward in [false, true] {
            assert!(graph
                .set_coupled_resources(&[CoupledResource::CustomResource {
                    increments: vec![1.],
                    weights: vec![1.],
                    forward_bound: forward.then_some(invalid),
                    backward_bound: (!forward).then_some(invalid),
                }])
                .is_err());
        }
    }
}

#[test]
fn custom_arithmetic_overflow_is_an_error_instead_of_infeasibility() {
    let graph = chain(custom(&[1e308, 1e308], &[0., 0.]), 2);
    assert!(graph.coupled_kernel().unwrap().replay(&[0, 1]).is_err());
}

#[cfg(feature = "clp")]
#[test]
fn custom_json_round_trip_preserves_kind_constants_and_fixed_initial_state() {
    let definition = CoupledResource::CustomResource {
        increments: vec![2., 1.],
        weights: vec![3., 4.],
        forward_bound: Some(7.),
        backward_bound: Some(3.),
    };
    let json = serde_json::to_value(&definition).unwrap();
    assert_eq!(json["kind"], "custom_resource");
    assert_eq!(json["forward_bound"], 7.);
    assert_eq!(json["backward_bound"], 3.);
    assert!(json.get("initial").is_none());
    assert_eq!(
        serde_json::from_value::<CoupledResource>(json.clone()).unwrap(),
        definition
    );
    let mut invalid = json;
    invalid["initial"] = 1.into();
    assert!(serde_json::from_value::<CoupledResource>(invalid).is_err());
    let default: CoupledResource =
        serde_json::from_str(r#"{"kind":"custom_resource","increments":[2,1],"weights":[3,4]}"#)
            .unwrap();
    assert_eq!(default, custom(&[2., 1.], &[3., 4.]));
    let graph = chain(default, 2);
    assert_eq!(
        graph.coupled_kernel().unwrap().replay(&[0, 1]).unwrap(),
        Some(18.)
    );
}

#[test]
fn cumulative_costs_remain_invariant_when_original_vertex_ids_are_permuted() {
    // Full reference preparation controls retain the counterexample: one
    // prepared head indexes the original vertex table at the wrong position.
    // BALDES resource arrays remain in the unchanged physical arc order.
    for (source, sink, interior) in [(0, 3, [1, 2]), (2, 0, [1, 3])] {
        let mut graph = Graph::<2>::new(5, 0).unwrap();
        graph.source = source;
        graph.sink = sink;
        graph.add_arc(4, 4, 0.).unwrap();
        for (tail, head, cost) in [
            (source, interior[0], -20.),
            (interior[0], sink, 0.),
            (source, interior[1], -20.),
            (interior[1], sink, 0.),
        ] {
            graph.add_arc(tail, head, cost).unwrap();
        }
        graph
            .set_coupled_resources(&[CoupledResource::CumulativeCost {
                initial: 0.,
                increments: vec![0., 2., 1., 1., 2.],
                weights: vec![0., 5., 0., 7., 0.],
                sink_upper: Some(13.),
                source_upper: Some(17.),
            }])
            .unwrap();
        let kernel = graph.coupled_kernel().unwrap();
        assert_eq!(kernel.replay(&[1, 2]).unwrap(), Some(10.));
        assert_eq!(kernel.replay(&[3, 4]).unwrap(), Some(7.));
        for direction in [0., 1., 2.] {
            let mut solver = crate::pricing::Solver::<2>::new(5, 0).unwrap();
            solver.graph = graph.clone();
            solver.enumeration_preserve_arc_paths = true;
            solver
                .set_option("BALDESuseBidirectionalSearch", direction)
                .unwrap();
            let priced = solver.price_phase(0, 8, false).unwrap();
            assert!(priced.exact);
            assert_eq!(priced.minimum_reduced_cost, -13.);
            let mut routes: Vec<_> = solver
                .enumerate(0.)
                .unwrap()
                .into_iter()
                .map(|route| (route.arcs, route.original_cost))
                .collect();
            routes.sort_by(|a, b| a.1.total_cmp(&b.1));
            assert_eq!(routes, vec![(vec![3, 4], -13.), (vec![1, 2], -10.)]);
        }
    }
}

#[test]
fn unspecified_custom_constants_do_not_assume_the_reference_zero_defaults_are_bounds() {
    // Reference registration defaults both constants to zero, without deriving
    // a bound from its positive future weights. None is not such a promise.
    let mut graph = Graph::<2>::new(3, 0).unwrap();
    for (tail, head) in [(0, 1), (0, 1), (1, 2)] {
        graph.add_arc(tail, head, 0.).unwrap();
    }
    graph
        .set_coupled_resources(&[custom(&[2., 1., 0.], &[0., 0., 1.])])
        .unwrap();
    let kernel = graph.coupled_kernel().unwrap();
    let root = kernel.root(true);
    let (high, high_cost) = kernel.extend(&root, 0, true).unwrap().unwrap();
    let (low, low_cost) = kernel.extend(&root, 1, true).unwrap().unwrap();
    assert_eq!((high_cost, low_cost), (0., 0.));
    assert!(!kernel
        .dominance_penalty(&high, &low, true)
        .is_some_and(|penalty| high_cost + penalty <= low_cost));
    assert_eq!(kernel.replay(&[0, 2]).unwrap(), Some(2.));
    assert_eq!(kernel.replay(&[1, 2]).unwrap(), Some(1.));
}
