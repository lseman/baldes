use super::*;
fn slope(value: f64, policy: EvaluationPolicy) -> EncodedFunction {
    EncodedFunction::SlopeSegments {
        function: PreparedSlopeFunction::from_prepared_pairs(vec![(value, 0.), (0., 10.)]).unwrap(),
        policy,
    }
}
fn fixture(policy: EvaluationPolicy) -> Graph<2> {
    let mut graph = Graph::new(3, 0).unwrap();
    graph.add_arc(0, 1, 0.).unwrap();
    graph.add_arc(1, 2, 0.).unwrap();
    graph
        .set_coupled_resources(&[
            CoupledResource::EncodedPiecewiseLinearResource {
                initial: -0.00005,
                bounds: Window::new(-1., 10.).unwrap(),
                functions: vec![slope(1., policy); 2],
            },
            CoupledResource::EncodedPiecewiseLinearDependent {
                parent: 0,
                initial: 0.,
                bounds: Window::new(0., 10.).unwrap(),
                functions: vec![slope(2., EvaluationPolicy::StrictDomain); 2],
            },
            CoupledResource::EncodedPiecewiseLinearCost {
                parent: 1,
                functions: vec![slope(3., EvaluationPolicy::StrictDomain); 2],
            },
        ])
        .unwrap();
    graph
}
#[test]
fn native_tolerance_survives_preprocessing_forward_and_every_suffix_join() {
    let graph = fixture(EvaluationPolicy::NativeTolerance);
    let kernel = graph.coupled_kernel().unwrap();
    assert_eq!(kernel.preprocessing().usable_arcs, 2);
    let (state, cost) = kernel.replay_state(&[0, 1]).unwrap().unwrap();
    assert_eq!(state.numeric_values(), Some(&[1., 4.][..]));
    assert_eq!(cost, 6.);
    for split in 0..=2 {
        let prefix: Vec<_> = (0..split).collect();
        let f = kernel.replay_state(&prefix).unwrap().unwrap().0;
        let mut b = kernel.root(false);
        for arc in (split..2).rev() {
            b = kernel.extend(&b, arc, false).unwrap().unwrap().0;
        }
        assert_eq!(kernel.join_resource_objective(&f, &b).unwrap(), Some(6.));
    }
    let snapshot = graph.clone();
    assert_eq!(
        snapshot.coupled_kernel().unwrap().replay(&[0, 1]).unwrap(),
        Some(6.)
    );
    let strict = fixture(EvaluationPolicy::StrictDomain);
    assert_eq!(
        strict.coupled_kernel().unwrap().replay(&[0, 1]).unwrap(),
        None
    );
}
#[test]
fn encoded_numeric_pricing_and_enumeration_share_literal_program_in_all_modes() {
    let mut graph = Graph::<2>::new(4, 1).unwrap();
    for v in 0..4 {
        graph.set_bounds(v, 0, 0., 2.).unwrap();
    }
    for (u, v) in [(0, 1), (1, 3), (0, 2), (2, 3)] {
        let arc = graph.add_arc(u, v, 0.).unwrap();
        graph.arc_mut(arc).unwrap().consumption[0] = 1.;
    }
    graph
        .set_coupled_resources(&[
            CoupledResource::EncodedPiecewiseLinearResource {
                initial: -0.00005,
                bounds: Window::new(-1., 10.).unwrap(),
                functions: vec![slope(1., EvaluationPolicy::NativeTolerance); 4],
            },
            CoupledResource::EncodedPiecewiseLinearDependent {
                parent: 0,
                initial: 0.,
                bounds: Window::new(0., 10.).unwrap(),
                functions: vec![slope(2., EvaluationPolicy::StrictDomain); 4],
            },
            CoupledResource::EncodedPiecewiseLinearCost {
                parent: 1,
                functions: [-4., -4., -1., -1.]
                    .into_iter()
                    .map(|cost| slope(cost, EvaluationPolicy::StrictDomain))
                    .collect(),
            },
        ])
        .unwrap();
    for mode in [0., 1., 2.] {
        let mut solver = crate::pricing::Solver::<2>::new(4, 1).unwrap();
        solver.graph = graph.clone();
        solver.enumeration_preserve_arc_paths = true;
        solver
            .set_option("BALDESuseBidirectionalSearch", mode)
            .unwrap();
        let result = solver.price_phase(0, 20, false).unwrap();
        assert!(result.exact);
        assert_eq!(result.minimum_reduced_cost, -8.);
        let mut costs: Vec<_> = solver
            .enumerate(0.)
            .unwrap()
            .into_iter()
            .map(|route| {
                assert_eq!(route.original_cost, route.cost);
                route.cost
            })
            .collect();
        costs.sort_by(f64::total_cmp);
        assert_eq!(costs, vec![-8., -2.]);
    }
}
#[test]
fn unknown_and_empty_enclosures_are_not_conflated() {
    let raw = slope(2., EvaluationPolicy::StrictDomain);
    assert!(matches!(
        raw.image(Range { lo: 0., hi: 1. }),
        FunctionEnclosure::Unknown
    ));
    assert!(matches!(
        raw.image(Range { lo: -2., hi: -1. }),
        FunctionEnclosure::Empty
    ));
    assert!(matches!(
        raw.inverse_image(Range { lo: 0., hi: 1. }, Range::point(999.)),
        FunctionEnclosure::Unknown
    ));
    assert_eq!(
        raw.preimage(Range { lo: 0., hi: 1. }, Range::point(999.))
            .unwrap(),
        Range { lo: 0., hi: 1. }
    );
    let tolerant = slope(2., EvaluationPolicy::NativeTolerance);
    assert!(matches!(
        tolerant.image(Range::point(-0.00005)),
        FunctionEnclosure::Unknown
    ));
    assert_eq!(tolerant.value(-0.00005).unwrap(), Some(2.));
    assert_eq!(raw.value(-0.00005).unwrap(), None);
}
#[test]
fn prepared_point_functions_execute_without_an_endpoint_knot_adapter() {
    let raw = EncodedFunction::SlopeSegments {
        function: PreparedSlopeFunction::constant_point(7., 2.).unwrap(),
        policy: EvaluationPolicy::StrictDomain,
    };
    let mut graph = Graph::<2>::new(2, 0).unwrap();
    graph.add_arc(0, 1, 0.).unwrap();
    graph
        .set_coupled_resources(&[
            CoupledResource::CumulativeCost {
                initial: 2.,
                increments: vec![0.],
                weights: vec![0.],
                sink_upper: None,
                source_upper: None,
            },
            CoupledResource::EncodedPiecewiseLinearCost {
                parent: 0,
                functions: vec![raw],
            },
        ])
        .unwrap();
    assert_eq!(
        graph.coupled_kernel().unwrap().replay(&[0]).unwrap(),
        Some(7.)
    );
}
#[cfg(feature = "clp")]
#[test]
fn encoded_function_deserialization_validates_both_encodings() {
    assert!(serde_json::from_str::<EncodedFunction>(
        r#"{"encoding":"knots","function":{"points":[[0,1]]}}"#
    )
    .is_err());
    assert!(serde_json::from_str::<EncodedFunction>(r#"{"encoding":"slope_segments","function":{"pairs":[[0,1],[2,0]]},"policy":"StrictDomain"}"#).is_err());
    assert!(serde_json::from_str::<EncodedFunction>(
        r#"{"encoding":"slope_segments","function":{"pairs":[[0,1]]}}"#
    )
    .is_err());
}
#[cfg(feature = "clp")]
#[test]
fn phase_i_keeps_literal_domains_and_round_trips_encoding() {
    let graph = fixture(EvaluationPolicy::NativeTolerance);
    let definitions = graph.coupled_kernel().unwrap().definitions();
    let json = serde_json::to_string(definitions).unwrap();
    let restored: Vec<CoupledResource> = serde_json::from_str(&json).unwrap();
    assert_eq!(restored, definitions);
    let mut kernel = graph.coupled_kernel().unwrap().clone();
    kernel.suppress_costs();
    assert_eq!(kernel.replay(&[0, 1]).unwrap(), Some(0.));
    assert!(kernel.certify(&[0, 1]).is_none());
    let strict = fixture(EvaluationPolicy::StrictDomain);
    let mut strict = strict.coupled_kernel().unwrap().clone();
    strict.suppress_costs();
    assert_eq!(strict.replay(&[0, 1]).unwrap(), None);
}
