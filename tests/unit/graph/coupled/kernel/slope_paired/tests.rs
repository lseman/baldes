use super::*;
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
