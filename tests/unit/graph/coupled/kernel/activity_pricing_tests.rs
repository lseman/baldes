use super::*;

fn model() -> Graph<2> {
    let mut g = Graph::new(4, 1).unwrap();
    g.elementary = true;
    g.vertices[0].elements.clear();
    g.vertices[3].elements.clear();
    for v in 0..4 {
        g.set_bounds(v, 0, 0., 2.).unwrap();
    }
    for (tail, head, cost) in [(0, 1, -20.), (1, 3, 0.), (0, 2, -20.), (2, 3, 0.)] {
        let id = g.add_arc(tail, head, cost).unwrap();
        g.arc_mut(id).unwrap().consumption[0] = 1.;
    }
    g.set_coupled_resources(&[
        CoupledResource::Selection {
            initial: 3,
            allowed: vec![3; 4],
            vertices: vec![],
            relaxed: false,
        },
        CoupledResource::CustomResource {
            increments: vec![1.; 4],
            weights: vec![3., 4., 0., 1.],
            forward_bound: None,
            backward_bound: None,
        },
    ])
    .unwrap();
    g
}

#[test]
fn pricing_adjustments_preserve_original_costs_and_every_directional_join() {
    for adjustment in [-3., -1., -0.5, 0., 0.5, 2.] {
        let mut graph = model();
        graph
            .set_coupled_pricing_adjustments(&[(1, adjustment)])
            .unwrap();
        let k = graph.coupled_kernel().unwrap();
        for (path, activity) in [([0, 1], 11.), ([2, 3], 2.)] {
            assert_eq!(k.replay(&path).unwrap(), Some(activity));
            assert_eq!(
                k.replay_pricing(&path).unwrap(),
                Some((1. + adjustment) * activity)
            );
            for split in 0..=2 {
                let (f, _) = k.replay_state(&path[..split]).unwrap().unwrap();
                let mut b = k.root(false);
                for &arc in path[split..].iter().rev() {
                    b = k
                        .extend_internal(
                            &b,
                            arc,
                            false,
                            false,
                            false,
                            &mut CoupledWorkspace::default(),
                        )
                        .unwrap()
                        .unwrap()
                        .0;
                }
                assert_eq!(
                    k.join_resource_objective(&f, &b).unwrap(),
                    Some((1. + adjustment) * activity)
                );
                let correction = k.join(&f, &b).unwrap().unwrap();
                assert_eq!(
                    f.pricing_objective().unwrap() + b.pricing_objective().unwrap() + correction,
                    (1. + adjustment) * activity
                );
            }
        }
        let old = graph.coupled_kernel().unwrap().identity.clone();
        graph
            .set_coupled_pricing_adjustments(&[(1, adjustment)])
            .unwrap();
        assert!(std::sync::Arc::ptr_eq(
            &old,
            &graph.coupled_kernel().unwrap().identity
        ));
    }
}

#[test]
fn numerical_pricing_and_retained_paths_do_not_replace_original_objectives() {
    for direction in [0., 1., 2.] {
        let mut solver = crate::pricing::Solver::<2>::new(4, 1).unwrap();
        solver.graph = model();
        solver
            .set_option("BALDESuseBidirectionalSearch", direction)
            .unwrap();
        solver.enumeration_preserve_arc_paths = true;
        for adjustment in [1., -2., -0.5, 0.] {
            solver
                .graph
                .set_coupled_pricing_adjustments(&[(1, adjustment)])
                .unwrap();
            let priced = solver.price_phase(0, 100, false).unwrap();
            let expected = [
                -20. + (1. + adjustment) * 11.,
                -20. + (1. + adjustment) * 2.,
            ];
            assert!(priced.exact);
            assert!(
                (priced.minimum_reduced_cost - expected[0].min(expected[1])).abs() < 1e-9,
                "{direction} {adjustment}: {priced:?}"
            );
            for route in priced.columns {
                let activity = if route.arcs == [0, 1] { 11. } else { 2. };
                assert_eq!(route.original_cost, -20. + activity);
                assert!((route.cost - (-20. + (1. + adjustment) * activity)).abs() < 1e-9);
            }
            let routes = solver.enumerate(100.).unwrap();
            assert_eq!(routes.len(), 2);
            for route in routes {
                let activity = if route.arcs == [0, 1] { 11. } else { 2. };
                assert_eq!(route.original_cost, -20. + activity);
                assert!((route.cost - (-20. + (1. + adjustment) * activity)).abs() < 1e-9);
            }
        }
    }
}

#[test]
fn rejected_updates_are_atomic_and_forks_keep_independent_adjustments() {
    let mut graph = model();
    graph.set_coupled_pricing_adjustments(&[(1, -2.)]).unwrap();
    let original = graph.coupled_kernel().unwrap().clone();
    for adjustments in [
        vec![(1, 0.), (1, 1.)],
        vec![(0, 1.)],
        vec![(9, 1.)],
        vec![(1, f64::NAN)],
    ] {
        assert!(graph.set_coupled_pricing_adjustments(&adjustments).is_err());
        assert_eq!(graph.coupled_kernel().unwrap(), &original);
    }
    let mut child = graph.clone();
    child.set_coupled_pricing_adjustments(&[]).unwrap();
    assert_eq!(
        graph.coupled_pricing_path_cost(&[0, 1]).unwrap(),
        Some(-11.)
    );
    assert_eq!(child.coupled_pricing_path_cost(&[0, 1]).unwrap(), Some(11.));
    assert_eq!(graph.coupled_path_cost(&[0, 1]).unwrap(), Some(11.));
}

#[cfg(feature = "clp")]
#[test]
fn phase_one_pricing_keeps_activity_duals_after_suppressing_model_costs() {
    let mut graph = model();
    graph.suppress_coupled_costs();
    graph.set_coupled_pricing_adjustments(&[(1, -1.)]).unwrap();
    let k = graph.coupled_kernel().unwrap();
    assert!(k.has_costs());
    assert_eq!(k.replay(&[0, 1]).unwrap(), Some(0.));
    assert_eq!(k.replay_pricing(&[0, 1]).unwrap(), Some(-11.));
    assert_eq!(k.accumulated_activity(1, &[0, 1]).unwrap(), 11.);
    assert_eq!(
        k.certify_accumulated_activity(1, &[0, 1], &mut 100_000, None),
        Some((true, [11., 11.]))
    );
    let (valid, bounds) = k
        .certify_pricing_with_budget(&[0, 1], &mut 100_000, None)
        .unwrap();
    assert!(valid && bounds[0] <= -11. && bounds[1] >= -11.);
    assert_eq!(k.certify(&[0, 1]), Some((true, [0., 0.])));
    assert_eq!(k.certify_pricing_with_budget(&[0, 1], &mut 0, None), None);
    let bound = k
        .pricing_objective_walk_lower_bound(
            &graph,
            2,
            &mut 100_000,
            std::time::Instant::now() + std::time::Duration::from_secs(2),
        )
        .unwrap();
    assert!(bound <= -11., "{bound}");
}

#[cfg(feature = "clp")]
#[test]
fn retained_reference_activity_duals_price_all_columns_at_the_independent_lp_points() {
    let fixture: serde_json::Value = serde_json::from_str(include_str!(
        "../../../../fixtures/coupled/metasolver-derived-activity.json"
    ))
    .unwrap();
    let rational = |value: &serde_json::Value| {
        let mut pieces = value.as_str().unwrap().split('/');
        let numerator: f64 = pieces.next().unwrap().parse().unwrap();
        numerator
            / pieces
                .next()
                .map(|p| p.parse::<f64>().unwrap())
                .unwrap_or(1.)
    };
    let mut comparisons = 0;
    for case in fixture["cases"].as_array().unwrap() {
        let q = case["objective_resource_coefficient"].as_f64().unwrap();
        let dual = rational(&case["oracle"]["row_dual"]);
        let reduced_coefficient = rational(&case["oracle"]["resource_reduced_coefficient"]);
        let arc_row = case["row"]["coefficient_on_first_path_arc"]
            .as_f64()
            .unwrap_or(0.);
        for direction in [0., 1., 2.] {
            let mut solver = crate::pricing::Solver::<2>::new(4, 1).unwrap();
            solver.graph = model();
            if case["kind"] == "cumulative" {
                let mut definitions = solver
                    .graph
                    .coupled_kernel()
                    .unwrap()
                    .definitions()
                    .to_vec();
                definitions[1] = CoupledResource::CumulativeCost {
                    initial: 0.,
                    increments: vec![2., 1., 3., 1.],
                    weights: vec![2., 2., 1., 1.],
                    sink_upper: None,
                    source_upper: None,
                };
                solver.graph.set_coupled_resources(&definitions).unwrap();
            }
            // q-1 supplies the extra objective coefficient only for this pricing
            // check; original-variable objective registration is a later layer.
            solver
                .graph
                .set_coupled_pricing_adjustments(&[(1, q - 1. - dual)])
                .unwrap();
            solver.graph.arcs[0].reduced_cost -= dual * arc_row;
            solver
                .set_option("BALDESuseBidirectionalSearch", direction)
                .unwrap();
            let priced = solver.price_phase(0, 100, false).unwrap();
            let activity = case["activity_values_by_path"].as_array().unwrap();
            let expected = [
                -20. - dual * arc_row + reduced_coefficient * activity[0].as_f64().unwrap(),
                -20. + reduced_coefficient * activity[1].as_f64().unwrap(),
            ];
            assert!(priced.exact);
            assert!(
                (priced.minimum_reduced_cost - expected[0].min(expected[1])).abs() < 1e-8,
                "{} {direction}",
                case["name"]
            );
            for route in priced.columns {
                let path = usize::from(route.arcs != [0, 1]);
                assert!((route.cost - expected[path]).abs() < 1e-8);
                assert_eq!(route.original_cost, -20. + activity[path].as_f64().unwrap());
            }
            comparisons += 1;
        }
    }
    assert_eq!(comparisons, 24);
}

#[cfg(feature = "clp")]
#[test]
fn original_and_pricing_bounds_remain_distinct_with_other_cost_fronts() {
    let mut graph = model();
    let mut definitions = graph.coupled_kernel().unwrap().definitions().to_vec();
    definitions.push(CoupledResource::IntervalCost {
        initial: Window::new(0., 1.).unwrap(),
        bounds: Window::new(0., 1.).unwrap(),
        increments: vec![0.; 4],
        functions: vec![PiecewiseLinearFunction::new(vec![[0., 5.], [1., 5.]]).unwrap(); 4],
    });
    graph.set_coupled_resources(&definitions).unwrap();
    graph.set_coupled_pricing_adjustments(&[(1, -2.)]).unwrap();
    let k = graph.coupled_kernel().unwrap();
    assert_eq!(k.replay(&[0, 1]).unwrap(), Some(21.));
    assert_eq!(k.replay_pricing(&[0, 1]).unwrap(), Some(-1.));
    let (valid, bounds) = k
        .certify_pricing_with_budget(&[0, 1], &mut 100_000, None)
        .unwrap();
    assert!(valid && bounds[0] <= -1. && bounds[1] >= -1.);
    assert_eq!(k.certify(&[0, 1]), Some((true, [21., 21.])));
    let f = k.replay_state(&[0]).unwrap().unwrap().0;
    let b = k
        .extend_internal(
            &k.root(false),
            1,
            false,
            false,
            false,
            &mut CoupledWorkspace::default(),
        )
        .unwrap()
        .unwrap()
        .0;
    assert_eq!(k.join_objective_costs(&f, &b).unwrap(), Some((-1., 21.)));
}

#[test]
fn raw_cost_forks_reload_changed_activity_adjustments() {
    let make = || {
        let mut solver = crate::pricing::Solver::<2>::new(4, 1).unwrap();
        solver.graph = model();
        for (position, arc) in solver.graph.arcs.iter_mut().enumerate() {
            arc.cost = 0.;
            arc.variables = vec![(position, 1.)];
        }
        solver.cache_variable_costs(&[-20., 0., -20., 0.]).unwrap();
        solver
    };
    let mut reused = make();
    for adjustment in [1., -2., -0.5, 0.] {
        reused
            .graph
            .set_coupled_pricing_adjustments(&[(1, adjustment)])
            .unwrap();
        let actual = reused.price_functor_phase(0, 100).unwrap();
        let mut fresh = make();
        fresh
            .graph
            .set_coupled_pricing_adjustments(&[(1, adjustment)])
            .unwrap();
        let expected = fresh.price_functor_phase(0, 100).unwrap();
        assert_eq!(actual.minimum_reduced_cost, expected.minimum_reduced_cost);
        let routes = |result: crate::PricingResult| {
            result
                .columns
                .into_iter()
                .map(|r| (r.arcs, r.cost, r.original_cost))
                .collect::<Vec<_>>()
        };
        assert_eq!(routes(actual), routes(expected));
    }
}

#[test]
fn multiple_activity_adjustments_keep_nonzero_starts_and_network_pricing_directions() {
    for direction in [0., 1., 2.] {
        let mut solver = crate::pricing::Solver::<2>::new(4, 1).unwrap();
        solver.graph = model();
        let mut definitions = solver
            .graph
            .coupled_kernel()
            .unwrap()
            .definitions()
            .to_vec();
        definitions.push(CoupledResource::CumulativeCost {
            initial: 3.,
            increments: vec![2., 1., 3., 1.],
            weights: vec![2., 2., 1., 1.],
            sink_upper: None,
            source_upper: None,
        });
        solver.graph.set_coupled_resources(&definitions).unwrap();
        solver
            .set_option("BALDESuseBidirectionalSearch", direction)
            .unwrap();
        for (first, second) in [(-2., 0.5), (-0.5, -2.), (0., 0.)] {
            // Setter order does not change component identity or accumulation.
            solver
                .graph
                .set_coupled_pricing_adjustments(&[(2, second), (1, first)])
                .unwrap();
            let expected = [
                -20. + 33. + first * 11. + second * 22.,
                -20. + 15. + first * 2. + second * 13.,
            ];
            let priced = solver.price_phase(0, 100, false).unwrap();
            assert!(priced.exact);
            assert_eq!(priced.minimum_reduced_cost, expected[0].min(expected[1]));
            for route in priced.columns {
                let path = usize::from(route.arcs != [0, 1]);
                assert_eq!(route.cost, expected[path]);
                assert_eq!(route.original_cost, [-20. + 33., -20. + 15.][path]);
            }
        }
    }
}
