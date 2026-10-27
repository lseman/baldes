use super::*;
fn cumulative(increments: Vec<f64>, weights: Vec<f64>) -> CoupledResource {
    CoupledResource::CumulativeCost {
        initial: 0.,
        increments,
        weights,
        sink_upper: None,
        source_upper: None,
    }
}

#[test]
fn piecewise_cost_can_compose_with_unequal_additive_numeric_states() {
    let piecewise = CoupledResource::PiecewiseLinearCost {
        parent: 0,
        functions: vec![crate::graph::coupled::PiecewiseLinearFunction::new(vec![
            [0., 0.],
            [10., 2.],
        ])
        .unwrap()],
    };
    let definitions = [cumulative(vec![1.], vec![1.]), piecewise];
    let order = NumericCostOrder::new(&definitions, &[Some(0), None], 1);
    assert!(order.deploys_unequal());
    assert!(order.preserves(&[1.], &[2.], 0., 2.));
}

#[test]
fn accepted_positive_and_negative_cones_preserve_literal_costs_over_all_small_words() {
    for sign in [-1., 1.] {
        let increments = [sign, 2. * sign];
        let weights = [sign, 3. * sign];
        let order = NumericCostOrder::new(
            &[cumulative(increments.to_vec(), weights.to_vec())],
            &[Some(0)],
            1,
        );
        for smaller in 0..=3 {
            for larger in smaller..=3 {
                let a0 = sign * f64::from(smaller);
                let b0 = sign * f64::from(larger);
                for ac in 0..=3 {
                    for bc in ac..=3 {
                        assert!(order.preserves(&[a0], &[b0], f64::from(ac), f64::from(bc)));
                        for length in 0..=4 {
                            for word in 0..(1usize << length) {
                                let (mut a, mut b) = (a0, b0);
                                let (mut ca, mut cb) = (f64::from(ac), f64::from(bc));
                                // Independent physical recurrence; this
                                // does not call NumericOperation::apply.
                                for step in 0..length {
                                    let arc = (word >> step) & 1;
                                    a += increments[arc];
                                    b += increments[arc];
                                    ca += a * weights[arc];
                                    cb += b * weights[arc];
                                    assert!(ca.is_finite() && cb.is_finite() && ca <= cb);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
#[test]
fn sink_bound_lets_a_larger_state_pay_for_every_short_completion() {
    let (increments, weights) = ([1., 2.], [1., 3.]);
    let bounded = |sink_upper| {
        NumericCostOrder::new(
            &[CoupledResource::CumulativeCost {
                initial: 0.,
                increments: increments.to_vec(),
                weights: weights.to_vec(),
                sink_upper,
                source_upper: None,
            }],
            &[Some(0)],
            1,
        )
    };
    // At most four future arcs, each weighing at most 3.
    let order = bounded(Some(12.));
    assert!(order.preserves(&[3.], &[0.], 0., 36.));
    assert!(!order.preserves(&[3.], &[0.], 0., 35.));
    assert!(!bounded(None).preserves(&[3.], &[0.], 0., 36.));
    let mut accepted = 0;
    for a0 in 0..=3 {
        for b0 in 0..=3 {
            for ac in 0..=40 {
                for bc in ac..=40 {
                    let (a0, b0) = (f64::from(a0), f64::from(b0));
                    if !order.preserves(&[a0], &[b0], f64::from(ac), f64::from(bc)) {
                        continue;
                    }
                    accepted += usize::from(a0 > b0);
                    for length in 0..=4 {
                        for word in 0..(1usize << length) {
                            let (mut a, mut b) = (a0, b0);
                            let (mut ca, mut cb) = (f64::from(ac), f64::from(bc));
                            for step in 0..length {
                                let arc = (word >> step) & 1;
                                a += increments[arc];
                                b += increments[arc];
                                ca += a * weights[arc];
                                cb += b * weights[arc];
                            }
                            assert!(ca <= cb, "{a0} {b0} {ac} {bc} {word}");
                        }
                    }
                }
            }
        }
    }
    assert!(accepted > 0);
}
#[test]
fn signed_shared_charges_require_identical_cost_inputs_and_prefix_costs() {
    let definitions = [
        cumulative(vec![0., 1.], vec![-1., 1.]),
        cumulative(vec![0., 0.], vec![0., 0.]),
    ];
    let order = NumericCostOrder::new(&definitions, &[Some(0), Some(1)], 2);
    assert!(order.preserves(&[2., -7.], &[2., 9.], 3., 3.));
    assert!(!order.preserves(&[1., -7.], &[2., 9.], 3., 3.));
    assert!(!order.preserves(&[2., -7.], &[2., 9.], 2., 3.));
    for length in 0..=5 {
        for word in 0..(1usize << length) {
            let (mut a, mut b) = (2., 2.);
            let (mut ca, mut cb) = (3., 3.);
            for step in 0..length {
                let arc = (word >> step) & 1;
                a += [0., 1.][arc];
                b += [0., 1.][arc];
                ca += a * [-1., 1.][arc];
                cb += b * [-1., 1.][arc];
                assert_eq!(ca, cb);
            }
        }
    }
}
#[test]
fn large_finite_products_and_zero_weight_state_updates_keep_overflow_guards() {
    let order = NumericCostOrder::new(&[cumulative(vec![0.], vec![f64::MAX])], &[Some(0)], 1);
    // The wrong-sign dominator product overflows while B stays finite.
    let a: f64 = -2.;
    let b: f64 = 0.;
    assert!(!(a * f64::MAX).is_finite());
    assert!((b * f64::MAX).is_finite());
    assert!(!order.preserves(&[a], &[b], 0., 0.));
    assert!(order.preserves(&[0.], &[0.5], 0., 0.));
    let zero_weight = NumericCostOrder::new(&[cumulative(vec![f64::MAX], vec![0.])], &[Some(0)], 1);
    let a = f64::MAX;
    assert!(!(a + f64::MAX).is_finite());
    assert!((0. + f64::MAX).is_finite());
    assert!(!zero_weight.preserves(&[a], &[0.], 0., 0.));
    assert!(zero_weight.preserves(&[0.], &[a], 0., 0.));
}
#[test]
fn signed_zero_and_reachable_nonnegative_prefix_policy_are_explicit() {
    let order = NumericCostOrder::new(&[cumulative(vec![-0., 1.], vec![1., 2.])], &[Some(0)], 1);
    assert!(order.preserves(&[-0.], &[0.], -0., 0.));
    assert!(order.preserves(&[0.], &[f64::from_bits(1)], 0., f64::from_bits(1)));
    assert!(!order.preserves(&[0.], &[1.], -1., 0.));
    let signed = NumericCostOrder::new(&[cumulative(vec![0.], vec![-1.])], &[Some(0)], 1);
    // Prefix equality prevents a shared negative charge from overflowing
    // only the putative dominator's already very negative cost.
    assert!(!signed.preserves(&[2.], &[2.], -f64::MAX, 0.));
}

fn dag_order(increments: &[f64], weights: &[f64], cyclic: bool) -> NumericCostOrder {
    let mut graph = Graph::<2>::new(6, 0).unwrap();
    for tail in 0..5 {
        graph.add_arc(tail, tail + 1, 0.).unwrap();
    }
    if cyclic {
        graph.add_arc(4, 1, 0.).unwrap();
    }
    let definitions = [cumulative(increments.to_vec(), weights.to_vec())];
    NumericCostOrder::new(&definitions, &[Some(0)], 1).with_horizon(
        crate::graph::coupled::kernel::bounds::completion_hops(&graph),
        &definitions,
    )
}

#[test]
fn dag_signed_order_preserves_every_short_literal_continuation() {
    for weights in [[1., 2.], [-1., -2.], [-1., 2.]] {
        let increments = [-2., 1.];
        let order = dag_order(&increments, &weights, false);
        let mut accepted = 0;
        for a in -3..=3 {
            for b in -3..=3 {
                for ac in -6..=0 {
                    for bc in ac..=1 {
                        if !order.preserves(
                            &[f64::from(a)],
                            &[f64::from(b)],
                            f64::from(ac),
                            f64::from(bc),
                        ) {
                            continue;
                        }
                        accepted += 1;
                        for length in 0..=5 {
                            for word in 0..(1usize << length) {
                                let (mut a, mut b, mut ca, mut cb) =
                                    (f64::from(a), f64::from(b), f64::from(ac), f64::from(bc));
                                for step in 0..length {
                                    let arc = (word >> step) & 1;
                                    a += increments[arc];
                                    b += increments[arc];
                                    ca += a * weights[arc];
                                    cb += b * weights[arc];
                                    assert!(ca.is_finite() && cb.is_finite() && ca <= cb);
                                }
                            }
                        }
                    }
                }
            }
        }
        assert!(accepted > 0);
    }
}

#[test]
fn signed_horizon_rejects_cycles_overflow_and_wrong_weight_order() {
    let order = dag_order(&[-2., 1.], &[1., 2.], false);
    assert!(order.preserves(&[-2.], &[1.], -3., -2.));
    assert!(!order.preserves(&[1.], &[-2.], -3., -2.));
    assert!(!dag_order(&[-2., 1.], &[1., 2.], true).preserves(&[-2.], &[1.], -3., -2.));
    assert!(!order.preserves(&[-f64::MAX], &[0.], -3., -2.));
    assert!(!order.preserves(&[-2.], &[1.], -f64::MAX, -2.));
    let mixed = dag_order(&[-2., 1.], &[-1., 2.], false);
    assert!(mixed.preserves(&[1.], &[1.], -3., -2.));
    assert!(!mixed.preserves(&[0.], &[1.], -3., -2.));
    assert!(mixed.preserves(&[0.], &[1.], -100., 0.));
    assert!(!mixed.preserves(&[0.5], &[1.5], -100., 0.));
    assert!(!mixed.preserves(&[0.], &[1.], -100.5, 0.));
    let limit = (1u64 << 53) as f64;
    assert!(!mixed.preserves(&[limit - 1.], &[limit], -100., 0.));
}

#[test]
fn explicit_bounded_walk_horizon_admits_signed_cycles_without_using_ng() {
    let mut graph = Graph::<2>::new(4, 0).unwrap();
    for (tail, head) in [(0, 1), (1, 2), (2, 1), (2, 3)] {
        graph.add_arc(tail, head, 0.).unwrap();
    }
    let definitions = [cumulative(vec![-2., 1., -2., 1.], vec![1., 2., 1., 2.])];
    let prepare = |g: &Graph<2>| {
        NumericCostOrder::new(&definitions, &[Some(0)], 1).with_horizon(
            crate::graph::coupled::kernel::bounds::completion_hops(g),
            &definitions,
        )
    };
    assert!(!prepare(&graph).preserves(&[-2.], &[1.], -3., -2.));
    graph
        .set_route_domain(crate::graph::RouteDomain::BoundedWalks { max_arcs: 4 })
        .unwrap();
    assert!(prepare(&graph).preserves(&[-2.], &[1.], -3., -2.));
    for length in 0..=4 {
        for word in 0..(1usize << length) {
            let (mut a, mut b, mut ca, mut cb) = (-2., 1., -3., -2.);
            for step in 0..length {
                let (inc, weight) = if (word >> step) & 1 == 0 {
                    (-2., 1.)
                } else {
                    (1., 2.)
                };
                a += inc;
                b += inc;
                ca += a * weight;
                cb += b * weight;
                assert!(ca <= cb);
            }
        }
    }
    // A larger cap can destroy the finite-operation envelope. It is not
    // replaced with V-1 just because the graph is small.
    graph
        .set_route_domain(crate::graph::RouteDomain::BoundedWalks {
            max_arcs: usize::MAX,
        })
        .unwrap();
    assert!(!prepare(&graph).preserves(&[-2.], &[1.], -3., -2.));
}

#[test]
fn changing_route_cap_refreshes_original_and_pricing_horizons_without_resetting_costs() {
    let mut graph = Graph::<2>::new(4, 0).unwrap();
    for (tail, head) in [(0, 1), (1, 2), (2, 1), (2, 3)] {
        graph.add_arc(tail, head, 0.).unwrap();
    }
    let bounded = |max_arcs| crate::graph::RouteDomain::BoundedWalks { max_arcs };
    graph.set_route_domain(bounded(4)).unwrap();
    graph
        .set_coupled_resources(&[cumulative(vec![-2., 1., -2., 1.], vec![1., 2., 1., 2.])])
        .unwrap();
    graph
        .set_coupled_objective_coefficients(&[(0, 2.)])
        .unwrap();
    graph.set_coupled_pricing_adjustments(&[(0, -0.5)]).unwrap();
    let previous = graph.coupled_kernel().unwrap().cache_identity().clone();
    for cap in [6, usize::MAX, 4] {
        graph.set_route_domain(bounded(cap)).unwrap();
        let kernel = graph.coupled_kernel().unwrap();
        assert_eq!(kernel.optimization.continuation_hops, Some(cap));
        let fresh = CoupledKernel::prepare(&graph, kernel.definitions()).unwrap();
        assert_eq!(kernel.route_domains, fresh.route_domains);
        for pricing in kernel.pricing_activities.iter() {
            assert_eq!(pricing.optimization.continuation_hops, Some(cap));
            let fresh = CoupledKernel::prepare(&graph, pricing.definitions()).unwrap();
            assert_eq!(pricing.route_domains, fresh.route_domains);
        }
        assert_eq!(kernel.objective_coefficient(0), Some(2.));
        assert_eq!(kernel.replay(&[0, 1, 3]).unwrap(), Some(-8.));
        assert_eq!(kernel.replay_pricing(&[0, 1, 3]).unwrap(), Some(-6.));
        assert!(!std::sync::Arc::ptr_eq(&previous, kernel.cache_identity()));
    }
    let current = graph.coupled_kernel().unwrap().cache_identity().clone();
    graph.set_route_domain(bounded(4)).unwrap();
    assert!(std::sync::Arc::ptr_eq(
        &current,
        graph.coupled_kernel().unwrap().cache_identity()
    ));
    assert!(graph.set_route_domain(bounded(0)).is_err());
    assert_eq!(graph.max_route_arcs(), Some(4));
    assert!(std::sync::Arc::ptr_eq(
        &current,
        graph.coupled_kernel().unwrap().cache_identity()
    ));
    assert_eq!(
        graph
            .coupled_kernel()
            .unwrap()
            .replay_pricing(&[0, 1, 3])
            .unwrap(),
        Some(-6.)
    );
}

#[test]
fn backward_vertex_entry_keeps_a_real_negative_prefix_counterexample() {
    for negative_prefix in [false, true] {
        let mut graph = Graph::<2>::new(3, 0).unwrap();
        for (tail, head) in [(0, 1), (1, 2), (1, 2)] {
            graph.add_arc(tail, head, 0.).unwrap();
        }
        let mut increments = vec![3., 0., 0.];
        let mut weights = vec![0., -0.5, 0.];
        if negative_prefix {
            graph.add_arc(0, 1, 0.).unwrap();
            increments.push(-3.);
            weights.push(0.);
        }
        graph
            .set_coupled_resources(&[cumulative(increments, weights)])
            .unwrap();
        let kernel = graph.coupled_kernel().unwrap();
        let root = kernel.root(false);
        let (a, ac) = kernel.extend(&root, 1, false).unwrap().unwrap();
        let (b, bc) = kernel.extend(&root, 2, false).unwrap().unwrap();
        let penalty = graph
            .coupled_dominance_penalty(Some(&a), Some(&b), false, 1)
            .unwrap();
        assert_eq!(
            penalty,
            kernel
                .dominance_penalty_at_vertex(&a, &b, false, 1)
                .unwrap()
        );
        assert_eq!(ac + penalty <= bc, !negative_prefix);
        for arc in [0, if negative_prefix { 3 } else { 0 }] {
            let (prefix, _) = kernel
                .extend(&kernel.root(true), arc, true)
                .unwrap()
                .unwrap();
            let ca = kernel.join(&prefix, &a).unwrap().unwrap();
            let cb = kernel.join(&prefix, &b).unwrap().unwrap();
            if arc == 0 {
                assert!(ca <= cb);
            } else {
                assert!(ca > cb);
            }
        }
    }
}

#[test]
fn vertex_summary_excludes_prefix_weights_but_retains_real_negative_continuations() {
    for negative_future in [false, true] {
        let mut graph = Graph::<2>::new(3, 0).unwrap();
        for (tail, head) in [(0, 1), (0, 1), (1, 2)] {
            graph.add_arc(tail, head, 0.).unwrap();
        }
        let mut increments = vec![-0.25, 0.5, 0.];
        let mut weights = vec![-1., 0.5, 1.];
        if negative_future {
            graph.add_arc(1, 2, 0.).unwrap();
            increments.push(0.);
            weights.push(-1.);
        }
        graph
            .set_coupled_resources(&[cumulative(increments, weights)])
            .unwrap();
        let kernel = graph.coupled_kernel().unwrap();
        let root = kernel.root(true);
        let (a, ac) = kernel.extend(&root, 0, true).unwrap().unwrap();
        let (b, bc) = kernel.extend(&root, 1, true).unwrap().unwrap();
        assert_eq!((ac, bc), (0.25, 0.25));
        assert_eq!(kernel.dominance_penalty(&a, &b, true), None);
        let expected = if negative_future { None } else { Some(0.) };
        assert_eq!(
            kernel.dominance_penalty_at_vertex(&a, &b, true, 1),
            expected
        );
        assert_eq!(
            graph.coupled_dominance_penalty(Some(&a), Some(&b), true, 1),
            expected
        );
        assert_eq!(kernel.dominance_penalty_at_vertex(&a, &b, true, 0), None);
        assert_eq!(kernel.dominance_penalty_at_vertex(&a, &b, true, 99), None);
        for arc in 2..graph.arcs.len() {
            let (_, ca) = kernel.extend(&a, arc, true).unwrap().unwrap();
            let (_, cb) = kernel.extend(&b, arc, true).unwrap().unwrap();
            if arc == 2 {
                assert!(ac + ca <= bc + cb);
            } else {
                assert!(ac + ca > bc + cb);
            }
        }
    }
}
