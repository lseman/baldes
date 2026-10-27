use super::*;
use crate::graph::coupled::PiecewiseLinearFunction;

#[test]
fn unbounded_relation_shift_pays_for_a_higher_nonconstant_front() {
    use crate::graph::coupled::{PairedResourceArc, PairedResourceMode};
    let line = |a, b| PiecewiseLinearFunction::new(vec![[0., a], [2., b]]).unwrap();
    let mut graph = Graph::<2>::new(2, 0).unwrap();
    graph.add_arc(0, 1, 0.).unwrap();
    graph.add_arc(0, 1, 0.).unwrap();
    let arc = |increment| PairedResourceArc {
        arrival: line(0., 2.),
        increment,
        lower: line(0., 0.),
        upper: line(1e12, 1e12),
        waiting: 1,
    };
    graph
        .set_coupled_resources(&[CoupledResource::PairedPiecewiseLinear {
            normalization: crate::graph::coupled::PairedCostNormalization::FinalMinusInitialMinimum,
            initial: line(0., 0.),
            initial_base: Window::new(0., 2.).unwrap(),
            arcs: vec![arc(line(0., 2.)), arc(line(0., 0.))],
            mode: PairedResourceMode::CostUnbounded,
        }])
        .unwrap();
    let kernel = graph.coupled_kernel().unwrap();
    let root = kernel.root(true);
    let mut high = kernel.extend(&root, 0, true).unwrap().unwrap().0;
    let mut low = kernel.extend(&root, 1, true).unwrap().unwrap().0;
    high.account_extension(&root, 0., 0.).unwrap();
    low.account_extension(&root, 0., 0.).unwrap();
    let penalty = kernel.dominance_penalty(&high, &low, true).unwrap();
    assert!((2. ..=2. + 1e-12).contains(&penalty), "{penalty}");
    assert_eq!(kernel.dominance_penalty(&low, &high, true), Some(0.));
}

#[test]
fn backward_paired_fronts_compare_different_suffix_programs() {
    use crate::graph::coupled::{PairedResourceArc, PairedResourceMode};
    let line = |a, b| PiecewiseLinearFunction::new(vec![[0., a], [2., b]]).unwrap();
    let mut graph = Graph::<2>::new(2, 0).unwrap();
    graph.add_arc(0, 1, 0.).unwrap();
    graph.add_arc(0, 1, 0.).unwrap();
    let arc = |upper| PairedResourceArc {
        arrival: line(0., 2.),
        increment: line(1., 1.),
        lower: line(0., 0.),
        upper: line(upper, upper),
        waiting: 0,
    };
    graph
        .set_coupled_resources(&[CoupledResource::PairedPiecewiseLinear {
            normalization: crate::graph::coupled::PairedCostNormalization::FinalMinusInitialMinimum,
            initial: line(5., 5.),
            initial_base: Window::new(0., 2.).unwrap(),
            arcs: vec![arc(10.), arc(8.)],
            mode: PairedResourceMode::Feasibility,
        }])
        .unwrap();
    let kernel = graph.coupled_kernel().unwrap();
    let root = kernel.root(false);
    let wide = kernel.extend(&root, 0, false).unwrap().unwrap().0;
    let narrow = kernel.extend(&root, 1, false).unwrap().unwrap().0;
    assert_ne!(
        wide.suffix.as_ref().unwrap().program,
        narrow.suffix.as_ref().unwrap().program
    );
    assert_eq!(kernel.dominance_penalty(&wide, &narrow, false), Some(0.));
    assert_eq!(kernel.dominance_penalty(&narrow, &wide, false), None);
}

#[test]
fn paired_terminal_above_numerical_infinity_retains_suffix_replay() {
    use crate::graph::coupled::{PairedResourceArc, PairedResourceMode};
    let line = |value| PiecewiseLinearFunction::new(vec![[0., value], [2., value]]).unwrap();
    let mut graph = Graph::<2>::new(2, 0).unwrap();
    graph.add_arc(0, 1, 0.).unwrap();
    graph
        .set_coupled_resources(&[CoupledResource::PairedPiecewiseLinear {
            normalization: crate::graph::coupled::PairedCostNormalization::FinalMinusInitialMinimum,
            initial: line(1.5e12),
            initial_base: Window::new(0., 2.).unwrap(),
            arcs: vec![PairedResourceArc {
                arrival: PiecewiseLinearFunction::new(vec![[0., 0.], [2., 2.]]).unwrap(),
                increment: line(1.),
                lower: line(0.),
                upper: line(2e12),
                waiting: 0,
            }],
            mode: PairedResourceMode::Feasibility,
        }])
        .unwrap();
    let kernel = graph.coupled_kernel().unwrap();
    let suffix = kernel
        .extend(&kernel.root(false), 0, false)
        .unwrap()
        .unwrap()
        .0;
    assert!(suffix.relations[0].backward_upper().is_none());
    assert!(kernel.join(&kernel.root(true), &suffix).unwrap().is_some());
}

#[test]
fn rounded_future_charge_cannot_prune_the_better_physical_route() {
    let large = 9_007_199_254_740_994.;
    let mut graph = Graph::<2>::new(4, 0).unwrap();
    for (tail, head, cost) in [(0, 1, -1.), (0, 1, 0.), (1, 2, -large + 2.), (2, 3, -10.)] {
        graph.add_arc(tail, head, cost).unwrap();
    }
    let function = |points| PiecewiseLinearFunction::new(points).unwrap();
    graph
        .set_coupled_resources(&[CoupledResource::IntervalCost {
            initial: Window::new(0., 1.).unwrap(),
            bounds: Window::new(0., 1.).unwrap(),
            increments: vec![0.; 4],
            functions: vec![
                function(vec![[0., 0.], [1., 1.]]),
                function(vec![[0., 0.], [1., 0.]]),
                function(vec![[1., large], [2., large]]),
                function(vec![[0., 0.], [1., 0.]]),
            ],
        }])
        .unwrap();
    let kernel = graph.coupled_kernel().unwrap();
    let root = kernel.root(true);
    let (a, _) = kernel.extend(&root, 0, true).unwrap().unwrap();
    let (b, _) = kernel.extend(&root, 1, true).unwrap().unwrap();
    // A snapshot difference of one grows to two after common addition.
    assert_eq!(kernel.dominance_penalty(&a, &b, true), None);
    let (_, ad) = kernel.extend(&a, 2, true).unwrap().unwrap();
    let (_, bd) = kernel.extend(&b, 2, true).unwrap().unwrap();
    assert_eq!(ad - bd, 2.);
    // The scalar sums are independently evaluated in physical arc order.
    // Resource values below are literal endpoint sums, not front replay.
    let scalar_a = ((0. - 1.) + (-large + 2.)) - 10.;
    let scalar_b = ((0. + 0.) + (-large + 2.)) - 10.;
    assert_eq!(scalar_a + (1. + large), -6.);
    assert_eq!(scalar_b + (0. + large), -8.);
    for direction in [0., 1., 2.] {
        let mut solver = crate::pricing::Solver::<2>::new(4, 0).unwrap();
        solver.graph = graph.clone();
        solver
            .set_option("BALDESuseBidirectionalSearch", direction)
            .unwrap();
        let result = solver.price_phase(0, 100, false).unwrap();
        assert_eq!(result.minimum_reduced_cost, -8.);
        assert!(result
            .columns
            .iter()
            .any(|r| r.arcs == [1, 2, 3] && r.original_cost == -8.));
    }
}

#[test]
fn frozen_future_does_not_repair_an_already_rounded_prefix_key() {
    let large = 9_007_199_254_740_994.;
    let mut graph = Graph::<2>::new(4, 0).unwrap();
    for (tail, head, cost) in [(0, 1, -large), (1, 2, 0.), (0, 2, 3.), (2, 3, -10.)] {
        graph.add_arc(tail, head, cost).unwrap();
    }
    let constant = |value| PiecewiseLinearFunction::new(vec![[0., value], [1., value]]).unwrap();
    graph
        .set_coupled_resources(&[CoupledResource::IntervalCost {
            initial: Window::new(0., 1.).unwrap(),
            bounds: Window::new(0., 1.).unwrap(),
            increments: vec![0.; 4],
            functions: vec![constant(1.), constant(large), constant(0.), constant(0.)],
        }])
        .unwrap();
    // Prefix A's interleaved scalar/min-delta key is four, but its
    // physical scalar sum plus final resource minimum is two. B's key
    // and canonical cost are three. Zero future charges cannot repair this
    // discrepancy before choosing which complete route to retain.
    assert_eq!((-large + 1.) + ((1. + large) - 1.), 4.);
    assert_eq!(-large + (1. + large), 2.);
    for direction in [0., 1., 2.] {
        let mut solver = crate::pricing::Solver::<2>::new(4, 0).unwrap();
        solver.graph = graph.clone();
        solver
            .set_option("BALDESuseBidirectionalSearch", direction)
            .unwrap();
        let result = solver.price_phase(0, 100, false).unwrap();
        assert_eq!(result.minimum_reduced_cost, -8.);
        assert!(result
            .columns
            .iter()
            .any(|r| r.arcs == [0, 1, 3] && r.original_cost == -8.));
    }
}

#[test]
fn equal_final_fronts_do_not_hide_different_physical_scalar_prefixes() {
    let large = 9_007_199_254_740_994.;
    let mut graph = Graph::<2>::new(5, 0).unwrap();
    for (tail, head, cost) in [
        (0, 1, -large),
        (1, 3, 0.),
        (0, 2, -(large + 2.)),
        (2, 3, 3.),
        (3, 4, -10.),
    ] {
        graph.add_arc(tail, head, cost).unwrap();
    }
    let constant = |value| PiecewiseLinearFunction::new(vec![[0., value], [1., value]]).unwrap();
    graph
        .set_coupled_resources(&[CoupledResource::IntervalCost {
            initial: Window::new(0., 1.).unwrap(),
            bounds: Window::new(0., 1.).unwrap(),
            increments: vec![0.; 5],
            functions: vec![
                constant(1.),
                constant(large),
                constant(large + 2.),
                constant(0.),
                constant(0.),
            ],
        }])
        .unwrap();
    // Both final resource functions are the same constant large+2.
    // Independent physical scalar sums distinguish the complete costs.
    assert_eq!((-large - 10.) + (1. + large), -8.);
    assert_eq!(((-(large + 2.) + 3.) - 10.) + (large + 2.), -6.);
    for direction in [0., 1., 2.] {
        let mut solver = crate::pricing::Solver::<2>::new(5, 0).unwrap();
        solver.graph = graph.clone();
        solver
            .set_option("BALDESuseBidirectionalSearch", direction)
            .unwrap();
        let result = solver.price_phase(0, 100, false).unwrap();
        assert_eq!(result.minimum_reduced_cost, -8.);
        assert!(result
            .columns
            .iter()
            .any(|r| r.arcs == [0, 1, 4] && r.original_cost == -8.));
    }
}

#[test]
fn numeric_only_piecewise_costs_preserve_independent_scalar_components() {
    let large = 9_007_199_254_740_994.;
    let mut graph = Graph::<2>::new(4, 0).unwrap();
    for (tail, head, cost) in [(0, 1, -1.), (0, 1, 0.), (1, 2, -large + 2.), (2, 3, -10.)] {
        graph.add_arc(tail, head, cost).unwrap();
    }
    let constant = |value| PiecewiseLinearFunction::new(vec![[-1., value], [1., value]]).unwrap();
    graph
        .set_coupled_resources(&[
            CoupledResource::CumulativeCost {
                initial: 0.,
                increments: vec![0.; 4],
                weights: vec![0.; 4],
                sink_upper: None,
                source_upper: None,
            },
            CoupledResource::PiecewiseLinearCost {
                parent: 0,
                functions: vec![constant(1.), constant(0.), constant(large), constant(0.)],
            },
        ])
        .unwrap();
    let kernel = graph.coupled_kernel().unwrap();
    let root = kernel.root(true);
    let (a, _) = kernel.extend(&root, 0, true).unwrap().unwrap();
    let (b, _) = kernel.extend(&root, 1, true).unwrap().unwrap();
    assert_eq!(a.numeric, b.numeric);
    assert!(!CoupledKernel::dominates(Some(&a), Some(&b)));
    assert_eq!(kernel.dominance_penalty(&a, &b, true), None);
    // Independent scalar sum plus literal numeric charge accumulation.
    let scalar_a = ((0. - 1.) + (-large + 2.)) - 10.;
    let scalar_b = (0. + (-large + 2.)) - 10.;
    assert_eq!(scalar_a + (1. + large), -6.);
    assert_eq!(scalar_b + large, -8.);
    for direction in [0., 1., 2.] {
        let mut solver = crate::pricing::Solver::<2>::new(4, 0).unwrap();
        solver.graph = graph.clone();
        solver
            .set_option("BALDESuseBidirectionalSearch", direction)
            .unwrap();
        let result = solver.price_phase(0, 100, false).unwrap();
        assert_eq!(result.minimum_reduced_cost, -8.);
        assert!(result
            .columns
            .iter()
            .any(|r| r.arcs == [1, 2, 3] && r.original_cost == -8.));
    }
}

#[test]
fn cumulative_source_bound_prices_backward_weight_differences() {
    let prepare = |source_upper| {
        let mut graph = Graph::<2>::new(3, 0).unwrap();
        for (tail, head) in [(0, 1), (1, 2), (1, 2), (2, 1)] {
            graph.add_arc(tail, head, 0.).unwrap();
        }
        graph
            .set_coupled_resources(&[CoupledResource::CumulativeCost {
                initial: 0.,
                // Zero suffix increments isolate the entry-state correction from
                // the independent accumulated-cost ordering precondition.
                increments: vec![1., 0., 0., 0.],
                weights: vec![0., 1., 3., 0.],
                sink_upper: None,
                source_upper,
            }])
            .unwrap();
        graph
    };
    let suffixes = |graph: &Graph<2>| {
        let kernel = graph.coupled_kernel().unwrap();
        let root = kernel.root(false);
        let light = kernel.extend(&root, 1, false).unwrap().unwrap().0;
        let heavy = kernel.extend(&root, 2, false).unwrap().unwrap().0;
        (kernel.clone(), light, heavy)
    };
    let (kernel, light, heavy) = suffixes(&prepare(None));
    assert_eq!(kernel.dominance_penalty(&heavy, &light, false), None);
    let (kernel, light, heavy) = suffixes(&prepare(Some(5.)));
    // Entry states lie in [0, 5]: the heavier suffix costs at most
    // (3 - 1) * 5 more, plus the dot-product rounding allowance.
    let penalty = kernel.dominance_penalty(&heavy, &light, false).unwrap();
    assert!((10. ..=10. + 1e-9).contains(&penalty), "{penalty}");
    let penalty = kernel.dominance_penalty(&light, &heavy, false).unwrap();
    assert!((0. ..=1e-9).contains(&penalty), "{penalty}");
    let mut graph = prepare(Some(5.));
    let mut definitions = graph.coupled_kernel().unwrap().definitions().to_vec();
    let CoupledResource::CumulativeCost { increments, .. } = &mut definitions[0] else {
        unreachable!()
    };
    increments.fill(1.);
    graph.set_coupled_resources(&definitions).unwrap();
    let (kernel, light, heavy) = suffixes(&graph);
    assert!(heavy.numeric_cost > light.numeric_cost);
    assert_eq!(kernel.dominance_penalty(&heavy, &light, false), None);
}

#[test]
fn nonnegative_numeric_continuations_restore_zero_compensation_state_dominance() {
    let mut graph = Graph::<2>::new(3, 0).unwrap();
    for (tail, head) in [(0, 1), (0, 1), (1, 2)] {
        graph.add_arc(tail, head, -20.).unwrap();
    }
    graph
        .set_coupled_resources(&[CoupledResource::CumulativeCost {
            initial: 0.,
            increments: vec![1., 2., 1.],
            weights: vec![1., 1., 3.],
            sink_upper: None,
            source_upper: None,
        }])
        .unwrap();
    let kernel = graph.coupled_kernel().unwrap();
    let root = kernel.root(true);
    let (a, _) = kernel.extend(&root, 0, true).unwrap().unwrap();
    let (b, _) = kernel.extend(&root, 1, true).unwrap().unwrap();
    assert_ne!(a.numeric, b.numeric);
    assert_eq!(kernel.dominance_penalty(&a, &b, true), Some(0.));
    assert_eq!(kernel.dominance_penalty(&b, &a, true), None);
    // Independent resource recurrences: route A pays 1 + 2*3 = 7;
    // route B pays 2 + 3*3 = 11. Physical scalar sums are both -40.
    for direction in [0., 1., 2.] {
        let mut solver = crate::pricing::Solver::<2>::new(3, 0).unwrap();
        solver.graph = graph.clone();
        solver
            .set_option("BALDESuseBidirectionalSearch", direction)
            .unwrap();
        let result = solver.price_phase(0, 100, false).unwrap();
        assert_eq!(result.minimum_reduced_cost, -33.);
        assert!(result
            .columns
            .iter()
            .any(|r| r.arcs == [0, 2] && r.original_cost == -33.));
    }
}

#[test]
fn identical_numeric_inputs_keep_finite_signed_and_nonnegative_prefix_order() {
    for [a_charge, b_charge, future] in [[1., 2., 3.], [-3., -1., -2.]] {
        let mut graph = Graph::<2>::new(3, 0).unwrap();
        for (tail, head) in [(0, 1), (0, 1), (1, 2)] {
            graph.add_arc(tail, head, 0.).unwrap();
        }
        let constant =
            |value| PiecewiseLinearFunction::new(vec![[-1., value], [1., value]]).unwrap();
        graph
            .set_coupled_resources(&[
                CoupledResource::CumulativeCost {
                    initial: 0.,
                    increments: vec![0.; 3],
                    weights: vec![0.; 3],
                    sink_upper: None,
                    source_upper: None,
                },
                CoupledResource::PiecewiseLinearCost {
                    parent: 0,
                    functions: vec![constant(a_charge), constant(b_charge), constant(future)],
                },
            ])
            .unwrap();
        let kernel = graph.coupled_kernel().unwrap();
        let root = kernel.root(true);
        let (a, _) = kernel.extend(&root, 0, true).unwrap().unwrap();
        let (b, _) = kernel.extend(&root, 1, true).unwrap().unwrap();
        assert_eq!(a.numeric, b.numeric);
        assert_eq!(kernel.dominance_penalty(&a, &b, true), Some(0.));
        assert_eq!(kernel.dominance_penalty(&b, &a, true), None);
        // Independent literal objective recurrence on each physical path.
        assert_eq!(kernel.replay(&[0, 2]).unwrap(), Some(a_charge + future));
        assert_eq!(kernel.replay(&[1, 2]).unwrap(), Some(b_charge + future));
    }
}

#[test]
fn identical_numeric_signed_costs_preserve_baseline_and_overflow_is_global_unknown() {
    let maximum = f64::MAX;
    let mut graph = Graph::<2>::new(3, 0).unwrap();
    for (tail, head, cost) in [(0, 1, -1.), (0, 1, 0.), (1, 2, 0.)] {
        graph.add_arc(tail, head, cost).unwrap();
    }
    let constant = |value| PiecewiseLinearFunction::new(vec![[-1., value], [1., value]]).unwrap();
    graph
        .set_coupled_resources(&[
            CoupledResource::CumulativeCost {
                initial: 0.,
                increments: vec![0.; 3],
                weights: vec![0.; 3],
                sink_upper: None,
                source_upper: None,
            },
            CoupledResource::PiecewiseLinearCost {
                parent: 0,
                functions: vec![
                    constant(-maximum),
                    constant(-maximum / 2.),
                    constant(-maximum / 2.),
                ],
            },
        ])
        .unwrap();
    let kernel = graph.coupled_kernel().unwrap();
    let root = kernel.root(true);
    let (a, _) = kernel.extend(&root, 0, true).unwrap().unwrap();
    let (b, _) = kernel.extend(&root, 1, true).unwrap().unwrap();
    assert_eq!(a.numeric, b.numeric);
    assert!(a.numeric_cost < b.numeric_cost);
    assert_eq!(kernel.dominance_penalty(&a, &b, true), Some(0.));
    assert!(kernel.extend(&a, 2, true).is_err());
    assert_eq!(kernel.replay(&[1, 2]).unwrap(), Some(-maximum));
    for direction in [0., 1., 2.] {
        let mut solver = crate::pricing::Solver::<2>::new(3, 0).unwrap();
        solver.graph = graph.clone();
        solver
            .set_option("BALDESuseBidirectionalSearch", direction)
            .unwrap();
        if let Ok(result) = solver.price_phase(0, 100, false) {
            // A scheduler that never explores the unsupported branch can
            // return the valid finite best route. It cannot silently
            // turn overflow into empty/complete infeasibility evidence.
            assert_eq!(result.minimum_reduced_cost, -maximum);
            assert!(!result.columns.is_empty());
            assert!(result
                .columns
                .iter()
                .all(|r| r.arcs == [1, 2] && r.original_cost == -maximum));
        }
    }
}

#[test]
fn complex_numeric_programs_keep_baseline_identity_activation_policy() {
    let mut graph = Graph::<2>::new(3, 0).unwrap();
    for (tail, head) in [(0, 1), (0, 1), (1, 2)] {
        graph.add_arc(tail, head, -20.).unwrap();
    }
    graph
        .set_coupled_resources(&[
            CoupledResource::CumulativeCost {
                initial: 0.,
                increments: vec![1., 2., 1.],
                weights: vec![1., 1., 3.],
                sink_upper: None,
                source_upper: None,
            },
            CoupledResource::Dependent {
                parent: 0,
                initial: 0.,
                bounds: Window::new(0., 100.).unwrap(),
                offsets: vec![0.; 3],
                factors: vec![1.; 3],
            },
        ])
        .unwrap();
    let kernel = graph.coupled_kernel().unwrap();
    let root = kernel.root(true);
    let (a, _) = kernel.extend(&root, 0, true).unwrap().unwrap();
    let (b, _) = kernel.extend(&root, 1, true).unwrap().unwrap();
    // The proof can establish the arithmetic order; deployment keeps
    // richer programs on their measured baseline identity path.
    assert!(kernel.numeric_cost_order.preserves(
        &a.numeric,
        &b.numeric,
        a.numeric_cost,
        b.numeric_cost
    ));
    assert!(!kernel.numeric_cost_order.deploys_unequal());
    assert_eq!(kernel.dominance_penalty(&a, &b, true), None);
    assert_eq!(kernel.dominance_penalty(&a, &a, true), Some(0.));
}

#[test]
fn encoded_cost_with_equal_parent_allows_unrelated_uncharged_ordering() {
    let mut graph = Graph::<2>::new(3, 0).unwrap();
    for (tail, head) in [(0, 1), (0, 1), (1, 2)] {
        graph.add_arc(tail, head, 0.).unwrap();
    }
    let signed_charge = crate::graph::coupled::EncodedFunction::from(
        PiecewiseLinearFunction::new(vec![[0., -2.], [10., -1.]]).unwrap(),
    );
    graph
        .set_coupled_resources(&[
            CoupledResource::CumulativeCost {
                initial: 0.,
                increments: vec![0.; 3],
                weights: vec![0.; 3],
                sink_upper: None,
                source_upper: None,
            },
            CoupledResource::EncodedPiecewiseLinearCost {
                parent: 0,
                functions: vec![signed_charge; 3],
            },
            CoupledResource::CustomResource {
                increments: vec![0., 1., 0.],
                weights: vec![0.; 3],
                forward_bound: None,
                backward_bound: None,
            },
        ])
        .unwrap();

    let kernel = graph.coupled_kernel().unwrap();
    let root = kernel.root(true);
    let (lower, _) = kernel.extend(&root, 0, true).unwrap().unwrap();
    let (higher, _) = kernel.extend(&root, 1, true).unwrap().unwrap();
    assert_eq!(lower.numeric_cost, higher.numeric_cost);
    assert_eq!(kernel.dominance_penalty(&lower, &higher, true), Some(0.));
    assert_eq!(kernel.dominance_penalty(&higher, &lower, true), None);
}

#[test]
fn permanently_zero_fronts_keep_domain_inclusion_dominance() {
    let mut graph = Graph::<2>::new(3, 0).unwrap();
    for (tail, head) in [(0, 1), (0, 1), (1, 2)] {
        graph.add_arc(tail, head, 0.).unwrap();
    }
    let zero = |lo, hi| PiecewiseLinearFunction::new(vec![[lo, 0.], [hi, 0.]]).unwrap();
    graph
        .set_coupled_resources(&[CoupledResource::IntervalCost {
            initial: Window::new(0., 4.).unwrap(),
            bounds: Window::new(0., 4.).unwrap(),
            increments: vec![0.; 3],
            functions: vec![zero(0., 4.), zero(1., 3.), zero(0., 4.)],
        }])
        .unwrap();
    let kernel = graph.coupled_kernel().unwrap();
    let root = kernel.root(true);
    let (a, _) = kernel.extend(&root, 0, true).unwrap().unwrap();
    let (b, _) = kernel.extend(&root, 1, true).unwrap().unwrap();
    assert_eq!(kernel.dominance_penalty(&a, &b, true), Some(0.));
    assert_eq!(kernel.dominance_penalty(&b, &a, true), None);
}

#[test]
fn signed_paired_forward_dominance_requires_both_envelopes_and_equal_accounting() {
    use crate::graph::coupled::{PairedResourceArc, PairedResourceMode};
    let line = |a, b| PiecewiseLinearFunction::new(vec![[0., a], [2., b]]).unwrap();
    let mut graph = Graph::<2>::new(2, 0).unwrap();
    graph.add_arc(0, 1, 0.).unwrap();
    graph.add_arc(0, 1, 0.).unwrap();
    let arc = |increment| PairedResourceArc {
        arrival: line(0., 2.),
        increment: line(increment, increment),
        lower: line(0., 0.),
        upper: line(10., 10.),
        waiting: 0,
    };
    graph
        .set_coupled_resources(&[CoupledResource::PairedPiecewiseLinear {
            normalization: crate::graph::coupled::PairedCostNormalization::FinalMinusInitialMinimum,
            initial: line(0., 0.),
            initial_base: Window::new(0., 2.).unwrap(),
            arcs: vec![arc(0.), arc(1.)],
            mode: PairedResourceMode::CostUnbounded,
        }])
        .unwrap();
    graph
        .set_coupled_objective_coefficients(&[(0, -1.)])
        .unwrap();
    let kernel = graph.coupled_kernel().unwrap();
    let root = kernel.root(true);
    let mut low = kernel.extend(&root, 0, true).unwrap().unwrap().0;
    let mut high = kernel.extend(&root, 1, true).unwrap().unwrap().0;
    low.account_extension(&root, 0., 0.).unwrap();
    high.account_extension(&root, 0., 0.).unwrap();
    assert_eq!(kernel.dominance_penalty(&low, &high, true), Some(0.));
    assert_eq!(kernel.dominance_penalty(&high, &low, true), None);
    let mut unknown = low.clone();
    if let super::super::paired::PairedLabel::Forward { upper_negative, .. } =
        &mut unknown.relations[0]
    {
        *upper_negative = None;
    }
    assert_eq!(kernel.dominance_penalty(&unknown, &high, true), None);
    low.accounting = None;
    assert_eq!(kernel.dominance_penalty(&low, &high, true), None);
}
