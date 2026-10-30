//! Direct physical-graph coupled-resource regressions; no expanded graph oracle.
use baldes::graph::coupled::{CoupledResource, PiecewiseLinearFunction};
use baldes::{Graph, Window};

#[test]
fn whole_interval_front_dominance_declines_unproved_continuation_compensation() {
    let mut graph = Graph::<2>::new(3, 0).unwrap();
    for (tail, head) in [(0, 1), (0, 1), (1, 2)] {
        graph.add_arc(tail, head, 0.).unwrap();
    }
    let pwl = |ys: [f64; 2]| PiecewiseLinearFunction::new(vec![[0., ys[0]], [4., ys[1]]]).unwrap();
    graph
        .set_coupled_resources(&[CoupledResource::IntervalCost {
            initial: Window::new(0., 4.).unwrap(),
            bounds: Window::new(0., 4.).unwrap(),
            increments: vec![0.; 3],
            functions: vec![pwl([8., 8.]), pwl([-2., 2.]), pwl([4., -4.])],
        }])
        .unwrap();
    let kernel = graph.coupled_kernel().unwrap();
    for forward in [true, false] {
        let root = kernel.root(forward);
        let (constant, _) = kernel.extend(&root, 0, forward).unwrap().unwrap();
        let (rising, _) = kernel.extend(&root, 1, forward).unwrap().unwrap();
        assert_eq!(kernel.dominance_penalty(&constant, &rising, forward), None);
        assert_eq!(kernel.dominance_penalty(&rising, &constant, forward), None);
        assert_eq!(kernel.dominance_penalty(&constant, &rising, !forward), None);
        assert!(!baldes::graph::coupled::CoupledKernel::dominates(
            Some(&constant),
            Some(&rising)
        ));
        assert_eq!(
            kernel.dominance_penalty(&constant, &constant, forward),
            Some(0.)
        );
    }
}

#[test]
fn paired_waiting_cost_matches_native_join_at_every_split() {
    use baldes::graph::coupled::descriptor::{
        FrontierBudget, FrontierOutcome, MachineRootInput, PreparedSlopeFunction, RawAffine,
    };
    use baldes::graph::coupled::{MachineArcFunctionInput, PairedResourceMode};
    let constant = |lo: f64, hi: f64, value: f64| match lo == hi {
        true => PreparedSlopeFunction::constant_point(value, lo).unwrap(),
        false => PreparedSlopeFunction::from_prepared_pairs(vec![(value, lo), (0., hi)]).unwrap(),
    };
    let affine = |value| MachineRootInput::Affine(RawAffine::new(value, 0.).unwrap());
    // All data are integer affine functions. On this two-step model every
    // continuous optimum occurs at an integer control or domain boundary, so
    // enumerating departure/waiting controls gives an independent oracle for
    // replay and for joins with a nonempty backward suffix. The full-forward
    // join (split 2) is not that optimum: the pinned binary returns
    // `3 + 3 * first_wait`, which exceeds it when the second rate is lower.
    // Native values: internal/reference/metasolver_paired_waiting_probe.py.
    // Function domains are the tail windows, as in the native program.
    let windows = [
        Window::new(0., 1.).unwrap(),
        Window::new(3., 8.).unwrap(),
        Window::new(6., 10.).unwrap(),
    ];
    for first_wait in 0..4 {
        for second_wait in 0..4 {
            let mut graph = Graph::<2>::new(3, 0).unwrap();
            graph.add_arc(0, 1, -20.).unwrap();
            graph.add_arc(1, 2, -20.).unwrap();
            graph
                .set_coupled_resources(&[CoupledResource::PairedFront {
                    mode: PairedResourceMode::CostBounded,
                }])
                .unwrap();
            let base: Vec<_> = (0..2)
                .map(|arc| constant(windows[arc].lower, windows[arc].upper, 1.))
                .collect();
            let increments = [2., 1.];
            let waiting = [first_wait, second_wait];
            let inputs: Vec<_> = (0..2)
                .map(|arc| MachineArcFunctionInput {
                    physical_arc: arc,
                    tail_bounds: windows[arc],
                    waiting_bounds: windows[arc + 1],
                    waiting_rate: waiting[arc],
                    base: &base[arc],
                    dependent: affine(increments[arc]),
                    dependent_lower: affine(0.),
                    dependent_upper: affine(100.),
                })
                .collect();
            let prepared = graph.prepare_paired_front_program(
                0,
                &inputs,
                affine(0.),
                windows[0],
                windows[2],
                FrontierBudget::default(),
            );
            assert!(
                matches!(prepared, FrontierOutcome::Complete(())),
                "{prepared:?}"
            );
            let expected = (3..=8)
                .flat_map(|x| {
                    ((x + 1).max(6)..=10).map(move |y| {
                        3. + f64::from(first_wait * (x - 2) + second_wait * (y - x - 1))
                    })
                })
                .reduce(f64::min)
                .unwrap();
            let kernel = graph.coupled_kernel().unwrap();
            assert_eq!(kernel.replay(&[0, 1]).unwrap(), Some(expected));
            let native = [expected, expected, 3. + 3. * f64::from(first_wait)];
            for split in 0..=2 {
                let (mut f, mut b) = (kernel.root(true), kernel.root(false));
                let mut cost = 0.;
                for arc in 0..split {
                    let (next, delta) = kernel.extend(&f, arc, true).unwrap().unwrap();
                    f = next;
                    cost += delta;
                }
                for arc in (split..2).rev() {
                    let (next, delta) = kernel.extend(&b, arc, false).unwrap().unwrap();
                    b = next;
                    cost += delta;
                }
                assert_eq!(
                    cost + kernel.join(&f, &b).unwrap().unwrap(),
                    native[split],
                    "waiting=({first_wait},{second_wait}), split={split}"
                );
            }
            let mut solver = baldes::pricing::Solver::<2>::new(3, 0).unwrap();
            solver.graph = graph;
            let result = solver.price_phase(0, 100, false).unwrap();
            // Default forward pricing completes at the sink, so it reports the
            // full-forward join value.
            assert_eq!(result.minimum_reduced_cost, native[2] - 40.);
            assert_eq!(result.columns[0].original_cost, native[2] - 40.);
        }
    }
}

fn function() -> PiecewiseLinearFunction {
    PiecewiseLinearFunction::new(vec![[0., 0.], [1., 2.], [2., 1.], [4., 3.]]).unwrap()
}
fn nonlinear_chain() -> Graph<2> {
    let mut graph = Graph::new(4, 0).unwrap();
    graph.elementary = true;
    for tail in 0..3 {
        graph.add_arc(tail, tail + 1, -10.).unwrap();
    }
    graph
        .set_coupled_resources(&[
            CoupledResource::CumulativeCost {
                initial: 0.,
                increments: vec![1.; 3],
                weights: vec![2., -1., 3.],
                sink_upper: None,
                source_upper: None,
            },
            CoupledResource::PiecewiseLinearDependent {
                parent: 0,
                initial: 0.,
                bounds: Window {
                    lower: 0.,
                    upper: 10.,
                },
                functions: vec![function(); 3],
            },
            CoupledResource::Dependent {
                parent: 1,
                initial: 10.,
                bounds: Window {
                    lower: -20.,
                    upper: 20.,
                },
                offsets: vec![0.; 3],
                factors: vec![-1.; 3],
            },
        ])
        .unwrap();
    graph
}
#[test]
fn nonmonotone_piecewise_dependency_replays_and_joins_at_every_split() {
    let graph = nonlinear_chain();
    let kernel = graph.coupled_kernel().unwrap();
    // Independent hand recurrence: x=1,2,3; f(x)=2,1,2;
    // y=2,3,5; z=8,5,0; objective=1*2 + 2*(-1) + 3*3.
    let (state, objective) = kernel.replay_state(&[0, 1, 2]).unwrap().unwrap();
    assert_eq!(state.numeric_values().unwrap(), [3., 5., 0.]);
    assert_eq!(objective, 9.);
    for split in 0..=3 {
        let (mut f, mut b) = (kernel.root(true), kernel.root(false));
        let (mut fc, mut bc) = (0., 0.);
        for arc in 0..split {
            let (next, cost) = kernel.extend(&f, arc, true).unwrap().unwrap();
            f = next;
            fc += cost;
        }
        for arc in (split..3).rev() {
            let (next, cost) = kernel.extend(&b, arc, false).unwrap().unwrap();
            b = next;
            bc += cost;
        }
        assert_eq!(
            fc + bc + kernel.join(&f, &b).unwrap().unwrap(),
            9.,
            "split={split}"
        );
    }
}
#[test]
fn graph_wide_reduction_eliminates_an_infeasible_branch_without_renumbering() {
    let mut graph: Graph<2> = Graph::new(4, 0).unwrap();
    for (tail, head) in [(0, 1), (1, 3), (0, 2), (2, 3)] {
        graph.add_arc(tail, head, 0.).unwrap();
    }
    graph
        .set_coupled_resources(&[
            CoupledResource::CumulativeCost {
                initial: 0.,
                increments: vec![0.; 4],
                weights: vec![0.; 4],
                sink_upper: None,
                source_upper: None,
            },
            CoupledResource::Dependent {
                parent: 0,
                initial: 0.,
                bounds: Window {
                    lower: 0.,
                    upper: 10.,
                },
                offsets: vec![6., 6., 4., 4.],
                factors: vec![0.; 4],
            },
        ])
        .unwrap();
    let kernel = graph.coupled_kernel().unwrap();
    assert_eq!(graph.arcs.len(), 4);
    assert!(!kernel.route_arc_possible(0).unwrap());
    assert!(!kernel.route_arc_possible(1).unwrap());
    assert!(kernel.route_arc_possible(2).unwrap());
    assert!(kernel.route_arc_possible(3).unwrap());
    assert_eq!(kernel.preprocessing().usable_arcs, 2);
    // Standalone resource replay retains partial paths, even when they cannot
    // finish a complete route. Label expansion uses complete-route filtering.
    assert!(kernel.replay(&[0]).unwrap().is_some());
    assert!(kernel
        .extend(&kernel.root(true), 0, true)
        .unwrap()
        .is_none());
    assert!(kernel.replay(&[0, 1]).unwrap().is_none());
    assert!(kernel.replay(&[2, 3]).unwrap().is_some());
}
#[test]
fn piecewise_domains_are_enforced_and_knot_values_are_exact() {
    let f = function();
    for [x, y] in &f.points {
        assert_eq!(f.value(*x).unwrap(), Some(*y));
    }
    assert_eq!(f.value(3.).unwrap(), Some(2.));
    assert_eq!(f.value(-1.).unwrap(), None);
    assert_eq!(f.value(5.).unwrap(), None);
    assert!(f.value(f64::NAN).is_err());
    assert!(PiecewiseLinearFunction::new(vec![[0., 0.], [0., 1.]]).is_err());
    assert!(PiecewiseLinearFunction::new(vec![[0., 0.], [1., f64::INFINITY]]).is_err());
}
#[test]
fn domain_reduction_preserves_a_feasible_cyclic_route() {
    let mut graph: Graph<2> = Graph::new(3, 0).unwrap();
    for (tail, head) in [(0, 1), (1, 1), (1, 2)] {
        graph.add_arc(tail, head, 0.).unwrap();
    }
    graph
        .set_coupled_resources(&[
            CoupledResource::CumulativeCost {
                initial: 0.,
                increments: vec![0.; 3],
                weights: vec![0.; 3],
                sink_upper: None,
                source_upper: None,
            },
            CoupledResource::PiecewiseLinearDependent {
                parent: 0,
                initial: 0.,
                bounds: Window {
                    lower: 0.,
                    upper: 10.,
                },
                functions: vec![
                    PiecewiseLinearFunction::new(vec![[-1., 0.], [1., 0.]]).unwrap();
                    3
                ],
            },
        ])
        .unwrap();
    let kernel = graph.coupled_kernel().unwrap();
    let mut label = kernel.root(true);
    for arc in [0, 1, 1, 1, 2] {
        assert!(kernel.route_arc_possible(arc).unwrap());
        label = kernel.extend(&label, arc, true).unwrap().unwrap().0;
    }
}
#[cfg(feature = "clp")]
#[test]
fn piecewise_network_json_round_trip_preserves_knots_and_domains() {
    let resource = CoupledResource::PiecewiseLinearDependent {
        parent: 0,
        initial: 0.,
        bounds: Window {
            lower: 0.,
            upper: 10.,
        },
        functions: vec![function()],
    };
    let encoded = serde_json::to_string(&resource).unwrap();
    assert!(encoded.contains("piecewise_linear_dependent"));
    let decoded: CoupledResource = serde_json::from_str(&encoded).unwrap();
    assert_eq!(decoded, resource);
}

#[test]
fn piecewise_pricing_agrees_in_both_directions_and_directional_enumeration() {
    use baldes::enumeration::{Controls as EnumerationControls, Enumeration};
    use baldes::labeling::engine::{EngineControls, LabelingEngine};
    use baldes::pricing::phases::Phase;
    let graph = nonlinear_chain();
    let phase = Phase {
        check_ng: true,
        jump_arcs: true,
        labels_per_bucket: i64::MAX,
    };
    for bidirectional in [false, true] {
        let mut engine = LabelingEngine::prepare(
            &graph,
            EngineControls {
                bidirectional,
                ..Default::default()
            },
            phase,
        )
        .unwrap();
        let result = engine.solve(&graph, &Default::default()).unwrap();
        assert_eq!(result.minimum_reduced_cost, -21.);
        assert_eq!(result.columns[0].original_cost, -21.);
        assert_eq!(result.columns[0].arcs, [0, 1, 2]);
        let controls = EnumerationControls {
            cutoff: 0.,
            ..Default::default()
        };
        let enumeration =
            Enumeration::run_directional(&graph, &engine.grid, &controls, |forward, label| {
                engine.enumeration_completion(&graph, forward, label, controls.cutoff, true)
            })
            .unwrap();
        assert_eq!(enumeration.paths.len, 1);
    }
}

#[test]
fn piecewise_resources_survive_completion_fixing_enumeration_and_node_forks() {
    let mut solver = baldes::pricing::Solver::<2>::new(4, 0).unwrap();
    solver.graph = nonlinear_chain();
    assert_eq!(
        solver
            .price_phase(0, 10, false)
            .unwrap()
            .minimum_reduced_cost,
        -21.
    );
    solver.complete_labels(true).unwrap();
    solver.fix(0.).unwrap();
    let routes = solver.enumerate(0.).unwrap();
    assert_eq!(routes.len(), 1);
    assert_eq!(routes[0].original_cost, -21.);
    let mut child = solver.fork_for_node();
    assert_eq!(
        child
            .price_phase(0, 10, false)
            .unwrap()
            .minimum_reduced_cost,
        -21.
    );
    assert_eq!(
        solver
            .price_phase(0, 10, false)
            .unwrap()
            .minimum_reduced_cost,
        -21.
    );
}

fn base_and_cost_chain() -> Graph<2> {
    let mut graph = Graph::new(4, 0).unwrap();
    for tail in 0..3 {
        graph.add_arc(tail, tail + 1, -10.).unwrap();
    }
    let base = PiecewiseLinearFunction::new(vec![[0., 1.], [1., 2.], [2., 4.], [4., 4.]]).unwrap();
    let charge =
        PiecewiseLinearFunction::new(vec![[0., 4.], [1., -3.], [2., 1.], [4., -2.]]).unwrap();
    let dependent_charge =
        PiecewiseLinearFunction::new(vec![[0., 0.], [1., 2.], [3., -1.], [7., 4.], [10., 2.]])
            .unwrap();
    graph
        .set_coupled_resources(&[
            CoupledResource::PiecewiseLinearResource {
                initial: 0.,
                bounds: Window {
                    lower: 0.,
                    upper: 4.,
                },
                functions: vec![base; 3],
            },
            CoupledResource::Dependent {
                parent: 0,
                initial: 0.,
                bounds: Window {
                    lower: 0.,
                    upper: 10.,
                },
                offsets: vec![0.; 3],
                factors: vec![1.; 3],
            },
            CoupledResource::PiecewiseLinearCost {
                parent: 0,
                functions: vec![charge; 3],
            },
            CoupledResource::PiecewiseLinearCost {
                parent: 1,
                functions: vec![dependent_charge; 3],
            },
        ])
        .unwrap();
    graph
}
#[test]
fn base_state_maps_and_signed_state_costs_replay_at_every_join() {
    let graph = base_and_cost_chain();
    let kernel = graph.coupled_kernel().unwrap();
    // x=1,2,4; y=1,3,7. Charges (-3+2)+(1-1)+(-2+4)=1.
    let (state, cost) = kernel.replay_state(&[0, 1, 2]).unwrap().unwrap();
    assert_eq!(state.numeric_values().unwrap(), [4., 7.]);
    assert_eq!(cost, 1.);
    for split in 0..=3 {
        let (mut f, mut b) = (kernel.root(true), kernel.root(false));
        let (mut fc, mut bc) = (0., 0.);
        for arc in 0..split {
            let (s, c) = kernel.extend(&f, arc, true).unwrap().unwrap();
            f = s;
            fc += c;
        }
        for arc in (split..3).rev() {
            let (s, c) = kernel.extend(&b, arc, false).unwrap().unwrap();
            b = s;
            bc += c;
        }
        assert_eq!(
            fc + bc + kernel.join(&f, &b).unwrap().unwrap(),
            1.,
            "split={split}"
        );
    }
}
#[test]
fn base_and_cost_maps_price_and_enumerate_without_a_cumulative_cost_coordinate() {
    let mut solver = baldes::pricing::Solver::<2>::new(4, 0).unwrap();
    solver.graph = base_and_cost_chain();
    assert_eq!(
        solver
            .price_phase(0, 10, false)
            .unwrap()
            .minimum_reduced_cost,
        -29.
    );
    solver.complete_labels(true).unwrap();
    solver.fix(0.).unwrap();
    let routes = solver.enumerate(0.).unwrap();
    assert_eq!(routes.len(), 1);
    assert_eq!(routes[0].original_cost, -29.);
}
#[test]
fn base_resource_and_state_cost_descriptors_round_trip() {
    let graph = base_and_cost_chain();
    assert!(graph.coupled_kernel().unwrap().has_costs());
    #[cfg(feature = "clp")]
    {
        let definitions = graph.coupled_kernel().unwrap().definitions();
        let encoded = serde_json::to_string(definitions).unwrap();
        let decoded: Vec<CoupledResource> = serde_json::from_str(&encoded).unwrap();
        assert_eq!(decoded, definitions);
    }
}

#[test]
fn nonlinear_future_cost_prevents_discarding_a_more_expensive_prefix() {
    let mut graph: Graph<2> = Graph::new(5, 0).unwrap();
    for (tail, head, cost) in [(0, 1, -5.), (0, 2, 0.), (1, 3, 0.), (2, 3, 0.), (3, 4, 0.)] {
        graph.add_arc(tail, head, cost).unwrap();
    }
    let identity = PiecewiseLinearFunction::new(vec![[0., 0.], [3., 3.]]).unwrap();
    let constant = |y| PiecewiseLinearFunction::new(vec![[0., y], [3., y]]).unwrap();
    let zero = constant(0.);
    graph
        .set_coupled_resources(&[
            CoupledResource::PiecewiseLinearResource {
                initial: 0.,
                bounds: Window {
                    lower: 0.,
                    upper: 3.,
                },
                functions: vec![
                    constant(1.),
                    constant(2.),
                    identity.clone(),
                    identity.clone(),
                    identity,
                ],
            },
            CoupledResource::PiecewiseLinearCost {
                parent: 0,
                functions: vec![
                    zero.clone(),
                    zero.clone(),
                    zero.clone(),
                    zero,
                    PiecewiseLinearFunction::new(vec![[0., 0.], [1., 10.], [2., -10.], [3., 0.]])
                        .unwrap(),
                ],
            },
        ])
        .unwrap();
    let mut solver = baldes::pricing::Solver::<2>::new(5, 0).unwrap();
    solver.graph = graph;
    let result = solver.price_phase(0, 10, false).unwrap();
    assert_eq!(result.minimum_reduced_cost, -10.);
    assert_eq!(result.columns[0].nodes, [0, 2, 3, 4]);
    solver.complete_labels(true).unwrap();
    solver.fix(0.).unwrap();
    let routes = solver.enumerate(0.).unwrap();
    assert_eq!(routes.len(), 1);
    assert_eq!(routes[0].nodes, [0, 2, 3, 4]);
    assert_eq!(routes[0].original_cost, -10.);
}

#[test]
fn selection_simultaneous_loads_and_signed_cumulative_dependencies_match_tiny_path_oracle() {
    use std::collections::BTreeMap;
    let mut graph: Graph<2> = Graph::new(5, 1).unwrap();
    graph.elementary = true;
    for vertex in 0..5 {
        graph.set_bounds(vertex, 0, 0., 4.).unwrap();
    }
    for tail in 0..4 {
        for head in 1..5 {
            if tail != head {
                let id = graph.add_arc(tail, head, -5.).unwrap();
                graph.arc_mut(id).unwrap().consumption[0] = 1.;
            }
        }
    }
    let deliveries = [0., 2., 1., 0., 0.];
    let pickups = [0., 0., 2., 3., 0.];
    let masks = [3u64, 1, 2, 3, 3];
    let increments = [0., 1., -1., 2., 0.];
    let weights = [0., -1., 2., -1., -2.];
    let quantities = |values: &[f64; 5]| graph.arcs.iter().map(|arc| values[arc.head]).collect();
    let definitions = vec![
        CoupledResource::Selection {
            initial: 3,
            allowed: graph.arcs.iter().map(|arc| masks[arc.head]).collect(),
            vertices: vec![],
            relaxed: false,
        },
        CoupledResource::SimultaneousPickupDelivery {
            capacity: 4.,
            deliveries: quantities(&deliveries),
            pickups: quantities(&pickups),
            tolerance: 0.,
        },
        CoupledResource::CumulativeCost {
            initial: 0.,
            increments: quantities(&increments),
            weights: quantities(&weights),
            sink_upper: None,
            source_upper: None,
        },
        CoupledResource::Dependent {
            parent: 2,
            initial: 2.,
            bounds: Window {
                lower: 0.,
                upper: 8.,
            },
            offsets: vec![0.; graph.arcs.len()],
            factors: vec![1.; graph.arcs.len()],
        },
    ];
    graph.set_coupled_resources(&definitions).unwrap();
    fn paths(prefix: &mut Vec<usize>, out: &mut Vec<Vec<usize>>) {
        let mut finished = prefix.clone();
        finished.push(4);
        out.push(finished);
        for next in 1..4 {
            if !prefix.contains(&next) {
                prefix.push(next);
                paths(prefix, out);
                prefix.pop();
            }
        }
    }
    let mut candidates = Vec::new();
    paths(&mut vec![0], &mut candidates);
    let mut oracle = BTreeMap::new();
    for nodes in candidates {
        // Independent physical-load replay: load all deliveries at departure,
        // unload then collect at each stop. No label-summary formulas are used.
        let mut load: f64 = nodes[1..].iter().map(|&head| deliveries[head]).sum();
        let (mut mask, mut x, mut y, mut cost) = (3u64, 0., 2., 0.);
        let mut feasible = load <= 4.;
        for &head in &nodes[1..] {
            mask &= masks[head];
            load -= deliveries[head];
            load += pickups[head];
            x += increments[head];
            y += x;
            cost += -5. + x * weights[head];
            feasible &= mask != 0 && load >= 0. && load <= 4. && y >= 0. && y <= 8.;
        }
        if feasible && cost < 0. {
            let arcs: Vec<_> = nodes
                .windows(2)
                .map(|pair| {
                    graph
                        .arcs
                        .iter()
                        .find(|arc| arc.tail == pair[0] && arc.head == pair[1])
                        .unwrap()
                        .id
                })
                .collect();
            oracle.insert(arcs, cost);
        }
    }
    assert!(oracle.len() >= 3);
    let expected = oracle.values().copied().fold(f64::INFINITY, f64::min);
    let mut solver = baldes::pricing::Solver::<2>::new(5, 1).unwrap();
    solver.graph = graph;
    assert_eq!(
        solver
            .price_phase(0, 100, false)
            .unwrap()
            .minimum_reduced_cost,
        expected
    );
    solver.complete_labels(true).unwrap();
    solver.fix(0.).unwrap();
    let routes = solver.enumerate(0.).unwrap();
    let actual: BTreeMap<_, _> = routes
        .into_iter()
        .map(|route| (route.arcs, route.original_cost))
        .collect();
    assert_eq!(actual, oracle);
}

#[cfg(feature = "clp")]
#[test]
fn root_phase_i_recovers_nonlinear_cost_columns_then_restores_original_objective() {
    use baldes::master::{
        cg::{FeasibilityStatus, Status},
        routing::{self, RootOptions, RoutingContext},
    };
    let mut graph: Graph<2> = Graph::new(4, 1).unwrap();
    for vertex in 0..4 {
        graph.set_bounds(vertex, 0, 0., 10.).unwrap();
    }
    for (tail, head) in [(0, 1), (1, 3), (0, 2), (2, 3)] {
        graph.add_arc(tail, head, 0.).unwrap();
    }
    graph
        .set_coupled_resources(&[
            CoupledResource::PiecewiseLinearResource {
                initial: 0.,
                bounds: Window {
                    lower: 0.,
                    upper: 2.,
                },
                functions: vec![
                    PiecewiseLinearFunction::new(vec![[0., 1.], [1., 2.], [2., 2.]])
                        .unwrap();
                    4
                ],
            },
            CoupledResource::PiecewiseLinearCost {
                parent: 0,
                functions: vec![
                    PiecewiseLinearFunction::new(vec![[0., 0.], [1., 10.], [2., 20.]])
                        .unwrap();
                    4
                ],
            },
        ])
        .unwrap();
    let mut options = RootOptions::default();
    options.enumerate = false;
    options.round_reduced_costs = false;
    options.cg.artificial_cost = 1.;
    options.cg.max_penalty_updates = 0;
    options.cg.insert_non_negative = false;
    options.cg.stabilize = false;
    let output = routing::solve(
        RoutingContext::set_partitioning(graph, 2.).unwrap(),
        options,
    )
    .unwrap();
    assert_eq!(output.master.status, Status::Optimal, "{output:?}");
    assert!((output.master.lp - 60.).abs() < 1e-7, "{output:?}");
    assert!(output
        .master
        .feasibility
        .iter()
        .any(|round| round.status == FeasibilityStatus::Feasible));
    assert!(output
        .master
        .artificials
        .iter()
        .all(|&column| output.values[column].abs() < 1e-7));
    assert_eq!(
        output
            .route_values()
            .iter()
            .map(|(route, value)| route.original_cost * value)
            .sum::<f64>(),
        60.
    );
}

#[test]
fn nonlinear_state_costs_compose_with_cut_bearing_joins_and_dual_refresh() {
    use baldes::cuts::state::LimitedInput;
    use baldes::labeling::engine::{EngineControls, LabelingEngine};
    use baldes::pricing::phases::Phase;
    let graph = base_and_cost_chain();
    let phase = Phase {
        check_ng: true,
        jump_arcs: true,
        labels_per_bucket: i64::MAX,
    };
    for bidirectional in [false, true] {
        let mut engine = LabelingEngine::prepare(
            &graph,
            EngineControls {
                bidirectional,
                ..Default::default()
            },
            phase,
        )
        .unwrap();
        let cut = |dual| LimitedInput {
            covering: false,
            type2: false,
            arc_memory: false,
            denominator: 2,
            dual,
            vertex_coefficient: vec![0, 1, 1, 0],
            arc_coefficient: Default::default(),
            memory_vertices: (0..4).collect(),
            memory_arcs: Default::default(),
        };
        assert!(engine.load_cuts(&graph, vec![cut(-2.)], vec![]));
        let first = engine.solve(&graph, &Default::default()).unwrap();
        // floor((1+1)/2) = 1. Inputs carry minus the master dual,
        // so these negative cut prices reduce the priced objective only.
        assert_eq!(first.minimum_reduced_cost, -31.);
        assert_eq!(first.columns[0].original_cost, -29.);
        engine.update_cut_duals(vec![cut(-7.)], vec![]);
        let refreshed = engine.solve(&graph, &Default::default()).unwrap();
        assert_eq!(refreshed.minimum_reduced_cost, -36.);
        assert_eq!(refreshed.columns[0].original_cost, -29.);
    }
}

fn interval_cost_fixture(seed: usize) -> Graph<2> {
    let mut graph = Graph::<2>::new(4, 1).unwrap();
    graph.elementary = true;
    graph.vertices[0].elements.clear();
    graph.vertices[3].elements.clear();
    for v in 0..4 {
        graph.set_bounds(v, 0, 0., 3.).unwrap();
    }
    for (tail, head) in [(0, 1), (1, 2), (2, 3), (0, 2), (2, 1), (1, 3)] {
        let id = graph.add_arc(tail, head, -4.).unwrap();
        graph.arc_mut(id).unwrap().consumption[0] = 1.;
    }
    let functions = (0..6)
        .map(|i| {
            let knot = ((i * 3 + seed) % 5) as f64;
            PiecewiseLinearFunction::new(vec![
                [-8., 7. + i as f64],
                [knot, -3. - (seed % 3) as f64],
                [12., 9. - i as f64],
            ])
            .unwrap()
        })
        .collect();
    graph
        .set_coupled_resources(&[
            CoupledResource::Selection {
                initial: 3,
                allowed: vec![3; 6],
                vertices: vec![],
                relaxed: false,
            },
            CoupledResource::SimultaneousPickupDelivery {
                capacity: 3.,
                deliveries: vec![1.; 6],
                pickups: vec![1.; 6],
                tolerance: 0.,
            },
            CoupledResource::IntervalCost {
                initial: Window {
                    lower: 0.,
                    upper: 4.,
                },
                bounds: Window {
                    lower: -4.,
                    upper: 8.,
                },
                increments: (0..6).map(|i| ((i + seed) % 3) as f64 - 1.).collect(),
                functions,
            },
        ])
        .unwrap();
    graph
}

// Independent optimization in the original initial coordinate. For a fixed
// walk every charge is PWL in x0; its optimum occurs at a feasible endpoint
// or at a translated original charge knot. Never consult production fronts.
fn interval_path_oracle(graph: &Graph<2>, path: &[usize]) -> Option<f64> {
    let CoupledResource::IntervalCost {
        initial,
        bounds,
        increments,
        functions,
    } = &graph.coupled_kernel().unwrap().definitions()[2]
    else {
        unreachable!()
    };
    let (mut lower, mut upper, mut shift) = (initial.lower, initial.upper, 0.);
    let mut knots = vec![lower, upper];
    for &arc in path {
        shift += increments[arc];
        let points = &functions[arc].points;
        lower = lower.max(bounds.lower - shift).max(points[0][0] - shift);
        upper = upper
            .min(bounds.upper - shift)
            .min(points.last().unwrap()[0] - shift);
        knots.extend(points.iter().map(|p| p[0] - shift));
    }
    if lower > upper {
        return None;
    }
    knots.extend([lower, upper]);
    knots
        .into_iter()
        .filter(|x| *x >= lower && *x <= upper)
        .map(|mut x| {
            let mut cost = 0.;
            for &arc in path {
                x += increments[arc];
                let points = &functions[arc].points;
                let pair = points
                    .windows(2)
                    .find(|p| p[0][0] <= x && x <= p[1][0])
                    .unwrap();
                let value = pair[0][1]
                    + (pair[1][1] - pair[0][1]) * ((x - pair[0][0]) / (pair[1][0] - pair[0][0]));
                cost += value;
            }
            cost
        })
        .reduce(f64::min)
}

#[test]
fn interval_fronts_match_original_coordinate_oracle_at_every_join_split() {
    for seed in 0..16 {
        let graph = interval_cost_fixture(seed);
        let kernel = graph.coupled_kernel().unwrap();
        for path in [vec![0, 1, 2], vec![0, 5], vec![3, 2], vec![3, 4, 5]] {
            let expected = interval_path_oracle(&graph, &path).unwrap();
            assert!((kernel.replay(&path).unwrap().unwrap() - expected).abs() < 1e-10);
            for split in 0..=path.len() {
                let (mut f, mut b) = (kernel.root(true), kernel.root(false));
                let (mut fc, mut bc) = (0., 0.);
                for &a in &path[..split] {
                    let (next, delta) = kernel.extend(&f, a, true).unwrap().unwrap();
                    f = next;
                    fc += delta;
                }
                for &a in path[split..].iter().rev() {
                    let (next, delta) = kernel.extend(&b, a, false).unwrap().unwrap();
                    b = next;
                    bc += delta;
                }
                let actual = fc + bc + kernel.join(&f, &b).unwrap().unwrap();
                assert!(
                    (actual - expected).abs() < 1e-10,
                    "seed={seed} split={split} path={path:?}: {actual} != {expected}"
                );
            }
        }
    }
}

#[test]
fn interval_front_pricing_cuts_fixing_retention_and_forks_preserve_physical_paths() {
    use std::collections::BTreeMap;
    for seed in 0..8 {
        for mode in [0., 1., 2.] {
            let graph = interval_cost_fixture(seed);
            let paths = [vec![0usize, 1, 2], vec![0, 5], vec![3, 2], vec![3, 4, 5]];
            let mut solver = baldes::pricing::Solver::<2>::new(4, 1).unwrap();
            solver.graph = graph;
            solver.enumeration_preserve_arc_paths = true;
            solver
                .set_option("BALDESuseBidirectionalSearch", mode)
                .unwrap();
            solver
                .cuts
                .add_strong_k_path_cut(4, &[1, 2], 1, -1.25)
                .unwrap();
            let oracle = |g: &Graph<2>| -> BTreeMap<Vec<i32>, f64> {
                paths
                    .iter()
                    .map(|path| {
                        let value = interval_path_oracle(g, path).unwrap()
                            + path.iter().map(|&a| g.arcs[a].reduced_cost).sum::<f64>()
                            + 1.25;
                        (path.iter().map(|&a| g.arcs[a].id).collect(), value)
                    })
                    .collect()
            };
            let expected = oracle(&solver.graph);
            let priced = solver.price_phase(0, 100, false).unwrap();
            assert!(priced.exact);
            assert!(
                (priced.minimum_reduced_cost
                    - expected.values().copied().reduce(f64::min).unwrap())
                .abs()
                    < 1e-9
            );
            let check = |routes: Vec<baldes::Route>, expected: &BTreeMap<Vec<i32>, f64>| {
                assert_eq!(routes.len(), expected.len());
                for route in routes {
                    assert!(
                        (route.cost - expected[&route.arcs]).abs() < 1e-9,
                        "{:?}",
                        route
                    );
                }
            };
            check(solver.enumerate(1000.).unwrap(), &expected);
            let mut sibling = solver.fork_for_node();
            for (i, arc) in solver.graph.arcs.iter_mut().enumerate() {
                arc.reduced_cost += (i % 3) as f64;
            }
            let refreshed = oracle(&solver.graph);
            check(
                solver.inspect_enumeration(1000.).unwrap().unwrap(),
                &refreshed,
            );
            solver.fix(-12.).unwrap();
            let retained = refreshed
                .into_iter()
                .filter(|(_, cost)| *cost < -12.)
                .collect();
            check(solver.enumerate(-12.).unwrap(), &retained);
            check(
                sibling.inspect_enumeration(1000.).unwrap().unwrap(),
                &expected,
            );
            let mut child = sibling.fork_for_node();
            child.set_forbidden_arcs(vec![0]).unwrap();
            let child_expected = expected
                .iter()
                .filter(|(p, _)| !p.contains(&0))
                .map(|(p, c)| (p.clone(), *c))
                .collect();
            check(child.enumerate(1000.).unwrap(), &child_expected);
            check(
                sibling.inspect_enumeration(1000.).unwrap().unwrap(),
                &expected,
            );
        }
    }
}

#[cfg(feature = "clp")]
#[test]
fn interval_cost_phase_i_keeps_domains_and_restores_original_front_objective() {
    use baldes::master::{
        cg::{FeasibilityStatus, Status},
        routing::{self, RootOptions, RoutingContext},
    };
    let mut graph = Graph::<2>::new(4, 1).unwrap();
    for v in 0..4 {
        graph.set_bounds(v, 0, 0., 10.).unwrap();
    }
    for (u, v) in [(0, 1), (1, 3), (0, 2), (2, 3)] {
        graph.add_arc(u, v, 0.).unwrap();
    }
    graph
        .set_coupled_resources(&[CoupledResource::IntervalCost {
            initial: Window {
                lower: 0.,
                upper: 4.,
            },
            bounds: Window {
                lower: 0.,
                upper: 4.,
            },
            increments: vec![0.; 4],
            functions: vec![
                PiecewiseLinearFunction::new(vec![[0., 30.], [2., 10.], [4., 30.]])
                    .unwrap();
                4
            ],
        }])
        .unwrap();
    let kernel = graph.coupled_kernel().unwrap();
    assert_eq!(kernel.replay(&[0, 1]).unwrap(), Some(20.));
    let mut options = RootOptions {
        enumerate: false,
        round_reduced_costs: false,
        ..Default::default()
    };
    options.cg.artificial_cost = 1.;
    options.cg.max_penalty_updates = 0;
    options.cg.insert_non_negative = false;
    options.cg.stabilize = false;
    let output = routing::solve(
        RoutingContext::set_partitioning(graph, 2.).unwrap(),
        options,
    )
    .unwrap();
    assert_eq!(output.master.status, Status::Optimal, "{output:?}");
    assert!((output.master.lp - 40.).abs() < 1e-8);
    assert!(output
        .master
        .feasibility
        .iter()
        .any(|r| r.status == FeasibilityStatus::Feasible));
    assert!(
        (output
            .route_values()
            .iter()
            .map(|(r, x)| r.original_cost * x)
            .sum::<f64>()
            - 40.)
            .abs()
            < 1e-8
    );
}

#[test]
fn interval_cost_domains_reject_a_walk_even_when_each_arc_has_a_cheaper_point() {
    let mut graph = Graph::<2>::new(3, 0).unwrap();
    graph.add_arc(0, 1, 0.).unwrap();
    graph.add_arc(1, 2, 0.).unwrap();
    graph
        .set_coupled_resources(&[CoupledResource::IntervalCost {
            initial: Window {
                lower: 0.,
                upper: 4.,
            },
            bounds: Window {
                lower: 0.,
                upper: 4.,
            },
            increments: vec![0.; 2],
            functions: vec![
                PiecewiseLinearFunction::new(vec![[0., 0.], [1., -1.]]).unwrap(),
                PiecewiseLinearFunction::new(vec![[2., -1.], [4., 0.]]).unwrap(),
            ],
        }])
        .unwrap();
    assert!(graph
        .coupled_kernel()
        .unwrap()
        .replay(&[0, 1])
        .unwrap()
        .is_none());
    let mut solver = baldes::pricing::Solver::<2>::new(3, 0).unwrap();
    solver.graph = graph;
    assert!(solver
        .price_phase(0, 100, false)
        .unwrap()
        .columns
        .is_empty());
    assert!(solver.enumerate(100.).unwrap().is_empty());
}

#[test]
fn interval_front_decimal_translations_and_multiple_resources_keep_full_functions() {
    let mut graph = Graph::<2>::new(3, 0).unwrap();
    graph.add_arc(0, 1, -2.).unwrap();
    graph.add_arc(1, 2, -2.).unwrap();
    let functions =
        vec![PiecewiseLinearFunction::new(vec![[-1., 2.], [0.3, -1.], [2., 3.]]).unwrap(); 2];
    graph
        .set_coupled_resources(&[
            CoupledResource::IntervalCost {
                initial: Window {
                    lower: 0.,
                    upper: 1.,
                },
                bounds: Window {
                    lower: -1.,
                    upper: 2.,
                },
                increments: vec![0.1, -0.2],
                functions,
            },
            CoupledResource::IntervalCost {
                initial: Window {
                    lower: 0.,
                    upper: 4.,
                },
                bounds: Window {
                    lower: 0.,
                    upper: 4.,
                },
                increments: vec![0.; 2],
                functions: vec![
                    PiecewiseLinearFunction::new(vec![[0., 1.], [2., -3.], [4., 1.]])
                        .unwrap();
                    2
                ],
            },
        ])
        .unwrap();
    // The competing knots are x0=0.2 and x0=0.4. The first is cheaper:
    // states 0.3 and 0.1 cost -2 + 0.2*3/1.3. The second front costs -6.
    let expected = -8. + 0.2 * 3. / 1.3;
    let kernel = graph.coupled_kernel().unwrap();
    let (state, cost) = kernel.replay_state(&[0, 1]).unwrap().unwrap();
    assert!((cost - expected).abs() < 1e-12);
    assert_eq!(state.interval_cost_fronts().count(), 2);
    let (f, fc) = kernel.extend(&kernel.root(true), 0, true).unwrap().unwrap();
    let (b, bc) = kernel
        .extend(&kernel.root(false), 1, false)
        .unwrap()
        .unwrap();
    assert!((fc + bc + kernel.join(&f, &b).unwrap().unwrap() - expected).abs() < 1e-12);
}

#[cfg(feature = "clp")]
#[test]
fn interval_cost_json_round_trip_preserves_initial_interval_and_charge_knots() {
    let resource = interval_cost_fixture(0)
        .coupled_kernel()
        .unwrap()
        .definitions()[2]
        .clone();
    let encoded = serde_json::to_string(&resource).unwrap();
    assert!(encoded.contains("interval_cost"));
    assert_eq!(
        serde_json::from_str::<CoupledResource>(&encoded).unwrap(),
        resource
    );
}

#[test]
fn interval_front_cut_dual_refresh_reprices_joins_without_changing_original_objectives() {
    use baldes::cuts::state::LimitedInput;
    use baldes::labeling::engine::{EngineControls, LabelingEngine};
    use baldes::pricing::phases::Phase;
    let graph = interval_cost_fixture(3);
    let paths = [vec![0usize, 1, 2], vec![0, 5], vec![3, 2], vec![3, 4, 5]];
    let cut = |dual| LimitedInput {
        covering: false,
        type2: false,
        arc_memory: false,
        denominator: 2,
        dual,
        vertex_coefficient: vec![0, 1, 1, 0],
        arc_coefficient: Default::default(),
        memory_vertices: (0..4).collect(),
        memory_arcs: Default::default(),
    };
    for bidirectional in [false, true] {
        let mut engine = LabelingEngine::prepare(
            &graph,
            EngineControls {
                bidirectional,
                ..Default::default()
            },
            Phase {
                check_ng: true,
                jump_arcs: true,
                labels_per_bucket: i64::MAX,
            },
        )
        .unwrap();
        assert!(engine.load_cuts(&graph, vec![cut(-2.)], vec![]));
        for dual in [-2., -7.] {
            engine.update_cut_duals(vec![cut(dual)], vec![]);
            let result = engine.solve(&graph, &Default::default()).unwrap();
            let expected = paths
                .iter()
                .map(|path| {
                    interval_path_oracle(&graph, path).unwrap() - 4. * path.len() as f64
                        + dual * ((path.len() - 1) / 2) as f64
                })
                .reduce(f64::min)
                .unwrap();
            assert!((result.minimum_reduced_cost - expected).abs() < 1e-9);
            for route in result.columns {
                let path: Vec<_> = route
                    .arcs
                    .iter()
                    .map(|id| graph.arcs.iter().position(|a| a.id == *id).unwrap())
                    .collect();
                let original =
                    interval_path_oracle(&graph, &path).unwrap() - 4. * path.len() as f64;
                assert!((route.original_cost - original).abs() < 1e-9);
            }
        }
    }
}

#[test]
fn zero_charge_interval_suffix_retains_physical_scalar_costs_in_every_search_mode() {
    let mut graph = Graph::<2>::new(3, 1).unwrap();
    graph.add_arc(0, 1, 1.).unwrap();
    graph.add_arc(1, 2, 2.).unwrap();
    for arc in &mut graph.arcs {
        arc.consumption[0] = 1.;
    }
    graph
        .set_coupled_resources(&[CoupledResource::IntervalCost {
            initial: Window::new(0., 1.).unwrap(),
            bounds: Window::new(0., 1.).unwrap(),
            increments: vec![0.; 2],
            functions: vec![PiecewiseLinearFunction::new(vec![[0., 0.], [1., 0.]]).unwrap(); 2],
        }])
        .unwrap();
    let reference = baldes::labeling::reference_price(&graph, &Default::default()).unwrap();
    assert_eq!(reference.minimum_reduced_cost, 3.);
    for mode in [0., 1., 2.] {
        let mut solver = baldes::pricing::Solver::<2>::new(3, 1).unwrap();
        solver.graph = graph.clone();
        solver
            .set_option("BALDESuseBidirectionalSearch", mode)
            .unwrap();
        let result = solver.price_phase(0, 10, false).unwrap();
        assert!(result.exact);
        assert_eq!(result.minimum_reduced_cost, 3., "mode={mode}");
        assert!(result
            .columns
            .iter()
            .all(|route| route.cost == 3. && route.original_cost == 3.));
    }
}

#[test]
fn signed_custom_and_cumulative_states_and_every_join_match_metasolver_observations() {
    // Regenerate with internal/reference/metasolver_custom.py. Expectations come
    // from the pinned modern ELF helpers, independently of BALDES arithmetic.
    let fixture = include_str!("../fixtures/coupled/metasolver-custom-signed-splits.txt");
    let mut observations = 0;
    for line in fixture.lines().filter(|line| !line.starts_with('#')) {
        let row: Vec<f64> = line
            .split_whitespace()
            .map(|value| value.parse().unwrap())
            .collect();
        assert_eq!(row.len(), 20);
        let split = row[1] as usize;
        let increments = row[2..8].to_vec();
        let weights = row[8..14].to_vec();
        for definition in [
            CoupledResource::CustomResource {
                increments: increments.clone(),
                weights: weights.clone(),
                forward_bound: None,
                backward_bound: None,
            },
            CoupledResource::CumulativeCost {
                initial: 0.,
                increments: increments.clone(),
                weights: weights.clone(),
                sink_upper: None,
                source_upper: None,
            },
        ] {
            let mut graph = Graph::<2>::new(7, 0).unwrap();
            for arc in 0..6 {
                graph.add_arc(arc, arc + 1, 0.).unwrap();
            }
            graph.set_coupled_resources(&[definition]).unwrap();
            let kernel = graph.coupled_kernel().unwrap();
            let mut forward = kernel.root(true);
            let mut forward_cost = 0.;
            for arc in 0..split {
                let (next, cost) = kernel.extend(&forward, arc, true).unwrap().unwrap();
                forward = next;
                forward_cost += cost;
            }
            let mut backward = kernel.root(false);
            let mut backward_cost = 0.;
            for arc in (split..6).rev() {
                let (next, cost) = kernel.extend(&backward, arc, false).unwrap().unwrap();
                backward = next;
                backward_cost += cost;
            }
            let cross = kernel.join(&forward, &backward).unwrap().unwrap();
            assert_eq!(
                forward_cost, row[14],
                "forward seed={} split={split}",
                row[0]
            );
            assert_eq!(forward.numeric_values().unwrap(), &[row[15]]);
            assert_eq!(
                backward_cost, row[16],
                "backward seed={} split={split}",
                row[0]
            );
            assert_eq!(cross, row[18], "join seed={} split={split}", row[0]);
            assert_eq!(forward_cost + backward_cost + cross, row[19]);
            assert_eq!(kernel.replay(&[0, 1, 2, 3, 4, 5]).unwrap(), Some(row[19]));
        }
        observations += 1;
    }
    assert_eq!(observations, 336);
}

#[test]
fn signed_resource_pricing_and_enumeration_match_metasolver_path_costs_in_every_direction() {
    let fixture = include_str!("../fixtures/coupled/metasolver-custom-signed-splits.txt");
    let paths: Vec<Vec<f64>> = fixture
        .lines()
        .filter(|line| !line.starts_with('#'))
        .map(|line| {
            line.split_whitespace()
                .map(|value| value.parse().unwrap())
                .collect::<Vec<f64>>()
        })
        .filter(|row| row[1] == 6.)
        .collect();
    assert_eq!(paths.len(), 48);
    for pair in paths.chunks_exact(2) {
        let increments: Vec<_> = pair
            .iter()
            .flat_map(|row| row[2..8].iter().copied())
            .collect();
        let weights: Vec<_> = pair
            .iter()
            .flat_map(|row| row[8..14].iter().copied())
            .collect();
        let expected = [pair[0][19] - 256., pair[1][19] - 256.];
        for definition in [
            CoupledResource::CustomResource {
                increments: increments.clone(),
                weights: weights.clone(),
                forward_bound: None,
                backward_bound: None,
            },
            CoupledResource::CumulativeCost {
                initial: 0.,
                increments: increments.clone(),
                weights: weights.clone(),
                sink_upper: None,
                source_upper: None,
            },
        ] {
            let mut graph = Graph::<2>::new(12, 1).unwrap();
            for vertex in 0..12 {
                graph.set_bounds(vertex, 0, 0., 6.).unwrap();
            }
            for path in [[0, 1, 2, 3, 4, 5, 11], [0, 6, 7, 8, 9, 10, 11]] {
                for (index, ends) in path.windows(2).enumerate() {
                    let arc = graph
                        .add_arc(ends[0], ends[1], if index == 0 { -256. } else { 0. })
                        .unwrap();
                    graph.arc_mut(arc).unwrap().consumption[0] = 1.;
                }
            }
            graph.set_coupled_resources(&[definition]).unwrap();
            for direction in [0., 1., 2.] {
                let mut solver = baldes::pricing::Solver::<2>::new(12, 1).unwrap();
                solver.graph = graph.clone();
                solver.enumeration_preserve_arc_paths = true;
                solver
                    .set_option("BALDESuseBidirectionalSearch", direction)
                    .unwrap();
                let result = solver.price_phase(0, 100, false).unwrap();
                assert!(result.exact);
                assert_eq!(
                    result.minimum_reduced_cost,
                    expected[0].min(expected[1]),
                    "seed={} direction={direction}",
                    pair[0][0]
                );
                let routes = solver.enumerate(1000.).unwrap();
                assert_eq!(routes.len(), 2);
                for route in routes {
                    let branch = usize::from(route.arcs[0] == graph.arcs[6].id);
                    assert_eq!(route.arcs.len(), 6);
                    assert_eq!(route.original_cost, expected[branch]);
                    assert_eq!(route.cost, expected[branch]);
                }
            }
        }
    }
}

#[test]
fn directional_cost_bounds_match_retained_metasolver_dominance_controls() {
    let fixture = include_str!("../fixtures/coupled/metasolver-directional-dominance.txt");
    let mut comparisons = 0;
    let mut positive_comparisons = 0;
    let mut conservative = 0;
    let mut extra = 0;
    let mut scoped_forward_gaps = 0;
    let mut scoped_backward_gaps = 0;
    for line in fixture.lines().filter(|line| !line.starts_with('#')) {
        let row: Vec<f64> = line
            .split_whitespace()
            .map(|value| value.parse().unwrap())
            .collect();
        assert_eq!(row.len(), 13);
        let forward = row[0] == 1.;
        let initial = row[1];
        let signed = row[2] == 1.;
        let (ai, bi, aw, bw) = (row[4], row[5], row[6], row[7]);
        let (increments, weights) = if forward {
            (vec![ai, bi, 0.], vec![aw, bw, 1.])
        } else {
            (vec![3., ai, bi], vec![0., aw, bw])
        };
        for custom in [false, true] {
            if custom && initial != 0. {
                continue; // A custom resource has a fixed zero initial state.
            }
            let definition = if custom {
                CoupledResource::CustomResource {
                    increments: increments.clone(),
                    weights: weights.clone(),
                    forward_bound: Some(3.),
                    backward_bound: Some(7.),
                }
            } else {
                CoupledResource::CumulativeCost {
                    initial,
                    increments: increments.clone(),
                    weights: weights.clone(),
                    sink_upper: Some(3.),
                    source_upper: Some(7.),
                }
            };
            let mut graph = Graph::<2>::new(3, 0).unwrap();
            for (tail, head) in if forward {
                [(0, 1), (0, 1), (1, 2)]
            } else {
                [(0, 1), (1, 2), (1, 2)]
            } {
                graph.add_arc(tail, head, 0.).unwrap();
            }
            graph.set_coupled_resources(&[definition]).unwrap();
            let kernel = graph.coupled_kernel().unwrap();
            let root = kernel.root(forward);
            let (a, ac) = kernel
                .extend(&root, if forward { 0 } else { 1 }, forward)
                .unwrap()
                .unwrap();
            let (b, bc) = kernel
                .extend(&root, if forward { 1 } else { 2 }, forward)
                .unwrap()
                .unwrap();
            assert_eq!(ac, row[8]);
            assert_eq!(bc, row[9]);
            if forward {
                assert_eq!(a.numeric_values(), Some(&row[10..11]));
                assert_eq!(b.numeric_values(), Some(&row[11..12]));
            }
            let accepted = kernel
                .dominance_penalty(&a, &b, forward)
                .is_some_and(|penalty| ac + penalty <= bc);
            conservative += usize::from(!accepted && row[12] <= 0.);
            extra += usize::from(accepted && row[12] > 0.);
            let scoped = kernel
                .dominance_penalty_at_vertex(&a, &b, forward, 1)
                .is_some_and(|penalty| ac + penalty <= bc);
            if row[12] <= 0. {
                assert!(scoped, "vertex context: {line} custom={custom}");
                if forward {
                    scoped_forward_gaps += usize::from(!accepted);
                } else {
                    scoped_backward_gaps += usize::from(!accepted);
                }
            }
            if !accepted && row[12] <= 0. && std::env::var_os("BALDES_DOMINANCE_AUDIT").is_some() {
                let (ca, cb) = if forward {
                    (
                        (initial + ai) * aw + (initial + ai),
                        (initial + bi) * bw + (initial + bi),
                    )
                } else {
                    ((initial + 3. + ai) * aw, (initial + 3. + bi) * bw)
                };
                eprintln!("DOMINANCE_REJECTION forward={forward} custom={custom} input={line} completed_a={ca} completed_b={cb}");
            }
            if !signed {
                // The retained reference charge already includes ac-bc.
                assert_eq!(accepted, row[12] <= 0., "{line} custom={custom}");
                positive_comparisons += 1;
            }
            if accepted || scoped {
                // Independent literal completion: forward appends weight1;
                // backward prepends a zero-charge increment3. All values are
                // small dyadics, so these products/sums are exact binary64.
                let (completed_a, completed_b) = if forward {
                    (
                        (initial + ai) * aw + (initial + ai) * 1.,
                        (initial + bi) * bw + (initial + bi) * 1.,
                    )
                } else {
                    ((initial + 3. + ai) * aw, (initial + 3. + bi) * bw)
                };
                assert!(completed_a <= completed_b, "{line} custom={custom}");
            }
            comparisons += 1;
        }
    }
    assert_eq!(comparisons, 768);
    assert_eq!(positive_comparisons, 384);
    assert_eq!(extra, 0);
    assert_eq!(scoped_forward_gaps, 42);
    assert_eq!(scoped_backward_gaps, 28);
    eprintln!("vertex context closes {scoped_forward_gaps} forward and {scoped_backward_gaps} backward retained gaps");
    eprintln!(
        "directional dominance: {conservative} conservative rejections, {extra} extra acceptances"
    );
}

#[cfg(feature = "clp")]
#[test]
fn signed_dag_numeric_prefix_pays_for_both_future_weight_signs() {
    let mut graph = Graph::<2>::new(3, 0).unwrap();
    for (tail, head) in [(0, 1), (0, 1), (1, 2), (1, 2)] {
        graph.add_arc(tail, head, 0.).unwrap();
    }
    graph
        .set_coupled_resources(&[CoupledResource::CumulativeCost {
            initial: 100.,
            increments: vec![-1., 0., 0., 0.],
            weights: vec![-1., 0., -1., 1.],
            sink_upper: None,
            source_upper: None,
        }])
        .unwrap();
    let kernel = graph.coupled_kernel().unwrap();
    let root = kernel.root(true);
    let (a, ac) = kernel.extend(&root, 0, true).unwrap().unwrap();
    let (b, bc) = kernel.extend(&root, 1, true).unwrap().unwrap();
    assert_eq!((ac, bc), (-99., 0.));
    assert_eq!(kernel.dominance_penalty(&a, &b, true), Some(0.));
    for arc in [2, 3] {
        let (_, da) = kernel.extend(&a, arc, true).unwrap().unwrap();
        let (_, db) = kernel.extend(&b, arc, true).unwrap().unwrap();
        assert!(ac + da <= bc + db);
    }
    assert_eq!(kernel.dominance_penalty(&b, &a, true), None);
}

#[test]
fn integral_backward_signed_suffix_has_zero_compensation_without_rounding_slack() {
    for initial in [0., 2.] {
        let mut graph = Graph::<2>::new(3, 0).unwrap();
        for (tail, head) in [(0, 1), (1, 2), (1, 2)] {
            graph.add_arc(tail, head, 0.).unwrap();
        }
        graph
            .set_coupled_resources(&[CoupledResource::CumulativeCost {
                initial,
                increments: vec![1., 0., 0.],
                weights: vec![0., -1., 0.],
                sink_upper: None,
                source_upper: None,
            }])
            .unwrap();
        let kernel = graph.coupled_kernel().unwrap();
        let root = kernel.root(false);
        let (a, ac) = kernel.extend(&root, 1, false).unwrap().unwrap();
        let (b, bc) = kernel.extend(&root, 2, false).unwrap().unwrap();
        assert!(ac <= bc);
        assert_eq!(kernel.dominance_penalty(&a, &b, false), Some(0.));
        let (prefix, _) = kernel.extend(&kernel.root(true), 0, true).unwrap().unwrap();
        assert!(
            kernel.join(&prefix, &a).unwrap().unwrap()
                <= kernel.join(&prefix, &b).unwrap().unwrap()
        );
    }
}
