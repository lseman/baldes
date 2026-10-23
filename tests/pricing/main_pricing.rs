use baldes::graph::ng::PreparedNg;
use baldes::labeling::{extend_resources, NO_ROUTE_COST};
use baldes::{price, Controls, Graph, ResourceKind};
use std::time::Duration;

fn graph(nodes: usize, resources: usize, arcs: &[(usize, usize, f64, f64)]) -> Graph<2> {
    let mut g = Graph::<2>::new(nodes, resources).unwrap();
    for &(tail, head, cost, use_) in arcs {
        let a = g.add_arc(tail, head, cost).unwrap();
        if resources != 0 {
            g.arc_mut(a).unwrap().consumption[0] = use_;
        }
    }
    if resources != 0 {
        for v in 0..nodes {
            g.set_bounds(v, 0, 0., 4.).unwrap();
        }
    }
    g
}
fn cycle_graph() -> Graph<2> {
    graph(
        5,
        1,
        &[
            (0, 1, 3., 1.),
            (1, 4, -8., 1.),
            (0, 2, -10., 5.),
            (2, 4, 0., 5.),
            (1, 2, -1., 1.),
            (2, 1, -1., 1.),
        ],
    )
}
#[test]
fn cpp_cycle_fixture_and_reconstruction() {
    let result = price(&cycle_graph(), &Controls::default()).unwrap();
    let best = &result.columns[0];
    assert_eq!(best.nodes, [0, 1, 2, 1, 4]);
    assert_eq!(best.arcs, [0, 4, 5, 1]);
    assert_eq!(best.cost, -7.);
    assert_eq!(best.resources, [4.]);
    assert!(result.exact);
}
#[test]
fn ng_memory_and_full_elementarity_prevent_repeated_visits() {
    let mut g = cycle_graph();
    g.set_ng(1, &[1, 2]).unwrap();
    g.set_ng(2, &[1, 2]).unwrap();
    let result = price(&g, &Controls::default()).unwrap();
    assert_eq!(result.minimum_reduced_cost, -5.);
    g.elementary = true;
    assert_eq!(
        price(&g, &Controls::default()).unwrap().columns[0].nodes,
        [0, 1, 4]
    );
}
#[test]
fn waiting_and_equality_are_different_transitions() {
    let mut g = graph(3, 1, &[(0, 1, -2., 1.), (1, 2, -3., 1.)]);
    g.set_bounds(1, 0, 3., 4.).unwrap();
    assert_eq!(
        price(&g, &Controls::default()).unwrap().columns[0].resources,
        [4.]
    );
    g.resources[0] = ResourceKind::Equality;
    assert!(price(&g, &Controls::default()).unwrap().columns.is_empty());
}
#[test]
fn backward_resource_clipping_happens_before_subtraction() {
    let mut g = graph(3, 1, &[(0, 1, -2., 1.), (1, 2, -3., 1.)]);
    g.set_bounds(0, 0, 0., 2.).unwrap();
    g.set_bounds(1, 0, 0., 3.).unwrap();
    assert_eq!(
        extend_resources(&g, &g.arcs[0], [4., 0.], false),
        Some([2., 0.].into())
    );
    g.resources[0] = ResourceKind::Equality;
    assert!(extend_resources(&g, &g.arcs[0], [4., 0.], false).is_none());
}
#[test]
fn sparse_physical_ids_never_index_label_or_arc_vectors() {
    let mut g = Graph::<2>::new(3, 0).unwrap();
    g.add_arc_with_id(100, 0, 1, -3.).unwrap();
    g.add_arc_with_id(7000, 1, 2, 1.).unwrap();
    assert_eq!(
        price(&g, &Controls::default()).unwrap().columns[0].arcs,
        [100, 7000]
    );
}
#[test]
fn dominance_is_local_to_a_vertex_and_preserves_cost_resource_tradeoffs() {
    let g = graph(
        5,
        1,
        &[
            (0, 1, -10., 3.),
            (0, 2, -1., 1.),
            (2, 1, 0., 1.),
            (1, 4, -3., 2.),
        ],
    );
    let r = price(&g, &Controls::default()).unwrap();
    assert_eq!(r.columns[0].nodes, [0, 2, 1, 4]);
    assert_eq!(r.columns[0].cost, -4.);
}
#[test]
fn label_limit_and_time_fail_without_partial_exact_results() {
    let g = cycle_graph();
    let c = Controls {
        max_labels: 2,
        ..Controls::default()
    };
    assert!(price(&g, &c).unwrap_err().message.contains("label limit"));
    let c = Controls {
        time_limit: Duration::ZERO,
        ..Controls::default()
    };
    assert!(price(&g, &c).unwrap_err().message.contains("time limit"));
    assert_eq!(
        price(&g, &Controls::default())
            .unwrap()
            .minimum_reduced_cost,
        -7.
    );
}
#[test]
fn zero_resource_negative_cycle_is_budgeted_not_certified() {
    let g = graph(
        4,
        0,
        &[
            (0, 1, 0., 0.),
            (1, 2, -1., 0.),
            (2, 1, -1., 0.),
            (2, 3, 0., 0.),
        ],
    );
    assert!(price(
        &g,
        &Controls {
            max_labels: 100,
            ..Controls::default()
        }
    )
    .is_err());
}
#[test]
fn hop_bound_is_part_of_dominance() {
    let g = graph(
        5,
        0,
        &[
            (0, 1, 0., 0.),
            (1, 2, -10., 0.),
            (0, 2, 1., 0.),
            (2, 3, 0., 0.),
            (3, 4, 0., 0.),
        ],
    );
    let r = price(
        &g,
        &Controls {
            max_route_arcs: Some(3),
            ..Controls::default()
        },
    )
    .unwrap();
    assert_eq!(r.columns[0].nodes, [0, 2, 3, 4]);
    assert_eq!(r.columns[0].cost, 1.);
}
#[test]
fn repricing_is_transactional_and_previous_routes_are_owned() {
    let mut g = graph(3, 0, &[(0, 1, 1., 0.), (1, 2, 1., 0.), (0, 2, 3., 0.)]);
    let saved = price(&g, &Controls::default()).unwrap().columns[0].clone();
    g.update_reduced_costs(&[1., 1., 3.], &[0., 5., 0.], true)
        .unwrap();
    assert_eq!(
        price(&g, &Controls::default())
            .unwrap()
            .minimum_reduced_cost,
        -3.
    );
    assert!(g
        .update_reduced_costs(&[1., 1., 3.], &[0., f64::NAN, 0.], true)
        .is_err());
    assert_eq!(
        price(&g, &Controls::default())
            .unwrap()
            .minimum_reduced_cost,
        -3.
    );
    assert_eq!(saved.cost, 2.);
}
#[test]
fn grouped_elements_forbid_different_vertices_with_the_same_element() {
    let mut g = graph(
        4,
        0,
        &[
            (0, 1, -3., 0.),
            (1, 2, -3., 0.),
            (2, 3, 0., 0.),
            (1, 3, 1., 0.),
        ],
    );
    g.add_elementarity_set(&[1, 2]).unwrap();
    g.elementary = true;
    assert_eq!(
        price(&g, &Controls::default()).unwrap().columns[0].nodes,
        [0, 1, 3]
    );
}
#[test]
fn arc_elements_are_checked_and_remembered() {
    let mut g = graph(
        4,
        0,
        &[
            (0, 1, -3., 0.),
            (1, 2, -3., 0.),
            (2, 3, 0., 0.),
            (1, 3, 1., 0.),
        ],
    );
    g.elementary = true;
    g.arcs[0].elements.push(2);
    assert_eq!(
        price(&g, &Controls::default()).unwrap().columns[0].nodes,
        [0, 1, 3]
    );
}
#[test]
fn input_validation_precedes_search_and_ignores_no_resources() {
    assert!(Graph::<2>::new(0, 0).is_err());
    assert!(Graph::<2>::new(3, 3).is_err());
    let mut g = graph(2, 1, &[(0, 1, 0., 1.)]);
    g.arcs[0].consumption[0] = f64::NAN;
    assert!(price(&g, &Controls::default()).is_err());
    g.arcs[0].consumption[0] = -1.;
    assert!(price(&g, &Controls::default()).is_err());
    g.arcs[0].consumption[0] = 1.;
    g.source = g.sink;
    assert!(price(&g, &Controls::default()).is_err());
}

// Independent complete walk enumeration: no dominance, queues, NG tables or
// predecessor machinery shared with the implementation under test.
fn oracle(g: &Graph<2>) -> Option<f64> {
    fn walk(
        g: &Graph<2>,
        v: usize,
        q: [f64; 2],
        seen: u64,
        cost: f64,
        depth: usize,
        best: &mut Option<f64>,
    ) {
        if v == g.sink {
            if best.is_none_or(|b| cost < b) {
                *best = Some(cost);
            }
            return;
        }
        if depth >= 6 {
            return;
        }
        for a in &g.arcs {
            if a.tail != v || a.head == g.source || (g.elementary && seen & (1 << a.head) != 0) {
                continue;
            }
            let mut next = q;
            let mut feasible = true;
            for (r, value) in next.iter_mut().enumerate().take(g.resources.len()) {
                *value += a.consumption[r];
                let w = g.vertices[a.head].bounds[r];
                if *value < w.lower {
                    if g.resources[r] == ResourceKind::Equality {
                        feasible = false;
                        break;
                    }
                    *value = w.lower;
                }
                if *value > w.upper {
                    feasible = false;
                    break;
                }
            }
            if feasible {
                walk(
                    g,
                    a.head,
                    next,
                    seen | (1 << a.head),
                    cost + a.reduced_cost,
                    depth + 1,
                    best,
                );
            }
        }
    }
    let mut best = None;
    let mut q = [0.; 2];
    for (r, value) in q.iter_mut().enumerate().take(g.resources.len()) {
        *value = g.vertices[g.source].bounds[r].lower;
    }
    walk(g, g.source, q, 1 << g.source, 0., 0, &mut best);
    best
}
#[test]
fn randomized_small_graphs_match_independent_complete_enumeration() {
    let mut rng = baldes::rng::Xoroshiro128Plus::from_seed(0x9e3779b97f4a7c15u64);
    for case in 0..160 {
        let mut next = || rng.next() >> 32;
        let mut g = Graph::<2>::new(6, 2).unwrap();
        g.elementary = case >= 80;
        if case % 3 == 0 {
            g.resources[1] = ResourceKind::Equality;
        }
        for v in 0..6 {
            g.set_bounds(v, 0, if v == 0 { 0. } else { (next() % 3) as f64 }, 5.)
                .unwrap();
            g.set_bounds(v, 1, 0., 4.).unwrap();
        }
        for tail in 0..5 {
            for head in 1..6 {
                if tail == head || (!g.elementary && head <= tail) || next() % 3 == 0 {
                    continue;
                }
                let a = g
                    .add_arc(tail, head, (next() % 21) as f64 / 4. - 3.)
                    .unwrap();
                g.arc_mut(a).unwrap().consumption = [1., (next() % 2) as f64].into();
            }
        }
        let expected = oracle(&g);
        let actual = price(&g, &Controls::default()).unwrap();
        match expected {
            Some(cost) => assert!(
                (cost - actual.minimum_reduced_cost).abs() < 1e-12,
                "case {case}: {cost} vs {}",
                actual.minimum_reduced_cost
            ),
            None => assert!(actual.columns.is_empty(), "case {case}"),
        }
        for route in &actual.columns {
            assert_eq!(route.nodes.first(), Some(&(g.source as i32)));
            assert_eq!(route.nodes.last(), Some(&(g.sink as i32)));
            assert_eq!(route.nodes.len(), route.arcs.len() + 1);
            let replay: f64 = route
                .arcs
                .iter()
                .map(|id| g.arcs.iter().find(|a| a.id == *id).unwrap().reduced_cost)
                .sum();
            assert_eq!(replay, route.cost);
        }
    }
}

fn scalar_dimensions_fixture<const D: usize>() -> Graph<D> {
    let dimensions = D;
    let mut g = Graph::<D>::new(6, dimensions).unwrap();
    g.elementary = true;
    for v in 0..6 {
        for r in 0..dimensions {
            g.set_bounds(v, r, 0., 10.).unwrap();
        }
    }
    // At vertex 3, the cheaper prefix consumes too much of the last resource
    // to reach the sink. Dominance must retain the more expensive prefix.
    for (u, v, cost, use_) in [
        (0, 1, -9., 8.),
        (1, 3, 0., 0.),
        (0, 2, -5., 2.),
        (2, 3, 0., 0.),
        (3, 4, 0., 0.),
        (4, 5, 0., 4.),
    ] {
        let id = g.add_arc(u, v, cost).unwrap();
        g.arc_mut(id).unwrap().consumption[dimensions - 1] = use_;
    }
    g
}

#[test]
fn all_scalar_dimensions_participate_in_dominance_and_reconstruction() {
    use baldes::labeling::engine::{EngineControls, LabelingEngine};
    use baldes::pricing::phases::Phase;
    fn check<const D: usize>() {
        let dimensions = D;
        let mut g = scalar_dimensions_fixture::<D>();
        for main in [0, dimensions - 1] {
            g.main_resource = Some(main);
            for bidirectional in [false, true] {
                let mut engine = LabelingEngine::prepare(
                    &g,
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
                let result = engine.solve(&g, &Controls::default()).unwrap();
                assert_eq!(
                    result.minimum_reduced_cost, -5.,
                    "dimensions={dimensions}, main={main}, bidirectional={bidirectional}"
                );
                let route = &result.columns[0];
                assert_eq!(route.nodes, [0, 2, 3, 4, 5]);
                assert_eq!(route.resources.len(), dimensions);
                assert_eq!(route.resources[dimensions - 1], 6.);
                assert!(route
                    .resource_profile
                    .iter()
                    .all(|state| state.len() == dimensions));
            }
        }
    }
    check::<3>();
    check::<4>();
    check::<9>();
}

#[test]
fn extra_equality_resources_and_arc_windows_are_enforced() {
    let mut g = scalar_dimensions_fixture::<4>();
    g.resources[3] = ResourceKind::Equality;
    g.set_bounds(g.sink, 3, 6., 6.).unwrap();
    let result = price(&g, &Controls::default()).unwrap();
    assert_eq!(result.minimum_reduced_cost, -5.);
    g.arcs.last_mut().unwrap().bounds[3] = Some(baldes::Window::new(7., 8.).unwrap());
    assert!(price(&g, &Controls::default()).unwrap().columns.is_empty());
}

#[test]
fn general_dimensions_match_independent_dag_path_search() {
    // Exhaustive search intentionally uses separate scalar transition code.
    fn visit<const D: usize>(g: &Graph<D>, vertex: usize, q: &[f64], cost: f64, best: &mut f64) {
        if vertex == g.sink {
            *best = best.min(cost);
            return;
        }
        for a in g.arcs.iter().filter(|a| a.tail == vertex) {
            let mut next = q.to_vec();
            let mut feasible = true;
            for r in 0..g.resources.len() {
                let w = g.vertices[a.head].bounds[r];
                next[r] = (next[r] + a.consumption[r]).max(w.lower);
                if next[r] > w.upper {
                    feasible = false;
                    break;
                }
            }
            if feasible {
                visit(g, a.head, &next, cost + a.cost, best);
            }
        }
    }
    fn check<const D: usize>() {
        let dimensions = D;
        let mut rng = baldes::rng::Xoroshiro128Plus::from_seed(739u64);
        for _ in 0..20 {
            let mut next = || rng.next() >> 32;
            let mut g = Graph::<D>::new(7, dimensions).unwrap();
            for v in 0..7 {
                for r in 0..dimensions {
                    g.set_bounds(v, r, 0., 8.).unwrap();
                }
            }
            for u in 0..6 {
                for v in u + 1..7 {
                    let id = g.add_arc(u, v, (next() % 17) as f64 - 9.).unwrap();
                    for r in 0..dimensions {
                        g.arc_mut(id).unwrap().consumption[r] = (next() % 5) as f64;
                    }
                }
            }
            let mut best = NO_ROUTE_COST;
            visit(&g, 0, &vec![0.; dimensions], 0., &mut best);
            let actual = price(&g, &Controls::default()).unwrap();
            assert_eq!(actual.minimum_reduced_cost, best, "dimensions={dimensions}");
        }
    }
    check::<3>();
    check::<5>();
    check::<8>();
}

#[test]
fn general_dimensions_work_with_cuts_enumeration_and_path_domains() {
    use baldes::cuts::state::LimitedInput;
    use baldes::enumeration::{
        CompleteCanonicalPaths, CompleteResourceRelaxation, Enumeration, PathDomainLimits,
    };
    use baldes::labeling::engine::{EngineControls, LabelingEngine};
    use baldes::pricing::phases::Phase;
    let g = scalar_dimensions_fixture::<5>();
    let canonical = CompleteCanonicalPaths::collect(&g, PathDomainLimits::default()).unwrap();
    assert_eq!(canonical.paths().len(), 1);
    assert!(canonical.paths()[0]
        .resource_profile()
        .iter()
        .all(|q| q.len() == 5));
    let relaxed = CompleteResourceRelaxation::collect(&g, PathDomainLimits::default()).unwrap();
    assert_eq!(relaxed.paths().len(), 1);
    let phase = Phase {
        check_ng: true,
        jump_arcs: true,
        labels_per_bucket: i64::MAX,
    };
    for bidirectional in [false, true] {
        let mut engine = LabelingEngine::prepare(
            &g,
            EngineControls {
                bidirectional,
                ..Default::default()
            },
            phase,
        )
        .unwrap();
        let cut = LimitedInput {
            covering: false,
            type2: false,
            arc_memory: false,
            denominator: 2,
            dual: 2.,
            vertex_coefficient: vec![0, 0, 1, 1, 0, 0],
            arc_coefficient: Default::default(),
            memory_vertices: (0..6).collect(),
            memory_arcs: Default::default(),
        };
        assert!(engine.load_cuts(&g, vec![cut], vec![]));
        let result = engine.solve(&g, &Controls::default()).unwrap();
        assert_eq!(result.minimum_reduced_cost, -3.);
    }
    let engine = LabelingEngine::prepare(&g, EngineControls::default(), phase).unwrap();
    let controls = baldes::enumeration::Controls {
        cutoff: 1.,
        preserve_arc_paths: true,
        ..Default::default()
    };
    let paths = Enumeration::run(&g, &engine.grid, &controls, None)
        .unwrap()
        .paths
        .routes(&g)
        .unwrap();
    assert_eq!(paths.len(), 1);
    assert_eq!(paths[0].resources[4], 6.);
    let paths = Enumeration::run_directional(&g, &engine.grid, &controls, |_, _| true)
        .unwrap()
        .paths
        .routes(&g)
        .unwrap();
    assert_eq!(paths.len(), 1);
    assert_eq!(paths[0].resources[4], 6.);
}

#[test]
fn models_cannot_exceed_their_compiled_capacity() {
    assert!(Graph::<4>::new(3, 5).is_err());
    let mut g = Graph::<4>::new(3, 4).unwrap();
    assert!(g.add_resource(ResourceKind::Disposable).is_err());
    // Public metadata is validated before any scalar coordinate is accessed.
    g.resources.push(ResourceKind::Disposable);
    assert!(price(&g, &Controls::default()).is_err());
}

#[test]
fn extra_resource_branch_survives_enumeration_and_forks() {
    use baldes::pricing::{restrictions::AccumulatedResourceBranch, Solver};
    let mut solver = Solver::<5>::new(6, 5).unwrap();
    solver.graph = scalar_dimensions_fixture::<5>();
    assert_eq!(
        solver
            .price_phase(0, 20, false)
            .unwrap()
            .minimum_reduced_cost,
        -5.
    );
    let paths = solver.enumerate(1.).unwrap();
    assert_eq!(paths.len(), 1);
    assert_eq!(paths[0].resources[4], 6.);
    let mut child = solver.fork_for_node();
    child
        .add_accumulated_resource_branch(AccumulatedResourceBranch {
            arc: 2,
            resource: 4,
            threshold: 1.,
            upper: false,
        })
        .unwrap();
    assert!(child.price_phase(0, 20, false).unwrap().columns.is_empty());
    assert_eq!(
        solver
            .price_phase(0, 20, false)
            .unwrap()
            .minimum_reduced_cost,
        -5.
    );
}

#[test]
fn knapsack_charges_can_use_extra_resource_coordinates() {
    use baldes::cuts::knapsack::{KnapsackKind, ResourceKnapsackCut, RouteLoadKnapsackCut};
    use baldes::pricing::Solver;
    let mut solver = Solver::<5>::new(6, 5).unwrap();
    solver.graph = scalar_dimensions_fixture::<5>();
    solver.cuts.knapsack_cuts.push(ResourceKnapsackCut {
        id: 0,
        row: 0,
        cut_class: 0,
        graph_resources: std::collections::BTreeMap::from([(0, (4, 1.))]),
        dual: -1.,
        cut: RouteLoadKnapsackCut {
            kind: KnapsackKind::Steps,
            numerator: 1,
            denominator: 1,
            rhs: 3,
            lhs: 0.,
            violation: 0.,
            thresholds: vec![6],
            coefficients: std::collections::BTreeMap::from([(6, 3)]),
        },
    });
    let result = solver.price_phase(0, 20, false).unwrap();
    assert_eq!(result.minimum_reduced_cost, -2.);
    let paths = solver.enumerate(1.).unwrap();
    assert_eq!(paths.len(), 1);
    assert_eq!(paths[0].cost, -2.);
}

#[test]
fn signed_extra_equality_resource_joins_at_nonzero_initial_state() {
    let mut g = Graph::<4>::new(4, 4).unwrap();
    for vertex in 0..4 {
        for r in 0..4 {
            g.set_bounds(vertex, r, 0., 10.).unwrap();
        }
    }
    g.resources[3] = ResourceKind::Equality;
    g.initial[3] = Some(5.);
    g.set_bounds(0, 3, 5., 5.).unwrap();
    g.set_bounds(1, 3, 3., 3.).unwrap();
    g.set_bounds(3, 3, 4., 4.).unwrap();
    for (u, v, cost, consumption) in [
        (0, 1, -2., -2.),
        (1, 3, 0., 1.),
        (0, 2, -9., -1.),
        (2, 3, 0., 1.),
    ] {
        let id = g.add_arc(u, v, cost).unwrap();
        g.arc_mut(id).unwrap().consumption[3] = consumption;
    }
    let result = price(&g, &Controls::default()).unwrap();
    assert_eq!(result.minimum_reduced_cost, -2.);
    assert_eq!(result.columns[0].nodes, [0, 1, 3]);
    assert_eq!(result.columns[0].resources[3], 4.);
    assert_eq!(
        result.columns[0]
            .resource_profile
            .iter()
            .map(|q| q[3])
            .collect::<Vec<_>>(),
        [5., 3., 4.]
    );
}

#[test]
fn zero_and_one_compiled_dimension_models_price() {
    fn check<const D: usize>() {
        let mut graph = Graph::<D>::new(3, D).unwrap();
        let first = graph.add_arc(0, 1, -2.).unwrap();
        let second = graph.add_arc(1, 2, -3.).unwrap();
        for r in 0..D {
            for v in 0..3 {
                graph.set_bounds(v, r, 0., 10.).unwrap();
            }
            graph.arc_mut(first).unwrap().consumption[r] = 1.;
            graph.arc_mut(second).unwrap().consumption[r] = 2.;
        }
        let result = price(&graph, &Controls::default()).unwrap();
        assert_eq!(result.minimum_reduced_cost, -5.);
        assert_eq!(result.columns[0].resources, vec![3.; D]);
    }
    check::<0>();
    check::<1>();
}

#[test]
fn engine_label_budgets_remain_enforced_after_unlimited_sweeps() {
    use baldes::labeling::engine::{EngineControls, LabelingEngine};
    use baldes::pricing::phases::Phase;
    let g = scalar_dimensions_fixture::<3>();
    for bidirectional in [false, true] {
        let mut engine = LabelingEngine::prepare(
            &g,
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
        let unlimited = engine.solve(&g, &Controls::default()).unwrap();
        assert_eq!(unlimited.minimum_reduced_cost, -5.);
        let error = engine
            .solve(
                &g,
                &Controls {
                    max_labels: 2,
                    ..Default::default()
                },
            )
            .unwrap_err();
        assert!(error.message.contains("label limit"), "{error}");
        let bounded = engine
            .solve(
                &g,
                &Controls {
                    max_labels: 100_000,
                    ..Default::default()
                },
            )
            .unwrap();
        assert_eq!(bounded.columns, unlimited.columns);
        let repeated = engine.solve(&g, &Controls::default()).unwrap();
        assert_eq!(repeated.columns, unlimited.columns);
    }
}

#[path = "presence.rs"]
mod presence_tests;

#[test]
fn bucket_snapshot_eligibility_preserves_hop_limits_and_finite_budgets() {
    use baldes::labeling::engine::{EngineControls, LabelingEngine};
    use baldes::pricing::phases::Phase;

    let g = cycle_graph();
    for bidirectional in [false, true] {
        for max_route_arcs in [None, Some(3), Some(4)] {
            let mut engine = LabelingEngine::prepare(
                &g,
                EngineControls {
                    bidirectional,
                    use_symmetric_execution: false,
                    steps: [10., 0.].into(),
                    target_buckets: 1,
                    ..Default::default()
                },
                Phase {
                    check_ng: true,
                    jump_arcs: true,
                    labels_per_bucket: i64::MAX,
                },
            )
            .unwrap();
            let controls = Controls {
                max_route_arcs,
                ..Default::default()
            };
            let unlimited = engine.solve(&g, &controls).unwrap();
            assert_eq!(
                unlimited.minimum_reduced_cost,
                if max_route_arcs == Some(3) { -5. } else { -7. }
            );
            let bounded = engine
                .solve(
                    &g,
                    &Controls {
                        max_labels: 100_000,
                        ..controls.clone()
                    },
                )
                .unwrap();
            assert_eq!(bounded.columns, unlimited.columns);
            let error = engine
                .solve(
                    &g,
                    &Controls {
                        max_labels: 2,
                        ..controls.clone()
                    },
                )
                .unwrap_err();
            assert!(error.message.contains("label limit"), "{error}");
            let repeated = engine.solve(&g, &controls).unwrap();
            assert_eq!(repeated.columns, unlimited.columns);
        }
    }
}
