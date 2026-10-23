use baldes::graph::Window;
use baldes::labeling::{extend_resources, reference_price};
use baldes::{price, Controls, Graph};

fn windows(pairs: &[(f64, f64)]) -> Vec<Window> {
    pairs
        .iter()
        .map(|&(a, b)| Window::new(a, b).unwrap())
        .collect()
}
fn chain() -> Graph<2> {
    let mut g = Graph::new(4, 1).unwrap();
    for v in 0..4 {
        g.set_bounds(v, 0, 0., 30.).unwrap();
    }
    for (u, v, c, t) in [(0, 1, -3., 4.), (1, 2, -4., 3.), (2, 3, -2., 2.)] {
        let id = g.add_arc(u, v, c).unwrap();
        g.arc_mut(id).unwrap().consumption[0] = t;
    }
    g.set_time_windows(1, 0, &windows(&[(0., 2.), (8., 12.)]))
        .unwrap();
    g.set_time_windows(2, 0, &windows(&[(0., 5.), (15., 20.)]))
        .unwrap();
    g
}

#[test]
fn forward_waits_across_gaps_and_backward_selects_latest_window() {
    let g = chain();
    assert_eq!(
        extend_resources(&g, &g.arcs[0], [0., 0.], true).unwrap()[0],
        8.
    );
    assert_eq!(
        extend_resources(&g, &g.arcs[1], [8., 0.], true).unwrap()[0],
        15.
    );
    // The old ascending backward loop would choose the first interval.
    assert_eq!(
        extend_resources(&g, &g.arcs[2], [30., 0.], false).unwrap()[0],
        20.
    );
    assert_eq!(
        extend_resources(&g, &g.arcs[1], [20., 0.], false).unwrap()[0],
        12.
    );
}

#[test]
fn bucket_pricing_matches_reference_and_reconstructs_admissible_service_starts() {
    let g = chain();
    let result = price(&g, &Controls::default()).unwrap();
    let reference = reference_price(&g, &Controls::default()).unwrap();
    assert!(result.exact);
    assert_eq!(result.minimum_reduced_cost, reference.minimum_reduced_cost);
    assert_eq!(result.columns[0].nodes, [0, 1, 2, 3]);
    assert_eq!(
        result.columns[0].resource_profile,
        vec![vec![0.], vec![8.], vec![15.], vec![17.]]
    );
}

#[test]
fn arc_bounds_intersect_windows_and_never_fill_a_gap() {
    let mut g = chain();
    g.arcs[0].bounds[0] = Some(Window::new(3., 7.).unwrap());
    assert!(extend_resources(&g, &g.arcs[0], [0., 0.], true).is_none());
    assert!(extend_resources(&g, &g.arcs[0], [20., 0.], false).is_none());
    assert!(price(&g, &Controls::default()).unwrap().columns.is_empty());
}

#[test]
fn normalization_is_transactional_and_topology_tracks_window_changes() {
    let mut g = chain();
    g.set_time_windows(1, 0, &windows(&[(8., 10.), (2., 4.), (4., 6.), (9., 12.)]))
        .unwrap();
    assert_eq!(g.time_windows(1, 0), windows(&[(2., 6.), (8., 12.)]));
    let before = g.clone();
    assert!(g.set_time_windows(1, 0, &[]).is_err());
    assert!(g
        .set_time_windows(
            1,
            0,
            &[Window {
                lower: f64::NAN,
                upper: 1.
            }]
        )
        .is_err());
    assert_eq!(g, before);
    g.set_time_windows(1, 0, &windows(&[(2., 5.), (8., 12.)]))
        .unwrap();
    assert!(!g.same_topology(&before));
    g.resources[0] = baldes::ResourceKind::Equality;
    assert!(price(&g, &Controls::default()).is_err());
}

#[test]
fn source_sink_unions_and_later_bound_tightening_are_enforced() {
    let mut g = chain();
    g.set_time_windows(g.source, 0, &windows(&[(3., 4.), (6., 7.)]))
        .unwrap();
    g.set_time_windows(g.sink, 0, &windows(&[(0., 16.), (22., 25.)]))
        .unwrap();
    let route = &price(&g, &Controls::default()).unwrap().columns[0];
    assert_eq!(route.resource_profile[0], [3.]);
    assert_eq!(route.resources, [22.]);
    g.initial[0] = Some(5.);
    assert!(price(&g, &Controls::default()).is_err());
    g.initial[0] = None;
    g.set_bounds(g.source, 0, 5., 7.).unwrap();
    let result = price(&g, &Controls::default()).unwrap();
    assert_eq!(result.columns[0].resource_profile[0], [6.]);
    assert_eq!(result.columns[0].resources, [22.]);
}

// Independent small-integer earliest schedule, without the production resource
// extension, union projection, bucket graph or dominance implementations.
fn brute<const D: usize>(g: &Graph<D>) -> f64 {
    fn visit<const D: usize>(
        g: &Graph<D>,
        v: usize,
        q: [f64; D],
        cost: f64,
        best: &mut f64,
        seen: u64,
    ) {
        if v == g.sink {
            *best = best.min(cost);
            return;
        }
        for a in g
            .arcs
            .iter()
            .filter(|a| a.tail == v && seen & (1 << a.head) == 0)
        {
            let mut next = q;
            let mut feasible = true;
            for r in 0..g.resources.len() {
                let bound = g.vertices[a.head].bounds[r];
                let arc_bound = a.bounds[r].unwrap_or(bound);
                let arrival = q[r] + a.consumption[r];
                let union = g.time_windows(a.head, r);
                let choices = if union.is_empty() {
                    std::slice::from_ref(&bound)
                } else {
                    union
                };
                let value = choices
                    .iter()
                    .filter_map(|w| {
                        let low = w.lower.max(bound.lower).max(arc_bound.lower);
                        let high = w.upper.min(bound.upper).min(arc_bound.upper);
                        let t = arrival.max(low);
                        (t <= high).then_some(t)
                    })
                    .min_by(f64::total_cmp);
                match value {
                    Some(t) => next[r] = t,
                    None => {
                        feasible = false;
                        break;
                    }
                }
            }
            if feasible {
                visit(
                    g,
                    a.head,
                    next,
                    cost + a.reduced_cost,
                    best,
                    seen | (1 << a.head),
                );
            }
        }
    }
    let mut best = baldes::labeling::NO_ROUTE_COST;
    let q = std::array::from_fn(|r| {
        if r < g.resources.len() {
            g.initial[r].unwrap_or(g.vertices[g.source].bounds[r].lower)
        } else {
            0.
        }
    });
    visit(g, g.source, q, 0., &mut best, 1 << g.source);
    best
}

#[test]
fn randomized_bucket_pricing_matches_independent_integer_path_search() {
    use baldes::labeling::engine::{EngineControls, LabelingEngine};
    use baldes::pricing::phases::Phase;
    let mut rng = baldes::rng::Xoroshiro128Plus::from_seed(0x4d54575f2026u64);
    let mut random = |n: u64| rng.uniform_below(n);
    for case in 0..120 {
        let mut g = Graph::<2>::new(7, 2).unwrap();
        g.elementary = true;
        for v in 0..7 {
            for r in 0..2 {
                g.set_bounds(v, r, 0., 18.).unwrap();
                if v > 0 && random(3) != 0 {
                    g.set_time_windows(v, r, &windows(&[(0., 2.), (5., 8.), (12., 18.)]))
                        .unwrap();
                }
            }
        }
        for tail in 0..6 {
            for head in 1..7 {
                if head == tail || (case < 80 && head <= tail) {
                    continue;
                }
                if head != tail + 1 && random(3) == 0 {
                    continue;
                }
                let id = g.add_arc(tail, head, -((random(9) + 1) as f64)).unwrap();
                let a = g.arc_mut(id).unwrap();
                a.consumption[0] = random(6) as f64;
                a.consumption[1] = random(6) as f64;
                if random(4) == 0 {
                    a.bounds[0] = Some(Window::new(3., 15.).unwrap());
                }
            }
        }
        let expected = brute(&g);
        for bidirectional in [false, true] {
            let mut engine = LabelingEngine::prepare(
                &g,
                EngineControls {
                    bidirectional,
                    target_buckets: 16,
                    steps: [if case % 2 == 0 { 3. } else { 5. }, 6.].into(),
                    ..Default::default()
                },
                Phase {
                    check_ng: true,
                    jump_arcs: true,
                    labels_per_bucket: i64::MAX,
                },
            )
            .unwrap();
            assert!(!engine.consumption_joins);
            let result = engine.solve(&g, &Controls::default()).unwrap();
            assert_eq!(
                result.minimum_reduced_cost, expected,
                "case={case} bidirectional={bidirectional}"
            );
            for route in result.columns {
                for (&vertex, q) in route.nodes.iter().zip(&route.resource_profile) {
                    for (r, &t) in q.iter().enumerate() {
                        let union = g.time_windows(vertex as usize, r);
                        assert!(
                            union.is_empty() || union.iter().any(|w| w.lower <= t && t <= w.upper)
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn enumeration_and_canonical_collection_obey_window_unions() {
    use baldes::enumeration::{CompleteCanonicalPaths, Enumeration, PathDomainLimits};
    let mut g = chain();
    let bypass = g.add_arc(0, 2, -1.).unwrap();
    g.arc_mut(bypass).unwrap().consumption[0] = 7.;
    let grid = baldes::graph::buckets::BucketGraph::prepare(&g, [3., 0.], 10, true).unwrap();
    let controls = baldes::enumeration::Controls {
        cutoff: 100.,
        preserve_arc_paths: true,
        ..Default::default()
    };
    let canonical = CompleteCanonicalPaths::collect(&g, PathDomainLimits::default()).unwrap();
    assert_eq!(canonical.paths().len(), 2);
    let forward = Enumeration::run(&g, &grid, &controls, None).unwrap();
    let directional = Enumeration::run_directional(&g, &grid, &controls, |_, _| true).unwrap();
    for enumeration in [forward, directional] {
        let routes = enumeration.paths.routes(&g).unwrap();
        assert_eq!(routes.len(), canonical.paths().len());
        assert!(routes
            .iter()
            .all(|r| r.resource_profile[r.nodes.iter().position(|&v| v == 2).unwrap()][0] == 15.));
    }
}

#[test]
fn nonindexed_resource_windows_and_zero_width_endpoints_are_enforced() {
    let mut g = Graph::<3>::new(3, 3).unwrap();
    g.resources[1] = baldes::ResourceKind::Equality;
    for v in 0..3 {
        g.set_bounds(v, 0, 0., 20.).unwrap();
        g.set_bounds(v, 1, 0., 0.).unwrap();
        g.set_bounds(v, 2, 0., 20.).unwrap();
    }
    for (u, v) in [(0, 1), (1, 2)] {
        g.add_arc(u, v, -1.).unwrap();
    }
    g.arcs[0].consumption[2] = 3.;
    g.set_time_windows(1, 2, &windows(&[(1., 1.), (7., 7.)]))
        .unwrap();
    let result = price(&g, &Controls::default()).unwrap();
    assert_eq!(result.columns[0].resource_profile[1], [0., 0., 7.]);
    g.arcs[0].bounds[2] = Some(Window::new(0., 6.).unwrap());
    assert!(price(&g, &Controls::default()).unwrap().columns.is_empty());
}

#[cfg(feature = "clp")]
#[test]
fn network_json_loads_and_intersects_unions() {
    use baldes::master::mp::format::Model;
    let text = r#"{"rows":[],"networks":[{"id":0,"vertices":3,"multiplicity":[0,1],
      "resources":[{"time_windows":[null,[[8,12],[0,2]],null],"bounds":[[0,20],[0,10],[0,20]]}],
      "arcs":[{"tail":0,"head":1,"cost":-1,"consumption":[4]},
                {"tail":1,"head":2,"cost":0,"consumption":[2]}]}]}"#;
    let p = Model::from_json(text).unwrap().problem::<2>().unwrap();
    let g = &p.networks[0].graph;
    assert_eq!(g.time_windows(1, 0), windows(&[(0., 2.), (8., 12.)]));
    let result = price(g, &Controls::default()).unwrap();
    assert_eq!(result.columns[0].resource_profile[1], [8.]);
    assert_eq!(result.columns[0].resources, [10.]);
}

#[test]
fn window_endpoints_are_closed_without_extending_them_by_tolerance() {
    let mut g = chain();
    g.arcs[0].consumption[0] = 0.;
    let upper = 2f64;
    let above = f64::from_bits(upper.to_bits() + 1);
    let lower = 8f64;
    let below = f64::from_bits(lower.to_bits() - 1);
    assert_eq!(
        extend_resources(&g, &g.arcs[0], [upper, 0.], true).unwrap()[0],
        upper
    );
    assert_eq!(
        extend_resources(&g, &g.arcs[0], [above, 0.], true).unwrap()[0],
        lower
    );
    g.clear_time_windows(2, 0);
    assert_eq!(
        extend_resources(&g, &g.arcs[1], [lower + 3., 0.], false).unwrap()[0],
        lower
    );
    // Restrict the head to the first interval, then reverse to a tail bound
    // lying one representable value below the second interval's lower edge.
    g.arcs[1].consumption[0] = 0.;
    assert_eq!(
        extend_resources(&g, &g.arcs[1], [below, 0.], false).unwrap()[0],
        upper
    );
}

#[test]
fn empty_bound_intersection_is_infeasible_and_clearing_restores_ordinary_bounds() {
    let mut g = chain();
    g.set_bounds(1, 0, 3., 7.).unwrap();
    assert!(price(&g, &Controls::default()).unwrap().columns.is_empty());
    g.clear_time_windows(1, 0);
    assert_eq!(g.vertices[1].bounds[0], Window::new(3., 7.).unwrap());
    assert!(!price(&g, &Controls::default()).unwrap().columns.is_empty());
}

#[test]
fn repricing_rebuilds_after_a_union_change_with_the_same_hull() {
    let mut solver = baldes::pricing::Solver::<2>::new(4, 1).unwrap();
    solver.graph = chain();
    let first = solver.price(150, false).unwrap();
    assert_eq!(first.columns[0].resource_profile[1], [8.]);
    // Keep the hull [0,12], changing only its interior domain.
    solver
        .graph
        .set_time_windows(1, 0, &windows(&[(0., 6.), (10., 12.)]))
        .unwrap();
    let second = solver.price(150, false).unwrap();
    assert_eq!(second.columns[0].resource_profile[1], [4.]);
    assert_eq!(first.columns[0].resource_profile[1], [8.]);
}

#[test]
fn incoming_hull_ending_in_a_sink_gap_keeps_the_correct_backward_root() {
    let mut g = Graph::<2>::new(3, 1).unwrap();
    for v in 0..3 {
        g.set_bounds(v, 0, 0., 20.).unwrap();
    }
    for (u, v) in [(0, 1), (1, 2)] {
        let a = g.add_arc(u, v, -1.).unwrap();
        g.arc_mut(a).unwrap().consumption[0] = 1.;
    }
    g.set_time_windows(2, 0, &windows(&[(0., 2.), (8., 10.)]))
        .unwrap();
    g.arcs[1].bounds[0] = Some(Window::new(0., 7.).unwrap());
    let result = price(&g, &Controls::default()).unwrap();
    assert_eq!(result.minimum_reduced_cost, -2.);
    assert_eq!(result.columns[0].resources, [2.]);
    g.arcs[1].bounds[0] = Some(Window::new(3., 7.).unwrap());
    assert!(price(&g, &Controls::default()).unwrap().columns.is_empty());
}

#[test]
fn halfway_continuation_keeps_backward_crossings_inside_a_discontinuous_cell() {
    use baldes::labeling::engine::{EngineControls, LabelingEngine};
    use baldes::pricing::phases::Phase;
    let mut g = Graph::<2>::new(4, 1).unwrap();
    for v in 0..4 {
        g.set_bounds(v, 0, 0., 20.).unwrap();
    }
    for (u, v) in [(0, 1), (1, 2), (2, 3)] {
        g.add_arc(u, v, -1.).unwrap();
    }
    g.set_time_windows(1, 0, &windows(&[(0., 2.), (18., 18.)]))
        .unwrap();
    g.arcs[2].bounds[0] = Some(Window::new(0., 17.5).unwrap());
    let mut engine = LabelingEngine::prepare(
        &g,
        EngineControls {
            steps: [1., 0.].into(),
            ..Default::default()
        },
        Phase {
            check_ng: true,
            jump_arcs: true,
            labels_per_bucket: i64::MAX,
        },
    )
    .unwrap();
    let controls = Controls::default();
    assert_eq!(
        engine.solve(&g, &controls).unwrap().minimum_reduced_cost,
        -3.
    );
    engine.continue_after_halfway(&g, &controls).unwrap();
    assert!(engine.full_ready);
    let internal = engine.grid.internal[1];
    assert!(engine.backward.buckets[internal]
        .iter()
        .flat_map(|b| &b.active)
        .any(|&id| engine.backward.labels[id].resources[0] == 2.));
}
