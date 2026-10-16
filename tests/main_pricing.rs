use baldes::graph::ng::PreparedNg;
use baldes::labeling::{extend_resources, NO_ROUTE_COST};
use baldes::{price, Controls, Graph, ResourceKind};
use std::time::Duration;

fn graph(nodes: usize, resources: usize, arcs: &[(usize, usize, f64, f64)]) -> Graph {
    let mut g = Graph::new(nodes, resources).unwrap();
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
fn cycle_graph() -> Graph {
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
        Some([2., 0.])
    );
    g.resources[0] = ResourceKind::Equality;
    assert!(extend_resources(&g, &g.arcs[0], [4., 0.], false).is_none());
}
#[test]
fn sparse_physical_ids_never_index_label_or_arc_vectors() {
    let mut g = Graph::new(3, 0).unwrap();
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
fn empty_and_positive_pools_keep_the_native_minimum_contract() {
    let g = graph(3, 0, &[(0, 1, 1., 0.)]);
    let r = price(&g, &Controls::default()).unwrap();
    assert!(r.columns.is_empty());
    assert_eq!(r.minimum_reduced_cost, NO_ROUTE_COST);
    assert!(r.exact);
    let g = graph(2, 0, &[(0, 1, 2., 0.)]);
    let r = price(
        &g,
        &Controls {
            negative_only: true,
            ..Controls::default()
        },
    )
    .unwrap();
    assert!(r.columns.is_empty());
    assert_eq!(r.minimum_reduced_cost, 2.);
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
fn native_local_ng_capacity_is_enforced_instead_of_truncating_bits() {
    let mut g = Graph::new(17, 0).unwrap();
    g.elementary = true;
    assert!(PreparedNg::new(&g).unwrap_err().message.contains("fifteen"));
}
#[test]
fn input_validation_precedes_search_and_ignores_no_resources() {
    assert!(Graph::new(0, 0).is_err());
    assert!(Graph::new(3, 3).is_err());
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
fn oracle(g: &Graph) -> Option<f64> {
    fn walk(
        g: &Graph,
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
    let mut seed = 0x9e3779b97f4a7c15u64;
    for case in 0..160 {
        let mut next = || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            seed >> 32
        };
        let mut g = Graph::new(6, 2).unwrap();
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
                g.arc_mut(a).unwrap().consumption = [1., (next() % 2) as f64];
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
