use super::*;
use crate::graph::ResourceKind;
#[test]
fn acyclic_resource_minima_match_independent_quarter_grid_paths() {
    let mut compared = 0;
    for kind in [ResourceKind::Disposable, ResourceKind::Equality] {
        for capacity in [4, 6] {
            for first in [1, 2, 4] {
                for second in [1, 2, 4] {
                    for direct in [1, 2, 4] {
                        for costs in [[-3., 2., 1., 4., -2.], [2., -4., 3., -1., 5.]] {
                            let mut graph = Graph::<5>::new(4, 5).unwrap();
                            graph.resources[4] = kind;
                            for v in 0..4 {
                                for r in 0..5 {
                                    graph.set_bounds(v, r, 0., capacity as f64 / 4.).unwrap();
                                }
                            }
                            graph.set_bounds(1, 4, 0.75, capacity as f64 / 4.).unwrap();
                            for (tail, head, q) in [
                                (0, 1, first),
                                (1, 2, second),
                                (2, 3, 0),
                                (1, 3, 0),
                                (0, 2, direct),
                            ] {
                                let arc = graph.add_arc(tail, head, 0.).unwrap();
                                graph.arc_mut(arc).unwrap().consumption[4] = q as f64 / 4.;
                            }
                            let mut expected: Option<f64> = None;
                            for path in [vec![0, 1, 2], vec![0, 3], vec![4, 2]] {
                                let mut q = 0;
                                let mut feasible = true;
                                for &a in &path {
                                    q += [first, second, 0, 0, direct][a];
                                    let lower = if graph.arcs[a].head == 1 { 3 } else { 0 };
                                    if q < lower {
                                        if kind == ResourceKind::Equality {
                                            feasible = false;
                                            break;
                                        }
                                        q = lower;
                                    }
                                    if q > capacity {
                                        feasible = false;
                                        break;
                                    }
                                }
                                if feasible {
                                    let cost = path.iter().map(|&a| costs[a]).sum::<f64>();
                                    expected = Some(expected.map_or(cost, |v| v.min(cost)));
                                }
                            }
                            let mut work = 1_000_000;
                            assert_eq!(
                                minimum(
                                    &graph,
                                    &costs,
                                    &mut work,
                                    Instant::now() + std::time::Duration::from_secs(10)
                                ),
                                Some(expected)
                            );
                            compared += 1;
                        }
                    }
                }
            }
        }
    }
    assert_eq!(compared, 216);
}
#[test]
fn cycles_unsupported_resources_and_budgets_never_return_partial_path_proofs() {
    let mut graph = Graph::<5>::new(3, 5).unwrap();
    for v in 0..3 {
        for r in 0..5 {
            graph.set_bounds(v, r, 0., 3.).unwrap();
        }
    }
    graph.add_arc(0, 1, 0.).unwrap();
    graph.add_arc(1, 1, -1.).unwrap();
    graph.add_arc(1, 2, 0.).unwrap();
    graph.arcs[1].consumption[4] = 1.;
    let end = Instant::now() + std::time::Duration::from_secs(10);
    let mut work = 1_000_000;
    assert_eq!(minimum(&graph, &[0., -1., 0.], &mut work, end), None);
    graph.arcs.remove(1);
    graph.arcs[0].consumption[4] = -1.;
    let mut work = 1_000_000;
    assert_eq!(minimum(&graph, &[0., 0.], &mut work, end), None);
    graph.arcs[0].consumption[4] = 1.;
    let mut work = 0;
    assert_eq!(minimum(&graph, &[0., 0.], &mut work, end), None);
    let mut work = 1_000_000;
    assert_eq!(minimum(&graph, &[0., 0.], &mut work, Instant::now()), None);
}
