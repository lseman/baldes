use super::*;
use crate::graph::ResourceKind;
fn limits() -> PathDomainLimits {
    PathDomainLimits {
        deadline: None,
        ..Default::default()
    }
}
fn graph() -> Graph<2> {
    let mut graph = Graph::<2>::new(4, 1).unwrap();
    for v in 0..4 {
        graph.set_bounds(v, 0, 0., 5.).unwrap();
    }
    graph.set_bounds(1, 0, 2., 5.).unwrap();
    graph.set_bounds(2, 0, 0., 3.).unwrap();
    for (u, v, c) in [
        (0, 1, 0.),
        (0, 1, 0.),
        (1, 2, 2.),
        (2, 1, 0.),
        (1, 3, 1.),
        (2, 3, 1.),
        (0, 2, 1.),
    ] {
        let id = graph.add_arc(u, v, 1.).unwrap();
        graph.arc_mut(id).unwrap().consumption[0] = c;
    }
    graph
}
#[test]
fn parallel_arcs_cycles_waiting_and_equality_have_explicit_canonical_paths() {
    let graph = graph();
    let domain = CompleteCanonicalPaths::collect(&graph, limits()).unwrap();
    let paths = domain
        .paths()
        .iter()
        .map(|p| p.arc_positions().to_vec())
        .collect::<Vec<_>>();
    assert_eq!(paths, [vec![0, 4], vec![1, 4], vec![6, 3, 4], vec![6, 5]]);
    assert_eq!(
        domain.paths()[0].resource_profile(),
        &[[0., 0.], [2., 0.], [3., 0.]]
    );
    let mut equality = graph.clone();
    equality.resources[0] = ResourceKind::Equality;
    let domain = CompleteCanonicalPaths::collect(&equality, limits()).unwrap();
    assert_eq!(domain.paths().len(), 1);
    assert_eq!(domain.paths()[0].arc_positions(), &[6, 5]);
}
#[test]
fn limits_reject_partial_results_and_accept_exact_boundaries() {
    let graph = graph();
    let domain = CompleteCanonicalPaths::collect(&graph, limits()).unwrap();
    let exact = PathDomainLimits {
        max_paths: domain.paths().len(),
        max_stored_arcs: domain.stored_arcs(),
        max_work: domain.work_used(),
        deadline: None,
    };
    assert!(CompleteCanonicalPaths::collect(&graph, exact).is_ok());
    for (limit, expected) in [
        (
            PathDomainLimits {
                max_paths: exact.max_paths - 1,
                ..exact
            },
            PathDomainFailure::PathLimit,
        ),
        (
            PathDomainLimits {
                max_stored_arcs: exact.max_stored_arcs - 1,
                ..exact
            },
            PathDomainFailure::StoredArcLimit,
        ),
        (
            PathDomainLimits {
                max_work: exact.max_work - 1,
                ..exact
            },
            PathDomainFailure::WorkLimit,
        ),
        (
            PathDomainLimits {
                deadline: Some(Instant::now()),
                ..exact
            },
            PathDomainFailure::TimeLimit,
        ),
    ] {
        assert_eq!(
            CompleteCanonicalPaths::collect(&graph, limit).unwrap_err(),
            expected
        );
    }
}
#[test]
fn empty_domain_and_impossible_initial_state_are_complete() {
    let graph = Graph::<2>::new(2, 0).unwrap();
    let limit = PathDomainLimits {
        max_paths: 0,
        max_stored_arcs: 0,
        ..limits()
    };
    assert!(CompleteCanonicalPaths::collect(&graph, limit)
        .unwrap()
        .paths()
        .is_empty());
    let mut graph = super::tests::graph();
    graph.initial[0] = Some(10.);
    assert!(CompleteCanonicalPaths::collect(&graph, limits())
        .unwrap()
        .paths()
        .is_empty());
}
#[test]
fn direct_paths_and_cost_overflow_are_not_silently_omitted() {
    let mut graph = Graph::<2>::new(2, 0).unwrap();
    graph.add_arc(0, 1, 1.).unwrap();
    let domain = CompleteCanonicalPaths::collect(&graph, limits()).unwrap();
    assert_eq!(domain.paths()[0].arc_positions(), &[0]);
    let mut graph = Graph::<2>::new(3, 0).unwrap();
    graph.add_arc(0, 1, f64::MAX).unwrap();
    graph.add_arc(1, 2, f64::MAX).unwrap();
    let domain = CompleteCanonicalPaths::collect(&graph, limits()).unwrap();
    assert_eq!(domain.paths()[0].arc_positions(), &[0, 1]);
}
#[test]
fn malformed_graphs_are_not_completed_domains() {
    let mut graph = graph();
    graph.arcs[0].head = 99;
    assert_eq!(
        CompleteCanonicalPaths::collect(&graph, limits()).unwrap_err(),
        PathDomainFailure::InvalidGraph
    );
}

#[test]
fn sparse_path_search_does_not_rescan_unrelated_arcs() {
    let mut graph = Graph::<2>::new(80, 0).unwrap();
    graph.add_arc(0, 1, 1.).unwrap();
    graph.add_arc(1, 79, 1.).unwrap();
    for tail in 2..79 {
        for head in 2..79 {
            if tail != head {
                graph.add_arc(tail, head, 1.).unwrap();
            }
        }
    }
    // Validation and adjacency construction are linear in graph size.
    // The reachable DFS has only two arcs, despite the dense unreachable
    // component. A full-graph scan at each frame exceeds this budget.
    let limits = PathDomainLimits {
        max_work: 2 * (graph.vertices.len() + graph.arcs.len()) + 16,
        deadline: None,
        ..limits()
    };
    let domain = CompleteCanonicalPaths::collect(&graph, limits).unwrap();
    assert_eq!(domain.paths().len(), 1);
    assert_eq!(domain.paths()[0].arc_positions(), &[0, 1]);
}

fn chain(consumption: &[f64], upper: f64) -> Graph<2> {
    let mut g = Graph::<2>::new(consumption.len() + 1, 1).unwrap();
    g.initial[0] = Some(0.);
    for vertex in 0..g.vertices.len() {
        g.set_bounds(vertex, 0, 0., upper).unwrap();
    }
    for (i, &value) in consumption.iter().enumerate() {
        let id = g.add_arc(i, i + 1, 1.).unwrap();
        g.arc_mut(id).unwrap().consumption[0] = value;
    }
    g
}
#[test]
fn directed_relaxation_retains_an_exact_real_feasible_rounding_boundary() {
    // The exact sum is 1 + 3 ulps. Two nearest/even additions of 1.5 ulps
    // round upward, producing 1 + 4 ulps and a spurious window rejection.
    let delta = 3. * 2f64.powi(-53);
    let upper = f64::from_bits(1f64.to_bits() + 3);
    let g = chain(&[1., delta, delta], upper);
    assert!(CompleteCanonicalPaths::collect(&g, limits())
        .unwrap()
        .paths()
        .is_empty());
    let domain = CompleteResourceRelaxation::collect(&g, limits()).unwrap();
    assert_eq!(domain.paths().len(), 1);
    assert_eq!(domain.paths()[0].arc_positions(), &[0, 1, 2]);
    assert!(domain.paths()[0].resource_lower_bounds()[3][0] <= upper);
}
#[test]
fn relaxation_may_retain_infeasible_paths_but_rejects_proven_violations() {
    let g = chain(&[0.7, 0.3], f64::from_bits(1f64.to_bits() - 1));
    assert_eq!(
        CompleteResourceRelaxation::collect(&g, limits())
            .unwrap()
            .paths()
            .len(),
        1
    );
    let g = chain(&[0.7, 0.4], 1.);
    assert!(CompleteResourceRelaxation::collect(&g, limits())
        .unwrap()
        .paths()
        .is_empty());
    let g = graph();
    let a = CompleteCanonicalPaths::collect(&g, limits()).unwrap();
    let b = CompleteResourceRelaxation::collect(&g, limits()).unwrap();
    assert_eq!(
        a.paths()
            .iter()
            .map(|p| p.arc_positions())
            .collect::<Vec<_>>(),
        b.paths()
            .iter()
            .map(|p| p.arc_positions())
            .collect::<Vec<_>>()
    );
}
#[test]
fn relaxed_collection_has_atomic_limits_and_explicit_unknown_arithmetic() {
    let g = chain(&[1., 0.5], 2.);
    let domain = CompleteResourceRelaxation::collect(&g, limits()).unwrap();
    let exact = PathDomainLimits {
        max_paths: 1,
        max_stored_arcs: 2,
        max_work: domain.work_used(),
        deadline: None,
    };
    assert!(CompleteResourceRelaxation::collect(&g, exact).is_ok());
    for (l, want) in [
        (
            PathDomainLimits {
                max_paths: 0,
                ..exact
            },
            PathDomainFailure::PathLimit,
        ),
        (
            PathDomainLimits {
                max_stored_arcs: 1,
                ..exact
            },
            PathDomainFailure::StoredArcLimit,
        ),
        (
            PathDomainLimits {
                max_work: exact.max_work - 1,
                ..exact
            },
            PathDomainFailure::WorkLimit,
        ),
        (
            PathDomainLimits {
                deadline: Some(Instant::now()),
                ..exact
            },
            PathDomainFailure::TimeLimit,
        ),
    ] {
        assert_eq!(
            CompleteResourceRelaxation::collect(&g, l).unwrap_err(),
            want
        );
    }
    let g = chain(&[f64::MAX, f64::MAX], f64::MAX);
    assert_eq!(
        CompleteResourceRelaxation::collect(&g, limits()).unwrap_err(),
        PathDomainFailure::Arithmetic
    );
    let mut g = chain(&[1., 1.], 2.);
    g.resources[0] = ResourceKind::Equality;
    assert_eq!(
        CompleteResourceRelaxation::collect(&g, limits()).unwrap_err(),
        PathDomainFailure::UnsupportedResources
    );
}

#[test]
fn deadline_after_a_materialized_path_discards_the_partial_collection() {
    let mut graph = Graph::<5>::new(80, 5).unwrap();
    let sink = graph.sink;
    graph.add_arc(0, 1, 0.).unwrap();
    graph.add_arc(1, sink, 0.).unwrap();
    graph.add_arc(0, 2, 0.).unwrap();
    for vertex in 2..sink {
        graph.add_arc(vertex, vertex + 1, 0.).unwrap();
    }
    let original = graph.clone();
    let created = std::cell::Cell::new(0);
    let extensions = std::cell::Cell::new(0);
    let started = Instant::now();
    let failed = collect_paths_with_clock(
        &graph,
        PathDomainLimits {
            deadline: Some(started + Duration::from_secs(1)),
            ..limits()
        },
        |graph, arc, quantities| {
            extensions.set(extensions.get() + 1);
            extend_resource_lower_bounds(graph, arc, quantities)
        },
        |arcs, _| {
            created.set(created.get() + 1);
            arcs
        },
        || {
            if created.get() == 0 {
                started
            } else {
                started + Duration::from_secs(2)
            }
        },
    );
    assert!(matches!(failed, Err(PathDomainFailure::TimeLimit)));
    assert_eq!(created.get(), 1);
    assert!(
        extensions.get() > 2,
        "must interrupt the second path's traversal"
    );
    assert_eq!(graph, original);
    let retried = CompleteResourceRelaxation::collect(&graph, limits()).unwrap();
    assert_eq!(retried.paths().len(), 2);
    assert_eq!(retried.paths()[0].arc_positions(), [0, 1]);
}

#[test]
fn cost_bounded_relaxation_matches_exhaustive_signed_parallel_cycle_domains() {
    use std::collections::BTreeSet;
    let graph = graph();
    // Independently listed feasible simple paths for this resource schedule.
    let paths = [vec![0, 4], vec![1, 4], vec![6, 3, 4], vec![6, 5]];
    for code in 0..3usize.pow(7) {
        let mut remaining = code;
        let mut weights = vec![];
        for _ in 0..7 {
            weights.push((remaining % 3) as f64 - 1.);
            remaining /= 3;
        }
        for ceiling in -3..=3 {
            let domain =
                CostBoundedResourceRelaxation::collect(&graph, &weights, ceiling as f64, limits())
                    .unwrap();
            assert_eq!(domain.coverage_ceiling(), ceiling as f64);
            let expected: BTreeSet<_> = paths
                .iter()
                .filter(|p| p.iter().map(|&i| weights[i]).sum::<f64>() <= ceiling as f64)
                .cloned()
                .collect();
            let actual: BTreeSet<_> = domain
                .paths()
                .iter()
                .map(|p| p.arc_positions().to_vec())
                .collect();
            assert_eq!(actual, expected, "weights={weights:?}, ceiling={ceiling}");
        }
    }
}

#[test]
fn cost_pruning_precedes_storage_limits_and_keeps_negative_suffixes() {
    let mut graph = Graph::<7>::new(4, 0).unwrap();
    for (tail, head, cost) in [(0, 1, 1.), (1, 3, 1.), (2, 3, 0.)] {
        graph.add_arc(tail, head, cost).unwrap();
    }
    for _ in 0..32 {
        graph.add_arc(0, 2, 100.).unwrap();
    }
    let weights: Vec<_> = graph.arcs.iter().map(|a| a.cost).collect();
    let small = PathDomainLimits {
        max_paths: 1,
        max_stored_arcs: 2,
        ..limits()
    };
    assert!(matches!(
        CompleteResourceRelaxation::collect(&graph, small),
        Err(PathDomainFailure::PathLimit)
    ));
    let domain = CostBoundedResourceRelaxation::collect(&graph, &weights, 2., small).unwrap();
    assert_eq!(domain.paths().len(), 1);
    assert_eq!(domain.paths()[0].arc_positions(), &[0, 1]);
    assert_eq!(domain.stored_arcs(), 2);
    let exact = PathDomainLimits {
        max_work: domain.work_used(),
        ..small
    };
    assert!(CostBoundedResourceRelaxation::collect(&graph, &weights, 2., exact).is_ok());
    assert!(matches!(
        CostBoundedResourceRelaxation::collect(
            &graph,
            &weights,
            2.,
            PathDomainLimits {
                max_work: exact.max_work - 1,
                ..exact
            }
        ),
        Err(PathDomainFailure::WorkLimit)
    ));
    let mut signed = weights;
    signed[0] = 100.;
    signed[1] = -101.;
    let domain = CostBoundedResourceRelaxation::collect(&graph, &signed, -1., small).unwrap();
    assert_eq!(domain.paths()[0].arc_positions(), &[0, 1]);
}

#[test]
fn cost_coverage_is_outward_and_invalid_or_partial_searches_return_no_domain() {
    let mut graph = Graph::<2>::new(3, 0).unwrap();
    graph.add_arc(0, 1, 0.).unwrap();
    graph.add_arc(1, 2, 0.).unwrap();
    let at = 0.3f64;
    let above = f64::from_bits(at.to_bits() + 1);
    // Exact binary64 0.1 + 0.2 is below `above`; downward pruning must keep it.
    let weights = [0.1, 0.2];
    assert_eq!(
        CostBoundedResourceRelaxation::collect(&graph, &weights, above, limits())
            .unwrap()
            .paths()
            .len(),
        1
    );
    let below = f64::from_bits(at.to_bits() - 1);
    assert_eq!(
        CostBoundedResourceRelaxation::collect(&graph, &weights, below, limits())
            .unwrap()
            .paths()
            .len(),
        0
    );
    for (costs, ceiling) in [
        (vec![], 1.),
        (vec![f64::NAN, 0.], 1.),
        (vec![0., f64::INFINITY], 1.),
        (vec![0., 0.], f64::INFINITY),
    ] {
        assert!(matches!(
            CostBoundedResourceRelaxation::collect(&graph, &costs, ceiling, limits()),
            Err(PathDomainFailure::Arithmetic)
        ));
    }
    assert!(matches!(
        CostBoundedResourceRelaxation::collect(
            &graph,
            &weights,
            1.,
            PathDomainLimits {
                deadline: Some(Instant::now()),
                ..limits()
            }
        ),
        Err(PathDomainFailure::TimeLimit)
    ));
    assert!(matches!(
        CostBoundedResourceRelaxation::collect(
            &graph,
            &weights,
            1.,
            PathDomainLimits {
                max_paths: 0,
                ..limits()
            }
        ),
        Err(PathDomainFailure::PathLimit)
    ));
    assert!(matches!(
        CostBoundedResourceRelaxation::collect(
            &graph,
            &weights,
            1.,
            PathDomainLimits {
                max_stored_arcs: 1,
                ..limits()
            }
        ),
        Err(PathDomainFailure::StoredArcLimit)
    ));
    let mut equality = graph;
    equality.resources.push(ResourceKind::Equality);
    assert!(matches!(
        CostBoundedResourceRelaxation::collect(&equality, &weights, 1., limits()),
        Err(PathDomainFailure::UnsupportedResources)
    ));
}
