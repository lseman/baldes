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
