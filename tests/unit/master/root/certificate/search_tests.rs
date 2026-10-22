use super::*;

fn problem() -> RootProblem<2> {
    let mut graph = crate::Graph::<2>::new(4, 0).unwrap();
    for (tail, head) in [(0, 1), (0, 2), (1, 2), (2, 1), (1, 3), (2, 3)] {
        graph.add_arc(tail, head, 1.).unwrap();
    }
    RootProblem::set_partitioning(graph, 2.).unwrap()
}

#[test]
fn deterministic_domain_failures_are_not_repeated_but_larger_budgets_retry() {
    let problem = problem();
    let generous = PathDomainLimits {
        deadline: None,
        ..PathDomainLimits::default()
    };
    let domain = ConservativeCoverRelaxation::collect(&problem, generous).unwrap();
    let exact = PathDomainLimits {
        max_paths: domain.paths().paths().len(),
        max_stored_arcs: domain.paths().stored_arcs(),
        max_work: domain.work_used(),
        deadline: None,
    };
    for insufficient in [
        PathDomainLimits {
            max_work: exact.max_work - 1,
            ..exact
        },
        PathDomainLimits {
            max_paths: exact.max_paths - 1,
            ..exact
        },
        PathDomainLimits {
            max_stored_arcs: exact.max_stored_arcs - 1,
            ..exact
        },
    ] {
        let cache = SearchCache::default();
        assert!(cache.collect(&problem, insufficient).is_none());
        assert_eq!(cache.attempts.get(), 1);
        assert!(cache.collect(&problem, insufficient).is_none());
        assert_eq!(cache.attempts.get(), 1);
        let recovered = cache.collect(&problem, exact).unwrap();
        assert_eq!(cache.attempts.get(), 2);
        assert_eq!(recovered.paths().paths().len(), exact.max_paths);
        assert_eq!(recovered.work_used(), exact.max_work);
    }
}

#[test]
fn expired_deadline_does_not_poison_a_later_certificate_attempt() {
    let problem = problem();
    let cache = SearchCache::default();
    let limits = PathDomainLimits {
        deadline: Some(Instant::now()),
        ..PathDomainLimits::default()
    };
    assert!(cache.collect(&problem, limits).is_none());
    assert!(cache.failed_domain_limits.get().is_none());
    assert!(cache
        .collect(
            &problem,
            PathDomainLimits {
                deadline: None,
                ..limits
            }
        )
        .is_some());
    assert_eq!(cache.attempts.get(), 2);
}
