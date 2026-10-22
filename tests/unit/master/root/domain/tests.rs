use super::*;
use crate::Graph;
fn problem() -> RootProblem<2> {
    let mut g = Graph::<2>::new(4, 1).unwrap();
    for (u, v) in [
        (0, 1),
        (1, 3),
        (0, 2),
        (2, 3),
        (1, 2),
        (2, 1),
        (0, 3),
        (1, 1),
    ] {
        let id = g.add_arc(u, v, 1.).unwrap();
        g.arc_mut(id).unwrap().consumption[0] = 1.;
    }
    RootProblem::set_partitioning(g, 2.).unwrap()
}
fn limits() -> PathDomainLimits {
    PathDomainLimits {
        deadline: None,
        ..Default::default()
    }
}
#[test]
fn cycles_are_excluded_by_unit_covers_and_extra_paths_are_retained() {
    let p = problem();
    let domain = CanonicalCoverRelaxation::collect(&p, limits()).unwrap();
    assert_eq!(domain.customer_rows(), &[(1, 1), (2, 2)]);
    assert_eq!(domain.paths().paths().len(), 5);
    assert!(domain
        .paths()
        .paths()
        .iter()
        .any(|p| p.arc_positions() == [6]));
    assert!(!domain
        .paths()
        .paths()
        .iter()
        .any(|p| p.arc_positions().contains(&7)));
}
#[test]
fn fixed_routes_must_match_the_exact_unit_cover_residual() {
    let mut p = problem();
    p.fixed_routes
        .push((p.elementary_route(&[0, 1]).unwrap(), 1));
    p.rows[0].rhs = 1.;
    assert!(matches!(
        CanonicalCoverRelaxation::collect(&p, limits()),
        Err(CoverDomainFailure::UnsupportedModel)
    ));
    p.rows[1].rhs = 0.;
    assert_eq!(
        CanonicalCoverRelaxation::collect(&p, limits())
            .unwrap()
            .paths()
            .paths()
            .len(),
        5
    );
    p.fixed_routes[0].1 = 2;
    assert!(matches!(
        CanonicalCoverRelaxation::collect(&p, limits()),
        Err(CoverDomainFailure::UnsupportedModel)
    ));
}
#[test]
fn fractional_or_schedule_sensitive_cover_models_are_not_eligible() {
    let mut p = problem();
    p.arc_entries[0][0].1 = 0.5;
    assert!(matches!(
        CanonicalCoverRelaxation::collect(&p, limits()),
        Err(CoverDomainFailure::UnsupportedModel)
    ));
    p = problem();
    p.graph.resources[0] = ResourceKind::Equality;
    assert!(matches!(
        CanonicalCoverRelaxation::collect(&p, limits()),
        Err(CoverDomainFailure::UnsupportedModel)
    ));
    p = problem();
    p.rows[1].rhs = 0.5;
    assert!(matches!(
        CanonicalCoverRelaxation::collect(&p, limits()),
        Err(CoverDomainFailure::UnsupportedModel)
    ));
    p = problem();
    p.rows.push(super::super::Row {
        sense: RowSense::Equal,
        rhs: f64::NAN,
    });
    assert!(matches!(
        CanonicalCoverRelaxation::collect(&p, limits()),
        Err(CoverDomainFailure::UnsupportedModel)
    ));
    p = problem();
    p.arc_index.insert(999, 0);
    assert!(matches!(
        CanonicalCoverRelaxation::collect(&p, limits()),
        Err(CoverDomainFailure::UnsupportedModel)
    ));
}
#[test]
fn eligibility_and_path_collection_share_work_and_deadline() {
    let p = problem();
    let mut l = limits();
    let used = CanonicalCoverRelaxation::collect(&p, l)
        .unwrap()
        .work_used();
    l.max_work = used;
    assert_eq!(
        CanonicalCoverRelaxation::collect(&p, l)
            .unwrap()
            .work_used(),
        used
    );
    l.max_work -= 1;
    assert!(matches!(
        CanonicalCoverRelaxation::collect(&p, l),
        Err(CoverDomainFailure::Paths(PathDomainFailure::WorkLimit))
    ));
    l = limits();
    l.deadline = Some(Instant::now());
    assert!(matches!(
        CanonicalCoverRelaxation::collect(&p, l),
        Err(CoverDomainFailure::Paths(PathDomainFailure::TimeLimit))
    ));
}
