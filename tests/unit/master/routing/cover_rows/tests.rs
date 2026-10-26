use super::*;
use crate::{enumeration::PathDomainLimits, master::routing::RoutingContext, Graph};
fn problem() -> RoutingContext<2> {
    let mut g = Graph::<2>::new(4, 0).unwrap();
    for (u, v) in [(0, 1), (1, 3), (0, 2), (2, 3), (1, 2), (2, 1), (0, 3)] {
        g.add_arc(u, v, 1.).unwrap();
    }
    RoutingContext::set_partitioning(g, 2.).unwrap()
}
fn limits() -> CoverRowLimits {
    CoverRowLimits {
        deadline: None,
        ..Default::default()
    }
}
#[test]
fn weighted_envelopes_prove_rows_and_reject_a_single_violating_path() {
    let p = problem();
    let domain = ConservativeCoverRelaxation::collect(
        &p,
        PathDomainLimits {
            deadline: None,
            ..Default::default()
        },
    )
    .unwrap();
    for weights in [[0, 1, 1, 0], [0, -3, 2, 0]] {
        let coefficients = domain
            .paths()
            .paths()
            .iter()
            .map(|path| {
                path.arc_positions()
                    .iter()
                    .map(|&a| weights[p.model.networks[0].graph.arcs[a].head])
                    .sum::<i64>()
                    .div_euclid(2)
            })
            .collect::<Vec<_>>();
        let audit = domain
            .audit_weighted_upper_row(&coefficients, &weights, 2, limits())
            .unwrap();
        let exact = domain
            .audit_cover_row(&coefficients, limits())
            .unwrap()
            .range
            .unwrap();
        assert!(exact.maximum <= audit.upper_bound.unwrap());
        assert_eq!(
            audit.upper_bound,
            Some(i128::from(weights[1] + weights[2]).div_euclid(2))
        );
        let mut bad = coefficients.clone();
        bad[0] += 1;
        assert_eq!(
            domain
                .audit_weighted_upper_row(&bad, &weights, 2, limits())
                .unwrap()
                .upper_bound,
            None
        );
        let mut exhausted = limits();
        exhausted.max_work = audit.work_used - 1;
        assert_eq!(
            domain.audit_weighted_upper_row(&coefficients, &weights, 2, exhausted),
            Err(CoverRowFailure::WorkLimit)
        );
        exhausted = limits();
        exhausted.deadline = Some(Instant::now());
        assert_eq!(
            domain.audit_weighted_upper_row(&coefficients, &weights, 2, exhausted),
            Err(CoverRowFailure::TimeLimit)
        );
    }
    let mut fixed = problem();
    let route = fixed.elementary_route(&[0, 1]).unwrap();
    fixed.fixed_routes.push((route, 1));
    fixed.model.rows[0].rhs -= 1.;
    fixed.model.rows[1].rhs = 0.;
    let domain = ConservativeCoverRelaxation::collect(
        &fixed,
        PathDomainLimits {
            deadline: None,
            ..Default::default()
        },
    )
    .unwrap();
    let weights = [0, 3, 2, 0];
    let coefficients = domain
        .paths()
        .paths()
        .iter()
        .map(|path| {
            path.arc_positions()
                .iter()
                .map(|&a| weights[fixed.model.networks[0].graph.arcs[a].head])
                .sum::<i64>()
                .div_euclid(2)
        })
        .collect::<Vec<_>>();
    assert_eq!(
        domain
            .audit_weighted_upper_row(&coefficients, &weights, 2, limits())
            .unwrap()
            .upper_bound,
        Some(1)
    );
}
#[test]
fn exact_partition_extrema_reject_a_cut_that_excludes_a_cover() {
    let p = problem();
    let domain = CanonicalCoverRelaxation::collect(
        &p,
        PathDomainLimits {
            deadline: None,
            ..Default::default()
        },
    )
    .unwrap();
    let coefficients = domain
        .paths()
        .paths()
        .iter()
        .map(|path| match path.arc_positions().len() {
            1 => i64::MIN,
            2 => 10,
            _ => -3,
        })
        .collect::<Vec<_>>();
    let range = domain
        .audit_cover_row(&coefficients, limits())
        .unwrap()
        .range
        .unwrap();
    assert_eq!(
        range,
        CoverRowRange {
            minimum: -3,
            maximum: 20
        }
    );
    assert!(range.satisfies(RowSense::LessEqual, 20.5));
    assert!(!range.satisfies(RowSense::LessEqual, 19.9));
    assert!(range.satisfies(RowSense::GreaterEqual, -3.5));
    assert!(!range.satisfies(RowSense::GreaterEqual, -2.9));
    assert!(!range.satisfies(RowSense::Equal, 20.));
    assert!(!range.satisfies(RowSense::LessEqual, f64::INFINITY));
}
#[test]
fn missing_and_fully_fixed_covers_are_distinct() {
    let mut g = Graph::<2>::new(3, 0).unwrap();
    g.add_arc(0, 2, 1.).unwrap();
    let mut p = RoutingContext::set_partitioning(g, 1.).unwrap();
    let d = CanonicalCoverRelaxation::collect(
        &p,
        PathDomainLimits {
            deadline: None,
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(d.audit_cover_row(&[123], limits()).unwrap().range, None);
    let a = p.model.networks[0].graph.add_arc(0, 1, 1.).unwrap();
    let b = p.model.networks[0].graph.add_arc(1, 2, 1.).unwrap();
    p = RoutingContext::set_partitioning(p.model.networks.remove(0).graph, 1.).unwrap();
    p.fixed_routes
        .push((p.elementary_route(&[a, b]).unwrap(), 1));
    p.model.rows[0].rhs = 0.;
    p.model.rows[1].rhs = 0.;
    let d = CanonicalCoverRelaxation::collect(
        &p,
        PathDomainLimits {
            deadline: None,
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(
        d.audit_cover_row(&vec![i64::MAX; d.paths().paths().len()], limits())
            .unwrap()
            .range,
        Some(CoverRowRange {
            minimum: 0,
            maximum: 0
        })
    );
}
#[test]
fn dimensions_and_exact_work_state_deadline_limits_are_enforced() {
    let p = problem();
    let d = CanonicalCoverRelaxation::collect(
        &p,
        PathDomainLimits {
            deadline: None,
            ..Default::default()
        },
    )
    .unwrap();
    let coefficients = vec![1; d.paths().paths().len()];
    let audit = d.audit_cover_row(&coefficients, limits()).unwrap();
    let mut l = limits();
    l.max_work = audit.work_used;
    l.max_states = audit.states;
    assert!(d.audit_cover_row(&coefficients, l).is_ok());
    l.max_work -= 1;
    assert_eq!(
        d.audit_cover_row(&coefficients, l).unwrap_err(),
        CoverRowFailure::WorkLimit
    );
    l = limits();
    l.max_states = audit.states - 1;
    assert_eq!(
        d.audit_cover_row(&coefficients, l).unwrap_err(),
        CoverRowFailure::StateLimit
    );
    l = limits();
    l.deadline = Some(Instant::now());
    assert_eq!(
        d.audit_cover_row(&coefficients, l).unwrap_err(),
        CoverRowFailure::TimeLimit
    );
    assert_eq!(
        d.audit_cover_row(&[], limits()).unwrap_err(),
        CoverRowFailure::Dimension
    );
}
#[test]
fn conservative_row_domain_includes_an_exact_real_feasible_rounding_boundary() {
    let mut graph = Graph::<2>::new(4, 1).unwrap();
    let upper = f64::from_bits(1f64.to_bits() + 3);
    let delta = 3. * 2f64.powi(-53);
    graph.initial[0] = Some(0.);
    for v in 0..4 {
        graph.set_bounds(v, 0, 0., upper).unwrap();
    }
    for (i, consumption) in [1., delta, delta].into_iter().enumerate() {
        let id = graph.add_arc(i, i + 1, 1.).unwrap();
        graph.arc_mut(id).unwrap().consumption[0] = consumption;
    }
    let p = RoutingContext::set_partitioning(graph, 1.).unwrap();
    let domain = ConservativeCoverRelaxation::collect(
        &p,
        PathDomainLimits {
            deadline: None,
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(domain.paths().paths().len(), 1);
    let row = domain
        .audit_cover_row(&[1], limits())
        .unwrap()
        .range
        .unwrap();
    assert_eq!(
        row,
        CoverRowRange {
            minimum: 1,
            maximum: 1
        }
    );
    assert!(row.satisfies(RowSense::LessEqual, 1.));
    assert!(!row.satisfies(RowSense::LessEqual, 0.));
    // The retained path can support an actual finite-matrix lower bound
    // even though strict forward accumulation omitted its column.
    let entries = domain.paths().paths()[0]
        .arc_positions()
        .iter()
        .flat_map(|&a| p.model.networks[0].arc_entries[a].iter().copied())
        .collect::<Vec<_>>();
    let costs = domain.paths().paths()[0]
        .arc_positions()
        .iter()
        .map(|&a| p.model.networks[0].graph.arcs[a].cost)
        .collect::<Vec<_>>();
    let rhs = p.model.rows.iter().map(|r| r.rhs).collect::<Vec<_>>();
    let senses = p.model.rows.iter().map(|r| r.sense).collect::<Vec<_>>();
    let columns = [crate::master::certificates::Column {
        cost_terms: &costs,
        entries: &entries,
    }];
    let matrix = crate::master::certificates::FiniteMaster {
        rhs: &rhs,
        senses: &senses,
        columns: &columns,
        offset_terms: &[],
        multiplicity_row: 0,
    };
    assert_eq!(matrix.lower_bound(&[0., 1., 2.]), Some(3.));
}
