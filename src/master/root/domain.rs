//! Eligibility for a canonical relaxation of an integer unit-cover model.
use super::{RootProblem, RowSense};
use crate::enumeration::{
    CompleteCanonicalPaths, CompleteResourceRelaxation, PathDomainFailure, PathDomainLimits,
};
use crate::graph::ResourceKind;
use std::time::Instant;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CoverDomainFailure {
    UnsupportedModel,
    Paths(PathDomainFailure),
}
impl From<PathDomainFailure> for CoverDomainFailure {
    fn from(value: PathDomainFailure) -> Self {
        Self::Paths(value)
    }
}
impl std::fmt::Display for CoverDomainFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "canonical cover relaxation: {self:?}")
    }
}
impl std::error::Error for CoverDomainFailure {}

/// A complete canonical path relaxation for an eligible integer unit-cover model.
///
/// Each nonterminal has a nonnegative visit-count equality with residual RHS
/// zero or one. Consequently every positive integer route in a feasible residual
/// solution is physically vertex-simple. Disposable scalar extension is
/// monotone: any waiting schedule projects to its earliest profile on that path.
/// Original arc costs and the problem's logical row coefficients depend on paths.
///
/// Paths excluded by element sets, branches, fixing or other restrictions may
/// remain here; including them gives a relaxation. Direct terminal arcs are
/// included too. This is an integer-domain projection, not completeness of an
/// NG/non-elementary LP. It does not authorize cut rows: resource-dependent cuts
/// need a separate domination proof, and all added rows need original validity.
/// Nor does it validate a floating LP objective or publish a root certificate.
#[derive(Debug)]
pub struct CanonicalCoverRelaxation<'a> {
    problem: &'a RootProblem,
    paths: CompleteCanonicalPaths<'a>,
    /// (physical customer vertex, logical covering row).
    customer_rows: Vec<(usize, usize)>,
    work_used: usize,
}

fn spend(limits: &mut PathDomainLimits, amount: usize) -> Result<(), CoverDomainFailure> {
    if limits.deadline.is_some_and(|end| Instant::now() >= end) {
        return Err(PathDomainFailure::TimeLimit.into());
    }
    limits.max_work = limits
        .max_work
        .checked_sub(amount)
        .ok_or(PathDomainFailure::WorkLimit)?;
    Ok(())
}

fn validate_cover_structure(
    problem: &RootProblem,
    limits: &mut PathDomainLimits,
) -> Result<Vec<(usize, usize)>, CoverDomainFailure> {
    spend(limits, 0)?;
    let g = &problem.graph;
    let n = g.vertices.len();
    if !(2..=1024).contains(&n)
        || g.source >= n
        || g.sink >= n
        || g.source == g.sink
        || g.resources.len() > 2
    {
        return Err(PathDomainFailure::InvalidGraph.into());
    }
    if g.resources
        .iter()
        .any(|kind| *kind != ResourceKind::Disposable)
        || problem.rows.len() < n - 1
        || problem.arc_entries.len() != g.arcs.len()
        || problem.arc_index.len() != g.arcs.len()
        || problem.rows[0].sense != RowSense::LessEqual
        || !problem.rows[0].rhs.is_finite()
        || problem.rows[0].rhs < 0.
    {
        return Err(CoverDomainFailure::UnsupportedModel);
    }
    for row in &problem.rows {
        spend(limits, 1)?;
        if !row.rhs.is_finite() {
            return Err(CoverDomainFailure::UnsupportedModel);
        }
    }
    spend(limits, n)?;
    let mut customer_rows = Vec::new();
    customer_rows
        .try_reserve_exact(n - 2)
        .map_err(|_| PathDomainFailure::Allocation)?;
    for vertex in 0..n {
        if vertex != g.source && vertex != g.sink {
            let row = customer_rows.len() + 1;
            if problem.rows[row].sense != RowSense::Equal
                || !matches!(problem.rows[row].rhs, 0. | 1.)
                || problem.customers.get(row - 1) != Some(&vertex)
            {
                return Err(CoverDomainFailure::UnsupportedModel);
            }
            customer_rows.push((vertex, row));
        }
    }
    if problem.customers.len() != customer_rows.len() {
        return Err(CoverDomainFailure::UnsupportedModel);
    }
    // Check the raw core coefficients, without relying on rounded sparse
    // aggregation or on the mutable physical master row map.
    for (position, (arc, entries)) in g.arcs.iter().zip(&problem.arc_entries).enumerate() {
        spend(limits, 1)?;
        if arc.tail >= n || arc.head >= n {
            return Err(PathDomainFailure::InvalidGraph.into());
        }
        if problem.arc_position(arc.id) != Some(position) {
            return Err(CoverDomainFailure::UnsupportedModel);
        }
        let expected = if arc.head == g.sink {
            Some(0)
        } else {
            customer_rows
                .iter()
                .find(|&&(v, _)| v == arc.head)
                .map(|&(_, row)| row)
        };
        spend(limits, customer_rows.len())?;
        let mut found = false;
        for &(row, value) in entries {
            spend(limits, 1)?;
            if row >= problem.rows.len() || !value.is_finite() {
                return Err(CoverDomainFailure::UnsupportedModel);
            }
            if row < n - 1 && value != 0. {
                if Some(row) != expected || value != 1. || found {
                    return Err(CoverDomainFailure::UnsupportedModel);
                }
                found = true;
            }
        }
        if found != expected.is_some() {
            return Err(CoverDomainFailure::UnsupportedModel);
        }
    }
    for &(row, value) in &problem.route_entries {
        spend(limits, 1)?;
        if row >= problem.rows.len() || !value.is_finite() || (row < n - 1 && value != 0.) {
            return Err(CoverDomainFailure::UnsupportedModel);
        }
    }
    let mut fixed_counts = Vec::new();
    fixed_counts
        .try_reserve_exact(n)
        .map_err(|_| PathDomainFailure::Allocation)?;
    fixed_counts.resize(n, 0u32);
    for (route, quantity) in &problem.fixed_routes {
        spend(limits, route.arcs.len())?;
        let checked = problem
            .elementary_route(&route.arcs)
            .ok_or(CoverDomainFailure::UnsupportedModel)?;
        if *quantity == 0 {
            return Err(CoverDomainFailure::UnsupportedModel);
        }
        for &vertex in &checked.nodes {
            spend(limits, 1)?;
            let vertex = vertex as usize;
            if vertex != g.source && vertex != g.sink {
                fixed_counts[vertex] = fixed_counts[vertex]
                    .checked_add(*quantity)
                    .ok_or(CoverDomainFailure::UnsupportedModel)?;
            }
        }
    }
    for &(vertex, row) in &customer_rows {
        spend(limits, 1)?;
        if fixed_counts[vertex] > 1 || problem.rows[row].rhs != f64::from(1 - fixed_counts[vertex])
        {
            return Err(CoverDomainFailure::UnsupportedModel);
        }
    }
    Ok(customer_rows)
}
impl<'a> CanonicalCoverRelaxation<'a> {
    /// Validate the cover structure and fixed residual, then exhaust the path
    /// search under the same work/deadline budget. Failure returns no domain.
    /// Storage limits describe path payload, not total validation heap usage.
    pub fn collect(
        problem: &'a RootProblem,
        mut limits: PathDomainLimits,
    ) -> Result<Self, CoverDomainFailure> {
        let initial_work = limits.max_work;
        let customer_rows = validate_cover_structure(problem, &mut limits)?;
        let g = &problem.graph;
        let eligibility_work = initial_work - limits.max_work;
        let paths = CompleteCanonicalPaths::collect(g, limits)?;
        let work_used = eligibility_work + paths.work_used();
        Ok(Self {
            problem,
            paths,
            customer_rows,
            work_used,
        })
    }
    pub fn problem(&self) -> &'a RootProblem {
        self.problem
    }
    pub fn paths(&self) -> &CompleteCanonicalPaths<'a> {
        &self.paths
    }
    pub fn customer_rows(&self) -> &[(usize, usize)] {
        &self.customer_rows
    }
    pub fn work_used(&self) -> usize {
        self.work_used
    }
}

/// Physical-path relaxation covering every exact-real feasible integer cover
/// under disposable scalar resources. Resource states are downward bounds,
/// not feasible schedules. Extra infeasible paths may remain. The unit-cover
/// structure and fixed residual are checked as for `CanonicalCoverRelaxation`.
/// Original costs and logical coefficients depend on paths. Added matrix rows
/// still need validity over this relaxed domain; no native/tolerant resource
/// equivalence or root certificate is implied.
#[derive(Debug)]
pub struct ConservativeCoverRelaxation<'a> {
    problem: &'a RootProblem,
    paths: CompleteResourceRelaxation<'a>,
    customer_rows: Vec<(usize, usize)>,
    work_used: usize,
}
impl<'a> ConservativeCoverRelaxation<'a> {
    pub fn collect(
        problem: &'a RootProblem,
        mut limits: PathDomainLimits,
    ) -> Result<Self, CoverDomainFailure> {
        let initial_work = limits.max_work;
        let customer_rows = validate_cover_structure(problem, &mut limits)?;
        let eligibility_work = initial_work - limits.max_work;
        let paths = CompleteResourceRelaxation::collect(&problem.graph, limits)?;
        let work_used = eligibility_work + paths.work_used();
        Ok(Self {
            problem,
            paths,
            customer_rows,
            work_used,
        })
    }
    pub fn problem(&self) -> &'a RootProblem {
        self.problem
    }
    pub fn paths(&self) -> &CompleteResourceRelaxation<'a> {
        &self.paths
    }
    pub fn customer_rows(&self) -> &[(usize, usize)] {
        &self.customer_rows
    }
    pub fn work_used(&self) -> usize {
        self.work_used
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Graph;
    fn problem() -> RootProblem {
        let mut g = Graph::new(4, 1).unwrap();
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
}
