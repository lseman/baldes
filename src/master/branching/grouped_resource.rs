//! Packing-group accumulated-resource disjunctions over physical arc IDs.
use super::{BranchConstraint, RowSense};
use crate::{
    invalid, master::root::RootProblem, pricing::restrictions::AccumulatedResourceBranch, Graph,
    Result, Route,
};

/// Apply one resource split to every occurrence of a verified packing group.
/// The backing row must prove at most one group occurrence in an integer
/// solution; arbitrary arc sets do not provide a valid binary disjunction.
#[derive(Clone, Debug, PartialEq)]
pub struct AccumulatedResourceGroupBranch {
    packing_row: usize,
    arcs: Vec<i32>,
    resource: usize,
    threshold: f64,
    upper: bool,
    // Frozen on attachment: later restrictions cannot change an older row's
    // additive/nonlinear decomposition or its pricing coefficients.
    blocked: Vec<i32>,
}

impl AccumulatedResourceGroupBranch {
    pub fn new(
        packing_row: usize,
        mut arcs: Vec<i32>,
        resource: usize,
        threshold: f64,
        upper: bool,
    ) -> Result<Self> {
        arcs.sort_unstable();
        if arcs.is_empty() || arcs.windows(2).any(|p| p[0] == p[1]) || !threshold.is_finite() {
            return Err(invalid(
                "resource group requires unique physical arcs and a finite threshold",
            ));
        }
        Ok(Self {
            packing_row,
            arcs,
            resource,
            threshold,
            upper,
            blocked: vec![],
        })
    }
    pub fn packing_row(&self) -> usize {
        self.packing_row
    }
    pub fn arcs(&self) -> &[i32] {
        &self.arcs
    }
    pub fn resource(&self) -> usize {
        self.resource
    }
    pub fn threshold(&self) -> f64 {
        self.threshold
    }
    pub fn upper(&self) -> bool {
        self.upper
    }
    pub fn boundary(&self) -> f64 {
        self.threshold - AccumulatedResourceBranch::THRESHOLD_OFFSET
    }
    pub fn with_upper(&self, upper: bool) -> Self {
        Self {
            upper,
            blocked: vec![],
            ..self.clone()
        }
    }
    fn rule(&self, arc: i32) -> AccumulatedResourceBranch {
        AccumulatedResourceBranch {
            arc,
            resource: self.resource,
            threshold: self.threshold,
            upper: self.upper,
        }
    }
    pub(crate) fn validate<const D: usize>(&self, problem: &RootProblem<D>) -> Result<()> {
        self.validate_proof(
            &problem.graph,
            &problem.rows,
            &problem.arc_entries,
            &problem.route_entries,
            |id| problem.arc_position(id),
        )
    }
    pub(crate) fn validate_support<const D: usize>(
        &self,
        graph: &Graph<D>,
        rows: &[crate::master::root::Row],
        arc_entries: &[Vec<(usize, f64)>],
        route_entries: &[(usize, f64)],
    ) -> Result<()> {
        let index: std::collections::HashMap<_, _> = graph
            .arcs
            .iter()
            .enumerate()
            .map(|(i, a)| (a.id, i))
            .collect();
        self.validate_proof(graph, rows, arc_entries, route_entries, |id| {
            index.get(&id).copied()
        })
    }
    fn validate_proof<const D: usize>(
        &self,
        graph: &Graph<D>,
        rows: &[crate::master::root::Row],
        arc_entries: &[Vec<(usize, f64)>],
        route_entries: &[(usize, f64)],
        position: impl Fn(i32) -> Option<usize>,
    ) -> Result<()> {
        let row = rows
            .get(self.packing_row)
            .ok_or_else(|| invalid("resource group packing row"))?;
        if !matches!(row.sense, RowSense::LessEqual | RowSense::Equal)
            || !row.rhs.is_finite()
            || row.rhs > 1.
            || route_entries
                .iter()
                .any(|&(r, c)| r == self.packing_row && (!c.is_finite() || c < 0.))
            || arc_entries.iter().any(|entries| {
                entries
                    .iter()
                    .any(|&(r, c)| r == self.packing_row && (!c.is_finite() || c < 0.))
            })
        {
            return Err(invalid(
                "resource group needs a nonnegative unit packing row",
            ));
        }
        let mut identities = std::collections::HashSet::with_capacity(graph.arcs.len());
        if graph.arcs.iter().any(|arc| !identities.insert(arc.id)) {
            return Err(invalid(
                "resource group requires unambiguous physical arc IDs",
            ));
        }
        for &arc in &self.arcs {
            self.rule(arc).window(graph)?;
            let position = position(arc).ok_or_else(|| invalid("resource group arc"))?;
            // A single exact input coefficient >=1 is enough; do not turn a
            // rounded sum of sub-unit coefficients into an occurrence proof.
            if !arc_entries
                .get(position)
                .ok_or_else(|| invalid("resource group arc coefficients"))?
                .iter()
                .any(|&(r, c)| r == self.packing_row && c >= 1.)
            {
                return Err(invalid(
                    "resource group arc is not covered by its packing row",
                ));
            }
        }
        Ok(())
    }
    pub(crate) fn prepare<const D: usize>(mut self, graph: &Graph<D>) -> Result<Self> {
        self.blocked.clear();
        for &arc in &self.arcs {
            if self.rule(arc).window(graph)?.is_none() {
                self.blocked.push(arc);
            }
        }
        Ok(self)
    }
    pub(crate) fn install<const D: usize>(&self, graph: &mut Graph<D>) -> Result<()> {
        for &arc in &self.arcs {
            if let Some(window) = self.rule(arc).window(graph)? {
                graph.arc_mut(arc)?.bounds[self.resource] = Some(window);
            }
        }
        Ok(())
    }
    pub(crate) fn additive(&self, arc: i32) -> f64 {
        if self.blocked.binary_search(&arc).is_ok() {
            1.
        } else {
            0.
        }
    }
    pub(crate) fn coefficient(&self, route: &Route, nonlinear_only: bool) -> f64 {
        route
            .arcs
            .iter()
            .enumerate()
            .filter(|&(i, &arc)| {
                self.arcs.binary_search(&arc).is_ok()
                    && (!nonlinear_only || self.blocked.binary_search(&arc).is_err())
                    && route
                        .resource_profile
                        .get(i + 1)
                        .and_then(|q| q.get(self.resource))
                        .map_or(true, |&q| !self.rule(arc).satisfied_by(q))
            })
            .count() as f64
    }
    pub(crate) fn same_split(&self, other: &Self) -> bool {
        self.packing_row == other.packing_row
            && self.arcs == other.arcs
            && self.resource == other.resource
            && self.boundary() == other.boundary()
    }
}

impl BranchConstraint {
    pub(crate) fn nonlinear_coefficient<const D: usize>(
        &self,
        problem: &RootProblem<D>,
        route: &Route,
    ) -> f64 {
        match self {
            Self::AccumulatedResourceGroup(c) => c.coefficient(route, true),
            _ => self.coefficient(problem, route),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::master::tree::candidates::resource_groups::tests::fixture;

    #[test]
    fn blocked_members_remain_optional_and_old_row_decompositions_are_frozen() {
        let (mut problem, routes, arcs) = fixture();
        let packing = problem.arc_entries[problem.arc_position(arcs[0]).unwrap()][0].0;
        let upper =
            AccumulatedResourceGroupBranch::new(packing, arcs.clone(), 4, 1. + 1e-6, true).unwrap();
        let row = problem.rows.len();
        problem
            .branch(BranchConstraint::AccumulatedResourceGroup(upper.clone()))
            .unwrap();
        let value = |p: &RootProblem<5>, r: &Route, row| {
            p.base_entries(r)
                .unwrap()
                .iter()
                .filter(|(i, _)| *i == row)
                .map(|(_, c)| c)
                .sum::<f64>()
        };
        assert_eq!(value(&problem, &routes[0].0, row), 1.);
        assert_eq!(value(&problem, &routes[1].0, row), 0.);
        let lower =
            AccumulatedResourceGroupBranch::new(packing, arcs, 4, 2. + 1e-6, false).unwrap();
        problem
            .branch(BranchConstraint::AccumulatedResourceGroup(lower))
            .unwrap();
        assert_eq!(value(&problem, &routes[0].0, row), 1.);
        assert_eq!(value(&problem, &routes[1].0, row), 0.);
        assert_eq!(value(&problem, &routes[1].0, row + 1), 1.);
    }

    #[test]
    fn group_disjunction_requires_an_actual_unit_packing_proof() {
        let (mut problem, _, arcs) = fixture();
        let packing = problem.arc_entries[problem.arc_position(arcs[0]).unwrap()][0].0;
        let group = AccumulatedResourceGroupBranch::new(packing, arcs, 4, 1., false).unwrap();
        group.validate(&problem).unwrap();
        problem.rows[packing].rhs = 2.;
        assert!(group.validate(&problem).is_err());
        problem.rows[packing].rhs = 1.;
        problem.arc_entries[0].push((packing, -0.25));
        assert!(group.validate(&problem).is_err());
        assert!(AccumulatedResourceGroupBranch::new(0, vec![1, 1], 0, 1., false).is_err());
    }
}
