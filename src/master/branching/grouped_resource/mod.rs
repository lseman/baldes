//! Packing-group accumulated-resource disjunctions over physical arc IDs.
use super::{BranchConstraint, RowSense};
use crate::{
    invalid, master::routing::RoutingContext, pricing::restrictions::AccumulatedResourceBranch,
    Graph, Result, Route,
};
mod name;

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
    /// Resource-generator label using the packing row as its group identity.
    /// Display rounds to six significant digits; exact split identity remains
    /// separate so nearby thresholds with equal labels cannot merge branches.
    pub fn generator_name(&self) -> String {
        name::generator_name(self.packing_row, self.resource, self.threshold)
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
    pub(crate) fn validate<const D: usize>(&self, problem: &RoutingContext<D>) -> Result<()> {
        self.validate_proof(
            &problem.model.networks[0].graph,
            &problem.model.rows,
            &problem.model.networks[0].arc_entries,
            &problem.model.networks[0].route_entries,
            |id| problem.arc_position(id),
        )
    }
    pub(crate) fn validate_support<const D: usize>(
        &self,
        graph: &Graph<D>,
        rows: &[crate::master::routing::Row],
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
        rows: &[crate::master::routing::Row],
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
        let violation = route.arcs.iter().enumerate().any(|(i, &arc)| {
            self.arcs.binary_search(&arc).is_ok()
                && route
                    .resource_profile
                    .get(i + 1)
                    .and_then(|q| q.get(self.resource))
                    .map_or(true, |&q| !self.rule(arc).satisfied_by(q))
        }) as u8 as f64;
        if nonlinear_only && super::enforcement::ACCUMULATED_RESOURCE_IN_PRICING {
            // Blocked arcs already contribute through the additive entries.
            // Correct that decomposition even for a repeated group occurrence:
            // the complete route coefficient is a Boolean violation indicator.
            violation
                - route
                    .arcs
                    .iter()
                    .map(|&arc| self.additive(arc))
                    .sum::<f64>()
        } else {
            violation
        }
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
        problem: &RoutingContext<D>,
        route: &Route,
    ) -> f64 {
        match self {
            Self::AccumulatedResourceGroup(c) => c.coefficient(route, true),
            _ => self.coefficient(problem, route),
        }
    }
}

#[cfg(test)]
#[path = "../../../../tests/unit/master/branching/grouped_resource/tests.rs"]
mod tests;
