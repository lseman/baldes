//! Owned immutable improvement-domain view; it cannot replace original pricing.
use super::*;

/// Directed cutoff metadata at the view's retained dual point.
#[derive(Clone, Debug)]
pub struct CutoffCostBound {
    pub network: i32,
    /// Lower bound on contributions other than the selected route's reduced cost.
    pub objective_intercept: f64,
    pub route_constant_lower_bound: f64,
    pub arc_ceiling: f64,
}

/// Complete coverage only for integer solutions strictly below the objective
/// cutoff. The internal restricted snapshot cannot be resumed as an original
/// LP, used for an original infeasibility claim, or attached to another node.
pub struct CutoffEnumeration<const D: usize = { crate::DEFAULT_RESOURCE_DIMENSIONS }> {
    node: NodeSnapshot<D>,
    cutoff: f64,
    duals: Vec<f64>,
    cost_bounds: Vec<CutoffCostBound>,
}
impl<const D: usize> CutoffEnumeration<D> {
    pub fn dual_point(&self) -> &[f64] {
        &self.duals
    }
    pub fn cost_bounds(&self) -> &[CutoffCostBound] {
        &self.cost_bounds
    }
    pub fn coverage_cutoff(&self) -> f64 {
        self.cutoff
    }
    /// Completion is relative to `coverage_cutoff`, never the full domain.
    pub fn report(&self) -> &EnumerationReport {
        &self.node.state.enumeration_report
    }
    pub fn routes(&self) -> impl Iterator<Item = &NetworkRoute> {
        self.node
            .state
            .universe
            .iter()
            .flatten()
            .map(|&tag| self.node.state.routes[tag].as_ref())
    }
    /// Solve an independent integer copy. A finite cutoff does not imply that
    /// the caller has a feasible incumbent. Missing improving solutions report
    /// ObjectiveCutoff, including when the restricted model is infeasible.
    pub fn solve_mip(&self, options: &MipOptions) -> Result<EnumeratedMipOutput> {
        use crate::master::solvers::mip::MipStatus;
        if options.objective_cutoff.is_nan() {
            return Err(invalid("invalid MIP cutoff"));
        }
        let mut options = options.clone();
        options.objective_cutoff = options.objective_cutoff.min(self.cutoff);
        let mut result = self
            .node
            .solve_enumerated_mip(&options)?
            .ok_or_else(|| invalid("missing cutoff enumeration domain"))?;
        result.coverage_cutoff = self.cutoff;
        result.mip.bound = result.mip.bound.min(options.objective_cutoff);
        if result.mip.status == MipStatus::Infeasible {
            result.mip.status = MipStatus::ObjectiveCutoff;
            result.mip.bound = options.objective_cutoff;
        }
        Ok(result)
    }
}
impl<const D: usize> NodeSnapshot<D> {
    /// Derive a common directed original-domain proof and collect all routes
    /// needed by any improving integer solution. Errors leave this snapshot
    /// unchanged. The returned owned view exposes no original-domain LP API.
    pub fn enumerate_below(
        &self,
        cutoff: f64,
        options: &EnumerationOptions,
    ) -> Result<CutoffEnumeration<D>> {
        if !cutoff.is_finite() {
            return Err(invalid("finite enumeration cutoff required"));
        }
        let deadline = Instant::now()
            .checked_add(options.time_limit)
            .ok_or_else(|| invalid("enumeration time limit"))?;
        let duals = self.master.duals();
        let mut node = self.fork()?;
        node.state.deadline = deadline;
        node.state.enumeration_report = EnumerationReport {
            coverage_cutoff: Some(cutoff),
            ..Default::default()
        };
        let mut work = options.max_work;
        let plan = super::super::bounds::cutoff_plan(
            &node.state,
            &node.master,
            &duals,
            cutoff,
            &mut work,
            deadline,
        )
        .ok_or_else(|| incomplete("original-domain cutoff proof unavailable"))?;
        let mut collection = options.clone();
        collection.max_work = work;
        collection.time_limit = deadline.saturating_duration_since(Instant::now());
        let (routes, counts) =
            collect_routes_with_plan(&node.state, &collection, Some(&plan)).map_err(incomplete)?;
        if Instant::now() >= deadline {
            return Err(incomplete("cutoff enumeration TimeLimit"));
        }
        let mut cg = cg::Options::default();
        cg.warm_start = Some(node.warm);
        node.state.enumeration_report.attempts += 1;
        install_collected_with(
            &mut node.master,
            &mut node.state,
            &mut cg,
            routes,
            counts,
            deadline,
            |destination, column| {
                destination
                    .add_column(column.cost, &column.entries)
                    .map(|_| ())
            },
            Instant::now,
        )?;
        node.warm = cg.warm_start.unwrap();
        if !node.state.enumeration_report.complete
            || node.state.enumeration_report.failure.is_some()
        {
            return Err(incomplete("cutoff transfer interrupted"));
        }
        let cost_bounds = plan
            .iter()
            .filter_map(|p| match p {
                super::super::bounds::NetworkCutoff::Empty { .. } => None,
                super::super::bounds::NetworkCutoff::Cost {
                    network,
                    other_contributions,
                    route_constant_lower_bound,
                    arc_ceiling,
                    ..
                } => Some(CutoffCostBound {
                    network: *network,
                    objective_intercept: *other_contributions,
                    route_constant_lower_bound: *route_constant_lower_bound,
                    arc_ceiling: *arc_ceiling,
                }),
            })
            .collect();
        Ok(CutoffEnumeration {
            node,
            cutoff,
            duals,
            cost_bounds,
        })
    }
}

#[cfg(all(test, feature = "highs"))]
#[path = "../../../../tests/unit/master/model/enumeration/cutoff/tests.rs"]
mod tests;

#[cfg(all(test, feature = "highs"))]
#[path = "../../../../tests/unit/master/model/enumeration/cutoff/limit_tests.rs"]
mod limit_tests;
