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
            .map(|&tag| &self.node.state.routes[tag])
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
            collect_with_plan(&node.state, &collection, Some(&plan)).map_err(incomplete)?;
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
mod tests {
    use super::*;
    use crate::master::solvers::mip::MipStatus;
    fn root() -> Output<7> {
        let problem = super::super::super::format::Model::from_json(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/examples/networks/triangle.json"
        )))
        .unwrap()
        .problem()
        .unwrap();
        let mut options = Options::default();
        options.cg.stabilize = false;
        options.cg.artificial_cost = 100.;
        solve(problem, options).unwrap()
    }
    #[test]
    fn all_network_cutoff_transfer_and_mip_keep_original_pricing_available() {
        let root = root();
        assert_eq!(root.master.lp, 1.5);
        let source_routes = root.snapshot.state.routes.len();
        let limits = EnumerationOptions {
            max_routes: 3,
            ..Default::default()
        };
        // The full nine-route domain exceeds this cap; only its three cheap
        // pair routes are needed below 1.75 at the common root dual point.
        let full = root.snapshot.enumerate(&limits).unwrap();
        assert!(!full.enumeration_report().complete);
        let cut = root.snapshot.enumerate_below(1.75, &limits).unwrap();
        assert_eq!(cut.coverage_cutoff(), 1.75);
        assert_eq!(cut.report().routes, 3);
        assert_eq!(cut.report().networks, [(7, 2), (42, 1)]);
        assert_eq!(cut.routes().count(), 3);
        let result = cut.solve_mip(&MipOptions::default()).unwrap();
        assert_eq!(result.coverage_cutoff, 1.75);
        assert_eq!(result.mip.status, MipStatus::ObjectiveCutoff);
        assert_eq!(result.mip.bound, 1.75);
        assert!(result.mip.solution.is_none());
        assert!(root.snapshot.state.universe.is_none());
        assert_eq!(root.snapshot.state.routes.len(), source_routes);
        let mut options = Options::default();
        options.cg.stabilize = false;
        let resumed = root.snapshot.fork().unwrap().solve(options).unwrap();
        assert_eq!(resumed.master.status, cg::Status::Optimal);
        assert_eq!(resumed.master.lp, 1.5);
        // A larger cutoff retains the global integer optimum and its witness.
        let cut = root
            .snapshot
            .enumerate_below(3., &EnumerationOptions::default())
            .unwrap();
        let result = cut.solve_mip(&MipOptions::default()).unwrap();
        assert_eq!(result.mip.status, MipStatus::Optimal);
        assert_eq!(result.mip.solution.as_ref().unwrap().objective, 2.);
        assert_eq!(
            result.route_values.iter().map(|r| r.quantity).sum::<f64>(),
            2.
        );
        assert!(result.mip.bound <= 2.);
        let stricter = cut
            .solve_mip(&MipOptions {
                objective_cutoff: 1.75,
                ..Default::default()
            })
            .unwrap();
        assert_eq!(stricter.mip.status, MipStatus::ObjectiveCutoff);
        assert_eq!(stricter.mip.bound, 1.75);
    }
    #[test]
    fn failed_cutoff_collection_fixed_copies_and_retry_preserve_source() {
        let root = root();
        let tag = root
            .snapshot
            .state
            .routes
            .iter()
            .position(|r| r.network == 7 && r.route.arcs == [0, 1])
            .unwrap();
        let child = root.snapshot.fix_route(tag, 1.).unwrap();
        for options in [
            EnumerationOptions {
                max_work: 0,
                ..Default::default()
            },
            EnumerationOptions {
                max_routes: 0,
                ..Default::default()
            },
            EnumerationOptions {
                time_limit: Duration::ZERO,
                ..Default::default()
            },
        ] {
            assert!(child.enumerate_below(3., &options).is_err());
            assert!(child.state.universe.is_none());
        }
        let cut = child
            .enumerate_below(3., &EnumerationOptions::default())
            .unwrap();
        let mip = cut.solve_mip(&MipOptions::default()).unwrap();
        assert_eq!(mip.mip.status, MipStatus::Optimal);
        assert_eq!(mip.mip.solution.unwrap().objective, 2.);
        assert!(mip
            .route_values
            .iter()
            .any(|r| r.route.network == 7 && r.route.route.arcs == [0, 1] && r.quantity >= 1.));
        assert!(child.state.universe.is_none());
        assert!(root.snapshot.state.fixed_routes.is_empty());
        assert!(root
            .snapshot
            .enumerate_below(f64::NAN, &EnumerationOptions::default())
            .is_err());
    }
}

#[cfg(all(test, feature = "highs"))]
mod limit_tests {
    use super::*;
    use crate::master::solvers::mip::MipStatus;
    #[test]
    fn cutoff_mip_limits_and_invalid_requests_keep_the_view_retryable() {
        let problem = super::super::super::format::Model::from_json(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/examples/networks/triangle.json"
        )))
        .unwrap()
        .problem::<7>()
        .unwrap();
        let mut options = Options::default();
        options.cg.stabilize = false;
        options.cg.artificial_cost = 100.;
        let root = solve(problem, options).unwrap();
        let view = root
            .snapshot
            .enumerate_below(3., &EnumerationOptions::default())
            .unwrap();
        for (limits, status) in [
            (
                MipOptions {
                    node_limit: 0,
                    ..Default::default()
                },
                MipStatus::NodeLimit,
            ),
            (
                MipOptions {
                    time_limit: Duration::ZERO,
                    ..Default::default()
                },
                MipStatus::TimeLimit,
            ),
        ] {
            let result = view.solve_mip(&limits).unwrap();
            assert_eq!(result.mip.status, status);
            assert!(result.mip.solution.is_none());
            assert!(result.route_values.is_empty());
            assert!(result.mip.bound <= view.coverage_cutoff());
            assert_eq!(view.report().coverage_cutoff, Some(3.));
        }
        assert!(view
            .solve_mip(&MipOptions {
                objective_cutoff: f64::NAN,
                ..Default::default()
            })
            .is_err());
        let retry = view.solve_mip(&MipOptions::default()).unwrap();
        assert_eq!(retry.mip.status, MipStatus::Optimal);
        assert_eq!(retry.mip.solution.unwrap().objective, 2.);
        assert!(root.snapshot.state.universe.is_none());
    }
}
