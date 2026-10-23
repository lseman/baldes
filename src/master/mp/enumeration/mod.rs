//! Atomic all-network full-domain transfer. No reduced-cost cutoff is used.
pub(super) use super::route_bounds::upper as route_upper;
use super::*;
use crate::{
    enumeration::{CompleteResourceRelaxation, CostBoundedResourceRelaxation, PathDomainLimits},
    master::solvers::mip::{MipOptions, MipOutput},
};
use std::time::Duration;
mod collection;
mod cutoff;
mod mip;
mod pricing;
mod transfer;
use collection::{collect_routes, collect_routes_with_plan};
pub use cutoff::{CutoffCostBound, CutoffEnumeration};
pub(super) use pricing::price_universe as price;
use transfer::install_collected_with;
pub(super) use transfer::transfer;
#[cfg(all(test, feature = "highs"))]
use transfer::transfer_with;

#[derive(Clone, Debug)]
pub struct EnumerationOptions {
    pub max_routes: usize,
    pub max_work: usize,
    pub max_stored_arcs: usize,
    pub time_limit: Duration,
}
impl Default for EnumerationOptions {
    fn default() -> Self {
        Self {
            max_routes: 10000,
            max_work: 10000000,
            max_stored_arcs: 1000000,
            time_limit: Duration::from_secs(1),
        }
    }
}
#[derive(Clone, Debug, Default, serde::Serialize)]
pub struct EnumerationReport {
    pub attempts: usize,
    pub complete: bool,
    /// None covers the full domain; Some covers only objectives below this cutoff.
    pub coverage_cutoff: Option<f64>,
    pub networks: Vec<(i32, usize)>,
    pub routes: usize,
    pub transferred: usize,
    pub failure: Option<String>,
}
#[derive(Debug)]
pub struct EnumeratedMipOutput {
    pub mip: MipOutput,
    pub route_values: Vec<FixedRoute>,
    pub master_values: Vec<(i32, f64)>,
    /// Full original elementary domain: no objective cutoff excluded routes.
    pub coverage_cutoff: f64,
}

impl<const D: usize> NodeSnapshot<D> {
    pub fn enumeration_report(&self) -> &EnumerationReport {
        &self.state.enumeration_report
    }
    /// Prepare a private complete master. Failure keeps all original columns and
    /// pricing state, with an explicit unavailable report; the parent is untouched.
    pub fn enumerate(&self, options: &EnumerationOptions) -> Result<Self> {
        let mut node = self.fork()?;
        node.state.deadline = Instant::now()
            .checked_add(options.time_limit)
            .ok_or_else(|| invalid("enumeration time limit"))?;
        let mut cg = cg::Options::default();
        cg.warm_start = Some(node.warm);
        transfer(&mut node.master, &mut node.state, &mut cg, options)?;
        node.warm = cg.warm_start.unwrap();
        Ok(node)
    }
}

#[cfg(all(test, feature = "highs"))]
#[path = "../../../../tests/unit/master/mp/enumeration/tests.rs"]
mod tests;
