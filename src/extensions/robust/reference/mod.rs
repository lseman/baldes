//! Validation-only elementary forward labeler and minimal LP master.
//!
//! This independent implementation is retained as a correctness oracle for
//! small instances. Production solves use [`Problem::pricing_graph`] with
//! `crate::master::routing::solve` or the CLI's `--robust` option.
use super::Problem;
use crate::Result;

mod master;
mod pricing;
pub use master::{
    RobustColumn, RobustColumnGenerationOptions, RobustColumnGenerationResult,
    RobustColumnGenerationStatus, RobustIteration, RobustMaster,
};
pub use pricing::{PricedRoute, PricingOptions, PricingResult, PricingStatistics, PricingStatus};

/// Independently price elementary routes for validation on small instances.
/// This forward-only oracle does not use bucket graphs, NG relaxation or cuts.
pub fn price(
    problem: &Problem,
    duals: &[f64],
    fleet_dual: f64,
    options: PricingOptions,
) -> Result<PricingResult> {
    problem.reference_price(duals, fleet_dual, options)
}
