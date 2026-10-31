//! Solution validation and route-file compatibility exports.
pub use crate::extensions::common::solution::{format_solution, parse_solution};
use crate::inputs::Instance;

/// Check coverage, fleet, capacity, service-start windows and depot return;
/// return the recomputed distance. No rounding is applied to the total.
pub fn validate_plan(instance: &Instance, routes: &[Vec<usize>]) -> Result<f64, String> {
    crate::extensions::common::validate_plan_with_fleet(instance, routes, true)
}

/// Replay all constraints except the fleet bound while repairing a clustered plan.
pub(super) fn validate_plan_relaxed_fleet(
    instance: &Instance,
    routes: &[Vec<usize>],
) -> Result<f64, String> {
    crate::extensions::common::validate_plan_with_fleet(instance, routes, false)
}
