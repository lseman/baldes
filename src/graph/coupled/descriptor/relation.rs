//! Paired slope transitions. Unsupported arithmetic is never infeasibility.
use super::*;
use std::time::Instant;
mod backward_dominance;
mod forward_dominance;

/// Explicit inputs to the recovered machine-front continuation. The forward
/// lookup, inverse and sum vector are distinct prepared binary operands.
/// These inputs do not assert source-control feasibility.
pub struct MachineContinuationInputs<'a> {
    pub tail_limit: f64,
    pub head_limit: f64,
    pub forward_mapping: &'a PreparedSlopeFunction,
    pub inverse: &'a PreparedSlopeFunction,
    pub sum_vector: &'a PreparedSlopeFunction,
    pub waiting_lower: f64,
    pub waiting_upper: f64,
    pub waiting_rate: i32,
    pub dependent_lower: &'a PreparedSlopeFunction,
    pub dependent_upper: &'a PreparedSlopeFunction,
}

/// Prepared operands for a backward machine transition.
pub struct MachineBackwardInputs<'a> {
    pub waiting_lower: f64,
    pub waiting_upper: f64,
    pub forward_lower: f64,
    pub mapped_upper: f64,
    pub tail_lower: f64,
    pub tail_upper: f64,
    pub waiting_rate: i32,
    pub mapping: &'a PreparedSlopeFunction,
    pub inverse: &'a PreparedSlopeFunction,
    pub dependent: &'a PreparedSlopeFunction,
    pub dependent_lower: &'a PreparedSlopeFunction,
    pub dependent_upper: &'a PreparedSlopeFunction,
}

/// Machine-only reconstruction from a joined base/dependent value through a
/// backward suffix. Values retain the solution recorder's ordered arithmetic;
/// they do not certify a source model or a normalized objective.
#[derive(Clone, Debug)]
pub struct MachinePathReconstruction {
    pub base: Vec<f64>,
    pub dependent: Vec<f64>,
    /// One waiting amount per traversed arc, in physical traversal order.
    pub waiting: Vec<f64>,
}

/// Operations on prepared paired front-program operands.
pub struct PairedFrontOperations;

mod backward;
mod continuation;
mod minimum;
mod reconstruction;
fn finite(x: f64) -> Result<f64, FrontierUnknown> {
    if x.is_finite() {
        Ok(x)
    } else {
        Err(FrontierUnknown::Arithmetic(
            RawFunctionError::ArithmeticOverflow("paired slope operation"),
        ))
    }
}
fn near(a: f64, b: f64) -> Result<bool, FrontierUnknown> {
    let distance = finite(a - b)?.abs();
    Ok(distance < 0.0001 || distance < finite(finite(a + b)?.abs() * 2.220446049250313e-10)?)
}
fn value(y: f64, s: f64, dx: f64) -> Result<f64, FrontierUnknown> {
    finite(y + finite(s * dx)?)
}

#[cfg(test)]
#[path = "../../../../tests/unit/graph/coupled/descriptor/relation/tests.rs"]
mod tests;
