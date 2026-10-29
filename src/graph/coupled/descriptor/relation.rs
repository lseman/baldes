//! Paired slope transitions. Unsupported arithmetic is never infeasibility.
use super::*;
use std::{sync::Arc, time::Instant};
mod backward_dominance;

#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "clp", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "clp", serde(deny_unknown_fields))]
pub struct SlopeRelationArc {
    pub arrival: PreparedSlopeFunction,
    pub increment: PreparedSlopeFunction,
    pub lower: PreparedSlopeFunction,
    pub upper: PreparedSlopeFunction,
    pub waiting: i32,
}

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

/// Immutable preparation is shared across labels; raw functions are not rebuilt.
#[derive(Clone, Debug)]
pub struct PreparedSlopeRelationArc {
    source: Arc<SlopeRelationArc>,
    policy: FrontOperationPolicy,
    shape: Result<Translation, &'static str>,
    inverse: FrontierOutcome<Arc<PreparedSlopeFunction>>,
    composition: FrontierOutcome<PreparedCompositionMap>,
    increment_front: FrontierOutcome<SlopeFront>,
    bounds: PreparedClampFilter,
}
impl PartialEq for PreparedSlopeRelationArc {
    fn eq(&self, other: &Self) -> bool {
        self.source == other.source && self.policy == other.policy
    }
}
#[derive(Clone, Debug, PartialEq)]
struct Translation {
    delta: f64,
    phi: f64,
    tail_lo: f64,
    tail_hi: f64,
    head_lo: f64,
    head_hi: f64,
}
#[derive(Clone, Copy)]
struct Segment {
    lo: f64,
    hi: f64,
    y: f64,
    slope: f64,
}

mod backward;
mod continuation;
mod extension;
mod minimum;
mod reconstruction;
// Retaining slopes through translation agrees with recovered secant
// composition only when all intermediate endpoint arithmetic is exact.
// Other geometry uses the general prepared composition operation instead.
fn exact_translation(front: &SlopeFront, t: &Translation) -> bool {
    let integer = |x: f64| x.is_finite() && x.abs() <= ((1u64 << 40) as f64) && x.fract() == 0.;
    if !integer(t.delta) || !integer(t.phi) || !integer(t.tail_lo) || !integer(t.tail_hi) {
        return false;
    }
    for f in front.components() {
        let mut x = f.pairs()[0].1;
        let mut y = f.pairs()[0].0 + t.phi;
        if !integer(x) || !integer(x + t.delta) || !integer(f.pairs()[0].0) || !integer(y) {
            return false;
        }
        for &(s, end) in &f.pairs()[1..] {
            let product = s * (end - x);
            y += product;
            if !integer(s)
                || !integer(end)
                || !integer(end + t.delta)
                || !integer(product)
                || !integer(y)
            {
                return false;
            }
            x = end;
        }
    }
    true
}
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
fn push(out: &mut Vec<Segment>, s: Segment, b: FrontierBudget) -> Result<(), FrontierUnknown> {
    if out.len() >= b.max_pairs || out.len() >= b.max_operations {
        return Err(FrontierUnknown::PairLimit);
    }
    out.push(s);
    Ok(())
}

#[cfg(test)]
#[path = "../../../../tests/unit/graph/coupled/descriptor/relation/tests.rs"]
mod tests;
