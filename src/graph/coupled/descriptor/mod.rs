//! Explicit affine descriptors and prepared slope functions.
//!
//! Prepared pairs are `(initial_value, initial_coordinate)`, followed by
//! `(slope, segment_end)`. They are not coordinate/value knots. Evaluation keeps
//! the ordered subtraction, multiplication, and addition of this encoding.
mod bounds;
mod composition;
mod normalization;
pub use bounds::{clamp_filter, PreparedClampFilter};
pub use composition::{compose_increasing, inverse_increasing, PreparedCompositionMap};
mod relation;
mod slope;
pub use relation::{PreparedSlopeRelationArc, SlopeRelationArc};

pub use normalization::PairedCostNormalization;
pub use slope::{EvaluationPolicy, PreparedSlopeFunction, RawAffine, RawFunctionError};

mod frontier;
pub use frontier::{
    FrontAlgebraWitness, FrontOperationPolicy, FrontierBudget, FrontierOutcome, FrontierProvenance,
    FrontierUnknown, SlopeFront, SourceReplayWitness,
};
