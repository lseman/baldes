//! Explicit affine descriptors and prepared slope functions.
//!
//! Prepared pairs are `(initial_value, initial_coordinate)`, followed by
//! `(slope, segment_end)`. They are not coordinate/value knots. Evaluation keeps
//! the ordered subtraction, multiplication, and addition of this encoding.
mod adjoint;
mod bounds;
pub use adjoint::PreparedSlopeAdjointArc;
mod composition;
mod concatenation;
pub use concatenation::{
    MachineConcatenation, MachineConcatenationScan, MachineJoinedPathReconstruction, MachineLabelConcatenation,
};
mod normalization;
pub use bounds::{clamp_filter, PreparedClampFilter};
pub use composition::{
    compose_increasing, inverse_increasing, machine_initial_limits, machine_inverse,
    prepare_machine_arc, prepare_machine_inputs, MachineArcPreparation, MachineInitialLimits,
    MachinePreparedInputs, PreparedCompositionMap,
};
mod relation;
mod slope;
pub use relation::{
    MachineBackwardInputs, MachineContinuationInputs, MachinePathReconstruction,
    PreparedSlopeRelationArc, SlopeRelationArc,
};

pub use normalization::PairedCostNormalization;
#[cfg(feature = "clp")]
pub(crate) use slope::deserialize_machine_function;
pub use slope::{EvaluationPolicy, PreparedSlopeFunction, RawAffine, RawFunctionError};

mod frontier;
pub use frontier::{
    FrontAlgebraWitness, FrontOperationPolicy, FrontierBudget, FrontierOutcome, FrontierProvenance,
    FrontierUnknown, MachineForwardState, MachinePathHistory, MachinePathSnapshot,
    MachineRootInput, MachineRootState, MachineWaitingEnvelope, SlopeFront, SourceReplayWitness,
};
