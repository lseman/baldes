//! Prepared operands for the paired machine recurrence.
use super::*;

/// Additional prepared operands for the backward paired recurrence.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "clp", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "clp", serde(deny_unknown_fields))]
pub struct PreparedMachineBackwardInputs {
    /// Mapped tail's lower bound.
    pub tail_lower: f64,
    /// Mapped tail's upper bound.
    pub tail_upper: f64,
    /// Prepared forward vector evaluated at the tail lower bound.
    pub forward_at_tail_lower: f64,
    /// Prepared mapped upper limit.
    pub mapped_upper: f64,
    /// Prepared dependent vector before composition with the inverse.
    #[cfg_attr(
        feature = "clp",
        serde(deserialize_with = "descriptor::deserialize_machine_function")
    )]
    pub dependent: PreparedSlopeFunction,
}

/// Exact prepared machine operands for one physical arc. Installing these
/// values does not select a model domain or change pricing by itself.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "clp", derive(serde::Serialize, serde::Deserialize))]
pub struct PreparedMachineArc {
    /// Exact first scalar argument; it is not inferred from a vertex bound.
    pub tail_limit: f64,
    /// Exact third scalar argument.
    pub head_limit: f64,
    pub waiting_lower: f64,
    pub waiting_upper: f64,
    pub waiting_rate: i32,
    #[cfg_attr(
        feature = "clp",
        serde(deserialize_with = "descriptor::deserialize_machine_function")
    )]
    pub forward_mapping: PreparedSlopeFunction,
    #[cfg_attr(
        feature = "clp",
        serde(deserialize_with = "descriptor::deserialize_machine_function")
    )]
    pub inverse: PreparedSlopeFunction,
    #[cfg_attr(
        feature = "clp",
        serde(deserialize_with = "descriptor::deserialize_machine_function")
    )]
    pub sum_vector: PreparedSlopeFunction,
    #[cfg_attr(
        feature = "clp",
        serde(deserialize_with = "descriptor::deserialize_machine_function")
    )]
    pub dependent_lower: PreparedSlopeFunction,
    #[cfg_attr(
        feature = "clp",
        serde(deserialize_with = "descriptor::deserialize_machine_function")
    )]
    pub dependent_upper: PreparedSlopeFunction,
    /// Additional operands used by the backward machine recurrence.
    #[cfg_attr(
        feature = "clp",
        serde(default, skip_serializing_if = "Option::is_none")
    )]
    pub backward: Option<PreparedMachineBackwardInputs>,
}
impl PreparedMachineArc {
    /// Borrow the additional operands required by a backward transition.
    pub fn backward_inputs(&self) -> Option<descriptor::MachineBackwardInputs<'_>> {
        let backward = self.backward.as_ref()?;
        Some(descriptor::MachineBackwardInputs {
            waiting_lower: self.waiting_lower,
            waiting_upper: self.waiting_upper,
            forward_lower: backward.forward_at_tail_lower,
            mapped_upper: backward.mapped_upper,
            tail_lower: backward.tail_lower,
            tail_upper: backward.tail_upper,
            waiting_rate: self.waiting_rate,
            mapping: &self.forward_mapping,
            inverse: &self.inverse,
            dependent: &backward.dependent,
            dependent_lower: &self.dependent_lower,
            dependent_upper: &self.dependent_upper,
        })
    }
    /// Borrow the exact operands for the explicit machine continuation API.
    pub fn continuation_inputs(&self) -> descriptor::MachineContinuationInputs<'_> {
        descriptor::MachineContinuationInputs {
            tail_limit: self.tail_limit,
            head_limit: self.head_limit,
            forward_mapping: &self.forward_mapping,
            inverse: &self.inverse,
            sum_vector: &self.sum_vector,
            waiting_lower: self.waiting_lower,
            waiting_upper: self.waiting_upper,
            waiting_rate: self.waiting_rate,
            dependent_lower: &self.dependent_lower,
            dependent_upper: &self.dependent_upper,
        }
    }
    pub(super) fn valid(&self) -> bool {
        [
            self.tail_limit,
            self.head_limit,
            self.waiting_lower,
            self.waiting_upper,
        ]
        .iter()
        .all(|value| value.is_finite())
            && self.waiting_lower <= self.waiting_upper
            && [
                &self.forward_mapping,
                &self.inverse,
                &self.sum_vector,
                &self.dependent_lower,
                &self.dependent_upper,
            ]
            .iter()
            .all(|function| {
                function
                    .pairs()
                    .iter()
                    .all(|(a, b)| a.is_finite() && b.is_finite())
            })
            && self.backward.as_ref().map_or(true, |backward| {
                [
                    backward.tail_lower,
                    backward.tail_upper,
                    backward.forward_at_tail_lower,
                    backward.mapped_upper,
                ]
                .iter()
                .all(|value| value.is_finite())
                    && backward.tail_lower <= backward.tail_upper
                    && backward
                        .dependent
                        .pairs()
                        .iter()
                        .all(|(a, b)| a.is_finite() && b.is_finite())
            })
    }
    fn same_bits(&self, other: &Self) -> bool {
        fn function(a: &PreparedSlopeFunction, b: &PreparedSlopeFunction) -> bool {
            a.has_machine_only_geometry() == b.has_machine_only_geometry()
                && a.pairs().len() == b.pairs().len()
                && a.pairs()
                    .iter()
                    .zip(b.pairs())
                    .all(|(a, b)| a.0.to_bits() == b.0.to_bits() && a.1.to_bits() == b.1.to_bits())
        }
        self.tail_limit.to_bits() == other.tail_limit.to_bits()
            && self.head_limit.to_bits() == other.head_limit.to_bits()
            && self.waiting_lower.to_bits() == other.waiting_lower.to_bits()
            && self.waiting_upper.to_bits() == other.waiting_upper.to_bits()
            && self.waiting_rate == other.waiting_rate
            && function(&self.forward_mapping, &other.forward_mapping)
            && function(&self.inverse, &other.inverse)
            && function(&self.sum_vector, &other.sum_vector)
            && function(&self.dependent_lower, &other.dependent_lower)
            && function(&self.dependent_upper, &other.dependent_upper)
            && match (&self.backward, &other.backward) {
                (None, None) => true,
                (Some(a), Some(b)) => {
                    a.tail_lower.to_bits() == b.tail_lower.to_bits()
                        && a.tail_upper.to_bits() == b.tail_upper.to_bits()
                        && a.forward_at_tail_lower.to_bits() == b.forward_at_tail_lower.to_bits()
                        && a.mapped_upper.to_bits() == b.mapped_upper.to_bits()
                        && function(&a.dependent, &b.dependent)
                }
                _ => false,
            }
    }
}
pub(super) fn machine_inputs_equal(
    a: Option<&[PreparedMachineArc]>,
    b: Option<&[PreparedMachineArc]>,
) -> bool {
    match (a, b) {
        (None, None) => true,
        (Some(a), Some(b)) => a.len() == b.len() && a.iter().zip(b).all(|(a, b)| a.same_bits(b)),
        _ => false,
    }
}
