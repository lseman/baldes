//! Independent raw slope-front intervals with ordered suffix-operator replay.
use super::*;
use descriptor::{FrontierBudget, FrontierOutcome, SlopeFront};

#[derive(Clone, Debug)]
pub(super) struct SlopeLabel {
    pub(super) front: Option<SlopeFront>,
    pub(super) minimum: f64,
    pub(super) upper: Option<relation::RelationFront>,
}
impl PartialEq for SlopeLabel {
    fn eq(&self, other: &Self) -> bool {
        if self.minimum.to_bits() != other.minimum.to_bits() || self.upper != other.upper {
            return false;
        }
        match (&self.front, &other.front) {
            (None, None) => true,
            (Some(a), Some(b)) => {
                a.used_tolerant_alignment() == b.used_tolerant_alignment()
                    && a.components().len() == b.components().len()
                    && a.components().iter().zip(b.components()).all(|(a, b)| {
                        a.pairs().len() == b.pairs().len()
                            && a.pairs().iter().zip(b.pairs()).all(|(a, b)| {
                                a.0.to_bits() == b.0.to_bits() && a.1.to_bits() == b.1.to_bits()
                            })
                    })
            }
            _ => false,
        }
    }
}
/// Shared leaf fronts prepared once per arc, including phase-I domain-only
/// fronts. The zero objective field denotes unexecuted metadata, not its value.
#[derive(Clone, Debug, PartialEq)]
pub(super) struct PreparedSlopeCharge {
    normal: SlopeLabel,
    suppressed: SlopeLabel,
}
impl PreparedSlopeCharge {
    pub(super) fn new(function: &PreparedSlopeFunction) -> Result<Self> {
        let normal = SlopeFront::from_prepared(function.clone());
        let zeros = function
            .pairs()
            .iter()
            .map(|&(_, coordinate)| (0., coordinate))
            .collect();
        let zero = PreparedSlopeFunction::from_prepared_pairs(zeros)
            .map_err(|error| invalid(error.to_string()))?;
        let suppressed = SlopeFront::from_prepared(zero);
        Ok(Self {
            normal: SlopeLabel::root(&normal, true),
            suppressed: SlopeLabel::root(&suppressed, true),
        })
    }
    fn front(&self, scale: f64) -> Result<&SlopeFront> {
        let label = if scale == 0. {
            &self.suppressed
        } else if scale == 1. {
            &self.normal
        } else {
            return Err(crate::incomplete(
                "unsupported raw slope-front objective scale",
            ));
        };
        Ok(label.front.as_ref().expect("prepared charge front"))
    }
}
/// Per-resource facts behind forward slope-interval dominance. Under the
/// strict policy unequal costly fronts never compare: a pointwise order is
/// not preserved by future operations, because translation re-rounds every
/// segment span and addition re-accumulates split segments, so an earlier gap
/// can reverse. The native policy instead follows the reference's tolerant
/// front comparison; it is a parity rule, not a binary64 proof.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct PreparedSlopeDominance {
    policy: FrontOperationPolicy,
    zero_charges: bool,
}
impl PreparedSlopeDominance {
    pub(super) fn new(definition: &CoupledResource) -> Self {
        let CoupledResource::SlopeIntervalCost {
            functions, policy, ..
        } = definition
        else {
            unreachable!()
        };
        Self {
            policy: *policy,
            zero_charges: functions.iter().all(zero_valued),
        }
    }
    /// Continuation penalty of unequal forward labels a and b, or None.
    ///
    /// Strict policy: both fronts are identically zero, every future charge is
    /// zero (or suppressed) and each component of b lies in one component of
    /// a. Zero values survive every operation exactly, and literal
    /// translation, restriction and strict intersection are monotone in each
    /// coordinate, so a stays feasible wherever b does at the same zero
    /// minimum; the penalty is zero.
    ///
    /// Native policy: a covers b's domain and exceeds it by at most delta
    /// (values within the reference tolerance count as equal). Translation,
    /// restriction and common charges keep that pointwise gap, so a's final
    /// minimum exceeds b's by at most delta, paid as
    /// scale * (delta + min_b - min_a) against the current objectives. A
    /// charged comparison needs `tracked_order`, the independent scalar order.
    pub(super) fn penalty(
        &self,
        a: &SlopeLabel,
        b: &SlopeLabel,
        scale: f64,
        tracked_order: bool,
    ) -> Option<f64> {
        let (Some(front_a), Some(front_b)) = (&a.front, &b.front) else {
            return None;
        };
        match self.policy {
            FrontOperationPolicy::StrictDomain => ((scale == 0. || self.zero_charges)
                && front_a
                    .components()
                    .iter()
                    .chain(front_b.components())
                    .all(zero_valued)
                && front_b.components().iter().all(|b| {
                    let (lo, hi) = b.domain();
                    front_a.components().iter().any(|a| {
                        let (start, end) = a.domain();
                        start <= lo && hi <= end
                    })
                }))
            .then_some(0.),
            FrontOperationPolicy::NativeFrontAlgebra => {
                if scale != 0. && !tracked_order {
                    return None;
                }
                let excess = front_a.excess(front_b, self.policy)?;
                let gap = bounds::Range::point(excess)
                    .multiply(bounds::Range::point(scale))
                    .add(bounds::Range::point(b.minimum))
                    .subtract(bounds::Range::point(a.minimum))
                    .hi;
                gap.is_finite().then_some(gap.max(0.))
            }
        }
    }
}
fn zero_valued(function: &PreparedSlopeFunction) -> bool {
    function.pairs().iter().all(|pair| pair.0 == 0.)
}
pub(super) fn outcome<T>(value: FrontierOutcome<T>) -> Result<Option<T>> {
    match value {
        FrontierOutcome::Complete(value) => Ok(Some(value)),
        FrontierOutcome::Empty => Ok(None),
        FrontierOutcome::Unknown(reason) => Err(crate::incomplete(format!(
            "raw slope-front operation is unknown: {reason:?}"
        ))),
    }
}
impl SlopeLabel {
    pub(super) fn root(front: &SlopeFront, forward: bool) -> Self {
        Self {
            front: forward.then(|| front.clone()),
            minimum: 0.,
            upper: None,
        }
    }
    pub(super) fn extend(
        &self,
        definition: &CoupledResource,
        charge: &PreparedSlopeCharge,
        arc: usize,
        forward: bool,
        scale: f64,
    ) -> Result<Option<Self>> {
        if !forward {
            return Ok(Some(self.clone()));
        }
        let CoupledResource::SlopeIntervalCost {
            bounds,
            increments,
            policy,
            ..
        } = definition
        else {
            unreachable!()
        };
        let budget = FrontierBudget::default();
        let old = self
            .front
            .as_ref()
            .ok_or_else(|| invalid("missing forward raw slope front"))?;
        let Some(translated) = outcome(old.translate(increments[arc], budget))? else {
            return Ok(None);
        };
        let Some(restricted) =
            outcome(translated.restrict(bounds.lower, bounds.upper, *policy, budget))?
        else {
            return Ok(None);
        };
        // The phase-I leaf is precomputed with identical full domain and
        // zero values. No raw pair Vec is cloned or revalidated per extension.
        let charge = charge.front(scale)?;
        let Some(front) = outcome(restricted.add(charge, *policy, budget))? else {
            return Ok(None);
        };
        let Some(witness) = outcome(front.minimum(budget))? else {
            return Ok(None);
        };
        Ok(Some(Self {
            front: Some(front),
            minimum: finite(witness.value)?,
            upper: None,
        }))
    }
}
impl CoupledKernel {
    pub(crate) fn has_slope_fronts(&self) -> bool {
        !self.slope_indices.is_empty() || !self.slope_paired.is_empty()
    }
}
impl CoupledLabelState {
    /// Computed raw operator fronts, in slope-interval declaration order.
    /// Backward labels retain an arc program and expose no invented front.
    pub fn slope_cost_fronts(&self) -> impl Iterator<Item = Option<&SlopeFront>> {
        self.slope_fronts.iter().map(|label| label.front.as_ref())
    }
}

#[cfg(test)]
#[path = "../../../../tests/unit/graph/coupled/kernel/slope_interval/tests.rs"]
mod tests;
