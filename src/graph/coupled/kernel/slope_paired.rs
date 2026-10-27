//! Paired raw slope relations: physical dependent state and objective stay separate.
use super::bounds::Range;
use super::*;
use descriptor::{
    FrontierBudget, PairedCostNormalization, PreparedSlopeRelationArc, SlopeFront, SlopeRelationArc,
};
use relation::{PreparedRelationArc, RelationFront};
use slope_interval::{outcome, SlopeLabel};

#[derive(Clone, Debug)]
pub(super) struct PreparedSlopePaired {
    pub(super) objective_coefficient: f64,
    initial: SlopeLabel,
    upper_initial: Option<SlopeFront>,
    initial_minimum: f64,
    arcs: std::sync::Arc<[PreparedSlopeRelationArc]>,
    upper_arcs: Option<std::sync::Arc<[PreparedSlopeRelationArc]>>,
    sources: std::sync::Arc<[SlopeRelationArc]>,
    backward: std::sync::Arc<[Option<PreparedRelationArc>]>,
    mode: PairedResourceMode,
    normalization: PairedCostNormalization,
    policy: FrontOperationPolicy,
    /// Cost-unbounded native mode without a reachable upper bound (the
    /// reference's 1e12 infinity): a higher front may pay its excess.
    shifts: bool,
}
impl PartialEq for PreparedSlopePaired {
    fn eq(&self, b: &Self) -> bool {
        self.initial == b.initial
            && self.objective_coefficient == b.objective_coefficient
            && self.initial_minimum.to_bits() == b.initial_minimum.to_bits()
            && self.sources == b.sources
            && self.mode == b.mode
            && self.normalization == b.normalization
            && self.policy == b.policy
    }
}
fn constant(function: &PreparedSlopeFunction) -> Option<f64> {
    function
        .pairs()
        .iter()
        .skip(1)
        .all(|p| p.0 == 0.)
        .then(|| function.pairs()[0].0)
}
fn negate_function(function: &PreparedSlopeFunction) -> Result<PreparedSlopeFunction> {
    PreparedSlopeFunction::from_prepared_pairs(
        function
            .pairs()
            .iter()
            .map(|&(slope, coordinate)| (-slope, coordinate))
            .collect(),
    )
    .map_err(|error| invalid(error.to_string()))
}
fn negate_relation_arc(arc: &SlopeRelationArc) -> Result<SlopeRelationArc> {
    Ok(SlopeRelationArc {
        arrival: arc.arrival.clone(),
        increment: negate_function(&arc.increment)?,
        lower: negate_function(&arc.upper)?,
        upper: negate_function(&arc.lower)?,
        waiting: arc
            .waiting
            .checked_neg()
            .ok_or_else(|| invalid("signed raw paired waiting overflow"))?,
    })
}
// Conversion is advisory. Singleton domains and unrepresentable slopes keep
// suffix replay as the authoritative feasibility check.
fn relation_function(function: &PreparedSlopeFunction) -> Option<PiecewiseLinearFunction> {
    let pairs = function.pairs();
    let mut points = vec![[pairs[0].1, pairs[0].0]];
    for &(slope, end) in &pairs[1..] {
        let [start, value] = *points.last()?;
        let next = value + slope * (end - start);
        if (next - value) / (end - start) != slope {
            return None;
        }
        points.push([end, next]);
    }
    PiecewiseLinearFunction::new(points).ok()
}
fn backward_arc(arc: &SlopeRelationArc) -> Option<PreparedRelationArc> {
    PreparedRelationArc::new(&PairedResourceArc {
        arrival: relation_function(&arc.arrival)?,
        increment: relation_function(&arc.increment)?,
        lower: relation_function(&arc.lower)?,
        upper: relation_function(&arc.upper)?,
        waiting: arc.waiting,
    })
    .ok()
}
impl PreparedSlopePaired {
    pub(super) fn new(
        initial: &PreparedSlopeFunction,
        base: Window,
        arcs: &[SlopeRelationArc],
        mode: PairedResourceMode,
        normalization: PairedCostNormalization,
        policy: FrontOperationPolicy,
    ) -> Result<Self> {
        let budget = FrontierBudget::default();
        let Some(front) = outcome(
            SlopeFront::from_prepared(initial.clone())
                .restrict(base.lower, base.upper, policy, budget),
        )?
        else {
            return Err(invalid("empty raw paired initial front"));
        };
        let Some(witness) = outcome(front.minimum(budget))? else {
            return Err(invalid("empty raw paired initial front"));
        };
        let upper_initial = negate_function(initial)
            .and_then(|function| {
                Ok(SlopeFront::from_prepared(function)
                    .restrict(base.lower, base.upper, policy, budget))
            })
            .ok()
            .and_then(|front| outcome(front).ok().flatten());
        let upper_arcs = arcs
            .iter()
            .map(negate_relation_arc)
            .collect::<Result<Vec<_>>>()
            .and_then(|arcs| {
                arcs.iter()
                    .map(|arc| PreparedSlopeRelationArc::new(arc, policy))
                    .collect::<std::result::Result<Vec<_>, _>>()
                    .map_err(|error| invalid(error.to_string()))
            })
            .ok()
            .map(std::sync::Arc::from);
        let mut shifts = false;
        if mode != PairedResourceMode::Feasibility {
            let first = arcs
                .first()
                .ok_or_else(|| invalid("raw paired cost requires physical arcs"))?;
            let lower = constant(&first.lower).ok_or_else(|| {
                invalid("raw paired cost requires shared constant dependent bounds")
            })?;
            let upper = constant(&first.upper).ok_or_else(|| {
                invalid("raw paired cost requires shared constant dependent bounds")
            })?;
            if lower > upper
                || arcs.iter().any(|arc| {
                    constant(&arc.lower) != Some(lower) || constant(&arc.upper) != Some(upper)
                })
            {
                return Err(invalid(
                    "raw paired cost requires shared ordered constant dependent bounds",
                ));
            }
            shifts = mode == PairedResourceMode::CostUnbounded
                && policy == FrontOperationPolicy::FrontAlgebra
                && upper >= 1e12;
            if mode == PairedResourceMode::CostBounded {
                if witness.value < lower || arcs.iter().any(|arc| arc.waiting < 0) {
                    return Err(invalid("bounded raw paired cost requires nonnegative waiting and initial minimum above the lower bound"));
                }
                for arc in arcs {
                    let Some(minimum) =
                        outcome(SlopeFront::from_prepared(arc.increment.clone()).minimum(budget))?
                    else {
                        return Err(invalid("empty raw paired increment"));
                    };
                    if minimum.value <= 0. {
                        return Err(invalid(
                            "bounded raw paired cost requires positive increment functions",
                        ));
                    }
                }
            }
        }
        let prepared = arcs
            .iter()
            .map(|arc| {
                PreparedSlopeRelationArc::new(arc, policy)
                    .map_err(|error| invalid(error.to_string()))
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            objective_coefficient: 1.,
            initial: SlopeLabel::root(&front, true),
            upper_initial,
            initial_minimum: finite(witness.value)?,
            arcs: prepared.into(),
            upper_arcs,
            sources: arcs.to_vec().into(),
            backward: arcs
                .iter()
                .map(|arc| {
                    (mode == PairedResourceMode::Feasibility
                        && policy == FrontOperationPolicy::StrictDomain)
                        .then(|| backward_arc(arc))
                        .flatten()
                })
                .collect::<Vec<_>>()
                .into(),
            mode,
            normalization,
            policy,
            shifts,
        })
    }
    /// Forward fronts cover and stay below under this resource's policy.
    /// Backward feasibility uppers cover and stay above; unknowns require
    /// replay. A shifting resource's forward front may instead exceed by
    /// delta: every future step is max(value + increment, lower), which keeps
    /// the pointwise gap, so a's terminal minimum exceeds b's by at most delta.
    /// The objective is scale times that minimum plus a common baseline, so
    /// the penalty is scale * delta + objective_b - objective_a. Returns the
    /// penalty, or None.
    pub(super) fn penalty(
        &self,
        a: &SlopeLabel,
        b: &SlopeLabel,
        forward: bool,
        scale: f64,
    ) -> Option<f64> {
        let scale = if self.has_costs() {
            scale * self.objective_coefficient
        } else {
            0.
        };
        if a == b {
            return Some(0.);
        }
        if !forward {
            return match (&a.upper, &b.upper) {
                (Some(a), Some(b)) if a.covers_above(b) => Some(0.),
                _ => None,
            };
        }
        if scale < 0. {
            if self.shifts {
                return None;
            }
            let (Some(front_a), Some(front_b)) = (&a.upper_negative, &b.upper_negative) else {
                return None;
            };
            return front_a.covers_below(front_b, self.policy).then_some(0.);
        }
        let (Some(front_a), Some(front_b)) = (&a.front, &b.front) else {
            return None;
        };
        if !(self.shifts && scale != 0.) {
            return front_a.covers_below(front_b, self.policy).then_some(0.);
        }
        let excess = front_a.excess(front_b, self.policy)?;
        let gap = Range::point(excess)
            .multiply(Range::point(scale))
            .add(Range::point(b.minimum))
            .subtract(Range::point(a.minimum))
            .hi;
        gap.is_finite().then_some(gap.max(0.))
    }
    pub(super) fn has_costs(&self) -> bool {
        self.mode != PairedResourceMode::Feasibility
    }
    pub(super) fn selected_value(&self, label: &SlopeLabel, scale: f64) -> Result<f64> {
        if self.has_costs() && finite(scale * self.objective_coefficient)? < 0. {
            let upper = label
                .upper_negative
                .as_ref()
                .ok_or_else(|| invalid("signed raw paired upper frontier is unavailable"))?;
            let witness = outcome(upper.minimum(FrontierBudget::default()))?
                .ok_or_else(|| invalid("empty signed raw paired upper frontier"))?;
            finite(-witness.value)
        } else {
            let front = label
                .front
                .as_ref()
                .ok_or_else(|| invalid("missing raw paired front"))?;
            let witness = outcome(front.minimum(FrontierBudget::default()))?
                .ok_or_else(|| invalid("empty raw paired front"))?;
            finite(witness.value)
        }
    }
    pub(super) fn root(&self, forward: bool) -> SlopeLabel {
        if forward {
            let mut root = self.initial.clone();
            root.upper_negative = self.upper_initial.clone();
            root
        } else {
            SlopeLabel {
                front: None,
                upper_negative: None,
                minimum: 0.,
                upper: None,
            }
        }
    }
    #[cfg(feature = "clp")]
    pub(super) fn initial_minimum(&self) -> f64 {
        self.initial_minimum
    }
    pub(super) fn final_objective(&self, value: f64, scale: f64) -> Result<f64> {
        if !self.has_costs() {
            return Ok(0.);
        }
        self.normalization
            .charge_selected(
                self.initial_minimum,
                value,
                finite(scale * self.objective_coefficient)?,
            )
            .map_err(|error| invalid(error.to_string()))
    }

    pub(super) fn extend(
        &self,
        old: &SlopeLabel,
        arc: usize,
        forward: bool,
        first_backward: bool,
        scale: f64,
    ) -> Result<Option<SlopeLabel>> {
        let scale = finite(scale * self.objective_coefficient)?;
        let signed = self.has_costs() && scale < 0.;
        if !forward {
            let mut next = old.clone();
            next.upper = self.backward[arc].as_ref().and_then(|prepared| {
                let root;
                let upper = if let Some(upper) = old.upper.as_ref() {
                    upper
                } else if first_backward {
                    let arc = prepared.arc();
                    let lo = arc.lower.prepared_domain();
                    let hi = arc.upper.prepared_domain();
                    let (lower, upper) = (lo.lower.max(hi.lower), lo.upper.min(hi.upper));
                    if lower > upper || arc.upper.points.iter().any(|p| p[1] > 1e12) {
                        return None;
                    }
                    root = RelationFront::unbounded_upper(lower, upper);
                    &root
                } else {
                    return None;
                };
                upper
                    .extend_backward_upper_prepared(prepared)
                    .ok()
                    .flatten()
            });
            return Ok(Some(next));
        }
        let front = old
            .front
            .as_ref()
            .ok_or_else(|| invalid("missing raw paired forward state"))?;
        let Some(front) = outcome(self.arcs[arc].extend(front, FrontierBudget::default()))? else {
            return Ok(None);
        };
        let Some(witness) = outcome(front.minimum(FrontierBudget::default()))? else {
            return Ok(None);
        };
        let upper_negative = match (&old.upper_negative, self.upper_arcs.as_ref()) {
            (Some(old), Some(arcs)) => match arcs[arc].extend(old, FrontierBudget::default()) {
                FrontierOutcome::Complete(front) => Some(front),
                FrontierOutcome::Empty | FrontierOutcome::Unknown(_) => None,
            },
            _ => None,
        };
        if signed && upper_negative.is_none() {
            return Err(crate::incomplete(
                "signed raw paired upper frontier is unavailable",
            ));
        }
        let selected = if signed {
            let witness = outcome(
                upper_negative
                    .as_ref()
                    .expect("checked signed envelope")
                    .minimum(FrontierBudget::default()),
            )?
            .ok_or_else(|| invalid("empty signed raw paired upper frontier"))?;
            -witness.value
        } else {
            witness.value
        };
        let objective = if self.mode == PairedResourceMode::Feasibility {
            0.
        } else {
            self.normalization
                .charge_selected(self.initial_minimum, selected, scale)
                .map_err(|error| invalid(error.to_string()))?
        };
        Ok(Some(SlopeLabel {
            front: Some(front),
            upper_negative,
            minimum: finite(objective)?,
            upper: None,
        }))
    }
}
impl CoupledLabelState {
    /// Computed forward paired slope fronts. Backward states retain the operator
    /// arc program plus an advisory feasibility upper front when available.
    pub fn slope_relation_fronts(&self) -> impl Iterator<Item = Option<&SlopeFront>> {
        self.slope_relations
            .iter()
            .map(|label| label.front.as_ref())
    }
}

#[cfg(test)]
#[path = "../../../../tests/unit/graph/coupled/kernel/slope_paired/tests.rs"]
mod tests;
