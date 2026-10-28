//! Paired raw slope relations: physical dependent state and objective stay separate.
use super::bounds::Range;
use super::*;
use descriptor::{
    FrontierBudget, FrontierOutcome, FrontierUnknown, PairedCostNormalization,
    PreparedSlopeRelationArc, SlopeFront, SlopeRelationArc,
};
use relation::{BoundedCostFront, PreparedRelationArc, RelationFront};
use slope_interval::{outcome, SlopeLabel};

#[derive(Clone, Debug)]
pub(super) struct PreparedSlopePaired {
    pub(super) objective_coefficient: f64,
    initial: SlopeLabel,
    upper_initial: Option<SlopeFront>,
    initial_minimum: f64,
    arcs: std::sync::Arc<[PreparedSlopeRelationArc]>,
    upper_arcs: Option<std::sync::Arc<[SlopeFront]>>,
    sources: std::sync::Arc<[SlopeRelationArc]>,
    backward: std::sync::Arc<[Option<PreparedRelationArc>]>,
    exact_cost: bool,
    ceiling: f64,
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
fn negated_upper_on(
    feasible: &SlopeFront,
    negative_upper: &SlopeFront,
    policy: FrontOperationPolicy,
    budget: FrontierBudget,
) -> FrontierOutcome<SlopeFront> {
    let zero_components = match feasible
        .components()
        .iter()
        .map(|component| {
            let (lower, upper) = component.domain();
            if lower == upper {
                PreparedSlopeFunction::constant_point(0., lower)
            } else {
                PreparedSlopeFunction::from_prepared_pairs(vec![(0., lower), (0., upper)])
            }
        })
        .collect::<std::result::Result<Vec<_>, _>>()
    {
        Ok(components) => components,
        Err(error) => return FrontierOutcome::Unknown(FrontierUnknown::Arithmetic(error)),
    };
    match SlopeFront::from_components(zero_components, budget) {
        FrontierOutcome::Complete(mask) => negative_upper.add(&mask, policy, budget),
        FrontierOutcome::Empty => FrontierOutcome::Empty,
        FrontierOutcome::Unknown(reason) => FrontierOutcome::Unknown(reason),
    }
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
            .map(|arc| negate_function(&arc.upper).map(SlopeFront::from_prepared))
            .collect::<Result<Vec<_>>>()
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
        let backward: std::sync::Arc<[Option<PreparedRelationArc>]> = arcs
            .iter()
            .map(|arc| {
                (policy == FrontOperationPolicy::StrictDomain)
                    .then(|| backward_arc(arc))
                    .flatten()
            })
            .collect::<Vec<_>>()
            .into();
        let integer = |x: f64| x.is_finite() && x.fract() == 0. && x.abs() <= 1_099_511_627_776.;
        // The raw/knot conversion preserves every stored segment slope. Only
        // the existing exact translation/positive-increment contract may use
        // the converted cache for cost dominance; general raw programs replay.
        let exact_cost = mode == PairedResourceMode::CostBounded
            && integer(base.lower)
            && integer(base.upper)
            && constant(initial).is_some_and(integer)
            && initial
                .pairs()
                .iter()
                .all(|pair| integer(pair.0) && integer(pair.1))
            && backward.iter().all(|arc| {
                arc.as_ref()
                    .is_some_and(PreparedRelationArc::exact_cache_eligible)
            });
        let ceiling = arcs
            .first()
            .and_then(|arc| constant(&arc.upper))
            .unwrap_or(0.);
        Ok(Self {
            objective_coefficient: 1.,
            initial: SlopeLabel::root(&front, true),
            upper_initial,
            initial_minimum: finite(witness.value)?,
            arcs: prepared.into(),
            upper_arcs,
            sources: arcs.to_vec().into(),
            backward,
            exact_cost,
            ceiling,
            mode,
            normalization,
            policy,
            shifts,
        })
    }
    /// Forward fronts cover and stay below under this resource's policy.
    /// Backward feasibility uppers cover and stay above; exact bounded-cost
    /// transfers additionally preserve the incremental suffix objective.
    /// Unknowns require replay. A shifting forward front may instead exceed by
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
            if self.has_costs() {
                return (scale.is_finite() && scale >= 0.)
                    .then_some(())
                    .and_then(|()| match (&a.bounded_cost, &b.bounded_cost) {
                        (Some(a), Some(b)) if a.dominates(b) => Some(0.),
                        _ => None,
                    });
            }
            return match (&a.upper, &b.upper) {
                (Some(a), Some(b)) if a.covers_above(b) => Some(0.),
                _ => None,
            };
        }
        if scale < 0. {
            if self.shifts || self.policy != FrontOperationPolicy::StrictDomain {
                return None;
            }
            let (Some(front_a), Some(front_b)) = (&a.upper_negative, &b.upper_negative) else {
                return None;
            };
            let (Some(lower_a), Some(lower_b)) = (&a.front, &b.front) else {
                return None;
            };
            return (lower_a.covers_below(lower_b, self.policy)
                && front_a.covers_below(front_b, self.policy))
            .then_some(0.);
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
    pub(super) fn same_arc_operation(&self, a: usize, b: usize) -> bool {
        let (a, b) = (&self.sources[a], &self.sources[b]);
        a.waiting == b.waiting
            && [&a.arrival, &a.increment, &a.lower, &a.upper]
                .into_iter()
                .zip([&b.arrival, &b.increment, &b.lower, &b.upper])
                .all(|(a, b)| {
                    a.pairs().len() == b.pairs().len()
                        && a.pairs().iter().zip(b.pairs()).all(|(a, b)| {
                            a.0.to_bits() == b.0.to_bits() && a.1.to_bits() == b.1.to_bits()
                        })
                })
    }
    pub(super) fn uses_strict_front_policy(&self) -> bool {
        self.policy == FrontOperationPolicy::StrictDomain
    }
    pub(super) fn selected_value(&self, label: &SlopeLabel, scale: f64) -> Result<f64> {
        if self.has_costs() && finite(scale * self.objective_coefficient)? < 0. {
            let upper = label.upper_negative.as_ref().ok_or_else(|| {
                crate::incomplete("signed raw paired upper frontier is unavailable")
            })?;
            let witness = outcome(upper.minimum(FrontierBudget::default()))?
                .ok_or_else(|| crate::incomplete("empty signed raw paired upper frontier"))?;
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
                bounded_cost: None,
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
            next.bounded_cost = (if self.exact_cost {
                self.backward[arc].as_ref().and_then(|prepared| {
                    if let Some(old) = &old.bounded_cost {
                        old.checked_prepend(prepared)
                    } else if first_backward {
                        BoundedCostFront::from_terminal_arc(prepared, self.ceiling)
                    } else {
                        None
                    }
                })
            } else {
                None
            })
            .map(Box::new);
            // Backward bounded cost is an incremental suffix key, so do not
            // apply the forward final-minus-initial normalization here.
            next.minimum = if scale > 0. {
                finite(
                    scale
                        * next
                            .bounded_cost
                            .as_ref()
                            .and_then(|cache| cache.minimum_additional())
                            .unwrap_or(0.),
                )?
            } else {
                0.
            };
            next.upper = (self.mode == PairedResourceMode::Feasibility)
                .then(|| self.backward[arc].as_ref())
                .flatten()
                .and_then(|prepared| {
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
        let upper_negative = self.upper_arcs.as_ref().and_then(|arcs| {
            match negated_upper_on(&front, &arcs[arc], self.policy, FrontierBudget::default()) {
                FrontierOutcome::Complete(front) => Some(front),
                FrontierOutcome::Empty | FrontierOutcome::Unknown(_) => None,
            }
        });
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
            bounded_cost: None,
        }))
    }
}
impl CoupledLabelState {
    /// Computed forward paired slope fronts. Backward states retain the operator
    /// arc program plus a feasibility upper or exact bounded-cost transfer
    /// when available; they expose no invented forward frontier.
    pub fn slope_relation_fronts(&self) -> impl Iterator<Item = Option<&SlopeFront>> {
        self.slope_relations
            .iter()
            .map(|label| label.front.as_ref())
    }
}

#[cfg(test)]
#[path = "../../../../tests/unit/graph/coupled/kernel/slope_paired/tests.rs"]
mod tests;
