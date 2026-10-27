//! Prepared paired relations and direction-specific physical-label state.
use super::*;
use relation::{
    BoundedCostFront, PreparedRelationArc, RelationArc, RelationFront, RelationWorkspace,
};

#[derive(Clone, Debug, PartialEq)]
pub(super) enum PairedLabel {
    Forward {
        lower: RelationFront,
        upper_negative: Option<RelationFront>,
    },
    /// `upper`: in feasibility mode, the highest dependent value at each base
    /// value from which the suffix stays feasible; None when unknown.
    Backward {
        cache: Option<BoundedCostFront>,
        upper: Option<RelationFront>,
    },
}
impl PairedLabel {
    pub(super) fn forward(&self) -> Option<&RelationFront> {
        match self {
            Self::Forward { lower, .. } => Some(lower),
            _ => None,
        }
    }
    /// Negated maximum feasible dependent value, when its advisory recurrence
    /// stayed representable. Its minimum is the negative of the attainable
    /// maximum and therefore supports a negative linear objective coefficient.
    pub(super) fn forward_upper_negative(&self) -> Option<&RelationFront> {
        match self {
            Self::Forward { upper_negative, .. } => upper_negative.as_ref(),
            _ => None,
        }
    }
    pub(super) fn backward_upper(&self) -> Option<&RelationFront> {
        match self {
            Self::Backward { upper, .. } => upper.as_ref(),
            _ => None,
        }
    }
}
#[derive(Clone, Debug, PartialEq)]
pub(super) struct PreparedPaired {
    pub(super) objective_coefficient: f64,
    pub(super) normalization: PairedCostNormalization,
    initial: std::sync::Arc<RelationFront>,
    upper_initial: Option<std::sync::Arc<RelationFront>>,
    arcs: std::sync::Arc<[PreparedRelationArc]>,
    upper_arcs: Option<std::sync::Arc<[PreparedRelationArc]>>,
    mode: PairedResourceMode,
    exact: bool,
    ceiling: f64,
}
impl PreparedPaired {
    pub(super) fn new(
        initial: &PiecewiseLinearFunction,
        domain: Window,
        arcs: &[PairedResourceArc],
        mode: PairedResourceMode,
        normalization: PairedCostNormalization,
    ) -> Result<Self> {
        let front = RelationFront::initial(initial, domain)?
            .ok_or_else(|| invalid("empty paired initial frontier"))?;
        // The signed front is advisory. If negation/preparation overflows or
        // becomes numerically unsupported, retain the valid ordinary front.
        let signed = (|| -> Result<_> {
            let initial = negate_function(initial)?;
            let initial = RelationFront::initial(&initial, domain)?;
            let initial = initial.ok_or_else(|| invalid("empty signed paired frontier"))?;
            let arcs = arcs
                .iter()
                .map(negate_relation_arc)
                .map(|arc| arc.and_then(|arc| PreparedRelationArc::new(&arc)))
                .collect::<Result<Vec<_>>>()?;
            Ok((std::sync::Arc::new(initial), std::sync::Arc::from(arcs)))
        })()
        .ok();
        let prepared = arcs
            .iter()
            .map(PreparedRelationArc::new)
            .collect::<Result<Vec<_>>>()?;
        let integer = |x: f64| x.is_finite() && x.fract() == 0. && x.abs() <= 1_099_511_627_776.;
        let exact = mode == PairedResourceMode::CostBounded
            && integer(domain.lower)
            && integer(domain.upper)
            && initial
                .points
                .iter()
                .all(|p| integer(p[0]) && integer(p[1]) && p[1] == initial.points[0][1])
            && prepared
                .iter()
                .all(PreparedRelationArc::exact_cache_eligible);
        Ok(Self {
            objective_coefficient: 1.,
            normalization,
            initial: std::sync::Arc::new(front),
            upper_initial: signed.as_ref().map(|(initial, _)| initial.clone()),
            arcs: prepared.into(),
            upper_arcs: signed.map(|(_, arcs)| arcs),
            mode,
            exact,
            ceiling: arcs.first().map_or(0., |a| a.upper.points[0][1]),
        })
    }
    #[cfg(feature = "clp")]
    pub(super) fn initial_minimum(&self) -> f64 {
        self.initial.minimum()
    }
    pub(super) fn has_costs(&self) -> bool {
        self.mode != PairedResourceMode::Feasibility
    }
    /// Cost-unbounded mode with no reachable upper bound (the reference's
    /// 1e12 infinity): every future step is max(value, lower), which grows
    /// by at most a common shift, so a higher front can pay its excess.
    pub(super) fn shifts(&self) -> bool {
        self.mode == PairedResourceMode::CostUnbounded && self.ceiling >= 1e12
    }
    pub(super) fn final_objective(&self, minimum: f64, scale: f64) -> Result<f64> {
        let scale = finite(scale * self.objective_coefficient)?;
        if self.has_costs() && scale != 0. {
            self.normalization
                .charge_selected(self.initial.minimum(), minimum, scale)
                .map_err(|error| invalid(error.to_string()))
        } else {
            Ok(0.)
        }
    }
    pub(super) fn root(&self, forward: bool) -> PairedLabel {
        if forward {
            PairedLabel::Forward {
                lower: (*self.initial).clone(),
                upper_negative: self.upper_initial.as_ref().map(|front| (**front).clone()),
            }
        } else {
            PairedLabel::Backward {
                cache: None,
                upper: None,
            }
        }
    }
    pub(super) fn objective(&self, state: &PairedLabel, scale: f64) -> Result<f64> {
        let scale = finite(scale * self.objective_coefficient)?;
        if self.mode == PairedResourceMode::Feasibility || scale == 0. {
            return Ok(0.);
        }
        let selected = if scale < 0. {
            -state
                .forward_upper_negative()
                .ok_or_else(|| invalid("signed paired upper frontier is unavailable"))?
                .minimum()
        } else {
            state
                .forward()
                .ok_or_else(|| invalid("paired forward join expected"))?
                .minimum()
        };
        self.normalization
            .charge_selected(self.initial.minimum(), selected, scale)
            .map_err(|error| invalid(error.to_string()))
    }
    pub(super) fn extend(
        &self,
        state: &PairedLabel,
        arc: usize,
        forward: bool,
        first_backward: bool,
        workspace: &mut RelationWorkspace,
    ) -> Result<Option<PairedLabel>> {
        match (forward, state) {
            (
                true,
                PairedLabel::Forward {
                    lower,
                    upper_negative,
                },
            ) => {
                let Some(lower) = lower.extend_prepared(&self.arcs[arc], workspace)? else {
                    return Ok(None);
                };
                let upper_negative = match (upper_negative, self.upper_arcs.as_ref()) {
                    (Some(old), Some(arcs)) => {
                        old.extend_prepared(&arcs[arc], workspace).ok().flatten()
                    }
                    _ => None,
                };
                Ok(Some(PairedLabel::Forward {
                    lower,
                    upper_negative,
                }))
            }
            (false, PairedLabel::Backward { cache, upper }) => {
                let next = if self.exact {
                    if let Some(old) = cache {
                        old.checked_prepend(&self.arcs[arc])
                    } else if first_backward {
                        let a = self.arcs[arc].arc();
                        let lo = a.lower.prepared_domain();
                        let hi = a.upper.prepared_domain();
                        let domain = Window {
                            lower: lo.lower.max(hi.lower),
                            upper: lo.upper.min(hi.upper),
                        };
                        BoundedCostFront::terminal(domain, self.ceiling)
                            .ok()
                            .and_then(|root| root.checked_prepend(&self.arcs[arc]))
                    } else {
                        None
                    }
                } else {
                    None
                };
                // The adjoint is advisory: a numerical failure leaves the
                // front unknown and never rejects the suffix.
                let upper = if self.mode == PairedResourceMode::Feasibility {
                    match upper {
                        Some(old) => old
                            .extend_backward_upper_prepared(&self.arcs[arc])
                            .ok()
                            .flatten(),
                        None if first_backward => {
                            let a = self.arcs[arc].arc();
                            let lo = a.lower.prepared_domain();
                            let hi = a.upper.prepared_domain();
                            let (lower, upper) = (lo.lower.max(hi.lower), lo.upper.min(hi.upper));
                            // The numerical infinity root must not truncate a
                            // genuinely larger terminal dependent domain.
                            (lower <= upper && a.upper.points.iter().all(|p| p[1] <= 1e12))
                                .then(|| RelationFront::unbounded_upper(lower, upper))
                                .and_then(|root| {
                                    root.extend_backward_upper_prepared(&self.arcs[arc])
                                        .ok()
                                        .flatten()
                                })
                        }
                        None => None,
                    }
                } else {
                    None
                };
                Ok(Some(PairedLabel::Backward { cache: next, upper }))
            }
            _ => Err(invalid("paired label direction mismatch")),
        }
    }
    /// A proven restricted-cache result or unknown. General numerical
    /// nonintersection never deletes a route; ordered replay remains fallback.
    pub(super) fn cached_join(&self, f: &PairedLabel, b: &PairedLabel) -> Option<Option<f64>> {
        if !self.exact {
            return None;
        }
        let (PairedLabel::Forward { lower: f, .. }, PairedLabel::Backward { cache: Some(b), .. }) =
            (f, b)
        else {
            return None;
        };
        b.checked_compatible_join(f)
    }
    pub(super) fn replay_suffix(
        &self,
        front: &PairedLabel,
        suffix: &continuation::Continuation,
        workspace: &mut RelationWorkspace,
        maximize: bool,
    ) -> Result<Option<f64>> {
        let source = front
            .forward()
            .ok_or_else(|| invalid("paired forward join expected"))?;
        let mut current: Option<RelationFront> = None;
        let mut current_upper = front.forward_upper_negative().cloned();
        for arc in suffix.iter() {
            let input = current.as_ref().unwrap_or(source);
            let Some(next) = input.extend_prepared(&self.arcs[arc], workspace)? else {
                return Ok(None);
            };
            current = Some(next);
            current_upper = match (current_upper.as_ref(), self.upper_arcs.as_ref()) {
                (Some(input), Some(arcs)) => {
                    input.extend_prepared(&arcs[arc], workspace).ok().flatten()
                }
                _ => None,
            };
        }
        if maximize {
            return current_upper
                .map(|upper| Some(-upper.minimum()))
                .ok_or_else(|| invalid("signed paired upper frontier is unavailable"));
        }
        Ok(Some(current.as_ref().unwrap_or(source).minimum()))
    }
}

fn negate_function(function: &PiecewiseLinearFunction) -> Result<PiecewiseLinearFunction> {
    PiecewiseLinearFunction::new(
        function
            .points
            .iter()
            .map(|point| [point[0], -point[1]])
            .collect(),
    )
}

fn negate_relation_arc(arc: &RelationArc) -> Result<RelationArc> {
    Ok(RelationArc {
        arrival: arc.arrival.clone(),
        increment: negate_function(&arc.increment)?,
        lower: negate_function(&arc.upper)?,
        upper: negate_function(&arc.lower)?,
        waiting: arc
            .waiting
            .checked_neg()
            .ok_or_else(|| invalid("signed paired waiting coefficient overflow"))?,
    })
}

/// Borrowed subdivision is only an indexing precheck. Whole original pieces
/// and global normalization survive; hull overlap never proves feasibility.
pub(super) fn domains_may_intersect(
    f: &PairedLabel,
    b: &PairedLabel,
    policy: bucketing::FrontBucketPolicy,
) -> bool {
    if policy == bucketing::FrontBucketPolicy::Disabled {
        return true;
    }
    let (PairedLabel::Forward { lower: f, .. }, PairedLabel::Backward { cache: Some(b), .. }) =
        (f, b)
    else {
        return true;
    };
    bucketing::FrontBucketViews::new(f, policy).any(|view| {
        let Some([lo, hi]) = view.base_hull() else {
            return false;
        };
        debug_assert!(std::ptr::eq(view.source, f));
        debug_assert_eq!(view.normalization_minimum, f.minimum());
        b.upper()
            .pieces()
            .iter()
            .any(|p| lo <= p.right[0] && p.left[0] <= hi)
    })
}
