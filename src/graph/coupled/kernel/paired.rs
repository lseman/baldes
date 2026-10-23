//! Prepared paired relations and direction-specific physical-label state.
use super::*;
use relation::{BoundedCostFront, PreparedRelationArc, RelationFront, RelationWorkspace};

#[derive(Clone, Debug, PartialEq)]
pub(super) enum PairedLabel {
    Forward(RelationFront),
    Backward { cache: Option<BoundedCostFront> },
}
impl PairedLabel {
    pub(super) fn forward(&self) -> Option<&RelationFront> {
        match self {
            Self::Forward(f) => Some(f),
            _ => None,
        }
    }
}
#[derive(Clone, Debug, PartialEq)]
pub(super) struct PreparedPaired {
    initial: std::sync::Arc<RelationFront>,
    arcs: std::sync::Arc<[PreparedRelationArc]>,
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
    ) -> Result<Self> {
        let front = RelationFront::initial(initial, domain)?
            .ok_or_else(|| invalid("empty paired initial frontier"))?;
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
            initial: std::sync::Arc::new(front),
            arcs: prepared.into(),
            mode,
            exact,
            ceiling: arcs.first().map_or(0., |a| a.upper.points[0][1]),
        })
    }
    pub(super) fn has_costs(&self) -> bool {
        self.mode != PairedResourceMode::Feasibility
    }
    pub(super) fn final_objective(&self, minimum: f64, scale: f64) -> Result<f64> {
        if self.has_costs() && scale != 0. {
            finite(scale * finite(minimum - self.initial.minimum())?)
        } else {
            Ok(0.)
        }
    }
    pub(super) fn root(&self, forward: bool) -> PairedLabel {
        if forward {
            PairedLabel::Forward((*self.initial).clone())
        } else {
            PairedLabel::Backward { cache: None }
        }
    }
    pub(super) fn objective(&self, state: &PairedLabel, scale: f64) -> Result<f64> {
        match state.forward() {
            Some(f) if self.mode != PairedResourceMode::Feasibility && scale != 0. => {
                finite(scale * finite(f.minimum() - self.initial.minimum())?)
            }
            _ => Ok(0.),
        }
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
            (true, PairedLabel::Forward(front)) => Ok(front
                .extend_prepared(&self.arcs[arc], workspace)?
                .map(PairedLabel::Forward)),
            (false, PairedLabel::Backward { cache }) => {
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
                Ok(Some(PairedLabel::Backward { cache: next }))
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
        let (PairedLabel::Forward(f), PairedLabel::Backward { cache: Some(b) }) = (f, b) else {
            return None;
        };
        b.checked_compatible_join(f)
    }
    pub(super) fn replay_suffix(
        &self,
        front: &PairedLabel,
        suffix: &continuation::Continuation,
        workspace: &mut RelationWorkspace,
    ) -> Result<Option<f64>> {
        let source = front
            .forward()
            .ok_or_else(|| invalid("paired forward join expected"))?;
        let mut current: Option<RelationFront> = None;
        for arc in suffix.iter() {
            let input = current.as_ref().unwrap_or(source);
            let Some(next) = input.extend_prepared(&self.arcs[arc], workspace)? else {
                return Ok(None);
            };
            current = Some(next);
        }
        Ok(Some(current.as_ref().unwrap_or(source).minimum()))
    }
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
    let (PairedLabel::Forward(f), PairedLabel::Backward { cache: Some(b) }) = (f, b) else {
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
