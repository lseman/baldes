//! Immutable arc transforms and caller-owned temporary frontier storage.
use super::*;

#[derive(Clone, Debug, PartialEq)]
pub(in crate::graph::coupled::kernel) struct PreparedRelationArc {
    arc: std::sync::Arc<RelationArc>,
    exact: bool,
    pub(super) departure_knots: Vec<f64>,
    pub(in crate::graph::coupled::kernel) inverse: PiecewiseLinearFunction,
    pub(in crate::graph::coupled::kernel) arrival_charge: PiecewiseLinearFunction,
}
impl PreparedRelationArc {
    pub(in crate::graph::coupled::kernel) fn new(arc: &RelationArc) -> Result<Self> {
        arc.validate()?;
        // Inversion and arrival-space composition require distinguishable
        // interior coordinates, not merely different stored endpoints. A
        // narrow output span at a large offset is not a safe affine relation
        // under the public literal binary64 evaluator.
        for pair in arc.arrival.points.windows(2) {
            let span = pair[1][1] - pair[0][1];
            let scale = pair[0][1].abs().max(pair[1][1].abs());
            if span <= 64. * f64::EPSILON * scale {
                return Err(invalid(
                    "paired prepared arrival has an unresolved interior rounding plateau",
                ));
            }
        }
        let inverse = PiecewiseLinearFunction::new(
            arc.arrival.points.iter().map(|p| [p[1], p[0]]).collect(),
        )?;
        let ad = arc.arrival.prepared_domain();
        let id = arc.increment.prepared_domain();
        let lo = ad.lower.max(id.lower);
        let hi = ad.upper.min(id.upper);
        // Empty common domains remain an infeasible arc at runtime. The
        // prepared charge cache is not consulted outside this common domain.
        let arrival_charge = if lo < hi {
            let mut knots = vec![lo, hi];
            for p in arc.arrival.points.iter().chain(&arc.increment.points) {
                if lo < p[0] && p[0] < hi {
                    knots.push(p[0]);
                }
            }
            ordered(&mut knots);
            PiecewiseLinearFunction::new(
                knots
                    .into_iter()
                    .map(|x| Ok([evaluate(&arc.arrival, x)?, evaluate(&arc.increment, x)?]))
                    .collect::<Result<Vec<_>>>()?,
            )?
        } else {
            // A harmless placeholder; the empty-domain check precedes use.
            arc.increment.clone()
        };
        let mut departure_knots = arc
            .arrival
            .points
            .iter()
            .chain(&arc.increment.points)
            .map(|p| p[0])
            .collect();
        ordered(&mut departure_knots);
        let exact = Self::exact_arc_eligible(arc);
        Ok(Self {
            exact,
            departure_knots,
            arc: std::sync::Arc::new(arc.clone()),
            inverse,
            arrival_charge,
        })
    }
    /// Restricted exact waiting-LP contract. General numeric arcs retain
    /// ordered replay and cannot turn a cache nonintersection into pruning.
    pub(in crate::graph::coupled::kernel) fn exact_cache_eligible(&self) -> bool {
        self.exact
    }
    fn exact_arc_eligible(a: &RelationArc) -> bool {
        let integer = |x: f64| x.is_finite() && x.fract() == 0. && x.abs() <= 1_099_511_627_776.;
        let shift = a.arrival.points[0][1] - a.arrival.points[0][0];
        integer(shift)
            && shift >= 0.
            && a.waiting >= 0
            && a.arrival
                .points
                .iter()
                .all(|p| integer(p[0]) && integer(p[1]) && p[1] - p[0] == shift)
            && a.increment.points.iter().all(|p| {
                integer(p[0]) && integer(p[1]) && p[1] == a.increment.points[0][1] && p[1] > 0.
            })
            && [&a.lower, &a.upper].iter().all(|f| {
                f.points
                    .iter()
                    .all(|p| integer(p[0]) && integer(p[1]) && p[1] == f.points[0][1])
            })
            && a.waiting as f64
                * a.lower.points[0][0]
                    .abs()
                    .max(a.lower.points.last().unwrap()[0].abs())
                <= 1_099_511_627_776.
    }
    pub(in crate::graph::coupled::kernel) fn arc(&self) -> &RelationArc {
        &self.arc
    }
}

/// One workspace per pricing call/thread; it is never shared through a kernel
/// clone. Returned fronts own their retained pieces, not this scratch storage.
#[derive(Default)]
pub(in crate::graph::coupled::kernel) struct RelationWorkspace {
    pub(super) transported: Vec<RelationPiece>,
    pub(super) knots: Vec<f64>,
    pub(super) points: Vec<[f64; 2]>,
}

#[cfg(test)]
#[path = "../../../../../tests/unit/graph/coupled/kernel/relation/prepared/tests.rs"]
mod tests;
