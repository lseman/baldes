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
mod tests {
    use super::*;
    fn f(points: Vec<[f64; 2]>) -> PiecewiseLinearFunction {
        PiecewiseLinearFunction::new(points).unwrap()
    }
    #[test]
    fn prepared_workspace_reuses_buffers_without_aliasing_retained_fronts() {
        let a = RelationArc {
            arrival: f(vec![[0., 1.], [4., 5.]]),
            increment: f(vec![[0., 2.], [4., 2.]]),
            lower: f(vec![[1., 0.], [6., 0.]]),
            upper: f(vec![[1., 100.], [6., 100.]]),
            waiting: 1,
        };
        let prepared = PreparedRelationArc::new(&a).unwrap();
        assert!(prepared.exact_cache_eligible());
        let front = RelationFront::initial(
            &f(vec![[0., 0.], [4., 0.]]),
            Window {
                lower: 0.,
                upper: 0.,
            },
        )
        .unwrap()
        .unwrap();
        let mut workspace = RelationWorkspace::default();
        let first = front
            .extend_prepared(&prepared, &mut workspace)
            .unwrap()
            .unwrap();
        let saved = first.clone();
        let second = first
            .extend_prepared(&prepared, &mut workspace)
            .unwrap()
            .unwrap();
        assert_eq!(first, saved);
        assert_eq!(second.minimum(), 4.);
        assert!(!workspace.transported.is_empty());
    }
    #[test]
    fn interior_arrival_plateau_is_unknown_instead_of_false_infeasibility() {
        let base = 1e16;
        let a = RelationArc {
            arrival: f(vec![[0., base], [1., base + 2.]]),
            increment: f(vec![[0., 0.], [1., 0.]]),
            lower: f(vec![[base, -10.], [base + 2., -10.]]),
            upper: f(vec![[base, -0.2], [base + 2., -2.]]),
            waiting: 0,
        };
        let initial = f(vec![[0., 0.], [1., -1.]]);
        let t = 0.25;
        let arrival = a.arrival.value(t).unwrap().unwrap();
        assert!(initial.value(t).unwrap().unwrap() <= a.upper.value(arrival).unwrap().unwrap());
        let front = RelationFront::initial(
            &initial,
            Window {
                lower: 0.,
                upper: 1.,
            },
        )
        .unwrap()
        .unwrap();
        assert!(front.extend(&a).is_err());
    }
}
