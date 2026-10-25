//! Cached bounded-cost upper front and a segment-stream concatenation query.
//! This numerical cache is not itself a directed certificate. Callers retain
//! ordered replay for unproved feasibility rejection or objective arithmetic.
use super::*;

#[derive(Clone, Debug, PartialEq)]
pub(in crate::graph::coupled::kernel) struct BoundedCostFront {
    upper: RelationFront,
    ceiling: f64,
    exact: bool,
}
impl BoundedCostFront {
    pub(in crate::graph::coupled::kernel) fn terminal(
        domain: Window,
        ceiling: f64,
    ) -> Result<Self> {
        finite(domain.lower)?;
        finite(domain.upper)?;
        finite(ceiling)?;
        if domain.lower > domain.upper {
            return Err(invalid("invalid bounded cost terminal domain"));
        }
        let exact = [domain.lower, domain.upper, ceiling]
            .iter()
            .all(|x| x.fract() == 0. && x.abs() <= 1_099_511_627_776.);
        Ok(Self {
            exact,
            ceiling,
            upper: RelationFront::from_pieces(vec![RelationPiece {
                metadata: 0,
                left: [domain.lower, ceiling],
                right: [domain.upper, ceiling],
                waiting: false,
            }]),
        })
    }
    pub(in crate::graph::coupled::kernel) fn prepend(
        &self,
        arc: &PreparedRelationArc,
    ) -> Result<Option<Self>> {
        Ok(self
            .upper
            .extend_backward_upper_prepared(arc)?
            .map(|upper| Self {
                exact: upper.exact_cache_eligible(),
                upper,
                ceiling: self.ceiling,
            }))
    }
    /// None means unknown and requires ordered replay, including when the
    /// numerical adjoint reports no domain. This never proves infeasibility.
    pub(in crate::graph::coupled::kernel) fn checked_prepend(
        &self,
        arc: &PreparedRelationArc,
    ) -> Option<Self> {
        if !self.exact || !arc.exact_cache_eligible() {
            return None;
        }
        // Constant upper fronts under the restricted translation/constant-phi
        // contract need only a forced-wait ramp followed by a constant. Avoid
        // negate/clip/reflect/composition scratch for this frequent case.
        if self.upper.pieces.len() == 1
            && self.upper.pieces[0].left[1] == self.upper.pieces[0].right[1]
        {
            return self.prepend_constant(arc);
        }
        let next = self.prepend(arc).ok().flatten()?;
        next.upper.exact_cache_eligible().then_some(next)
    }
    fn prepend_constant(&self, prepared: &PreparedRelationArc) -> Option<Self> {
        let a = prepared.arc();
        let old = &self.upper.pieces[0];
        let head_lo = old.left[0]
            .max(a.lower.prepared_domain().lower)
            .max(a.upper.prepared_domain().lower);
        let head_hi = old.right[0]
            .min(a.lower.prepared_domain().upper)
            .min(a.upper.prepared_domain().upper);
        let allowance = old.left[1].min(a.upper.points[0][1]);
        if head_lo > head_hi || allowance < a.lower.points[0][1] {
            return None;
        }
        let shift = a.arrival.points[0][1] - a.arrival.points[0][0];
        let lo = a
            .arrival
            .prepared_domain()
            .lower
            .max(a.increment.prepared_domain().lower);
        let hi = a
            .arrival
            .prepared_domain()
            .upper
            .min(a.increment.prepared_domain().upper)
            .min(head_hi - shift);
        if lo > hi {
            return None;
        }
        let breakpoint = head_lo - shift;
        let constant = allowance - a.increment.points[0][1];
        let value = |t: f64| {
            let product = f64::from(a.waiting) * (breakpoint - t).max(0.);
            let y = constant - product;
            (product.is_finite()
                && product.fract() == 0.
                && product.abs() <= 1_099_511_627_776.
                && y.is_finite()
                && y.fract() == 0.
                && y.abs() <= 1_099_511_627_776.)
                .then_some(y)
        };
        let mut pieces = Vec::with_capacity(2);
        if lo < breakpoint && breakpoint < hi {
            pieces.push(RelationPiece {
                metadata: 0,
                left: [lo, value(lo)?],
                right: [breakpoint, constant],
                waiting: true,
            });
            pieces.push(RelationPiece {
                metadata: 0,
                left: [breakpoint, constant],
                right: [hi, constant],
                waiting: old.waiting,
            });
        } else {
            pieces.push(RelationPiece {
                metadata: 0,
                left: [lo, value(lo)?],
                right: [hi, value(hi)?],
                waiting: old.waiting || hi <= breakpoint,
            });
        }
        let upper = RelationFront::from_pieces(pieces);
        upper.exact_cache_eligible().then_some(Self {
            upper,
            ceiling: self.ceiling,
            exact: true,
        })
    }
    /// Outer None is unknown; Some(None) is an exact restricted-contract
    /// nonintersection, and Some(Some(value)) the exact compatible objective.
    pub(in crate::graph::coupled::kernel) fn checked_compatible_join(
        &self,
        forward: &RelationFront,
    ) -> Option<Option<f64>> {
        if !self.exact
            || !forward.exact_cache_eligible()
            || self.ceiling.fract() != 0.
            || self.ceiling.abs() > 1_099_511_627_776.
        {
            return None;
        }
        self.compatible_join_impl(forward, true).ok()
    }
    fn compatible_join_impl(&self, forward: &RelationFront, exact: bool) -> Result<Option<f64>> {
        let evaluate = |p: &RelationPiece, x: f64| -> Result<f64> {
            if !exact {
                return p.value(x);
            }
            if p.left[0] == p.right[0] {
                return Ok(p.left[1]);
            }
            let slope = (p.right[1] - p.left[1]) / (p.right[0] - p.left[0]);
            let product = finite(slope * finite(x - p.left[0])?)?;
            if product.fract() != 0. || product.abs() > 1_099_511_627_776. {
                return Err(invalid(
                    "bounded cost cache integer product is not certified",
                ));
            }
            finite(p.left[1] + product)
        };
        fn endpoints(front: &RelationFront) -> impl Iterator<Item = f64> + '_ {
            let mut previous = None;
            front
                .pieces
                .iter()
                .flat_map(|p| [p.left[0], p.right[0]])
                .filter(move |&x| {
                    let keep = previous != Some(x);
                    previous = Some(x);
                    keep
                })
        }
        let mut fxs = endpoints(forward).peekable();
        let mut bxs = endpoints(&self.upper).peekable();
        let mut fp = 0;
        let mut bp = 0;
        let mut best: Option<f64> = None;
        while fxs.peek().is_some() || bxs.peek().is_some() {
            let x = match (fxs.peek(), bxs.peek()) {
                (Some(&a), Some(&b)) => a.min(b),
                (Some(&a), None) => a,
                (None, Some(&b)) => b,
                _ => break,
            };
            if fxs.peek() == Some(&x) {
                fxs.next();
            }
            if bxs.peek() == Some(&x) {
                bxs.next();
            }
            while forward.pieces.get(fp).is_some_and(|p| p.right[0] < x) {
                fp += 1;
            }
            while self.upper.pieces.get(bp).is_some_and(|p| p.right[0] < x) {
                bp += 1;
            }
            let mut fvalue = f64::INFINITY;
            let mut bvalue = f64::NEG_INFINITY;
            for p in &forward.pieces[fp..] {
                if p.left[0] > x {
                    break;
                }
                if p.right[0] >= x {
                    fvalue = fvalue.min(evaluate(p, x)?);
                }
            }
            for p in &self.upper.pieces[bp..] {
                if p.left[0] > x {
                    break;
                }
                if p.right[0] >= x {
                    bvalue = bvalue.max(evaluate(p, x)?);
                }
            }
            if fvalue.is_finite() && bvalue.is_finite() && fvalue <= bvalue {
                let value = finite(self.ceiling + finite(fvalue - bvalue)?)?;
                best = Some(best.map_or(value, |old| old.min(value)));
            }
        }
        Ok(best)
    }
    pub(in crate::graph::coupled::kernel) fn upper(&self) -> &RelationFront {
        &self.upper
    }
}

#[cfg(test)]
#[path = "../../../../../tests/unit/graph/coupled/kernel/relation/backward/tests.rs"]
mod tests;

#[cfg(test)]
#[path = "../../../../../tests/unit/graph/coupled/kernel/relation/backward/constant_cache_tests.rs"]
mod constant_cache_tests;
