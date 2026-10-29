//! Cached bounded-cost upper front and a segment-stream concatenation query.
//! This numerical cache is not itself a directed certificate. Callers retain
//! ordered replay for unproved feasibility rejection or objective arithmetic.
use super::*;

#[derive(Clone, Debug, PartialEq)]
pub(in crate::graph::coupled::kernel) struct BoundedCostFront {
    upper: RelationFront,
    ceiling: f64,
    exact: bool,
    unit: f64,
}

/// Exact power-of-two change of dependent units. Base coordinates and their
/// literal source operations remain unchanged. The scaled program must pass
/// every existing integer-cache guard before a suffix can use it.
#[derive(Clone, Debug, PartialEq)]
pub(in crate::graph::coupled::kernel) struct PreparedCostLattice {
    unit: f64,
    ceiling: f64,
    arcs: Vec<PreparedRelationArc>,
}

impl PreparedCostLattice {
    pub(in crate::graph::coupled::kernel) fn new(
        initial: f64,
        arcs: &[PreparedRelationArc],
    ) -> Option<Self> {
        let first = arcs.first()?.arc();
        let lower = first.lower.points.first()?[1];
        let upper = first.upper.points.first()?[1];
        if !initial.is_finite() || initial < lower {
            return None;
        }
        let mut exponent = None;
        let mut include = |value: f64| -> Option<()> {
            if !value.is_finite() {
                return None;
            }
            let bits = value.to_bits() & 0x7fff_ffff_ffff_ffff;
            if bits == 0 {
                return Some(());
            }
            let encoded = ((bits >> 52) & 0x7ff) as i32;
            let significand =
                (bits & 0x000f_ffff_ffff_ffff) | if encoded == 0 { 0 } else { 1u64 << 52 };
            let power = (if encoded == 0 {
                -1074
            } else {
                encoded - 1023 - 52
            }) + significand.trailing_zeros() as i32;
            exponent = Some(exponent.map_or(power, |old: i32| old.min(power)));
            Some(())
        };
        include(initial)?;
        let mut waiting_charges = Vec::with_capacity(arcs.len());
        for prepared in arcs {
            let arc = prepared.arc();
            if arc
                .lower
                .points
                .iter()
                .any(|p| p[1].to_bits() != lower.to_bits())
                || arc
                    .upper
                    .points
                    .iter()
                    .any(|p| p[1].to_bits() != upper.to_bits())
                || arc
                    .increment
                    .points
                    .iter()
                    .any(|p| p[1].to_bits() != arc.increment.points[0][1].to_bits() || p[1] < 0.)
            {
                return None;
            }
            include(arc.increment.points[0][1])?;
            let waiting = prepared.fixed_waiting_charge()?;
            include(waiting)?;
            waiting_charges.push(waiting);
        }
        include(lower)?;
        include(upper)?;
        // An entirely zero dependent program needs no finer unit. It is an
        // exact identity recurrence, including through repeated zero arcs.
        let exponent = exponent.unwrap_or(0);
        // Keep all nonzero lattice values normal before and after scaling.
        // With at most 2^40 units, every accepted addition is exact and its
        // result remains well within the finite normal binary64 range.
        if !(-512..=512).contains(&exponent) {
            return None;
        }
        let unit = f64::from_bits(((exponent + 1023) as u64) << 52);
        let scale = |value: f64| -> Option<f64> {
            let value = value / unit;
            (value.is_finite() && value.fract() == 0. && value.abs() <= 1_099_511_627_776.)
                .then_some(value)
        };
        scale(initial)?;
        let ceiling = scale(upper)?;
        let function = |source: &PiecewiseLinearFunction| -> Option<PiecewiseLinearFunction> {
            PiecewiseLinearFunction::new(
                source
                    .points
                    .iter()
                    .map(|p| Some([p[0], scale(p[1])?]))
                    .collect::<Option<Vec<_>>>()?,
            )
            .ok()
        };
        let arcs = arcs
            .iter()
            .zip(waiting_charges)
            .map(|(prepared, waiting)| {
                let source = prepared.arc();
                // Both source additions are exact on this common lattice. Folding
                // the fixed waiting contribution therefore preserves their
                // ordered binary64 result without changing either base domain.
                let increment = scale(source.increment.points[0][1])? + scale(waiting)?;
                if increment < 0. || increment > 1_099_511_627_776. {
                    return None;
                }
                let arc = PreparedRelationArc::new(&RelationArc {
                    arrival: source.arrival.clone(),
                    increment: PiecewiseLinearFunction::new(
                        source
                            .increment
                            .points
                            .iter()
                            .map(|p| [p[0], increment])
                            .collect(),
                    )
                    .ok()?,
                    lower: function(&source.lower)?,
                    upper: function(&source.upper)?,
                    waiting: 0,
                })
                .ok()?;
                arc.exact_cache_eligible().then_some(arc)
            })
            .collect::<Option<Vec<_>>>()?;
        Some(Self {
            unit,
            ceiling,
            arcs,
        })
    }

    pub(in crate::graph::coupled::kernel) fn prepend(
        &self,
        previous: Option<&BoundedCostFront>,
        arc: usize,
        terminal: bool,
    ) -> Option<BoundedCostFront> {
        let arc = self.arcs.get(arc)?;
        if let Some(previous) = previous {
            if previous.unit.to_bits() != self.unit.to_bits() {
                return None;
            }
            previous.checked_prepend(arc)
        } else if terminal {
            let mut front = BoundedCostFront::from_terminal_arc(arc, self.ceiling)?;
            front.unit = self.unit;
            Some(front)
        } else {
            None
        }
    }
}
impl BoundedCostFront {
    /// Start the backward recurrence at the shared dependent ceiling, then
    /// prepend the terminal physical arc. Unknown arithmetic keeps replay.
    pub(in crate::graph::coupled::kernel) fn from_terminal_arc(
        arc: &PreparedRelationArc,
        ceiling: f64,
    ) -> Option<Self> {
        let source = arc.arc();
        let lower = source.lower.prepared_domain();
        let upper = source.upper.prepared_domain();
        Self::terminal(
            Window {
                lower: lower.lower.max(upper.lower),
                upper: lower.upper.min(upper.upper),
            },
            ceiling,
        )
        .ok()?
        .checked_prepend(arc)
    }

    /// Both suffixes have the same terminal ceiling. A larger admitted entry
    /// front preserves feasibility and decreases `ceiling + entry - upper`
    /// for every common prefix. Require equal numerical minima as an
    /// additional conservative condition; the outer dominance gate still
    /// requires equivalent ordered programs for cost-bearing suffixes.
    pub(in crate::graph::coupled::kernel) fn dominates(&self, other: &Self) -> bool {
        self.exact
            && other.exact
            && self.unit.to_bits() == other.unit.to_bits()
            && self.ceiling.to_bits() == other.ceiling.to_bits()
            && self
                .upper
                .pieces()
                .iter()
                .chain(other.upper.pieces())
                .all(|piece| piece.left[1].to_bits() == piece.right[1].to_bits())
            && matches!(
                (self.minimum_additional(), other.minimum_additional()),
                (Some(a), Some(b)) if a.to_bits() == b.to_bits()
            )
            && self.upper.covers_above_strict(&other.upper)
    }

    /// Caller proves all reachable dependent inputs are bounded exact integer
    /// multiples of the same power-of-two unit, waiting is zero or has been
    /// folded exactly from forced controls, increments are nonnegative constants
    /// on that lattice, and bounds are shared. Then
    /// the constant ceiling-minus-entry value is exactly the
    /// suffix's total increment. A smaller increment and covering base domain
    /// preserve feasibility and objective order for every reachable prefix.
    pub(in crate::graph::coupled::kernel) fn dominates_exact_lattice(&self, other: &Self) -> bool {
        let constant = |front: &Self| {
            let value = front.upper.pieces.first()?.left[1];
            front
                .upper
                .pieces
                .iter()
                .all(|piece| {
                    piece.left[1].to_bits() == value.to_bits()
                        && piece.right[1].to_bits() == value.to_bits()
                })
                .then_some(value)
        };
        self.exact
            && other.exact
            && self.unit.to_bits() == other.unit.to_bits()
            && self.ceiling.to_bits() == other.ceiling.to_bits()
            && constant(self).is_some()
            && constant(other).is_some()
            && matches!((self.minimum_additional(), other.minimum_additional()),
                (Some(a), Some(b)) if a <= b)
            && self.upper.covers_above_strict(&other.upper)
    }

    /// Minimum incremental suffix charge, before objective scaling. The
    /// backward state stores admitted entry values, so the numerical value is
    /// the terminal ceiling minus the largest admitted entry. It is not used
    /// as a key until its bound is proved for fractional prefixes.
    pub(in crate::graph::coupled::kernel) fn minimum_additional(&self) -> Option<f64> {
        if !self.exact {
            return None;
        }
        let maximum = self
            .upper
            .pieces()
            .iter()
            .flat_map(|piece| [piece.left[1], piece.right[1]])
            .max_by(f64::total_cmp)?;
        let lo = crate::core::rounding::add_down(self.ceiling, -maximum)?;
        let hi = crate::core::rounding::add_up(self.ceiling, -maximum)?;
        let value = lo * self.unit;
        (lo == hi && lo >= 0. && value.is_finite() && value / self.unit == lo).then_some(value)
    }

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
            unit: 1.,
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
                unit: self.unit,
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
        // Integer translation endpoints do not generally prove that the
        // literal divide-then-multiply evaluator is an exact translation at
        // interior values. The unit identity is exact for every binary64
        // input in its domain: subtraction by zero, division and multiply by
        // one, and addition of zero preserve its value (with endpoint lookup
        // handling zero). On sufficiently distant power-of-two intervals,
        // Sterbenz's lemma makes x - x0 exact. Division and multiplication by
        // the interval width are exact power-of-two scalings without
        // underflow under the integer/range guards above; adding x0 then
        // recovers the representable input. Other domains require replay.
        let source = arc.arc();
        let arrival = source.arrival.prepared_domain();
        let increment = source.increment.prepared_domain();
        let lo = arrival.lower.max(increment.lower);
        let hi = arrival.upper.min(increment.upper);
        let exact_identity = source.arrival.points.len() == 2
            && source
                .arrival
                .points
                .iter()
                .all(|p| p[0].to_bits() == p[1].to_bits())
            && {
                let start = source.arrival.points[0][0];
                let end = source.arrival.points[1][0];
                let width = end - start;
                let bits = width.to_bits();
                let normal_power_of_two = width.is_finite()
                    && width > 0.
                    && (bits & 0x000f_ffff_ffff_ffff) == 0
                    && (bits & 0x7ff0_0000_0000_0000) != 0;
                normal_power_of_two
                    && ((start.to_bits() == 0.0f64.to_bits() && width == 1.)
                        || (start.to_bits() == 0.0f64.to_bits() && lo >= 1.)
                        || start >= width
                        || end <= -width)
            };
        if lo != hi && !exact_identity {
            return None;
        }
        if lo == hi {
            let shift = source.arrival.points[0][1] - source.arrival.points[0][0];
            let translated = finite(lo + shift).ok()?;
            if evaluate(&source.arrival, lo).ok()?.to_bits() != translated.to_bits() {
                return None;
            }
        }
        // Constant upper fronts under the restricted translation/constant-phi
        // contract need only a forced-wait ramp followed by a constant. Avoid
        // negate/clip/reflect/composition scratch for this frequent case.
        if self.upper.pieces.len() == 1
            && self.upper.pieces[0].left[1] == self.upper.pieces[0].right[1]
        {
            return self.prepend_constant(arc).filter(|next| {
                next.upper
                    .pieces()
                    .iter()
                    .all(|piece| piece.left[1].to_bits() == piece.right[1].to_bits())
            });
        }
        let next = self.prepend(arc).ok().flatten()?;
        (next.upper.exact_cache_eligible()
            && next
                .upper
                .pieces()
                .iter()
                .all(|piece| piece.left[1].to_bits() == piece.right[1].to_bits()))
        .then_some(next)
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
            unit: self.unit,
        })
    }
    /// Outer None is unknown; Some(None) is an exact restricted-contract
    /// nonintersection, and Some(Some(value)) the exact compatible objective.
    pub(in crate::graph::coupled::kernel) fn checked_compatible_join(
        &self,
        forward: &RelationFront,
    ) -> Option<Option<f64>> {
        let scaled = if self.unit.to_bits() == 1.0f64.to_bits() {
            None
        } else {
            let mut pieces = Vec::with_capacity(forward.pieces.len());
            for piece in &forward.pieces {
                // Change only dependent units. Keep all original base-domain
                // endpoints and provenance; the existing geometry proof below
                // must still succeed on the resulting constant prefix.
                if piece.left[1].to_bits() != piece.right[1].to_bits() {
                    return None;
                }
                let value = piece.left[1] / self.unit;
                if !value.is_finite()
                    || value.fract() != 0.
                    || value.abs() > 1_099_511_627_776.
                    || value * self.unit != piece.left[1]
                {
                    return None;
                }
                let mut scaled = piece.clone();
                scaled.left[1] = value;
                scaled.right[1] = value;
                pieces.push(scaled);
            }
            Some(RelationFront::from_pieces(pieces))
        };
        let forward = scaled.as_ref().unwrap_or(forward);
        if !self.exact
            || !forward.exact_cache_eligible()
            // The cache's integer-slope evaluator reassociates the literal
            // source expression. Integral endpoints and slopes do not make
            // those two binary64 programs equivalent at interior points.
            // Constant pieces avoid that reassociation; all other joins use
            // the existing ordered suffix replay.
            || forward
                .pieces()
                .iter()
                .chain(self.upper.pieces())
                .any(|piece| piece.left[1].to_bits() != piece.right[1].to_bits())
            || self.ceiling.fract() != 0.
            || self.ceiling.abs() > 1_099_511_627_776.
        {
            return None;
        }
        match self.compatible_join_impl(forward, true).ok()? {
            None => Some(None),
            Some(value) => {
                let original = value * self.unit;
                (original.is_finite() && original / self.unit == value).then_some(Some(original))
            }
        }
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
