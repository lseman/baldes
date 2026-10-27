//! Function-valued base/dependent relations with explicit waiting choices.
//!
//! A front stores the least attainable dependent value at each base value.
//! Each closed affine piece owns its endpoints; gaps remain infeasible. Waiting
//! provenance is retained even when two neighboring affine pieces are equal.
//! This is a numerical frontier. It deliberately supplies no directed proof
//! certificate and no algebraic inverse of a waiting relation.
use super::*;
mod backward;
mod prepared;
pub(super) use backward::BoundedCostFront;
pub(super) use prepared::{PreparedRelationArc, RelationWorkspace};

// Explicit numerical work limit: exceeding this returns unsupported arithmetic
// rather than an empty/complete pricing result. No fixed native knot ABI limit.
const MAX_RELATION_PIECES: usize = 4096;
const MAX_ENVELOPE_PAIRS: usize = 1_000_000;

fn check_size(size: usize) -> Result<()> {
    if size > MAX_RELATION_PIECES {
        return Err(invalid("paired relation frontier work limit exceeded"));
    }
    Ok(())
}

/// Native cost-mode restrictions are separate from the broader feasibility
/// recurrence. Bounds must be constant and shared across all arcs; bounded
/// cost mode also requires nonnegative waiting, strictly positive increments,
/// and an initial dependent front no lower than that shared bound.
pub(in crate::graph::coupled) fn validate_cost_mode(
    arcs: &[RelationArc],
    initial_minimum: f64,
    bounded: bool,
) -> Result<()> {
    let Some(first) = arcs.first() else {
        return Ok(());
    };
    finite(initial_minimum)?;
    first.validate()?;
    let low = first.lower.points[0][1];
    let high = first.upper.points[0][1];
    for arc in arcs {
        arc.validate()?;
        if arc.lower.points.iter().any(|p| p[1] != low)
            || arc.upper.points.iter().any(|p| p[1] != high)
        {
            return Err(invalid(
                "paired cost bounds must be constant and shared across arcs",
            ));
        }
        if bounded && (arc.waiting < 0 || arc.increment.points.iter().any(|p| p[1] <= 0.)) {
            return Err(invalid(
                "bounded paired cost needs nonnegative waiting and positive increments",
            ));
        }
    }
    if bounded && initial_minimum < low {
        return Err(invalid(
            "bounded paired initial cost violates the shared lower bound",
        ));
    }
    Ok(())
}

#[repr(C)]
#[derive(Clone, Debug)]
pub(super) struct RelationPiece {
    pub(super) left: [f64; 2],
    pub(super) right: [f64; 2],
    pub(super) waiting: bool,
    // Explicit safe storage occupying formerly unused tail padding. It is
    // derived frontier metadata, never part of the resource relation.
    metadata: u32,
}
impl PartialEq for RelationPiece {
    fn eq(&self, other: &Self) -> bool {
        self.left == other.left && self.right == other.right && self.waiting == other.waiting
    }
}
impl RelationPiece {
    fn value(&self, x: f64) -> Result<f64> {
        if x == self.left[0] {
            return Ok(self.left[1]);
        }
        if x == self.right[0] {
            return Ok(self.right[1]);
        }
        let t = finite((x - self.left[0]) / (self.right[0] - self.left[0]))?;
        finite(self.left[1] + finite(t * finite(self.right[1] - self.left[1])?)?)
    }
    fn clipped(&self, lower: f64, upper: f64) -> Result<Option<Self>> {
        let lower = lower.max(self.left[0]);
        let upper = upper.min(self.right[0]);
        if lower > upper {
            return Ok(None);
        }
        Ok(Some(Self {
            metadata: 0,
            left: [lower, self.value(lower)?],
            right: [upper, self.value(upper)?],
            waiting: self.waiting,
        }))
    }
}

fn strict_piece_value_range(
    piece: &RelationPiece,
    lower: f64,
    upper: f64,
) -> Option<super::bounds::Range> {
    use super::bounds::Range;

    if lower > upper || lower < piece.left[0] || upper > piece.right[0] {
        return None;
    }
    if lower == upper {
        return Some(Range::point(piece.value(lower).ok()?));
    }
    let dx = finite(piece.right[0] - piece.left[0]).ok()?;
    let dy = finite(piece.right[1] - piece.left[1]).ok()?;
    if dx <= 0. {
        return None;
    }
    let t = Range {
        lo: lower,
        hi: upper,
    }
    .subtract(Range::point(piece.left[0]))
    .divide_positive(dx);
    let mut values = Range::point(piece.left[1]).add(t.multiply(Range::point(dy)));
    for x in [lower, upper] {
        let value = piece.value(x).ok()?;
        values.lo = values.lo.min(value);
        values.hi = values.hi.max(value);
    }
    values.finite().then_some(values)
}

fn strict_piece_below(
    left: &RelationPiece,
    right: &RelationPiece,
    lower: f64,
    upper: f64,
    work: &mut usize,
    depth: usize,
) -> bool {
    if left == right {
        return true;
    }
    if *work == 0 {
        return false;
    }
    *work -= 1;
    let (Some(left_values), Some(right_values)) = (
        strict_piece_value_range(left, lower, upper),
        strict_piece_value_range(right, lower, upper),
    ) else {
        return false;
    };
    if left_values.subtract(right_values).hi <= 0. {
        return true;
    }
    if depth >= 48 || lower == upper {
        return false;
    }
    let middle = lower * 0.5 + upper * 0.5;
    if !middle.is_finite() || middle <= lower || middle >= upper {
        return false;
    }
    strict_piece_below(left, right, lower, middle, work, depth + 1)
        && strict_piece_below(left, right, middle, upper, work, depth + 1)
}

#[derive(Clone, Debug)]
pub(in crate::graph::coupled) struct RelationFront {
    // Immutable once published. Cached metadata lives in explicit fields in
    // the first pieces, so the frontier header remains one Vec.
    pieces: Vec<RelationPiece>,
}
impl PartialEq for RelationFront {
    fn eq(&self, other: &Self) -> bool {
        self.pieces.len()==other.pieces.len()
            && !matches!((self.rejection_fingerprint(),other.rejection_fingerprint()),(Some(a),Some(b)) if a!=b)
            // Digest matches, missing metadata and collisions all retain the
            // full original floating-value/provenance comparison.
            && self.pieces==other.pieces
    }
}

/// Prepared local contract. `arrival(t)` is the total base arrival value,
/// `increment(t)` the dependent increment at departure, and lower/upper are
/// dependent bounds at the head base value. Waiting has an explicit integer
/// slope; no unchecked conversion from a public floating descriptor occurs.
pub(super) type RelationArc = PairedResourceArc;
impl RelationArc {
    pub(in crate::graph::coupled) fn validate(&self) -> Result<()> {
        for f in [&self.arrival, &self.increment, &self.lower, &self.upper] {
            f.validate()?;
        }
        if self.arrival.points.windows(2).any(|p| p[0][1] >= p[1][1])
            || self.arrival.points.iter().any(|p| p[1] < p[0])
        {
            return Err(invalid(
                "paired base arrival must be strictly FIFO and nonnegative",
            ));
        }
        Ok(())
    }
}

impl RelationFront {
    const META_VALID: u32 = 1 << 30;
    const META_EXACT: u32 = 1 << 31;
    const META_INDEX: u32 = Self::META_VALID - 1;

    fn scan_exact(pieces: &[RelationPiece]) -> bool {
        let integer = |x: f64| x.is_finite() && x.fract() == 0. && x.abs() <= 1_099_511_627_776.;
        pieces.iter().all(|p| {
            let dx = p.right[0] - p.left[0];
            let dy = p.right[1] - p.left[1];
            p.left.iter().chain(&p.right).all(|&x| integer(x))
                && (dx == 0.
                    || (dx > 0. && integer(dx) && integer(dy) && dy % dx == 0. && integer(dy / dx)))
        })
    }
    fn from_pieces(mut pieces: Vec<RelationPiece>) -> Self {
        let integer = |x: f64| x.is_finite() && x.fract() == 0. && x.abs() <= 1_099_511_627_776.;
        let mut minimum = f64::INFINITY;
        let mut minimum_index = None;
        let mut exact = true;
        let mut all_finite = true;
        // Local performance policy, not a native constant. Common short
        // fronts avoid hashing; wide fronts cache a rejection-only digest.
        let wide = pieces.len() >= 8;
        let mut fingerprint = if wide {
            0xcbf29ce484222325u64 ^ (pieces.len() as u64)
        } else {
            0
        };
        for (i, p) in pieces.iter_mut().enumerate() {
            // Intermediates may clone original pieces. Clear every old marker
            // before publishing this newly owned frontier.
            p.metadata = 0;
            for (side, &value) in [p.left[1], p.right[1]].iter().enumerate() {
                let next = minimum.min(value);
                if next.to_bits() != minimum.to_bits()
                    || (minimum_index.is_none() && next.to_bits() == value.to_bits())
                {
                    minimum_index = i.checked_mul(2).and_then(|x| x.checked_add(side));
                }
                minimum = next;
            }
            all_finite &= p.left.iter().chain(&p.right).all(|x| x.is_finite());
            if wide {
                for &x in p.left.iter().chain(&p.right) {
                    let bits = if x == 0. { 0 } else { x.to_bits() };
                    fingerprint = (fingerprint ^ bits).wrapping_mul(0x100000001b3);
                }
                fingerprint = (fingerprint ^ u64::from(p.waiting)).wrapping_mul(0x100000001b3);
            }
            if exact {
                let dx = p.right[0] - p.left[0];
                let dy = p.right[1] - p.left[1];
                exact = p.left.iter().chain(&p.right).all(|&x| integer(x))
                    && (dx == 0.
                        || (dx > 0.
                            && integer(dx)
                            && integer(dy)
                            && dy % dx == 0.
                            && integer(dy / dx)));
            }
        }
        if let Some(index) = minimum_index.filter(|&x| all_finite && x <= Self::META_INDEX as usize)
        {
            pieces[0].metadata =
                Self::META_VALID | if exact { Self::META_EXACT } else { 0 } | index as u32;
            if wide {
                pieces[1].metadata = fingerprint as u32;
                pieces[2].metadata = (fingerprint >> 32) as u32;
            }
        }
        Self { pieces }
    }
    fn metadata(&self) -> Option<u32> {
        let metadata = self.pieces.first()?.metadata;
        if metadata & Self::META_VALID == 0 {
            return None;
        }
        self.pieces
            .get((metadata & Self::META_INDEX) as usize / 2)?;
        Some(metadata)
    }
    fn rejection_fingerprint(&self) -> Option<u64> {
        if self.pieces.len() < 8 || self.metadata().is_none() {
            return None;
        }
        Some(u64::from(self.pieces[1].metadata) | (u64::from(self.pieces[2].metadata) << 32))
    }

    pub(in crate::graph::coupled) fn initial(
        function: &PiecewiseLinearFunction,
        domain: Window,
    ) -> Result<Option<Self>> {
        function.validate()?;
        finite(domain.lower)?;
        finite(domain.upper)?;
        if domain.lower > domain.upper {
            return Err(invalid("invalid paired initial interval"));
        }
        let mut pieces = Vec::new();
        for p in function.points.windows(2) {
            let piece = RelationPiece {
                metadata: 0,
                left: p[0],
                right: p[1],
                waiting: false,
            };
            if let Some(p) = piece.clipped(domain.lower, domain.upper)? {
                pieces.push(p);
            }
        }
        Ok((!pieces.is_empty()).then_some(Self::from_pieces(pieces)))
    }
    #[inline]
    pub(super) fn exact_cache_eligible(&self) -> bool {
        self.metadata().map_or_else(
            || Self::scan_exact(&self.pieces),
            |m| m & Self::META_EXACT != 0,
        )
    }
    pub(super) fn pieces(&self) -> &[RelationPiece] {
        &self.pieces
    }
    #[inline]
    pub(in crate::graph::coupled) fn minimum(&self) -> f64 {
        if let Some(metadata) = self.metadata() {
            let index = (metadata & Self::META_INDEX) as usize;
            if let Some(piece) = self.pieces.get(index / 2) {
                return if index % 2 == 0 {
                    piece.left[1]
                } else {
                    piece.right[1]
                };
            }
        }
        self.pieces.iter().fold(f64::INFINITY, |minimum, p| {
            minimum.min(p.left[1]).min(p.right[1])
        })
    }
    /// Negated maximum dependent bound on this front's feasible base domain.
    /// Once the lower recurrence is feasible, any dependent value up to
    /// `upper(x)` is allowed; negating and replaying the lower recurrence
    /// would therefore compute the wrong maximum envelope.
    pub(super) fn negated_upper_on(&self, upper: &PiecewiseLinearFunction) -> Result<Option<Self>> {
        let mut pieces = Vec::new();
        for feasible in &self.pieces {
            let domain = (feasible.left[0], feasible.right[0]);
            if upper.points.len() == 1 {
                let x = upper.points[0][0];
                if domain.0 <= x && x <= domain.1 {
                    let value = finite(-upper.points[0][1])?;
                    pieces.push(RelationPiece {
                        metadata: 0,
                        left: [x, value],
                        right: [x, value],
                        waiting: false,
                    });
                }
                continue;
            }
            for segment in upper.points.windows(2) {
                let lo = domain.0.max(segment[0][0]);
                let hi = domain.1.min(segment[1][0]);
                if lo > hi {
                    continue;
                }
                pieces.push(RelationPiece {
                    metadata: 0,
                    left: [lo, finite(-evaluate(upper, lo)?)?],
                    right: [hi, finite(-evaluate(upper, hi)?)?],
                    waiting: false,
                });
            }
        }
        Ok((!pieces.is_empty()).then(|| Self::from_pieces(pieces)))
    }
    /// This forward front is defined wherever `other` is and is no higher
    /// there, comparing values as the reference does: within 1e-4, or a
    /// relative 2.22e-10, counts as equal. Forward extension is monotone in
    /// the front, so every completion of `other` stays available.
    #[cfg(test)]
    pub(in crate::graph::coupled) fn covers_below(&self, other: &Self) -> bool {
        self.excess(other).is_some_and(|excess| excess <= 0.)
    }
    /// Directed, no-tolerance continuation coverage for signed dominance.
    /// Interval subdivision may fail to prove close fronts; failure only
    /// retains the label. The bounded work cap keeps this proof off the
    /// unbounded path when frontiers are large or nearly coincident.
    pub(in crate::graph::coupled) fn covers_below_strict(&self, other: &Self) -> bool {
        const WORK_LIMIT: usize = 200_000;
        let mut work = WORK_LIMIT;
        for other_piece in &other.pieces {
            let mut knots = vec![other_piece.left[0], other_piece.right[0]];
            for piece in &self.pieces {
                if work == 0 {
                    return false;
                }
                work -= 1;
                for x in [piece.left[0], piece.right[0]] {
                    if other_piece.left[0] < x && x < other_piece.right[0] {
                        knots.push(x);
                    }
                }
            }
            knots.sort_by(f64::total_cmp);
            knots.dedup();
            if knots.len() == 1 {
                if !self.covers_piece_interval(other_piece, knots[0], knots[0], &mut work, 0) {
                    return false;
                }
                continue;
            }
            for span in knots.windows(2) {
                if !self.covers_piece_interval(other_piece, span[0], span[1], &mut work, 0) {
                    return false;
                }
            }
        }
        true
    }

    fn covers_piece_interval(
        &self,
        other: &RelationPiece,
        lower: f64,
        upper: f64,
        work: &mut usize,
        depth: usize,
    ) -> bool {
        if *work == 0 {
            return false;
        }
        *work -= 1;
        for piece in &self.pieces {
            if *work == 0 {
                return false;
            }
            *work -= 1;
            if piece.left[0] <= lower
                && upper <= piece.right[0]
                && strict_piece_below(piece, other, lower, upper, work, 0)
            {
                return true;
            }
        }
        if depth >= 48 || lower == upper {
            return false;
        }
        let middle = lower * 0.5 + upper * 0.5;
        if !middle.is_finite() || middle <= lower || middle >= upper {
            return false;
        }
        self.covers_piece_interval(other, lower, middle, work, depth + 1)
            && self.covers_piece_interval(other, middle, upper, work, depth + 1)
    }
    /// This backward upper front is defined wherever `other` is and is no
    /// lower there: it admits every prefix `other` admits.
    pub(in crate::graph::coupled) fn covers_above(&self, other: &Self) -> bool {
        self.signed_excess(other, -1.)
            .is_some_and(|excess| excess <= 0.)
    }
    /// The backward root: any dependent value up to the reference's 1e12
    /// infinity over the head domain; the adjoint applies the arc bounds.
    pub(in crate::graph::coupled) fn unbounded_upper(lower: f64, upper: f64) -> Self {
        Self::from_pieces(vec![RelationPiece {
            metadata: 0,
            left: [lower, 1e12],
            right: [upper, 1e12],
            waiting: false,
        }])
    }
    /// None unless this front is defined wherever `other` is; otherwise an
    /// upper bound on max(this - other) over `other`'s domain, with
    /// tolerance-equal values counting as 0. Fronts are piecewise linear, so
    /// each of `other`'s pieces is checked at the merged breakpoints; one
    /// piece of this front must cover each sub-interval by itself (its
    /// values bound the front's minimum there). Overlapping pieces are never
    /// combined, which can only overestimate.
    pub(in crate::graph::coupled) fn excess(&self, other: &Self) -> Option<f64> {
        self.signed_excess(other, 1.)
    }
    /// `excess` of sign * (this - other).
    fn signed_excess(&self, other: &Self, sign: f64) -> Option<f64> {
        let near = |a: f64, b: f64| {
            let d = (a - b).abs();
            d < 0.0001 || d < (a + b).abs() * 2.220446049250313e-10
        };
        let mut knots = Vec::new();
        let mut excess = f64::NEG_INFINITY;
        for b in &other.pieces {
            knots.clear();
            knots.push(b.left[0]);
            knots.extend(
                self.pieces
                    .iter()
                    .flat_map(|a| [a.left[0], a.right[0]])
                    .filter(|&x| b.left[0] < x && x < b.right[0]),
            );
            knots.push(b.right[0]);
            knots.sort_by(f64::total_cmp);
            knots.dedup();
            let gap = |a: &RelationPiece, x: f64| {
                let (va, vb) = (a.value(x).ok()?, b.value(x).ok()?);
                let d = finite(sign * (va - vb)).ok()?;
                Some(if near(va, vb) { d.min(0.) } else { d })
            };
            // The smallest bound any single covering piece gives on [lo, hi].
            let cover = |lo: f64, hi: f64| {
                self.pieces
                    .iter()
                    .filter(|a| a.left[0] <= lo && hi <= a.right[0])
                    .filter_map(|a| Some(gap(a, lo)?.max(gap(a, hi)?)))
                    .min_by(f64::total_cmp)
            };
            let spans: Vec<(f64, f64)> = if knots.len() == 1 {
                vec![(knots[0], knots[0])]
            } else {
                knots.windows(2).map(|w| (w[0], w[1])).collect()
            };
            for (lo, hi) in spans {
                excess = excess.max(cover(lo, hi)?);
            }
        }
        excess.is_finite().then_some(excess)
    }
    #[cfg(test)]
    pub(super) fn value(&self, x: f64) -> Result<Option<f64>> {
        let mut best: Option<f64> = None;
        for p in &self.pieces {
            if p.left[0] <= x && x <= p.right[0] {
                let v = p.value(x)?;
                best = Some(best.map_or(v, |old| old.min(v)));
            }
        }
        Ok(best)
    }

    /// Exact real-arithmetic recurrence, evaluated with checked binary64:
    /// H(q(t)) = F(t)+phi(t),
    /// W(x) = min_{q(t)<=x} [H(q(t))+kappa*(x-q(t))],
    /// F_next(x) = max(W(x),lower(x)), provided F_next(x)<=upper(x).
    /// The lower bound permits increasing the dependent resource; it is not
    /// an additional objective charge. Disconnected upper-bound intersections
    /// remain separate pieces rather than being replaced by a feasible hull.
    #[cfg(test)]
    pub(super) fn extend(&self, arc: &RelationArc) -> Result<Option<Self>> {
        self.extend_workspace(arc, &mut RelationWorkspace::default(), None)
    }
    pub(super) fn extend_prepared(
        &self,
        arc: &PreparedRelationArc,
        workspace: &mut RelationWorkspace,
    ) -> Result<Option<Self>> {
        self.extend_workspace(arc.arc(), workspace, Some(&arc.departure_knots))
    }
    fn extend_workspace(
        &self,
        arc: &RelationArc,
        workspace: &mut RelationWorkspace,
        prepared_knots: Option<&[f64]>,
    ) -> Result<Option<Self>> {
        let ad = arc.arrival.prepared_domain();
        let id = arc.increment.prepared_domain();
        let ld = arc.lower.prepared_domain();
        let ud = arc.upper.prepared_domain();
        let lower = ld.lower.max(ud.lower);
        let upper = ld.upper.min(ud.upper);
        if lower > upper {
            return Ok(None);
        }
        check_size(self.pieces.len())?;
        let transported = &mut workspace.transported;
        transported.clear();
        for p in &self.pieces {
            let lo = p.left[0].max(ad.lower).max(id.lower);
            let hi = p.right[0].min(ad.upper).min(id.upper);
            if lo > hi {
                continue;
            }
            let knots = &mut workspace.knots;
            knots.clear();
            if let Some(prepared) = prepared_knots {
                knots.push(lo);
                if hi > lo {
                    let start = prepared.partition_point(|&x| x <= lo);
                    let end = prepared.partition_point(|&x| x < hi);
                    knots.extend_from_slice(&prepared[start..end]);
                    knots.push(hi);
                }
            } else {
                knots.extend([lo, hi]);
                for point in arc.arrival.points.iter().chain(&arc.increment.points) {
                    if lo < point[0] && point[0] < hi {
                        knots.push(point[0]);
                    }
                }
                ordered(knots);
            }
            let points = &mut workspace.points;
            points.clear();
            for &t in knots.iter() {
                let x = evaluate(&arc.arrival, t)?;
                let y = finite(p.value(t)? + evaluate(&arc.increment, t)?)?;
                if points
                    .last()
                    .is_some_and(|previous: &[f64; 2]| previous[0] >= x)
                {
                    return Err(invalid("paired arrival merges distinct front coordinates"));
                }
                points.push([x, y]);
            }
            // A tiny arrival span at a huge offset can collapse many distinct
            // departure choices inside a segment although endpoint knots are
            // distinct. A single-valued affine transport then loses literal
            // feasible witnesses. Reject this unresolved numerical relation.
            for pair in points.windows(2) {
                let span = finite(pair[1][0] - pair[0][0])?;
                let magnitude = pair[0][0].abs().max(pair[1][0].abs());
                if pair[0][1] != pair[1][1] && span <= 64. * f64::EPSILON * magnitude {
                    return Err(invalid(
                        "paired arrival interior rounding plateau cannot be represented safely",
                    ));
                }
            }
            add_points(transported, points, p.waiting);
            check_size(transported.len())?;
        }
        if transported.is_empty() {
            return Ok(None);
        }
        let envelope = waiting_envelope(transported, lower, upper, f64::from(arc.waiting))?;
        let bounded = apply_bounds(&envelope, &arc.lower, &arc.upper)?;
        check_size(bounded.len())?;
        Ok((!bounded.is_empty()).then_some(Self::from_pieces(bounded)))
    }
    /// Feasibility adjoint. Here this front is an upper admissible dependent
    /// value B(x), rather than a forward lower front. The predecessor is
    /// sup_{x>=q(t), lower(x)<=C(x)} [C(x)-phi(t)-kappa*(x-q(t))],
    /// where C(x)=min(B(x),upper(x)). Cost joins still need the full ordered
    /// suffix transfer because function-valued lower clamps are not additive.
    // Retained as a separately verified feasibility operation. Production
    // backward costs use suffix replay until a certified pruning policy exists.
    #[allow(dead_code)]
    pub(super) fn extend_backward_upper(&self, arc: &RelationArc) -> Result<Option<Self>> {
        let inverse = PiecewiseLinearFunction::new(
            arc.arrival.points.iter().map(|p| [p[1], p[0]]).collect(),
        )?;
        self.extend_backward_upper_impl(arc, &inverse)
    }
    pub(super) fn extend_backward_upper_prepared(
        &self,
        arc: &PreparedRelationArc,
    ) -> Result<Option<Self>> {
        self.extend_backward_upper_impl(arc.arc(), &arc.inverse)
    }
    fn extend_backward_upper_impl(
        &self,
        arc: &RelationArc,
        inverse: &PiecewiseLinearFunction,
    ) -> Result<Option<Self>> {
        check_size(self.pieces.len())?;
        let negate = |f: &PiecewiseLinearFunction| PiecewiseLinearFunction {
            points: f.points.iter().map(|p| [p[0], -p[1]]).collect(),
        };
        let negated: Vec<_> = self
            .pieces
            .iter()
            .map(|p| RelationPiece {
                metadata: 0,
                left: [p.left[0], -p.left[1]],
                right: [p.right[0], -p.right[1]],
                waiting: p.waiting,
            })
            .collect();
        let head = arc.lower.prepared_domain();
        let upper_head = arc.upper.prepared_domain();
        let mut trimmed = Vec::new();
        for piece in negated {
            if let Some(p) = piece.clipped(
                head.lower.max(upper_head.lower),
                head.upper.min(upper_head.upper),
            )? {
                trimmed.push(p);
            }
        }
        let clipped = apply_bounds(&trimmed, &negate(&arc.upper), &negate(&arc.lower))?;
        let reflected: Vec<_> = clipped
            .iter()
            .map(|p| RelationPiece {
                metadata: 0,
                left: [-p.right[0], p.right[1]],
                right: [-p.left[0], p.left[1]],
                waiting: p.waiting,
            })
            .collect();
        let domain = arc.arrival.prepared_domain();
        let consumption = arc.increment.prepared_domain();
        let lo = domain.lower.max(consumption.lower);
        let hi = domain.upper.min(consumption.upper);
        if lo > hi {
            return Ok(None);
        }
        let arrival_lo = evaluate(&arc.arrival, lo)?;
        let arrival_hi = evaluate(&arc.arrival, hi)?;
        let envelope =
            waiting_envelope(&reflected, -arrival_hi, -arrival_lo, f64::from(arc.waiting))?;
        let mut output = Vec::new();
        for p in envelope {
            let low = evaluate(inverse, -p.right[0])?.max(lo);
            let high = evaluate(inverse, -p.left[0])?.min(hi);
            if low > high {
                continue;
            }
            let mut knots = vec![low, high];
            for knot in arc.arrival.points.iter().chain(&arc.increment.points) {
                if low < knot[0] && knot[0] < high {
                    knots.push(knot[0]);
                }
            }
            ordered(&mut knots);
            let mut points = Vec::with_capacity(knots.len());
            for t in knots {
                let arrival = -evaluate(&arc.arrival, t)?;
                // Inverse/forward interpolation may disagree at an endpoint
                // by an ulp; clamp only within this already intersected piece.
                let arrival = arrival.clamp(p.left[0], p.right[0]);
                points.push([
                    t,
                    finite(-p.value(arrival)? - evaluate(&arc.increment, t)?)?,
                ]);
            }
            add_points(&mut output, &points, p.waiting);
            check_size(output.len())?;
        }
        output.sort_by(|a, b| {
            a.left[0]
                .total_cmp(&b.left[0])
                .then(a.right[0].total_cmp(&b.right[0]))
        });
        Ok((!output.is_empty()).then_some(Self::from_pieces(output)))
    }
}

fn waiting_envelope(
    transported: &[RelationPiece],
    lower: f64,
    upper: f64,
    k: f64,
) -> Result<Vec<RelationPiece>> {
    if lower > upper || transported.is_empty() {
        return Ok(Vec::new());
    }
    // Transport preserves coordinate order. The reflected backward adjoint
    // reverses it; normalize that one order once, rather than intersecting
    // every pair of waiting rays. Rays all have the same slope k.
    let mut reordered;
    let ordered_pieces = if transported.windows(2).any(|p| p[0].left[0] > p[1].left[0]) {
        reordered = transported.to_vec();
        reordered.sort_by(|a, b| {
            a.left[0]
                .total_cmp(&b.left[0])
                .then(a.right[0].total_cmp(&b.right[0]))
        });
        reordered.as_slice()
    } else {
        transported
    };
    let mut result = Vec::new();
    let mut best: Option<[f64; 2]> = None;
    let mut cursor = ordered_pieces[0].left[0];
    let ray_value =
        |origin: [f64; 2], x: f64| finite(origin[1] + finite(k * finite(x - origin[0])?)?);
    let mut emit = |piece: RelationPiece| -> Result<()> {
        if let Some(piece) = piece.clipped(lower, upper)? {
            result.push(piece);
            check_size(result.len())?;
        }
        Ok(())
    };
    let mut pair_knots = Vec::with_capacity(6);
    let mut pair_output = Vec::with_capacity(6);
    for p in ordered_pieces {
        if p.left[0] > upper {
            break;
        }
        if p.left[0] < cursor && p.right[0] > cursor {
            return Err(invalid(
                "paired waiting input has overlapping interior pieces",
            ));
        }
        if let Some(origin) = best {
            if cursor < p.left[0] {
                emit(RelationPiece {
                    metadata: 0,
                    left: [cursor, ray_value(origin, cursor)?],
                    right: [p.left[0], ray_value(origin, p.left[0])?],
                    waiting: true,
                })?;
            }
        }
        let origin = match best {
            Some(origin) if ray_value(origin, p.left[0])? <= p.left[1] => origin,
            _ => p.left,
        };
        let ray = RelationPiece {
            metadata: 0,
            left: [p.left[0], ray_value(origin, p.left[0])?],
            right: [p.right[0], ray_value(origin, p.right[0])?],
            waiting: true,
        };
        // Only one direct piece and one parallel waiting ray compete here.
        // The existing guarded crossing operator handles their sole switch,
        // including literal point minima and adjacent-float cells.
        lower_envelope_into(&[p.clone(), ray], &mut pair_knots, &mut pair_output)?;
        for piece in pair_output.drain(..) {
            emit(piece)?;
        }
        best = Some(if p.right[1] < ray_value(origin, p.right[0])? {
            p.right
        } else {
            origin
        });
        cursor = cursor.max(p.right[0]);
    }
    if let Some(origin) = best {
        if cursor < upper {
            emit(RelationPiece {
                metadata: 0,
                left: [cursor, ray_value(origin, cursor)?],
                right: [upper, ray_value(origin, upper)?],
                waiting: true,
            })?;
        }
    }
    Ok(result)
}

fn evaluate(f: &PiecewiseLinearFunction, x: f64) -> Result<f64> {
    f.evaluate(x)?
        .ok_or_else(|| invalid("paired function domain mismatch"))
}
fn ordered(knots: &mut Vec<f64>) {
    knots.sort_by(f64::total_cmp);
    knots.dedup();
}
fn add_points(output: &mut Vec<RelationPiece>, points: &[[f64; 2]], waiting: bool) {
    if points.len() == 1 {
        output.push(RelationPiece {
            metadata: 0,
            left: points[0],
            right: points[0],
            waiting,
        });
    } else {
        output.extend(points.windows(2).map(|p| RelationPiece {
            metadata: 0,
            left: p[0],
            right: p[1],
            waiting,
        }));
    }
}
fn crossing(a: &RelationPiece, b: &RelationPiece, lo: f64, hi: f64) -> Result<Option<f64>> {
    if lo >= hi {
        return Ok(None);
    }
    let dl = finite(a.value(lo)? - b.value(lo)?)?;
    let dr = finite(a.value(hi)? - b.value(hi)?)?;
    if (dl < 0. && dr > 0.) || (dl > 0. && dr < 0.) {
        let ratio = finite(dl / finite(dl - dr)?)?;
        let x = finite(lo + finite(ratio * finite(hi - lo)?)?)?;
        if x <= lo || x >= hi {
            return Err(invalid(
                "paired front crossing collapses at binary64 precision",
            ));
        }
        Ok(Some(x))
    } else {
        Ok(None)
    }
}
#[cfg(test)]
fn lower_envelope(candidates: &[RelationPiece]) -> Result<Vec<RelationPiece>> {
    let mut knots = Vec::new();
    let mut pieces = Vec::new();
    lower_envelope_into(candidates, &mut knots, &mut pieces)?;
    Ok(pieces)
}
fn lower_envelope_into(
    candidates: &[RelationPiece],
    knots: &mut Vec<f64>,
    pieces: &mut Vec<RelationPiece>,
) -> Result<()> {
    if candidates.len().saturating_mul(candidates.len()) / 2 > MAX_ENVELOPE_PAIRS {
        return Err(invalid("paired relation envelope work limit exceeded"));
    }
    knots.clear();
    pieces.clear();
    for (i, a) in candidates.iter().enumerate() {
        knots.extend([a.left[0], a.right[0]]);
        for b in &candidates[..i] {
            let lo = a.left[0].max(b.left[0]);
            let hi = a.right[0].min(b.right[0]);
            if let Some(x) = crossing(a, b, lo, hi)? {
                knots.push(x);
                check_size(knots.len())?;
            }
        }
    }
    ordered(knots);
    for pair in knots.windows(2) {
        let lo = pair[0];
        let hi = pair[1];
        // A manufactured crossing can round to one side of the actual
        // intersection. Its endpoint comparison must not select the branch
        // for an entire interval: a one-ulp residual could otherwise reverse
        // a material stretch of the envelope. Use an interior witness.
        let midpoint = finite(lo + finite((hi - lo) * 0.5)?)?;
        let interior = lo < midpoint && midpoint < hi;
        let mut best: Option<(&RelationPiece, f64, f64)> = None;
        let mut best_witness = f64::INFINITY;
        for p in candidates {
            if p.left[0] <= lo && hi <= p.right[0] {
                let left = p.value(lo)?;
                let right = p.value(hi)?;
                let witness = if interior { p.value(midpoint)? } else { left };
                if witness < best_witness
                    || (witness == best_witness && best.is_some_and(|(_, _, r)| right < r))
                {
                    best_witness = witness;
                    best = Some((p, left, right));
                }
            }
        }
        if interior {
            if let Some((winner, _, _)) = best {
                // A rounded crossing may be several representable values
                // away from the switch. Do not silently pick one branch if
                // its order still reverses inside this cell. Such arithmetic
                // remains explicitly unsupported, rather than infeasible.
                let first = next_toward(lo, hi);
                let last = next_toward(hi, lo);
                for competitor in candidates {
                    if competitor.left[0] <= lo
                        && hi <= competitor.right[0]
                        && (winner.value(first)? > competitor.value(first)?
                            || winner.value(last)? > competitor.value(last)?)
                    {
                        return Err(invalid(
                            "paired envelope crossing has an unresolved interior branch switch",
                        ));
                    }
                }
            }
        }
        if let Some((p, left, right)) = best {
            pieces.push(RelationPiece {
                metadata: 0,
                left: [lo, left],
                right: [hi, right],
                waiting: p.waiting,
            });
        }
    }
    // Preserve isolated feasible coordinates and lower values at branch
    // endpoints; they need not agree with the adjacent open-interval branch.
    for x in knots.iter().copied() {
        let mut best: Option<(&RelationPiece, f64)> = None;
        for p in candidates {
            if p.left[0] <= x && x <= p.right[0] {
                let value = p.value(x)?;
                if best.map_or(true, |(_, old)| value < old) {
                    best = Some((p, value));
                }
            }
        }
        if let Some((p, value)) = best {
            if !pieces
                .iter()
                .any(|q| (q.left == [x, value]) || (q.right == [x, value]))
            {
                pieces.push(RelationPiece {
                    metadata: 0,
                    left: [x, value],
                    right: [x, value],
                    waiting: p.waiting,
                });
            }
        }
    }
    pieces.sort_by(|a, b| {
        a.left[0]
            .total_cmp(&b.left[0])
            .then(a.right[0].total_cmp(&b.right[0]))
    });
    Ok(())
}
fn apply_bounds(
    front: &[RelationPiece],
    lower: &PiecewiseLinearFunction,
    upper: &PiecewiseLinearFunction,
) -> Result<Vec<RelationPiece>> {
    let mut result = Vec::new();
    for p in front {
        let mut knots = vec![p.left[0], p.right[0]];
        for q in lower.points.iter().chain(&upper.points) {
            if p.left[0] < q[0] && q[0] < p.right[0] {
                knots.push(q[0]);
            }
        }
        ordered(&mut knots);
        if knots.len() == 1 {
            let x = knots[0];
            let y = p.value(x)?.max(evaluate(lower, x)?);
            if y <= evaluate(upper, x)? {
                result.push(RelationPiece {
                    metadata: 0,
                    left: [x, y],
                    right: [x, y],
                    waiting: p.waiting,
                });
            }
            continue;
        }
        for pair in knots.windows(2) {
            let lo = pair[0];
            let hi = pair[1];
            let l = RelationPiece {
                metadata: 0,
                left: [lo, evaluate(lower, lo)?],
                right: [hi, evaluate(lower, hi)?],
                waiting: false,
            };
            let u = RelationPiece {
                metadata: 0,
                left: [lo, evaluate(upper, lo)?],
                right: [hi, evaluate(upper, hi)?],
                waiting: false,
            };
            let mut split = vec![lo, hi];
            if let Some(x) = crossing(p, &l, lo, hi)? {
                split.push(x);
            }
            if let Some(x) = crossing(p, &u, lo, hi)? {
                split.push(x);
            }
            if let Some(x) = crossing(&l, &u, lo, hi)? {
                split.push(x);
            }
            ordered(&mut split);
            for endpoints in split.windows(2) {
                let a = endpoints[0];
                let b = endpoints[1];
                let av = p.value(a)?.max(l.value(a)?);
                let bv = p.value(b)?.max(l.value(b)?);
                let au = u.value(a)?;
                let bu = u.value(b)?;
                // A rounded intersection can land one ulp on the rejected
                // side. Preserve the adjacent feasible interval instead of
                // dropping it because only that manufactured endpoint fails.
                let midpoint = finite(a + finite((b - a) * 0.5)?)?;
                let middle_feasible =
                    p.value(midpoint)?.max(l.value(midpoint)?) <= u.value(midpoint)?;
                if av <= au && bv <= bu {
                    result.push(RelationPiece {
                        metadata: 0,
                        left: [a, av],
                        right: [b, bv],
                        waiting: p.waiting,
                    });
                } else if a < midpoint && midpoint < b && middle_feasible {
                    let mut left = a;
                    let mut right = b;
                    for _ in 0..64 {
                        if p.value(left)?.max(l.value(left)?) <= u.value(left)? {
                            break;
                        }
                        left = next_toward(left, b);
                    }
                    for _ in 0..64 {
                        if p.value(right)?.max(l.value(right)?) <= u.value(right)? {
                            break;
                        }
                        right = next_toward(right, a);
                    }
                    let lv = p.value(left)?.max(l.value(left)?);
                    let rv = p.value(right)?.max(l.value(right)?);
                    if left > right || lv > u.value(left)? || rv > u.value(right)? {
                        return Err(invalid(
                            "paired bound crossing cannot be resolved at binary64 precision",
                        ));
                    }
                    result.push(RelationPiece {
                        metadata: 0,
                        left: [left, lv],
                        right: [right, rv],
                        waiting: p.waiting,
                    });
                } else {
                    if av <= au {
                        result.push(RelationPiece {
                            metadata: 0,
                            left: [a, av],
                            right: [a, av],
                            waiting: p.waiting,
                        });
                    }
                    if bv <= bu {
                        result.push(RelationPiece {
                            metadata: 0,
                            left: [b, bv],
                            right: [b, bv],
                            waiting: p.waiting,
                        });
                    }
                }
            }
        }
    }
    Ok(result)
}

fn next_toward(x: f64, target: f64) -> f64 {
    if x == target {
        return x;
    }
    if x == 0. {
        return if target > 0. {
            f64::from_bits(1)
        } else {
            -f64::from_bits(1)
        };
    }
    let bits = x.to_bits();
    f64::from_bits(if (target > x) == (x > 0.) {
        bits + 1
    } else {
        bits - 1
    })
}

#[cfg(test)]
#[path = "../../../../../tests/unit/graph/coupled/kernel/relation/tests.rs"]
mod tests;

#[cfg(test)]
#[path = "../../../../../tests/unit/graph/coupled/kernel/relation/metadata_tests.rs"]
mod metadata_tests;

#[cfg(test)]
#[path = "../../../../../tests/unit/graph/coupled/kernel/relation/covering_tests.rs"]
mod covering_tests;
