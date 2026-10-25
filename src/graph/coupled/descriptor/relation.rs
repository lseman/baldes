//! Paired slope transitions. Unsupported arithmetic is never infeasibility.
use super::*;
use std::{sync::Arc, time::Instant};

#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "clp", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "clp", serde(deny_unknown_fields))]
pub struct SlopeRelationArc {
    pub arrival: PreparedSlopeFunction,
    pub increment: PreparedSlopeFunction,
    pub lower: PreparedSlopeFunction,
    pub upper: PreparedSlopeFunction,
    pub waiting: i32,
}

/// Immutable preparation is shared across labels; raw functions are not rebuilt.
#[derive(Clone, Debug)]
pub struct PreparedSlopeRelationArc {
    source: Arc<SlopeRelationArc>,
    policy: FrontOperationPolicy,
    shape: Result<Translation, &'static str>,
    inverse: FrontierOutcome<Arc<PreparedSlopeFunction>>,
    composition: FrontierOutcome<PreparedCompositionMap>,
    increment_front: FrontierOutcome<SlopeFront>,
    bounds: PreparedClampFilter,
}
impl PartialEq for PreparedSlopeRelationArc {
    fn eq(&self, other: &Self) -> bool {
        self.source == other.source && self.policy == other.policy
    }
}
#[derive(Clone, Debug, PartialEq)]
struct Translation {
    delta: f64,
    phi: f64,
    tail_lo: f64,
    tail_hi: f64,
    head_lo: f64,
    head_hi: f64,
}
#[derive(Clone, Copy)]
struct Segment {
    lo: f64,
    hi: f64,
    y: f64,
    slope: f64,
}

impl PreparedSlopeRelationArc {
    pub fn new(
        source: &SlopeRelationArc,
        policy: FrontOperationPolicy,
    ) -> Result<Self, RawFunctionError> {
        if source.arrival.pairs().iter().skip(1).any(|p| p.0 <= 0.) {
            return Err(RawFunctionError::Unsupported(
                "arrival must be strictly increasing",
            ));
        }
        let mut previous = source.arrival.pairs()[0].1;
        let mut mapped = source.arrival.pairs()[0].0;
        if mapped < previous {
            return Err(RawFunctionError::Unsupported("arrival precedes departure"));
        }
        for &(slope, x) in &source.arrival.pairs()[1..] {
            mapped += slope * (x - previous);
            if !mapped.is_finite() {
                return Err(RawFunctionError::ArithmeticOverflow("arrival validation"));
            }
            if mapped < x {
                return Err(RawFunctionError::Unsupported("arrival precedes departure"));
            }
            previous = x;
        }
        let shape = (|| {
            let constant = |f: &PreparedSlopeFunction| {
                f.pairs()
                    .iter()
                    .skip(1)
                    .all(|p| p.0 == 0.)
                    .then_some(f.pairs()[0].0)
            };
            if source.arrival.pairs().len() > 2
                || !source.arrival.pairs().iter().skip(1).all(|p| p.0 == 1.)
                || source.arrival.pairs()[0].1 != 0.
            {
                return Err("general arrival composition");
            }
            let delta = source.arrival.pairs()[0].0;
            if delta < 0. {
                return Err("non-FIFO arrival translation");
            }
            let phi = constant(&source.increment).ok_or("nonconstant paired increment")?;
            let (al, ah) = source.arrival.domain();
            let (pl, ph) = source.increment.domain();
            let (ll, lh) = source.lower.domain();
            let (ul, uh) = source.upper.domain();
            Ok(Translation {
                delta,
                phi,
                tail_lo: al.max(pl),
                tail_hi: ah.min(ph),
                head_lo: ll.max(ul),
                head_hi: lh.min(uh),
            })
        })();
        let inverse = match inverse_increasing(&source.arrival, FrontierBudget::default()) {
            FrontierOutcome::Complete(f) => FrontierOutcome::Complete(Arc::new(f)),
            FrontierOutcome::Empty => FrontierOutcome::Empty,
            FrontierOutcome::Unknown(e) => FrontierOutcome::Unknown(e),
        };
        let composition = match &inverse {
            FrontierOutcome::Complete(f) => {
                PreparedCompositionMap::new(f, FrontierBudget::default())
            }
            FrontierOutcome::Empty => FrontierOutcome::Empty,
            FrontierOutcome::Unknown(e) => FrontierOutcome::Unknown(e.clone()),
        };
        // Native preparation constructs Phi = phi composed with q inverse.
        // Runtime composes F separately, then adds this cached Phi.
        let increment_front = match (&composition, &inverse) {
            (FrontierOutcome::Complete(map), FrontierOutcome::Complete(inverse)) => {
                let (lo, hi) = inverse.domain();
                match map.compose_clipped(
                    &source.increment,
                    lo,
                    hi,
                    policy,
                    FrontierBudget::default(),
                ) {
                    FrontierOutcome::Complete(f) => {
                        FrontierOutcome::Complete(SlopeFront::from_prepared(f))
                    }
                    FrontierOutcome::Empty => FrontierOutcome::Empty,
                    FrontierOutcome::Unknown(e) => FrontierOutcome::Unknown(e),
                }
            }
            (FrontierOutcome::Unknown(e), _) => FrontierOutcome::Unknown(e.clone()),
            (_, FrontierOutcome::Unknown(e)) => FrontierOutcome::Unknown(e.clone()),
            _ => FrontierOutcome::Empty,
        };
        Ok(Self {
            source: Arc::new(source.clone()),
            policy,
            shape,
            inverse,
            composition,
            increment_front,
            bounds: PreparedClampFilter::new(&source.lower, &source.upper)?,
        })
    }

    /// Ordered slope addition and monotone secant composition followed by a
    /// running-min waiting envelope, lower clamping and upper filtering.
    /// Unresolved numerical crossings return Unknown, never infeasibility.
    pub fn extend(
        &self,
        front: &SlopeFront,
        budget: FrontierBudget,
    ) -> FrontierOutcome<SlopeFront> {
        match self.extend_checked(front, budget) {
            Ok(Some(out)) => FrontierOutcome::Complete(out),
            Ok(None) => FrontierOutcome::Empty,
            Err(reason) => FrontierOutcome::Unknown(reason),
        }
    }
    fn extend_checked(
        &self,
        front: &SlopeFront,
        budget: FrontierBudget,
    ) -> Result<Option<SlopeFront>, FrontierUnknown> {
        if front.components().len() > budget.max_components {
            return Err(FrontierUnknown::ComponentLimit);
        }
        let mut pairs = 0usize;
        for f in front.components() {
            pairs = pairs
                .checked_add(f.pairs().len())
                .ok_or(FrontierUnknown::PairLimit)?;
            if pairs > budget.max_pairs {
                return Err(FrontierUnknown::PairLimit);
            }
            if pairs > budget.max_operations / 2 {
                return Err(FrontierUnknown::WorkLimit);
            }
        }
        // Covers the preflight and exact-transport eligibility scan before
        // partitioning the remaining allowance between actual operators.
        let budget = FrontierBudget {
            max_operations: budget.max_operations - 2 * pairs,
            ..budget
        };
        match self.shape.as_ref() {
            Ok(t) if exact_translation(front, t) => self.extend_waiting(front, front, t, budget),
            _ => self.extend_composed(front, budget),
        }
    }
    fn extend_waiting(
        &self,
        front: &SlopeFront,
        origin: &SlopeFront,
        t: &Translation,
        budget: FrontierBudget,
    ) -> Result<Option<SlopeFront>, FrontierUnknown> {
        let start = budget.max_elapsed.map(|_| Instant::now());
        let unsupported = |s| FrontierUnknown::Unsupported(s);
        let a = &self.source;
        // Independent bounded passes share one aggregate operation allowance:
        // input scan, envelope, geometry assembly, validation and bound filter.
        let budget = FrontierBudget {
            max_operations: budget.max_operations / 5,
            ..budget
        };
        let (delta, phi) = (t.delta, t.phi);
        let (tail_lo, tail_hi, head_lo, head_hi) = (t.tail_lo, t.tail_hi, t.head_lo, t.head_hi);
        if tail_lo > tail_hi || head_lo > head_hi {
            return Ok(None);
        }
        let mut input = Vec::new();
        let mut work = 0usize;
        for f in front.components() {
            work = work.checked_add(1).ok_or(FrontierUnknown::WorkLimit)?;
            if work > budget.max_operations || input.len() >= budget.max_pairs {
                return Err(FrontierUnknown::WorkLimit);
            }
            if start
                .zip(budget.max_elapsed)
                .is_some_and(|(s, d)| s.elapsed() > d)
            {
                return Err(FrontierUnknown::Deadline);
            }
            let mut x = f.pairs()[0].1;
            // Constant Phi is added once to the component anchor. Re-adding it
            // independently at each source knot changes ordered front algebra.
            let mut y = finite(f.pairs()[0].0 + phi)?;
            if f.pairs().len() == 1 && x >= tail_lo && x <= tail_hi {
                input.push(Segment {
                    lo: finite(x + delta)?,
                    hi: finite(x + delta)?,
                    y,
                    slope: 0.,
                });
            }
            for &(s, e) in &f.pairs()[1..] {
                work = work.checked_add(1).ok_or(FrontierUnknown::WorkLimit)?;
                if work > budget.max_operations || work > budget.max_pairs {
                    return Err(FrontierUnknown::WorkLimit);
                }
                let lo = x.max(tail_lo);
                let hi = e.min(tail_hi);
                if lo <= hi {
                    if self.policy == FrontOperationPolicy::NativeFrontAlgebra
                        && ((lo != x && near(lo, x)?) || (hi != e && near(hi, e)?))
                    {
                        return Err(unsupported("unresolved native tail restriction alignment"));
                    }
                    if lo < hi && finite(lo + delta)? >= finite(hi + delta)? {
                        return Err(unsupported("collapsed arrival coordinates"));
                    }
                    input.push(Segment {
                        lo: finite(lo + delta)?,
                        hi: finite(hi + delta)?,
                        y: finite(y + finite(s * finite(lo - x)?)?)?,
                        slope: s,
                    });
                }
                y = finite(y + finite(s * finite(e - x)?)?)?;
                x = e;
            }
        }
        if input.is_empty() {
            return Ok(None);
        }
        // Running record at the current coordinate. A waiting ray bridges gaps.
        let k = f64::from(a.waiting);
        let mut output = Vec::new();
        let mut end = input[0].lo;
        let mut record = input[0].y;
        for (index, seg) in input.into_iter().enumerate() {
            if seg.lo > end {
                push(
                    &mut output,
                    Segment {
                        lo: end,
                        hi: seg.lo,
                        y: record,
                        slope: k,
                    },
                    budget,
                )?;
                record = value(record, k, seg.lo - end)?;
            }
            if seg.y < record {
                if index > 0 {
                    return Err(unsupported("downward discontinuity in waiting envelope"));
                }
                record = seg.y;
            }
            let difference = finite(seg.y - record)?;
            if seg.slope < k && difference > 0. {
                let crossing = finite(seg.lo + finite(difference / finite(k - seg.slope)?)?)?;
                if crossing < seg.hi {
                    // A rounded crossing must not silently create a false feasible
                    // interval. Unresolved branch equality is an explicit unknown.
                    if value(seg.y, seg.slope, crossing - seg.lo)?
                        != value(record, k, crossing - seg.lo)?
                    {
                        return Err(unsupported("rounded waiting crossing"));
                    }
                    push(
                        &mut output,
                        Segment {
                            lo: seg.lo,
                            hi: crossing,
                            y: record,
                            slope: k,
                        },
                        budget,
                    )?;
                    let cy = value(seg.y, seg.slope, crossing - seg.lo)?;
                    push(
                        &mut output,
                        Segment {
                            lo: crossing,
                            hi: seg.hi,
                            y: cy,
                            slope: seg.slope,
                        },
                        budget,
                    )?;
                    record = value(cy, seg.slope, seg.hi - crossing)?;
                } else {
                    push(
                        &mut output,
                        Segment {
                            lo: seg.lo,
                            hi: seg.hi,
                            y: record,
                            slope: k,
                        },
                        budget,
                    )?;
                    record = value(record, k, seg.hi - seg.lo)?;
                }
            } else {
                let slope = if difference == 0. {
                    seg.slope.min(k)
                } else {
                    k
                };
                push(
                    &mut output,
                    Segment {
                        lo: seg.lo,
                        hi: seg.hi,
                        y: record,
                        slope,
                    },
                    budget,
                )?;
                record = value(record, slope, seg.hi - seg.lo)?;
            }
            end = seg.hi;
        }
        if end < head_hi {
            push(
                &mut output,
                Segment {
                    lo: end,
                    hi: head_hi,
                    y: record,
                    slope: k,
                },
                budget,
            )?;
        }
        let mut functions = Vec::new();
        for s in output {
            work = work.checked_add(1).ok_or(FrontierUnknown::WorkLimit)?;
            if work > budget.max_operations {
                return Err(FrontierUnknown::WorkLimit);
            }
            if functions.len().saturating_add(1) > budget.max_pairs {
                return Err(FrontierUnknown::PairLimit);
            }
            if start
                .zip(budget.max_elapsed)
                .is_some_and(|(s, d)| s.elapsed() > d)
            {
                return Err(FrontierUnknown::Deadline);
            }
            let lo = s.lo.max(head_lo);
            let hi = s.hi.min(head_hi);
            if lo > hi {
                continue;
            }
            if self.policy == FrontOperationPolicy::NativeFrontAlgebra
                && ((lo != s.lo && near(lo, s.lo)?) || (hi != s.hi && near(hi, s.hi)?))
            {
                return Err(unsupported(
                    "unresolved native waiting restriction alignment",
                ));
            }
            let y = value(s.y, s.slope, lo - s.lo)?;
            let function = if lo == hi {
                PreparedSlopeFunction::constant_point(y, lo)
            } else {
                PreparedSlopeFunction::from_prepared_pairs(vec![(y, lo), (s.slope, hi)])
            }
            .map_err(FrontierUnknown::Arithmetic)?;
            functions.push(function);
        }
        // Adjacent pieces are one component, retaining literal segment slopes.
        let mut components: Vec<PreparedSlopeFunction> = Vec::new();
        let mut current: Vec<(f64, f64)> = Vec::new();
        let mut current_y = 0.;
        for f in functions {
            if let Some(last) = current.last() {
                if f.domain().0 < last.1 {
                    if f.pairs().len() == 1 {
                        continue;
                    }
                    return Err(unsupported("overlapping waiting output"));
                }
                if last.1 == f.domain().0 {
                    if current_y != f.pairs()[0].0 {
                        return Err(unsupported("rounded waiting boundary"));
                    }
                    current_y = f
                        .evaluate(f.domain().1, EvaluationPolicy::StrictDomain)
                        .map_err(FrontierUnknown::Arithmetic)?
                        .ok_or_else(|| unsupported("missing waiting endpoint"))?;
                    current.extend_from_slice(&f.pairs()[1..]);
                    continue;
                }
                components.push(
                    PreparedSlopeFunction::from_prepared_pairs(std::mem::take(&mut current))
                        .map_err(FrontierUnknown::Arithmetic)?,
                );
            }
            current_y = f
                .evaluate(f.domain().1, EvaluationPolicy::StrictDomain)
                .map_err(FrontierUnknown::Arithmetic)?
                .ok_or_else(|| unsupported("missing waiting endpoint"))?;
            current.extend_from_slice(f.pairs());
        }
        if !current.is_empty() {
            components.push(
                PreparedSlopeFunction::from_prepared_pairs(current)
                    .map_err(FrontierUnknown::Arithmetic)?,
            );
        }
        match SlopeFront::from_components(components, budget) {
            FrontierOutcome::Complete(out) => match self.bounds.apply(
                &out,
                self.policy,
                FrontierBudget {
                    max_elapsed: start
                        .zip(budget.max_elapsed)
                        .map(|(s, d)| d.saturating_sub(s.elapsed())),
                    ..budget
                },
            ) {
                FrontierOutcome::Complete(out) => Ok(Some(out.paired_result(
                    origin,
                    self.source.clone(),
                    self.policy,
                    front.used_tolerant_alignment(),
                ))),
                FrontierOutcome::Empty => Ok(None),
                FrontierOutcome::Unknown(e) => Err(e),
            },
            FrontierOutcome::Empty => Ok(None),
            FrontierOutcome::Unknown(e) => Err(e),
        }
    }

    fn extend_composed(
        &self,
        front: &SlopeFront,
        budget: FrontierBudget,
    ) -> Result<Option<SlopeFront>, FrontierUnknown> {
        let start = budget.max_elapsed.map(|_| Instant::now());
        let inverse = match &self.inverse {
            FrontierOutcome::Complete(f) => f,
            FrontierOutcome::Empty => return Ok(None),
            FrontierOutcome::Unknown(e) => return Err(e.clone()),
        };
        let composition = match &self.composition {
            FrontierOutcome::Complete(c) => c,
            FrontierOutcome::Empty => return Ok(None),
            FrontierOutcome::Unknown(e) => return Err(e.clone()),
        };
        let stages = front
            .components()
            .len()
            .checked_add(3)
            .ok_or(FrontierUnknown::WorkLimit)?;
        let stage_budget = || FrontierBudget {
            max_operations: budget.max_operations / stages,
            max_elapsed: start
                .zip(budget.max_elapsed)
                .map(|(s, d)| d.saturating_sub(s.elapsed())),
            ..budget
        };
        let mut components = Vec::new();
        let mut pairs = 0usize;
        for f in front.components() {
            let (lo, hi) = inverse.domain();
            match composition.compose_clipped(f, lo, hi, self.policy, stage_budget()) {
                FrontierOutcome::Complete(f) => {
                    pairs = pairs
                        .checked_add(f.pairs().len())
                        .ok_or(FrontierUnknown::PairLimit)?;
                    if pairs > budget.max_pairs {
                        return Err(FrontierUnknown::PairLimit);
                    }
                    components.push(f);
                }
                FrontierOutcome::Empty => {}
                FrontierOutcome::Unknown(e) => return Err(e),
            }
        }
        let transformed = match SlopeFront::from_components(components, stage_budget()) {
            FrontierOutcome::Complete(f) => f,
            FrontierOutcome::Empty => return Ok(None),
            FrontierOutcome::Unknown(e) => return Err(e),
        };
        let phi = match &self.increment_front {
            FrontierOutcome::Complete(f) => f,
            FrontierOutcome::Empty => return Ok(None),
            FrontierOutcome::Unknown(e) => return Err(e.clone()),
        };
        let charged = match transformed.add(phi, self.policy, stage_budget()) {
            FrontierOutcome::Complete(f) => f,
            FrontierOutcome::Empty => return Ok(None),
            FrontierOutcome::Unknown(e) => return Err(e),
        };
        let (tail_lo, tail_hi) = inverse.domain();
        let (ll, lh) = self.source.lower.domain();
        let (ul, uh) = self.source.upper.domain();
        let shape = Translation {
            delta: 0.,
            phi: 0.,
            tail_lo,
            tail_hi,
            head_lo: ll.max(ul),
            head_hi: lh.min(uh),
        };
        self.extend_waiting(&charged, front, &shape, stage_budget())
    }
}
// Retaining slopes through translation agrees with recovered secant
// composition only when all intermediate endpoint arithmetic is exact.
// Other geometry uses the general prepared composition operation instead.
fn exact_translation(front: &SlopeFront, t: &Translation) -> bool {
    let integer = |x: f64| x.is_finite() && x.abs() <= ((1u64 << 40) as f64) && x.fract() == 0.;
    if !integer(t.delta) || !integer(t.phi) || !integer(t.tail_lo) || !integer(t.tail_hi) {
        return false;
    }
    for f in front.components() {
        let mut x = f.pairs()[0].1;
        let mut y = f.pairs()[0].0 + t.phi;
        if !integer(x) || !integer(x + t.delta) || !integer(f.pairs()[0].0) || !integer(y) {
            return false;
        }
        for &(s, end) in &f.pairs()[1..] {
            let product = s * (end - x);
            y += product;
            if !integer(s)
                || !integer(end)
                || !integer(end + t.delta)
                || !integer(product)
                || !integer(y)
            {
                return false;
            }
            x = end;
        }
    }
    true
}
fn finite(x: f64) -> Result<f64, FrontierUnknown> {
    if x.is_finite() {
        Ok(x)
    } else {
        Err(FrontierUnknown::Arithmetic(
            RawFunctionError::ArithmeticOverflow("paired slope operation"),
        ))
    }
}
fn near(a: f64, b: f64) -> Result<bool, FrontierUnknown> {
    let distance = finite(a - b)?.abs();
    Ok(distance < 0.0001 || distance < finite(finite(a + b)?.abs() * 2.220446049250313e-10)?)
}
fn value(y: f64, s: f64, dx: f64) -> Result<f64, FrontierUnknown> {
    finite(y + finite(s * dx)?)
}
fn push(out: &mut Vec<Segment>, s: Segment, b: FrontierBudget) -> Result<(), FrontierUnknown> {
    if out.len() >= b.max_pairs || out.len() >= b.max_operations {
        return Err(FrontierUnknown::PairLimit);
    }
    out.push(s);
    Ok(())
}

#[cfg(test)]
#[path = "../../../../tests/unit/graph/coupled/descriptor/relation/tests.rs"]
mod tests;
