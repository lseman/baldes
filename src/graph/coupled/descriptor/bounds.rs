//! Function-valued lower clamps and upper filters on computed slope fronts.
//! Crossings are checked, never repaired by a feasibility epsilon. This is
//! front algebra; its output is not a directed source-program certificate.
use super::*;
use std::{sync::Arc, time::Instant};

type Answer<T> = Result<T, FrontierUnknown>;
#[derive(Clone, Copy, Debug)]
struct Segment {
    lo: f64,
    hi: f64,
    value: f64,
    slope: f64,
}
impl Segment {
    fn at(self, x: f64) -> Answer<f64> {
        finite(self.value + finite(self.slope * finite(x - self.lo)?)?)
    }
}
#[derive(Clone, Debug)]
pub struct PreparedClampFilter {
    lower: Arc<PreparedSlopeFunction>,
    upper: Arc<PreparedSlopeFunction>,
    lower_segments: Arc<[Segment]>,
    upper_segments: Arc<[Segment]>,
}
impl PartialEq for PreparedClampFilter {
    fn eq(&self, b: &Self) -> bool {
        self.lower == b.lower && self.upper == b.upper
    }
}
impl PreparedClampFilter {
    pub fn new(
        lower: &PreparedSlopeFunction,
        upper: &PreparedSlopeFunction,
    ) -> Result<Self, RawFunctionError> {
        Ok(Self {
            lower: Arc::new(lower.clone()),
            upper: Arc::new(upper.clone()),
            lower_segments: segments(lower)?,
            upper_segments: segments(upper)?,
        })
    }
    pub fn apply(
        &self,
        front: &SlopeFront,
        policy: FrontOperationPolicy,
        budget: FrontierBudget,
    ) -> FrontierOutcome<SlopeFront> {
        match self.apply_checked(front, policy, budget) {
            Ok(Some(front)) => FrontierOutcome::Complete(front),
            Ok(None) => FrontierOutcome::Empty,
            Err(reason) => FrontierOutcome::Unknown(reason),
        }
    }
    fn apply_checked(
        &self,
        front: &SlopeFront,
        policy: FrontOperationPolicy,
        budget: FrontierBudget,
    ) -> Answer<Option<SlopeFront>> {
        let mut work = Work::new(budget);
        let mut accepted = Vec::new();
        let (ll, lh) = self.lower.domain();
        let (ul, uh) = self.upper.domain();
        for component in front.components() {
            work.charge(component.pairs().len())?;
            let fs = segments(component).map_err(FrontierUnknown::Arithmetic)?;
            let domain = component.domain();
            let lo = domain.0.max(ll).max(ul);
            let hi = domain.1.min(lh).min(uh);
            if lo > hi {
                if policy == FrontOperationPolicy::NativeFrontAlgebra && near(lo, hi)? {
                    return Err(unsupported("near-disjoint bound domains"));
                }
                continue;
            }
            // Native truncation can snap a nonidentical endpoint. This helper
            // does not infer a topology from that tolerance.
            if policy == FrontOperationPolicy::NativeFrontAlgebra {
                for endpoint in [lo, hi] {
                    for source in [domain.0, domain.1, ll, lh, ul, uh] {
                        if endpoint != source && near(endpoint, source)? {
                            return Err(unsupported("near-nonidentical bound endpoint"));
                        }
                    }
                }
            }
            let mut fi = first(&fs, lo);
            let mut li = first(&self.lower_segments, lo);
            let mut ui = first(&self.upper_segments, lo);
            if lo == hi {
                point(
                    &mut accepted,
                    lo,
                    fs[fi],
                    self.lower_segments[li],
                    self.upper_segments[ui],
                    &mut work,
                )?;
                continue;
            }
            let mut x = lo;
            while x < hi {
                work.charge(1)?;
                while fs[fi].hi <= x && fi + 1 < fs.len() {
                    fi += 1;
                }
                while self.lower_segments[li].hi <= x && li + 1 < self.lower_segments.len() {
                    li += 1;
                }
                while self.upper_segments[ui].hi <= x && ui + 1 < self.upper_segments.len() {
                    ui += 1;
                }
                let f = fs[fi];
                let l = self.lower_segments[li];
                let u = self.upper_segments[ui];
                let end = hi.min(f.hi).min(l.hi).min(u.hi);
                if end <= x {
                    return Err(unsupported("nonadvancing bound segment stream"));
                }
                let mut cuts = [0.; 5];
                cuts[0] = x;
                cuts[1] = end;
                let mut count = 2;
                for (a, b) in [(f, l), (f, u), (l, u)] {
                    if let Some(cross) = crossing(a, b, x, end, policy, &mut work)? {
                        cuts[count] = cross;
                        count += 1;
                    }
                }
                cuts[..count].sort_by(f64::total_cmp);
                let mut unique = 1;
                for index in 1..count {
                    if cuts[index] != cuts[unique - 1] {
                        cuts[unique] = cuts[index];
                        unique += 1;
                    }
                }
                let cuts = &cuts[..unique];
                if policy == FrontOperationPolicy::NativeFrontAlgebra {
                    for pair in cuts.windows(2) {
                        if near(pair[0], pair[1])? {
                            return Err(unsupported("near-coincident bound crossings"));
                        }
                    }
                }
                for &cut in cuts {
                    point(&mut accepted, cut, f, l, u, &mut work)?;
                }
                for pair in cuts.windows(2) {
                    cell(&mut accepted, pair[0], pair[1], f, l, u, &mut work)?;
                }
                x = end;
            }
        }
        if accepted.is_empty() {
            return Ok(None);
        }
        accepted.sort_by(|a, b| a.lo.total_cmp(&b.lo).then(a.hi.total_cmp(&b.hi)));
        let functions = assemble(accepted, &mut work)?;
        match SlopeFront::from_components(functions, work.remaining()) {
            FrontierOutcome::Complete(out) => Ok(Some(out.with_provenance(
                FrontierProvenance::ClampFilter {
                    input: front.provenance().clone(),
                    lower: self.lower.clone(),
                    upper: self.upper.clone(),
                    policy,
                },
                front.used_tolerant_alignment(),
            ))),
            FrontierOutcome::Empty => Ok(None),
            FrontierOutcome::Unknown(reason) => Err(reason),
        }
    }
}
/// Clamp the complete front from below, then retain only coordinates meeting
/// the upper function. Unknown arithmetic/topology is distinct from empty.
pub fn clamp_filter(
    front: &SlopeFront,
    lower: &PreparedSlopeFunction,
    upper: &PreparedSlopeFunction,
    policy: FrontOperationPolicy,
    budget: FrontierBudget,
) -> FrontierOutcome<SlopeFront> {
    let Some(preparation) = lower.pairs().len().checked_add(upper.pairs().len()) else {
        return FrontierOutcome::Unknown(FrontierUnknown::WorkLimit);
    };
    if preparation > budget.max_operations {
        return FrontierOutcome::Unknown(FrontierUnknown::WorkLimit);
    }
    let start = budget.max_elapsed.map(|_| Instant::now());
    let prepared = match PreparedClampFilter::new(lower, upper) {
        Ok(prepared) => prepared,
        Err(error) => return FrontierOutcome::Unknown(FrontierUnknown::Arithmetic(error)),
    };
    let max_elapsed = match start.zip(budget.max_elapsed) {
        Some((start, limit)) => match limit.checked_sub(start.elapsed()) {
            Some(remaining) => Some(remaining),
            None => return FrontierOutcome::Unknown(FrontierUnknown::Deadline),
        },
        None => None,
    };
    prepared.apply(
        front,
        policy,
        FrontierBudget {
            max_operations: budget.max_operations - preparation,
            max_elapsed,
            ..budget
        },
    )
}
fn segments(function: &PreparedSlopeFunction) -> Result<Arc<[Segment]>, RawFunctionError> {
    let mut result = Vec::with_capacity(function.pairs().len());
    let (mut value, mut x) = function.pairs()[0];
    if function.pairs().len() == 1 {
        result.push(Segment {
            lo: x,
            hi: x,
            value,
            slope: 0.,
        });
    }
    for &(slope, end) in &function.pairs()[1..] {
        result.push(Segment {
            lo: x,
            hi: end,
            value,
            slope,
        });
        let product = slope * (end - x);
        let next = value + product;
        if !product.is_finite() || !next.is_finite() {
            return Err(RawFunctionError::ArithmeticOverflow(
                "bound segment preparation",
            ));
        }
        value = next;
        x = end;
    }
    Ok(result.into())
}
fn first(segments: &[Segment], x: f64) -> usize {
    segments
        .partition_point(|s| s.hi < x)
        .min(segments.len() - 1)
}
fn crossing(
    a: Segment,
    b: Segment,
    lo: f64,
    hi: f64,
    policy: FrontOperationPolicy,
    work: &mut Work,
) -> Answer<Option<f64>> {
    work.charge(1)?;
    let al = a.at(lo)?;
    let bl = b.at(lo)?;
    let ah = a.at(hi)?;
    let bh = b.at(hi)?;
    if al == bl && ah == bh && a.slope != b.slope {
        return Err(unsupported(
            "rounded coincident bounds have different slopes",
        ));
    }
    if (al <= bl && ah <= bh) || (al >= bl && ah >= bh) {
        return Ok(None);
    }
    let denominator = finite(a.slope - b.slope)?;
    if denominator == 0. {
        return Err(unsupported("parallel rounded bound crossing"));
    }
    let x = match policy {
        FrontOperationPolicy::StrictDomain => finite(lo + finite(finite(bl - al)? / denominator)?)?,
        // Match the expanded-intercept ordering recovered for native bound
        // crossing arithmetic. If its rounding does not yield a common value,
        // decline rather than snap a feasibility boundary.
        FrontOperationPolicy::NativeFrontAlgebra => finite(
            finite(finite(bh - finite(hi * b.slope)?)? - finite(ah - finite(hi * a.slope)?)?)?
                / denominator,
        )?,
    };
    if x <= lo || x >= hi || a.at(x)? != b.at(x)? {
        return Err(unsupported("unverified bound crossing"));
    }
    Ok(Some(x))
}
fn point(
    out: &mut Vec<Segment>,
    x: f64,
    f: Segment,
    l: Segment,
    u: Segment,
    work: &mut Work,
) -> Answer<()> {
    work.charge(1)?;
    let fv = f.at(x)?;
    let lv = l.at(x)?;
    let value = if fv >= lv { fv } else { lv };
    if value <= u.at(x)? {
        work.push(
            out,
            Segment {
                lo: x,
                hi: x,
                value,
                slope: 0.,
            },
        )?;
    }
    Ok(())
}
fn cell(
    out: &mut Vec<Segment>,
    lo: f64,
    hi: f64,
    f: Segment,
    l: Segment,
    u: Segment,
    work: &mut Work,
) -> Answer<()> {
    work.charge(1)?;
    let middle = finite(lo * 0.5 + hi * 0.5)?;
    if middle <= lo || middle >= hi {
        return Err(unsupported(
            "bound cell has no representable interior witness",
        ));
    }
    let selected = if f.at(middle)? >= l.at(middle)? { f } else { l };
    let other = if f.at(middle)? >= l.at(middle)? { l } else { f };
    if selected.at(lo)? < other.at(lo)? || selected.at(hi)? < other.at(hi)? {
        return Err(unsupported("unresolved lower-clamp branch"));
    }
    let left = selected.at(lo)?;
    let right = selected.at(hi)?;
    let upper_left = u.at(lo)?;
    let upper_right = u.at(hi)?;
    if left <= upper_left && right <= upper_right {
        let piece = Segment {
            lo,
            hi,
            value: left,
            slope: selected.slope,
        };
        // Reanchoring is itself a front operation. Verify its endpoint before
        // retaining a new segment; no source-expression equivalence is assumed.
        if piece.at(hi)? != right {
            return Err(unsupported("rounded bound segment reanchoring"));
        }
        work.push(out, piece)?;
    } else if left >= upper_left && right >= upper_right {
        if left == upper_left && right == upper_right {
            return Err(unsupported("ambiguous upper-bound plateau"));
        }
    } else {
        return Err(unsupported("unresolved upper-filter branch"));
    }
    Ok(())
}
fn assemble(input: Vec<Segment>, work: &mut Work) -> Answer<Vec<PreparedSlopeFunction>> {
    let mut result = Vec::new();
    let mut pairs: Vec<(f64, f64)> = Vec::new();
    let mut end = 0.;
    let mut value = 0.;
    for piece in input {
        work.charge(1)?;
        if !pairs.is_empty() && piece.lo < end {
            return Err(unsupported("overlapping bound output pieces"));
        }
        if !pairs.is_empty() && piece.lo == end {
            if value != piece.value {
                return Err(unsupported("discontinuous touching bound output"));
            }
            if piece.hi == end {
                continue;
            }
            pairs.push((piece.slope, piece.hi));
            end = piece.hi;
            value = piece.at(end)?;
            continue;
        }
        if !pairs.is_empty() {
            work.component(result.len() + 1)?;
            result.push(
                PreparedSlopeFunction::from_prepared_pairs(std::mem::take(&mut pairs))
                    .map_err(FrontierUnknown::Arithmetic)?,
            );
        }
        pairs.push((piece.value, piece.lo));
        end = piece.lo;
        value = piece.value;
        if piece.hi > piece.lo {
            pairs.push((piece.slope, piece.hi));
            end = piece.hi;
            value = piece.at(end)?;
        }
    }
    if !pairs.is_empty() {
        work.component(result.len() + 1)?;
        result.push(
            PreparedSlopeFunction::from_prepared_pairs(pairs)
                .map_err(FrontierUnknown::Arithmetic)?,
        );
    }
    Ok(result)
}
fn finite(x: f64) -> Answer<f64> {
    if x.is_finite() {
        Ok(x)
    } else {
        Err(FrontierUnknown::Arithmetic(
            RawFunctionError::ArithmeticOverflow("bound front arithmetic"),
        ))
    }
}
fn unsupported(message: &'static str) -> FrontierUnknown {
    FrontierUnknown::Unsupported(message)
}
fn near(a: f64, b: f64) -> Answer<bool> {
    let difference = finite(a - b)?.abs();
    let sum = finite(a + b)?.abs();
    Ok(difference < 0.0001 || difference < finite(sum * 2.220446049250313e-10)?)
}
struct Work {
    budget: FrontierBudget,
    operations: usize,
    pieces: usize,
    start: Option<Instant>,
}
impl Work {
    fn new(budget: FrontierBudget) -> Self {
        Self {
            budget,
            operations: 0,
            pieces: 0,
            start: budget.max_elapsed.map(|_| Instant::now()),
        }
    }
    fn charge(&mut self, n: usize) -> Answer<()> {
        self.operations = self
            .operations
            .checked_add(n)
            .ok_or(FrontierUnknown::WorkLimit)?;
        if self.operations > self.budget.max_operations {
            return Err(FrontierUnknown::WorkLimit);
        }
        if self
            .start
            .zip(self.budget.max_elapsed)
            .is_some_and(|(start, limit)| start.elapsed() >= limit)
        {
            return Err(FrontierUnknown::Deadline);
        }
        Ok(())
    }
    fn push(&mut self, out: &mut Vec<Segment>, piece: Segment) -> Answer<()> {
        self.pieces = self
            .pieces
            .checked_add(1)
            .ok_or(FrontierUnknown::PairLimit)?;
        if self.pieces > self.budget.max_pairs {
            return Err(FrontierUnknown::PairLimit);
        }
        out.push(piece);
        Ok(())
    }
    fn component(&self, count: usize) -> Answer<()> {
        if count > self.budget.max_components {
            Err(FrontierUnknown::ComponentLimit)
        } else {
            Ok(())
        }
    }
    fn remaining(&self) -> FrontierBudget {
        FrontierBudget {
            max_operations: self.budget.max_operations - self.operations,
            max_elapsed: self
                .start
                .zip(self.budget.max_elapsed)
                .map(|(start, limit)| limit.saturating_sub(start.elapsed())),
            ..self.budget
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn function(pairs: &[(f64, f64)]) -> PreparedSlopeFunction {
        PreparedSlopeFunction::from_prepared_pairs(pairs.to_vec()).unwrap()
    }
    fn line(value: f64, slope: f64, end: f64) -> PreparedSlopeFunction {
        function(&[(value, 0.), (slope, end)])
    }
    fn complete(outcome: FrontierOutcome<SlopeFront>) -> SlopeFront {
        match outcome {
            FrontierOutcome::Complete(front) => front,
            other => panic!("expected complete front: {other:?}"),
        }
    }
    fn value(front: &SlopeFront, x: f64) -> Option<f64> {
        front.components().iter().find_map(|component| {
            component
                .evaluate(x, EvaluationPolicy::StrictDomain)
                .unwrap()
        })
    }
    #[test]
    fn general_sloped_lower_and_upper_split_at_all_verified_crossings() {
        let front = SlopeFront::from_prepared(line(0., 0., 4.));
        let lower = line(-1., 1., 4.);
        let upper = line(3., -1., 4.);
        let result = complete(clamp_filter(
            &front,
            &lower,
            &upper,
            FrontOperationPolicy::StrictDomain,
            FrontierBudget::default(),
        ));
        assert_eq!(result.components().len(), 1);
        assert_eq!(result.components()[0].domain(), (0., 2.));
        for (x, y) in [(0., 0.), (0.5, 0.), (1., 0.), (1.5, 0.5), (2., 1.)] {
            assert_eq!(value(&result, x), Some(y));
        }
        assert_eq!(value(&result, 2.5), None);
        assert!(matches!(
            result.provenance().as_ref(),
            FrontierProvenance::ClampFilter { .. }
        ));
    }
    #[test]
    fn upper_filter_preserves_disconnected_components_and_lower_is_a_clamp() {
        let hill = SlopeFront::from_prepared(function(&[(0., 0.), (2., 1.), (-2., 2.)]));
        let low = line(-100., 0., 2.);
        let high = line(1., 0., 2.);
        let result = complete(clamp_filter(
            &hill,
            &low,
            &high,
            FrontOperationPolicy::StrictDomain,
            FrontierBudget::default(),
        ));
        let domains: Vec<_> = result
            .components()
            .iter()
            .map(PreparedSlopeFunction::domain)
            .collect();
        assert_eq!(domains, vec![(0., 0.5), (1.5, 2.)]);
        assert_eq!(value(&result, 1.), None);
        let valley = SlopeFront::from_prepared(function(&[(2., 0.), (-2., 1.), (2., 2.)]));
        let result = complete(clamp_filter(
            &valley,
            &high,
            &line(100., 0., 2.),
            FrontOperationPolicy::StrictDomain,
            FrontierBudget::default(),
        ));
        assert_eq!(result.components().len(), 1);
        assert_eq!(result.components()[0].domain(), (0., 2.));
        for (x, y) in [(0., 2.), (0.5, 1.), (1., 1.), (1.5, 1.), (2., 2.)] {
            assert_eq!(value(&result, x), Some(y));
        }
    }
    #[test]
    fn isolated_feasible_point_is_distinct_from_proven_empty() {
        let valley = SlopeFront::from_prepared(function(&[(2., 0.), (-2., 1.), (2., 2.)]));
        let low = line(-100., 0., 2.);
        let high = line(0., 0., 2.);
        let result = complete(clamp_filter(
            &valley,
            &low,
            &high,
            FrontOperationPolicy::StrictDomain,
            FrontierBudget::default(),
        ));
        assert_eq!(result.components()[0].pairs(), &[(0., 1.)]);
        let above = SlopeFront::from_prepared(line(2., 0., 2.));
        assert!(matches!(
            clamp_filter(
                &above,
                &low,
                &high,
                FrontOperationPolicy::StrictDomain,
                FrontierBudget::default()
            ),
            FrontierOutcome::Empty
        ));
    }
    #[test]
    fn cached_function_bounds_apply_independently_to_each_input_component() {
        let input = complete(SlopeFront::from_components(
            vec![line(0., 0., 1.), function(&[(0., 2.), (0., 3.)])],
            FrontierBudget::default(),
        ));
        let prepared = PreparedClampFilter::new(&line(-1., 1., 3.), &line(10., 0., 3.)).unwrap();
        let result = complete(prepared.apply(
            &input,
            FrontOperationPolicy::StrictDomain,
            FrontierBudget::default(),
        ));
        assert_eq!(
            result
                .components()
                .iter()
                .map(PreparedSlopeFunction::domain)
                .collect::<Vec<_>>(),
            vec![(0., 1.), (2., 3.)]
        );
        assert_eq!(value(&result, 2.), Some(1.));
        assert_eq!(value(&result, 3.), Some(2.));
        assert_eq!(value(&result, 1.5), None);
    }
    #[test]
    fn rounding_ambiguity_near_topology_and_work_exhaustion_are_unknown() {
        let input = SlopeFront::from_prepared(line(0.1, 0.3, 2.));
        let lower = line(-100., 0., 2.);
        let upper = line(0.05, 0.5, 2.);
        assert!(matches!(
            clamp_filter(
                &input,
                &lower,
                &upper,
                FrontOperationPolicy::NativeFrontAlgebra,
                FrontierBudget::default()
            ),
            FrontierOutcome::Unknown(_)
        ));
        let near = function(&[(0., 0.00001), (0., 2.)]);
        assert!(matches!(
            clamp_filter(
                &input,
                &near,
                &line(100., 0., 2.),
                FrontOperationPolicy::NativeFrontAlgebra,
                FrontierBudget::default()
            ),
            FrontierOutcome::Unknown(_)
        ));
        let exhausted = FrontierBudget {
            max_operations: 0,
            ..Default::default()
        };
        assert!(matches!(
            clamp_filter(
                &input,
                &lower,
                &upper,
                FrontOperationPolicy::StrictDomain,
                exhausted
            ),
            FrontierOutcome::Unknown(FrontierUnknown::WorkLimit)
        ));
    }
}
