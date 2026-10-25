//! Slope FRONT algebra; deliberately distinct from source-expression replay.
use super::{EvaluationPolicy, PreparedSlopeFunction, RawFunctionError};
use std::{
    sync::Arc,
    time::{Duration, Instant},
};

/// Source evaluation policy does not implicitly select an operation policy.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "clp", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "clp", serde(rename_all = "snake_case"))]
pub enum FrontOperationPolicy {
    /// Exact domain intersection and exact coordinate comparisons.
    StrictDomain,
    /// Native left-coordinate choices at near-aligned segment ends and starts.
    NativeFrontAlgebra,
}
#[derive(Clone, Copy, Debug)]
pub struct FrontierBudget {
    pub max_operations: usize,
    pub max_components: usize,
    pub max_pairs: usize,
    pub max_elapsed: Option<Duration>,
}
impl Default for FrontierBudget {
    fn default() -> Self {
        Self {
            max_operations: 1_000_000,
            max_components: 4096,
            max_pairs: 4096,
            max_elapsed: None,
        }
    }
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FrontierUnknown {
    WorkLimit,
    ComponentLimit,
    PairLimit,
    Deadline,
    Arithmetic(RawFunctionError),
    Unsupported(&'static str),
}
#[derive(Clone, Debug)]
pub enum FrontierOutcome<T> {
    Complete(T),
    Empty,
    Unknown(FrontierUnknown),
}
/// Persistent provenance of performed front operations, not literal replay.
#[derive(Clone, Debug)]
pub enum FrontierProvenance {
    ClampFilter {
        input: Arc<Self>,
        lower: Arc<PreparedSlopeFunction>,
        upper: Arc<PreparedSlopeFunction>,
        policy: FrontOperationPolicy,
    },
    /// Paired transition retains the original raw descriptors and waiting rate.
    Paired {
        input: Arc<Self>,
        arc: Arc<super::SlopeRelationArc>,
        policy: FrontOperationPolicy,
        intermediate_tolerant_alignment: bool,
    },
    Source(Arc<[PreparedSlopeFunction]>),
    Addition {
        left: Arc<Self>,
        right: Arc<Self>,
        policy: FrontOperationPolicy,
    },
    Translation {
        input: Arc<Self>,
        delta: f64,
    },
    Restriction {
        input: Arc<Self>,
        lower: f64,
        upper: f64,
        policy: FrontOperationPolicy,
    },
}
#[derive(Clone, Debug)]
pub struct SlopeFront {
    components: Arc<[PreparedSlopeFunction]>,
    provenance: Arc<FrontierProvenance>,
    tolerant_alignment: bool,
}
/// Result of the front algebra's ordered endpoint scan. Re-evaluation under
/// a source function's tolerance/snap policy is a different operation and is
/// not asserted to reproduce this value or certify a source-program minimum.
#[derive(Clone, Debug)]
pub struct FrontAlgebraWitness {
    pub coordinate: f64,
    pub value: f64,
    pub component: usize,
    pub provenance: Arc<FrontierProvenance>,
}
/// Verified ordered charges evaluated at a shared argument. This is a distinct
/// witness scope; its value can differ from the added front's value/minimum.
#[derive(Clone, Debug)]
pub struct SourceReplayWitness {
    argument: f64,
    value: f64,
    charges: Arc<[f64]>,
}
impl SourceReplayWitness {
    pub fn evaluate_shared_argument(
        functions: &[PreparedSlopeFunction],
        argument: f64,
        policy: EvaluationPolicy,
    ) -> Result<Option<Self>, RawFunctionError> {
        if !argument.is_finite() {
            return Err(RawFunctionError::NonFinite("argument"));
        }
        let mut value = 0.;
        let mut charges = Vec::with_capacity(functions.len());
        for function in functions {
            let Some(charge) = function.evaluate(argument, policy)? else {
                return Ok(None);
            };
            value = checked(value + charge, "source replay accumulation")?;
            charges.push(charge);
        }
        Ok(Some(Self {
            argument,
            value,
            charges: charges.into(),
        }))
    }
    pub fn argument(&self) -> f64 {
        self.argument
    }
    pub fn value(&self) -> f64 {
        self.value
    }
    pub fn charge_values(&self) -> &[f64] {
        &self.charges
    }
}
impl SlopeFront {
    pub(super) fn with_provenance(
        mut self,
        provenance: FrontierProvenance,
        tolerant_alignment: bool,
    ) -> Self {
        self.provenance = Arc::new(provenance);
        self.tolerant_alignment |= tolerant_alignment;
        self
    }
    pub(super) fn paired_result(
        mut self,
        input: &Self,
        arc: Arc<super::SlopeRelationArc>,
        policy: FrontOperationPolicy,
        intermediate_tolerant_alignment: bool,
    ) -> Self {
        self.provenance = Arc::new(FrontierProvenance::Paired {
            input: input.provenance.clone(),
            arc,
            policy,
            intermediate_tolerant_alignment,
        });
        self.tolerant_alignment |= input.tolerant_alignment || intermediate_tolerant_alignment;
        self
    }
    pub fn zero(lower: f64, upper: f64) -> Result<Self, RawFunctionError> {
        let function = super::RawAffine::new(0., 0.)?.prepare(lower, upper)?;
        Ok(Self::from_prepared(function))
    }

    /// Translate every retained coordinate with a literal addition. Slopes
    /// and anchor values remain unchanged. A rounded coordinate collapse is
    /// unsupported, rather than a reason to drop one of the source pieces.
    pub fn translate(&self, delta: f64, budget: FrontierBudget) -> FrontierOutcome<Self> {
        if !delta.is_finite() {
            return FrontierOutcome::Unknown(FrontierUnknown::Arithmetic(
                RawFunctionError::NonFinite("translation"),
            ));
        }
        let mut work = Work::new(budget);
        if self.components.len() > budget.max_components {
            return FrontierOutcome::Unknown(FrontierUnknown::ComponentLimit);
        }
        let mut components = Vec::with_capacity(self.components.len());
        for component in self.components.iter() {
            let mut pairs = Vec::with_capacity(component.pairs().len());
            let mut previous = None;
            for &(value_or_slope, coordinate) in component.pairs() {
                if let Err(reason) = work.pair() {
                    return FrontierOutcome::Unknown(reason);
                }
                let translated =
                    match arithmetic(coordinate + delta, "front coordinate translation") {
                        Ok(value) => value,
                        Err(reason) => return FrontierOutcome::Unknown(reason),
                    };
                if previous.is_some_and(|previous| translated <= previous) {
                    return FrontierOutcome::Unknown(FrontierUnknown::Unsupported(
                        "translation collapsed consecutive coordinates",
                    ));
                }
                pairs.push((value_or_slope, translated));
                previous = Some(translated);
            }
            match PreparedSlopeFunction::from_prepared_pairs(pairs) {
                Ok(component) => {
                    if components.len() >= budget.max_components {
                        return FrontierOutcome::Unknown(FrontierUnknown::ComponentLimit);
                    }
                    components.push(component);
                }
                Err(error) => return FrontierOutcome::Unknown(FrontierUnknown::Arithmetic(error)),
            }
        }
        if let Err(reason) = validate(&components, &mut work) {
            return FrontierOutcome::Unknown(reason);
        }
        FrontierOutcome::Complete(Self {
            components: components.into(),
            provenance: Arc::new(FrontierProvenance::Translation {
                input: self.provenance.clone(),
                delta,
            }),
            tolerant_alignment: self.tolerant_alignment,
        })
    }

    /// Intersect the component union with the exact requested closed window.
    /// Re-anchor at its lower coordinate and retain native slopes; no secant
    /// fitting or bridging a gap occurs. Unrecovered near-boundary native
    /// truncation cases remain Unknown instead of guessing their topology.
    pub fn restrict(
        &self,
        lower: f64,
        upper: f64,
        policy: FrontOperationPolicy,
        budget: FrontierBudget,
    ) -> FrontierOutcome<Self> {
        if !lower.is_finite() || !upper.is_finite() {
            return FrontierOutcome::Unknown(FrontierUnknown::Arithmetic(
                RawFunctionError::NonFinite("restriction window"),
            ));
        }
        if lower > upper {
            return FrontierOutcome::Unknown(FrontierUnknown::Arithmetic(
                RawFunctionError::InvalidDomain,
            ));
        }
        let mut work = Work::new(budget);
        let mut components = Vec::new();
        for function in self.components.iter() {
            if let Err(reason) = work.charge(function.pairs().len()) {
                return FrontierOutcome::Unknown(reason);
            }
            let (start, end) = function.domain();
            let lo = start.max(lower);
            let hi = end.min(upper);
            if lo > hi {
                continue;
            }
            if policy == FrontOperationPolicy::NativeFrontAlgebra {
                for &(_, boundary) in function.pairs() {
                    for coordinate in [lo, hi] {
                        if coordinate != boundary {
                            match near(coordinate, boundary, policy) {
                                Ok(true) => {
                                    return FrontierOutcome::Unknown(FrontierUnknown::Unsupported(
                                        "nonexact near-boundary truncation needs a topology proof",
                                    ))
                                }
                                Err(reason) => return FrontierOutcome::Unknown(reason),
                                Ok(false) => (),
                            }
                        }
                    }
                }
            }
            let anchor = match value_at(function, lo, policy) {
                Ok(value) => value,
                Err(reason) => return FrontierOutcome::Unknown(reason),
            };
            if let Err(reason) = work.pair() {
                return FrontierOutcome::Unknown(reason);
            }
            let mut pairs = vec![(anchor, lo)];
            if lo != hi {
                for &(slope, segment_end) in &function.pairs()[1..] {
                    if segment_end <= lo {
                        continue;
                    }
                    if let Err(reason) = work.pair() {
                        return FrontierOutcome::Unknown(reason);
                    }
                    pairs.push((slope, segment_end.min(hi)));
                    if segment_end >= hi {
                        break;
                    }
                }
            }
            match PreparedSlopeFunction::from_prepared_pairs(pairs) {
                Ok(component) => {
                    if components.len() >= budget.max_components {
                        return FrontierOutcome::Unknown(FrontierUnknown::ComponentLimit);
                    }
                    components.push(component);
                }
                Err(error) => return FrontierOutcome::Unknown(FrontierUnknown::Arithmetic(error)),
            }
        }
        if components.is_empty() {
            return FrontierOutcome::Empty;
        }
        if let Err(reason) = validate(&components, &mut work) {
            return FrontierOutcome::Unknown(reason);
        }
        FrontierOutcome::Complete(Self {
            components: components.into(),
            provenance: Arc::new(FrontierProvenance::Restriction {
                input: self.provenance.clone(),
                lower,
                upper,
                policy,
            }),
            tolerant_alignment: self.tolerant_alignment,
        })
    }
    pub fn from_prepared(function: PreparedSlopeFunction) -> Self {
        let components: Arc<[PreparedSlopeFunction]> = vec![function].into();
        Self {
            provenance: Arc::new(FrontierProvenance::Source(components.clone())),
            components,
            tolerant_alignment: false,
        }
    }
    /// Components must be strictly disjoint and sorted. Overlaps/touching
    /// components need an envelope, which this addition primitive does not fake.
    pub fn from_components(
        components: Vec<PreparedSlopeFunction>,
        budget: FrontierBudget,
    ) -> FrontierOutcome<Self> {
        if components.is_empty() {
            return FrontierOutcome::Empty;
        }
        let mut work = Work::new(budget);
        if let Err(reason) = validate(&components, &mut work) {
            return FrontierOutcome::Unknown(reason);
        }
        let components: Arc<[PreparedSlopeFunction]> = components.into();
        FrontierOutcome::Complete(Self {
            provenance: Arc::new(FrontierProvenance::Source(components.clone())),
            components,
            tolerant_alignment: false,
        })
    }
    pub fn components(&self) -> &[PreparedSlopeFunction] {
        &self.components
    }
    pub fn provenance(&self) -> &Arc<FrontierProvenance> {
        &self.provenance
    }
    pub fn used_tolerant_alignment(&self) -> bool {
        self.tolerant_alignment
    }
    /// Add active slopes directly, without fitting rounded endpoint values.
    pub fn add(
        &self,
        other: &Self,
        policy: FrontOperationPolicy,
        budget: FrontierBudget,
    ) -> FrontierOutcome<Self> {
        let mut work = Work::new(budget);
        let mut output = Vec::new();
        let mut tolerant = self.tolerant_alignment || other.tolerant_alignment;
        let (mut i, mut j) = (0, 0);
        while i < self.components.len() && j < other.components.len() {
            if let Err(reason) = work.charge(1) {
                return FrontierOutcome::Unknown(reason);
            }
            let a = &self.components[i];
            let b = &other.components[j];
            match add_component(a, b, policy, &mut work) {
                Ok(Some((component, aligned))) => {
                    if output.len() >= budget.max_components {
                        return FrontierOutcome::Unknown(FrontierUnknown::ComponentLimit);
                    }
                    output.push(component);
                    tolerant |= aligned;
                }
                Ok(None) => (),
                Err(reason) => return FrontierOutcome::Unknown(reason),
            }
            let ae = a.domain().1;
            let be = b.domain().1;
            if ae <= be {
                i += 1;
            }
            if be <= ae {
                j += 1;
            }
        }
        if output.is_empty() {
            return FrontierOutcome::Empty;
        }
        if let Err(reason) = validate(&output, &mut work) {
            return FrontierOutcome::Unknown(reason);
        }
        FrontierOutcome::Complete(Self {
            components: output.into(),
            provenance: Arc::new(FrontierProvenance::Addition {
                left: self.provenance.clone(),
                right: other.provenance.clone(),
                policy,
            }),
            tolerant_alignment: tolerant,
        })
    }
    /// This front is defined wherever `other` is and is no higher there.
    /// A front's value is the minimum over its components. Each of
    /// `other`'s components is checked at its knots merged with this
    /// front's knots inside its domain; one component of this front must
    /// cover each sub-interval by itself (components are continuous, so its
    /// endpoint values bound it). Under the native policy, domains snap and
    /// values compare as the reference does (1e-4, or relative 2.22e-10);
    /// under the strict policy both are exact. Arithmetic failure rejects.
    pub(crate) fn covers_below(&self, other: &Self, policy: FrontOperationPolicy) -> bool {
        let evaluation = match policy {
            FrontOperationPolicy::StrictDomain => EvaluationPolicy::StrictDomain,
            FrontOperationPolicy::NativeFrontAlgebra => EvaluationPolicy::NativeTolerance,
        };
        let value = |f: &PreparedSlopeFunction, x: f64| f.evaluate(x, evaluation).ok().flatten();
        let below = |a: f64, b: f64| a <= b || near(a, b, policy).unwrap_or(false);
        let mut knots = Vec::new();
        other.components.iter().all(|b| {
            let (lo, hi) = b.domain();
            knots.clear();
            knots.extend(b.pairs().iter().map(|p| p.1));
            knots.extend(
                self.components
                    .iter()
                    .flat_map(|a| a.pairs().iter().map(|p| p.1))
                    .filter(|&x| lo < x && x < hi),
            );
            knots.sort_by(f64::total_cmp);
            knots.dedup();
            let covers = |l: f64, h: f64| {
                let (Some(bl), Some(bh)) = (value(b, l), value(b, h)) else {
                    return false;
                };
                self.components.iter().any(|a| {
                    matches!((value(a, l), value(a, h)),
                        (Some(al), Some(ah)) if below(al, bl) && below(ah, bh))
                })
            };
            if knots.len() == 1 {
                return covers(knots[0], knots[0]);
            }
            knots.windows(2).all(|w| covers(w[0], w[1]))
        })
    }

    /// Minimum of the computed slope front, not of a separate source program.
    /// Ordered endpoint accumulation suffices for each monotone slope segment.
    pub fn minimum(&self, budget: FrontierBudget) -> FrontierOutcome<FrontAlgebraWitness> {
        let mut work = Work::new(budget);
        let mut best: Option<(f64, f64, usize)> = None;
        for (component, function) in self.components.iter().enumerate() {
            let (mut value, mut x) = function.pairs()[0];
            for (index, &(slope, end)) in function.pairs().iter().enumerate() {
                if let Err(reason) = work.charge(1) {
                    return FrontierOutcome::Unknown(reason);
                }
                if index > 0 {
                    let next = checked(end - x, "minimum span")
                        .and_then(|span| checked(slope * span, "minimum multiplication"))
                        .and_then(|product| checked(value + product, "minimum accumulation"));
                    match next {
                        Ok(next) => value = next,
                        Err(error) => {
                            return FrontierOutcome::Unknown(FrontierUnknown::Arithmetic(error))
                        }
                    }
                    x = end;
                }
                if best.as_ref().map_or(true, |old| value < old.1) {
                    best = Some((x, value, component));
                }
            }
        }
        match best {
            Some((coordinate, value, component)) => {
                FrontierOutcome::Complete(FrontAlgebraWitness {
                    coordinate,
                    value,
                    component,
                    provenance: self.provenance.clone(),
                })
            }
            None => FrontierOutcome::Empty,
        }
    }
}
type FrontResult<T> = Result<T, FrontierUnknown>;
struct Work {
    budget: FrontierBudget,
    operations: usize,
    pairs: usize,
    timer: Option<(Instant, Duration)>,
}
impl Work {
    fn new(budget: FrontierBudget) -> Self {
        Self {
            budget,
            operations: 0,
            pairs: 0,
            timer: budget.max_elapsed.map(|limit| (Instant::now(), limit)),
        }
    }
    fn charge(&mut self, count: usize) -> FrontResult<()> {
        self.operations = self
            .operations
            .checked_add(count)
            .ok_or(FrontierUnknown::WorkLimit)?;
        if self.operations > self.budget.max_operations {
            return Err(FrontierUnknown::WorkLimit);
        }
        if self
            .timer
            .is_some_and(|(start, limit)| start.elapsed() >= limit)
        {
            return Err(FrontierUnknown::Deadline);
        }
        Ok(())
    }
    fn pair(&mut self) -> FrontResult<()> {
        self.charge(1)?;
        self.pairs = self
            .pairs
            .checked_add(1)
            .ok_or(FrontierUnknown::PairLimit)?;
        if self.pairs > self.budget.max_pairs {
            Err(FrontierUnknown::PairLimit)
        } else {
            Ok(())
        }
    }
}
fn validate(components: &[PreparedSlopeFunction], work: &mut Work) -> FrontResult<()> {
    if components.len() > work.budget.max_components {
        return Err(FrontierUnknown::ComponentLimit);
    }
    let mut end = None;
    let mut count = 0usize;
    for component in components {
        work.charge(component.pairs().len())?;
        count = count
            .checked_add(component.pairs().len())
            .ok_or(FrontierUnknown::PairLimit)?;
        if count > work.budget.max_pairs {
            return Err(FrontierUnknown::PairLimit);
        }
        if end.is_some_and(|end| component.domain().0 <= end) {
            return Err(FrontierUnknown::Unsupported(
                "overlapping or unordered components require an envelope",
            ));
        }
        end = Some(component.domain().1);
    }
    Ok(())
}
fn checked(value: f64, operation: &'static str) -> Result<f64, RawFunctionError> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err(RawFunctionError::ArithmeticOverflow(operation))
    }
}
fn arithmetic(value: f64, operation: &'static str) -> FrontResult<f64> {
    checked(value, operation).map_err(FrontierUnknown::Arithmetic)
}
fn near(a: f64, b: f64, policy: FrontOperationPolicy) -> FrontResult<bool> {
    if a == b {
        return Ok(true);
    }
    if policy == FrontOperationPolicy::StrictDomain {
        return Ok(false);
    }
    let difference = arithmetic(a - b, "front alignment difference")?.abs();
    if difference < 0.0001 {
        return Ok(true);
    }
    let sum = arithmetic(a + b, "front alignment sum")?.abs();
    Ok(difference < arithmetic(2.220446049250313e-10 * sum, "front alignment band")?)
}
fn value_at(
    function: &PreparedSlopeFunction,
    x: f64,
    policy: FrontOperationPolicy,
) -> FrontResult<f64> {
    let policy = match policy {
        FrontOperationPolicy::StrictDomain => EvaluationPolicy::StrictDomain,
        FrontOperationPolicy::NativeFrontAlgebra => EvaluationPolicy::NativeTolerance,
    };
    function
        .evaluate(x, policy)
        .map_err(FrontierUnknown::Arithmetic)?
        .ok_or(FrontierUnknown::Unsupported(
            "alignment escaped a source domain",
        ))
}
fn add_component(
    a: &PreparedSlopeFunction,
    b: &PreparedSlopeFunction,
    policy: FrontOperationPolicy,
    work: &mut Work,
) -> FrontResult<Option<(PreparedSlopeFunction, bool)>> {
    let input_pairs = a
        .pairs()
        .len()
        .checked_add(b.pairs().len())
        .ok_or(FrontierUnknown::PairLimit)?;
    if input_pairs > work.budget.max_pairs {
        return Err(FrontierUnknown::PairLimit);
    }
    work.charge(input_pairs)?;
    let (as_, ae) = a.domain();
    let (bs, be) = b.domain();
    if as_ > be || bs > ae {
        if policy == FrontOperationPolicy::NativeFrontAlgebra
            && (near(as_, be, policy)? || near(bs, ae, policy)?)
        {
            return Err(FrontierUnknown::Unsupported(
                "near-disjoint domains need a topology proof",
            ));
        }
        return Ok(None);
    }
    let same = near(as_, bs, policy)?;
    let start = if same { as_ } else { as_.max(bs) };
    let mut tolerant = same && as_ != bs;
    let (av, bv) = if same {
        (a.pairs()[0].0, b.pairs()[0].0)
    } else {
        (value_at(a, start, policy)?, value_at(b, start, policy)?)
    };
    work.pair()?;
    let mut pairs = vec![(arithmetic(av + bv, "front initial addition")?, start)];
    if as_ == ae || bs == be || start == ae.min(be) {
        if start != ae.min(be) {
            return Err(FrontierUnknown::Unsupported(
                "tolerant singleton alignment is unresolved",
            ));
        }
        return PreparedSlopeFunction::from_prepared_pairs(pairs)
            .map(|f| Some((f, tolerant)))
            .map_err(FrontierUnknown::Arithmetic);
    }
    let (mut i, mut j) = (1, 1);
    while i < a.pairs().len() && a.pairs()[i].1 <= start {
        i += 1;
    }
    while j < b.pairs().len() && b.pairs()[j].1 <= start {
        j += 1;
    }
    let mut previous = start;
    while i < a.pairs().len() && j < b.pairs().len() {
        work.charge(1)?;
        let (sa, xa) = a.pairs()[i];
        let (sb, xb) = b.pairs()[j];
        let aligned = near(xa, xb, policy)?;
        let end = if aligned { xa } else { xa.min(xb) };
        if end <= previous {
            return Err(FrontierUnknown::Unsupported(
                "alignment reversed consecutive coordinates",
            ));
        }
        work.pair()?;
        pairs.push((arithmetic(sa + sb, "front slope addition")?, end));
        previous = end;
        if aligned {
            tolerant |= xa != xb;
            i += 1;
            j += 1;
        } else if xa < xb {
            i += 1;
        } else {
            j += 1;
        }
    }
    PreparedSlopeFunction::from_prepared_pairs(pairs)
        .map(|f| Some((f, tolerant)))
        .map_err(FrontierUnknown::Arithmetic)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn function(pairs: &[(f64, f64)]) -> PreparedSlopeFunction {
        PreparedSlopeFunction::from_prepared_pairs(pairs.to_vec()).unwrap()
    }
    fn complete<T: std::fmt::Debug>(outcome: FrontierOutcome<T>) -> T {
        match outcome {
            FrontierOutcome::Complete(value) => value,
            other => panic!("unexpected {other:?}"),
        }
    }

    #[test]
    fn native_slope_addition_and_literal_replay_have_distinct_witnesses() {
        let a = function(&[(1073741824., 0.), (1., 1.)]);
        let b = function(&[(-1073741824., 0.), (0., 1.)]);
        let c = function(&[(0., 0.), (-1., 1.)]);
        let budget = FrontierBudget::default();
        let ab = complete(SlopeFront::from_prepared(a.clone()).add(
            &SlopeFront::from_prepared(b.clone()),
            FrontOperationPolicy::NativeFrontAlgebra,
            budget,
        ));
        assert_eq!(ab.components()[0].pairs(), &[(0., 0.), (1., 1.)]);
        let abc = complete(ab.add(
            &SlopeFront::from_prepared(c.clone()),
            FrontOperationPolicy::NativeFrontAlgebra,
            budget,
        ));
        assert_eq!(abc.components()[0].pairs(), &[(0., 0.), (0., 1.)]);
        assert_eq!(complete(abc.minimum(budget)).value, 0.);
        // This argument is outside the native start-snap band. The difference
        // is operation scope, not a difference in leaf boundary policy.
        let x = 2_f64.powi(-12) + 2_f64.powi(-24);
        let replay = SourceReplayWitness::evaluate_shared_argument(
            &[a, b, c],
            x,
            EvaluationPolicy::NativeTolerance,
        )
        .unwrap()
        .unwrap();
        assert_eq!(replay.value(), -2_f64.powi(-24));
        assert!(matches!(
            &**abc.provenance(),
            FrontierProvenance::Addition { .. }
        ));
    }

    #[test]
    fn native_alignment_preserves_left_coordinates_and_active_slopes() {
        let budget = FrontierBudget::default();
        let a = SlopeFront::from_prepared(function(&[(0., 0.), (1., 5.), (2., 10.)]));
        let b = SlopeFront::from_prepared(function(&[(0., 0.), (3., 5.00001), (4., 10.)]));
        let native = complete(a.add(&b, FrontOperationPolicy::NativeFrontAlgebra, budget));
        assert_eq!(
            native.components()[0].pairs(),
            &[(0., 0.), (4., 5.), (6., 10.)]
        );
        assert!(native.used_tolerant_alignment());
        let strict = complete(a.add(&b, FrontOperationPolicy::StrictDomain, budget));
        assert_eq!(
            strict.components()[0].pairs(),
            &[(0., 0.), (4., 5.), (5., 5.00001), (6., 10.)]
        );
        assert!(!strict.used_tolerant_alignment());
    }

    #[test]
    fn strict_policy_does_not_expand_a_nearby_source_start() {
        let a = SlopeFront::from_prepared(function(&[(1., 0.), (2., 10.)]));
        let b = SlopeFront::from_prepared(function(&[(3., 0.00001), (4., 10.)]));
        let native = complete(a.add(
            &b,
            FrontOperationPolicy::NativeFrontAlgebra,
            FrontierBudget::default(),
        ));
        assert_eq!(native.components()[0].pairs(), &[(4., 0.), (6., 10.)]);
        let strict = complete(a.add(
            &b,
            FrontOperationPolicy::StrictDomain,
            FrontierBudget::default(),
        ));
        assert_eq!(strict.components()[0].domain().0, 0.00001);
        assert_eq!(strict.components()[0].pairs()[0].0, 4.00002);
        let a = SlopeFront::from_prepared(function(&[(1., 0.), (2., 10.00001)]));
        let b = SlopeFront::from_prepared(function(&[(3., 0.), (4., 10.)]));
        let native = complete(a.add(
            &b,
            FrontOperationPolicy::NativeFrontAlgebra,
            FrontierBudget::default(),
        ));
        assert_eq!(native.components()[0].pairs(), &[(4., 0.), (6., 10.00001)]);
        let strict = complete(a.add(
            &b,
            FrontOperationPolicy::StrictDomain,
            FrontierBudget::default(),
        ));
        assert_eq!(strict.components()[0].domain().1, 10.);
    }

    #[test]
    fn disconnected_domains_and_singleton_minima_are_not_bridged() {
        let budget = FrontierBudget::default();
        let components = complete(SlopeFront::from_components(
            vec![function(&[(3., 0.), (1., 1.)]), function(&[(-2., 3.)])],
            budget,
        ));
        let other = SlopeFront::from_prepared(function(&[(0., 0.), (1., 4.)]));
        let result = complete(components.add(&other, FrontOperationPolicy::StrictDomain, budget));
        assert_eq!(result.components().len(), 2);
        assert_eq!(result.components()[0].domain(), (0., 1.));
        assert_eq!(result.components()[1].pairs(), &[(1., 3.)]);
        let witness = complete(result.minimum(budget));
        assert_eq!(
            (witness.coordinate, witness.value, witness.component),
            (3., 1., 1)
        );
    }

    #[test]
    fn empty_unknown_limits_and_arithmetic_are_distinct() {
        let a = SlopeFront::from_prepared(function(&[(1., 0.), (0., 1.)]));
        let far = SlopeFront::from_prepared(function(&[(2., 2.), (0., 3.)]));
        assert!(matches!(
            a.add(
                &far,
                FrontOperationPolicy::StrictDomain,
                FrontierBudget::default()
            ),
            FrontierOutcome::Empty
        ));
        let zero = FrontierBudget {
            max_operations: 0,
            ..FrontierBudget::default()
        };
        assert!(matches!(
            a.add(&a, FrontOperationPolicy::StrictDomain, zero),
            FrontierOutcome::Unknown(FrontierUnknown::WorkLimit)
        ));
        assert!(matches!(
            a.minimum(zero),
            FrontierOutcome::Unknown(FrontierUnknown::WorkLimit)
        ));
        let huge = SlopeFront::from_prepared(function(&[(f64::MAX, 0.)]));
        assert!(matches!(
            huge.add(
                &huge,
                FrontOperationPolicy::StrictDomain,
                FrontierBudget::default()
            ),
            FrontierOutcome::Unknown(FrontierUnknown::Arithmetic(_))
        ));
        let near_point = SlopeFront::from_prepared(function(&[(0., 1.00001)]));
        assert!(matches!(
            a.add(
                &near_point,
                FrontOperationPolicy::NativeFrontAlgebra,
                FrontierBudget::default()
            ),
            FrontierOutcome::Unknown(FrontierUnknown::Unsupported(_))
        ));
    }

    #[test]
    fn zero_translation_and_restriction_preserve_slope_arithmetic() {
        let budget = FrontierBudget::default();
        assert_eq!(
            SlopeFront::zero(2., 2.).unwrap().components()[0].pairs(),
            &[(0., 2.)]
        );
        assert_eq!(
            SlopeFront::zero(0., 2.).unwrap().components()[0].pairs(),
            &[(0., 0.), (0., 2.)]
        );
        let source = SlopeFront::from_prepared(function(&[(1., 0.), (2., 2.)]));
        let interior =
            complete(source.restrict(0.5, 1.5, FrontOperationPolicy::NativeFrontAlgebra, budget));
        assert_eq!(interior.components()[0].pairs(), &[(2., 0.5), (2., 1.5)]);
        let singleton =
            complete(source.restrict(1., 1., FrontOperationPolicy::NativeFrontAlgebra, budget));
        assert_eq!(singleton.components()[0].pairs(), &[(3., 1.)]);
        let shifted = complete(interior.translate(4., budget));
        assert_eq!(shifted.components()[0].pairs(), &[(2., 4.5), (2., 5.5)]);
        assert!(matches!(
            &**shifted.provenance(),
            FrontierProvenance::Translation { .. }
        ));
    }

    #[test]
    fn restriction_retains_gaps_and_translation_collapse_is_unknown() {
        let budget = FrontierBudget::default();
        let source = complete(SlopeFront::from_components(
            vec![
                function(&[(0., 0.), (1., 1.)]),
                function(&[(2., 3.), (-1., 4.)]),
            ],
            budget,
        ));
        let restricted =
            complete(source.restrict(0.5, 3.5, FrontOperationPolicy::StrictDomain, budget));
        assert_eq!(restricted.components().len(), 2);
        assert_eq!(restricted.components()[0].pairs(), &[(0.5, 0.5), (1., 1.)]);
        assert_eq!(restricted.components()[1].pairs(), &[(2., 3.), (-1., 3.5)]);
        assert!(matches!(
            source.restrict(1.5, 2., FrontOperationPolicy::StrictDomain, budget),
            FrontierOutcome::Empty
        ));
        assert!(matches!(
            source.translate(1e20, budget),
            FrontierOutcome::Unknown(FrontierUnknown::Unsupported(_))
        ));
        assert!(matches!(
            source.restrict(
                0.00001,
                0.5,
                FrontOperationPolicy::NativeFrontAlgebra,
                budget
            ),
            FrontierOutcome::Unknown(FrontierUnknown::Unsupported(_))
        ));
        assert!(matches!(
            source.translate(
                1.,
                FrontierBudget {
                    max_operations: 0,
                    ..budget
                }
            ),
            FrontierOutcome::Unknown(FrontierUnknown::WorkLimit)
        ));
    }
}
