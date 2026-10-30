//! Backward paired slope fronts, kept separate from source-control proofs.
//!
//! For an admitted head frontier B, the adjoint is
//! max_{x >= q(t)} (min(B(x), upper(x)) - w*(x-q(t))) - phi(t),
//! restricted to heads whose admitted value reaches lower(x). Reflection
//! turns the reverse waiting envelope into the forward running-min operator.
//! No conversion to coordinate/value knots or fixed-size segment array occurs.
use super::*;
use std::{
    sync::{Arc, OnceLock},
    time::Instant,
};

type Checked<T> = std::result::Result<T, FrontierUnknown>;

#[derive(Clone, Debug)]
struct Operators {
    bounds: PreparedClampFilter,
    waiting: PreparedSlopeRelationArc,
    arrival: PreparedCompositionMap,
    negative_increment: SlopeFront,
    departure: (f64, f64),
}

/// Prepared reverse-waiting, base-composition and dependent-increment operators.
/// A completed numerical front is advisory: it is neither a source-control
/// certificate nor a proof that backward and forward IEEE programs coincide.
#[derive(Clone, Debug)]
pub struct PreparedSlopeAdjointArc {
    source: Arc<SlopeRelationArc>,
    policy: FrontOperationPolicy,
    operators: OnceLock<Operators>,
    terminal: SlopeFront,
}
impl PartialEq for PreparedSlopeAdjointArc {
    fn eq(&self, other: &Self) -> bool {
        self.source == other.source && self.policy == other.policy
    }
}

impl PreparedSlopeAdjointArc {
    pub fn new(
        source: &SlopeRelationArc,
        policy: FrontOperationPolicy,
    ) -> std::result::Result<Self, RawFunctionError> {
        if [
            &source.arrival,
            &source.increment,
            &source.lower,
            &source.upper,
        ]
        .iter()
        .any(|function| function.has_machine_only_geometry())
        {
            return Err(RawFunctionError::Unsupported(
                "machine zero-length segment as paired source",
            ));
        }
        // Check the same original arrival contract as forward preparation.
        let mut x = source.arrival.pairs()[0].1;
        let mut y = source.arrival.pairs()[0].0;
        if y < x {
            return Err(RawFunctionError::Unsupported("arrival precedes departure"));
        }
        for &(slope, end) in &source.arrival.pairs()[1..] {
            if slope <= 0. {
                return Err(RawFunctionError::Unsupported(
                    "arrival must be strictly increasing",
                ));
            }
            y += slope * (end - x);
            if !y.is_finite() {
                return Err(RawFunctionError::ArithmeticOverflow("arrival validation"));
            }
            if y < end {
                return Err(RawFunctionError::Unsupported("arrival precedes departure"));
            }
            x = end;
        }
        Ok(Self {
            source: Arc::new(source.clone()),
            policy,
            operators: OnceLock::new(),
            terminal: SlopeFront::from_prepared(source.upper.clone()),
        })
    }

    /// Start at the declared terminal dependent upper function. This avoids
    /// truncating a genuine bound above a numerical infinity sentinel.
    pub fn prepend_terminal(&self, budget: FrontierBudget) -> FrontierOutcome<SlopeFront> {
        self.prepend(&self.terminal, budget)
    }

    /// Prepend this arc to an admitted head frontier. Empty and Unknown are
    /// distinguished for operator diagnostics; neither alone certifies default
    /// source-domain infeasibility. All stages share one work/deadline budget.
    pub fn prepend(
        &self,
        upper: &SlopeFront,
        budget: FrontierBudget,
    ) -> FrontierOutcome<SlopeFront> {
        match self.prepend_checked(upper, budget) {
            Ok(Some(front)) => FrontierOutcome::Complete(front),
            Ok(None) => FrontierOutcome::Empty,
            Err(reason) => FrontierOutcome::Unknown(reason),
        }
    }

    fn prepend_checked(
        &self,
        upper: &SlopeFront,
        budget: FrontierBudget,
    ) -> Checked<Option<SlopeFront>> {
        if upper.has_machine_zero_length() {
            return Err(FrontierUnknown::Unsupported(
                "machine zero-length segment adjoint",
            ));
        }
        if budget.max_operations == 0 {
            return Err(FrontierUnknown::WorkLimit);
        }
        if budget.max_elapsed.is_some_and(|limit| limit.is_zero()) {
            return Err(FrontierUnknown::Deadline);
        }
        let mut work = Work::new(budget);
        if self.operators.get().is_none() {
            let Some(operators) = prepare(&self.source, self.policy, &mut work)? else {
                return Ok(None);
            };
            work.remaining_time(0)?;
            // Cache only successful preparation. A budget-limited attempt
            // must not permanently poison a later call with a larger budget.
            let _ = self.operators.set(operators);
        }
        let operators = self.operators.get().expect("adjoint operators initialized");
        let Some(reflected) = reflect_front(upper, &mut work)? else {
            return Ok(None);
        };
        let Some(clipped) = completed(operators.bounds.apply(
            &reflected,
            self.policy,
            work.pass()?,
        ))?
        else {
            return Ok(None);
        };
        let Some(waited) = completed(operators.waiting.extend(&clipped, work.pass()?))? else {
            return Ok(None);
        };
        let Some(reflected) = reflect_front(&waited, &mut work)? else {
            return Ok(None);
        };
        let count = reflected.components().len();
        let allowance = work.pass()?.max_operations / count.max(1);
        if allowance == 0 {
            return Err(FrontierUnknown::WorkLimit);
        }
        let mut components = Vec::with_capacity(count);
        for component in reflected.components() {
            if let Some(composed) = completed(operators.arrival.compose_clipped(
                component,
                operators.departure.0,
                operators.departure.1,
                self.policy,
                work.remaining_time(allowance)?,
            ))? {
                components.push(composed);
            }
        }
        let Some(composed) = completed(SlopeFront::from_components(components, work.pass()?))?
        else {
            return Ok(None);
        };
        let Some(result) =
            completed(composed.add(&operators.negative_increment, self.policy, work.pass()?))?
        else {
            return Ok(None);
        };
        work.remaining_time(0)?;
        Ok(Some(result.paired_adjoint_result(
            upper,
            self.source.clone(),
            self.policy,
            clipped.used_tolerant_alignment() || waited.used_tolerant_alignment(),
        )))
    }
}

fn prepare(
    source: &SlopeRelationArc,
    policy: FrontOperationPolicy,
    work: &mut Work,
) -> Checked<Option<Operators>> {
    let sizes = [
        source.arrival.pairs().len(),
        source.increment.pairs().len(),
        source.lower.pairs().len(),
        source.upper.pairs().len(),
    ];
    if sizes.iter().any(|&size| size > work.budget.max_pairs) {
        return Err(FrontierUnknown::PairLimit);
    }
    // Preparation performs several linear scans, including reflection and
    // function construction. Reserve a conservative amount before scanning;
    // the composition builder receives its own share of the same budget.
    let scan = sizes
        .iter()
        .try_fold(0usize, |sum, size| sum.checked_add(*size))
        .and_then(|sum| sum.checked_mul(8))
        .ok_or(FrontierUnknown::WorkLimit)?;
    work.charge(scan)?;
    let (al, ah) = source.arrival.domain();
    let (pl, ph) = source.increment.domain();
    let departure = (al.max(pl), ah.min(ph));
    let (ll, lh) = source.lower.domain();
    let (ul, uh) = source.upper.domain();
    let head = (ll.max(ul), lh.min(uh));
    if departure.0 > departure.1 || head.0 > head.1 {
        // The raw front policy can snap near endpoints. An exact-domain gap
        // within that band therefore cannot establish an empty machine front.
        let near = |lo: f64, hi: f64| {
            let distance = (lo - hi).abs();
            let magnitude = (lo + hi).abs();
            distance.is_finite()
                && (distance < 0.0001
                    || (magnitude.is_finite() && distance < magnitude * 2.220446049250313e-10))
        };
        if policy == FrontOperationPolicy::FrontAlgebra
            && ((departure.0 > departure.1 && near(departure.0, departure.1))
                || (head.0 > head.1 && near(head.0, head.1)))
        {
            return Err(FrontierUnknown::Unsupported(
                "near-disjoint adjoint domains",
            ));
        }
        return Ok(None);
    }
    let arrival_lo = at(&source.arrival, departure.0)?;
    let arrival_hi = at(&source.arrival, departure.1)?;
    let bounds = PreparedClampFilter::new(
        &reflect_function(&source.upper)?,
        &reflect_function(&source.lower)?,
    )
    .map_err(FrontierUnknown::Arithmetic)?;
    let identity = function(-head.1, -head.1, -head.0, 1.)?;
    let waiting = PreparedSlopeRelationArc::new(
        &SlopeRelationArc {
            arrival: identity,
            increment: function(0., -head.1, -head.0, 0.)?,
            lower: function(-f64::MAX, -arrival_hi, -arrival_lo, 0.)?,
            upper: function(f64::MAX, -arrival_hi, -arrival_lo, 0.)?,
            waiting: source.waiting,
        },
        policy,
    )
    .map_err(FrontierUnknown::Arithmetic)?;
    let Some(arrival) = completed(PreparedCompositionMap::new(&source.arrival, work.pass()?))?
    else {
        return Ok(None);
    };
    let negative_increment = PreparedSlopeFunction::from_prepared_pairs(
        source
            .increment
            .pairs()
            .iter()
            .map(|&(value, coordinate)| (-value, coordinate))
            .collect(),
    )
    .map_err(FrontierUnknown::Arithmetic)?;
    Ok(Some(Operators {
        bounds,
        waiting,
        arrival,
        negative_increment: SlopeFront::from_prepared(negative_increment),
        departure,
    }))
}

fn function(value: f64, lo: f64, hi: f64, slope: f64) -> Checked<PreparedSlopeFunction> {
    let pairs = if lo == hi {
        vec![(value, lo)]
    } else {
        vec![(value, lo), (slope, hi)]
    };
    PreparedSlopeFunction::from_prepared_pairs(pairs).map_err(FrontierUnknown::Arithmetic)
}

fn at(function: &PreparedSlopeFunction, x: f64) -> Checked<f64> {
    function
        .evaluate(x, EvaluationPolicy::StrictDomain)
        .map_err(FrontierUnknown::Arithmetic)?
        .ok_or(FrontierUnknown::Unsupported(
            "adjoint argument outside source domain",
        ))
}

/// Reflection negates both the argument and value; segment slopes retain
/// their signs and reverse order. Reanchoring uses the original endpoint scan.
fn reflect_function(source: &PreparedSlopeFunction) -> Checked<PreparedSlopeFunction> {
    let pairs = source.pairs();
    let end = source.domain().1;
    let mut reflected = Vec::with_capacity(pairs.len());
    reflected.push((-at(source, end)?, -end));
    for index in (1..pairs.len()).rev() {
        reflected.push((pairs[index].0, -pairs[index - 1].1));
    }
    PreparedSlopeFunction::from_prepared_pairs(reflected).map_err(FrontierUnknown::Arithmetic)
}

fn reflect_front(front: &SlopeFront, work: &mut Work) -> Checked<Option<SlopeFront>> {
    if front.components().len() > work.budget.max_components {
        return Err(FrontierUnknown::ComponentLimit);
    }
    let mut pairs = 0usize;
    let mut components = Vec::with_capacity(front.components().len());
    for component in front.components().iter().rev() {
        pairs = pairs
            .checked_add(component.pairs().len())
            .ok_or(FrontierUnknown::PairLimit)?;
        if pairs > work.budget.max_pairs {
            return Err(FrontierUnknown::PairLimit);
        }
        work.charge(
            component
                .pairs()
                .len()
                .checked_mul(3)
                .ok_or(FrontierUnknown::WorkLimit)?,
        )?;
        components.push(reflect_function(component)?);
    }
    completed(SlopeFront::from_components(components, work.pass()?))
}

fn completed<T>(outcome: FrontierOutcome<T>) -> Checked<Option<T>> {
    match outcome {
        FrontierOutcome::Complete(value) => Ok(Some(value)),
        FrontierOutcome::Empty => Ok(None),
        FrontierOutcome::Unknown(reason) => Err(reason),
    }
}

struct Work {
    budget: FrontierBudget,
    started: Instant,
    remaining: usize,
    quantum: usize,
}
impl Work {
    fn new(budget: FrontierBudget) -> Self {
        Self {
            budget,
            started: Instant::now(),
            remaining: budget.max_operations,
            quantum: budget.max_operations / 10,
        }
    }
    fn charge(&mut self, count: usize) -> Checked<()> {
        self.remaining = self
            .remaining
            .checked_sub(count)
            .ok_or(FrontierUnknown::WorkLimit)?;
        self.remaining_time(0).map(|_| ())
    }
    fn pass(&mut self) -> Checked<FrontierBudget> {
        if self.quantum == 0 {
            return Err(FrontierUnknown::WorkLimit);
        }
        self.charge(self.quantum)?;
        self.remaining_time(self.quantum)
    }
    fn remaining_time(&self, operations: usize) -> Checked<FrontierBudget> {
        let max_elapsed = match self.budget.max_elapsed {
            Some(limit) => {
                let elapsed = self.started.elapsed();
                if elapsed >= limit {
                    return Err(FrontierUnknown::Deadline);
                }
                Some(limit - elapsed)
            }
            None => None,
        };
        Ok(FrontierBudget {
            max_operations: operations,
            max_elapsed,
            ..self.budget
        })
    }
}
