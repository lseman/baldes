//! Independent outward reachability and explicit-control witnesses for paired
//! resources. Numerical optimized relation fronts are never proof inputs.
//!
//! The outer interval state contains every feasible control sequence. A separate
//! bounded search retains only independently certified control sequences. Their
//! final dependent values give a lower/upper enclosure of the optimal charge.
//! A failed witness search is unknown, not an infeasibility proof.
use super::*;
use crate::graph::coupled::descriptor::{FrontOperationPolicy, PreparedSlopeFunction};

const MAX_WORK: usize = 100_000;
const MAX_STATES: usize = 512;
const MAX_COORDINATES: usize = 256;

#[derive(Clone, Copy)]
enum Function<'a> {
    Knots(&'a PiecewiseLinearFunction),
    Slopes(&'a PreparedSlopeFunction),
}

struct Work(usize, Option<Instant>);
impl Work {
    fn take(&mut self, amount: usize) -> Option<()> {
        if self.1.is_some_and(|limit| Instant::now() >= limit) {
            return None;
        }
        self.0 = self.0.checked_sub(amount)?;
        Some(())
    }
}

fn checked(range: bounds::Range) -> Option<bounds::Range> {
    range.finite().then_some(range)
}

pub(super) fn slope_range(
    function: &PreparedSlopeFunction,
    argument: bounds::Range,
    shared_work: &mut usize,
    deadline: Option<Instant>,
) -> Option<bounds::Range> {
    let mut work = Work(*shared_work, deadline);
    let result = Function::Slopes(function).range(argument, &mut work);
    *shared_work = work.0;
    result
}

fn intersect(a: bounds::Range, b: bounds::Range) -> Option<bounds::Range> {
    let range = bounds::Range {
        lo: a.lo.max(b.lo),
        hi: a.hi.min(b.hi),
    };
    (range.lo <= range.hi).then_some(range)
}

fn max(a: bounds::Range, b: bounds::Range) -> bounds::Range {
    bounds::Range {
        lo: a.lo.max(b.lo),
        hi: a.hi.max(b.hi),
    }
}

impl Function<'_> {
    fn domain(self) -> bounds::Range {
        match self {
            Self::Knots(f) => {
                let domain = f.prepared_domain();
                bounds::Range {
                    lo: domain.lower,
                    hi: domain.upper,
                }
            }
            Self::Slopes(f) => {
                let (lo, hi) = f.domain();
                bounds::Range { lo, hi }
            }
        }
    }

    fn coordinates(self, out: &mut Vec<f64>, work: &mut Work) -> Option<()> {
        match self {
            Self::Knots(f) => {
                work.take(f.points.len())?;
                out.extend(f.points.iter().map(|p| p[0]));
            }
            Self::Slopes(f) => {
                work.take(f.pairs().len())?;
                out.extend(f.pairs().iter().map(|p| p.1));
            }
        }
        Some(())
    }

    fn range(self, argument: bounds::Range, work: &mut Work) -> Option<bounds::Range> {
        match self {
            Self::Knots(f) => {
                work.take(f.points.len())?;
                checked(f.range(argument)?)
            }
            Self::Slopes(f) => {
                work.take(f.pairs().len())?;
                let argument = intersect(argument, self.domain())?;
                let (value, mut previous) = f.pairs()[0];
                let mut prefix = bounds::Range::point(value);
                let mut result = if argument.lo == previous {
                    Some(prefix)
                } else {
                    None
                };
                for &(slope, end) in &f.pairs()[1..] {
                    if let Some(part) = intersect(
                        argument,
                        bounds::Range {
                            lo: previous,
                            hi: end,
                        },
                    ) {
                        let next = checked(
                            prefix.add(
                                part.subtract(bounds::Range::point(previous))
                                    .multiply(bounds::Range::point(slope)),
                            ),
                        )?;
                        result = Some(result.map_or(next, |old: bounds::Range| bounds::Range {
                            lo: old.lo.min(next.lo),
                            hi: old.hi.max(next.hi),
                        }));
                    }
                    if argument.hi <= end {
                        break;
                    }
                    // Preserve the source encoding's ordered segment sums.
                    prefix = checked(
                        prefix.add(
                            bounds::Range::point(end)
                                .subtract(bounds::Range::point(previous))
                                .multiply(bounds::Range::point(slope)),
                        ),
                    )?;
                    previous = end;
                }
                result
            }
        }
    }

    // A source-control point can use an exactly verified affine slope. This
    // proves the real knot interpolant algebraically; it does not claim the
    // reordered binary64 expression reproduces the numerical evaluator.
    fn control_point(self, argument: f64, work: &mut Work) -> Option<bounds::Range> {
        if let Self::Knots(function) = self {
            work.take(function.points.len())?;
            let domain = self.domain();
            if argument < domain.lo || argument > domain.hi {
                return None;
            }
            if let Some(point) = function.points.iter().find(|point| point[0] == argument) {
                return Some(bounds::Range::point(point[1]));
            }
            for pair in function.points.windows(2) {
                let ([x0, y0], [x1, y1]) = (pair[0], pair[1]);
                if x0 < argument && argument < x1 {
                    let dx = bounds::Range::point(x1).subtract(bounds::Range::point(x0));
                    let dy = bounds::Range::point(y1).subtract(bounds::Range::point(y0));
                    let slope = (y1 - y0) / (x1 - x0);
                    let product = bounds::Range::point(slope).multiply(dx);
                    if dx.lo == dx.hi
                        && dy.lo == dy.hi
                        && slope.is_finite()
                        && product.lo == dy.lo
                        && product.hi == dy.hi
                    {
                        return checked(
                            bounds::Range::point(y0).add(
                                bounds::Range::point(argument)
                                    .subtract(bounds::Range::point(x0))
                                    .multiply(bounds::Range::point(slope)),
                            ),
                        );
                    }
                    break;
                }
            }
        }
        self.range(bounds::Range::point(argument), work)
    }

    // Suggestions only: every proposed control is checked independently by
    // outward replay before it can establish feasibility or an upper bound.
    fn inverse_suggestions(self, target: f64, out: &mut Vec<f64>, work: &mut Work) -> Option<()> {
        match self {
            Self::Knots(f) => {
                work.take(f.points.len())?;
                for pair in f.points.windows(2) {
                    let ([x0, y0], [x1, y1]) = (pair[0], pair[1]);
                    if y0 < y1 && y0 <= target && target <= y1 {
                        neighbors(x0 + (target - y0) / (y1 - y0) * (x1 - x0), out);
                    }
                }
            }
            Self::Slopes(f) => {
                work.take(f.pairs().len())?;
                let (mut value, mut start) = f.pairs()[0];
                for &(slope, end) in &f.pairs()[1..] {
                    let next = value + slope * (end - start);
                    if slope > 0. && value <= target && target <= next {
                        neighbors(start + (target - value) / slope, out);
                    }
                    value = next;
                    start = end;
                }
            }
        }
        Some(())
    }
}

fn neighbors(x: f64, out: &mut Vec<f64>) {
    if !x.is_finite() {
        return;
    }
    out.push(x);
    if x == 0. {
        out.extend([-f64::from_bits(1), f64::from_bits(1)]);
    } else {
        let bits = x.to_bits();
        out.push(f64::from_bits(if x > 0. { bits - 1 } else { bits + 1 }));
        out.push(f64::from_bits(if x > 0. { bits + 1 } else { bits - 1 }));
    }
}

#[derive(Clone, Copy)]
struct ArcFunctions<'a> {
    arrival: Function<'a>,
    increment: Function<'a>,
    lower: Function<'a>,
    upper: Function<'a>,
    waiting: i32,
}

impl ArcFunctions<'_> {
    fn departure(self) -> Option<bounds::Range> {
        intersect(self.arrival.domain(), self.increment.domain())
    }

    fn head(self) -> Option<bounds::Range> {
        intersect(self.lower.domain(), self.upper.domain())
    }

    fn extend(
        self,
        base: bounds::Range,
        dependent: bounds::Range,
        head: bounds::Range,
        work: &mut Work,
    ) -> Option<(bounds::Range, bounds::Range)> {
        self.extend_using(base, dependent, head, work, |function, argument, work| {
            function.range(argument, work)
        })
    }

    fn extend_using(
        self,
        base: bounds::Range,
        dependent: bounds::Range,
        head: bounds::Range,
        work: &mut Work,
        mut evaluate: impl FnMut(Function<'_>, bounds::Range, &mut Work) -> Option<bounds::Range>,
    ) -> Option<(bounds::Range, bounds::Range)> {
        let arrival = evaluate(self.arrival, base, work)?;
        let increment = evaluate(self.increment, base, work)?;
        let wait = intersect(
            checked(head.subtract(arrival))?,
            bounds::Range {
                lo: 0.,
                hi: f64::INFINITY,
            },
        )?;
        let next = checked(
            dependent
                .add(increment)
                .add(wait.multiply(bounds::Range::point(f64::from(self.waiting)))),
        )?;
        Some((
            max(next, evaluate(self.lower, head, work)?),
            evaluate(self.upper, head, work)?,
        ))
    }
}

/// Independent source-control feasibility and terminal dependent enclosure.
/// This is neither an optimum certificate nor a machine-front domain proof.
/// Other descriptors and physical path/master constraints are checked separately.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PairedControlCertificate {
    pub feasible: bool,
    pub dependent: [f64; 2],
}
impl PairedControlCertificate {
    /// Only a singleton enclosure proves this exact selected value. Merely
    /// lying inside a non-singleton enclosure does not establish attainability.
    pub fn certifies_value(&self, value: f64) -> bool {
        self.feasible && value.is_finite() && self.dependent == [value, value]
    }
}

// One lowering for both path-optimum and explicit source-control certificates.
fn source_functions<'a>(
    definition: &'a CoupledResource,
    path: &[usize],
) -> Option<(Function<'a>, Window, Vec<ArcFunctions<'a>>)> {
    match definition {
        CoupledResource::PairedPiecewiseLinear {
            initial,
            initial_base,
            arcs,
            ..
        } => Some((
            Function::Knots(initial),
            *initial_base,
            path.iter()
                .map(|&i| {
                    let a = arcs.get(i)?;
                    Some(ArcFunctions {
                        arrival: Function::Knots(&a.arrival),
                        increment: Function::Knots(&a.increment),
                        lower: Function::Knots(&a.lower),
                        upper: Function::Knots(&a.upper),
                        waiting: a.waiting,
                    })
                })
                .collect::<Option<Vec<_>>>()?,
        )),
        CoupledResource::SlopePaired {
            initial,
            initial_base,
            arcs,
            ..
        } => Some((
            Function::Slopes(initial),
            *initial_base,
            path.iter()
                .map(|&i| {
                    let a = arcs.get(i)?;
                    Some(ArcFunctions {
                        arrival: Function::Slopes(&a.arrival),
                        increment: Function::Slopes(&a.increment),
                        lower: Function::Slopes(&a.lower),
                        upper: Function::Slopes(&a.upper),
                        waiting: a.waiting,
                    })
                })
                .collect::<Option<Vec<_>>>()?,
        )),
        _ => None,
    }
}

fn control_certificate(
    definition: &CoupledResource,
    path: &[usize],
    base: &[f64],
    work: &mut Work,
) -> Option<PairedControlCertificate> {
    work.take(path.len().checked_mul(5)?.checked_add(1)?)?;
    let rejected = || {
        Some(PairedControlCertificate {
            feasible: false,
            dependent: [0.; 2],
        })
    };
    let (initial, window, arcs) = source_functions(definition, path)?;
    let contains = |domain: bounds::Range, value: f64| domain.lo <= value && value <= domain.hi;
    let Some(domain) = intersect(
        initial.domain(),
        bounds::Range {
            lo: window.lower,
            hi: window.upper,
        },
    ) else {
        return rejected();
    };
    if !contains(domain, base[0]) {
        return rejected();
    }
    let mut dependent = initial.control_point(base[0], work)?;
    for (arc, choices) in arcs.into_iter().zip(base.windows(2)) {
        work.take(1)?;
        let (Some(departure), Some(head)) = (arc.departure(), arc.head()) else {
            return rejected();
        };
        if !contains(departure, choices[0]) || !contains(head, choices[1]) {
            return rejected();
        }
        let start = bounds::Range::point(choices[0]);
        let end = bounds::Range::point(choices[1]);
        let arrival = arc.arrival.control_point(choices[0], work)?;
        if choices[1] < arrival.lo {
            return rejected();
        }
        if choices[1] < arrival.hi {
            return None;
        }
        let (next, upper) =
            arc.extend_using(start, dependent, end, work, |function, coordinate, work| {
                function.control_point(coordinate.lo, work)
            })?;
        if next.lo > upper.hi {
            return rejected();
        }
        if next.hi > upper.lo {
            return None;
        }
        dependent = next;
    }
    work.take(0)?;
    Some(PairedControlCertificate {
        feasible: true,
        dependent: [dependent.lo, dependent.hi],
    })
}

fn coordinates(mut values: Vec<f64>, domain: bounds::Range, work: &mut Work) -> Option<Vec<f64>> {
    values.extend([domain.lo, domain.hi, domain.lo / 2. + domain.hi / 2.]);
    // Debit a conservative comparison-sort bound before sorting, including
    // duplicate/off-domain suggestions. Large inputs cannot bypass the budget.
    work.take(values.len().checked_mul(usize::BITS as usize)?)?;
    values.retain(|x| x.is_finite() && domain.lo <= *x && *x <= domain.hi);
    values.sort_by(f64::total_cmp);
    values.dedup();
    (values.len() <= MAX_COORDINATES).then_some(values)
}

#[derive(Clone, Copy)]
struct State {
    base: bounds::Range,
    dependent: bounds::Range,
}

fn certify(
    initial: Function<'_>,
    domain: Window,
    arcs: &[ArcFunctions<'_>],
    mode: PairedResourceMode,
    normalization: descriptor::PairedCostNormalization,
    scale: f64,
    work: &mut Work,
) -> Option<(bool, [f64; 2])> {
    let Some(base) = intersect(
        initial.domain(),
        bounds::Range {
            lo: domain.lower,
            hi: domain.upper,
        },
    ) else {
        return Some((false, [0.; 2]));
    };
    if arcs.is_empty() {
        return Some((true, [0.; 2]));
    }
    if scale < 0. {
        return None;
    }
    let initial_range = initial.range(base, work)?;
    let mut outer = State {
        base,
        dependent: initial_range,
    };
    // The global initial minimum is enclosed independently of the eventual
    // starting control chosen by the feasible witness.
    let mut knots = vec![];
    initial.coordinates(&mut knots, work)?;
    let mut witnesses = vec![];
    let mut initial_min_upper = f64::INFINITY;
    for x in coordinates(knots, base, work)? {
        let d = initial.range(bounds::Range::point(x), work)?;
        initial_min_upper = initial_min_upper.min(d.hi);
        witnesses.push(State {
            base: bounds::Range::point(x),
            dependent: d,
        });
    }
    for (position, &arc) in arcs.iter().enumerate() {
        work.take(1)?;
        let Some(departure) = arc.departure().and_then(|d| intersect(outer.base, d)) else {
            return Some((false, [0.; 2]));
        };
        let Some(mut head) = arc.head() else {
            return Some((false, [0.; 2]));
        };
        let arrival = arc.arrival.range(departure, work)?;
        head.lo = head.lo.max(arrival.lo);
        if head.lo > head.hi {
            return Some((false, [0.; 2]));
        }
        let (mut dependent, upper) = arc.extend(departure, outer.dependent, head, work)?;
        if dependent.lo > upper.hi {
            return Some((false, [0.; 2]));
        }
        dependent.hi = dependent.hi.min(upper.hi);
        outer = State {
            base: head,
            dependent,
        };

        let mut suggested = vec![];
        arc.lower.coordinates(&mut suggested, work)?;
        arc.upper.coordinates(&mut suggested, work)?;
        if let Some(next) = arcs.get(position + 1) {
            next.arrival.coordinates(&mut suggested, work)?;
            next.increment.coordinates(&mut suggested, work)?;
            let mut targets = vec![];
            next.lower.coordinates(&mut targets, work)?;
            next.upper.coordinates(&mut targets, work)?;
            for target in targets {
                next.arrival
                    .inverse_suggestions(target, &mut suggested, work)?;
            }
        }
        let static_coordinates = coordinates(suggested, head, work)?;
        let mut next_witnesses: Vec<State> = vec![];
        for witness in witnesses {
            work.take(1)?;
            if intersect(witness.base, arc.departure()?).is_none() {
                continue;
            }
            let arrival = arc.arrival.range(witness.base, work)?;
            let mut proposed = static_coordinates.clone();
            neighbors(head.lo.max(arrival.hi), &mut proposed);
            for x in coordinates(proposed, head, work)? {
                work.take(1)?;
                if x < arrival.hi {
                    continue;
                }
                let point = bounds::Range::point(x);
                let (dependent, upper) =
                    arc.extend(witness.base, witness.dependent, point, work)?;
                if dependent.hi > upper.lo {
                    continue;
                }
                work.take(usize::BITS as usize)?;
                match next_witnesses.binary_search_by(|old| old.base.lo.total_cmp(&x)) {
                    Ok(index) => {
                        if dependent.hi < next_witnesses[index].dependent.hi {
                            next_witnesses[index].dependent = dependent;
                        }
                    }
                    Err(index) => {
                        if next_witnesses.len() == MAX_STATES {
                            return None;
                        }
                        work.take(next_witnesses.len() + 1)?;
                        next_witnesses.insert(
                            index,
                            State {
                                base: point,
                                dependent,
                            },
                        );
                    }
                }
            }
        }
        if next_witnesses.len() > MAX_STATES {
            return None;
        }
        witnesses = next_witnesses;
    }
    let witness_upper = witnesses
        .iter()
        .map(|state| state.dependent.hi)
        .reduce(f64::min)?;
    if mode == PairedResourceMode::Feasibility || scale == 0. {
        return Some((true, [0.; 2]));
    }
    let baseline = match normalization {
        descriptor::PairedCostNormalization::AbsoluteFinalValue => bounds::Range::point(0.),
        descriptor::PairedCostNormalization::FinalMinusInitialMinimum => bounds::Range {
            lo: initial_range.lo,
            hi: initial_min_upper,
        },
    };
    let cost = checked(
        bounds::Range {
            lo: outer.dependent.lo,
            hi: witness_upper,
        }
        .subtract(baseline)
        .multiply(bounds::Range::point(scale)),
    )?;
    Some((true, [cost.lo, cost.hi]))
}

/// Independently enclose the global initial minimum over the declared base
/// domain, rather than the initial value chosen by one route witness.
fn initial_minimum(definition: &CoupledResource, work: &mut Work) -> Option<bounds::Range> {
    let (initial, window, _) = source_functions(definition, &[])?;
    let domain = intersect(
        initial.domain(),
        bounds::Range {
            lo: window.lower,
            hi: window.upper,
        },
    )?;
    let lower = initial.range(domain, work)?.lo;
    let mut knots = vec![];
    initial.coordinates(&mut knots, work)?;
    let mut upper = f64::INFINITY;
    for x in coordinates(knots, domain, work)? {
        upper = upper.min(initial.control_point(x, work)?.hi);
    }
    checked(bounds::Range {
        lo: lower,
        hi: upper,
    })
}

impl CoupledKernel {
    /// Independently replay an explicit source control for one paired resource.
    /// Universal source terminal enclosure from declared dependent bounds.
    /// This ignores feasibility restrictions and is not an attainment proof.
    pub(crate) fn paired_terminal_bounds(
        &self,
        resource: usize,
        path: &[usize],
        shared_work: &mut usize,
        deadline: Option<Instant>,
    ) -> Option<[f64; 2]> {
        if self.front_program_domain {
            return None;
        }
        let allowance = (*shared_work).min(MAX_WORK);
        let mut work = Work(allowance, deadline);
        let result = (|| {
            work.take(path.len().checked_add(1)?)?;
            let (initial, window, arcs) = source_functions(self.definitions.get(resource)?, path)?;
            if let Some(last) = arcs.last() {
                let domain = intersect(last.lower.domain(), last.upper.domain())?;
                Some([
                    last.lower.range(domain, &mut work)?.lo,
                    last.upper.range(domain, &mut work)?.hi,
                ])
            } else {
                let domain = intersect(
                    initial.domain(),
                    bounds::Range {
                        lo: window.lower,
                        hi: window.upper,
                    },
                )?;
                let range = initial.range(domain, &mut work)?;
                Some([range.lo, range.hi])
            }
        })();
        *shared_work = shared_work.checked_sub(allowance - work.0)?;
        result
    }

    /// Physical arc positions and base coordinates must have valid shapes and
    /// finite entries; malformed arguments are errors. A certificate with
    /// feasible == false rejects only this control, never the entire path.
    /// None means arithmetic ambiguity, exhausted work or expired deadline.
    ///
    /// This uses declared source functions under strict domains, even when the
    /// graph selects machine-front semantics for ordinary pricing. It is not
    /// an optimized-front certificate and does not certify other resources,
    /// physical continuity, scalar windows, master rows or a minimum value.
    /// Objective weights, normalization and phase-I suppression are irrelevant.
    pub fn certify_paired_control(
        &self,
        path: &[usize],
        control: &PairedControl,
        shared_work: &mut usize,
        deadline: Option<Instant>,
    ) -> Result<Option<PairedControlCertificate>> {
        let definition = self
            .definitions
            .get(control.resource)
            .ok_or_else(|| invalid("unknown paired control descriptor"))?;
        if !matches!(
            definition,
            CoupledResource::PairedPiecewiseLinear { .. } | CoupledResource::SlopePaired { .. }
        ) || control.base.len()
            != path
                .len()
                .checked_add(1)
                .ok_or_else(|| invalid("paired control shape overflow"))?
            || control.base.iter().any(|value| !value.is_finite())
            || path.iter().any(|&arc| arc >= self.arc_count)
        {
            return Err(invalid("paired control requires a paired descriptor, valid arcs and one finite base coordinate per vertex"));
        }
        let allowance = (*shared_work).min(MAX_WORK);
        let mut work = Work(allowance, deadline);
        let certificate = control_certificate(definition, path, &control.base, &mut work);
        *shared_work -= allowance - work.0;
        Ok(certificate)
    }

    /// Source initial-front minimum, enclosing the numerical normalization
    /// baseline too. It is independent of the chosen starting control.
    #[cfg(feature = "clp")]
    pub(crate) fn paired_initial_minimum(
        &self,
        resource: usize,
        shared_work: &mut usize,
        deadline: Option<Instant>,
    ) -> Option<[f64; 2]> {
        let definition = self.definitions.get(resource)?;
        let allowance = (*shared_work).min(MAX_WORK);
        let mut work = Work(allowance, deadline);
        let result = work
            .take(self.definitions.len())
            .and_then(|()| initial_minimum(definition, &mut work));
        *shared_work = shared_work.checked_sub(allowance - work.0)?;
        let source = result?;
        let numeric = match definition {
            CoupledResource::SlopePaired { .. } => {
                let index = self.definitions[..resource]
                    .iter()
                    .filter(|d| matches!(d, CoupledResource::SlopePaired { .. }))
                    .count();
                self.slope_paired.get(index)?.initial_minimum()
            }
            CoupledResource::PairedPiecewiseLinear { .. } => {
                let index = self.relation_indices.iter().position(|&r| r == resource)?;
                self.paired.get(index)?.initial_minimum()
            }
            _ => return None,
        };
        numeric
            .is_finite()
            .then_some([source.lo.min(numeric), source.hi.max(numeric)])
    }

    fn control_charge(
        &self,
        path: &[usize],
        control: &PairedControl,
        work: &mut Work,
    ) -> Option<(bool, Interval)> {
        let certificate = self
            .certify_paired_control(path, control, &mut work.0, work.1)
            .ok()??;
        if !certificate.feasible {
            return Some((false, Interval::point(0.)));
        }
        let value = certificate.dependent[0];
        if !certificate.certifies_value(value) {
            return None;
        }
        let definition = self.definitions.get(control.resource)?;
        let (mode, normalization) = match definition {
            CoupledResource::SlopePaired {
                mode,
                normalization,
                ..
            }
            | CoupledResource::PairedPiecewiseLinear {
                mode,
                normalization,
                ..
            } => (*mode, *normalization),
            _ => return None,
        };
        let scale = self.cost_scale * self.objective_coefficients[control.resource];
        if mode == PairedResourceMode::Feasibility || scale == 0. || path.is_empty() {
            return Some((true, Interval::point(0.)));
        }
        let mut dependent = bounds::Range::point(value);
        if normalization == PairedCostNormalization::FinalMinusInitialMinimum {
            dependent = checked(dependent.subtract(initial_minimum(definition, work)?))?;
        }
        let range = checked(dependent.multiply(bounds::Range::point(scale)))?;
        Some((
            true,
            Interval {
                lo: range.lo,
                hi: range.hi,
            },
        ))
    }

    pub(super) fn certify_relations(
        &self,
        path: &[usize],
        shared_work: &mut usize,
        deadline: Option<Instant>,
        controls: &[PairedControl],
    ) -> Option<(bool, [f64; 2])> {
        let allowance = (*shared_work).min(MAX_WORK);
        let mut work = Work(allowance, deadline);
        let result = self.certify_relations_inner(path, &mut work, controls, None);
        *shared_work = shared_work.checked_sub(allowance - work.0)?;
        result
    }

    /// Isolate one paired charge without subtracting already rounded totals.
    /// Full-path feasibility remains the caller's separate responsibility.
    #[cfg(feature = "clp")]
    pub(crate) fn certify_paired_objective_component(
        &self,
        path: &[usize],
        resource: usize,
        shared_work: &mut usize,
        deadline: Option<Instant>,
    ) -> Option<(bool, [f64; 2])> {
        if !matches!(
            self.definitions.get(resource),
            Some(
                CoupledResource::SlopePaired { .. } | CoupledResource::PairedPiecewiseLinear { .. }
            )
        ) {
            return None;
        }
        let allowance = (*shared_work).min(MAX_WORK);
        let mut work = Work(allowance, deadline);
        let result = self.certify_relations_inner(path, &mut work, &[], Some(resource));
        *shared_work = shared_work.checked_sub(allowance - work.0)?;
        result
    }

    fn certify_relations_inner(
        &self,
        path: &[usize],
        work: &mut Work,
        controls: &[PairedControl],
        only: Option<usize>,
    ) -> Option<(bool, [f64; 2])> {
        work.take(self.definitions.len())?;
        let mut total = Interval::point(0.);
        for (resource, definition) in self.definitions.iter().enumerate() {
            if only.is_some_and(|index| index != resource) {
                continue;
            }
            if !matches!(
                definition,
                CoupledResource::PairedPiecewiseLinear { .. } | CoupledResource::SlopePaired { .. }
            ) {
                continue;
            }
            if let Some(control) = controls.iter().find(|control| control.resource == resource) {
                let (valid, cost) = self.control_charge(path, control, work)?;
                if !valid {
                    return Some((false, [0.; 2]));
                }
                total = total.add(cost)?;
                continue;
            }
            work.take(path.len().checked_mul(5)?.checked_add(1)?)?;
            let scale = self.cost_scale * self.objective_coefficients[resource];
            let (mode, normalization) = match definition {
                CoupledResource::PairedPiecewiseLinear {
                    mode,
                    normalization,
                    ..
                }
                | CoupledResource::SlopePaired {
                    mode,
                    normalization,
                    policy: FrontOperationPolicy::StrictDomain,
                    ..
                } => (*mode, *normalization),
                CoupledResource::SlopePaired { .. } => {
                    let (feasible, cost) = if self.front_program_domain {
                        super::paired_machine::certify(
                            definition,
                            path,
                            scale,
                            &mut work.0,
                            work.1,
                        )?
                    } else {
                        operators::front_algebra_pair(definition, path, scale, &mut work.0, work.1)?
                    };
                    if !feasible {
                        return Some((false, [0.; 2]));
                    }
                    total = total.add(Interval {
                        lo: cost[0],
                        hi: cost[1],
                    })?;
                    continue;
                }
                _ => continue,
            };
            let (initial, base, arcs) = source_functions(definition, path)?;
            let (feasible, cost) = certify(initial, base, &arcs, mode, normalization, scale, work)?;
            if !feasible {
                return Some((false, [0.; 2]));
            }
            total = total.add(Interval {
                lo: cost[0],
                hi: cost[1],
            })?;
        }
        Some((true, [total.lo, total.hi]))
    }
}

#[cfg(test)]
#[path = "../../../../../tests/unit/graph/coupled/kernel/paired_certificate_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "../../../../../tests/unit/graph/coupled/kernel/paired_control_tests.rs"]
mod control_tests;
