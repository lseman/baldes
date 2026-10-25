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
        let arrival = self.arrival.range(base, work)?;
        let increment = self.increment.range(base, work)?;
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
            max(next, self.lower.range(head, work)?),
            self.upper.range(head, work)?,
        ))
    }
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

impl CoupledKernel {
    pub(super) fn certify_relations(
        &self,
        path: &[usize],
        shared_work: &mut usize,
        deadline: Option<Instant>,
    ) -> Option<(bool, [f64; 2])> {
        let allowance = (*shared_work).min(MAX_WORK);
        let mut work = Work(allowance, deadline);
        let result = self.certify_relations_inner(path, &mut work);
        *shared_work = shared_work.checked_sub(allowance - work.0)?;
        result
    }

    fn certify_relations_inner(&self, path: &[usize], work: &mut Work) -> Option<(bool, [f64; 2])> {
        work.take(self.definitions.len())?;
        let mut total = Interval::point(0.);
        for definition in self.definitions.iter() {
            if !matches!(
                definition,
                CoupledResource::PairedPiecewiseLinear { .. } | CoupledResource::SlopePaired { .. }
            ) {
                continue;
            }
            work.take(path.len().checked_mul(5)?.checked_add(1)?)?;
            let (initial, base, mode, normalization, arcs) = match definition {
                CoupledResource::PairedPiecewiseLinear {
                    initial,
                    initial_base,
                    arcs,
                    mode,
                } => (
                    Function::Knots(initial),
                    *initial_base,
                    *mode,
                    descriptor::PairedCostNormalization::FinalMinusInitialMinimum,
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
                ),
                CoupledResource::SlopePaired {
                    initial,
                    initial_base,
                    arcs,
                    mode,
                    normalization,
                    policy: FrontOperationPolicy::StrictDomain,
                } => (
                    Function::Slopes(initial),
                    *initial_base,
                    *mode,
                    *normalization,
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
                ),
                CoupledResource::SlopePaired { .. } => {
                    let (feasible, cost) = operators::native_pair(
                        definition,
                        path,
                        self.cost_scale,
                        &mut work.0,
                        work.1,
                    )?;
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
            let (feasible, cost) = certify(
                initial,
                base,
                &arcs,
                mode,
                normalization,
                self.cost_scale,
                work,
            )?;
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
