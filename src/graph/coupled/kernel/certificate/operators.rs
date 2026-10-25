//! Exact operator certificates for integer-aligned raw slope geometry.
//!
//! This does not reinterpret a general numerical front as source replay.
//! Coordinates are bounded integers, so native tolerance cannot align distinct
//! coordinates; every retained endpoint operation is an exact binary64 integer.
use super::*;
use crate::graph::coupled::descriptor::{PairedCostNormalization, PreparedSlopeFunction};

const LIMIT: f64 = 1_000_000.;

fn small(x: f64) -> bool {
    x.is_finite() && x.fract() == 0. && x.abs() <= LIMIT
}
fn exact(x: f64) -> bool {
    x.is_finite() && x.fract() == 0. && x.abs() <= 4_503_599_627_370_496.
}

fn shape(
    function: &PreparedSlopeFunction,
    charged: bool,
    work: &mut usize,
    deadline: Option<Instant>,
) -> Option<(f64, f64)> {
    debit(work, function.pairs().len(), deadline)?;
    if !function.pairs().iter().all(|p| small(p.1)) {
        return None;
    }
    if !charged {
        return Some((0., 0.));
    }
    let (mut value, mut previous) = function.pairs()[0];
    if !small(value) {
        return None;
    }
    let (mut magnitude, mut slope_magnitude) = (value.abs(), 0f64);
    for &(slope, end) in &function.pairs()[1..] {
        if !small(slope) {
            return None;
        }
        let product = slope * (end - previous);
        let next = value + product;
        if !exact(product) || !small(next) {
            return None;
        }
        value = next;
        previous = end;
        magnitude = magnitude.max(value.abs());
        slope_magnitude = slope_magnitude.max(slope.abs());
    }
    Some((magnitude, slope_magnitude))
}

fn constant(
    function: &PreparedSlopeFunction,
    work: &mut usize,
    deadline: Option<Instant>,
) -> Option<f64> {
    shape(function, true, work, deadline)?;
    function
        .pairs()
        .iter()
        .skip(1)
        .all(|p| p.0 == 0.)
        .then_some(function.pairs()[0].0)
}

pub(super) fn native_pair(
    definition: &CoupledResource,
    path: &[usize],
    scale: f64,
    work: &mut usize,
    deadline: Option<Instant>,
) -> Option<(bool, [f64; 2])> {
    let CoupledResource::SlopePaired {
        initial,
        initial_base,
        arcs,
        mode,
        normalization,
        ..
    } = definition
    else {
        return None;
    };
    let baseline = constant(initial, work, deadline)?;
    if ![initial_base.lower, initial_base.upper]
        .into_iter()
        .all(small)
        || scale < 0.
    {
        return None;
    }
    let domain = initial.domain();
    let (mut lo, mut hi) = (
        domain.0.max(initial_base.lower),
        domain.1.min(initial_base.upper),
    );
    if lo > hi {
        return Some((false, [0.; 2]));
    }
    let mut dependent = baseline;
    for &i in path {
        debit(work, 1, deadline)?;
        let arc = arcs.get(i)?;
        if arc.waiting != 0 {
            return None;
        }
        shape(&arc.arrival, true, work, deadline)?;
        if !arc.arrival.pairs().iter().skip(1).all(|p| p.0 == 1.) {
            return None;
        }
        let increment = constant(&arc.increment, work, deadline)?;
        let lower = constant(&arc.lower, work, deadline)?;
        let upper = constant(&arc.upper, work, deadline)?;
        let ad = arc.arrival.domain();
        let id = arc.increment.domain();
        lo = lo.max(ad.0).max(id.0);
        hi = hi.min(ad.1).min(id.1);
        if lo > hi {
            return Some((false, [0.; 2]));
        }
        let delta = arc.arrival.pairs()[0].0 - ad.0;
        if !small(delta) || delta < 0. {
            return None;
        }
        let ld = arc.lower.domain();
        let ud = arc.upper.domain();
        lo = (lo + delta).max(ld.0).max(ud.0);
        hi = ld.1.min(ud.1);
        if !small(lo) || !small(hi) {
            return None;
        }
        if lo > hi {
            return Some((false, [0.; 2]));
        }
        dependent = (dependent + increment).max(lower);
        if !small(dependent) {
            return None;
        }
        if dependent > upper {
            return Some((false, [0.; 2]));
        }
    }
    if path.is_empty() || *mode == PairedResourceMode::Feasibility || scale == 0. {
        return Some((true, [0.; 2]));
    }
    let baseline = match normalization {
        PairedCostNormalization::AbsoluteFinalValue => 0.,
        PairedCostNormalization::FinalMinusInitialMinimum => baseline,
    };
    let cost = bounds::Range::point(dependent)
        .subtract(bounds::Range::point(baseline))
        .multiply(bounds::Range::point(scale));
    cost.finite().then_some((true, [cost.lo, cost.hi]))
}

impl CoupledKernel {
    pub(super) fn certify_slope_fronts(
        &self,
        path: &[usize],
        work: &mut usize,
        deadline: Option<Instant>,
    ) -> Option<(bool, [f64; 2])> {
        let mut total = Interval::point(0.);
        for &index in &self.slope_indices {
            let CoupledResource::SlopeIntervalCost {
                initial,
                bounds,
                increments,
                functions,
                ..
            } = &self.definitions[index]
            else {
                return None;
            };
            if ![initial.lower, initial.upper, bounds.lower, bounds.upper]
                .into_iter()
                .all(small)
                || !(self.cost_scale == 0. || self.cost_scale == 1.)
            {
                return None;
            }
            let (mut lo, mut hi, mut shift) = (initial.lower, initial.upper, 0.);
            let mut knots = vec![lo, hi];
            let mut shifts = Vec::with_capacity(path.len());
            let (mut magnitude, mut slopes) = (0., 0.);
            for &arc in path {
                debit(work, 1, deadline)?;
                let increment = *increments.get(arc)?;
                if !small(increment) {
                    return None;
                }
                shift += increment;
                if !small(shift) {
                    return None;
                }
                let function = functions.get(arc)?;
                let (m, s) = shape(function, self.cost_scale != 0., work, deadline)?;
                magnitude += m;
                slopes += s;
                // This bounds every intermediate anchor value and summed
                // slope. With integer coordinates <=2*LIMIT, all segment
                // products are below2^53 and therefore exactly represented.
                if !small(magnitude) || !small(slopes) {
                    return None;
                }
                let (left, right) = function.domain();
                lo = lo.max(bounds.lower - shift).max(left - shift);
                hi = hi.min(bounds.upper - shift).min(right - shift);
                shifts.push(shift);
                knots.extend(function.pairs().iter().map(|p| p.1 - shift));
            }
            if lo > hi {
                return Some((false, [0.; 2]));
            }
            knots.extend([lo, hi]);
            debit(
                work,
                knots.len().checked_mul(usize::BITS as usize)?,
                deadline,
            )?;
            knots.retain(|x| *x >= lo && *x <= hi);
            knots.sort_by(f64::total_cmp);
            knots.dedup();
            let mut minimum = f64::INFINITY;
            for x in knots {
                let mut cost = Interval::point(0.);
                for (&arc, &shift) in path.iter().zip(&shifts) {
                    if self.cost_scale == 0. {
                        continue;
                    }
                    let value = paired::slope_range(
                        &functions[arc],
                        bounds::Range::point(x + shift),
                        work,
                        deadline,
                    )?;
                    if value.lo != value.hi || !small(value.lo) {
                        return None;
                    }
                    cost = cost.add(Interval::point(value.lo))?;
                }
                if cost.lo != cost.hi || !small(cost.lo) {
                    return None;
                }
                minimum = minimum.min(cost.lo);
            }
            if !minimum.is_finite() {
                return None;
            }
            total = total.add(Interval::point(minimum))?;
        }
        Some((true, [total.lo, total.hi]))
    }
}
